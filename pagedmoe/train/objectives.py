# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training objectives, written against per-token target logprobs.

Every objective is `loss(lp, batch) -> (scalar, {metric: scalar})` where `lp`
is [B, T] log p(target_t | prefix) and `batch.mask` marks the positions that
count. That is all SFT, CPT, ORPO, DPO, IPO, SimPO, CPO, KTO and GRPO need,
and it is what lets the layered executor backprop the head in chunks
instead of holding [B, T, vocab] logits.

Normalizers come from `StepNorm`, computed over the whole optimizer step
before any micro-batch runs. A micro-batch's loss is its share of the step's
loss, so summing micro-batches gives exactly the loss of one big batch.

Custom objectives: subclass `Objective`, or wrap a function with `Custom`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class StepNorm:
    tokens: float = 1.0  # loss tokens in the step
    rows: float = 1.0  # sequences in the step
    pairs: float = 1.0  # preference pairs in the step
    chosen_tokens: float = 1.0  # loss tokens in chosen rows
    max_len: float = 1.0  # longest completion in the step (Dr. GRPO)


def _seq(lp, mask):
    """(sum logprob, token count) per row."""
    s = (lp * mask).sum(axis=-1)
    n = mask.sum(axis=-1)
    return s, n


def _split(x, pairs: int):
    return x[:pairs], x[pairs:]


class Objective:
    name = "objective"
    # Wants reference logprobs: batch.meta["ref_seq"] (masked sum per row,
    # precomputed once per dataset) or, for GRPO, "ref_lp" per token.
    needs_reference = False
    paired = False  # rows are [chosen..., rejected...]

    def __init__(self):
        self.norm = StepNorm()

    def set_norm(self, norm: StepNorm) -> None:
        self.norm = norm

    def __call__(self, lp, batch):
        return self.loss(lp, batch)

    def loss(self, lp, batch):  # pragma: no cover - interface
        raise NotImplementedError


class NLL(Objective):
    """Token-mean negative log likelihood over the masked tokens (SFT, CPT)."""

    name = "sft"

    def loss(self, lp, batch):
        nll = -(lp * batch.mask).sum() / self.norm.tokens
        return nll, {"nll": nll}


class ORPO(Objective):
    """Odds ratio preference optimization (Hong et al., 2024). No reference model.

    loss = NLL(chosen) + lam * -log sigmoid(log odds(chosen) - log odds(rejected))
    with odds computed from the per-token mean logprob, as in the paper.
    """

    name = "orpo"
    paired = True

    def __init__(self, lam: float = 0.1):
        super().__init__()
        self.lam = float(lam)

    def loss(self, lp, batch):
        P = int(batch.meta["pairs"])
        s, n = _seq(lp, batch.mask)
        n = mx.maximum(n, 1.0)
        avg = s / n
        (sc, _sr), (ac, ar) = _split(s, P), _split(avg, P)
        # log(p / (1 - p)) with p = exp(avg); clamp keeps log1p finite at p -> 1.
        def log_odds(a):
            return a - mx.log1p(-mx.exp(mx.minimum(a, -1e-6)))

        ratio = log_odds(ac) - log_odds(ar)
        or_term = -nn.log_sigmoid(ratio)
        nll = -sc.sum() / self.norm.chosen_tokens
        loss = nll + self.lam * or_term.sum() / self.norm.pairs
        pn = self.norm.pairs
        return loss, {
            "nll": nll,
            "or_loss": or_term.sum() / pn,
            "log_odds": ratio.sum() / pn,
            "acc": (ratio > 0).astype(mx.float32).sum() / pn,
            "margin": (ac - ar).sum() / pn,
        }


class DPO(Objective):
    """Direct preference optimization against a frozen reference.

    loss_type: sigmoid (DPO), ipo (IPO, length-averaged), hinge (SLiC).
    """

    name = "dpo"
    paired = True
    needs_reference = True

    def __init__(self, beta: float = 0.1, loss_type: str = "sigmoid", label_smoothing: float = 0.0):
        super().__init__()
        self.beta = float(beta)
        self.loss_type = loss_type
        self.label_smoothing = float(label_smoothing)

    def loss(self, lp, batch):
        P = int(batch.meta["pairs"])
        s, n = _seq(lp, batch.mask)
        r = batch.meta["ref_seq"]
        if self.loss_type == "ipo":
            n = mx.maximum(n, 1.0)
            s, r = s / n, r / n
        (sc, sr), (rc, rr) = _split(s, P), _split(r, P)
        logits = (sc - rc) - (sr - rr)
        b = self.beta
        if self.loss_type == "sigmoid":
            ls = self.label_smoothing
            per = -(1 - ls) * nn.log_sigmoid(b * logits) - ls * nn.log_sigmoid(-b * logits)
        elif self.loss_type == "ipo":
            per = (logits - 1.0 / (2.0 * b)) ** 2
        elif self.loss_type == "hinge":
            per = mx.maximum(0.0, 1.0 - b * logits)
        else:
            raise ValueError(f"unknown dpo loss_type {self.loss_type!r}")
        pn = self.norm.pairs
        rw_c, rw_r = b * (sc - rc), b * (sr - rr)
        return per.sum() / pn, {
            "reward_chosen": rw_c.sum() / pn,
            "reward_rejected": rw_r.sum() / pn,
            "margin": (rw_c - rw_r).sum() / pn,
            "acc": (rw_c > rw_r).astype(mx.float32).sum() / pn,
        }


class SimPO(Objective):
    """SimPO (Meng et al., 2024): length-normalized, reference free, target margin."""

    name = "simpo"
    paired = True

    def __init__(self, beta: float = 2.0, gamma: float = 1.0, sft_weight: float = 0.0):
        super().__init__()
        self.beta, self.gamma, self.sft_weight = float(beta), float(gamma), float(sft_weight)

    def loss(self, lp, batch):
        P = int(batch.meta["pairs"])
        s, n = _seq(lp, batch.mask)
        avg = s / mx.maximum(n, 1.0)
        ac, ar = _split(avg, P)
        logits = self.beta * (ac - ar) - self.gamma
        per = -nn.log_sigmoid(logits)
        pn = self.norm.pairs
        loss = per.sum() / pn
        out = {
            "margin": (ac - ar).sum() / pn,
            "acc": (ac > ar).astype(mx.float32).sum() / pn,
        }
        if self.sft_weight:
            nll = -_split(s, P)[0].sum() / self.norm.chosen_tokens
            loss = loss + self.sft_weight * nll
            out["nll"] = nll
        return loss, out


class CPO(Objective):
    """Contrastive preference optimization: sigmoid on summed logprobs plus chosen NLL."""

    name = "cpo"
    paired = True

    def __init__(self, beta: float = 0.1, nll_weight: float = 1.0):
        super().__init__()
        self.beta, self.nll_weight = float(beta), float(nll_weight)

    def loss(self, lp, batch):
        P = int(batch.meta["pairs"])
        s, _ = _seq(lp, batch.mask)
        sc, sr = _split(s, P)
        per = -nn.log_sigmoid(self.beta * (sc - sr))
        nll = -sc.sum() / self.norm.chosen_tokens
        pn = self.norm.pairs
        return per.sum() / pn + self.nll_weight * nll, {
            "nll": nll,
            "acc": (sc > sr).astype(mx.float32).sum() / pn,
        }


class KTO(Objective):
    """Kahneman-Tversky optimization on unpaired good / bad completions.

    batch.meta["label"] is 1 for desirable rows and 0 for undesirable ones.
    The KL reference point z0 is a running mean of the detached policy /
    reference log ratio, updated after each step, instead of the paper's
    mismatched-pair estimate (that costs a second forward per step, which on
    a streamed model is a second pass over the expert mass).
    """

    name = "kto"
    needs_reference = True

    def __init__(self, beta: float = 0.1, desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.wd, self.wu = float(desirable_weight), float(undesirable_weight)
        self.z0 = 0.0
        self._acc = [0.0, 0.0]

    def loss(self, lp, batch):
        s, _ = _seq(lp, batch.mask)
        ratio = s - batch.meta["ref_seq"]
        lab = batch.meta["label"].astype(mx.float32)
        z0 = max(0.0, self.z0)
        good = self.wd * (1 - mx.sigmoid(self.beta * (ratio - z0)))
        bad = self.wu * (1 - mx.sigmoid(self.beta * (z0 - ratio)))
        per = lab * good + (1 - lab) * bad
        self._acc[0] += float(mx.stop_gradient(ratio).sum().item())
        self._acc[1] += float(ratio.shape[0])
        rn = self.norm.rows
        # metrics are summed over micro-batches, so z0 is weighted by rows too
        return per.sum() / rn, {
            "ratio": mx.stop_gradient(ratio).sum() / rn,
            "z0": mx.array(z0 * ratio.shape[0] / rn),
        }

    def end_step(self) -> None:
        if self._acc[1]:
            mean = self._acc[0] / self._acc[1]
            self.z0 = 0.9 * self.z0 + 0.1 * mean if self.z0 else mean
        self._acc = [0.0, 0.0]


class GRPO(Objective):
    """Group relative policy optimization with the clipped surrogate.

    batch.meta: "adv" [B] group-normalized advantages, optional "old_lp"
    [B, T] (sampling-time logprobs when a batch is reused for several
    updates; absent means on-policy, ratio 1 with the policy gradient
    intact) and "ref_lp" [B, T] when beta > 0.

    loss_type:
      dapo     token mean over the whole step (default; no length bias)
      grpo     per-sequence token mean, then mean over sequences (paper)
      dr_grpo  token sum / (sequences * longest completion) (Dr. GRPO)
    """

    name = "grpo"

    def __init__(
        self,
        beta: float = 0.0,
        epsilon: float = 0.2,
        epsilon_high: float | None = 0.28,
        loss_type: str = "dapo",
    ):
        super().__init__()
        self.beta = float(beta)
        self.eps_lo = float(epsilon)
        self.eps_hi = float(epsilon if epsilon_high is None else epsilon_high)
        self.loss_type = loss_type

    @property
    def needs_reference(self):  # type: ignore[override]
        return self.beta > 0

    def loss(self, lp, batch):
        mask = batch.mask
        adv = batch.meta["adv"].astype(mx.float32)[:, None]
        old = batch.meta.get("old_lp")
        old = mx.stop_gradient(lp) if old is None else old
        ratio = mx.exp(lp - old)
        clipped = mx.clip(ratio, 1 - self.eps_lo, 1 + self.eps_hi)
        per = -mx.minimum(ratio * adv, clipped * adv)
        kl = None
        if self.beta > 0:
            d = batch.meta["ref_lp"] - lp
            kl = mx.exp(d) - d - 1
            per = per + self.beta * kl
        if self.loss_type == "dapo":
            loss = (per * mask).sum() / self.norm.tokens
        elif self.loss_type == "grpo":
            n = mx.maximum(mask.sum(axis=-1), 1.0)
            loss = ((per * mask).sum(axis=-1) / n).sum() / self.norm.rows
        elif self.loss_type == "dr_grpo":
            loss = (per * mask).sum() / (self.norm.rows * self.norm.max_len)
        else:
            raise ValueError(f"unknown grpo loss_type {self.loss_type!r}")
        tn = self.norm.tokens
        is_clipped = ((ratio < 1 - self.eps_lo) | (ratio > 1 + self.eps_hi)).astype(mx.float32)
        out = {"clip_frac": (is_clipped * mask).sum() / tn}
        if kl is not None:
            out["kl"] = (kl * mask).sum() / tn
        return loss, out


class Custom(Objective):
    """Wrap fn(lp, batch, norm) -> loss or (loss, metrics)."""

    name = "custom"

    def __init__(self, fn, needs_reference: bool = False, paired: bool = False):
        super().__init__()
        self.fn = fn
        self.needs_reference = needs_reference
        self.paired = paired

    def loss(self, lp, batch):
        out = self.fn(lp, batch, self.norm)
        if isinstance(out, tuple):
            return out
        return out, {}

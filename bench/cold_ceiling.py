#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How good can a cold detector get, before we build one into the cache?

bench/belady_anatomy.py established the shape of the prize and the shape of
the problem:

  * Coarsened Belady - evict ANY entry that will not be touched for 32+ tokens,
    not the furthest one - captures 95% of the oracle's win (-31.6% vs -33.2%
    misses). So the target is a CLASSIFIER, not a regression onto next-use
    order, which is a far lower bar than Belady itself.
  * But the payoff in detector accuracy is violently nonlinear. A detector
    right 90% of the time buys -5.8%; right 100% of the time buys -31.6%.
    Eviction errors compound, because a wrong eviction causes the miss that
    forces the next eviction.
  * A linear detector on 10 causal features reaches AUC 0.774, fires on 4.5%
    of candidates at 74% precision, and yields -2.0% to -2.3% misses when
    gated with an LRU fallback.

So the whole question is whether 0.774 is the information limit of causal
features or just the limit of a linear model on them. If a gradient-boosted
tree on the same inputs reaches 0.9+, precision and coverage both rise and the
prize could be much larger than -2%. If it lands near 0.78, the residency
question is closed and the -2% gated nudge is the whole take.

This script answers that with the strongest off-the-shelf model, on the exact
same samples, so the comparison is apples to apples.

Usage:
    python bench/cold_ceiling.py --bank <bank.npz> --slots 754
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "bench"))
from belady_anatomy import (
    COLD_FEATURES,
    auc,
    build_trace,
    fit_logistic,
    study_cold,
)

from pagedmoe.sidecar.bank import load_bank


def coverage_at_precision(p, y, target=0.7):
    """Largest share of candidates the detector can fire on at >= `target`.

    This is the number that matters operationally: precision alone can always
    be bought by firing almost never (the 0.9 bar measured -0.4% because it
    was silent), so the detector has to be judged on precision AND volume.
    """
    order = np.argsort(-p)
    ys = y[order]
    csum = np.cumsum(ys)
    prec = csum / np.arange(1, len(ys) + 1)
    ok = np.nonzero(prec >= target)[0]
    if ok.size == 0:
        return 0.0, 0.0
    k = int(ok[-1]) + 1
    return k / len(ys), float(prec[k - 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--min-traj", type=int, default=16)
    ap.add_argument("--cold-tokens", type=float, default=32.0)
    args = ap.parse_args()

    trajectories, meta = load_bank(Path(args.bank).expanduser())
    trajectories = [t for t in trajectories if len(t) >= args.min_traj]
    n_layers = int(meta.get("n_layers") or 0) or 1 + max(
        (max(f) for traj in trajectories for f in traj if f), default=0
    )
    kept, total = [], 0
    for traj in trajectories:
        if total >= args.max_tokens:
            break
        kept.append(traj)
        total += len(traj)

    key_ids, layers_of, _tokens, boundaries, n_keys = build_trace(kept, n_layers)
    key_layer = np.zeros(n_keys, dtype=np.int32)
    key_layer[key_ids] = layers_of
    per_token = len(key_ids) / max(1, total)

    X, y = study_cold(
        key_ids,
        layers_of,
        boundaries,
        args.slots,
        n_layers=n_layers,
        key_layer=key_layer,
        per_token=per_token,
        cold_tokens=args.cold_tokens,
    )
    cut = int(len(X) * 0.7)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    print(f"samples {len(X)} ({len(Xtr)} train / {len(Xte)} test), "
          f"base rate {y.mean():.3f}, label = cold for {args.cold_tokens:.0f}+ "
          f"tokens\n")

    w, mu, sd = fit_logistic(Xtr, ytr)
    z = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], axis=1) @ w
    p_lin = 1.0 / (1.0 + np.exp(-z))

    from sklearn.ensemble import HistGradientBoostingClassifier

    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=6, early_stopping=True
    )
    gb.fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]

    print(f"{'model':<22} {'AUC':>6} {'cov@0.7p':>9} {'cov@0.8p':>9} "
          f"{'cov@0.9p':>9}")
    for name, p in (("linear (10 feats)", p_lin), ("gradient boosting", p_gb)):
        c7, _ = coverage_at_precision(p, yte, 0.7)
        c8, _ = coverage_at_precision(p, yte, 0.8)
        c9, _ = coverage_at_precision(p, yte, 0.9)
        print(f"{name:<22} {auc(p, yte):>6.3f} {c7:>9.3f} {c8:>9.3f} "
              f"{c9:>9.3f}")

    print("\nfeature importance by single-feature AUC:")
    for i, nm in enumerate(COLD_FEATURES):
        a = auc(Xte[:, i], yte)
        print(f"  {nm:<9} {a:.3f}" + ("  (inverted)" if a < 0.5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

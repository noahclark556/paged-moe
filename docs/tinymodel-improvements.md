# Multi-head tiny model (2026-09-14, v4 2026-09-15)

Per-model online learner with four independently gated heads. Master switch
`PAGED_MOE_SIDECAR=0` -> never constructed (zero cost, and the decode path
does not even slice out the hidden state). Per-head flags:

| Head | Flag (default) | Quality | Actuation |
|---|---|---|---|
| wrap | `SIDECAR_HEAD_WRAP=1` | safe | next-token early-layer `prefetch_many`, incremental over the engine's last-token replay |
| prefill | `SIDECAR_HEAD_PREFILL=0` | safe | chunk **hot-set** prefetch + post-leave warm |
| residency | `SIDECAR_HEAD_RESIDENCY=0` | safe | eviction protect scores |
| prune | `SIDECAR_HEAD_PRUNE=0` | **affects logits** | adaptive prune/wait (shadow->promote, mass-gated) |

Package layout: `pagedmoe/sidecar/` (`service.py` facade, `heads/*`,
shared `slot.py` / `features.py`). Persist:
`~/.paged_moe/sidecar/<id>_{wrap,prefill,residency,prune}.npz` (or `PAGED_MOE_SIDECAR_DIR`).

## v4 changes

**Features.** The vector is now a fixed four-block layout: hashed age-decayed
demand, per-target-layer stripes, a signed random projection of the last hidden
state, and named scalar channels. The hidden-state block is the substantive
addition - it is what the LM head turns into the next token, so it is by far the
strongest predictor of that token's early-layer routing, and without it the
model can only learn expert co-occurrence. The projection is derived
deterministically from the slot id so persisted weights stay meaningful across
restarts. The demand block is maintained as an in-place EMA rather than rebuilt
per token, which removed ~90% of the sidecar's per-token cost.

**Learner.** Weights are one `(heads, feat, experts)` tensor plus a per-expert
bias, so a token's prediction is a single BLAS call instead of a Python loop.
The bias alone learns each layer's expert popularity, which is most of the
reachable accuracy in the first few hundred tokens. Added AdaGrad, weight decay,
a frequency prior that blends in while the head is cold and decays out as it
learns, and preallocated gradient scratch.

**Actuation and gating.** The wrap head issues only the experts its model
predicts that the previous token did *not* already cover, because the engine's
own prefetch ring already replays those. Coverage therefore cannot regress, and
the head's value becomes directly measurable: `gain` is the recall its extra
reads added, `waste` the share of them nothing wanted. Promote and demote gate on
`gain`, not absolute recall - an absolute bar promotes heads that merely
re-derive the baseline and blocks heads that add real coverage on top of a low
one. Earned actuation now survives a restart.

**Prefill head.** Predicting a chunk's expert *union* was close to worthless: a
few hundred tokens route across nearly every expert, so copying the previous
chunk scored ~1.00 recall while a 12-per-layer guess scored ~0.01. It now targets
the chunk's hot set (experts many tokens shared) and ships the copy baseline as a
free warm whenever one exists, with the model's additions counted separately.

**Residency head (bug).** Reuse windows were appended once per MoE layer rather
than once per token, so on any model with more MoE layers than the horizon a
key's own next appearance always fell outside the window. Every score stayed at
its 0.15 initial value against a 0.35 bar - `protect=0` forever. Now token-
indexed, vectorized over flat `(layer x expert)` tables, lazily aged on read, and
rate-limited so a uniformly hot cache still evicts.

**Geometry.** Feature width, wrap depth, prefill breadth and top-k all scale
from the model's own MoE shape unless an env var pins them, and top-k tracks the
router's observed fan-out instead of assuming top-3 of 8.

**Cost guard.** The sidecar runs inline between tokens, so its own time is token
latency. It measures itself and sheds work past a wall-clock budget - shadow
scoring, then training stride, then top-k, then actuation - so a model it cannot
help does not pay for it. Measured on GLM geometry (89 layers, 160 experts,
4096 hidden): 795 us/token doing all of the above, down from 1756 us for the
smaller v3 feature set.

**Self-tuning.** Top-k walks down while most extra reads go unused and back up
when they pay, bounded by the observed fan-out. The residency bar rises while
protection is pinned at its rate cap (cap-bound protection is no better than
LRU). The prune head's quality gate is relative to the shipped `PRUNE`
threshold's retained mass, not an absolute bar - `PRUNE=0.7` keeps well under
half the mixture mass on GLM and was measured fine, so an absolute bar would
have vetoed the head forever.

## Measured live (glm-47-4bit: 89 MoE layers, 160 experts, 5120 hidden)

| | v3 | v4 |
|---|---|---|
| wrap marginal gain over baseline | not measured | +0.12 ... +0.22 |
| prefill recall vs its baseline | 0.01 vs ceil 1.00 (`plift -0.99`) | 0.71 vs ceil 0.29 (`plift +0.42`) |
| residency protects | 0, always (bug) | firing, `reuse 0.86` |
| prune live | promoted on agreement alone | only at >=0.98 of baseline retained mass |
| sidecar cost / token | 1756 us | ~1000 us (`ovh` 0.5-0.7%) |
| earned actuation across restart | re-earned from zero | resumed |

`lift` and `gain` answer different questions and both matter: `lift` near 0 with
`gain` +0.15 means the model is no smarter than the copy baseline alone, but the
union of the two covers 15% more than the baseline - real value that an
absolute-recall gate reads as failure.

Two latent bugs elsewhere surfaced once the residency head could actually
protect: `ExpertCache._evict_to_budget` called `move_to_end` while iterating its
own `OrderedDict` (now collected and applied after the walk, bounded by
`_PROTECT_SCAN`), and `hash_demand`'s uniform-width fast path mis-detected the
ragged per-layer demand that decode pruning produces.

`glm-47-4bit` ships with all four heads on. Watch `[sidecar] tick` for
`wrap_r/p`, `ceil`, `lift`, `gain`, `waste`, `topk`, `prefill_r`, `residency`,
`prune_live`, `tok/s`, and `us`/`ovh`.

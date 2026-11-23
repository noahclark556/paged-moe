# What lets PagedMoE train models that shouldn't fit

Fine-tuning normally needs more memory than running. On top of the weights
you hold every layer's activations for the backward pass, the gradients, and
the optimizer state. A 235B MoE that needs ~132 GB just to load is not
something you fine-tune on a laptop.

Paged training does it on a 48 GB Mac. The weights stay where the engine
already keeps them (backbone in RAM, experts on the SSD), and the training
loop is built around the one thing that is expensive here: reading experts.

| Model | Checkpoint | Training peak (LoRA, 48 GB Mac) | Mode |
| --- | ---: | ---: | --- |
| **Qwen3-Coder-480B** (4-bit) | **~270 GB** | **~30 GB** | streamed |
| **Qwen3-235B-A22B** (4-bit) | **~132 GB** | **~30 GB** | streamed |
| **PagedMoE-35B-A3B** (4-bit), planned as a 16 GB Mac | **20.4 GB** | **~10 GB** | streamed |
| **PagedMoE-27B-A3B** (4-bit), 4 GB expert cache | **15.9 GB** | **~11 GB** | streamed |
| **PagedMoE-27B-A3B** (4-bit) | **15.9 GB** | **~26 GB** | resident |

PagedMoE-35B-A3B is Qwen3.6-35B-A3B at 4-bit, and the 27B is the same model
with 192 of its 256 routed experts per layer kept. "Training peak" is
measured Metal memory during real steps. Streamed runs
read the experts from disk every step; resident runs are the normal path,
used when the model fits.

The ideas that make that work, one sentence each and then a little more.

---

**Only small adapters train; the base weights never change.**

LoRA adds a pair of thin matrices next to each linear layer and trains only
those, about 0.06% of the parameters on the 35B. The base weights are
frozen, so the experts can stay on disk as read-only data, and gradients and
optimizer state exist only for the adapters (a few hundred MB, not hundreds
of GB). Routers are never adapted, so the trained model still picks experts
the way the weights on disk expect.

---

**A step walks the model one layer at a time, for every micro-batch at once.**

The usual loop runs a whole micro-batch through the model, then the next.
Streamed, that would read every expert once per micro-batch. The layered
executor runs layer 0 for every micro-batch in the step, then layer 1, and
so on. Each layer's experts are needed for one short window on the way
forward and one on the way back, so a step reads the expert mass about
twice no matter how big it is. Same lesson as prefill: cost is priced in
expert bytes, so touch each expert as few times as possible.

---

**The backward pass recomputes each layer instead of storing everything.**

Holding every intermediate value of a 94-layer model for a big batch would
not fit. The forward pass keeps only each layer's input (one hidden state per
token per layer). On the way back, each layer is run again under its own
gradient call, its gradient is taken, and its memory is released before the
next layer. Peak memory is one layer's worth of work plus those saved inputs.

---

**The output head runs in chunks, so full logits never exist.**

A 150k to 250k vocabulary is up to 1 MB of float32 logits per token. The
loss only needs each target token's log probability, so the head runs over
slices of the sequence, keeps those numbers, and does its backward pass slice
by slice too. Losses are written against those per-token log probabilities,
which is also why every method (SFT, DPO, GRPO and the rest) fits the same
loop.

---

**Micro-batching is exact, not an approximation.**

Because the head runs after every micro-batch's forward pass, the loss sees
the whole step at once. Token means, pair counts and GRPO group stats are
computed over the full step before anything is divided up, so splitting a
step into micro-batches changes memory, not the math.

---

**Experts are fetched exactly as the router picks them, and read ahead early.**

A training stand-in for the MoE layer reads the router's picks, fetches those
experts through the same budgeted cache and F_NOCACHE read path as the
engine, and runs the same expert math as a fully loaded model. The backward
pass knows exactly which experts each layer used on the way forward, so it
starts those reads one layer early (~99% hit rate). Read-ahead only changes
timing; a layer always computes with its router's real picks.

---

**Frozen quantized weights get a cheaper backward pass.**

The gradient of a 4-bit matmul with respect to its input runs MLX's
quantized kernel in its slow orientation, about 4x the cost of the forward.
Since the weights never change, the backward instead unpacks the weights it
needs to 16-bit once and does a plain matmul. The forward is the same op as
before, and the gradients match to bf16 rounding. That covers every
quantized layer: experts, attention projections, shared experts, the output
head. On the 235B it took a layer's forward plus backward from ~1.0 s to
~0.6 s.

---

**Hybrid linear attention trains in chunks instead of one token at a time.**

The Qwen3.5 / Qwen3.8 hybrids use a gated delta recurrence in 3 of every 4
layers. mlx-lm's fast kernel for it has no gradient, and the fallback is a
per-token loop that was ~95% of training time. Training swaps in the chunked
form of the same recurrence (matmuls inside 128-token chunks, a short
sequential pass between chunks) with a closed-form backward for its
triangular inverse. Same math, rearranged, so it matches the loop to float32
rounding.

---

**The plan is measured, not guessed.**

Before the first step a short probe runs a few synthetic steps and fits peak
memory and step time against tokens, plus what one more micro-batch costs.
When the size it wants is well past what it measured, it measures that size
too instead of trusting the line. From that it picks the micro-batch size,
the step size (big enough that compute covers the fixed expert reads), and
gives the expert cache whatever
the measured step leaves inside the budget. If the longest rows can't fit
even with the smallest cache, it says which max_seq_len would, and it refuses
to start a step that would go past the GPU working set. The result is cached,
so the second run on the same setup starts right away.

---

**Rows are grouped by length so padding costs less.**

Each step sorts its rows by length and cuts them into micro-batches with the
fewest padded tokens, weighed against what an extra micro-batch costs. A long
row no longer drags a group of short ones up to its length.

---

**Training runs the exact model, never the decode shortcuts.**

Pruning, approximate routing, drafts and the rest of the engine's speed
options are all off and checked after load. Compile is off too, so the
forward pass, the recompute and the reference scores all see identical
numbers. Gradients from the streamed path match the fully loaded path
(tests compare them on every change).

---

**If the model already fits, it trains resident.**

Auto mode only streams when the weights don't fit the budget. A model that
fits keeps the simpler fused path, which is faster. Same config, same
adapter format either way; the adapter loads in `mlx_lm` or
`pagedmoe.load`.

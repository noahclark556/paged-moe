# What lets PagedMoE run models that shouldn't fit

On a normal load, these MoEs need roughly their full checkpoint in RAM.
PagedMoE keeps only a working set resident and frees the rest as headroom -
often tens to hundreds of GB versus a full resident load. Numbers below are
the **48 GB baseline** (M5 Pro); smaller and larger Macs scale from it.

| Model | Normally (full resident) | With PagedMoE | Headroom |
| --- | ---: | ---: | ---: |
| **Qwen3-Coder-Next** (6-bit) | **~65 GB** - won't load | **~30 GB** peak | **+35 GB** |
| **Qwen3-235B-A22B** (4-bit) | **~132 GB** - won't load | **~29 GB** peak | **+104 GB** |
| **GLM-4.7** (4-bit) | **~199 GB** - won't load | **~29 GB** peak | **+170 GB** |

"Normally" is the checkpoint size you'd need to hold the model all at once.
"With PagedMoE" is measured peak Metal memory while generating - the rest
of the weights stay on disk and page in as needed. Headroom is that gap.

A short list of the ideas that make that possible. Each item is one sentence,
then a little more detail - not the full engineering story.

---

**Only the always-on backbone stays in RAM; routed experts live on disk until needed.**

A Mixture-of-Experts model is huge mostly because of experts that almost never
fire on any given token. PagedMoE keeps the small shared pieces (attention,
routers, embeddings, and similar) resident, and treats the expert mass as a
library on the SSD. That is the single reason a 65 GB coder can start on a
16 GB Air, or a 199 GB GLM on a 32 GB Pro.

---

**Experts are fetched from the SSD on demand, instead of loading the whole checkpoint.**

When the router picks experts for a token, those weights are read from disk into
a working set, used, and can be dropped later. The model never has to be fully
materialized in memory the way a normal load would demand.

---

**A large in-memory expert cache keeps recently used experts warm.**

Disk is slow if you hit it every time. A big LRU-style cache holds as many
experts as your RAM budget allows, so repeats and nearby tokens often find what
they need already loaded. The machine still "owns" far less than the full model,
but the hot path feels closer to RAM.

---

**Expert reads bypass the OS page cache so memory isn't double-spent.**

Without this, every SSD read also piles into the kernel's cache - a second copy
of the same bytes that can quietly swap the machine to death. Skipping that
hidden pile-on keeps the budget honest: RAM goes to the backbone, the expert
cache, and KV, not to an invisible duplicate.

---

**The engine starts reading likely next experts before the GPU needs them.**

There is a short window between finishing one token and needing the next token's
early experts. Prefetch and route prediction use that window to get bytes moving
early, so disk wait is hidden behind useful work instead of sitting on the
critical path.

---

**A tiny online predictor (the sidecar) guesses which experts come next and prefetches them.**

Copying "whatever the last token used" already helps; a small continuously
trained linear model does better by learning patterns from recent demand and the
last hidden state. It only pulls extras the simple replay would miss, and it can
step back if those extras stop paying off. This is why long agent sessions feel
smoother than raw streaming alone.

---

**Optional decode pruning skips weak experts so fewer SSD reads happen per token.**

When enabled, experts that carry almost none of the mixture can be skipped
instead of blocking on a full read. Quality is traded carefully (and measured);
the win is fewer bytes and less stall on disk-bound models.

---

**Memory is reserved for context (KV) so a long conversation doesn't steal the expert cache.**

Full-attention models burn real RAM as the prompt grows. The system sizes the
expert cache knowing how much context you intend to keep, so mid-chat you don't
suddenly dump the experts that were keeping decode fast.

---

**Long prefills temporarily shrink the expert footprint so Metal doesn't OOM.**

A big prompt touches almost everything once. During that pass the system can
release or shrink aggressive expert residency, finish the prompt, then warm back
up for decode. That keeps huge turns from crashing the process even when peak
demand is higher than steady decode.

---

**Agent turns reuse the prompt prefix instead of redoing the whole history.**

After the first turn, only new tokens need to be prefilled. That matters more
than raw tok/s for agents: each tool round would otherwise re-walk a huge
expert mass. Prefix reuse is a big part of why multi-step sessions stay usable.

---

**If the model already fits, it can run fully resident; streaming is the escape hatch.**

Auto mode only pages from disk when the checkpoint won't fit the RAM budget.
Small models keep the simple path. Oversized MoEs get the streamed path. Same
API either way - the capacity leap is for the cases that used to be a hard no.

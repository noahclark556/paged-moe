# Paged training

Fine-tune MoE checkpoints on a Mac, including ones that do not fit in memory.
Same idea as the engine: the backbone stays resident, the experts page from
the SSD through a budgeted cache, and everything that can be measured or
detected is planned for you. LoRA adapters come out in mlx-lm format, so they
load with `mlx_lm`, `pagedmoe.load(adapter_path=...)` and anything built on
those, and `export` writes a PEFT copy for GPU users (see
[On a GPU](#on-a-gpu-peft-export)).

Methods: SFT, CPT (continued pretraining), ORPO, DPO (sigmoid / IPO / hinge),
SimPO, CPO, KTO, GRPO. Custom objectives, custom rows, custom input
embeddings and trainable encoders plug into the same loop.

The package lives in `pagedmoe/train/`. If you'd rather not touch a
terminal, [Paged Trainer](https://pagedmoe.com/train) is a Mac app that
drives this same trainer from a UI: every method and setting, live stats,
checkpoints, evaluation and the bench. The short version of why any of this
fits is in [why-it-trains.md](./why-it-trains.md).

- [Quick start](#quick-start)
- [Testing it](#testing-it)
- [Methods and data](#methods-and-data)
- [What gets planned](#what-gets-planned)
- [Config reference](#config-reference)
- [Output, checkpoints, resume](#output-checkpoints-resume)
- [Stats and logging](#stats-and-logging)
- [Using the adapter](#using-the-adapter)
- [Evaluating a run](#evaluating-a-run)
- [Benchmark](#benchmark)
- [Custom data, inputs and objectives](#custom-data-inputs-and-objectives)
- [How it works](#how-it-works)
- [Measured](#measured)
- [Architectures](#architectures)
- [Limits](#limits)

---

## Quick start

With the package installed (`./install_mlx.sh`, see the README) and
`~/.mlx-env` active. `paged-moe-train` and `python -m pagedmoe.train` are the
same command, and `python -m` also works from a clone without installing:

```bash
# see the plan first: mode, memory, batch, speed estimate. Nothing trains.
python -m pagedmoe.train plan sft --model ~/mlx-models/Qwen3.6-35B-A3B-4bit --data data/chat.jsonl

# train
python -m pagedmoe.train sft --model ~/mlx-models/Qwen3.6-35B-A3B-4bit --data data/chat.jsonl \
    --epochs 2 --template-kwargs '{"enable_thinking": false}'

# or from a config file (examples in pagedmoe/train/examples/)
paged-moe-train pagedmoe/train/examples/sft.yaml

# continue a stopped run
python -m pagedmoe.train resume ~/.paged_moe/train/runs/<run>
```

Any config field works as a flag: `--batch-size 16`, `--lora.rank 32`,
`--mode streamed`, `--cache-gb 6`. Values parse as JSON when they can.

From Python:

```python
from pagedmoe.train import SFTConfig, SFTTrainer

trainer = SFTTrainer(SFTConfig(
    model="~/mlx-models/Qwen3.6-35B-A3B-4bit",
    data="data/chat.jsonl",
    epochs=2,
    template_kwargs={"enable_thinking": False},
))
print(trainer.plan)        # the decisions, as a dataclass
trainer.train()
```

Every trainer takes the same keyword arguments on top of the config:
`train_data=`, `eval_data=`, `formatter=`, `encoder=`, `objective=`,
`callbacks=`, `tokenizer=`, plus any config field as an override
(`SFTTrainer(cfg, learning_rate=5e-5)`).

`plan_only=True` reads the data and plans without loading the model (under a
second). `trainer.plan_info()` returns the plan, data summary, skipped rows
and estimate as a dict, which is what the UI shows.

Parquet files and Hugging Face datasets need two extra packages:
`pip install -e ".[train]"` (pandas and datasets). Everything else is in the
base install.

---

## Testing it

`pagedmoe/train/examples/sandbox.sh` runs everything.

```bash
# unit checks in a throwaway venv under /tmp (pinned to the tested versions, needs uv)
pagedmoe/train/examples/sandbox.sh

# same checks in the env you are in
source ~/.mlx-env/bin/activate
pagedmoe/train/examples/sandbox.sh --here

# plus short SFT, streamed SFT, ORPO and GRPO runs on a real model
MODEL=~/mlx-models/Qwen3.6-35B-A3B-4bit pagedmoe/train/examples/sandbox.sh --here

# same, planned as if this were a 16 GB Mac (the planner goes streamed on its own)
SIM_RAM_GB=16 MODEL=~/mlx-models/Qwen3.6-35B-A3B-4bit pagedmoe/train/examples/sandbox.sh --here
```

It runs from source through `PYTHONPATH`, so nothing gets installed into
your env. Outputs go to `/tmp/paged-train-check`.

The checks:

| Test | What it proves |
| --- | --- |
| `tests/test_train_linear_attn.py` | the chunked gated delta matches mlx-lm's per-token loop (outputs, state, gradients), including the worst case for its inverse |
| `tests/test_train_parity.py` | fused, checkpointed, layered and streamed-layered gradients all match plain `value_and_grad`; read-ahead hits on the second step |
| `tests/test_train_methods.py` | every method trains end to end on a tiny MoE, resident and streamed (streamed losses equal resident ones), resume, adapter round trip through `mlx_lm` and `pagedmoe.load`, per-row embeds, trainable encoder |
| `tests/test_train_fastgrad.py` | the fast backwards give MLX's forward exactly and its gradients to bf16 rounding (dense, expert, paired gate / up, token sort, long causal attention, LoRA through an installed model) |
| `tests/test_train_data.py` | chat templates that reject null content, broken templates, and the micro-batch split (limits held, never worse than a greedy fill) |
| `tests/test_train_export.py` | PEFT export: name mapping (multimodal split, mtp skipped), transposes and alpha, DoRA, routed experts skipped, wrong-model errors, `export_peft` at the end of a run and the CLI, and a PEFT load when torch and peft are installed |
| `bench/train_parity_real.py` | the same parity on a real checkpoint: `python bench/train_parity_real.py ~/mlx-models/Qwen3.6-35B-A3B-4bit --cache-gb 4` |

One GPU job at a time: stop any engine server or other MLX app before a real run.
Each test file is a script that prints OK at the end:
`for t in tests/test_train_*.py; do python "$t"; done`. pytest skips them,
since they switch the whole process into training mode.

---

## Methods and data

Rows are dicts from a `.jsonl` / `.json` / `.parquet` / `.csv` file, a
directory holding `train.*` and `valid.*`, a `.npy` / `.npz` of token ids, a
Hugging Face dataset (`hf:org/name` or `org/name`), or a Python list.

| Method | Row format | Notes |
| --- | --- | --- |
| `sft` | `{"messages": [...], "tools": [...]?}` | assistant turns are trained, the rest is context. ShareGPT `{"conversations": [{"from", "value"}]}` works too |
| | `{"prompt": ..., "completion": ...}` | completion trained |
| | `{"input_ids": [...], "labels" or "loss_mask"?}` | already tokenized; labels of -100 are ignored |
| `cpt` | `{"text": ...}` | packed into max_seq_len windows, every token trained |
| `orpo`, `dpo`, `simpo`, `cpo` | `{"prompt": ..., "chosen": ..., "rejected": ...}` | strings or message lists. Without a prompt, chosen / rejected are whole conversations and the shared prefix is the prompt |
| `kto` | `{"prompt": ..., "completion": ..., "label": true/false}` | unpaired good / bad examples |
| `grpo` | `{"prompt": ..., <anything else>}` | other fields go to the reward functions (`answer`, test cases, a file path) |

`prompt`, `completion`, `chosen` and `rejected` can be strings or message
lists. Chat templates come from the tokenizer; `--template-kwargs` passes
through to it (`enable_thinking`, etc), `system_prompt` adds a system turn
when a row has none, `chat_template: false` trains on raw text.

When a template rewrites earlier turns (some strip old reasoning), only the
final assistant turn has a stable prefix, so only that turn is trained and
a one-time note says so.

Method knobs, on top of the shared ones:

| Method | Fields (defaults) | Default lr |
| --- | --- | --- |
| sft | `packing` (false) | 1e-4 |
| cpt | `packing` (true) | 1e-4 |
| orpo | `beta` (0.1, the odds-ratio weight) | 5e-5 |
| dpo | `beta` (0.1), `loss_type` sigmoid / ipo / hinge, `label_smoothing` | 2e-5 |
| simpo | `beta` (2.0), `gamma` (1.0), `sft_weight` (0) | 2e-5 |
| cpo | `beta` (0.1), `nll_weight` (1.0) | 2e-5 |
| kto | `beta` (0.1), `desirable_weight`, `undesirable_weight` | 2e-5 |
| grpo | `num_generations` (8), `max_completion_tokens` (512), `temperature`, `top_p`, `beta` (0, KL weight), `epsilon` (0.2), `epsilon_high` (0.28), `loss_type` dapo / grpo / dr_grpo, `scale_rewards`, `iterations` (1), `reward_weights`, `skip_flat_groups` | 5e-6 |

DPO and KTO need reference logprobs. They are scored once before training
with the LoRA switched off (same weights, no second model in memory) and
cached in the run dir as `reference_logps.npy`.

### GRPO rewards

```bash
--rewards exact_answer,think_format
--rewards 'exact_answer,regex:<answer>\d+</answer>'
--rewards my_rewards.py:score_fn,exact_answer --reward-weights '[1.0, 0.2]'
```

Builtins: `exact_answer` (last number, `\boxed{}` or `#### x` against
`row["answer"]`), `think_format`, `regex:<pattern>`, `length_penalty` (from
Python). Your own functions take either signature:

```python
def per_sample(completion: str, row: dict) -> float: ...
def trl_style(prompts, completions, **columns) -> list[float]: ...   # columns = dataset fields
```

Return None for "no opinion". Rewards see the completion without the stop
token; training keeps it, so the model learns to stop. Groups where every
completion scored the same carry no signal and are skipped, with the reward
and a sample printed so you can see why.

Rollouts run on the engine's own decode path (streamed experts swapped back
in, compiled kernels on) with all prompts x generations in one batch.

---

## What gets planned

Construction prints the plan, for example (Qwen3.6-35B-A3B on a simulated
16 GB Mac; PagedMoE-35B-A3B is mlx-community's 4-bit conversion of it):

```
paged training plan
  method    sft
  model     PagedMoE-35B-A3B-4bit (qwen3_5_moe), 40 layers, 256 experts / 8 active
  size      20.4 GB (experts 18.1 GB, backbone 2.3 GB)
  memory    11.7 GB training budget
  mode      streamed  (20.4 GB of weights against a 11.7 GB budget)
  executor  layered  (streamed experts: layer-major so each step reads them ~twice)
  cache     5.51 GB of experts held between reads
  lora      LoRA 21.17M trainable (0.060%) on 40 layers, 12 module kinds
  data      64 examples, mean 42 tokens (p95 42, max 42), max_seq_len 128
  batch     16 examples/step (~666 tokens), micro-batches <= 1024 tokens  (...)
  steps     4/epoch, 4 total, lr 0.0001 cosine
  estimate  ~143 tok/s, ~4.7 s/step, ~18s total  (probe: measured)
  reads     ~17 GB/step (0.9x the expert mass)
```

| Decision | Auto rule | Override |
| --- | --- | --- |
| mode | resident when the weights plus headroom (12% of the budget, at least 2 GB) fit, else streamed. Budget is the engine's `RAM_FRACTION` of RAM, capped by the Metal working set, and lowered at start when the Mac is already under memory pressure | `mode` |
| executor | streamed: layered. Resident: fused when a fused micro-batch of a few of the longest rows fits, else checkpoint | `executor` |
| micro-batch | estimated from per-token activation cost, then fit to the probe's measured memory line | `micro_batch_tokens` |
| step size | resident: ~16k tokens. Streamed: at least ~16k tokens, since each step pays a fixed cost for every expert it touches whatever its size. Step tokens can take memory from the expert cache down to its floor to get there: on a model many times the budget the cache hits ~15% at best, and bigger steps are worth more. Small datasets keep at least 4 to 8 steps per epoch | `batch_size` |
| cache | streamed: small enough for the probe to run safely, then refit to what the measured step leaves after the backbone and activations. The backbone is what a streamed load keeps resident: PLE n-gram tables (read from disk per token) and vision / MTP weights do not count | `cache_gb` |
| max_seq_len | p99.5 of the formatted rows, up to 8192 on a 48 GB Mac and more on bigger ones (32768 max) | `max_seq_len` |
| head chunk | about 3% of the budget in fp32 logits at a time, 0.25 to 4 GB (a 248k vocab is 1 MB per token) | none |
| learning rate | per method (table above), cosine with 3% warmup | `learning_rate`, `lr_schedule`, `warmup_ratio` |

The probe runs a warm-up and three short synthetic steps (one micro-batch,
one twice as big, and the same tokens as two micro-batches), fits peak
memory and step time as lines in tokens plus the cost of one more
micro-batch. When the micro-batch it picks is more than twice the biggest
size it measured, it steps up in doublings, predicting each from the
steepest slope so far, and stops before a size that would not fit. The
micro-batch is never more than one doubling past a measured size.
It caches the result in `~/.paged_moe/train/profiles/` (keyed
by model, RAM, mode, executor, LoRA shape, a profile version, the fast
backward switch, and cache size when you set one; an auto cache is refit
from the size the profile was measured at). The estimate runs the real
schedule to count padded tokens and micro-batches per step. The second run
on the same setup skips the probe. If even the smallest cache leaves the
planned step over the budget (very long rows on a big model), the plan warns
with a `max_seq_len` that fits, and refuses to start when the step would go
past the GPU working set. `probe: always` forces it, `probe: never`
skips it (and the estimate). With an encoder there are no synthetic rows, so
there is no probe.

`PAGED_MOE_SIMULATE_RAM_GB=16` plans as if the Mac had 16 GB (Metal still
uses the real GPU).

Bigger and smaller Macs. Everything above is relative to the budget, so
nothing needs setting by hand. The caps that would otherwise hold a big Mac
back (micro-batch size, step size, max_seq_len, scoring groups, head chunk)
grow with the budget past a 48 GB Mac, in powers of two. Disk speed comes
from the engine's hardware probe, so read times are planned from this SSD
and not a fixed number. The step size target itself is not scaled: batch
size is a training choice, and it stays the same on every Mac unless you
set it.

Memory guard. While training, the trainer watches macOS memory pressure and
the compressor. When pressure goes critical, or the compressor grows by a
real amount, it gives memory back: first the streamed expert cache shrinks
(down to its floor, unless you pinned `cache_gb`), then the micro-batch
halves. Rows per step stay the same, so the gradients do not change, only
the speed. It prints a `[train] memory pressure` line when it acts.
`memory_guard: false` turns it off.

The first real steps also check the plan. If a streamed step peaks over the
planned memory (the probe line can miss by a few GB on some models), the
expert cache shrinks by the overshoot and the correction is saved with the
probe profile, so the next run on that model plans it in from the start. It
prints a `[train] step peaked at` line. On resident runs MLX's pool of freed
buffers is also kept to what the measured step leaves of the budget, so it
can't push the Mac into swap while the step itself fits.

---

## Config reference

Shared fields (`TrainConfig`). Anything left at None is planned.

| Field | Default | |
| --- | --- | --- |
| `model` | required | checkpoint dir (mlx-community safetensors) |
| `data`, `eval_data` | | see [Methods and data](#methods-and-data) |
| `output_dir`, `name` | `~/.paged_moe/train/runs/<name>` | `PAGED_MOE_TRAIN_DIR` moves the root |
| `epochs`, `max_steps` | 1, None | max_steps wins when set |
| `batch_size` | planned | rows / pairs / prompts per optimizer step |
| `micro_batch_tokens`, `max_seq_len` | planned | |
| `learning_rate` | per method | |
| `lr_schedule`, `warmup_ratio`, `min_lr_ratio` | cosine, 0.03, 0.1 | cosine / linear / constant |
| `optimizer` | adamw | adamw / adam / sgd / adafactor |
| `weight_decay`, `adam_betas`, `max_grad_norm` | 0, (0.9, 0.999), 1.0 | |
| `lora` | rank 16, alpha 32, target all-linear | see below |
| `mode`, `executor`, `cache_gb`, `probe` | auto | see [What gets planned](#what-gets-planned) |
| `memory_guard` | true | give memory back under pressure, see above |
| `chat_template`, `completion_only`, `template_kwargs`, `system_prompt` | auto, true, {}, None | |
| `seed` | 0 | data order, LoRA init, dropout |
| `save_steps`, `save_minutes`, `keep_checkpoints` | 50, 20, 3 | whichever comes first; 0 / None turns one off |
| `resume` | false | true / "latest" / a checkpoint path |
| `eval_steps`, `eval_batches` | None, 16 | |
| `log_steps`, `report_to`, `verbose` | 1, [], true | `report_to: [wandb]` |
| `export_peft` | false | also write `<run>/peft/` (PEFT format) when the run ends, see [On a GPU](#on-a-gpu-peft-export) |

LoRA (`lora:` block or `--lora.<field>`):

| Field | Default | |
| --- | --- | --- |
| `rank`, `alpha`, `dropout` | 16, 32, 0 | scale = alpha / rank |
| `target` | all-linear | `attention`, `all-linear`, `experts` (routed experts too, resident only), `custom` |
| `keys` | None | module names for `target: custom`, e.g. `["self_attn.q_proj", "mlp.shared_expert.down_proj"]` |
| `num_layers` | -1 | last N decoder layers |
| `dora` | false | not with DPO / KTO (their reference is LoRA-off) |

Routers are never adapted: changing what the router picks is the one thing
that would make the trained model route differently from the weights on disk.

---

## Output, checkpoints, resume

```
<run>/
  adapters.safetensors        latest adapter (mlx-lm format)
  adapter_config.json
  encoder.safetensors         when training an encoder
  metrics.jsonl               one line per step and per eval
  run.json                    config, plan, CLI extras, summary
  reference_logps.npy/.json   DPO / KTO reference cache
  peft/                       PEFT copy of the adapter, after an export
  checkpoints/
    latest
    step-000050/              adapters, optimizer state, trainer_state.json (+ encoder)
```

Saves happen every `save_steps` or `save_minutes` and at the end. SIGTERM
finishes the step in flight, saves and stops; Ctrl-C stops right away and
saves the last finished step. The last `keep_checkpoints` are kept.

Resume restores adapters, optimizer state, step, epoch and position in the
epoch. It refuses a checkpoint whose LoRA shape or data does not match the
run, and drops metrics past the checkpoint so the log has no repeats. The
micro-batch is planned fresh for this Mac (it does not change the math). Data order is a pure function of (seed, epoch), so a resumed run sees
exactly the batches it would have seen.

```bash
python -m pagedmoe.train resume ~/.paged_moe/train/runs/<run>            # same settings
python -m pagedmoe.train resume ~/.paged_moe/train/runs/<run> --max-steps 400
```

From Python: `resume=True` (latest), or a path to a specific checkpoint. To
pick a run back up with the settings it was started with:

```python
from pagedmoe.train.cli import resume
resume("~/.paged_moe/train/runs/<run>", {"max_steps": 400}).train()
```

---

## Stats and logging

Each step:

```
step 2/4 | loss 0.0693 | lr 8.7e-05 | gn 2.17 | 203 tok/s | 5.1s | read 11.4 GB (0.6x) | peak 10.3 GB | eta 11s
```

`tok/s` counts every position processed, which is what the estimate
predicts. `read` is expert bytes read from disk that step and how many
passes over the expert mass that is. Preference methods add `acc` and
`margin`, GRPO adds `reward`, `kl` and `clip_frac`. Every 25 steps a second
line breaks the step time down (forward, head, backward, fetch) and for
streamed runs shows experts fetched, read-ahead hit rate, cache hit rate and
disk wait.

`metrics.jsonl` has all of it per step, plus `loss_tokens`, the timing
breakdown, and for GRPO `gen_s`, `gen_tok_s`, `completion_len`,
`finished`, `live_groups` and one `r_<name>` per reward function.

When a run starts, the CLI and Python print one line pointing at
[Paged Trainer](https://pagedmoe.com/train). `PAGED_MOE_TRAIN_HINT=0` hides it.

Callbacks:

```python
from pagedmoe.train import Callback

class Notify(Callback):
    def on_step_end(self, trainer, step, logs): ...
    def on_evaluate(self, trainer, step, logs): ...
    def on_save(self, trainer, step, path): ...
    def on_train_end(self, trainer, logs): ...

SFTTrainer(cfg, callbacks=[Notify()]).train()
```

---

## Using the adapter

```python
from pagedmoe import load
model, tok = load("~/mlx-models/Qwen3.6-35B-A3B-4bit", adapter_path="~/.paged_moe/train/runs/<run>")

# or plain mlx-lm on a model that fits
from mlx_lm import load
model, tok = load("~/mlx-models/Qwen3.6-35B-A3B-4bit", adapter_path="<run>")
```

```bash
mlx_lm.generate --model ~/mlx-models/Qwen3.6-35B-A3B-4bit --adapter-path <run> --prompt "..."
mlx_lm.fuse --model ~/mlx-models/Qwen3.6-35B-A3B-4bit --adapter-path <run>   # bake it in
```

A trained encoder is a normal MLX module: rebuild it and
`encoder.load_weights("<run>/encoder.safetensors")`.

### On a GPU (PEFT export)

The mlx-lm adapter only loads in MLX. For NVIDIA / AMD, export it to Hugging
Face PEFT format, which transformers, vLLM and SGLang load directly and
llama.cpp converts to GGUF:

```bash
python -m pagedmoe.train export ~/.paged_moe/train/runs/<run>                         # -> <run>/peft/
python -m pagedmoe.train export <run> --checkpoint step-000200 --out ~/my-adapter-peft
python -m pagedmoe.train export <run> --base Qwen/Qwen3.6-35B-A3B   # when it can't tell the original model
```

```python
from pagedmoe.train import to_peft
to_peft("~/.paged_moe/train/runs/<run>")      # or a checkpoint folder
```

`export_peft: true` in a config (or `--export-peft true`) does it at the end
of the run. Paged Trainer has an **Export for GPU (PEFT)** button on each run
and an option to export when a run finishes.

What it does:

- Converts only the adapter (lora_a / lora_b transposed into PEFT's lora_A /
  lora_B, `lora_alpha = scale * rank`, DoRA magnitudes when it's DoRA), so it
  takes seconds on any size of model.
- Finds the original HF repo from the checkpoint's model card (`base_model:`
  or "converted to MLX format from ..."). Runs also save that repo in
  `adapter_config.json`, so a moved or unmounted model folder doesn't matter.
- Maps the MLX layer names to the original model's and checks every name and
  shape against its safetensors headers on Hugging Face (headers only, no
  weights). A multimodal Qwen3.5 / 3.6 is `language_model.model.layers.N` in
  MLX and `model.language_model.layers.N` in HF, for example. Anything that
  doesn't line up stops the export. Offline it still exports from the MLX
  names and says they weren't checked.
- Writes `adapter_model.safetensors`, `adapter_config.json` and a README with
  the load commands.

Load it on the original full precision model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "<run>/peft")
```

```bash
vllm serve Qwen/Qwen3.6-35B-A3B --enable-lora --lora-modules mine=<run>/peft
python convert_lora_to_gguf.py <run>/peft --base <original model folder>   # llama.cpp
```

Things to know:

- It was trained against the 4-bit weights, so on the full precision model it
  works like a QLoRA adapter: same fine-tune, small numeric drift. If you want
  it exact, `mlx_lm.fuse --dequantize` still works on a model that fits.
- Routed-expert LoRA (`lora.target: experts`) is skipped with a note: MLX
  stacks the experts into one tensor and there's no one PEFT layout for that.
  Attention, shared experts and dense MLPs export.
- A run on a pruned PagedMoE package (the 27B) exports for the full model and
  says so. The full model also has the experts the pruned copy dropped.
- A trained input encoder isn't part of a PEFT adapter and isn't exported.

Checked by loading the export with PEFT and comparing the adapter's effect
against MLX (LoRA and DoRA) on Llama, Qwen3-MoE, Qwen3-Next, DeepSeek V3 /
Kimi, GLM-4.5 and up and Mixtral, plus name and shape checks on gpt-oss and
the real Qwen3.6-35B, 27B and Qwen3-Coder-480B. `tests/test_train_export.py`
covers the mapping, DoRA, the errors and the end-of-run export.

---

## Evaluating a run

Any checkpoint of a run can be loaded and compared against the base model,
during training or after it (not at the same time as a training job, one
GPU job at a time).

```bash
# talk to it. /use step-000050, /base, /reset, /quit
python -m pagedmoe.train chat ~/.paged_moe/train/runs/<run>

# loss and answers per checkpoint, base model included
python -m pagedmoe.train eval ~/.paged_moe/train/runs/<run> \
    --data eval.jsonl --checkpoints base,step-000050,final --generate \
    --max-tokens 512 --out results.csv
```

`eval` scores loss on each row's completion (the same masking training
uses) and with `--generate` writes an answer per row, scored against the
row's reference as exact match, contains and token F1. GRPO reward
functions work too (`--rewards`). Without `--data` it uses the run's
eval data, else its training rows (that measures fit, not
generalization). Each report is saved under `<run>/evals/` as JSON;
`--out` also writes `.csv`, `.jsonl` or `.json`.

From Python:

```python
from pagedmoe.train.evaluate import Evaluator

ev = Evaluator("~/.paged_moe/train/runs/<run>", checkpoint="final")
for piece in ev.chat([{"role": "user", "content": "hi"}]):
    print(piece.get("text", ""), end="")
report = ev.run(rows, checkpoints=["base", "final"], generate=True)
ev.close()
```

Adapters on the routed experts load resident (the streamed path has no
expert LoRA).

---

## Benchmark

`python -m pagedmoe.train bench` measures training speed and memory on this
Mac with the same trainer you train with. Each case runs in its own process
on a fixed synthetic corpus (same rows on every Mac), so numbers from
different machines compare.

```bash
python -m pagedmoe.train bench status              # ladder models found here, then every other checkpoint
python -m pagedmoe.train bench run                 # every ladder model found, SFT, 2048 tokens, LoRA r16
python -m pagedmoe.train bench run ~/mlx-models/Qwen3.6-35B-A3B-4bit ~/mlx-models/other-model \
    --method sft,dpo --seq 2048,8192 --rank 16,64 --steps 6
python -m pagedmoe.train bench table               # Markdown; --format csv / json
python -m pagedmoe.train bench table --doc TRAIN_BENCHMARK.md
```

Per case: a probe, one warm-up step, then the timed steps. The result has
the machine (chip, RAM, SSD speed, budget, versions), the plan (mode,
executor, cache, batch, micro-batch), tok/s, s/step, hours per 1M tokens,
an epoch of 1k rows, how close the planner's estimate was, peak Metal and
process memory, swap and pressure during the run, GB read per step for
streamed runs, load and probe time, and the checkpoint (adapter size, save
and load time). LoRA on a 4-bit base is the QLoRA setup; the table says
which.

Any checkpoint can be benched, not only the ladder. `bench run` takes a
ladder id, a folder name or a path. Models are looked for in
`~/mlx-models` and `~/models` (two levels down), or the folders in
`PAGED_MOE_BENCH_DIR` (`:` separated). A ladder model is recognized by its
config (architecture, layers, hidden size, experts, bits), so a renamed
folder still lands on the right row; anything else is benched under its
folder name.

Memory in the tables is GiB, the unit a Mac's RAM is sold in (a 48 GB Mac
has 48 GiB, about 51.5 decimal GB). The trainer logs and the JSON keep
decimal GB, and each case's details show both. Model size and disk reads
are decimal GB, like Finder.

Results go to `~/.paged_moe/train/bench/results/` as one JSON per case, so
a ladder can be stopped and continued (`--skip-done`). Ctrl-C stops the
case in flight; it is saved as stopped and left out of the table, and
`--skip-done` runs it again. The Paged Trainer app has the same thing in
its Bench tab.

---

## Custom data, inputs and objectives

The loop does not care what a row is, as long as something turns it into
token ids and a loss mask. Batching, masking, streaming, checkpoints and
stats all stay the same.

### Custom rows

```python
from pagedmoe.train import Example, SFTTrainer

def format_row(row, fmt):
    # fmt is the run's Formatter: .encode(text), .render(messages), .tok, .pad_id, .eos
    q = fmt.encode(row["question"])
    a = fmt.encode(row["answer"]) + [fmt.eos]
    return Example(q + a, [0] * len(q) + [1] * len(a), row=row)

SFTTrainer(cfg, formatter=format_row).train()
```

Pair trainers expect a `Pair(chosen, rejected)` and GRPO a `Prompt(ids,
prompt, row)`. CLI: `--formatter my_rows.py:format_row`.

### Inputs that are not tokens

An `Example` can carry `embeds` ([P, hidden] floats) and `embeds_at`: those
vectors replace P placeholder positions in the input. Fixed features (a
frozen encoder's output, precomputed in the formatter) go there directly.

For an encoder that trains, pass `encoder=` (any `nn.Module` whose
`__call__(batch)` returns [B, P, hidden]). It gets `batch.rows`, so it can
load whatever the row points at. Its gradient comes from the model's
backward pass, it has its own optimizer state, and it is saved and resumed
with the adapter. It works in streamed mode too.

`pagedmoe/train/examples/points_encoder.py` is a worked example: each row
points at a `.npy` point cloud, which is furthest point sampled to N points,
centered, Fourier encoded and projected to the hidden size, followed by the
CadQuery code that builds the shape.

```bash
POINTS_ROOT=~/data/cadquery-points \
python -m pagedmoe.train sft --model ~/mlx-models/Qwen3.6-35B-A3B-4bit \
    --data $POINTS_ROOT/train.jsonl \
    --formatter pagedmoe/train/examples/points_encoder.py:format_row \
    --encoder pagedmoe/train/examples/points_encoder.py:make
```

An `--encoder` factory gets `hidden=<model hidden size>` when it takes that
argument. `POINTS_N` and `POINTS_FREQS` set the cloud size and Fourier bands.

Swap the encoder for an image tower, an audio frontend, or a frozen
pretrained encoder with a small trainable adapter and nothing else changes.

### Custom objectives

Objectives see per-token target logprobs `lp` [B, T] and the batch (mask,
meta, rows), never the full logits. That is what lets the head run in
chunks.

```python
import mlx.core as mx
from pagedmoe.train import Custom, SFTTrainer

def focal(lp, batch, norm):
    p = mx.exp(lp)
    loss = -(((1 - p) ** 2) * lp * batch.mask).sum() / norm.tokens
    return loss, {"p_mean": (p * batch.mask).sum() / norm.tokens}

SFTTrainer(cfg, objective=Custom(focal)).train()
```

`norm` is the whole step's `StepNorm` (tokens, rows, pairs, chosen_tokens,
max_len), computed before any micro-batch runs, so dividing by it makes the
micro-batch losses add up to the full-batch loss. Metrics are scalars.
`Custom(fn, paired=True)` gets pair batches (chosen rows first, then
rejected, `batch.meta["pairs"]`), `needs_reference=True` gets
`batch.meta["ref_seq"]`.

### A new method

A method is a subclass with its own objective and batch maker:

```python
class MyTrainer(Trainer):
    method = "sft"                       # which config class / defaults to use
    def make_objective(self): ...
    def next_batches(self, micro_units): ...   # list[Batch] for one step, or None to skip it
```

GRPO is one of these: `next_batches` samples, scores and builds the batches.

---

## How it works

The engine's rule for prefill holds for training too: cost is priced in
expert bytes, so the trick is to touch each expert as few times per step as
possible.

**Layered executor (streamed).** The forward pass runs layer 0 for every
micro-batch in the step, then layer 1, and so on, keeping only each layer's
input (one hidden state per token per layer). The head runs in chunks over
the whole step, the objective turns logprobs into d loss / d logprob, and
the backward pass walks the layers in reverse, recomputing each one under
its own vjp. Each layer's experts are needed for two short windows, so a
step reads the expert mass about twice however many micro-batches it holds.
And because the head sees the whole step, step-level normalizers (token
means, GRPO groups) are exact.

**Expert fetch.** A training stand-in for the switch layer reads the router's
picks, fetches exactly those experts through the LRU cache, and runs the
same `gather_qmm` math as mlx-lm. Read-ahead is timing only: the backward
pass prefetches the experts the forward pass used for that layer, the next
step's forward prefetches the previous step's set. A layer always computes
with the router's actual picks. Expert reads go through the same F_NOCACHE,
1 MB-capped path as the engine.

**Fused executor (resident).** One `value_and_grad` per micro-batch,
optionally with per-layer checkpointing and a chunked, checkpointed head.
Fastest when everything is in memory; not usable streamed, since MLX builds
the whole backward graph before running it.

**Exact model.** Training runs the engine's exact profile: prune, route
top-k/p, draft, slab, stitch, fused prefill, verify, sidecar, predict and
learned margin are all off, and checked after load. Two MLX details on top:

- MoE routers do `take_along_axis(scores, argpartition(scores))`, and MLX
  refuses to differentiate through the integer indices. The arg* ops return
  stop_gradient'ed indices while training, which is exact (indices never
  carry a gradient).
- `mx.compile` fuses differently when a function runs plain than when it is
  traced for a gradient, so a no-grad forward and the same forward under
  `value_and_grad` disagree by more than rounding. Compile is off while
  training and back on for rollouts, so the layered recompute, reference
  logprobs and GRPO's old-policy ratio see identical numbers.

**Chunked gated delta (qwen3_next / qwen3_5 hybrids).** mlx-lm's gated delta
kernel has no gradient, so train mode falls back to a Python loop with one
step per token, and autodiff through that was ~95% of training time on a
3:1 linear / full hybrid. Training swaps in the chunked form of the same
recurrence (the one flash-linear-attention trains with): matmuls inside
128-token chunks, a sequential state only between chunks. The in-chunk
triangular inverse uses block recursion (a power series in L overflows
before it cancels) and has a closed-form backward (`dL = W^T dW W^T`)
instead of autodiff through the block gathers. 128 differentiates ~15%
faster than 64 at the same accuracy; with the closed-form backward the
gated delta forward + backward is ~27% faster than before. On CPU it
matches the loop to 1e-7 (gradients 5e-6); on the GPU float32 matmuls run at
reduced precision, ~2e-3, under bf16 rounding.
`PAGED_MOE_TRAIN_CHUNKED_DELTA=0` keeps the loop.

**Fast backward for frozen quantized weights.** The gradient of a quantized
matmul with respect to its input runs MLX's quantized kernel with
`transpose=False`, about 4x slower than the forward. The weights are frozen,
so `train/fastgrad.py` wraps `QuantizedLinear`, tied quantized heads,
`QuantizedSwitchLinear` and the streamed expert matmuls in custom functions:
the forward is the same op, the backward dequantizes the weight and runs a
dense matmul (`gather_mm` with the expert sort kept). Gate and up share one
function so only one dense copy is alive at a time; the expert token sort
gets gather backwards instead of scatter-adds. Rows of 4096+ tokens also get
a chunked causal attention backward that only builds the lower triangle
(about half the stock peak at the same speed). Gradients match to bf16
rounding. On a 235B layer at 8k tokens, forward + backward went from 1002 ms
to ~590 ms (MoE 602 -> 326, attention 357 -> 225); the cost is ~1.5 GB
transient during an MoE backward, which the probe measures.
`PAGED_MOE_TRAIN_FASTGRAD=0` turns it off. Rollouts uninstall it.

**Micro-batch split.** Inside a step rows are sorted by length and cut where
the padded tokens plus a fixed per-micro-batch cost are lowest, under the
same memory limit. Each micro-batch restacks and dequantizes every layer's
experts; the probe measures that cost (12.6 s per step on the 235B, ~1.3k
tokens of compute) and `Schedule.MICRO_COST` (2048) is the fallback. That is 8-14% less work
per step than filling micro-batches greedily on typical length mixes. The
loss is normalized over the step, so the split never changes the math.

**Ground rules.** It never raises Metal's wired limit, and it expects to be
the only GPU job on the Mac. Checkpoints are the normal safetensors dirs, no
conversion. Runs, probe profiles and bench results live under
`~/.paged_moe/train` (`PAGED_MOE_TRAIN_DIR` moves it).

---

## Measured

PagedMoE-27B-A3B is Qwen3.6-35B-A3B (4-bit) with 192 of its 256 routed
experts per layer kept. PagedMoE-35B-A3B is the full model.

M-series, 48 GB, PagedMoE-27B-A3B-4bit, LoRA rank 8 on all linear layers,
2 x 256 tokens per step:

| Path | Before chunked delta | Now | Peak |
| --- | --- | --- | --- |
| resident fused (checkpoint) | 17.0 s | 3.6 s | 18.0 GB |
| resident layered | 16.8 s | 3.7 s | 17.2 GB |
| streamed layered, 4 GB cache | 24.2 s | 6.7 s | 7.7 GB |

All three give the same loss (0.785215) and gradients that agree to cosine
0.99995 (bf16 accumulation order). Read-ahead hit rate 95%, disk wait under
a second.

Short real runs (`sandbox.sh`, 48 GB M-series):

| Run | Speed | Peak |
| --- | --- | --- |
| 27B SFT resident | ~315 tok/s (estimate 370), 2 s/step | 26 GB |
| 27B SFT streamed, 4 GB cache | ~245 tok/s (estimate 205) | 11 GB |
| 27B ORPO resident | ~394 tok/s (estimate 372) | 26.5 GB |
| 27B GRPO resident | generation ~160 tok/s for 16 rollouts, 3.2 s/update | 21 GB |
| 35B SFT, simulated 16 GB Mac (auto streamed) | ~205 tok/s (estimate 143), 0.6x expert mass read per step | 10.3 GB |
| 35B ORPO, simulated 16 GB | ~228 tok/s | 10.5 GB |

Rows are short (40 to 60 tokens), so these are per-step overhead heavy;
longer rows raise tok/s, mostly in streamed mode where the reads are fixed
per step.

Qwen3-235B-A22B 4-bit (132 GB), streamed layered, 10 GB cache, LoRA on 94
layers, 12 rows per step (mean 646 tokens, max 1024), micro-batches <= 8192
tokens:

| | Before fast backward | Now |
| --- | --- | --- |
| Step | ~245 s | ~157 s |
| Forward / backward | ~55 s / ~185 s | ~41 s / ~113 s |
| Padded tokens per step | ~9.6k | ~8.8k |
| Read per step | ~217 GB | ~225 GB |
| Peak | 33.1 GB | 27-29 GB |

Same loss path, no stalls (worst UI gap 0.04 s). The estimate was 125 s;
the probe's synthetic rows run a little faster than real data.

---

## Architectures

The executors never look inside a decoder layer. A dry call with every layer
swapped for a tap records what the model passes each one (hidden state,
masks, per-layer inputs), and the head is reached by injecting a hidden
state at layer 0. So the requirements are:

- `model.layers` is the live list of decoder layers
- each layer takes the hidden state first and returns one array
- the MoE block uses mlx-lm's `SwitchGLU` (for streaming; resident works with anything)

Tested: `qwen3_moe` (Qwen3-235B-A22B), `qwen3_5_moe` (27B-A3B and 35B-A3B,
hybrid gated delta plus full attention), `qwen3_next` (Qwen3-Coder-Next),
`qwen4_exp` (Qwen3.8-Flash-Next, with the model file that ships in its
checkpoint). Most of mlx-lm's MoE models share that layout
(DeepSeek V3, GLM-4 MoE, Mixtral among them) and should work, but they have
not been run here yet. A model that breaks a requirement fails on the first
step (the probe, usually during construction) with a message saying which.

---

## Limits

- LoRA on routed experts is resident only. Streamed, the experts stay frozen
  (they page from disk) and everything else adapts.
- Full fine-tuning is not wired up; LoRA / DoRA only.
- GRPO's plan estimate is a worst case (every completion at
  `max_completion_tokens`) and leaves out generation, which is measured live
  (`gen_s`, `gen_tok_s`).
- No probe (and no estimate) when training an encoder.
- Rows in one micro-batch must share an `embeds` shape and `embeds_at`; pad
  them in the formatter.
- The PEFT export leaves out routed-expert LoRA and trained encoders.

Files: `pagedmoe/train/` (`trainer.py` loop and methods, `executor.py`,
`experts.py`, `fastgrad.py`, `linear_attn.py`, `objectives.py`, `data.py`, `plan.py`,
`lora.py`, `rollout.py`, `rewards.py`, `stats.py`, `env.py`, `cli.py`, `export.py`),
examples in `pagedmoe/train/examples/`.

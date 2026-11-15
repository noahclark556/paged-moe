# Prefix cache on disk

Every prompt on a streamed MoE costs a prefill, and a prefill is a pass over
the expert weights. A big system prompt with tool schemas can be 10k tokens or
more, and paying for that again every time the server restarts adds up fast
(about 40 s on Flash for 14k tokens).

The server can save the KV for the part of your prompt that never changes and
load it back after a restart, a model switch, or a crash. Then the first
request only prefills what's new.

It's on by default, but it only kicks in when your request tells the server
which part of the prompt is stable. A plain OpenAI client that only sends
`messages` never triggers it. It still gets the in-memory prompt cache while
the server is up, that just doesn't survive a restart.

## How it works

Add `stable_prefix_messages: N` to the request. That says the first N
messages are byte-identical on every request (usually the system prompt, which
is also where the template puts your tool schemas).

- The first request prefills as normal, pins the KV at the end of those N
  messages, and writes it to disk.
- After a restart, a request that starts with the same tokens loads that file
  instead of prefilling them.
- If the start changes, it's a miss and the request prefills like normal. The
  new prefix gets saved in its place.

A file only loads if everything that could change the result matches: the
weights, config, engine version, the settings that affect prefill, and the
exact token ids. It also has to pass a CRC check. Anything that doesn't match
is deleted on the next lookup and the log says why, so a cold prefill is never
silent.

The stable part has to be between 1,024 and 32,768 tokens. Shorter than that
and a prefill is cheap anyway, so nothing is saved.

## Quick start

Start the server:

```bash
paged-moe serve ~/mlx-models/<model> --port 8080
```

You need a front matter of at least 1,024 tokens, so this builds a request
from the repo's own README as the system prompt. Run it from the repo root:

```bash
ask() {
  python3 - "$1" > /tmp/body.json <<'EOF'
import json, sys
system = "You answer questions about the project below. Keep answers short.\n\n" + open("README.md").read()
print(json.dumps({
    "model": "default_model",
    "stable_prefix_messages": 1,
    "max_tokens": 100,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": sys.argv[1]},
    ],
}))
EOF
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @/tmp/body.json | python3 -c "import json, sys; print(json.load(sys.stdin)['usage'])"
}

ask "What does --speed auto do?"
```

The usage shows `"cached_tokens": 0`, it was a cold prefill. The server log
shows:

```
[prefix]   pinned stable front matter @ 9573 tok (0.26 GB)
[prefix-disk] saved front matter 9573 tok (0.26 GB, 4 checkpoint(s) at [2048, 4096, 6144, 8192]) in 0.24s
```

At this point the engine has the system prompt saved to disk. The checkpoints
only show up on hybrid models, they're extra resume points inside the file. You
may also see `no engine check for this file`, that's fine, the file still
loads.

Stop the server (Ctrl+C), start it again with the same command, and ask
something else:

```bash
ask "How do I turn off tool call repair?"
```

This time the log shows:

```
[prefix-disk] loaded front matter 9573 tok from disk (0.26 GB, 0.09s)
```

and the usage shows `"cached_tokens": 9573`. Only the new question was
prefilled. Token counts depend on the model's tokenizer, so yours will be a
little different.

Leave out `"stable_prefix_messages": 1` and run the same thing: nothing is
saved and every restart starts cold.

## More cut points

These are optional. They help when part of your front matter changes now and
then, so a change doesn't throw away the whole file.

| Field | What it saves |
| --- | --- |
| `stable_prefix_messages: N` | The first N messages. The main file |
| `stable_prefix_shared_messages: M` | The first M messages (M < N), the part that's the same everywhere, like the system prompt and tools before any per-project context. A new or changed project resumes from here |
| `stable_prefix_head_messages: H` | The first H messages (H < N), everything before a block you edit often, like a memory note. An edit to that block resumes from here |
| `stable_prefix_chat_id` + `stable_prefix_chat_messages: C` | A whole conversation up to message C, tagged with your id, so reopening an old chat doesn't prefill it again. Needs at least 2,048 tokens |

On models that can trim their cache (most MoEs), one file covers any shorter
match anyway, so the extra cut points matter most on hybrid models
(Qwen3.6, qwen38-coder-flash), where a cache can only resume at a point it was
saved at.

## Where the files go

`~/.paged_moe/prefix_cache` by default. The limits:

| Setting | Default | What it does |
| --- | --- | --- |
| `PAGED_MOE_PREFIX_DISK` | `1` | `0` turns the disk cache off |
| `PAGED_MOE_PREFIX_DISK_DIR` | `~/.paged_moe/prefix_cache` | where the files go |
| `PAGED_MOE_PREFIX_DISK_MAX_GB` | `12` | total size on disk |
| `PAGED_MOE_PREFIX_DISK_MODELS` | `6` | models kept, most recent first |
| `PAGED_MOE_PREFIX_DISK_PER_MODEL` | `2` | files per model for each kind |
| `PAGED_MOE_PREFIX_DISK_CHATS` | `6` | saved conversations per model |

A file is around 0.2 to 2.6 GB depending on the model and the prompt length.
Deleting the folder is always safe, the next request just prefills cold.

## Tips

- Keep the stable messages byte-identical. A date, a counter, or a reworded
  line in there means a miss every time. Put anything that changes after the
  stable messages.
- The tool list is part of the front matter for most templates. Changing it
  changes the prefix.
- Same goes for template kwargs like `enable_thinking` or `reasoning_effort`.
  Some templates write those into the system block.
- `PAGED_MOE_PREFIX_DEBUG=1` logs how much each request reused and where it
  diverged, which is the quickest way to find what's changing.

The rest of the host side (how to compact history, `warm_prefix`, what not to
rewrite) is in [CLIENT_CONTEXT.md](./CLIENT_CONTEXT.md).

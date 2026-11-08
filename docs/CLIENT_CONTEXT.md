# Client context management for PagedMoE

PagedMoE owns **inference**: expert paging, the prompt-cache LRU,
`warm_prefix`, fused prefill, KV policy. It does **not** own your chat
history. Every `/v1/chat/completions` request is scored against the token
sequence you send. If that sequence rewrites earlier bytes, the prefix cache
misses and you pay another full pass over the expert mass.

This is the contract for any host that talks to the server: an agent, an IDE
plugin, a custom OpenAI client. Follow it and agent turns stay fast on a
streamed MoE. Ignore it and every turn is a cold prefill.

---

## The physics in one line

Prefill cost ~ **new tokens x expert union of those tokens**.

Diverse tool dumps (file reads, greps) route across essentially the whole
expert mass for each chunk. Prefix reuse only skips tokens that match
**byte for byte from token 0**. Timing tricks (prefetch, the sidecar) don't
shrink that mass, they only hide the wait.

So the host has to:

1. Keep the stable prefix identical across rounds.
2. Minimize how many new tokens enter each completion.
3. When history has to shrink, rewrite once, then **warm** the new short
   prefix before the next real ask.

---

## What the engine can't fix for you

These are host bugs:

- Rewriting system / memory / map / task-state / digests mid-prefix every turn
- Moving a block from the middle of the prompt to the tail (or the reverse)
- Re-sending full tool bodies after you already replaced them with stubs on a
  previous request (unless you re-apply the same stub bytes)
- Expecting the server to "remember" a compacted history while you keep
  sending the long one
- Character-mangling or ad-hoc text compression as KV savings (wrong tokens,
  wrong attention, and it usually busts the prefix too)

The engine can cache and warm. It can't invent a shorter history than the one
in the POST body.

---

## Cache breakpoints (prefix only)

Causal attention: the KV at token i is a function of tokens `0..i`. A constant
span in the *middle* of the prompt can't be cached and spliced after a
different prefix. Tags like `<cache>The current date is</cache>` as isolated
islands aren't valid and the engine won't treat them that way.

What works is **prefix breakpoints**. Snapshot the KV at token N. The next
request reuses it only if its first N tokens match byte for byte.

The engine:

1. Auto-snapshots the first message boundary (usually system + tool schemas).
2. Pins a copy of that head so the LRU can't drop it. The pin is sized against
   the prompt (up to half of it, 32k ceiling) because a system prompt plus a
   couple dozen tool schemas is well past 8k on its own, and a head that can't
   be pinned means every front-matter edit re-prefills from token 0.
3. Honors host marks:

```json
{
  "messages": [
    { "role": "system", "content": "...", "cache_control": { "type": "ephemeral" } },
    { "role": "user", "content": "..." }
  ],
  "cache_breaks": [1840]
}
```

`cache_control` on a message, `cache_breaks: [token_index]`, or a
`<|cache_break|>` sentinel at the end of a message are the same idea: "the
tokens up to here are the reusable head." They don't cache a later island.

Trimmable vs not matters here. Most MoEs (DeepSeek-V3.2, Qwen3-MoE, GLM) can
trim a KV cache, so the engine prefills once and forks any boundary afterwards.
Hybrid attention models (Qwen3-Next, Qwen3.8-Flash-Next) can't, and a cache
can only be pinned at an offset it already sits at. So on those the pin is the
prefix the store just served, and a partial pin match is refused rather than
trimmed. Reusing KV for more tokens than the prefix actually shares is wrong
output, not a slow turn, so the engine takes the `kv=0` instead.

What that means for hosts: on hybrid models the prefix has to match a stored
snapshot or a recurrent-state checkpoint (each turn's resume point and its
mid-snap). Branching back to one of those boundaries is cheap. Anywhere else
there's no partial credit and you pay for the whole prompt.

Keep side requests (retries, recoveries, warm prefixes) on the same template
kwargs as the turn they follow. Some templates write the think switch or the
reasoning effort into the system block, and a side request that flips either
one shares nothing with the chat past the first few tokens.

If your host rewrites history (compaction, summaries), send
`stable_prefix_messages: N` with the count of leading messages you keep
byte-identical. The engine pins the KV at the end of those N messages, stopping
prefill there once on hybrid caches, so a rewrite resumes from that pin
instead of token 0. Keep those N messages truly stable. If they change, the pin
misses once and gets rebuilt.

---

## Do

### 1. Treat the prompt as append-only during a live tool loop

Once a message has been prefilled into the prompt cache, don't edit its bytes.
Append tool results and new assistant turns. Put volatile reminders at the
**tail**, not by rewriting the head.

### 2. Full tool body on first sight, park it after the model continues

The model can't reason about a file it hasn't attended to. The first
completion that includes a large tool result **has** to pay the expert-mass
prefill once.

After an assistant message follows that tool result:

- Replace the raw tool body with a stable stub (path, size, short preview,
  archive id).
- Keep the tool **call** and the assistant's own prose (what it decided).
- Offer a re-read tool for the rare case it needs the bytes again.

Don't stub before first sight. That causes guess-loops and re-reads. And don't
`warm_prefix` in the middle of a turn after parking: the rewritten history no
longer matches the cached prefix, so the warm would be a full cold prefill.

### 3. Collapse at task done (or cancel), then warm

When a task finishes cleanly, **or** the user cancels mid-run after real tool
work:

1. Replace the long tool transcript with a short durable summary (original
   ask, files touched, before/after intent, verify status, or on cancel what
   landed so far and what's still open).
2. Archive the raw turns somewhere you can reload them.
3. POST that exact short message list with `warm_prefix: true`, `max_tokens: 0`.
4. Next user ask = the **same** short prefix bytes + the new user message only.

`kv=0` on the warm of a brand-new collapsed prefix is expected. Success looks
like `[prefix] warm_prefix=1` ... prefill ... `[prefix] warm complete`. The win
is the **next** ask, not the warm itself.

Skipping the cancel path leaves the fat transcript in place, so the follow-up
re-prefills everything.

### 4. Match warm bytes to the next real request

Whatever you warm has to be identical to the leading messages of the next
completion (same system prompt shape, same stubs, same bridge text). Extra
reminders that show up only on the real ask and not on the warm truncate the
hit to the divergence point.

If the host injects front matter (project memory, a codebase map) ahead of the
conversation, the warm has to include it or the hit ends at the first injected
block. Render it for the warm POST and then throw it away. Don't write it into
stored history (see Don't #2).

### 5. Ask-shaped text goes at the tail, never in front

A block that restates the current request is the worst thing to put in front
of the conversation: it changes on every ask and moves every token behind it.
Keep front matter ask-independent for the whole session and put per-ask
routing, file leads and plan state right before the new user message.

### 6. Await in-flight warms before the next model call

If a done-warm runs in the background, the next `/chat/completions` has to
wait for it.

### 7. Keep one Metal owner

Don't run a second MLX / PagedMoE load on the same GPU while the agent server
is hot. Tear one down before starting the other.

### 8. Slim follow-up user turns

The first ask can carry the workspace tree, the date, reminders. Every later
ask should be the new request only. Re-sending a 4k-token tree dump after a
17k cached prefix is a fresh expert-mass pass (that's the `7170 tok at
kv=17619` shape in the logs).

### 9. Keep tool schemas and phase text still

If the chat template serializes `tools=` at the front, changing the tool list
per phase rewrites token 0 and you get `kv=0`. Put phase limits in a **tail**
note, or pin the first phase block and leave it.

---

## Don't

### 1. Don't auto-reinflate done archives into every follow-up

A done summary already carries the undo / continue facts. Blindly reloading
the archive on every follow-up dumps the whole tool transcript back into the
prompt and **defeats** the warm for that turn.

Instead:

- Act from the summary when it's enough ("change it back" -> path +
  before/after in the summary).
- Let the **model** ask for the archive only when a path or a before/after
  fact is missing, and return a compact extract (paths, tool names, final
  prose), not the raw transcript.
- Never prefetch done archives automatically.

Mid-task compaction is different. There, auto-prefetch can stop the model from
re-exploring after bodies disappeared.

### 2. Don't persist injected blocks into stored history

Tempting shortcut when a collapse has to match the next turn: write the
injected front matter into the saved message list. It backfires. If your
injectors strip and re-render their own blocks every turn, a pin layer sees a
rewrite instead of an insertion and can't freeze it anymore. History holds
conversation only. Front matter is rebuilt per call and pinned per call.

### 3. Don't rewrite pins every turn

Refreshing task state, project memory or a codebase map with new wording each
round forces a full re-prefill from the edit point. Pin the first rendering
and append deltas, or update off the hot path.

### 4. Don't digest mid-assistant-message

Collapse tool bodies only after the assistant turn that used them is
committed. History that silently drops evidence the assistant just cited
causes confusion and invented detail.

### 5. Don't use empty or lying stubs

A stub has to admit what was removed and how to get it back. Deleting a
5k-token read while the assistant said "based on the full file..." with no
archive id is how agents thrash.

### 6. Don't "compress" by deleting letters inside tokens

KV entries are per tokenizer id, not per word. Mangling `the` -> `th` changes
the sequence, usually destroys the prefix match, and doesn't make the model
"infer" the missing characters from the cache.

---

## Suggested host pipeline (agent turn)

```
user ask
  -> stable system + pinned front matter (byte-stable)
  -> model

tool round
  -> append tool results (full body on first sight)
  -> model completion (pays expert-mass prefill for new tokens once)
  -> if assistant continued past large tools:
        park bodies -> stubs (mutate host history)
        // no warm_prefix here, a mid-turn rewrite means a cold full prefill
  -> execute next tools
  -> next completion (delta from the digest point + new tool tails)

done
  -> summarize + archive raw turns
  -> replace live history with summary + bridge
  -> warm_prefix(short prefix)
  -> await warm before the next user ask

follow-up ask ("change it back")
  -> short prefix + slim user request only (no second workspace tree)
  -> don't auto-dump the archive
  -> the model can ask for it if a path is missing, return a compact extract
```

---

## How to read the logs

| Log | Meaning |
| --- | --- |
| `chunk N tok at kv=K` | Prefilling N new tokens, K already matched in the prompt cache |
| `kv=0` on a warm of a new summary | Expected, that prefix was never cached |
| `kv=0` on a normal tool round after a long session | Prefix bust, something rewrote earlier messages |
| `warm_prefix=1` ... `warm complete` | Host warm worked |
| `stable-pin hit N tok` | LRU missed (or was short), the pinned system/tools head still matched |
| `[prefix] hybrid checkpoint hit` | A hybrid cache branched back to a held recurrent state |
| `[prefix-disk] loaded ...` | Front matter came off disk instead of a prefill |
| A giant archive payload right after done | Host re-inflated the transcript, the warm win is gone for that request |

`PAGED_MOE_PREFIX_DEBUG=1` logs how much each request reused and, when reuse
is partial, the decoded text on both sides of the divergence. That's the
fastest way to find which block your host is rewriting.

---

## Checklist for a new host

1. Append-only history during tool loops, a done summary, and `warm_prefix`.
2. First sight, then stub, for large tool results (or hard-cap tool payloads
   and make the model re-fetch).
3. Don't assume the server compacts messages for you.
4. Send `stable_prefix_messages` if you rewrite history.
5. Keep archive reload **opt-in** after done. Default follow-ups stay on the
   short summary.

The engine stays fast as long as your messages stay honest about what
changed.

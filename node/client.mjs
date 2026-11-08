// Copyright (C) 2026 Noah Clark
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * paged-moe Node client.
 *
 * The engine itself is Python/MLX (that's where Apple-silicon GPU compute
 * lives), but this module makes Node the front door: it can spawn/supervise
 * the server and gives you a typed-ish client for it. No dependencies  - 
 * built on Node 18+ global fetch.
 *
 * Two ways to use it:
 *
 *   // 1. Talk to a server that's already running
 *   import { ExpertStream } from "./client.mjs";
 *   const es = new ExpertStream({ baseUrl: "http://127.0.0.1:8080" });
 *   const reply = await es.chat([{ role: "user", content: "hi" }]);
 *
 *   // 2. Let Node own the whole lifecycle (spawn server, wait until ready)
 *   const es = await ExpertStream.launch({ model: "path-or-hf-repo" });
 *   for await (const delta of es.chatStream([{ role: "user", content: "hi" }])) {
 *     process.stdout.write(delta);
 *   }
 *   await es.close();
 *
 * The wire format is OpenAI-compatible, so you can also point any OpenAI
 * client (openai npm package, LangChain, etc.) at
 * `${baseUrl}/v1` and skip this file entirely.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = join(fileURLToPath(new URL(".", import.meta.url)), "..");

function defaultPython() {
  if (process.env.PAGED_MOE_PYTHON) return process.env.PAGED_MOE_PYTHON;
  const venv = join(homedir(), ".mlx-env", "bin", "python");
  return existsSync(venv) ? venv : "python3";
}

export class ExpertStream {
  constructor({ baseUrl = "http://127.0.0.1:8080", model = "default_model" } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.model = model;
    this.proc = null; // set by launch()
  }

  /**
   * Spawn the Python server and resolve once it's serving.
   *
   * @param {object} opts
   * @param {string} opts.model        local model dir or HF repo id (required)
   * @param {number} [opts.port=8080]
   * @param {string} [opts.host="127.0.0.1"]
   * @param {string} [opts.python]     python executable (default: ~/.mlx-env)
   * @param {string} [opts.mode]       "auto" | "resident" | "streamed"
   * @param {number} [opts.cacheGb]    expert cache budget in GB
   * @param {string[]} [opts.extraArgs] extra flags passed to the server
   * @param {function} [opts.onLog]    receives server stderr lines
   * @param {number} [opts.readyTimeoutMs=600000]  big models load slowly
   */
  static async launch({
    model,
    port = 8080,
    host = "127.0.0.1",
    python = defaultPython(),
    mode,
    cacheGb,
    extraArgs = [],
    onLog = () => {},
    readyTimeoutMs = 600_000,
  }) {
    if (!model) throw new Error("launch requires a model");

    const env = { ...process.env };
    if (mode) env.PAGED_MOE_MODE = mode;
    if (cacheGb != null) env.PAGED_MOE_CACHE_GB = String(cacheGb);

    const args = [
      "-m", "pagedmoe.server",
      "--model", model,
      "--host", host,
      "--port", String(port),
      ...extraArgs,
    ];
    const proc = spawn(python, args, { cwd: REPO_ROOT, env });
    proc.stdout.on("data", (d) => onLog(d.toString()));
    proc.stderr.on("data", (d) => onLog(d.toString()));

    const client = new ExpertStream({ baseUrl: `http://${host}:${port}`, model });
    client.proc = proc;

    const deadline = Date.now() + readyTimeoutMs;
    // The first request triggers the model load, so /health returning is
    // enough to declare the server ready to accept work.
    while (Date.now() < deadline) {
      if (proc.exitCode !== null) {
        throw new Error(`server exited with code ${proc.exitCode}`);
      }
      try {
        const ok = await client.health();
        if (ok) return client;
      } catch {
        /* not up yet */
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    proc.kill();
    throw new Error(`server not ready after ${readyTimeoutMs} ms`);
  }

  async close() {
    if (this.proc && this.proc.exitCode === null) {
      this.proc.kill("SIGTERM");
      await new Promise((r) => this.proc.once("exit", r));
    }
  }

  async health() {
    const res = await fetch(`${this.baseUrl}/health`);
    return res.ok;
  }

  async models() {
    const res = await fetch(`${this.baseUrl}/v1/models`);
    return (await res.json()).data;
  }

  /**
   * Non-streaming chat completion. Returns the assistant message content.
   * opts: { maxTokens, temperature, topP, stop, model, raw }
   */
  async chat(messages, opts = {}) {
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.#body(messages, opts, false)),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    const json = await res.json();
    return opts.raw ? json : json.choices[0].message.content;
  }

  /**
   * Streaming chat completion. Async generator yielding text deltas.
   */
  async *chatStream(messages, opts = {}) {
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.#body(messages, opts, true)),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);

    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of res.body) {
      buf += decoder.decode(chunk, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        const delta = JSON.parse(payload).choices?.[0]?.delta?.content;
        if (delta) yield delta;
      }
    }
  }

  #body(messages, opts, stream) {
    return {
      model: opts.model ?? this.model,
      messages,
      stream,
      max_tokens: opts.maxTokens ?? 1024,
      ...(opts.temperature != null && { temperature: opts.temperature }),
      ...(opts.topP != null && { top_p: opts.topP }),
      ...(opts.stop != null && { stop: opts.stop }),
    };
  }
}

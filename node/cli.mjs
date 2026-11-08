#!/usr/bin/env node
// Copyright (C) 2026 Noah Clark
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * paged-moe Node CLI.
 *
 *   # one-shot (spawns the server, streams the answer, shuts down)
 *   node cli.mjs --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
 *       "Explain expert streaming in one paragraph"
 *
 *   # interactive chat
 *   node cli.mjs --model <path-or-repo> --chat
 *
 *   # against an already-running server (fast startup, shared model)
 *   node cli.mjs --url http://127.0.0.1:8080 "hello"
 *
 * Flags: --model, --url, --port, --chat, --max-tokens, --temp,
 *        --mode auto|resident|streamed, --cache-gb N, --verbose
 */

import { createInterface } from "node:readline/promises";
import { ExpertStream } from "./client.mjs";

function parseArgs(argv) {
  const args = { positional: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) { args.positional.push(a); continue; }
    const key = a.slice(2);
    const flags = new Set(["chat", "verbose", "help"]);
    if (flags.has(key)) args[key] = true;
    else args[key] = argv[++i];
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
if (args.help || (!args.model && !args.url)) {
  console.log("usage: node cli.mjs (--model <path-or-repo> | --url <server>) [--chat] [prompt]");
  process.exit(args.help ? 0 : 1);
}

const genOpts = {
  maxTokens: args["max-tokens"] ? Number(args["max-tokens"]) : 1024,
  ...(args.temp != null && { temperature: Number(args.temp) }),
};

let es;
if (args.url) {
  es = new ExpertStream({ baseUrl: args.url });
} else {
  const log = args.verbose ? (s) => process.stderr.write(s) : () => {};
  process.stderr.write("starting server (model load can take a while)...\n");
  es = await ExpertStream.launch({
    model: args.model,
    port: args.port ? Number(args.port) : 8080,
    mode: args.mode,
    cacheGb: args["cache-gb"] ? Number(args["cache-gb"]) : undefined,
    onLog: log,
  });
  process.stderr.write("server ready\n");
}

async function streamReply(messages) {
  let text = "";
  for await (const delta of es.chatStream(messages, genOpts)) {
    process.stdout.write(delta);
    text += delta;
  }
  process.stdout.write("\n");
  return text;
}

try {
  if (args.chat) {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    const messages = [];
    console.log("chat mode - empty line to exit");
    for (;;) {
      const user = (await rl.question("\nyou> ")).trim();
      if (!user) break;
      messages.push({ role: "user", content: user });
      process.stdout.write("model> ");
      const reply = await streamReply(messages);
      messages.push({ role: "assistant", content: reply });
    }
    rl.close();
  } else {
    const prompt = args.positional.join(" ") || "Hello!";
    await streamReply([{ role: "user", content: prompt }]);
  }
} finally {
  await es.close();
}

# Gemma 4 E2B chat demo

An interactive, line-oriented chat REPL for the Gemma 4 E2B pipeline. It is the worked example for
`Gemma4ChatSession`: one conversation, one KV sequence, string in and string out — no token ids, no
`<bos>`, no manual prompt assembly.

```
deno task demo:gemma4
```

The script needs a WebGPU adapter and a local distribution directory (see `--source` below).

## Talking to it

One line is one user turn. While a turn is generating, the reply streams out as it is decoded.

| Input            | Effect                                                                      |
| ---------------- | --------------------------------------------------------------------------- |
| `/reset`         | Throw the conversation away and start a fresh session with the same setup.  |
| `/exit`, `/quit` | Leave.                                                                      |
| `Ctrl+D`         | Same as `/exit` (end of stdin).                                             |
| `Ctrl+C`         | Cancel the turn that is generating; with no turn running, exit the process. |

## Where the model comes from

`fromPretrained` has no built-in default, so the source is always spelled by the caller. The two
forms are mutually exclusive:

- `--source <dir>` reads a local distribution directory (one that carries `karume.json`) directly.
  This is the default, at `models/karume-gemma4-e2b` — the layout that
  `tools/export-recipes/gemma4` produces.
- `--repo <owner/name[@revision]>` fetches from Hugging Face. Gemma 4 has no published repository
  yet, so there is no pinned revision constant for it; write the revision yourself when you want a
  reproducible fetch.

## Options

All options are `--key value` pairs except `--diagnostics`, which is a bare switch. Unknown keys are
rejected rather than silently ignored.

| Option                           | Default                    | What it does                                                                |
| -------------------------------- | -------------------------- | --------------------------------------------------------------------------- |
| `--source <dir>`                 | `models/karume-gemma4-e2b` | Read a local distribution directory.                                        |
| `--repo <owner/name[@revision]>` | —                          | Fetch from Hugging Face instead.                                            |
| `--system <text>`                | none                       | System turn placed at the head of the conversation.                         |
| `--max-new-tokens <n>`           | `256`                      | Per-turn generation cap (stop tokens are not counted).                      |
| `--temperature <x>`              | asset default              | Sampling temperature; `0` is greedy.                                        |
| `--top-k <n>` / `--top-p <x>`    | asset default              | Candidate truncation.                                                       |
| `--seed <n>`                     | asset default              | Sampling seed.                                                              |
| `--capacity <n>`                 | asset default              | Logical positions this conversation reserves KV for.                        |
| `--chunk-length <n>`             | asset default              | Rows per prefill run.                                                       |
| `--max-resident-ple-bytes <n>`   | two largest shards         | Host RAM budget for the resident PLE sidecar.                               |
| `--diagnostics`                  | off                        | Print the per-op GPU time breakdown of the last run of each turn to stderr. |

Any sampling flag you pass is layered on top of the recommended values the asset declares; the ones
you leave out keep their declared value.

### Capacity and chunk length are runtime knobs

The distribution declares `capacity` and `chunkLength`, but both are _defaults_, not baked-in
properties: RoPE cos/sin tables are derived per chunk on the host rather than stored in the graph, so
you can pick different values without re-exporting the model.

- `--capacity` trades GPU memory for conversation length. It must satisfy
  `chunkLength <= capacity <= maxPosition`, and it is also the yardstick the session uses to decide
  when to drop old turns.
- `--chunk-length` trades prefill run count (and therefore fence waits) against the temporary
  buffers and attention score matrix of a single run. The generated token sequence does not change.

Both are validated at startup, by the same memory estimate that prints the `GPU 見積り` line — you
find out before the first turn, not on the first `send`.

### `--diagnostics` is not free

Per-op GPU timing needs the `timestamp-query` feature, which can only be requested when the device
is created, so the script acquires its own device in this mode. A device with timing enabled opens
one pass per dispatch, which stretches wall-clock time. Do not read the `tok/s` figure of a
`--diagnostics` run as a speed measurement; take speed numbers without the flag.

## What the startup lines mean

The banner is shaped like this — the byte figures depend on the asset and the device:

```
[gemma4] ready（12.3s） / capacity 4096 / maxPosition 131072 / chunk 768
         GPU 見積り resident 3812 MiB / peakAccounted 4205 MiB（上限ではない — 勘定外 5 項目）
         sampler {"temperature":1,"topK":64,"topP":0.95} / max-new-tokens 256
```

`resident` is what the session holds for the lifetime of the model (weights plus KV state slots);
`peakAccounted` adds the largest single run scenario on top. It is an estimate of the accounted
categories, not an upper bound — the final gate on admission is still the out-of-memory error scope.

## Per-turn output

A long prompt spends its first seconds in prefill, where nothing has been decoded yet. When the
prompt spans more than one chunk, the script overwrites a `prefill n/m` line on stderr until the
first piece of the reply arrives.

Each turn closes with a bracketed summary: stop reason, tokens generated, elapsed seconds, tok/s, and
the number of turns in the conversation. When a turn does not fit, the session drops the oldest
user/assistant pair and says so; when there is nothing left to drop, it reports the numbers that
decide the case (limit, past length, prompt length, and the largest `max-new-tokens` that would have
fit).

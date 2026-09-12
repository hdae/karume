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
  This is the default, at `models/karume-gemma4` — the layout that
  `tools/export-recipes/gemma4` produces.
- `--repo <owner/name[@revision]>` fetches from Hugging Face. Gemma 4 has no published repository
  yet, so there is no pinned revision constant for it; write the revision yourself when you want a
  reproducible fetch.

## Options

All options are `--key value` pairs except `--speculative` and `--diagnostics`, which are bare
switches. Unknown keys are rejected rather than silently ignored.

| Option                           | Default                 | What it does                                                                                                       |
| -------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--source <dir>`                 | `models/karume-gemma4`  | Read a local distribution directory.                                                                               |
| `--repo <owner/name[@revision]>` | —                       | Fetch from Hugging Face instead.                                                                                   |
| `--system <text>`                | none                    | System turn placed at the head of the conversation.                                                                |
| `--max-new-tokens <n>`           | `256`                   | Per-turn generation cap (stop tokens are not counted).                                                             |
| `--temperature <x>`              | asset default           | Sampling temperature; `0` is greedy.                                                                               |
| `--top-k <n>` / `--top-p <x>`    | asset default           | Candidate truncation.                                                                                              |
| `--seed <n>`                     | asset default           | Sampling seed.                                                                                                     |
| `--capacity <n>`                 | asset default           | Logical positions this conversation reserves KV for.                                                               |
| `--chunk-length <n>`             | asset default           | Rows per prefill run.                                                                                              |
| `--max-resident-ple-bytes <n>`   | twice the largest shard | Host RAM budget for the resident PLE sidecar.                                                                      |
| `--speculative`                  | off                     | Build the MTP drafter and speculate while decoding. Speed only — the token sequence does not change.               |
| `--diagnostics`                  | off                     | Print the per-op GPU time breakdown of the last run of each turn to stderr. Not usable on macOS/Metal — see below. |

Any sampling flag you pass is layered on top of the recommended values the asset declares; the ones
you leave out keep their declared value.

With `--temperature 0`, decoding reads back only the selected value and token id when logits need
no repetition penalty or logit bias, and speculation and diagnostics are disabled. Prefill keeps
the regular path. This uses the existing model assets and also applies to the mobile QAT family.
Sampling at other temperatures keeps the regular path.

A flag value may not start with `--`. The parser treats such a value as a swallowed flag and stops,
so that a mistyped knob never runs silently on its default. The cost is that a system prompt whose
text begins with `--` (say `--- rules ---`) is rejected; reword it or drop the leading dashes.

### Capacity and chunk length are runtime knobs

The distribution declares `capacity` and `chunkLength`, but both are _defaults_, not baked-in
properties: RoPE cos/sin tables are derived per chunk on the host rather than stored in the graph, so
you can pick different values without re-exporting the model.

- `--capacity` trades GPU memory for conversation length. It must satisfy
  `chunkLength <= capacity <= maxPosition`, and it is also the yardstick the session uses to decide
  when to drop old turns.
- `--chunk-length` trades prefill run count (and therefore fence waits) against the temporary
  buffers and attention score matrix of a single run. The generated token sequence does not change.
  It is capped by the `maxChunkLength` the distribution declares (768 for E2B) — the traced upper
  bound of the chunk symbol, which the graph itself does not carry.

Both are validated at startup — chunk length when the pipeline is built, capacity by the memory
estimate that prints the `GPU 見積り` line. You find out before the first turn, not on the first
`send`.

### `--speculative` buys speed and nothing else

With the flag the pipeline also fetches the distribution's `drafter` weights and opens one drafter
session that borrows the target session's embedding table (no bytes are duplicated), and the
conversation speculates by default. Each cycle drafts `k` tokens on the drafter and confirms them in a
single verify run of `k + 1` rows, so one target run can commit up to `k + 1` tokens instead of
exactly one. `k` is the number of steps the distribution's drafter graph was exported with, and the
script does not expose a knob for it.

Acceptance draws from the same logits, with the same history, in the same order as a plain decode
would at that position, so the flag moves speed only: the same seed produces the same tokens with and
without it. Compare wall clock and tok/s, never output.

A distribution that carries no drafter rejects the flag while `fromPretrained` resolves what to fetch.
The script does not inspect the distribution first — the gate lives in the library, and a second copy
of it here would be a second thing to keep true.

### `--diagnostics` is not free

Per-op GPU timing needs the `timestamp-query` feature, which can only be requested when the device
is created, so the script acquires its own device in this mode. A device with timing enabled opens
one pass per dispatch, which stretches wall-clock time. Do not read the `tok/s` figure of a
`--diagnostics` run as a speed measurement; take speed numbers without the flag.

On macOS/Metal the flag currently loses the whole device: the counter sample buffers fail to
allocate — whether the trigger is the number of live query sets or their retention is not yet
isolated — and wgpu turns that failure into a lost device rather than an error you can catch. The run
dies on the first turn. Take the breakdown on a Vulkan or D3D12 machine instead; the details are in
[`docs/limitations.md`](../../docs/limitations.md) and the state of the workaround (and the ruling
on when it gets investigated) in [`docs/known-issues.md`](../../docs/known-issues.md). The script warns on stderr at startup when you
pass the flag on macOS, and any fatal error is unwrapped in full — `SuppressedError` pairs,
`AggregateError` members, and `cause` chains — to stderr before the script exits with status 1.

## What the startup lines mean

The banner is shaped like this — the byte figures depend on the asset and the device:

```
[gemma4] loaded（12.3s） / capacity 4096 / maxPosition 131072 / chunk 768
         GPU 見積り resident 3812 MiB / peakAccounted 4205 MiB（上限ではない — 勘定外 5 項目）
         sampler {"temperature":1,"topK":64,"topP":0.95} / max-new-tokens 256 / 投機なし
```

The last field says whether this run carries the drafter: `投機なし` without `--speculative`,
`投機あり（k は配布形の段数）` with it — the banner does not spell the number, because the effective
`k` is the step count baked into the drafter graph and the public surface carries no constant for it.
The per-turn line below does spell it, read out of the acceptance tally.

`resident` is what the session holds for the lifetime of the model (weights plus KV state slots);
`peakAccounted` adds the largest single run scenario on top. It is an estimate of the accounted
categories, not an upper bound — the final gate on admission is still the out-of-memory error scope.

## Per-turn output

A long prompt spends its first seconds in prefill, where nothing has been decoded yet. When the
prompt spans more than one chunk, the script overwrites a `prefill n/m` line on stderr until the
first piece of the reply arrives.

Each turn closes with a bracketed summary: stop reason, tokens generated, TTFT, decode tok/s, total seconds,
then — on a speculating turn — `投機 k=3 · 2.15 tok/cycle（cycles 40）`, and finally the number of
turns in the conversation. `k` is the drafter's step count, taken from the length of the acceptance
histogram (the tally is built with one bucket per acceptance count, `0..k`); the tokens committed per
verify run are `1.00` when nothing was accepted and at most `k + 1`. When a turn does not fit, the
session drops the oldest user/assistant pair and says so; when there is nothing left to drop, it
reports the numbers that decide the case (limit, past length, prompt length, and the largest
`max-new-tokens` that would have fit).

## Timing and warmup

Startup runs a short warmup with separate generation state, then discards that state before
accepting conversation input. Warmup does not add messages to the conversation. It prepares the
main prefill/decode paths, not every possible prompt length or speculative execution path.
Use `--no-warmup` to measure a run without this startup step; driver caches may still be warm.

Each reply reports **TTFT** (time to the first non-stop token), **decode tok/s** (subsequent
non-stop tokens divided by the time from the first token to completion), and **total** turn time.
These are wall-clock measurements including host work and output handling, not GPU-only timings.
The displayed total token count includes EOS; the decode numerator excludes EOS and the first
token. With no delivered tokens TTFT is unavailable; with fewer than two, decode speed is unavailable.
Token timing precedes text decoding, so buffered text does not postpone TTFT.
Warmup and model loading are reported separately and excluded from turn timing.

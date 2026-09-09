# mtp-bench — what speculative decoding is actually worth

A CLI that measures the **end-to-end payoff of speculative decoding** (MTP, ADR 0096) for Gemma 4
E2B: it runs the non-speculative path (`plain`) and the speculative path (`speculative`) alternately
**in one process, on one pipeline, with one prompt**, and writes one JSON line.

Measuring the two paths in separate processes would compare two runs that differ in driver clock
state, PLE residency and shader compilation as well as in the thing under test, so the difference
could not be attributed. One process, alternating turns, is the whole point of this tool.

Kernels and the generation loop are **not modified**. Per-run wall clock is taken by wrapping
`Session.prototype.run` from the tool, and what each run was (`prefill` / `decode` / `draft` /
`verify`) is named by the pipeline's own observation seat (`onRunDiagnostics`), which fires in the
synchronous region right after the run returns.

## Usage

```
deno run -A tools/mtp-bench/main.ts --workload <name> --sampler <greedy|recommended> [options]
```

| Option                           | Default                                                | Meaning                                                                                                                                                          |
| -------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--source <dir>`                 | `models/karume-gemma4`                                 | Distribution mirror that carries the drafter weights, read through `denoDirectory`                                                                               |
| `--workload <name>`              | required                                               | `extract` / `summarize` / `dialogue` / `freeform` (see below)                                                                                                    |
| `--sampler <name>`               | required                                               | `greedy` = `{ temperature: 0 }`; `recommended` = the mirror's declared sampler plus `--seed`                                                                     |
| `--seed <int>`                   | `42`                                                   | Only used by `--sampler recommended`. Passing it together with `--sampler greedy` is rejected: a knob that cannot take effect is never accepted silently         |
| `--k <int>`                      | library default: the step count baked into the drafter | Drafts per cycle. Omitting it passes `speculative: {}`, so the drafter graph's own step count decides. The range gate lives in the library                       |
| `--new-tokens <int>`             | `200`                                                  | `maxNewTokens` per turn                                                                                                                                          |
| `--capacity <int>`               | `8192`                                                 | KV capacity of every turn's sequence. Above the mirror's default 4096 because the long-context prompt is ≈4.8K tokens                                            |
| `--document-chars <int>`         | `20000`                                                | Character budget for the document workloads, cut at a paragraph boundary. Rejected for the other workloads                                                       |
| `--rounds <int>`                 | `3`                                                    | ABBA repetitions after the warm-up                                                                                                                               |
| `--max-resident-ple-bytes <int>` | whole PLE sidecar                                      | Host RAM budget for the PLE sidecar. The default is derived from the mirror's index so that **no shard is ever re-read** during a turn                           |
| `--out <file.jsonl>`             | —                                                      | Append the JSON line to this file as well (stdout always gets it). The parent directory must exist — it is checked before the model is loaded, and never created |
| `--gpu-timing` (switch)          | off                                                    | Collect per-op GPU time via `acquireGpu({ gpuTiming: true })`                                                                                                    |

Unknown options fail loudly: a mistyped knob that silently fell back to a default would make the
recorded configuration disagree with what was actually measured.

Progress goes to stderr, one line per turn; **stdout carries only the final JSON line**.

```sh
# The headline number: greedy, long-context extraction
deno run -A tools/mtp-bench/main.ts --workload extract --sampler greedy \
  --out outputs/bench/karume-gemma4/2026-09-09_mtp-stage4/turns.jsonl

# Sampling as shipped (temperature > 0 must still produce the same token ids)
deno run -A tools/mtp-bench/main.ts --workload dialogue --sampler recommended --seed 42

# Per-op GPU breakdown (a separate process — see the warning below)
deno run -A tools/mtp-bench/main.ts --workload freeform --sampler greedy --gpu-timing
```

## Protocol

One process = one configuration (the convention of `tools/ram-peak/measure.ts`).

1. Acquire the GPU, wrap `Session.prototype.run`, open the pipeline with the drafter
   (`speculative: {}`, or `{ k }` when `--k` was given) and the full PLE budget.
2. Build the workload's chat messages, encode them once with `gemma4ChatPrompt`. Both modes send the
   **same prompt token ids** from the same starting position; each turn gets a fresh sequence.
3. **Warm up** with one `plain` and one `speculative` turn. They are recorded (`warmup: true`) but
   excluded from the summary: the first turns include shader translation and params construction.
4. Run `--rounds` repetitions of **ABBA** — `plain`, `speculative`, `speculative`, `plain` — which
   cancels the order effect (later turns being systematically faster or slower).
5. Summarise with **medians**, never means: a single turn can spike (PLE shard re-reads, clock state
   changes), and a mean carries the spike into the ratio.

### Denominators

- `msPerToken` divides by **`tokens - 1`**. The first token comes out of the prefill run, so the
  decode/cycle work of a turn covers `tokens - 1` tokens. Dividing by `tokens` makes both modes look
  faster and distorts the ratio.
- `tokensPerCycle` is `(accepted + cycles) / cycles`: a rejected cycle still commits one token, so a
  cycle confirms `1 + a` tokens.
- `hostMsPerToken` (plain) and `hostMsPerCycle` (speculative) subtract the run walls from the
  generation wall. That remainder is the time spent outside `Session.run` — sampling, detokenising,
  event delivery, PLE reads — the part speculation cannot remove.

**The `plain` baseline is measured on a device that holds the drafter.** One process with alternating
turns is what makes the two paths comparable at all, and that process opened the pipeline with the
drafter, so the `plain` turns also run with the drafter's weights resident and with whatever pressure
the drafter puts on the buffer pool. A `plain` turn here is therefore not the same thing as a `plain`
turn of a drafter-less distribution: if this one is the slower of the two, `speedup` is overstated by
that difference. The MTP head is small next to the target model, so the effect is expected to be
small — but that expectation is not measured.

**`tokensPerCycle` and `msPerToken` do not count the same tokens.** When a stop token lands in the
middle of a cycle, the rows behind it are not delivered, so that turn has `tokens - 1 <
accepted + cycles`. `msPerToken` is per **delivered** token and `tokensPerCycle` is per **committed**
token; the two denominators are not interchangeable, and multiplying one by the other does not give
`cycleMs`.

## Workloads

Prompt material comes only from **git-tracked files inside the repository** (`outputs/` is untracked,
so reading from there would mean the same script produced a different prompt), and the document is
cut at a **paragraph boundary** (the last blank line before the limit). Cutting on a character count
ends mid-sentence and makes generation degenerate into a repeated phrase, which changes the
acceptance rate for a reason that has nothing to do with the drafter. Both rules are the ones
`tools/export-recipes/gemma4/export_drafter.py` uses for the drafter's golden material.

| Name        | Shape                                                                                                             | Why it is here                                                          |
| ----------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `extract`   | One user turn: instruction + `tools/exporter/README.md` (cut to `--document-chars`) + "list every shell command…" | Long context, copied verbatim from the prompt — the high-acceptance end |
| `summarize` | Same instruction and same document + "summarize in 10 bullet points"                                              | Same context, own words — acceptance drops                              |
| `dialogue`  | user → a **fixed** assistant answer (a baked constant) → "expand on the second point"                             | A continuing conversation, the practical shape                          |
| `freeform`  | One short creative request (the same text as the drafter golden's short case)                                     | The thinnest context — the low-acceptance end                           |

No system message is added in any of them. The assistant turn of `dialogue` is a constant in
`workloads.ts`: if it were generated, the prompt would differ between runs and neither ABBA nor the
median could cancel that.

## Output

One JSON line with these top-level keys:

| Key       | Contents                                                                                                                                                                                                                                                                                                                                                 |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tool`    | `"mtp-bench"`                                                                                                                                                                                                                                                                                                                                            |
| `host`    | `os`, `arch`, `deno`, and the `adapter` (`vendor` / `architecture` / `device` / `description`)                                                                                                                                                                                                                                                           |
| `config`  | The effective value of every option. Knobs that are not in effect are `null` (`k` is `null` when the drafter's own step count was used). `config.asset` identifies what was measured: `defaultModel`, `defaultQuant`, and `manifestSha256` — the SHA-256 of the manifest body, which pins the asset because a distribution carries no version of its own |
| `prompt`  | `tokens` (the encoded prompt length) and the `messages` that produced it                                                                                                                                                                                                                                                                                 |
| `turns`   | One record per turn, warm-ups included: mode, round, `warmup`, `tokens`, `stopReason`, the token `ids` and their decoded `text`, `turnMs` / `firstTokenMs` / `generationMs`, per-kind run counts and wall totals, and `speculation` for speculative turns                                                                                                |
| `summary` | See below                                                                                                                                                                                                                                                                                                                                                |
| `gpu`     | Present only with `--gpu-timing`                                                                                                                                                                                                                                                                                                                         |

`summary` (warm-ups excluded, every value a median over turns):

| Field                           | Meaning                                                                                                                                                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plain` / `speculative`         | `turns`, `msPerToken`, `generationMs`, `turnMs`, `firstTokenMs`                                                                                                  |
| `plain.runs[kind]`              | `countPerTurn` and `msPerRun` per run kind. A kind with no runs has no `msPerRun` field (writing `0` would read as "measured, and it was 0 ms")                  |
| `plain.hostMsPerToken`          | `(generationMs − decode wall) / (tokens − 1)`                                                                                                                    |
| `speculative.tokensPerCycle`    | `(accepted + cycles) / cycles`                                                                                                                                   |
| `speculative.k`                 | The effective step count, `acceptedHistogram.length − 1` (the tally has one bucket per acceptance count, `0..k`). Turns that disagree are an error, not a median |
| `speculative.cycleMs`           | `generationMs / cycles`                                                                                                                                          |
| `speculative.hostMsPerCycle`    | `(generationMs − draft wall − verify wall) / cycles`                                                                                                             |
| `speculative.acceptedHistogram` | Element-wise sum of the per-turn histograms (index = accepted drafts in a cycle)                                                                                 |
| `speedup`                       | `plain.msPerToken / speculative.msPerToken`                                                                                                                      |
| `identity`                      | `plainConsistent`, `speculativeConsistent`, `identical`, and `firstDivergence` when the two modes disagree                                                       |

`identity` is a correctness check, not a performance one: speculation is a speed-only knob, so the
token ids of the two modes must match exactly. If `identical` is false, read that before the ratio —
the ratio is comparing two different generations.

## `--gpu-timing`

With the switch, each run's `lastRunTiming` is accumulated per run kind (warm-ups excluded) into
`gpu[kind]`: `runs`, `msPerRun`, `dispatchesPerRun`, `clampedNegativeSamples`, and the top 12
pipeline keys by GPU time with their own `msPerRun` and `dispatchesPerRun`.

**Never compare a timing-on wall clock with a timing-off one.** A timing-enabled device opens one
pass per dispatch, so the wall clock (and therefore `speedup`) grows. Take the ratio from a run
without the switch and the breakdown from a run with it. `clampedNegativeSamples` above zero means
the driver's timestamps were non-monotonic and the breakdown itself is suspect.

On macOS (Metal) the switch prints a warning: timestamp collection loses the device there
(`docs/limitations.md`). The tool does not refuse — the constraint belongs to the OS/driver stack, so
no removable gate is added to karume.

## Notes

- Generated files belong under `outputs/bench/<model>/<YYYY-MM-DD>_<purpose>/`, which is untracked
  (`docs/assets-layout.md`). Numbers meant to last go into `docs/research/`, and adoption decisions
  into `docs/perf-ledger.md`.
- The PLE budget default is computed from the mirror's `ple_index` asset the same way
  `packages/models/tests/helpers/ple-budget.ts` does it. Byte budgets, not shard counts: shard width
  changes with the asset generation, so "N shards" means a different amount of RAM per generation
  (ADR 0085).
- `--k` is passed through to the library, which owns the range gate. This tool does not re-implement
  a second gate for the same knob.

# flag-bench — one execution flag at a time, relative to a reference

A CLI that compares **Gemma 4 execution options** (the static session flags a pipeline accepts —
`linearGemvReduce`, `fuseRmsNormAdd`, `fuseLinearStaticQuantize`, `packedStaticQuantize`,
`stateAttentionReduce`, …) **against a reference configuration on the same machine**, for the regular
and the QAT distributions, E2B and E4B. It is built for hosts without a browser: it measures under
Deno, with the GPU time (`timestamp-query`) as the primary metric and the wall clock as the secondary
one.

Each `--set <label>=<json>` is one configuration: a JSON object of pipeline options that override the
selected quant's declaration. The **first** `--set` is the reference; every other set is reported as a
percentage against it.

Kernels, pipelines and the generation loop are **not modified**. Options reach the library exactly as
an application would pass them to `Gemma4Pipeline.fromPretrained` / `Gemma4QatPipeline.fromPretrained`.

## Usage

```
deno run -A tools/flag-bench/main.ts --source <mirror dir> --family <normal|qat> --model <e2b|e4b> \
  --quant <name> --set <label>=<json> [--set ...] [options] --out <file.jsonl>
```

| Option                  | Default  | Meaning                                                                                                                                                                                      |
| ----------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--source <dir>`        | required | Local distribution mirror (`models/karume-gemma4`, `models/karume-gemma4-qat`), read through `denoDirectory`                                                                                 |
| `--family <name>`       | required | `normal` = `Gemma4Pipeline`, `qat` = `Gemma4QatPipeline`                                                                                                                                     |
| `--model <name>`        | required | `e2b` / `e4b` — must exist in the mirror's manifest                                                                                                                                          |
| `--quant <name>`        | required | The quant to load (for example `i4`, `i4-gemvpar`, `i4-fast`). No default: the manifest's default quant changes between releases, and the measured quant must be the one on the command line |
| `--set <label>=<json>`  | required | One configuration (repeatable, at least one). The first is the reference. See below                                                                                                          |
| `--rounds <int>`        | `2`      | ABBA repetitions (see Protocol)                                                                                                                                                              |
| `--new-tokens <int>`    | `96`     | `maxNewTokens` per generation; at least 2 (the decode rate needs a second token)                                                                                                             |
| `--capacity <int>`      | `4096`   | KV capacity of every sequence                                                                                                                                                                |
| `--gpu-timing` (switch) | off      | Acquire every device with `acquireGpu({ gpuTiming: true })` and record the per-run GPU time. **Run it as a separate invocation into a separate file** — see Discipline                       |
| `--out <file.jsonl>`    | required | JSONL output. The parent directory must exist and the file must **not** exist (it is created before any model is loaded, never appended to — one file is one invocation and one timing mode) |

Unknown options, unknown `--set` keys and values of the wrong type fail loudly: a mistyped knob that
silently fell back to a default would make the recorded configuration disagree with what was measured.

### `--set` keys

A set accepts a subset of `Gemma4PipelineOptions` — the options that shape execution:

| Key                          | Value                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `linearGemvReduce`           | `"sequential"` / `"parallel"` / `"parallel-subgroup32"`                                   |
| `fuseRmsNormAdd`             | boolean                                                                                   |
| `fuseLinearStaticQuantize`   | boolean                                                                                   |
| `packedStaticQuantize`       | boolean                                                                                   |
| `stateAttentionReduce`       | `"sequential"` / `"parallel"` / `"parallel-fused"`                                        |
| `rmsNormReduce`              | `"workgroup"` / `"subgroup32"`                                                            |
| `submitPolicy`               | all four fields: `timeBudgetMs`, `initialChunkSize`, `minChunkSize`, `maxChunkSize`       |
| `planBackingBudgetBytes`     | integer                                                                                   |
| `linearGemvRowsThreadTarget` | integer                                                                                   |
| `chunkLength`                | integer                                                                                   |
| `chunkBuckets`               | array of integers                                                                         |
| `pleResidency`               | `"host"` / `"gpu"` (`"gpu"` cannot be combined with `--gpu-timing` — the library says so) |
| `maxResidentPleBytes`        | integer                                                                                   |

The tool checks only the **type** of each value. Ranges and combinations (for example
`fuseLinearStaticQuantize` requiring `linearGemvReduce: "parallel"`) are checked by the library, which
is the single owner of those rules. The quant-declared keys are resolved with the pipeline's own
function before any model is loaded, so an invalid combination fails in the first second.

`speculative` is not accepted: the QAT pipeline has no drafter, and what speculation is worth is a
different question, answered by `tools/mtp-bench`.

A subgroup variant (`"parallel-subgroup32"`, `"subgroup32"`) makes the tool acquire that set's device
with `subgroups: true`. On an adapter without subgroups (for example the Intel Arc B570) this fails
loudly in a pre-flight step, before any model is loaded — it never falls back to the non-subgroup
variant.

**The reference condition is explicit.** Precedence is: explicit override > the quant's `session`
declaration > the family / runtime default. The Gemma family's `stateAttentionReduce` default is
`"parallel"`, not the runtime's reference `"sequential"`, so a reference set meant to be the reference
path must say `{"stateAttentionReduce":"sequential"}`. An empty `{}` means "the quant's declaration
and the family defaults, as shipped".

```sh
# Wall clock (timing off): reference path vs. one flag
deno run -A tools/flag-bench/main.ts --source models/karume-gemma4-qat --family qat --model e2b \
  --quant i4 \
  --set 'ref={"stateAttentionReduce":"sequential"}' \
  --set 'gemvpar={"stateAttentionReduce":"sequential","linearGemvReduce":"parallel"}' \
  --out outputs/bench/karume-gemma4-qat/2026-09-26_flags/e2b-wall.jsonl

# GPU time: the same sets, a separate invocation and a separate file
deno run -A tools/flag-bench/main.ts --source models/karume-gemma4-qat --family qat --model e2b \
  --quant i4 \
  --set 'ref={"stateAttentionReduce":"sequential"}' \
  --set 'gemvpar={"stateAttentionReduce":"sequential","linearGemvReduce":"parallel"}' \
  --gpu-timing --out outputs/bench/karume-gemma4-qat/2026-09-26_flags/e2b-gpu.jsonl
```

Progress goes to stderr, one line per generation; **stdout carries only the final summary line**.

## Protocol

1. Parse and type-check every set; resolve the quant's `session` declaration with each set's
   overrides (the pipeline's own function); create the output file.
2. **Pre-flight**: acquire and release one device per set with that set's requirements
   (`timestamp-query`, subgroups), so an unsupported set fails before any model is read.
3. Visit the sets in **ABBA** order: one round is `S0 … Sk` then `Sk … S0`, repeated `--rounds` times.
   Each set appears twice per round at mirrored positions, so a drift that grows with process time
   (clock warm-up, page cache filling) lands evenly on every set instead of on the later ones.
4. Every visit builds a **fresh `GpuContext` and pipeline** for that set — the flags are static
   Session options, fixed at construction — and disposes both afterwards (pipeline first, then the
   device).
5. Inside a visit, each of the two prompts of `tools/llm-speed/browser/cases.json` (the same English
   and Japanese chat prompts the browser benchmark uses; their token ids are checked against the chat
   template, and the stop set against the tokenizer) runs **one warm-up and two measured
   generations**: greedy (`temperature: 0`), a fresh sequence each time, the fixture's stop tokens.
   The warm-up absorbs shader translation and first-use allocation; it is recorded but excluded from
   the summary.
6. Summarise with **medians** over the measured generations of all visits and both prompts — a single
   spike (a PLE block re-read, a clock change) would otherwise be carried into the ratio.

## What is measured

Per generation (one JSONL line, `type: "run"`):

| Field                                              | Meaning                                                                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `set`, `visit`, `round`, `prompt`, `rep`, `warmup` | Where the line comes from (`rep` 0 is the warm-up)                                                                              |
| `gpuTiming`                                        | Whether the device had GPU timing enabled                                                                                       |
| `ttftMs`                                           | `generate()` call → first token event. The sequence (KV allocation) is created before the clock starts                          |
| `decodeMsPerToken`                                 | `(completion − first token) / (delivered − 1)`                                                                                  |
| `delivered`                                        | Token events received (a stop token counts)                                                                                     |
| `tokensSha256`                                     | SHA-256 of the token ids spelled as JSON (`[1,2,…]`, UTF-8)                                                                     |
| `gpu`                                              | With `--gpu-timing` only: per run kind (`prefill` / `decode`), `runs`, `msPerRun`, `dispatchesPerRun`, `clampedNegativeSamples` |

The final line (`type: "summary"`, also printed to stdout) carries the derived values:

| Field                                                              | Meaning                                                                                                                                                                |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gpuTiming`                                                        | The timing mode of the whole file                                                                                                                                      |
| `host`                                                             | `os`, `arch`, `deno`, and the `adapter` (`vendor` / `architecture` / `device` / `description`)                                                                         |
| `config`                                                           | Source, family, model, quant, `manifestSha256` (SHA-256 of the manifest body), the quant's `quantSession` declaration, rounds, tokens, capacity, prompts               |
| `reference`                                                        | The first set's label                                                                                                                                                  |
| `sets[]`                                                           | Per set: `overrides`, `resolvedSession` (declaration + overrides, resolved as the pipeline does, plus `stateAttentionReduce`), `deviceFeatures`, and the medians below |
| `decodeMsPerToken`, `ttftMs`                                       | Medians of the measured generations                                                                                                                                    |
| `gpuDecodeMsPerStep`, `gpuDispatchesPerStep`, `gpuPrefillMsPerRun` | With `--gpu-timing` only: medians of the per-generation values                                                                                                         |
| `deltaPercent`                                                     | `(value / reference − 1) × 100` for each of the above — negative is faster / fewer                                                                                     |
| `tokensIdenticalAcrossVisits`                                      | Every generation of this set (warm-ups included) produced one token sequence per prompt                                                                                |
| `tokensMatchReference`                                             | …and it is the reference's sequence. `false` means the flag changed the output — expected for flags that change summation order                                        |

## Discipline

- **Never mix a timing-on file with a timing-off one.** A timing-enabled device opens one compute pass
  per dispatch, and passing the observation callback (`onRunDiagnostics`, needed to read the per-run
  GPU time) makes the pipeline skip its on-GPU greedy output path. A timing-on wall clock is therefore
  neither the product path nor comparable: in a short smoke run on the Arc B570 (E2B `i4`, 8 tokens)
  the same decode read ≈36 ms/token with timing off and ≈138 ms/token with it on. The tool passes the callback only with `--gpu-timing`, so a
  timing-off run takes the shipping path. Take wall-clock ratios from timing-off files and GPU ratios
  from timing-on files; the summary refuses to fold both kinds of line together.
- **The Deno wall clock has a floor.** Under Deno every decode step waits on a fence whose polling adds
  about 10 ms per token regardless of the work (`docs/research/2026-09-19-qat-speed-recon.md` §3.2), so
  a GPU-side gain is compressed in the wall-clock ratio. The GPU time is the primary metric; the wall
  clock is a sanity check that the gain is not eaten elsewhere.
- **Under Deno the GPU time is in device ticks, not nanoseconds.** Deno returns `timestamp-query`
  values without applying the adapter's timestamp period, and the WebGPU API does not expose that
  period (`docs/known-issues.md`, the Intel Arc B570 section). On the B570 the period is 52.0833 ns, so
  every `ms` of GPU time is about **52 times too small** in absolute terms. The period is a constant of
  the device, so `deltaPercent` is unaffected — read GPU results as ratios on such adapters.
- **PLE placement is the shipping default** (a two-block host cache) unless a set says otherwise. Put
  `maxResidentPleBytes` or `pleResidency` into **every** set if the host-side PLE reads should be
  pinned; they do not affect the GPU time.

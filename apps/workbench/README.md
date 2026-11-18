# Karume workbench

A bare browser app for measurements and checks on a real GPU. One file is one **entry**; the page
lists the entries, shows the selected entry's parameters, runs it on a click on **Run**, and keeps the
result as a JSON **record**. A [batch](#batches) runs several entries in a row. The same entry modules
also run under Node + Dawn (`run.ts`). The app is private (not published).

| Entry | What it does |
| --- | --- |
| `anima-single-file` | Reads a local distribution and a file under `inputs/`. A single Anima weights file through `anima.Pipeline.fromSingleFile` ([the spec](../../docs/spec/single-file.md)) against the `anima` mount: the pipeline is built and makes two 512² images of one request, then, with `compare` on, `fromPretrained` of the same model and quant does the same. Parameters: `file` (relative to `inputs/`; default the upstream `anima-base-v1.0` single file), `model` (`anima-v1.0`), `quant` (empty: the model's default), `textConditioner` (`file` or `distribution`), `retainConverted` (off: `fromSingleFile` keeps the converted weights in memory, so its second generation reads nothing of the file), `compare` (on), `steps` (4), `seed` (42), `source` (`cached` or `direct`). In a browser the file is read by ranged requests to the dev server's `/inputs/` route. No verdict; the record (`karume-anima-single-file/1`) has which components came from the file, the build time and the two generations' times and pixel sha256 per pipeline, and whether the images are equal. |
| `attention-bench` | The two a8 attention routes — S stored in f16 over three dispatches (s16), and the online int8 kernel — timed per kernel at Wan2.2 TI2V-5B's shapes with the kernels, row blocks and variants production would use on this host (`runAttentionRouteBench` of `@karume/runtime/tune`), apart from a generation. Parameters: `shapes` (default: all three), `rowsPerDispatch` (default 0: the first row block production would dispatch; a number is capped at each shape's query count), `rounds` (5; the min is kept), `quantization` (also time the shared quantization dispatches; default on), `onlineTiled` (also time the tiled key-axis candidate of perf-ledger K-106 against the online kernel in the same rounds, with its producer and an identity check; default off). Times are GPU timestamps when the adapter lists `timestamp-query`, else the wall clock: compare ratios then, not absolute values. The record's `data` is the runtime's `karume-attention-bench/2` report. |
| `gemma4-bench` | Reads a local distribution. The Gemma 4 decode bench: its arms (a model × a preset × switches — [the grammar](#the-decode-benchs-arms)) are visited in A B B A rounds (even rounds in order, odd rounds in reverse), and each visit builds its arm's pipeline (the workbench's own builder, `src/gemma4/pipeline.ts`), runs `chats` fresh chats per prompt of a fixed two-prompt set (chat 0 is the cold one, recorded and left out of the summary) and releases it before the next visit, so two models are never resident at once. The device is acquired through a GPU tap (`src/gemma4-bench/tap.ts`) that requests `timestamp-query` itself and hides it from the runtime, so karume's timing mode stays off: a tapped arm's chats get pass-boundary timestamps and pass, dispatch and submit counts in the product's own pass structure; a split arm times every dispatch in a pass of its own for the per-family and per-key breakdown (the extra pass boundaries inflate its times: read its shares); an arm in the tap's host mode gets wall-clock marks at the WebGPU calls instead of pass timing, and its chats are cut into [the host's segments of a decode token](../../docs/spec/workbench.md#the-decode-benchs-host-segments); a tap-off arm is the wall-clock control. Parameters: `arms` ([the grammar](#the-decode-benchs-arms); empty: one arm per Gemma 4 mount whose mirror is present, at its default model and preset, the tap on, regular Gemma 4 first), `chats` (3, 2–1000), `newTokens` (128, 2–32768), `rounds` (4, 1–1000), `sampler` (`greedy`, or `distribution`: the distribution's sampler with seed 0; an arm with the switch `sampled` runs the distribution's whatever this is), `prompts` (`few-ids`, `many-ids`: one on each side of 32 distinct token ids; default both), `subgroups` (on), `source` (`cached` or `direct`). No verdict when it finished; `incomplete` when a failure stopped it after a chat (what finished is kept). The summary has one line per arm and prompt over the warm chats (the median decode tokens/s and the max, the median ms/token, the ratio to arm 1's, the median GPU ms/token of a tapped arm, whether the outputs are identical, and — when the chats' stops report [the candidate selection](../../docs/spec/generation.md#candidate-selection) — the leading logits a sampled decode step asked for, the steps drawn from them and the steps that read the whole row), the split arms' breakdown and the host arms' segments; `data` has the workload, the arms with their distributions and options, every visit and chat, and the summary. |
| `gemma4-session-ab` | Reads a local distribution. The wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, judged by `judgeSessionAb`) of a Gemma 4 greedy decode with and without the candidate rule `candidate/gelu_mul` of `@karume/candidate-rules`: both arms load the same model and preset (the workbench's own builder, `src/gemma4/pipeline.ts`), every timed run is one generation from a fixed prompt, and identity is token identity. The arms are built without `onRunDiagnostics`; a count probe of each arm before the A/B reads the rule's hits and the dispatches. Parameters: `model` (`<mount>/<model>`; empty: the default model of the first of `gemma4`, `gemma4-qat`, `gemma4-qat-mobile` whose mirror is present), `preset` (empty: the manifest's default), `tokens` (96, 2–512), `rounds` (4, ≥ 3), `minGain` (0.01), `openPerVisit` (on), `probeCounts` (on), `f16Kv` (off), and for `gemma4-qat/26b` only `expertCacheSlots` (a count, a comma list or `fitted`), `hostTier` (GiB or `all`) and `depth` (0 = off, up to 4), `source` (`cached` or `direct`). The verdict is token identity; the judgement is in the summary. |
| `gemv-sweep` | The GEMV choices of `linear` measured on this host's GPU with `runGemvSweep` of `@karume/runtime/tune`, one call per case with idle time before every case but the first (the per-case loop of `geometry-sweep`, `src/per-case-sweep.ts`), merged into one `karume-gemv-sweep/4` record (`data`: each row carries the fingerprint of its kernel, each case its own). The cases are the shapes of the default profile's GEMV rows (Gemma 4 E2B, E4B, 12B, 26B A4B, regular and QAT), the other integer-storage linears a decode token or a draft step of those models runs at one row (the int2 and int4 output heads, the MTP drafters) and the linears of regular Gemma 4 E2B and E4B as they ship, in bf16 (measured at rows 1, 4 and 8; past one row their baseline is the GEMM, `gemm`, and they have no row-block height and no subgroup candidate). Per case and row count it measures the baseline — what a Session at the level `exact` runs with the default profile —, the identical class (the sequential M = 1 variants and the row-block heights; every output word must be the baseline's) and the reorder class (the parallel GEMV at 2–32 lanes, plain, staged and — where the device has the subgroup features — subgroup, at 1 to 8 rows; held to the band of `linear` against an f64 oracle on the host), measured together: a warm start on the baseline, a quick look at every kernel (one short timed pass; only the references — the baseline and the default profile's choice at `reorder` — and the kernels within the margin of their class's bar go on, the others are screened rows), the kernels that go on calibrated and checked, then rounds that each time every one of them once in a rotating order; at rows 4 and 8 a reorder candidate is measured only when the case's M = 1 group of the same run leaves it adoptable (timed, its check held and, in a steady group, ×1.10 or more over the default's choice at `reorder`), the others are skipped rows with their kernel and launch only ([the spec](../../docs/spec/workbench.md#how-the-gemv-sweep-times-a-row-count)). Parameters: `candidateSet` (`quick` — M = 1, the identical class on the 3 × 3 grid around the baseline, the reorder class in full —, `full` — rows 1, 4 and 8, every row count a derivation rule reads, and the whole grid — or `row-blocks` — rows 16 and 64), `cases` (default none: every case, the 25 bf16 ones included — 115 in all), `rows` (default none: every row count of the set), `candidates` (default none: every candidate; the references are measured either way), `quickLookMargin` (×1.25; 0 times every kernel), `rounds` (5), `idleSeconds` (10). The generator's `--gemv-plan` writes a batch of such runs that measures what the records at hand lack ([its README](../../tools/geometry-profiles/README.md#the-command)). The packed-activation form is not measured. The summary has the kernels screened (and skipped, when there are any) in its totals, and per case and row count the baseline's time and the spread of its rounds (its largest ÷ its smallest: how steady the device was), the fastest identical and the fastest reorder candidate (×: speed against the baseline), the default profile's choice at `exact` and at `reorder` (×: the fastest of its class's time ÷ the choice's, ×1.000 when the choice is the fastest) and the kernels screened and skipped (`· 12 screened · 640 skipped (not adoptable from M = 1)`). The verdict is `fail` when an identical candidate's words differ, a reorder candidate is outside the band or a row of a finished case errored, `incomplete` when a device loss or an error stopped the run; a screened or skipped row has no check; timing is not part of it. `deriveLinearGemvChoices` (the generator's `--gemv-records`) reads the record. |
| `geometry-sweep` | The GEMM tile geometries measured on this host's GPU with `runGeometrySweep` of `@karume/runtime/tune`, one call per case with idle time before every case but the first (cooling: heat moves the default geometry's re-measurement, which the derivation checks), merged into one `karume-geometry-sweep/4` record. Each case starts with a quick look: only the default geometry and the candidates whose quick time is within the margin of the default's are timed in full, the others are screened rows. The record's `data` is that record, one input of `deriveGeometryProfile` as it is. Parameters: `candidateSet` (`quick`, `quick+` — the default —, `full`, or the evaluation set `k88`, which sweeps a case table of its own and from whose record no table is derived — its summary compares every candidate with the table's geometry, so run it at margin 0; `quick+` is `quick%2B` in a URL; in `quick+` and `full` a linear / matmul / bmm case measures every tile also with each skeleton form (`…:tb-un`, `…:tas-tb-un`) and a conv2d case also with the counter decode (`…:dec`) and with both, so a table derived from the record can adopt them), `ops` (default: every op family), `cases` (default none: every case of the chosen op families in the chosen set; a case the chosen set does not have is an error, a case of an op family not chosen is not swept), `geometries` (default none: every candidate; comma-separated names, each a candidate of every chosen case), `quickLookMargin` (×1.25; 0 times every geometry), `rounds` (5), `idleSeconds` (10). A cancel or a lost device keeps the cases that finished. `deriveGeometryProfile` rejects a record whose fingerprints (`caseSets`, `defaultKernels`, `candidateKernels`) differ from the deriving code's, so run it from the checkout that will derive the table and leave `packages/runtime` alone until then (the dev server reloads edited modules); keep the tab in front and nothing else on the GPU. Without `timestamp-query` the sweep times by the wall clock, and the derivation rejects such a record. The verdict is `fail` when a candidate of a finished case differs from its default or errored, `incomplete` when a device loss or an error stopped the run, `pass` when every planned case finished clean; timing is not part of it. |
| `kernel-check` | The kernel identity checks of `@karume/kernel-checks` — the cases and criteria of the runtime's node GPU tests — on this host's backend (D3D12 in Chrome on Windows, Metal on macOS, Dawn under Node). Parameter `checks`: the checks to run (default: all twelve, the release check). The verdict rules are in `src/kernel-check/verdict.ts`, the checks in [the package README](../../packages/kernel-checks/README.md). |
| `session-ab` | Reads a local distribution. The wall-clock A/B of one component forward with and without the fusion rule its pipeline passes by default: `target` `moss-dit-forward` (the MOSS-Image `dit` at the 1024² sequence, `models/ew_chain`), `wan22-vae-next` (the Wan2.2 `vae_decoder_next` chunk graph) or `anima-vae-decode` (the anima `vae_decoder` tile graph, the default; both `models/vae_rms_silu`). The component of the manifest's default model at `preset` (empty: the default preset) is opened once; the arms are `sessionArm`s over it, every visit opens and closes its Session, and a run is one forward on seeded feeds whose every output word the runner compares. Parameters: `target`, `preset`, `rounds` (4, ≥ 3), `minGain` (0.01), `source` (`cached` or `direct`). The verdict is output identity; the judgement and the counts per forward are in the summary. |

## Run in a browser

```sh
export PATH=$HOME/.local/share/vite-plus/bin:$PATH   # Node 24 + pnpm from Vite+
pnpm install
pnpm --filter @karume/workbench dev
```

The dev server listens on **http://127.0.0.1:8791** (loopback only; the playground uses 8790). From
another machine, open it through an SSH tunnel as described in
[the playground README](../playground/README.md#from-another-machine-ssh-tunnel), with port 8791.
Every response carries COOP / COEP, so the page is cross-origin isolated.

The URL holds the selected entry, its parameters and the declared backend
(`?entry=kernel-check&checks=gemv-integer,gemv-planes&backend=d3d12`), so a link reproduces a run
setup; nothing runs before a click on Run. WebGPU does not tell the backend: type it in the
**backend (optional)** field, or leave it empty and the record has none. Cancel stops the run at the
entry's next check point; a cancelled or failed run still makes a record.

Under the dev server the page POSTs each record to the server, which writes it under `outputs/` and
shows the path. **Download record** and **Copy record** give the same text. A production build
(`pnpm --filter @karume/workbench build`, served by `vp preview`) writes nothing: download the record.

## Run under Node

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts kernel-check
flock /tmp/karume-gpu.lock node apps/workbench/run.ts kernel-check --checks=gemv-integer,gemv-planes
node apps/workbench/run.ts --list
```

Parameters are `--name=value` (a `choices` list is comma-separated). The registry and the entries
load through Vite's module runner, so the run sees the same modules as the page. Dawn's toggles come
from `KARUME_DAWN_FLAGS` as in the tests; a `backend=<name>` toggle selects the backend and is
recorded, otherwise the record names Dawn's default for the platform (Vulkan on Linux). Progress goes
to stderr; the outcome, the summary and the record's path go to stdout; Ctrl-C cancels the run. The
exit codes are in [the spec](../../docs/spec/workbench.md#the-node-command).

## Example setups

The geometry sweep's two standing setups are plain parameter values.

**V-15** — every conv3d w8a8 case with every i8a8 geometry of the `full` set, 5 rounds, 90 s idle:
the record from which the derivation chooses `i8a8.conv3d`.

```text
http://127.0.0.1:8791/?entry=geometry-sweep&candidateSet=full&ops=i8a8-conv3d&rounds=5&idleSeconds=90
```

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts geometry-sweep --candidateSet=full --ops=i8a8-conv3d --rounds=5 --idleSeconds=90
```

**K-88** — every case of the `k88` table (linear, matmul, conv2d), 5 rounds, 90 s idle: per case
the tile of this adapter's table geometry (without the fields, also where the table's rule carries
them) against the same tile with its fastest skeleton form, with the conv2d counter decode and with
both, and the census-weighted sums per unit. No table is derived from this record.

```text
http://127.0.0.1:8791/?entry=geometry-sweep&candidateSet=k88&ops=linear,matmul,conv2d&rounds=5&idleSeconds=90
```

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts geometry-sweep --candidateSet=k88 --ops=linear,matmul,conv2d --rounds=5 --idleSeconds=90
```

### The decode bench's arms

The `arms` of `gemma4-bench` are comma-separated, arm 1 first (the one the summary compares the
others with); an arm is `<mount>/<model>[@<preset>][:<switch>]…`:

- the mount is `gemma4`, `gemma4-qat` or `gemma4-qat-mobile`, the model one of its manifest's;
- without `@<preset>` the arm runs the manifest's default preset;
- the switches: `tap-off` (the wall-clock control; the tap is on otherwise), `split` (one timed pass
  per dispatch; not with `tap-off`), `host` (the tap's host mode: wall-clock marks at the WebGPU
  calls and at the checks of the chat's signal, no pass timing — the host's segments of a decode
  token; not with `tap-off` or `split`), `linear-reorder` (the Session option
  `numerics: { ops: { linear: "reorder" } }` on a preset that does not declare linear at `reorder` —
  the arm against the same preset at `exact`; refused on a preset that declares it),
  `integer` (the Session option `integerGemv: true`, with
  `packedStaticQuantize: true` and linear at `reorder` where the preset does not declare them),
  `ple-gpu` (the pipeline option `pleResidency: "gpu"`), `ahead2` / `ahead4` / `ahead8` (the
  pipeline option `decodeAhead`; it needs `ple-gpu`, except on 12B, which has no PLE), `no-signal`
  (the chats are sent without an `AbortSignal`, so the generation loop does not yield to the event
  loop before a run; a cancel then takes effect between two chats) and `sampled` (this arm's chats
  use the distribution's sampler with seed 0, whatever the `sampler` parameter is).

None of `,`, `@` and `:` occurs in the mirrors' model and preset names, and none needs quoting in a
shell or escaping in a URL query. An arm that does not parse, names what its manifest does not have
or asks for a pair the pipeline refuses is an error that names the arm, before any GPU work.

The mobile QAT E2B against itself with the integer GEMV:

```text
http://127.0.0.1:8791/?entry=gemma4-bench&arms=gemma4-qat-mobile/e2b,gemma4-qat-mobile/e2b:integer
```

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench --arms=gemma4-qat-mobile/e2b,gemma4-qat-mobile/e2b:integer
```

Regular E2B in bf16 at `exact` against the same preset with linear at `reorder` (the parallel GEMV on
the shapes the profile names):

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench --arms=gemma4/e2b,gemma4/e2b:linear-reorder
```

The q4_0 QAT E2B at `i4-fast` split per dispatch, and with the PLE on the GPU and four decode steps
per batch, tap off, over two rounds:

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench --arms=gemma4-qat/e2b@i4-fast:split,gemma4-qat/e2b@i4-fast:ple-gpu:ahead4:tap-off --rounds=2
```

The host side of a q4_0 QAT E2B decode token at `i4-fast`: the GPU time (the tapped arm), the wall
clock (tap off), and the host's segments with the signal, without it, and with the distribution's
sampler:

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts gemma4-bench --arms=gemma4-qat/e2b@i4-fast,gemma4-qat/e2b@i4-fast:tap-off,gemma4-qat/e2b@i4-fast:host,gemma4-qat/e2b@i4-fast:host:no-signal,gemma4-qat/e2b@i4-fast:host:sampled --rounds=3
```

## Batches

A **batch** is a JSON file of steps — an entry id and the parameter values that differ from the
defaults — run in order on one click (or one command). Write it as `<id>.json` (kebab-case) in one of
two directories:

- `apps/workbench/batches/` — tracked; the dev server and a production build list it (a production
  build only when every step names an entry it holds, so not one that reads a local distribution);
- `inputs/workbench/batches/` under the repository root — untracked (`inputs/` is outside git); only
  the dev server lists it, marked `local`, read when the page starts.

```json
{
  "format": "karume-workbench-batch/1",
  "title": "Kernel checks, then a quick sweep",
  "description": "Optional, one paragraph.",
  "steps": [
    { "id": "kernel-check" },
    { "id": "geometry-sweep", "parameters": { "candidateSet": "quick", "rounds": 1, "idleSeconds": 0 } }
  ]
}
```

A step is the `entry` object of a saved record, so a record's `entry` pastes in as a step. Values
are read as the URL reads them (`3` and `"3"` alike, a list is comma-joined); nothing else is allowed
in the file — no repeats, conditions or expected results, and no backend (the backend stays the
page's field). The whole file is checked before anything runs: a batch with problems is listed with
every problem, each naming its step (`step 2 (geometry-sweep): Parameter rounds: …`), and has no Run.

In the page, the batches follow the entries in the menu in their own colour (`?batch=<id>` selects
one). Uncheck the steps to leave out, press **Run**: a failed or erroring step does not stop the
batch, Cancel stops the running step (its record is kept) and the rest. Each step leaves an ordinary
record with a `batch` field. Under Node:

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts --batch=apps/workbench/batches/synthetic-quick.json
```

`synthetic-quick` runs the three synthetic entries as a short run (the sweep at `quick`, one round,
no idle time) — not the evidence run. The rules are in
[the spec](../../docs/spec/workbench.md#batches).

## Records

A run leaves one JSON record under `outputs/workbench/` (outside git), in a directory per host and
adapter (`node-nvidia-geforce-rtx-3080-ti`, `browser-nvidia-geforce-rtx-5070-ti`); no file is ever
replaced. The fields are in [the record](../../docs/spec/workbench.md#the-record), the path rule in
[where records go](../../docs/spec/workbench.md#where-records-go).

## Add an entry

Add one file `src/entries/<id>.ts` (kebab-case) whose default export is an `Entry`
(`src/entry.ts`): `id` (the file name), `title`, a one- or two-sentence `description`, the declared
`parameters` (`boolean`, `integer` and `number` with min / max, `text`, `choice`, `choices`) and `run`. `run` gets the
typed parameters, `gpuOptions` to spread into `acquireGpu` (it acquires and destroys its own GPU
context), an `AbortSignal`, `progress(text)`, `openDistribution` (below) and `backend` (the backend's
name as the record's `environment.backend` holds it, absent when a browser run declared none), and
resolves to `{ verdict?, summary, data }`. An entry holds no UI code and no Node API: the shell and `run.ts` render and drive every entry the same
way. The registry (`src/entries.ts`) picks the file up; nothing else lists entries.

An entry that reads a local distribution (a mirror under `models/`) is a file
`src/entries/local/<id>.ts` (the directory is the declaration) and opens the mount with
`openDistribution(mount, source)` from its run context: a reference for `loadManifest` or a family's
`fromPretrained`, and the facts its record carries (`directory`, `manifestSha256`, `read`). In a
browser `source` picks the cached route (`/hf/`, the hub's CacheStorage) or the direct one
(`/models/<mount>/`); under Node the mirror's directory is read. The dev server serves the same
`/models/` and `/hf/` routes as the playground, from the shared mount table of
[`@karume/local-distributions`](../../packages/local-distributions/README.md). A production build
leaves these entries out.

## Develop

```sh
pnpm --filter @karume/workbench test        # vp test: parameters, registry, record, write-back, batches, the entries' pure helpers (no GPU)
pnpm --filter @karume/workbench typecheck
pnpm --filter @karume/workbench build
```

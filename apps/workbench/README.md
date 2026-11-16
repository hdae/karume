# Karume workbench

A bare browser app for measurements and checks on a real GPU. One file is one **entry**; the page
lists the entries, shows the selected entry's parameters, runs it on a click on **Run**, and keeps the
result as a JSON **record**. The same entry modules also run under Node + Dawn (`run.ts`). The app is
private (not published).

| Entry | What it does |
| --- | --- |
| `attention-bench` | The two a8 attention routes — S stored in f16 over three dispatches (s16), and the online int8 kernel — timed per kernel at Wan2.2 TI2V-5B's shapes with the kernels, row blocks and variants production would use on this host (`runAttentionRouteBench` of `@karume/runtime/tune`), apart from a generation. Parameters: `shapes` (default: all three), `rowsPerDispatch` (default 0: the first row block production would dispatch; a number is capped at each shape's query count), `rounds` (5; the min is kept), `quantization` (also time the shared quantization dispatches; default on), `onlineTiled` (also time the tiled key-axis candidate of perf-ledger K-106 against the online kernel in the same rounds, with its producer and an identity check; default off). Times are GPU timestamps when the adapter lists `timestamp-query`, else the wall clock: compare ratios then, not absolute values. The record's `data` is the runtime's `karume-attention-bench/2` report. |
| `gemma4-session-ab` | Reads a local distribution. The wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, judged by `judgeSessionAb`) of a Gemma 4 greedy decode with and without the candidate rule `candidate/gelu_mul` of `@karume/candidate-rules`: both arms load the same model and preset (the workbench's own builder, `src/gemma4/pipeline.ts`), every timed run is one generation from a fixed prompt, and identity is token identity. The arms are built without `onRunDiagnostics`; a count probe of each arm before the A/B reads the rule's hits and the dispatches. Parameters: `model` (`<mount>/<model>`; empty: the default model of the first of `gemma4`, `gemma4-qat`, `gemma4-qat-mobile` whose mirror is present), `preset` (empty: the manifest's default), `tokens` (96, 2–512), `rounds` (4, ≥ 3), `minGain` (0.01), `openPerVisit` (on), `probeCounts` (on), `f16Kv` (off), and for `gemma4-qat/26b` only `expertCacheSlots` (a count, a comma list or `fitted`), `hostTier` (GiB or `all`) and `depth` (0 = off, up to 4), `source` (`cached` or `direct`). The verdict is token identity; the judgement is in the summary. |
| `geometry-sweep` | The GEMM tile geometries measured on this host's GPU with `runGeometrySweep` of `@karume/runtime/tune`, one call per case with idle time before every case but the first (cooling: heat moves the default geometry's re-measurement, which the derivation checks), merged into one `karume-geometry-sweep/3` record. The record's `data` is that record, one input of `deriveGeometryProfile` as it is. Parameters: `candidateSet` (`quick`, `quick+` — the default —, `full`, or the evaluation set `k88`, which sweeps a case table of its own and from whose record no table is derived; `quick+` is `quick%2B` in a URL), `ops` (default: every op family), `cases` (default none: every case of the chosen op families in the chosen set; a case the chosen set does not have is an error, a case of an op family not chosen is not swept), `rounds` (5), `idleSeconds` (90). A cancel or a lost device keeps the cases that finished. `deriveGeometryProfile` rejects a record whose fingerprints (`caseSets`, `defaultKernels`, `candidateKernels`) differ from the deriving code's, so run it from the checkout that will derive the table and leave `packages/runtime` alone until then (the dev server reloads edited modules); keep the tab in front and nothing else on the GPU. Without `timestamp-query` the sweep times by the wall clock, and the derivation rejects such a record. |
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
to stderr; the outcome, the summary and the record's path go to stdout. Exit code: 0 when the run is
done and its verdict is pass, incomplete or none; 1 for fail, error or cancelled (Ctrl-C cancels); 2
for a command line that names no entry or a parameter that does not fit.

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
this adapter's table geometry against the same geometry with its fastest skeleton form, with the
conv2d counter decode and with both, and the census-weighted sums per unit. No table is derived from
this record.

```text
http://127.0.0.1:8791/?entry=geometry-sweep&candidateSet=k88&ops=linear,matmul,conv2d&rounds=5&idleSeconds=90
```

```sh
flock /tmp/karume-gpu.lock node apps/workbench/run.ts geometry-sweep --candidateSet=k88 --ops=linear,matmul,conv2d --rounds=5 --idleSeconds=90
```

## Records

A record lands at `outputs/workbench/<host>-<adapter name>/<entry id>_<UTC stamp>.json` (outside git):
`node-nvidia-geforce-rtx-3080-ti`, `browser-nvidia-geforce-rtx-5070-ti`. The adapter name is the
description in a browser and the device under Node (Dawn puts the driver version in the description),
else vendor-architecture. A second record in the same second gets `-1`, `-2`, …; no file is ever
replaced. The rules are in [docs/spec/workbench.md](../../docs/spec/workbench.md).

```text
format       "karume-workbench/1"
entry        { id, parameters }              the typed parameter values of the run
recordedAt   when the run ended (ISO 8601)
seconds      wall time of the entry's run
outcome      "done" | "error" | "cancelled"   (error: the stack in `error`)
environment  host ("browser" | "node"), userAgent, crossOriginIsolated (browser), backend
             ({ name, source: "declared" | "dawn-flag" | "dawn-default" }, when known), the adapter
             (vendor, architecture, device, description), its features and limits, WGSL language
             features — read from an adapter the host requests itself
checkout     { commit, dirty, runtimeVersion }   (dev server: at run start; build: at build time)
verdict      "pass" | "fail" | "incomplete"   (entries that conclude something)
summary      the lines the page shows first
data         the entry's own record body
```

## Add an entry

Add one file `src/entries/<id>.ts` (kebab-case) whose default export is an `Entry`
(`src/entry.ts`): `id` (the file name), `title`, a one- or two-sentence `description`, the declared
`parameters` (`boolean`, `integer` and `number` with min / max, `text`, `choice`, `choices`) and `run`. `run` gets the
typed parameters, `gpuOptions` to spread into `acquireGpu` (it acquires and destroys its own GPU
context), an `AbortSignal` and `progress(text)`, and resolves to `{ verdict?, summary, data }`. An
entry holds no UI code and no Node API: the shell and `run.ts` render and drive every entry the same
way. The registry (`src/entries.ts`) picks the file up; nothing else lists entries.

An entry that reads a local distribution (a mirror under `models/`) is a file
`src/entries/local/<id>.ts` that declares `readsLocalDistribution: true` and opens the mount with
`openDistribution(mount, source)` from its run context: a reference for `loadManifest` or a family's
`fromPretrained`, and the facts its record carries (`directory`, `manifestSha256`, `read`). In a
browser `source` picks the cached route (`/hf/`, the hub's CacheStorage) or the direct one
(`/models/<mount>/`); under Node the mirror's directory is read. The dev server serves the same
`/models/` and `/hf/` routes as the playground, from the shared mount table of
[`@karume/local-distributions`](../../packages/local-distributions/README.md). A production build
leaves these entries out.

## Develop

```sh
pnpm --filter @karume/workbench test        # vp test: parameters, registry, record, write-back, the entries' pure helpers (no GPU)
pnpm --filter @karume/workbench typecheck
pnpm --filter @karume/workbench build
```

# The workbench

What `apps/workbench` does now: the browser app (and the Node command) that runs the repository's checks and
measurements on a GPU and leaves one JSON record per run. It is the app ADR
[0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md) decision 9 calls "the harness"; the
maintainer named it the workbench, because "harness" already names the test harness (`packages/test-support`) and the
sweep's measurement core (`packages/runtime/src/tune/harness.ts`).

The workbench is private (not published) and is not part of the lanes: its tests run with
`pnpm --filter @karume/workbench test` and need no GPU.

## An entry

- One file `apps/workbench/src/entries/<id>.ts` is one entry (`src/entries/local/<id>.ts` for an entry that reads a
  local distribution). Its default export declares an `id` (kebab-case, equal
  to the file name), a `title`, a `description`, its `parameters` and `run`. Nothing else lists the entries: the
  registry collects the files, and an id that is not its file name is an error at start.
- An entry holds no UI code. The browser shell and the Node command render and drive every entry the same way, and the
  same module runs in both.
- Parameters are declared, of six kinds: `boolean`, `integer` and `number` (each with a minimum and a maximum; a
  number is finite), `text` (any string — the entry gives it its meaning when it runs), `choice` (one of a list) and
  `choices` (a subset of a list, written as a comma-separated list). Values arrive as strings — a URL query,
  `--name=value` on the command line, the form — and one function turns them into typed values. A missing name takes
  its default; a name the entry does not declare and a value that does not fit are errors that name the parameter.
- `run` receives the typed parameters, the GPU options of the host (`{}` in a browser, Dawn's `GPU` under Node), an
  abort signal, a function that reports one status line, a function that opens a local distribution and the
  backend's name as the host knows it (what the record's `environment.backend` holds, absent when a browser run
  declared none). An entry acquires and releases its own GPU context.
- `run` resolves to an optional `verdict`, a `summary` (the lines a person reads first) and `data` (the entry's own
  record body). A check concludes `pass` or `fail`; `incomplete` says the run did not cover what was asked (a check
  the device cannot run, a sweep that a device loss or an error stopped). A sweep that finished concludes `pass` or
  `fail` too: every candidate's output is checked — against its default's words, or, for a GEMV candidate that sums in
  another order, against the band of `linear`. An entry that only measures and finished has no verdict.

### Local distributions

- An entry that reads a model distribution is a file under `src/entries/local/`; the directory is the declaration.
  The dev server and the Node command list it; a production build neither lists nor bundles it (the registry's module
  of local entries resolves to an empty one in a build), because a production build serves no distribution.
- Such an entry opens a mount of the shared mount table (`@karume/local-distributions`) through the function the host
  gives it, naming how a browser should read it: `cached` (the dev server's HF-compatible `/hf/` route into the hub's
  CacheStorage, so a second run does not move the weights again) or `direct` (plain requests to `/models/<mount>/`).
  Under Node the mirror's directory is read whichever is named, and no other mount's mirror is read but those of the
  repositories the manifest references, so a broken mirror of an unrelated mount does not stop the run. A repository
  the manifest references is read from the mount that holds it.
- What it gets back is the reference the hub and a family's `fromPretrained` take, and the facts a record carries:
  the mount's directory, the sha256 of its manifest and how it was read (`cached`, `direct` or `directory`).
- A mount whose mirror is missing, and a mount without an HF-compatible route asked for `cached`, are errors that
  name the mount.
- The cache of `cached` belongs to the page's origin: the workbench does not see what the Playground cached.

### The entries

| Entry | What it runs | Inputs |
| --- | --- | --- |
| `kernel-check` | The twelve kernel identity checks of `@karume/kernel-checks` — the cases and criteria of the runtime's node GPU tests — on the host's backend. Parameter `checks`: which of the twelve (default: all). `pass` needs every check run and passed; a check the device cannot run is not available, never a pass. | Synthetic (built in memory); no distribution |
| `gemv-sweep` | `runGemvSweep` of `@karume/runtime/tune`, one call per planned case with idle time before every case but the first, the per-case records merged into one `karume-gemv-sweep/2` record (`data`, [below](#the-gemv-sweeps-record)) through the per-case loop the `geometry-sweep` entry uses. Per case (a shape of the default profile's GEMV rows) and row count: the baseline — what a Session at the level `exact` runs with the default profile —, the identical class (the sequential M = 1 variants, the row-block heights) and the reorder class (the parallel GEMV at each lane count, plain, staged and, where the device has the features, subgroup, at 1 to 8 rows), measured together: a warm start on the baseline, every kernel calibrated and checked, then rounds that each time every kernel once in a rotating order ([below](#how-the-gemv-sweep-times-a-row-count)). Parameters: `candidateSet` (`quick` — M = 1, the identical class on the 3 × 3 grid of columns and read-ahead around the baseline, the reorder class in full; sized for about half an hour on a desktop GPU —, or `full` — every row count and the whole grid), `cases` (none: every case), `rounds` (5), `idleSeconds` (10). The summary has a totals line, then per case and row count the baseline's time and the spread of its rounds, the fastest identical and the fastest reorder candidate with their speed against the baseline, and the default profile's choice at each level with its share of the fastest of its class's speed. The verdict, checked in this order: `fail` when a row of a finished case is in the identical class and its words differ from the baseline's, or in the reorder class and outside the band; `incomplete` when a device loss or an error stopped the run; `fail` when a row of a finished case errored; none when the run was cancelled; otherwise `pass`. Timing is not part of the verdict. | Synthetic; no distribution |
| `geometry-sweep` | `runGeometrySweep` of `@karume/runtime/tune`, one call per planned case with idle time before every case but the first, the per-case records merged into one `karume-geometry-sweep/3` record (`data`) — what `deriveGeometryProfile` reads. Parameters: `candidateSet` (`quick`, `quick+`, `full`, or the evaluation set `k88`), `ops`, `cases` (none: every case of the chosen op families in the chosen set; a case that is not in the chosen set is an error), `rounds`, `idleSeconds`. A cancelled run, a lost device and a sweep that threw keep the cases that finished. The verdict, checked in this order: `fail` when a row of a finished case (one that neither a cancel nor a device loss cut short) is not identical to its case's default; `incomplete` when a device loss or an error stopped the run short of its plan; `fail` when a row of a finished case errored; none when the run was cancelled; otherwise `pass`. Every candidate of every set is expected to match its default (f32 by the K reduction order, i8a8 by exact integer arithmetic); timing is not part of the verdict. | Synthetic; no distribution |
| `attention-bench` | `runAttentionRouteBench` of `@karume/runtime/tune` at Wan2.2 TI2V-5B's attention shapes: the two int8 attention routes with the kernels and row blocks production uses on the host, optionally the shared quantization and the tiled-key-axis candidate. `data` is the runtime's `karume-attention-bench/2` record. Parameters: `shapes`, `rowsPerDispatch` (0: production's split), `rounds`, `quantization`, `onlineTiled`. Without `timestamp-query` it times with the wall clock. | Synthetic; no distribution |
| `session-ab` (local) | The wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, judged by `judgeSessionAb`) of one component forward with and without the fusion rule its pipeline passes: the MOSS-Image DiT (`models/ew_chain`), the Wan2.2 VAE next chunk or the anima VAE tile (`models/vae_rms_silu`), on seeded feeds, every visit opening its own Session. Parameters: `target`, `preset` (empty: the manifest's default), `rounds`, `minGain`, `source`. The verdict is identity — every arm's outputs equal the base's first run, word for word; the judgement (adopt, reject, inconclusive) is in the summary and the data. | The mount of the target |
| `gemma4-session-ab` (local) | The same A/B on a Gemma 4 decode: one greedy generation of a fixed prompt per run, the rule arm with the candidate rule `candidate/gelu_mul`; an optional count probe before the A/B. Parameters: `model` (`<mount>/<model>`; empty: the default model of the first Gemma 4 mount whose mirror is present), `preset`, `tokens`, `rounds`, `minGain`, `openPerVisit`, `probeCounts`, `f16Kv`, `subgroups` (whether the device is acquired with the subgroup features when it has them), the expert cache of the 26B model (`expertCacheSlots`, `hostTier`, `depth`), `source`. The verdict is token identity. | A Gemma 4 mount |
| `gemma4-bench` (local) | A decode A/B of Gemma 4 arms — a model × a preset × the GPU tap on, split per dispatch or off × the integer GEMV × the PLE on the GPU × the decode run-ahead depth — visited in A B B A rounds over two fixed prompts, one pipeline resident at a time, `chats` fresh chats per prompt and visit (the first is the cold one and is left out of the summary). It reports the decode tokens per second and ms per token and, through a tap around the host's GPU, the GPU time, dispatches, passes and submits per token. Parameters: `arms` (comma-separated, arm 1 the base: `<mount>/<model>[@<preset>][:<switch>]…` with the switches `tap-off`, `split`, `integer`, `ple-gpu`, `ahead2` / `ahead4` / `ahead8`; empty: one arm per Gemma 4 mount whose mirror is present), `chats`, `newTokens`, `rounds`, `sampler`, `prompts`, `subgroups`, `source`. No verdict; a run that a failure stopped after a chat keeps what finished and is `incomplete`. | The Gemma 4 mounts of its arms |

### How the GEMV sweep times a row count

The kernels of one case at one row count are measured together, on the case's buffers (made once per case, at its
largest row count, with the host's f64 oracle):

1. Every kernel's pipeline is made first, so that no compile leaves the device idle between the steps below.
2. A warm start on the baseline: passes of its dispatch, each kept near the 80 ms target, back to back until three
   consecutive passes agree within 2 % (largest ÷ smallest), spending at least 1 s and at most 5 s of load, and at
   most 125 passes (a dispatch too short to fill a pass at the bound of 16,384 dispatches would otherwise take far
   longer than its load). The case is not measured on a device that is still on its idle clocks.
3. Every kernel, the baseline first, runs once from a zeroed output and is calibrated: the dispatches per pass that
   fill the target, from the wall clock of passes of growing length. A calibration whose pass comes out below a
   quarter of the target (with fewer dispatches than the bound of 16,384) is done again after another warm start, up
   to three calibrations. The kernel's output is checked right after (its words against the baseline's, or its band
   against the oracle), while the buffer still holds it.
4. The rounds: each times every kernel once, the baseline included, in plan order rotated so that round r starts at
   kernel ⌊r · kernels / rounds⌋ (at kernel r mod kernels when there are more rounds than kernels). A kernel's time is
   the min of its rounds ÷ its dispatches per pass.

A kernel that fails a step gets a failure row and leaves the steps after it; a lost device makes every kernel without
all its rounds a failure row. Interleaving puts every kernel under the
device states the row count went through: a device that heats during the rounds slows every kernel alike, instead of
whichever was measured last. How far the state moved is the spread of the baseline's rounds. The rows of a row count
are final when its rounds are done; a cancel during a row count leaves no row of it.

### The GEMV sweep's record

`karume-gemv-sweep/2`, the `data` of a `gemv-sweep` run (and what `runGemvSweep` returns). Nothing reads the records
of `karume-gemv-sweep/1`, whose rounds were taken kernel after kernel:

| Field | Content |
| --- | --- |
| `format`, `startedAt`, `date`, `aborted` | The format; when the sweep started and when its record was assembled; whether a cancel cut it short |
| `caseTable`, `kernels` | Fingerprints, built as the geometry sweep's: of the case table's dispatch-deciding fields, and of the kernels the rows without an error measured (a merged record recomputes it over its rows) |
| `userAgent`, `adapter` | The host and the four adapter fields |
| `gates` | The device facts that decided which candidates exist: the subgroup features (`subgroups`, `subgroup-size-control`, the WGSL `subgroup_id`; the subgroup form needs all three) and `maxComputeWorkgroupsPerDimension` (a candidate whose grid exceeds it, such as 32 lanes on a 262144-column head, is not measured) |
| `gpuTiming` | Whether `timestamp-query` was enabled, the unit (`ns` or `wall`), whether every round looks quantized |
| `settings` | The candidate set, the cases when narrowed, the rounds and the timing protocol's constants (`targetPassMs`, `maxReps`, the warm start's `warmStart` — `minNs`, `maxNs`, `maxPasses`, `stablePasses`, `tolerance` —, `recalibrateBelow` and `calibrationAttempts`), the band (`tolerance`, the band of `linear`) and `notMeasured` (the packed-activation form: it reads the quantizing producer's codes, which the sweep does not run) |
| `cases` | The planned cases: id, storage, i4 group, `n`, `k`, the row counts measured and where the shape comes from (the model and the role of the linear) |
| `summaries` | Per case and row count: the default profile's choice at `exact` and at `reorder` (by candidate name), the wall time, and how steady the device was over the rounds: the baseline's largest round ÷ its smallest (`baselineSpread`; rounds clamped to 0 left out, 1 with one round), absent when the baseline's row failed |
| `rows` | One per measured kernel: the case, the row count `m`, the candidate's name, its class, whether it is the baseline, its choice (the member of the GEMV family, lanes, column and row form), the pipeline keys, the workgroups, the time per dispatch (`perDispatch`) with its `rounds` (in round order, so round i of two rows was taken in the same round) and wall-clock rounds, the speed against the baseline; for the identical class whether every word equals the baseline's and how many differ; for the reorder class — and for the baseline at a row count with reorder rows — the largest absolute error against the host's f64 oracle, its use of the band, whether it is inside and the non-finite outputs; an `error` when it failed |
| `deviceLost` | The loss that ended the sweep, or `null` |

## The record

Every run leaves one record, format `karume-workbench/1`:

| Field | Content |
| --- | --- |
| `format` | `karume-workbench/1` |
| `entry` | `id` and the typed `parameters` the run used |
| `batch` | Only in the record of a run that was a step of a [batch](#batches): the batch's `id`, `startedAt` (when the batch run started, ISO 8601 — the same in every record of that batch run), `step` (the step's 1-based number in the file) and `steps` (the number of steps in the file). The record of a single run has no such key |
| `recordedAt`, `seconds` | When the run ended (ISO 8601) and the wall time of `run` |
| `outcome` | `done`; `error` (`run` threw: `error` holds the stack or message); `cancelled` (the signal was aborted before `run` settled — what the entry returned by then is kept) |
| `environment` | `host` (`browser` or `node`), `userAgent`, `crossOriginIsolated` (browser only), `backend` (optional), `adapter` (vendor, architecture, device, description), the adapter's `features` (sorted), every numeric adapter `limits` value and `wgslLanguageFeatures` (sorted) |
| `checkout` | `commit`, `dirty` and `runtimeVersion` (the version of `packages/runtime/package.json`) |
| `verdict`, `summary`, `data` | What the entry returned |

- The environment is read from an adapter the host requests itself, not from the entry.
- WebGPU does not tell the backend. In a browser a person may declare it (the shell's field or `backend=` in the
  URL; `source: "declared"`), and an undeclared backend leaves the field out. Under Node it is the `backend=<name>`
  token of the Dawn flags (`source: "dawn-flag"`), else Dawn's default for the platform (`source: "dawn-default"`).
- The checkout is read when the run starts: by the dev server at that moment for a browser run, from git for a Node
  run. A production build carries the checkout its build read.
- The saved text is the record as JSON with one-space indentation and a final newline. The download, the copy, the
  dev server's file and the Node command's file are that same text.

## Where records go

- `outputs/workbench/<host>-<adapter name>/<entry id>_<UTC time>.json`, outside git. `<host>` is `browser` or `node`.
  The adapter name follows the rule of the tests' environment key: the adapter's `description` in a browser and its
  `device` under Node (Dawn puts the driver version in the description, and the directory does not change with a
  driver update), in lower case with every run of characters other than `a`–`z` and `0`–`9` as one hyphen (at most 80
  characters); an empty name falls back to vendor and architecture, then to `unknown-adapter`. Examples:
  `node-nvidia-geforce-rtx-3080-ti`, `browser-nvidia-geforce-rtx-5070-ti`.
- The time is the writer's clock (`YYYY-MM-DDTHH-MM-SS`); a taken name gets `-1`, `-2`, …, and an existing file is never
  replaced.
- A browser run under the dev server posts its record, and the dev server writes it; the page says where. A machine
  that reaches the dev server through an SSH tunnel therefore leaves its records on the development machine.
- A record the dev server does not take — it cannot be reached, it answers anything but `201` with a path, or its
  answer cannot be read — is not saved: the run's outcome stands, the page says why the record was not saved and offers
  the download, and a batch goes on with its next step.
- A production build writes nothing: the page offers the download and says the record is not saved.
- A cancelled run and a failed run leave a record like any other.

## The dev server

`vp dev` in `apps/workbench` listens on `127.0.0.1:8791` only. Besides the endpoints below it serves the local
distributions as the Playground's dev server does (`/models/index.json`, `/models/<mount>/…` and `/hf/…`, read only).

| Request | Answer |
| --- | --- |
| `POST /workbench/results` | Writes the body as a record and answers `201` with `{ "path": … }` (repository-relative). `405` for another method; `415` unless `Content-Type` is `application/json`; `403` unless the `Origin` header is the server's own origin; `413` for a body over 64 MiB (not read past the limit); `422` unless the body is JSON with the format above, a kebab-case `entry.id`, an `environment.host` of `browser` or `node` and an `environment.adapter` object. The bytes written are the body, unchanged. |
| `GET /workbench/checkout` | The checkout at the moment of the request. `405` for another method. |
| `GET /workbench/batches` | The untracked batches, read at the moment of the request: `{ "batches": [{ "id": …, "text": … }] }` for every `*.json` file directly in `inputs/workbench/batches/` (the id is the file name without `.json`), in file-name order; a missing directory is an empty list. A file over 1 MiB (not read past the cap) or not UTF-8 is listed with `"problem"` in place of `"text"`. `Cache-Control: no-store`. `405` for another method. |

- Nothing the client sends becomes part of a path except through the two rules above (the kebab-case entry id and the
  adapter directory); the request for the batches names no path.
- The preview server and a production build have none of these endpoints. The preview server serves the distributions.
- Every response of the dev and preview servers carries `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`, so the page is cross-origin isolated.

## The shell

- A list of the entries, then the list of the [batches](#batches) in a colour of their own (a tinted background and a
  coloured left border); an untracked batch carries the mark `local`, a batch with problems the mark `has problems`.
- The selected entry's panel: its parameter form, generated from its declaration; an optional backend field; Run and
  Cancel; one status line.
- After a run: the verdict (or the outcome of a run that did not finish), the summary, where the record was saved (or
  why it was not), the buttons "Download record" and "Copy record", and the record's text (its first 200,000
  characters).
- The selected batch's panel: its title, description and source (tracked or untracked). A batch with problems shows
  them, and no Run. Otherwise a table of the steps — a checkbox (all checked at first; not in the URL), the step's
  number, the entry id, the parameters the file writes (`name=value`; not the defaults), a link that opens the step in
  the entry's form (`?entry=<id>&<name>=<value>…` from the step's typed values, the declared backend carried over) and
  the step's state — then the backend field, Run and Cancel, and one status line.
- While a batch runs, the status line is the running step's progress prefixed with its number and the file's step
  count (`step 3 of 9 · …`) and its row is marked. A finished step's row shows the outcome and the verdict, the seconds,
  the first summary line (the first line of the error for `error`) and where the record was saved or why not; the
  result view of a single run (above) opens under the row. At the end the status line sums the run up as the Node
  command's last line does, after the reason when a failure around a step's run stopped the batch.
- The selected entry and its parameter values, or the selected batch (`?batch=<id>`), and the declared backend are in
  the URL; an entry and a batch in one URL is an error the panel shows. Selecting from the menu writes the URL. Nothing
  runs without a click on Run. One run — of an entry or of a batch — at a time; the menu is off while it runs.
- Without WebGPU or without an adapter the page says so in place of Run.

## The Node command

`node apps/workbench/run.ts <entry id> [--name=value …]` runs an entry on Dawn and writes its record through the same
path rule; `node apps/workbench/run.ts --batch=<path>` runs a batch ([under Node](#under-node)); `--list` prints the
entries. The command's body and the entries load through one module runner, so both sides hold the same instance of
every shared module. It uses the GPU, so it runs under the GPU lock (`flock /tmp/karume-gpu.lock …`).

| Exit code | When |
| --- | --- |
| 0 | The run is done and its verdict is `pass`, `incomplete` or none (`--list` is 0 too); for a batch, every step's run is so |
| 1 | The verdict is `fail`, `run` threw, the run was cancelled (SIGINT), or the command failed around the run (no adapter, a record that cannot be written); for a batch, a step ended as `error` or `cancelled`, concluded `fail`, failed around its run, or did not run |
| 2 | The command line names no entry, an unknown entry, an argument that is not `--name=value`, or a parameter the entry does not declare or whose value does not fit its declaration; `--batch` together with an entry id, a `--name=value` or a second `--batch`, a path whose file name is not `<id>.json`, a file that cannot be read, or a batch with problems |

- A value that fits its declaration but that the entry refuses when it runs (the `arms` text of `gemma4-bench`, for
  example) is not a command-line error: `run` throws, the record's outcome is `error`, the record is written, and the
  exit code is 1.

Progress goes to stderr; the outcome, the summary and the record's path go to stdout.

## Batches

A batch is a JSON file that lists steps — an entry id and parameter values each — which the shell and the Node command
run in order on one click or one command.

### The batch file

```json
{
  "format": "karume-workbench-batch/1",
  "title": "Session A/B on three targets",
  "description": "optional, one paragraph",
  "steps": [
    { "id": "session-ab" },
    { "id": "session-ab", "parameters": { "source": "direct" } },
    { "id": "gemma4-bench", "parameters": { "arms": "gemma4-qat/e2b:split", "chats": 2, "prompts": ["few-ids"] } }
  ]
}
```

- `format` (required, exactly `karume-workbench-batch/1`), `title` (required, a string that is not blank),
  `description` (optional string), `steps` (required, at least one).
- A step is `{ id, parameters? }` — the shape of a record's `entry`, so the `entry` of a saved record is a step. `id`
  names a registered entry; `parameters` maps parameter names to a JSON boolean, number, string or list of strings.
- A parameter a step does not write takes the entry's default. A value is read in its string form (a list joined with
  commas) by the parser of the URL and the command line, so `3` and `"3"` are the same for an `integer` parameter.
- A key the format does not name, at the top or in a step, is a problem. A batch is a list: no expressions, conditions,
  loops, repeat counts, pauses, steps that expand into several, or expected results. It carries no backend: that is
  the shell's field and `backend=` in the URL, and Dawn's flags under Node.
- A batch's id is its file name without `.json`, kebab-case as an entry id is.
- Validation is whole and comes before anything runs: the JSON, the shape, and every step's entry and values. Every
  problem is reported, and a problem of a step names the step by its 1-based number
  (`step 3 (session-ab): Parameter rounds: …`). A batch with a problem runs nothing.

### Where batches come from

| Source | Where | How it reaches the page |
| --- | --- | --- |
| Tracked | `apps/workbench/batches/<id>.json` | Bundled, in the dev server and in a production build |
| Untracked | `inputs/workbench/batches/<id>.json` under the repository root (outside git) | `GET /workbench/batches` of the dev server, asked once when the page starts |
| Node | any path | `node apps/workbench/run.ts --batch=<path>` |

- The dev server lists every batch, one with problems with its problems: a step naming an entry the registry does not
  hold, a file name that is not kebab-case, and an untracked batch with a tracked batch's id (which is not runnable).
- A production build asks for no untracked batch and lists only the tracked batches whose every step names an entry it
  holds: it holds no entry that reads a local distribution, and so no batch that needs one.
- The page has no file picker and no paste box: the menu is the way in.
- `synthetic-quick` is the tracked quick look at three of the synthetic entries: `kernel-check` with its defaults,
  `geometry-sweep` at the `quick` set with one round and no idle time, `attention-bench` with its defaults. It is not
  the evidence run. `gemv-sweep` is not a step of it: its `quick` set measures about 1,600 kernels (about 1,250 without the
  subgroup features), each with its own calibration after a warm start per case, which is no quick look even at one
  round.

### A batch run

- The checked steps run in file order, each as one ordinary run: the environment and the checkout are read when the
  step starts, and its record is saved where and how a single run's record is, with the `batch` field
  ([the record](#the-record)). No record is made for the batch as a whole.
- A step that ends as `error` or with verdict `fail` does not stop the batch: the next checked step runs.
- Cancel (SIGINT under Node) cancels the running step — its record is `cancelled` and saved — and the remaining steps do
  not run.
- A step whose run fails around the entry — the environment or the checkout cannot be read, or (under Node) its record
  cannot be written — stops the batch with the reason: that failure is the host's, not the step's.

### Under Node

- `node apps/workbench/run.ts --batch=<path>` reads the file (a relative path against the current directory) and runs
  every step. `--batch` with an entry id or a `--name=value`, a file name that is not `<id>.json`, a file that cannot be
  read and a batch with problems are usage errors (exit 2): every problem goes to stderr and nothing runs.
- Each step prints what a single run prints, its first line prefixed with its position (`[3/9] session-ab: done,
  verdict pass (6.2 s)`); its progress goes to stderr prefixed the same way. A failure around a step's run goes to stderr
  with the same prefix.
- The last line sums the run up, counting what happened and what did not run: `batch <id>: 9 steps — 7 done (1 with
  verdict fail), 1 error, 1 did not run`.
- The first SIGINT cancels the running step, and the rest do not run. The exit codes are in
  [the Node command](#the-node-command).

## What it does not do

- It collects nobody else's results: a record stays where it was written.
- It downloads no model: an entry reads a local mirror or nothing.
- It keeps no history and compares no two records.
- A batch decides nothing from a result: it is a list of runs, and it leaves no record of its own.

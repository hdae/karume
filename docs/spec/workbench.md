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
  abort signal, a function that reports one status line, a function that opens a local distribution, a function that
  opens a [file under `inputs/`](#input-files) and the backend's name as the host knows it (what the record's `environment.backend` holds, absent when a browser run
  declared none). An entry acquires and releases its own GPU context.
- `run` resolves to an optional `verdict`, a `summary` (the lines a person reads first) and `data` (the entry's own
  record body). A check concludes `pass` or `fail`; `incomplete` says the run did not cover what was asked (a check
  the device cannot run, a sweep that a device loss or an error stopped). A sweep that finished concludes `pass` or
  `fail` too: every candidate it times is checked — against its default's words, or, for a GEMV candidate that sums in
  another order, against the band of `linear`; one its quick look screened out is not. An entry that only measures and
  finished has no verdict.

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

### Input files

- A local entry may read a file under the repository's `inputs/` (outside git — a user's weights file) through the
  function the host gives it, naming the file by its path relative to `inputs/`: `/`-separated, with no empty, `.` or
  `..` segment, no backslash and no NUL (refused naming the path before anything is read).
- What it gets back is the file as a ranged reader (`size`, `read(offset, length)`) and the facts a record carries:
  its repository path (`inputs/…`), its size and how it was read.
- Under Node the file is read from disk (`directory`), one positional read per range, the file opened for each read;
  its real path must stay inside `inputs/`, which may itself be a link. In a browser it is read by ranged requests to the
  dev server's `/inputs/` route (`direct`): its size from a `HEAD`, each range a `GET` that must answer `206` with
  exactly the bytes asked for. A production build serves no input file.

### The entries

| Entry | What it runs | Inputs |
| --- | --- | --- |
| `kernel-check` | The fourteen kernel identity checks of `@karume/kernel-checks` — the cases and criteria of the runtime's node GPU tests — on the host's backend. Parameter `checks`: which of the fourteen (default: all). `pass` needs every check run and passed; a check the device cannot run is not available, never a pass. | Synthetic (built in memory); no distribution |
| `gemv-sweep` | `runGemvSweep` of `@karume/runtime/tune`, one call per planned case with idle time before every case but the first, the per-case records merged into one `karume-gemv-sweep/4` record (`data`, [below](#the-gemv-sweeps-record)) through the per-case loop the `geometry-sweep` entry uses. Per case (a shape of the case table: the default profile's GEMV rows, the other integer-storage linears a decode token or a draft step of the Gemma 4 distributions runs at one row — the int2 and int4 output heads, the MTP drafters —, or a linear of regular Gemma 4 E2B / E4B in bf16) and row count: the baseline — what a Session at the level `exact` runs with the default profile —, the identical class (the sequential M = 1 variants, the row-block heights) and the reorder class (the parallel GEMV at each lane count, plain, staged and, where the device has the features, subgroup, at 1 to 8 rows, and at 4 and 8 rows its row-blocked form `staged-l<lanes>r<rows>` at each height of at most that many rows the kernel exists at), measured together; a bf16 case is measured at 1, 4 and 8 rows, its baseline past one row is the GEMM (`gemm`: such a weight has no sequential GEMV there, and no row-block height or subgroup candidate): a warm start on the baseline, a quick look at every kernel, the kernels that go on calibrated and checked, then rounds that each time every one of them once in a rotating order; at 4 and 8 rows only the reorder candidates that the case's M = 1 group of the same run leaves adoptable are measured, the others are skipped rows ([below](#how-the-gemv-sweep-times-a-row-count)). Parameters: `candidateSet` (`quick` — M = 1, the identical class on the 3 × 3 grid of columns and read-ahead around the baseline, the reorder class in full —, `full` — M = 1, 4 and 8, every row count a derivation rule reads, and the whole grid —, `row-blocks` — M = 16 and 64, the row-block heights, which no rule reads yet — or `explore` — the row counts of `full` in a short run: the [sentinels](#the-explore-set) of the case table with every candidate, first, and every other case with the candidates they predict), `cases` (none: every case of the table — the 91 integer-storage shapes, which are those of the default profile's rows and the output heads and MTP drafter shapes that have none, and the 25 bf16 linears of regular Gemma 4 E2B / E4B, 116 in all), `rows` (row counts of the set; none: all of them), `candidates` (names; none: every candidate — the references are measured either way; refused with `explore`, which chooses its own), `quickLookMargin` (empty: the set's default, ×1.25 and ×1.10 for `explore`; 0 times every kernel in full; a value between 0 and 1, or one that is no number, is refused), `rounds` (5), `idleSeconds` (10). With `explore` the per-case loop measures the sentinels among the planned cases first (each part in the table's order) and hands every call the sentinels the calls before it measured, so the merged record is what one `runGemvSweep` call over the same cases returns. The summary has a totals line (with the kernels the quick look screened out, and the kernels skipped as not adoptable from M = 1 when there are any — `not carried from M = 1` for `explore`), then per case and row count the baseline's time and the spread of its rounds, the fastest identical and the fastest reorder candidate with their speed against the baseline, the default profile's choice at each level with its share of the fastest of its class's speed, for an `explore` group narrowed to its prediction how many candidates it was narrowed to (`· narrowed to 9 predicted candidates`), and the kernels screened and skipped (`· 12 screened · 640 skipped (not adoptable from M = 1)`). The verdict, checked in this order: `fail` when a row of a finished case is in the identical class and its words differ from the baseline's, or in the reorder class and outside the band; `incomplete` when a device loss or an error stopped the run; `fail` when a row of a finished case errored; none when the run was cancelled; otherwise `pass`. A screened or skipped row has no check, so it is never wrong. Timing is not part of the verdict. | Synthetic; no distribution |
| `geometry-sweep` | `runGeometrySweep` of `@karume/runtime/tune`, one call per planned case with idle time before every case but the first, the per-case records merged into one `karume-geometry-sweep/4` record (`data`) — what `deriveGeometryProfile` reads. Each case after a quick look: only the default geometry and the candidates whose quick time is within the margin of the default's are timed in full; the others are screened rows. Parameters: `candidateSet` (`quick`, `quick+`, `full`, `explore` — `full`'s candidates on its cases in fewer rows: the cases class by class, a candidate that an earlier case of its class screened out, failed, found not identical or timed slower than the default skipped in the later ones ([below](#the-geometry-sweeps-explore-set)) —, or the evaluation set `k88` — whose summary compares every candidate with the table's geometry: run it at margin 0; in `quick+` and `full` a linear / matmul / bmm case measures every tile also with each skeleton form and a conv2d case also with the counter decode and with both, [geometry-profiles.md](geometry-profiles.md#the-skeleton-form-and-the-counter-decode)), `ops`, `cases` (none: every case of the chosen op families in the chosen set; a case that is not in the chosen set is an error), `geometries` (comma-separated names of candidate geometries; none: every candidate of the set; the default geometry is measured either way, and a name that is not a candidate of every chosen case makes the run fail before the GPU is touched; refused with `explore`, which chooses what it measures), `quickLookMargin` (×1.25; 0 times every geometry in full; a value between 0 and 1 is refused), `rounds`, `idleSeconds`. With `explore` the per-case loop measures the planned cases class by class and hands every call the rows the calls before it measured, so the merged record is what one `runGeometrySweep` call over the same cases returns; the summary's totals line then counts the rows skipped as dropped for their class (`· 412 skipped as dropped for their class`). A cancelled run, a lost device and a sweep that threw keep the cases that finished. The verdict, checked in this order: `fail` when a row of a finished case (one that neither a cancel nor a device loss cut short) is not identical to its case's default; `incomplete` when a device loss or an error stopped the run short of its plan; `fail` when a row of a finished case errored; none when the run was cancelled; otherwise `pass`. Every candidate that is timed is expected to match its default (f32 by the K reduction order, i8a8 by exact integer arithmetic); a screened or skipped candidate has no output checked; timing is not part of the verdict. | Synthetic; no distribution |
| `attention-bench` | `runAttentionRouteBench` of `@karume/runtime/tune` at Wan2.2 TI2V-5B's attention shapes: the two int8 attention routes with the kernels and row blocks production uses on the host, optionally the shared quantization and the tiled-key-axis candidate. `data` is the runtime's `karume-attention-bench/2` record. Parameters: `shapes`, `rowsPerDispatch` (0: production's split), `rounds`, `quantization`, `onlineTiled`. Without `timestamp-query` it times with the wall clock. | Synthetic; no distribution |
| `session-ab` (local) | The wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, judged by `judgeSessionAb`) of one component forward with and without the fusion rule its pipeline passes: the MOSS-Image DiT (`models/ew_chain`), the Wan2.2 VAE next chunk or the anima VAE tile (`models/vae_rms_silu`), on seeded feeds, every visit opening its own Session. Parameters: `target`, `preset` (empty: the manifest's default), `rounds`, `minGain`, `source`. The verdict is identity — every arm's outputs equal the base's first run, word for word; the judgement (adopt, reject, inconclusive) is in the summary and the data. | The mount of the target |
| `gemma4-session-ab` (local) | The same A/B on a Gemma 4 decode: one greedy generation of a fixed prompt per run, the rule arm with the candidate rule `candidate/gelu_mul`; an optional count probe before the A/B. Parameters: `model` (`<mount>/<model>`; empty: the default model of the first Gemma 4 mount whose mirror is present), `preset`, `tokens`, `rounds`, `minGain`, `openPerVisit`, `probeCounts`, `f16Kv`, `subgroups` (whether the device is acquired with the subgroup features when it has them), the expert cache of the 26B model (`expertCacheSlots`, `hostTier`, `depth`), `source`. The verdict is token identity. | A Gemma 4 mount |
| `embeddinggemma-preset-ab` (local) | The same A/B on two presets of the EmbeddingGemma 2 mirror (mount `embeddinggemma-2`), through the public pipeline: each arm is `embeddinggemma.Pipeline.fromPretrained` of one preset of the manifest's default model on one shared device (acquired as the pipeline acquires its own: no feature requested), opened once and resident for the whole A/B, and a run is one `embed` of one text, without a signal or the diagnostics callback; the timed span is the whole call, the tokenization of the text on the host and the readback of the vector included. The text is a prefix of a golden case of the recipe (`packages/models/tests/fixtures/embeddinggemma2/golden-cases.json` — the shortest case of at least the asked length, with its prompt), in whole code points, whose ids from `Pipeline.tokenize` are exactly the asked length: a short prefix found by bisection over the code points (it assumes the count mostly grows with the prefix) and a scan of at most 64 more code points, its length checked exactly; a length the scan does not hit is an error. Before the A/B each preset is probed by a pipeline of its own built with the diagnostics callback: its `fromPretrained` timed, the text embedded twice, the second run's diagnostics read (the runtime builds the plan backing on a run that finds its prepared plan), the pipeline disposed; the compared probe's ids must equal the base's. Parameters: `base` (empty: the manifest's default preset), `compared` (`i8`), `tokens` (512, 2 to the longest golden case), `rounds` (9, ≥ 3), `minGain` (0.01), `source` (`cached`). One run measures one length. No verdict; `judgeSessionAb` runs without identity (the presets' vectors differ by design), and each arm's report compares its vector with the arm's own first ([below](#the-embeddinggemma-preset-abs-record)). | The `embeddinggemma-2` mount |
| `anima-single-file` (local) | A single Anima weights file through `anima.Pipeline.fromSingleFile` ([single-file](single-file.md)) against the `anima` mount: the pipeline is built and makes two images of one request (512², the model's default guidance and negative prompt), then, when `compare` is on, `fromPretrained` of the same model and quant does the same; each pipeline is disposed before the next. Parameters: `file` (a path under `inputs/`; default the upstream `anima-base-v1.0` single file), `model` (`anima-v1.0`), `quant` (empty: the model's default), `textConditioner` (`file`: the file's when it holds one; `distribution`: `components: { text_conditioner }` of the mount), `retainConverted` (off: `retainConverted` of `fromSingleFile`, which keeps the converted weights in memory — [single-file](single-file.md#keeping-the-converted-weights)), `compare` (on), `steps` (4), `seed` (42), `source`. No verdict; `data` is a `karume-anima-single-file/1` record ([below](#the-anima-single-file-record)). | The `anima` mount and a file under `inputs/` |
| `anima-step-timing` (local) | The time of every denoise step of Anima generations per preset: for each preset, in the order of the list, a device of its own (`acquireGpu`, with `shaderF16` when the preset declares it) and `anima.Pipeline.fromPretrained` of the model at that preset on it, then `generations` images of one fixed request (a neutral prompt, the model's default steps, guidance and negative prompt, seeds 42, 43, …, the same for every preset) in the default per-stage residency, then the pipeline disposed and the device destroyed. Every `stage`, `denoise-step` and `residency` callback of `onEvent` is timed by the wall clock ([below](#the-anima-step-timing-record)); no image is kept. Parameters: `mount` (`anima-2.9b`, `anima` or `anima-extra`), `model` (empty: `anima-v1.0` — the 20-step base with guidance — on `anima`, whose manifest's default is a turbo model, and the manifest's default elsewhere), `presets` (default `bf16+dit8-a8` and `bf16+dit8-a8-attn8-s16`: one without and one with int8 attention), `generations` (3), `width` and `height` (1024), `source` (`cached`). Every preset is resolved against the manifest before the GPU is touched. No verdict; the summary says per preset its median step and how many steps stalled. | The mount named |
| `gemma4-bench` (local) | A decode A/B of Gemma 4 arms — a model × a preset × the GPU tap on, split per dispatch, in its host mode or off × linear at the level `reorder` × the GEMV choices of the device's profile, of the default profile, or without the sequential variant's per-shape exceptions × the integer GEMV × the PLE on the GPU × the decode run-ahead depth × the model's MTP drafter or not × a signal passed or not × the bench's sampler or the distribution's — visited in A B B A rounds over two fixed prompts, one pipeline resident at a time, `chats` fresh chats per prompt and visit (the first is the cold one and is left out of the summary). It reports the decode tokens per second and ms per token and, through a tap around the host's GPU, the GPU time, dispatches, passes and submits per token, or in the tap's host mode [the host's segments of a decode token](#the-decode-benchs-host-segments). Parameters: `arms` (comma-separated, arm 1 the base: `<mount>/<model>[@<preset>][:<switch>]…` with the switches `tap-off`, `split`, `host`, `linear-reorder` (the Session option `numerics: { ops: { linear: "reorder" } }` on a preset that does not declare linear at `reorder`; refused on one that does), `default-gemv` and `general-sequential` ([the profile switches](#the-decode-benchs-profile-switches)), `integer`, `ple-gpu`, `ahead2` / `ahead4` / `ahead8`, `drafter` ([the drafter arm](#the-decode-benchs-drafter-arm)), `no-signal`, `sampled`; empty: one arm per Gemma 4 mount whose mirror is present), `chats`, `newTokens`, `rounds`, `sampler`, `prompts`, `subgroups`, `source`. No verdict; a run that a failure stopped after a chat keeps what finished and is `incomplete`. | The Gemma 4 mounts of its arms |

### The decode bench's host segments

An arm of `gemma4-bench` with the switch `host` runs the tap in its host mode: the tap leaves the
compute passes as the pipeline encodes them (no timestamps, no count per dispatch) and records the
wall clock at the WebGPU calls that bound the host's work on a run — `queue.writeBuffer`,
`createCommandEncoder`, `queue.submit` and its return, and for a buffer created with `MAP_READ` its
`mapAsync`, the settlement of that map, `getMappedRange` and `unmap`. The chats of such an arm are
sent with a signal of their own that follows the run's and records when it is checked: the generation
loop checks its signal at the boundary before a run — at once when the event loop turned while the run
before it was awaited, after a yield to the event loop otherwise
([generation](generation.md#abort-checks)) — and again right before it issues the run. The switch does
not go with `tap-off` or `split`.

Each decode token's interval — from the previous token's `onToken` to its own, and from the last
delivered token to the end of the chat when a run drew a stop token — is cut into seven segments that
add up to it:

| Segment | From → to | What the host does in it |
| --- | --- | --- |
| `between` | the previous token → the signal's first check | the previous token's way to the consumer, and the loop's yield to the event loop where the boundary takes one (the run before it did not let the event loop turn) |
| `derive` | → the signal's last check before the first WebGPU call | the inputs the pipeline derives on the host (the per-layer-embedding rows) |
| `start` | → the return of the interval's first submit | preparing the run, the queue writes, encoding and submitting the first chunk of passes; the GPU has no work of this token until it ends |
| `encode` | → the return of the interval's last submit | encoding and submitting the remaining chunks while the GPU runs the ones before them |
| `wait` | → the settlement of the last read-back map | nothing: the GPU finishes the run |
| `read` | → the last `unmap` | the mapped ranges copied to the host |
| `draw` | → the token's `onToken` | the selection or sampling of the token, and its way to the consumer |

A chat sent without a signal (`no-signal`) has no check and no yield: `between` runs to the first
WebGPU call and `derive` is 0. An interval without a submit — a token handed out of a run-ahead batch
that was already read back — is all `between`. The time after the last delivered token of a chat that
ended at its token limit is in no interval.

A chat's measurement keeps, per segment, the mean over its decode tokens and the median over them, and
per decode token the submits, queue writes, maps and mapped bytes (`measurement.host`; it has no
`measurement.tap`). The summary spreads both over the warm chats (`summary[].host`) and ends with a
table per arm and prompt: the median and the minimum over the chats of the chat's mean, and the median
over the chats of the chat's per-token median. `wait` holds the GPU's time for the token; the GPU time
itself is what a tapped arm of the same run reports.

### The decode bench's drafter arm

An arm of `gemma4-bench` with the switch `drafter` decodes with its model's MTP drafter: its pipeline is built with
the drafter (`fromPretrained` with `speculative`, at the drafter's own draft count) and every chat is sent with
`speculative: "always"` — speculation in every cycle, without the self-tuning gate, so the arm times speculative
decode itself. An arm without the switch builds no drafter and passes no `speculative`: it decodes one run per token.
So two arms of one model and preset, with and without the switch, compare speculative decode with plain decode; under
greedy decoding their outputs are the same tokens.

The switch is refused, naming the arm, before any GPU work: on a model whose manifest entry declares no `drafter`
weights (it ships none), on the mobile QAT line (`gemma4QatMobile.Pipeline` takes no drafter), with `ple-gpu` and with
a run-ahead depth (the pipeline refuses both pairs), and with `host` ([the host segments](#the-decode-benchs-host-segments)
cut each interval between two tokens as one decode step, and the tokens of one cycle arrive together, so most intervals
hold no run and one holds the whole cycle). It goes with `split`: the breakdown divides the whole decode window by the
decode tokens, as the decode ms per token does, so it is the GPU time per token committed, the drafter's runs and the
verify runs together.

The arm's record says `drafter: true` (`data.arms[]`), and its chats' stops carry the speculation's counters
(`data.chats[].stop.speculation`: cycles, draft runs, drafts produced and accepted, tokens committed, the histogram of
accepted drafts). The summary sums them over the warm chats (`summary[].speculation`) and the arm's line ends with the
tokens per cycle (tokens committed ÷ cycles) and the drafts accepted. The arm's line leaves out the token interval: the
tokens of one cycle arrive together, so the interval between two tokens is not a decode step's time (the data keeps
it); the decode ms per token and its ratio to arm 1's are the figures to compare.

### The decode bench's profile switches

Two switches of a `gemma4-bench` arm change the geometry profile the arm's devices are acquired with, so that one
interleaved run compares what a profile's GEMV choices of linear
([geometry-profiles.md](geometry-profiles.md#the-gemv-choices-of-linear)) buy on the host's GPU, in a browser or
under Node:

- `default-gemv` — the table the adapter selects, without GEMV choices of its own: the device runs with the default
  profile's, as it would if its table carried none. On an adapter whose table carries none the arm runs what an arm
  without the switch runs.
- `general-sequential` — the GEMV choices without any per-shape exception of the sequential variant: every
  sequential GEMV of one row runs the general variant.
- Both on one arm apply in that order, whatever order the arm names them in: the default profile's choices without
  its exceptions.
- The tile geometry and every other field of the profile are the selected table's in every arm. The switches change
  which kernel a linear runs; a sequential variant writes the same words, and the parallel rows of a table and of the
  default profile can differ, so the outputs of two arms are compared as for any other pair of arms.

The geometry profile is a property of the device:

- A bench in which no arm carries a profile switch acquires one device and runs every visit on it.
- A bench in which an arm carries one acquires a device for every visit of every arm — with its arm's profile, or
  with none for an arm without a switch — and releases it with the visit's pipeline, so the arms are treated alike.
  Its figures are not comparable with those of a bench on one device.
- The device takes the profile through `acquireGpu({ geometryProfile })`. The bench builds it from the table the
  runtime selects for the adapter and from the default profile's choices, which it imports from the runtime's source
  (the runtime exports neither as a value).

Such a bench states, per arm, the choices its devices ran with: `data.arms[].gemvChoices` holds the arm's switches
in the order they applied, the id of the table the adapter selects (`table`), whether that table carries GEMV choices
of its own (`tableCarriesGemv`) and the choices with the switches applied (`linearGemv`); the summary has one line
per arm after the arms' lines (the table, whose choices, the sequential variant with the number of its exceptions,
the number of parallel rows). An arm none of whose visits began has no `gemvChoices`, and a bench without a profile
switch has none at all.

### How the GEMV sweep times a row count

The kernels of one case at one row count are measured together, on the case's buffers (made once per case, at its
largest row count, with the host's f64 oracle). A case's row counts are measured in ascending order, so at 4 and 8
rows — where the derivation only checks that a candidate M = 1 would adopt is not slower than the default's choice at
`reorder` — the run has the case's M = 1 group, when it measured it. There a reorder candidate that is not a reference
(the baseline or the default's choice at `reorder` of that row count) is measured only if its row of the same name at
M = 1 has a time (not screened by the quick look, not failed), held its check (inside the band, no non-finite output),
and — when the M = 1 group was steady (its baseline spread at most 1.25) and the default's choice at `reorder` has a
time there — is at least ×1.10 faster than that choice (gemv-candidates.ts's `gemvCarriedFromM1`, with the
derivation's threshold and the sweep's own steadiness bound, which the derivation does not read). Any other is skipped: it takes none of the steps below, and its row has
its kernel's fingerprint, its pipeline key and workgroups and `skipped: true`. Nothing is skipped in a run that has
no M = 1 group of the case to read (narrowed by `rows`, stopped before it, or its baseline was not timed there), for a
candidate without a row at M = 1, for the identical class or at any other row count.

A row-blocked candidate has no row at M = 1: it is the form a parallel row of its lanes takes at 2 rows and more. It
is measured when the default's choice at `reorder` at M = 1 has its lanes (the row a derivation keeps), or when the
plain or the staged candidate of its lanes is measured by the rule above; otherwise it is a skipped row. The
default's choice at `reorder` at M = 1 is measured at 4 and 8 rows whatever its own row says: where the default's
row carries a row block, the reference at those row counts is the row-blocked kernel, and the row's kernel of M = 1
is what shows whether the block pays on the device.

The `explore` set carries less ([below](#the-explore-set)): of the reorder candidates the rule above measures, only
the three fastest plain or staged rows of the case's M = 1 group (`gemvExploreCarriedFromM1`) and the row-blocked
forms of their lanes; the subgroup form is not carried. The references, the default's choice at `reorder` at M = 1
and the row-blocked forms of its lanes are measured as above. The others are skipped rows. Its quick look at 4 and 8
rows (step 3) does not screen out one of those three rows — unless it is the default's choice at `reorder` at M = 1 —
when a row-blocked form of its lanes goes on (`gemvExploreQuickLookGoesOn`): the derivation reads every row block of
a candidate against the candidate's own kernel at that row count, so without that kernel's time no row block has
evidence and the candidate is not adopted. A row-blocked form that is screened out itself keeps nothing; without the
run's M = 1 group of the case the margin alone decides.

1. Every kernel's pipeline is made first, so that no compile leaves the device idle between the steps below. The
   kernels are the baseline, the default profile's choice at `reorder` and the candidates of the set (those the
   parameter `candidates` names, when given; for an `explore` case that is no sentinel, at M = 1, those its
   sentinels predict).
2. A warm start on the baseline: passes of its dispatch, each kept near the 80 ms target, back to back until three
   consecutive passes agree within 2 % (largest ÷ smallest), spending at least 1 s and at most 5 s of load, and at
   most 125 passes (a dispatch too short to fill a pass at the bound of 16,384 dispatches would otherwise take far
   longer than its load). The case is not measured on a device that is still on its idle clocks.
3. The quick look (skipped at a margin of 0): the baseline runs once from a zeroed output, so its words are in its
   buffer; then every kernel, in an order shuffled from the case and row count (the same in every run, not the
   plan's), gets the calibration's probe — passes of growing length until one reaches half the target by the wall
   clock or the bound of 16,384 dispatches — with timestamps, a candidate's first pass from a zeroed output; the
   last pass ÷ its dispatches is its quick time. Right after its probe, while the buffer holds its output, the
   kernel's check is read by the comparison of step 4 (the baseline's holds; a candidate's does not when the
   baseline has no output). The references (the baseline and the default's choice at `reorder`) go on, and so does
   every kernel whose quick time is at most the margin times its class's bar: the fastest quick time of the class
   among the kernels whose check held at M = 1, where the derivation picks a winner, the subgroup form left out (a
   Session option no table row holds) — a kernel that can never be adopted does not screen out one that can; the
   quick time of the default's choice at `reorder` for the reorder class at 4 and 8 rows, where it only checks that
   a candidate is not slower than that choice (whatever that choice's own check); the fastest of the class among
   the kernels whose check held elsewhere, again without the subgroup form. A class in which no kernel that could be
   adopted has a quick time has no bar: all its kernels go on. A subgroup kernel is screened against the bar like
   any other. A kernel whose check failed goes on or
   is screened by its own time like any other, and its row records the check of step 4. The `explore` set also lets
   the base of a row block go on at 4 and 8 rows ([above](#how-the-gemv-sweep-times-a-row-count)). The others are
   screened rows. The rule and its evidence are measurement.ts's `QUICK_LOOK_MARGIN` (×1.25) and, for the `explore`
   set, `EXPLORE_QUICK_LOOK_MARGIN` (×1.10: replayed on the GEMV sweep records of 2026-11-18/19, it left the
   derivation's decisions unchanged on the RTX 5070 Ti, 0 of 116, and changed 2 of 214 on the Apple M2, each within
   2 %; ×1.05 lost adoptions on both).
4. Every kernel that goes on, the baseline first, runs once from a zeroed output and is calibrated from its quick
   look's probe (from a probe of its own with the quick look off): the dispatches per pass that fill the target, from
   the wall clock of passes of growing length. A calibration whose pass comes out below a quarter of the target (with
   fewer dispatches than the bound of 16,384) is done again after another warm start, up to three calibrations. The
   kernel's output is checked right after (its words against the baseline's, or its band against the oracle), while
   the buffer still holds it.
5. The rounds: each times every kernel that went on once, the baseline included, in plan order rotated so that round
   r starts at kernel ⌊r · kernels / rounds⌋ (at kernel r mod kernels when there are more rounds than kernels). A
   kernel's time is the min of its rounds ÷ its dispatches per pass.

A kernel that fails a step gets a failure row and leaves the steps after it; a lost device makes every kernel without
all its rounds a failure row (a screened kernel is no failure: its row has its quick time; a skipped kernel keeps its
skipped row). Interleaving puts every kernel under the
device states the row count went through: a device that heats during the rounds slows every kernel alike, instead of
whichever was measured last. How far the state moved is the spread of the baseline's rounds. The rows of a row count
are final when its rounds are done; a cancel during a row count leaves no row of it.

Besides one status line per step, `runGemvSweep` reports a `plan` event when a case starts — its place among the
planned cases and, per row count, the kernels planned (the baseline, the default's choice at `reorder` and the
candidates; at 4 and 8 rows those M = 1 does not carry are skipped, not prepared) — and a `step` event per unit of a
row count: its phase (`prepare`, `warm-start`, `quick-look`, `calibrate`, `rounds`), how many units are done and how
many there are — a kernel's pipeline, the warm start (one unit), a kernel's quick look and check, a kernel's
calibration and check, a kernel's pass in a round (rounds × the kernels timed). A phase reports 0 of its total when it
begins and each unit when it ends, a unit that failed included; the quick look reports nothing at a margin of 0.

### The explore set

A short run of the `full` set's row counts, for measuring a device in minutes (gemv-explore.ts):

- The **sentinels** are twelve cases of the whole case table (`gemvSentinels`; whatever the run is narrowed to, so
  every run and every device names the same): the case nearest to all, then again and again the case farthest from
  every one chosen (a tie goes to the earlier case of the table). Two cases are apart by the octaves between the
  words of their weight rows and between their column counts, plus 100 for another storage and 0.5 for another i4
  group length (`gemvCaseDistance`). The count is where the decisions of replayed records stopped improving fast
  (2026-11-19: 6, 8, 12 and 16 sentinels left 18, 14, 8 and 5 of 214 decisions different on the Apple M2 and 1, 0, 0
  and 0 of 116 on the RTX 5070 Ti).
- The planned sentinels are measured first, with every candidate, then the other cases; each part in the table's
  order.
- Every other case measures at M = 1, besides the references, the fastest sequential variant and the fastest plain
  or staged parallel row — each among the rows whose check held, timed (a finite time), not screened, skipped or
  failed — of the nearest sentinel of the case's storage that has one (the first of equally near ones; the i4 group
  length may differ), each with every other kernel of its kind at that sentinel, among the same rows, that the
  derivation could adopt there: at least ×1.10 faster than its class's reference there — the default's choice at
  `exact` (the baseline) for a sequential variant, at `reorder` for a plain or staged row; none when that reference
  has no time there —, and their neighbours on the candidate grid: one step of the columns per workgroup or of the
  words read ahead; one step of the lanes, and the other of plain and staged at the same lanes
  (`gemvExploreSelection`). A sentinel of another storage
  does not predict: the storage decides the decode. A name
  the case has no candidate of on the device selects nothing. When no sentinel of the case's storage has a winner of
  one of the two classes, the case measures every candidate. The
  sentinels read are the ones the call measured before the case and those given as `sentinelEvidence` (measured by
  earlier calls — the workbench makes one call per case).
- At 4 and 8 rows a case measures what its own M = 1 group carries ([above](#how-the-gemv-sweep-times-a-row-count)),
  and the identical class there is the baseline alone (no row-block height: no rule reads one).
- The quick look's default margin is ×1.10; at 4 and 8 rows it does not screen out the base of a row block that goes
  on ([above](#how-the-gemv-sweep-times-a-row-count)).
- Option errors, before the GPU is touched: `candidates` with the set; `sentinelEvidence` with another set, or
  naming a case that is no sentinel, one sentinel twice, a sentinel the call measures itself, or a shape other than
  the table's.

### The GEMV sweep's record

`karume-gemv-sweep/4`, the `data` of a `gemv-sweep` run (and what `runGemvSweep` returns). Nothing reads the records
of `karume-gemv-sweep/3`, whose rows had no skipped state, of `/2`, whose fingerprints were one per record, or of `/1`,
whose rounds were taken kernel after kernel. A record of the `explore` set is a `/4` record with two optional fields
no reader needs (`settings.sentinels`, `narrowedTo` of a summary) and rows only of the kernels it measured; a reader
that predates the set refuses it by its candidate set:

| Field | Content |
| --- | --- |
| `format`, `startedAt`, `date`, `aborted` | The format; when the sweep started and when its record was assembled; whether a cancel cut it short |
| `userAgent`, `adapter` | The host and the four adapter fields |
| `geometryProfile` | The id of the geometry profile the sweep's device resolved to (the adapter's embedded table, or `default`). The GEMM that is the baseline of a float-stored case past one row was built with its tile geometry; no other kernel of the sweep depends on it. The entry merges per-case records of one profile only |
| `gates` | The device facts that decided which candidates exist: the subgroup features (`subgroups`, `subgroup-size-control`, the WGSL `subgroup_id`; the subgroup form needs all three) and `maxComputeWorkgroupsPerDimension` (a candidate whose grid exceeds it, such as 32 lanes on a 262144-column head, is not measured) |
| `gpuTiming` | Whether `timestamp-query` was enabled, the unit (`ns` or `wall`), whether every round looks quantized |
| `settings` | The candidate set, for `explore` its sentinels (`sentinels`: the ids of the whole table's twelve), the cases (in measuring order: the table's, the sentinels first for `explore` — one `runGemvSweep` call and the workbench's merged record alike), row counts and candidates when narrowed, the rounds, the quick look's margin (`quickLookMargin`, 0 when it was off) and the timing protocol's constants (`targetPassMs`, `maxReps`, the warm start's `warmStart` — `minNs`, `maxNs`, `maxPasses`, `stablePasses`, `tolerance` —, `recalibrateBelow` and `calibrationAttempts`), the band (`tolerance`, the band of `linear`) and `notMeasured` (the packed-activation form: it reads the quantizing producer's codes, which the sweep does not run) |
| `cases` | The planned cases: id, storage (`i2`, `i4`, `i8` or `bf16`), i4 group, `n`, `k`, the row counts measured, where the shape comes from (the model and the role of the linear) and the case's `fingerprint` (of its id, storage, group, `n` and `k`, built as the geometry sweep's fingerprints: FNV-1a 64 of canonical JSON) |
| `summaries` | Per case and row count: the default profile's choice at `exact` and at `reorder` (by candidate name), the wall time, how steady the device was over the rounds: the baseline's largest round ÷ its smallest (`baselineSpread`; rounds clamped to 0 left out, 1 with one round), absent when the baseline's row failed, how many kernels the quick look screened out (`screened`) and how many were skipped as not carried by the run's M = 1 group of the case (`skipped`, 0 at M = 1); for `explore`, at M = 1 of a case that is no sentinel and was narrowed, the candidate names it was narrowed to (`narrowedTo`, the references measured besides) |
| `rows` | One per planned kernel: the case, the row count `m`, the candidate's name, its class, whether it is the baseline, its choice (the member of the GEMV family, lanes, column and row form; for the GEMM that is the baseline of a float-stored case past one row, `gemm-f32` with the storage and whether it writes `vec4` outputs), the fingerprint of its kernel (`kernel`: its program's pipeline keys, params, WGSL tokens and workgroups), the pipeline keys, the workgroups, its quick time (`quickPerDispatch`, with the quick look on) — and for a kernel the quick look screened out `screened: true` and nothing more —; for a kernel skipped at 4 or 8 rows (above) `skipped: true` after the workgroups and nothing more (no quick time, never the baseline); the time per dispatch (`perDispatch`) with its `rounds` (in round order, so round i of two rows was taken in the same round) and wall-clock rounds, the speed against the baseline; for the identical class whether every word equals the baseline's and how many differ; for the reorder class — and for the baseline at a row count with reorder rows — the largest absolute error against the host's f64 oracle, its use of the band, whether it is inside and the non-finite outputs; an `error` when it failed |
| `deviceLost` | The loss that ended the sweep, or `null` |

### The geometry sweep's explore set

The `explore` set of `geometry-sweep` measures the candidates of `full` on its case table in fewer rows
(geometry-explore.ts). It rests on the derivation's rule
([geometry-profiles.md](geometry-profiles.md#where-a-table-comes-from)): a geometry is adopted for a profile field — a
class of cases — only if, in every case of the class, its output is the default's and it is at least ×1.05 faster. A
candidate that loses in one case of a class is never adopted there, so the other cases of the class need not measure it.

- The cases are measured class by class: the classes in the order of their first case in the table, the cases of a
  class in the table's order (`geometryExploreOrder`). The class of a case is the field the derivation reads it in — the
  `gemmRows` or `i8a8.linear` bucket of its M, the stage of the fused attention, conv2d's m tile of its default
  geometry, conv3d —, from the one function the derivation uses (`sweepCaseSlot`, over derive.ts's `slotOf`).
- The first case of a class measures every candidate of `full`, after the quick look against the default at the margin
  given (×1.25 by default), as every set does.
- A candidate is **dropped** for the class once, in a case of the class (measured by the same call or handed in), the
  quick look screened it out, it failed, its output was not the default's, or its ratio to the default was below ×1
  (`EXPLORE_KEEP_SPEEDUP`: not slower than the default — below the derivation's ×1.05, so that the geometric mean of
  two records can still reach that). A row without a ratio or a comparison (the default's row failed) drops nothing.
  The default geometry is always measured.
- A later case of the class neither prepares nor measures a dropped candidate. Its row there is a skipped row
  (`skipped: true`): the case, op, shape, census count, geometry and its parameters, and nothing more — no key, no
  quick time, no rounds, no digest, no comparison. A skipped row drops the candidate too. In a case that cannot be
  measured (its plan, its inputs or its prelude failed) a dropped candidate keeps its skipped row while the others get
  failure rows.
- The source of ×1: replayed on the two `full` records of an RTX 5070 Ti under Chrome of 2026-11-18 (78 cases, 6,191
  candidate rows each), dropping below ×1 prepares 2,198 and 2,201 of the candidate rows and times 1,439 and 1,443
  where `full` timed 3,278 and 3,279 — an estimated 58 % less time in the cases —, and the fields the derivation adopts
  from the two records together are the same eleven decisions. Dropping below ×1.05 left them unchanged too (1,141
  and 1,152 timed rows); ×1 keeps the slack.
- A call that measures some cases of a run takes the rows of the cases earlier calls measured as `exploreEvidence`.
  The workbench's per-case loop orders the planned cases class by class and hands every call the rows of the calls
  before it, so its merged record holds what one call over the cases holds. `settings.cases` of an explore record lists
  the cases in that measuring order.
- Option errors, before the GPU is touched: `geometries` with the set; `exploreEvidence` with another set, or holding
  a row of a case outside the set's table, of a case the call measures itself, or with another op or shape than the
  table's.
- The record is a `karume-geometry-sweep/4` record with `settings.candidateSet: "explore"`. Its case-set versions and
  default fingerprints are `full`'s, and its `candidateKernels` fingerprints the rows that are not skipped (a skipped
  row built no kernel). A reader that predates the set refuses it: by `candidateKernels` when the record has skipped
  rows (that reader's fingerprint counts their kernels), by its candidate set otherwise.

Besides one status line per step, `runGeometrySweep` reports, for every set, a `plan` event when a case starts — its
place among the planned cases, the geometries it plans (the default and the candidates not dropped) and the candidates
it skips — and a `step` event per unit of the case: its phase (`prepare`, `warm-start`, `quick-look`, `timing`,
`repeat`), how many units are done and how many there are — a geometry's pipeline, the warm start (one unit), a
geometry's quick look, a geometry's timing (its calibration, idle runs and rounds), the default's re-measurement (one
unit). A phase reports 0 of its total when it begins and each unit when it ends, a unit that failed included; the quick
look reports nothing at a margin of 0. The callbacks run between timed passes and must stay cheap.

### The anima-single-file record

`karume-anima-single-file/1`, the `data` of an `anima-single-file` run:

| Field | Content |
| --- | --- |
| `format` | `karume-anima-single-file/1` |
| `request` | `model`, `quant` (the one the run used, the default resolved), `textConditioner`, `retainConverted`, `compare`, the prompt, the resolution, `steps`, `seed` |
| `file` | The input file's facts (path, size, how it was read), the name map's layout that recognised it and the keys it placed per component |
| `distribution` | The mount's facts (directory, the sha256 of its manifest, how it was read) |
| `fromFile` | The components that came from the file: the transformer, and the text conditioner when the run takes it from the file and the file holds one (read from the file's placement) |
| `arms` | `fromSingleFile`, then `fromPretrained` when compared: per arm the seconds of building the pipeline and, per generation (two), its seconds and the sha256 of the image's RGB8 pixels (what the Anima e2e hashes) |
| `equal` | `all`: every image of the run has the same pixels; `arms`: the first images of the two arms do (`null` without a comparison) |

- The second generation of a pipeline shows what one costs once the first has run: in the default per-stage residency
  every generate builds its Sessions again, so a component from the file is read and converted again — unless
  `retainConverted` is on, when it is served from memory. Two runs that differ only in `retainConverted` compare the
  second generation's time with and without; the first line of the summary says when it was on.

### The anima-step-timing record

`karume-anima-step-timing/1`, the `data` of an `anima-step-timing` run:

| Field | Content |
| --- | --- |
| `format` | `karume-anima-step-timing/1` |
| `request` | `mount`, `model` (the default resolved), `presets` (in the order they ran), `generations`, the resolution, `steps` and `guidanceScale` (the model's `pipelineConfig.defaults` the generations ran with; `null` when it declares none), the prompt and the `seeds` (generation g of every preset uses the g-th) |
| `distribution` | The mount's facts (directory, the sha256 of its manifest, how it was read) |
| `clock` | How the times are taken, in words (below) |
| `stallRule` | `factor` (3) and `firstStepLeftOut` (`true`) |
| `presets` | Per preset: its name, its Session options as the manifest declares them (`session`), the seconds of `fromPretrained` (`buildSeconds`), per generation its seed, its total (`generate` from the call until it resolved), the seconds of every step (`stepSeconds`, step 1 first), each stage's seconds (`stages`, in the order they ended) and every residency event with its time from the start of `generate`; then the median step (`medianStepSeconds`), how many steps it read (`stepsCounted`) and the stalled steps (`stalled`: generation and step, both 1-based, the seconds and the ratio to the median) |

- The times are the wall clock: `performance.now()` read at the top of each `onEvent` callback. A step runs from the
  previous `denoise-step` callback to its own; step 1 runs from the `transformer` stage's start, so it includes building
  the DiT Session, which the per-stage residency does in every generate. A stage runs from its `start` callback to its
  `end` callback, its Session's build and release included. No GPU timestamp is read.
- The median and the stalled steps read every step but step 1 of every generation of the preset. A stalled step is one
  slower than 3 × that median.
- The summary's first line names the mount, the model, the resolution, the steps and the guidance; one line per preset
  gives its median step, how many steps stalled with each one's generation, step, seconds and ratio, and the
  generations' totals; one line per generation lists its step times; the last line lists the residency events, or says
  there were none.
- A step event out of order and a stage that ends without having started are errors: the timeline would not mean what
  the record says.
- As in [the EmbeddingGemma preset A/B's record](#the-embeddinggemma-preset-abs-record), the record holds no GPU clock,
  power or thermal reading. The adapter, its description and features and the backend are the record's `environment`.

### The EmbeddingGemma preset A/B's record

`karume-embeddinggemma-preset-ab/2`, the `data` of an `embeddinggemma-preset-ab` run. Nothing reads the records of
`karume-embeddinggemma-preset-ab/1`, whose arms had no GC reading:

| Field | Content |
| --- | --- |
| `format` | `karume-embeddinggemma-preset-ab/2` |
| `request` | `mount`, `model` (the manifest's default), `base` and `compared` (the presets, the default resolved), `tokens`, `rounds`, `minGain` |
| `distribution` | The mount's facts (directory, the sha256 of its manifest, how it was read) |
| `text` | The golden case the text is a prefix of, its prompt (absent without one), the prefix's length in code points and its length in token ids; not the text itself |
| `probes` | Per preset, base first: the wall time of `fromPretrained` (`loadMs`: the manifest, part 0, admission, reading and uploading the weight parts — under Node from the mirror's directory), of the first and the second `embed`, and from the second run's diagnostics the Session build's breakdown (`build`: `SessionBuildStats` — `supplyWaitMs` is the wait for the supply, the read of the parts), the weight arena (`weights`), the low-precision storage (`storage`), the plan backing (`planBacking`), the planned dispatches summed over the pipeline keys and the geometry profile |
| `arms` | Per arm, base first: its preset's name, the median over the rounds of the mean of the round's two visits (`medianMs`; the base's is the judge's `baseMedianMs`), the fastest timed visit (`minMs`), the host's garbage collections in each timed visit (`gc`, [below](#garbage-collections-of-the-host)), each open's wall time (`openMs`, `fromPretrained` after the probe read the files once), the runs compared with the arm's own first vector and how many differed |
| `cosine` | The cosine of the compared arm's first vector against the base's |
| `modelsVersion` | The version of `@karume/models` |
| `record`, `judgement` | The runtime's `karume-session-ab/1` record, unchanged (its identity holds against the base's first vector, so the compared arm's differs), and `judgeSessionAb(record, { minGain, requireIdentity: false })` of the compared arm |

- The record holds no GPU clock, power or thermal reading: WebGPU exposes none, and an entry holds no Node API. A
  reading belongs next to the run.
- The record keeps the fastest visit beside the median: under Node a major GC of the process can fall on the same
  visit of every round. What the development machine showed in such runs (slow stretches in both arms at 2,048
  tokens, and one visit per round stalled by a major GC at 8,192) is in
  [research 2026-11-22](../research/2026-11-22-embeddinggemma2-quant-screen.md).

#### Garbage collections of the host

- An observer of the Web-standard `PerformanceObserver` entry type `gc` runs from just before the runner opens the
  arms until the runner has closed them. The type is feature-detected through `PerformanceObserver.supportedEntryTypes`:
  Node reports it, browsers do not.
- Each arm notes `performance.now()` (the entries' timeline) right before and right after the `embed` of every run.
  A GC belongs to a visit when its `startTime` lies inside that span, both ends included; one that begins in the
  runner's span but outside the `embed` (the call around it) is not counted.
- On a host that reports GCs, an arm's `gc` is `{ "observable": true, "visits": … }`: `visits[r][v]` is round
  `r + 1`, visit `v + 1`, the layout of the record's `timesMs`, each with `major` (the count of major GCs — Node's
  `detail.kind` 4, V8's mark-sweep-compact), `majorPauseMs` (their durations summed), `other` and `otherPauseMs`
  (every other kind together: scavenges, minor mark-sweeps, incremental marking, weak callbacks). The warm-up round
  and the probes are not in it. Run spans that do not match the runner's runs of that arm are an error.
- On a host that does not, it is `{ "observable": false }`, with no counts.
- Node delivers a `gc` entry on a later task than the GC; the observer is read after the runner returns, two tasks of
  the event loop later, and keeps every entry it was handed for its whole life.
- The GC reading changes neither the times nor the judgement.
- The summary's fourth line says per arm how many timed visits held a major GC and the range of those visits' summed
  major pauses (`<preset> <n> of <visits> visits (<lowest> ms to <highest> ms of pause each)`, one value when
  they are equal, no range when none held one), or that the host cannot observe GC.

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
- The envelope, the typed parameters, the path rule below and the two sweep entries' parameters, plans, runs, merged
  records and summaries are the private package `@karume/sweep-app-support`, which the
  [benchmark app](benchmark.md) uses too: a sweep's record is the same whichever app made it.

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
distributions as the Playground's dev server does (`/models/index.json`, `/models/<mount>/…` and `/hf/…`, read only),
and the files under the repository's `inputs/` at `/inputs/<path>` (GET and HEAD with `Range`, read only, nothing
outside `inputs/`, no listing — the Playground's dev server does not serve it).

| Request | Answer |
| --- | --- |
| `POST /workbench/results` | Writes the body as a record and answers `201` with `{ "path": … }` (repository-relative). `405` for another method; `415` unless `Content-Type` is `application/json`; `403` unless the `Origin` header is the server's own origin; `413` for a body over 64 MiB (not read past the limit); `422` unless the body is JSON with the format above, a kebab-case `entry.id`, an `environment.host` of `browser` or `node` and an `environment.adapter` object. The bytes written are the body, unchanged. |
| `GET /workbench/checkout` | The checkout at the moment of the request. `405` for another method. |
| `GET /workbench/batches` | The untracked batches, read at the moment of the request: `{ "batches": [{ "id": …, "text": … }] }` for every `*.json` file directly in `inputs/workbench/batches/` (the id is the file name without `.json`), in file-name order; a missing directory is an empty list. A file over 1 MiB (not read past the cap) or not UTF-8 is listed with `"problem"` in place of `"text"`. `Cache-Control: no-store`. `405` for another method. |

- Nothing the client sends becomes part of a path except through the two rules above (the kebab-case entry id and the
  adapter directory); the request for the batches names no path.
- The preview server and a production build have none of these endpoints. The preview server serves the distributions
  and `/inputs/`.
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
- `synthetic-quick` is the tracked short run of three of the synthetic entries: `kernel-check` with its defaults,
  `geometry-sweep` at the `quick` set with one round and no idle time, `attention-bench` with its defaults. It is not
  the evidence run. `gemv-sweep` is not a step of it: its `quick` set holds about 2,600 kernels (about 2,150 without the
  subgroup features), each with a probe after a warm start per row count, which is no short run even at one round.

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
- It downloads no model: an entry reads a local mirror, a file under `inputs/`, or nothing.
- It keeps no history and compares no two records.
- A batch decides nothing from a result: it is a list of runs, and it leaves no record of its own.

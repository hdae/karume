# The benchmark app

What `apps/benchmark` does now: a small browser page that an outside collaborator opens on their own computer to run
the exploratory sweeps on their GPU and send the result back as one file, and the Node command with which the
maintainer turns that file into workbench records. It measures with `@karume/runtime/tune` and only drives it; it
does not measure the speed of a model.

The app is private (not published) and independent of the workbench: it needs no workbench, no local mirror under
`models/`, no dev-server endpoint and no file of the repository's asset roots. `vp build` gives a static site
(relative asset paths) that runs from any static host. What it shares with the workbench — the typed parameters, the
record's envelope and its path rule, the per-case loop with idle time and its cancel and device-loss rules, the two
sweeps' plans, runs, merged records and summaries — is the private package `@karume/sweep-app-support`, which both
apps import; nothing of a record is built twice. Its tests run with `pnpm --filter @karume/benchmark test` and need no
GPU.

## What a run measures

- Three sweeps, in this order: the GEMV sweep (`gemv-sweep`) and the geometry sweep (`geometry-sweep`), each at its
  `explore` set ([workbench](workbench.md#the-explore-set),
  [workbench](workbench.md#the-geometry-sweeps-explore-set)), then the geometry sweep at its `grid` set — the ladder of
  the grid-fill rule of the GEMM ([workbench](workbench.md#the-geometry-sweeps-grid-set)) —, each with its entry's
  defaults for that set — 5 rounds, the quick look's margin of the set (×1.10 for the GEMV sweep's, ×1.25 for the
  geometry sweep's `explore` set; the `grid` set takes none), every case of its table (118, 78 and 33) — and no idle
  time between cases: the page gives every sweep `idleSeconds` 0 (the workbench's defaults for these sets are none for
  the GEMV sweep's and 10 s for the geometry sweep's). An idle time the trial sets is waited before every case but the
  first, a later sweep's first case too.
- Each sweep's parameters are parsed from `candidateSet=explore` (the `grid` set's from `candidateSet=grid`) and the
  run's `idleSeconds` by the workbench entry's own declarations, and planned by the shared settings, so a sweep's
  record holds the `entry.parameters` the workbench writes for the same run, in the same order. A grid record of a
  collaborator's device is therefore what the generator's `--grid-records` reads
  ([geometry-profiles](geometry-profiles.md#deriving-the-grid-fill-rule)): GPU timestamps that are not rounded, a
  steady record alone or any other beside a second.
- Three checkboxes (all on) choose the sweeps — the GEMV sweep, the tile geometry sweep and the GEMM tile ladder —;
  there is no other setting on the page.
- For a short trial the URL takes `?cases=` (comma-separated case ids of any of the three tables: each sweep runs the
  ids of its own table, and a sweep the list names none of does not run) and `?idleSeconds=` (an integer of at least
  0, for every sweep, in place of 0). Both reach the records' parameters and the file's `trial`. Any other parameter, an unknown
  case id, an empty list and an idle time that does not fit are shown as a problem, and nothing can start.

## Before a run

- The page shows the adapter it gets from `requestAdapter()`: its vendor, architecture and description (`(not
  reported by the browser)` when empty), and whether it lists `timestamp-query`.
- Without WebGPU or without an adapter the page says so. Without `timestamp-query` it says that the sweeps would time
  with the wall clock and such a run could not be used as evidence, and Start stays off.
- An adapter without a description gets a note: in Chrome, `chrome://flags/#enable-webgpu-developer-features` makes it
  report one, so the result can be matched to that GPU model ([geometry-profiles](geometry-profiles.md#the-three-levels)),
  and the flag also stops Chrome from rounding the timestamps to 100 µs, which the GEMV derivation and the grid
  derivation refuse (a quantized record).
- A short note in plain words says what will run (the chosen sweeps with their case counts), the estimated time (the
  estimate below, before anything is measured), to keep the tab in front and the computer awake and use the GPU for
  nothing else meanwhile, that one run is enough for a table when the GPU's timings hold steady through it and that
  otherwise the page says so at the end and a second run at another time is what makes a table, and
  what the file holds (the GPU's vendor, architecture and description, its features and limits, the browser's user
  agent and the timings — nothing else about the person or the computer) and that it is to be sent to the maintainer.
- Start and Cancel. Start is off while no sweep is chosen.
- The footer names the checkout the page was built from (the commit, whether it had uncommitted changes, the runtime's
  version).

## While it runs

- The sweeps run one call per case through the shared per-case loop (its cancel points and device-loss rule:
  [workbench](workbench.md#the-entries)), each sweep's record made by the shared record code (`recordRun`), with the
  environment read from an adapter the page requests itself when the sweep starts (`host: "browser"`, the user agent,
  `crossOriginIsolated`; no backend is declared) and the checkout baked in at build time.
- The page makes its session once, when it loads (`gemvSweepSession`: a random id and its start — the id a version 4
  UUID built from `crypto.getRandomValues`, so a page served over plain HTTP from a LAN address, where
  `crypto.randomUUID` is absent, still starts and says it has no WebGPU), and hands it to
  every GEMV sweep call, so the GEMV record of each run names it with the time since the page loaded
  ([workbench](workbench.md#the-gemv-sweeps-record)): two runs in one page load are told from runs in two.
- The sweeps' callbacks only store the event (with `performance.now()`); the page folds the stored events and paints
  on `requestAnimationFrame`.
- Two levels of progress:
  - the whole run: how many of its cases are done, of how many, over the sweeps (and per sweep). A case is done when
    its sweep call returned a record that neither a cancel nor a device loss cut short (`cutShort`); a case a cancel
    stopped midway is not done, and its rows are not counted;
  - the case being measured: its place in the run, its sweep, its id and its kind (a GEMV sentinel or a case measuring
    what its sentinels predict; a geometry case first of its class or a later one; a grid case, every tile of the
    ladder), then the step it is at, from the sweeps' `plan` and `step` events — for the GEMV sweep its row count
    (`M = 4 (row count 2 of 3)`), for the geometry sweep the geometries it plans and skips, and the phase in words with
    its units done of all (`Timing rounds: 24 of 60`; a grid case's timing counts its rounds of every tile,
    interleaved); before the first step, "Acquiring the GPU"; during the idle wait, the seconds left. The sweep's own status
    line is shown under it.
- The time left ([below](#the-time-left)), from the start of the run.
- The patterns found so far ([below](#the-patterns-found)), updated as cases finish.
- A device loss is shown as it is reported; the run stops after that case.
- A screen wake lock is held while the run goes where the browser has the API (`wake-lock.ts`). Its state follows the
  lock: `held` once the browser grants it, `not-held` when the browser releases it (as it does while the page is
  hidden) or has not granted it yet, `failed` when it refuses the request. While the run goes, the page asks again
  whenever it is shown and holds no lock, with no request pending; the file records the state when the run ended. The
  page counts how often and how long it was hidden during the run, shows it, and the file records it.
- Cancel stops the run at the loop's next cancel point.
- When the page itself fails at the end of a run (the file cannot be made, or showing the ending throws), the progress
  stops, Cancel is off, Start is on again and the page says the run stopped on an error of the page; when a file was
  made and holds measurements, it is offered with "Download what was measured".

## The time left

A pure function of the run's cases and what finished (`estimate.ts`): a case is expected to take the mean of the
finished cases of the same kind in the run; before one finished, the prior of its kind — 29 s for a GEMV sentinel, 17 s
for another GEMV case, 58 s for a geometry case first of its class, 19 s for a later one, 10 s for a grid case —,
scaled by how the finished cases of the same sweep (the other kind) fared against their priors when there are any (the
grid set has one kind: a grid case is read by its prior, then by the finished grid cases alone). The time left is what the
running case has left of its expectation (never below 0), the rest of the idle wait, and every case after it with the
idle time before it. Before any case finished it is labelled as a first estimate from other machines' runs.

The priors come from the maintainer's records: the GEMV sweep's `full` records of an Apple M5 (2026-11-23) and the
`explore` set's replay (half the in-group time), the geometry sweep's `full` records of an RTX 5070 Ti (2026-11-18) and
the `explore` set's estimate on them (58 % less time in the cases). The GEMV priors were 27 and 15 s until the sweep
re-timed each group's references and carried candidates from M = 1 down to ×1.03 (2026-11-28): on the `explore`
records of 2026-11-27 that is about 6 to 7 % more passes than the rounds take on a device without a table and 8.5 to
8.9 % on one with an embedded table (its choices are re-timed too), and 1.4 to 3.9 % more timed rows, about 4 % more
of a GEMV case; and the explore set's prediction at ×1.03 instead of ×1.10 adds 0.6 to 7.5 % to the other cases' timed
passes (2.8 % on average on the four devices' records), an embedded table 1.3 to 4.8 % to the whole set's. Over the
four devices with and without their table that gives 29 and 17 s. The grid prior is the page's own run of the grid set
at its default 5 rounds on the RTX 3080 Ti under Node on Dawn (2026-11-28): 337 s for the 33 cases, 10.2 s a case
(6.2 to 13.5) — the same day's two workbench grid records at 7 rounds, less two rounds, read 12.0 and 12.1 s. The full
run of the three sweeps is then about 1 h 17 min (about 1 h 55 min with `?idleSeconds=10`); without the grid set about
1 h 12 min.

## The patterns found

Counted over the per-case records of the cases that finished (done, [above](#while-it-runs)), with the derivations'
own rules, predicates and exclusions of `@karume/runtime/tune` (nothing restated):

- **GEMV**, by the rule of one steady record ([geometry-profiles](geometry-profiles.md#deriving-the-gemv-choices)): of
  the (case, row count) groups at which the derivation picks a winner (`gemvGroupRole`'s `winner`: M = 1), those in
  which a candidate a table can hold (`gemvTableCandidate`: a sequential variant, or a plain or staged parallel row —
  not the subgroup form) reads against its class's reference in the same group (`gemvSteadyReading`: the reference —
  the current default profile's choice at `exact` or at `reorder`, `gemvClassReference`, whatever the group's
  summary names — is one, `gemvReferenceRefusal`; the candidate's row is evidence with its check held,
  `gemvRowReading`; both rows have a pass of at least 20 ms and two positive rounds; the record times in ns) with a
  break-even margin of at least the margin and a re-timing of the reference within the bar (`gemvReadingIsEvidence`).
  Where the group's summary names the device table's choice for the class (`table`) and it is another kernel than
  the default's, the count reads as the derivation keeps a table's choice: the group holds a pattern when a candidate
  clears its bar over the default's choice and over the table's in the same group, or the default's choice clears it
  over the table's; a group without a time of the table's choice holds none (a record that did not time it does not
  move it), and one where that choice failed or its check failed is read against the default's choice. A record
  without `table` (taken before it) is read against the default's choice. Counted at the default margin, ×1.065 (`GEMV_MARGIN`), and at the lowest one, ×1.03 (`GEMV_LOWEST_MARGIN`), side
  by side. Beside the count the page states the run's class, `gemvSteadiness` over the summaries the run's GEMV
  record merges — the finished cases' and the groups a cancel or a device loss left finished in a case cut short, which
  the merge keeps (the count itself reads finished cases only) —, so the line states the class the record states, read as the derivation will read that record
  (`gemvRecordReading` of `@karume/runtime/tune`, the derivation's own test: the finished cases' timing as the merge
  states it — the first's unit, rounded when any was): GPU timestamps the derivation refuses — rounded (`quantized`
  not `false`) or none (the wall clock) — "no evidence for a table, however steady its timings"; steady and in ns —
  "this run alone is enough for a table"; steady in another unit (Deno's raw ticks) — "a table needs a second such
  run"; unsteady — more than 20 % of its groups spread beyond ×1.05, "please run the page once more at another time —
  a table needs two such runs"; unclassified — fewer than 20 groups so far, too few to tell.
- **GEMM tile ladder**: no count. The grid derivation (`deriveGemmGrid`) takes two integers for the whole set — every
  case against the table's own tile, weighted by how often its shape runs — and decides no case on its own; the line
  says how many grid cases are measured, and that the choice is made from all of them after the run.
- **Tile geometry**: of the classes the finished cases belong to (`geometryCaseClass`, the field the derivation reads a
  case in), those in which one candidate is at least `GEOMETRY_MIN_SPEEDUP` (×1.05) faster than the default, with the
  default's output, in every case of the class measured so far, none of those cases dropped from the ratio material
  (`geometryCaseExclusion`: the default's re-measurement failed, is missing or out of range, or the default row's
  rounding-error bound is over 1%) and the candidate's observation in none of them dropped (`geometryRowExclusion`:
  its ratio's rounding-error bound is over 1%). The rounding step is the one of the record the finished cases' records
  merge into (100 µs when any of them looked rounded — `mergedQuantized`, `geometryTimestampQuantum`).

The page shows both as "n of m" and says what they are: "beats today's choice" — the device table's where the record
names it —; where the run held steady and its timestamps count, a table can take them from it alone; where it did
not, only after a second run on the same machine agrees. The derivation reads more than this count does — whether a
row is of the current runtime (its kernel fingerprint against the table derived for), the table it derives for
where that is not the one the record names, for a parallel row its kernels at the other row counts of its range, a
sequential key's other cases, and on an unsteady device a second record —, so the count is no promise of a table.

## The file

`karume-benchmark/1`, one per run, one-space JSON with a final newline:

| Field | Content |
| --- | --- |
| `format` | `karume-benchmark/1` |
| `startedAt`, `endedAt` | When the run started and ended (ISO 8601) |
| `complete` | Every planned sweep ran to its end and the run was never cancelled |
| `stop` | Why the run ended: `completed`, `cancelled` (also when Cancel was pressed after the last measurement began), `deviceLost` (with the reason and message) or `error` (with its message) |
| `sweeps` | The entries of the planned sweeps, in run order |
| `trial` | What the URL asked for (`cases`, `idleSeconds`; `{}` for a run of the defaults) |
| `page` | How often and how long the page was hidden during the run (`hiddenCount`, `hiddenSeconds`) and the wake lock's state when the run ended: `held`, `not-held`, `failed` or `unavailable` ([above](#while-it-runs)) |
| `records` | One workbench record (`karume-workbench/1`) per sweep that ran, exactly as the workbench writes it: the same envelope ([workbench](workbench.md#the-record)) and data (the merged `karume-gemv-sweep/4` or `karume-geometry-sweep/4` record), entries `gemv-sweep` and `geometry-sweep` — the geometry sweep once per set, `explore` then `grid` |

- The name is `karume-benchmark_<host>-<adapter name>_<UTC end time>.json` (the adapter name by the rule of
  [where records go](workbench.md#where-records-go)), with `_incomplete` before `.json` when the run did not complete.
- A run is complete only when every planned sweep finished and its signal never aborted: a Cancel that lands while
  the last case re-measures its default (where the sweep takes no signal) leaves the sweeps' loop completed but the
  run cancelled, and its records say `cancelled`.
- A complete run downloads the file at once (a `Blob` through `a[download]`); the page then offers "Download
  again". When that download throws, the page says so and offers the file with "Download the result". A run a
  cancel, a device loss or an error stopped keeps what was measured and offers it with a button, marked as
  incomplete; it is not downloaded by itself. A run that measured nothing offers nothing.
- After a run whose GEMV record's timing the GEMV derivation refuses (`gemvTimingRefusal`), the page says in plain
  words that the GEMV part will not be accepted as evidence: for quantized GPU timing (its `gpuTiming.quantized` is not
  `false`) that the browser rounded the timestamps to 100 µs, which moves the short GEMV kernels' timings by more than
  the differences looked for, and that in Chrome `chrome://flags/#enable-webgpu-developer-features` stops the
  rounding; for a record timed without GPU timestamps, that the wall clock adds a delay that moves from pass to pass.
  When a grid record is in the file, its timing is read by the same test (the grid derivation refuses rounded
  timestamps and the wall clock too), and the note says the GEMM tile ladder part will not count either — or alone,
  when the GEMV sweep did not run or its timing counts. When a record of the geometry sweep's `explore` set is in the
  file too, it says that part is still used (`afterRunNotes` in `notes.ts`).
  Otherwise, after a run whose GEMV
  record states itself unsteady (its `steadiness`), the page asks in plain words for one more run at another time
  (another day is best) and both files: two such runs are what makes a table; after one that states too few groups for a
  class (a short trial), it says so.
- The page sends nothing itself: it says to send the downloaded file to the maintainer.

## Unpacking the file

`node apps/benchmark/unpack.ts <file>…` (plain Node) reads each file and writes each of its records to
`outputs/workbench/<host>-<adapter name>/<entry id>_<UTC time>.json` under the repository root — where the workbench
would have written it, the time the record's `recordedAt` — with the text the shared record code gives it, byte for
byte what the workbench writes. The table generator (`tools/geometry-profiles`) reads those files as it reads the
workbench's.

- Every file is read and checked (its format, its fields, every record's envelope and entry) and every target is
  checked before anything is written. A target that exists, or two records of two files that would take one name (the
  same file given twice), refuse the whole command: nothing is overwritten and nothing is written. Two records of one
  file that would take one name (the geometry sweep's two sets ending within one second) take the workbench's own
  suffix for a taken name, `-1` after the stamp, in run order.
- An incomplete file's records are written too, with a line that says why the run stopped.
- Beside each GEMV record it writes, the line says how the derivation will read it (`gemvClassText`, by
  `gemvRecordReading` as the page's line): rounded GPU timestamps or none (no GEMV evidence, whatever its class), or
  else the class the record states — steady and in ns (one record is enough for a table), steady in another unit (a
  table needs a second record), unsteady (a table needs a second record of the device), too few groups for a class
  (read as unsteady), or no class stated (a record written before the records stated one). Beside each grid record
  it says that the generator's `--grid-records` reads it, or that it refuses its timing (rounded GPU timestamps, or
  none).
- Exit code 0 when every record was written; 1 for a file that cannot be read, is no bundle, or would overwrite; 2 for
  a command line without a file.

## The dev server and the build

- `vp dev` in `apps/benchmark` listens on `127.0.0.1:8792` only; `vp preview` on the same port. Both send
  `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, as the workbench's do;
  a static host sends its own headers (the record says whether the page was cross-origin isolated).
- The checkout the records carry is read when the config loads: when the dev server starts, or when the site is built.

## What it does not do

- It sends nothing: no form, no upload, no endpoint.
- It has no setting beyond the three checkboxes and the two URL parameters, keeps no history and derives no table.
- It has no Node driver of its own; under Node the same sweeps run through the workbench's command.

# 0106: Keep sha256 reference values as rows per environment key

- Status: accepted (2026-09-20 — step 1 of the test reorganisation; implemented). Amended by its own
  addenda (the reference guard's green condition, 2026-10-03; PNG rows, 2026-10-17); partly superseded by
  [0127](0127-node-dawn-toolchain.md) (the key base on Node, 2026-10-16). On 2026-10-18 decision 7 and the
  Intel Arc B570 rows of addendum decisions 1 and 8 were withdrawn, the rule of the 2026-09-26 addendum
  that deletes practical rows was kept with the `deno-*` rows exempt, and its rejected alternative was
  re-graded to MAY. [0129](0129-test-layout-and-lanes.md) decision 3 keeps the rows, the three modes and
  the reference guard as MUST.
- Date: 2026-09-20
- Scope: `packages/runtime/tests/helpers/{environment,reference,results}.ts` (environment key; reading and
  writing the fixtures and the reference guard; results directory); unit tests `environment_key_test.ts` /
  `reference_fixture_test.ts` / `results_writer_test.ts` in `packages/runtime/tests/`; tracked fixtures
  `packages/models/tests/fixtures/references/*.json`; the consuming e2e files (under `packages/e2e/tests/`
  since ADR 0129). The 2026-09-22 addendum adds `packages/runtime/tests/e2e_golden_test.ts`, the real-weight
  golden e2e files and `tools/verify-diff/`. **The runtime, the distributions and the exporter are
  unchanged** (test and verification tooling only).
- Related: ADR [0005](0005-verification.md) ("an all-SKIP is an explicit FAIL", the lane split of
  2026-09-20) / [limitations](../limitations.md) (why bit identity across devices is not guaranteed) /
  [known-issues](../known-issues.md) (the Intel Arc B570 section) / [assets-layout](../assets-layout.md)
  (the `outputs/` root)

## Context

The e2e sha256 regression checks (anima, sbv2 WAV, irodori WAV) test only whether the sha256 of the output
PNG / WAV is **bit-identical** to the expected value. They can claim that "the numbers did not move by one
bit **on that machine**"; bit identity across devices is not guaranteed by specification (limitations:
driver-dependent transcendental functions, the shader compiler's fma contraction, Tint / naga paths).

The expected value was one constant in the test source. When the development GPU was swapped from the RTX
3080 Ti to the Intel Arc B570 on 2026-09-20, all 16 sha256 checks turned red (the PNGs looked correct).
An all-red check catches no regression; a tolerance is forbidden (it erases what "the port worked"
means); re-recording the constant removes the old machine's regression detector. **Two machines competed
for one slot.** There was also no place for the **artifacts** (PNG / WAV) of a run, so the material for
an A/B existed only after a failure.

## Decision

### 1. Reference values are **rows per environment key** (tracked JSON)

They live in `packages/models/tests/fixtures/references/<series>.json` (tracked), shaped
`{ "schema": 1, "kind": "sha256", "cases": { "<case ID>": { "<environment key>": "<sha256>" } } }`. A file
that cannot be read, is broken JSON, or has a different `schema` / `kind` throws (never treated as empty).
Writes are stable — case names and keys in **lexicographic order**, two-space indentation, trailing
newline — because a diff of a file shared by several environments is required to show only the rows that
actually changed.

### 2. Environment key = `<runtime>-<adapter name slug>`

`helpers/environment.ts` builds it from `GPUAdapterInfo`. The base is `description` when not empty, else
`${vendor}-${architecture}` (Deno fills `description`, Chrome fills `vendor` / `architecture`). (Superseded
for Node by ADR 0127 on 2026-10-16: the base is `GPUAdapterInfo.device`, because Dawn's `description` is
the driver version; the test harness builds `node-` keys only. Deno and Chrome keep this rule.) The base is
lower-cased, `(r)` / `(tm)` / `(c)` are dropped, `[^a-z0-9]+` becomes `-`, and outer `-` are trimmed:
`deno-intel-graphics-bmg-g21`, `chrome-apple-metal-3`, `node-nvidia-geforce-rtx-3080-ti`.

An environment that yields no name **throws** (an empty-key row would compare another machine's values as
this machine's). An environment without a GPU adapter has **no** key (not an empty string).
`environment_key_test.ts` pins the spelling.

**The key MUST NOT contain the driver version.** With it, every driver update would silently escape to "no
row yet" = explicit SKIP; without it, a driver update that moves the numbers **breaks the row of the same
key** and turns red. (Dawn does expose the version; it is left out on purpose.)

**The OS is not in the key** (ruling of 2026-09-21): no machine uses one GPU under several OSes, so a
longer key would only respell rows without preventing a real collision. If one occurs, it is reconsidered
then; `results.json` records the OS, so a row's machine can be traced.

### 3. Three modes (environment variable `KARUME_REFERENCE`)

| Mode      | Case without a row                                      | Case with a row                                                   |
| --------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| unset     | explicit SKIP (warns once at registration how to write) | compare (a mismatch is red; MUST NOT be turned into a tolerance) |
| `write`   | **adds** the measurement, green                         | compare (**does not overwrite**)                                  |
| `rewrite` | adds the measurement, green                             | **overwrites** with the measurement, green (prints old → new)     |

An unknown spelling throws. `rewrite` is for when you can say **in advance what changed**, and it **never
touches another environment's rows**. `reference_fixture_test.ts` pins the three modes without a real GPU
(environment and mode injected as arguments).

### 4. The reference guard (`registerReferenceGate`) and its opt-out

A guard at the end of each sha256 file keeps "no reference value here yet, so every check was skipped"
from **becoming a silent green** — ADR 0005's rule, with the opt-out `KARUME_ALLOW_NO_REFERENCE=1`. Where
neither GPU nor assets exist and the checks do not run, the guard does not fire (`runnable`). The failure
names where the rows live and how to write them (`KARUME_REFERENCE=write`). **It counts only the case IDs
the file registered in this run** (the same set as `warnMissing`):

- Red when **none** of the registered cases has a row for the current environment.
- Green when only **some** do. (Superseded by the addendum of 2026-10-03: green requires a row for every
  registered case, or a hold for each id without one.)
- **Orphan rows** of renamed or deleted cases are not counted (one orphan row would hide an all-SKIP of
  the live cases).
- A writing mode (`KARUME_REFERENCE`) is green with zero rows (that run writes them); the release machine
  must not keep it set ([release-runbook](../release-runbook.md) §1).

### 5. Results go to `outputs/verify/<environment key>/<date>_<series>/`

Reference values (tracked, per environment) and results (untracked, growing per run) live apart. Results
are **safe to delete** (`rm -rf outputs/verify`; the `outputs/` convention of
[assets-layout](../assets-layout.md); paired with `VERIFY_ROOT` in `tools/export-recipes/_shared/paths.py`):

- `results.json` — `schema` / `family` / `environment` (runtime name, version, V8, TypeScript version, the
  four adapter fields, the OS) / `checkout` (HEAD sha and dirty flag; `null` without `git`) / `startedAt`
  / `cases[]` (`id` / `status` = `pass` | `fail` | `written` | `rewritten` / `expected?` / `actual?` /
  `artifact?` / `elapsedMs` / `note?`), rewritten whole after every case so a crash keeps what came before.
- **Artifacts** (PNG / WAV) — **every run, pass or fail**; otherwise a matching run leaves no A/B material.

Two runs of one series on one day **keep the last** (split by date, not by run). Choosing and creating the
directory are **deferred until use** (modules load on machines without a GPU too; demanding a key at load
time would fail files that should be skipped).

### 6. Twin cases compare against "the twin's **row** and the measurement"

Cases that should produce the same bytes from different paths (anima `fromPretrained-512` /
`onEvent-1024` / `base-cfg-fromAssets-shards`) keep **a row under their own case ID** and also check that
the measurement equals the twin's row; with per-environment rows, one constant can no longer serve both.
The check lapses only while the twin's row is missing here.

### 7. The RTX 3080 Ti row spelling was derived, not observed

Withdrawn 2026-10-18: the spelling `deno-nvidia-geforce-rtx-3080-ti`, derived from the recorded adapter
name while that GPU was out of the machine, was confirmed after it returned on 2026-10-06 (addendum
decision 9: the `deno-` and `node-` rows agree on all 11 anima cases).

## Alternatives considered

1. **An "environment" column in the test source's reference arrays** — rejected: every row becomes a code
   change, nothing like `KARUME_REFERENCE=write` is possible, and review diffs fill with hashes. Reference
   values are data written and read by machines.
2. **Reference values under `outputs/`** — rejected: the local roots are untracked
   ([assets-layout](../assets-layout.md)), so a fresh clone has no detector. A reference value declares the
   bytes **this machine must produce**, so it is tracked; the split from results is decision 5.
3. **Absorbing mismatches with a tolerance** — rejected by ADR 0005: loosening erases what "the port
   worked" means. Sanity on another device is checked by a **self A/B** (limitations), not the reference sha.
4. **Golden tolerances in tracked fixtures, derived from measurements** — rejected: the band would follow
   each run, against ADR 0005's ban. A band is a **specification-derived declaration**, not a measurement.
5. **A tracked ledger of results with cross-environment differences as a check** — rejected: bit identity
   across devices is not guaranteed, and untracked results would move to the tracked side (decision 5).

## Consequences

- **Bit identity across devices is still not guaranteed.** More rows add machines on which a regression is
  detected, not a guarantee; the by-design entry in limitations stays.
- **The first run on a new machine is an all-SKIP plus a red reference guard.** `KARUME_REFERENCE=write`
  on the same lane writes the rows but **accepts that run as correct**, so the machine's sanity is checked
  first (inspecting the output, a self A/B).
- Review reads which environment's rows moved (lexicographic order, and `rewrite` touches only the current
  environment). Each new series adds a fixture and a reference guard; `results.json` holds the environment
  and checkout for later comparison (addendum decision 3).

## Addendum (2026-09-22) — golden tolerances per environment and comparing results

### Addendum decision 1: the golden second stage (WGSL specification band) is also kept per environment key

`OUTPUT_TOLERANCE` in `packages/runtime/tests/e2e_golden_test.ts` is a two-level table,
`<model>/<output name>` → **environment key** → `{ spec }`. The second stage fires only **for outputs with
a row for the running machine**; elsewhere only the first stage (Karume's own criterion) applies. The first
stage, the failure messages and the `note` record are unchanged. Rows are split so that **a loosening added
for one machine does not spread** to machines that round the op almost exactly. Golden expected values come
from torch on the CPU and do not depend on the device, so only the **tolerances** are per machine (sha256
rows keep the expected value itself per machine).

Withdrawn 2026-10-18: the one row, `activations/sin` × `deno-intel-graphics-bmg-g21` (atol 2⁻¹¹, measured
on the Intel Arc B570). It remains in `e2e_golden_test.ts` but never fires; the harness builds `node-` keys.

### Addendum decision 2: a passing run records its differences in `results.json` (optional field `measurements`)

A case gains the optional `measurements`: `output` / `maxAbs` / `maxRel` / `tolerance` (the band used) /
`stage` (`karume` | `spec`; on a fail, the last stage measured). A tolerance check shows numbers only on
failure; recording passing differences gives the material for judging a band before anything breaks.

- **No derived values** (ratios to the band are derived by the reader, `tools/verify-diff`); two forms of
  one number allow fixing one alone.
- `schema` **stays 1** (optional field). Non-finite values become `null` through `JSON.stringify`.
- Judgement, bands, asserts and `ignore` do not depend on the field (a record, not an input).

The real-weight golden e2e files (11 then) write to results directories named `<series>-golden`, so they
do not compete with the models-side e2e files that use the plain series name on the same day.

### Addendum decision 3: comparing across environments is a read-only tool, not a check (`tools/verify-diff`)

It reads only `outputs/verify/<environment key>/<date>_<series>/results.json` and writes nothing. It takes
the newest directory per environment (`--date` pins one) and prints, in Markdown or `--json`, a "case ×
environment key" matrix per series with differences in `status`, artifact sha and one-sided cases, plus
warnings for `checkout` mismatches and dirty checkouts.

**It is not a check**: cross-device bit identity is not guaranteed, so differences cannot be red. The exit
code is 0 with differences and 1 only for unreadable documents or a different `schema` (never a silent "no
differences"). A second machine's results are copied **by hand** (no sync — decision 5's split); results
kept elsewhere are read with `--root`.

## Addendum (2026-09-25) — the date of the results directory is UTC

The `<date>` of decision 5 is the **UTC** date of the run's start (`toISOString()` —
`packages/runtime/tests/helpers/results.ts`); behaviour is unchanged. On a JST machine a run before 09:00
lands on the previous day, "the same day" is a UTC day, and `tools/verify-diff --date` takes a UTC date.
UTC dates are monotonic in run time and give every machine one date basis.

## Addendum (2026-09-26) — row classes (reference rows / practical rows) and re-recording practical rows

[ADR 0110](0110-practical-tier-numerics-contract.md) decision 7 applied to these rows.

- The class is **derived** from whether the `session` of the case's quant preset is empty (no separate
  table). Empty = **reference row** (frozen — `rewrite` only for a change that moves the reference path's
  numerics). Non-empty = **practical row** (regression and determinism detector of the practical preset).
- A commit that intentionally changes practical numerics rewrites the practical rows in the same commit and
  records that the reference rows stay unchanged and green (ADR 0058 decision 4 ①) and that the A/B check
  and the census are green.
- Practical rows of environments that could not be rewritten in that commit are **deleted** (decision 3
  rewrites only the current environment, so an old row would leave another machine stuck on red). That
  environment rebuilds them: all-SKIP plus red guard → self A/B → `write`. (Kept by the ruling of
  2026-10-18: the `deno-*` rows are exempt as the frozen baseline of the migration until Deno is retired —
  ADR 0127 open point.)
- At the time, 10 of anima's 11 cases (`f16+dit8-a8-attn8-s16`) were practical rows and `f16-1024` the
  reference row; all sbv2 / irodori rows were reference rows.
- Alternative not taken then: move every anima case to the reference-path preset (`f16+dit8`), which would
  have removed the only E2E detector of the practical preset while the A/B check and the census were
  missing. Both now exist (`packages/e2e/tests/e2e_anima_ab_test.ts`,
  `packages/runtime/tests/helpers/pipeline-census.ts`), so the alternative MAY be reconsidered (re-graded
  2026-10-18 to MAY: the condition set for reconsidering it is met).

## Addendum (2026-09-26, second) — the optional `results.json` field `comparisons`

Self A/B measurements (ADR 0110 decision 5) are recorded in `comparisons`
(`{ output, reference, practical, relRms, maxAbs, band? }`): optional under schema 1, no derived values,
non-finite values as `null`; `tools/verify-diff` prints an A/B table per case. Bands are not per
environment key (ADR 0110 decision 4).

## Addendum (2026-09-26, third) — more series with rows, the runtime-side location, artifact rules, held rows

### Addendum decision 4: the series with rows

Rows extend beyond anima / sbv2 / irodori (the maintainer's ruling of 2026-09-26):

| Series (fixture)                                             | Location | Case ID           | Artifact (bytes hashed)                                  |
| ------------------------------------------------------------ | -------- | ----------------- | -------------------------------------------------------- |
| anima / sbv2 / irodori                                       | models   | case name         | PNG pixels (addendum decision 9) / WAV                   |
| siglip2 golden (`siglip2-golden`)                            | runtime  | `<series>/<case>` | f32 safetensors of the pooled output                     |
| gemma4 golden (`gemma4-golden`)                              | runtime  | case name         | f32 safetensors of the last-position logits row `[1,V]`  |
| birefnet real images (`birefnet`)                            | models   | `<series>/<case>` | f32 safetensors of the matte logits                      |
| depth-anything real images (`depth-anything`)                | models   | case name         | f32 safetensors of the depth                             |
| vowel-detector full chain (`vowel-detector`)                 | models   | case name         | f32 safetensors of the logits                            |
| gemma4 / gemma4-qat quant presets (`gemma4` / `gemma4-qat`)  | models   | `<model>/<quant>` | JSON of the ids of 64 greedy tokens for a fixed prompt   |

`vowel-detector.json` has had no rows since it was added (`"cases": {}`, 639993d6). Wan2.1 (`wan.json`)
and Wan2.2 (`wan-ti2v.json`) were added later.

- The gemma presets are enumerated from **every quant preset** in the mirror's manifest (`mirrorSeats` in
  `packages/models/tests/helpers/gemma-reference.ts`), not a fixed table, which would silently leave a new
  preset without a row; the row class is derived from the same enumeration.
- The ids JSON is `{ "ids": [...] }` (two spaces, trailing newline). Decoded text and stop reason go only
  into the `note` (the text derives from the ids). The prompt asks for more than 64 tokens, and no row is
  written from a run that stopped early (a shorter sequence weakens the check).

### Addendum decision 5: runtime-side fixtures live in `packages/runtime/tests/fixtures/references/`

Rows of the runtime-side golden e2e go to `packages/runtime/tests/fixtures/references/<results family
name>.json` (`gemma4-golden` / `siglip2-golden`), named after the results directory (addendum decision 2)
so that a series with e2e files on both sides (gemma4) does not confuse them with the models-side fixture
(`gemma4.json`). Shape, order and modes follow decisions 1 and 3.

### Addendum decision 6: artifact rules

- An f32 tensor artifact is **one safetensors file** (`f32ArtifactBytes` in
  `packages/runtime/tests/helpers/reference.ts`): tensor name = graph output name, dtype `F32`, shape as
  output, **no metadata** (per-run values in metadata would make the row capture the run time, not the
  numbers). A shape whose element count disagrees with the data length throws.
- The sha is taken over **the artifact file's bytes** (`actual` always equals the artifact's sha).
  (Superseded for PNG by addendum decision 9 on 2026-10-17; it holds for safetensors and WAV, which depend
  on the numbers alone.)
- birefnet / depth-anything also write PNGs for inspection, but the sha covers the f32 output before
  quantization (8-bit conversion misses differences that cross no quantization boundary).

### Addendum decision 7: order of sha checks added to existing tests

Series whose rows were added **on top of** an existing check (siglip2 / gemma4 golden = tolerance,
birefnet / depth-anything real images = discrimination, vowel-detector = exact `.lab` match) follow these
rules (canonical: the module doc "caller types" of `reference.ts` and `settleOrObserve`):

1. Rows are written or compared **after** the original check passes.
2. In compare mode, a case without a row skips only the comparison, writes the artifact and records
   `status: "pass"` + `actual` + `artifact` (no `expected`), so `tools/verify-diff` can compare `actual`
   across environments. This branch lives in `settleOrObserve` alone, not per series.
3. The outcome is **recorded before** a mismatch fails (otherwise the measured sha is lost).

Series whose test is itself the sha256 regression check (anima / sbv2 / irodori, the gemma presets) keep
marking a case without a row as an explicit SKIP via `ignore`.

### Addendum decision 8: held rows — environment-key rows that stop a series on a machine

Environment-key rows also mark **series that do not run on a machine**: `HELD_SERIES` (series →
environment key → reason) in `packages/e2e/tests/e2e_birefnet_test.ts` (moved from `packages/runtime/tests/`
by ADR 0129).

- A row MUST be added only when running the series there **crashes the whole process** (taking later
  verification down). It is not for series whose numbers disagree (those stay red until fixed).
- Only device tests (golden comparison, geometry discrimination) are stopped, not the asset integrity tests.
- A stopped series announces itself once with `console.warn`; a self-test checks that the inner keys have
  the environment-key format (a misspelled row would silently stop working).
- Rows are per environment key; a machine without a GPU has no key and matches none.
- The reason states the evidence and the removal condition; removing the row signals the cause is resolved.

Withdrawn 2026-10-18: the three rows for the 2048² series (`birefnet-hr-2048-f16` / `lucida-2048` /
`lucida-2048-f16`) × `deno-intel-graphics-bmg-g21` (Intel Arc B570 under Linux xe: `deform_conv2d`
exceeded the 5 s job limit and Deno panicked). The B570 left the development machine on 2026-10-06 and the
harness builds `node-` keys, so the rows, still in the code, never match.

## Addendum (2026-10-03) — decision 4 revised: the reference guard is green when every registered case has a row here or its id is held

Decision 4's "green when only some cases have rows" let a forgotten row for a new case ID pass as a SKIP
with a warning (ADR 0119 decision 8 relies on closing this). Revised in `7dc17309`:

- Green: every registered case ID has a row for the current environment, or each id without one is
  **held** here (`ReferenceHolds` passed to `registerReferenceGate` — the shape of `HELD_SERIES`, for cases
  that crash the process only). An id both held and with a row is red (the stale hold is named). Zero
  registered cases stays red. A writing mode is always green.
- A case without a row in compare mode behaves as before (explicit SKIP, artifact written); the failure
  names the ids without rows and the `write` procedure.
- Opt-in cases (enabled by an environment variable) are not registered while disabled (precedent: Wan's
  conditional `CASE_IDS`).

## Addendum (2026-10-17) — PNG rows use the sha of the pixels before compression

### Addendum decision 9: for cases whose artifact is a PNG, the sha is taken over the RGB8 pixels that `encodePng` writes into IDAT, not over the PNG bytes

In the port to Node (ADR [0127](0127-node-dawn-toolchain.md)), Deno and Node on the same RTX 3080 Ti
differed by zero in every series, yet anima's 11 PNG rows broke: IDAT is `CompressionStream("deflate")`
output, and its zlib differs per runtime. The rows should capture the numbers, not the compressor, so the
sha covers the pixels `encodePng` writes into IDAT (row-major RGB8 — the RGBA of `GeneratedImage.data`
without alpha, which is always 255 and checked by `encodePng`). The PNG stays as an artifact for inspection.

- The sha of the pixels of the artifact decoded with `decodePng` (`@karume/test-support/png-decode`)
  equals the row.
- Existing rows were recomputed on 2026-10-17 from the frozen artifacts (not re-run); the
  `deno-nvidia-geforce-rtx-3080-ti` and `node-nvidia-geforce-rtx-3080-ti` rows then agreed on all 11 cases
  (Dawn and wgpu produce bit-identical pixels on this GPU). A Deno upgrade can no longer break PNG rows.
- `tools/verify-diff` compares `actual` and is unaffected.

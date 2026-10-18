# 0129: Test layout — package tests, a real-weight e2e package, lanes, and what verification keeps

- Status: proposed (2026-10-16; to be accepted by the maintainer after review)
- Date: 2026-10-16
- Related: [0005](0005-verification.md) (verification strategy — this ADR amends its 2026-09-20 "test
  lanes" addendum and qualifies its 2026-10-11 upstream-parity addendum for the test suite) /
  [0106](0106-device-keyed-references.md) (per-environment sha256 rows, the `write` / `rewrite` modes
  and the no-tolerance rule — kept unchanged) / [0008](0008-public-api.md) (the public-surface gate
  — kept, re-implemented without Deno) / [0110](0110-practical-tier-numerics-contract.md) (the
  op-level oracle and its f64 bands — kept) / [0125](0125-birefnet-matte-postprocess.md) (an example
  of an upstream-parity golden that becomes per-family) / [0065](0065-exporter-core-recipe-split.md)
  (the Python side; not touched by this ADR)

## Context

The 0.15.0 cycle moves the TypeScript side of the repository from Deno to Node 24 with Dawn as the
WebGPU implementation and Vite+ (Vitest, Oxlint) as the toolchain. The toolchain change itself is
recorded in ADR [0127](0127-node-dawn-toolchain.md) (proposed alongside this one); this ADR records how the test suite is laid out on the new toolchain and
which verification rules survive the move. The maintainer decided the points below on 2026-10-15/16
(post-0.14.0 plan, decisions 2, 3 and 4, and the approved layout).

What the suite looked like at 0.14.0 (Deno):

- Tests lived in `packages/runtime/tests` and `packages/models/tests`, mixed: unit tests, op-level
  oracle tests, tests on repository fixtures, the all-skip gates, and real-weight end-to-end tests
  (`e2e_*_test.ts`, which need local model assets under `models/`, `outputs/series/` or `inputs/`
  and a real GPU). Real-weight e2e for one family could sit in either package (for example
  `e2e_sbv2_test.ts` in runtime, `e2e_sbv2_wav_test.ts` in models).
- Lanes (ADR 0005, addendum 2026-09-20) were `deno.json` tasks: `test:core` plus
  `test:models:<family>`. `verify_lanes_test.ts` parsed those task strings and checked coverage,
  disjointness and non-empty lanes.
- A full run took 118–134 minutes on the RTX 3080 Ti, about one hour of it in the Wan lanes.

What the port changes:

- `deno.json` tasks are no longer how tests are selected. Vitest selects files through
  `test.projects`, so the lane table needs a new home and the coverage gate a new source of truth.
- The public-surface gate (`deno doc`) and the dependency-graph gate (`deno info`) depended on Deno
  tools that do not exist on Node.
- Deno's test API (`Deno.test`, `@std/testing/bdd`, `t.step`) is used by about 3,200 tests and 692
  steps. Porting them and restructuring them at the same time would mix two kinds of change; the
  maintainer ruled that porting and fixing do not happen in the same phase (plan decision 9).
- Decision 2 of the plan separates two kinds of correctness evidence that the old suite mixed: the
  op-level oracle (Karume against its own TypeScript mirror or an f64 reference, within a derived
  band) and parity with the upstream implementation at the model exit (torch goldens, model-exit
  bands). ADR 0005's 2026-10-11 addendum already said that 100% upstream parity was a tool for the
  early, low-trust phase of the project.

## Decision

### 1. Layout: three published packages, two private packages

`packages/runtime`, `packages/hub` and `packages/models` MUST keep only tests that need no local
model assets: unit tests, op-level oracle tests, tests on fixtures tracked in the repository, and the
gates. Real-weight end-to-end tests live in the private workspace package `@karume/e2e`
(`packages/e2e/tests`), grouped by model family through the lane table (decision 2). The harness
lives in the private workspace package `@karume/test-support`.

Rationale: plan decision 3 asks that a runtime change be verifiable by the runtime tests alone and
that model checks run before a release. A package whose tests read multi-gigabyte local assets
cannot be verified on its own on a machine without them; a separate package makes "needs local
assets" a property of where the file is, not of a skip condition inside it.

`@karume/test-support` holds:

| Entry | Contents |
| --- | --- |
| `@karume/test-support/setup` | Vitest `setupFiles` entry: one Dawn `GPU` per worker, installed as `navigator.gpu`, plus the WebGPU globals (`GPUBufferUsage`, ...). Runs without `navigator.gpu` where Dawn cannot create an adapter. |
| `@karume/test-support/test` | The Deno-shaped registration shim on Vitest (decision 6). |
| `@karume/test-support/fs` | Deno-shaped file helpers on `node:fs` (byte results own their `ArrayBuffer`). |
| `@karume/test-support/webgpu` | `platformObject<T>()` for hand-built WebGPU fakes. |
| `@karume/test-support/lanes` | The lane table (decision 2). |
| `module-graph`, `local-dist-server` | The oxc-parser module graph for the dependency gate; the `node:http` server for the local-distribution tests. |
| `codemod/` | The porting codemods (`port-deno-test.mjs`, `check-ported.mjs`, `deno-baseline.mjs`, `move-e2e.mjs`). |

Both private packages are `"private": true`. They are never published, and the dependency-graph
gate does not treat them as members whose entries must stay Web-standard (they may still be edge
targets).

### 2. Lanes are Vitest projects generated from one table

The lanes are defined once, in `packages/test-support/src/lanes.ts`, and the root
`vite.config.ts` builds `test.projects` from that table (`laneProjects()`); nothing else may define a
lane. A lane is run with `vp test --project <lane>`; `vp test` runs every project, that is, the full
suite.

- A family lane (`models:<family>`) claims the files whose name carries the family token, in
  `packages/models/tests` (host-side tests of the family) and in `packages/e2e/tests`. Family names
  follow the distribution spelling, as before.
- `core` is everything the family lanes do not claim, over the population
  `{packages,tools}/*/tests/**/*_test.ts`. It holds the runtime, hub and test-support tests, the
  family-neutral models tests, the tool tests, and the gates.
- Family lanes MAY overlap each other (`e2e_gemma4_ple_gpu_test.ts` is in `models:gemma4` and
  `models:gemma4-qat`). A family lane MUST NOT overlap `core`.

`packages/runtime/tests/verify_lanes_test.ts` is the coverage gate and reads the same table. It MUST
fail when:

1. some test file is in no lane (core ∪ lanes ≠ all tests);
2. a family lane shares a file with `core` (core ∩ lane ≠ ∅);
3. a lane, or `core`, collects no file;
4. a file looks like a test but is not spelled `*_test.ts` (`*.test.ts`, `test.ts`, ...), because
   Vitest's include would silently not collect it;
5. `vite.config.ts` does not build its projects from the table, or does not pass `--expose-gc` to
   the workers (memory tests call `globalThis.gc()`; without the flag they silently lose effect).

The gate expands the globs itself and MUST reject glob syntax it does not understand (`?`, `[...]`,
`**` without a separator) rather than disagree silently with Vitest.

Rationale: ADR 0005's addendum named the only danger of lanes as a test that no lane runs. With the
table as the single source and the config generated from it, the gate checks what actually runs,
not a parallel description of it. The four task-string parser tests of the Deno version went away
with the parser.

### 3. What stays mandatory

The following MUST stay in the suite through the migration and after it:

- **The op-level oracle.** Kernel and op tests against Karume's own TypeScript mirror
  (`packages/runtime/src/reference/`) with `atol = 0` where ADR 0110 claims bit equality, and against
  the f64 reference within the op bands (`GEMM_TOLERANCE` and the other derived bands of ADR 0110).
  This is the evidence that the WGSL is correct independently of any upstream.
- **The per-environment sha256 rows** of ADR 0106, with their three modes (unset = compare, missing
  row = explicit SKIP; `KARUME_REFERENCE=write` adds missing rows only; `rewrite` overwrites the
  current environment's rows and prints old → new) and the reference gate. A sha gate MUST NOT be
  turned into a tolerance: a tolerance would erase what the gate proves (the numbers did not move on
  this machine).
- **The all-skip gates** of ADR 0005 — `gpu_gate_test.ts` (no adapter, no `shader-f16`, no
  `timestamp-query`), `assets_gate_test.ts` (no real-weight assets) and `distribution_gate_test.ts`
  (no distribution mirrors) — with their explicit opt-outs (`KARUME_ALLOW_NO_GPU`,
  `KARUME_ALLOW_NO_SHADER_F16`, `KARUME_ALLOW_NO_TIMESTAMP_QUERY`, `KARUME_ALLOW_NO_ASSETS`,
  `KARUME_ALLOW_NO_DISTRIBUTION`). A run that sets an opt-out is not release evidence.
- **The public-surface gate** (ADR 0008: per-package snapshot `tests/fixtures/public-surface.json`
  plus the named-export gates and `e2e_public_api_test.ts`) and **the dependency-graph gate**
  (runtime dependencies are Web-standard only). Their extraction moves from `deno doc` /
  `deno info` to TypeScript 7's API (`typescript/unstable/sync`, in
  `packages/runtime/tests/helpers/public-surface.ts`) and oxc-parser
  (`@karume/test-support/module-graph`). What they assert does not change.

Rationale: plan decision 2 keeps the oracle and relaxes only upstream parity. The sha rows and the
all-skip gates are the two mechanisms that stop "not verified" from reading as "verified"; nothing in
the toolchain change weakens the reason for them.

### 4. What is relaxed: upstream parity at the model exit

Upstream-parity checks at the model exit — torch goldens compared at a model's output and model-exit
bands — become a per-family choice and MAY be dropped for a family after the triage (decision 5).
They are no longer a suite-wide requirement.

Rationale: ADR 0110 recorded that model-exit bands become vacuous (any implementation passes them)
once activation quantisation and stacked opt-ins are involved, and ADR 0005's 2026-10-11 addendum
already set parity at the level of numbers or their intent, not bit-for-bit. Where a family has a
parity check that still discriminates (for example the BiRefNet matte golden of ADR 0125, which is
CPU-only and exact), the family keeps it; where it does not, it costs GPU hours without evidence.
This ADR does not change what counts as matching upstream (ADR 0005, 2026-10-11) — only whether a
family must carry such a test.

### 5. Triage before anything is dropped

After each layer (runtime → hub → models → e2e) is ported and runs on Node, a reviewed list is
produced with one row per test file: file → kind (unit / oracle / fixture / gate / upstream parity /
real-weight e2e) → needs GPU / needs local assets → lane → keep / move / drop. The maintainer
approves the drops. Nothing MUST be deleted, skipped or weakened without appearing in an approved
list.

The omission check is the frozen Deno JUnit baseline
(`outputs/misc/0.15-baseline-2026-10-16/deno-junit/`, one XML per lane): per file, the number of
passed tests and of skipped (Deno "ignored") tests on Node MUST match the Deno baseline, unless the
difference is listed and approved in the triage. `node packages/test-support/codemod/deno-baseline.mjs
<files>` prints the Deno counts.

Rationale: the port touches about 430 test files; a test that stops running looks exactly like a
test that passes. The baseline turns "did we lose anything" into a per-file count comparison.

### 6. The Deno-shaped shim is a phase-1 scaffold

Phase 1 ports tests behaviour-preservingly through `@karume/test-support/test`: `test(name, fn)` /
`test({ name, ignore, only, fn })`, `describe` / `it`, the hooks, and `t.step` running sub-steps
sequentially inside the test. Deno's sanitizer options are accepted and ignored. In phase 2 the tests
are rewritten in Vitest idioms (`describe` / `it` / `expect`, nested `describe` instead of `t.step`)
and the shim is removed.

Rationale: phase 1 must show that results did not change; rewriting assertions at the same time
would make a difference in counts or numbers ambiguous between a porting defect and a test rewrite.

### 7. No time target; which suite verifies what

There is no time target for the full suite. A runtime (or hub, or core-layer) change SHOULD be
verified with the `core` lane; a change confined to one family SHOULD be verified with that family's
lane; the family lanes MUST be run before a release.

Rationale: plan decision 3. The old addendum already split execution units for the same reason
(most of the time went to families unrelated to the change); this decision makes it the stated
policy instead of a convenience, and drops any target duration for the full run.

## Consequences

### For users

None. Test layout is repository-internal; no published entry point, type or behaviour changes.
`@karume/e2e` and `@karume/test-support` are never published.

### For the repository

- The lane tasks in `deno.json` (`test:core`, `test:models:*`) are no longer the source of truth and
  go away with `deno.json` as the port proceeds (they are still present in the worktree on
  2026-10-16, unused by the Node suite). The lane table in `lanes.ts` replaces them.
- 61 `e2e_*_test.ts` files moved to `packages/e2e/tests/` (12 from `packages/runtime/tests`, 49 from
  `packages/models/tests`; see the appendix). The move re-relativised 464 relative specifiers
  (`move-e2e.mjs`). The moved files still import `packages/runtime/tests/helpers`,
  `packages/models/tests/helpers` and package `src/` internals by relative path; `@karume/e2e` is not
  a public-surface consumer (the public-surface E2E, `e2e_public_api_test.ts`, stays in runtime).
- The phase-1 move was by file name (`e2e_*`). Two tests that start with `e2e_` stayed in runtime
  because they need no local assets: `e2e_golden_test.ts` (torch goldens tracked under
  `packages/runtime/tests/fixtures/golden/`) and `e2e_public_api_test.ts` (the ADR 0008 gate).
  Conversely, some files left in the packages still read local assets when present and skip
  explicitly otherwise — for example `assets_fusion_counts_test.ts` (runtime, `core`),
  `image_preprocess_real_test.ts` and `wan_ti2v_i2v_preprocess_real_test.ts` (models). A grep for
  local-asset paths on 2026-10-16 matches 32 files in `packages/models/tests` (29 of them in family
  lanes), 4 in `packages/runtime/tests` (two of them gates) and 2 in `packages/hub/tests`; matches
  include comments. Decision 1 is the target; the triage (decision 5) brings these files in line with
  it.
- The sha256 reference rows stay where they are (`packages/models/tests/fixtures/references/`,
  `packages/runtime/tests/fixtures/references/`), as do the helpers that read and write them
  (`packages/runtime/tests/helpers/{reference,environment,results}.ts`).
- Test file names stay `*_test.ts` in phase 1; Vitest's include uses that spelling.

### For verification

- CI runs `vp check` and `vp test --project core` with `KARUME_ALLOW_NO_GPU`,
  `KARUME_ALLOW_NO_ASSETS` and `KARUME_ALLOW_NO_DISTRIBUTION` set: CI runners have neither an adapter
  nor the local assets. The CI run is therefore a regression check of the non-GPU tests, not release
  evidence (ADR 0005 unchanged on this point).
- Family lanes run locally on the GPU machine. All files run in one forked worker, one file at a
  time (`pool: "forks"`, `fileParallelism: false`), because GPU tests share one device and the sha
  rows assume one run at a time. There is no per-test timeout (`testTimeout: 0`).
- The environment key on Node is `node-<adapter device slug>` (Dawn puts the driver version into
  `description`, so the key is built from `device`), for example `node-nvidia-geforce-rtx-3080-ti`
  next to Deno's `deno-nvidia-geforce-rtx-3080-ti`. Node therefore starts with no rows: every sha
  case is an explicit SKIP and the reference gate is red until rows are written.
- Node rows are written per lane with `KARUME_REFERENCE=write`, and only after that lane's Node
  artifacts (PNG / WAV / tensors) have been compared with the frozen Deno artifacts in
  `outputs/misc/0.15-baseline-2026-10-16/` (157 files from the 0.14.0 release run, with
  `SHA256SUMS`). Writing rows before the comparison would freeze whatever Node produced as "correct".
- Lane membership is compared with the Deno baseline per lane. On 2026-10-16 the `core`, `anima`,
  `irodori` and `sbv2` lanes matched with zero difference; the other lanes were waiting for their
  baseline run.
- Dawn exposes features the Deno (wgpu) runs did not (`subgroups`,
  `packed_4x8_integer_dot_product` on the development machine), so some cases that were explicit
  SKIPs in the baseline run on Node. In the decision-5 count comparison these show up as
  skipped → passed and are expected differences, to be listed.
- Vitest has no counterpart of Deno's per-test resource / op sanitizers; that leak detection is lost
  for now (open point).

### Explicitly left open

See "Open points".

## Staging

1. **Phase 1 — port (behaviour unchanged).** Skeleton and harness (`@karume/test-support`) →
   codemod port of every test file → move of the real-weight e2e into `@karume/e2e` → lane table and
   coverage gate → re-implement the public-surface and dependency-graph gates on TypeScript 7 /
   oxc-parser → run runtime → hub → models → e2e on Node, comparing counts with the Deno JUnit
   baseline (decision 5) and artifacts with the frozen Deno artifacts, then `write` the Node rows per
   lane. A triage list is produced after each layer.
2. **Phase 2 — reorganise (behaviour unchanged).** Apply the approved triage (moves and drops);
   rewrite tests in Vitest idioms and remove the shim (decision 6); redefine lanes where the triage
   moved files, with the coverage gate rebuilt on the same table.
3. **Phase 3 — behaviour changes** (numerics profile, default preset — separate ADRs). Their
   intended differences are re-recorded in the sha rows with `rewrite`, per ADR 0106 and ADR 0110.
   *(2026-10-18: the numerics profile was deferred — ADR 0126 — and the default presets stay — ADR 0128
   decision 6 — so 0.15.0 rewrites no sha row for either.)*

## Open points

- The concrete triage lists and which families keep upstream parity at the model exit (decision 4)
  are decided per layer by the maintainer; none is approved yet.
- Whether the op/block-level torch goldens of `e2e_golden_test.ts` (repository fixtures, no local
  assets) count as part of the kept oracle or as relaxable upstream parity. ADR 0005's step 3 ("add a
  golden for each new op") is not revised by this ADR.
- Whether the release judgement stays "one green full run" (ADR 0005, 2026-09-20 addendum: lane
  greens do not substitute for it) or becomes the union of lane runs. Decision 7 says when lanes run,
  not how a release is judged.
- Whether the shared test helpers (`packages/{runtime,models}/tests/helpers`) and the reference-row
  fixtures move into `@karume/test-support` / `@karume/e2e`, and whether e2e should import package
  internals at all.
- The replacement for Deno's resource / op sanitizers.
- The file-name spelling in phase 2 (`*_test.ts` → `*.test.ts`) and the matching change of the
  coverage gate's spelling rule.
- The Japanese term for "all-skip gate" (plan, open).

## Appendix: real-weight e2e files moved to `packages/e2e/tests` (2026-10-16)

Every file below was `packages/<origin>/tests/<name>` and is now `packages/e2e/tests/<name>`. Lanes
as in `lanes.ts`; `e2e_gemma4_ple_gpu_test.ts` is listed under `models:gemma4` and also belongs to
`models:gemma4-qat`.

| Lane | From `packages/runtime/tests` (12) | From `packages/models/tests` (49) |
| --- | --- | --- |
| `models:anima` | — | `e2e_anima_ab_test.ts`, `e2e_anima_test.ts` |
| `models:sbv2` | `e2e_sbv2_test.ts` | `e2e_sbv2_ab_test.ts`, `e2e_sbv2_wav_test.ts` |
| `models:irodori` | `e2e_irodori_test.ts` | `e2e_irodori_ab_test.ts`, `e2e_irodori_cleanup_test.ts`, `e2e_irodori_codec_test.ts`, `e2e_irodori_context_test.ts`, `e2e_irodori_latent_test.ts`, `e2e_irodori_reference_test.ts`, `e2e_irodori_w8a8_test.ts`, `e2e_irodori_wav_test.ts` |
| `models:gemma4` | `e2e_gemma4_test.ts` | `e2e_gemma4_chat_test.ts`, `e2e_gemma4_directory_test.ts`, `e2e_gemma4_drafter_test.ts`, `e2e_gemma4_greedy_test.ts`, `e2e_gemma4_ple_gpu_test.ts`, `e2e_gemma4_pretrained_test.ts`, `e2e_gemma4_product_test.ts`, `e2e_gemma4_quant_test.ts`, `e2e_gemma4_reduce_parity_test.ts`, `e2e_gemma4_reference_test.ts`, `e2e_gemma4_sampler_test.ts`, `e2e_gemma4_sequence_test.ts`, `e2e_gemma4_speculative_test.ts`, `e2e_gemma4_token_exit_test.ts` |
| `models:gemma4-qat` | — | `e2e_gemma4_qat_packed_static_quantize_test.ts`, `e2e_gemma4_qat_reference_test.ts`, `e2e_gemma4_qat_test.ts` |
| `models:birefnet` | `e2e_birefnet_test.ts` | `e2e_birefnet_real_test.ts` |
| `models:depth-anything` | `e2e_depth_anything_test.ts` | `e2e_depth_anything_real_test.ts` |
| `models:siglip2` | `e2e_siglip2_test.ts` | — |
| `models:vowel-detector` | `e2e_vowel_detector_test.ts` | `e2e_vowel_detector_chain_test.ts`, `e2e_vowel_detector_lab_test.ts` |
| `models:minicpm5` | `e2e_minicpm5_test.ts` | `e2e_minicpm5_greedy_test.ts` |
| `models:embeddinggemma` | `e2e_embeddinggemma_test.ts` | — |
| `models:deberta` | `e2e_deberta_test.ts`, `e2e_deberta_w8a8_test.ts` | — |
| `models:dacvae` | `e2e_dacvae_test.ts` | — |
| `models:wan` | — | `e2e_wan_ab_test.ts`, `e2e_wan_dit_test.ts`, `e2e_wan_pipeline_test.ts`, `e2e_wan_swap_test.ts`, `e2e_wan_umt5_test.ts`, `e2e_wan_vae_chunks_test.ts`, `e2e_wan_vae_tiles_test.ts` |
| `models:wan-ti2v` | — | `e2e_wan_ti2v_ab_test.ts`, `e2e_wan_ti2v_dit_test.ts`, `e2e_wan_ti2v_i2v_pipeline_test.ts`, `e2e_wan_ti2v_pipeline_test.ts`, `e2e_wan_ti2v_vae_chunks_test.ts`, `e2e_wan_ti2v_vae_encoder_test.ts`, `e2e_wan_ti2v_vae_tiles_test.ts` |
| `models:umt5` | — | `e2e_umt5_encoder_test.ts` |

## Addendum 2026-10-17 — the triage was applied

The maintainer ruled on the triage list (`docs/research/2026-10-16-test-triage.md`, option a):
every drop, merge and move row was applied (its "Applied" section is the ledger, including the
cases dropped as duplicates and the deviations decided at acceptance), the review rows wait for a
per-row ruling. The lane membership after the change is whatever `packages/test-support/src/lanes.ts`
claims; the tables above describe the phase-1 move and are kept as the record of it. The dacvae
lane was retired with its only test. One rule was added during acceptance and holds from here on:
**no runtime or models test imports from `packages/e2e`** (the series table lives in
`packages/models/tests/helpers`, the A/B-gate and fusion-count helpers in `packages/e2e/tests/helpers`).

## Addendum 2026-10-17 (2) — upstream parity is opt-in; two families are not run by default

The maintainer ruled on the review rows of the triage list: upstream-parity cases (comparisons of
Karume's model-exit or per-component outputs with torch / diffusers goldens under a tolerance) stay
in the code but run only with `KARUME_UPSTREAM_PARITY=1` (`packages/e2e/tests/helpers/upstream-parity.ts`;
the default lanes skip them with that reason). Parity is proven when a family is implemented; later
changes are caught by the sha256 reference rows and the self A/B checks. Record-only cases were
deleted. The token-only Gemma 4 exit (ADR 0068 decision 4) is not a product feature and its test
and recipe entry point were removed. Two families are maintained but not run by default
(`LOW_PRIORITY_FAMILIES` in the lane table: minicpm5, siglip2): their lanes exist, `core` still
excludes their files, and `vp test` includes them only with `KARUME_ALL_FAMILIES=1`; the release run
uses the default set. EmbeddingGemma keeps its parity test in the default run because it is the
family's only check until it has reference rows.

## Addendum 2026-10-17 (3) — decision 6 is fulfilled: the shim is gone

Phase 2 step 2-2 replaced the Deno-shaped registration with Vitest's own forms: a codemod
(`packages/test-support/codemod/port-shim.mjs`) rewrote the object and three-argument forms to
`test.skipIf(cond)(name, fn)` and moved the imports to `vite-plus/test` (308 files), and the 54
files that used sequential sub-steps were restructured by hand into `describe` / `it` with
`beforeAll` / `afterAll` (557 steps became `it`s; titles, case ids and reference-row keys
unchanged). `@karume/test-support/test` no longer exists. `@std/assert` stays as the assertion
library.

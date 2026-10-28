# Backlog — the source of truth for wave order and work items

> This file alone is the source of truth for the project's **priorities, wave order and unfinished items**.
> Status vocabulary: `now` (current wave) / `next` (the next big wave) / `later` / `release` / `parked` (with a revival condition).
> Operating contract: ① completed items are **deleted** (git and the ADRs / research hold the history) ② measured values and design arguments are
> **not written** here — point to the source (ADR / research / ledger) ③ the source of truth for filing, adopting or rejecting performance candidates and for their kill criteria is
> [perf-ledger](perf-ledger.md); this file only refers to them as waves ④ the source of truth for by-design constraints is
> [limitations](limitations.md) — a constraint appears here only when a ruling turns it into work.

## now — after the 0.15.0 release (released 2026-10-31)

0.15.0 is out (JSR `@karume/runtime` / `@karume/hub` / `@karume/models` 0.15.0; HF re-uploads of anima, anima-extra,
wan2.2, irodori v4.1-small and gemma4, card-only updates of the other published repositories; published-smoke green on
Node and Deno). The next wave is 0.15.1 — Gemma 4 QAT decode speed (the first item of *later*), started in a new session.

## later

- **Gemma 4 QAT decode speed (0.15.1 — maintainer 2026-10-30)**: QAT E2B decodes slower than regular E2B in the
  browser. The attribution of 2026-09-19 ([research](research/2026-09-19-qat-speed-recon.md) §6) names four kernel gaps
  (standalone SRQ dispatches, the occupancy of the INT2 GEMVs, the f32 `per_layer_model_projection`, the sequential INT2
  `lm_head`) and perf-ledger K-45 (integer dot product) as the root fix. Re-measure first and check which GEMV changes
  since then reach the QAT shapes; QAT E4B's parallel-GEMV rows are unmeasured copies (K-59). `karume-gemma4-qat` stays
  unpublished on HF.
- **Translation remainder of ADR [0130](decisions/0130-project-language.md) (0.15.x)**: the comments under
  `packages/*/tests` and `tools/export-recipes`, the areas the language allow-list marks `until: "0.15.x"`.
- **Playground small item (review notes of 2026-10-28)**: the Wan2.2 / SBV2 pipeline hooks swallow a rejection from disposing an aborted build
  (`use-wan-pipeline.ts` / `use-sbv2-pipeline.ts`).
- **Remainder of the speed / memory wave (filed 2026-09-25; maintainer's ruling = split into two tiers, reference tier / practical tier; within-device determinism MUST hold in the practical tier too)**:
  The finished records of iterations 1 to 3 (contract and groundwork, B3 anima DiT residency, B4 irodori context K/V projection, H-35, K-70, E1 stage 1) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Candidates for the next iteration = the rest of wave 2 (DiT speed: B1 → B2) / wave 3 (memory: rest of E1 → E2 → E5 → E4) / wave 4 (practical-tier kernels: C2 VAE conv i8a8 → D2 → D1 → D3).
  The candidates come from `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md` §9.3 (not tracked by git) and the perf-ledger section of 2026-09-26 (K-59 to K-69, L-20 to L-23).
  Rest of E1 (BiRefNet's f16 series — ADR [0113](decisions/0113-birefnet-weight-series.md)) = golden comparison on a real GPU, derivation of the Lucida f16 tolerance, creation of the sha reference rows for real images, and stage 2 (the i8 plan + i8 series).
- **Remainder of geometry profiles (K-71 — ADR [0115](decisions/0115-geometry-profiles.md) / [0116](decisions/0116-geometry-profile-row-buckets.md) / [0117](decisions/0117-app-geometry-tuning.md)) (filed 2026-09-27 to 10-03)**:
  The finished records (registration of `nvidia-blackwell`, the 7 row-count buckets, integration into applications (ADR 0117, all 7 stages ✅), the Codex review follow-up, 4 small items) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  ② Regenerating `apple-metal-3` with Chrome 154 alone happens the next time a sweep is taken on M2 (ruling 2026-10-02)
  ③ The Pixel 10a (Mali-G715) table is not registered; the re-run on the new version has not been done (research K-70 §15)
- **Remainder of the container-format wave (ADR [0108](decisions/0108-container-format.md)) = stages 4 to 6** (upstream intake of the anima transformer f16 / LoRA runtime A/B / new bit widths, Range fetching and so on —
  the staged-decomposition table of ADR 0108 is the source of truth for what is built and how it is accepted). The records of stages 0 to 3 and up to the 0.13.0 publication (2026-09-25) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
- **Remainder of the cross-model follow-up investigation (2026-09-10 to 11)**: the source of truth for the measurements and the unfinished parts is the [follow-up investigation](research/2026-09-10-codex-mtp-optimization.md#追加-llm-の実行と量子化別比較).
  The acceptance of Qwen3 / MiniCPM5 / E4B / gemma4-qat and the adoption records of K-25 to K-44 are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Remaining = GPU cost attribution on M2; the decision on applying the submission policy and prefill buckets to the model defaults; integration of broad combinations and a cost study of sharing the gate/up input; isolating the cause of the GPU Instance loss in Chrome;
  distribution recipe / source table; long-text and broad quality evaluation; automatic per-shape numeric acceptance of f16 / f32 / RMS128; transfer reduction for general sampling (with temperature, penalty / bias, speculative) and a follow-up test on M2.
- **Continued verification of the 9/11 review** (investigation 2026-09-10, [follow-up record](research/2026-09-10-codex-mtp-optimization.md)):
  Decide whether `artifacts.staged_publication` allows several writers to the same final, and if needed design locking and recovery from interruption.
  The follow-up tests of the GPU edge-channel protection on M2 / browsers and whole-model performance measurement remain.
  Do not turn review suggestions into new kernels without verification.
- **Move `planBackingBudgetBytes` into the common options** (filed 2026-09-07 — consequence of ADR 0095): every family except gemma4 builds its
  Session options from the manifest's `session`, so there is no way to change the budget (the default 256 MiB applies). Add it the same way `onRetry` was
  unified into `FromPretrainedHubOptions`. At the same time, merge the budget range check duplicated in estimate / session-build into one function.
- **Anima: add `text+transformer` to the residency values (filed 2026-10-01, maintainer)**: today `transformer` keeps only the DiT Session across generate calls, and
  text_encoder / text_conditioner (1,396 MiB of weights) redo "read → upload → build the Session → tear down" every time (the default VRAM assumption is "one largest stage" —
  ADR [0112](decisions/0112-anima-transformer-residency.md)). The de facto practice in PyTorch / ONNX Runtime is "once loaded, resident until explicitly freed", and
  the maintainer's policy is "the API shape is explicit release; the default is resident as far as VRAM allows; demote automatically when it does not fit". Put the two text Sessions on the existing eviction
  state machine (trial allocation for the next stage → evict if it does not fit → retry). The default is unchanged. The gain depends on the environment (with local serving on M2,
  re-reading the text path ≈ 1.7 s / image; the slower the fetch, the larger the gain). Before starting, measure the VRAM peak including the resident part on B570 / M2 / Pixel.
- **Remaining test coverage (filed 2026-09-05)**: the copy branch in `#encodeTimedChunk` of `SubmitScheduler`
  (`packages/runtime/src/gpu/submit.ts`) is still untested.
- **Isolation experiment for Metal `--diagnostics`**: an A/B of which dominates, the number of query sets alive at once or the
  backlog of `destroy()`. The relevant section of [known-issues](known-issues.md) is the source of truth for steps ①② and the fix candidates.
  It needs real hardware, so the user runs it.
- **The two medium-priority items of the diff review's deferred table** (source of truth = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`
  — not tracked by git): ② W-G4-4 record the origin of the chunk limit in the provenance field `sym_max` (**rides along with a re-export** — on the occasion of wave b or a series
  update) ③ the missing path for ADR [0033](decisions/0033-vae-fixed-tile-decode.md) decision 5 "the TS side compares the geometry
  itself" (a ruling on whether to emit one geometry JSON or to amend decision 5 to the actual state).
- **Peak VRAM column on the model card (filed 2026-09-04)**: `karume dist` calls the TS-side estimate
  (the `prepareContainer(...).estimate()` family) when generating the card and adds a peak VRAM column to the quant table. Today's
  card shows only the stored bytes, so readers cannot judge whether the model runs on their GPU.
- Follow-up test of the RAM peak of the HF path in Chrome (measured on the Deno side —
  [research 2026-09-24](research/2026-09-24-part-length-ram-peak.md)).
  **Browser performance measurement is dropped from this wave (a) (maintainer ruling 2026-09-04)** — what a browser can measure is
  only the implementation difference between Dawn / wgpu, which does not bear on adopting kernel candidates. Alternative = add a feature to the TS packages that collects
  performance information, and gather the results of users running a sample collection (name and location undecided) on several environments
  (**filed only** — the items collected, the location and the opt-in form are not designed).
- Re-assessing `err.cause` on Pixel (8 GB class Android Chrome) — [known-issues](known-issues.md).
- **Follow-ups of accepting ADRs 0127 / 0128 / 0129 / 0131 (filed 2026-10-29, the maintainer's ruling 34 a — each ADR's addendum of
  2026-10-29)**. Items under `packages/runtime` wait until the pending nvidia-blackwell geometry derivation lifts the freeze.
  - Gate C2 counts `convCompute` (ADR [0128](decisions/0128-default-preset-original-weights.md) open point 5): `defaultPresetViolations` in
    `packages/runtime/tests/distribution_gate_test.ts` counts only `linearCompute` / `attentionCompute`; add `convCompute: "a8"` (ADR 0133)
    with fault-injection cases. No GPU; every current mirror already passes.
  - Borrowed bf16 weights fail the residency check (ADR [0131](decisions/0131-bf16-weight-execution.md) open point 3): `expectedSharedSeat`
    (`packages/runtime/src/runtime/weight-residency.ts`) classes bf16 as `raw`, so borrowing a bf16 initializer always throws (loudly). Treat
    bf16 like f16 there and add a bf16 borrow case to `gpu_borrowed_context_test.ts` (GPU). No shipped family borrows bf16; if it misses
    0.15.0, record the limit in [limitations](limitations.md).
  - Geometry sweep identity check for a bf16-stored linear (ADR 0131 open point 2, low priority), after the pending derivation.
  - A check that `version` and `exports` agree between each published package's `package.json` and `deno.json` (ADR
    [0127](decisions/0127-node-dawn-toolchain.md)): JSR publishes from `deno.json`, `published-smoke` reads `packages/models/package.json`.
  - Move the Node read-strategy default into `@hdae/fetch-cache` (ADR 0127 open point, after 0.15); hub's probe in
    `packages/hub/src/sources/hf.ts` can then go.
  - Deno's resource / op sanitizers have no replacement under Vitest (ADRs 0127 / [0129](decisions/0129-test-layout-and-lanes.md) open
    point 5): record the loss in [limitations](limitations.md); file a GPU resource-leak check only if a leak shows up.
  - The order in which the removed 0.14.0 tools and examples (v0.14.0 tag) come back as `apps/playground` pages (ADR 0127 decision 5).
  - Gate the test import rules of ADR 0129 (open point 4): no runtime / hub / models test imports `packages/e2e`; only the runtime GPU
    tests and the playground import `@karume/kernel-checks`. Both hold today, but `dependency_graph_test.ts` checks published members only.
  - Move the kept optional asset cases (`gemma4_config`, `gemma4_tokenizer`, `gemma4_chat`, `wan_pipeline`, `census_table` in
    `packages/models/tests`) into `packages/e2e` (ADR 0129 decision 1).
  - Run the playground test suite (`pnpm --filter @karume/playground test`, 45 files outside the lanes and `vp test`) in CI and in
    [release-runbook](release-runbook.md) §1 (ADR 0129 decision 2).
  - Small doc edits: ADR 0001 still says npm publishing is open (settled JSR-only, 0127 addendum 2026-10-28); ADR 0086 quotes 0127
    decision 2's old "read local bytes" wording; ADR 0005 decision 4 names an ADR 0127 open point as the real-browser acceptance home;
    `packages/test-support/package.json` describes "Deno-shaped test and fs helpers"; `packages/e2e/README.md` says every test there needs
    assets and a GPU (the three core gates and the `*_host*_test.ts` files do not); ADR 0120 points at `WAN_DEFAULT_QUANT` in
    `tools/export-recipes/wan/distribution.py:343` (now `:440`); release-runbook §1 omits the recording mode `KARUME_WGSL_SNAPSHOT=write`.
- **Barrel imports keep every family's source table (filed 2026-10-28 — found in the naming design of ADR [0136](decisions/0136-models-public-naming.md))**:
  importing anything from the `@karume/models` barrel, even one shared helper such as `encodeWav`, keeps all nine
  source tables (twelve `hdae/karume-*` entries) and a few top-level expressions a bundler cannot prove pure — about 1.4–1.7 KB minified
  (≈0.8 KB gzip) over the same import from a subpath (rolldown 1.2.13, 2026-10-28). Bundle size of barrel consumers only; no
  behaviour change; independent of the naming. The retained expressions: `Object.freeze([...])` in `src/anima/config.ts`,
  `src/anima/residency.ts` and `src/gemma/pipeline.ts`, the spread of a call in `src/gemma/speculative.ts`, the
  `indexOf` / `map` calls in `src/vowel-detector/postprocess.ts` and the object spread of `KARUME_SOURCES` in
  `src/sources.ts`. Fix: `"sideEffects": false` in `packages/models/package.json` (measured: the barrel then costs
  within 5 B of the subpath) once its survival through JSR's npm layer is verified (the open point of ADR
  [0037](decisions/0037-karume-monorepo.md) §4), or `/* @__PURE__ */` on those expressions.

- **MOSS-Image (branch `wip/moss-image`, not part of 0.15 — ADR [NNNN](decisions/NNNN-moss-image-dit-vae.md), proposed)**: the DiT and the untiled FLUX.2 VAE are done on the branch (text as precomputed embeddings, the presets `f32+dit8` and `f32+dit8-a8-attn8-online`, an internal family with no subpath or distribution — stages 0–5 of the ADR; the measurements are in [research 2026-10-28](research/2026-10-28-moss-image-dit-vae.md)). The branch stays apart from `wip/0.15` and gets the ADR's number when it is taken in.
  Next = the text encoder (Qwen3.5-9B; its Gated DeltaNet layers need the scan op of ADR [0135](decisions/0135-gated-delta-rule-scan.md)) — publication waits for it (ADR NNNN decision 12). The default preset is `f32+dit8-a8-attn8-online` (the maintainer's visual review passed — ADR 0128 decision 1); the manifest carries it. Its speed candidates are perf-ledger F-10 to F-12.

- **gemma4 prefill in w4a8 (perf-ledger K-60, maintainer 2026-10-17 "in the backlog")**: design of an acceptance table that keeps decode on the M=1 GEMV and puts only prefill (chunks of M=768) on `linearCompute: "a8"` × i4 (switching by M). Today's a8 falls to the GEMM skeleton even at M=1 and is 8.45 times slower than the GEMV ([research 2026-09-19](research/2026-09-19-qat-speed-recon.md)), so opening the acceptance table is not enough.

- **Root fix for the rows that collapse under umT5's a8 (perf-ledger K-81, filed 2026-10-18, maintainer: "root fix first, not even opt-in")**: per-token int8 crushes the remaining components in the rows of function words (spaces, commas, articles, suffixes) ([research 2026-10-18](research/2026-10-18-a8-umt5-sbv2.md) §6; the main cause is token-specific outliers in the input of the FFN exit `wo`; mixing a8 per linear or layer does not remove it). Candidate = per-row mixing (only rows whose crest exceeds a threshold take the f32 path — row routing in the runtime's `linear-i8a8`). An equivalent transform on the export side only helps fixed channels. Once this is closed, re-evaluate umt5's `i8-a8` (the adoption line and the work needed are in K-81).

- **Scope the geometry sweep's `caseSet` per op family (filed 2026-10-27; the maintainer's ruling 2026-10-26: an amendment of ADR 0117 decision 8, better long-term, later)**: `caseSet` (`sweepCaseSetId()` — the provenance field of ADR [0117](decisions/0117-app-geometry-tuning.md) decision 4, carried in the sweep record and checked by the generator under decision 8; the change touches decisions 4 / 8) is one hash over every case of `SWEEP_CASES`, so adding one op family's cases moves it for every table: every embedded table's provenance and every saved table mismatches (`geometryProfileMismatch`) and needs a re-sweep or a re-derivation, and a table whose records have no case of a class still carries the new `caseSet` — after perf-ledger V-5, `nvidia-blackwell` and `apple-metal-3` carry it although their `i8a8.conv3d` is the default by absence ([ADR 0115](decisions/0115-geometry-profiles.md) addendum decision 11).

- **Execution form with f16 activations (w16a16) (filed 2026-10-11, maintainer: "will do it next time")**: today the runtime can make only the weight storage f16; compute and activations are f32 (only attention has `attentionCompute: "f16"` — ADR 0006 rejected f16 activations). The value is the intermediate memory, which pays off where intermediate tensors take most of the VRAM, as in BiRefNet's 2048² (ADR 0113). It requires `shader-f16`. Bit-exact agreement with upstream w16a16 is not expected (the kernels differ — handled at the numeric level of the ADR 0005 addendum of 2026-10-11). Preliminary research: [research 2026-08-04](research/2026-08-04-intermediate-f16-design.md).

- **Decouple the test helpers results.ts / reference.ts from GPU discovery (filed 2026-10-11, Codex deep review R8-008, maintainer's ruling = file only)**:
  `ENVIRONMENT` in `packages/runtime/tests/helpers/environment.ts` is fixed by calling `requestAdapter()` at module evaluation.
  `helpers/results.ts` and `helpers/reference.ts` import it as a value. So `results_writer_test.ts` / `reference_fixture_test.ts`, which say
  "no real GPU needed", also search for an adapter at import time. They do not take a device, and on a machine without an adapter
  the value stays null and they proceed, so the tests do not fail. The only real cost is the `requestAdapter` call. Both functions already take `options.environment`.
  What remains is that the value import of `ENVIRONMENT` runs at module evaluation. Lazy acquisition does not fix it — `openReferences`
  reads the environment key synchronously when tests are registered, and `openResults` is also a synchronous API. The fix would split the two into "a body that takes the environment as an argument" and
  "a thin wrapper that binds the default `ENVIRONMENT`", and point only the CPU tests at the body (the GPU tests' callers do not change).
  After the fix, confirm that with `navigator.gpu` undefined, importing the two tests does not call `requestAdapter`.
- **Out-of-scope remainder of the Codex deep-review fix wave (filed 2026-10-11)**: of the observations that the fix wave's implementers and reviewers recorded as out of scope,
  those not in backlog / known-issues and confirmed still present on main on 2026-10-11 (one per line; the IDs in parentheses are
  findings in `.claude/reviews/2026-10-04_de127de1/` (not tracked by git)). The order of work is undecided.
  - The estimate's copy of the states form (review of ruling 6): `transientSlotBytes` in `packages/runtime/src/runtime/estimate.ts` counts the row statistics with `allocBefore = base + 1`, but execution (`buildStateAttention`) allocates both S and the row statistics before ① QK. Depending on the placement, `workspaceBytes` can disagree with the actual plan (the shapes tried agree).
  - Row-block estimate of decomposed attention (review of ruling 6): without `maxStorageBufferBindingSize`, the count in `rowBlockAttentionSteps` still divides by the WebGPU core default. As with fused attention, align it with the rule that requires the limit whenever the count depends on it (ADR 0124 decision 7).
  - Anima's per-stage estimate (review of ruling 6): `stageNeed` in `packages/models/src/anima/pipeline.ts` does not pass the quant preset's `attentionCompute` / `attentionScoreStorage` to `EstimateOptions` (the VAE's fused attention is counted as f32; for today's shapes the total does not differ).
  - Disposal of the Gemma pipeline and the fork wrapper (review of ruling 8): in `#hand` of `packages/models/src/gemma/pipeline.ts` (shared by `sequence()` and `fork`), the dispose of the inner after the await is not ordered against the stages of the pipeline's dispose (drafter Session → target Session). Today it only gets queued first because of the number of awaits.
  - safetensors read helpers: `readBuffer` / `viewOf` / `floatsOf` live in `packages/e2e/tests/helpers/wan-ti2v-dit.ts`, made for Wan's DiT (and `helpers/umt5-golden.ts` has its own `readBuffer` / `floatsOf`); nine test files import them from there. Move them to a general `helpers/safetensors.ts`.
  - `tools/export-recipes/tests/test_optional_group_imports.py` uses the distribution name as the import name, so it cannot catch module-level imports of packages whose names differ, such as pillow → PIL (the check's doc also admits this miss in the loosening direction).
  - Backing difference between Deno and Chrome: the cause of 5.97 GiB vs 4.80 GiB for Wan2.2's 121 frames with the practical preset (ADR 0121) is not investigated. The two candidate explanations are the geometry profile (Deno uses `default`) and `maxBufferSize` (how regions are split) (ADR 0124's risks; material recorded in stage 4b).
  - Flat indices of the GEMM family (rest of R2B-009): `gemmParams` in `packages/runtime/src/kernels/gemm.ts` only checks m / n / k individually as u32 and does not reject shapes where matmul / bmm's `M·K` or `K·N`, bmm's flat indices multiplied by the batch (`batch·M·K` and so on) or linear's activation-side `m·k` exceed 2^32 (linear's weight side `n·k` is checked by `linearParams`).
  - Flat indices of deform_conv2d (rest of R2B-009): `deformConv2dParams` in `packages/runtime/src/kernels/deform-conv2d.ts` checks only the output element count as u32 and does not reject shapes where the flat indices of the input, weight, offset or modulator exceed 2^32 (there is no check corresponding to conv3d's `assertConvFlatExtents`).
  - In-row loops of row kernels (rest of R2B-007): the in-row loop `i = i + workgroup size` of reduce / RMS / softmax / layer_norm and others (for example `packages/runtime/src/kernels/rms-norm.ts`) wraps around and never ends on rows where `dim + workgroup size` exceeds 2^32, and since this does not go through the dispatch count, the `gridStrideWorkgroups` check cannot block it — each row kernel's params need a `dim + workgroup size ≤ 2^32` check.
  - When Wan copies `latents` (rest of R6A-005 / former P-3): `generate` in `packages/models/src/wan/pipeline.ts` and `ti2v-pipeline.ts` runs its checks and the copy after waiting its turn in the serialization chain, so if the caller rewrites `latents` during the wait, generation uses the rewritten values — to guarantee the values at call time, copy outside the chain (at the entry of `generate`).
  - Added vocabulary of the T5-family tokenizer: `packages/models/src/text/t5-tokenizer.ts` uses the shared `parseAddedTokens` (which checks only the i32 range) and does not check whether an added token's id is below the vocabulary row count or matches the vocabulary row of the same spelling (it lacks the check that irodori's `parseIrodoriAddedTokens` has; affects both anima's T5 and Wan's umT5; the distributed assets satisfy both).
  - f32 overflow of SBV2's `z_p`: even when `noiseScale` is finite in double precision, `buildZp` in `packages/models/src/sbv2/host/latent.ts` puts products above the f32 maximum (for example `noiseScale = 1e39`) into a `Float32Array`, turning them into ±Inf, and passes them to the voice graph as they are (the same hole as the `z_noise` check; the GPU outputs `m_p` / `logs_p` are mixed in, so a design for which error type is thrown is needed).
  - Output dtype of BiRefNet / Depth Anything (output side of R6B-013): `packages/models/src/birefnet/pipeline.ts` and `depth-anything/pipeline.ts` check only the output shape before construction, not the dtype, so a non-f32 graph fails only at `asF32` after one run.
  - A helper for the checks after GPU acquisition (root of R6A-006): 8 families — Wan (`wan/family.ts`) / Anima / SBV2 / Irodori / BiRefNet / Depth Anything / SigLIP2 / vowel-detector — each hold the same check of features and `requiredLimits` after GPU acquisition in their construction code (`pipeline.ts` / `family.ts` / `admission.ts`) — a candidate for one helper (a check missed in one family caused R6A-006).
  - Quadratic copying in hub's `parseWeightEntry`: `parseWeightEntry` in `packages/hub/src/manifest.ts` copies the accumulated object with a spread for every dtype label, so the work grows with the square of the label count, and the label count of one entry has no count limit (a scale limit in the sense of ADR 0041 decision 7) (only the 1 MiB limit on the whole manifest applies) — build it at once with `Object.fromEntries`, or add a limit on the label count.
  - Integer-valued floats in Python integer fields (rest of R7-013): `verify._as_reader_index`, `states._window`, `migrate._count` and the `pipelineConfig` capacity in `limits` under `tools/exporter/src/karume/` reject integer-valued floats (for example `4.0`), diverging from the set that the TS parser (`Number.isSafeInteger`) accepts.
  - Unicode `\d` in Python regular expressions (rest of R7-014): `PIPELINE_RE` in `tools/exporter/src/karume/migrate.py`, `_PIECE_PATTERN` in `legacy.py` and the match of sequence-numbered file names in `container.py` let `\d` accept non-ASCII digits (for example the full-width `１`) — use `[0-9]` or `re.ASCII` (`_REPR_PATTERN` in `container.py` only reads repr output, so it is out of scope).
  - The copy of the repository name rule in models: `REPO_SEGMENT_RE` / `assertRepoSpelling` in `packages/models/src/hub/repo-ref.ts` hold a copy of the same rule as hub's `isRepoName` (`packages/hub/src/manifest.ts`) — making it one requires exporting the predicate from hub's mod.ts (an addition to the public API).
  - Duplicated range checks of the declared size in hub: `packages/hub/src/sources/local.ts`, `packages/hub/src/container.ts` and `packages/hub/src/fetch.ts` each have a range check of the same form — a candidate for one internal helper.
  - Dimension checks of the CPU references for state_append / readonly: `referenceStateAppend` and `referenceStateAttentionReadonly` in `packages/runtime/src/reference/state-attention.ts` check dimensions only with a `< 1` comparison, so NaN or non-integer dimensions pass through (the states-form entry `assertShape` has been fixed to check for an integer first).
  - Length 0 in the CPU reference's `allclose`: `packages/runtime/src/reference/allclose.ts` returns `pass: true` for two length-0 inputs — whether to allow empty input or throw has not been ruled on, and `packages/runtime/tests/reference_ops_test.ts` pins only the currently observed value (review L-R3-4).
  - Writes to pad rows in state attention: the variants in `packages/runtime/src/kernels/state-attention.ts` and `state-attention-stats-pv.ts` each write the same guard separately — a candidate for one shared WGSL fragment (today a static test pins the form of all variants, so a recurring mismatch would be detected).
  - Bilinear interpolation weights: unlike torch's `guard_index_and_lambda`, λ in `packages/runtime/src/kernels/upsample-bilinear2d.ts` is not clamped to [0, 1], so only for shapes where one axis exceeds 2^24 and the source coordinate goes past the edge might the end points not be exact (not reproduced; out-of-range index reads are blocked).
  - Pinned value of Wan's ftfy badness: the TS side (`packages/models/src/wan/text/prompt-clean.ts`) pins the sha256 of the asset's `badness` with `FTFY_BADNESS_SHA256`, but the recipe's writer (`tools/export-recipes/wan/prompt_clean.py`) does not check the same value — after bumping the ftfy version and rebuilding, the first sign is a red TS test.
  - Test subjects of the exporter: around lines 995 and 1646 of `tools/exporter/tests/test_emit.py` the tests use the graph of `int4_weight_graph(conv=True, conv_groups=2)` with its default Cout 3 (not divisible by groups 2 — a violation of the conv1d contract) — they do not go through the public path, so they stay green.
  - Random numbers in the tokenizer test: in `nextRandom` of `packages/models/tests/gemma4_tokenizer_test.ts`, the plain multiplication `state * 1103515245` exceeds 2^53 and the low bits are lost, so the random choices vary little — use the `Math.imul` form.
  - Type arguments of DataView / ArrayBufferView (outside the lint rule `karume/typed-array-buffer`): since TS 5.7, `DataView` and `ArrayBufferView` also take a buffer type argument like TypedArray, and omitting it gives the default `ArrayBufferLike`. src still has 4 bare `DataView` type annotations (`packages/runtime/src/gpu/attention-dp4a-canary.ts`, `packages/runtime/src/format/container/header.ts` and 2 in `packages/models/src/audio/wav.ts` — `ArrayBufferView` has 0) — whether to add it to the rule's set of names is a separate ruling.
  - hub: decide whether the form where one manifest puts the same path with the same sha256 at different revisions of the same cross-repository target (two references,
    one cache entity — the hub cache inventory section of [limitations](limitations.md)) is reworked to match by entity key
    (recommendation a of Codex deep review R4-001), or rejected at parse.
  - hub: rejecting a cross-repository reference that points at its own repository applies only to the parse in `loadManifest` from HF. The distribution gate
    (`distribution_gate_test`) and `local-dist-server` (now in `packages/test-support`) use the public
    `parseManifest(text)`, so a manifest that passed on the local mirror can first fail after it is uploaded to HF (`karume dist` already rejects it —
    ADR [0038](decisions/0038-manifest-v1.md) §7 addendum 2026-10-28). Open: whether the gate checks with the destination repository name too, for a mirror the exporter did not build.
- **Merge the duplicated helpers of Wan2.2's T2V / I2V end-to-end pipeline e2e tests (filed 2026-10-10, maintainer's ruling = later)**: `e2e_wan_ti2v_pipeline_test.ts` (T2V) and
  `e2e_wan_ti2v_i2v_pipeline_test.ts` (I2V — ADR 0121 stage 9b) each hold the same small utilities — fixture loading (`readFixture`), difference computation and formatting (`difference` /
  `formatDifference`), bit equality (`bitsEqual`), RGB extraction (`rgbBytes`), counting non-finite values (`countNonFinite`), prompt resolution (`textOf`), band derivation
  (`bandOf` / `roundUpTwoDigits`), formatting of observations (`observe` / `formatObserved`). Move them into `helpers/`. After the move, confirm on the GPU that both bands and sha rows are unchanged
  (the same procedure as when stage 7's A/B check was moved into `helpers/wan-ab-gate.ts`).
- **Return Wan2.2's 121-frame self A/B check to the default lane — dealing with the duration of the full verify (filed 2026-10-09, maintainer's ruling = later)**: ADR 0121 stage 7's
  self A/B check is held at two points, 33 frames (S = 7,920) and 121 frames (S = 27,280) (the same "default and longest" rule as Wan2.1 — attention's 8-bit error grows
  with row length). The 121 part takes about 6 minutes per run (estimate), and putting it in the default lane would lengthen the full verify of about 118 minutes, so it is opt-in for now
  (`KARUME_WAN_TI2V_121F=1`). Collapse on long rows is caught only when the opt-in is run. Measured (2026-10-09, RTX 3080 Ti): the 121 A/B check takes
  5 min 53 s. The stage 7 parts put in the default lane (the 33 A/B check 2 min 1 s, one submit of the DiT e2e with the practical preset about 55 s, the seed path of the end-to-end e2e with the practical preset about 2 min 30 s)
  lengthened the default lane by about 5.5 minutes. Measures (shortening the duration, changing where it lives) are to be considered separately.
- **In-flight submit bound for enqueue in browsers (filed 2026-10-07, maintainer's ruling = later)**: ADR 0123 put the bound on run only
  (if enqueue waits, the call order against another Session's enqueue breaks). To bound enqueue as well, first add a mechanism that runs the bodies of the enqueues in a batch one at a time
  in acceptance order (a per-batch FIFO), then wait at the end. The same mechanism also closes the existing case where the first enqueue (a body with a derivation `await`) can be
  overtaken by a later enqueue of another Session. When to start: when running long batches (video VAE, generation loops) in Chrome
  shows congestion in submission or device lost. There is no actual failure so far (the 121-frame VAE, about 20,000 submits, also completed).

- **Two anima parts fragmented by the 0.13.0 re-upload (filed 2026-09-24; ruling 2026-09-24 = publish as is and act after measuring the real download speed)**: in the fragmentation table of `hf-upload.zsh`,
  among the 255 MiB-class parts, 2 of anima (`text_conditioner` part 3 of `anima-turbo-v1.1` / `anima-v1.0`, 3.6 MiB/term) fell below 10 MiB/term
  (the [research](research/2026-08-09-xet-fragmentation.md)). Repositories with a published pin do not use delete → recreate (runbook §2 MUST NOT), so
  decide whether to accept it based on the measured download speed, or another remedy (whether it resolves itself the next time a re-upload changes the bytes).
  In the local mirror (0.13.0),
  part 3 of `text_conditioner` has zero zero-pages in all 5 variants, so the zero-run mechanism does not apply. The two fragmented parts
  (`anima-turbo-v1.1` / `anima-v1.0`) are byte-identical to each other and differ in content from part 3 of the other 3 variants, which were healthy.
  Re-checked after the 0.15.0 re-upload (2026-10-30, global dedup off, 0 CAS queries): the two parts read 3.7 MiB/term
  (69 terms over 6 xorbs, all written by that upload) and the new `shared/text_conditioner/model.bf16-00003` 2.4 MiB/term —
  sharing between byte-identical or near-identical parts inside the repository, not cross-repository reuse; every other
  part of the five re-uploaded repositories is at 25 MiB/term or more.

- **Revisit the structure of the manifest (`karume.json`) — make optimization flags and presets easier to handle (filed by the maintainer 2026-10-06)**: today the bundle of execution knobs
  is held in the single `session` of a quant preset, and the preset name doubles as the bundle's name (ADR [0074](decisions/0074-quant-seat-naming.md) /
  [0110](decisions/0110-practical-tier-numerics-contract.md) / [0111](decisions/0111-session-options-composition.md)).
  We want a form in which optimization flags and presets (bundles) are easier to handle. What belongs where (storage dtype, execution knobs, bundles, per-component declarations and so on)
  is to be sorted out later. The material is the open points of the inventory of how fast paths are expressed (2026-10-06, `.claude/reviews/2026-10-06_fast-path-inventory/`, not tracked by git)
  — there is no per-component session (the fast paths of the VAE / text encoder cannot be declared), the keys of new kernel variants, and whether changing a bundle
  changes an existing preset or creates a new one. **Work starts later** (maintainer 2026-10-06 — Wan2.2's optimization proceeds within today's `session`).
- **Candidate for bringing HTTP Range fetching (ADR 0108 stage 6) forward (filed 2026-09-24; decided after the release)**: the condition for bringing it forward in ADR
  [0109](decisions/0109-manifest-v5-container.md) decision 7 is "in the RAM peak harness of stage 2, the cold
  peak exceeds 'part length + overlap'". Stage 3e removed the duplicated holding, but the cold peak of the scan type (Deno via HF)
  still exceeds part length + the largest block. What remains is hub's single holding slot and the part waiting for GC; the cause has moved to whole-part
  reads (the fetch granularity). The condition formally still holds ([research record](research/2026-09-24-part-length-ram-peak.md)
  item 7, ADR 0108 addendum 5). It applies only to Deno via HF. Its motivation differs from that of parked "hub Range parallel + prefetch"
  (a fragmentation measure).
- **With an unverified source (local), host PLE row reads apply sha256 to the whole block every time (filed 2026-09-24)**:
  host PLE reopens the reader for every row read, so with an unverified source one row read costs a whole block read and
  sha256 every time (28 × 32 MiB per run). The longer gemma4 run on M2 in stage 3e has been attributed to this and to the cost of memory allocation
  ([research record](research/2026-09-24-prerelease-gpu-measurements.md) item 1). The fix is either for the opened container to
  remember the verified blocks and do range reads from the second time on (a verified cache), or to digest each block only once.
  Deciding in an ADR how to handle a local file being replaced during reading (TOCTOU) comes first.
- **Fence count of PLE's in-memory container (filed 2026-09-24)**: the PLE of the GPU-resident preset hands one piece = one part to
  the in-memory container, so Session construction raises one fence per piece
  (the part split in `packages/runtime/src/format/container/memory.ts`). Measured (QAT E2B / E4B —
  [research record](research/2026-09-24-prerelease-gpu-measurements.md) item 2): bundling pieces into parts brings E2B's fences from
  37 to 6 and PLE construction about −0.2 s, at the cost of +192 MiB of VRAM during construction. **Ruling 2026-09-24 = do not bundle for now**
  (re-confirmed in the pre-release review of 0.13.0; the implementation size is the conclusion of item 2 of the research record).
  Alternative (filed 2026-09-25, not started): without bundling, let the fences run **one ahead** (start supplying the next piece before the previous fence completes).
  Staging then holds two parts alive at once, so running ahead is allowed only when the total bytes are within the limit (ADR 0108 decision 9:
  "limited by both count and total bytes"). With 256 MiB weight parts this would be +256 MiB, more than the +192 MiB of the bundling option, so
  the target is only the PLE preset (32 MiB pieces). Kill: PLE construction improves by less than 0.3 s (the adoption of the candidate is judged together with L-16 in [perf-ledger](perf-ledger.md)).
- **The 25 `gemma4_qat` fixture tests are skipped wholesale in CI (filed 2026-09-25)**: `series_fixture.py` and `test_fixed.py` under `tools/export-recipes/gemma4_qat/tests/`
  have a module-level `pytest.importorskip("transformers")`, and `transformers` exists only in the
  `gemma4-qat` group, so in CI's recipes job, which runs with the default sync, the 25 tests (test_audit 1, test_fixed 12,
  test_plan 12) are skipped at collection. To decide: a) stub `transformers` and keep 1 to 2 upstream comparisons
  (mind `isinstance(QuantizedEmbedding)` in `checkpoint.py`), or b) add `--group gemma4-qat` to CI
  (a trade between CI time and the strength of the upstream comparison).

- **Leftovers of the code-quality wave (filed 2026-09-22 — source: the same wave in the
  [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md) (after the 0.12.0 release))**:
  - Where to put the four functions that take `gemma.PipelineOptions` (`assertSpeculative` / `resolveGemma4PleResidency` /
    `buildGemma4Program` / `speculativeSetup`) (creating `pipeline-options.ts` was postponed by the ruling of 2026-09-21).
  - The checks of the Unicode range tables differ in wording and remain in each family (only the binary search was moved to `text/code-ranges.ts`).
  - Splitting `reference/ops.ts` and `ops/shapes.ts` (info — no mutable state and no cycles, zero real harm).

- **Remainder of decode speed (moved from now on 2026-09-20)**: H-27 stage ② (run-ahead, an opt-in exception to ADR 0066, greedy only, expected Deno −5 / Chrome −2.2 ms),
  small items K-48 stage 1 (rms_norm→SRQ fusion, 70 instances, 0.19 ms) / K-49 (slice aliasing) / K-50 (k+v concatenated GEMV), K-46 (int8 KV — a memory item).
  Revival condition: when decode speed becomes the topic again (H-27's prerequisite = H-28's GPU-resident preset is done; re-measure the TTFT attribution of research 2026-09-19 §15.2 first
  — "the prefill arena path" is not in hit runs, as corrected in the addendum of 2026-09-25).
  The adoption of candidates and the kill criteria are in [perf-ledger](perf-ledger.md), the attribution in [research 2026-09-19](research/2026-09-19-qat-speed-recon.md).

- **Ruling on the three contracts that block large in-layer fusion (filed 2026-09-19)**: of the dispatch gap to WebML (1,132 → about 316), about 480 are
  blocked jointly by `windowTouchesState` MUST (ADR 0067), FusedStep single output MUST (ADR 0068 decision 1) and the portability judgment of the atomic last-arriver merge.
  The individual candidates (attention→SRQ, residual add→SRQ, attention in 1 dispatch, Q/K/V with 3 outputs) were all closed by refutation, so a design ruling on whether to relax the contracts comes first
  ([research](research/2026-09-19-qat-speed-recon.md) §9.1). Do not file speed candidates before the ruling.
- **Remaining items of the video-generation foundation (investigation 2026-09-10; ADRs [0118](decisions/0118-wan21-video-generation.md) /
  [0119](decisions/0119-wan-umt5-gpu-text-encoder.md) / [0120](decisions/0120-wan-dit-w8a8-seat.md))**:
  - MiniMax H3 (the maintainer's final goal, 2026-10-02): H3 stays at a structural study + per-component candidates
    ([research](research/2026-10-02-video-gen-recon.md)). H3 exists and is open-weight (HF MiniMaxAI/MiniMax-H3; by the letter of the terms Japan is a licensed region (not legal advice)),
    but with 33B dense + Qwen3-VL-32B + VAE 2.6B it is about 144 GB per task and assumes offload even natively (a measured example of 112 s on an RTX 5090);
    there is no browser WebGPU case.
  - In safetensors 0.8.0 the key order of `__metadata__` in the header changes per process, so regenerated fixtures are not byte-identical (seen in vae_chunks.band / accept; the tensors are identical) — check regeneration by the tensors' sha, or fold the metadata into one key (filed 2026-10-03).
  - Wan small items (filed 2026-10-03, diff review of the ADR 0118 wave — the IDs in parentheses are items of `.claude/reviews/2026-10-03_diff-review-wan/` and `2026-10-03_codex-review-wan/` (not tracked by git)):
    ① measuring the GPU time per submit with the submit layout of normal execution (M8) (the "1 submit ≤ 1 s" check and its recorded values come from timing mode (`gpuTiming`, 1 dispatch = 1 pass), and the chunk boundaries after backing can differ from normal execution — ADR 0118 stage 8 result "scope of the submit check"; needs a GPU)
    ③ a fourth case in golden `conv3d_block` that exercises the v4 and m-tile 64 variants (F4) (the current 3 cases are Wout 5 / 5 / 6 and Cout 6 / 3 / 2 and hit only the scalar variant and m 32. v4 and m 64 are exercised only by the GPU parity tests; needs golden regeneration and confirmation on the GPU)
    ⑤ the CPU-only DiT host test reads and copies the whole 2.6 GB reference (TG-11) (`reference.full-{band,accept}-s14040-t0999` are 2,598,638,524 B each; `ArrayBuffer.slice` in `readBuffer` temporarily doubles it; RSS not measured)
    ⑥ the note separator " / " in results.json collides with " / " inside labels (TG-18) (labels in general, such as `formatNormalized`; no consumer splits it today; a fix would go on the separator side)
    ⑦ narrow the entity-candidate check of `prompt_clean` into an exact check that "rejects only when the string actually changes", using the name table (Python's `html.entities.html5`, 2,231 names) (accepting sequences that do not form an entity, such as `R&D` — ADR 0119 ruling 2026-10-03, maintainer: "a workaround if it can be avoided"; once someone asks for it).- **Remaining small items of the Wan2.2 acceptance (ADR [0121](decisions/0121-wan22-ti2v-5b.md))**:
  **Adjacent small items of stage 9 (filed 2026-10-10, review)**:
  (a) the distribution assembly (`tools/export-recipes/wan/distribution.py`) does not check the input/output contract of the VAE encoder's 3 containers (input names, symbols, the shape chain pre → attn → post) (the decoder's 2 containers have the pair check `assert_vae_chunk_pair` — a series with only one of them rebuilt passes assembly and is first rejected at admission in the user's `fromPretrained`; filed 2026-10-10, review G4-01, a candidate for the fix wave)
  **Adjacent small items of stages 5 / 6 (filed 2026-10-05)**:
  (a) the tile reference reader and helper functions (`readTileFixture`, `planMeta`, `compare`, `absMax`, `poisonFrames`, `roundUpTwoDigits`) are duplicated across 3 e2e tests → into a shared test helper
  (b) helpers such as the recipe's `_peak_rss_gib` are duplicated in 3 places → into `_shared`
  (c) the existing 5 DiT inputs have no dtype check (only the 2 inputs added for 2.2 check the dtype)
  (d) rounding up to 2 significant digits in `bandCandidate` of `e2e_wan_ti2v_dit_test.ts` can round one unit too high because of floating-point error (`roundUpTwoDigits` in the end-to-end e2e `e2e_wan_ti2v_pipeline_test.ts` was fixed by inserting `toPrecision(12)`. `roundUpTwoDigits` in the tile e2e `e2e_wan_ti2v_vae_tiles_test.ts` and in stage 4's chunk e2e `e2e_wan_ti2v_vae_tiles_test.ts` does not have it — it can be fixed together when moving into the shared test helper of (a))
  (e) the messages of the `pipelineConfig` reader have no owner prefix (the message cannot tell whether the 2.1 or the 2.2 class failed)
  (f) the 2.1 and 2.2 end-to-end e2e tests hold copies of the same helper functions (`readFixture`, `difference`, the contact-sheet image and so on)
  (g) there is no host test combining the admission of the gpu path (`admitWanText`) with the 48 ch, 7-input DiT (stage 6 results)
  (h) `video_size` in `tools/export-recipes/wan/pipeline_ref.py` hard-codes 16 ch and compression 8 (unusable for 2.2 — stage 6 results)
  (i) `planWanRequest` checks only the element count of the injected `latents`, not the shape (stage 6 results)
  **Adjacent small items of stage 4 (filed 2026-10-05, deferred in review)**:
  (a) move `filePresent` into a neutral test helper (today the VAE helper `wan-ti2v-vae.ts` imports it from the DiT helper `wan-ti2v-dit.ts` — the VAE tests depend on importing the DiT helper)
  (b) make the drop-cache fault injection of the shared helper `wan-vae-chunk-loop.ts` assert that the named cache exists in the graph (with a missing name it silently injects no fault — run both the 2.1 and the 2.2 lane)
  (c) a closed form of DupUp3D (put up0 / up1 on the fused upsample2x) — perf-ledger K-77
  (d) in the step "the fault-injection loop matches the product path at Uint32 when there is no fault" of the 2.1 VAE chunk e2e (`e2e_wan_vae_tiles_test.ts`), refill the frames with poison values before `decodeWithFault(none)` as well and assert 0 non-finite (the 2.2 e2e was fixed in stage 4 — after the fix, run the 2.1 VAE chunk e2e for real)
  **Adjacent small items of stage 3 (filed 2026-10-04, proposals)**:
  (b) add a paragraph "how to swap it into Wan" to the model card of the experimental mirror (the fine-tuned umT5)
  (c) add a check that compares the model card of the local distribution mirror with the card the recipe draws today (to find missed rebuilds)
  (d) make Wan's text stage check the `session` declared by the umT5 preset instead of silently ignoring it (same root as the component replacement item in known-issues)
  (e) make `e2e_wan_dit_test.ts` emit the VRAM record even for a failed step (to split the in-lane OOM of known-issues into ① leftovers / ② another process)
  **To add later (filed 2026-10-04, maintainer)**: (a) an f16 preset for TI2V-5B — we want the weights ready even if it does not run on the development machine (material: the f16 export is at the RAM boundary or
  needs a change in the exporter core; the development machine can hold neither checks nor sha rows — ADR 0121 decision 7 / ruling 2).
  **Related**: the 2 items of ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) "Open points" (the ruling on publishing third-party compatible components,
  agreement of graph descriptors across versions).
  **umT5 / Wan2.2 small items (filed 2026-10-04)**: (a) the original's index (`pytorch_model.bin.index.json`) and `config.json` are not pinned by sha256 (if the index disagrees with the actual files,
  it fails with a bare `KeyError` — there is no path by which values get swapped; ADR 0122 addendum "stage a results") (b) umT5's `write --check` opens a temporary directory inside the series
  (`.check-*`) — it remains if the process is killed (c) loading `ContainerDitWeights` (the layer-sequential entry point that reads the DiT weights from the container) takes 25 s per forward.
- **Quality improvement of anima base i4 (filed 2026-08-24 — the revival lever of the ruling to skip distribution)**: what remains is only
  **the list of possibilities left unverified** for the turbo-side i4 preset (dedicated geometry, g16, calibration amount, one more
  degradation mechanism — all of them "not tried", not "tried and failed"), and the source of truth is
  [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md) §8.
  g16 remains as one revival lever in parked "anima g16 evaluation". Making adaLN i8 was rejected by a ruling after a visual
  sweep (perf-ledger Q-9); for locating the places with high quantization sensitivity, the source of truth is item ④ of
  "3 known issues + anima base i4 sensitivity" in the [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md),
  to which the same ruling attaches "the anima DiT i4 line is
  on hold for a while". The calibrated series are kept as `*-i4-dyn` in `outputs/series/`
  (source of truth for the measurements =
  [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5).
- **irodori adaLN i8 A/B on the shipping rig (filed 2026-08-24)**: whether adaLN i8 (+13.1 MiB), which helped in the sim,
  also improves the reading on the shipping rig is unverified (the transfer limit sim → shipping — same research §2).
  Revival = dissatisfaction with the quality of the `i8+dit4` preset (formerly `w4`), or real demand for size optimization.
- **Bring measure_quant's distribution estimate up to J-5b** (found during J-3, 2026-08-22): sbv2's
  `project_distribution` still rests on the pre-J-5b premise "only linear weight slots; conv / embedding i4 has neither a storage form
  nor an execution path" (in fact shipped — ADR 0069 addenda 6/7). Harmless for relative
  comparison, but the estimate is too low and the docstring is stale. Update the target set and the description in one item.
- **Continuing model expansion**: Kokoro-82M (waiting for the exporter's multi-output `getitem` wiring + the ruling to add `lstm_scan`
  (ADR 0056 decision 8) — node-level multiple outputs themselves were allowed by ADR 0068), MobileSAM / SAM 2
  (conv_transpose2d), DA-V2 variable resolution (upsample_bicubic2d). The latter two need ops outside the vocabulary.
  The point-in-time record of the candidate survey is [recon-2](research/2026-08-14-model-expansion-recon-2.md).
- **Performance candidates**: the 🚧 rows of [perf-ledger](perf-ledger.md) are the source of truth for filing, adoption and order.
- Completing EmbeddingGemma (models pipeline / distribution, batch>1 export, runtime attention_mask wiring).
  **The implementation of the tokenizer (Gemma SPM BPE + byte_fallback) and the EG asset compile rode along with stage 1a
  of the generation API wave** (ruling 9 of 2026-08-31 — ADR [0084](decisions/0084-gemma-tokenizer-chat.md) decision 6. The implementation is
  shared; the assets are compiled separately). The wall in the conversion stage of the batch>1 export is in [known-issues](known-issues.md).
- **Practice of benchmarking against ORT Web** (maintainer ruling 2026-08-16): measure models supported by both regularly, and
  investigate whether we are too slow and where the bottleneck is. **The practice is already running for gemma4**
  (the harness was `tools/llm-speed/browser` = `deno task bench:llm-browser`, removed in 0.15; the record is
  [research 2026-09-12](research/2026-09-12-webml-browser-speed.md)). Remaining targets =
  EmbeddingGemma (requires completion on the models
  side) and **KokoroTTS** (corrected 2026-08-16 — the original Irodori was a typo. Kokoro
  runs on the Transformers stack, so it is easy to compare; requires Kokoro-82M support on the karume side = the model expansion
  candidate above, waiting for LSTM multi-output). In future, widen it into a comparison
  matrix that includes browser ONNX + native PyTorch (ruled unnecessary for now). The norms for measurement conditions (graph capture ON /
  freeDimensionOverrides / IO binding / checking EP partitioning, recording native EP or JSEP) have their source of truth in
  [runtime-landscape §4](research/2026-08-16-runtime-landscape.md).
- **Rolling generation events out to other families (waiting for demand)**: stage
  events for sbv2 / birefnet / depth / siglip2 / vowel (no step loop, so only stage transitions can be offered), and an AbortSignal cancellation slot in the **generation loop**
  of anima / irodori (today a throw from onEvent is the step-granular means of cancellation — the slot is kept in reserve).
  **AbortSignal on the construction path is implemented for anima / irodori** (`anima.PipelineOptions.signal` —
  checked at stage boundaries, passed through to the fetch layer, `signal.reason` propagated as is), so that is the precedent for the style
  (the `signal` of the other 6 families is only passed through to the fetch layer, with no checks at stage boundaries). **The generation-loop slot is
  implemented on the LLM surface** (ADR [0083](decisions/0083-generation-api-surface.md) decision 5 —
  rolling it out to other families still waits for demand).
- **RAM-saving sequential tile assembly for large anima resolutions (filed by the ruling of 2026-08-24)**: if VAE decode tiles
  can be composited sequentially without accumulating them, the peak RAM drops (`decodeTiled` still accumulates all tiles in an array before
  compositing). The original motivation, restoring acceptance of the 8 resolutions, was achieved separately by the evenly spaced snap placement of the ADR
  [0033](decisions/0033-vae-fixed-tile-decode.md) addendum.
- **RAM-saving sequential tile assembly for Wan's VAE (filed 2026-10-11, Codex deep review R6A-016)**: the Wan version of the anima
  candidate above. `decodeWanVaeTiles` (`packages/models/src/wan/vae-tiles.ts`) accumulates all tiles in an array, then allocates the final video and
  composites, so for Wan2.1's 832×480, 81 frames the 12 tiles' 764,411,904 B and the output's 388,177,920 B are alive at the same time
  (computed from the shapes; RSS not measured). Candidate: decode row by row, paste into the output, and keep only the row above and the tile to the left. Conditions: keep the blend
  order (vertical → horizontal), and decide adoption after confirming bit identity with today's output by a Uint32 comparison. To measure: RSS for 33 / 81
  frames, readback wait, GPU time (for Wan2.2, measure the same quantities at 1280×704 and at the exit of patchify space).
- **Avoiding duplicate downloads of sha256-identical files across repositories in hub (external feedback proposal ⑥, low priority)**:
  the fetch layer's cache key is already a content key (`['hf', kind, repo, path, sha256]` — ADR
  [0080](decisions/0080-hub-fetch-cache-050.md)), which removed duplication across revisions, but `repo`
  remains a key element, so across repositories the file is fetched again. Proposal: drop `repo` (or add an alias keyed by sha256 alone).
- **cache-less streaming mode** (a prerequisite for the 26B A4B class — the CacheStorage quota is the first wall): phase 1
  (streaming prefetch) cannot fall back to plain fetch without CacheStorage and stays fail-loud
  ([limitations](limitations.md)). Revival = real demand at the 26B A4B class. The first unit is one design memo on "whether sequential reads of phase 2 alone
  work with a source that has no CacheStorage".
- Pinning down the cause of the Metal numeric difference (known-issues); lifting the diagnostics / measurement constraints of the resident path.
- **MoE page-fault**: PoC done in a spike outside the repository (2026-09-01) — the mechanism works (the readback of a miss
  can ride on the round trip that decode already pays, with zero extra synchronization; output bit-identical to direct binding), but **practical
  viability is uncertain** (the miss cost is dominated not by transfer but by one re-execution fence, and making experts smaller does not
  make it cheaper. The fence floor in browsers (Dawn) and the real hit rate are unmeasured; one layer only). The start condition is
  subordinate to parked "value-dependent execution selection in the IR". The first unit stops at transcribing the spike's measurements into
  [research](research/) as a point-in-time snapshot.
- The MoE seam (fixed-k routing can be expressed with static shapes — do not bake the absence of experts into the dense API).

- **Remaining items of GEMV row blocks (filed 2026-09-07, consequence of K-21)**: ① the parallelism target 16384 is the saturation point of the RTX 3080 Ti (limitations) —
  the M2 re-sweep is done (the default 16384 is best); what remains is a sweep on integrated GPUs. The override is the static Session option
  `linearGemvRowsThreadTarget` ② the shader analysis cost in browsers (Chrome / Tint) is unmeasured (on Deno / naga the first turn costs
  +85 ms) ③ the Metal u32 check is also unmeasured for the 13 row-block shapes (known-issues).

- **Anima: iteration residency inside the DiT stage only** (filed 2026-09-07 — Codex performance investigation 04 §Anima): keeping the current design that creates and returns a Session per stage
  (a VRAM invariant), patchify the initial latent once inside the DiT stage, reuse the RoPE / cond and uncond embeddings /
  timestep material across iterations, run DiT output → CFG → Euler / DPM++2M update in the same token layout, and unpatchify only at the end.
  The responsiveness of `copyLatents` / onEvent / abort is a contract of the public surface, so it is not cut (also measure a boundary that finishes every few steps).
  **Making only the fixed inputs resident was measured — no gain** (2026-09-10, 1024px / CFG1 / Euler8, all steps / PNG identical).
  Iteration residency including latent / scheduler, and CFG>1, are unverified ([measurements](research/2026-09-10-codex-mtp-optimization.md#ホスト待ちと既存融合の追加測定)).

- **Whether to retire the slot backing before the bucket run** (filed 2026-09-07, limitations "prefill buckets"): the bucket run of the last chunk
  (the miss run) allocates temporaries in the arena while the chunkLength-shaped backing is still loaded, and the uncounted window carries
  "the miss run's arena temporaries (at most one bucket shape)" (with the default ladder, 768 shape + 256 shape ≈ 1.33 times).
  Retiring the activation backing first in generation's miss run removes the window, but the next 768 chunk then needs one more
  rebuild. Rule after seeing whether it helps on 4 GB class devices.

- **A summarizing overflow policy for ChatSession**: `onOverflow` is replaceable, so it goes in as one policy
  implementation. The reconsideration condition "after widening the window" was met by ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md) (capacity is a runtime knob)
  — this is where the deferral statement in the ADR [0083](decisions/0083-generation-api-surface.md) addendum points.

- **Measuring HF CDN concurrency**: once the per-connection limit is measured, re-file raising the download parallelism constant or splitting the parts
  toward the end more finely (the download slot improvement itself is killed —
  [research 2026-09-02](research/2026-09-02-cold-load-dl-timeline.md)).

- **Re-evaluating the GPTQ sweep**: the default stays as is (settled); the opt-in implementation is kept (source of truth =
  [research 2026-08-31](research/2026-08-31-gptq-axes-sweep.md)). Revival condition = re-evaluation with **multiple models ×
  16× calibration data** (a new gemma4 calibration rig is also on hold until then).

- **Moving norm's 1/dim to the host is on hold** (a real-GPU probe comes first; low cost-effectiveness. Making the reduce identity's
  params −inf stays as is = ruled as a set in the same frame as W-2/W-3) / **tolerance option B** (a public API change that adds a reduction-scale term to `allclose`
  — option A's per-op table is implemented.
  [research 2026-08-31](research/2026-08-31-op-tolerance-measurement.md) §8.2).

- **minicpm5's `export_decode` still bakes the RoPE table** (an asymmetry: only gemma4 moved to host supply, with ADR
  [0091](decisions/0091-gemma4-host-rope-variable-capacity.md)).
  Whether to align it with host supply is a separate ruling.

- **Adjacent filings of the memory management wave** (source of truth = ADR [0089](decisions/0089-memory-limits-preflight.md)
  Consequences / ADR [0090](decisions/0090-shard-spec-v3-tensor-pieces.md)):
  allocation in `GpuContext.createResident` still relies on errorScope (run-time transients fail before allocation in the planning-time
  preflight) / the positioning of `fromAssets` / the reference-first general rule for
  large assets.

- **Handle the N % 4 remainder of fused attention's a8 / s16 on the declared path (filed 2026-10-10)**: today, for shapes where the key length N
  is not a multiple of 4, ③ PV and the S storage fall back to f32 (docs/limitations.md; decided by `attentionRoute`). Key points:
  ③ PV's i8a8 needs the remainder in the A tile fill (n ≥ N as 0) and a `quantize_rows` variant that zero-pads the row width to a multiple of 4.
  The f16 storage of S needs the write race on u32 words that span rows removed structurally, passing S's row width as a uniform separate from N
  to ①②③ (both dense / i8a8). Both touch kernels shared with linear (`quantize_rows`, the GEMM skeleton), so
  keep the output for existing keys with N % 4 == 0 byte-unchanged (the snapshots are the detector). Start condition: when there is a request to use Anima resolutions that are not
  multiples of 32 px in practice, or when a family that accepts N % 4 != 0 shapes has an a8 / s16 preset.
  MOSS-Image (branch `wip/moss-image`) accepts such shapes with an a8 preset, but its preset runs them on the online
  kernel, which pads the key axis (ADR 0132 addendum of 2026-10-27); it meets the condition only on a device with
  neither online form, where the practical preset runs the three dispatches with S in f32 and, where S % 4 != 0,
  ③PV in f32 too (ADR [NNNN](decisions/NNNN-moss-image-dit-vae.md) decision 10).
  Kill criterion: measured VRAM and speed on those shapes show no difference against today's path that falls back to f32.


## release — the release preparation wave (some way off)

- Real-asset CI gate (the problem that GitHub CI does not exercise local assets). **The all-skip guard is done**
  (`packages/e2e/tests/e2e_assets_gate_test.ts` (moved from `packages/runtime/tests/` in 0.15) + CI env `KARUME_ALLOW_NO_ASSETS=1` —
  2026-09-05). What remains is the decision on promoting goldens to fixtures / fetching assets in the release gate
- Writing up the repository root README; verifying sideEffects of the JSR npm compatibility layer. **Out of scope for 0.8.0
  (ruling 2026-09-03 — replaced with a form whose Status line alone says it is a stub). Revival condition = 1.0 or
  a public announcement**, and starting requires re-measuring the bundle size (the gzip measurement of 2026-08-16
  is outdated by the additions of the gemma4 generation API, the GEMV family and the tokenizer)
- Licence interview (human confirmation of export-recipes' per-family provenance per upstream revision
  — the release gate of the reorganization. **Done ahead for the 4 published repositories at the human gate of wave K-4**).
  **Carried out on 2026-09-25** (the maintainer checked the upstream model pages + transcribed Revision used from `refs/main`
  of the HF cache). Counted from the distribution mirrors in `models/`, **7 families, 11 repositories** are published (anima 2,
  birefnet 2 (birefnet-hr / lucida), depth_anything, gemma4, irodori 2, sbv2-jvnv, siglip2).
  The ledgers filled in are anima / gemma4 (a new block for the drafter `google/gemma-4-E2B-it-assistant`) /
  irodori / sbv2 / siglip2, 5 in all. Rows left as Unrecorded with a reason = the revision of modernbert-ja-310m
  (no local record), the revision of Semantic-DACVAE-Japanese (parent) (not fetched).
  Rows left as Unverified = whether the Civitai permission field of anima-wai-v1.0 can widen the upstream licence, the Weights license of depth_anything (before revision matching).
  **FN (`rufflet17/voice_models`) has neither a licence statement nor a README on HF, so it cannot be published** (until this is clarified).
  **Ruled (2026-10-29, ruling 37 a)**: anima's licence is v1.2, the only text it bundles — Base's weights are byte-identical
  to files `circlestone-labs/Anima` publishes under v1.2, and the v1.0 `LICENSE.md` of `Anima-Base-v1.0-Diffusers` is stale
  (evidence and the v1.0 → v1.2 differences: [anima's THIRD_PARTY_NOTICES](../tools/export-recipes/anima/THIRD_PARTY_NOTICES.md)).
  anima-copycat-20260610's revision is recorded there too (Civitai version 3020450, with its sha256).
  The Code license / Attribution of the code-dependency blocks (transformers / PyTorch and so on) were filled in on 2026-09-05 from the actual
  upstream LICENSE files. The machine-readable place for the upstream revision is the container's
  `provenance.upstreamRevision` ([container-v1](container-v1.md) §2.3). Today the recipes of sbv2, minicpm5, siglip2, depth_anything, birefnet, embeddinggemma, deberta and wan / umT5 fill it
  (through `tools/export-recipes/_shared/upstream.py`), so irodori and the others **fill it at their next re-export** (on the same occasion as the `sym_max` field of W-G4-4)
- Re-ruling on the policy of "separating the semantic surface from the implemented subset" (attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — a unified rule that does not turn the observed subset into op semantics)

## parked (with revival conditions)

- **Gemma 4 MTP (speculative decoding with the official drafter)**: the source of truth for the current state and the remaining items is the K-20 row of [perf-ledger](perf-ledger.md)
  (remaining = M2 warm re-measurement of exploreMax 512, ⑥ k=7 + 3-value gate (awaiting a ruling), attribution of ⑩, 4-C docs). The history of stages 1 to 4-B and K-21 / H-15 is in
  the item "performance wave K-21 → H-15 done" of the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).

- **Value-dependent execution selection in the IR (prerequisite for dynamic MoE expert residency)** (ruling 2026-08-31 — not adopted).
  Per-expert load / eviction is a triple conflict among ① the container's merge layer (`bindDeclarations` — the krm entry `bindGraphs` also delegates to it; the binding table's
  full comparison is `validateAgainstGraph`), a check over all entries that matches the bindings of every initializer before the weights are fetched, ② the immutable Map of resident weights, and ③ the absence of value-dependent execution selection in the IR, so it is not a feature addition but a redesign across 3 modules (measurement record =
  [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md)). The official
  stance for now = **MoE keeps all experts resident in VRAM and budgets by total parameters** ([limitations](limitations.md)).
  Adding a "not-yet-arrived initializer" slot is also deferred as subordinate to this item (same ruling). Revival = real shipping demand for an MoE
  that does not fit in VRAM. The first homework then = a close first-hand reading of predictive offloading (SiDA arXiv:2310.18859 / HOBBIT
  arXiv:2411.01433) (the current reading: the readback can be removed, but the wall of keeping the host-side pool resident remains).
  The page-fault mechanism is **PoC'd in a spike outside the repository; practical viability uncertain** (on revival, transcribe the spike's measurements into
  research before entering a design spike — later section "MoE page-fault").

- **HF publication of karume-sbv2-fn** (ruling to hold, 2026-08-20 — in wave K it briefly swung to "publish with source attribution",
  but that was withdrawn). The upstream's written terms = only "commercial use OK, no credit needed, merging
  free" on the Booth distribution page; **redistribution is not mentioned**, and the distributor's identity is unconfirmed. Revival = confirming with the distributor whether redistribution is allowed,
  or a new maintainer ruling. The build method (separate pipeline `--pipeline sbv2-fn` — carrying the FN line's attribution) is kept.
  The attribution is minimal (quoting the terms; no details of the distributor — ruling 2026-08-20, `a7ed68e0`). No permanent local mirror
  (ruling 2026-08-30 — the e2e check was moved to jvnv, whose licence statement is positive; the fn mirror was deleted.
  Regeneration = the dist command in assets-layout, from `inputs/sbv2/FN*`)
- **Port of SBV2 `adjust_word2ph`** (ruling not to adopt, 2026-08-21 — ADR 0072 decision 8). The word2ph redistribution for accepting edits that change
  the phoneme count (replacing readings that do not align with word boundaries). The reference is the **upstream**
  Style-Bert-VITS2 semantics (LCS diff + 1..6 clamp, the residue raises), not the fork that AivisSpeech
  pins, which "forces consistency by even increments and decrements" (no silent approximation).
  Revival = real demand for reading edits that cannot be expressed with overlays.
- **anima g16 evaluation** (deferred 2026-08-23 — maintainer ruling to prioritize extending GPTQ. SBV2's
  g-axis ruling does not generalize across model lines (research 2026-08-22), so the evaluation itself keeps its value.
  Revival = quality dissatisfaction in the visual check of base i4. Bringing it forward as a background run that does not collide with the main work is allowed
  (maintainer ruling of the same day; safest to run after J-4 ②, which touches the anima calibration rig, has landed))
- hub Range parallel + prefetch — revival = fragmented repositories coming back (perf L-3)
- Making GPU timestamps an estimation source, full f16 (option γ), SBV2 NFC chip, regenerating the f32 anima series
- The by-design constraints (rank≤4 / OOB NaN / non-finite input / zero-element dimensions / cancellation granularity and others) have their source of truth in
  limitations.md — only items with real demand are promoted here
- **Prerequisite for the first publication of vowel-detector (filed 2026-09-24)** (first publication on hold — maintainer's ruling 2026-10-15): the recipe reads the upstream `feature_config.json`
  directly under `inputs/vowel-detector/`. This development machine holds the whole upstream repository, so before assembly
  run `cp inputs/vowel-detector/assets/feature_config.json inputs/vowel-detector/` once.

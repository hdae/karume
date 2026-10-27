# Backlog — the source of truth for wave order and work items

> This file alone is the source of truth for the project's **priorities, wave order and unfinished items**.
> Status vocabulary: `now` (current wave) / `next` (the next big wave) / `later` / `release` / `parked` (with a revival condition).
> Operating contract: ① completed items are **deleted** (git and the ADRs / research hold the history) ② measured values and design arguments are
> **not written** here — point to the source (ADR / research / ledger) ③ the source of truth for filing, adopting or rejecting performance candidates and for their kill criteria is
> [perf-ledger](perf-ledger.md); this file only refers to them as waves ④ the source of truth for by-design constraints is
> [limitations](limitations.md) — a constraint appears here only when a ruling turns it into work.

## now — after the 0.14.0 release (2026-10-16)

- **Migration to 0.15 (filed 2026-10-15, maintainer's ruling — ADR [0126](decisions/0126-numerics-profiles.md) / [0127](decisions/0127-node-dawn-toolchain.md) / [0128](decisions/0128-default-preset-original-weights.md) / [0129](decisions/0129-test-layout-and-lanes.md) / [0130](decisions/0130-project-language.md); 0126 deferred on 2026-10-18, 0130 accepted on 2026-10-18, the others proposed)**:
  branch `wip/0.15` (the worktree is outside the repository at `~/workspace/karume-wt-0.15`; main stays at 0.14.0).
  **Phase 1 (port) ✅ 2026-10-16**: ported to Node 24 + Dawn (npm `webgpu`) + Vite+ (Vitest projects = lanes, Oxlint, Oxfmt, TypeScript 7).
  The test suite was ported mechanically; the 61 real-weight e2e tests went to the private `packages/e2e`, and the shared groundwork (Dawn injection, fs and so on) to the private `packages/test-support`.
  The four checks that depended on Deno tooling (public surface, dependency graph, barrel, local-dist-server) were replaced, and the measurement tools / examples were deleted (to be rebuilt as browser applications — ADR 0127).
  Result: all lanes green (core 2,929 + 16 family lanes 2,119 passed, 17 skipped (opt-in cases), 0 failed — 2026-10-16 to 17, `26053ef2` (core re-run at `1c8f6f8b`, which fixed one spurious timer failure)); the artifact comparison against the Deno baseline shows zero numeric difference ([research](research/2026-10-16-node-port-results.md)).
  **Phase 2 (cleanup — the content of 0.15.0)**: ① test triage ([list](research/2026-10-16-test-triage.md): keep 306 / move 16 / drop 4 / merge 42 / review 24 — carried out on the maintainer's ruling)
  ② ADR cleanup and translation, English code comments (ADR 0130 — steps 2-9a / 2-9b ADRs / 2-9c docs and agent files / 2-9d comments, JSDoc and WGSL of `packages/*/src` ✅ 2026-10-18 / 2-9e error messages and their assertions ✅ 2026-10-19 / 2-9f the Python exporter ✅ 2026-10-19 / 2-9g the remaining tools and the hand-written fixture notes ✅ 2026-10-19 / 2-9h the language gate with its allow-list ✅ 2026-10-19 — step 2-9 closed; `packages/*/tests` and `tools/export-recipes` follow in 0.15.x) ③ ✅ the decision on taking in numerics profiles (2026-10-18, step 2-8 — ADR 0126 deferred: the default preset carries the practical bundle (ADR 0128), upstream parity became opt-in in 2-1, and auto-selection has no variant it could choose today (subgroup32 cannot be expressed in the manifest and lacks ② and ④ of the four-point check) → the runtime default stays the reference path; the re-entry conditions are in the ADR 0126 addendum) ④ ✅ the default preset policy (ADR 0128 rewritten — the default stays a practical preset chosen and recorded per model by a procedure; the manifest constraints C1 to C3 are checked by the distribution gate; transparency through the README and the card; the runtime warns when a8 is requested on a device without dp4a; the original-weights preset is a best-effort goal; not breaking; 2026-10-17, step 2-7)
  ②′ **K-82 online int8 attention (ADR 0132) ✅ 2026-10-23** — done before 2-10 at the maintainer's request (the Wan2.2 default preset is `f16+dit8-a8-attn8-online`; the 121-frame A/B passes under Node since the backing-build release wait of 2026-10-24 (relRMS 4.675e-2); **K-84 ✅ 2026-10-25** — the kernel's vec4 tiles (ADR 0132 addendum): self ×1.44–1.46 and cross ×2.0 on the RTX 3080 Ti, the 33-frame DiT forward −9.7%, bit-identical; **wave a ✅ 2026-10-25** — F-4, F-8 and F-3's input side ([research](research/2026-10-25-wave-a-dit-fusions.md)): the 33-frame DiT GPU −3.5%, bit-identical; follow-ups **H-42 ✅ 2026-10-26** (transient placement by least region growth, best of two orders — ADR 0093 addendum: summed region totals 30.26 → 24.18 GiB over 49 scenarios, Wan2.2 DiT 33f 880.3 → 717.8 MiB) → **F-3's output permute ✅ 2026-10-26** (A4 — the online attention stores O token-major: −14.9 ms per 33f DiT forward, −49 ms per 121f, bit-identical), H-43; follow-ups: the host-side Vᵀ key order (+10% in the spike), the subgroup-matrix opt-in for Node; ✅ 2026-10-23 the ampere geometry profile (`nvidia-ampere` registered, K-83 — ×1.03 on the DiT's linear) and the VAE recon (V-1 … V-4 filed; **V-1 conv3d w8a8 ✅ 2026-10-23** — ADR 0133, the Wan2.2 presets `i8` / `i8-a8-attn8-online-conv8`, VAE stage −46% at 1280×704×17; the maintainer's visual check passed and **`i8-a8-attn8-online-conv8` is the Wan2.2 default since 2026-10-23**; **V-2 ✅ 2026-10-23** — ADR 0134, the 32-latent pair that borrows the 16-tile weights, `vaeTile: 32` opt-in: VAE stage 0.70–0.83× of tile 16 (the 17-frame kill line missed by 1–3%, the 33-frame clip meets it — confirmed by the maintainer 2026-10-24); **the VAE wave b ✅ 2026-10-26 to 27** (bit-identical; the RTX 3080 Ti measurements are in the perf-ledger rows) — K-72 revived (`ResidentTensor.clear()` zeroes the tile caches, `8f6b0afb`), V-7 (split-tree axis reduce, `f5c18bad`), V-6 (channel-major quantiser of conv3d w8a8, `7f540580` / `21ec9fa8`), V-5 (the profile field `i8a8.conv3d` and the re-derived `nvidia-ampere`, `fde7f6e2`, [research](research/2026-10-26-conv3d-geometry-sweep.md)); follow-ups filed in the perf-ledger: V-12 … V-14, and V-15 (the Blackwell conv3d sweep, at a later sitting on the RTX 5070 Ti); the sweep case set per op family is in later; next = step 2-10, starting with a browser check page for Wan2.2 on the RTX 5070 Ti)).
  ⑤ ✅ CacheStorage for Node (`nodeCacheStorage` of `@karume/hub/node`, 2026-10-17, step 2-4) ⑥ ✅ PNG sha rows switched to the pixel sha (ADR 0106 addendum 2026-10-17, step 2-5) ⑦ ✅ the lint rule set (331 warnings under the default rules — settled 2026-10-17, step 2-3, `7cb010cf`)
  ⑧ ✅ the `Deno.test`-shaped shim replaced by Vitest's own forms (2026-10-17, step 2-2). ⑨ ✅ bf16 storage (ADR 0131 — runtime / exporter / anima's faithful preset `bf16` (base, 4 roles), 2026-10-17, step 2-6. Follow-up ✅ the same day: the remaining 6 models (turbo / aesthetic × 2 generations, wai / copycat) also get `bf16`; the bf16 conditioner is each model's own except for base; the export matches the upstream checkpoint byte for byte for all 7 models). ⑩ a8 measurement ✅ measured (2026-10-18, [research](research/2026-10-18-a8-umt5-sbv2.md), perf-ledger K-80): umt5's a8 is 2.1 to 2.7 times faster at L ≥ 118 (L 28 to 50 fluctuates; L 8 is 0.4 to 0.5 times), but the added embedding error is 4 to 8 times that of i8 rounding (a few rows of real prompts collapse); on the Wan step 1 latent the effect is smaller than that of the DiT's own a8; sbv2's `i8-a8` re-confirmed to have no speed benefit. Visual check: the maintainer found it "mostly fine"; root-cause investigation ✅ (§6: per-token int8 crushes the rows of function words; the main cause is the input of `wo`; mixing a8 per linear or layer does not remove it). Ruling (2026-10-18, maintainer): **not added even as opt-in — the root fix comes first** (umt5's current speed is not a problem; the root fix is perf-ledger K-81 under later). By-products: a fix of the Qwen3 text encoder export (broken since `a5427e6b`); bf16 rounding drops the non-persistent buffers. **Phase 3**: the 0.15.0 release (JSR only, or npm as well — undecided in ADR 0127).
  Candidates to ride along (decided before the release): the cross-repository reference for the duplication between the two irodori repositories, a redesign of the sbv2 subpath surface, removal of the glue that reclaims the old namespace of hub ≤ 0.4, HTTP Range fetching.

- **Remainder of the speed / memory wave (filed 2026-09-25; maintainer's ruling = split into two tiers, reference tier / practical tier; within-device determinism MUST hold in the practical tier too)**:
  The finished records of iterations 1 to 3 (contract and groundwork, B3 anima DiT residency, B4 irodori context K/V projection, H-35, K-70, E1 stage 1) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Candidates for the next iteration = the rest of wave 2 (DiT speed: B1 → B2) / wave 3 (memory: rest of E1 → E2 → E5 → E4) / wave 4 (practical-tier kernels: C2 VAE conv i8a8 → C1 → D2 → D1 → D3).
  The candidates come from `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md` §9.3 (not tracked by git) and the perf-ledger section of 2026-09-26 (K-59 to K-69, L-20 to L-23).
  Rest of E1 (BiRefNet's f16 series — ADR [0113](decisions/0113-birefnet-weight-series.md)) = golden comparison on a real GPU, derivation of the Lucida f16 tolerance, creation of the sha reference rows for real images, and stage 2 (the i8 plan + i8 series).

- **✅ To consider: whether the default quant preset is quantized or opt-in (original weights) (filed 2026-09-27, maintainer — ruling 2026-10-17: the default stays a practical preset (per-model procedure + manifest constraints + transparency); a device without dp4a gets a warning; original weights are a best-effort goal — ADR [0128](decisions/0128-default-preset-original-weights.md), step 2-7)**: today the default selects a quantized preset such as w8a8.
  Review whether that is justified. The decision "drop the a8 of the default quant on Metal" (perf-ledger K-70, ADR 0115 Consequences) was put on hold and merged into this item.
  **Material gathering done (2026-09-29)**: `.claude/reviews/2026-09-29_quant-default-recon/SUMMARY.md` (not tracked by git) — how presets are decided (recipe constants, hub resolution, the user's entry point),
  a 33-row table of series × preset (storage dtype, session declaration, download size, contract class), per-preset measurements (B570 / M2 / RTX, with the unmeasured ones marked), quotes of the past grounds for a quantized default, and what each of the three options protects. Next = discuss with the maintainer.
  **Added material (2026-10-04, the maintainer's policy — in the ruling on Wan's default preset)**: the default quant preset is "the most practical preset today" (light, with small quality loss).
  Room is left to switch the default to "the preset closest to the original weights" in future (ADR 0120 ruling 2026-10-04, item 4). Wan2.1 switched its default to the practical preset under this policy.

- **Remainder of geometry profiles (K-71 — ADR [0115](decisions/0115-geometry-profiles.md) / [0116](decisions/0116-geometry-profile-row-buckets.md) / [0117](decisions/0117-app-geometry-tuning.md)) (filed 2026-09-27 to 10-03)**:
  The finished records (registration of `nvidia-blackwell`, the 7 row-count buckets, integration into applications (ADR 0117, all 7 stages ✅), the Codex review follow-up, 4 small items) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  ① The PNG sha match before and after registration on the RTX 5070 Ti (Chrome) is unconfirmed, because there is no RTX record from before registration (optional: once on the check page of `cd11cdc8`, by the maintainer)
  ② Regenerating `apple-metal-3` with Chrome 154 alone happens the next time a sweep is taken on M2 (ruling 2026-10-02)
  ③ The Pixel 10a (Mali-G715) table is not registered; the re-run on the new version has not been done (research K-70 §15)
  ④ Adjacent small item (filed 2026-10-03): the known problem of the OOM check in `gpu_generation_context_test.ts` going red on B570 can probably be fixed on the mechanism side only, by having the test wait for `onSubmittedWorkDone` after destroy() before allocating 1 GiB (the same settle as the e2e of stages 3 / 6) — the assert is not changed
  ⑤ Adjacent small item (filed 2026-10-02): the default id of gpu-lab's generated table can collide with the id of an embedded table (apple-metal-3) → a default such as `<vendor>-<arch>-generated` (the application flow then works in Chrome without flags). (gpu-lab was removed in 0.15.)

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

**Remaining items**:

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
- **The three medium-priority items of the diff review's deferred table** (source of truth = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`
  — not tracked by git): ① W-G5-7 merge the asset resolution of opbench / fusion-hints into `tools/_shared/assets.ts` (opbench, fusion-hints and `tools/_shared` were removed in 0.15)
  ② W-G4-4 record the origin of the chunk limit in the provenance field `sym_max` (**rides along with a re-export** — on the occasion of wave b or a series
  update) ③ the missing path for ADR [0033](decisions/0033-vae-fixed-tile-decode.md) decision 5 "the TS side compares the geometry
  itself" (a ruling on whether to emit one geometry JSON or to amend decision 5 to the actual state).
- **Peak VRAM column on the model card (filed 2026-09-04)**: `karume dist` calls the TS-side estimate
  (the `prepareContainer(...).estimate()` family) when generating the card and adds a peak VRAM column to the quant table. Today's
  card shows only the stored bytes, so readers cannot judge whether the model runs on their GPU.

**Remainder of decode speed** (H-27 stage ② / small items K-48, K-49, K-50 / K-46): the item of the same name under later is the source of truth.

**On the maintainer's hardware (Claude cannot run these)**:

- Follow-up test of the RAM peak of the HF path in Chrome (measured on the Deno side —
  [research 2026-09-24](research/2026-09-24-part-length-ram-peak.md)).
  **Browser performance measurement is dropped from this wave (a) (maintainer ruling 2026-09-04)** — what a browser can measure is
  only the implementation difference between Dawn / wgpu, which does not bear on adopting kernel candidates. Alternative = add a feature to the TS packages that collects
  performance information, and gather the results of users running a sample collection (name and location undecided) on several environments
  (**filed only** — the items collected, the location and the opt-in form are not designed).
- Re-assessing `err.cause` on Pixel (8 GB class Android Chrome) — [known-issues](known-issues.md).

## later

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
  - The I2V section of Wan's model card (review of ruling 40): `tools/export-recipes/wan/card.py` says it carries the same facts as the example / gpu-lab READMEs (both removed in 0.15), but it does not carry that the example and gpu-lab applied the EXIF orientation (diverging from upstream) (`test_distribution.py` pins only the old wording). Add one sentence and an assert the next time the distribution is rebuilt (the 0.14.0 preparation).
  - Backing difference between Deno and Chrome: the cause of 5.97 GiB vs 4.80 GiB for Wan2.2's 121 frames with the practical preset (ADR 0121) is not investigated. The two candidate explanations are the geometry profile (Deno uses `default`) and `maxBufferSize` (how regions are split) (ADR 0124's risks; material recorded in stage 4b).
  - Flat indices of the GEMM family (rest of R2B-009): `gemmParams` in `packages/runtime/src/kernels/gemm.ts` only checks m / n / k individually as u32 and does not reject shapes where matmul / bmm's `M·K` or `K·N`, bmm's flat indices multiplied by the batch (`batch·M·K` and so on) or linear's activation-side `m·k` exceed 2^32 (linear's weight side `n·k` is checked by `linearParams`).
  - Flat indices of deform_conv2d (rest of R2B-009): `deformConv2dParams` in `packages/runtime/src/kernels/deform-conv2d.ts` checks only the output element count as u32 and does not reject shapes where the flat indices of the input, weight, offset or modulator exceed 2^32 (there is no check corresponding to conv3d's `assertConvFlatExtents`).
  - In-row loops of row kernels (rest of R2B-007): the in-row loop `i = i + workgroup size` of reduce / RMS / softmax / layer_norm and others (for example `packages/runtime/src/kernels/rms-norm.ts`) wraps around and never ends on rows where `dim + workgroup size` exceeds 2^32, and since this does not go through the dispatch count, the `gridStrideWorkgroups` check cannot block it — each row kernel's params need a `dim + workgroup size ≤ 2^32` check.
  - When Wan copies `latents` (rest of R6A-005 / former P-3): `generate` in `packages/models/src/wan/pipeline.ts` and `ti2v-pipeline.ts` runs its checks and the copy after waiting its turn in the serialization chain, so if the caller rewrites `latents` during the wait, generation uses the rewritten values — to guarantee the values at call time, copy outside the chain (at the entry of `generate`).
  - Added vocabulary of the T5-family tokenizer: `packages/models/src/text/t5-tokenizer.ts` uses the shared `parseAddedTokens` (which checks only the i32 range) and does not check whether an added token's id is below the vocabulary row count or matches the vocabulary row of the same spelling (it lacks the check that irodori's `parseIrodoriAddedTokens` has; affects both anima's T5 and Wan's umT5; the distributed assets satisfy both).
  - f32 overflow of SBV2's `z_p`: even when `noiseScale` is finite in double precision, `buildZp` in `packages/models/src/sbv2/host/latent.ts` puts products above the f32 maximum (for example `noiseScale = 1e39`) into a `Float32Array`, turning them into ±Inf, and passes them to the voice graph as they are (the same hole as the `z_noise` check; the GPU outputs `m_p` / `logs_p` are mixed in, so a design for which error type is thrown is needed).
  - Output dtype of BiRefNet / Depth Anything (output side of R6B-013): `packages/models/src/birefnet/pipeline.ts` and `depth-anything/pipeline.ts` check only the output shape before construction, not the dtype, so a non-f32 graph fails only at `asF32` after one run.
  - A helper for the checks after GPU acquisition (root of R6A-006): 8 families — Wan (`wan/family.ts`) / Anima / SBV2 / Irodori / BiRefNet / Depth Anything / SigLIP2 / vowel-detector — each hold the same check of features and `requiredLimits` after GPU acquisition in their construction code (`pipeline.ts` / `family.ts` / `admission.ts`) — a candidate for one helper (a check missed in one family caused R6A-006).
  - The recipe pytest goes red without the wan dependency group: the 5 tests concerned in `test_dit_reference.py` (2), `test_i2v_preprocess_ref.py`, `test_sources.py` and `test_vae_tiling_ti2v.py` under `tools/export-recipes/wan/tests/` do not go through `pytest.importorskip` and hit `ModuleNotFoundError` from a lazy import inside the module under test — in a venv without the opt-in `wan` group (diffusers / huggingface_hub) they fail instead of skipping (a plain `uv run pytest` goes red). **→ ✅ fixed 2026-10-15 (`7f3955c5` moves `importorskip` before the optional imports; on 2026-10-18, with diffusers / huggingface_hub made unimportable, the four files give 72 passed / 68 skipped / 0 failed)**
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
  - Count mismatch in limitations: the introduction of the "generation sequence" section of `docs/limitations.md` says "there are two constraints", but there are three bullets.
  - ADR 0075's description of deduplication: "the fetch layer deduplicates by path before summing" in `docs/decisions/0075-quant-presentation.md` disagrees with the switch to `fileRefKey` (`repo@revision/path`) in the ADR 0038 addendum. **→ ✅ resolved by the English translation of ADR 0075 (`9f328336`, 2026-10-18), which says to deduplicate by file identity, not by path alone (`selectionRefs(resolveSelection(…))`)**
  - Type arguments of DataView / ArrayBufferView (outside the lint rule `karume/typed-array-buffer`): since TS 5.7, `DataView` and `ArrayBufferView` also take a buffer type argument like TypedArray, and omitting it gives the default `ArrayBufferLike`. src still has 4 bare `DataView` type annotations (`packages/runtime/src/gpu/attention-dp4a-canary.ts`, `packages/runtime/src/format/container/header.ts` and 2 in `packages/models/src/audio/wav.ts` — `ArrayBufferView` has 0) — whether to add it to the rule's set of names is a separate ruling.
  - hub: decide whether the form where one manifest puts the same path with the same sha256 at different revisions of the same cross-repository target (two references,
    one cache entity — the hub cache inventory section of [limitations](limitations.md)) is reworked to match by entity key
    (recommendation a of Codex deep review R4-001), or rejected at parse.
  - hub: rejecting a cross-repository reference that points at its own repository applies only to the parse in `loadManifest` from HF. The distribution gate
    (`distribution_gate_test`) and `local-dist-server` (now in `packages/test-support`; the examples' `local-assets` and gpu-lab were removed in 0.15) use the public
    `parseManifest(text)`, so a manifest that passed on the local mirror can first fail after it is uploaded to HF. Decide whether the gate
    checks with the destination repository name, or whether the exporter's `karume dist` rejects `--ref-repo` pointing at its own repository (the card's repository
    name).
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

- **Export of regular E4B (gemma4) (ruling 4 = b, 2026-09-26)**: export on a machine with 48 GB of RAM or more. The recipe (`--model e4b`,
  meta construction on load + assign) stays as is (it OOMs on a 31 GiB machine). The regular E4B presets provisionally declare the E2B
  bundle; default `i4`.

- **Five parts fragmented by the 0.13.0 re-upload (filed 2026-09-24; ruling 2026-09-24 = publish as is and act after measuring the real download speed)**: in the fragmentation table of `hf-upload.zsh`,
  among the 255 MiB-class parts, 2 of anima (`text_conditioner` part 3 of `anima-turbo-v1.1` / `anima-v1.0`, 3.6 MiB/term),
  1 of birefnet-hr (`2048/matte` part 3, 4.1) and 2 of lucida (`2048/matte` parts 3 / 6, 0.9 / 8.8) fell below 10 MiB/term
  (the other 620-plus are healthy; extra is 0). **Speculation**: chunks byte-identical to the old safetensors were deduplicated against existing xorbs in the repository
  (global dedup is already stopped in the script). Repositories with a published pin do not use delete → recreate (runbook §2 MUST NOT), so
  decide whether to accept it based on the measured download speed, or another remedy (whether it resolves itself the next time a re-upload changes the bytes).
  **The three BiRefNet parts: cause identified and fixed on the distribution side (2026-10-16)**: the speculation above was wrong; the cause was long runs of zeros from the
  window mask and the zero padding baked in by the recipe (part 3 of 2048² is 86% zeros). With the mask turned into a table of window kinds + indices and the zero padding into
  `expand`, the zero runs disappeared from all parts (GPU output unchanged; ADR [0113](decisions/0113-birefnet-weight-series.md)
  addendum 2026-10-16; the 2026-10-16 addendum of the [research](research/2026-08-09-xet-fragmentation.md)). After the 0.14.0 re-upload,
  check the term counts in the fragmentation table **→ ✅ 2026-10-16: large parts are 42 to 64 MiB/term; terms = ranges = xorbs (no duplicate references). The other 5 repositories uploaded in 0.14.0 also have 25 MiB/term or more for their large parts**. **The two anima parts are a separate case and unresolved**: in the local mirror (0.13.0),
  part 3 of `text_conditioner` has zero zero-pages in all 5 variants, so the zero-run mechanism does not apply. The two fragmented parts
  (`anima-turbo-v1.1` / `anima-v1.0`) are byte-identical to each other and differ in content from part 3 of the other 3 variants, which were healthy.

- **Make cross-repository references easier to handle in the tooling (filed by the maintainer 2026-09-24)**: a form in which a recipe declares cross-repository component references
  (today the five `--ref-*` options of `dist.py` — ADR [0038](decisions/0038-manifest-v1.md) §7 addendum), and a feature
  in the export tooling that resolves the dependencies (the commit SHA of the referenced repository's main) and uploads starting from the referenced repositories.
  Two motivations: removing the duplication between `irodori-v4-small` and `irodori-v4.1-small` (83 files with the same sha256, 5,605 MiB)
  through cross-repository references needs the references to be declared, and publishing `anima` → `anima-extra` is serial, waiting for the referenced SHA, and
  manual ([release-runbook](release-runbook.md) §0). Design not started.

- **Revisit the structure of the manifest (`karume.json`) — make optimization flags and presets easier to handle (filed by the maintainer 2026-10-06)**: today the bundle of execution knobs
  is held in the single `session` of a quant preset, and the preset name doubles as the bundle's name (ADR [0074](decisions/0074-quant-seat-naming.md) /
  [0110](decisions/0110-practical-tier-numerics-contract.md) / [0111](decisions/0111-session-options-composition.md)).
  We want a form in which optimization flags and presets (bundles) are easier to handle. What belongs where (storage dtype, execution knobs, bundles, per-component declarations and so on)
  is to be sorted out later. The material is the open points of the inventory of how fast paths are expressed (2026-10-06, `.claude/reviews/2026-10-06_fast-path-inventory/`, not tracked by git)
  — there is no per-component session (the fast paths of the VAE / text encoder cannot be declared), the keys of new kernel variants, and whether changing a bundle
  changes an existing preset or creates a new one. **Work starts later** (maintainer 2026-10-06 — Wan2.2's optimization proceeds within today's `session`).
- **Remove the 5,605 MiB duplication between irodori v4-small and v4.1-small with cross-repository references (filed 2026-09-24)**: 83 files of the two repositories
  have the same sha256. Skipped in the 0.13.0 re-upload; to be decided in the next breaking wave (maintainer ruling 2026-09-24). Not bundled in 0.14.0 either (maintainer's ruling 2026-10-15).
  Done on the same occasion as the re-upload of the two irodori repositories (release section — addition of `dit_context`).
  Requires "Make cross-repository references easier to handle in the tooling" above.
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
- **Removal conditions for the glue that reclaims the `karume/1` namespace of hub ≤ 0.4 (filed 2026-09-25; unchanged in 0.13.0)**:
  `purgeLegacyCaches` in `packages/hub/src/cache.ts` runs at every entry to `loadManifest` regardless of the source, and
  `clearHubCache` also reclaims the old namespace. Neither ADR 0080 nor the backlog has a removal condition. To decide: ① when to remove it
  (where to close the migration period from old versions) ② whether enumeration failures, currently reported with op `"delete"`, move to `"open"`.
- **The 25 `gemma4_qat` fixture tests are skipped wholesale in CI (filed 2026-09-25)**: `series_fixture.py` and `test_fixed.py` under `tools/export-recipes/gemma4_qat/tests/`
  have a module-level `pytest.importorskip("transformers")`, and `transformers` exists only in the
  `gemma4-qat` group, so in CI's recipes job, which runs with the default sync, the 25 tests (test_audit 1, test_fixed 12,
  test_plan 12) are skipped at collection. To decide: a) stub `transformers` and keep 1 to 2 upstream comparisons
  (mind `isinstance(QuantizedEmbedding)` in `checkpoint.py`), or b) add `--group gemma4-qat` to CI
  (a trade between CI time and the strength of the upstream comparison).
- **The depth / birefnet real-asset e2e tests call the result recording outside the wrapper (filed 2026-09-24)**:
  `packages/e2e/tests/e2e_depth_anything_real_test.ts` and `e2e_birefnet_real_test.ts` (moved from `packages/models/tests/` in 0.15) call `results.record` directly in
  catch, so if recording fails on I/O, the original verification exception is replaced (aligning them with the form of `runRecordedCase` /
  `recordFailure` prevents this).

- **Leftovers of the code-quality wave (filed 2026-09-22 — source: the same wave in the
  [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md) (after the 0.12.0 release))**:
  - Where to put the four functions that take `Gemma4PipelineOptions` (`assertSpeculative` / `resolveGemma4PleResidency` /
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
- **Foundation for video generation in the browser** (investigation 2026-09-10, **maintainer's request 2026-10-02: first get Wan2.1 T2V 1.3B running; the final goal is
  MiniMax H3**): a plan to accept Wan2.1-T2V-1.3B in the order small DiT alone → attention / FFN at real token length → causal Conv3d VAE → scheduler and
  stage lifetimes. Decisions on the runtime vocabulary and the numeric contracts come first.
  [Structure and capacity estimate](research/2026-09-10-codex-mtp-optimization.md#動画生成の事前調査-wan-と-minimax-h3).
  **Main investigation done (2026-10-02, [research](research/2026-10-02-video-gen-recon.md), **ruling 2026-10-02, all as recommended**: H3 stays at a structural study + per-component candidates / conv3d adds an op to the IR (implicit GEMM, ADR drafted) / the text encoder is staged (precomputed embeddings → i8 on the GPU, f32 activations) / the original 50 steps / first target = 832×480, 33 frames, bring-up on Deno (maintainer: the published 8.19 GB is the value with offload + t5_cpu; with precomputed embeddings it should fit even on the local B570) → 81 frames → Chrome. **ADR [0118](decisions/0118-wan21-video-generation.md) accepted (2026-10-02; the 4 ruling points also as recommended = VAE always tiled at 32 / default shift 3.0 / the fixed prompts are the official examples / repository name karume-wan2.1). Stage 0 ✅ (B570 ceiling 9,600 MiB, `327818f0`), stage 1 ✅ (conv3d op), stages 2 / 4 ✅ (`4af318e6`, DiT S-shaped export + ratio band 9.6e-5, 2 kinds of VAE chunk graphs + cache normalization bit-exact, lane test:models:wan). Stage 5 ✅ (tiled decode, 12 tiles, degenerate-case check Uint32, 33 frames in 128 s), stage 3 ✅ (S=14,040 completed, at most 270 ms per submit, VRAM 5.15 GiB, GPU 16.75 s, band 75 with the f64 reference + normalized ratio r, acceptance r 2.9 / 4.6, attribution = precision difference in GEMM's K reduction (ruling: K-75 kept as is)). Stage 6 ✅ (`3de7d277`, @karume/models/wan, 2-step e2e sha row, 50 steps completed in 1,811 s, 33 PNGs = outputs/verify/deno-intel-graphics-bmg-g21/2026-10-02_wan-pipeline-full/). Stage 7 ✅ (`67c8d1d2`, distribution models/karume-wan2.1, fromPretrained, distribution gate, model card, WAN_SOURCES at publication, full verify 3,547 passed (red: 1 known OOM check + 2 missing acceptance-table registrations → fixed), the maintainer's visual check of 12 videos OK (2026-10-03, as intended, no seams → K-75 confirmed as kept)). Stage 8 ✅ (2026-10-03, 81 frames = S 32,760, limit 81 frames (default 33; now `WAN21_GENERATION.maxFrames`), 8 full-size DiT cases (f64 reference, band 104 = worst r 20.8 × 5, acceptance 13.1 / 4.89), 24 row blocks, at most 459 ms per submit, VAE 21 chunks long ratio 2.98e-6, 81f 50 steps 7,173 s / VRAM 7.31 GiB, sha rows, an 81f table in the card's Resources) → stage 9 (Chrome). **Adjacent small item (filed 2026-10-03)**: in safetensors 0.8.0 the key order of `__metadata__` in the header changes per process, so regenerated fixtures are not byte-identical (seen in vae_chunks.band / accept; the tensors are identical) — check regeneration by the tensors' sha, or fold the metadata into one key. The design of a release wait before the VAE was closed as unnecessary (the DiT stage and the VAE stage do not overlap — measured in stage 6). **Adjacent small item (filed 2026-10-03)**: the reference check `referenceGatePasses` (runtime/tests/helpers/reference.ts) is green when any one of the caseIds has a row, so a forgotten row for a new case id passes as a SKIP with a warning — tighten to "every caseId has a row, or a missing id is explicitly held". **✅ 2026-10-03 `7dc17309` (ADR 0106 addendum)**. **Adjacent small item (filed 2026-10-03)**: the diffusers ports in packages/models (wan's UniPC / blend, anima's sampler and so on) have no third-party notice (pre-existing state; published on JSR) **→ ✅ 2026-10-15 (notices in the same form as the recipes, in 7 places across 6 files)**. **Wan small items (filed 2026-10-03, diff review of the ADR 0118 wave — the IDs in parentheses are items of `.claude/reviews/2026-10-03_diff-review-wan/` and `2026-10-03_codex-review-wan/` (not tracked by git; the facts needed for decisions are copied into each item))**: ① measuring the GPU time per submit with the submit layout of normal execution (M8) (the "1 submit ≤ 1 s" check and its recorded values come from timing mode (`gpuTiming`, 1 dispatch = 1 pass), and the chunk boundaries after backing can differ from normal execution — ADR 0118 stage 8 result "scope of the submit check"; needs a GPU) ② `signal` (AbortSignal) support for Wan (P-4) (the other families have it; anima's form of passing it to the fetch layer and to the stage boundaries is the precedent; the lack of support is recorded in the Wan section of limitations) ③ a fourth case in golden `conv3d_block` that exercises the v4 and m-tile 64 variants (F4) (the current 3 cases are Wout 5 / 5 / 6 and Cout 6 / 3 / 2 and hit only the scalar variant and m 32. v4 and m 64 are exercised only by the GPU parity tests; needs golden regeneration and confirmation on the GPU) ④ a boundary audit of the coordinate arithmetic in the conv1d / conv2d WGSL (the expressions that turn dilation and padding into i32 coordinates) (an out-of-scope observation from the Codex review; neither reproduced nor asserted) **✅ 2026-10-11 — reproduced the wraparound on the CPU and closed it by putting the same `input length + 2·padding ≤ 2^31 − 1` check as conv3d into the params (Codex deep review R2A-003 / R2B-004)** ⑤ the CPU-only DiT host test reads and copies the whole 2.6 GB reference (TG-11) (`reference.full-{band,accept}-s14040-t0999` are 2,598,638,524 B each; `ArrayBuffer.slice` in `readBuffer` temporarily doubles it; RSS not measured) ⑥ the note separator " / " in results.json collides with " / " inside labels (TG-18) (labels in general, such as `formatNormalized`; no consumer splits it today; a fix would go on the separator side). ⑦ narrow the entity-candidate check of `prompt_clean` into an exact check that "rejects only when the string actually changes", using the name table (Python's `html.entities.html5`, 2,231 names) (accepting sequences that do not form an entity, such as `R&D` — ADR 0119 ruling 2026-10-03, maintainer: "a workaround if it can be avoided"; once someone asks for it). Perf filings K-72 (clearBuffer entry point) / K-73 (the reshape condition of the adaLN fusion rule) / K-74 (the frequency table of timesteps_proj) (2026-10-03: K-72 / K-74 ❌ as their kill conditions were met; K-73 🚧 as the adaLN share needed for the judgment is not recorded)**)**: on the DiT side the vocabulary
  is nearly sufficient, and recipe patches (making the complex-form RoPE real, lowering unpatchify's rank) are expected to suffice. Gaps = the video VAE's conv3d
  (option A: add an op to the IR (extended composite layer + ADR, about 1,000 lines of src (estimate)) / option B: decompose into conv2d in the recipe) and carrying over
  feat_cache (an export plan of one-chunk graphs × 2 kinds), upsample3d's temporal interleave (rank 6), how to load umT5-XXL 5.68B
  (i8 / i4 / f16 / precomputed embeddings / CPU), full attention over 32,760 tokens (about 283 TFLOP per forward; 50 steps × CFG
  is about 28.3 PFLOP) and TDR. H3 exists and is open-weight (HF MiniMaxAI/MiniMax-H3; by the letter of the terms Japan is a licensed region (not legal advice)),
  but with 33B dense + Qwen3-VL-32B + VAE 2.6B it is about 144 GB per task and assumes offload even natively (a measured example of 112 s on an RTX 5090);
  there is no browser WebGPU case.
  **Review processed (2026-10-03)**: 75 diff-review items + 12 Codex items processed in 37 commits, one per fix (ADR 0118 addendum 2026-10-03).
  **Follow-ups of Wan2.1 (drafted 2026-10-03; 0119 proposed / 0120 accepted)**: [ADR 0119](decisions/0119-wan-umt5-gpu-text-encoder.md) (stage 10 = umT5 on the GPU, free
  prompts, AbortSignal, 10a to 10e) / [ADR 0120](decisions/0120-wan-dit-w8a8-seat.md) (w8a8 preset, stages 1 to 7) / stage 9 Chrome (build gpu-lab's
  Wan page; the maintainer checks on the RTX 5070 Ti machine). Order: stage 9 ∥ 10a → 10b (export host RAM; run alone) → 0120 stages 1 to 2
  (CPU) → GPU stages serialized by a lock (10c → 0120 stages 3 to 5 → 10d) → visual A/B (0120 stage 6, 10d) → H3. **Progress (2026-10-03 evening)**: stage 9 = page finished, waiting for the real run on the maintainer's RTX machine / 10a ✅ / 0120 stages 1 to 2 ✅ (i8 series, 3 presets) / 0120 stages 3 to 5 ✅ (bands settled by measurements on B570; the practical preset takes about half the time of f16; step 1 relRMS is 1.07e-1 at 33f / 2.10e-1 at 81f, mostly from attention a8 → ruled by the visual check of stage 6) / 10c preparation ✅ (parity of the bucket table, export coverage with a small model, gelu replaced by tanh) / collecting the i8 golden at S=14,040 → 10b ✅ (meta trace + i8 per row chunk + fixed_weights, no core change, peak 6.79 GiB, containers 5.69 GB) → the CPU side of 10c ✅ (layer-sequential f64 / f32 reference, 10 goldens, e2e band not derived) → 10d-1 ✅ (karume-umt5-xxl, Wan's cross-repository reference, the placeholder-SHA check, 2 mirrors), 10d-2 ✅ (textEncoder path selection, text stage, signal, examples / gpu-lab) → 10d-3 ✅ (tightening of the reference check `7dc17309`) → the check of 10c ✅ (band 79), 10d-4 ✅ (2 gpu-text sha rows, text stage 10.4 s, VRAM peak 6.30 GiB, residue +0.08 GiB) → next is 10e (optional) → the visual A/B of 0120 stage 6 (12 videos of material ✅ 2026-10-03, sent to the maintainer, awaiting a ruling).
  **Progress (2026-10-04)**: stage 9 ✅ (81 frames, 50 steps completed in Chrome on the RTX 5070 Ti, `f16` preset, precomputed path — ADR 0118 addendum 2026-10-04.
  Ruling: no sha rows for this environment key) / 0120 stage 6 ✅ → **ADR 0120 accepted** (the practical preset `f16+dit8-a8-attn8-s16` is settled with today's bundle, **default preset = practical preset**,
  `f16` is the reference-side preset selected explicitly, and the existing sha rows, bands and case ids are kept by naming `f16` explicitly — ADR 0120 ruling / addendum 2026-10-04) / gpu-lab's Wan tab gets
  quant selection, `demo:wan` gets `--quant`. The distribution mirror `models/karume-wan2.1` is updated (only `defaultQuant` in `karume.json` and the README; containers byte-identical; host checks green).
  The 2 sha rows of ADR 0120's 50-step opt-in for the practical preset ✅ (`d5b3daa4`, B570; the practical preset's 81 frames take wall 3,703 s = DiT stage 3,383 s + VAE stage 319 s,
  DiT stage VRAM peak 6.18 GiB — ADR 0120 addendum "two 50-step sha rows for the practical preset"). Remaining: `test:models:wan` on the GPU (ADR 0120 addendum 2026-10-04 "remaining work" —
  after the GPU serial queue) (→ all green in the full verify on the RTX 3080 Ti — ADR 0121 addendum "RTX 3080 Ti lanes and full verify", 0 failed at `7fefb2ad`).
  ADR 0119's visual material (umT5 GPU path + practical preset, 3 free prompts and 3 fixed ones × seed 42 to 45) is done (ADR 0119 addendum (2026-10-04) "rest of stage 10d").
  **Wan small items (filed 2026-10-04)**: (a) real runs in Chrome of the GPU path (umT5 on the GPU) and of the practical preset are not done (the default path of gpu-lab's Wan tab is still
  `precomputed` — moving the default to `gpu` also waits for the same check; gpu-lab was removed in 0.15) (b) ✅ the model card generation (`tools/export-recipes/wan/card.py`): "browser not confirmed" and
  "not run" for the practical preset's 81 frames corrected to measurements (2026-10-04) — reflected in the distribution mirror at the next rebuild (c) ✅ in `tools/gpu-lab/README.md`, added the RTX results to the
  "Checking Wan in Chrome" section (d) the output names of the anima / irodori / sbv2 examples are `default` when the preset is omitted, so they are overwritten under the same name when the default preset changes (the examples were removed in 0.15).
- **Acceptance of Wan2.2 TI2V-5B (in progress, filed 2026-10-04, ADR [0121](decisions/0121-wan22-ti2v-5b.md) accepted)**: reference preset `f16+dit8` /
  practical preset `f16+dit8-a8-attn8-s16` (2 presets), one DiT graph with I2V support, tiled decode of VAE 2.2, stages 0 to 10. MiniMax H3 comes after this.
  **Ruling (2026-10-04, maintainer)**: the accepted resolutions and frame counts are built and tried within what the development machine can run (the split of 1280×704 up to 33 frames / 832×480 up to 81 frames
  is being confirmed — stages 0 / 1 do not depend on the split); no f16 preset for now; the I2V aspect ratio first follows the official behaviour (resize to cover, then center crop).
  **Ruling 1 confirmed (2026-10-04)**: accept and try both splits (the 832×480 family up to 81 frames, the 1280×704 family up to 33 frames — ADR 0121 addendum).
  (→ revised on 2026-10-05: only 1280×704 / 704×1280 × 4n+1 with 5 to 49 frames are accepted — "Change of the accepted sizes" below)
  **Progress (2026-10-04)**: stage 0 ✅ (fetch entry point, RAM measurement of the 5B, layer-sequential f64 / f32 reference `59367d44`) / stage 1 ✅ (DiT patch with I2V support, the 5B i8 series
  `wan2.2-ti2v-5b-i8-dyn`, 10 goldens at S = 192, `export_dit --check` `25b6962a`. The golden context is a real prompt's embedding — synthetic random values amplify the difference between f32 and f64)
  / stage 2 ✅ (the r check of the reference preset is green; both presets are within the memory threshold; 121 frames also fit — ADR 0121 addendum "stage 2 results". Whether to widen acceptance to 121 frames
  was for the maintainer to rule before stage 6 — moot since the 832×480 family was dropped from acceptance) / stage 3 = code landed (extraction of shared modules and generalization of 2.1 — Python `1e4d7cbb`, TS `f25907f5` /
  `e54e6dc6` / `cf321598`, ADR 0121 addendum "stage 3 results"). **GPU pass judgment of stage 3**: the 2.1 lane gave 394 passed, 1 failed (sha rows, bands and case ids all match; the red is the known in-lane OOM),
  and the four 50-step runs of `KARUME_WAN_FULL_PIPELINE=1` match the existing sha rows. **Remaining = all green on a lane re-run** (in-lane OOM = Deno's `GPUDevice.destroy()` does not return VRAM —
  known-issues. A symptomatic mitigation that prompts GC at the tests' GPU acquisition point was added (maintainer's ruling 2026-10-05); the re-run with the mitigation gave 395 passed, 0 failed (margin about 0.15 GiB — known-issues)). **Stage 3 is complete** (ADR 0121 addendum "stage 3 closed"). The comparison of the 1.3B f16 series
  (`export_dit --check`) matches for 69 files. **Stage 4 ✅** (2 kinds of chunk graphs for the VAE 2.2 decoder, no new op, the GPU chunk sequence within band 4.8e-5 on B570,
  the 1.3B VAE outputs byte-unchanged — Python `0f1ff88a` / `734077ae` / `2763651b`, TS `3ab40a8d` / `08547483` / `b004af49` / `8166a18a`, ADR 0121 addendum
  "stage 4 results". `src` was not touched, so the 2.1 lane was not re-run; the helper move is bit-identical in a standalone run of the 2.1 VAE chunk e2e).
  **Stage 5 ✅** (tiled decode — blending in patchify space → stitching → host unpatchify; the 2.2 tile plan frozen; the element-count check turned into a shape check.
  The GPU tile e2e on B570 is within band 5.0e-5; host unpatchify matches upstream at Uint32; the 2.1 tile reference was regenerated and its tensors and metadata match —
  Python `4436457f`, TS `567fb9f0` / `0d5fc54d` / `f91b5b39`, ADR 0121 addendum "stage 5 results". The 2.1 GPU lane runs once together with the final state of stage 6 — not reached with 1 failed; what remains is the same 3080 Ti lane as stage 6 (→ 0 failed on the 3080 Ti lane — "3080 Ti lane ✅" below)).
  **Change of the accepted sizes (2026-10-05, maintainer's ruling)**: Wan2.2 accepts only 1280×704 / 704×1280 × 4n+1 with 5 to 49 frames (the 832×480 family, when generated on B570,
  collapsed whatever the preset and shift; it is outside the official supported sizes). The default shift is 5.0. The frame limit of 49 is the value at which both presets completed 50 steps on B570
  (57 also completed, but with the practical preset it is outside the 8.0 GiB guideline of decision 8; the default stays at 33 provisionally). The official default of 121 frames waits until it is confirmed with gpu-lab + RTX 5070 Ti in stage 8 —
  ADR 0121 addendum "accepted sizes moved to the two official sizes", `59557732`. (→ limit raised to 121 on 2026-10-07 — stage 10 below)
  **Stage 6 = code and GPU checks done** (T2V of `WanTi2vPipeline` — D1 `59557732`, `8a3a99db` (2 extra DiT inputs, `ditInputForm`), `0fb5c005` (`family.ts`), `e0c50cea` (class),
  `992ca738` (helper and e2e), `24ffd87b` (Python 2-step reference), `3df18612` (bands), `7173c91d` (limit 49), `128b511e` (6 sha rows of the reference preset), ADR 0121 addendum
  "stage 6 results"). The 2-step end-to-end run is within the band on B570 (acceptance too; the 3 fault injections fall outside the floor (twice the band)). 50 steps × 1280×704 × 33 frames completed in 43 min 3 s with the reference preset and
  23 min 19 s with the practical preset, 0 non-finite. The 33 PNGs of the practical preset are byte-identical to the generation script's output (script ≡ product). The lanes (`128b511e`):
  `test:models:wan-ti2v` 41 passed, 0 failed; `test:models:wan` 403 passed, 1 failed (the known in-lane OOM — a standalone re-run gave 2 passed; the numeric checks and
  sha rows all passed; known-issues). Remaining = the 2.1 / 2.2 lanes on the 3080 Ti (a run that writes rows on the frozen copy `128b511e` → a plain run at HEAD, both 0 failed). The 2.1 lane on B570
  OOMed at the same step both times (cause = the runner holding the step — known-issues). (→ both 0 failed on the 3080 Ti lanes — "3080 Ti lane ✅" below)
  Maintainer's ruling (2026-10-05): no remedy on B570; pass it on the RTX 3080 Ti after the hardware swap. After the swap, the new GPU's sha rows are created on the frozen copy `128b511e`
  with `KARUME_REFERENCE=write` (the B570 rows are kept; writing at HEAD would let regressions in the commits after `128b511e` escape the sha — ruling 2026-10-06), and from then on matching those rows is the ADR's pass condition (going back to B570 is not a condition). From stage 4 on, the
  condition "run the 2.1 lane on B570" is read as the RTX 3080 Ti rows (ADR 0121 addendum (2026-10-06) "the development machine changed"). The swap was completed on 2026-10-06
  (environment key `deno-nvidia-geforce-rtx-3080-ti`).
  The bands of stages 4 / 5 / 6 (measured on B570) are **run unchanged on the 3080 Ti too** (maintainer's ruling 2026-10-06 — if a result falls outside a band, consult with the measured value attached; no re-derivation in advance).
  **Stage 8a ✅ (2026-10-06)** = distribution `models/karume-wan2.2` (recipe parameterized by the generation table, `dist.py --pipeline wan-ti2v`, provenance check split, own repository 6.714 GiB + cross-repository umT5 5.296 GiB) and
  generation selection in gpu-lab's Wan tab (`/models/wan22/`; up to 121 frames it calls the internal API of `family.ts` with a widened descriptor; the product's acceptance of 49 is unchanged (→ on 2026-10-07 the product's acceptance became 121 and the widening override was removed — stage 10)), the 2.2 e2e goes through the distribution,
  `karume-wan2.2` added to the distribution gate — ADR 0121 addendum "stage 8a results". **Default preset = practical preset ✅ (2026-10-06)** — 12 videos of the practical preset (seed 42 to 45 × 3 prompts,
  1280×704×33, 50 steps, 3080 Ti, 753 to 792 s each, `outputs/misc/wan22-visual-2026-10-06/`) checked visually by the maintainer with no breakdown → `WAN22_DEFAULT_QUANT`, the card text and 3 places in the tests were changed and the distribution rebuilt.
  **8b** (the TI2V entry point of `examples/wan`, the docs that close T2V) comes after the 5070 Ti check. (→ ✅ 2026-10-09 — "8b ✅ → stage 8 ✅" below) **Chrome on the 5070 Ti ✅ (2026-10-06)** — 1280×704×121, 50 steps with the practical preset completed (40.3 min, DiT diagnostic value 9.47 GiB,
  about 10.5 GiB for the DiT stage by the maintainer's eye (including Windows) — ADR 0121 addendum "RTX 5070 Ti Chrome"). How to widen acceptance to 121 is decided after running one 121 on this machine (ruling 2026-10-06).
  **3080 Ti lane ✅, stages 5 / 6 ✅ (2026-10-06)** — rows written on the frozen copy `128b511e` (`03a5a7ba`), and in the full verify at HEAD `b744a9fd` both Wan lanes are all green
  (the 8 failed are all outside Wan — known-issues). **1280×704×121 also completes on this machine** (63.1 min including thermal throttling, DiT stage peak 11,361 / 12,288 MiB). The 8b example ✅ (`b744a9fd`)
  — ADR 0121 addendum "RTX 3080 Ti lanes and full verify".
  **Stage 10 = landed (2026-10-07, maintainer's ruling "both open points OK as recommended")** — Wan2.2's accepted frame limit 49 → 121 (descriptor, card and the accepted-set fixture,
  gpu-lab's override removed; limitations records that the 10 GB class (B570) was confirmed only up to 57 frames (longer lengths unsupported)). The development machine's reference value is the 2-step × 121-frame case
  `f16+dit8-2step-boxing-cats-seed42-1280x704-121f-shift5` (opt-in `KARUME_WAN_TI2V_121F=1` — about 12 minutes for the VAE stage alone), and the 3080 Ti row is filled in (the device lost of this case with the reference preset was resolved by ADR 0123 (in-flight submit bound) — completed all 5 times).
  Maintainer's visual check: the 3080 Ti's 121 frames look the same as the 5070 Ti's — ADR 0121 addendum "stage 10: 121 frames accepted".
  **8b ✅ → stage 8 ✅ (2026-10-09 — the conditions of stage 8 are met; closing T2V comes after stage 7)** — the "close T2V" docs (the document row of ADR 0121 "Affected files" — quantization / limitations / glossary /
  assets-layout / perf-ledger / backlog / ACTIVE_DESIGN / CHANGELOG, and the target of ADR 0118's "Open points") were brought to the current state. The full verify after stage 8 is
  `7fefb2ad` (4,003 passed, 0 failed — ADR 0123) — ADR 0121 addendum "stage 8b results".
  **Stage 7 checks and sha rows ✅ (2026-10-09)** — the self A/B check at 1280×704 with 33 frames (S = 7,920, relRMS 4.39e-2, band 8.8e-2) and 121 frames (S = 27,280,
  4.92e-2, band 9.9e-2, opt-in), census, determinism, 5 fault injections, 1 submit ≤ 1 s with the practical preset (max 413.2 ms), the 2-step sha row of the practical preset, 2 sha rows for 50 steps
  (RTX 3080 Ti); the default settled as 1280×704, 33 frames — ADR 0121 addendum "stage 7 results". **Visual check ✅ (2026-10-09)** — all 12 pairs of reference and practical preset OK;
  2.2 follows free prompts better (2.1's weakness is a property of the model's capacity — ADR 0119 addendum) → the card was rewritten to the measurements and the distribution rebuilt. **Stage 7 ✅ → T2V closed**.
  **Next** → stage 9 (I2V) → **a break** (the maintainer tries it to see how it is used).
  **Stage 9 plan ✅ (ruling 2026-10-09), 9a ✅ (2026-10-09)** — the encoder is 3 symbolic-length graphs split before and after the mid attention (assets independent of resolution, 108 MB),
  preprocessing (a verbatim port of Pillow's fixed-point LANCZOS, sizes, patchify, normalization), GPU check (band 7.8e-5, 3 sizes in the same Session) — `0b34c674` to `b20fdba7`,
  ADR 0121 addendum "stage 9 plan rulings and stage 9a results". **9b ✅ (2026-10-10)** — I2V conditioning, public API (`WanTi2vGenerateRequest`), encoder in the distribution, CPU reference, end-to-end bands and 5 sha rows, 2.1 / 2.2 lanes green —
  `c98d4747` to `063dea42`, ADR 0121 addendum "stage 9b results". **9c ✅ (2026-10-10)** — the example's `--image` / `--fit` (`48f204ed`), gpu-lab's condition image (`b630417b`),
  card and NOTICE (`7d729dcb`), host-side README (`dfe5ab3f`), docs `2ac1ac62` — ADR 0121 addendum "stage 9c results". **9d ✅ (2026-10-10)** — the I2V self A/B check
  (1280×704×33, relRMS 4.6146e-2, band 9.3e-2, opt-in `KARUME_WAN_TI2V_I2V_FULL=1`) and the 50-step sha row check (`136cee60`), 2 sha rows for I2V's 50
  steps (reference and practical preset, RTX 3080 Ti) (`28ab44fe`), 12 visual pairs (reference vs practical preset, 3 images × seed 42 to 45; the maintainer's judgment = all 12 pairs OK, quality no different from T2V),
  VRAM observation (for the same request the DiT stage peak is I2V 7,927 / T2V 7,931 MiB — the encoder's allocations do not remain), full verify (HEAD `6679d123`, 4,035 passed, 0 failed),
  Opus diff review (28 items, 0 bugs, the 2 medium items in `08f21894` / `6679d123`), card and docs (`620a0766` / `732f507b`), distribution rebuilt (only README.md changed)
  — ADR 0121 addendum "stage 9d results". One check remaining: the floor of `cba99c24` for T2V's 50 steps with the practical preset has not run on the GPU (the next time T2V's 50 steps are run) **→ ✅ 2026-10-15 (sha rows match for both the reference and the practical preset, floor green too, RTX 3080 Ti, 34 min 45 s)**.
  **Stage 9 ✅ → I2V closed. Next = a break** (the maintainer tries it to see how it is used) → next week's order (ruling 2026-10-10): ② archiving the now section (✅ 2026-10-10 — finished records moved to the research archive) → ③ the Codex deep-review fix wave (✅ 2026-10-11 — the findings fixable as recommended are taken in: 9 medium; of 183 low / info, 100 fixed and 52 docs fixes; making the fractional fallback of Anima's int8 attention visible; 7 missing tests; typesDependency in the dependency-graph check; withdrawal of ADR 0108 decision 12; decision material for the errors of unsupported ops. The items awaiting decisions were settled by rulings 20 to 27 and taken in. Features of rulings 4 / 6 / 8 / 18 ✅ 2026-10-11 — ruling 4 = BiRefNet's post-processing method declared per series in the distribution (HR uses the upstream method `u8-pil-bicubic`; the blind A/B could not tell any of the 8 pairs apart, so upstream per the rule; pipeline `birefnet/2`; ADR [0125](decisions/0125-birefnet-matte-postprocess.md)); ruling 6 = the attention row-block budget (runtime's `SessionOptions.attentionRowBlockMaxBytes` only; output bit-identical; with the practical preset at 121 frames, 512 MiB lowers the VRAM peak by 1.46 GiB; ADR [0124](decisions/0124-attention-row-block-budget.md)); ruling 8 = the conversation fork `fork` (addenda to ADR 0066 / 0083); ruling 18 → 40 = the image's EXIF orientation is applied in both the CLI and gpu-lab (M2's Chrome applied the orientation even with `imageOrientation: "none"`, so we aligned with the side that intentionally differs from upstream's `generate.py`; ADR 0121 addendum; manual check on M2's Chrome OK); gpu-lab saves Wan's results as a zip of numbered PNGs. The upstream parity policy was added to ADR 0005. No row-block budget in Wan's default (ruling 39 = a; no public API entry point either). Retirement of HR's f32 preset ✅ 2026-10-11 (HR has only the `f16` preset; presets are derived from the upstream storage type; the mix-up check uses a fixed upstream revision; the card's execution resources are measured per checkpoint (the conversion error in total allocation 1.7 / 4.1 → 1.6 / 4.0 GiB also corrected); Lucida's f32 series rebuilt with the upstream revision; the 2 local mirrors rebuilt with `birefnet/2`; ADR 0113 addendum). Next = ④ H-37 measurement)
  (+ the review's 7 missing tests, typesDependency in the dependency-graph check, gpu-lab's EXIF) → ④ H-37 measurement (✅ 2026-10-15 — the seams are estimated at 0.34 to 1.30% of one step (after correction within generation), below the 5% kill line → perf-ledger H-37 ❌, [research](research/2026-10-15-wan-dit-seams-h37.md)) → ⑤ cleanup of outputs/diag and tools/shared
  (✅ 2026-10-15 — tools/shared merged into tools/_shared; 3 regularly used scripts (2 GPU-busy checks, a probe of one timestamp tick) moved to tools/diag; the 4 stages of deriving DeBERTa's collapse ceiling brought under version control (`deberta/collapse_ceilings.py` and `tools/diag/deberta-collapse.ts`; the re-taken results are byte-identical to those of 2026-10-07); the nature of outputs/diag stated in [assets-layout](assets-layout.md); investigation scripts not moved, to keep the paths the records point to) → next = ⑥ preparing 0.14.0 (first publication of the Wan family; whether to upload Wan2.1 decided then) **✅ 2026-10-16 — 0.14.0 released (3 JSR packages, 7 HF repositories, connectivity from the published versions OK). Maintainer's rulings (2026-10-15 to 16): Wan2.1 is not published on HF (ADR 0118 addendum); vowel-detector on hold; gemma4-qat not published; the irodori de-duplication not included; the umT5 / Wan licences checked by the maintainer; third-party notices added to the TS ports from diffusers; umT5 included in a form usable on its own, independent of Wan (`@karume/models/umt5` — ADR 0122 addendum 2026-10-15); `THIRD_PARTY_NOTICES.md` (the list of diffusers ports + the full Apache-2.0 text, Pillow's copyright notice and permission text) bundled with `@karume/models` (maintainer's ruling 2026-10-16 = 6 a); the sbv2 AGPL matter stays as is in 0.14.0 (same ruling, 7 a — still "awaiting a ruling" in the release section). Scope: the release section below**.
  (What follows is the list from before the work started) **Next** → stage 7 (the self A/B check of the practical preset (the set of S undecided — the 832×480 values can no longer be used), census, determinism, the maintainer's visual check (doubling as the ruling on the reference preset's own quality, 4 or more seeds,
  1280×704, the free-prompt comparison deferred to it by the ADR 0119 ruling (2026-10-04)), the default size / frame count (provisionally 1280×704, 33 frames for now), the 2-step sha row of the practical preset, the 50-step sha rows (reference and practical preset — not written in stage 6)) → stage 9 (I2V) → **a break** (the maintainer tries it to see how it is used — 2026-10-06).
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
  (a) ✅ gpu-lab's Wan tab reads the playback fps and the default frame count from the descriptor and `video.fps` (stage 8a, 2026-10-06)
  (b) add a paragraph "how to swap it into Wan" to the model card of the experimental mirror (the fine-tuned umT5)
  (c) add a check that compares the model card of the local distribution mirror with the card the recipe draws today (to find missed rebuilds)
  (d) make Wan's text stage check the `session` declared by the umT5 preset instead of silently ignoring it (same root as the component replacement item in known-issues)
  (e) make `e2e_wan_dit_test.ts` emit the VRAM record even for a failed step (to split the in-lane OOM of known-issues into ① leftovers / ② another process)
  **To add later (filed 2026-10-04, maintainer)**: (a) an f16 preset for TI2V-5B — we want the weights ready even if it does not run on the development machine (material: the f16 export is at the RAM boundary or
  needs a change in the exporter core; the development machine can hold neither checks nor sha rows — ADR 0121 decision 7 / ruling 2) (b) ✅ an option to choose the I2V aspect handling (direct resize — diffusers'
  behaviour — made selectable; the form of the option decided in stage 9) — `fit: "stretch"` of stage 9b (`c98d4747`, 2026-10-10).
  **Related (accepted, ruling 2026-10-04)**: ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) (umT5's provenance moved to the original google/umt5-xxl,
  a path to build compatible text encoders, replacement stays on today's components preset, stages a to d). Stages a to d complete (4 addenda to ADR 0122). Remaining = the 2 items of ADR 0122 "Open points" (the ruling on publishing third-party compatible components,
  agreement of graph descriptors across versions — the visual check of the fine-tuned version was ruled fine by the maintainer on 2026-10-05).
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
- **Revisit the barrel's family prefixes (filed 2026-08-25 — maintainer's intent "want to revisit
  later")**: symbols get a family-name prefix because `mod.ts` exports every family flat, but
  in the SBV2 family the "2" is misread as the conversion idiom (x2y), with real harm (`sbv2Utterance` →
  renamed to `toSbv2Utterance` to avoid it — ADR
  [0079](decisions/0079-sbv2-two-layer-input.md) decision 2). Redesign across all families the options such as bare-name exports on the subpath surfaces or
  namespace objects (once the prefix is gone,
  it can converge on `toUtterance`). It is breaking, so it rides along with the next breaking wave.
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
  **AbortSignal on the construction path is implemented for anima / irodori** (`AnimaPipelineOptions.signal` —
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
  Kill criterion: measured VRAM and speed on those shapes show no difference against today's path that falls back to f32.

- **Promote Wan's visual-check tools into tools/ (filed 2026-10-15, ruling of ⑤)**: the scripts that produce videos for seed × prompt combinations,
  `outputs/diag/wan22-visual-dist.ts` / `wan22-visual-free.ts` (not tracked by git), and the 2 Python scripts for WebP conversion and side-by-side layout
  (`wan22-frames-to-webp.py` / `wan22-ab-sbs.py`) were used repeatedly for the visual checks of ADR 0121 stages 7 and 9d. The 2 Deno scripts largely
  overlap (the differences are the text path (precomputed embeddings / free text from JSON through umT5 on the GPU) and the I2V condition-image option), so
  when promoted they are merged into one and put under fmt / lint / check (the placement rule is the entry for [assets-layout](assets-layout.md)'s
  `outputs/diag/`). Start condition: the next time Wan's visual A/B is run (the floor of T2V's 50 steps with the practical preset, the decision on whether to upload
  Wan2.1).

## release — the release preparation wave (some way off)

- **✅ 0.14.0 (2026-10-16) HF re-upload + pin update of gemma4 / gemma4-qat (ruling 8 = a, 2026-09-26)** (0.14.0 uploads only `karume-gemma4` — `karume-gemma4-qat` stays unpublished (it would be a first publication, which separately needs the upstream revision recorded and the licence checked; maintainer's ruling 2026-10-15)): reflect in the distribution the promotion of the manifest vocabulary
  `stateAttentionReduce` (Breaking — CHANGELOG `[Unreleased]`) and E4B QAT's `i4-gemvpar` / `i4-fast`.
  Bundled into the next release (how pins are baked: ADR [0073](decisions/0073-models-source-pin.md),
  [release-runbook](release-runbook.md) §2 / §3). The local mirror is rebuilt.
- **✅ 0.14.0 (2026-10-16 — re-uploaded with the window mask as table + indices) HF re-upload + pin update of birefnet-hr / lucida (E1 stage 1 — ADR [0113](decisions/0113-birefnet-weight-series.md))**:
  reflect in the distribution, including the rebuild of Lucida's f32 series (with the upstream revision — `--dtype f32`) and the f16 preset. **Maintainer's ruling
  2026-09-26 / 09-27: HR has only the `f16` preset (the upstream `ZhengPeng7/BiRefNet_HR` has a single f16 checkpoint and no f32
  — an f32 preset would only hold the same values in twice the VRAM). Lucida has f32 by default + optional f16 (f16 is lossy — mean absolute α difference at most 7.0e-5).
  The recipe changes (`dist` able to build HR with a single f16 preset, the default preset, pytest, the card's relation) and the retirement of HR's 2 f32 series +
  the f32 rows of golden / sha reference rows / `HELD_SERIES` are done when the distribution is updated (ADR 0113 addendum).** → done ahead of time on 2026-10-11 (addendum below).
  **Addendum 2026-10-11**: on the same occasion, upload the distribution with the pipeline `birefnet/2` (the post-processing method field `postprocess` — HR = `u8-pil-bicubic`,
  Lucida = `float-bilinear`, ADR [0125](decisions/0125-birefnet-matte-postprocess.md)) and update the pin of `BIREFNET_SOURCES`
  (the proviso of ruling 28 = manifest changes are bundled into the next release). main reads only `birefnet/2`, so until then
  `fromPretrained` from the published revision fails at the major-version check. The retirement of HR's f32 preset (recipe, tests,
  golden / sha reference rows / `HELD_SERIES` rows), the rebuild of Lucida's f32 series with the upstream revision and the rebuild of the 2 local mirrors with
  `birefnet/2` were done on 2026-10-11 (ADR 0113 addendum 2026-10-11). What remains for the distribution update is only the re-upload to HF
  and the pin update of `BIREFNET_SOURCES`.
- **✅ 0.14.0 (2026-10-16 — only the addition of `dit_context`; de-duplication not bundled) HF re-upload + pin update of the 2 irodori repositories (`irodori-v4-small` / `irodori-v4.1-small`) (B4 = ADR [0114](decisions/0114-irodori-dit-context-split.md))**:
  the distribution gained the role `dit_context` (breaking — a distribution without it fails with the new models as a missing component). Done on the same breaking-wave occasion as later's
  "Remove the 5,605 MiB duplication between irodori v4-small and v4.1-small with cross-repository references" (→ in 0.14.0 uploaded with only the addition of `dit_context`, without bundling the de-duplication — maintainer's ruling 2026-10-15). Local mirrors:
  v4-small rebuilt, v4.1-small being rebuilt (2026-09-27).
- **✅ 0.14.0 (2026-10-16 — `karume-umt5-xxl` and `karume-wan2.2`, new `UMT5_SOURCES` / `WAN_SOURCES`) First publication of the Wan family (filed 2026-10-09)** (0.14.0 uploads the 2 repositories `karume-umt5-xxl` → `karume-wan2.2` — `karume-wan2.1` is not published (maintainer's ruling 2026-10-15, ADR 0118 addendum). umT5 became a family with its own entry point `Umt5Encoder`, so `UMT5_SOURCES` (key `umt5-xxl`) is also added — ADR 0122 addendum 2026-10-15): in the order `karume-umt5-xxl` → `karume-wan2.1` / `karume-wan2.2` (the 2 Wan repositories reference umT5 across repositories, so
  rebuild the 2 Wan repositories with umT5's real SHA before uploading them — ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) decision 8).
  The `WAN_SOURCES` table (keys `wan2.1` / `wan2.2`) and baking the pins. §0 and "First publication of a family" in [release-runbook](release-runbook.md)
  describe only `karume-wan2.1`, so add the procedure for `karume-wan2.2` (`dist.py --pipeline wan-ti2v --ref-*`).
- **Prerequisite for the first publication of vowel-detector (filed 2026-09-24)** (first publication on hold — maintainer's ruling 2026-10-15): the recipe reads the upstream `feature_config.json`
  directly under `inputs/vowel-detector/`. This development machine holds the whole upstream repository, so before assembly
  run `cp inputs/vowel-detector/assets/feature_config.json inputs/vowel-detector/` once.
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
  Rows left as Unverified = the revision of anima-copycat-20260610 (placed by hand, no version id / sha256),
  whether the Civitai permission field of anima-wai-v1.0 can widen the upstream licence, the attribution of sbv2's style-bert-vits2 (AGPL)
  (awaiting the ruling below), the Weights license of depth_anything (before revision matching).
  **FN (`rufflet17/voice_models`) has neither a licence statement nor a README on HF, so it cannot be published** (until this is clarified).
  **Awaiting a ruling**: the `LICENSE.md` that anima bundles is the v1.2 text, but Base (`Anima-Base-v1.0-Diffusers`) is
  v1.0 (the difference is the commercial terms of §2(b)/(c)/(e) — v1.2 is looser). Recorded in the ledger as a fact.
  The Code license / Attribution of the code-dependency blocks (transformers / PyTorch and so on) were filled in on 2026-09-05 from the actual
  upstream LICENSE files. The machine-readable place for the upstream revision is the container's
  `provenance.upstreamRevision` ([container-v1](container-v1.md) §2.3). Today the recipes of sbv2, minicpm5, siglip2, depth_anything, birefnet, embeddinggemma, deberta and wan / umT5 fill it
  (through `tools/export-recipes/_shared/upstream.py`), so irodori and the others **fill it at their next re-export** (on the same occasion as the `sym_max` field of W-G4-4)
- Re-ruling on the policy of "separating the semantic surface from the implemented subset" (attention / deform_conv2d /
  gather / conv_transpose1d / upsample_bilinear2d — a unified rule that does not turn the observed subset into op semantics)
- **Consolidate CHANGELOG.md's `[Unreleased]` before 0.15.0 (filed 2026-10-27)**: the section has to describe the change from
  0.14.0, not the history of features that were added and then changed again while unreleased. Entries known now:
  ① Changed, the online route absorbing the output permute: the online route itself is unreleased (Added, online int8
  attention), so its "`FusionCounts.attentionPermute` … 180 → 240" compares two unreleased states — fold it into the Added
  entries. ② Changed, the estimate with `attentionScoreStorage: "none"` and no `attentionDevice` "now fails for every graph,
  not only …": `"none"` is unreleased, so only the current rule belongs in the Added entry. ③ Timings measured before later
  unreleased speed-ups: the Wan2.2 conv8 preset's VAE stage 87 s against the online preset's 168 s and clip 533 s /
  611 s and `vaeTile: 32`'s
  0.70–0.83× of tile 16 (both 2026-10-23, before the VAE wave b), and the Wan DiT fusion entry's 2,825 / 2,927 ms (2026-10-25,
  before A4) — re-measure on the release tree or state the date. ④ "about 45 ms" per 121-frame forward for A4, where ADR
  [0040](decisions/0040-fusion-pass.md)'s addendum of 2026-10-26 gives −44 / −49 ms. ⑤ The kernel-fingerprint entry's "the two
  embedded tables" — three are embedded now (`nvidia-ampere` was added). ⑥ The house word "seat" in a user-facing entry (the
  2026-10-25 DiT fusion entry: "under an a8 attention seat"). ⑦ A stray blank line inside the Added list (before the
  `runAttentionRouteBench` entry). ⑧ The Fixed entry (the release wait before a slot backing is built) is framed on Node +
  Dawn — say what it changes for a 0.14.0 user.

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

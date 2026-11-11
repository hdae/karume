# Backlog — the source of truth for wave order and work items

> This file alone is the source of truth for the project's **priorities, wave order and unfinished items**.
> Status vocabulary: `now` (current wave) / `next` (the next big wave) / `later` / `release` / `parked` (with a revival condition).
> Operating contract: ① completed items are **deleted** (git and the ADRs / research hold the history) ② measured values and design arguments are
> **not written** here — point to the source (ADR / research / ledger) ③ the source of truth for filing, adopting or rejecting performance candidates and for their kill criteria is
> [perf-ledger](perf-ledger.md); this file only refers to them as waves ④ the source of truth for by-design constraints is
> [limitations](limitations.md) — a constraint appears here only when a ruling turns it into work.

## now — after the 0.20.0 release (released 2026-11-10)

0.20.0 is out (JSR `@karume/runtime` / `@karume/hub` / `@karume/models` 0.20.0 — the exact expert cache of ADR 0146 with
its host tier, self-speculation and per-layer slots; the parallel-GEMV rows and the RMS→add width for 26B, with the 26B
presets `i4-gemvpar` / `i4-fast`). `hdae/karume-gemma4-qat` was re-uploaded with them (no FAILED row, the large parts at
51.2 MiB per term or more), and `gemma.SOURCES["gemma4-qat"]` moved. Nothing is breaking. Published-smoke is green on
Node and Deno.

## next — Gemma 4, in the maintainer's order (2026-11-08)

The items go in this order. The candidates of the other families wait until Gemma 4 has been gone through (the
maintainer's ruling of 2026-11-06).

1. **26B A4B (MoE)** (maintainer 2026-11-08): every expert resident ✅ 2026-11-08 (ADR
   [0145](decisions/0145-gemma4-26b-moe-resident.md) and its addendum; in the local mirror, ran in Chrome on the RTX 5070 Ti
   with about 0.8 GB of 16 GB left; released in 0.19.0). Left: the grouped prefill form of `expert_linear` (perf-ledger K-104), the drafter and vision of 26B. The dense parts'
   fusion and parallel-GEMV rows at width 2816 ✅ 2026-11-10 (`i4-gemvpar` / `i4-fast` for 26B, a decode forward ×0.70 on the
   RTX 3080 Ti, and 30.1 against 22.8 tokens per second under the expert cache in Chrome on the RTX 5070 Ti —
   [research 2026-11-10](research/2026-11-10-gemma4-26b-fast.md)). Then see whether a MoE cache that loads the
   needed experts from RAM can work, and look for a way to pay a miss as a transfer rather than a re-run (the PoC of [research 2026-08-31](research/2026-08-31-freetoken-moe-over-arraybuffer.md)
   paid it with one fence). Pre-research done 2026-11-08: the maintainer prefers an exact cache (bit-identical to
   full residency) and wants the approximate modes (skip a miss, bias routing to the cache) measured too; CPU measurements of
   decode routing on all 30 layers, the re-run count of the exact cache and the approximate modes' KL ✅ 2026-11-09
   ([research](research/2026-11-08-gemma4-26b-moe-cache.md)); the Chrome check of the fence and the per-expert upload ✅
   2026-11-09 (RTX 5070 Ti: about 2.5 ms per fence, about 0.43 ms per expert by a staging ring).
   Resuming a retry at the first missed layer stays exact but saves only about a fifth of the retry layers (the early layers
   miss most); a K that differs per layer and self-speculation over the cache each cut the time at K = 64 by about 12–13%,
   together about 15%, still about 2.3 times full residency. Ruling 2026-11-09: build it even if slow (a model whose experts
   do not fit in VRAM becomes possible in principle) — ADR [0146](decisions/0146-exact-expert-cache.md) (proposed): stage 0 the
   cached-read measurement, then the whole-token re-run, then self-speculation over the cache. The cached read ✅ 2026-11-09
   (about 2.9 ms per expert in Chrome, so a RAM tier comes after a real chat's counters). The whole-token re-run ✅
   2026-11-09 (runtime, Gemma pipeline, Playground; the cut model at 16 and 64 slots is bit-identical on the RTX 3080 Ti).
   The maintainer's chat with the full model in Chrome on the RTX 5070 Ti ✅ 2026-11-09 (32 slots: 264 ms per decode token,
   77% of the time in Blob reads — research §6.2). The host tier in RAM ✅ 2026-11-10 (`hostBytes`, plain `ArrayBuffer`s
   and the staging ring — ADR 0146 addendum 2026-11-10; the cut model is bit-identical with it). The maintainer's
   chat with the tier in Chrome on the RTX 5070 Ti ✅ 2026-11-09 (every expert in RAM: 9.7 tokens per second at 32 slots,
   18.1 at 64, 52.3 resident; the same tokens — research §6.3). Self-speculation over the cache ✅ 2026-11-10 (`expertCache.depth` — ADR 0146 addendum
   "scheme B as built"; the cut model gives the tokens without the cache at temperature 0 and 1). The maintainer's
   Chrome chat at 64 slots depth 4 ✅ 2026-11-10 (the same tokens as resident; 17.4 → 20.8 tokens per second, 1.6 runs
   per decode token — research §6.4). A K per layer ✅ 2026-11-10 (`slots` as an array, the Playground preset `fitted` —
   ADR 0146 addendum "per-layer slots"; the cut model gives the tokens without the cache). The maintainer's Chrome A/B ✅
   2026-11-10 (the same tokens; `fitted` at depth 4 decodes 22.6 tokens per second against 20.5 at 64 — research §6.5).
   The cost per run ✅ 2026-11-10 (the drafts' whole forwards and the verify padded to 8 rows — research §6.6); the
   verify in a bucket of `depth + 1` rows ✅ 2026-11-10. Next, by the measured share: `expert_linear` reading each
   distinct expert once and skipping pad rows, a depth sweep in Chrome, drafts chained on the GPU, and the load after a
   cut for its last row alone.
2. **Vision on E2B / E4B** (V8 of ADR [0144](decisions/0144-gemma4-12b-vision.md); later by the maintainer's ruling of
   2026-11-07). 12B's vision shipped with 0.18.0.

## later

- **ADR 0147 follow-ups — numerics and kernel adoption (filed 2026-11-10; ADR [0147](decisions/0147-numerics-and-kernel-adoption.md)
  accepted, nothing below is built; the order among them and against the other items is the maintainer's)**. Each code item is its
  own commit with its behaviour tests and states its blast radius:
  - **Code** (the decisions name what each changes):
    1. The three fusion keys leave `SessionOptions`, the type checks and the combination rule; the fusions run by default
       (decision 5). The test-only switch that disables the fusion rules has landed (`DISABLED_FUSION_RULES`, with
       `checkFusionTwin` in `packages/kernel-checks` as the comparison — 2026-11-11); the tests that obtain the unfused
       plan by omitting the option (the Session baseline of `gpu_linear_static_quantize_session_test.ts`, the references
       of `gpu_rms_norm_add_fusion_test.ts` and `gpu_gelu_mul_static_quantize_fusion_test.ts`, and the grid-stride
       reference of the RMS→add kernel check) move to it in this change. The reference sha rows stay; the fusion-count
       tests, `EXPECTED_FUSIONS` and the census rows that pin fused counts change. Breaking (surface snapshot, CHANGELOG); a
       cross-cutting change, so the full `vp test`.
    2. The same keys leave the manifest vocabulary (`SESSION_KEYS`, `SessionSpec`, `parseSession`) and the recipes' quant
       tables (`gemma4_qat`, the q4_0 QAT quants of `gemma4`); the mirrors are regenerated and `verify_dist` is green; the
       re-upload, the source pins and the published smoke wait for the maintainer's explicit go-ahead (decision 5).
    3. The Session canary for every default-on fusion: fused against unfused on a canary input as u32, a mismatch warns and
       drops the rule on that device, memoised per device, with a fault-injection test (decision 9). The input size is
       decided and the start-up cost measured at implementation, and the cost is reported to the maintainer.
    4. The derived applicability check at admission replaces the twelve `FamilySessionPolicy` tables: one pure runtime
       function shared with the recipe builders' predicates; the structurally impossible combinations of decision 7 move
       from the first run to admission (tested with a fetch spy); the linear rejections become one shared violation function
       used by admission, `buildLinear` and the estimate; `WRITERS` exhaustiveness, the error type by origin and the routing
       to component Sessions stay (decision 7). MOSS-Image's knob table stops gating, and in the same commit a test compares
       the `session` each preset declares in the distribution mirror with the knob values its checks were run on.
    5. The public applicability query API (user story, export diff, surface snapshot, CHANGELOG) and one line on how to query
       in the model card; the fixed override type covering every numerics option of `SessionOptions` replaces
       `SessionOverrides`, and gemma's explicit `rmsNormReduce` / `fullStateStorage` fold into it (decision 7).
    6. Device fallbacks become warned switches: f16 without `shader-f16` → `"f32"` (the throw is removed); `parallel-subgroup32`
       → `"parallel"` and `rmsNormReduce: "subgroup32"` → `"workgroup"` without subgroups; a warning on online attention's
       three-dispatch fallback; a shared GPU warns at admission, a self-acquired device requests only the available features;
       a manifest's `gpuFeatures` is a request; the attention dp4a canary with both arms outside the band warns and runs that
       attention in f32, and its `emu-exact` verdict warns once (decision 8). Today `sessionGpuFeatures` covers only
       `shader-f16`, so a subgroup32 option on a shared GPU without subgroups throws at construction after the fetch; this item
       removes that.
    7. Run-time canaries for linear and conv3d a8's dp4a choice and for online attention's `sg32` / `wg` choice on devices
       where the path is requested (decisions 6 and 9).
    8. `/kernel-check` covers the fusions (fused against unfused, u32) and the default kernels; the tune sweep record and
       the `/kernel-check` record gain a backend field (how it is filled is an implementation-time item); where the per-GPU
       evidence records live is decided with it (decision 9). (Updated 2026-11-11: the fusions landed — the page's
       "Fusion rules: fused vs unfused" runs the built-in fusion twins from `packages/kernel-checks/src/fusion-twins/`, the
       cases the runtime's twin-graph tests run; so far run on Dawn only. The default kernels, the backend field and where
       the records live remain. Updated 2026-11-11: the page also runs "rms_norm: prefetch form vs runtime-trip form"
       (`checkRmsNormPrefetch`, the cases of `gpu_rms_norm_prefetch_test.ts`), and the rmsNormRope twin runs at five more
       row widths, to tell a backend difference of the prefetch form from one of long rows.)
    9. Wiring: gemma forwards `linearCompute` once the derived check admits it; Wan's text stage passes umT5's options
       (wiring only — no umT5 `i8-a8` preset, perf-ledger K-80 / K-81) (decision 7).
    10. `packedStaticQuantize` passes NaN through (so it becomes E) (decisions 2 and 5).
    11. `packedStaticQuantize`, `integerGemv` and `linearGemvPlanes` assessed against decision 6's test (certainly on / off);
        each that fails becomes a string union at the next breaking change of this area or at 1.0, whichever comes first.
    12. GPU checks: no grid-stride kernel's numbers depend on the workgroup count (a `LIMIT_CAPS` Uint32 test); bit identity
        under an injected geometry profile (decision 8, device-driven choices).
  - **Checks**:
    1. Fault injection on the core ops (matmul, conv, …) for check C (decision 4).
    2. Gemma's E2E self A/B check, held under the grace clause until it exists (decision 2).
    3. The grace clause's missing points: parallel GEMV's ④ and worst-case record; the subgroup32 reductions' ② and ④
       (decision 2).
    4. The checks of decision 4 (A, P, F, B, G, E): before a `KARUME_UPSTREAM_PARITY` file is retired it is mapped to its
       replacement (a file without one keeps its opt-in lane); patch fidelity for the recipes without a `--verify` mode
       (deberta, embeddinggemma, gemma4, gemma4_qat, irodori, minicpm5, moss, qwen3_5 on 2026-11-09); deviation records per
       family, starting from BiRefNet's post-processing (ADR 0125) and Gemma 4's vision masks (ADR 0144 decision 1).
  - **Records**: Wan2.2's untiled VAE distance (torch tiled against untiled on the same latents — an ADR 0121 addendum and
    ADR 0147 decision 10); the inventory of published presets with no recorded verification (the self A/B check, the
    census-table row, the quality record — human, machine or none; for example the regular Gemma 4 `bf16` default has no
    recorded human review, ADR 0128 addendum 2026-11-04, while decision 3 (2) never skips the default preset's); the full
    inventory of device-driven choices of 2026-11-09 into `docs/research/` (decision 8); the first per-GPU evidence records on
    the two everyday environments (decision 9).
  - **For a later ruling**: whether Anima's sha rows, recorded on its practical preset, move to its reference preset (decision
    11; re-recording needs the GPU); the wording of ADR 0126's re-entry condition 2 and of perf-ledger K-80 / K-81, which name
    the acceptance tables decision 7 removes (changed only with the maintainer's agreement); whether the old rule "a band's
    derivation table sits in the docstring of the test constant" (ADR 0058 decision 4 ②, not restated by ADR 0147) should be
    restated.
  - **Stale statements found on 2026-11-09 (report only, not yet fixed)**: the `DECIDED` comment above `PARALLEL_SHAPES`
    ("only measured shapes are eligible") against its 22 rows marked unmeasured (ADR 0098 decision 4 was re-graded to SHOULD);
    ADR 0111 decision 2's list of tables; the table comments of gemma (`session-options.ts`) and anima (`pipeline.ts`); the
    rmsNormAdd widths in the `SessionOptions` doc (3840 is missing); `session_options_test.ts` checking "all 10 families" (moot once the tables go); ADR 0033 decision 4's test
    cite; ADR 0132 decision 5's v1 workgroup-storage figures (the v2 figures are in its later addendum); the unreachable w4a8 `k % group`
    check; the online kernel's `:emu` against `:dp4aEmu` in the other i8a8 pipeline keys.
- **Gemma 4 mobile QAT decode speed (maintainer 2026-10-30)**: 0.15.1 took K-97, K-48 step 1b and K-99; the integer GEMV
  (K-45, ADR [0140](decisions/0140-integer-gemv.md)) and the byte-plane lm_head (K-52, ADR
  [0141](decisions/0141-byte-plane-gemv.md)) are opt-in and declared by no preset. Left: their speed away from the Apple
  M2 (the RTX under Dawn, D3D12), the decision to declare them in a preset (ADR 0147 decision 3 (2)), K-48 step 1, K-98, and QAT
  E4B's parallel-GEMV rows, which are unmeasured copies (K-59). The filed comparison with regular E2B's int4 presets
  ended when they were removed on 2026-11-04 (ADR 0128), so the baseline has to be chosen again.
- **Regular E4B's MTP drafter (maintainer 2026-11-01)**: `karume-gemma4`'s E4B ships no drafter:
  `google/gemma-4-E4B-it-assistant` is not exported, and `gemma4.export_drafter` accepts E2B only (the q4_0 QAT line's
  E4B has its QAT drafter from 0.16.0). Scoping memo `.claude/reviews/2026-11-02_gemma4-expansion/e4b-drafter.md`
  (local, untracked).
- **Follow-ups of the Gemma 4 lines (filed 2026-11-04)**:
  - `tools/llm-baseline`: the mobile QAT profiles score with their checkpoint's own tokenizer, which may differ from the
    one `gemma4QatMobile.SOURCES` ships (the regular `-it` repositories' fixed tokenizer — 0.16.0 Changed); its README
    Validation section still says `deno task verify`. No scoring run since the schema 4 / 5 reader (local `inputs/`
    lacks some checkpoints).
  - Function calling and thinking (ADR 0084 addendum 2026-11-04): the `Pipeline.chat` one-shot entry takes no `tools` /
    `enableThinking` (only `chatPrompt` and `ChatSession` do); the parser's exact-integer check reads `JSON.parse`'s
    reviver `context.source`, not verified on Safari.
- **Remainder of the speed / memory wave (filed 2026-09-25; maintainer's ruling = split into two tiers, reference tier / practical tier; within-device determinism MUST hold in the practical tier too)**:
  The finished records of iterations 1 to 3 (contract and groundwork, B3 anima DiT residency, B4 irodori context K/V projection, H-35, K-70, E1 stage 1) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Candidates for the next iteration = wave 2 (DiT speed: B1 → the rest of B2 = perf-ledger F-6) / wave 3 (memory: rest of E1 → E2 → E5 → E4) / wave 4 (practical-tier kernels: the conv2d half of C2 = perf-ledger K-2 → D1 → the rest of D3 = perf-ledger K-95, K-57).
  The candidates come from `.claude/reviews/2026-09-25_perf-recon/SUMMARY.md` §9.3 (not tracked by git) and the perf-ledger section of 2026-09-26 (K-59 to K-69, L-20 to L-23).
  Rest of E1 (BiRefNet's f16 series — ADR [0113](decisions/0113-birefnet-weight-series.md)) = derivation of the Lucida f16 2048² synthetic tolerance (`LUCIDA_F16_2048_SYNTHETIC_TOLERANCE` in `packages/e2e/tests/e2e_birefnet_test.ts` is still a placeholder taken from the f32 series; the RTX 3080 Ti can run it) and phase 2 (the i8 plan + i8 series).
- **Remainder of geometry profiles (K-71 — ADR [0115](decisions/0115-geometry-profiles.md) / [0116](decisions/0116-geometry-profile-row-buckets.md) / [0117](decisions/0117-app-geometry-tuning.md)) (filed 2026-09-27 to 10-03)**:
  The finished records (registration of `nvidia-blackwell`, the 7 row-count buckets, integration into applications (ADR 0117, all 7 stages ✅), the Codex review follow-up, 4 small items) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  ② Regenerating `apple-metal-3` from the sweeps of one Chrome version (the table mixes Chrome 153 and 154) happens the next time a sweep is taken on M2, on the playground's `/geometry-sweep` page (ruling 2026-10-02)
  ③ The Pixel 10a (Mali-G715) table is not registered; the re-run on the new version has not been done (research K-70 §15)
- **Remainder of the container-format wave (ADR [0108](decisions/0108-container-format.md)) = stages 4 to 6** (upstream intake of the anima transformer f16 / LoRA runtime A/B / new bit widths, Range fetching and so on —
  the staged-decomposition table of ADR 0108 is the source of truth for what is built and how it is accepted). The records of stages 0 to 3 and up to the 0.13.0 publication (2026-09-25) are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
- **Remainder of the cross-model follow-up investigation (2026-09-10 to 11)**: the source of truth for the measurements and the unfinished parts is the [follow-up investigation](research/2026-09-10-codex-mtp-optimization.md#追加-llm-の実行と量子化別比較).
  The acceptance of Qwen3 / MiniCPM5 / E4B / gemma4-qat and the adoption records of K-25 to K-44 are in the [archive](research/2026-10-10-backlog-archive-0.12.0-to-0.13.0.md).
  Remaining = the cost study of sharing the gate/up input (perf-ledger K-50, whose step 0 measurement has not run); isolating the cause of the GPU Instance loss in Chrome;
  distribution recipe / source table and long-text and broad quality evaluation for Qwen3-0.6B and MiniCPM5; transfer reduction for general sampling (with temperature, penalty / bias, speculative) and a follow-up test on M2.
- **Continued verification of the 9/11 review** (investigation 2026-09-10, [follow-up record](research/2026-09-10-codex-mtp-optimization.md)):
  Decide whether `artifacts.staged_publication` allows several writers to the same final, and if needed design locking and recovery from interruption.
  The follow-up tests of the GPU edge-channel protection on M2 / browsers and whole-model performance measurement remain.
  Do not turn review suggestions into new kernels without verification.
- **Move `planBackingBudgetBytes` into the common options** (filed 2026-09-07 — consequence of ADR 0095): every family except gemma4 builds its
  Session options from the manifest's `session`, so there is no way to change the budget (the default applies). Add it the same way `onRetry` was
  unified into `FromPretrainedHubOptions`. At the same time, merge the budget range check duplicated in `estimate.ts` and `session-options-check.ts` into one function.
- **Anima: add `text+transformer` to the residency values (filed 2026-10-01, maintainer)**: today `transformer` keeps only the DiT Session across generate calls, and
  text_encoder / text_conditioner redo "read → upload → build the Session → tear down" every time (the default VRAM assumption is "one largest stage" —
  ADR [0112](decisions/0112-anima-transformer-residency.md)). The de facto practice in PyTorch / ONNX Runtime is "once loaded, resident until explicitly freed", and
  the maintainer's policy is "the API shape is explicit release; the default is resident as far as VRAM allows; demote automatically when it does not fit". Put the two text Sessions on the existing eviction
  state machine (trial allocation for the next stage → evict if it does not fit → retry). The default is unchanged. The gain depends on the environment (the slower the fetch, the larger the gain). Before starting, measure the VRAM peak including the resident part on M2 / Pixel.
- **Remaining test coverage (filed 2026-09-05)**: the copy branch in `#encodeTimedChunk` of `SubmitScheduler`
  (`packages/runtime/src/gpu/submit.ts`) is still untested.
- **Isolation experiment for `gpuTiming: true` on Metal through Deno's wgpu**: an A/B of which dominates, the number of query sets alive at once or the
  backlog of `destroy()`. The `--diagnostics` flag left with `examples/gemma4` in 0.15, so the run needs a small Deno script. The relevant section of [known-issues](known-issues.md) is the source of truth for steps ①② and the fix candidates.
  It needs real hardware, so the user runs it.
- **The remaining medium-priority item of the diff review's deferred table** (source of truth = `.claude/reviews/2026-09-03_7fc4ada/ROADMAP.md`
  — not tracked by git): W-G4-4 record the origin of the chunk limit in the provenance field `sym_max` and have dist compare it (**rides along with the next Gemma 4 re-export**).
- **Peak VRAM column on the model card (filed 2026-09-04)**: `karume dist` calls the TS-side estimate
  (the `prepareContainer(...).estimate()` family) when generating the card and adds a peak VRAM column to the quant table. Today's
  card shows only the stored bytes, so readers cannot judge whether the model runs on their GPU.
- Follow-up test of the RAM peak of the HF path in Chrome (measured on the Deno side —
  [research 2026-09-24](research/2026-09-24-part-length-ram-peak.md)).
- Collecting performance information from users' environments: a feature in the TS packages that collects it, and the
  results of users running a sample collection on several environments (**filed only** — the items collected, the
  location and the opt-in form are not designed; the per-run JSON records of `apps/playground` are the nearest existing
  piece).
- Re-assessing `err.cause` on Pixel (8 GB class Android Chrome) — [known-issues](known-issues.md).
- **Follow-ups of accepting ADRs 0127 / 0128 / 0129 / 0131 (filed 2026-10-29, the maintainer's ruling 34 a — each ADR's addendum of
  2026-10-29)**.
  - Geometry sweep identity check for a bf16-stored linear (ADR 0131 open point 2, low priority).
  - Move the Node read-strategy default into `@hdae/fetch-cache` (ADR 0127 open point, after 0.15); hub's probe in
    `packages/hub/src/sources/hf.ts` can then go.
  - Deno's resource / op sanitizers have no replacement under Vitest (ADRs 0127 / [0129](decisions/0129-test-layout-and-lanes.md) open
    point 5): record the loss in [limitations](limitations.md); file a GPU resource-leak check only if a leak shows up.
  - The order in which the rest of the removed 0.14.0 tools and examples (v0.14.0 tag) come back as `apps/playground`
    pages (ADR 0127 decision 5): the tools `anima-residency`, `birefnet-ab`, `flag-bench`, `fusion-hints`, `llm-speed`,
    `mtp-bench`, `opbench`, `ram-peak` and `wan-seams`, and the examples `minicpm5`, `qwen3`, `vowel-detector` and `wan`
    (Wan2.1). `fusion-hints`' search itself is back (2026-11-11) as the fusion candidate search of `packages/e2e`
    (`tests/helpers/fusion-candidates.ts`, and the opt-in report `KARUME_FUSION_CANDIDATES=<dir>` of
    `e2e_fusion_candidates_report_test.ts`); only its page is still to come.
  - Move the kept optional asset cases (`gemma4_config`, `gemma4_tokenizer`, `gemma4_chat`, `wan_pipeline`, `census_table` in
    `packages/models/tests`) into `packages/e2e` (ADR 0129 decision 1).
  - Small doc edits: `packages/e2e/README.md` names `outputs/series/` but not `outputs/test-assets/` (ADR
    [0139](decisions/0139-tests-on-distributions-and-test-assets.md)), where the goldens now live; ADR 0120's pointer at
    `WAN_DEFAULT_QUANT` in `tools/export-recipes/wan/distribution.py` carries a line number that has drifted again —
    drop the number.
- **Bundle-size remainders after the models barrel fix (filed 2026-11-04 — ADR [0037](decisions/0037-karume-monorepo.md) §4 addendum)**:
  - A bare import of the `@karume/runtime` barrel still keeps a large bundle under Rolldown (not investigated — the size is in the ADR).
  - esbuild still keeps part of the `@karume/models` barrel that Rolldown drops.
  - `KARUME_SOURCES`'s type (`packages/models/src/sources.ts`) is spelled by hand next to the pure-annotated build, so a
    family table added to one and not the other goes unnoticed by the type check — derive one from the other.
  - `@hdae/yomi` is a dependency of `@karume/models` that only tests use.

- **MOSS-Image (the subpath `@karume/models/moss`, published on Hugging Face as `hdae/karume-moss-image` with 0.15.1 — ADRs [0138](decisions/0138-moss-image-dit-vae.md) and [0137](decisions/0137-qwen3_5-text-encoder.md), accepted)**: what remains is other GPUs' sha rows (this family's rows exist for `node-nvidia-geforce-rtx-3080-ti` only — ADR 0106) and the text encoder at T = 4,096 (ADR 0137 open point 2). Its speed candidates are perf-ledger F-10 to F-12.

- **gemma4 prefill in w4a8 (perf-ledger K-60, maintainer 2026-10-17 "in the backlog")**: design of an acceptance table that keeps decode on the M=1 GEMV and puts only prefill (chunks of M=768) on `linearCompute: "a8"` × i4 (switching by M). Today's a8 falls to the GEMM skeleton even at M=1 and is 8.45 times slower than the GEMV ([research 2026-09-19](research/2026-09-19-qat-speed-recon.md)), so opening the acceptance table is not enough.

- **Root fix for the rows that collapse under umT5's a8 (perf-ledger K-81, filed 2026-10-18, maintainer: "root fix first, not even opt-in")**: per-token int8 crushes the remaining components in the rows of function words (spaces, commas, articles, suffixes) ([research 2026-10-18](research/2026-10-18-a8-umt5-sbv2.md) §6; the main cause is token-specific outliers in the input of the FFN exit `wo`; mixing a8 per linear or layer does not remove it). Candidate = per-row mixing (only rows whose crest exceeds a threshold take the f32 path — row routing in the runtime's `linear-i8a8`). An equivalent transform on the export side only helps fixed channels. Once this is closed, re-evaluate umt5's `i8-a8` (the adoption line and the work needed are in K-81).

- **Scope the geometry sweep's `caseSet` per op family (filed 2026-10-27; the maintainer's ruling 2026-10-26: an amendment of ADR 0117 decision 8, better long-term, later)**: `caseSet` (`sweepCaseSetId()` — the provenance field of ADR [0117](decisions/0117-app-geometry-tuning.md) decision 4, carried in the sweep record and checked by the generator under decision 8; the change touches decisions 4 / 8) is one hash over every case of `SWEEP_CASES`, so adding one op family's cases moves it for every table: every embedded table's provenance and every saved table mismatches (`geometryProfileMismatch`) and needs a re-sweep or a re-derivation, and a table whose records have no case of a class still carries the new `caseSet` — `apple-metal-3` carries it although its `i8a8.conv3d` is the default by absence ([ADR 0115](decisions/0115-geometry-profiles.md) addendum decision 11).

- **Execution form with f16 activations (w16a16) (filed 2026-10-11, maintainer: "will do it next time")**: today the runtime can store weights in f16 or bf16, and `linearCompute` / `attentionCompute: "f16"` drop only the kernels' shared tiles to f16 (ADR 0028); the activations between ops stay f32 (ADR 0006 rejected f16 activations). The value is the intermediate memory, which pays off where intermediate tensors take most of the VRAM, as in BiRefNet's 2048² (ADR 0113). It requires `shader-f16`. Bit-exact agreement with upstream w16a16 is not expected (the kernels differ — handled at the numeric level of the ADR 0005 addendum of 2026-10-11). Preliminary research: [research 2026-08-04](research/2026-08-04-intermediate-f16-design.md).

- **Decouple the test helpers results.ts / reference.ts from GPU discovery (filed 2026-10-11, Codex deep review R8-008, maintainer's ruling = file only)**:
  `ENVIRONMENT` in `packages/runtime/tests/helpers/environment.ts` is fixed by calling `requestAdapter()` at module evaluation.
  `helpers/results.ts` and `helpers/reference.ts` import it as a value. So `results_writer_test.ts` / `reference_fixture_test.ts`, which say
  "no real GPU needed", also search for an adapter at import time. They do not take a device, and on a machine without an adapter
  the value stays null and they proceed, so the tests do not fail. The only real cost is the `requestAdapter` call. Both functions already take `options.environment`.
  What remains is that the value import of `ENVIRONMENT` runs at module evaluation. Lazy acquisition does not fix it — `openReferences`
  reads the environment key synchronously when tests are registered, and `openResults` is also a synchronous API. The fix would split the two into "a body that takes the environment as an argument" and
  "a thin wrapper that binds the default `ENVIRONMENT`", and point only the CPU tests at the body (the GPU tests' callers do not change).
  After the fix, confirm that with `navigator.gpu` undefined, importing the two tests does not call `requestAdapter`.
  A draft of this split (bodies `results-core.ts` / `reference-core.ts` and a gate test) was made on 2026-11-04 outside
  the approved scope and kept as a local patch, not applied: `.claude/reviews/2026-11-04_overnight/decouple-results-reference.patch`.
- **Out-of-scope remainder of the Codex deep-review fix wave (filed 2026-10-11)**: of the observations that the fix wave's implementers and reviewers recorded as out of scope,
  those not in backlog / known-issues and confirmed still present on main on 2026-10-11 (one per line; the IDs in parentheses are
  findings in `.claude/reviews/2026-10-04_de127de1/` (not tracked by git)). The order of work is undecided.
  - Anima's per-stage estimate (review of ruling 6): `stageNeed` in `packages/models/src/anima/pipeline.ts` estimates the text and VAE stages with no attention knobs, and `withStage` opens those Sessions with `{}`; the two agree only because both default to f32 (the preset's knobs apply to the DiT alone). Candidate: one shared constant for the non-DiT stage options, used by both. Passing the preset's `attentionCompute` / `attentionScoreStorage` there would count a form those Sessions do not run.
  - Output extent of unbatched GEMM params (filed 2026-11-04): `linearParams` (and the GEMV / i8a8 params that share `gemmParams`) do not check the output `m·n` against u32 (matmul and bmm do). `codegen_wgsl_test` pins `[65, 268435455, 16]` as accepted for linear, so that case needs a ruling first.
  - Batch-multiplied flat indices of fused attention params (filed 2026-11-04): `attentionQkParams` / `attentionPvParams` and their i8a8 variants do not receive the batch (B·H on `wid.z`), so `B·H·M·K` and the like are not checked against u32.
  - deform_conv2d i32 coordinates (filed 2026-11-04): the WGSL computes `i32(oy) − i32(dims.padding_h)`, and neither the params (padding is checked only as u32, with no `assertConvAxisExtent` equivalent) nor the contract layer (the attr is a safe integer ≥ 0, the shape only requires an output length ≥ 1) bounds it.
  - The post-acquisition GPU check `assertAcquiredGpuCapabilities` (`packages/models/src/session/gpu-features.ts`, filed 2026-11-04) is not used by moss (`src/moss/pipeline.ts`), the umT5 encoder (`src/umt5/encoder.ts`) or Gemma (`src/gemma/pipeline.ts`), nor by the shared-GPU admission sites of the families; they still write the features + limits checks by hand.
  - PLE index schema check in Python (filed 2026-11-04): `read_ple_index` in `tools/exporter/src/karume/migrate.py` tests `schema in (1, 2)`, so `true` and `1.0` / `2.0` pass (Python's `True == 1`); align it with `packages/models/src/gemma/ple-index.ts` (reject bool first, as `_require_version` in `container.py` does).
  - `packages/models/tests/helpers/moss-dit.ts` keeps its own `readBuffer` / `viewOf` / `floatsOf` (the contract of `helpers/safetensors.ts`); fold it in (importers: `e2e_moss_ab_test.ts`, `e2e_moss_dit_test.ts`, `e2e_moss_generate_test.ts`, `moss_sampler_test.ts`, `moss_dit_host_test.ts`, `helpers/moss-text.ts`, `helpers/moss-vae.ts`) (filed 2026-11-04).
  - A vacuous negative assertion (filed 2026-11-04): `packages/runtime/tests/estimate_test.ts` asserts that the estimate's notes do not include a phrase the estimate never prints (already so before the ADR 0130 translation — the notes were translated in 2-9), so it cannot fail. Pin the phrase the real notes would print instead.
  - Backing difference between runtimes: the cause of the different DiT backing between Deno and Chrome for Wan2.2's 121 frames with the preset `f16+dit8-a8-attn8-s16` (ADR 0124, "Alternatives not taken"; material in [research 2026-10-11](research/2026-10-11-row-block-budget-stage-4b.md)) is not investigated, and Node + Dawn, the development runtime since 0.15, has not been compared. The candidate explanations recorded then are the geometry profile (Deno ran on `default`) and `maxBufferSize` (how regions are split).
  - f32 overflow of SBV2's `z_p`: even when `noiseScale` is finite in double precision, `buildZp` in `packages/models/src/sbv2/host/latent.ts` puts products above the f32 maximum (for example `noiseScale = 1e39`) into a `Float32Array`, turning them into ±Inf, and passes them to the voice graph as they are (the same hole as the `z_noise` check; the GPU outputs `m_p` / `logs_p` are mixed in, so a design for which error type is thrown is needed).
  - The copy of the repository name rule in models: `REPO_SEGMENT_RE` / `assertRepoSpelling` in `packages/models/src/hub/repo-ref.ts` hold a copy of the same rule as hub's `isRepoName` (`packages/hub/src/manifest.ts`) — making it one requires exporting the predicate from hub's mod.ts (an addition to the public API).
  - Length 0 in the CPU reference's `allclose`: `packages/runtime/src/reference/allclose.ts` returns `pass: true` for two length-0 inputs — whether to allow empty input or throw has not been ruled on, and `packages/runtime/tests/reference_ops_test.ts` pins only the currently observed value (review L-R3-4).
  - Writes to pad rows in state attention: the variants in `packages/runtime/src/kernels/state-attention.ts` and `state-attention-stats-pv.ts` each write the same guard separately — a candidate for one shared WGSL fragment (today a static test pins the form of all variants, so a recurring mismatch would be detected).
  - Bilinear interpolation weights: unlike torch's `guard_index_and_lambda`, λ in `packages/runtime/src/kernels/upsample-bilinear2d.ts` is not clamped to [0, 1], so only for shapes where one axis exceeds 2^24 and the source coordinate goes past the edge might the end points not be exact (not reproduced; out-of-range index reads are blocked).
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
- **Return Wan2.2's 121-frame self A/B check to the default lane — dealing with the duration of the full verify (filed
  2026-10-09, maintainer's ruling = later)**: ADR 0121 stage 7's self A/B check is held at two points, 33 frames and 121
  frames (the same "default and longest" rule as Wan2.1 — attention's 8-bit error grows with row length). Putting the
  121-frame part in the default lane would lengthen the full verify, so it is opt-in for now (`KARUME_WAN_TI2V_121F=1`),
  and collapse on long rows is caught only when the opt-in is run. Measures (shortening the duration, changing where it
  lives) are to be considered separately.
- **In-flight submit bound for enqueue in browsers (filed 2026-10-07, maintainer's ruling = later)**: ADR 0123 put the bound on run only
  (if enqueue waits, the call order against another Session's enqueue breaks). To bound enqueue as well, first add a mechanism that runs the bodies of the enqueues in a batch one at a time
  in acceptance order (a per-batch FIFO), then wait at the end. The same mechanism also closes the existing case where the first enqueue (a body with a derivation `await`) can be
  overtaken by a later enqueue of another Session. When to start: when running long batches (video VAE, generation loops) in Chrome
  shows congestion in submission or device lost. There is no actual failure so far (the 121-frame VAE, about 20,000 submits, also completed).

- **Two anima parts read as fragmented in the upload check (filed 2026-09-24; ruling 2026-09-24 = publish as is and act after measuring the real download speed)**: in the fragmentation table of `tools/release/hf-upload.zsh`, part 3 of `text_conditioner` in `anima-turbo-v1.1` and `anima-v1.0` (byte-identical to each other, different from the same part of the other 3 variants) and `shared/text_conditioner/model.bf16-00003` stay below the 10 MiB/term guideline (the [research](research/2026-08-09-xet-fragmentation.md)). The 0.15.0 re-upload did not clear them: the cause is sharing between byte-identical or near-identical parts inside the repository, not cross-repository reuse. Repositories with a published pin do not use delete → recreate (runbook §2 MUST NOT), so decide whether to accept it once the real download speed has been measured (not measured yet), or find another remedy.

- **Revisit the structure of the manifest (`karume.json`) — make optimization flags and presets easier to handle (filed by the maintainer 2026-10-06)**: today the bundle of execution knobs
  is held in the single `session` of a quant preset, and the preset name doubles as the bundle's name (ADR [0074](decisions/0074-quant-seat-naming.md) /
  [0147](decisions/0147-numerics-and-kernel-adoption.md) / [0111](decisions/0111-session-options-composition.md)).
  We want a form in which optimization flags and presets (bundles) are easier to handle. What belongs where (storage dtype, execution knobs, bundles, per-component declarations and so on)
  is to be sorted out later. The material is the open points of the inventory of how fast paths are expressed (2026-10-06, `.claude/reviews/2026-10-06_fast-path-inventory/`, not tracked by git)
  — there is no per-component session (the fast paths of the VAE / text encoder cannot be declared), the keys of new kernel variants, and whether changing a bundle
  changes an existing preset or creates a new one. **Work starts later** (maintainer 2026-10-06 — Wan2.2's optimization proceeds within today's `session`).
- **Candidate for bringing HTTP Range fetching (ADR 0108 stage 6) forward (filed 2026-09-24; not decided yet)**: the
  condition for bringing it forward in ADR [0109](decisions/0109-manifest-v5-container.md) decision 7 is "in the RAM
  peak harness of stage 2, the cold peak exceeds 'part length + overlap'". Stage 3e removed the duplicated holding, but
  the cold peak of the scan type (Deno and Node via HF, the fetch layer's `stream` strategy) still exceeds part length +
  the largest block. The condition formally still holds ([research record](research/2026-09-24-part-length-ram-peak.md)
  item 7, ADR 0108 addendum 5). Browsers (Blob = seek) and local sources are not affected. Its motivation differs from
  that of parked "hub Range parallel + prefetch" (a fragmentation measure).
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
- **The `gemma4_qat` fixture tests are skipped wholesale in CI (filed 2026-09-25)**: `series_fixture.py` and
  `test_fixed.py` under `tools/export-recipes/gemma4_qat/tests/` have a module-level
  `pytest.importorskip("transformers")`, and `transformers` exists only in the `gemma4-qat` group, so in CI's recipes
  job, which runs with the default sync, `test_audit.py`, `test_fixed.py` and `test_plan.py` are skipped at collection.
  To decide: a) stub `transformers` and keep 1 to 2 upstream comparisons (mind `isinstance(QuantizedEmbedding)` in
  `checkpoint.py`), or b) add `--group gemma4-qat` to CI (a trade between CI time and the strength of the upstream
  comparison).

- **Leftovers of the code-quality wave (filed 2026-09-22 — source: the same wave in the
  [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md) (after the 0.12.0 release))**:
  - Where to put the four functions of `packages/models/src/gemma/pipeline.ts` tied to `gemma.PipelineOptions`
    (`resolveGemma4PleResidency` / `buildGemma4Program` take it, `assertSpeculative` takes its `speculative` field;
    `speculativeSetup` no longer depends on the type, so it may not belong in the move) (creating `pipeline-options.ts`
    was postponed by the ruling of 2026-09-21).
  - sbv2's check of the Unicode range tables (`parseRanges` in `sbv2/text/asset.ts`) differs in wording and stays separate from the shared `parseCodeRanges` in `text/code-ranges.ts`.
  - Splitting `reference/ops.ts` and `ops/shapes.ts` (info — no mutable state and no cycles, zero real harm).

- **Remainder of decode speed (moved from now on 2026-09-20)**: small items K-48 stage 1 (rms_norm→SRQ fusion) / K-49
  (slice aliasing) / K-50 (k+v concatenated GEMV), K-46 (int8 KV — a memory item). Revival condition: when decode speed
  becomes the topic again (re-measure the TTFT attribution of research 2026-09-19 §15.2 first — "the prefill arena path"
  is not in hit runs, as corrected in the addendum of 2026-09-25). The adoption of candidates and the kill criteria are
  in [perf-ledger](perf-ledger.md), the attribution in [research 2026-09-19](research/2026-09-19-qat-speed-recon.md).

- **Fusion hit-count checks for the families without one (filed 2026-11-11, with the fusion candidate search)**: a fusion that
  stops matching only moves performance, so a family without a hit-count check (`packages/e2e/tests/e2e_<family>_fusion_counts_test.ts`)
  would not notice. MOSS-Image's DiT stage gained one with the candidate search; still without: sbv2 (DeBERTa included),
  umT5, BiRefNet, Depth Anything, SigLIP2, vowel-detector, Qwen3.5 (MOSS-Image's text encoder), Gemma 4 26B and the q4_0 QAT
  line, and Irodori v4.1-small. The opt-in candidate report already binds and plans those with a local mirror.

- **Ruling on the contracts that block large in-layer fusion (filed 2026-09-19)**: part of the dispatch gap to WebML is
  blocked jointly by `windowTouchesState` MUST (ADR 0067) and the single-output MUST of fused steps
  (`windowIsSingleOutput`, ADR 0068 addendum 1). The third blocker, the atomic last-arriver merge, is now a MUST NOT of
  ADR 0147 decision 11. The individual candidates (attention→SRQ, residual add→SRQ, attention in 1 dispatch, Q/K/V with 3
  outputs) were all closed by refutation, so a design ruling on whether to relax the contracts comes first
  ([research](research/2026-09-19-qat-speed-recon.md) §9.1). Do not file in-layer fusion candidates before the ruling.
- **Remaining items of the video-generation foundation (investigation 2026-09-10; ADRs [0118](decisions/0118-wan21-video-generation.md) /
  [0119](decisions/0119-wan-umt5-gpu-text-encoder.md) / [0120](decisions/0120-wan-dit-w8a8-seat.md))**:
  - MiniMax H3 (the maintainer's final goal, 2026-10-02): H3 stays at a structural study + per-component candidates
    ([research](research/2026-10-02-video-gen-recon.md)). H3 exists and is open-weight (HF MiniMaxAI/MiniMax-H3; by the letter of the terms Japan is a licensed region (not legal advice)),
    but with 33B dense + Qwen3-VL-32B + VAE 2.6B it is about 144 GB per task and assumes offload even natively (a measured example of 112 s on an RTX 5090);
    there is no browser WebGPU case.
  - Wan small items (filed 2026-10-03, diff review of the ADR 0118 wave — the IDs in parentheses are items of `.claude/reviews/2026-10-03_diff-review-wan/` and `2026-10-03_codex-review-wan/` (not tracked by git)):
    ① measuring the GPU time per submit with the submit layout of normal execution (M8) (the "1 submit ≤ 1 s" check and its recorded values come from timing mode (`gpuTiming`, 1 dispatch = 1 pass), and the chunk boundaries after backing can differ from normal execution — ADR 0118 stage 8 result "scope of the submit check"; needs a GPU)
    ③ a fourth case in golden `conv3d_block` that exercises the v4 and m-tile 64 variants (F4) (the current 3 cases are Wout 5 / 5 / 6 and Cout 6 / 3 / 2 and hit only the scalar variant and m 32. v4 and m 64 are exercised only by the GPU parity tests; needs golden regeneration and confirmation on the GPU)
    ⑤ the CPU-only DiT host test reads and copies the whole 2.6 GB reference (TG-11) (`reference.full-{band,accept}-s14040-t0999` are 2,598,638,524 B each; `ArrayBuffer.slice` in `readBuffer` temporarily doubles it; RSS not measured)
    ⑥ the note separator " / " in results.json collides with " / " inside labels (TG-18) (labels in general, such as `formatNormalized`; no consumer splits it today; a fix would go on the separator side)
    ⑦ narrow the entity-candidate check of `prompt_clean` into an exact check that "rejects only when the string actually changes", using the name table (Python's `html.entities.html5`, 2,231 names) (accepting sequences that do not form an entity, such as `R&D` — ADR 0119 ruling 2026-10-03, maintainer: "a workaround if it can be avoided"; once someone asks for it).
- **Remaining small items of the Wan2.2 acceptance (ADR [0121](decisions/0121-wan22-ti2v-5b.md))**:
  **Adjacent small items of stage 9 (filed 2026-10-10, review)**:
  (a) the distribution assembly (`tools/export-recipes/wan/distribution.py`) does not check the input/output contract of the VAE encoder's 3 containers (input names, symbols, the shape chain pre → attn → post) (the decoder's 2 containers have the pair check `assert_vae_chunk_pair` — a series with only one of them rebuilt passes assembly and is first rejected at admission in the user's `fromPretrained`; filed 2026-10-10, review G4-01, a candidate for the fix wave)
  **Adjacent small items of stages 5 / 6 (filed 2026-10-05)**:
  (a) the tile reference reader and helper functions are duplicated across the Wan VAE e2e tests (`readTileFixture` in `e2e_wan_ti2v_vae_tiles_test.ts` and `e2e_wan_ti2v_vae_tile32_test.ts`; `planMeta` in those two and `e2e_wan_vae_tiles_test.ts`; `compare` and `absMax` in `e2e_wan_vae_tiles_test.ts` and `e2e_wan_ti2v_vae_tiles_test.ts`; `poisonFrames` in those two plus `e2e_wan_ti2v_vae_tile32_test.ts` and `e2e_wan_ti2v_vae_a8_test.ts`; `roundUpTwoDigits` — see (d)) → into a shared test helper
  (b) helpers such as the recipe's `_peak_rss_gib` are duplicated in `tools/export-recipes/wan/` (`text_embeds.py`, `vae_tiling.py`, `few_step_ref.py`, `ti2v_few_step_ref.py`, `export_vae.py`, `export_vae_encoder.py`) → into `_shared`
  (c) the existing 5 DiT inputs have no dtype check (only the 2 inputs added for 2.2 check the dtype)
  (d) rounding up to 2 significant digits can round one unit too high because of floating-point error. `e2e_wan_ti2v_pipeline_test.ts` and `e2e_wan_ti2v_i2v_pipeline_test.ts` are fixed by inserting `toPrecision(12)`. Still without it: `roundUpTwoDigits` in `e2e_wan_ti2v_vae_tiles_test.ts`, `e2e_wan_ti2v_vae_tile32_test.ts` and `e2e_wan_ti2v_vae_encoder_test.ts`, and the same expression inline in `bandCandidate` of `e2e_wan_ti2v_dit_test.ts` and `e2e_wan_dit_test.ts`, in `e2e_wan_umt5_test.ts`, `e2e_moss_generate_test.ts` and `e2e_moss_dit_test.ts` (can be fixed together when moving into the shared test helper of (a))
  (e) the messages of the `pipelineConfig` reader have no owner prefix (the message cannot tell whether the 2.1 or the 2.2 class failed)
  (f) the 2.1, 2.2 T2V and 2.2 I2V end-to-end e2e tests (`e2e_wan_pipeline_test.ts`, `e2e_wan_ti2v_pipeline_test.ts`, `e2e_wan_ti2v_i2v_pipeline_test.ts`) hold copies of the same helper functions (`readFixture`, `difference`), and the 2.1 test keeps its own verbatim copy of the contact sheet that the two 2.2 tests import from `helpers/wan-video-artifacts.ts`
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
  **To add later (filed 2026-10-04, maintainer)**: (a) a TI2V-5B preset with a non-quantized DiT — we want the weights
  ready even if it does not run on the development machine. The maintainer's request named f16; under ADR 0128 decision
  4 the original-weights preset of this upstream is F32 (best effort, opt-in), so settle which one is meant first. The
  RAM premise of the original material is gone with the current machine (ADR 0121 decision 2 and ruling 2). Whether
  checks and sha rows fit on the current GPU has not been measured.
  **Related**: the 2 items of ADR [0122](decisions/0122-umt5-upstream-and-compatible-encoders.md) "Open points" (the ruling on publishing third-party compatible components,
  agreement of graph descriptors across versions).
  **umT5 / Wan2.2 small items (filed 2026-10-04)**: (a) the original's index (`pytorch_model.bin.index.json`) and `config.json` are not pinned by sha256 (if the index disagrees with the actual files,
  it fails with a bare `KeyError` — there is no path by which values get swapped; ADR 0122 addendum "stage a results") (b) umT5's `write --check` opens a temporary directory inside the series
  (`.check-*`) — it remains if the process is killed (c) loading `ContainerDitWeights` (the layer-sequential entry point that reads the DiT weights from the container) takes 25 s per forward.
- **Quality improvement of anima base i4 (filed 2026-08-24 — the revival lever of the ruling to skip distribution)**:
  what remains is only **the list of possibilities left unverified** for the i4 preset of the old fused turbo (dedicated
  geometry, g16, calibration amount, one more degradation mechanism — all of them "not tried", not "tried and failed"),
  and the source of truth is [research/2026-08-21-anima-i4-seat-speed.md](research/2026-08-21-anima-i4-seat-speed.md)
  §8. g16 remains as one revival lever in parked "anima g16 evaluation". Making adaLN i8 was rejected by a ruling after
  a visual sweep (perf-ledger Q-9); for locating the places with high quantization sensitivity, the source of truth is
  item ④ of "3 known issues + anima base i4 sensitivity" in the
  [archived completed section](research/2026-09-22-backlog-archive-0.5.0-to-0.12.0.md), to which the same ruling
  attaches "the anima DiT i4 line is on hold for a while". The calibrated `*-i4-dyn` series are no longer under
  `outputs/` on the development machine, so a revival starts from a new calibrated export (source of truth for the
  measurements = [research/2026-08-24-gptq-expansion-quality.md](research/2026-08-24-gptq-expansion-quality.md) §5).
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
- **Practice of benchmarking against ORT Web** (maintainer ruling 2026-08-16): measure models supported by both
  regularly, and investigate whether we are too slow and where the bottleneck is. **It has run for gemma4 only**, and
  the record is [research 2026-09-12](research/2026-09-12-webml-browser-speed.md). Its harness was removed in 0.15 and
  no ORT comparison harness exists today (the Playground's Gemma 4 bench compares Karume's own arms), so resuming needs
  one as a browser application (ADR 0127). Remaining targets = EmbeddingGemma (requires completion on the models side)
  and **KokoroTTS** (corrected 2026-08-16 — the original Irodori was a typo. Kokoro runs on the Transformers stack, so
  it is easy to compare; requires Kokoro-82M support on the karume side = the model expansion candidate above, waiting
  for LSTM multi-output). In future, widen it into a comparison matrix that includes browser ONNX + native PyTorch
  (ruled unnecessary for now). The norms for measurement conditions (graph capture ON / freeDimensionOverrides / IO
  binding / checking EP partitioning, recording native EP or JSEP) have their source of truth in
  [runtime-landscape §4](research/2026-08-16-runtime-landscape.md).
- **Rolling generation events out to other families (waiting for demand)**: stage events for sbv2 / birefnet / depth /
  siglip2 / vowel (no step loop, so only stage transitions can be offered), and an AbortSignal cancellation slot in the
  **generation loop** of anima / irodori (today a throw from onEvent is the step-granular means of cancellation — the
  slot is kept in reserve). MOSS-Image's generation entry has neither events nor cancellation. **AbortSignal on the
  construction path is implemented for anima / irodori / Wan / MOSS** (`anima.PipelineOptions.signal` — checked at stage
  boundaries, passed through to the fetch layer, `signal.reason` propagated as is), so that is the precedent for the
  style (the construction `signal` of sbv2 / birefnet / depth / siglip2 / vowel / gemma / umT5 is only passed through to
  the fetch layer, with no checks at stage boundaries). **The generation-loop slot is implemented on the LLM surface**
  (ADR [0083](decisions/0083-generation-api-surface.md) decision 5) **and on Wan** (`wan.GenerateRequest.signal`) —
  rolling it out to other families still waits for demand.
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
  viability is uncertain** (the PoC's miss cost was dominated by one re-execution fence of about 11 ms, which came from Deno's 10 ms
  sleep; Chrome's floor is expected near 1.5–3 ms but is unmeasured. On all 30 layers of 26B A4B the CPU measurement of
  [research 2026-11-08](research/2026-11-08-gemma4-26b-moe-cache.md) found the cost is set by re-runs: 2.41 forwards per decode token
  at 64 experts per layer with whole-token re-runs). The start condition is
  subordinate to parked "value-dependent execution selection in the IR". The first unit stops at transcribing the spike's measurements into
  [research](research/) as a point-in-time snapshot.
- The MoE seam (fixed-k routing can be expressed with static shapes — do not bake the absence of experts into the dense API).

- **Remaining items of GEMV row blocks (filed 2026-09-07, consequence of K-21)**: ① the parallelism target 16384 is the saturation point of the RTX 3080 Ti (limitations) —
  the M2 re-sweep is done (the default 16384 is best); what remains is a sweep on integrated GPUs. The override is the static Session option
  `linearGemvRowsThreadTarget` ② the shader analysis cost on Tint (Chrome, and Dawn under Node) is unmeasured; the only
  measurement is on naga ([research 2026-09-07](research/2026-09-07-gemv-rows-k21.md)) ③ the Metal u32 check is also
  unmeasured for the 13 row-block shapes (known-issues).

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
  params −inf stays as is = ruled as a set in the same frame as W-2/W-3) / **tolerance option B** (adds a
  reduction-scale term to `allclose` and the `Tolerance` type in `packages/runtime/src/reference/allclose.ts`, which are
  internal but shared by every test tolerance, the e2e ones included — option A's per-op table is implemented.
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
  MOSS-Image accepts such shapes with an a8 preset, but its preset runs them on the online
  kernel, which pads the key axis (ADR 0132 addendum of 2026-10-27); it meets the condition only on a device with
  neither online form, where the practical preset runs the three dispatches with S in f32 and, where S % 4 != 0,
  ③PV in f32 too (ADR [0138](decisions/0138-moss-image-dit-vae.md) decision 10).
  Kill criterion: measured VRAM and speed on those shapes show no difference against today's path that falls back to f32.


## release — the release preparation wave (some way off)

- Real-asset CI gate (GitHub CI does not exercise local assets; the all-skip guard covers the empty case): decide on promoting goldens to fixtures / fetching assets in the release gate
- Writing up the repository root README; verifying sideEffects of the JSR npm compatibility layer. **Out of scope for 0.8.0
  (ruling 2026-09-03 — replaced with a form whose Status line alone says it is a stub). Revival condition = 1.0 or
  a public announcement**, and starting requires re-measuring the bundle size (the gzip measurement of 2026-08-16
  is outdated by the additions of the gemma4 generation API, the GEMV family and the tokenizer)
- Licence interview (human confirmation of export-recipes' per-family provenance per upstream revision; it runs at every publication — release-runbook). What the interview of 2026-09-25 left open:
  rows left as Unrecorded = the revision of modernbert-ja-310m (no local record), the revision of Semantic-DACVAE-Japanese (parent) (not fetched).
  Rows left as Unverified = whether the Civitai permission field of anima-wai-v1.0 can widen the upstream licence, the Weights license of depth_anything (before revision matching).
  **FN (`rufflet17/voice_models`) has neither a licence statement nor a README on HF, so it cannot be published** (until this is clarified).
  The Code license of the code-dependency block is still Unverified in the deberta, minicpm5 and embeddinggemma ledgers
  (deberta's Attribution too).
  The machine-readable place for the upstream revision is the container's `provenance.upstreamRevision`
  ([container-v1](container-v1.md) §2.3). The recipes of anima, irodori (main and dacvae) and vowel_detector, and
  gemma4's `export.py`, still declare a `PROVENANCE` without it, so they **start filling it at their next re-export**
  (through `tools/export-recipes/_shared/upstream.py`; on the same occasion as the `sym_max` field of W-G4-4, which
  `Provenance` does not have yet)
- **Gate for the first PyPI release of `karume` (tools/exporter): no upstream copy in the core wheel** (ruling 57 a,
  2026-11-01). `custom_ops.py` holds a verbatim copy of transformers 5.14.1's `l2norm` / `torch_chunk_gated_delta_rule`
  as the eager body of `karume::gated_delta_scan` (ADR 0135 decision 5), with a third-party notice comment. That
  conflicts with ADR 0065 decision 7 (the core wheel holds no copy of an upstream model implementation; the boundary
  test catches imports, not copies), and the wheel's `license-files` carry only MIT. Fix before the first upload: core
  keeps the op's schema and fake only, and the qwen3_5 recipe (which already depends on transformers) registers the
  eager kernel by importing upstream's function — no copy anywhere, and the "verbatim copy" MUST disappears with it.
  ADR 0065 itself stays: core does not take a transformers dependency
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
  (maintainer ruling of the same day))
- hub Range parallel + prefetch — revival = fragmented repositories coming back (perf L-3)
- Making GPU timestamps an estimation source, full f16 (option γ), SBV2's engine-dependent NFC
  (`packages/models/src/sbv2/text/tokenizer.ts` uses plain `normalize("NFC")`; first check that DeBERTa's normalizer
  needs NFC, then take the segment-table method of `packages/models/src/text/qwen2-tokenizer.ts` —
  [research 2026-08-03](research/2026-08-03-demo-w8-perf-recon.md)), regenerating the f32 anima series
- The by-design constraints (rank≤4 / OOB NaN / non-finite input / zero-element dimensions / cancellation granularity and others) have their source of truth in
  limitations.md — only items with real demand are promoted here
- **Prerequisite for the first publication of vowel-detector (filed 2026-09-24)** (first publication on hold — maintainer's ruling 2026-10-15): the recipe reads the upstream `feature_config.json`
  directly under `inputs/vowel-detector/`. This development machine holds the whole upstream repository, so before assembly
  run `cp inputs/vowel-detector/assets/feature_config.json inputs/vowel-detector/` once.

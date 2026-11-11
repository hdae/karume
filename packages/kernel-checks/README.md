# @karume/kernel-checks

Private workspace package (not published): the kernel identity checks that must run on every
backend a kernel rewrite relies on, in one implementation shared by

- the runtime's node GPU tests (`packages/runtime/tests/gpu_linear_gemv_subgroup_test.ts`,
  `gpu_rms_norm_subgroup_test.ts`, `gpu_state_attention_fold_test.ts`,
  `gpu_linear_gemv_integer_test.ts`, `gpu_linear_gemv_planes_test.ts`, `gpu_state_kv16_test.ts`, `gpu_score_f16_store_test.ts`,
  `gpu_state_bidirectional_test.ts`, `gpu_expert_linear_test.ts`, `gpu_fusion_twin_test.ts` and the
  twin-graph tests of the fusion rules — Dawn on the development machine), and
- the playground's `/kernel-check` page (`apps/playground` — the browser's own backend: D3D12 in
  Chrome on Windows, Metal on macOS).

So the page runs exactly the cases and criteria the tests run; there is no copy to drift.

| Check | Function | Needs |
| --- | --- | --- |
| Parallel GEMV `linearGemvReduce: "parallel-subgroup32"` vs parallel, u32 identity (ADR 0101) | `checkGemvSubgroup` | `subgroups` + `subgroup-size-control` + WGSL `subgroup_id` (`subgroup32Available`) |
| RMS `rmsNormReduce: "subgroup32"`: CPU-reference tolerance, row identity across M, fused vs unfused identity, grid-stride (ADR 0100) | `checkRmsSubgroup` | the same |
| States attention rewrites (perf-ledger K-55 / K-91): folded ①' vs ①', unrolled ③' and fused stats + PV vs their v1 WGSL, u32 identity, plus order-only mutants that must mismatch | `checkStateQkFold`, `checkStatePvUnroll`, `checkStateStatsPvUnroll`, `checkStateAttentionMutants`, and `checkStateAttentionMutantMatrix` (each mutant on every case of its shape — the page runs it, the node test does not; all five: `checkStateAttentionRewrites`) | nothing optional |
| Integer GEMV `integerGemv`: every output word equals the CPU reference, rows past M unwritten (mirror class E — ADR 0140) | `checkIntegerGemv` (and `benchIntegerGemv`, a single-kernel timing, not a check) | nothing optional |
| Byte-plane GEMV `linearGemvPlanes`: the f64 band, the A/B band against the sequential GEMV, two dispatches identical, nothing written past the outputs, NaN propagation (class R — ADR 0141; the fault injection runs in the node test only, through `PlanesGemvCheckOptions.mutate`) | `checkPlanesGemv` (and `benchPlanesGemv`, a single-kernel timing, not a check) | nothing optional |
| f16 full state slots `fullStateStorage`: every reader vs its f32 form over a rounded slot, the append vs host rounding on every rounding boundary (class C — ADR 0143) | `checkStateKv16` (and `checkStateKv16Mutants`, node test only) | nothing optional |
| f16 attention scores `attentionScoreStorage`: the store vs host rounding (ADR 0031) | `checkScoreF16Store` | nothing optional |
| Bidirectional block of the sliding ①ₜ (`GenerationRun.bidirectionalRows`): every S word vs the same kernel keeping every score plus a host mirror of the predicate, u32 identity (ADR 0144) | `checkStateBidirectional` (and `stateBidirectionalMutants`, node test only) | nothing optional |
| `expert_linear` with f16 steps at the 26B shapes: every pick's output words vs `linear`'s sequential i4 GEMV on that expert with the steps widened to f32, u32 identity (ADR 0145) | `checkExpertLinear` | nothing optional |
| A fusion against the unfused plan of the same graph (a rule of `FUSION_RULES` or an annotation fold): two open Sessions run the same feeds, every compared output u32-identical with NaN by class or by payload, the fusion counter at the expected hits on the fused side and at 0 on the plain side, optionally each side's dispatches (class E — ADR 0147 decision 2) | `checkFusionTwin` | nothing optional |

The page does not run `checkFusionTwin` yet: the caller builds both Sessions, and the graphs and the
ways of making the plain side (an interposed zero-dispatch reshape, an opt-in on one side only, a
forced split, the test-only `DISABLED_FUSION_RULES` key) live in the runtime's twin-graph tests
(`gpu_*_fusion_test.ts`, `gpu_fusion_twin_test.ts`). It returns one case per call, so it exports no
case count.

A check does not assert. It returns structured results — per case, per criterion, the number of
failing words and the first (identity) or worst (tolerance band) failing word as u32 — and the
callers decide: the tests assert every count is 0, the page shows PASS / FAIL. A failure of the
harness itself throws: every case runs its GPU work under the validation, out-of-memory and
internal error scopes and also fails on an uncaptured WebGPU error or a lost device
(`scopedGpuWork`), since an invalid shader or a failed backend compile does not throw and would
otherwise read back the buffers' initial contents for both runs of a case; a mutation anchor that
no longer matches throws as well. Each check exports the number of cases each of its parts returns
(`GEMV_SUBGROUP_CASES`, `RMS_SUBGROUP_CASES`, `STATE_ATTENTION_CASES`, `GEMV_INTEGER_CASES`,
`GEMV_PLANES_CASES`, `STATE_KV16_CASES`, `SCORE_F16_CASES`, `STATE_BIDIRECTIONAL_CASES`, `EXPERT_LINEAR_CASES`) and the
mutant count
(`STATE_ATTENTION_MUTANTS`); the tests and the page hold a run to them, so a part that ran no
cases cannot pass.

The checks drive runtime internals (kernel WGSL generators, the dispatch harness and fixtures under
`packages/runtime/tests/helpers` and `tests/fixtures/wgsl-baseline`) through relative imports, as
`@karume/e2e` does; that is why this is a private package and not a subpath of `@karume/runtime`
(which would publish test machinery). The modules are browser-safe: no `node:` imports, no
assertion library, and no WebGPU global read at module scope (usage flags come from the runtime's
`webgpu-constants.ts`), so a browser without WebGPU can load the page and show that it cannot
run. The baseline WGSL is imported as text (`?raw`, which Vite and Vitest both load).

`package.json` declares no dependency on `@karume/runtime` although the sources import it by
relative path: `@karume/runtime` devDepends on this package (its GPU tests import it), and
declaring the edge back would make a workspace cycle. So workspace filters such as
`pnpm --filter @karume/kernel-checks...` do not see the edge; the package has no scripts, its CPU
tests (`tests/`: the error-scope handling of `scopedGpuWork` on a fake device and the exported
counts) run in the core lane, and its GPU code runs through the runtime's GPU tests and the
playground.

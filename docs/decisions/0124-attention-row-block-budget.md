# 0124: A capacity budget for attention row-block temporaries (`attentionRowBlockMaxBytes`)

- Status: accepted (2026-10-11, the maintainer: a user budget may split row blocks finer without
  changing result bits; the open design questions answered as recommended); implemented 2026-10-11.
  The gpu-lab part of decision 10 lapsed with `tools/gpu-lab` (`bb35fe2a`, 2026-10-16). Design notes:
  `.claude/reviews/2026-10-04_de127de1/DESIGN-4-6-8-v2-2026-10-11.md` (d6, untracked)
- Date: 2026-10-11
- Related: ADR [0060](0060-row-block-attention.md) (row-block decomposed attention — the basis of bit
  identity) / [0023](0023-fused-attention.md) (fused attention) /
  [0067](0067-autoregressive-attention-vocabulary.md) decision 7 (states-form row blocks) /
  [0093](0093-transient-liveness-packing.md) (intermediate placement and estimate — addendum
  2026-10-10) / [0070](0070-shard-loading-admission.md) decision 5 (the estimate is not compared with
  free VRAM) / [0022](0022-gemm-register-blocking.md) (no runtime autotuning) /
  [0058](0058-numerics-opt-in-contract.md) (numeric opt-in options) / [0121](0121-wan22-ti2v-5b.md)
  (Wan2.2 — the development machine's 1280×704×121 measurements)

## Context

Three attention forms materialise the score matrix S: fused attention (ADR 0023), the states form
(ADR 0067) and the decomposed `rowBlockAttention` (ADR 0060). Each computes S in blocks of query rows;
the count is the smallest that makes one S block and one row-statistics block each fit the device's
`maxStorageBufferBindingSize`, so only the binding limit decides it. With a limit of about 2 GiB (RTX
3080 Ti, Arc B570), Wan2.2's practical preset at 1280×704×121 (S = 27,280) has one S block of about
1.96 GiB (17 blocks); on the development machine (RTX 3080 Ti, 12,288 MiB) the DiT stage peaked at
11,361 MiB (ADR 0121). Splitting finer shrinks the temporaries alive during attention and lowers the
VRAM peak. The aim is headroom on 10–12 GB class devices, not new shapes: 832×480 is outside
acceptance, and 121 frames already completed on the development machine.

Deciding facts:

- **WebGPU does not expose free VRAM.** The estimate is never compared with free memory (ADR 0070
  decision 5), and runtime autotuning is ruled out (ADR 0022). Neither the runtime nor a pipeline can
  decide when to trade speed for capacity; only the user (the application) can.
- **Row blocks are bit-identical.** A row window only reassigns rows (ADR 0060 decision 3), so the
  budget is a resource policy like `submitPolicy`, not a numeric option.
- **Speed cost at Wan scale was unmeasured** at the time. Small shapes slowed down (Anima's
  `[1,16,4096,128]`, a8 / s16: 6.46 / 6.98 ms in 1 block, 8.42 / 10.90 ms in 8 — [research
  2026-09-10](../research/2026-09-10-codex-mtp-optimization.md)); a decomposed case sped up (ADR 0060,
  S = 750: 46.8 ms in 2 blocks, 48.9 ms in 1).

## Decision

### 1. Meaning — a limit per row-block temporary, not a total

- `SessionOptions.attentionRowBlockMaxBytes` is added (optional, positive safe integer, bytes).
- The row-block limit is **min(granted `maxStorageBufferBindingSize`, budget)**.
- The limit applies **to each temporary**, not to their sum: one S block and one row-statistics block
  are each within it, and what is alive together may exceed it. The i8a8 quantization temporaries
  (outside the row-block loop) and other intermediates are not bound by it.
- One pure function derives it (`rowBlockLimit` in
  `packages/runtime/src/runtime/fusion-rules/row-block-attention.ts`); fused attention, the states
  form, the read-only states form, the decomposed `rowBlockAttention` and the estimate all go through
  it.
- The placement limit check for intermediate buffers (ADR 0093 decision 5) keeps the granted value;
  FFN intermediates and embedding bindings are not bound.

### 2. The default is unchanged

Without the field, block counts, pipeline keys, WGSL, dispatch sequences and backing are byte for
byte as before; existing sha rows, WGSL snapshots and the census pass unchanged.

### 3. A resource policy, not a numeric option

It does not enter presets, the manifest's `session` or the hub, because it does not change output
bits. Its name is in the census exclusion type `NonNumericField`
(`packages/runtime/tests/helpers/pipeline-census.ts`), declaring it a field that does not change
numbers.

### 4. Basis of bit identity

Structurally:

1. a row window only reassigns rows (ADR 0060 decision 3); the K reduction order and the rounding
   sequence of each output element do not move;
2. row statistics and softmax are closed within a row; cutting between rows does not change a row's
   operation sequence;
3. i8a8 quantization is outside the row-block loop; its values do not depend on the block count.

Going from n = 1 to n ≥ 2 switches to a row-window key and a different WGSL. Bit identity at that
boundary is covered by the enumerated variants that tests pin:

| Form                                 | Variants                                                         | Test (forced split and budget split both equal 1-block execution in Uint32) |
| ------------------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| fused attention                      | f32, f32 + mask, s16, a8, a8 + s16 (GQA shapes included)         | `packages/runtime/tests/gpu_attention_row_block_test.ts`                      |
| fused attention (f16 compute)        | f16, f16 + mask (GQA included; needs `shader-f16`, separate case) | same file (added by this ADR)                                                 |
| decomposed attention (9-node window) | f32                                                              | `packages/runtime/tests/gpu_row_block_attention_test.ts`                      |
| states form                          | f32 only (numeric variants fail loudly — ADR 0058 decision 3)    | `packages/runtime/tests/gpu_state_execution_test.ts`                          |

A budget-split run checks at once that the count is 2 or more and that the intermediate peak
(`peakTransientBytes`) is strictly below the unbudgeted one, so a budget that has no effect cannot
pass by running the same path twice.

### 5. When it fails

- A shape whose one-query-row temporary exceeds the budget fails loudly, naming the budget, **at the
  first run that derives the plan for that shape**: at Session construction shapes are still symbolic
  and the count is unknown. In Wan this is after the weights are loaded. A read-only states form (one
  row = decode) with a long capacity and a small budget fails at decode.
- The construction-time range check (`sessionOptionsViolation`) rejects only values known to be
  invalid then (0, negative, non-integer, NaN, non-number).
- Passing the same value to the estimate (decision 7) reveals a violation before weights load.

### 6. Precedence over a forced count

When the test-only `ROW_BLOCK_SPLIT` forces a count, it passes only if that count fits
min(binding limit, budget); otherwise it fails loudly naming the budget (forcing never loosens the
budget). Existing forced-split tests without a budget keep their meaning.

### 7. The estimate

- `EstimateOptions.attentionRowBlockMaxBytes` is added, range-checked with the same message as Session
  construction, even for graphs without row blocks.
- Existing accounting (states-form S and row statistics, the decomposed row-block window temporaries)
  derives its counts as execution does (decision 1). A budget below one row fails with the message
  execution uses.
- **The in-node temporaries of fused attention ops (S, row statistics, i8a8 quantization
  intermediates) are counted too**, with execution's pure functions (`planAttentionBlocks`,
  `attentionScoreElementBytes`, `attentionI8a8TempBytes` in `recipe-builders/attention.ts`;
  `attentionRoute` in `attention-route.ts`, both under `packages/runtime/src/runtime/`). This changes
  the default for every series with fused attention ops, budget or not, so the before / after table
  for existing callers came first
  ([research 2026-10-11](../research/2026-10-11-fused-attention-estimate-before-after.md)), and the
  accounting and `unaccounted` message changes were separate commits.
- A fused attention op's count depends on the binding limit. An estimate without
  `maxStorageBufferBindingSize` counts only shapes whose count is the same on every device: shapes
  that fit one block at the WebGPU core default (128 MiB; a granted value is never lower) and shapes
  whose budget is at or below the core default (the limit is then the budget everywhere). Other
  shapes fail loudly asking for the limit (as for the states form: dividing by the core default could
  claim a count no device would produce).
- Combinations execution rejects (mask × a8, GQA × a8, the i8a8 i32 reduction check, states form ×
  non-f32) get no estimate either, decided by the same pure functions (`fusedAttentionViolation` /
  `stateAttentionVariantViolation`) with the same messages.
- `EstimateOptions` also accepts `attentionCompute` / `attentionScoreStorage`; omitted, they count as
  the runtime default f32 (the precedent of `stateAttentionReduce`).

### 8. Bit-identity check in two layers

- **Layer 1**: the DiT e2e (now `packages/e2e/tests/e2e_wan_ti2v_dit_test.ts` — ADR 0129) runs a u32
  A/B with `SessionOptions` passed directly: the same input without a budget, with 512 MiB and with
  256 MiB, comparing output bits, at 17 frames (default lane) and 121 frames (opt-in
  `KARUME_WAN_TI2V_121F=1`), for the practical and the reference preset. The expected block counts are
  derived independently of the runtime (self / cross separately, with the preset's storage width); the
  number of ① QK dispatches of the row windows must equal them and differ from the unbudgeted run. It
  does not depend on a models knob.
- **Layer 2**: comparison with the pipeline's sha rows, only once a models knob exists; without one
  no wiring can change the output.

### 9. Boundary with online softmax (an opt-in fast path without S)

Outside this ADR, except: this knob is a resource policy and stays out of presets; whether a path
without S ignores or rejects it is that path's ADR's decision (per-phase GPU times were taken in step
4b).

### 10. Public surface

- The runtime's `SessionOptions` / `EstimateOptions` each gain one optional field; `mod.ts` exports no
  new name.
- **No knob in models** (`WanTi2vPipeline`) **and no budget in Wan's default**: ruled on 2026-10-11
  after step 4b; added when a problem needs it (ADR [0005](0005-verification.md) addendum 2026-10-11:
  no options without a use). Reason: the 121-frame practical preset fits the 12 GiB class development
  machine without a budget (DiT-only peak 11,308 MiB), and a wall-clock cost could not be ruled out.
  That peak was measured on Deno 2.9.6 only; the Node + Dawn peak is not measured. (The gpu-lab Wan
  tab named here was removed with `tools/gpu-lab` in `bb35fe2a`, 2026-10-16.)
- "61 frames and more are unsupported on 10 GB class devices" (`docs/limitations.md`, Wan2.2 sizes)
  stays: no such device is at hand to check. Its basis is the Deno total-allocation ceiling of about
  9.4 GiB measured on the former Arc B570; Node + Dawn's ceiling is not measured.

## Alternatives not taken

- **A fixed runtime default budget (for example 1 GiB for all series)**: free VRAM is invisible, so
  devices with headroom pay the split cost too; the default performance of every series would change
  silently in either direction (against the spirit of ADR 0058 decision 3), and a fixed value could
  rest only on one machine. (Step 4b later measured the Wan-scale cost; the other reasons stand.)
- **Derive the limit from a total VRAM budget via the estimate**: needed the estimate to count fused
  attention first, and backing differed by 1.17 GiB between Deno (5.97 GiB) and Chrome (4.80 GiB) for
  the same run for an unknown reason (ADR 0121), so a derived value can miss and fail out of memory.
- **Reuse `planBackingBudgetBytes`**: mixes in a different meaning (how many backings are kept).
- **Lower the device binding limit itself** (`requiredLimits` in `acquireGpu`): would also bind FFN
  intermediates and embeddings.
- **Promote `ROW_BLOCK_SPLIT` (a count) to a public knob**: the same count gives different block sizes
  per shape; a byte limit is used instead.

## Consequences

- One optional field each in the runtime's `SessionOptions` / `EstimateOptions` (additive; existing
  callers unaffected). Hub, manifest and presets unchanged.
- A budget adds dispatches (3 per block for fused attention): for the 121-frame practical preset at
  512 MiB, about 4,590 more per forward (self (67 − 17) × 3 × 30 layers + cross (2 − 1) × 3 × 30
  layers).
- A budget also splits cross-attention (512 text columns; 2 blocks at 512 MiB for the 121-frame
  practical preset). Where users set a budget, other series (Anima, Irodori, …) newly cross the
  n = 1 to n ≥ 2 boundary; the forced-split parity of decision 4 covers that across shapes.
- Failures come late (decision 5): in Wan after the weights are loaded.
- The estimate extension (decision 7) may raise `workspaceBytes` / `peakAccountedBytes`. Existing
  callers were unchanged (Anima's text stage has no fused attention op and its VAE temporary fits the
  existing region; Gemma uses only the states form), so Anima's pre-emptive eviction of the resident
  DiT gets the same inputs. Wan2.2's DiT (60 fused attention ops; practical preset, 121 frames, no
  budget) rises from 3.54 to 5.61 GiB `workspaceBytes`; Wan's pipeline does not call the estimate.
  Callers that omit the binding limit are not affected by decision 7's rejection: Anima passes it and
  Gemma has no fused attention op.

## Acceptance

Completed 2026-10-11: host tests (`rowBlockLimit`, Wan2.2 counts, loud failures, forced counts, range
checks, hand-computed estimate constants), the real-GPU table of decision 4, layer 1 of decision 8,
and the estimate extension (`workspaceBytes` equals execution's `planBacking.residentBytes` for a
single fused attention node — `packages/runtime/tests/gpu_estimate_test.ts`). WGSL snapshots and the
census were unchanged.

## Results

Measured on one machine only (RTX 3080 Ti, Deno 2.9.6, 2026-10-11); tables and method in
[research 2026-10-11](../research/2026-10-11-row-block-budget-stage-4b.md). Chrome and 10 GB class
devices were not measured.

- **Layer 1**: for 17 and 121 frames and both presets, outputs at 512 MiB and 256 MiB equal the
  unbudgeted output in Uint32, with ① QK counts as expected. The 17-frame A/B runs in the default lane
  (now the Vitest project `models:wan-ti2v` — ADR 0129). Layer 2 was not run (no models knob).
- **Step 4b** (121-frame practical preset, one DiT forward): backing 5.97 GiB without a budget, −0.98
  at 1 GiB, −1.46 at 512 MiB, −1.65 GiB at 256 MiB; the peak fell from 11,308 to 9,622 MiB. GPU time
  in measurement mode stayed within noise at 1 GiB and 512 MiB and rose by 24.9% at 256 MiB (almost
  all `attention_pv`); wall-clock runs were thermally throttled and not used. Decision 10 records what
  followed.

## Addendum (2026-11-05) — `gemma.PipelineOptions` passes the limit through (update to decision 10)

Decision 10 kept the knob out of models until a problem needed it. Gemma 4 has that problem: the
prefill score matrix of states-form attention is heads × chunk rows × capacity × 4 bytes, so it grows
with the capacity a caller chooses (1.5 GiB at 8 heads, 768 chunk rows and capacity 65,536, computed
from that formula), and the gemma surface had no way to cap it.

- `attentionRowBlockMaxBytes` joins `gemma.PipelineOptions`, unset by default. Unset leaves the
  Session options byte for byte as they were (the field is absent), so nothing changes for existing
  callers.
- The value goes to the target Session and the drafter Session, and
  `Pipeline.estimateSessionMemory` passes the same value to the estimate of both graphs, so the
  estimate counts the split the Sessions run. This is the same rule as `planBackingBudgetBytes` there
  (ADR [0095](0095-plan-backing-budget.md)). A limit below one row therefore fails in the estimate too,
  before the first run does.
- It stays a resource policy: it is not a manifest `session` field or part of a quant preset, and token
  sequences do not change (decision 4 — bit identity comes from row blocks only reassigning rows; no
  Gemma-specific A/B was run for this addendum). It trades prefill speed for VRAM; that cost was not
  measured on Gemma, and step 4b measured it only on Wan2.2's DiT.
- Other families are unchanged: `wan.Ti2vPipeline` and the rest still have no knob, and no default
  budget is added anywhere.

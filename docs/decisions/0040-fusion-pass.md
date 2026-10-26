# 0040: Runtime fusion is a separate pure-function pass (exact matching plus permanent counters)

- Status: accepted (2026-08-08 — the four PNG sha256 checks did not move, and SBV2 WAV sha256 is
  identical with fusion on and off)
- Date: 2026-08-08
- Related: ADR [0007](0007-op-vocabulary.md) (op vocabulary allowlist — this ADR adds no public op) /
  [0012](0012-attrs-and-fused-ops.md) (contract for fused ops placed in the IR — this ADR is the
  opposite case, private kernels that never appear in the IR) / [0011](0011-layout-strategy.md)
  (layout strategy — aliasing of identity `expand` is recorded there) /
  [0023](0023-fused-attention.md) (an existing fused kernel published as an IR op) /
  [0021](0021-gpu-timing-diagnostics.md) (style of permanent diagnostics) /
  [0004](0004-execution-model.md) (arena reference counting) /
  [0042](0042-prepared-execution-plan.md) (prepared execution plan — where step bookkeeping lives
  today)
- Triage of the reference branch: [research/2026-08-06-kernel-triage/](../research/2026-08-06-kernel-triage/)
  (OP-007 / OP-017 / OP-018 / OP-019); adoption and measurements in
  [research/2026-08-08-branch-adoption-perf.md](../research/2026-08-08-branch-adoption-perf.md)

## Context

The exporter's node sequence contains decompositions that always appear in the same order: the seven
nodes of half-split RoPE (eight dispatches, because `cat` emits one copy per input), SiLU as
`sigmoid → mul`, the six `reshape/expand` nodes of the VAE's nearest-exact ×2 upsample, and identity
`expand` whose shapes match exactly. On the Anima 1024 / 8-step default path these alone account
for thousands of dispatches and GiB-scale intermediate writes and reads.

Adding an op to the IR is expensive under ADR 0007's allowlist and ADR 0012's contract discipline
(TypeScript contract table, Python contract table, goldens and exporter move together). RoPE and SiLU
have no single corresponding torch op and appear only in decomposed form, so they do not justify new
public vocabulary. A peephole that collapses them at run time is needed.

The question is where it lives. The direct form puts `if (op === "sigmoid" && next.op === "mul")`
into the executor's node loop, which causes two structural failures:

1. **Matching is reachable only from GPU tests.** Covering the counterexamples (use count 2, graph
   output, near shapes, other dtypes, other orders) needs a real GPU.
2. **Release bookkeeping is duplicated per fusion.** A fused step consumes inputs fewer times than the
   original sequence (internal values are not materialised), so retain / release would be hand-written
   per fusion. An off-by-one raises no exception (released too early: the pool reuses the buffer and
   values change; too late: peak memory does not drop). Arena reference counting (ADR 0004) would turn
   into a source of silently wrong values.

## Decision

### 1. Fusion is a pure-function pass in `src/runtime/fusion.ts` (no matching inside the executor)

The conversion from the planned node sequence (`NodePlan[]` from `plan.ts`) to the execution step
sequence (`ExecStep[]` — plain nodes or fused steps) is closed in one pure function that never touches
the GPU, `planFusions`. The execution side receives the step sequence and holds no matching logic.

- **Rules form a declaration table**, `FUSION_RULES` (seven rules as of 2026-10-18:
  linearStaticQuantize / rmsNormAdd / silu / upsample2x / rope / adaln / rowBlockAttention). Each rule
  separates `match` (capture) from `build` (declare binds, kernel key and params), and `defineRule`
  enforces the separation through types.
- **MUST: the basis of release bookkeeping (the multiset of external inputs, `ins`) and the scan width
  are derived from the captured chain**; rules do not declare them. `externalIns(chain)` builds "the
  original `node.ins` minus internal values" mechanically, so the same fact cannot be duplicated per
  rule.
- **MUST: plain nodes and fused steps share one bookkeeping path.** "Allocate → retain → body →
  release inputs (multiset) → release per definition" is written once for both kinds; only the body
  differs. Today this path is `#buildStep` in `src/runtime/recipe-builder.ts` with its replay in
  `src/runtime/transient-plan.ts` (ADR 0042). The bind surface of a fused kernel is "params, inputs…,
  output", with params at binding 0 (a uniform by default).
- Rules are applied in the order of `FUSION_RULES`. Their head ops are pairwise disjoint, so the order
  does not change the result; **a test checks the disjointness mechanically from each rule's `heads`**
  (`packages/runtime/tests/runtime_fusion_test.ts`), because order starts to matter the moment heads
  overlap. (One declared exception since 2026-10-25, `rms_norm` of rmsNormAdd and rmsNormRope, which
  never accept the same index — the addendum of 2026-10-25, stage 3.)
- Identity `expand` (input and output shapes identical after binding, rank included) becomes a buffer
  alias with zero dispatches, like `reshape` (ADR 0011). A non-identity expand keeps the strided copy.

### 2. Matchers accept exact matches only (never widen the acceptance set to "similar formulas")

A shape that is not captured falls back to the plain node sequence. That fallback is the existing,
always-correct path, and it is the whole basis of the pass's correctness. Widening the acceptance set
to "the formula looks similar" leaves "not captured means correct", and the fallback stops being a
guarantee. Therefore:

- **Op sequence, wiring, attrs, resolved shapes and dtypes are all compared, and any mismatch returns
  `undefined`.** RoPE accepts only `[1,H,S,D]` (D a positive even number; measured: Anima 128, Gemma
  256), table `[1,1,S,D]`, and the half split `0..D/2` / `D/2..D` on dim 3 (head width derived from the
  split since 2026-08-11; BSHD added 2026-09-12, see addenda). SiLU accepts only f32 with the same
  shape in every slot; upsample2x accepts only f32 rank-4 NCHW with each spatial axis exactly doubled.
  Interleaved (pair-form) RoPE is accepted only in the exact node sequence of the addendum of
  2026-10-24; there is no generalisation to broadcast SiLU or general resize.
- **Every internal value of a chain must have exactly one consumer and must not be a graph output**
  (`internalsArePrivate`, common to all rules). A fused step creates no internal buffer, so any
  external consumer or readback would lose the value. (The rmsNormRope rule, whose chain reads two
  internal values twice, uses the sibling `internalsStayInChain` — every use inside the chain; the
  addendum of 2026-10-25, stage 3.)
- **MUST: use counts and release bookkeeping follow the actual node order.** RoPE accepts two orders
  (slice-first and direct-mul-first); a sequence reordered by role would shift the set of internal
  values for direct-first only.
- Fusion collapses the op sequence but **does not change values** on the default path. Preserving the
  rounding points is the kernel's job; the means RoPE and SiLU use (a rounding barrier through a
  workgroup-memory round trip) is **measured behaviour, not a WGSL guarantee**, and each kernel's
  docstring says so. upsample2x copies u32 bits, so rounding does not arise. The opt-in
  `rmsNormReduce: "subgroup32"` variant of the rmsNormAdd rule sums in a different order from the
  reference path; it is governed by the opt-in contract of ADR [0058](0058-numerics-opt-in-contract.md)
  and ADR [0100](0100-rms-subgroup-reduction.md), not by this decision.

### 3. Application counts are exposed as `lastRunFusions` in the permanent diagnostics

Fusion **stops silently when the exporter's node emission order changes by one node**. Values stay
correct (the fallback is correct), only performance regresses, and nothing throws or warns. This is the
only observation point, so per-rule application counts are permanent:
`SessionDiagnostics.lastRunFusions`, with the same lifetime as `lastRun` / `lastRunTiming` (the most
recent run, replaced as a whole on every run).

The counters are the fusion rules plus `identityExpand` (and `packedStaticQuantize`, added later).
**`reshape` aliasing is not counted**: it always applies, so there is nothing to observe; identity
expand depends on a shape condition and can fail, so it is counted.

### 4. Division of graph rewrites: exporter normalisation or runtime fusion

Removing wasteful nodes has two homes. The criterion is **whether the rewrite stays inside the IR
vocabulary**:

| Rewrite                                                     | Home                                                     | Examples                                                                                 |
| ----------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Stays inside the IR** (the result is expressible in IR ops) | exporter `normalize.py` (`tools/exporter/src/karume/`)   | composing adjacent permutes `p∘q`, removing identity permute, identity clone / repeat     |
| **Collapses into a private kernel not in the IR**           | runtime `fusion.ts`                                      | RoPE 7 nodes → 1 dispatch, SiLU 2 → 1, upsample2x 6 → 1, identity expand alias            |

Putting the first kind in the runtime scatters optimisations that cannot be seen from the IR (IR node
count diverges from the real dispatch count). Putting the second kind in the exporter publishes
private ops in the IR and breaks ADR 0007's allowlist.

**Asymmetry:** `normalize.py` works only on the torch.export path (not on IR written by other paths or
on already exported assets); `fusion.ts` works on any IR, because it matches on the shapes seen at run
time. If a rewrite can be written in both places, the fusion pass wins on reach and normalisation wins
on IR readability; the choice is made case by case, and this ADR sets no default.

## Consequences

- **The public IR vocabulary does not grow.** The suspicion in `op-vocabulary.md` that `silu` in
  `UnaryOp` was unreachable was resolved by an internal fusion rule, without a public op.
- **Adding a fusion is cheap** (one rule in `FUSION_RULES`, one kernel, matcher tests). In return,
  matcher acceptance is fixed to the measured shapes, so a change in the exporter's emission order or
  shapes disables a fusion silently. Watching `lastRunFusions` — and the fusion-count tests
  `packages/e2e/tests/e2e_*_fusion_counts_test.ts` — is part of operating it.
- Fusion kernels are f32 by default and use no optional feature on the default path. The exception is
  the opt-in subgroup variant of rmsNormAdd (`packages/runtime/src/kernels/rms-norm-subgroup.ts`,
  ADR 0100). Fusion on an f16 compute path is out of scope.
- Of the other candidates in the reference triage, VAE channel L2 (30 chains) remains unimplemented;
  the conditioner's D64 RoPE (22 chains then) fuses since the BSHD addendum of 2026-09-12 (24 chains,
  `packages/e2e/tests/e2e_anima_fusion_counts_test.ts`). VAE channel L2 loses the f32 rounding boundary that `mul`'s
  storage write creates, so reproducing the reduction order with u32 staging must be proven first
  ([large-designs.md](../research/2026-08-06-kernel-triage/large-designs.md) F3). Adaptive norm was
  implemented on 2026-08-10 (addendum below).

## Addendum (2026-08-08) — measurements

- Fusion hits on Anima 1024 / 8 steps on an RTX 3080 Ti: rope 503, silu 305, upsample2x 27,
  identityExpand 160 (matching the reference branch's static count); SBV2 fires no rule. End-to-end wall
  time is in [research/2026-08-08-branch-adoption-perf.md](../research/2026-08-08-branch-adoption-perf.md);
  the share of fusion alone was not separated.

## Addendum (2026-08-10) — in-window passthrough and adaLN fusion (OP-008)

Adaptive norm (DiT modulation) is the fourth rule, `adaln`. Decision 2's exact matching and decision
1's bookkeeping are unchanged; the scan gains one concept: part of a contiguous window passes through
without being folded.

In the real IR the `layer_norm → mul → add` chain is not adjacent: 2–3 `reshape` nodes (unsqueezing
shift / scale / gate) sit between `layer_norm` and `add(scale, const[1])`. They do not consume the
`layer_norm` output, so moving them before the fused step is legal.

1. **A rule declares a contiguous window (`window`) and the subsequence to fold (`chain`); the
   remainder is emitted as plain nodes before the fused step**, and `planFusions` advances by the window
   width.
   - MUST: scan width, passthrough nodes, release bookkeeping (`ins`) and the folded count are all
     derived by `defineRule` from the window and the chain; rules do not declare them (decision 1's MUST
     extended to windows). If the chain is not a subsequence of the window, `defineRule` fails loudly
     (a missed node is never executed, an extra node runs twice; both are silent wrong values).
   - MUST: passthrough nodes consume no value defined by the chain (`passthroughIsIndependent`). For
     adaln this is implied by `internalsArePrivate`, but it matters on its own for a future window whose
     final chain node precedes a passthrough node (`internalsArePrivate` does not look at the final
     output).
   - `reshape` is also the head op of upsample2x, so a matcher test checks that skipping a window never
     takes an opportunity from another rule (every window start × every other rule captures nothing).
2. **Acceptance is fixed to the measured shapes (decision 2 unchanged).** The passthrough is only the
   2–3 consecutive `reshape` nodes right after `layer_norm`, the modulation only a `[1,…,1,dim]`
   broadcast, the constant only of shape `[1]`, and the `mul` / `add` input order only the measured
   one. Constants are not assumed: baking `1.0` would leave "not captured means correct", so `one` is
   bound as a buffer and the kernel reads `one[0]` (storage 6 in + 1 out = 7, within the default limit
   of 8).
3. **Kernel `src/kernels/adaln-norm.ts`** uses the plain layer_norm structure (one row per workgroup of
   256, grid-stride along the row) and shares the row-statistics and affine WGSL text with it
   (`LAYER_NORM_ROW_STATS_WGSL` / `LAYER_NORM_AFFINE_WGSL`), so the plain `layer_norm` WGSL did not move
   by a byte. The rounding barrier is u32 staging plus barrier at two points (`1 + scale` and `t · s`). Of the
   three materialisation points of the plain chain, `t` needs no barrier (its consumer is a
   multiplication, so there is no reduction to reorder). The output loop is the block loop of silu: an
   `o = lid` while loop is workgroup-nonuniform and cannot hold a barrier.
4. **`adaln` is added to `FusionCounterName`**, and `FusedDispatch.workgroupSize` is documented as the
   grid-stride divisor, not `@workgroup_size` (row-per-workgroup kernels pass 1).
5. **The execution side is unchanged**; the rule fits the common "params, inputs…, output" bind surface.

On the RTX 3080 Ti / Vulkan the twin-graph check (`packages/runtime/tests/gpu_adaln_fusion_test.ts`,
adversarial inputs including subnormals, ±0, Inf/NaN rows and distributivity-provoking scales) was
bit-identical for windows 6 and 7, which settled the form that also folds `1 + scale`. GPU time dropped
by 12.7 ms/step (w8a8-1024) and the DiT step count by 255 per run.

## Addendum (2026-08-11) — rope rule: derived head width and in-window passthrough

RoPE did not fire on EmbeddingGemma-300m (head width 256), so the acceptance set gained two measured
shapes of the same half-split `rotate_half`, not a generalisation by similarity:

1. **The head width D is derived from the split position** (it was fixed to 64 / 128). The kernel
   takes head_dim / half_dim as uniforms, so its WGSL did not change.
2. **In-window passthrough for rope.** The `sym_prefix_slice` that trims the cos / sin table to the
   run-time T appears once per θ family, and its first occurrence falls between `cat` and the cross
   `mul`. Exactly one `sym_prefix_slice` may be skipped (another op, or two or more, is rejected;
   pinned by counterexample tests).

EmbeddingGemma rope went from 0 to 48 hits and the Anima text encoder from 55 to 56; the PNG checks
stayed bit-identical.

## Addendum (2026-09-12) — BSHD half-split RoPE

Gemma 4's real graph uses `[1,S,H,D]` with table `[1,S,1,D]`. BSHD is accepted only when both tables
match that form exactly. The table token index is `row / H`, under its own key
`rope:v1:half:bshd:f32:wg256` (`ROPE_BSHD_KEY` in `src/kernels/rope.ts`), separate from BHSD's
`row % S`; a degenerate shape that fits both keeps BHSD and its WGSL. Kernel generation takes the axis
order as an argument and shares everything except the table index; it runs nothing at module import.
The other conditions (u32 workgroup staging of the two products with a barrier, grid-stride, B = 1,
even D, half split on dim 3, private internals, wiring, dtype, in-window passthrough) are unchanged, as
are the IR, the public API, stored weights and the numerics contract; non-matching shapes run as
primitive ops. The 50 chains of regular / QAT E2B and the 24 chains of the Anima text conditioner match.
BSHD has GPU bit-identity tests, counterexamples and snapshots, and the fusion-count tests were
updated. Because the rounding barrier depends on the backend, the bit-identity twin-graph tests that run
in the default lanes (`packages/runtime/tests/gpu_rope_fusion_test.ts` and the other
`gpu_*_fusion_test.ts`) are the check of record. Confirmation on another backend such as an Apple M2
SHOULD be run when such a reference machine is available (re-graded 2026-10-18 to SHOULD: the
repository has no M2 check — the reference rows cover only the RTX 3080 Ti and Intel BMG environments —
and upstream golden comparison is opt-in, `KARUME_UPSTREAM_PARITY=1`, per ADR
[0129](0129-test-layout-and-lanes.md), addendum 2026-10-17 (2)).
Rationale and Chrome measurements: [research](../research/2026-09-12-chrome-gemma-optimization.md).

## Addendum (2026-09-21) — splitting the fusion pass into files (entry stays `fusion.ts`)

Decision 1's "pure-function pass in `src/runtime/fusion.ts`" stays as the name of the pass's entry
point; the body is split in three:

- **One rule per file**: `src/runtime/fusion-rules/<rule>.ts`. The pure row-block function
  `planRowBlocks` lives with `rowBlockAttention`.
- **Common types and the eligibility conditions shared by all rules** (the execution step types such
  as `FusedStep`, `windowIsSingleOutput` / `windowTouchesState` / `allF32` / `internalsArePrivate` /
  `externalIns` / `passthroughIsIndependent`) and `defineRule` are in `src/runtime/fusion-rule.ts`.
- **The entry point** (`planFusions`, the declaration table `FUSION_RULES`, aliasing `planAliases`,
  candidate enumeration `enumerateUnfusedWindows`, the diagnostic counters) stays in
  `src/runtime/fusion.ts`, which re-exports the public types and `planRowBlocks`; **consumers import
  through `fusion.ts`**.

Dependencies run one way, fusion-rule → fusion-rules → fusion; rules never import `fusion.ts` (the
table imports the rules, so that would be a cycle).

MUST: `FUSION_RULES` stays **one hand-written array**. Self-registration at rule import would violate
the zero-side-effect invariant (no execution at import), and the mechanical disjointness check of head
ops in `tests/runtime_fusion_test.ts` reads **this array**. Decision 1's property that order does not
affect the result is unchanged. Matching, generated output and the public surface did not change by a
byte (lines moved only).

## Addendum (2026-10-24) — three accepted forms for the Wan DiT

Only `silu` fired on the Wan DiT, whose small ops took about 17% of a forward in Chrome
([research](../research/2026-10-24-wan22-chrome-kernel-breakdown.md)). Three forms are added, each matched exactly (op sequence, wiring, shapes,
dtypes, private internals) and computed with the same f32 operations in the same order:

- `rope`, pair form: `slice(odd), neg, slice(even), cat, reshape, mul(x, cos), mul(rot, sin), add` on
  BSHD with tables `[1,S,1,D]` becomes one dispatch (`rope:v1:pair:bshd`); rot[2p] = −x[2p+1],
  rot[2p+1] = x[2p]. It fires on Wan2.1 / 2.2 (60 each) and on the Irodori sites with the same sequence
  (DiT 23, speaker 15); the first Irodori site, which slices its tables inside the window, does not match.
- `adaln`, per-token form: `layer_norm, where(mask, scale_c, scale_g), add(+1), mul, where(mask, shift_c,
  shift_g), add` becomes a pack step (the four vectors into one table) plus the body, so no kernel needs
  more than 8 storage bindings and fusion does not depend on device limits.
- `gatedResidual` (new rule, first op `where`): `where(mask, gate_c, gate_g), mul(y, g), add(residual, p)`
  becomes one dispatch.

The rounding points are kept with the workgroup-memory barrier described above (measured, not a WGSL
guarantee); per-rule GPU tests compare fused and plain runs bit for bit. The existing keys' WGSL is
unchanged, and the Wan sha256 rows did not move.

## Addendum (2026-10-25) — a consumer that absorbs its producer (linear prologue, perf-ledger F-4)

A new kind of fusion: **a consumer absorbs its producer through an annotation on its `NodeStep`;
decided by `planFusions` with the consumer's own eligibility predicate; executed by the consumer's
recipe.** It is not a `FusedStep`: the fused form is the consumer's own dispatch list with one kernel
swapped, so the consumer's choices (here the i8a8 route, the dot product form, the geometry, the i32
guard and the step temporaries of `buildLinearI8a8`) stay on its one path. It is the same split as the
packed activations of ADR [0105](0105-packed-static-quantize-activations.md).

The first instance: an elementwise `gelu_tanh` that feeds an a8 linear. The linear's step ① quantizes
`gelu_tanh(x)` directly (`quantizeRowsPrologueWgsl` in `packages/runtime/src/kernels/quantize-rows.ts`)
and the gelu node disappears, so its f32 output (33 frames of Wan2.2: `[7920, 14336]` f32 = 454 MB
per FFN) is never written or read.

- **The annotation**: `NodeStep.prologue = { op }` (`LinearPrologue` in
  `packages/runtime/src/runtime/fusion-rule.ts`). The linear's `NodePlan` in that step has its first
  input re-pointed to the producer's input (same shape and dtype — asserted), and the producer
  produces no step. Release bookkeeping stays decision 1's one path (`plan.node.ins` of a node step):
  the producer's input loses the producer's use and gains the linear's, and the folded value, which
  had exactly one use and is not a graph output, is never defined.
- **The exact match** (decision 2 — `linearPrologueAt` in `packages/runtime/src/runtime/fusion.ts`),
  on a two-node window: (1) the producer's op is in `LINEAR_PROLOGUE_OPS` (`gelu_tanh` and, since the
  measurement below, the erf `gelu`), f32 in and out; (2)
  its output has one use and is not a graph output (`internalsArePrivate`); (3) the next node is a
  `linear` that reads it in `ins[0]`; (4) that linear takes the i8a8 path by the **same predicate as
  the recipe** — `linearTakesI8a8` (`packages/runtime/src/runtime/linear-route.ts`: linearCompute
  `"a8"`, weight resident as i8 or i4, k > 0, k % 4 == 0) on the weight storage the executor projects
  for the fusion pass — and k fits the prologue kernel (`quantizeRowsPrologueEligible`: k % 4 == 0,
  k ≤ 16,384); (5) at least one row (the kernel binds `x` as `array<vec4<f32>>`, whose 16-byte minimum
  binding size an empty tensor does not meet). Anything else leaves both nodes as they are.
- **Fail loudly on the execution side**: a linear that carries a prologue but does not take the i8a8
  path with a prologue-eligible row, and any non-linear node that carries one, throw
  `ExecutionError` (the producer's op would otherwise never run).
- **The counter**: `linearPrologue`, once per folded pair, in the public `FusionCounts`
  (`lastRunFusions`). The Wan2.2 and Wan2.1 DiTs fold 31 pairs each under `linearCompute: "a8"` on
  their i8 seats — the FFN down projection of the 30 blocks (k 14,336 / 8,960) and the text
  embedder's second linear (k 3,072 / 1,536); `packages/e2e/tests/e2e_wan*_fusion_counts_test.ts`.
  anima's DiT folds 28 erf `gelu` pairs on its i8 seat under `"a8"` (the FFN down projection of the
  28 blocks, k 8,192); its text conditioner's 6 gelu → linear pairs stay unfolded, because that
  component's weights are f16 / bf16 on every preset; `packages/e2e/tests/e2e_anima_fusion_counts_test.ts`.
- **The estimate** sees the same decision: `EstimateOptions.linearCompute` and the weight storage of
  the residency plan go into the same `planFusions`, and a folded producer's output is not counted.
- **No rounding barrier** sits between `y = gelu_tanh(x)` and the quantization. The kernel's
  docstring holds the argument: "No rounding barrier sits between `y` and the quantization: the next
  operations are `abs` and `y / s` (a multiplication by `rcp(s)` after lowering), so no contraction
  `a·b+c` spans the boundary. WGSL also allows reassociation, so this is a fact measured at adoption,
  not a specification guarantee (the argument for `t` in `adaln-norm.ts`); the per-environment sha256
  rows are the check."
- **Measured**: `packages/runtime/tests/gpu_quantize_rows_prologue_test.ts` compares the kernel with
  the pair "elementwise op → quantize_rows" word for word for both ops (`xq` and `xs`, NaN scales included)
  and `packages/runtime/tests/gpu_linear_prologue_fusion_test.ts` compares Sessions with and without
  the fold (both dot product forms), on Vulkan / NVIDIA; the D3D12 / NVIDIA spike measurement is in
  [research 2026-10-25-speed-recon-5070ti](../research/2026-10-25-speed-recon-5070ti.md) §8.
- **The erf form (`gelu`) — measured 2026-10-25 before it entered `LINEAR_PROLOGUE_OPS`** (RTX 3080 Ti,
  Node 24.22 + Dawn and Chromium 156, both Vulkan; the production WGSL of `gelu`, `quantize_rows` and
  the prologue kernel): 0 differing 32-bit words in `xq`, `xs` and the f32 `y = gelu(x)` (read from a
  variant of the fused kernel that also stores `y`) against the pair, in two Node runs and one
  Chromium run, at anima's row (K 8,192, M 4,096 / 1,024) and DeBERTa's (K 4,096, M 256 / 64), on
  normal()·2 rows and on special values (subnormals, ±0 rows, 2e19, erf's negative tail −5.5..−2.5
  and the underflow to −0 at −15..−6, NaN, ±inf — the NaN payloads matched as well). The fused form
  takes 0.29–0.30 of the pair's time at anima's shapes (×3.3–3.4; −479 µs per call at M 4,096) and
  0.31 / 0.45 at K 4,096, M 256 / 64; it runs at the memory bound, and per element it stays within
  1.3% of the gelu-free register quantize up to K 14,336 (no register step below the kernel's bound
  of 16,384). **What this check does not see**: a fault of 1e-7 in one erf constant changed 24% of
  `y`'s words but only 1–3 of 2 M `xq` words and no `xs` word, so a comparison of `xq` / `xs`
  alone (the GPU tests above) hardly detects a gelu mismatch; the end-to-end check is the models'
  per-environment sha256 rows.

### The second instance: an attention that absorbs the permutes of its q / k / v (perf-ledger F-3 A4 / F-8)

The Wan DiTs project q / k / v token-major (`[B, S, H·D]` reshaped to `[B, S, H, D]`) and permute each
to head-major `[B, H, S, D]` with a materialised strided copy right before the fused `attention` (and
permute O back right after it). On an a8 attention seat the attention **absorbs the input permutes as
their consumer**: its a8 quantisation reads the token-major tensors itself, so none of those three
copies is a node output. The permute after the attention stays a node.

- **The annotation**: `NodeStep.tokenMajor = { q, k, v }` (`AttentionTokenMajor` in
  `packages/runtime/src/runtime/fusion-rule.ts`). The attention's `NodePlan` in that step has each
  absorbed operand re-pointed to the permute's input (name, token-major shape and dtype); its output
  stays the declared head-major O. The absorbed permutes produce no step. Release bookkeeping stays
  decision 1's one path, as for the linear prologue: each permute's input trades the permute's use for
  the attention's, and the absorbed values are never defined. `nodeStepSpan` gives the number of
  original nodes a node step stands for (the candidate enumeration and the estimate read it).
- **The gate is the a8 attention seat** (`FusionContext.attentionCompute`: the executor passes
  `SessionOptions.attentionCompute`, the estimate `EstimateOptions.attentionCompute` — a session knob,
  no device facts). Only the a8 quantisation reads a token-major operand natively: the token-major
  `quantize_rows` writes the head-major outputs of "permute → quantize_rows", and one tiled transpose
  per batch builds Vᵀ from a token-major V in place of "permute → the strided Vᵀ copy". A dense seat
  (f32 / f16) reads q / k / v as stored, so an absorbed permute would come back as the same strided
  copy inside the attention step — no dispatch saved — while the token-major input stays live next to
  its copy for the whole step (estimated for the Wan2.2 DiT, `linearCompute` `"a8"`: the live peak
  would rise 2,385.1 → 2,663.5 MiB at S 7,920 and 3,610.4 → 4,569.4 MiB at S 27,280). So a dense seat
  absorbs nothing.
- **The seat, not the route**: `attentionRoute` (`packages/runtime/src/runtime/attention-route.ts`)
  stays the one decision of which kernels run. Inside an a8 seat a stage the shape keeps off i8a8 (①QK
  at D % 4 ≠ 0, ③PV at N % 4 ≠ 0) reads a head-major copy the recipe makes with the absorbed node's own
  strided permute, into a node temporary (`attentionHeadMajorCopies` in
  `packages/runtime/src/runtime/recipe-builders/attention.ts`, shared with the estimate), so that
  stage's work and values are the graph's byte for byte. A route-aware match would have to copy the
  per-stage eligibility and the online kernel's device facts into the fusion pass — the second
  spelling `attentionRoute`'s module doc forbids.

  | Route of an a8 seat | q / k token-major | V token-major |
  | --- | --- | --- |
  | online (`"none"`) and the three a8 dispatches (S in f32 / f16) | the a8 quantisation reads them (`quantizeRowsBshd*`) | Vᵀ by the tiled transpose (`transpose_tiled`) |
  | ①QK kept on f32 by the shape (D % 4 ≠ 0) | the absorbed node's strided permute into a node temporary, before ①QK | Vᵀ by the tiled transpose |
  | ③PV kept on f32 by the shape (N % 4 ≠ 0) | the a8 quantisation reads them | the absorbed node's strided permute into a node temporary, before ③PV |

  The quantisation stays the a8 path's own dispatches (`quantizeAttentionQk` / `quantizeAttentionPv`,
  shared by the online and the three-dispatch routes), so both routes switch together. Only addresses
  change (the transpose is a copy), so the values are bit-identical to the permutes as nodes.
- **The output permute stays a node.** On the three a8 dispatches ③PV stores O head-major, so absorbing
  the permute after it would only move the same strided copy into the attention step and add an O
  temporary. On the online route the kernel could store O token-major (one dispatch less, the same live
  bytes), but the first-fit placement of the transient plan (`planTransients`, ADR
  [0093](0093-transient-liveness-packing.md) decision 1) turns the removal of that node into a larger
  backing: the Wan2.2 DiT's region total — what the slot backing allocates — is 888.0 MiB with the
  permute as a node and 1,042.7 MiB with it absorbed at S 7,920 (3,043.3 / 3,576.1 MiB at S 27,280),
  the live peak unchanged at 634.2 / 2,169.2 MiB. The output permute stays a node until the transient
  placement no longer turns its removal into a larger backing.
- **The exact match** (decision 2 — `attentionPermutesAt` in `packages/runtime/src/runtime/fusion.ts`),
  on an a8 seat, on the window from the scan index: a run of `p ≥ 1` nodes directly followed by a
  fused `attention` (no states field — the states form is another kernel family), f32, one output,
  where **every** node of the run is a permute the attention may absorb: dims `[0, 2, 1, 3]` on a
  rank-4 f32 tensor, materialised (a permute that aliases its input — an axis of length 1 — costs
  nothing, and absorbing it would put the copy back on the stages that make it themselves), its output
  with one use, not a graph output, consumed in the attention's q, k or v slot (never the mask). A run
  holding anything else does not match at that index, and its tail is tried again from the next one,
  so the absorbed permutes are the longest such run that ends at the attention — contiguous, so each
  re-pointed input lives only across the other absorbed permutes. Anything else leaves every node as
  it is.
- **Fail loudly**: a node other than a fused attention that carries the annotation throws
  `ExecutionError` in the recipe builder and in the estimate (it would read the re-pointed inputs in
  the wrong layout).
- **The counter**: `attentionPermute`, once per absorbed permute (up to three per attention), in the
  public `FusionCounts` (`lastRunFusions`). Counted 2026-10-25 from the distribution graphs on the a8
  attention seat (`packages/e2e/tests/e2e_*_fusion_counts_test.ts`): the Wan2.2 and Wan2.1 DiTs 180
  each (q, k and v of 60 attentions); anima's DiT 56 (v of 56 attentions — its q / k reach the
  attention through the per-head `rms_norm` and the RoPE). On their f32 / f16 attention seats they
  count 0, and so does every other family: only the Wan and anima presets set `attentionCompute`
  `"a8"`.
- **The estimate** sees the same decision (`EstimateOptions.attentionCompute` goes into the same
  `planFusions`; `postFusionSteps` maps the node step back over `nodeStepSpan` nodes) and counts the
  attention step's own temporaries from the same pure functions as execution
  (`attentionHeadMajorCopies`, with `attentionI8a8TempBytes` unchanged).
- **Memory**: where the a8 quantisation reads an operand natively, its token-major input replaces the
  head-major copy in the live set and the copy step (input and output live together) is gone, so the
  live peak does not rise; a stage the shape keeps on f32 holds the token-major input next to its copy
  for the whole step. Estimated (i8 seat, `linearCompute` `"a8"`, 2 GiB − 4 B binding limit; live
  peak / region total, MiB, with the permutes as nodes → absorbed):

  | DiT, seat | S | live peak | region total |
  | --- | --- | --- | --- |
  | Wan2.2, online | 7,920 / 27,280 | 634.2 → 634.2 / 2,169.2 → 2,169.2 | 888.0 → 880.3 / 3,043.3 → 3,016.7 |
  | Wan2.2, s16 | 7,920 / 27,280 | 2,070.6 → 2,070.6 / 4,173.7 → 4,173.7 | 2,325.1 → 2,163.4 / 5,052.6 → 4,493.4 |
  | Wan2.1, s16 | 14,040 / 32,760 | 2,064.0 → 2,064.0 / 3,349.0 → 3,349.0 | 2,411.9 → 2,247.8 / 4,161.5 → 3,777.9 |
  | anima, s16 (v only) | 4,096 / 9,216 | 729.0 → 729.0 / 1,783.7 → 1,783.7 | 880.5 → 896.6 / 2,124.6 → 2,160.6 |

  The region total is first-fit placement over the live intervals, so it can move either way while
  the live peak stays put: anima's DiT, which absorbs only v, grows by 16.1 / 36.0 MiB (the same on
  its S-in-f32 a8 seat), the same placement effect as the output permute's.
  `packages/e2e/tests/e2e_wan_ti2v_attention_permute_estimate_test.ts` holds the Wan2.2 rows: on the
  online and s16 seats at S 7,920 / 27,280 the region total of the distributed graph is at most that
  of the same graph with a zero-dispatch reshape between each input permute and its attention.
- **Measured** (2026-10-25, RTX 3080 Ti, Node 24.22 + Dawn, Vulkan):
  `packages/runtime/tests/gpu_attention_token_major_kernels_test.ts` compares each kernel form with the
  composition it replaces, from the production generators — the token-major `quantize_rows` with
  "strided permute → quantize_rows" (D 64 / 128 / 1024, H 12 / 24, S with tails, B 1 / 2, a zero and a
  NaN row, a clamped grid), the tiled transpose with "the graph's permute → the recipe's strided Vᵀ
  copy" (tails in both axes, batch 2) — 0 differing words;
  `packages/runtime/tests/gpu_attention_permute_fusion_test.ts` compares Sessions with and without the
  absorption (a zero-dispatch reshape breaks the match) — bit-identical outputs on the online route,
  the three-dispatch s16 route, and the a8 shapes whose ①QK (D 14) or ③PV (N 43) stays on f32, with
  9 → 6, 11 → 8, 9 → 8 and 9 → 7 dispatches; on the f32 seat nothing is absorbed (7 → 7);
  `packages/runtime/tests/gpu_estimate_test.ts` checks that the estimate equals the slot table on the
  four a8 routes. Not measured here: other backends (D3D12, Metal), the timings, and the models'
  per-environment sha256 rows (bit identity predicts them unchanged).

### Stage 3 of the same wave: rms_norm → pair RoPE in one kernel (rule `rmsNormRope`, perf-ledger F-3)

Not a consumer annotation: a `FusedStep` rule like those of decision 1, `rmsNormRope`
(`packages/runtime/src/runtime/fusion-rules/rms-norm-rope.ts`), f32 in and out. The Wan DiTs normalise
each token row of the self-attention q and k (`rms_norm` over H·D) and rotate it with the pair-form
RoPE through two reshape aliases — two dispatches and one stored f32 row per projection. The rule
folds the chain into one dispatch of `RMS_NORM_ROPE_WGSL` (`packages/runtime/src/kernels/rms-norm-rope.ts`,
key `rms_norm_rope:v1:pair:bshd:f32:wg256`), which never stores the normalised row; the attention's own
a8 quantize still reads the RoPE output token-major.

- **The exact match** (decision 2), from the scan index: `rms_norm(x, weight) → n` with x
  `[1,S,H·D]` f32 (last-dim and weight-only by its contract, ADR 0017); `reshape(n) → v`
  `[1,S,H,D]`; `reshape(v) → pairs` `[1,S,H·D/2,2]`; then, directly after, exactly what the rope rule
  accepts for the pair / BSHD form of the addendum of 2026-10-24 — the same matcher
  (`matchRopePairForm`), so D is even and a pair never straddles a head. All f32, no graph output and
  no other consumer inside, no state, bind surface x / weight / cos / sin without duplicates. Between
  the chain's nodes only `rms_norm` and `reshape` nodes may sit (the other projections'); anything
  else, a bias `add` after the norm, a per-head norm (`[1,S,H,D]` rows — anima's spelling, which also
  feeds the half split in BHSD), a half-split RoPE, an odd D or one more consumer of `n`, `v` or
  `pairs` leaves the nodes (and the RoPE then folds alone under `rope`).
- **Only where the plain path runs the kernel it is bit-identical to**: the 256-wide workgroup
  `rms_norm` (`rms_norm:v1:f32:lastdim:wg256`). The choice is one predicate, `rmsNormKernel` in
  `packages/runtime/src/kernels/rms-norm.ts`, which `buildRmsNorm` and the rule both call: a row of at
  most 128 (the 128-wide form) or `rmsNormReduce: "subgroup32"` (another summation order, ADR
  [0100](0100-rms-subgroup-reduction.md)) keeps the nodes.
- **Internal values read twice.** `v` feeds the pair reshape and the RoPE's direct `mul`; `pairs`
  feeds both slices. Decision 2's `internalsArePrivate` (exactly one consumer) rejects that, so the
  rule uses a sibling, `internalsStayInChain` (`packages/runtime/src/runtime/fusion-rule.ts`): every
  use of every internal value is an input slot of a chain node, there is at least one, and none is a
  graph output — the fold loses no value for the same reason. It does **not** replace
  `internalsArePrivate` for every rule: counted reads do not know which slot reads the value, so
  under it `add(n, n)` after `rms_norm → n` would pass and the rmsNormAdd matcher would bind `n`, a
  value fusion never materialises, as its residual. Only a matcher that pins every slot an internal
  value may occupy, with a bind surface of external values only, uses it.
- **Deferred nodes — a change to the window mechanism.** In the Wan IR the q and k chains
  interleave: `rms_norm` q, `rms_norm` k, the reshapes of q / k / v, q's pair reshape and its 8 RoPE
  nodes, k's pair reshape and its 8 RoPE nodes. q's window therefore holds k's `rms_norm` and reshape;
  as in-window passthroughs (the addendum of 2026-08-10, placed before the fused step) they would be
  emitted as plain nodes and k could never fold. A rule may now **defer** its window's other nodes
  (`FusionMatch.others: "deferred"`; only this rule does): the fused step takes the window's place,
  and the deferred nodes are written into the window's last slots of the scanned list, where the scan
  resumes, so they come after the fused step and any rule may grab them again — k's `rms_norm` heads
  k's window, with only v's reshape between its nodes (deferred once more, then a plain alias).
  - MUST: legal only when the chain and the deferred nodes are independent both ways —
    `passthroughIsIndependent` (they read no chain value) and `deferralIsIndependent` (the chain
    reads none of their outputs), both checked by `defineRule`. A table reshaped inside the window
    and read by the RoPE is rejected by the second.
  - The step list is then a valid order that differs from the declaration order only inside the
    deferred region. The estimate's `postFusionSteps` places the steps it keeps in post-fusion form
    by the running node count, which is the position in the scanned list there; MUST: a rule defers
    only nodes that cannot head such a step (a row-block attention's `bmm`, a linear-prologue
    producer, a permute an attention absorbs) — `rms_norm` and `reshape` head none.
  - Checked once on the distribution graphs without a GPU (2026-10-25): every step of the Wan2.2 and
    Wan2.1 DiTs reads only values defined before it, and every value that exists is read exactly as
    often as the graph reads it (the release bookkeeping of decision 1), with the rule and under
    `"subgroup32"`; `runtime_fusion_test.ts` checks the same on its test graphs.
- **Decision 1's disjoint heads gain one declared exception.** `rmsNormRope` and `rmsNormAdd` share
  the head `rms_norm`. They never accept the same index: rmsNormAdd needs the normalised value's single
  use to be the next node, an `add`; rmsNormRope needs every use inside its chain, which reads it only
  in a `reshape`. The mechanical head check of `packages/runtime/tests/runtime_fusion_test.ts` now
  allows exactly this pair, and a test applies every rule of `FUSION_RULES` at every index of graphs
  holding both forms (and one normalised value feeding both) and finds at most one accepting rule, so
  the order still cannot change the result.
- **Why it is bit-identical** — the kernel's docstring: "Bit identity with the pair "rms_norm →
  rope:v1:pair:bshd" (measured at adoption on Vulkan / NVIDIA, Node + Dawn and Chromium: 0 differing
  words at the Wan2.2 and Wan2.1 shapes, special values and NaN payloads included — research of
  2026-10-25, wave a stage 3): the reduction, the normalised value and the RoPE products are the plain
  kernels' strings, so the compiler's contraction decisions are theirs; the normalised value is
  parenthesised where it feeds the products, the form the adoption measurement compiled; the
  normalised value meets only multiplications before the products are staged, so no contraction spans
  the stored-then-read boundary the pair had; the staging itself is the RoPE rounding barrier (its
  MUST — without it 19–25% of the words move by one ulp on this backend); the backend measured flushes
  subnormals in both kernels alike; a backend that keeps them is unchecked."
- **The counter**: `rmsNormRope`, once per folded chain, in the public `FusionCounts`
  (`lastRunFusions`). Counted 2026-10-25 from the distribution graphs
  (`packages/e2e/tests/e2e_wan*_fusion_counts_test.ts`, and the GPU lanes' `EXPECTED_FUSIONS` of
  `e2e_wan_ti2v_dit_test.ts` / `e2e_wan_dit_test.ts`): the Wan2.2 DiT 60 (q and k of 30 blocks, row
  3,072) and the Wan2.1 DiT 60 (row 1,536) on every seat and S checked, and `rope` drops 60 → 0 on
  both — each pair-form RoPE of those DiTs now folds with its `rms_norm` (under `"subgroup32"` the
  counts are `rope` 60, `rmsNormRope` 0). Every other family counts 0 and keeps its `rope` count:
  anima's q / k reach the half split through a per-head `rms_norm` in BHSD, and Irodori's pair RoPE
  (DiT 23, speaker 15) is fed by `rms_norm → mul(weight) → slice` of the heads, not by two reshapes.
- **The estimate does not replay it** (the precedent of the addendum of 2026-10-24: only row-block
  attention, the linear prologue and the absorbed permutes are counted in post-fusion form). It counts
  the chain in its pre-fusion form, so besides the RoPE intermediates it already over-counted it now
  also over-counts the normalised row (`S · H · D · 4` bytes per chain — 93 MiB at the Wan2.2 S 7,920,
  only where that row is live at the peak).
- **Measured** (2026-10-25, RTX 3080 Ti, Node 24.22 + Dawn, Vulkan):
  `packages/runtime/tests/gpu_rms_norm_rope_test.ts` compares the kernel with the pair "rms_norm
  (`rms_norm:v1:f32:lastdim:wg256`) → `rope:v1:pair:bshd`", the normalised row bound as both `x` and
  `pairs`, from the production generators — H·D 24·128 / 12·128 / 3·128 (a partial last chunk), S 1 /
  7 / 300, a two-workgroup grid over 301 rows, normal()·2 rows, special rows (zero, −0, subnormals,
  ±2e19, underflowing squares, NaN, ±inf, two NaN payloads) and special weights and tables — 0
  differing words, NaN words included; `packages/runtime/tests/gpu_rms_norm_rope_fusion_test.ts`
  compares Sessions on the interleaved q / k form with and without the match (a zero-dispatch reshape
  after each `rms_norm` breaks it) — bit-identical outputs at H 24 / 12 / 3 (D 128) and over a
  two-workgroup grid, `rmsNormRope` 2 / `rope` 0 against 0 / 2, and 3 against 5 dispatches. Not
  measured here: Chromium and other backends (the adoption spike covered Chromium on Vulkan), the
  timings (the spike: 0.57–0.61 of the pair's time), and the models' per-environment sha256 rows (bit
  identity predicts them unchanged).

## Addendum (2026-10-26) — the attention absorbs its output permute on the online route (perf-ledger F-3 A4)

This revises the bullet "The output permute stays a node" of the addendum of 2026-10-25 (its second
instance). **A fused attention without states absorbs the `[0, 2, 1, 3]` permute that directly follows
it only when `attentionRoute` chooses the online kernel for that node** (ADR
[0132](0132-online-int8-attention.md)). The online kernel then stores O token-major `[B, M, H, D]`
itself, so the permute's strided copy is never a dispatch. The absorption of the input permutes (q /
k / v on the a8 seat) is unchanged.

- **The annotation**: `AttentionTokenMajor` regains `o`. With `o`, the attention's `NodePlan` in its
  node step has its output re-pointed to the output permute's output (name and token-major shape), and
  the permute produces no step. Release bookkeeping stays decision 1's one path, as in the 2026-10-25
  addendum: the attention's head-major output had exactly one use and is not a graph output, so it is
  never defined. `nodeStepSpan` and the `attentionPermute` counter include the output permute.
- **The exact match** (decision 2 — `attentionPermutesAt` in `packages/runtime/src/runtime/fusion.ts`),
  on an a8 seat: the attention's output has one use and is not a graph output, that use is the next
  node, and that node is a permute the attention may absorb (dims `[0, 2, 1, 3]` on a rank-4 f32
  tensor, materialised). In addition the route must be the online kernel: `attentionRoute` on the
  Session's seats (`attentionCompute` `"a8"` × `attentionScoreStorage` `"none"`), the node's D and key
  count N, and the device's facts (D ∈ {64, 128}, N % 4 == 0, and a device that holds the `sg32` or
  the `wg` reduction form). The output permute is also absorbed when no input permute is (at least one
  permute of the four is absorbed). Anything else leaves the permute as a node.
- **The kernel form**: the online kernel's token-major store (`AttentionOnlineOutput` `"bshd"` in
  `packages/runtime/src/kernels/attention-online-i8a8.ts`, key suffix `:obshd`, H in a second uniform
  block). Only the store address changes, so O is bit-identical to "head-major store → strided permute".
  `buildAttention` passes `"bshd"` with H only on the online route.
- **Why the online route only.** The addendum of 2026-10-25 kept the output permute as a node because
  the first-fit placement turned its removal into a larger backing. The least-growth placement (ADR
  [0093](0093-transient-liveness-packing.md) addendum of 2026-10-26) removes that effect on the online
  route: the Wan2.2 DiT's region total is 717.8 MiB at S 7,920 and 2,457.2 MiB at S 27,280 with or
  without the output permute absorbed (live peak 634.2 / 2,169.2 MiB). On every other route ③PV stores
  O head-major. There, absorbing the permute would need an O temporary plus the same strided copy
  inside the attention step. That adds one O of live bytes (+92.8 / +319.7 MiB on the Wan2.2 DiT at S
  7,920 / 27,280) and saves no dispatch. So those routes keep the permute as a node.
- **Why the gate is the route, decided in the pass.** For the input permutes the 2026-10-25 addendum
  chose the seat, because every route an a8 seat can take honours a token-major q / k / v. That does
  not hold for O:
  - **No fallback path.** The token-major O exists only in the online kernel. The recipe builds no O
    temporary and no copy for the other routes. Instead `attentionHeadMajorCopies`
    (`packages/runtime/src/runtime/recipe-builders/attention.ts`, shared with the estimate) fails
    loudly when `o` reaches a route other than the online kernel. A seat-only gate would hand `o` to
    the shapes and devices whose `"none"` seat falls back to the three a8 dispatches.
  - **One decision function.** The pass calls the same `attentionRoute` as `buildAttention`, with the
    same seats, device facts, D and N. The route is not written a second time, so the pass and the
    recipe cannot disagree. The objection of the 2026-10-25 addendum to a route-aware match was a
    second spelling of the per-stage eligibility and the device facts inside the pass; calling the
    function itself avoids that.
  - **Device facts in the fusion context have a precedent**: `FusionContext.limits`, inputs that
    decide the plan and are fixed per device (ADR 0022 forbids measuring and choosing again).
    `FusionContext` gains `attentionScoreStorage` (the session knob) and `attentionDevice` (the
    `AttentionDeviceFacts` the executor fixes once per Session for `buildAttention`; the estimate
    passes `EstimateOptions.attentionDevice`, already required with `"none"` and now checked at the
    estimate's entry whatever the graph). On `"a8"` × `"none"` without the device facts the decision
    fails loudly instead of guessing a device.
- **The estimate** replays the same decision: `postFusionSteps` passes the seats and the device
  facts to the same `planFusions`. On the online route the attention step writes the token-major
  output itself, so `workspaceBytes` still equals the slot table.
- **Effect on dispatches**: on the online route each attention runs one dispatch fewer (the output
  permute's strided copy). Where q / k / v are absorbed too and the online kernel takes one row window,
  an attention goes from 6 dispatches to 5: the quantisation of q and of k, the tiled Vᵀ transpose and
  Vᵀ's quantisation, and the online kernel (the counts move with the absorbed input permutes and the
  row windows). On the Wan2.2 DiT this removes the 60 strided copies after `attention_online` per
  33-frame forward (30 self- and 30 cross-attentions): 15.20 ms of GPU time in wave a's final
  per-kernel trace on the RTX 3080 Ti (the 60 strided copies after `attention_online` in one 33-frame
  forward; research [2026-10-25-wave-a-dit-fusions](../research/2026-10-25-wave-a-dit-fusions.md)).
- **Blast radius**: Sessions on `attentionCompute` `"a8"` × `attentionScoreStorage` `"none"` whose
  attentions take the online route and are followed by such a permute. By default that is the Wan2.2
  DiT's online presets: `attentionPermute` 180 → 240, the region total unchanged (717.8 / 2,457.2 MiB
  at S 7,920 / 27,280). Any other family takes it only where its user picks `"a8"` × `"none"` and the
  route is online. On the transformer i8 containers, on a device with subgroups and 48 KiB of
  workgroup storage, every attention of the anima DiT (56) and of the Wan2.1 DiT (60) takes the online
  route (`sg32`, D 128): `attentionPermute` 56 → 112 and 180 → 240, and the estimate's region total
  is the same with O absorbed and with O kept as a node (anima 224.0 / 504.0 MiB at S 4,096 / 9,216;
  Wan2.1 729.7 / 1,698.7 MiB at S 14,040 / 32,760; measured 2026-10-26 with `linearCompute` `"a8"`).
  The s16 seat and the a8 seat with S in f32 stay as before (Wan2.2 180, Wan2.1 180, anima 56), and
  so does every other seat (`packages/e2e/tests/e2e_*_fusion_counts_test.ts`).
- **Measured** (2026-10-26, RTX 3080 Ti, Node 24.22 + Dawn, Vulkan):
  `packages/runtime/tests/gpu_attention_token_major_kernels_test.ts` compares the token-major O store
  with "head-major store → strided permute" word for word (sg32 / wg, dp4a and the emulated dot, row
  windows); `packages/runtime/tests/gpu_attention_permute_fusion_test.ts` compares Sessions with and
  without the absorption (a zero-dispatch reshape before the attention's inputs and after its output
  breaks the match) — bit-identical outputs on the online route at D 128 (`sg32` and `wg`) and D 64
  (`sg32`), 9 → 5 dispatches (6 before this addendum), and the other routes keep their counts (s16
  11 → 8, an online seat whose shape falls back at D 96 11 → 8 with the census of a8 × `"f32"`, ①QK
  on f32 9 → 8, ③PV on f32 9 → 7, f32 seat 7 → 7);
  `packages/runtime/tests/gpu_estimate_test.ts` checks that the estimate equals the slot table on the
  online route with the output permute absorbed, including a shape whose O is larger than the
  attention's temporaries (there a head-major O next to the permute's output would raise the region
  total); `packages/e2e/tests/e2e_wan_ti2v_attention_permute_estimate_test.ts` checks that the Wan2.2
  DiT's region total with every absorbed permute is at most that of the graph with the permutes as
  nodes, on the online and s16 seats at S 7,920 / 27,280, and on the online seat at most that of the
  graph with only O's permute as a node. The models' sha256 rows of the Wan2.2, Wan2.1 and anima
  lanes are unchanged on this machine. Speed (in-process kernel A/B, [research](../research/2026-10-26-a4-output-permute.md)):
  −14.8 / −14.9 ms per 33-frame forward and −44 / −49 ms per 121-frame forward; the token-major store
  itself costs nothing measurable. Not measured here: other backends (D3D12, Metal).

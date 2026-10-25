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
  overlap.
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
  external consumer or readback would lose the value.
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
  on a two-node window: (1) the producer's op is in `LINEAR_PROLOGUE_OPS`, which holds only
  `gelu_tanh` (the erf `gelu` is one edit plus its tests away, once measured), f32 in and out; (2)
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
- **The estimate** sees the same decision: `EstimateOptions.linearCompute` and the weight storage of
  the residency plan go into the same `planFusions`, and a folded producer's output is not counted.
- **No rounding barrier** sits between `y = gelu_tanh(x)` and the quantization. The kernel's
  docstring holds the argument: "No rounding barrier sits between `y` and the quantization: the next
  operations are `abs` and `y / s` (a multiplication by `rcp(s)` after lowering), so no contraction
  `a·b+c` spans the boundary. WGSL also allows reassociation, so this is a fact measured at adoption,
  not a specification guarantee (the argument for `t` in `adaln-norm.ts`); the per-environment sha256
  rows are the check."
- **Measured**: `packages/runtime/tests/gpu_quantize_rows_prologue_test.ts` compares the kernel with
  the pair "elementwise gelu_tanh → quantize_rows" word for word (`xq` and `xs`, NaN scales included)
  and `packages/runtime/tests/gpu_linear_prologue_fusion_test.ts` compares Sessions with and without
  the fold (both dot product forms), on Vulkan / NVIDIA; the D3D12 / NVIDIA spike measurement is in
  [research 2026-10-25-speed-recon-5070ti](../research/2026-10-25-speed-recon-5070ti.md) §8.

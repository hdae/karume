# Numerics: the numeric requirement and what runs under it

What the runtime does now. Reasons are in ADR [0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md)
(decision 2) and ADR [0147](../decisions/0147-numerics-and-kernel-adoption.md); work that is not built is in the
[backlog](../backlog.md).

## The reference kernel and the band

- Every op has one reference kernel. Its output on a device is that device's reference for the op.
- Every op has a band: the tolerance a kernel's output must stay inside against an f64 evaluation of the op
  (`packages/runtime/src/reference/op-tolerance.ts`). An op without a row in that table fails; there is no default band.
- Bits are compared on one device. No level requires two devices or two backends to give the same bits.

## Levels

| Level | Meaning |
| --- | --- |
| `exact` | The reference kernel's bits on this device. |
| `reorder` | The summation order may differ from the reference kernel's. The error stays inside the op's band, and two runs of the same input on one device give the same bits. |
| a named approximation | Anything else. It is asked for by its own option and is never implied by a level. |

`exact` does not mean "accurate" or "the same on every GPU". A `reorder` kernel can be closer to f64 than the
reference kernel.

## The declaration

`SessionOptions.numerics` is `{ level?, ops? }`.

- `level` is `"exact"` or `"reorder"` for the whole model. Omitted, it is `"exact"`.
- `ops` gives a level per op kind: `linear`, `attention`, `rms_norm`. An op kind's own level wins over the model's, in
  either direction.
- There is no per-node scope.
- A level admits strategies; it does not name one. Declaring `reorder` for an op kind that has no `reorder` strategy
  is not an error: that op kind runs its reference kernel.
- A value outside this shape fails at Session construction: a value that is not an object, a level outside the two, an
  unknown field, an unknown op kind. An unknown key is never read as "not declared".
- The memory estimate takes the same declaration and derives the same kernels from it, up to what a kernel choice
  takes from the adapter's geometry profile: an estimate has no adapter and reads the default profile's
  ([geometry-profiles.md](geometry-profiles.md#the-gemv-choices-of-linear)).

## What a level selects

The selection is one derivation, shared by Session construction and the memory estimate
(`executionKnobDefaults` in `packages/runtime/src/runtime/session-options-check.ts`).

| Op kind | At `exact` | At `reorder` |
| --- | --- | --- |
| `linear` | The reference GEMM, and the GEMV kernels checked to give its bits. | With `linearCompute: "f32"`: the parallel GEMV on the shapes of its eligibility table, at 1 to 8 rows; other shapes and row counts keep the `exact` kernels. With `linearCompute: "a8"` or `"f16"`: that approximation's kernels, unchanged. |
| `attention` | The sequential reductions of the states form. | The parallel reductions of the states form, with the row statistics folded into PV where the shape admits the fold. The other form of attention has no `reorder` strategy. |
| `rms_norm` | The workgroup reduction. | The workgroup reduction (no `reorder` strategy is selected). |

The eligibility table of the parallel GEMV is the Session's geometry profile's (`linearGemv.parallelShapes` —
[geometry-profiles.md](geometry-profiles.md#the-gemv-choices-of-linear)). A row names a weight storage, a weight shape
and, for int4, the group length:

- Every storage the GEMV kernels read can have a row: int2, int4, int8, bf16, f16 and f32.
- A row serves 1 to 8 rows of the activation with one kernel and one summation order, so a decode step and the
  verification of a few rows give the same words for the same row.
- A float-stored weight (bf16, f16, f32) has no sequential GEMV past one row. At `exact` its 2 to 8 rows run the
  GEMM; at `reorder`, on a shape with a row, they run the parallel GEMV.
- A float-stored weight takes the parallel GEMV's plain form or its staged-activation form, as its row says. The
  packed-activation form, the linear→static_quantize fusion and the subgroup form exist for the integer storages
  only: a float-stored linear is never paired with packed activations, is not fused with a following
  static_quantize, and under the subgroup strategy runs the plain or staged form, which sums in the same order.

Each strategy's level and the kind of evidence it carries are stated in the strategy tables
(`packages/runtime/src/runtime/linear-strategies.ts`, `state-attention-strategies.ts`).

An op outside these kinds runs the same kernels at every level. Where it has more than one, the others are checked to
give its reference kernel's bits and the choice is a static rule of the shape, as for `linear`'s GEMV kernels:
`expert_linear`'s reference kernel is its one-row form, and a run of 2 to 768 tokens over weight rows of 64 words or
more takes the grouped form, whose words are the one-row form's (`expertLinearGrouped` in
`packages/runtime/src/kernels/expert-linear.ts`; the GPU tests and the `kernel-check` entry compare them).

### Strategies that depend on the adapter

The subgroup GEMV, the subgroup RMS reduction and the byte-plane GEMV are `reorder` strategies whose gain depends on
the adapter. A strategy that was not measured to gain on an adapter is not selected there, and nothing is tuned at
run time. No adapter has such a measurement, so no Session selects them. They run through the private Session key
`REORDER_STRATEGIES`, which the kernel checks and tests pass; it is not part of the public surface.

### Strategies admitted only by a private key

The split-K GEMM (`packages/runtime/src/kernels/linear-splitk.ts`) is a `reorder` strategy of `linear`, contract
class R (only the order of the additions differs):

- It is the split of the reference GEMM: the same kernel, at the tile geometry the reference GEMM takes for the row
  count, runs over one slice of the K tiles per workgroup of the dispatch's z axis. The tiles are dealt out in whole
  tiles, as evenly as they go. Each slice is summed from 0 in the reference GEMM's ascending order; a second pass adds
  the slices' partial sums in slice order, then the bias, last.
- `exact` never selects it. With one slice its two passes give the reference GEMM's words on every tile geometry the
  linear GEMM takes; that is the kernel's structural check, not a form any Session runs.
- No level, preset or geometry profile selects it. It runs only through the private Session key `LINEAR_SPLIT_K`,
  which its kernel tests pass; it is not part of the public surface. Under that key a linear takes it when its weight
  is stored as f32, `linearCompute` is `"f32"`, the GEMV family does not take the linear, and its K is cut by the
  slice rule: the largest of 16, 8, 4 and 2 slices that divides K and leaves each slice at least 256 terms.
- Its partial sums are a temporary of `slices × m × n` f32 values between its two passes.

## Named approximations

Each is its own Session option. A level never turns one on.

| Option | Values | Requires |
| --- | --- | --- |
| `linearCompute` | `"f32"` (default), `"a8"`, `"f16"` | — |
| `convCompute` | `"f32"` (default), `"a8"` | — |
| `attentionCompute` | `"f32"` (default), `"f16"`, `"a8"` | — |
| `attentionScoreStorage` | `"f32"` (default), `"f16"`, `"none"` | — |
| `fullStateStorage` | `"f32"` (default), `"f16"` | — |
| `packedStaticQuantize` | boolean | `linear` at `reorder` and `linearCompute: "f32"` (it is a form of the parallel GEMV) |
| `integerGemv` | boolean | `packedStaticQuantize: true` |

A combination that breaks a requirement fails at Session construction.

## Fusions

- A fusion that spans several ops applies only when its numbers meet the strictest level among those ops.
- Three built-in fusions give the bits of the unfused plan, so they apply at every level, wherever they match:
  rms_norm→add (`rmsNormAdd`), linear→static_quantize on the parallel GEMV (`linearStaticQuantize`) and
  gelu_tanh→mul→static_quantize (`geluMulStaticQuantize`). They have no option.
- Each is checked against the same graph without the rule, compared as u32 words
  (`packages/kernel-checks/src/fusion-twins/`). The test-only Session key `DISABLED_FUSION_RULES` leaves rules out by
  name to obtain that side; it is not part of the public surface.

## Presets and explicit options

- Order of precedence: an explicit option, then the preset's declaration, then the runtime's default (`exact`, no
  approximation).
- A preset declares its requirement in the manifest: `session.numerics` of a quant, with the same shape as the Session
  option, beside the approximation options it names. A `session` key outside the vocabulary rejects the manifest
  (`packages/hub/src/manifest.ts`).
- A pipeline composes the preset's `numerics` with an explicit one per field: the explicit `level` if given, else the
  declared one; per op kind, the explicit level if given, else the declared one. An explicit
  `{ ops: { attention: "exact" } }` keeps the levels the preset declares for the other op kinds
  (`resolveSessionOptions` in `packages/models/src/session/options.ts`).
- The check of the combination runs on the composed settings, not on the declaration alone.
- A family accepts `numerics` only where its policy table says so. Today that is Gemma 4.

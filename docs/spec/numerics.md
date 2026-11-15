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
- The memory estimate takes the same declaration and derives the same kernels from it.

## What a level selects

The selection is one derivation, shared by Session construction and the memory estimate
(`executionKnobDefaults` in `packages/runtime/src/runtime/session-options-check.ts`).

| Op kind | At `exact` | At `reorder` |
| --- | --- | --- |
| `linear` | The reference GEMM, and the GEMV kernels checked to give its bits. | With `linearCompute: "f32"`: the parallel GEMV on the shapes of its eligibility table; other shapes keep the `exact` kernels. With `linearCompute: "a8"` or `"f16"`: that approximation's kernels, unchanged. |
| `attention` | The sequential reductions of the states form. | The parallel reductions of the states form, with the row statistics folded into PV where the shape admits the fold. The other form of attention has no `reorder` strategy. |
| `rms_norm` | The workgroup reduction. | The workgroup reduction (no `reorder` strategy is selected). |

Each strategy's level and the kind of evidence it carries are stated in the strategy tables
(`packages/runtime/src/runtime/linear-strategies.ts`, `state-attention-strategies.ts`).

### Strategies that depend on the adapter

The subgroup GEMV, the subgroup RMS reduction and the byte-plane GEMV are `reorder` strategies whose gain depends on
the adapter. A strategy that was not measured to gain on an adapter is not selected there, and nothing is tuned at
run time. No adapter has such a measurement, so no Session selects them. They run through the private Session key
`REORDER_STRATEGIES`, which the kernel checks and tests pass; it is not part of the public surface.

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

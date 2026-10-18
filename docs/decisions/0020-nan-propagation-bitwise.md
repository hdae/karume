# 0020: NaN propagation on the GPU is guaranteed by a bitwise check

- Status: accepted (2026-08-03; the maintainer's ruling: option a, all five ops at once); the
  safe-softmax exemption of decision 5 overtaken by [0044](0044-runtime-attention-mask.md) (2026-08-31)
- Date: 2026-08-03
- Scope: `clamp` / `clamp_min` / `relu` (elementwise) and `amax` / `amin` (row reduce)
- Evidence (real GPU, 2026-08-02): `clamp(NaN, -1, 1) = -1`, `clamp_min(NaN, 0) = 0`, `relu(NaN) = 0`,
  while the CPU reference propagates NaN — breaking the agreement [limitations.md](../limitations.md)
  promises.

## Context: the mechanism

A WGSL comparison alone (`select(0.0, 1.0, NaN < m)`) is false as specified, but the shader compiler
folds the **whole** `select(x, m, x < m)` into the `max` idiom, and the driver's `max` drops NaN. So
"the select form propagates NaN" (ADRs 0015 / 0017) does not hold there; this ADR overrides them.
`leaky_relu` propagated only because x appears **in both branches**.

## Decision

1. NaN is detected **from the bits, without floating-point comparison**:
   `(bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u` (sign dropped; exponent all ones, mantissa
   non-zero). Integer `&` and `>` are not folded into min / max idioms.
2. Elementwise ops **wrap the existing value expression unchanged** in
   `select(finite, x, is_nan_bits(x))` (non-NaN WGSL textually identical — no numeric regression).
   Reduces combine with `nan_max` / `nan_min` in **both the per-thread scan and the workgroup tree**
   (one stage alone lets the ±F32_MAX identity swallow NaN).
3. Pipeline keys are bumped **per family** (`ew:v1 → v2` / `reduce:v1 → v2`; `ew` is at v3 today for an
   unrelated change, `packages/runtime/src/codegen/elementwise.ts:379`). No per-op exception table (it
   would duplicate "which op is at which version"); unchanged ops pay one in-process cache miss.
4. MUST: a new kernel that uses `max` / `min` / a comparison select on **data values**, for an op where
   torch propagates NaN, guarantees propagation with the same bitwise check.
5. Deliberately out of scope: `leaky_relu` keeps the select form (both branches carry x; pinned by a
   real-GPU test). Out-of-range NaN in gather / embedding (limitations.md) is a separate ruling. The max
   inside safe-softmax stayed plain `max`; current fact: softmax / safe_softmax and attention row
   statistics use `nan_max` since 2026-08-31, because an all-NaN row silently became 0 (ADR 0044,
   addendum of 2026-08-31; `packages/runtime/src/kernels/softmax.ts:27`).

## Verification

- Generated WGSL: `packages/runtime/tests/codegen_wgsl_test.ts` pins "bits, not comparison" for the five
  ops and that the helper does not leak into other ops (snapshots included).
- Real GPU: `packages/runtime/tests/gpu_ops_test.ts` varies the NaN position (first / middle / last;
  for reductions also the second pass of the scan) and asserts the CPU reference's **isNaN pattern**,
  exact non-NaN elements, and control elements (excluding an always-NaN implementation). Fault
  injection (relu or the amax combine to plain max) turns it red.
- NaN **bit patterns** need not match (GPU quiet NaN and CPU NaN bits are not guaranteed equal).

## Consequences

- GPU / CPU-reference propagation agreement holds again for these ops; limitations.md lists the covered
  ops. An upstream NaN no longer turns finite in a clamp-like op.
- Residual risk: the outer `select` could in principle be folded; with an integer condition this is
  unlikely, and the real-GPU test is the standing detector.

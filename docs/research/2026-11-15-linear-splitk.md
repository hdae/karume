> This document is a point-in-time snapshot (measurements of 2026-11-15, RTX 3080 Ti, Node + Dawn / Vulkan) of the
> split-K linear against the reference kernel: the error of each against f64, and wall time. It is not a design source
> of truth: the kernel's contract is in `packages/runtime/src/kernels/linear-splitk.ts` and its checks in
> `packages/runtime/tests/gpu_linear_splitk_test.ts`; whether the reference GEMM's reduction is replaced is an open
> point of ADR [0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md). The script is under
> `outputs/diag/restructure-2026-11-15/` (local, untracked); the numbers below are its output.

# The split-K linear: accuracy and time against the reference kernel

## 1. What was asked

The reference kernel of `linear` adds the K products of one output into a single f32 accumulator in ascending k. Its
rounding error grows with K ([perf-ledger](../perf-ledger.md) K-75). The split-K linear (a `reorder` strategy, admitted
only by a private option) adds the same products slice by slice — 2 to 16 slices of at least 256 terms — and then adds
the partial sums. The questions: how much smaller is its error against f64, and what does it cost.

## 2. Reading

- **Error.** With inputs uniform in [−1, 1), the root-mean-square error against f64 is about half the reference
  kernel's at K = 1536 (4 slices) and about a quarter at K = 8960 (16 slices); the worst error at K = 8960 is about a
  ninth to a seventh.
- **At K = 8960 the reference kernel's worst error is of the size of the op's band on these inputs.** It is 4.3e-4 to
  5.0e-4; the band of the GEMM class is `atol 2e-4 + rtol 4e-6 · |reference|`, about 3.3e-4 for a sum of typical size
  here (|reference| ≈ 32). Whether an element is outside its own band was not evaluated element by element. The band
  was derived at K = 72 and says it has to be re-derived for a K orders of magnitude larger; on the smoother inputs of
  the kernel's own test the reference kernel uses a third of the band. The split-K kernel's worst error is 5.4e-5 to
  6.1e-5.
- **The reference kernel is not the plain f32 sum bit for bit.** Its error statistics equal those of a sequential f32
  sum computed in JS (multiply, round, add, round), but only 7 to 14% of the outputs have the same bits — the sign of a
  fused multiply-add on this driver. This is known behaviour (bit identity between forms is measured, not promised);
  it means "the reference kernel's bits" are those of this kernel on this device, not of an abstract f32 sum.
- **Time.** The split-K kernel reads its operands without a shared tile, two loads per multiply-add. Wall time of one
  `Session.run` (15 runs after 3 warm-ups, read-back included, so a fixed cost of several tenths of a millisecond is
  in every number): equal within noise at K = 8960 with 16,384 outputs, and 1.9 times the reference kernel at K = 1536
  with 262,144 outputs. It is not a speed candidate as written.
- **Not measured:** real activations and weights (the inputs here are synthetic), other GPUs and backends, f16 / bf16 /
  integer storage (the kernel takes f32 storage only), the effect on a model's output.

## 3. Measurement

Inputs: `x`, `w` and `bias` from one linear congruential generator, uniform in [−1, 1). "JS sequential" is the f32 sum
in ascending k computed on the host.

| m × n, K | Kernel | Worst error against f64 | RMS error against f64 | Same bits as JS sequential |
| --- | --- | --- | --- | --- |
| 16 × 256, 1536 | JS sequential | 5.97e-5 | 9.26e-6 | — |
| | reference kernel | 5.58e-5 | 9.18e-6 | 552 / 4,096 |
| | split-K (4 slices) | 2.33e-5 | 4.57e-6 | 182 / 4,096 |
| 16 × 256, 8960 | JS sequential | 4.98e-4 | 5.29e-5 | — |
| | reference kernel | 4.98e-4 | 5.28e-5 | 308 / 4,096 |
| | split-K (16 slices) | 5.41e-5 | 1.39e-5 | 61 / 4,096 |
| 64 × 256, 8960 | JS sequential | 4.07e-4 | 5.31e-5 | — |
| | reference kernel | 4.30e-4 | 5.31e-5 | 1,231 / 16,384 |
| | split-K (16 slices) | 6.08e-5 | 1.37e-5 | 300 / 16,384 |
| 256 × 1024, 1536 | JS sequential | 9.91e-5 | 9.19e-6 | — |
| | reference kernel | 9.91e-5 | 9.18e-6 | 31,102 / 262,144 |
| | split-K (4 slices) | 3.62e-5 | 4.63e-6 | 11,317 / 262,144 |

| m × n, K | Kernel | Median wall of a run | Fastest run |
| --- | --- | --- | --- |
| 64 × 256, 8960 | reference kernel | 1.26 ms | 0.94 ms |
| | split-K | 1.15 ms | 0.84 ms |
| 256 × 1024, 1536 | reference kernel | 1.06 ms | 0.65 ms |
| | split-K | 2.02 ms | 1.69 ms |

The kernel's own test (smooth sine and cosine inputs) logs, for the worst error against f64: 1.92e-5 against the
reference kernel's 1.77e-5 at K = 512 (2 slices), 2.74e-5 against 4.11e-5 at K = 1536, 5.88e-5 against 7.77e-5 at
K = 8960 — between 10% and 34% of the band for both kernels.

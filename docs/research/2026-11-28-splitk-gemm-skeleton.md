> This document is a point-in-time snapshot (measurements of 2026-11-27 and 2026-11-28, one RTX 3080 Ti, Node + Dawn /
> Vulkan) of the split-K linear rebuilt as a split of the GEMM skeleton, against the reference GEMM: kernel time and
> error. It is not a design source of truth: what runs is [spec/numerics.md](../spec/numerics.md#strategies-admitted-only-by-a-private-key),
> the kernel's contract is in `packages/runtime/src/kernels/linear-splitk.ts` and its checks in
> `packages/runtime/tests/gpu_linear_splitk_test.ts`. The scripts are under `outputs/misc/splitk-recon-2026-11-28/`
> (the design recon, `bench*.ts`) and its `impl/` folder (this note, `bench-split.ts`), local and untracked; the
> numbers below are their output.

# The split-K linear as a split of the GEMM skeleton

## 1. What was asked

The split-K linear (a `reorder` strategy of `linear`, admitted only by the private Session key `LINEAR_SPLIT_K`) was
built on 2026-11-15 as an accuracy measure ([research 2026-11-15](2026-11-15-linear-splitk.md)): one thread per output,
two global loads per multiply-add, no shared tile. It was 4 to 16 times slower than the reference GEMM on every
shipped shape class. The recon of 2026-11-27 measured a prototype that reuses the GEMM skeleton instead — the same
kernel with its K tile loop cut to one slice's run and its epilogue writing partial sums without the bias — and found it
faster than the reference where the reference's grid is under-filled. This note records the built form of that
design: the split of the GEMM skeleton as a codegen option (`splitSlices` in `packages/runtime/src/kernels/gemm.ts`),
with the existing combine pass.

## 2. Reading

- **The built kernel times as the prototype did.** On the shipped shapes the ratios to the reference GEMM agree with
  the recon's prototype within its run-to-run spread (depth-anything fc2 0.463 against 0.456; siglip2 base fc2 0.435
  against 0.434; lucida 0.597 against 0.598). The so400m fc2 (K = 4304, 269 K tiles, a prime) now runs: its slices are
  dealt out in whole tiles, one tile apart.
- **Where the reference GEMM's grid is under-filled, the split is faster: ×1.15 to ×4.1.** A 128 × 128 tile grid of 12
  to 96 workgroups (of 256 threads) on 80 SMs walks every K tile serially in each workgroup. Cutting K into slices
  multiplies the workgroups; at Gemma 4 prefill's 768-row chunk the narrow projections take 0.24 to 0.29 (k / v), 0.73
  (q) and 0.76 to 0.84 (down) of the reference's time at their best slice count.
- **Where the grid is already full, the split is slower: up to ×1.33.** Gemma 4's gate / up projections (288 to 2,560
  workgroups) and the PLE projection at 256 rows gain nothing at any slice count. The private option's slice rule
  (the largest of 16 / 8 / 4 / 2 that divides K, at least 256 terms each) does not look at the grid, so on those shapes
  it picks a losing count (gate / up at M = 768: 1.05 to 1.14). A selection by level needs a rule that reads the grid's
  thread count against a per-adapter target — not built.
- **Bits.** With one slice the two dispatches give the reference GEMM's words on every tile geometry the linear GEMM
  takes (the default profile's and every embedded profile's row-count rules, scalar and vec4). On the kernel's own test
  inputs its error against f64 is the naive kernel's to the printed digit (the recon found the prototype's words equal
  to the naive kernel's on every shape it ran — an observation of this driver's FMA contraction, not a contract).

## 3. Method

- Kernel time: GPU timestamps around one compute pass of `reps` repeated dispatches (calibrated to about 6 ms on the
  reference arm). One round runs every arm, then every arm in reverse order; a round's ratio is the arm's mean over
  the reference's mean in that round; the median of 5 rounds is reported. Only ratios within one round are compared
  (the GPU is shared and its clocks drift).
- Arms: the reference GEMM with the `nvidia-ampere` profile's geometry for the row count (what a Session on this
  device runs), and the split at 2, 4, 8 and 16 slices with the same geometry, partial pass plus combine pass.
- Inputs: activations and f32 weights uniform in [−1, 1). The weights stay warm in the 6 MB L2 across repeated
  dispatches of small shapes, so these are kernel ratios, not end-to-end ones.
- Device: RTX 3080 Ti (80 SMs), Dawn / Vulkan, the development machine; `gpu-busy` read 0% before each run.

## 4. Kernel time, f32 storage

Shipped shapes (the recon's census of f32-stored linears; `rule` is the private option's slice count):

| Shape | M | N | K | Tiles | Reference geometry | Workgroups | Reference µs | s2 | s4 | s8 | s16 | rule | rule/ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth-anything mlp.fc2 | 1370 | 384 | 1536 | 96 | 128 × 128 | 33 | 379.6 | 0.547 | 0.463 | 0.498 | 0.569 | 4 | 0.463 |
| siglip2-base mlp.fc2 | 196 | 768 | 3072 | 192 | 64 × 64 | 48 | 257.7 | 0.656 | 0.508 | 0.435 | 0.456 | 8 | 0.435 |
| siglip2-so400m mlp.fc2 | 729 | 1152 | 4304 | 269 | 128 × 128 | 54 | 1023.1 | 0.818 | 0.652 | 0.638 | 0.642 | 16 | 0.642 |
| siglip2-so400m attn | 729 | 1152 | 1152 | 72 | 128 × 128 | 54 | 280.7 | 0.846 | 0.725 | 0.800 | 0.936 | 4 | 0.725 |
| lucida 1024 fc2 (stage) | 1024 | 768 | 3072 | 192 | 128 × 128 | 48 | 735.6 | 0.820 | 0.662 | 0.597 | 0.664 | 8 | 0.597 |
| lucida 1024 fc2 (deep) | 256 | 1536 | 6144 | 384 | 64 × 64 | 96 | 752.0 | 0.688 | 0.565 | 0.555 | 0.568 | 16 | 0.568 |
| gemma4-qat-mobile E2B PLE proj | 32 | 8960 | 1536 | 96 | 16 × 16 | 1120 | 266.3 | 0.974 | 0.971 | 0.990 | 1.043 | 4 | 0.971 |
| gemma4-qat-mobile E2B PLE proj | 256 | 8960 | 1536 | 96 | 64 × 64 | 560 | 650.4 | 1.024 | 1.054 | 1.122 | 1.265 | 4 | 1.054 |
| gemma4-qat-mobile E4B PLE proj | 8 | 10752 | 2560 | 160 | 16 × 16 | 672 | 284.4 | 1.004 | 0.920 | 0.881 | 0.894 | 8 | 0.881 |

None of the families of the first six rows accepts `numerics` ([spec/numerics.md](../spec/numerics.md#presets-and-explicit-options)),
so they cannot select a `reorder` strategy; the recon measured an exact smaller tile at 0.53 to 0.81 of the reference
on them (a geometry question, outside this note).

Gemma 4 prefill shapes, f32 storage (bf16 and int4 in the next table):

| Shape | M | N | K | Workgroups | Reference µs | s2 | s4 | s8 | s16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2B k / v | 64 | 256 | 1536 | 64 | 72.4 | 0.585 | 0.382 | 0.350 | 0.336 |
| E2B q | 64 | 2048 | 1536 | 512 | 139.5 | 0.889 | 0.853 | 0.863 | 0.925 |
| E2B o | 64 | 1536 | 2048 | 384 | 131.5 | 0.974 | 0.940 | 0.939 | 0.959 |
| E2B gate / up | 64 | 6144 | 1536 | 1536 | 360.4 | 0.988 | 0.966 | 0.976 | 1.031 |
| E2B down | 64 | 1536 | 6144 | 384 | 396.3 | 0.962 | 0.913 | 0.891 | 0.873 |
| E4B k / v | 64 | 512 | 2560 | 128 | 122.6 | 0.634 | 0.488 | 0.446 | 0.438 |
| E4B o | 64 | 2560 | 2048 | 640 | 205.6 | 0.945 | 0.935 | 0.945 | 0.990 |
| E4B gate / up | 64 | 10240 | 2560 | 2560 | 992.3 | 0.987 | 0.987 | 1.003 | 1.029 |
| E4B down | 64 | 2560 | 10240 | 640 | 1051.8 | 0.966 | 0.932 | 0.915 | 0.912 |
| E2B k / v | 768 | 256 | 1536 | 12 | 362.7 | 0.519 | 0.289 | 0.243 | 0.272 |
| E2B q | 768 | 2048 | 1536 | 96 | 589.1 | 0.830 | 0.731 | 0.808 | 0.968 |
| E2B o | 768 | 1536 | 2048 | 72 | 483.4 | 0.858 | 0.879 | 0.974 | 1.073 |
| E2B gate / up | 768 | 6144 | 1536 | 288 | 1224.5 | 1.054 | 1.054 | 1.135 | 1.326 |
| E2B down | 768 | 1536 | 6144 | 72 | 1455.1 | 0.831 | 0.821 | 0.844 | 0.839 |
| E4B k / v | 768 | 512 | 2560 | 24 | 603.4 | 0.523 | 0.429 | 0.375 | 0.377 |
| E4B o | 768 | 2560 | 2048 | 120 | 808.7 | 0.827 | 0.849 | 0.928 | 1.075 |
| E4B gate / up | 768 | 10240 | 2560 | 480 | 3215.4 | 1.036 | 1.068 | 1.138 | 1.267 |
| E4B down | 768 | 2560 | 10240 | 120 | 4070.9 | 0.785 | 0.756 | 0.758 | 0.779 |

At M = 64 the reference runs the 16 × 16 tile of 64 threads (`nvidia-ampere`'s rule for up to 64 rows), so its grid has
more workgroups but each is small; the split still pays on the narrow k / v projections (0.34 to 0.44).

The same shapes with the weights the Gemma 4 presets store, bf16 and int4 at group 32, at M = 768 (the split reads them
with the reference GEMM's decode at its tile fill):

| Shape | N | K | Workgroups | bf16: ref µs | bf16: s2 / s4 / s8 / s16 | int4 g32: ref µs | int4 g32: s2 / s4 / s8 / s16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2B k / v | 256 | 1536 | 12 | 338.9 | 0.539 / 0.298 / 0.263 / 0.284 | 328.9 | 0.549 / 0.299 / 0.272 / 0.293 |
| E2B q | 2048 | 1536 | 96 | 598.4 | 0.827 / 0.746 / 0.824 / 0.931 | 610.2 | 0.824 / 0.737 / 0.810 / 0.929 |
| E2B o | 1536 | 2048 | 72 | 484.4 | 0.889 / 0.928 / 1.007 / 1.102 | 470.0 | 0.902 / 0.939 / 1.021 / 1.112 |
| E2B gate / up | 6144 | 1536 | 288 | 1244.4 | 1.056 / 1.051 / 1.146 / 1.310 | 1237.2 | 1.060 / 1.066 / 1.180 / 1.330 |
| E2B down | 1536 | 6144 | 72 | 1425.7 | 0.861 / 0.863 / 0.880 / 0.879 | 1404.4 | 0.872 / 0.881 / 0.901 / 0.889 |
| E4B k / v | 512 | 2560 | 24 | 590.6 | 0.519 / 0.450 / 0.384 / 0.390 | 580.3 | 0.524 / 0.451 / 0.388 / 0.397 |
| E4B o | 2560 | 2048 | 120 | 811.3 | 0.819 / 0.847 / 0.917 / 1.073 | 815.2 | 0.830 / 0.861 / 0.915 / 1.057 |
| E4B gate / up | 10240 | 2560 | 480 | 3130.9 | 1.038 / 1.073 / 1.141 / 1.272 | 3097.1 | 1.040 / 1.078 / 1.142 / 1.289 |
| E4B down | 2560 | 10240 | 120 | 4251.1 | 0.774 / 0.754 / 0.758 / 0.782 | 4111.4 | 0.782 / 0.761 / 0.769 / 0.797 |

At M = 64 (not tabulated; `outputs/misc/splitk-recon-2026-11-28/impl/gemma-bf16-i4.tsv`) the narrow k / v projections take
0.41 to 0.58 (bf16) and 0.35 to 0.49 (int4) at the best slice count, against 0.34 to 0.44 with f32; the other shapes
stay within 0.04 of the f32 ratios. An int4 linear of 64 rows or fewer does not run the GEMM in a
Session: the GEMV family takes it before the split is considered, so those int4 rows compare two kernels a Session
would not choose between.

## 5. Error

The kernel's own test inputs (`irregular` of the GPU test: `sin(0.037 i) · 0.75` activations, `cos(0.011 i + 0.3) · 0.5`
weights) through a Session on this device, the worst over each shape's outputs, as a share of the op band
(`GEMM_TOLERANCE`: atol 2e-4 + rtol 4e-6 · |f64|):

| m × k × n, slices | Split vs f64 | Reference vs f64 | Split vs reference |
| --- | --- | --- | --- |
| 3 × 512 × 5, 2 | 1.92e-5 (9.5%) | 1.77e-5 (8.8%) | 5.96e-6 (2.8%) |
| 16 × 1536 × 64, 4 | 2.74e-5 (12.1%) | 4.11e-5 (15.9%) | 3.34e-5 (14.4%) |
| 9 × 8960 × 37, 16 | 5.88e-5 (28.6%) | 7.77e-5 (33.5%) | 9.25e-5 (39.3%) |

Every storage, through a Session at 128 × 1536 × 64 (4 slices; the weights the `irregular` fixture decodes to, about
±0.5): against f64 10.6% (int4 g32) to 17.5% (bf16) of the band, against the reference kernel 12.2% (int4 g16) to 19.2%
(int2). By raw dispatches at 37 × 4096 × 68 and 37 × 4300 × 69 (int4: 37 × 4224 × 69), 2 to 16 slices, two geometries
and eight storages: at worst 32.4% of the band against f64 and 39.0% against the reference GEMM (bf16, K = 4096, 4
slices).

The A/B band of the test is declared at 2 × the op band (ADR 0147 decision 2: at least 4 × the worst measured, 0.393 ×
4 = 1.57, and at most 2 × the f64 band).

## 6. Not measured

- Other devices. With fewer SMs a grid fills earlier and a split stops paying earlier; WebGPU does not report the SM
  count, so a selection needs a measured thread target per adapter class.
- End to end. No Session A/B of a model was run: the kernel ratios come from repeated dispatches with warm caches, and
  the end-to-end share of the narrow projections in a Gemma 4 prefill chunk is the recon's inference (about 8% of a
  768-row chunk), not a measurement.
- The browser backends (D3D12, Metal).
- The combine pass alone on these shapes (the recon measured 1.4 to 40 µs where a split is wanted).

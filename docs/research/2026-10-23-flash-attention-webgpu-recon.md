> This document is a point-in-time snapshot (2026-10-23) of a feasibility investigation and two measurements, not a design
> source of truth. The measurements are from one machine (RTX 3080 Ti, Node + Dawn on Vulkan) and one run each; the
> web findings are dated where the date matters. Decisions taken from it live in the perf-ledger (K-82).

# FlashAttention-2 in WGSL — feasibility for the Wan2.2 DiT

Question: can a FlashAttention-2-style fused attention (online softmax; the score matrix S is never written to memory) be
built in WGSL, and what would it buy on consumer NVIDIA GPUs for the Wan2.2 TI2V-5B DiT at 1280×704×121 (S = 27,280,
D = 128, 24 heads, 30 layers, CFG = 2 forwards per step)? The maintainer asked for the angle of **subgroups** and
**`maxComputeWorkgroupStorageSize`**. Starting point: `.claude/reviews/2026-10-11_wan-attention-recon/NOTES.md` (untracked);
the web sweep (four investigation legs, 2026-10-23) and the probe scripts are kept next to it.

## 1. Summary

- **Feasible, with a bounded payoff without matrix units.** Measured on the 3080 Ti (§4.2), one DiT forward at S = 27,280 is
  24.2 s of GPU time; the three attention kernels are 64% of it, and ② row statistics alone 16.4% (4.0 s). ①QK and ③PV are
  compute-bound on dp4a (①: 32 TOPS = 50% of the measured 65 TOPS ceiling; ③: 19 TOPS, limited by building P̃ in the tile
  fill), so the S write / read they carry mostly hides behind arithmetic. What an online kernel removes is ② and the S
  round trip; what it keeps is the integer dot products and the per-element `exp` + quantisation of ③. Upper bound:
  **×1.34 on the attention kernels, 16% of the forward** — on the kill line of K-5b (×1.3) before the online kernel's own
  costs (rescale per key tile, K / V re-streamed N / Br times) are paid.
- **The matrix units are the real lever, and they are not reachable from stable WebGPU.** Dawn exposes
  `chromium-experimental-subgroup-matrix` (i8→i32 16×16×32 and f16→f32 16×16×16 on the 3080 Ti) only behind the
  `allow_unsafe_apis` toggle, Chrome only behind `--enable-unsafe-webgpu`, the gpuweb proposal is Draft, Firefox and WebKit
  have nothing (§5). It would attack ①③'s arithmetic (16 of the 24 s), which no plain-WGSL kernel can.
- **Subgroups are optional on NVIDIA.** The public WGSL flash kernels that run on NVIDIA avoid `subgroupShuffle` (bank
  conflicts; onnxruntime reverted it) and keep one row per thread, or use `subgroupMax` / `subgroupAdd` only (§6).
- **Workgroup storage decides the key-tile width, not the feasibility.** 48 KiB (Dawn / Vulkan on NVIDIA) fits
  Bc = 128 int8 key / value tiles, 32 KiB (Chrome on D3D12, Metal) Bc = 64, the core default 16 KiB Bc = 32 (§3). A kernel
  must be keyed by the granted limit; the geometry-profile mechanism (per-adapter static tables) already carries that.

## 2. Device facts (probed 2026-10-23)

| | RTX 3080 Ti, Node + Dawn (Vulkan, driver 615.78) | RTX 5070 Ti, Chrome 155 on Windows (D3D12) | WebGPU core default |
| --- | --- | --- | --- |
| `maxComputeWorkgroupStorageSize` | 49,152 | 32,768 (bench JSON of 2026-10-10) | 16,384 |
| subgroup size (min / max) | 32 / 32 | not recorded | — |
| features | `subgroups`, `subgroup-size-control`, `shader-f16`, `timestamp-query`; WGSL `packed_4x8_integer_dot_product`, `subgroup_id`, `subgroup_uniformity` | subgroups since Chrome 134, size control since 151 | — |
| subgroup matrix | with `allow_unsafe_apis`: f16→f16 / f32 at 16×16×16, 16×8×16, 16×8×8; i8→i32 and u8→u32 at 16×16×32, 16×8×32. A device with the feature was created; no kernel was run | D3D12 needs SM 6.10 + Agility SDK; i8 disabled there | proposal Draft |

Dawn's D3D12 backend hardcodes 32,768 (the D3D thread-group shared-memory limit), Metal reports 32,768 for Apple4+, Vulkan
passes the driver's `maxComputeSharedMemorySize` through (NVIDIA 49,152). Chrome rounds adapter limits to tiers
(16 / 32 / 48 / 64 KiB); a device gets a value above the default only when `requiredLimits` asks for it.

## 3. Arithmetic (S = 27,280, D = 128, H = 24, 30 layers, 2 forwards per step)

| | per step |
| --- | --: |
| attention FLOP (QK + PV) | 549 T |
| S bytes per layer (f16 pairs) | 35.7 GB |
| S traffic of the three-pass design (write, two stats reads, one PV read) | 8.57 TB → 9.4 s at 912 GB/s |
| of which ② alone (two reads) | 4.7 s |
| compute time at 30 / 65 TOPS | 18.3 / 8.4 s |

①QK writes 2 B of S per 2·D = 256 FLOP → 128 FLOP/B, bandwidth-bound only above 117 TFLOP/s on 912 GB/s; ③PV reads the
same amount. ② is pure traffic. An online kernel also halves the `exp` count (2 → 1 per S element) but re-streams K / V
N / Br times (71 GB per layer at Br = 64 before L2 reuse) and rescales the running output per key tile.

Workgroup storage of candidate tiles (D = 128):

| Br × Bc | K + V int8 | + Q int8 | K + V f16 | S tile f32 | O accumulator f32 per thread (256 threads) |
| --- | --: | --: | --: | --: | --: |
| 64 × 64 | 16 KiB | 24 KiB | 32 KiB | 16 KiB | 128 B (32 registers) |
| 64 × 128 | 32 KiB | 40 KiB | 64 KiB | 32 KiB | 128 B |
| 128 × 64 | 16 KiB | 32 KiB | 32 KiB | 32 KiB | 256 B |

WGSL has no register control; a dynamically indexed accumulator array spills to local memory (the same reason the i8a8
skeleton spells `acc{row}_{quad}` as named variables). FlashAttention-2 itself notes that past a certain block size
register spilling causes a significant slowdown.

## 4. Measurements (RTX 3080 Ti, Node + Dawn, 2026-10-23)

### 4.1 dp4a and f32 fma issue rate (standalone microbench)

256 threads × 1,280 workgroups, 4 independent chains × 8 unrolled instructions × 4,096 iterations, timestamp queries,
7 repetitions, two rounds (the slow-kernel episodes of 2026-10-18 did not occur; all 7 repetitions agree to 0.1 ms):

| kernel | GPU time | rate | instructions / clk / SM at 1,665 MHz |
| --- | --: | --: | --: |
| `fma` f32 | 2.9 ms | 29.5 TFLOP/s | 110 (datapath 128) |
| `dot4I8Packed` | 5.3 ms | 65 TOPS (8 ops per instruction) | 61 (datapath 64) |

dp4a issues at 64 per clock per SM on GA102 — the INT32 datapath, half the FP32 width — so the **int8 MAC ceiling is about
65 TOPS**, twice the f32 fma rate in operations. NVIDIA's instruction throughput table has no dp4a row; this is the measured
value.

### 4.2 Per-key GPU time of one DiT forward (practical preset `f16+dit8-a8-attn8-s16`, S = 27,280)

Taken with the opt-in measured-mode case of `packages/e2e/tests/e2e_wan_ti2v_dit_test.ts` (`KARUME_WAN_TI2V_121F=1`, filter
`27280`) and a temporary, uncommitted dump of `SessionDiagnostics.lastRunTiming` after the second run (the run whose
chunks are the backed ones of generation). GPU time 24.24 s, wall 24.68 s, 4,387 dispatches, 0 clamped samples
(the first run: 23.74 s GPU).

| group | ms | share | dispatches |
| --- | --: | --: | --: |
| attention ①QK (`attention_qk:v3:i8a8:tile128x64…:dp4a:s16:rwa`) | 4,318 | 17.8% | 540 |
| attention ② stats (`attention_stats:v2:f32:lastdim:safe:wg256:s16`) | 3,981 | 16.4% | 540 |
| attention ③PV (`attention_pv:v3:i8a8:tile64x128…:dp4a:s16:rwc`) | 7,260 | 30.0% | 540 |
| linear (`linear:v4:i8a8:tile128x64…:dp4a`) | 6,821 | 28.1% | 310 |
| quantize_rows (activations) | 349 | 1.4% | 490 |
| strided / permute | 463 | 1.9% | 904 |
| norm / RoPE / elementwise / other | 1,044 | 4.3% | 1,063 |

Derived (self-attention only; cross-attention with 512 keys is a small part of the same keys):

| | FLOP per forward | time | achieved | S traffic |
| --- | --: | --: | --: | --: |
| ①QK | 137 T | 4.24 s | 32 TOPS = 50% of 65 | 1.07 TB written → 253 GB/s |
| ③PV | 137 T | 7.16 s | 19 TOPS = 29% of 65 | 1.07 TB read → 150 GB/s |
| ② stats | — | 3.98 s | — | 2.14 TB read → 538 GB/s (59% of 912) |

Reading: ② runs at memory speed, as expected. ① is at half the dp4a ceiling. ③ is at under a third: its A-tile fill
unpacks s16, evaluates `exp(S − m)`, scales, rounds and packs each P̃ element before the dot product, and that per-element
work is the bound, not the dot products. An online kernel keeps exactly that work (one `exp` + quantisation per S element) and
removes ② and the S round trip; so the attainable gain is the ② share plus the fraction of ①③ that is traffic, not more.

## 5. Subgroup matrix (cooperative matrix) status

- Spec: gpuweb issue #4195 is open (Milestone 2); `proposals/subgroup-matrix.md` is Draft (the proposals README says Draft
  proposals must not be exposed by implementations). Recent changes: load / store array requirements relaxed (#6345,
  2026-08), a stride restriction (#6365, 2026-08-12), D3D12 support and `minSubgroupSize` / `maxSubgroupSize` in the config
  (#10964, open since 2026-09-22), FP8 operands proposed (#11051, 2026-10).
- WGSL surface: `subgroup_matrix_left<T, K, M>`, `subgroup_matrix_right<T, N, K>`, `subgroup_matrix_result<T, N, M>`
  with T in f32 / f16 / u32 / i32 / u8 / i8; `subgroupMatrixLoad` / `Store` / `Multiply` / `MultiplyAccumulate` and scalar
  add / subtract / multiply. **No per-element access, no conversion, no transpose, no reduction** (listed as future work).
  Loads and stores only from the storage and workgroup address spaces; subgroup-uniform control flow; `workgroup_size.x` a
  multiple of `subgroupMaxSize`. Dawn's surface differs from the draft (`enable chromium_experimental_subgroup_matrix`,
  `subgroupMatrixLoad<T>(p, offset, col_major, stride)` with the stride in component elements; the draft has
  `enable subgroup_matrix`, a `Majorness` template parameter and the stride in array elements) — shaders written today
  would be rewritten.
- Dawn / Chrome: the feature is `FeatureState::Experimental` (`allow_unsafe_apis` required); Chrome exposes it only at
  `SafetyLevel::kUnsafe` (`--enable-unsafe-webgpu`). Vulkan needs `VK_KHR_cooperative_matrix`, the Vulkan memory model,
  `subgroupSizeControl` + `computeFullSubgroups`; Metal has f32 / f16 8×8×8 on Apple7+; D3D12 needs SM 6.10 in Agility SDK
  builds and skips i8 / u8. No blink-dev intent or shipping timeline was found; a May 2026 paper calls it unavailable in
  stable browsers.
- Firefox (wgpu) has only a native `EXPERIMENTAL_COOPERATIVE_MATRIX` with a different syntax and 8×8 f32; WebKit has only
  the shared header's struct definitions.
- Consequence for attention: with no element access, softmax goes S → workgroup memory → ordinary threads (max, `exp`,
  sum) → P written back (f16 / i8) → reloaded as a left operand; llama.cpp's `flash_attn.wgsl` does exactly this and rounds
  O through workgroup memory as well. The S / P tile competes with K / V for the 32 / 48 KiB. In Dawn an i8 matrix loads
  from an `array<i32>` with byte offsets; whether the runtime's packed-int8 layout (4 × i8 per u32, dp4a order) reads as-is
  is untested.
- Performance: no published NVIDIA numbers with and without subgroup matrix (two papers use it, neither isolates it). Apple
  M2 via Dawn / Metal: 3.2 TFLOP/s raw MMA peak, 2.0–2.45 from workgroup memory, about 1.9 for a GEMM; llama.cpp on M4 Pro:
  the tile kernel without matrices is about half of the subgroup-matrix kernel.

## 6. Public WGSL flash kernels (prefill, S never materialised)

| implementation | structure | storage | subgroups | measured |
| --- | --- | --- | --- | --- |
| onnxruntime WebGPU EP `flash_attention.wgsl.template` | 64 threads (128 on Apple), one Q row per thread, Q / O / scores in `var<private>`, softmax serial per thread | K / V tiles only; Bc from a fixed 16 KiB budget (f16 hd128: Bc = 32), although the device requests the adapter's limit | NVIDIA, Apple and no-subgroup devices take the shared-memory path; `subgroupShuffle` was dropped on NVIDIA (lanes 16–31 hit one bank; PR #25777, ~10% faster without) | RTX 5060 Ti: Bc 16 → 32 gave +10–15% prefill; RTX 5080 D3D12: loop unrolling +11–47% (open PR #33395; Vulkan flat); no kernel-only comparison with a materialised-S kernel on NVIDIA |
| llama.cpp ggml-webgpu `flash_attn.wgsl` / `flash_attn_tile.wgsl` | subgroup-matrix version: Q tile = matrix M, KV tile sized to the device's storage limit, S / P and O in workgroup f16; tile version: 4 Q rows, one subgroup per row, KV ≤ 64, P in workgroup f16, O in f32 registers | requests the adapter's limit; `allow_unsafe_apis` for the matrix path | `subgroupMax` / `subgroupAdd` per row | Apple M4 Pro only (prefill-like shape 152.9 → 64.1 ms with software pipelining; tile ≈ ½ of matrix) |
| TVM / web-llm | 32 × 32 tiles, 128 threads, Q / K / V / S in shared, O in registers, serial softmax per row with `exp2` | requires 32 KiB | none | none found |
| kevala `attention_tile.wgsl` | register-tiled FA2, 256 threads, 64 q × 32 k, hd64, ~25 KiB | 25 KiB | 16 lanes per row via `subgroupShuffleXor` | Apple M4 Max only, vs its own earlier kernel |

No public WGSL kernel runs the online softmax on int8 / dp4a; int8 P̃ with a fixed 1/127 scale inside an online loop has
no prior art in WGSL.

## 7. Algorithm notes for an int8 online kernel

- FlashAttention-2 splits Q rows across warps, K / V are shared by all warps, no cross-warp communication in the forward,
  O is kept unscaled and divided by l once at the end. Its speed-up came with tensor cores (A100: 312 TFLOP/s matmul vs
  19.5 non-matmul); no published ablation separates the tensor-core gain from the S-round-trip gain. Third-party ports to
  GPUs without tensor cores lost against the materialised form (P100: 0.45–0.78×; llama.cpp's FP32 vector kernel on P40 lost
  at large batch).
- SageAttention 1 / 2 quantise Q and K per block (smooth-K), keep P̃·V in fp16 / fp8 with fp32 accumulation; SageBwd
  (SageAttention3 paper, Algorithm 2) runs P̃·V in int8 with a per-token P̃ scale and a per-block V scale, folds the integer
  block result into an fp32 O rescaled by `exp(m_old − m_new)`. So "integer dot products inside the block, fp32 running
  output" has a precedent. The same paper states that a static 1/127 scale for P̃ is inaccurate (the current a8 ③PV uses
  that scale; recorded as an adjacent observation, not acted on).
- FlashAttention-3 / 4: at D = 128 the exponential can cost half the matmul cycles on H100 (SFU 16 ops / clk / SM); FA4
  uses a polynomial `exp2` on the FMA units and a conditional rescale. On the 3080 Ti the 536 G `exp` of one forward are
  0.25 s per pass at SFU rate — small next to ③'s 7.2 s, so the P̃ construction cost is the unpack / scale / round / pack
  around it, not the `exp` itself.

## 8. What this does not settle

- Whether an online a8 kernel reaches today's ① and ③ efficiency while re-streaming K / V and rescaling (only a spike
  measures it; the kill line of K-5b, ×1.3 on the attention kernels, is barely inside the upper bound).
- The subgroup-matrix path on the 3080 Ti: no kernel has run; the i8 matrix load layout against the runtime's packing and
  the attainable TOPS are unknown.
- Per-vendor subgroup sizes beyond NVIDIA (Intel 8 / 16 / 32, AMD RDNA 32 / 64, Apple 32, Adreno unverified) only matter
  once a kernel depends on subgroups; Firefox and Safari ship no subgroups at all.
- The 5070 Ti (Chrome, 32 KiB) was not re-measured; the 36 s / step there is the 2026-10-10 bench.

## 9. Sources

- gpuweb: issue #4195, `proposals/subgroup-matrix.md` (Draft), PRs #6345 / #6365 / #10964 / #11051; WGSL `index.bs`
  §Subgroup built-in functions; spec `maxComputeWorkgroupStorageSize` default 16384.
- Dawn: `docs/dawn/features/subgroup_matrix.md`, `src/dawn/native/Features.cpp` (Experimental), `PhysicalDeviceVk.cpp`,
  `PhysicalDeviceD3D12.cpp` (32,768; SM 6.10), `PhysicalDeviceMTL.mm`, `Limits.cpp` (tiers), `docs/tint/extensions/subgroup_size_control.md`;
  Chromium `webgpu_decoder_impl.cc` (`SafetyLevel::kUnsafe`).
- Browsers: Chrome blog "New in WebGPU 134" (subgroups) and "151–152" (subgroup size control); blink-dev intent for subgroup
  size control (2026-06); Firefox bug 1955417 and `dom/webgpu/Adapter.cpp`; caniuse `wf-webgpu-subgroups`; wgpu
  `wgpu-hal` adapter limits.
- Kernels: microsoft/onnxruntime `onnxruntime/contrib_ops/webgpu/bert/flash_attention.{wgsl.template,cc}`, PRs #23576 /
  #25777 / #26715 / #28389 / #28511 / #28520 / #33395; ggml-org/llama.cpp `ggml/src/ggml-webgpu/wgsl-shaders/flash_attn*.wgsl`,
  PRs #18610 / #19151 / #22199; apache/tvm `python/tvm/relax/frontend/nn/llm/_kernel_common.py`, `web/src/webgpu.ts`;
  bvolpato/kevala `attention_tile.wgsl`; tracel-ai/burn `burn-cubecl/src/kernel/attention/base.rs`.
- Papers: FlashAttention (arXiv 2205.14135), FlashAttention-2 (2307.08691), FlashAttention-3 (2407.08608), FA4 (Tri Dao's
  blog, 2026), SageAttention (2410.02367), SageAttention2 (2411.10958), SageAttention3 / SageBwd (2505.11594),
  "Llamas on the Web" (2605.20706); NVIDIA CUDA Best Practices Guide table 5, Ampere GA102 and Ada / Blackwell whitepapers,
  Ampere tuning guide (shared memory per SM).
- Local: `.claude/reviews/2026-10-11_wan-attention-recon/` (untracked) — `NOTES.md`, `2026-10-23-fa2-web-sweep-digest.txt`,
  `2026-10-23-fa2-probe-results.txt`, `probe*.mjs`, `arith.mjs`; the microbench and aggregation scripts of §4 are kept there too.

## 10. Spikes (2026-10-23, same machine, after the maintainer's ruling to run both)

Standalone scripts next to the recon notes (`.claude/reviews/2026-10-11_wan-attention-recon/`, untracked); nothing in the
runtime changed. Timestamp queries, medians of 3–7 repetitions; the slow-kernel episodes of 2026-10-18 showed up as single
outliers (for example 449 ms among 347 ms) and are excluded by the median.

### 10.1 Subgroup matrix (`chromium-experimental-subgroup-matrix`, `allow_unsafe_apis`)

- **Canary**: an i8 → i32 16×16×32 GEMM reading the runtime's packed layout directly — A as `subgroup_matrix_left` row-major
  with the K stride, B (stored `[N][K]` like `k` and `Vᵀ`) as `subgroup_matrix_right` column-major with the same stride — is
  **exact against the CPU integer product** (32,768 and 131,072 elements, several tilings). So `qq`, `kq`, `vq` and a P̃
  tile can feed the matrix units without repacking. Tint warns that the Dawn-form `subgroupMatrixLoad` / `Store` are
  deprecated builtins (the surface is moving toward the draft's).
- **i8 GEMM throughput** (M = N = K = 4,096; 256 threads = 8 subgroups per workgroup; fragments loaded straight from
  storage, no workgroup tiles):

| result block per subgroup | workgroup tile | TOPS |
| --- | --- | --: |
| 32 × 32 (2 × 2 tiles) | 128 × 64 | 62.4 |
| 64 × 32 | 256 × 64 | 83.6 |
| 32 × 64 | 64 × 256 | 78.2 |
| **64 × 64** | 256 × 128 | **91.4** |
| 64 × 64 | 128 × 256 | 84.4 |

  Staging the tiles through workgroup memory first was slower (K step 32 / 64 / 128: 49.7 / 39.2 / 22.5 TOPS) — the
  scalar copy loop and the workgroup-memory fragment loads cost more than the L1 / L2 reuse they buy. The naive 64 × 64
  kernel is already **1.4× the dp4a ceiling (65 TOPS) and 2.6× the runtime's current i8a8 linear (34.7 TOPS at
  S = 27,280)**, at a third of the GA102 tensor peak; the remaining gap is fragment-load bound and would need a real
  pipeline (double-buffered tiles, larger blocks) to close.
- **f16 → f32**: 31.3 TFLOP/s at 32 × 32 (no better than the f32 fma rate), 6.4 at 64 × 64 (register spill). The int8 path
  is the one worth having.

### 10.2 Online a8 attention (plain WGSL, dp4a, one dispatch per 64-row query block)

Kernel: 256 threads; Br = 64 query rows, Bc = 64 keys per tile; `q` tile loaded once, `k` and `Vᵀ` tiles per key tile;
S tile by `dot4I8Packed` (thread = 4 rows × 4 keys), row max across the 16 threads of a row, running `m` / `l` / f32 `O`
with the `exp(m_old − m_new)` rescale, P̃ = `round(127·exp(S − m))` packed to a workgroup tile, then P̃·Vᵀ by
`dot4I8Packed` (thread = 4 rows × 8 columns), `vs[d]·(1/127)/l` at the end. Checked against an f64 reference of the
dequantised inputs with an unquantised P: relative RMS 1.4% and max 2.5% of the output range (the int8 P̃ lattice; the
current ③PV quantises P̃ the same way — not compared head-to-head yet).

| version | S = 7,920 (ms / layer, 24 heads) | S = 27,280 |
| --- | --: | --: |
| current three dispatches, self-attention keys only (§4.2 and a second run at 7,920) | 32.7 | 507.7 |
| v1: natural layouts (`k[key][pack]`, `Vᵀ[d][pack]`), row max through workgroup memory | 70.0 (×0.47) | 825 (×0.62) |
| v2: padded `k` rows (33 words), keys interleaved 16 apart per lane, `Vᵀ` as `[pack][d]`, workgroup-memory reduce | 34.0 (×0.96) | 430 (×1.18) |
| **v2 with `subgroupShuffleXor` for the row max and the final `l`** | **28.5–31.4 (×1.04–1.15)** | **348 (×1.46)** |
| v2 subgroup, Br = 128 (8 rows per thread) | 33.9 | 399 |
| v2 subgroup, `Vᵀ` tile copied without the byte gather (timing only — wrong values) | 25.6 (×1.28) | 315 (×1.61) |

Reading:

- The first version lost to the three-pass design by 2×: the `k` and `Vᵀ` tile reads put the 16 lanes of a row on one
  shared-memory bank (16-way conflicts). Fixing the layouts is what made the kernel win; the algorithm itself was never
  the bound.
- Subgroups matter here: the shuffle reduce beats the workgroup-memory reduce by 1.23× at the long shape (the ORT
  finding against `subgroupShuffle` on NVIDIA was about broadcasting K / V, not about reductions).
- The gain is **×1.46 at S = 27,280 (above the K-5b kill line ×1.3) and ×1.04–1.15 at S = 7,920 (below it)**: at the
  short shape the three passes are already close to their compute bound and ② is a smaller share. In the 121-frame
  forward the attention kernels would go from 15.6 s to about 10.7 s (−20% of the forward).
- About 10% more is available by giving `Vᵀ` the interleaved key order on the host (the permute + quantise step already
  rewrites `Vᵀ`; the kernel's byte gather is 4 loads per tile word): ×1.61 / ×1.28. Larger row tiles do not help (register
  pressure); 26–29 TOPS is still under half the dp4a ceiling, so there is room (double-buffered tiles, `exp2`, Bc = 128 on
  48 KiB devices), unmeasured.
- The kernel needs 32.9 KiB of workgroup storage at Bc = 64 (q 8.4, k 8.4, Vᵀ 8, P̃ 4.3, reduce 4; 28.9 KiB with the subgroup
  reduce) — inside the 48 KiB of Dawn / Vulkan and the 32 KiB of D3D12 / Metal only in the subgroup form; the core
  default 16 KiB needs Bc = 32 and no `q` tile.

### 10.3 Not settled by the spikes

- A head-to-head of the online kernel's error against the current a8 path (both quantise P̃; the online one against a
  running max) on real activations — the ADR 0110 four-point check would settle the contract class.
- Whether the subgroup-matrix GEMM reaches its 91 TOPS inside an attention loop (the S → workgroup memory → P̃ → reload
  detour, §5) — not built.
- The 5070 Ti (Chrome, 32 KiB) was not measured for either spike.

## 11. In the runtime (2026-10-23, after productising — ADR 0132)

The first Session-path measurement of the productised kernel was slower than the three dispatches
(20.5 s against 16.3 s of attention per forward at S = 27,280). Two causes, both found by running
the production WGSL in the spike's harness on the same inputs:

| variant (one dispatch, S = 27,280 × 24 heads) | ms / layer |
| --- | --: |
| spike v2 (subgroup reduce) | 352 |
| production, lanes identified by `subgroup_invocation_id` / `subgroup_id` | 407 |
| production, lanes by `local_invocation_index` | 350 |
| production, inlined dot product / branch-free gather (no effect) | 407 / 415 |
| spike, 768-row windows (36 dispatches of 288 workgroups) | 403 |
| spike, 6,528-row windows (5 dispatches) | 357 |

The lane identification costs 16% (the tile addresses no longer derive from the thread index), the
small windows 15% (1.2 waves on 80 SMs). After both corrections, and with the measured-mode device
acquired with subgroups as the Wan family does, each seat in a fresh process after cooling the GPU to
60 °C, the DiT e2e at S = 27,280 gives:

| seat | attention | linear | forward (2nd run) |
| --- | --: | --: | --: |
| `f16+dit8-a8-attn8-s16` (three dispatches) | 18.4 s | 8.9 s | 29.3 s |
| `f16+dit8-a8-attn8-online` (`:sg32`, 180 dispatches) | 12.5 s | 7.3 s | 21.7 s |

The s16 run was throttled (its linear is 8.9 s against 6.8–7.3 s in other runs); against the
morning's cold s16 baseline (attention 15.6 s, forward 24.2 s) the gain is ×1.25 on attention and
−10% on the forward. The 33-frame A/B of the online preset is relRMS 4.23e-2 (s16: 4.4e-2). The
121-frame A/B could not run (the reference preset's step runs out of device memory under Node —
known-issues). The machine's thermal state moves these numbers by 20–30% between runs; compare seats
only within one cold run each.

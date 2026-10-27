> Point-in-time record (2026-10-27) of the Gated DeltaNet (GDN) scan spike on the RTX 3080 Ti under Node + Dawn (Vulkan): four GPU forms of the recurrence, their speed and their error against an f64 oracle. Not a design source of truth (that is [ADR 0135](../decisions/0135-gated-delta-rule-scan.md), proposed). Raw material (spike kernels and harness, fixed inputs, raw JSON, logs, the design note) is under `outputs/diag/gdn-scan-2026-10-27/` of the main checkout (not tracked).

# GDN scan spike: sequential, blocked and chunked forms of the gated delta rule

Question: can the recurrence of a Gated DeltaNet layer (the MOSS-Image text encoder's shape: 32
value heads, 16 key heads, key and value head dim 128) run on WebGPU in f32 without subgroups, fast
enough to be a small part of the layer and accurate enough to match upstream's own f32 functions?
Which kernel form should the op use?

Answer: yes. Every form built meets the proposed acceptance (error no more than twice the larger of
the two upstream f32 functions' errors, against an f64 oracle) on all 10 cases, provided the
sequential forms compute `exp(g)` with an accurate exp. At T = 4096 one layer takes 2.13 ms
(blocked), 2.36 ms (sequential, 19.6 KiB of workgroup storage), 2.56 ms (sequential, default
limits), 3.41 ms (chunked) and 3.56 ms (one thread per column). The best is 9.6 % of the layer's
`in_proj_qkv` projection measured with the product's f16-weight linear. Decode (T = 1, state in
place) takes 4.2 µs per layer. The chunked form, upstream's own prefill algorithm, is slower than the
sequential forms at every T on this GPU.

## 1. Setup

- Machine: RTX 3080 Ti (driver 615.78.08), Node 24.22.1 + Dawn (npm `webgpu` 0.6.2, Vulkan),
  timestamp quantisation off. transformers 5.14.1, torch 2.13.0+cpu.
- Code base: the 0.15 worktree at `55fd3db8` when the spike started. Another session committed
  `e95a4ceb` / `f71b0bea` there during it (states-form decode attention and docs). Nothing under
  `packages/runtime/src/tune`, `gpu`, the GEMM / linear kernels or `packages/test-support` changed,
  which is all the yardstick imports. The spike changed nothing in either repository.
- Device: explicit `requiredLimits` (the adapter's values, plus one run per form with the WebGPU core
  defaults), `pushErrorScope('validation')` around module and pipeline creation, f32 only, no
  subgroups. The adapter grants 49,152 B of workgroup storage and 1,024 invocations per workgroup
  (defaults 16,384 B and 256).
- Shape: one MOSS text-encoder GDN layer, B = 1, Hv = 32 value heads, Hk = 16 key heads, dk = dv =
  128. Value head h reads key head ⌊h/2⌋.
- Inputs (`gen_inputs.py`, seeded): q / k / v = silu(N(0,1)), a / b ~ N(0,1), A_log = log U(0,16)
  (one draw for all cases, seed 20261027), dt_bias = 1, `beta = sigmoid(b)`,
  `g = −exp(A_log)·softplus(a + dt_bias)` (regime `init`) or that × 0.001 (regime `long`, long
  memory). Activation seed 20261027 + T. At T = 4096, g spans [−77, −0.08] (init) and
  [−0.077, −8e-5] (long). Cases: 2 regimes × T ∈ {1, 64, 512, 2048, 4096}. The T = 1 case continues
  the T = 64 case: its initial state is the f32-rounded f64 final state of the T = 64 case.
- Upstream outputs: `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` in f32,
  called as `Qwen3_5GatedDeltaNet` calls them (q / k repeated to 32 heads,
  `use_qk_l2norm_in_kernel=True`), with the final state returned.
- Oracle: the recurrence in f64 on exactly the f32 inputs the kernels consume (normalised q / k, v,
  g, beta, initial state). `l2norm` before and after the head repeat gives identical bits (checked
  for every case), so upstream and the kernels consume the same normalised bits. A second oracle that
  normalises the raw q / k in f64 differs from this one by o rel_l2 ≈ 7e-8 (the l2norm's own f32
  rounding, which is outside the op in the chosen contract). The TS f64 reference (`lib/ref_f64.ts`)
  matches the Python oracle to o rel_l2 4.8e-16 and S rel_l2 3.9e-16 at most over the 10 cases.

## 2. Forms built

All forms take the op contract of ADR 0135 (q / k already l2-normalised, the head broadcast ⌊h/r⌋
and the `dk^-½` scale inside, zero initial state for prefill, a carried initial state for decode).

| form | description | form name in the spike |
| --- | --- | --- |
| (a1) | sequential token loop, one thread per value column (128 state floats in registers) | `seq:s1:…` |
| (a2) | sequential token loop, one value column split over 8 threads (key slices); one workgroup-memory exchange and one barrier per token | `seq:s8:…` |
| (a3) | blocked recurrence (not in the design note): state in registers as in (a2), advanced 8 or 16 tokens at a time with the UT transform of the chunked form, so the exchange happens once per block; a small parallel prep dispatch builds the per-block coefficient matrices | `blk:…` |
| (b) | chunked WY / UT form with 64-token chunks as upstream (prep / state pass / output, 3 dispatches) | `chunk:…` |
| (c1) | decode: (a2) with the state read and overwritten in place (and (a3) in place) | `…:inplace` |

Form names: `seq:s<split>:c<columns per workgroup>:tb<token block>[:u][:kreg][:lazy][:xexp][:inplace]`,
`blk:s<split>:c<columns>:b<8|16>[:inplace]`, `chunk:bv<16|32>`. `u` = token loop unrolled, `kreg` =
the token's k kept in registers, `xexp` = `α = exp(g)` by an explicit range-reduced polynomial (once
per token at staging) instead of WGSL's built-in `exp`.

## 3. Method

- **Timing** (`bench.ts`, after the protocol of `packages/runtime/src/tune/harness.ts`): one process
  per (form, T); warm-up of at least 500 ms of GPU time and 5 runs; **stacked** = 30 rounds of one
  compute pass holding `reps` back-to-back executions of the layer (`reps` chosen so a pass is about
  20 ms; per layer = pass time ÷ reps); **single** = 30 rounds of one execution per pass (includes
  about 4 µs of pass overhead, visible at T = 1); **per kernel** = one pass per dispatch; **24
  layers** = 24 layers with distinct o / state buffers back to back in one pass, at least 20 rounds.
  Median and min are both reported (this GPU has a known slow mode). Two repetitions per (form, T),
  round-robin; the tables show the repetition with the lower median.
- **Shared GPU guard**: the GPU was shared with another session that did not always hold the lock
  (observed: busy 100 %, VRAM up to 9 GB, 88–92 °C with `sw_thermal` throttling). The runner waited
  for busy < 10 % and VRAM < 200 MiB before each process, sampled VRAM every second, and dropped runs
  whose sampled VRAM exceeded the process's own allocations by more than 600 MiB.
- **Correctness** (`correct.ts`): outputs pre-filled with a NaN bit pattern (an unwritten element
  counts as non-finite), one execution, read back, metrics against the oracle. The upstream metrics
  are recomputed in TS and agree with the Python ones.
- **Metrics**: max_abs = max |x − ref|; max_rel = max over value heads of (head max |x − ref|) /
  (head max |ref|); rel_l2 = ‖x − ref‖₂ / ‖ref‖₂, for the output o and the final state S.
- **Acceptance** (proposed in the design note): finite, and max_abs and rel_l2 of o and S each at
  most 2 × the larger of the two upstream f32 errors on the same case.
- **Non-tautology**: five fault injections on (a2), each of which must fail the acceptance: drop the
  decay, drop β, use key head `h % Hk` instead of `⌊h/r⌋`, read o before the state update, skip the
  initial state. The r = 1 layout (q / k repeated to 32 heads) is run against r = 2 and compared bit
  for bit.

## 4. Speed

### 4.1 Per layer

One layer = B=1, Hv=32, Hk=16, dk=dv=128, init regime inputs. Time in us per layer; T=1 runs carry the T=64 final state as S_0 (separate S_0 / S_T buffers unless marked in place).

| form | name | dispatches | T=1 | T=64 | T=512 | T=2048 | T=4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (a1) one thread per column | `seq:s1:c128:tb8:xexp` | 1 | 6.8 / 6.8 | 50 / 50 | 449 / 449 | 1778 / 1777 | 3563 / 3560 |
| (a2) key split 8, default limits | `seq:s8:c32:tb8:u:kreg:xexp` | 1 | 4.7 / 4.7 | 41 / 40 | 331 / 322 | 1296 / 1266 | 2556 / 2528 |
| (a2) key split 8, token block 16 | `seq:s8:c16:tb16:u:kreg:xexp` | 1 | 5.2 / 5.2 | 39 / 38 | 306 / 288 | 1187 / 1144 | 2357 / 2269 |
| (a3) blocked b=8, default limits | `blk:s4:c16:b8` | 2 | 13 / 13 | 41 / 40 | 301 / 288 | 1201 / 1174 | 2368 / 2305 |
| (a3) blocked b=8, 32 columns | `blk:s4:c32:b8` | 2 | 13 / 13 | 38 / 38 | 268 / 255 | 1076 / 1050 | 2131 / 2084 |
| (a3) blocked b=16 | `blk:s4:c32:b16` | 2 | 23 / 22 | 46 / 45 | 308 / 299 | 1174 / 1140 | 2307 / 2255 |
| (b) chunked C=64, default limits | `chunk:bv16` | 3 | 128 / 128 | 144 / 143 | 511 / 497 | 1741 / 1679 | 3461 / 3342 |
| (b) chunked C=64, 32-column state pass | `chunk:bv32` | 3 | 126 / 126 | 143 / 142 | 499 / 485 | 1709 / 1649 | 3411 / 3314 |
| (c1) decode, state in place | `seq:s4:c16:tb2:u:kreg:inplace:xexp` | 1 | 4.2 / 4.2 | - | - | - | - |
| (a3) decode, state in place | `blk:s4:c16:b8:inplace` | 2 | 13 / 13 | - | - | - | - |

### 4.2 Twenty-four layers back to back

| form | T=1 measured (ms) | T=1 24 x layer | T=4096 measured (ms) | T=4096 24 x layer | T=4096 single-dispatch median (us) |
| --- | --- | --- | --- | --- | --- |
| `seq:s1:c128:tb8:xexp` | 0.19 (min 0.19) | 0.16 | 85.94 (min 85.48) | 85.51 | 3567 |
| `seq:s8:c32:tb8:u:kreg:xexp` | 0.13 (min 0.13) | 0.11 | 62.01 (min 61.17) | 61.34 | 2560 |
| `seq:s8:c16:tb16:u:kreg:xexp` | 0.15 (min 0.15) | 0.13 | 57.56 (min 56.56) | 56.57 | 2343 |
| `blk:s4:c16:b8` | 0.35 (min 0.35) | 0.32 | 57.01 (min 56.66) | 56.84 | 2369 |
| `blk:s4:c32:b8` | 0.36 (min 0.36) | 0.32 | 51.39 (min 50.95) | 51.13 | 2127 |
| `blk:s4:c32:b16` | 0.56 (min 0.56) | 0.54 | 55.42 (min 54.99) | 55.38 | 2315 |
| `chunk:bv16` | 3.09 (min 3.08) | 3.07 | 83.67 (min 82.80) | 83.06 | 3453 |
| `chunk:bv32` | 3.04 (min 3.03) | 3.02 | 82.46 (min 81.69) | 81.87 | 3409 |
| `seq:s4:c16:tb2:u:kreg:inplace:xexp` | 0.15 (min 0.15) | 0.10 | - | - | - |
| `blk:s4:c16:b8:inplace` | 0.37 (min 0.37) | 0.32 | - | - | - |

At T = 4096 the measured 24-layer time equals 24 × one layer within 2 %. At T = 1 it does not: the
2 MiB state of one layer stays in L2 when the same layer repeats, while 24 layers' 48 MiB do not.

### 4.3 Per kernel of the multi-dispatch forms

| form | T | kernels (median us) |
| --- | --- | --- |
| `blk:s4:c16:b8` | 1 | blk_prep 11 / blk_state 10 |
| `blk:s4:c16:b8` | 512 | blk_prep 28 / blk_state 277 |
| `blk:s4:c16:b8` | 4096 | blk_prep 278 / blk_state 2094 |
| `blk:s4:c32:b8` | 1 | blk_prep 11 / blk_state 10 |
| `blk:s4:c32:b8` | 512 | blk_prep 28 / blk_state 248 |
| `blk:s4:c32:b8` | 4096 | blk_prep 279 / blk_state 1852 |
| `blk:s4:c32:b16` | 1 | blk_prep 16 / blk_state 14 |
| `blk:s4:c32:b16` | 512 | blk_prep 66 / blk_state 253 |
| `blk:s4:c32:b16` | 4096 | blk_prep 391 / blk_state 1909 |
| `chunk:bv16` | 1 | chunk_prep 59 / chunk_state 27 / chunk_out 54 |
| `chunk:bv16` | 512 | chunk_prep 223 / chunk_state 157 / chunk_out 158 |
| `chunk:bv16` | 4096 | chunk_prep 1358 / chunk_state 1139 / chunk_out 957 |
| `chunk:bv32` | 1 | chunk_prep 59 / chunk_state 25 / chunk_out 54 |
| `chunk:bv32` | 512 | chunk_prep 218 / chunk_state 143 / chunk_out 155 |
| `chunk:bv32` | 4096 | chunk_prep 1369 / chunk_state 1092 / chunk_out 959 |

### 4.4 Default limits against adapter limits

| form | T=1 default | T=1 adapter | T=4096 default | T=4096 adapter |
| --- | --- | --- | --- | --- |
| `seq:s1:c128:tb8` | 6.7 | 6.7 | 3536 | 3511 |
| `seq:s8:c32:tb8:u:kreg` | 4.6 | 4.6 | 2579 | 2550 |
| `seq:s8:c32:tb8:u:kreg:xexp` | 4.7 | 4.7 | 2611 | 2556 |
| `blk:s4:c16:b8` | 13 | 13 | 2377 | 2368 |
| `chunk:bv16` | 128 | 128 | 3447 | 3461 |

The largest difference in this table is 2.2 % (`seq:s8:c32:tb8:u:kreg:xexp` at T = 4096).

### 4.5 Limits each kernel needs

| form | kernels | fits the WebGPU defaults (256 invocations, 16 KiB) |
| --- | --- | --- |
| `seq:s1:c128:tb8:xexp` | 128 / 12.6 KiB | yes |
| `seq:s8:c32:tb8:u:kreg:xexp` | 256 / 13.6 KiB | yes |
| `seq:s8:c16:tb16:u:kreg:xexp` | 128 / 19.6 KiB (over default) | no |
| `blk:s4:c16:b8` | 64 / 0.8 KiB; 64 / 14.8 KiB | yes |
| `blk:s4:c32:b8` | 64 / 0.8 KiB; 128 / 20.8 KiB (over default) | no |
| `blk:s4:c32:b16` | 256 / 3.1 KiB; 128 / 43.1 KiB (over default) | no |
| `chunk:bv16` | 256 / 9.4 KiB; 128 / 12.5 KiB; 256 / 8.4 KiB | yes |
| `chunk:bv32` | 256 / 9.4 KiB; 256 / 24.5 KiB (over default); 256 / 8.4 KiB | no |
| `seq:s4:c16:tb2:u:kreg:inplace:xexp` | 64 / 3.6 KiB | yes |

The chunked form's chunk-start states are 128 MiB per layer at T = 4096, exactly the default
`maxStorageBufferBindingSize`; its scratch in total is 320.5 MiB per layer at T = 4096 (the blocked
form's 13.25 MiB; the sequential forms use none — `scratchBytes` in the bench records).

### 4.6 Projection yardstick

| T | linear, f16 weights (median / min us) | matmul f32 x f32 (median / min us) | best scan form at this T (median us) | scan / linear |
| --- | --- | --- | --- | --- |
| 1 | 371 / 369 | 283 / 277 | 4.2 (`seq:s4:c16:tb2:u:kreg:inplace:xexp`) | 1.1% |
| 64 | 1359 / 1341 | 1034 / 998 | 38 (`blk:s4:c32:b8`) | 2.8% |
| 512 | 3774 / 3737 | 3041 / 2990 | 268 (`blk:s4:c32:b8`) | 7.1% |
| 2048 | 11513 / 11390 | 11615 / 10726 | 1076 (`blk:s4:c32:b8`) | 9.3% |
| 4096 | 22099 / 21869 | 19587 / 19100 | 2131 (`blk:s4:c32:b8`) | 9.6% |


The yardstick varies by about 15 % between processes: a second measurement at T = 4096 (in the
process that also ran the bmm shapes) gave 25.0 ms (linear) / 23.2 ms (matmul) against 22.1 / 19.6 ms
in its own process. **[estimate]** A GDN layer's projections counted together — `in_proj_qkv`,
`in_proj_z`, `in_proj_b` / `in_proj_a` and `out_proj` (4096 · 16448 multiply-accumulates per token)
— are about 2.0 × `in_proj_qkv` by multiply-accumulate count (the four `nn.Linear` before the
recurrence, `in_proj_qkv` / `z` / `b` / `a`, are about 1.51 × by themselves). So the best scan
((a3) `blk:s4:c32:b8`) is about 5 % of them at T = 4096 and the default (a2) `seq:s8:c32:tb8:u:kreg:xexp`
about 6 % (2556 / (2 × 22099) ≈ 5.8 %), and far less of the whole encoder (attention layers and MLPs
not counted).

## 5. Error

### 5.1 Against the f64 oracle

max_abs = max |x - ref|; max_rel = max over value heads of (head max |x - ref|) / (head max |ref|); rel_l2 = ||x - ref||_2 / ||ref||_2. The oracle is the recurrence in f64 on the same f32 inputs (normalised q/k, v, g, beta, S_0). `pass` = finite, and max_abs and rel_l2 of o and S both <= 2 x the larger of the two upstream f32 errors.

| case | form | o max_abs | o max_rel | o rel_l2 | S max_abs | S max_rel | S rel_l2 | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| init_T1 | upstream torch_chunk_gated_delta_rule f32 | 8.3e-9 | 1.6e-6 | 1.3e-7 | 5.0e-8 | 1.1e-7 | 4.4e-8 | - |
| init_T1 | upstream torch_recurrent_gated_delta_rule f32 | 5.6e-9 | 1.1e-6 | 9.1e-8 | 7.1e-8 | 1.3e-7 | 4.5e-8 | - |
| init_T1 | (a1) xexp | 5.6e-9 | 1.0e-6 | 8.4e-8 | 7.1e-8 | 1.3e-7 | 4.5e-8 | yes |
| init_T1 | (a2) xexp | 5.0e-9 | 7.6e-7 | 7.0e-8 | 7.1e-8 | 1.3e-7 | 4.5e-8 | yes |
| init_T1 | (a2) built-in exp | 5.0e-9 | 7.6e-7 | 7.0e-8 | 7.1e-8 | 1.3e-7 | 4.5e-8 | yes |
| init_T1 | (a3) b=8 | 5.6e-9 | 5.0e-7 | 9.4e-8 | 4.6e-8 | 1.1e-7 | 4.0e-8 | yes |
| init_T1 | (a3) b=16 | 5.6e-9 | 5.0e-7 | 9.4e-8 | 4.6e-8 | 1.1e-7 | 4.0e-8 | yes |
| init_T1 | (b) chunked | 7.0e-9 | 4.3e-7 | 1.3e-7 | 4.8e-8 | 1.0e-7 | 3.9e-8 | yes |
| init_T1 | (c1) decode in place | 4.7e-9 | 7.9e-7 | 6.7e-8 | 7.1e-8 | 1.3e-7 | 4.5e-8 | yes |
| init_T1 | (a3) decode in place | 5.6e-9 | 5.0e-7 | 9.4e-8 | 4.6e-8 | 1.1e-7 | 4.0e-8 | yes |
| init_T64 | upstream torch_chunk_gated_delta_rule f32 | 1.5e-7 | 4.2e-6 | 4.7e-7 | 8.2e-7 | 3.4e-6 | 4.7e-7 | - |
| init_T64 | upstream torch_recurrent_gated_delta_rule f32 | 1.1e-8 | 2.6e-7 | 8.5e-8 | 9.4e-8 | 1.2e-7 | 4.6e-8 | - |
| init_T64 | (a1) xexp | 9.9e-9 | 2.5e-7 | 9.2e-8 | 9.4e-8 | 1.2e-7 | 4.6e-8 | yes |
| init_T64 | (a2) xexp | 8.5e-9 | 2.1e-7 | 7.0e-8 | 9.4e-8 | 1.2e-7 | 4.6e-8 | yes |
| init_T64 | (a2) built-in exp | 8.5e-9 | 2.1e-7 | 7.1e-8 | 9.4e-8 | 1.2e-7 | 4.7e-8 | yes |
| init_T64 | (a3) b=8 | 9.9e-9 | 2.4e-7 | 9.8e-8 | 1.6e-7 | 4.0e-7 | 8.7e-8 | yes |
| init_T64 | (a3) b=16 | 2.0e-8 | 3.8e-7 | 1.1e-7 | 2.3e-7 | 5.7e-7 | 1.2e-7 | yes |
| init_T64 | (b) chunked | 1.6e-7 | 4.4e-6 | 3.5e-7 | 8.2e-7 | 1.7e-6 | 4.1e-7 | yes |
| init_T512 | upstream torch_chunk_gated_delta_rule f32 | 7.0e-7 | 1.3e-5 | 5.6e-7 | 6.8e-7 | 1.9e-6 | 4.5e-7 | - |
| init_T512 | upstream torch_recurrent_gated_delta_rule f32 | 1.3e-8 | 1.7e-7 | 8.5e-8 | 7.7e-8 | 1.2e-7 | 4.9e-8 | - |
| init_T512 | (a1) xexp | 1.3e-8 | 2.0e-7 | 9.1e-8 | 7.7e-8 | 1.2e-7 | 4.9e-8 | yes |
| init_T512 | (a2) xexp | 1.2e-8 | 1.7e-7 | 7.0e-8 | 7.7e-8 | 1.2e-7 | 4.9e-8 | yes |
| init_T512 | (a2) built-in exp | 1.2e-8 | 1.7e-7 | 7.0e-8 | 7.7e-8 | 1.2e-7 | 4.9e-8 | yes |
| init_T512 | (a3) b=8 | 3.3e-8 | 6.6e-7 | 1.0e-7 | 1.0e-7 | 2.9e-7 | 5.9e-8 | yes |
| init_T512 | (a3) b=16 | 9.4e-8 | 1.5e-6 | 1.3e-7 | 1.4e-7 | 2.9e-7 | 7.9e-8 | yes |
| init_T512 | (b) chunked | 2.0e-7 | 3.6e-6 | 3.7e-7 | 5.0e-7 | 1.8e-6 | 3.8e-7 | yes |
| init_T2048 | upstream torch_chunk_gated_delta_rule f32 | 6.9e-7 | 9.4e-6 | 5.2e-7 | 1.1e-6 | 5.7e-6 | 4.6e-7 | - |
| init_T2048 | upstream torch_recurrent_gated_delta_rule f32 | 1.6e-8 | 1.8e-7 | 8.5e-8 | 7.7e-8 | 1.3e-7 | 4.8e-8 | - |
| init_T2048 | (a1) xexp | 2.3e-8 | 2.6e-7 | 9.1e-8 | 7.7e-8 | 1.3e-7 | 4.8e-8 | yes |
| init_T2048 | (a2) xexp | 1.7e-8 | 1.6e-7 | 7.0e-8 | 7.7e-8 | 1.3e-7 | 4.8e-8 | yes |
| init_T2048 | (a2) built-in exp | 1.7e-8 | 1.6e-7 | 7.1e-8 | 7.7e-8 | 1.3e-7 | 4.8e-8 | yes |
| init_T2048 | (a3) b=8 | 5.1e-8 | 5.8e-7 | 1.0e-7 | 6.6e-8 | 3.5e-7 | 5.7e-8 | yes |
| init_T2048 | (a3) b=16 | 9.2e-8 | 1.2e-6 | 1.3e-7 | 6.7e-8 | 3.5e-7 | 6.3e-8 | yes |
| init_T2048 | (b) chunked | 3.9e-7 | 5.8e-6 | 3.6e-7 | 1.1e-6 | 5.7e-6 | 4.4e-7 | yes |
| init_T4096 | upstream torch_chunk_gated_delta_rule f32 | 7.2e-7 | 6.5e-6 | 5.1e-7 | 3.1e-6 | 1.3e-5 | 1.5e-6 | - |
| init_T4096 | upstream torch_recurrent_gated_delta_rule f32 | 1.8e-8 | 2.1e-7 | 8.5e-8 | 7.2e-8 | 1.0e-7 | 4.6e-8 | - |
| init_T4096 | (a1) xexp | 2.0e-8 | 2.4e-7 | 9.1e-8 | 7.2e-8 | 1.0e-7 | 4.6e-8 | yes |
| init_T4096 | (a2) xexp | 1.3e-8 | 1.5e-7 | 7.0e-8 | 7.2e-8 | 1.0e-7 | 4.6e-8 | yes |
| init_T4096 | (a2) built-in exp | 1.3e-8 | 1.5e-7 | 7.0e-8 | 7.2e-8 | 1.0e-7 | 4.6e-8 | yes |
| init_T4096 | (a3) b=8 | 3.4e-8 | 4.6e-7 | 1.0e-7 | 5.6e-7 | 2.3e-6 | 2.3e-7 | yes |
| init_T4096 | (a3) b=16 | 7.0e-8 | 9.5e-7 | 1.2e-7 | 5.6e-7 | 2.3e-6 | 2.6e-7 | yes |
| init_T4096 | (b) chunked | 3.4e-7 | 4.7e-6 | 3.5e-7 | 3.2e-6 | 1.3e-5 | 1.3e-6 | yes |
| long_T1 | upstream torch_chunk_gated_delta_rule f32 | 4.1e-8 | 5.1e-7 | 1.9e-7 | 1.1e-7 | 1.0e-7 | 4.2e-8 | - |
| long_T1 | upstream torch_recurrent_gated_delta_rule f32 | 1.0e-8 | 1.7e-7 | 8.2e-8 | 9.3e-8 | 8.8e-8 | 4.0e-8 | - |
| long_T1 | (a1) xexp | 1.2e-8 | 1.9e-7 | 9.0e-8 | 9.1e-8 | 1.0e-7 | 3.2e-8 | yes |
| long_T1 | (a2) xexp | 1.1e-8 | 1.5e-7 | 6.7e-8 | 6.9e-8 | 6.7e-8 | 3.1e-8 | yes |
| long_T1 | (a2) built-in exp | 1.0e-8 | 1.5e-7 | 8.0e-8 | 9.1e-8 | 1.0e-7 | 4.6e-8 | yes |
| long_T1 | (a3) b=8 | 1.3e-8 | 1.9e-7 | 8.1e-8 | 9.1e-8 | 1.0e-7 | 4.6e-8 | yes |
| long_T1 | (a3) b=16 | 1.3e-8 | 1.9e-7 | 8.1e-8 | 9.1e-8 | 1.0e-7 | 4.6e-8 | yes |
| long_T1 | (b) chunked | 2.2e-8 | 4.1e-7 | 1.9e-7 | 1.6e-7 | 1.5e-7 | 5.4e-8 | yes |
| long_T1 | (c1) decode in place | 9.6e-9 | 1.4e-7 | 6.8e-8 | 6.9e-8 | 7.2e-8 | 3.1e-8 | yes |
| long_T1 | (a3) decode in place | 1.3e-8 | 1.9e-7 | 8.1e-8 | 9.1e-8 | 1.0e-7 | 4.6e-8 | yes |
| long_T64 | upstream torch_chunk_gated_delta_rule f32 | 3.8e-8 | 4.5e-7 | 2.3e-7 | 3.7e-7 | 4.1e-7 | 1.7e-7 | - |
| long_T64 | upstream torch_recurrent_gated_delta_rule f32 | 2.5e-8 | 2.7e-7 | 1.3e-7 | 5.3e-7 | 5.2e-7 | 2.0e-7 | - |
| long_T64 | (a1) xexp | 2.4e-8 | 2.6e-7 | 1.2e-7 | 3.6e-7 | 4.1e-7 | 1.6e-7 | yes |
| long_T64 | (a2) xexp | 1.9e-8 | 2.1e-7 | 1.0e-7 | 3.8e-7 | 4.1e-7 | 1.6e-7 | yes |
| long_T64 | (a2) built-in exp | 7.2e-8 | 8.3e-7 | 3.5e-7 | 1.1e-6 | 1.3e-6 | 6.4e-7 | **no** |
| long_T64 | (a3) b=8 | 2.8e-8 | 3.1e-7 | 1.3e-7 | 5.4e-7 | 5.6e-7 | 1.6e-7 | yes |
| long_T64 | (a3) b=16 | 2.9e-8 | 3.2e-7 | 1.3e-7 | 3.3e-7 | 3.5e-7 | 1.5e-7 | yes |
| long_T64 | (b) chunked | 3.2e-8 | 3.8e-7 | 1.7e-7 | 3.7e-7 | 4.2e-7 | 1.7e-7 | yes |
| long_T512 | upstream torch_chunk_gated_delta_rule f32 | 7.1e-8 | 5.9e-7 | 2.5e-7 | 3.9e-7 | 4.3e-7 | 1.9e-7 | - |
| long_T512 | upstream torch_recurrent_gated_delta_rule f32 | 5.2e-8 | 4.3e-7 | 2.0e-7 | 9.0e-7 | 7.1e-7 | 2.9e-7 | - |
| long_T512 | (a1) xexp | 4.3e-8 | 3.1e-7 | 1.7e-7 | 5.9e-7 | 4.6e-7 | 2.2e-7 | yes |
| long_T512 | (a2) xexp | 4.1e-8 | 3.0e-7 | 1.5e-7 | 5.9e-7 | 4.6e-7 | 2.2e-7 | yes |
| long_T512 | (a2) built-in exp | 1.7e-7 | 1.5e-6 | 8.3e-7 | 2.1e-6 | 1.9e-6 | 1.4e-6 | **no** |
| long_T512 | (a3) b=8 | 5.5e-8 | 4.4e-7 | 2.0e-7 | 8.2e-7 | 6.2e-7 | 3.1e-7 | yes |
| long_T512 | (a3) b=16 | 4.8e-8 | 3.9e-7 | 1.8e-7 | 7.7e-7 | 6.1e-7 | 2.4e-7 | yes |
| long_T512 | (b) chunked | 8.7e-8 | 6.3e-7 | 2.4e-7 | 8.2e-7 | 6.4e-7 | 2.4e-7 | yes |
| long_T2048 | upstream torch_chunk_gated_delta_rule f32 | 7.5e-8 | 5.4e-7 | 2.6e-7 | 5.0e-7 | 4.5e-7 | 1.9e-7 | - |
| long_T2048 | upstream torch_recurrent_gated_delta_rule f32 | 6.5e-8 | 4.0e-7 | 2.2e-7 | 8.3e-7 | 7.1e-7 | 3.1e-7 | - |
| long_T2048 | (a1) xexp | 7.6e-8 | 4.6e-7 | 1.8e-7 | 7.1e-7 | 5.2e-7 | 2.3e-7 | yes |
| long_T2048 | (a2) xexp | 6.5e-8 | 4.0e-7 | 1.7e-7 | 6.2e-7 | 5.1e-7 | 2.2e-7 | yes |
| long_T2048 | (a2) built-in exp | 2.1e-7 | 1.5e-6 | 9.2e-7 | 2.1e-6 | 2.0e-6 | 1.4e-6 | **no** |
| long_T2048 | (a3) b=8 | 9.6e-8 | 5.8e-7 | 2.4e-7 | 1.0e-6 | 7.6e-7 | 3.2e-7 | yes |
| long_T2048 | (a3) b=16 | 6.5e-8 | 4.0e-7 | 1.9e-7 | 7.7e-7 | 5.7e-7 | 2.5e-7 | yes |
| long_T2048 | (b) chunked | 8.7e-8 | 7.0e-7 | 2.6e-7 | 6.9e-7 | 5.1e-7 | 2.4e-7 | yes |
| long_T4096 | upstream torch_chunk_gated_delta_rule f32 | 8.8e-8 | 5.8e-7 | 2.6e-7 | 3.9e-7 | 3.7e-7 | 1.9e-7 | - |
| long_T4096 | upstream torch_recurrent_gated_delta_rule f32 | 7.0e-8 | 4.3e-7 | 2.2e-7 | 9.1e-7 | 7.0e-7 | 3.0e-7 | - |
| long_T4096 | (a1) xexp | 6.1e-8 | 3.7e-7 | 1.8e-7 | 6.1e-7 | 4.8e-7 | 2.3e-7 | yes |
| long_T4096 | (a2) xexp | 6.3e-8 | 3.8e-7 | 1.7e-7 | 6.7e-7 | 5.2e-7 | 2.3e-7 | yes |
| long_T4096 | (a2) built-in exp | 2.0e-7 | 1.4e-6 | 9.4e-7 | 2.0e-6 | 1.9e-6 | 1.4e-6 | **no** |
| long_T4096 | (a3) b=8 | 8.7e-8 | 5.5e-7 | 2.4e-7 | 1.0e-6 | 7.9e-7 | 3.2e-7 | yes |
| long_T4096 | (a3) b=16 | 7.1e-8 | 4.7e-7 | 1.9e-7 | 7.4e-7 | 5.7e-7 | 2.3e-7 | yes |
| long_T4096 | (b) chunked | 9.2e-8 | 5.6e-7 | 2.6e-7 | 8.0e-7 | 6.1e-7 | 2.4e-7 | yes |

### 5.2 Acceptance over every case run

| form [limits] | passed / cases | failed cases |
| --- | --- | --- |
| `blk:s4:c16:b8 [adapter]` | 10 / 10 | - |
| `blk:s4:c16:b8 [default]` | 10 / 10 | - |
| `blk:s4:c16:b8:inplace [adapter]` | 2 / 2 | - |
| `blk:s4:c32:b16 [adapter]` | 10 / 10 | - |
| `blk:s4:c32:b8 [adapter]` | 10 / 10 | - |
| `chunk:bv16 [adapter]` | 10 / 10 | - |
| `chunk:bv16 [default]` | 10 / 10 | - |
| `chunk:bv32 [adapter]` | 10 / 10 | - |
| `seq:s1:c128:tb8 [adapter]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s1:c128:tb8 [default]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s1:c128:tb8:xexp [adapter]` | 10 / 10 | - |
| `seq:s4:c16:tb2:u:kreg:inplace [adapter]` | 2 / 2 | - |
| `seq:s4:c16:tb2:u:kreg:inplace:xexp [adapter]` | 2 / 2 | - |
| `seq:s4:c16:tb2:u:kreg:inplace:xexp [default]` | 2 / 2 | - |
| `seq:s8:c16:tb16:u:kreg [adapter]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s8:c16:tb16:u:kreg:xexp [adapter]` | 10 / 10 | - |
| `seq:s8:c16:tb2:u:inplace [adapter]` | 2 / 2 | - |
| `seq:s8:c16:tb2:u:inplace:xexp [adapter]` | 2 / 2 | - |
| `seq:s8:c32:tb16:u:kreg [adapter]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s8:c32:tb8:u:kreg [adapter]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s8:c32:tb8:u:kreg [default]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s8:c32:tb8:u:kreg:inplace [adapter]` | 2 / 2 | - |
| `seq:s8:c32:tb8:u:kreg:lazy [adapter]` | 6 / 10 | long_T2048, long_T4096, long_T512, long_T64 |
| `seq:s8:c32:tb8:u:kreg:xexp [adapter]` | 10 / 10 | - |
| `seq:s8:c32:tb8:u:kreg:xexp [default]` | 10 / 10 | - |

r = 1 check (q / k repeated to 32 heads, Hk = 32, against the r = 2 run of the same form): 90 / 90
runs bit-identical in o and S.

### 5.3 Fault injection

| fault | failed / cases | cases that passed | largest o rel_l2 |
| --- | --- | --- | --- |
| no-decay | 10 / 10 | - | 8.5e+0 |
| no-beta | 10 / 10 | - | 1.1e+0 |
| head-mod | 10 / 10 | - | 9.5e-1 |
| o-before-update | 10 / 10 | - | 1.5e+0 |
| no-initial | 2 / 10 | init_T2048, init_T4096, init_T512, init_T64, long_T2048, long_T4096, long_T512, long_T64 | 9.5e-1 |


The missing-initial-state fault can only act at T = 1 (prefill has no initial state).

### 5.4 Accuracy of f32 exp on this GPU

| range | built-in exp: max / mean abs / mean signed ULP | exp_acc: max / mean abs / mean signed ULP |
| --- | --- | --- |
| long memory g in [-0.1, 0) | 3.00 / 0.535 / -0.337 | 1.00 / 0.288 / -0.091 |
| g in [-1, -0.1) | 4.19 / 0.534 / -0.383 | 1.00 / 0.293 / -0.057 |
| fast decay g in [-80, -1) | 115.04 / 12.223 / 6.490 | 7.00 / 1.492 / 1.472 |

The built-in `exp` has a negative bias on the long-memory range (mean −0.34 ULP). A per-token
product of `exp(g)` carries that bias across the memory, which is why the sequential forms with the
built-in `exp` fail the long-memory cases (o rel_l2 up to 9.4e-7 against a bound of 5.1e-7 at
T = 4096) and pass with the accurate exp (1.7e-7, below upstream's recurrent function's 2.2e-7). The
blocked and chunked forms take `exp` of block-local cumulative sums and pass with the built-in `exp`.

## 6. Read from the raw JSON at write-up (2026-10-27)

These are comparisons of the recorded metrics, not new measurements. The records hold metrics, not
output hashes, so equal metrics suggest but do not prove bit identity; §6.1 has the raw-word
comparison of the independent check's re-run.

- Geometry within a form: `seq:s8:c32:tb8:u:kreg:xexp` and `seq:s8:c16:tb16:u:kreg:xexp` (columns
  per workgroup and token block changed, key split kept) have identical o / S max_abs and rel_l2 on
  all 10 cases, and so do the same form under adapter and default limits, `blk:s4:c16:b8` and
  `blk:s4:c32:b8`, and `chunk:bv16` and `chunk:bv32`. `blk:s4:c32:b8` and `blk:s4:c32:b16` (block
  size changed) differ on 8 of 10 cases. At T = 1, the decode geometry `seq:s4:c16:tb2:u:kreg:inplace:xexp`
  (key split 4) differs from `seq:s8:c32:tb8:u:kreg:xexp` (key split 8) on both cases.
- Share of the bound used (the largest of the four ratios kernel error / bound, over the 10 cases):
  (a2) `seq:s8:c32:tb8:u:kreg:xexp` 0.500 (init, T = 1), (a3) `blk:s4:c32:b8` 0.634 (long,
  T = 2048), (b) `chunk:bv16` 0.751 (long, T = 1); (a2) with the built-in `exp` 2.455 (long, T = 512).
- Share of the o band only (o max_abs and o rel_l2; the value-form op exposes only o): (a2) with the
  accurate exp 0.434 at worst (long, T = 2048). (a2) with the built-in `exp`
  (`long_T*__seq_s8_c32_tb8_u_kreg__adapter.json`), max_abs / rel_l2: long T = 64 0.937 / 0.747
  (passes), T = 512 1.157 / 1.658, T = 2048 1.366 / 1.805, T = 4096 1.120 / 1.834 (these three fail).
  The 2.455 above is S's rel_l2 at long T = 512, so on o the regression's margin over the band is
  1.83× at worst and does not exist at T = 64. In the fast-decay regime the band is set by the
  upstream chunked function's large error, and (a2) uses about 0.07 of the o band there (§6.1), so a
  numeric regression of roughly 14× would pass there; the coarse faults of §5.3 (share ≥ 1e6) fail
  wherever they can act.
- Kernel against upstream's prefill function `torch_chunk_gated_delta_rule` in f32 (o max_abs, the
  largest over the cases): 7.19e-7 for (a2) with the accurate exp, 7.17e-7 for `blk:s4:c32:b8`,
  9.09e-7 for `chunk:bv16`, all at init T = 4096, where upstream's chunked function itself is
  7.2e-7 from the oracle.

### 6.1 Raw output words (the independent check's re-run, 2026-10-27)

The independent check re-ran kernels on the same GPU and fixed inputs and compared the raw output
words (Uint32) of o and the final state S: `outputs/diag/gdn-scan-2026-10-27/check/run_gpu.ts`
writes the words, `check/metrics.py` compares them and `check/metrics.json` holds the result.

- Repeated runs: `seq:s8:c32:tb8:u:kreg:xexp`, `seq:s8:c16:tb16:u:kreg:xexp` and the built-in-exp
  `seq:s8:c32:tb8:u:kreg`, two runs each at long T = 4096: the second run is identical to the first
  in every word of o and S for all three.
- Geometry: `s8:c32:tb8` against `s8:c16:tb16` (both with the accurate exp): 0 differing words in o
  and in S at init T = 4096 and at long T = 4096.
- Share of the band in the re-run: (a2) 0.068 at init T = 4096 (o rel_l2; both geometries), 0.386
  at long T = 4096; the built-in-exp form 2.27 at long T = 4096.

Other cases, B > 1, and other GPUs and backends are not measured.

## 7. What was tried (tuning notes)

Single runs of 10 rounds unless stated; times per layer at T = 4096 unless stated. Runs made before
the VRAM guard of §3 are kept under `results/explore/` of the raw material and quoted only as tuning
notes.

- **(a1)**: 128 columns per workgroup, token loop not unrolled: 3.52 ms; the same with the token
  block unrolled: 5.56 ms (instruction-cache or register pressure suspected — not verified); 64
  columns: 4.01 ms; 32 columns: 5.66 ms. Kept `seq:s1:c128:tb8`.
- **(a2)**: split 8, 32 columns, token block 8: 2.87 ms → k in registers 2.57 ms → token loop also
  unrolled 2.53 ms (unrolling alone: 2.86 ms). Lazy decay (S = λR, 3 instead of 4 floating-point
  operations per state element, renormalised below 2^-80): no gain (2.53 → 2.53 ms with k in
  registers; 2.86 → 2.74 ms without), so the floating-point work is not the limit. Token block 16:
  2.36 ms (22.6 KiB); 16 columns with block 16: 2.33 ms (19.6 KiB). Split 4 (32 columns, unrolled):
  2.74 ms. At T = 512: split 2 with 16 / 32 / 64 columns 605 / 539 / 341 µs, split 4 with 16 / 64
  columns 358 / 373 µs, split 8 with 8 / 16 columns 361 / 314 µs, token block 4 / 16 347 / 300 µs,
  against 325–335 µs for the kept geometry.
- **Where (a2)'s time goes** (`seq:s8:c32:tb8:kreg`, T = 512, results deliberately wrong): full
  369 µs; without the per-token barrier 331; without the staging loads 315; without the partial-sum
  reads 301; with k / q of the token body replaced by constants 306; with no state work at all
  (exchange, barrier, staging, output only) 173; that skeleton without staging 114, without barrier
  144, without the partial reads 165, without all three 95. No single part dominates. With fewer
  heads (same work per workgroup, fewer workgroups) Hv = 32 / 16 / 8 / 2 → 371 / 297 / 293 / 259 µs:
  even 8 workgroups (one per SM) need 0.5 µs per token (about 960 cycles at 1.9 GHz). The sequential
  form is bound by the latency of one workgroup's token step, not by the GPU's throughput; the design
  note's estimate of 300–600 cycles per step was low by about 2×.
- **(a3)**: split ∈ {2, 4, 8}, columns ∈ {8, 16, 32, 64}, block ∈ {8, 16}. Split 4 / 16 columns /
  b = 8 2.33 ms; split 8 / 8 columns 2.65 ms; split 8 / 16 columns 2.37 ms; split 8 / 16 columns /
  b = 16 2.89 ms; split 4 / 16 columns / b = 16 3.36 ms; split 4 / 32 columns / b = 16 2.23 ms;
  split 4 / 32 columns / b = 8 2.08 ms; split 2 with 32 / 64 columns 3.64 / 2.69 ms; split 2 / 16
  columns / b = 16 10.9 ms (register spill suspected — not verified); the last four are console
  readings, not kept as files. Ablation of the state kernel (split 4 / 16 columns / b = 8): full
  2.29 ms; without the partial dots 1.56; without the state update 1.47 (this removes the dots too);
  without the per-column scalar work 2.05; without the staging loads 1.87; without the exchange
  barriers 2.22; with dots, update and scalar work all removed 1.26, and also without staging 0.42.
  Kept `blk:s4:c16:b8` (default limits), `blk:s4:c32:b8` (20.8 KiB), `blk:s4:c32:b16` (43.1 KiB).
- **(b)**: first version 5.28 ms (prep 2.16 / state 2.10 / output 1.03, from the run's console
  line). Runtime-bounded inner loops replaced by constant-bound guarded loops: 4.38 ms (prep 1.26).
  State-pass loads issued in explicit batches of 8–16 before use: state pass 2.11 → 1.05–1.10 ms.
  The forward substitution's inner sum split over 4 threads per column and the U / W loads batched by
  4 tokens: prep 1.26 → 1.29 ms (no gain); the output kernel's loads batched: 1.02 → 0.95 ms; together
  3.30–3.38 ms (the campaign's split for `chunk:bv32`: prep 1.37 / state 1.09 / output 0.96 ms —
  §4.3). The 32-column state pass (24.5 KiB) is 4–5 % faster than the 16-column one. Running the
  matmul-shaped parts on the product's bmm alone would cost 3.57 ms by itself (Gram
  `[1024,64,128]×[128,64]` 311 µs ×2, triangular `[2048,64,64]×[64,128]` 607 µs ×3,
  `(Q e^Γ) S_c` `[2048,64,128]×[128,128]` 1,122 µs ×1) — slower than the spike's own prep + output
  kernels (2.3 ms). Kept `chunk:bv16` (default limits) and `chunk:bv32`.
- **(c1) decode** (T = 1, state in place): split 4 / 16 columns / token block 2 with k in registers
  4.1–4.2 µs; split 8 with 16 or 8 columns 4.3–4.4 µs; split 2 4.6 µs; the prefill geometry
  `seq:s8:c32:tb8:u:kreg` in place 4.8 µs (with the built-in exp). (a3) in place 13 µs; (b)
  126–128 µs (it pads to one 64-token chunk).
- **Dawn robustness off** (a diagnostic toggle, not a product option): (a2) `seq:s8:c16:tb16:u:kreg`
  2.47 ms (2.36 with robustness), (a3) `blk:s4:c32:b8` 2.05 (2.13), (b) `chunk:bv32` 3.28 (3.41),
  (a1) 3.57 (3.51) — within ±5 %, so bounds checks do not explain the gap to the FMA throughput.
- **What did not work**: the design note's speed estimate for the sequential form (0.8–1.5 ms per
  layer; measured 2.36 ms); lazy decay and unrolling alone; the chunked form against the sequential
  ones (≈ 1.6× slower than (a3) at T = 4096, 3.8× at T = 64, ≈ 10× at T = 1); the built-in `exp`
  in the sequential forms. Two harness bugs were found and fixed (the in-place state buffer was
  poisoned after the initial state was copied into it; the missing-initial-state fault had to keep
  reading S₀ × 0 so the binding stayed in the automatic layout); the kernels were not at fault.

## 8. Toolchain findings (beyond this op)

1. **Built-in `exp` accuracy** (§5.4): up to 3 ULP with a −0.34 ULP mean bias on [−0.1, 0), up to
   4.2 ULP on [−1, −0.1), up to 115 ULP on [−80, −1) (WGSL allows 3 + 2|x| ULP). Any kernel that
   multiplies per-step `exp` outputs over long sequences inherits the bias. The range-reduced
   polynomial (`exp_acc` in the spike's `kernels/seq.ts`) has at most 1 ULP and a −0.09 ULP mean on
   the long-memory range. Its domain is `x ≤ 0`: below −87 it returns 0 (which also flushes the
   normal results on (−87.34, −87) to 0, negligible); for x above about 88.4 the true result
   overflows f32 (`n = round(x·log2 e) ≥ 128`), and WGSL's `ldexp` gives an indeterminate value once
   the exponent exceeds bias + 1 = 128 (a reading of the WGSL spec, not tried on the GPU), so +inf
   is not guaranteed. Its code comment's truncation bound (`< 2e-9 relative`) is low: the bound is
   r⁸/8! ≈ 5.2e-9 for |r| ≤ ln 2 / 2, about 7e-9 relative to e^r — still well under 1 ULP.
2. **Runtime-bounded loop trips are expensive**: a loop whose bound is not a compile-time constant
   costs about 60–67 cycles per trip here whatever its body; hoisting the bound into a `let` does not
   help; a constant bound costs 5 cycles per trip; with Dawn's robustness disabled the runtime-bounded
   loop costs 8. On top of the bare trip a barrier gives 107 cycles per trip, a workgroup exchange
   (write, barrier, read) 132, the same exchange reading 8 partials 239. `gru_scan`'s dot loop
   (`packages/runtime/src/kernels/gru-scan.ts`, `for (var k = 0u; k < hidden; …)`) is such a loop
   (not measured, not touched).
3. **The Dawn instance must stay referenced**: with the `create()` result held only in a local of the
   acquire function, later pipeline creation aborted inside Dawn (`std::system_error`, a futex
   error); holding it at module scope (as `packages/test-support/src/setup.ts` does on a global)
   fixed it — most likely the instance was garbage-collected while the device lived. In that first
   setup `createComputePipelineAsync` also never settled; the spike switched to the synchronous call
   and did not re-test the async one after the fix.
4. **Slow phases**: some probe runs show medians 5–7× their minimum (the chunked prep kernel at
   7.7 ms median / 1.46 ms min), the machine's known slow mode. The chunked-kernel ablation is
   therefore not used. The campaign records have min and median within a few per cent and two
   repetitions within 2.2 %.

## 9. Not measured

- The export side of the design note's step 7: the length-axis specialisation of `torch.export` on
  the upstream functions, node counts after `run_decompositions` at fixed T, the custom op with
  `register_fake`, and a 1-layer `Qwen3_5GatedDeltaNet` through the exporter (expected blockers:
  `rsqrt` in `l2norm`, the slice on the symbolic axis after the causal conv).
- The decomposed recurrence on the GPU (its dispatch count and intermediate traffic).
- Bit identity between geometries of one form and run-to-run identity on one device beyond the
  cases of §6.1 (§6 compares metrics only; §6.1 compares raw words on 2 geometry cases and 3 forms
  at long T = 4096).
- Real MOSS `A_log` / `dt_bias`, bf16 inputs, B > 1 (the kernels take B through the grid but ran
  with B = 1 only), other head dims, Chrome and other GPUs or backends, subgroup variants (excluded by
  the product rule).
- (b) with workgroup-staged K chunks and a blocked triangular inverse; hazard canaries for `exp` at
  g = 0, −0, −inf and NaN.

## 10. Raw material and reproduction

Under `outputs/diag/gdn-scan-2026-10-27/` of the main checkout: `DESIGN.md` (the design note),
`upstream_error_probe.py`, `limits.mjs`, and `spike/` with `RESULTS.md`, the scripts (`gen_inputs.py`,
`correct.ts`, `bench.ts`, `yardstick.ts`, `probe_*.ts`, `kernels/`, `lib/`, the `run_*.zsh` campaigns
and `gpu_run.zsh`), the fixed inputs in `data/` (1.4 GB) and the records in `results/` (`bench/`,
`correct/`, `yardstick/`, `runs.jsonl`, logs). Every table in §4 and §5 is the output of
`node report_tables.ts` over `results/` (`results/tables.md`), copied verbatim; the table lines were
checked against a fresh regeneration on 2026-10-27.

> Point-in-time record (2026-10-31) of the first GPU runs of the Qwen3.5 text encoder (`Qwen3_5TextEncoder` over the i8 series `qwen3.5-9b-text-moss-image-sft-i8-dyn`) on one machine: bit identity of the host-built RoPE rows with the recipe's, the GPU-versus-CPU-reference error, fault injections against the band, encode times and device memory. Measurement material for [ADR 0137](../decisions/0137-qwen3_5-text-encoder.md) (proposed); not a design source of truth. The one-off measurement script and its raw JSON are under `outputs/diag/qwen3_5-encoder-gpu-2026-10-31/` of the main checkout (not tracked).

# Qwen3.5 text encoder on the GPU: host inputs, parity with the CPU reference, time and memory

Question: does the exported text tower run on the GPU through the new encoder, how close are its
rows to Karume's CPU reference execution of the same container, how tight can the band be while
still catching a broken layer, and what do an encoding and the Session cost on a 12 GB GPU?

Answer: it runs. The host-built RoPE rows are bit-identical to the recipe's for every position up
to 4,095. The GPU rows sit at a relative RMS error of 5.09e-6 (T = 10) and 3.62e-6 (T = 63) from the
CPU reference; a band of 2.6e-5 leaves every injected fault 65× or more outside it. A warm encoding
takes 60 ms at T = 10, 196 ms at T = 63 and 1.40 s at T = 1,000; the Session holds 7.40 GiB of
weights and the device peaked at 7,850 MiB.

## 1. Setup

- RTX 3080 Ti (12 GB, driver 615.78.08), Linux 7.2.7, Node 24.22.2, Dawn (npm `webgpu` 0.6.2,
  Vulkan), geometry profile `nvidia-ampere`. Checkout `ecffbf8e` plus the uncommitted encoder
  module of ADR 0137. The GPU was idle (`tools/diag/gpu-busy.zsh 5`: 0 %).
- The container: 34 parts, part 0 sha256 `e9b581dc…260ff`, exported 2026-10-31 09:02 UTC.
- The oracle: `outputs/diag/qwen3_5-export-2026-10-31/ir.full-i8.{negative,fridge-note}.f32` —
  `ir_cpu_reference.ts` there walks the container's graph op by op with the runtime's
  `applyReferenceOp`, on the inputs `inputs.full.<case>.safetensors` (the recipe's `inputs_for` over
  the ids of MOSS-Image's prompt asset); 11 min 31 s for both prompts on the CPU.
- Prompts: `negative` (the empty prompt — the chat template alone, T = 10), `fridge-note` (the
  README's quick-start prompt, T = 63) and a synthetic T = 1,000 sequence for timing only (ids
  `i · 7919 mod 248320`, the recipe's `example_inputs` formula).
- Metric: relative RMS error `‖GPU − CPU‖₂ / ‖CPU‖₂` over all rows of one encoding (f64
  accumulation), with the maximum absolute difference and the smallest per-row cosine similarity.

## 2. Host inputs

The TypeScript rows (`buildQwen3_5RopeRows`, f64 then f32) against the recipe's `rope_rows(range(4096),
config)` with the pinned checkpoint's config (theta 1e7, rotary width 64): **0 of 262,144 words
differ in `rope_cos` and 0 of 262,144 in `rope_sin`**. The rows and `positions` the encoder builds
from the recorded ids are byte-identical to the two `inputs.full.*` files the CPU reference was fed
(the e2e's host check). The bound against upstream's own f32 rotary module (ADR 0091 decision 2) was
not re-measured: the rows are the recipe's, and the recipe's record holds that comparison.

## 3. GPU against the CPU reference

| Case | T | relRms | max abs. difference | min row cosine | bit-identical |
| --- | --: | --: | --: | --: | --- |
| negative | 10 | 5.09e-6 | 1.26e-4 | 0.99999999997 | no |
| fridge-note | 63 | 3.62e-6 | 1.75e-4 | 0.99999999979 | no |

Seven warm repetitions at each T returned the cold run's rows bit for bit, and the later e2e runs (the
comparison run and the lane run) recorded the same errors to the last digit. The sha256 rows of `node-nvidia-geforce-rtx-3080-ti`
were written from these rows (`417f90dc…` negative, `50c2cdd5…` fridge-note).

Band: the worst error × 5, rounded up to two significant figures (ADR 0119 decision 8's rule) =
**2.6e-5**.

## 4. Faults against the same references

Weight faults change one per-channel scale block on the way into the Session (the container's
files are untouched); each is a separate Session. Input faults reuse the main Session.

| Fault | negative | fridge-note | × band (smaller case) |
| --- | --: | --: | --: |
| layer 15 MLP `down_proj` scale × 1.01 | 2.87e-3 | 1.70e-3 | 65 |
| layer 15 MLP `down_proj` scale × 0 (the layer's MLP output zeroed) | 0.271 | 0.160 | 6,200 |
| layer 14 GDN `out_proj` scale × 0 | 0.147 | 0.161 | 5,700 |
| `rope_cos` / `rope_sin` swapped | 0.080 | 0.067 | 2,600 |
| one token of fridge-note replaced by its neighbour | — | 0.198 | 7,600 |

The e2e keeps the swapped-RoPE case as its liveness check. With a fault in the host code (cos and
sin swapped in `qwen3_5SessionInputs`), the e2e failed four of its six tests: the host
byte-identity check (at byte 2 of `rope_cos`), both band checks (0.080 and 0.067) and the liveness
check (the double swap restored the order).

## 5. Time

Wall time of `encode` (host input building, upload, run, readback). Cold = the first encoding at
that T on the Session (the first one also creates the pipelines; the Session had used 36 pipeline
keys by the end); warm = 7 repetitions.

| T | cold | warm median | warm min |
| --: | --: | --: | --: |
| 10 | 400 ms | 59.7 ms | 59.3 ms |
| 63 | 268 ms | 196.1 ms | 194.5 ms |
| 1,000 | 1,466 ms | 1,397.8 ms | 1,391.2 ms |

Building the Session took 8.30 s for 7,946,672,780 bytes in 33 weight parts, with the parts in the
page cache from a run a minute earlier (supply wait 5.53 s, `writeBuffer` issue 2.22 s, upload fence
0.29 s). A cold disk adds the read time. At T = 1,000 the fusion pass counted 16 `rope` and 80
`silu` fusions per run.

## 6. Device memory

| After | runtime-accounted | of which weights | plan backing | device peak (nvidia-smi, GPU-wide) |
| --- | --: | --: | --: | --: |
| Session build | — | — | — | 7,603 MiB |
| T = 10 | 7.40 GiB | 7.40 GiB | 1.6 MB | 7,659 MiB |
| T = 63 | 7.41 GiB | 7.40 GiB | 12.0 MB | 7,678 MiB |
| T = 1,000 | 7.60 GiB | 7.40 GiB | 194.7 MB | 7,850 MiB |

Runtime-accounted = weights + the last run's arena + plan backing (resident and input). The idle
GPU showed 52 MiB. The plan backing is cumulative per Session (one per T seen). The admission
estimate for T ≤ 1,000 was 8.22 GB (7.65 GiB) and refused T = 4,096 under the WebGPU default
`maxStorageBufferBindingSize` (the fused attention's temporaries); T = 4,096 was not run.

## 7. Not measured

- T above 1,000 on the GPU (ADR 0137 open point 2).
- Other GPUs and backends (D3D12 on the RTX 5070 Ti).
- The bound of the host rows against upstream's f32 rotary module (section 2).

> Point-in-time record (2026-10-27 to 2026-10-28) of the measurements of the MOSS-Image DiT and VAE port (stages 2–5 of [ADR NNNN](../decisions/NNNN-moss-image-dit-vae.md), branch `wip/moss-image`) on the RTX 3080 Ti under Node + Dawn (Vulkan): the parity bands against the CPU references, the census, the timing, the VRAM held and needed per preset and size, the largest areas tried and a per-op profile. Not a design source of truth (that is ADR NNNN, proposed). Raw material of stages 2–4 is under `outputs/bench/karume-moss-image/` and `outputs/verify/node-nvidia-geforce-rtx-3080-ti/` of the main checkout (not tracked); the stage-5 logs (timing, VRAM, area probes, profile) were taken in a scratch directory and are transcribed here.

# MOSS-Image DiT and VAE: parity, census, timing and VRAM

Question: does the port of MOSS-Image's DiT (i8 weights) and its untiled FLUX.2 decoder match
upstream within bands derived from measurement, which preset runs how fast, and what does each
stage hold and need on a 12 GiB device? Answer, in short: every parity gate holds with its fault
injections outside the band; the practical preset `f32+dit8-a8-attn8-online` runs upstream's
default request (1024² × 50 steps) in 216.3–238.0 s against 574.8 s for the reference preset
`f32+dit8`; every stage held less than its headroom need; on this device the VAE's area cap binds
the practical preset and the DiT pair binds the reference preset (§5.5, §5.6).

## 1. Setup

- Machine: RTX 3080 Ti (12,288 MiB), Node 24.22 + Dawn (npm `webgpu`, Vulkan), timestamp
  quantisation off. Every GPU run was serialised with the other sessions through
  `flock /tmp/karume-gpu.lock` after `tools/diag/gpu-busy.zsh 5` read under 10 %; the runs of §3.4
  and §4.3 also recorded no other GPU user (the display held 33 MiB). The CPU references ran with
  torch 2.13.0+cpu (6 threads, AVX512), diffusers 0.39.0 and transformers 5.14.1 in the shared
  recipe venv.
- Upstream: `OpenMOSS-Team/MOSS-Image-SFT` at revision `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`
  (`tools/export-recipes/moss/sources.py`).
- Series (`outputs/series/`, written by the recipe `tools/export-recipes/moss/`):
  `moss-image-sft-i8-dyn` (the DiT graphs `dit`, `dit_skip22`, `dit_text`; i8 linear weights with
  one f32 scale per row; the assets `rope_base` and `training_grid`), `moss-image-sft-f32-dyn` (the
  three decoder graphs, f32; the asset `latent_batch_norm`), the text-embedding asset
  `moss-image-sft-text-embeds` and the goldens `moss-image-sft-i8-dyn-reference`.
- Text: the fixed prompt `fridge-note` (T₊ = 63 tokens) and the empty negative (T₋ = 10). At 1024²
  the joint sequences are S₊ = 4,159 and S₋ = 4,106.
- Metrics: r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|) against the
  layer-by-layer CPU references (ADR 0118 decision 8, ADR 0120 decision 3); m = max|GPU − ref| ÷
  max|ref| against an f32 reference (ADR 0118 decision 8); relRMS between two GPU runs (the self A/B
  check, ADR 0110 decision 5). Bands are the worst band case × 5 rounded up to two significant
  digits (A/B ceilings: the measurement × 2, the same rounding).

## 2. The DiT on the reference path (stage 2, 2026-10-27)

### 2.1 References and goldens

- `python -m moss.export_dit write` (with the asset `training_grid`): 265.3 s, peak RSS
  28.52 GiB. Its three gates passed — eager bit identity with upstream's `forward` in 8 cases, the
  container weights (`dit` 258 i8 linears, `dit_text` 11) bit-identical to the fake-quant values,
  and the written `training_grid` (1,032,232 B) bit-identical to the stage-0 host record. Adding the
  asset changed only `dit`'s part 0 and its asset part; the asset part's first 2,622,032 bytes are
  the former one (`rope_base`).
- `moss.dit_reference` runs the graph modules on the CPU in f32 and in f64 from the container's
  dequantised weights, one block alive at a time. `text` cases about 1 s each; `small` (256²) 30–34 s
  (f32 about 10 s, f64 about 21 s), peak RSS 7.15 GiB; `full` (1024²) 303–394 s (f32 104–128 s),
  peak RSS 8.80 GiB. 30 goldens, 151 MB. The f64 output is stored as two f32 parts (writer version
  2 — rewritten in 3,173 s, peak RSS 8.82 GiB): rounded to one f32, the reference had scaled the
  denominator of r by 0.69–1.63 at `dit_text`'s outlier channel (max |value| about 1,800), where the
  CPU f32 error is only 1–2 ULP.

### 2.2 r and its bands (goldens of writer version 2)

Band cases: `dit_text` at T = 2, 10, 31, 62, 63, 134; the DiT pair at training-grid indices 0, 400,
850 × the two branches (positive `dit` with `fridge-note`, negative `dit_skip22` with the empty
negative; the same latents per index). Acceptance cases: `dit_text` at T = 1, 7, 45, 100; the pair
at indices 200 and 950.

| Group | Band-case r | Band | Acceptance r |
| --- | --- | --- | --- |
| `dit_text` (r) | 0.748, 6.59, 2.80, 11.1, 7.42, 5.75 | 11.1 × 5 → 56 | 1.15 (T = 1), 9.75, 3.64, 20.4 |
| `dit_text` (per-channel r) | 2.35, 3.58, 5.19, 6.79, 6.49, 7.23 | 7.23 × 5 → 37 | 1.93, 3.23, 4.52, 5.06 |
| 256² | positive 2.76, 1.95, 4.26 / negative 3.22, 1.51, 2.19 | 4.26 × 5 → 22 | positive 0.847, 3.00 / negative 1.98, 1.56 |
| 1024² | positive 2.21, 2.57, 3.01 / negative 4.86, 1.78, 3.09 | 4.86 × 5 → 25 | positive 5.09, 5.31 / negative 3.22, 2.38 |

The per-channel r divides each channel's error by its scale (max(its max |value|, the median
channel's)) before the maxima. With goldens of writer version 1 the product path (the pair fed the
text context that `dit_text` produced on the GPU) gave r 0.968 / 9.66 (positive) and 2.39 / 2.38
(negative) at 256², 3.98 / 7.42 and 3.28 / 3.47 at 1024². Two runs on this machine agreed to three
digits.

### 2.3 Fault injection

Each fault must land outside its band by a margin of 2 (ADR 0118's addendum). A uniform shift of
every position does not move RoPE scores (they depend on relative positions only), so the
position faults are non-uniform.

- With writer-version-1 goldens (bands 45 / 22 / 24): the next training-grid row of the time input
  r 1.27e3–8.81e3 at 256² and 964–7.54e3 at 1024²; image tokens one position further from the text
  (row T + 1 of the `f` table) r 6.55e3 / 2.77e4 at 256², 6.13e3 / 3.60e3 at 1024²; `dit_text` with
  position 2i r 2.86e4–1.88e5, the last embedding row replaced by the one before r 1.39e5–1.16e6.
- With writer-version-2 goldens: the smallest position fault, `dit_text`'s last token one position
  further, at 184–2,500 × the per-channel band and 58–590 × the band of r.
- T = 1 has one token, so neither a position fault nor a row fault can act on it; it is checked
  against the bands only.

### 2.4 The borrower and the pair's need

At 1024² the borrower `dit_skip22` (S₋ = 4,106) estimates 0 B of weights and a
`peakAccountedBytes` of 2,135,761,320 B; the lender `dit` (S₊ = 4,159) 4,849,617,932 B of weights
(4,848,316,928 B compressed) and 7,034,107,240 B. The pair's need is 10,070,775,568 B (9.38 GiB),
with the weights counted once. On the GPU the borrower's `storage.residentCompressedBytes` is 0 and
the lender's equals its estimate. Both Sessions build their plan backing once over a generation
(`planBacking.buildCount` 1 at 256² over 3 steps and over the r gate's 7 predictions).

Waiting for the release of the previous stage's buffers before the trial allocation matters on
Dawn: without the wait, a 7.9 GB pair probe right after the 10.1 GB pair of 1024² was closed was
refused on an empty 12 GiB device; with it the same sequence passed.

## 3. The VAE (stage 3, 2026-10-27)

### 3.1 Op inventory of the decoder (`moss.export_vae inspect`, symbolic h, w / S)

| Graph | conv2d | group_norm | SiLU (sigmoid + mul) | add / div | Other |
| --- | --- | --- | --- | --- | --- |
| `vae_decoder_pre` | 4 | 2 | 2 | 1 / 1 | — |
| `vae_decoder_attn` | 0 | 1 | 0 | 1 / 1 | linear 4, attention 1 (one head, D = 384), permute 6, reshape 4 |
| `vae_decoder_post` | 32 | 27 | 27 | 13 / 13 | three ×2 upsamples = expand 6 + reshape 12 |

30 GroupNorms in all (32 groups, eps 1e-6) and 36 conv2d. On the GPU the post graph's three
upsamples run as the fusion `upsample2x` (`lastRunFusions.upsample2x` = 3).
`python -m moss.export_vae write --cap`: 267 s, peak RSS 8.37 GiB; parts `vae_decoder_pre`
11,083,870 B, `vae_decoder_attn` 2,376,845 B, `vae_decoder_post` 98,424,857 B; the eager chain and
each boundary bit-identical to upstream's untiled f32 `decode` in 11 cases.

### 3.2 The `group_norm` kernel's four-point check (`gpu_group_norm_test.ts`)

Error unit e = max|GPU − f64| ÷ (2⁻²³ · (rstd_g · max_g|x| · max|w| + max|b|)).

| Point | Content | Measured |
| --- | --- | --- |
| ① f64 band | six band cases (`[2,64,9,11]`, `[1,192,40,40]`, `[1,96,128,96]` × mean 0 / +100), worst × 4 | e 1.290, 0.940, 1.117, 1.117, 1.363, 1.363 → band 5.5; acceptance (mean −50, +100, a rank-3 +3) 1.258, 1.002, 1.277; variance of the order of eps 0.927, 1.055; the summation depth's bound ≈ 54 |
| ② A/B against the decomposition | reshape → layer_norm → reshape → mul → add on the same GPU | 0.490–1.702 (9 cases) → band 7 |
| ③ structure | two runs; a device lowered to 3 workgroups per dimension; a zero-variance group; NaN / +Inf | Uint32-identical; exactly the bias; poison stays in its own group |
| ④ faults | one-pass variance (mean +100), the last partial dropped, another group's affine, eps outside the square root and eps × 10 (variance ≈ eps) | e 226, 3.55e5, 5.61e6, 4.67e6, 4.24e6 (all outside 2 × 5.5) |
| estimate | a lone `group_norm` Session's plan backing against the runtime estimate's workspace | equal (two temporaries included) |
| large group | `[1,192,1696,1648]`: 16,770,048 elements per group, 524,064 partial workgroups (the grid-stride wraps 8 times at 65,535) | e 1.476, about 7 s |

The VAE's upstream band below does not see a one-pass variance: with that fault in the kernel the
decoder's m stays 3.0e-6–9.2e-6 (random latents keep the GroupNorm inputs out of the cancelling
range), so ① and ④ alone hold that property.

### 3.3 Against upstream's untiled f32 decode (`e2e_moss_vae_test.ts`, opt-in)

m of the clamped images (no golden pixel saturates; max |decoded| 1.042):

| Size | Band-case m | Band | Acceptance m |
| --- | --- | --- | --- |
| 256² | 3.878e-6, 3.054e-6, 3.018e-6 | 1.939e-5 → 2.0e-5 | 3.567e-6, 4.489e-6 |
| 512×768 | 6.385e-6, 6.586e-6, 5.722e-6 | 3.293e-5 → 3.3e-5 | 7.978e-6, 5.005e-6 |

relRMS 9.5e-7–1.0e-6 in every case; two runs agreed in every digit. Faults (BatchNorm mean / std
swapped, another seed's latents, the unpatchify's 2×2 transposed) m 1.529–1.936. At the area cap
(1696×1648, opt-in `KARUME_MOSS_VAE_CAP=1`) m 1.272e-5, relRMS 9.77e-7, decode 2.6 s.

### 3.4 At the area cap before the row-block budget

The cap of a 2 GiB binding: 49,152 B per VAE cell (`[1,192,8h,8w]` f32 — 768 B per pixel), so
⌊2,147,483,644 ÷ 49,152⌋ × 64 = 2,796,160 pixels. At 1696×1648 (sampled every 50 ms with
`nvidia-smi` during the decode; no other GPU user, the display held 33 MiB):

| Item | Value |
| --- | --- |
| Process peak (`nvidia-smi`) | 8,709 MiB (device 8,769 MiB) |
| Runtime-accounted held bytes | 9,096,149,120 B = plan backings pre 201,306,368 + attn 2,175,651,696 + post 6,439,698,688, weights 111,791,888, residents 167,700,480 (9,101,739,136 B with the pre input buffer in the re-run) |
| post's backing per pixel | 2,304 B (the full-resolution resnets hold the input, the GroupNorm output and the SiLU output — 576 channels) |
| Hand count before the run | 5.37 GB (1,920 B per pixel) — it left out the attention Session's backing (the three Sessions are open together) and the GroupNorm output |
| Headroom need | 12,252,394,824 B (11.41 GiB): estimates pre 352,065,412, attn 2,312,180,596, post 8,785,231,120 (post's workspace 3,072 B per pixel — the unfused sigmoid output counted) + residents + part + margin |
| Trial allocation | admitted on the idle device (about 0.5 GiB to spare); it occupied 11,695 MiB for a moment |

The largest accepted area (2,796,032 px, 688×4064) decoded finite through the product's own
check (probe 11,734 MiB, held 8,713 MiB, 2.5 s). Raw samples:
`outputs/bench/karume-moss-image/2026-10-27_vae-cap-vram/`.

## 4. The sampler and end to end (stage 4, 2026-10-27)

### 4.1 The σ schedule

- The requested grid transcribes `torch.linspace` with **one** rounding per element (the CPU
  kernel of torch 2.13.0+cpu contracts the product and the sum into an FMA — an implementation
  fact, not a torch guarantee). It is Uint32-identical to the stage-0 record at all 202 recorded
  counts (1–200, 500, 1,000); with two roundings 183 of the 202 grids differ in some last bit, but
  the snapped schedule is the same for every count 1–1,000 either way. Snapping, de-duplication, σ
  and timesteps are Uint32-identical at all 202 counts (500 → 415 steps, 1,000 → 615).
- After the stage-4 review the host record gained eleven shifts (0.1, 0.35, 0.9999, 1, 1.0001,
  2.2, 3, 3.7, 7.25, 12, 50) × eight counts (2, 7, 30, 50, 77, 199, 1,024, 5,000) and upstream's
  CFG line at nine guidance scales (one step of `__call__` on a 4-block DiT); all Uint32-identical in
  TS. A double-precision scale would differ in 85–4,384 of 16,384 elements.
- Large step counts: the number of snapped indices stays at most the 1,000 training points and
  monotonic up to about 2·10⁶ requested steps (977 at shift 5). At 5·10⁶ there are 989 indices
  with 6 decreases, at 10⁷ 1,099 with 61 (the σ column is not monotonic; some steps have dt > 0).
  Upstream's own `set_timesteps` gives the same indices at 12 counts from 10⁴ to 2·10⁷. The host
  loop is linear in the count, about 0.25 s at 10⁷.
- A shift below about 2.98·10⁻⁸ makes `f32(shift − 1)` −1 and the first requested σ infinite;
  upstream snaps it to the end of the grid (2 steps at shift 10⁻⁹ give σ = [0.999, 0.001]).

### 4.2 Two steps against upstream's `pipe(...)` (`e2e_moss_generate_test.ts`, opt-in)

The reference is upstream's pipeline on the CPU in f32 with the DiT's linear weights rounded to the
series' i8 values (checked bit-identical to the containers: `dit` 258 i8 + 174 f32, `dit_text`
11 + 19) and upstream's f32 VAE (`moss.few_step_ref`, 304.5 s, peak RSS 27.86 GiB). Band cases seed
20261201–03, acceptance 20261204–05:

| Size (W×H) | Point | Band-case m | Band | Acceptance m |
| --- | --- | --- | --- | --- |
| 256² | latents | 9.643e-6, 9.444e-6, 1.054e-5 | 5.270e-5 → 5.3e-5 | 1.846e-5, 1.156e-5 |
| 256² | image | 2.571e-5, 1.749e-5, 1.776e-5 | 1.286e-4 → 1.3e-4 | 1.735e-5, 1.481e-5 |
| 384×256 | latents | 1.821e-5, 2.204e-5, 1.082e-5 | 1.102e-4 → 1.2e-4 | 1.194e-5, 8.928e-6 |
| 384×256 | image | 9.288e-6, 1.242e-5, 1.291e-5 | 6.455e-5 → 6.5e-5 | 7.544e-6, 4.580e-6 |

Faults on the first acceptance case: the prompts swapped 1.56–2.24, guidance 3.3 7.6e-2–1.1e-1,
shift 3 1.7e-2–3.6e-2, another seed's latents 0.25–0.41, the first step at the next training-grid
point 1.255e-1 / 1.764e-1 (256²) and 2.074e-1 / 1.077e-1 (384×256), guidance 3.003 9.751e-4 /
1.060e-3 and 9.556e-4 / 8.329e-4 — 8.0–18.4 × the band, the fault closest to it. The band cases
and the acceptance cases were measured in the same run.

### 4.3 The 1024² × 50-step reference run (opt-in `KARUME_MOSS_FULL_PIPELINE=1`)

Seed 42, `fridge-note`, guidance 3, shift 5, `f32+dit8`; no other GPU user (the display held
33 MiB). Each Session's open and dispose marks the
intervals; `nvidia-smi` sampled every 50 ms, and a rerun wrapped `device.createBuffer` / `destroy`
to log the WebGPU buffers exactly (`postfix_*`, 573.8 s):

| Interval | Time | Process peak | Held (buffer log) |
| --- | --- | --- | --- |
| first check (pair need 10,070,775,568 B) + `dit_text` | 0.97 s + 0.06 s | 9,615 MiB (probe) → 1,236 MiB | 250,626,824 B |
| DiT pair check + creation (4.85 GB of weights uploaded) | 5.13 s | 9,633 MiB (probe) | — |
| 50 steps (100 forwards) | 564.1 s (11.3 s per step) | 8,792 MiB | 9,167,768,808 B (8,743 MiB) |
| VAE check (need 5,562,198,232 B) + creation | 0.39 s | 5,383 MiB (probe) | — |
| VAE decode | 0.73 s | 3,726 MiB | 3,855,364,144 B (3,677 MiB) |
| whole generation | 571.4 s | — | 0 B left after it |

A probe is the stage's need itself (need + the device's base), not what the stage uses; every
probe ran with 0 B held. The DiT stage's free memory measured by what it held is about 3.4 GiB
(12,288 − 8,792 MiB). Raw samples: `outputs/bench/karume-moss-image/2026-10-27_generate-1024-vram/`.

## 5. Presets (stage 5, 2026-10-27 to 2026-10-28)

### 5.1 The online kernel's key valid length

The checks are those of the ADR 0132 addendum of 2026-10-27: N = 65 / 4,106 / 203 (D 64 / 128 /
128, batch 2 / 2 / 3) pass the four-point check, using 34.6 % / 3.6 % / 20.4 % of the f64 error
model on dense inputs; padding bytes 0 and 127 give Uint32-identical outputs; the mask bound moved
to `ceil(N/4)·4` is red at 591.8 / 34.1 / 175.6 × the model. Two spike-V inputs see a wrong row
length (`floor(N/4)`: 613.4 / 249.6 / 461.0 ×) and a dropped last word (252.4 / 86.3 / 252.7 ×) at
every padded shape, which the dense inputs at N = 4,106 did not (1.33 × and 0.15 ×). The existing
codegen snapshots did not move; the Wan2.2 seed-path rows (7) and every Anima row compared equal;
the core lane passed (256 files, 3,583 passed / 1 skipped). The Wan2.2 golden S = 8,190 now runs
online (1 submit at most 151.1 ms).

### 5.2 Self A/B check (`e2e_moss_ab_test.ts`)

Request: `fridge-note` / empty negative, seed 42, guidance 3, shift 5, 50 steps. Step 1 starts at
σ = 1 (training-grid index 0) at every step count, so the branch outputs do not depend on the
count (2 and 50 steps gave relRMS identical in every digit). At 50 steps it goes to σ 0.996; at 2
steps it goes to σ 0.024 and the latent after it is nearly the x₀ prediction, which another seed
moves only 1.5423e-1 relRMS. Observation points: step 1's `dit`
output (positive head rows), `dit_skip22` output (negative) and the latent after step 1.

| Size | Point | relRMS | maxAbs | Ceiling |
| --- | --- | --- | --- | --- |
| 1024² | `dit` | 1.8482e-2 | 1.1560e-1 | 3.7e-2 |
| 1024² | `dit_skip22` | 1.5683e-2 | 8.5594e-2 | 3.2e-2 |
| 1024² | latent | 2.4677e-4 | 1.3812e-3 | 5.0e-4 |
| 1696×1648 (opt-in `KARUME_MOSS_AB_CAP=1`) | `dit` | 1.7865e-2 | 9.6400e-2 | 3.6e-2 |
| 1696×1648 | `dit_skip22` | 1.6198e-2 | 9.7922e-2 | 3.3e-2 |
| 1696×1648 | latent | 2.3550e-4 | 1.1912e-3 | 4.8e-4 |

- Census: i8a8 linears exactly 258 (`dit`) / 252 (`dit_skip22`) — the four M = 1 linears (time
  MLP 2, modulation 2) run i8a8 too; online dispatches exactly the attention nodes (42 / 41) × the
  query-row windows (1 up to the cap with these texts), none of the three dispatches. At 1024²
  (S₊ 4,159 / S₋ 4,106) all 42 + 41 online keys carry `:kpad`; at 1696×1648 (S₊ 10,981 / S₋
  10,928) the positive 42 carry `:kpad` and the negative 41 are whole. `dit_text` is bit-identical
  between the presets with no knob variant in its census.
- Determinism: two practical runs bit-identical at every point (both sizes).
- Faults (1024²): no knob → relRMS 0 (the floor); another seed (43) → determinism red and the
  ceilings red (`dit` 1.3566, `dit_skip22` 1.3910, latent 1.4137 — 37–2,827 × the ceilings);
  `attentionScoreStorage` dropped (the three-dispatch a8 attention, S and ③PV f32 at these S) →
  census red while the bands stay inside (1.6550e-2 / 1.5294e-2 / 2.1969e-4 — 89–98 % of the
  bundle's); the linear or attention node count ± 1 → census red.
- Single knobs: without `linearCompute` (a8 attention and online kept) the difference is 54 %
  (`dit` 1.0032e-2), 28 % (`dit_skip22` 4.3333e-3) and 53 % (latent 1.3191e-4) of the bundle's;
  linear a8 alone 89 % / 97 % / 89 % (1.6467e-2 / 1.5177e-2 / 2.1866e-4). In quadrature the linear
  a8 is most of the difference (√(1.6467² + 1.0032²) = 1.93 against 1.85, in units of 1e-2), but
  the positive branch's attention part is not negligible.
- Order of magnitude against theory: the a8 row rounding (peak/rms)/(127·√12), which ADR 0025's
  per-layer w8a8 measurement matched at a median of 1.8 %, gives per-layer relRMS of the 1e-2
  order, as the branches' 1.6–1.8e-2; MOSS's per-layer peak/rms was not measured.
- The 50-step latent after step 1 is a dependent point: its difference is the CFG of the branches'
  differences scaled by |dt| = 0.004 (2.5e-4 predicted, 2.4677e-4 measured), and another seed is red
  there by the two noises' distance (1.4137 ≈ √2). At 2 steps the practical preset moves the x₀
  side by 9.0852e-2 (59 % of another seed's 1.5423e-1; 9.4005e-2 at the cap) — recorded for the
  visual review, not checked.
- The cap's reference side cannot open its pair there (§5.5), so its observation alone runs with
  the pair's estimate replaced by 0 (the practical side goes through the product's check). Runs:
  6 passed in 158 s (2026-10-28), twice more after the review with relRMS identical in every digit.

### 5.3 One forward before and after the key valid length (1024², practical preset)

Wall time of `Session.run`, the median and minimum of 7 runs after one warm-up run. "Before" is a
copy of the tree with only the online eligibility reverted to `N % 4 == 0` (its census: the three
dispatches `attention_qk:…:i8a8`, `attention_stats:…:f32`, `attention_pv:…:f32`); "after" runs
`attention_online:…:sg32:obshd:kpad` on both branches.

| Tree | Branch | Run 1 (median / min) | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| before | `dit` (S₊ 4,159) | 2,212 / 2,170 ms | 2,182 / 2,131 | 2,176 / 2,143 |
| before | `dit_skip22` (S₋ 4,106) | 2,169 / 2,101 | 2,120 / 2,065 | 2,107 / 2,072 |
| after | `dit` | 1,667 / 1,655 | 1,694 / 1,676 | 1,697 / 1,679 |
| after | `dit_skip22` | 1,623 / 1,608 | 1,652 / 1,629 | 1,663 / 1,631 |

About −22 % at the median and the minimum. Run 1 followed directly (after: from 33 °C, before:
from 63 °C); runs 3 and 4 waited for 50 °C. Run 2 started hot (84 °C) without cooling and hit this
machine's slow state (before 3,518 / 2,395 and 3,402 / 2,349 ms, after 2,323 / 1,839 and
2,450 / 1,783 ms — medians far above the minima), so it is excluded. In measured mode (three runs
after a warm-up) the attention dispatches with every `quantize_rows` — about 46 ms of it the q / k
and the linears' activations, the same in both trees — took 754–772 → 323–344 ms of a `dit`
forward and 726–764 → 321–333 ms of a `dit_skip22` forward.

### 5.4 Upstream's default request (1024² × 50 steps, one sample each)

In one process (`KARUME_MOSS_FULL_PIPELINE=1`, 2026-10-28 01:11–01:25): `f32+dit8` 574.8 s (the
DiT stage 568.1 s, 11.4 s per step), then `f32+dit8-a8-attn8-online` 238.0 s (231.4 s, 4.6 s per
step) — −59 %. One more practical run started at 33 °C: 216.3 s (−62 %). Per step that is
4.2–4.6 s against the 3.3–3.4 s of the two cooled forwards of §5.3; the rest of the gap is not
decomposed (speculation: the host's per-step work and the GPU warming during 50 steps). The
reference run matches stage 4's 571.0–573.8 s.

### 5.5 VRAM per preset and size

The peak of the WebGPU buffers the process held per stage, beside each stage's need (estimate +
largest part + margin). "Row-block budget" = the VAE mid attention's 256 MiB budget (§5.7). The
DiT-pair and with-budget columns come from the generate test's buffer log (1024² at 50 steps with
`KARUME_MOSS_FULL_PIPELINE=1`; 1456² and 1696×1648 at 2 steps with `KARUME_MOSS_VRAM=1`;
2026-10-28 01:11–01:27). The without-budget column comes from a scratch script with the same buffer
accounting, at 2 steps for every size (00:22–00:26).

| Preset | Size | DiT pair held / need | VAE with the budget: held / need | VAE without it: held / need |
| --- | --- | --- | --- | --- |
| `f32+dit8` | 1024² | 9,167,768,808 / 10,070,775,568 B | 3,049,959,696 / 4,756,793,560 B | 3,855,364,144 / 5,562,198,232 B |
| `f32+dit8` | 1456² | 10,261,736,896 / 11,269,763,608 B | 5,767,450,392 / 8,294,919,952 B | 6,972,308,368 / 9,499,778,824 B |
| `f32+dit8` | 1696×1648 | refused at the first check (pair need 12,519,585,664 B = 11,940 MiB); with the probe bypassed it held 11,445,429,304 B (10,915 MiB) over 3 steps | — | — |
| `f32+dit8-a8-attn8-online` | 1024² | 6,430,398,328 / 7,333,404,976 B | 3,049,959,696 / 4,756,793,560 B | 3,855,364,144 / 5,562,198,232 B |
| `f32+dit8-a8-attn8-online` | 1456² | 8,031,016,512 / 9,039,043,576 B | 5,767,450,392 / 8,294,919,952 B | 6,972,308,368 / 9,499,778,824 B |
| `f32+dit8-a8-attn8-online` | 1696×1648 | 9,039,595,176 / 10,113,779,296 B | 7,491,042,096 / 10,608,156,072 B | 9,135,279,232 / 12,252,394,824 B |

Every stage held less than its need; the practical pair holds 0.9–1.1 GB under its need, so the
margin covers the i8a8 linears' quantised intermediates the estimate leaves out. The text stage
held 250,626,824 B against 1,284,624,392 B. 1456² is upstream's 2M-pixel example (the GitHub
README). Without the device facts the practical pair's estimate would count the three-dispatch S:
10,070,775,568 B instead of 7,333,404,976 B at 1024², 12,519,585,664 B instead of 10,113,779,296 B
at the cap. The VAE columns count the process's buffers; the runtime's own account of the VAE
stage (plan backings, inputs, weights, residents) is 33,540,096 B lower at the cap
(9,101,739,136 / 7,457,502,000 B without / with the budget).

### 5.6 The largest squares tried for the DiT pair

The DiT stage alone (3 steps, `--dit-only`), one device at one time, each a pass or a refusal of
the stage's own trial allocation. The need is not monotonic in S (the row blocks), so a passing
size says nothing about the smaller ones.

| Preset | Passed (probe / held) | Refused (probe) |
| --- | --- | --- |
| `f32+dit8-a8-attn8-online` | 1984² (11,378 / 10,247 MiB), 2048² (11,770 / 10,615 MiB), 2064² (11,870 / 10,708 MiB) | 2112² (12,174 MiB) |
| `f32+dit8` | 1536² (11,747 / 10,763 MiB), 1568² (11,334 / 10,341 MiB), 1600² (11,723 / 10,721 MiB), 1664² (11,845 / 10,823 MiB) | 1616² (11,926 MiB), 1632² (12,133 MiB), 1696×1648 (11,940 MiB), 1696² (12,239 MiB), 1728² (12,055 MiB) |

The idle device admitted every probe up to 11,870 MiB and refused every one from 11,926 MiB (over
two sessions). Every practical size that passed is above the VAE's area cap (2,796,160 px), so the
cap binds that preset's generations; the reference preset is refused at the cap and at some areas
below it, so its DiT pair binds first. 1696² and 1728² lie above the area cap too.

### 5.7 The VAE mid attention's row-block budget (256 MiB)

With the practical preset the VAE was the largest stage need at the cap (12,252,394,824 B against
the pair's 10,113,779,296 B; the idle device admitted it with about 0.5 GiB to spare). With
`attentionRowBlockMaxBytes` 256 MiB on the `vae_decoder_attn` Session and in its estimate:

- the need at the cap 12,252,394,824 → 10,608,156,072 B (11.41 → 9.88 GiB), the attention's row
  blocks 4 → 29, the runtime-accounted hold 9,101,739,136 → 7,457,502,000 B;
- the values do not change: the cap's image Uint32-identical with and without the budget (two runs
  each); m against upstream still 1.272e-5 with relRMS 9.773e-7; the 256² / 512×768 bands and
  acceptance m unchanged in every digit (below the budget); the reference preset's 1024² × 50-step
  sha row (row blocks 1 → 4) compared equal; 1024² decodes bit-identical with and without it
  (12 attention dispatches against 3);
- decode at the cap 2.4 s with the budget, 2.4–2.5 s without (same process). One parity run gave
  18.5 s for the cap's decode; the same case alone gave 2.3 s with the same m, relRMS and held bytes
  (this machine's transient slow state);
- below about 8,200 attention positions (an S × S f32 score of 256 MiB, about 0.52M px) the budget
  is not reached. The reference preset's limit does not move (its pair binds).

### 5.8 sha256 rows

`packages/models/tests/fixtures/references/moss.json`, environment key
`node-nvidia-geforce-rtx-3080-ti` only. Written with `KARUME_UPSTREAM_PARITY=1 KARUME_REFERENCE=write`
after the parity cases of the same run passed; the practical rows after its A/B run had passed.

| Row | Lane | sha256 |
| --- | --- | --- |
| `f32+dit8-2step-fridge-note-seed42-256x256-g3-shift5` | default | `6f5685a5da162144aadbfae443085e3f516b90aa153e8d62be709548a90b76ba` |
| `f32+dit8-2step-fridge-note-seed42-384x256-g3-shift5` | default | `c732dddb3653aa5a6abf16c3335aca1b192486b61d091a7b86154ed150e8792b` |
| `f32+dit8-2step-fridge-note-seed42-256x256-g1-shift5` | default | `2366d8663efbf0b08d12fc53dc31b81469914b612882c7edbd93bda5077f51f4` |
| `i8-dit_text-fridge-note-t1` | default | `2869acc9169f102399bf571cade1e54eb30d5bb32abeec2aecc3402b2f9899bb` |
| `f32+dit8-50step-fridge-note-seed42-1024x1024-g3-shift5` | opt-in | `052f093ad3446d3f94818bc1a6a3f4774a15448eb0a8b300674c7a05b0b81a30` |
| `f32+dit8-a8-attn8-online-2step-fridge-note-seed42-256x256-g3-shift5` | default | `46c0cfe14667e4310c5f722660a99e366e6372f0a29c6a7b98dc31e60071c80a` |
| `f32+dit8-a8-attn8-online-2step-fridge-note-seed42-384x256-g3-shift5` | default | `112562caced0b04def9662a194cd8599c53f858ecedc6b206b5502d5de082464` |
| `f32+dit8-a8-attn8-online-50step-fridge-note-seed42-1024x1024-g3-shift5` | opt-in | `e2845c3cbf12c82938d55cca95505357dfc52db4e86008ce6ff90a31b2ad2ac9` |

The reference rows were first written in stage 4 as `i8-…` and renamed with values unchanged (the
renaming run compared them equal). The images are the clamped f32 `[3, H, W]`; the `dit_text` row is
its output `[1,1,3072]` at T = 1.

## 6. Per-op profile of one practical forward at 1024² (2026-10-28)

`f32+dit8-a8-attn8-online`, measured mode (`gpuTiming`), three runs per graph after the text stage,
one process at 01:29 (54 °C at the start). By family, the minimum over the three runs, in ms:

| Graph | Total | linear (i8a8) | attention_online | ew | strided | quantize_rows | rms_norm | strided_write | transpose_tiled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dit` | 1,666.9 | 1,021.2 | 252.4 | 237.0 | 52.8 | 50.9 | 34.4 | 11.8 | 5.6 |
| `dit_skip22` | 1,613.5 | 992.2 | 243.0 | 228.4 | 50.9 | 49.3 | 33.0 | 11.4 | 5.3 |

An earlier process (01:28) gave 1,668.9 / 1,618.8 ms in total (`dit` linear 1,024.5, attention
251.8, ew 237.0, strided 52.8, quantize_rows 51.0, rms_norm 34.3).

The `dit` forward by pipeline key (the last of the three runs):

| Key (abridged) | Dispatches | ms |
| --- | ---: | ---: |
| `linear:v4:i8a8:tile64x64r8x4w16x8k16v4:dp4a` | 258 | 1,023.8 |
| `attention_online:v2:i8a8:d128:r64k64:dp4a:sg32:obshd:kpad` | 42 | 253.2 |
| `ew mul r3` | 337 | 92.0 |
| `strided r4` | 346 | 52.9 |
| `quantize_rows pertoken` | 256 | 36.7 |
| `ew tanh r3` | 84 | 28.8 |
| `ew div r3` | 84 | 28.6 |
| `ew add r3` | 172 | 27.1 |
| `ew mul r4` | 168 | 24.7 |
| `rms_norm lastdim wg128` (per head) | 84 | 19.5 |
| `ew add r4` | 84 | 15.5 |
| `rms_norm lastdim wg256` | 85 | 15.0 |
| `ew sigmoid r3` | 42 | 14.1 |
| `strided_write r4` | 168 | 11.8 |
| `quantize_rows pertoken r8w32` (q / k) | 85 | 9.5 |
| `ew neg r4` | 84 | 6.0 |
| `transpose_tiled t32x33` | 42 | 5.6 |
| `quantize_rows pertoken r1w256:pad` (V) | 42 | 4.7 |

Fusion census of `dit` / `dit_skip22`: `silu` 1 / 1, `attentionPermute` 84 / 82, `rope` 0,
`rmsNormRope` 0, `adaln` 0, `gatedResidual` 0. By element count, about 67 ms of the 92.0 ms of
rank-3 muls are `situ`'s four multiplies over `[S, 8192]` per block (168 of the 337, the rest over
`[S, 3072]`), and the RoPE's slices are about 26 ms of the 52.9 ms of strided copies (168 of 346).

`vae_decoder_post` alone at 1024² (input `[1, 384, 128, 128]`, the same process, minimum of three
runs): 534.1 ms — conv2d 474.7 (`igemm64x128v4` ×24 349.6, `igemm32x128v4` ×8 130.9 in the last
run), group_norm 28.0 (partial 14.5, apply 13.0, total 0.5), ew 13.6, silu 13.0 (27 fused
`silu`), upsample2x 2.0 (3 fused).

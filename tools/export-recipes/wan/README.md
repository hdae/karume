# Wan2.1 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)). The design is ADR
[0118](../../../docs/decisions/0118-wan21-video-generation.md). Scripts are started as modules from
`tools/export-recipes/` (`uv run --group wan python -m wan.<module> …`).

## Overview

Text-to-video with Wan2.1 T2V 1.3B: the DiT (S form, one symbol for the token length), the video VAE
decoder (two chunk graphs with the causal cache as resident tensors) and host-side reference
pipelines. The reference is diffusers 0.39.0 `WanPipeline` on CPU f32, built with
`text_encoder=None` / `tokenizer=None` and fed precomputed text embeddings (`wan/pipeline_ref.py`).

```bash
uv sync --group wan --inexact   # accelerate / diffusers==0.39.0 / ftfy==6.3.1 / huggingface-hub / transformers==5.14.1
uv run --group wan python -m wan.pipeline_ref --smoke   # load + one CFG step + VAE decode
```

`--inexact` keeps the other families' groups installed in the shared venv (a plain `uv sync --group
wan` removes them).

## Sources

The upstream repo and its pinned 40-hex revision live in `wan/sources.py` (`SOURCES`). Only the
`transformer` (about 5.7 GB, fp32), `vae` (about 0.5 GB) and `scheduler` subfolders are fetched; the
umT5-XXL text encoder (about 22.7 GB fp32) and its tokenizer are not.

```bash
uv run --group wan python -m wan.sources --fetch   # download the pinned revision into the HF cache
uv run --group wan python -m wan.sources           # check the cached snapshot (parameter counts)
```

## DiT (stage 2)

The DiT graph is the **token form** (S form, ADR 0034): it starts after the patchify and ends before
the unpatchify, so every value is written with one symbol, the token length `S`
(`Dim("S", max=32760)` = 81 frames). `wan/dit_patch.py` holds the wrapper and the host-side
reference functions; the host steps are mirrored in TypeScript under `packages/models/src/wan/`.

| Step                            | Where | How                                                                                                                                        |
| ------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| patchify + patch embedding      | host  | latent `[16,F,H,W]` → tokens `[1,S,64]` in `(c, pt, ph, pw)` order; `Conv3d(k=s=(1,2,2))` → `Linear(64→1536)` on a view of the conv weight |
| sinusoidal timestep (256)       | host  | graph input `timesteps_proj [1,256]` (cos first); the MLP stays in the graph                                                               |
| 3D RoPE tables                  | host  | reordered from the per-axis base tables in the container asset `rope_base` (role `rope-base`) into `rope_cos` / `rope_sin [1,S,1,128]`     |
| interleaved RoPE application    | graph | rewritten as "swap adjacent pairs + elementwise cos / sin" at rank 4 (bit-exact)                                                           |
| adaLN, qk-norm, cross-attention | graph | upstream as is; the 512-row text context is projected without a mask                                                                       |
| unpatchify                      | host  | output tokens `[1,S,64]` are in `(pt, ph, pw, c)` order — not the patchify order                                                           |

```bash
uv run --group wan --inexact python -m wan.export_dit            # f16 graph + golden → outputs/series/wan2.1-t2v-1.3b-f16-dyn/transformer/
uv run --group wan --inexact python -m wan.export_dit --layers   # per-block outputs (measurement only) → …-f16-dyn-probe/transformer/
uv run --group wan --inexact python -m wan.export_dit --verify   # eager equivalence of the patches only
uv run --group wan --inexact python -m wan.dit_host_fixture      # TS host fixture → packages/models/tests/fixtures/wan-dit/
```

Weights are rounded to f16-representable values before any reference is taken (ADR 0006); the
rounding goes through the wrapper, so the upstream modules it shares get the same values while the
RoPE buffers of `model.rope` stay f32. The golden for each case is two files: `io.<case>` (graph
inputs and the patched torch output) and `reference.<case>` (the input latent, the timestep, the
unpatched upstream output and each block's output from the plain diffusers forward).

The real-GPU comparison is `packages/models/tests/e2e_wan_dit_test.ts`: the band is about 5× the
worst of six decision cases spread over the sampling schedule, and three separate acceptance cases
and four fault injections (RoPE h/w swap, unpatchify order, flipped timestep halves, timestep off by
one) check it.

Stage 3 adds eight real-size cases (832×480, 33 frames: latent `[16,9,60,104]`, S = 14,040). Six
`full-band` cases spread over the schedule (t = 999 with two seeds, 750, 500, 250, 113) set a
separate band, and two `full-accept` cases (t = 999 and 600, seeds 777006 / 777007, generated after
the band was fixed) are checked against it. Each real-size case has two references: a float64 one
(`dit_patch.reference_dit_f64` runs the same plain upstream forward with the activations in float64
and the same f16-rounded weights; stored as `output.f64` in `reference.<case>`, rounded to f32)
and the CPU f32 one (`output`). The metric is a normalized ratio,
r = (max|GPU − f64| / max|f64|) / (max|CPU f32 − f64| / max|f64|): how many times the CPU f32
reference's own error the GPU error is. The plain ratio moves 240× with the input, because inputs
that amplify rounding pull the CPU f32 reference away from float64 just as much; dividing by the CPU
f32 error cancels that, and r stays between 1.6 and 15 on the decision cases. The band is 75 (5×
the worst decision r, 14.9). r above 1 is a precision difference, not a porting bug: the GEMM kernel
reduces K in one f32 accumulator in ascending order. The four fault injections must land outside
the band, and the off-by-one timestep at least 2× the band (observed 4.4× to 80×; the fault itself
moves about 3× with the input).

Only `full-band-s14040-t0999` and `full-accept-s14040-t0999` keep the 30 block outputs (2.6 GB
each) for the per-layer record; the other real-size cases hold the final outputs only. On a 6-core
desktop CPU (2026-10-02) one real-size case takes about 370 s for the float64 forward and 140 s
each for the f32 and the patched ones, so the whole command takes about 88 minutes and writes about
6.2 GB of golden files (plus the container).

As of 2026-10-02 both acceptance cases are inside the real-size band (r = 2.93 and 4.60) and all
four fault injections are outside it, but the off-by-one timestep at t = 600 lands at r = 332, only
4.4× the band, so the real-size comparison is red until that is resolved (the NOTE on
`DIT_FULL_NORMALIZED_BAND` in the e2e test has the numbers).

## VAE (stage 4)

The upstream decode runs the decoder one latent frame at a time and carries a causal cache
(`feat_cache`) in a Python list with a `'Rep'` sentinel. `wan/vae_patch.py` turns that into two
static **chunk graphs** on unbatched rank-4 tensors `[C, T, H, W]` (ADR 0118 decision 2):

| Graph               | Inputs                              | Outputs                                             |
| ------------------- | ----------------------------------- | --------------------------------------------------- |
| `vae_decoder_first` | `latent [16,1,t,t]` + 30 `cache_NN` | frame `[3,1,8t,8t]` (before clamp) + the 30 caches  |
| `vae_decoder_next`  | `latent [16,1,t,t]` + 32 `cache_NN` | frames `[3,4,8t,8t]` (before clamp) + the 32 caches |

- **Cache normalization.** Every cache is always 2 frames, starts at zero and is updated to the last
  2 frames of `cat(cache, x)`. Each time-kernel-3 causal conv becomes `conv3d(cat(cache, x))` with
  zero time padding (spatial padding goes into the conv3d attrs). The zeros the upstream code pads
  with and the zero cache are the same values, so the conv inputs match element by element; the two
  `time_conv` caches (`cache_11`, `cache_18`) are not touched by `first` and stay zero for the first
  `next` chunk, which is the upstream `'Rep'` path.
- **Output order.** Output `1 + k` is the updated value of cache input `k` (the host pairs them by
  position; `assert_chunk_graph` checks that each one is sliced from `cat(cache_k, …)`).
- **Other rewrites.** RMS_norm as `sum → sqrt → clamp_min → div` on the channel axis, the
  `upsample3d` time interleave (rank 6 upstream) as rank-4 reshape / permute, and per-frame 2D work
  (nearest ×2 + conv2d, mid attention) with the frames folded into the batch by a rank-4 permute.
- **Host side.** `z·std + mean` before the graph and `clamp(-1, 1)` after the tile blend stay on the
  host. The tile side `t` is an export argument (default 32 = diffusers' 256 px tile); the host
  reads it back from the asset's `latent` input.

```bash
uv run --group wan --inexact python -m wan.export_vae            # both graphs + chunk fixtures (f16)
uv run --group wan --inexact python -m wan.export_vae --verify   # eager equivalence on the real weights
```

The graphs go to `outputs/series/wan2.1-t2v-1.3b-f16-dyn/{vae_decoder_first,vae_decoder_next}/`
(f16 storage; the weights are rounded to f16 before the references are taken). The series root also
gets `vae_chunks.{band,accept}.safetensors`: a seeded de-normalized latent (9 and 5 chunks) and the
upstream non-tiled `_decode` chunk loop before its clamp. `band` sets the GPU tolerance and
`accept` (another latent, other chunk boundaries) is the one judged against it.

Measured on 2026-10-02 (tile 32):

| Check                                                 | Result                                         |
| ----------------------------------------------------- | ---------------------------------------------- |
| cache normalization only (upstream code) vs `_decode` | bit-exact (9 chunks)                           |
| rewritten chunk graphs (eager) vs upstream            | max abs 3.90e-6 / reference max 1.40 = 2.79e-6 |
| `first` / `next` graph                                | 445 / 457 nodes, 143,195,654 / 146,744,483 B   |
| GPU (B570, Deno) chunk loop vs upstream, `band`       | max abs 4.65e-6 (ratio 3.58e-6)                |
| GPU chunk loop vs upstream, `accept`                  | max abs 3.55e-6 (ratio 2.77e-6)                |

The GPU check is `packages/models/tests/e2e_wan_vae_chunks_test.ts` (one tile = one batch, only the
frames are read back; fault injections must leave the tolerance).

### Tiled decode (stage 5)

The chunk graphs take a fixed `t×t` latent tile, so the host tiles the full frame: tiles outside,
chunks inside (the cache is zeroed again at every tile, like the upstream `tiled_decode`).
`wan/vae_tiling.py` is the Python side of the geometry and writes the references; the TypeScript
side is `packages/models/src/wan/vae-tiles.ts`.

- **Geometry.** Rounded equal spacing with the last tile snapped to `extent − tile` (the Anima rule,
  ADR 0033), using the fewest tiles whose neighbours overlap by at least 8 latents (64 px — the
  upstream default blend, 256 − 192). The upstream `range(0, H, stride)` walk leaves a short last
  tile, which a fixed-shape graph cannot take. 832×480 (latent 60×104) is 3 × 4 = 12 tiles: rows
  start at 0 / 14 / 28 and columns at 0 / 24 / 48 / 72 (overlaps of 144 px and 64 px; 1.97× the
  untiled area). The tile side and the scale are read from the asset's declared shapes.
- **Blend and paste.** `blend_v` / `blend_h` are copied verbatim from `AutoencoderKLWan` (vertical
  first, in place, over every frame). Tile `i` pastes the region `[starts[i], starts[i+1])`, the
  last one up to the edge. `clamp(-1, 1)` comes after the paste.
- **Frozen plan.** The tile starts are pinned as literal tables on both sides (`MIRRORED_STARTS` in
  `wan/tests/test_vae_tiling.py`, `AXIS_STARTS` in `packages/models/tests/wan_vae_tiles_test.ts`),
  and the reference fixtures carry the plan in their metadata for the GPU test to compare.

```bash
uv run --group wan --inexact python -m wan.vae_tiling   # vae_tiles.{band,accept}.safetensors (CPU, about 30 minutes)
```

`band` is a seeded latent `[16,9,60,104]` (832×480, 33 frames) with the tiled reference and the
upstream untiled `_decode` chunk loop, both before the clamp. `accept` is another seed in portrait,
`[16,3,104,60]` (480×832, 9 frames), where a row / column mix-up shows in the values.

Measured on 2026-10-02 (B570, Deno 2.9.6):

| Check                                                      | Result                                            |
| ---------------------------------------------------------- | ------------------------------------------------- |
| one 32×32 tile (GPU) vs the untiled chunk loop             | bit-exact (Uint32)                                |
| GPU tiled decode vs the tiled reference, `band`            | max abs 3.87e-6 (ratio 2.65e-6), tolerance 2e-5   |
| GPU tiled decode vs the tiled reference, `accept`          | max abs 4.16e-6 (ratio 3.15e-6)                   |
| 832×480, 33 frames, wall time                              | 128–129 s (12 tiles × 9 chunks)                   |
| VRAM during the decode (DRM fdinfo, 10 ms samples)         | vram0 2.870 GiB before, 2.876 GiB peak; gtt 0.063 |
| tiled vs untiled upstream decode (observation, not a gate) | max abs 0.120, mean 2.26e-3 (ratio 8.17e-2)       |
| CPU reference (`band`): tiled / untiled                    | 836 s / 491 s, peak RSS 9.2 GiB                   |

## Tests

```bash
uv run --group wan --inexact pytest wan   # from tools/export-recipes/
deno task test:models:wan                 # from the repository root (packages/models/tests/*wan*_test.ts)
```

The Deno lane runs the host-function tests and the real-GPU parity gates for the DiT, the VAE chunk
graphs and the tiled decode. The GPU gates read the series under `outputs/series/` written by
`wan.export_dit`, `wan.export_vae` and `wan.vae_tiling`, and skip explicitly (with the generating
command) when those assets or a GPU adapter are missing.

Tests that need the real weights take the `wan_snapshot` fixture (`wan/tests/conftest.py`) and skip
when the pinned snapshot is not in the HF cache.

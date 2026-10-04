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
uv run --group wan --inexact python -m wan.export_dit --dtype i8 # int8 graph + golden → outputs/series/wan2.1-t2v-1.3b-i8-dyn/transformer/
uv run --group wan --inexact python -m wan.dit_host_fixture      # TS host fixture → packages/models/tests/fixtures/wan-dit/
```

Weights are rounded to f16-representable values before any reference is taken (ADR 0006); the
rounding goes through the wrapper, so the upstream modules it shares get the same values while the
RoPE buffers of `model.rope` stay f32. The golden for each case is two files: `io.<case>` (graph
inputs and the patched torch output) and `reference.<case>` (the input latent, the timestep, the
unpatched upstream output and each block's output from the plain diffusers forward).

The writer checks every case before anything is published, and `--verify` applies the same checks
and exits non-zero: the trunk (every rewrite except the patch embedding) must match the upstream
forward bit for bit, the linear form of the patch embedding must stay within the summation-order
bound `2·γ_{K+1}·(|x|·|W|ᵀ + |b|)` of the conv3d, and the patched output must be finite. A failing
case stops the export before the series is touched.

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
desktop CPU (2026-10-02) one S = 14,040 case takes about 370 s for the float64 forward and 140 s
each for the f32 and the patched ones, so the cases up to stage 3 take about 88 minutes and write
about 6.2 GB of golden files (plus the container).

As of 2026-10-02 the real-size comparison is green: both acceptance cases are inside the band of 75
(r = 2.93 and 4.60) and all four fault injections are outside it. The off-by-one timestep's smallest
margin is 4.4× the band (r = 332 at t = 600), above the 2× floor (`SUBTLE_FAULT_MARGIN`). The band,
the metric and the decision cases were not changed after the acceptance run (the NOTE on
`DIT_FULL_NORMALIZED_BAND` in the e2e test has the numbers).

Stage 8 adds eight cases at 81 frames (latent `[16,21,60,104]`, S = 32,760 — exactly the `Dim("S")`
ceiling). They repeat the timesteps and text lengths of the S = 14,040 cases with new seeds
(`full-band` `SEED` + 20 to 25, `full-accept` 777008 / 777009), and their band is derived from them
alone: the S = 14,040 band is not carried over (decision 8). None of them keeps block outputs — the
per-layer probe would need about 6 GB of readback staging for 30 block outputs, which does not fit
on the B570.

As of 2026-10-03 the S = 32,760 comparison is green. The band is 104 (5× the worst decision r, 20.8
at t = 999; the other five decision cases are 3.9 to 12.2, in line with S = 14,040), both
acceptance cases are inside it (r = 13.1 and 4.89), and all four fault injections are outside it;
the off-by-one timestep's smallest margin is 2.65× the band (r = 276 at t = 600). One forward splits
the self-attention into 24 row blocks, and its longest submit took about 459 ms of GPU time (the
gate is 1 s). On the CPU one S = 32,760 case takes about 1,505 s for the float64 forward, 630 s for
the f32 one and 546 s for the patched one; regenerating all 27 cases and the container took
27,271 s (7.6 hours, sharing the machine with GPU generation jobs).

The CPU references (f32 and float64) and the patched eager forward run with attention pinned to
PyTorch's CPU flash kernel (`dit_patch.flash_attention_only`). At S = 32,760 a fallback to the math
kernel would allocate the full score matrix, 51.5 GB in f32 and 103 GB in float64, and die out of
memory; with the pin, an input the flash kernel cannot take raises instead. PyTorch 2.13 already
picks the flash kernel for these shapes, so the pin changes no value: the reference and the patched
output of `band-s00192-t0999` (all 30 block outputs included) were reproduced byte for byte against
the existing golden.

### int8 series (ADR 0120)

`--dtype i8` writes the transformer series behind the int8 quants `f16+dit8` and
`f16+dit8-a8-attn8-s16` (ADR [0120](../../../docs/decisions/0120-wan-dit-w8a8-seat.md)); the VAE
stays in the f16 series. The weights of all 307 linear layers (the patch-embedding linear included)
are rounded to per-channel symmetric int8 (round to nearest, one f32 scale per output channel)
through the token-form wrapper, and stored as int8. Biases, normalization weights and the
`scale_shift_table` keep the upstream f32 values — the f16 rounding of the f16 series is not
applied. There is no calibration: the weights are rounded to nearest and the activations are
quantized per token at run time.

The golden cases are the S = 192 set (`band` / `accept`, including the S = 128 case) and the eight
S = 14,040 cases of the f16 series, with the same seeds and timesteps; `growth` and S = 32,760 are
not retaken. Every case carries a float64 reference (`output.f64`), since the normalized-ratio gate
covers S = 192 as well. `--no-full` leaves out the S = 14,040 cases (about 1.5 hours on the CPU) for
a staged run; a complete series comes from a run without it. The writer applies the same eager
checks as for the f16 series, on the rounded weights that the reference shares.

On the 6-core desktop CPU (2026-10-03) the S = 192 run (`--no-full`) took 104 s with a peak RSS of
13.8 GiB, and wrote a 1,426,806,948-byte container (307 int8 weights, 700,480 scales).

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

The two graphs and the fixtures are written to staging seats and published together, only after
every check has passed, so a failed run never leaves a mixed set (a new `first` next to an old
`next`). The graphs go to
`outputs/series/wan2.1-t2v-1.3b-f16-dyn/{vae_decoder_first,vae_decoder_next}/`
(f16 storage; the weights are rounded to f16 before the references are taken). The series root also
gets `vae_chunks.{band,accept,long}.safetensors`: a seeded de-normalized latent (9, 5 and 21 chunks)
and the upstream non-tiled `_decode` chunk loop before its clamp. `band` sets the GPU tolerance and
`accept` (another latent, other chunk boundaries) is the one judged against it. `long` (stage 8: 21
chunks = 81 frames, the cache carried over 20 times) is a second acceptance case against the same
`band` tolerance.

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

## Text embeddings, sampler and few-step reference (stage 6)

### Text embeddings

The first text stage ships precomputed umT5-XXL outputs (ADR 0118 decision 4). `wan/text_embeds.py`
fetches the pinned `text_encoder` / `tokenizer` (about 23 GB) and runs them **in a separate process
on the CPU in bf16** (the upstream `t5_dtype`), through the upstream
`WanPipeline._get_t5_prompt_embeds` (`prompt_clean` with ftfy 6.3.1 → `padding="max_length"`, 512 →
encoder). Only the valid rows `[L_valid, 4096]` are stored, in f32 (the bf16 values widen exactly).

```bash
uv run --group wan --inexact python -m wan.text_embeds --fetch   # text_encoder + tokenizer at the pinned revision
uv run --group wan --inexact python -m wan.text_embeds           # → outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors
```

| Name             | Role     | Source                                                                     | `L_valid` |
| ---------------- | -------- | -------------------------------------------------------------------------- | --------: |
| `boxing-cats`    | positive | official README, t2v-1.3B example (same as `generate.py` `EXAMPLE_PROMPT`) |        28 |
| `ferret`         | positive | Diffusers docs, Wan T2V example (multi-line string kept verbatim)          |       118 |
| `cat-dog-baking` | positive | Diffusers `WanPipeline.__call__` example                                   |        50 |
| `negative`       | negative | official `sample_neg_prompt` (`wan/configs/shared_config.py`, Chinese)     |       126 |

One tensor per prompt (`F32 [L_valid, 4096]`). The metadata is a **single key**
(`karume.wan.text_embeds`) holding sorted-key JSON: source repo / revision, encoder dtype, library
versions, and per prompt the original text, the normalized text, the token count and the source URL
(pinned to a commit). A single key keeps the file byte-identical across runs: safetensors writes the
metadata map in a per-process order, so several keys give a different header each run. ftfy turns
the full-width commas of the negative prompt into ASCII commas, which is why the normalized text is
recorded.

Measured on 2026-10-02 (6-core desktop CPU): about 2.2–2.5 minutes per prompt, 9–10 minutes in all,
5,280,288 bytes, and the second run wrote the same bytes. Peak anonymous memory is 11.4 GiB; with
a warm page cache the RSS reaches 24 GiB because the mapped fp32 shards (15.3 GiB, reclaimable)
count toward it.

### UniPC and CFG fixture

`wan/scheduler_ref.py` drives the diffusers 0.39.0 `UniPCMultistepScheduler` with the pinned
scheduler config (flow, shift 3.0, bh2, order 2) on seeded synthetic model outputs (latent
`[1,16,3,16,16]`) and writes `packages/models/tests/fixtures/wan-scheduler/unipc.{safetensors,json}`:
the σ (f32) and timestep (int64) columns, the trajectory after each of the 50 steps, the step
orders, and one CFG combination (guide 5.0, the upstream expression). The JSON also holds the σ
columns in float64 for 2 and 50 steps: they are rebuilt with the upstream numpy expression and
checked against the scheduler's f32 σ (bit-exact) and timesteps (exact) before they are written,
since the scheduler does not expose its float64 intermediate. σ[0] is 0.999999 (1 − 1e-6) and the
timesteps start 999, 993, 986, 979, 971.

```bash
uv run --group wan --inexact python -m wan.scheduler_ref   # seconds; no weights
```

### Few-step reference

`wan/few_step_ref.py` runs the plain diffusers `WanPipeline` on CPU f32 (f16-rounded DiT and VAE
weights) for 2 steps with CFG (guide 5.0, shift 3.0) at 832×480, 33 frames, with the text
embeddings from the asset and the seeded torch `randn` noise injected through `latents=`. It writes
`pipeline_steps.<case>.safetensors` at the series root: the injected noise, each step's cond /
uncond DiT outputs (recorded with a forward hook) and latents, and the frames from the stage-5 tiled
decode before the clamp. Two cases set the tolerance — `band-boxing-cats` (seed 20261030) and
`band-cat-dog-baking` (seed 20261032) — and `accept-ferret` (seed 20261033) is judged against it.

```bash
uv run --group wan --inexact python -m wan.few_step_ref   # all three cases (CPU, about 20 min each)
```

### The pipeline these feed (`@karume/models/wan`)

`WanPipeline` (`packages/models/src/wan/`) loads the three containers and the embedding asset from
the distribution (stage 7 below) and runs `generate` in three stages: the asset lookup (only the stored prompts are accepted, by original or normalized text), the DiT with
CFG as two batch-1 passes and the host UniPC, then the tiled VAE decode and the clamp. Measured on
the B570 (2026-10-02, 832×480, 33 frames, 2 steps):

- From the cond / uncond DiT outputs recorded in `pipeline_steps.*`, the host CFG + UniPC reproduces
  the reference latents bit for bit, so the GPU-vs-reference difference comes from the DiT alone.
- GPU against the CPU f32 reference (max |diff| ÷ max |reference|): the bands are five times the
  worse of the two band-setting cases at each point — latents after step 1 8.5e-5, after step 2
  9.1e-4, clamped frames 8.5e-3. The acceptance case lands at 0.16 / 0.56 / 0.67 of them. This gate
  checks the wiring (prompt order, guidance, steps, the hand-off between stages); the numerics of a
  single DiT forward are checked in stage 3 against the f64 reference.
- Stage times: the DiT stage (4 forwards) about 70 s and the VAE stage about 129 s. VRAM (fdinfo
  `drm-total-vram0`) peaks at 5.7 GiB in the DiT stage and 2.9 GiB in the VAE stage, and is back
  to 0.4 GiB right after the DiT session is disposed, so the two stages never overlap and no extra
  wait for released memory is needed between them.
- The 50-step run with the default settings is opt-in (`KARUME_WAN_FULL_PIPELINE=1`); it runs the 33-frame
  and the 81-frame clip and writes every frame and a 4×8 contact sheet as PNG under
  `outputs/verify/<environment>/<date>_wan-pipeline-full/`. The 81-frame clip (2026-10-03) took
  7,173 s: 68.6 s per DiT pass (100 passes) and 315 s for the VAE decode, with VRAM peaks of
  7.31 GiB in the DiT stage and 3.78 GiB in the VAE stage.

The UniPC port is checked against `wan-scheduler/unipc.*`: σ bit for bit, timesteps exactly, the
50-step trajectory within an absolute 2e-5 (torch's float32 `log` differs from the correctly rounded
value by one ULP at six of the σ), and the CFG combination bit for bit.

## umT5 text encoder distribution (ADR 0119 stage 10d)

The GPU text path runs the umT5-XXL encoder with int8 weights (ADR
[0119](../../../docs/decisions/0119-wan-umt5-gpu-text-encoder.md)). Its upstream is the encoder of
[`google/umt5-xxl`](https://huggingface.co/google/umt5-xxl) at a pinned commit (`UMT5_SOURCES` in
`wan/sources.py` — ADR [0122](../../../docs/decisions/0122-umt5-upstream-and-compatible-encoders.md)),
whose float32 weights are bit-identical to the checkpoint's `text_encoder` folder. The container is
the series `outputs/series/umt5-xxl-i8-dyn/text_encoder/` (26 parts, graph name `text_encoder`),
written by `wan/umt5_export.py` from the pinned F32 checkpoint one row block at a time.

`google/umt5-xxl` ships pickle `.bin` shards only. The fetch derives the shards that hold the encoder
(`shared.weight` and `encoder.*`) from the index and takes just those (3 of 6, about 29.8 GB); the
writer checks each shard's SHA-256 against the table before it unpickles anything, and then opens it
with `torch.load(mmap=True, weights_only=True)`. A directory that merely contains
`pytorch_model.bin.index.json` is never unpickled — only the pinned row of the table opens `.bin`
files. The tokenizer asset, `check-mask`, the precomputed embeddings and the ids of the golden cases
still come from the Wan snapshot (`--model`, as in the sibling Wan scripts); the umT5 upstream is
`--upstream` (default `xxl`), and `reference` records both axes in the golden metadata.

```bash
uv run --group wan --inexact python -m wan.sources --umt5 xxl --fetch     # the encoder shards of google/umt5-xxl into the HF cache
uv run --group wan --inexact python -m wan.umt5_export write --check   # write to a scratch seat, compare every part with the series by SHA-256
uv run --group wan --inexact python -m wan.umt5_export write           # replace the series container (then rerun `reference` for the goldens)
```

It ships in its own repository, `models/karume-umt5-xxl/` (`wan/umt5_distribution.py`), which the
Wan distribution references (below). There is no TypeScript family for it; the pipeline name only
names the part:

```bash
uv run python dist.py --pipeline umt5   # default model xxl, out models/karume-umt5-xxl
```

| Manifest entry (`karume/5`) | Value                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| model                       | `xxl` (pipeline `umt5-encoder/1`)                                                                                                           |
| `weights`                   | `text_encoder` with one `i8` container (`text_encoder/model.i8.krm`)                                                                        |
| `assets`                    | none — the tokenizer is bundled with Wan's prompt cleaning and stays in the Wan repository                                                  |
| `quants`                    | `i8` only; the vocabulary embedding (one 1,050,148,864-byte buffer) makes the build declare `maxBufferSize` / `maxStorageBufferBindingSize` |
| `pipelineConfig`            | `{}`                                                                                                                                        |

Before anything is placed, the plan checks the container's provenance against the pinned
`google/umt5-xxl` revision, the binding table per kind of weight (linear layers and the vocabulary
embedding in int8, RMSNorm weights and relative-position tables in float32 — with the real series:
169 int8 and 73 float32 weights), and the graph contract (`input_ids [1, L]` and `relative_position_buckets [L, L]`,
both int32, one symbol `L`, one float32 output `[1, L, W]`). The repository root gets `LICENSE.md`
(Apache 2.0) and `NOTICE.md` (the encoder only, int8 conversion, `gelu_new` →
`GELU(approximate="tanh")`, valid tokens only; the upstream is float32 and bit-identical to the Wan
checkpoint's `text_encoder`).

### Third-party compatible encoders (ADR 0122 stage b)

A third-party umT5-XXL-compatible encoder (for example a fine-tune published on Hugging Face) can be
converted for local experiments. `wan/umt5_intake.py` takes one safetensors file from a pinned
commit, checks its SHA-256 against the Hub API, and places it unchanged (same name, same bytes — no
float32 copy) under `inputs/umt5/<name>/` together with `intake.json`, a machine-written record of the
repository, commit, file SHA-256, source dtype (`BF16` or `F32`), declared license and base model, and
where `config.json` came from (the upstream's own, checked against `google/umt5-xxl`, or a copy of
`google/umt5-xxl`'s — the configuration is then assumed) with the SHA-256 of the bytes placed. Every
reader of the record (`write`, `prepare`, `reference` and the mirror) checks `config.json` against that
SHA-256 first, so a configuration edited after the intake (for example
`relative_attention_max_distance`) fails instead of producing a container or golden from a
configuration never checked against `google/umt5-xxl`; records of an older format are refused — run
the intake again with the same arguments. It accepts safetensors only (never pickle),
the encoder key set of the Wan `text_encoder` (the tied pair `shared.weight` /
`encoder.embed_tokens.weight` counts once; both names must then be bit-identical), and one dtype for
every tensor; FP8, mixed dtypes, the old key spelling and a config whose graph fields or
relative-position buckets differ from `google/umt5-xxl` all fail. It records the license but does not
judge it: when the repository declares none, or declares a value that names no terms (`unknown`,
empty, `other` without a `license_name`), the record says `NOASSERTION`. Keep your own review of the
terms in `license-review.md` next to the record. The intake directory must not resolve into `models/`
(checked on real paths before the Hub API is called), and it may hold only the recorded file,
`config.json`, `intake.json` and `license-review.md` (names compared exactly; anything else — another
weight file in any format or letter case, an index, a partial `.part` download, a subdirectory —
fails).

```bash
uv run --group wan --inexact python -m wan.umt5_intake --repo <owner/name> --revision <40-hex commit> \
    --file <file>.safetensors --name <name>
uv run --group wan --inexact python -m wan.umt5_export write --intake ../../inputs/umt5/<name> [--allow-undeclared-license]
uv run --group wan --inexact python -m wan.umt5_export reference --intake ../../inputs/umt5/<name>
uv run python dist.py --pipeline umt5 --intake ../../inputs/umt5/<name> [--allow-undeclared-license]
```

The writer opens only the file the record names, checks its SHA-256 through the file descriptor it
then reads from (an index next to it is never consulted, and replacing the file after the check does
not change what is read), reads only the dtype the record names and widens bfloat16 to float32 as it
reads (lossless), so the container is byte-identical to one written from the same values stored as
float32; the series is `outputs/series/umt5-xxl-<name>-i8-dyn/` (host checks read only the series they
list, so an intake series needs no entry in any tracked table). The `--upstream` path keeps accepting float32
only. The mirror goes to `outputs/misc/local-dist/<name>/` by default and never into `models/`
(whatever the license; the check compares real paths, so symlinks and `..` do not get around it). It
does not use the name `karume-umt5-xxl`, and its root files come from the record: `LICENSE.md` is the
verbatim text of a declared license (only licenses with a bundled text), `NOTICE.md` names the upstream
repository and commit and the changes made here, and the card states the source dtype and the
configuration origin — never the Apache license and "no retraining" notice of `karume-umt5-xxl`.

When the upstream declares no license, `write` and `dist.py` stop before writing anything unless
`--allow-undeclared-license` is given; the container's `provenance.license` then carries
`NOASSERTION`, the mirror has no `LICENSE.md`, and the card opens with "do not redistribute".
`tools/release/hf-upload.zsh upload` reads `provenance.license` from part 0 of every container before
uploading and refuses `NOASSERTION` in any letter case, `unknown` and `none`, as well as a directory
with no container at all, so a mirror copied or linked into `models/` by hand is stopped by its
contents. There is no publishing path for third-party encoders here.

## Distribution (stage 7)

`wan/distribution.py` assembles the series into the distribution `models/karume-wan2.1/` (ADR 0118
decision 7 — one repository per family generation; not published on Hugging Face yet). `karume dist`
does the copying, hashing and verification; the recipe only says what goes where. The text encoder
is a cross-repository reference to `karume-umt5-xxl` (ADR 0119 stage 10d, decision A), so build that
repository first and pass the five `--ref-*` options:

```bash
uv run python dist.py --pipeline umt5
uv run python dist.py --pipeline wan \
    --ref-repo hdae/karume-umt5-xxl --ref-revision <main SHA of karume-umt5-xxl> \
    --ref-dist ../../models/karume-umt5-xxl --ref-model xxl --ref-role text_encoder
```

Until `karume-umt5-xxl` is published there is no commit SHA. The development mirror uses the
placeholder `0000000000000000000000000000000000000000` and must say so with
`--allow-placeholder-ref`; without that flag the driver refuses the placeholder before writing
anything, so a publishing build has to pass the real SHA (`docs/release-runbook.md` §0). A local
reader resolves the reference through `crossRepo` and does not look at the revision.

| Manifest entry (`karume/5`) | Value                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                       | `t2v-1.3b` (the only one; pipeline `wan/1`)                                                                                                                                                                                                                                                                                                                                                                                                        |
| `weights`                   | `text_encoder` with one `i8` container referenced from `karume-umt5-xxl` (`xxl/text_encoder/model.i8-*.krm`, every part pinned by repo, commit, size and SHA-256); `transformer` with an `f16` and an `i8` container (`model.f16.krm` / `model.i8.krm`); `vae_decoder_first` / `vae_decoder_next` with one `f16` container each — the key is the container's graph name (container-v1 §2.1)                                                        |
| `assets`                    | `text_embeds` (`text_embeds/text_embeds.safetensors`) and `umt5_tokenizer` (`umt5_tokenizer/tokenizer.json` — format `karume-wan-umt5-tokenizer/1`), model-level and quant-independent. `rope_base` stays a container asset of `transformer` (ADR 0109)                                                                                                                                                                                            |
| `quants`                    | `f16` (no session knobs — the seat closest to the original weights, chosen explicitly); `f16+dit8` (int8 transformer, no session knobs — the reference seat); `f16+dit8-a8-attn8-s16` (default; int8 transformer with `linearCompute` / `attentionCompute` `a8` and `attentionScoreStorage` `f16`) — ADR 0120. Every seat selects the `i8` text encoder, so every seat declares the vocabulary embedding's 1,050,148,864 bytes as `requiredLimits` |
| `pipelineConfig`            | `scheduler.shift` 3.0, `defaults.steps` 50, `defaults.guidance` 5.0 (the reference setting, decision 5)                                                                                                                                                                                                                                                                                                                                            |

Before anything is placed, the plan checks that each container stores the format of its seat and
no other compressed format (int8 for the text encoder, f16 for the f16 transformer and the VAE
graphs, int8 for the int8 transformer), names the pinned upstream revision and license in its
provenance (`wan/sources.py`), that the text encoder passes the same gates as in its own repository
and that its output width is the transformers' context width, that the two VAE graphs belong to one
set (the same latent shape, and the caches of `first` appear in `next` with the same names, shapes
and order — the rule the TypeScript loader applies), that both transformers declare the same
`rope_base` asset byte for byte, that the embedding asset was made from the pinned revision with the
bfloat16 encoder and the pinned diffusers / ftfy, carries exactly the prompts of `wan/prompts.py`
with a normalized text for each that no other row claims, and fits both transformers'
`encoder_hidden_states [1, 512, 4096]` input, and that the tokenizer asset has the format the
TypeScript reader knows, the pinned source and transformers / ftfy versions, and a `maxLength` that
fits the context rows. The golden files of the series (`io.*`, `reference.*`, `vae_*`,
`pipeline_steps.*`) are never copied, and with the reference the text encoder is not copied either
(`karume dist` checks that the referenced parts are byte-identical to the series container).

The repository root gets `LICENSE.md` (Apache 2.0, verbatim) and `NOTICE.md` (the changes: container
format, f16 rounding, the int8 transformer, the transformer and VAE rewrites, the text encoder
referenced from `karume-umt5-xxl`, the precomputed text embeddings kept for use without it, the
tokenizer converted together with the prompt-cleaning tables), and `README.md` is the model card
rendered from the manifest by `wan/card.py`: the pinned upstream, where the text encoder comes from,
the `textEncoder` choice (`"gpu"` by default, `"precomputed"` for the fixed prompts only), the
prompt rules, the fixed prompts with their sources, the accepted inputs, how the outputs are
verified, the quant table (with the `dit` abbreviation spelled out), the defaults, the measured
resources (each stage with the `f16` quant and the precomputed embeddings, the text encoder stage
from a 2-step run, and the transformer of each quant, with the cells not measured marked as such)
and the declared limits. Re-running the command writes the same bytes.

## Tests

```bash
uv run --group wan --inexact pytest wan   # from tools/export-recipes/
deno task test:models:wan                 # from the repository root (packages/models/tests/*wan*_test.ts)
```

`wan/tests/test_distribution.py` assembles the distribution once from minimal synthetic containers
(layout, manifest, every gate above, the card) — with the text encoder referenced from a synthetic
`karume-umt5-xxl`, as published — checks the placeholder-SHA gate of the driver, and, when the real
series exist, builds the real plan without copying anything. `wan/tests/test_umt5_distribution.py`
does the same for the umT5 repository and, with the real series, pins its binding table (169 int8
and 73 float32 weights).

The stage-6 tests check the fixed prompts (commit-pinned sources, the pipeline example matching the
pinned diffusers), the normalization (ftfy is required), the asset format (round trip, rejection of
tensors outside the contract, byte-identical output from two processes), and, when the generated
files exist, the asset (f32 `[L_valid, 4096]`, `L_valid` equal to the tokenizer mask length,
normalized text equal to `prompt_clean`), the scheduler columns (first timestep 999, strictly
decreasing, the 1e-6 correction), the scheduler fixture being current, and the few-step reference
(shapes, schedule and tile plan in the metadata, the embeddings' sha256, the seeded noise).

The Deno lane runs the host-function tests (including the UniPC / CFG fixture and the pipeline's input
gates) and the real-GPU parity gates for the DiT, the VAE chunk graphs, the tiled decode and the
two-step pipeline run (with its sha256 row in `packages/models/tests/fixtures/references/wan.json`,
written with `KARUME_REFERENCE=write`). The DiT and VAE gates read the series under `outputs/series/`
written by `wan.export_dit`, `wan.export_vae` and `wan.vae_tiling`; the pipeline run loads the
distribution `models/karume-wan2.1/` through `fromPretrained` and compares it with the references
written by `wan.few_step_ref` — its sha256 rows were first written from the series, so the
distribution path has to reproduce them bit for bit. The gates skip explicitly (with the generating
command) when those assets or a GPU adapter are missing. `KARUME_WAN_FULL_PIPELINE=1` adds the
50-step runs: the 33-frame and the 81-frame clip, each with the `f16` quant and with
`f16+dit8-a8-attn8-s16` (four runs). On the B570, the `f16` quant takes about half an hour for the
33-frame clip and about two hours for the 81-frame clip; `f16+dit8-a8-attn8-s16` takes about 16
minutes for the 33-frame clip and about 62 minutes for the 81-frame clip (3,703 s, of which the
transformer stage is 3,383 s).

Tests that need the real weights take the `wan_snapshot` fixture (`wan/tests/conftest.py`) and skip
when the pinned snapshot is not in the HF cache.

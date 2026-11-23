# Wan export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)). The design is ADR
[0118](../../../docs/decisions/0118-wan21-video-generation.md), with ADR
[0121](../../../docs/decisions/0121-wan22-ti2v-5b.md) for Wan2.2. Scripts are started as modules
from `tools/export-recipes/` (`uv run --group wan python -m wan.<module> …`).

## Overview

Text and image to video with Wan2.2 TI2V 5B: the DiT (S form, one symbol for the token length —
`ti2v_export_dit`), the video VAE decoder (two chunk graphs with the causal cache as resident
tensors — `export_vae`, with the tile references of `vae_tiling`), the VAE encoder for the condition
image (`export_vae_encoder`) and host-side reference pipelines (`ti2v_few_step_ref`,
`ti2v_i2v_few_step_ref`). The two text assets the distribution ships (the precomputed umT5
embeddings of the fixed prompts and the tokenizer — `text_embeds`, `umt5_tokenizer`) and two
TypeScript host fixtures (`scheduler_ref`, `dit_host_fixture`) are made from the Wan2.1 T2V 1.3B
checkpoint. The references are diffusers 0.39.0 pipelines and modules on CPU f32, built without the
text encoder and fed precomputed text embeddings (the shared settings are in `wan/pipeline_ref.py`).

```bash
uv sync --group wan --inexact   # accelerate / diffusers==0.39.0 / ftfy==6.3.1 / huggingface-hub / transformers==5.14.1
```

`--inexact` keeps the other families' groups installed in the shared venv (a plain `uv sync --group
wan` removes them).

## Sources

The upstream repositories and their pinned 40-hex revisions live in `wan/sources.py` (`SOURCES`).
Only the `transformer`, `vae` and `scheduler` subfolders are fetched (about 20 GB, 2.8 GB and a
config for TI2V-5B); the umT5-XXL text encoder of the Wan2.1 T2V 1.3B checkpoint (about 22.7 GB
fp32) and its tokenizer are fetched separately, by `wan.text_embeds --fetch`.

```bash
uv run --group wan python -m wan.sources --model ti2v-5b --fetch   # download the pinned TI2V-5B revision into the HF cache
uv run --group wan python -m wan.sources --model ti2v-5b           # check the cached snapshot (parameter counts)
uv run --group wan python -m wan.sources --fetch                   # the Wan2.1 T2V 1.3B checkpoint (text assets, host fixtures)
```

## DiT

The DiT graph is the **token form** (S form, ADR 0034): it starts after the patchify and ends before
the unpatchify, so every value is written with one symbol, the token length `S`
(`Dim("S", max=27280)` = 1280×704 with 121 frames for TI2V-5B). `wan/dit_patch.py` holds the
wrappers and the host-side reference functions, and `wan/dit_common.py` the golden form, the
eager-equivalence check and the weight rounding they share; the host steps are mirrored in
TypeScript under `packages/models/src/wan/`.

| Step                            | Where | How                                                                                                                                         |
| ------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| patchify + patch embedding      | host  | latent `[C,F,H,W]` → tokens `[1,S,C·pt·ph·pw]` in `(c, pt, ph, pw)` order; `Conv3d(k=s=(1,2,2))` → `Linear` on a view of the conv weight |
| sinusoidal timestep (256)       | host  | graph input `timesteps_proj [1,256]` (cos first); the MLP stays in the graph                                                                |
| 3D RoPE tables                  | host  | reordered from the per-axis base tables in the container asset `rope_base` (role `rope-base`) into `rope_cos` / `rope_sin [1,S,1,128]`      |
| interleaved RoPE application    | graph | rewritten as "swap adjacent pairs + elementwise cos / sin" at rank 4 (bit-exact)                                                            |
| adaLN, qk-norm, cross-attention | graph | upstream as is; the 512-row text context is projected without a mask                                                                       |
| unpatchify                      | host  | output tokens `[1,S,pt·ph·pw·C]` are in `(pt, ph, pw, c)` order — not the patchify order                                                    |

The TI2V-5B graph (`dit_patch.WanDitTokensTi2v`) adds two inputs for image to video, the
condition-side `timesteps_proj_condition [1,256]` and `condition_mask [1,S,1]`, and selects each
modulation component per token with a `where` (ADR 0121 decision 3). `wan/ti2v_export_dit.py`
writes it as the int8 series `outputs/series/wan2.2-ti2v-5b-i8-dyn/transformer/` with its goldens
(the module doc has the cases, the IR check and the real-size goldens):

```bash
uv run --group wan --inexact python -m wan.ti2v_export_dit inspect      # trace + structure check (config only)
uv run --group wan --inexact python -m wan.ti2v_export_dit write        # container + S = 192 goldens
uv run --group wan --inexact python -m wan.ti2v_export_dit write-full   # real-size goldens
uv run --group wan --inexact python -m wan.ti2v_export_dit eager-full   # eager equivalence on the whole f32 model
uv run --group wan --inexact python -m wan.dit_host_fixture             # TS host fixture → packages/models/tests/fixtures/wan-dit/ (Wan2.1 checkpoint)
```

The golden for each case is two files: `io.<case>` (the graph inputs and the patched torch output)
and `reference.<case>` (the input latent, the timesteps, the upstream output and `output.f64`, the
same forward with the activations in float64, rounded to f32). The references read the same int8
weights as the container (`write` reads them back from the container it has just written), and the
RoPE buffers of `model.rope` stay f32.

The writer checks every case before anything is published (`dit_common.eager_failures`): the trunk
(every rewrite except the patch embedding) must match the upstream forward bit for bit, the linear
form of the patch embedding must stay within the summation-order bound `2·γ_{K+1}·(|x|·|W|ᵀ + |b|)`
of the conv3d, and the patched output must be finite. A failing case stops the export before the
series is touched. The real-GPU gate is `packages/e2e/tests/e2e_wan_ti2v_dit_test.ts`: its metric is
the normalized ratio r = (max|GPU − f64| / max|f64|) / (max|CPU f32 − f64| / max|f64|), and the band
is 5× the worst r of the decision cases.

The CPU references (f32 and float64) and the patched eager run with attention pinned to PyTorch's
CPU flash kernel (`dit_patch.flash_attention_only`). At a large S a fallback to the math kernel
would allocate the full score matrix and die out of memory; with the pin, an input the flash kernel
cannot take raises instead. PyTorch 2.13 already picks the flash kernel for these shapes, so the pin
changes no value. The float64 reference runs one block at a time (`wan/dit_reference.py` — the whole
5B in float64 does not fit in host RAM), and `wan/dit_probe.py` measures the RAM of the export and of
the references.

## VAE (stage 4)

The upstream decode runs the decoder one latent frame at a time and carries a causal cache
(`feat_cache`) in a Python list with a `'Rep'` sentinel. `wan/vae_patch.py` turns that into two
static **chunk graphs** on unbatched rank-4 tensors `[C, T, H, W]` (ADR 0118 decision 2 — the
shapes of the TI2V-5B pair are in the table of the TI2V-5B subsection below):

| Graph               | Inputs                              | Outputs                                             |
| ------------------- | ----------------------------------- | --------------------------------------------------- |
| `vae_decoder_first` | `latent [z,1,t,t]` + 30 `cache_NN`  | frame `[C,1,8t,8t]` (before clamp) + the 30 caches  |
| `vae_decoder_next`  | `latent [z,1,t,t]` + 32 `cache_NN`  | frames `[C,4,8t,8t]` (before clamp) + the 32 caches |

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
- **Host side.** `z·std + mean` before the graph, and the unpatchify and `clamp(-1, 1)` after the
  tile blend, stay on the host. The tile side `t` is an export argument (default 16 = diffusers'
  256 px tile ÷ the spatial compression 16); the host reads it back from the asset's `latent` input.

The two graphs and the fixtures are written to staging seats and published together, only after
every check has passed, so a failed run never leaves a mixed set (a new `first` next to an old
`next`). The weights are rounded to f16 before the references are taken.

### Tiled decode (stage 5)

The chunk graphs take a fixed `t×t` latent tile, so the host tiles the full frame: tiles outside,
chunks inside (the cache is zeroed again at every tile, like the upstream `tiled_decode`).
`wan/vae_tiling.py` is the Python side of the geometry and writes the references; the TypeScript
side is `packages/models/src/wan/vae-tiles.ts`.

- **Geometry.** Rounded equal spacing with the last tile snapped to `extent − tile` (the Anima rule,
  ADR 0033), using the fewest tiles whose neighbours overlap by at least 64 px (the upstream default
  blend, 256 − 192) — 4 latents for Wan2.2 (64 px ÷ the spatial compression 16). The upstream
  `range(0, H, stride)` walk leaves a short last tile, which a fixed-shape graph cannot take. The
  tile side and the scale are read from the asset's declared shapes.
- **Blend and paste.** `blend_v` / `blend_h` are copied verbatim from `AutoencoderKLWan` (vertical
  first, in place, over every frame). Tile `i` pastes the region `[starts[i], starts[i+1])`, the
  last one up to the edge. `clamp(-1, 1)` comes after the paste.
- **Frozen plan.** The tile starts are pinned as literal tables on both sides
  (`MIRRORED_STARTS_TI2V` in `wan/tests/test_vae_tiling_ti2v.py`, `AXIS_STARTS_TI2V` in
  `packages/models/tests/wan_ti2v_vae_tiles_host_test.ts`; the planner is also pinned at tile 32 with
  an overlap of 8 latents in `wan/tests/test_vae_tiling.py` and
  `packages/models/tests/wan_vae_tiles_test.ts`), and the reference fixtures carry the plan in their
  metadata for the GPU test to compare.

To show that a recipe change leaves the series unchanged, run the writers into an empty scratch
directory (`--out` on both — without it they overwrite the series itself) and compare only the named
items with `wan.series_check`; a copy of the series would match too, so the check means something
only for a fresh run:

```bash
uv run --group wan --inexact python -m wan.series_check --written <scratch> \
  --existing ../../outputs/series/wan2.2-ti2v-5b-f16-dyn \
  --require vae_decoder_first --require vae_decoder_next --require vae_chunks.band.safetensors
```

Safetensors files are compared tensor by tensor (names, dtype, shape and raw bits) plus their
metadata as a dictionary, because safetensors 0.8.0 writes the metadata keys in an order that varies
between processes; every other file is compared by size and SHA-256. It prints one JSON summary and
exits 1 on any mismatch, including an item or file missing on either side, a required directory with
no file in it and a symlink inside the compared items, which it never follows (2 is a usage error, 3
an unreadable file).

### TI2V-5B (Wan2.2, ADR 0121 stage 4)

The Wan2.2 TI2V-5B decoder is exported as the same two chunk graphs, with the same cache
normalization and the same 15 IR ops (ADR [0121](../../../docs/decisions/0121-wan22-ti2v-5b.md)
decision 6). `wan/vae_patch.py` accepts exactly two generations and adds two things for Wan2.2:

- **Residual up blocks.** Each up block that upsamples (three of the four) has a shortcut outside
  its residual path, `DupUp3D` (channel repeat, then depth-to-space along T, H and W). `_shared/vae_rank4.dup_up_3d` writes it
  with rank-4 values only and is bit-exact against the upstream module; the first chunk drops the
  leading `ft − 1` frames, as upstream does.
- **Output in patchify space.** The graphs return 12 channels (RGB folded 2×2) before the
  unpatchify; the unpatchify and the clamp stay on the host (ADR 0121 stage 5). The latent tile is
  16 (256 px ÷ a spatial compression of 16). The tiled decode described above also covers Wan2.2
  since ADR 0121 stage 5: `wan.vae_tiling --model ti2v-5b` blends and pastes in patchify space with
  an overlap of 4 latents (64 px ÷ 16) and applies the unpatchify and the clamp after the paste.

| Graph               | Inputs                                | Outputs                                                                |
| ------------------- | ------------------------------------- | ---------------------------------------------------------------------- |
| `vae_decoder_first` | `latent [48,1,16,16]` + 30 `cache_NN` | frame `[12,1,128,128]` (patchify space, before clamp) + the 30 caches  |
| `vae_decoder_next`  | `latent [48,1,16,16]` + 32 `cache_NN` | frames `[12,4,128,128]` (patchify space, before clamp) + the 32 caches |

```bash
uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b            # both graphs + chunk fixtures (f16)
uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --verify   # eager equivalence, tile 16, 9 chunks (writes nothing)
uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --dtype i8 # both graphs in int8 (no fixtures)
uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --tile 32 --shared            # the 32-tile pair (f16 series)
uv run --group wan --inexact python -m wan.export_vae --model ti2v-5b --dtype i8 --tile 32 --shared # the 32-tile pair (i8 series)
```

`--shared` (ADR [0134](../../../docs/decisions/0134-wan22-vae-tile32-shared.md) decision 1) writes
`vae_decoder_first_32` / `vae_decoder_next_32` beside the 16-tile pair of the same series — the same
graphs traced at a 32-latent tile, every initializer borrowing the 16-tile pair's weights — so the
Wan2.2 distribution ships this second decoder pair with no extra weight download, and the application
chooses the tile explicitly (16 by default; the 10 GB class keeps 16).
`uv run --group wan --inexact python -m wan.vae_tiling --model ti2v-5b --tile 32` writes the 32-tile
references as `vae_tiles.<case>.t32.safetensors` beside the 16-tile ones (decision 6; same 4-latent
overlap, `tile` in the meta). Only `wide` (1280×704) fits a 32 tile; `band` and `accept` are 30
latents on one axis, so they are skipped and named in the summary.

`--dtype i8` (ADR [0133](../../../docs/decisions/0133-conv3d-w8a8.md) decision 7) rounds the weights
to f16 as above, then quantizes every convolution weight of the decoder to int8 with one scale per
output channel, and writes the same two graphs to
`outputs/series/wan2.2-ti2v-5b-i8-dyn/{vae_decoder_first,vae_decoder_next}/` (beside the int8
transformer's `transformer/`). It writes no chunk fixtures: those are the f16 reference of the e2e
tests, and the int8 graphs are compared against the f16 decode on the GPU.

`--model` defaults to `ti2v-5b`, the one row of the table in `export_vae.py` (`VAE_SERIES`) that
picks the series, the tile and the fixture seeds, and the writer refuses a VAE whose `patch_size` or
spatial compression does not match it. The run writes
`outputs/series/wan2.2-ti2v-5b-f16-dyn/{vae_decoder_first,vae_decoder_next}/` and
`vae_chunks.{band,accept,long}.safetensors` at the series root (9, 5 and 21 chunks; the reference is
the upstream non-tiled `_decode` chunk loop before the unpatchify and the clamp). `--verify` reports
whether the cache normalization alone is bit-exact against the upstream `_decode`, whether
`clamp(unpatchify(reference))` equals the upstream `_decode`, and the ratio of the rewritten graphs
to the upstream.

Measured on 2026-10-05:

| Check                                                     | Result                                              |
| --------------------------------------------------------- | --------------------------------------------------- |
| cache normalization only vs `_decode` (tile 16, 9 chunks) | bit-exact                                           |
| rewritten chunk graphs (eager) vs upstream                | max abs 7.09e-6 / reference max 1.40 = 5.07e-6      |
| `first` / `next` graph                                    | 481 / 491 nodes, 1,085,152,447 / 1,110,338,097 B    |
| second export into an empty directory vs the series       | all 17 files match (`wan.series_check`)             |
| export / `--verify` (CPU)                                 | 538 s, peak RSS 4.85 GiB / 368 s, peak RSS 5.91 GiB |
| GPU (B570, Deno) chunk loop vs upstream, `band`           | max abs 9.48e-6 (tolerance 4.8e-5)                  |
| GPU chunk loop vs upstream, `accept` / `long` (21 chunks) | max abs 7.42e-6 / 1.15e-5                           |

The GPU check is `packages/e2e/tests/e2e_wan_ti2v_vae_tiles_test.ts` and the asset check
`packages/e2e/tests/e2e_wan_ti2v_vae_chunks_host_test.ts`, both in `vp test --project models:wan-ti2v`.

The pytest side is `wan/tests/test_dup_up_3d.py`, `wan/tests/test_vae_patch_ti2v.py` and the
TI2V-5B classes of `wan/tests/test_export_vae.py`. With the `wan` group installed they always run on
a synthetic Wan2.2 VAE (the
upstream `AutoencoderKLWan` with the 5B structure and small channels — `ti2v_synthetic_vae` in
`wan/tests/conftest.py`). The real-weight tests take the `wan22_snapshot` fixture and skip when the
pinned TI2V-5B snapshot is not in the HF cache; they read the whole VAE in f32 (a peak RSS of about
3.5 GiB, about 2 minutes), so leave them out with `-k 'not RealWeights'` next to another long CPU
job.

The VAE writers (`export_vae.py` and `vae_tiling.py`) import `vae_patch.py` and
`_shared/vae_rank4.py`, so a change to any of these four files can change the Wan2.2 VAE series.
After such a change, rewrite the VAE items into a fresh directory and compare all of them with
`wan.series_check` (see the end of the previous subsection for how it compares).
`_shared/vae_rank4.py` is also imported by the Anima recipe (`anima/patch.py`), so run
`anima/tests/test_patch.py` as well after touching it.

```bash
OUT=$(mktemp -d); S=../../outputs/series/wan2.2-ti2v-5b-f16-dyn
uv run --group wan --inexact python -m wan.export_vae --out $OUT
uv run --group wan --inexact python -m wan.series_check --written $OUT --existing $S \
  --require vae_decoder_first --require vae_decoder_next --require vae_chunks.band.safetensors \
  --require vae_chunks.accept.safetensors --require vae_chunks.long.safetensors
uv run --group wan --inexact python -m wan.vae_tiling --out $OUT
uv run --group wan --inexact python -m wan.series_check --written $OUT --existing $S \
  --require vae_tiles.band.safetensors --require vae_tiles.accept.safetensors \
  --require vae_tiles.wide.safetensors
```

### TI2V-5B VAE encoder for image to video (Wan2.2, ADR 0121 stage 9a)

Image to video encodes one condition image with the Wan2.2 VAE encoder (chunk 0 only).
`wan/export_vae_encoder.py` writes it as three graphs split around the middle attention, so the
sizes stay symbolic: the attention's sequence length h·w cannot be written as an IR dimension
expression (one symbol, linear), so it gets a graph of its own over `S`. One container holds one
graph (`karume/6`), so the encoder is three parts. `wan/vae_encoder_patch.py` holds the rewrites:
the causal conv3d folded to the last time slice that sees real data in chunk 0 (an identity in
f64), `AvgDown3D` in the closed form of chunk 0 (mean as sum × 1/4), the asymmetric `ZeroPad2d` as
a pad of the last axis with a permute, the attention in sequence form (row L2 normalization and
`linear`), and `post` returning the first 48 channels of `quant_conv` (the mean `mu`).

| Graph (part)       | Input → output                         | Symbols | Nodes | Container    |
| ------------------ | -------------------------------------- | ------- | ----- | ------------ |
| `vae_encoder_pre`  | `image` `[12,1,8h,8w]` → `[640,1,h,w]` | h, w    | 266   | 88,699,409 B |
| `vae_encoder_attn` | `tokens` `[640,S]` → `[640,S]`         | S       | 19    | 3,295,014 B  |
| `vae_encoder_post` | `hidden` `[640,1,h,w]` → `[48,1,h,w]`  | h, w    | 36    | 15,893,533 B |

The weights are f16 (107.9 MB in all) and go into the same series as the decoder,
`outputs/series/wan2.2-ti2v-5b-f16-dyn/{vae_encoder_pre,vae_encoder_attn,vae_encoder_post}/`.
`[640,1,h,w]` and `[640,S]` are the same bytes read two ways. The graphs accept any multiple-of-16
size; the accepted sizes are checked by the pipeline.

```bash
uv run --group wan --inexact python -m wan.export_vae_encoder            # 3 graphs + goldens
uv run --group wan --inexact python -m wan.export_vae_encoder --verify   # eager equivalence (writes nothing)
```

The goldens `vae_encoder.<case>.safetensors` at the series root cover three test images
(`inputs/wan-i2v/<name>-832x480.png` — not tracked by git, pinned by SHA-256 in `IMAGES`) at
1280×704, 704×1280 and 256×160 (outside the accepted set, a downscaling path) with `fit` `crop`;
two images set the tolerance and the third is judged against it. One more case keeps only the
`stretch` RGB8 at 1280×704 for the TypeScript host test. Against the upstream untiled encode
(diffusers 0.39.0, CPU f32), the nine cases differ by 5.3e-7 to 2.1e-6 (largest absolute difference
over the reference's largest value); the differences come only from the folded conv3d, the L2
normalization and the attention's reduction order. The GPU gate is
`packages/e2e/tests/e2e_wan_ti2v_vae_encoder_test.ts` (tolerance 7.8e-5) in
`vp test --project models:wan-ti2v`. On a machine with the series root, the asset gate requires the
three encoder parts as well, so a machine that has only the decoder needs the first command above.

The image preprocessing has a reference of its own, `wan/i2v_preprocess_ref.py` (no command — the
encoder goldens and the image-to-video reference below call it): `crop` as in the official Wan2.2
code (scale to cover, Python rounding, LANCZOS, center crop) and `stretch` as in diffusers'
`WanImageToVideoPipeline` (LANCZOS straight to the output size), then `[-1, 1]` and the upstream
patchify. Its values depend on Pillow's LANCZOS, so the `wan` group pins `pillow==12.3.0`.

`wan/lanczos_fixture.py` writes the git-tracked fixture that the TypeScript LANCZOS port must
match byte for byte (45 cases, synthetic images only):

```bash
uv run --group wan --inexact python -m wan.lanczos_fixture
(cd ../.. && vp fmt packages/models/tests/fixtures/wan-i2v/lanczos.json)   # the formatter's form is the committed form
```

It writes `packages/models/tests/fixtures/wan-i2v/lanczos.json` (cases and provenance) and
`lanczos.safetensors` (the pixels). The pytest side is `wan/tests/test_vae_encoder_patch.py`,
`wan/tests/test_export_vae_encoder.py`, `wan/tests/test_i2v_preprocess_ref.py` and
`wan/tests/test_lanczos_fixture.py` (the last one rebuilds the fixture with the installed Pillow
and fails when it no longer matches the committed one). They run on the synthetic Wan2.2 VAE and
synthetic images; the real-weight tests and the goldens on disk skip when they are missing.

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

### Few-step reference, TI2V-5B (Wan2.2, ADR 0121 stage 6)

`wan/ti2v_few_step_ref.py` runs the plain diffusers `WanPipeline` on CPU f32 for 2 steps with CFG (guide
5.0) at 1280×704, 17 frames (latent `[48,5,44,80]`, S = 4,400) — one of the two officially
supported sizes. The scheduler is loaded with `flow_shift=5.0` passed explicitly and checked after
loading (the pinned config holds the same value). The DiT is the full f32 checkpoint rounded to the
per-channel int8 grid (RTN fake-quant, the same values as the int8 series) and the VAE is
f16-rounded. The run has two phases in one process:

1. **DiT.** `WanPipeline(expand_timesteps=False)`, so the timestep is one value per pass (the T2V
   form of ADR 0121 decision 4). A forward hook records the cond / uncond outputs, labelled by the
   text context they received. The two band cases also run with `expand_timesteps=True` (the
   upstream default, timestep `[1,S]`) through a separate hook and store those latents as an
   observation, not a gate.
2. **VAE.** The DiT is released first (the run stops if a reference to it is left), then the
   stage-5 tiled decode in patchify space (28 tiles), the upstream unpatchify and the clamp.

It writes `pipeline_steps.<case>.safetensors` at the int8 DiT series root
(`outputs/series/wan2.2-ti2v-5b-i8-dyn/`): the injected noise, each step's cond / uncond DiT outputs
and latents, the clamped RGB frames `[3,17,704,1280]`, the 48 latent mean / std values, and for
the band cases `observed_m_s.latents.<i>`. The metadata holds the seed, the prompt names, the
schedule (timesteps, σ, `flow_shift` `"5.0"`), the tile plan, the source pin and the numeric
environment (torch version, threads, `MKL_*` / `OMP_*` / `KMP_*`). Two cases set the tolerance —
`band-boxing-cats` (seed 20262101) and `band-cat-dog-baking` (seed 20262102) — and
`accept-ferret` (seed 20262103) is judged against it; all three have the same shape.

```bash
uv run --group wan --inexact python -m wan.ti2v_few_step_ref                        # all three cases
uv run --group wan --inexact python -m wan.ti2v_few_step_ref --case accept-ferret   # one case
```

The first phase needs about 24 GiB of RAM (estimate 23.7 GiB: the f32 DiT, the measured fake-quant
overhead, the activations at S = 4,400 and the f32 VAE). Before loading, the script stops unless
`MemAvailable` covers the estimate plus 1 GiB of headroom; the estimate is not a guarantee, so a
run that passes the check can still run short at the peak. Expect about three hours for the three
cases (estimate: 20 DiT passes and 28 tiles × 5 chunks per case on the CPU). The first-phase results
are kept in memory only: if the run stops before or during the decode, the DiT phase (about 40 min)
has to be run again (fixtures already written stay). Do not run it next to another heavy CPU job. The pytest side
(`wan/tests/test_ti2v_few_step_ref.py`) runs the whole two-phase script on a small synthetic Wan2.2
DiT and VAE.

### Few-step reference, TI2V-5B image to video (Wan2.2, ADR 0121 stage 9b)

`wan/ti2v_i2v_few_step_ref.py` is the image-to-video counterpart and reuses the parts of
`ti2v_few_step_ref.py` (the weight reader, the scheduler, the prompt rows, the tiled decode and the
two phases). It builds diffusers' `WanImageToVideoPipeline` from the pinned snapshot (the
`expand_timesteps` branch, the Wan2.2 5B image-to-video form) and runs it on CPU f32 for 2 steps
with CFG (guidance 5.0, shift 5.0) at 1280×704 with 9 frames (latent `[48,3,44,80]`, S = 2,640, the
first 880 tokens being the condition). The condition image is a stage-9a test image cropped by
`i2v_preprocess_ref.crop_resize`; before writing, the script checks that the condition latent, the
source image and the cropped image match the stage-9a encoder golden bit for bit (a missing golden
is reported as `missing` in the summary). The timestep MLP runs once per value (M = 1: the
generation timestep and 0 for the condition tokens — ADR 0121 decision 4); the two deciding cases
also run the unchanged pipeline (M = S) as an observation, which differs from the reference by
6.1e-6 to 1.9e-5.

It writes `pipeline_steps_i2v.<case>.safetensors` at the int8 DiT series root
(`outputs/series/wan2.2-ti2v-5b-i8-dyn/`): the injected noise, the condition latent, each step's
cond / uncond outputs and latents (the scheduler's state, before the replacement), `latents_final`
(after it), the frames `[3,9,704,1280]`, the latent mean / std, the source and the cropped RGB8, and
for the deciding cases the M = S observations. `band-boxing-cats` and `band-cat-dog-baking` set the
tolerance and `accept-ferret` is judged against it.

```bash
uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref                       # all three cases
uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref --case accept-ferret  # one case
```

Measured on 2026-10-09 (Ryzen 5 7600): the three cases took 41 min 48 s with a peak RSS of
24.07 GiB. The RAM check adds the image-to-video overhead to the text-to-video estimate and stops
unless `MemAvailable` covers it plus 1 GiB; the first-phase results are kept in memory only, as in
the text-to-video reference, so do not run it next to another heavy CPU job. The official Wan2.2
form (replacing the scheduler's state after every step) and the diffusers form (replacing only the
transformer's input) give bit-identical final outputs with the pinned UniPC.
`wan/tests/test_ti2v_i2v_few_step_ref.py` keeps that check for 2, 8 and 50 steps on a stand-in
model and runs the whole script on a small synthetic Wan2.2 model. The GPU side is
`packages/models/tests/e2e_wan_ti2v_i2v_pipeline_test.ts`.

### The pipeline these feed (`@karume/models/wan`)

`wan.Ti2vPipeline` runs the host steps these references pin; its GPU gates against the few-step
references are `packages/e2e/tests/e2e_wan_ti2v_pipeline_test.ts` (text to video) and
`packages/e2e/tests/e2e_wan_ti2v_i2v_pipeline_test.ts` (image to video).

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
files. The Wan tokenizer asset (with the prompt-cleaning tables), `check-mask`, the precomputed
embeddings and the ids of the golden cases still come from the Wan snapshot (`--model`, as in the sibling Wan scripts); the umT5 upstream is
`--upstream` (default `xxl`), and `reference` records both axes in the golden metadata.

```bash
uv run --group wan --inexact python -m wan.sources --umt5 xxl --fetch     # the encoder shards of google/umt5-xxl into the HF cache
uv run --group wan --inexact python -m wan.umt5_export write --check   # write to a scratch seat, compare every part with the series by SHA-256
uv run --group wan --inexact python -m wan.umt5_export write           # replace the series container (then rerun `reference` for the goldens)
```

It ships in its own repository, `models/karume-umt5-xxl/` (`wan/umt5_distribution.py`), together
with the tokenizer of `google/umt5-xxl` converted into one JSON table, so it can be used without Wan
(`Encoder` from `@karume/models/umt5`); the Wan2.2 distribution references its encoder (below).
The tokenizer asset (format `karume-umt5-tokenizer/1` — the T5 tables of the Wan asset without the
prompt cleaning) is built by `wan/umt5_xxl_tokenizer.py` from `tokenizer.json` and
`tokenizer_config.json` at the pinned commit (their SHA-256 are in `UMT5_SOURCES`), into the series
`outputs/series/umt5-xxl-tokenizer/tokenizer.json`, together with the TypeScript fixture
`packages/models/tests/fixtures/umt5/tokenizer-cases.json`. Without Wan's prompt cleaning, two more
input shapes make `transformers` 5.14.1 and the original `tokenizer.json` disagree — whitespace
other than U+0020 and a trailing space — so they are rejected as well (the module docstring has
the measurements):

```bash
uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer [--fetch]   # asset + fixture (then vp fmt the fixture from the repository root)
uv run python dist.py --pipeline umt5   # default model xxl, out models/karume-umt5-xxl
```

| Manifest entry (`karume/6`) | Value                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| model                       | `xxl` (pipeline `umt5-encoder/1`)                                                                                                           |
| `weights`                   | `text_encoder` with one `i8` container (`text_encoder/model.i8.krm`)                                                                        |
| `assets`                    | `tokenizer` (`tokenizer/tokenizer.json` — format `karume-umt5-tokenizer/1`, no prompt cleaning)                                             |
| `quants`                    | `i8` only; the vocabulary embedding (one 1,050,148,864-byte buffer) makes the build declare `maxBufferSize` / `maxStorageBufferBindingSize` |
| `pipelineConfig`            | `{}`                                                                                                                                        |

Before anything is placed, the plan checks the container's provenance against the pinned
`google/umt5-xxl` revision, the binding table per kind of weight (linear layers and the vocabulary
embedding in int8, RMSNorm weights and relative-position tables in float32 — with the real series:
169 int8 and 73 float32 weights), and the graph contract (`input_ids [1, L]` and `relative_position_buckets [L, L]`,
both int32, one symbol `L`, one float32 output `[1, L, W]`), and the tokenizer asset against its
format, its field set, the pinned repository, commit and file SHA-256, the `transformers` version
and the maximum length of 512. The repository root gets `LICENSE.md` (Apache 2.0) and `NOTICE.md`
(the encoder only, int8 conversion, `gelu_new` → `GELU(approximate="tanh")`, valid tokens only, the
converted tokenizer and the inputs the host rejects; the upstream is float32 and bit-identical to
the Wan checkpoint's `text_encoder`).

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

The whole path, run from `tools/export-recipes` (placeholders in angle brackets; `<name>` is your
own short name for the encoder, used for the intake directory, the series and the mirror):

```bash
# 1. Intake: one safetensors file at a pinned commit -> ../../inputs/umt5/<name>/ (+ intake.json).
uv run --group wan --inexact python -m wan.umt5_intake --repo <owner/name> --revision <40-hex commit> \
    --file <file>.safetensors --name <name>

# 2. License review (by hand): record what the repository declares (license, base model, training
#    data) and your conclusion in ../../inputs/umt5/<name>/license-review.md. The intake never
#    writes this file and never judges the license; intake.json says NOASSERTION when none is declared.

# 3. Convert: the int8 series ../../outputs/series/umt5-xxl-<name>-i8-dyn/ (bfloat16 is widened to
#    float32 as it is read). Add --allow-undeclared-license only when intake.json records NOASSERTION.
uv run --group wan --inexact python -m wan.umt5_export write --intake ../../inputs/umt5/<name> [--allow-undeclared-license]
uv run --group wan --inexact python -m wan.umt5_export reference --intake ../../inputs/umt5/<name>   # optional: goldens

# 4. Local experiment mirror: ../../outputs/misc/local-dist/<name>/ (model <name>, quant i8). An --out
#    under models/ is refused, whatever the license.
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
`NOASSERTION`, the mirror has no `LICENSE.md`, and the card's first note, under the house
non-endorsement sentence every card carries, says "do not redistribute". That sentence names the
upstream repository and, as its publisher, the HF account it lives under (the record holds nothing
else about the publisher).
`tools/release/hf-upload.zsh upload` reads `provenance.license` from part 0 of every container before
uploading and refuses `NOASSERTION` in any letter case, `unknown` and `none`, as well as a directory
with no container at all, so a mirror copied or linked into `models/` by hand is stopped by its
contents. There is no publishing path for third-party encoders here.

#### Swapping the mirror into Wan (ADR 0122 stage d)

After step 4 above, swap the mirror into Wan. The command-line swap (`--swap-text-encoder` of the
`examples/wan` CLI, whose README explained the rules) was removed in 0.15 with no replacement yet;
in code, the swap is the load option `components: { text_encoder: { source } }` (ADR 0122 —
`packages/models/tests/wan_components_test.ts` exercises it).

The swap is admitted only if the mirror's `text_encoder` has the same graph description SHA-256 as
the one the Wan manifest declares (the `descriptor.graph.sha256` of `text_encoder` in both
`karume.json` files). The writer traces a meta model built from `config.json`, so an encoder whose
configuration matches `google/umt5-xxl` gets the same graph; a mismatch (for example a different
exporter, torch or transformers version) fails at construction before any weight is downloaded,
with `the graph descriptor does not match the manifest's declaration` and both hashes. Matching
graphs do not prove a matching tokenizer or relative-position buckets — see `docs/limitations.md`.

The mirror stays local. Before any upload, `tools/release/hf-upload.zsh upload <repo-dir-name>` runs
the pre-publish gate on `models/<repo-dir-name>`; the same check can be run by hand on any
directory, and exits 1 on a `NOASSERTION` container (run from `tools/export-recipes`, like the steps
above):

```bash
node ../release/container_license.ts ../../outputs/misc/local-dist/<name>
```

## Distribution (stage 7)

`wan/distribution.py` assembles the series into the distribution `models/karume-wan2.2/` (ADR 0121
stage 8 — model `ti2v-5b`, pipeline `wan-ti2v/1`; one repository per family generation, ADR 0092).
`karume dist` does the copying, hashing and verification; the recipe only says what goes where. The
text encoder is a cross-repository reference to `karume-umt5-xxl` (ADR 0119 stage 10d, decision A)
that the recipe declares (`wan_references` — model `xxl`, role `text_encoder`; ADR 0038 §7 addendum
2026-10-28), so build that repository first and give the build the commit it is pinned at (the
revision sources are in [`../README.md`](../README.md#dist-driver)):

```bash
uv run python dist.py --pipeline umt5
uv run python dist.py --pipeline wan-ti2v --resolve-refs hf    # the publication build
```

The distribution carries the int8 transformer (`wan2.2-ti2v-5b-i8-dyn`), the VAE graphs of
`wan2.2-ti2v-5b-f16-dyn`, and the two text assets of the Wan2.1 text series copied byte for byte —
their provenance is checked against the Wan2.1 pin, while the transformer and VAE containers are
checked against the Wan2.2 pin. The VAE graphs include the three encoder parts for image to video
(`vae_encoder_pre` / `vae_encoder_attn` / `vae_encoder_post`, f16, 107,887,956 B in all — ADR 0121
stage 9b); export them with `python -m wan.export_vae_encoder` (see the VAE section) before
assembling, as the plan needs all three. The decoder graphs also come in int8 from
`wan2.2-ti2v-5b-i8-dyn` (`model.i8.krm`, written by `python -m wan.export_vae --model ti2v-5b --dtype
i8`), and a 32-latent tile pair of each (ADR 0134) borrows the weights of the 16-latent pair.

A local reader resolves the text encoder reference through `crossRepo` and does not look at the
revision. A build without a revision source fails before writing anything, so a publishing build
has to pass the real SHA or `--resolve-refs hf` (`docs/release-runbook.md` §0); the placeholder
`0000000000000000000000000000000000000000` that only `--allow-placeholder-ref` produces is for a
development mirror.

| Manifest entry (`karume/6`) | Value |
| --------------------------- | ----- |
| model                       | `ti2v-5b` (the only one; pipeline `wan-ti2v/1`) |
| `weights`                   | `text_encoder` with one `i8` container referenced from `karume-umt5-xxl` (`xxl/text_encoder/model.i8-*.krm`, every part pinned by repo, commit, size and SHA-256); `transformer` with one `i8` container; `vae_decoder_first` / `vae_decoder_next` and their 32-latent pair with an `f16` and an `i8` container each; `vae_encoder_pre` / `vae_encoder_attn` / `vae_encoder_post` with one `f16` container each — the key is the container's graph name (container-v1 §2.1) |
| `assets`                    | `text_embeds` (`text_embeds/text_embeds.safetensors`) and `umt5_tokenizer` (`umt5_tokenizer/tokenizer.json` — format `karume-wan-umt5-tokenizer/1`), model-level and quant-independent. `rope_base` stays a container asset of `transformer` (ADR 0109) |
| `quants`                    | `f16+dit8` (the reference seat), `f16+dit8-a8-attn8-s16`, `f16+dit8-a8-attn8-online`, `i8` (the reference seat of the int8 VAE decoder) and `i8-a8-attn8-online-conv8` (default) — ADR 0121 decision 2, ADR 0132, ADR 0133. Every seat selects the `i8` text encoder, so every seat declares the vocabulary embedding's 1,050,148,864 bytes as `requiredLimits` |
| `pipelineConfig`            | `scheduler.shift` 5.0, `defaults.steps` 50, `defaults.guidance` 5.0 |

Before anything is placed, the plan checks that each container stores the format of its seat and
no other compressed format, names the pinned upstream revision and license in its provenance
(`wan/sources.py`), that the text encoder passes the same gates as in its own repository and that
its output width is the transformer's context width, that each VAE decoder pair belongs to one set
(the same latent shape, and the caches of `first` appear in `next` with the same names, shapes and
order — the rule the TypeScript loader applies), that each 32-latent container borrows every weight
from its lender, that the transformer declares the `rope_base` asset, that the embedding asset was
made from the pinned revision with the bfloat16 encoder and the pinned diffusers / ftfy, carries
exactly the prompts of `wan/prompts.py` with a normalized text for each that no other row claims,
and fits the transformer's `encoder_hidden_states [1, 512, 4096]` input, and that the tokenizer
asset has the format the TypeScript reader knows, the pinned source and transformers / ftfy
versions, and a `maxLength` that fits the context rows. The golden files of the series (`io.*`,
`reference.*`, `vae_*`, `pipeline_steps.*`) are never copied, and with the reference the text
encoder is not copied either (`karume dist` checks that the referenced parts are byte-identical to
the series container).

The repository root gets `LICENSE.md` (Apache 2.0, verbatim) and `NOTICE.md` (the changes made to
the upstream weights), and `README.md` is the model card rendered from the manifest by `wan/card.py`.
Re-running the command writes the same bytes.

## Tests

```bash
uv run --group wan --inexact pytest wan   # from tools/export-recipes/
vp test --project models:wan             # from the repository root (*wan*_test.ts in packages/models/tests and packages/e2e/tests, minus *wan_ti2v*)
vp test --project models:wan-ti2v         # Wan2.2 TI2V-5B (*wan_ti2v*_test.ts in the same two directories)
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
decreasing, the 1e-6 correction), the scheduler fixture being current, and the TI2V few-step
references on a small synthetic model (the tensors and metadata they write, the seeded noise, the
latents rebuilt bit for bit from the recorded DiT outputs).

The `models:wan` lane runs the family's host-function tests (including the UniPC / CFG fixture, the
DiT host fixture and the pipeline's input gates on a test generation) and the umT5 port check
(`e2e_wan_umt5_test.ts`). The real-GPU parity gates of Wan2.2 run in `models:wan-ti2v`; they skip
explicitly (with the generating command) when their assets or a GPU adapter are missing.

Tests that need the real weights take the `wan_snapshot` fixture (`wan/tests/conftest.py`) and skip
when the pinned snapshot is not in the HF cache. The Wan2.2 TI2V-5B tests take `wan22_snapshot` the
same way (the VAE ones are listed in the TI2V-5B subsection of the VAE section).

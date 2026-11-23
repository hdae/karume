> A point-in-time measurement record (2026-11-23): a torch-CPU screening of int8 weight forms for the vision and
> audio towers of EmbeddingGemma 2, against the embeddings of the checkpoint's own weights, run twice that day — first
> with every tensor each tower's bf16 series stores as bf16 rounded, then again with the form the `i8` series ships
> (the audio tower's two subsampling convolutions kept bf16, the maintainer's decision of that day). It is the
> evidence of the towers' part of the `i8` preset (step 3 of the EmbeddingGemma 2 item in [backlog](../backlog.md)).
> Every number is copied from the two runs' reports named below, from the exports' summaries, or from the decision
> of 2026-11-23 where a section says so. Nothing here is a design source of truth or a decision, and it is not
> edited to follow later events.

# EmbeddingGemma 2 tower quant screening (torch CPU fake-quant)

## What was measured and how

- **Script**: [`tools/export-recipes/embeddinggemma2/measure_quant_towers.py`](../../tools/export-recipes/embeddinggemma2/measure_quant_towers.py),
  run as `uv run --offline --with 'transformers==5.23.0' python -m embeddinggemma2.measure_quant_towers
  --quality-set ../../inputs/embeddinggemma2/quality-set --date 2026-11-23` from `tools/export-recipes/` (6 threads on
  the development machine's Ryzen 7600). The raw records (untracked; every per-item cosine at every width is in
  them):
  - **first run** (08:31): `outputs/bench/embeddinggemma-2/2026-11-23_tower-quant-screen/report.json`;
  - **re-run** (09:25, 6 min 43 s, with `--out …/2026-11-23_tower-quant-screen-shipped --write-clips
    ../../inputs/embeddinggemma2/quality-set-16k`): `outputs/bench/embeddinggemma-2/2026-11-23_tower-quant-screen-shipped/report.json`.

  Both runs' verification gates were green: each configuration rounded exactly its intended set, each tensor once,
  a restore gave the reference back bit for bit, and the reference reproduced the multimodal fixture's upstream
  vectors (`image-only` 5.87e-8, `audio-only` 6.98e-8, ids equal).
- **Model**: the pinned snapshot of `google/embeddinggemma-2` (revision `f9972ac2…`), loaded and wrapped by the
  exports' own code: the text graph's wrapper (`EmbeddingWrapper`), the vision tower's (`VisionTowerWrapper`) and the
  audio tower's (`AudioTowerWrapper`). The **reference** is those wrappers with the checkpoint's bf16 weights upcast
  to f32 and run in f32 (the goldens' arithmetic). An item is one input as the pipeline builds it: `<bos>`, the image
  or audio block alone (its begin marker, its soft tokens, its end marker), `<eos>`, no prompt. The captions and
  transcripts are embedded as text with the `query` prompt.
- **Configurations**: `vision-i8` (the vision tower rounded, measured on images), `audio-i8` (the audio tower rounded,
  measured on clips), `text-i8` (the text tower as the shipped `i8` preset rounds it — the token table and every
  linear, 219 tensors), `all-i8` (every tower an item passes through rounded: text + vision for an image, text +
  audio for a clip).
- **Rounding**: per-row symmetric int8 (`karume.quantize.fake_quant_int8`, one F32 scale per output row).
  - **Vision tower**, both runs: the 114 linear weights and the two learned position tables `pos_x` / `pos_y`
    (`[10240, 768]`, per position row) — 116 tensors, 167,706,624 elements, every tensor the bf16 series stores as
    bf16. The first run rounded the tables through a temporary `nn.Embedding` holding the wrapper's parameter; the
    re-run through the export's own rounding (`export_vision.round_weights_to_i8`, the tables named to
    `fake_quant_int8` as `embedding` weights). The images' numbers of the two runs are equal to every printed digit.
  - **Audio tower, first run**: every tensor the bf16 series stores as bf16 but the one-hot window selector (no
    weight; rounding it per row gives it back exactly) — the 123 linears (`input_proj_linear` included), the 12
    depthwise convolutions and the two subsampling convolutions: 137 tensors, 292,914,304 elements.
  - **Audio tower, re-run**: the shipped form (`export_audio.round_weights_to_i8`) — the 123 linears and the 12
    depthwise convolutions, 135 tensors, 292,876,288 elements; the two subsampling convolutions (`layer0.conv`
    `[128, 1, 3, 3]`, `layer1.conv` `[32, 128, 3, 3]`, 38,016 elements) and the selector stay bf16.
- **Metrics**, each at the widths 768, 512, 256 and 128 (all four are in the reports; 768 and 128 are tabulated
  here): the cosine of each item's embedding against the reference (min and mean over the quality set and over the
  synthetic items); the cosine of each soft token against the reference's (min and mean over every token); retrieval
  over the quality set — item → text among the distinct captions / transcripts and text → item among the items, top-1
  hits of the reference and of the configuration, top-1 agreement with the reference and the rank inversion rate over
  (query, candidate i < j) triples; the largest absolute change of any item × text cosine; the projected container
  payload.

## The quality set and its limits

`inputs/embeddinggemma2/quality-set/` (untracked; written by `tools/embeddinggemma2-quality-set/generate.ts`):

- **32 images**: 16 everyday subjects drawn twice, by Anima (`anima-turbo-v1.1`) and by MOSS-Image (`sft`) with the
  same subject, seed and size — 16 distinct captions, two styles each. Generated images only; no photograph, no text
  in images, no diagram.
- **32 clips**: 8 Japanese sentences spoken by the four SBV2 JVNV speakers (F1, F2, M1, M2) — 8 distinct transcripts.
  Japanese synthetic speech only: no other language, no real recording, no noise, music or non-speech sound. The
  files are 44,100 Hz mono; the script resamples them to 16 kHz with its own Kaiser-windowed sinc (32 zero crossings,
  beta 8.6, cutoff 0.95 of the lower Nyquist frequency). The re-run then puts every clip on the 16-bit PCM grid (the
  first run measured the resampled f32 samples): at most half a step, 1.53e-5, per sample, no sample clipped. The
  re-run wrote those samples as 16 kHz 16-bit mono WAV files with the images and a manifest of the same format to
  `inputs/embeddinggemma2/quality-set-16k/` (the inputs of the GPU check of the shipped preset; the library takes
  16 kHz only).
- **Synthetic items** besides it: the vision export's five golden images and the audio export's six golden
  waveforms (sweeps, noise, a click, tones, 30 s).

Retrieval among 16 captions or 8 transcripts is coarse: a top-1 flip is one of 32 or 8. Even the reference ranks the
right transcript first for 26 of the 32 clips only (item → text), so speech retrieval says little about the towers.

## Results — images (width 768; the two runs equal)

| config      | quality set min / mean | synthetic min / mean | soft tokens min / mean | item → text top-1 (ref) / agree / inversion | text → item top-1 (ref) / agree / inversion | max Δ cosine |
| ----------- | ---------------------- | -------------------- | ---------------------- | ------------------------------------------- | ------------------------------------------- | ------------ |
| `vision-i8` | 0.999743 / 0.999930    | 0.999786 / 0.999912  | 0.899765 / 0.999465    | 32/32 (32) / 32 / 0.47%                     | 16/16 (16) / 16 / 0.66%                     | 3.3923e-03   |
| `text-i8`   | 0.999844 / 0.999877    | 0.999896 / 0.999906  | 1 / 1                  | 32/32 (32) / 32 / 0.81%                     | 16/16 (16) / 16 / 0.69%                     | 3.9192e-03   |
| `all-i8`    | 0.999648 / 0.999812    | 0.999752 / 0.999834  | 0.899765 / 0.999465    | 32/32 (32) / 32 / 0.91%                     | 16/16 (16) / 16 / 0.91%                     | 4.0993e-03   |

At width 128 (quality set min / mean): `vision-i8` 0.999844 / 0.999960, `text-i8` 0.999904 / 0.999930, `all-i8`
0.999802 / 0.999892 — no further off than at 768.

## Results — speech (width 768)

| run       | config     | quality set min / mean | synthetic min / mean | soft tokens min / mean (worst clip) | item → text top-1 (ref) / agree / inversion | text → item top-1 (ref) / agree / inversion | max Δ cosine |
| --------- | ---------- | ---------------------- | -------------------- | ----------------------------------- | ------------------------------------------- | ------------------------------------------- | ------------ |
| first     | `audio-i8` | 0.998920 / 0.999873    | 0.995955 / 0.998783  | 0.201818 / 0.998685 (M2-02)         | 26/32 (26) / 32 / 0.11%                     | 8/8 (8) / 7 / 0.78%                         | 6.6724e-03   |
| first     | `text-i8`  | 0.999612 / 0.999821    | 0.999574 / 0.999696  | 1 / 1                               | 26/32 (26) / 32 / 0.45%                     | 8/8 (8) / 7 / 0.81%                         | 5.6032e-03   |
| first     | `all-i8`   | 0.998856 / 0.999699    | 0.996142 / 0.998580  | 0.201818 / 0.998685 (M2-02)         | 26/32 (26) / 32 / 0.33%                     | 8/8 (8) / 7 / 1.23%                         | 7.2324e-03   |
| re-run    | `audio-i8` | 0.999792 / 0.999935    | 0.999504 / 0.999801  | 0.814561 / 0.999431 (F2-06)         | 26/32 (26) / 32 / 0.22%                     | 8/8 (8) / 7 / 0.63%                         | 3.2830e-03   |
| re-run    | `text-i8`  | 0.999614 / 0.999821    | 0.999574 / 0.999696  | 1 / 1                               | 26/32 (26) / 32 / 0.56%                     | 8/8 (8) / 7 / 0.76%                         | 5.7086e-03   |
| re-run    | `all-i8`   | 0.999493 / 0.999759    | 0.999153 / 0.999521  | 0.814561 / 0.999431 (F2-06)         | 26/32 (26) / 32 / 0.33%                     | 8/8 (8) / 7 / 1.11%                         | 6.8936e-03   |

At width 128 (quality set min / mean; synthetic min): first run `audio-i8` 0.999364 / 0.999924 (0.997745),
`all-i8` 0.999309 / 0.999822 (0.997927); re-run `audio-i8` 0.999861 / 0.999960 (0.999804), `text-i8` 0.999743 /
0.999894, `all-i8` 0.999651 / 0.999858 (0.999651).

The re-run changed two things at once — the subsampling convolutions kept bf16 and the clips on the 16-bit grid —
so its differences from the first run are not attributed to one of them per item. The `text-i8` rows, whose towers
are not rounded, show what the 16-bit grid alone moved: the quality set's lowest cosine 0.999612 → 0.999614.

## The subsampling convolutions (the decision of 2026-11-23)

The first run's lowest soft-token cosines (M2-02 0.201818, M1-08 0.631504 with every tensor rounded) led to an
isolation of the audio tower's layer groups, made on torch CPU the same morning over the 32 clips and not kept as a
file; its numbers are those the decision of 2026-11-23 records. Rounding the two subsampling convolutions alone moved
single soft tokens to a cosine of 0.66 against the bf16 tower (`layer0.conv` alone, clip M1-08 token 68) and 0.20
(both, clip M2-02 token 26), while everything else rounded with them kept those tokens at 0.9998 or more;
`input_proj_linear` rounded alone moved nothing. The two convolutions hold 38,016 elements, so keeping them bf16
costs 37,376 bytes in the projection below (about 40 KB). The `i8` series keeps them bf16 and rounds
`input_proj_linear` (`export_audio.I8_KEPT_BF16_WEIGHTS` declares it, with that reason).

## Soft-token outliers

The lowest single soft-token cosines remain far below the final embeddings' in both towers: the final embedding
pools every token of the input through the text tower, and in the shipped form (the re-run) no item's embedding is
below 0.9995 at 768 in `vision-i8` or `audio-i8` (in the first run `audio-i8` reached 0.995955, the synthetic
`click`).

| run    | config      | the five lowest items: min token cosine (item's mean, tokens)                                                                                                                         | items with a token below 0.99 |
| ------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| both   | `vision-i8` | anima-storm 0.899765 (0.999011, 264), anima-bicycle 0.952143 (0.999399, 260), anima-cactus 0.953640 (0.999316, 260), anima-cat 0.966749 (0.999304, 260), moss-ramen 0.968233 (0.999433, 256) | 17 of 37                      |
| first  | `audio-i8`  | M2-02 0.201818 (0.969820, 35), M1-08 0.631504 (0.997618, 351), long 0.730038 (0.996618, 750), noise 0.931250 (0.997620, 63), F2-06 0.936679 (0.998992, 134)                            | 10 of 38                      |
| re-run | `audio-i8`  | F2-06 0.814561 (0.997198, 134), long 0.852937 (0.997961, 750), F1-08 0.950703 (0.999604, 344), noise 0.952138 (0.998681, 63), M2-03 0.966643 (0.998612, 48)                            | 6 of 38                       |

F2-06's worst token fell from 0.936679 to 0.814561 between the runs while the overall worst rose from 0.201818; which
of the two changes of the re-run did that was not measured.

## Container sizes

The screening's projection (16 bits per bf16 element, 8 per i8 element plus 32 per row for its scale, 32 per f32
element of the remainder) against the series the exports wrote the same day:

| tower  | form                                   | projected payload | series written                     |
| ------ | -------------------------------------- | ----------------- | ---------------------------------- |
| vision | bf16                                   | 335,853,840 B     | 336,093,696 B (4 parts)            |
| vision | i8                                     | 168,873,232 B     | 169,138,366 B (3 parts; 116 i8, 107 f32 initializers) |
| audio  | bf16                                   | 587,291,312 B     | 587,590,521 B (5 parts)            |
| audio  | i8, the convolutions rounded too        | 295,274,672 B     | —                                  |
| audio  | i8, the shipped form                    | 295,312,048 B     | 295,640,914 B (4 parts; 135 i8, 3 bf16, 148 f32 initializers) |

With both towers the `i8` preset downloads 798,296,220 bytes (761 MiB) against 1,526,175,876 for `bf16` (the text
graph and the tokenizer alone: 333,516,940 against 602,491,659). The local mirror holds 2,314,952,320 bytes in all.

## The i8 series' own checks (export, torch CPU)

The exports round before the reference and the goldens, so each i8 series' goldens are upstream's forward over the
rounded weights (the wrappers share every weight with upstream's modules). On the rounded towers:

- vision: the wrapper within 1.19e-6 to 1.43e-6 of upstream on the same rows and 1.86e-5 to 4.94e-5 of the golden
  (upstream on the processor's padded rows) over the five golden images — the bound is 1e-4;
- audio: the wrapper within 3.10e-6 (`click`) to 2.82e-4 (`long`) of the golden over the six golden waveforms — the
  bound is 3e-4, so `long` sits at 94% of it (the f32 series' worst is 1.01e-4, `tones`, and `long` there 6.51e-5).

## What was not measured

- Anything on a GPU: the i8 towers' series e2e, their sha256 rows, the pipeline over the mirror's `i8` preset with
  the towers, and its speed (the next stage).
- Photographs, real recordings, other languages, non-speech audio, video; retrieval among more than 16 captions or 8
  transcripts.
- Which of the re-run's two changes moved F2-06's worst token; the audio tower's layer groups beyond the isolation
  above; the effect of each layer group on the final embedding (the isolation read soft tokens).
- Whether the audio i8 wrapper's distance to upstream on `long` (2.82e-4) is f32 summation order alone — the f32
  series' wrapper equals upstream in f64 to 1.3e-12, but the rounded tower was not run in f64.

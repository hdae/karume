# EmbeddingGemma 2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)); it writes series and the
distribution `karume-embeddinggemma-2` (published as `hdae/karume-embeddinggemma-2` since 0.24.0;
the pin is `embeddinggemma.SOURCES["embeddinggemma-2"]` of `@karume/models`). The **text tower** of `google/embeddinggemma-2` is the distribution's one model, `text`;
the **vision tower** is exported as series of its own (`export_vision.py`) and ships in the same
model as the weights role `vision` (since 2026-11-22), and the **audio tower** is exported as series
of its own (`export_audio.py`) and ships in the same model as the weights role `audio` (since
2026-11-22). Each tower ships in the storage of each preset's text tower: bf16 in `bf16`, int8 in `i8`
(since 2026-11-23).

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstring of `export.py`; this file is the entry point only. Every file a
script reads is checked against the pin in `sources.py` first (the snapshot is fetched once into
`inputs/embeddinggemma2/embeddinggemma-2/`; the pin's docstring has the command).

## What `export.py` emits

The text tower, masked mean pooling (prompt tokens, `<bos>` and `<eos>` included) and L2
normalisation as **one graph**, whose output is a single unit vector `[1, 768]`. The inputs are
`input_ids` and `pool_mask` (all ones) `[1, T]`, and the soft rows of a tower: `soft_embeds`
`[1, T, 512]` f32 and `soft_mask` `[1, T, 1]` bool (the names and the mask form of the Gemma 4 12B
product graph). The rows the tower runs on are `where(soft_mask, soft_embeds, embed_tokens(input_ids))`:
a soft row takes the tower's soft token as it is, after the lookup's `sqrt(512)` scaling, as
upstream's composite model overwrites the scaled lookup (`masked_scatter`); the per-layer embedding
is projected from the merged rows, so nothing else changes. A text binds zeros and an all-false
mask, which `where` turns into the text forward bit for bit. The token length `T` is symbolic up
to 8,192, the upstream model's context (the model card's; no configuration file states it). Every layer is bidirectional:
the 20 sliding layers see `|q − kv| <= 512` through `attrs.band` 512 on their attention (ADR
[0155](../../../docs/decisions/0155-attention-band.md) — checked in the score kernel from
positions), the 4 full layers carry no mask, and grouped-query attention stays in its true form
(2 KV heads × 256 on the sliding layers, 1 KV head × 512 on the full ones). The export traces the
band as one `[1, 1, 8192, 8192]` constant (256 MiB) sliced by `sym_prefix_slice`, checks it against
the config's window, and declares the band in its place with `karume.band.to_band_form` — the
constant never reaches the container. The bound of `T` is carried by the folded RoPE tables
(`[1, 1, 8192, 256]` and `[1, 1, 8192, 512]`). Prompt prefixing and tokenization stay on the host.

```
outputs/series/embeddinggemma-2-text/model-NNNNN-of-NNNNN.krm  the container: graph + weights/constants as a part sequence
outputs/series/embeddinggemma-2-text/io.<case>.safetensors      input tensors and expected outputs from torch CPU (f32)
outputs/series/embeddinggemma-2-text-bf16/…                     the same with `--dtype bf16` (the distribution's `bf16` preset)
outputs/series/embeddinggemma-2-text-i8/…                       `--dtype i8`, with its own goldens (the distribution's `i8` preset)
```

Every run also writes the golden texts (case name, prompt name, body, T) to the git-tracked fixture
`packages/models/tests/fixtures/embeddinggemma2/golden-cases.json`, which the TypeScript pipeline's
e2e over the distribution reads (`packages/e2e/tests/e2e_embeddinggemma2_pipeline_test.ts`); a
recipe test holds it to `GOLDEN_CASES`. After editing the cases without an export, rewrite it with
`uv run python -c "from embeddinggemma2.export import write_golden_case_fixture; write_golden_case_fixture()"`.

`--dtype bf16` stores every weight of a weight slot (the token table and the linears) as bf16 and
the rest (norms, per-layer scalars, folded constants) as f32. The checkpoint is BF16, so nothing is
rounded (the writer stores a weight as bf16 only when the round trip is exact): measured on
2026-11-21, with the graph before the band was declared, all 427 tensors of the bf16 series decoded
bit for bit to the f32 series' values (219 of them, 270,925,824 elements, stored as bf16) and the
graph descriptors were equal. Measured on 2026-11-22: the bf16 series holds 426 initializers (219
bf16, 207 f32 — the band constant is gone) and is 592,932,727 bytes in 5 parts, and its ten io files
are byte-identical to the f32 series'.

`--dtype i8` rounds the same 219 weights (the token table and the 218 linears) to per-row symmetric int8 with
`karume.quantize.fake_quant_int8` **before** the window evidence, the trace and the goldens, so the series' goldens are
the quantized model's own outputs and its e2e tolerance measures implementation error alone (ADR
[0006](../../../docs/decisions/0006-quantization.md)). The script checks that the rounding reached exactly the linears
and the token table, and that the stored weight slots are exactly those weights as i8; the writer stores each with the
scale its rounding used and checks that `q · scale` gives the rounded value back bit for bit. Measured on 2026-11-22:
the i8 series holds 426 initializers (219 i8, 207 f32) in 323,958,008 bytes over 4 parts (the screening projected
323,468,384 for the payload alone), its graph descriptor equals the bf16 series', the window evidence reads 1.11e-2 to
4.97e-2 for the band and 0 against upstream's forward on the rounded model, and the cosine of each i8 golden to the f32
golden equals the screening's `i8-all` value for that text to 2e-16 (the same rounding). One run of the i8 export took
about 3 minutes on the development machine (CPU).

Ten golden cases: four short ones with and without a prompt, and six long documents at T 609 and
1,508 (past the first window cut at T = 514), 2,048 (the bound before the band was declared), 3,062,
5,935 and exactly 8,192 (the bound). Their texts, and those of the seven cases before them, are
unchanged when cases are added. Before anything is published the script checks the traced IR form
(the band constant equal to the config's inclusive window), that the band equals upstream's own
mask function, that the band really moves the long outputs and that its wrapper equals the upstream
model's own forward on them, the band form it writes (the band on exactly the sliding layers, no
mask left, the bound still readable at 8,192), and a cosine sanity order.

Measured on 2026-11-21 (the first three) and 2026-11-22 (the rest) (CPU, f32):

| Case          | T     | band vs every position visible (max abs) | wrapper vs upstream forward |
| ------------- | ----- | ---------------------------------------- | --------------------------- |
| document-609  | 609   | 1.09e-2                                  | 0 (output and hidden)       |
| document-1508 | 1,508 | 2.91e-2                                  | 0 (output and hidden)       |
| document-2048 | 2,048 | 3.89e-2                                  | 0 (output and hidden)       |
| document-3062 | 3,062 | 3.95e-2                                  | 0 (output and hidden)       |
| document-5935 | 5,935 | 5.09e-2                                  | 0 (output and hidden)       |
| document-8192 | 8,192 | 4.98e-2                                  | 0 (output and hidden)       |

Moving the window by one position (511 or 513) moves the first three outputs by 2.5e-4 to 5.6e-4
(2026-11-21). Declaring the band left the seven earlier io files byte-identical and the f32 series
is 1,134,785,941 bytes in 7 parts (f32 weights; 1,113,815,088 before, with the 2,048 bound). One run
of the f32 export takes about 3 minutes on the development machine (CPU). The real-GPU check is
`packages/e2e/tests/e2e_embeddinggemma2_test.ts` (lane `models:embeddinggemma2`, run with
`KARUME_SERIES=1`).

Adding the soft inputs (2026-11-22) left every output of the thirty io files (ten per storage)
byte-identical, and every container part but the first (the one holding the graph descriptor,
264 bytes longer) byte-identical: the f32 series is 1,134,786,205 bytes, the bf16 592,932,991 and the
i8 323,958,272. The export checks on every long case that the wrapper fed no soft row equals the
`input_ids` path bit for bit, and the form of the merge (one `where(soft_mask, soft_embeds, ·)`
whose text operand is the scaled lookup and is read by nothing else).

## What `export_vision.py` emits (the vision tower, stage 1 of image input)

Upstream's vision tower (`vision_config`, `model_type` `gemma4_vision`: 16 layers of 768, 12 heads
of 64, 2-D axial RoPE, the 3 × 3 average pooling and `× sqrt(768)`) and its projector `embed_vision`
(scale-free RMSNorm → linear 768 → 512) as **one graph** `vision`, whose output is ready to be the
text graph's soft rows. The authority for the design is the module docstring of `export_vision.py`.

```
outputs/series/embeddinggemma-2-vision/model-NNNNN-of-NNNNN.krm  the container (f32 weights)
outputs/series/embeddinggemma-2-vision/io.<image>.safetensors   the graph's inputs and upstream's soft tokens
outputs/series/embeddinggemma-2-vision/reference.json            per image: sizes, digests, soft tokens, measured differences
outputs/series/embeddinggemma-2-vision-bf16/…                    `--dtype bf16`: the linears and the position tables as bf16
outputs/series/embeddinggemma-2-vision-i8/…                      `--dtype i8`: the same tensors as per-row int8, its own goldens
```

### The io contract

| Name         | Dtype | Shape         | What the host binds                                                           |
| ------------ | ----- | ------------- | ----------------------------------------------------------------------------- |
| `pixels`     | f32   | `[1, 9n, 768]` | one patch per row: 16 × 16 × 3 values in (row, column, channel) order, each `u8 · f32(1/255)` |
| `position_x` | i32   | `[1, 9n]`     | the patch's column in the resized image's patch grid                          |
| `position_y` | i32   | `[1, 9n]`     | the patch's row                                                               |
| → output     | f32   | `[1, n, 512]` | the soft tokens, in the order of the image's placeholder rows                 |

`n` is the image's soft-token count (the graph's symbol, 139 to 280 at the snapshot's budget of 280);
the patch axis is `9n`. A patch row is exactly upstream's processor's row (it does not normalise —
the tower applies `2 · (x − 0.5)` itself). **The host passes only the image's real patches** (upstream
pads to 2,520 rows and masks them; the real rows' values do not depend on them), **in block order**:
with the resized image cut into a `W × H` patch grid (both multiples of 3), the rows are the 3 × 3
blocks in raster order (row of blocks by row of blocks, left to right), and inside each block its 9
patches in raster order:

    row i  →  block b = i // 9 (bx = b mod (W / 3), by = b div (W / 3)),  (dy, dx) = divmod(i mod 9, 3)
              position_x = 3 · bx + dx,  position_y = 3 · by + dy

Soft token `j` is block `j`, which is upstream's pooled order. The order is what makes the pooling a
reshape `[1, n, 9, 768]` and a mean over the 9 (upstream pools by positions with a one-hot product);
rows in another order still run and pool the wrong patches together — the processor's raster order
moves the soft tokens by 5.9 to 6.5 (measured in torch). Positions must be below 840: the 2-D RoPE is
two folded tables `[840, 32]` gathered by `position_x` / `position_y` (840 patches is the longest side
the resize rule gives at 280 soft tokens, 13,440 px), as are the two learned position tables
`[10240, 768]` split from upstream's one.

The smallest soft-token count of upstream's resize rule at the budget is **139**, not 140: in exact
arithmetic an image whose sides are 70:1 resizes to 2 × 140 blocks, but for 1190 × 17 the f64 `sqrt`
lands both sides a hair under their multiples of 48 and both round down (48 × 6672 px, 1 × 139
blocks). A host port of the rule in f64 (`Math.sqrt` is correctly rounded, like Python's) gives the
same counts.

### Checks and measurements

Before anything is published the script checks, on five synthetic images
(`images.py`: `gradient-square` 64 × 64 → 256 soft tokens, `shapes-wide` 128 × 64 → 253,
`noise-tall` 48 × 112 → 250, `stripes-edge` 1190 × 17 → 139, and `gradient-strip` 840 × 3 → 280,
whose 13,440 px side is the longest the rule gives, so its positions reach 839), that the gathered RoPE equals
upstream's rotary module bit for bit, that the wrapper's soft tokens equal upstream's within 1e-4,
the traced form (three inputs on `9n`, 16 attention nodes without a mask and with upstream's scale
1, the six lookups, no `where`) and the storage. Measured on 2026-11-22 (CPU, f32; soft tokens reach
5.0 to 6.2):

| Image           | n   | wrapper vs upstream (padded rows — the golden) | wrapper vs upstream (the same rows) | upstream: same rows vs padded |
| --------------- | --- | ---------------------------------------------- | ----------------------------------- | ----------------------------- |
| gradient-square | 256 | 2.00e-5                                        | 1.19e-6                             | 1.98e-5                       |
| shapes-wide     | 253 | 1.86e-5                                        | 1.67e-6                             | 1.86e-5                       |
| noise-tall      | 250 | 2.54e-5                                        | 1.19e-6                             | 2.50e-5                       |
| stripes-edge    | 139 | 3.87e-5                                        | 1.31e-6                             | 3.85e-5                       |
| gradient-strip  | 280 | 4.45e-5                                        | 1.43e-6                             | 4.45e-5                       |

The wrapper equals upstream on the same rows to the pooling's summation order (about 1e-6); what is
left against the golden is upstream's own f32 run over 2,520 padded rows against the real ones
(`gradient-strip` fills the budget, so it has no padding row — its difference is the processor's
raster order against the block order). The
tower is sensitive in f32: torch's own f32 run is up to 9.8e-5 from the same graph in f64. The real-GPU
check is `packages/e2e/tests/e2e_embeddinggemma2_vision_test.ts` (lane `models:embeddinggemma2`,
`KARUME_SERIES=1`): both series against the goldens at about three times each image's own error
(measured on the RTX 3080 Ti at 6.5e-5 to 3.35e-4, atol 2e-4 to 1e-3) with a floor on every soft
token's cosine; its comment has the table against f64. The two series are bit-identical to each
other, with sha256 rows per series.

Storage: the f32 series is 671,506,936 bytes in 5 parts; the bf16 series 336,093,696 bytes in 4 parts
(116 initializers bf16 — the 114 linears and the two learned position tables, the checkpoint's own
values — and 107 f32: the norms and the two RoPE tables, which sit in an `embedding` weight slot but
are not bf16 values and are kept f32 explicitly). Its goldens are the f32 series' (the io files are
byte-identical). One export takes about 35 s on the development machine (CPU).

`--dtype i8` (the distribution's `i8` preset, the text tower's method) rounds the same 116 tensors to per-row
symmetric int8 (`karume.quantize.fake_quant_int8`; the two position tables are bare parameters read by `embedding`,
named to it as `embedding` weights through `functional_weights`, one scale per position row) **before** the reference
and the goldens, so the series' goldens are upstream's forward on the rounded weights: the wrapper holds no copy of a
weight (the position tables are views of upstream's table, every other module upstream's own), so the rounding is
upstream's too. The RoPE tables stay f32. Measured on 2026-11-23: 169,138,366 bytes in 3 parts (116 initializers i8,
107 f32 — 167,706,624 elements rounded), the wrapper within 1.19e-6 to 1.43e-6 of upstream on the same rows and
1.86e-5 to 4.94e-5 of the golden. Making the wrapper share upstream's table left the bf16 series byte-identical (its
four parts, five io files and record re-exported that day). The torch-CPU screening of the form: [research 2026-11-23](../../../docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md).
On the RTX 3080 Ti (the same real-GPU check, 2026-11-23) the series sits 5.15e-5 to 3.56e-4 from its own goldens — the
f32 series' order — and is held at about three times each image's error, with sha256 rows of its own.

## What `export_audio.py` emits (the audio tower)

Upstream's audio tower (`audio_config`, `model_type` `gemma4_audio` — Gemma 4 E2B / E4B's, every
tensor unchanged: a subsampling of two stride-2 convolutions, 12 conformer layers of 1024 with 8
heads of 128, chunked local attention with a relative-position term and a soft cap of 50, clipped
linears, `output_proj` to 1536) and its projector `embed_audio` (scale-free RMSNorm → linear
1536 → 512) as **one graph** `audio`, whose output is ready to be the text graph's soft rows. The
authority for the design is the module docstring of `export_audio.py`. The distribution carries
the bf16 series as the weights role `audio` (below).

```
outputs/series/embeddinggemma-2-audio/model-NNNNN-of-NNNNN.krm  the container (f32 weights)
outputs/series/embeddinggemma-2-audio/io.<waveform>.safetensors the graph's inputs and upstream's soft tokens
outputs/series/embeddinggemma-2-audio/reference.json            per waveform: lengths, digests, soft tokens, measured differences
outputs/series/embeddinggemma-2-audio-bf16/…                    `--dtype bf16`: every linear and convolution weight as bf16
outputs/series/embeddinggemma-2-audio-i8/…                      `--dtype i8`: per-row int8 but the subsampling convolutions, its own goldens
```

### The io contract

| Name         | Dtype | Shape          | What the host binds                                                         |
| ------------ | ----- | -------------- | --------------------------------------------------------------------------- |
| `features`   | f32   | `[1, 4n, 128]` | the audio's `L` valid log-mel frames as upstream's feature extractor gives them, then `4n − L` zero frames |
| `frame_mask` | f32   | `[1, 4n]`      | 1 on the first `L` frames, 0 on the padding (upstream's `input_features_mask`) |
| → output     | f32   | `[1, n, 512]`  | the soft tokens, in the order of the audio's placeholder rows               |

`n = ceil(L / 4)` is the audio's soft-token count (the graph's symbol, 1 to 750), so the host pads
0 to 3 zero frames; every output row is a soft token (upstream keeps the same `n`). **The mask
matters exactly when `L ≡ 2 (mod 4)`**: the first convolution's last output frame then reads the
last real frame, and upstream masks it before the second convolution — dropped, the last token moves
by 4.2 (`edge`); at the other residues the mask changes no bit. A mask-free form exists only with
frame counts `4s + 2` and one output token more than the audio's for half the residues, so the mask
keeps the output at `n`. An audio of at most 160 samples has no valid frame and no soft token.

The frames (upstream's `Gemma4AudioFeatureExtractor`, f32 out — the host front end,
`packages/models/src/embeddinggemma/audio.ts`, computes the same steps with the windowed product in
f32 as upstream's and the FFT and the rest in f64, rounding once, so its frames are within one ulp
of upstream's; and it refuses an audio over 30 s or at another rate instead of truncating or
misreading it):

- 16 kHz mono f32 samples, **not resampled** (an array is read as 16 kHz whatever its rate),
  **truncated at 480,000 samples (30 s)** without a word; zero-padded on the right to a multiple of
  128 samples; 160 zeros prepended (semicausal);
- frames of 321 samples by a hop of 160, the first 320 of each used (preemphasis 0), times a
  periodic Hann window `0.5 − 0.5 · cos(2πk / 320)` stored as f32; `rfft` of length 512 (the frame
  zero-padded), its **magnitude**; an HTK mel bank of 128 triangles from 0 to 8,000 Hz
  (`2595 · log10(1 + f / 700)`, centres `linspace` over 130 mel points, FFT bins `linspace(0, 8000,
  257)`, no area normalisation, f64, filter 0 all zero); `log(mel + 1e-3)` (natural log; silence is
  −6.9077554); no per-bin normalisation. numpy multiplies the f32 samples by the f32 window in f32
  (each product rounded to f32) and runs the FFT in f32 (complex64), the mel product and the log in
  f64, then casts to f32. Of the two f32 steps only the product moves the frames visibly: left in
  f64 it shifts near-silent mel bins by up to 5.12e-4, while the f32 FFT adds one ulp;
- frame `i` is valid iff sample `160 · i + 160` exists, so `s` samples have
  `L = max(0, ceil(s / 160) − 1)` valid frames (1 s → 99 → 25 soft tokens, 30 s → 2,999 → 750); an
  invalid frame is zeros. The 128-sample padding can add one invalid frame, which the host drops.

### Checks and measurements

Before anything is published the script checks, on six synthetic waveforms (`waveforms.py`: every
residue of `L` mod 4, 1 s, 10 s and the 30 s maximum), that the wrapper's soft tokens equal
upstream's within 3e-4 (8e-4 for the i8 series, below), that the all-ones mask is bit-identical at the other residues and far off at
2, the traced form (two inputs on `4n`, no fused attention and no `where`, 12 softmaxes on the
12-key window, 24 window copies over one selector, one `[750, 12]` bias prefix) and the storage.
Measured on 2026-11-22 (CPU, f32; soft tokens reach 5.5 to 6.1):

| Waveform | Samples | L     | n   | wrapper vs upstream (the golden) | upstream: graph's frames vs extractor's | without the mask |
| -------- | ------- | ----- | --- | -------------------------------- | --------------------------------------- | ---------------- |
| sweep    | 16,000  | 99    | 25  | 1.55e-5                          | 0                                       | 0                |
| noise    | 40,000  | 249   | 63  | 7.44e-5                          | 0                                       | 0                |
| click    | 16,100  | 100   | 25  | 5.10e-6                          | 0                                       | 0                |
| edge     | 16,400  | 102   | 26  | 3.30e-5                          | 0                                       | 4.16             |
| tones    | 160,000 | 999   | 250 | 1.01e-4                          | 0                                       | 0                |
| long     | 480,000 | 2,999 | 750 | 6.51e-5                          | 0                                       | 0                |

The rewrite is exact in arithmetic: in f64 (upstream's f32 casts lifted) the wrapper equals upstream
to 1.3e-12, while upstream's own f32 run is up to 3.0e-4 from that f64 run and the wrapper's up to
3.4e-4 — the differences above are f32 summation order. The unit tests pin the window across
upstream's 12-query chunks on a tiny random tower (`tests/test_export_audio.py`, 1 to 40 soft
tokens and a partial fifth chunk): every layer gives nonzero weight to exactly the keys upstream's
returned attention weights do, and in f64 the wrapper's soft tokens equal upstream's within 1e-13
(measured 3.6e-15; a window one key off past the first chunk moves them by 2.2e-2 or more). The
real-GPU check is `packages/e2e/tests/e2e_embeddinggemma2_audio_test.ts` (lane
`models:embeddinggemma2`, `KARUME_SERIES=1`): both series against the goldens, each waveform at
about three times its own measured error, as an atol (4.2e-5 for `sweep` to 1.75e-3 for `noise`;
measured on the RTX 3080 Ti at 1.41e-5 to 5.82e-4, the GPU up to 5.0e-4 from f64) and as a floor on
every soft token's cosine (measured `1 − 3.7e-12` to `1 − 7.0e-9`) — its comment has the table —,
the two series bit-identical to each other, with sha256 rows per series, and the 30 s golden on a
device capped at 128 MiB (the WebGPU default binding and buffer limits) bit-identical to the
uncapped run.

On the RTX 3080 Ti under Node (2026-11-22, five warm runs after the first, the median):

| Audio            | n   | f32 series: warm run / first run | bf16 series: warm run / first run | plan backing |
| ---------------- | --- | -------------------------------- | --------------------------------- | ------------ |
| 1 s (`sweep`)    | 25  | 17.6 ms / 82 ms                  | 16.5 ms / 67 ms                   | 4.2 MiB      |
| 10 s (`tones`)   | 250 | 34.3 ms / 98 ms                  | 33.3 ms / 95 ms                   | 42.0 MiB     |
| 30 s (`long`)    | 750 | 94.4 ms / 171 ms                 | 94.2 ms / 154 ms                  | 126.0 MiB    |

A second run of the lane that day read the same within 0.1 to 4.9 ms. The Session builds in 1.02 to
1.22 s (f32, 1,119.1 MiB of weights) and 0.48 to 0.63 s (bf16, 560.1 MiB). The graph has 1,202 nodes (312 `clamp`, 127 `reshape`, 123 `linear`, 122 `mul`, 113
`permute`, 109 `rms_norm`, 72 `add`, 48 `sigmoid`, 36 each of `pad`, `conv1d` and `bmm`, 25 `slice`,
12 each of `div`, `tanh` and `softmax`, 2 each of `conv2d`, `layer_norm` and `relu`, one
`sym_prefix_slice`) and 286 initializers (293,353,708 elements). Its largest intermediate is the
first convolution's output `[1, 128, 2n, 64]` f32 (46.9 MiB at 750), its largest stored tensor 16 MiB
(f32) or 8 MiB (bf16): the tower needs no `requiredLimits`.

Storage: the f32 series is 1,173,714,150 bytes in 7 parts; the bf16 series 587,590,521 bytes in 5
parts (138 initializers bf16 — the 123 linears, the 2 subsampling and 12 depthwise convolutions and
the window selector, the checkpoint's own values or 0 and 1 — and 148 f32: the norms, the biases and
the folded constants). Its goldens are the f32 series' (the io files are byte-identical). One export
takes about 25 s on the development machine (CPU).

`--dtype i8` (the distribution's `i8` preset, the text tower's method) rounds every tensor the bf16 series stores as
bf16 to per-row symmetric int8 **before** the reference and the goldens — the 123 linears (`input_proj_linear`
included) and the 12 depthwise convolutions — **except** the two subsampling convolutions (`layer0.conv`
`[128, 1, 3, 3]`, `layer1.conv` `[32, 128, 3, 3]`), which stay bf16, and the window selector (no weight), which stays
bf16 too. The exception is declared once, `export_audio.I8_KEPT_BF16_WEIGHTS`, with its reason: rounded alone, the
two convolutions moved single soft tokens of speech to a cosine of 0.66 and 0.20 against the bf16 tower, while
everything else rounded with them kept those tokens at 0.9998 or more; keeping their 38,016 elements costs about
40 KB (the decision of 2026-11-23, [research 2026-11-23](../../../docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md)).
The wrapper's layers are copied with upstream's parameters shared, so the goldens are upstream's forward on the
rounded weights. Measured on 2026-11-23: 295,640,914 bytes in 4 parts (135 initializers i8, 3 bf16, 148 f32 —
292,876,288 elements rounded); the wrapper within 3.10e-6 (`click`) to 2.82e-4 (`long`) of the golden (the f32
series' worst is 1.01e-4). That difference is f32 noise of the rounded weights, not the rewrite: in f64 (upstream's
f32 casts lifted, as above) the wrapper equals upstream to 6.3e-13, while upstream's own f32 run is up to 4.0e-4 from
that f64 run (`long`) and the wrapper's up to 1.8e-4 (`noise`). So the series has a bound of its own, derived as the
f32 series' was — three times its worst difference, 8e-4 (`export_audio.UPSTREAM_TOLERANCE_I8`) — instead of sitting
at 94% of the bf16 weights' 3e-4. Re-exported with it that day, the series changed only the bound its
`reference.json` records (its four parts and six io files byte-identical), and the bf16 series exported again kept
every file byte-identical. Sharing upstream's parameters left the bf16 series byte-identical
(its five parts, six io files and record re-exported that day). On the RTX 3080 Ti (the same real-GPU check,
2026-11-23) the series sits 1.03e-5 to 2.49e-4 from its own goldens — the f32 series' order — and is held at about
three times each waveform's error, with sha256 rows of its own; its 30 s golden on the 128 MiB device is bit-identical
to the uncapped run.

## What `multimodal_reference.py` writes (the reference vectors of whole inputs)

The fixture `packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.{json,safetensors}`
(tracked in git, about 520 KB) for the TypeScript pipeline's stages: upstream's
`SentenceTransformer` (f32, CPU, the releases `parity.py` runs) on five whole inputs built from the
synthetic images — an image alone, a text then an image, an image then a text, a text with the
`query` prompt and an image, and two images (`cases`) — and on four holding an audio built from the
synthetic waveforms — an audio alone (`sweep`), a text then an audio (`click`), an audio with the
`query` prompt (`edge`, `L ≡ 2 mod 4`) and an image then an audio (`audioCases`, a list of its own,
so the image pipeline reads `cases` as before; adding them left the five image cases, their tensors
and every earlier field byte-identical). Each case holds the content parts and the prompt name, the
exact token ids the composite model received (`<bos>`, the text runs, each image as `<|image>` + n ×
`<|image|>` + `<image|>`, each audio as `<|audio>` + m × `<|audio|>` + `<audio|>`, `<eos>`), each
block's start and soft-token count, and the 768-wide vector; the images are the RGB8 bytes the
processor read, the audios the f32 samples it read (`audio.<name>`) with the valid log-mel frames its
feature extractor made of them (`audio.<name>.features` `[L, 128]` — the script checks they are the
frames the composite model received, bit for bit). Seven more inputs pin only the ids upstream runs
where a text run meets an image (`idCases`, no vector): a trailing space before an image, a leading
space after one, an empty text between two images, a newline and a tab at the boundaries, Japanese
around an image, the `query` prompt before an input that starts with an image, and the `document`
prompt before an image, a text and an image. Each is one user message of content parts in order
(upstream's chat form, which holds a repeated or an empty text part); their texts are pieces of the
tokenizer parity fixture's, so the host test checks them with that fixture's tokenizer, without the
mirror. Adding them and the fifth image left every earlier case, tensor and field byte-identical.
The JSON's `_doc` says how it was made.

The script also composes Karume's graphs in torch (the tower wrappers' soft tokens placed at the
placeholder rows of the text wrapper, over the ids upstream ran) and holds them to upstream's
vectors within 1e-5. Measured on 2026-11-22: max abs 5.9e-8 to 1.8e-6 for the image cases and
5.0e-8 to 1.2e-7 for the audio cases, cosine at least 0.99999999991 (`composition` of each case).

## What `audio_front_end.py` writes (the host front end's fixture)

`packages/models/tests/fixtures/embeddinggemma2/audio-front-end.json` (tracked in git) ties the TypeScript front end to
upstream without the snapshot: the settings of the pinned `preprocessor_config.json` that change a frame, as the file
states them; the extractor's `max_length` call default (480,000 samples); and per waveform of the multimodal fixture
(`sweep`, `click`, `edge`) the SHA-256 of the `[4n, 128]` f32 tower input by upstream's arithmetic with the FFT in
f64 — the f32 product of the samples and the f32 window, numpy's `rfft` in f64, the magnitude, the f64 mel product,
`log(mel + 0.001)`, cast to f32. Before writing, the script checks that the same steps with the FFT in f32 are
upstream's extractor's run bit for bit. `tests/test_audio_front_end.py` holds `export_audio`'s settings and
`MAX_SAMPLES` and the distribution's mirror to the file (no snapshot or transformers needed) and, with transformers at
the pinned release, recomputes the digests; `packages/models/tests/embeddinggemma2_audio_input_test.ts` holds
`AUDIO_FEATURE_EXTRACTOR` and `AUDIO_MAX_SAMPLES` to it and its frames' SHA-256 to the digests.

## What `tokenizer.py` emits

The compiled tokenizer asset `outputs/series/embeddinggemma-2-tokenizer/tokenizer.json` and the
git-tracked parity fixture `packages/models/tests/fixtures/gemma-text/embeddinggemma-2-parity.json`,
through the same machinery as gemma4 and the first EmbeddingGemma (`_shared/gemma_tokenizer.py`).
The vocabulary and merges are Gemma 4's; the post-processor wraps the sequence in `<bos>` … `<eos>`.

## What `parity.py` measures (opt-in)

SentenceTransformer loaded from the pinned snapshot against the wrapper, on the golden texts (text
only). It writes nothing and runs in no default lane. Measured on 2026-11-21:

| Comparison                                         | lowest cosine | largest max abs |
| -------------------------------------------------- | ------------- | --------------- |
| SentenceTransformer f32 vs wrapper                 | 1.0000000     | 5.96e-8         |
| SentenceTransformer bf16 vs wrapper (f32)          | 0.9998619     | 2.22e-3         |
| wrapper with the bf16 embedding scale (22.625)     | 0.9999999     | 2.03e-5         |

The bf16 gap comes almost entirely from bf16 arithmetic, not from the embedding scale. A text
holding the image placeholder is outside the goldens' domain: SentenceTransformer refuses it, and
the composite model fed the ids directly swaps the placeholder for the pad id, which the text tower
does not (max abs 9.25e-5 on one such text; 0 once the pad id is substituted).

## The distribution (`distribution.py`, `card.py`)

`dist.py --pipeline embeddinggemma-2` assembles `models/karume-embeddinggemma-2/` (the directory
`tools/release/hf-upload.zsh` uploads to `hdae/karume-embeddinggemma-2`): the model `text` (contract `embeddinggemma/1`, read by the TypeScript
family `embeddinggemma`) with the graph as the weights `model` in two storages — `bf16` from the bf16 series and `i8`
from the i8 series, each container checked to carry the bf16 one's graph —, the vision tower as the weights `vision` and
the audio tower as the weights `audio` in the same two storages (`bf16` from the series `embeddinggemma-2-vision-bf16`
/ `embeddinggemma-2-audio-bf16`, `i8` from `embeddinggemma-2-vision-i8` / `embeddinggemma-2-audio-i8`, each tower's
i8 container checked to carry its bf16 one's graph; the pipeline fetches a tower only with `vision: true` /
`audio: true`), and the compiled tokenizer as the asset `tokenizer`, two presets with an empty session, each mapping
every graph to its own storage (`bf16`, the original weights and the default; `i8`, the int8 storage of the three
graphs), `LICENSE.md` (the Apache-2.0 text) and `NOTICE.md` (the modification notice, which states the int8 copies,
the audio tower's kept convolutions and the two tower graphs). The card's title
and text follow the towers the manifest ships. Its `pipelineConfig` carries the model's own values, each read from its source and
checked against the others when the plan is built:

| Field               | Value                                         | Source and check                                                                                      |
| ------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `prompts`           | prompt name → prefix (20 names, 8 prefixes)   | `config_sentence_transformers.json` at the pin, verbatim; no default prompt, cosine similarity       |
| `dimensions`        | `[768, 512, 256, 128]`                        | the upstream model card; the first equals the graph's output width                                    |
| `maxTokens`         | `8192`                                        | the bound the graph's folded RoPE tables carry for `T`, held to the export's `SYM_MAX`               |
| `placeholderTokens` | `{image: 258880, audio: 258881, video: 258884}` | `config.json` at the pin; each a special token of the tokenizer asset                                 |
| `vision`            | `{softTokenBudget: 280, imageTokenId: 258880, beginImageTokenId: 255999, endImageTokenId: 258882, hiddenSize: 512}` (the Gemma 4 family's shape) | `processor_config.json`'s `max_soft_tokens`; `config.json`'s `image_token_id` / `boi_token_id` / `eoi_token_id` (each a special token of the asset, distinct); `text_config.hidden_size`, held to the text graph's soft rows and the tower's output |
| `audio`             | `{audioTokenId: 258881, beginAudioTokenId: 256000, endAudioTokenId: 258883, hiddenSize: 512, maxSoftTokens: 750, sampleRate: 16000}` | `config.json`'s `audio_token_id` / `boa_token_id` / `eoa_token_index` (each a special token of the asset, distinct); `text_config.hidden_size`, held to the tower's output; the audio graph's bound of `n` (its folded window bias), held to the soft tokens of the feature extractor's longest audio (480,000 samples, `export_audio.MAX_SAMPLES` mirrored) and to the audio series' record; `preprocessor_config.json`'s `sampling_rate` |

The plan also checks each storage (`model_bf16` must hold bf16 and none of the other compressed layouts, `model_i8`
the same for i8 — so the f32 series, or the two series swapped, are refused; `vision_bf16` / `audio_bf16` the same as
`model_bf16`, `vision_i8` the same as `model_i8`, `audio_i8` i8 and bf16 beside it — its bf16 tensors exactly the two
subsampling convolutions and one window selector, the export's declared exceptions),
that the i8 container carries the bf16 container's graph, the graph's inputs and output, the tokenizer asset against the
pinned `tokenizer.json` and the graph's vocabulary, each container's licence and upstream revision against the pin, the
processor's image settings against what the host implements (`Gemma4ImageProcessor`, bicubic, the 1/255 rescale, no
normalisation), the tower graph (`pixels` f32 `[1, 9n, 768]`, `position_x` / `position_y` i32 `[1, 9n]`, one f32
`[1, n, 512]`, position tables covering `3 · budget` patches) and the vision series' record (its series, its budget
and the pin); for the audio tower, the feature extractor's settings in both
`preprocessor_config.json` and `processor_config.json`'s `feature_extractor` block against what the host front end
implements (`Gemma4AudioFeatureExtractor`, 16 kHz, 128 mel filters 0–8,000 Hz, frames of 320 by 160, an FFT of 512, a
mel floor of 0.001, and no preemphasis, dither, input scale, per-bin normalisation or FFT overdrive — a `False` is not
taken for a `0`), the tower graph (`features` f32 `[1, 4n, 128]`, `frame_mask` f32 `[1, 4n]`, one f32 `[1, n, 512]`,
the bound of `n` 750) and the audio series' record (its series, the pin, the sample rate and `softTokensMax`).

`processor_config.json` also sets `audio_seq_length` to 280 (its processor's default is 750). Upstream reads it only in
its serving helper (`_compute_audio_num_tokens`, the count `_get_num_multimodal_tokens` reports to a vLLM-style
server), which caps an audio's soft tokens there — about 11.2 s at 40 ms a token. The processor path that
sentence-transformers runs (`replace_audio_token`, the count from the frame mask) has no cap, and Karume follows it: up
to 750 soft tokens for 30 s (a 30 s clip is 754 ids in sentence-transformers and here). The plan reads the value and
holds it to the recorded 280 (`EMBEDDINGGEMMA2_AUDIO_SEQ_LENGTH`), so a snapshot that changes it fails the build and
the difference (`docs/limitations.md`) is looked at again.

Assembled again on 2026-11-23 with the towers' i8 storages, the mirror holds 2,314,952,320 bytes in all (the vision
i8 container 169,138,366 more, the audio i8 container 295,640,914): every file of the `bf16` preset (its fourteen
container parts and the tokenizer) and the text tower's i8 container are byte-identical to the assembly before it,
and in `karume.json` every entry of the `bf16` preset is unchanged; the `i8` preset now maps both towers to their i8
containers and downloads 798,296,220 bytes with both towers (333,516,940 without them). Assembled once more that day
with the `i8` preset's description reworded to cover the towers, only `karume.json` (that one string) and `README.md`
(its quants row) changed; every container part and the tokenizer kept its sha256 (2,314,952,503 bytes in all). Assembled on 2026-11-22 with
the audio tower, the mirror held 1,850,169,018 bytes in all (the audio
container 587,590,521 more); the text and vision containers and the tokenizer are byte-identical to the assembly
before it, and each preset downloads 587,590,521 bytes more for the tower with `audio: true`. Assembled before with
the vision tower, it held 1,262,574,677 bytes (the containers 592,932,991 + 323,958,272 + 336,093,696 and the
9,558,668-byte tokenizer); the text containers were byte-identical to the assembly before that. The `bf16` preset downloads 592,932,991 + 9,558,668 bytes, the `i8` preset 323,958,272 +
9,558,668, and each 336,093,696 more for the tower with `vision: true`. Before the tower, with the soft inputs, it held
926,477,217 bytes. The dist driver
derives `requiredLimits` `{ maxStorageBufferBindingSize: 268435456 }` for `bf16` (its bf16 token table, 256 MiB) and
none for `i8` (its int8 table is 134,217,728 bytes, exactly the WebGPU default, which the bake leaves out; the vision
tower's largest table, 15 MiB of bf16 or 7.5 MiB of i8, and the audio tower's, 8 MiB of bf16 or 4 MiB of i8, add
none). Against the
mirror before the soft inputs, only the first part of each container and the graph descriptor's length and digest
in `karume.json` changed.

How far `i8` is from the original weights, as measured once on a small set (not a benchmark): the torch-CPU screening
of [research 2026-11-22](../../../docs/research/2026-11-22-embeddinggemma2-quant-screen.md) (configuration `i8-all`,
the rounding the i8 series stores) put the cosine to the original weights' embeddings at 0.999396 at the lowest and
0.999842 on average over 58 texts at 768 values, with 0.74% of the 3,800 ranking triples inverted; on the GPU the
cosine of each golden text's shipped `i8` vector to its `bf16` vector equals the screening's value for that text to
the seventh digit (the per-text values are in the screening's untracked `report.json`). The towers' int8 forms were
screened the same way on 32 generated images and 32 clips of Japanese synthetic speech
([research 2026-11-23](../../../docs/research/2026-11-23-embeddinggemma2-tower-quant-screen.md), configuration
`all-i8`, the shipped form): lowest cosine 0.999648 and mean 0.999812 over the images, 0.999493 and 0.999759 over the
clips, at 768 values. The card prints both screenings' numbers with their sources.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.tokenizer
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype bf16
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype i8
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision --dtype bf16
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision --dtype i8
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio --dtype bf16
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_audio --dtype i8
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.audio_front_end
uv run python dist.py --pipeline embeddinggemma-2
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.measure_quant_towers \
    --quality-set ../../inputs/embeddinggemma2/quality-set \
    --write-clips ../../inputs/embeddinggemma2/quality-set-16k
uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \
    python -m embeddinggemma2.parity
uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \
    python -m embeddinggemma2.multimodal_reference
```

`transformers` is pinned to 5.23.0 (a change in the modeling code changes the graph's form, and
EmbeddingGemma 2 first ships in 5.19.0); it is brought in per run with `--with` rather than declared
as a group, and each script refuses any other release. `sentence-transformers` 6.1.0 (the release
the snapshot declares) is needed by `parity.py` and `multimodal_reference.py` alone.

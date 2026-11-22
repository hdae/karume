# EmbeddingGemma 2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)); it writes series and the
distribution `karume-embeddinggemma-2`, which is built as a local mirror only (nothing is
published). The **text tower** of `google/embeddinggemma-2` is the distribution's one model, `text`;
the **vision tower** is exported as series of its own (`export_vision.py`) and ships in the same
model as the weights role `vision` (since 2026-11-22), and the audio tower is never built.

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

Before anything is published the script checks, on four synthetic images
(`images.py`: `gradient-square` 64 × 64 → 256 soft tokens, `shapes-wide` 128 × 64 → 253,
`noise-tall` 48 × 112 → 250, `stripes-edge` 1190 × 17 → 139), that the gathered RoPE equals
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

The wrapper equals upstream on the same rows to the pooling's summation order (about 1e-6); what is
left against the golden is upstream's own f32 run over 2,520 padded rows against the real ones. The
tower is sensitive in f32: torch's own f32 run is up to 9.8e-5 from the same graph in f64. The real-GPU
check is `packages/e2e/tests/e2e_embeddinggemma2_vision_test.ts` (lane `models:embeddinggemma2`,
`KARUME_SERIES=1`): both series against the goldens within atol 1e-3 (measured on the RTX 3080 Ti at
6.5e-5 to 3.35e-4; its comment has the table against f64), the two series bit-identical to each
other, with sha256 rows per series.

Storage: the f32 series is 671,506,936 bytes in 5 parts; the bf16 series 336,093,696 bytes in 4 parts
(116 initializers bf16 — the 114 linears and the two learned position tables, the checkpoint's own
values — and 107 f32: the norms and the two RoPE tables, which sit in an `embedding` weight slot but
are not bf16 values and are kept f32 explicitly). Its goldens are the f32 series' (the io files are
byte-identical). One export takes about 35 s on the development machine (CPU).

## What `multimodal_reference.py` writes (the reference vectors of whole inputs)

The fixture `packages/models/tests/fixtures/embeddinggemma2/multimodal-reference.{json,safetensors}`
(tracked in git, about 140 KB) for the TypeScript pipeline's next stage: upstream's
`SentenceTransformer` (f32, CPU, the releases `parity.py` runs) on five whole inputs built from the
synthetic images — an image alone, a text then an image, an image then a text, a text with the
`query` prompt and an image, and two images. Each case holds the content parts and the prompt name,
the exact token ids the composite model received (`<bos>`, the text runs, each image as `<|image>` +
n × `<|image|>` + `<image|>`, `<eos>`), each image's block start and soft-token count, and the
768-wide vector; the images are the RGB8 bytes the processor read. The JSON's `_doc` says how it was
made.

The script also composes Karume's two graphs in torch (the tower wrapper's soft tokens placed at the
image's placeholder rows of the text wrapper, over the ids upstream ran) and holds them to upstream's
vectors within 1e-5. Measured on 2026-11-22: max abs 5.9e-8 to 1.8e-6, cosine at least
0.99999999991 (`composition` of each case).

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

`dist.py --pipeline embeddinggemma-2` assembles `models/karume-embeddinggemma-2/` (local only — no
upload, no published pin): the model `text` (contract `embeddinggemma/1`, read by the TypeScript
family `embeddinggemma`) with the graph as the weights `model` in two storages — `bf16` from the bf16 series and `i8`
from the i8 series, each container checked to carry the bf16 one's graph —, the vision tower as the weights `vision` in
one storage, `bf16`, from the series `embeddinggemma-2-vision-bf16` (so both presets map it — no rounded tower was
screened, and the card says so; the pipeline fetches it only with `vision: true`), and the compiled tokenizer as the
asset `tokenizer`, two presets with an empty session (`bf16`, the original weights and the default; `i8`, the int8
storage of the text tower), `LICENSE.md` (the Apache-2.0 text) and `NOTICE.md` (the modification notice, which states
the int8 copy and the vision graph). Its `pipelineConfig` carries the model's own values, each read from its source and
checked against the others when the plan is built:

| Field               | Value                                         | Source and check                                                                                      |
| ------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `prompts`           | prompt name → prefix (20 names, 8 prefixes)   | `config_sentence_transformers.json` at the pin, verbatim; no default prompt, cosine similarity       |
| `dimensions`        | `[768, 512, 256, 128]`                        | the upstream model card; the first equals the graph's output width                                    |
| `maxTokens`         | `8192`                                        | the bound the graph's folded RoPE tables carry for `T`, held to the export's `SYM_MAX`               |
| `placeholderTokens` | `{image: 258880, audio: 258881, video: 258884}` | `config.json` at the pin; each a special token of the tokenizer asset                                 |
| `vision`            | `{softTokenBudget: 280, imageTokenId: 258880, beginImageTokenId: 255999, endImageTokenId: 258882, hiddenSize: 512}` (the Gemma 4 family's shape) | `processor_config.json`'s `max_soft_tokens`; `config.json`'s `image_token_id` / `boi_token_id` / `eoi_token_id` (each a special token of the asset, distinct); `text_config.hidden_size`, held to the text graph's soft rows and the tower's output |

The plan also checks each storage (`model_bf16` must hold bf16 and none of the other compressed layouts, `model_i8`
the same for i8 — so the f32 series, or the two series swapped, are refused; `vision_bf16` bf16 and no rounded layout),
that the i8 container carries the bf16 container's graph, the graph's inputs and output, the tokenizer asset against the
pinned `tokenizer.json` and the graph's vocabulary, each container's licence and upstream revision against the pin, the
processor's image settings against what the host implements (`Gemma4ImageProcessor`, bicubic, the 1/255 rescale, no
normalisation), the tower graph (`pixels` f32 `[1, 9n, 768]`, `position_x` / `position_y` i32 `[1, 9n]`, one f32
`[1, n, 512]`, position tables covering `3 · budget` patches) and the vision series' record (its series, its budget
and the pin).

Assembled again on 2026-11-22 with the vision tower, the mirror holds 1,262,574,677 bytes in all (the containers
592,932,991 + 323,958,272 + 336,093,696 and the 9,558,668-byte tokenizer); the text containers are byte-identical to the
assembly before it. The `bf16` preset downloads 592,932,991 + 9,558,668 bytes, the `i8` preset 323,958,272 +
9,558,668, and each 336,093,696 more for the tower with `vision: true`. Before the tower, with the soft inputs, it held
926,477,217 bytes. The dist driver
derives `requiredLimits` `{ maxStorageBufferBindingSize: 268435456 }` for `bf16` (its bf16 token table, 256 MiB) and
none for `i8` (its int8 table is 134,217,728 bytes, exactly the WebGPU default, which the bake leaves out; the tower's
largest table, 15 MiB of bf16, adds none). Against the
mirror before the soft inputs, only the first part of each container and the graph descriptor's length and digest
in `karume.json` changed.

How far `i8` is from the original weights, as measured once on a small set (not a benchmark): the torch-CPU screening
of [research 2026-11-22](../../../docs/research/2026-11-22-embeddinggemma2-quant-screen.md) (configuration `i8-all`,
the rounding the i8 series stores) put the cosine to the original weights' embeddings at 0.999396 at the lowest and
0.999842 on average over 58 texts at 768 values, with 0.74% of the 3,800 ranking triples inverted; on the GPU the
cosine of each golden text's shipped `i8` vector to its `bf16` vector equals the screening's value for that text to
the seventh digit (the per-text values are in the screening's untracked `report.json`). The card prints the
screening's numbers with that source.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.tokenizer
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype bf16
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype i8
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export_vision --dtype bf16
uv run python dist.py --pipeline embeddinggemma-2
uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \
    python -m embeddinggemma2.parity
uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \
    python -m embeddinggemma2.multimodal_reference
```

`transformers` is pinned to 5.23.0 (a change in the modeling code changes the graph's form, and
EmbeddingGemma 2 first ships in 5.19.0); it is brought in per run with `--with` rather than declared
as a group, and each script refuses any other release. `sentence-transformers` 6.1.0 (the release
the snapshot declares) is needed by `parity.py` and `multimodal_reference.py` alone.

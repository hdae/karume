# EmbeddingGemma (host side)

The `./embeddinggemma` subpath of `@karume/models`: `embeddinggemma.Pipeline` turns a text, an image,
an audio or any mix of them into a sentence embedding — a unit vector, so the dot product of two
results is their cosine similarity, an image's, an audio's and a text's alike. One family serves the EmbeddingGemma models; today it
serves EmbeddingGemma 2 (`google/embeddinggemma-2`, Apache-2.0 — the distribution
`karume-embeddinggemma-2`, model `text`, pipeline `embeddinggemma/1`, presets `bf16` — the default, the
checkpoint's own weights — and `i8` — the text tower's weight slots as per-row int8; both compute in
f32 and declare no knob, and `quant` picks one). Its vision tower is the weights role `vision`, the
checkpoint's bf16 values in both presets, built only with `vision: true`; its audio tower likewise
the weights role `audio`, built only with `audio: true`. The first EmbeddingGemma
joins the family later with its own manifest values.
The public surface is [`embeddinggemma.ts`](../../embeddinggemma.ts) (also re-exported from the
barrel as the namespace `embeddinggemma`); everything here is internal.

```ts
import { Pipeline } from "@karume/models/embeddinggemma";
import { nodeDirectory } from "@karume/hub/node";

const pipeline = await Pipeline.fromPretrained(nodeDirectory("models/karume-embeddinggemma-2"), {
  gpu,
  vision: true, // also fetch the vision tower, for image parts
  audio: true, // also fetch the audio tower, for audio parts
});
const query = await pipeline.embed("how do I reset my password", { prompt: "query", dimension: 256 });
const count = pipeline.tokenize("how do I reset my password", { prompt: "query" }).length;
// image: decoded RGB8 — { data: Uint8Array (width × height × 3), width, height }
const picture = await pipeline.embed([{ type: "image", image }], { dimension: 256 });
const captioned = await pipeline.embed([{ type: "text", text: "a red bicycle" }, { type: "image", image }]);
// audio: mono f32 at 16 kHz, at most 30 s — { data: Float32Array, sampleRate: 16000 } (decodeWav's shape)
const clip = await pipeline.embed([{ type: "audio", audio }]);
await pipeline.dispose();
```

The distribution is not published: the recipe `tools/export-recipes/embeddinggemma2/distribution.py`
assembles a local mirror under `models/`, and there is no `SOURCES` table until the first
publication (docs/release-runbook.md §3). A local distribution is passed to `fromPretrained` as a
source handle (`nodeDirectory` / `denoDirectory`); `fromAssets` takes the manifest and the bytes.

## One call

1. **Host** — the input becomes token ids, upstream's sentence-transformers sequence: `<bos>`, then
   the parts in order — each run of adjacent text parts, the prompt's prefix in front of the first
   part, tokenized as one string by the Gemma tokenizer of `src/gemma/text/` (the distribution's
   asset); each image as `<|image>` (255999), n × the image placeholder 258880 and `<image|>`
   (258882), and each audio as `<|audio>` (256000), n × the audio placeholder 258881 and `<audio|>`
   (258883), with no separator — then `<eos>`. A text-only input is the prefix and the joined text,
   tokenized once, as before images. An image's n comes from its size alone: the aspect-preserving
   resize rule of the Gemma 4 family (`imageTargetSize` — largest multiples of 48 px within
   `pipelineConfig.vision.softTokenBudget`, upstream's default 280, in f64 as upstream computes it:
   139 to 280 soft tokens). An audio's n comes from its length alone: `ceil(L / 4)` of its
   `L = ceil(samples / 160) − 1` valid mel frames (`audio.ts` — 25 for 1 s, 750 for 30 s). An input
   that starts with an image or an audio under a prompt gets the prefix as a run of its own before it
   (sentence-transformers renders the prompt as a system message before the first part).
2. **Host, at issue** — each image becomes the tower's input (`vision.ts`): torch's uint8
   antialiased bicubic resize (`src/image/torch-resample.ts`) to the rule's size, the image's real
   16 × 16 patches as rows of 768 values (row, column, channel; each `f32(u8 · f32(1/255))`) in block
   order (the 3 × 3 blocks in raster order, each block's 9 patches in raster order), and each row's
   column and row in the patch grid. Each audio becomes its log-mel frames (`audio.ts` — upstream's
   `Gemma4AudioFeatureExtractor` as the snapshot configures it: 160 zeros prepended, frames of 320
   samples by a hop of 160 times a periodic Hann window, an rfft of 512, the magnitude, 128 HTK mel
   filters from 0 to 8,000 Hz, `ln(mel + 0.001)`; every step in f64 with the FFT of
   `src/audio/fft.ts`, rounded to f32 once), padded with zero frames to `4n`, and its frame mask (1
   on the valid frames). Both are read when `embed` is called, so a later write into the pixels or
   the samples does not reach the call.
3. **GPU, images and audios only** — one tower run per block, in sequence order: the graph `vision`
   per image (`pixels` `[1, 9n, 768]`, `position_x` / `position_y` `[1, 9n]` → `[1, n, 512]`) and the
   graph `audio` per audio (`features` `[1, 4n, 128]`, `frame_mask` `[1, 4n]` → `[1, n, 512]`); each
   output is copied into `soft_embeds` at its block's soft rows, with `soft_mask` set there.
4. **GPU** — the graph `model` runs once on `input_ids` `[1, T]` (the placeholder id stays under each
   soft row), `pool_mask` `[1, T]` (all ones — the pooling averages every token, the prompt's and
   the soft ones included, as upstream's `include_prompt`), `soft_embeds` `[1, T, 512]` and
   `soft_mask` `[1, T, 1]` (zeros and all false for a text — the graph then computes the text forward
   bit for bit). A soft row runs on its soft token, unscaled, as upstream's composite model overwrites
   the scaled lookup. The graph holds the text tower, the masked mean pooling and the L2
   normalisation, and returns `[1, D]`.
5. **Host** — a shorter width is the leading slice divided by `max(norm, 1e-12)` (computed in f64,
   stored as f32 — sentence-transformers' `truncate_dim` followed by its normalisation).

The Sessions (the text graph's, the vision tower's with `vision: true` and the audio tower's with
`audio: true`) are opened at construction and kept until `dispose`; calls queue on one
serialization chain (a refused request returns at once without waiting for its turn), and `dispose`
waits for the calls in flight, disposes the Sessions, then destroys the GPU only when the pipeline
acquired it. Every Session runs the attention scores in row blocks of at most 128 MiB (`attentionRowBlockMaxBytes`, a fixed value of the family — the
measurement and the reasons are at the constant in `pipeline.ts`): the output is bit-identical to
the unbudgeted run; at 8,192 tokens the text plan backing is 944 MiB instead of 1,648 MiB on the
RTX 3080 Ti, for 2–7% of the run's time, and the vision tower's at 280 soft tokens 135 MiB instead
of 329 MiB (the audio tower's scores, `[8n, 1, 12]`, never reach the budget).
`embed`'s `signal` is checked after the turn and before the first run, then between the runs of
one call; construction's `signal` reaches the fetch layer. `onRunDiagnostics` reports the `model`
runs only (the towers' are not reported, as the Gemma family's image embedder's are not). There is
no batch API and no similarity helper; the images and the audios of one call run one at a time.

### The towers are chosen at construction

`PipelineOptions.vision` follows the Gemma family's option for its image embedder: `true` fetches the
distribution's `vision` weights as well (the hub selection is narrowed to the roles the pipeline
runs, so without it not a byte of the tower is fetched) and opens their Session; `fromAssets` takes
the component `vision` among the assets then. Without it an image part is refused with a
`ModelInputError` that says to build the pipeline with `{ vision: true }`, and with it a text call
runs no tower and binds what it bound without it (its vector keeps the text pipeline's sha256 row).
`PipelineOptions.audio` does the same for the `audio` weights and audio parts (`{ audio: true }`);
the two may be on together, and one input may hold images and audios.

## Accepted requests

Every value below comes from the manifest's `pipelineConfig` (`config.ts` — four required fields and
two optional ones, unknown and missing ones fail): `prompts` (name → prefix, upstream's
`config_sentence_transformers.json`), `dimensions` (descending, the first is the full width and the
default), `maxTokens` (`<bos>` and `<eos>` included), `placeholderTokens` (modality → id), for a
distribution with the vision tower `vision` (the Gemma 4 family's shape and reader:
`softTokenBudget` 280, `imageTokenId` 258880 — it must be `placeholderTokens.image` —,
`beginImageTokenId` 255999, `endImageTokenId` 258882, `hiddenSize` 512) and, for one with the audio
tower, `audio` (`audioTokenId` 258881 — it must be `placeholderTokens.audio` —,
`beginAudioTokenId` 256000, `endAudioTokenId` 258883, `hiddenSize` 512, `maxSoftTokens` 750 — the
tower graph's bound, at most the soft tokens of 30 s — and `sampleRate` 16000, the only rate the
front end is built for). The front end's own settings are the host's constants
(`AUDIO_FEATURE_EXTRACTOR` in `audio.ts`), which the dist recipe holds to the snapshot's
`preprocessor_config.json`. The budget is the manifest's, upstream's default — not a caller option.
`embed` refuses with a `ModelInputError`, before the GPU is touched:

- an input that is neither a string nor an array, an empty array, a part that is not an object, a
  part of a type the pipeline does not accept (named with the accepted types — `"text"`, `"image"`
  with the vision tower and `"audio"` with the audio tower; an image or audio part on a pipeline
  without its tower says to build it with `{ vision: true }` / `{ audio: true }`, or that the
  distribution ships no such tower), a text part whose text is not a string, an image part whose
  image is not decoded RGB8 pixels of its size (`Rgb8Image`: `data` a `Uint8Array` of
  `width × height × 3` bytes — alpha is not taken, and decoding is the caller's);
- an audio part whose audio is not mono f32 samples (`MonoAudio`: `data` a `Float32Array` over an
  `ArrayBuffer`, `sampleRate` a number), at another rate than 16,000 Hz (nothing is resampled —
  resampling is the caller's), of more than 480,000 samples (30 s — upstream's feature extractor
  truncates a longer audio silently; here nothing is truncated), of at most 160 samples (no mel
  frame), of more soft tokens than `maxSoftTokens`, or holding a non-finite sample;
- a prompt name the manifest does not declare (the message lists the declared names);
- a width the manifest does not declare (the message lists them);
- a text whose tokens include a placeholder id — the image, audio or video placeholder of
  EmbeddingGemma 2 (258880 / 258881 / 258884), named in the message with its place in the whole
  sequence. The text tower alone has no use for them: upstream's composite model replaces such an id
  with the pad id before the lookup, and sentence-transformers refuses the text. Placeholders are
  never written in text: an image is an image part and an audio an audio part, and the ids of
  their blocks are the host's; video is not taken yet (a later video input is an array of frames);
- a token sequence longer than `maxTokens` (8,192 for EmbeddingGemma 2, upstream's context —
  docs/limitations.md), every image and every audio counting its soft tokens and its two markers;
  nothing is truncated;
- a token sequence longer than this device runs: the graph's largest intermediate grows with the
  length (EmbeddingGemma 2's per-layer-embedding projection, 49,152 B per token — 384 MiB at 8,192)
  and must fit one storage buffer, so the device's limit is the smaller of its
  `maxStorageBufferBindingSize` and `maxBufferSize` over those bytes (2,730 tokens at the WebGPU
  default of 128 MiB — the `i8` preset needs no binding or buffer size above it —, 5,461 on a device that binds 256 MiB — the `bf16`
  preset's `requiredLimits` —, 43,690 at 2 GiB). The message names the count, the device's limit and the
  buffer; a device whose limit is at or above `maxTokens` never refuses this way.

The empty string is accepted (`<bos>` `<eos>` alone, the shortest sequence the graph runs). Text
holding other special spellings (`<bos>`, the turn and tool markup, the media begin / end markers)
is tokenized as upstream's tokenizer does and is not refused.

`tokenize(input, { prompt })` returns the token ids `embed` would run for the same input and prompt
(`Int32Array` — `<bos>`, the prefix and the parts, each image's and each audio's soft rows as its
placeholder id between its markers, `<eos>`; its length is the count the limits apply to),
synchronously, on the host (an image read for its size only, an audio for its rate, length and
samples) and without waiting for the calls in flight. It refuses what `embed` refuses before tokenizing (the
input, the parts, the prompt) with the same `ModelInputError`, and fails after `dispose`. A sequence
over `maxTokens` and a text holding a placeholder id are returned, not refused, so a caller can show
the count first; `embed` still refuses both.

## Construction checks

`admitEmbeddingGemma` runs before a weight part is fetched: the pipeline name and major, the
`pipelineConfig` schema, the quant (its `session` must declare no knob —
`EMBEDDINGGEMMA_SESSION_POLICY`), the tokenizer asset's declaration, a shared GPU's features and
limits, and the graph contract (`embeddingGemmaGraphContract`): one symbolic length T, exactly the
inputs `input_ids` i32 `[1, T]`, `pool_mask` f32 `[1, T]`, `soft_embeds` f32 `[1, T, W]` and
`soft_mask` bool `[1, T, 1]` in that order, one f32 output `[1, D]` with D the first declared width,
one `embedding` lookup of `input_ids` (its table `[V, W]`: its rows bound the token ids, its width is
the soft rows'),
the graph's maximum of T — read from the `sym_prefix_slice` of its folded constants (the RoPE
tables; the sliding window is a band declared on the attention, not a constant) — equal to
`maxTokens`, and every symbolic value holding T on exactly one axis (the largest gives the bytes
per token of the device's limit; a `[T, T]` value fails). With `vision: true` it also checks the
tower (`embeddingGemmaVisionContract` in `vision.ts`): `pipelineConfig.vision` is declared, its
`hiddenSize` is the text graph's soft-row width and its three ids are rows of the text graph's table;
the tower graph has one symbol n, exactly `pixels` f32 `[1, 9n, 768]`, `position_x` / `position_y`
i32 `[1, 9n]` in that order, one f32 output `[1, n, hiddenSize]`, no state, and position tables
(the `embedding` tables its positions gather — 840 rows) that cover the longest side the resize rule
gives at the budget (`3 · budget` patches). With `audio: true` it checks the audio tower
(`embeddingGemmaAudioContract` in `audio.ts`, then `admitAudio`): `pipelineConfig.audio` is
declared, its `hiddenSize` is the soft-row width and its three ids are rows of the table; the tower
graph has one symbol n, exactly `features` f32 `[1, 4n, 128]` and `frame_mask` f32 `[1, 4n]` in
that order, one f32 output `[1, n, hiddenSize]`, no state, and its maximum of n — read from the
`sym_prefix_slice` of its folded window bias — equal to `maxSoftTokens`. Once the asset arrives,
and before the GPU is acquired, the tokenizer must wrap the sequence in `<bos>` … `<eos>`, every
placeholder id must be one of its added tokens (and the two markers of each tower built), and its
largest token id must lie inside the embedding table.

## Numerics

The real-GPU e2e over the local mirror (`packages/e2e/tests/e2e_embeddinggemma2_pipeline_test.ts`)
embeds the ten golden texts of the series with their prompts (up to 8,192 tokens): the token ids equal the golden's
(upstream's `tokenizers` over prefix + body), the vectors equal the torch CPU goldens within the
series e2e's tolerance (atol 1e-6), and each vector is pinned by a sha256 row per environment, as
are two shorter widths, for each preset (`bf16` against the f32 series' goldens, whose values bf16
holds bit for bit; `i8` against the i8 series' goldens, the quantized model's own outputs). On the RTX
3080 Ti each preset's vectors are bit-identical to its series' Session outputs, and each golden text's
`i8` vector keeps a cosine of at least 0.999 to its `bf16` vector at every width (measured 0.99940 to
0.99993 at 768 — the torch-CPU screening's values for the same rounding,
[research 2026-11-22](../../../../docs/research/2026-11-22-embeddinggemma2-quant-screen.md)).

Images: the host preprocessing equals the vision series' io inputs byte for byte on its five images
(`packages/models/tests/embeddinggemma2_image_input_test.ts` — the pixels and both position arrays;
the 1190 × 17 rounding edge, 139 soft tokens, and the 840 × 3 strip whose columns reach position 839
among them), the ids of seven inputs where a text run meets an image (spaces, newlines and tabs at the
boundary, an empty text between images, Japanese, a prompt before an image) equal upstream's in the
same host test, and the pipeline e2e over the mirror
(`packages/e2e/tests/e2e_embeddinggemma2_image_pipeline_test.ts`) runs the five inputs of
upstream's SentenceTransformer fixture (an image alone, a text then an image, an image then a text,
a `query`-prompted text with an image, two images): the token ids equal upstream's exactly, and on
the RTX 3080 Ti the `bf16` vectors are within 5.44e-6 of upstream's (cosine at least
0.999999999136; atol 2e-5 — the text graph fed upstream's own soft tokens is within 9.8e-7, checked
at atol 3e-6, and the rest is the tower's GPU noise), the `i8` vectors at a cosine of 0.99976 to
0.99991 (floor 0.999), each pinned by a sha256 row per preset.

Audio: the host front end multiplies the samples by the window in f32, as upstream's f32 product
does (one correctly rounded multiply — upstream's values exactly), and runs the FFT and the rest in
f64. The frames of the fixture's three waveforms
(`packages/models/tests/embeddinggemma2_audio_input_test.ts`) are within 4.77e-7 of upstream's (one
ulp — upstream's complex64 FFT against the host's f64 one), and their bits equal upstream's
arithmetic with an f64 FFT (the digests of `fixtures/embeddinggemma2/audio-front-end.json`, which
the recipe writes and checks against upstream's extractor); the frame and soft-token counts are
checked on every residue of the frame count mod 4 and at the 30 s maximum. The pipeline e2e (`packages/e2e/tests/e2e_embeddinggemma2_audio_pipeline_test.ts`)
runs the four audio inputs of upstream's fixture (an audio alone, a text then an audio, a
`query`-prompted audio of 102 frames — the frame mask's case —, an image then an audio): the token
ids equal upstream's exactly, and on the RTX 3080 Ti the `bf16` vectors are within 2.44e-7 of
upstream's (atol 7.5e-7 — the GPU's own f32 noise; with the windowed product in f64 they were
2.45e-5 off), the text graph fed upstream's soft tokens within
1.12e-7 (atol 3.5e-7), the `i8` vectors at a cosine of 0.99982 to 0.99991 (floor 0.999), a 30 s
audio bit-identical on a device capped at the preset's limits, each pinned by a sha256 row per
preset.

| Files         | Contents                                                                                                                                                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.ts`   | the pipeline name `embeddinggemma/1` and the `pipelineConfig` schema (`prompts`, `dimensions`, `maxTokens`, `placeholderTokens`, the optional `vision` and `audio`)                                                                                         |
| `input.ts`    | the public input types (`EmbedContentPart`, `EmbedOptions`, `TokenizeOptions`), the tokenizer checks (`embedTextRules`), the sequence of an input (`tokenizeEmbedInput`), a request on the host (`planEmbedRequest`) and the width (`truncateEmbedding`) |
| `vision.ts`   | the image preprocessing (`imagePatchRows`), the tower graph's contract (`embeddingGemmaVisionContract`) and one tower run (`runVisionTower`)                                                                                                         |
| `audio.ts`    | the `MonoAudio` type, the frame and soft-token counts, the log-mel front end (`AudioFrontEnd`), the audio tower graph's contract (`embeddingGemmaAudioContract`) and one tower run (`runAudioTower`)                                                    |
| `pipeline.ts` | the graph contract, the admission, the resident Sessions with their chain (`Embedder`) and the public `Pipeline` (`fromPretrained`, `fromAssets`, `embed`, `tokenize`, `dispose`)                                                                       |

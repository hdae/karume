# EmbeddingGemma (host side)

The `./embeddinggemma` subpath of `@karume/models`: `embeddinggemma.Pipeline` turns a text into a
sentence embedding — a unit vector, so the dot product of two results is their cosine similarity.
One family serves the EmbeddingGemma models; today it serves the text model of EmbeddingGemma 2
(`google/embeddinggemma-2`, Apache-2.0 — the distribution `karume-embeddinggemma-2`, model `text`,
pipeline `embeddinggemma/1`, presets `bf16` — the default, the checkpoint's own weights — and `i8` — the same
weight slots as per-row int8; both compute in f32 and declare no knob, and `quant` picks one). The first EmbeddingGemma joins it later with its own manifest values.
The public surface is [`embeddinggemma.ts`](../../embeddinggemma.ts) (also re-exported from the
barrel as the namespace `embeddinggemma`); everything here is internal.

```ts
import { Pipeline } from "@karume/models/embeddinggemma";
import { nodeDirectory } from "@karume/hub/node";

const pipeline = await Pipeline.fromPretrained(nodeDirectory("models/karume-embeddinggemma-2"), { gpu });
const query = await pipeline.embed("how do I reset my password", { prompt: "query", dimension: 256 });
const count = pipeline.tokenize("how do I reset my password", { prompt: "query" }).length;
await pipeline.dispose();
```

The distribution is not published: the recipe `tools/export-recipes/embeddinggemma2/distribution.py`
assembles a local mirror under `models/`, and there is no `SOURCES` table until the first
publication (docs/release-runbook.md §3). A local distribution is passed to `fromPretrained` as a
source handle (`nodeDirectory` / `denoDirectory`); `fromAssets` takes the manifest and the bytes.

## One call

1. **Host** — the input becomes one text: a string, or an array of content parts whose text parts
   are concatenated in order with nothing between them. The prompt's prefix goes in front, and the
   Gemma tokenizer of `src/gemma/text/` (the distribution's asset, with its `<bos>` … `<eos>`
   post-processor) gives the token ids.
2. **GPU** — the graph `model` runs once on `input_ids` `[1, T]` and `pool_mask` `[1, T]` (all ones —
   the pooling averages every token, the prompt's included, as upstream's `include_prompt`). The graph
   holds the text tower, the masked mean pooling and the L2 normalisation, and returns `[1, D]`.
3. **Host** — a shorter width is the leading slice divided by `max(norm, 1e-12)` (computed in f64,
   stored as f32 — sentence-transformers' `truncate_dim` followed by its normalisation).

The Session is opened at construction and kept until `dispose`; calls queue on one serialization
chain (a refused request returns at once without waiting for its turn), and `dispose` waits for the
calls in flight, disposes the Session, then destroys the GPU only when the pipeline acquired it.
The Session runs the attention scores in row blocks of at most 128 MiB
(`attentionRowBlockMaxBytes`, a fixed value of the family — the measurement and the reasons are at
the constant in `pipeline.ts`): the output is bit-identical to the unbudgeted run, and at 8,192
tokens the plan backing is 944 MiB instead of 1,648 MiB on the RTX 3080 Ti, for 2–7% of the run's
time.
`embed`'s `signal` is checked after the turn and before the run; construction's `signal` reaches the
fetch layer. There is no batch API and no similarity helper.

## Accepted requests

Every value below comes from the manifest's `pipelineConfig` (`config.ts` — four fields, unknown
and missing ones fail): `prompts` (name → prefix, upstream's `config_sentence_transformers.json`),
`dimensions` (descending, the first is the full width and the default), `maxTokens` (`<bos>` and
`<eos>` included) and `placeholderTokens` (modality → id). `embed` refuses with a `ModelInputError`,
before the GPU is touched:

- an input that is neither a string nor an array, an empty array, a part that is not an object, a
  part of a type the model does not accept (named with the accepted types — today `"text"`), a text
  part whose text is not a string;
- a prompt name the manifest does not declare (the message lists the declared names);
- a width the manifest does not declare (the message lists them);
- a text whose tokens include a placeholder id — the image, audio or video placeholder of
  EmbeddingGemma 2 (258880 / 258881 / 258884), named in the message. The text tower alone has no
  use for them: upstream's composite model replaces such an id with the pad id before the lookup,
  and sentence-transformers refuses the text. Placeholders are never written in text; a later model
  that takes images, audio or video takes them as content parts (a video as an array of frames);
- a token sequence longer than `maxTokens` (8,192 for EmbeddingGemma 2, upstream's context —
  docs/limitations.md); nothing is truncated;
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
(`Int32Array` — the prefix, the joined text parts, `<bos>` … `<eos>`), synchronously, on the host
and without waiting for the calls in flight. It refuses what `embed` refuses before tokenizing (the
input, the parts, the prompt) with the same `ModelInputError`, and fails after `dispose`. A sequence
over `maxTokens` and a text holding a placeholder id are returned, not refused, so a caller can show
the count first; `embed` still refuses both.

## Construction checks

`admitEmbeddingGemma` runs before a weight part is fetched: the pipeline name and major, the
`pipelineConfig` schema, the quant (its `session` must declare no knob —
`EMBEDDINGGEMMA_SESSION_POLICY`), the tokenizer asset's declaration, a shared GPU's features and
limits, and the graph contract (`embeddingGemmaGraphContract`): one symbolic length T, exactly the
inputs `input_ids` i32 `[1, T]` and `pool_mask` f32 `[1, T]`, one f32 output `[1, D]` with D the
first declared width, one `embedding` lookup of `input_ids` (its table's rows bound the token ids),
the graph's maximum of T — read from the `sym_prefix_slice` of its folded constants (the RoPE
tables; the sliding window is a band declared on the attention, not a constant) — equal to
`maxTokens`, and every symbolic value holding T on exactly one axis (the largest gives the bytes
per token of the device's limit; a `[T, T]` value fails). Once the asset arrives, and before the GPU is acquired, the tokenizer must wrap the
sequence in `<bos>` … `<eos>`, every placeholder id must be one of its added tokens, and its largest
token id must lie inside the embedding table.

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

| Files         | Contents                                                                                                                                                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.ts`   | the pipeline name `embeddinggemma/1` and the `pipelineConfig` schema (`prompts`, `dimensions`, `maxTokens`, `placeholderTokens`)                                                                                                                          |
| `input.ts`    | the public input types (`EmbedContentPart`, `EmbedOptions`, `TokenizeOptions`), the tokenizer checks (`embedTextRules`), the token ids of an input (`tokenizeEmbedInput`), a request on the host (`planEmbedRequest`) and the width (`truncateEmbedding`) |
| `pipeline.ts` | the graph contract, the admission, the resident Session with its chain (`Embedder`) and the public `Pipeline` (`fromPretrained`, `fromAssets`, `embed`, `tokenize`, `dispose`)                                                                            |

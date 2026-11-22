# EmbeddingGemma (host side)

The `./embeddinggemma` subpath of `@karume/models`: `embeddinggemma.Pipeline` turns a text into a
sentence embedding — a unit vector, so the dot product of two results is their cosine similarity.
One family serves the EmbeddingGemma models; today it serves the text model of EmbeddingGemma 2
(`google/embeddinggemma-2`, Apache-2.0 — the distribution `karume-embeddinggemma-2`, model `text`,
pipeline `embeddinggemma/1`). The first EmbeddingGemma joins it later with its own manifest values.
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
- a token sequence longer than `maxTokens` (2,048 for EmbeddingGemma 2, against upstream's 8,192 —
  docs/limitations.md); nothing is truncated.

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
and the graph's maximum of T — read from the `sym_prefix_slice` of its folded constants — equal to
`maxTokens`. Once the asset arrives, and before the GPU is acquired, the tokenizer must wrap the
sequence in `<bos>` … `<eos>`, every placeholder id must be one of its added tokens, and its largest
token id must lie inside the embedding table.

## Numerics

The real-GPU e2e over the local mirror (`packages/e2e/tests/e2e_embeddinggemma2_pipeline_test.ts`)
embeds the seven golden texts of the series with their prompts: the token ids equal the golden's
(upstream's `tokenizers` over prefix + body), the vectors equal the torch CPU goldens within the
series e2e's tolerance (atol 1e-6), and each vector is pinned by a sha256 row per environment, as
are two shorter widths. On the RTX 3080 Ti the pipeline's vectors over the bf16 distribution are
bit-identical to the f32 series' Session outputs.

| Files         | Contents                                                                                                                                                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.ts`   | the pipeline name `embeddinggemma/1` and the `pipelineConfig` schema (`prompts`, `dimensions`, `maxTokens`, `placeholderTokens`)                                                                                                                          |
| `input.ts`    | the public input types (`EmbedContentPart`, `EmbedOptions`, `TokenizeOptions`), the tokenizer checks (`embedTextRules`), the token ids of an input (`tokenizeEmbedInput`), a request on the host (`planEmbedRequest`) and the width (`truncateEmbedding`) |
| `pipeline.ts` | the graph contract, the admission, the resident Session with its chain (`Embedder`) and the public `Pipeline` (`fromPretrained`, `fromAssets`, `embed`, `tokenize`, `dispose`)                                                                            |

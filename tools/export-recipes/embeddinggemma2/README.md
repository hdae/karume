# EmbeddingGemma 2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)); it writes series and the
distribution `karume-embeddinggemma-2`, which is built as a local mirror only (nothing is
published). Only the **text tower** of `google/embeddinggemma-2` is in scope; the checkpoint's
vision and audio towers are never built.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstring of `export.py`; this file is the entry point only. Every file a
script reads is checked against the pin in `sources.py` first (the snapshot is fetched once into
`inputs/embeddinggemma2/embeddinggemma-2/`; the pin's docstring has the command).

## What `export.py` emits

The text tower, masked mean pooling (prompt tokens, `<bos>` and `<eos>` included) and L2
normalisation as **one graph**, whose output is a single unit vector `[1, 768]`. The inputs are
`input_ids` and `pool_mask` (all ones); the token length `T` is symbolic up to 8,192, the upstream
model's context (the model card's; no configuration file states it). Every layer is bidirectional:
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
outputs/series/embeddinggemma-2-text-bf16/…                     the same with `--dtype bf16` (what the distribution ships)
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
family `embeddinggemma`) with the bf16 series' graph as the weights `model` and the compiled
tokenizer as the asset `tokenizer`, one preset `bf16` (the original weights, the default),
`LICENSE.md` (the Apache-2.0 text) and `NOTICE.md` (the modification notice). Its
`pipelineConfig` carries the model's own values, each read from its source and checked against the
others when the plan is built:

| Field               | Value                                         | Source and check                                                                                      |
| ------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `prompts`           | prompt name → prefix (20 names, 8 prefixes)   | `config_sentence_transformers.json` at the pin, verbatim; no default prompt, cosine similarity       |
| `dimensions`        | `[768, 512, 256, 128]`                        | the upstream model card; the first equals the graph's output width                                    |
| `maxTokens`         | `8192`                                        | the bound the graph's folded RoPE tables carry for `T`, held to the export's `SYM_MAX`               |
| `placeholderTokens` | `{image: 258880, audio: 258881, video: 258884}` | `config.json` at the pin; each a special token of the tokenizer asset                                 |

The plan also checks the bf16 storage (the f32 series is refused), the graph's inputs and output,
the tokenizer asset against the pinned `tokenizer.json` and the graph's vocabulary, and the
container's licence and upstream revision against the pin.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.tokenizer
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export --dtype bf16
uv run python dist.py --pipeline embeddinggemma-2
uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \
    python -m embeddinggemma2.parity
```

`transformers` is pinned to 5.23.0 (a change in the modeling code changes the graph's form, and
EmbeddingGemma 2 first ships in 5.19.0); it is brought in per run with `--with` rather than declared
as a group, and each script refuses any other release. `sentence-transformers` 6.1.0 (the release
the snapshot declares) is needed by `parity.py` alone.

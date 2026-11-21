# EmbeddingGemma 2 export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)); it writes series, not a
distribution, and has no dist recipe yet. Only the **text tower** of `google/embeddinggemma-2` is in
scope; the checkpoint's vision and audio towers are never built.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstring of `export.py`; this file is the entry point only. Every file a
script reads is checked against the pin in `sources.py` first (the snapshot is fetched once into
`inputs/embeddinggemma2/embeddinggemma-2/`; the pin's docstring has the command).

## What `export.py` emits

The text tower, masked mean pooling (prompt tokens, `<bos>` and `<eos>` included) and L2
normalisation as **one graph**, whose output is a single unit vector `[1, 768]`. The inputs are
`input_ids` and `pool_mask` (all ones); the token length `T` is symbolic up to 2,048. Every layer is
bidirectional: the 20 sliding layers see `|q − kv| <= 512` through one folded
`[1, 1, 2048, 2048]` band constant sliced by `sym_prefix_slice`, the 4 full layers carry no mask,
and grouped-query attention stays in its true form (2 KV heads × 256 on the sliding layers, 1 KV
head × 512 on the full ones). Prompt prefixing and tokenization stay on the host.

```
outputs/series/embeddinggemma-2-text/model-NNNNN-of-NNNNN.krm  the container: graph + weights/constants as a part sequence
outputs/series/embeddinggemma-2-text/io.<case>.safetensors      input tensors and expected outputs from torch CPU (f32)
```

Seven golden cases: four short ones with and without a prompt, and three long documents at T 609 and
1,508 (past the first window cut at T = 514) and exactly 2,048 (the whole folded band). Before
anything is published the script checks the IR form, that the band equals upstream's own mask
function, that the band really moves the long outputs and that its wrapper equals the upstream
model's own forward on them, and a cosine sanity order.

Measured on 2026-11-21 (CPU, f32):

| Case          | T     | band vs every position visible (max abs) | wrapper vs upstream forward |
| ------------- | ----- | ---------------------------------------- | --------------------------- |
| document-609  | 609   | 1.09e-2                                  | 0 (output and hidden)       |
| document-1508 | 1,508 | 2.91e-2                                  | 0 (output and hidden)       |
| document-2048 | 2,048 | 3.89e-2                                  | 0 (output and hidden)       |

Moving the window by one position (511 or 513) moves those outputs by 2.5e-4 to 5.6e-4. The series is
1,113,815,088 bytes in 7 parts (f32 weights). The real-GPU check is
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

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.tokenizer
uv run --with 'transformers==5.23.0' python -m embeddinggemma2.export
uv run --with 'transformers==5.23.0' --with 'sentence-transformers==6.1.0' \
    python -m embeddinggemma2.parity
```

`transformers` is pinned to 5.23.0 (a change in the modeling code changes the graph's form, and
EmbeddingGemma 2 first ships in 5.19.0); it is brought in per run with `--with` rather than declared
as a group, and each script refuses any other release. `sentence-transformers` 6.1.0 (the release
the snapshot declares) is needed by `parity.py` alone.

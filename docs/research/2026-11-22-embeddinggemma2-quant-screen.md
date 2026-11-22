> A point-in-time measurement record (2026-11-22): a torch-CPU screening of quantized weight forms for the text tower
> of EmbeddingGemma 2, against the embeddings of the checkpoint's own weights. It is stage A of the model's quant
> presets (step 3 of the EmbeddingGemma 2 item in [backlog](../backlog.md)). Every number below is copied from one run
> of the script named beside it; nothing here is a design source of truth or a decision, and it is not edited to
> follow later events.

# EmbeddingGemma 2 quant screening (torch CPU fake-quant)

## What was measured and how

- **Script**: [`tools/export-recipes/embeddinggemma2/measure_quant.py`](../../tools/export-recipes/embeddinggemma2/measure_quant.py),
  run as `uv run --offline --with 'transformers==5.23.0' python -m embeddinggemma2.measure_quant` from
  `tools/export-recipes/` (torch 2.13.0+cpu, 6 threads, the development machine's Ryzen 7600). The raw record is
  `outputs/bench/embeddinggemma-2/2026-11-22_quant-screen/report.json` (untracked; every per-text cosine at every
  width is in it). One run took 3 min 47 s.
- **Model**: the pinned snapshot of `google/embeddinggemma-2` (revision `f9972ac2…`), loaded and wrapped by the
  export's own code (`EmbeddingWrapper`: the text tower, the band 512 on the sliding layers, masked mean, L2
  normalisation). The **reference** is that wrapper with the checkpoint's bf16 weights upcast to f32 and run in f32
  (the goldens' arithmetic).
- **Weights rounded**: the set the bf16 series stores as bf16 — the token table (`embed_tokens`, 262,144 × 512) and
  the 218 linears (attention, MLP, the per-layer-embedding projections and gates, the final 512 → 768 projection):
  219 tensors, 270,925,824 elements. The rounding is the exporter's own: `karume.quantize.fake_quant_int8`
  (symmetric, one F32 scale per output row; per vocabulary row for the table) and `karume.quantize.fake_quant_int4`
  (symmetric round to nearest, one F32 scale per 32 elements along the in axis). The script checks that each
  configuration rounded exactly its intended set, each tensor once, and that a restore gives the reference back bit
  for bit.
- **Activations (`i8-all+a8`)**: `karume.act_quant.attach_act_quant`, the torch mirror of the runtime's
  `quantize_rows` (per-token symmetric int8 of each linear's input, scale `amax · f32(1/127)`, values closed at
  ±127) — the same kind of CPU simulation as the umT5 a8 investigation
  ([research 2026-10-18](2026-10-18-a8-umt5-sbv2.md) §6). It attached to all 218 linears, and every linear was
  checked to be called as a module (the hooks see only module calls).

| config               | linear weights (218) | token table (1) | linear inputs  |
| -------------------- | -------------------- | --------------- | -------------- |
| `i8-all`             | i8 per row           | i8 per row      | f32            |
| `i8-linear`          | i8 per row           | checkpoint bf16 | f32            |
| `i4-linear+i8-table` | i4 group 32          | i8 per row      | f32            |
| `i4-all`             | i4 group 32          | i4 group 32     | f32            |
| `i8-all+a8`          | i8 per row           | i8 per row      | per-token i8   |

- **Texts (58)**: the 10 golden cases of the export with their prompts (T 12 to 8,192; the four from 2,048 tokens on
  are "long" below), and the 48 texts of the first EmbeddingGemma's calibration corpus
  (`tools/export-recipes/embeddinggemma/calib_texts.py`, T 14 to 45 here) with the prompt kind each carries there,
  prefixed with EmbeddingGemma 2's own `query` / `document` prompts. The corpus holds 20 query → document pairs (a
  query directly followed by its document) and 8 bare texts, 10 of the 48 in Japanese. None of the 8,192-token case
  had to be dropped: it took 17 to 18.5 s per configuration.
- **Metrics**, each at the widths 768, 512, 256 and 128 (the leading slice re-normalised, as the model card's
  Matryoshka widths are used):
  1. cosine of each text's embedding against the reference — min and mean over all 58 texts, the 54 short ones
     (T < 2,048) and the 4 long ones;
  2. retrieval over the 20 pairs (20 queries × 20 documents by cosine): how many queries rank their own document
     first ("own top-1"; 20/20 for the reference at every width), how many put the same document first as the
     reference ("top-1 agree"), and the rank inversion rate — the share of the 20 × 190 = 3,800 (query, document i,
     document j) triples whose order differs from the reference's;
  3. the largest absolute change of any entry of the 58 × 58 text-by-text cosine matrix;
  4. the projected container payload (below).

## Results

### Width 768

| config             | cos min  | cos mean | short min | short mean | long min | long mean | own top-1 | top-1 agree | inversion | max Δ cos matrix |
| ------------------ | -------- | -------- | --------- | ---------- | -------- | --------- | --------- | ----------- | --------- | ---------------- |
| reference          | 1.000000 | 1.000000 | 1.000000  | 1.000000   | 1.000000 | 1.000000  | 20/20     | 20/20       | 0.00%     | 0.0000e+00       |
| i8-all             | 0.999396 | 0.999842 | 0.999396  | 0.999847   | 0.999756 | 0.999767  | 20/20     | 20/20       | 0.74%     | 5.4263e-03       |
| i8-linear          | 0.999641 | 0.999852 | 0.999641  | 0.999858   | 0.999765 | 0.999773  | 20/20     | 20/20       | 0.71%     | 5.3603e-03       |
| i4-linear+i8-table | 0.965902 | 0.980455 | 0.965902  | 0.981215   | 0.968902 | 0.970193  | 20/20     | 20/20       | 8.21%     | 4.1697e-02       |
| i4-all             | 0.959838 | 0.979753 | 0.959838  | 0.980429   | 0.969600 | 0.970636  | 20/20     | 20/20       | 8.45%     | 5.1949e-02       |
| i8-all+a8          | 0.995631 | 0.998880 | 0.995631  | 0.998859   | 0.999115 | 0.999174  | 20/20     | 20/20       | 2.39%     | 1.4877e-02       |

### Width 512

| config             | cos min  | cos mean | short min | short mean | long min | long mean | own top-1 | top-1 agree | inversion | max Δ cos matrix |
| ------------------ | -------- | -------- | --------- | ---------- | -------- | --------- | --------- | ----------- | --------- | ---------------- |
| reference          | 1.000000 | 1.000000 | 1.000000  | 1.000000   | 1.000000 | 1.000000  | 20/20     | 20/20       | 0.00%     | 0.0000e+00       |
| i8-all             | 0.999411 | 0.999848 | 0.999411  | 0.999852   | 0.999775 | 0.999785  | 20/20     | 20/20       | 0.66%     | 5.8315e-03       |
| i8-linear          | 0.999654 | 0.999858 | 0.999654  | 0.999863   | 0.999784 | 0.999791  | 20/20     | 20/20       | 0.63%     | 5.8427e-03       |
| i4-linear+i8-table | 0.967778 | 0.981266 | 0.967778  | 0.981999   | 0.970257 | 0.971370  | 20/20     | 20/20       | 8.11%     | 3.9923e-02       |
| i4-all             | 0.962042 | 0.980624 | 0.962042  | 0.981248   | 0.971223 | 0.972192  | 20/20     | 20/20       | 8.21%     | 5.3965e-02       |
| i8-all+a8          | 0.995779 | 0.998920 | 0.995779  | 0.998899   | 0.999124 | 0.999203  | 20/20     | 20/20       | 2.50%     | 1.6844e-02       |

### Width 256

| config             | cos min  | cos mean | short min | short mean | long min | long mean | own top-1 | top-1 agree | inversion | max Δ cos matrix |
| ------------------ | -------- | -------- | --------- | ---------- | -------- | --------- | --------- | ----------- | --------- | ---------------- |
| reference          | 1.000000 | 1.000000 | 1.000000  | 1.000000   | 1.000000 | 1.000000  | 20/20     | 20/20       | 0.00%     | 0.0000e+00       |
| i8-all             | 0.999474 | 0.999867 | 0.999474  | 0.999871   | 0.999795 | 0.999813  | 20/20     | 20/20       | 0.53%     | 6.0337e-03       |
| i8-linear          | 0.999697 | 0.999876 | 0.999697  | 0.999880   | 0.999805 | 0.999816  | 20/20     | 20/20       | 0.61%     | 5.0962e-03       |
| i4-linear+i8-table | 0.971372 | 0.983476 | 0.971372  | 0.984003   | 0.975575 | 0.976367  | 20/20     | 20/20       | 7.97%     | 4.7096e-02       |
| i4-all             | 0.964045 | 0.982820 | 0.964045  | 0.983237   | 0.976359 | 0.977195  | 20/20     | 20/20       | 7.84%     | 4.6774e-02       |
| i8-all+a8          | 0.996248 | 0.999042 | 0.996248  | 0.999023   | 0.999191 | 0.999303  | 20/20     | 20/20       | 2.05%     | 1.5926e-02       |

### Width 128

| config             | cos min  | cos mean | short min | short mean | long min | long mean | own top-1 | top-1 agree | inversion | max Δ cos matrix |
| ------------------ | -------- | -------- | --------- | ---------- | -------- | --------- | --------- | ----------- | --------- | ---------------- |
| reference          | 1.000000 | 1.000000 | 1.000000  | 1.000000   | 1.000000 | 1.000000  | 20/20     | 20/20       | 0.00%     | 0.0000e+00       |
| i8-all             | 0.999620 | 0.999902 | 0.999620  | 0.999905   | 0.999856 | 0.999872  | 20/20     | 20/20       | 0.66%     | 5.3969e-03       |
| i8-linear          | 0.999796 | 0.999909 | 0.999796  | 0.999911   | 0.999857 | 0.999869  | 20/20     | 20/20       | 0.79%     | 5.4115e-03       |
| i4-linear+i8-table | 0.980793 | 0.988496 | 0.980793  | 0.988852   | 0.983174 | 0.983685  | 20/20     | 20/20       | 8.63%     | 4.9097e-02       |
| i4-all             | 0.977839 | 0.988122 | 0.977839  | 0.988399   | 0.983944 | 0.984379  | 20/20     | 20/20       | 8.47%     | 5.7004e-02       |
| i8-all+a8          | 0.997606 | 0.999309 | 0.997606  | 0.999294   | 0.999451 | 0.999521  | 20/20     | 20/20       | 2.58%     | 1.4322e-02       |

The lowest cosine at 768 is the golden case `bare` (12 tokens, no prompt) for every configuration; the next lowest
(at 768) are, per configuration: `i8-all` a 45-token corpus document, `document-609`, `document-8192`; `i8-linear`
`document-8192`, `document-2048`, `document-1508`; `i4-linear+i8-table` `document-8192`, `document-5935`, a 36-token
corpus document; `i4-all` a 24-token bare corpus text, `document-8192`, `document-5935`; `i8-all+a8` `document-en`
and two corpus documents.

### Projected container payload

Bits per weight group: bf16 `16·N`; i8 `8·N + 32·rows` (one F32 scale per row); i4 `4·N + 32·N/32` (one F32 scale
per group). The f32 remainder is the 194 state-dict tensors outside the weight set (norm weights and per-layer
scalars, 76,824 elements) plus the folded RoPE tables — cos and sin of `[8192, 256]` (sliding) and `[8192, 512]`
(full) = 12,582,912 elements — 12,659,736 elements in all. Linear: 136,708,096 elements over 213,760 rows; table:
134,217,728 elements over 262,144 rows.

| config             | bytes       | MiB   | arithmetic (bits)                                                                                       |
| ------------------ | ----------- | ----- | ------------------------------------------------------------------------------------------------------- |
| bf16 (shipped)     | 592,490,592 | 565.0 | linear 16·136,708,096 + table 16·134,217,728 + remainder 32·12,659,736                                  |
| i8-all             | 323,468,384 | 308.5 | linear 8·136,708,096 + 32·213,760 + table 8·134,217,728 + 32·262,144 + remainder 32·12,659,736          |
| i8-linear          | 456,637,536 | 435.5 | linear 8·136,708,096 + 32·213,760 + table 16·134,217,728 + remainder 32·12,659,736                      |
| i4-linear+i8-table | 271,347,808 | 258.8 | linear 4·136,708,096 + 32·4,272,128 + table 8·134,217,728 + 32·262,144 + remainder 32·12,659,736        |
| i4-all             | 219,967,584 | 209.8 | linear 4·136,708,096 + 32·4,272,128 + table 4·134,217,728 + 32·4,194,304 + remainder 32·12,659,736      |
| i8-all+a8          | 323,468,384 | 308.5 | the weights of `i8-all` (activation quantization stores nothing)                                       |

Check against a measured series: the projection for bf16 (592,490,592 bytes) is 442,135 bytes under the bf16
series' measured size, 592,932,727 bytes (the recipe README, 2026-11-22); the difference is what the formula leaves
out — the graph descriptors, part and block alignment and the small folded constants.

### Wall time per configuration (CPU, apply + all 58 texts)

| config             | wall s | document-2048 | document-3062 | document-5935 | document-8192 |
| ------------------ | ------ | ------------- | ------------- | ------------- | ------------- |
| reference          | 35.9   | 1.92          | 3.43          | 9.63          | 17.5          |
| i8-all             | 36.3   | 1.98          | 3.49          | 9.83          | 17.07         |
| i8-linear          | 36.1   | 1.99          | 3.5           | 9.82          | 17.25         |
| i4-linear+i8-table | 36.4   | 1.99          | 3.54          | 9.81          | 17.06         |
| i4-all             | 36.5   | 1.98          | 3.54          | 9.91          | 17.08         |
| i8-all+a8          | 39.9   | 2.23          | 3.8           | 10.93         | 18.45         |

These are torch CPU times of a fake-quant model (f32 arithmetic throughout); they say nothing about the speed of any
preset.

## What was not measured

- Bytes on the GPU and the shipped containers: the screening rounds weights in torch; no series was exported and no
  container was read. The shipped values (and the runtime's i8 / i4 / a8 kernels on them) still need the GPU check.
- Speed of any preset on any device.
- Any public benchmark (MTEB or similar) or any retrieval set beyond the 20 pairs here.
- Calibrated rounding (GPTQ / AWQ), other 4-bit grids, other group lengths, f16 activations, and a8 on an i4 weight
  set (w4a8).
- The embedding of a text holding a multimodal placeholder (outside the golden domain).

## Observations

- The two i8 weight configurations stay at cosine ≥ 0.99939 against the reference at every width; rounding the token
  table as well (`i8-all` against `i8-linear`) lowers the minimum from 0.99964 to 0.99940 at 768 and leaves the mean,
  the inversion rate (0.71 → 0.74%) and the matrix change (5.4e-3) about where they were.
- Both i4 configurations sit an order of magnitude further out: minimum cosine 0.960 to 0.966 at 768, mean about
  0.980, 8.2 to 8.5% of the ranking triples inverted and a largest matrix change of 4.2e-2 to 5.2e-2 (all at 768;
  the four widths together span 7.8 to 8.6% and 4.0e-2 to 5.7e-2). Rounding the
  table to i4 as well (`i4-all`) moves the minimum from 0.9659 to 0.9598 and the matrix change from 4.2e-2 to 5.2e-2
  at 768.
- Under i4 the long texts' mean cosine (0.970) is lower than the short texts' (0.980 to 0.981) at 768; under i8 the
  two are within 1e-4.
- `i8-all+a8` lands between the i8 and i4 weight configurations: minimum 0.9956, mean 0.9989, 2.4% inverted at 768 —
  against `i8-all`, 1 − mean cosine is 7 times larger (1.12e-3 against 1.58e-4), the inversion rate 3 times (2.39
  against 0.74%) and the matrix change 2.7 times (1.49e-2 against 5.43e-3). Its long texts (≥ 0.99911) stay closer
  than its short ones.
- Narrower Matryoshka widths do not degrade further: at 128 every configuration's minimum cosine is at or above its
  768 value, and the inversion rates stay in the same band.
- Every configuration kept 20/20 own top-1 and 20/20 top-1 agreement at every width: the 20 pairs are 20 distinct
  topics, so this set does not separate the configurations at top-1; the inversion rate and the matrix change are the
  metrics that move here.
- The golden case `bare` (12 tokens, no prompt) is the lowest cosine of every configuration.

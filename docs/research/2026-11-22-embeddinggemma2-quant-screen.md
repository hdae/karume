> A point-in-time measurement record (2026-11-22): a torch-CPU screening of quantized weight forms for the text tower
> of EmbeddingGemma 2, against the embeddings of the checkpoint's own weights. It is stage A of the model's quant
> presets (step 3 of the EmbeddingGemma 2 item in [backlog](../backlog.md)). Every number of the screening is copied
> from one run of the script named beside it; the section on the speed of the shipped `i8` preset, added the same day,
> copies its numbers from the GPU workbench records named there and from the `nvidia-smi` samples kept beside them.
> Nothing here is a design source of truth or a decision, and it is not edited to follow later events.

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

## Speed of the shipped `i8` preset against `bf16` (RTX 3080 Ti, Node + Dawn)

Added the same day after the `i8` preset was built into the local mirror; every number below is copied from the
workbench records named beside it, not from the screening script, except the clock, power, temperature and
utilisation figures, which come from the `nvidia-smi` samples in `outputs/diag/eg2-preset-ab-2026-11-22/` (untracked:
`t<T>.smi.csv` per run, `t8192-swap.smi.csv` for the swapped run, the console output beside them and the sampler's
scripts in `scripts/`).

### What was measured and how

- **Entry**: the workbench's `embeddinggemma-preset-ab` (`apps/workbench/src/entries/local/embeddinggemma-preset-ab.ts`,
  [the spec](../spec/workbench.md#the-embeddinggemma-preset-abs-record)), run as
  `flock /tmp/karume-gpu.lock node apps/workbench/run.ts embeddinggemma-preset-ab --tokens=<T> --rounds=<R>` on the
  development machine (RTX 3080 Ti, Dawn's Vulkan backend under Node, geometry profile `nvidia-ampere`), the mirror
  `models/karume-embeddinggemma-2` with both presets, checkout `cd475469` with the uncommitted `i8` work (`dirty`).
- **Arms**: `embeddinggemma.Pipeline.fromPretrained` of `bf16` (the base) and of `i8`, both on one device acquired
  without features, opened once and resident for the whole A/B. A run is one `embed` of one text (no signal, no
  diagnostics callback; the timed span is the whole call, so it holds the tokenization of the text on the host and
  the readback of the vector besides the GPU work), timed by `runSessionAb` of `@karume/runtime/tune`: one untimed
  warm-up round, then rounds that
  visit the arms A B B A; `judgeSessionAb` without identity (the presets' vectors differ by design), minimum gain 1%.
- **Texts**: a prefix of a golden case of the recipe whose ids (`Pipeline.tokenize` with the case's prompt, `<bos>`
  and `<eos>` included) are exactly T: T = 16 is `query-en` whole, 512 the first 2,258 code points of
  `document-609`, 2,048 `document-2048` (7,895 of 7,898 code points), 8,192 `document-8192` (29,544 of 29,547).
- **One run per length**, in the order 16, 512, 2,048, 8,192, 90 s apart (the GPU back at 32 to 36 °C and idle
  clocks before each). Rounds: 101 at 16 tokens, 61 at 512, 31 at 2,048, 15 at 8,192 — enough that the first
  rounds, during which the GPU leaves its idle clocks, weigh little (at 16 tokens an earlier run of 21 rounds,
  `…T11-52-49.json`, spent its first half second at 525 to 825 MHz — read from a side sampler whose samples were not
  kept).
- **Clocks**: the record cannot read them; `nvidia-smi` was sampled every 100 ms beside each run (SM and memory
  clock, temperature, power, the power-cap and thermal reasons), and the figures below are the samples between the
  first timed run and the closing of the pipelines.
- **Probes**: before the A/B each preset was loaded once more by a pipeline of its own with the diagnostics callback,
  the text embedded twice; load figures and bytes are that pipeline's (the second run's diagnostics). Loads read the
  mirror from the local disk, so `supply wait` holds the disk read (the page cache warm after the first run of the
  day), not a download.

### Speed per length

| T     | Rounds | Record (`outputs/workbench/node-nvidia-geforce-rtx-3080-ti/…`) | bf16 median | i8 median | bf16 fastest | i8 fastest | median Δ (i8 − bf16) | interval of the median Δ   | noise floor | judge        |
| ----- | ------ | --------------------------------------------------------------- | ----------- | --------- | ------------ | ---------- | -------------------- | -------------------------- | ----------- | ------------ |
| 16    | 101    | `embeddinggemma-preset-ab_2026-11-22T12-00-57.json`             | 13.4 ms     | 8.3 ms    | 12.7 ms      | 6.1 ms     | −5.25 ms (−39.25%)   | −5.30 to −5.21 ms (92.7%)  | 0.30 ms     | adopt        |
| 512   | 61     | `embeddinggemma-preset-ab_2026-11-22T12-02-49.json`             | 30.4 ms     | 30.7 ms   | 27.1 ms      | 29.5 ms    | +0.35 ms (+1.15%)    | +0.23 to +0.51 ms (92.8%)  | 0.63 ms     | inconclusive |
| 2,048 | 31     | `embeddinggemma-preset-ab_2026-11-22T12-05-37.json`             | 440.3 ms    | 119.2 ms  | 114.9 ms     | 115.5 ms   | −1.18 ms (−0.27%)    | −1.42 to +0.49 ms (92.9%)  | 1.15 ms     | inconclusive |
| 8,192 | 15     | `embeddinggemma-preset-ab_2026-11-22T12-08-37.json`             | 1,378.5 ms  | 952.9 ms  | 936.5 ms     | 933.7 ms   | −424.25 ms (−30.78%) | −433.64 to −407.78 ms (96.5%) | 835.12 ms | inconclusive |

"Median" is the median over the rounds of the mean of the round's two visits; "fastest" the fastest timed visit;
Δ is the judge's paired difference per round (the percentage is the judge's gain against the bf16 median, with the
sign of Δ); the interval is the order-statistic interval of the median Δ with its coverage. Each arm repeated its own
first vector bit for bit in every run of every record, and the cosine of the `i8` vector against the `bf16` one was
0.9999201 (16), 0.9997485 (512), 0.9997587 (2,048) and 0.9997382 (8,192).

At 16 tokens `i8` is faster. At 512 it is not: both runs of that length read it slower (below). At 2,048 the two
arms run at the same speed outside periodic stretches in which both are slow, and the arms' medians are those
stretches' doing (next section). At 8,192 the `bf16` arm is slower on average, through an alternation that only
`bf16` shows (next section).

### What moved the long rows

- **2,048 tokens: periodic stretches in which an embed takes about 11 times longer, in both arms.** Outside the
  stretches every run took 114.9 to 124.7 ms (median 117.7 ms). A first stretch, from the first `i8` visit of round
  4 to the second of round 7 (14 runs), held runs of 439 to 1,100 ms, rounds 5 and 6 wholly inside it at 439 to
  447 ms. From round 11 on a stretch of eight consecutive runs recurs every 6 rounds (every 12 runs of each arm): it
  starts at the last run of rounds 11, 17, 23 and 29 and covers rounds 12, 18, 24 and 30 whole. Inside it the first
  run and every third one after it took 1,528 to 1,901 ms, the others 1,364 to 1,375 ms (11.6 times the median fast
  run), so per stretch the extra-slow runs fall once on `bf16` and twice on `i8` — where the stretch starts in the
  A B B A order decides it. The earlier run of the same length (`…T11-57-57.json`) shows the same pattern: a first
  stretch from the last run of round 3 to the first of round 8 (18 runs, most at 301 to 305 ms), then eight-run
  stretches from the first `i8` run of rounds 12, 18, 24 and 30 (the last cut short by the end of the run), there at
  1,357 to 1,363 ms with the extra-slow runs at 1,539 to 2,152 ms — once on `i8` and twice on `bf16`. During the
  stretches the GPU was at 100% utilisation, 1,770 to 1,965 MHz SM, 9,501 MHz memory and about 326 W: the clock does
  not explain them. Of the 31 rounds of `…T12-05-37.json`, 14 have a paired difference beyond 100 ms (5 negative, 9
  positive) and 17 are within 10 ms: the 15 rounds without a slow run and rounds 5 and 6. The median paired difference
  (−1.18 ms) and its interval rest on those 17 clean rounds; the `bf16` arm's median (440.3 ms) is a round inside a
  stretch.
- **8,192 tokens: every other `bf16` run is about 1.8 times slower; `i8` is not.** With `bf16` as the base
  (`…T12-08-37.json`), its first visit of each round took 1,731 to 1,816 ms from round 3 on (1,152 ms in round 2,
  936 ms in round 1) and its second 942 to 974 ms, while both `i8` visits took 934 to 962 ms. Swapped
  (`--base=i8 --compared=bf16`, `…T12-12-10.json`), `bf16`'s first visit took 1,514 to 1,730 ms in every round and its
  second 938 to 971 ms, `i8` 936 to 964 ms. In run order the `bf16` runs alternate slow and fast in both arm orders
  (from its third timed run with `bf16` as the base, from its first swapped), wherever the arm stands in the round;
  the median slow run is 1.89 times the median fast one with `bf16` as the base and 1.74 times swapped. `i8` shows no
  such alternation, and neither did the probes' two back-to-back runs of `bf16` (1,048 and 996 ms; 1,080 and 981 ms
  swapped). Over the 30 timed runs of each arm the mean embed is 1,326 ms (`bf16`) against 951 ms (`i8`), and 1,307
  against 954 ms swapped: `bf16`'s is 1.39 and 1.37 times `i8`'s. This is a measured, reproducible effect of running
  the `bf16` preset at this length on this machine, not noise; its cause is unknown
  ([known issues](../known-issues.md)). The judge calls both orders inconclusive because the `bf16` arm's own two
  visits differ by 708 to 835 ms per round (its noise floor), more than the paired differences (−424 and +357 ms).
  The fastest visits (936.5 against 933.7 ms) agree, but they leave out the slow half of the `bf16` runs.
- Power was at the card's cap through the long runs: `sw_power_cap` active in 614 of 643 samples (2,048) and 607 of
  670 (8,192), SM 1,635 to 1,965 MHz, memory at 9,501 MHz throughout; `sw_thermal` in 12 samples at 8,192 (up to
  77.8 °C). At 512 tokens the cap was active in 75 of 88 samples at 1,785 to 1,965 MHz; at 16 tokens never (about
  177 W). The arms are interleaved per round, so the cap applies to both alike.
- The cause of either slow pattern was not investigated. The arms run the public pipeline and nothing else, so it
  lies in the runtime, the driver or the device. The 2,048-token stretches reach both arms alike and leave the clean
  rounds' paired differences as the reading; the 8,192-token alternation reaches only `bf16` and is part of what a
  caller of that preset sees on this machine.

### Load, resident weights and plan backing (the probes, 2026-11-22)

| Preset | Load (`fromPretrained`) | supply wait | buffer create | write issue | upload fence | decode | uploaded       | weight arena           | low-precision as stored | first / second embed (16 · 512 · 2,048 · 8,192 tokens)        |
| ------ | ----------------------- | ----------- | ------------- | ----------- | ------------ | ------ | -------------- | ---------------------- | ----------------------- | ------------------------------------------------------------- |
| bf16   | 1,091 to 1,121 ms       | 384–412 ms  | 16.5–17.2 ms  | 147–154 ms  | 15–29 ms     | 0      | 592,560,232 B  | 592,562,268 B (565.1 MiB; 592,563,324 B at 8,192) | 541,851,648 B          | 84.0 / 30.4 · 104.7 / 52.5 · 207.4 / 137.5 · 1,047.6 / 995.9 ms |
| i8     | 814 to 843 ms (1,208 ms once) | 222–243 ms (587 once) | 7.8–8.4 ms | 80.5–81.4 ms | 5.7–9.8 ms | 0 | 323,538,024 B | 323,540,060 B (308.6 MiB; 323,541,116 B at 8,192) | 272,829,440 B          | 50.1 / 24.0 · 81.5 / 52.4 · 167.9 / 136.4 · 1,001.2 / 995.2 ms   |

- The ranges are over the four records of the table; the `i8` load of 1,208 ms (supply wait 587 ms) is the 8,192
  record's, the second load of that run (the base's probe loads first). In the swapped 8,192 record the second probe,
  there `bf16`'s, was slow too (1,521 ms, supply wait 669 ms); why was not looked into. The arms' own loads in the A/B
  (the files read once already) were 1,019 to 1,063 ms (`bf16`) and 816 to 890 ms (`i8`).
- The weight arena is the bytes the weights' buffers hold (the params cache's buffers too: 2,036 B, 3,092 B at 8,192
  tokens); `i8` holds 54.6% of `bf16`'s. Nothing is decoded on the host for either preset.
- Plan backing (the intermediates of one length, the same for both presets): 1.8 MiB at 16 tokens, 59.0 MiB at 512,
  236.1 MiB at 2,048, 944.0 MiB at 8,192; dispatches 853 per run up to 2,048 tokens, 1,357 at 8,192 (the attention
  row blocks).

### What this shows and does not show

- On this machine under Node, `i8` is faster at 16 tokens: 5.2 ms of 13.4 ms.
- At 512 tokens `i8` is not faster. Both runs of that length read it slower: `…T12-02-49.json` by +0.35 ms (1.15%),
  with the whole interval (+0.23 to +0.51 ms) on the slower side and judged inconclusive only because the difference
  is under the noise floor; the earlier `…T11-55-27.json` by +1.24 ms (4.13%), judged reject. The fastest visits of
  `…T12-02-49.json` were 27.1 ms (`bf16`) against 29.5 ms (`i8`); `…T11-55-27.json` predates the fastest-visit field.
- At 2,048 tokens the clean rounds give a paired difference of −1.18 ms (0.27%), inconclusive: the same speed.
- At 8,192 tokens `i8`'s mean embed is shorter, because every other `bf16` run is slow (the mean of `bf16` is 1.37 to
  1.39 times that of `i8`); the fast `bf16` runs and the `i8` runs take the same time.
- Its load is 251 to 298 ms shorter in three of the four probes (less to read and upload); in the 8,192 record the
  `i8` probe, the second load, took 117 ms longer. Its first embed is 23 to 46 ms shorter and its weights 256.6 MiB
  smaller on the GPU.
- Not measured: any other GPU, any browser (Chrome's D3D12 or Metal backends), Deno, a device with the WebGPU default
  limits (the `i8` preset's reason to exist on such devices; the pipeline e2e caps only the binding and buffer sizes
  at the default, so it shows that `i8` needs no buffer or binding size above the default, not that every other
  limit at its default suffices, and it measures no speed), batches of texts or any length other than these four, and
  the cause of the two slow patterns above.

# Gemma E2B browser speed comparison

Run from the repository root (tested with Deno 2.9.6):

```sh
deno task bench:llm-browser
```

Open **http://localhost:8787** in Chrome on your Mac and click **計測開始**. The
current defaults compare **QAT E2B with packed int8 activations disabled and enabled**,
in off → on → on → off order: 4 model loads and 40 generations. Parallel GEMV,
dense prefill buckets, RMS-add fusion, linear-to-SRQ fusion, submission limit 768, and
unfused parallel attention stay fixed. The fusion and packed flags are explicit
benchmark overrides; model defaults remain unchanged. The M2 attention comparison showed no speed benefit, so attention
fusion remains optional.
Transformers.js, reference settings, and earlier comparisons remain available.
**JSONを保存** downloads all timings, generated token IDs, output text,
GPU information, model references, dependency versions, and the benchmark bundle hash.
Keep the tab in the foreground and avoid other GPU workloads during measurement.
Reload the page to stop a run; stop the server with Ctrl+C.

Karume uses your existing converted distributions:

- `models/karume-gemma4` for normal E2B
- `models/karume-gemma4-qat` for QAT E2B

Both paths refer to the distribution directory containing `karume.json`, not its
`e2b/` subdirectory. Override paths or the port when needed:

```sh
deno task bench:llm-browser --port 8787 \
  --normal /path/to/karume-gemma4 --qat /path/to/karume-gemma4-qat
```

Transformers.js downloads pinned public ONNX text models from Hugging Face on first
use (approximately 3.1 GB normal / 2.3 GB QAT, excluding browser/GPU working memory).
It uses the browser cache. Download and model initialization are included in the
separate loading time, never in generation throughput. No Python or Node installation
is needed. Deno bundles the local karume modules into a temporary directory; the
server serves pinned official Transformers.js and ONNX Runtime artifacts from
jsDelivr. The server listens only on the loopback interface and reads model assets
without modifying them.

The ONNX variants require `shader-f16`. The page reports support and rejects an
unsupported GPU rather than switching to CPU. The Linux/NVIDIA benchmark documented
in the research note uses an experimental Chrome flag; it is **not** part of the
Mac launch command and is not necessary for ordinary supported M2 Chrome.

## Linear output fusion

The **Karumeの行列出力** selector compares `fuseLinearStaticQuantize: false` with
`true`. This combines packed parallel GEMV and the following fixed-scale
requantization (SRQ), using the existing weight-scale buffer. It requires
`linearGemvReduce: "parallel"` and f32 compute; unsupported combinations are rejected.
Only validated INT2/4/8 shapes with 1–8 physical rows are fused. The normal E2B graph
has no applicable SRQ nodes. JSON and table labels record the selected flag.
See [the decision record](../../../docs/decisions/0103-linear-static-quantize-fusion.md).

When comparing any earlier optimization, set **Karumeの行列出力** to **従来** to
avoid adding another comparison axis.

## Packed activations

The **Karumeの活性の受け渡し** selector compares `packedStaticQuantize: false` with
`true`. This hands the int8 output of a fixed-scale requantization (SRQ) to parallel
GEMV packed four codes per `u32`, instead of one f32 per element. It is a transport
change only: outputs are bit-identical to the f32 path, with two input values excepted
(`-0.0` collapses into `+0.0`, and `NaN` saturates instead of propagating). It requires
`linearGemvReduce: "parallel"` and f32 compute; unsupported combinations are rejected.
The normal E2B graph has no applicable SRQ nodes. JSON and table labels record the
selected flag.
See [the decision record](../../../docs/decisions/0105-packed-static-quantize-activations.md).

When comparing any earlier optimization, set **Karumeの活性の受け渡し** to
**従来（f32）** to avoid adding another comparison axis.

## State attention fusion

The **Karumeのattention** selector compares `stateAttentionReduce: "parallel"`
with `"parallel-fused"`. The fused path combines normalization and value
aggregation for state attention with at most 8 physical rows and 1024 score
columns. Other shapes and readonly attention retain their existing parallel path.
The JSON records `stateAttentionReduce` for each run. See
[the decision record](../../../docs/decisions/0102-state-attention-stats-pv-fusion.md).

When testing only an earlier optimization below, select **従来の並列経路** for
attention to avoid adding another comparison axis.

## Quant defaults and parallel GEMV

Selecting **quant定義に従う** follows the distribution's E2B `defaultQuant` and its
`session.linearGemvReduce` setting. Newly assembled E2B distributions default to
`i4-fast`, which also declares fusion flags; this page applies its own normalization,
linear SRQ, and packed activation selections instead, so only `linearGemvReduce` is
taken from the quant.
Older local distributions keep their existing `i4` or `i4-gemvpar` default. No files
are rewritten by this benchmark. The table and JSON include the selected quant and
effective reduction mode; JSON also records whether the mode was explicitly overridden.

Select **karume** under 比較対象 and **逐次と並列を比較** under Karumeの行列計算 to run
both overrides in fresh iframes, with the same quant and weights. Set
Karumeの正規化・投入設定 to **従来** to isolate that comparison. Parallel GEMV changes
the summation order for selected packed INT2/INT4/INT8 matrices with f32 arithmetic
and 1–8 input rows. Larger batches and unmeasured shapes keep their existing kernels.
Token sequences can differ, especially for QAT. The M2 results and quality limits are
recorded in [the adoption note](../../../docs/research/2026-09-13-m2-gemv-adoption.md).

## Prefill bucket experiment

The initial selection uses QAT E2B, **karume**, and **細分化**,
following the [M2 validation](../../../docs/research/2026-09-13-m2-prefill-adoption.md).
To repeat only the bucket comparison, change Karumeの入力バケット to
**3種類を往復比較**, Karumeの行列計算 to **並列加算を指定**, and
Karumeの正規化・投入設定 to **従来**.
Explicit parallel selection also works with older local distributions. Save the JSON
after completion; each result includes `prefillBuckets` and `chunkBuckets`.

| Selection           | Physical row buckets below chunk length 64 |
| ------------------- | ------------------------------------------ |
| 標準 (`default`)    | 4, 8, 32                                   |
| 少数追加 (`sparse`) | 4, 8, 16, 32, 48                           |
| 細分化 (`dense`)    | 4, 8, 16, 24, 32, 40, 48, 56               |

The browser initially selects **細分化**; **標準** remains available. Library defaults
are unchanged. A 37-token Japanese input uses 64 physical rows with
standard buckets, 48 with sparse buckets, and 40 with dense buckets. The 31-token
English input uses 32 rows in all three. This primarily targets TTFT, not decode
throughput. It does not change the stored weights, quant selection, or library defaults.

The combined comparison runs `default → sparse → dense → dense → sparse → default`,
with a fresh iframe/model for each job, to expose run-order effects. Each job retains
the initial, warmup, and three measured generations described below. Selecting both
GEMV modes doubles the six Karume jobs per model. Transformers.js runs only once per
model regardless of these Karume settings.

This is a **chunk-64 experiment**, not a recommendation to extend all chunk sizes.
More buckets consume more execution-plan and buffer-cache entries; long inputs and
alternating context capacities can erase the benefit. RTX cache-stress results and
the scope of the experiment are recorded in [the prefill note](../../../docs/research/2026-09-13-prefill-buckets.md).

## Subgroup GEMV comparison

To repeat the comparison, select both E2B models and **並列GEMVの2経路を往復比較**:
`parallel → parallel-subgroup32 → parallel-subgroup32 → parallel`, for eight jobs
and 80 generations. Dense chunk-64 buckets, RMS-add fusion, and submission limit
768 stay fixed. This compares the matrix reduction only; RMS subgroup reduction
remains a separate option.

`linearGemvReduce: "parallel-subgroup32"` preserves the existing parallel kernel's
input partition and addition tree while exchanging partial sums within a fixed
32-lane subgroup. RTX and M2 comparisons preserved all generated tokens. M2 normal
E2B was about 9.3% slower; QAT also showed no speed benefit, with substantial
variation during the run. The browser therefore initially selects existing `parallel`.
Missing `subgroups`, `subgroup-size-control`, or WGSL `subgroup_id`
support is an error; Deno 2.9.6 does not provide the required features.

Both Gemma pipelines accept this option. With an external GPU context, first call
`acquireGpu({ subgroups: true })`. Existing `parallel`, `sequential`, quant defaults,
and model assets retain their meanings. This mode is currently an explicit runtime
option and is not added to the distribution's quant vocabulary. See
[ADR 0101](../../../docs/decisions/0101-linear-gemv-subgroup.md) and
[measurements](../../../docs/research/2026-09-13-gemv-subgroup.md).

## RMS normalization and GPU submission comparison

The optional **RMSの2経路を往復比較** runs `fused → subgroup32 → subgroup32 → fused`
for QAT E2B. Each job uses a fresh iframe with the same weights, parallel GEMV,
dense prefill buckets, RMS-add fusion, and submission limit 768. `subgroup32`
changes the reduction order and can change generated text. It requires Chrome
with `subgroups`, `subgroup-size-control`, and the WGSL `subgroup_id` language
feature; missing support is an error. Deno 2.9.6 cannot run this path.

The previous **従来の3設定を往復比較** remains available as
`reference → submit768 → fused → fused → submit768 → reference` to separate
submission frequency from fusion.

| Setting                                      | RMS-add fusion | Maximum dispatches per submission |
| -------------------------------------------- | -------------- | --------------------------------: |
| 従来 (`reference`)                           | Off            |                              1024 |
| 投入上限のみ768 (`submit768`)                | Off            |                               768 |
| RMSと加算を融合 + 上限768 (`fused`)          | On             |                               768 |
| 32レーン縮約 + 融合 + 上限768 (`subgroup32`) | On             |                               768 |

The fusion is opt-in and keeps the original RMS kernels available. It combines
adjacent RMS normalization and addition for validated f32 widths; the integer
rounding barrier matched the reference on the tested RTX backends and the M2
fusion comparison preserved all 120 generated sequences. The M2 RMS subgroup comparison showed only about 0.5% difference;
its speed benefit is not established on M2. Library, CLI, and quant defaults stay unchanged.
Results include `normalization`, `rmsNormReduce`, `fuseRmsNormAdd`,
`submitMaxChunkSize`, and the enabled GPU and WGSL features.

Both Gemma pipelines also accept `fuseRmsNormAdd` and the runtime `submitPolicy`
as independent options. They also accept `rmsNormReduce: "subgroup32"`; when you
supply a GPU context, acquire it with `acquireGpu({ subgroups: true })` first.
The fusion flag alone does not change the submission limit. See
[ADR 0100](../../../docs/decisions/0100-rms-subgroup-reduction.md),
[the subgroup measurements and quality limits](../../../docs/research/2026-09-13-rms-subgroup-reduction.md), [ADR 0099](../../../docs/decisions/0099-rms-norm-add-fusion.md) and
[measurements](../../../docs/research/2026-09-13-rms-norm-add-fusion.md).

## What is measured

The fixed English and Japanese prompts and token IDs are in [cases.json](cases.json).
Each engine loads its model in a fresh iframe and disposes it before the next job.
For each prompt it runs an initial generation, one additional warmup, then three
measured repetitions, with greedy decoding and at most 64 new tokens. Every run
starts with a fresh conversation/KV cache; there is no reuse of a previous answer.
The table shows the median of the three measured runs. Only the English initial
run is the first generation in that iframe; the Japanese initial run follows it.

- **TTFT:** generation start to delivery of the first non-stop token.
- **Decode tok/s:** `(delivered tokens - 1) / (completion time - first token time)`.
- **Elapsed time:** the full generation call/stream, recorded in JSON.
- Tokenization, detokenization, and UI rendering are outside the measured interval.
  Karume sequence creation is timed. Transformers.js preparation and its internal
  KV disposal before `generate()` resolves are timed; karume's final sequence
  disposal after stream completion is outside the timer, matching the Deno baseline.
- Zero or one delivered token yields no decode rate. Stop tokens are not counted.
- Repetition mismatches are retained and flagged in the output details and JSON.
  Do not treat such runs as a deterministic quality reference.

## Interpreting the comparison

This compares usable model deployments, **not identical weights or numerical
contracts**. Karume uses its converted packed weights and f32 arithmetic. The public
normal ONNX uses q4f16 (including 4-bit embeddings and output projection); QAT ONNX
uses mixed 2/4/8-bit weights and f16 arithmetic. The QAT label alone does not establish
SRQ equivalence. Karume uses capacity 128, chunk length 64, and the default PLE host
cache; ONNX uses its exported attention graph and dynamic KV cache. Generated token
sequences can differ between engines even with identical prompt IDs.

Sources and exact versions are pinned in [config.ts](config.ts). The Transformers.js
browser bundle imports the same ONNX Runtime ESM module for both runtime execution
and tensor construction; its JS and WASM versions are kept together. Only the text
embeddings and decoder sessions load, with no vision/audio encoders. WASM uses one
thread. This is separate from the WebML Community **custom kernel** demo, which is
not a Transformers.js/ONNX Runtime benchmark.

## Offline ONNX and dependencies

For previously downloaded ONNX assets, provide a directory containing `normal/`
and `qat/`, each with its pinned repository's `config.json`, tokenizer files,
`generation_config.json`, and the corresponding `onnx/embed_tokens_*` and
`onnx/decoder_model_merged_*` files (both `.onnx` and `.onnx_data`):

```sh
deno task bench:llm-browser --onnx /path/to/local-onnx
```

Local assets must match [config.ts](config.ts); the server does not convert or update
them, nor verify them against Hugging Face. Archive their hashes alongside research
results when using custom local files. Optional `--vendor /path/to/dependencies`
serves these unmodified official artifacts locally:

- Transformers.js: `transformers.web.js`, saved as `transformers.js`
- ONNX Runtime: `ort.webgpu.min.mjs`, `ort-wasm-simd-threaded.asyncify.mjs`, and
  `ort-wasm-simd-threaded.asyncify.wasm`

Use the versions in `config.ts`; mixing JS/WASM versions is unsupported. Keep the
upstream dependency and model license/notice files with your downloaded assets.

The automated RTX results and browser launch conditions are documented in
[the research note](../../../docs/research/2026-09-12-browser-llm-speed.md).

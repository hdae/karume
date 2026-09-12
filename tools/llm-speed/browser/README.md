# Gemma E2B browser speed comparison

Run from the repository root (tested with Deno 2.9.6):

```sh
deno task bench:llm-browser
```

Open **http://localhost:8787** in Chrome on your Mac. Select Gemma 4 E2B, QAT E2B,
or both, then click **計測開始**. The default compares karume and Transformers.js
sequentially. **JSONを保存** downloads all timings, generated token IDs, output text,
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

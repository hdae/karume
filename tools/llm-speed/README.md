# LLM generation speed baseline

Compare Deno/WebGPU with the official PyTorch/Transformers implementations using the same local
model assets, input token IDs, stop tokens, greedy sampling, and timing definitions. Supports
`gemma4-e2b`, `gemma4-e4b`, `gemma4-qat-e2b`, `gemma4-qat-e4b`, `minicpm5-2b`, and `qwen3-06b`.
Run commands from the repository root. Only the selected model needs to be present locally.

## Prepare an isolated Python environment

Keep the speed environment separate from the exporter's CPU environment. The recorded RTX run used
Python 3.14.6, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, and Transformers 5.14.1:

```sh
speed_dir=$(mktemp -d outputs/bench/karume/llm-speed-$(date +%F)-XXXXXX)
uv venv --python 3.14 "$speed_dir/venv"
uv pip install --python "$speed_dir/venv/bin/python" \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.13.0' 'torchvision==0.28.0'
uv pip install --python "$speed_dir/venv/bin/python" \
  'transformers==5.14.1' accelerate safetensors pytest
```

For Apple Silicon, install the same version pins from the default package index instead of the CUDA
index, then select `--device mps`. The runner implements MPS synchronization, but **this baseline was
only executed on CUDA**. Unsupported device operations fail explicitly; there is no automatic CPU
fallback. CPU is available through `--device cpu`, but its numbers are a separate comparison.

## Freeze inputs for one model

```sh
"$speed_dir/venv/bin/python" tools/llm-speed/prepare.py \
  --model qwen3-06b \
  --checkpoint inputs/qwen3/Qwen3-0.6B \
  --source outputs/series/qwen3-06b-gptq-i4-2026-09-10-probe \
  --out "$speed_dir/inputs"
```

`--checkpoint` points to the official local configuration and tokenizer directory. `--source` points
to the converted series for MiniCPM5/Qwen3, or the distribution root for normal/QAT Gemma. Defaults
are the explicit local profiles in `../llm-baseline/data.py`; override them when your paths differ.
Preparation reads only local files. If the checkpoint lacks its official chat template, provide that
file with `--chat-template /path/to/chat_template.jinja`; the tool does not invent a template.

The fixture contains two fixed English/Japanese chat prompts, token IDs, stop IDs, local paths, and
hashes of the tokenizer, template, and generation configuration. Generation uses at most 64 tokens
(default `--max-new-tokens 64`), with a capacity of 128. Thinking is disabled where the official
chat template supports it. Longer-context performance is outside this initial baseline.

## Run each engine separately

```sh
deno run -A tools/llm-speed/main.ts \
  --inputs "$speed_dir/inputs/inputs.json" --out "$speed_dir/deno"

PYTHONPATH=tools/exporter/src:tools/export-recipes \
  "$speed_dir/venv/bin/python" tools/llm-speed/torch_bench.py \
  --inputs "$speed_dir/inputs/inputs.json" --out "$speed_dir/torch-f32" \
  --device cuda --dtype float32
```

Run only one GPU job at a time, including GPU tests. Every output directory must be new. Failed and
interrupted runs preserve their artifacts; retry in a new directory. The tools never modify model
assets. Normal models can also be measured with `--dtype bfloat16` or `float16`, as a separate
precision condition. Normal E4B's dense float32 weights alone exceed the RTX 3080 Ti's memory;
`capacity.json` records that rejection before transfer, with no hidden offload or dtype change.

Both runners support `--warmups 1 --repeats 3` (defaults). PyTorch additionally accepts
`--attention sdpa|eager`, `--threads 4`, and `--weights stored|source`.

## What the measurements mean

- **TTFT:** request start to delivery of the first non-stop token ID to the host.
- **Decode tok/s:** `(delivered tokens - 1) / (completion time - first token time)`.
  Stop IDs do not count as delivered tokens. Zero/one-token runs have no decode rate.
- **Total time:** includes sequence/cache creation, prefill, generation, and stop handling.
  Model loading, input tokenization, text decoding, terminal output, and outer sequence disposal
  are excluded. Cleanup performed internally before a Deno stream completes is included. PyTorch
  cache references are released after timing. GPU work is synchronized before each run; token
  delivery requires its result on the host.
- Save the first run separately, exclude one additional full warmup, then measure three repetitions.
  Each run starts with a fresh sequence/cache. Only the first case is marked `firstInProcess`;
  the second case has already benefited from the first case's initialization. Driver and filesystem
  caches are not cleared, so this is not a controlled cold-machine benchmark.
- Save every run, generated token IDs, timing, versions, device, and source fingerprints. Repetitions
  within an engine must produce identical tokens and stop IDs. Cross-engine agreement is assessed
  separately: equal stored weights do not guarantee identical GPU reductions or greedy output.

### Quantization and execution differences

`--weights stored` (default) reconstructs the **existing integer payloads and stored scales**, without
recalibration. Normal-model PyTorch runs expand them into dense float32 tensors; BF16/FP16 introduces
additional rounding. Deno retains packed integer weights with float32 computation. These are explicit
execution/storage differences, not native quantized CUDA kernel comparisons.

Mobile QAT uses the official packed mixed INT2/INT4/INT8 modules and their fixed SRQ activation
rounding. The loader checks converted weights/scales/PLE against the original checkpoint. QAT requires
`--weights stored --dtype float32`; changing those conditions fails. Its official forward path unpacks
weights and applies SRQ during execution. This measures that implementation, not a fused CUDA kernel
or the maximum attainable PyTorch throughput.

QAT defaults to `--qat-model text`: the official `Gemma4ForCausalLM` reuses the verified checkpoint's
exact language-model and output-head module objects. This avoids the multimodal wrapper's full-table
dequantization just to read a PAD row. Use `--qat-model conditional` to measure the original
`Gemma4ForConditionalGeneration` separately. The original E4B path ran out of CUDA memory in that
PAD lookup; the text path is validated separately and is not a silent fallback.

Normal Gemma's large per-layer embedding (PLE) table uses CPU row lookup and transfers selected rows
in PyTorch. Deno uses its default host PLE cache budget (two largest shards); QAT PyTorch keeps packed
PLE on the device. These placements are recorded. PyTorch uses dynamic KV cache, SDPA attention by
default, and no `torch.compile`. Float32 matrix multiplication precision is set to `highest`.

`--weights source` measures the unconverted normal checkpoint and requires its original weight
files. It is a different weight condition. Stored normal runs only need the official configuration,
tokenizer, and converted assets; QAT also needs its original packed checkpoint for verification.

## Validation and results

```sh
PYTHONPATH=tools/exporter/src:tools/export-recipes tools/.venv/bin/python -m pytest -q \
  tools/llm-speed tools/llm-baseline
uvx ruff check tools/llm-speed
uvx ruff format --check --line-length 100 tools/llm-speed
deno check tools/llm-speed/main.ts
```

See the [recorded speed comparison](../../docs/research/2026-09-12-llm-speed-baseline.md).
The separate [quality baseline](../llm-baseline/README.md) remains useful for tracking changes in
model outputs; its CPU scoring duration is not used as a generation-speed result.

## Chrome / M2 comparison

Run `deno task bench:llm-browser` from the repository root and open
http://localhost:8787 in Chrome. The [browser benchmark](browser/README.md) compares
local karume Gemma E2B / QAT E2B with pinned Transformers.js ONNX models, separates
TTFT from warmed decode throughput, and exports JSON results.

# MiniCPM5-2B CLI (experimental)

Generate a single response using a locally converted MiniCPM5-2B model in Deno + WebGPU.
The tokenizer also runs in TypeScript; Python is not required to run this CLI.

From the repository root:

```sh
deno task demo:minicpm5 --prompt "日本の首都を都市名だけで答えてください。"
deno task demo:minicpm5 --prompt "WebGPUとは何ですか？日本語で一文で答えてください。"
deno task demo:minicpm5 --quant i8 --prompt "What is the capital of France?"
deno task demo:minicpm5 --completion --prompt "The capital of France is" --max-new-tokens 8
printf '%s\n' '日本の首都を都市名だけで答えてください。' | deno task demo:minicpm5
```

The CLI prefers local GPTQ i4, then i8, RTN i4, f16, and f32. It searches `outputs/series/` for
`minicpm5-2b-<quant>/`, then a matching dated experiment directory. For f32, the dated directory omits
`-f32`. If several dated directories match, select one with `--source`:

```sh
deno task demo:minicpm5 \
  --source outputs/series/minicpm5-2b-gptq-i4-2026-09-10-probe \
  --tokenizer inputs/minicpm5/MiniCPM5-2B/tokenizer.json \
  --prompt "日本の首都を都市名だけで答えてください。" --json
```

`--source` takes an exporter **series directory** containing `model.safetensors` or its numbered shards.
Use either `--source` or `--quant`, not both. The default tokenizer is
`inputs/minicpm5/MiniCPM5-2B/tokenizer.json` from the official local model. Nothing is downloaded or
converted automatically; original Hugging Face weights cannot be used directly as a series.
The selected path is printed to stderr.

Options and behavior match [the Qwen3 CLI](../qwen3/README.md): `--prompt`, `--system`,
`--max-new-tokens` (default 64), `--completion`, `--json`, and `--help`.
Chat uses the official non-thinking template, including MiniCPM's beginning-of-sequence token.
Generation is greedy (temperature 0) and stops at either official EOS token.
Each invocation handles one prompt; history is not retained. Without `--prompt`, stdin is read to EOF.

This example uses a 128-token KV cache and 64 rows per prefill chunk. Input length plus
`max-new-tokens - 1` must fit within 128; oversized requests fail before allocating GPU weights.
This is a short-context experimental example, not the original model's context limit.
The measured GPTQ i4 weights occupy about 1.78 GB on disk; GPU usage also includes cache and workspace.

See [the measurement record](../../docs/research/2026-09-10-codex-mtp-optimization.md).
A public distribution and a published `MiniCPM5Pipeline` API are still pending.

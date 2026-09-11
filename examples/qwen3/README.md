# Qwen3-0.6B CLI (experimental)

Chat with a locally converted Qwen3-0.6B model. Inference and tokenization
run in Deno + WebGPU; Python is only needed to regenerate the tokenizer test fixtures.

From the repository root:

```sh
deno task demo:qwen3
deno task demo:qwen3 --prompt "What is the capital of France? Answer with the city name only."
deno task demo:qwen3 --prompt "日本の首都を都市名だけで答えてください。"
deno task demo:qwen3 --quant i8 --prompt "Explain WebGPU briefly."
deno task demo:qwen3 --completion --prompt "The capital of France is" --max-new-tokens 8
printf '%s\n' 'Explain WebGPU briefly.' | deno task demo:qwen3
```

The CLI prefers local GPTQ i4, then i8, RTN i4, f16, and f32. It searches `outputs/series/` for
`qwen3-06b-<quant>/`, then for the matching dated experiment directory
(e.g. `qwen3-06b-gptq-i4-2026-09-10-probe/`). For f32, the dated directory omits `-f32`.
Multiple dated directories for the same quant require an explicit selection; it never silently
chooses the newest weights. The selected path is printed to stderr.

```sh
deno task demo:qwen3 \
  --source outputs/series/qwen3-06b-gptq-i4-2026-09-10-probe \
  --tokenizer inputs/qwen3/Qwen3-0.6B/tokenizer.json \
  --prompt "日本の首都を都市名だけで答えてください。" --json
```

`--source` takes an exporter **series directory**, containing `model.safetensors` or its numbered
shards. These experimental models do not yet have a public karume distribution. No model is downloaded
or converted automatically, and original Hugging Face weights cannot be passed directly to `--source`.
The default tokenizer is `inputs/qwen3/Qwen3-0.6B/tokenizer.json` from the official local model.
Use `--source` to select another series, or `--quant` to select a quantization; the two options are
mutually exclusive.

| Option                 | Default | Meaning                                                                  |
| ---------------------- | ------- | ------------------------------------------------------------------------ |
| `--prompt <text>`      | omitted | Generate one response and exit; omit for interactive, line-by-line chat. |
| `--system <text>`      | omitted | Optional system message.                                                 |
| `--max-new-tokens <n>` | `64`    | Maximum generated tokens, including EOS.                                 |
| `--completion`         | off     | Continue raw text once; without `--prompt`, read stdin until EOF.        |
| `--json`               | off     | Print one JSON result per response, including input/output token IDs.    |
| `--help`               |         | Show usage without loading assets or requesting a GPU.                   |

Without `--prompt`, the CLI follows `demo:gemma4`: enter one message per line, use `/reset` to clear
history, and `/exit`, `/quit`, or Ctrl+D to leave. Ctrl+C during generation cancels the current turn;
the next message can continue from the displayed partial response. Ctrl+C while idle exits the process.
Piped input also uses one line per message. For a multiline single message, pass `--prompt`.

Generation is greedy (temperature 0), with thinking disabled. Weights stay loaded between turns.
The official Qwen template removes the empty thinking block from past assistant messages, so the CLI
rebuilds the KV cache when the rendered token prefix changes. It only reuses an exactly matching
committed prefix. `/reset` clears both conversation history and its cache.

Interactive text and turn summaries go to stdout; loading and prefill status go to stderr. With
`--json`, stdout contains only one JSON object per response, including `reusedTokens` and `droppedTurns`;
status goes to stderr. `--prompt --json` keeps the single-response JSON format.

This example uses the short-context experimental graphs: 64 rows per prefill chunk and a 128-token
KV cache. The number of input tokens plus `max-new-tokens - 1` must fit within 128. Oversized requests
fail before allocating GPU weights in single-response mode. In chat mode, the CLI removes the oldest
user/assistant pairs until the next request fits and prints a notice; it keeps the system message and
current question. If those alone do not fit, the question is rejected and the conversation remains
available. Reduce the input or token limit. The default generation budget is 64 tokens; use
`--max-new-tokens 32` to leave more room for history. This limit belongs to the example, not to the
original model. Generation quality, especially after quantization, is still experimental.

See [the measurement record](../../docs/research/2026-09-10-codex-mtp-optimization.md) and
[the MiniCPM5 CLI](../minicpm5/README.md). The shared loader and tokenizer currently use repository
internals; this is not a published `Qwen3Pipeline` API.

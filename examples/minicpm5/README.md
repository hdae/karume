# MiniCPM5-2B CLI (experimental)

Chat with a locally converted MiniCPM5-2B model in Deno + WebGPU.
The tokenizer also runs in TypeScript; Python is not required to run this CLI.

From the repository root:

```sh
deno task demo:minicpm5
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
Without `--prompt`, interaction follows `demo:gemma4`: enter one message per line, `/reset` clears the
conversation, and `/exit`, `/quit`, or Ctrl+D exits. Ctrl+C during generation cancels only that turn;
displayed partial text is kept for the next message. Piped input uses the same line-by-line interaction.
`--prompt` generates one response and exits. `--completion` without `--prompt` reads stdin until EOF.
`--json` emits one JSON object per response, with all status on stderr.

Weights stay loaded between turns. After EOS, the CLI reuses the KV cache only when the next official
template's token prefix exactly matches the committed tokens. MiniCPM keeps the empty thinking block
in past assistant messages, so ordinary turns can reuse their cache. A changed prefix or an interrupted
or length-limited answer is rebuilt from the displayed conversation. `/reset` releases the cache.

This example uses a 128-token KV cache and 64 rows per prefill chunk. Input length plus
`max-new-tokens - 1` must fit within 128. Chat removes the oldest complete question/answer pairs with a
notice, keeping the system message and current question. A question that still does not fit is rejected
without losing the conversation. Single-response requests are checked before allocating GPU weights.
Use `--max-new-tokens 32` to leave more space for history.
This is a short-context experimental example, not the original model's context limit.
The measured GPTQ i4 weights occupy about 1.78 GB on disk; GPU usage also includes cache and workspace.

See [the measurement record](../../docs/research/2026-09-10-codex-mtp-optimization.md).
A public distribution and a published `MiniCPM5Pipeline` API are still pending.

## Timing and warmup

Startup runs a short warmup with separate generation state, then discards that state before
accepting conversation input. Warmup does not add messages to the conversation. It prepares the
main prefill/decode paths, not every possible prompt length or speculative execution path.
Use `--no-warmup` to measure a run without this startup step; driver caches may still be warm.

Each reply reports **TTFT** (time to the first non-stop token), **decode tok/s** (subsequent
non-stop tokens divided by the time from the first token to completion), and **total** turn time.
These are wall-clock measurements including host work and output handling, not GPU-only timings.
The displayed total token count includes EOS; the decode numerator excludes EOS and the first
token. With no delivered tokens TTFT is unavailable; with fewer than two, decode speed is unavailable.
Token timing precedes text decoding, so buffered text does not postpone TTFT.
Warmup and model loading are reported separately and excluded from turn timing.

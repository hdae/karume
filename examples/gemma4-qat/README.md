# Gemma 4 mobile QAT chat demo

An experimental interactive CLI for the fixed mobile QAT E2B and E4B models. It shares the
[Gemma 4 CLI](../gemma4/README.md) implementation, including streaming, conversation history,
KV cache reuse, `/reset`, `/exit`, and `Ctrl+C` cancellation.

```sh
deno task demo:gemma4-qat --model e2b --temperature 0
deno task demo:gemma4-qat --model e4b --temperature 0
```

The default source is the existing local distribution at `models/karume-gemma4-qat/`. It is read
directly without copying it to a download cache. `--source <dir>` selects another local distribution;
`--repo <owner/name[@revision]>` explicitly selects a remote distribution. Omitting `--model` uses
the manifest's default. There is no published QAT source pin yet.

Create the local distribution using the [QAT recipe](../../tools/export-recipes/gemma4_qat/README.md).
It must use pipeline `gemma4-qat/1`; ordinary Gemma distributions are rejected. Its `i4` quant label
represents the original fixed mixture of INT2, INT4, and INT8 with static activation rounding (SRQ).
The CLI does not requantize or select a replacement model.

The initial distribution defaults are capacity 128 and prefill chunk length 32. This CLI defaults
to at most 64 new tokens per turn. It drops old conversation pairs when needed, as the Gemma CLI
does. Longer contexts are not validated. `--capacity`, `--chunk-length`, sampling flags, and PLE
memory budget flags have the same meaning as in the Gemma CLI. Use `--help` for the option list.
MTP (`--speculative`), vision, and audio are not supported for this QAT family.

CPU and GPU floating-point reductions can cross an SRQ rounding boundary and select different
tokens. Deno and Chrome agreed on the tested short RTX cases; this does not establish broad model
quality or Apple GPU equivalence. The experimental status is also printed at startup.
`--diagnostics` changes execution timing and currently fails on macOS/Metal; use normal runs there.

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

`--temperature 0` uses the Gemma pipeline's small-output decode path when logits need no repetition
penalty or logit bias and diagnostics are disabled. It reads back the selected value and token id;
prefill and sampling at other temperatures keep their regular paths. Existing model assets work
without conversion.

CPU and GPU floating-point reductions can cross an SRQ rounding boundary and select different
tokens. Deno and Chrome agreed on the tested short RTX cases; this does not establish broad model
quality or Apple GPU equivalence. The experimental status is also printed at startup.
`--diagnostics` changes execution timing and currently fails on macOS/Metal; use normal runs there.

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

## Experimental parallel GEMV

```sh
deno task demo:gemma4-qat --linear-gemv-reduce parallel
```

This opt-in changes the summation order of selected packed INT2/INT4/INT8 matrices
for 1–8 rows. The default is `sequential`. It uses the same local converted weights
and can change generated tokens, especially with QAT. The measured shape set targets
E2B; other shapes keep their existing kernels. Measure on your GPU before choosing it.

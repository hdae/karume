# Gemma 4 mobile QAT text recipe

This experimental recipe converts the official `google/gemma-4-E2B-it-qat-mobile-transformers`
and `google/gemma-4-E4B-it-qat-mobile-transformers` checkpoints. It preserves fixed INT2/INT4/INT8
integers and scales, including static range quantization (SRQ) around quantized linear layers.
It does not requantize the checkpoint. The text decoder, tokenizer, and packed per-layer embeddings
(PLE) are exported; vision, audio, and the MTP drafter are not included.

The upstream weights are Apache 2.0. Distribution assembly includes the license text and a
modification notice. This recipe imports the installed Transformers implementation and does not
copy its source. Use the repository's pinned Transformers dependency (`5.14.1`).

From `tools/export-recipes`:

```sh
uv run python -m gemma4_qat.export --model e2b
uv run python -m gemma4_qat.export --model e4b
uv run python dist.py --pipeline gemma4-qat --model e2b --model e4b \
  --out ../../models/karume-gemma4-qat
```

Place original checkpoints in `inputs/gemma4-qat/<checkpoint-name>/`, or pass `--input`.
Exports go to `outputs/series/gemma4-qat-<model>-product/`, or `--out`. The export verifies every
fixed packed payload and scale after sharding and publishes all files together only after checks
pass. `reference.json` records checkpoint fingerprints and trace bounds. Exporting into an existing
output replaces that complete series; use a new directory to retain previous measurements.

The distribution family is `gemma4-qat/1`, with models `e2b` and `e4b`. Its single `i4` quant is
explicitly labeled as fixed mixed INT2/INT4/INT8 with SRQ. Default capacity is 128 tokens and prefill
chunk length is 32 (trace maximum 128). Larger contexts and broad quality remain unvalidated.
CPU and GPU floating-point reductions can cross SRQ rounding boundaries and select different
tokens. Short Deno and Chrome comparisons on an RTX 3080 Ti agreed with each other, but did not
always match the official CPU output. This is not a claim of CPU/GPU bit identity or complete model
quality validation. See [the research record](../../../docs/research/2026-09-10-codex-mtp-optimization.md).

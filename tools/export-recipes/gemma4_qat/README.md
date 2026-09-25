# Gemma 4 mobile QAT text recipe

This experimental recipe converts the official `google/gemma-4-E2B-it-qat-mobile-transformers`
and `google/gemma-4-E4B-it-qat-mobile-transformers` checkpoints. It preserves fixed INT2/INT4/INT8
integers and scales, including static range quantization (SRQ) around quantized linear layers.
It does not requantize the checkpoint. The text decoder, tokenizer, and packed per-layer embeddings
(PLE) are exported; vision, audio, and the MTP drafter are not included.

The upstream weights are Apache 2.0. Distribution assembly includes the license text and a
modification notice. This recipe imports the installed Transformers implementation and does not
copy its source. Use the repository's pinned Transformers dependency (`5.14.1`), declared as the
`gemma4-qat` dependency group; without that group the export cannot run and the tests that build
upstream modules are skipped.

From `tools/export-recipes`:

```sh
uv sync --group gemma4-qat
uv run python -m gemma4_qat.export --model e2b
uv run python -m gemma4_qat.export --model e4b
uv run python dist.py --pipeline gemma4-qat --model e2b --model e4b \
  --out ../../models/karume-gemma4-qat
```

Place original checkpoints in `inputs/gemma4-qat/<checkpoint-name>/`, or pass `--input`. Exports go
to `outputs/series/gemma4-qat-<model>-product/`, or `--out`. The export verifies every fixed packed
payload and scale after writing the container, reads the PLE assets back from it, and publishes all
files together only after checks pass. `reference.json` records checkpoint fingerprints, trace
bounds, how many fixed weights and PLE blocks were verified, and how many upstream tensors this
text-only conversion never read (KV cache scales, vision, audio). Distribution assembly reconciles
those counts against the series itself. Exporting into an existing output replaces that complete
series; use a new directory to retain previous measurements.

The distribution family is `gemma4-qat/1`, with models `e2b` and `e4b`. The `i4` quant is
fixed mixed INT2/INT4/INT8 with SRQ and reference GEMV summation. E2B defaults to
`i4-fast`, which uses the same weights and declares parallel GEMV, RMS-add fusion,
linear-to-SRQ fusion, and packed int8 activations. `i4-gemvpar` retains parallel GEMV
without the fusions or packed activations.
E4B keeps `i4` as its default. The `karume/5` manifest needs a reader of 0.13.0 or later.
Explicit runtime options override quant settings, including
setting a fusion flag back to `false`. To use sequential GEMV with
`i4-fast`, also set `fuseLinearStaticQuantize: false` and `packedStaticQuantize: false`,
or select `i4`.
Rebuild the distribution to obtain the new declaration; no checkpoint requantization
is needed. Submission policy and prefill buckets remain separate host options.
See [the decision record](../../../docs/decisions/0104-gemma-fast-quant.md).
Default capacity is 4096 tokens and prefill chunk length is 768 (trace maximum 768) — the same
defaults as ordinary Gemma. Both are runtime knobs a caller can override. Output quality beyond
512-token contexts has not been accepted yet.
CPU and GPU floating-point reductions can cross SRQ rounding boundaries and select different
tokens. Short Deno and Chrome comparisons on an RTX 3080 Ti agreed with each other, but did not
always match the official CPU output. This is not a claim of CPU/GPU bit identity or complete model
quality validation. See [the research record](../../../docs/research/2026-09-10-codex-mtp-optimization.md).

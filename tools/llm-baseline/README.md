# PyTorch LLM quality baseline

Small, reproducible reference scores for Gemma 4 E2B and Gemma 4 mobile QAT E2B/E4B — the local
distributions in container form listed in the profile table of `data.py`. This tool runs the
installed **Transformers 5.14.1** model implementations on **CPU with float32 arithmetic and eager
attention**. It is a quality reference, not a speed benchmark. It does not use Karume's GPU executor
to produce its reference scores.

Run from the repository root using the exporter Python environment (`tools/.venv`), with PyTorch,
Transformers 5.14.1, and Accelerate installed. The existing `anima` dependency group supplies these:

```sh
uv sync --project tools/export-recipes --group anima
baseline_dir=$(mktemp -d outputs/bench/karume/2026-09-12_llm-quality-XXXXXX)
uv run --no-project --with 'pyarrow==25.0.1' python tools/llm-baseline/fetch_data.py \
  --out "$baseline_dir/data"
PYTHONPATH=tools/export-recipes tools/.venv/bin/python tools/llm-baseline/data.py \
  --data "$baseline_dir/data" --out "$baseline_dir/suite"
PYTHONPATH=tools/export-recipes tools/.venv/bin/python tools/llm-baseline/run.py \
  --suite "$baseline_dir/suite/suite.json" --model gemma4-e2b --weights stored \
  --out "$baseline_dir/gemma4-e2b-stored"
```

Each output directory must be new. Interrupted or failed runs retain their logs and intermediate
records; rerun into a different directory. Model checkpoints and converted assets are read locally,
without modifying or requantizing them. The local profile paths are explicit in `data.py`.

For `gemma4-e2b`, run both `--weights source` and `--weights stored` into separate directories.
`source` loads the original checkpoint values; `stored` restores the exact integer payloads and
scales from the distribution's containers. There is no new calibration step. Normal Gemma's PLE
table is read by row to avoid allocating a second large floating-point embedding table.

For `gemma4-qat-e2b` and `gemma4-qat-e4b`, use `--weights stored` once: the official checkpoint
already contains the fixed mixed INT2/INT4/INT8 weights. The tool verifies the stored integers,
scales, PLE, and the positive SRQ scales next to each linear operation against it, then uses the
official SRQ activation rounding implementation.
QAT uses a separate checkpoint; a difference from the normal model is not an isolated measurement
of storage quantization. PyTorch reductions still differ from WebGPU; matching quantization is not
a claim of bit identity.
Run one model at a time. E4B needs substantial CPU memory; avoid overlap with GPU verification.

## Evaluation protocol

- **ARC-Easy:** the first 64 test questions whose complete choices fit every profile's tokenizer
  within 128 tokens. Selection depends only on token length, before inference. All models use the
  same question IDs. Score each continuation in `Question: {question}\nAnswer: {choice}` by teacher
  forcing. Report accuracy from summed continuation log probability and separately from mean log
  probability per continuation token. Ties select the earliest choice. No chat template, examples,
  generated reasoning, or answer parsing are used.
- **WikiText-2 raw:** the first 8,192 Unicode characters of test rows joined by two newlines.
  Use 128-token windows with stride 64, resetting positions for each window and scoring each
  token after the first exactly once. Prepend BOS once if the tokenizer declares one. Report
  mean negative log likelihood and its exponential (perplexity).
- Fix dataset revisions, file hashes, tokenizer hashes, input token IDs, model weight hashes,
  dependency versions, CPU thread count, and per-question/per-window scores in the artifacts.
  The default is four CPU threads. `--arc-limit` and `--wiki-limit` are smoke-test controls;
  their results are marked `partial` and must not be reported as the complete baseline.

These are **small regression baselines**, not full ARC leaderboard or standard long-context
WikiText scores. Token-normalized choice accuracy is named explicitly because normalization
conventions differ between evaluation harnesses. Compare perplexity only with the same tokenizer,
text, and window policy. This initial suite does not evaluate Japanese, long conversations, or
free-form generation. Confidence intervals describe sampling uncertainty, not general capability.

## Data and attribution

[ARC](https://huggingface.co/datasets/allenai/ai2_arc) is from the Allen Institute for AI;
its dataset card declares CC BY-SA 4.0.
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext) is from Salesforce Research;
its metadata lists CC BY-SA 3.0 and GFDL, while the card prose links CC BY-SA 4.0. The tool saves
the exact pinned cards with the downloaded data so those declarations remain available.
Dataset text is stored in ignored experiment outputs, not vendored into the project source.

The overlapping-window approach follows the
[Transformers perplexity guide](https://huggingface.co/docs/transformers/en/perplexity).
The implementation counts shifted targets explicitly to avoid double-counting overlap or losing
one target at each window boundary. No external model implementation source is copied.

## Validation and recorded results

```sh
PYTHONPATH=tools/export-recipes tools/.venv/bin/python -m pytest -q tools/llm-baseline
cd tools/export-recipes
uv run ruff check --config pyproject.toml ../llm-baseline
uv run ruff format --check --config pyproject.toml ../llm-baseline
```

The [initial evaluation record](../../docs/research/2026-09-12-llm-quality-baseline.md) documents
its exact inputs, limitations, and artifact paths.

"""The export recipe of Gemma 4 E2B / E4B (causal LM) (ADR 0065 decision 2 — outside the wheel,
repository only).

The export scripts pick the model with `--model e2b|e4b` (default e2b). The MTP drafter and the
tokenizer asset compile exist for E2B only (E4B has no drafter, and its tokenizer is the same
upstream, so it shares E2B's series — {@link gemma4.distribution}).

Only what **cannot sit in the general-purpose exporter core** lives here: five export scripts
(the one-shot {@link gemma4.export}, the states-form decode {@link gemma4.export_decode}, the
token-only exit {@link gemma4.export_token}, the exit with the PLE taken out and the last row's
logits {@link gemma4.export_product}, and the MTP drafter {@link gemma4.export_drafter}), the
tokenizer asset compile
({@link gemma4.tokenizer} — ADR 0084 decision 1; it never touches the graph and handles only
"string ⇄ id sequence"), and the harvesting of the chat format's parity fixtures
({@link gemma4.chat} — ADR 0084 decisions 5 / 7).
There is no separate patch layer — the upstream modeling code is not replaced; adding one
attention implementation that keeps MQA / GQA through transformers' public extension point
(`AttentionInterface.register`) is enough
(it lives inside the script — `gemma4.export.gqa_sdpa_attention`).
It also holds the distribution recipe ({@link gemma4.distribution} / {@link gemma4.card}): the
product container (the PLE is an asset of the container), the drafter container and the tokenizer
asset are folded into one repository, `karume-gemma4`
(ADR 0092 decision 1 — one repository per family).

The scripts and the distribution side share three helpers: the per-layer split of the single PLE
table ({@link gemma4.ple}), the provenance record of a series that takes no golden
({@link gemma4.provenance}), and the formula of the host-generated RoPE
({@link gemma4.rope} — ADR 0091). Each is split out so that one rule is not written in two
places, and the number of readers differs per module (`rope` is read by the distribution side too).
MUST: `rope` does not import torch (to keep the distribution side's imports light — the module
itself states this with its reason; this is a restatement of the MUST, not a check). The same role
on the tokenizer side is
`_shared/gemma_tokenizer.py` (shared with EmbeddingGemma — ADR 0084 decision 6).

12B (`--model 12b`) is another upstream class (`gemma4_unified`: no per-layer embeddings, no
KV sharing, K = V on the full layers) and has its own export script,
{@link gemma4.export_unified} (the product graph and its goldens, in the meta-trace / row-block
form of `qwen3_5.export`) and {@link gemma4.export_vision} (its encoder-free vision embedder and the
goldens with an image). Where every model's checkpoint is read from — and 12B's hub-snapshot pin —
is {@link gemma4.sources} (torch-free, like `rope`).

26B A4B (`--model 26b`) is E2B / E4B's class with a block of routed experts in every layer and has
its own script too, {@link gemma4.export_moe} (12B's product contract and meta-trace form, the
experts through `karume::expert_linear` — {@link gemma4.experts} — and a cut model for this
machine's GPU — ADR 0145).

E2B / E4B's q4_0 QAT line (Google's `-qat-q4_0-unquantized` checkpoints, every 2-D text weight on a
4-bit grid) has its own entry, {@link gemma4.export_product_qat}: the product series' graph with the
grid imported as i4, a schema-4 PLE asset and its own goldens. Its MTP drafter has one too,
{@link gemma4.export_drafter_qat}: the regular drafter's borrower graph from Google's q4_0 QAT
assistants, for E2B and E4B (the one E4B drafter here), every weight imported from its grid as
i4 and the borrowed table the QAT product series' i4 table.

The dependency direction is **recipe → core only**
(`tools/exporter/tests/test_architecture_boundary.py` guards it by machine).

MUST: **do not re-export** (no form in which `from gemma4 import X` drags transformers in).
Callers import the submodule by name.

Scripts are launched from the export-recipes root (e.g.
`uv run --with 'transformers==5.14.1' python -m gemma4.export`).
"""

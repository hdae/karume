# BiRefNet export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstrings (`export.py` / `patch.py` / `distribution.py`); this file is the
entry point only.

## What it emits

One graph per model × resolution — the **matte logits** (`preds[-1]`, before the sigmoid) of the
BiRefNet family, with batch static at 1 and no symbolic axis. The series name is the `--model-dir`
directory name plus `--resolution` (default `birefnet-hr-1024`), because a BiRefNet graph is a
different graph per resolution: window masks and padding constants are baked per resolution.
`--resolution` accepts multiples of 64 only. Two checkpoints share the recipe (BiRefNet_HR and its
fine-tune Lucida — identical structure, different weights and attribution), so they are one
`--model-dir` axis rather than two recipes.

## Running

```sh
cd tools/export-recipes
uv run --group birefnet python -m birefnet.export
uv run --group birefnet python -m birefnet.export --verify     # eager equivalence vs the unpatched model (3 stages)
uv run --group birefnet python -m birefnet.export --resolution 2048
uv run --group birefnet python -m birefnet.export --model-dir ../../inputs/birefnet/lucida
uv run --group birefnet python -m birefnet.export --real-images
uv run python dist.py --pipeline birefnet --model 1024 --model 2048 \
    --out ../../models/karume-birefnet-hr
uv run python dist.py --pipeline lucida --model 1024 --model 2048 \
    --out ../../models/karume-lucida
```

`patch.py` rewrites the upstream model in place before the export; its docstring is the
authoritative list. The last rewrite folds the decoder tail's 1×1 convolution through the bilinear
upsample it used to follow — the two are linear on disjoint axes, so they commute, and the two
full-resolution intermediates the upstream tail materialized (3.22GB and 4.03GB at 2048²) are gone.
`--verify` therefore measures the equivalence against the unpatched eager model in **three stages**:
the layout-only rewrites must be bit-exact, while the module and tail rewrites report a maximum
absolute difference.

**The two axes swap roles between the export script and `dist`.** For the export script the
checkpoint is `--model-dir` and the resolution is `--resolution`. For `dist` the checkpoint is the
**pipeline seat** (`--pipeline birefnet` / `--pipeline lucida`) and the resolution is `--model`:
Lucida is a derivative and ships as its own repository (ADR
[0092](../../../docs/decisions/0092-distribution-repos-and-sources.md) decision 1), and the MIT
copyright lines that go into each repository's `LICENSE.md` differ, so a single pipeline seat
cannot carry both. Within one repository the two resolutions live side by side as two models named
`1024` and `2048` (decisions 1 and 8, the same shape as SigLIP2's base / so400m), with `1024` as
the default because it is the first `--model` on the command line. Assembling two models at once
needs an explicit `--out`: a repository name cannot be derived from a list of model names.

`transformers` is pinned with `==` (the patch layer replaces class attributes of the module the
`trust_remote_code` loader produced); the group is declared in [`../pyproject.toml`](../pyproject.toml).

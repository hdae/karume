# Karume export recipes

**This directory contains upstream-derived code and is not covered by the project's MIT license.**
Each family here carries its own `THIRD_PARTY_NOTICES.md` recording what the recipe derives from,
and the upstream terms recorded there govern that family. A derivative whose redistribution terms
the project cannot vouch for — unknown training data, decensored fine-tunes — does not belong here
at all: build it as a separate library on top of the exporter core, distributed from its own Hugging
Face repository.

Model-specific export recipes for the Karume exporter. Everything here is **repo-only**: it is never
published to PyPI, and the wheel ([`../exporter/`](../exporter/README.md) = PyPI `karume`) contains
none of it. The split is ADR
[0065](../../docs/decisions/0065-exporter-core-recipe-split.md) — the generic `torch.export` → IR v1
core is one distribution; upstream-derived model code and the family knowledge around it (patch
layers, export scripts, reference pipelines, dist recipes, card templates, provenance) is not
shipped with it. Dependencies point **one way, recipe → core**, and that is a machine gate
(`../exporter/tests/test_architecture_boundary.py`), not a convention.

Helpers shared by several families that cannot be promoted into the generic core live in
[`_shared/`](_shared/) — the repository's own path spellings (`models/` / `outputs/` / `inputs/`),
which are repo topology and therefore never core knowledge, plus the gates that several families
must judge identically (the decode-series contract, and whether an i4 series' calibration record is
good enough to ship).

## Setup (uv workspace)

`tools/` is the workspace root; `exporter` and `export-recipes` are its two members and share one
lock and one virtualenv (no second copy of torch for the recipes).

```sh
cd tools/export-recipes
uv sync --all-groups   # MUST be run from here, not from tools/exporter/
```

The virtualenv is shared, so a bare `uv sync` run in [`../exporter/`](../exporter/README.md)
afterwards resolves the exporter's dependencies alone and **prunes the family groups back out of the
environment** (measured: it drops `style-bert-vits2`, `transformers`, `timm`, …). That failure is
silent rather than red — the recipe tests guard every upstream package with `pytest.importorskip` /
`skipif`, so `uv run pytest` still passes green while none of the model cases actually run. Whenever
the environment has been rebuilt, re-run the `--all-groups` command above from this directory.

The family dependency groups (`anima` / `sbv2` / `siglip2` / `siglip2-preprocess` / `birefnet` /
`depth-anything` / `depth-anything-preprocess`) are declared in this directory's
[`pyproject.toml`](pyproject.toml) and are deliberately **not** part of a bare `uv sync` — the base
dependencies alone must keep the core tests and the tiny goldens running. Upstream packages that
would otherwise pin the whole environment are taken temporarily with `uv run --with …` instead of a
group. Inside a group, the package whose modeling code a patch layer replaces is pinned with `==`
(a minor update silently changes the graph shape); packages with nothing to replace stay on `>=`.
The reasoning is written per group in [`pyproject.toml`](pyproject.toml).

Scripts are started as **modules from this directory**, never by path:

```sh
cd tools/export-recipes
uv run --group anima python -m anima.export --dtype f16
uv run --group sbv2 python -m sbv2.export --target front
uv run --with 'transformers==5.14.1' python -m deberta.export --layers 2
```

## dist driver

`karume dist` assembles a distribution directory but holds no family knowledge — the pipeline
registry is injected, and the registry inside the wheel is empty. `dist.py` in this directory is
that injection: it composes the 8 family pipelines with the core engine and passes the
repository's own spellings for `--series` (`outputs/series/`) and `--out` (`models/`).

```sh
cd tools/export-recipes
uv run python dist.py                                   # default = anima
uv run python dist.py --pipeline irodori
uv run python dist.py --pipeline sbv2 --card-profile fn
uv run python dist.py --pipeline sbv2 --card-profile jvnv \
    --model F1 --model F2 --out ../../models/karume-sbv2-jvnv
```

The accepted set is `anima` / `anima-turbo` / `sbv2` / `irodori` / `siglip2` / `birefnet` /
`depth-anything` / `vowel-detector`. What the flags mean — `--model` for assembling several models into one
repository, `--card-profile` for attribution, and the model card written after `verify_dist` — is
the engine's contract and is documented in [`../exporter/README.md`](../exporter/README.md).

## THIRD_PARTY_NOTICES

Every family directory carries a `THIRD_PARTY_NOTICES.md` recording what that recipe is known to
derive from, next to the code it describes rather than in one repository-wide file (provenance is
per family — ADR 0065 decision 2). They are skeletons for the pre-release provenance review (ADR
0065 stage 6) and are **not** license determinations: the upstream-revision license interview is a
release gate, and this reorganization only creates its precondition.

## Families

| Family                                | What it emits                                                                                                                              | README                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `anima` (text-to-image)               | 4 graphs (text_encoder / text_conditioner / transformer / vae_decoder) + host reference fixtures, tokenizer assets                         | [anima/README.md](anima/README.md)                   |
| `sbv2` (text-to-speech)               | 5 graphs (dp / front / flow / dec / voice) + demo assets and the torch reference for the voice demo                                        | [sbv2/README.md](sbv2/README.md)                     |
| `deberta` (text encoder)              | the real-weight DeBERTa-v2 series — shipped as the `text_encoder` seat of the SBV2 distribution, no dist recipe here                       | [deberta/README.md](deberta/README.md)               |
| `irodori` (text-to-speech)            | 6 text-side graphs + the 2 DACVAE codec graphs, tokenizer asset and full-loop reference fixtures                                           | [irodori/README.md](irodori/README.md)               |
| `siglip2` (image feature extraction)  | the vision tower's `pooler_output` as one graph + the preprocessing parity fixture                                                         | [siglip2/README.md](siglip2/README.md)               |
| `birefnet` (image segmentation)       | matte logits as one graph per model × resolution (BiRefNet_HR / Lucida)                                                                    | [birefnet/README.md](birefnet/README.md)             |
| `depth-anything` (depth estimation)   | relative depth as one graph at the pretrained 518² point (Small is the only distributable license)                                         | [depth_anything/README.md](depth_anything/README.md) |
| `embeddinggemma` (sentence embedding) | one graph covering all 5 SentenceTransformer modules — series only, no distribution                                                        | [embeddinggemma/README.md](embeddinggemma/README.md) |
| `vowel-detector` (lip-sync vowels)    | the CRNN as one graph with a symbolic length                                                                                               | [vowel_detector/README.md](vowel_detector/README.md) |
| `minicpm5` (causal LM, 1-shot)        | MiniCPM5-1B as one prefill-shaped graph — the GQA acceptance fixture (ADR 0067), series only                                               | [minicpm5/README.md](minicpm5/README.md)             |
| `gemma4` (causal LM, 1-shot + decode) | Gemma 4 E2B as 3 series (1-shot / states-form decode / token-only exit) — the mixed i8 × i4 fixture, plus the `karume-gemma4` distribution | [gemma4/README.md](gemma4/README.md)                 |

## Patch layers

**The two patch layers have different motivations** (ADR 0016): `sbv2.patch` targets "exportability
itself" (without the branch-free form, `torch.export` fails on a data-dependent guard).
`anima.patch` is a quality layer on top of the fact that **all 4 targets export even unpatched** —
it exists for 3 things: not growing the op set, staying within rank ≤ 4, and not baking runtime
knobs into the graph.

Each family's patch layer is documented in its own README; what they have in common is that they
replace attributes of already-imported upstream classes (monkeypatch) **process-wide**, which is why
every script that both emits and verifies takes one target per process.

## Verification commands (all of them, after any change)

```sh
uv run pytest             # run in tools/export-recipes/
uv run ruff check .
uv run ruff format --check .
```

The core has its own set (run in `tools/exporter/`) — see
[`../exporter/README.md`](../exporter/README.md).

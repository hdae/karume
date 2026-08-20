# Third-party notices — Anima

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/anima/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [circlestone-labs/Anima-Base-v1.0-Diffusers](https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers).
  Loaded from a local checkout; nothing from it is copied into this directory.
- **Model implementation** — the `diffusers` and `transformers` packages (pinned `diffusers==0.39.0`
  / `transformers==5.14.1`). `patch.py` replaces class attributes on already-imported classes and
  never touches the installed packages, but its module docstring self-reports wrappers written out
  verbatim from `AnimaTextConditioner` and `CosmosTransformer3DModel`, and `tiling.py` self-reports
  verbatim ports of `AutoencoderKLQwenImage.blend_v` / `blend_h`. `pipeline_ref.py` self-reports a
  verbatim transcription of the upstream pipeline blocks.
- **Distilled LoRA** — the official CircleStone Labs "Anima Turbo LoRA" (see the inventory row
  below), baked into the weights at export time (`lora.py`). The conversion to diffusers naming
  uses the diffusers-supplied function.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### circlestone-labs/Anima-Base-v1.0-Diffusers (weights)

| Item                     | Value                                                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers>                                                                                                                                     |
| Revision used            | `073c3a9db359c31ad0e8aa268d15775473c2176c` (fetched 2026-08-16 into the HF cache)                                                                                                                       |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                                                                                                                           |
| Code license             | n/a (weights only)                                                                                                                                                                                      |
| Weights license          | `card.py` records `license: other` / `circlestone-labs-non-commercial-license`, pointing at <https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md>. Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                                                                                                                              |

### diffusers / transformers (model implementation)

| Item                     | Value                                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://github.com/huggingface/diffusers> / <https://github.com/huggingface/transformers>                                     |
| Revision used            | `diffusers==0.39.0` / `transformers==5.14.1` (pinned in `pyproject.toml`)                                                      |
| Form of copy             | Monkeypatch of imported classes, plus verbatim-derived wrappers self-reported in `patch.py` / `tiling.py` / `pipeline_ref.py`. |
| Code license             | Unverified                                                                                                                     |
| Weights license          | n/a                                                                                                                            |
| Attribution requirements | Unverified                                                                                                                     |

### Distilled LoRA (baked in at export time)

| Item                     | Value                                                                                                                                                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | Official CircleStone Labs "Anima Turbo LoRA" — <https://civitai.com/models/2560840/anima-turbo-lora> (author `circlestone_labs`; identified 2026-08-20, user-confirmed provenance)                                           |
| Revision used            | v0.2 (`inputs/anima/anima-turbo-lora-v0.2.safetensors`, 148,902,616 B — matches the v0.2 SafeTensor size listed on the Civitai page as of 2026-08-20)                                                                        |
| Form of copy             | Merged into the exported weights (`lora.py`).                                                                                                                                                                                |
| Code license             | n/a (weights only)                                                                                                                                                                                                           |
| Weights license          | "Anima License" per the Civitai page = the CircleStone Non-Commercial License (same terms as the base weights; Copyright CircleStone Labs LLC). Checked on the model page 2026-08-20.                                        |
| Attribution requirements | Covered by the base-model obligations: the distribution bundles `LICENSE.md` + `NOTICE.md` (Attribution Notice, modification statement naming the baked LoRA, non-endorsement) per License §3(a)/(b)/(d) — wired 2026-08-20. |

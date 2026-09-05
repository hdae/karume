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
- **Official single-file checkpoints** — [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima),
  `split_files/diffusion_models/` (fetched by hand into `inputs/anima/upstream-2458426/`). These are
  the four official variants the distribution now carries besides the base: `anima-turbo-v1.1`,
  `anima-turbo-v1.0`, `anima-aesthetic-v1.1`, `anima-aesthetic-v1.0` (ADR 0087).
- **Third-party fine-tunes** — `anima-wai-v1.0` and `anima-copycat-20260610`, published on Civitai
  and redistributed from the separate `karume-anima-extra` repository (ADR 0087 / 0088). Their
  weights derive from the same CircleStone base, so the upstream license flows through; the
  per-source permission fields differ and are listed below.
- **Distilled LoRA** — the official CircleStone Labs "Anima Turbo LoRA" (see the inventory row
  below). It used to be baked into the weights at export time (`lora.py`); **that path is retired
  from the distribution** (ADR 0087 — the official Turbo checkpoint replaced it). The machinery is
  kept for possible future fine-tune intake, and the conversion to diffusers naming uses the
  diffusers-supplied function.

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

| Item                     | Value                                                                                                                                                                                                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/diffusers> / <https://github.com/huggingface/transformers>                                                                                                                                                                                                                                  |
| Revision used            | `diffusers==0.39.0` / `transformers==5.14.1` (pinned in `pyproject.toml`)                                                                                                                                                                                                                                                   |
| Form of copy             | Monkeypatch of imported classes, plus verbatim-derived wrappers self-reported in `patch.py` / `tiling.py` / `pipeline_ref.py`.                                                                                                                                                                                              |
| Code license             | Apache-2.0 for both — read from each installed wheel's own `LICENSE` (`diffusers 0.39.0` / `transformers 5.14.1`, checked 2026-09-05; the `transformers` file carries "Copyright 2018- The Hugging Face team").                                                                                                             |
| Weights license          | n/a                                                                                                                                                                                                                                                                                                                         |
| Attribution requirements | Apache 2.0 §4(a)/(b) attaches to the verbatim-derived wrappers, which live only in this repo-only directory and are attributed here and in the files themselves. Nothing from either package enters the published distribution (import-time dependencies), so the distribution's `LICENSE.md` / `NOTICE.md` are unaffected. |

### circlestone-labs/Anima (official single-file checkpoints)

| Item                     | Value                                                                                                                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/circlestone-labs/Anima> (`card.py` records it as the same bytes Civitai distributes as model version 2458426)                                                                           |
| Revision used            | Unverified — the hand-placed copy under `inputs/anima/upstream-2458426/` carries no snapshot metadata. Files used: `split_files/diffusion_models/anima-{turbo,aesthetic}-v1.{0,1}.safetensors`                  |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form (`karume-anima`).                                                                                                                                  |
| Code license             | n/a (weights only)                                                                                                                                                                                              |
| Weights license          | CircleStone Labs Non-Commercial License v1.2 — the verbatim text is vendored as `circlestone_license.txt` and shipped as the distribution's `LICENSE.md`. The v1.2 wording was compared verbatim on 2026-09-01. |
| Attribution requirements | License §3(a)/(b)/(d): the distribution bundles `LICENSE.md` + `NOTICE.md` (Attribution Notice verbatim, modification statement, non-endorsement). The card also carries them (`card.py` `ATTRIBUTION_NOTICE`). |

### anima-wai-v1.0 (third-party fine-tune)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://civitai.com/models/2544636/wai-anima> (author `WAI0731`; AIR `urn:air:anima:checkpoint:civitai:2544636@2983680`)                                                                                                                                                                                                                                              |
| Revision used            | version 2983680 — `inputs/anima/civitai-2983680/` with `civitai.json`; file `waiANIMA_v10Base10.safetensors`, sha256 `9d5a1e1393c2978d6a979fab38fb0dee00bc2a94e354196c9f3cf2f6f56d5fbf` (byte-matched)                                                                                                                                                                 |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form (`karume-anima-extra`).                                                                                                                                                                                                                                                                                   |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                     |
| Weights license          | Derivative of the CircleStone base, so the CircleStone Non-Commercial License flows through. Civitai permissions as of 2026-08-22 (re-checked 2026-09-01, unchanged): `allowNoCredit` true / `allowCommercialUse` Image, RentCivit / `allowDerivatives` true / `allowDifferentLicense` true. Whether those page fields can broaden the upstream license is Unverified. |
| Attribution requirements | The card lists the source page, author and the four permission fields with an "as of" date (`card.py`). Base-model obligations are covered by the bundled `LICENSE.md` / `NOTICE.md`.                                                                                                                                                                                  |

### anima-copycat-20260610 (third-party fine-tune)

| Item                     | Value                                                                                                                                                                                                                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://civitai.com/models/2377376/copycat-anima> (author `calculater`)                                                                                                                                                                                                                                                  |
| Revision used            | `inputs/anima/copycatAnima_20260610.safetensors`, placed by hand on 2026-08-22 — predates the `anima.civitai` intake command (ADR 0088), so there is no `civitai.json` beside it and no version id or sha256 was recorded. Unverified.                                                                                    |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form (`karume-anima-extra`).                                                                                                                                                                                                                                      |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                        |
| Weights license          | Derivative of the CircleStone base, so the CircleStone Non-Commercial License flows through. Civitai permissions as of 2026-08-22: `allowNoCredit` true / `allowCommercialUse` Image, RentCivit / `allowDerivatives` true / **`allowDifferentLicense` false** — this source requires redistribution under the same terms. |
| Attribution requirements | The card lists the source page, author and the four permission fields with an "as of" date (`card.py`), including the `allowDifferentLicense` false row. Base-model obligations are covered by the bundled `LICENSE.md` / `NOTICE.md`.                                                                                    |

### Distilled LoRA (retired from the distribution path)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | Official CircleStone Labs "Anima Turbo LoRA" — <https://civitai.com/models/2560840/anima-turbo-lora> (author `circlestone_labs`; identified 2026-08-20, user-confirmed provenance)                                                                                                                                                                                                                                |
| Revision used            | v0.2 (`inputs/anima/anima-turbo-lora-v0.2.safetensors`, 148,902,616 B — matches the v0.2 SafeTensor size listed on the Civitai page as of 2026-08-20)                                                                                                                                                                                                                                                             |
| Form of copy             | Was merged into the exported weights (`lora.py`). **No published form derives from it any more** — ADR 0087 replaced the baked Turbo with the official checkpoint; the merge path is kept but unused.                                                                                                                                                                                                             |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                |
| Weights license          | "Anima License" per the Civitai page = the CircleStone Non-Commercial License (same terms as the base weights; Copyright CircleStone Labs LLC). Checked on the model page 2026-08-20.                                                                                                                                                                                                                             |
| Attribution requirements | Applies only if a future distribution bakes a LoRA in; the current `NOTICE.md` does not name one (ADR 0087 retired the baked Turbo, and every model now declares `lora_sha256=None`). Were one baked in again, the base-model obligations would cover it: `LICENSE.md` + `NOTICE.md` with the Attribution Notice, a modification statement naming the baked LoRA, and non-endorsement, per License §3(a)/(b)/(d). |

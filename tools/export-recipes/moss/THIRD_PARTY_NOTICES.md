# Third-party notices — MOSS-Image

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records
what this recipe directory is known to derive from. **It is not a license
determination.**

This directory (`tools/export-recipes/moss/`) is repo-only: it is never
published to PyPI and none of it enters the `karume` wheel (ADR 0065 decision
1). Upstream-derived code lives next to the recipe that needs it so its
provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision
actually used. Checking license compatibility per revision is a human review
scheduled before release (ADR 0065 decision 7); filling this table in _is_ that
review, and this file only lays out the questions it has to answer. Nothing
built from this recipe is published yet (ADR NNNN decision 12).

## Upstream sources

- **Weights and pipeline code** —
  [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT),
  pinned in `sources.py` (`SOURCE`, with the SHA-256 of
  `pipeline_moss_image.py`). The DiT (`transformer`) and the scheduler config
  are read for the references and, later, the export. The
  `pipeline_moss_image.py` file (`trust_remote_code` code) is imported after its
  SHA-256 is checked. The `text_encoder` (Qwen3.5) and `tokenizer` are run
  locally to produce the precomputed text embeddings (`text_embeds.py`).
- **VAE** — bundled in the same repository (`vae/`), but not MOSS's own: it is
  the FLUX.2 VAE with the FLUX.2 small decoder. `sources.py --vae-origin`
  (`VAE_ORIGINS`) compares it tensor by tensor with its two origins — see the
  VAE block below.
- **Model implementation** — `diffusers` (`DiffusionPipeline`,
  `FlowMatchEulerDiscreteScheduler`, `AutoencoderKLFlux2`) and `transformers`
  (`Qwen3_5Model`, `Qwen2Tokenizer`), pinned `diffusers==0.39.0` /
  `transformers==5.14.1`, imported only.
- **Adapted code** — short functions transcribed from upstream code, each under
  a "Third-party code notice" comment: `pipeline_ref.py` (`denormalize`,
  `unpatchify`), `reference.py` (`head`, and `cfg` — the CFG line of
  `MOSSImagePipeline.__call__`) and `dit_patch.py` (`MossDit.forward` and
  `MossDitText.forward` — `MOSSImageTransformer.forward` / `_TextProjection.forward`
  for a batch of one with the host-side steps moved out — and the `RopePair`
  branch of the patched `_rotate`) from `pipeline_moss_image.py`; `reference.py`
  (`euler` — the update line of `FlowMatchEulerDiscreteScheduler.step`) and
  `vae_patch.py` (`upsample_2d` — `Upsample2D.forward` with the nearest ×2 written
  as a repetition — and the three decoder graphs' `forward` — the statements of
  `AutoencoderKLFlux2._decode`, `Decoder.forward`, `UNetMidBlock2D.forward`,
  `UpDecoderBlock2D.forward` and `AttnProcessor2_0.__call__` split across them)
  from diffusers. `cfg` and `euler` are one-line formulas, listed here all the same
  so the inventory does not depend on a judgement of how short a copy may be.
- **Fixed prompts** — `prompts.py` quotes three prompt texts verbatim from the
  HF model card and the upstream user guide; the fourth (the negative) is the
  empty string.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this
recipe is published.

### OpenMOSS-Team/MOSS-Image-SFT (DiT, scheduler, pipeline code)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT> (fine-tuned from `OpenMOSS-Team/MOSS-Image-Base`; code also at <https://github.com/OpenMOSS/MOSS-Image>)                                                                                                                                                                                                                                                                                                                   |
| Revision used            | `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9` (`sources.py` `SOURCE`; `main` on 2026-10-27). `pipeline_moss_image.py` SHA-256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`, the same for its copies under `scheduler/` and `transformer/` (checked before import).                                                                                                                                                                                            |
| Form of copy             | Loaded, not copied. Nothing is distributed yet. The pipeline code is imported, not vendored; four short functions adapted from it (`denormalize`, `unpatchify`, `head`, `cfg`) live in `pipeline_ref.py` / `reference.py`, and the patched forwards of the export graphs (`MossDit`, `MossDitText`, the `RopePair` form of `_rotate`) in `dit_patch.py`, each with a notice. `export_dit.py` writes the DiT weights (i8, local series under `outputs/series/`, not distributed). |
| Code license             | Apache-2.0 — the model card's "License and acknowledgements" ("The model weights and bundled inference code are released under the Apache License, Version 2.0") and the repository's `LICENSE` (the Apache 2.0 text) at the revision used. Checked 2026-10-27; not yet reviewed by a human.                                                                                                                                                                                     |
| Weights license          | `apache-2.0` — the model card front matter (`cardData.license`, HF API, 2026-10-27; not gated). Not yet reviewed by a human.                                                                                                                                                                                                                                                                                                                                                     |
| Attribution requirements | Unverified. Apache 2.0 §4 would ask a distribution for `LICENSE.md` (verbatim `../_shared/licenses/apache_license_2_0.txt`) and a `NOTICE.md` stating the changes; the repository has no NOTICE file.                                                                                                                                                                                                                                                                            |

### VAE — FLUX.2 VAE with the FLUX.2 small decoder (bundled in `vae/` of MOSS-Image-SFT)

| Item                     | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | Decoder and `post_quant_conv` (the part text-to-image runs): <https://huggingface.co/black-forest-labs/FLUX.2-small-decoder>. Encoder, `quant_conv` and the latent BatchNorm statistics (`bn.*` — the decode path de-normalises with them): <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B> (`vae/`). Black Forest Labs.                                                                                                                                                                                                  |
| Revision used            | small decoder `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a` (`diffusion_pytorch_model.safetensors`, SHA-256 `d8d52ba0…d013b`); klein 4B `e7b7dc27f91deacad38e78976d1f2b499d76a294` (`vae/diffusion_pytorch_model.safetensors`, SHA-256 `ca70d220…e0f04`) — `sources.py` `VAE_ORIGINS`.                                                                                                                                                                                                                                                  |
| Form of copy             | MOSS's `vae/diffusion_pytorch_model.safetensors` (SHA-256 `c3bdf94d…f17a`, 251 tensors, f32) was compared on 2026-10-27 (`sources.py --vae-origin`): the 140 decoder / `post_quant_conv` tensors equal the small-decoder file's f32 tensors byte for byte; the 111 encoder / `quant_conv` / `bn` tensors equal the klein 4B VAE's bf16 tensors cast to f32 (exact). The small-decoder file's own encoder and BatchNorm statistics are different values and are not the ones bundled. Loaded, not copied; nothing is distributed yet. |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Weights license          | `apache-2.0` for both — the small decoder's front matter and its card ("Available under the Apache 2.0 license"; License section: Apache 2.0); klein 4B's front matter and License section ("Open weights available for commercial use under the Apache 2.0 license"). HF API `cardData.license`, not gated, 2026-10-27. The same klein VAE file also sits in the non-commercial FLUX.2 [klein] 9B repository; the copy used here is the Apache-2.0 4B one. Not yet reviewed by a human.                                             |
| Attribution requirements | Unverified. Apache 2.0 §4 as above; both cards also carry a trademark note (no use of the Black Forest Labs / FLUX marks that implies endorsement) and point to BFL's usage policy.                                                                                                                                                                                                                                                                                                                                                  |

### Text encoder and tokenizer (`text_encoder/`, `tokenizer/` of MOSS-Image-SFT)

| Item                     | Value                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | The MOSS-Image-SFT repository above. The pipeline code's docstring loads the text encoder from `Qwen/Qwen3.5-9B` (`apache-2.0` per its HF `cardData`, 2026-10-27); whether the bundled weights equal that release is Unverified. |
| Revision used            | as the MOSS-Image-SFT block                                                                                                                                                                                                      |
| Form of copy             | Run locally (bf16, CPU) by `text_embeds.py`; its outputs for the fixed prompts form the text-embedding asset (`outputs/series/moss-image-sft-text-embeds/`). The weights and tokenizer files are not copied.                     |
| Code license             | n/a                                                                                                                                                                                                                              |
| Weights license          | `apache-2.0` (the MOSS-Image-SFT card covers the whole repository). Not yet reviewed by a human.                                                                                                                                 |
| Attribution requirements | Unverified (a distribution of the embeddings would list them in `NOTICE.md`, as Wan's does).                                                                                                                                     |

### diffusers / transformers (imported; one line transcribed)

| Item                     | Value                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/diffusers> / <https://github.com/huggingface/transformers>                                                                          |
| Revision used            | `diffusers==0.39.0` / `transformers==5.14.1` (pinned in `pyproject.toml`, group `moss`)                                                                             |
| Form of copy             | Import-time dependencies. One line is transcribed: `reference.py` `euler` is the update of diffusers 0.39.0 `FlowMatchEulerDiscreteScheduler.step` (with a notice). |
| Code license             | Apache-2.0 for both (as recorded for the Wan recipe, which pins the same versions). Unverified against the repositories' `LICENSE` files.                           |
| Weights license          | n/a                                                                                                                                                                 |
| Attribution requirements | Unverified. The `euler` line is repo-only and not distributed; Apache 2.0 §4 would apply to a distribution of it.                                                   |

### Fixed prompt texts

| Item                     | Value                                                                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | The MOSS-Image-SFT model card (`README.md`, quick-start example) and <https://github.com/OpenMOSS/MOSS-Image> (`USERGUIDE.md`, the two recommended example prompts)                     |
| Revision used            | Commit-pinned URLs in `prompts.py` (HF `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`; GitHub `a211e398a33cbf4dd7a1cc52aba0dfdbac3b0281`, whose `USERGUIDE.md` has SHA-256 `46dc6966…e2a2`) |
| Form of copy             | Three short texts quoted verbatim (one is Chinese, two are English with a Chinese note); they are also stored in the text-embedding asset's metadata.                                   |
| Code license             | Apache-2.0 — the model card (above) and the GitHub repository's `LICENSE` (GitHub API: `Apache-2.0`, 2026-10-27). Not yet reviewed by a human.                                          |
| Weights license          | n/a                                                                                                                                                                                     |
| Attribution requirements | Unverified (Wan's model card links each text to its commit-pinned source; the same would apply here).                                                                                   |

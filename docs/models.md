# Models

The standing of every model family in this repository as the tree builds it now — a description of the current state,
not a work list (work items are in [backlog.md](backlog.md)); for the details of a family, the specs under
[spec/](spec/) and the recipes under [tools/export-recipes/](../tools/export-recipes/README.md) are the source.

A distribution repository is published when its family's `SOURCES` table in `packages/models/src/<family>/config.ts`
pins it to a commit (folded into one table in `packages/models/src/sources.ts`); the pins are written by §3 of the
[release runbook](release-runbook.md). The default preset is the recipe's `default_quant` (`defaultQuant` in the
manifest). The lanes are `FAMILY_LANES` in `packages/test-support/src/lanes.ts`; a lane in `LOW_PRIORITY_FAMILIES`
runs only with `KARUME_ALL_FAMILIES=1`.

## Published families

| Repository | Models | Subpath | Default preset | Lane (runs by default) |
| --- | --- | --- | --- | --- |
| `karume-anima` | `anima-turbo-v1.1` (default), `anima-v1.0`, `anima-aesthetic-v1.1`, `anima-turbo-v1.0`, `anima-aesthetic-v1.0` | `./anima` | `bf16+dit8-a8-attn8-s16` | `models:anima` (yes) |
| `karume-anima-extra` | `anima-wai-v1.0`, `anima-copycat-20260610` (third-party fine-tunes; the text stack is referenced from `karume-anima`) | `./anima` | `bf16+dit8-a8-attn8-s16` | `models:anima` (yes) |
| `karume-anima-2.9b` | `anima-2.9b-preview-v1` (a layer-expanded Anima; the text stack is referenced from `karume-anima`) | `./anima` | `bf16+dit8-a8-attn8-s16` | `models:anima` (yes) |
| `karume-irodori-v4-small` | `v4-small` | `./irodori` | `i8-a8` | `models:irodori` (yes) |
| `karume-irodori-v4.1-small` | `v4.1-small` | `./irodori` | `i8-a8` | `models:irodori` (yes) |
| `karume-irodori-v4.1-small-mf` | `v4.1-small-mf` | `./irodori` | `i8` | `models:irodori` (yes) |
| `karume-irodori-v4-large` | `v4-large` | `./irodori` | `i8-a8` | `models:irodori` (yes) |
| `karume-sbv2-jvnv` | `F1` (default), `F2`, `M1`, `M2` | `./sbv2` | `i8+bert4` | `models:sbv2`, `models:deberta` (yes) |
| `karume-gemma4` | `e2b`, `e4b` | `./gemma` | `bf16` | `models:gemma4` (yes) |
| `karume-gemma4-qat` | `e2b`, `e4b`, `12b`, `26b` (q4_0 QAT) | `./gemma` | `i4` | `models:gemma4` (yes) |
| `karume-gemma4-qat-mobile` | `e2b`, `e4b` | `./gemma4-qat-mobile` | `i4-fast` | `models:gemma4-qat-mobile` (yes) |
| `karume-siglip2` | `base` (default), `so400m` | `./siglip2` | `f32` | `models:siglip2` (**no** — maintained, not run by default) |
| `karume-depth-anything-v2` | `small` | `./depth-anything` | `f32` | `models:depth-anything` (yes) |
| `karume-birefnet-hr` | `1024` (default), `2048` | `./birefnet` | `f16` | `models:birefnet` (yes) |
| `karume-lucida` | `1024` (default), `2048` (a BiRefNet_HR derivative) | `./birefnet` | `f32` | `models:birefnet` (yes) |
| `karume-wan2.2` | `ti2v-5b` (Wan2.2 TI2V 5B; the `text_encoder` is referenced from `karume-umt5-xxl`) | `./wan` | `i8-a8-attn8-online-conv8` | `models:wan-ti2v`, `models:wan` (yes) |
| `karume-umt5-xxl` | `xxl` (the umT5-XXL encoder of Wan2.2 — see the components below) | `./umt5` | `i8` | `models:umt5` (yes) |
| `karume-moss-image` | `sft` (MOSS-Image-SFT, with its Qwen3.5 text tower) | `./moss` | `f32+dit8-a8-attn8-online` | `models:moss`, `models:qwen3_5` (yes) |
| `karume-embeddinggemma-2` | `text` (EmbeddingGemma 2's text tower) | `./embeddinggemma` | `bf16` | `models:embeddinggemma2` (yes) |

## Built but not yet published

None today. A repository the recipe builds and the e2e lanes run on the local mirror, with no `SOURCES` entry yet,
belongs here until its first publication (a checklist in §3 of the [release runbook](release-runbook.md); a new family
follows "First publication of a family").

## Components of another model

A component is exported, tested and shipped as a part of its host model, not as a model of its own. The rule: a
component gets a repository of its own once it is useful alone or two models share the same weights, and the hosts
then reference it there.

- **DeBERTa** (`ku-nlp/deberta-v2-large-japanese-char-wwm`, the BERT of Style-Bert-VITS2) — recipe
  `tools/export-recipes/deberta/`, which has no dist recipe of its own; its series is the `text_encoder` of
  `karume-sbv2-jvnv`. Lane `models:deberta`.
- **Qwen3.5** (MOSS-Image's text encoder) — recipe `tools/export-recipes/qwen3_5/`; the text tower is MOSS-Image-SFT's
  own `text_encoder` folder, so it lives in `karume-moss-image`
  ([ADR 0138](decisions/0138-moss-image-dit-vae.md) addendum of 2026-11-01). Lane `models:qwen3_5`.
- **umT5** (Wan2.2's text encoder) — recipe `tools/export-recipes/wan/umt5_*.py`; it already has its own repository,
  `karume-umt5-xxl` (a repository of its own that other families can reference — [ADR 0119](decisions/0119-wan-umt5-gpu-text-encoder.md)), which
  `karume-wan2.2` references for its `text_encoder`, and its own subpath `./umt5`.

## Removed

Exported only to validate the stack, never published, and no longer built or tested. Each can be restored from git
history (the parent of the commits named) or rebuilt if it is needed again.

- **Wan2.1 T2V 1.3B** — validated the Wan port (DiT, VAE tiles, pipeline) before Wan2.2; removed in `1da4e8d4` (e2e),
  `38b313a0` (pipeline), `491d0a8a` (distribution), `94f542d7` and `e125c265` (export scripts).
- **Style-Bert-VITS2 FN voices** — validated the Style-Bert-VITS2 port; removed in `33b88096` (the e2e checks had
  already moved to JVNV).
- **MiniCPM5-1B** — validated true GQA on real weights; removed in `0d6ff6d4` (e2e and lane) and `c416cb43` (recipe).
- **EmbeddingGemma (`google/embeddinggemma-300m`)** — validated a single-vector text model on real weights; removed in
  `5efa49ed` (e2e and lane) and `96abccf5` (recipe; the parts EmbeddingGemma 2 uses moved into its recipe).
- **vowel-detector** — validated the GRU scan and the derived-dimension binding; removed in `3291df31` (subpath and
  pipeline) and `9aa1c36f` (recipe).

## Adding a model

- Decide its standing first: a published family, a component of another model, or validation only (validation-only
  models leave the tree once the validation is done).
- A third-party fine-tune is read with `fromSingleFile` ([spec/single-file.md](spec/single-file.md)) rather than added
  to a repository; today only Anima's `Pipeline` has `fromSingleFile`.
- A modified model (a changed architecture) gets a repository of its own, as `karume-anima-2.9b` does.

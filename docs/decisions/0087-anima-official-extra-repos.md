# 0087: Split the Anima distribution repositories into "official" and "fine-tuned"

- Status: accepted (2026-09-01 — the maintainer's ruling; the official Turbo checkpoint replaces the
  LoRA-merged karume-anima-turbo); partly superseded by [0092](0092-distribution-repos-and-sources.md)
  (decision 2's pin constant, replaced by `ANIMA_SOURCES`, 2026-09-04)
- Date: 2026-09-01
- Related: [0041](0041-manifest-v2.md) (several models per repository) / [0073](0073-models-source-pin.md)
  (one pin constant per public repository) / [0077](0077-model-version-naming.md) (upstream names) /
  [0038](0038-manifest-v1.md) §7 (cross-repo) / [0016](0016-anima-chain-export.md) (LoRA-merged export)

## Context

Upstream CircleStone republished Anima as **three official variants: Base, Aesthetic and Turbo**
(HF `circlestone-labs/Anima` = civitai 2458426, each one bf16 safetensors file). With Turbo an
official checkpoint, the distribution that merged the turbo LoRA v0.2 into base (ADR 0016,
`hdae/karume-anima-turbo`) lost its reason to exist. The upstream README recommends Turbo first and
positions Base for LoRA training.

The old split was "LoRA merged or not = a different NOTICE (modification notice)". The plain
repository mixed the official base with third-party fine-tunes (wai, copycat) and answered neither
"where do new official variants go" nor "where can fine-tunes be added by Civitai URL / AIR"
(backlog N3).

## Decision

1. **The repositories are split by "official / fine-tuned".**
   - `karume-anima` (official): `anima-turbo-v1.1` (**default** — upstream's recommendation, ruling of
     2026-09-01) / `anima-v1.0` / `anima-aesthetic-v1.1` / `anima-turbo-v1.0` /
     `anima-aesthetic-v1.0` = five variants (2026-09-01 addendum).
   - `karume-anima-extra` (fine-tuned): `anima-wai-v1.0` / `anima-copycat-20260610`, plus fine-tunes
     added later by Civitai URL / AIR (the home for N3).
2. **The LoRA-merged `anima-turbo` model and the `hdae/karume-anima-turbo` repository are retired.**
   The pin constant `ANIMA_TURBO_CURRENT` was folded into `ANIMA_CURRENT` (breaking — unreleased JSR
   surface; the published old repository was left to the release) (superseded by ADR 0092 decision 4
   on 2026-09-04: `*_CURRENT` constants became the table `ANIMA_SOURCES`, keys `anima` and
   `anima-extra` — `packages/models/src/anima/config.ts`).
3. **The text stack (text_encoder / vae_decoder / two tokenizers / a shared text_conditioner — the
   last corrected by the 2026-09-03 addendum) is referenced cross-repo from extra to official** (ADR
   0038 §7), as turbo → base was (publishing order: upload official first, record its main SHA, then
   build extra — release runbook).
4. NOTICE stays one per Pipeline = repository pair (the MUST that the modification notice must be
   correct is unchanged — only the axis moved from "merged or not" to "official / fine-tuned").

## Consequences

- New official variants stay in `karume-anima`, and the default experience (`fromPretrained` with
  the `anima` entry of `ANIMA_SOURCES`) is upstream's recommended Turbo.
- The LoRA-merge path (`anima/lora.py`, `export --lora`) is no longer used for distribution but stays
  (future fine-tunes may need a merged LoRA; `assert_lora_provenance`'s absence check keeps rejecting
  an old fused series inserted by mistake).
- The old series `outputs/series/anima-turbo-*-dyn` are unreferenced (`lora_provenance.json` keeps
  them out of the new presets); cleanup is a separate ruling.
- Extra depends on the official repository's published SHA (re-uploading the target means rebuilding
  extra — the runbook's publishing-order MUST).

## Addendum (2026-09-01) — five official variants

`anima-turbo-v1.0` / `anima-aesthetic-v1.0` join decision 1's official list (the maintainer's
additional ruling that day — preferences differ between versions, so a way back is kept; the
motivation of ADR [0077](0077-model-version-naming.md)). The Context's "three official variants" are
upstream's lines: five models = three lines × versions (v1.0 / v1.1; Base has v1.0 only).

## Addendum (2026-09-03) — the roles referenced cross-repo, and repositories with several models

- **Decision 3's "shared text_conditioner" was wrong.** Both extra models have
  `own_text_conditioner=True` (`tools/export-recipes/anima/distribution.py`). The roles referenced
  cross-repo are **`text_encoder` / `vae_decoder` / `tokenizer` / `tokenizer_2`**.
- **Cross-repo references also apply to repositories with several models.** The check: the files of
  the named roles are **byte-identical to the target in every model**, compared per plan, with all
  plans' references agreeing. The former check "cross-repo only when building one model"
  (2026-08-24, when `karume-anima-turbo` held one model) stopped holding once extra held two; the real
  ambiguity, "one role name pointing at different bytes", is checked directly instead.

## Addendum (2026-11-24) — Anima-2.9B gets a repository of its own

- **Context.** `Gazingstars123/Anima-2.9B` (`anima-2.9b-preview-v1`) is not a fine-tune of the base: its transformer
  was deepened from 28 to 40 blocks of the same width (12 inserted blocks, about 2.9B parameters) and trained further.
  Decision 1's split had two places for it: the extra repository ("fine-tuned", where third-party models go), or a
  repository of its own.
- **Decision (the maintainer's ruling of 2026-11-24).** It goes in **`karume-anima-2.9b`**, a third repository with its
  own Pipeline (`--pipeline anima-2.9b`), notice and card — not in `karume-anima-extra`. The reason: it is a modified
  model, not a fine-tune, and ordinary fine-tunes are now read with `fromSingleFile` (an upstream single file converted
  while a Session reads it), so the extra repository is expected to be deprecated later; a model that will outlive it
  does not go in. Nothing is added to the extra repository, and it is not deprecated by this addendum.
- **What stays as decided.** Decision 3 applies unchanged: its text encoder, VAE decoder and both tokenizers are the
  official base's bytes and are referenced across repositories from `karume-anima` (model `anima-v1.0`), the same roles
  extra borrows; its text conditioner and transformer are its own. Decision 4 applies too: one notice per Pipeline, and
  its notice names the third party's layer expansion instead of a community fine-tune.
- **Alternative not chosen.** Adding it to `karume-anima-extra` would have reused that repository's notice, card and
  Playground entry, and cost a rebuild and upload of the whole extra repository; it would also tie the model to a
  repository expected to be retired, and the extra notice ("community fine-tune") would not describe it.
- **Consequences.** Until its first publication the model has no published pin: its own roles are built from its
  series and the borrowed ones from the official mirror (`anima_unpublished_sources` in
  `tools/export-recipes/anima/distribution.py`), and it has no `SOURCES` entry (the release runbook adds one at the first
  publication). The publication order is the extra repository's: `karume-anima` first, then this one.

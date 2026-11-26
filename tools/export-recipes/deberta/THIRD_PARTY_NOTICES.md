# Third-party notices — DeBERTa-v2 text encoder

Provenance record of this recipe directory and of the `text_encoder` component it builds for the
distribution `karume-sbv2-jvnv`: for each upstream, where it comes from, the revision used, its
license and what was checked. **It is not a license determination.**

This directory (`tools/export-recipes/deberta/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used. This
record is updated at each publication of a distribution built from this recipe, by the license
check of `docs/release-runbook.md` §1 ("License check"); a field that names a date records the
check made on that date, and moving a pin needs the check again.

## Upstream sources

- **Weights** — [ku-nlp/deberta-v2-large-japanese-char-wwm](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm).
  This recipe exports the `text_encoder` seat of the SBV2 distribution (see `../sbv2/`).
- **Model implementation** — the `transformers` package. `patch.py` replaces
  `DisentangledSelfAttention.disentangled_attention_bias` with an equivalent that takes the
  relative-position tables as graph inputs. The equivalent (`_external_disentangled_attention_bias`)
  is adapted from the upstream method and carries the in-place notice; the package itself is not
  modified, and no file of it is copied.

## Upstream inventory

One block per upstream above. A block is completed before a distribution that carries its upstream
is published, and checked again when its pin moves.

### ku-nlp/deberta-v2-large-japanese-char-wwm

| Item                     | Value                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm>                                   |
| Revision used            | `547b0e8b044fba3f9b84d0ab9f990440bd130c8b` (pinned as `export.MODEL_REVISION`; fetched 2026-08-02)   |
| Form of copy             | Loaded, not copied. Truncated to its first 22 encoder layers and re-distributed in converted storage form in `karume-sbv2-jvnv` (its `text_encoder` component). |
| Code license             | n/a (weights only)                                                                                   |
| Weights license          | `cc-by-sa-4.0`, unchanged at the revision used (interview 2026-09-25 — recorded in `../sbv2/THIRD_PARTY_NOTICES.md`; HF models API at this revision, 2026-11-26). `../sbv2/card.py` records the same (checked 2026-08-07). |
| Attribution requirements | CC BY-SA 4.0 attribution and the indication of changes (truncation to 22 layers, quantization), carried by `karume-sbv2-jvnv`'s model card and `NOTICE.md`. The upstream gives no citation format (interview 2026-09-25). |

### transformers (model implementation)

| Item                     | Value                                                       |
| ------------------------ | ----------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>               |
| Revision used            | `transformers==5.14.1` (via the `sbv2` dependency group)    |
| Form of copy             | Monkeypatch of imported classes. `patch.py`'s `_external_disentangled_attention_bias` restates `DisentangledSelfAttention.disentangled_attention_bias` for the shared-key c2p + p2c path, receiving the relative-position tables instead of building them. |
| Code license             | Apache-2.0 — the installed wheel's own `LICENSE` (`transformers 5.14.1`; recorded in `../irodori/THIRD_PARTY_NOTICES.md`, 2026-09-05), identical to the `LICENSE` at the `v5.14.1` tag (checked 2026-11-26). |
| Weights license          | n/a                                                         |
| Attribution requirements | Nothing from `transformers` enters the distribution. In this recipe, `_external_disentangled_attention_bias` carries the in-place Apache 2.0 notice — the upstream file's copyright line, `Copyright 2020 Microsoft and the Hugging Face Inc. team.`, and what was changed — in the form the irodori and wan recipes use (decided 2026-11-26). |

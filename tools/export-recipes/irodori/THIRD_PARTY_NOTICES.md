# Third-party notices — Irodori-TTS v4

Skeleton for the pre-release provenance review (ADR 0065 stage 6). It records what this recipe
directory is known to derive from. **It is not a license determination.**

This directory (`tools/export-recipes/irodori/`) is repo-only: it is never published to PyPI and
none of it enters the `karume` wheel (ADR 0065 decision 1). Upstream-derived code lives next to
the recipe that needs it so its provenance travels with the code.

`Unverified` marks a field nobody has checked against the upstream revision actually used.
Checking license compatibility per revision is a human review scheduled before release
(ADR 0065 decision 7); filling this table in _is_ that review, and this file only lays out the
questions it has to answer.

## Upstream sources

- **Weights** — [Aratako/Irodori-TTS-v4-Small](https://huggingface.co/Aratako/Irodori-TTS-v4-Small).
- **Model implementation** — [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
  imported from a local clone through `sys.path` (`--source-dir`); no copy lives here. `patch.py`
  replaces class attributes and self-reports that the untouched parts of the replaced forwards are
  verbatim.
- **Text backbone** — [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m),
  the ModernBERT-ja checkpoint the text encoder was built from, re-distributed inside the Karume
  repository. Its modeling code comes from `transformers` (pinned `transformers==5.14.1`).
- **Codec weights** — [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim),
  also re-distributed.
- **Codec implementation** — <https://github.com/facebookresearch/dacvae>, pinned at commit
  `414c20785fc3a28373073ea8ef7a1316eeeaca6e` (`dacvae/export.py`), imported from a local clone
  through `sys.path`; no copy lives here.

## Release-gate inventory

One block per upstream above. Complete every row before anything built from this recipe is
published.

### Aratako/Irodori-TTS-v4-Small (weights)

| Item                     | Value                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Aratako/Irodori-TTS-v4-Small>                                                     |
| Revision used            | Unverified                                                                                                |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                             |
| Code license             | n/a (weights only)                                                                                        |
| Weights license          | `card.py` records `mit` (checked on the HF models API, 2026-08-12). Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                                |

### Aratako/Irodori-TTS (model implementation)

| Item                     | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Upstream repository      | <https://github.com/Aratako/Irodori-TTS>                         |
| Revision used            | Unverified                                                       |
| Form of copy             | Imported from a clone via `sys.path`; no copy in this directory. |
| Code license             | `card.py` records MIT. Unverified against the revision used.     |
| Weights license          | n/a                                                              |
| Attribution requirements | Unverified                                                       |

### sbintuitions/modernbert-ja-310m (text backbone)

| Item                     | Value                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/sbintuitions/modernbert-ja-310m>                                                  |
| Revision used            | Unverified                                                                                                |
| Form of copy             | Re-distributed inside the Irodori distribution.                                                           |
| Code license             | n/a (weights only; modeling code is `transformers`)                                                       |
| Weights license          | `card.py` records `mit` (checked on the HF models API, 2026-08-12). Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                                |

### Aratako/Semantic-DACVAE-Japanese-32dim (codec weights)

| Item                     | Value                                                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>                                                              |
| Revision used            | Unverified                                                                                                                   |
| Form of copy             | Re-distributed inside the Irodori distribution.                                                                              |
| Code license             | n/a (weights only)                                                                                                           |
| Weights license          | `card.py` records MIT (checked in `docs/research/2026-08-11-irodori-source-recon.md`). Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                                                   |

### facebookresearch/dacvae (codec implementation)

| Item                     | Value                                                                     |
| ------------------------ | ------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/facebookresearch/dacvae>                              |
| Revision used            | `414c20785fc3a28373073ea8ef7a1316eeeaca6e` (pinned in `dacvae/export.py`) |
| Form of copy             | Imported from a clone via `sys.path`; no copy in this directory.          |
| Code license             | Unverified                                                                |
| Weights license          | n/a                                                                       |
| Attribution requirements | Unverified                                                                |

### transformers (ModernBERT implementation)

| Item                     | Value                                                       |
| ------------------------ | ----------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>               |
| Revision used            | `transformers==5.14.1`                                      |
| Form of copy             | Monkeypatch of imported classes; no copy in this directory. |
| Code license             | Unverified                                                  |
| Weights license          | n/a                                                         |
| Attribution requirements | Unverified                                                  |

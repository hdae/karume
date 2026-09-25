# Third-party notices — Irodori-TTS v4 / v4.1

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

- **Weights** — [Aratako/Irodori-TTS-v4-Small](https://huggingface.co/Aratako/Irodori-TTS-v4-Small)
  and [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)
  (one distribution repository per model; `card.py`'s `IRODORI_UPSTREAMS` picks by model name).
- **Model implementation** — [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
  imported from a local clone through `sys.path` (`--source-dir`). `patch.py` replaces class
  attributes, and one of its replacement forwards is a verbatim copy of the upstream forward except
  for one line (see the block below).
- **Text backbone** — [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m),
  the ModernBERT-ja checkpoint the text encoder was built from, re-distributed inside the Irodori
  distribution repositories. Its modeling code comes from `transformers` (pinned `transformers==5.14.1`).
- **Codec weights** — [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim),
  also re-distributed. Its weights derive from
  [Aratako/Semantic-DACVAE-Japanese](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese)
  (MIT), whose weights in turn derive from
  [facebook/dacvae-watermarked](https://huggingface.co/facebook/dacvae-watermarked) (Apache-2.0) —
  the chain recorded in `card.py` (checked 2026-09-24).
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

### Aratako/Irodori-TTS-v4.1-Small (weights)

| Item                     | Value                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small>                                                   |
| Revision used            | Unverified                                                                                                |
| Form of copy             | Loaded, not copied. Re-distributed in converted storage form.                                             |
| Code license             | n/a (weights only)                                                                                        |
| Weights license          | `card.py` records `mit` (checked on the HF models API, 2026-09-01). Unverified against the revision used. |
| Attribution requirements | Unverified                                                                                                |

### Aratako/Irodori-TTS (model implementation)

| Item                     | Value                                                                                                                                                                                                                                                           |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/Aratako/Irodori-TTS>                                                                                                                                                                                                                        |
| Revision used            | Unverified                                                                                                                                                                                                                                                      |
| Form of copy             | Imported from a clone via `sys.path`. `patch.py`'s `_folded_rms_low_rank_adaln_forward` is a verbatim copy of `irodori_tts.model.LowRankAdaLN.forward` except for the normalization line (its docstring says so).                                               |
| Code license             | MIT — read from the `LICENSE` of the clone this recipe imports (`inputs/irodori/Irodori-TTS`, checked 2026-09-05; "Copyright (c) 2026 Aratako").                                                                                                                |
| Weights license          | n/a                                                                                                                                                                                                                                                             |
| Attribution requirements | Undecided (human review). MIT requires the copyright notice and permission text to travel with copies of the software; the verbatim fragment in `patch.py` (Form of copy) lives in this repository, while no upstream source enters the published distribution. |

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

| Item                     | Value                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Upstream repository      | <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim>                                                                                                                                              |
| Revision used            | Unverified                                                                                                                                                                                                   |
| Form of copy             | Re-distributed inside the Irodori distribution.                                                                                                                                                              |
| Code license             | n/a (weights only)                                                                                                                                                                                           |
| Weights license          | `card.py` records MIT (checked in `docs/research/2026-08-11-irodori-source-recon.md`). Unverified against the revision used. The weights derive from the two blocks below, and the chain ends in Apache-2.0. |
| Attribution requirements | Unverified                                                                                                                                                                                                   |

### Aratako/Semantic-DACVAE-Japanese (codec weights — parent)

| Item                     | Value                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/Aratako/Semantic-DACVAE-Japanese>                                                                           |
| Revision used            | n/a — not loaded; `card.py` records the revision seen on the HF models API (`96adcf19…`, checked 2026-09-24).                       |
| Form of copy             | Not copied directly. It is the `base_model` of the codec weights above.                                                             |
| Code license             | n/a (weights only)                                                                                                                  |
| Weights license          | `card.py` records `mit` (HF models API, checked 2026-09-24). Its README says the weights derive from `facebook/dacvae-watermarked`. |
| Attribution requirements | Unverified                                                                                                                          |

### facebook/dacvae-watermarked (codec weights — origin)

| Item                     | Value                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://huggingface.co/facebook/dacvae-watermarked>                                                                                                                                                                                                                                                                         |
| Revision used            | n/a — not loaded; `card.py` records the revision seen on the HF models API (`8680102d…`, checked 2026-09-24).                                                                                                                                                                                                                |
| Form of copy             | Not copied directly. It is the `base_model` of the parent above, so the redistributed codec weights derive from it.                                                                                                                                                                                                          |
| Code license             | n/a (weights only)                                                                                                                                                                                                                                                                                                           |
| Weights license          | `card.py` records `apache-2.0` (the HF license tag, checked 2026-09-24). The repository README says "SAM License"; the upstream discussions/1 confirms that the README is wrong (user-confirmed 2026-09-24 — `card.py`).                                                                                                     |
| Attribution requirements | Apache 2.0 §4(a)(b): the distribution repositories carry the Apache 2.0 text and the attribution / change notice (`distribution.py`'s `irodori_root_files` — `LICENSE.md` / `NOTICE.md`). Neither the HF nor the GitHub upstream carries a `NOTICE` file, so §4(d) has nothing to propagate (`card.py`, checked 2026-09-24). |

### facebookresearch/dacvae (codec implementation)

| Item                     | Value                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/facebookresearch/dacvae>                                                                                                                                                                                     |
| Revision used            | `414c20785fc3a28373073ea8ef7a1316eeeaca6e` (pinned in `dacvae/export.py`)                                                                                                                                                        |
| Form of copy             | Imported from a clone via `sys.path`; no copy in this directory.                                                                                                                                                                 |
| Code license             | Apache-2.0 — read from the `LICENSE` of the pinned clone (`inputs/irodori/dacvae-src`, whose HEAD is the pinned `414c2078…`; checked 2026-09-05). The upstream tree carries no `NOTICE` file, so §4(d) has nothing to propagate. |
| Weights license          | n/a                                                                                                                                                                                                                              |
| Attribution requirements | None attach here: nothing is copied into this directory and no upstream source enters the published distribution (import-time only), and Apache 2.0 §4 starts at redistribution. The codec **weights** are the block above.      |

### transformers (ModernBERT implementation)

| Item                     | Value                                                                                                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream repository      | <https://github.com/huggingface/transformers>                                                                                                                                                                  |
| Revision used            | `transformers==5.14.1`                                                                                                                                                                                         |
| Form of copy             | Monkeypatch of imported classes. `patch.py`'s `_flat_qkv_attention_forward` is a verbatim copy of `ModernBertAttention.forward` except for how qkv is split (its docstring says so).                           |
| Code license             | Apache-2.0 — read from the installed wheel's own `LICENSE` (`transformers 5.14.1`, checked 2026-09-05; "Copyright 2018- The Hugging Face team").                                                               |
| Weights license          | n/a                                                                                                                                                                                                            |
| Attribution requirements | Undecided (human review). Apache 2.0 §4 starts at redistribution; no `transformers` code enters the published distribution, while the verbatim fragment in `patch.py` (Form of copy) lives in this repository. |

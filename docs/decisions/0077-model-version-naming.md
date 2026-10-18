# 0077: Model names carry the upstream version

- Status: accepted (2026-08-22, the maintainer's ruling; applied at once to the anima base
  repository); partly superseded by [0088](0088-civitai-air-intake.md) ("not normalized" revised
  for Civitai intake, 2026-09-01); the non-retroactivity clause re-graded to MAY on 2026-10-18
- Date: 2026-08-22
- Related: [0041](0041-manifest-v2.md) (`models` map, per-model subtrees) /
  [0073](0073-models-source-pin.md) (default source pin) / [0074](0074-quant-seat-naming.md) (preset
  naming — another axis) / [0088](0088-civitai-air-intake.md) (Civitai intake defaults to a
  machine-normalized name; the provenance keeps the verbatim spelling)

## Context

Third-party fine-tunes ship several versions side by side, and the newest is not always the best
(the maintainer, 2026-08-22). copycat-anima had six versions on Civitai from 2026-02-16 to 06-10;
WAI-ANIMA ships `v1.0` and `PW3`, two lines with no order between them.

A model name in the distribution manifest (ADR 0041) is a key of the `models` map and also the name
of the subtree in the repository. Without a version in the name, the only way to ship a new version
is to replace the weights under the same name. Then:

- Users who follow `revision: "main"` silently switch to different weights.
- A pin (ADR 0073) protects pinned users, but the old bytes still leave the repository; a user who
  preferred the previous version cannot go back.
- The card already says `Converted from: copycatAnima_20260610.safetensors`, a version, and so
  disagrees with an unversioned model name.

## Decision

**A model name is `<family>-<upstream version>`.** The version is the one the upstream uses for
itself, reduced only to `[A-Za-z0-9._-]` (the set `assert_model_name` in
`tools/exporter/src/karume/dist.py` accepts).

| Model name               | Upstream name                |
| ------------------------ | ---------------------------- |
| `anima-v1.0`             | Anima Base **v1.0**          |
| `anima-wai-v1.0`         | WAI-ANIMA **v1.0**(base 1.0) |
| `anima-copycat-20260610` | copycat-anima **20260610**   |

- **The spelling is not normalized.** Upstreams mix semver-like and date versions; rewriting them
  (for example into dates) would break the match with the source page. Unordered lines follow the
  same rule (WAI's `PW3` becomes `anima-wai-pw3`). (Revised by ADR 0088 for Civitai intake.)
- **`defaultModel` holds the recommendation.** A ruling moves the default, not a newer version.
- **The series directory name derives from the model name** (`{model}-{dtype}-dyn`), so series
  of two versions cannot be mixed up.

## Consequences

- One more version costs only its transformer and text_conditioner (6.14 GB for anima); the text
  encoder, VAE and tokenizer sit in `shared/` (ADR 0041 §5). Users fetch only the model they choose.
- **No policy yet for dropping old versions**; it is ruled when a limit becomes necessary (the
  repository grows linearly; 3 models, 19 GB on 2026-08-22).
- Not applied retroactively: SBV2's `F1` / `FN4` are speaker identifiers, not versions. A family
  that publishes one model per repository MAY carry the version in the repository name instead
  (re-graded 2026-10-18 to MAY: irodori, kept as is when it had one model, now ships v4-small and
  v4.1-small as separate repositories — `IRODORI_SOURCES` in `packages/models/src/irodori/config.ts`
  — and Wan2.2 has its own repository `hdae/karume-wan2.2`).
- The published `anima-turbo` was not renamed (pins pointed at it); ADR
  [0087](0087-anima-official-extra-repos.md) retired it with its repository on 2026-09-01.

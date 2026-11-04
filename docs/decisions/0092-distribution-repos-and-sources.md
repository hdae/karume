# 0092: Granularity and naming of distribution repositories, and the public repository table in models

- Status: accepted (2026-09-04 — repository rules fixed before the first publication of the families
  not yet distributed). The path-dependency clause of decision 6 was re-graded to MAY on 2026-10-18
- Date: 2026-09-04
- Related: [0038](0038-manifest-v1.md) §7 (cross-repository component references — the referenced
  revision is baked) / [0041](0041-manifest-v2.md) (`karume-` repository prefix; several models per
  repository + `defaultModel`) / [0065](0065-exporter-core-recipe-split.md) (core = PyPI, recipes =
  repository-only) / [0073](0073-models-source-pin.md) (why sources are pinned — **the shape of the
  constants is replaced here**) / [0087](0087-anima-official-extra-repos.md) (anima's "official /
  fine-tuned" split, the precedent for the granularity rule)

## Context

**Repository splits were ruled case by case.** The six existing repositories (`karume-anima`,
`karume-anima-extra`, `karume-sbv2-jvnv`, `karume-irodori-v4-small`, `karume-irodori-v4.1-small`,
`karume-gemma4-e2b`) were each decided individually; no rule was written down. The next wave publishes
families not yet distributed (siglip2 base / so400m, BiRefNet HR / Lucida, depth-anything v2,
vowel-detector), and each raises "together or separate" first.

**The pin constants kept the shape of the single-default era.** ADR 0073 decision 2 (one default
source per family) was withdrawn on 2026-08-25, and the constants became
`<FAMILY>[_<VARIANT>]_CURRENT`, "the source this package version was verified against". The shape
stayed "one public repository = one constant" (six then): the type has no place for "all public
repositories of this family", and adding one means a constant plus a barrel re-export — manual work
that can be forgotten. The pin decision was fine; the container was not.

**The revision can only live on the TS side.** An HF model card cannot state its own revision (the
commit does not exist yet when the card is written). The rule not to copy SHAs into docs
(release-runbook §3) follows from this; copies go stale on every rebuild. "Public repository → its
revision" therefore lives in exactly one place, the constants in models.

**The motive for splitting the recipes out is licensing, not structure.** The parked backlog item
"move export-recipes to a separate repository" came from the concern that upstream-derived code
(patch layers, reference pipelines, export scripts) reads as if it were under this repository's MIT
license. ADR 0065 already separated the structure, enforced by a machine check; a separate repository
would buy only appearance, at the cost of dismantling the uv workspace, injecting asset roots and
fixture destinations, and CI on both sides.

**Derivatives with awkward licensing need a home.** For assets with unknown training data or
uncensored derivatives the project does not want to carry the written basis for redistribution
(`karume-sbv2-fn`, on hold, is the precedent); listing them in models would make it accountable.

## Decision

### 1. Granularity = one repository per family; derivatives, generations and versions separate

- **Variants of one family share a repository** (Gemma 4 E2B / E4B / 12B in `karume-gemma4`; amended
  2026-11-04, see the addendum — today Gemma 4 E2B / E4B in `karume-gemma4`), using
  ADR 0041's "several models per repository + `defaultModel`".
- **A derived model gets its own repository** (Lucida, derived from BiRefNet, is `karume-lucida`).
- **A family with many derivatives bundles them in `-extra`** (anima, ADR 0087's precedent as a rule).
- **A new generation or version gets a new repository** (irodori v4 and v4.1 separate; Gemma 5 would
  be new).
- **BiRefNet and BiRefNet HR are separate** (same family name, distributed as different models).

### 2. Repository naming

- Official: `karume-<family>[-<generation or version>][-<variant>]` (`karume-gemma4`,
  `karume-irodori-v4.1-small`, `karume-birefnet-hr`, `karume-depth-anything-v2`, `karume-siglip2`).
- Bundle of derivatives: `karume-<family>-extra`. A single derivative: `karume-<derivative>`
  (`karume-lucida`).
- **Dots in versions are allowed** (`karume-irodori-v4.1-small` keeps upstream's naming — the same
  motive as ADR [0077](0077-model-version-naming.md)).

The `karume-` prefix is inherited from ADR 0041 (namespacing by repository name without an HF org).

### 3. The public repository table in models = `<FAMILY>_SOURCES`

Each family keeps **all its public repositories** in one table.

```ts
export const IRODORI_SOURCES = {
  "irodori-v4-small": { repo: "hdae/karume-irodori-v4-small", revision: "<40hex>" },
  "irodori-v4.1-small": { repo: "hdae/karume-irodori-v4.1-small", revision: "<40hex>" },
} as const satisfies Record<string, HubRepoRef>;
```

- **Key = the HF repository basename without `karume-`.** Hyphens and dots stay, so access uses
  brackets (`IRODORI_SOURCES["irodori-v4.1-small"]`). An invented identifier-friendly alias would
  leave the mapping to human memory.
- **A single-repository family also uses the table, keyed by its name** (`GEMMA4_SOURCES["gemma4"]`).
  That the shape does not change when repositories are added is the point of this decision.
- **The barrel holds `KARUME_SOURCES`, the union of all families** (same keys;
  `packages/models/src/sources.ts`). It is the one place listing every public repository; the
  connectivity smoke test (`tools/published-smoke`) and the completeness check of pin updates iterate
  it.
- **No default source** (ADR 0073's withdrawal of 2026-08-25 stands — the caller always names the
  source; the opening of ADR [0086](0086-distribution-source.md) points the same way).
- **Each revision is the commit SHA at publication, updated on every bump** (ADR 0073 decision 3: hand
  edit + runbook step, with the same maintenance duty).
- **The invariants are machine checks** (`packages/models/tests/sources_test.ts`): `"karume-" + key`
  equals the repository basename, the owner is `hdae`, `revision` is 40 hex digits.

### 4. `*_CURRENT` is removed (breaking, 0.9.0)

The six `<FAMILY>[_<VARIANT>]_CURRENT` constants move into the tables of decision 3. A breaking change
of the JSR surface while unreleased, without a compatibility shim.

### 5. export-recipes stays in this repository (the licensing motive is solved by a carve-out)

ADR 0065 decision 1 (repository-only) is unchanged; **no separate repository**. The licensing concern
is settled in writing:

- The License section of the root README and the top of the `tools/export-recipes/` README state that
  **this directory contains upstream-derived code, follows each family's `THIRD_PARTY_NOTICES.md`, and
  is not covered by the project's MIT license**.
- **Families that declare verbatim ports of MIT code** (birefnet `patch.py` / `export.py`) carry
  **upstream's copyright notice itself** in their `THIRD_PARTY_NOTICES.md` (MIT's condition to include
  the copyright and permission notice).

This closes the parked backlog item "move export-recipes to a separate repository".

### 6. Derivatives with awkward licensing stay out of this repository

Assets whose redistribution the project cannot back in writing (unknown training data, uncensored
derivatives, …) are built as **a separate library using exporter core + a separate HF repository**.
No pipeline is added to models and they do not enter the table of decision 3. Such a library MAY
reference core through a path dependency (re-graded 2026-10-18 to MAY: the rule rested on PyPI
`karume` being unreleased; once it is published a PyPI dependency is equally acceptable. As of
2026-10-18 [release-runbook](../release-runbook.md) §4 still records PyPI `karume` as unreleased,
while `tools/exporter/pyproject.toml` carries version 0.14.0).

### 7. `LICENSE.md` / `NOTICE.md` ship at the root of each distribution repository

Only `LICENSE.md` / `NOTICE.md` may sit at a distribution repository's root (`LEGAL_PATHS` in
`tools/exporter/src/karume/dist.py`). By upstream license:

- **Apache-2.0**: full text + the modification notice of §4(b).
- **MIT**: full text + the copyright line.

This applied to siglip2 / birefnet-hr / lucida / depth-anything-v2. The already published
`karume-irodori-*` (MIT) and `karume-sbv2-jvnv` (CC BY-SA) lacked them, to be fixed at their next
upload (filed in the backlog).

**Addendum (2026-09-24) — the CC BY-SA 4.0 form and the three published repositories.** The
maintainer ruled: irodori gets the MIT text with `Copyright (c) 2026 Aratako`; sbv2-jvnv gets the
CC BY-SA 4.0 legal code; the SBV2 pipeline is split by voice family. The implementation added
`Copyright (c) 2025 SB Intuitions` to irodori (from the `LICENSE` of upstream
`sbintuitions/modernbert-ja-310m`) and named the split pipelines `sbv2` (JVNV, keeping the published
spelling) and `sbv2-fn` (FN). The resulting form:

- **CC BY-SA 4.0**: the full legal code (verbatim plain text from creativecommons.org —
  `tools/export-recipes/_shared/licenses/cc_by_sa_4_0.txt`) + attribution and modification notice of
  §3(a)(1). A URI would satisfy §3(a)(1)(C) / §3(b)(2), but the full text in `LICENSE.md` travels with
  copies outside HF. `NOTICE.md` holds the upstream origin and authors, training data, the
  modifications and the share-alike statement; without an upstream copyright notice no line is
  invented.
- **sbv2-jvnv**: both upstreams (voice `litagin/style_bert_vits2_jvnv` and text encoder
  `ku-nlp/deberta-v2-large-japanese-char-wwm`) are CC BY-SA 4.0, so one set of the form above. SBV2
  licenses differ per voice family, so the pipeline is **split by family** (`--pipeline sbv2` = JVNV,
  `--pipeline sbv2-fn` = FN). FN's only written terms are on its Booth distribution page, with no text
  to bundle, and its HF publication is on hold, so its legal text is empty.
- **irodori**: MIT. The line `Copyright (c) 2026 Aratako` from the upstream implementation repository
  (GitHub `Aratako/Irodori-TTS`) is applied to the same author's weights (model and codec), together
  with `Copyright (c) 2025 SB Intuitions` from the text backbone's base
  `sbintuitions/modernbert-ja-310m` (the backbone is its fine-tune, so the upstream notice must stay).
  `NOTICE.md` refers to "the checkpoint listed in `README.md`" without naming a version, because one
  set of legal text goes to both the v4 and v4.1 repositories.

### 8. siglip2: one repository, two models (default = base)

`karume-siglip2` holds base and so400m (decision 1). The siglip2 recipe's "one model per repository"
constraint is **withdrawn**.

### 9. BiRefNet: one repository, two models named by resolution (addendum 2026-09-05, maintainer's ruling)

BiRefNet needs a **separate graph per resolution** (the shifted-window mask and padding constants
depend on it — the "resolution axis" of `birefnet/export.py`), so 1024² and 2048² of one checkpoint
share a repository as in decision 8. **The manifest model names are the resolutions `"1024"` /
`"2048"`** (default `"1024"`; only squares can be baked, so no `x`). Per decisions 1 / 2 there are
two repositories, `karume-birefnet-hr` (upstream BiRefNet_HR) and `karume-lucida` (derivative);
`karume-birefnet` stays free for plain BiRefNet. Callers write
`fromPretrained(BIREFNET_SOURCES["birefnet-hr"], { model: "2048" })` (`BirefnetPipeline` already
reads `model` / `defaultModel`). 2048² became distributable through recipe patch ⑨ and ADR
[0093](0093-transient-liveness-packing.md) (measurements in its Consequences).

## Consequences

- **Breaking in 0.9.0**: the six `*_CURRENT` constants become `<FAMILY>_SOURCES` + `KARUME_SOURCES`;
  examples, `tools/published-smoke` and the model cards' usage examples follow.
- **Runbook §3 changes** from "find the constant and write it" to "fill the key → check that
  `KARUME_SOURCES` is complete → connectivity check". The entry for a newly published repository is
  **created at that point** (an entry before publication would only 404 — ADR 0073 decision 1).
- **`karume-gemma4-e2b` is renamed to `karume-gemma4`** (decision 1 — the repository will hold E4B /
  12B too); HF redirects the old name. The rename belongs to the release procedure (runbook §2).
- **Scope of the publication wave**: siglip2 (one repository, two models), birefnet-hr and lucida
  (**after confirming 2048² is feasible**), depth-anything-v2. **vowel-detector is left out** (upstream
  presentation first — ruling of 2026-09-04).
- **SHAs are still not copied into docs** (runbook §3 check unchanged); only the table holds them.
- **`*_CURRENT` spellings in ADRs [0050](0050-irodori-quant-series.md) /
  [0073](0073-models-source-pin.md) / [0087](0087-anima-official-extra-repos.md) are historical** and
  no longer exist on the public surface.
- With decision 5 the uv workspace (`tools/` as root, exporter and export-recipes as members) stays,
  and so does running `uv run pytest` on both sides.

## Addendum (2026-11-04) — the Gemma 4 lines and decision 1

The maintainer's ruling of 2026-11-04 (ADR [0097](0097-gemma4-qat-integration.md) addendum 10, ADR
[0128](0128-default-preset-original-weights.md) addendum 2026-11-04) lays Gemma 4 out over three
repositories:

| Repository                 | Models                                     | Weights                                                | Contract / reader                                  |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------ | -------------------------------------------------- |
| `karume-gemma4`            | E2B (drafter), E4B                         | `google/gemma-4-E{2,4}B-it`                            | `gemma4/1`, `@karume/models/gemma`                 |
| `karume-gemma4-qat`        | E2B / E4B / 12B, each with its QAT drafter | `google/gemma-4-{E2B,E4B,12B}-it-qat-q4_0-unquantized` | `gemma4/1`, `@karume/models/gemma`                 |
| `karume-gemma4-qat-mobile` | E2B / E4B                                  | `google/gemma-4-E{2,4}B-it-qat-mobile-transformers`    | `gemma4-qat/1`, `@karume/models/gemma4-qat-mobile` |

Decision 1's example "Gemma 4 E2B / E4B / 12B in `karume-gemma4`" no longer describes the layout: 12B
is only in `karume-gemma4-qat`. The ruling did not cite this ADR. Read against decision 1, the layout
fits its derived-model rule: the q4_0 QAT checkpoints are other upstream checkpoints, quantization-aware
fine-tunes whose model cards name the `-it` models as `base_model`, so they get a repository of their
own even though the same reader (`gemma4/1`) loads both. The same reading applies to the mobile line,
which also has a contract of its own. Decision 1's rule for variants still holds inside each
repository (several models, one `defaultModel`). One family reader may therefore serve two
repositories; decision 3's table (`gemma.SOURCES`) has one key per repository, and the q4_0 line gets
its key when it is first published (release-runbook "First publication of a family").

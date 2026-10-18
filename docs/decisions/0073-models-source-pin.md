# 0073: Default sources of models are pinned to the commit SHA at publication

- Status: accepted (2026-08-20 — maintainer ruling "please implement the pin"; pins are written right
  after the HF upload, once the SHA exists; procedure canonical in
  [release-runbook.md](../release-runbook.md) §3); partly superseded: decision 2 was withdrawn on
  2026-08-25 (addendum), and decision 1's constant shape by [0092](0092-distribution-repos-and-sources.md)
  decision 3 (2026-09-04: per-family tables `<FAMILY>_SOURCES` and `KARUME_SOURCES`); decision 3's
  rejection of automated stamping was re-graded to MAY on 2026-10-18
- Date: 2026-08-20
- Related: ADR [0038](0038-manifest-v1.md) (revision resolution — passing a SHA issues no resolution
  request) / [0041](0041-manifest-v2.md) (repository naming with the `karume-` prefix) /
  [0071](0071-manifest-v3-shards.md) (the pin absorbs format breaks)

## Context

The models package held neither repository names nor revisions, and the hub defaulted to
`revision ?? "main"` (now `packages/hub/src/sources/hf.ts`). With that, **what a published models
package reads changes silently** the moment the distribution repository's manifest or assets change
(presets added or removed, `defaultQuant` changed, a format version bump breaks or alters it).
Branches and tags can be moved on HF; only a commit SHA is immutable. A SHA pin plus the manifest's
sha256 checks fix the bytes a published package reads without relying on the network side's good
will. As a side benefit, a SHA issues no revision resolution request, so **the default path starts
offline when everything is cached**.

## Decision

### 1. Families with an official repository get pinned source constants

Each family config carries a constant `{ repo, revision: "<40hex>" } as const satisfies HubRepoRef`,
exported from the subpath, for families whose public repository exists. Families without one keep
requiring a ref. No pin constant for fn: `karume-sbv2-fn` is not published (2026-08-20 ruling,
backlog parked; it is added when published — today `SBV2_SOURCES` has only `sbv2-jvnv`). This is
**the first link from models to its official repositories** (models were repository-agnostic
before), chosen for the out-of-the-box experience.
**(superseded by ADR 0092 decision 3 on 2026-09-04: the constants are per-family tables
`<FAMILY>_SOURCES` in each family's `config.ts`, folded by `KARUME_SOURCES` in
`packages/models/src/sources.ts`, nine families. The repositories listed in 2026-08 included
`hdae/karume-anima-turbo`, retired by ADR 0087 on 2026-09-01; anima's current table is
`ANIMA_SOURCES` with `anima` / `anima-extra`.)**

### 2. `fromPretrained` without a ref (families with a pin)

Withdrawn 2026-08-25 (addendum): `ref` is required again for every family.

### 3. Pins are updated by hand through the runbook check

Every publication updates the pins by the procedure of release-runbook §3 (get the SHA → write the
table entry → connectivity check → verify → commit). Stamping by a generator plus a matching task is
not adopted for now; the maintainer MAY adopt it (re-graded 2026-10-18 to MAY: the premise "few
publications so far" no longer holds — about ten releases from 0.5.0 to 0.14.0 and 12 public
repositories in the source tables — and hand copies of SHAs into docs disagreed in two consecutive releases; the
invariants 40-hex revision, key derived from the repository name and owner are already checked by
`packages/models/tests/sources_test.ts`). Using HF git tags as versions is not adopted: tags can be
moved, so immutability would depend on discipline.

## Consequences

- Release order is not commutative: **HF upload → write the pins → JSR publish** (runbook §0).
- A fix on the asset side reaches users only with a models patch release (a deliberate trade for
  stability). The opposite accident — a published package silently broken — is unrecoverable, so the
  design leans to the recoverable side.
- The hub's default `revision ?? "main"` stays (the semantics of using the hub alone are unchanged).

## Addenda

- 2026-08-24: **non-default public repositories get the same pin.** MUST: **every constant through
  which models names a repository is pinned to a commit SHA.** The reason is the same: "not the
  default, so following `main` is fine" does not hold — silent changes to what a published package
  reads are unrecoverable whether or not the source is the default. Decision 3's procedure applies to
  every entry.
- 2026-08-25: **decision 2 withdrawn** (0.5.0 breaking wave — maintainer ruling). `fromPretrained`
  requires `ref` for every family. Examples and model cards had spread the "pass the repository
  string" form, and nothing led users to the baked-in default; the per-family optional branch and
  the explanation of "where does a call without arguments read" did not pay for themselves. In
  TypeScript the missing argument fails type checking; an untyped JavaScript call without arguments
  fails loudly in the shared check (`toRepoRef` in `packages/models/src/hub/repo-ref.ts`) with two
  spelling examples, instead of letting `undefined` reach the hub. A string ref still means "follow
  `main`".
- 2026-08-25: the constants were renamed `*_CURRENT` (later replaced by ADR 0092's tables). The
  maintenance duty carries over: **on every version bump the pins are updated (decision 3) and, as
  needed, checked by running the models, before the table claims them as verified.** An unverified
  pin under that name would be a lie.
- 2026-08-25: the hub, as the counterpart, **warns once via `console.warn` when it resolves `main`
  implicitly** because `revision` was omitted (`warnImplicitMain` in `packages/hub/src/sources/hf.ts`).
  It prints the resolved SHA and offers two options: copy that line into `revision`, or use the
  `@karume/models` source tables. An explicit `"main"`, a tag or a SHA does not warn (an explicit ref
  states that a mutable ref is acceptable). This keeps a path to pins for users of the hub alone.
- 2026-08-25: HF git tags were reconsidered and rejected again (maintainer ruling); one more reason —
  "point at the version matching the package version" is already met by the source tables, so tags
  would duplicate them.
- 2026-08-31: **the "no default source" MUST extends to source handles** (ADR
  [0086](0086-distribution-source.md)). The first argument of `fromPretrained` is
  `string | HubRepoRef | DistributionSource`, and **no spelling may be omitted** — the caller always
  names the source. A source handle **does not pass through `toRepoRef`** (it is not an HF
  repository, so requiring `owner/name` would reject local distributions). The distinction is the hub
  type predicate `isDistributionSource` alone (`toManifestSource`), and the handle goes straight to
  `loadManifest`. Handles have no pin counterpart: a source without generations treats the bytes at
  hand as canonical (ADR 0086 decision 1; no implicit-`main` warning either).
- 2026-09-04: the constant shape was replaced by ADR [0092](0092-distribution-repos-and-sources.md)
  decision 3 (0.9.0 breaking). **The reason for pinning (Context), decision 3's procedure and the
  maintenance duty of 2026-08-25 stay with this ADR** — only the container changed.

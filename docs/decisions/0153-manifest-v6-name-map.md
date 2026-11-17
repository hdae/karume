# 0153: Manifest `karume/6` — a distribution carries its family's name map

- Status: accepted (2026-11-17 — the maintainer's decision; implemented in `@karume/hub`, the exporter's `karume dist`
  and `karume upgrade`, and Anima's recipe)
- Date: 2026-11-17
- Related: [0109](0109-manifest-v5-container.md) (`karume/5`, one format read — decision 1) /
  [0038](0038-manifest-v1.md) §1 (the bump rule: a member that is not a listed extension point is a new major) and §7
  with its addendum of 2026-10-28 (cross-repository references) / [0041](0041-manifest-v2.md) decision 9 (the layout of
  a repository) / [0152](0152-maintained-specs-and-declared-numeric-requirements.md) (item 7 of its backlog list:
  weights from outside the exporter) / [spec/convert](../spec/convert.md#the-name-map) (the name map) /
  [release-runbook](../release-runbook.md) §0 (the publication order of repositories with cross-repository
  references)

## Context

`@karume/convert` places the tensors of an upstream weights file with a **name map** (format `karume-name-map/1`): data
that says which tensor of the file is which initializer of which component, per layout the upstream files exist in.
Anima's map is written beside its export recipe (`tools/export-recipes/anima/name-map.json`), because the recipe is the
one place that knows both names — the upstream tensor names it reads and the initializer names of the components it
exports. A client that converts a user's upstream file against a distributed preset needs the map of that preset's
family, and today the map exists only in this repository.

`karume/5` has no member for it. The hub refuses unknown top-level keys, and a member that is not one of the listed
extension points takes a new major (ADR 0038 §1). The 18 local mirrors and every published repository hold `karume/5`;
every published client pins its repositories' revisions (40-digit commits).

## Decision

1. **`karume/6` is `karume/5` plus one optional top-level member, `nameMap`**: the reference of the repository's name
   map file, the way the manifest references any file — `{ "path", "size", "sha256" }`. It is checked like every file
   reference (the path allow-list, `1 ≤ size ≤ 16 GiB`, 64 lowercase hex digits, agreement with any other reference of
   the same path). It names a file of the repository itself: a cross-repository pair (`repo` / `revision`) is refused.
   A repository without a name map does not write the member. The file's content is a `karume-name-map/1` document.
2. **The hub reads `karume/6` only** — one format, as with `karume/5`. A `karume/5` manifest fails as an unsupported
   major, and the message says what to do: the repository needs a `karume/6` manifest, which the exporter's upgrade
   command writes in place. Nothing in the hub or in `@karume/models` reads the map's content, and the map belongs to
   no selection (it is not among the files a selection fetches, counts or evicts) until a reader uses it.
3. **The writer.** The manifest format of `karume dist` is `karume/6`. A recipe declares its family's name map
   (`Pipeline.name_map`); the assembly checks that the file is JSON with the format tag `karume-name-map/1` (it does not
   interpret the rules), copies it to the repository root as `name-map.json` and writes the reference. The writer's
   structural checks know the member: the top-level members are a closed set, and the reference is the 3-field set at
   `name-map.json`. Anima declares its map for both of its repositories, `karume-anima` and `karume-anima-extra`.
4. **The upgrade lives in the migration layer** (the one place that reads an old form): `karume upgrade <repository
   directory> [--name-map <file>]` reads a repository holding a `karume/5` manifest and writes its `karume/6` manifest in
   place — the same content (`generator` and every cross-repository reference as they are) with the new format and,
   when a name map is given, the copied file and its reference. Before writing, the repository as it is must pass the
   current checks (structure, limits, every declared file present with its size, no undeclared file). Each new file is
   written under a temporary name inside the directory, read back, and then put in place, the manifest last. No weights
   file is touched. The card is rewritten only where it states the format — the statement `` `karume.json`
   (`karume/5`) `` that every recipe's card spells from the manifest — and is otherwise left byte for byte; a card
   that does not state the format is not touched. On a `karume/6` repository the command reports that there is
   nothing to do (a name map given for it must be the one it references).

## Alternatives

- **The hub reads both `karume/5` and `karume/6`.** Only the two Anima repositories would change, and the other
  repositories would keep their manifests. Not chosen: two formats would be read, which ADR 0109 decision 1 ruled out
  for the same reason as here — the dual reading stays forever, and every later rule is written twice.
- **The manifest stays as it is, and the map sits at a conventional path** (`name-map.json` beside `karume.json`). No
  format change at all. Not chosen: there is no recorded size and sha256 for the file, so it would be the one file of a
  repository a client reads without checking it against the manifest, and a stale map could not be told from the right
  one.

## Consequences

- **Breaking**: the hub refuses every `karume/5` repository. Published clients pin their repositories' revisions, so
  none of them is affected; the clients of the next release read only repositories upgraded or built as `karume/6`.
- **Every repository's manifest is rewritten at the next release**, with the card where it states the format; the
  weights are not — no container changes, so no weights file is uploaded again. The local mirrors are upgraded with
  `karume upgrade`.
- **Cross-repository references are rebuilt in the runbook's order**: a referenced repository is uploaded first, and a
  repository that references it is built again against its new main and uploaded after it (release-runbook §0). The
  upgrade itself keeps every reference as it is: the commits a reference names keep holding the bytes it borrows.
- The upgraded manifest keeps its `generator` — the field names the exporter that converted the weights.
- `name-map.json` is the one file at a repository's root that the manifest declares (the layout of ADR 0041 decision 9
  had only the per-model subtrees and `shared/` declared).

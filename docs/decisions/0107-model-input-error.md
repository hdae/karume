# 0107: Represent input-caused failures with one cross-family `ModelInputError`

- Status: accepted (2026-09-22 — the type, its owner and the public surface, then the replacement in the
  families, completed the same day). The rule of decision 4 was restated on 2026-10-18 as its criterion
  instead of a count.
- Date: 2026-09-22
- Scope: `packages/models/src/errors.ts` (new — the type) / `packages/models/src/request-gates.ts` (new —
  owner of the cross-family acceptance sets) / `packages/models/src/sbv2/errors.ts` (re-parented) /
  `packages/models/src/generation/sequence.ts` (`GenerationCapacityError` re-parented) / the public
  surface (`packages/models/mod.ts` and every subpath entry that exports a pipeline); unit test
  `packages/models/tests/errors_test.ts`; public-surface fixture
  `packages/models/tests/fixtures/public-surface.json`. **The runtime, the hub, the distributions and the
  exporter are unchanged.**
- Related: ADR [0008](0008-public-api.md) (a thin public surface; additions are justified by a user story)
  / [0072](0072-sbv2-text-injection.md) decision 6 (which SBV2 checks are input-caused and which are
  internal invariants — this ADR only extends that line across families and does not move it) /
  [0083](0083-generation-api-surface.md) decision 10 (`GenerationCapacityError` carries the truncation
  values in fields) / ADR [0038](0038-manifest-v1.md) §4-era hub `HubError` (where the classification
  axis was borrowed from) / [0119](0119-wan-umt5-gpu-text-encoder.md) (`PromptCleanError`) /
  [limitations](../limitations.md) (the exception spellings documented as public behaviour) /
  [glossary](../glossary.md)

## Context

`@karume/models` ships the pipelines of eight families (at the time) in one package, and each family
writes its own checks of a generation request. Before the replacement, the **input-caused** throws (the
request the caller passed cannot be accepted = fixing the input makes it pass) numbered 127 and were split
across three types: plain `Error` (93), `RangeError` (19) and `Sbv2InputError` (15, SBV2 only — ADR 0072
decision 6).

In that shape, **a host that serves several families** (HTTP server, CLI, UI) has no way to tell 400 from
500. A plain `Error` is the same type as a broken internal invariant, an asset mismatch (manifest /
tokenizer / shard) or a lack of GPU capacity, so `instanceof` says nothing. What is left is **reading the
message string**, a coupling that breaks silently the moment the wording changes.

`Sbv2InputError` solves this for one family, but repeating that solution eight times makes the host learn
eight types — for one single branch.

Also, **the seed acceptance set** (non-negative safe integers) was held separately by anima / sbv2 /
irodori with the same condition. Copying one range to three places invites drift: one copy loosens, one
uses a falsy check.

## Decision

### 1. One type — `ModelInputError extends Error`

It lives in `packages/models/src/errors.ts`. `name` is `"ModelInputError"`, `cause` passes through
`ErrorOptions`, and it **has no additional fields**. The failures that need fields carry them on derived
types (decision 4); an empty field on the parent would leave the caller unable to tell when it is filled.

### 2. The classification axis is **where the caller branches**

The same approach as the hub's `HubError` and SBV2's `Sbv2InputError`. This type is thrown only when
"**the request itself cannot be accepted = fixing the input makes it pass**" — a 400 for an HTTP server.
The following are not this type (they are 500s — fixing the input does not help):

- broken internal invariants (`sum(word2ph)` mismatch, a broken tile scan)
- asset mismatches (manifest / tokenizer / shard disagree with what was declared)
- lack of GPU capacity

They stay plain `Error` or their existing types.

### 3. Scope: checks of the **range, type and combination of a generation request** only

Two things look input-caused but are not mixed into this type, because the caller's remedy differs:

- **misspelled model / quant / sampler names** (the remedy is "look up the accepted set again", not "fix
  the value")
- **call-order violations** (reusing after `dispose`, double generation — the remedy is "fix the call
  order")

Both stay plain `Error`. **Widening** the scope later is compatible for callers; narrowing it breaks
callers that already branch on it, so it is a breaking change (the asymmetry in Consequences).

### 4. Derived types only where they carry more information

A derived type is created only when the failure carries **more information than `instanceof`** (a
family-specific line or structured fields); a branch that adds no information is not split into a type, so
**no per-family types**. The derived types today:

- `Sbv2InputError extends ModelInputError` (`name` stays `"Sbv2InputError"`; the line of ADR 0072
  decision 6 is kept as specified)
- `GenerationCapacityError extends ModelInputError` (carries the values needed for truncation in fields —
  ADR 0083 decision 10)
- `PromptCleanError extends ModelInputError` (carries the rejection `reason` in a field — ADR 0119,
  `packages/models/src/wan/text/prompt-clean.ts`)

(Restated 2026-10-18: the original text said "only two derived types"; ADR 0119 added the third on the
same criterion, so the criterion is the rule.)

### 5. Input-caused `RangeError`s also become `ModelInputError`

The targets are the seed range, sampler settings and image / audio dimensions (21 throw sites then). This
changes public behaviour, so it is listed under Breaking in the CHANGELOG (code that branched on
`instanceof RangeError` has to follow).

### 6. The seed acceptance set has one owner

`assertAcceptableSeed(seed: number): void` in `packages/models/src/request-gates.ts` (throws
`ModelInputError`). The accepted set is the non-negative safe integers, shared by every family that takes
a seed. The check runs **at the entry of generation**: the random generator is created after the weights
are on the GPU, so a check only in the generator's constructor would make a typo fail after a
gigabyte-scale load.

## Alternatives considered

1. **Use the standard types (`RangeError` / `TypeError`)** (no new type) — rejected. They are not a
   discriminator. `packages/models/src/` had 54 `RangeError`s before the replacement, and the 33 left
   afterwards are **preconditions of internal helpers** (assumptions of the wiring, not the caller's input),
   so 500-class failures would share the type. Carving out 400s with `instanceof RangeError` would tell the
   caller to "fix the input" for a broken wiring. ADR 0072 decision 6 puts `RangeError` outside the
   classification axis for this reason, and there is no ground to reverse it here.
2. **One dedicated type per family, eight of them** (spreading the shape of `Sbv2InputError`) — rejected.
   It does not remove the reason for this ADR: a host serving several families would import eight types
   and write `instanceof` eight times, or read messages after all. The branch is the single "fix the
   input", so more types do not add branches for the caller. The one family that needs a family-specific
   line (SBV2) is covered by a derived type (decision 4).

## Consequences

- **The public surface gains one type** (the same type from the barrel and from every pipeline subpath —
  the public-surface fixture diff is the `ModelInputError` line only). A consumer that uses a single family
  can take the type from its subpath.
- **The change from `RangeError` is breaking.** Code that wrote `instanceof RangeError` for the seed /
  sampler range / image and audio dimensions switches to `ModelInputError`. `ModelInputError` derives from
  `Error`, so code with a bare `catch (e) { ... }` is unaffected.
- **Widening the scope only adds** (moving a check that throws a plain `Error` today into this type turns
  a 500 into a 400 for callers — a new branch, no broken one). **Narrowing is breaking**, so when in doubt,
  leaving a check out is the cheaper side.
- `GenerationCapacityError` only gained a parent; its `name`, fields and constructor are unchanged, and
  code branching on `instanceof GenerationCapacityError` is unaffected.
- The replacement in the families was completed on 2026-09-22 (95 `new ModelInputError` sites then, 133 on
  2026-10-18). Messages were kept and only types changed, except two whose wording moved: anima's `steps`
  (its check moved to the owner `sigmaSchedule`) and gemma's `capacity` diagnostic in
  `estimateSessionMemory` (split into an integer check and an ordering check, matching
  `createGenerationSequence` / `Gemma4ChatSession`).
- Three acceptance sets written in two places each now have **one owner**: seed (four copies →
  `request-gates.ts`), anima's `steps` (entry and `sigmaSchedule`) and sbv2's `styleWeight` (entry and table
  lookup). The entries call the owner and hold no condition themselves. Only sbv2's entry re-wraps the seed
  check in `Sbv2InputError` (so as not to break `instanceof` for hosts that serve SBV2 alone — a derived
  type of decision 4, so it also fits the cross-family branch).

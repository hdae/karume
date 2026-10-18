# 0075: Display name and description of a quant preset — fields for an application's selection UI

- Status: implemented (2026-08-25, the 0.5.0 breaking release, together with the preset renames of
  [0074](0074-quant-seat-naming.md)); partly superseded by [0109](0109-manifest-v5-container.md)
  (decision 4 and its consequence: the manifest format is `karume/5` since 2026-09-22, marked
  2026-10-18). Filed on 2026-08-21 from a proposal of the maintainer
- Date: 2026-08-21
- Related: [0038](0038-manifest-v1.md) (format bump rule, session allowlist) /
  [0041](0041-manifest-v2.md) (the `quants` vocabulary) / [0071](0071-manifest-v3-shards.md)
  (`karume/3`) / [0074](0074-quant-seat-naming.md) (preset naming — a separate role from this ADR)

## Context

A quant preset had only a machine name (`w8a8-s16`, `w8-bert4`). The name folds the storage and the
knobs into one identifier. An application with a selection UI (a demo or a tool that lets the user
pick the quantization) cannot show it as is, and the naming rule of [0074](0074-quant-seat-naming.md)
makes machine names longer still (`i8-a8-attn8-s16`).

The recipe author knows what a preset is good for, but that knowledge lived only in the model card
prose, a README that an application cannot read.

Compatibility constraint: the manifest parser has a strict allowlist for quant entries and fails
loudly on an unknown key (`QUANT_KEYS`, checked by `assertAllowedKeys` in
`packages/hub/src/manifest.ts`; the list then held `weights`, `session` and `gpuFeatures`). Adding a
field is therefore not backward compatible even when it is optional: an old client cannot read the
new manifest. This agrees with ADR [0038](0038-manifest-v1.md) decision 1, which rejects unknown
keys in the envelope and allows additive change only inside `pipelineConfig` and in the fields its
§7 lists ("an addition not listed here is `karume/2`"). The display fields are not in that list, so
a major bump is the procedure 0038 prescribed from the start.

A note of 2026-08-21 found that two fields §7 listed as addable (file reference `repo` / `revision`,
preset `requiredLimits`) were missing from the allowlists too; fixed in the same release (ADR 0038
addendum 2026-08-25).

## Decision

### 1. Add optional `label` and `description` to a quant entry

```jsonc
"i8-a8-attn8-s16": {
  "weights": { /* … */ },
  "session": { /* … */ },
  "label": "Balanced (int8)",
  "description": "Half the download of f16 with no visible difference."
}
```

- `label`: a short display name for the choice (at most 64 characters).
- `description`: a one-line explanation (at most 200 characters).
- Both are English, like the distribution README and the model card. There is no multilingual
  field; an application can key its own translation on the preset id.
- Both are optional; without them the caller shows the id.

### 2. The hub checks the limits and does not interpret the meaning

The length limits are a boundary check, because a manifest is external input. Whether the text
matches the preset cannot be checked and is not checked; display is the caller's responsibility.

### 3. No `recommended` field and no ordering field

- `defaultQuant` already names the recommended preset.
- An application can order presets roughly by download size. Download size is not a total order:
  presets that differ only in compute knobs (`linearCompute` / `attentionCompute`) point at the same
  weights and tie (anima's seven presets fall into three sizes). Within a tie the primary key is
  the declaration order of `quants` (JSON insertion order).
- The download size is not the sum of the files resolved from `weights`: the resolved selection also
  holds the `assets` (independent of the preset), and several keys can point at the same file.
  Count it with the deduplication the fetch layer uses — by file identity, not by path alone —
  which is `selectionRefs(resolveSelection(…))` in `packages/hub/src/resolve.ts`.

These fields are left out because derivable state in an independently updated field has no source
of truth once the two disagree (CLAUDE.md: no derivable state in denormalized fields).

### 4. Bump the format to `karume/4` (superseded by ADR 0109 on 2026-09-22: the format is `karume/5`)

Because of the allowlist, a manifest with the new fields cannot be read by an old client. The break
is declared with a format major instead of being left silent; this rides on ADR 0041 §1 ("an unknown
major fails loudly"). The bump went into the 0.5.0 breaking release, so the four public repositories
were re-uploaded and the pins ([0073](0073-models-source-pin.md)) updated once.

### 5. The model card's quant table prints the same strings

Wording grown separately in the manifest and the card goes stale on one side. Card generation
(`karume.modelcard`) already MUST print only values from the manifest; this rides on that rule.

The English prose a recipe already has (modification notices such as `*_NOTICE_MARKDOWN`) is a
separate responsibility and is neither merged nor cross-referenced. A notice is one fixed text about
what was done to the weights of the repository; `description` says per preset what that preset is.
Preset descriptions in the notice would disagree with the content whenever presets are added or
removed (for the same reason the notice does not depend on the list of models).

## Consequences

- The `karume/4` bump shipped only together with the renames of [0074](0074-quant-seat-naming.md),
  so the format broke once (superseded by ADR 0109 on 2026-09-22: the format is `karume/5`).
- The hub adds two keys to `QUANT_KEYS` and the length check. Resolution is unchanged.
- A recipe writes two English lines per preset. A preset can omit them (optional), so the single
  `f32` preset of a small family stays as it is.
- The UI can absorb the cost of the longer names of [0074](0074-quant-seat-naming.md); the two ADRs
  make sense as a pair.

## Addendum (2026-10-11) — the hub parser defines the accepted set of the display fields

- Accepted: the field is absent, or a string within the limit (counted in code points). `null` and
  non-strings are rejected. An empty string, whitespace only and values with line breaks are
  accepted (decision 2: the meaning is not interpreted; "one line" in decision 1 is guidance for the
  author, not a check).
- `karume dist` (Python) follows this set. Both sides run the shared table
  `packages/hub/tests/fixtures/quant-presentation.json` to stay in agreement.
- The model card's quant table folds every run of whitespace (line breaks included) into one space
  so the value fits one cell, and writes `—` for a field that holds only whitespace, as for an
  absent field. Whitespace is what `str.split()` without arguments splits on (characters for which
  `str.isspace()` is true): Unicode whitespace as well as ASCII spaces and line breaks, so NBSP
  (U+00A0), the ideographic space (U+3000) and the line separator (U+2028) also fold; the zero-width
  space (U+200B) is not whitespace. `|` would split the table column and is escaped as `\|`. These
  two transformations are the only departures from "the same strings" of decision 5.

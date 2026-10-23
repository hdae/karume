# 0074: Preset naming — one storage vocabulary plus per-axis knobs

- Status: implemented (2026-08-25, 0.5.0 breaking wave — every preset renamed by decision 6, the
  `linearCompute` / `attentionCompute` value `"i8a8"` renamed `"a8"`, published together with
  [0075](0075-quant-presentation.md); proposed on 2026-08-21 by the maintainer's ruling). Still in
  force; [0128](0128-default-preset-original-weights.md) keeps decision 5.
- Date: 2026-08-21
- Related: ADR [0041](0041-manifest-v2.md) (the `presets` → `quants` vocabulary) /
  [0025](0025-w8a8-linear-execution.md) and [0030](0030-attention-a8-execution.md) (origin of
  `i8a8`) / [0069](0069-packed-w4-storage.md) (i4 storage) / [0075](0075-quant-presentation.md)
  (display label and description) / [0073](0073-models-source-pin.md) (pins)

## Context

Preset names grew independently per family and had four flaws on 2026-08-21.

| Family                                      | Presets                                                           | Default    |
| ------------------------------------------- | ----------------------------------------------------------------- | ---------- |
| anima                                       | `f16` `i8` `w8a8` `w8a8-a8` `w8a8-s16` `f16-c16` `w4` `w4-a8-s16` | `w8a8-s16` |
| sbv2                                        | `f16` `w8` `w8a8` `w8-bert4` `w4`                                 | `w8-bert4` |
| irodori                                     | `f32` `f16` `w8` `w8a8`                                           | `w8a8`     |
| birefnet / depth-anything / siglip2 / vowel | `f32`                                                             | `f32`      |

1. **Two spellings for one thing** — "i8 storage, plain compute" was `i8` in anima and `w8` in sbv2 /
   irodori.
2. **One spelling for two axes** — the `a8` of `w8a8` is the linear activation; the second `-a8` of
   `w8a8-a8` is the attention activation. `w4-a8-s16`, added on 2026-08-21, inherited the ambiguity.
3. **Names that contradict the content** — sbv2's `f16` preset was not pure f16: only `text_encoder`
   was i8 (`sbv2/distribution.py`).
4. **Axes with different grammar mixed** — `w8-bert4` is a per-component override, built differently
   from the other names.

The hub treats a preset name **only as an opaque key** (`resolveSelection` in
`packages/hub/src/resolve.ts` checks `Object.hasOwn(entry.quants, quant)`), so renaming changes data
and documents, not parsers or the runtime.

## Decision

### 1. Grammar

    <storage>[+<component><bits>]…[-<knob>]…

- `<storage>` = the **base** storage common to every role
- `+<component><bits>` = per-role overrides (zero or more)
- `-<knob>` = runtime knobs (zero or more, in the order defined below)

### 2. The storage vocabulary matches the asset's storage one-to-one (`f32` / `f16` / `i8` / `i4`)

The second vocabulary `w8` / `w4` is dropped. The dtype label in a preset's `weights` declaration,
the storage that `STORAGE_REQUIREMENTS` demands and the preset name share one spelling, so the
question "which one is right" never arises. `i4` labels mixed storage (eligible weights in i4, the
rest in i8), the meaning the anima and sbv2 presets already used. The vocabulary has grown since:
`i2` (addendum of 2026-09-19) and `bf16` (ADR [0131](0131-bf16-weight-execution.md), 2026-10-17 — the
anima preset `bf16`, a bare storage name); the full list and each type's state are indexed in
[quantization](../quantization.md) table 1. Since ADR 0108 the "asset's storage" is the container's
codec (addendum of 2026-09-25).

### 3. Knob spellings — `a8` / `attn8` / `s16` / `c16`

| Token   | Axis                                | Session value                               |
| ------- | ----------------------------------- | ------------------------------------------- |
| `a8`    | linear activations to i8            | `linearCompute: "a8"`                       |
| `attn8` | attention activations to i8         | `attentionCompute: "a8"`                    |
| `s16`   | attention score storage to f16      | `attentionScoreStorage: "f16"`              |
| `c16`   | compute in f16 (requires shaderF16) | `linearCompute` / `attentionCompute: "f16"` |

**The value vocabulary (`a8`) and the name tokens (`a8` / `attn8`) are separate layers.** The option
key already names the axis, so the value carries none (`attentionCompute: "a8"`). A preset name folds
the axes into one string, so the token tells which activation.

`linearCompute`'s value becomes `a8` instead of `i8a8` because the knob decides **only how activations
are handled** (the asset decides the weight storage). The `i8` of the old spelling folded the weight
storage discriminator `:wi8` ([dp4a-w8a8-design](../research/2026-08-03-dp4a-w8a8-design.md)), which
becomes false once w4a8 puts i4-resident weights on the same path. **Writing the storage into the
value would keep it twice, in the asset and in the manifest** (the rule against maintaining derivable
state in an independently updated field), so adding values per storage (`i4a8`) is not adopted — it
would also accelerate only one side of a mixed asset (anima's i4 series has 453 i4 tensors and one
i8).

### 4. Component abbreviations are defined by the recipe and shown on the card

An abbreviation such as the `bert` of `i8+bert4` is shorter than the role name (`text_encoder`) but
becomes vocabulary itself. **The recipe (`distribution.py`) defines the abbreviations, and the
generated model card's quant table always shows the mapping** (the `abbreviations` argument of
`quants` in `tools/exporter/src/karume/modelcard.py`). With [0075](0075-quant-presentation.md)'s
label and description, a UI need not show abbreviations at all.

### 5. The default is not expressed in the name

Preset names do not contain "recommended" or "default": `defaultQuant` already points at the default,
and a name would keep it twice (for the same reason [0075](0075-quant-presentation.md) has no
`recommended` field).

### 6. Migration (all at once in 0.5.0)

Applied on 2026-08-25 in the 0.5.0 breaking wave, with no compatibility shim. `<storage>` is taken
literally as the base common to every role: anima keeps text_encoder / text_conditioner /
vae_decoder in f16 and compresses only the transformer, so its base is `f16` with `+dit<bits>`
(for example `w8a8-s16` → `f16+dit8-a8-attn8-s16`; `w4-a8-s16` → `f16+dit4-attn8-s16`, whose old
`-a8` meant attention); sbv2 / irodori use one base word (`w8-bert4` → `i8+bert4`, sbv2 `f16` →
`f16+bert8`, irodori `w4` → `i8+dit4`). Names grow longer (`i8-a8-attn8-s16`), but **the id serves
the machine; people read [0075](0075-quant-presentation.md)'s `label` / `description`.**

## Consequences

- The manifests of every published repository changed, so they were re-uploaded and their pins
  updated ([0073](0073-models-source-pin.md)), together with the other 0.5.0 breaking changes.
- The hub and runtime need no change for preset names (opaque keys). Renaming the
  `linearCompute` / `attentionCompute` value touches the runtime types (`ComputePrecision` /
  `SessionOptions`) and the hub's `LINEAR_COMPUTE` / `ATTENTION_COMPUTE` allowlists (type checking
  finds every reference).
- **No compatibility shim** for old names. A breaking switch at once misleads less than a period where
  both spellings work; 0.4.0 clients pinned to old names keep reading the old revision, so the pin
  absorbs the break.
- Small families (`f32` only) are unchanged.

## Addendum (2026-08-24) — presets added in wave J-4

irodori gained a `w4` preset (ADR [0050](0050-irodori-quant-series.md) addendum 2); its 0.5.0 name is
`i8+dit4` (base i8 for all seven roles, the DiT alone mixed i4). Applied with decision 6.

## Addendum (2026-09-13) — parallel GEMV reduction

The knob `gemvpar` is added at the end of the knob order and means `session.linearGemvReduce:
"parallel"`. `i4-gemvpar` has the same weight mapping as `i4`; only the reduction differs
([ADR 0098](0098-linear-gemv-parallel.md)). The default is still chosen by `defaultQuant`, not by
name.

## Addendum (2026-09-19) — the `fast` token, `i2` in the storage vocabulary, ADRs that extend this rule

### The `fast` token

The knob spelling `fast` is added. It is not one knob but **the bundle of speed-up knobs the family
has accepted**, defined by each `distribution.py` quant table (for plain Gemma: parallel GEMV plus
RMS→add fusion; QAT adds linear→SRQ fusion — [ADR 0104](0104-gemma-fast-quant.md)). The bundle may
differ per family and cannot be read from the name; [0075](0075-quant-presentation.md)'s `label` /
`description` explain it.

### `i2` joins the storage vocabulary of decision 2

INT2 storage (ADR [0097](0097-gemma4-qat-integration.md) addendum 1)
makes **`i2` a further word**. Decision 2's rule (preset name, `weights` dtype label and the storage
`STORAGE_REQUIREMENTS` demands share one spelling) is unchanged.

### ADRs that add to or make exceptions to this rule

| ADR                                               | What it adds                                                                                                           |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [0098](0098-linear-gemv-parallel.md)              | knob `gemvpar` (addendum of 2026-09-13)                                                                                |
| [0097](0097-gemma4-qat-integration.md) addendum 5 | QAT preset names — mixed INT2 / INT4 / INT8 under the single label `i4`; the breakdown goes in `label` / `description` |
| [0104](0104-gemma-fast-quant.md)                  | knob `fast` (above); existing assets are not updated automatically                                                     |
| [0131](0131-bf16-weight-execution.md)             | storage word and preset name `bf16` (2026-10-17)                                                                       |

## Addendum (2026-09-25) — the canonical storage vocabulary after the container

Decision 2's "asset's storage" means, since ADR [0108](0108-container-format.md), the container
descriptor's `encoding.codec` (codec ledger — 0108 decisions 12 / 13, [container-v1](../container-v1.md)
§6). The dtype label and the preset name match the layout derived from the codec (`codecLayout`), and
`karume dist` still checks that the three spellings agree (`storage_dtypes` in
`tools/exporter/src/karume/dist.py`). Manifest `karume/5` keeps the same field (`quants[].weights`
dtype label — ADR [0109](0109-manifest-v5-container.md) decision 2).

## Addendum (2026-10-23) — the `online` token (ADR 0132)

| Token    | Axis                                           | Session value                   |
| -------- | ---------------------------------------------- | ------------------------------- |
| `online` | attention scores never stored (online kernel) | `attentionScoreStorage: "none"` |

It takes the slot of `s16` (both are the score-storage axis): `f16+dit8-a8-attn8-online`, the Wan2.2
default since 2026-10-23.

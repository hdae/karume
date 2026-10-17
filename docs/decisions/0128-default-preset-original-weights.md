# 0128: How a distribution chooses its default preset

- Status: proposed (2026-10-16; rewritten 2026-10-17 after the maintainer's rulings of the same day;
  to be accepted by the maintainer after review)
- Date: 2026-10-16 (rewritten 2026-10-17)
- Related: ADR [0074](0074-quant-seat-naming.md) (preset naming — the name never says "default";
  `defaultQuant` points at it) / [0075](0075-quant-presentation.md) (`label` / `description`;
  decision 3 keeps "recommended" out of the manifest, which this ADR keeps) /
  [0058](0058-numerics-opt-in-contract.md) (its 2026-09-13 addendum lets a distribution put a
  speed-bundled preset into `defaultQuant` beside a reference preset with the same weights — the
  form decision 1 keeps) / [0110](0110-practical-tier-numerics-contract.md) (decision 7 classifies
  sha rows by the preset's `session`; decision 8 ties the human quality review to entering
  `defaultQuant` — kept as part of decision 1's procedure) / [0111](0111-session-options-composition.md)
  (explicit option > preset declaration > runtime default; unchanged) /
  [0113](0113-birefnet-weight-series.md) (2026-10-11 addendum: the BiRefNet default is the preset
  whose storage equals the upstream checkpoint's; an instance of decision 1 and the origin of
  decision 4's "f16 only for an f16 upstream") / [0115](0115-geometry-profiles.md) (Context: on Apple
  Metal the i8a8 kernels gain nothing without dp4a; Consequences deferred "drop a8 from the default
  on Metal" into this question — closed by decision 5) / [0120](0120-wan-dit-w8a8-seat.md)
  (2026-10-04 ruling 4: "the most practical preset is the default" — an instance of decision 1) /
  [0121](0121-wan22-ti2v-5b.md) (decision 2 and the 2026-10-06 ruling: the Wan2.2 default is the
  practical preset, and Wan2.2 has no f16 preset — an instance of decisions 1 and 4) /
  [0109](0109-manifest-v5-container.md) (`karume/5`; the `defaultQuant` field is unchanged) /
  [0106](0106-device-keyed-references.md) (sha256 reference rows per environment key) /
  [0126](0126-numerics-profiles.md) (numerics profiles; activation quantization stays out of them —
  decision 7) / [0131](0131-bf16-weight-execution.md) (bf16 storage; Anima's original-weights preset
  `bf16`) / [0073](0073-models-source-pin.md) (revision pins).

## Context

### How the default is resolved

`defaultQuant` is a required string in every model entry of a distribution manifest; the parser
only checks that it names one of the entry's `quants` (`packages/hub/src/manifest.ts:1147-1162`).
The hub treats it as an opaque name: `resolveSelection` uses `options.quant ?? entry.defaultQuant`
(`packages/hub/src/resolve.ts:124`, same in `inventory.ts:151`). Every family pipeline in
`@karume/models` passes the caller's `quant` through to that resolution. The default is therefore
pure data, chosen by one constant per recipe (`ANIMA_DEFAULT_QUANT`, `SBV2_DEFAULT_QUANT`,
`IRODORI_DEFAULT_QUANT`, `WAN_DEFAULT_QUANT`, `WAN22_DEFAULT_QUANT`, `UMT5_DEFAULT_QUANT`,
`GEMMA4_DEFAULT_QUANT`, `QAT_DEFAULT_QUANT`, …) and, for BiRefNet only, derived from the upstream
storage type (`birefnet_default_quant` in `tools/export-recipes/birefnet/distribution.py`).

### What the default is today

Local distribution mirrors on 2026-10-17 (`models/*/karume.json`), with the storage type of the
upstream checkpoint read from its safetensors header (weight tensors only):

| Distribution (models)                      | Presets                                                                                                       | `defaultQuant`          | Upstream storage                                                                    | Preset holding the original values                                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| anima, anima-extra (7)                     | `f16` `f16+dit8` `f16+dit8-a8` `f16+dit8-a8-attn8` `f16+dit8-a8-attn8-s16` `f16-c16` `bf16`                                              | `f16+dit8-a8-attn8-s16` | BF16 in all four components                                                         | `bf16` (ADR 0131; all seven models since 2026-10-17)                                                                                |
| sbv2-jvnv (4)                              | `f16+bert8` `i8` `i8-a8` `i8+bert4` `i4`                                                                      | `i8+bert4`              | F32                                                                                 | none; an f32 preset was not shipped ("1.32 GB impractical", `sbv2/distribution.py:11`) and `f16+bert8` keeps the text encoder in i8 |
| irodori v4 / v4.1-small                    | `f32` `f16` `i8` `i8-a8` `i8+dit4`                                                                            | `i8-a8`                 | F32                                                                                 | `f32`                                                                                                                               |
| wan2.1 t2v-1.3b                            | `f16` `f16+dit8` `f16+dit8-a8-attn8-s16`                                                                      | `f16+dit8-a8-attn8-s16` | DiT F32                                                                             | none; `f16` is the widest (its text encoder is the cross-referenced umT5 i8)                                                        |
| wan2.2 ti2v-5b                             | `f16+dit8` `f16+dit8-a8-attn8-s16`                                                                            | `f16+dit8-a8-attn8-s16` | DiT F32                                                                             | none; ADR 0121 decision 2 ships no f16 preset (the f16 DiT is 9.31 GiB)                                                             |
| umt5-xxl                                   | `i8`                                                                                                          | `i8`                    | F32 (about 22.7 GB, `wan/text_embeds.py`)                                           | none                                                                                                                                |
| gemma4 e2b                                 | `i4` `i4-gemvpar` `i4-fast`                                                                                   | `i4-fast`               | BF16                                                                                | none                                                                                                                                |
| gemma4-qat e2b / e4b                       | `i4` `i4-gemvpar` `i4-fast`                                                                                   | `i4-fast`               | mixed F32 / BF16 / I8 / U8 (the upstream is itself a quantization-aware checkpoint) | `i4`: the upstream's own fixed integer weights, carried without re-quantization (ADR 0097)                                          |
| birefnet-hr 1024 / 2048                    | `f16`                                                                                                         | `f16`                   | F16                                                                                 | `f16`                                                                                                                               |
| lucida 1024 / 2048                         | `f32` `f16`                                                                                                   | `f32`                   | F32                                                                                 | `f32`                                                                                                                               |
| depth-anything-v2, siglip2, vowel-detector | `f32`                                                                                                         | `f32`                   | F32                                                                                 | `f32`                                                                                                                               |

Eleven models (anima ×7, irodori ×2, wan2.1, wan2.2) default to a preset that quantizes
activations (`linearCompute: "a8"`, `attentionCompute: "a8"`); every one of those presets has a
sibling preset with the same weight files and no activation quantization (`f16+dit8` for anima and
Wan, `i8` for irodori). No default requires a GPU feature (`shaderF16`).

### How the defaults came about

- sbv2: maintainer ruling 2026-08-20, "quality first" among the presets that were practical to
  ship; an f32 preset was ruled out by size.
- [0058](0058-numerics-opt-in-contract.md) addendum 2026-09-13: the runtime keeps reference
  numerics by default, and distributions put a preset with the speed bundle into `defaultQuant`
  while keeping a reference preset with the same weights.
- [0110](0110-practical-tier-numerics-contract.md) decision 8: machine checks show a preset is
  safe as an opt-in; a human quality review is required only when a preset enters `defaultQuant`.
- [0120](0120-wan-dit-w8a8-seat.md) ruling 2026-10-04 (4): the default is "the most practical
  preset right now" (light, small quality loss). The ruling kept open a later switch to "the
  preset closest to the original weights". Wan2.1 and then Wan2.2 (ADR 0121, 2026-10-06) were
  switched to the practical preset under that policy.
- The backlog item "要検討: 既定の quant 席を量子化にするか opt-in（元の重み）にするか"
  (opened 2026-09-27) collected the question. The "drop a8 from the default on Metal" decision
  ([0115](0115-geometry-profiles.md) Consequences, perf-ledger K-70) was deferred into it.

### The first draft, and why it was withdrawn

The first draft of this ADR (2026-10-16) ruled that `defaultQuant` MUST name the preset holding the
original weights, that quantization is selected explicitly, and that 0.15.0 announces this as a
breaking change. Its reasons were real and are kept here as facts: the speed case for int8
activations is not portable (on Apple Metal Tint expands `dot4I8Packed` into scalar code, so the
i8a8 kernels stay as slow as f32 — ADR 0115); low-bit storage is about memory, not speed; and a
default that silently rounds the model's weights is the failure shape a library should avoid.

The maintainer withdrew the draft on 2026-10-17 on these grounds:

- f32 / bf16 / f16 presets are often too large for browsers, the primary target. The practical
  presets that are the defaults today were measured and reviewed, and they are not a bad default.
- The two concerns behind the draft can be met without moving the default: (a) a lower-precision
  model being used without the user knowing — met by stating the policy and by the model card
  (decision 3); (b) adopting int8 activations on a Mac without dp4a — met by a warning in
  environments without dp4a (decision 5).
- What stays unsatisfactory is that "the default is decided per model" was practice, not a rule.
  This ADR writes the practice down as a procedure with mechanical constraints.
- An original-weights preset is wanted where it is feasible (bf16 storage computed in f32 was added
  for it — ADR 0131; f16 counts as original only when the upstream itself is f16), but it is best
  effort: no size cap decides it, and a family whose export is impractical on the maintainer's
  machine may skip it.
- A device without dp4a gets a warning, not a fallback and not an automatic switch to the sibling
  preset: an automatic quality increase is as surprising as an automatic decrease, and the numbers
  are the same either way.

## Decision

1. **The default is chosen per model by a recorded procedure, and the current defaults stand.** For
   every model the maintainer picks `defaultQuant` among the shipped presets from measurements on
   the reference devices — speed, resident memory, download size and output quality; the human
   quality review of ADR 0110 decision 8 remains the gate for entering `defaultQuant` — and records
   the candidates, the measurements and the reason in the decision record (the family's ADR, or the
   table in Context for families without one). The default is normally the practical preset: the
   lightest preset whose quality loss the review accepted. The rulings for sbv2 (2026-08-20),
   Wan2.1 (ADR 0120 ruling 4, 2026-10-04), Wan2.2 (ADR 0121 decision 2, 2026-10-06), gemma4 (ADR
   0104) and BiRefNet (ADR 0113 addendum 2026-10-11) are instances of this procedure, and the form
   of ADR 0058's addendum (a speed-bundled default beside a reference preset with the same weights)
   is what the procedure produces. Rationale: the measured facts differ per model and per device
   class, so no formula yields the default; what can be a rule is that the choice is measured,
   reviewed and written down.
2. **Three constraints hold for every distribution and are checked mechanically from the manifest
   alone**, on every local mirror, by the distribution gate
   (`packages/runtime/tests/distribution_gate_test.ts`):
   - **C1 — the default runs on baseline WebGPU.** The default preset declares no `gpuFeatures`
     (no `shaderF16` requirement). Rationale: a default that needs a feature fails on devices the
     user did not get to choose for.
   - **C2 — every preset that quantizes activations has a sibling with the same weights and no
     activation quantization.** For every preset whose `session` has `linearCompute: "a8"` or
     `attentionCompute: "a8"`, another preset of the same model has an identical `weights` map and
     neither value. Rationale: that sibling is what the warning of decision 5 points to; identical
     weights means switching to it costs no download.
   - **C3 — a model with two or more presets labels every preset.** Every preset has a `label` and
     a `description` (ADR 0075), non-blank after trimming; a model with exactly one preset is
     exempt. Rationale: the choice has to be readable where it is offered, and the card prints
     these strings.
     What the manifest cannot show is recorded instead of checked (decision 4's table): whether an
     original-weights preset ships, and that f16 counts as original only for an f16 upstream (the
     upstream storage type is not in the manifest).
3. **Transparency.** The model card marks the default and lists every preset's storage per
   component and its compute knobs (`karume.modelcard`, exists); the repository README states this
   policy ("Presets and the default"); the runtime warns when activation quantization runs without
   dp4a (decision 5). Examples and gpu-lab print the resolved preset name when they are rebuilt as
   browser applications (ADR 0127). Rationale: concern (a) of the withdrawn draft — a user who did
   not choose can see what they got, where they got it.
4. **An original-weights preset is best effort, opt-in, and defined by the upstream storage.**
   "Original weights" means the upstream checkpoint's values with no rounding by Karume: f32 for an
   f32 upstream, bf16 for a bf16 upstream (ADR 0131), f16 only where the upstream itself stores f16
   (BiRefNet-HR — ADR 0113), and for a quantization-aware upstream its own fixed integer weights
   (gemma4-qat — ADR 0097). Compute is f32 (f16 compute is a separate knob, `-c16`). Each family
   SHOULD ship one; it MAY be skipped when exporting it is impractical on the maintainer's machine,
   and the gap is recorded here — no size cap decides it. It is never the default by rule (decision
   1 decides the default) and is selected explicitly through `quant`.

   | Family                                             | Original-weights preset | Status on 2026-10-17                                                   |
   | -------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
   | anima (all seven models)                           | `bf16`                  | shipped (ADR 0131; every checkpoint is BF16, no LoRA fusion)           |
   | irodori v4 / v4.1-small                            | `f32`                   | shipped                                                                |
   | birefnet-hr                                        | `f16`                   | shipped (upstream f16)                                                 |
   | lucida, depth-anything-v2, siglip2, vowel-detector | `f32`                   | shipped                                                                |
   | gemma4-qat                                         | `i4`                    | shipped (upstream fixed integers, not re-quantized)                    |
   | sbv2-jvnv                                          | —                       | skipped (f32 1.32 GB, 2026-08); may be added, no cap decides it        |
   | wan2.1 / wan2.2                                    | —                       | skipped (DiT f32 about 5.6 / 20 GB; the text encoder is umT5 i8)       |
   | umt5-xxl                                           | —                       | skipped (f32 about 22.7 GB)                                            |
   | gemma4 e2b                                         | —                       | pending (bf16 upstream; needs the PLE in bf16 — ADR 0131 open point)   |

5. **A device without dp4a gets a warning, not a fallback.** When a Session is built with
   `linearCompute: "a8"` or `attentionCompute: "a8"` on a device whose WGSL language features lack
   `packed_4x8_integer_dot_product`, `@karume/runtime` prints one `console.warn` per device
   (`GpuContext`) that says what happened — the integer dot products run emulated: same numbers,
   no speed-up, slower on Apple Metal — and the two ways out: the sibling preset of C2 (same
   weights, no activation quantization, no extra download), or `"f32"` for the knob. The runtime
   does not change the knob, does not pick the sibling and does not fail; the choice stays the
   user's. The predicate is a pure function (`packages/runtime/src/runtime/a8-emulation-warning.ts`),
   the once-per-device memo lives on the `GpuContext`. Rationale: the maintainer's ruling of
   2026-10-17; the emulated path is numerically identical (ADR 0025), so this is a speed fact the
   user needs to see, not a correctness event. To be revisited if field evidence shows the warning
   is ignored.
6. **No API change and no data change.** `defaultQuant` keeps its meaning and today's values; the
   `quant` option keeps its meaning; ADR 0075 decision 3 stands (no `recommended` field). No
   distribution is rebuilt for this ADR and 0.15.0 carries no breaking change on this account. The
   withdrawn draft's decisions — default = original weights, quantization explicit-only, the
   0.15.0 announcement, superseding the per-family rulings, ADR 0110 decision 8 losing its trigger
   — do not apply.
7. **Activation quantization stays a property of the preset, not of the numerics profile.** ADR
   0126's automatic selection covers kernel variants of classes E / C / R only; `a8` reaches a
   Session only through a preset's `session` declaration or an explicit option. ADR 0126 decision
   2's sentence "quantization stays an explicit choice in 0.15.0" is to be read as "the runtime
   never selects it automatically" — a default preset may declare it; the wording is adjusted when
   ADR 0126 is implemented (phase 2 step 2-8).

## Consequences

### For users

- `fromPretrained(...)` without `quant` behaves as in 0.14.x.
- On a GPU without dp4a, a default that quantizes activations (anima, irodori, wan2.1, wan2.2
  today) prints the warning once when the first Session is built; passing the sibling preset
  (`f16+dit8` for anima and Wan, `i8` for irodori) removes it with no extra download, and so does
  `linearCompute: "f32"` / `attentionCompute: "f32"`.
- The original-weights preset, where it exists, is chosen by name (`quant: "bf16"` for
  `anima-v1.0`, `"f32"` for irodori, …).

### For the repository

- The distribution gate asserts C1–C3 on every mirror under `models/`, so a recipe that adds an
  activation-quantized preset without its sibling, a default that requires `shaderF16`, or an
  unlabeled preset turns the gate red before anything is published.
- The table of decision 4 is maintained here; a family that gains or drops an original-weights
  preset updates its row.
- CHANGELOG: the runtime warning under Added; no Breaking entry.

### For verification

- Tests name their preset where the value matters (unchanged practice); the sha rows of ADR 0106
  are unaffected because no default moved.
- The warning is covered by unit tests of the predicate and of the once-per-device memo (no GPU),
  and by a GPU test that builds an a8 Session through a `GpuContext` whose language-feature set is
  emptied — the real device runs the emulated kernels and exactly one warning is observed across
  three Sessions (`packages/runtime/tests/gpu_i8a8_test.ts`).
- The constraints are covered by fault-injection tests on fabricated manifests next to the
  per-mirror checks.

## Staging

Implemented in phase 2 step 2-7 of the 0.15 plan (2026-10-17): the runtime warning with its tests,
the gate constraints with their fault-injection tests, the README section and the documentation
sync. No distribution was rebuilt.

## Open points

1. gemma4 `bf16` (PLE); whether to add sbv2 `f32`. Each closes a row of decision 4's table.
2. Revisiting decision 5 (warning versus automatic sibling) once there is field evidence.
3. The memory pre-check's "which presets fit this device" report (building blocks exist:
   `estimateGraphMemory`, `fitsHeadroom`); not scheduled.
4. The wording of ADR 0126 decision 2 (decision 7), at step 2-8.

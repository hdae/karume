# 0128: The default quantization preset is the original weights

- Status: proposed (2026-10-16; to be accepted by the maintainer after review)
- Date: 2026-10-16
- Related: ADR [0074](0074-quant-seat-naming.md) (preset naming — the name never says "default";
  `defaultQuant` points at it) / [0075](0075-quant-presentation.md) (`label` / `description`;
  decision 3 keeps "recommended" out of the manifest, which this ADR keeps) /
  [0058](0058-numerics-opt-in-contract.md) (its 2026-09-13 addendum let distributions put a
  speed-bundled preset into `defaultQuant`; superseded here) /
  [0110](0110-practical-tier-numerics-contract.md) (decision 7 classifies sha rows by the preset's
  `session`; decision 8 ties the human quality review to entering `defaultQuant`; amended here) /
  [0111](0111-session-options-composition.md) (explicit option > preset declaration > runtime
  default; unchanged) / [0113](0113-birefnet-weight-series.md) (2026-10-11 addendum: the BiRefNet
  default is the preset whose storage equals the upstream checkpoint's; the precedent this ADR
  generalises) / [0115](0115-geometry-profiles.md) (Consequences: "drop a8 from the default on
  Metal" was deferred into this question) / [0120](0120-wan-dit-w8a8-seat.md) (2026-10-04 ruling 4:
  "the most practical preset is the default"; superseded here) /
  [0121](0121-wan22-ti2v-5b.md) (decision 2 and the 2026-10-06 ruling: the Wan2.2 default is the
  practical preset, and Wan2.2 has no f16 preset) / [0109](0109-manifest-v5-container.md)
  (`karume/5`; the `defaultQuant` field itself is unchanged) / [0106](0106-device-keyed-references.md)
  (sha256 reference rows per environment key) / [0126](0126-numerics-profiles.md) (numerics
  profiles; activation quantization stays out of them) / [0073](0073-models-source-pin.md) (revision pins;
  re-upload means re-pinning).

## Context

### How the default is resolved

`defaultQuant` is a required string in every model entry of a distribution manifest; the parser
only checks that it names one of the entry's `quants` (`packages/hub/src/manifest.ts:1159-1174`).
The hub treats it as an opaque name: `resolveSelection` uses `options.quant ?? entry.defaultQuant`
(`packages/hub/src/resolve.ts:124`, same in `inventory.ts:151`). Every family pipeline in
`@karume/models` passes the caller's `quant` through to that resolution. The default is therefore
pure data, chosen by one constant per recipe (`ANIMA_DEFAULT_QUANT`, `SBV2_DEFAULT_QUANT`,
`IRODORI_DEFAULT_QUANT`, `WAN_DEFAULT_QUANT`, `WAN22_DEFAULT_QUANT`, `UMT5_DEFAULT_QUANT`,
`GEMMA4_DEFAULT_QUANT`, `QAT_DEFAULT_QUANT`, …) and, for BiRefNet only, derived from the upstream
storage type (`birefnet_default_quant` in `tools/export-recipes/birefnet/distribution.py`).

### What the default is today

Local distribution mirrors on 2026-10-16 (`models/*/karume.json`), with the storage type of the
upstream checkpoint read from its safetensors header (weight tensors only):

| Distribution (models)                      | Presets                                                                              | `defaultQuant` today    | Upstream storage                                                                    | Preset holding the original values                                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------ | ----------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| anima, anima-extra (7)                     | `f16` `f16+dit8` `f16+dit8-a8` `f16+dit8-a8-attn8` `f16+dit8-a8-attn8-s16` `f16-c16` | `f16+dit8-a8-attn8-s16` | DiT BF16                                                                            | none exactly; `f16` is the widest                                                                                                   |
| sbv2-jvnv (4)                              | `f16+bert8` `i8` `i8-a8` `i8+bert4` `i4`                                             | `i8+bert4`              | F32                                                                                 | none; an f32 preset was not shipped ("1.32 GB impractical", `sbv2/distribution.py:11`) and `f16+bert8` keeps the text encoder in i8 |
| irodori v4 / v4.1-small                    | `f32` `f16` `i8` `i8-a8` `i8+dit4`                                                   | `i8-a8`                 | F32                                                                                 | `f32`                                                                                                                               |
| wan2.1 t2v-1.3b                            | `f16` `f16+dit8` `f16+dit8-a8-attn8-s16`                                             | `f16+dit8-a8-attn8-s16` | DiT F32                                                                             | none exactly; `f16` is the widest (its text encoder is the cross-referenced umT5 i8)                                                |
| wan2.2 ti2v-5b                             | `f16+dit8` `f16+dit8-a8-attn8-s16`                                                   | `f16+dit8-a8-attn8-s16` | DiT F32                                                                             | none; ADR 0121 decision 2 ships no f16 preset (the f16 DiT is 9.31 GiB)                                                             |
| umt5-xxl                                   | `i8`                                                                                 | `i8`                    | F32 (about 22.7 GB, `wan/text_embeds.py`)                                           | none                                                                                                                                |
| gemma4 e2b                                 | `i4` `i4-gemvpar` `i4-fast`                                                          | `i4-fast`               | BF16                                                                                | none                                                                                                                                |
| gemma4-qat e2b / e4b                       | `i4` `i4-gemvpar` `i4-fast`                                                          | `i4-fast`               | mixed F32 / BF16 / I8 / U8 (the upstream is itself a quantization-aware checkpoint) | not defined                                                                                                                         |
| birefnet-hr 1024 / 2048                    | `f16`                                                                                | `f16`                   | F16                                                                                 | `f16`                                                                                                                               |
| lucida 1024 / 2048                         | `f32` `f16`                                                                          | `f32`                   | F32                                                                                 | `f32`                                                                                                                               |
| depth-anything-v2, siglip2, vowel-detector | `f32`                                                                                | `f32`                   | F32                                                                                 | `f32`                                                                                                                               |

So five families already default to the original weights, and six do not. Of those six, only
irodori already ships a preset that holds the upstream values exactly.

### How the quantized defaults came about

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

### What changed the answer

- **The speed case is not portable.** On Apple M2 (Chrome, Metal), Tint expands `dot4I8Packed` into
  scalar code. The i8a8 linear and attention kernels therefore stay as slow as f32, even after
  geometry tuning ([0115](0115-geometry-profiles.md) Context and Consequences). A preset whose
  value is speed on Intel and NVIDIA is the wrong default for the same library on a Mac.
- **Low-bit storage is about memory, not speed.** Its benefit is a smaller download and a smaller
  resident footprint. Whether that matters depends on the user's device, and the library cannot
  see the user's priorities.
- **The common practice points the other way.** Transformers.js defaults to fp32 on WebGPU (q8 on
  WASM) and requires q4 to be named explicitly.
- **Faithful output is the safer default for a library.** A user who does not choose gets the
  model's own output; any quality loss from quantization is then the user's choice.

## Decision

1. **`defaultQuant` MUST name the preset that stores the original weights of the upstream model.**
   "Original weights" means: every weight role is stored as f32, or as f16 where the upstream
   checkpoint itself stores f16. The stored values are the upstream values, widened where needed
   and never rounded. The preset declares no activation quantization (`linearCompute: "a8"`,
   `attentionCompute: "a8"`). Rationale: the default must not change the model's numbers without
   the user asking. "Widened, never rounded" is the rule ADR 0113 (2026-10-11 addendum) already
   applies to BiRefNet: a default that silently rounds is the failure shape this ADR removes.
2. **Quantized presets (`i8`, `i4`, `i2` storage, `w8a8`-style activation quantization, and the
   `fast` bundles built on them) are selected explicitly by the user** through the existing
   `quant` option. Rationale: their benefit (memory, and speed on some GPUs) depends on the device
   and the user's priorities, which only the caller knows.
3. **No API changes; only the default's behaviour changes.** The manifest keeps the `defaultQuant`
   field (`karume/5`, ADR 0109), the hub and the family pipelines keep their resolution code, and
   the `quant` option keeps its meaning. ADR 0075 decision 3 stands: no "recommended" field is
   added. The `label` / `description` of each preset carry the trade-off. Rationale: the change is
   a data change in the distributions, and the existing surface already expresses explicit
   choice.
4. **This is a breaking change shipped in 0.15.0.** The release notes MUST announce it as
   "quantization now has to be selected explicitly". The maintainer's Japanese wording is
   「量子化は明示指定が必要になりました」. Rationale: code that omits `quant` keeps compiling and
   running, but it downloads more, holds more memory, and may run slower. That must be stated, not
   discovered.
5. **The 2026-10-04 policy ("the default is the most practical preset") is superseded**, together
   with:
   - the default-selection bullet of the [0058](0058-numerics-opt-in-contract.md) addendum
     2026-09-13;
   - the per-family default rulings that rest on it (sbv2 2026-08-20, [0120](0120-wan-dit-w8a8-seat.md)
     ruling 2026-10-04 (4), [0121](0121-wan22-ti2v-5b.md) 2026-10-06).
     The deferred "drop a8 from the default on Metal" question is closed: no default preset carries
     a8 on any platform. Rationale: one rule for every family replaces per-family rulings that each
     weighed speed on the GPUs that happened to be measured.
6. **[0110](0110-practical-tier-numerics-contract.md) decision 8 loses its trigger.** The default
   preset holds the upstream values, so choosing it needs no human quality review. Quantized
   presets remain opt-in and keep every machine check of ADR 0110 decisions 2 to 7 (census,
   difference floor, collapse ceiling, determinism, row classes). Rationale: the review existed to
   protect users who did not choose. After this ADR, no user receives a quantized preset without
   choosing it.
7. **Activation quantization stays a property of the preset, not of the numerics profile.** The
   numerics-profile ADR [0126](0126-numerics-profiles.md) makes the practical profile the runtime
   default; it covers kernel variants of contract classes E / C / R only. Rationale: a8 is
   quantization and changes quality; it must stay where this ADR makes it explicit.

## Consequences

### For users

- `fromPretrained(...)` without `quant` resolves to the original-weights preset. The output equals
  the reference behaviour for that model, but downloads and resident memory grow. For example, the
  irodori default moves from `i8-a8` to `f32`, and the anima DiT moves from i8 to the widest
  original-weights storage. On GPUs with dp4a, the default may also run slower.
- Code that wants the previous behaviour passes the previous default's name as `quant` (the
  "today" column in the table above). Presets keep their names; only the pointer moves.
- Pinned clients of 0.14.x keep reading the revisions they pinned (ADR 0073), so they do not change
  behaviour until they upgrade.

### For the repository

- Every recipe's default constant is set by decision 1. Every distribution is rebuilt (manifest and
  model card: the card's `(default)` marker and the Usage snippet are derived from `defaultQuant`).
  Every distribution that is published on Hugging Face is re-uploaded in the 0.15.0 release, and
  its revision pin is updated. The published set today: anima, anima-extra, anima-turbo,
  birefnet-hr, depth-anything-v2, gemma4, irodori v4 / v4.1-small, lucida, sbv2-jvnv, siglip2,
  umt5-xxl, wan2.2.
- Families without an original-weights preset (the table above: anima, sbv2, wan2.1, wan2.2,
  umt5, gemma4, gemma4-qat) need either a new preset or a recorded exception before their
  distribution can satisfy decision 1. Neither is decided here (Open points).
- Examples and gpu-lab print the resolved preset name, not only the manifest default. Today
  `examples/wan/main.ts:327` and `examples/sbv2/main.ts:114` print "（manifest の既定）" when
  `--quant` is omitted.
- The memory pre-check tells users which presets fit their device. The building blocks exist in
  `@karume/runtime` (`estimateGraphMemory`, `fitsHeadroom` added in 0.14.0) and in the gemma
  pipeline's session-memory estimate; a cross-preset report does not exist yet. Rationale: the
  default now costs the most memory, so the user needs a cheap way to find the lighter preset that
  fits.

### For verification

- **Tests MUST name their preset.** A test case that omits `quant` silently changes its values
  when the default moves. ADR 0120's "default switch" addendum and ADR 0121 decision 2 already
  enforce this for Wan; this ADR makes it the rule for every family. Each case that relies on the
  default today is re-keyed in one of two ways:
  - it names its old preset explicitly, which keeps its ID, sha rows and bands unchanged; or
  - it moves to a new ID for the new default preset, and its rows are written fresh (ADR 0106
    `write`).
    No case keeps an ID whose preset silently changed.
- Host checks that assert today's default (for example the Wan "omitting `quant` resolves to the
  practical preset" check in `packages/models/tests/wan_pipeline_test.ts`), the distribution gate
  (`packages/runtime/tests/distribution_gate_test.ts`, which follows the default selection), and
  the per-recipe distribution tests change with the constants.
- A mechanical check SHOULD assert decision 1 on every distribution mirror: the default preset's
  `weights` labels are all `f32` (or `f16` where the upstream is f16), and its `session` declares
  no `a8`. Rationale: the rule is checkable from the manifest alone, so it should not depend on
  review.
- Row classes (ADR 0110 decision 7) are derived from the preset's `session`, so they do not change.
  A default preset with an empty `session` yields reference rows.

### Explicitly left open

See "Open points". This ADR does not decide the numerics profile ([0126](0126-numerics-profiles.md)),
the toolchain ([0127](0127-node-dawn-toolchain.md)), the test layout
([0129](0129-test-layout-and-lanes.md)) or the repository language ([0130](0130-project-language.md)).

## Staging

Phase 3-2 of the 0.15 migration, after phase 3-1 (numerics profile, ADR 0126) and before 3-3
(release).
Each step lists its check.

| Step | Work                                                                                                                                                                 | Check                                                                                                                                                                                       |
| ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | Decide, per family without an original-weights preset, between adding one and a recorded exception (Open points 1-4).                                                | Every row of the Context table has an entry in this ADR's addendum.                                                                                                                         |
|    2 | Make every test case that relies on the default name its preset (or move it to a new ID).                                                                            | With the mirrors still at today's defaults, every lane is green and no sha row changed.                                                                                                     |
|    3 | Set every recipe's default constant by decision 1. Add any new original-weights presets from step 1. Add the mechanical check of decision 1.                         | `uv run pytest` in `tools/exporter` and `tools/export-recipes`. The new check is red on today's mirrors and green after step 4.                                                             |
|    4 | Rebuild every distribution mirror.                                                                                                                                   | `verify_dist` is green. Only `karume.json` (`defaultQuant`, any new preset) and the card changed, plus new containers for new presets. The distribution gate and the host checks are green. |
|    5 | Print the resolved preset in examples and gpu-lab. Add the "which presets fit" report to the memory pre-check.                                                       | Example output names the preset; a unit test covers the report on a fake device.                                                                                                            |
|    6 | Run the lanes and write reference rows for the new default IDs.                                                                                                      | Phase 3 acceptance: only the intended differences (new IDs, new presets) appear against the phase 2 output.                                                                                 |
|    7 | Re-upload in the 0.15.0 release and update the pins (phase 3-3, release runbook). Write the CHANGELOG entry with the decision 4 announcement under breaking changes. | The published smoke test resolves the new defaults from the pinned revisions.                                                                                                               |

## Open points

1. **Is f16 an acceptable "original" for an f32 or bf16 upstream?** Decision 1 as ruled says f32,
   or f16 only where the upstream ships f16. Under that wording, anima (BF16 DiT), wan2.1 (F32 DiT)
   and wan2.2 (F32 DiT) have no conforming preset today. The glossary draft (`docs/GLOSSARY.ja.md`,
   "original weights") describes them as "f16 / f32 written by the exporter without quantization",
   which would admit their `f16` presets. BF16 to f16 is exact only for values inside f16's normal
   range; this has not been measured per checkpoint.
2. **Families whose original weights are impractical to ship or to run.** wan2.2 (f16 DiT 9.31
   GiB, f32 about twice that), umt5-xxl (f32 about 22.7 GB), gemma4 e2b (BF16 upstream, only i4
   presets) and sbv2 (f32 1.32 GB, ruled impractical in 2026-08) each need either a new preset or
   an exception, together with the device class it targets.
3. **gemma4-qat.** The upstream is already a quantization-aware checkpoint with mixed storage, so
   "original weights" is not defined by decision 1's wording.
4. **Cross-referenced components.** The Wan presets take their text encoder from the umT5
   distribution, which ships only `i8`. Whether a Wan default can satisfy decision 1 while its text
   encoder is i8 is not decided.
5. The exact English wording of the CHANGELOG entry, and whether the release notes list each
   family's previous default.
6. The API shape of the memory pre-check's "which presets fit" report, and where it lives
   (runtime, hub or models).
7. Whether a human quality review is still required before a quantized preset is published, now
   that decision 6 removes the default trigger. The `description` of a preset is itself a quality
   claim.
8. The release date of 0.15.0, and whether it is published on JSR only or on npm as well (PLAN
   open point; it affects the re-upload step only through timing).

# 0034: S-form DiT (a dynamic graph with one token-length symbol)

- Status: accepted (2026-08-04); partly superseded by [0038](0038-manifest-v1.md) §4, still
  canonical (distribution and models pipeline use the S form only; no `--dit-graph` runtime knob),
  and [0109](0109-manifest-v5-container.md) decision 4 (rope tables are a container asset, addendum
  2026-09-24); decision 1 re-graded to MAY and decision 4's check withdrawn on 2026-10-18
- Date: 2026-08-04
- Related: recon [2026-08-03-dynres-vae-tiling.md](../research/2026-08-03-dynres-vae-tiling.md) §2.2
  (the S form, the obstacles) / [0016](0016-anima-chain-export.md) (resolution-baked graphs; this
  adds a series beside them) / [0010](0010-symbolic-constant-folding.md) (why `Dim("S")` has an
  upper bound) / [0013](0013-sbv2-chain-export.md) (a table promoted to an input) /
  [0033](0033-vae-fixed-tile-decode.md) (the VAE side of variable resolution)

## Context

The DiT depends on resolution through **three constants only** (the zero padding channel
`[1,1,H,W]` and the rope cos / sin tables `[1,1,S,128]` × 2). The IR's dimension language (a linear
expression in one symbol per dimension) cannot write `H·W`, but moving the graph entry **after**
patchify lets every node be written in the one token-length symbol `S` (recon §2.2). The runtime
accepts spatial symbols without changes (plans are per run; keys carry no shape).

## Decision

1. **An additional series, not a replacement.** `python -m anima.export --dit-graph dyn`
   (transformer only) emits a `…-dyn` series under `outputs/series/`. The static series MAY be kept
   as a reference series; the recipe default stays `static` (`DIT_GRAPHS` in
   `tools/export-recipes/anima/export.py`). (Re-graded 2026-10-18 to MAY: ruling 7a withdraws the
   MUST "do not move the static assets, e2e checks, tolerances or snapshots", which protected them
   during the transition; only the S form is distributed and only `dyn` series carry `rope_base`.)
2. **The graph entry is after patchify.** Inputs are `tokens [1,S,68]` (68 = 17·2·2, including the
   constant-zero padding channel) and `rope_cos` / `rope_sin [1,1,S,128]` (**promoted to inputs**,
   as with the SBV2 relative-position table). The output is the token sequence before unpatchify.
   Patchify, unpatchify and rope assembly are host-side pure functions
   (`packages/models/src/anima/dit-tokens.ts`). **No S-dependent constant is baked in**: zero
   initializers of rank ≥ 3, fixed by a pytest in `tools/export-recipes/anima/tests/test_patch.py`
   (the static form has 3, so the check is not vacuous).
3. **The rope tables are a permutation of per-axis base tables, not a computation** (a design change
   of the implementation leg, accepted by ruling). torch's f32 cos / sin differ from correct rounding
   by 1 ulp (472 cos / 231 sin cases of 8,192), while JS `Math.cos` rounds correctly, so **porting
   the formula to TS cannot be bit-identical in principle**. The exporter cuts per-axis base tables
   out of the output of `model.rope` (`rope_base`, 66 KB, resolution-independent, 128 rows = latent
   256 = the model's ceiling of 2048 px); the host only arranges them as `[t,h,w,t,h,w]` and never
   calls a trigonometric function (the same choice as baking a tokenizer into tables).
4. Withdrawn 2026-10-18: the main check "S form ≡ static graph, Uint32-identical, never loosened"
   (held in three cases on 2026-08-04) lost its test
   (`packages/runtime/tests/e2e_anima_dyn_test.ts`) in the models move of 2026-08-15; the DiT is
   checked by the sha256 reference rows ([ADR 0106](0106-device-keyed-references.md)) and
   `packages/e2e/tests/e2e_anima_ab_test.ts`. Still in force: `Dim("S", min=2, max=16384)`
   (`MAX_DIT_TOKENS` in `packages/models/src/anima/resolution.ts`); `torch.export` created no
   S-dependent guard.
5. `--verify` compares "host patchify → S form → host unpatchify" with the unpatched diffusers model
   through a per-case adapter (bit-exact in 2 cases). The two golden points change **both resolution
   and timestep** (S = 1,024 / 4,096 — the same S twice would not show a lost binding).

## Measurements (2026-08-04)

Uint32-identical to the static graph in 3 cases (512 px f16, 1024 px i8, 512 px i8 with w8a8 +
attention a8 + s16); rebuilt rope tables byte-identical to the baked ones; demo PNG sha256
identical; per step 1,046 vs 1,051 ms, dispatches 2,633 → 2,623. Full record: commit `99a3c649`.

## Detection limits

1. **Swapping h and w in the rope base tables cannot be detected numerically** — Anima's
   `rope_scale` is `[1.0, 4.0, 4.0]` (h and w equal), so `cos_h == cos_w` byte for byte; a trap that
   depends on the model configuration. The working detector is the implementation form itself
   (cutting the tables out of upstream output leaves no room for a copying error).
2. **torch's f32 trigonometric functions are not correctly rounded** (SLEEF, 1.0 ULP bound) —
   "writing the same formula as the reference gives bit identity" does not hold for transcendental
   functions; baking a table is the only way to identity.
3. A TypedArray passed to `assertEquals` makes failure formatting practically endless (cut off after
   15 minutes); compare large arrays with a "first mismatch + count" assert.

## Consequences

- **One DiT asset covers every resolution** (up to 2048 px). With the VAE tiles of ADR 0033 the
  parts for variable resolution are complete; arbitrary and non-square resolutions were wired by
  [ADR 0036](0036-freeform-resolution.md).
- At 2048 px the intermediate S of DiT attention (16 × 16,384² elements, 8 GiB even with s16) did not
  fit at the time; that was left as separate memory work.

## Addendum (2026-09-24) — the rope base tables are the container asset `rope_base`

Only the location of decision 3's tables changed: with [ADR 0109](0109-manifest-v5-container.md)
decision 4 the manifest's `extras` were retired, and the tables are the asset `rope_base` (role
`rope-base`) of the `transformer` container, so swapping the transformer through `components` also
swaps them. The payload is still safetensors, read with `parseSafetensors`
(`packages/models/src/anima/rope-base.ts`).

# 0125: BiRefNet alpha matte post-processing — per-checkpoint upstream procedures, the golden contract, the A/B rule

- Status: accepted (2026-10-11, the maintainer's rulings on the parity level, per-checkpoint
  procedures, the A/B form and material, one Pillow port, adopting the upstream procedure for HR and
  adding no public option); the A/B result and the distribution declaration `postprocess`
  (`birefnet/2`) were settled the same day (see "Results")
- Date: 2026-10-11
- Related: ADR [0008](0008-public-api.md) (thin public surface) / [0038](0038-manifest-v1.md)
  (extending `pipelineConfig` and the pipeline major) / [0113](0113-birefnet-weight-series.md)
  (BiRefNet weight series) / [0121](0121-wan22-ti2v-5b.md) decision 11 (Wan2.2 I2V conditioning image
  bit-identical with Pillow's LANCZOS) / [0005](0005-verification.md) addendum 2026-10-11 (upstream
  parity policy)

## Context

### What happened

A review (R6B-001) showed that BiRefNet post-processing (graph logit → sigmoid → resize to the input
image → 8-bit alpha) differed from upstream in interpolation and in where quantization happens, moving
edge alpha by up to 10–29 levels (upscaling cases, measured against an in-repository f64 Pillow-style
helper, not Pillow itself). The maintainer's ruling: implement the upstream procedure and decide by an
A/B against the current path.

### Upstream differs per checkpoint

- **BiRefNet_HR** (bundled `handler.py:131-137`; the model card's example has the same order):
  half-precision model → half-precision `sigmoid` → `transforms.ToPILImage()` (torchvision 0.28:
  `(x * 255).astype(np.uint8)`, truncation) → Pillow `pred_pil.resize(image.size)` (BICUBIC by
  default, 8-bit, one band). Quantization comes **before** the resize.
- **Lucida** (an HR fine-tune; model card `README.md:86-89`): f32 model → `sigmoid` →
  `transforms.functional.resize` (tensor branch, bilinear, antialias by default) →
  `(alpha.numpy() * 255).astype("uint8")` (truncation).
- **The path before this ADR** (`packages/models/src/birefnet/matte.ts`): σ computed in f64 and stored
  as f32 → `resizePlaneF32` (antialiased bilinear, f64, no rounding between passes) →
  `Math.round(255·α)`. Its interpolation matched Lucida's upstream; it differed from HR's upstream in
  interpolation and quantization position, and from both in rounding (round vs truncate).

### Quantization thresholds (own calculation, checked with torch on CPU)

The smallest logit giving alpha 255 is 6.23 (ln 509) with rounding (f32 / f64), 8.32 (ln 4095) with
HR's upstream (half precision + truncation), and 16.64 (24·ln 2) with f32 truncation (Lucida's
upstream). On the real-image e2e's f32 matte logits (2026-10-11, 2 images with foreground × 2
checkpoints), f32 truncation turns 60.1–85.3% of the foreground (rounded alpha ≥ 128) into 254
instead of 255.

## Decision

1. **Level of upstream parity**: parity is reached at the level of computed values or of their
   intent. Upstream errors, bugs and precision choices made for speed are not copied; what is copied
   per checkpoint is **the interpolation kernel and the order of stages**.
2. **Each checkpoint follows its own upstream**, declared in the distribution's `pipelineConfig`
   (settled in "Results": the field was added).
3. **Two procedures** (`MATTE_PROCEDURES` in `matte.ts`):
   - `u8-pil-bicubic` (HR's upstream intent): σ in f64 → `Math.round(255σ)` into an 8-bit plane →
     Pillow 12.3.0's 8-bit one-band BICUBIC (22-bit fixed point, uint8 intermediate, a pass is skipped
     on an axis whose size is unchanged) → uint8. σ is not stored as f32 before rounding.
   - `float-bilinear` (Lucida's upstream intent = the earlier path itself): unchanged to the bit,
     including storing σ as f32.
   - The procedure argument of `matteFromLogits` is internal, not exported from `mod.ts` or a subpath
     (ADR 0008). It has no default (revised in "Results"): the pipeline passes the distribution's
     `config.postprocess`, so a missing declaration cannot silently run one checkpoint's procedure.
4. **Every procedure rounds** (`Math.round(255·α)`, 0.5 rounds up). Upstream truncation is the
   `astype(uint8)` idiom, not intent — a foreground of confidence 1 is meant to be opaque (255). As a
   consequence every procedure saturates from logit 6.23.
5. **Precision**: HR's upstream half precision is not copied. It is a speed choice for the whole model
   (`handler.py:101` runs `.half()`; the upstream weights in `inputs/birefnet/BiRefNet_HR/model.safetensors`
   are F16 for every float tensor — F16 687, I64 67), so σ in half precision follows from it. Karume's
   logits come from an f32-compute graph (HR's preset stores f16 weights and computes in f32 — ADR 0113
   addendum 2026-10-11: HR keeps only the f16 preset), so they never match the half-precision
   activations, and imitating half precision in the last stage alone does not bring the result closer.
   Upstream bugs are not copied either (for example torch 2.13.0's CPU antialiased bilinear breaks the
   vertical axis for width-1, height ≥ 2 inputs, repeating the first row — measured).
6. **Golden contract** (a permanent check, CPU only, environment-independent):
   - Writer: `tools/export-recipes/birefnet/matte_fixture.py`. It calls each checkpoint's upstream
     functions as they are and replaces only quantization with `floor(x + 0.5)`. Fixtures:
     `packages/models/tests/fixtures/birefnet-matte/` (tracked).
   - Inputs: windows (40×32) near the contour of the real-image e2e's matte logits for the default
     presets (portrait / street; HR from the f16 preset, Lucida from the f32 preset — ADR 0113
     addendum 2026-10-11), a 1024-long row and a 2-wide column through each window centre (the
     production coefficient tables 1024 → 832 / 480 / 2048), and synthetic edge cases. Provenance (file,
     sha256, window) is in the input table and the torch / torchvision / Pillow / numpy versions in the
     JSON; the recipe's pytest rebuilds from the tracked inputs and checks byte identity. Target sizes
     other than 1024² (up, down, non-square, one axis only, identity).
   - `u8-pil-bicubic`: **zero mismatches**. JS `Math.exp` and numpy `exp` may differ in the last ulp,
     so the generator rejects samples whose 255σ lies within 1e-9 of `k + 0.5`.
   - `float-bilinear`: equals the rounding of the golden (torch's unrounded `255α`), or differs by one
     level only where the golden `255α` is within **ε = 1e-2** of `k + 0.5`. ε is the smallest power of
     ten at least twice the measured maximum difference between torch's f32 `255α` and the TS f64 path
     (4.2e-3 on 2026-10-11, the 1024 → 480 downscale column; torch keeps coordinates and weights in
     f32, which matters more at large indices). The generator fails when the measurement exceeds ε/2.
     In the 2026-10-11 fixture, 66 of 76,767 samples are one-level differences within ε, at most
     7.6e-6 (2^-17) from `k + 0.5` (case `thin.up`).
   - Pillow in the birefnet group is pinned `==12.3.0` (the same version as the wan group —
     `tools/export-recipes/pyproject.toml`).
   - The check also runs under the default `uv run pytest`: Pillow reaches the default sync through
     torchvision (an exporter core dependency) and `uv.lock` resolves it to 12.3.0 under the pin
     (confirmed 2026-10-11 in a default-group-only venv: 14 golden tests and 11 A/B verification tests
     ran without SKIP). `importorskip("PIL")` in the tests guards environments without Pillow and does
     not skip under the current dependencies.
7. **One Pillow port**: the verbatim port of Pillow's 8-bit separable resample lives only in the
   shared core `packages/models/src/image/pillow-resample.ts` (filter and channel count as
   arguments). `u8-pil-bicubic` (one band, BICUBIC) and Wan2.2 I2V's `lanczos.ts` (three channels,
   LANCZOS — ADR 0121 decision 11) build on it; Wan's existing LANCZOS fixtures still match.
8. **A/B for HR only, three choices**: Lucida already matched its upstream at the value level through
   the fixtures of decision 6. HR was judged on 8 blind pairs (4 down, 4 up) with "A / same / B"; the
   current path would be kept only with 5 or more "current" and 0 "upstream" answers, otherwise the
   upstream procedure is adopted. Done 2026-10-11 (see "Results").
9. **Numeric columns are records only** (contour-band differences between the two procedures, and
   SAD / MSE / gradient error against the 2048² model's σ for 2048² targets), never a decision input.
10. **A/B material was generated with anima** (4 images at 2048×2048, 4 at 832×480 cut from 832×512;
    hair, fur and thin objects, neutral subjects only; images only under `outputs/`). The corpus
    definition `tools/birefnet-ab/corpus.ts` was removed with `tools/birefnet-ab/` in `bb35fe2a`
    (2026-10-16) and remains in the 0.14.0 history; the checker
    `tools/export-recipes/birefnet/matte_ab_check.py` remains.
11. **Distribution change only if HR adopts the upstream procedure**: settled in "Results" — the
    `postprocess` field was added (pipeline major `birefnet/2`). Moving the fixture inputs from the f32
    to the f16 preset left the check unchanged (same values and sha256 — ADR 0113 addendum
    2026-10-11).

## Alternatives not taken

- **Fix one procedure in code**: upstream differs per checkpoint, so at least one checkpoint would
  leave its upstream; aligning with HR would move Lucida further away.
- **A checkpoint → procedure table in code**: puts model-specific values into TS (against the MUST in
  `birefnet/config.ts`) and misses third-party fine-tunes.
- **Copy HR's half precision (`Math.f16round`)**: its only purpose was to reproduce the 8.32 saturation
  threshold under truncation, which decision 4 removes.
- **Truncate as upstream does**: 60–85% of the foreground would be 254 (Context), breaking uses that
  treat 255 as opaque.
- **Five-level answers**: assume finer visual distinctions; three choices give steadier answers.
- **Separate material per checkpoint (photos for HR, illustrations for Lucida)**: suitable photos are
  hard to prepare.

## Consequences

- HR's `segment` output changes only at contours, and only when the input size differs from the model
  resolution (equal sizes skip the resample, leaving only the f32-storage rounding of σ). Lucida's
  output does not change.
- The contract covers "Karume's logits through each checkpoint's upstream procedure"; no match with
  the upstream half-precision model's output is claimed.
- Raising Pillow requires rebuilding the BiRefNet and Wan LANCZOS fixtures; both recipe pytests compare
  against the installed Pillow and turn a forgotten rebuild red.
- The A/B checker (`birefnet/matte_ab_check.py`) verifies tool output under decision 6's contract,
  excluding output pixels reached through the BICUBIC support by tied samples (255σ within
  `TIE_MARGIN` of `k + 0.5`; count in `excludedTiePixels`) and requiring zero mismatches elsewhere.

## Results (addendum 2026-10-11) — A/B answers, rulings and the distribution declaration

- **A/B**: all 8 HR pairs answered "same" (one pair, down-fern, had no detected foreground and zero
  difference); the checker found zero mismatches against the upstream functions. Records:
  `outputs/birefnet-ab/2026-10-11/` (untracked).
- **Ruling**: under decision 8's rule, HR adopts `u8-pil-bicubic`; Lucida stays `float-bilinear`. No
  public option for choosing a procedure is added: the upstream procedure needed no workaround, so
  following upstream adds no choice (ADR [0005](0005-verification.md) addendum 2026-10-11).
- **Schema**: the distribution's `pipelineConfig` gains the required field `postprocess`, a closed set
  `u8-pil-bicubic` / `float-bilinear` with no default; anything else fails loudly. A new required field
  raises the pipeline major to `birefnet/2` (ADR [0038](0038-manifest-v1.md)); `birefnet/1`
  distributions fail the major check. The distribution builder writes it from the checkpoint
  (`BIREFNET_POSTPROCESS` in `tools/export-recipes/birefnet/distribution.py`: HR → `u8-pil-bicubic`,
  Lucida → `float-bilinear`); TS holds no checkpoint table. Model cards describe each model's procedure
  from the declaration.
- Published as `birefnet/2` with the re-uploads of `karume-birefnet-hr` and `karume-lucida` (CHANGELOG
  0.14.0).

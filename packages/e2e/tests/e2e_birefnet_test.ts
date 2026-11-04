// The real-GPU golden E2E of the real-weight BiRefNet family (background removal / salient object
// segmentation) (stage 3 of ADR 0005).
//
// Where the tiny golden (tests/e2e_golden_test.ts) covers "the op contracts" and SigLIP2
// (tests/e2e_siglip2_test.ts) covers "the image family with a single-vector output", this one
// covers **the image family with a per-pixel output** (a `[1,1,S,S]` map of 1,048,576 elements).
//
// The containers come from the distribution mirrors and the goldens from the test-asset root (both
// outside git — "Where the assets come from" below). The goldens are recipe outputs of
// `tools/export-recipes/birefnet/export.py` (the command per series is {@link Series.generate}).
//
// There is **one series per model × resolution** ({@link SERIES} below). BiRefNet_HR (upstream's
// high-resolution checkpoint) and Lucida (its fine-tune — the structure is identical, only the
// weights differ) run as separate series. Both the shifted-window mask and the zero constants of
// the H/W padding are baked as resolution-dependent constants, so another resolution is another
// graph. The resolutions are **1024²** and **2048²** (General-HR of the upstream handler); both are
// in {@link SERIES}, and the tolerance is measured per series (2048² is all green on the RTX 3080
// Ti). However, on the Intel Arc B570 machine the tests that use the GPU are held for 2048²
// ({@link HELD_SERIES} — the last dispatch of the decoder exceeds the driver's job limit and takes
// the whole process down).
//
// **The storage dtype is an axis of the series too** (ADR 0113): one series per preset of the
// distribution, and a `--dtype f16` series is a separate `<model>-<resolution>-f16` series.
// BiRefNet_HR has only the f16 series (the upstream checkpoint is f16 and the f32 series merely
// widened the same values, so it was retired — ADR 0113 addendum 2026-10-11); Lucida has both an
// f32 and an f16 series. The goldens of an f16 series are torch outputs taken with the rounded
// weights, so here too **only the runtime's numeric error** shows up in the comparison (the
// quantisation error is held by `quality.json` at export time). A mix-up of the series root (f16
// assets on the f32 preset or the reverse) cannot be detected numerically (the tolerances are of
// the same order, so each passes the other's assets — ADR 0027 / 0029), so a separate check (no GPU
// needed) compares the container's **set of compressed storage dtypes** with the series'
// declaration.
//
// **The tolerances are measured independently per series** (each row of `SERIES` has its own
// tolerance). Sharing them would let one silently loosen when the other is re-measured — the 2
// series have the same structure but logit ranges an order of magnitude apart (measured: an \|ref\|
// upper end of 64.1 on BiRefNet_HR against 1078.1 on Lucida).
//
// There is one graph, and one output, **the pre-sigmoid logit `[1,1,S,S]`** (the matte's α is the
// host-side value after `sigmoid`). The input is one **normalised** `pixel_values f32 [1,3,S,S]`,
// with no symbolic dimensions.
//
// ## The two groups of goldens (synthetic images + real images) — both are kept
//
// In **both groups the input is the `pixel_values` baked into the golden as is** (bit-identical),
// so **only the runtime's numeric error** shows up in the comparison. They are kept as two groups
// because they exercise different distributions:
//
// - **4 synthetic-image cases** (`checker` / `disc` / `noise` / `ramp`): they reach the ends of the
//   value range and gradients, so they detect numeric regressions sharply. `disc` is a salient
//   object — a bright disc on a dark background — and also serves as the base for applying a
//   geometric discrimination (mean logit inside the disc > outside) to the real-GPU output.
// - **4 real-image cases** (`photo-*`): the runtime fidelity at a point of the natural-image
//   distribution. The tolerance is derived from measurement independently per group and per series
//   ({@link HR_F16_REAL_TOLERANCE}).
//
// ## The chain including the TS preprocessing is held by **composing** two checks
//
// The chain "pass a PNG and get the same matte as Python" is held not by this test alone but by
// composing the **input-side parity check** of `packages/models/tests/e2e_birefnet_real_test.ts`
// (the same PNG yields the same `pixel_values` as Python — input difference ≤ 1e-6) with this
// test's **fidelity on the golden input**. The preprocessing does not run here because of the
// dependency direction (a runtime test importing the models implementation by relative path is the
// wrong direction). As a by-product of the split, when something fails the test name tells whether
// the preprocessing or the inference moved.
//
// Of the semantic discrimination, the **real-image side** (the foreground ratio of the 2 images
// with a salient object exceeds that of the 2 without) and the matte PNGs written along with it are
// on the models side for the same reason (`encodePng` is the models implementation, and the input
// also goes through the TS preprocessing). **The geometric discrimination on the synthetic side**
// (`disc`) is complete with the golden's input alone, so it stays here.
//
// Where the assets are absent the cases **SKIP explicitly** (independently per series). The
// real-image group SKIPs independently too (assets emitted without `--real-images` hold only the 4
// synthetic cases). The gate of ADR 0005's "all cases SKIP = explicit FAIL"
// (tests/gpu_gate_test.ts) looks only at *whether a GPU adapter exists*, independently of this
// SKIP. Conversely, when the assets are present **only in part** (missing cases) it is a FAIL
// rather than a SKIP (the "asset completeness" test below) — there it would be a silent apparent
// success.
//
// ## Where the assets come from (ADR 0139)
//
// The containers are read from the distribution mirrors (`models/karume-birefnet-hr/`,
// `models/karume-lucida/`) through their manifests — model = resolution, quant = storage dtype
// ({@link Series.mirror}). The goldens (`io.*.safetensors`) are read from the test-asset root,
// `goldenDir(<series>)` = `outputs/test-assets/goldens/<series>/` (fetched with `node
// tools/test-assets/fetch.ts`). Each case requires exactly the assets it reads: the storage case
// the mirror only, the geometry case the mirror plus `io.disc`, the parity cases the mirror plus
// their golden. The series names stay as the golden directory names, the case IDs and the keys of
// {@link HELD_SERIES}. A golden that is absent SKIPs only while the test-asset root is not a
// complete fetch of the pin (the assets gate is red then); absent from a complete fetch, it FAILs —
// the dataset at the pin lacks it. An absent mirror SKIPs (the distribution gate owns that red).

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  type CodecLayout,
  codecLayout,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Tensor,
} from "../../runtime/mod.ts";
import {
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../../runtime/src/reference/allclose.ts";
import {
  assertAdapterMatchesEnvironment,
  ENVIRONMENT,
} from "../../runtime/tests/helpers/environment.ts";
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
import { GPU_AVAILABLE } from "../../runtime/tests/helpers/gpu.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type MirrorChoice,
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * The tolerance of the comparison of the **BiRefNet_HR / synthetic-image** cases (series
 * `birefnet-hr-1024-f16` — the input is bit-identical to the golden).
 *
 * **The measurement table below (and the tables of {@link HR_F16_REAL_TOLERANCE} /
 * {@link HR_F16_2048_SYNTHETIC_TOLERANCE}) were taken on the f32 series (`birefnet-hr-{1024,2048}`
 * — retired in the ADR 0113 addendum 2026-10-11)**, and they apply to the f16 series as is: the
 * upstream BiRefNet_HR checkpoint is f16, so rounding to f16 is the identity and the goldens are
 * bit-identical to those of the f32 series (enforced on 8 cases by the quality check at export
 * time; 1024² compared on both inputs and outputs on 2026-09-26). The real-GPU outputs match too —
 * the sha256 reference rows of the matte logits of the 4 real-image cases
 * (`packages/models/tests/fixtures/references/birefnet.json`) match the f32 series' rows exactly in
 * both environments (RTX 3080 Ti and Arc B570), and in the measurement of 2026-10-11 (RTX 3080 Ti)
 * the largest absolute error of the disc case against the golden was the same value on both series
 * (1024² 1.297e-4, 2048² 8.60e-4).
 *
 * Measured (a plain comparison with `atol=rtol=0`, 4 cases × 1 output `[1,1,1024,1024]`):
 *
 * | Case    | maxAbs  | maxRel  | \|ref\| upper end | \|ref\| smallest nonzero |
 * | ------- | ------- | ------- | ----------------- | ------------------------ |
 * | checker | 2.38e-5 | 3.04e-6 | 11.677            | 3.50                     |
 * | disc    | 1.26e-4 | 7.69e-3 | 23.123            | 5.36e-4                  |
 * | noise   | 8.30e-5 | 1.00e-5 | 15.302            | 3.16                     |
 * | ramp    | 2.11e-4 | 6.73e-1 | 10.765            | 2.58e-4                  |
 *
 * atol 1e-3 is about 4.7 × the measured worst 2.11e-4 (ramp).
 *
 * **rtol is 0.** This output is a map of logits that cross 0, and in cases with a boundary between
 * foreground and background such as `disc` / `ramp`, \|ref\| thins down to 2.6e-4 (boundary pixels
 * are logit ≈ 0 by definition). Making rtol the main term would make the verdict diverge at the
 * boundary — even the element with the measured maxRel 0.673 (ramp) has an absolute error of only
 * 1.7e-4.
 *
 * The sources of error are the same as SigLIP2 (fma fusion, a different reduction order of linear /
 * conv from torch, implementation differences of transcendental functions), but the value range is
 * wide (\|ref\| upper end 23.1), so the absolute amount is an order of magnitude larger. In
 * relative terms it is 2.11e-4 / 10.8 ≈ 2.0e-5, the same order as the real-image side (1.09e-3 /
 * 64.1 ≈ 1.7e-5 — {@link HR_F16_REAL_TOLERANCE}).
 *
 * The error of an implementation bug (mixed-up deform_conv2d offsets, a phase shift of the window
 * mask, a wrong upsample axis, a wrong order of BatchNorm's per-channel constants) is O(1) to
 * O(20), the same as the output's value range, four or more orders of magnitude above this
 * threshold.
 */
const HR_F16_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1e-3, rtol: 0 };

/**
 * The tolerance of the comparison of the **BiRefNet_HR / real-image** cases (`photo-*` — series
 * `birefnet-hr-1024-f16`).
 *
 * Measured (a plain comparison with `atol=rtol=0`, 4 cases × 1 output `[1,1,1024,1024]` — a table
 * taken on the retired f32 series. Why it applies to the f16 series as is: see
 * {@link HR_F16_SYNTHETIC_TOLERANCE}):
 *
 * | Case            | maxAbs (TS input) | \|ref\| upper end | **maxAbs (golden input)** |
 * | --------------- | ----------------- | ----------------- | ------------------------- |
 * | photo-corridor  | 4.77e-5           | 12.729            | 4.01e-5                   |
 * | photo-landscape | 5.25e-5           | 12.662            | 3.24e-5                   |
 * | photo-portrait  | 1.78e-3           | 64.094            | 1.09e-3                   |
 * | photo-street    | 8.58e-5           | 16.069            | 9.63e-5                   |
 *
 * The input this check uses is **the golden's `pixel_values`, the same as the rightmost column**
 * (the chain through the TS preprocessing moved to the models side — the "composing two checks" of
 * the module docstring). The effective measured worst is 1.09e-3 (photo-portrait), and **atol 5e-3
 * is kept as is** (a margin of about 4.6 ×. It only became wider than the 2.8 × of when it was
 * derived from the left column; tightening it needs a re-measurement in an environment with the
 * real assets).
 *
 * That the left and right columns are of about the same order is **the evidence of attribution**:
 * the error on the real-image side is large because of **the value range**, not a preprocessing
 * difference — photo-portrait's logits swing up to 64.1 (the salient object fills 55% of the frame,
 * and the saturation inside is deep). In relative terms it is 1.09e-3 / 64.1 ≈ 1.7e-5, the same
 * order as the synthetic side (≈2.0e-5). **The constant is still not shared** — re-measuring one
 * would silently loosen the other.
 *
 * **rtol is 0** (the same reason as {@link HR_F16_SYNTHETIC_TOLERANCE} — a map of logits that cross
 * 0).
 *
 * **A replaced image** (the generation script was rerun without retaking the goldens) is not
 * visible to this check — its sha256 comparison is held by name in
 * `packages/models/tests/e2e_birefnet_real_test.ts`.
 */
const HR_F16_REAL_TOLERANCE: Tolerance = { atol: 5e-3, rtol: 0 };

/**
 * The tolerance of the comparison of the **BiRefNet_HR 2048² / synthetic-image** cases (series
 * `birefnet-hr-2048-f16`).
 *
 * Measured (a plain comparison with `atol=rtol=0`, 4 cases × 1 output `[1,1,2048,2048]`, RTX 3080
 * Ti — a table taken on the retired f32 series `birefnet-hr-2048` [added on 2026-09-05, when recipe
 * patch ⑨ + ADR 0093 let the execution stage run]. Why it applies to the f16 series as is: see
 * {@link HR_F16_SYNTHETIC_TOLERANCE}):
 *
 * | Case    | maxAbs  | maxRel  | \|ref\| upper end | \|ref\| smallest nonzero |
 * | ------- | ------- | ------- | ----------------- | ------------------------ |
 * | checker | 4.29e-5 | 4.92e-6 | 12.121            | 3.61                     |
 * | disc    | 8.60e-4 | 6.33e-3 | 47.904            | 1.06e-3                  |
 * | noise   | 5.72e-5 | 6.62e-6 | 16.170            | 2.51                     |
 * | ramp    | 1.15e-4 | 6.51e-3 | 10.696            | 6.84e-4                  |
 *
 * atol 4e-3 is about 4.7 × the measured worst 8.60e-4 (disc). It is wider than the 1e-3 of 1024²
 * because of **the value range** (disc's \|ref\| upper end doubles from 23.1 to 47.9); in relative
 * terms 8.60e-4 / 47.9 ≈ 1.8e-5, the same order as 1024²'s 2.0e-5. **rtol is 0** (the same reason
 * as {@link HR_F16_SYNTHETIC_TOLERANCE}). The binary-mask disagreement is 0 on all 4 cases. No
 * real-image goldens are taken at 2048² (the corpus is 1024², so the resize would not be the
 * identity — the real-image check is held by the 1024² series).
 */
const HR_F16_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 4e-3, rtol: 0 };

/**
 * The tolerance of the comparison of the **Lucida / synthetic-image** cases.
 *
 * Measured (a plain comparison with `atol=rtol=0`, 4 cases × 1 output `[1,1,1024,1024]`):
 *
 * | Case    | maxAbs  | maxRel  | \|ref\| upper end | \|ref\| smallest nonzero |
 * | ------- | ------- | ------- | ----------------- | ------------------------ |
 * | checker | 1.53e-5 | 2.68e-6 | 13.554            | 3.34                     |
 * | disc    | 1.11e-4 | 2.83e-3 | 17.400            | 1.19e-3                  |
 * | noise   | 5.29e-5 | 1.85e-2 | 13.574            | 1.76e-4                  |
 * | ramp    | 4.41e-5 | 2.89e-3 | 13.210            | 3.04e-4                  |
 *
 * atol 5e-4 is about 4.5 × the measured worst 1.11e-4 (disc). **rtol is 0** (the same reason as
 * {@link HR_F16_SYNTHETIC_TOLERANCE} — a map of logits that cross 0).
 *
 * It is a step smaller than BiRefNet_HR because Lucida's response to the synthetic images is
 * shallow (it does not find a salient object, so the value range does not widen). This asymmetry is
 * also why **the two series do not use the same constant**.
 */
const LUCIDA_SYNTHETIC_TOLERANCE: Tolerance = { atol: 5e-4, rtol: 0 };

/**
 * The tolerance of the comparison of the **Lucida / real-image** cases (`photo-*`).
 *
 * Measured (a plain comparison with `atol=rtol=0`, 4 cases × 1 output `[1,1,1024,1024]`):
 *
 * | Case            | maxAbs (TS input) | \|ref\| upper end | **maxAbs (golden input)** |
 * | --------------- | ----------------- | ----------------- | ------------------------- |
 * | photo-corridor  | 1.25e-4           | 12.950            | 5.34e-5                   |
 * | photo-landscape | 1.00e-4           | 13.711            | 8.77e-5                   |
 * | photo-portrait  | 9.80e-3           | **1078.080**      | 4.30e-3                   |
 * | photo-street    | 8.77e-4           | 210.791           | 1.05e-3                   |
 *
 * The input this check uses is the golden's `pixel_values`, the same as the rightmost column. The
 * effective measured worst is 4.30e-3 (photo-portrait), and **atol 3e-2 is kept as is** (a margin
 * of about 7.0 ×).
 *
 * It is an order of magnitude looser than BiRefNet_HR's real-image check (5e-3) only because **the
 * difference in value range shows through as is**, not because the precision drops — in relative
 * terms 4.30e-3 / 1078.1 ≈ 4.0e-6, **smaller** than BiRefNet_HR's 1.7e-5. The fine-tuned Lucida is
 * an order of magnitude more confident about the foreground (logit 1078 = a saturation whose
 * distance from 1 after sigmoid cannot be represented even in f64).
 */
const LUCIDA_REAL_TOLERANCE: Tolerance = { atol: 3e-2, rtol: 0 };

/**
 * The tolerance of the comparison of the **Lucida 2048² / synthetic-image** cases (series
 * `lucida-2048` — added on 2026-09-05).
 *
 * Measured (a plain comparison with `atol=rtol=0`, 4 cases × 1 output `[1,1,2048,2048]`, RTX 3080
 * Ti):
 *
 * | Case    | maxAbs  | maxRel  | \|ref\| upper end | \|ref\| smallest nonzero |
 * | ------- | ------- | ------- | ----------------- | ------------------------ |
 * | checker | 3.48e-5 | 4.65e-6 | 13.710            | 2.86                     |
 * | disc    | 2.86e-4 | 8.44e-1 | 16.182            | 3.05e-5                  |
 * | noise   | 5.34e-5 | 1.06e-5 | 13.084            | 9.35e-1                  |
 * | ramp    | 1.39e-4 | 3.89e-5 | 12.868            | 1.43                     |
 *
 * atol 1.5e-3 is about 5.2 × the measured worst 2.86e-4 (disc). It is wider than the 5e-4 of 1024²
 * because at 2048² disc has more boundary pixels, and absolute error accumulates where \|ref\|
 * thins down to 3.05e-5 (the maxRel 0.844 is such a boundary pixel — its absolute error is only
 * 2.9e-4). **rtol is 0** (the same reason as {@link HR_F16_SYNTHETIC_TOLERANCE}). The binary-mask
 * disagreement is 0 on all 4 cases. No real-image goldens are taken at 2048² (the same reason as
 * {@link HR_F16_2048_SYNTHETIC_TOLERANCE}).
 */
const LUCIDA_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1.5e-3, rtol: 0 };

/**
 * The tolerances of **Lucida's f16 series** (`lucida-{1024,2048}-f16` — ADR 0113).
 *
 * 1024² was derived from measurement (Intel Arc B570, 2026-09-26, a plain comparison with
 * `atol=rtol=0`, 1 output):
 *
 * | Case           | maxAbs   | maxRel   |
 * | -------------- | -------- | -------- |
 * | checker        | 2.29e-5  | 3.87e-6  |
 * | disc           | 8.97e-5  | 3.38e-3  |
 * | noise          | 3.77e-5  | 1.72e-1  |
 * | ramp           | 5.34e-4  | 2.35e-1  |
 * | photo-portrait | 7.02e-3  | 4.46e-2  |
 * | photo-street   | 1.28e-3  | 5.46e-2  |
 * | photo-corridor | 3.48e-5  | 4.78e-6  |
 * | photo-landscape| 4.72e-5  | 7.16e-3  |
 *
 * The synthetic atol 2.5e-3 is about 4.7 × the worst 5.34e-4 (ramp), and the real-image atol 3e-2
 * about 4.3 × the worst 7.02e-3 (portrait) (the same derivation as the f32 series). The synthetic
 * worst is an order of magnitude above the f32 series (ramp at the 1.1e-4 level) because the
 * implementation error against goldens taken with rounded weights hits different rounding
 * boundaries with the f16 value layout (a guess — the quantisation error itself is in the golden
 * and does not show up here).
 *
 * TODO: 2048² **cannot be measured on the B570** (`HELD_SERIES` — deform_conv2d's job limit). The
 * same value as the f32 series' {@link LUCIDA_2048_SYNTHETIC_TOLERANCE} is a placeholder, to be
 * replaced by one derived from a plain comparison with `atol=rtol=0` on a machine that can run it.
 */
const LUCIDA_F16_SYNTHETIC_TOLERANCE: Tolerance = { atol: 2.5e-3, rtol: 0 };
const LUCIDA_F16_REAL_TOLERANCE: Tolerance = { atol: 3e-2, rtol: 0 };
const LUCIDA_F16_2048_SYNTHETIC_TOLERANCE: Tolerance = { atol: 1.5e-3, rtol: 0 };

/**
 * The fraction of pixels whose binary mask (`logit > 0` = `sigmoid > 0.5`) may disagree with torch.
 *
 * This check is needed apart from the output-side tolerance **because the artifact is a mask**. A
 * tolerance only says "the values are close", and a boundary pixel (logit ≈ 0) can flip its sign
 * while staying close. Conversely this alone cannot catch a regression of the values (an error in
 * the saturated region does not change the sign), so both are kept.
 *
 * Measured: **0 / 1,048,576 on all 2 f32 series × 8 cases** (4 synthetic + 4 real — for HR, values
 * taken on the retired f32 series, which apply as is to the f16 series whose outputs match the f32
 * series — the grounds are in {@link HR_F16_SYNTHETIC_TOLERANCE}). Lucida's 2 f16 series (ADR 0113)
 * await measurement and get the same limit provisionally. 0 is not used as the check as is because
 * the sign of a boundary pixel can move with another backend or driver — 1e-4 is 104 pixels at
 * 1024², an amount that does not show in the matte. An implementation bug flips tens of thousands
 * to hundreds of thousands of pixels, three or more orders of magnitude above this threshold. **It
 * is shared across series** because it is not a numeric quantity but one and the same claim, "the
 * decisions as a mask agree" (both series measure 0, and neither has a reason to loosen it).
 */
const MASK_DISAGREEMENT_LIMIT = 1e-4;

/** One series that runs (model × resolution). */
type Series = {
  /**
   * The series name (`birefnet.export.default_out_dir`): the golden directory
   * (`goldenDir(name)`), the prefix of the case IDs and the key of {@link HELD_SERIES}.
   */
  readonly name: string;
  /** The distribution mirror that ships this series' container, and its selection. */
  readonly mirror: { readonly repo: string } & Required<MirrorChoice>;
  /** The export command that regenerates this series' goldens (pasted into failure messages). */
  readonly generate: string;
  /**
   * The tolerance of the synthetic-image cases (**measured independently per series** — the module
   * docstring).
   */
  readonly tolerance: Tolerance;
  /** The tolerance of the real-image cases (likewise). */
  readonly realTolerance: Tolerance;
  /**
   * What should be declared as the container's **set of compressed storage dtypes** (empty for an
   * f32 series — the form of ADR 0029 decision 2). A mix-up of the series root and a forgotten
   * `--dtype` are invisible numerically, so this alone tells them apart.
   */
  readonly compressedStorage: readonly CompressedStorage[];
};

/**
 * The compressed storages of the vocabulary this family's series may hold (stage 1 goes up to f16 —
 * ADR 0113).
 */
type CompressedStorage = "f16";

/**
 * Whether a layout is a compressed storage counted by the storage check (**everything except** the
 * plain storages = f32 and the i32 index tables). It is written as an exclusion rather than an
 * enumeration so that a new compressed storage is not silently left out of the count when the
 * codec vocabulary gains a layout.
 */
const isCompressedLayout = (layout: CodecLayout): boolean => layout !== "f32" && layout !== "i32";

const EXPORT_PREFIX =
  "cd tools/export-recipes && uv run --group birefnet python -m birefnet.export";

/**
 * The series that run (model × resolution). **They are pinned here, not enumerated** — relying on
 * enumeration alone, tests silently vanish in an environment where only part of the generation ran,
 * and it becomes "green but unverified". 2048² holds no real-image goldens, so its real-image cases
 * SKIP (`realTolerance` formally shares the 1024² value but is never read).
 */
const SERIES: readonly Series[] = [
  {
    name: "lucida-1024",
    mirror: { repo: "karume-lucida", model: "1024", quant: "f32" },
    generate: `${EXPORT_PREFIX} --model-dir <repo>/inputs/birefnet/lucida --real-images`,
    tolerance: LUCIDA_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_REAL_TOLERANCE,
    compressedStorage: [],
  },
  {
    name: "lucida-2048",
    mirror: { repo: "karume-lucida", model: "2048", quant: "f32" },
    generate: `${EXPORT_PREFIX} --model-dir <repo>/inputs/birefnet/lucida --resolution 2048`,
    tolerance: LUCIDA_2048_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_REAL_TOLERANCE,
    compressedStorage: [],
  },
  {
    name: "birefnet-hr-1024-f16",
    mirror: { repo: "karume-birefnet-hr", model: "1024", quant: "f16" },
    generate: `${EXPORT_PREFIX} --dtype f16 --real-images`,
    tolerance: HR_F16_SYNTHETIC_TOLERANCE,
    realTolerance: HR_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
  {
    name: "birefnet-hr-2048-f16",
    mirror: { repo: "karume-birefnet-hr", model: "2048", quant: "f16" },
    generate: `${EXPORT_PREFIX} --dtype f16 --resolution 2048`,
    tolerance: HR_F16_2048_SYNTHETIC_TOLERANCE,
    realTolerance: HR_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
  {
    name: "lucida-1024-f16",
    mirror: { repo: "karume-lucida", model: "1024", quant: "f16" },
    generate: `${EXPORT_PREFIX} --model-dir <repo>/inputs/birefnet/lucida --dtype f16 --real-images`,
    tolerance: LUCIDA_F16_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
  {
    name: "lucida-2048-f16",
    mirror: { repo: "karume-lucida", model: "2048", quant: "f16" },
    generate: `${EXPORT_PREFIX} --model-dir <repo>/inputs/birefnet/lucida --dtype f16 --resolution 2048`,
    tolerance: LUCIDA_F16_2048_SYNTHETIC_TOLERANCE,
    realTolerance: LUCIDA_F16_REAL_TOLERANCE,
    compressedStorage: ["f16"],
  },
];

/**
 * Why the 2048² series are held on the Intel Arc B570 (shared by the rows of {@link HELD_SERIES}).
 */
const B570_JOB_TIMEOUT =
  "the one deform_conv2d dispatch at the end of the decoder (output [1,256,1024,1024]; expected ≈ 6.6 s from 1.66 s at 1024²)" +
  " exceeds the 5 s compute job limit of the Linux xe driver and loses the device, and Deno panics the whole process instead of" +
  ' raising an exception (stopping the verification that follows too — the "BiRefNet family" section of docs/limitations.md).' +
  " Remove this row once deform_conv2d is split or sped up";

/**
 * The series whose **GPU tests are held** per environment key. The outer key is a series name of
 * {@link SERIES}, the inner one an **environment key** (`<runtime>-<adapter name slug>` — ADR 0106
 * decision 2, the same approach as the reference rows).
 *
 * The rows are held per environment key because **the reason to hold exists only on that machine**.
 * Holding on every machine would silently drop the verification on the machines that can run it
 * (2048² is all green on the RTX 3080 Ti too). Even on a machine with a row, **the asset
 * completeness test (no GPU needed) runs** — only the tests that touch the device (golden
 * comparison, geometric discrimination) are held.
 *
 * This SKIP is independent of the gate of ADR 0005's "all cases SKIP = explicit FAIL"
 * (tests/gpu_gate_test.ts — it looks only at whether a GPU adapter exists) and **applies only to
 * the environments with a row**. A machine without a GPU has no environment key, so it hits no row
 * (there the GPU tests SKIP anyway). A held series announces itself once with `console.warn` at
 * registration (not a silent SKIP).
 *
 * MUST: add a row only when running it on that machine **takes the whole process down** (= drags
 * the following verification down with it), and write the grounds and "remove this row once it is
 * resolved" in the reason. This is not the place to hold a series whose numbers do not match (that
 * one is fixed while red).
 *
 * - The 3 series at 2048² (`birefnet-hr-2048-f16` / `lucida-2048` / `lucida-2048-f16`) on
 *   `deno-intel-graphics-bmg-g21` (Intel Arc B570, Linux xe driver): the "BiRefNet family" section
 *   of docs/limitations.md (measured 2026-09-20, ruled 2026-09-26). The f16 series are held too
 *   because the failing dispatch is deform_conv2d, whose weights stay in f32 storage even in the
 *   f16 series (not eligible), so the same dispatch runs whatever the storage dtype.
 */
const HELD_SERIES: Readonly<Record<string, Readonly<Record<string, { readonly reason: string }>>>> =
  {
    "lucida-2048": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
    "birefnet-hr-2048-f16": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
    "lucida-2048-f16": { "deno-intel-graphics-bmg-g21": { reason: B570_JOB_TIMEOUT } },
  };

/** The reason to hold on the machine of this run (`undefined` = not held). */
const heldReason = (series: Series): string | undefined => {
  const environment = ENVIRONMENT.key;
  if (environment === undefined) return undefined;
  if (!Object.hasOwn(HELD_SERIES, series.name)) return undefined;
  const byEnvironment = HELD_SERIES[series.name];
  if (!Object.hasOwn(byEnvironment, environment)) return undefined;
  return byEnvironment[environment].reason;
};

/** The graph name in the container = the manifest's weights key (container-v1 §2.1). */
const GRAPH = "matte";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";
/** The command that fetches the goldens (pasted into the skip warnings). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The golden directory of a series (`outputs/test-assets/goldens/<series>/`). */
const goldenRoot = (series: Series): URL => goldenDir(series.name);

/**
 * Whether `outputs/test-assets/` holds a complete fetch of the pinned revision. Then a golden that
 * is absent is a FAIL (the dataset at the pin lacks it), not a SKIP.
 */
const FETCHED = testAssetsFetched();
/**
 * The failure message of a test asset the case reads that is absent: from a complete fetch of the
 * pin (the dataset lacks it), or from a partial test-asset root.
 */
const absentAsset = (url: URL): string =>
  FETCHED
    ? `${url.pathname} is absent, but outputs/test-assets/ holds a complete fetch of ` +
      `${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks it`
    : `${url.pathname} is absent (fetch: ${FETCH_COMMAND})`;
/** Title suffixes of a case that SKIPs for want of an asset, so the test list says why. */
const NOT_FETCHED = " (SKIP: goldens not fetched)";
const MIRROR_ABSENT = " (SKIP: distribution mirror absent)";
/** The distribution mirror that ships the series' container (`models/<repo>/`). */
const mirrorOf = (series: Series): URL => mirrorRoot(series.mirror.repo);

/** Opens the series' `matte` container from its mirror (through the manifest — ADR 0139). */
const openMatte = async (series: Series): Promise<PreparedModel> => {
  const { repo, ...choice } = series.mirror;
  const opened = await openSelectedContainer(
    await selectMirrorComponent(mirrorRoot(repo), GRAPH, choice),
  );
  return prepareContainer(opened, GRAPH);
};

/**
 * The **synthetic-image** cases that should have been generated. The source of truth is
 * `build_cases` of `birefnet/export.py` (the same 4 images regardless of the model axis).
 */
const EXPECTED_CASES = ["checker", "disc", "noise", "ramp"] as const;

/**
 * The **real-image** cases (held only by an emit with `--real-images`). The source of truth is
 * `REAL_CASES` of `birefnet/export.py`. Only the golden case names are needed here; the mapping to
 * the source PNGs is held by `packages/models/tests/e2e_birefnet_real_test.ts` (this test does not
 * read the PNGs — the "composing two checks" of the module docstring).
 */
const REAL_CASES = ["photo-portrait", "photo-landscape", "photo-corridor", "photo-street"] as const;

/**
 * The synthetic case used for the geometric discrimination, and its disc radius (as a fraction of
 * the image's shorter side). The source of truth is `DISC_CASE` / `DISC_RADIUS` of
 * `birefnet/export.py` (there the same formula is applied to the torch output).
 */
const DISC_CASE = "disc";
const DISC_RADIUS = 0.3;

/**
 * Lists an asset directory. It degrades to empty only when the directory does not exist.
 * MUST: anything other than NotFound propagates — reading a permission error and the like as "no
 * assets" would turn a verification that did not run into a quietly green SKIP.
 */
const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return [...readDirSync(url)];
  } catch (cause) {
    if (isNotFound(cause)) {
      return [];
    }
    throw cause;
  }
};

const discoverCases = (root: URL): readonly string[] =>
  listDir(root)
    .filter(
      (entry) => entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await readFile(new URL(file, root));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

/**
 * Builds the golden inputs as views of the declared dtype (there are no symbolic dimensions, so no
 * explicit bindings either).
 */
const goldenInputs = (parsed: PreparedModel, io: SafetensorsFile): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} is missing from the golden`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return inputs;
};

/**
 * The static dimension of a graph input (there are no symbolic dimensions — `symbol_names=()` of
 * `birefnet/export.py`).
 */
const staticDim = (parsed: PreparedModel, axis: number): number => {
  const dim = parsed.graph.inputs[0].shape[axis];
  assert(
    typeof dim === "number",
    `axis ${axis} of pixel_values is the symbolic dimension '${String(dim)}'`,
  );
  return dim;
};

/**
 * The fraction of pixels whose binary mask (foreground = `logit > 0`) disagrees
 * ({@link MASK_DISAGREEMENT_LIMIT}).
 */
const maskDisagreement = (got: Float32Array, expected: Float32Array): number => {
  assertEquals(got.length, expected.length, "matte length");
  let differing = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (got[index] > 0 !== expected[index] > 0) differing += 1;
  }
  return differing / expected.length;
};

/**
 * The inside of the disc of the `disc` case (a `[S, S]` bool). **The same formula** as `disc_mask`
 * of `birefnet/export.py` (that one makes the image; this one cuts the real-GPU output with the
 * same disc).
 */
const discMask = (size: number): Uint8Array => {
  const axis = new Float64Array(size);
  for (let index = 0; index < size; index += 1) axis[index] = ((index + 0.5) / size) * 2 - 1;
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      mask[y * size + x] = axis[y] ** 2 + axis[x] ** 2 <= DISC_RADIUS ** 2 ? 1 : 0;
    }
  }
  return mask;
};

/** The mean logit inside / outside the disc (cut by {@link discMask}). */
const discMeans = (logits: Float32Array, size: number): { inside: number; outside: number } => {
  const mask = discMask(size);
  let inside = 0;
  let insideCount = 0;
  let outside = 0;
  let outsideCount = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (mask[index] === 1) {
      inside += logits[index];
      insideCount += 1;
    } else {
      outside += logits[index];
      outsideCount += 1;
    }
  }
  return { inside: inside / insideCount, outside: outside / outsideCount };
};

/** The asset state of one series (listed synchronously because it is needed at registration). */
type Discovery = {
  readonly cases: readonly string[];
  readonly realCases: readonly string[];
  /**
   * Whether any golden of the series is present. None and not a complete fetch = the goldens were
   * not fetched, so the golden cases and the completeness test SKIP; none in a complete fetch, or a
   * partial set, is a FAIL of the completeness test.
   */
  readonly available: boolean;
  /** The real-image group (absent from assets emitted without `--real-images`). */
  readonly realAvailable: boolean;
  /** Whether the golden of the geometry case (`io.disc`) is present. */
  readonly discAvailable: boolean;
  /** Whether the distribution mirror that ships the container is present (`karume/5`). */
  readonly mirror: boolean;
};

const realNames = new Set<string>(REAL_CASES);

const discover = (series: Series): Discovery => {
  const discovered = discoverCases(goldenRoot(series));
  const realCases = discovered.filter((name) => realNames.has(name));
  const available = discovered.length > 0;
  return {
    cases: discovered.filter((name) => !realNames.has(name)),
    realCases,
    available,
    realAvailable: available && realCases.length > 0,
    discAvailable: discovered.includes(DISC_CASE),
    mirror: mirrorPresent(mirrorOf(series)),
  };
};

const DISCOVERED: ReadonlyMap<string, Discovery> = new Map(
  SERIES.map((series) => [series.name, discover(series)]),
);

const discoveryOf = (series: Series): Discovery => {
  const found = DISCOVERED.get(series.name);
  if (found === undefined) throw new Error(`no listing for series ${series.name}`);
  return found;
};

for (const series of SERIES) {
  const found = discoveryOf(series);
  const held = heldReason(series);
  if (held !== undefined) {
    console.warn(`[karume] ${series.name} SKIPs in this environment (${ENVIRONMENT.key}): ${held}`);
  }
  if (!found.mirror) {
    console.warn(
      `[karume] The distribution mirror ${mirrorOf(series).pathname} is absent (or not karume/5), ` +
        `so the real-weight BiRefNet E2E of ${series.name} that runs the container is skipped ` +
        "(place the mirror of the published distribution under models/).",
    );
  }
  if (!found.available && !FETCHED) {
    console.warn(
      `[karume] The goldens ${goldenRoot(series).pathname} are absent, so the golden cases of ` +
        `${series.name} are skipped. Fetch: ${FETCH_COMMAND}`,
    );
  } else if (found.available && !found.realAvailable) {
    console.warn(
      `[karume] the real-image cases of ${series.name} SKIP (golden ${found.realCases.length}/` +
        `${REAL_CASES.length}). Generate: ${series.generate}`,
    );
  }
}

/**
 * The format of an environment key (the spelling `environmentKey` of `helpers/environment.ts`
 * produces — `<runtime>-<slug>`, where the slug joins lowercase alphanumerics with single `-` and
 * has no `-` at either end).
 */
const ENVIRONMENT_KEY_FORMAT = /^(deno|chrome)-[a-z0-9]+(-[a-z0-9]+)*$/;

test("BiRefNet per-environment SKIP table: the rows point at series that run, and the inner keys have the environment-key format", () => {
  // A misspelled series name silently disables the row (a series meant to be held runs and takes
  // the process down).
  const names = new Set(SERIES.map((series) => series.name));
  assertEquals(
    Object.keys(HELD_SERIES).filter((name) => !names.has(name)),
    [],
  );
  // A misspelled environment key (uppercase, a trademark sign, leftover whitespace) likewise
  // silently disables the row — `environmentKey` produces nothing outside this format, so a row
  // outside it hits no machine.
  const malformed = Object.entries(HELD_SERIES).flatMap(([series, byEnvironment]) =>
    Object.keys(byEnvironment)
      .filter((key) => !ENVIRONMENT_KEY_FORMAT.test(key))
      .map((key) => `${series} / ${key}`),
  );
  assertEquals(malformed, []);
});

/**
 * Where the outcomes and measurements go (`outputs/verify/<environment
 * key>/<date>_birefnet-golden/` — safe to delete).
 *
 * The series name is `<family>-golden` because the models-side e2e (`birefnet`) writes to the same
 * root (two modules writing to the same directory overwrite each other's `cases`).
 */
const results = openResults("birefnet-golden");

for (const series of SERIES) {
  const found = discoveryOf(series);
  const root = goldenRoot(series);
  /**
   * Whether this machine holds the series' GPU tests ({@link HELD_SERIES} — not applied to the
   * asset completeness test).
   */
  const held = heldReason(series) !== undefined;

  // The completeness test covers the goldens only: the container moved to the mirror, whose parts
  // the distribution gate checks (packages/runtime/tests/distribution_gate_test.ts). It sleeps only
  // on a golden directory that is entirely absent while the test-asset root is not a complete fetch
  // of the pin; absent from a complete fetch, the directory FAILs here.
  const goldensMissing = !found.available && !FETCHED;
  test.skipIf(goldensMissing)(
    `BiRefNet goldens: ${series.name} — the expected cases are present${goldensMissing ? NOT_FETCHED : ""}`,
    () => {
      assert(found.available, absentAsset(root));
      assertEquals(
        found.cases,
        [...EXPECTED_CASES],
        `synthetic-image golden cases of ${root.pathname}`,
      );
      // The real images are **optional but all or nothing** (an emit with `--real-images` writes
      // all 4 together). Rounding a partial absence to a SKIP would let assets that broke midway
      // through a retake pass silently.
      assert(
        found.realCases.length === 0 || found.realCases.length === REAL_CASES.length,
        `${root.pathname} has ${found.realCases.length}/${REAL_CASES.length} real-image goldens` +
          ` (retake: ${series.generate})`,
      );
    },
  );

  test.skipIf(!found.mirror)(
    `BiRefNet storage: ${series.name} — the container's set of compressed storage dtypes matches the series' declaration` +
      (found.mirror ? "" : MIRROR_ABSENT),
    async () => {
      // The storage dtype of the series and of the assets agree (the only detector of a root mix-up
      // / a forgotten `--dtype` — {@link Series.compressedStorage}). It looks at the **set** rather
      // than the count so that an f16 series with another compressed storage mixed in does not pass
      // as "there is at least one compression" (ADR 0029 decision 2, the same form as e2e_sbv2). On
      // the mirror this also checks that the manifest's quant label names the container's storage
      // (the f16 quant of the mirror must hold f16 storage, the f32 quant none).
      const parsed = await openMatte(series);
      const compressed = [
        ...new Set(
          Object.values(parsed.graph.initializers)
            .flatMap((initializer) =>
              initializer.storage === undefined ? [] : [codecLayout(initializer.storage.codec)],
            )
            .filter(isCompressedLayout),
        ),
      ].sort();
      assertEquals(
        compressed,
        [...series.compressedStorage].sort(),
        `${series.name}: the set of compressed storage dtypes disagrees with the series (retake: ${series.generate})`,
      );
    },
  );

  /**
   * The cases the comparison runs. Synthetic and real images **build their input the same way**
   * (both are the golden's `pixel_values`); what differs is only the distribution they exercise and
   * the tolerance derived from it independently.
   */
  const goldenCases = [
    ...found.cases.map((name) => ({
      name,
      label: "golden comparison",
      tolerance: series.tolerance,
      ignore: !found.available,
    })),
    ...found.realCases.map((name) => ({
      name,
      label: "real-image golden comparison",
      tolerance: series.realTolerance,
      ignore: !found.realAvailable,
    })),
  ];

  for (const entry of goldenCases) {
    const caseName = entry.name;
    /** The case ID (the same case name exists in other series, so it is held as a pair). */
    const caseId = `${series.name}/${caseName}`;
    // The whole case is upstream parity (model-exit torch goldens under a tolerance, plus the mask
    // agreement with torch), so it is opt-in. Its io-key / shape / dtype assertions only check the
    // golden file against the graph as preconditions of that comparison; the default run keeps the
    // output shape / dtype check per series in the geometry case below.
    test.skipIf(entry.ignore || !found.mirror || !GPU_AVAILABLE || held || !UPSTREAM_PARITY)(
      `BiRefNet ${entry.label}: ${series.name} / ${caseName} (real GPU / torch CPU expected values)${UPSTREAM_PARITY_OPT_IN}` +
        (found.mirror ? "" : MIRROR_ABSENT),
      async () => {
        await runRecordedCase(results, { id: caseId }, async ({ measurements }) => {
          const [parsed, ioBytes] = await Promise.all([
            openMatte(series),
            readBuffer(root, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
          ]);
          const io = parseSafetensors(ioBytes);

          // Every tensor of io corresponds exactly to the graph's inputs and outputs (nothing
          // extra, nothing missing).
          const expectedKeys = [
            ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
            ...parsed.graph.outputs.map((_, index) => `output.${index}`),
          ].sort();
          assertEquals(
            [...io.tensors.keys()].sort(),
            expectedKeys,
            "tensor keys of io.safetensors",
          );

          const inputs = goldenInputs(parsed, io);

          const gpu = await acquireGpu();
          // MUST: the device is destroyed on failures of the adapter check and `createSession` too.
          // Missing that, the rest of the run carries on holding a device that is never destroyed,
          // and later tests turn red with OOM (the delayed release of known-issues).
          try {
            // This route keeps reference values and results as this machine's rows, so check here
            // that the adapter the key was taken from is the running adapter (mixing them up on a
            // multi-GPU machine would measure against another machine's band).
            assertAdapterMatchesEnvironment(gpu);
            const session = await parsed.createContainerSession(gpu);
            try {
              const outputs = await session.run(inputs);
              assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

              const [name] = parsed.graph.outputs;
              const view = io.tensors.get("output.0");
              assert(view !== undefined, "output.0 is missing from the golden");
              const where = `${series.name} / ${caseName} output.0 ('${name}')`;
              const declared = parsed.graph.values[name].dtype;
              assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
              assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
              const expected = ioTensor(io, view, declared);
              const report = compareTensors(outputs[name], expected, entry.tolerance);
              measurements.push({
                output: name,
                maxAbs: report.maxAbsError,
                maxRel: report.maxRelError,
                tolerance: entry.tolerance,
                stage: "karume",
              });
              assert(report.pass, `${where}: ${formatAllclose(report)}`);

              // Apart from the closeness of the values, **the decisions as a mask** agree with
              // torch.
              assert(
                outputs[name].dtype === "f32" && expected.dtype === "f32",
                `${where}: not f32`,
              );
              const disagreement = maskDisagreement(outputs[name].data, expected.data);
              assert(
                disagreement <= MASK_DISAGREEMENT_LIMIT,
                `${where}: binary-mask disagreement ${(disagreement * 100).toFixed(4)}%` +
                  ` (limit ${MASK_DISAGREEMENT_LIMIT * 100}%)`,
              );
            } finally {
              await session.dispose();
            }
          } finally {
            gpu.destroy();
          }
        });
      },
    );
  }

  /**
   * The geometry case reads `io.disc`: absent and not fetched = SKIP, absent from a fetch = FAIL.
   */
  const discMissing = !found.discAvailable && !FETCHED;
  test.skipIf(discMissing || !found.mirror || !GPU_AVAILABLE || held)(
    `BiRefNet geometric discrimination: ${series.name} — the mean logit inside the disc exceeds that outside` +
      (discMissing ? NOT_FETCHED : "") +
      (found.mirror ? "" : MIRROR_ABSENT),
    async () => {
      assert(
        found.discAvailable,
        absentAsset(new URL(`${IO_PREFIX}${DISC_CASE}${IO_SUFFIX}`, root)),
      );
      // A golden comparison alone only says "it matches the expected values"; whether the output is
      // meaningful as a matte is another matter (a uniformly collapsed output passes if the
      // expected values collapsed the same way). This one cuts by **a known geometry in the image**
      // (the disc), so a uniform output fails as the means line up. No threshold (the order itself
      // is what is checked — the same form as `_sanity` of `birefnet/export.py`, which is applied
      // on the torch side). Measured inside / outside the disc: +10.96 / −9.64 on BiRefNet_HR and
      // +3.37 / −9.69 on Lucida (the depth of the response to the synthetic images differs per
      // series).
      const parsed = await openMatte(series);
      const ioBytes = await readBuffer(root, `${IO_PREFIX}${DISC_CASE}${IO_SUFFIX}`);
      const io = parseSafetensors(ioBytes);
      const size = staticDim(parsed, 3);
      assertEquals(size, staticDim(parsed, 2), "the disc discrimination assumes a square input");

      const gpu = await acquireGpu();
      // MUST: the device is destroyed on a failure of `createSession` too. Missing that, the rest
      // of the run carries on holding a device that is never destroyed, and later tests turn red
      // with OOM (the delayed release of known-issues).
      try {
        const session = await parsed.createContainerSession(gpu);
        try {
          const [name] = parsed.graph.outputs;
          const output = (await session.run(goldenInputs(parsed, io)))[name];
          assert(output.dtype === "f32", `the dtype of disc's matte is ${output.dtype}`);
          assertEquals(output.shape, [1, 1, size, size], "matte shape");
          const { inside, outside } = discMeans(output.data, size);
          assert(
            inside > outside,
            `disc's mean logit inside the disc ${inside} is not above the outside ${outside} — the salient object is not separated`,
          );
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    },
  );
}

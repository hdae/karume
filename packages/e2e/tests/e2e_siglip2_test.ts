// Real-GPU golden E2E of the real-weight SigLIP2 **vision tower** (stage 3 of ADR 0005).
//
// The tiny golden (tests/e2e_golden_test.ts) covers "the coverage of the op contracts", the
// real-weight SBV2 / Irodori cover "real weights on the audio-chain side", and EmbeddingGemma
// (tests/e2e_embeddinggemma_test.ts) covers "text models with a single-vector output"; this one
// covers **image models with a single-vector output**.
//
// The container comes from the distribution mirror `models/karume-siglip2/` (model `base` /
// `so400m`, quant `f32`, graph `vision`, read through the manifest) and the goldens from the
// test-asset root (`goldenDir(<series>)`, fetched with `node tools/test-assets/fetch.ts`) — ADR
// 0139. The goldens are recipe outputs of `tools/export-recipes/siglip2/export.py` (the command is
// {@link realGenerateCommand}). Each case requires the mirror plus exactly the goldens it reads. A
// golden that is absent SKIPs only while the test-asset root is not a complete fetch of the pin (the
// assets gate is red then); absent from a complete fetch, it FAILs — the dataset at the pin lacks
// it. An absent mirror SKIPs (the distribution gate owns that red).
//
// There is **one series per model** (SERIES below). base (`patch16-224` — hidden 768 / 12 layers /
// 196 patches) and so400m (`patch14-384` — hidden 1152 / 27 layers / 729 patches) are a small and
// a large point of the same path, with the same graph shape and op table. **Only the tolerance is
// derived from measurements per series** (one's measurement does not pass the other — another
// order of magnitude means another error). There is only one storage-dtype series, f32, so there
// is no axis for that.
//
// The graph is the vision tower alone, and so is the output: **only pooler_output (`[1,hidden]`
// through the MAP head)** (neither the text tower nor `last_hidden_state` is included — the
// docstring of export_siglip2.py). The input is one **normalized** `pixel_values f32
// [1,3,resolution,resolution]`, with no symbolic dimension (the resolution and the patch count are
// fixed).
//
// ## Two groups of goldens (synthetic images + real images) — both are kept
//
// The input of **both groups is the very `pixel_values` baked into the golden** (bit-identical),
// so the comparison sees **only the runtime's numerical error**. They are kept as two groups
// because they exercise different distributions:
//
// - **4 synthetic-image cases** (`ramp` / `ramp-dim` / `checker` / `noise`): they hit the edges of
//   the value range and gradients, so they catch numerical regressions sharply (measured 6.4e-6 to
//   1.05e-5).
// - **4 real-image cases** (`photo-*`): they cover the runtime fidelity at the distribution point
//   of natural images, and the **discrimination** of whether the output means something as an
//   embedding (the cosine order — {@link REAL_PERSON_CASES}). The error is of the same order as the
//   synthetic side, but the tolerance is derived from measurements independently per group and per
//   series ({@link REAL_BASE_TOLERANCE}).
//
// ## The chain including the TS preprocessing is held by the **composition** of two checks
//
// The chain "pass a PNG and get the same embedding as Python" is held not by this test alone but by
// the composition of the **input-side parity check** of
// `packages/models/tests/e2e_siglip2_image_preprocess_real_test.ts` (the same PNG gives the same
// `pixel_values` as Python — an input difference ≤ 7.85e-3 = about one uint8 LSB) and this test's
// **fidelity on the golden input**. The preprocessing is not run here because of the dependency
// direction (a runtime test importing the models implementation by a relative path is backwards).
// As a side benefit of the split, when one fails, the test name tells whether the preprocessing or
// the inference moved.
//
// Where the assets are absent it **SKIPs explicitly** (independently per series — in an environment
// that generated only one, the present one still runs). The real-image group also SKIPs
// independently (assets emitted without `--real-images` have only the 4 synthetic cases). ADR
// 0005's "all-SKIP is an explicit FAIL" guard (tests/gpu_gate_test.ts) looks only at *whether a GPU
// adapter exists*, independent of this SKIP. Conversely, assets that are **partially** present
// (missing cases) FAIL rather than SKIP (the "asset completeness" test below) — that would be a
// silent apparent success.

import { type DirEntry, isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
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
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
import { GPU_AVAILABLE } from "../../runtime/tests/helpers/gpu.ts";
import {
  f32ArtifactBytes,
  openReferences,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleOrObserve,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";

/**
 * The tolerance for comparing the real-weight SigLIP2 **base** (`patch16-224`) against the torch
 * CPU expected values.
 *
 * Measured (raw comparison with `atol=rtol=0`, 4 cases × 1 output `[1,768]`):
 *
 * | Case     | maxAbs  | maxRel  | \|ref\| max  | \|ref\| min nonzero |
 * | -------- | ------- | ------- | ------------ | ------------------- |
 * | checker  | 1.05e-5 | 7.45e-3 | 6.842        | 2.04e-4            |
 * | noise    | 9.54e-6 | 6.11e-3 | 7.147        | 1.02e-4            |
 * | ramp     | 6.44e-6 | 4.13e-3 | 6.909        | 1.70e-4            |
 * | ramp-dim | 1.05e-5 | 3.03e-3 | 6.950        | 2.36e-4            |
 *
 * **atol drives the decision** (`|x−y| ≤ atol + rtol·|ref|`): pooler_output is a raw vector, not
 * L2-normalized, whose values straddle 0 and thin out down to a \|ref\| min nonzero of 1.02e-4, so
 * making rtol the main term would diverge near 0 (even the element with the measured maxRel
 * 7.45e-3 has an absolute error of only `7.45e-3 × 1.4e-3 ≈ 1.0e-5`). rtol 1e-6 contributes 7.1e-6
 * even at the \|ref\| max of 7.147, 1/7 of atol, so it does not drive the decision.
 *
 * atol 5e-5 is about 4.8× the worst measured 1.05e-5 (checker / ramp-dim). **It is 2 orders of
 * magnitude looser than EmbeddingGemma's `EMBEDDINGGEMMA_TOLERANCE` (atol 1e-6) only because of the
 * value range** — that output is an L2-normalized unit vector (\|ref\| max 0.25), and the error
 * relative to the value range is about 1.5e-6 for both (1.05e-5 / 7.15 ≈ 1.5e-6 vs 2.8e-7 / 0.25 ≈
 * 1.1e-6). The sources of error are the same too (fma fusion, a linear / attention reduction order
 * that differs from torch, implementation differences of the transcendental functions (tanh / exp /
 * rsqrt)), and all 13 SDPAs go through the decomposed path (mul×2 + bmm + softmax + bmm).
 *
 * NOTE: the difference from the patch that makes the MAP head's q/k/v explicit (measured 7.75e-7 to
 * 2.38e-6 by `export_siglip2.py --verify`) **is not included here** — the golden's expected values
 * are taken from the module **after** the replacement, and this comparison sees only the runtime's
 * error.
 *
 * The error of implementation bugs (mixed-up channels in the patch embedding, a wrong order of the
 * position embeddings, a wrong attention head split, a mixed-up MAP head probe) is O(1) to O(7),
 * the same as the output's value range, 5 or more orders of magnitude above this threshold.
 */
const BASE_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * The tolerance for comparing the real-weight SigLIP2 **so400m** (`patch14-384`) against the torch
 * CPU expected values.
 *
 * MUST: derived from measurements **independently** of base (raw comparison with `atol=rtol=0`, 4
 * cases × 1 output `[1,1152]`). On the same path, hidden 1152 / 27 layers / 729 patches give other
 * reduction lengths, and the error accumulates differently:
 *
 * | Case     | maxAbs  | maxRel  | \|ref\| max  | \|ref\| min nonzero |
 * | -------- | ------- | ------- | ------------ | ------------------- |
 * | checker  | 8.46e-6 | 4.64e-3 | 10.389       | 1.00e-4            |
 * | noise    | 1.05e-5 | 9.76e-3 | 8.451        | 1.28e-4            |
 * | ramp     | 6.68e-6 | 2.55e-2 | 10.753       | 4.68e-6            |
 * | ramp-dim | 6.08e-6 | 8.35e-3 | 10.874       | 1.89e-4            |
 *
 * atol 5e-5 is about 4.8× the worst measured 1.05e-5 (noise). **It came out the same as base as a
 * consequence of the measurements**, and it is not shared (re-measuring one would silently loosen
 * the other). atol drives the decision for the same reason as base — the \|ref\| min nonzero thins
 * out down to 4.68e-6 (ramp), and even the element with maxRel 2.55e-2 has an absolute error of
 * only 1.2e-7. rtol 1e-6 contributes 1.1e-5 even at the \|ref\| max of 10.874, 1/5 of atol, so it
 * does not drive the decision.
 *
 * NOTE: as for base, the difference from the patch that makes the MAP head's q/k/v explicit
 * (so400m measured 1.07e-6 to 2.86e-6 by `--verify`; the shape folding is bit-identical in all 4
 * cases) is not included here.
 */
const SO400M_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 1e-6 };

/**
 * The tolerance for comparing the **real-image** cases (`photo-*`) — SigLIP2 **base**.
 *
 * As for the synthetic images, the input is the very `pixel_values` baked into the golden, so only
 * the runtime error shows, of the same order as the synthetic side. It is still **not shared** with
 * {@link BASE_TOLERANCE} because the distribution exercised differs and so does the error (not
 * letting a re-measurement of one silently loosen the other is the same reason as not sharing
 * across series).
 *
 * Measured (raw comparison with `atol=rtol=0`, 4 cases × 1 output `[1,768]`):
 *
 * | Case            | maxAbs  | maxRel  | \|ref\| max  | \|ref\| min nonzero |
 * | --------------- | ------- | ------- | ------------ | ------------------- |
 * | photo-portrait  | 3.34e-6 | 2.16e-3 | 5.314        | 8.77e-5            |
 * | photo-landscape | 1.34e-5 | 6.30e-3 | 4.550        | 1.82e-4            |
 * | photo-corridor  | 3.82e-6 | 1.73e-2 | 6.727        | 7.22e-5            |
 * | photo-street    | 5.72e-6 | 2.49e-3 | 5.319        | 1.55e-4            |
 *
 * atol 5e-5 is about 3.7× the worst measured 1.34e-5 (photo-landscape). That it came out the same
 * as the synthetic side is a consequence of the measurements, not a reason to share it.
 *
 * rtol is **0** (not even the synthetic side's 1e-6). The worst measured 1.34e-5 fits within atol
 * alone, and adding rtol 1e-6 would contribute 6.7e-6 at the \|ref\| max of 6.727 = 1/7 of atol,
 * not moving the decision — what does not move the decision is not set. maxRel reaching 1.73e-2 is
 * an artifact of elements near 0, whose absolute error is only 3.8e-6 (the same reasoning as the
 * synthetic side's docstring).
 *
 * The error of runtime-side implementation bugs (those {@link BASE_TOLERANCE} lists) is O(1) to
 * O(7), the same as the output's value range, 5 or more orders of magnitude above this threshold.
 * On the other hand, **errors on the TS preprocessing side** (a wrong resolution, reversed channel
 * order, a wrong interpolation) **are not caught by this check** — the input is the golden's
 * `pixel_values`. That is covered by
 * `packages/models/tests/e2e_siglip2_image_preprocess_real_test.ts` ("the composition of two
 * checks" at the top of the file).
 */
const REAL_BASE_TOLERANCE: Tolerance = { atol: 5e-5, rtol: 0 };

/**
 * The tolerance of the **real-image** cases — SigLIP2 **so400m**. MUST: derived from measurements
 * independently of base.
 *
 * | Case            | maxAbs  | maxRel  | \|ref\| max  | \|ref\| min nonzero |
 * | --------------- | ------- | ------- | ------------ | ------------------- |
 * | photo-portrait  | 9.06e-6 | 2.85e-3 | 5.845        | 3.24e-4            |
 * | photo-landscape | 7.63e-6 | 1.20e-2 | 5.883        | 3.19e-4            |
 * | photo-corridor  | 9.54e-6 | 1.42e-3 | 6.584        | 1.47e-4            |
 * | photo-street    | 3.34e-5 | 3.95e-3 | 5.329        | 4.22e-4            |
 *
 * atol 1e-4 is about 3.0× the worst measured 3.34e-5 (photo-street). **It is looser than the
 * synthetic side's {@link SO400M_TOLERANCE} (5e-5) as a consequence of the measurements** — only
 * photo-street comes out at 3× the worst of the 4 synthetic cases (1.05e-5), which is itself the
 * measurement showing that a natural-image distribution point can land on the worse side than
 * synthetic images on a path with long reductions (hidden 1152 / 27 layers / 729 patches). It is
 * also why it cannot be reused for base (worst measured 1.34e-5).
 *
 * rtol is 0 for the same reason as base (rtol 1e-6 contributes 6.6e-6 even at the \|ref\| max of
 * 6.584, 1/15 of atol).
 */
const REAL_SO400M_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 0 };

/**
 * The declaration of one series (= model). The series name is the directory name of the exporter's
 * `--model-dir`.
 */
type Series = {
  /**
   * The series name: the golden directory (`goldenDir(name)`) and the prefix of the sha256 case
   * IDs (frozen — ADR 0139 decision 7).
   */
  readonly name: string;
  /** The model of the distribution mirror `models/karume-siglip2/` that ships this series. */
  readonly model: string;
  /**
   * The quant of that model: the storage dtype the series used. Pinned, not left to the manifest's
   * default, so a new default quant cannot change the container these cases and their sha256 rows
   * test.
   */
  readonly quant: string;
  readonly tolerance: Tolerance;
  /**
   * The tolerance of the real-image cases (derived from measurements independently of the synthetic
   * images — {@link REAL_BASE_TOLERANCE}).
   */
  readonly realTolerance: Tolerance;
};

/**
 * The series that run. **Fixed here rather than enumerated** — picking up whatever was generated
 * would give "green but unchecked" in an environment that forgot to generate a whole series.
 */
const SERIES: readonly Series[] = [
  {
    name: "siglip2-base-patch16-224",
    model: "base",
    quant: "f32",
    tolerance: BASE_TOLERANCE,
    realTolerance: REAL_BASE_TOLERANCE,
  },
  {
    name: "siglip2-so400m-patch14-384",
    model: "so400m",
    quant: "f32",
    tolerance: SO400M_TOLERANCE,
    realTolerance: REAL_SO400M_TOLERANCE,
  },
];

/** The distribution mirror that ships both models. */
const MIRROR = mirrorRoot("karume-siglip2");
/** The graph name in the container = the manifest's weights key (container-v1 §2.1). */
const GRAPH = "vision";
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

/** Opens the series' `vision` container from the mirror (through the manifest — ADR 0139). */
const openVision = async (series: Series): Promise<PreparedModel> =>
  prepareContainer(
    await openSelectedContainer(
      await selectMirrorComponent(MIRROR, GRAPH, { model: series.model, quant: series.quant }),
    ),
    GRAPH,
  );

/**
 * The export command that regenerates the goldens, real-image cases included
 * (`tools/export-recipes/siglip2/export.py`; `--real-images` needs the `siglip2-preprocess` group).
 * The series name is also the directory name of the real weights (both are the HF repository name).
 */
const realGenerateCommand = (series: Series): string =>
  "cd tools/export-recipes && uv run --group siglip2-preprocess python -m siglip2.export" +
  ` --real-images --model-dir ../../inputs/siglip2/${series.name}`;

/**
 * The **synthetic-image** cases that should have been generated. **Fixed here rather than
 * enumerated** — relying on an enumeration alone, tests would silently vanish in an environment
 * that ran only part of the generation, giving "green but unchecked". The source of truth is
 * `build_cases` in `tools/exporter/export_siglip2.py`.
 */
const EXPECTED_CASES = ["checker", "noise", "ramp", "ramp-dim"] as const;

/**
 * The **real-image** cases (only an emit with `--real-images` has them). The source of truth is
 * `REAL_CASES` in `export_siglip2.py`. Only the golden case names are needed here; the mapping to
 * the source PNGs is held by `packages/models/tests/e2e_siglip2_image_preprocess_real_test.ts`
 * (this test does not read PNGs — "the composition of two checks" at the top of the file).
 */
const REAL_CASES = ["photo-portrait", "photo-landscape", "photo-corridor", "photo-street"] as const;

/**
 * The two groups the real-image discrimination looks at (2 images with a person / 2 without). The
 * source of truth is `REAL_PERSON_CASES` / `REAL_SCENE_CASES` in `export_siglip2.py`; that one is
 * applied to the torch output, this one to the **real-GPU output**.
 *
 * NOTE: the synthetic-image pairs (`ramp`×`ramp-dim` vs `ramp`×`checker`) are not here — the
 * discrimination was **moved** to the real images ("is there a person" is a more meaningful
 * discrimination than the strength of the same structure, and the synthetic side stays as the
 * numerical-regression check). On every emit, `_sanity` in `export_siglip2.py` applies the torch-
 * side order check to both the synthetic and the real images.
 */
const REAL_PERSON_CASES = ["photo-portrait", "photo-street"] as const;
const REAL_SCENE_CASES = ["photo-landscape", "photo-corridor"] as const;

/**
 * Lists an asset directory. Only a missing directory degrades to empty.
 * MUST: anything other than NotFound propagates — reading a permission error or the like as "no
 * asset" would quietly turn a check that never ran green as a SKIP.
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
 * Builds the golden's inputs as views of the declared dtype (no symbolic dimension, so no explicit
 * bindings either).
 */
const goldenInputs = (parsed: PreparedModel, io: SafetensorsFile): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const spec of parsed.graph.inputs) {
    const view = io.tensors.get(`input.${spec.name}`);
    assert(view !== undefined, `input.${spec.name} is not in the golden`);
    inputs[spec.name] = ioTensor(io, view, spec.dtype);
  }
  return inputs;
};

const cosine = (first: Float32Array, second: Float32Array): number => {
  let dot = 0;
  let firstNorm = 0;
  let secondNorm = 0;
  for (let i = 0; i < first.length; i += 1) {
    dot += first[i] * second[i];
    firstNorm += first[i] * first[i];
    secondNorm += second[i] * second[i];
  }
  return dot / Math.sqrt(firstNorm * secondNorm);
};

/**
 * Where the verdicts and measurements go (`outputs/verify/<environment key>/<date>_siglip2-golden/`
 * — safe to delete).
 *
 * The series name is `<family>-golden` because the models-side e2e writes to the same root under
 * the bare family name (two modules writing to the same slot would overwrite each other's `cases`).
 */
const results = openResults("siglip2-golden");

/**
 * The sha256 reference values of the pooled output (rows per environment — ADR 0106; the row ID is
 * `<series>/<case>`, the same as the results).
 *
 * The tolerance comparison only states closeness to torch **across machines**; a regression that
 * moved a value by one bit on this machine (a change of reduction order or fma fusion) hides inside
 * the band. The sha rows are an additional check that catches that, and they do not replace the
 * tolerance comparison.
 */
const references = openReferences(
  new URL("../../runtime/tests/fixtures/references/siglip2-golden.json", import.meta.url),
);

/**
 * The IDs of the registered golden cases (the set the reference gate and `warnMissing` count —
 * fixed synchronously at registration). Only cases of series with assets go in, so empty = no
 * series has assets.
 */
const caseIds: string[] = [];

/** Whether the distribution mirror is present (`karume/5`). */
const MIRROR_PRESENT = mirrorPresent(MIRROR);
if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] The distribution mirror ${MIRROR.pathname} is absent (or not karume/5), so the ` +
      "real-weight SigLIP2 vision E2E is skipped (place the published distribution under models/).",
  );
}

for (const series of SERIES) {
  const root = goldenRoot(series);
  /** Enumerated synchronously, as registration needs it (the same reason as `test.skipIf`). */
  const discovered = discoverCases(root);
  const realNames = new Set<string>(REAL_CASES);
  const cases = discovered.filter((name) => !realNames.has(name));
  const realCases = discovered.filter((name) => realNames.has(name));
  /**
   * Whether any golden is present. None and not a complete fetch = the goldens were not fetched, so
   * every case of the series and the completeness test SKIP; none in a complete fetch, or a partial
   * set, is a FAIL of the completeness test.
   */
  const available = discovered.length > 0;
  /** The real-image group (absent from assets emitted without `--real-images`). */
  const realAvailable = available && realCases.length > 0;
  const goldensMissing = !available && !FETCHED;
  /** The discrimination reads the real-image goldens: absent from a complete fetch, it FAILs. */
  const realMissing = !realAvailable && !FETCHED;

  if (goldensMissing) {
    console.warn(
      `[karume] The goldens ${root.pathname} are absent, so the real-weight SigLIP2 vision E2E ` +
        `of ${series.name} is skipped. Fetch: ${FETCH_COMMAND}`,
    );
  } else if (available && !realAvailable) {
    console.warn(
      `[karume] skipping the real-image cases (${series.name}) (goldens ${realCases.length}/` +
        `${REAL_CASES.length}). Generate: ${realGenerateCommand(series)}`,
    );
  }

  // The completeness test covers the goldens only: the container moved to the mirror, whose parts
  // the distribution gate checks (packages/runtime/tests/distribution_gate_test.ts). It sleeps only
  // on a golden directory that is entirely absent while the test-asset root is not a complete fetch
  // of the pin; absent from a complete fetch, the directory FAILs here.
  test.skipIf(goldensMissing)(
    `SigLIP2 goldens: ${series.name} — the expected cases are present${goldensMissing ? NOT_FETCHED : ""}`,
    () => {
      assert(available, absentAsset(root));
      assertEquals(cases, [...EXPECTED_CASES], `${root.pathname}: synthetic-image golden cases`);
      // The real images are **optional, but all or none** (an emit with `--real-images` writes all
      // 4 together). Rounding a partial gap to SKIP would let assets lost mid-regeneration pass
      // silently.
      assert(
        realCases.length === 0 || realCases.length === REAL_CASES.length,
        `${root.pathname} has ${realCases.length}/${REAL_CASES.length} real-image goldens` +
          ` (regenerate: ${realGenerateCommand(series)})`,
      );
    },
  );

  /**
   * The cases that run the comparison. Synthetic and real images **build the input the same way**
   * (both from the golden's `pixel_values`); only the distribution exercised and the tolerance
   * derived independently from it differ.
   */
  const goldenCases = [
    ...cases.map((name) => ({
      name,
      label: "golden comparison",
      tolerance: series.tolerance,
      ignore: !available,
    })),
    ...realCases.map((name) => ({
      name,
      label: "real-image golden comparison",
      tolerance: series.realTolerance,
      ignore: !realAvailable,
    })),
  ];

  for (const entry of goldenCases) {
    const caseName = entry.name;
    /** The case ID (another series has the same case names, so it is kept as a pair). */
    const caseId = `${series.name}/${caseName}`;
    if (!entry.ignore && MIRROR_PRESENT) caseIds.push(caseId);
    test.skipIf(entry.ignore || !MIRROR_PRESENT || !GPU_AVAILABLE)(
      `SigLIP2 ${entry.label}: ${series.name} / ${caseName} (real GPU / torch CPU expected values)` +
        (MIRROR_PRESENT ? "" : MIRROR_ABSENT),
      async () => {
        /**
         * The sha settlement (only when compared — carried outside so a mismatch fails after the
         * recording).
         */
        let settlement: ReferenceSettlement | undefined;
        await runRecordedCase(results, { id: caseId }, async ({ measurements }) => {
          const [parsed, ioBytes] = await Promise.all([
            openVision(series),
            readBuffer(root, `${IO_PREFIX}${caseName}${IO_SUFFIX}`),
          ]);
          const io = parseSafetensors(ioBytes);

          // Every tensor of io maps exactly onto the graph's inputs and outputs (nothing extra,
          // nothing missing).
          const expectedKeys = [
            ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
            ...parsed.graph.outputs.map((_, index) => `output.${index}`),
          ].sort();
          assertEquals([...io.tensors.keys()].sort(), expectedKeys, "io.safetensors tensor keys");

          const inputs = goldenInputs(parsed, io);

          const gpu = await acquireGpu();
          // MUST: the device is destroyed even when the adapter check or `createSession` fails.
          // Missing that, the rest of the run carries an undestroyed device and later cases go red
          // with OOM (the deferred release in known-issues).
          try {
            // This path keeps the reference values and results as this machine's rows, so check
            // here that the adapter the key was taken from is the one running (mixing them up on a
            // multi-GPU machine would measure against another machine's band).
            assertAdapterMatchesEnvironment(gpu);
            const session = await parsed.createContainerSession(gpu);
            try {
              const outputs = await session.run(inputs);
              assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());

              parsed.graph.outputs.forEach((name, index) => {
                const view = io.tensors.get(`output.${index}`);
                assert(view !== undefined, `output.${index} is not in the golden`);
                const where = `${series.name} / ${caseName} output.${index} ('${name}')`;
                const declared = parsed.graph.values[name].dtype;
                assertEquals(outputs[name].shape, view.shape, `${where}: shape`);
                assertEquals(outputs[name].dtype, declared, `${where}: dtype`);
                const report = compareTensors(
                  outputs[name],
                  ioTensor(io, view, declared),
                  entry.tolerance,
                );
                measurements.push({
                  output: name,
                  maxAbs: report.maxAbsError,
                  maxRel: report.maxRelError,
                  tolerance: entry.tolerance,
                  stage: "karume",
                });
                assert(report.pass, `${where}: ${formatAllclose(report)}`);
              });

              // Turn the output that passed the tolerance comparison into an artifact (one
              // safetensors, no metadata) and compare it with this machine's reference row (in
              // compare mode with no row, only the comparison is skipped — settleOrObserve). If
              // outputs are added, do not silently pin only the first.
              assertEquals(parsed.graph.outputs.length, 1, `${series.name}: graph output count`);
              const [pooled] = parsed.graph.outputs;
              const pooledTensor = outputs[pooled];
              assert(
                pooledTensor.dtype === "f32",
                `${pooled}: the output to make an artifact of has dtype ${pooledTensor.dtype}`,
              );
              const outcome = await settleOrObserve(references, results, {
                id: caseId,
                artifact: `${series.name}-${caseName}.safetensors`,
                bytes: f32ArtifactBytes(pooled, pooledTensor.shape, pooledTensor.data),
              });
              settlement = outcome.settlement;
              return outcome.fields;
            } finally {
              await session.dispose();
            }
          } finally {
            gpu.destroy();
          }
        });
        // Fail after the settlement (the measured sha) is in the results — throwing first would
        // leave the measurement out of the results.
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(caseId, settlement, references));
        }
      },
    );
  }

  test.skipIf(realMissing || !MIRROR_PRESENT || !GPU_AVAILABLE)(
    `SigLIP2 discrimination: ${series.name} — the cosine of the 2 person images is higher than any person-scene pair` +
      (realMissing ? NOT_FETCHED : "") +
      (MIRROR_PRESENT ? "" : MIRROR_ABSENT),
    async () => {
      for (const real of REAL_CASES) {
        const golden = new URL(`${IO_PREFIX}${real}${IO_SUFFIX}`, root);
        assert(realCases.includes(real), absentAsset(golden));
      }
      // A golden comparison alone says only "it matches the expected values"; whether the output
      // means something as an embedding is another matter (an output collapsed to one point passes
      // if the expected values collapsed the same way). This looks at **the order between distinct
      // inputs**, so an output constant across cases makes every cosine 1 and fails. No threshold
      // is set (the order itself is what is checked — the same form as `_real_sanity` in
      // export_siglip2.py, which applies to the torch side).
      //
      // The input is the `pixel_values` baked into the golden (= the same as the comparison
      // cases). The discrimination no longer runs on input from the TS preprocessing, but the claim
      // holds — the preprocessing-side parity is pinned to about one input LSB by
      // `packages/models/tests/e2e_siglip2_image_preprocess_real_test.ts`, and it was measured that
      // the order does not flip within that width (2026-08-14: the gap between the person pair and
      // the worst cross pair is base 0.1196 / so400m 0.0747, while swapping the input for the TS
      // preprocessing moves each cosine by at most 1.6e-3 — 2 orders of magnitude smaller).
      const parsed = await openVision(series);
      const [outputName] = parsed.graph.outputs;

      const pooled = new Map<string, Float32Array>();
      const gpu = await acquireGpu();
      // MUST: the device is destroyed even when `createSession` fails. Missing that, the rest of
      // the run carries an undestroyed device and later cases go red with OOM (the deferred
      // release in known-issues).
      try {
        const session = await parsed.createContainerSession(gpu);
        try {
          // Run the 4 cases on one Session (the weights are 350 MB to 1.7 GB — no reason to
          // rebuild per case).
          for (const real of REAL_CASES) {
            const io = parseSafetensors(await readBuffer(root, `${IO_PREFIX}${real}${IO_SUFFIX}`));
            const output = (await session.run(goldenInputs(parsed, io)))[outputName];
            // Narrow by the discriminant (a cast to Float32Array would silently pass a wrong
            // dtype).
            assert(output.dtype === "f32", `${real}: pooler_output has dtype ${output.dtype}`);
            pooled.set(real, output.data);
          }
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }

      const cosineOf = (first: string, second: string): number => {
        const [left, right] = [pooled.get(first), pooled.get(second)];
        assert(left !== undefined && right !== undefined, `no output for ${first}×${second}`);
        return cosine(left, right);
      };
      const person = cosineOf(REAL_PERSON_CASES[0], REAL_PERSON_CASES[1]);
      for (const personCase of REAL_PERSON_CASES) {
        for (const sceneCase of REAL_SCENE_CASES) {
          const cross = cosineOf(personCase, sceneCase);
          assert(
            person > cross,
            `cosine order reversed: ${REAL_PERSON_CASES.join("×")}=${person} <=` +
              ` ${personCase}×${sceneCase}=${cross}`,
          );
        }
      }
    },
  );
}

const RUNNABLE = caseIds.length > 0 && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(caseIds);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds });

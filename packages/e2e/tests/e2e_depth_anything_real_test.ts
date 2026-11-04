// The **real-image** checks of Depth Anything V2 — two claims that go through the TS
// preprocessing layer (`src/image/preprocess.ts`, **bicubic**) and the PNG encoder
// (`src/image/png.ts`).
//
// ## What this file asserts (after the 0.15 test triage)
//
// The torch golden-parity file (`e2e_depth_anything_real_test.ts`) was merged into this one on
// 2026-10-17 and its model-exit parity cases were dropped: the model output is pinned per
// environment by the sha256 rows of the discrimination test below (ADR 0106), and the
// asset-completeness gate moved here unchanged. What remains:
//
//  - **input side**: the TS preprocessing (`src/image/preprocess.ts`, **bicubic**) yields the same
//    `pixel_values` as `DPTImageProcessor` from the same PNG — no GPU and no model body (99 MB).
//  - **output side**: the real-image discrimination test runs the model on the GPU, asserts the
//    composition-level ordering of near and far regions, writes the depth PNGs and checks their
//    sha256 rows.
//
// MUST: DA-V2's resize is **bicubic** (`"resample": 3` in `preprocessor_config.json`). It
// **differs** from SigLIP2 / BiRefNet's bilinear (2), so falling back to the default without
// passing {@link IMAGE_FILTER} shifts `pixel_values` by up to 0.59 (34 times 1 LSB = 0.0175 —
// measured). The parity check below fails such a mix-up by name.
//
// ## The semantic discrimination and the PNG artifacts stay here (no golden involved)
//
// The **discrimination** "a near region > a far region, as the composition shows" reads not a
// single golden tensor (it reads only the model itself and the PNGs), so there is nothing to split
// off. It was moved here **verbatim**, with the input still going through the TS preprocessing —
// the chain claim "pass a PNG and a meaningful depth comes back" is held directly here, without
// relying on synthetic images. Writing the depth PNGs is here for the same reason (`encodePng` is
// the models implementation).
//
// ## Assets (ADR 0139)
//
// The container comes from the distribution mirror `models/karume-depth-anything-v2/` (model
// `small`, quant `f32`, graph `depth`, read through the manifest); the goldens from the test-asset root
// (`goldenDir("depth-anything-v2-small-hf")`); the real images from the hand-frozen corpus
// (`corpusDir("anima")`, the same file names). Both test-asset folders are fetched with
// `node tools/test-assets/fetch.ts`. The depth maps are written to
// `outputs/verify/<environment key>/<date>_depth-anything/`, which is always safe to delete. Each case
// requires exactly what it reads: the preprocessing parity the goldens and the images, the
// discrimination (and its sha256 rows) the mirror and the images — it reads no golden tensor. A
// test asset that is absent SKIPs explicitly only while the test-asset root is not a complete fetch
// of the pin (the assets gate is red then); absent from a complete fetch, it FAILs — the dataset at
// the pin lacks it. An absent mirror SKIPs (the distribution gate owns that red). **A partially
// missing golden set = FAIL** (the completeness gate of this file names it).
// ADR 0005's "an all-SKIP run is an explicit FAIL" guard looks only at *whether a GPU adapter
// exists* and is independent of this SKIP.

import {
  type DirEntry,
  isNotFound,
  readDirSync,
  readFile,
  statSync,
  writeFile,
} from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
} from "@karume/runtime";
import {
  normalizeToNchw,
  type ResampleFilter,
  resizeRgb8,
} from "../../models/src/image/preprocess.ts";
import { encodePng } from "../../models/src/image/png.ts";
// NOTE: by design karume itself has no PNG decoder (the module doc of preprocess.ts — the entry
// point is RGB8 pixels). The tool used only for the test to read real images is borrowed from the
// runtime's tests. models → runtime is a **forward** dependency, and both exclude `tests/` from
// publishing (`publish.exclude` in packages/*/deno.json), so the published packages are not
// affected.
import { decodePng } from "@karume/test-support/png-decode";
import {
  corpusDir,
  goldenDir,
  TEST_ASSETS,
  testAssetsFetched,
} from "@karume/test-support/test-assets";
import {
  type MirrorChoice,
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { openResults, recordFailure } from "../../runtime/tests/helpers/results.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import {
  f32ArtifactBytes,
  openReferences,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleOrObserve,
} from "../../runtime/tests/helpers/reference.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

/**
 * The tolerance of the **input side** of the real-image cases (TS preprocessing vs
 * `DPTImageProcessor`).
 *
 * The real images are 1024² and the series is 518², so unlike BiRefNet **the resize is not the
 * identity**. Two things remain: ① the precision of accumulating the downscale weights (the
 * reference uses f32, this side f64) and ② the boundaries of the uint8 rounding between passes;
 * both stay within **1 uint8 LSB**. One LSB after normalisation is `1 / (std · 255)`, at most
 * `1 / (0.224 · 255) = 1.7507e-2` on the green channel. Measured maxAbs is exactly **1.7507e-2**
 * in all four cases (= never more than 1 LSB).
 *
 * atol 1.8e-2 is about 1.03 times 1 LSB. The width is kept so that **2 LSB (3.5e-2) does not
 * pass**.
 *
 * With only the upper bound, "an implementation that shifts every pixel by 1" (a rounding or filter
 * mix-up that happens to land close) would pass, so it is used together with
 * {@link PIXEL_MISMATCH_RATE_CAP}.
 *
 * This comparison is kept apart from the output side so that a failure tells whether the
 * preprocessing or the inference moved (with the output side alone, a statistics mix-up and an
 * implementation bug look the same: "a large maxAbs"). A statistics mix-up (applying SigLIP2's
 * mean = std = 0.5 / a different RGB order) and a filter mix-up (applying bilinear — measured
 * maxAbs 0.593) both show up here at least an order of magnitude above.
 *
 * NOTE: the verdict is `|got − expected| ≤ atol` (rtol was 0) plus **a failure on non-finite
 * values** — the same formula as the runtime's `compareTensors` with rtol 0, closing the same hole
 * where NaN / ±Inf would pass as "difference 0".
 */
const PIXEL_ATOL = 1.8e-2;

/**
 * The fraction of samples of `pixel_values` allowed to disagree with the reference (paired with
 * {@link PIXEL_ATOL}).
 *
 * A "disagreement" is a sample whose difference exceeds {@link PIXEL_MISMATCH_EPSILON}. Measured
 * over the four cases: 0.425% / 0.448% / 0.474% / 0.504% (804,972 samples each). The cap of 2% is
 * about 4 times the measured worst case.
 */
const PIXEL_MISMATCH_RATE_CAP = 0.02;

/**
 * The smallest difference counted as a "disagreement". It is 1/17 of 1 uint8 LSB (1.75e-2) and
 * four orders of magnitude above 1 f32 ULP (2.4e-7 at \|x\| ≈ 2) — the boundary that keeps samples
 * differing only in the last rounding bit out of the count. The reference is f32 while this side
 * accumulates in f64 and then drops to f32, so even with matching values the last bit moves in
 * nearly half the samples (measured: with this floor at 0 the disagreement rate is 50–67%).
 */
const PIXEL_MISMATCH_EPSILON = 1e-3;

/**
 * The series name (`depth_anything.export.default_out_dir`): the golden directory
 * (`goldenDir(SERIES_NAME)`) and the id of the per-run settlement.
 */
const SERIES_NAME = "depth-anything-v2-small-hf";

/**
 * The generate command, ready to paste when the cases SKIP (the form that includes the real-image
 * goldens).
 */
const GENERATE =
  "cd tools/export-recipes && uv run --group depth-anything-preprocess" +
  " python -m depth_anything.export --real-images";

/** The command that fetches the goldens and the corpus (pasted into the skip warnings). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";

/** The golden directory of the series (`outputs/test-assets/goldens/<series>/`). */
const GOLDEN_ROOT = goldenDir(SERIES_NAME);
/** The real-image corpus of the inputs (copies frozen by hand — the test-asset dataset's `corpus/anima/`). */
const CORPUS_DIR = corpusDir("anima");
/** The distribution mirror that ships the container. */
const MIRROR = mirrorRoot("karume-depth-anything-v2");
/**
 * The selection in the mirror: the model and the storage dtype the series used (`f32`). The quant
 * is pinned, not left to the manifest's default, so a new default quant cannot change the container
 * these cases and their sha256 rows test.
 */
const MIRROR_CHOICE: Required<MirrorChoice> = { model: "small", quant: "f32" };
/** The graph name in the container = the manifest's weights key (container-v1 §2.1). */
const MODEL_GRAPH = "depth";
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

/**
 * Where the depth artifacts and settlements go (`outputs/verify/<environment
 * key>/<date>_depth-anything/` — safe to delete).
 */
const results = openResults("depth-anything");
/**
 * sha256 reference values of the depth (rows per environment key —
 * `runtime/tests/helpers/reference.ts`). The case ID is the case name itself (there is one series).
 *
 * The artifact is **the graph output itself** (the depth `[1,H,W]` f32 as one safetensors file —
 * tensor name = the graph's output name, no metadata). It is the output before quantization, so
 * any single bit moving anywhere in the f32 moves the sha. The depth PNG for viewing is an image
 * reduced to 8-bit pseudo-colour after min-max normalisation; it misses differences that do not
 * cross a quantization boundary (and PNG compression may change with the runtime version), so it
 * is not hashed.
 */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/depth-anything.json", import.meta.url),
);

/**
 * The **real-image** cases (only an emit with `--real-images` has them). The source of truth for
 * the case and file names is `REAL_CASES` in `depth_anything/export.py`; for the images themselves
 * (prompt / seed / resolution) it is `examples/anima/eval-images.ts`. **Pinned here rather than
 * taken from the listing** — relying on the listing, an environment that generated only part of
 * them would lose cases silently and end up "green but unchecked".
 */
const REAL_CASES = [
  { name: "photo-portrait", file: "anima-default-1024x1024-defaultstep-seed42.png" },
  { name: "photo-landscape", file: "anima-default-1024x1024-defaultstep-seed43.png" },
  { name: "photo-corridor", file: "anima-default-1024x1024-defaultstep-seed44.png" },
  { name: "photo-street", file: "anima-default-1024x1024-defaultstep-seed45.png" },
] as const;

/** A relative rectangle of the depth map (fractions 0–1; the top / left edge is 0). */
type Region = {
  readonly label: string;
  readonly top: number;
  readonly bottom: number;
  readonly left: number;
  readonly right: number;
};

/**
 * The "near region / far region" pairs the real-image discrimination looks at. The source of truth
 * is `REAL_REGIONS` in `depth_anything/export.py`; **that side applies the same rectangles to the
 * torch output, this side to the real-GPU output** (change the rectangles on both — moving only one
 * silently turns the claim "the two agree" into something else).
 *
 * Only pairs the composition determines unambiguously are listed:
 *
 * - `photo-corridor` — the **floor** at the bottom centre (at the camera's feet) and the corridor's
 *   **vanishing point**
 * - `photo-street` — the **person** in the centre (torso to legs) and the **sky** above the
 *   buildings
 * - `photo-landscape` — the **grass** in the foreground and the **mountains and sky** behind
 * - `photo-portrait` — the **subject** at the bottom of the frame (shoulders) and the
 *   **background** at the left edge (the distant row of trees)
 */
const REAL_REGIONS: Readonly<Record<string, readonly [Region, Region]>> = {
  "photo-portrait": [
    { label: "subject", top: 0.85, bottom: 1.0, left: 0.35, right: 0.65 },
    { label: "background", top: 0.3, bottom: 0.45, left: 0.02, right: 0.12 },
  ],
  "photo-landscape": [
    { label: "foreground-ground", top: 0.92, bottom: 1.0, left: 0.3, right: 0.7 },
    { label: "mountain-sky", top: 0.05, bottom: 0.2, left: 0.35, right: 0.65 },
  ],
  "photo-corridor": [
    { label: "floor", top: 0.92, bottom: 1.0, left: 0.4, right: 0.6 },
    { label: "vanishing-point", top: 0.64, bottom: 0.76, left: 0.4, right: 0.56 },
  ],
  "photo-street": [
    { label: "person", top: 0.55, bottom: 0.85, left: 0.44, right: 0.56 },
    { label: "sky", top: 0.02, bottom: 0.12, left: 0.4, right: 0.6 },
  ],
};

/**
 * The key of the golden's input tensor (source of truth: `INPUT_NAME` in
 * `depth_anything/export.py`).
 */
const PIXEL_INPUT_KEY = "input.pixel_values";

/**
 * Fields of the real-image golden's `__metadata__` (source of truth: the constants of the same name
 * in `depth_anything/export.py`).
 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

/**
 * The preprocessing normalisation constants (ImageNet statistics). The source of truth is
 * `image_mean` / `image_std` in the bundled `preprocessor_config.json`; the copy is
 * `IMAGENET_MEAN` / `IMAGENET_STD` in `depth_anything/export.py`. **These are different statistics
 * from SigLIP2 (mean = std = 0.5)**, so they are not shared.
 */
const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/**
 * The interpolation filter. `"resample": 3` in `preprocessor_config.json` = **BICUBIC as a PIL
 * constant** (`depth_anything.export.check_processor` measures it from the real processor on every
 * emit). It differs from BiRefNet / SigLIP2's bilinear — the MUST of the module docstring.
 */
const IMAGE_FILTER: ResampleFilter = "bicubic";

/**
 * Whether the file exists.
 * MUST: anything but NotFound propagates — reading a permission error or the like as "no asset"
 * would turn a check that never ran into a quiet green SKIP.
 */
const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const goldenUrl = (caseName: string): URL =>
  new URL(`${IO_PREFIX}${caseName}${IO_SUFFIX}`, GOLDEN_ROOT);

/**
 * Counted synchronously because it is needed at registration (used for `Deno.test`'s ignore
 * decision).
 */
const GOLDEN_COUNT = REAL_CASES.filter((entry) => exists(goldenUrl(entry.name))).length;

/**
 * Whether all four real images are present (`corpus/anima/` holds the images copied and frozen by
 * hand).
 * MUST: anything but NotFound propagates (the same reason as `exists`).
 */
const IMAGES_PRESENT: boolean = REAL_CASES.every((entry) =>
  exists(new URL(entry.file, CORPUS_DIR)),
);

/**
 * Whether `outputs/test-assets/` holds a complete fetch of the pinned revision. Then a golden or an
 * image that is absent is a FAIL (the dataset at the pin lacks it), not a SKIP.
 */
const FETCHED = testAssetsFetched();

/**
 * Asserts that a test asset the case reads is present. A case reaches this with the asset absent
 * when the test-asset root is a complete fetch (otherwise it SKIPs) or a partial one.
 */
const assertAssetPresent = (url: URL): void =>
  assert(
    exists(url),
    FETCHED
      ? `${url.pathname} is absent, but outputs/test-assets/ holds a complete fetch of ` +
          `${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks it`
      : `${url.pathname} is absent (fetch: ${FETCH_COMMAND})`,
  );

/** Title suffixes of a case that SKIPs for want of an asset, so the test list says why. */
const NOT_FETCHED = " (SKIP: test assets not fetched)";
const MIRROR_ABSENT = " (SKIP: distribution mirror absent)";

/**
 * The preprocessing parity: it needs the goldens and the images (absent from a complete fetch of
 * the pin, they FAIL the cases instead).
 */
const PARITY_AVAILABLE = (GOLDEN_COUNT > 0 && IMAGES_PRESENT) || FETCHED;

/** Whether the distribution mirror is present (`karume/5`). */
const MIRROR_PRESENT = mirrorPresent(MIRROR);
/**
 * The discrimination and its sha256 rows: they need the mirror and the images (no golden tensor is
 * read — the golden files are not a run condition of this case).
 */
const IMAGES_RUNNABLE = IMAGES_PRESENT || FETCHED;
const DISCRIMINATION_AVAILABLE = MIRROR_PRESENT && IMAGES_RUNNABLE;

/** Opens the `depth` container from the mirror (through the manifest — ADR 0139). */
const openDepth = async (): Promise<PreparedModel> =>
  prepareContainer(
    await openSelectedContainer(await selectMirrorComponent(MIRROR, MODEL_GRAPH, MIRROR_CHOICE)),
    MODEL_GRAPH,
  );

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/**
 * The bytes of one real image (before decoding — kept raw because the sha256 comparison uses them
 * too).
 */
const readImage = (file: string): Promise<Uint8Array<ArrayBuffer>> =>
  readFile(new URL(file, CORPUS_DIR));

/**
 * A static dimension of the graph input (there are no symbolic dimensions — `symbol_names=()` in
 * `depth_anything/export.py`).
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
 * Turns one real image into `pixel_values` through **the TS path** (PNG decode → bicubic resize →
 * normalize). What it goes through is the models package's preprocessing layer itself — this
 * chain is exactly the claim of this check.
 */
const preprocessImage = async (
  png: Uint8Array<ArrayBuffer>,
  file: string,
  width: number,
  height: number,
): Promise<Float32Array<ArrayBuffer>> =>
  normalizeToNchw(
    resizeRgb8(await decodePng(png, file), width, height, IMAGE_FILTER),
    IMAGE_MEAN,
    IMAGE_STD,
  );

/**
 * The fraction of samples that disagree with the reference (difference above {@link
 * PIXEL_MISMATCH_EPSILON}).
 */
const mismatchRate = (got: Float32Array, expected: Float32Array): number => {
  assertEquals(got.length, expected.length, "pixel_values length");
  let differing = 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (Math.abs(got[index] - expected[index]) > PIXEL_MISMATCH_EPSILON) differing += 1;
  }
  return differing / expected.length;
};

/**
 * The mean depth inside a relative rectangle (the discrimination looks only at the order of this
 * quantity).
 */
const regionMean = (depth: Float32Array, size: number, region: Region): number => {
  const top = Math.trunc(region.top * size);
  const bottom = Math.trunc(region.bottom * size);
  const left = Math.trunc(region.left * size);
  const right = Math.trunc(region.right * size);
  let sum = 0;
  let count = 0;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      sum += depth[y * size + x];
      count += 1;
    }
  }
  assert(count > 0, `region ${region.label} is empty on a ${size}×${size} map`);
  return sum / count;
};

/**
 * The stops of the pseudo-colour map for depth (far = dark navy → near = pale yellow). **Greyscale
 * is not used** because relative depth crowds on the far side (measured: the sky and the flat
 * ground at the vanishing point are exactly 0, and only the subject reaches the top of the range).
 * With lightness alone the far field crushes to black and the eye cannot tell "the far field has
 * structure" from "the output is dead". This is an artifact, not a check, so readability wins over
 * colour accuracy.
 */
const DEPTH_COLORS: readonly (readonly [number, number, number])[] = [
  [12, 16, 48],
  [58, 26, 110],
  [150, 44, 96],
  [232, 108, 48],
  [252, 236, 160],
];

/** The PNG bytes of a depth map (min-max normalised → coloured with {@link DEPTH_COLORS}). */
const depthToPng = (depth: Float32Array, size: number): Promise<Uint8Array<ArrayBuffer>> => {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of depth) {
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  const span = maximum - minimum;
  const last = DEPTH_COLORS.length - 1;
  const rgba = new Uint8Array(size * size * 4);
  for (let index = 0; index < depth.length; index += 1) {
    const normalized = span > 0 ? (depth[index] - minimum) / span : 0;
    const position = normalized * last;
    const anchor = Math.min(last - 1, Math.floor(position));
    const weight = position - anchor;
    for (let channel = 0; channel < 3; channel += 1) {
      const low = DEPTH_COLORS[anchor][channel];
      const high = DEPTH_COLORS[anchor + 1][channel];
      rgba[index * 4 + channel] = Math.round(low + (high - low) * weight);
    }
    rgba[index * 4 + 3] = 255;
  }
  return encodePng(rgba, size, size);
};

if (!PARITY_AVAILABLE) {
  console.warn(
    `[karume] The Depth Anything V2 real-image preprocessing parity is skipped (goldens ` +
      `${GOLDEN_COUNT}/${REAL_CASES.length} in ${GOLDEN_ROOT.pathname}, images ` +
      `${IMAGES_PRESENT ? "present" : `absent from ${CORPUS_DIR.pathname}`}). Fetch: ${FETCH_COMMAND}`,
  );
}
if (!DISCRIMINATION_AVAILABLE) {
  console.warn(
    `[karume] The Depth Anything V2 real-image discrimination is skipped (mirror ` +
      `${MIRROR_PRESENT ? "present" : `${MIRROR.pathname} absent — place the published distribution under models/`}, ` +
      `images ${IMAGES_PRESENT ? "present" : `absent from ${CORPUS_DIR.pathname} — fetch: ${FETCH_COMMAND}`}).`,
  );
}

// Asset completeness gate (merged from the former e2e_depth_anything_real_test.ts in the 0.15 test
// triage — docs/research/2026-10-16-test-triage.md). The skips above only say "nothing to run";
// this gate turns a *partially* present golden directory into a FAIL instead of a silent green
// SKIP, naming what is missing. It covers the goldens only: the container moved to the mirror,
// whose parts the distribution gate checks (packages/runtime/tests/distribution_gate_test.ts).

/**
 * Synthetic golden cases every emit writes. Source of truth: `build_cases` in
 * `depth_anything/export.py` (the same 4 images regardless of the model axis).
 */
const SYNTHETIC_CASES = ["checker", "disc", "noise", "ramp"] as const;

/**
 * Lists a directory; only a missing directory degrades to empty.
 * MUST: propagate everything but NotFound — reading a permission error as "no assets" turns a
 * check that never ran into a silent green SKIP.
 */
const listDir = (url: URL): readonly DirEntry[] => {
  try {
    return [...readDirSync(url)];
  } catch (cause) {
    if (isNotFound(cause)) return [];
    throw cause;
  }
};

/** Golden case names found in the golden directory (`io.<case>.safetensors`). */
const discoverGoldenCases = (root: URL): readonly string[] =>
  listDir(root)
    .filter(
      (entry) => entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
    )
    .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
    .sort();

const DISCOVERED_GOLDEN = discoverGoldenCases(GOLDEN_ROOT);
const realCaseNames = new Set<string>(REAL_CASES.map((entry) => entry.name));
const FOUND_SYNTHETIC = DISCOVERED_GOLDEN.filter((name) => !realCaseNames.has(name));
const FOUND_REAL = DISCOVERED_GOLDEN.filter((name) => realCaseNames.has(name));
/**
 * Whether any golden is present. The gate may sleep only on a golden directory that is entirely
 * absent while the test-asset root is not a complete fetch of the pin; absent from a complete
 * fetch, the directory FAILs the gate.
 */
const ANY_PRESENT = DISCOVERED_GOLDEN.length > 0;
const GOLDENS_MISSING = !ANY_PRESENT && !FETCHED;

test.skipIf(GOLDENS_MISSING)(
  `Depth Anything goldens: the expected cases are present${GOLDENS_MISSING ? NOT_FETCHED : ""}`,
  () => {
    assert(
      ANY_PRESENT,
      `${GOLDEN_ROOT.pathname} is absent, but outputs/test-assets/ holds a complete fetch of ` +
        `${TEST_ASSETS.repo}@${TEST_ASSETS.revision}: the dataset at the pin lacks it`,
    );
    assertEquals(
      FOUND_SYNTHETIC,
      [...SYNTHETIC_CASES],
      `synthetic-image golden cases of ${GOLDEN_ROOT.pathname}`,
    );
    // Real-image goldens are optional but all-or-none (an emit with `--real-images` writes all 4).
    // Rounding a partial set to SKIP would let assets lost mid-regeneration pass silently.
    assert(
      FOUND_REAL.length === 0 || FOUND_REAL.length === REAL_CASES.length,
      `${GOLDEN_ROOT.pathname} has ${FOUND_REAL.length}/${REAL_CASES.length} real-image goldens` +
        ` (re-take them: ${GENERATE})`,
    );
  },
);

for (const entry of REAL_CASES) {
  test.skipIf(!PARITY_AVAILABLE)(
    `Depth Anything real-image preprocessing parity: ${entry.name}` +
      " (TS bicubic + folded normalisation vs DPTImageProcessor)" +
      (PARITY_AVAILABLE ? "" : NOT_FETCHED),
    async () => {
      assertAssetPresent(goldenUrl(entry.name));
      assertAssetPresent(new URL(entry.file, CORPUS_DIR));
      const [goldenBytes, png] = await Promise.all([
        readBuffer(goldenUrl(entry.name)),
        readImage(entry.file),
      ]);
      const io = parseSafetensors(goldenBytes);

      // ① The image the golden was taken from is the image being read now. A **difference no
      // tolerance absorbs** (the generation script was re-run without re-taking the golden) fails
      // by name before the comparison. If this loosened, "green from comparing two different
      // images" could happen.
      assertEquals(
        io.metadata.get(SOURCE_IMAGE_KEY),
        entry.file,
        "the source image the golden points to",
      );
      assertEquals(
        io.metadata.get(SOURCE_SHA256_KEY),
        await sha256Hex(png),
        `${entry.file} differs from the image the golden was taken from (re-take it: ${GENERATE})`,
      );

      const view = io.tensors.get(PIXEL_INPUT_KEY);
      assert(view !== undefined, `${PIXEL_INPUT_KEY} is missing from the golden`);
      assertEquals(view.dtype, "F32", `storage dtype of ${PIXEL_INPUT_KEY}`);
      assertEquals(view.shape.length, 4, `rank of ${PIXEL_INPUT_KEY}`);
      const [batch, channels, height, width] = view.shape;
      assertEquals([batch, channels], [1, 3], `first 2 axes of ${PIXEL_INPUT_KEY}`);
      const expected = new Float32Array(io.buffer, view.byteOffset, view.byteLength / 4);

      // ② The TS path (PNG decode → bicubic resize → fused normalisation).
      const got = await preprocessImage(png, entry.file, width, height);

      assertEquals(got.length, expected.length, `${entry.name}: length of pixel_values`);
      let maxAbs = 0;
      let worst = 0;
      let nonFinite = 0;
      for (let index = 0; index < got.length; index += 1) {
        if (!Number.isFinite(got[index]) || !Number.isFinite(expected[index])) {
          nonFinite += 1;
          continue;
        }
        const delta = Math.abs(got[index] - expected[index]);
        if (delta > maxAbs) {
          maxAbs = delta;
          worst = index;
        }
      }
      // MUST: NaN / ±Inf fail. A naive difference check lets NaN pass as "difference 0".
      assertEquals(nonFinite, 0, `${entry.name}: non-finite samples in pixel_values`);
      assert(
        maxAbs <= PIXEL_ATOL,
        `${entry.name}: pixel_values maxAbs=${maxAbs}` +
          ` (sample ${worst}: ${got[worst]} vs ${expected[worst]}) exceeds ${PIXEL_ATOL}`,
      );

      // ③ With only the upper bound, "an implementation that shifts every pixel by 1" would pass,
      // so the disagreement rate is checked alongside.
      const rate = mismatchRate(got, expected);
      assert(
        rate <= PIXEL_MISMATCH_RATE_CAP,
        `${entry.name} pixel_values disagreement rate ${(rate * 100).toFixed(3)}%` +
          ` (cap ${PIXEL_MISMATCH_RATE_CAP * 100}%) — a shift over the whole image, even within 1 LSB, ` +
          "is a filter or rounding mix-up",
      );
    },
  );
}

test.skipIf(!DISCRIMINATION_AVAILABLE || !GPU_AVAILABLE)(
  "Depth Anything real-image discrimination: the depth of the near region the composition shows exceeds the far region" +
    " (also writes the depth PNGs)" +
    (IMAGES_RUNNABLE ? "" : NOT_FETCHED) +
    (MIRROR_PRESENT ? "" : MIRROR_ABSENT),
  async () => {
    for (const real of REAL_CASES) assertAssetPresent(new URL(real.file, CORPUS_DIR));
    // The discrimination looks only at **the order of the region means** (no threshold — the same
    // form as `_real_sanity` in `depth_anything/export.py`, which applies it on the torch side). A
    // uniformly collapsed output, an input-independent output and an upside-down one all fail,
    // because the two region means tie or swap. Measured: corridor 3.8544 / 0.2055, street
    // 3.1659 / 0.0044, landscape 3.5228 / 0.0855, portrait 4.9632 / 0.2736 (3 digits agree with
    // the torch side).
    //
    // The input is **the real image passed through the TS preprocessing** (not the golden's
    // input) — the check reaches as far as "pass a PNG and a meaningful depth comes back". The
    // depth maps are written out as PNGs on the way (a numeric check alone does not show the
    // shape — an artifact for viewing, not hashed). What is hashed is the output before
    // quantization itself (the depth `[1,H,W]` f32 as safetensors — {@link references}), which
    // makes a claim separate from the discrimination: "not a single bit moved on the same
    // machine". The order is: run the four images → discriminate → record the sha settlement per
    // image → record the series settlement → fail on sha mismatches (no reference row is made
    // from an output whose discrimination split).
    const started = performance.now();
    const elapsedMs = (): number => Math.round(performance.now() - started);
    /**
     * The region means per image (the material of the discrimination; a run that fails midway still
     * leaves what was measured in the note).
     */
    const means = new Map<string, { near: number; far: number }>();
    const meansNote = (): string =>
      [...means]
        .map(([name, measured]) => `${name} ${measured.near.toFixed(4)}/${measured.far.toFixed(4)}`)
        .join(" / ");
    /**
     * The images whose sha disagreed with the reference (they fail together after every image's
     * settlement is recorded).
     */
    const mismatches: { readonly label: string; readonly settlement: ReferenceSettlement }[] = [];
    try {
      const parsed = await openDepth();
      const [outputName] = parsed.graph.outputs;
      const width = staticDim(parsed, 3);
      const height = staticDim(parsed, 2);
      // The relative rectangles and the PNG both index by one side only (a non-square map would
      // silently measure somewhere else).
      assertEquals(width, height, "the region discrimination assumes a square map");
      const inputName = parsed.graph.inputs[0].name;

      /** The sha artifacts per image (compared once the discrimination passes). */
      const depths: {
        readonly caseId: string;
        readonly artifact: string;
        readonly bytes: Uint8Array<ArrayBuffer>;
        readonly elapsedMs: number;
      }[] = [];
      const gpu = await acquireGpu();
      try {
        // This path may write the depth sha as this machine's row, so it first checks that the
        // adapter the key was taken from is the one running (a mix-up on a multi-GPU machine
        // would become another machine's row).
        assertAdapterMatchesEnvironment(gpu);
        const session = await parsed.createContainerSession(gpu);
        try {
          // The four images run on one Session (the weights are 99 MB — no reason to rebuild per
          // image).
          for (const real of REAL_CASES) {
            const imageStarted = performance.now();
            const pixels = await preprocessImage(
              await readImage(real.file),
              real.file,
              width,
              height,
            );
            const output = (
              await session.run({
                [inputName]: { dtype: "f32", shape: [1, 3, height, width], data: pixels },
              })
            )[outputName];
            // Narrow by the discriminant (a cast to Float32Array would pass silently when the dtype
            // is off).
            assert(output.dtype === "f32", `${real.name}: the depth dtype is ${output.dtype}`);
            assertEquals(output.shape, [1, height, width], `${real.name}: shape of the depth map`);
            const [near, far] = REAL_REGIONS[real.name];
            means.set(real.name, {
              near: regionMean(output.data, width, near),
              far: regionMean(output.data, width, far),
            });
            depths.push({
              caseId: real.name,
              artifact: `${real.name}-depth.safetensors`,
              bytes: f32ArtifactBytes(outputName, output.shape, output.data),
              elapsedMs: Math.round(performance.now() - imageStarted),
            });
            // The PNG for viewing (min-max normalisation → 8-bit pseudo-colour — not hashed).
            await writeFile(
              results.artifact(`${real.name}-depth.png`),
              await depthToPng(output.data, width),
            );
          }
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }

      for (const real of REAL_CASES) {
        const measured = means.get(real.name);
        assert(measured !== undefined, `${real.name} has no region means`);
        const [near, far] = REAL_REGIONS[real.name];
        assert(
          measured.near > measured.far,
          `${real.name}: the near ${near.label} has mean depth ${measured.near}, which is` +
            ` at most the far ${far.label}'s ${measured.far} — the composition's near and far are not recovered`,
        );
      }

      // Only outputs that passed the discrimination are compared with the reference rows (in
      // compare mode without a row, only the comparison is skipped and the measured sha is
      // recorded — settleOrObserve).
      for (const entry of depths) {
        const outcome = await settleOrObserve(references, results, {
          id: entry.caseId,
          artifact: entry.artifact,
          bytes: entry.bytes,
        });
        await results.record({
          id: entry.caseId,
          ...outcome.fields,
          elapsedMs: entry.elapsedMs,
        });
        if (outcome.settlement?.check.status === "fail") {
          mismatches.push({ label: entry.caseId, settlement: outcome.settlement });
        }
      }
    } catch (cause) {
      // Record the settlement before failing (without the region means at hand, reading the red
      // cannot start; leaving without a settlement would leave the previous run of the same day's
      // settlement in place). An I/O failure while recording does not replace the original
      // exception (recordFailure).
      await recordFailure(results, {
        id: SERIES_NAME,
        status: "fail",
        elapsedMs: elapsedMs(),
        note: `${meansNote()} / failure: ${cause instanceof Error ? cause.message : String(cause)}`,
      });
      throw cause;
    }
    // sha mismatches also go into the series settlement (reading only the series row shows the
    // red).
    await results.record({
      id: SERIES_NAME,
      status: mismatches.length === 0 ? "pass" : "fail",
      elapsedMs: elapsedMs(),
      note:
        mismatches.length === 0
          ? meansNote()
          : `${meansNote()} / sha mismatch: ${mismatches.map(({ label }) => label).join(", ")}`,
    });
    console.log(
      `[karume] wrote the Depth Anything depths to ${results.dir.pathname}` +
        " (<case>-depth.safetensors / -depth.png)",
    );
    // The settlements per image and per series are all recorded (throwing earlier would leave the
    // measured sha out of the results).
    assert(
      mismatches.length === 0,
      mismatches
        .map(({ label, settlement }) => referenceMismatchMessage(label, settlement, references))
        .join("\n"),
    );
  },
);

/** When the sha256 gate runs: the same condition as the discrimination test (mirror + images + GPU). */
const RUNNABLE = DISCRIMINATION_AVAILABLE && GPU_AVAILABLE;
const CASE_IDS: readonly string[] = REAL_CASES.map((entry) => entry.name);
if (RUNNABLE) references.warnMissing(CASE_IDS);
// A guard so that "this environment has no reference value yet" is not a silent green (the same
// approach as ADR 0005).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });

// The **real-image** check of the BiRefNet family — it holds two claims that go through the TS
// preprocessing layer (`src/image/preprocess.ts`) and PNG encoding (`src/image/png.ts`).
//
// ## The claims this check holds, and the split with the runtime side
//
// It originally lived in `packages/runtime/tests/e2e_birefnet_test.ts` (it ran inference on the
// `pixel_values` produced by the TS preprocessing, checking the input side and the output side in
// one test). It was split in two — the same shape first met with SigLIP2 (the head of
// `e2e_siglip2_image_preprocess_real_test.ts`):
//
//  - **Input side (here)**: "the same PNG yields the same `pixel_values` as Python" — a pure parity
//    of the preprocessing layer alone, needing neither a GPU nor the model itself (964MB per
//    series).
//  - **output side**: the real-GPU fidelity on the golden `pixel_values` (the real-image golden
//    cases of `e2e_birefnet_test.ts`). They are upstream parity and opt-in
//    (`KARUME_UPSTREAM_PARITY=1`); the default lanes pin the output per environment by the
//    sha256 rows of the discrimination below (ADR 0106).
//
// The chain claim "pass a PNG and get the same matte as Python" is held by **composing** the two
// (this side guarantees an input difference ≤ {@link PIXEL_ATOL}, the runtime side guarantees the
// fidelity on that input). The reason for the split is the dependency direction — the runtime tests
// imported the models implementation by relative path (the wrong direction).
//
// ## The semantic discrimination and the PNG artifacts stay here (no golden involved)
//
// The **discrimination** — the foreground ratio of the 2 images with a salient object exceeds that
// of the 2 without — reads not a single golden tensor (it reads only the model itself and the
// PNGs), so it cannot be split. It was moved here **verbatim**, with the input as it comes out of
// the TS preprocessing — the chain claim "pass a PNG and get a meaningful matte" is held directly
// here, without relying on composition. Writing the matte PNGs / the PNGs composited on white is
// here for the same reason (`encodePng` is the models implementation).
//
// ## Assets (ADR 0139)
//
// The containers come from the distribution mirrors (`models/karume-birefnet-hr/`,
// `models/karume-lucida/`, read through their manifests); the goldens from the test-asset root
// (`goldenDir(<series>)`); the real images from the hand-frozen corpus (`corpusDir("anima")`, the
// same file names). Both test-asset folders are fetched with `node tools/test-assets/fetch.ts`. The
// matte PNGs are written to `outputs/verify/`, which is always safe to delete. Each case requires
// exactly what it reads: the preprocessing parity the goldens and the images, the discrimination
// (and its sha256 rows) the mirror and the images — it reads no golden tensor. A test asset that is
// absent SKIPs explicitly only while the test-asset root is not a complete fetch of the pin (the
// assets gate is red then); absent from a complete fetch, it FAILs — the dataset at the pin lacks
// it. An absent mirror SKIPs (the distribution gate owns that red). A partially missing golden set
// also FAILs (the completeness test of `e2e_birefnet_test.ts` names it).
// The gate of ADR 0005's "all cases SKIP = explicit FAIL" looks only at *whether a GPU adapter
// exists*, independently of this SKIP.

import { isNotFound, readFile, statSync, writeFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
} from "@karume/runtime";
import { normalizeToNchw, resizeRgb8 } from "../../models/src/image/preprocess.ts";
import { encodePng } from "../../models/src/image/png.ts";
// NOTE: karume itself is designed without a PNG decoder (the module doc of preprocess.ts — the
// entry point is the RGB8 pixel sequence). A tool used only so that the tests can read real images
// is borrowed from under the runtime tests. models → runtime is the **forward** dependency
// direction, and both exclude `tests/` from publishing (`publish.exclude` in packages/*/deno.json),
// so the published packages are unaffected.
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
 * The tolerance of the **input side** of the real-image cases (TS preprocessing vs Python
 * preprocessing).
 *
 * The real images and the series are both 1024², so **the resize is the identity** (the support of
 * `resizeRgb8` shrinks to a single point of weight 1 at scale 1). What remains is only the
 * difference in how the normalisation is folded: TS does `(u8 − mean·255) / (std·255)` in one step,
 * while the torch side does `ToTensor` (`u8 / 255`) → `Normalize` (`(x − mean) / std`) of the
 * bundled `handler.py` — **one more rounding**. The measurement shows only this difference,
 * **maxAbs 4.768e-7 for all 4 series × 4 cases** (measured on the 4 series of the time, including
 * the retired `birefnet-hr-1024`; 2 ulp against an \|ref\| upper end of 2.64. The input goldens of
 * the 2 f16 series are bit-identical to those of the f32 series — compared on 2026-09-26).
 *
 * atol 1e-6 is about 2.1 × the measured 4.768e-7.
 *
 * **This one alone is shared across series** (the runtime side's output tolerances are not shared).
 * What it compares is only the chain that produces `pixel_values`, and neither side ever touches
 * the model's weights — same image, same preprocessing constants, same dimensions, so measuring per
 * series can only give **the same result of the same computation**.
 *
 * This comparison is kept apart from the output side so that, when it fails, it tells whether the
 * preprocessing or the inference moved (with only the output side, a mix-up of the statistics and
 * an implementation bug both look like the same "large maxAbs"). A mix-up of the statistics
 * (applying SigLIP2's mean = std = 0.5 / a wrong RGB order) gives a difference of O(0.1) to O(1)
 * here, four or more orders of magnitude above.
 *
 * NOTE: the verdict is `|got − expected| ≤ atol` (rtol was 0) plus **a failure on non-finite
 * values** — the same formula as the runtime's `compareTensors` with rtol 0, closing the same hole
 * where NaN / ±Inf would pass as "difference 0".
 */
const PIXEL_ATOL = 1e-6;

/**
 * One series that runs (model × resolution). The source of truth is `SERIES` of the runtime-side
 * e2e.
 */
type Series = {
  /**
   * The series name (`birefnet.export.default_out_dir`): the golden directory
   * (`goldenDir(name)`) and the prefix of the sha256 case IDs (frozen — ADR 0139 decision 7).
   */
  readonly name: string;
  /** The distribution mirror that ships this series' container, and its selection. */
  readonly mirror: { readonly repo: string } & Required<MirrorChoice>;
  /** The export command that regenerates this series' goldens (pasted into failure messages). */
  readonly generate: string;
};

const EXPORT_PREFIX =
  "cd tools/export-recipes && uv run --group birefnet python -m birefnet.export";

/**
 * The series with real-image goldens (1024² only — no real-image goldens are taken at 2048²). The
 * f16 series of the storage dtype (ADR 0113) go through the same discrimination and sha reference
 * values. BiRefNet_HR has only the f16 series (the upstream checkpoint is f16 and the f32 series
 * merely widened the same values, so it was retired — ADR 0113 addendum 2026-10-11. Before
 * retirement, the sha rows of the f32 series matched the f16 series' rows exactly in 2 environments
 * × 4 cases).
 */
const SERIES: readonly Series[] = [
  {
    name: "lucida-1024",
    mirror: { repo: "karume-lucida", model: "1024", quant: "f32" },
    generate: `${EXPORT_PREFIX} --model-dir <repo>/inputs/birefnet/lucida --real-images`,
  },
  {
    name: "birefnet-hr-1024-f16",
    mirror: { repo: "karume-birefnet-hr", model: "1024", quant: "f16" },
    generate: `${EXPORT_PREFIX} --dtype f16 --real-images`,
  },
  {
    name: "lucida-1024-f16",
    mirror: { repo: "karume-lucida", model: "1024", quant: "f16" },
    generate: `${EXPORT_PREFIX} --model-dir <repo>/inputs/birefnet/lucida --dtype f16 --real-images`,
  },
];

/**
 * The real-image corpus of the inputs (copies frozen by hand — the test-asset dataset's
 * `corpus/anima/`).
 */
const CORPUS_DIR = corpusDir("anima");
/** The command that fetches the goldens and the corpus (pasted into the skip warnings). */
const FETCH_COMMAND = "node tools/test-assets/fetch.ts";
/** The graph name in the container = the manifest's weights key (container-v1 §2.1). */
const GRAPH = "matte";
/**
 * Where the matte PNGs and the outcomes go (`outputs/verify/<environment key>/<date>_birefnet/` —
 * safe to delete).
 */
const results = openResults("birefnet");
/**
 * The sha256 reference values of the matte logits (rows per environment key —
 * `runtime/tests/helpers/reference.ts`).
 *
 * The artifact is **the graph output itself** (the pre-sigmoid logit `[1,1,H,W]` f32 as one
 * safetensors file — tensor name = the graph's output name, no metadata). It is the output before
 * quantisation, so one bit moving anywhere in the f32 moves the sha. The matte PNG / the PNG
 * composited on white for visual inspection are images reduced to 8 bit, which miss differences
 * that do not cross a quantisation boundary (and PNG compression can change with the runtime's
 * version), so they are not what the sha covers.
 */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/birefnet.json", import.meta.url),
);
const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

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
 * The names of the artifacts (the matte PNG / the PNG composited on white for visual inspection,
 * and the matte logits safetensors the sha covers). **The series name is prefixed** — written under
 * the same name, the matte of the series that ran later would silently replace the earlier series'
 * one. `file` is the kind and extension (`matte.png` / `cutout.png` / `matte.safetensors`).
 */
const artifactName = (series: Series, caseName: string, file: string): string =>
  `${series.name}-${caseName}-${file}`;

/**
 * The case ID of a reference value (series × image — the same image gives a different matte on
 * another series).
 */
const referenceCaseId = (series: Series, caseName: string): string => `${series.name}/${caseName}`;

/**
 * The **real-image** cases (held only by an emit with `--real-images`). The source of truth for the
 * case names and file names is `REAL_CASES` of `birefnet/export.py`; for the images themselves
 * (prompt / seed / resolution), `examples/anima/eval-images.ts`. **They are pinned here, not
 * enumerated** — relying on enumeration, cases silently vanish in an environment that generated
 * only some of them and it becomes "green but unverified".
 */
const REAL_CASES = [
  { name: "photo-portrait", file: "anima-default-1024x1024-defaultstep-seed42.png" },
  { name: "photo-landscape", file: "anima-default-1024x1024-defaultstep-seed43.png" },
  { name: "photo-corridor", file: "anima-default-1024x1024-defaultstep-seed44.png" },
  { name: "photo-street", file: "anima-default-1024x1024-defaultstep-seed45.png" },
] as const;

/**
 * The two groups the real-image discrimination looks at (salient object = the 2 images with a
 * person / the 2 without). The source of truth is `REAL_PERSON_CASES` / `REAL_SCENE_CASES` of
 * `birefnet/export.py`; there they are applied to the torch output, here to **the real-GPU
 * output**.
 */
const REAL_PERSON_CASES = ["photo-portrait", "photo-street"] as const;
const REAL_SCENE_CASES = ["photo-landscape", "photo-corridor"] as const;

/**
 * The key of the golden's input tensor (the source of truth is `INPUT_NAME` of
 * `birefnet/export.py`).
 */
const PIXEL_INPUT_KEY = "input.pixel_values";

/**
 * The `__metadata__` fields of a real-image golden (the source of truth is the constants of the
 * same name in `birefnet/export.py`).
 */
const SOURCE_IMAGE_KEY = "source_image";
const SOURCE_SHA256_KEY = "source_sha256";

/**
 * The preprocessing normalisation constants (ImageNet statistics). The source of truth is
 * `ImagePreprocessor` of the bundled `handler.py`; the copy is `IMAGENET_MEAN` / `IMAGENET_STD` of
 * `birefnet/export.py`. **Different statistics from SigLIP2 (mean = std = 0.5)**, so they are not
 * shared — {@link PIXEL_ATOL} catches a mix-up. The 2 series share the statistics because Lucida is
 * a fine-tune of BiRefNet_HR that did not change the preprocessing (the usage example in the
 * upstream model card spells out the same 3 steps).
 */
const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/**
 * Whether a file exists.
 * MUST: anything other than NotFound propagates — reading a permission error and the like as "no
 * assets" would turn a verification that did not run into a quietly green SKIP.
 */
const exists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const goldenUrl = (series: Series, caseName: string): URL =>
  new URL(`${IO_PREFIX}${caseName}${IO_SUFFIX}`, goldenDir(series.name));

/**
 * Counted synchronously because it is needed at registration (used for the `Deno.test` ignore
 * decision).
 */
const goldenCount = (series: Series): number =>
  REAL_CASES.filter((entry) => exists(goldenUrl(series, entry.name))).length;

/**
 * Whether all 4 real images are present (`corpus/anima/` is the place of images copied and frozen
 * by hand).
 * MUST: anything other than NotFound propagates (the same reason as `exists`).
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

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/**
 * The bytes of one real image (before decode — kept raw because the sha256 comparison uses them
 * too).
 */
const readImage = (file: string): Promise<Uint8Array<ArrayBuffer>> =>
  readFile(new URL(file, CORPUS_DIR));

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
 * The area ratio of the foreground (`logit > 0`). The discrimination (salient object or not) looks
 * only at the order of this quantity.
 */
const foregroundRatio = (logits: Float32Array): number => {
  let foreground = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (logits[index] > 0) foreground += 1;
  }
  return foreground / logits.length;
};

/** The α of the matte (`sigmoid` is the host side's job — the graph outputs logits). */
const alphaFromLogits = (logits: Float32Array): Float32Array => {
  const alpha = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    alpha[index] = 1 / (1 + Math.exp(-logits[index]));
  }
  return alpha;
};

/** The PNG bytes of the grayscale (α taken as the lightness as is). */
const matteToPng = (
  alpha: Float32Array,
  width: number,
  height: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const rgba = new Uint8Array(width * height * 4);
  for (let index = 0; index < alpha.length; index += 1) {
    const value = Math.round(alpha[index] * 255);
    rgba[index * 4] = value;
    rgba[index * 4 + 1] = value;
    rgba[index * 4 + 2] = value;
    rgba[index * 4 + 3] = 255;
  }
  return encodePng(rgba, width, height);
};

/**
 * The PNG bytes of the foreground composited on white (the look of the background removal itself).
 */
const cutoutToPng = (
  image: { readonly data: Uint8Array; readonly width: number; readonly height: number },
  alpha: Float32Array,
): Promise<Uint8Array<ArrayBuffer>> => {
  const rgba = new Uint8Array(image.width * image.height * 4);
  for (let index = 0; index < alpha.length; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const source = image.data[index * 3 + channel];
      rgba[index * 4 + channel] = Math.round(source * alpha[index] + 255 * (1 - alpha[index]));
    }
    rgba[index * 4 + 3] = 255;
  }
  return encodePng(rgba, image.width, image.height);
};

/**
 * The series whose real-image groups are complete (the range of cases the reference gate counts —
 * decided synchronously at registration).
 */
const RUNNABLE_SERIES: Series[] = [];

for (const series of SERIES) {
  const goldens = goldenCount(series);
  /**
   * The preprocessing parity: it needs the goldens and the images (absent from a complete fetch of
   * the pin, they FAIL the cases instead).
   */
  const parityAvailable = (goldens > 0 && IMAGES_PRESENT) || FETCHED;
  /**
   * The discrimination and its sha256 rows: they need the mirror and the images (no golden tensor
   * is read — the golden files are not a run condition of this case).
   */
  const mirror = mirrorPresent(mirrorOf(series));
  const imagesRunnable = IMAGES_PRESENT || FETCHED;
  const discriminationAvailable = mirror && imagesRunnable;
  if (discriminationAvailable) RUNNABLE_SERIES.push(series);

  if (!parityAvailable) {
    console.warn(
      `[karume] The real-image preprocessing parity of ${series.name} is skipped (goldens ` +
        `${goldens}/${REAL_CASES.length} in ${goldenDir(series.name).pathname}, images ` +
        `${IMAGES_PRESENT ? "present" : `absent from ${CORPUS_DIR.pathname}`}). Fetch: ${FETCH_COMMAND}`,
    );
  }
  if (!discriminationAvailable) {
    console.warn(
      `[karume] The real-image discrimination of ${series.name} is skipped (mirror ` +
        `${mirror ? "present" : `${mirrorOf(series).pathname} absent — place the published distribution under models/`}, ` +
        `images ${IMAGES_PRESENT ? "present" : `absent from ${CORPUS_DIR.pathname} — fetch: ${FETCH_COMMAND}`}).`,
    );
  }

  for (const entry of REAL_CASES) {
    test.skipIf(!parityAvailable)(
      `BiRefNet real-image preprocessing parity: ${series.name} / ${entry.name}` +
        " (TS's folded normalisation vs handler.py's two steps)" +
        (parityAvailable ? "" : NOT_FETCHED),
      async () => {
        assertAssetPresent(goldenUrl(series, entry.name));
        assertAssetPresent(new URL(entry.file, CORPUS_DIR));
        const [goldenBytes, png] = await Promise.all([
          readBuffer(goldenUrl(series, entry.name)),
          readImage(entry.file),
        ]);
        const io = parseSafetensors(goldenBytes);

        // ① The image the golden was taken from and the image read now are the same. **A
        // difference no tolerance absorbs** (the generation script was rerun without retaking the
        // goldens) is failed by name before the comparison. If this loosens, "green from comparing
        // two different images" becomes possible.
        assertEquals(
          io.metadata.get(SOURCE_IMAGE_KEY),
          entry.file,
          "the source image the golden points to",
        );
        assertEquals(
          io.metadata.get(SOURCE_SHA256_KEY),
          await sha256Hex(png),
          `${entry.file} differs from the image the golden was taken from (retake: ${series.generate})`,
        );

        const view = io.tensors.get(PIXEL_INPUT_KEY);
        assert(view !== undefined, `${PIXEL_INPUT_KEY} is missing from the golden`);
        assertEquals(view.dtype, "F32", `storage dtype of ${PIXEL_INPUT_KEY}`);
        assertEquals(view.shape.length, 4, `rank of ${PIXEL_INPUT_KEY}`);
        const [batch, channels, height, width] = view.shape;
        assertEquals([batch, channels], [1, 3], `leading 2 axes of ${PIXEL_INPUT_KEY}`);
        const expected = new Float32Array(io.buffer, view.byteOffset, view.byteLength / 4);

        // ② The TS route (PNG decode → resize → fused normalisation).
        const decoded = await decodePng(png, entry.file);
        const got = normalizeToNchw(resizeRgb8(decoded, width, height), IMAGE_MEAN, IMAGE_STD);

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
          `${series.name} / ${entry.name}: pixel_values maxAbs=${maxAbs}` +
            ` (sample ${worst}: ${got[worst]} vs ${expected[worst]}) exceeds ${PIXEL_ATOL}`,
        );
      },
    );
  }

  test.skipIf(!discriminationAvailable || !GPU_AVAILABLE)(
    `BiRefNet real-image discrimination: ${series.name} — the foreground ratio of the 2 images with a salient object exceeds that of the 2` +
      " without (it also writes the matte PNGs)" +
      (imagesRunnable ? "" : NOT_FETCHED) +
      (mirror ? "" : MIRROR_ABSENT),
    async () => {
      for (const real of REAL_CASES) assertAssetPresent(new URL(real.file, CORPUS_DIR));
      // The discrimination looks only at **the order of the foreground ratios** (no threshold — the
      // same form as `_real_sanity` of `birefnet/export.py`, which is applied on the torch side). A
      // uniformly collapsed output (all foreground / all background) and an output independent of
      // the input both fail, since the two groups' foreground ratios line up. Measured on the
      // person side: 0.5512 / 0.1139 on BiRefNet_HR and 0.5545 / 0.1134 on Lucida; on the scene
      // side under 0.005 on both.
      //
      // The input is **the real image passed through the TS preprocessing** (not the golden's
      // input) — the check goes as far as "pass a PNG and get a meaningful matte". Along the way it
      // writes the α matte and the composite on white as PNGs (a numeric check alone does not show
      // the shape — artifacts for visual inspection, not covered by the sha). The sha covers the
      // output before quantisation itself (the f32 safetensors of the matte logits —
      // {@link references}), and says something other than the discrimination: "not a single bit
      // moved on the same machine". The order is: run the 4 images → discriminate → append the sha
      // outcome per image → append the series outcome → fail on sha mismatches (no reference row is
      // created from an output whose discrimination broke).
      const started = performance.now();
      const elapsedMs = (): number => Math.round(performance.now() - started);
      /**
       * The foreground ratio per image (material for the discrimination; even a run that fails
       * midway keeps what was measured in the note).
       */
      const ratios = new Map<string, number>();
      const ratioNote = (): string =>
        [...ratios].map(([name, ratio]) => `${name} ${ratio.toFixed(4)}`).join(" / ");
      /**
       * The images whose sha disagreed with the reference (failed together after every image's
       * outcome is appended).
       */
      const mismatches: { readonly label: string; readonly settlement: ReferenceSettlement }[] = [];
      try {
        const parsed = await openMatte(series);
        const [outputName] = parsed.graph.outputs;
        const width = staticDim(parsed, 3);
        const height = staticDim(parsed, 2);
        const inputName = parsed.graph.inputs[0].name;

        /** The per-image sha artifacts (compared after the discrimination passes). */
        const logits: {
          readonly caseId: string;
          readonly artifact: string;
          readonly bytes: Uint8Array<ArrayBuffer>;
          readonly elapsedMs: number;
        }[] = [];
        const gpu = await acquireGpu();
        try {
          // This route can write the matte logits' sha as this machine's row, so first check that
          // the adapter the key was taken from is the running adapter (mixing them up on a
          // multi-GPU machine would make it another machine's row).
          assertAdapterMatchesEnvironment(gpu);
          const session = await parsed.createContainerSession(gpu);
          try {
            // Run the 4 images on one Session (the weights are 964MB — no reason to rebuild per
            // image).
            for (const real of REAL_CASES) {
              const imageStarted = performance.now();
              const image = resizeRgb8(
                await decodePng(await readImage(real.file), real.file),
                width,
                height,
              );
              const pixels = normalizeToNchw(image, IMAGE_MEAN, IMAGE_STD);
              const output = (
                await session.run({
                  [inputName]: { dtype: "f32", shape: [1, 3, height, width], data: pixels },
                })
              )[outputName];
              // Narrow by the discriminant (a cast to Float32Array would pass silently on a dtype
              // drift).
              assert(output.dtype === "f32", `${real.name}: the matte's dtype is ${output.dtype}`);
              ratios.set(real.name, foregroundRatio(output.data));
              logits.push({
                caseId: referenceCaseId(series, real.name),
                artifact: artifactName(series, real.name, "matte.safetensors"),
                bytes: f32ArtifactBytes(outputName, output.shape, output.data),
                elapsedMs: Math.round(performance.now() - imageStarted),
              });

              // The PNGs for visual inspection (images reduced to 8 bit — not covered by the sha).
              const alpha = alphaFromLogits(output.data);
              await writeFile(
                results.artifact(artifactName(series, real.name, "matte.png")),
                await matteToPng(alpha, width, height),
              );
              await writeFile(
                results.artifact(artifactName(series, real.name, "cutout.png")),
                await cutoutToPng(image, alpha),
              );
            }
          } finally {
            await session.dispose();
          }
        } finally {
          gpu.destroy();
        }

        const ratioOf = (name: string): number => {
          const ratio = ratios.get(name);
          assert(ratio !== undefined, `no foreground ratio for ${name}`);
          return ratio;
        };
        for (const personCase of REAL_PERSON_CASES) {
          for (const sceneCase of REAL_SCENE_CASES) {
            assert(
              ratioOf(personCase) > ratioOf(sceneCase),
              `the foreground ratios are in the wrong order: ${personCase}=${ratioOf(personCase)} <=` +
                ` ${sceneCase}=${ratioOf(sceneCase)}`,
            );
          }
        }

        // Only outputs whose discrimination passed are compared with the reference rows (in compare
        // mode, without a row only the comparison is skipped and the measured sha is appended —
        // settleOrObserve).
        for (const entry of logits) {
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
        // Record the outcome before failing (without the foreground ratios at hand, reading the red
        // cannot even start. Leaving without an outcome lets the outcome of the previous run of the
        // same day stay in the results directory). An I/O failure of the record does not replace
        // the original exception (recordFailure).
        await recordFailure(results, {
          id: series.name,
          status: "fail",
          elapsedMs: elapsedMs(),
          note: `${ratioNote()} / failure: ${cause instanceof Error ? cause.message : String(cause)}`,
        });
        throw cause;
      }
      // A sha mismatch shows in the series outcome too (the red is visible from the series row
      // alone).
      await results.record({
        id: series.name,
        status: mismatches.length === 0 ? "pass" : "fail",
        elapsedMs: elapsedMs(),
        note:
          mismatches.length === 0
            ? ratioNote()
            : `${ratioNote()} / sha mismatch: ${mismatches.map(({ label }) => label).join(", ")}`,
      });
      console.log(
        `[karume] wrote the BiRefNet mattes to ${results.dir.pathname}` +
          ` (${series.name}-<case>-matte.safetensors / -matte.png / -cutout.png)`,
      );
      // The outcomes per image and per series are all appended already (throwing earlier would
      // leave no measured sha in the results).
      assert(
        mismatches.length === 0,
        mismatches
          .map(({ label, settlement }) => referenceMismatchMessage(label, settlement, references))
          .join("\n"),
      );
    },
  );
}

// Only the series with both a GPU and the assets run the sha check (the cases of a series without
// them are not counted as missing reference values — the whole section SKIPs).
const RUNNABLE = GPU_AVAILABLE && RUNNABLE_SERIES.length > 0;
const CASE_IDS: readonly string[] = RUNNABLE_SERIES.flatMap((series) =>
  REAL_CASES.map((entry) => referenceCaseId(series, entry.name)),
);
if (RUNNABLE) references.warnMissing(CASE_IDS);
// The gate that keeps "no reference value in this environment yet" from going silently green (the
// same approach as ADR 0005).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });

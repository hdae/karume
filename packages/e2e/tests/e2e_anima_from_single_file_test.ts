/**
 * `anima.Pipeline.fromSingleFile` on the upstream Anima base single file, against the local
 * mirror (real GPU). Each case builds two pipelines on one shared GPU, one after the other, makes
 * one image with each from the same request, and compares the sha256 of their pixels inside the run:
 * the image of `fromPretrained` is the reference, so no sha256 reference row is read or written.
 *
 * 1. `fromSingleFile(file, mirror, { model: "anima-v1.0", quant: "bf16" })` equals
 *    `fromPretrained(mirror, { model: "anima-v1.0", quant: "bf16" })`: every component of the bf16
 *    preset is the upstream bits, so the file converted against its plans gives the same weights.
 * 2. The default quant with `components: { text_conditioner: { source: mirror } }` equals
 *    `fromPretrained` with the default quant: the transformer converted against the `i8` plan gives
 *    the `i8` files' weights.
 * 3. The default quant with the text conditioner from the file, stored against the plan of the
 *    `bf16` text conditioner the preset itself uses: the file's BF16 tensors are the upstream bits
 *    that container holds, so the image equals `fromPretrained`'s at the default quant.
 * 4. Case 2 with `retainConverted: true`, through a reader that counts its reads: the two images of
 *    one pipeline are equal and equal `fromPretrained`'s at the default quant, and the second
 *    generation reads nothing of the file (the transformer's converted weights are kept in memory).
 *
 * The inputs are the upstream file under `inputs/anima/upstream-single-file/` (circlestone-labs/Anima,
 * about 4.2 GB) and the mirror `models/karume-anima/` (both untracked); where one is missing, or
 * there is no GPU, the cases SKIP explicitly with the reason (ADR 0005).
 */

import { isNotFound, readRange, stat } from "@karume/test-support/fs";
import { afterAll, describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import { acquireGpu, type GpuContext } from "@karume/runtime";
import {
  type FromPretrainedOptions,
  type GeneratedImage,
  Pipeline,
  type SingleFileSource,
} from "../../models/anima.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const FILE = new URL(
  "../../../inputs/anima/upstream-single-file/split_files/diffusion_models/anima-base-v1.0.safetensors",
  import.meta.url,
);
const MIRROR_DIR = new URL("../../../models/karume-anima/", import.meta.url);
const MODEL = "anima-v1.0";

const exists = async (url: URL): Promise<boolean> => {
  try {
    await stat(url);
    return true;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const FILE_READY = await exists(FILE);
const MIRROR_READY = await exists(new URL("karume.json", MIRROR_DIR));
if (!FILE_READY || !MIRROR_READY) {
  console.warn(
    `[karume] fromSingleFile on a GPU SKIPs: it needs the upstream file ${FILE.pathname}` +
      ` (${FILE_READY ? "present" : "missing"}) and the mirror ${MIRROR_DIR.pathname}` +
      ` (${MIRROR_READY ? "present" : "missing"})`,
  );
}
const RUNNABLE = GPU_AVAILABLE && FILE_READY && MIRROR_READY;

/** A small generation of the base model (its defaults: CFG 4 with the default negative prompt). */
const REQUEST = {
  prompt:
    "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
    "smile, upper body, masterpiece, best quality",
  resolution: { width: 512, height: 512 },
  steps: 4,
  seed: 42,
} as const;

/** The upstream file as a ranged reader (a 4 GB file is never read whole). */
const upstreamFile = async (): Promise<SingleFileSource> => ({
  size: (await stat(FILE)).size,
  read: (offset, length) => readRange(FILE, offset, length),
});

/** The RGB8 pixels the sha is taken over (`GeneratedImage.data` is RGBA, alpha always 255). */
const rgbPixels = (image: GeneratedImage): Uint8Array<ArrayBuffer> => {
  const pixels = new Uint8Array(new ArrayBuffer(image.width * image.height * 3));
  for (let source = 0, target = 0; source < image.data.length; source += 4, target += 3) {
    pixels[target] = image.data[source];
    pixels[target + 1] = image.data[source + 1];
    pixels[target + 2] = image.data[source + 2];
  }
  return pixels;
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");

/** Builds a pipeline, makes one image, disposes it, and returns the image's pixel sha256. */
const imageSha = async (label: string, build: () => Promise<Pipeline>): Promise<string> => {
  const started = performance.now();
  const pipeline = await build();
  const built = performance.now();
  try {
    const image = await pipeline.generate(REQUEST);
    const sha = await sha256Hex(rgbPixels(image));
    console.log(
      `[e2e] ${label}: built in ${((built - started) / 1000).toFixed(1)} s, generated in ${(
        (performance.now() - built) /
        1000
      ).toFixed(1)} s, pixels sha256 ${sha}`,
    );
    return sha;
  } finally {
    await pipeline.dispose();
  }
};

/** One shared GPU for every pipeline of the file (each pipeline is disposed before the next). */
let gpu: Promise<GpuContext> | undefined;
const sharedGpu = (): Promise<GpuContext> => (gpu ??= acquireGpu());

const optionsOf = async (quant?: string): Promise<FromPretrainedOptions> => ({
  gpu: await sharedGpu(),
  model: MODEL,
  ...(quant === undefined ? {} : { quant }),
});

/** The reference image of `fromPretrained` at the default quant (shared by cases 2 and 3). */
let defaultReference: Promise<string> | undefined;
const defaultPretrainedSha = (): Promise<string> =>
  (defaultReference ??= (async () =>
    imageSha(`fromPretrained ${MODEL} (default quant)`, async () =>
      Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), await optionsOf()),
    ))());

describe("anima.Pipeline.fromSingleFile on the upstream base file (real GPU)", () => {
  afterAll(async () => {
    if (gpu !== undefined) (await gpu).destroy();
  });

  it.skipIf(!RUNNABLE)(
    "quant bf16: the image equals fromPretrained's of the same model and quant",
    async () => {
      const mirror = nodeDirectory(MIRROR_DIR);
      const fromFile = await imageSha(`fromSingleFile ${MODEL} / bf16`, async () =>
        Pipeline.fromSingleFile(await upstreamFile(), mirror, await optionsOf("bf16")),
      );
      const pretrained = await imageSha(`fromPretrained ${MODEL} / bf16`, async () =>
        Pipeline.fromPretrained(mirror, await optionsOf("bf16")),
      );
      assertEquals(fromFile, pretrained);
    },
  );

  it.skipIf(!RUNNABLE)(
    "the default quant with the distribution's text conditioner: the image equals fromPretrained's at the default quant",
    async () => {
      const mirror = nodeDirectory(MIRROR_DIR);
      const fromFile = await imageSha(
        `fromSingleFile ${MODEL} (default quant, the distribution's text conditioner)`,
        async () =>
          Pipeline.fromSingleFile(await upstreamFile(), mirror, {
            ...(await optionsOf()),
            components: { text_conditioner: { source: mirror } },
          }),
      );
      assertEquals(fromFile, await defaultPretrainedSha());
    },
  );

  it.skipIf(!RUNNABLE)(
    "the default quant with the text conditioner from the file: the image equals fromPretrained's at the default quant",
    async () => {
      const fromFile = await imageSha(
        `fromSingleFile ${MODEL} (default quant, the file's text conditioner)`,
        async () =>
          Pipeline.fromSingleFile(
            await upstreamFile(),
            nodeDirectory(MIRROR_DIR),
            await optionsOf(),
          ),
      );
      // The preset's own text conditioner is the `bf16` container, which holds the upstream bits,
      // and the file's BF16 tensors are stored as bf16 against its plan: the same weights.
      assertEquals(fromFile, await defaultPretrainedSha());
    },
  );

  it.skipIf(!RUNNABLE)(
    "retainConverted at the default quant with the distribution's text conditioner: two equal images, equal to fromPretrained's, and the second generation reads nothing of the file",
    async () => {
      const mirror = nodeDirectory(MIRROR_DIR);
      const size = (await stat(FILE)).size;
      let reads = 0;
      const file: SingleFileSource = {
        size,
        read: (offset, length) => {
          reads += 1;
          return readRange(FILE, offset, length);
        },
      };
      const pipeline = await Pipeline.fromSingleFile(file, mirror, {
        ...(await optionsOf()),
        components: { text_conditioner: { source: mirror } },
        retainConverted: true,
      });
      const shas: string[] = [];
      const readsAfter: number[] = [reads];
      try {
        for (const index of [1, 2]) {
          const started = performance.now();
          shas.push(await sha256Hex(rgbPixels(await pipeline.generate(REQUEST))));
          readsAfter.push(reads);
          console.log(
            `[e2e] fromSingleFile ${MODEL} (default quant, retainConverted): generation ${index} in ${(
              (performance.now() - started) /
              1000
            ).toFixed(
              1,
            )} s, ${readsAfter[index] - readsAfter[index - 1]} reads of the file, pixels sha256 ${shas[index - 1]}`,
          );
        }
      } finally {
        await pipeline.dispose();
      }
      const [afterBuild, afterFirst, afterSecond] = readsAfter;
      assert(afterFirst > afterBuild, "the first generation reads the transformer's tensors");
      assertEquals(afterSecond, afterFirst, "the second generation read the file");
      assertEquals(shas[1], shas[0]);
      assertEquals(shas[0], await defaultPretrainedSha());
    },
  );
});

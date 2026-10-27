/**
 * What the MOSS-Image VAE tests share (ADR NNNN decision 8, plan step 3): where the VAE series and
 * its goldens live, the golden case table, the golden reader and the components and contract opened
 * from the series containers.
 *
 * The case table mirrors `tools/export-recipes/moss/export_vae.py` (`CASES` / `CASES_WITH_CAP`) and
 * is **fixed here, not enumerated from the files** (a partial generation must not make cases
 * disappear silently — the Wan rule).
 */

import { readFile } from "@karume/test-support/fs";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../../runtime/tests/helpers/container-files.ts";
import {
  MOSS_VAE_DECODER_ATTN,
  MOSS_VAE_DECODER_POST,
  MOSS_VAE_DECODER_PRE,
  mossVaeContract,
  type MossVaeContract,
  mossVaeGraphContract,
  readMossLatentBatchNorm,
} from "../../src/moss/vae-contract.ts";
import type { MossVaeComponents } from "../../src/moss/vae-stage.ts";
import { componentOf, filePresent, floatsOf, readBuffer } from "./moss-dit.ts";
import { seriesGraph } from "./series-graphs.ts";

export const MOSS_VAE_SERIES_NAME = "moss-image-sft-f32-dyn";
const SERIES_DIR = new URL(`../../../../outputs/series/${MOSS_VAE_SERIES_NAME}/`, import.meta.url);

const RECIPE = "cd tools/export-recipes && uv run --group moss --inexact python -m";
export const MOSS_GENERATE_VAE_SERIES = `${RECIPE} moss.export_vae write`;
export const MOSS_GENERATE_VAE_CAP = `${RECIPE} moss.export_vae write --cap`;
export const MOSS_GENERATE_VAE_STAGE0 = `${RECIPE} moss.vae_reference`;

/** The component directories of the series (= graph names — `series-graphs.ts`). */
export const MOSS_VAE_COMPONENTS = [
  MOSS_VAE_DECODER_PRE,
  MOSS_VAE_DECODER_ATTN,
  MOSS_VAE_DECODER_POST,
] as const;
type ComponentDir = (typeof MOSS_VAE_COMPONENTS)[number];

const modelUrl = (component: ComponentDir): URL => new URL(`${component}/model.krm`, SERIES_DIR);

/** Every container of the series is written. */
export const mossVaeSeriesPresent = (): boolean =>
  MOSS_VAE_COMPONENTS.every((c) => modelPresent(modelUrl(c)));

/**
 * The sha256 of each VAE container's part 0 (the two documents — they name every block's sha256),
 * as the end-to-end goldens of `moss.few_step_ref` record them beside the DiT's
 * (`containerDigests` of `./moss-dit.ts`).
 */
export const mossVaeContainerDigests = async (): Promise<Readonly<Record<string, string>>> => {
  const digests: Record<string, string> = {};
  for (const component of MOSS_VAE_COMPONENTS) {
    const [part0] = resolveParts(modelUrl(component));
    const digest = await crypto.subtle.digest("SHA-256", await readFile(part0));
    digests[component] = [...new Uint8Array(digest)]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }
  return digests;
};

export type MossVaeRole = "band" | "accept" | "cap";

/** One VAE golden (`export_vae.VaeCase`). */
export type MossVaeCase = {
  readonly name: string;
  readonly height: number;
  readonly width: number;
  readonly seed: number;
  readonly role: MossVaeRole;
};

const caseOf = (height: number, width: number, seed: number, role: MossVaeRole): MossVaeCase => ({
  name: `${height}x${width}-${role}-s${seed}`,
  height,
  width,
  seed,
  role,
});

/** The stage-0 VAE seed (`moss.vae_reference.SEED`) — one acceptance case per size. */
export const MOSS_VAE_STAGE0_SEED = 20261028;

const BAND_SEEDS = [20261101, 20261102, 20261103] as const;
const ACCEPT_SEEDS = [MOSS_VAE_STAGE0_SEED, 20261104] as const;

/** The default cases: 256² and 512×768 (height × width), three band and two acceptance each. */
export const MOSS_VAE_CASES: readonly MossVaeCase[] = (
  [
    [256, 256],
    [512, 768],
  ] as const
).flatMap(([height, width]) => [
  ...BAND_SEEDS.map((seed) => caseOf(height, width, seed, "band")),
  ...ACCEPT_SEEDS.map((seed) => caseOf(height, width, seed, "accept")),
]);

/** The area-cap case (written by `write --cap` — 1696×1648, the largest area at a 2 GiB binding). */
export const MOSS_VAE_CAP_CASE: MossVaeCase = caseOf(1696, 1648, 20261105, "cap");

export const vaeGoldenUrl = (testCase: MossVaeCase): URL =>
  new URL(`vae_decoder.${testCase.name}.safetensors`, SERIES_DIR);

export const vaeGoldenPresent = (testCase: MossVaeCase): boolean =>
  filePresent(vaeGoldenUrl(testCase));

const METADATA_KEY = "karume.moss.vae_golden";

/** One golden's tensors (`latents` / `z` / `decoded` / `image`, f32), its metadata checked. */
export type MossVaeGolden = {
  readonly latents: Float32Array<ArrayBuffer>;
  readonly z: Float32Array<ArrayBuffer>;
  readonly decoded: Float32Array<ArrayBuffer>;
  readonly image: Float32Array<ArrayBuffer>;
};

export const loadVaeGolden = async (testCase: MossVaeCase): Promise<MossVaeGolden> => {
  const file: SafetensorsFile = parseSafetensors(await readBuffer(vaeGoldenUrl(testCase)));
  const raw = file.metadata.get(METADATA_KEY);
  if (raw === undefined) throw new Error(`${testCase.name}: no '${METADATA_KEY}' metadata`);
  const metadata: unknown = JSON.parse(raw);
  if (typeof metadata !== "object" || metadata === null) {
    throw new Error(`${testCase.name}: the golden metadata is not an object`);
  }
  const fields = Object.fromEntries(Object.entries(metadata));
  const expected = {
    case: testCase.name,
    role: testCase.role,
    seed: testCase.seed,
    size: [testCase.height, testCase.width],
  };
  for (const [key, value] of Object.entries(expected)) {
    if (JSON.stringify(fields[key]) !== JSON.stringify(value)) {
      throw new Error(`${testCase.name}: golden '${key}' is ${JSON.stringify(fields[key])}`);
    }
  }
  const where = `golden ${testCase.name}`;
  return {
    latents: floatsOf(file, "latents", where),
    z: floatsOf(file, "z", where),
    decoded: floatsOf(file, "decoded", where),
    image: floatsOf(file, "image", where),
  };
};

/** The opened VAE series: its components and its contract (the asset read). */
export type MossVaeSeries = {
  readonly components: MossVaeComponents;
  readonly contract: MossVaeContract;
};

export const openMossVaeSeries = async (): Promise<MossVaeSeries> => {
  const component = async (dir: ComponentDir) =>
    componentOf(await openSeriesContainer(modelUrl(dir)), seriesGraph(MOSS_VAE_SERIES_NAME, dir));
  const components: MossVaeComponents = {
    pre: await component(MOSS_VAE_DECODER_PRE),
    attn: await component(MOSS_VAE_DECODER_ATTN),
    post: await component(MOSS_VAE_DECODER_POST),
  };
  const graph = mossVaeGraphContract(components, "MOSS-Image");
  return {
    components,
    contract: mossVaeContract(graph, await readMossLatentBatchNorm(components.pre), "MOSS-Image"),
  };
};

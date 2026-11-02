/**
 * What the MOSS-Image DiT tests share (ADR 0138, stage 2): where the containers, the goldens and the
 * stage-0 records live, the golden case table, the golden readers and the components opened from
 * the distribution's containers.
 *
 * Where the assets come from (ADR 0139): the containers are read from the distribution mirror
 * `models/karume-moss-image/` through its manifest (model `sft`; the graph name is the manifest's
 * weights key). The goldens of `tools/export-recipes/moss/dit_reference.py` and upstream's prompt
 * embeddings (`moss.text_embeds`) are read from the test-asset root (`goldenDir(<series>)`, fetched
 * with `node tools/test-assets/fetch.ts`) at the relative paths they had under `outputs/series/`.
 * The stage-0 records of upstream's own run (`moss.host_fixture`, `moss.reference`,
 * `moss.vae_reference` — written under `outputs/bench/karume-moss-image/<date>_stage0-reference/`)
 * are uploaded to and read from the test assets too ({@link stage0Record}).
 *
 * The case table mirrors `tools/export-recipes/moss/dit_reference.py` (`DIT_CASES` / `TEXT_CASES`)
 * and is **fixed here, not enumerated from the files** (a partial generation must not make cases
 * disappear silently — the Wan rule, `wan-ti2v-dit.ts`).
 */

import { assert } from "@std/assert";
import { isNotFound, readFile, readRange, statSync } from "@karume/test-support/fs";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import {
  type GpuContext,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type ResidentTensor,
  type SafetensorsFile,
  type TensorView,
} from "@karume/runtime";
import { parseSafetensorsHeader } from "../../../runtime/src/format/safetensors.ts";
import type { ModelComponent } from "../../src/hub/components.ts";
import type { MossDitComponents } from "../../src/moss/dit-stage.ts";
import type { MossTrainingGrid } from "../../src/moss/dit-timestep.ts";
import {
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "./mirror.ts";

/** The distribution mirror every MOSS-Image test opens its containers from (ADR 0139 decision 1). */
export const MOSS_MIRROR: URL = mirrorRoot("karume-moss-image");
/** The mirror's one model (upstream's SFT checkpoint). */
const MOSS_MODEL = "sft";
/** How to get the mirror (pasted into the SKIP messages). */
export const MOSS_GET_MIRROR =
  "hf download hdae/karume-moss-image --local-dir models/karume-moss-image";
/** How to get the goldens (pasted into the SKIP messages — ADR 0139 decision 2). */
export const MOSS_FETCH_GOLDENS = "node tools/test-assets/fetch.ts";

/** The mirror holds a manifest this version reads (a broken part fails loudly at open). */
export const mossMirrorPresent = (): boolean => mirrorPresent(MOSS_MIRROR);

/** The series name of the DiT (the golden directory names derive from it). */
export const MOSS_SERIES_NAME = "moss-image-sft-i8-dyn";
export const MOSS_REFERENCE_DIR: URL = goldenDir(`${MOSS_SERIES_NAME}-reference`);
export const MOSS_TEXT_EMBEDS: URL = new URL(
  "text_embeds.safetensors",
  goldenDir("moss-image-sft-text-embeds"),
);

const RUN = "uv run --group moss --inexact python -m";
const RECIPE = `cd tools/export-recipes && ${RUN}`;
export const MOSS_GENERATE_GOLDENS = `${RECIPE} moss.dit_reference write`;
// One `cd`: a second one would resolve inside tools/export-recipes and fail.
export const MOSS_GENERATE_STAGE0 = `${RECIPE} moss.host_fixture && ${RUN} moss.reference`;

/** The DiT components (= graph names = the manifest's weights keys). */
export const MOSS_COMPONENTS = ["dit", "dit_skip22", "dit_text"] as const;
type ComponentDir = (typeof MOSS_COMPONENTS)[number];

/** Opens the mirror's container of `key` (descriptors only — weights are read by Sessions). */
export const openMossMirrorContainer = async (key: string): Promise<OpenedContainer> =>
  await openSelectedContainer(await selectMirrorComponent(MOSS_MIRROR, key, { model: MOSS_MODEL }));

export type MossCaseRole = "band" | "accept";

/** One DiT golden (`dit_reference.DitCase`). */
export type MossDitCase = {
  readonly name: string;
  readonly group: "small" | "full";
  readonly role: MossCaseRole;
  readonly branch: "positive" | "negative";
  readonly gridIndex: number;
};

/** One `dit_text` golden (`dit_reference.TextCase`). */
export type MossTextCase = {
  readonly name: string;
  readonly role: MossCaseRole;
  readonly prompt: string;
  readonly tokens: number;
};

const BAND_INDICES = [0, 400, 850] as const;
const ACCEPT_INDICES = [200, 950] as const;

const ditCases = (group: "small" | "full"): MossDitCase[] =>
  (
    [
      ["band", BAND_INDICES],
      ["accept", ACCEPT_INDICES],
    ] as const
  ).flatMap(([role, indices]) =>
    (["positive", "negative"] as const).flatMap((branch) =>
      indices.map((gridIndex) => ({
        name: `${group}-${role}-${branch}-g${String(gridIndex).padStart(4, "0")}`,
        group,
        role,
        branch,
        gridIndex,
      })),
    ),
  );

/** The DiT goldens: per group six band and four acceptance cases, both branches. */
export const MOSS_DIT_CASES: readonly MossDitCase[] = [...ditCases("small"), ...ditCases("full")];

const textCase = (role: MossCaseRole, prompt: string, tokens: number): MossTextCase => ({
  name: `text-${role}-${prompt}-t${String(tokens).padStart(4, "0")}`,
  role,
  prompt,
  tokens,
});

/** The `dit_text` goldens: six band and four acceptance cases (T = 1 is an acceptance case). */
export const MOSS_TEXT_CASES: readonly MossTextCase[] = [
  textCase("band", "guide-whiteboard", 2),
  textCase("band", "negative", 10),
  textCase("band", "fridge-note", 31),
  textCase("band", "guide-fridge-note", 62),
  textCase("band", "fridge-note", 63),
  textCase("band", "guide-whiteboard", 134),
  textCase("accept", "fridge-note", 1),
  textCase("accept", "negative", 7),
  textCase("accept", "guide-fridge-note", 45),
  textCase("accept", "guide-whiteboard", 100),
];

/** The file of a golden. */
export const goldenUrl = (name: string): URL => new URL(`${name}.safetensors`, MOSS_REFERENCE_DIR);

/** A file exists (NotFound only — another I/O error propagates, a broken mount is not a SKIP). */
export const filePresent = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/**
 * Whether a default-lane case that reads the test assets `urls` runs (the rule of
 * `testAssetsFetched`, ADR 0139): when every file is present, and also when one is absent from a
 * complete fetch of the pinned dataset — the case then FAILs in {@link assertTestAssetsPresent},
 * since the dataset at the pin lacks it. Otherwise an absent file SKIPs the case (the assets gate is
 * red then).
 */
export const testAssetsRunnable = (urls: readonly URL[]): boolean =>
  urls.every(filePresent) || testAssetsFetched();

/** FAILs naming every absent file of `urls` (see {@link testAssetsRunnable}). */
export const assertTestAssetsPresent = (urls: readonly URL[]): void => {
  const missing = urls.filter((url) => !filePresent(url)).map((url) => url.pathname);
  if (missing.length > 0) {
    throw new Error(
      `${missing.join(", ")}: absent, but outputs/test-assets/ holds a complete fetch of` +
        ` ${TEST_ASSETS.repo}@${TEST_ASSETS.revision} — the dataset at the pin lacks it`,
    );
  }
};

export const readBuffer = async (url: URL): Promise<ArrayBuffer> =>
  (await readFile(url)).slice().buffer;

/** One golden: its tensors and its metadata (the single JSON key of `dit_reference`). */
export type MossGolden = {
  readonly file: SafetensorsFile;
  readonly metadata: Readonly<Record<string, unknown>>;
};

const METADATA_KEY = "karume.moss.dit_reference";

export const loadGolden = async (name: string): Promise<MossGolden> => {
  const file = parseSafetensors(await readBuffer(goldenUrl(name)));
  const raw = file.metadata.get(METADATA_KEY);
  if (raw === undefined) throw new Error(`${name}: the golden has no '${METADATA_KEY}' metadata`);
  const metadata: unknown = JSON.parse(raw);
  if (typeof metadata !== "object" || metadata === null) {
    throw new Error(`${name}: the golden metadata is not an object`);
  }
  return { file, metadata: Object.fromEntries(Object.entries(metadata)) };
};

export const viewOf = (file: SafetensorsFile, key: string, where: string): TensorView => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`${where}: no '${key}'`);
  return view;
};

export const floatsOf = (
  file: SafetensorsFile,
  key: string,
  where: string,
): Float32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "F32") throw new Error(`${where}: '${key}' is ${view.dtype}`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

export const intOf = (file: SafetensorsFile, key: string, where: string): number => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "I32" || view.byteLength !== 4)
    throw new Error(`${where}: '${key}' is not one I32`);
  return new Int32Array(file.buffer, view.byteOffset, 1)[0];
};

/** The bool tensor of a golden as u32 0 / 1 (safetensors stores torch bool as one byte). */
export const boolsOf = (
  file: SafetensorsFile,
  key: string,
  where: string,
): Uint32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "BOOL") throw new Error(`${where}: '${key}' is ${view.dtype}`);
  return Uint32Array.from(new Uint8Array(file.buffer, view.byteOffset, view.byteLength));
};

/** The index of the first element whose bits differ (-1 = identical; 0 when the lengths differ). */
export const firstBitMismatch = (
  actual: Float32Array | Uint32Array,
  expected: Float32Array | Uint32Array,
): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

const hex = (digest: ArrayBuffer): string =>
  [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");

/**
 * The sha256 of an opened container's part 0 (the two documents — they name every block's sha256),
 * read through the manifest's source.
 */
export const part0Digest = async (opened: OpenedContainer): Promise<string> =>
  hex(
    await crypto.subtle.digest(
      "SHA-256",
      await opened.source.read(0, 0, opened.source.partLength(0)),
    ),
  );

/**
 * The sha256 of each DiT container's part 0, as the goldens record them
 * (`dit_reference.part0_sha256` — all three containers: a rewritten `dit_skip22` also makes the
 * negative goldens stale). The mirror's containers are byte copies of the series the goldens were
 * taken from (ADR 0139), so the recorded digests hold against them.
 */
export const containerDigests = async (): Promise<Readonly<Record<string, string>>> => {
  const digests: Record<string, string> = {};
  for (const component of MOSS_COMPONENTS) {
    digests[component] = await part0Digest(await openMossMirrorContainer(component));
  }
  return digests;
};

/**
 * Fails loudly when a golden was taken from other containers than the mirror's (re-export, re-run
 * the golden writer after `moss.export_dit write`, and upload the goldens — ADR 0139 decision 8).
 */
export const assertGoldenOfSeries = (
  golden: MossGolden,
  digests: Readonly<Record<string, string>>,
  where: string,
): void => {
  const recorded = golden.metadata["containers"];
  const same =
    typeof recorded === "object" &&
    recorded !== null &&
    Object.keys(digests).every(
      (key) =>
        Object.hasOwn(recorded, key) &&
        Object.getOwnPropertyDescriptor(recorded, key)?.value === digests[key],
    );
  if (!same) {
    throw new Error(
      `${where}: the golden was taken from other containers (${JSON.stringify(recorded)}) than the` +
        ` mirror (${JSON.stringify(digests)}) — regenerate: ${MOSS_GENERATE_GOLDENS}`,
    );
  }
};

/** The metadata a golden of `testCase` carries (`dit_reference.write`). */
const expectedMetadata = (
  testCase: MossDitCase | MossTextCase,
): Readonly<Record<string, unknown>> =>
  "gridIndex" in testCase
    ? {
        case: testCase.name,
        group: testCase.group,
        role: testCase.role,
        graph: testCase.branch === "positive" ? "dit" : "dit_skip22",
        prompt: testCase.branch === "positive" ? "fridge-note" : "negative",
        grid_index: testCase.gridIndex,
      }
    : {
        case: testCase.name,
        group: "text",
        role: testCase.role,
        graph: "dit_text",
        prompt: testCase.prompt,
        T: testCase.tokens,
      };

/**
 * Fails loudly when a golden is not the case the table names (a renamed file, or a writer whose case
 * table drifted from this one).
 */
export const assertGoldenOfCase = (
  golden: MossGolden,
  testCase: MossDitCase | MossTextCase,
): void => {
  const wrong = Object.entries(expectedMetadata(testCase)).filter(
    ([key, value]) => !Object.hasOwn(golden.metadata, key) || golden.metadata[key] !== value,
  );
  if (wrong.length > 0) {
    throw new Error(
      `${testCase.name}: the golden's metadata does not match the case table (${wrong
        .map(
          ([key, value]) =>
            `${key} ${JSON.stringify(golden.metadata[key])}, expected ${JSON.stringify(value)}`,
        )
        .join("; ")})`,
    );
  }
};

/** A case's golden, refused when stale ({@link assertGoldenOfSeries}) or of another case. */
export const loadCaseGolden = async (
  testCase: MossDitCase | MossTextCase,
  digests: Readonly<Record<string, string>>,
): Promise<MossGolden> => {
  const golden = await loadGolden(testCase.name);
  assertGoldenOfSeries(golden, digests, testCase.name);
  assertGoldenOfCase(golden, testCase);
  return golden;
};

/** The opened DiT: per component its container, prepared model and {@link ModelComponent}. */
export type MossDit = {
  readonly opened: Readonly<Record<ComponentDir, OpenedContainer>>;
  readonly components: MossDitComponents;
};

/**
 * A {@link ModelComponent} over an opened container (the fields `hub/components.ts`'s
 * `containerComponent` builds from a fetched one; the largest weight part is part 1 onward).
 */
export const componentOf = (opened: OpenedContainer, graph: string): ModelComponent => {
  const prepared = prepareContainer(opened, graph);
  let maxPartBytes = 0;
  for (let index = 1; index < opened.source.partCount; index += 1) {
    maxPartBytes = Math.max(maxPartBytes, opened.source.partLength(index));
  }
  return {
    graph: prepared.graph,
    createSession: (gpu, options = {}) => prepared.createContainerSession(gpu, options),
    assets: Object.fromEntries(
      Object.entries(opened.model?.assets ?? {}).map(([name, asset]) => [name, asset.role]),
    ),
    asset: (name) => opened.asset(name),
    estimate: (options) => prepared.estimate(options),
    maxPartBytes,
  };
};

/**
 * Opens the mirror's three DiT containers (descriptors only — weights are read by Sessions); each
 * graph is named by its weights key.
 */
export const openMossDit = async (): Promise<MossDit> => {
  const opened = {
    dit: await openMossMirrorContainer("dit"),
    dit_skip22: await openMossMirrorContainer("dit_skip22"),
    dit_text: await openMossMirrorContainer("dit_text"),
  };
  return {
    opened,
    components: {
      dit: componentOf(opened.dit, "dit"),
      ditSkip22: componentOf(opened.dit_skip22, "dit_skip22"),
      ditText: componentOf(opened.dit_text, "dit_text"),
    },
  };
};

/**
 * A strictly descending training grid of `points` from 1 (σ_i = 1 − i·(1 − 1/points)/(points − 1),
 * timesteps σ × points) for the tests that need a grid's structure but not upstream's values.
 */
export const syntheticTrainingGrid = (points: number): MossTrainingGrid => {
  const sigmas = Float32Array.from({ length: points }, (_, index) =>
    Math.fround(1 - (index * (1 - 1 / points)) / (points - 1)),
  );
  return {
    points,
    width: 1,
    sigmas,
    timesteps: sigmas.map((sigma) => Math.fround(sigma * points)),
    sinusoid: new Float32Array(points),
  };
};

/**
 * Runs `body` with every resident tensor `gpu.createResident` makes collected into the list it is
 * given (an instance property shadowing the class's method, removed afterwards).
 */
export const withResidentLog = async <T>(
  gpu: GpuContext,
  body: (residents: readonly ResidentTensor[]) => Promise<T>,
): Promise<T> => {
  assert(!Object.hasOwn(gpu, "createResident"), "createResident is already shadowed");
  const residents: ResidentTensor[] = [];
  const create = gpu.createResident.bind(gpu);
  Object.defineProperty(gpu, "createResident", {
    configurable: true,
    value: async (byteLength: number, label?: string) => {
      const resident = await create(byteLength, label);
      residents.push(resident);
      return resident;
    },
  });
  try {
    return await body(residents);
  } finally {
    Reflect.deleteProperty(gpu, "createResident");
  }
};

/**
 * MUST: a run that writes this family's sha rows (`KARUME_REFERENCE=write` or `rewrite` — `mode`)
 * settles a row only after every upstream parity case of the same test file (`parity.expected`) has
 * passed earlier in the same run (`parity.passed` — a parity test adds its id only after all its
 * assertions held, and the file declares its parity before its row-settling tests). A row is
 * written the moment it is settled and the default lane checks only the shape and finiteness of an
 * output, so a broken but finite build would otherwise be pinned. Compare mode (`mode` undefined)
 * does not depend on the parity.
 */
export const assertMossRowWritable = (
  mode: string | undefined,
  parity: { readonly expected: readonly string[]; readonly passed: ReadonlySet<string> },
  caseId: string,
): void => {
  if (mode === undefined) return;
  const failed = parity.expected.filter((id) => !parity.passed.has(id));
  if (failed.length === 0) return;
  throw new Error(
    `${caseId}: KARUME_REFERENCE=${mode} writes MOSS-Image sha rows only after every upstream` +
      " parity case of this file passed earlier in the same run — not passed:" +
      ` ${failed.join(", ")} (rerun the whole file with KARUME_UPSTREAM_PARITY=1 and the parity` +
      " goldens present, so that a finite but wrong output is not pinned)",
  );
};

/** One stage-0 record in the test assets (`goldens/moss-image-sft-stage0-reference/<file>`). */
export const stage0Record = (file: string): URL =>
  new URL(file, goldenDir("moss-image-sft-stage0-reference"));

/**
 * Selected F32 tensors of a large safetensors record, read by range (the stage-0 DiT records are
 * 0.6–0.9 GB; the tests need a few MB of them).
 */
export const readRecordTensors = async (
  url: URL,
  keys: readonly string[],
): Promise<Readonly<Record<string, Float32Array<ArrayBuffer>>>> => {
  const prefix = await readRange(url, 0, 8);
  const headerLength = Number(new DataView(prefix.buffer).getBigUint64(0, true));
  const head = await readRange(url, 0, 8 + headerLength);
  const header = parseSafetensorsHeader(head, statSync(url).size);
  const out: Record<string, Float32Array<ArrayBuffer>> = {};
  for (const key of keys) {
    const view = header.tensors.get(key);
    if (view === undefined) throw new Error(`${url.pathname}: no '${key}'`);
    if (view.dtype !== "F32") throw new Error(`${url.pathname}: '${key}' is ${view.dtype}`);
    const bytes = await readRange(url, view.byteOffset, view.byteLength);
    out[key] = new Float32Array(bytes.buffer, 0, view.byteLength / 4);
  }
  return out;
};

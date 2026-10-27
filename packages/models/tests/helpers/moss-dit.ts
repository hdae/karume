/**
 * What the MOSS-Image DiT tests share (ADR NNNN, plan step 2): where the series, the goldens and the
 * stage-0 records live, the golden case table, the golden readers and the components opened from
 * the series containers.
 *
 * The case table mirrors `tools/export-recipes/moss/dit_reference.py` (`DIT_CASES` / `TEXT_CASES`)
 * and is **fixed here, not enumerated from the files** (a partial generation must not make cases
 * disappear silently — the Wan rule, `wan-ti2v-dit.ts`).
 */

import { isNotFound, readDirSync, readFile, readRange, statSync } from "@karume/test-support/fs";
import {
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type SafetensorsFile,
  type TensorView,
} from "@karume/runtime";
import { parseSafetensorsHeader } from "../../../runtime/src/format/safetensors.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../../runtime/tests/helpers/container-files.ts";
import type { ModelComponent } from "../../src/hub/components.ts";
import type { MossDitComponents } from "../../src/moss/dit-stage.ts";
import { seriesGraph } from "./series-graphs.ts";

export const MOSS_SERIES_NAME = "moss-image-sft-i8-dyn";
const OUTPUTS = new URL("../../../../outputs/", import.meta.url);
const SERIES_DIR = new URL(`series/${MOSS_SERIES_NAME}/`, OUTPUTS);
export const MOSS_REFERENCE_DIR = new URL(`series/${MOSS_SERIES_NAME}-reference/`, OUTPUTS);
export const MOSS_TEXT_EMBEDS = new URL(
  "series/moss-image-sft-text-embeds/text_embeds.safetensors",
  OUTPUTS,
);
const BENCH_DIR = new URL("bench/karume-moss-image/", OUTPUTS);

const RUN = "uv run --group moss --inexact python -m";
const RECIPE = `cd tools/export-recipes && ${RUN}`;
export const MOSS_GENERATE_SERIES = `${RECIPE} moss.export_dit write`;
export const MOSS_GENERATE_GOLDENS = `${RECIPE} moss.dit_reference write`;
// One `cd`: a second one would resolve inside tools/export-recipes and fail.
export const MOSS_GENERATE_STAGE0 = `${RECIPE} moss.host_fixture && ${RUN} moss.reference`;
export const MOSS_GENERATE_EMBEDS = `${RECIPE} moss.text_embeds`;

/** The component directories of the series (= graph names — `series-graphs.ts`). */
export const MOSS_COMPONENTS = ["dit", "dit_skip22", "dit_text"] as const;
type ComponentDir = (typeof MOSS_COMPONENTS)[number];

const modelUrl = (component: ComponentDir): URL => new URL(`${component}/model.krm`, SERIES_DIR);

/** Every container of the series is written. */
export const mossSeriesPresent = (): boolean =>
  MOSS_COMPONENTS.every((c) => modelPresent(modelUrl(c)));

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
 * The sha256 of each written container's part 0 (the two documents — they name every block's
 * sha256), as the goldens record them (`dit_reference.part0_sha256` — all three containers: a
 * rewritten `dit_skip22` also makes the negative goldens stale).
 */
export const containerDigests = async (): Promise<Readonly<Record<string, string>>> => {
  const digests: Record<string, string> = {};
  for (const component of MOSS_COMPONENTS) {
    const [part0] = resolveParts(modelUrl(component));
    digests[component] = hex(await crypto.subtle.digest("SHA-256", await readFile(part0)));
  }
  return digests;
};

/**
 * Fails loudly when a golden was taken from other containers than the series on disk (re-run the
 * golden writer after `moss.export_dit write`).
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
        ` series (${JSON.stringify(digests)}) — regenerate: ${MOSS_GENERATE_GOLDENS}`,
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

/** The opened series: per component its container, prepared model and {@link ModelComponent}. */
export type MossSeries = {
  readonly opened: Readonly<Record<ComponentDir, OpenedContainer>>;
  readonly components: MossDitComponents;
};

/**
 * A {@link ModelComponent} over an opened series container (the fields `hub/components.ts`'s
 * `containerComponent` builds from a fetched one; the largest weight part is part 1 onward).
 */
const componentOf = (opened: OpenedContainer, graph: string): ModelComponent => {
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

/** Opens the three containers of the series (descriptors only — weights are read by Sessions). */
export const openMossSeries = async (): Promise<MossSeries> => {
  const opened = {
    dit: await openSeriesContainer(modelUrl("dit")),
    dit_skip22: await openSeriesContainer(modelUrl("dit_skip22")),
    dit_text: await openSeriesContainer(modelUrl("dit_text")),
  };
  return {
    opened,
    components: {
      dit: componentOf(opened.dit, seriesGraph(MOSS_SERIES_NAME, "dit")),
      ditSkip22: componentOf(opened.dit_skip22, seriesGraph(MOSS_SERIES_NAME, "dit_skip22")),
      ditText: componentOf(opened.dit_text, seriesGraph(MOSS_SERIES_NAME, "dit_text")),
    },
  };
};

/** The newest stage-0 record directory (`<date>_stage0-reference`), or undefined. */
export const stage0Dir = (): URL | undefined => {
  let names: string[];
  try {
    names = readDirSync(BENCH_DIR)
      .filter((entry) => entry.isDirectory && entry.name.endsWith("_stage0-reference"))
      .map((entry) => entry.name)
      .sort();
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
  const last = names.at(-1);
  return last === undefined ? undefined : new URL(`${last}/`, BENCH_DIR);
};

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

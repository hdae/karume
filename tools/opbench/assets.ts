/**
 * census の入力側: **資産ディレクトリ → コンポーネントごとのグラフ shard** の解決。
 *
 * 読むのは safetensors の**ヘッダ JSON だけ**（IR は先頭 shard の `__metadata__.karume_ir` に
 * 載る — ADR 0070 決定 3 / ADR 0081）。実体は合計 GB 級なので 1 バイトも読まない。
 *
 * 資産は 2 形ある:
 *
 * - **配布形** — `karume.json`（manifest）を持つディレクトリ。shard の綴りを決めるのは manifest
 *   なので、ファイル名の推測をせず manifest の `shards[0]` をそのまま引く（sbv2 のように
 *   先頭 shard だけ `shared/` に居る形があり、コンポーネントのディレクトリを走査すると
 *   グラフ shard を取り落とす）
 * - **系列出力** — `outputs/series/<名前>/` 以下。manifest が無いので、ファイル名から代表 path を
 *   起こして {@link resolveShards}（Python 側 `karume.shards.resolve_shards` の鏡像）に解かせる
 */

import { resolveShards } from "../../packages/runtime/tests/helpers/shard-files.ts";
import { type IrGraph, parseIrGraph } from "../../packages/runtime/src/format/ir.ts";

/** census を掛ける 1 コンポーネント。 */
export type ComponentTarget = {
  readonly component: string;
  /** 配布形の quant 表が選んだ格納 dtype キー（系列出力ではファイル名の infix）。 */
  readonly componentDtype: string;
  /** グラフを載せている先頭 shard。 */
  readonly graphShard: URL;
};

/** 1 資産ぶんの解決結果。 */
export type AssetTargets = {
  readonly family: string;
  readonly model: string;
  readonly quant: string;
  readonly components: readonly ComponentTarget[];
};

/**
 * safetensors のヘッダ JSON だけを読んで IR を取り出す。
 *
 * MUST: `karume_ir` が無い shard は fail loudly。重み shard（metadata 無し）を先頭と取り違えた
 * ときに、空グラフの census が「ノード 0 本」として静かに出力されるのを防ぐ。
 */
export const readIrGraph = async (source: URL): Promise<IrGraph> => {
  const file = await Deno.open(source, { read: true });
  try {
    const lengthBytes = new Uint8Array(8);
    await readExact(file, lengthBytes, source);
    const length = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));
    const headerBytes = new Uint8Array(length);
    await readExact(file, headerBytes, source);
    const header: unknown = JSON.parse(new TextDecoder().decode(headerBytes));
    const metadata = (header as { readonly __metadata__?: Record<string, string> }).__metadata__;
    const ir = metadata?.karume_ir;
    if (ir === undefined) {
      throw new Error(
        `${source.pathname}: safetensors ヘッダに __metadata__.karume_ir が無い` +
          "（グラフを載せるのは先頭 shard だけ — 重み shard を指していないか）",
      );
    }
    return parseIrGraph(ir);
  } finally {
    file.close();
  }
};

/** `into.length` バイトちょうど読む（短い読み返しで黙ってゼロ埋めのヘッダを作らない）。 */
const readExact = async (handle: Deno.FsFile, into: Uint8Array, where: URL): Promise<void> => {
  for (let read = 0; read < into.length;) {
    const chunk = await handle.read(into.subarray(read));
    if (chunk === null) throw new Error(`${where.pathname}: ${into.length} バイトを読み切れない`);
    read += chunk;
  }
};

/** manifest のうち census が引く欄だけ（綴りの正本は配布形なので、ここに焼かず読む）。 */
type Manifest = {
  readonly defaultModel: string;
  readonly models: Readonly<Record<string, ManifestModel>>;
};
type ManifestModel = {
  readonly pipeline: string;
  readonly defaultQuant: string;
  readonly quants: Readonly<Record<string, { readonly weights: Readonly<Record<string, string>> }>>;
  readonly weights: Readonly<
    Record<string, Readonly<Record<string, { readonly shards: readonly ManifestShard[] }>>>
  >;
};
type ManifestShard = { readonly path: string; readonly repo?: string };

/** ディレクトリ列挙（不在は空・それ以外の I/O 異常は伝播させる）。 */
const listDir = (dir: URL): readonly Deno.DirEntry[] => {
  try {
    return [...Deno.readDirSync(dir)];
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return [];
    throw cause;
  }
};

const isFile = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/**
 * 資産ディレクトリを解決する。
 *
 * @param model 配布形の model 名（省略時は `defaultModel`）。系列出力では受けない。
 * @param quant 配布形の quant 名（省略時は `defaultQuant`）。系列出力では格納 dtype の
 *   グループ名（`model.i8-*.safetensors` の `i8`）。
 */
export const resolveAsset = async (
  root: URL,
  model: string | undefined,
  quant: string | undefined,
  family: string | undefined,
): Promise<AssetTargets> => {
  const manifest = new URL("karume.json", root);
  return isFile(manifest)
    ? await resolveDistribution(root, manifest, model, quant, family)
    : resolveSeries(root, model, quant, family);
};

/** 配布形（manifest 正本）。 */
const resolveDistribution = async (
  root: URL,
  manifestUrl: URL,
  model: string | undefined,
  quant: string | undefined,
  family: string | undefined,
): Promise<AssetTargets> => {
  const manifest: Manifest = JSON.parse(await Deno.readTextFile(manifestUrl));
  const modelName = model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${manifestUrl.pathname}: model '${modelName}' が無い` +
        `（既知: ${Object.keys(manifest.models).join(" / ")}）`,
    );
  }
  const entry = manifest.models[modelName];
  const quantName = quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${manifestUrl.pathname}: model '${modelName}' に quant '${quantName}' が無い` +
        `（既知: ${Object.keys(entry.quants).join(" / ")}）`,
    );
  }
  const selection = entry.quants[quantName].weights;
  const components = Object.keys(entry.weights).map((component): ComponentTarget => {
    if (!Object.hasOwn(selection, component)) {
      throw new Error(
        `quant '${quantName}' が component '${component}' の格納 dtype を選んでいない`,
      );
    }
    const dtype = selection[component];
    const variants = entry.weights[component];
    if (!Object.hasOwn(variants, dtype)) {
      throw new Error(
        `component '${component}' に格納 dtype '${dtype}' が無い` +
          `（既知: ${Object.keys(variants).join(" / ")}）`,
      );
    }
    const [head] = variants[dtype].shards;
    // 越境参照（ADR 0038 §7）はローカルミラーの綴りを持たない。黙って root 直下として解くと
    // 存在しない path を読みに行くだけなので、ここで理由ごと落とす。
    if (head.repo !== undefined) {
      throw new Error(
        `component '${component}' の先頭 shard が越境参照（repo '${head.repo}'）— ` +
          "ローカルに落とした配布形を --source に渡す",
      );
    }
    return { component, componentDtype: dtype, graphShard: new URL(head.path, root) };
  });
  return {
    family: family ?? familyOfPipeline(entry.pipeline),
    model: modelName,
    quant: quantName,
    components,
  };
};

/** `anima/1` → `anima`。 */
const familyOfPipeline = (pipeline: string): string => {
  const [name] = pipeline.split("/");
  if (name === "") throw new Error(`pipeline '${pipeline}' から家族名を取れない`);
  return name;
};

/**
 * 系列出力（`outputs/series/<名前>/`）。コンポーネントの単位はディレクトリで、重みファイルが
 * 直下にある形は単一コンポーネント `model` として扱う。
 */
const resolveSeries = (
  root: URL,
  model: string | undefined,
  quant: string | undefined,
  family: string | undefined,
): AssetTargets => {
  if (model !== undefined) {
    throw new Error("系列出力には model の選択が無い（--model は配布形でだけ使える）");
  }
  const directories = componentDirectories(root);
  const components = directories.map(([component, dir]) => {
    const groups = storageGroups(dir);
    if (groups.length === 0) {
      throw new Error(`${dir.pathname}: model*.safetensors が無い`);
    }
    const chosen = quant === undefined
      ? soleGroup(groups, dir)
      : groups.find(([name]) => name === quant);
    if (chosen === undefined) {
      throw new Error(
        `${dir.pathname}: 格納 dtype '${quant}' が無い（既知: ${
          groups.map(([name]) => name).join(" / ")
        }）`,
      );
    }
    const [dtype, representative] = chosen;
    return {
      component,
      componentDtype: dtype,
      graphShard: resolveShards(representative)[0],
    };
  });
  const groupNames = new Set(components.map((target) => target.componentDtype));
  return {
    family: family ?? familyOfSeriesDirectory(root),
    model: directoryName(root),
    quant: quant ?? [...groupNames].join("+"),
    components,
  };
};

/**
 * 格納 dtype グループが 1 つだけならそれを返す。
 * MUST: 複数あるまま黙って 1 つ選ばない — census 加重は格納で変わる（i4 の group 数など）ので、
 * どちらを測ったのか分からない表が出る。
 */
const soleGroup = (
  groups: readonly (readonly [string, URL])[],
  dir: URL,
): readonly [string, URL] => {
  if (groups.length > 1) {
    throw new Error(
      `${dir.pathname}: 格納 dtype が複数ある（${
        groups.map(([name]) => name).join(" / ")
      }）— --quant で 1 つ選ぶ`,
    );
  }
  return groups[0];
};

/** `<名前>/` の最終要素。 */
const directoryName = (dir: URL): string =>
  decodeURIComponent(dir.pathname).replace(/\/$/, "").split("/").pop() ?? "";

/**
 * 系列ディレクトリ名 → 家族名。`--family` の既定値でしかないので、知らない綴りは
 * fail loudly（家族名を誤ると既定シナリオの束縛が別家族のものになる）。
 */
const SERIES_FAMILIES: readonly string[] = [
  "anima",
  "birefnet",
  "depth-anything",
  "gemma4",
  "irodori",
  "sbv2",
  "siglip2",
  "vowel-detector",
];

const familyOfSeriesDirectory = (root: URL): string => {
  const name = directoryName(root);
  const match = SERIES_FAMILIES.find((family) => name === family || name.startsWith(`${family}-`));
  if (match === undefined) {
    throw new Error(
      `系列ディレクトリ '${name}' から家族名を推せない（既知: ${SERIES_FAMILIES.join(" / ")}）` +
        " — --family で明示する",
    );
  }
  return match;
};

/** 重みファイルを持つディレクトリの列（直下にある形は単一コンポーネント `model`）。 */
const componentDirectories = (root: URL): readonly (readonly [string, URL])[] => {
  const entries = listDir(root);
  if (entries.some((entry) => entry.isFile && MODEL_FILE.test(entry.name))) {
    return [["model", root]];
  }
  const found = entries
    .filter((entry) => entry.isDirectory)
    .map((entry): readonly [string, URL] => [entry.name, new URL(`${entry.name}/`, root)])
    .filter(([, dir]) => listDir(dir).some((entry) => entry.isFile && MODEL_FILE.test(entry.name)));
  if (found.length === 0) {
    throw new Error(`${root.pathname}: model*.safetensors を持つディレクトリが無い`);
  }
  return found.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
};

/**
 * `model.safetensors` / `model.<dtype>.safetensors` と、その shard 連番形。
 * NOTE: 連番の綴り（`-NNNNN-of-NNNNN`）を解くのは {@link resolveShards} の仕事で、ここは
 * 「代表 path がどれか」だけを拾う。
 */
const MODEL_FILE = /^model(?:\.(?<dtype>[^.\-]+))?(?:-\d{5}-of-\d{5})?\.safetensors$/;

/** ディレクトリ内の格納 dtype グループ（グループ名 → 代表 path）。 */
const storageGroups = (dir: URL): readonly (readonly [string, URL])[] => {
  const groups = new Map<string, URL>();
  for (const entry of listDir(dir)) {
    if (!entry.isFile) continue;
    const match = MODEL_FILE.exec(entry.name);
    if (match === null) continue;
    const dtype = match.groups?.dtype ?? "";
    const name = dtype === "" ? "native" : dtype;
    groups.set(
      name,
      new URL(dtype === "" ? "model.safetensors" : `model.${dtype}.safetensors`, dir),
    );
  }
  return [...groups].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
};

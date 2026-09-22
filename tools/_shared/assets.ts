/**
 * tools 共有: **資産ディレクトリ → 部品ごとのグラフの在り処**の解決と、その IR の読み出し。
 *
 * opbench（静的 census）と fusion-hints（融合候補列挙）が同じ資産を同じ規則で見つけるための
 * 1 本。読むのは**グラフの宣言が載っている先頭だけ**（配布形なら容器の part 0 = ヘッダ +
 * 2 文書・系列出力なら safetensors ヘッダの `__metadata__.karume_ir`）。実体は合計 GB 級なので
 * 重みの block は 1 バイトも読まない。
 *
 * 資産は 2 形ある:
 *
 * - **配布形** — `karume.json`（manifest `karume/5`）を持つディレクトリ。部品 1 つが `krm`
 *   コンテナ 1 本で、グラフ記述と束縛表は part 0 に載る（ADR 0109 決定 3）。ファイル名の推測を
 *   せず manifest の `container.parts[0]` をそのまま引く
 * - **系列出力** — `outputs/series/<名前>/` 以下。manifest が無いので、ファイル名から代表 path を
 *   起こして {@link resolveShards}（Python 側 `karume.shards.resolve_shards` の鏡像）に解かせる
 *   （こちらは旧 IR の safetensors 方言のまま）
 *
 * 公開する解決口は {@link resolveAsset} 1 本で、**格納 dtype は quant 表に従う**（配布形は
 * manifest の quant が選んだ dtype・系列出力は `--quant` かディレクトリに 1 つだけある
 * グループ）。census 加重は格納で変わる（i4 の group 数など）ので、どの dtype を測ったのかが
 * 表の意味そのもの。融合候補の列挙は格納に依らない（候補はノード列だけで決まる）が、道具に
 * よって別の dtype の shard を開くと 2 つの表が同じ資産の別の面を指すことになるので、
 * 解決口を分けない。
 */

import { resolveShards } from "../../packages/runtime/tests/helpers/shard-files.ts";
import { type IrGraph, parseIrGraph } from "../../packages/runtime/src/format/ir.ts";
import type { FusionLimits } from "../../packages/runtime/src/runtime/fusion.ts";
import { bindGraphs, mergedGraph } from "../../packages/runtime/src/format/container/bind.ts";
import {
  parseGraphDescriptor,
  parseModelDescriptor,
} from "../../packages/runtime/src/format/container/descriptor.ts";
import { readHeader } from "../../packages/runtime/src/format/container/header.ts";
import { HEADER_BYTES } from "../../packages/runtime/src/format/container/limits.ts";

/**
 * 融合計画に渡す device 能力。**WebGPU core の既定値**（128MiB / 65535）を固定で使う。
 *
 * MUST: 実機の granted limit を読まない。census も候補表も機に依らない静的な表であるべきで、
 * 行ブロック枚数の分かれ目（`rowBlockAttention`）が測った機ごとに動くと、別の機で採った表と
 * 融合の有無が黙って食い違う。実機の値で見たいときは実行時の `lastRunFusions` が正本。
 */
export const CORE_LIMITS: FusionLimits = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
};

/** 相対 path を cwd 基準のディレクトリ URL にする（末尾 `/` を必ず付ける）。 */
export const directoryUrl = (path: string): URL =>
  new URL(path.endsWith("/") ? path : `${path}/`, `file://${Deno.cwd()}/`);

/**
 * CLI の path 文字列を、**外部プロセスへ渡せる素の絶対 path** にする（末尾 `/` は落とす）。
 *
 * MUST: 子プロセスへ渡す path に {@link directoryUrl} の `pathname` を使わない。URL の
 * `pathname` は percent encode 済みなので、空白や非 ASCII を含む path は `%20` を含む
 * **リテラルなディレクトリ名**として子へ渡り、Deno 側の書き出し先と食い違う。`#` / `?` は
 * さらに悪く、`new URL` の時点で fragment / query として `pathname` から落ちるため、
 * 警告なく別のディレクトリを指す。
 */
export const externalPath = (path: string): string => {
  const absolute = path.startsWith("/") ? path : `${Deno.cwd()}/${path}`;
  return absolute.length > 1 ? absolute.replace(/\/+$/, "") : absolute;
};

/**
 * グラフ宣言の在り処（2 形）。どちらで読むかは**解決の時点で決まっている**ので、読み手が
 * バイト列を覗いて推測しない（推測すると、取り違えた資産が「別の形だった」という理由で
 * 静かに読み飛ばされる）。
 */
export type GraphSource =
  /** 配布形: `krm` の part 0（ヘッダ + グラフ記述 + モデル記述）。グラフ名 = 部品名（決定 8）。 */
  | { readonly kind: "container"; readonly url: URL; readonly graph: string }
  /** 系列出力: 旧 IR（safetensors の `__metadata__.karume_ir`）を載せた先頭 shard。 */
  | { readonly kind: "shard"; readonly url: URL };

/** 先頭から `length` バイトだけ読む（重みの block へは進まない）。 */
const readPrefix = async (source: URL, length: number): Promise<Uint8Array<ArrayBuffer>> => {
  const file = await Deno.open(source, { read: true });
  try {
    const bytes = new Uint8Array(new ArrayBuffer(length));
    await readExact(file, bytes, source);
    return bytes;
  } finally {
    file.close();
  }
};

/**
 * 容器の part 0 から 1 グラフの IR を読む（グラフ記述 × 束縛表の合流まで — 格納 codec は
 * 束縛表にしか無いので、合流しないと census の `storage` 欄が埋まらない）。
 *
 * MUST: 名指しのグラフが無ければ在るグラフを並べて落とす。黙って 1 本目を採ると、部品名と
 * グラフ名がずれた容器で「別の部品を数えた表」が静かに出る。
 */
const readContainerGraph = async (url: URL, graphName: string): Promise<IrGraph> => {
  const header = readHeader(await readPrefix(url, HEADER_BYTES));
  const documents = await readPrefix(
    url,
    HEADER_BYTES + header.graphDescriptorLength + header.modelDescriptorLength,
  );
  const graph = parseGraphDescriptor(
    documents.slice(HEADER_BYTES, HEADER_BYTES + header.graphDescriptorLength),
  );
  const model = header.kind === "model"
    ? parseModelDescriptor(documents.slice(HEADER_BYTES + header.graphDescriptorLength))
    : undefined;
  const bound = bindGraphs(graph, model)[graphName];
  if (bound === undefined) {
    throw new Error(
      `${url.pathname}: 容器にグラフ '${graphName}' が無い` +
        `（在るのは ${Object.keys(graph.graphs).join(" / ")}）`,
    );
  }
  return mergedGraph(bound, graphName);
};

/**
 * safetensors のヘッダ JSON だけを読んで旧 IR を取り出す（系列出力）。
 *
 * MUST: `karume_ir` が無い shard は fail loudly。重み shard（metadata 無し）を先頭と取り違えた
 * ときに、空グラフの census が「ノード 0 本」として静かに出力されるのを防ぐ。
 */
const readShardGraph = async (source: URL): Promise<IrGraph> => {
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

/** グラフ宣言を 1 本読む（形は解決済み — {@link GraphSource}）。 */
export const readIrGraph = async (source: GraphSource): Promise<IrGraph> =>
  source.kind === "container"
    ? await readContainerGraph(source.url, source.graph)
    : await readShardGraph(source.url);

/**
 * `into.length` バイトちょうど読む。
 *
 * MUST: 短い読み返しを検査する。`Deno.FsFile.read` は要求より短い長さも `null`（EOF）も返せる
 * ので、返り値を捨てると残りが 0 のままのヘッダを黙って組み立ててしまう。長さ 8 バイトが
 * ゼロ埋めなら別の長さで JSON を切り出し、「IR を載せない資産」という誤った理由でグラフが
 * 1 本静かに落ちる。
 */
const readExact = async (handle: Deno.FsFile, into: Uint8Array, where: URL): Promise<void> => {
  for (let read = 0; read < into.length;) {
    const chunk = await handle.read(into.subarray(read));
    if (chunk === null) throw new Error(`${where.pathname}: ${into.length} バイトを読み切れない`);
    read += chunk;
  }
};

/**
 * 配布形の quant が宣言する**実行変種**（manifest の `quants[<名前>].session`）。
 *
 * MUST: キーも値も manifest 所有の語彙のまま持ち、runtime の `SessionOptions` へ翻訳しない
 * （綴りの正本は hub の allowlist — ADR 0038 §3）。census が写すのは「配布形が何を宣言して
 * いるか」であって runtime のノブではないので、翻訳を挟むと表の綴りが正本と黙って割れる。
 */
export type SessionDeclaration = Readonly<Record<string, string>>;

/**
 * manifest のうち資産解決が引く欄だけ（綴りの正本は配布形なので、ここに焼かず読む）。検査は
 * hub の `parseManifest` が持つので、ここは「読む欄の形」だけを名乗る型である。
 */
type Manifest = {
  /** `karume/<major>`。この道具が読むのは {@link MANIFEST_FORMAT} だけ。 */
  readonly format: string;
  readonly defaultModel: string;
  readonly models: Readonly<Record<string, ManifestModel>>;
};
type ManifestModel = {
  readonly pipeline: string;
  readonly defaultQuant: string;
  readonly quants: Readonly<Record<string, ManifestQuant>>;
  readonly weights: Readonly<
    Record<string, Readonly<Record<string, { readonly container: ManifestContainer }>>>
  >;
};
type ManifestQuant = {
  readonly weights: Readonly<Record<string, string>>;
  /** 省略可（hub 側も未宣言を「ノブを 1 つも指定しない」として読む）。 */
  readonly session?: SessionDeclaration;
};
/** 部品 1 つ = コンテナ 1 本。先頭が part 0（ヘッダ + 2 文書）— ADR 0109 決定 3。 */
type ManifestContainer = {
  readonly parts: readonly ManifestPart[];
};
/** `size` は part のファイル長（ADR 0038 §2 の 3 点セットの 1 つ — 必ず在る）。 */
type ManifestPart = {
  readonly path: string;
  readonly size: number;
  readonly repo?: string;
};

/** この版が読む配布 manifest の major（旧版は読まない — ADR 0109 決定 1）。 */
export const MANIFEST_FORMAT = "karume/5";

/**
 * 配布形ミラーが名乗る `format`（`karume.json` が無ければ `undefined`）。
 *
 * 実資産テストの SKIP 判定に使う（ADR 0005 の「無い環境は明示 SKIP」）。**真偽ではなく値を
 * 返す**のは、SKIP の warn に「何が在ったのか」を書かせるため — 「無い」と「旧 major のまま」は
 * 読み手のやることが違う（前者は生成、後者は移行）。
 *
 * MUST: NotFound 以外は伝播させる。全 I/O エラーを「未生成」に丸めると、資産ルートのマウント
 * 異常が SKIP に化けて、実行されていない検証が静かに緑になる。
 */
export const distributionFormat = (root: URL): string | undefined => {
  let text: string;
  try {
    text = Deno.readTextFileSync(new URL("karume.json", root));
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
  const format = (JSON.parse(text) as { readonly format?: unknown }).format;
  return typeof format === "string" ? format : undefined;
};

/** 配布形の model 1 件（`--model` 省略時は `defaultModel`）。 */
const manifestModel = (
  manifest: Manifest,
  manifestUrl: URL,
  model: string | undefined,
): readonly [string, ManifestModel] => {
  const name = model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, name)) {
    throw new Error(
      `${manifestUrl.pathname}: model '${name}' が無い` +
        `（既知: ${Object.keys(manifest.models).join(" / ")}）`,
    );
  }
  return [name, manifest.models[name]];
};

/**
 * コンテナの part 列から part 0（グラフ記述の置き場）のローカル URL を作る。
 *
 * MUST: 越境参照（ADR 0038 §7）はローカルミラーの綴りを持たないので、ここで理由ごと落とす。
 * 黙って root 直下として解くと存在しない path を読みに行くだけだし、飛ばすと「候補ゼロの
 * 部品」として表に出てしまう。
 */
const localPart0 = (container: ManifestContainer | undefined, root: URL, where: string): URL => {
  // MUST: 空の part 列を診断無しの TypeError にしない（manifest が壊れている、という理由が
  // 読める形で落とす — part 0 はグラフの置き場なので 0 本はあり得ない）。
  const head = container?.parts?.[0];
  if (head === undefined) throw new Error(`${where}: manifest の container.parts が空`);
  if (head.repo !== undefined) {
    throw new Error(
      `${where}: part 0 が越境参照（repo '${head.repo}'）— ` +
        "ローカルに落とした配布形を --source に渡す",
    );
  }
  return new URL(head.path, root);
};

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

/** `<名前>/` の最終要素。 */
const directoryName = (dir: URL): string =>
  decodeURIComponent(dir.pathname).replace(/\/$/, "").split("/").pop() ?? "";

/**
 * `model.safetensors` / `model.<dtype>.safetensors` と、その shard 連番形。
 * NOTE: 連番の綴り（`-NNNNN-of-NNNNN`）を解くのは {@link resolveShards} の仕事で、ここは
 * 「代表 path がどれか」だけを拾う。
 */
const MODEL_FILE = /^model(?:\.(?<dtype>[^.\-]+))?(?:-\d{5}-of-\d{5})?\.safetensors$/;

/** 系列出力のコンポーネント 1 件（重みファイルを持つディレクトリ）。 */
type ComponentDirectory = {
  /** 資産根からの相対ディレクトリ名。重みが根直下にある形は空。 */
  readonly relative: string;
  readonly dir: URL;
};

/**
 * 重みファイルを持つディレクトリの列。1 件も無ければ空を返す（診断の綴りは呼び手が持つ）。
 *
 * NOTE: 探すのは根の直下 1 階層まで。配布形と同じく `<コンポーネント>/model*.safetensors` が
 * 系列出力の綴りで、それより深い置き方をする資産は無い。
 */
const componentDirectories = (root: URL): readonly ComponentDirectory[] => {
  const entries = listDir(root);
  if (entries.some((entry) => entry.isFile && MODEL_FILE.test(entry.name))) {
    return [{ relative: "", dir: root }];
  }
  return entries
    .filter((entry) => entry.isDirectory)
    .map((entry): ComponentDirectory => ({
      relative: entry.name,
      dir: new URL(`${entry.name}/`, root),
    }))
    .filter(({ dir }) => listDir(dir).some((entry) => entry.isFile && MODEL_FILE.test(entry.name)))
    .sort((a, b) => (a.relative < b.relative ? -1 : a.relative > b.relative ? 1 : 0));
};

/** ディレクトリ内の格納 dtype グループ（グループ名 → 代表 path・グループ名順）。 */
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

/** census を掛ける 1 部品。 */
export type ComponentTarget = {
  readonly component: string;
  /** 配布形の quant 表が選んだ格納 dtype キー（系列出力ではファイル名の infix）。 */
  readonly componentDtype: string;
  /** グラフ宣言の在り処（配布形は容器の part 0・系列出力は先頭 shard）。 */
  readonly graph: GraphSource;
};

/** 1 資産ぶんの解決結果。 */
export type AssetTargets = {
  readonly family: string;
  readonly model: string;
  readonly quant: string;
  /**
   * 選んだ quant が宣言した実行変種。**系列出力は manifest を持たないので `undefined`**
   * （= 実行変種は宣言されておらず、呼び手が与える）。
   *
   * census 行の形は IR だけで決まる一方、その形を**どのカーネルで走らせるか**は manifest 側の
   * 宣言（`linearCompute` など）なので、加重表だけでは本番の実行変種が決まらない。表の読み手が
   * 「この加重はどの席の話か」を辿れるよう写す。
   */
  readonly session: SessionDeclaration | undefined;
  readonly components: readonly ComponentTarget[];
};

/**
 * 資産ディレクトリを解決する（census 用 — 格納 dtype は quant 表に従う）。
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
  // MUST: major を見る。旧版の manifest は欄の形が違うので、見ないと `container` が無い形を
  // 「part 列が空」という遠い理由で落とすことになる（ADR 0109 決定 1）。
  if (manifest.format !== MANIFEST_FORMAT) {
    throw new Error(
      `${manifestUrl.pathname}: format '${manifest.format}' はこの版が読めない` +
        `（読めるのは ${MANIFEST_FORMAT} — 配布形を移行する）`,
    );
  }
  const [modelName, entry] = manifestModel(manifest, manifestUrl, model);
  const quantName = quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${manifestUrl.pathname}: model '${modelName}' に quant '${quantName}' が無い` +
        `（既知: ${Object.keys(entry.quants).join(" / ")}）`,
    );
  }
  const quantEntry = entry.quants[quantName];
  const selection = quantEntry.weights;
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
    return {
      component,
      componentDtype: dtype,
      // グラフ名 = 部品名（書き手の規約 — ADR 0109 決定 8）。
      graph: {
        kind: "container",
        url: localPart0(variants[dtype].container, root, `component '${component}'`),
        graph: component,
      },
    };
  });
  return {
    family: family ?? familyOfPipeline(entry.pipeline),
    model: modelName,
    quant: quantName,
    // 欄ごと無い quant は「ノブを 1 つも指定しない」= 空の宣言（hub の読みと同じ）。
    // 「宣言が無い」を表す `undefined` は manifest を持たない系列出力だけに使う。
    session: quantEntry.session ?? {},
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
  if (directories.length === 0) {
    throw new Error(`${root.pathname}: model*.safetensors を持つディレクトリが無い`);
  }
  const components = directories.map(({ relative, dir }) => {
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
      component: relative === "" ? "model" : relative,
      componentDtype: dtype,
      graph: { kind: "shard", url: resolveShards(representative)[0] },
    } satisfies ComponentTarget;
  });
  const groupNames = new Set(components.map((target) => target.componentDtype));
  return {
    family: family ?? familyOfSeriesDirectory(root),
    model: directoryName(root),
    quant: quant ?? [...groupNames].join("+"),
    session: undefined,
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

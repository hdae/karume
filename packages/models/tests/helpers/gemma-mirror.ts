/**
 * 配布形ミラー（`karume/5`）から gemma の全量面の入力（`Gemma4Assets`）と PLE の読み口を組む
 * — **実資産 e2e が共有する 1 本**。
 *
 * ## なぜ系列出力ではなくミラーなのか
 *
 * PLE は ADR 0109 決定 4 で `model` 容器の資産（`ple_index` + `ple-values` / `ple-scales` の
 * block 列）へ移り、recipe の系列出力（`outputs/series/**` の素の safetensors + `ple.json`）は
 * **段 3 まで旧形のまま**である（ADR 0109 決定 8）。したがって `fromAssets` / `createGemma4Ple`
 * に渡せる実資産は移行済みミラーだけで、系列出力からは組めない。重みも PLE も同じ焼き直し
 * なので、golden の断定はそのまま保てる。
 *
 * NOTE: hub / runtime の**テストの都合**は import しない（`helpers/memory-cache.ts` の規律）。
 * ここが触るのは公開面（`@karume/hub` / `@karume/hub/deno` / `@karume/runtime` と models の
 * 公開サブパス）だけである。
 */

import { loadManifest, openContainerSource, resolveSelection } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import {
  type AssetReader,
  openContainer,
  type OpenedContainer,
  prepareContainer,
  type PreparedModel,
} from "@karume/runtime";
import { type Gemma4Assets, parseGemma4PipelineConfig } from "../../gemma.ts";
import {
  gemma4PleAssetSource,
  type Gemma4PleIndex,
  readGemma4PleIndex,
} from "../../src/gemma/ple-index.ts";

/** manifest の選択軸（省略時はミラーの既定）。 */
export type MirrorChoice = {
  readonly model?: string;
  readonly quant?: string;
};

const MODEL = "model";

const selectionOf = async (mirror: URL, choice: MirrorChoice) => {
  const loaded = await loadManifest(denoDirectory(mirror.pathname));
  const selection = resolveSelection(loaded.manifest, {
    ...(choice.model === undefined ? {} : { model: choice.model }),
    ...(choice.quant === undefined ? {} : { quant: choice.quant }),
    weights: [MODEL],
  });
  const container = selection.containers[MODEL];
  if (container === undefined) {
    throw new Error(`test: ${mirror.pathname} の manifest に部品 '${MODEL}' の容器が無い`);
  }
  return { loaded, selection, container };
};

/**
 * ミラーの `model` 容器を**区間読みで**開く（part 0 と、以後に触った block だけを読む）。
 *
 * 全 part を RAM に載せる {@link gemma4MirrorAssets} と違い、PLE の読み口はこちらから取る。
 */
const openModelContainer = async (
  mirror: URL,
  choice: MirrorChoice,
): Promise<OpenedContainer> => {
  const { loaded, container } = await selectionOf(mirror, choice);
  return await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );
};

/**
 * ミラーの部品 1 本を**区間読みで**開き、宣言と Session の入口を返す。
 *
 * 旧 shard 面（`prepareModel` + `streamShards`）の置き換え — 部品は容器 1 本になり、block は
 * Session を組むその瞬間に part 順で読まれる（ADR 0109 決定 7）。
 */
export const openMirrorComponent = async (
  mirror: URL,
  key: string,
  choice: MirrorChoice = {},
): Promise<{
  readonly graph: PreparedModel["graph"];
  readonly createSession: PreparedModel["createContainerSession"];
  readonly opened: OpenedContainer;
}> => {
  const loaded = await loadManifest(denoDirectory(mirror.pathname));
  const selection = resolveSelection(loaded.manifest, {
    ...(choice.model === undefined ? {} : { model: choice.model }),
    ...(choice.quant === undefined ? {} : { quant: choice.quant }),
    weights: [key],
  });
  const container = selection.containers[key];
  if (container === undefined) {
    throw new Error(`test: ${mirror.pathname} の manifest に部品 '${key}' の容器が無い`);
  }
  const opened = await openContainer(
    { kind: "source", source: openContainerSource(loaded, container) },
    container.descriptor,
  );
  // グラフ名 = 役割名（書き手の規約 — ADR 0109 決定 8）。
  const prepared = prepareContainer(opened, key);
  return {
    graph: prepared.graph,
    createSession: (gpu, options) => prepared.createContainerSession(gpu, options),
    opened,
  };
};

/** PLE の索引と block の読み口（`createGemma4Ple` / `createGemma4PleResident` の入力）。 */
export type Gemma4PleHandle = {
  readonly index: Gemma4PleIndex;
  readonly openBlock: (asset: string) => AssetReader;
};

/**
 * ミラーの `model` 容器から PLE の索引を読み、block の読み口を配る。
 *
 * 索引と容器の資産の突合（`readGemma4PleIndex`）も通るので、片方だけ焼き直したミラーは
 * ここで落ちる。
 */
export const openGemma4Ple = async (
  mirror: URL,
  choice: MirrorChoice = {},
): Promise<Gemma4PleHandle> => {
  const opened = await openModelContainer(mirror, choice);
  return {
    index: await readGemma4PleIndex(
      `test: ${mirror.pathname}`,
      gemma4PleAssetSource(opened),
    ),
    openBlock: (asset) => opened.asset(asset),
  };
};

/**
 * ミラーから `fromAssets` の入力を組む（`model` 容器の part 列 + tokenizer）。
 *
 * MUST: 長さ 0 の part も並べる — 添字が容器の中の part id なので、飛ばすと以降が 1 つずつ
 * 繰り上がって別の part として読まれる。
 */
export const gemma4MirrorAssets = async (
  mirror: URL,
  choice: MirrorChoice = {},
): Promise<Gemma4Assets> => {
  const { loaded, selection, container } = await selectionOf(mirror, choice);
  const modelName = choice.model ?? loaded.manifest.defaultModel;
  const tokenizer = selection.assets["tokenizer"];
  if (tokenizer === undefined) {
    throw new Error(`test: ${mirror.pathname} の manifest に資産 'tokenizer' が無い`);
  }
  return {
    config: parseGemma4PipelineConfig(loaded.manifest.models[modelName].pipelineConfig),
    model: await Promise.all(
      container.parts.map((part) => Deno.readFile(new URL(part.path, mirror))),
    ),
    tokenizer: await Deno.readFile(new URL(tokenizer.path, mirror)),
  };
};

/** この版が読む配布 manifest の major（旧版は読まない — ADR 0109 決定 1）。 */
export const MIRROR_FORMAT = "karume/5";

/**
 * ミラーが**この版が読める配布形**（`karume/5`）か。
 *
 * MUST: 「ファイルが在る」ではなく format で判定する。移行前のミラーが残っている機では
 * `parseManifest` が落ちるので、存在だけを条件にすると門が赤くなるだけで何も言えない
 * （ADR 0109 決定 9 の共存期間 — `tools/_shared/assets.ts` の `distributionFormat` と同じ規律）。
 */
export const mirrorAvailable = (mirror: URL): boolean => {
  let text: string;
  try {
    text = Deno.readTextFileSync(new URL("karume.json", mirror));
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
  return (JSON.parse(text) as { readonly format?: unknown }).format === MIRROR_FORMAT;
};

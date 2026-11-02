/**
 * 配布形ミラー（`karume/5`）から gemma の全量面の入力（`Assets`）と PLE の読み口を組む
 * — **実資産 e2e が共有する 1 本**。
 *
 * ## なぜ系列出力ではなくミラーなのか
 *
 * 系列出力も `krm` を書くようになったので、容器を開いて 1 部品を走らせる
 * だけなら `helpers/ple-series.ts` の経路で足りる。ミラーが要るのは**manifest が持つもの**を
 * 見る席だけである — `fromAssets` の全量面が受ける part 列のキー（`<部品>[i]`）、model /
 * quant の選択軸、2 文書の期待 hash（`container.descriptor`）。系列出力には manifest が
 * 無いのでこれらは組めない。重みも PLE も同じ焼き直しなので、golden の断定はどちらから
 * 読んでも同じである。
 *
 * NOTE: hub / runtime の**テストの都合**は import しない（`helpers/memory-cache.ts` の規律）。
 * ここが触るのは公開面（`@karume/hub` / `@karume/hub/node` / `@karume/runtime` と models の
 * 公開サブパス）だけである。
 */

import { readFile } from "@karume/test-support/fs";
import type { AssetReader, OpenedContainer } from "@karume/runtime";
import { type Assets, parsePipelineConfig } from "../../gemma.ts";
import {
  gemma4PleAssetSource,
  type Gemma4PleIndex,
  readGemma4PleIndex,
} from "../../src/gemma/ple-index.ts";
import { type MirrorChoice, openSelectedContainer, selectMirrorComponent } from "./mirror.ts";

// The mirror resolver, the presence check and the component opener are shared with every family
// (`helpers/mirror.ts`); they stay exported here under their existing names.
export {
  MIRROR_FORMAT,
  mirrorPresent as mirrorAvailable,
  type MirrorChoice,
  openMirrorComponent,
} from "./mirror.ts";

const MODEL = "model";

const selectionOf = (mirror: URL, choice: MirrorChoice) =>
  selectMirrorComponent(mirror, MODEL, choice);

/**
 * ミラーの `model` 容器を**区間読みで**開く（part 0 と、以後に触った block だけを読む）。
 *
 * 全 part を RAM に載せる {@link gemma4MirrorAssets} と違い、PLE の読み口はこちらから取る。
 */
const openModelContainer = async (mirror: URL, choice: MirrorChoice): Promise<OpenedContainer> =>
  await openSelectedContainer(await selectionOf(mirror, choice));

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
    index: await readGemma4PleIndex(`test: ${mirror.pathname}`, gemma4PleAssetSource(opened)),
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
): Promise<Assets> => {
  const { loaded, selection, container } = await selectionOf(mirror, choice);
  const modelName = choice.model ?? loaded.manifest.defaultModel;
  const tokenizer = selection.assets["tokenizer"];
  if (tokenizer === undefined) {
    throw new Error(`test: ${mirror.pathname} の manifest に資産 'tokenizer' が無い`);
  }
  return {
    config: parsePipelineConfig(loaded.manifest.models[modelName].pipelineConfig),
    model: await Promise.all(container.parts.map((part) => readFile(new URL(part.path, mirror)))),
    tokenizer: await readFile(new URL(tokenizer.path, mirror)),
  };
};

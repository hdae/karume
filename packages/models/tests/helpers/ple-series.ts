/**
 * recipe の**系列出力**（`outputs/series/**` の `model.krm`）が持つ PLE を、ミラーと同じ面
 * （{@link Gemma4PleHandle}）で開く。
 *
 * ## なぜ 1 本の合成関数で済むのか
 *
 * PLE は ADR 0109 決定 4 で `model` 容器の資産（索引 `ple_index` + 役割 `ple-values` /
 * `ple-scales` の block 列）へ移り、recipe も段 3 で `krm` を直接書くようになった。つまり
 * 系列出力とミラーの違いは**容器の開け方だけ**（代表 path を直に指すか manifest 経由か）で、
 * 索引の読み口は models の公開部品（`gemma4PleAssetSource` + `readGemma4PleIndex`）が
 * そのまま使える。旧 sidecar（`ple.json` + `ple-NNNNN.safetensors`）を畳む暫定アダプタは
 * ここから消えた。
 *
 * NOTE: `helpers/memory-cache.ts` の規律（他パッケージのテスト**内部**へ依存しない — 向こうの
 * 都合がこちらへ漏れる）はそのまま効いている。`container-files.ts` から借りているのは
 * **容器という形式の綴り**（part 連番の見つけ方と区間の読み方 = Python 側 `container_parts` の
 * 鏡像）であって、runtime のテストの都合ではない。形式の綴りを models 側へ写すと、焼く側と
 * 読み返す側で規則が割れる。
 */

import type { OpenedContainer } from "@karume/runtime";
import { openSeriesContainer } from "../../../runtime/tests/helpers/container-files.ts";
import { gemma4PleAssetSource, readGemma4PleIndex } from "../../src/gemma/ple-index.ts";
import type { Gemma4PleHandle } from "./gemma-mirror.ts";

/** 系列出力の `model` 容器の代表 path（実体は part 連番 — 見つけ方は `resolveParts` が持つ）。 */
export const SERIES_MODEL_FILE = "model.krm";

/**
 * 既に開いている容器から PLE の索引と block の読み口を組む。
 *
 * `readGemma4PleIndex` は索引と容器の資産を**両方向で**突き合わせる（索引が指す資産が在るか /
 * 容器の PLE 資産が索引に載っているか / 役割と論理長）ので、片方だけ焼き直した系列出力は
 * ここで落ちる。
 */
export const seriesPleHandle = async (
  opened: OpenedContainer,
  where: string,
): Promise<Gemma4PleHandle> => ({
  index: await readGemma4PleIndex(where, gemma4PleAssetSource(opened)),
  openBlock: (asset) => opened.asset(asset),
});

/**
 * 系列出力の PLE を代表 path から開く（容器を自分で開く呼び手のための 1 行）。
 *
 * 重みも同じターンで読む呼び手は、容器を 2 度開かずに {@link seriesPleHandle} へ
 * `openSeriesContainer` の結果を渡す。
 */
export const openSeriesPle = async (root: URL): Promise<Gemma4PleHandle> =>
  await seriesPleHandle(
    await openSeriesContainer(new URL(SERIES_MODEL_FILE, root)),
    `test: ${decodeURIComponent(root.pathname)}`,
  );

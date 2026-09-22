// `assetComponentOpener` の門が使う最小の資産フィクスチャ
// （asset_container_components_test.ts と gpu_asset_container_components_test.ts の共有）。
//
// 容器の書き手は `container-fixture.ts` 経由（形式の道具 — 向こうの NOTE）。

import { assetComponentOpener, type ComponentOpener } from "../../src/hub/components.ts";
import { linearComponent, type TestContainer, writeContainer } from "./container-fixture.ts";

/** 実行できる部品 1 本（`linear` 1 段）の容器。重みは part 2 にしか無い。 */
const ditContainer: TestContainer = await writeContainer(linearComponent("dit"));

/** 単一形（全量 1 本）。 */
export const single = (): Uint8Array<ArrayBuffer> => ditContainer.single;

/** 分割形の part 1 本（`index` は part 番号 — 0 が descriptor）。 */
export const part = (index: number): Uint8Array<ArrayBuffer> => ditContainer.parts[index];

/** 分割形の part 列を `<役割>[i]` のキーで並べる（長さ 0 の part も並べる）。 */
export const parts = (key: string): Record<string, Uint8Array<ArrayBuffer>> =>
  Object.fromEntries(ditContainer.parts.map((bytes, index) => [`${key}[${index}]`, bytes]));

/** 家族側の資産アクセサ（`assetBuffer`）と同じ姿の最小実装。 */
const bufferOf =
  (assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>) => (key: string): ArrayBuffer => {
    if (!Object.hasOwn(assets, key)) {
      throw new Error(
        `test: 資産 '${key}' が無い（揃っているキー: ${Object.keys(assets).join(" / ")}）`,
      );
    }
    return assets[key].buffer;
  };

export const openerOf = (
  assets: Record<string, Uint8Array<ArrayBuffer>>,
  componentKeys: readonly string[] = ["dit"],
): Promise<ComponentOpener> =>
  assetComponentOpener("test", assets, bufferOf(assets), componentKeys);

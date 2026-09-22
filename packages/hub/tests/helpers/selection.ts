/**
 * 選択を扱うテスト側の道具。`karume/5` の hub は「取得キー → FileRef」の平坦な表を持たない
 * （選択は構造型 — ADR 0109 決定 6）ので、`fetchAssets` に渡す表と mock の配信表はテストが組む。
 *
 * MUST: path を手で写さない — manifest から導く形にしておくと、fixture を焼き直したときに
 * テスト側の表も一緒に動く（写すと「fixture を直したのにテストが古い path を配信し続ける」が
 * 黙って起きる）。
 */

import type { FileRef, Manifest, ResolveOptions } from "../../mod.ts";
import { resolveSelection } from "../../mod.ts";

/**
 * manifest が宣言する全 FileRef を**宣言順・path 一意**で並べる（長さ 0 の part は除く —
 * 配信する中身が無い）。mock の配信表を作るのに使う。
 */
export const declaredRefs = (manifest: Manifest): readonly FileRef[] => {
  const seen = new Map<string, FileRef>();
  const put = (ref: FileRef): void => {
    if (ref.size > 0 && !seen.has(ref.path)) seen.set(ref.path, ref);
  };
  for (const model of Object.keys(manifest.models)) {
    const entry = manifest.models[model];
    for (const weight of Object.keys(entry.weights)) {
      const dtypes = entry.weights[weight];
      for (const dtype of Object.keys(dtypes)) {
        for (const ref of dtypes[dtype].container.parts) put(ref);
      }
    }
    for (const name of Object.keys(entry.assets)) put(entry.assets[name]);
  }
  return [...seen.values()];
};

/**
 * 選択を「取得キー → FileRef」の表にする。キーはテストが読むための綴りで、hub はこの綴りを
 * 持たない（`fetchAssets` は渡された表のキーをそのまま返すだけ）:
 *
 * - コンテナの part … `<部品名>[<part の添字>]`（長さ 0 の part は表に出ない）
 * - assets … 資産名そのまま
 *
 * 並びは `selectionRefs` と同じ宣言順。**同じ実体を指すキーは落とさない**（別名の資産のように
 * 2 つの名前が 1 本を指す形があるため — 取得と進捗の一意化は `fetchAssets` が行う）。
 */
export const selectionFiles = (
  manifest: Manifest,
  options: ResolveOptions = {},
): Record<string, FileRef> => {
  const selection = resolveSelection(manifest, options);
  const files: Record<string, FileRef> = {};
  for (const name of Object.keys(selection.containers)) {
    selection.containers[name].parts.forEach((ref, index) => {
      if (ref.size > 0) files[`${name}[${index}]`] = ref;
    });
  }
  for (const name of Object.keys(selection.assets)) files[name] = selection.assets[name];
  return files;
};

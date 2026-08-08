/**
 * ローカルディレクトリ（`karume.json` を持つ配布形）から manifest + 資産を読む
 * — `fetchAssets` のローカル版。
 *
 * 公開面のデモ（`main.ts`）と torch 参照突合の dump 経路（`dump.ts`）が共有する。同じ資産の
 * 読み方が 2 本に割れると、「同じ preset なのに片方だけ別のファイルを開いていた」形の差が
 * 波形の突合に混ざる。
 */

import { parseManifest, resolveFiles } from "../../packages/hub/mod.ts";
import type { Sbv2Assets } from "../../packages/models/mod.ts";

export const MANIFEST_FILE = "karume.json";

/** `--source` がローカルの配布形ディレクトリか（そうでなければ HF リポ名として扱う）。 */
export const isLocalDist = (dir: string): Promise<boolean> =>
  Deno.stat(`${dir}/${MANIFEST_FILE}`).then(() => true, () => false);

export const loadLocalAssets = async (dir: string, preset?: string): Promise<Sbv2Assets> => {
  const manifest = parseManifest(await Deno.readTextFile(`${dir}/${MANIFEST_FILE}`));
  const files = resolveFiles(manifest, preset);
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const key of Object.keys(files)) {
    const { path } = files[key];
    const bytes = byPath.get(path) ?? await Deno.readFile(`${dir}/${path}`);
    byPath.set(path, bytes);
    assets = { ...assets, [key]: bytes };
  }
  return { manifest, assets };
};

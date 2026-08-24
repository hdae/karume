/**
 * ローカルディレクトリ（`karume.json` を持つ配布形）から manifest + 資産を読む
 * — `fetchAssets` のローカル版。
 *
 * 4 ファミリのデモ（`examples/<ファミリ>/main.ts`）と torch 参照突合の dump 経路
 * （`examples/sbv2/dump.ts`）が共有する。同じ資産の読み方が複数本に割れると、「同じ quant
 * なのに片方だけ別のファイルを開いていた」形の差が突合（波形・画像）に混ざる。
 */

import {
  type Manifest,
  parseManifest,
  resolveFiles,
  type ResolveOptions,
} from "../../packages/hub/mod.ts";

export const MANIFEST_FILE = "karume.json";

/**
 * ローカル配布形の資産束。4 ファミリの `<Family>Assets` はいずれもこの構造なので、
 * どの `fromAssets` へもそのまま渡せる（ファミリごとの写しを作らない）。
 */
export type LocalAssets = {
  readonly manifest: Manifest;
  readonly assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>;
};

/**
 * `--source` がローカルの配布形ディレクトリか（そうでなければ HF リポ名として扱う）。
 *
 * MUST: 「ローカルではない」と読むのは `NotFound` だけ。他の失敗（権限異常など）を握り潰すと、
 * そのパス文字列が黙って HF リポ名としてネットワークへ飛ぶ（未対応・想定外は fail loudly）。
 */
export const isLocalDist = async (dir: string): Promise<boolean> => {
  try {
    await Deno.stat(`${dir}/${MANIFEST_FILE}`);
    return true;
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) return false;
    throw error;
  }
};

export const loadLocalAssets = async (
  dir: string,
  selection: ResolveOptions = {},
): Promise<LocalAssets> => {
  const manifest = parseManifest(await Deno.readTextFile(`${dir}/${MANIFEST_FILE}`));
  const files = resolveFiles(manifest, selection);
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

/**
 * ローカルディレクトリ（`karume.json` を持つ配布形）から manifest + 資産を読む
 * — `fetchAssets` のローカル版。
 *
 * ## 役割は 2 つに割れている
 *
 * - {@link isLocalDist}（ローカル配布形かの判定）は**デモと dump の共有**。デモはこれで
 *   使い捨ての HF 形サーバ（`local-dist-server.ts`）へ回すかを決める。
 * - {@link loadLocalAssets}（全量読み）は **`examples/sbv2/dump.ts` 専用**。torch 参照突合の
 *   dump は分割対象外の小資産しか触らないので、この面のままでよい。
 *
 * NOTE: 全量読みは shard 分割された配布形も読める（`resolveFiles` が返す `<役割>[i]` の
 * キーをそのまま並べるだけ — 連結はしない）。`from*Assets` はその形の Record を受けて
 * shard 逐次面へ流す（X2-101 — `packages/models/src/hub/components.ts`）。ただし**ホスト RAM に
 * 全 shard が同時に載る**面であることは変わらないので、デモは従来どおり `fromPretrained`
 * （取得層の shard 面 = 常に「今の 1 本」だけ）越しに回す。
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

/**
 * ローカルディレクトリ（`karume.json` を持つ配布形）から manifest + 資産を読む
 * — `fetchAssets` のローカル版。
 *
 * ## 役割は 2 つに割れている
 *
 * - {@link isLocalDist}（ローカル配布形かの判定）は**デモと dump の共有**。デモはこれで
 *   `--source` を取得元ハンドル（`denoDirectory`）へ回すか HF リポジトリ名として読むかを
 *   決める（`local-source.ts`）。
 * - {@link loadLocalAssets}（全量読み）は **`examples/sbv2/dump.ts` 専用**。torch 参照突合の
 *   dump は分割対象外の小資産しか触らないので、この面のままでよい。
 *
 * NOTE: 全量読みはコンテナの part 列をキー `<部品>[i]`（part 0 から添字順・長さ 0 の part も
 * 含む）で並べるだけで、連結はしない。`from*Assets` はその形の Record を受けて part 列のまま
 * `openContainer` へ渡す（X2-101 — `packages/models/src/hub/components.ts`）。ただし**ホスト
 * RAM に全 part が同時に載る**面であることは変わらないので、デモは従来どおり
 * `fromPretrained`（取得面 = 区間読み）越しに回す。
 */

import {
  type FileRef,
  type Manifest,
  parseManifest,
  type ResolveOptions,
  resolveSelection,
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
  const selected = resolveSelection(manifest, selection);
  const byPath = new Map<string, Uint8Array<ArrayBuffer>>();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  const readInto = async (key: string, ref: FileRef): Promise<void> => {
    // MUST: 越境参照（ADR 0038 §7）は (repo, commit SHA) からしか取れない — 全量面では解けない
    // のでキーごと落とす。`${dir}/${path}` を無条件に開くと、同名 path がローカルにも在る
    // ときに**別リポのバイト列が黙って差し替わる**（`path` だけでは 1 本のファイルを指さない
    // — packages/hub/src/manifest.ts の fileRefKey がその理由）。
    // 双子実装 = packages/models/tests/e2e_anima_test.ts の readLocalAssets。
    if (ref.repo !== undefined) {
      throw new Error(`全量面では越境参照 '${ref.repo}' を解けない（キー ${key}）`);
    }
    const bytes = byPath.get(ref.path) ?? await Deno.readFile(`${dir}/${ref.path}`);
    byPath.set(ref.path, bytes);
    assets = { ...assets, [key]: bytes };
  };
  // MUST: 長さ 0 の part も並べる — part の添字が容器の中の id なので、飛ばすと以降の part が
  // 1 つずつ繰り上がって別の part として読まれる。
  for (const name of Object.keys(selected.containers)) {
    const { parts } = selected.containers[name];
    for (const [index, ref] of parts.entries()) await readInto(`${name}[${index}]`, ref);
  }
  for (const name of Object.keys(selected.assets)) await readInto(name, selected.assets[name]);
  return { manifest, assets };
};

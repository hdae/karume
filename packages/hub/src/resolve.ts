/**
 * preset → 取得すべきファイル表の解決（ADR 0038 §5 の `resolve(preset?)`）。
 */

import { ManifestReferenceError } from "./errors.ts";
import type { ComponentFiles, FileRef, Manifest } from "./manifest.ts";

/**
 * 取得キー → ファイル参照。キーは `"<component>"` と `"<component>.<extra>"`。
 *
 * NOTE: 同一 `path` を指すキーが複数あっても**キーは落とさない**（落とすと呼び出し側が
 * そのコンポーネントの bytes を引けなくなる）。取得と進捗総量の path 一意化は取得層
 * （`fetchAssets`）が行い、network 取得も進捗加算もちょうど 1 回になる。
 */
export type ResolvedFiles = Readonly<Record<string, FileRef>>;

const addFiles = (
  manifest: Manifest,
  files: ResolvedFiles,
  component: string,
  chosen: ComponentFiles,
): ResolvedFiles => {
  let next = files;
  const put = (key: string, ref: FileRef): void => {
    if (Object.hasOwn(next, key)) {
      throw new ManifestReferenceError(
        `取得キー '${key}' が衝突した（コンポーネント名と extras 名の綴りを見直すこと）`,
        { available: manifest.available },
      );
    }
    next = { ...next, [key]: ref };
  };
  put(component, chosen.file);
  for (const extra of Object.keys(chosen.extras)) {
    put(`${component}.${extra}`, chosen.extras[extra]);
  }
  return next;
};

/**
 * preset を選んで取得すべきファイル表を作る。preset 省略時は manifest の `defaultPreset`。
 *
 * 解決可能性（weights の完全写像・ラベルの実在）は parse 時に検査済みなので、ここで新たに
 * 落ちるのは「存在しない preset 名を指定した」場合だけ。
 */
export const resolveFiles = (manifest: Manifest, preset?: string): ResolvedFiles => {
  const name = preset ?? manifest.defaultPreset;
  if (!Object.hasOwn(manifest.presets, name)) {
    throw new ManifestReferenceError(
      `preset '${name}' は manifest に無い（利用可能: ${manifest.available.presets.join(" / ")}）`,
      { available: manifest.available },
    );
  }
  const weights = manifest.presets[name].weights;
  let files: ResolvedFiles = {};
  for (const component of Object.keys(manifest.components)) {
    const entry = manifest.components[component];
    const chosen = entry.kind === "single" ? entry.files : entry.variants[weights[component]];
    files = addFiles(manifest, files, component, chosen);
  }
  return files;
};

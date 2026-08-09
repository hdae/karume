/**
 * model / quant → 取得すべきファイル表の解決（ADR 0041 §8 の `resolveFiles(manifest, {…})`）。
 */

import { ManifestReferenceError } from "./errors.ts";
import type { FileRef, Manifest, ModelEntry, WeightFiles } from "./manifest.ts";

/**
 * 取得キー → ファイル参照。キーは weights 名 / `"<weights>.<extra>"` / assets 名。
 *
 * NOTE: 同一 `path` を指すキーが複数あっても**キーは落とさない**（落とすと呼び出し側が
 * その weights の bytes を引けなくなる）。取得と進捗総量の path 一意化は取得層
 * （`fetchAssets`）が行い、network 取得も進捗加算もちょうど 1 回になる。
 */
export type ResolvedFiles = Readonly<Record<string, FileRef>>;

/** {@link resolveFiles} の選択軸。どちらも省略すると manifest の既定が使われる。 */
export type ResolveOptions = {
  /** モデル名（省略時は `defaultModel`）。 */
  readonly model?: string;
  /** quant 名（省略時はそのモデルの `defaultQuant`）。 */
  readonly quant?: string;
};

/**
 * モデルを選ぶ。未知のモデル名は**利用可能な一覧**を添えて落とす（v2 で初めて列挙が機械可読に
 * なった — ADR 0041 §8）。
 */
const selectModel = (manifest: Manifest, model?: string): ModelEntry => {
  const name = model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, name)) {
    throw new ManifestReferenceError(
      `model '${name}' は manifest に無い（利用可能: ${manifest.available.models.join(" / ")}）`,
      { available: manifest.available },
    );
  }
  return manifest.models[name];
};

const addWeightFiles = (
  entry: ModelEntry,
  files: ResolvedFiles,
  name: string,
  chosen: WeightFiles,
): ResolvedFiles => {
  let next = files;
  const put = (key: string, ref: FileRef): void => {
    if (Object.hasOwn(next, key)) {
      throw new ManifestReferenceError(
        `取得キー '${key}' が衝突した（weights 名・extras 名・assets 名の綴りを見直すこと）`,
        { available: entry.available },
      );
    }
    next = { ...next, [key]: ref };
  };
  put(name, chosen.file);
  for (const extra of Object.keys(chosen.extras)) {
    put(`${name}.${extra}`, chosen.extras[extra]);
  }
  return next;
};

/**
 * model と quant を選んで取得すべきファイル表を作る。省略時は `defaultModel` / `defaultQuant`。
 *
 * 解決可能性（weights 写像の完全性・dtype ラベルの実在）は parse 時に検査済みなので、ここで
 * 新たに落ちるのは「存在しない model / quant 名を指定した」場合と、weights と assets が同名を
 * 主張した場合だけ。
 */
export const resolveFiles = (
  manifest: Manifest,
  options: ResolveOptions = {},
): ResolvedFiles => {
  const entry = selectModel(manifest, options.model);
  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new ManifestReferenceError(
      `quant '${quantName}' は manifest に無い（利用可能: ${entry.available.quants.join(" / ")}）`,
      { available: entry.available },
    );
  }
  const chosen = entry.quants[quantName].weights;
  let files: ResolvedFiles = {};
  for (const name of Object.keys(entry.weights)) {
    files = addWeightFiles(entry, files, name, entry.weights[name][chosen[name]]);
  }
  for (const name of Object.keys(entry.assets)) {
    if (Object.hasOwn(files, name)) {
      throw new ManifestReferenceError(
        `取得キー '${name}' が衝突した（weights 名・extras 名・assets 名の綴りを見直すこと）`,
        { available: entry.available },
      );
    }
    files = { ...files, [name]: entry.assets[name] };
  }
  return files;
};

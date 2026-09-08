/**
 * model / quant → 取得すべきファイル表の解決（ADR 0041 §8 の `resolveFiles(manifest, {…})`）。
 */

import { ManifestReferenceError } from "./errors.ts";
import type { FileRef, Manifest, ModelEntry, WeightFiles } from "./manifest.ts";

/**
 * 取得キー → ファイル参照。キーは weights 名 / `"<weights>[<shard>]"`（複数 shard のとき）/
 * `"<weights>.<extra>"` / assets 名。
 *
 * NOTE: 同一 `path` を指すキーが複数あっても**キーは落とさない**（落とすと呼び出し側が
 * その weights の bytes を引けなくなる）。取得と進捗総量の path 一意化は取得層
 * （`fetchAssets`）が行い、network 取得も進捗加算もちょうど 1 回になる。
 */
export type ResolvedFiles = Readonly<Record<string, FileRef>>;

/** {@link resolveFiles} の選択軸。どれも省略すると manifest の既定が使われる。 */
export type ResolveOptions = {
  /** モデル名（省略時は `defaultModel`）。 */
  readonly model?: string;
  /** quant 名（省略時はそのモデルの `defaultQuant`）。 */
  readonly quant?: string;
  /**
   * 取得する weights の**部分集合**（省略時はそのモデルの weights 全数）。
   *
   * 1 つのモデルが「本体だけでも動き、追加の役割を足すこともできる」形（gemma4 の
   * `model` + 投機の `drafter`）を配布形の分割なしに扱うための軸である。使わない役割の
   * shard は取得キーの表に**現れない**ので、DL も進捗の総量も在庫の勘定もその役割ぶんだけ
   * 減る（表に残すと `@karume/models` の shard 面がそれを「用途不明の資産」として全量取得する）。
   *
   * MUST: 未知の weights 名は {@link ManifestReferenceError}（綴り間違いを黙って
   * 「取らない」に畳まない）。同じ名前を 2 度並べるのも拒否する — 部分集合の指定として
   * 意味が無く、取得キーの衝突と区別が付かない綴りだから。
   *
   * NOTE: 表の並びは**宣言順**（manifest の `weights` の順）で、この配列の順ではない —
   * 呼び手の並べ方で取得キーの列が動くと、shard の位置で引き当てる層（`@karume/models` の
   * `loadShardComponents`）が呼び手ごとに別の順序を見ることになる。
   * NOTE: 空配列は「weights を 1 本も取らない（assets だけ）」— 拒否はしない（部分集合として
   * 定義された値で、要求どおり weights の無い表が返る）。
   */
  readonly weights?: readonly string[];
};

/**
 * 取得する weights 名を**宣言順**で決める（省略時は全数）。
 *
 * MUST: 実在検査はここ 1 本（`quants` の完全写像は parse が保証済みなので、実在しさえすれば
 * dtype ラベルは必ず引ける）。
 */
const selectWeightNames = (
  entry: ModelEntry,
  weights: readonly string[] | undefined,
): readonly string[] => {
  const declared = Object.keys(entry.weights);
  if (weights === undefined) return declared;
  const chosen = new Set<string>();
  for (const name of weights) {
    if (!Object.hasOwn(entry.weights, name)) {
      throw new ManifestReferenceError(
        `weights '${name}' は manifest に無い（利用可能: ${declared.join(" / ")}）`,
        { available: entry.available },
      );
    }
    if (chosen.has(name)) {
      throw new ManifestReferenceError(
        `weights '${name}' が 2 度指定された（取得する weights の部分集合は重複なしで並べる）`,
        { available: entry.available },
      );
    }
    chosen.add(name);
  }
  return declared.filter((name) => chosen.has(name));
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

/**
 * shard の取得キー。1 shard なら weights 名そのもの（単一ファイル配布のキーを動かさない）、
 * 複数 shard なら宣言順の位置を添えた `<weights>[i]` に展開する。
 *
 * MUST: 区切りに `.` を使わない — `<weights>.<extra>` は extras の名前空間で、shard を同じ形に
 * すると extras 名と shard 番号が同じキーを主張し得る（この表はキーで引かれるので衝突は取り違え
 * そのもの）。`[i]` は名前空間として交わらない。
 */
const shardKey = (name: string, index: number, count: number): string =>
  count === 1 ? name : `${name}[${index}]`;

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
  // 宣言順のまま並べる（配列位置 = shard id — 並べ替えると識別子が壊れる）。
  chosen.shards.forEach((ref, index) => put(shardKey(name, index, chosen.shards.length), ref));
  for (const extra of Object.keys(chosen.extras)) {
    put(`${name}.${extra}`, chosen.extras[extra]);
  }
  return next;
};

/**
 * model と quant を選んで取得すべきファイル表を作る。省略時は `defaultModel` / `defaultQuant`。
 * {@link ResolveOptions.weights} を渡すとその部分集合だけを並べる（assets は常に全数）。
 *
 * 解決可能性（weights 写像の完全性・dtype ラベルの実在）は parse 時に検査済みなので、ここで
 * 新たに落ちるのは「存在しない model / quant / weights 名を指定した」場合と、weights と assets が
 * 同名を主張した場合だけ。
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
  for (const name of selectWeightNames(entry, options.weights)) {
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

/**
 * `pipelineConfig` を読む基本 reader（family 横断の下位層）。
 *
 * ADR 0038 §1 のとおり `pipelineConfig` のスキーマは各パイプライン実装が所有する。ここに置くのは
 * **どの family でも同じ骨格になる読み取りの手続きだけ**で、許可キー集合・既定値・許容する数値
 * 条件・受理する固定値といった**宣言の正本は各 family の `config.ts` に残す**。どの欄を読んだかを
 * 示す文言（`where` / `key`）と条件の説明（`requirement` / `why`）も呼び手から受け取るので、
 * エラー文言の責任も family 側にある。
 *
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 * MUST: 未知キーは fail loudly（綴り違いが黙って既定へ縮退すると、配布者の意図した前処理と
 * 実行が食い違ったまま気づけない）。
 */

/** mean / std の要素数（RGB — アルファは入口で受け取らない）。 */
const CHANNELS = 3;

/**
 * 素のオブジェクトか（配列は**含めない** — 欄の集まりとして読む先で `["a"]` が黙って
 * `{ "0": "a" }` として通ると、未知キー検査も既定値も別の意味で動く）。
 */
export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** 欄の集まりとして読む（欠落とオブジェクトでないのは同じ直し方なので 1 文言）。 */
export const readRecord = (raw: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(raw)) throw new Error(`${where}: 無い / オブジェクトでない`);
  return raw;
};

/** 寸法・件数の欄の受理条件（{@link readNumber} の `check` に渡す）。 */
export const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;

/** 許可集合の外にあるキーを 1 つでも見つけたら落とす。 */
export const assertAllowedKeys = (
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void => {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      throw new Error(`${where}: 未知キー '${key}'（許可: ${allowed.join(" / ")}）`);
    }
  }
};

/** 数の欄を読む（値域の判定 `check` と、外れたときの文言 `requirement` は呼び手が渡す）。 */
export const readNumber = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (typeof value !== "number" || !check(value)) {
    throw new Error(`${where}.${key}: ${requirement}（${String(value)}）`);
  }
  return value;
};

/**
 * チャネルごとの定数 3 本を読む。
 *
 * MUST: `std` は 0 を弾く（呼び手の `parse*PipelineConfig` が `check` で渡す）— 0 除算は例外を
 * 出さず `±Infinity` の `pixel_values` を作り、グラフは NaN を吐きながら shape だけ合う。
 */
export const readChannels = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): readonly [number, number, number] => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (!Array.isArray(value) || value.length !== CHANNELS) {
    throw new Error(`${where}.${key}: 長さ ${CHANNELS} の配列でない（${JSON.stringify(value)}）`);
  }
  for (const entry of value) {
    if (typeof entry !== "number" || !check(entry)) {
      throw new Error(`${where}.${key}: ${requirement}（${JSON.stringify(value)}）`);
    }
  }
  return [value[0], value[1], value[2]];
};

/** 受理集合が 1 値しかない欄。綴り違いも対応外も同じ文言で落とす。 */
export const readOnly = <T extends string>(
  raw: Record<string, unknown>,
  key: string,
  where: string,
  accepted: T,
  why: string,
): T => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (value !== accepted) {
    throw new Error(
      `${where}.${key}: この実装が対応するのは '${accepted}' だけ（${String(value)}）— ${why}`,
    );
  }
  return accepted;
};

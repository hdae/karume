/**
 * `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、手書きの検査を全て parse 時に走らせる。
 *
 * MUST: 未知キーは fail loudly（`{ imageMean: …, image_mean: … }` のような綴り違いが黙って
 * 既定へ縮退すると、配布者の意図した前処理と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * ## MUST: 補間は `"bicubic"` **だけ**を受理する（BiRefNet / SigLIP2 と受理集合が違う）
 *
 * Depth Anything V2 の上流 `preprocessor_config.json` は `"resample": 3`（**PIL の定数で
 * BICUBIC**）で、SigLIP2 / BiRefNet の bilinear（2）とは別のフィルタ。`resizeRgb8` の既定は
 * 既存 2 ファミリの値（bilinear）なので、宣言を分岐に使わず**受理集合を 1 値に絞る**形にして
 * おかないと、取り違えが `pixel_values` の最大 0.59 ずれ（uint8 1 LSB = 0.0175 の 34 倍 —
 * 実測）として通る。値は shape も値域も合ったままなので、深度地図は「それらしく」出る。
 *
 * 欄名も値域も BiRefNet の 5 欄と同じだが、**スキーマは共有しない**（ADR 0038 §1 の流儀 —
 * 受理集合がファミリごとに違うものを 1 本に畳むと、片方を緩めたときにもう片方が黙って緩む）。
 *
 * ## NOTE: 解像度は宣言であってグラフの正本ではない
 *
 * `imageWidth` / `imageHeight` は焼かれたグラフの入力宣言と**同じ数**で、組み立て段
 * （`tools/exporter/karume/dist.py` の Depth Anything 節）が上流 `preprocessor_config.json` の
 * `size` から書き、同じ席でグラフと突き合わせる。それでも宣言を置くのは、①前処理の resize 先は
 * グラフを開く前に読めるべき欄で ②モデルカードが解像度を説明できるようにするため。二重化した
 * 分は {@link DepthAnythingPipelineConfig} を使う側（`pipeline.ts` の `assertStaticDim`）が
 * **毎回グラフと突き合わせる**ので、食い違ったまま走ることはない。
 *
 * NOTE: rescale の除数（255）はここに無い。`normalizeToNchw` の入口が 8bit の画素列
 * （`Rgb8Image`）で閉じており、実行時に選べない数を宣言だけ持たせても正本が 2 つ増える
 * （`src/siglip2/config.ts` / `src/birefnet/config.ts` と同じ判断）。
 *
 * NOTE: 上流 config の `keep_aspect_ratio` / `ensure_multiple_of` はここに無い。焼かれた
 * グラフが**正方 1 点**（518² = patch 14 × 37）でしか受け取らないので、アスペクト比を保つ
 * 経路には行き先が無い（伸縮しか選べない）。宣言だけ置くと「保てるのに保っていない」と
 * 読める欄になるため、事実はモデルカードの散文が持つ。
 */

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const DEPTH_ANYTHING_PIPELINE_NAME = "depth-anything";
export const DEPTH_ANYTHING_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "interpolation",
];

/** この実装が受理する唯一の補間（モジュール doc の MUST）。 */
const INTERPOLATION = "bicubic";

/** mean / std の要素数（RGB — アルファは入口で受け取らない）。 */
const CHANNELS = 3;

/** 正規化の定数（`[0, 1]` 尺度 — 上流 `preprocessor_config.json` の綴りのまま）。 */
export type DepthAnythingChannels = readonly [number, number, number];

export type DepthAnythingPipelineConfig = {
  /** 前処理の resize 先（= 焼かれたグラフの入力幅）。 */
  readonly imageWidth: number;
  /** 前処理の resize 先（= 焼かれたグラフの入力高さ）。 */
  readonly imageHeight: number;
  readonly imageMean: DepthAnythingChannels;
  readonly imageStd: DepthAnythingChannels;
  /** ADR 0038 §1 の流儀で**宣言**として持つ（分岐用ではない — モジュール doc の MUST）。 */
  readonly interpolation: typeof INTERPOLATION;
};

const assertAllowedKeys = (
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

const readNumber = (
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

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;

/**
 * チャネルごとの定数 3 本を読む。
 *
 * MUST: `std` は 0 を弾く（{@link parseDepthAnythingPipelineConfig}）— 0 除算は例外を出さず
 * `±Infinity` の `pixel_values` を作り、グラフは NaN を吐きながら shape だけ合う。
 */
const readChannels = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): DepthAnythingChannels => {
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
const readOnly = <T extends string>(
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

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseDepthAnythingPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): DepthAnythingPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "正の整数でない";
  return {
    imageWidth: readNumber(raw, "imageWidth", where, isPositiveInteger, positive),
    imageHeight: readNumber(raw, "imageHeight", where, isPositiveInteger, positive),
    // mean は負でも構わない（`(x − mean·255) / (std·255)` の平行移動）が、有限でなければ
    // 全画素が NaN になる。
    imageMean: readChannels(raw, "imageMean", where, Number.isFinite, "有限の数でない要素がある"),
    imageStd: readChannels(
      raw,
      "imageStd",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "正の有限数でない要素がある",
    ),
    interpolation: readOnly(
      raw,
      "interpolation",
      where,
      INTERPOLATION,
      "上流 preprocessor_config.json の resample は 3（PIL の BICUBIC）で、" +
        "bilinear で通すと pixel_values が uint8 1 LSB の 34 倍ずれる",
    ),
  };
};

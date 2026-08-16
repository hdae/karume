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
 * ## MUST: モデル固有の数はここ（= manifest）が正本で、TS 側に定数を置かない
 *
 * BiRefNet 系は上流に `preprocessor_config.json` が無く、正規化定数は同梱 `handler.py` の
 * `ImagePreprocessor`（ImageNet 統計）にしか無い。だからといって TS に 0.485 を直書きすると、
 * 別の統計で学習された派生モデル（この形の配布形は fine-tune 違いが並ぶ）で**沈黙誤値**に
 * なる — 正規化がずれても shape は合い、マットは「それらしく」出るからである。
 * `imageMean` / `imageStd` を宣言として持ち、配布形ごとに読む。
 *
 * ## MUST: 補間は**宣言**として持ち、対応外は値を保持せずパース時に拒否する
 *
 * `src/image/preprocess.ts` が実装しているのは antialias 付き **bilinear** だけ。上流
 * （`handler.py` / モデルカードの利用例）は `torchvision.transforms.Resize((S, S))` を既定の
 * 補間で通す = bilinear なので現状は一致するが、bicubic を要求する派生を黙って bilinear で
 * 通すと **resize の値が最大 47/255 ずれたまま**ロードも実行も通る（実測 — 前処理層の
 * モジュール doc）。分岐を持つのではなく**受理しない**（型としても `"bilinear"` しか
 * 表せない）。
 *
 * NOTE: rescale の除数（255）はここに無い。`normalizeToNchw` の入口が 8bit の画素列
 * （`Rgb8Image`）で閉じており、実行時に選べない数を宣言だけ持たせても正本が 2 つ増える
 * （`src/siglip2/config.ts` と同じ判断）。
 *
 * NOTE: `imageWidth` / `imageHeight` は焼かれたグラフの入力宣言と**同じ数**で、組み立て段
 * （`tools/exporter/karume/dist.py` の BiRefNet 節）がグラフから導いて書く。それでも宣言を
 * 置くのは、①前処理の resize 先はグラフを開く前に読めるべき欄で ②モデルカードが解像度を
 * 説明できるようにするため。二重化した分は {@link BirefnetPipelineConfig} を使う側
 * （`pipeline.ts` の `assertStaticDim`）が**毎回グラフと突き合わせる**ので、食い違ったまま
 * 走ることはない。
 */

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const BIREFNET_PIPELINE_NAME = "birefnet";
export const BIREFNET_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = [
  "imageWidth",
  "imageHeight",
  "imageMean",
  "imageStd",
  "interpolation",
];

/** この実装が受理する唯一の補間（モジュール doc の MUST）。 */
const INTERPOLATION = "bilinear";

/** mean / std の要素数（RGB — アルファは入口で受け取らない）。 */
const CHANNELS = 3;

/** 正規化の定数（`[0, 1]` 尺度 — 上流 `handler.py` の綴りのまま）。 */
type BirefnetChannels = readonly [number, number, number];

export type BirefnetPipelineConfig = {
  /** 前処理の resize 先（= 焼かれたグラフの入力幅）。 */
  readonly imageWidth: number;
  /** 前処理の resize 先（= 焼かれたグラフの入力高さ）。 */
  readonly imageHeight: number;
  readonly imageMean: BirefnetChannels;
  readonly imageStd: BirefnetChannels;
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
 * MUST: `std` は 0 を弾く（{@link parseBirefnetPipelineConfig}）— 0 除算は例外を出さず
 * `±Infinity` の `pixel_values` を作り、グラフは NaN を吐きながら shape だけ合う。
 */
const readChannels = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
  check: (value: number) => boolean,
  requirement: string,
): BirefnetChannels => {
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
export const parseBirefnetPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): BirefnetPipelineConfig => {
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
      "前処理の resize が antialias 付き bilinear の 1 本しかない（src/image/preprocess.ts）",
    ),
  };
};

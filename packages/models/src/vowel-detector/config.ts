/**
 * `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、手書きの検査を全て parse 時に走らせる。
 *
 * MUST: 未知キーは fail loudly（`{ maxFrames: …, max_frames: … }` のような綴り違いが
 * 黙って既定へ縮退すると、配布者の意図した上限と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * ## ここに並ぶ 4 欄のうち、**実行時に効くのは `maxFrames` だけ**
 *
 * 他の 3 欄（`sampleRate` / `featureDim` / `classes`）は BiRefNet の `interpolation` と同じ
 * **宣言**で、受理集合はこの実装が持つ 1 値きり。分岐を持たせるのではなく、外れた配布形を
 * パース時に**受理しない**:
 *
 * - `sampleRate` — 特徴抽出はリサンプラを持たず、窓長・hop・mel 基底の帯域が 16kHz 前提で
 *   焼かれている（`features.ts` のモジュール doc）。別の周波数で学習された派生を通すと、
 *   `detect` は最後まで走って**それらしい別の母音列**を出す（形は合う）。
 * - `featureDim` — グラフ入力の最終軸と突き合わせる先（`pipeline.ts` の `assertGraph`）。
 * - `classes` — **並びが id**（`postprocess.ts` の `LIPSYNC_CLASSES`）。並びが違う配布形は
 *   ラベルが置換されるだけで、区間割りも `.lab` の書式も完全に成立する = 最も気づけない。
 *
 * ## MUST: `maxFrames` は配布形が宣言する運用上限（TS 側に定数を持たない）
 *
 * グラフの時間軸は記号なので、**IR は上限を持たない**（`docs/ir-v1.md` の `symbols` は名前の
 * 列挙だけ）。焼くときに `Dim(max=…)` で宣言した値を配布形から受け、`detect` が特徴抽出の
 * 直後に落とす（正本は exporter の `dist.py`）。ここに既定値や定数を置くと、別の上限で焼いた
 * 配布形が来たときにホストだけが古い数を持つ形になる（SBV2 の `maxTokens` と同じ判断）。
 *
 * ## MUST: 後処理の定数（`switchPenalty` / `minDurationFrames` / `frameSec`）はここに無い
 *
 * 正本は `postprocess.ts` の module 定数（= 学習時の平滑化そのもの）。宣言の席を作ると
 * 「学習時と違う平滑化を配布形が指定できる」ことになるが、そこを動かして良い根拠が無い。
 * 値の一致は `tests/vowel_detector_host_test.ts` が上流 `feature_config.json` の写しと
 * 突き合わせる。
 */

import { FEATURE_DIM, SAMPLE_RATE } from "./features.ts";
import { LIPSYNC_CLASSES } from "./postprocess.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const VOWEL_DETECTOR_PIPELINE_NAME = "vowel-detector";
export const VOWEL_DETECTOR_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = [
  "sampleRate",
  "featureDim",
  "classes",
  "maxFrames",
];

/** 入力長の刻み（出力は 20ms 格子 = 入力 2 フレームで 1 本 — グラフ入力は `2T`）。 */
const LENGTH_MULTIPLE = 2;

export type VowelDetectorPipelineConfig = {
  /** 入力波形のサンプリング周波数。宣言のみ（受理集合は {@link SAMPLE_RATE} の 1 値）。 */
  readonly sampleRate: typeof SAMPLE_RATE;
  /** グラフ入力の特徴次元。宣言のみ（受理集合は {@link FEATURE_DIM} の 1 値）。 */
  readonly featureDim: typeof FEATURE_DIM;
  /** 出力クラス（**並びが id**）。受理するのは {@link LIPSYNC_CLASSES} と同一の並びだけ。 */
  readonly classes: typeof LIPSYNC_CLASSES;
  /**
   * 1 回の `detect` が受ける 10ms フレーム数の上限（= グラフを焼いたときの記号次元の上限）。
   * **2 の倍数**（グラフ入力は `2T`）。
   */
  readonly maxFrames: number;
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

/** 受理集合が 1 値しかない数の欄。綴り違いも対応外も同じ文言で落とす。 */
const readOnlyNumber = <T extends number>(
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
      `${where}.${key}: この実装が対応するのは ${accepted} だけ（${String(value)}）— ${why}`,
    );
  }
  return accepted;
};

/**
 * クラス語彙を読む。**並びまで**一致を要求する（モジュール doc の 3 つ目）。
 */
const readClasses = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
): typeof LIPSYNC_CLASSES => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (
    !Array.isArray(value) || value.length !== LIPSYNC_CLASSES.length ||
    value.some((entry, index) => entry !== LIPSYNC_CLASSES[index])
  ) {
    throw new Error(
      `${where}.${key}: この実装が対応するのは [${LIPSYNC_CLASSES.join(", ")}] の並びだけ` +
        `（${JSON.stringify(value)}）— 並びがそのままクラス id なので、` +
        "違う並びはラベルが置換されるだけで区間割りも .lab も成立してしまう",
    );
  }
  return LIPSYNC_CLASSES;
};

/**
 * 運用上限を読む。
 *
 * MUST: 2 の倍数であることまで見る — グラフ入力は `2T` なので、奇数の上限は「その 1 本だけは
 * 通らない上限」という、宣言としては成立するが意味の壊れた数になる。
 */
const readMaxFrames = (
  raw: Record<string, unknown>,
  key: string,
  where: string,
): number => {
  if (!Object.hasOwn(raw, key)) throw new Error(`${where}.${key}: 無い`);
  const value = raw[key];
  if (
    typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0 ||
    value % LENGTH_MULTIPLE !== 0
  ) {
    throw new Error(
      `${where}.${key}: 正の ${LENGTH_MULTIPLE} の倍数でない（${JSON.stringify(value)}）` +
        "— 出力は 20ms 格子で、入力 2 フレームが 1 本になる",
    );
  }
  return value;
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseVowelDetectorPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): VowelDetectorPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  return {
    sampleRate: readOnlyNumber(
      raw,
      "sampleRate",
      where,
      SAMPLE_RATE,
      "特徴抽出はリサンプラを持たず、窓長 / hop / mel 基底が 16kHz 前提で焼かれている",
    ),
    featureDim: readOnlyNumber(
      raw,
      "featureDim",
      where,
      FEATURE_DIM,
      "80 次元 log-mel + DSP 3 次元は学習時の契約そのもの（src/vowel-detector/features.ts）",
    ),
    classes: readClasses(raw, "classes", where),
    maxFrames: readMaxFrames(raw, "maxFrames", where),
  };
};

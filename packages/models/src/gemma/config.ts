/**
 * gemma4 の `pipelineConfig` スキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・
 * 検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）ので、**形の正本は
 * このモジュール**で、手書きの検査を全て parse 時に走らせる。焼く側の正本は
 * `tools/export-recipes/gemma4/distribution.py`（`gemma4_pipeline_config` / `gemma4_sampler`）。
 *
 * MUST: 未知キーは fail loudly（`{ capacity: 640, capasity: 640 }` のような綴り違いが黙って
 * 既定へ縮退すると、配布者の意図した宣言と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * ## NOTE: 公開配布リポの pin 定数（`*_CURRENT` — ADR 0073）はまだ無い
 *
 * gemma4 の配布形は**まだ公開していない**。ADR 0073 決定 1 のとおり、pin 定数を持つのは公開
 * リポを持つファミリだけで（`packages/models/tests/current_source_test.ts` の対象もそれ）、
 * 公開前に「それらしい SHA」を置くと 404 にしかならない定数が公開面に生える。`fromPretrained`
 * に既定は無いので、取得元は呼び手が `{ repo, revision }` で明示する。公開時に
 * `docs/release-runbook.md` §3 の手順でここへ `GEMMA4_CURRENT` を足す。
 */

import type { SamplerSpec } from "../generation/sampler.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const GEMMA4_PIPELINE_NAME = "gemma4";
export const GEMMA4_PIPELINE_MAJOR = 1;

/**
 * 静的配線のうち**資産世代ごとに動く数**（配布形の `pipelineConfig` が宣言する）。
 *
 * NOTE: 記号（full スロットの容量記号）はここに置かない — グラフから導出できるものを宣言に
 * 二重持ちすると、片方だけ古びる（`pipeline.ts` の `capacitySymbolOf`）。
 */
export type Gemma4PipelineConfig = {
  /** 固定長 prefill chunk の行数（ADR 0066 決定 4 — context の計画時定数）。 */
  readonly chunkLength: number;
  /** 資産が引ける絶対位置の排他的上限（= 焼き込んだ RoPE 表の行数）。 */
  readonly maxPosition: number;
  /** full スロットの容量（会話が使える最大の論理長 — 実行時に選ぶ）。 */
  readonly capacity: number;
  /**
   * 配布形が宣言する sampler の既定（ADR 0083 決定 7）。
   *
   * リクエストが `sampler` を省略したときに使われる。宣言が無ければ低層の既定（温度 0 =
   * greedy）のままで、これは「sampler 未宣言 = 温度 0 の縮退形」という位置づけである
   * （parity 門はこの経路で生き続ける）。
   */
  readonly sampler?: SamplerSpec;
};

const ROOT_KEYS: readonly string[] = ["chunkLength", "maxPosition", "capacity", "sampler"];

/**
 * 配布形が宣言できる sampler の欄。
 *
 * MUST: `SamplerSpec` 全部を開けない — `logitBias` は `Map` で JSON に載らず、`seed` /
 * `repetitionPenalty` は「配布者が推奨する」性質の値ではない（上流 `generation_config.json`
 * が持つのもこの 3 つだけ）。欄を増やすのは実需が出てからで、それまでは未知キーとして落ちる。
 */
const SAMPLER_KEYS: readonly string[] = ["temperature", "topK", "topP"];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

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

/**
 * `sampler` を読む（欄ごと無ければ `undefined` = 低層の既定 = greedy）。
 *
 * 値域は `generation/sampler.ts` の `assertSpec` と同じ — 配布形を読んだ時点で落とし、
 * 「1 回生成させてから初めて分かる」形を作らない。
 */
const parseSampler = (raw: unknown, where: string): SamplerSpec => {
  const record = isRecord(raw) ? raw : undefined;
  if (record === undefined) throw new Error(`${where}: 無い / オブジェクトでない`);
  assertAllowedKeys(record, SAMPLER_KEYS, where);
  return {
    temperature: readNumber(
      record,
      "temperature",
      where,
      (value) => Number.isFinite(value) && value >= 0,
      "0 以上の有限数でない",
    ),
    topK: readNumber(
      record,
      "topK",
      where,
      (value) => Number.isSafeInteger(value) && value >= 1,
      "1 以上の整数でない",
    ),
    topP: readNumber(
      record,
      "topP",
      where,
      (value) => Number.isFinite(value) && value > 0 && value <= 1,
      "0 < topP ≤ 1 の範囲にない",
    ),
  };
};

/**
 * manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。
 *
 * MUST: 3 つの数の関係まで見る — `chunkLength ≤ capacity ≤ maxPosition`。容量いっぱいの会話は
 * 位置 `capacity - 1` まで進むので、RoPE 表の行数を超える宣言は**長い会話でだけ**実行時に
 * 落ちる（焼く側の `gemma4_pipeline_config` が同じ関係を見るが、他人の配布形は検査を通って
 * いない前提で読む）。
 */
export const parseGemma4PipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): Gemma4PipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positiveInteger = (value: number): boolean => Number.isSafeInteger(value) && value >= 1;
  const chunkLength = readNumber(raw, "chunkLength", where, positiveInteger, "1 以上の整数でない");
  const maxPosition = readNumber(raw, "maxPosition", where, positiveInteger, "1 以上の整数でない");
  const capacity = readNumber(raw, "capacity", where, positiveInteger, "1 以上の整数でない");
  if (chunkLength > capacity) {
    throw new Error(
      `${where}: chunkLength ${chunkLength} が capacity ${capacity} を超えた` +
        `（1 chunk すら入らない容量）`,
    );
  }
  if (capacity > maxPosition) {
    throw new Error(
      `${where}: capacity ${capacity} が maxPosition ${maxPosition} を超えた` +
        `（容量いっぱいの会話が位置表の外を引く）`,
    );
  }
  if (!Object.hasOwn(raw, "sampler")) return { chunkLength, maxPosition, capacity };
  return {
    chunkLength,
    maxPosition,
    capacity,
    sampler: parseSampler(raw["sampler"], `${where}.sampler`),
  };
};

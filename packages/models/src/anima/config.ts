/**
 * `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、手書きの検査を全て parse 時に走らせる。
 *
 * MUST: 未知キーは fail loudly（`{ steps: 8, step: 8 }` のような綴り違いが黙って既定へ
 * 縮退すると、配布者の意図した既定と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * NOTE: 公開配布リポの既定ソース（{@link ANIMA_DEFAULT_SOURCE}）もここに置く。manifest から
 * 導ける値ではなく「どの manifest を取りに行くか」の側なので、配布形が持てない（ADR 0073）。
 */

import type { HubRepoRef } from "@karume/hub";

import { assertAcceptableResolution, type ImageSize } from "./resolution.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const ANIMA_PIPELINE_NAME = "anima";
export const ANIMA_PIPELINE_MAJOR = 1;

/**
 * 公開配布リポの既定ソース（ADR 0073 決定 1）。`ref` を省略した `AnimaPipeline.fromPretrained`
 * はここを読む。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。追従が要る利用者は
 * `{ ...ANIMA_DEFAULT_SOURCE, revision: "main" }` を明示的に選ぶ。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 — 手書き + 手順書ゲート）。
export const ANIMA_DEFAULT_SOURCE = {
  repo: "hdae/karume-anima-turbo",
  revision: "5aa15e4b79feb71662fc8520860976e8d9ea27cb",
} as const satisfies HubRepoRef;

const ROOT_KEYS: readonly string[] = ["scheduler", "defaults"];
const SCHEDULER_KEYS: readonly string[] = ["shift", "numTrainTimesteps"];
const DEFAULTS_KEYS: readonly string[] = [
  "steps",
  "guidanceScale",
  "resolution",
  "negativePrompt",
];
const RESOLUTION_KEYS: readonly string[] = ["width", "height"];

/** `FlowMatchEulerDiscreteScheduler` の構成（配布物ごとに変わりうるので manifest が持つ）。 */
type AnimaScheduler = {
  readonly shift: number;
  readonly numTrainTimesteps: number;
};

/** 配布者の推奨既定（`generate` の未指定欄を埋める）。 */
type AnimaDefaults = {
  readonly steps: number;
  readonly guidanceScale: number;
  readonly resolution: ImageSize;
  readonly negativePrompt?: string;
};

export type AnimaPipelineConfig = {
  readonly scheduler: AnimaScheduler;
  readonly defaults: AnimaDefaults;
};

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

const readRecord = (raw: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(raw)) throw new Error(`${where}: 無い / オブジェクトでない`);
  return raw;
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

const parseResolutionEntry = (raw: unknown, where: string): ImageSize => {
  const record = readRecord(raw, where);
  assertAllowedKeys(record, RESOLUTION_KEYS, where);
  const size: ImageSize = {
    width: readNumber(record, "width", where, Number.isInteger, "正の整数でない"),
    height: readNumber(record, "height", where, Number.isInteger, "正の整数でない"),
  };
  // 既定値も受理集合の内側でなければならない（配布時に外れていたら fromAssets で落ちる —
  // 「生成を 1 回走らせて初めて分かる」を作らない）。
  assertAcceptableResolution(size);
  return size;
};

const parseDefaults = (raw: unknown): AnimaDefaults => {
  const where = "pipelineConfig.defaults";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, DEFAULTS_KEYS, where);
  const steps = readNumber(
    record,
    "steps",
    where,
    (value) => Number.isInteger(value) && value >= 2,
    "2 以上の整数でない",
  );
  const guidanceScale = readNumber(
    record,
    "guidanceScale",
    where,
    Number.isFinite,
    "有限の数でない",
  );
  const resolution = parseResolutionEntry(record["resolution"], `${where}.resolution`);
  if (!Object.hasOwn(record, "negativePrompt")) {
    return { steps, guidanceScale, resolution };
  }
  const negativePrompt = record["negativePrompt"];
  if (typeof negativePrompt !== "string") {
    throw new Error(`${where}.negativePrompt: 文字列でない`);
  }
  return { steps, guidanceScale, resolution, negativePrompt };
};

const parseScheduler = (raw: unknown): AnimaScheduler => {
  const where = "pipelineConfig.scheduler";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, SCHEDULER_KEYS, where);
  return {
    shift: readNumber(
      record,
      "shift",
      where,
      (value) => Number.isFinite(value) && value > 0,
      "正の有限数でない",
    ),
    numTrainTimesteps: readNumber(
      record,
      "numTrainTimesteps",
      where,
      (value) => Number.isInteger(value) && value > 0,
      "正の整数でない",
    ),
  };
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseAnimaPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): AnimaPipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  return {
    scheduler: parseScheduler(Object.hasOwn(raw, "scheduler") ? raw["scheduler"] : undefined),
    defaults: parseDefaults(Object.hasOwn(raw, "defaults") ? raw["defaults"] : undefined),
  };
};

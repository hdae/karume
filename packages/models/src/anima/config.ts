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
 * NOTE: 公開配布リポの pin 定数（{@link ANIMA_CURRENT}）もここに置く。manifest から導ける
 * 値ではなく「どの manifest を取りに行くか」の側なので、配布形が持てない（ADR 0073）。
 */

import type { HubRepoRef } from "@karume/hub";

import { assertAcceptableResolution, type ImageSize } from "./resolution.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const ANIMA_PIPELINE_NAME = "anima";
export const ANIMA_PIPELINE_MAJOR = 1;

/**
 * `hdae/karume-anima`（公式モデルが**同居する**リポ — 再構造後は Turbo / Base / Aesthetic の
 * 3 変種・既定 = Turbo）を**このパッケージ版が検証した取得元**（pin 済み commit SHA —
 * ADR 0073）。1 リポ = 複数モデルなので、リポ参照だけでは 1 本に決まらない —
 * 既定以外を使うときは `fromPretrained(ANIMA_CURRENT, { model })` と綴る。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン**として渡す — 再現性を自分で
 * 固定したい場合は、この定数ではなく自分の `{ repo, revision }` を書く（`fromPretrained` に
 * 既定は無い）。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。main 追従が要る利用者は
 * `{ ...ANIMA_CURRENT, revision: "main" }` を明示的に選ぶ。
 *
 * NOTE: 旧 `ANIMA_TURBO_CURRENT`（hdae/karume-anima-turbo — LoRA 焼き込みの旧 turbo 専用
 * リポ）は廃止（2026-09-01 裁定・breaking）。公式 Turbo checkpoint（anima-turbo-v1.1）が
 * このリポの既定モデルとして同居する形に統合された。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 — 手書き + 手順書ゲート）。
export const ANIMA_CURRENT = {
  repo: "hdae/karume-anima",
  revision: "d4dbb35fcb5e7146a5845cfc657fe39aa169c788",
} as const satisfies HubRepoRef;

const ROOT_KEYS: readonly string[] = ["scheduler", "defaults"];
const SCHEDULER_KEYS: readonly string[] = ["type", "shift", "numTrainTimesteps"];
const DEFAULTS_KEYS: readonly string[] = [
  "steps",
  "guidanceScale",
  "resolution",
  "negativePrompt",
];
const RESOLUTION_KEYS: readonly string[] = ["width", "height"];

/**
 * denoise の更新則（サンプラ種別）。manifest の宣言が**既定**を決め、生成要求側の `sampler`
 * 席（`AnimaGenerateRequest`）が呼び出しごとに上書きする（再裁定 2026-08-25 — 配布既定は
 * 配布者の推奨、更新則の選択は利用者のノブ）。どちらの値も資産に依らず常に有効。
 *
 * - `"euler"` — `FlowMatchEulerDiscreteScheduler` の 1 次更新（`sampler.ts` の `cfgEulerStep`）。
 * - `"dpmpp-2m"` — DPM++ 2M（`src/generation/dpm-solver-multistep.ts`）。
 */
export type AnimaSamplerType = "euler" | "dpmpp-2m";

const SAMPLER_TYPES: readonly AnimaSamplerType[] = ["euler", "dpmpp-2m"];

/**
 * `scheduler.type` 省略時の値。
 *
 * MUST: 既定は `"euler"` — `type` は additive に足した optional 席（ADR 0038 §1）なので、
 * **この欄を持たない既存の配布 manifest は 1 バイトも変えずに従来の更新則のまま**でなければ
 * ならない（既に配られたリポは migrate できない）。
 */
const DEFAULT_SAMPLER_TYPE: AnimaSamplerType = "euler";

/**
 * scheduler の構成（配布物ごとに変わりうるので manifest が持つ）。
 * `type` は省略可（既定 {@link DEFAULT_SAMPLER_TYPE}）で、parse 済みの値は**常に埋まっている**
 * — 既定の解決はこの 1 箇所に閉じ、消費側（pipeline）は分岐を持たない。
 */
type AnimaScheduler = {
  readonly type: AnimaSamplerType;
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

/**
 * サンプラ種別の綴りを**期待と実際を並べて**検査する（manifest 側 / 生成要求側の共通口）。
 *
 * MUST: 語彙の正本は {@link SAMPLER_TYPES} 1 本 — 綴り違いが黙って既定へ縮退すると、宣言・
 * 指定した更新則と実行が食い違ったまま気づけない。`where` は綴りの出所（`scheduler.type` /
 * request の `sampler`）を指す。
 */
export const assertAnimaSamplerType = (value: unknown, where: string): AnimaSamplerType => {
  const accepted = typeof value === "string"
    ? SAMPLER_TYPES.find((candidate) => candidate === value)
    : undefined;
  if (accepted === undefined) {
    throw new Error(
      `${where}: 期待 ${SAMPLER_TYPES.map((name) => `'${name}'`).join(" / ")}` +
        `（実際 ${typeof value === "string" ? `'${value}'` : String(value)}）`,
    );
  }
  return accepted;
};

/** `scheduler.type` を読む（欄ごと無ければ既定 — 未知値は {@link assertAnimaSamplerType} が落とす）。 */
const parseSamplerType = (
  record: Record<string, unknown>,
  where: string,
): AnimaSamplerType =>
  Object.hasOwn(record, "type")
    ? assertAnimaSamplerType(record["type"], `${where}.type`)
    : DEFAULT_SAMPLER_TYPE;

const parseScheduler = (raw: unknown): AnimaScheduler => {
  const where = "pipelineConfig.scheduler";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, SCHEDULER_KEYS, where);
  return {
    type: parseSamplerType(record, where),
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

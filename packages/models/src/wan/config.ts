/**
 * Wan2.1 の `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、焼く側（recipe `tools/export-recipes/wan/distribution.py` の
 * `WAN_PIPELINE_CONFIG`）の欄名とは recipe 側のテストが突き合わせる。
 *
 * 宣言するのは生成の既定だけ（ADR 0118 決定 5 — 製品の既定は manifest の `pipelineConfig.scheduler` で
 * 起こし、視認の A/B で変えるかを決める）:
 *
 * - `scheduler.shift` — flow matching の shift（参照の設定 3.0）
 * - `defaults.steps` / `defaults.guidance` — denoise の step 数と CFG の強さ（参照の設定 50 / 5.0）
 *
 * UniPC の構造（次数・bh2・`num_train_timesteps`）は宣言の席を持たない — 移植が実装している分岐は
 * 1 つきりで（`scheduler.ts` の `WAN_UNIPC_CONFIG`）、宣言できるようにすると検証していない組み合わせを
 * 配布側が選べてしまう。
 *
 * MUST: 未知キーは fail loudly（綴り違いが黙って既定へ縮退すると、配布者の意図した既定と実行が
 * 食い違ったまま気づけない）。値域は生成の要求の門（`family.ts` の `planWanGeneration`）と同じで、
 * 外れた宣言は**資産の齟齬**として素の `Error` で落とす（入力起因ではない — ADR 0107 決定 2）。
 * 既定の steps × shift の組で UniPC の σ 列が組めることもここで見る — 見ないと、要求が何も渡さない
 * `generate` が `planWanGeneration` で入力起因（`ModelInputError`）として落ちる。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 */

import { assertAllowedKeys, readNumber, readRecord } from "../config/readers.ts";
import { WAN_UNIPC_CONFIG, wanUniPcSchedule } from "./scheduler.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const WAN_PIPELINE_NAME = "wan";
export const WAN_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = ["scheduler", "defaults"];
const SCHEDULER_KEYS: readonly string[] = ["shift"];
const DEFAULTS_KEYS: readonly string[] = ["steps", "guidance"];

/** 配布者の推奨既定（`generate` の未指定欄を埋める）。 */
export type WanPipelineConfig = {
  readonly scheduler: { readonly shift: number };
  readonly defaults: { readonly steps: number; readonly guidance: number };
};

/** 欄の集まりを読む（欠落・オブジェクトでない・未知キーを落とす）。 */
const readSection = (
  raw: Readonly<Record<string, unknown>>,
  key: string,
  allowed: readonly string[],
): Record<string, unknown> => {
  const where = `pipelineConfig.${key}`;
  const record = readRecord(Object.hasOwn(raw, key) ? raw[key] : undefined, where);
  assertAllowedKeys(record, allowed, where);
  return record;
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseWanPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): WanPipelineConfig => {
  assertAllowedKeys(raw, ROOT_KEYS, "pipelineConfig");
  const scheduler = readSection(raw, "scheduler", SCHEDULER_KEYS);
  const defaults = readSection(raw, "defaults", DEFAULTS_KEYS);
  const shift = readNumber(
    scheduler,
    "shift",
    "pipelineConfig.scheduler",
    (value) => Number.isFinite(value) && value > 0,
    "正の有限数でない",
  );
  const steps = readNumber(
    defaults,
    "steps",
    "pipelineConfig.defaults",
    (value) => Number.isInteger(value) && value >= 1,
    "1 以上の整数でない",
  );
  const guidance = readNumber(
    defaults,
    "guidance",
    "pipelineConfig.defaults",
    // CFG は f32 で掛ける（要求の門と同じ — f64 で有限でも f32 で Infinity なら合成が NaN）。
    (value) => Number.isFinite(Math.fround(value)) && value >= 1,
    "1 以上で f32 に収まる有限の数でない",
  );
  try {
    wanUniPcSchedule(steps, shift, WAN_UNIPC_CONFIG.numTrainTimesteps);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(
      `pipelineConfig: 既定の defaults.steps ${steps} と scheduler.shift ${shift} の組では UniPC の ` +
        `σ 列が組めない（${reason}）`,
      { cause: error },
    );
  }
  return { scheduler: { shift }, defaults: { steps, guidance } };
};

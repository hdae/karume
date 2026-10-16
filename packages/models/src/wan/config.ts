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
 * - `scheduler.shift` — flow matching の shift（参照の設定 — Wan2.1 は 3.0・Wan2.2 は 5.0）
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

import type { HubRepoRef } from "@karume/hub";
import { assertAllowedKeys, readNumber, readRecord } from "../config/readers.ts";
import { WAN_UNIPC_CONFIG, wanUniPcSchedule } from "./scheduler.ts";

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const WAN_PIPELINE_NAME = "wan";
export const WAN_PIPELINE_MAJOR = 1;

/**
 * Wan2.2 TI2V の `pipeline` の契約名と major（`WanTi2vPipeline` — ADR 0121 決定 10）。`pipelineConfig` の
 * スキーマは Wan2.1 と同じ 1 本（{@link parseWanPipelineConfig}）。
 */
export const WAN_TI2V_PIPELINE_NAME = "wan-ti2v";
export const WAN_TI2V_PIPELINE_MAJOR = 1;

/**
 * Wan ファミリの**公開配布リポ対応表**（ADR 0092 — 家族 1 つにつき 1 表・**既定の席は無い**）。
 * 値は**このパッケージ版が検証した取得元**（pin 済み commit SHA — ADR 0073）。
 *
 * キーは HF リポ名の basename から `karume-` を落とした綴り（`"karume-" + key` がリポ名の
 * basename に戻る — この不変条件は `tests/sources_test.ts` の門が見る）。世代は別リポ（ADR 0092 決定 1）:
 *
 * - `"wan2.2"` = `hdae/karume-wan2.2`（Wan2.2 TI2V 5B — `WanTi2vPipeline` が読む・umT5 の text_encoder は
 *   `karume-umt5-xxl` を越境参照する）
 *
 * Wan2.1（`karume-wan2.1`）は HF に公開しない（ADR 0118 追記（2026-10-15））ので**キーを持たない** — 公開リポの
 * 無いキーは 404 にしかならない（ADR 0073 決定 1 の理由）。`WanPipeline` には手元の配布形を渡す。
 *
 * **パッケージ版に合わせて自動追従したい場合のオプトイン**として渡す — 再現性を自分で
 * 固定したい場合は、この表ではなく自分の `{ repo, revision }` を書く（`fromPretrained` に
 * 既定は無い）。
 *
 * MUST: revision は commit SHA で固定する — ブランチ・タグは配布側で付け替えられるので、
 * 公開済みのこのパッケージが読むバイト列がネットワーク側の都合で黙って変わる（回復不能側の
 * 事故）。SHA 指定は revision 解決要求そのものを消すため、完全キャッシュ時のオフライン起動も
 * 同時に成立する（ADR 0038）。main 追従が要る利用者は
 * `{ ...WAN_SOURCES["wan2.2"], revision: "main" }` を明示的に選ぶ。
 */
// NOTE: revision はリリース手順書（docs/release-runbook.md）§3 で、アップロード後の main の
// SHA に更新する（ADR 0073 決定 3 の維持義務を継承 — 手書き + 手順書ゲート）。
export const WAN_SOURCES = {
  "wan2.2": {
    repo: "hdae/karume-wan2.2",
    revision: "791c3e92dedeb7988f6adc739a5139fbfd876823",
  },
} as const satisfies Record<string, HubRepoRef>;

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
    // 上限は要求の門と同じ `num_train_timesteps`（`scheduler.ts` の `wanUniPcSchedule` の MUST）。
    (value) => Number.isInteger(value) && value >= 1 && value <= WAN_UNIPC_CONFIG.numTrainTimesteps,
    `1 以上 ${WAN_UNIPC_CONFIG.numTrainTimesteps} 以下の整数でない`,
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

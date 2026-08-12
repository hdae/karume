/**
 * `pipelineConfig` のスキーマ検証（ADR 0038 §1 — スキーマは各パイプライン実装が所有・検証）。
 *
 * hub は `pipelineConfig` を素通しする（禁止キーの一掃と規模上限だけを見る）。したがって
 * **形の正本はこのモジュール**で、手書きの検査を全て parse 時に走らせる。
 *
 * MUST: 未知キーは fail loudly（`{ steps: 40, step: 40 }` のような綴り違いが黙って既定へ
 * 縮退すると、配布者の意図した既定と実行が食い違ったまま気づけない）。
 * MUST: マップは `Object.hasOwn` 経由でのみ引く（横断不変条件）。
 *
 * ## MUST: モデル固有の数はここ（= manifest）が正本で、TS 側に定数を置かない
 *
 * Irodori は「グラフの宣言長」と「ホストの式」が噛み合って初めて正しく動く（条件 state の
 * Tmax・speaker 行数・latent 幅・t_embed 幅・S の clamp 範囲）。TS に直書きすると、重みを
 * 差し替えたときにホストだけが古い数を持ったまま **shape は合う**形で沈黙誤値になる。
 * 実グラフとの突合は {@link IrodoriPipeline.fromAssets} が資産に対して行う。
 *
 * ## MUST: 対応外の CFG モードは**値を保持せず**パース時に拒否する（ADR 0047 決定 1）
 *
 * この実装の uncond は「cond の state + 該当区間のマスク全 False」で、これが上流と一致するのは
 * `speaker_uncond_mode="mask"` かつ `cfg_guidance_mode="independent"` のときだけ。他のモード
 * （`"noise"` / `joint` / `alternating`）は同値が成り立たないので、分岐を持つのではなく
 * **受理しない**（型としても "mask" / "independent" しか表せない）。
 */

/** `pipeline` の契約名と、この実装が受け付ける major（ADR 0038 §1）。 */
export const IRODORI_PIPELINE_NAME = "irodori";
export const IRODORI_PIPELINE_MAJOR = 1;

const ROOT_KEYS: readonly string[] = [
  "maxTextLen",
  "maxCaptionLen",
  "speakerRows",
  "ditSymMax",
  "frameRate",
  "latentDim",
  "speakerPatchSize",
  "speakerDim",
  "textDim",
  "captionDim",
  "timestepEmbedDim",
  "steps",
  "initScale",
  "cfgMinT",
  "cfgMaxT",
  "cfgScales",
  "minSeconds",
  "maxSeconds",
  "speakerUncondMode",
  "cfgGuidanceMode",
];

const CFG_SCALE_KEYS: readonly string[] = ["text", "speaker", "caption"];

/** この実装が受理する唯一の uncond の作り方（ADR 0047 決定 1）。 */
const SPEAKER_UNCOND_MODE = "mask";
/** この実装が受理する唯一の CFG 合成（`v_cond + Σ s_k(v_cond − v_k)`）。 */
const CFG_GUIDANCE_MODE = "independent";

/** CFG の強さ（条件ごと）。合成順は text → speaker → caption で固定（ADR 0047）。 */
export type IrodoriCfgScales = {
  readonly text: number;
  readonly speaker: number;
  readonly caption: number;
};

export type IrodoriPipelineConfig = {
  /** text 側 token 列の上限（BOS 込み）。`dit` の `text_state` の宣言長でもある。 */
  readonly maxTextLen: number;
  /** caption 側 token 列の上限（BOS 込み）。`dit` の `caption_state` の宣言長でもある。 */
  readonly maxCaptionLen: number;
  /** `dit` の `speaker_state` の宣言行数（参照 latent の上限 + 平均トークン 1 本）。 */
  readonly speakerRows: number;
  /** `dit` の記号次元 S の上限。 */
  readonly ditSymMax: number;
  /** codec のフレームレート（Hz）— 秒 ↔ latent フレームの換算。 */
  readonly frameRate: number;
  /** patch 済み latent の幅（`x_t` の最終次元）。 */
  readonly latentDim: number;
  /** 参照 latent を speaker エンコーダへ渡すときの時間方向 patch 幅。 */
  readonly speakerPatchSize: number;
  readonly speakerDim: number;
  readonly textDim: number;
  readonly captionDim: number;
  /** `t_embed` の幅（前半 cos / 後半 sin）。 */
  readonly timestepEmbedDim: number;
  /** Euler の step 数。 */
  readonly steps: number;
  /** t スケジュールの初期倍率（`t_i = initScale·(1 − i/steps)`）。 */
  readonly initScale: number;
  /** CFG を掛ける t の下限 / 上限（両端を含む）。 */
  readonly cfgMinT: number;
  readonly cfgMaxT: number;
  readonly cfgScales: IrodoriCfgScales;
  /** S の clamp 範囲を決める秒数。 */
  readonly minSeconds: number;
  readonly maxSeconds: number;
  /** ADR 0047 が受理する唯一の値。**分岐用ではなく、配布形の宣言を検査するために持つ**。 */
  readonly speakerUncondMode: typeof SPEAKER_UNCOND_MODE;
  readonly cfgGuidanceMode: typeof CFG_GUIDANCE_MODE;
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

const isPositiveInteger = (value: number): boolean => Number.isInteger(value) && value > 0;
const isPositiveFinite = (value: number): boolean => Number.isFinite(value) && value > 0;
const isNonNegativeFinite = (value: number): boolean => Number.isFinite(value) && value >= 0;

/** 受理集合が 1 値しかない欄（ADR 0047 決定 1）。綴り違いも対応外も同じ文言で落とす。 */
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

const parseCfgScales = (raw: unknown): IrodoriCfgScales => {
  const where = "pipelineConfig.cfgScales";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, CFG_SCALE_KEYS, where);
  // 0 は「その条件の CFG を回さない」を意味する正規の値（上流 `has_*_cfg = scale > 0`）。
  const requirement = "非負の有限数でない";
  return {
    text: readNumber(record, "text", where, isNonNegativeFinite, requirement),
    speaker: readNumber(record, "speaker", where, isNonNegativeFinite, requirement),
    caption: readNumber(record, "caption", where, isNonNegativeFinite, requirement),
  };
};

/** manifest の `pipelineConfig`（hub が素通しした生の値）を検査して読む。 */
export const parseIrodoriPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): IrodoriPipelineConfig => {
  const where = "pipelineConfig";
  assertAllowedKeys(raw, ROOT_KEYS, where);
  const positive = "正の整数でない";
  // token 列は BOS を必ず 1 本置く（`text/tokenizer.ts`）ので、本文の予算が残る 2 以上が要る。
  const withBos = (value: number): boolean => Number.isInteger(value) && value >= 2;
  const config: IrodoriPipelineConfig = {
    maxTextLen: readNumber(raw, "maxTextLen", where, withBos, "2 以上の整数でない"),
    maxCaptionLen: readNumber(raw, "maxCaptionLen", where, withBos, "2 以上の整数でない"),
    speakerRows: readNumber(raw, "speakerRows", where, isPositiveInteger, positive),
    ditSymMax: readNumber(raw, "ditSymMax", where, isPositiveInteger, positive),
    frameRate: readNumber(raw, "frameRate", where, isPositiveInteger, positive),
    latentDim: readNumber(raw, "latentDim", where, isPositiveInteger, positive),
    speakerPatchSize: readNumber(raw, "speakerPatchSize", where, isPositiveInteger, positive),
    speakerDim: readNumber(raw, "speakerDim", where, isPositiveInteger, positive),
    textDim: readNumber(raw, "textDim", where, isPositiveInteger, positive),
    captionDim: readNumber(raw, "captionDim", where, isPositiveInteger, positive),
    // 前半 cos / 後半 sin に割るので奇数幅は組めない。
    timestepEmbedDim: readNumber(
      raw,
      "timestepEmbedDim",
      where,
      (value) => isPositiveInteger(value) && value % 2 === 0,
      "正の偶数でない",
    ),
    steps: readNumber(raw, "steps", where, isPositiveInteger, positive),
    initScale: readNumber(raw, "initScale", where, isPositiveFinite, "正の有限数でない"),
    cfgMinT: readNumber(raw, "cfgMinT", where, Number.isFinite, "有限の数でない"),
    cfgMaxT: readNumber(raw, "cfgMaxT", where, Number.isFinite, "有限の数でない"),
    cfgScales: parseCfgScales(Object.hasOwn(raw, "cfgScales") ? raw["cfgScales"] : undefined),
    minSeconds: readNumber(raw, "minSeconds", where, isPositiveFinite, "正の有限数でない"),
    maxSeconds: readNumber(raw, "maxSeconds", where, isPositiveFinite, "正の有限数でない"),
    speakerUncondMode: readOnly(
      raw,
      "speakerUncondMode",
      where,
      SPEAKER_UNCOND_MODE,
      "uncond を『cond の state + 該当区間のマスク全 False』で表す（ADR 0047 決定 1）",
    ),
    cfgGuidanceMode: readOnly(
      raw,
      "cfgGuidanceMode",
      where,
      CFG_GUIDANCE_MODE,
      "CFG の合成が `v_cond + Σ s_k(v_cond − v_k)` の形に限られる（ADR 0047 決定 1）",
    ),
  };
  // 区間が空 / 逆順の宣言は「CFG が 1 度も掛からない」形で沈黙する（forward 数だけが減る）。
  if (config.cfgMinT > config.cfgMaxT) {
    throw new Error(
      `${where}: cfgMinT ${config.cfgMinT} が cfgMaxT ${config.cfgMaxT} より大きい（CFG が 1 度も掛からない）`,
    );
  }
  if (config.minSeconds > config.maxSeconds) {
    throw new Error(
      `${where}: minSeconds ${config.minSeconds} が maxSeconds ${config.maxSeconds} より大きい`,
    );
  }
  return config;
};

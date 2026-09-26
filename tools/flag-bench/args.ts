/**
 * flag-bench の CLI 引数と `--set <label>=<json>` の解析（**純関数だけ** — GPU も I/O も触らない）。
 *
 * MUST: 未知のオプション・未知の上書きキーは落とす。打ち間違えたノブが黙って既定で走ると、
 * JSONL に残る条件と実際に測った条件が食い違う（`tools/mtp-bench` と同じ原則）。
 * MUST: 全モジュール副作用ゼロ（CLAUDE.md）。
 */

import type { Gemma4PipelineOptions } from "../../packages/models/gemma.ts";
import type {
  LinearGemvReduce,
  RmsNormReduce,
  StateAttentionReduce,
} from "../../packages/runtime/mod.ts";

/**
 * `--set` が上書きできるキー（pipeline が Session / PLE の実行形として受けるノブ）。
 *
 * `gpu` / `onRunDiagnostics` は台本の持ち物なので載せない。`speculative` も載せない — QAT は
 * 受けない（`Gemma4QatPipelineOptions`）うえ、投機の取り分は `tools/mtp-bench` が測る別の問い。
 */
export const FLAG_KEYS = [
  "linearGemvReduce",
  "fuseRmsNormAdd",
  "fuseLinearStaticQuantize",
  "packedStaticQuantize",
  "stateAttentionReduce",
  "rmsNormReduce",
  "submitPolicy",
  "planBackingBudgetBytes",
  "linearGemvRowsThreadTarget",
  "chunkLength",
  "chunkBuckets",
  "pleResidency",
  "maxResidentPleBytes",
] as const;
export type FlagKey = (typeof FLAG_KEYS)[number];

/** 1 つの set が pipeline へ渡す上書き（通常 / QAT 共通の部分集合）。 */
export type FlagOverrides = Pick<Gemma4PipelineOptions, FlagKey>;
type MutableOverrides = { -readonly [K in FlagKey]?: Gemma4PipelineOptions[K] };

/**
 * 文字列ノブの値域（`Record<…, true>` なのは、型の union に値が増えた日に**型検査で**気づく
 * ため — 配列で写すと足し忘れが黙って通る）。値の意味と組合せの門は library 側 1 箇所で、
 * ここが見るのは「型として何か」だけである。
 */
const LINEAR_GEMV_REDUCES: Readonly<Record<LinearGemvReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-subgroup32": true,
};
const STATE_ATTENTION_REDUCES: Readonly<Record<StateAttentionReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-fused": true,
};
const RMS_NORM_REDUCES: Readonly<Record<RmsNormReduce, true>> = {
  workgroup: true,
  subgroup32: true,
};
const PLE_RESIDENCIES: Readonly<
  Record<NonNullable<Gemma4PipelineOptions["pleResidency"]>, true>
> = { host: true, gpu: true };

const SUBMIT_POLICY_KEYS = [
  "timeBudgetMs",
  "initialChunkSize",
  "minChunkSize",
  "maxChunkSize",
] as const;

const oneOf = <T extends string>(
  domain: Readonly<Record<T, true>>,
  value: unknown,
  where: string,
): T => {
  const isMember = (candidate: string): candidate is T => Object.hasOwn(domain, candidate);
  if (typeof value !== "string" || !isMember(value)) {
    throw new Error(
      `${where}: ${JSON.stringify(value)} は受けない（既知: ${Object.keys(domain).join(" / ")}）`,
    );
  }
  return value;
};

const bool = (value: unknown, where: string): boolean => {
  if (typeof value !== "boolean") throw new Error(`${where}: boolean でない`);
  return value;
};

/** 整数ノブ。値域（正か非負か・上限）は library の門 1 箇所に任せ、ここは型だけを見る。 */
const integer = (value: unknown, where: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new Error(`${where}: 安全な整数でない（${JSON.stringify(value)}）`);
  }
  return value;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const submitPolicy = (
  value: unknown,
  where: string,
): NonNullable<Gemma4PipelineOptions["submitPolicy"]> => {
  // 部分指定は受けない — `SubmitPolicy` は 4 欄が揃って 1 つの政策で、欠けた欄を既定で埋めると
  // 記録に残る値と効いた値が食い違う。
  if (!isRecord(value)) throw new Error(`${where}: オブジェクトでない`);
  for (const key of Object.keys(value)) {
    if (!(SUBMIT_POLICY_KEYS as readonly string[]).includes(key)) {
      throw new Error(`${where}: 未知の欄 ${key}（既知: ${SUBMIT_POLICY_KEYS.join(" / ")}）`);
    }
  }
  const timeBudgetMs = value.timeBudgetMs;
  if (typeof timeBudgetMs !== "number" || !Number.isFinite(timeBudgetMs)) {
    throw new Error(`${where}.timeBudgetMs: 有限の数でない`);
  }
  return {
    timeBudgetMs,
    initialChunkSize: integer(value.initialChunkSize, `${where}.initialChunkSize`),
    minChunkSize: integer(value.minChunkSize, `${where}.minChunkSize`),
    maxChunkSize: integer(value.maxChunkSize, `${where}.maxChunkSize`),
  };
};

/**
 * `--set` の JSON（`unknown` 境界）を上書きへ写す。
 *
 * 空の `{}` は「上書き無し」= manifest の quant 宣言と家族既定そのまま（基準に置く形）。
 */
export const parseOverrides = (raw: unknown, where: string): FlagOverrides => {
  if (!isRecord(raw)) throw new Error(`${where}: JSON オブジェクトでない`);
  const out: MutableOverrides = {};
  for (const [key, value] of Object.entries(raw)) {
    const at = `${where}.${key}`;
    switch (key) {
      case "linearGemvReduce":
        out.linearGemvReduce = oneOf(LINEAR_GEMV_REDUCES, value, at);
        break;
      case "fuseRmsNormAdd":
        out.fuseRmsNormAdd = bool(value, at);
        break;
      case "fuseLinearStaticQuantize":
        out.fuseLinearStaticQuantize = bool(value, at);
        break;
      case "packedStaticQuantize":
        out.packedStaticQuantize = bool(value, at);
        break;
      case "stateAttentionReduce":
        out.stateAttentionReduce = oneOf(STATE_ATTENTION_REDUCES, value, at);
        break;
      case "rmsNormReduce":
        out.rmsNormReduce = oneOf(RMS_NORM_REDUCES, value, at);
        break;
      case "submitPolicy":
        out.submitPolicy = submitPolicy(value, at);
        break;
      case "planBackingBudgetBytes":
        out.planBackingBudgetBytes = integer(value, at);
        break;
      case "linearGemvRowsThreadTarget":
        out.linearGemvRowsThreadTarget = integer(value, at);
        break;
      case "chunkLength":
        out.chunkLength = integer(value, at);
        break;
      case "chunkBuckets":
        if (!Array.isArray(value)) throw new Error(`${at}: 配列でない`);
        out.chunkBuckets = value.map((rows, index) => integer(rows, `${at}[${index}]`));
        break;
      case "pleResidency":
        out.pleResidency = oneOf(PLE_RESIDENCIES, value, at);
        break;
      case "maxResidentPleBytes":
        out.maxResidentPleBytes = integer(value, at);
        break;
      default:
        throw new Error(`${at}: 未知のキー（既知: ${FLAG_KEYS.join(" / ")}）`);
    }
  }
  return out;
};

/**
 * その set の device に 32 レーン subgroup が要るか。
 *
 * 台本が device を持つ（pipeline は渡された device をそのまま使う）ので、要求するのも台本の
 * 責務である。持たないアダプタ（例: Arc B570）では `acquireGpu` が `GpuFeatureError` で落ちる —
 * 黙って非 subgroup 変種へ戻さない。
 */
export const needsSubgroups = (overrides: FlagOverrides): boolean =>
  overrides.linearGemvReduce === "parallel-subgroup32" ||
  overrides.rmsNormReduce === "subgroup32";

/** 比べる 1 構成（`--set` 1 本）。 */
export type FlagSet = { readonly label: string; readonly overrides: FlagOverrides };

/** ラベルは JSONL の突合キーなので、空白・記号の揺れで別物に割れない綴りに限る。 */
const LABEL = /^[A-Za-z0-9][A-Za-z0-9._+-]*$/;

/** `--set <label>=<json>` 1 本を解析する（`=` の最初の 1 個で割る — JSON 側の `=` は残す）。 */
export const parseSet = (spec: string): FlagSet => {
  const cut = spec.indexOf("=");
  if (cut < 0) throw new Error(`--set ${spec}: <label>=<json> の形でない`);
  const label = spec.slice(0, cut);
  if (!LABEL.test(label)) {
    throw new Error(`--set ${spec}: ラベル '${label}' は ${LABEL.source} に合わない`);
  }
  let raw: unknown;
  try {
    raw = JSON.parse(spec.slice(cut + 1));
  } catch (error) {
    throw new Error(`--set ${label}: JSON として読めない`, { cause: error });
  }
  return { label, overrides: parseOverrides(raw, `--set ${label}`) };
};

export type Family = "normal" | "qat";
export type ModelName = "e2b" | "e4b";

/** 解析済みの CLI（既定値は埋めた後の実効値）。 */
export type BenchArgs = {
  readonly source: string;
  readonly family: Family;
  readonly model: ModelName;
  readonly quant: string;
  /** 1 本目が基準（相対値の分母）。 */
  readonly sets: readonly FlagSet[];
  readonly rounds: number;
  readonly newTokens: number;
  readonly capacity: number;
  readonly gpuTiming: boolean;
  readonly out: string;
};

export const DEFAULT_ROUNDS = 2;
export const DEFAULT_NEW_TOKENS = 96;
export const DEFAULT_CAPACITY = 4096;

export const USAGE = "--source <mirror dir> --family <normal|qat> --model <e2b|e4b>" +
  " --quant <name> --set <label>=<json> [--set ...] [--rounds 2] [--new-tokens 96]" +
  " [--capacity 4096] [--gpu-timing] --out <file.jsonl>";

const VALUED = new Set([
  "source",
  "family",
  "model",
  "quant",
  "set",
  "rounds",
  "new-tokens",
  "capacity",
  "out",
]);
const SWITCHES = new Set(["gpu-timing"]);

/** 正の安全な整数（`--rounds` 等）。 */
const positive = (key: string, raw: string | undefined, fallback: number): number => {
  if (raw === undefined) return fallback;
  if (!/^[1-9]\d*$/.test(raw) || !Number.isSafeInteger(Number(raw))) {
    throw new Error(`--${key} ${raw} が正の整数でない`);
  }
  return Number(raw);
};

/**
 * argv を解析する。
 *
 * MUST: 次のフラグを値として食わない。MUST: `--set` 以外の重複は落とす（後勝ちにすると、
 * どちらで測ったかが JSONL から読めない）。
 */
export const parseArgs = (argv: readonly string[]): BenchArgs => {
  const values = new Map<string, string>();
  const setSpecs: string[] = [];
  const switches = new Set<string>();
  for (let at = 0; at < argv.length;) {
    const key = argv[at];
    if (!key.startsWith("--")) {
      throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
    }
    const name = key.slice(2);
    if (SWITCHES.has(name)) {
      if (switches.has(name)) throw new Error(`${key} が 2 回ある`);
      switches.add(name);
      at += 1;
      continue;
    }
    if (!VALUED.has(name)) throw new Error(`未知のオプション ${key}（使い方: ${USAGE}）`);
    const value = argv[at + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`引数 ${key} が --key value の対になっていない（使い方: ${USAGE}）`);
    }
    if (name === "set") {
      setSpecs.push(value);
    } else {
      if (values.has(name)) throw new Error(`${key} が 2 回ある`);
      values.set(name, value);
    }
    at += 2;
  }
  const required = (name: string): string => {
    const value = values.get(name);
    if (value === undefined) throw new Error(`--${name} は必須（使い方: ${USAGE}）`);
    return value;
  };
  const family = required("family");
  if (family !== "normal" && family !== "qat") {
    throw new Error(`--family ${family} は受けない（normal / qat）`);
  }
  const model = required("model");
  if (model !== "e2b" && model !== "e4b") {
    throw new Error(`--model ${model} は受けない（e2b / e4b）`);
  }
  if (setSpecs.length === 0) throw new Error(`--set が 1 本も無い（使い方: ${USAGE}）`);
  const sets = setSpecs.map(parseSet);
  const labels = new Set<string>();
  for (const { label } of sets) {
    if (labels.has(label)) throw new Error(`--set のラベル ${label} が重複している`);
    labels.add(label);
  }
  const newTokens = positive("new-tokens", values.get("new-tokens"), DEFAULT_NEW_TOKENS);
  // decode ms/token は (完了 − 先頭 token) / (配送 − 1) なので、2 token 未満では分母が無い。
  if (newTokens < 2) throw new Error(`--new-tokens ${newTokens}: decode を測るには 2 以上が要る`);
  return {
    source: required("source"),
    family,
    model,
    quant: required("quant"),
    sets,
    rounds: positive("rounds", values.get("rounds"), DEFAULT_ROUNDS),
    newTokens,
    capacity: positive("capacity", values.get("capacity"), DEFAULT_CAPACITY),
    gpuTiming: switches.has("gpu-timing"),
    out: required("out"),
  };
};

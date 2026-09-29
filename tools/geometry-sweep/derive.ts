/**
 * 掃引の記録（`karume-geometry-sweep/2` — `report.ts`）から、adapter 1 種ぶんの**幾何プロファイル**
 * （runtime の `src/kernels/geometry-profile.ts` の `GeometryProfile`）を導く純関数（perf-ledger K-71）。
 * Deno の API に依らない — CLI（`profile.ts` = `main.ts profile`）とブラウザのページ（`tools/gpu-lab` の
 * プロファイルタブ）が同じ規則で表を作る（2 本が別々に持つと、ページで作った表と CLI の生成物が黙ってずれる）。
 *
 * 幾何の選択は runtime では「shape × adapter の静的な表」で、実行時オートチューンは禁止のまま
 * （ADR 0022 決定 3 の MUST）。この道具は**明示のチューニング（掃引）で測った結果をソースへ焼き込む
 * 側**で、runtime は生成物を adapter の (vendor, architecture, description) で 1 本選ぶだけ — 実行時には
 * 測らない。`match` を省いた表（`--opt-in`）は自動選択されない注入専用の表になる。利用者が自分の掃引から
 * 作った表は `acquireGpu({ geometryProfile })` で注入もできる（{@link profileJson}）。
 *
 * ## 規則の導出
 *
 * プロファイルの欄 1 つ（= クラス）ごとに、そのクラスの**全ケース**で「出力が既定と一致
 * （`identicalToDefault`）」かつ「既定比（`speedupVsDefault`）≥ `--min-speedup`」を満たす幾何のうち、
 * ケース間の幾何平均が最大のものを採る。無ければ既定（掃引の既定の行の幾何）を書く。
 *
 * - 全ケースで勝つことを要求するのは、1 ケースの大勝ちで別のケースの退行を覆い隠さないため
 *   （クラスの幾何はそのクラスの全 shape に効く）。
 * - 平均を幾何平均にするのは、比の平均だから（×2 と ×0.5 の算術平均は ×1.25 になり、勝ち負けが
 *   相殺されない）。
 * - 出力の一致を門にするのは、f32 の K 縮約順が幾何に依存しないこと（ADR 0022）の実測上の裏付けが
 *   ここにしか無いから — 不一致の幾何はどれほど速くても候補にしない。
 * - 同じ (ケース, 幾何) を複数の掃引が測っていれば、比は観測の幾何平均・一致と失敗は全観測で見る
 *   （quick と full を重ねる用途 — 1 度でも不一致・失敗した幾何は採らない）。
 * - 掃引の既定の行の幾何が今の runtime の既定と違えば生成しない（fail loudly）。既定比は既定の行に
 *   対する比なので、土台が今の既定と別物なら「既定より速い」が今の runtime では成り立たない。
 *
 * 材料の門: 掃引ごとに、既定の再測定比（`cases[].defaultRepeat.driftRatio`）が
 * `DEFAULT_DRIFT_RANGE`（report.ts — 0.9〜1.1）の外か、再測定が失敗 / 無いケースは、その掃引の比の観測から
 * 外す（{@link deriveProfile}）。比の土台（既定の値）がケースの途中で動いた疑いがあり、どの幾何の既定比も
 * 信用できないから。外すのは比だけで、出力の不一致と失敗はそのケースでも判定に効く（出力の一致は正しさの
 * 門で熱に依らない）。外したケースは採否の行に全部残す。
 *
 * 入力の門（ADR 0115 §4）: GPU の timestamp で測った掃引（`gpuTiming.unit` が `ns` か
 * `deno-raw-tick`、かつ `gpuTiming.quantized` が false）だけを受ける。壁時計と 100 µs 丸めは
 * 幾何どうしの比を 1 へ縮めるので、勝ち負けの判定に使えない。
 *
 * クラスの境界は runtime の既定の表と同じ: linear / matmul / bmm の行数 ≤ 64 / ≤ 512 / それ以上
 * （`GEMM_ROWS_BUCKETS`）、融合 attention の ①QK / ③PV、conv2d の m タイル 64 / 32 行
 * （既定の行の tileM）、i8a8 の linear / ①QK / ③PV。掃引にケースが無いクラスは runtime の既定を
 * そのまま書く（理由を生成物のコメントに残す）。
 */
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_ROWS_BUCKETS,
  type GemmGeometry,
  gemmTileM,
} from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import { gemmMTileGeometry } from "../../packages/runtime/src/kernels/gemm.ts";
import type { GeometryProfile } from "../../packages/runtime/src/kernels/geometry-profile.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
} from "../../packages/runtime/src/kernels/i8a8-geometry.ts";
import { SWEEP_OPS, type SweepOp } from "./cases.ts";
import { conv2dCandidate, gemmCandidate, i8a8Candidate } from "./geometries.ts";
import { DEFAULT_DRIFT_RANGE, driftOutOfRange, REPORT_FORMAT } from "./report.ts";

/** `--min-speedup` の既定（既定比がこれ未満の勝ちは測定の揺れと区別しない）。 */
export const DEFAULT_MIN_SPEEDUP = 1.05;

/** プロファイル id（ファイル名 `<id>.ts` と export 名 `<ID を大文字 snake>` の元）。 */
export const PROFILE_ID = /^[a-z][a-z0-9]*(-[a-z0-9]+)*$/;

/**
 * 表を当てる adapter（生成物の `match`）。`vendor` / `architecture` / `description` は runtime の選択と
 * 同じ文字列の完全一致で、`description` は `architecture` と組でだけ指定できる（runtime の門と同じ条件）。
 * `optIn` は `match` を省いた**注入専用**の表 — runtime は自動では選ばず、アプリが id で引いて
 * `acquireGpu({ geometryProfile })` へ渡す（ADR 0115 追記決定 7）。
 */
export type ProfileTarget =
  | {
    readonly optIn?: undefined;
    readonly vendor: string;
    readonly architecture?: string;
    readonly description?: string;
  }
  | {
    readonly optIn: true;
    readonly vendor?: undefined;
    readonly architecture?: undefined;
    readonly description?: undefined;
  };

/**
 * 生成の入力のうち掃引の記録以外（CLI の `--from` / `--id` / `--vendor` / `--architecture` /
 * `--description` / `--opt-in` / `--out` / `--min-speedup`）。`from` と `out` は生成物のコメントの
 * 再生成コマンドに載る path。
 */
export type ProfileSpec = ProfileTarget & {
  readonly from: readonly string[];
  readonly id: string;
  readonly out: string;
  readonly minSpeedup: number;
};

type Geometry = GemmGeometry | I8a8Geometry;

type SweepAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly description: string;
};

/** 掃引の 1 行のうち生成が読む欄。 */
type SweepObservation = {
  readonly caseId: string;
  readonly op: SweepOp;
  readonly shape: string;
  readonly geometry: string;
  readonly geometryParams: Geometry;
  readonly isDefault: boolean;
  readonly speedupVsDefault?: number;
  readonly identicalToDefault?: boolean;
  readonly error?: string;
};

/** ケース末尾の既定の再測定（`cases[]` のうち生成が読む欄 — report.ts の `CaseSummary`）。 */
type SweepCaseRepeat = {
  readonly caseId: string;
  /** 再測定 ÷ 初回（`defaultRepeat.driftRatio`）。 */
  readonly driftRatio?: number;
  /** 再測定の失敗（`defaultRepeatError`）。 */
  readonly error?: string;
};

/** 掃引の記録 1 本（unknown 境界で検査したもの）。 */
export type SweepSource = {
  /** 表示と再生成コマンドに載せる path（リポ直下からの相対）。 */
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly adapter: SweepAdapter;
  readonly cases: readonly SweepCaseRepeat[];
  readonly rows: readonly SweepObservation[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const requireString = (record: Record<string, unknown>, key: string, where: string): string => {
  const value = record[key];
  if (typeof value !== "string") throw new Error(`${where}: ${key} が文字列でない`);
  return value;
};

const optional = <T>(
  record: Record<string, unknown>,
  key: string,
  accept: (value: unknown) => value is T,
  where: string,
): T | undefined => {
  const value = record[key];
  if (value === undefined) return undefined;
  if (!accept(value)) throw new Error(`${where}: ${key} の型が違う（${JSON.stringify(value)}）`);
  return value;
};

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value > 0;

const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";

const isString = (value: unknown): value is string => typeof value === "string";

const isSweepOp = (value: unknown): value is SweepOp =>
  typeof value === "string" && (SWEEP_OPS as readonly string[]).includes(value);

const isI8a8Op = (op: SweepOp): boolean => op === "i8a8-linear" || op === "i8a8-attention";

/** 幾何の欄（f32 族は 4 値・i8a8 族は tileK 込みの 5 値 — 族と欄の組が違えば落とす）。 */
const parseGeometryParams = (value: unknown, op: SweepOp, where: string): Geometry => {
  if (!isRecord(value)) throw new Error(`${where}: geometryParams がオブジェクトでない`);
  const keys = isI8a8Op(op)
    ? ["regM", "regN", "wgX", "wgY", "tileK"] as const
    : ["regM", "regN", "wgX", "wgY"] as const;
  const extra = Object.keys(value).filter((key) => !(keys as readonly string[]).includes(key));
  if (extra.length > 0) {
    throw new Error(`${where}: geometryParams に想定外の欄 ${extra.join(", ")}`);
  }
  const params: Record<string, number> = {};
  for (const key of keys) {
    const field = value[key];
    if (typeof field !== "number" || !Number.isSafeInteger(field) || field < 1) {
      throw new Error(`${where}: geometryParams.${key} は正整数（${JSON.stringify(field)}）`);
    }
    params[key] = field;
  }
  return isI8a8Op(op)
    ? {
      regM: params.regM,
      regN: params.regN,
      wgX: params.wgX,
      wgY: params.wgY,
      tileK: params.tileK,
    } satisfies I8a8Geometry
    : { regM: params.regM, regN: params.regN, wgX: params.wgX, wgY: params.wgY };
};

const parseRow = (value: unknown, where: string): SweepObservation => {
  if (!isRecord(value)) throw new Error(`${where}: 行がオブジェクトでない`);
  const op = value.op;
  if (!isSweepOp(op)) throw new Error(`${where}: op が ${SWEEP_OPS.join(" / ")} のどれでもない`);
  const isDefault = value.isDefault;
  if (!isBoolean(isDefault)) throw new Error(`${where}: isDefault が真偽値でない`);
  const speedupVsDefault = optional(value, "speedupVsDefault", isPositiveNumber, where);
  const identicalToDefault = optional(value, "identicalToDefault", isBoolean, where);
  const error = optional(value, "error", isString, where);
  return {
    caseId: requireString(value, "caseId", where),
    op,
    shape: requireString(value, "shape", where),
    geometry: requireString(value, "geometry", where),
    geometryParams: parseGeometryParams(value.geometryParams, op, where),
    isDefault,
    ...(speedupVsDefault === undefined ? {} : { speedupVsDefault }),
    ...(identicalToDefault === undefined ? {} : { identicalToDefault }),
    ...(error === undefined ? {} : { error }),
  };
};

const parseCaseRepeat = (value: unknown, where: string): SweepCaseRepeat => {
  if (!isRecord(value)) throw new Error(`${where}: オブジェクトでない`);
  const repeat = value.defaultRepeat;
  if (repeat !== undefined && !isRecord(repeat)) {
    throw new Error(`${where}: defaultRepeat がオブジェクトでない`);
  }
  const driftRatio = repeat === undefined
    ? undefined
    : optional(repeat, "driftRatio", isPositiveNumber, `${where} defaultRepeat`);
  const error = optional(value, "defaultRepeatError", isString, where);
  return {
    caseId: requireString(value, "caseId", where),
    ...(driftRatio === undefined ? {} : { driftRatio }),
    ...(error === undefined ? {} : { error }),
  };
};

/**
 * 掃引の記録を読む（unknown 境界 — 生成が読む欄だけを検査して fail loudly）。
 *
 * MUST: ケースごとに既定の行がちょうど 1 本あること。比の土台が無い・2 つあるケースは、
 * どの幾何を既定比で比べたのかが決まらない。
 */
export const parseSweepReport = (
  parsed: unknown,
  meta: { readonly path: string; readonly sha256: string },
): SweepSource => {
  const where = meta.path;
  if (!isRecord(parsed)) throw new Error(`${where}: JSON がオブジェクトでない`);
  if (parsed.format !== REPORT_FORMAT) {
    throw new Error(`${where}: format が ${REPORT_FORMAT} でない（${String(parsed.format)}）`);
  }
  const adapter = parsed.adapter;
  if (!isRecord(adapter)) throw new Error(`${where}: adapter が無い`);
  // MUST: GPU の timestamp で測った掃引だけを受ける（ADR 0115 §4）。壁時計は submit → 完了の床を
  // 含むので幾何どうしの比が 1 へ縮み、「既定比 ≥ --min-speedup」の門が意味を失う
  const timingFix = "Deno の CLI は adapter が timestamp-query を列挙すれば timestamp で測る" +
    "（単位 deno-raw-tick）ので timestamp-query を持つ adapter で、Chrome のページは" +
    "「GPU の timestamp で測る」にチェックを入れて測り直す";
  const gpuTiming = parsed.gpuTiming;
  if (!isRecord(gpuTiming)) {
    throw new Error(`${where}: gpuTiming が無い（時間の単位が分からない）— ${timingFix}`);
  }
  const unit = gpuTiming.unit;
  if (unit !== "ns" && unit !== "deno-raw-tick") {
    throw new Error(
      `${where}: gpuTiming.unit が ${JSON.stringify(unit)}（ns か deno-raw-tick であること — ` +
        `wall は壁時計で比が 1 へ縮む）— ${timingFix}`,
    );
  }
  // Chrome の 100 µs 量子化の下では比が刻みに潰れる（README の Caveats）— 表の材料にしない
  if (gpuTiming.quantized === true) {
    throw new Error(
      `${where}: timestamp が 100 µs に量子化されている（gpuTiming.quantized）— ` +
        "Chrome の WebGPU developer features を有効にして測り直す",
    );
  }
  if (gpuTiming.quantized !== false) {
    throw new Error(
      `${where}: gpuTiming.quantized が真偽値でない（${JSON.stringify(gpuTiming.quantized)}）`,
    );
  }
  // 既定の再測定（cases[]）は材料の門が読む — 無い記録は「どのケースも再測定が無い」になるので、
  // 黙って全ケースを外さずに落とす
  if (!Array.isArray(parsed.cases)) {
    throw new Error(`${where}: cases が配列でない（ケースごとの既定の再測定が読めない）`);
  }
  const cases = parsed.cases.map((entry, index) =>
    parseCaseRepeat(entry, `${where} cases[${index}]`)
  );
  const caseIds = new Set<string>();
  for (const { caseId } of cases) {
    if (caseIds.has(caseId)) throw new Error(`${where}: cases に ${caseId} が 2 本`);
    caseIds.add(caseId);
  }
  if (!Array.isArray(parsed.rows)) throw new Error(`${where}: rows が配列でない`);
  const rows = parsed.rows.map((row, index) => parseRow(row, `${where} rows[${index}]`));
  const defaults = new Map<string, number>();
  for (const row of rows) {
    defaults.set(row.caseId, (defaults.get(row.caseId) ?? 0) + (row.isDefault ? 1 : 0));
  }
  for (const [caseId, count] of defaults) {
    if (count !== 1) {
      throw new Error(`${where}: ${caseId} の既定の行が ${count} 本（1 本であること）`);
    }
  }
  return {
    path: meta.path,
    sha256: meta.sha256,
    date: requireString(parsed, "date", where),
    adapter: {
      vendor: requireString(adapter, "vendor", `${where} adapter`),
      architecture: requireString(adapter, "architecture", `${where} adapter`),
      description: requireString(adapter, "description", `${where} adapter`),
    },
    cases,
    rows,
  };
};

/** プロファイルの欄（= クラス）。生成物のコメントでもこの綴りで欄を指す。 */
export type ProfileSlot =
  | "gemmRows[0]"
  | "gemmRows[1]"
  | "gemmRows[2]"
  | "attention.qk"
  | "attention.pv"
  | "conv2d.rows64"
  | "conv2d.rows32"
  | "i8a8.linear"
  | "i8a8.attentionQk"
  | "i8a8.attentionPv";

type SlotSpec = {
  readonly slot: ProfileSlot;
  /** 欄が効く範囲（コメント用）。 */
  readonly scope: string;
  /**
   * 今の runtime の既定と、その表示名（掃引の `geometry` と同じ綴り）。掃引にケースが無いクラスに
   * 書く値で、掃引の既定の行と突き合わせる比の土台でもある。
   */
  readonly fallback: () => { readonly name: string; readonly geometry: Geometry };
};

/**
 * 行数バケットの上限（runtime の `GEMM_ROWS_BUCKETS`〈src/kernels/gemm-geometry.ts〉の `maxRows` —
 * 末尾は上限なし）。書き写さずに導くのは、境界がずれると、あるクラスの実測で選んだ幾何が別の
 * 行数帯へ効くから（既定プロファイルとの一致は profile_test.ts が見る）。
 * NOTE: 境界と既定は `gemm-geometry.ts` から取る（生成物の値を境界・既定の正本にしない）。ただし生成器は
 * 生成物へ依存している — 候補集合 `quick+`（geometries.ts が `geometry-profiles/index.ts` を読む）と
 * runtime の mod.ts（report.ts → anima-residency/timing.ts）を通じて。登録済みの生成物が構文的に壊れると
 * 生成器自体が読めなくなるので、その場合は index.ts の登録を外してから再生成する。
 */
export const ROWS_BUCKETS: readonly number[] = GEMM_ROWS_BUCKETS.map((rule) => rule.maxRows);

/** {@link ROWS_BUCKETS} と同じ順の欄。 */
const ROWS_SLOTS = ["gemmRows[0]", "gemmRows[1]", "gemmRows[2]"] as const;

/** runtime の行数バケットの `index` 段の既定の幾何（段が足りなければ落とす）。 */
const rowsBucketGeometry = (index: number): GemmGeometry => {
  const rule = GEMM_ROWS_BUCKETS[index];
  if (rule === undefined) throw new Error(`runtime の行数バケットに ${index} 段目が無い`);
  return rule.geometry;
};

/** 欄の一覧（生成物の欄の順・コメントの採否の順）。 */
const SLOTS: readonly SlotSpec[] = [
  {
    slot: "gemmRows[0]",
    scope: "linear / matmul / bmm の行数 ≤ 64",
    fallback: () => gemmCandidate(rowsBucketGeometry(0)),
  },
  {
    slot: "gemmRows[1]",
    scope: "linear / matmul / bmm の行数 65〜512",
    fallback: () => gemmCandidate(rowsBucketGeometry(1)),
  },
  {
    slot: "gemmRows[2]",
    scope: "linear / matmul / bmm の行数 > 512",
    fallback: () => gemmCandidate(rowsBucketGeometry(2)),
  },
  {
    slot: "attention.qk",
    scope: "融合 attention f32 ①QK",
    fallback: () => gemmCandidate(defaultGemmGeometry()),
  },
  {
    slot: "attention.pv",
    scope: "融合 attention f32 ③PV",
    fallback: () => gemmCandidate(defaultGemmGeometry()),
  },
  {
    slot: "conv2d.rows64",
    scope: "conv2d implicit GEMM の m タイル 64 行",
    fallback: () => conv2dCandidate(gemmMTileGeometry(64)),
  },
  {
    slot: "conv2d.rows32",
    scope: "conv2d implicit GEMM の m タイル 32 行",
    fallback: () => conv2dCandidate(gemmMTileGeometry(32)),
  },
  {
    slot: "i8a8.linear",
    scope: "i8a8 linear",
    fallback: () => i8a8Candidate(defaultI8a8Geometry("linear")),
  },
  {
    slot: "i8a8.attentionQk",
    scope: "i8a8 attention ①QK",
    fallback: () => i8a8Candidate(defaultI8a8Geometry("attention_qk")),
  },
  {
    slot: "i8a8.attentionPv",
    scope: "i8a8 attention ③PV",
    fallback: () => i8a8Candidate(defaultI8a8Geometry("attention_pv")),
  },
];

/** linear / matmul の行数（`caseShape` の `M{m} N{n} K{k}`）。 */
const LINEAR_SHAPE = /^M(\d+) N\d+ K\d+$/;
/** bmm の行列 1 枚の行数（`caseShape` の `B{batch} M{m} N{n} K{k}` — バッチは z 軸でバケットに効かない）。 */
const BMM_SHAPE = /^B\d+ M(\d+) N\d+ K\d+$/;
/** 融合 attention の段（`caseShape` の `{qk|pv} BH…`）。 */
const ATTENTION_SHAPE = /^(qk|pv) BH\d+ M\d+ N\d+ D\d+$/;

const attentionStage = (row: SweepObservation, where: string): "qk" | "pv" => {
  const matched = ATTENTION_SHAPE.exec(row.shape);
  if (matched === null) {
    throw new Error(`${where}: ${row.caseId} の shape を読めない（${row.shape}）`);
  }
  return matched[1] === "qk" ? "qk" : "pv";
};

/**
 * 行の欄。linear / matmul / bmm は shape の M（本番の 3 経路が同じ gemmRows の表を M で引く —
 * src/runtime/recipe-builders/linear.ts）・attention は shape の段・conv2d は**そのケースの既定の行の tileM**
 * （本番の m タイルの選択 `conv2dIgemmMTile` が選んだ側 — 掃引が既定として測った幾何が正本）。
 */
const slotOf = (
  row: SweepObservation,
  conv2dDefaultTileM: ReadonlyMap<string, number>,
  where: string,
): ProfileSlot => {
  switch (row.op) {
    case "linear":
    case "matmul":
    case "bmm": {
      const matched = (row.op === "bmm" ? BMM_SHAPE : LINEAR_SHAPE).exec(row.shape);
      if (matched === null) {
        throw new Error(`${where}: ${row.caseId} の shape を読めない（${row.shape}）`);
      }
      const rows = Number(matched[1]);
      return ROWS_SLOTS[ROWS_BUCKETS.findIndex((maxRows) => rows <= maxRows)];
    }
    case "attention":
      return attentionStage(row, where) === "qk" ? "attention.qk" : "attention.pv";
    case "conv2d": {
      const tileM = conv2dDefaultTileM.get(row.caseId);
      if (tileM === 64) return "conv2d.rows64";
      if (tileM === 32) return "conv2d.rows32";
      throw new Error(`${where}: ${row.caseId} の既定の m タイルが 64 / 32 でない（${tileM}）`);
    }
    case "i8a8-linear":
      return "i8a8.linear";
    case "i8a8-attention":
      return attentionStage(row, where) === "qk" ? "i8a8.attentionQk" : "i8a8.attentionPv";
  }
};

const geometryKey = (geometry: Geometry): string =>
  "tileK" in geometry
    ? `${geometry.regM},${geometry.regN},${geometry.wgX},${geometry.wgY},${geometry.tileK}`
    : `${geometry.regM},${geometry.regN},${geometry.wgX},${geometry.wgY}`;

const geometricMean = (values: readonly number[]): number =>
  Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);

/** 比の表示（生成物のコメントと標準出力で同じ綴り）。 */
export const formatRatio = (value: number): string => `×${value.toFixed(3)}`;

export type SlotVerdict = {
  readonly slot: ProfileSlot;
  readonly scope: string;
  /** クラスのケース（id の昇順）。 */
  readonly cases: readonly string[];
  readonly outcome:
    | {
      readonly kind: "adopted";
      readonly name: string;
      readonly geometry: Geometry;
      /** ケース間の幾何平均。 */
      readonly geomean: number;
      readonly min: number;
      readonly max: number;
    }
    | {
      readonly kind: "default";
      readonly name: string;
      readonly geometry: Geometry;
      readonly reason: string;
    };
  /** 採らなかった幾何と理由（名前の昇順）。 */
  readonly rejected: readonly { readonly name: string; readonly reason: string }[];
  /** このクラスのケースのうち、掃引ごとに比の材料から外したもの（掃引を渡した順 → ケース id の昇順）。 */
  readonly excluded: readonly ExcludedCase[];
};

/** 掃引 1 本の中で比の材料から外したケース（{@link exclusionReason}）— 出力の一致と失敗は見る。 */
export type ExcludedCase = {
  readonly path: string;
  readonly caseId: string;
  /** 「既定の再測定比 ×1.160 が範囲外」など（{@link excludedCaseLine} が文にする）。 */
  readonly reason: string;
};

/** 外したケースの 1 行（生成物のコメント・標準出力・ページで同じ文）。 */
export const excludedCaseLine = (excluded: ExcludedCase): string =>
  `掃引 ${excluded.path}: ${excluded.caseId} は${excluded.reason}のため比の材料から外した（出力の一致と失敗は見る）`;

/**
 * ケースをその掃引の比の材料から外す理由（外さないなら undefined）。
 *
 * 既定の再測定比（ケースの末尾で既定をもう 1 度測った値 ÷ 初回）が {@link DEFAULT_DRIFT_RANGE} の外なら、
 * 熱・クロックがケースの途中で動いた疑いがあり、比の土台（既定の値）ごと揺れている — そのケースの既定比は
 * どの幾何についても信用できない。再測定が失敗した・無いケースは揺れを確かめられないので同じ扱い。
 */
const exclusionReason = (repeat: SweepCaseRepeat | undefined): string | undefined => {
  if (repeat?.error !== undefined) return `既定の再測定が失敗（${repeat.error}）`;
  if (repeat?.driftRatio === undefined) return "既定の再測定が無い";
  return driftOutOfRange(repeat.driftRatio)
    ? `既定の再測定比 ${formatRatio(repeat.driftRatio)} が範囲外`
    : undefined;
};

/** クラスに振り分けた行（`included` = 材料に残したか）。 */
type SlotRow = { readonly row: SweepObservation; readonly included: boolean };

type Eligibility =
  | {
    readonly eligible: true;
    readonly geomean: number;
    readonly min: number;
    readonly max: number;
  }
  | { readonly eligible: false; readonly reason: string };

/**
 * 候補 1 つの判定。クラスの**全ケース**を見る — 測っていない・失敗・不一致・比が無いケースが
 * 1 つでもあれば落とし（ケース id の昇順で最初のもの）、全ケースが揃えば最も負けたケースで門を見る。
 *
 * 失敗と出力の不一致は、比の材料から外した行（`included: false`）も含めて見る — 出力の一致は正しさの門で
 * 熱・クロックに依らないから、どれか 1 本の掃引で不一致 / 失敗ならその幾何は採らない。比は材料に残した行
 * だけから取り、全掃引で比の材料から外したケース（`unmeasured`）は、どの幾何も比を測っていない扱い。
 */
const judgeCandidate = (
  cases: readonly string[],
  observations: ReadonlyMap<string, readonly SlotRow[]>,
  unmeasured: ReadonlySet<string>,
  minSpeedup: number,
): Eligibility => {
  const perCase: { readonly caseId: string; readonly speedup: number }[] = [];
  for (const caseId of cases) {
    const all = (observations.get(caseId) ?? []).map(({ row }) => row);
    const failed = all.find((row) => row.error !== undefined);
    if (failed !== undefined) {
      return { eligible: false, reason: `${caseId} で失敗（${failed.error}）` };
    }
    if (all.some((row) => row.identicalToDefault === false)) {
      return { eligible: false, reason: `${caseId} で出力が既定と不一致` };
    }
    const seen = (observations.get(caseId) ?? []).flatMap(({ row, included }) =>
      included ? [row] : []
    );
    if (seen.length === 0) {
      return {
        eligible: false,
        reason: unmeasured.has(caseId)
          ? `${caseId} で測っていない（全掃引で比の材料から外した）`
          : `${caseId} で測っていない`,
      };
    }
    const speedups: number[] = [];
    for (const row of seen) {
      if (row.identicalToDefault !== true || row.speedupVsDefault === undefined) {
        return { eligible: false, reason: `${caseId} で既定との比較が無い（既定の行の失敗など）` };
      }
      speedups.push(row.speedupVsDefault);
    }
    perCase.push({ caseId, speedup: geometricMean(speedups) });
  }
  const worst = perCase.reduce((low, entry) => entry.speedup < low.speedup ? entry : low);
  if (worst.speedup < minSpeedup) {
    return {
      eligible: false,
      reason: `${worst.caseId} で ${formatRatio(worst.speedup)} < ${formatRatio(minSpeedup)}`,
    };
  }
  const speedups = perCase.map((entry) => entry.speedup);
  return {
    eligible: true,
    geomean: geometricMean(speedups),
    min: Math.min(...speedups),
    max: Math.max(...speedups),
  };
};

const judgeSlot = (
  spec: SlotSpec,
  slotRows: readonly SlotRow[],
  excluded: readonly ExcludedCase[],
  minSpeedup: number,
): SlotVerdict => {
  if (slotRows.length === 0) {
    const { name, geometry } = spec.fallback();
    return {
      slot: spec.slot,
      scope: spec.scope,
      cases: [],
      outcome: { kind: "default", name, geometry, reason: "掃引にこのクラスのケースが無い" },
      rejected: [],
      excluded,
    };
  }
  // 既定の行の門とケースの一覧は比の材料から外した行も含めて見る（外すのは比の観測だけ — 土台の検査と
  // 出力の一致・失敗は記録そのものに掛ける）
  const rows = slotRows.map(({ row }) => row);
  const cases = [...new Set(rows.map((row) => row.caseId))].sort();
  const measured = new Set(
    slotRows.filter(({ included }) => included).map(({ row }) => row.caseId),
  );
  const unmeasured = new Set(cases.filter((caseId) => !measured.has(caseId)));
  const defaults = rows.filter((row) => row.isDefault);
  const defaultKeys = new Set(defaults.map((row) => geometryKey(row.geometryParams)));
  // MUST: クラスの既定は 1 つ。割れていれば比の土台がケースごとに違う（掃引の版違いなど）
  if (defaultKeys.size !== 1) {
    throw new Error(
      `${spec.slot}: 既定の行の幾何が割れている（${
        [...new Set(defaults.map((row) => row.geometry))].join(" / ")
      }）`,
    );
  }
  const defaultRow = defaults[0];
  const runtimeDefault = spec.fallback();
  // MUST: 比の土台（掃引の既定の行）が今の runtime の既定であること。違えば既定比は別の幾何に対する
  // 比で、「既定より速い」が今の runtime では成り立たない
  if (geometryKey(defaultRow.geometryParams) !== geometryKey(runtimeDefault.geometry)) {
    throw new Error(
      `${spec.slot}: 掃引の既定の行（${defaultRow.geometry}）が今の runtime の既定（${runtimeDefault.name}）` +
        "と違う — 古い runtime で取った掃引か境界のずれ。比の土台が別物なので生成しない",
    );
  }
  const byName = new Map<
    string,
    { geometry: Geometry; observations: Map<string, SlotRow[]> }
  >();
  for (const slotRow of slotRows) {
    const { row } = slotRow;
    if (row.isDefault || row.geometry === defaultRow.geometry) continue;
    const entry = byName.get(row.geometry) ??
      { geometry: row.geometryParams, observations: new Map<string, SlotRow[]>() };
    if (geometryKey(entry.geometry) !== geometryKey(row.geometryParams)) {
      throw new Error(`${spec.slot}: 幾何 ${row.geometry} の geometryParams が行ごとに違う`);
    }
    // 外した行も観測に載せる — 比には使わないが、失敗と出力の不一致は判定に効く（judgeCandidate）
    entry.observations.set(row.caseId, [...(entry.observations.get(row.caseId) ?? []), slotRow]);
    byName.set(row.geometry, entry);
  }
  const judged = [...byName.entries()]
    .map(([name, entry]) => ({
      name,
      geometry: entry.geometry,
      eligibility: judgeCandidate(cases, entry.observations, unmeasured, minSpeedup),
    }))
    .sort((left, right) => left.name < right.name ? -1 : left.name > right.name ? 1 : 0);
  const eligible = judged
    .flatMap((candidate) =>
      candidate.eligibility.eligible ? [{ ...candidate, ...candidate.eligibility }] : []
    )
    // 幾何平均の降順・同値は名前の昇順（入力の順に依らず決まる）
    .sort((left, right) => right.geomean - left.geomean || (left.name < right.name ? -1 : 1));
  const winner = eligible[0];
  const rejected = judged
    .filter((candidate) => candidate.name !== winner?.name)
    .map((candidate) => ({
      name: candidate.name,
      reason: candidate.eligibility.eligible
        ? `幾何平均 ${formatRatio(candidate.eligibility.geomean)}（採用 ${
          formatRatio(winner.geomean)
        } に届かない）`
        : candidate.eligibility.reason,
    }));
  return {
    slot: spec.slot,
    scope: spec.scope,
    cases,
    outcome: winner === undefined
      ? {
        kind: "default",
        name: defaultRow.geometry,
        geometry: defaultRow.geometryParams,
        reason: unmeasured.size === cases.length
          ? "クラスの全ケースを全掃引で比の材料から外した"
          : `全ケースで出力が一致し ${formatRatio(minSpeedup)} 以上の幾何が無い`,
      }
      : {
        kind: "adopted",
        name: winner.name,
        geometry: winner.geometry,
        geomean: winner.geomean,
        min: winner.min,
        max: winner.max,
      },
    rejected,
    excluded,
  };
};

/** 表の相手の表示（`vendor / architecture / description` — 指定した欄だけ）。 */
const targetLabel = (target: ProfileTarget): string =>
  target.optIn === true
    ? "（注入専用 — match なし）"
    : [target.vendor, target.architecture, target.description]
      .filter((part) => part !== undefined).join(" / ");

/**
 * 掃引の記録（1 本以上・同じ adapter）から欄ごとの採否を出す（純関数）。
 *
 * 材料の門: 掃引ごとに、既定の再測定比が範囲外か再測定が失敗 / 無いケース（{@link exclusionReason}）は
 * **その掃引の**比の観測から外す。同じケースを別の掃引が測っていればそちらの比で判定し、どの掃引にも
 * 残らないケースはどの幾何も比を測っていない扱い（クラスの全ケースで勝つ規則により欄は既定のまま）。
 * 出力の不一致と失敗は外したケースでも判定に残す — 出力の一致は正しさの門で熱に依らないので、外した掃引で
 * 不一致 / 失敗の幾何は他の掃引で一致していても候補にしない。外すかどうかの選択肢は持たない（規則は 1 つ）。
 */
export const deriveProfile = (
  sources: readonly SweepSource[],
  options: ProfileTarget & { readonly minSpeedup: number },
): readonly SlotVerdict[] => {
  if (sources.length === 0) throw new Error("掃引の記録が 1 本も無い");
  // 生成物の gemmRows は 3 段の欄で書く — runtime の段数が変われば欄と境界の対応が崩れる
  if (ROWS_BUCKETS.length !== ROWS_SLOTS.length) {
    throw new Error(
      `runtime の行数バケットが ${ROWS_BUCKETS.length} 段（生成器の欄は ${ROWS_SLOTS.length} 段）`,
    );
  }
  // runtime の門（assertGeometryProfile）と同じ条件 — 生成してから注入・登録の段で落ちる前に止める。
  // description は機種の名前で、vendor / architecture の内側を分けるためだけの欄
  if (options.description === "") throw new Error("description は空文字にしない（未指定は省く）");
  if (options.description !== undefined && options.architecture === undefined) {
    throw new Error("description は vendor と architecture の両方と組で指定する");
  }
  const seen = new Map<string, string>();
  for (const source of sources) {
    const twin = seen.get(source.sha256);
    // 同じ記録を 2 度数えると、そのケースの観測だけが重く効く
    if (twin !== undefined) {
      throw new Error(`同じ掃引を 2 度渡している（${twin} と ${source.path}）`);
    }
    seen.set(source.sha256, source.path);
    // MUST: 別の adapter の実測を混ぜない（表は adapter ごとの答え — B570 と M2 は逆を指す）。
    // description を指定したら、その機種の掃引だけ（同じ vendor / architecture の別機種を混ぜない）
    const { vendor, architecture, description } = source.adapter;
    if (
      options.optIn !== true && (
        vendor !== options.vendor ||
        (options.architecture !== undefined && architecture !== options.architecture) ||
        (options.description !== undefined && description !== options.description)
      )
    ) {
      throw new Error(
        `${source.path}: adapter ${vendor} / ${architecture} / ${description} が指定した ${
          targetLabel(options)
        } と合わない`,
      );
    }
    // MUST: 掃引どうしも description まで一致（空どうしは一致・片方だけ空は不一致）— --opt-in や
    // --description 省略では上の門が description を見ないので、同じ vendor / architecture の別機種
    // （M2 と M5 など）が混ざるのをここで止める
    const first = sources[0].adapter;
    if (
      vendor !== first.vendor || architecture !== first.architecture ||
      description !== first.description
    ) {
      throw new Error(
        `${source.path}: adapter ${vendor} / ${architecture} / ${description} が ${
          sources[0].path
        } の ${first.vendor} / ${first.architecture} / ${first.description} と違う`,
      );
    }
  }
  const bySlot = new Map<ProfileSlot, SlotRow[]>();
  const excludedBySlot = new Map<ProfileSlot, ExcludedCase[]>();
  for (const source of sources) {
    const conv2dDefaultTileM = new Map(
      source.rows
        .filter((row) => row.op === "conv2d" && row.isDefault)
        .map((row) => [row.caseId, gemmTileM(row.geometryParams)] as const),
    );
    const repeats = new Map(source.cases.map((repeat) => [repeat.caseId, repeat] as const));
    const excludedHere = new Map<string, { readonly slot: ProfileSlot; readonly reason: string }>();
    for (const row of source.rows) {
      const slot = slotOf(row, conv2dDefaultTileM, source.path);
      const reason = exclusionReason(repeats.get(row.caseId));
      if (reason !== undefined) excludedHere.set(row.caseId, { slot, reason });
      bySlot.set(slot, [...(bySlot.get(slot) ?? []), { row, included: reason === undefined }]);
    }
    const byCaseId = [...excludedHere.entries()]
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0);
    for (const [caseId, { slot, reason }] of byCaseId) {
      excludedBySlot.set(slot, [
        ...(excludedBySlot.get(slot) ?? []),
        { path: source.path, caseId, reason },
      ]);
    }
  }
  return SLOTS.map((spec) =>
    judgeSlot(
      spec,
      bySlot.get(spec.slot) ?? [],
      excludedBySlot.get(spec.slot) ?? [],
      options.minSpeedup,
    )
  );
};

/** `apple-metal-3` → `APPLE_METAL_3`。 */
export const profileConstName = (id: string): string => id.toUpperCase().replaceAll("-", "_");

/** 採否の一覧（生成物の冒頭コメントと標準出力で同じ行）。 */
export const verdictLines = (verdicts: readonly SlotVerdict[]): string[] =>
  verdicts.flatMap((verdict) => {
    const head = `${verdict.slot}（${verdict.scope}・${verdict.cases.length} ケース）`;
    const { outcome } = verdict;
    const line = outcome.kind === "adopted"
      ? `- ${head}: 採用 ${outcome.name} ${formatRatio(outcome.geomean)}（${
        formatRatio(outcome.min)
      }〜${formatRatio(outcome.max)}）`
      : `- ${head}: 既定 ${outcome.name} のまま（${outcome.reason}）`;
    return [
      line,
      ...verdict.excluded.map((excluded) => `  - ${excludedCaseLine(excluded)}`),
      ...verdict.rejected.map(({ name, reason }) => `  - ${name}: ${reason}`),
    ];
  });

/** シェルにそのまま貼れる形（空白・記号を含む語だけ単引用符で包む）。 */
const shellWord = (word: string): string =>
  /^[A-Za-z0-9_./:=@%+-]+$/.test(word) ? word : `'${word.replaceAll("'", `'\\''`)}'`;

/** 表の相手を指す CLI の引数（`--opt-in` か `--vendor` [`--architecture`] [`--description`]）。 */
const targetFlags = (target: ProfileTarget): string[] =>
  target.optIn === true ? ["--opt-in"] : [
    "--vendor",
    shellWord(target.vendor),
    ...(target.architecture === undefined
      ? []
      : ["--architecture", shellWord(target.architecture)]),
    ...(target.description === undefined ? [] : ["--description", shellWord(target.description)]),
  ];

/** 再生成コマンド（`--check` 抜き・行継続つきの複数行）。 */
export const regenerateCommand = (flags: ProfileSpec): string[] => [
  "deno run -A tools/geometry-sweep/main.ts profile \\",
  ...flags.from.map((path) => `  --from ${shellWord(path)} \\`),
  `  --id ${shellWord(flags.id)} ${targetFlags(flags).join(" ")} \\`,
  `  --out ${shellWord(flags.out)} --min-speedup ${flags.minSpeedup}`,
];

const renderGeometry = (geometry: Geometry, slot: ProfileSlot): string => {
  if ("tileK" in geometry) {
    assertI8a8Geometry(geometry, slot);
    return `{ regM: ${geometry.regM}, regN: ${geometry.regN}, wgX: ${geometry.wgX}, wgY: ${geometry.wgY}, tileK: ${geometry.tileK} }`;
  }
  assertGemmGeometry(geometry, slot);
  return `{ regM: ${geometry.regM}, regN: ${geometry.regN}, wgX: ${geometry.wgX}, wgY: ${geometry.wgY} }`;
};

/** adapter の表示（空の欄は落とす — Deno は architecture を空で返す）。 */
const adapterLabel = (sources: readonly SweepSource[]): string => {
  const { vendor, architecture } = sources[0].adapter;
  const descriptions = [...new Set(sources.map((source) => source.adapter.description))];
  return [vendor, architecture, ...descriptions].filter((part) => part !== "").join(" / ");
};

/**
 * 生成物の TS（整形前）。同じ入力からは常に同じ文字列（時刻・環境を読まない — 日付は掃引の記録の
 * `date`）。整形は {@link formatTypeScript} が担う。
 */
export const renderProfileSource = (
  flags: ProfileSpec,
  sources: readonly SweepSource[],
  verdicts: readonly SlotVerdict[],
): string => {
  // 値は注入の表（{@link buildGeometryProfile}）から書く — TS の生成物と注入の JSON を 1 本の経路で作る
  const profile = buildGeometryProfile(flags, sources, verdicts);
  const adapter = profile.provenance.adapter;
  // 行には掃引の記録由来の文字列（失敗の error 文）が入るので、コメントを閉じる綴りと改行を潰す
  const comment = (lines: readonly string[]): string[] =>
    lines.map((line) =>
      line === "" ? " *" : ` * ${line.replaceAll("*/", "* /").replaceAll(/\r?\n/g, " ")}`
    );
  return [
    "/**",
    ...comment([
      `幾何プロファイル \`${flags.id}\`（**生成物 — 手で編集しない**）。`,
      "",
      ...(flags.optIn === true
        ? [
          "tools/geometry-sweep の `profile` が掃引の記録から書いた、タイル幾何の静的な表（perf-ledger K-71）。",
          "**注入専用**（`match` を省いた表）: runtime は自動では選ばない — アプリが `BUILTIN_GEOMETRY_PROFILES`",
          "から id で引いて `acquireGpu({ geometryProfile })` に渡す（ADR 0115 追記決定 7）。",
        ]
        : [
          `tools/geometry-sweep の \`profile\` が掃引の記録から書いた、adapter \`${
            targetLabel(flags)
          }\` 用のタイル幾何の`,
          "静的な表（perf-ledger K-71）。runtime は adapter の `match`（vendor / architecture / description の",
          "完全一致）でこの表を選ぶだけ。",
        ]),
      "実行時には測らない（オートチューン禁止 — ADR 0022 決定 3）。値を変えるときは掃引を取り直して",
      "下のコマンドで再生成する。",
      "",
      "再生成（リポ直下から・`--check` を足すと再生成とバイト同一かだけを見る）:",
      "",
      ...regenerateCommand(flags).map((line) => `  ${line}`),
      "",
      `掃引（adapter ${adapter}）:`,
      "",
      ...sources.map((source) => `- ${source.path}（sha256 ${source.sha256}・${source.date}）`),
      "",
      "採否の基準: クラスの全ケースで出力が既定と一致し、既定比が " +
      `${formatRatio(flags.minSpeedup)} 以上の幾何のうち、`,
      "ケース間の幾何平均が最大のもの。無ければ既定（掃引の既定の行の幾何）。同じケースを複数の掃引が",
      "測っていれば、比はその観測の幾何平均。gemmRows は掃引にある linear / matmul / bmm のケースで決め、3 経路に同じ表が効く。",
      `材料の門: 掃引ごとに、既定の再測定比（cases[].defaultRepeat.driftRatio）が ${DEFAULT_DRIFT_RANGE.min}〜${DEFAULT_DRIFT_RANGE.max} の外か、`,
      "再測定が失敗 / 無いケースはその掃引の比の材料から外す（出力の一致と失敗は見る — 外した掃引で不一致 /",
      "失敗の幾何は採らない。比は同じケースを他の掃引が測っていればそちらで判定し、どの掃引にも残らなければ",
      "測っていない扱い）。外したケースは採否の欄ごとに「掃引 …」の行で示す。",
      "",
      "採否:",
      "",
      ...verdictLines(verdicts),
    ]),
    " */",
    'import type { GeometryProfile } from "../geometry-profile.ts";',
    "",
    ...profileDeclaration(profile),
    "",
  ].join("\n");
};

/**
 * 表の値の宣言（`export const <ID>: GeometryProfile = { … };` の行 — 整形前）。リポへ登録する生成物
 * （{@link renderProfileSource}）とアプリ用の TS（{@link renderAppProfileSource}）が同じ行を書く。
 */
const profileDeclaration = (profile: GeneratedProfile): string[] => {
  const { provenance, match } = profile;
  // match を省いた表（注入専用）は欄ごと書かない — `match: {}` は runtime の門が落とす別物
  const matchLine = match === undefined ? [] : [
    `match: { ${
      (["vendor", "architecture", "description"] as const)
        .flatMap((key) => match[key] === undefined ? [] : [`${key}: ${JSON.stringify(match[key])}`])
        .join(", ")
    } },`,
  ];
  const maxRows = (value: number): string =>
    value === Number.POSITIVE_INFINITY ? "Number.POSITIVE_INFINITY" : String(value);
  const gemmRows = profile.gemmRows.map((rule, index) =>
    `{ maxRows: ${maxRows(rule.maxRows)}, geometry: ${
      renderGeometry(rule.geometry, ROWS_SLOTS[index])
    } },`
  );
  return [
    `export const ${profileConstName(profile.id)}: GeometryProfile = {`,
    `id: ${JSON.stringify(profile.id)},`,
    ...matchLine,
    "gemmRows: [",
    ...gemmRows,
    "],",
    `attention: { qk: ${renderGeometry(profile.attention.qk, "attention.qk")}, pv: ${
      renderGeometry(profile.attention.pv, "attention.pv")
    } },`,
    `conv2d: { rows64: ${renderGeometry(profile.conv2d.rows64, "conv2d.rows64")}, rows32: ${
      renderGeometry(profile.conv2d.rows32, "conv2d.rows32")
    } },`,
    `i8a8: { linear: ${renderGeometry(profile.i8a8.linear, "i8a8.linear")}, attentionQk: ${
      renderGeometry(profile.i8a8.attentionQk, "i8a8.attentionQk")
    }, attentionPv: ${renderGeometry(profile.i8a8.attentionPv, "i8a8.attentionPv")} },`,
    `provenance: { sweep: ${JSON.stringify(provenance.sweep)}, sha256: ${
      JSON.stringify(provenance.sha256)
    }, date: ${JSON.stringify(provenance.date)}, adapter: ${JSON.stringify(provenance.adapter)} },`,
    "};",
  ];
};

/**
 * アプリに置く TS（整形前）: 公開 API の型（`@karume/runtime` の `GeometryProfile`）で表の値を
 * 定数として書く。アプリはこの定数を `acquireGpu({ geometryProfile })` へ渡して注入する（リポへ
 * 登録しない使い方 — ADR 0115 追記決定 6）。末尾の `maxRows` は `Number.POSITIVE_INFINITY`。
 */
export const renderAppProfileSource = (profile: GeneratedProfile): string => {
  const constName = profileConstName(profile.id);
  return [
    "/**",
    ` * 幾何プロファイル \`${profile.id}\`（GPU lab のプロファイルタブが掃引から作った表）。`,
    ` * acquireGpu({ geometryProfile: ${constName} }) に渡す。`,
    " */",
    'import type { GeometryProfile } from "@karume/runtime";',
    "",
    ...profileDeclaration(profile),
    "",
  ].join("\n");
};

/** 掃引から作った表（{@link buildGeometryProfile} — `provenance` を必ず持つ）。 */
export type GeneratedProfile = GeometryProfile & {
  readonly provenance: NonNullable<GeometryProfile["provenance"]>;
};

const slotGeometry = (verdicts: readonly SlotVerdict[], slot: ProfileSlot): Geometry => {
  const verdict = verdicts.find((entry) => entry.slot === slot);
  if (verdict === undefined) throw new Error(`${slot} の採否が無い`);
  return verdict.outcome.geometry;
};

const gemmSlot = (verdicts: readonly SlotVerdict[], slot: ProfileSlot): GemmGeometry => {
  const geometry = slotGeometry(verdicts, slot);
  if ("tileK" in geometry) throw new Error(`${slot} に i8a8 の幾何が入っている`);
  return geometry;
};

const i8a8Slot = (verdicts: readonly SlotVerdict[], slot: ProfileSlot): I8a8Geometry => {
  const geometry = slotGeometry(verdicts, slot);
  if (!("tileK" in geometry)) throw new Error(`${slot} に f32 骨格の幾何が入っている`);
  return geometry;
};

/**
 * 採否から表の値を組む（{@link renderProfileSource} の値と、`acquireGpu({ geometryProfile })` に
 * 注入する値の両方の正本）。`provenance` は掃引の path / sha256 / 日付を渡した順に `", "` で連結する。
 */
export const buildGeometryProfile = (
  spec: { readonly id: string } & ProfileTarget,
  sources: readonly SweepSource[],
  verdicts: readonly SlotVerdict[],
): GeneratedProfile => {
  const lastRows = ROWS_BUCKETS[ROWS_BUCKETS.length - 1];
  // MUST: 末尾の規則は全行数に当たること（runtime の門と同じ条件 — 生成時に先に落とす）
  if (lastRows !== Number.POSITIVE_INFINITY) {
    throw new Error(`runtime の行数バケットの末尾が Infinity でない（${lastRows}）`);
  }
  const joined = (pick: (source: SweepSource) => string): string => sources.map(pick).join(", ");
  return {
    id: spec.id,
    ...(spec.optIn === true ? {} : {
      match: {
        vendor: spec.vendor,
        ...(spec.architecture === undefined ? {} : { architecture: spec.architecture }),
        ...(spec.description === undefined ? {} : { description: spec.description }),
      },
    }),
    gemmRows: ROWS_SLOTS.map((slot, index) => ({
      maxRows: ROWS_BUCKETS[index],
      geometry: gemmSlot(verdicts, slot),
    })),
    attention: {
      qk: gemmSlot(verdicts, "attention.qk"),
      pv: gemmSlot(verdicts, "attention.pv"),
    },
    conv2d: {
      rows64: gemmSlot(verdicts, "conv2d.rows64"),
      rows32: gemmSlot(verdicts, "conv2d.rows32"),
    },
    i8a8: {
      linear: i8a8Slot(verdicts, "i8a8.linear"),
      attentionQk: i8a8Slot(verdicts, "i8a8.attentionQk"),
      attentionPv: i8a8Slot(verdicts, "i8a8.attentionPv"),
    },
    provenance: {
      sweep: joined((source) => source.path),
      sha256: joined((source) => source.sha256),
      date: joined((source) => source.date),
      adapter: adapterLabel(sources),
    },
  };
};

/** {@link profileJson} が Infinity の位置に一時的に置く印（表の文字列に NUL は現れない）。 */
const INFINITY_MARK = "\u0000karume-infinity\u0000";

/**
 * 注入に使う表の JSON（`JSON.parse` の結果をそのまま `acquireGpu({ geometryProfile })` へ渡せる）。
 *
 * `gemmRows` の末尾の `maxRows`（Infinity）は JSON の値に無いので `1e999` と書く — `JSON.parse` は
 * 範囲外の数を Infinity に読む。素の `JSON.stringify` は Infinity を null にし、null の表は runtime の門
 * （最後の規則は Infinity）で落ちる。
 */
export const profileJson = (profile: GeometryProfile): string => infinityJson(profile);

/**
 * `JSON.stringify(value, null, 2)` と同じ形で、`Infinity` だけを `1e999` と書く（{@link profileJson}
 * と同じ綴り — 注入した表を載せる記録〈GPU lab の Anima の JSON〉も表の値を null に落とさない）。
 */
export const infinityJson = (value: unknown): string => {
  const plain = JSON.stringify(value);
  if (plain.includes(JSON.stringify(INFINITY_MARK).slice(1, -1))) {
    throw new Error("JSON の文字列に Infinity の印と同じ綴りが含まれている");
  }
  return JSON.stringify(
    value,
    (_key, entry: unknown) => entry === Number.POSITIVE_INFINITY ? INFINITY_MARK : entry,
    2,
  ).replaceAll(JSON.stringify(INFINITY_MARK), "1e999");
};

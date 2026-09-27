/**
 * 掃引の記録（`karume-geometry-sweep/2` — `report.ts`）から、adapter 1 種ぶんの**幾何プロファイル**
 * （runtime の `src/kernels/geometry-profile.ts` の `GeometryProfile`）を TS の生成物として書く
 * （perf-ledger K-71・`main.ts profile`）。
 *
 * 幾何の選択は runtime では「shape × adapter の静的な表」で、実行時オートチューンは禁止のまま
 * （ADR 0022 決定 3 の MUST）。この道具は**明示のチューニング（掃引）で測った結果をソースへ焼き込む
 * 側**で、runtime は生成物を adapter の (vendor, architecture) で 1 本選ぶだけ — 実行時には測らない。
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
 * 入力の門（ADR 0115 §4）: GPU の timestamp で測った掃引（`gpuTiming.unit` が `ns` か
 * `deno-raw-tick`、かつ `gpuTiming.quantized` が false）だけを受ける。壁時計と 100 µs 丸めは
 * 幾何どうしの比を 1 へ縮めるので、勝ち負けの判定に使えない。
 *
 * クラスの境界は runtime の既定の表と同じ: linear の行数 ≤ 64 / ≤ 512 / それ以上
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
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
} from "../../packages/runtime/src/kernels/i8a8-geometry.ts";
import { SWEEP_OPS, type SweepOp } from "./cases.ts";
import { conv2dCandidate, gemmCandidate, i8a8Candidate } from "./geometries.ts";
import { REPORT_FORMAT } from "./report.ts";

/** `--min-speedup` の既定（既定比がこれ未満の勝ちは測定の揺れと区別しない）。 */
const DEFAULT_MIN_SPEEDUP = 1.05;

/** 整形に使う設定（cwd に依らず `deno fmt --check` と同じ lineWidth で整形する）。 */
const REPO_CONFIG = new URL("../../deno.json", import.meta.url);

const DECODER = new TextDecoder();

/** プロファイル id（ファイル名 `<id>.ts` と export 名 `<ID を大文字 snake>` の元）。 */
const PROFILE_ID = /^[a-z][a-z0-9]*(-[a-z0-9]+)*$/;

export type ProfileFlags = {
  readonly from: readonly string[];
  readonly id: string;
  readonly vendor: string;
  readonly architecture?: string;
  readonly out: string;
  readonly minSpeedup: number;
  readonly check: boolean;
};

/** `--check` 以外は `--key value` の対（未知のキーは落とす — 綴り違いが既定で走らない）。 */
export const parseProfileFlags = (argv: readonly string[]): ProfileFlags => {
  const from: string[] = [];
  let id: string | undefined;
  let vendor: string | undefined;
  let architecture: string | undefined;
  let out: string | undefined;
  let minSpeedup = DEFAULT_MIN_SPEEDUP;
  let check = false;
  for (let at = 0; at < argv.length; at += 1) {
    const key = argv[at];
    if (key === "--check") {
      check = true;
      continue;
    }
    const value = argv[at + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`引数を読めない: ${key}`);
    }
    at += 1;
    switch (key) {
      case "--from":
        from.push(value);
        break;
      case "--id":
        id = value;
        break;
      case "--vendor":
        vendor = value;
        break;
      case "--architecture":
        architecture = value;
        break;
      case "--out":
        out = value;
        break;
      case "--min-speedup":
        minSpeedup = Number(value);
        // 1 未満を許すと既定より遅い幾何を採りうる（比の門の意味が消える）
        if (!Number.isFinite(minSpeedup) || minSpeedup < 1) {
          throw new Error(`--min-speedup は 1 以上の数（${value}）`);
        }
        break;
      default:
        throw new Error(`引数を読めない: ${key} ${value}`);
    }
  }
  if (from.length === 0) throw new Error("--from <掃引の JSON> が要る（複数可）");
  if (id === undefined || !PROFILE_ID.test(id)) {
    throw new Error(`--id は英小文字始まりの kebab-case（${id ?? "無し"}）`);
  }
  if (vendor === undefined || vendor === "") throw new Error("--vendor が要る");
  if (out === undefined) throw new Error("--out <生成物の .ts> が要る");
  // MUST: ファイル名 = id（index.ts の一覧と runtime の診断が id からファイルを辿れる形を保つ）
  const basename = out.slice(out.lastIndexOf("/") + 1);
  if (basename !== `${id}.ts`) {
    throw new Error(`--out のファイル名は ${id}.ts（${basename}）`);
  }
  return {
    from,
    id,
    vendor,
    ...(architecture === undefined ? {} : { architecture }),
    out,
    minSpeedup,
    check,
  };
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

/** 掃引の記録 1 本（unknown 境界で検査したもの）。 */
export type SweepSource = {
  /** 表示と再生成コマンドに載せる path（リポ直下からの相対）。 */
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly adapter: SweepAdapter;
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
 * NOTE: 境界と既定は `gemm-geometry.ts` から取る（生成物を import しない）。`geometry-profile.ts` は
 * `geometry-profiles/index.ts` 経由で生成物を import するので直接は読まない。ただし生成器も runtime の
 * mod.ts（report.ts → anima-residency/timing.ts）を通じて生成物へ推移的に依存する — 登録済みの生成物が
 * 構文的に壊れると生成器自体が読めなくなるので、その場合は index.ts の登録を外してから再生成する。
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

/** linear の行数（`caseShape` の `M{m} N{n} K{k}`）。 */
const LINEAR_SHAPE = /^M(\d+) N\d+ K\d+$/;
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
 * 行の欄。linear は shape の M・attention は shape の段・conv2d は**そのケースの既定の行の tileM**
 * （本番の m タイルの選択 `conv2dIgemmMTile` が選んだ側 — 掃引が既定として測った幾何が正本）。
 */
const slotOf = (
  row: SweepObservation,
  conv2dDefaultTileM: ReadonlyMap<string, number>,
  where: string,
): ProfileSlot => {
  switch (row.op) {
    case "linear": {
      const matched = LINEAR_SHAPE.exec(row.shape);
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
const formatRatio = (value: number): string => `×${value.toFixed(3)}`;

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
};

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
 */
const judgeCandidate = (
  cases: readonly string[],
  observations: ReadonlyMap<string, readonly SweepObservation[]>,
  minSpeedup: number,
): Eligibility => {
  const perCase: { readonly caseId: string; readonly speedup: number }[] = [];
  for (const caseId of cases) {
    const seen = observations.get(caseId) ?? [];
    if (seen.length === 0) return { eligible: false, reason: `${caseId} で測っていない` };
    const failed = seen.find((row) => row.error !== undefined);
    if (failed !== undefined) {
      return { eligible: false, reason: `${caseId} で失敗（${failed.error}）` };
    }
    if (seen.some((row) => row.identicalToDefault === false)) {
      return { eligible: false, reason: `${caseId} で出力が既定と不一致` };
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
  rows: readonly SweepObservation[],
  minSpeedup: number,
): SlotVerdict => {
  if (rows.length === 0) {
    const { name, geometry } = spec.fallback();
    return {
      slot: spec.slot,
      scope: spec.scope,
      cases: [],
      outcome: { kind: "default", name, geometry, reason: "掃引にこのクラスのケースが無い" },
      rejected: [],
    };
  }
  const cases = [...new Set(rows.map((row) => row.caseId))].sort();
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
    { geometry: Geometry; observations: Map<string, SweepObservation[]> }
  >();
  for (const row of rows) {
    if (row.isDefault || row.geometry === defaultRow.geometry) continue;
    const entry = byName.get(row.geometry) ??
      { geometry: row.geometryParams, observations: new Map<string, SweepObservation[]>() };
    if (geometryKey(entry.geometry) !== geometryKey(row.geometryParams)) {
      throw new Error(`${spec.slot}: 幾何 ${row.geometry} の geometryParams が行ごとに違う`);
    }
    entry.observations.set(row.caseId, [...(entry.observations.get(row.caseId) ?? []), row]);
    byName.set(row.geometry, entry);
  }
  const judged = [...byName.entries()]
    .map(([name, entry]) => ({
      name,
      geometry: entry.geometry,
      eligibility: judgeCandidate(cases, entry.observations, minSpeedup),
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
        reason: `全ケースで出力が一致し ${formatRatio(minSpeedup)} 以上の幾何が無い`,
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
  };
};

/** 掃引の記録（1 本以上・同じ adapter）から欄ごとの採否を出す（純関数）。 */
export const deriveProfile = (
  sources: readonly SweepSource[],
  options: {
    readonly vendor: string;
    readonly architecture?: string;
    readonly minSpeedup: number;
  },
): readonly SlotVerdict[] => {
  if (sources.length === 0) throw new Error("掃引の記録が 1 本も無い");
  // 生成物の gemmRows は 3 段の欄で書く — runtime の段数が変われば欄と境界の対応が崩れる
  if (ROWS_BUCKETS.length !== ROWS_SLOTS.length) {
    throw new Error(
      `runtime の行数バケットが ${ROWS_BUCKETS.length} 段（生成器の欄は ${ROWS_SLOTS.length} 段）`,
    );
  }
  const seen = new Map<string, string>();
  for (const source of sources) {
    const twin = seen.get(source.sha256);
    // 同じ記録を 2 度数えると、そのケースの観測だけが重く効く
    if (twin !== undefined) {
      throw new Error(`同じ掃引を 2 度渡している（${twin} と ${source.path}）`);
    }
    seen.set(source.sha256, source.path);
    // MUST: 別の adapter の実測を混ぜない（表は adapter ごとの答え — B570 と M2 は逆を指す）
    const { vendor, architecture } = source.adapter;
    if (
      vendor !== options.vendor ||
      (options.architecture !== undefined && architecture !== options.architecture)
    ) {
      throw new Error(
        `${source.path}: adapter ${vendor} / ${architecture} が --vendor ${options.vendor}${
          options.architecture === undefined ? "" : ` --architecture ${options.architecture}`
        } と合わない`,
      );
    }
    const first = sources[0].adapter;
    if (vendor !== first.vendor || architecture !== first.architecture) {
      throw new Error(
        `${source.path}: adapter ${vendor} / ${architecture} が ${
          sources[0].path
        } の ${first.vendor} / ${first.architecture} と違う`,
      );
    }
  }
  const bySlot = new Map<ProfileSlot, SweepObservation[]>();
  for (const source of sources) {
    const conv2dDefaultTileM = new Map(
      source.rows
        .filter((row) => row.op === "conv2d" && row.isDefault)
        .map((row) => [row.caseId, gemmTileM(row.geometryParams)] as const),
    );
    for (const row of source.rows) {
      const slot = slotOf(row, conv2dDefaultTileM, source.path);
      bySlot.set(slot, [...(bySlot.get(slot) ?? []), row]);
    }
  }
  return SLOTS.map((spec) => judgeSlot(spec, bySlot.get(spec.slot) ?? [], options.minSpeedup));
};

/** `apple-metal-3` → `APPLE_METAL_3`。 */
const profileConstName = (id: string): string => id.toUpperCase().replaceAll("-", "_");

/** 採否の一覧（生成物の冒頭コメントと標準出力で同じ行）。 */
const verdictLines = (verdicts: readonly SlotVerdict[]): string[] =>
  verdicts.flatMap((verdict) => {
    const head = `${verdict.slot}（${verdict.scope}・${verdict.cases.length} ケース）`;
    const { outcome } = verdict;
    const line = outcome.kind === "adopted"
      ? `- ${head}: 採用 ${outcome.name} ${formatRatio(outcome.geomean)}（${
        formatRatio(outcome.min)
      }〜${formatRatio(outcome.max)}）`
      : `- ${head}: 既定 ${outcome.name} のまま（${outcome.reason}）`;
    return [line, ...verdict.rejected.map(({ name, reason }) => `  - ${name}: ${reason}`)];
  });

/** シェルにそのまま貼れる形（空白・記号を含む語だけ単引用符で包む）。 */
const shellWord = (word: string): string =>
  /^[A-Za-z0-9_./:=@%+-]+$/.test(word) ? word : `'${word.replaceAll("'", `'\\''`)}'`;

/** 再生成コマンド（`--check` 抜き・行継続つきの複数行）。 */
const regenerateCommand = (flags: ProfileFlags): string[] => [
  "deno run -A tools/geometry-sweep/main.ts profile \\",
  ...flags.from.map((path) => `  --from ${shellWord(path)} \\`),
  `  --id ${shellWord(flags.id)} --vendor ${shellWord(flags.vendor)}${
    flags.architecture === undefined ? "" : ` --architecture ${shellWord(flags.architecture)}`
  } \\`,
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
  flags: ProfileFlags,
  sources: readonly SweepSource[],
  verdicts: readonly SlotVerdict[],
): string => {
  const value = (slot: ProfileSlot): string => {
    const verdict = verdicts.find((entry) => entry.slot === slot);
    if (verdict === undefined) throw new Error(`${slot} の採否が無い`);
    return renderGeometry(verdict.outcome.geometry, slot);
  };
  const adapter = adapterLabel(sources);
  const match = flags.architecture === undefined
    ? `{ vendor: ${JSON.stringify(flags.vendor)} }`
    : `{ vendor: ${JSON.stringify(flags.vendor)}, architecture: ${
      JSON.stringify(flags.architecture)
    } }`;
  const joined = (pick: (source: SweepSource) => string): string =>
    JSON.stringify(sources.map(pick).join(", "));
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
      `tools/geometry-sweep の \`profile\` が掃引の記録から書いた、adapter \`${flags.vendor}${
        flags.architecture === undefined ? "" : ` / ${flags.architecture}`
      }\` 用のタイル幾何の`,
      "静的な表（perf-ledger K-71）。runtime は adapter の (vendor, architecture) でこの表を選ぶだけで、",
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
      "測っていれば、比はその観測の幾何平均。gemmRows は linear の実測で決め、同じ骨格の matmul / bmm にも効く。",
      "",
      "採否:",
      "",
      ...verdictLines(verdicts),
    ]),
    " */",
    'import type { GeometryProfile } from "../geometry-profile.ts";',
    "",
    `export const ${profileConstName(flags.id)}: GeometryProfile = {`,
    `id: ${JSON.stringify(flags.id)},`,
    `match: ${match},`,
    "gemmRows: [",
    `{ maxRows: ${ROWS_BUCKETS[0]}, geometry: ${value("gemmRows[0]")} },`,
    `{ maxRows: ${ROWS_BUCKETS[1]}, geometry: ${value("gemmRows[1]")} },`,
    `{ maxRows: Number.POSITIVE_INFINITY, geometry: ${value("gemmRows[2]")} },`,
    "],",
    `attention: { qk: ${value("attention.qk")}, pv: ${value("attention.pv")} },`,
    `conv2d: { rows64: ${value("conv2d.rows64")}, rows32: ${value("conv2d.rows32")} },`,
    `i8a8: { linear: ${value("i8a8.linear")}, attentionQk: ${
      value("i8a8.attentionQk")
    }, attentionPv: ${value("i8a8.attentionPv")} },`,
    `provenance: { sweep: ${joined((source) => source.path)}, sha256: ${
      joined((source) => source.sha256)
    }, date: ${joined((source) => source.date)}, adapter: ${JSON.stringify(adapter)} },`,
    "};",
    "",
  ].join("\n");
};

/**
 * `deno fmt` を通す（リポの設定で — 生成物が `deno fmt --check` を通り、生成 → 再生成でバイト同一に
 * なる形）。
 */
export const formatTypeScript = async (source: string): Promise<string> => {
  const child = new Deno.Command(Deno.execPath(), {
    args: ["fmt", "--config", decodeURIComponent(REPO_CONFIG.pathname), "--ext=ts", "-"],
    stdin: "piped",
    stdout: "piped",
    stderr: "piped",
  }).spawn();
  const writer = child.stdin.getWriter();
  await writer.write(new TextEncoder().encode(source));
  await writer.close();
  const output = await child.output();
  if (!output.success) {
    throw new Error(`deno fmt が生成物を整形できない: ${DECODER.decode(output.stderr)}`);
  }
  return DECODER.decode(output.stdout);
};

/**
 * 表示と再生成コマンドに載せる path。cwd（リポ直下）の下の絶対 path は相対にし、先頭の `./` を
 * 落とす — 同じファイルを別の綴りで渡しても生成物が同じバイトになるように。
 */
export const displayPath = (path: string, cwd: string): string => {
  const prefix = cwd.endsWith("/") ? cwd : `${cwd}/`;
  const relative = path.startsWith(prefix) ? path.slice(prefix.length) : path;
  return relative.replace(/^(\.\/)+/, "");
};

const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");

/** 掃引の記録を読む（path は {@link displayPath} で正規化したもの）。 */
const readSweepSource = async (path: string): Promise<SweepSource> => {
  const bytes = await Deno.readFile(path);
  let parsed: unknown;
  try {
    parsed = JSON.parse(DECODER.decode(bytes));
  } catch (cause) {
    const reason = cause instanceof Error ? cause.message : String(cause);
    throw new Error(`${path}: JSON として読めない（${reason}）`);
  }
  return parseSweepReport(parsed, { path, sha256: await sha256Hex(bytes) });
};

/**
 * 既存の生成物と再生成の差の要約（共通の先頭行と末尾行を除いた 1 塊を `-` / `+` で出す —
 * 採否の変化はコメントの採否の行か値の行に現れる）。
 */
const summarizeDifference = (
  before: string,
  after: string,
  limit = 20,
): string[] => {
  const left = before.split("\n");
  const right = after.split("\n");
  let head = 0;
  while (head < left.length && head < right.length && left[head] === right[head]) head += 1;
  let tail = 0;
  while (
    tail < left.length - head && tail < right.length - head &&
    left[left.length - 1 - tail] === right[right.length - 1 - tail]
  ) {
    tail += 1;
  }
  const removed = left.slice(head, left.length - tail);
  const added = right.slice(head, right.length - tail);
  const clip = (lines: readonly string[], mark: string): string[] =>
    lines.length > limit
      ? [
        ...lines.slice(0, limit).map((line) => `${mark} ${line}`),
        `${mark} …（ほか ${lines.length - limit} 行）`,
      ]
      : lines.map((line) => `${mark} ${line}`);
  return [
    `@@ ${head + 1} 行目から（既存 ${removed.length} 行 → 再生成 ${added.length} 行）`,
    ...clip(removed, "-"),
    ...clip(added, "+"),
  ];
};

const readTextOrUndefined = async (path: string): Promise<string | undefined> => {
  try {
    return await Deno.readTextFile(path);
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return undefined;
    throw cause;
  }
};

/**
 * `geometry-profiles/index.ts` の一覧（別の担当の手書き — 生成器は書き換えない）に載っているかの
 * 案内。載っていなければ足す行を出す。
 */
const indexHint = async (flags: ProfileFlags): Promise<string[]> => {
  const directory = flags.out.slice(0, Math.max(0, flags.out.lastIndexOf("/"))) || ".";
  const indexPath = `${directory}/index.ts`;
  const index = await readTextOrUndefined(indexPath);
  const name = profileConstName(flags.id);
  if (index?.includes(`"./${flags.id}.ts"`) === true) {
    return [`[geometry-profile] ${indexPath} には ${name} が既に載っている`];
  }
  return [
    `[geometry-profile] ${indexPath} の BUILTIN_GEOMETRY_PROFILES にまだ載っていない — 次を足す:`,
    `  import { ${name} } from "./${flags.id}.ts";`,
    `  （BUILTIN_GEOMETRY_PROFILES の配列に ${name} を加える）`,
  ];
};

/** `main.ts profile …` の本体。終了コードを返す（`--check` の不一致は 1）。 */
export const runProfileCommand = async (argv: readonly string[]): Promise<number> => {
  const parsed = parseProfileFlags(argv);
  const cwd = Deno.cwd();
  const flags: ProfileFlags = {
    ...parsed,
    from: parsed.from.map((path) => displayPath(path, cwd)),
    out: displayPath(parsed.out, cwd),
  };
  const sources = await Promise.all(flags.from.map(readSweepSource));
  const verdicts = deriveProfile(sources, flags);
  const generated = await formatTypeScript(renderProfileSource(flags, sources, verdicts));
  if (flags.check) {
    const existing = await readTextOrUndefined(flags.out);
    if (existing === generated) {
      console.log(`[geometry-profile] ${flags.out} は再生成とバイト同一`);
      return 0;
    }
    console.log(
      existing === undefined
        ? `[geometry-profile] ${flags.out} が無い`
        : `[geometry-profile] ${flags.out} が再生成と違う:\n${
          summarizeDifference(existing, generated).join("\n")
        }`,
    );
    return 1;
  }
  for (const line of verdictLines(verdicts)) console.log(line);
  await Deno.mkdir(flags.out.slice(0, Math.max(0, flags.out.lastIndexOf("/"))) || ".", {
    recursive: true,
  });
  await Deno.writeTextFile(flags.out, generated);
  console.log(`\n[geometry-profile] ${flags.out}`);
  for (const line of await indexHint(flags)) console.log(line);
  return 0;
};

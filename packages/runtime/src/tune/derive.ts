/**
 * 掃引の記録（`karume-geometry-sweep/2` — `report.ts`）から、adapter 1 種ぶんの**幾何プロファイル**
 * （runtime の `src/kernels/geometry-profile.ts` の `GeometryProfile`）を導く純関数（perf-ledger K-71）。
 * Deno の API に依らない — 公開面 `@karume/runtime/tune` の {@link deriveGeometryProfile}（利用者アプリ）・
 * CLI（`tools/geometry-sweep` の `main.ts profile`）・ブラウザのページ（`tools/gpu-lab` のプロファイルタブ）が
 * 同じ規則で表を作る（別々に持つと、ページやアプリで作った表と CLI の生成物が黙ってずれる）。リポへ登録する
 * 生成物の TS の描画は道具側（`tools/geometry-sweep/render.ts`）に置く（ADR 0117 決定 1）。
 *
 * 幾何の選択は runtime では「shape × adapter の静的な表」で、実行時オートチューンは禁止のまま
 * （ADR 0022 決定 3 の MUST）。この生成器は**明示のチューニング（掃引）で測った結果を表にする側**で、
 * runtime は埋め込みの表を adapter の (vendor, architecture, description) で 1 本選ぶか、注入された表を
 * 使うだけ — acquire / Session の経路では測らない（ADR 0117 決定 9）。`match` を省いた表（`--opt-in`）は
 * 自動選択されない注入専用の表になる。利用者が自分の掃引から作った表は `acquireGpu({ geometryProfile })`
 * で注入できる（{@link geometryProfileJson}）。
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
 * 入力の門（ADR 0115 §4 を ADR 0117 決定 3 で改定）: GPU の timestamp で測った掃引（`gpuTiming.unit` が
 * `ns` か `deno-raw-tick`）だけを受ける。壁時計は submit → 完了の床を含み、その誤差は pass ごとに動くので
 * 記録だけから上界を出せない。timestamp の丸め（Chrome のフラグ無しの 100 µs）は掃引ごと拒まず、観測
 * （掃引 1 本の中の 1 行）ごとに比の誤差の上界 E = e(行) + e(既定の行)（e = 刻み ÷ 最小の round）を出し、
 * {@link ROUNDING_ERROR_LIMIT} を超える観測をその掃引の比の材料から外す（{@link roundingBound}）。
 * 既定の行の e が超えるケースは全観測を外す。外した観測は採否の行に E の値つきで残す。
 *
 * クラスの境界: linear / matmul / bmm は行数 M の段（境界は掃引の形状表の `PROFILE_GEMM_ROWS_BOUNDS` =
 * ≤ 16 / 17〜32 / 33〜64 / 65〜128 / 129〜256 / 257〜512 / > 512 — 既定の表 `GEMM_ROWS_BUCKETS` の
 * 64 / 512 / ∞ を細分した 7 段・ADR 0116）、融合 attention の ①QK / ③PV、conv2d の m タイル 64 / 32 行
 * （既定の行の tileM）、i8a8 の linear / ①QK / ③PV。掃引にケースが無いクラスは runtime の既定を
 * そのまま書く（理由を生成物のコメントに残す）。gemmRows の段の既定は、その段の範囲を覆う既定の表の
 * 段の幾何（{@link profileRowsSegments}）。
 */
import {
  defaultGemmGeometry,
  GEMM_ROWS_BUCKETS,
  type GemmGeometry,
  gemmTileM,
} from "../kernels/gemm-geometry.ts";
import { gemmMTileGeometry } from "../kernels/gemm.ts";
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import { defaultI8a8Geometry, type I8a8Geometry } from "../kernels/i8a8-geometry.ts";
import { PROFILE_GEMM_ROWS_BOUNDS, SWEEP_OPS, type SweepOp } from "./cases.ts";
import {
  geometryProfileKernelsId,
  INFINITY_JSON,
  KernelsIdError,
  sweepCaseSetId,
} from "./fingerprint.ts";
import {
  CANDIDATE_SETS,
  type CandidateSet,
  conv2dCandidate,
  gemmCandidate,
  i8a8Candidate,
  isCandidateSet,
} from "./geometries.ts";
import { CHROME_TIMESTAMP_QUANTUM_NS, driftOutOfRange, REPORT_FORMAT } from "./report.ts";

/** `--min-speedup` の既定（既定比がこれ未満の勝ちは測定の揺れと区別しない）。 */
export const DEFAULT_MIN_SPEEDUP = 1.05;

/**
 * 観測の比の丸め誤差の上界 E のしきい値（ADR 0117 決定 3）。採用の閾値 ×1.05 の余地（5%）の 1/5、
 * 再測定比の許容幅（0.9〜1.1）の 1/10。80 ms の pass なら E は約 0.25% で、超えるのは reps の見積りが
 * 外れた短い pass だけ。規則は 1 つ — 引数にも options にもしない（除外を無効にする口も作らない）。
 */
export const ROUNDING_ERROR_LIMIT = 0.01;

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

type Geometry = GemmGeometry | I8a8Geometry;

/** 掃引した adapter（記録の `adapter` — `GPUAdapterInfo` の 4 欄・空文字も値のまま）。 */
type SweepAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
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
  /**
   * 計測 round ごとの pass の時間（単位は `gpuTiming.unit`・負だった round は 0 に丸めて残る — report.ts）。
   * 丸め誤差の上界（{@link roundingBound}）だけが読む。
   */
  readonly rounds?: readonly number[];
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
  /** 候補集合（`settings.candidateSet` — 欄の無い古い記録は `settings.quick` から `quick` / `full`）。 */
  readonly candidateSet: CandidateSet;
  /**
   * timestamp の量子化の刻み q（`rounds` と同じ単位）。`gpuTiming.quantized` が true なら Chrome の 100 µs、
   * false なら 0（丸めが無い）。
   */
  readonly timestampQuantum: number;
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

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value > 0;

/** `rounds` の形（report.ts — 負だった round は 0 に丸めて残すので、負の値は記録の形として壊れている）。 */
const isRounds = (value: unknown): value is readonly number[] =>
  Array.isArray(value) &&
  value.every((entry) => typeof entry === "number" && Number.isFinite(entry) && entry >= 0);

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
  const rounds = optional(value, "rounds", isRounds, where);
  // reps は検査だけ: 丸め誤差の上界 e = q ÷ min(round) が perDispatch（= min ÷ reps）の相対誤差の上界に
  // なるのは、reps が整数で誤差を持たないから（ADR 0117 決定 3）。値そのものは上界に効かない
  optional(value, "reps", isPositiveInteger, where);
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
    ...(rounds === undefined ? {} : { rounds }),
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
 * 記録の候補集合（表の `provenance.candidateSet` に載る — ADR 0117 決定 4）。`settings.candidateSet` が
 * 正本で、欄の無い古い記録（quick+ の導入前）は `settings.quick` から `quick` / `full` を読む。
 */
const parseCandidateSet = (settings: unknown, where: string): CandidateSet => {
  if (!isRecord(settings)) throw new Error(`${where}: settings が無い（候補集合が読めない）`);
  const { candidateSet, quick } = settings;
  if (candidateSet !== undefined) {
    if (typeof candidateSet !== "string" || !isCandidateSet(candidateSet)) {
      throw new Error(
        `${where}: settings.candidateSet が ${CANDIDATE_SETS.join(" / ")} のどれでもない（${
          JSON.stringify(candidateSet)
        }）`,
      );
    }
    return candidateSet;
  }
  if (!isBoolean(quick)) {
    throw new Error(`${where}: settings に candidateSet も quick も無い（候補集合が読めない）`);
  }
  return quick ? "quick" : "full";
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
  // MUST: GPU の timestamp で測った掃引だけを受ける（ADR 0115 §4・ADR 0117 決定 3）。壁時計は
  // submit → 完了の床を含むので幾何どうしの比が 1 へ縮み、床は pass ごとに動くので丸めのような
  // 既知の定数で誤差を押さえられない（行ごとの上界を記録だけから出せない）
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
  // 量子化（Chrome の 100 µs 丸め）は掃引ごとは拒まない — 丸めの刻み q を観測ごとの誤差の上界に使う
  // （deriveProfile・ADR 0117 決定 3）
  const quantized = gpuTiming.quantized;
  if (!isBoolean(quantized)) {
    throw new Error(
      `${where}: gpuTiming.quantized が真偽値でない（${JSON.stringify(quantized)}）`,
    );
  }
  // 書き手（report.ts の roundsLookQuantized）は単位 ns でだけ量子化を判定する。raw tick に ns の刻みを
  // 当てると上界が別物になるので、組が崩れた記録は落とす
  if (quantized && unit !== "ns") {
    throw new Error(
      `${where}: gpuTiming.quantized が true なのに単位が ${unit}（ns でだけ判定する）`,
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
      device: requireString(adapter, "device", `${where} adapter`),
      description: requireString(adapter, "description", `${where} adapter`),
    },
    candidateSet: parseCandidateSet(parsed.settings, where),
    timestampQuantum: quantized ? CHROME_TIMESTAMP_QUANTUM_NS : 0,
    cases,
    rows,
  };
};

/**
 * gemmRows の段の欄名。段を行数の範囲で綴る（`gemmRows ≤ 16`・`gemmRows 17〜32`・`gemmRows > 512` —
 * {@link profileRowsSegments}）。添字（`gemmRows[1]`）で綴らないのは、3 段時代の `gemmRows[1]`（65〜512）と
 * 7 段の 2 番目（17〜32）が同じ綴りで別の範囲を指すから（ADR 0116 決定 6）。
 */
export type GemmRowsSlot = `gemmRows ${string}`;

/** プロファイルの欄（= クラス）。生成物のコメントでもこの綴りで欄を指す。 */
export type ProfileSlot =
  | GemmRowsSlot
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

/** gemmRows の段 1 つ（{@link profileRowsSegments}）。 */
export type RowsSegment = {
  readonly slot: GemmRowsSlot;
  /** 欄が効く範囲（コメント用）。 */
  readonly scope: string;
  readonly maxRows: number;
  /**
   * 段の範囲を覆う既定の表（`GEMM_ROWS_BUCKETS`）の段の幾何。採用が無い・掃引にケースが無い段に書く値で、
   * 掃引の既定の行（掃引は既定の表 `gemmGeometryForRows(m)` で組む）と突き合わせる比の土台でもある。
   */
  readonly fallback: GemmGeometry;
};

/** 段 `index` の行数の範囲の綴り（`≤ 16`・`17〜32`・`> 512`）。境界は検査済みの列（正整数・狭義昇順・末尾 Infinity）。 */
export const rowsRange = (bounds: readonly number[], index: number): string => {
  const maxRows = bounds[index];
  if (index === 0) return `≤ ${maxRows}`;
  const previous = bounds[index - 1];
  return maxRows === Number.POSITIVE_INFINITY ? `> ${previous}` : `${previous + 1}〜${maxRows}`;
};

/**
 * gemmRows の段の境界（`maxRows` の列）を検査して段の一覧にする純関数（ADR 0116 決定 3）。生成器は
 * 掃引の形状表の `PROFILE_GEMM_ROWS_BOUNDS` を渡す（境界の定数は 1 か所 — ここに複製しない）。
 *
 * MUST: 境界は末尾が Infinity・末尾以外は正整数で狭義昇順・既定の表（`GEMM_ROWS_BUCKETS`）の `maxRows` を
 * 全て含む細分であること。満たさなければ生成を止める（fail loudly）。細分でない段は、掃引の既定の行の幾何が
 * 段の中で 2 種類に割れ（比の土台が 2 つになる）、段の既定（fallback）も 1 つに決まらないから。
 * 段の既定は、その段の範囲を覆う既定の段の幾何（細分なので段の全行数が同じ既定の段に入る）。
 *
 * NOTE: 既定は `gemm-geometry.ts` から取る（生成物の値を既定の正本にしない）。ただし生成器は
 * 生成物へ依存している — 候補集合 `quick+`（geometries.ts が `geometry-profiles/index.ts` を読む）を
 * 通じて。登録済みの生成物が構文的に壊れると生成器自体が読めなくなるので、その場合は index.ts の登録を
 * 外してから再生成する。
 */
export const profileRowsSegments = (bounds: readonly number[]): readonly RowsSegment[] => {
  const where = `gemmRows の段の境界 [${bounds.join(", ")}]`;
  if (bounds.at(-1) !== Number.POSITIVE_INFINITY) {
    throw new Error(`${where}: 末尾が Infinity でない（それより大きい M に当たる段が無い）`);
  }
  bounds.slice(0, -1).forEach((maxRows, index) => {
    if (!Number.isSafeInteger(maxRows) || maxRows < 1) {
      throw new Error(`${where}: 末尾以外の境界は正整数（${maxRows}）`);
    }
    if (!(maxRows < bounds[index + 1])) {
      throw new Error(
        `${where}: 狭義昇順でない（${maxRows} → ${
          bounds[index + 1]
        } — 後ろの段に当たる行数が無くなる）`,
      );
    }
  });
  const missing = GEMM_ROWS_BUCKETS
    .map((rule) => rule.maxRows)
    .filter((maxRows) => !bounds.includes(maxRows));
  if (missing.length > 0) {
    throw new Error(
      `${where}: 既定の表の境界 ${missing.join(" / ")} を含まない（既定の細分でない）— ` +
        "既定の境界をまたぐ段は、段の中で掃引の既定の行の幾何が割れ、段の既定も 1 つに決まらない",
    );
  }
  return bounds.map((maxRows, index) => {
    const range = rowsRange(bounds, index);
    const covering = GEMM_ROWS_BUCKETS.find((rule) => maxRows <= rule.maxRows) ??
      GEMM_ROWS_BUCKETS[GEMM_ROWS_BUCKETS.length - 1];
    return {
      slot: `gemmRows ${range}`,
      scope: `linear / matmul / bmm の行数 ${range}`,
      maxRows,
      fallback: covering.geometry,
    };
  });
};

/** gemmRows 以外の欄（生成物の欄の順・コメントの採否の順で、gemmRows の段の後ろに並ぶ）。 */
const FIXED_SLOTS: readonly SlotSpec[] = [
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

/** 欄の一覧（生成物の欄の順・コメントの採否の順 — gemmRows の段 → それ以外）。 */
const profileSlots = (segments: readonly RowsSegment[]): readonly SlotSpec[] => [
  ...segments.map((segment) => ({
    slot: segment.slot,
    scope: segment.scope,
    fallback: () => gemmCandidate(segment.fallback),
  })),
  ...FIXED_SLOTS,
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
 * 行の欄。linear / matmul / bmm は shape の M が `rows <= maxRows` で最初に当たる段（本番の 3 経路が同じ gemmRows の
 * 表を M で引く規則と同じ — src/runtime/recipe-builders/linear.ts・geometry-profile.ts の gemmRowsGeometry）・attention は shape の段・conv2d は**そのケースの既定の行の tileM**
 * （本番の m タイルの選択 `conv2dIgemmMTile` が選んだ側 — 掃引が既定として測った幾何が正本）。
 */
const slotOf = (
  row: SweepObservation,
  segments: readonly RowsSegment[],
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
      // 段の列は末尾が Infinity（profileRowsSegments の検査済み）なので必ずどれかに当たる
      return (segments.find((segment) => rows <= segment.maxRows) ??
        segments[segments.length - 1]).slot;
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
  /**
   * このクラスのケース・観測のうち、掃引ごとに比の材料から外したもの（掃引を渡した順 → ケース id の昇順 →
   * ケース単位の除外 → 幾何名の昇順）。ケース単位（再測定比の範囲外・既定の行の丸め誤差）と観測単位
   * （丸め誤差の上界 E > 1% — `geometry` あり）がある。
   */
  readonly excluded: readonly ExcludedCase[];
};

/**
 * 掃引 1 本の中で比の材料から外したケース（{@link exclusionReason}）か、ケースの 1 幾何の観測
 * （`geometry` あり — 丸め誤差の上界が超えた行・{@link roundingBound}）— 出力の一致と失敗は見る。
 */
export type ExcludedCase = {
  readonly path: string;
  readonly caseId: string;
  /** 外したのがケースの 1 幾何の観測だけのとき、その幾何の名前（無ければケースの全観測）。 */
  readonly geometry?: string;
  /** 「既定の再測定比 ×1.160 が範囲外」など（{@link excludedCaseLine} が文にする）。 */
  readonly reason: string;
};

/** 外したケース・観測の 1 行（生成物のコメント・標準出力・ページで同じ文）。 */
export const excludedCaseLine = (excluded: ExcludedCase): string =>
  `掃引 ${excluded.path}: ${excluded.caseId}${
    excluded.geometry === undefined ? "" : ` の ${excluded.geometry}`
  } は${excluded.reason}のため比の材料から外した（出力の一致と失敗は見る）`;

/**
 * ケースをその掃引の比の材料から外す理由（外さないなら undefined）。
 *
 * 既定の再測定比（ケースの末尾で既定をもう 1 度測った値 ÷ 初回）が report.ts の `DEFAULT_DRIFT_RANGE` の外なら、
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

/** 丸め誤差の上界（`bound` = 相対誤差）か、上界を出せない理由。 */
type RoundingBound = { readonly bound: number } | { readonly unknown: string };

/**
 * 行の丸め誤差の上界 e(r) = q ÷ min(rounds)（ADR 0117 決定 3）。timestamp 2 つの差の丸め誤差は 1 刻み
 * 未満なので、e は min の round（= `perDispatch` × reps）の相対誤差の上界になる（1 次の近似）。
 *
 * 0 の round は min の候補にしない — 負だった round は記録に 0 で残り（report.ts）、`perDispatch` の
 * min の候補から外れているから。負でなく丸めで 0 になった round が min なら `perDispatch` が 0 で、その行は
 * 既定比を持たない（report.ts の比較）ので、外しても判定は変わらない。
 */
const roundingBound = (row: SweepObservation, quantum: number): RoundingBound => {
  // 丸めが無ければ誤差も無い — rounds の無い古い記録もここで通る
  if (quantum === 0) return { bound: 0 };
  if (row.rounds === undefined) return { unknown: "rounds が無い" };
  const kept = row.rounds.filter((round) => round > 0);
  if (kept.length === 0) return { unknown: "正の round が無い" };
  return { bound: quantum / Math.min(...kept) };
};

/** 百分率の表示（丸め誤差の上界 — 採否の行の綴り）。 */
const formatPercent = (value: number): string => `${(value * 100).toFixed(2)}%`;

const LIMIT_LABEL = `${ROUNDING_ERROR_LIMIT * 100}%`;

/**
 * 既定の行の丸め誤差の上界 e(d) がしきい値を超える・出せないケースは、そのケースの全観測の E が超える
 * （E = e(r) + e(d) ≥ e(d)）ので、ケースごと外す理由（外さないなら undefined）。
 */
const defaultRoundingReason = (bound: RoundingBound): string | undefined => {
  if ("unknown" in bound) return `既定の行の丸め誤差の上界が不明（${bound.unknown}）`;
  return bound.bound > ROUNDING_ERROR_LIMIT
    ? `既定の行の丸め誤差の上界 e ${formatPercent(bound.bound)}（> ${LIMIT_LABEL}）`
    : undefined;
};

/** 観測の比の丸め誤差の上界 E = e(r) + e(d) がしきい値を超える・出せないとき、外す理由。 */
const rowRoundingReason = (bound: RoundingBound, defaultBound: number): string | undefined => {
  if ("unknown" in bound) return `丸め誤差の上界が不明（${bound.unknown}）`;
  const total = bound.bound + defaultBound;
  return total > ROUNDING_ERROR_LIMIT
    ? `丸め誤差の上界 E ${formatPercent(total)}（> ${LIMIT_LABEL}）`
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
export const targetLabel = (target: ProfileTarget): string =>
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
 *
 * 丸めの門（ADR 0117 決定 3）: 既定の行の丸め誤差の上界 e(d) が 1% を超える（か出せない）ケースも同じくその掃引の
 * 比の観測から外し（{@link defaultRoundingReason}）、それ以外の行は E = e(r) + e(d) が 1% を超える観測だけを外す
 * （{@link rowRoundingReason}）。非量子化（q = 0）なら e = 0 で rounds を見ない。
 */
export const deriveProfile = (
  sources: readonly SweepSource[],
  options: ProfileTarget & { readonly minSpeedup: number },
): readonly SlotVerdict[] => {
  if (sources.length === 0) throw new Error("掃引の記録が 1 本も無い");
  // 段の境界が既定の表の細分でなければここで止める（段の中で比の土台が割れる — profileRowsSegments）
  const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
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
    // MUST: 掃引どうしは adapter の 4 欄が全て一致（空どうしは一致・片方だけ空は不一致 — ADR 0117
    // 決定 4）。表の provenance は adapter を 1 つだけ持ち、照合（geometryProfileMismatch）はその 4 欄で
    // 見る。--opt-in や --description 省略では上の門が description を見ないので、同じ vendor /
    // architecture の別機種（M2 と M5 など）が混ざるのもここで止める
    const first = sources[0].adapter;
    for (const field of ["vendor", "architecture", "device", "description"] as const) {
      if (source.adapter[field] !== first[field]) {
        throw new Error(
          `${source.path}: adapter の ${field} ${JSON.stringify(source.adapter[field])} が ${
            sources[0].path
          } の ${JSON.stringify(first[field])} と違う（4 欄の揃った掃引だけを 1 本の表に合わせる）`,
        );
      }
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
    const defaultBounds = new Map(
      source.rows
        .filter((row) => row.isDefault)
        .map((row) => [row.caseId, { row, bound: roundingBound(row, source.timestampQuantum) }]),
    );
    const excludedHere: { readonly slot: ProfileSlot; readonly excluded: ExcludedCase }[] = [];
    const excludedCases = new Set<string>();
    for (const row of source.rows) {
      const slot = slotOf(row, segments, conv2dDefaultTileM, source.path);
      const base = defaultBounds.get(row.caseId);
      // parseSweepReport がケースごとに既定の行 1 本を検査済み
      if (base === undefined) throw new Error(`${source.path}: ${row.caseId} の既定の行が無い`);
      // ケースごと外す理由（再測定比 → 既定の行の丸め）。既定の行が失敗したケースは比を持たないので
      // 丸めを見ない（比の材料が無い — 失敗は judgeCandidate が見る）
      const caseReason = exclusionReason(repeats.get(row.caseId)) ??
        (base.row.error === undefined ? defaultRoundingReason(base.bound) : undefined);
      // 観測ごとに外す理由（E = e(r) + e(d)）。既定比を持たない行（失敗など）は比の材料でないので見ない
      const rowReason = caseReason !== undefined || row.isDefault ||
          row.speedupVsDefault === undefined || "unknown" in base.bound
        ? undefined
        : rowRoundingReason(roundingBound(row, source.timestampQuantum), base.bound.bound);
      if (caseReason !== undefined && !excludedCases.has(row.caseId)) {
        excludedCases.add(row.caseId);
        excludedHere.push({
          slot,
          excluded: { path: source.path, caseId: row.caseId, reason: caseReason },
        });
      }
      if (rowReason !== undefined) {
        excludedHere.push({
          slot,
          excluded: {
            path: source.path,
            caseId: row.caseId,
            geometry: row.geometry,
            reason: rowReason,
          },
        });
      }
      bySlot.set(slot, [
        ...(bySlot.get(slot) ?? []),
        { row, included: caseReason === undefined && rowReason === undefined },
      ]);
    }
    // ケース id の昇順 → ケースごとの除外が先 → 幾何の名前の昇順
    const order = (excluded: ExcludedCase): string =>
      `${excluded.caseId}\u0000${excluded.geometry ?? ""}`;
    const sorted = [...excludedHere].sort((left, right) => {
      const [a, b] = [order(left.excluded), order(right.excluded)];
      return a < b ? -1 : a > b ? 1 : 0;
    });
    for (const { slot, excluded } of sorted) {
      excludedBySlot.set(slot, [...(excludedBySlot.get(slot) ?? []), excluded]);
    }
  }
  return profileSlots(segments).map((spec) =>
    judgeSlot(
      spec,
      bySlot.get(spec.slot) ?? [],
      excludedBySlot.get(spec.slot) ?? [],
      options.minSpeedup,
    )
  );
};

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
 * 採否から表の値を組む（リポへ登録する生成物の TS〈`tools/geometry-sweep/render.ts`〉の値と、
 * `acquireGpu({ geometryProfile })` に注入する値の両方の正本）。
 *
 * MUST: `provenance` を書くのはこの関数だけ（ADR 0117 決定 4）。掃引の path / sha256 / 日付 / 候補集合は
 * 渡した順に `", "` で連結し、adapter は 4 欄をそのまま（{@link deriveProfile} が全ての記録で一致を検査済み）、
 * カーネルの指紋とケース集合の版は今の runtime で導く（照合 `geometryProfileMismatch` が同じ関数で導き直す）。
 */
export const buildGeometryProfile = (
  spec: { readonly id: string } & ProfileTarget,
  sources: readonly SweepSource[],
  verdicts: readonly SlotVerdict[],
): GeneratedProfile => {
  // MUST: 末尾の規則は全行数に当たること（runtime の門と同じ条件 — 生成時に先に落とす）。
  // 末尾 Infinity と既定の細分は profileRowsSegments が検査する
  const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
  if (sources.length === 0) throw new Error("掃引の記録が 1 本も無い（provenance を書けない）");
  const joined = (pick: (source: SweepSource) => string): string => sources.map(pick).join(", ");
  const table: GeometryProfile = {
    id: spec.id,
    ...(spec.optIn === true ? {} : {
      match: {
        vendor: spec.vendor,
        ...(spec.architecture === undefined ? {} : { architecture: spec.architecture }),
        ...(spec.description === undefined ? {} : { description: spec.description }),
      },
    }),
    gemmRows: segments.map((segment) => ({
      maxRows: segment.maxRows,
      geometry: gemmSlot(verdicts, segment.slot),
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
  };
  const { vendor, architecture, device, description } = sources[0].adapter;
  return {
    ...table,
    provenance: {
      sweep: joined((source) => source.path),
      sha256: joined((source) => source.sha256),
      date: joined((source) => source.date),
      candidateSet: joined((source) => source.candidateSet),
      adapter: { vendor, architecture, device, description },
      kernels: generatedKernelsId(table),
      caseSet: sweepCaseSetId(),
    },
  };
};

/**
 * 生成した表のカーネルの指紋。表の幾何から掃引の shape のカーネルを組めない（codegen の門・dispatch 数の
 * 上限 — 指紋は全 device が保証する 65535 で組むので、上限の大きい device の掃引を通った幾何でも落ちうる）
 * ときは、表を作らずに理由（ケース・欄・幾何）を名指して投げる — 照合で必ず不一致になる表を保存させない。
 */
const generatedKernelsId = (table: GeometryProfile): string => {
  try {
    return geometryProfileKernelsId(table);
  } catch (cause) {
    if (cause instanceof KernelsIdError) {
      throw new Error(
        `生成した表 '${table.id}' の provenance.kernels（カーネルの指紋）を導けない — ${cause.message}`,
        { cause },
      );
    }
    throw cause;
  }
};

/**
 * 公開の生成器（{@link deriveGeometryProfile}）の入力 1 本 — 掃引の記録と、表の `provenance` に載せる識別。
 * `report` は unknown 境界（保存した JSON を `JSON.parse` した値でも `runGeometrySweep` の戻りでもよい —
 * 生成が読む欄を {@link parseSweepReport} で検査する）。
 */
export type GeometrySweepInput = {
  readonly report: unknown;
  /** 記録の名前（ファイルの path など — `provenance.sweep` に載る）。 */
  readonly path: string;
  /** 記録のバイト列の SHA-256（16 進 — `provenance.sha256` に載る。同じ値の記録を 2 本渡すと落ちる）。 */
  readonly sha256: string;
};

/** {@link deriveGeometryProfile} の options（表の id・表を当てる adapter・採用の閾値）。 */
export type DeriveGeometryProfileOptions = ProfileTarget & {
  readonly id: string;
  /** 既定比の採用の閾値（既定 {@link DEFAULT_MIN_SPEEDUP}・1 以上 — CLI の `--min-speedup`）。 */
  readonly minSpeedup?: number;
};

/** {@link deriveGeometryProfile} の戻り（表と採否の行）。 */
export type GeometryProfileDerivation = {
  /** 生成した表（`provenance` 付き — {@link geometryProfileJson} で保存し、`acquireGpu` に注入する）。 */
  readonly profile: GeometryProfile;
  /**
   * 欄ごとの採否・退けた幾何と理由・比の材料から外したケース（{@link verdictLines} — リポの生成物の
   * 冒頭コメントと CLI の標準出力と同じ行）。
   */
  readonly verdicts: readonly string[];
};

/**
 * 掃引の記録 1 本以上から表を作る（公開面 `@karume/runtime/tune` の生成器 — ADR 0117 決定 2）。規則は
 * {@link deriveProfile}（ADR 0115 決定 4・ADR 0116・ADR 0117 決定 3）、表の値は {@link buildGeometryProfile} の 1 本で、
 * CLI と GPU lab が作る表と同じ値になる。門に落ちた入力は理由つきで投げる（fail loudly）。
 */
export const deriveGeometryProfile = (
  reports: readonly GeometrySweepInput[],
  options: DeriveGeometryProfileOptions,
): GeometryProfileDerivation => {
  const minSpeedup = options.minSpeedup ?? DEFAULT_MIN_SPEEDUP;
  // 1 未満を許すと既定より遅い幾何を採りうる（比の門の意味が消える — CLI の --min-speedup と同じ条件）
  if (!Number.isFinite(minSpeedup) || minSpeedup < 1) {
    throw new Error(`minSpeedup は 1 以上の数（${minSpeedup}）`);
  }
  const sources = reports.map(({ report, path, sha256 }) =>
    parseSweepReport(report, { path, sha256 })
  );
  const spec = { ...options, minSpeedup };
  const verdicts = deriveProfile(sources, spec);
  return {
    profile: buildGeometryProfile(spec, sources, verdicts),
    verdicts: verdictLines(verdicts),
  };
};

/** {@link geometryProfileJson} が Infinity の位置に一時的に置く印（表の文字列に NUL は現れない）。 */
const INFINITY_MARK = "\u0000karume-infinity\u0000";

/**
 * 注入に使う表の JSON（`JSON.parse` の結果をそのまま `acquireGpu({ geometryProfile })` へ渡せる）。
 *
 * `gemmRows` の末尾の `maxRows`（Infinity）は JSON の値に無いので `1e999` と書く — `JSON.parse` は
 * 範囲外の数を Infinity に読む。素の `JSON.stringify` は Infinity を null にし、null の表は runtime の門
 * （最後の規則は Infinity）で落ちる。
 */
export const geometryProfileJson = (profile: GeometryProfile): string => infinityJson(profile);

/**
 * `JSON.stringify(value, null, 2)` と同じ形で、`Infinity` だけを `1e999` と書く（{@link geometryProfileJson}
 * と同じ綴り — 注入した表を載せる記録〈GPU lab の Anima の JSON〉も表の値を null に落とさない）。綴りは
 * fingerprint.ts の `INFINITY_JSON`（照合キーの正規の JSON と 1 つ）。
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
  ).replaceAll(JSON.stringify(INFINITY_MARK), INFINITY_JSON);
};

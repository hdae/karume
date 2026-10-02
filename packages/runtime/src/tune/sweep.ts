/**
 * 掃引の入口（公開面 `@karume/runtime/tune` の `runGeometrySweep` — ADR 0117 決定 2）。
 *
 * 専用の device を自分で取って掃引し、終わったら捨てる。呼び手の `GpuContext` を受けないのは、掃引と
 * 推論が同じ device を共有する形を型の上で作れなくするため（決定 9 の MUST を構造で守る）。Deno の CLI
 * （`tools/geometry-sweep/main.ts`）と GPU lab の掃引タブもこれを呼ぶ殻で、device の取り直し・計測・
 * 記録の組み立てはここ 1 本（別々に持つと、2 本の記録が黙ってずれる）。
 *
 * MUST: acquire / Session の経路からは呼ばない — runtime は実行時に幾何を測らない（ADR 0022 追記の
 * オートチューン禁止・ADR 0117 決定 9）。測るのは利用者が明示的に呼んだときだけ。
 */
import { acquireGpu } from "../gpu/device.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../kernels/geometry-profile.ts";
import { SWEEP_CASES, SWEEP_OPS, type SweepCase, type SweepOp } from "./cases.ts";
import {
  CANDIDATE_SETS,
  type CandidateSet,
  DEFAULT_CANDIDATE_SET,
  isCandidateSet,
} from "./geometries.ts";
import {
  geometryProfileKernelsId,
  sweepCandidateKernelsId,
  sweepCaseSetId,
} from "./fingerprint.ts";
import { createSweepContext, destroySweepContext, runSweep, SWEEP_MAX_REPS } from "./harness.ts";
import { ROUNDS, TARGET_PASS_MS, WARMUP_MIN_RUNS, WARMUP_NS } from "./measurement.ts";
import {
  type CaseSummary,
  type Report,
  REPORT_FORMAT,
  type ReportAdapter,
  roundsLookQuantized,
  type SweepRow,
  type TimingUnit,
  WALL_TIMING_NOTE,
} from "./report.ts";

/** adapter の列挙を見る feature 名（`acquireGpu({ gpuTiming: true })` が要求するもの）。 */
const TIMESTAMP_QUERY: GPUFeatureName = "timestamp-query";

/**
 * 掃引の進捗（{@link GeometrySweepOptions.onProgress} に渡す — ケース / 幾何の単位）。
 *
 * - `started` — device を取り、計測の文脈ができた（単位・dp4a・ケース数が決まった）。
 * - `status` — 今どこを測っているか（「<ケース>: <幾何>（i/n）」など — 表示用の 1 行）。
 * - `row` — (ケース, 幾何) の行が 1 つ確定した。ケースの最初の行は既定の幾何（`isDefault`）で、
 *   続く行は既定比（`speedupVsDefault`）と出力の一致（`identicalToDefault`）を持つ。
 * - `case` — ケースを測り終えた（末尾で既定を測り直した比 `defaultRepeat.driftRatio` を持つ）。
 * - `deviceLost` — device を失った（残りの行は失敗の行になる）。
 */
export type GeometrySweepProgress =
  | {
    readonly kind: "started";
    readonly adapter: ReportAdapter;
    readonly unit: TimingUnit;
    readonly dp4a: boolean;
    readonly caseCount: number;
  }
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "row"; readonly row: SweepRow }
  | { readonly kind: "case"; readonly summary: CaseSummary }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string };

/** {@link runGeometrySweep} の options（全て省略可）。 */
export type GeometrySweepOptions = {
  /** `requestAdapter` に渡す（`powerPreference` 等 — `acquireGpu({ adapter })` と同じ）。 */
  readonly adapter?: GPURequestAdapterOptions;
  /** 候補集合（既定 `quick+`）。 */
  readonly candidateSet?: CandidateSet;
  /** 掃引する op 族（既定は全族）。 */
  readonly ops?: readonly SweepOp[];
  /** ケース id で絞る（既定は `ops` の全ケース — CLI の `--case`）。 */
  readonly cases?: readonly string[];
  /** 幾何ごとの計測 round の数（既定 5 — 代表値はその min）。 */
  readonly rounds?: number;
  /**
   * adapter が `timestamp-query` を列挙すれば GPU の timestamp で測る（既定 true）。`false` か、列挙
   * しない adapter では壁時計で測る（単位 `wall` — 記録は残せるが、生成器は表の材料にしない）。
   */
  readonly timestamps?: boolean;
  /** 中断（今の幾何を測り終えたところで止める — 記録は測れた行までを持つ）。 */
  readonly signal?: AbortSignal;
  /** 進捗の通知（{@link GeometrySweepProgress}）。 */
  readonly onProgress?: (progress: GeometrySweepProgress) => void;
};

/** options を検査して既定を埋めた掃引の設定と、選んだケース。 */
type SweepPlan = {
  readonly candidateSet: CandidateSet;
  readonly ops: readonly SweepOp[];
  readonly caseIds: readonly string[];
  readonly rounds: number;
  readonly cases: readonly SweepCase[];
};

const isSweepOp = (value: string): value is SweepOp =>
  (SWEEP_OPS as readonly string[]).includes(value);

/**
 * options を検査してケースを選ぶ（GPU に触れる前 — 綴り違いの集合・op・ケースを既定で走らせない）。
 * ケースは op と id の両方に合うもの（id を渡さなければ op の全ケース）で、並びは形状表の順。
 */
const planSweep = (options: GeometrySweepOptions): SweepPlan => {
  const candidateSet = options.candidateSet ?? DEFAULT_CANDIDATE_SET;
  if (!isCandidateSet(candidateSet)) {
    throw new Error(`候補集合 ${candidateSet} は ${CANDIDATE_SETS.join(" / ")} のどれでもない`);
  }
  const ops = options.ops ?? SWEEP_OPS;
  const unknownOp = ops.find((op) => !isSweepOp(op));
  if (unknownOp !== undefined) {
    throw new Error(`op ${unknownOp} は ${SWEEP_OPS.join(" / ")} のどれでもない`);
  }
  const caseIds = options.cases ?? [];
  const unknownCase = caseIds.find((id) => !SWEEP_CASES.some((sweepCase) => sweepCase.id === id));
  if (unknownCase !== undefined) throw new Error(`ケース ${unknownCase} が形状表に無い`);
  const rounds = options.rounds ?? ROUNDS;
  if (!Number.isInteger(rounds) || rounds < 1) {
    throw new Error(`rounds は 1 以上の整数（${rounds}）`);
  }
  const cases = SWEEP_CASES.filter((sweepCase) =>
    ops.includes(sweepCase.op) && (caseIds.length === 0 || caseIds.includes(sweepCase.id))
  );
  if (cases.length === 0) {
    throw new Error(`op [${ops.join(", ")}] とケース [${caseIds.join(", ")}] に合うケースが無い`);
  }
  return { candidateSet, ops, caseIds, rounds, cases };
};

/**
 * timestamp の差の単位。Deno の WebGPU（wgpu）は timestamp を ns に換算せず raw tick のまま返す
 * （docs/known-issues.md「Intel Arc B570」節 — B570 で 1 tick = 52.08 ns）ので `deno-raw-tick`、それ以外
 * （Chrome / Dawn）は仕様どおり `ns`。実装の判別は `navigator.userAgent`（Deno は `Deno/<版>` を返す）。
 */
export const timestampUnitFor = (userAgent: string): "ns" | "deno-raw-tick" =>
  userAgent.startsWith("Deno/") ? "deno-raw-tick" : "ns";

/**
 * 専用の device を取ってケースを掃引し、記録（`karume-geometry-sweep/2`）を返す。device は終わったら
 * （中断・失敗でも）捨てる。行ごとの失敗（確保・コンパイル・device lost）は記録の行に残して掃引を続け、
 * options の誤りと device が取れない環境は GPU に触れる前 / 取得の段で投げる。
 *
 * MUST: 推論と並走させない（測定が汚れ、GPU も取り合う）— runtime は別の device の使われ方を見られない
 * ので、これは呼び手との契約（公開面の doc に書く）。
 */
export const runGeometrySweep = async (
  options: GeometrySweepOptions = {},
): Promise<Report> => {
  const plan = planSweep(options);
  // 開始時刻は device を取る前（`date` は記録を組む時点 = 終了時刻 — 差が所要・ADR 0117 決定 8）
  const startedAt = new Date().toISOString();
  const notify = options.onProgress;
  // timestamp-query は adapter が列挙すれば要求する（決定 2）。acquireGpu の gpuTiming は「必須」か
  // 「要求しない」で「あれば使う」を持たないので、列挙を見るためだけに adapter を 1 度取って捨てる。
  // NOTE: requestAdapter を 2 回呼んで同じ物理 adapter が選ばれる保証は仕様に無い — 選び直された
  // adapter が timestamp-query を持たなければ acquireGpu が GpuFeatureError で落ちる（黙って壁時計に
  // 落とさない）。navigator.gpu が無い環境は acquireGpu が GpuUnavailableError で落とす。
  const webgpu: GPU | undefined = navigator.gpu;
  const probe = options.timestamps === false || webgpu === undefined
    ? null
    : await webgpu.requestAdapter(options.adapter);
  const timestamps = probe?.features.has(TIMESTAMP_QUERY) === true;
  let deviceLost: Report["deviceLost"] = null;
  const gpu = await acquireGpu({
    ...(options.adapter === undefined ? {} : { adapter: options.adapter }),
    ...(timestamps ? { gpuTiming: true } : {}),
    onDeviceLost: (info) => {
      deviceLost = { reason: info.reason, message: info.message };
      notify?.({ kind: "deviceLost", reason: info.reason, message: info.message });
    },
  });
  try {
    const timestampUnit = timestampUnitFor(navigator.userAgent);
    const context = await createSweepContext(gpu, timestampUnit);
    const info = gpu.adapterInfo;
    const adapter: ReportAdapter = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };
    let rows: readonly SweepRow[];
    let cases: readonly CaseSummary[];
    let aborted: boolean;
    try {
      notify?.({
        kind: "started",
        adapter,
        unit: context.unit,
        dp4a: context.dp4a,
        caseCount: plan.cases.length,
      });
      ({ rows, cases, aborted } = await runSweep(context, plan.cases, {
        rounds: plan.rounds,
        candidateSet: plan.candidateSet,
        timestampUnit,
      }, {
        signal: options.signal,
        onProgress: (message) => notify?.({ kind: "status", message }),
        onRow: (row) => notify?.({ kind: "row", row }),
        onCase: (summary) => notify?.({ kind: "case", summary }),
      }));
    } finally {
      destroySweepContext(context);
    }
    return {
      format: REPORT_FORMAT,
      date: new Date().toISOString(),
      startedAt,
      aborted,
      // 比の土台・測った候補のカーネルとケースの版を記録に焼く（生成器が今の runtime と照合する — ADR 0117
      // 決定 8）。どれも GPU を読まない純関数で、掃引に比べて無視できる所要（既定の指紋は約 13 ms・候補の指紋は
      // full の全行でも 1 秒未満）
      caseSet: sweepCaseSetId(),
      defaultKernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
      candidateKernels: sweepCandidateKernelsId(rows, context.dp4a),
      userAgent: navigator.userAgent,
      adapter,
      gpuTiming: {
        feature: gpu.gpuTimingEnabled,
        unit: context.unit,
        quantized: roundsLookQuantized(rows, context.unit),
      },
      dp4a: context.dp4a,
      settings: {
        candidateSet: plan.candidateSet,
        quick: plan.candidateSet === "quick",
        ops: plan.ops,
        ...(plan.caseIds.length === 0 ? {} : { cases: plan.caseIds }),
        rounds: plan.rounds,
        targetPassMs: TARGET_PASS_MS,
        maxReps: SWEEP_MAX_REPS,
        warmupNs: WARMUP_NS,
        warmupMinRuns: WARMUP_MIN_RUNS,
        wallTimingNote: WALL_TIMING_NOTE,
      },
      deviceLost,
      cases,
      rows,
    };
  } finally {
    gpu.destroy();
  }
};

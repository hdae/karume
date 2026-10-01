/**
 * 掃引の記録（`karume-geometry-sweep/2`）の形と、行を既定幾何と突き合わせる純関数。
 *
 * 記録を組むのは掃引の入口 `runGeometrySweep`（`sweep.ts`）1 本で、Deno の CLI（`tools/geometry-sweep`）・
 * ブラウザのページ（`tools/gpu-lab` の掃引タブ）・利用者アプリが同じ形の JSON を得る — 別々に組むと、
 * M2 / Chrome と B570 / Deno の JSON を並べる段で形が黙ってずれる。
 */
import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import type { I8a8Geometry } from "../kernels/i8a8-geometry.ts";
import type { SweepOp } from "./cases.ts";
import type { CandidateSet } from "./geometries.ts";

/**
 * 記録の形式。/2 = /1 の `nsPerDispatch` を単位中立な `perDispatch` に改名し、ケース単位の
 * `cases[]`（既定幾何の再測定）と `settings` の空回し・壁時計の注記を足した形。
 */
export const REPORT_FORMAT = "karume-geometry-sweep/2";

/**
 * 既定幾何の再測定の比（再測定 ÷ 初回）がこの範囲の外なら、そのケースの測り直しの目安にする
 * （熱・クロックが掃引の間に動いた疑い — 比の土台の既定の値が揺れている）。
 */
export const DEFAULT_DRIFT_RANGE = { min: 0.9, max: 1.1 } as const;

/** 既定幾何の再測定の比が {@link DEFAULT_DRIFT_RANGE} の外か。 */
export const driftOutOfRange = (driftRatio: number): boolean =>
  !(driftRatio >= DEFAULT_DRIFT_RANGE.min && driftRatio <= DEFAULT_DRIFT_RANGE.max);

/**
 * 壁時計由来の値の注記（JSON の `settings.wallTimingNote` に載せる — README の Output 節と同文）。
 * 壁時計は submit → 完了の床（Deno で ≈11 ms / pass）を含むので、1 dispatch あたりに割ると床の
 * 按分が乗る。
 */
export const WALL_TIMING_NOTE =
  'Wall-clock values (unit "wall", wallNsPerDispatch, and TFLOPS under deno-raw-tick) include ' +
  "the submit-to-completion floor (about 11 ms per pass on Deno): TFLOPS derived from them reads " +
  "low, and ratios between geometries shrink toward 1.";

/**
 * 時間の単位。`ns` = timestamp-query（Chrome / Dawn は仕様どおり ns）、`deno-raw-tick` =
 * timestamp-query を Deno で採った値（wgpu は raw tick を換算しない — docs/known-issues.md
 * 「Intel Arc B570」節。B570 では 1 tick = 52.08 ns）、`wall` = timestamp-query が無い device で
 * submit → `onSubmittedWorkDone` の壁時計（ns）。
 */
export type TimingUnit = "ns" | "deno-raw-tick" | "wall";

/** (case, geometry) 1 組の結果。失敗した組は `error` だけを持って残る（掃引は止めない）。 */
export type SweepRow = {
  readonly caseId: string;
  readonly op: SweepOp;
  readonly shape: string;
  readonly censusCount: number;
  /** 幾何の表示名（キー断片と同じ綴り）。 */
  readonly geometry: string;
  readonly geometryParams: GemmGeometry | I8a8Geometry;
  /** 既定プロファイルの幾何（比の土台）か。Session が実際に使う幾何は adapter のプロファイル。 */
  readonly isDefault: boolean;
  readonly key?: string;
  readonly workgroups?: readonly [number, number, number];
  /** 1 pass に積んだ dispatch の本数。 */
  readonly reps?: number;
  /** 計測 round ごとの pass 全体の時間（単位はレポートの `gpuTiming.unit`）。 */
  readonly rounds?: readonly number[];
  /** 計測 round ごとの pass の壁時計（submit → 完了・ns）。 */
  readonly wallRounds?: readonly number[];
  /**
   * timestamp の差が負だった round の数。`rounds` には 0 に丸めて残す（黙って捨てない）が、
   * `perDispatch` の min の候補からは外す。全 round が負なら行は `error` になる。
   */
  readonly clampedNegativeSamples?: number;
  /** `min(負でなかった rounds) / reps`（単位は `gpuTiming.unit`）。 */
  readonly perDispatch?: number;
  /** `min(wallRounds) / reps`（ns — submit → 完了の床を含む・{@link WALL_TIMING_NOTE}）。 */
  readonly wallNsPerDispatch?: number;
  /**
   * 演算数 / 1 dispatch の時間（TFLOPS・i8a8 は TOPS 相当）。時間は単位が ns のとき
   * `perDispatch`、それ以外は `wallNsPerDispatch`（raw tick は ns ではない — 床を含むので
   * 低めに出る・{@link WALL_TIMING_NOTE}）。
   */
  readonly tflops?: number;
  /** 既定幾何の `perDispatch` / この行の `perDispatch`（1 より大きい = 既定より速い）。 */
  readonly speedupVsDefault?: number;
  /** 出力の digest（{@link "./harness.ts"} の `outputDigest` — 64 MiB 区切りの 2 段 SHA-256）。 */
  readonly outputSha256?: string;
  /** 既定幾何と出力 digest が一致したか（f32 は K 縮約順の契約・i8a8 は整数厳密で一致が期待）。 */
  readonly identicalToDefault?: boolean;
  readonly error?: string;
};

/**
 * ケース単位の記録。`defaultRepeat` はケースの**末尾で既定幾何をもう 1 度**測った値
 * （`perDispatch` は行と同じ単位・`driftRatio` = 再測定 ÷ 初回）。既定の初回が失敗したケースは
 * 再測定しない（比の土台が無い）。再測定そのものが失敗したら `defaultRepeatError`。
 */
export type CaseSummary = {
  readonly caseId: string;
  readonly defaultRepeat?: { readonly perDispatch: number; readonly driftRatio: number };
  readonly defaultRepeatError?: string;
};

/** 掃引した adapter（`GPUAdapterInfo` の 4 欄 — 空文字も値のまま）。 */
export type ReportAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/** 掃引の記録（`karume-geometry-sweep/2` — 生成器 `derive.ts` の入力・JSON に書く形）。 */
export type Report = {
  readonly format: typeof REPORT_FORMAT;
  readonly date: string;
  /** ブラウザは `navigator.userAgent`、Deno の CLI は `{ deno: Deno.version.deno }`。 */
  readonly userAgent: string | { readonly deno: string };
  readonly adapter: ReportAdapter;
  readonly checkout?: string;
  readonly checkoutDirty?: boolean;
  /** ページの bundle の sha256（CLI は bundle しないので無い）。 */
  readonly bundleSha256?: string;
  readonly gpuTiming: {
    /** device が `timestamp-query` を有効にしたか。 */
    readonly feature: boolean;
    readonly unit: TimingUnit;
    /** 全 round の値が 100 µs の倍数（Chrome の timestamp 量子化の疑い — 単位 ns のときだけ判定）。 */
    readonly quantized: boolean;
  };
  /** i8a8 の整数内積に `dot4I8Packed` を使ったか（WGSL 言語機能の列挙 — 数値は同一）。 */
  readonly dp4a: boolean;
  readonly settings: {
    /** 候補集合（`quick` / `quick+` / `full`）。この欄より前の JSON は `quick` だけを持つ。 */
    readonly candidateSet: CandidateSet;
    /** 互換の欄: `candidateSet === "quick"` のときだけ true（`quick+` は false）。 */
    readonly quick: boolean;
    readonly ops: readonly SweepOp[];
    /** `--case` で絞ったとき（無ければ op の全ケース）。 */
    readonly cases?: readonly string[];
    readonly rounds: number;
    readonly targetPassMs: number;
    /**
     * この掃引の `reps` の上限（harness.ts の `SWEEP_MAX_REPS`）。掃引専用の上限を分ける前の JSON は
     * opbench と共有の 1024 — どの上限で取られた掃引かをこの欄で見分ける。
     */
    readonly maxReps: number;
    /** 幾何ごとの空回しの下限（累計時間 ns — measurement.ts の `WARMUP_NS`）。 */
    readonly warmupNs: number;
    /** 幾何ごとの空回しの回数下限（同 `WARMUP_MIN_RUNS`）。 */
    readonly warmupMinRuns: number;
    /** {@link WALL_TIMING_NOTE}。 */
    readonly wallTimingNote: string;
  };
  readonly deviceLost: { readonly reason: string; readonly message: string } | null;
  readonly cases: readonly CaseSummary[];
  readonly rows: readonly SweepRow[];
};

/** 既定幾何の行から採る参照（`perDispatch` と digest）。 */
export type DefaultReference = {
  readonly perDispatch?: number;
  readonly outputSha256?: string;
};

/**
 * 行に既定幾何との比較（速さの比と digest の一致）を足す。既定の行が失敗していれば比較できない
 * ので足さない（黙って `false` にすると「幾何が値を変えた」と読める）。
 */
export const compareToDefault = (row: SweepRow, reference: DefaultReference): SweepRow => {
  if (row.error !== undefined) return row;
  return {
    ...row,
    ...(reference.perDispatch === undefined || row.perDispatch === undefined ||
        row.perDispatch <= 0
      ? {}
      : { speedupVsDefault: reference.perDispatch / row.perDispatch }),
    ...(reference.outputSha256 === undefined || row.outputSha256 === undefined
      ? {}
      : { identicalToDefault: reference.outputSha256 === row.outputSha256 }),
  };
};

/** Chrome が timestamp を丸める刻み（WebGPU Developer Features を切っているとき）。 */
export const CHROME_TIMESTAMP_QUANTUM_NS = 100_000;

/**
 * {@link looksQuantized} が読む GPU 時間の集計（`tools/anima-residency` の段の集計 `StageGpuTiming`
 * もこの形を満たす — 判定が読む欄だけを要求する）。
 */
export type QuantizationSample = {
  readonly totalNs: number;
  readonly entries: readonly { readonly ns: number }[];
};

/**
 * 全キーの `ns` が 100 µs の倍数なら、Chrome の timestamp 量子化が効いている疑いが濃い。
 *
 * WHY: 量子化下では 1 dispatch = 1 pass の各差分が 0 か 100 µs の倍数になり、和も倍数のまま残る。
 * 短い dispatch が大半を占める DiT では内訳の読みが荒れるので、表示で気づけるようにする
 * （判定は表示だけ — 数値は触らない）。掃引と GPU lab の Anima の段の集計が共有する（ADR 0117 決定 1 で
 * `tools/anima-residency/timing.ts` からここへ移した）。
 */
export const looksQuantized = (gpu: QuantizationSample): boolean =>
  gpu.totalNs > 0 && gpu.entries.every((entry) => entry.ns % CHROME_TIMESTAMP_QUANTUM_NS === 0);

/**
 * Chrome の 100 µs 量子化の疑い（{@link looksQuantized} を pass 単位の値へ流用する）。単位が ns の
 * ときだけ判定する — raw tick と壁時計は量子化の刻みを持たない。
 */
export const roundsLookQuantized = (rows: readonly SweepRow[], unit: TimingUnit): boolean => {
  if (unit !== "ns") return false;
  const entries = rows.flatMap((row) => (row.rounds ?? []).map((ns) => ({ ns })));
  return looksQuantized({
    totalNs: entries.reduce((sum, entry) => sum + entry.ns, 0),
    entries,
  });
};

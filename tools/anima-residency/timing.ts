/**
 * anima の段ごとの op 別 GPU 時間の集計（perf-ledger K-70 — 確認ページと Deno の双子 CLI で共有）。
 *
 * pipeline の `onRunDiagnostics` は `Session.run` 1 回ごとに呼ばれる（DiT は 1 step・CFG の枝ごと、
 * text 系はプロンプトごと、VAE はタイルごと）。`SessionDiagnostics` は run のたびに置き換わるので、
 * 呼ばれた時点で {@link snapshotRun} で数値を写し取り、段が閉じた後に {@link aggregateRuns} で
 * パイプラインキー別に足し合わせる。どちらも純関数（GPU に触らない）。
 */
import type { SessionDiagnostics } from "../../packages/runtime/mod.ts";

/** パイプラインキー 1 本ぶんの GPU 時間（`ns` は計測の単位そのまま — 下の NOTE）。 */
export type TimingEntry = {
  readonly key: string;
  readonly ns: number;
  readonly dispatchCount: number;
};

/** パイプラインキー 1 本ぶんの計画上の dispatch 本数（計測に依らず常に採れる）。 */
export type PipelineCount = { readonly key: string; readonly dispatchCount: number };

/** 集計が読む診断の 2 欄（`onRunDiagnostics` が渡す `SessionDiagnostics` はそのまま渡せる）。 */
export type RunDiagnostics = Pick<SessionDiagnostics, "lastRunTiming" | "lastRunPipelines">;

/** run 1 回の写し。`timing` は計測が無効な device では無い。 */
export type RunSample = {
  readonly timing?: {
    readonly totalNs: number;
    readonly clampedNegativeSamples: number;
    readonly entries: readonly TimingEntry[];
  };
  readonly pipelines: readonly PipelineCount[];
};

/**
 * 段 1 回ぶんの GPU 時間の合計。
 *
 * NOTE: `ns` は実行環境が返した timestamp の差をそのまま足した値。Chrome（Dawn）は仕様どおり ns
 * だが、Deno は wgpu の raw tick を換算せずに返す（B570 の Vulkan では 1 tick = 52.08 ns —
 * docs/known-issues.md「Intel Arc B570」節）。単位はレポートの `gpuTiming.unit` が表す。
 */
export type StageGpuTiming = {
  readonly totalNs: number;
  /** 足し合わせた run の回数。 */
  readonly runs: number;
  readonly clampedNegativeSamples: number;
  /** 全キー・`ns` の降順（同値はキーの辞書順 — runtime の `GpuTimingStats` と同じ並び）。 */
  readonly entries: readonly TimingEntry[];
};

export type StageProfile = {
  /** 計測が無効なら無い（run が 1 回も終わらなかった段でも無い）。 */
  readonly gpu?: StageGpuTiming;
  /** キーの辞書順（runtime の `lastRunPipelines` と同じ並び）。 */
  readonly pipelines: readonly PipelineCount[];
};

/**
 * `onRunDiagnostics` が受けた診断から、集計に要る数値だけを写し取る（参照を持ち越さない）。
 *
 * `lastRunPipelines` は run が成功すれば必ず埋まる（`onRunDiagnostics` は成功した run の後にしか
 * 呼ばれない）ので、無ければ落とす — 黙って空にすると「dispatch 0 本の段」に化ける。
 */
export const snapshotRun = (diagnostics: RunDiagnostics): RunSample => {
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) {
    throw new Error("run 直後の診断に lastRunPipelines が無い（run が導出相で落ちた形）");
  }
  const timing = diagnostics.lastRunTiming;
  return {
    ...(timing === undefined ? {} : {
      timing: {
        totalNs: timing.totalNs,
        clampedNegativeSamples: timing.clampedNegativeSamples,
        entries: timing.entries.map(({ key, ns, dispatchCount }) => ({ key, ns, dispatchCount })),
      },
    }),
    pipelines: pipelines.map(({ key, dispatchCount }) => ({ key, dispatchCount })),
  };
};

const byKey = (a: { readonly key: string }, b: { readonly key: string }): number =>
  a.key < b.key ? -1 : a.key > b.key ? 1 : 0;

/**
 * 段 1 回ぶんの run の写しをパイプラインキー別に足し合わせる。
 *
 * 計測ありの run と無しの run が混ざっていたら落とす — 1 本の device の上では起きない形で、
 * 起きたら集計の前提（同じ単位の足し算）が崩れている。
 */
export const aggregateRuns = (samples: readonly RunSample[]): StageProfile => {
  const pipelineTotals = new Map<string, number>();
  for (const { key, dispatchCount } of samples.flatMap((sample) => sample.pipelines)) {
    pipelineTotals.set(key, (pipelineTotals.get(key) ?? 0) + dispatchCount);
  }
  const pipelines = [...pipelineTotals]
    .map(([key, dispatchCount]) => ({ key, dispatchCount }))
    .sort(byKey);
  const timed = samples.filter((sample) => sample.timing !== undefined);
  if (timed.length === 0) return { pipelines };
  if (timed.length !== samples.length) {
    throw new Error(
      `GPU 時間の有る run（${timed.length}）と無い run（${
        samples.length - timed.length
      }）が同じ段に混ざった`,
    );
  }
  let totalNs = 0;
  let clampedNegativeSamples = 0;
  const entryTotals = new Map<string, { ns: number; dispatchCount: number }>();
  for (const { timing } of timed) {
    if (timing === undefined) continue;
    totalNs += timing.totalNs;
    clampedNegativeSamples += timing.clampedNegativeSamples;
    for (const entry of timing.entries) {
      const total = entryTotals.get(entry.key) ?? { ns: 0, dispatchCount: 0 };
      total.ns += entry.ns;
      total.dispatchCount += entry.dispatchCount;
      entryTotals.set(entry.key, total);
    }
  }
  const entries = [...entryTotals]
    .map(([key, { ns, dispatchCount }]) => ({ key, ns, dispatchCount }))
    .sort((a, b) => b.ns - a.ns || byKey(a, b));
  return {
    gpu: { totalNs, runs: timed.length, clampedNegativeSamples, entries },
    pipelines,
  };
};

/** 表示用の上位 `limit` キー（`share` = 段の GPU 合計に占める割合 0〜1）。 */
export const topEntries = (
  gpu: StageGpuTiming,
  limit: number,
): (TimingEntry & { readonly share: number })[] =>
  gpu.entries.slice(0, limit).map((entry) => ({
    ...entry,
    share: gpu.totalNs === 0 ? 0 : entry.ns / gpu.totalNs,
  }));

/** Chrome が timestamp を丸める刻み（WebGPU Developer Features を切っているとき）。 */
export const CHROME_TIMESTAMP_QUANTUM_NS = 100_000;

/**
 * 全キーの `ns` が 100 µs の倍数なら、Chrome の timestamp 量子化が効いている疑いが濃い。
 *
 * WHY: 量子化下では 1 dispatch = 1 pass の各差分が 0 か 100 µs の倍数になり、和も倍数のまま残る。
 * 短い dispatch が大半を占める DiT では内訳の読みが荒れるので、表示で気づけるようにする
 * （判定は表示だけ — 数値は触らない）。
 */
export const looksQuantized = (gpu: StageGpuTiming): boolean =>
  gpu.totalNs > 0 && gpu.entries.every((entry) => entry.ns % CHROME_TIMESTAMP_QUANTUM_NS === 0);

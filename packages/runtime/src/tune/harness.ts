/**
 * タイル幾何の掃引の計測核（perf-ledger K-70 — 掃引の入口 `runGeometrySweep`〈sweep.ts〉が使い、Deno の
 * CLI・ブラウザのページ `tools/gpu-lab` の掃引タブ・利用者アプリがその入口を共有する。使うのは WebGPU
 * 標準 API と runtime の src だけ）。
 *
 * 同じ shape・同じ入力のまま**幾何だけ**を変えて 1 dispatch の時間を測る。WGSL とキーは runtime の
 * 生成入口に明示の幾何を渡して作る（`linearWgsl(…, geometry)` 等 — Session の経路は通らない）ので、
 * 測っているカーネルは本番のカーネルと同じ生成器の出力そのもの。
 *
 * ## 計測の規約（measurement.ts — tools/opbench と共有する）
 *
 * 1. 1 compute pass に同じ dispatch を `reps` 本積み、pass の `timestampWrites`（beginning / end）の
 *    差 ÷ `reps` を 1 dispatch の時間とする。`reps` は 1 pass ≈ {@link TARGET_PASS_MS} になる本数
 *    （{@link calibrateReps}・上限は掃引専用の {@link SWEEP_MAX_REPS}）。
 * 2. 代表値は `rounds` 回の **min**（熱ドリフトを吸う）。timestamp の差が負だった round は
 *    0 に丸めて記録に残すが min の候補から外す（全 round が負なら行を失敗にする）。
 * 3. 幾何ごとに、計測の前に pass を「累計 ≥ {@link WARMUP_NS} かつ ≥ {@link WARMUP_MIN_RUNS} 回」まで
 *    空回しする（opbench の `pinClocks` の下限と同じ・打ち切りも同じ {@link WARMUP_MAX_PASSES} 回）。
 *    累計は単位 `ns` なら GPU 時間、`deno-raw-tick` / `wall` なら壁時計で数える（raw tick は ns
 *    ではない）。
 * 4. ケースの末尾で既定幾何をもう 1 度測り、初回との比（`driftRatio`）をケース単位で残す — 掃引の
 *    間に熱やクロックが動いたかの目安（比の土台の既定の値が揺れていないか）。
 * 5. 絶対値を別機と比べない — 同じ機の中の比（既定幾何に対する速さ）だけが有効。
 *
 * NOTE: `reps` の見積りの入力は opbench と違い **壁時計**（`reps` を倍々に積んで pass の壁時計が
 * 目標の 1/2 に届いたところで、直前の 2 点の傾き = 1 dispatch の増分を採る）。Deno は timestamp を
 * ns に換算しない（B570 で 1 tick = 52.08 ns）ので timestamp から見積ると本数が 52 倍に膨らみ、
 * Chrome は開発者フラグ無しでは 100 µs に丸めるので短い dispatch が 0 に読めて上限まで積む。
 * 傾きを採るのは submit → 完了の床（Deno で ≈ 11 ms — B570 の実測で平均だと本数が半分に出た）を
 * 消すため。timestamp-query が無い device は同じ規約を壁時計で回す（単位 `wall`）。
 *
 * ## 束縛の寸法（写し元）
 *
 * 束縛は本番と同じく**位置**で渡す: 各 {@link Launch} の `bindings` は recipe-builders が同じカーネルに
 * 渡している `binds` の並び（binding 1 から順）を写した表。表は PipelineCache が WGSL から採る
 * storage の役割（`roles` — src/gpu/pipeline-cache.ts `parseStorageRoles`）と突き合わせ、束縛の本数と
 * 書き込み先の位置が食い違えば行を失敗にする（{@link assertBindingRoles}）。役割の同じ束縛どうしの
 * 入れ替わりは役割では見えない — linear / matmul / bmm は CPU 参照の突合テスト（tests/tune_harness_test.ts）が検出器。寸法の
 * 式は各 recipe-builder の確保と params 関数から写す（{@link casePlan} の各枝に `file:line`）。
 *
 * ## ビット同一の確認
 *
 * 幾何ごとに出力を 0 に戻してから回し、出力の digest（{@link outputDigest}）を既定幾何の行と比べる。
 * f32 は ADR 0022 決定 3（K 縮約順は幾何に依らない）で、i8a8 は整数縮約の厳密性で一致が期待される。
 * 一致しない幾何は、速くても採用条件を満たさない。
 */
// WHY: 公開面（mod.ts）には errorScope のロック・device lost の競走・パイプラインキャッシュが無い。
// 本番と同じキャッシュ（キー衝突の検出込み）とスコープ規律で測るため、内部面に意図して結合する。
import { type GpuContext, RUNTIME_INTERNAL } from "../gpu/device.ts";
import type { StorageRoles } from "../gpu/pipeline-cache.ts";
import { discardFailureScopes, popFailureScopes, pushFailureScopes } from "../gpu/error-scope.ts";
import { BUFFER_USAGE, MAP_MODE } from "../gpu/webgpu-constants.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import { gemmMTileGeometry, gemmUsesVec4 } from "../kernels/gemm.ts";
import {
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmTileM,
  gemmTileN,
} from "../kernels/gemm-geometry.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8TileM,
  i8a8TileN,
} from "../kernels/i8a8-geometry.ts";
import { linearKey, linearParams, linearWgsl } from "../kernels/linear.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../kernels/matmul.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../kernels/bmm.ts";
import {
  dp4aAvailable,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../kernels/linear-i8a8.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsWgsl,
} from "../kernels/attention.ts";
import {
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../kernels/attention-i8a8.ts";
import {
  type Conv2dDims,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dUsesVec4,
} from "../kernels/conv2d.ts";
import { scoreStorageBytes } from "../kernels/score-storage.ts";
import { calibrateReps, TARGET_PASS_MS, WARMUP_MIN_RUNS, WARMUP_NS } from "./measurement.ts";
import {
  type AttentionCase,
  type BmmCase,
  caseFlops,
  caseShape,
  type Conv2dCase,
  type LinearCase,
  type MatmulCase,
  type SweepCase,
} from "./cases.ts";
import {
  type CandidateSet,
  conv2dCandidate,
  conv2dCandidatesIn,
  gemmCandidate,
  gemmCandidatesIn,
  type GeometryCandidate,
  i8a8Candidate,
  i8a8CandidatesIn,
} from "./geometries.ts";
import {
  type CaseSummary,
  compareToDefault,
  type DefaultReference,
  type SweepRow,
  type TimingUnit,
} from "./report.ts";

/**
 * 掃引の `reps` の上限（計測の規約 1）。opbench の `MAX_REPS`（1024）は出力の readback とメモリが
 * 反復に比例して増えるので打ち切る上限だが、掃引は同じバッファに重ね打ちするので反復を増やしても
 * readback もメモリも増えない。1 dispatch 15 µs で 1 pass ≈ {@link TARGET_PASS_MS} に要る反復は
 * 5,334（1024 では 15 ms にしかならず、既定幾何の再測定比が 0.70〜1.55 と揺れた）・5 µs でも
 * 16,000 で上限内。opbench の `MAX_REPS` は変えない（あちらの打ち切りの理由はそのまま残る）。
 */
export const SWEEP_MAX_REPS = 16384;

/**
 * 空回しの打ち切り（計測の規約 3 — opbench の `pinClocks` と同じ 64 回）。reps が上限で頭打ちの
 * 短い dispatch で累計が {@link WARMUP_NS} に届かないときに、空回しを際限なく続けない。
 */
export const WARMUP_MAX_PASSES = 64;

/** `reps` の見積りを締める pass の壁時計（目標長の 1/2 — モジュール doc の NOTE）。 */
const PROBE_MIN_NS = (TARGET_PASS_MS / 2) * 1e6;

/** 出力の読み戻しと入力の書き込みの 1 区切り（64 MiB — 1 GiB の S を丸ごと JS に載せない）。 */
export const DIGEST_CHUNK_BYTES = 64 * 1024 * 1024;

/** 入力を書き込む区切り（生成した乱数列を持つ JS 側の峰を抑える）。 */
const FILL_CHUNK_BYTES = 16 * 1024 * 1024;

export type SweepSettings = {
  /** 計測 round の数（既定は measurement.ts の `ROUNDS`）。 */
  readonly rounds: number;
  /** 候補集合（geometries.ts の {@link CandidateSet}）。 */
  readonly candidateSet: CandidateSet;
  /** timestamp-query が有効なときの単位（Chrome = `ns`・Deno = `deno-raw-tick`）。 */
  readonly timestampUnit: "ns" | "deno-raw-tick";
};

export type SweepHooks = {
  /** 進捗 1 行（ページの状態行・CLI の標準エラー）。 */
  readonly onProgress?: (message: string) => void;
  /** 行が 1 つ確定するたびに呼ぶ（既定幾何の行が先に来る）。 */
  readonly onRow?: (row: SweepRow) => void;
  /** ケースを測り終えるたびに呼ぶ（そのケースの最後の行の後・既定幾何の再測定を含む）。 */
  readonly onCase?: (summary: CaseSummary) => void;
  /** 中断（今の幾何を測り終えたところで止める — 残りは行を作らない）。 */
  readonly signal?: AbortSignal;
};

/** 入力 1 本の中身（決定的な擬似乱数 — seed は資源の並び順で固定）。 */
export type Fill = "f32" | "f16" | "i8" | "scale" | "none";

/** 資源 1 本。名前は資源表の中の識別子（カーネルを跨いで同じ名前は同じバッファ — WGSL の束縛名に揃える）。 */
type ResourceSpec = {
  readonly name: string;
  readonly bytes: number;
  readonly fill: Fill;
};

/** 1 dispatch ぶんの生成物と寸法。 */
type Launch = {
  readonly key: string;
  readonly wgsl: string;
  readonly params: Uint32Array<ArrayBuffer>;
  /**
   * binding 1 から順に渡す資源の名前（recipe-builders の `binds` の並びを写した表 — 本番と同じ
   * 位置束縛）。`bindings[i]` が binding `i + 1`。
   */
  readonly bindings: readonly string[];
  /** そのカーネルが書く資源（WGSL の `read_write` と突き合わせる — {@link assertBindingRoles}）。 */
  readonly writes: readonly string[];
  readonly workgroups: readonly [number, number, number];
};

export type CasePlan = {
  readonly resources: readonly ResourceSpec[];
  /** 既定幾何で 1 度だけ回す前段（③PV の S と行統計を作る ①QK → ②）。 */
  readonly prelude: readonly Launch[];
  readonly output: string;
  readonly defaultCandidate: GeometryCandidate;
  readonly launch: (candidate: GeometryCandidate) => Launch;
};

const F32 = 4;

const gemmOf = (candidate: GeometryCandidate, where: string): GemmGeometry => {
  if (candidate.family !== "gemm") {
    throw new Error(`${where}: f32 骨格の幾何ではない（${candidate.name}）`);
  }
  return candidate.geometry;
};

const i8a8Of = (candidate: GeometryCandidate, where: string): I8a8Geometry => {
  if (candidate.family !== "i8a8") {
    throw new Error(`${where}: i8a8 の幾何ではない（${candidate.name}）`);
  }
  return candidate.geometry;
};

/**
 * 生の重みバッファの寸法 = 格納バイト長の 4 バイト切り上げ（src/runtime/session-build.ts:960 の
 * 生バイト席と同じ — f16 は 2 B / 要素・i8 は 1 B / 要素）。
 */
const rawWeightBytes = (payloadBytes: number): number =>
  payloadBytes + ((4 - (payloadBytes % 4)) % 4);

/** f32 linear（重み f16 格納）— src/runtime/recipe-builders/linear.ts:310-337 の GEMM 経路。 */
const linearPlan = (sweepCase: LinearCase, limit: number): CasePlan => {
  const { m, n, k } = sweepCase;
  const v4 = gemmUsesVec4(k, n);
  return {
    resources: [
      // x[m,k] / W[n,k] / b[n] / out[m,n]（束縛 1..4 = binds の順・linear.ts:327-331。f16 は scale 無し）
      { name: "x", bytes: m * k * F32, fill: "f32" },
      { name: "w", bytes: rawWeightBytes(n * k * 2), fill: "f16" },
      { name: "bias", bytes: n * F32, fill: "f32" },
      { name: "out", bytes: m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "out",
    // 掃引の既定 = 既定プロファイルの幾何（行数バケット `gemmGeometryForRows(m)` — 比の土台）。
    // Session が実際に使う幾何は adapter のプロファイル（src/kernels/geometry-profile.ts）
    defaultCandidate: gemmCandidate(gemmGeometryForRows(m)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        key: linearKey("f16", v4, "f32", m, undefined, geometry),
        wgsl: linearWgsl("f16", v4, "f32", m, undefined, geometry),
        params: linearParams(m, n, k),
        bindings: ["x", "w", "bias", "out"],
        writes: ["out"],
        // linear.ts:332-336
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
          1,
        ],
      };
    },
  };
};

/** f32 matmul — src/runtime/recipe-builders/linear.ts:65-107（buildMatmul）。 */
const matmulPlan = (sweepCase: MatmulCase, limit: number): CasePlan => {
  const { m, n, k } = sweepCase;
  // linear.ts:77
  const v4 = gemmUsesVec4(k, n);
  return {
    resources: [
      // a[m,k] / b[k,n] / c[m,n]（束縛 1..3 = binds[0] / binds[1] / outs[0] の順・linear.ts:96-100。
      // 名前は WGSL の束縛名 — kernels/gemm.ts の denseWgsl）
      { name: "a", bytes: m * k * F32, fill: "f32" },
      { name: "b", bytes: k * n * F32, fill: "f32" },
      { name: "c", bytes: m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "c",
    // 掃引の既定 = 既定プロファイルの幾何（行数バケット `gemmGeometryForRows(m)` — 比の土台）。
    // Session が実際に使う幾何は adapter のプロファイル（linear.ts:81 の gemmRowsGeometry）
    defaultCandidate: gemmCandidate(gemmGeometryForRows(m)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        // linear.ts:82-87
        key: matmulKey(v4, m, geometry),
        wgsl: matmulWgsl(v4, m, geometry),
        params: matmulParams(m, n, k),
        bindings: ["a", "b", "c"],
        writes: ["c"],
        // linear.ts:101-105
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
          1,
        ],
      };
    },
  };
};

/** f32 bmm（バッチは z 軸）— src/runtime/recipe-builders/linear.ts:113-153（buildBmm）。 */
const bmmPlan = (sweepCase: BmmCase, limit: number): CasePlan => {
  const { batch, m, n, k } = sweepCase;
  // linear.ts:123
  const v4 = gemmUsesVec4(k, n);
  return {
    resources: [
      // a[batch,m,k] / b[batch,k,n] / c[batch,m,n]（束縛 1..3 = linear.ts:141-145）
      { name: "a", bytes: batch * m * k * F32, fill: "f32" },
      { name: "b", bytes: batch * k * n * F32, fill: "f32" },
      { name: "c", bytes: batch * m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "c",
    // 掃引の既定 = 既定プロファイルの幾何（行列 1 枚の m のバケット — linear.ts:124-126）
    defaultCandidate: gemmCandidate(gemmGeometryForRows(m)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        // linear.ts:127-132（行窓無し）
        key: bmmKey(v4, m, undefined, geometry),
        wgsl: bmmWgsl(v4, m, undefined, geometry),
        params: bmmParams(m, n, k),
        bindings: ["a", "b", "c"],
        writes: ["c"],
        // linear.ts:146-151（バッチは z 軸の 1 workgroup = 1 バッチ）
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
          tiledWorkgroups(batch, 1, limit, sweepCase.id),
        ],
      };
    },
  };
};

/** i8a8 linear（重み i8 per-channel）— src/runtime/recipe-builders/linear.ts:438-523 の ② GEMM。 */
const i8a8LinearPlan = (sweepCase: LinearCase, limit: number, dp4a: boolean): CasePlan => {
  const { m, n, k } = sweepCase;
  const v4 = linearI8a8UsesVec4(n);
  return {
    resources: [
      // 量子化済み活性 xq（i8 を 4 詰め・linear.ts:462）と per-token scale xs（linear.ts:463）。
      // 本番は quantize_rows が作る — ここは同じ形式（kernels/quantize-rows.ts: 行 = token・
      // `row·dim/4` 語・±127）の値を直接書く（① は幾何を持たないので掃引の外）
      { name: "xq", bytes: m * (k / 4) * 4, fill: "i8" },
      { name: "xscale", bytes: m * F32, fill: "scale" },
      // 重み i8 [n,k]（生バイト席）と per-channel scale [n,1]（format/container/codecs.ts の
      // groupScaleShape + scaleBytes = n·4 B）。束縛は linear.ts:505-515
      { name: "w", bytes: rawWeightBytes(n * k), fill: "i8" },
      { name: "wscale", bytes: n * F32, fill: "scale" },
      { name: "bias", bytes: n * F32, fill: "f32" },
      { name: "out", bytes: m * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "out",
    // 掃引の既定 = 既定プロファイルの i8a8 linear 幾何（比の土台）
    defaultCandidate: i8a8Candidate(defaultI8a8Geometry("linear")),
    launch: (candidate) => {
      const geometry = i8a8Of(candidate, sweepCase.id);
      return {
        key: linearI8a8Key(v4, dp4a, geometry, "i8"),
        wgsl: linearI8a8Wgsl(v4, dp4a, geometry, "i8"),
        params: linearI8a8Params(m, n, k),
        bindings: ["xq", "w", "bias", "out", "wscale", "xscale"],
        writes: ["out"],
        // linear.ts:516-520
        workgroups: [
          tiledWorkgroups(n, i8a8TileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(m, i8a8TileM(geometry), limit, sweepCase.id),
          1,
        ],
      };
    },
  };
};

/**
 * ② 行統計（S の格納形に合わせた変種・regcache 無しの 2 回読み — ③PV の入力を作るだけなので
 * 速さは問わない）。params と dispatch は src/runtime/recipe-builders/attention.ts:432-443。
 */
const statsLaunch = (sweepCase: AttentionCase, limit: number): Launch => {
  const rows = sweepCase.batchHeads * sweepCase.m;
  return {
    key: attentionStatsKey("f32", sweepCase.score),
    wgsl: attentionStatsWgsl("f32", sweepCase.score),
    params: attentionStatsParams(rows, sweepCase.n),
    bindings: ["s", "stats"],
    writes: ["stats"],
    workgroups: [gridStrideWorkgroups(rows, 1, limit), 1, 1],
  };
};

/**
 * 融合 attention 共通の資源（src/runtime/recipe-builders/attention.ts）。S = `B·H·M·N·格納幅`
 * （:289 の scoreBytes・:370）・行統計 = `B·H·M·2` 語（:371）・O = `B·H·M·D` f32。
 */
const attentionShared = (sweepCase: AttentionCase): ResourceSpec[] => {
  const { batchHeads, m, n, d } = sweepCase;
  const scores: ResourceSpec = {
    name: "s",
    bytes: batchHeads * m * n * scoreStorageBytes(sweepCase.score),
    fill: "none",
  };
  if (sweepCase.stage === "qk") return [scores];
  return [
    scores,
    { name: "stats", bytes: batchHeads * m * ATTENTION_STATS_STRIDE * F32, fill: "none" },
    { name: "o", bytes: batchHeads * m * d * F32, fill: "none" },
  ];
};

/** f32 融合 attention（S は f32 格納）— src/runtime/recipe-builders/attention.ts:401-491。 */
const attentionPlan = (sweepCase: AttentionCase, limit: number): CasePlan => {
  const { batchHeads, m, n, d, scale } = sweepCase;
  const qkV4 = gemmUsesVec4(d, n);
  const pvV4 = gemmUsesVec4(n, d);
  const qk = (geometry: GemmGeometry): Launch => ({
    key: attentionQkKey(qkV4, "f32", "f32", false, false, false, geometry),
    wgsl: attentionQkWgsl(qkV4, "f32", "f32", false, false, false, geometry),
    params: attentionQkParams(m, n, d, scale),
    // attention.ts:417-421（mask 無し）
    bindings: ["q", "k", "s"],
    writes: ["s"],
    // attention.ts:423-427
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const pv = (geometry: GemmGeometry): Launch => ({
    key: attentionPvKey(pvV4, "f32", "f32", false, false, geometry),
    wgsl: attentionPvWgsl(pvV4, "f32", "f32", false, false, geometry),
    // attention.ts:476-478（m = M・n = D・k = N）
    params: attentionPvParams(m, d, n),
    // attention.ts:480-485
    bindings: ["s", "v", "stats", "o"],
    writes: ["o"],
    // attention.ts:486-490
    workgroups: [
      tiledWorkgroups(d, gemmTileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, gemmTileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const stagePv = sweepCase.stage === "pv";
  // 掃引の既定 = 既定プロファイルの幾何（①QK / ③PV とも `defaultGemmGeometry()` — 比の土台）。
  // Session が実際に使う幾何は adapter のプロファイル（src/kernels/geometry-profile.ts）
  const fallback = defaultGemmGeometry();
  return {
    resources: [
      // q[B·H,M,D] / k[B·H,N,D] / v[B·H,N,D]（rank-4 head-first の入力をバッチ軸に畳んだ形）
      { name: "q", bytes: batchHeads * m * d * F32, fill: "f32" },
      { name: "k", bytes: batchHeads * n * d * F32, fill: "f32" },
      ...(stagePv ? [{ name: "v", bytes: batchHeads * n * d * F32, fill: "f32" as const }] : []),
      ...attentionShared(sweepCase),
    ],
    prelude: stagePv ? [qk(fallback), statsLaunch(sweepCase, limit)] : [],
    output: stagePv ? "o" : "s",
    defaultCandidate: gemmCandidate(fallback),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return stagePv ? pv(geometry) : qk(geometry);
    },
  };
};

/**
 * i8a8 融合 attention（S は f16 格納 = s16）— src/runtime/recipe-builders/attention.ts:378-469 と
 * 前段の確保 :1044-1047（q / k の量子化）・:1141-1142（Vᵀ の量子化）。
 */
const i8a8AttentionPlan = (sweepCase: AttentionCase, limit: number, dp4a: boolean): CasePlan => {
  const { batchHeads, m, n, d, scale, score } = sweepCase;
  const qkV4 = attentionQkI8a8UsesVec4(n);
  const pvV4 = attentionPvI8a8UsesVec4(d);
  const qk = (geometry: I8a8Geometry): Launch => ({
    key: attentionQkI8a8Key(qkV4, dp4a, score, geometry),
    wgsl: attentionQkI8a8Wgsl(qkV4, dp4a, score, geometry),
    params: attentionQkI8a8Params(m, n, d, scale),
    // attention.ts:387-393
    bindings: ["qq", "kq", "s", "qscale", "kscale"],
    writes: ["s"],
    // attention.ts:394-398
    workgroups: [
      tiledWorkgroups(n, i8a8TileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const pv = (geometry: I8a8Geometry): Launch => ({
    key: attentionPvI8a8Key(pvV4, dp4a, score, geometry),
    wgsl: attentionPvI8a8Wgsl(pvV4, dp4a, score, geometry),
    // attention.ts:453-455（m = M・n = D・k = N）
    params: attentionPvI8a8Params(m, d, n),
    // attention.ts:457-463
    bindings: ["s", "vq", "stats", "o", "vscale"],
    writes: ["o"],
    // attention.ts:464-468
    workgroups: [
      tiledWorkgroups(d, i8a8TileN(geometry), limit, sweepCase.id),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, sweepCase.id),
      tiledWorkgroups(batchHeads, 1, limit, sweepCase.id),
    ],
  });
  const stagePv = sweepCase.stage === "pv";
  // 掃引の既定 = 既定プロファイルの i8a8 attention 幾何（①QK / ③PV で別・比の土台）
  const qkDefault = defaultI8a8Geometry("attention_qk");
  const pvDefault = defaultI8a8Geometry("attention_pv");
  return {
    resources: [
      // qq / kq は i8 を 4 詰め（B·H·M·D / B·H·N·D バイト）・qs / ks は per-token scale（:1044-1047）
      { name: "qq", bytes: batchHeads * m * d, fill: "i8" },
      { name: "qscale", bytes: batchHeads * m * F32, fill: "scale" },
      { name: "kq", bytes: batchHeads * n * d, fill: "i8" },
      { name: "kscale", bytes: batchHeads * n * F32, fill: "scale" },
      ...(stagePv
        ? [
          // Vᵀ[B·H,D,N] の per-column 量子化（行 = (b,h,d)・N 連続 — :1141-1142）
          { name: "vq", bytes: batchHeads * d * n, fill: "i8" as const },
          { name: "vscale", bytes: batchHeads * d * F32, fill: "scale" as const },
        ]
        : []),
      ...attentionShared(sweepCase),
    ],
    prelude: stagePv ? [qk(qkDefault), statsLaunch(sweepCase, limit)] : [],
    output: stagePv ? "o" : "s",
    defaultCandidate: i8a8Candidate(stagePv ? pvDefault : qkDefault),
    launch: (candidate) => {
      const geometry = i8a8Of(candidate, sweepCase.id);
      return stagePv ? pv(geometry) : qk(geometry);
    },
  };
};

/** conv2d の implicit GEMM（重み f16 格納）— src/runtime/recipe-builders/conv.ts:357-397。 */
const conv2dPlan = (sweepCase: Conv2dCase, limit: number): CasePlan => {
  const { channelsIn, channelsOut, height, width } = sweepCase;
  const dims: Conv2dDims = {
    batch: 1,
    channelsIn,
    channelsOut,
    heightIn: height,
    widthIn: width,
    heightOut: height,
    widthOut: width,
    kernelH: 3,
    kernelW: 3,
    strideH: 1,
    strideW: 1,
    paddingH: 1,
    paddingW: 1,
    dilationH: 1,
    dilationW: 1,
    groups: 1,
  };
  // conv.ts:366-370（M = Cout・N = Hout·Wout・K = Cin·Kh·Kw）
  const n = height * width;
  const kFlat = channelsIn * 9;
  const v4 = conv2dUsesVec4(kFlat, width, 1);
  const mTile = conv2dIgemmMTile(channelsOut);
  return {
    resources: [
      // x[B,Cin,H,W] / W[Cout,Cin,3,3] / b[Cout] / out[B,Cout,H,W]（束縛 1..4 = conv.ts:387-391）
      { name: "x", bytes: channelsIn * n * F32, fill: "f32" },
      { name: "w", bytes: rawWeightBytes(channelsOut * kFlat * 2), fill: "f16" },
      { name: "bias", bytes: channelsOut * F32, fill: "f32" },
      { name: "out", bytes: channelsOut * n * F32, fill: "none" },
    ],
    prelude: [],
    output: "out",
    // 掃引の既定 = 既定プロファイルの conv2d 幾何（m タイルのクラス 64 / 32 行ごと・比の土台）
    defaultCandidate: conv2dCandidate(gemmMTileGeometry(mTile)),
    launch: (candidate) => {
      const geometry = gemmOf(candidate, sweepCase.id);
      return {
        key: conv2dIgemmKey("f16", v4, mTile, geometry),
        wgsl: conv2dIgemmWgsl("f16", v4, mTile, geometry),
        params: conv2dIgemmParams(dims),
        bindings: ["x", "w", "bias", "out"],
        writes: ["out"],
        // conv.ts:392-396
        workgroups: [
          tiledWorkgroups(n, gemmTileN(geometry), limit, sweepCase.id),
          tiledWorkgroups(channelsOut, gemmTileM(geometry), limit, sweepCase.id),
          tiledWorkgroups(dims.batch, 1, limit, sweepCase.id),
        ],
      };
    },
  };
};

/** ケースの資源表・前段・候補ごとの dispatch（テストは資源の並びと中身をここから引く）。 */
export const casePlan = (sweepCase: SweepCase, limit: number, dp4a: boolean): CasePlan => {
  switch (sweepCase.op) {
    case "linear":
      return linearPlan(sweepCase, limit);
    case "matmul":
      return matmulPlan(sweepCase, limit);
    case "bmm":
      return bmmPlan(sweepCase, limit);
    case "i8a8-linear":
      return i8a8LinearPlan(sweepCase, limit, dp4a);
    case "attention":
      return attentionPlan(sweepCase, limit);
    case "i8a8-attention":
      return i8a8AttentionPlan(sweepCase, limit, dp4a);
    case "conv2d":
      return conv2dPlan(sweepCase, limit);
  }
};

/** op 族ごとの候補（`set` の集合）。既定幾何は含まれていなくても {@link sweepCase} が先頭に足す。 */
export const candidatesFor = (sweepCase: SweepCase, set: CandidateSet): GeometryCandidate[] => {
  switch (sweepCase.op) {
    case "linear":
    case "matmul":
    case "bmm":
    case "attention":
      return gemmCandidatesIn(set);
    case "conv2d":
      return conv2dCandidatesIn(set);
    case "i8a8-linear":
    case "i8a8-attention":
      return i8a8CandidatesIn(set);
  }
};

const sortedList = (values: Iterable<number>): string =>
  [...values].sort((a, b) => a - b).join(", ");

/**
 * 写した束縛表（{@link Launch} の `bindings` / `writes`）が WGSL の storage の役割（PipelineCache の
 * `roles`）と噛み合うかを見る: storage 束縛が binding 1..n にちょうど揃い、書き込み先（`read_write`）
 * の位置が表の `writes` の位置と一致すること。食い違いは写し元のカーネルか recipe-builders の並びが
 * 変わった印なので、推測で束縛せずに行を失敗にする。
 */
export const assertBindingRoles = (
  launch: Pick<Launch, "key" | "bindings" | "writes">,
  roles: StorageRoles,
): void => {
  const declared = sortedList([...roles.reads, ...roles.writes]);
  const expected = sortedList(launch.bindings.map((_, index) => index + 1));
  if (declared !== expected) {
    throw new Error(
      `${launch.key}: WGSL の storage 束縛 [${declared}] が写した表の位置 [${expected}]（${
        launch.bindings.join(", ")
      }）と違う`,
    );
  }
  const expectedWrites = sortedList(
    launch.bindings.flatMap((name, index) => launch.writes.includes(name) ? [index + 1] : []),
  );
  const declaredWrites = sortedList(roles.writes);
  if (declaredWrites !== expectedWrites) {
    throw new Error(
      `${launch.key}: WGSL の書き込み束縛 [${declaredWrites}] が写した表の書き込み先 [${expectedWrites}]（${
        launch.writes.join(", ")
      }）と違う`,
    );
  }
};

/** xorshift32（決定的・seed ≠ 0）。 */
const xorshift = (seed: number): () => number => {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state ^ (state << 13)) >>> 0;
    state = (state ^ (state >>> 17)) >>> 0;
    state = (state ^ (state << 5)) >>> 0;
    return state;
  };
};

const TWO_POW_32 = 2 ** 32;

/** 1 語ぶんの値を `fill` の形式で作る（u32 のビット列として返す）。 */
const wordFiller = (fill: Exclude<Fill, "none">, next: () => number): () => number => {
  const bits = new Float32Array(1);
  const word = new Uint32Array(bits.buffer);
  switch (fill) {
    case "f32":
      // [-1, 1) の f32
      return () => {
        bits[0] = (next() / TWO_POW_32) * 2 - 1;
        return word[0];
      };
    case "scale":
      // 正の scale（|x| ≤ 1 を ±127 に載せた値の近傍 — [0.5, 1.5) / 127）
      return () => {
        bits[0] = (0.5 + next() / TWO_POW_32) / 127;
        return word[0];
      };
    case "f16": {
      // f16 の正規数（|v| ∈ [2^-4, 1)・符号は乱数）を 2 つ詰める（下位 = 偶数添字 —
      // kernels/weight-storage.ts の `unpack2x16float(w[i >> 1])[i & 1]`）
      const half = (): number => {
        const r = next();
        return ((r >>> 12) & 1) << 15 | (11 + (r & 3)) << 10 | ((r >>> 2) & 0x3ff);
      };
      return () => (half() | (half() << 16)) >>> 0;
    }
    case "i8":
      // i8 を 4 つ詰める（レーン j = バイト j — `pack4xI8` / `unpack4xI8` と同じ並び）。値は
      // [-127, 127]（量子化の格子 — kernels/quantize-rows.ts の ±127）
      return () => {
        const r = next();
        let packed = 0;
        for (let lane = 0; lane < 4; lane += 1) {
          const byte = (r >>> (lane * 8)) & 0xff;
          packed |= ((byte === 255 ? 0 : byte - 127) & 0xff) << (lane * 8);
        }
        return packed >>> 0;
      };
  }
};

/**
 * 資源表の `index` 番目を埋める u32 語の列（決定的 — {@link createResources} とテストの CPU 参照が
 * 共有する 1 本の経路）。
 */
export const resourceWordStream = (fill: Exclude<Fill, "none">, index: number): () => number =>
  wordFiller(fill, xorshift(0x9e3779b9 ^ (index + 1)));

/** 計測に使う device まわりの束（掃引 1 回ぶん — 後始末は {@link destroySweepContext}）。 */
export type SweepContext = {
  readonly gpu: GpuContext;
  readonly unit: TimingUnit;
  readonly dp4a: boolean;
  /** 出力の読み戻し（{@link DIGEST_CHUNK_BYTES} 区切り）。 */
  readonly staging: GPUBuffer;
  /** pass の begin / end の timestamp（timestamp-query が無い device では無い）。 */
  readonly timestamps?: {
    readonly querySet: GPUQuerySet;
    readonly resolve: GPUBuffer;
    readonly read: GPUBuffer;
  };
};

/**
 * errorScope（validation + out-of-memory）で包んで `body` を同期実行する。区間は device 単位の
 * スコープロックの中（src/gpu/context.ts「errorScope 区間の不変条件」）。
 */
const scoped = <T>(gpu: GpuContext, label: string, body: () => T): Promise<T> =>
  gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
    pushFailureScopes(gpu.device);
    let value: T;
    try {
      value = body();
    } catch (cause) {
      await discardFailureScopes(gpu.device);
      throw cause;
    }
    const failure = await popFailureScopes(gpu.device, label);
    if (failure !== undefined) throw failure;
    return value;
  });

/**
 * 1 command buffer を積んで完了まで待ち、submit → 完了の壁時計（ns）を返す。errorScope の pop は
 * submit と同じ同期区間で発行し、完了待ちと並べて待つ（pop の往復を壁時計に直列で足さない）。
 */
const submitAndWait = (
  gpu: GpuContext,
  label: string,
  encode: (encoder: GPUCommandEncoder) => void,
): Promise<number> =>
  gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
    const { device } = gpu;
    pushFailureScopes(device);
    let started: number;
    try {
      const encoder = device.createCommandEncoder({ label });
      encode(encoder);
      const commands = encoder.finish();
      started = performance.now();
      device.queue.submit([commands]);
    } catch (cause) {
      await discardFailureScopes(device);
      throw cause;
    }
    const failure = popFailureScopes(device, label);
    const finished = gpu[RUNTIME_INTERNAL].raceDeviceLost(
      device.queue.onSubmittedWorkDone().then(() => performance.now()),
      label,
    );
    const [error, doneAt] = await Promise.all([failure, finished]);
    if (error !== undefined) throw error;
    return (doneAt - started) * 1e6;
  });

/**
 * 掃引の文脈を作る（timestamp の query set と読み戻しの staging を 1 本ずつ）。単位は
 * timestamp-query が有効なら `timestampUnit`、無ければ `wall`。
 */
export const createSweepContext = async (
  gpu: GpuContext,
  timestampUnit: SweepSettings["timestampUnit"],
): Promise<SweepContext> => {
  const { device } = gpu;
  const unit: TimingUnit = gpu.gpuTimingEnabled ? timestampUnit : "wall";
  const stagingBytes = Math.min(DIGEST_CHUNK_BYTES, Math.floor(gpu.limits.maxBufferSize / 8) * 8);
  const created = await scoped(gpu, "geometry-sweep の文脈", () => ({
    staging: device.createBuffer({
      label: "geometry-sweep-staging",
      size: stagingBytes,
      usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    }),
    timestamps: gpu.gpuTimingEnabled
      ? {
        querySet: device.createQuerySet({ type: "timestamp", count: 2 }),
        resolve: device.createBuffer({
          label: "geometry-sweep-resolve",
          size: 16,
          usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
        }),
        read: device.createBuffer({
          label: "geometry-sweep-timestamps",
          size: 16,
          usage: BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
        }),
      }
      : undefined,
  }));
  return {
    gpu,
    unit,
    // i8a8 の整数内積の変種は数値が同一（kernels/linear-i8a8.ts）— 選択は本番の linear と同じ
    // 「WGSL 言語機能の列挙」（src/runtime/session-build.ts:698 / :1090）。attention の本番は
    // カナリア（gpu/attention-dp4a-canary.ts）で選ぶが、掃引は速さを見るだけなので回さない。
    dp4a: dp4aAvailable(gpu.wgslLanguageFeatures),
    staging: created.staging,
    ...(created.timestamps === undefined ? {} : { timestamps: created.timestamps }),
  };
};

/** 文脈の資源を返す（device は呼び手のもの — ここでは破棄しない）。 */
export const destroySweepContext = (context: SweepContext): void => {
  context.staging.destroy();
  context.timestamps?.querySet.destroy();
  context.timestamps?.resolve.destroy();
  context.timestamps?.read.destroy();
};

/** 資源を確保して決定的な値で埋める（失敗したら確保済みを返してから投げる）。 */
const createResources = async (
  context: SweepContext,
  specs: readonly ResourceSpec[],
): Promise<Map<string, GPUBuffer>> => {
  const { gpu } = context;
  const buffers = new Map<string, GPUBuffer>();
  try {
    for (const [index, spec] of specs.entries()) {
      const size = Math.max(4, Math.ceil(spec.bytes / 4) * 4);
      if (size > gpu.limits.maxStorageBufferBindingSize) {
        throw new Error(
          `${spec.name}: ${size} B が maxStorageBufferBindingSize ${gpu.limits.maxStorageBufferBindingSize} を超える`,
        );
      }
      const buffer = await scoped(gpu, `createBuffer(${spec.name})`, () =>
        gpu.device.createBuffer({
          label: `geometry-sweep-${spec.name}`,
          size,
          usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC,
        }));
      buffers.set(spec.name, buffer);
      if (spec.fill === "none") continue;
      const word = resourceWordStream(spec.fill, index);
      for (let offset = 0; offset < size; offset += FILL_CHUNK_BYTES) {
        const chunk = new Uint32Array(Math.min(FILL_CHUNK_BYTES, size - offset) / 4);
        for (let at = 0; at < chunk.length; at += 1) chunk[at] = word();
        await scoped(
          gpu,
          `writeBuffer(${spec.name})`,
          () => gpu.device.queue.writeBuffer(buffer, offset, chunk),
        );
      }
    }
  } catch (cause) {
    for (const buffer of buffers.values()) buffer.destroy();
    throw cause;
  }
  return buffers;
};

/** パイプライン・params・bind group まで解決した dispatch。 */
type ReadyLaunch = {
  readonly launch: Launch;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly params: GPUBuffer;
};

const prepareLaunch = async (
  context: SweepContext,
  launch: Launch,
  resources: ReadonlyMap<string, GPUBuffer>,
): Promise<ReadyLaunch> => {
  const { gpu } = context;
  // パイプラインは device 寿命のキャッシュから引く（同じキーに別の WGSL が来たら
  // PipelineKeyConflictError — 明示幾何のキーが判別力を失っていないことの検出器にもなる）
  const { pipeline, layout, roles } = await gpu[RUNTIME_INTERNAL].withScopeLock(() =>
    gpu[RUNTIME_INTERNAL].pipelines().get(launch.key, launch.wgsl)
  );
  assertBindingRoles(launch, roles);
  const params = await scoped(gpu, `params(${launch.key})`, () => {
    const buffer = gpu.device.createBuffer({
      label: `geometry-sweep-params`,
      size: Math.max(16, launch.params.byteLength),
      usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
    });
    gpu.device.queue.writeBuffer(buffer, 0, launch.params);
    return buffer;
  });
  try {
    const entries: GPUBindGroupEntry[] = [{ binding: 0, resource: { buffer: params } }];
    for (const [index, name] of launch.bindings.entries()) {
      const buffer = resources.get(name);
      if (buffer === undefined) throw new Error(`${launch.key}: 資源 ${name} が無い`);
      entries.push({ binding: index + 1, resource: { buffer } });
    }
    const bindGroup = await scoped(
      gpu,
      `createBindGroup(${launch.key})`,
      () => gpu.device.createBindGroup({ layout, entries }),
    );
    return { launch, pipeline, bindGroup, params };
  } catch (cause) {
    params.destroy();
    throw cause;
  }
};

type PassSample = {
  readonly wallNs: number;
  /** timestamp の差（単位 = 文脈の unit）。計測しない pass では無い。 */
  readonly gpu?: number;
  readonly clamped: boolean;
};

/**
 * 1 compute pass に同じ dispatch を `reps` 本積んで回す。`timed` なら pass の begin / end に
 * timestamp を書いて差を返す。`clear` を渡すと pass の前にそのバッファを 0 に戻す。
 */
const runPass = async (
  context: SweepContext,
  ready: ReadyLaunch,
  reps: number,
  timed: boolean,
  clear?: GPUBuffer,
): Promise<PassSample> => {
  const timestamps = timed ? context.timestamps : undefined;
  const [x, y, z] = ready.launch.workgroups;
  const wallNs = await submitAndWait(context.gpu, ready.launch.key, (encoder) => {
    if (clear !== undefined) encoder.clearBuffer(clear);
    const pass = encoder.beginComputePass(
      timestamps === undefined ? {} : {
        timestampWrites: {
          querySet: timestamps.querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      },
    );
    pass.setPipeline(ready.pipeline);
    pass.setBindGroup(0, ready.bindGroup);
    for (let rep = 0; rep < reps; rep += 1) pass.dispatchWorkgroups(x, y, z);
    pass.end();
    if (timestamps !== undefined) {
      encoder.resolveQuerySet(timestamps.querySet, 0, 2, timestamps.resolve, 0);
      encoder.copyBufferToBuffer(timestamps.resolve, 0, timestamps.read, 0, 16);
    }
  });
  if (timestamps === undefined) return { wallNs, clamped: false };
  await context.gpu[RUNTIME_INTERNAL].raceDeviceLost(
    timestamps.read.mapAsync(MAP_MODE.READ),
    "timestamp の読み戻し",
  );
  const stamps = new BigUint64Array(timestamps.read.getMappedRange());
  const delta = stamps[1] - stamps[0];
  timestamps.read.unmap();
  // MUST: 負値は 0 に丸めて件数を残す（黙って捨てない — ADR 0021 と同じ）。min の候補からは
  // 呼び手（timeLaunch）が外す
  return { wallNs, gpu: delta < 0n ? 0 : Number(delta), clamped: delta < 0n };
};

const hex = (bytes: ArrayBuffer): string =>
  Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");

/**
 * 出力の digest = **64 MiB 区切りの各 SHA-256 を連結したものの SHA-256**（2 段）。S は 1 GiB に
 * なるので、区切りごとに staging へ写して読み、JS 側の峰を区切り 1 本に抑える。同じ出力 ⇔ 同じ
 * digest なので、幾何どうしの一致の判定には平の SHA-256 と同じ力を持つ。
 */
export const outputDigest = async (context: SweepContext, buffer: GPUBuffer): Promise<string> => {
  const { gpu, staging } = context;
  const chunks: ArrayBuffer[] = [];
  for (let offset = 0; offset < buffer.size; offset += staging.size) {
    const length = Math.min(staging.size, buffer.size - offset);
    await submitAndWait(gpu, "出力の読み戻し", (encoder) => {
      encoder.copyBufferToBuffer(buffer, offset, staging, 0, length);
    });
    await gpu[RUNTIME_INTERNAL].raceDeviceLost(
      staging.mapAsync(MAP_MODE.READ, 0, length),
      "出力の読み戻し",
    );
    // digest は呼び出し時点で入力を写す（WebCrypto の規定）が、実装差に寄らないよう unmap の前に
    // JS 側へ複製してから渡す
    const copy = new Uint8Array(staging.getMappedRange(0, length)).slice();
    staging.unmap();
    chunks.push(await crypto.subtle.digest("SHA-256", copy));
  }
  return await joinChunkDigests(chunks);
};

/** 区切りごとの SHA-256 を連結して SHA-256 を採る（{@link outputDigest} の 2 段目 — テストも使う）。 */
export const joinChunkDigests = async (chunks: readonly ArrayBuffer[]): Promise<string> => {
  const joined = new Uint8Array(chunks.length * 32);
  chunks.forEach((chunk, index) => joined.set(new Uint8Array(chunk), index * 32));
  return hex(await crypto.subtle.digest("SHA-256", joined));
};

const describeCause = (cause: unknown): string =>
  cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);

/** 失敗した組の行（掃引は止めない）。 */
const errorRow = (
  sweepCase: SweepCase,
  candidate: GeometryCandidate,
  isDefault: boolean,
  cause: unknown,
): SweepRow => ({
  caseId: sweepCase.id,
  op: sweepCase.op,
  shape: caseShape(sweepCase),
  censusCount: sweepCase.censusCount,
  geometry: candidate.name,
  geometryParams: candidate.geometry,
  isDefault,
  error: describeCause(cause),
});

/**
 * 空回し（計測の規約 3）。累計は単位 `ns` なら pass の GPU 時間、それ以外は壁時計で数える
 * （Deno の raw tick は ns ではないので下限の ns と比べられない）。
 */
const warmUp = async (context: SweepContext, ready: ReadyLaunch, reps: number): Promise<void> => {
  const timed = context.unit === "ns";
  let total = 0;
  for (let pass = 0; pass < WARMUP_MAX_PASSES; pass += 1) {
    if (pass >= WARMUP_MIN_RUNS && total >= WARMUP_NS) return;
    const sample = await runPass(context, ready, reps, timed);
    total += timed ? sample.gpu ?? 0 : sample.wallNs;
  }
};

/** 1 幾何ぶんの時間の記録（行の計測欄）。 */
type Timing = {
  readonly reps: number;
  readonly rounds: readonly number[];
  readonly wallRounds: readonly number[];
  readonly clampedNegativeSamples: number;
  /** 負でなかった round の min ÷ reps（全 round が負なら無い）。 */
  readonly perDispatch?: number;
  readonly wallNsPerDispatch: number;
};

/**
 * 1 幾何を測る: 出力を 0 に戻して 1 本流し → `reps` を見積り → 空回し → `rounds` 回の計測
 * （計測の規約 1〜3）。
 */
const timeLaunch = async (
  context: SweepContext,
  ready: ReadyLaunch,
  output: GPUBuffer,
  rounds: number,
): Promise<Timing> => {
  // 1 本目は出力を 0 に戻してから（書き漏らしたタイルが digest の不一致として見える）
  await runPass(context, ready, 1, false, output);
  // reps の見積り（モジュール doc の NOTE — 壁時計で倍々に積み、最後の 2 点の傾きを採る）
  let reps = 1;
  let previous: { readonly reps: number; readonly wallNs: number } | undefined;
  let probeNs: number;
  for (;;) {
    const { wallNs } = await runPass(context, ready, reps, false);
    if (wallNs >= PROBE_MIN_NS || reps >= SWEEP_MAX_REPS) {
      const average = wallNs / reps;
      const slope = previous === undefined
        ? 0
        : (wallNs - previous.wallNs) / (reps - previous.reps);
      // 傾きが立たない（1 点目で届いた / 揺れで負）なら平均へ倒す。揺れで傾きが潰れても pass が
      // 目標の 4 倍を超えないよう、平均の 1/4 を下限に置く
      probeNs = slope > 0 ? Math.max(slope, average / 4) : average;
      break;
    }
    previous = { reps, wallNs };
    reps = Math.min(SWEEP_MAX_REPS, reps * 2);
  }
  reps = calibrateReps(probeNs, TARGET_PASS_MS, SWEEP_MAX_REPS);
  await warmUp(context, ready, reps);
  const gpuRounds: number[] = [];
  const wallRounds: number[] = [];
  const kept: number[] = [];
  let clamped = 0;
  for (let round = 0; round < rounds; round += 1) {
    const sample = await runPass(context, ready, reps, context.unit !== "wall");
    const value = sample.gpu ?? sample.wallNs;
    gpuRounds.push(value);
    wallRounds.push(sample.wallNs);
    // MUST: 0 に丸めた sample は記録に残すが min の候補から外す（0 が min を取ると速さの比が
    // 無限大に化ける）
    if (sample.clamped) clamped += 1;
    else kept.push(value);
  }
  return {
    reps,
    rounds: gpuRounds,
    wallRounds,
    clampedNegativeSamples: clamped,
    ...(kept.length === 0 ? {} : { perDispatch: Math.min(...kept) / reps }),
    wallNsPerDispatch: Math.min(...wallRounds) / reps,
  };
};

/** 候補 1 つの dispatch を用意して測る（params は測り終えたら返す）。 */
const timeCandidate = async (
  context: SweepContext,
  plan: CasePlan,
  resources: ReadonlyMap<string, GPUBuffer>,
  candidate: GeometryCandidate,
  rounds: number,
): Promise<{ readonly launch: Launch; readonly timing: Timing }> => {
  const output = resources.get(plan.output);
  if (output === undefined) throw new Error(`出力 ${plan.output} が無い`);
  const ready = await prepareLaunch(context, plan.launch(candidate), resources);
  try {
    return { launch: ready.launch, timing: await timeLaunch(context, ready, output, rounds) };
  } finally {
    ready.params.destroy();
  }
};

const ALL_ROUNDS_NEGATIVE =
  "timestamp が全 round で負（0 に丸めた sample しか無く min を採れない）";

const measureGeometry = async (
  context: SweepContext,
  sweepCase: SweepCase,
  plan: CasePlan,
  resources: ReadonlyMap<string, GPUBuffer>,
  candidate: GeometryCandidate,
  isDefault: boolean,
  rounds: number,
): Promise<SweepRow> => {
  const { launch, timing } = await timeCandidate(context, plan, resources, candidate, rounds);
  const measured = {
    caseId: sweepCase.id,
    op: sweepCase.op,
    shape: caseShape(sweepCase),
    censusCount: sweepCase.censusCount,
    geometry: candidate.name,
    geometryParams: candidate.geometry,
    isDefault,
    key: launch.key,
    workgroups: launch.workgroups,
    reps: timing.reps,
    rounds: timing.rounds,
    wallRounds: timing.wallRounds,
    clampedNegativeSamples: timing.clampedNegativeSamples,
  } satisfies SweepRow;
  if (timing.perDispatch === undefined) {
    return { ...measured, wallNsPerDispatch: timing.wallNsPerDispatch, error: ALL_ROUNDS_NEGATIVE };
  }
  // raw tick は ns ではないので TFLOPS は壁時計から出す（report.ts の `tflops` の定義）
  const realNs = context.unit === "ns" ? timing.perDispatch : timing.wallNsPerDispatch;
  const output = resources.get(plan.output);
  if (output === undefined) throw new Error(`${sweepCase.id}: 出力 ${plan.output} が無い`);
  return {
    ...measured,
    perDispatch: timing.perDispatch,
    wallNsPerDispatch: timing.wallNsPerDispatch,
    ...(realNs > 0 ? { tflops: caseFlops(sweepCase) / realNs / 1e3 } : {}),
    outputSha256: await outputDigest(context, output),
  };
};

/** 1 ケースの結果（全幾何の行と、ケース単位の記録）。 */
export type CaseResult = {
  readonly rows: readonly SweepRow[];
  readonly summary: CaseSummary;
};

/**
 * 1 ケースを候補の全幾何で測る（既定幾何を先頭に置き、残りは候補の順）。測り終えたら既定幾何を
 * もう 1 度測って初回との比をケース単位の記録に残す（計測の規約 4）。計画・資源の確保・前段が
 * 失敗したら全幾何を失敗の行にする。device を失ったら残りを失敗の行にし、中断されたら残りは
 * 行を作らない（再測定もしない）。
 */
export const sweepCase = async (
  context: SweepContext,
  target: SweepCase,
  candidates: readonly GeometryCandidate[],
  settings: Pick<SweepSettings, "rounds">,
  hooks: SweepHooks = {},
): Promise<CaseResult> => {
  const { gpu } = context;
  const rows: SweepRow[] = [];
  const emit = (row: SweepRow): void => {
    rows.push(row);
    hooks.onRow?.(row);
  };
  const finish = (summary: CaseSummary): CaseResult => {
    hooks.onCase?.(summary);
    return { rows, summary };
  };
  let plan: CasePlan;
  try {
    plan = casePlan(target, gpu.limits.maxComputeWorkgroupsPerDimension, context.dp4a);
  } catch (cause) {
    // 既定幾何も解けていないので、候補の行だけを失敗として残す
    candidates.forEach((candidate) => emit(errorRow(target, candidate, false, cause)));
    return finish({ caseId: target.id });
  }
  const ordered = [
    plan.defaultCandidate,
    ...candidates.filter((candidate) => candidate.name !== plan.defaultCandidate.name),
  ];
  let resources: Map<string, GPUBuffer>;
  try {
    hooks.onProgress?.(`${target.id}: 入力を用意中`);
    resources = await createResources(context, plan.resources);
  } catch (cause) {
    ordered.forEach((candidate, index) => emit(errorRow(target, candidate, index === 0, cause)));
    return finish({ caseId: target.id });
  }
  try {
    try {
      for (const launch of plan.prelude) {
        const ready = await prepareLaunch(context, launch, resources);
        try {
          await runPass(context, ready, 1, false);
        } finally {
          ready.params.destroy();
        }
      }
    } catch (cause) {
      ordered.forEach((candidate, index) => emit(errorRow(target, candidate, index === 0, cause)));
      return finish({ caseId: target.id });
    }
    let reference: DefaultReference = {};
    for (const [index, candidate] of ordered.entries()) {
      if (hooks.signal?.aborted === true) break;
      const isDefault = index === 0;
      if (gpu.lost !== undefined) {
        emit(errorRow(target, candidate, isDefault, new Error(`device lost: ${gpu.lost.message}`)));
        continue;
      }
      hooks.onProgress?.(`${target.id}: ${candidate.name}（${index + 1}/${ordered.length}）`);
      let row: SweepRow;
      try {
        row = await measureGeometry(
          context,
          target,
          plan,
          resources,
          candidate,
          isDefault,
          settings.rounds,
        );
      } catch (cause) {
        row = errorRow(target, candidate, isDefault, cause);
      }
      if (isDefault) {
        reference = { perDispatch: row.perDispatch, outputSha256: row.outputSha256 };
      }
      emit(compareToDefault(row, reference));
    }
    const initial = reference.perDispatch;
    if (initial === undefined || hooks.signal?.aborted === true || gpu.lost !== undefined) {
      return finish({ caseId: target.id });
    }
    hooks.onProgress?.(`${target.id}: 既定幾何の再測定`);
    try {
      const { timing } = await timeCandidate(
        context,
        plan,
        resources,
        plan.defaultCandidate,
        settings.rounds,
      );
      if (timing.perDispatch === undefined) throw new Error(ALL_ROUNDS_NEGATIVE);
      return finish({
        caseId: target.id,
        defaultRepeat: {
          perDispatch: timing.perDispatch,
          driftRatio: timing.perDispatch / initial,
        },
      });
    } catch (cause) {
      return finish({ caseId: target.id, defaultRepeatError: describeCause(cause) });
    }
  } finally {
    for (const buffer of resources.values()) buffer.destroy();
  }
};

/** 掃引全体の結果（行と、ケース単位の記録）。 */
export type SweepResult = {
  readonly rows: readonly SweepRow[];
  readonly cases: readonly CaseSummary[];
};

/** ケースの列を順に掃引する（device を失ったか中断されたら、残りのケースは回さない）。 */
export const runSweep = async (
  context: SweepContext,
  cases: readonly SweepCase[],
  settings: SweepSettings,
  hooks: SweepHooks = {},
): Promise<SweepResult> => {
  const rows: SweepRow[] = [];
  const summaries: CaseSummary[] = [];
  for (const target of cases) {
    if (context.gpu.lost !== undefined || hooks.signal?.aborted === true) break;
    const result = await sweepCase(
      context,
      target,
      candidatesFor(target, settings.candidateSet),
      settings,
      hooks,
    );
    rows.push(...result.rows);
    summaries.push(result.summary);
  }
  return { rows, cases: summaries };
};

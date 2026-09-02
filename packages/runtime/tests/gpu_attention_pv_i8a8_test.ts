// 融合 attention ③PV の **i8a8 変種**（P̃ = round(127·exp(S−m)) の非実体化 × V の per-column i8）
// の実行経路。
//
// 設計 = docs/research/2026-08-04-attention-a8-design.md §2.2 / §2.3 / §4.2、実装 =
// src/codegen/strided.ts（Vᵀ・無改変で再利用）+ src/kernels/quantize-rows.ts（無改変）+
// src/kernels/attention-i8a8.ts、opt-in = `SessionOptions.attentionCompute: "a8"`（既定 "f32"）。
//
// ## このファイルが固定する数値契約（①QK と決定的に違う — 設計 §4.2）
//
// ③PV の O に**素朴な atol=0 は立たない**。A 側の量子化値が `round(127·exp(S−m))` で、
// `exp` は WGSL と JS でビット一致する保証が無く、`127·exp` は [0,127] に密に分布するので
// 丸め境界に落ちる要素が必ず出るため。そこで契約を **2 段に割る**:
//
// (a) **整数を受け取ってからの純関数**（`referenceAttentionPvI8a8Core`）とは **atol=0**。
//     GPU が使った `qP` は観測できない（P は実体化されない）ので、**V を単位行列にした
//     もう 1 本の実行を「qP の読み出し器」として使う**（vq = 127·I / vs = 1/127 になるので
//     O が 127·qP·prow·vs にしかならない）。vq / vs は一時バッファをそのまま読み戻す。
// (a2) **Vᵀ の量子化**（strided permute + quantize_rows）は TS 参照と atol=0。
//     permute の軸取り違えと per-column 量子化の唯一の機械的検出器。
// (b) **`qP` の生成**は別の門: TS 参照との不一致は**必ず ±1 段**（±2 が 1 件でも出たら
//     実装バグ）で、不一致率そのものを記録する。恒真化しない門にするための形。
// (c) dp4a とエミュは atol=0（拡張の有無は速度にしか効かない）。
//
// ## この波で意図的に検証していないもの
//
// - grid-stride の縮退ハーネス（tests/gpu_gridstride_test.ts）への追加は**不要**。新カーネルは
//   「1 workgroup = 1 出力タイル」で全域を覆う形（上限超過は `tiledWorkgroups` の fail loudly）。
//   同時に走る `quantize_rows` と `strided` は既に載っている。
// - 分解経路とのビット同一（tests/gpu_attention_parity_test.ts）は i8a8 では原理的に成立
//   しない。同テストは f32 / f16 変種専用の門として据え置く（別キーなので構造的に混ざらない）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../src/codegen/strided.ts";
import { openModel } from "../src/format/container.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { defaultI8a8Geometry, i8a8TileM, i8a8TileN } from "../src/kernels/i8a8-geometry.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../src/kernels/quantize-rows.ts";
import { LINEAR_I8A8_MAX_K } from "../src/kernels/linear-i8a8.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8UsesVec4,
} from "../src/kernels/attention-i8a8.ts";
import {
  attentionPvKey,
  attentionQkKey,
  attentionStatsKey,
  attentionStatsRegCache,
} from "../src/kernels/attention.ts";
import {
  quantizeRowsReference,
  quantizeRowsTieMargin,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
} from "../src/reference/i8a8.ts";
import {
  createSession,
  I8A8_DOT,
  type I8a8Dot,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** atol=0 を主張してよい丸め境界からの余裕（2.5 ULP ≈ 1.9e-5 に対し 50 倍以上）。 */
const TIE_MARGIN = 1e-3;

/**
 * P̃ の格子と `1/127`（**観測の算術にだけ使う** — 参照側の定数は src/reference/i8a8.ts が
 * 独立に持つ）。
 */
const P_ABS_MAX = 127;
const INV_P_ABS_MAX = Math.fround(1 / P_ABS_MAX);

/**
 * atol=0 の全数比較。`===` なので ±0 は同一視するが、それ以外は 1 ビットの差も許さない。
 */
const assertExact = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: 長さ`);
  for (let i = 0; i < actual.length; i += 1) {
    if (Number.isNaN(expected[i])) {
      assert(Number.isNaN(actual[i]), `${where}[${i}]: 参照 NaN に対し ${actual[i]}`);
      continue;
    }
    assert(actual[i] === expected[i], `${where}[${i}]: GPU ${actual[i]} / 参照 ${expected[i]}`);
  }
};

// ---------------------------------------------------------------------------
// 形状群とデータ
// ---------------------------------------------------------------------------

type PvShape = {
  readonly name: string;
  /** B·H を畳んだバッチ軸。**2 以上が batch base の唯一の検出器**。 */
  readonly batch: number;
  /** 出力の行 = M。 */
  readonly m: number;
  /** 縮約軸 = N（`N % 4 == 0` が i8a8 の適格条件）。 */
  readonly n: number;
  /** 出力の列 = D（`D % 4 == 0` が v4 書き出しの条件）。 */
  readonly d: number;
};

/**
 * 形の踏み分け。
 *
 * MUST: **B·H ≥ 2 かつ head ごとに統計もパターンも違う**形を持つ（バッチ base の唯一の
 * 検出器 — head ごとに大きさだけを変えると量子化後の整数が同じになり、ペイロード側の base の
 * 取り違えが値に出ない。波 1 の教訓）。
 * MUST: **m / d をタイル辺 64 を跨がせる**（行 inv と列 scale の取り違えの検出器）。
 * MUST: **v4 経路（D%4==0）とスカラ経路を対で持つ**（O の書き出しが変種ごとに別の式）。
 * MUST: **N % 16 != 0** を含む（K タイル 16 の端数 = P̃ タイルの 0 埋めが効いていることの
 * 検出器。N=20 は k4=5 で 2 タイル目の 4 パック中 1 パックしか有効でない）。
 */
const PV_SHAPES: readonly PvShape[] = [
  // v4 / m は 64 タイル 2 枚 / N=68 は k4=17 で K タイル 5 枚（端数 1 パック）
  { name: "v4 B·H6 M65 N68 D20", batch: 6, m: 65, n: 68, d: 20 },
  // スカラ（D % 4 != 0）で 1 タイルに収まる形 + head 2 枚 / N=20 は端数タイル
  { name: "スカラ B·H2 M17 N20 D19", batch: 2, m: 17, n: 20, d: 19 },
  // K タイルが 9 枚（k4=33・端数 1 パック）/ D=5 はスカラ / M=70 で行タイル 2 枚
  { name: "スカラ 長K B·H2 M70 N132 D5", batch: 2, m: 70, n: 132, d: 5 },
  // 1 タイルに収まる最小形（ガードが全て真のまま通る経路）
  { name: "v4 小 B·H1 M3 N8 D4", batch: 1, m: 3, n: 8, d: 4 },
  // 実測形（DiT の self-attention）— **列タイル 2 枚**（D=128）で列 scale の束縛を踏む
  { name: "DiT 形 B·H4 M64 N64 D128", batch: 4, m: 64, n: 64, d: 128 },
];

/**
 * S の素の値（決定的 — 乱数は使わない）。head / 行ごとに**パターンが変わる**ので、
 * P̃ は行内で 0（丸めで消える要素）から 127（行の最大）まで広がる。
 *
 * MUST: 刻みを**密**にする（`0.5` 刻みのような「きれいな」値だけだと `127·exp` が丸め境界の
 * 近傍に一度も落ちず、(b) の門が実質恒真になる — 実測で 62,088 要素中 0 件だった）。
 * 2 項を非共約の刻みで重ねて値域 15 を細かく埋める。
 */
const scoreAt = (b: number, row: number, col: number): number => {
  const index = col * 7 + row * 3 + b * 5;
  return ((index % 23) - 11) * 0.6373 + ((index * 13) % 11) * 0.1237 - 0.4291;
};

/**
 * V の素の値。**列（D）方向にも行（N）方向にも統計が偏る**形にしてある — per-column scale を
 * per-token（行 n ごと）へ取り違える誤りは、この偏りがあるときだけ値に出る（設計 §2.3）。
 * head ごとにパターンをずらすのは、大きさだけの差では量子化後の整数が一致してしまい
 * ペイロード base の取り違えが検出できないため（波 1 の教訓）。
 */
const valueAt = (b: number, row: number, col: number, d: number): number => {
  // MUST: 刻みを scale の格子と共約にしない（`x/s` が半整数に乗ると、GPU の除算が 2.5 ULP
  // まで許されているぶん量子化値が ±1 段揺れて (a2) の atol=0 が「たまたま落ちる」）。
  // 余裕は {@link quantizeRowsTieMargin} で毎回実測して門にする。
  const base = ((((row * d + col + b * 11) * 5) % 17) - 8) * 0.2731 + 0.1237;
  const colGain = 0.04 * Math.pow(3.7, col % 5);
  const rowGain = 0.25 + (row % 6) * 0.83;
  return base * colGain * rowGain * (1 + 0.53 * b);
};

type PreparedPv = {
  readonly s: Float32Array<ArrayBuffer>;
  /** 行統計 `[batch·m, 2]`（`[0]` = 行の最大 / `[1]` = `1/Σexp(S−m)`）。 */
  readonly stats: Float32Array<ArrayBuffer>;
  readonly rowInv: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
  /** Vᵀ`[batch, d, n]`（GPU の strided permute が作るはずの中身）。 */
  readonly vt: Float32Array<ArrayBuffer>;
  /** Vᵀ の `x/s` の丸め境界からの余裕。 */
  readonly margin: number;
};

const preparePv = (shape: PvShape): PreparedPv => {
  const { batch, m, n, d } = shape;
  const s = new Float32Array(batch * m * n);
  const stats = new Float32Array(batch * m * 2);
  const rowInv = new Float32Array(batch * m);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      let max = Number.NEGATIVE_INFINITY;
      for (let col = 0; col < n; col += 1) {
        const value = Math.fround(scoreAt(b, row, col));
        s[(b * m + row) * n + col] = value;
        if (value > max) max = value;
      }
      let sum = 0;
      for (let col = 0; col < n; col += 1) {
        sum = Math.fround(sum + Math.fround(Math.exp(s[(b * m + row) * n + col] - max)));
      }
      const inv = Math.fround(1 / sum);
      stats[(b * m + row) * 2] = max;
      stats[(b * m + row) * 2 + 1] = inv;
      rowInv[b * m + row] = inv;
    }
  }
  const v = new Float32Array(batch * n * d);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < d; col += 1) {
        v[(b * n + row) * d + col] = valueAt(b, row, col, d);
      }
    }
  }
  const vt = transpose(v, batch, n, d);
  return { s, stats, rowInv, v, vt, margin: quantizeRowsTieMargin(vt, batch * d, n) };
};

/** `[batch, n, d]` → `[batch, d, n]`（GPU 側は既存の strided permute がやる仕事）。 */
const transpose = (
  v: ArrayLike<number>,
  batch: number,
  n: number,
  d: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * d * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < d; col += 1) {
        out[(b * d + col) * n + row] = v[(b * n + row) * d + col];
      }
    }
  }
  return out;
};

/** head ごとの単位行列 `[batch, n, n]`（qP の読み出し器 — 下の {@link observeQuantizedP}）。 */
const identityV = (batch: number, n: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * n * n);
  for (let b = 0; b < batch; b += 1) {
    for (let i = 0; i < n; i += 1) out[(b * n + i) * n + i] = 1;
  }
  return out;
};

// ---------------------------------------------------------------------------
// 生 dispatch（executor の ③PV i8a8 と同じ 3 dispatch を流し、一時バッファも読み戻す）
// ---------------------------------------------------------------------------

const readbackBytes = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  bytes: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const size = Math.max(4, bytes);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Uint8Array(copy, 0, bytes);
};

type PvRun = {
  readonly o: Float32Array<ArrayBuffer>;
  /** GPU が使った Vᵀ の量子化値（`[batch, d, n]` の平坦 i8 — 4 詰めをそのまま読んだもの）。 */
  readonly vq: Int8Array<ArrayBuffer>;
  /** GPU が使った Vᵀ の行 scale（`[batch·d]`）。 */
  readonly vs: Float32Array<ArrayBuffer>;
};

/**
 * `strided`（Vᵀ）→ `quantize_rows` → i8a8 GEMM の 3 dispatch を生で流す。
 * S と行統計は**テストが直接与える**（③ の契約は「S と stats を受け取って O を書く」なので、
 * ①② を通す必要が無い）。
 */
const runPvI8a8 = async (
  gpu: GpuContext,
  shape: { readonly batch: number; readonly m: number; readonly n: number; readonly d: number },
  input: {
    readonly s: Float32Array<ArrayBuffer>;
    readonly stats: Float32Array<ArrayBuffer>;
    readonly v: Float32Array<ArrayBuffer>;
  },
  dot: I8a8Dot,
  where: string,
): Promise<PvRun> => {
  const { batch, m, n, d } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const host = (source: Float32Array<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, source);
      return buffer;
    };
    const scores = host(input.s);
    const stats = host(input.stats);
    const v = host(input.v);

    // (a) v[batch,N,D] → vT[batch,D,N]（既存の strided 読みコピー族 = permute）
    const stridedSpec = { dtype: "f32" } as const;
    const permuteKey = stridedKey(stridedSpec);
    const { pipeline: permutePipeline, layout: permuteLayout } = await cache.get(
      permuteKey,
      stridedWgsl(stridedSpec),
    );
    const permuteParams = arena.allocHostWritten(40, STORAGE_IN);
    gpu.device.queue.writeBuffer(
      permuteParams,
      0,
      stridedParams([batch, d, n], permuteSrcStrides([batch, n, d], [0, 2, 1]), 0),
    );
    const vt = arena.allocStorage(Math.max(4, batch * d * n * 4));
    arena.retain(vt, 0, { pinned: true });
    scheduler.dispatch(
      permutePipeline,
      gpu.device.createBindGroup({
        layout: permuteLayout,
        entries: [
          { binding: 0, resource: { buffer: permuteParams } },
          { binding: 1, resource: { buffer: v } },
          { binding: 2, resource: { buffer: vt } },
        ],
      }),
      [gridStrideWorkgroups(batch * d * n, STRIDED_WORKGROUP_SIZE, limit), 1, 1],
      permuteKey,
    );

    // (b) Vᵀ の per-row 量子化（= V の per-column 量子化）
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    const quantizeParams = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(quantizeParams, 0, quantizeRowsParams(batch * d, n));
    const vq = arena.allocStorage(Math.max(4, batch * d * n));
    arena.retain(vq, 0, { pinned: true });
    const vs = arena.allocStorage(Math.max(4, batch * d * 4));
    arena.retain(vs, 0, { pinned: true });
    scheduler.dispatch(
      quantizePipeline,
      gpu.device.createBindGroup({
        layout: quantizeLayout,
        entries: [
          { binding: 0, resource: { buffer: quantizeParams } },
          { binding: 1, resource: { buffer: vt } },
          { binding: 2, resource: { buffer: vq } },
          { binding: 3, resource: { buffer: vs } },
        ],
      }),
      [gridStrideWorkgroups(batch * d, 1, limit), 1, 1],
      QUANTIZE_ROWS_KEY,
    );

    // (c) 整数内積の GEMM（P̃ は A タイル充填で作る）
    const v4 = attentionPvI8a8UsesVec4(d);
    const dp4a = dot === "dp4a";
    const pvGeometry = defaultI8a8Geometry("attention_pv");
    const pipelineKey = attentionPvI8a8Key(v4, dp4a);
    const { pipeline, layout } = await cache.get(pipelineKey, attentionPvI8a8Wgsl(v4, dp4a));
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionPvI8a8Params(m, d, n));
    const out = arena.allocStorage(Math.max(4, batch * m * d * 4));
    arena.retain(out, 0, { pinned: true });
    scheduler.dispatch(
      pipeline,
      gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: scores } },
          { binding: 2, resource: { buffer: vq } },
          { binding: 3, resource: { buffer: stats } },
          { binding: 4, resource: { buffer: out } },
          { binding: ATTENTION_PV_V_SCALE_BINDING, resource: { buffer: vs } },
        ],
      }),
      [
        // MUST: 辺は ③PV 自身の幾何から導く（①QK とは別の既定 — キーと生成物の解決点と同じ）
        tiledWorkgroups(d, i8a8TileN(pvGeometry), limit, where),
        tiledWorkgroups(m, i8a8TileM(pvGeometry), limit, where),
        tiledWorkgroups(batch, 1, limit, where),
      ],
      pipelineKey,
    );
    await scheduler.flush();

    const outBytes = await readbackBytes(gpu.device, out, batch * m * d * 4);
    const vqBytes = await readbackBytes(gpu.device, vq, batch * d * n);
    const vsBytes = await readbackBytes(gpu.device, vs, batch * d * 4);
    return {
      o: new Float32Array(outBytes.buffer, 0, batch * m * d),
      // pack4xI8 は成分 i を i バイト目へ置くので、平坦 i8 の並びはバイト列そのもの
      vq: new Int8Array(vqBytes.buffer, 0, batch * d * n),
      vs: new Float32Array(vsBytes.buffer, 0, batch * d),
    };
  } finally {
    await arena.destroy();
  }
};

/**
 * GPU が A タイルで作った `qP` を観測する。
 *
 * V を head ごとの単位行列にすると `vq = 127·I` / `vs = 1/127` になるので、
 * `O[m,j] = f32(127·qP[m,j]) · (prow · vs[j])` = **qP の読み出し器**になる。
 * MUST: 前提（vq が対角 127・非対角 0）を assert してから割る — 前提が崩れたまま割ると
 * 観測値が静かにずれ、(a) の atol=0 が「たまたま通る」形になる。
 */
const observeQuantizedP = (
  run: PvRun,
  shape: { readonly batch: number; readonly m: number; readonly n: number },
  rowInv: ArrayLike<number>,
): Int8Array<ArrayBuffer> => {
  const { batch, m, n } = shape;
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      assertEquals(run.vs[b * n + row], INV_P_ABS_MAX, `単位行列の列 scale [${b},${row}]`);
      for (let col = 0; col < n; col += 1) {
        assertEquals(
          run.vq[(b * n + row) * n + col],
          row === col ? P_ABS_MAX : 0,
          `単位行列の量子化値 [${b},${row},${col}]`,
        );
      }
    }
  }
  const qp = new Int8Array(batch * m * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const prow = Math.fround(rowInv[b * m + row] * INV_P_ABS_MAX);
      for (let col = 0; col < n; col += 1) {
        const combined = Math.fround(prow * run.vs[b * n + col]);
        const raw = run.o[(b * m + row) * n + col] / (P_ABS_MAX * combined);
        const rounded = Math.round(raw);
        assert(
          Math.abs(raw - rounded) < 1e-3 && rounded >= 0 && rounded <= P_ABS_MAX,
          `qP の読み出しが整数にならない [${b},${row},${col}]: ${raw}`,
        );
        qp[(b * m + row) * n + col] = rounded;
      }
    }
  }
  return qp;
};

// ---------------------------------------------------------------------------
// 丸め境界からの余裕（Vᵀ の量子化を atol=0 で突合してよいことの門）
// ---------------------------------------------------------------------------

/** `quantize_rows` の dispatch 数（小 D 変種でキーが幾何ごとに割れるので接頭辞で束ねる）。 */
const quantizeRowsDispatches = (byKey: ReadonlyMap<string, number>): number =>
  [...byKey].filter(([key]) => key.startsWith(QUANTIZE_ROWS_KEY)).reduce(
    (sum, [, count]) => sum + count,
    0,
  );

Deno.test("attention_pv i8a8: 形状群の Vᵀ は丸め境界から十分離れている（vq 突合の前提）", () => {
  // GPU の除算（`x/s`）は 2.5 ULP まで許されるので、`x/s` が半整数の近傍にある要素は ±1 段
  // 揺れうる。余裕を**毎回実測**して門にしないと、(a2) の atol=0 は「たまたま通っているだけ」。
  let worst = Number.POSITIVE_INFINITY;
  for (const shape of PV_SHAPES) {
    const { margin } = preparePv(shape);
    assert(margin > TIE_MARGIN, `${shape.name}: 丸め境界からの余裕が ${margin} しかない`);
    worst = Math.min(worst, margin);
  }
  assert(worst > TIE_MARGIN, `最悪余裕 ${worst}`);
});

// ---------------------------------------------------------------------------
// (a) 整数を受け取ってからの純関数との atol=0 / (a2) Vᵀ の量子化の atol=0
// ---------------------------------------------------------------------------

Deno.test({
  name:
    "attention_pv i8a8 の O が TS 参照（整数の手前で割った純関数）と atol=0 で一致する（v4 / スカラ / タイル端 / K 端数 / B·H≥2・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of PV_SHAPES) {
        const { batch, m, n, d } = shape;
        const prepared = preparePv(shape);
        const run = await runPvI8a8(gpu, shape, prepared, "dp4a", shape.name);

        // (a2) Vᵀ（permute の実体化）とその per-column 量子化が TS 参照とビット一致する。
        // **permute の軸取り違えと per-token 化の唯一の機械的検出器**。
        const expectedV = quantizeRowsReference(prepared.vt, batch * d, n);
        assertExact(run.vq, expectedV.q, `${shape.name}: Vᵀ の量子化値`);
        assertExact(run.vs, expectedV.scale, `${shape.name}: Vᵀ の列 scale`);
        // 恒真化の門: 列 scale が全部同じなら「列で引く」ことを何も検証していない
        assert(
          new Set([...run.vs]).size > Math.max(2, batch),
          `${shape.name}: 列 scale が散らばっていない（軸の検出器が成立しない）`,
        );

        // (a) GPU が使った qP を観測してから、整数だけの参照と突き合わせる
        const observed = await runPvI8a8(
          gpu,
          { batch, m, n, d: n },
          { s: prepared.s, stats: prepared.stats, v: identityV(batch, n) },
          "dp4a",
          `${shape.name}（qP 観測）`,
        );
        const qp = observeQuantizedP(observed, shape, prepared.rowInv);
        const expected = referenceAttentionPvI8a8Core({
          qp,
          vq: run.vq,
          rowInv: prepared.rowInv,
          vs: run.vs,
          batch,
          m,
          n,
          d,
        });
        assertExact(run.o, expected, shape.name);

        // 恒真化の門: O が定数なら「一致」は何も検証していない
        assert(new Set([...run.o]).size > 1, `${shape.name}: O が定数`);
        // qP は 0 と 127 の両端に届いている（丸めで消える要素と行の最大が同居する形）
        assert([...qp].includes(P_ABS_MAX), `${shape.name}: qP に 127 が無い`);
        assert([...qp].includes(0), `${shape.name}: qP に 0 が無い`);
        // head ごとに統計が違うこと（バッチ base の検出器が成立する前提そのもの）
        if (batch > 1) {
          const head0 = run.o.slice(0, m * d);
          const head1 = run.o.slice(m * d, 2 * m * d);
          assert(
            [...head0].some((value, index) => value !== head1[index]),
            `${shape.name}: head 0 と head 1 の O が同一（base の取り違えが値に出ない）`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (b) qP の生成（exp の実装差 — atol=0 は原理的に立たない）
// ---------------------------------------------------------------------------

Deno.test({
  name: "attention_pv i8a8: qP は TS 参照と必ず ±1 段以内で一致する（exp の実装差・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    let total = 0;
    let mismatched = 0;
    try {
      for (const shape of PV_SHAPES) {
        const { batch, m, n } = shape;
        const prepared = preparePv(shape);
        const observed = await runPvI8a8(
          gpu,
          { batch, m, n, d: n },
          { s: prepared.s, stats: prepared.stats, v: identityV(batch, n) },
          "dp4a",
          `${shape.name}（qP 観測）`,
        );
        const qp = observeQuantizedP(observed, shape, prepared.rowInv);
        const rowMax = new Float32Array(batch * m);
        for (let row = 0; row < batch * m; row += 1) rowMax[row] = prepared.stats[row * 2];
        const expected = referenceAttentionPvQuant(prepared.s, rowMax, batch * m, n);
        for (let i = 0; i < qp.length; i += 1) {
          const diff = qp[i] - expected[i];
          // MUST: ±2 段以上は 1 件でも実装バグ（exp の数 ULP 差では届かない）
          assert(
            Math.abs(diff) <= 1,
            `${shape.name}: qP[${i}] が GPU ${qp[i]} / 参照 ${expected[i]}（±2 段以上）`,
          );
          if (diff !== 0) mismatched += 1;
          total += 1;
        }
      }
    } finally {
      gpu.destroy();
    }
    // 実測の記録（設計 §4.2 の「不一致率そのものが記録になる」）。恒真化しないよう上限も置く
    const rate = mismatched / total;
    console.log(
      `qP の GPU/TS 不一致率: ${mismatched} / ${total} = ${(rate * 100).toFixed(4)}%（全て ±1 段）`,
    );
    assert(rate < 0.02, `qP の不一致率 ${rate} が 2% を超えた（exp の実装差にしては大きい）`);
  },
});

// ---------------------------------------------------------------------------
// (c) dp4a とエミュ
// ---------------------------------------------------------------------------

Deno.test({
  name:
    "attention_pv i8a8: dot4I8Packed 版とエミュ版が atol=0 で一致する（拡張の有無は速度だけ・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of PV_SHAPES) {
        const prepared = preparePv(shape);
        const dp4a = await runPvI8a8(gpu, shape, prepared, "dp4a", shape.name);
        const emu = await runPvI8a8(gpu, shape, prepared, "emu", shape.name);
        assertExact(emu.o, dp4a.o, `${shape.name}: dp4a vs エミュ`);
        assert(new Set([...dp4a.o]).size > 1, `${shape.name}: O が定数`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// Session 経路（attentionCompute: "a8"）— 段ごとの適格判定と混成
// ---------------------------------------------------------------------------

const QUERY = (i: number): number => (((i * 3) % 29) - 14) * 0.3717 + 0.0419;
const KEY = (i: number): number => (((i * 3) % 41) - 20) * 0.2917 - 0.0173;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/** 半スケール（torch math decomp の `√scale_factor`）— ADR 0023 の契約どおりの値。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

type RunResult = {
  readonly output: Float32Array<ArrayBuffer>;
  /** 走ったパイプラインキー（GPU 時間計測が無い環境では空）。 */
  readonly entries: readonly { readonly key: string; readonly dispatchCount: number }[];
};

const runAttention = async (
  gpu: GpuContext,
  shape: { readonly b: number; readonly h: number; readonly m: number; readonly n: number },
  d: number,
  options: SessionOptions,
): Promise<RunResult> => {
  const { b, h, m, n } = shape;
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, h, n, d], KEY);
  const v = fill([b, h, n, d], VALUE);
  const graph = singleOpGraph("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
    attrs: { scale: halfScale(d) },
  });
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
  try {
    const outputs = await session.run({ x0: q, x1: k, x2: v });
    const timing = session.diagnostics().lastRunTiming;
    return {
      output: (outputs["y"] as Tensor).data as Float32Array<ArrayBuffer>,
      entries: (timing?.entries ?? []).map((entry) => ({
        key: entry.key,
        dispatchCount: entry.dispatchCount,
      })),
    };
  } finally {
    await session.dispose();
  }
};

const relativeRms = (actual: ArrayLike<number>, expected: ArrayLike<number>): number => {
  let diff = 0;
  let norm = 0;
  for (let i = 0; i < actual.length; i += 1) {
    diff += (actual[i] - expected[i]) ** 2;
    norm += expected[i] ** 2;
  }
  return Math.sqrt(diff / norm);
};

Deno.test({
  name: "attentionCompute:'a8' は ①QK と ③PV を整数内積にし、②行統計は f32 のまま走る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // MUST: **shader-f16 を要求しない既定の device** で走ること（i8a8 は feature ゲートの外）。
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const shape = { b: 2, h: 3, m: 65, n: 68 };
      const d = 20;
      const i8a8 = await runAttention(gpu, shape, d, { attentionCompute: "a8" });
      const plain = await runAttention(gpu, shape, d, {});
      assert(
        [...i8a8.output].some((value, index) => value !== plain.output[index]),
        "i8a8 の出力が f32 経路と 1 ビットも違わない（opt-in が効いていない）",
      );
      assert(new Set([...i8a8.output]).size > 1, "出力が定数（比較が恒真になっている）");
      // 経路全体が正しく結線されていることの粗い門（V の scale 軸を取り違える等の沈黙誤値は
      // 桁で外れる。数値契約そのものは上の生 dispatch が atol=0 で持つ）。
      // 実測 6.09e-2（決定的なデータなので値は動かない）に対し門は 1e-1。この形は S の値域が
      // 広く softmax が鋭いので P̃ の peak/rms が大きい = 設計 §2.2 の理論どおり誤差も大きい
      const rms = relativeRms(i8a8.output, plain.output);
      assert(rms < 0.1, `i8a8 と f32 の相対 RMS が ${rms}（結線を疑う）`);
      if (i8a8.entries.length > 0) {
        const byKey = new Map(i8a8.entries.map((entry) => [entry.key, entry.dispatchCount]));
        // ①QK / ③PV とも i8a8 変種で、f32 変種は 1 本も走らない
        assertEquals(byKey.get(attentionQkI8a8Key(attentionQkI8a8UsesVec4(shape.n), true)), 1);
        assertEquals(byKey.get(attentionPvI8a8Key(attentionPvI8a8UsesVec4(d), true)), 1);
        for (const v4 of [false, true]) {
          assertEquals(byKey.has(attentionQkKey(v4)), false, "f32 の attention_qk が残っている");
          assertEquals(byKey.has(attentionPvKey(v4)), false, "f32 の attention_pv が残っている");
        }
        // ②行統計は f32 のまま（S の格納形も f32 のまま — 設計 §4.3 の分母量子化は不採用）。
        // regcache（S 1 回読み）は dim 依存の生成なので epc がキーに載る（値はビット同一）
        assertEquals(
          byKey.get(attentionStatsKey("f32", "f32", attentionStatsRegCache(shape.n))),
          1,
          "②行統計は f32 のまま",
        );
        // 量子化は q / k / Vᵀ の 3 本、permute は Vᵀ の 1 本（ノード全体で 7 dispatch）
        assertEquals(quantizeRowsDispatches(byKey), 3, "quantize_rows が q / k / v の 3 本でない");
        assertEquals(byKey.get(stridedKey({ dtype: "f32" })), 1, "Vᵀ の permute が 1 本でない");
        assertEquals(
          [...byKey.values()].reduce((sum, count) => sum + count, 0),
          7,
          "1 ノード = 7 dispatch",
        );
      }

      // 整数内積変種のノブは attention の**両段**に効く（linear と同じ 1 つのノブ）
      const emu = await runAttention(gpu, shape, d, {
        attentionCompute: "a8",
        [I8A8_DOT]: "emu",
      });
      assertExact(emu.output, i8a8.output, "dp4a vs エミュ（Session 経路）");
      if (emu.entries.length > 0) {
        const keys = new Set(emu.entries.map((entry) => entry.key));
        assertEquals(keys.has(attentionPvI8a8Key(attentionPvI8a8UsesVec4(d), false)), true);
        assertEquals(keys.has(attentionPvI8a8Key(attentionPvI8a8UsesVec4(d), true)), false);
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "attentionCompute:'a8' の適格判定は段ごとに独立で、片方だけ f32 へ縮退する混成が起こる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      // ① N % 4 != 0 → ③PV だけ f32（①QK は D=20 で i8a8 のまま）
      const mixedPv = { b: 1, h: 2, m: 17, n: 19 };
      const pvF32 = await runAttention(gpu, mixedPv, 20, { attentionCompute: "a8" });
      if (pvF32.entries.length > 0) {
        const byKey = new Map(pvF32.entries.map((entry) => [entry.key, entry.dispatchCount]));
        assertEquals(byKey.get(attentionQkI8a8Key(attentionQkI8a8UsesVec4(19), true)), 1);
        assertEquals(byKey.get(attentionPvKey(gemmUsesVec4(19, 20))), 1, "③PV は f32 へ縮退する");
        for (const v4 of [false, true]) {
          assertEquals(byKey.has(attentionPvI8a8Key(v4, true)), false);
        }
        // 量子化は q / k の 2 本だけ・Vᵀ の permute は走らない
        assertEquals(quantizeRowsDispatches(byKey), 2, "quantize_rows が q / k の 2 本でない");
        assertEquals(byKey.has(stridedKey({ dtype: "f32" })), false, "Vᵀ の permute が走っている");
      }

      // ② D % 4 != 0 → ①QK だけ f32（③PV は N=20 で i8a8 のまま）— 逆向きの混成
      const mixedQk = { b: 1, h: 2, m: 17, n: 20 };
      const qkF32 = await runAttention(gpu, mixedQk, 13, { attentionCompute: "a8" });
      if (qkF32.entries.length > 0) {
        const byKey = new Map(qkF32.entries.map((entry) => [entry.key, entry.dispatchCount]));
        assertEquals(byKey.get(attentionQkKey(gemmUsesVec4(13, 20))), 1, "①QK は f32 へ縮退する");
        assertEquals(byKey.get(attentionPvI8a8Key(attentionPvI8a8UsesVec4(13), true)), 1);
        for (const v4 of [false, true]) {
          assertEquals(byKey.has(attentionQkI8a8Key(v4, true)), false);
        }
        // 量子化は Vᵀ の 1 本だけ（q / k は f32 のまま読まれる）
        assertEquals(quantizeRowsDispatches(byKey), 1, "quantize_rows が Vᵀ の 1 本でない");
        assertEquals(byKey.get(stridedKey({ dtype: "f32" })), 1, "Vᵀ の permute が 1 本でない");
      }

      // ③ 両方満たさない形は f32 経路とビット同一（縮退は沈黙 — 検出器はキーと値の 2 本）
      const both = { b: 1, h: 2, m: 17, n: 19 };
      const degraded = await runAttention(gpu, both, 13, { attentionCompute: "a8" });
      const plain = await runAttention(gpu, both, 13, {});
      assertExact(degraded.output, plain.output, "両段の縮退が f32 経路と一致しない");
      if (degraded.entries.length > 0) {
        const keys = new Set(degraded.entries.map((entry) => entry.key));
        assertEquals(
          [...keys].some((key) => key.startsWith(QUANTIZE_ROWS_KEY)),
          false,
          "縮退したのに量子化が走っている",
        );
        assertEquals(keys.has(stridedKey({ dtype: "f32" })), false, "縮退したのに permute が走る");
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "attention_pv i8a8: N が i32 縮約の門（2^17）を超えたら fail loudly（黙って巻き戻さない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 門のすぐ外側（4 の倍数で 2^17 + 4）。|acc| ≤ N·127² は i32 に収まるが、門は安全側。
    const n = LINEAR_I8A8_MAX_K + 4;
    const shape = { b: 1, h: 1, m: 1, n };
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      await assertRejects(
        () => runAttention(gpu, shape, 4, { attentionCompute: "a8" }),
        ExecutionError,
        "i32 縮約の門",
      );
      // 既定の f32 経路では同じ形が普通に走る（門が i8a8 経路だけのものであること）
      const baseline = await runAttention(gpu, shape, 4, {});
      assertEquals(baseline.output.length, 4);
    } finally {
      gpu.destroy();
    }
  },
});

// 融合 attention ①QK の **i8a8 変種**（q / k とも per-token i8 の整数内積）の実行経路。
//
// 設計 = docs/research/2026-08-04-attention-a8-design.md §2.1 / §4.1、実装 =
// src/kernels/quantize-rows.ts（無改変で再利用）+ src/kernels/attention-i8a8.ts、
// opt-in = `SessionOptions.attentionCompute: "i8a8"`（既定 "f32"）。
//
// ## このファイルが固定する数値契約
//
// **S（①QK の出力）について GPU と TS 参照が atol=0 で一致する**。成立の根拠は linear の
// w8a8（tests/gpu_i8a8_test.ts）と同じ 2 点だけ — 「内積が i32 の厳密加算（順序非依存）」と
// 「浮動小数の演算が出力 1 要素あたり数個で全て round-to-nearest-even」。したがってタイル
// 分割・K 端数の 0 埋め・共有メモリの配置をどう変えても値は 1 ビットも動かないはずで、
// 動いたらそれは実装の誤り。
//
// **契約の対象は S であって O ではない**。②行統計の後段に `exp` の実装差が乗るため O に
// atol=0 は立たない（③PV の i8a8 変種でも同じで、そちらは契約を 2 段に割る — 設計 §4.2 /
// tests/gpu_attention_pv_i8a8_test.ts）。したがって S を直接読む**生 dispatch**の突合が
// 本ファイルの主検出器で、Session 経路のテストは「経路選択（キー）」と「D%4≠0 の f32 縮退」
// だけを見る（③ 側の経路選択と混成は PV 側のファイルが持つ）。
//
// ## この波で意図的に検証していないもの
//
// - grid-stride の縮退ハーネス（tests/gpu_gridstride_test.ts）への追加は**不要**。新カーネルは
//   「1 workgroup = 1 出力タイル」で全域を覆う形（上限超過は `tiledWorkgroups` の fail loudly）
//   なので、行方向 grid-stride 族ではない。同時に走る `quantize_rows` は既に載っている。
// - 分解経路とのビット同一（tests/gpu_attention_parity_test.ts）は i8a8 では原理的に成立
//   しない。同テストは f32 / f16 変種専用の門として据え置く（別キーなので構造的に混ざらない）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { openModel } from "../src/format/container.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { GEMM_TILE } from "../src/kernels/gemm.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../src/kernels/quantize-rows.ts";
import { LINEAR_I8A8_MAX_K } from "../src/kernels/linear-i8a8.ts";
import {
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../src/kernels/attention-i8a8.ts";
import {
  attentionPvKey,
  attentionQkKey,
  attentionStatsKey,
  attentionStatsRegCache,
} from "../src/kernels/attention.ts";
import { quantizeRowsTieMargin, referenceAttentionQkI8a8 } from "../src/reference/i8a8.ts";
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
 * 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。
 *
 * MUST: 刻みを scale の格子と共約にしない（`x/s` が半整数に乗ると、GPU の除算が 2.5 ULP まで
 * 許されているぶん量子化値が ±1 段揺れて atol=0 が「たまたま落ちる」ようになる）。余裕は
 * {@link quantizeRowsTieMargin} で毎回実測して門にする。
 */
const QUERY = (i: number): number => (((i * 3) % 29) - 14) * 0.3717 + 0.0419;
const KEY = (i: number): number => (((i * 3) % 41) - 20) * 0.2917 - 0.0173;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/** 半スケール（torch math decomp の `√scale_factor`）— ADR 0023 の契約どおりの値。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/**
 * atol=0 の全数比較。`===` なので ±0 は同一視するが、それ以外は 1 ビットの差も許さない。
 * 参照が NaN の位置は「実測も NaN」であることだけを見る（非有限行の量子化値は契約の外）。
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
    assert(
      actual[i] === expected[i],
      `${where}[${i}]: GPU ${actual[i]} / 参照 ${expected[i]}`,
    );
  }
};

// ---------------------------------------------------------------------------
// 形状群とデータ
// ---------------------------------------------------------------------------

type QkShape = {
  readonly name: string;
  /** B·H を畳んだバッチ軸。**2 以上が batch base の唯一の検出器**。 */
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * 形の踏み分け。
 *
 * MUST: **B·H ≥ 2 かつ head ごとに統計が違う**形を持つ（バッチ base の唯一の検出器 —
 * head ごとに大きさだけを変えると量子化後の整数が同じになり、ペイロード側の base の
 * 取り違えが値に出ない。**パターンごと**変えること）。
 * MUST: **m / n をタイル辺 64 を跨がせる**（行 scale と列 scale の取り違えの検出器）。
 * MUST: **v4 経路とスカラ経路を対で持つ**（S の書き出しが変種ごとに別の式）。
 * MUST: **D % 16 != 0** を含む（K 端数の 0 埋めが効いていることの検出器）。
 */
const QK_SHAPES: readonly QkShape[] = [
  // v4（N % 4 == 0）/ m・n とも 64 タイル 2 枚 / D=20 は k4=5 で K タイル 2 枚（端数 1 パック）
  { name: "v4 B·H6 M65 N68 D20", batch: 6, m: 65, n: 68, d: 20 },
  // スカラ（N % 4 != 0）で 1 タイルに収まる形 + head 2 枚
  { name: "スカラ B·H2 M17 N19 D20", batch: 2, m: 17, n: 19, d: 20 },
  // K タイルが 9 枚（k4=33・端数 1 パック）/ n=5 はスカラ / m=70 で行タイル 2 枚
  { name: "スカラ 長K B·H2 M70 N5 D132", batch: 2, m: 70, n: 5, d: 132 },
  // 1 タイルに収まる最小形（ガードが全て真のまま通る経路）
  { name: "v4 小 B·H1 M3 N8 D4", batch: 1, m: 3, n: 8, d: 4 },
  // 実測形（DiT の self-attention）— タイル辺にちょうど乗る形
  { name: "DiT 形 B·H4 M64 N64 D128", batch: 4, m: 64, n: 64, d: 128 },
];

/**
 * head ごとに**パターンも大きさも**変えたデータ（バッチ base の検出器の前提）。
 * 大きさだけ変えると量子化後の整数が head 間で同一になり、ペイロード base の取り違えが
 * 値に出なくなる。
 */
const headVaried = (
  base: (index: number) => number,
  batch: number,
  rows: number,
  d: number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(batch * rows * d);
  const stride = rows * d;
  for (let head = 0; head < batch; head += 1) {
    for (let i = 0; i < stride; i += 1) {
      data[head * stride + i] = base(i + head * 37) * (1 + head * 0.37);
    }
  }
  return data;
};

type PreparedQk = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly k: Float32Array<ArrayBuffer>;
  readonly scale: number;
  readonly expected: Float32Array<ArrayBuffer>;
  /** `x/s` の丸め境界からの余裕（q / k の最小値）。 */
  readonly margin: number;
};

const prepareQk = (shape: QkShape): PreparedQk => {
  const { batch, m, n, d } = shape;
  const q = headVaried(QUERY, batch, m, d);
  const k = headVaried(KEY, batch, n, d);
  const scale = halfScale(d);
  const margin = Math.min(
    quantizeRowsTieMargin(q, batch * m, d),
    quantizeRowsTieMargin(k, batch * n, d),
  );
  const expected = referenceAttentionQkI8a8({ q, k, batch, m, n, d, scale });
  return { q, k, scale, expected, margin };
};

// ---------------------------------------------------------------------------
// 生 dispatch（S を直接読む — atol=0 の主検出器）
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

/**
 * executor の ①QK i8a8 と同じ 3 dispatch（`quantize_rows` ×2 → i8a8 GEMM）を生で流し、
 * **S をそのまま読み戻す**。Session 経路では S が中間バッファなので観測できない。
 */
const runAttentionQkI8a8 = async (
  gpu: GpuContext,
  shape: QkShape,
  prepared: PreparedQk,
  dot: I8a8Dot,
): Promise<Float32Array<ArrayBuffer>> => {
  const { batch, m, n, d } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const quantizePipeline = await cache.get(QUANTIZE_ROWS_KEY, QUANTIZE_ROWS_WGSL);
    const quantize = (
      source: Float32Array<ArrayBuffer>,
      rows: number,
    ): { readonly payload: GPUBuffer; readonly scales: GPUBuffer } => {
      const params = arena.allocHostWritten(16, UNIFORM_IN);
      gpu.device.queue.writeBuffer(params, 0, quantizeRowsParams(rows, d));
      const src = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(src, 0, source);
      const payload = arena.allocStorage(Math.max(4, rows * d));
      arena.retain(payload, 0, { pinned: true });
      const scales = arena.allocStorage(Math.max(4, rows * 4));
      arena.retain(scales, 0, { pinned: true });
      const bindGroup = gpu.device.createBindGroup({
        layout: quantizePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: payload } },
          { binding: 3, resource: { buffer: scales } },
        ],
      });
      scheduler.dispatch(quantizePipeline, bindGroup, [
        gridStrideWorkgroups(rows, 1, limit),
        1,
        1,
      ], QUANTIZE_ROWS_KEY);
      return { payload, scales };
    };
    const query = quantize(prepared.q, batch * m);
    const key = quantize(prepared.k, batch * n);

    const v4 = attentionQkI8a8UsesVec4(n);
    const dp4a = dot === "dp4a";
    const pipelineKey = attentionQkI8a8Key(v4, dp4a);
    const pipeline = await cache.get(pipelineKey, attentionQkI8a8Wgsl(v4, dp4a));
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionQkI8a8Params(m, n, d, prepared.scale));
    const scores = arena.allocStorage(Math.max(4, batch * m * n * 4));
    arena.retain(scores, 0, { pinned: true });
    const bindGroup = gpu.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: query.payload } },
        { binding: 2, resource: { buffer: key.payload } },
        { binding: 3, resource: { buffer: scores } },
        { binding: ATTENTION_QK_Q_SCALE_BINDING, resource: { buffer: query.scales } },
        { binding: ATTENTION_QK_K_SCALE_BINDING, resource: { buffer: key.scales } },
      ],
    });
    scheduler.dispatch(pipeline, bindGroup, [
      tiledWorkgroups(n, GEMM_TILE, limit, shape.name),
      tiledWorkgroups(m, GEMM_TILE, limit, shape.name),
      tiledWorkgroups(batch, 1, limit, shape.name),
    ], pipelineKey);
    await scheduler.flush();
    const bytes = await readbackBytes(gpu.device, scores, batch * m * n * 4);
    return new Float32Array(bytes.buffer, 0, batch * m * n);
  } finally {
    await arena.destroy();
  }
};

// ---------------------------------------------------------------------------
// 丸め境界からの余裕（atol=0 を主張してよいデータであることの門）
// ---------------------------------------------------------------------------

Deno.test("attention_qk i8a8: 形状群の q / k は丸め境界から十分離れている（atol=0 の前提）", () => {
  // GPU の除算は 2.5 ULP まで許されるので、`x/s` が半整数の近傍にある要素は ±1 段揺れうる。
  // 余裕を**毎回実測**して門にしないと、atol=0 は「たまたま通っているだけ」になる。
  let worst = Number.POSITIVE_INFINITY;
  for (const shape of QK_SHAPES) {
    const { margin } = prepareQk(shape);
    assert(margin > TIE_MARGIN, `${shape.name}: 丸め境界からの余裕が ${margin} しかない`);
    worst = Math.min(worst, margin);
  }
  // 全形状を通した最悪余裕（データ列を変えたらこの値が動く = 門が教えてくれる）
  assert(worst > TIE_MARGIN, `最悪余裕 ${worst}`);
});

// ---------------------------------------------------------------------------
// GPU vs TS 参照（atol=0）
// ---------------------------------------------------------------------------

Deno.test({
  name:
    "attention_qk i8a8 の S が TS 参照と atol=0 で一致する（v4 / スカラ / タイル端 / K 端数 / B·H≥2・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of QK_SHAPES) {
        const prepared = prepareQk(shape);
        const actual = await runAttentionQkI8a8(gpu, shape, prepared, "dp4a");
        assertExact(actual, prepared.expected, shape.name);
        // 恒真化の門: S が定数なら「一致」は何も検証していない
        assert(new Set([...actual]).size > 1, `${shape.name}: S が定数`);
        // head ごとに統計が違うこと（バッチ base の検出器が成立する前提そのもの）
        if (shape.batch > 1) {
          const head0 = actual.slice(0, shape.m * shape.n);
          const head1 = actual.slice(shape.m * shape.n, 2 * shape.m * shape.n);
          assert(
            [...head0].some((value, index) => value !== head1[index]),
            `${shape.name}: head 0 と head 1 の S が同一（base の取り違えが値に出ない）`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "attention_qk i8a8: NaN 行は行全体・NaN 列は列全体へ伝播し、ゼロ行は厳密に 0（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const shape: QkShape = { name: "nan/zero B·H2 M8 N12 D20", batch: 2, m: 8, n: 12, d: 20 };
    const { batch, m, n, d } = shape;
    const q = headVaried(QUERY, batch, m, d);
    const k = headVaried(KEY, batch, n, d);
    // head 1 の q 行 2 に NaN・q 行 5 を全ゼロ（tiny 床）・head 0 の k 行（= 出力列）3 に NaN
    const nanRow = 2;
    const zeroRow = 5;
    const nanCol = 3;
    q[(1 * m + nanRow) * d + d - 3] = Number.NaN;
    for (let i = 0; i < d; i += 1) q[(1 * m + zeroRow) * d + i] = 0;
    k[(0 * n + nanCol) * d + 1] = Number.NaN;
    const scale = halfScale(d);
    const expected = referenceAttentionQkI8a8({ q, k, batch, m, n, d, scale });
    // 非有限行の比は `quantizeRowsTieMargin` が読み飛ばすので、余裕の門はそのまま掛かる
    const margin = Math.min(
      quantizeRowsTieMargin(q, batch * m, d),
      quantizeRowsTieMargin(k, batch * n, d),
    );
    assert(margin > TIE_MARGIN, `${shape.name}: 丸め境界からの余裕が ${margin} しかない`);
    const prepared: PreparedQk = { q, k, scale, expected, margin };

    const gpu = await acquireGpu();
    let actual: Float32Array<ArrayBuffer>;
    try {
      actual = await runAttentionQkI8a8(gpu, shape, prepared, "dp4a");
    } finally {
      gpu.destroy();
    }
    assertExact(actual, expected, shape.name);
    // NaN 行は**全列**が NaN（scale が NaN になり dequant で行全体へ乗る）
    for (let col = 0; col < n; col += 1) {
      assert(Number.isNaN(actual[(1 * m + nanRow) * n + col]), `NaN 行 col=${col}`);
    }
    // NaN 列は**全行**が NaN（列 scale の側の伝播 — 行と列で別の経路）
    for (let row = 0; row < m; row += 1) {
      assert(Number.isNaN(actual[(0 * m + row) * n + nanCol]), `NaN 列 row=${row}`);
    }
    // 隣の head は無傷（q の NaN 行が head を跨いでいない — NaN 列は head 0 の側の設計）
    for (let col = 0; col < n; col += 1) {
      if (col === nanCol) continue;
      assert(Number.isFinite(actual[(0 * m + nanRow) * n + col]), `隣接 head col=${col}`);
    }
    // 全ゼロ行は q = 0・s = tiny なので S も厳密に 0（±0 は同一視）。NaN 列は head 0 の側
    // なので、head 1 のこの行は全列が有限で 0 になる
    for (let col = 0; col < n; col += 1) {
      assertEquals(actual[(1 * m + zeroRow) * n + col], 0, `ゼロ行 col=${col}`);
    }
  },
});

Deno.test({
  name:
    "attention_qk i8a8: dot4I8Packed 版とエミュ版が atol=0 で一致する（拡張の有無は速度だけ・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // 本機は packed_4x8_integer_dot_product を列挙するので**両変種とも実走できる**。
      // この 1 本が「エミュは数値同一」という主張の attention 側の機械的検出器。
      for (const shape of QK_SHAPES) {
        const prepared = prepareQk(shape);
        const dp4a = await runAttentionQkI8a8(gpu, shape, prepared, "dp4a");
        const emu = await runAttentionQkI8a8(gpu, shape, prepared, "emu");
        assertExact(emu, dp4a, `${shape.name}: dp4a vs エミュ`);
        // どちらも TS 参照と一致する（両者が同じだけずれている形を塞ぐ）
        assertExact(emu, prepared.expected, `${shape.name}: エミュ vs 参照`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// Session 経路（attentionCompute: "i8a8"）— 経路選択と縮退
// ---------------------------------------------------------------------------

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
  const graph = singleOpGraph("attention", [q.shape, k.shape, v.shape], [b, h, m, d], {
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

Deno.test({
  name: "attentionCompute:'i8a8' は ①QK を整数内積にし、②行統計は f32 のまま走る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // MUST: **shader-f16 を要求しない既定の device** で走ること（i8a8 は feature ゲートの外）。
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const shape = { b: 2, h: 3, m: 65, n: 68 };
      const d = 20;
      const i8a8 = await runAttention(gpu, shape, d, { attentionCompute: "i8a8" });
      const plain = await runAttention(gpu, shape, d, {});
      // opt-in が実際に効いている（活性量子化が O を変える）
      assert(
        [...i8a8.output].some((value, index) => value !== plain.output[index]),
        "i8a8 の出力が f32 経路と 1 ビットも違わない（opt-in が効いていない）",
      );
      assert(new Set([...i8a8.output]).size > 1, "出力が定数（比較が恒真になっている）");
      if (i8a8.entries.length > 0) {
        const byKey = new Map(i8a8.entries.map((entry) => [entry.key, entry.dispatchCount]));
        // ①QK は i8a8 変種・f32 変種は 1 本も走らない
        assertEquals(byKey.get(attentionQkI8a8Key(attentionQkI8a8UsesVec4(shape.n), true)), 1);
        assertEquals(byKey.has(attentionQkKey(true)), false, "f32 の attention_qk が残っている");
        assertEquals(byKey.has(attentionQkKey(false)), false, "f32 の attention_qk が残っている");
        // ②行統計は f32 のまま（S の格納形も f32 のまま — 設計 §4.3 の分母量子化は不採用）。
        // regcache（S 1 回読み）は dim 依存の生成なので epc がキーに載る（値はビット同一）
        assertEquals(
          byKey.get(attentionStatsKey("f32", "f32", attentionStatsRegCache(shape.n))),
          1,
          "②行統計は f32 のまま",
        );
        // ③PV も i8a8 へ移っている（N=68 % 4 == 0 で適格 — 経路と本数の検査は
        // tests/gpu_attention_pv_i8a8_test.ts が持つ）
        for (const v4 of [false, true]) {
          assertEquals(byKey.has(attentionPvKey(v4)), false, "f32 の attention_pv が残っている");
        }
        // 量子化は q / k / Vᵀ の 3 本（linear と同じキーを共有する）
        assertEquals(byKey.get(QUANTIZE_ROWS_KEY), 3, "quantize_rows が q / k / v の 3 本でない");
      }

      // 整数内積変種のノブが attention 側にも結線されている（linear と同じ 1 つのノブ）。
      // **どちらでも O は 1 ビットも変わらない**（S が atol=0 で同じなら後段も同じ）。
      const emu = await runAttention(gpu, shape, d, {
        attentionCompute: "i8a8",
        [I8A8_DOT]: "emu",
      });
      assertExact(emu.output, i8a8.output, "dp4a vs エミュ（Session 経路）");
      if (emu.entries.length > 0) {
        const keys = new Set(emu.entries.map((entry) => entry.key));
        const v4 = attentionQkI8a8UsesVec4(shape.n);
        assertEquals(keys.has(attentionQkI8a8Key(v4, false)), true, "エミュ変種のキー");
        assertEquals(keys.has(attentionQkI8a8Key(v4, true)), false, "dp4a 変種が残っている");
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "attentionCompute:'i8a8' は D % 4 != 0 で f32 経路へ縮退する（出力はビット同一・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      // D = 13（4 の倍数でない）は ①QK の語境界条件を満たさないので f32 経路のまま。
      // この形は N = 19 も 4 の倍数でないので ③PV も同時に縮退する（段ごとの独立な判定と
      // 混成は tests/gpu_attention_pv_i8a8_test.ts が持つ）。
      // MUST: 縮退は**沈黙**（linear の k%4 と同じ流儀）なので、検出器はキーと「f32 と
      // ビット同一」の 2 本しかない。
      const shape = { b: 1, h: 2, m: 17, n: 19 };
      const degraded = await runAttention(gpu, shape, 13, { attentionCompute: "i8a8" });
      const plain = await runAttention(gpu, shape, 13, {});
      assertExact(degraded.output, plain.output, "D%4!=0 の縮退が f32 経路と一致しない");
      if (degraded.entries.length > 0) {
        const keys = new Set(degraded.entries.map((entry) => entry.key));
        assertEquals(keys.has(attentionQkKey(false)), true, "f32 の attention_qk が走っていない");
        assertEquals(keys.has(QUANTIZE_ROWS_KEY), false, "縮退したのに量子化が走っている");
        for (const dp4a of [false, true]) {
          assertEquals(
            keys.has(attentionQkI8a8Key(false, dp4a)),
            false,
            "縮退したのに i8a8 が走っている",
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "attention i8a8: D が i32 縮約の門（2^17）を超えたら fail loudly（黙って巻き戻さない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 門のすぐ外側（4 の倍数で 2^17 + 4）。|acc| ≤ D·127² は i32 に収まるが、門は安全側。
    const d = LINEAR_I8A8_MAX_K + 4;
    const shape = { b: 1, h: 1, m: 1, n: 4 };
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => runAttention(gpu, shape, d, { attentionCompute: "i8a8" }),
        ExecutionError,
        "i32 縮約の門",
      );
      // 既定の f32 経路では同じ形が普通に走る（門が i8a8 経路だけのものであること）
      const baseline = await runAttention(gpu, shape, d, {});
      assertEquals(baseline.output.length, d);
    } finally {
      gpu.destroy();
    }
  },
});

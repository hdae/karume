// w8a8 / w4a8 linear（活性 per-token i8 × 重み i8 per-channel / i4 group の整数内積）の
// 実行経路。
//
// 設計 = docs/research/2026-08-03-dp4a-w8a8-design.md（w4a8 は perf-ledger Q-8）、実装 =
// src/kernels/quantize-rows.ts + src/kernels/linear-i8a8.ts、opt-in =
// `SessionOptions.linearCompute: "a8"`（既定 "f32"・重みの格納形は別軸）。
//
// ## このファイルが固定する数値契約
//
// **GPU と TS 参照が atol=0 で一致する**。他のカーネル（f32 の縮約順序が実装で変わる）とは
// 質の違う契約で、成立の根拠は「内積が i32 の厳密加算（順序非依存）」+「浮動小数の演算が
// 出力 1 要素あたり 3 つだけ（f32(acc) / xs·wscale / fma）」の 2 点だけ。したがって、
// タイル分割・K 端数の 0 埋め・共有メモリの配置をどう変えても値は 1 ビットも動かないはずで、
// 動いたらそれは実装の誤り — tolerance で吸収する余地が構造的に無い。
// w4a8 も同じ性質だが根拠がずれる（group の中だけが i32 厳密で、浮動小数の演算は
// `k/g + 1` 回）— 節ごとの docstring を参照。
//
// MUST: 比較は `compareTensors` ではなく {@link assertExact}（`===` の全数比較）で行う。
// allclose は非有限を全て不合格にするので、NaN 行の伝播（設計の一部）を検証できない。
//
// NOTE: 参照オラクル（src/reference/i8a8.ts）そのものの既知値門は tests/i8a8_reference_test.ts
// が **CPU 単独**で持つ。ここの突合は「参照と GPU が同じ向きに壊れた」形を通してしまうので、
// 参照の性質はアダプタ無しでも走る側に置く（そちらが落ちれば突合の土台が崩れている）。
//
// ## 非有限値の扱い（契約の外側）
//
// NaN 行は `xs[row] = NaN` 経由で**行全体が NaN**になる（f32 経路と同じ粒度）。Inf 行は
// `xs[row] = Inf` になり、出力は ±Inf / NaN のどれか — **符号は f32 経路と一致しない**
// （docs/limitations.md に記載）。量子化値 `xq` そのものは、行 scale が非有限のとき
// `vec4<i32>(NaN)` が不定値になるので**契約の外**（突合は scale と最終出力だけ）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import { openModel } from "../src/format/container.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  type QuantizeRowsGeometry,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../src/kernels/quantize-rows.ts";
import {
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8UsesVec4,
} from "../src/kernels/linear-i8a8.ts";
import {
  quantizeRowsReference,
  quantizeRowsTieMargin,
  referenceLinearI8a8,
  referenceLinearW4a8,
  roundTiesToEven,
} from "../src/reference/i8a8.ts";
import {
  createSession,
  I8A8_DOT,
  type I8a8Dot,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { buildSafetensors, f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { fill, type FilledTensor } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * 決定的なデータ列（乱数は使わない — 失敗が再現しないため）。
 *
 * MUST: 活性の刻みを **scale の格子と共約にしない**。`0.75` 刻みのような素直な列は
 * `x/s` がちょうど半整数（例: 127·3/6 = 63.5）に乗り、GPU の除算が 2.5 ULP まで許されている
 * ぶんだけ量子化値が ±1 段揺れて atol=0 が「たまたま落ちる」ようになる。境界からの余裕は
 * {@link quantizeRowsTieMargin} で毎回実測して門にする（この列を変えたら門が教えてくれる）。
 */
const SIGNED = (i: number): number => ((i % 29) - 14) * 0.3717 + 0.0131;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

/** atol=0 を主張してよい丸め境界からの余裕（2.5 ULP ≈ 1.9e-5 に対し 50 倍以上）。 */
const TIE_MARGIN = 1e-3;

/**
 * atol=0 の全数比較。`===` なので ±0 は同一視するが、それ以外は 1 ビットの差も許さない。
 * 参照が NaN の位置は「実測も NaN」であることだけを見る（値の同一性は契約の外）。
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
// 丸めの同点規則（GPU 非依存の側）
// ---------------------------------------------------------------------------

Deno.test("roundTiesToEven は偶数丸め（Math.round の half-up と割れる点を固定する）", () => {
  const cases: readonly (readonly [number, number])[] = [
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [3.5, 4],
    [-0.5, 0],
    [-1.5, -2],
    [-2.5, -2],
    [0.49999997, 0],
    [1.4999999, 1],
    [126.5, 126],
    [127.5, 128],
  ];
  for (const [value, expected] of cases) {
    assertEquals(roundTiesToEven(value), expected, `round(${value})`);
  }
  // MUST: Math.round では代用できない（同点で +∞ 方向へ倒れる）
  assertEquals(Math.round(0.5), 1);
  assertEquals(Math.round(2.5), 3);
  // 非有限はそのまま通す（clamp が受け取って ±127 に倒す）
  assertEquals(roundTiesToEven(Number.POSITIVE_INFINITY), Number.POSITIVE_INFINITY);
  assertEquals(Number.isNaN(roundTiesToEven(Number.NaN)), true);
});

// ---------------------------------------------------------------------------
// quantize_rows 単体（生 dispatch）
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

type QuantizeResult = {
  /** 4 詰めから展開した i8 値（`[rows, dim]` の平坦）。 */
  readonly q: Int8Array<ArrayBuffer>;
  readonly scale: Float32Array<ArrayBuffer>;
};

/** `quantize_rows` を 1 dispatch だけ流す（`groups` を絞れば縮退経路も踏める）。 */
const runQuantizeRows = async (
  gpu: GpuContext,
  x: Float32Array<ArrayBuffer>,
  rows: number,
  dim: number,
  groups?: number,
  // 既定は従来形（1 行 = 1 workgroup）。小 D 変種の門は幾何を明示して呼ぶ。
  geometry: QuantizeRowsGeometry = { lanesPerRow: 256, rowsPerGroup: 1 },
): Promise<QuantizeResult> => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const key = quantizeRowsKey(geometry);
  try {
    const { pipeline, layout } = await cache.get(key, quantizeRowsWgsl(geometry));
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, quantizeRowsParams(rows, dim));
    const src = arena.allocHostWritten(Math.max(4, x.byteLength), STORAGE_IN);
    gpu.device.queue.writeBuffer(src, 0, x);
    const xq = arena.allocStorage(Math.max(4, rows * dim));
    arena.retain(xq, 0, { pinned: true });
    const xs = arena.allocStorage(Math.max(4, rows * 4));
    arena.retain(xs, 0, { pinned: true });
    const bindGroup = gpu.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: xq } },
        { binding: 3, resource: { buffer: xs } },
      ],
    });
    scheduler.dispatch(pipeline, bindGroup, [
      groups ??
        gridStrideWorkgroups(
          rows,
          geometry.rowsPerGroup,
          gpu.limits.maxComputeWorkgroupsPerDimension,
        ),
      1,
      1,
    ], key);
    await scheduler.flush();
    const qBytes = await readbackBytes(gpu.device, xq, rows * dim);
    const scaleBytes = await readbackBytes(gpu.device, xs, rows * 4);
    return {
      q: new Int8Array(qBytes.buffer, 0, rows * dim),
      scale: new Float32Array(scaleBytes.buffer, 0, rows),
    };
  } finally {
    await arena.destroy();
  }
};

Deno.test({
  name: "quantize_rows: WGSL の round は偶数丸め（同点 3 点をカーネル内で実測固定・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // amax = 127 にすると s = f32(127 · (1/127)) = 1.0 ちょうどになり、x/s が入力そのものに
    // なる（除数 1.0 の除算は 2.5 ULP の余地が無く**厳密**なので、同点規則だけを切り出せる）。
    // 同点（±0.5 / ±1.5 / 2.5 / ±3.5）の丸め先がそのまま量子化値として観測できる。
    const values = [127, 0.5, 1.5, -0.5, -1.5, 2.5, 3.5, -3.5];
    const x = Float32Array.from(values);
    const gpu = await acquireGpu();
    try {
      const actual = await runQuantizeRows(gpu, x, 1, values.length);
      assertEquals(actual.scale[0], 1, "s は 1.0 ちょうど（同点が格子の境界に乗る形）");
      // 偶数丸め: 0.5→0 / 1.5→2 / -0.5→-0 / -1.5→-2 / 2.5→2 / 3.5→4 / -3.5→-4
      assertEquals([...actual.q], [127, 0, 2, 0, -2, 2, 4, -4]);
      // TS 参照（roundTiesToEven）と同じ列であることも同時に固定する
      assertEquals([...actual.q], [...quantizeRowsReference(x, 1, values.length).q]);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "quantize_rows: 小 D 変種は従来形と q / scale ともビット同一（幾何 4 通り × 縮退経路・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 幾何は dim から決まる: 128 → 8 行 × 32 レーン / 64 → 16 × 16 / 20 → 32 × 8 / 8 → 128 × 2。
    // 行ごとに大きさを変える（全行同じ scale だと「別の行の amax を使う」誤りが値に出ない）。
    // 行数は rowsPerGroup の倍数から外し、末尾の「行が無い組」の経路も踏む。
    const gpu = await acquireGpu();
    try {
      for (const dim of [128, 64, 20, 8]) {
        const geometry = quantizeRowsGeometry(dim);
        assertEquals(geometry.lanesPerRow * geometry.rowsPerGroup, 256);
        assertEquals(geometry.lanesPerRow >= dim / 4, true, `dim ${dim}: レーン幅が quad 数未満`);
        const rows = 1_003;
        const x = new Float32Array(rows * dim);
        for (let row = 0; row < rows; row += 1) {
          for (let i = 0; i < dim; i += 1) {
            x[row * dim + i] = row % 97 === 3 ? 0 : SIGNED(row * dim + i) * (1 + (row % 11) * 0.75);
          }
        }
        const baseline = await runQuantizeRows(gpu, x, rows, dim);
        // 変種同士の突合は GPU 同士（同じ除算）なので同点の余裕は要らない。TS 参照との突合だけ
        // 余裕がある形に限る（短い行は比が粗く、この列では dim 8 が境界近傍に乗る）。
        if (quantizeRowsTieMargin(x, rows, dim) > TIE_MARGIN) {
          const reference = quantizeRowsReference(x, rows, dim);
          assertEquals([...baseline.q], [...reference.q], `dim ${dim}: 従来形 q`);
        }
        // 自然な本数と、縮退（2 workgroup で grid-stride を回す）の両方。
        for (const groups of [undefined, 2]) {
          const grouped = await runQuantizeRows(gpu, x, rows, dim, groups, geometry);
          const label =
            `dim ${dim} r${geometry.rowsPerGroup}w${geometry.lanesPerRow} groups ${groups}`;
          assertEquals([...grouped.q], [...baseline.q], `${label}: q`);
          assertEquals(
            Array.from(new Uint32Array(grouped.scale.buffer)),
            Array.from(new Uint32Array(baseline.scale.buffer)),
            `${label}: scale のビット列`,
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
    "quantize_rows が TS 参照と一致する（絶対値最大の厳密復元 / 全ゼロ行 / 行別 scale / 端数・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // dim は 4 の倍数だが **256（workgroup 幅）の倍数でも 16 の倍数でもない**形を選ぶ
    // （1 スレッドが複数 quad を担当する経路と、担当が 0 個のスレッドが出る経路の両方を踏む）。
    const rows = 7;
    const dim = 20;
    const x = new Float32Array(rows * dim);
    for (let row = 0; row < rows; row += 1) {
      for (let i = 0; i < dim; i += 1) {
        // 行ごとに大きさを変える（行別 scale の独立性 — 全行同じだと行の取り違えが値に出ない）
        x[row * dim + i] = row === 3 ? 0 : SIGNED(row * dim + i) * (1 + row * 3);
      }
    }
    // atol=0 を主張してよいデータであることの門（WGSL の除算は 2.5 ULP まで許される）
    assert(
      quantizeRowsTieMargin(x, rows, dim) > TIE_MARGIN,
      `丸め境界からの余裕が ${quantizeRowsTieMargin(x, rows, dim)} しかない`,
    );
    const gpu = await acquireGpu();
    let actual: QuantizeResult;
    try {
      actual = await runQuantizeRows(gpu, x, rows, dim);
    } finally {
      gpu.destroy();
    }
    const expected = quantizeRowsReference(x, rows, dim);
    assertExact(actual.scale, expected.scale, "quantize_rows scale");
    assertExact(actual.q, expected.q, "quantize_rows q");
    // 行別 scale の独立性（同じ値ばかりなら以降の検査が恒真になる）
    assertEquals(new Set([...actual.scale]).size >= rows - 1, true, "行ごとに違う scale");
    // 全ゼロ行は s = f32 tiny・q = 0（0 · tiny = 0 で厳密に復元される）
    assertEquals(actual.scale[3], 1.1754943508222875e-38, "全ゼロ行の scale は f32 tiny");
    for (let i = 0; i < dim; i += 1) assertEquals(actual.q[3 * dim + i], 0, `全ゼロ行の q[${i}]`);
    // 絶対値最大の要素は ±127 へちょうど乗る（−128 を使わない規約の直接の帰結）
    for (let row = 0; row < rows; row += 1) {
      if (row === 3) continue;
      let amax = 0;
      for (let i = 0; i < dim; i += 1) amax = Math.max(amax, Math.abs(x[row * dim + i]));
      let sawExtreme = false;
      for (let i = 0; i < dim; i += 1) {
        if (Math.abs(x[row * dim + i]) === amax) {
          assertEquals(Math.abs(actual.q[row * dim + i]), 127, `行 ${row} の絶対値最大`);
          sawExtreme = true;
        }
      }
      assert(sawExtreme, `行 ${row} に絶対値最大の要素が無い`);
    }
  },
});

Deno.test({
  name: "quantize_rows は NaN / Inf を行 scale へ伝播する（素の max では消える・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 行 0 = 通常 / 行 1 = 末尾に NaN / 行 2 = 中央に +Inf / 行 3 = 中央に −Inf
    const rows = 4;
    const dim = 8;
    const x = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i += 1) x[i] = SIGNED(i);
    x[1 * dim + dim - 1] = Number.NaN;
    x[2 * dim + 3] = Number.POSITIVE_INFINITY;
    x[3 * dim + 3] = Number.NEGATIVE_INFINITY;
    const gpu = await acquireGpu();
    let actual: QuantizeResult;
    try {
      actual = await runQuantizeRows(gpu, x, rows, dim);
    } finally {
      gpu.destroy();
    }
    const expected = quantizeRowsReference(x, rows, dim);
    // 非有限行も含めた全数突合（assertExact は参照が NaN の位置だけ「実測も NaN」で見る）
    assertExact(actual.scale, expected.scale, "quantize_rows scale");
    assertEquals(Number.isFinite(actual.scale[0]), true, "通常行の scale は有限");
    // MUST: NaN は行 scale へ（素の max だと NaN が飲まれて有限の scale になる）
    assertEquals(Number.isNaN(actual.scale[1]), true, "NaN 行の scale");
    assertEquals(actual.scale[2], Number.POSITIVE_INFINITY, "+Inf 行の scale");
    // |−Inf| = +Inf なので絶対値最大は +Inf（scale に符号は残らない）
    assertEquals(actual.scale[3], Number.POSITIVE_INFINITY, "−Inf 行の scale");
    // 通常行の量子化値は参照と一致（非有限行の q は契約の外 — ファイル冒頭）
    assertExact(
      actual.q.slice(0, dim),
      expected.q.slice(0, dim),
      "quantize_rows q（通常行）",
    );
  },
});

// ---------------------------------------------------------------------------
// Session 経路（linearCompute: "a8"）
// ---------------------------------------------------------------------------

type LinearCase = {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** 既定は SIGNED。NaN / ゼロ行のケースだけ差し替える。 */
  readonly x?: (index: number, m: number, k: number) => number;
};

/** `linear(x, w, b)` 1 本のグラフ（w は i8 + per-channel scale）。 */
const i8LinearModel = (
  m: number,
  n: number,
  k: number,
  quantized: ReturnType<typeof quantizeI8>,
  bias: FilledTensor,
): ArrayBuffer => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  // MUST: I8（1 バイト要素）はファイル末尾（後続 F32 の絶対 offset が 4 の倍数から外れる）
  const tensors: TensorSpec[] = [
    { name: "m.b", dtype: "F32", shape: [n], data: new Uint8Array(bias.data.buffer.slice(0)) },
    { name: "m.s", dtype: "F32", shape: [n, 1], data: f32Bytes(quantized.scale) },
    { name: "m.w", dtype: "I8", shape: [n, k], data: quantized.bytes },
  ];
  return buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });
};

type PreparedLinear = {
  readonly model: ArrayBuffer;
  readonly x: Tensor;
  readonly expected: Float32Array<ArrayBuffer>;
};

const prepareLinear = (testCase: LinearCase): PreparedLinear => {
  const { m, n, k } = testCase;
  // 行（= 出力チャネル）ごとに大きさが違う重み。全行同じ scale だと軸の取り違えが値に出ない。
  const weight = fill([n, k], (i) => POSITIVE(i) * (1 + (Math.floor(i / k) % 7) * 0.25));
  const quantized = quantizeI8(weight.data, [n, k], 0);
  const bias = fill([n], SIGNED);
  const x = fill([m, k], (i) => (testCase.x ?? ((index) => SIGNED(index)))(i, m, k));
  // atol=0 を主張してよいデータであることの門（ファイル冒頭の {@link SIGNED} の MUST）
  const margin = quantizeRowsTieMargin(x.data, m, k);
  if (!(margin > TIE_MARGIN)) {
    throw new Error(`${testCase.name}: 丸め境界からの余裕が ${margin} しかない`);
  }
  const expected = referenceLinearI8a8({
    x: x.data,
    weight: new Int8Array(quantized.bytes.buffer, quantized.bytes.byteOffset, n * k),
    weightScale: quantized.scale,
    bias: bias.data,
    m,
    n,
    k,
  });
  return { model: i8LinearModel(m, n, k, quantized, bias), x, expected };
};

const runLinear = async (
  gpu: GpuContext,
  prepared: PreparedLinear,
  options: SessionOptions,
): Promise<{ readonly y: Tensor; readonly pipelineCount: number; readonly keys: string[] }> => {
  const session = await createSession(gpu, openModel(prepared.model), options);
  try {
    const outputs = await session.run({ x: prepared.x });
    const diagnostics = session.diagnostics();
    return {
      y: outputs["y"],
      pipelineCount: diagnostics.pipelineCount,
      keys: (diagnostics.lastRunTiming?.entries ?? []).map((entry) => entry.key).sort(),
    };
  } finally {
    await session.dispose();
  }
};

/**
 * 形の踏み分け。
 *
 * MUST: **v4 経路とスカラ経路を対で持つ**（書き出しのガードが変種ごとに別の式）。
 * MUST: **m / n をタイル辺 64 を跨がせる**（1 タイル未満に潰れるとガードが一度も偽にならない）。
 * MUST: **k % 16 != 0** を含む（K 端数の 0 埋めが効いていることの検出器。ただし ADR 0022 の
 * 検出限界①のとおり、A/B 片側だけの 0 埋め退行は相殺で見えない）。
 */
const LINEAR_CASES: readonly LinearCase[] = [
  // v4（n % 4 == 0）/ m = 65 で行タイル 2 枚 / n = 68 で列タイル 2 枚（最終タイルの
  // 有効 quad は 16 中 1）/ k = 20 は k4 = 5 で K タイル 2 枚（2 枚目は 4 パック中 1 パック）
  { name: "i8a8 v4 [65,20] × W[68,20]", m: 65, n: 68, k: 20 },
  // スカラ（n % 4 != 0）
  { name: "i8a8 スカラ [17,20] × W[19,20]", m: 17, n: 19, k: 20 },
  // K タイルが 9 枚（k4 = 33）で最終タイルが 1 パック / n = 5 はスカラ / m = 70 で行タイル 2 枚
  { name: "i8a8 スカラ 長 K [70,132] × W[5,132]", m: 70, n: 5, k: 132 },
  // 1 タイルに収まる小さい形（ガードが全て真のまま通る経路）
  { name: "i8a8 v4 小 [3,4] × W[8,4]", m: 3, n: 8, k: 4 },
];

Deno.test({
  name:
    "linearCompute:'a8' の linear が TS 参照と atol=0 で一致する（v4 / スカラ / タイル端 / K 端数・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const testCase of LINEAR_CASES) {
        const prepared = prepareLinear(testCase);
        const actual = await runLinear(gpu, prepared, { linearCompute: "a8" });
        assertEquals(actual.y.shape, [testCase.m, testCase.n], testCase.name);
        assertExact(actual.y.data, prepared.expected, testCase.name);
        // 恒真化の門: 出力が定数なら「一致」は何も検証していない
        assert(new Set([...actual.y.data]).size > 1, `${testCase.name}: 出力が定数`);
        // 実際に i8a8 経路を通っている（quantize_rows + i8a8 GEMM の 2 本）
        assertEquals(actual.pipelineCount, 2, `${testCase.name}: パイプライン本数`);
        if (actual.keys.length > 0) {
          assertEquals(
            actual.keys,
            [
              linearI8a8Key(linearI8a8UsesVec4(testCase.n), true),
              quantizeRowsKey(quantizeRowsGeometry(testCase.k)),
            ].sort(),
            `${testCase.name}: 走ったパイプラインキー`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "i8a8: NaN 行は行全体へ伝播し、ゼロ行と負値は厳密に一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 行 2 に NaN を 1 つ・行 5 を全ゼロ・残りは正負混在
    const nanRow = 2;
    const zeroRow = 5;
    const prepared = prepareLinear({
      name: "nan/zero",
      m: 8,
      n: 12,
      k: 20,
      x: (index, _m, k) => {
        const row = Math.floor(index / k);
        if (row === zeroRow) return 0;
        if (row === nanRow && index % k === k - 3) return Number.NaN;
        return SIGNED(index);
      },
    });
    const gpu = await acquireGpu();
    let y: Tensor;
    try {
      y = (await runLinear(gpu, prepared, { linearCompute: "a8" })).y;
    } finally {
      gpu.destroy();
    }
    assertExact(y.data, prepared.expected, "NaN / ゼロ行");
    // NaN 行は**全列**が NaN（f32 経路と同じ粒度 — 縮約が行全体に掛かるため）
    for (let col = 0; col < 12; col += 1) {
      assert(Number.isNaN(y.data[nanRow * 12 + col]), `NaN 行 col=${col}`);
    }
    // 隣の行は無傷（伝播が行を跨いでいない）
    for (let col = 0; col < 12; col += 1) {
      assert(Number.isFinite(y.data[(nanRow + 1) * 12 + col]), `隣接行 col=${col}`);
    }
    // ゼロ行は bias そのもの（q = 0・s = tiny なので f32(0)·(tiny·ws) + bias = bias）
    for (let col = 0; col < 12; col += 1) {
      assertEquals(
        y.data[zeroRow * 12 + col],
        prepared.expected[zeroRow * 12 + col],
        `ゼロ行 col=${col}`,
      );
    }
  },
});

Deno.test({
  name: "i8a8: dot4I8Packed 版とエミュ版が atol=0 で一致する（拡張の有無は速度だけ・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      // 本機は packed_4x8_integer_dot_product を列挙するので**両変種とも実走できる**。
      // この 1 本が「エミュは数値同一」という主張の linear 側の機械的検出器（設計 §4.4。
      // 融合 attention の QK 側は tests/gpu_attention_i8a8_test.ts が同じ形で持つ）。
      const dots: readonly I8a8Dot[] = ["dp4a", "emu"];
      for (const testCase of LINEAR_CASES) {
        const prepared = prepareLinear(testCase);
        const results: Record<string, Tensor> = {};
        for (const dot of dots) {
          const options: SessionOptions = { linearCompute: "a8", [I8A8_DOT]: dot };
          const actual = await runLinear(gpu, prepared, options);
          results[dot] = actual.y;
          if (actual.keys.length > 0) {
            assertEquals(
              actual.keys.includes(linearI8a8Key(linearI8a8UsesVec4(testCase.n), dot === "dp4a")),
              true,
              `${testCase.name}: ${dot} のキー`,
            );
          }
        }
        assertExact(results["emu"].data, results["dp4a"].data, `${testCase.name}: dp4a vs エミュ`);
        // どちらも TS 参照と一致する（両者が同じだけずれている形を塞ぐ）
        assertExact(results["emu"].data, prepared.expected, `${testCase.name}: エミュ vs 参照`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "i8a8 は opt-in × i8 常駐 × k%4==0 のときだけ効く（既定は従来経路のまま・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      // ① 既定（linearCompute 省略）は i8 資産でも従来の f32 経路（linear 1 本だけ）
      const prepared = prepareLinear({ name: "default", m: 17, n: 19, k: 20 });
      const baseline = await runLinear(gpu, prepared, {});
      assertEquals(baseline.pipelineCount, 1, "既定はパイプライン 1 本（quantize_rows が出ない）");
      if (baseline.keys.length > 0) {
        // 見たいのは「i8a8 ではなく f32 linear（wi8 格納）が 1 本だけ走った」こと。キーの
        // 綴りそのものは tests/gemm_geometry_test.ts が固定するので、ここは同じ形状
        //（m=17 → 小 M のタイル幾何バケット / n=19 でスカラ変種）から引く。
        assertEquals(baseline.keys, [linearKey("i8", false, "f32", 17)], "既定のキー");
      }
      // 既定の値は w8（重みだけ量子化）なので、w8a8 の参照とは**一致しない**
      // （一致してしまうなら活性量子化が効いていない）
      const i8a8 = await runLinear(gpu, prepared, { linearCompute: "a8" });
      assertEquals(
        [...baseline.y.data].some((value, index) => value !== i8a8.y.data[index]),
        true,
        "活性量子化が出力を変えていない（opt-in が効いていない）",
      );

      // ② k % 4 != 0 は opt-in でも従来経路へ落ちる（i8 ペイロードの語境界条件）
      const odd = prepareLinear({ name: "k=7", m: 5, n: 6, k: 7 });
      const oddRun = await runLinear(gpu, odd, { linearCompute: "a8" });
      assertEquals(oddRun.pipelineCount, 1, "k % 4 != 0 は f32 経路");
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * K=0（in_features 0）の退化 shape。契約（src/ops.ts の linear）では有効な形だが、`k % 4 == 0`
 * を満たすので選択述語が `k > 0` を見ていないと i8a8 経路へ入り、① `quantize_rows` の
 * `dim >= 1` と衝突して **i8a8 固有の CodegenError** になる（量子化する活性がそもそも存在せず、
 * i8a8 の門として意味を持たない例外）。
 *
 * 固定するのは「**opt-in が K=0 の経路を変えない**」こと — 既定 session と**同じ**失敗で
 * 落ちる。K=0 自体は既定の f32 経路でも 0 バイト束縛の最小サイズ違反で落ちる（linearCompute
 * とは無関係の別要因）ので、ここは 2 経路の一致だけを見る。
 */
Deno.test({
  name:
    "i8a8: K=0 の i8 常駐 linear は opt-in でも既定と同じ経路（i8a8 固有の失敗を出さない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const m = 3;
    const n = 8;
    const bias = fill([n], SIGNED);
    // 重みは 0 要素（[n,0]）。scale は per-channel の形だけ保つ（縮約が空なので読まれない）。
    const quantized = quantizeI8(new Float32Array(0), [n, 0], 0);
    const model = i8LinearModel(m, n, 0, quantized, bias);
    const x = fill([m, 0], SIGNED);
    const runK0 = async (gpu: GpuContext, options: SessionOptions): Promise<Error> => {
      const session = await createSession(gpu, openModel(model), options);
      try {
        return await assertRejects(() => session.run({ x }), Error);
      } finally {
        await session.dispose();
      }
    };
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const baseline = await runK0(gpu, {});
      const optIn = await runK0(gpu, { linearCompute: "a8" });
      assertEquals(optIn.name, baseline.name, "K=0: opt-in で例外の種類が変わっている");
      assertEquals(optIn.message, baseline.message, "K=0: opt-in で失敗の理由が変わっている");
      // 恒真化の門: 述語が K=0 を拾っていたときに出る ① の門とは別物であること
      assert(
        !optIn.message.includes("quantize_rows"),
        `K=0: i8a8 の ① へ入っている（${optIn.message}）`,
      );
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "i8a8: k が i32 縮約の門（2^17）を超えたら fail loudly（黙って巻き戻さない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 門のすぐ外側（4 の倍数で 2^17 + 4）。|acc| ≤ k·127² は i32 に収まるが、門は安全側。
    const k = LINEAR_I8A8_MAX_K + 4;
    const prepared = prepareLinear({ name: "overflow gate", m: 1, n: 4, k });
    const gpu = await acquireGpu();
    try {
      const session = await createSession(gpu, openModel(prepared.model), {
        linearCompute: "a8",
      });
      try {
        await assertRejects(
          () => session.run({ x: prepared.x }),
          ExecutionError,
          "i32 縮約の門",
        );
      } finally {
        await session.dispose();
      }
      // 既定の f32 経路では同じモデルが普通に走る（門が i8a8 経路だけのものであること）
      const baseline = await runLinear(gpu, prepared, {});
      assertEquals(baseline.y.shape, [1, 4]);
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// w4a8（i4 常駐の重み × per-token i8 活性 — perf-ledger Q-8）
// ---------------------------------------------------------------------------

/**
 * この節が固定する数値契約は i8 変種と**同じ atol=0** で、成立の根拠だけが違う:
 * group の中は i32 の厳密内積・group 境界でだけ f32 へ 1 回 flush・最後に xs を 1 回。
 * したがって浮動小数の演算は出力 1 要素あたり `k/g + 1` 回に固定され、タイル分割・
 * 充填の担当・dp4a / エミュのどれを変えても値は 1 ビットも動かない。
 *
 * MUST: 期待値に **f32 i4 経路（fake-quant 重みの f32 GEMM）を使わない**。あちらは
 * 活性が f32 のままなので、一致してしまうなら活性量子化が効いていない — 恒真化そのもの。
 * 下の「既定経路と値が割れる」門がその裏取りになる。
 * MUST: 重みは**隣接要素の符号を交互**にする（nibble の上下を取り違えても対称パターンでは
 * 値が合う — ADR 0069 決定 4 ①。i4lanes の並びに対する唯一の値の側の検出器）。
 * MUST: 重みは **group ごとに振幅を変える**（全 group 同じ scale だと group 添字の
 * 取り違え〈行/group の入れ替え・shift 誤り〉が一切値に出ない）。
 */
const groupVarying = (cols: number, groupSize: number) => (i: number): number => {
  const group = Math.floor((i % cols) / groupSize);
  const row = Math.floor(i / cols);
  const base = (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);
  return base * (1 + group * 0.75 + (row % 5) * 0.25);
};

/** `linear(x, w, b)` 1 本のグラフ（w は i4 + group scale）。 */
const i4LinearModel = (
  m: number,
  n: number,
  k: number,
  groupSize: number,
  quantized: ReturnType<typeof quantizeI4>,
  bias: FilledTensor,
): ArrayBuffer => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: groupSize } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return buildSafetensors(
    [
      { name: "m.w", dtype: "I4", shape: [n, k], data: quantized.bytes },
      {
        name: "m.s",
        dtype: "F32",
        shape: [...quantized.scaleShape],
        data: f32Bytes([...quantized.scale]),
      },
      { name: "m.b", dtype: "F32", shape: [n], data: f32Bytes([...bias.data]) },
    ],
    { karume_ir: JSON.stringify(graph) },
  );
};

type W4a8Case = {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly groupSize: number;
};

/**
 * 形の踏み分け。
 *
 * MUST: **v4 経路とスカラ経路を対で持つ**（書き出しのガードが変種ごとに別の式）。
 * MUST: **m / n をタイル辺（128 × 64）を跨がせる**（1 タイル未満に潰れるとガードが一度も
 * 偽にならない）。
 * MUST: **group が行内で 2 回以上変わる形**を含める（1 group / 行だと flush が 1 回きりに
 * なり、group 添字も畳み順も検証されない）。
 * MUST: **group = K タイル（g16）と 2 タイル / group（g32）の両方**を持つ（内側ループの
 * 上限が 1 のときだけ通る誤り〈gbase の掛け忘れ〉を塞ぐ）。
 */
const W4A8_CASES: readonly W4a8Case[] = [
  // v4 / m = 130 で行タイル 2 枚・n = 68 で列タイル 2 枚（最終タイルの有効 quad は 16 中 1）
  // / k = 96 g32 = 行内 3 group・1 group が K タイル 2 枚
  { name: "w4a8 v4 [130,96] × W[68,96] g32", m: 130, n: 68, k: 96, groupSize: 32 },
  // スカラ（n % 4 != 0）/ m = 129・n = 67 でどちらもタイル辺を跨ぐ / k = 48 g16 = 行内
  // 3 group・**1 group がちょうど K タイル 1 枚**
  { name: "w4a8 スカラ [129,48] × W[67,48] g16", m: 129, n: 67, k: 48, groupSize: 16 },
  // 1 タイルに収まる小さい形（ガードが全て真のまま通る経路）・行内 1 group
  { name: "w4a8 v4 小 [3,32] × W[8,32] g32", m: 3, n: 8, k: 32, groupSize: 32 },
  // スカラ小 / k = 96 g32 = 行内 3 group（v4 の A と group 構成だけ揃えた対照）
  { name: "w4a8 スカラ 小 [17,96] × W[19,96] g32", m: 17, n: 19, k: 96, groupSize: 32 },
];

type PreparedW4a8 = {
  readonly model: ArrayBuffer;
  readonly x: Tensor;
  readonly expected: Float32Array<ArrayBuffer>;
};

const prepareW4a8 = (
  testCase: W4a8Case,
  xValue: (index: number, m: number, k: number) => number = SIGNED,
): PreparedW4a8 => {
  const { m, n, k, groupSize } = testCase;
  const weight = fill([n, k], groupVarying(k, groupSize));
  const quantized = quantizeI4(weight.data, [n, k], groupSize);
  // 恒真化の門: 隣接要素の符号が交互でないと nibble の取り違えが値に出ない
  let alternating = 0;
  for (let i = 0; i + 1 < quantized.q.length; i += 1) {
    if (quantized.q[i] > 0 && quantized.q[i + 1] < 0) alternating += 1;
  }
  if (alternating * 4 < quantized.q.length) {
    throw new Error(`${testCase.name}: 符号が交互になっている隣接対が ${alternating} 組しかない`);
  }
  const bias = fill([n], SIGNED);
  const x = fill([m, k], (index) => xValue(index, m, k));
  // atol=0 を主張してよいデータであることの門（ファイル冒頭の {@link SIGNED} の MUST）。
  // 非有限行・全ゼロ行は丸めが走らないので対象外へ置き換える（門を残したまま非有限を通す）。
  const gated = Float32Array.from(x.data);
  for (let row = 0; row < m; row += 1) {
    const slice = gated.subarray(row * k, row * k + k);
    const exempt = slice.some((value) => !Number.isFinite(value)) ||
      slice.every((value) => value === 0);
    if (exempt) {
      for (let i = 0; i < k; i += 1) slice[i] = SIGNED(row * k + i);
    }
  }
  const margin = quantizeRowsTieMargin(gated, m, k);
  if (!(margin > TIE_MARGIN)) {
    throw new Error(`${testCase.name}: 丸め境界からの余裕が ${margin} しかない`);
  }
  const expected = referenceLinearW4a8({
    x: x.data,
    weight: quantized.q,
    weightScale: quantized.scale,
    bias: bias.data,
    m,
    n,
    k,
    groupSize,
  });
  return { model: i4LinearModel(m, n, k, groupSize, quantized, bias), x, expected };
};

Deno.test({
  name:
    "linearCompute:'a8' × i4 常駐（w4a8）が TS 参照と atol=0 で一致する（v4 / スカラ / タイル端 / group 2 種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const testCase of W4A8_CASES) {
        const prepared = prepareW4a8(testCase);
        const actual = await runLinear(gpu, prepared, { linearCompute: "a8" });
        assertEquals(actual.y.shape, [testCase.m, testCase.n], testCase.name);
        assertExact(actual.y.data, prepared.expected, testCase.name);
        // 恒真化の門: 出力が定数なら「一致」は何も検証していない
        assert(new Set([...actual.y.data]).size > 1, `${testCase.name}: 出力が定数`);
        // 実際に w4a8 経路を通っている（quantize_rows + w4a8 GEMM の 2 本）
        assertEquals(actual.pipelineCount, 2, `${testCase.name}: パイプライン本数`);
        if (actual.keys.length > 0) {
          const key = linearI8a8Key(
            linearI8a8UsesVec4(testCase.n),
            true,
            undefined,
            "i4",
            testCase.groupSize,
          );
          assertEquals(
            actual.keys,
            [key, quantizeRowsKey(quantizeRowsGeometry(testCase.k))].sort(),
            `${testCase.name}: 走ったパイプラインキー`,
          );
          // 診断で「i4 常駐 × その group 長」が読めること（ADR 0021）
          assertEquals(
            key.endsWith(`:wi4g${testCase.groupSize}`),
            true,
            `${testCase.name}: 判別子`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "w4a8: NaN 行は行全体へ伝播し、ゼロ行と負値は厳密に一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // 非有限の伝播は i8a8 と**同じ根拠に立たない** — w4a8 は `xs · wscale` の早期畳み込みを
    // 使えず（wscale が group ごとに変わる）、`xs` を最後の fma 1 回へ回す。したがって
    // `xs = ±Inf` かつ group 縮約が 0 の組で `fma(0, Inf, bias)` のような分岐が割れる余地があり、
    // ファイル冒頭が全体の契約として書いている「NaN 行の伝播（設計の一部）」を新設変種でも
    // 別に固定する。
    const testCase = W4A8_CASES[0];
    const { m, n } = testCase;
    const nanRow = 2;
    const zeroRow = 5;
    const prepared = prepareW4a8(testCase, (index, _m, cols) => {
      const row = Math.floor(index / cols);
      if (row === zeroRow) return 0;
      if (row === nanRow && index % cols === cols - 3) return Number.NaN;
      return SIGNED(index);
    });

    const gpu = await acquireGpu();
    let y: Tensor;
    try {
      y = (await runLinear(gpu, prepared, { linearCompute: "a8" })).y;
    } finally {
      gpu.destroy();
    }

    assertExact(y.data, prepared.expected, "w4a8 NaN / ゼロ行");
    for (let col = 0; col < n; col += 1) {
      assert(Number.isNaN(y.data[nanRow * n + col]), `NaN 行 col=${col}`);
      assert(Number.isFinite(y.data[(nanRow + 1) * n + col]), `隣接行 col=${col}`);
      assertEquals(
        y.data[zeroRow * n + col],
        prepared.expected[zeroRow * n + col],
        `ゼロ行 col=${col}`,
      );
    }
    // 恒真化の門: NaN 行・ゼロ行の外が定数なら「一致」は何も検証していない。
    const others = [...y.data].filter((_value, index) => {
      const row = Math.floor(index / n);
      return row !== nanRow && row !== zeroRow;
    });
    assert(new Set(others).size > 1, "NaN / ゼロ行の外が定数");
    assertEquals(y.shape, [m, n]);
  },
});

Deno.test({
  name:
    "w4a8: dot4I8Packed 版とエミュ版が atol=0 で一致する（i4 レーン展開は内積変種に依らない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const dots: readonly I8a8Dot[] = ["dp4a", "emu"];
      for (const testCase of W4A8_CASES) {
        const prepared = prepareW4a8(testCase);
        const results: Record<string, Tensor> = {};
        for (const dot of dots) {
          const options: SessionOptions = { linearCompute: "a8", [I8A8_DOT]: dot };
          const actual = await runLinear(gpu, prepared, options);
          results[dot] = actual.y;
          if (actual.keys.length > 0) {
            assertEquals(
              actual.keys.includes(
                linearI8a8Key(
                  linearI8a8UsesVec4(testCase.n),
                  dot === "dp4a",
                  undefined,
                  "i4",
                  testCase.groupSize,
                ),
              ),
              true,
              `${testCase.name}: ${dot} のキー`,
            );
          }
        }
        assertExact(results["emu"].data, results["dp4a"].data, `${testCase.name}: dp4a vs エミュ`);
        // どちらも TS 参照と一致する（両者が同じだけずれている形を塞ぐ）
        assertExact(results["emu"].data, prepared.expected, `${testCase.name}: エミュ vs 参照`);
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "w4a8 は opt-in のときだけ効き、既定の f32 i4 経路とは値が割れる（活性量子化の裏取り・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const testCase = W4A8_CASES[3];
      const prepared = prepareW4a8(testCase);
      // ① 既定（linearCompute 省略）は i4 資産でも従来の f32 計算経路（linear 1 本だけ）
      const baseline = await runLinear(gpu, prepared, {});
      assertEquals(baseline.pipelineCount, 1, "既定はパイプライン 1 本（quantize_rows が出ない）");
      if (baseline.keys.length > 0) {
        assertEquals(
          baseline.keys,
          [linearKey("i4", false, "f32", testCase.m, testCase.groupSize)],
          "既定のキー",
        );
      }
      // MUST: 既定の値は活性 f32 のままなので w4a8 の参照とは**一致しない**。一致するなら
      // 活性量子化が効いていない = 期待値に f32 i4 経路を使ったのと同じ恒真化。
      const w4a8 = await runLinear(gpu, prepared, { linearCompute: "a8" });
      assertEquals(
        [...baseline.y.data].some((value, index) => value !== w4a8.y.data[index]),
        true,
        "活性量子化が出力を変えていない（opt-in が効いていない）",
      );
    } finally {
      gpu.destroy();
    }
  },
});

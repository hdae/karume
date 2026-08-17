// f16 **計算**変種（ADR 0028 — `SessionOptions.attentionCompute` / `linearCompute` の `"f16"`）の
// 数値契約。**重み格納 f16（ADR 0018）とは別の軸**で、こちらは共有タイルを f16 に落とす。
//
// ## このファイルが固定する数値契約: atol=0
//
// カーネルは共有タイルへ書く直前に 1 度だけ f16 へ丸め、内側ループでは f32 へ広げてから
// （拡幅は厳密）f32 で累積する。したがって
//
//     f16 変種の出力 ≡ 入力を f16 に丸めた f32 変種の出力     … 1 ビットも違わない
//
// が成立するはずで、tolerance で吸収する余地が構造的に無い（w8a8 の atol=0 と同じ質の契約）。
//
// ## オラクルの取り方（丸めだけをホストが担い、f32 の縮約は GPU に任せる）
//
// MUST: f32 の縮約を JS で書き直さない。WGSL は `a + b * c` を FFMA へ融合してよく（単一丸め）、
// `exp` の実装も規定されていないので、CPU で書いた f32 参照は**ドライバの裁量ぶん**だけ
// 割れうる。それは f16 変種の正しさとは無関係な差なので、契約に持ち込むと atol=0 が壊れる。
// 代わりに **f32 変種そのものをオラクルにし、ホストは丸め（{@link roundToF16}）だけを担う**:
//
// | 対象      | 実測                                  | オラクル                                                            |
// | --------- | ------------------------------------- | ------------------------------------------------------------------- |
// | linear    | `linearCompute:"f16"` × 素の x / w    | 既定 × **f16 に丸めた** x / w                                       |
// | attention | `attentionCompute:"f16"` × 素の q/k/v | ① bmm(f16(q·s), f16(k·s)ᵀ) → ② softmax(f16(S)) → ③ bmm(f16(P), f16(v)) |
//
// attention の 3 段は融合前の分解経路そのもの（tests/gpu_attention_parity_test.ts が
// ビット同一を既に固定している経路）で、**丸めが起きる 3 点だけ**をホストが挟む。これは
// ADR 0023 の「② は softmax のパス①②と逐語一致 / ③ の A 要素は softmax のパス③の出力と同じ式」
// という不変条件に直接乗っている — 崩れたらここが赤くなる。
//
// MUST: 恒真化しないこと。**f16 変種の出力が f32 変種と違う**ことを毎ケース確かめる
// （丸めが 1 度も起きていない実装でもオラクル一致は成立してしまうため）。
//
// 資産不要。`shader-f16` を列挙しないアダプタでは SKIP（既定の f32 経路の検証は無傷）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { f16BitsToF32, roundToF16 } from "../src/format/f16.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { attentionPvKey, attentionQkKey, attentionStatsKey } from "../src/kernels/attention.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { f32ToF16Bits, quantizeF16 } from "./helpers/f16.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";

// ---------------------------------------------------------------------------
// 丸めの正本（GPU 非依存）
// ---------------------------------------------------------------------------

Deno.test("roundToF16 は独立実装の RTNE 符号化器と f32 入力の全域で一致する（丸めの正本の交差検証）", () => {
  // MUST: 2 つの実装を突き合わせる。片方だけだと「両方同じ間違い方」を検出できない
  // （tests/helpers/f16.ts はビット演算の自前実装・src/format/f16.ts は Math.f16round）。
  //
  // MUST: 入力は **f32 の値**（`Math.fround` 済み）に揃える。ビット実装は double を一度 f32 へ
  // 落としてから f16 へ丸めるので、f32 で表せない double を渡すと**二重丸め**の差が出る
  // （例: 65519.999 は double のままなら 65504・f32 を経由すると同点で偶数側 = Inf）。
  // カーネルが丸めるのは常に f32 の値なので、契約に関わるのは f32 入力の側だけ。
  const viaBits = (value: number): number => f16BitsToF32(f32ToF16Bits(value));
  const cases: readonly number[] = [
    0,
    -0,
    1,
    // 同点（1 + 2^-11）は偶数側 1.0 へ / 1 + 3·2^-12 は上へ
    1 + 2 ** -11,
    1 + 3 * 2 ** -12,
    -(1 + 2 ** -11),
    0.1,
    -0.1,
    1 / 3,
    65504, // f16 の最大有限値
    65519, // まだ 65504 へ落ちる
    65520, // 同点 → 偶数側（仮数 0）= Inf
    65536,
    -65536,
    2 ** -14, // 最小 normal
    2 ** -24, // 最小 subnormal
    2 ** -25, // 半分 → 偶数丸めで 0
    3 * 2 ** -25, // subnormal の同点の上側
    1e-9,
    Infinity,
    -Infinity,
  ].map(Math.fround);
  for (const value of cases) {
    assertEquals(
      Object.is(roundToF16(value), viaBits(value)),
      true,
      `roundToF16(${value}) = ${roundToF16(value)} / ビット実装 = ${viaBits(value)}`,
    );
  }
  // 刻みの違う 3 本の走査で全体像も突き合わせる（境界だけ合っている実装を弾く）
  for (let i = -20000; i < 20000; i += 7) {
    for (const step of [1e-4, 3.7e-2, 11.3]) {
      const value = Math.fround(i * step);
      assert(
        Object.is(roundToF16(value), viaBits(value)),
        `roundToF16(${value}) が割れた`,
      );
    }
  }
  assert(Number.isNaN(roundToF16(Number.NaN)));
});

// ---------------------------------------------------------------------------
// 実 GPU（f16 計算変種）
// ---------------------------------------------------------------------------

const F16_GPU = GPU_AVAILABLE && SHADER_F16_AVAILABLE;

/** 決定的なデータ列（乱数は使わない）。MUST: f16 で厳密に表せない刻みにする。 */
const XS = (i: number): number => (((i * 7) % 23) - 11) * 0.1713 + 0.0271;
const WS = (i: number): number => (((i * 11) % 19) - 9) * 0.2317 - 0.0139;
const BS = (i: number): number => ((i % 5) - 2) * 0.4271;
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.1731;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.2317;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/** 半スケール（torch math decomp の `√scale_factor`）— ADR 0023 の契約どおりの値。 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** f32 のビット列（±0 の差も末尾 1 ulp の差も見える形）。 */
const bits = (data: Float32Array): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const assertBitEqual = (actual: Float32Array, expected: Float32Array, where: string): void => {
  assertEquals(actual.length, expected.length, `${where}: 長さ`);
  const a = bits(actual);
  const e = bits(expected);
  const mismatches: string[] = [];
  for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
    if (a[i] !== e[i]) mismatches.push(`[${i}] ${actual[i]} vs ${expected[i]}`);
  }
  assertEquals(mismatches, [], `${where}: ビット列が違う（${mismatches.join(" / ")}）`);
};

/** 2 つの結果が**違う**ことの確認（丸めが実際に効いていることの恒真化対策）。 */
const assertDiffers = (a: Float32Array, b: Float32Array, where: string): void => {
  const x = bits(a);
  const y = bits(b);
  let differ = false;
  for (let i = 0; i < x.length; i += 1) {
    if (x[i] !== y[i]) {
      differ = true;
      break;
    }
  }
  assert(differ, `${where}: f16 変種の出力が f32 変種と 1 ビットも違わない（丸めが効いていない）`);
  assert(new Set(x).size > 1, `${where}: 出力が定数（比較が恒真になっている）`);
};

const roundedCopy = (data: Float32Array): Float32Array<ArrayBuffer> =>
  Float32Array.from(data, roundToF16);

const roundedTensor = (tensor: FilledTensor): FilledTensor => ({
  dtype: "f32",
  shape: tensor.shape,
  data: roundedCopy(tensor.data as Float32Array),
});

type RunResult = {
  readonly output: Float32Array<ArrayBuffer>;
  /** 走ったパイプラインキー（GPU 時間計測が無い環境では空）。 */
  readonly keys: readonly string[];
};

const runModel = async (
  gpu: GpuContext,
  buffer: ArrayBuffer,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
): Promise<RunResult> => {
  const model = openModel(buffer);
  const session = await createSession(gpu, model, options);
  try {
    const outputs = await session.run(inputs);
    const tensor = outputs[model.graph.outputs[0]];
    return {
      output: tensor.data as Float32Array<ArrayBuffer>,
      keys: (session.diagnostics().lastRunTiming?.entries ?? []).map((entry) => entry.key),
    };
  } finally {
    await session.dispose();
  }
};

const runGraph = (
  gpu: GpuContext,
  graph: GraphJson,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
  tensors: readonly TensorSpec[] = [],
): Promise<RunResult> => runModel(gpu, graphModelBuffer(graph, tensors), inputs, options);

// ---------------------------------------------------------------------------
// linear
// ---------------------------------------------------------------------------

/** `linear(x, w, b)` 1 ノード（重みは**グラフ入力** = 格納 f32 の経路）。 */
const linearInputGraph = (m: number, n: number, k: number): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [m, k] },
    { name: "w", dtype: "f32", shape: [n, k] },
    { name: "b", dtype: "f32", shape: [n] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [m, n] } },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** `linear(x, w, b)` 1 ノード（重みは f16 格納の initializer = 格納 f16 × 計算 f16 の組）。 */
const linearWf16Graph = (m: number, n: number, k: number): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f16" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [m, n] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/**
 * 形状の選定（既存の GEMM parity テストに倣う）。
 *
 * MUST: v4 経路とスカラ経路を**両方**持ち、行 / 列とも 64 タイルを跨ぐケースを 1 本入れる。
 * K の偶奇・端数は共有タイルの端で 0 埋めが起きる場所なので、`k % 16 ≠ 0` を必ず含める。
 */
const LINEAR_SHAPES: readonly {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
}[] = [
  // v4（k%4 && n%4）で m / n とも 64 タイルを跨ぐ
  { name: "v4 m65 n68 k20", m: 65, n: 68, k: 20 },
  // スカラ変種（k も n も 4 の倍数でない）+ K タイル端数
  { name: "スカラ m70 n23 k37", m: 70, n: 23, k: 37 },
  // K タイルを何枚も跨ぐ v4（丸め誤差が縮約に積まれる形）
  { name: "v4 m9 n64 k132", m: 9, n: 64, k: 132 },
  // k が奇数 1（K 端数 0 埋めが 15 語）
  { name: "スカラ m5 n7 k1", m: 5, n: 7, k: 1 },
];

Deno.test({
  name: "linear の f16 計算変種は「入力を f16 に丸めた f32 変種」とビット単位で一致する（実 GPU）",
  ignore: !F16_GPU,
  fn: async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      for (const shape of LINEAR_SHAPES) {
        const { name, m, n, k } = shape;
        const x = fill([m, k], XS);
        const w = fill([n, k], WS);
        const b = fill([n], BS);
        const graph = linearInputGraph(m, n, k);

        // 実測: 素の入力 × f16 計算
        const actual = await runGraph(gpu, graph, { x, w, b }, { linearCompute: "f16" });
        // オラクル: f16 に丸めた入力 × 既定の f32 計算（bias は f32 のまま — 丸めない）
        const oracle = await runGraph(gpu, graph, {
          x: roundedTensor(x),
          w: roundedTensor(w),
          b,
        });
        // 対照: 素の入力 × 既定の f32 計算（丸めが効いていることの確認用）
        const plain = await runGraph(gpu, graph, { x, w, b });

        assertBitEqual(actual.output, oracle.output, `linear ${name}`);
        assertDiffers(actual.output, plain.output, `linear ${name}`);

        if (actual.keys.length > 0) {
          const v4 = k % 4 === 0 && n % 4 === 0;
          assertEquals(
            actual.keys.filter((key) => key === linearKey("f32", v4, "f16")).length,
            1,
            `linear ${name}: f16 変種のキーが走っていない（${actual.keys.join(", ")}）`,
          );
          assertEquals(
            actual.keys.filter((key) => key === linearKey("f32", v4)).length,
            0,
            `linear ${name}: f32 変種のキーが残っている`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "重み f16 格納 × f16 計算の組は往復恒等（unpack2x16float の値は f16 に厳密に戻る）",
  ignore: !F16_GPU,
  fn: async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      for (const shape of LINEAR_SHAPES) {
        const { name, m, n, k } = shape;
        const x = fill([m, k], XS);
        const raw = Float32Array.from({ length: n * k }, (_, i) => WS(i));
        const weight = quantizeF16(raw);
        const bias = Float32Array.from({ length: n }, (_, i) => BS(i));
        const graph = linearWf16Graph(m, n, k);
        // MUST: F32 を先に置く（F16 の要素数が奇数だと後続 F32 の絶対 offset が
        // 4 バイト整列を外れ、safetensors のリーダが正当に落ちる）。
        const tensors: readonly TensorSpec[] = [
          { name: "m.b", dtype: "F32", shape: [n], data: f32Bytes(bias) },
          { name: "m.w", dtype: "F16", shape: [n, k], data: weight.bytes },
        ];

        const actual = await runGraph(gpu, graph, { x }, { linearCompute: "f16" }, tensors);
        // 重みは既に f16 の格子上にあるので、丸めるのは x だけ（往復恒等の主張そのもの）
        const oracle = await runGraph(gpu, graph, { x: roundedTensor(x) }, {}, tensors);
        const plain = await runGraph(gpu, graph, { x }, {}, tensors);

        assertBitEqual(actual.output, oracle.output, `linear wf16 ${name}`);
        assertDiffers(actual.output, plain.output, `linear wf16 ${name}`);

        if (actual.keys.length > 0) {
          const v4 = k % 4 === 0 && n % 4 === 0;
          assertEquals(
            actual.keys.filter((key) => key === linearKey("f16", v4, "f16")).length,
            1,
            `linear wf16 ${name}: f16 計算のキーが走っていない`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 融合 attention
// ---------------------------------------------------------------------------

type AttentionShape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/** MUST: 既存の f32 parity テストと同じ形状群（B と H の畳み込み・端数・v4 の踏み分け）。 */
const ATTENTION_SHAPES: readonly AttentionShape[] = [
  { name: "全異 B2 H3 M5 N11 D7", b: 2, h: 3, m: 5, n: 11, d: 7 },
  { name: "端数 B3 H1 M17 N19 D13", b: 3, h: 1, m: 17, n: 19, d: 13 },
  { name: "v4 B1 H2 M68 N20 D12", b: 1, h: 2, m: 68, n: 20, d: 12 },
  { name: "混成 B2 H2 M9 N8 D6", b: 2, h: 2, m: 9, n: 8, d: 6 },
  { name: "DiT 形 B1 H4 M64 N64 D128", b: 1, h: 4, m: 64, n: 64, d: 128 },
];

const tensorOf = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/**
 * 分解経路 3 段のオラクル。丸めが起きる点だけをホストが挟む（ADR 0028 の丸め列 1 / 4）:
 *
 * 1. `f16(q·scale)` と `f16(k·scale)` — ①QK のタイル充填。**scale の乗算は f32 で済ませてから**
 *    丸める（半スケール契約は ADR 0023）。k は `[n,d]` → `[d,n]` へホストで転置する
 *    （カーネル側の「共有メモリで転置」と同じ配置になる）。
 * 2. `f16(S)` — ①QK の書き出し（S は f16 で受け渡す）。
 * 3. `f16(P)` と `f16(v)` — ③PV のタイル充填。`P = exp(S−m)·inv` は softmax のパス③の出力
 *    そのもの（ADR 0023 の成立根拠 4）。
 */
const attentionOracle = async (
  gpu: GpuContext,
  shape: AttentionShape,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  scale: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;

  // ① q·scale を f16 へ / k·scale を [heads, d, n] へ転置しつつ f16 へ
  const qs = Float32Array.from(q, (value) => roundToF16(Math.fround(value * scale)));
  const kt = new Float32Array(heads * d * n);
  for (let head = 0; head < heads; head += 1) {
    for (let col = 0; col < n; col += 1) {
      for (let depth = 0; depth < d; depth += 1) {
        kt[(head * d + depth) * n + col] = roundToF16(
          Math.fround(k[(head * n + col) * d + depth] * scale),
        );
      }
    }
  }
  const scores = await runGraph(
    gpu,
    singleOpGraph("bmm", [[heads, m, d], [heads, d, n]], [[heads, m, n]]),
    { x0: tensorOf([heads, m, d], qs), x1: tensorOf([heads, d, n], kt) },
  );

  // ② S を f16 へ（① の書き出しの丸め）→ 行統計と P は softmax がそのまま作る
  const probs = await runGraph(
    gpu,
    singleOpGraph("softmax", [[heads, m, n]], [[heads, m, n]], { attrs: { dim: 2 } }),
    { x0: tensorOf([heads, m, n], roundedCopy(scores.output)) },
  );

  // ③ P と v を f16 へ（③ のタイル充填の丸め）
  const out = await runGraph(
    gpu,
    singleOpGraph("bmm", [[heads, m, n], [heads, n, d]], [[heads, m, d]]),
    {
      x0: tensorOf([heads, m, n], roundedCopy(probs.output)),
      x1: tensorOf([heads, n, d], roundedCopy(v as Float32Array<ArrayBuffer>)),
    },
  );
  return out.output;
};

Deno.test({
  name: "融合 attention の f16 変種は分解経路 + 丸め 3 点とビット単位で一致する（実 GPU）",
  ignore: !F16_GPU,
  fn: async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      for (const shape of ATTENTION_SHAPES) {
        const { name, b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const scale = halfScale(d);
        const graph = singleOpGraph("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
          attrs: { scale },
        });

        const actual = await runGraph(gpu, graph, { x0: q, x1: k, x2: v }, {
          attentionCompute: "f16",
        });
        const plain = await runGraph(gpu, graph, { x0: q, x1: k, x2: v });
        const oracle = await attentionOracle(
          gpu,
          shape,
          q.data as Float32Array,
          k.data as Float32Array,
          v.data as Float32Array,
          scale,
        );

        assertBitEqual(actual.output, oracle, `attention ${name}`);
        assertDiffers(actual.output, plain.output, `attention ${name}`);

        if (actual.keys.length > 0) {
          const qkV4 = d % 4 === 0 && n % 4 === 0;
          const pvV4 = n % 4 === 0 && d % 4 === 0;
          const running = new Set(actual.keys);
          for (
            const key of [
              attentionQkKey(qkV4, "f16"),
              attentionStatsKey("f16"),
              attentionPvKey(pvV4, "f16"),
            ]
          ) {
            assert(running.has(key), `attention ${name}: '${key}' が走っていない`);
          }
          // 3 カーネルは**同時に**切り替わる（S の格納形が書き手と読み手で一致する条件）
          for (
            const key of [
              attentionQkKey(qkV4),
              attentionStatsKey(),
              attentionPvKey(pvV4),
            ]
          ) {
            assert(!running.has(key), `attention ${name}: f32 変種 '${key}' が残っている`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// fail loudly（黙って f32 へ落とさない）
// ---------------------------------------------------------------------------

Deno.test({
  name: "f16 計算の要求は shader-f16 無しの device では Session 構築で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // MUST: **shaderF16 を要求していない** device を作る（既定の acquireGpu）。
    const gpu = await acquireGpu();
    try {
      assertEquals(gpu.shaderF16Enabled, false, "既定の acquireGpu は shader-f16 を要求しない");
      const buffer = graphModelBuffer(linearInputGraph(4, 8, 4));
      for (const options of [{ linearCompute: "f16" }, { attentionCompute: "f16" }] as const) {
        const error = await assertRejects(
          () => createSession(gpu, openModel(buffer), options),
          ExecutionError,
        );
        assert(
          error.message.includes("acquireGpu({ shaderF16: true })"),
          `案内が足りない: ${error.message}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "i8 常駐の重み × linearCompute 'f16'（w8a16）は fail loudly（実 GPU）",
  ignore: !F16_GPU,
  fn: async () => {
    const m = 4;
    const n = 8;
    const k = 8;
    const raw = Float32Array.from({ length: n * k }, (_, i) => WS(i));
    const weight = quantizeI8(raw, [n, k], 0);
    const graph: GraphJson = {
      format: "karume-ir",
      version: 1,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
      outputs: ["y"],
      initializers: {
        w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.w_scale" } },
        b: { tensor: "m.b", storage: { dtype: "f32" } },
      },
      values: {
        w: { dtype: "f32", shape: [n, k] },
        b: { dtype: "f32", shape: [n] },
        y: { dtype: "f32", shape: [m, n] },
      },
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    };
    const tensors: readonly TensorSpec[] = [
      { name: "m.w", dtype: "I8", shape: [n, k], data: weight.bytes },
      {
        name: "m.w_scale",
        dtype: "F32",
        shape: [...weight.scaleShape],
        data: f32Bytes(weight.scale),
      },
      {
        name: "m.b",
        dtype: "F32",
        shape: [n],
        data: f32Bytes(Float32Array.from({ length: n }, (_, i) => BS(i))),
      },
    ];
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      const buffer = graphModelBuffer(graph, tensors);
      const x = fill([m, k], XS);
      // MUST: i8a8 と f32 は通る（落とすのは w8a16 の組だけ）
      await runModel(gpu, buffer, { x }, {});
      await runModel(gpu, buffer, { x }, { linearCompute: "i8a8" });
      const error = await assertRejects(
        () => runModel(gpu, buffer, { x }, { linearCompute: "f16" }),
        ExecutionError,
      );
      assert(error.message.includes("w8a16"), `案内が足りない: ${error.message}`);
    } finally {
      gpu.destroy();
    }
  },
});

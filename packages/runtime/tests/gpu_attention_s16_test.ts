// attention スコア S の **f16 格納**変種（案 γ 波 1 — `SessionOptions.attentionScoreStorage`）の
// 実行経路。実装 = src/kernels/score-storage.ts（部品の正本）+ src/kernels/gemm.ts /
// src/kernels/attention.ts / src/kernels/attention-i8a8.ts（挿入点）+ src/runtime/executor.ts
// （適格判定と S の確保幅）。設計 = docs/research/2026-08-04-intermediate-f16-design.md §3.4②。
//
// ## このファイルが固定する数値契約: atol=0
//
// s16 は S を `array<u32>` に `pack2x16float` で 2 要素／語詰める。**丸めは格納の 1 回だけ**で
// 読み側の `unpack2x16float` は厳密なので、
//
//     s16 変種の出力 ≡ S をホストで f16 に丸めた f32 変種の出力     … 1 ビットも違わない
//
// が成立するはず（演算は f32 のままなので定義上の恒等式 — ADR 0028 の f16 **タイル計算**より
// 契約が強い）。tolerance で吸収する余地は構造的に無い。
//
// ## オラクルの取り方（丸めだけをホストが担い、f32 の縮約は GPU に任せる）
//
// MUST: f32 の縮約を JS で書き直さない（WGSL は `a + b*c` を FFMA へ融合してよく `exp` の実装も
// 規定されていない — ADR 0028 のオラクル設計と同じ理由）。代わりに**分解経路そのものを
// オラクルにし、ホストは S の丸め 1 点だけを担う**:
//
//     ① bmm(q·scale, (k·scale)ᵀ) → S      … 融合 ①QK とビット同一
//        （tests/gpu_attention_parity_test.ts が固定している経路）
//     ② softmax(**f16 に丸めた** S) → P   … 融合 ②③ の m / inv / P と同じ式・同じ順
//     ③ bmm(P, v) → O
//
// ADR 0028 のオラクルとの違いは**丸めの点が 3 つではなく 1 つ**なところだけ。q / k / P / v は
// 素のまま = 「格納だけを f16 にし、計算は f32 のまま」という案 γ の骨格そのもの。
//
// MUST: 恒真化しないこと。**s16 の出力が f32 変種と違う**ことを毎ケース確かめる
// （丸めが 1 度も起きていない実装でもオラクル一致は成立してしまうため）。
//
// ## この波で意図的に検証していないもの
//
// - grid-stride の縮退ハーネス（tests/gpu_gridstride_test.ts）へは **②行統計の s16 変種**を
//   足してある（行方向 grid-stride の族が 1 本増えたので族の MUST）。①③ はタイル系で
//   カテゴリ違い（安全網は `tiledWorkgroups` の fail loudly と full-write の毒値注入）。
// - スカラ経路（`D % 4 != 0` or `N % 4 != 0`）に s16 は**配線していない**。非適格は f32 格納へ
//   沈黙で縮退し、検出器は下の「縮退 2 方向」テストのパイプラインキー検査 1 本になる。
//
// ## 検出限界（故障注入で実測 — 2026-08-04）
//
// **②行統計のスカラ読み `score_at` の偶奇選択は数値網では検出できない**。`select` の 2 枝を
// 入れ替える注入（要素 `i` の代わりに `i^1` を読む）を入れても本ファイルも
// tests/gpu_gridstride_test.ts も緑のまま通る — 適格条件が `N % 4 == 0` を課すので行頭が
// 必ず偶数添字になり、入れ替えは**行内の対ごとの置換**にしかならないからである（`amax` は
// 置換不変・`Σexp` も本形状では最終 ulp まで一致した）。ADR 0018 の「行内の相対添字で偶奇を
// 取る罠」が、s16 では適格条件によって**無害化されている**という形。
// 実働の検出器は **WGSL スナップショット**（tests/fixtures/wgsl/attention_stats_s16.wgsl）
// ただ 1 本で、これは意図した構造の固定であって数値の門ではない。
// 他の 5 件（2 語の詰め順 / quad 読みの 2 語目 / 適格判定の `D % 4` / キーの語 / opt-in の
// 沈黙失効）は全て本ファイルか E2E が赤で捕まえる。
//
// 資産不要。**`shader-f16` も不要**（core WGSL の pack2x16float — ADR 0030 決定 1 が無傷で
// あることの実行側の証拠）。アダプタ無し環境は明示 SKIP。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { openModel } from "../src/format/container.ts";
import { roundToF16 } from "../src/format/f16.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  attentionPvKey,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsRegCache,
} from "../src/kernels/attention.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
} from "../src/kernels/attention-i8a8.ts";
import { GEMM_TILE } from "../src/kernels/gemm.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import { referenceAttentionPvQuant } from "../src/reference/i8a8.ts";
import { stridedKey } from "../src/codegen/strided.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { f32ToF16Bits } from "./helpers/f16.ts";
import { fill, type FilledTensor, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

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
  assert(
    [...x].some((value, index) => value !== y[index]),
    `${where}: s16 の出力が f32 格納と 1 ビットも違わない（丸めが効いていない）`,
  );
  assert(new Set(x).size > 1, `${where}: 出力が定数（比較が恒真になっている）`);
};

const roundedCopy = (data: Float32Array): Float32Array<ArrayBuffer> =>
  Float32Array.from(data, roundToF16);

/** S を GPU と同じ格納形（`pack2x16float` 相当の 2 要素／語）へ詰める。 */
const packScores = (scores: Float32Array): Uint32Array<ArrayBuffer> => {
  assertEquals(scores.length % 2, 0, "s16 の適格形は N % 4 == 0 なので要素数は必ず偶数");
  const words = new Uint32Array(scores.length / 2);
  for (let i = 0; i < words.length; i += 1) {
    // MUST: 下位半分が偶数添字（`pack2x16float(vec2(x, y))` の x 側）。入れ替えると
    // GPU の読みと 1 要素ずれ、行の端だけが静かに誤る
    words[i] = (f32ToF16Bits(scores[2 * i]) & 0xffff) |
      ((f32ToF16Bits(scores[2 * i + 1]) & 0xffff) << 16);
  }
  return words;
};

// ---------------------------------------------------------------------------
// 形状群とデータ
// ---------------------------------------------------------------------------

type AttentionShape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * 適格形（`D % 4 == 0 && N % 4 == 0` = 書き手 ①QK が v4 経路を取る条件）。
 *
 * MUST: **B·H ≥ 2 かつ head ごとにパターンが違う**形を持つ（バッチ base の唯一の検出器 —
 * ADR 0030 の検出限界 5）。
 * MUST: **m / n をタイル辺 64 を跨がせる**形を含む（端タイルの書き出しが 2 語ちょうどで
 * 収まっていることの検出器）。
 * MUST: **N % 16 != 0** を含む（③PV の K タイル 16 の端数 = S 側の 0 埋めが効いている形）。
 */
const S16_SHAPES: readonly AttentionShape[] = [
  // B·H = 6 / m は 64 タイル 2 枚 / N=68 は k4=17 で K タイル 5 枚（端数 1 quad）
  { name: "B2 H3 M65 N68 D20", b: 2, h: 3, m: 65, n: 68, d: 20 },
  // N=20 は K タイル端数（k4=5）・D=12 で列 quad が 3 本
  { name: "B3 H1 M17 N20 D12", b: 3, h: 1, m: 17, n: 20, d: 12 },
  // 1 タイルに収まる最小形（ガードが全て真のまま通る経路）
  { name: "B1 H2 M3 N8 D4", b: 1, h: 2, m: 3, n: 8, d: 4 },
  // n が 64 タイル 2 枚（S の行が複数の workgroup に割れる形）
  { name: "B2 H2 M9 N132 D8", b: 2, h: 2, m: 9, n: 132, d: 8 },
  // 実測形（DiT の self-attention を縮めた形）
  { name: "DiT 形 B1 H4 M64 N64 D128", b: 1, h: 4, m: 64, n: 64, d: 128 },
];

/** 非適格形（s16 は f32 格納へ沈黙で縮退する）。両方向（D 側 / N 側）を持つ。 */
const DEGRADED_SHAPES: readonly (AttentionShape & { readonly why: string })[] = [
  { name: "D%4!=0 B1 H2 M17 N20 D13", why: "D", b: 1, h: 2, m: 17, n: 20, d: 13 },
  { name: "N%4!=0 B1 H2 M17 N19 D20", why: "N", b: 1, h: 2, m: 17, n: 19, d: 20 },
];

/**
 * 決定的なデータ列（乱数は使わない）。
 *
 * MUST: f16 で厳密に表せない刻みにする（表せる値ばかりだと丸めが 1 度も起きず、
 * オラクル一致が恒真になる）。周期を互いに非共約にしてあるので、head 境界（`m·d` 要素）と
 * 揃わず **head ごとにパターンが変わる**。
 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.1731 + 0.0271;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.2317 - 0.0139;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119 + 0.0413;

const tensorOf = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

// ---------------------------------------------------------------------------
// Session 経路の実行（診断のパイプラインキーごと拾う）
// ---------------------------------------------------------------------------

type RunResult = {
  readonly output: Float32Array<ArrayBuffer>;
  readonly entries: readonly { readonly key: string; readonly dispatchCount: number }[];
  /** 直近 run の中間バッファ実績（S の半減が見える唯一の観測点）。 */
  readonly peakTransientBytes: number;
};

const runAttention = async (
  gpu: GpuContext,
  shape: AttentionShape,
  options: SessionOptions,
): Promise<RunResult> => {
  const { b, h, m, n, d } = shape;
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, h, n, d], KEY);
  const v = fill([b, h, n, d], VALUE);
  const graph = singleOpGraph("attention", [q.shape, k.shape, v.shape], [b, h, m, d], {
    attrs: { scale: halfScale(d) },
  });
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
  try {
    const outputs = await session.run({ x0: q, x1: k, x2: v });
    const diagnostics = session.diagnostics();
    return {
      output: outputs["y"].data as Float32Array<ArrayBuffer>,
      entries: (diagnostics.lastRunTiming?.entries ?? []).map((entry) => ({
        key: entry.key,
        dispatchCount: entry.dispatchCount,
      })),
      peakTransientBytes: diagnostics.lastRun?.peakTransientBytes ?? 0,
    };
  } finally {
    await session.dispose();
  }
};

const runGraph = async (
  gpu: GpuContext,
  graph: ReturnType<typeof singleOpGraph>,
  inputs: Readonly<Record<string, Tensor | FilledTensor>>,
): Promise<Float32Array<ArrayBuffer>> => {
  const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
  try {
    const outputs = await session.run(inputs);
    return outputs["y"].data as Float32Array<ArrayBuffer>;
  } finally {
    await session.dispose();
  }
};

/**
 * 分解経路 3 段のオラクル。**丸めが起きるのは S の 1 点だけ**（冒頭の設計）。
 *
 * `q·scale` / `k·scale` は f32 のまま（半スケール契約は ADR 0023）。k は `[n,d]` → `[d,n]` へ
 * ホストで転置する（カーネル側の「共有メモリで転置」と同じ配置になる）。
 */
const s16Oracle = async (
  gpu: GpuContext,
  shape: AttentionShape,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  scale: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;
  const qs = Float32Array.from(q, (value) => Math.fround(value * scale));
  const kt = new Float32Array(heads * d * n);
  for (let head = 0; head < heads; head += 1) {
    for (let col = 0; col < n; col += 1) {
      for (let depth = 0; depth < d; depth += 1) {
        kt[(head * d + depth) * n + col] = Math.fround(k[(head * n + col) * d + depth] * scale);
      }
    }
  }
  const scores = await runGraph(
    gpu,
    singleOpGraph("bmm", [[heads, m, d], [heads, d, n]], [heads, m, n]),
    { x0: tensorOf([heads, m, d], qs), x1: tensorOf([heads, d, n], kt) },
  );
  // ここが唯一の丸め（①QK の書き出し = pack2x16float に対応する）
  const probs = await runGraph(
    gpu,
    singleOpGraph("softmax", [[heads, m, n]], [heads, m, n], { attrs: { dim: 2 } }),
    { x0: tensorOf([heads, m, n], roundedCopy(scores)) },
  );
  return await runGraph(
    gpu,
    singleOpGraph("bmm", [[heads, m, n], [heads, n, d]], [heads, m, d]),
    {
      x0: tensorOf([heads, m, n], probs),
      x1: tensorOf([heads, n, d], v as Float32Array<ArrayBuffer>),
    },
  );
};

// ---------------------------------------------------------------------------
// (1) 格納の丸め = pack2x16float が RTE であることの直接の門
// ---------------------------------------------------------------------------

/**
 * 丸め境界（f16 の同点）を狙い撃つ f32 の値。`Math.f16round` の正本テスト
 * （tests/gpu_f16_compute_test.ts）と同じ列を使う — あちらは WGSL の `f16()` を、ここは
 * `pack2x16float` を突き合わせる。
 *
 * MUST: `Math.fround` 済みにする（f32 で表せない double を渡すと**二重丸め**の差が出る）。
 */
const TIE_VALUES: readonly number[] = [
  0,
  1,
  1 + 2 ** -11, // 同点 → 偶数側 1.0
  1 + 3 * 2 ** -12, // 同点の上側
  -(1 + 2 ** -11),
  0.1,
  -0.1,
  1 / 3,
  65504, // f16 の最大有限値
  65519, // まだ 65504 へ落ちる
  65520, // 同点 → 偶数側（仮数 0）= Inf
  -65536,
  2 ** -14, // 最小 normal
  2 ** -24, // 最小 subnormal
  2 ** -25, // 半分 → 偶数丸めで 0
  3 * 2 ** -25, // subnormal の同点の上側
  5 * 2 ** -25,
  1e-9,
  1024 + 0.5,
  1024 + 1.5,
  2048 + 1,
  2048 + 3,
].map(Math.fround);

/** f16 の 16bit パターン → f32（本番 {@link "../src/format/f16.ts"} と独立の読み戻し）。 */
const decodeHalf = (pattern: number): number => {
  const sign = (pattern & 0x8000) !== 0 ? -1 : 1;
  const exponent = (pattern >> 10) & 0x1f;
  const mantissa = pattern & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

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

Deno.test({
  name:
    "①QK の s16 書き出しは Math.f16round とビット一致する（pack2x16float の丸めが RTE であることの直接の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // `scale = 1` かつ k を単位ベクトルにすると `S[m][n] = q[m][n % D]` が**厳密**に出る
    // （0 との積と 0 の加算しか混ざらないので内積の丸めがゼロ）。したがって S の各要素は
    // 上の同点値そのもので、書き出された 16bit が Math.f16round と一致するかだけが残る。
    const m = 32;
    const n = 8;
    const d = 4;
    const q = new Float32Array(m * d);
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < d; col += 1) {
        q[row * d + col] = TIE_VALUES[(row * d + col) % TIE_VALUES.length];
      }
    }
    const k = new Float32Array(n * d);
    for (let col = 0; col < n; col += 1) k[col * d + (col % d)] = 1;

    const gpu = await acquireGpu();
    // MUST: **shader-f16 を要求しない既定の device** で走ること（s16 は core WGSL）
    assertEquals(gpu.shaderF16Enabled, false, "既定の acquireGpu は shader-f16 を要求しない");
    const scheduler = new SubmitScheduler(gpu);
    const cache = new PipelineCache(gpu.device);
    const arena = new RunArena(gpu.device, () => scheduler.flush());
    try {
      const host = (source: Float32Array<ArrayBuffer>): GPUBuffer => {
        const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
        gpu.device.queue.writeBuffer(buffer, 0, source);
        return buffer;
      };
      const params = arena.allocHostWritten(16, UNIFORM_IN);
      gpu.device.queue.writeBuffer(params, 0, attentionQkParams(m, n, d, 1));
      const scores = arena.allocStorage(m * n * 2);
      arena.retain(scores, 0, { pinned: true });
      const key = attentionQkKey(true, "f32", "f16");
      const { pipeline, layout } = await cache.get(key, attentionQkWgsl(true, "f32", "f16"));
      const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
      scheduler.dispatch(
        pipeline,
        gpu.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: host(q) } },
            { binding: 2, resource: { buffer: host(k) } },
            { binding: 3, resource: { buffer: scores } },
          ],
        }),
        [
          tiledWorkgroups(n, GEMM_TILE, limit, "s16 丸めの門 ①QK"),
          tiledWorkgroups(m, GEMM_TILE, limit, "s16 丸めの門 ①QK"),
          1,
        ],
        key,
      );
      await scheduler.flush();

      const raw = await readbackBytes(gpu.device, scores, m * n * 2);
      const words = new Uint32Array(raw.buffer, 0, (m * n) / 2);
      let rounded = 0;
      for (let index = 0; index < m * n; index += 1) {
        const source = q[Math.floor(index / n) * d + (index % n) % d];
        const word = words[index >> 1];
        const got = decodeHalf((index & 1) === 1 ? word >>> 16 : word & 0xffff);
        const want = roundToF16(source);
        assert(
          Object.is(got, want) || (Number.isNaN(got) && Number.isNaN(want)),
          `S[${index}]: GPU ${got} / Math.f16round(${source}) = ${want}`,
        );
        if (!Object.is(got, source)) rounded += 1;
      }
      // 恒真化の門: 1 要素も丸まっていないなら「一致」は何も検証していない
      assert(rounded > 0, `丸めが 1 度も起きていない（${rounded} / ${m * n}）`);
    } finally {
      await arena.destroy();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (2) atol=0（s16 変種 ≡ S をホストで f16 に丸めた f32 変種）
// ---------------------------------------------------------------------------

Deno.test({
  name:
    "attentionScoreStorage:'f16' は「S を f16 に丸めた f32 変種」とビット単位で一致する（v4 / タイル端 / K 端数 / B·H≥2・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const shape of S16_SHAPES) {
        const { name, b, h, m, n, d } = shape;
        assert(attentionScoreUsesF16(d, n), `${name}: 適格形のはず`);
        const scale = halfScale(d);
        const actual = await runAttention(gpu, shape, { attentionScoreStorage: "f16" });
        const plain = await runAttention(gpu, shape, {});
        const oracle = await s16Oracle(
          gpu,
          shape,
          fill([b, h, m, d], QUERY).data as Float32Array,
          fill([b, h, n, d], KEY).data as Float32Array,
          fill([b, h, n, d], VALUE).data as Float32Array,
          scale,
        );

        assertBitEqual(actual.output, oracle, `attention s16 ${name}`);
        assertDiffers(actual.output, plain.output, `attention s16 ${name}`);
        // head ごとに値が違うこと（バッチ base の検出器が成立する前提そのもの）
        if (b * h > 1) {
          const head0 = actual.output.slice(0, m * d);
          const head1 = actual.output.slice(m * d, 2 * m * d);
          assert(
            [...head0].some((value, index) => value !== head1[index]),
            `${name}: head 0 と head 1 の出力が同一（base の取り違えが値に出ない）`,
          );
        }
        if (actual.entries.length > 0) {
          const running = new Map(actual.entries.map((entry) => [entry.key, entry.dispatchCount]));
          // 3 カーネルは**同時に**切り替わる（S の格納形が書き手と読み手で一致する条件）
          for (
            const key of [
              attentionQkKey(true, "f32", "f16"),
              attentionStatsKey("f32", "f16", attentionStatsRegCache(n)),
              attentionPvKey(true, "f32", "f16"),
            ]
          ) {
            assertEquals(running.get(key), 1, `${name}: '${key}' が 1 本走っていない`);
          }
          for (
            const key of [
              attentionQkKey(true),
              attentionStatsKey("f32", "f32", attentionStatsRegCache(n)),
              attentionPvKey(true),
            ]
          ) {
            assertEquals(running.has(key), false, `${name}: f32 格納の '${key}' が残っている`);
          }
        }
        // S の確保が半分になっている（案 γ の本体価値 — RunArena の実測で見える形）
        if (plain.peakTransientBytes > 0) {
          assertEquals(
            plain.peakTransientBytes - actual.peakTransientBytes,
            b * h * m * n * 2,
            `${name}: peakTransient の差が S の半減ぶんと一致しない`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (3) 適格判定の 2 方向（適格 → s16 キー / 非適格 → f32 キー + ビット同一）
// ---------------------------------------------------------------------------

Deno.test({
  name:
    "attentionScoreStorage:'f16' の非適格形（D%4 / N%4）は f32 格納へ沈黙で縮退する（検出器はキー検査・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (const shape of DEGRADED_SHAPES) {
        const { name, why, d, n } = shape;
        assertEquals(attentionScoreUsesF16(d, n), false, `${name}: 非適格形のはず`);
        const requested = await runAttention(gpu, shape, { attentionScoreStorage: "f16" });
        const plain = await runAttention(gpu, shape, {});
        // 縮退は**沈黙**（値は f32 経路とビット同一 — 丸めが 1 度も起きない）
        assertBitEqual(requested.output, plain.output, `${name}: 縮退が f32 経路と一致しない`);
        assert(new Set(bits(plain.output)).size > 1, `${name}: 出力が定数`);
        if (requested.entries.length > 0) {
          const keys = new Set(requested.entries.map((entry) => entry.key));
          for (const key of [...keys]) {
            assertEquals(key.endsWith(":s16"), false, `${name}（${why} 側）: s16 キーが走っている`);
          }
          assertEquals(
            keys.has(attentionStatsKey("f32", "f32", attentionStatsRegCache(n))),
            true,
            `${name}: f32 格納の ②行統計`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (4) fail loudly（c16 との併用は黙ってどちらかに解釈しない）
// ---------------------------------------------------------------------------

Deno.test({
  name: "attentionScoreStorage 'f16' × attentionCompute 'f16' は Session 構築で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const graph = singleOpGraph("attention", [[1, 1, 4, 4], [1, 1, 4, 4], [1, 1, 4, 4]], [
        1,
        1,
        4,
        4,
      ], { attrs: { scale: halfScale(4) } });
      const buffer = graphModelBuffer(graph);
      const error = await assertRejects(
        () =>
          createSession(gpu, openModel(buffer), {
            attentionCompute: "f16",
            attentionScoreStorage: "f16",
          }),
        ExecutionError,
      );
      assert(
        error.message.includes("同時に指定できない"),
        `案内が足りない: ${error.message}`,
      );
      // MUST: **s16 単独は shader-f16 無しの device で通る**（ADR 0030 決定 1 と同じ規律 —
      // ここが落ちると案 γ の「feature 非依存」という前提そのものが崩れる）
      const session = await createSession(gpu, openModel(buffer), {
        attentionScoreStorage: "f16",
      });
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (5) 本命の組（i8a8 × s16）— 直交していることのキー検査
// ---------------------------------------------------------------------------

Deno.test({
  name:
    "attentionCompute:'i8a8' × attentionScoreStorage:'f16' は直交して同時に立つ（shader-f16 不要・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      const shape: AttentionShape = { name: "B2 H3 M65 N68 D20", b: 2, h: 3, m: 65, n: 68, d: 20 };
      const both = await runAttention(gpu, shape, {
        attentionCompute: "i8a8",
        attentionScoreStorage: "f16",
      });
      const i8a8Only = await runAttention(gpu, shape, { attentionCompute: "i8a8" });
      // S の丸めが 1 段増えるので値は動く（opt-in が効いていることの数値側の証拠）
      assertDiffers(both.output, i8a8Only.output, "i8a8 × s16");
      // S が半分になっている（i8a8 でも同じ 1 箇所の確保が効く）
      if (i8a8Only.peakTransientBytes > 0) {
        assertEquals(
          i8a8Only.peakTransientBytes - both.peakTransientBytes,
          shape.b * shape.h * shape.m * shape.n * 2,
          "peakTransient の差が S の半減ぶんと一致しない",
        );
      }
      if (both.entries.length > 0) {
        const byKey = new Map(both.entries.map((entry) => [entry.key, entry.dispatchCount]));
        // ①QK / ③PV は i8a8 のまま、S だけが f16 格納（3 つの軸が同時に立つ）
        assertEquals(byKey.get(attentionQkI8a8Key(true, true, "f16")), 1);
        assertEquals(byKey.get(attentionPvI8a8Key(true, true, "f16")), 1);
        // ②行統計は f32 計算のまま s16 を読む
        assertEquals(
          byKey.get(attentionStatsKey("f32", "f16", attentionStatsRegCache(shape.n))),
          1,
        );
        // f32 格納の変種は 1 本も残らない
        for (
          const key of [
            attentionQkI8a8Key(true, true),
            attentionPvI8a8Key(true, true),
            attentionStatsKey("f32", "f32", attentionStatsRegCache(shape.n)),
          ]
        ) {
          assertEquals(byKey.has(key), false, `f32 格納の '${key}' が残っている`);
        }
        // i8a8 の前段（q / k / Vᵀ の量子化と Vᵀ の permute）は s16 に影響されない
        assertEquals(byKey.get(QUANTIZE_ROWS_KEY), 3, "quantize_rows が q / k / v の 3 本でない");
        assertEquals(byKey.get(stridedKey({ dtype: "f32" })), 1, "Vᵀ の permute が 1 本でない");
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// (6) ③PV i8a8 の qP を s16 で再導出（±1 段 + 不一致率の門 — ADR 0030 の門の s16 版）
// ---------------------------------------------------------------------------

/** P̃ の格子と `1/127`（**観測の算術にだけ使う** — 参照側の定数は src/reference/i8a8.ts が持つ）。 */
const P_ABS_MAX = 127;
const INV_P_ABS_MAX = Math.fround(1 / P_ABS_MAX);

type QpShape = {
  readonly name: string;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
};

/** MUST: `N % 4 == 0`（③PV i8a8 の適格条件）かつ B·H ≥ 2（バッチ base の検出器）。 */
const QP_SHAPES: readonly QpShape[] = [
  { name: "B·H6 M65 N68", batch: 6, m: 65, n: 68 },
  { name: "B·H2 M17 N20", batch: 2, m: 17, n: 20 },
];

/** S の素の値（決定的）。行ごとにパターンが変わるので P̃ は 0 から 127 まで広がる。 */
const scoreAt = (b: number, row: number, col: number): number => {
  const index = col * 7 + row * 3 + b * 5;
  return ((index % 23) - 11) * 0.6373 + ((index * 13) % 11) * 0.1237 - 0.4291;
};

/**
 * ③PV i8a8 を **1 dispatch だけ**生で流す（S と行統計はテストが与え、Vᵀ の量子化結果は
 * ホストで組む）。V を head ごとの単位行列にすると `vq = 127·I` / `vs = 1/127` に**構造的に**
 * なるので、`O[m,j] = f32(127·qP[m,j]) · (prow · vs[j])` が **qP の読み出し器**になる。
 *
 * MUST: 前提（vq が対角 127・vs が 1/127）は tests/gpu_attention_pv_i8a8_test.ts が
 * `quantize_rows` の実走で固定済み。ここはその値をホストで再構成して S の格納形だけを変える。
 */
const observeQpS16 = async (
  gpu: GpuContext,
  shape: QpShape,
  packed: Uint32Array<ArrayBuffer>,
  stats: Float32Array<ArrayBuffer>,
  rowInv: Float32Array<ArrayBuffer>,
): Promise<Int8Array<ArrayBuffer>> => {
  const { batch, m, n } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const host = (source: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, source);
      return buffer;
    };
    // Vᵀ = 単位行列を量子化した結果（行 = (b, d)・N 連続の i8 を 4 詰め）
    const vq = new Int8Array(batch * n * n);
    const vs = new Float32Array(batch * n);
    for (let head = 0; head < batch; head += 1) {
      for (let row = 0; row < n; row += 1) {
        vq[(head * n + row) * n + row] = P_ABS_MAX;
        vs[head * n + row] = INV_P_ABS_MAX;
      }
    }
    const out = arena.allocStorage(Math.max(4, batch * m * n * 4));
    arena.retain(out, 0, { pinned: true });
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionPvI8a8Params(m, n, n));
    const v4 = attentionPvI8a8UsesVec4(n);
    const key = attentionPvI8a8Key(v4, true, "f16");
    const { pipeline, layout } = await cache.get(key, attentionPvI8a8Wgsl(v4, true, "f16"));
    scheduler.dispatch(
      pipeline,
      gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: host(packed) } },
          { binding: 2, resource: { buffer: host(new Uint8Array(vq.buffer)) } },
          { binding: 3, resource: { buffer: host(stats) } },
          { binding: 4, resource: { buffer: out } },
          { binding: ATTENTION_PV_V_SCALE_BINDING, resource: { buffer: host(vs) } },
        ],
      }),
      [
        tiledWorkgroups(n, GEMM_TILE, limit, `${shape.name} ③PV i8a8 s16`),
        tiledWorkgroups(m, GEMM_TILE, limit, `${shape.name} ③PV i8a8 s16`),
        tiledWorkgroups(batch, 1, limit, `${shape.name} ③PV i8a8 s16`),
      ],
      key,
    );
    await scheduler.flush();
    const bytes = await readbackBytes(gpu.device, out, batch * m * n * 4);
    const o = new Float32Array(bytes.buffer, 0, batch * m * n);
    const qp = new Int8Array(batch * m * n);
    for (let head = 0; head < batch; head += 1) {
      for (let row = 0; row < m; row += 1) {
        const prow = Math.fround(rowInv[head * m + row] * INV_P_ABS_MAX);
        for (let col = 0; col < n; col += 1) {
          const combined = Math.fround(prow * INV_P_ABS_MAX);
          const value = o[(head * m + row) * n + col] / (P_ABS_MAX * combined);
          const nearest = Math.round(value);
          // MUST: 前提が崩れたまま割ると観測値が静かにずれる（門が恒真化する）
          assert(
            Math.abs(value - nearest) < 1e-3 && nearest >= 0 && nearest <= P_ABS_MAX,
            `qP の読み出しが整数にならない [${head},${row},${col}]: ${value}`,
          );
          qp[(head * m + row) * n + col] = nearest;
        }
      }
    }
    return qp;
  } finally {
    await arena.destroy();
  }
};

Deno.test({
  name:
    "③PV i8a8 × s16: qP は「f16 に丸めた S」から作った TS 参照と必ず ±1 段以内で一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    let total = 0;
    let mismatched = 0;
    try {
      for (const shape of QP_SHAPES) {
        const { batch, m, n } = shape;
        // MUST: 参照も**丸め済み** S を読む（S の格納形が変わったのだから、行統計も qP も
        // 丸めた値から作らないと「s16 の門」ではなく「丸めの検出器」になる）
        const raw = new Float32Array(batch * m * n);
        for (let head = 0; head < batch; head += 1) {
          for (let row = 0; row < m; row += 1) {
            for (let col = 0; col < n; col += 1) {
              raw[(head * m + row) * n + col] = Math.fround(scoreAt(head, row, col));
            }
          }
        }
        const rounded = roundedCopy(raw);
        const stats = new Float32Array(batch * m * 2);
        const rowInv = new Float32Array(batch * m);
        const rowMax = new Float32Array(batch * m);
        for (let row = 0; row < batch * m; row += 1) {
          let max = Number.NEGATIVE_INFINITY;
          for (let col = 0; col < n; col += 1) max = Math.max(max, rounded[row * n + col]);
          let sum = 0;
          for (let col = 0; col < n; col += 1) {
            sum = Math.fround(sum + Math.fround(Math.exp(rounded[row * n + col] - max)));
          }
          const inv = Math.fround(1 / sum);
          stats[row * 2] = max;
          stats[row * 2 + 1] = inv;
          rowInv[row] = inv;
          rowMax[row] = max;
        }
        const qp = await observeQpS16(gpu, shape, packScores(rounded), stats, rowInv);
        const expected = referenceAttentionPvQuant(rounded, rowMax, batch * m, n);
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
        // 恒真化の門: qP が両端に届いていなければ格子を何も検証していない
        assert([...qp].includes(P_ABS_MAX), `${shape.name}: qP に 127 が無い`);
        assert([...qp].includes(0), `${shape.name}: qP に 0 が無い`);
      }
    } finally {
      gpu.destroy();
    }
    // 実測の記録（ADR 0030 の門を s16 で再導出した値）。恒真化しないよう上限も置く
    const rate = mismatched / total;
    console.log(
      `qP（S は f16 格納）の GPU/TS 不一致率: ${mismatched} / ${total} = ${
        (rate * 100).toFixed(4)
      }%（全て ±1 段）`,
    );
    assert(rate < 0.02, `qP の不一致率 ${rate} が 2% を超えた（exp の実装差にしては大きい）`);
  },
});

// ---------------------------------------------------------------------------
// (7) NaN 行の伝播（格納形が変わっても非有限値の扱いが動かない）
// ---------------------------------------------------------------------------

Deno.test({
  name: "s16 でも NaN を含む行はそのまま NaN を伝播する（f32 格納と同じ行が NaN になる・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const shape: AttentionShape = { name: "NaN 行 B1 H2 M8 N8 D4", b: 1, h: 2, m: 8, n: 8, d: 4 };
    const { b, h, m, n, d } = shape;
    const q = fill([b, h, m, d], QUERY);
    const k = fill([b, h, n, d], KEY);
    const v = fill([b, h, n, d], VALUE);
    // head 1 の行 3 だけを NaN にする（S の 1 行が丸ごと NaN になる）
    (q.data as Float32Array)[(1 * m + 3) * d] = Number.NaN;
    const graph = singleOpGraph("attention", [q.shape, k.shape, v.shape], [b, h, m, d], {
      attrs: { scale: halfScale(d) },
    });
    const gpu = await acquireGpu();
    try {
      const run = async (options: SessionOptions): Promise<Float32Array> => {
        const session = await createSession(gpu, openModel(graphModelBuffer(graph)), options);
        try {
          return (await session.run({ x0: q, x1: k, x2: v }))["y"].data as Float32Array;
        } finally {
          await session.dispose();
        }
      };
      const s16 = await run({ attentionScoreStorage: "f16" });
      const plain = await run({});
      const nanRows = (data: Float32Array): readonly number[] =>
        Array.from({ length: b * h * m }, (_, row) => row).filter((row) =>
          [...data.slice(row * d, (row + 1) * d)].some(Number.isNaN)
        );
      // 汚染する行は f32 格納と完全に同じ（pack2x16float が NaN を 0 や有限値へ潰していない）
      assertEquals(nanRows(s16), nanRows(plain), "NaN 行の集合が f32 格納と違う");
      assertEquals(nanRows(s16), [1 * m + 3], "NaN が 1 行に閉じていない");
      // 汚染していない行は 1 つも NaN でない（行が全滅していたら比較が恒真になる）
      assert(
        [...s16].filter(Number.isNaN).length === d,
        "NaN が 1 行ぶんを超えて広がっている",
      );
    } finally {
      gpu.destroy();
    }
  },
});

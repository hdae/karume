// states 形実行の**受入テスト群**（波 D-5 — ADR 0066 受入条件② / ADR 0067 受入条件⑤ ほか）。
//
// gpu_state_execution_test.ts（実行結線）と gpu_state_attention_test.ts（カーネル単体）の門は、
// どちらも **D-2 製の CPU 参照**（src/reference/state-attention.ts）をオラクルにしている。参照と
// カーネルが同じ誤りを共有すれば全部が恒真になるので、本ファイルはその穴を塞ぐ 5 群を置く:
//
//   1. **交差オラクル** — 波 A 以前から信頼済みの機構（mask 第 4 入力つき 1-shot 融合 attention）
//      で同じ計算を組み、多 step の states 形と突き合わせる（参照に一切依存しない検証鎖）
//   2. **ADR 0066 受入条件②** — pad あり / なしで **KV スロットのバイト列が完全一致** + 出力一致
//   3. **ADR 0067 受入条件⑤** — 空行 → 厳密 0 を `Session.run` 経由で（executor 越しの直接門）
//   4. **容量非依存** — C=8 と C=4096 で全 step ビット同一（決定 3 の仕事量合格条件の値側の裏）
//   5. **KV 共有層**（ADR 0067 決定 4 — Gemma 4 E2B の 20 層パターン）の**実行**
//
// MUST: 各群は「門が空振りしていない」ことを自前で持つ（期待値の非自明性・対照・故障注入）。
// states 形は厳密 0 が正規に出る形なので、確認が無いと「何も計算していない」実装が緑で通る。

import { assert, assertEquals } from "@std/assert";
import { compareTensors, formatAllclose, type Tolerance } from "../src/reference/allclose.ts";
import { referenceStateAppend, referenceStateAttention } from "../src/reference/state-attention.ts";
import { acquireGpu, type GpuContext, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import { openModel } from "../src/format/container.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import type { GenerationContext } from "../src/runtime/generation-context.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { halfScale, seeded } from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** {@link compareTensors} の戻り（allclose.ts は型を輸出しないので実体から取る）。 */
type Report = ReturnType<typeof compareTensors>;

/** 決定的な入力列（乱数は使わない — 失敗が再現しないため）。 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** 1 モデルぶんの形（`B = 1` 固定 — batch 軸は ADR 0066 決定 8 のスコープ外）。 */
type StateModel = {
  readonly heads: number;
  readonly kvHeads: number;
  readonly depth: number;
  /** スロットの行容量。記号 `"C"` にすると `createGenerationContext(bindings)` が決める。 */
  readonly capacity: number | "C";
  /** `W`（省略 = 全 context）。 */
  readonly window?: number;
};

/**
 * states 形のグラフ（読者 `readers` 本 + `state_append` 2 本）。
 *
 * ノード配列順は ADR 0067 決定 5b の発行規約どおり「**当該スロットの全読者 → 書き**」。
 * `readers = 2` が KV 共有層（所有層 + 自前 KV を持たない層）の形で、共有層は自層で projection を
 * 計算せず**所有層の k/v 値をそのまま ins に配線**する（決定 4）— それがこのグラフの `k` / `v` を
 * 2 本の attention が共有している形そのもので、共有は「同一スロット名 + 同一値名」で表れる。
 */
const stateGraph = (model: StateModel, readers = 1): GraphJson => {
  const { heads, kvHeads, depth, capacity, window } = model;
  const windowAttrs = window === undefined ? {} : { window };
  const names = Array.from({ length: readers }, (_, index) => index);
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["attention", "state_append"] },
    symbols: capacity === "C" ? ["M", "C"] : ["M"],
    inputs: [
      ...names.map((index) => ({
        name: `q${index}`,
        dtype: "f32",
        shape: [1, heads, "M", depth] as (number | string)[],
      })),
      { name: "k", dtype: "f32", shape: [1, kvHeads, "M", depth] },
      { name: "v", dtype: "f32", shape: [1, kvHeads, "M", depth] },
    ],
    outputs: names.map((index) => `o${index}`),
    initializers: {},
    values: Object.fromEntries(
      names.map((index) => [`o${index}`, { dtype: "f32", shape: [1, heads, "M", depth] }]),
    ),
    states: {
      kslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
      vslot: { dtype: "f32", shape: [1, kvHeads, capacity, depth] },
    },
    nodes: [
      ...names.map((index) => ({
        op: "attention",
        ins: [`q${index}`, "k", "v"],
        outs: [`o${index}`],
        attrs: { scale: halfScale(depth), ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      })),
      {
        op: "state_append",
        ins: ["k"],
        outs: [],
        attrs: { ...windowAttrs },
        states: { slot: "kslot" },
      },
      {
        op: "state_append",
        ins: ["v"],
        outs: [],
        attrs: { ...windowAttrs },
        states: { slot: "vslot" },
      },
    ],
  };
};

const stateSession = (
  gpu: GpuContext,
  model: StateModel,
  readers = 1,
): Promise<Session> => createSession(gpu, openModel(graphModelBuffer(stateGraph(model, readers))));

const tensor = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** 1 step ぶんの入力（`M` 行ぶん — 有効なのは先頭 `queryLength` 行）。 */
type StepInputs = {
  /** 読者ごとの q（KV 共有層は層ごとに別の q を持つ）。 */
  readonly q: readonly Float32Array<ArrayBuffer>[];
  readonly k: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
};

/**
 * step ごとの入力。`salt` を step で変えることで「前 step の入力を読み直している」実装が値で
 * 落ちるようにする（同じ入力を配ると鎖の誤りが自己相殺しうる）。読者ごとにも salt をずらす
 * （2 本の読者が同じ q を持つと、共有層の配線ミスが値に出ない）。
 */
const stepInputs = (
  model: StateModel,
  chunkRows: number,
  salt: number,
  readers = 1,
): StepInputs => ({
  q: Array.from(
    { length: readers },
    (_, index) =>
      seeded(model.heads * chunkRows * model.depth, (i) => QUERY(i + salt + index * 97)),
  ),
  k: seeded(model.kvHeads * chunkRows * model.depth, (i) => KEY(i + salt)),
  v: seeded(model.kvHeads * chunkRows * model.depth, (i) => VALUE(i + salt)),
});

/** GPU 側の 1 step（`run` の第 3 引数が generation 面）。返すのは読者ごとの `[1,H,M,D]`。 */
const runStep = async (
  session: Session,
  context: GenerationContext,
  model: StateModel,
  inputs: StepInputs,
  chunkRows: number,
  query: number,
): Promise<readonly Float32Array<ArrayBuffer>[]> => {
  const tensors: Record<string, Tensor> = {
    k: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.k),
    v: tensor([1, model.kvHeads, chunkRows, model.depth], inputs.v),
  };
  inputs.q.forEach((data, index) => {
    tensors[`q${index}`] = tensor([1, model.heads, chunkRows, model.depth], data);
  });
  const outputs = await session.run(tensors, {}, { context, queryLength: query });
  return inputs.q.map((_, index) => {
    const out = outputs[`o${index}`];
    assertEquals(out.dtype, "f32");
    return out.data as Float32Array<ArrayBuffer>;
  });
};

/** `[1,H,M,D]` の先頭 `rows` 行を抜き出す（有効行 = 論理座標 `[P, P+Q)` に対応する行）。 */
const takeRows = (
  data: Float32Array<ArrayBuffer>,
  planes: number,
  chunkRows: number,
  depth: number,
  rows: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(planes * rows * depth);
  for (let plane = 0; plane < planes; plane += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let d = 0; d < depth; d += 1) {
        out[(plane * rows + row) * depth + d] = data[(plane * chunkRows + row) * depth + d];
      }
    }
  }
  return out;
};

const bitsOf = (data: Float32Array<ArrayBuffer>): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

/** ビット列の食い違い（先頭 4 件だけ）。 */
const mismatches = (actual: ArrayLike<number>, expected: ArrayLike<number>): readonly string[] => {
  const found: string[] = [];
  for (let i = 0; i < actual.length && found.length < 4; i += 1) {
    if (actual[i] !== expected[i]) found.push(`[${i}] ${actual[i]} vs ${expected[i]}`);
  }
  return found;
};

const assertBitsEqual = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: 長さが違う`);
  assertEquals(mismatches(actual, expected), [], `${where}: ビット列が一致しない`);
};

/**
 * state スロットの**生バイト列**を読み戻す（ADR 0066 受入条件②の突合相手）。
 *
 * 実行経路はスロットの内容を 1 度も出力に出さないので、`RUNTIME_INTERNAL` のバッファを直接
 * copyBufferToBuffer + mapAsync で覗く以外に観測点が無い（helpers/state-dispatch.ts の
 * `readFloats` と同じ流儀）。返すのは **`Uint32Array`** — 「バイト列が完全一致」は f32 の値比較
 * では表せない（`-0` と `+0`・NaN の payload が値比較では潰れる）。
 *
 * MUST: 呼ぶのは `run` の解決後（`Session.run` は出力を読み戻して決着するので、その時点で
 * 当該 run の dispatch は全て submit 済み）。
 */
const readSlotBits = async (
  gpu: GpuContext,
  context: GenerationContext,
  slot: string,
): Promise<Uint32Array> => {
  const backing = context[RUNTIME_INTERNAL].slots.get(slot);
  if (backing === undefined) throw new Error(`state スロット '${slot}' が無い`);
  const staging = gpu.device.createBuffer({
    size: backing.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(backing.buffer, 0, staging, 0, backing.byteLength);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    return new Uint32Array(copy);
  } finally {
    staging.destroy();
  }
};

// ---------------------------------------------------------------------------
// 群 1: 交差オラクル（帯 mask つき 1-shot 融合 attention との多 step 突合）
// ---------------------------------------------------------------------------

/**
 * 帯 mask の「遮蔽」値。
 *
 * MUST: **有限**の大負値（`-inf` を入れない）。融合 attention の mask は加算型で safe_softmax
 * ではない（ADR 0044 決定 3 / names.ts の ATTENTION_OP 契約）ため、`-inf` を入れると
 * 「行 max も `-inf` になる行」が 1 本でも生じた瞬間に `exp(-inf − (-inf)) = NaN` へ落ちる形が
 * 実装の分岐次第で生まれる。有限値なら加算も減算も f32 の有限域に閉じたまま、
 * `exp(-3e38 − max)` は f32 の指数下限（≈ −88）を 36 桁下回るので**厳密に 0** へ潰れる。
 * `-3e38` は f32 の最大有限値 ≈ 3.4028e38 の内側なので、`S + mask` が `-inf` へ溢れることもない。
 *
 * NOTE: どの行も causal 対角（`col = P + row`）を必ず 1 本含むので「全 −inf 行」は構造的に
 * 生じない — この mask は融合 attention の契約（全 −inf 行は契約違反）を破らない。
 */
const MASK_BLOCK = -3e38;

/**
 * 論理座標の帯 mask `[1,1,Q,Ntot]`（causal + sliding の**両側**述語 — ADR 0067 決定 4）。
 *
 * 行 `i` は論理位置 `P + i`。許可は `max(0, P+i−W+1) ≤ col ≤ P+i`（`W = 0` は下限なし）。
 */
const bandMask = (
  past: number,
  query: number,
  total: number,
  window: number,
): Float32Array<ArrayBuffer> => {
  const mask = new Float32Array(query * total);
  for (let row = 0; row < query; row += 1) {
    const limit = past + row;
    const lower = window > 0 ? Math.max(0, limit - window + 1) : 0;
    for (let col = 0; col < total; col += 1) {
      mask[row * total + col] = col >= lower && col <= limit ? 0 : MASK_BLOCK;
    }
  }
  return mask;
};

/**
 * 論理 KV 履歴（1 行 = `[Hkv·D]`）。states 側のスロットとは**別の器**で持つ — 容量も ring も
 * 持たない素の追記列なので、写像（`col % W`）の誤りが交差突合に出る。
 */
type History = Float32Array[];

const appendHistory = (
  history: History,
  data: Float32Array<ArrayBuffer>,
  model: StateModel,
  chunkRows: number,
  query: number,
): void => {
  const { kvHeads, depth } = model;
  for (let row = 0; row < query; row += 1) {
    const logical = new Float32Array(kvHeads * depth);
    for (let kv = 0; kv < kvHeads; kv += 1) {
      for (let d = 0; d < depth; d += 1) {
        logical[kv * depth + d] = data[(kv * chunkRows + row) * depth + d];
      }
    }
    history.push(logical);
  }
};

/**
 * 履歴を `[1,H,N,D]` へ**実体化**する（repeat_kv — GQA 系列の 1-shot 側は整除 broadcast を
 * 使わない形で組む。ADR 0067 決定 2 の受入条件と同じ流儀で、1-shot 側が波 A の変更に依存しない）。
 */
const materializeHistory = (history: History, model: StateModel): Float32Array<ArrayBuffer> => {
  const { heads, kvHeads, depth } = model;
  const repeat = heads / kvHeads;
  const total = history.length;
  const out = new Float32Array(heads * total * depth);
  for (let head = 0; head < heads; head += 1) {
    const kv = Math.floor(head / repeat);
    for (let n = 0; n < total; n += 1) {
      for (let d = 0; d < depth; d += 1) {
        out[(head * total + n) * depth + d] = history[n][kv * depth + d];
      }
    }
  }
  return out;
};

/** 1-shot 側の Session（`M` / `N` を記号にして全 step を 1 本で賄う）。 */
const oneShotSession = (gpu: GpuContext, model: StateModel): Promise<Session> => {
  const { heads, depth } = model;
  const graph = singleOpGraph(
    "attention",
    [[1, heads, "M", depth], [1, heads, "N", depth], [1, heads, "N", depth], [1, 1, "M", "N"]],
    [[1, heads, "M", depth]],
    { attrs: { scale: halfScale(depth) }, symbols: ["M", "N"] },
  );
  return createSession(gpu, openModel(graphModelBuffer(graph)));
};

/**
 * 交差オラクルの許容誤差。
 *
 * 両経路は**別族のカーネル**（states 形は state_qk / state_stats / state_pv の 3 本 — census が
 * 「融合 attention のキーが 1 本も出ない」ことを別途固定している / 1-shot は融合 attention）で、
 * 縮約の分け方も読み出し元も違う（過去は ring 物理順 + current は ins ／ 連結済みの論理順）ため、
 * 一般には丸め列が一致する保証は無い。よって allclose で判定する。
 *
 * MUST: `rtol = 0`（states 形は厳密 0 が正解の要素を正規に含み、相対項は 0 近傍で効かない）。
 * MUST: この幅は「窓外 key 1 本の混入」を必ず検出できること — 故障注入（window +1 のグラフ）で
 * 実証する。実測の効きは 0.054〜2.12 で、この幅の 1.8e4〜7e5 倍。
 *
 * NOTE: 幅の値そのものは**余裕でしかない**。参照機の実測は 4 系列 20 step すべて
 * `maxAbs = 0`（= 両経路がビット同一）で、丸め差は 1 度も観測されていない。「縮約順が違うから
 * 誤差が出るはず」という着手時の想定は実測に反証された — 遮蔽列の寄与が厳密 0
 * （`exp(-3e38 − max)` も `0 · v` も）で、有効列の並びと D 方向の縮約順が両族で一致しているため。
 * MUST: その「ビット同一」を**門に焼かない**（波 D-7 で `worstAbs <= 0` の assert を撤去し、
 * 実測最悪値は stdout へ出すだけにした）。fma の使い方はドライバ依存で、別機では正常に丸め差が
 * 出る — 機材依存のビット挙動を可搬な門にすると、正しい実装がその機でだけ赤くなる
 * （golden io のバイト突合を参照環境専用に分けた 2026-08-16 の裁定と同じ教訓）。判定は
 * {@link CROSS_TOLERANCE} + 故障注入の検出力マージンに一本化する。
 */
const CROSS_TOLERANCE: Tolerance = { atol: 3e-6, rtol: 0 };

type CrossSeries = {
  readonly name: string;
  readonly model: StateModel;
  readonly chunkLength: number;
  /** `[M（物理 chunk 行数）, Q（queryLength）]` の列。 */
  readonly steps: readonly (readonly [number, number])[];
};

const CROSS_SERIES: readonly CrossSeries[] = [
  {
    // 論理長 0 → 4 → 5 → 6 → 7 → 9（末尾は pad つき chunk）
    name: "full r=1",
    model: { heads: 4, kvHeads: 4, depth: 4, capacity: 16 },
    chunkLength: 4,
    steps: [[4, 4], [1, 1], [1, 1], [1, 1], [4, 2]],
  },
  {
    // pad つき prefill から始める GQA（0 → 3 → 4 → 5 → 9）
    name: "full r=2 (GQA)",
    model: { heads: 4, kvHeads: 2, depth: 8, capacity: 16 },
    chunkLength: 4,
    steps: [[4, 3], [1, 1], [1, 1], [4, 4]],
  },
  {
    // 0 → 3 → 4 → 5 → 6 → 7 → 9。W=4 の ring は past=4 で一周し、以後 毎 step エビクト
    name: "sliding W=4 r=1（W 跨ぎエビクション）",
    model: { heads: 2, kvHeads: 2, depth: 4, capacity: 4, window: 4 },
    chunkLength: 3,
    steps: [[3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [3, 2]],
  },
  {
    // 0 → 4 → 5 → 6 → 7 → 10。エビクション × GQA × 末尾 pad つき chunk
    name: "sliding W=6 r=2（GQA × エビクション）",
    model: { heads: 4, kvHeads: 2, depth: 4, capacity: 6, window: 6 },
    chunkLength: 4,
    steps: [[4, 4], [1, 1], [1, 1], [1, 1], [4, 3]],
  },
];

/**
 * 1 系列を回して step ごとの突合レポートを返す。
 *
 * `windowDelta` は**故障注入**の口 — states 側だけ窓を広げてグラフを組み直す（1-shot 側の期待値は
 * 常に本来の窓で作る）。窓が 1 広い実装は「窓外 key 1 本を沈黙混入」させる形そのもの
 * （ADR 0067 決定 4 が両側述語を MUST にした理由）。
 */
const crossSeriesReports = async (
  gpu: GpuContext,
  series: CrossSeries,
  windowDelta = 0,
): Promise<readonly Report[]> => {
  const { model, steps } = series;
  const capacity = model.capacity;
  if (windowDelta !== 0 && (model.window === undefined || typeof capacity !== "number")) {
    throw new Error("故障注入（窓を広げる）は具体容量の sliding 系列にしか適用できない");
  }
  const stateModel: StateModel = windowDelta === 0 ? model : {
    ...model,
    window: (model.window ?? 0) + windowDelta,
    capacity: (capacity as number) + windowDelta,
  };
  const states = await stateSession(gpu, stateModel);
  const oneShot = await oneShotSession(gpu, model);
  const context = await states.createGenerationContext({ chunkLength: series.chunkLength });
  const kHistory: History = [];
  const vHistory: History = [];
  const reports: Report[] = [];
  let past = 0;
  try {
    for (const [index, [chunkRows, query]] of steps.entries()) {
      const inputs = stepInputs(model, chunkRows, 7 + index * 29);
      const actual = (await runStep(states, context, model, inputs, chunkRows, query))[0];
      appendHistory(kHistory, inputs.k, model, chunkRows, query);
      appendHistory(vHistory, inputs.v, model, chunkRows, query);
      const total = past + query;
      const expected = (await oneShot.run({
        x0: tensor(
          [1, model.heads, query, model.depth],
          takeRows(inputs.q[0], model.heads, chunkRows, model.depth, query),
        ),
        x1: tensor([1, model.heads, total, model.depth], materializeHistory(kHistory, model)),
        x2: tensor([1, model.heads, total, model.depth], materializeHistory(vHistory, model)),
        x3: tensor([1, 1, query, total], bandMask(past, query, total, model.window ?? 0)),
      }))["y"].data as Float32Array<ArrayBuffer>;
      // MUST: 期待値が自明でないこと（両側が ~0 なら突合は恒真）。
      assert(
        expected.some((value) => Math.abs(value) > 1e-3),
        `${series.name} step ${index}: 1-shot 側の期待出力が自明（全 ~0）`,
      );
      reports.push(compareTensors(
        {
          dtype: "f32",
          data: takeRows(actual, model.heads, chunkRows, model.depth, query),
        },
        { dtype: "f32", data: expected },
        CROSS_TOLERANCE,
      ));
      past += query;
    }
    assertEquals(context.pastLength, past, `${series.name}: 論理長がホスト側の進行とずれた`);
  } finally {
    await context.dispose();
    await states.dispose();
    await oneShot.dispose();
  }
  return reports;
};

Deno.test({
  name:
    "states 形の多 step が帯 mask つき 1-shot 融合 attention と一致する（交差オラクル・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    let worstAbs = 0;
    try {
      for (const series of CROSS_SERIES) {
        const reports = await crossSeriesReports(gpu, series);
        reports.forEach((report, index) => {
          assertEquals(
            report.pass,
            true,
            `${series.name} step ${index}: ${formatAllclose(report)}`,
          );
          worstAbs = Math.max(worstAbs, report.maxAbsError);
        });
      }
    } finally {
      gpu.destroy();
    }
    // 実測最悪値は**診断として出すだけ**（判定は CROSS_TOLERANCE 側 — 上の doc）。参照機では
    // 0（両族ビット同一）が出るが、fma の使い方はドライバ依存なので他機の非ゼロは正常。
    console.log(`[states cross-oracle] maxAbs=${worstAbs}`);
  },
});

Deno.test({
  name: "窓を 1 広げた states 形は交差オラクルと一致しない（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // W=4 の系列を W=5（容量も 5）で組み直す。論理長の進行は 0 → 3 → 4 → 5 → 6 → 7 なので、
      // **step 2 以降**（`past + row ≥ W` になる step）だけが「resident な窓外 key を 1 本余計に
      // 見る」形になり、正しい帯 mask の期待値から外れる。
      const reports = await crossSeriesReports(gpu, CROSS_SERIES[2], 1);
      const failing = reports.map((report) => report.pass);
      assertEquals(
        failing,
        [true, true, false, false, false, false],
        `窓 +1 の実装が検出されない step がある（突合が空振り）: ${
          reports.map(formatAllclose).join(" / ")
        }`,
      );
      // 検出力の余裕: 混入 1 本の効きが許容幅の 3 桁以上（実測 0.054〜2.12 — 幅を 2 桁緩めても
      // 残る）。前半 2 step が緑なのは「何でも赤くする注入ではない」ことの裏。
      for (const [index, report] of reports.slice(2).entries()) {
        assert(
          report.maxAbsError > CROSS_TOLERANCE.atol * 1000,
          `step ${
            index + 2
          }: 窓外 key 混入の効き ${report.maxAbsError} が許容幅 ${CROSS_TOLERANCE.atol} に近すぎる`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 群 2: ADR 0066 受入条件② — padding 行 no-op の機械証明
// ---------------------------------------------------------------------------

const PAD_CAPACITY = 8;
const PAD_MODEL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: PAD_CAPACITY };
/** pad ありの物理 chunk 行数（`M = 8` — 有効は先頭 Q 行だけ）。 */
const PAD_WIDE_ROWS = 8;
/** pad なし側の prefill 形の物理 chunk 行数（`M = Q = 3`）。 */
const PAD_TIGHT_ROWS = 3;

/**
 * `M` 行の入力を作る（先頭 `query` 行は「論理内容」・残りは pad）。
 *
 * `padFill` が `undefined` なら pad 行は **0 埋め**（ADR 0066 追記 6 のホスト側の値契約）。
 * 値を渡すと pad 行に非ゼロのゴミが入る — スロットが pad 行を読まないことの直接証明用。
 */
const paddedInputs = (
  model: StateModel,
  chunkRows: number,
  query: number,
  salt: number,
  padFill?: (index: number) => number,
): StepInputs => {
  const build = (planes: number, generator: (i: number) => number): Float32Array<ArrayBuffer> =>
    seeded(planes * chunkRows * model.depth, (i) => {
      const row = Math.floor(i / model.depth) % chunkRows;
      if (row < query) {
        return generator(
          (Math.floor(i / (chunkRows * model.depth)) * query + row) * model.depth +
            (i % model.depth) + salt,
        );
      }
      return padFill === undefined ? 0 : padFill(i);
    });
  return {
    q: [build(model.heads, QUERY)],
    k: build(model.kvHeads, KEY),
    v: build(model.kvHeads, VALUE),
  };
};

/** pad なし（`M = query`）の同一論理内容。 */
const tightInputs = (model: StateModel, query: number, salt: number): StepInputs =>
  paddedInputs(model, query, query, salt);

Deno.test({
  name: "pad 行は KV スロットのバイト列にも出力にも現れない（ADR 0066 受入条件②・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, PAD_MODEL);
    // 3 本の context を同じ Session で回す。pad の有無は物理 chunk 行数 M の違いなので、
    // 固定 chunk 契約（`M ∈ {chunkLength, 1}` — ADR 0066 決定 4）の下では **chunkLength も
    // 別**になる（`tight` は M=Q=3 の prefill 形 = chunkLength 3）。主張（KV バイト同一）は
    // 同じ Session を跨いだ context 間の比較なので、そのまま成立する。
    const [wide, tight, garbage] = await Promise.all([
      session.createGenerationContext({ chunkLength: PAD_WIDE_ROWS }),
      session.createGenerationContext({ chunkLength: PAD_TIGHT_ROWS }),
      session.createGenerationContext({ chunkLength: PAD_WIDE_ROWS }),
    ]);
    try {
      // 2 step 回す（2 本目は過去ありの形 — 論理長が進んだ後も同じ 2 点が成り立つこと）。
      // 1 本目は prefill 形（tight は M=Q=3）・2 本目は decode 形（tight は M=Q=1）で、
      // どちらの実行形でも同じ 2 点が成り立つことを併せて見る。
      for (const [index, query] of [PAD_TIGHT_ROWS, 1].entries()) {
        const salt = 11 + index * 37;
        const wideOut = (await runStep(
          session,
          wide,
          PAD_MODEL,
          paddedInputs(PAD_MODEL, PAD_WIDE_ROWS, query, salt),
          PAD_WIDE_ROWS,
          query,
        ))[0];
        const tightOut = (await runStep(
          session,
          tight,
          PAD_MODEL,
          tightInputs(PAD_MODEL, query, salt),
          query,
          query,
        ))[0];
        // NOTE: pad 行の 0 埋めは**ホスト側の値契約**（ADR 0066 追記 6 — pad 行の k/v が非有限だと
        // 「−inf 加算 + exp」経路で valid 行へ NaN が漏れるのを値で遮断する）。KV スロットには
        // そもそも pad 行が書かれないので、この契約は KV には効いていない — それを示すのが下の
        // `garbage`（pad 行に非ゼロを詰めても KV のバイト列は 1 ビットも動かない）。
        const garbageOut = (await runStep(
          session,
          garbage,
          PAD_MODEL,
          paddedInputs(PAD_MODEL, PAD_WIDE_ROWS, query, salt, (i) => (((i * 13) % 29) - 14) * 0.41),
          PAD_WIDE_ROWS,
          query,
        ))[0];

        const label = `step ${index}（Q=${query}）`;
        // ① KV スロットのバイト列が完全一致（pad の有無・pad の中身に依らない）
        for (const slot of ["kslot", "vslot"]) {
          const expected = await readSlotBits(gpu, tight, slot);
          assertBitsEqual(await readSlotBits(gpu, wide, slot), expected, `${label}: ${slot} / pad`);
          assertBitsEqual(
            await readSlotBits(gpu, garbage, slot),
            expected,
            `${label}: ${slot} / pad にゴミ`,
          );
          // MUST: 突合が空振りでないこと — 書かれた行は非ゼロで、`pastLength` 以降は 0 のまま
          // （`state_append` が先頭 queryLength 行しか書かない = 受入条件②の直接の言い換え。
          // full スロットなので論理 col = 物理 row で、境界がそのまま `pastLength` に出る）。
          const past = tight.pastLength;
          const rowFloats = PAD_MODEL.depth;
          for (let plane = 0; plane < PAD_MODEL.kvHeads; plane += 1) {
            const base = plane * PAD_CAPACITY * rowFloats;
            const written = expected.subarray(base, base + past * rowFloats);
            const beyond = expected.subarray(
              base + past * rowFloats,
              base + PAD_CAPACITY * rowFloats,
            );
            assert(
              written.some((bits) => bits !== 0),
              `${label}: ${slot} 面 ${plane} の書込み範囲が全 0（空振り）`,
            );
            assertEquals(
              beyond.every((bits) => bits === 0),
              true,
              `${label}: ${slot} 面 ${plane} の論理長より先（行 ${past}..${
                PAD_CAPACITY - 1
              }）に書込みがある`,
            );
          }
        }
        // ② 有効出力行 [0,Q) が Uint32 完全一致
        const valid = (data: Float32Array<ArrayBuffer>, rows: number): Uint32Array =>
          bitsOf(takeRows(data, PAD_MODEL.heads, rows, PAD_MODEL.depth, query));
        const tightBits = valid(tightOut, query);
        assert(new Set(tightBits).size > 1, `${label}: 出力が定数（ビット一致が恒真）`);
        assertBitsEqual(valid(wideOut, PAD_WIDE_ROWS), tightBits, `${label}: 出力 / pad`);
        // 行局所性（追記 6）— pad 行のゴミは valid 行へ 1 ビットも漏れない。
        assertBitsEqual(valid(garbageOut, PAD_WIDE_ROWS), tightBits, `${label}: 出力 / pad にゴミ`);
        assertEquals(wide.pastLength, tight.pastLength, `${label}: 論理長が pad の有無で違う`);
      }
      assertEquals(tight.pastLength, PAD_TIGHT_ROWS + 1);
    } finally {
      for (const context of [wide, tight, garbage]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 群 3: ADR 0067 受入条件⑤ — 空行 → 0 の直接門（executor 経由）
// ---------------------------------------------------------------------------

const EMPTY_MODEL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: 2, window: 2 };

/**
 * 行 `row` が**空行**か（述語を満たす実在 col が 1 本も無い）。
 *
 * MUST: カーネル側のヘルパ（`stateColumnBase` / `stateLiveColumns`）を使わず論理座標だけで書く —
 * 実装の列範囲の取り方が誤っていても、この判定は誤りに追随しない。実在する col は `[0, P+Q)`
 * （過去 ∪ 今 step）で、許可は `max(0, P+row−W+1) ≤ col ≤ P+row`。
 */
const isEmptyRow = (past: number, query: number, window: number, row: number): boolean => {
  const limit = past + row;
  const lower = window > 0 ? Math.max(0, limit - window + 1) : 0;
  return Math.min(limit, past + query - 1) < lower;
};

Deno.test({
  name: "pad 行が窓から落ちた出力行は厳密 0（ADR 0067 受入条件⑤・Session.run 経由・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, EMPTY_MODEL);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const { heads, depth, window } = EMPTY_MODEL;
    const chunkRows = 4;
    try {
      // W=2 に対し M=4 の chunk を Q=1 で回す。row 2 / 3 は causal 上限が届く実在 col が
      // どれも窓の外（`col + W ≤ P + row`）に落ちるため、正規に空行になる。
      for (const [index, past] of [0, 1, 2].entries()) {
        assertEquals(context.pastLength, past, `step ${index}: 前提の論理長がずれた`);
        const out = (await runStep(
          session,
          context,
          EMPTY_MODEL,
          stepInputs(EMPTY_MODEL, chunkRows, 5 + index * 19),
          chunkRows,
          1,
        ))[0];
        const empties = Array.from({ length: chunkRows }, (_, row) => row)
          .filter((row) => isEmptyRow(past, 1, window ?? 0, row));
        // MUST: 空行が実在すること（この形で 0 本なら門ごと空振り）。
        assertEquals(empties, [2, 3], `step ${index}: 空行の想定が崩れている`);
        // MUST: 空行は pad 行（`row ≥ Q`）の**部分集合**（Q=1 なので pad は 1,2,3）。波 D-7 で
        // 「空行 → 厳密 0」は「pad 行 → 厳密 0」が構造的に包含する形になったので、包含が崩れて
        // いないことも併せて見る（valid 行は causal 自己参照で必ず非空）。
        assertEquals(empties.filter((row) => row < 1), [], `step ${index}: 空行が valid 行に出た`);
        for (let head = 0; head < heads; head += 1) {
          // pad 行（空行を含む）は厳密 0（ADR 0067 受入条件⑤ / ADR 0066 追記 6 の値契約）。
          for (let row = 1; row < chunkRows; row += 1) {
            for (let d = 0; d < depth; d += 1) {
              const value = out[(head * chunkRows + row) * depth + d];
              assertEquals(
                Object.is(value, 0),
                true,
                `step ${index}: pad 行 (head ${head}, row ${row}, d ${d}) が厳密 0 でない（${value}）`,
              );
            }
          }
          // 対照: **valid 行**は非ゼロ（「全部 0 を書く実装」がこの門を通らないことの裏）。
          const slice = out.subarray(head * chunkRows * depth, (head * chunkRows + 1) * depth);
          assert(
            slice.some((value) => Math.abs(value) > 1e-3),
            `step ${index}: valid 行 (head ${head}, row 0) まで ~0`,
          );
        }
        // 全出力に NaN が 1 つも無い（空行の構成が「0 ガード」で、`exp(-inf − (-inf))` へ
        // 落ちていないことの直接の裏 — ADR 0067 決定 6）。
        assertEquals(
          out.some((value) => Number.isNaN(value)),
          false,
          `step ${index}: 出力に NaN が居る`,
        );
      }
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 群 4: 容量非依存（ADR 0066 決定 3 — 仕事量が論理長のみで決まることの値側）
// ---------------------------------------------------------------------------

const SYMBOLIC_MODEL: StateModel = { heads: 2, kvHeads: 2, depth: 4, capacity: "C" };

Deno.test({
  name: "容量 C=8 と C=4096 で全 step の出力がビット同一（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, SYMBOLIC_MODEL);
    const small = await session.createGenerationContext({ chunkLength: 4, bindings: { C: 8 } });
    const large = await session.createGenerationContext({ chunkLength: 4, bindings: { C: 4096 } });
    try {
      // MUST: 容量が**実際に効いている**こと（束縛が黙って無視されていれば 2 本は同じ context で、
      // ビット同一は恒真になる）。容量は計画鍵そのもの（ADR 0066 決定 3）なので、ここが違えば
      // 2 本は別レシピ・別 S ストライド・別確保サイズで走っている。
      assertEquals(small[RUNTIME_INTERNAL].slots.get("kslot")?.shape, [1, 2, 8, 4]);
      assertEquals(large[RUNTIME_INTERNAL].slots.get("kslot")?.shape, [1, 2, 4096, 4]);

      // prefill（M=4 / Q=4）→ decode ×3。容量は S の列ストライドと確保サイズにしか効かない
      // ので、同じ論理長の同じ入力なら**丸め列まで含めて**同じでなければならない。
      for (
        const [index, [chunkRows, query]] of ([[4, 4], [1, 1], [1, 1], [1, 1]] as const).entries()
      ) {
        const inputs = stepInputs(SYMBOLIC_MODEL, chunkRows, 3 + index * 41);
        const a = (await runStep(session, small, SYMBOLIC_MODEL, inputs, chunkRows, query))[0];
        const b = (await runStep(session, large, SYMBOLIC_MODEL, inputs, chunkRows, query))[0];
        const bits = bitsOf(a);
        assert(new Set(bits).size > 1, `step ${index}: 出力が定数（ビット一致が恒真）`);
        assertBitsEqual(bitsOf(b), bits, `step ${index}: 容量で出力が変わった`);
      }
      assertEquals(small.pastLength, 7);
      assertEquals(large.pastLength, 7);
    } finally {
      for (const context of [small, large]) await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
});

// ---------------------------------------------------------------------------
// 群 5: KV 共有層の実行（ADR 0067 決定 4 — Gemma 4 E2B の 20 層パターン）
// ---------------------------------------------------------------------------

/**
 * 共有層の突合に使う許容誤差（オラクルは D-2 の CPU 参照鎖 — この群が見るのは**実行**で、
 * 参照の独立性は群 1 が別途担保している）。根拠は gpu_state_execution_test.ts の
 * `STATE_TOLERANCE` と同じ（参照は f64 積算・GPU は f32 逐次累積）。
 */
const SHARED_TOLERANCE: Tolerance = { atol: 5e-6, rtol: 0 };

const asF32 = (data: Float32Array | Int32Array | Uint32Array): Float32Array<ArrayBuffer> => {
  if (!(data instanceof Float32Array)) throw new Error("参照の戻りが f32 でない");
  return data as Float32Array<ArrayBuffer>;
};

type SharedSeries = {
  readonly name: string;
  readonly model: StateModel;
  readonly capacity: number;
  readonly chunkLength: number;
  readonly steps: readonly (readonly [number, number])[];
  /**
   * 「append が読者より先に走った」オラクルとの差を見る step（省略 = 見ない）。ring の wrap が
   * 今 step の読者の過去行を潰す形でのみ差が出る（ADR 0067 決定 5b の順序が値に効く唯一の窓）。
   */
  readonly orderProbeStep?: number;
};

const SHARED_SERIES: readonly SharedSeries[] = [
  {
    name: "full",
    model: { heads: 2, kvHeads: 2, depth: 4, capacity: 16 },
    capacity: 16,
    chunkLength: 3,
    steps: [[3, 3], [3, 3], [3, 3], [1, 1]],
  },
  {
    // W=4 / C=4 の ring。step 1（past=3 / Q=3）の append は物理 3,0,1 を書くので、読者が先に
    // 読む物理 0,1,2 のうち 2 行が潰れる = 順序違反が値に出る位置。
    name: "sliding W=4",
    model: { heads: 2, kvHeads: 2, depth: 4, capacity: 4, window: 4 },
    capacity: 4,
    chunkLength: 3,
    steps: [[3, 3], [3, 3], [3, 3], [1, 1]],
    orderProbeStep: 1,
  },
];

Deno.test({
  name: "KV 共有層（読者 2 本 → append 1 本）が実行できる（ADR 0067 決定 4 / 5b・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const series of SHARED_SERIES) {
        const { model, capacity } = series;
        const session = await stateSession(gpu, model, 2);
        const context = await session.createGenerationContext({ chunkLength: series.chunkLength });
        let slotK = new Float32Array(model.kvHeads * capacity * model.depth);
        let slotV = new Float32Array(model.kvHeads * capacity * model.depth);
        let past = 0;
        try {
          for (const [index, [chunkRows, query]] of series.steps.entries()) {
            const inputs = stepInputs(model, chunkRows, 17 + index * 23, 2);
            const outs = await runStep(session, context, model, inputs, chunkRows, query);
            const common = {
              chunkRows,
              depth: model.depth,
              capacity,
              window: model.window ?? 0,
              past,
              query,
            };
            const read = (
              q: Float32Array<ArrayBuffer>,
              k: Float32Array<ArrayBuffer>,
              v: Float32Array<ArrayBuffer>,
            ): Float32Array<ArrayBuffer> =>
              asF32(
                referenceStateAttention({
                  ...common,
                  batch: 1,
                  heads: model.heads,
                  kvHeads: model.kvHeads,
                  q,
                  insK: inputs.k,
                  insV: inputs.v,
                  slotK: k,
                  slotV: v,
                  scale: halfScale(model.depth),
                }).data,
              );

            // MUST: **両読者とも append の前の**スロットを読む（決定 5b の発行規約）。
            const expected = inputs.q.map((q) => read(q, slotK, slotV));
            outs.forEach((actual, layer) => {
              const label = `${series.name} step ${index} layer ${layer}`;
              assert(
                expected[layer].some((value) => Math.abs(value) > 1e-3),
                `${label}: 期待出力が自明（全 ~0）`,
              );
              const report = compareTensors(
                { dtype: "f32", data: actual },
                { dtype: "f32", data: expected[layer] },
                SHARED_TOLERANCE,
              );
              assertEquals(report.pass, true, `${label}: ${formatAllclose(report)}`);
            });
            // 所有層と共有層は**同じ KV・違う q** なので出力は違う（2 本目が 1 本目の複製に
            // なっている形＝配線の取り違えを、ここで落とす）。
            assertEquals(
              mismatches(bitsOf(outs[0]), bitsOf(outs[1])).length > 0,
              true,
              `${series.name} step ${index}: 2 層の出力がビット一致（共有層が所有層の複製）`,
            );

            // 順序の検出力: 「append を読者より先に走らせた」オラクルとは一致しないこと。
            if (series.orderProbeStep === index) {
              const appendBase = { ...common, kvPlanes: model.kvHeads };
              const wrongK = asF32(
                referenceStateAppend({ ...appendBase, x: inputs.k, slot: slotK }).data,
              );
              const wrongV = asF32(
                referenceStateAppend({ ...appendBase, x: inputs.v, slot: slotV }).data,
              );
              outs.forEach((actual, layer) => {
                const report = compareTensors(
                  { dtype: "f32", data: actual },
                  { dtype: "f32", data: read(inputs.q[layer], wrongK, wrongV) },
                  SHARED_TOLERANCE,
                );
                assertEquals(
                  report.pass,
                  false,
                  `${series.name} step ${index} layer ${layer}: append 先行のオラクルとも一致した` +
                    "（決定 5b の順序が値に出ない位置を probe に選んでいる = 空振り）",
                );
              });
            }

            const appendBase = { ...common, kvPlanes: model.kvHeads };
            slotK = asF32(referenceStateAppend({ ...appendBase, x: inputs.k, slot: slotK }).data);
            slotV = asF32(referenceStateAppend({ ...appendBase, x: inputs.v, slot: slotV }).data);
            past += query;
            assertEquals(
              context.pastLength,
              past,
              `${series.name} step ${index}: 論理長がずれた（append 2 本で 2 度進んでいる）`,
            );
          }
        } finally {
          await context.dispose();
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

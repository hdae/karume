// 送り込みの先行の上限（ADR 0123）の実 GPU の門。
//
// 固定するのは ①長い run の値が変わらず、ステップの間で前の印から submit 64 回ごとに完了印が 1 本ずつ出る
// こと ②run は上限を超えたらステップの間で印を待つこと（区間ロックを持ったまま）③enqueue は印を置かず待たない
// こと — 非 await で続けた別 Session の enqueue が、常駐テンソル越しに前の enqueue の出力を読む形で呼び出し順を
// 追い越さない（ADR 0123 決定 4）。
//
// ③ が無いと、enqueue で待つ実装へ戻しても単独の enqueue の値は正しいまま緑になる。崩れるのは
// 「待ちの間に別 Session の本体が走り、書かれる前の常駐テンソルを読む」形だけで、例外も警告も出ない。
// enqueue に印を置く形は、Deno では batch の生成ループを遅くする（印が GPU の完了までホストを止める）。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type BatchScope, type GpuContext } from "../src/gpu/device.ts";
import {
  IN_FLIGHT_MARKER_INTERVAL,
  IN_FLIGHT_MAX_MARKERS,
  type SubmitPolicy,
} from "../src/gpu/submit.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, openGraphModel } from "./helpers/model-fixture.ts";
import { countFences } from "./helpers/fences.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;
const BYTES = COUNT * 4;

/** 上限を超える（未完了の印が上限 + 1 本に届く）だけの add の段数。 */
const CHAIN = IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 1) + 8;

/** 1 dispatch = 1 submit（印の位置を dispatch の数から決められる形）。 */
const ONE_PER_SUBMIT: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 1,
  minChunkSize: 1,
  maxChunkSize: 1,
};

/** y = x + x + … + x（add を {@link CHAIN} 段つなぐ）。 */
const CHAIN_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: Object.fromEntries(
    Array.from({ length: CHAIN }, (_, i) => [
      i === CHAIN - 1 ? "y" : `v${i}`,
      { dtype: "f32", shape: [ROWS, COLS] },
    ]),
  ),
  nodes: Array.from({ length: CHAIN }, (_, i) => ({
    op: "add",
    ins: [i === 0 ? "x" : `v${i - 1}`, "x"],
    outs: [i === CHAIN - 1 ? "y" : `v${i}`],
    attrs: {},
  })),
};

/** w = z * z（{@link CHAIN_GRAPH} の出力を常駐テンソル経由で受ける）。 */
const SQUARE_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["mul"] },
  symbols: [],
  inputs: [{ name: "z", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["w"],
  initializers: {},
  values: { w: { dtype: "f32", shape: [ROWS, COLS] } },
  nodes: [{ op: "mul", ins: ["z", "z"], outs: ["w"], attrs: {} }],
};

/** `phase` ごとに値が変わる入力（同じ値を配ると古い値を読んだことが検出できない）。 */
const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => (i + phase * 5) % 9 - 4),
});

/** 参照値（ノードごとに f32 へ丸めながら手計算する — 実装とは独立）。 */
const expectedChain = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => {
    let value = x[i];
    for (let step = 0; step < CHAIN; step += 1) value = Math.fround(value + x[i]);
    return value;
  });
};

const expectedSquare = (phase: number): Float32Array<ArrayBuffer> =>
  Float32Array.from(expectedChain(phase), (value) => Math.fround(value * value));

/** ビット列比較（丸めの取り違えを許容しない）。常駐テンソルの読み戻しは素の ArrayBuffer。 */
const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const session = async (
  gpu: GpuContext,
  json: DeclarationJson,
  submitPolicy?: SubmitPolicy,
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(json), "model", { submitPolicy });

Deno.test({
  name:
    "長い run は値を変えず、ステップの間で前の印から submit 64 回ごとに完了印を 1 本置く（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const chain = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    const fences = countFences(gpu);
    try {
      // 1 run 目は非 backed（アリーナ経路）、2 run 目は backed。どちらもステップの間で待つ。
      for (const phase of [0, 1]) {
        const before = { fences: fences.count(), submits: chain.diagnostics().submit.submitCount };
        const outputs = await chain.run({ x: input(phase) });
        assertEquals(bits(outputs["y"].data), bits(expectedChain(phase)), `phase ${phase}`);
        const submits = chain.diagnostics().submit.submitCount - before.submits;
        assert(
          submits > IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 1),
          `上限を超える長さの run になっていない（submit ${submits} 回 — 門が空振りする）`,
        );
        // run のフェンスは mapAsync の 1 本（H-1）なので、onSubmittedWorkDone は完了印だけ。印はステップの
        // 間でだけ置くので、最後のステップの後に積む読み戻しの写し（1 本 = 1 submit）は数えない。
        assertEquals(
          fences.count() - before.fences,
          Math.floor((submits - 1) / IN_FLIGHT_MARKER_INTERVAL),
          `phase ${phase}: submit ${submits} 回に対する完了印の本数`,
        );
      }
    } finally {
      fences.restore();
      await chain.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "enqueue は待たない — 非 await で 2 周続けた別 Session の enqueue が常駐テンソル越しの順序を追い越さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const producer = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    const consumer = await session(gpu, SQUARE_GRAPH);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    const later = await gpu.createResident(BYTES, "later");
    try {
      // 1 batch 目は await で回して両 Session を導出済み計画 + backing にする（本体に await の無い
      // 経路 — 本体は呼び出し順にマイクロタスクの内で走り切る）。
      const warm = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch: warm, copyOutputs: { y: carrier } });
      await consumer.enqueue({ z: carrier }, { batch: warm, copyOutputs: { w: sink } });
      await warm.finish();
      assertEquals(bits(await sink.read()), bits(expectedSquare(0)), "1 batch 目");

      // 2 batch 目は非 await で 2 周積む。producer が途中で待つと、その間に consumer の本体が走って先に
      // submit し、前の carrier を読む。末尾で待つ形でも、2 周目の producer は同じ Session の直列化で
      // 1 周目の待ちに止められ、2 周目の consumer だけが先に走る。
      const batch = await gpu.beginBatch();
      await Promise.all([
        producer.enqueue({ x: input(1) }, { batch, copyOutputs: { y: carrier } }),
        consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } }),
        producer.enqueue({ x: input(2) }, { batch, copyOutputs: { y: carrier } }),
        consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: later } }),
      ]);
      await batch.finish();
      assertEquals(bits(await sink.read()), bits(expectedSquare(1)), "2 batch 目の 1 周目");
      assertEquals(bits(await later.read()), bits(expectedSquare(2)), "2 batch 目の 2 周目");
    } finally {
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      later.dispose();
      gpu.destroy();
    }
  },
});

/**
 * `onSubmittedWorkDone`（完了印とフェンス）の解決を手で止める。GPU の実行と `mapAsync` は止めない。
 * MUST: `release` を finally で必ず呼ぶ（開けて元の面へ戻す — 後始末の flush が固まる / 他のテストへ漏れる）。
 */
const gateWorkDone = (gpu: GpuContext): {
  /** 止めた `onSubmittedWorkDone` の呼び出しの回数（完了印とフェンス）。 */
  readonly calls: () => number;
  readonly open: () => void;
  readonly release: () => void;
} => {
  const queue = gpu.device.queue;
  const original = queue.onSubmittedWorkDone.bind(queue);
  const gate = Promise.withResolvers<void>();
  let calls = 0;
  Object.defineProperty(queue, "onSubmittedWorkDone", {
    configurable: true,
    writable: true,
    value: () => {
      calls += 1;
      return original().then(() => gate.promise);
    },
  });
  return {
    calls: () => calls,
    open: () => gate.resolve(),
    release: () => {
      gate.resolve();
      Object.defineProperty(queue, "onSubmittedWorkDone", {
        configurable: true,
        writable: true,
        value: original,
      });
    },
  };
};

/** 決着したかを外から読める形で Promise を見張る。 */
const watch = <T>(
  promise: Promise<T>,
): { readonly settled: () => boolean; readonly done: Promise<T> } => {
  let settled = false;
  const done = promise.finally(() => {
    settled = true;
  });
  return { settled: () => settled, done };
};

Deno.test({
  name:
    "run は上限を超えたらステップの間で完了印を待ち、それまで決着せず区間ロックも離さない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const chain = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    let gate: ReturnType<typeof gateWorkDone> | undefined;
    let entered: ReturnType<typeof watch<BatchScope>> | undefined;
    try {
      // 導出済み計画 + backing にしておく（パイプラインの生成の待ちを外す）。
      await chain.run({ x: input(0) });

      gate = gateWorkDone(gpu);
      const running = watch(chain.run({ x: input(1) }));
      // run が上限を超える本数の印を置くまで進める（印は区間ロックの中で置かれる）。先に beginBatch を
      // 呼ぶと、run の本体（マイクロタスクの先）より前に batch がロックを取ってしまう。
      for (let tick = 0; tick < 200 && gate.calls() <= IN_FLIGHT_MAX_MARKERS; tick += 1) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      assert(gate.calls() > IN_FLIGHT_MAX_MARKERS, "run が上限を超える本数の完了印を置いていない");
      // 待ちの間も run は errorScope 区間ロックを持ったまま — batch はロックを取れない（取れると、待ちの
      // 間に別 Session の enqueue が同じ queue へ割り込む）。
      entered = watch(gpu.beginBatch());
      // run のフェンスは mapAsync（止めていない）なので、印を待たない run はこの間に決着する。
      await new Promise((resolve) => setTimeout(resolve, 200));
      assertEquals(running.settled(), false, "完了印が解けるまで run は決着しない");
      assertEquals(entered.settled(), false, "待ちの間は batch が区間ロックを取れない");

      gate.open();
      const outputs = await running.done;
      assertEquals(bits(outputs["y"].data), bits(expectedChain(1)));
    } finally {
      gate?.release();
      // batch を閉じてから Session を畳む（batch が区間ロックを持ったままだと dispose が固まる）。
      if (entered !== undefined) await (await entered.done).finish();
      await chain.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "enqueue は完了印を置かず待たない — 上限を超える長さでも onSubmittedWorkDone を呼ばずに決着する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const producer = await session(gpu, CHAIN_GRAPH, ONE_PER_SUBMIT);
    const carrier = await gpu.createResident(BYTES, "carrier");
    let gate: ReturnType<typeof gateWorkDone> | undefined;
    try {
      // 導出済み計画 + backing にしておく（本体に await の無い経路 — 待たなければ本体は
      // マイクロタスクの内に決着する）。
      const warm = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch: warm, copyOutputs: { y: carrier } });
      await warm.finish();

      gate = gateWorkDone(gpu);
      const batch = await gpu.beginBatch();
      const enqueued = watch(
        producer.enqueue({ x: input(1) }, { batch, copyOutputs: { y: carrier } }),
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
      assertEquals(enqueued.settled(), true, "上限を超える長さでも enqueue は印を待たない");
      assertEquals(gate.calls(), 0, "enqueue は完了印を置かない");

      gate.open();
      await enqueued.done;
      await batch.finish();
      assertEquals(bits(await carrier.read()), bits(expectedChain(1)));
    } finally {
      gate?.release();
      await producer.dispose();
      carrier.dispose();
      gpu.destroy();
    }
  },
});

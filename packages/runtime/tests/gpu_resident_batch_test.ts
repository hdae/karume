// 常駐テンソル（第 4 の寿命クラス）とフェンス無し enqueue（H-5）の門。
//
// 狙いは「生成ループの run 境界フェンスを数本まで落とす」ことなので、この門が固定するのは
// ①値が正しいこと（常駐入力・写し先とも、ホスト経由の run とビット単位で一致）②**フェンスが
// 実際に減っていること**（batch 全体で onSubmittedWorkDone が 1 回）③追い越しが起きないこと
// （eager submit の不変条件）④寿命の破れが fail loudly になること（破棄後利用・参照中の破棄）。
//
// ② が無いと、内部で flush へ退避していても値は正しいまま緑になる（例外も警告も出ない）。
// ③ が無いと、前後の enqueue の入力が入れ替わっても「どちらも計算はされている」ので気づけない。

import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import {
  acquireGpu,
  BatchScopeError,
  type GpuContext,
  GpuDeviceLostError,
  ResidentTensorError,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { createSession, type Session, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { countFences } from "./helpers/fences.ts";
import type { GraphJson } from "./helpers/format.ts";
import { graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE } from "./helpers/gpu.ts";

const ROWS = 4;
const COLS = 3;
const COUNT = ROWS * COLS;
const BYTES = COUNT * 4;

/** y = x + x（= 2x）。生産側。 */
const PRODUCER: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [ROWS, COLS] } },
  nodes: [{ op: "add", ins: ["x", "x"], outs: ["y"], attrs: {} }],
};

/** w = z * z。消費側（生産側の出力を常駐テンソル経由で受ける）。 */
const CONSUMER: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["mul"] },
  symbols: [],
  inputs: [{ name: "z", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["w"],
  initializers: {},
  values: { w: { dtype: "f32", shape: [ROWS, COLS] } },
  nodes: [{ op: "mul", ins: ["z", "z"], outs: ["w"], attrs: {} }],
};

/** `phase` ごとに値が変わる入力（同じ値を配ると取り違えが検出できない）。 */
const input = (phase: number): Tensor => ({
  dtype: "f32",
  shape: [ROWS, COLS],
  data: Float32Array.from({ length: COUNT }, (_, i) => ((i + phase * 3) % 9 - 4) * 0.5),
});

/** 参照値。ノードごとに f32 へ丸めながら手計算する（実装とは独立）。 */
const expectedChain = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  const out = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i += 1) {
    const doubled = Math.fround(x[i] + x[i]);
    out[i] = Math.fround(doubled * doubled);
  }
  return out;
};

/** ビット列比較（丸めの取り違えを許容しない）。常駐テンソルの読み戻しは素の ArrayBuffer。 */
const bits = (data: Tensor["data"] | ArrayBuffer): readonly number[] =>
  Array.from(
    data instanceof ArrayBuffer
      ? new Uint32Array(data)
      : new Uint32Array(data.buffer, data.byteOffset, data.length),
  );

const producerSession = (gpu: GpuContext): Promise<Session> =>
  createSession(gpu, openModel(graphModelBuffer(PRODUCER)));

const consumerSession = (gpu: GpuContext): Promise<Session> =>
  createSession(gpu, openModel(graphModelBuffer(CONSUMER)));

Deno.test({
  name: "常駐入力は writeBuffer 無しで束ねられ、ホスト入力の run とビット一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const resident = await gpu.createResident(BYTES, "x");
    const session = await producerSession(gpu);
    const reference = await producerSession(gpu);
    try {
      for (const phase of [0, 1, 2]) {
        const tensor = input(phase);
        resident.write(tensor.data);
        // 1 run 目は非 backed（アリーナ経路）、2 run 目以降は backed。常駐入力は両経路とも
        // 「実体をそのまま束ねる」なので、どちらでも同じ値でなければならない。
        const actual = await session.run({ x: resident });
        const expected = await reference.run({ x: tensor });
        assertEquals(bits(actual["y"].data), bits(expected["y"].data), `phase ${phase}`);
      }
      assertEquals(session.diagnostics().lastRunPrepared?.hit, true);
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
    } finally {
      await session.dispose();
      await reference.dispose();
      resident.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "enqueue の連鎖は常駐テンソル経由で値を渡し、フェンスは batch の 1 本だけ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const producer = await producerSession(gpu);
    const consumer = await consumerSession(gpu);
    const carrier = await gpu.createResident(BYTES, "carrier");
    const sink = await gpu.createResident(BYTES, "sink");
    const fences = countFences(gpu);
    try {
      const batch = await gpu.beginBatch();
      await producer.enqueue({ x: input(0) }, { batch, copyOutputs: { y: carrier } });
      // 生産側の出力は常駐テンソルにしか残らない。消費側はそれを**そのまま**入力に取る。
      await consumer.enqueue({ z: carrier }, { batch, copyOutputs: { w: sink } });
      assertEquals(fences.count(), 0, "enqueue はフェンスを 1 本も張らない");

      await batch.finish();
      assertEquals(fences.count(), 1, "batch 全体で onSubmittedWorkDone は 1 回");

      assertEquals(bits(await sink.read()), bits(expectedChain(0)));
      // どちらの enqueue も初回から backed（アリーナ経路へ黙って退避していない）。
      assertEquals(producer.diagnostics().planBacking.buildCount, 1);
      assertEquals(consumer.diagnostics().planBacking.buildCount, 1);
      assertEquals(producer.diagnostics().lastRun, undefined, "enqueue はアリーナを使わない");
    } finally {
      fences.restore();
      await producer.dispose();
      await consumer.dispose();
      carrier.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "連続 enqueue の入力書き込みは先行 dispatch を追い越さない（eager submit の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const first = await gpu.createResident(BYTES, "first");
    const second = await gpu.createResident(BYTES, "second");
    try {
      // 同一 signature = 入力バッファは backing 所有の**1 本**。enqueue → writeBuffer（次の値）→
      // enqueue の順で積むので、eager submit が抜けると 2 本目の書き込みが 1 本目の dispatch を
      // 追い越し、両方が新しい値になる（例外は出ない）。
      const batch = await gpu.beginBatch();
      await session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: first } });
      await session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: second } });
      await batch.finish();

      const doubled = (phase: number): Float32Array<ArrayBuffer> => {
        const x = input(phase).data;
        return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
      };
      // 恒真化の門: 2 つの phase の期待値が同じなら取り違えを検出できない。
      assert(
        JSON.stringify(bits(doubled(0))) !== JSON.stringify(bits(doubled(1))),
        "phase ごとに期待値が変わっていない（検出器として空振る）",
      );
      assertEquals(bits(await first.read()), bits(doubled(0)), "先の enqueue は古い値を読む");
      assertEquals(bits(await second.read()), bits(doubled(1)), "後の enqueue は新しい値を読む");
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        1,
        "同一 signature で作り直さない",
      );
    } finally {
      await session.dispose();
      first.dispose();
      second.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "常駐入力の識別子は導出済み計画のキーに効く（差し替えで再導出・戻すとヒット・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const a = await gpu.createResident(BYTES, "a");
    const b = await gpu.createResident(BYTES, "b");
    try {
      a.write(input(0).data);
      b.write(input(1).data);

      await session.run({ x: a });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 1 });
      await session.run({ x: a });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 1 });

      // 別の常駐テンソルは別 signature。焼き込み済み bind group を使い回してはいけない。
      const swapped = await session.run({ x: b });
      assertEquals(
        session.diagnostics().lastRunPrepared,
        { hit: false, cachedPlans: 2 },
        "常駐入力の差し替えは別キー",
      );
      const doubled = (phase: number): readonly number[] => {
        const x = input(phase).data;
        return bits(Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i])));
      };
      assertEquals(bits(swapped["y"].data), doubled(1), "差し替え後は新しい実体を読む");

      const back = await session.run({ x: a });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: true, cachedPlans: 2 });
      assertEquals(bits(back["y"].data), doubled(0), "戻すと元の実体を読む");

      // ホスト入力（Tensor）と常駐入力ではキーが分かれる（同じ shape でも別 signature）。
      await session.run({ x: input(0) });
      assertEquals(session.diagnostics().lastRunPrepared, { hit: false, cachedPlans: 3 });
    } finally {
      await session.dispose();
      a.dispose();
      b.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "焼き込みから参照中の常駐テンソルは破棄できない（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const bound = await gpu.createResident(BYTES, "bound");
    const loose = await gpu.createResident(BYTES, "loose");
    try {
      bound.write(input(0).data);
      assertEquals(bound.bakedReferences, 0);
      // 1 run 目は非 backed（焼き込み無し）— この時点ではまだ破棄できる状態のはず。
      await session.run({ x: bound });
      assertEquals(bound.bakedReferences, 0, "非 backed run は焼き込み参照を作らない");
      // 2 run 目で backing が構築され、bind group がこの実体を焼き込む。
      await session.run({ x: bound });
      assertEquals(bound.bakedReferences, 1);

      assertThrows(
        () => bound.dispose(),
        ResidentTensorError,
        "焼き込み bind group から参照中",
      );
      assertEquals(bound.disposed, false, "拒否した破棄で状態を壊さない");

      // 恒真化の門: 参照されていない常駐テンソルは同じ呼び出しで通る（検査が常に落ちる形に
      // なっていないことの裏）。
      loose.dispose();
      assertEquals(loose.disposed, true);

      // 参照を外す唯一の経路（Session の破棄 / signature 切替）を通れば破棄できる。
      await session.dispose();
      assertEquals(bound.bakedReferences, 0);
      bound.dispose();
      assertEquals(bound.disposed, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "破棄済み常駐テンソルの利用は全ての面で fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const dead = await gpu.createResident(BYTES, "dead");
    try {
      dead.dispose();
      dead.dispose(); // 2 度目は no-op（破棄の冪等性）

      assertThrows(() => dead.write(new Float32Array(COUNT)), ResidentTensorError, "破棄済み");
      await assertRejects(() => dead.read(), ResidentTensorError, "破棄済み");
      await assertRejects(() => session.run({ x: dead }), ExecutionError, "破棄済み");

      const batch = await gpu.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: dead } }),
          ExecutionError,
          "破棄済み",
        );
      } finally {
        await batch.finish();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

// 消失した device では `queue.writeBuffer` が破棄済みバッファへの**沈黙 no-op**（警告すら
// 出ない）になり、`createBuffer` の errorScope も消失を捕らえない（pop は null で resolve）ため
// 無効なバッファを掴んだ resident が成功として返る。どちらも loud になるのは次のフェンスで、
// その間の診断は誤導的 — 入口の同期検査でしか止められない。
Deno.test({
  name: "device.destroy() 直後（lost の反応前）から常駐テンソルの受付を拒否する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // この case は device を壊すので専用の GpuContext を取る。
    const gpu = await acquireGpu();
    const resident = await gpu.createResident(BYTES, "victim");

    gpu.destroy();
    // MUST: ここで lost が**未記録**であること。記録済みなら既存の lost 判定だけで通ってしまい、
    // 「destroy の同期フラグを見ている」ことの証拠にならない（門が恒真になる）。
    assertStrictEquals(gpu.lost, undefined, "destroy 直後に lost の reaction が走っている");
    assert(gpu.destroyRequested, "destroy 要求が同期に立っていない");

    assertThrows(
      () => resident.write(new Float32Array(COUNT)),
      GpuDeviceLostError,
      "device が失われた",
    );
    await assertRejects(
      () => gpu.createResident(BYTES, "after-destroy"),
      GpuDeviceLostError,
      "device が失われた",
    );

    await gpu.device.lost;
    assert(gpu.lost !== undefined, "消失が記録されていない（門が空振りする）");
    assertThrows(() => resident.write(new Float32Array(COUNT)), GpuDeviceLostError);
    await assertRejects(() => gpu.createResident(BYTES, "after-lost"), GpuDeviceLostError);
  },
});

Deno.test({
  name: "常駐テンソルの大きさ不一致は入力側・写し先側とも fail loudly（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const wrong = await gpu.createResident(BYTES + 4, "wrong");
    const right = await gpu.createResident(BYTES, "right");
    try {
      assertThrows(() => right.write(new Float32Array(COUNT + 1)), ResidentTensorError, "write");
      await assertRejects(() => session.run({ x: wrong }), ExecutionError, "バイトと合わない");

      const batch = await gpu.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: wrong } }),
          ExecutionError,
          "バイトと合わない",
        );
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch, copyOutputs: { nope: right } }),
          ExecutionError,
          "グラフ出力ではない",
        );
      } finally {
        // 落ちた enqueue の後も区間そのものは決着できる（残骸は discard 済み）。ただし決着は
        // **最初のホスト側失敗**を帰属する — 2 本目の「グラフ出力ではない」ではなく 1 本目の
        // 大きさ不一致が出るのが、記録を 1 件に絞っている根拠（派生失敗で根因が隠れない）。
        await assertRejects(() => batch.finish(), ExecutionError, "バイトと合わない");
      }
    } finally {
      await session.dispose();
      wrong.dispose();
      right.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "createResident は 4 の倍数の正の大きさしか受けない",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (const bad of [0, -4, 6, 1.5]) {
        await assertRejects(
          () => gpu.createResident(bad),
          ResidentTensorError,
          "4 の倍数",
          `byteLength=${bad}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "決着済みの batch には enqueue できず、finish は何度でも同じ決着を返す（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      await session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: sink } });
      await batch.finish();
      await batch.finish();
      assertEquals(batch.finished, true);
      await assertRejects(
        () => session.enqueue({ x: input(0) }, { batch }),
        BatchScopeError,
        "finish() 済み",
      );

      // 区間を閉じた後の通常 run は従来どおり動く（scope ロックが返っている）。
      const outputs = await session.run({ x: input(0) });
      assertEquals(outputs["y"].shape, [ROWS, COLS]);
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "計測が有効な device では batch を開けない（実 GPU・timestamp-query 必須）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    try {
      await assertRejects(() => gpu.beginBatch(), BatchScopeError, "gpuTiming");
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "別の GpuContext で開いた batch には enqueue できない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const owner = await acquireGpu();
    const other = await acquireGpu();
    const session = await producerSession(other);
    try {
      const batch = await owner.beginBatch();
      try {
        await assertRejects(
          () => session.enqueue({ x: input(0) }, { batch }),
          BatchScopeError,
          "別の GpuContext",
        );
      } finally {
        await batch.finish();
      }
    } finally {
      await session.dispose();
      owner.destroy();
      other.destroy();
    }
  },
});

/** y = reshape(x)（[ROWS,COLS] → [COUNT]）。出力が入力の**別名**になる縮退グラフ。 */
const ALIAS: GraphJson = {
  format: "karume-ir",
  version: 1,
  requires: { ops: ["reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [ROWS, COLS] }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [COUNT] } },
  nodes: [{ op: "reshape", ins: ["x"], outs: ["y"], attrs: {} }],
};

Deno.test({
  name: "未 await の enqueue を積んだ直後の finish() でも全て区間に入る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      // MUST: 1 本も await しない。in-flight リースが無いと finish() が先に決着し、
      // ①3 本とも BatchScopeError で reject ②sink は全 0（0 dispatch）③それでも finish は
      // 成功で返る、という沈黙の空振りになる。
      // NOTE: phase は 3 周期（`input` の式）なので、値が割れる範囲で 3 本積む。
      const pending = [0, 1, 2].map((phase) =>
        session.enqueue({ x: input(phase) }, { batch, copyOutputs: { y: sink } })
      );
      await batch.finish();
      await Promise.all(pending);

      const doubled = (phase: number): Float32Array<ArrayBuffer> => {
        const x = input(phase).data;
        return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
      };
      // 恒真化の門: 先頭と末尾の期待値が同じなら「最後の 1 本が入った」ことを確かめられない。
      assert(
        JSON.stringify(bits(doubled(0))) !== JSON.stringify(bits(doubled(2))),
        "phase ごとに期待値が変わっていない（検出器として空振る）",
      );
      assertEquals(bits(await sink.read()), bits(doubled(2)), "最後の enqueue まで実行されている");
      assertEquals(session.diagnostics().planBacking.buildCount, 1);
    } finally {
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "別 GpuContext の常駐テンソルはヒット経路でも fail loudly（ID 衝突の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const owner = await acquireGpu();
    const other = await acquireGpu();
    const session = await producerSession(owner);
    const mine = await owner.createResident(BYTES, "mine");
    const foreign = await other.createResident(BYTES, "foreign");
    try {
      // 前提の確認: 採番は GpuContext ごとに独立なので識別子が衝突する（衝突しないなら
      // 導出済み計画のキーが分かれてしまい、この門は何も検出していないことになる）。
      assertEquals(
        mine[RUNTIME_INTERNAL].id,
        foreign[RUNTIME_INTERNAL].id,
        "別 context の 1 本目どうしは同じ識別子（この門が塞ぐ前提）",
      );
      assertEquals(mine.byteLength, foreign.byteLength, "サイズ検査では区別できない");

      mine.write(input(0).data);
      await session.run({ x: mine }); // 1 run 目 = ミス（アリーナ経路）
      await session.run({ x: mine }); // 2 run 目 = ヒット。backing が mine を焼き込む
      assertEquals(session.diagnostics().planBacking.buildCount, 1);

      // 焼き込み済み backing はキー一致でそのまま再利用されるため、検査が無いと foreign は
      // 1 バイトも読まれないまま mine の古い値が返る（例外ゼロ・警告ゼロ）。
      await assertRejects(
        () => session.run({ x: foreign }),
        ResidentTensorError,
        "別の GpuContext",
      );
      assertEquals(
        session.diagnostics().planBacking.buildCount,
        1,
        "拒否した run は backing を作り直さない",
      );
    } finally {
      await session.dispose();
      mine.dispose();
      foreign.dispose();
      owner.destroy();
      other.destroy();
    }
  },
});

Deno.test({
  name: "copyOutputs の写し元と写し先が同一バッファなら enqueue 時点で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await createSession(gpu, openModel(graphModelBuffer(ALIAS)));
    const state = await gpu.createResident(BYTES, "state");
    try {
      const batch = await gpu.beginBatch();
      try {
        // y は x の別名（reshape）なので、写し元の実体は state そのもの。検査が無いと
        // 自己コピーが積まれ、validation 失敗は batch.finish() まで遅れる（= 同じ区間の
        // 無関係な enqueue まで巻き添えになり、原因の enqueue が特定できない）。
        await assertRejects(
          () => session.enqueue({ x: state }, { batch, copyOutputs: { y: state } }),
          ExecutionError,
          "写し元と写し先が同じバッファ",
        );
      } finally {
        // 落ちた enqueue はこの区間の最初のホスト側失敗として決着にも帰属する（RC1-2）。
        await assertRejects(() => batch.finish(), ExecutionError, "写し元と写し先が同じバッファ");
      }
    } finally {
      await session.dispose();
      state.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "進行中の run が束ねている常駐テンソルは破棄できない（故障注入・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const bound = await gpu.createResident(BYTES, "bound");
    const device = gpu.device;
    const original = device.createComputePipeline.bind(device);
    let injected = 0;
    try {
      bound.write(input(0).data);
      // ミス run は「env へ実体を束縛 → パイプライン生成を await → エンコード」の順で進む。
      // その await 窓（= パイプライン生成中）で dispose を試すのがこの注入。焼き込み参照は
      // まだ 0 本なので、束縛予約が無ければここは素通りしてしまう。
      device.createComputePipeline = ((
        descriptor: GPUComputePipelineDescriptor,
      ): GPUComputePipeline => {
        if (injected === 0) {
          injected += 1;
          assertEquals(bound.bakedReferences, 0, "焼き込み参照はまだ立っていない窓");
          assertEquals(bound.boundReferences, 1, "束縛予約が立っている");
          assertThrows(() => bound.dispose(), ResidentTensorError, "束縛中");
        }
        return original(descriptor);
      }) as typeof device.createComputePipeline;

      const outputs = await session.run({ x: bound });
      assertEquals(injected, 1, "注入が 1 度も走っていない（窓を踏めていない）");
      const x = input(0).data;
      assertEquals(
        bits(outputs["y"].data),
        bits(Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]))),
        "拒否された dispose は run の値を壊さない",
      );

      // run が完了すれば予約は返っている（返し損ねると以後ずっと破棄できなくなる）。
      assertEquals(bound.boundReferences, 0);
      assertEquals(bound.disposed, false, "拒否した破棄で状態を壊さない");
      bound.dispose();
      assertEquals(bound.disposed, true);
    } finally {
      device.createComputePipeline = original;
      await session.dispose();
      gpu.destroy();
    }
  },
});

/** 2x（PRODUCER の出力）の参照値。 */
const doubled = (phase: number): Float32Array<ArrayBuffer> => {
  const x = input(phase).data;
  return Float32Array.from({ length: COUNT }, (_, i) => Math.fround(x[i] + x[i]));
};

// 非 await の enqueue が本体で落ちても、区間は「dispatch が 1 本少ないまま成功」で決着して
// いた（errorScope が捕らえるのは GPU 側の失敗だけで、ホスト側の throw は戻り Promise にしか
// 出ない = 握っていなければ未処理拒否として抜けるだけ）。この門はその帰属を固定する。
Deno.test({
  name: "非 await の enqueue の失敗は finish() に帰属する（最初の 1 件・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const wrong = await gpu.createResident(BYTES + 4, "wrong");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      const batch = await gpu.beginBatch();
      // MUST: どちらも await しない（この面が明示的に許している形）。1 本目は写し先の大きさが
      // 合わずに本体で落ち、2 本目は正常に積まれる。
      const pending = [
        session.enqueue({ x: input(0) }, { batch, copyOutputs: { y: wrong } }),
        session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: sink } }),
      ];
      // 未処理拒否を作らないよう、発行と同じ同期区間でハンドラを付ける。
      const settled = Promise.allSettled(pending);

      const attributed = await assertRejects(
        () => batch.finish(),
        ExecutionError,
        "バイトと合わない",
      );

      const results = await settled;
      assert(results[0].status === "rejected", "落ちた enqueue の戻り Promise は reject する");
      assertStrictEquals(
        results[0].reason,
        attributed,
        "finish が投げるのは enqueue が落ちたその例外そのもの（同じ 1 事実が 2 経路で見える）",
      );
      assertEquals(results[1].status, "fulfilled", "正常な enqueue は巻き添えで落ちない");

      // 恒真化の門: 期待値が全 0 なら「写しが起きた」ことを確かめられない。
      assert(
        bits(doubled(1)).some((word) => word !== 0),
        "期待値が全 0（写しの有無を判定できない）",
      );
      assertEquals(bits(await sink.read()), bits(doubled(1)), "正常な enqueue の写し先は更新済み");
    } finally {
      await session.dispose();
      wrong.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
});

// 束縛予約（`#bindInput`）が立つのは実行本体 = マイクロタスク 1 段後なので、焼き込みも束縛も
// 0 本の resident には「API が受理した直後の同じ tick に dispose が通る」窓が開いていた。
// dispose の doc が謳う「誤りは dispose の呼び出し点で真因のまま落ちる」を、受理済みの
// run / enqueue に対しても成り立たせるのが発行時の使用予約。
Deno.test({
  name: "発行済み run / enqueue が使う常駐テンソルは同じ tick でも破棄できない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const fresh = await gpu.createResident(BYTES, "fresh");
    const sink = await gpu.createResident(BYTES, "sink");
    try {
      fresh.write(input(0).data);
      // 1 run 目はミス経路（backing 無し）— 焼き込み参照が 1 本も立たない窓であることの確認。
      assertEquals(fresh.bakedReferences, 0, "焼き込み参照が既に立っている（門が空振りする）");

      const pending = session.run({ x: fresh });
      assertEquals(fresh.useReferences, 1, "発行の同期区間で使用予約が立っていない");
      assertEquals(fresh.boundReferences, 0, "束縛予約は本体まで立たない（この窓の証拠）");
      assertThrows(() => fresh.dispose(), ResidentTensorError, "使用中");
      assertEquals(fresh.disposed, false, "拒否した破棄で状態を壊さない");

      // 拒否した dispose は run を壊さない（受理済みの実行はそのまま完走する）。
      assertEquals(bits((await pending)["y"].data), bits(doubled(0)));
      assertEquals(fresh.useReferences, 0, "決着で使用予約が返っていない");

      // 写し先（copyOutputs）も同じ寿命の保護を受ける。
      const batch = await gpu.beginBatch();
      const queued = session.enqueue({ x: input(1) }, { batch, copyOutputs: { y: sink } });
      assertEquals(sink.useReferences, 1, "写し先に使用予約が立っていない");
      assertThrows(() => sink.dispose(), ResidentTensorError, "使用中");
      await queued;
      await batch.finish();
      assertEquals(sink.useReferences, 0);
      assertEquals(bits(await sink.read()), bits(doubled(1)));

      // 決着後は通る（予約が返っていることの裏 — 検査が恒真になっていない）。
      fresh.dispose();
      sink.dispose();
      assertEquals(fresh.disposed, true);
      assertEquals(sink.disposed, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "使用予約の取得が途中で落ちても取得済みは漏れない（rollback の門・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const session = await producerSession(gpu);
    const alive = await gpu.createResident(BYTES, "alive");
    const spare = await gpu.createResident(BYTES, "spare");
    const dead = await gpu.createResident(BYTES, "dead");
    try {
      dead.dispose();
      const batch = await gpu.beginBatch();
      // 入力（alive）の予約を取った後に写し先（dead）で落ちる形。巻き戻さないと alive は
      // 以後ずっと破棄できない。
      await assertRejects(
        () => session.enqueue({ x: alive }, { batch, copyOutputs: { y: dead } }),
        ExecutionError,
        "破棄済み",
      );
      assertEquals(alive.useReferences, 0, "取得済みの使用予約が巻き戻っていない");
      // 区間に入る前に落ちた enqueue は区間の失敗ではない（リースを取っていない）。
      await batch.finish();

      // batch.enter() が落ちる経路でも同じ（ここでは入力と写し先の 2 本を取った後に落ちる）。
      await assertRejects(
        () => session.enqueue({ x: alive }, { batch, copyOutputs: { y: spare } }),
        BatchScopeError,
        "finish() 済み",
      );
      assertEquals(alive.useReferences, 0, "enter 失敗で使用予約が漏れている");
      assertEquals(spare.useReferences, 0, "enter 失敗で使用予約が漏れている");

      alive.dispose();
      spare.dispose();
      assertEquals(alive.disposed, true);
      assertEquals(spare.disposed, true);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

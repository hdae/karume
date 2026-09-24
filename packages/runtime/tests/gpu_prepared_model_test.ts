// 2 段境界（ADR 0070 決定 5 / graph-first）の実 GPU 門 — `prepareContainer(opened, graph) →
// estimate() → createContainerSession(gpu)` が、1 本にまとめた createSessionFromContainer と
// **同じ Session** を作ること。
//
// 検出器は 2 つ: 全重みを通る run の出力ビット同一と storage 診断の一致。fixture は 4 codec 混在
// （f32 / f16 / int8-sym / int4-sym-g）を使う — どれか 1 codec が 2 段の経路で欠けても沈黙しない
// ため。

import { assert, assertEquals, assertRejects } from "@std/assert";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { buildFixture, countingContainer } from "./helpers/mixed-codec-fixture.ts";
import {
  baseDeclaration,
  baseTensors,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const bitsOf = (tensor: Tensor): readonly number[] => [
  ...new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length),
];

Deno.test({
  name: "2 段境界は createSessionFromContainer と出力ビット同一・storage 診断一致（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const gpu = await acquireGpu();
    try {
      // 相 1: 開いた容器の宣言だけで admission（重みの block は 1 つも取っていない）
      const counted = countingContainer(await openModelBytes(fixture.declaration, fixture.tensors));
      const prepared = prepareContainer(counted.container, GRAPH_NAME);
      const estimate = prepared.estimate();
      assertEquals(counted.reads(), 0, "admission が重みの block を取っている");
      // 相 2: block を取って構築
      const twoPhase = await prepared.createContainerSession(gpu);
      const composed = await createSessionFromContainer(
        gpu,
        await openModelBytes(fixture.declaration, fixture.tensors),
        GRAPH_NAME,
      );
      try {
        const twoPhaseOut = (await twoPhase.run({ x: fixture.x }))["y"];
        const composedOut = (await composed.run({ x: fixture.x }))["y"];
        assertEquals(bitsOf(twoPhaseOut), bitsOf(composedOut), "出力がビット同一でない");
        assertEquals(twoPhase.diagnostics().storage, composed.diagnostics().storage);
        const storage = twoPhase.diagnostics().storage;
        assert(storage.residentCompressedBytes > 0, "圧縮常駐が 1 本も無い");
        // 重みを 1 バイトも取る前に出した数字が、実測の圧縮常駐と厳密一致する
        // （ADR 0070 決定 5 の対応表 — 2 段境界にした意味がここに出る）
        assertEquals(estimate.resident.weights.compressedBytes, storage.residentCompressedBytes);
      } finally {
        await twoPhase.dispose();
        await composed.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "admission は重みの block を 1 つも取る前に落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // bf16 は容器の codec 台帳にはあるが実行経路が無い（capability 不足）。
    const counted = countingContainer(
      await openModelBytes(baseDeclaration(), [
        {
          graph: GRAPH_NAME,
          initializer: "w",
          bytes: new Uint8Array(new ArrayBuffer(24)),
          encoding: { codec: "bf16" },
        },
        baseTensors()[1],
      ]),
    );
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, counted.container, GRAPH_NAME),
        RuntimeSupportError,
        "bf16",
      );
      assertEquals(counted.reads(), 0, "実行できないモデルなのに重みの block を取っている");
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * 重み `w`（`[4,3]` f32）を 2 本の piece で供給し、**2 本目の読みだけ**を落とすメモリ内容器。
 *
 * MUST: 落とし方は「構築の途中」でなければならない。admission（capability / op 契約）で
 * 落とすと GPU の資源を 1 つも作る前に戻るので、「半端な資源が残らない」という主張が
 * 空振りする（`:70` の門がその相を別に持っている）。piece 1 は実バイト列を返して
 * `queue.writeBuffer` まで進ませ、piece 2 の `read()` で初めて落とす。
 */
const halfUploadedContainer = (): { readonly container: BoundContainer; reads(): number } => {
  let reads = 0;
  const half = f32Bytes(new Array(6).fill(0.5));
  return {
    container: openMemoryContainer({
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(baseDeclaration()) },
      tensors: {
        [GRAPH_NAME]: {
          w: {
            encoding: { codec: "f32" },
            pieces: [
              {
                rows: [0, 2],
                read: () => {
                  reads += 1;
                  return Promise.resolve(half);
                },
              },
              {
                rows: [2, 4],
                read: () => {
                  reads += 1;
                  return Promise.reject(new Error("piece 2 の供給が落ちた"));
                },
              },
            ],
          },
          b: { encoding: { codec: "f32" }, bytes: f32Bytes([1, 2, 3]) },
        },
      },
    }),
    reads: () => reads,
  };
};

Deno.test({
  name: "構築の途中で落ちても同じ device で作り直せる（部分 Session が残らない・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // Session 構築が途中で失敗したとき、GPU 側に半端な資源や壊れた常駐を残さないこと。
    // 供給元が容器になっても成り立つべき不変条件で、admission の相だけを見る他の 2 本とは別。
    const broken = halfUploadedContainer();
    const healthy = await openModelBytes();
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, broken.container, GRAPH_NAME),
        Error,
        "piece 2 の供給が落ちた",
      );
      // 恒真でないことの確認 — 1 本目は実際に上がっており、落ちたのは転送が始まった後である。
      assertEquals(
        broken.reads(),
        2,
        "piece の読みが 2 回に達していない（転送が始まる前に落ちている）",
      );
      // 同じ device で正常な容器から作り直し、最後まで走ること（x[1,4]·w[4,3] + b）。
      const session = await createSessionFromContainer(gpu, healthy, GRAPH_NAME);
      try {
        const out = (await session.run({
          x: { dtype: "f32", shape: [1, 4], data: Float32Array.from([1, 1, 1, 1]) },
        }))["y"];
        // w は全要素 0.5 なので行和は 2・b は [1,2,3]。
        assertEquals([...out.data], [3, 4, 5]);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

// 2 段境界（ADR 0070 決定 5 / graph-first）の実 GPU 門 — `prepareModel(graphShard) →
// estimate() → createSession(gpu, weightShards)` が、1 本にまとめた createSessionFromShards と
// **同じ Session** を作ること。
//
// 検出器は A/B 門（gpu_shard_session_test.ts）と同じ 2 つ: 全重みを通る run の出力ビット同一と
// storage 診断の一致。fixture も同じ 4 格納混在（f32 / f16 / i8 / i4）を使う — どれか 1 格納が
// 2 段の経路で欠けても沈黙しないため。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { ContainerError } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  createSessionFromShards,
  type ModelShard,
  prepareModel,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { buildFixture, shardStream } from "./helpers/shard-fixture.ts";

/** 配布形のファイル名らしい実名（帰属の検出器が連番と取り違えないため）。 */
const IDS = [
  "fixture-v1.0/transformer/model-00001.safetensors",
  "fixture-v1.0/transformer/model-00002.safetensors",
  "fixture-v1.0/transformer/model-00003.safetensors",
];

const bitsOf = (tensor: Tensor): readonly number[] => [
  ...new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length),
];

Deno.test({
  name: "2 段境界は createSessionFromShards と出力ビット同一・storage 診断一致（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const shards = fixture.shards();
    const gpu = await acquireGpu();
    try {
      // 相 1: グラフ shard だけで admission（重み shard は 1 本も渡していない）
      const prepared = prepareModel({ id: IDS[0], bytes: new Uint8Array(shards[0]) });
      const estimate = prepared.estimate();
      // 相 2: 重み shard 列（グラフ shard を含まない）を渡して構築
      const twoPhase = await prepared.createSession(
        gpu,
        shardStream(shards.slice(1), IDS.slice(1)),
      );
      const composed = await createSessionFromShards(gpu, shardStream(fixture.shards(), IDS));
      try {
        const twoPhaseOut = (await twoPhase.run({ x: fixture.x }))["y"];
        const composedOut = (await composed.run({ x: fixture.x }))["y"];
        assertEquals(bitsOf(twoPhaseOut), bitsOf(composedOut), "出力がビット同一でない");
        assertEquals(twoPhase.diagnostics().storage, composed.diagnostics().storage);
        const storage = twoPhase.diagnostics().storage;
        assert(storage.residentCompressedBytes > 0, "圧縮常駐が 1 本も無い");
        // 重み shard を 1 バイトも取る前に出した数字が、実測の圧縮常駐と厳密一致する
        // （ADR 0070 決定 5 の対応表 — 2 段境界にした意味がここに出る）
        assertEquals(estimate.compressedWeightBytes, storage.residentCompressedBytes);
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
  name: "重み shard 列にグラフ shard を混ぜると既存文言で落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const shards = fixture.shards();
    const gpu = await acquireGpu();
    try {
      const prepared = prepareModel({ id: IDS[0], bytes: new Uint8Array(shards[0]) });
      // 重み側に「グラフ shard をもう 1 本」渡す取り違え（ADR 0070 決定 3 — 先頭の 1 本だけ）
      const duplicated = await assertRejects(
        () => prepared.createSession(gpu, shardStream([fixture.fullBuffer()], [IDS[1]])),
        ExecutionError,
        "グラフ shard が複数",
      );
      // 帰属は重み shard 側の連番（グラフ shard が [0]）
      assert(duplicated.message.includes(`shard [1] '${IDS[1]}'`), duplicated.message);

      // 落ちた後も同じ device で作り直せる（部分 Session や壊れた常駐が残っていない）
      const session = await prepared.createSession(gpu, shardStream(shards.slice(1), IDS.slice(1)));
      try {
        assertEquals((await session.run({ x: fixture.x }))["y"].shape, [2, 4]);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "admission は重み shard を 1 本も引く前に落ちる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const shards = fixture.shards();
    let pulledWeights = 0;
    // 先頭に重み shard を置いた取り違え（karume_ir が無い）。後続を引いたら数えるが、
    // admission が先に落ちるので 1 本も引かれない。
    const counting = async function* (): AsyncGenerator<ModelShard, void, unknown> {
      yield { id: IDS[0], bytes: new Uint8Array(shards[1]) };
      pulledWeights += 1;
      yield { id: IDS[1], bytes: new Uint8Array(shards[2]) };
    };
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromShards(gpu, counting()),
        ContainerError,
        "karume_ir",
      );
      assertEquals(pulledWeights, 0, "実行できないモデルなのに重み shard を引いている");
    } finally {
      gpu.destroy();
    }
  },
});

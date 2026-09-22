/**
 * PLE の GPU 常駐席（ADR 0085 追記〈GPU 常駐席〉）のうち、**GPU を要さない部分**の門。
 *
 * 見るのは 3 つ — ①席の指定の受理と拒否 ②gather IR の形（ノード列・shape・格納の宣言）
 * ③容器の資産 → 合成コンテナの突合（piece の並びと、繋ぎ直したバイト列が原本と一致すること）。
 * 実 GPU での値の一致は `e2e_gemma4_ple_gpu_test.ts` が持つ。
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, prepareModel } from "@karume/runtime";
import {
  buildGemma4PleGatherShards,
  gemma4PleGatherGraph,
  gemma4PleGpuBytes,
} from "../src/gemma/ple-gpu.ts";
import { ModelInputError } from "../src/errors.ts";
import { resolveGemma4PleResidency } from "../src/gemma/pipeline.ts";
import { pleFixture, type PleFixtureSpec } from "./helpers/ple-fixture.ts";

/** 索引 1 本ぶんの作り物（実資産と同じ綴り・同じ関係で、桁だけ小さい）。 */
const SPEC: PleFixtureSpec = {
  storage: "i4",
  tokens: 6,
  layers: 2,
  dim: 16,
  embedScale: 16,
  valueRows: 4,
};

/** 1 層ぶんのバイト数（i4 なので dim / 2）。 */
const ROW_BYTES = SPEC.dim / 2;
/** 重みの行数（token × 層）。 */
const WEIGHT_ROWS = SPEC.tokens * SPEC.layers;

const rig = pleFixture(SPEC);
const index = rig.index;

Deno.test("pleResidency: 受理と、併用できないノブの拒否", () => {
  assertEquals(resolveGemma4PleResidency("T", {}), "host");
  assertEquals(resolveGemma4PleResidency("T", { pleResidency: "host" }), "host");
  assertEquals(resolveGemma4PleResidency("T", { pleResidency: "gpu" }), "gpu");
  // 綴り違いを既定へ倒さない（型の外から来る指定）。
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "GPU" as "gpu" }),
    Error,
    "pleResidency",
  );
  // 併用できないノブの組合せは呼び手のオプションだけで決まるので入力起因（ADR 0107 決定 2）。
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", maxResidentPleBytes: 0 }),
    ModelInputError,
    "maxResidentPleBytes",
  );
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", speculative: {} }),
    ModelInputError,
    "speculative",
  );
  // 逆側: 綴り違いは「受理集合を引き直す」側なので入力起因にしない（決定 3 の除外）。
  const misspelled = assertThrows(() =>
    resolveGemma4PleResidency("T", { pleResidency: "GPU" as "gpu" })
  );
  assert(!(misspelled instanceof ModelInputError));
});

Deno.test("gather IR: embedding → mul の 2 ノードで、添字の形がそのまま per-layer の形になる", () => {
  const graph = gemma4PleGatherGraph(index) as {
    nodes: { op: string; ins: string[]; outs: string[] }[];
    inputs: { name: string; dtype: string; shape: (string | number)[] }[];
    outputs: string[];
    values: Record<string, { dtype: string; shape: (string | number)[] }>;
    initializers: Record<string, { tensor: string; storage: Record<string, unknown> }>;
  };
  assertEquals(graph.nodes.map((node) => node.op), ["embedding", "mul"]);
  assertEquals(graph.nodes[0].ins, ["ple", "ple_index"]);
  assertEquals(graph.nodes[1].ins, ["ple_raw", "embed_scale"]);
  assertEquals(graph.inputs, [{ name: "ple_index", dtype: "i32", shape: [1, "M", 2] }]);
  // 重みは (token, layer) を行に畳んだ形 — 1 行 = 1 層ぶんなので scale の並びが索引と揃う。
  assertEquals(graph.values.ple.shape, [WEIGHT_ROWS, SPEC.dim]);
  assertEquals(graph.values[graph.outputs[0]].shape, [1, "M", 2, SPEC.dim]);
  assertEquals(graph.initializers.ple.storage, {
    dtype: "i4",
    scale: "ple_scale",
    group_size: SPEC.dim,
  });
  // i8 は行 scale なので group を宣言しない。
  const i8 = gemma4PleGatherGraph({ ...index, storage: "i8" }) as typeof graph;
  assertEquals(i8.initializers.ple.storage, { dtype: "i8", scale: "ple_scale" });
  const i2 = gemma4PleGatherGraph({ ...index, storage: "i2" }) as typeof graph;
  assertEquals(i2.initializers.ple.storage, { dtype: "i2", scale: "ple_scale" });
});

Deno.test("gather IR: ランタイムの op 契約と格納宣言を通る", async () => {
  const { graphShard } = await buildGemma4PleGatherShards(index, rig.openBlock, "T");
  // prepareModel は非対応 op / 格納 dtype と IR 契約を全件見る（GPU は触らない）。
  const prepared = prepareModel(graphShard);
  assertEquals(prepared.graph.outputs, ["per_layer"]);
  assertEquals(prepared.graph.symbols, ["M"]);
});

Deno.test("常駐バイト数: 索引だけで決まる（E2B QAT の実値）", () => {
  const e2b = pleFixture({ storage: "i4", tokens: 4096, layers: 35, dim: 256, embedScale: 16 })
    .index;
  assertEquals(gemma4PleGpuBytes(e2b), {
    values: 4096 * 35 * 256 / 2,
    scales: 4096 * 35 * 4,
    total: 4096 * 35 * 256 / 2 + 4096 * 35 * 4,
  });
  // i8（通常 Gemma 4）は同じ寸法で倍になる。
  const i8 = pleFixture({ storage: "i8", tokens: 4096, layers: 35, dim: 256, embedScale: 16 })
    .index;
  assertEquals(gemma4PleGpuBytes(i8).values, 4096 * 35 * 256);
  const i2 = pleFixture({ storage: "i2", tokens: 4096, layers: 35, dim: 256, embedScale: 16 })
    .index;
  assertEquals(gemma4PleGpuBytes(i2).values, 4096 * 35 * 256 / 4);
});

Deno.test("合成コンテナ: piece が順に並び、繋ぎ直すと原本の values と一致する", async () => {
  const { graphShard, weightShards } = await buildGemma4PleGatherShards(index, rig.openBlock, "T");
  const graph = parseSafetensors(graphShard.bytes.buffer, graphShard.bytes.byteLength);
  // scale は piece 1 と同じ shard に置く（ADR 0090 決定 1 の co-shard）。
  const scale = graph.tensors.get("ple_scale");
  assertEquals(scale?.dtype, "F32");
  assertEquals(scale?.shape, [WEIGHT_ROWS, 1]);
  const scales = new Float32Array(
    graph.buffer.slice(scale!.byteOffset, scale!.byteOffset + scale!.byteLength),
  );
  // block 順に連結した scale が、平坦添字 `id * layers + layer` の並びになっている。
  assertEquals(
    [...scales],
    Array.from({ length: WEIGHT_ROWS }, (_row, at) => 1 / 2 ** ((at % SPEC.layers) + 1)),
  );
  const embedScale = graph.tensors.get("embed_scale");
  assertEquals(embedScale?.shape, [1]);

  const collected: Uint8Array[] = [];
  const keys: string[] = [];
  for (const [name, view] of graph.tensors) {
    if (!name.startsWith("ple#")) continue;
    keys.push(name);
    collected.push(new Uint8Array(graph.buffer, view.byteOffset, view.byteLength));
  }
  for await (const shard of weightShards) {
    const file = parseSafetensors(shard.bytes.buffer, shard.bytes.byteLength);
    assertEquals(file.tensors.size, 1);
    for (const [name, view] of file.tensors) {
      keys.push(name);
      // 器を使い回すので、次の shard へ進む前に写す。
      collected.push(
        new Uint8Array(file.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)),
      );
    }
  }
  assertEquals(keys, ["ple#00001-of-00003", "ple#00002-of-00003", "ple#00003-of-00003"]);
  // piece 1 = 先頭 1 行（graph shard に同居）・残りは values の block 順。
  assertEquals(collected.map((piece) => piece.byteLength / ROW_BYTES), [1, 7, 4]);
  const joined = new Uint8Array(WEIGHT_ROWS * ROW_BYTES);
  let cursor = 0;
  for (const piece of collected) {
    joined.set(piece, cursor);
    cursor += piece.byteLength;
  }
  const expected = new Uint8Array(WEIGHT_ROWS * ROW_BYTES);
  let at = 0;
  for (const block of index.values.blocks) {
    const bytes = rig.blocks.get(block.asset)!;
    expected.set(bytes, at);
    at += bytes.byteLength;
  }
  assertEquals(joined, expected);
});

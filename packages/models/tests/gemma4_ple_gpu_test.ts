/**
 * PLE の GPU 常駐席（ADR 0085 追記〈GPU 常駐席〉）のうち、**GPU を要さない部分**の門。
 *
 * 見るのは 3 つ — ①席の指定の受理と拒否 ②gather IR の形（ノード列・shape・格納の宣言）
 * ③sidecar → 合成コンテナの突合（piece の並びと、繋ぎ直したバイト列が原本と一致すること）。
 * 実 GPU での値の一致は `e2e_gemma4_ple_gpu_test.ts` が持つ。
 */
import { assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, prepareModel } from "@karume/runtime";
import {
  buildGemma4PleGatherShards,
  gemma4PleGatherGraph,
  gemma4PleGpuBytes,
} from "../src/gemma/ple-gpu.ts";
import { resolveGemma4PleResidency } from "../src/gemma/pipeline.ts";
import type { Gemma4PleIndex, Gemma4PleShard, Gemma4PleShardSource } from "../src/gemma/ple.ts";

/** 索引 1 本ぶんの作り物（実資産と同じ綴り・同じ関係で、桁だけ小さい）。 */
const index: Gemma4PleIndex = {
  schema: 2,
  storage: "i4",
  tokens: 6,
  layers: 2,
  dim: 16,
  embedScale: 16,
  shards: [
    { file: "ple-a.safetensors", start: 0, stop: 4 },
    { file: "ple-b.safetensors", start: 4, stop: 6 },
  ],
} as Gemma4PleIndex & { readonly schema: number };

/** 1 層ぶんのバイト数（i4 なので dim / 2）。 */
const ROW_BYTES = index.dim / 2;

const writeShard = (shard: Gemma4PleShard): ArrayBuffer => {
  const rows = shard.stop - shard.start;
  const values = new Uint8Array(rows * index.layers * ROW_BYTES);
  // 行ごとに違うバイト列にする（繋ぎ直しの取り違えが値で出るように）。
  values.forEach((_, position) => {
    values[position] = (shard.start * index.layers * ROW_BYTES + position) % 251;
  });
  const scales = new Float32Array(rows * index.layers);
  scales.forEach((_, position) => {
    scales[position] = (shard.start * index.layers + position + 1) / 64;
  });
  const header = {
    __metadata__: {
      karume_ple: JSON.stringify({
        schema: 2,
        storage: "i4",
        tokens: index.tokens,
        layers: index.layers,
        dim: index.dim,
        embedScale: index.embedScale,
        start: shard.start,
        stop: shard.stop,
      }),
    },
    values: {
      dtype: "I4",
      shape: [rows, index.layers, index.dim],
      data_offsets: [0, values.byteLength],
    },
    scales: {
      dtype: "F32",
      shape: [rows, index.layers],
      data_offsets: [values.byteLength, values.byteLength + scales.byteLength],
    },
  };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = Math.ceil(json.byteLength / 4) * 4;
  const buffer = new ArrayBuffer(8 + headerLength + values.byteLength + scales.byteLength);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(json, 8);
  bytes.fill(0x20, 8 + json.byteLength, 8 + headerLength);
  bytes.set(values, 8 + headerLength);
  bytes.set(new Uint8Array(scales.buffer), 8 + headerLength + values.byteLength);
  return buffer;
};

/** 全量読みだけを持つ読み口（区間読みを持たない取得元の経路をこちらで踏む）。 */
const openShard = (file: string): Promise<Gemma4PleShardSource> => {
  const shard = index.shards.find((entry) => entry.file === file);
  if (shard === undefined) throw new Error(`test: shard '${file}' が索引に無い`);
  const buffer = writeShard(shard);
  return Promise.resolve({ bytes: buffer.byteLength, readAll: () => Promise.resolve(buffer) });
};

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
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", maxResidentPleBytes: 0 }),
    Error,
    "maxResidentPleBytes",
  );
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", speculative: {} }),
    Error,
    "speculative",
  );
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
  // 重みは (token, layer) を行に畳んだ形 — 1 行 = 1 層ぶんなので scale の並びが sidecar と揃う。
  assertEquals(graph.values.ple.shape, [12, 16]);
  assertEquals(graph.values[graph.outputs[0]].shape, [1, "M", 2, 16]);
  assertEquals(graph.initializers.ple.storage, {
    dtype: "i4",
    scale: "ple_scale",
    group_size: 16,
  });
  // i8 sidecar（schema 1）は行 scale なので group を宣言しない。
  const i8 = gemma4PleGatherGraph({ ...index, storage: undefined }) as typeof graph;
  assertEquals(i8.initializers.ple.storage, { dtype: "i8", scale: "ple_scale" });
  const i2 = gemma4PleGatherGraph({ ...index, storage: "i2" }) as typeof graph;
  assertEquals(i2.initializers.ple.storage, { dtype: "i2", scale: "ple_scale" });
});

Deno.test("gather IR: ランタイムの op 契約と格納宣言を通る", async () => {
  const { graphShard } = await buildGemma4PleGatherShards(index, openShard, "T");
  // prepareModel は非対応 op / 格納 dtype と IR 契約を全件見る（GPU は触らない）。
  const prepared = prepareModel(graphShard);
  assertEquals(prepared.graph.outputs, ["per_layer"]);
  assertEquals(prepared.graph.symbols, ["M"]);
});

Deno.test("常駐バイト数: 索引だけで決まる（E2B QAT の実値）", () => {
  const e2b: Gemma4PleIndex = { ...index, tokens: 262144, layers: 35, dim: 256 };
  assertEquals(gemma4PleGpuBytes(e2b), {
    values: 262144 * 35 * 256 / 2,
    scales: 262144 * 35 * 4,
    total: 262144 * 35 * 256 / 2 + 262144 * 35 * 4,
  });
  // i8 sidecar（通常 Gemma 4）は同じ索引で倍になる。
  assertEquals(gemma4PleGpuBytes({ ...e2b, storage: undefined }).values, 262144 * 35 * 256);
  assertEquals(gemma4PleGpuBytes({ ...e2b, storage: "i2" }).values, 262144 * 35 * 256 / 4);
});

Deno.test("合成コンテナ: piece が順に並び、繋ぎ直すと原本の values と一致する", async () => {
  const { graphShard, weightShards } = await buildGemma4PleGatherShards(index, openShard, "T");
  const graph = parseSafetensors(graphShard.bytes.buffer, graphShard.bytes.byteLength);
  // scale は piece 1 と同じ shard に置く（ADR 0090 決定 1 の co-shard）。
  const scale = graph.tensors.get("ple_scale");
  assertEquals(scale?.dtype, "F32");
  assertEquals(scale?.shape, [12, 1]);
  const scales = new Float32Array(
    graph.buffer.slice(scale!.byteOffset, scale!.byteOffset + scale!.byteLength),
  );
  // shard 順に連結した scale が、平坦添字 `id * layers + layer` の並びになっている。
  assertEquals([...scales], Array.from({ length: 12 }, (_, at) => (at + 1) / 64));
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
  assertEquals(collected.map((piece) => piece.byteLength / ROW_BYTES), [1, 7, 4]);
  const joined = new Uint8Array(12 * ROW_BYTES);
  let cursor = 0;
  for (const piece of collected) {
    joined.set(piece, cursor);
    cursor += piece.byteLength;
  }
  const expected = new Uint8Array(12 * ROW_BYTES);
  let at = 0;
  for (const shard of index.shards) {
    const file = parseSafetensors(writeShard(shard));
    const view = file.tensors.get("values")!;
    expected.set(new Uint8Array(file.buffer, view.byteOffset, view.byteLength), at);
    at += view.byteLength;
  }
  assertEquals(joined, expected);
});

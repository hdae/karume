// shard 逐次面（ADR 0070 決定 3）の Session 構築 — 全量面との A/B・失敗の transaction 境界。
//
// 受入①（ADR 0070）: 同一資産を全量面（createSession）と shard 面（createSessionFromShards）で
// 開いたとき、GPU 常駐バイト列が一致すること。常駐バッファは直接読めない（MAP_READ を持たない）
// ため、検出器は「全重みを通る run の出力の**ビット同一**」+「storage 診断の一致」で張る —
// 重みのバイトが 1 つでも違えば dequant 後の内積が変わり、出力のビット列に必ず出る。
//
// MUST: 重みは f32 / f16 / i8 / i4 の 4 格納を混在させる（分岐ごとに経路が違う — どれか 1 つ
// でも shard 経路で欠けると、この門が沈黙する）。

import { assert, assertEquals, assertRejects } from "@std/assert";
import { ContainerError, openModel } from "../src/format/container.ts";
import { SafetensorsError } from "../src/format/safetensors.ts";
import { acquireGpu, popFailureScopes, pushFailureScopes } from "../src/gpu/device.ts";
import {
  createSession,
  createSessionFromShards,
  type ModelShard,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { buildSafetensors, type TensorSpec } from "./helpers/format.ts";
import { fill } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { quantizeF16 } from "./helpers/f16.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const VARYING = (i: number): number => (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);

/**
 * linear 3 段（w1 = i4 g16 / w2 = f16 / w3 = i8）+ f32 bias 群のモデル素材。
 * 4 つの格納 dtype が全て「圧縮のまま常駐」の適格になる形（消費は linear の重みスロットのみ）。
 */
const buildFixture = () => {
  const w1 = fill([16, 32], VARYING);
  const w2 = fill([8, 16], VARYING);
  const w3 = fill([4, 8], VARYING);
  const q1 = quantizeI4(w1.data, w1.shape, 16);
  const q2 = quantizeF16(w2.data);
  const q3 = quantizeI8(w3.data, w3.shape, 0);
  const graph = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 32] }],
    outputs: ["y"],
    initializers: {
      w1: { tensor: "m.w1", storage: { dtype: "i4", scale: "m.s1", group_size: 16 } },
      w2: { tensor: "m.w2", storage: { dtype: "f16" } },
      w3: { tensor: "m.w3", storage: { dtype: "i8", scale: "m.s3" } },
      b1: { tensor: "m.b1", storage: { dtype: "f32" } },
      b2: { tensor: "m.b2", storage: { dtype: "f32" } },
      b3: { tensor: "m.b3", storage: { dtype: "f32" } },
    },
    values: {
      w1: { dtype: "f32", shape: [16, 32] },
      w2: { dtype: "f32", shape: [8, 16] },
      w3: { dtype: "f32", shape: [4, 8] },
      b1: { dtype: "f32", shape: [16] },
      b2: { dtype: "f32", shape: [8] },
      b3: { dtype: "f32", shape: [4] },
      h1: { dtype: "f32", shape: [2, 16] },
      h2: { dtype: "f32", shape: [2, 8] },
      y: { dtype: "f32", shape: [2, 4] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w1", "b1"], outs: ["h1"], attrs: {} },
      { op: "linear", ins: ["h1", "w2", "b2"], outs: ["h2"], attrs: {} },
      { op: "linear", ins: ["h2", "w3", "b3"], outs: ["y"], attrs: {} },
    ],
  };
  const metadata = { karume_ir: JSON.stringify(graph) };
  const f32Tensor = (name: string, filled: ReturnType<typeof fill>): TensorSpec => ({
    name,
    dtype: "F32",
    shape: [...filled.shape],
    data: new Uint8Array(filled.data.buffer),
  });
  const biases = [
    f32Tensor("m.b1", fill([16], SIGNED)),
    f32Tensor("m.b2", fill([8], SIGNED)),
    f32Tensor("m.b3", fill([4], SIGNED)),
  ];
  // 各 shard 内の並びは整列降順（F32 → I4 → F16 → I8 — 先頭 offset の整列規則を満たす並び）
  const s1: TensorSpec = {
    name: "m.s1",
    dtype: "F32",
    shape: [...q1.scaleShape],
    data: new Uint8Array(q1.scale.buffer),
  };
  const s3: TensorSpec = {
    name: "m.s3",
    dtype: "F32",
    shape: [...q3.scaleShape],
    data: new Uint8Array(q3.scale.buffer),
  };
  const t = {
    w1: { name: "m.w1", dtype: "I4", shape: [16, 32], data: q1.bytes } satisfies TensorSpec,
    w2: { name: "m.w2", dtype: "F16", shape: [8, 16], data: q2.bytes } satisfies TensorSpec,
    w3: { name: "m.w3", dtype: "I8", shape: [4, 8], data: q3.bytes } satisfies TensorSpec,
    s1,
    s3,
  };
  return {
    metadata,
    biases,
    tensors: t,
    x: fill([2, 32], SIGNED),
    /** 全量面のファイル（グラフ + 全テンソル）。 */
    fullBuffer: (): ArrayBuffer =>
      buildSafetensors([...biases, s1, s3, t.w1, t.w2, t.w3], metadata),
    /** 既定の 3 分割（graph shard = bias 群 / shard1 = w1+s1 / shard2 = w3+s3+w2）。 */
    shards: (): ArrayBuffer[] => [
      buildSafetensors(biases, metadata),
      buildSafetensors([s1, t.w1], undefined),
      buildSafetensors([s3, t.w2, t.w3], undefined),
    ],
  };
};

/**
 * ArrayBuffer 列を shard 面の入力（実名 + tight view の逐次列）にする。id は既定で
 * 「配布形のファイル名らしい実名」を振る（帰属の検出器が連番と取り違えないため）。
 */
const shardStream = async function* (
  buffers: readonly ArrayBuffer[],
  ids?: readonly string[],
): AsyncGenerator<ModelShard, void, unknown> {
  for (const [index, buffer] of buffers.entries()) {
    yield {
      id: ids?.[index] ?? `fixture/model-0000${index}.safetensors`,
      bytes: new Uint8Array(buffer),
    };
  }
};

const bitsOf = (tensor: Tensor): readonly number[] => [
  ...new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length),
];

Deno.test({
  name: "shard 面は全量面と出力ビット同一・storage 診断一致（f32/f16/i8/i4 混在・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const gpu = await acquireGpu();
    try {
      const whole = await createSession(gpu, openModel(fixture.fullBuffer()));
      const sharded = await createSessionFromShards(gpu, shardStream(fixture.shards()));
      try {
        const wholeOut = (await whole.run({ x: fixture.x }))["y"];
        const shardedOut = (await sharded.run({ x: fixture.x }))["y"];
        assertEquals(bitsOf(shardedOut), bitsOf(wholeOut), "出力がビット同一でない");
        // 常駐の内訳（圧縮常駐 / CPU 展開）は 2 面で同じバイト数になる（受入①の第 2 の検出器）
        assertEquals(sharded.diagnostics().storage, whole.diagnostics().storage);
        assert(whole.diagnostics().storage.residentCompressedBytes > 0, "圧縮常駐が 1 本も無い");
      } finally {
        await whole.dispose();
        await sharded.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "shard 面は分割粒度に依存しない（全テンソル入りグラフ shard 1 本でも同一・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const gpu = await acquireGpu();
    try {
      const whole = await createSession(gpu, openModel(fixture.fullBuffer()));
      // 全量ファイルは「グラフ shard に全重みが同居した 1 shard の列」としても合法
      const single = await createSessionFromShards(gpu, shardStream([fixture.fullBuffer()]));
      try {
        const wholeOut = (await whole.run({ x: fixture.x }))["y"];
        const singleOut = (await single.run({ x: fixture.x }))["y"];
        assertEquals(bitsOf(singleOut), bitsOf(wholeOut));
      } finally {
        await whole.dispose();
        await single.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "shard 由来の失敗は連番でなく資産名（id）を名乗る（宣言違反 / parse 不能・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    // 「連番と取り違えない」ことが見えるよう、実名側は配布形のファイル名にする。
    const ids = [
      "fixture-v1.0/transformer/model-00001.safetensors",
      "fixture-v1.0/transformer/model-00002.safetensors",
      "fixture-v1.0/transformer/model-00003.safetensors",
    ];
    const gpu = await acquireGpu();
    try {
      // 宣言違反: どの initializer からも参照されない余剰テンソルが 3 本目に混ざる
      const surplus: TensorSpec = {
        name: "m.unused",
        dtype: "F32",
        shape: [4],
        data: new Uint8Array(new Float32Array([1, 2, 3, 4]).buffer),
      };
      const violation = await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            shardStream([
              buildSafetensors(fixture.biases, fixture.metadata),
              buildSafetensors([fixture.tensors.s1, fixture.tensors.w1], undefined),
              buildSafetensors(
                [fixture.tensors.s3, surplus, fixture.tensors.w2, fixture.tensors.w3],
                undefined,
              ),
            ], ids),
          ),
        ContainerError,
        "m.unused",
      );
      assert(violation.message.includes(ids[2]), violation.message);
      // 帰属は**落ちた shard** 1 本（別の shard の名前が混ざったら帰属が壊れている）
      assert(!violation.message.includes(ids[1]), violation.message);
      // 連番は補助として残す（届いた順は id と別の情報）
      assert(violation.message.includes("shard [2]"), violation.message);

      // parse 不能: safetensors ですらないバイト列が 2 本目に来る（帰属を足してもパーサ門の
      // クラスは保つ — 包み直すと呼び出し側の分岐が壊れる）
      const broken = await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            shardStream([
              buildSafetensors(fixture.biases, fixture.metadata),
              new Uint8Array(64).buffer,
            ], ids),
          ),
        SafetensorsError,
      );
      assert(broken.message.includes(ids[1]), broken.message);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "shard 面の失敗は transaction 境界で閉じる（違反列挙 + スコープ残高 + 再構築可・実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const fixture = buildFixture();
    const gpu = await acquireGpu();
    try {
      // 空列（グラフ shard すら無い）
      await assertRejects(
        () => createSessionFromShards(gpu, shardStream([])),
        ExecutionError,
        "shard 列が空",
      );
      // 最初の shard がグラフ shard でない
      await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            shardStream([buildSafetensors([fixture.tensors.s1, fixture.tensors.w1], undefined)]),
          ),
        ContainerError,
        "karume_ir",
      );
      // グラフ shard の重複（後続 shard に karume_ir が現れる）
      await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            shardStream([
              buildSafetensors(fixture.biases, fixture.metadata),
              buildSafetensors([fixture.tensors.s1, fixture.tensors.w1], fixture.metadata),
            ]),
          ),
        ExecutionError,
        "グラフ shard が複数",
      );
      // co-shard 分断（w1 と companion scale s1 が別 shard — ADR 0070 決定 1 の MUST）
      await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            shardStream([
              buildSafetensors(fixture.biases, fixture.metadata),
              buildSafetensors([fixture.tensors.w1], undefined),
              buildSafetensors(
                [fixture.tensors.s1, fixture.tensors.s3, fixture.tensors.w2, fixture.tensors.w3],
                undefined,
              ),
            ]),
          ),
        ContainerError,
      );
      // 欠け（最終 shard を渡し忘れる — 完全性は全 shard 読了後に全件列挙）
      const missing = await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            shardStream([
              buildSafetensors(fixture.biases, fixture.metadata),
              buildSafetensors([fixture.tensors.s1, fixture.tensors.w1], undefined),
            ]),
          ),
        ContainerError,
      );
      assert(missing.message.includes("m.w2"), missing.message);
      assert(missing.message.includes("m.w3"), missing.message);
      // 非 tight view（slice の混入 — RAM ピーク倍増の防波堤）
      const loose = new Uint8Array(new ArrayBuffer(16), 4, 8);
      await assertRejects(
        () =>
          createSessionFromShards(
            gpu,
            (async function* () {
              yield { id: "fixture/loose.safetensors", bytes: loose as Uint8Array<ArrayBuffer> };
            })(),
          ),
        ExecutionError,
        "buffer 全体",
      );

      // 途中失敗の後で errorScope が積み残されていないこと（積み残すと以後の検証結果が
      // 誤ったスコープに吸われ、次の失敗が恒久的に見えなくなる）
      pushFailureScopes(gpu.device);
      assertEquals(await popFailureScopes(gpu.device, "残高検査"), undefined);

      // 同じ device で正常構築 → 実行までできる（部分 Session や壊れた常駐が残っていない）
      const session = await createSession(gpu, openModel(fixture.fullBuffer()));
      try {
        const output = (await session.run({ x: fixture.x }))["y"];
        assertEquals(output.shape, [2, 4]);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

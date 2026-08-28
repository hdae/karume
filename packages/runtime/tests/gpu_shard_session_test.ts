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
import { createSession, createSessionFromShards, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { buildSafetensors, type TensorSpec } from "./helpers/format.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { buildFixture, shardStream } from "./helpers/shard-fixture.ts";

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

import { assert, assertEquals, assertInstanceOf, assertRejects } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, RUNTIME_INTERNAL } from "../src/gpu/device.ts";
import {
  GpuValidationError,
  popFailureScopes,
  pushFailureScopes,
  withPipelineScope,
} from "../src/gpu/error-scope.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { applyReferenceOp } from "../src/reference/ops.ts";
import { createSession } from "../src/runtime/executor.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const wgslWithWorkgroupSize = (size: number): string => `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(${size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x] + 1.0; }
}
`;

Deno.test({
  name: "workgroup サイズ超過のパイプラインは errorScope 経由で例外になる（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache = new PipelineCache(gpu.device);
    try {
      const tooLarge = gpu.limits.maxComputeWorkgroupSizeX + 1;

      // 生成そのものは同期例外にならない（沈黙 no-op 化の入口）。可視化は errorScope のみ。
      let threwSynchronously = false;
      gpu.device.pushErrorScope("validation");
      try {
        const module = gpu.device.createShaderModule({ code: wgslWithWorkgroupSize(tooLarge) });
        gpu.device.createComputePipeline({ layout: "auto", compute: { module } });
      } catch {
        threwSynchronously = true;
      }
      const raw = await gpu.device.popErrorScope();
      assert(threwSynchronously || raw !== null, "無効パイプラインが完全に無検出のまま通った");

      // PipelineCache 経由なら必ず例外になる（沈黙 no-op の検出網）
      const error = await assertRejects(
        () => cache.get("test:too-large", wgslWithWorkgroupSize(tooLarge)),
        GpuValidationError,
        "test:too-large",
      );
      assert(error.message.length > "test:too-large".length, "原因メッセージが保持される");
      assertEquals(cache.size, 0, "失敗したパイプラインはキャッシュしない");

      // 直前の失敗で errorScope が積み残されていないこと（後続の検証が誤ったスコープに
      // 吸われると、以後の validation エラーが恒久的に見えなくなる）
      const valid = await cache.get("test:valid", wgslWithWorkgroupSize(64));
      assert(valid !== undefined);
      assertEquals(cache.size, 1);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "withPipelineScope は body の同期例外でも 2 本とも pop する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      // 積み残し / 過剰 pop はスコープの中身を見ても分からない（自分の push と pop は LIFO で
      // 必ず対応する）ので、**外側の番兵スコープ**で間接的に門にする。番兵には先にエラーを
      // 1 つ入れておき、withPipelineScope の後にそれを取り返せるかを見る。
      const { layout } = await new PipelineCache(gpu.device).get(
        "pipeline-scope-probe",
        wgslWithWorkgroupSize(64),
      );
      gpu.device.pushErrorScope("validation");
      // binding 0 が欠落した bindGroup は同期例外にならず errorScope にだけ現れる
      gpu.device.createBindGroup({ layout, entries: [] });

      const thrown = new Error("body 側の失敗");
      const caught = await assertRejects(() =>
        withPipelineScope(gpu.device, "body-throws", () => {
          throw thrown;
        })
      );
      assertEquals(caught, thrown, "body の例外はそのまま伝播する");

      // 1 本しか pop していなければ、この pop が取るのは積み残された空のスコープ（= null）。
      // 3 本 pop していれば番兵は既に消えており、この pop 自体が失敗して落ちる。
      const sentinel = await gpu.device.popErrorScope();
      assert(
        sentinel !== null,
        "番兵スコープのエラーが取り返せない = withPipelineScope がスコープを積み残した",
      );
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "無効パイプラインは internal ではなく validation として返る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const cache = new PipelineCache(gpu.device);
    try {
      // internal エラー自体は実 GPU では作れないため、2 本張りの導入で「validation だった
      // ものが internal 扱いに化けない」ことだけを門にする（優先順位の誤りの検出）。
      const tooLarge = gpu.limits.maxComputeWorkgroupSizeX + 1;
      const error = await assertRejects(
        () => cache.get("test:internal-misfire", wgslWithWorkgroupSize(tooLarge)),
        GpuValidationError,
      );
      assertEquals(error.name, "GpuValidationError", "internal 分岐が誤爆している");
      assertEquals(cache.size, 0, "失敗したパイプラインはキャッシュしない");

      // 失敗経路でも 2 本とも pop されていること（積み残しがあると後続の検証が誤ったスコープに
      // 吸われる）。直後の正常な生成が通ることで間接的に確かめる。
      const valid = await cache.get("test:internal-misfire-after", wgslWithWorkgroupSize(64));
      assert(valid !== undefined);
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "同一 device 上の複数 Session の並行 run が互いに干渉せず完走する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    // run のスコープ区間は device 単位ロックで直列化される。ロック内から呼ばれる層
    // （PipelineCache 等）がロックを再取得すると自己デッドロックになるため、並行 run が
    // ハングせず完走すること自体が回帰の対象になる。
    const sessions = await Promise.all(
      (["relu", "neg"] as const).map((op) =>
        createSession(gpu, openModel(graphModelBuffer(singleOpGraph(op, [[8, 8]], [[8, 8]]))))
      ),
    );
    try {
      const x = fill([8, 8], (i) => (i % 13) - 6);

      // await を挟まずに発行する（Session 間で区間が重なる形）
      const [relu, neg] = await Promise.all(sessions.map((session) => session.run({ x0: x })));

      assertEquals([...relu.y.data], [...applyReferenceOp("relu", [x]).data]);
      assertEquals([...neg.y.data], [...applyReferenceOp("neg", [x]).data]);
    } finally {
      await Promise.all(sessions.map((session) => session.dispose()));
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "Session.run は自分のスコープ区間を device 単位ロックで守る（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    // プリミティブ（GpuContext.withScopeLock）の契約は gpu_device_test が固定しているが、
    // それを Session.run が**使っている**という結線はここでしか固定されない。executor から
    // ラップが外れる（あるいは readback / flush をロック外へ動かす）リファクタは、この 1 本が
    // 無いと全緑のまま通り、m0-review が major とした誤帰属がそのまま復活する。
    const gpu = await acquireGpu();
    const session = await createSession(
      gpu,
      openModel(graphModelBuffer(singleOpGraph("relu", [[8, 8]], [[8, 8]]))),
    );
    try {
      const { layout } = await new PipelineCache(gpu.device).get(
        "run-scope-lock-probe",
        wgslWithWorkgroupSize(64),
      );
      const x = fill([8, 8], (i) => (i % 13) - 6);

      // MUST: faulty 側を**先に**開かせる。pop はスタック先頭を無条件に取るので、誤帰属は
      // 「faulty が先に push し先に pop する」重なりでのみ起きる（run 先行だと両者とも正しい
      // 側に帰属し、ロックが外れても緑になってしまう）。
      const faulty = gpu[RUNTIME_INTERNAL].withScopeLock(async () => {
        pushFailureScopes(gpu.device);
        // binding 0 が欠落した bindGroup は同期例外にならず errorScope にだけ現れる
        gpu.device.createBindGroup({ layout, entries: [] });
        for (let i = 0; i < 4; i += 1) await Promise.resolve();
        return await popFailureScopes(gpu.device, "faulty");
      });
      const running = session.run({ x0: x });

      const [faultyResult, outputs] = await Promise.all([faulty, running]);

      assertInstanceOf(faultyResult, GpuValidationError, "エラーを起こした側が捕捉する");
      assert(faultyResult.message.startsWith("faulty:"), faultyResult.message);
      // ロックが外れると run が faulty のエラーで落ち（本来通るはずの run が他人のメッセージで
      // 落ちる）、faulty は自分の失敗を取り逃がす。
      assertEquals([...outputs.y.data], [...applyReferenceOp("relu", [x]).data]);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

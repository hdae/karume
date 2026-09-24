import { assert, assertEquals, assertRejects } from "@std/assert";
import { acquireGpu, createSessionFromContainer } from "../mod.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, fill, openGraphModel } from "./helpers/model-fixture.ts";
const graph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  symbols: [],
  requires: { ops: ["relu", "neg"] },
  inputs: [{ name: "x", dtype: "f32", shape: [4] }],
  outputs: ["a", "b"],
  initializers: {},
  values: { a: { dtype: "f32", shape: [4] }, b: { dtype: "f32", shape: [4] } },
  nodes: [{ op: "relu", ins: ["x"], outs: ["a"], attrs: {} }, {
    op: "neg",
    ins: ["a"],
    outs: ["b"],
    attrs: {},
  }],
});
for (const fail of [false, true]) {
  Deno.test({
    name: `レシピ並列準備: ${fail ? "失敗時も全件を待つ" : "完了順が逆でも依存順に実行する"}`,
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const gpu = await acquireGpu();
      const session = await createSessionFromContainer(gpu, await openGraphModel(graph()), "model");
      const original = gpu.device.createComputePipelineAsync.bind(gpu.device);
      const gate = Promise.withResolvers<void>(),
        both = Promise.withResolvers<void>();
      let count = 0, settled = false;
      const failure = new RangeError("injected compile failure");
      gpu.device.createComputePipelineAsync = async (descriptor) => {
        const index = count++;
        if (count === 2) both.resolve();
        if (fail && index === 0) throw failure;
        if (index === (fail ? 1 : 0)) await gate.promise;
        return await original(descriptor);
      };
      const inputs = { x: fill([4], (i) => i - 1) };
      const running = session.run(inputs);
      const checked = running.then(() => {
        settled = true;
      }, () => {
        settled = true;
      });
      let timeout: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          both.promise,
          new Promise<never>((_, reject) => {
            timeout = setTimeout(
              () => reject(Error("compile requests were serialized")),
              2000,
            );
          }),
        ]);
        assertEquals(count, 2);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assertEquals(settled, false, "未決着compileを残したままrunを返さない");
        gate.resolve();
        if (fail) {
          const error = await assertRejects(
            () => running,
            RangeError,
            "injected compile failure",
          );
          assertEquals(error, failure);
          assertEquals(session.diagnostics().lastRunPrepared, undefined);
        } else {
          const outputs = await running;
          assert(outputs.a.dtype === "f32" && outputs.b.dtype === "f32");
          assertEquals(Array.from(outputs.a.data), [0, 0, 1, 2]);
          assertEquals(Array.from(outputs.b.data), [-0, -0, -1, -2]);
        }
      } finally {
        if (timeout !== undefined) clearTimeout(timeout);
        gate.resolve();
        await checked;
        gpu.device.createComputePipelineAsync = original;
        await session.dispose();
        gpu.destroy();
      }
    },
  });
}

import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { openModel } from "../src/format/container.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession } from "../src/runtime/executor.ts";
import { fill, graphModelBuffer, singleOpGraph } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { permutedSourceIndices } from "./helpers/permute.ts";

describe({
  name: "要素順を保つpermuteの共有と実体化（実GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    for (const dtype of ["f32", "i32", "bool"] as const) {
      it(
        dtype === "f32"
          ? "f32の語・出力shapeを保ち、必要な並べ替えだけdispatchする"
          : `${dtype}は別名化する形でも既存の非対応dtype門で拒否する`,
        async () => {
          const gpu = await acquireGpu();
          try {
            for (const shape of [[1, 3, 1, 5], [1, 1, 4, 9], [1, 3, 4, 5]]) {
              const dims = [0, 2, 1, 3], outShape = dims.map((axis) => shape[axis]);
              const graph = singleOpGraph("permute", [shape], [outShape], {
                attrs: { dims },
                inDtypes: [dtype],
                outDtypes: [dtype],
              });
              // 現runtimeのpermuteはf32のみ。別名にできてもcapability門は迂回しない。
              if (dtype !== "f32") {
                await assertRejects(
                  () => createSession(gpu, openModel(graphModelBuffer(graph))),
                  Error,
                  "非対応 意味論 dtype",
                );
                continue;
              }
              // 全区間sliceは語を変えずに内部の実体を確保する。
              graph.requires.ops.push("slice");
              graph.values.h = { dtype: "f32", shape };
              graph.nodes[0].ins = ["h"];
              graph.nodes.unshift({
                op: "slice",
                ins: ["x0"],
                outs: ["h"],
                attrs: { dim: 0, start: 0, end: shape[0] },
              });
              const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
              try {
                for (const repeat of [0, 1]) {
                  const input = fill(
                    shape,
                    (i) => i * 3 + repeat,
                    dtype,
                  );
                  const words = new Uint32Array(input.data.buffer);
                  if (repeat === 0) {
                    words.set([0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc00123]);
                  }
                  const before = session.diagnostics().submit.dispatchCount;
                  const y = (await session.run({ x0: input })).y;
                  assertEquals(y.shape, outShape);
                  assertEquals(
                    new Uint32Array(y.data.buffer, y.data.byteOffset, y.data.length),
                    Uint32Array.from(permutedSourceIndices(shape, dims), (i) => words[i]),
                  );
                  assertEquals(
                    session.diagnostics().submit.dispatchCount - before,
                    shape[1] === 1 || shape[2] === 1 ? 1 : 2,
                  );
                }
              } finally {
                await session.dispose();
              }
            }
          } finally {
            gpu.destroy();
          }
        },
      );
    }

    it("別名の出力と共有元を後段まで保持し、再実行でも壊さない", async () => {
      const gpu = await acquireGpu(), shape = [1, 3, 1, 5], permuted = [1, 1, 3, 5];
      const graph = singleOpGraph("permute", [shape], [permuted], {
        attrs: { dims: [0, 2, 1, 3] },
      });
      graph.requires.ops.push("neg");
      graph.values.h = { dtype: "f32", shape };
      graph.values.back = { dtype: "f32", shape };
      graph.values.late = { dtype: "f32", shape };
      graph.nodes[0].ins = ["h"];
      graph.nodes.unshift({ op: "neg", ins: ["x0"], outs: ["h"], attrs: {} });
      graph.nodes.push(
        { op: "permute", ins: ["y"], outs: ["back"], attrs: { dims: [0, 2, 1, 3] } },
        { op: "neg", ins: ["h"], outs: ["late"], attrs: {} },
      );
      graph.outputs.push("back", "late");
      try {
        const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
        try {
          for (const repeat of [0, 1, 2]) {
            const input = fill(shape, (i) => i + repeat + .5);
            const before = session.diagnostics().submit.dispatchCount;
            const outputs = await session.run({ x0: input });
            assertEquals(outputs.y.shape, permuted);
            for (const key of ["y", "back"]) {
              assertEquals(Array.from(outputs[key].data), Array.from(input.data, (x) => -x));
            }
            assertEquals(outputs.late.data, input.data);
            assertEquals(session.diagnostics().submit.dispatchCount - before, 2);
          }
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    });

    it("入力由来のpermute出力は実体を保ち、同じ常駐入力へ書き戻せる", async () => {
      const gpu = await acquireGpu(), shape = [1, 2, 1, 4], permuted = [1, 1, 2, 4];
      const graph = singleOpGraph("permute", [shape], [permuted], {
        attrs: { dims: [0, 2, 1, 3] },
      });
      graph.requires.ops.push("reshape");
      graph.values.before = { dtype: "f32", shape };
      graph.values.after = { dtype: "f32", shape };
      graph.nodes[0].ins = ["before"];
      graph.nodes.unshift({ op: "reshape", ins: ["x0"], outs: ["before"], attrs: {} });
      graph.nodes.push({ op: "reshape", ins: ["y"], outs: ["after"], attrs: {} });
      graph.outputs = ["after"];
      const state = await gpu.createResident(32, "permute input and output");
      try {
        const session = await createSession(gpu, openModel(graphModelBuffer(graph)));
        try {
          for (const repeat of [0, 1]) {
            const values = Float32Array.from({ length: 8 }, (_, i) => i + repeat + .5);
            state.write(values);
            const batch = await gpu.beginBatch();
            try {
              await session.enqueue({ x0: state }, { batch, copyOutputs: { after: state } });
            } finally {
              await batch.finish();
            }
            assertEquals(new Float32Array(await state.read()), values);
          }
          assertEquals(session.diagnostics().submit.dispatchCount, 2);
        } finally {
          await session.dispose();
        }
      } finally {
        state.dispose();
        gpu.destroy();
      }
    });

    it("常駐initializerのコピーを維持し、複数runで重みを破棄しない", async () => {
      const gpu = await acquireGpu(), shape = [1, 2, 1, 4], outShape = [1, 1, 2, 4];
      const graph = singleOpGraph("permute", [shape], [outShape], {
        attrs: { dims: [0, 2, 1, 3] },
      });
      graph.inputs = [];
      graph.initializers.x0 = { tensor: "weight", storage: { dtype: "f32" } };
      graph.values.x0 = { dtype: "f32", shape };
      const values = Float32Array.from({ length: 8 }, (_, i) => i + .5);
      try {
        const session = await createSession(
          gpu,
          openModel(graphModelBuffer(graph, [
            { name: "weight", dtype: "F32", shape, data: new Uint8Array(values.buffer) },
          ])),
        );
        try {
          for (const _repeat of [0, 1]) {
            const output = await session.run({});
            assertEquals(output.y.shape, outShape);
            assertEquals(output.y.data, values);
          }
          assertEquals(session.diagnostics().submit.dispatchCount, 2);
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    });
  },
});

import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { permutedSourceIndices } from "./helpers/permute.ts";

describe.skipIf(!GPU_AVAILABLE)(
  "sharing and materialization of order-preserving permutes (real GPU)",
  () => {
    for (const dtype of ["f32", "i32", "bool"] as const) {
      it(
        dtype === "f32"
          ? "keeps the f32 words and output shape, and dispatches only the reordering that is needed"
          : `${dtype} is rejected by the existing unsupported-dtype check even in a form that would alias`,
        async () => {
          const gpu = await acquireGpu();
          try {
            for (const shape of [
              [1, 3, 1, 5],
              [1, 1, 4, 9],
              [1, 3, 4, 5],
            ]) {
              const dims = [0, 2, 1, 3],
                outShape = dims.map((axis) => shape[axis]);
              const graph = singleOpDeclaration("permute", [shape], [outShape], {
                attrs: { dims },
                inDtypes: [dtype],
                outDtypes: [dtype],
              });
              // The current runtime's permute is f32 only. Even when it can alias, the capability check is not bypassed.
              if (dtype !== "f32") {
                const opened = await openModelBytes(graph, []);
                await assertRejects(
                  () => createSessionFromContainer(gpu, opened, GRAPH_NAME),
                  RuntimeSupportError,
                  "unsupported semantic dtype",
                );
                continue;
              }
              // A full-range slice materializes its own buffer without changing the words.
              graph.requires.ops.push("slice");
              graph.values.h = { dtype: "f32", shape };
              graph.nodes[0].ins = ["h"];
              graph.nodes.unshift({
                op: "slice",
                ins: ["x0"],
                outs: ["h"],
                attrs: { dim: 0, start: 0, end: shape[0] },
              });
              const session = await createSessionFromContainer(
                gpu,
                await openModelBytes(graph, []),
                GRAPH_NAME,
              );
              try {
                for (const repeat of [0, 1]) {
                  const input = fill(shape, (i) => i * 3 + repeat, dtype);
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

    it("keeps the aliased output and its source alive to later stages, and does not corrupt them on re-run", async () => {
      const gpu = await acquireGpu(),
        shape = [1, 3, 1, 5],
        permuted = [1, 1, 3, 5];
      const graph = singleOpDeclaration("permute", [shape], [permuted], {
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
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, []),
          GRAPH_NAME,
        );
        try {
          for (const repeat of [0, 1, 2]) {
            const input = fill(shape, (i) => i + repeat + 0.5);
            const before = session.diagnostics().submit.dispatchCount;
            const outputs = await session.run({ x0: input });
            assertEquals(outputs.y.shape, permuted);
            for (const key of ["y", "back"]) {
              assertEquals(
                Array.from(outputs[key].data),
                Array.from(input.data, (x) => -x),
              );
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

    it("a permute output derived from an input keeps its own buffer and can be written back to the same resident input", async () => {
      const gpu = await acquireGpu(),
        shape = [1, 2, 1, 4],
        permuted = [1, 1, 2, 4];
      const graph = singleOpDeclaration("permute", [shape], [permuted], {
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
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, []),
          GRAPH_NAME,
        );
        try {
          for (const repeat of [0, 1]) {
            const values = Float32Array.from({ length: 8 }, (_, i) => i + repeat + 0.5);
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

    it("keeps the copy of a resident initializer and does not destroy weights across multiple runs", async () => {
      const gpu = await acquireGpu(),
        shape = [1, 2, 1, 4],
        outShape = [1, 1, 2, 4];
      const graph = singleOpDeclaration("permute", [shape], [outShape], {
        attrs: { dims: [0, 2, 1, 3] },
      });
      graph.inputs = [];
      graph.initializers.x0 = {};
      graph.values.x0 = { dtype: "f32", shape };
      const values = Float32Array.from({ length: 8 }, (_, i) => i + 0.5);
      try {
        const session = await createSessionFromContainer(
          gpu,
          await openModelBytes(graph, [
            {
              graph: GRAPH_NAME,
              initializer: "x0",
              bytes: new Uint8Array(values.buffer),
              encoding: { codec: "f32" },
            },
          ]),
          GRAPH_NAME,
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
);

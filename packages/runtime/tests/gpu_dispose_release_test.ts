// A disposal ends with the release fence: one empty submit made after its last buffer destroy, and
// the wait for it (SubmitScheduler.releaseDestroyed). On Dawn's Vulkan backend a destroyed buffer's
// memory is returned only when a submit made after the destroy has completed, so without the fence a
// caller that disposes one model and builds the next on the same device allocates the second on top
// of the first.
//
// What is checked is the order of the calls on a real device, not the memory: WebGPU does not expose
// how much a device holds. (Measured once by hand with the adapter's own tool — the commit that
// added the fence has the numbers.)
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { traceQueue } from "./helpers/queue-trace.ts";

const SHAPE = [4, 8];

const NEG_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: SHAPE }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: SHAPE } },
  nodes: [{ op: "neg", ins: ["x"], outs: ["y"], attrs: {} }],
};

test.skipIf(!GPU_AVAILABLE)(
  "Session.dispose ends with one submit made after its last destroy, and waits for it",
  async () => {
    const gpu = await acquireGpu();
    try {
      const session = await createSessionFromContainer(
        gpu,
        await openModelBytes(NEG_GRAPH, []),
        GRAPH_NAME,
      );
      await session.run({ x: fill(SHAPE, (i) => i) });
      const trace = traceQueue(gpu);
      try {
        await session.dispose();
      } finally {
        trace.restore();
      }
      const lastDestroy = trace.log.lastIndexOf("destroy");
      assert(lastDestroy >= 0, `the disposal destroyed a buffer (${trace.log.join(" ")})`);
      assertEquals(trace.log.slice(lastDestroy + 1), ["submit", "done"], trace.log.join(" "));
    } finally {
      gpu.destroy();
    }
  },
);

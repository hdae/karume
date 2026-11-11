/**
 * `stagedLoopWgsl` (src/kernels/rounding-barrier.ts), no GPU: the generated entry point's text and
 * the inputs it refuses. Whether the staged form acts as a rounding barrier on a backend is the GPU
 * test's question (gpu_rounding_barrier_test.ts).
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { type StagedLoop, stagedF32, stagedLoopWgsl } from "../src/kernels/rounding-barrier.ts";

const LOOP: StagedLoop = {
  workgroupSize: 64,
  count: "params.n",
  index: ["let row = i / params.dim;"],
  rounds: [
    [{ name: "product", value: "y[i] * gate[row]" }],
    [{ name: "scaled", value: `(bias[i] + ${stagedF32("product")}) * scale[row]` }],
  ],
  finish: [`out[i] = residual[i] + ${stagedF32("scaled")};`],
};

describe("stagedLoopWgsl", () => {
  it("writes each round's values to workgroup u32s, a barrier after every round and after the output", () => {
    assertEquals(
      stagedLoopWgsl(LOOP),
      `var<workgroup> staged_product: array<u32, 64>;
var<workgroup> staged_scaled: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * 64u;
  var base = wid.x * 64u;
  while (base < params.n) {
    let i = base + lid;
    if (i < params.n) {
      let row = i / params.dim;
      staged_product[lid] = bitcast<u32>(y[i] * gate[row]);
    }
    workgroupBarrier();
    if (i < params.n) {
      let row = i / params.dim;
      staged_scaled[lid] = bitcast<u32>((bias[i] + bitcast<f32>(staged_product[lid])) * scale[row]);
    }
    workgroupBarrier();
    if (i < params.n) {
      let row = i / params.dim;
      out[i] = residual[i] + bitcast<f32>(staged_scaled[lid]);
    }
    workgroupBarrier();
    base = base + stride;
  }
}
`,
    );
  });

  it("gives the same text for the same loop (the pipeline cache's determinism contract)", () => {
    assertEquals(stagedLoopWgsl(LOOP), stagedLoopWgsl({ ...LOOP, rounds: [...LOOP.rounds] }));
  });

  it("refuses a workgroup size outside 1 to 256 (the core limit)", () => {
    for (const workgroupSize of [0, 257, 1.5, Number.NaN]) {
      assertThrows(
        () => stagedLoopWgsl({ ...LOOP, workgroupSize }),
        CodegenError,
        "workgroup size",
        String(workgroupSize),
      );
    }
    stagedLoopWgsl({ ...LOOP, workgroupSize: 1 });
    stagedLoopWgsl({ ...LOOP, workgroupSize: 256 });
  });

  it("refuses a loop without rounds, an empty round and an empty finish", () => {
    assertThrows(() => stagedLoopWgsl({ ...LOOP, rounds: [] }), CodegenError, "at least one round");
    assertThrows(
      () => stagedLoopWgsl({ ...LOOP, rounds: [...LOOP.rounds, []] }),
      CodegenError,
      "no round is empty",
    );
    assertThrows(() => stagedLoopWgsl({ ...LOOP, finish: [] }), CodegenError, "finish");
  });

  it("refuses a name that is no WGSL identifier of its form, and a name used twice", () => {
    for (const name of ["", "Product", "1st", "a-b", "a b", "p;"]) {
      assertThrows(
        () => stagedLoopWgsl({ ...LOOP, rounds: [[{ name, value: "y[i]" }]] }),
        CodegenError,
        "not a staged value name",
        name,
      );
    }
    assertThrows(
      () =>
        stagedLoopWgsl({
          ...LOOP,
          rounds: [[{ name: "p", value: "y[i]" }], [{ name: "p", value: "y[i]" }]],
        }),
      CodegenError,
      "used twice",
    );
  });

  it("refuses staged values past the core workgroup storage (16 KiB: 16 values of 256 words)", () => {
    const values = (count: number) =>
      Array.from({ length: count }, (_, at) => ({ name: `v${at}`, value: "y[i]" }));
    stagedLoopWgsl({ ...LOOP, workgroupSize: 256, rounds: [values(16)] });
    assertThrows(
      () => stagedLoopWgsl({ ...LOOP, workgroupSize: 256, rounds: [values(17)] }),
      CodegenError,
      "workgroup storage",
    );
    // Each variable rounds up to 16 bytes: 1025 values of one word are 16400 bytes, not 4100
    stagedLoopWgsl({ ...LOOP, workgroupSize: 1, rounds: [values(1024)] });
    assertThrows(
      () => stagedLoopWgsl({ ...LOOP, workgroupSize: 1, rounds: [values(1025)] }),
      CodegenError,
      "workgroup storage",
    );
  });
});

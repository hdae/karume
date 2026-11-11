// Whether the staged grid-stride loop (stagedLoopWgsl — src/kernels/rounding-barrier.ts) acts as a
// rounding barrier on this backend: a kernel built from it writes the words of the unfused sequence
// (one dispatch per plain node, every intermediate through a storage buffer), and the same
// expression in one statement does not.
//
// Two forms: `r + y · g` (one round — mul → add) and `e + (c + a · b) · d` (two rounds, the second
// reading the first's staged value — mul → add → mul → add). Operand orders are the plain nodes'.
// Each runs with one workgroup per 256 elements and on three workgroups (the forced grid-stride),
// over a count that leaves the last workgroup partial. u32 identity, a NaN by class (the payload of
// a computed NaN is the backend's).
//
// MUST: the unstaged mutant differs from the plain sequence in at least one word, or the test
// fails. On a backend that does not contract these expressions the staged kernel would match
// with or without the barrier, and a zero-mismatch result would prove nothing about the helper;
// the case then needs an expression that backend does contract.
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import { stagedF32, stagedLoopWgsl } from "../src/kernels/rounding-barrier.ts";
import { wordIdentity } from "../src/tune/word-identity.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const WORKGROUP_SIZE = 256;
/** 37 full workgroups and a partial one. */
const COUNT = WORKGROUP_SIZE * 37 + 91;

/** `Params` and the bindings: the uniform, `inputs` read-only f32 arrays, then `out`. */
const header = (inputs: readonly string[]): string => `struct Params {
  n: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
${inputs.map((name, at) => `@group(0) @binding(${at + 1}) var<storage, read> ${name}: array<f32>;`).join("\n")}
@group(0) @binding(${inputs.length + 1}) var<storage, read_write> out: array<f32>;
`;

/** A grid-stride kernel writing `value` to every element, with no staging: a plain node, or a mutant. */
const unstaged = (inputs: readonly string[], value: string): string => `${header(inputs)}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${WORKGROUP_SIZE}u;
  var i = wid.x * ${WORKGROUP_SIZE}u + lid;
  while (i < params.n) {
    out[i] = ${value};
    i = i + stride;
  }
}
`;

const MUL = unstaged(["a", "b"], "a[i] * b[i]");
const ADD = unstaged(["a", "b"], "a[i] + b[i]");

const SPECIAL_WORDS = [
  0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 1, 0x80000001, 0x007fffff, 0x7f7fffff,
];

/** A deterministic operand: Gaussian values with the special words at its head. */
const operand = (seed: number): Uint32Array<ArrayBuffer> => {
  let state = seed >>> 0;
  const uniform = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const values = Float32Array.from({ length: COUNT }, () => {
    const gauss = Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
    return 1.5 * gauss;
  });
  const words = new Uint32Array(values.buffer);
  // Each operand carries the specials at a different offset, so they meet ordinary values too
  words.set(SPECIAL_WORDS, (seed % 7) * 3);
  return words;
};

describe.skipIf(!GPU_AVAILABLE)("the staged grid-stride loop on a real GPU", () => {
  it("writes the unfused sequence's words where the unstaged expression does not", async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    const pipelines = new Map<string, GPUComputePipeline>();
    const run = async (
      code: string,
      inputs: readonly Uint32Array<ArrayBuffer>[],
      groups: number,
    ): Promise<Uint32Array<ArrayBuffer>> => {
      device.pushErrorScope("validation");
      const owned: GPUBuffer[] = [];
      const make = (data: Uint32Array<ArrayBuffer> | number, usage: number): GPUBuffer => {
        const size = typeof data === "number" ? data : data.byteLength;
        const buffer = device.createBuffer({ size, usage: usage | U.COPY_DST });
        owned.push(buffer);
        if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      try {
        let compiled = pipelines.get(code);
        if (compiled === undefined) {
          compiled = await device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
          });
          pipelines.set(code, compiled);
        }
        const bytes = COUNT * 4;
        const out = make(new Uint32Array(COUNT).fill(0xdeadbeef), U.STORAGE | U.COPY_SRC);
        const staging = make(bytes, U.MAP_READ);
        const buffers = [
          make(new Uint32Array([COUNT, 0, 0, 0]), U.UNIFORM),
          ...inputs.map((input) => make(input, U.STORAGE)),
          out,
        ];
        const group = device.createBindGroup({
          layout: compiled.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(compiled);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(groups);
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(MAP_MODE.READ);
        const result = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return result;
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const buffer of owned) buffer.destroy();
        assertEquals(await device.popErrorScope(), null);
      }
    };

    const [a, b, c, d, e] = [11, 23, 37, 41, 53].map(operand);
    const forms = [
      {
        label: "r + y * g",
        inputs: [a, b, c],
        names: ["y", "g", "r"],
        staged: {
          rounds: [[{ name: "product", value: "y[i] * g[i]" }]],
          finish: [`out[i] = r[i] + ${stagedF32("product")};`],
        },
        mutant: "r[i] + y[i] * g[i]",
        plain: async (groups: number) =>
          await run(ADD, [c, await run(MUL, [a, b], groups)], groups),
      },
      {
        label: "e + (c + a * b) * d",
        inputs: [a, b, c, d, e],
        names: ["a", "b", "c", "d", "e"],
        staged: {
          rounds: [
            [{ name: "product", value: "a[i] * b[i]" }],
            [{ name: "scaled", value: `(c[i] + ${stagedF32("product")}) * d[i]` }],
          ],
          finish: [`out[i] = e[i] + ${stagedF32("scaled")};`],
        },
        mutant: "e[i] + (c[i] + a[i] * b[i]) * d[i]",
        plain: async (groups: number) => {
          const sum = await run(ADD, [c, await run(MUL, [a, b], groups)], groups);
          return await run(ADD, [e, await run(MUL, [sum, d], groups)], groups);
        },
      },
    ];
    try {
      for (const form of forms) {
        const fused =
          header(form.names) +
          stagedLoopWgsl({ workgroupSize: WORKGROUP_SIZE, count: "params.n", ...form.staged });
        const mutant = unstaged(form.names, form.mutant);
        for (const groups of [Math.ceil(COUNT / WORKGROUP_SIZE), 3]) {
          const label = `${form.label}, ${groups} workgroups`;
          const plain = await form.plain(groups);
          assertEquals(
            plain.includes(0xdeadbeef),
            false,
            `${label}: the plain run wrote every word`,
          );
          const staged = wordIdentity(await run(fused, form.inputs, groups), plain, "class");
          assertEquals(staged.failures, 0, `${label}: staged vs plain ${JSON.stringify(staged)}`);
          const unbarriered = wordIdentity(await run(mutant, form.inputs, groups), plain, "class");
          assert(
            unbarriered.failures > 0,
            `${label}: the unstaged mutant matched the plain sequence — this backend does not` +
              " contract the expression, so the staged kernel's match shows nothing",
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  });
});

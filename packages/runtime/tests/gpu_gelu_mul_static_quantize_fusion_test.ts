/**
 * The `geluMulStaticQuantize` fusion on a real GPU (perf-ledger K-48 step 1b): a Session with
 * `fuseGeluMulStaticQuantize` on returns the unfused Session's SRQ words as u32 — values landing on
 * SRQ boundaries after `gelu_tanh(gate) · ple`, ±0, ±inf, NaN payloads, subnormals, several scales
 * including 0 (the identity SRQ) — and a direct dispatch with the rounding masks injected proves both
 * barriers execute.
 *
 * The twin cases live in @karume/kernel-checks (checkGeluMulStaticQuantizeTwins), which the
 * playground's /kernel-check page runs on the browser's backend too; the check that the data reaches
 * the boundaries and the rounding-barrier dispatch stay here.
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  checkGeluMulStaticQuantizeTwins,
  disposeGeluMulChain,
  GELU_MUL_DIM,
  GELU_MUL_GROUPS,
  GELU_MUL_STATIC_QUANTIZE_TWIN_CASES,
  geluMulCase,
  geluMulF32Tensor,
  geluMulGateWords,
  geluMulInputs,
  geluMulPleWords,
  openGeluMulChain,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE,
  geluMulStaticQuantizeParams,
  geluMulStaticQuantizeWgsl,
} from "../src/kernels/gelu-mul-static-quantize.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { type DeclarationJson, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";

const DIM = GELU_MUL_DIM;

const words = (t: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(t.data.buffer, t.data.byteOffset, t.data.length);

describe.skipIf(!GPU_AVAILABLE)("gelu_tanh → mul → SRQ fusion (real GPU)", () => {
  it("returns the unfused SRQ words as u32 at rows 1 / 4 / 8 for every scale (the QAT operand form), keeping the node's operand order and reading a plain input operand", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkGeluMulStaticQuantizeTwins(gpu);
      assertEquals(
        [part.id, part.cases.length],
        Object.entries(GELU_MUL_STATIC_QUANTIZE_TWIN_CASES)[0],
      );
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });

  it("tunes the operand so the products land on the SRQ boundaries in every case", async () => {
    const gpu = await acquireGpu();
    try {
      for (const group of GELU_MUL_GROUPS) {
        const chain = await openGeluMulChain(gpu, group.options);
        try {
          for (const { seed, label } of group.cases) {
            const { inputs, targeted } = await geluMulCase(chain, group.scale, seed);
            // The data reaches the boundaries: the plain product equals the targeted word (the
            // host tunes in exact f32 arithmetic; a backend that flushes a subnormal factor misses
            // a few).
            const product = words((await chain.probe.run(inputs)).m);
            const hits = [...targeted].filter(([i, target]) => product[i] === target).length;
            assert(
              hits >= targeted.size * 0.95,
              `${label}: ${hits}/${targeted.size} on a boundary`,
            );
            if (group.scale !== 0) assert(targeted.size > 0, `${label}: no boundary targeted`);
          }
        } finally {
          await disposeGeluMulChain(chain);
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("executes both rounding barriers: a sign mask after gelu or after the mul negates the product, an LSB mask after gelu moves it", async () => {
    const gpu = await acquireGpu(),
      device = gpu.device;
    const rows = 8,
      n = rows * DIM,
      scale = Math.fround(0.0371);
    const owned: GPUBuffer[] = [];
    const make = (data: ArrayBufferView<ArrayBuffer> | number, usage: number): GPUBuffer => {
      const buffer = device.createBuffer({
        size: typeof data === "number" ? data : data.byteLength,
        usage,
      });
      owned.push(buffer);
      if (typeof data !== "number") device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    device.pushErrorScope("validation");
    const chain = await openGeluMulChain(gpu, { rows, scale, other: "input" });
    const srq: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["static_quantize"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [n] }],
      outputs: ["y"],
      initializers: {},
      values: { y: { dtype: "f32", shape: [n] } },
      nodes: [{ op: "static_quantize", ins: ["x"], outs: ["y"], attrs: { scale } }],
    };
    const standalone = await createSessionFromContainer(gpu, await openGraphModel(srq), GRAPH_NAME);
    try {
      const gate = geluMulGateWords(n, 5);
      const { ple } = await geluMulPleWords(chain, gate, scale, 5);
      const inputs = geluMulInputs(chain.options, gate, ple, 5);
      const probed = await chain.probe.run(inputs);
      const product = words(probed.m);
      /** The standalone SRQ of the product with every word XORed by `mask`. */
      const quantized = async (mask: number) =>
        words(
          (
            await standalone.run({
              x: geluMulF32Tensor(
                [n],
                product.map((bits) => (bits ^ mask) >>> 0),
              ),
            })
          ).y,
        );
      const params = make(geluMulStaticQuantizeParams(n, scale), U.UNIFORM | U.COPY_DST);
      const out = make(n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
      const stage = make(n * 4, U.COPY_DST | U.MAP_READ);
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code: geluMulStaticQuantizeWgsl("gelu-other") }),
          entryPoint: "main",
        },
      });
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          params,
          make(gate, U.STORAGE | U.COPY_DST),
          make(ple, U.STORAGE | U.COPY_DST),
          out,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const dispatch = async (
        maskAct: number,
        maskMul: number,
      ): Promise<Uint32Array<ArrayBuffer>> => {
        device.queue.writeBuffer(params, 4, new Uint32Array([maskAct, maskMul]));
        device.queue.writeBuffer(out, 0, new Uint32Array(n).fill(0xdeadbeef));
        const encoder = device.createCommandEncoder(),
          pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(n / GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE));
        pass.end();
        encoder.copyBufferToBuffer(out, 0, stage, 0, n * 4);
        device.queue.submit([encoder.finish()]);
        await stage.mapAsync(MAP_MODE.READ);
        const result = new Uint32Array(stage.getMappedRange().slice(0));
        stage.unmap();
        return result;
      };
      const unmasked = await dispatch(0, 0);
      assertEquals(unmasked, words(probed.y), "masks 0: the plain chain's words");
      const negated = await quantized(0x80000000);
      // After the mul the mask flips the stored product's bits exactly (NaN payloads included).
      assertEquals(await dispatch(0, 0x80000000), negated, "sign mask after the mul");
      // After gelu it flips the factor's sign: −act · ple = −(act · ple) for every non-NaN product
      // (which NaN a two-NaN product returns is not specified, so those words only stay NaN).
      const actSign = await dispatch(0x80000000, 0);
      const isNan = (bits: number): boolean => (bits & 0x7fffffff) > 0x7f800000;
      let nans = 0;
      for (let i = 0; i < n; i += 1) {
        if (isNan(product[i])) {
          nans += 1;
          assert(isNan(actSign[i]), `sign mask after gelu @${i}: NaN stays NaN`);
        } else assertEquals(actSign[i], negated[i], `sign mask after gelu @${i}`);
      }
      assert(nans < n / 2, "most products are not NaN");
      // An LSB flip of the gelu value moves some products (the barrier sits before the mul).
      const actLsb = await dispatch(1, 0);
      assert(
        unmasked.some((bits, i) => bits !== actLsb[i]),
        "an LSB mask after gelu changes some SRQ words",
      );
    } finally {
      await disposeGeluMulChain(chain);
      await standalone.dispose();
      device.queue.submit([]);
      await device.queue.onSubmittedWorkDone();
      const error = await device.popErrorScope();
      for (const buffer of owned) buffer.destroy();
      gpu.destroy();
      assertEquals(error, null);
    }
  });
});

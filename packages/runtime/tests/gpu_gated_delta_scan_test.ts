/**
 * The `gated_delta_scan` GPU kernel (ADR 0135, the value form) against the CPU reference, on a real
 * GPU. Without an adapter every case SKIPs explicitly.
 *
 * - **The band** (ADR 0135 decision 6): on every case of fixtures/gated-delta-scan.safetensors the
 *   output is finite and its max_abs and rel_l2 against the f64 reference are each at most twice the
 *   larger of upstream's two f32 functions' errors. The cases are slices of the 2026-10-27 spike's
 *   draw, so the spike's recorded GPU words (outputs/diag/gdn-scan-2026-10-27/check/out/, RTX
 *   3080 Ti, Vulkan) give the shares this form uses on them: `init_T64` 0.047 (max_abs) /
 *   0.123 (rel_l2), `long_T512` 0.234 / 0.299 — and the **built-in-exp regression** of the same
 *   form 1.413 / 2.006 on `long_T512`, outside the band. So this default-lane test catches a
 *   kernel that goes back to the built-in `exp` (the MUST of decision 6), together with the direct
 *   ULP test of the accurate exp below. `mixed_B2_T16` (B = 2, r = 2, two regimes) has no
 *   recorded words (its second batch element starts mid-draw). Other GPUs and backends are not
 *   measured (ADR 0135 open point 10).
 * - **The head broadcast** r = 2 against r = 1 (q / k repeated to the value heads): the same words.
 * - **Batch elements** are independent: B = 2 against each element run alone, the same words.
 * - **In-device determinism** (ADR 0110 decision 6): two runs, the same words.
 * - **Hazard canaries** (the numerically hazardous class — docs/op-vocabulary.md): g = +0 / −0 must
 *   give α = 1 exactly and g = −∞ α = 0 exactly (inputs chosen so the whole scan is exact in f32 —
 *   the expected words are known), and NaN in g or v propagates to exactly the reference's
 *   positions.
 * - **The accurate exp body** directly: at most 1.5 ULP and a mean signed error within ±0.2 ULP on
 *   the long-memory range [−0.1, 0) (measured 1.00 / −0.091 ULP; the built-in 3.00 / −0.337 ULP —
 *   research §5.4), and its contract's special values.
 *
 * Full write (NaN- / poison-filled outputs) is tests/gpu_full_write_test.ts.
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { EXP_ACCURATE_WGSL } from "../src/codegen/numerics-wgsl.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import { referenceGatedDeltaScanF64, type RefTensor } from "../src/reference/ops.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import {
  bandShare,
  describeBand,
  errorMetrics,
  type GatedDeltaCase,
  gatedDeltaCases,
} from "./helpers/gated-delta-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";

const CASES = await gatedDeltaCases();

const caseNamed = (name: string): GatedDeltaCase => {
  const found = CASES.find((entry) => entry.name === name);
  assert(found !== undefined, `fixture case '${name}' is missing`);
  return found;
};

type Inputs = readonly [RefTensor, RefTensor, RefTensor, RefTensor, RefTensor];

/** Runs one `gated_delta_scan` node on the GPU (a fresh Session — no state across calls). */
const runScan = async (gpu: GpuContext, inputs: Inputs): Promise<Tensor> => {
  const [q, , v] = inputs;
  const graph = singleOpDeclaration(
    "gated_delta_scan",
    inputs.map((input) => input.shape),
    [[q.shape[0], q.shape[1], v.shape[2], v.shape[3]]],
  );
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    const feeds = Object.fromEntries(inputs.map((input, index) => [`x${index}`, input]));
    return (await session.run(feeds))["y"];
  } finally {
    await session.dispose();
  }
};

const words = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

/** The first index where two word arrays differ (−1 when they are the same). */
const firstDifference = (a: Uint32Array, b: Uint32Array): number => {
  assertEquals(a.length, b.length);
  return a.findIndex((word, index) => word !== b[index]);
};

const f32 = (shape: readonly number[], value: (index: number) => number): RefTensor => ({
  dtype: "f32",
  shape,
  data: Float32Array.from({ length: shape.reduce((count, dim) => count * dim, 1) }, (_, index) =>
    value(index),
  ),
});

/** Repeats the head axis (axis 2) of `[B,T,H,d]` r times, as `repeat_interleave(r, dim=2)`. */
const repeatHeads = (tensor: RefTensor, ratio: number): RefTensor => {
  const [batch, time, heads, dim] = tensor.shape;
  return f32([batch, time, heads * ratio, dim], (index) => {
    const column = index % dim;
    const head = Math.floor(index / dim) % (heads * ratio);
    const row = Math.floor(index / (dim * heads * ratio));
    return tensor.data[(row * heads + Math.floor(head / ratio)) * dim + column];
  });
};

/** Batch element `item` of a `[B, ...]` tensor. */
const batchElement = (tensor: RefTensor, item: number): RefTensor => {
  const size = tensor.data.length / tensor.shape[0];
  return f32([1, ...tensor.shape.slice(1)], (index) => tensor.data[item * size + index]);
};

describe.skipIf(!GPU_AVAILABLE)("gated_delta_scan on the GPU (real GPU)", () => {
  it("stays inside the acceptance band on every fixture case", async () => {
    const gpu = await acquireGpu();
    try {
      for (const entry of CASES) {
        const output = await runScan(gpu, entry.inputs);
        const { B, T, Hv, dv } = entry.shape;
        assertEquals(output.shape, [B, T, Hv, dv], entry.name);
        const oracle = referenceGatedDeltaScanF64(...entry.inputs).data;
        const metrics = errorMetrics(output.data, oracle, Hv, dv);
        const where = `${entry.name}: ${describeBand(metrics, entry.band)}`;
        assertEquals(metrics.nonFinite, 0, where);
        assert(metrics.maxAbs <= entry.band.maxAbs, where);
        assert(metrics.relL2 <= entry.band.relL2, where);
        assert(bandShare(metrics, entry.band) <= 1, where);
      }
    } finally {
      gpu.destroy();
    }
  });

  it("gives the same words for r = 2 and for q / k repeated to the value heads (r = 1)", async () => {
    const gpu = await acquireGpu();
    try {
      const entry = caseNamed("mixed_B2_T16");
      const [q, k, v, g, beta] = entry.inputs;
      const ratio = entry.shape.Hv / entry.shape.Hk;
      assert(ratio > 1);
      const broadcast = words(await runScan(gpu, entry.inputs));
      const repeated = words(
        await runScan(gpu, [repeatHeads(q, ratio), repeatHeads(k, ratio), v, g, beta]),
      );
      assertEquals(firstDifference(broadcast, repeated), -1, "r = 2 against r = 1");
    } finally {
      gpu.destroy();
    }
  });

  it("computes each batch element as if it ran alone (B = 2)", async () => {
    const gpu = await acquireGpu();
    try {
      const entry = caseNamed("mixed_B2_T16");
      const together = words(await runScan(gpu, entry.inputs));
      const size = together.length / entry.shape.B;
      for (let item = 0; item < entry.shape.B; item += 1) {
        const alone = words(
          await runScan(gpu, [
            batchElement(entry.inputs[0], item),
            batchElement(entry.inputs[1], item),
            batchElement(entry.inputs[2], item),
            batchElement(entry.inputs[3], item),
            batchElement(entry.inputs[4], item),
          ]),
        );
        assertEquals(
          firstDifference(together.slice(item * size, (item + 1) * size), alone),
          -1,
          `batch element ${item}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });

  it("gives the same words in two runs (in-device determinism, ADR 0110 decision 6)", async () => {
    const gpu = await acquireGpu();
    try {
      const entry = caseNamed("long_T512");
      const first = words(await runScan(gpu, entry.inputs));
      const second = words(await runScan(gpu, entry.inputs));
      assertEquals(firstDifference(first, second), -1);
    } finally {
      gpu.destroy();
    }
  });
});

/**
 * Inputs on which the whole scan is exact in f32: k_t = q_t = e₀, so the state has one non-zero
 * row; v holds multiples of 1/8 in [−1, 1], so every subtraction and addition of that row is exact,
 * and o_t = row₀ · fround(128^-½) is one rounding. Head 0 has g = `gateHead0`, head 1 `gateHead1`.
 */
const exactInputs = (gateHead0: number, gateHead1: number, strength: number): Inputs => {
  const [time, dim] = [12, 128];
  const unit = (index: number): number => (index % dim === 0 ? 1 : 0);
  return [
    f32([1, time, 1, dim], unit),
    f32([1, time, 1, dim], unit),
    f32([1, time, 2, dim], (index) => ((((index * 7) % 17) - 8) / 8) * ((index >> 7) % 2 ? -1 : 1)),
    f32([1, time, 2], (index) => (index % 2 === 0 ? gateHead0 : gateHead1)),
    f32([1, time, 2], () => strength),
  ];
};

const SCALE_F32 = Math.fround(128 ** -0.5);

describe.skipIf(!GPU_AVAILABLE)("gated_delta_scan hazard canaries (real GPU)", () => {
  it("decays by exactly 1 at g = +0 and g = −0", async () => {
    // β = 1/2: with α = 1 the row accumulates, row_t = row_{t−1} + ½(v_t − row_{t−1}) (exact here);
    // an α that is not exactly 1 moves the last bits.
    const gpu = await acquireGpu();
    try {
      const inputs = exactInputs(0, -0, 0.5);
      const output = await runScan(gpu, inputs);
      const v = inputs[2].data;
      const [time, dim] = [12, 128];
      for (let head = 0; head < 2; head += 1) {
        for (let column = 0; column < dim; column += 1) {
          let row = 0;
          for (let t = 0; t < time; t += 1) {
            const value = v[(t * 2 + head) * dim + column];
            row = row + 0.5 * (value - row);
            const index = (t * 2 + head) * dim + column;
            // `===`: the sign of a zero result is not part of the check
            assert(
              output.data[index] === Math.fround(row * SCALE_F32),
              `head ${head} (g = ${head === 0 ? "+0" : "−0"}) t ${t} column ${column}: ${output.data[index]}`,
            );
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("forgets the state exactly at g = −∞ (α = 0)", async () => {
    const gpu = await acquireGpu();
    try {
      const inputs = exactInputs(-Infinity, -Infinity, 0.5);
      const output = await runScan(gpu, inputs);
      const v = inputs[2].data;
      output.data.forEach((value, index) => {
        // `===`: the sign of a zero result is not part of the check
        assert(value === Math.fround(0.5 * v[index] * SCALE_F32), `element ${index}: ${value}`);
      });
    } finally {
      gpu.destroy();
    }
  });

  it("propagates NaN in g and in v to exactly the reference's positions", async () => {
    const gpu = await acquireGpu();
    try {
      const entry = caseNamed("init_T64");
      const { T, Hv, dv } = entry.shape;
      const [q, k, v, g, beta] = entry.inputs;
      const gate = Float32Array.from(g.data);
      gate[20 * Hv + 1] = Number.NaN; // token 20, value head 1
      const value = Float32Array.from(v.data);
      value[(30 * Hv + 0) * dv + 7] = Number.NaN; // token 30, value head 0, column 7
      const inputs: Inputs = [
        q,
        k,
        { dtype: "f32", shape: v.shape, data: value },
        { dtype: "f32", shape: g.shape, data: gate },
        beta,
      ];
      const output = await runScan(gpu, inputs);
      const reference = referenceGatedDeltaScanF64(...inputs).data;
      let nan = 0;
      let worst = 0;
      for (let index = 0; index < reference.length; index += 1) {
        const expectedNan = Number.isNaN(reference[index]);
        assertEquals(Number.isNaN(output.data[index]), expectedNan, `element ${index}`);
        if (expectedNan) {
          nan += 1;
        } else {
          worst = Math.max(worst, Math.abs(output.data[index] - reference[index]));
        }
      }
      // head 1 from token 20 on (all columns) + head 0 column 7 from token 30 on
      assertEquals(nan, (T - 20) * dv + (T - 30));
      assert(worst <= entry.band.maxAbs, `finite elements: max_abs ${worst}`);
    } finally {
      gpu.destroy();
    }
  });
});

/** One f32 ULP at |value| (the gap to the next f32 away from zero). */
const ulp = (value: number): number => {
  const bits = new Uint32Array(Float32Array.of(Math.abs(value)).buffer);
  bits[0] += 1;
  return new Float32Array(bits.buffer)[0] - Math.fround(Math.abs(value));
};

/** Evaluates `exp_accurate` and the built-in `exp` on the GPU, element for element. */
const EXP_PROBE_WGSL = `@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
${EXP_ACCURATE_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = arrayLength(&x);
  let i = id.x;
  if (i >= n) { return; }
  y[i] = exp_accurate(x[i]);
  y[n + i] = exp(x[i]);
}
`;

const probeExp = async (gpu: GpuContext, x: Float32Array<ArrayBuffer>) => {
  const device = gpu.device;
  const owned: GPUBuffer[] = [];
  const make = (size: number, usage: number): GPUBuffer => {
    const buffer = device.createBuffer({ size, usage });
    owned.push(buffer);
    return buffer;
  };
  try {
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: EXP_PROBE_WGSL }), entryPoint: "main" },
    });
    const input = make(x.byteLength, U.STORAGE | U.COPY_DST);
    device.queue.writeBuffer(input, 0, x);
    const output = make(2 * x.byteLength, U.STORAGE | U.COPY_SRC);
    const stage = make(2 * x.byteLength, U.MAP_READ | U.COPY_DST);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input } },
          { binding: 1, resource: { buffer: output } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(x.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, stage, 0, 2 * x.byteLength);
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    assertEquals(error, null, error?.message);
    await stage.mapAsync(MAP_MODE.READ);
    const values = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    return { accurate: values.subarray(0, x.length), builtin: values.subarray(x.length) };
  } finally {
    for (const buffer of owned) buffer.destroy();
  }
};

/** Max |error| and mean signed error in ULP of the exact exp, over the normal results. */
const ulpStats = (x: Float32Array, y: Float32Array): { max: number; meanSigned: number } => {
  let max = 0;
  let signed = 0;
  let count = 0;
  x.forEach((input, index) => {
    const exact = Math.exp(input);
    if (exact < 1.2e-38) return;
    const error = (y[index] - exact) / ulp(exact);
    max = Math.max(max, Math.abs(error));
    signed += error;
    count += 1;
  });
  return { max, meanSigned: signed / count };
};

describe.skipIf(!GPU_AVAILABLE)("the accurate exp body on the GPU (real GPU)", () => {
  it("keeps the long-memory range [−0.1, 0) within 1.5 ULP and ±0.2 ULP of mean bias", async () => {
    const gpu = await acquireGpu();
    try {
      const count = 1 << 20;
      const x = Float32Array.from({ length: count }, (_, i) => -0.1 + (0.1 * (i + 0.5)) / count);
      const { accurate, builtin } = await probeExp(gpu, x);
      const stats = ulpStats(x, accurate);
      const reference = ulpStats(x, builtin);
      const where =
        `exp_accurate max ${stats.max.toFixed(3)} / mean signed ${stats.meanSigned.toFixed(3)} ULP` +
        ` (the built-in exp here: ${reference.max.toFixed(3)} / ${reference.meanSigned.toFixed(3)})`;
      assert(stats.max <= 1.5, where);
      assert(Math.abs(stats.meanSigned) <= 0.2, where);
    } finally {
      gpu.destroy();
    }
  });

  it("returns exactly 1 at ±0, 0 at −∞ and below −87, and NaN at NaN", async () => {
    const gpu = await acquireGpu();
    try {
      const x = Float32Array.of(0, -0, -Infinity, -87.5, -100, Number.NaN, -86.5);
      const { accurate } = await probeExp(gpu, x);
      assertEquals([...accurate.subarray(0, 5)], [1, 1, 0, 0, 0]);
      assert(Number.isNaN(accurate[5]), `NaN in: ${accurate[5]}`);
      assert(accurate[6] > 0 && Number.isFinite(accurate[6]), `−86.5: ${accurate[6]}`);
    } finally {
      gpu.destroy();
    }
  });
});

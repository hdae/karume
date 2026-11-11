/**
 * The `geluMulStaticQuantize` fusion on a real GPU (perf-ledger K-48 step 1b): a Session with
 * `fuseGeluMulStaticQuantize` on returns the unfused Session's SRQ words as u32 — values landing on
 * SRQ boundaries after `gelu_tanh(gate) · ple`, ±0, ±inf, NaN payloads, subnormals, several scales
 * including 0 (the identity SRQ) — and a direct dispatch with the rounding masks injected proves both
 * barriers execute.
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { checkFusionTwin } from "@karume/kernel-checks";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE,
  type GeluMulOrder,
  geluMulStaticQuantizeParams,
  geluMulStaticQuantizeWgsl,
} from "../src/kernels/gelu-mul-static-quantize.ts";
import { staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import {
  geluMulStaticQuantizeGraph,
  type GeluMulStaticQuantizeGraphOptions,
  PLE_LAYERS,
} from "./helpers/gelu-mul-static-quantize-graph.ts";
import { type DeclarationJson, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";

const DIM = 256;

const words = (t: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(t.data.buffer, t.data.byteOffset, t.data.length);

const scratch = new Float32Array(1);
const scratchBits = new Uint32Array(scratch.buffer);
const bitsOf = (value: number): number => {
  scratch[0] = value;
  return scratchBits[0];
};
const valueOf = (bits: number): number => {
  scratchBits[0] = bits;
  return scratch[0];
};

/** A seeded xorshift32 — the data is fixed by the seed (a failure reproduces). */
const generator = (seed: number) => {
  let state = seed >>> 0 || 1;
  const bits = (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state;
  };
  const uniform = (): number => (bits() + 0.5) / 2 ** 32;
  const gauss = (): number =>
    Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  return { bits, uniform, gauss };
};

/** ±0, ±inf, quiet and signalling NaN payloads of both signs, subnormals, the f32 extremes, 1. */
const SPECIALS = [
  0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00001, 0x7fa5a5a5, 1,
  0x80000001, 0x007fffff, 0x807fffff, 0x7f7fffff, 0xff7fffff, 0x00800000, 0x80800000, 0x3f800000,
];

/** The scales: QAT-sized, small, tiny, large, the smallest normal, and 0 (identity). */
const SCALES = [0.0371, 0.001, 1e-30, 3.3, 0.5, 123.25, 2 ** -126, 0].map(Math.fround);

/** The SRQ boundary words of `scale` ±2 ulp, both signs (finite only). Empty for the identity. */
const boundaryTargets = (scale: number): readonly number[] => {
  const table = staticQuantizeParams(DIM, scale);
  if (table[258] !== 0) return [];
  const targets: number[] = [];
  for (let level = 1; level <= 128; level += 1) {
    for (let ulp = -2; ulp <= 2; ulp += 1) {
      for (const sign of [0, 0x80000000]) {
        const bits = ((table[level] + ulp) | sign) >>> 0;
        if ((bits & 0x7fffffff) < 0x7f800000) targets.push(bits);
      }
    }
  }
  return targets;
};

/**
 * The f32 `w` with `fround(act · w)` = the target bits, stepping `w` by ulps from `target / act`
 * (an f32 multiply is correctly rounded), or `undefined`.
 */
const tune = (act: number, targetBits: number): number | undefined => {
  const target = valueOf(targetBits);
  if (!Number.isFinite(act) || act === 0) return undefined;
  let w = Math.fround(target / act);
  for (let step = 0; step < 16 && Number.isFinite(w) && w !== 0; step += 1) {
    const product = Math.fround(act * w);
    if (bitsOf(product) === targetBits) return w;
    const bits = bitsOf(w);
    w = valueOf((Math.abs(product) < Math.abs(target) ? bits + 1 : bits - 1) >>> 0);
  }
  return undefined;
};

/** The gate: normal, small, large (gelu saturation) and raw-bit values; specials at the end. */
const gateWords = (n: number, seed: number): Uint32Array<ArrayBuffer> => {
  const random = generator(seed);
  const gate = new Uint32Array(n);
  for (let i = 0; i < n; i += 1) {
    const kind = i % 6;
    gate[i] = kind === 4 ? random.bits() : bitsOf(random.gauss() * [1, 0.01, 5, 12, 0, 3][kind]);
  }
  SPECIALS.forEach((bits, k) => (gate[n - SPECIALS.length + k] = bits));
  return gate;
};

type Chain = {
  readonly options: GeluMulStaticQuantizeGraphOptions;
  /** The unfused chain with the gelu value and the product as graph outputs. */
  readonly probe: Session;
  readonly reference: Session;
  readonly fused: Session;
};

const openChain = async (
  gpu: GpuContext,
  options: GeluMulStaticQuantizeGraphOptions,
): Promise<Chain> => {
  const chain = await openGraphModel(geluMulStaticQuantizeGraph(options));
  const probe = await createSessionFromContainer(
    gpu,
    await openGraphModel(
      geluMulStaticQuantizeGraph({ ...options, geluOutput: true, mulOutput: true }),
    ),
    GRAPH_NAME,
  );
  const reference = await createSessionFromContainer(gpu, chain, GRAPH_NAME);
  const fused = await createSessionFromContainer(gpu, chain, GRAPH_NAME, {
    fuseGeluMulStaticQuantize: true,
  });
  return { options, probe, reference, fused };
};

const disposeChain = async (chain: Chain): Promise<void> => {
  await chain.probe.dispose();
  await chain.reference.dispose();
  await chain.fused.dispose();
};

const f32Tensor = (shape: readonly number[], data: Uint32Array<ArrayBuffer>) => ({
  dtype: "f32" as const,
  shape,
  data: new Float32Array(data.buffer, data.byteOffset, data.length),
});

/** The graph inputs carrying `ple` in the chain's operand form (the slice form: layer 1 of `pli`). */
const inputsOf = (
  options: GeluMulStaticQuantizeGraphOptions,
  gate: Uint32Array<ArrayBuffer>,
  ple: Uint32Array<ArrayBuffer>,
  seed: number,
): Readonly<Record<string, ReturnType<typeof f32Tensor>>> => {
  const rows = options.rows ?? 4;
  const shape = [1, rows, DIM];
  if ((options.other ?? "slice") === "input") {
    return { gate: f32Tensor(shape, gate), ple: f32Tensor(shape, ple) };
  }
  const random = generator(seed);
  const pli = new Uint32Array(rows * PLE_LAYERS * DIM);
  for (let i = 0; i < pli.length; i += 1) pli[i] = bitsOf(random.gauss());
  for (let row = 0; row < rows; row += 1) {
    pli.set(ple.subarray(row * DIM, (row + 1) * DIM), (row * PLE_LAYERS + 1) * DIM);
  }
  return { gate: f32Tensor(shape, gate), pli: f32Tensor([1, rows, PLE_LAYERS, DIM], pli) };
};

/**
 * The operand values for `gate`: every other element tuned so the product lands on an SRQ boundary
 * ±2 ulp (from the probe's gelu value), the rest random bits or normal values, specials at the end.
 */
const pleWords = async (
  chain: Chain,
  gate: Uint32Array<ArrayBuffer>,
  scale: number,
  seed: number,
): Promise<{ readonly ple: Uint32Array<ArrayBuffer>; readonly targeted: Map<number, number> }> => {
  const n = gate.length;
  const random = generator(seed ^ 0x9e3779b9);
  const ones = new Uint32Array(n).fill(bitsOf(1));
  const act = new Float32Array(
    (await chain.probe.run(inputsOf(chain.options, gate, ones, seed))).g.data,
  );
  const targets = boundaryTargets(scale);
  const ple = new Uint32Array(n);
  const targeted = new Map<number, number>();
  let next = 0;
  for (let i = 0; i < n - 2 * SPECIALS.length; i += 1) {
    let tuned: number | undefined;
    for (
      let attempt = 0;
      i % 2 === 0 && targets.length > 0 && attempt < 8 && tuned === undefined;
      attempt += 1
    ) {
      const target = targets[next % targets.length];
      next += 7;
      tuned = tune(act[i], target);
      if (tuned !== undefined) targeted.set(i, target);
    }
    ple[i] =
      tuned !== undefined
        ? bitsOf(tuned)
        : random.uniform() < 0.3
          ? random.bits()
          : bitsOf(random.gauss() * 2);
  }
  for (let i = n - 2 * SPECIALS.length; i < n; i += 1) ple[i] = SPECIALS[i % SPECIALS.length];
  return { ple, targeted };
};

/** Runs the chain flag off and on and compares the SRQ words (and the gate's late consumer) as u32. */
const assertIdentity = async (
  gpu: GpuContext,
  chain: Chain,
  scale: number,
  seed: number,
  label: string,
): Promise<void> => {
  const rows = chain.options.rows ?? 4;
  const gate = gateWords(rows * DIM, seed);
  const { ple, targeted } = await pleWords(chain, gate, scale, seed);
  const inputs = inputsOf(chain.options, gate, ple, seed);
  // The data reaches the boundaries: the plain product equals the targeted word (the host tunes in
  // exact f32 arithmetic; a backend that flushes a subnormal factor misses a few).
  const product = words((await chain.probe.run(inputs)).m);
  const hits = [...targeted].filter(([i, target]) => product[i] === target).length;
  assert(hits >= targeted.size * 0.95, `${label}: ${hits}/${targeted.size} on a boundary`);
  if (scale !== 0) assert(targeted.size > 0, `${label}: no boundary targeted`);
  // gelu_tanh + mul + static_quantize + neg (+ slice) → fused + neg (+ slice).
  const plain = (chain.options.other ?? "slice") === "slice" ? 3 : 2;
  // The SRQ words (y) and the gate after the fold (copy).
  assertCheckCase(
    await checkFusionTwin(gpu, {
      label,
      fused: chain.fused,
      plain: chain.reference,
      feeds: inputs,
      counter: "geluMulStaticQuantize",
      hits: 1,
      nan: "payload",
      dispatches: { fused: plain },
    }),
  );
};

describe.skipIf(!GPU_AVAILABLE)("gelu_tanh → mul → SRQ fusion (real GPU)", () => {
  it("returns the unfused SRQ words as u32 at rows 1 / 4 / 8 for every scale (the QAT operand form)", async () => {
    const gpu = await acquireGpu();
    try {
      for (const rows of [1, 4, 8]) {
        for (const [index, scale] of SCALES.entries()) {
          const chain = await openChain(gpu, { rows, scale });
          try {
            for (const seed of [11, 29]) {
              await assertIdentity(
                gpu,
                chain,
                scale,
                seed + index + rows * 101,
                `rows ${rows} / scale ${scale} / seed ${seed}`,
              );
            }
          } finally {
            await disposeChain(chain);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  });

  it("keeps the node's operand order and reads a plain input operand", async () => {
    const gpu = await acquireGpu();
    try {
      const forms: readonly (readonly [GeluMulOrder, "slice" | "input"])[] = [
        ["other-gelu", "slice"],
        ["other-gelu", "input"],
        ["gelu-other", "input"],
      ];
      for (const [order, other] of forms) {
        for (const scale of [Math.fround(0.0371), 0]) {
          const chain = await openChain(gpu, { rows: 8, order, other, scale });
          try {
            await assertIdentity(gpu, chain, scale, 7, `${order} / ${other} / scale ${scale}`);
          } finally {
            await disposeChain(chain);
          }
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
    const chain = await openChain(gpu, { rows, scale, other: "input" });
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
      const gate = gateWords(n, 5);
      const { ple } = await pleWords(chain, gate, scale, 5);
      const inputs = inputsOf(chain.options, gate, ple, 5);
      const probed = await chain.probe.run(inputs);
      const product = words(probed.m);
      /** The standalone SRQ of the product with every word XORed by `mask`. */
      const quantized = async (mask: number) =>
        words(
          (
            await standalone.run({
              x: f32Tensor(
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
      await disposeChain(chain);
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

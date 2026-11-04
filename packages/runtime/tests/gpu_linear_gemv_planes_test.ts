/**
 * The byte-plane GEMV (`SessionOptions.linearGemvPlanes`) on a real GPU:
 *
 * 1. The class-R check of `@karume/kernel-checks` (the cases the playground's kernel-check page
 *    runs): the f64 band, the A/B band against the sequential GEMV and the structural checks.
 * 2. Point 4 of ADR 0110 §3 — the check is not vacuous: two mutants of the WGSL turn its f64 band
 *    red (the decode off by a constant, on every case; two planes' activations swapped, on every
 *    mixed case — the saturated fixture holds one weight value, so the swap cannot change it).
 * 3. Through a Session at the measured shape (the QAT E2B lm_head, i2 262144 × 1536, M = 1): with
 *    the option on, the byte-plane key runs, the sequential GEMV's does not, and every output is
 *    within the f64 band; with it off, the sequential GEMV runs and no byte-plane key does.
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  checkPlanesGemv,
  type CheckPart,
  GEMV_PLANES_CASES,
  GEMV_PLANES_F64_BAND,
} from "@karume/kernel-checks";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { linearGemvKey } from "../src/kernels/linear-gemv.ts";
import { planesGemvKey } from "../src/kernels/linear-gemv-planes.ts";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
} from "./helpers/model-fixture.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { type CensusRow, countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

describe.skipIf(!GPU_AVAILABLE)("byte-plane GEMV: the class-R check (real GPU)", () => {
  it("passes every criterion of every case of the shared check", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkPlanesGemv(gpu);
      // K = 16 / 80 / 1536 mixed and K = 1536 saturated, each at 37 and 1029 columns, × 15
      // variants × M = 1 / 8.
      assertEquals(part.cases.length, GEMV_PLANES_CASES["gemv-planes"]);
      assertEquals(part.cases.length, 240);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });
});

/** Replaces every `from`; a missing anchor throws instead of leaving the kernel unmutated. */
const mustReplaceAll = (wgsl: string, from: string, to: string): string => {
  if (!wgsl.includes(from)) throw new Error(`mutation anchor not found: ${from}`);
  return wgsl.replaceAll(from, to);
};

/** The decode without its subtraction: every weight value reads as `u + 1.5 · 2^23`. */
const decodeOffByConstant = (wgsl: string): string =>
  mustReplaceAll(wgsl, " - vec4<f32>(12582914.0)", "");

/** Planes 1 and 2 multiply each other's activations (elements 4b + 1 and 4b + 2 swapped). */
const swapPlaneActivations = (wgsl: string): string =>
  mustReplaceAll(
    mustReplaceAll(mustReplaceAll(wgsl, "p1)", "p_swap)"), "p2)", "p1)"),
    "p_swap)",
    "p2)",
  );

/** The f64 band criterion of each case, by case label. */
const f64Band = (part: CheckPart): readonly { label: string; failures: number }[] =>
  part.cases.map(({ label, criteria }) => {
    const criterion = criteria.find(({ name }) => name === GEMV_PLANES_F64_BAND);
    assert(criterion !== undefined, `${label}: no '${GEMV_PLANES_F64_BAND}' criterion`);
    return { label, failures: criterion.failures };
  });

describe.skipIf(!GPU_AVAILABLE)("byte-plane GEMV: the check is not vacuous (real GPU)", () => {
  it("fails the f64 band on every case when the decode is off by a constant", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkPlanesGemv(gpu, { mutate: decodeOffByConstant });
      assertEquals(part.cases.length, GEMV_PLANES_CASES["gemv-planes"]);
      for (const { label, failures } of f64Band(part)) assert(failures > 0, label);
    } finally {
      gpu.destroy();
    }
  });

  it("fails the f64 band on every mixed case when two planes swap activations", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkPlanesGemv(gpu, { mutate: swapPlaneActivations });
      assertEquals(part.cases.length, GEMV_PLANES_CASES["gemv-planes"]);
      const mixed = f64Band(part).filter(({ label }) => label.includes("mixed"));
      // 3 K × 2 column counts × 15 variants × 2 M.
      assertEquals(mixed.length, 180);
      for (const { label, failures } of mixed) assert(failures > 0, label);
    } finally {
      gpu.destroy();
    }
  });
});

/** The QAT E2B lm_head. */
const N = 262144;
const K = 1536;

/** `x → linear (i2) → y`, one decode row. */
const sessionGraph = (n: number, k: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, k] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    y: { dtype: "f32", shape: [1, n] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

type Fixture = {
  readonly model: BoundContainer;
  readonly x: Float32Array<ArrayBuffer>;
  /** `(Σ_k q · x) · scale + bias` in f64. */
  readonly reference: Float64Array<ArrayBuffer>;
};

/** A 32-bit mix of `i`, so neighbouring words hold unrelated codes. */
const hash = (i: number): number => {
  let h = i >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
};

/** The model with random i2 weights (one scale per channel), its input and its f64 evaluation. */
const fixtureOf = (n: number, k: number): Fixture => {
  const words = new Uint32Array((n * k) / 16);
  for (let i = 0; i < words.length; i++) words[i] = hash(i + 1);
  const wscale = Float32Array.from({ length: n }, (_, c) => ((c % 17) + 1) * 0.0031);
  const bias = Float32Array.from({ length: n }, (_, c) => ((c % 7) - 3) * 0.11);
  const x = Float32Array.from({ length: k }, (_, i) => (4 * hash(i + 7)) / 2 ** 32 - 2);
  const reference = new Float64Array(n);
  for (let column = 0; column < n; column++) {
    let sum = 0;
    for (let word = 0; word < k / 16; word++) {
      const bits = words[(column * k) / 16 + word];
      for (let j = 0; j < 16; j++) sum += (((bits >>> (2 * j)) & 3) - 2) * x[word * 16 + j];
    }
    reference[column] = sum * wscale[column] + bias[column];
  }
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      // The i2 payload is bound as is: element j at bits 2·(j mod 16) of word j / 16.
      bytes: new Uint8Array(words.buffer),
      // One scale per output channel (the i2 codec's grouping is the whole row).
      encoding: {
        codec: "int2-off",
        groupSize: k,
        scale: { bytes: f32Bytes(wscale), dtype: "f32" },
      },
    },
    { graph: GRAPH_NAME, initializer: "b", bytes: f32Bytes(bias), encoding: { codec: "f32" } },
  ];
  return { model: memoryModel(sessionGraph(n, k), tensors), x, reference };
};

/** Runs the fixture twice (the second run hits the prepared plan); returns the output and census. */
const runTwice = async (
  fixture: Fixture,
  options: SessionOptions,
  where: string,
): Promise<{ readonly y: Float32Array<ArrayBuffer>; readonly census: readonly CensusRow[] }> => {
  const gpu = await acquireGpu();
  try {
    const session = await createSessionFromContainer(gpu, fixture.model, GRAPH_NAME, options);
    try {
      const input = { x: { dtype: "f32" as const, shape: [1, fixture.x.length], data: fixture.x } };
      const first = Float32Array.from((await session.run(input)).y.data);
      const second = Float32Array.from((await session.run(input)).y.data);
      assertEquals(
        new Uint32Array(second.buffer),
        new Uint32Array(first.buffer),
        `${where}: the prepared plan's run differs from the first`,
      );
      return { y: first, census: requireCensus(session.diagnostics().lastRunPipelines, where) };
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

const PLANES_FAMILY = "linear_gemv_planes";
const isPlanes = (key: string): boolean => key.split(":", 1)[0] === PLANES_FAMILY;

describe.skipIf(!GPU_AVAILABLE)("linearGemvPlanes through a Session (real GPU)", () => {
  // Built on first use (100 MB of weights), so collecting the skipped tests costs nothing.
  let fixture: Fixture | undefined;
  const fixtureFor = (): Fixture => (fixture ??= fixtureOf(N, K));

  it("runs the byte-plane key at the lm_head shape, every output within the f64 band", async () => {
    const where = "lm_head 262144×1536 (option on)";
    const { y, census } = await runTwice(fixtureFor(), { linearGemvPlanes: true }, where);
    assertEquals(countDispatches(census, planesGemvKey({ lanes: 8, rows: 4 })), 1, where);
    assertEquals(countDispatches(census, linearGemvKey("i2")), 0, where);
    const { reference } = fixtureFor();
    let failures = 0;
    let first = -1;
    for (let i = 0; i < N; i++) {
      const bound = GEMM_TOLERANCE.atol + GEMM_TOLERANCE.rtol * Math.abs(reference[i]);
      if (Number.isFinite(y[i]) && Math.abs(y[i] - reference[i]) <= bound) continue;
      failures += 1;
      if (first < 0) first = i;
    }
    assertEquals(
      failures,
      0,
      `${where}: ${failures} outputs outside the f64 band, the first at ${first}: ${y[first]} against ${reference[first]}`,
    );
  });

  it("runs the sequential GEMV's key at the lm_head shape with the option off", async () => {
    const where = "lm_head 262144×1536 (option off)";
    const { census } = await runTwice(fixtureFor(), {}, where);
    assertEquals(countDispatches(census, linearGemvKey("i2")), 1, where);
    assertEquals(countDispatches(census, isPlanes), 0, where);
  });

  it("rejects a non-boolean linearGemvPlanes at construction", async () => {
    const gpu = await acquireGpu();
    try {
      const { model } = fixtureOf(64, 64);
      const options: SessionOptions = {};
      Object.defineProperty(options, "linearGemvPlanes", { value: 1, enumerable: true });
      await assertRejects(
        () => createSessionFromContainer(gpu, model, GRAPH_NAME, options),
        ExecutionError,
        "options.linearGemvPlanes must be a boolean",
      );
    } finally {
      gpu.destroy();
    }
  });
});

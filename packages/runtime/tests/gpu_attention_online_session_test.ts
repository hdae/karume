// The online int8 attention (ADR 0132) through the Session path: `attentionCompute: "a8"` ×
// `attentionScoreStorage: "none"` on a single fused attention node.
//
//   (i)   the census shows the attention_online key in the reduction form the route chose for the
//         device (sg32 when the adapter has subgroups and the device was acquired with
//         `subgroups: "available"`, wg otherwise) and none of attention_qk / attention_stats /
//         attention_pv;
//   (ii)  the output is bit-identical to the raw kernel dispatched on the same quantised inputs
//         (the Session adds only the quantisation dispatches and the windows around the kernel);
//   (iii) a shape outside the kernel (D = 96) falls back to the three a8 dispatches with S in f32
//         and is bit-identical to the a8 Session;
//   (iv)  a shape over the work budget of one dispatch runs one dispatch per window of
//         `planAttentionOnlineWindows`, bit-identical to a single raw dispatch.
//
// The inputs are built so that their per-token quantisation is exact (every element is an integer
// multiple of its row's step and every row holds ±127 steps): the host quantisation of the raw run
// and the GPU quantisation of the Session then give the same bits without a tie margin to measure.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineReduce,
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
} from "../src/kernels/attention-online-i8a8.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { stridedKey } from "../src/codegen/strided.ts";
import { transposeValues } from "../src/reference/attention-online.ts";
import { quantizeRowsReference } from "../src/reference/i8a8.ts";
import { attentionDeviceFacts, attentionRoute } from "../src/runtime/attention-route.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import { planAttentionOnlineWindows } from "../src/runtime/recipe-builders/attention.ts";
import { GRAPH_NAME, openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type CensusRow, countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const ONLINE: SessionOptions = { attentionCompute: "a8", attentionScoreStorage: "none" };
const A8: SessionOptions = { attentionCompute: "a8" };

type Shape = {
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/** The half scale of ADR 0023 (`√scale_factor`, applied to both q and k). */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

// ---------------------------------------------------------------------------
// Inputs with an exact quantisation
// ---------------------------------------------------------------------------

/** A deterministic integer in [-126, 126] (varies with every index, per head and per tensor). */
const lattice = (head: number, token: number, column: number, salt: number): number =>
  ((token * 37 + column * 11 + head * 53 + salt) % 253) - 126;

/**
 * q or k `[B·H, rows, D]`: element `n · c_head` with |n| ≤ 126, and the element at column
 * `token % D` set to ±127 steps, so every token row (the quantisation row of q / k) has amax =
 * 127 · c and quantises to exactly n.
 */
const tokenRows = (
  batch: number,
  rows: number,
  depth: number,
  step: (head: number) => number,
  salt: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * rows * depth);
  for (let head = 0; head < batch; head += 1) {
    const c = step(head);
    for (let token = 0; token < rows; token += 1) {
      for (let column = 0; column < depth; column += 1) {
        const n =
          column === token % depth
            ? token % 2 === 0
              ? 127
              : -127
            : lattice(head, token, column, salt);
        out[(head * rows + token) * depth + column] = n * c;
      }
    }
  }
  return out;
};

/**
 * v `[B·H, N, D]`: element `n · c_head` with |n| ≤ 126, and the element of key `column % N` set to
 * ±127 steps in every column, so every row of Vᵀ (the quantisation row of v — one column of v over
 * the keys) has amax = 127 · c.
 */
const valueColumns = (
  batch: number,
  keys: number,
  depth: number,
  step: (head: number) => number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * keys * depth);
  for (let head = 0; head < batch; head += 1) {
    const c = step(head);
    for (let key = 0; key < keys; key += 1) {
      for (let column = 0; column < depth; column += 1) {
        const n =
          key === column % keys ? (column % 2 === 0 ? -127 : 127) : lattice(head, key, column, 7);
        out[(head * keys + key) * depth + column] = n * c;
      }
    }
  }
  return out;
};

type Inputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly k: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
  readonly scale: number;
};

const makeInputs = (shape: Shape): Inputs => {
  const batch = shape.b * shape.h;
  // Steps chosen so that S has a standard deviation of a few units (a softmax neither flat nor
  // one-hot); a different step per head keeps the heads apart.
  return {
    q: tokenRows(batch, shape.m, shape.d, (head) => 0.02 + 0.005 * head, 3),
    k: tokenRows(batch, shape.n, shape.d, (head) => 0.018 + 0.004 * head, 5),
    v: valueColumns(batch, shape.n, shape.d, (head) => 0.01 + 0.003 * head),
    scale: halfScale(shape.d),
  };
};

/** Pack flat i8 rows four per u32 (component i in byte i — the order of `pack4xI8`). */
const packWords = (values: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(values.length / 4);
  for (let w = 0; w < words.length; w += 1) {
    words[w] =
      ((values[w * 4] & 0xff) |
        ((values[w * 4 + 1] & 0xff) << 8) |
        ((values[w * 4 + 2] & 0xff) << 16) |
        ((values[w * 4 + 3] & 0xff) << 24)) >>>
      0;
  }
  return words;
};

// ---------------------------------------------------------------------------
// The two runs under comparison
// ---------------------------------------------------------------------------

type SessionRun = {
  readonly output: Float32Array<ArrayBuffer>;
  readonly census: readonly CensusRow[];
};

const runSession = async (
  gpu: GpuContext,
  shape: Shape,
  inputs: Inputs,
  options: SessionOptions,
): Promise<SessionRun> => {
  const { b, h, m, n, d } = shape;
  const graph = singleOpDeclaration(
    "attention",
    [
      [b, h, m, d],
      [b, h, n, d],
      [b, h, n, d],
    ],
    [[b, h, m, d]],
    { attrs: { scale: inputs.scale } },
  );
  const session = await createSessionFromContainer(
    gpu,
    await openGraphModel(graph),
    GRAPH_NAME,
    options,
  );
  try {
    const outputs = await session.run({
      x0: { dtype: "f32", shape: [b, h, m, d], data: inputs.q },
      x1: { dtype: "f32", shape: [b, h, n, d], data: inputs.k },
      x2: { dtype: "f32", shape: [b, h, n, d], data: inputs.v },
    });
    const y = outputs["y"];
    assert(y !== undefined && y.dtype === "f32", "the attention output is missing");
    return {
      output: y.data,
      census: requireCensus(session.diagnostics().lastRunPipelines, "attention"),
    };
  } finally {
    await session.dispose();
  }
};

/**
 * The raw kernel in one dispatch over every row, on inputs quantised on the host (the job of the
 * Session's `quantize_rows` / `strided` dispatches) — the dispatch of
 * tests/gpu_attention_online_i8a8_test.ts.
 */
const runRawKernel = async (
  gpu: GpuContext,
  shape: Shape,
  inputs: Inputs,
  reduce: AttentionOnlineReduce,
): Promise<Float32Array<ArrayBuffer>> => {
  const { m, n, d } = shape;
  const batch = shape.b * shape.h;
  const query = quantizeRowsReference(inputs.q, batch * m, d);
  const key = quantizeRowsReference(inputs.k, batch * n, d);
  const values = quantizeRowsReference(transposeValues(inputs.v, batch, n, d), batch * d, n);
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const host = (source: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, source);
      return buffer;
    };
    const bytes = batch * m * d * 4;
    const out = arena.allocRegion(bytes);
    const pipelineKey = attentionOnlineI8a8Key(true, d, reduce);
    const { pipeline, layout } = await cache.get(
      pipelineKey,
      attentionOnlineI8a8Wgsl(true, d, reduce),
    );
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionOnlineI8a8Params(m, n, 0, inputs.scale));
    const buffers = [
      params,
      host(packWords(query.q)),
      host(packWords(key.q)),
      host(packWords(values.q)),
      out,
      host(query.scale),
      host(key.scale),
      host(values.scale),
    ];
    scheduler.dispatch(
      pipeline,
      gpu.device.createBindGroup({
        layout,
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
      [Math.ceil(m / ATTENTION_ONLINE_ROWS), batch, 1],
      pipelineKey,
    );
    await scheduler.flush();
    const staging = gpu.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return new Float32Array(copy);
  } finally {
    await arena.destroy();
  }
};

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

const assertBitIdentical = (a: Float32Array, b: Float32Array, where: string): void => {
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  assertEquals(ua.length, ub.length, `${where}: length`);
  for (let i = 0; i < ua.length; i += 1) {
    if (ua[i] !== ub[i]) throw new Error(`${where}: bits differ at ${i} (${a[i]} vs ${b[i]})`);
  }
};

/** The key family (before the first `:`). */
const familyOf = (key: string): string => key.slice(0, key.indexOf(":"));

const ONLINE_FAMILY = familyOf(attentionOnlineI8a8Key(true, 128, "wg"));
const THREE_PASS_FAMILIES: ReadonlySet<string> = new Set([
  "attention_qk",
  "attention_stats",
  "attention_pv",
]);

/**
 * The census of an online node: only the online family among the attention kernels, one key in the
 * expected reduction form, one dispatch per window, and the four quantisation dispatches of a8.
 */
const assertOnlineCensus = (
  census: readonly CensusRow[],
  shape: Shape,
  reduce: AttentionOnlineReduce,
  where: string,
): void => {
  const keys = census.map((row) => `${row.key}×${row.dispatchCount}`).join(" / ");
  const online = census.filter((row) => familyOf(row.key) === ONLINE_FAMILY);
  assertEquals(online.length, 1, `${where}: one online key — ${keys}`);
  assert(online[0].key.endsWith(`:${reduce}`), `${where}: the ${reduce} form — ${keys}`);
  assertEquals(
    online[0].dispatchCount,
    planAttentionOnlineWindows(shape.m, { batch: shape.b * shape.h, cols: shape.n, depth: shape.d })
      .length,
    `${where}: one dispatch per window — ${keys}`,
  );
  assertEquals(
    countDispatches(census, (key) => THREE_PASS_FAMILIES.has(familyOf(key))),
    0,
    `${where}: no three-pass dispatch — ${keys}`,
  );
  assertEquals(
    countDispatches(census, (key) => key.startsWith(QUANTIZE_ROWS_KEY)),
    3,
    `${where}: q / k / Vᵀ quantisation — ${keys}`,
  );
  assertEquals(
    countDispatches(census, stridedKey({ dtype: "f32" })),
    1,
    `${where}: the Vᵀ permute — ${keys}`,
  );
};

/** Whether the adapter advertises everything the subgroup form needs. */
const subgroupsSupported = async (): Promise<boolean> => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return false;
  return (
    adapter.features.has("subgroups") &&
    adapter.features.has("subgroup-size-control") &&
    new Set(navigator.gpu.wgslLanguageFeatures).has("subgroup_id")
  );
};

/**
 * The reduction form the Session must take on this device (the same function as buildAttention);
 * `undefined` on a device that holds neither form (no subgroups and less workgroup storage than the
 * workgroup-memory form needs — the route falls back to the three dispatches).
 */
const expectedReduce = (gpu: GpuContext, shape: Shape): AttentionOnlineReduce | undefined =>
  attentionRoute(
    { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape.d,
    shape.n,
    attentionDeviceFacts(gpu),
  ).online;

describe.skipIf(!GPU_AVAILABLE)("online int8 attention through the Session (ADR 0132)", () => {
  const ELIGIBLE: Shape = { b: 1, h: 2, m: 150, n: 200, d: 128 };

  it("(i)(ii) D = 128: only attention_online runs, bit-identical to the raw kernel, in both reduction forms", async () => {
    const inputs = makeInputs(ELIGIBLE);
    const advertised = await subgroupsSupported();
    for (const subgroups of [false, "available"] as const) {
      const gpu = await acquireGpu({ subgroups });
      try {
        const reduce = expectedReduce(gpu, ELIGIBLE);
        if (subgroups === "available" && advertised) {
          assertEquals(reduce, "sg32", "subgroups 'available' on an adapter that has them");
        }
        const where = `subgroups=${subgroups} (${reduce ?? "fallback"})`;
        const online = await runSession(gpu, ELIGIBLE, inputs, ONLINE);
        if (reduce === undefined) {
          // Neither form fits this device: the route falls back (its values are case (iii)'s).
          assertEquals(
            countDispatches(online.census, (key) => familyOf(key) === ONLINE_FAMILY),
            0,
            `${where}: the online kernel ran on a device the route rejects`,
          );
          console.log(`[attention_online session] ${where}: the device holds neither form`);
          continue;
        }
        assertOnlineCensus(online.census, ELIGIBLE, reduce, where);
        console.log(
          `[attention_online session] ${where}: ${online.census.map((row) => row.key).join(" / ")}`,
        );
        assert(new Set(online.output).size > 1, `${where}: the output is constant`);
        // The raw kernel in the workgroup-memory form: the two forms are bit-identical (the kernel
        // test holds it), so the Session's form does not matter for the bits.
        const raw = await runRawKernel(gpu, ELIGIBLE, inputs, "wg");
        assertBitIdentical(online.output, raw, `${where}: Session vs raw kernel`);
      } finally {
        gpu.destroy();
      }
    }
  });

  it("(iii) D = 96 falls back to the three a8 dispatches with S in f32, bit-identical to a8", async () => {
    const shape: Shape = { b: 1, h: 2, m: 150, n: 200, d: 96 };
    const inputs = makeInputs(shape);
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      const requested = await runSession(gpu, shape, inputs, ONLINE);
      const a8 = await runSession(gpu, shape, inputs, A8);
      const keys = requested.census.map((row) => row.key).join(" / ");
      assertEquals(
        countDispatches(requested.census, (key) => familyOf(key) === ONLINE_FAMILY),
        0,
        `no online dispatch — ${keys}`,
      );
      for (const family of THREE_PASS_FAMILIES) {
        assertEquals(
          countDispatches(requested.census, (key) => familyOf(key) === family),
          1,
          `${family} — ${keys}`,
        );
      }
      // The same pipelines as the a8 Session (①③ i8a8, S in f32 — no ':s16')
      assertEquals(
        requested.census.map((row) => row.key),
        a8.census.map((row) => row.key),
      );
      assertBitIdentical(requested.output, a8.output, "D = 96: 'none' vs a8");
    } finally {
      gpu.destroy();
    }
  });

  it("(iv) a shape over one dispatch's budget runs one dispatch per window, bit-identical to one raw dispatch", async (context) => {
    // 2⁴⁰ / (65,536 keys · 128 · 2) = 65,536 rows per window: 65,568 rows make two windows, and
    // 65,568 = 65,536 + 32 leaves the second window's only workgroup partial and clamped at M
    const shape: Shape = { b: 1, h: 1, m: 65568, n: 65536, d: 128 };
    const windows = planAttentionOnlineWindows(shape.m, {
      batch: 1,
      cols: shape.n,
      depth: shape.d,
    });
    assertEquals(windows.length, 2);
    const inputs = makeInputs(shape);
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      const reduce = expectedReduce(gpu, shape);
      if (reduce === undefined) {
        context.skip();
        return;
      }
      const online = await runSession(gpu, shape, inputs, ONLINE);
      assertOnlineCensus(online.census, shape, reduce, "two windows");
      const raw = await runRawKernel(gpu, shape, inputs, "wg");
      assertBitIdentical(online.output, raw, "two windows: Session vs one raw dispatch");
    } finally {
      gpu.destroy();
    }
  });
});

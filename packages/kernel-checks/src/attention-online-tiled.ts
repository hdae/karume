/**
 * The tiled key axis of the online int8 attention (perf-ledger K-106 — a measure-first candidate;
 * `AttentionOnlineKeyWords` `tiled` in src/kernels/attention-online-i8a8.ts, its producer
 * `quantizeRowsTiledWgsl` in src/kernels/quantize-rows.ts): the producer writes Vᵀ already in the
 * kernel's key order, so the kernel's owner fill is one `vec4` load instead of a 16-byte gather. The
 * claim is bit identity with today's pair, checked in two halves:
 *
 * - **producer**: the tiled `quantize_rows` against the host permutation of the production
 *   producer's bytes (`attentionValueQuantization` — the plain form at `N % 4 == 0`, the padded one
 *   otherwise) on the same f32 Vᵀ, word for word, with the scales bit for bit, and nothing written
 *   past the tiled rows. The bytes are the same lattice values only by measured behaviour (the
 *   per-component `x / s` of two different shaders), so this runs on every backend.
 * - **kernel**: the online kernel reading the tiled Vᵀ against today's kernel reading the plain /
 *   padded Vᵀ of the same quantised values, every O word identical — both reduction forms (one part
 *   each), both output layouts, dp4a and the emulated dot product.
 *
 * Fault injection (the node test only, through {@link AttentionOnlineTiledCheckOptions.mutate}): a
 * wrong word index in the producer and a wrong lane in the kernel's load each turn their part red.
 */

import { gridStrideWorkgroups } from "../../runtime/src/codegen/dispatch.ts";
import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../../runtime/src/gpu/webgpu-constants.ts";
import {
  ATTENTION_ONLINE_KEYS,
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineKeyWords,
  type AttentionOnlineOutput,
  type AttentionOnlineReduce,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
  attentionOnlineI8a8WorkgroupStorageBytes,
  attentionOnlineKeyWords,
  attentionOnlineValueRowWords,
} from "../../runtime/src/kernels/attention-online-i8a8.ts";
import { dp4aAvailable } from "../../runtime/src/kernels/linear-i8a8.ts";
import {
  quantizeRowsTiledGeometry,
  quantizeRowsTiledParams,
  quantizeRowsTiledRowWords,
  quantizeRowsTiledWgsl,
} from "../../runtime/src/kernels/quantize-rows.ts";
import { quantizeRowsReference } from "../../runtime/src/reference/i8a8.ts";
import { transposeValues } from "../../runtime/src/reference/attention-online.ts";
import { attentionDeviceFacts } from "../../runtime/src/runtime/attention-device.ts";
import { attentionRoute } from "../../runtime/src/runtime/attention-route.ts";
import { attentionValueQuantization } from "../../runtime/src/runtime/recipe-builders/attention.ts";
import {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  compareCount,
  compareFill,
  compareWords,
  type ExpectedCases,
  notRunPart,
  scopedGpuWork,
  throwIfAborted,
} from "./result.ts";

/** Words per key tile of a Vᵀ column (64 keys, four per word). */
const TILE_WORDS = ATTENTION_ONLINE_KEYS / 4;
/** The value a written word cannot hold by chance in these fixtures (checked as left in place). */
const POISON = 0xa5a5a5a5;
/** Words after the tiled rows that MUST stay poisoned (an out-of-range store lands here first). */
const GUARD_WORDS = 64;

/**
 * Host permutation of a plain / padded Vᵀ (`rows` rows of `ceil(dim / 4)` words, four consecutive
 * keys per word, zero past `dim`) into the tiled layout: word `q` of a row packs the keys
 * `64·(q/16) + q%16 + 16·i` (zero from `dim` on), and the words of four consecutive rows are
 * interleaved — row `r`'s word `q` sits at `((r/4)·n4p + q)·4 + r%4`, `n4p = 16·ceil(dim / 64)`.
 * The oracle of both parts. MUST: `rows % 4 == 0`.
 */
export const tiledValueWords = (
  plain: Uint32Array<ArrayBuffer>,
  rows: number,
  dim: number,
): Uint32Array<ArrayBuffer> => {
  const plainWords = attentionOnlineValueRowWords(dim);
  const tiledWords = quantizeRowsTiledRowWords(dim);
  if (rows % 4 !== 0 || plain.length !== rows * plainWords) {
    throw new Error(
      `tiledValueWords: ${plain.length} words for ${rows} rows of ${dim} (rows must be a multiple of 4)`,
    );
  }
  const byteOf = (row: number, key: number): number =>
    key < dim ? (plain[row * plainWords + (key >> 2)] >>> (8 * (key & 3))) & 0xff : 0;
  const out = new Uint32Array(rows * tiledWords);
  for (let row = 0; row < rows; row += 1) {
    for (let q = 0; q < tiledWords; q += 1) {
      const first = Math.floor(q / TILE_WORDS) * ATTENTION_ONLINE_KEYS + (q % TILE_WORDS);
      let word = 0;
      for (let i = 0; i < 4; i += 1) word |= byteOf(row, first + 16 * i) << (8 * i);
      out[(Math.floor(row / 4) * tiledWords + q) * 4 + (row % 4)] = word >>> 0;
    }
  }
  return out;
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const seeded = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return state / 0x1_0000_0000;
  };
};

/**
 * The producer's row lengths: one key tile not reached (5, 40), N % 4 != 0 (5, 65, 203, 4,106),
 * N % 16 != 0 (40, 200), N % 64 != 0 at a multiple of 16 (7,920 and 27,280 — the Wan2.2 self
 * attention at 33 / 121 frames), whole tiles (512 — the cross attention). Rows are `(head, column)`.
 */
const PRODUCER_DIMS: readonly number[] = [5, 40, 65, 200, 203, 512, 4106, 7920, 27280];
/** 36 rows: nine groups of four, and a partial last workgroup at every lined-up geometry. */
const PRODUCER_ROWS = 36;
/** The production grid, and one workgroup walking every row through the grid-stride loop. */
const PRODUCER_GRIDS = ["production grid", "one workgroup"] as const;

/**
 * Rows 0–3 are the edge rows: all zero (s = tiny, every byte 0), ±3.0e38 alternating (the scale at
 * the top of f32, every byte ±127), all +1 (every element at the row max) and ±2.5 alternating;
 * the others are pseudo-random with a per-row gain and a spike every 37 elements.
 */
const producerInput = (dim: number): Float32Array<ArrayBuffer> => {
  const next = seeded(0x51ed + dim);
  const x = new Float32Array(PRODUCER_ROWS * dim);
  for (let row = 0; row < PRODUCER_ROWS; row += 1) {
    for (let i = 0; i < dim; i += 1) {
      const sign = i % 2 === 0 ? 1 : -1;
      x[row * dim + i] =
        row === 0
          ? 0
          : row === 1
            ? sign * 3.0e38
            : row === 2
              ? 1
              : row === 3
                ? sign * 2.5
                : (next() * 2 - 1) * (1 + (row % 7) * 0.9) * (i % 37 === row % 37 ? 9 : 1);
    }
  }
  return x;
};

type KernelShape = {
  readonly name: string;
  /** B·H folded. */
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * The shapes of the four-point check (tests/gpu_attention_online_i8a8_test.ts): every eligible
 * depth, a single key tile, key and row tiles with remainders, the cross-attention keys, a long key
 * axis, and N % 4 = 1 / 2 / 3 (today's padded key axis).
 */
const KERNEL_SHAPES: readonly KernelShape[] = [
  { name: "single tile D64", batch: 2, m: 64, n: 64, d: 64 },
  { name: "remainders D128", batch: 3, m: 150, n: 200, d: 128 },
  { name: "cross D128", batch: 2, m: 70, n: 512, d: 128 },
  { name: "long keys D64", batch: 2, m: 100, n: 1000, d: 64 },
  { name: "N%4=1 D64", batch: 2, m: 70, n: 65, d: 64 },
  { name: "N%4=2 D128 (S₋ 4,106)", batch: 2, m: 64, n: 4106, d: 128 },
  { name: "N%4=3 D128", batch: 3, m: 150, n: 203, d: 128 },
];
const OUTPUTS: readonly AttentionOnlineOutput[] = ["bhsd", "bshd"];
const DOTS = ["dp4a", "emu"] as const;

const PRODUCER_PART = "attention-online-tiled-producer";
const kernelPart = (reduce: AttentionOnlineReduce): string => `attention-online-tiled-${reduce}`;

export const ATTENTION_ONLINE_TILED_CASES = {
  [PRODUCER_PART]: PRODUCER_DIMS.length * PRODUCER_GRIDS.length,
  [kernelPart("wg")]: KERNEL_SHAPES.length * OUTPUTS.length * DOTS.length,
  [kernelPart("sg32")]: KERNEL_SHAPES.length * OUTPUTS.length * DOTS.length,
} as const satisfies ExpectedCases;

/** Packs flat i8 rows of `dim` into `ceil(dim / 4)` words each, zero past `dim` (the producer's layout). */
const packRows = (
  values: Int8Array<ArrayBuffer>,
  rows: number,
  dim: number,
): Uint32Array<ArrayBuffer> => {
  const words = attentionOnlineValueRowWords(dim);
  const out = new Uint32Array(rows * words);
  for (let row = 0; row < rows; row += 1) {
    for (let i = 0; i < dim; i += 1) {
      out[row * words + (i >> 2)] |= (values[row * dim + i] & 0xff) << (8 * (i & 3));
    }
  }
  return out;
};

type KernelInputs = {
  readonly qq: Uint32Array<ArrayBuffer>;
  readonly qs: Float32Array<ArrayBuffer>;
  readonly kq: Uint32Array<ArrayBuffer>;
  readonly ks: Float32Array<ArrayBuffer>;
  /** Today's Vᵀ (plain at N % 4 == 0, padded otherwise) and the same bytes in the tiled layout. */
  readonly vqPlain: Uint32Array<ArrayBuffer>;
  readonly vqTiled: Uint32Array<ArrayBuffer>;
  readonly vs: Float32Array<ArrayBuffer>;
  readonly scale: number;
};

/** Gaussian q / k / v with per-head statistics that differ, quantised on the host (the TS references). */
const kernelInputs = (shape: KernelShape): KernelInputs => {
  const { batch, m, n, d } = shape;
  const next = seeded(7 + d + n);
  const gaussian = (): number => {
    const u = next() || 1e-9;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  const q = new Float32Array(batch * m * d);
  const k = new Float32Array(batch * n * d);
  const v = new Float32Array(batch * n * d);
  for (let b = 0; b < batch; b += 1) {
    for (let i = 0; i < m * d; i += 1) q[b * m * d + i] = gaussian() * (0.6 + 0.5 * b);
    for (let i = 0; i < n * d; i += 1) k[b * n * d + i] = gaussian() * (1.2 - 0.3 * b);
    for (let i = 0; i < n * d; i += 1) v[b * n * d + i] = gaussian() * (0.8 + 0.4 * b);
  }
  const query = quantizeRowsReference(q, batch * m, d);
  const key = quantizeRowsReference(k, batch * n, d);
  const values = quantizeRowsReference(transposeValues(v, batch, n, d), batch * d, n);
  const vqPlain = packRows(values.q, batch * d, n);
  return {
    qq: packRows(query.q, batch * m, d),
    qs: query.scale,
    kq: packRows(key.q, batch * n, d),
    ks: key.scale,
    vqPlain,
    vqTiled: tiledValueWords(vqPlain, batch * d, n),
    vs: values.scale,
    scale: Math.sqrt(1 / Math.sqrt(d)) * 1.5,
  };
};

// ---------------------------------------------------------------------------
// Raw dispatches
// ---------------------------------------------------------------------------

/** Buffers, pipelines and one-shot dispatches of one case; `release` destroys what it made. */
const dispatcher = (gpu: GpuContext) => {
  const { device } = gpu;
  const owned: GPUBuffer[] = [];
  const buffer = (
    value: number | ArrayBufferView<ArrayBuffer>,
    usage: number = U.STORAGE | U.COPY_DST | U.COPY_SRC,
  ): GPUBuffer => {
    const size = typeof value === "number" ? value : value.byteLength;
    const made = device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
    owned.push(made);
    if (typeof value !== "number") device.queue.writeBuffer(made, 0, value);
    return made;
  };
  const poisoned = (words: number): GPUBuffer =>
    buffer(new Uint32Array(Math.ceil(words / 4) * 4).fill(POISON));
  const run = async (
    code: string,
    bindings: readonly GPUBuffer[],
    workgroups: readonly [number, number, number],
  ): Promise<void> => {
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((resource, binding) => ({ binding, resource: { buffer: resource } })),
      }),
    );
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  const read = async (source: GPUBuffer, words: number): Promise<Uint32Array<ArrayBuffer>> => {
    const staging = buffer(words * 4, U.MAP_READ | U.COPY_DST);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, words * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE.READ);
    const out = new Uint32Array(staging.getMappedRange(0, words * 4).slice(0));
    staging.unmap();
    return out;
  };
  const release = (): void => {
    for (const made of owned) made.destroy();
  };
  return { buffer, poisoned, run, read, release };
};

/** Rewrites a WGSL text (a fault); a mutation that changes nothing is a broken anchor and throws. */
const mutated = (
  wgsl: string,
  mutate: ((wgsl: string) => string) | undefined,
  where: string,
): string => {
  if (mutate === undefined) return wgsl;
  const out = mutate(wgsl);
  if (out === wgsl) throw new Error(`${where}: the mutation left the WGSL unchanged`);
  return out;
};

export type AttentionOnlineTiledCheckOptions = CheckOptions & {
  /** Fault injection (node test only): rewrites the tiled producer's / the tiled kernel's WGSL. */
  readonly mutate?: {
    readonly producer?: (wgsl: string) => string;
    readonly kernel?: (wgsl: string) => string;
  };
};

// ---------------------------------------------------------------------------
// The parts
// ---------------------------------------------------------------------------

const checkProducer = async (
  gpu: GpuContext,
  options: AttentionOnlineTiledCheckOptions,
): Promise<CheckPart> => {
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  const total = ATTENTION_ONLINE_TILED_CASES[PRODUCER_PART];
  const cases: CheckCase[] = [];
  for (const dim of PRODUCER_DIMS) {
    const x = producerInput(dim);
    const rows = PRODUCER_ROWS;
    for (const grid of PRODUCER_GRIDS) {
      throwIfAborted(options);
      const label = `N ${dim}, ${rows} rows, ${grid}`;
      const result = await scopedGpuWork(gpu, label, async () => {
        const gpuWork = dispatcher(gpu);
        try {
          // The production producer of today's key axis for this N, on the production grid
          const plain = attentionValueQuantization(rows, dim, attentionOnlineKeyWords(dim));
          const plainWords = rows * attentionOnlineValueRowWords(dim);
          const input = gpuWork.buffer(x);
          const plainQ = gpuWork.poisoned(plainWords);
          const plainS = gpuWork.poisoned(rows);
          await gpuWork.run(
            plain.wgsl,
            [gpuWork.buffer(plain.params, U.UNIFORM | U.COPY_DST), input, plainQ, plainS],
            [gridStrideWorkgroups(rows, plain.geometry.rowsPerGroup, limit), 1, 1],
          );
          const geometry = quantizeRowsTiledGeometry(dim);
          const tiledWords = rows * quantizeRowsTiledRowWords(dim);
          const tiledQ = gpuWork.poisoned(tiledWords + GUARD_WORDS);
          const tiledS = gpuWork.poisoned(rows);
          await gpuWork.run(
            mutated(quantizeRowsTiledWgsl(geometry), options.mutate?.producer, label),
            [
              gpuWork.buffer(quantizeRowsTiledParams(rows, dim), U.UNIFORM | U.COPY_DST),
              input,
              tiledQ,
              tiledS,
            ],
            [
              grid === "one workgroup"
                ? 1
                : gridStrideWorkgroups(rows, geometry.rowsPerGroup, limit),
              1,
              1,
            ],
          );
          return {
            plainQ: await gpuWork.read(plainQ, plainWords),
            plainS: await gpuWork.read(plainS, rows),
            tiledQ: await gpuWork.read(tiledQ, tiledWords + GUARD_WORDS),
            tiledS: await gpuWork.read(tiledS, rows),
            tiledWords,
          };
        } finally {
          gpuWork.release();
        }
      });
      const { plainQ, plainS, tiledQ, tiledS, tiledWords } = result;
      cases.push({
        label,
        criteria: [
          compareWords(
            "tiled words vs the host permutation of the production producer's words",
            tiledQ.slice(0, tiledWords),
            tiledValueWords(plainQ, rows, dim),
          ),
          compareWords("tiled scales vs the production producer's (bits)", tiledS, plainS),
          compareFill(
            `the ${GUARD_WORDS} words past the tiled rows stay unwritten`,
            tiledQ.slice(tiledWords),
            POISON,
          ),
          compareCount(
            "words of the production producer left unwritten (the oracle ran)",
            plainQ.filter((word) => word === POISON).length,
            0,
          ),
        ],
      });
      options.onProgress?.({ part: PRODUCER_PART, done: cases.length, total });
    }
  }
  return {
    id: PRODUCER_PART,
    title: "Tiled Vᵀ producer vs the host permutation of today's producer (words, scale bits)",
    cases,
    requirements: [],
  };
};

/** What the device lacks for a kernel part (`undefined` = it can run it). */
const kernelShortfall = (gpu: GpuContext, reduce: AttentionOnlineReduce): string | undefined => {
  if (!dp4aAvailable(gpu.wgslLanguageFeatures)) {
    return "WGSL packed_4x8_integer_dot_product is not listed (the dp4a cases cannot run)";
  }
  const storage = attentionOnlineI8a8WorkgroupStorageBytes(128, reduce);
  if (storage > gpu.limits.maxComputeWorkgroupStorageSize) {
    return `the ${reduce} form at D = 128 declares ${storage} B of workgroup storage, the device grants ${gpu.limits.maxComputeWorkgroupStorageSize} B`;
  }
  if (reduce === "sg32") {
    const form = attentionRoute(
      { attentionCompute: "a8", attentionScoreStorage: "none" },
      128,
      512,
      attentionDeviceFacts(gpu),
    ).online;
    if (form !== "sg32") {
      return `the online attention takes the ${form ?? "no online"} form on this device; sg32 needs subgroups, subgroup-size-control, WGSL subgroup_id and a linear lane mapping`;
    }
  }
  return undefined;
};

const runKernel = async (
  gpu: GpuContext,
  shape: KernelShape,
  inputs: KernelInputs,
  form: {
    readonly dp4a: boolean;
    readonly reduce: AttentionOnlineReduce;
    readonly output: AttentionOnlineOutput;
    readonly keyWords: AttentionOnlineKeyWords;
  },
  mutate: ((wgsl: string) => string) | undefined,
  where: string,
): Promise<Uint32Array<ArrayBuffer>> => {
  const { batch, m, n, d } = shape;
  const gpuWork = dispatcher(gpu);
  try {
    const tiled = form.keyWords === "tiled";
    const o = gpuWork.poisoned(batch * m * d);
    // The token-major O folds B·H as B = 1, H = batch
    const heads = form.output === "bshd" ? batch : undefined;
    await gpuWork.run(
      mutated(
        attentionOnlineI8a8Wgsl(form.dp4a, d, form.reduce, form.output, form.keyWords),
        tiled ? mutate : undefined,
        where,
      ),
      [
        gpuWork.buffer(
          attentionOnlineI8a8Params(m, n, 0, inputs.scale, form.keyWords, heads),
          U.UNIFORM | U.COPY_DST,
        ),
        gpuWork.buffer(inputs.qq),
        gpuWork.buffer(inputs.kq),
        gpuWork.buffer(tiled ? inputs.vqTiled : inputs.vqPlain),
        o,
        gpuWork.buffer(inputs.qs),
        gpuWork.buffer(inputs.ks),
        gpuWork.buffer(inputs.vs),
      ],
      [Math.ceil(m / ATTENTION_ONLINE_ROWS), batch, 1],
    );
    return await gpuWork.read(o, batch * m * d);
  } finally {
    gpuWork.release();
  }
};

const checkKernel = async (
  gpu: GpuContext,
  reduce: AttentionOnlineReduce,
  options: AttentionOnlineTiledCheckOptions,
): Promise<CheckPart> => {
  const id = kernelPart(reduce);
  const title = `Online attention (${reduce}) reading the tiled Vᵀ vs today's key axis (O words)`;
  const shortfall = kernelShortfall(gpu, reduce);
  if (shortfall !== undefined) return notRunPart(id, title, shortfall);
  const total = ATTENTION_ONLINE_TILED_CASES[id];
  const cases: CheckCase[] = [];
  for (const shape of KERNEL_SHAPES) {
    const inputs = kernelInputs(shape);
    for (const output of OUTPUTS) {
      for (const dot of DOTS) {
        throwIfAborted(options);
        const label = `${shape.name} (B·H ${shape.batch}, M ${shape.m}, N ${shape.n}), ${output}, ${dot}`;
        const form = { dp4a: dot === "dp4a", reduce, output };
        const [today, tiled] = await scopedGpuWork(gpu, label, async () => [
          await runKernel(
            gpu,
            shape,
            inputs,
            { ...form, keyWords: attentionOnlineKeyWords(shape.n) },
            undefined,
            label,
          ),
          await runKernel(
            gpu,
            shape,
            inputs,
            { ...form, keyWords: "tiled" },
            options.mutate?.kernel,
            label,
          ),
        ]);
        cases.push({
          label,
          criteria: [
            compareWords("O words: tiled vs today's key axis", tiled, today),
            compareCount(
              "O words today's kernel left unwritten (the comparison is not vacuous)",
              today.filter((word) => word === POISON).length,
              0,
            ),
          ],
        });
        options.onProgress?.({ part: id, done: cases.length, total });
      }
    }
  }
  return { id, title, cases, requirements: [] };
};

/**
 * Runs the three parts: the producer, then the kernel in the wg form and in the sg32 form. A kernel
 * part the device cannot run (no dp4a language feature, too little workgroup storage, or — for sg32
 * — no online sg32 form) comes back with `notRun`.
 */
export const checkAttentionOnlineTiled = async (
  gpu: GpuContext,
  options: AttentionOnlineTiledCheckOptions = {},
): Promise<CheckPart[]> => [
  await checkProducer(gpu, options),
  await checkKernel(gpu, "wg", options),
  await checkKernel(gpu, "sg32", options),
];

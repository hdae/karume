// The channel-major form of `quantize_rows` (perf-ledger V-6 — conv3d w8a8's activation quantiser,
// ADR 0133 addendum 2026-10-26) against the composition it replaced in the conv3d w8a8 recipe: the
// strided f32 transpose `x[C][P]` → `[P][C]` followed by `quantize_rows` in its
// `quantizeRowsGeometry(C)` form. Word for word on `xq` and `xs`, from the production generators
// (the keys, WGSL, params and workgroup counts a Session uses).
//
// The form changes only the order of memory accesses and the order in which a pixel's absolute
// maximum is folded (order-free for finite values), so the claim is bit identity at every
// geometry — the one production picks and ones it does not. Whether the backend keeps it is
// visible only here. No adapter = explicit SKIP.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../src/codegen/strided.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { CONV3D_I8A8_QUANTIZE_GEOMETRY } from "../src/kernels/conv3d-i8a8.ts";
import {
  type QuantizeRowsChannelMajorGeometry,
  quantizeRowsChannelMajorKey,
  quantizeRowsChannelMajorWgsl,
  quantizeRowsChannelMajorWorkgroups,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../src/kernels/quantize-rows.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
/** Fills the fused kernel's outputs before it runs, so a word it does not write cannot match. */
const POISON = 0xdeadbeef;
/** f32 bits of the scale floor `tiny` (an all-zero pixel's scale). */
const TINY_BITS = 0x00800000;

/** A deterministic pseudo-random `x[C][P]` whose magnitude varies by pixel (distinct scales). */
const channelMajor = (channels: number, pixels: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(channels * pixels);
  let state = (0x9e3779b9 ^ (channels * 7919 + pixels)) >>> 0;
  for (let i = 0; i < out.length; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const magnitude = 1 + ((i % pixels) % 7) * 0.5;
    out[i] = ((state / 2 ** 32) * 2 - 1) * magnitude;
  }
  return out;
};

/** One test bench: a scheduler, a cache, an arena, and helpers to dispatch and read back. */
const bench = (gpu: GpuContext) => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const host = (source: Float32Array<ArrayBuffer>): GPUBuffer => {
    const buffer = arena.allocHostWritten(Math.max(16, source.byteLength), STORAGE_IN);
    gpu.device.queue.writeBuffer(buffer, 0, source);
    return buffer;
  };
  const region = (bytes: number): GPUBuffer => arena.allocRegion(Math.max(16, bytes));
  /** A readable output region filled with {@link POISON} words. */
  const poisoned = (words: number): GPUBuffer => {
    const buffer = region(words * 4);
    gpu.device.queue.writeBuffer(buffer, 0, new Uint32Array(words).fill(POISON));
    return buffer;
  };
  const run = async (
    key: string,
    wgsl: string,
    params: Uint32Array<ArrayBuffer>,
    usage: number,
    buffers: readonly GPUBuffer[],
    groups: number,
  ): Promise<void> => {
    const { pipeline, layout } = await cache.get(key, wgsl);
    const uniform = arena.allocHostWritten(Math.max(16, params.byteLength), usage);
    gpu.device.queue.writeBuffer(uniform, 0, params);
    const bindGroup = gpu.device.createBindGroup({
      layout,
      entries: [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
  };
  const read = async (buffer: GPUBuffer, words: number): Promise<Uint32Array<ArrayBuffer>> => {
    await scheduler.flush();
    const size = Math.max(4, words * 4);
    const staging = gpu.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return new Uint32Array(copy, 0, words);
  };
  return { host, region, poisoned, run, read, dispose: () => arena.destroy() };
};

const assertWords = (
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
  where: string,
  skip: ReadonlySet<number> = new Set(),
): void => {
  assertEquals(actual.length, expected.length, `${where}: length`);
  let differing = 0;
  let first = -1;
  for (let i = 0; i < actual.length; i += 1) {
    if (skip.has(i)) continue;
    if (actual[i] !== expected[i]) {
      differing += 1;
      if (first < 0) first = i;
    }
  }
  assertEquals(differing, 0, `${where}: ${differing} differing words (first at ${first})`);
};

const isNanBits = (bits: number): boolean =>
  (bits & 0x7f800000) === 0x7f800000 && (bits & 0x7fffff) !== 0;

describe.skipIf(!GPU_AVAILABLE)(
  "the channel-major quantize_rows of conv3d w8a8 (perf-ledger V-6)",
  () => {
    it("writes the words of 'strided transpose → quantize_rows' on xq and xs at every geometry (Cin 4 / 48 / 256 / 512 / 1020 / 1024, P < 64 / P % 64 ≠ 0 / 768 / large; a zero, a one-large-channel and a NaN pixel)", async () => {
      type Case = {
        readonly channels: number;
        readonly pixels: number;
        /** Clamps the workgroup count to walk the pixel-block grid stride. */
        readonly groups?: number;
      };
      const cases: Case[] = [];
      for (const channels of [4, 48, 256, 512, 1020, 1024]) {
        for (const pixels of [37, 200, 768, 9001]) cases.push({ channels, pixels });
      }
      cases.push(
        { channels: 1020, pixels: 9001, groups: 3 },
        { channels: 48, pixels: 768, groups: 2 },
      );
      // The production geometry, and ones it does not pick: the outputs must not depend on it.
      const geometries: readonly QuantizeRowsChannelMajorGeometry[] = [
        CONV3D_I8A8_QUANTIZE_GEOMETRY,
        { pixels: 4 },
        { pixels: 64 },
      ];
      const gpu = await acquireGpu();
      try {
        const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
        for (const { channels, pixels, groups } of cases) {
          const where = `Cin ${channels} P ${pixels}${groups === undefined ? "" : ` groups ${groups}`}`;
          const quads = channels / 4;
          const words = pixels * quads;
          const x = channelMajor(channels, pixels);
          const zeroPixel = 1;
          const largePixel = pixels - 1;
          const largeChannel = channels >> 1;
          const nanPixel = pixels >> 1;
          for (let c = 0; c < channels; c += 1) x[c * pixels + zeroPixel] = 0;
          x[largeChannel * pixels + largePixel] = 1e4;
          x[(channels - 1) * pixels + nanPixel] = Number.NaN;
          const test = bench(gpu);
          try {
            const src = test.host(x);
            // The composition the recipe ran: transpose to [P][C], then quantize the rows.
            const transposed = test.region(x.byteLength);
            const strided = { dtype: "f32" } as const;
            await test.run(
              stridedKey(strided),
              stridedWgsl(strided),
              stridedParams([pixels, channels], permuteSrcStrides([channels, pixels], [1, 0]), 0),
              STORAGE_IN,
              [src, transposed],
              gridStrideWorkgroups(pixels * channels, STRIDED_WORKGROUP_SIZE, limit),
            );
            const rowGeometry = quantizeRowsGeometry(channels);
            const rowQ = test.region(words * 4);
            const rowS = test.region(pixels * 4);
            await test.run(
              quantizeRowsKey(rowGeometry),
              quantizeRowsWgsl(rowGeometry),
              quantizeRowsParams(pixels, channels),
              UNIFORM_IN,
              [transposed, rowQ, rowS],
              gridStrideWorkgroups(pixels, rowGeometry.rowsPerGroup, limit),
            );
            const expectedQ = await test.read(rowQ, words);
            const expectedS = await test.read(rowS, pixels);
            for (const geometry of geometries) {
              const at = `${where} ${quantizeRowsChannelMajorKey(geometry)}`;
              const fusedQ = test.poisoned(words);
              const fusedS = test.poisoned(pixels);
              await test.run(
                quantizeRowsChannelMajorKey(geometry),
                quantizeRowsChannelMajorWgsl(geometry),
                quantizeRowsParams(pixels, channels),
                UNIFORM_IN,
                [src, fusedQ, fusedS],
                groups ?? quantizeRowsChannelMajorWorkgroups(pixels, geometry, limit),
              );
              const actualQ = await test.read(fusedQ, words);
              const actualS = await test.read(fusedS, pixels);
              assertWords(actualQ, expectedQ, `${at}: xq`);
              // Which NaN payload reaches a NaN pixel's scale may depend on the fold order (the
              // form's documented caveat): that word is checked for NaN, not for its bits.
              assertWords(actualS, expectedS, `${at}: xs`, new Set([nanPixel]));
              assert(isNanBits(actualS[nanPixel]), `${at}: the NaN pixel keeps a NaN scale`);
              assert(isNanBits(expectedS[nanPixel]), `${at}: the baseline's NaN pixel scale`);
              assertEquals(
                actualS[zeroPixel],
                TINY_BITS,
                `${at}: an all-zero pixel's scale is tiny`,
              );
              assertEquals(
                [...actualQ.subarray(zeroPixel * quads, (zeroPixel + 1) * quads)].filter(
                  (w) => w !== 0,
                ),
                [],
                `${at}: an all-zero pixel quantises to 0`,
              );
              const largeWord = actualQ[largePixel * quads + (largeChannel >> 2)];
              assertEquals(
                (largeWord >>> ((largeChannel & 3) * 8)) & 0xff,
                127,
                `${at}: the large channel sits on the lattice edge`,
              );
            }
          } finally {
            await test.dispose();
          }
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);

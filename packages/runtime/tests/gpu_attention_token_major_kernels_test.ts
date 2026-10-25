// The three kernels that let a fused attention read and write token-major tensors (perf-ledger F-3
// A4 / F-8 — the permutes the fusion pass lets it absorb), each against the composition it
// replaces, from the production generators (the keys, WGSL and params a Session uses), word for
// word:
//
//   - the token-major `quantize_rows` (rows read from `[B, S, H, D]`, outputs head-major) against
//     the strided permute `[B, S, H, D]` → `[B, H, S, D]` followed by the plain `quantize_rows`;
//   - the tiled transpose `[batch, s, c]` → `[batch, c, s]` against the strided permute of a
//     token-major V to head-major followed by the strided `[B·H, N, D]` → `[B·H, D, N]` copy the
//     attention recipe makes for Vᵀ;
//   - the online kernel storing O token-major against its head-major store followed by the strided
//     permute `[B, H, M, D]` → `[B, M, H, D]`.
//
// Each new form changes only addresses, so the claim is bit identity; whether the backend keeps it
// is visible only here. No adapter = explicit SKIP.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
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
import {
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineOutput,
  type AttentionOnlineReduce,
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
  attentionOnlineI8a8WorkgroupStorageBytes,
} from "../src/kernels/attention-online-i8a8.ts";
import {
  quantizeRowsBshdKey,
  quantizeRowsBshdParams,
  quantizeRowsBshdWgsl,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../src/kernels/quantize-rows.ts";
import {
  TRANSPOSE_TILED_KEY,
  TRANSPOSE_TILED_TILE,
  TRANSPOSE_TILED_WGSL,
  transposeTiledParams,
} from "../src/kernels/transpose-tiled.ts";
import { attentionDeviceFacts } from "../src/runtime/attention-route.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const HEAD_SWAP = [0, 2, 1, 3];

/** A deterministic pseudo-random f32 in about ±(1 + 3·row-dependent magnitude). */
const values = (length: number, salt: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(length);
  let state = (0x9e3779b9 ^ salt) >>> 0;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const magnitude = 1 + ((i >>> 7) % 5) * 0.75;
    out[i] = ((state / 2 ** 32) * 2 - 1) * magnitude;
  }
  return out;
};

/** One test bench: a scheduler, a cache, an arena, and helpers to dispatch and read back. */
const bench = (gpu: GpuContext) => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const host = (source: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>): GPUBuffer => {
    const buffer = arena.allocHostWritten(Math.max(16, source.byteLength), STORAGE_IN);
    gpu.device.queue.writeBuffer(buffer, 0, source);
    return buffer;
  };
  const region = (bytes: number): GPUBuffer => arena.allocRegion(Math.max(16, bytes));
  const run = async (
    key: string,
    wgsl: string,
    params: Uint32Array<ArrayBuffer>,
    usage: number,
    buffers: readonly GPUBuffer[],
    groups: readonly [number, number, number],
  ): Promise<void> => {
    const { pipeline, layout } = await cache.get(key, wgsl);
    const uniform = arena.allocHostWritten(Math.max(16, params.byteLength), usage);
    gpu.device.queue.writeBuffer(uniform, 0, params);
    const bindGroup = gpu.device.createBindGroup({
      layout,
      entries: [uniform, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    scheduler.dispatch(pipeline, bindGroup, groups, key);
  };
  /** The strided permute of an `srcShape` tensor (the copy `buildStridedCopy` makes for a permute node). */
  const permute = async (
    src: GPUBuffer,
    dst: GPUBuffer,
    srcShape: readonly number[],
    dims: readonly number[],
  ): Promise<void> => {
    const outShape = dims.map((axis) => srcShape[axis]);
    const spec = { dtype: "f32" } as const;
    const count = outShape.reduce((product, dim) => product * dim, 1);
    await run(
      stridedKey(spec),
      stridedWgsl(spec),
      stridedParams(outShape, permuteSrcStrides(srcShape, dims), 0),
      STORAGE_IN,
      [src, dst],
      [
        gridStrideWorkgroups(
          count,
          STRIDED_WORKGROUP_SIZE,
          gpu.limits.maxComputeWorkgroupsPerDimension,
        ),
        1,
        1,
      ],
    );
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
  return { host, region, run, permute, read, dispose: () => arena.destroy() };
};

const assertWords = (
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: length`);
  let differing = 0;
  let first = -1;
  for (let i = 0; i < actual.length; i += 1) {
    if (actual[i] !== expected[i]) {
      differing += 1;
      if (first < 0) first = i;
    }
  }
  assertEquals(differing, 0, `${where}: ${differing} differing words (first at ${first})`);
};

describe.skipIf(!GPU_AVAILABLE)(
  "the token-major kernels of an attention that absorbs its permutes (perf-ledger F-3 / F-8)",
  () => {
    it("quantize_rows on token-major [B, S, H, D] writes the words of 'strided permute → quantize_rows' (D 64 / 128 / 1024, H 12 / 24, S with tails, B 1 / 2)", async () => {
      type Case = {
        readonly b: number;
        readonly s: number;
        readonly h: number;
        readonly d: number;
        /** Clamps the workgroup count to walk the grid-stride row loop. */
        readonly groups?: number;
      };
      const cases: readonly Case[] = [
        { b: 1, s: 1, h: 12, d: 128 },
        { b: 1, s: 77, h: 24, d: 128 },
        { b: 2, s: 33, h: 12, d: 128 },
        { b: 2, s: 45, h: 24, d: 64 },
        { b: 1, s: 7, h: 12, d: 64 },
        { b: 2, s: 9, h: 12, d: 1024 },
        { b: 2, s: 77, h: 24, d: 128, groups: 3 },
      ];
      const gpu = await acquireGpu();
      try {
        const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
        for (const { b, s, h, d, groups } of cases) {
          const where = `B ${b} S ${s} H ${h} D ${d}${groups === undefined ? "" : ` groups ${groups}`}`;
          const rows = b * h * s;
          const x = values(rows * d, rows + d);
          // A zero row and a NaN row of batch 0 — token 0 of head 1 (token-major offset (0·H + 1)·D,
          // head-major row 1·S + 0) and token S − 1 of head 0 (offset (S − 1)·H·D, head-major row
          // S − 1) — so the special rows land on their permuted addresses.
          x.fill(0, d, 2 * d);
          x[(s - 1) * h * d + 3] = Number.NaN;
          const test = bench(gpu);
          try {
            const geometry = quantizeRowsGeometry(d);
            const src = test.host(x);
            // The composition the graph and the recipe ran: permute to [B, H, S, D], then quantize.
            const permuted = test.region(x.byteLength);
            await test.permute(src, permuted, [b, s, h, d], HEAD_SWAP);
            const plainQ = test.region(rows * d);
            const plainS = test.region(rows * 4);
            await test.run(
              quantizeRowsKey(geometry),
              quantizeRowsWgsl(geometry),
              quantizeRowsParams(rows, d),
              UNIFORM_IN,
              [permuted, plainQ, plainS],
              [gridStrideWorkgroups(rows, geometry.rowsPerGroup, limit), 1, 1],
            );
            const bshdQ = test.region(rows * d);
            const bshdS = test.region(rows * 4);
            await test.run(
              quantizeRowsBshdKey(geometry),
              quantizeRowsBshdWgsl(geometry),
              quantizeRowsBshdParams(rows, d, s, h),
              UNIFORM_IN,
              [src, bshdQ, bshdS],
              [groups ?? gridStrideWorkgroups(rows, geometry.rowsPerGroup, limit), 1, 1],
            );
            const expectedS = await test.read(plainS, rows);
            assertWords(
              await test.read(bshdQ, (rows * d) / 4),
              await test.read(plainQ, (rows * d) / 4),
              `${where}: xq`,
            );
            assertWords(await test.read(bshdS, rows), expectedS, `${where}: xs`);
            const scales = new Float32Array(expectedS.buffer);
            assert(Number.isNaN(scales[s - 1]), `${where}: the NaN row keeps a NaN scale`);
          } finally {
            await test.dispose();
          }
        }
      } finally {
        gpu.destroy();
      }
    });

    it("the tiled transpose of a token-major V is the strided 'permute to head-major → Vᵀ' copy, word for word (tails in s and c, batch 2)", async () => {
      type Case = {
        readonly b: number;
        readonly n: number;
        readonly h: number;
        readonly d: number;
      };
      const cases: readonly Case[] = [
        { b: 2, n: 45, h: 3, d: 12 },
        { b: 1, n: 64, h: 2, d: 16 },
        { b: 2, n: 70, h: 12, d: 64 },
        { b: 1, n: 33, h: 24, d: 128 },
      ];
      const gpu = await acquireGpu();
      try {
        const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
        for (const { b, n, h, d } of cases) {
          const where = `B ${b} N ${n} H ${h} D ${d}`;
          const count = b * n * h * d;
          const v = values(count, n * 31 + h);
          const test = bench(gpu);
          try {
            const src = test.host(v);
            // The graph's permute [B, N, H, D] → [B, H, N, D], then the recipe's Vᵀ copy
            // [B·H, N, D] → [B·H, D, N].
            const headMajor = test.region(count * 4);
            await test.permute(src, headMajor, [b, n, h, d], HEAD_SWAP);
            const expected = test.region(count * 4);
            await test.permute(headMajor, expected, [b * h, n, d], [0, 2, 1]);
            const tiled = test.region(count * 4);
            const channels = h * d;
            await test.run(
              TRANSPOSE_TILED_KEY,
              TRANSPOSE_TILED_WGSL,
              transposeTiledParams(n, channels),
              UNIFORM_IN,
              [src, tiled],
              [
                tiledWorkgroups(channels, TRANSPOSE_TILED_TILE, limit, where),
                tiledWorkgroups(n, TRANSPOSE_TILED_TILE, limit, where),
                tiledWorkgroups(b, 1, limit, where),
              ],
            );
            assertWords(await test.read(tiled, count), await test.read(expected, count), where);
          } finally {
            await test.dispose();
          }
        }
      } finally {
        gpu.destroy();
      }
    });

    it("the online kernel's token-major O store is its head-major store followed by the strided permute, word for word (sg32 / wg where the device holds them, dp4a and the emulated dot, row windows)", async () => {
      type Case = {
        readonly b: number;
        readonly h: number;
        readonly m: number;
        readonly n: number;
        readonly d: number;
        /** Row windows as `[offset, rows]` (one window over every row when omitted). */
        readonly windows?: readonly (readonly [number, number])[];
      };
      const cases: readonly Case[] = [
        { b: 2, h: 3, m: 150, n: 200, d: 128 },
        { b: 1, h: 2, m: 70, n: 68, d: 64 },
        {
          b: 2,
          h: 3,
          m: 150,
          n: 200,
          d: 128,
          windows: [
            [0, 2 * ATTENTION_ONLINE_ROWS],
            [2 * ATTENTION_ONLINE_ROWS, 150 - 2 * ATTENTION_ONLINE_ROWS],
          ],
        },
      ];
      const codes = (count: number, salt: number): Uint32Array<ArrayBuffer> => {
        const words = new Uint32Array(count / 4);
        let state = (0x85ebca6b ^ salt) >>> 0;
        for (let w = 0; w < words.length; w += 1) {
          let word = 0;
          for (let byte = 0; byte < 4; byte += 1) {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            const code = (state % 255) - 127;
            word |= (code & 0xff) << (8 * byte);
          }
          words[w] = word >>> 0;
        }
        return words;
      };
      const scales = (count: number, salt: number, base: number): Float32Array<ArrayBuffer> => {
        const out = new Float32Array(count);
        for (let i = 0; i < count; i += 1) out[i] = base * (1 + ((i * 7 + salt) % 13) / 13);
        return out;
      };
      const gpu = await acquireGpu({ subgroups: "available" });
      try {
        const facts = attentionDeviceFacts(gpu);
        let checked = 0;
        for (const { b, h, m, n, d, windows } of cases) {
          const batch = b * h;
          const reduces = (["sg32", "wg"] as const).filter(
            (reduce: AttentionOnlineReduce) =>
              (reduce === "wg" || facts.subgroups) &&
              attentionOnlineI8a8WorkgroupStorageBytes(d, reduce) <=
                facts.maxComputeWorkgroupStorageSize,
          );
          for (const reduce of reduces) {
            for (const dp4a of [true, false]) {
              const where = `B ${b} H ${h} M ${m} N ${n} D ${d} ${reduce} ${dp4a ? "dp4a" : "emu"}${
                windows === undefined ? "" : " windows"
              }`;
              const test = bench(gpu);
              try {
                const inputs = [
                  test.host(codes(batch * m * d, 1)),
                  test.host(codes(batch * n * d, 2)),
                  test.host(codes(batch * d * n, 3)),
                ];
                const tail = [
                  test.host(scales(batch * m, 4, 0.02)),
                  test.host(scales(batch * n, 5, 0.018)),
                  test.host(scales(batch * d, 6, 0.01)),
                ];
                const scale = Math.fround(Math.sqrt(1 / Math.sqrt(d)));
                const bytes = batch * m * d * 4;
                const online = async (
                  output: AttentionOnlineOutput,
                  out: GPUBuffer,
                ): Promise<void> => {
                  for (const [offset, rows] of windows ?? [[0, m] as const]) {
                    await test.run(
                      attentionOnlineI8a8Key(dp4a, d, reduce, output),
                      attentionOnlineI8a8Wgsl(dp4a, d, reduce, output),
                      attentionOnlineI8a8Params(
                        m,
                        n,
                        offset,
                        scale,
                        output === "bshd" ? h : undefined,
                      ),
                      UNIFORM_IN,
                      [inputs[0], inputs[1], inputs[2], out, tail[0], tail[1], tail[2]],
                      [Math.ceil(rows / ATTENTION_ONLINE_ROWS), batch, 1],
                    );
                  }
                };
                const headMajor = test.region(bytes);
                await online("bhsd", headMajor);
                const expected = test.region(bytes);
                await test.permute(headMajor, expected, [b, h, m, d], HEAD_SWAP);
                const tokenMajor = test.region(bytes);
                await online("bshd", tokenMajor);
                const words = batch * m * d;
                const want = await test.read(expected, words);
                assert(new Set(want).size > 1, `${where}: the output is constant`);
                assertWords(await test.read(tokenMajor, words), want, where);
                checked += 1;
              } finally {
                await test.dispose();
              }
            }
          }
        }
        assert(checked > 0, "no reduction form fits this device");
        console.log(`[attention token-major] online bshd: ${checked} combinations bit-identical`);
      } finally {
        gpu.destroy();
      }
    });
  },
);

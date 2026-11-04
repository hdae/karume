/**
 * **Bit-identity checks** of packed int8 activations (ADR 0105) (real GPU).
 *
 * 1. Restore check: over all 256 codes × representative scales (including real scales of the QAT
 *    distribution), the consumer-side `vec4<f32>(unpack4xI8(w)) * scale` **matches the current SRQ f32 output
 *    as u32 per element**. The only exception is `-0.0`, which has no slot in int8 (it falls to code 0 =
 *    `+0.0`); its count is pinned as well.
 * 2. Path check: the packed variant of the parallel GEMV (i2 / i4 / i8 × lane 2–32 × with / without the fused
 *    SRQ epilogue) matches the current path's output **exactly as u32**. This is the detector for the claim that
 *    the `-0.0` fall does not move the output (the accumulator `acc` starts at `+0.0`, so adding `±0.0` does not
 *    change its value).
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvStaticQuantizePackedParams,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeParams,
  linearGemvStaticQuantizeWgsl,
  linearGemvUnit,
} from "../src/kernels/linear-gemv.ts";
import {
  STATIC_QUANTIZE_PACKED_WGSL,
  STATIC_QUANTIZE_WGSL,
  staticQuantizePackedParams,
  staticQuantizeParams,
} from "../src/kernels/static-quantize.ts";

/**
 * Representative scales. The last 4 are **values the QAT E2B distribution actually declares** (minimum / two
 * middle ones / maximum). The first 2 are a power of 2 (exact product) and a non-power-of-2 control.
 */
const SCALES = [
  0.00390625,
  Math.fround(0.00071),
  0.0015532826073467731,
  0.005826006643474102,
  3.334678888320923,
  27.842519760131836,
] as const;

/** A test shader that extracts just the consumer-side restore of one word (the text is the same as the kernel's). */
const RESTORE_WGSL = `
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.x) { return; }
  dst[gid.x] = vec4<f32>(unpack4xI8(src[gid.x])) * bitcast<f32>(params.y);
}
`;

const f32Bits = (value: number): number => new Uint32Array(Float32Array.of(value).buffer)[0];

describe.skipIf(!GPU_AVAILABLE)("bit-identity checks of packed int8 activations (real GPU)", () => {
  it("over all 256 codes × representative scales the restore matches the current SRQ output as u32 (only -0.0 falls to +0.0)", async () => {
    const gpu = await acquireGpu(),
      d = gpu.device;
    try {
      const cache = new Map<string, GPUComputePipeline>();
      const compile = (code: string): GPUComputePipeline => {
        const found = cache.get(code);
        if (found !== undefined) return found;
        const made = d.createComputePipeline({
          layout: "auto",
          compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
        });
        cache.set(code, made);
        return made;
      };
      let checks = 0;
      for (const scale of SCALES) {
        // The middle of codes -128..127 and its neighbours, plus ±0 / a tiny negative (= the side that produces -0.0).
        const values: number[] = [];
        for (let code = -128; code <= 127; code += 1) values.push(code * scale);
        values.push(0, -0, -scale * 1e-6, scale * 1e-6, -Infinity, Infinity);
        while (values.length % 4 !== 0) values.push(0);
        const count = values.length;
        const x = Float32Array.from(values);
        const owned: GPUBuffer[] = [];
        d.pushErrorScope("validation");
        let validationError: GPUError | null = null;
        try {
          const make = (
            v: number | ArrayBufferView<ArrayBuffer>,
            usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
          ): GPUBuffer => {
            const b = d.createBuffer({
              size: typeof v === "number" ? v : v.byteLength,
              usage,
            });
            owned.push(b);
            if (typeof v !== "number") d.queue.writeBuffer(b, 0, v);
            return b;
          };
          const input = make(x);
          const reference = make(count * 4);
          const codes = make(count);
          const restored = make(count * 4);
          const stage = make(count * 8, U.MAP_READ | U.COPY_DST);
          const sp = make(staticQuantizeParams(count, scale), U.UNIFORM | U.COPY_DST);
          const pp = make(staticQuantizePackedParams(count, scale), U.UNIFORM | U.COPY_DST);
          const rp = make(Uint32Array.of(count / 4, f32Bits(scale), 0, 0), U.UNIFORM | U.COPY_DST);
          const encoder = d.createCommandEncoder(),
            pass = encoder.beginComputePass();
          const dispatch = (code: string, buffers: GPUBuffer[], gx: number): void => {
            const pipeline = compile(code);
            pass.setPipeline(pipeline);
            pass.setBindGroup(
              0,
              d.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: buffers.map((buffer, binding) => ({
                  binding,
                  resource: { buffer },
                })),
              }),
            );
            pass.dispatchWorkgroups(gx);
          };
          dispatch(STATIC_QUANTIZE_WGSL, [sp, input, reference], Math.ceil(count / 128));
          // packed also uses "1 thread per element" like the f32 path (ADR 0105 addendum 1) = the same dispatch count
          dispatch(STATIC_QUANTIZE_PACKED_WGSL, [pp, input, codes], Math.ceil(count / 128));
          dispatch(RESTORE_WGSL, [rp, codes, restored], Math.ceil(count / 4 / 64));
          pass.end();
          encoder.copyBufferToBuffer(reference, 0, stage, 0, count * 4);
          encoder.copyBufferToBuffer(restored, 0, stage, count * 4, count * 4);
          d.queue.submit([encoder.finish()]);
          await stage.mapAsync(MAP_MODE.READ);
          const words = new Uint32Array(stage.getMappedRange().slice(0));
          stage.unmap();
          let negativeZero = 0,
            different = 0;
          for (let i = 0; i < count; i += 1) {
            const a = words[i],
              b = words[i + count];
            if (a === b) continue;
            if (a === 0x80000000 && b === 0) negativeZero += 1;
            else different += 1;
          }
          assertEquals(different, 0, `scale=${scale}: elements whose restore does not match`);
          // -0.0 comes only from "inputs that fall to level 0 with a sign": -0 and the tiny negative, 2 in all.
          assertEquals(negativeZero, 2, `scale=${scale}: number of values falling to -0.0`);
          checks += 1;
        } finally {
          d.queue.submit([]);
          await d.queue.onSubmittedWorkDone();
          validationError = await d.popErrorScope();
          for (const b of owned) b.destroy();
        }
        assertEquals(validationError, null);
      }
      assertEquals(checks, SCALES.length);
    } finally {
      gpu.destroy();
    }
  });

  /**
   * Edge lengths of the grid stride. Including `132` and `12288` (= 132 × 93 + …), which are not multiples of
   * `128`, checks that the packing through shared memory (ADR 0105 addendum 1) does not break on the edge tile.
   * The element counts are multiples of 4 because of the params check (4 elements per word).
   */
  const TAIL_COUNTS = [4, 16, 128, 132, 1536, 12288] as const;
  /** A sentinel that fails if even one out-of-range word is written. */
  const SENTINEL = 0xdeadbeef;

  it("the new kernel emits the same code sequence as the f32 path at edge lengths too (code cycle × 6 counts)", async () => {
    const gpu = await acquireGpu(),
      d = gpu.device;
    try {
      const cache = new Map<string, GPUComputePipeline>();
      const compile = (code: string): GPUComputePipeline => {
        const found = cache.get(code);
        if (found !== undefined) return found;
        const made = d.createComputePipeline({
          layout: "auto",
          compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
        });
        cache.set(code, made);
        return made;
      };
      let checks = 0;
      for (const count of TAIL_COUNTS) {
        for (const scale of [SCALES[0], SCALES[2]]) {
          // Cycle through all 256 codes, crossing a word boundary every 4 elements.
          const x = Float32Array.from(
            { length: count },
            (_, i) => ((i % 256) - 128 + (i % 3) * 0.37) * scale,
          );
          const words = count / 4;
          const owned: GPUBuffer[] = [];
          d.pushErrorScope("validation");
          let validationError: GPUError | null = null;
          try {
            const make = (
              v: number | ArrayBufferView<ArrayBuffer>,
              usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
            ): GPUBuffer => {
              const b = d.createBuffer({
                size: typeof v === "number" ? v : v.byteLength,
                usage,
              });
              owned.push(b);
              if (typeof v !== "number") d.queue.writeBuffer(b, 0, v);
              return b;
            };
            const input = make(x);
            const reference = make(count * 4);
            // Fill the last 8 words with the sentinel to detect over-writes on the edge tile.
            const codes = make(new Uint32Array(words + 8).fill(SENTINEL));
            const restored = make(count * 4);
            const stage = make((count * 2 + words + 8) * 4, U.MAP_READ | U.COPY_DST);
            const sp = make(staticQuantizeParams(count, scale), U.UNIFORM | U.COPY_DST);
            const pp = make(staticQuantizePackedParams(count, scale), U.UNIFORM | U.COPY_DST);
            const rp = make(Uint32Array.of(words, f32Bits(scale), 0, 0), U.UNIFORM | U.COPY_DST);
            const encoder = d.createCommandEncoder(),
              pass = encoder.beginComputePass();
            const dispatch = (code: string, buffers: GPUBuffer[], gx: number): void => {
              const pipeline = compile(code);
              pass.setPipeline(pipeline);
              pass.setBindGroup(
                0,
                d.createBindGroup({
                  layout: pipeline.getBindGroupLayout(0),
                  entries: buffers.map((buffer, binding) => ({
                    binding,
                    resource: { buffer },
                  })),
                }),
              );
              pass.dispatchWorkgroups(gx);
            };
            const groups = Math.ceil(count / 128);
            dispatch(STATIC_QUANTIZE_WGSL, [sp, input, reference], groups);
            dispatch(STATIC_QUANTIZE_PACKED_WGSL, [pp, input, codes], groups);
            dispatch(RESTORE_WGSL, [rp, codes, restored], Math.ceil(words / 64));
            pass.end();
            encoder.copyBufferToBuffer(reference, 0, stage, 0, count * 4);
            encoder.copyBufferToBuffer(restored, 0, stage, count * 4, count * 4);
            encoder.copyBufferToBuffer(codes, 0, stage, count * 8, (words + 8) * 4);
            d.queue.submit([encoder.finish()]);
            await stage.mapAsync(MAP_MODE.READ);
            const read = new Uint32Array(stage.getMappedRange().slice(0));
            stage.unmap();
            // The contract is "matches the f32 path's words, except that only `-0.0` falls to `+0.0`" (ADR 0105).
            let different = 0,
              negativeZero = 0;
            for (let i = 0; i < count; i += 1) {
              const reference = read[i],
                restoredWord = read[i + count];
              if (reference === 0x80000000) {
                negativeZero += 1;
                if (restoredWord !== 0) different += 1;
              } else if (reference !== restoredWord) different += 1;
            }
            assertEquals(different, 0, `count=${count} scale=${scale}: mismatching elements`);
            // `-0.0` comes only from "a negative value falling to level 0" = elements whose quotient is in (-0.5, 0).
            // With the input above, that is the single point `i%256 == 127 and i%3 == 2` (quotient -0.26) = `i ≡ 383 (mod 768)`.
            assertEquals(
              negativeZero,
              Math.floor((count + 384) / 768),
              `count=${count} scale=${scale}: number of values falling to -0.0`,
            );
            const tail = [...read.subarray(count * 2 + words, count * 2 + words + 8)];
            assertEquals(
              tail,
              Array.from({ length: 8 }, () => SENTINEL),
              `count=${count} scale=${scale}: wrote an out-of-range word`,
            );
            checks += 1;
          } finally {
            d.queue.submit([]);
            await d.queue.onSubmittedWorkDone();
            validationError = await d.popErrorScope();
            for (const b of owned) b.destroy();
          }
          assertEquals(validationError, null);
        }
      }
      assertEquals(checks, TAIL_COUNTS.length * 2);
    } finally {
      gpu.destroy();
    }
  });

  it("NaN saturates to +127 / -128 by its sign bit (quiet and signaling alike; one of the only 2 values that differ from the f32 path)", async () => {
    const gpu = await acquireGpu(),
      d = gpu.device;
    try {
      // Inputs are passed as bit patterns (the kernel reads them as u32, so NaN payloads arrive as well).
      // +qNaN / -qNaN / +sNaN / -sNaN / 1.0 / -1.0 / +0 / -0
      const bits = Uint32Array.of(
        0x7fc00000,
        0xffc00000,
        0x7f800001,
        0xff800001,
        0x3f800000,
        0xbf800000,
        0,
        0x80000000,
      );
      const count = bits.length,
        scale = 0.5;
      const owned: GPUBuffer[] = [];
      d.pushErrorScope("validation");
      let validationError: GPUError | null = null;
      let bytes: number[] = [];
      try {
        const make = (
          v: number | ArrayBufferView<ArrayBuffer>,
          usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
        ): GPUBuffer => {
          const b = d.createBuffer({ size: typeof v === "number" ? v : v.byteLength, usage });
          owned.push(b);
          if (typeof v !== "number") d.queue.writeBuffer(b, 0, v);
          return b;
        };
        const input = make(bits);
        const codes = make(count);
        const stage = make(count, U.MAP_READ | U.COPY_DST);
        const pp = make(staticQuantizePackedParams(count, scale), U.UNIFORM | U.COPY_DST);
        const pipeline = d.createComputePipeline({
          layout: "auto",
          compute: {
            module: d.createShaderModule({ code: STATIC_QUANTIZE_PACKED_WGSL }),
            entryPoint: "main",
          },
        });
        const encoder = d.createCommandEncoder(),
          pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(
          0,
          d.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [pp, input, codes].map((buffer, binding) => ({
              binding,
              resource: { buffer },
            })),
          }),
        );
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(codes, 0, stage, 0, count);
        d.queue.submit([encoder.finish()]);
        await stage.mapAsync(MAP_MODE.READ);
        bytes = [...new Uint8Array(stage.getMappedRange().slice(0))];
        stage.unmap();
      } finally {
        d.queue.submit([]);
        await d.queue.onSubmittedWorkDone();
        validationError = await d.popErrorScope();
        for (const b of owned) b.destroy();
      }
      assertEquals(validationError, null);
      // The magnitude of NaN exceeds every boundary, so the level is the limit given by the sign (+127 / 128 = -128).
      // 1.0 / 0.5 = 2 → 0x02, -2 → 0xFE, and -0 has no slot in int8, so 0x00.
      assertEquals(bytes, [0x7f, 0x80, 0x7f, 0x80, 0x02, 0xfe, 0x00, 0x00]);
    } finally {
      gpu.destroy();
    }
  });

  it("the packed variant of the parallel GEMV matches the current path exactly as u32 (i2/i4/i8 × lane × with/without fusion)", async () => {
    const gpu = await acquireGpu(),
      d = gpu.device;
    try {
      const cache = new Map<string, GPUComputePipeline>();
      const compile = (code: string): GPUComputePipeline => {
        const found = cache.get(code);
        if (found !== undefined) return found;
        const made = d.createComputePipeline({
          layout: "auto",
          compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
        });
        cache.set(code, made);
        return made;
      };
      const xScale = Math.fround(0.005826006643474102);
      const outScale = Math.fround(0.00071);
      let checks = 0;
      for (const storage of ["i2", "i4", "i8"] as const) {
        for (const k of [linearGemvUnit(storage) * 5, 1536]) {
          for (const lanes of [2, 4, 8, 16, 32] as const) {
            for (const m of [1, 4, 8]) {
              for (const scenario of ["normal", "codes", "special"] as const) {
                for (const quantize of [false, true]) {
                  const n = 36,
                    group = storage === "i4" ? (k === 1536 ? 512 : 32) : undefined,
                    bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
                  const owned: GPUBuffer[] = [];
                  d.pushErrorScope("validation");
                  let validationError: GPUError | null = null;
                  const make = (
                    v: number | ArrayBufferView<ArrayBuffer>,
                    usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
                  ): GPUBuffer => {
                    const b = d.createBuffer({
                      size: typeof v === "number" ? v : v.byteLength,
                      usage,
                    });
                    owned.push(b);
                    if (typeof v !== "number") d.queue.writeBuffer(b, 0, v);
                    return b;
                  };
                  try {
                    // Three activation shapes: one spanning the whole int8 range, one with special values, and a plain one.
                    const x = Float32Array.from({ length: m * k }, (_, i) =>
                      scenario === "codes"
                        ? ((i % 256) - 128 + (i % 3) * 0.4) * xScale
                        : scenario === "special"
                          ? [Infinity, -Infinity, -0, -xScale * 1e-6][i % 4]
                          : Math.sin(i * 0.037) * 0.75,
                    );
                    const w = Uint32Array.from(
                      { length: (n * k * bits) / 32 },
                      (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
                    );
                    const bias = Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);
                    const ws = Float32Array.from(
                      {
                        length: n * (group === undefined ? 1 : k / group),
                      },
                      (_, i) => ((i % 17) + 1) * 0.00017,
                    );
                    const input = make(x),
                      weight = make(w),
                      biasBuf = make(bias),
                      scaleBuf = make(ws);
                    const xf32 = make(m * k * 4),
                      xPacked = make(m * k);
                    const a = make(m * n * 4),
                      b = make(m * n * 4);
                    const stage = make(m * n * 8, U.MAP_READ | U.COPY_DST);
                    const sp = make(staticQuantizeParams(m * k, xScale), U.UNIFORM | U.COPY_DST),
                      pp = make(staticQuantizePackedParams(m * k, xScale), U.UNIFORM | U.COPY_DST);
                    const plainParams = quantize
                      ? linearGemvStaticQuantizeParams(storage, m, n, k, outScale, group)
                      : linearGemvParams(storage, m, n, k, group);
                    const packedParams = quantize
                      ? linearGemvStaticQuantizePackedParams(
                          storage,
                          m,
                          n,
                          k,
                          xScale,
                          outScale,
                          group,
                        )
                      : linearGemvParallelPackedParams(storage, m, n, k, xScale, group);
                    const lp = make(plainParams, U.UNIFORM | U.COPY_DST),
                      kp = make(packedParams, U.UNIFORM | U.COPY_DST);
                    const encoder = d.createCommandEncoder(),
                      pass = encoder.beginComputePass();
                    const dispatch = (
                      code: string,
                      buffers: GPUBuffer[],
                      gx: number,
                      gy = 1,
                    ): void => {
                      const pipeline = compile(code);
                      pass.setPipeline(pipeline);
                      pass.setBindGroup(
                        0,
                        d.createBindGroup({
                          layout: pipeline.getBindGroupLayout(0),
                          entries: buffers.map((buffer, binding) => ({
                            binding,
                            resource: { buffer },
                          })),
                        }),
                      );
                      pass.dispatchWorkgroups(gx, gy);
                    };
                    dispatch(STATIC_QUANTIZE_WGSL, [sp, input, xf32], Math.ceil((m * k) / 128));
                    dispatch(
                      STATIC_QUANTIZE_PACKED_WGSL,
                      [pp, input, xPacked],
                      Math.ceil((m * k) / 128),
                    );
                    const tiles = Math.ceil(n / (128 / lanes));
                    dispatch(
                      quantize
                        ? linearGemvStaticQuantizeWgsl(storage, group, lanes)
                        : linearGemvParallelWgsl(storage, group, lanes),
                      [lp, xf32, weight, biasBuf, a, scaleBuf],
                      tiles,
                      m,
                    );
                    dispatch(
                      quantize
                        ? linearGemvStaticQuantizePackedWgsl(storage, group, lanes)
                        : linearGemvParallelPackedWgsl(storage, group, lanes),
                      [kp, xPacked, weight, biasBuf, b, scaleBuf],
                      tiles,
                      m,
                    );
                    pass.end();
                    encoder.copyBufferToBuffer(a, 0, stage, 0, m * n * 4);
                    encoder.copyBufferToBuffer(b, 0, stage, m * n * 4, m * n * 4);
                    d.queue.submit([encoder.finish()]);
                    await stage.mapAsync(MAP_MODE.READ);
                    const words = new Uint32Array(stage.getMappedRange().slice(0));
                    stage.unmap();
                    let different = 0;
                    for (let i = 0; i < m * n; i += 1) {
                      if (words[i] !== words[i + m * n]) different += 1;
                    }
                    assertEquals(
                      different,
                      0,
                      JSON.stringify({ storage, k, group, lanes, m, scenario, quantize }),
                    );
                    checks += 1;
                  } finally {
                    d.queue.submit([]);
                    await d.queue.onSubmittedWorkDone();
                    validationError = await d.popErrorScope();
                    for (const buffer of owned) buffer.destroy();
                  }
                  assertEquals(validationError, null);
                }
              }
            }
          }
        }
      }
      assertEquals(checks, 540);
    } finally {
      gpu.destroy();
    }
  });
});

/**
 * Fused linear-GEMV + SRQ epilogue (non-packed) against the unfused pair
 * (parallel GEMV, then the standalone SRQ kernel): u32 parity over every
 * storage x lane x M, including the SRQ boundary, special values, identity
 * (scale 0) and the sign-mask rounding scenario.
 */
describe.skipIf(!GPU_AVAILABLE)("fused linear→SRQ kernel (real GPU)", () => {
  it("matches the original parallel + original SRQ at every boundary, on special values and at M1/4/8", async () => {
    const gpu = await acquireGpu(),
      d = gpu.device;
    let checks = 0;
    try {
      const cache = new Map<string, GPUComputePipeline>();
      const compile = (code: string): GPUComputePipeline => {
        let p = cache.get(code);
        if (!p) {
          p = d.createComputePipeline({
            layout: "auto",
            compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
          });
          cache.set(code, p);
        }
        return p;
      };
      const srq = compile(STATIC_QUANTIZE_WGSL);
      for (const storage of ["i2", "i4", "i8"] as const) {
        for (const k of [linearGemvUnit(storage) * 5, 1536]) {
          for (const lanes of [2, 4, 8, 16, 32] as const) {
            for (const m of [1, 4, 8]) {
              for (const scenario of [
                "normal",
                "wide",
                "boundary",
                "special",
                "identity",
                "rounding",
              ] as const) {
                const n = scenario === "boundary" ? 768 : 36,
                  group = storage === "i4" ? (k === 1536 ? 512 : 32) : undefined,
                  scale =
                    scenario === "identity" || scenario === "rounding" ? 0 : Math.fround(0.00071),
                  bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
                d.pushErrorScope("validation");
                const owned: GPUBuffer[] = [];
                let validationError: GPUError | null = null;
                const make = (
                  v: number | ArrayBufferView<ArrayBuffer>,
                  usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
                ): GPUBuffer => {
                  const b = d.createBuffer({
                    size: typeof v === "number" ? v : v.byteLength,
                    usage,
                  });
                  owned.push(b);
                  if (typeof v !== "number") d.queue.writeBuffer(b, 0, v);
                  return b;
                };
                try {
                  const x = Float32Array.from({ length: m * k }, (_, i) =>
                    scenario === "special"
                      ? [Infinity, -Infinity, NaN, -0][i % 4]
                      : scenario === "boundary"
                        ? 0
                        : Math.sin(i * 0.037) * (scenario === "wide" ? 1e4 : 0.75),
                  );
                  const w = Uint32Array.from(
                    { length: (n * k * bits) / 32 },
                    (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
                  );
                  const boundary = staticQuantizeParams(m * n, scale);
                  const bias =
                    scenario === "boundary"
                      ? new Float32Array(
                          Uint32Array.from(
                            { length: n },
                            (_, i) =>
                              ((boundary[1 + Math.floor(i / 6)] + Math.floor((i % 6) / 2) - 1) |
                                (i % 2 ? 0x80000000 : 0)) >>>
                              0,
                          ).buffer,
                        )
                      : Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);
                  const ws = Float32Array.from(
                    {
                      length: n * (group === undefined ? 1 : k / group),
                    },
                    (_, i) => ((i % 17) + 1) * 0.00017,
                  );
                  const input = make(x),
                    weight = make(w),
                    biasBuf = make(bias),
                    scaleBuf = make(ws);
                  const middle = make(m * n * 4),
                    a = make(m * n * 4),
                    b = make(m * n * 4),
                    stage = make(m * n * 8, U.MAP_READ | U.COPY_DST);
                  const lp = make(
                      linearGemvParams(storage, m, n, k, group),
                      U.UNIFORM | U.COPY_DST,
                    ),
                    sp = make(staticQuantizeParams(m * n, scale), U.UNIFORM | U.COPY_DST),
                    fp = make(
                      linearGemvStaticQuantizeParams(storage, m, n, k, scale, group),
                      U.UNIFORM | U.COPY_DST,
                    );
                  const mask = scenario === "rounding" ? 0x80000000 : 0;
                  d.queue.writeBuffer(fp, 12, Uint32Array.of(mask));
                  const encoder = d.createCommandEncoder(),
                    pass = encoder.beginComputePass();
                  const dispatch = (
                    pipeline: GPUComputePipeline,
                    buffers: GPUBuffer[],
                    gx: number,
                    gy = 1,
                  ): void => {
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(
                      0,
                      d.createBindGroup({
                        layout: pipeline.getBindGroupLayout(0),
                        entries: buffers.map((buffer, binding) => ({
                          binding,
                          resource: { buffer },
                        })),
                      }),
                    );
                    pass.dispatchWorkgroups(gx, gy);
                  };
                  dispatch(
                    compile(linearGemvParallelWgsl(storage, group, lanes)),
                    [lp, input, weight, biasBuf, middle, scaleBuf],
                    Math.ceil(n / (128 / lanes)),
                    m,
                  );
                  dispatch(srq, [sp, middle, a], Math.ceil((m * n) / 128));
                  dispatch(
                    compile(linearGemvStaticQuantizeWgsl(storage, group, lanes)),
                    [fp, input, weight, biasBuf, b, scaleBuf],
                    Math.ceil(n / (128 / lanes)),
                    m,
                  );
                  pass.end();
                  encoder.copyBufferToBuffer(a, 0, stage, 0, m * n * 4);
                  encoder.copyBufferToBuffer(b, 0, stage, m * n * 4, m * n * 4);
                  d.queue.submit([encoder.finish()]);
                  await stage.mapAsync(MAP_MODE.READ);
                  const words = new Uint32Array(stage.getMappedRange().slice(0));
                  stage.unmap();
                  let different = 0;
                  for (let i = 0; i < m * n; i++) {
                    if (
                      (words[i] ^ mask) >>> 0 !== words[i + m * n] &&
                      !(
                        (words[i] & 0x7fffffff) > 0x7f800000 &&
                        (words[i + m * n] & 0x7fffffff) > 0x7f800000
                      )
                    )
                      different++;
                  }
                  assertEquals(
                    different,
                    0,
                    JSON.stringify({ storage, k, group, lanes, m, scenario }),
                  );
                  checks++;
                } finally {
                  d.queue.submit([]);
                  await d.queue.onSubmittedWorkDone();
                  validationError = await d.popErrorScope();
                  for (const b of owned) b.destroy();
                }
                assertEquals(validationError, null);
              }
            }
          }
        }
      }
      assertEquals(checks, 540);
    } finally {
      gpu.destroy();
    }
  });
});

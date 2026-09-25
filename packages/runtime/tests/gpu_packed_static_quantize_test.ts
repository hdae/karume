/**
 * packed int8 活性（ADR 0105）の**ビット同一門**（実 GPU）。
 *
 * 1. 復元の門: 全 256 コード × 代表 scale（QAT 配布形の実 scale を含む）で、消費側の
 *    `vec4<f32>(unpack4xI8(w)) * scale` が現行 SRQ の f32 出力と**要素ごとに u32 一致**する。
 *    唯一の例外は int8 に席の無い `-0.0`（コード 0 = `+0.0` へ落ちる）で、本数まで固定する。
 * 2. 経路の門: 並列 GEMV（i2 / i4 / i8 × lane 2〜32 × SRQ 融合エピローグあり / なし）の
 *    packed 変種と現行経路の出力が **u32 完全一致**する。`-0.0` の落ちが出力を動かさないこと
 *    （積和の `acc` は `+0.0` 始まりなので `±0.0` の加算で値が動かない）は、ここが検出器。
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
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
 * 代表 scale。末尾 4 本は **QAT E2B 配布形が実際に宣言している値**（最小 / 中位 2 本 / 最大）。
 * 先頭 2 本は 2 冪（積が厳密）と非 2 冪の対照。
 */
const SCALES = [
  0.00390625,
  Math.fround(0.00071),
  0.0015532826073467731,
  0.005826006643474102,
  3.334678888320923,
  27.842519760131836,
] as const;

/** 消費側の復元 1 語ぶんだけを取り出した検査用シェーダ（字面はカーネルと同じ）。 */
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

describe({
  name: "packed int8 活性のビット同一門（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("全 256 コード × 代表 scale で復元が現行 SRQ 出力と u32 一致する（-0.0 だけ +0.0 へ落ちる）", async () => {
      const gpu = await acquireGpu(), d = gpu.device;
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
          // コード -128..127 の中央と両隣、加えて ±0 / 極小負（= -0.0 を作る側）。
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
            const rp = make(
              Uint32Array.of(count / 4, f32Bits(scale), 0, 0),
              U.UNIFORM | U.COPY_DST,
            );
            const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
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
            // packed も f32 経路と同じ「1 スレッド 1 要素」（ADR 0105 追記 1）= dispatch 数も同じ
            dispatch(STATIC_QUANTIZE_PACKED_WGSL, [pp, input, codes], Math.ceil(count / 128));
            dispatch(RESTORE_WGSL, [rp, codes, restored], Math.ceil(count / 4 / 64));
            pass.end();
            encoder.copyBufferToBuffer(reference, 0, stage, 0, count * 4);
            encoder.copyBufferToBuffer(restored, 0, stage, count * 4, count * 4);
            d.queue.submit([encoder.finish()]);
            await stage.mapAsync(MAP_MODE.READ);
            const words = new Uint32Array(stage.getMappedRange().slice(0));
            stage.unmap();
            let negativeZero = 0, different = 0;
            for (let i = 0; i < count; i += 1) {
              const a = words[i], b = words[i + count];
              if (a === b) continue;
              if (a === 0x80000000 && b === 0) negativeZero += 1;
              else different += 1;
            }
            assertEquals(different, 0, `scale=${scale}: 復元が一致しない要素`);
            // -0.0 になるのは「符号つきで level 0 へ落ちる入力」だけ: -0 と極小負の 2 本。
            assertEquals(negativeZero, 2, `scale=${scale}: -0.0 の落ちる本数`);
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
     * grid-stride の端の長さ。`128` の倍数でない `132` と `12288`（= 132 × 93 + …）を含めて、
     * 共有メモリ経由の詰め（ADR 0105 追記 1）が端のタイルで壊れないことを見る。
     * 要素数が 4 の倍数なのは params の門（1 語 4 要素）。
     */
    const TAIL_COUNTS = [4, 16, 128, 132, 1536, 12288] as const;
    /** 範囲外の語を 1 語でも書いたら落ちる番兵。 */
    const SENTINEL = 0xdeadbeef;

    it("新カーネルは端の長さでも f32 経路と同じコード列を出す（コード巡回 × count 6 種）", async () => {
      const gpu = await acquireGpu(), d = gpu.device;
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
            // 全 256 コードを巡回させ、4 要素ごとに 1 語の境界を跨がせる。
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
              // 末尾 8 語を番兵で埋めておき、端のタイルの書きすぎを検出する。
              const codes = make(new Uint32Array(words + 8).fill(SENTINEL));
              const restored = make(count * 4);
              const stage = make((count * 2 + words + 8) * 4, U.MAP_READ | U.COPY_DST);
              const sp = make(staticQuantizeParams(count, scale), U.UNIFORM | U.COPY_DST);
              const pp = make(staticQuantizePackedParams(count, scale), U.UNIFORM | U.COPY_DST);
              const rp = make(Uint32Array.of(words, f32Bits(scale), 0, 0), U.UNIFORM | U.COPY_DST);
              const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
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
              // 契約は「f32 経路の語と一致・ただし `-0.0` だけ `+0.0` へ落ちる」（ADR 0105）。
              let different = 0, negativeZero = 0;
              for (let i = 0; i < count; i += 1) {
                const reference = read[i], restoredWord = read[i + count];
                if (reference === 0x80000000) {
                  negativeZero += 1;
                  if (restoredWord !== 0) different += 1;
                } else if (reference !== restoredWord) different += 1;
              }
              assertEquals(different, 0, `count=${count} scale=${scale}: 一致しない要素`);
              // `-0.0` になるのは「負で level 0 へ落ちる」= 商が (-0.5, 0) の要素だけ。上の入力で
              // それを満たすのは `i%256 == 127 かつ i%3 == 2`（商 -0.26）の 1 点 = `i ≡ 383 (mod 768)`。
              assertEquals(
                negativeZero,
                Math.floor((count + 384) / 768),
                `count=${count} scale=${scale}: -0.0 の落ちる本数`,
              );
              const tail = [...read.subarray(count * 2 + words, count * 2 + words + 8)];
              assertEquals(
                tail,
                Array.from({ length: 8 }, () => SENTINEL),
                `count=${count} scale=${scale}: 範囲外の語を書いた`,
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

    it("NaN は符号ビットに従って +127 / -128 へ飽和する（quiet / signaling とも・f32 経路と違う唯一の 2 値の片方）", async () => {
      const gpu = await acquireGpu(), d = gpu.device;
      try {
        // 入力はビット列で渡す（カーネルは u32 として読むので NaN のペイロードも届く）。
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
        const count = bits.length, scale = 0.5;
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
          const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
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
        // NaN の magnitude は全境界を超えるので level は符号で決まる上限（+127 / 128 = -128）。
        // 1.0 / 0.5 = 2 → 0x02、-2 → 0xFE、-0 は int8 に席が無く 0x00。
        assertEquals(bytes, [0x7f, 0x80, 0x7f, 0x80, 0x02, 0xfe, 0x00, 0x00]);
      } finally {
        gpu.destroy();
      }
    });

    it("並列 GEMV の packed 変種は現行経路と u32 完全一致する（i2/i4/i8 × lane × 融合あり/なし）", async () => {
      const gpu = await acquireGpu(), d = gpu.device;
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
                      // 活性は int8 の全域を跨ぐ形・特殊値を含む形・素直な形の 3 通り。
                      const x = Float32Array.from(
                        { length: m * k },
                        (_, i) =>
                          scenario === "codes"
                            ? ((i % 256) - 128 + (i % 3) * 0.4) * xScale
                            : scenario === "special"
                            ? [Infinity, -Infinity, -0, -xScale * 1e-6][i % 4]
                            : Math.sin(i * 0.037) * 0.75,
                      );
                      const w = Uint32Array.from(
                        { length: n * k * bits / 32 },
                        (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
                      );
                      const bias = Float32Array.from(
                        { length: n },
                        (_, i) => (i % 7 - 3) * 0.11,
                      );
                      const ws = Float32Array.from({
                        length: n * (group === undefined ? 1 : k / group),
                      }, (_, i) => (i % 17 + 1) * 0.00017);
                      const input = make(x),
                        weight = make(w),
                        biasBuf = make(bias),
                        scaleBuf = make(ws);
                      const xf32 = make(m * k * 4), xPacked = make(m * k);
                      const a = make(m * n * 4), b = make(m * n * 4);
                      const stage = make(m * n * 8, U.MAP_READ | U.COPY_DST);
                      const sp = make(
                          staticQuantizeParams(m * k, xScale),
                          U.UNIFORM | U.COPY_DST,
                        ),
                        pp = make(
                          staticQuantizePackedParams(m * k, xScale),
                          U.UNIFORM | U.COPY_DST,
                        );
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
                      dispatch(STATIC_QUANTIZE_WGSL, [sp, input, xf32], Math.ceil(m * k / 128));
                      dispatch(
                        STATIC_QUANTIZE_PACKED_WGSL,
                        [pp, input, xPacked],
                        Math.ceil(m * k / 128),
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
  },
});

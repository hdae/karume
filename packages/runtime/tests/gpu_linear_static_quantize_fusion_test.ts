import { acquireGpu } from "../src/gpu/device.ts";
import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvUnit,
} from "../src/kernels/linear-gemv.ts";
import { STATIC_QUANTIZE_WGSL, staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import {
  linearGemvStaticQuantizeParams as linearSrqParams,
  linearGemvStaticQuantizeWgsl as linearSrqWgsl,
} from "../src/kernels/linear-gemv.ts";
describe({
  name: "linear→SRQの融合カーネル（実GPU）",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("元のparallel＋元のSRQと全境界・特殊値・M1/4/8で一致する", async () => {
      const gpu = await acquireGpu(), d = gpu.device;
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
                for (
                  const scenario of [
                    "normal",
                    "wide",
                    "boundary",
                    "special",
                    "identity",
                    "rounding",
                  ] as const
                ) {
                  const n = scenario === "boundary" ? 768 : 36,
                    group = storage === "i4" ? (k === 1536 ? 512 : 32) : undefined,
                    scale = scenario === "identity" || scenario === "rounding"
                      ? 0
                      : Math.fround(.00071),
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
                    const x = Float32Array.from(
                      { length: m * k },
                      (_, i) =>
                        scenario === "special"
                          ? [Infinity, -Infinity, NaN, -0][i % 4]
                          : scenario === "boundary"
                          ? 0
                          : Math.sin(i * .037) * (scenario === "wide" ? 1e4 : .75),
                    );
                    const w = Uint32Array.from(
                      { length: n * k * bits / 32 },
                      (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 0,
                    );
                    const boundary = staticQuantizeParams(m * n, scale);
                    const bias = scenario === "boundary"
                      ? new Float32Array(
                        Uint32Array.from({ length: n }, (_, i) =>
                          ((boundary[1 + Math.floor(i / 6)] + Math.floor(i % 6 / 2) - 1) |
                            (i % 2 ? 0x80000000 : 0)) >>> 0).buffer,
                      )
                      : Float32Array.from({ length: n }, (_, i) => (i % 7 - 3) * .11);
                    const ws = Float32Array.from({
                      length: n * (group === undefined ? 1 : k / group),
                    }, (_, i) => (i % 17 + 1) * .00017);
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
                        linearSrqParams(storage, m, n, k, scale, group),
                        U.UNIFORM | U.COPY_DST,
                      );
                    const mask = scenario === "rounding" ? 0x80000000 : 0;
                    d.queue.writeBuffer(fp, 12, Uint32Array.of(mask));
                    const encoder = d.createCommandEncoder(), pass = encoder.beginComputePass();
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
                    dispatch(srq, [sp, middle, a], Math.ceil(m * n / 128));
                    dispatch(
                      compile(linearSrqWgsl(storage, group, lanes)),
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
                        ((words[i] ^ mask) >>> 0) !== words[i + m * n] &&
                        !((words[i] & 0x7fffffff) > 0x7f800000 &&
                          (words[i + m * n] & 0x7fffffff) > 0x7f800000)
                      ) different++;
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
  },
});

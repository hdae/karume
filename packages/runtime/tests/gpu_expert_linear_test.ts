import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { f16BitsToF32 } from "../src/format/f16.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  EXPERT_LINEAR_OOB_BITS,
  type ExpertLinearStorage,
  expertLinearDispatch,
  expertLinearParams,
  expertLinearUnit,
  expertLinearWgsl,
} from "../src/kernels/expert-linear.ts";
import { linearGemvParams, linearGemvWgsl } from "../src/kernels/linear-gemv.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const GROUP = 32;
const NAN_FILL = 0x7fc00000;

/** Deterministic expert weights and steps; f16 steps are u16 patterns of positive normals. */
const fixture = (storage: ExpertLinearStorage, experts: number, n: number, k: number) => {
  const words = storage === "f32" ? 0 : (experts * n * k) / 8;
  const steps = (experts * n * k) / GROUP;
  return {
    f32: Float32Array.from({ length: experts * n * k }, (_, i) => Math.sin(i * 0.0131) * 0.2),
    packed: Uint32Array.from({ length: words }, (_, i) => Math.imul(i + 7, 0x9e3779b9) >>> 0),
    stepsF32: Float32Array.from({ length: steps }, (_, i) => ((i % 23) + 1) * 0.00021),
    // 0x1400..0x23ff: f16 normals between about 1e-3 and 1.5e-2, the range of 26B's expert steps.
    stepsF16: Uint16Array.from(
      { length: steps + (steps % 2) },
      (_, i) => 0x1400 + ((i * 131) % 0x1000),
    ),
  };
};

/** The steps the kernel reads, widened to f32 (what `linear`'s GEMV gets as its scale). */
const widened = (data: ReturnType<typeof fixture>, storage: ExpertLinearStorage) =>
  storage === "i4f16"
    ? Float32Array.from(data.stepsF16.subarray(0, data.stepsF32.length), (bits) =>
        f16BitsToF32(bits),
      )
    : data.stepsF32;

/** CPU value in f64 of one output (the tolerance check — the bit check is against `linear`). */
const cpuValue = (
  storage: ExpertLinearStorage,
  data: ReturnType<typeof fixture>,
  steps: Float32Array<ArrayBuffer>,
  shape: { n: number; k: number },
  expert: number,
  col: number,
  x: Float32Array<ArrayBuffer>,
): number => {
  const { n, k } = shape;
  const base = (expert * n + col) * k;
  let acc = 0;
  for (let i = 0; i < k; i++) {
    let weight: number;
    if (storage === "f32") weight = data.f32[base + i];
    else {
      const flat = base + i;
      const byte = (data.packed[flat >> 3] >>> (((flat >> 1) & 3) * 8)) & 0xff;
      const nibble = (flat & 1) === 1 ? byte >> 4 : byte & 0xf;
      weight = (nibble - 8) * steps[(expert * n + col) * (k / GROUP) + (i >> 5)];
    }
    acc += weight * x[i];
  }
  return acc;
};

describe.skipIf(!GPU_AVAILABLE)("expert_linear kernel (real GPU)", () => {
  it("writes, for every pick, the u32 words of linear's GEMV on that expert's rows, and NaN for an id out of range", async () => {
    const gpu = await acquireGpu();
    const device = gpu.device;
    const owned: GPUBuffer[] = [];
    const make = (
      value: number | ArrayBufferView<ArrayBuffer>,
      usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    ): GPUBuffer => {
      const size = typeof value === "number" ? value : value.byteLength;
      const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
      owned.push(buffer);
      if (typeof value !== "number") device.queue.writeBuffer(buffer, 0, value);
      return buffer;
    };
    const readBack = async (source: GPUBuffer, bytes: number, encoder: GPUCommandEncoder) => {
      const mapped = make(bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      encoder.copyBufferToBuffer(source, 0, mapped, 0, bytes);
      device.queue.submit([encoder.finish()]);
      await mapped.mapAsync(GPUMapMode.READ);
      const out = new Uint32Array(mapped.getMappedRange().slice(0));
      mapped.unmap();
      return out;
    };
    const pipelineOf = async (code: string) =>
      device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
    try {
      device.pushErrorScope("validation");
      const experts = 6;
      const n = 36; // not a multiple of the 32 columns per workgroup (edge columns)
      const tokens = 3;
      const picks = 2;
      // Ids repeat within and across tokens; the last token has one id below and one past E.
      const ids = [4, 1, 1, 5, -1, experts];
      for (const storage of ["f32", "i4", "i4f16"] as const) {
        const unit = expertLinearUnit(storage);
        // k: one prefetch round plus a remainder word, and the two 26B input widths.
        for (const k of [unit * 5, 704, 2816]) {
          const group = storage === "f32" ? undefined : GROUP;
          const data = fixture(storage, experts, n, k);
          const steps = widened(data, storage);
          const gemv = await pipelineOf(linearGemvWgsl(storage === "f32" ? "f32" : "i4", group));
          const kernel = await pipelineOf(expertLinearWgsl(storage, group));
          for (const sharedX of [true, false]) {
            const xRows = sharedX ? tokens : tokens * picks;
            const x = Float32Array.from({ length: xRows * k }, (_, i) => Math.cos(i * 0.029) * 0.9);
            const rows = tokens * picks;
            const out = make(
              rows * n * 4,
              GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            device.queue.writeBuffer(out, 0, new Uint32Array(rows * n).fill(NAN_FILL));
            const weight = storage === "f32" ? make(data.f32) : make(data.packed);
            const scale =
              storage === "i4"
                ? make(data.stepsF32)
                : storage === "i4f16"
                  ? make(new Uint32Array(data.stepsF16.buffer))
                  : undefined;
            const buffers = [
              make(
                expertLinearParams(storage, { tokens, picks, n, k, experts, sharedX }, group),
                GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
              ),
              make(x),
              make(Int32Array.from(ids)),
              weight,
              out,
              ...(scale === undefined ? [] : [scale]),
            ];
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(kernel);
            pass.setBindGroup(
              0,
              device.createBindGroup({
                layout: kernel.getBindGroupLayout(0),
                entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
              }),
            );
            // A y cap of 2 makes the 6 rows grid-stride.
            pass.dispatchWorkgroups(...expertLinearDispatch(rows, n, 2));
            pass.end();
            const got = await readBack(out, rows * n * 4, encoder);
            for (let row = 0; row < rows; row++) {
              const expert = ids[row];
              const where = `${storage} k${k} ${sharedX ? "shared" : "per-pick"} row ${row} (expert ${expert})`;
              const words = got.subarray(row * n, (row + 1) * n);
              if (expert < 0 || expert >= experts) {
                for (const word of words) assertEquals(word, EXPERT_LINEAR_OOB_BITS, where);
                continue;
              }
              // linear's M = 1 GEMV over this expert's rows and this row's x (bias 0).
              const xRow = x.slice(
                (sharedX ? Math.floor(row / picks) : row) * k,
                (sharedX ? Math.floor(row / picks) + 1 : row + 1) * k,
              );
              const expertWeight =
                storage === "f32"
                  ? data.f32.slice(expert * n * k, (expert + 1) * n * k)
                  : data.packed.slice((expert * n * k) / 8, ((expert + 1) * n * k) / 8);
              const reference = make(
                n * 4,
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
              );
              const gemvBuffers = [
                make(
                  linearGemvParams(storage === "f32" ? "f32" : "i4", 1, n, k, group),
                  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                ),
                make(xRow),
                make(expertWeight),
                make(new Float32Array(n)),
                reference,
                ...(storage === "f32"
                  ? []
                  : [make(steps.slice((expert * n * k) / GROUP, ((expert + 1) * n * k) / GROUP))]),
              ];
              const gemvEncoder = device.createCommandEncoder();
              const gemvPass = gemvEncoder.beginComputePass();
              gemvPass.setPipeline(gemv);
              gemvPass.setBindGroup(
                0,
                device.createBindGroup({
                  layout: gemv.getBindGroupLayout(0),
                  entries: gemvBuffers.map((buffer, binding) => ({
                    binding,
                    resource: { buffer },
                  })),
                }),
              );
              gemvPass.dispatchWorkgroups(Math.ceil(n / 32));
              gemvPass.end();
              const expected = await readBack(reference, n * 4, gemvEncoder);
              for (let col = 0; col < n; col++) {
                // linear adds its bias, so a -0 sum comes back +0 there.
                const word = words[col] === 0x80000000 ? 0 : words[col];
                assertEquals(word, expected[col], `${where} col ${col}`);
                const value = new Float32Array(new Uint32Array([words[col]]).buffer)[0];
                const cpu = cpuValue(storage, data, steps, { n, k }, expert, col, xRow);
                assert(
                  Math.abs(value - cpu) <= 1e-5 * k * Math.max(1, Math.abs(cpu)),
                  `${where} col ${col}: ${value} vs ${cpu} in f64`,
                );
              }
            }
          }
        }
      }
      const error = await device.popErrorScope();
      assertEquals(error, null, error?.message);
    } finally {
      for (const buffer of owned) buffer.destroy();
      gpu.destroy();
    }
  });
});

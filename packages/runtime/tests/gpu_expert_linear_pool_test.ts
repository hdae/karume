// The pooled form of the expert_linear kernel on a real GPU (ADR 0146 decision 2): the weight holds
// K slots and a table i32[E] maps an expert to its slot. Against the resident kernel over all E
// experts, on the same ids:
//
// - a pick whose expert is resident writes the resident kernel's words bit for bit (f32 / i4 /
//   i4f16, shared and per-pick x, an edge column count, k with a remainder word);
// - a pick whose table entry is −1 writes exactly 0 (+0.0 — every output still written, ADR 0014);
// - a table entry of K or more, and an id outside [0, E), write the quiet-NaN bits.
//
// The slots hold the experts in an order other than their ids, so an addressing slip (expert id for
// slot) reads another expert and fails the word comparison.

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  EXPERT_LINEAR_OOB_BITS,
  EXPERT_LINEAR_SCALE_BINDING,
  EXPERT_LINEAR_TABLE_BINDING,
  type ExpertLinearStorage,
  expertLinearDispatch,
  expertLinearParams,
  expertLinearUnit,
  expertLinearWgsl,
} from "../src/kernels/expert-linear.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const GROUP = 32;
const NAN_FILL = 0x7fc00000;
const EXPERTS = 6;
const SLOTS = 3;
/** Expert → slot: 1 → 0, 5 → 1, 4 → 2, 3 → K (a broken entry); 0 and 2 are not resident. */
const TABLE = [-1, 0, -1, SLOTS, 2, 1];
/** Four tokens of two picks: resident, broken (3), not resident (0), and two ids out of range. */
const IDS = [4, 1, 3, 5, 0, 1, -1, EXPERTS];

/** What the pooled kernel writes for a row (resident = the resident kernel's words). */
const rowKind = (id: number): "resident" | "zero" | "nan" => {
  if (id < 0 || id >= EXPERTS) return "nan";
  const slot = TABLE[id];
  return slot < 0 ? "zero" : slot >= SLOTS ? "nan" : "resident";
};

describe.skipIf(!GPU_AVAILABLE)("the pooled expert_linear kernel (real GPU)", () => {
  it("writes the resident kernel's words for a resident pick, 0 for a missing one, and NaN for a broken slot or id", async () => {
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
    const pipelineOf = async (code: string) =>
      device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
    /** One dispatch over `rows` rows; returns the out words. */
    const dispatch = async (
      pipeline: GPUComputePipeline,
      entries: readonly (readonly [number, GPUBuffer])[],
      out: GPUBuffer,
      rows: number,
      n: number,
    ): Promise<Uint32Array<ArrayBuffer>> => {
      device.queue.writeBuffer(out, 0, new Uint32Array(rows * n).fill(NAN_FILL));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
        }),
      );
      // A y cap of 3 makes the 8 rows grid-stride.
      pass.dispatchWorkgroups(...expertLinearDispatch(rows, n, 3));
      pass.end();
      const mapped = make(rows * n * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      encoder.copyBufferToBuffer(out, 0, mapped, 0, rows * n * 4);
      device.queue.submit([encoder.finish()]);
      await mapped.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(mapped.getMappedRange().slice(0));
      mapped.unmap();
      return words;
    };
    try {
      device.pushErrorScope("validation");
      const n = 36; // not a multiple of the 32 columns per workgroup (edge columns)
      const tokens = 4;
      const picks = 2;
      const rows = tokens * picks;
      for (const storage of [
        "f32",
        "i4",
        "i4f16",
      ] as const satisfies readonly ExpertLinearStorage[]) {
        const unit = expertLinearUnit(storage);
        // One prefetch round plus a remainder word, and a 26B width.
        for (const k of [unit * 5, 704]) {
          const group = storage === "f32" ? undefined : GROUP;
          const per = n * k; // elements of one expert
          const weightF32 = Float32Array.from(
            { length: EXPERTS * per },
            (_, i) => Math.sin(i * 0.0131) * 0.2,
          );
          const words = Uint32Array.from(
            { length: (EXPERTS * per) / 8 },
            (_, i) => Math.imul(i + 7, 0x9e3779b9) >>> 0,
          );
          const steps = (EXPERTS * per) / GROUP;
          const stepsF32 = Float32Array.from({ length: steps }, (_, i) => ((i % 23) + 1) * 0.00021);
          // f16 normals between about 1e-3 and 1.5e-2 (the range of 26B's expert steps).
          const stepsF16 = Uint16Array.from(
            { length: steps },
            (_, i) => 0x1400 + ((i * 131) % 0x1000),
          );
          // The same experts copied into their slots (the bytes a load writes).
          const pool = <
            T extends
              | Float32Array<ArrayBuffer>
              | Uint32Array<ArrayBuffer>
              | Uint16Array<ArrayBuffer>,
          >(
            full: T,
            perExpert: number,
            empty: (length: number) => T,
          ): T => {
            const slots = empty(SLOTS * perExpert);
            TABLE.forEach((slot, expert) => {
              if (slot >= 0 && slot < SLOTS) {
                slots.set(
                  full.subarray(expert * perExpert, (expert + 1) * perExpert),
                  slot * perExpert,
                );
              }
            });
            return slots;
          };
          const fullWeight = storage === "f32" ? make(weightF32) : make(words);
          const poolWeight =
            storage === "f32"
              ? make(pool(weightF32, per, (length) => new Float32Array(length)))
              : make(pool(words, per / 8, (length) => new Uint32Array(length)));
          const scales =
            storage === "i4"
              ? [
                  make(stepsF32),
                  make(pool(stepsF32, per / GROUP, (length) => new Float32Array(length))),
                ]
              : storage === "i4f16"
                ? [
                    make(new Uint32Array(stepsF16.buffer)),
                    make(
                      new Uint32Array(
                        pool(stepsF16, per / GROUP, (length) => new Uint16Array(length)).buffer,
                      ),
                    ),
                  ]
                : undefined;
          const resident = await pipelineOf(expertLinearWgsl(storage, group));
          const pooled = await pipelineOf(expertLinearWgsl(storage, group, true));
          for (const sharedX of [true, false]) {
            const x = make(
              Float32Array.from(
                { length: (sharedX ? tokens : rows) * k },
                (_, i) => Math.cos(i * 0.029) * 0.9,
              ),
            );
            const index = make(Int32Array.from(IDS));
            const out = make(
              rows * n * 4,
              GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            );
            const shape = { tokens, picks, n, k, experts: EXPERTS, sharedX };
            const expected = await dispatch(
              resident,
              [
                [
                  0,
                  make(
                    expertLinearParams(storage, shape, group),
                    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                  ),
                ],
                [1, x],
                [2, index],
                [3, fullWeight],
                [4, out],
                ...(scales === undefined
                  ? []
                  : [[EXPERT_LINEAR_SCALE_BINDING, scales[0]] as const]),
              ],
              out,
              rows,
              n,
            );
            const got = await dispatch(
              pooled,
              [
                [
                  0,
                  make(
                    expertLinearParams(storage, { ...shape, slots: SLOTS }, group),
                    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                  ),
                ],
                [1, x],
                [2, index],
                [3, poolWeight],
                [4, out],
                ...(scales === undefined
                  ? []
                  : [[EXPERT_LINEAR_SCALE_BINDING, scales[1]] as const]),
                [EXPERT_LINEAR_TABLE_BINDING, make(Int32Array.from(TABLE))],
              ],
              out,
              rows,
              n,
            );
            IDS.forEach((id, row) => {
              const where = `${storage} k${k} ${sharedX ? "shared" : "per-pick"} row ${row} (id ${id})`;
              const words = [...got.subarray(row * n, (row + 1) * n)];
              const kind = rowKind(id);
              if (kind === "resident") {
                assertEquals(words, [...expected.subarray(row * n, (row + 1) * n)], where);
              } else {
                assertEquals(
                  words,
                  new Array(n).fill(kind === "zero" ? 0 : EXPERT_LINEAR_OOB_BITS),
                  where,
                );
              }
            });
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

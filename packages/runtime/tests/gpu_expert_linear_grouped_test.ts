// The grouped form of the expert_linear kernel on a real GPU: the rows of a run that picked one
// expert share the walk of that expert's weight row. Against the one-row form on the same inputs
// (which gpu_expert_linear_test.ts holds to linear's GEMV), every output word MUST be the same:
//
// - an expert picked 2, 3, 5, 7 and 9 times — every walk the kernel generates (8, 4, 2 and 1 rows)
//   alone and after another (9 = 8 + 1, 7 = 4 + 2 + 1, 5 = 4 + 1, 3 = 2 + 1) — by rows of other
//   tokens and twice by one token;
// - resident and pooled (f32 / i4 / i4f16, shared and per-pick x, an edge column count, k with a
//   remainder word and a 26B width);
// - rows the kernel does not sum: an id out of range (NaN), and in the pool an expert that is not
//   resident (0) and a broken slot (NaN);
// - rows that grid-stride (the y cap below the row count), so a walk writes rows of other threads'
//   strides, and rows that do not.
//
// The out buffer starts as a NaN pattern no kernel writes, so a row nobody wrote mismatches.

import { assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import {
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
/** A NaN pattern that is neither the kernel's out-of-range NaN nor a sum. */
const UNWRITTEN = 0x7fc00123;
const EXPERTS = 7;
const SLOTS = 5;
const PICKS = 3;
/**
 * Ten tokens of three picks. Expert 4 is picked by nine tokens (one more row than a scan finds),
 * expert 1 by seven, expert 6 five times (twice by the last token), expert 2 three times (twice by
 * token 5), expert 5 twice; expert 0 is not resident in the pool, expert 3's slot is broken there
 * (resident, each is one more row on its own), and −1 and E are out of range.
 */
const IDS = [
  [4, 1, 5],
  [1, 4, 0],
  [4, 2, 1],
  [4, 1, EXPERTS],
  [4, 3, 1],
  [2, 2, -1],
  [4, 1, 6],
  [4, 6, 1],
  [6, 4, 5],
  [6, 4, 6],
].flat();
const TOKENS = IDS.length / PICKS;
/** Expert → slot: 0 is not resident, 3 has a slot of K (a broken entry). */
const TABLE = [-1, 3, 0, SLOTS, 1, 4, 2];

describe.skipIf(!GPU_AVAILABLE)("the grouped expert_linear kernel (real GPU)", () => {
  const withKernels = async (
    body: (run: {
      /** The out words of one dispatch of `wgsl`, per case, with the case's name. */
      readonly cases: (
        grouped: (
          storage: ExpertLinearStorage,
          group: number | undefined,
          pooled: boolean,
        ) => string,
        each: (
          where: string,
          got: Uint32Array<ArrayBuffer>,
          expected: Uint32Array<ArrayBuffer>,
          n: number,
        ) => void,
      ) => Promise<void>;
    }) => Promise<void>,
  ): Promise<void> => {
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
    const dispatch = async (
      pipeline: GPUComputePipeline,
      entries: readonly (readonly [number, GPUBuffer])[],
      out: GPUBuffer,
      rows: number,
      n: number,
      yCap: number,
    ): Promise<Uint32Array<ArrayBuffer>> => {
      device.queue.writeBuffer(out, 0, new Uint32Array(rows * n).fill(UNWRITTEN));
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
      pass.dispatchWorkgroups(...expertLinearDispatch(rows, n, yCap));
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
      await body({
        cases: async (grouped, each) => {
          const n = 36; // not a multiple of the 32 columns per workgroup (edge columns)
          const rows = TOKENS * PICKS;
          for (const storage of [
            "f32",
            "i4",
            "i4f16",
          ] as const satisfies readonly ExpertLinearStorage[]) {
            const unit = expertLinearUnit(storage);
            const group = storage === "f32" ? undefined : GROUP;
            // One prefetch round plus a remainder word, and a 26B width.
            for (const k of [unit * 5, 704]) {
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
              const stepsF32 = Float32Array.from(
                { length: steps },
                (_, i) => ((i % 23) + 1) * 0.00021,
              );
              const stepsF16 = Uint16Array.from(
                { length: steps + (steps % 2) },
                (_, i) => 0x1400 + ((i * 131) % 0x1000),
              );
              /** The experts copied into their slots (the bytes a load writes). */
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
              for (const pooled of [false, true]) {
                const weight = pooled
                  ? storage === "f32"
                    ? make(pool(weightF32, per, (length) => new Float32Array(length)))
                    : make(pool(words, per / 8, (length) => new Uint32Array(length)))
                  : storage === "f32"
                    ? make(weightF32)
                    : make(words);
                const pooledSteps = (): Uint32Array<ArrayBuffer> => {
                  const halves = pool(
                    stepsF16.subarray(0, steps),
                    per / GROUP,
                    (length) => new Uint16Array(length + (length % 2)),
                  );
                  return new Uint32Array(halves.buffer);
                };
                const scale =
                  storage === "i4"
                    ? make(
                        pooled
                          ? pool(stepsF32, per / GROUP, (length) => new Float32Array(length))
                          : stepsF32,
                      )
                    : storage === "i4f16"
                      ? make(pooled ? pooledSteps() : new Uint32Array(stepsF16.buffer))
                      : undefined;
                const oneRow = await pipelineOf(expertLinearWgsl(storage, group, pooled));
                const candidate = await pipelineOf(grouped(storage, group, pooled));
                for (const sharedX of [true, false]) {
                  const x = make(
                    Float32Array.from(
                      { length: (sharedX ? TOKENS : rows) * k },
                      (_, i) => Math.cos(i * 0.029) * 0.9,
                    ),
                  );
                  const out = make(
                    rows * n * 4,
                    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                  );
                  const entries: readonly (readonly [number, GPUBuffer])[] = [
                    [
                      0,
                      make(
                        expertLinearParams(
                          storage,
                          {
                            tokens: TOKENS,
                            picks: PICKS,
                            n,
                            k,
                            experts: EXPERTS,
                            sharedX,
                            ...(pooled ? { slots: SLOTS } : {}),
                          },
                          group,
                        ),
                        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                      ),
                    ],
                    [1, x],
                    [2, make(Int32Array.from(IDS))],
                    [3, weight],
                    [4, out],
                    ...(scale === undefined ? [] : [[EXPERT_LINEAR_SCALE_BINDING, scale] as const]),
                    ...(pooled
                      ? [[EXPERT_LINEAR_TABLE_BINDING, make(Int32Array.from(TABLE))] as const]
                      : []),
                  ];
                  // A y cap of 4 makes the 30 rows grid-stride; 64 gives every row its own thread.
                  for (const yCap of [4, 64]) {
                    const expected = await dispatch(oneRow, entries, out, rows, n, yCap);
                    const got = await dispatch(candidate, entries, out, rows, n, yCap);
                    each(
                      `${storage} k${k} ${pooled ? "pooled" : "resident"} ${sharedX ? "shared" : "per-pick"} x, y cap ${yCap}`,
                      got,
                      expected,
                      n,
                    );
                  }
                }
              }
            }
          }
        },
      });
      const error = await device.popErrorScope();
      assertEquals(error, null, error?.message);
    } finally {
      for (const buffer of owned) buffer.destroy();
      gpu.destroy();
    }
  };

  it("writes the one-row form's words for every row, however many rows picked the row's expert", async () => {
    await withKernels(async ({ cases }) => {
      await cases(
        (storage, group, pooled) => expertLinearWgsl(storage, group, pooled, true),
        (where, got, expected, n) => {
          // The one-row side wrote every row (a sum, 0 or the out-of-range NaN).
          assertEquals(expected.includes(UNWRITTEN), false, `${where}: the one-row form`);
          IDS.forEach((id, row) => {
            assertEquals(
              [...got.subarray(row * n, (row + 1) * n)],
              [...expected.subarray(row * n, (row + 1) * n)],
              `${where} row ${row} (id ${id})`,
            );
          });
        },
      );
    });
  });

  it("the comparison can fail: walks that write their first row's sum to their second row mismatch", async () => {
    let differing = 0;
    await withKernels(async ({ cases }) => {
      await cases(
        (storage, group, pooled) => {
          const wgsl = expertLinearWgsl(storage, group, pooled, true);
          const fault = wgsl.replaceAll(
            "out[m1 * dims.n + col] = acc1;",
            "out[m1 * dims.n + col] = acc0;",
          );
          assertNotEquals(fault, wgsl, "the fault changed no line");
          return fault;
        },
        (_where, got, expected) => {
          differing += got.reduce((count, word, i) => count + (word === expected[i] ? 0 : 1), 0);
        },
      );
    });
    assertNotEquals(differing, 0);
  });
});

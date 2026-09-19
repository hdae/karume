// 融合前の parallel と u32 で比較し、独立 f64 参照の既存 5e-6 も維持する。
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { acquireGpu } from "../src/gpu/device.ts";
import { referenceStateAttention } from "../src/reference/state-attention.ts";
import { stateStatsPvEligible } from "../src/kernels/state-attention-stats-pv.ts";
import {
  halfScale,
  runStateAttention,
  seeded,
  type StateCase,
  type StateInputs,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const inputsFor = (spec: StateCase): StateInputs => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = spec;
  const values = (count: number, salt: number): Float32Array<ArrayBuffer> =>
    seeded(count, (i) => Math.sin(i * 0.371 + salt) * 0.75);
  const inputs = {
    q: values(batch * heads * chunkRows * depth, 1),
    insK: values(batch * kvHeads * chunkRows * depth, 2),
    insV: values(batch * kvHeads * chunkRows * depth, 3),
    slotK: values(batch * kvHeads * capacity * depth, 4),
    slotV: values(batch * kvHeads * capacity * depth, 5),
  };
  const resident = new Set<number>();
  for (let col = window ? Math.max(0, past - window + 1) : 0; col < past; col++) {
    resident.add(window ? col % capacity : col);
  }
  for (let plane = 0; plane < batch * kvHeads; plane++) {
    for (let row = 0; row < capacity; row++) {
      if (resident.has(row)) continue;
      inputs.slotK.fill(20, (plane * capacity + row) * depth, (plane * capacity + row + 1) * depth);
      inputs.slotV.fill(
        400,
        (plane * capacity + row) * depth,
        (plane * capacity + row + 1) * depth,
      );
    }
    for (let row = query; row < chunkRows; row++) {
      inputs.insK.fill(
        20,
        (plane * chunkRows + row) * depth,
        (plane * chunkRows + row + 1) * depth,
      );
      inputs.insV.fill(
        400,
        (plane * chunkRows + row) * depth,
        (plane * chunkRows + row + 1) * depth,
      );
    }
  }
  return inputs;
};

describe("states attention の行統計/PV融合", () => {
  it("静的な適用境界を守る", () => {
    assert(stateStatsPvEligible(1, 1024));
    assert(stateStatsPvEligible(8, 1024));
    assertEquals(stateStatsPvEligible(9, 1024), false);
    assertEquals(stateStatsPvEligible(1, 1025), false);
  });

  it({
    name: "ring・GQA・端数D・pad・行分割でparallelと同じ値を返す",
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const gpu = await acquireGpu(), cache = new Map<string, GPUComputePipeline>();
      let maxAbs = 0;
      try {
        for (const depth of [1, 17, 256, 512]) {
          for (
            const [chunkRows, query, window, past] of [
              [1, 1, 0, 0],
              [4, 4, 0, 31],
              [8, 1, 0, 95],
              [8, 4, 1, 4095],
              [4, 1, 31, 4095],
              [8, 8, 127, 0xfffffff0],
              // live 701（= past + query）で行統計の 256 レーン grid-stride が 3 周回る形。
              // 他の 6 形は live ≤ 134 なので 1 周で終わり、多周回の加算順が ② と一致する
              // ことを実機で見る線が無い（colCap 717 ≤ 1024 で適用条件は満たす）。
              // 実行時間が depth ループ 4 本に掛かるので、長い列はこの 1 形だけにする。
              [1, 1, 0, 700],
            ]
          ) {
            const spec: StateCase = {
              name: "fused",
              batch: 2,
              heads: 4,
              kvHeads: depth === 1 ? 4 : 1,
              chunkRows,
              query,
              window,
              past,
              depth,
              capacity: window ? window + 8 : past + chunkRows + 16,
            };
            const inputs = inputsFor(spec);
            const ref = referenceStateAttention({ ...spec, ...inputs, scale: halfScale(depth) });
            const before = await runStateAttention(gpu.device, spec, inputs, {
              cache,
              qkReduce: "parallel",
              pvReduce: "parallel",
            });
            for (const rowsBlock of [chunkRows, 1]) {
              const after = await runStateAttention(gpu.device, { ...spec, rowsBlock }, inputs, {
                cache,
                qkReduce: "parallel",
                pvReduce: "parallel",
                statsPvFusion: true,
              });
              assertEquals(new Uint32Array(after.out.buffer), new Uint32Array(before.out.buffer));
              for (let i = 0; i < after.out.length; i++) {
                const error = Math.abs(after.out[i] - ref.data[i]);
                assert(error < 5e-6, "states attention の既存絶対許容差");
                maxAbs = Math.max(maxAbs, error);
              }
            }
          }
        }
        console.log("stats/PV融合 vs f64 maxAbs", maxAbs);
      } finally {
        gpu.destroy();
      }
    },
  });

  it({
    name: "NaN/Infの分類とpadの厳密0を維持する",
    ignore: !GPU_AVAILABLE,
    fn: async () => {
      const gpu = await acquireGpu(), cache = new Map<string, GPUComputePipeline>();
      try {
        const spec: StateCase = {
          name: "special",
          batch: 1,
          heads: 4,
          kvHeads: 1,
          chunkRows: 4,
          query: 1,
          window: 31,
          past: 40,
          capacity: 39,
          depth: 17,
        };
        for (const target of ["q", "insK", "insV", "slotK", "slotV"] as const) {
          const inputs = inputsFor(spec);
          inputs[target].set([NaN, Infinity, -Infinity, 0, -0]);
          const before = await runStateAttention(gpu.device, spec, inputs, {
            cache,
            qkReduce: "parallel",
            pvReduce: "parallel",
          });
          const after = await runStateAttention(gpu.device, spec, inputs, {
            cache,
            qkReduce: "parallel",
            pvReduce: "parallel",
            statsPvFusion: true,
          });
          const a = new Uint32Array(after.out.buffer), b = new Uint32Array(before.out.buffer);
          for (let i = 0; i < a.length; i++) {
            if (Number.isNaN(before.out[i])) assert(Number.isNaN(after.out[i]));
            else assertEquals(a[i], b[i]);
            if (Math.floor(i / spec.depth) % spec.chunkRows >= spec.query) assertEquals(a[i], 0);
          }
        }
      } finally {
        gpu.destroy();
      }
    },
  });
});

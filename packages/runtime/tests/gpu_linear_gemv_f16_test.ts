// f16 格納 M=1 の GEMV を、通常 GEMM と独立 CPU 参照の両方で検収する（ADR 0082 追記 6）。
// 実モデルは踏まない N の端、先読みの端、packed f16 の対・語の境界も含める。
import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { linearGemvKey } from "../src/kernels/linear-gemv.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, refTensor } from "../src/reference/ops.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
import { quantizeF16 } from "./helpers/f16.ts";
import { buildSafetensors, f32Bytes, type GraphJson } from "./helpers/format.ts";
import {
  GPU_AVAILABLE,
  SHADER_F16_AVAILABLE,
  TIMESTAMP_QUERY_AVAILABLE,
  TIMING_ACQUIRE_OPTIONS,
} from "./helpers/gpu.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";

const activation = (length: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length }, (_, i) => Math.sin(i * 0.73) * 1.17);

const fixture = (m: number, n: number, k: number, storage: "f16" | "f32" = "f16") => {
  // 隣接要素の符号・大きさを散らす。対の lo/hi 入れ替えを対称性で打ち消さない。
  const weight = quantizeF16(Float32Array.from(
    { length: n * k },
    (_, i) => ((i % 2 === 0 ? 1 : -1) * (0.0013 + (i % 23) * 0.017)),
  ));
  const bias = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.13);
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "w", storage: { dtype: storage } },
      b: { tensor: "b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const bytes = buildSafetensors([
    {
      name: "w",
      dtype: storage === "f16" ? "F16" : "F32",
      shape: [n, k],
      data: storage === "f16" ? weight.bytes : f32Bytes([...weight.values]),
    },
    { name: "b", dtype: "F32", shape: [n], data: f32Bytes([...bias]) },
  ], { karume_ir: JSON.stringify(graph) });
  const input = refTensor([m, k], activation(m * k));
  return {
    bytes,
    input,
    reference: () =>
      applyReferenceOp(
        "linear",
        [input, refTensor([n, k], weight.values), refTensor([n], bias)],
        {},
        [m, n],
      ),
  };
};

const run = async (
  gpu: GpuContext,
  data: ReturnType<typeof fixture>,
  options: SessionOptions = {},
): Promise<{ output: Tensor; keys: readonly string[] }> => {
  const session = await createSession(gpu, openModel(data.bytes), options);
  try {
    const { y } = await session.run({ x: data.input });
    return {
      output: y,
      keys: session.diagnostics().lastRunTiming?.entries.map((e) => e.key) ?? [],
    };
  } finally {
    await session.dispose();
  }
};

const bits = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

Deno.test({
  name: "f16 M=1 GEMV は端の形でも通常 GEMM とビット一致し CPU 参照を満たす（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const [n, k] of [[64, 128], [100, 40], [36, 48], [68, 56], [4, 8], [32, 24], [36, 136], [
          100,
          264,
        ]]
      ) {
        const data = fixture(1, n, k);
        const actual = await run(gpu, data);
        // f16 の門は M=1 だけ。M=2 は同じ M16N16 GEMM 骨格で、族内比較にはならない。
        const expected = await run(gpu, fixture(2, n, k));
        if (TIMESTAMP_QUERY_AVAILABLE) {
          assertEquals(actual.keys, [linearGemvKey("f16")]);
          assertEquals(expected.keys, [linearKey("f16", true, "f32", 2)]);
        }
        assertEquals(actual.output.shape, [1, n]);
        assertEquals(bits(actual.output), bits(expected.output).subarray(0, n), `n=${n} k=${k}`);
        const report = compareTensors(actual.output, data.reference(), GEMM_TOLERANCE);
        assert(report.pass, formatAllclose(report));
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "f16 GEMV の行数・K 整列・N 整列・格納の門は条件を外すと通常 GEMM を選ぶ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const c of [
          { m: 2, n: 36, k: 40, storage: "f16" },
          { m: 1, n: 36, k: 36, storage: "f16" },
          { m: 1, n: 35, k: 40, storage: "f16" },
          { m: 1, n: 36, k: 40, storage: "f32" },
        ] as const
      ) {
        const data = fixture(c.m, c.n, c.k, c.storage);
        const actual = await run(gpu, data);
        if (TIMESTAMP_QUERY_AVAILABLE) {
          assertEquals(actual.keys, [
            linearKey(c.storage, c.k % 4 === 0 && c.n % 4 === 0, "f32", c.m),
          ]);
        }
        const report = compareTensors(actual.output, data.reference(), GEMM_TOLERANCE);
        assert(report.pass, formatAllclose(report));
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "f16 格納でも f16 計算の指定は GEMV に置き換えない（実 GPU）",
  ignore: !SHADER_F16_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ ...TIMING_ACQUIRE_OPTIONS, shaderF16: true });
    try {
      const actual = await run(gpu, fixture(1, 36, 40), { linearCompute: "f16" });
      if (TIMESTAMP_QUERY_AVAILABLE) {
        assertEquals(actual.keys, [linearKey("f16", true, "f16", 1)]);
      }
    } finally {
      gpu.destroy();
    }
  },
});

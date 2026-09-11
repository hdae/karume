// f32 格納 M=1 の GEMV を、通常 GEMM と独立 CPU 参照の両方で検収する（ADR 0082 追記 7）。
// 実モデルは踏まない N の端、先読みの端、f32 の成分・語の境界も含める。
import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { linearGemvKey } from "../src/kernels/linear-gemv.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { applyReferenceOp, refTensor } from "../src/reference/ops.ts";
import { createSession, type SessionOptions, type Tensor } from "../src/runtime/executor.ts";
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

const fixture = (m: number, n: number, k: number) => {
  const weight = Float32Array.from(
    { length: n * k },
    (_, i) => (i % 2 === 0 ? 1 : -1) * (0.0013 + (i % 23) * 0.017),
  );
  const bias = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.13);
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "w", storage: { dtype: "f32" } },
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
      dtype: "F32",
      shape: [n, k],
      data: f32Bytes([...weight]),
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
        [input, refTensor([n, k], weight), refTensor([n], bias)],
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
  name: "f32 M=1 GEMV は端の形でも通常 GEMM とビット一致し CPU 参照を満たす（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const [n, k] of [[64, 128], [100, 20], [36, 24], [68, 28], [4, 4], [32, 12], [36, 132], [
          100,
          260,
        ]]
      ) {
        const data = fixture(1, n, k);
        const actual = await run(gpu, data);
        // f32 の門は M=1 だけ。M=2 は同じ M16N16 GEMM 骨格で、族内比較にはならない。
        const expected = await run(gpu, fixture(2, n, k));
        if (TIMESTAMP_QUERY_AVAILABLE) {
          assertEquals(actual.keys, [linearGemvKey("f32")]);
          assertEquals(expected.keys, [linearKey("f32", true, "f32", 2)]);
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
  name: "f32 GEMV の行数・K 整列・N 整列の門は条件を外すと通常 GEMM を選ぶ（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    try {
      for (
        const c of [
          { m: 2, n: 36, k: 20 },
          { m: 1, n: 36, k: 18 },
          { m: 1, n: 35, k: 20 },
        ] as const
      ) {
        const data = fixture(c.m, c.n, c.k);
        const actual = await run(gpu, data);
        if (TIMESTAMP_QUERY_AVAILABLE) {
          assertEquals(actual.keys, [
            linearKey("f32", c.k % 4 === 0 && c.n % 4 === 0, "f32", c.m),
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
  name: "f32 格納でも f16 計算の指定は GEMV に置き換えない（実 GPU）",
  ignore: !SHADER_F16_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ ...TIMING_ACQUIRE_OPTIONS, shaderF16: true });
    try {
      const actual = await run(gpu, fixture(1, 36, 40), { linearCompute: "f16" });
      if (TIMESTAMP_QUERY_AVAILABLE) {
        assertEquals(actual.keys, [linearKey("f32", true, "f16", 1)]);
      }
    } finally {
      gpu.destroy();
    }
  },
});

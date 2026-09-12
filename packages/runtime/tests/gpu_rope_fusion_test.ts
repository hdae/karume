// half-split RoPE の exact peephole（融合ルール rope — src/runtime/fusion.ts）:
// 既存 primitive 列とのビット parity と dispatch 削減を直接固定する。

import { assert, assertEquals } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { ROPE_BSHD_KEY, ROPE_KEY } from "../src/kernels/rope.ts";
import { createSession, type Tensor } from "../src/runtime/executor.ts";
import type { GraphJson } from "./helpers/format.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE, TIMESTAMP_QUERY_AVAILABLE, TIMING_ACQUIRE_OPTIONS } from "./helpers/gpu.ts";

// H=1, S=3 では n=384 となり、256 スレッド workgroup の末尾端数も通る。
type RopeOrder = "slice-first" | "direct-first";

/**
 * `interpose=true` は first slice の直後へ 0 dispatch の reshape を挟み、式を変えず exact
 * matcher だけを外す。これで同一バックエンド上の既存 8 dispatch 列を正本として直接比較できる。
 *
 * `heads` は x の H 軸。カーネルは `row = i / head_dim` を `sequence` で割って token を出す形で
 * head 方向を畳んでいるので、H≥2 を通さないとその添字が 1 度も踏まれない。
 */
const ropeGraph = (
  order: RopeOrder,
  interpose: boolean,
  heads = 1,
  headDim = 128,
  layout: "bhsd" | "bshd" = "bhsd",
): GraphJson => {
  const shape = layout === "bshd"
    ? [1, "S", heads, headDim] as const
    : [1, heads, "S", headDim] as const;
  const tableShape = layout === "bshd"
    ? [1, "S", 1, headDim] as const
    : [1, 1, "S", headDim] as const;
  const halfShape = layout === "bshd"
    ? [1, "S", heads, headDim / 2] as const
    : [1, heads, "S", headDim / 2] as const;
  const values: GraphJson["values"] = {
    first: { dtype: "f32", shape: [...halfShape] },
    second: { dtype: "f32", shape: [...halfShape] },
    negative: { dtype: "f32", shape: [...halfShape] },
    rotated: { dtype: "f32", shape: [...shape] },
    direct: { dtype: "f32", shape: [...shape] },
    cross: { dtype: "f32", shape: [...shape] },
    y: { dtype: "f32", shape: [...shape] },
  };
  if (interpose) values.first_alias = { dtype: "f32", shape: [...halfShape] };
  const direct: GraphJson["nodes"][number] = {
    op: "mul",
    ins: ["x", "cos"],
    outs: ["direct"],
    attrs: {},
  };
  const nodes: GraphJson["nodes"] = [];
  if (order === "direct-first") nodes.push(direct);
  nodes.push({
    op: "slice",
    ins: ["x"],
    outs: ["first"],
    attrs: { dim: 3, start: 0, end: headDim / 2 },
  });
  if (interpose) {
    nodes.push({ op: "reshape", ins: ["first"], outs: ["first_alias"], attrs: {} });
  }
  nodes.push(
    {
      op: "slice",
      ins: ["x"],
      outs: ["second"],
      attrs: { dim: 3, start: headDim / 2, end: headDim },
    },
    { op: "neg", ins: ["second"], outs: ["negative"], attrs: {} },
    {
      op: "cat",
      ins: ["negative", interpose ? "first_alias" : "first"],
      outs: ["rotated"],
      attrs: { dim: 3 },
    },
  );
  if (order === "slice-first") nodes.push(direct);
  nodes.push(
    { op: "mul", ins: ["rotated", "sin"], outs: ["cross"], attrs: {} },
    { op: "add", ins: ["direct", "cross"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: ["S"],
    inputs: [
      { name: "x", dtype: "f32", shape: [...shape] },
      { name: "cos", dtype: "f32", shape: [...tableShape] },
      { name: "sin", dtype: "f32", shape: [...tableShape] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertFloatParity = (actual: Uint32Array, expected: Uint32Array): void => {
  assertEquals(actual.length, expected.length);
  const word = new Uint32Array(1);
  const value = new Float32Array(word.buffer);
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] === expected[i]) continue;
    word[0] = actual[i];
    const actualValue = value[0];
    word[0] = expected[i];
    const expectedValue = value[0];
    assertEquals(
      Number.isNaN(actualValue) && Number.isNaN(expectedValue),
      true,
      "非 NaN はビット一致する index=" + i + " actual=0x" + actual[i].toString(16) +
        " expected=0x" + expected[i].toString(16),
    );
  }
};

/** x の先頭に ±0 / subnormal / ±Inf / NaN payload を差し込んだ入力一式。 */
const ropeInputs = (heads: number, sequence: number): Readonly<Record<string, Tensor>> => {
  const x = fill([1, heads, sequence, 128], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  new Uint32Array(x.data.buffer).set([
    0x00000000, // +0
    0x80000000, // -0
    0x00000001, // 最小正 subnormal
    0x80000001, // 最小負 subnormal
    0x007fffff, // 最大正 subnormal
    0x807fffff, // 最大負 subnormal
    0x7f800000, // +Inf
    0xff800000, // -Inf
    0x7fc01234, // quiet NaN payload
    0xff801234, // 負の signaling NaN payload
  ]);
  return {
    x,
    cos: fill([1, 1, sequence, 128], (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.23 - 0.2)),
  };
};

Deno.test({
  name:
    "half-split RoPE 融合は slice/direct-first 両順で有限値ビット / NaN 分類が一致し 1 dispatch へ畳む（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    const inputs = ropeInputs(1, 3);
    try {
      for (const order of ["slice-first", "direct-first"] as const) {
        const fused = await createSession(
          gpu,
          openModel(graphModelBuffer(ropeGraph(order, false))),
        );
        const split = await createSession(gpu, openModel(graphModelBuffer(ropeGraph(order, true))));
        try {
          const fusedOut = (await fused.run(inputs)).y;
          const splitOut = (await split.run(inputs)).y;
          assertFloatParity(bits(fusedOut), bits(splitOut));
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: RoPE 1 pass`);
          assertEquals(
            split.diagnostics().submit.dispatchCount,
            8,
            `${order}: slice×2 + neg + cat×2 + mul×2 + add`,
          );
          assertEquals(fused.diagnostics().lastRunFusions?.rope, 1, `${order}: 融合カウンタ`);
          assertEquals(
            split.diagnostics().lastRunFusions?.rope,
            0,
            `${order}: 反例のカウンタは 0`,
          );
        } finally {
          await fused.dispose();
          await split.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

/**
 * **census**（ADR 0058 決定 4）。融合が RoPE カーネル 1 本だけで済んでいることはキーでしか
 * 見えない（dispatch 数だけでは別カーネル 1 本へ化けた形を見逃す）。
 *
 * MUST: 計測を要求しない device（`TIMESTAMP_QUERY_AVAILABLE` が偽）では**明示 SKIP** し、
 * 走るときは内訳を無条件に検査する（`timing !== undefined` で守ると、計測なし機では
 * キー検査が 1 つも走らないまま緑になる — gpu_attention_gqa_test.ts の census と同じ形）。
 */
Deno.test({
  name: "half-split RoPE 融合が走らせるのは RoPE カーネル 1 本だけ（実 GPU / timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu(TIMING_ACQUIRE_OPTIONS);
    const inputs = ropeInputs(1, 3);
    try {
      for (const order of ["slice-first", "direct-first"] as const) {
        const fused = await createSession(
          gpu,
          openModel(graphModelBuffer(ropeGraph(order, false))),
        );
        try {
          await fused.run(inputs);
          const timing = fused.diagnostics().lastRunTiming;
          assert(timing !== undefined, `${order}: 内訳が空（キー検査が空振りしている）`);
          assertEquals(timing.entries.map((entry) => entry.key), [ROPE_KEY]);
        } finally {
          await fused.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

// H=1 だけでは `row / sequence` の head 方向の畳み込みが 1 度も踏まれない（row < sequence の
// ままなので token = row になってしまう）。実配布は H=16 なのでここが本番の添字。
Deno.test({
  name: "RoPE 融合は H≥2 でも head 方向の添字が primitive 列とビット一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const inputs = ropeInputs(3, 5);
    try {
      for (const order of ["slice-first", "direct-first"] as const) {
        const fused = await createSession(
          gpu,
          openModel(graphModelBuffer(ropeGraph(order, false, 3))),
        );
        const split = await createSession(
          gpu,
          openModel(graphModelBuffer(ropeGraph(order, true, 3))),
        );
        try {
          const fusedOut = (await fused.run(inputs)).y;
          const splitOut = (await split.run(inputs)).y;
          assertEquals(fusedOut.shape, [1, 3, 5, 128], `${order}: 出力 shape`);
          assertFloatParity(bits(fusedOut), bits(splitOut));
          assertEquals(fused.diagnostics().submit.dispatchCount, 1, `${order}: RoPE 1 pass`);
          assertEquals(fused.diagnostics().lastRunFusions?.rope, 1, `${order}: 融合カウンタ`);
        } finally {
          await fused.dispose();
          await split.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "RoPE の内部値が graph output または別 consumer を持てば融合しない（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    const sequence = 2;
    const inputs = {
      x: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.11) + i / 4093),
      cos: fill([1, 1, sequence, 128], (i) => Math.cos(i * 0.07)),
      sin: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.13)),
    };
    const x = inputs.x;
    try {
      const outputGraph = ropeGraph("slice-first", false);
      outputGraph.outputs = ["first", "y"];
      const outputSession = await createSession(gpu, openModel(graphModelBuffer(outputGraph)));
      try {
        const outputs = await outputSession.run(inputs);
        assertEquals(
          outputSession.diagnostics().submit.dispatchCount,
          8,
          "内部 output があれば fallback",
        );
        assertEquals(outputSession.diagnostics().lastRunFusions?.rope, 0, "融合カウンタ 0");
        const first = outputs.first.data;
        for (let row = 0; row < sequence; row++) {
          for (let d = 0; d < 64; d++) {
            assertEquals(first[row * 64 + d], x.data[row * 128 + d], `row=${row}:d=${d}`);
          }
        }
      } finally {
        await outputSession.dispose();
      }

      const directOutputGraph = ropeGraph("direct-first", false);
      directOutputGraph.outputs = ["direct", "y"];
      const directOutputSession = await createSession(
        gpu,
        openModel(graphModelBuffer(directOutputGraph)),
      );
      try {
        const outputs = await directOutputSession.run(inputs);
        assertEquals(
          directOutputSession.diagnostics().submit.dispatchCount,
          8,
          "direct-first の先頭内部 output があれば fallback",
        );
        assertEquals(directOutputSession.diagnostics().lastRunFusions?.rope, 0, "融合カウンタ 0");
        const direct = outputs.direct.data;
        for (let i = 0; i < direct.length; i++) {
          assertEquals(direct[i], Math.fround(x.data[i] * inputs.cos.data[i]), `direct i=${i}`);
        }
      } finally {
        await directOutputSession.dispose();
      }

      const consumerGraph = ropeGraph("slice-first", false);
      consumerGraph.values.first_copy = { dtype: "f32", shape: [1, 1, "S", 64] };
      consumerGraph.nodes.push({
        op: "neg",
        ins: ["first"],
        outs: ["first_copy"],
        attrs: {},
      });
      consumerGraph.outputs = ["first_copy", "y"];
      const consumerSession = await createSession(gpu, openModel(graphModelBuffer(consumerGraph)));
      try {
        const outputs = await consumerSession.run(inputs);
        assertEquals(
          consumerSession.diagnostics().submit.dispatchCount,
          9,
          "内部値の use count が 2 なら 8 dispatch 列 + consumer へ fallback",
        );
        assertEquals(consumerSession.diagnostics().lastRunFusions?.rope, 0, "融合カウンタ 0");
        const firstCopy = outputs.first_copy.data;
        for (let row = 0; row < sequence; row++) {
          for (let d = 0; d < 64; d++) {
            assertEquals(
              firstCopy[row * 64 + d],
              -x.data[row * 128 + d],
              `consumer row=${row}:d=${d}`,
            );
          }
        }
      } finally {
        await consumerSession.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "BSHD RoPEはtoken/head軸・端数・幅・特殊値をprimitive列とビット一致で処理する（実GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      for (
        const [heads, sequence, dim] of [[8, 1, 256], [16, 17, 64], [3, 17, 6], [4, 3, 128], [
          8,
          9,
          256,
        ], [
          2,
          5,
          512,
        ]]
      ) {
        const inputs = {
          x: fill([1, sequence, heads, dim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997),
          cos: fill([1, sequence, 1, dim], (i) => Math.cos(i * 0.19 + 0.3)),
          sin: fill([1, sequence, 1, dim], (i) => Math.sin(i * 0.23 - 0.2)),
        };
        new Uint32Array(inputs.x.data.buffer).set([
          0,
          0x80000000,
          1,
          0x80000001,
          0x007fffff,
          0x807fffff,
          0x7f800000,
          0xff800000,
          0x7fc01234,
          0xff801234,
        ]);
        for (const order of ["slice-first", "direct-first"] as const) {
          const fused = await createSession(
            gpu,
            openModel(graphModelBuffer(ropeGraph(order, false, heads, dim, "bshd"))),
          );
          const split = await createSession(
            gpu,
            openModel(graphModelBuffer(ropeGraph(order, true, heads, dim, "bshd"))),
          );
          try {
            const actual = (await fused.run(inputs)).y;
            const expected = (await split.run(inputs)).y;
            assertEquals(actual.shape, [1, sequence, heads, dim]);
            assertFloatParity(bits(actual), bits(expected));
            assertEquals(fused.diagnostics().lastRunFusions?.rope, 1);
            assertEquals(split.diagnostics().lastRunFusions?.rope, 0);
            assertEquals(fused.diagnostics().submit.dispatchCount, 1);
          } finally {
            await fused.dispose();
            await split.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "BSHD RoPEはgrid-strideの複数巡回でもprimitive列とビット一致する（実GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 2 } });
    const heads = 8, sequence = 17, dim = 128;
    const inputs = {
      x: fill([1, sequence, heads, dim], (i) => Math.sin(i * 0.013)),
      cos: fill([1, sequence, 1, dim], (i) => Math.cos(i * 0.017)),
      sin: fill([1, sequence, 1, dim], (i) => Math.sin(i * 0.019)),
    };
    try {
      assertEquals(gpu.limits.maxComputeWorkgroupsPerDimension, 2);
      const fused = await createSession(
        gpu,
        openModel(graphModelBuffer(ropeGraph("direct-first", false, heads, dim, "bshd"))),
      );
      const split = await createSession(
        gpu,
        openModel(graphModelBuffer(ropeGraph("direct-first", true, heads, dim, "bshd"))),
      );
      try {
        assertFloatParity(bits((await fused.run(inputs)).y), bits((await split.run(inputs)).y));
        assertEquals(fused.diagnostics().lastRunFusions?.rope, 1);
      } finally {
        await fused.dispose();
        await split.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "BSHD RoPEは専用キーだけで走る（実GPU/timestamp-query）",
  ignore: !GPU_AVAILABLE || !TIMESTAMP_QUERY_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu({ gpuTiming: true });
    try {
      const session = await createSession(
        gpu,
        openModel(graphModelBuffer(ropeGraph("direct-first", false, 3, 128, "bshd"))),
      );
      try {
        await session.run({
          x: fill([1, 5, 3, 128], (i) => Math.sin(i)),
          cos: fill([1, 5, 1, 128], (i) => Math.cos(i)),
          sin: fill([1, 5, 1, 128], (i) => Math.sin(i)),
        });
        const timing = session.diagnostics().lastRunTiming;
        assert(timing !== undefined);
        assertEquals(timing.entries.map((x) => x.key), [ROPE_BSHD_KEY]);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
});

import { assert, assertEquals, assertThrows } from "@std/assert";
import { admitGemma4Qat, assertGemma4QatPle } from "../src/gemma/qat.ts";
import { gemma4QatRopeInputs, gemma4RopeInputs, type Gemma4RopeSpec } from "../src/gemma/rope.ts";

type Graph = Parameters<typeof admitGemma4Qat>[0];
const graphOf = (model: "e2b" | "e4b"): Graph => {
  const hidden = model === "e2b" ? 1536 : 2560,
    layers = model === "e2b" ? 35 : 42;
  const raw = [
    { op: "embedding", ins: ["head", "input_ids"], outs: ["embedded"] },
    { op: "linear", ins: ["embedded", "projection"], outs: ["projected"] },
    { op: "static_quantize", ins: ["projected"], outs: ["input4"] },
    { op: "linear", ins: ["input4", "w4"], outs: ["linear4"] },
    { op: "static_quantize", ins: ["linear4"], outs: ["input8"] },
    { op: "linear", ins: ["input8", "w8"], outs: ["linear8"] },
    { op: "static_quantize", ins: ["linear8"], outs: ["input2"] },
    { op: "linear", ins: ["input2", "head"], outs: ["linear2"] },
    { op: "static_quantize", ins: ["linear2"], outs: ["logits"] },
  ];
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["embedding", "linear", "static_quantize"] },
    symbols: ["M", "R", "C"],
    inputs: [{ name: "input_ids", dtype: "i32", shape: [1, "M"] }, {
      name: "per_layer_inputs",
      dtype: "f32",
      shape: [1, "M", layers, 256],
    }],
    outputs: ["logits", "hidden"],
    values: {
      logits: { dtype: "f32", shape: [1, "R", 262144] },
      hidden: { dtype: "f32", shape: [1, "R", hidden] },
    },
    initializers: {
      head: {
        tensor: "head.weight",
        storage: { dtype: "i2", scale: "head.scale" },
      },
      projection: {
        tensor: "model.model.per_layer_model_projection.weight",
        storage: { dtype: "f32" },
      },
      w4: {
        tensor: "w4",
        storage: { dtype: "i4", scale: "s4", groupSize: 32 },
      },
      w8: { tensor: "w8", storage: { dtype: "i8", scale: "s8" } },
    },
    states: {},
    nodes: raw.map((node) => ({
      ...node,
      attrs: node.op === "static_quantize" ? { scale: 0.125 } : {},
      states: {},
    })),
  };
};

Deno.test("固定 QAT の family admission", async (t) => {
  for (const model of ["e2b", "e4b"] as const) {
    await t.step(`${model} の固定混成と共有 head を受ける`, () => {
      const graph = graphOf(model);
      assertEquals(admitGemma4Qat(graph, model), model);
      assertThrows(
        () => admitGemma4Qat(graph, model === "e2b" ? "e4b" : "e2b"),
        Error,
        "構成が違う",
      );
    });
  }
  for (const index of [2, 4, 6, 8]) {
    await t.step(`SRQ ${index} が落ちたグラフを拒否する`, () => {
      const graph = graphOf("e2b");
      assertThrows(
        () =>
          admitGemma4Qat({
            ...graph,
            nodes: graph.nodes.map((node, i) => i === index ? { ...node, op: "reshape" } : node),
          }),
        Error,
        "固定 SRQ",
      );
    });
  }
  await t.step("通常の I8 embedding を QAT として受けない", () => {
    const graph = graphOf("e2b");
    assertThrows(
      () =>
        admitGemma4Qat({
          ...graph,
          initializers: {
            ...graph.initializers,
            head: {
              tensor: "head.weight",
              storage: { dtype: "i8", scale: "s" },
            },
          },
        }),
      Error,
      "固定 INT2",
    );
  });
  await t.step("PLE の格納をモデル構成へ突合する", () => {
    const base = {
      tokens: 262144,
      layers: 35,
      dim: 256,
      embedScale: 16,
      shards: [{ file: "ple.safetensors", start: 0, stop: 262144 }],
    };
    assertGemma4QatPle(graphOf("e2b"), { ...base, storage: "i4" });
    assertGemma4QatPle(graphOf("e4b"), { ...base, layers: 42, storage: "i2" });
    assertThrows(
      () => assertGemma4QatPle(graphOf("e2b"), { ...base, storage: "i2" }),
      Error,
      "PLE",
    );
    assertThrows(() => assertGemma4QatPle(graphOf("e2b"), base), Error, "PLE");
  });
});

Deno.test("QAT のホスト RoPE", async (t) => {
  const spec = {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  } satisfies Gemma4RopeSpec;
  await t.step("pad と回転しない次元を厳密に保存し、入力4本を返す", () => {
    const input = gemma4QatRopeInputs(spec, [0, 17, 127]);
    assertEquals(Object.keys(input).length, 4);
    for (const [key, tensor] of Object.entries(input)) {
      const width = key.includes("full_attention") ? 512 : 256;
      assertEquals(tensor.shape, [1, 3, width]);
      assertEquals(tensor.dtype, "f32");
      assertEquals(
        Array.from(tensor.data.slice(0, width)),
        Array(width).fill(key.endsWith("cos") ? 1 : 0),
      );
      for (let i = 0; i < tensor.data.length; i++) {
        assert(Number.isFinite(tensor.data[i]));
      }
    }
    for (const part of ["cos", "sin"]) {
      const t = input[`rope_full_attention_${part}`];
      for (let row = 0; row < 3; row++) {
        for (let i = 64; i < 256; i++) {
          assertEquals(t.data[row * 512 + i], part === "cos" ? 1 : 0);
          assertEquals(t.data[row * 512 + i + 256], part === "cos" ? 1 : 0);
        }
      }
    }
  });
  await t.step("角度の f32 丸めを保ち、通常 Gemma の f64 契約へ戻らない", () => {
    // Torch f32: sin(tensor(127) * (1 / tensor(10))) = 0x3e086dee。
    // theta=10000、次元32/256は逆周波数1/10。べき乗そのものは厳密なので角度の丸めだけを検査する。
    const key = "rope_sliding_attention_sin";
    assertEquals(gemma4QatRopeInputs(spec, [127])[key].data[32], 0.13323184847831726);
    assertEquals(gemma4RopeInputs(spec, [127])[key].data[32], 0.13323204219341278);
  });
  await t.step("不正な位置と f32 で表せない逆周波数を拒否する", () => {
    assertThrows(() => gemma4QatRopeInputs(spec, [-1]), Error, "非負整数");
    assertThrows(
      () =>
        gemma4QatRopeInputs({
          ...spec,
          full_attention: { theta: 1e300, headDim: 4, rotaryDim: 4 },
        }, [1]),
      Error,
      "f32 逆周波数",
    );
  });
});

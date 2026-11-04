import { readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { admitGemma4Qat, assertGemma4QatModel, assertGemma4QatPle } from "../src/gemma/qat.ts";
import { ropeInputs, type RopeSpec } from "../src/gemma/rope.ts";
import * as gemma4QatMobile from "../gemma4-qat-mobile.ts";
import type { Gemma4PleQuantizedIndex } from "../src/gemma/ple-index.ts";
import { pleFixture } from "./helpers/ple-fixture.ts";

type Graph = Parameters<typeof admitGemma4Qat>[0];

/**
 * Modifications that exercise the structural checks one branch at a time (one fault triggers exactly one throw).
 *
 * `undefined` is the normal case. Only `head-without-srq` is a modification that **passes**: it represents the shape
 * without SRQ around the shared head (the official checkpoint's lm_head has SRQ scales of 0 = uncalibrated).
 */
type Fault =
  | "no-per-layer-inputs"
  | "layers"
  | "ple-dim"
  | "hidden"
  | "two-embeddings"
  | "embedding-i8"
  | "no-head"
  | "no-projection"
  | "two-projections"
  | "linear-not-initializer"
  | "linear-shared"
  | "linear-f32"
  | "only-i4"
  | "srq-before-missing"
  | "srq-after-missing"
  | "extra-consumer"
  | "head-without-srq";

/** The weight the single f32 linear reads (the check names the projection by this initializer name). */
const PROJECTION = "model.model.per_layer_model_projection.weight";

const graphOf = (model: "e2b" | "e4b", fault?: Fault): Graph => {
  const hidden = model === "e2b" ? 1536 : 2560,
    layers = model === "e2b" ? 35 : 42;
  let raw: { op: string; ins: string[]; outs: string[] }[] = [
    { op: "embedding", ins: ["head", "input_ids"], outs: ["embedded"] },
    { op: "linear", ins: ["embedded", PROJECTION], outs: ["projected"] },
    { op: "static_quantize", ins: ["projected"], outs: ["input4"] },
    { op: "linear", ins: ["input4", "w4"], outs: ["linear4"] },
    { op: "static_quantize", ins: ["linear4"], outs: ["input8"] },
    { op: "linear", ins: ["input8", "w8"], outs: ["linear8"] },
    { op: "static_quantize", ins: ["linear8"], outs: ["input2"] },
    { op: "linear", ins: ["input2", "head"], outs: ["linear2"] },
    { op: "static_quantize", ins: ["linear2"], outs: ["logits"] },
  ];
  // After the merge, an initializer's **name is the key to its data** (docs/ir-v2.md). The check also names the f32
  // projection by this name, so the declared name is the FQN itself.
  const initializers: Record<string, Graph["initializers"][string]> = {
    head: { storage: { codec: "int2-off" } },
    [PROJECTION]: { storage: { codec: "f32" } },
    w4: { storage: { codec: "int4-sym-g", groupSize: 32, rowAxis: 0 } },
    w8: { storage: { codec: "int8-sym" } },
  };
  let pleShape: (string | number)[] = [1, "M", layers, 256];
  let inputs: Graph["inputs"] = [
    { name: "input_ids", dtype: "i32", shape: [1, "M"] },
    { name: "per_layer_inputs", dtype: "f32", shape: pleShape },
  ];
  let hiddenWidth: string | number = hidden;
  switch (fault) {
    case "no-per-layer-inputs":
      inputs = [{ name: "input_ids", dtype: "i32", shape: [1, "M"] }];
      break;
    case "layers":
      pleShape = [1, "M", 7, 256];
      inputs = [inputs[0], { name: "per_layer_inputs", dtype: "f32", shape: pleShape }];
      break;
    case "ple-dim":
      pleShape = [1, "M", layers, 128];
      inputs = [inputs[0], { name: "per_layer_inputs", dtype: "f32", shape: pleShape }];
      break;
    case "hidden":
      hiddenWidth = hidden + 1;
      break;
    case "two-embeddings":
      raw = [...raw, { op: "embedding", ins: ["head", "input_ids"], outs: ["embedded2"] }];
      break;
    case "embedding-i8":
      initializers.head = { storage: { codec: "int8-sym" } };
      break;
    case "no-head":
      // The head's linear reads a weight other than the token embedding (no shared head remains).
      raw = raw.map((node, i) => (i === 7 ? { ...node, ins: ["input2", "w4"] } : node));
      break;
    case "no-projection":
      raw = raw.filter((_, i) => i !== 1);
      break;
    case "two-projections":
      raw = [...raw, { op: "linear", ins: ["embedded", PROJECTION], outs: ["projected2"] }];
      break;
    case "linear-not-initializer":
      raw = raw.map((node, i) => (i === 3 ? { ...node, ins: ["input4", "absent"] } : node));
      break;
    case "linear-shared":
      initializers.w8 = { shared: true };
      break;
    case "linear-f32":
      initializers.w4 = { storage: { codec: "f32" } };
      break;
    case "only-i4":
      initializers.w8 = { storage: { codec: "int4-sym-g", groupSize: 32, rowAxis: 0 } };
      break;
    case "srq-before-missing":
      raw = raw.map((node, i) => (i === 2 ? { ...node, op: "reshape" } : node));
      break;
    case "srq-after-missing":
      raw = raw.map((node, i) => (i === 4 ? { ...node, op: "reshape" } : node));
      break;
    case "extra-consumer":
      raw = [...raw, { op: "reshape", ins: ["linear4"], outs: ["spare"] }];
      break;
    case "head-without-srq":
      // Removes the SRQ around the shared head (a reshape before it; the graph output directly after it).
      raw = [
        ...raw.slice(0, 7),
        { op: "reshape", ins: ["input2"], outs: ["normed"] },
        { op: "linear", ins: ["normed", "head"], outs: ["logits"] },
      ];
      break;
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding", "linear", "static_quantize"] },
    symbols: ["M", "R", "C"],
    inputs,
    outputs: ["logits", "hidden"],
    values: {
      logits: { dtype: "f32", shape: [1, "R", 262144] },
      hidden: { dtype: "f32", shape: [1, "R", hiddenWidth] },
    },
    initializers,
    states: {},
    nodes: raw.map((node) => ({
      ...node,
      attrs: node.op === "static_quantize" ? { scale: 0.125 } : {},
      states: {},
    })),
  };
};

/** A single index (a synthetic one with one block — only the dimensions are checked, so the sizes can be small). */
const pleIndexOf = (model: "e2b" | "e4b"): Gemma4PleQuantizedIndex =>
  pleFixture({
    tokens: 64,
    layers: model === "e2b" ? 35 : 42,
    dim: 256,
    embedScale: 16,
    storage: model === "e2b" ? "i4" : "i2",
  }).index;

describe("family admission of fixed QAT", () => {
  for (const model of ["e2b", "e4b"] as const) {
    it(`accepts ${model}'s fixed mix and shared head`, () => {
      const graph = graphOf(model);
      assertEquals(admitGemma4Qat(graph, model), model);
      assertEquals(admitGemma4Qat(graph), model);
      assertThrows(
        () => admitGemma4Qat(graph, model === "e2b" ? "e4b" : "e2b"),
        Error,
        "configuration differ",
      );
    });
    it(`accepts ${model}'s shared head without SRQ around it`, () => {
      // The official checkpoint's lm_head has SRQ scales of 0 for both input and output (uncalibrated = identity),
      // so the recipe does not insert an identity SRQ into the IR. The shape with SRQ (the step above) is accepted too.
      assertEquals(admitGemma4Qat(graphOf(model, "head-without-srq"), model), model);
    });
  }

  // One rejection branch = one fault. The messages are pinned too so that rewriting one condition does not turn green
  // a case that "happens to fail for another reason" (splitting the check's three co-located conditions is this table's premise).
  const rejections: readonly [Fault, string][] = [
    ["no-per-layer-inputs", "per_layer_inputs is missing"],
    ["layers", "is not E2B 35 / E4B 42"],
    ["ple-dim", "PLE dimension"],
    ["hidden", "hidden width"],
    ["two-embeddings", "not exactly one token embedding"],
    ["embedding-i8", "token embedding needs fixed INT2"],
    ["no-head", "0 heads share the token embedding"],
    ["no-projection", "per_layer_model_projection has 0 f32 linears"],
    ["two-projections", "per_layer_model_projection has 2 f32 linears"],
    ["linear-not-initializer", "is not an initializer"],
    ["linear-shared", "shared initializer"],
    ["linear-f32", "needs fixed INT2/INT4/INT8"],
    ["only-i4", "does not include both fixed INT4 and INT8"],
    ["srq-before-missing", "fixed SRQ before and after"],
    ["srq-after-missing", "fixed SRQ before and after"],
    ["extra-consumer", "fixed SRQ before and after"],
  ];
  for (const [fault, message] of rejections) {
    it(`rejects modification ${fault}`, () => {
      assertThrows(() => admitGemma4Qat(graphOf("e2b", fault)), Error, message);
    });
  }

  it("rejects an unsupported model name", () => {
    assertThrows(() => assertGemma4QatModel("12b"), Error, "unsupported model '12b'");
    assertEquals(assertGemma4QatModel("e2b"), "e2b");
    assertEquals(assertGemma4QatModel("e4b"), "e4b");
    assertThrows(() => admitGemma4Qat(graphOf("e2b"), "12b"), Error, "unsupported model '12b'");
  });

  it("compares PLE storage, layer count and dimension with the graph", () => {
    for (const model of ["e2b", "e4b"] as const) {
      assertGemma4QatPle(model, graphOf(model), pleIndexOf(model));
    }
    const index = pleIndexOf("e2b");
    assertThrows(
      () => assertGemma4QatPle("e2b", graphOf("e2b"), { ...index, storage: "i2" }),
      Error,
      "PLE storage 'i2'",
    );
    assertThrows(
      () => assertGemma4QatPle("e4b", graphOf("e4b"), { ...pleIndexOf("e4b"), storage: "i4" }),
      Error,
      "PLE storage 'i4'",
    );
    assertThrows(
      () => assertGemma4QatPle("e2b", graphOf("e2b"), { ...index, layers: 42 }),
      Error,
      "PLE layer count 42",
    );
    assertThrows(
      () => assertGemma4QatPle("e2b", graphOf("e2b"), { ...index, dim: 128 }),
      Error,
      "PLE dimension 128",
    );
  });
});

describe("fixed QAT rejects an MTP setting before fetching assets", () => {
  // The type side is blocked with `speculative?: never`, but that does not apply to JS callers, `as` casts or
  // dynamically built options. A source that does not exist confirms the runtime rejection comes **before a single
  // byte of assets is fetched**.
  it("fromPretrained", async () => {
    await assertRejects(
      () =>
        gemma4QatMobile.Pipeline.fromPretrained("does-not-exist", { speculative: true } as never),
      Error,
      "MTP is not supported for the mobile QAT line",
    );
  });
  it("fromAssets", async () => {
    await assertRejects(
      () => gemma4QatMobile.Pipeline.fromAssets({} as never, { speculative: true } as never),
      Error,
      "speculative is not accepted",
    );
  });
});

describe("host RoPE of QAT", () => {
  const spec = {
    sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
    full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
  } satisfies RopeSpec;
  it("preserves pad and the non-rotated dimensions exactly, and returns 4 inputs", () => {
    const input = gemma4QatMobile.ropeInputs(spec, [0, 17, 127]);
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
  it("keeps the f32 rounding of the angle and does not fall back to regular Gemma's f64 contract", () => {
    // Torch f32: sin(tensor(127) * (1 / tensor(10))) = 0x3e086dee.
    // theta=10000; dimension 32/256 gives an inverse frequency of 1/10. The power itself is exact, so only the rounding of the angle is checked.
    const key = "rope_sliding_attention_sin";
    assertEquals(gemma4QatMobile.ropeInputs(spec, [127])[key].data[32], 0.13323184847831726);
    assertEquals(ropeInputs(spec, [127])[key].data[32], 0.13323204219341278);
  });
  it("compares cos / sin for both layer types × 5 positions bit for bit with the golden", async () => {
    // A single-element check stays green "even if the rounding of the power step is missing" when a dimension whose
    // inverse frequency is exactly representable is chosen. The golden freezes the whole table as u32 to catch that.
    // MUST: this is **a freeze of karume's own output**, not bit agreement with upstream Torch (ADR 0097 addendum 5
    // does not guarantee agreement in every bit). Regenerate it only when the rounding contract is changed on
    // purpose.
    const golden = JSON.parse(
      await readTextFile(new URL("./fixtures/gemma4-qat-rope-golden.json", import.meta.url)),
    ) as {
      readonly positions: readonly number[];
      readonly spec: RopeSpec;
      readonly tables: Readonly<Record<string, string>>;
    };
    assertEquals(golden.spec, spec);
    const inputs = gemma4QatMobile.ropeInputs(golden.spec, golden.positions);
    assertEquals(Object.keys(inputs).sort(), Object.keys(golden.tables).sort());
    for (const [name, base64] of Object.entries(golden.tables)) {
      const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
      const expected = new Uint32Array(bytes.buffer as ArrayBuffer);
      const data = inputs[name].data as Float32Array<ArrayBuffer>;
      const actual = new Uint32Array(data.buffer, data.byteOffset, data.length);
      assertEquals(actual.length, expected.length, `${name}: element count`);
      for (let i = 0; i < expected.length; i++) {
        if (actual[i] === expected[i]) continue;
        throw new Error(
          `${name}[${i}]: 0x${actual[i].toString(16)} differs from golden 0x${expected[i].toString(
            16,
          )}`,
        );
      }
    }
  });
  it("rejects an invalid position and an inverse frequency not representable in f32", () => {
    assertThrows(() => gemma4QatMobile.ropeInputs(spec, [-1]), Error, "non-negative integer");
    assertThrows(
      () =>
        gemma4QatMobile.ropeInputs(
          {
            ...spec,
            full_attention: { theta: 1e300, headDim: 4, rotaryDim: 4 },
          },
          [1],
        ),
      Error,
      "f32 inverse frequency",
    );
  });
});

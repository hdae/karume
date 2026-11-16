// The corpus's weight generation (tests/corpus/weights.ts) and its entries' fixtures. CPU only.
//
// The generator's bytes are an invariant of the corpus's sha256 rows, so a small tensor per codec
// and role is pinned by its sha256: a moved pin says the generator changed (every corpus row then
// moves), not the runtime.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { openContainer, type OpenedContainer } from "../src/format/container/open.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { CORPUS_ENTRIES } from "./corpus/entries.ts";
import { openCorpusFixture } from "./corpus/fixture.ts";
import {
  CorpusWeightError,
  generateTensor,
  generateWeights,
  type GeneratedTensor,
  initializerRole,
  PERTURBABLE_CODECS,
  perturbOneLevel,
  type StoredEncoding,
  type TensorSpec,
} from "./corpus/weights.ts";
import { writeGraphContainer } from "./helpers/container-write.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { sha256Hex } from "./helpers/reference.ts";

const SEED = 7;

/** sha256 over the payload, then the scale. */
const digest = (tensor: GeneratedTensor): Promise<string> => {
  const scale = tensor.scale ?? new Uint8Array(0);
  const joined = new Uint8Array(tensor.bytes.byteLength + scale.byteLength);
  joined.set(tensor.bytes, 0);
  joined.set(scale, tensor.bytes.byteLength);
  return sha256Hex(joined);
};

const INT4_LINEAR: TensorSpec = {
  name: "pin.linear.weight",
  shape: [4, 64],
  role: { kind: "linear", fanIn: 64 },
  encoding: { codec: "int4-sym-g", groupSize: 32 },
};
/** The int4-sym-g group sizes of the mobile entry (the rule is the same; the scale count is not). */
const INT4_GROUPS: readonly (TensorSpec & { readonly digest: string })[] = [
  {
    name: "pin.linear.weight",
    shape: [2, 1024],
    role: { kind: "linear", fanIn: 1024 },
    encoding: { codec: "int4-sym-g", groupSize: 512 },
    digest: "d79698cd9562d7375ee72072d8ee678a42284c0c2bc9c722217b37026e491e49",
  },
  {
    name: "pin.linear.weight",
    shape: [2, 4096],
    role: { kind: "linear", fanIn: 4096 },
    encoding: { codec: "int4-sym-g", groupSize: 2048 },
    digest: "cdd3b26d0d82b486c75d45fb1873a2f983563354d11fd3492adbe7905e0c2614",
  },
  {
    name: "pin.linear.weight",
    shape: [2, 4096],
    role: { kind: "linear", fanIn: 4096 },
    encoding: { codec: "int4-sym-g", groupSize: 4096 },
    digest: "e99379efc186ece2cefb34c0677cba69f8fbb168fb78b0a90582736cc41a08dd",
  },
];
const INT2_LINEAR: TensorSpec = {
  name: "pin.linear.weight",
  shape: [4, 32],
  role: { kind: "linear", fanIn: 32 },
  encoding: { codec: "int2-off", groupSize: 32 },
};
/** A row length of 18: the payload ends inside a random word. */
const INT8_LINEAR: TensorSpec = {
  name: "pin.linear.weight",
  shape: [4, 18],
  role: { kind: "linear", fanIn: 18 },
  encoding: { codec: "int8-sym", groupSize: 18 },
};
const F32_LINEAR: TensorSpec = {
  name: "pin.linear.weight",
  shape: [4, 8],
  role: { kind: "linear", fanIn: 8 },
  encoding: { codec: "f32" },
};
const F32_NORM: TensorSpec = {
  name: "pin.norm.weight",
  shape: [16],
  role: { kind: "norm" },
  encoding: { codec: "f32" },
};
const F32_OPERAND: TensorSpec = {
  name: "pin.layer_scalar",
  shape: [3],
  role: { kind: "operand" },
  encoding: { codec: "f32" },
};

const F32_BIAS: TensorSpec = {
  name: "pin.linear.bias",
  shape: [8],
  role: { kind: "bias" },
  encoding: { codec: "f32" },
};
const F32_ADDITIVE: TensorSpec = {
  name: "pin.scale_shift_table",
  shape: [1, 2, 4],
  role: { kind: "additive" },
  encoding: { codec: "f32" },
};

/** A conv3d weight [out, in, kt, kh, kw]: the linear rule with fan-in 2 · 1 · 3 · 3. */
const INT8_CONVOLUTION: TensorSpec = {
  name: "pin.conv.weight",
  shape: [4, 2, 1, 3, 3],
  role: { kind: "convolution", fanIn: 18 },
  encoding: { codec: "int8-sym", groupSize: 18 },
};
/** A conv2d weight [out, in, kh, kw]. */
const F32_CONVOLUTION: TensorSpec = {
  name: "pin.conv.weight",
  shape: [4, 2, 3, 3],
  role: { kind: "convolution", fanIn: 18 },
  encoding: { codec: "f32" },
};
/** An embedding table no linear reads: standard deviation 1. */
const INT8_EMBEDDING: TensorSpec = {
  name: "pin.embedding.weight",
  shape: [8, 6],
  role: { kind: "embedding" },
  encoding: { codec: "int8-sym", groupSize: 6 },
};
/** A table that is normalised (the data input of a layer norm). */
const F32_NORMALIZED: TensorSpec = {
  name: "pin.rel_embeddings.weight",
  shape: [4, 8],
  role: { kind: "normalized" },
  encoding: { codec: "f32" },
};

const f32Values = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> =>
  new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

/**
 * ids → `embedding(table)` → `rms_norm(·, norm)` → `mul(·, scalar)` → `add(·, bias)` →
 * `linear(·, w)` and the tied `linear(·, table)`. `bias` is the const.
 */
const toyDocument = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add", "embedding", "linear", "mul", "rms_norm"] },
  symbols: ["T"],
  inputs: [{ name: "ids", dtype: "i32", shape: [1, "T"] }],
  outputs: ["y", "logits"],
  initializers: { table: {}, norm: {}, scalar: {}, bias: {}, w: {} },
  values: {
    table: { dtype: "f32", shape: [16, 16] },
    norm: { dtype: "f32", shape: [16] },
    scalar: { dtype: "f32", shape: [1] },
    bias: { dtype: "f32", shape: [16] },
    w: { dtype: "f32", shape: [4, 16] },
    e: { dtype: "f32", shape: [1, "T", 16] },
    n: { dtype: "f32", shape: [1, "T", 16] },
    s: { dtype: "f32", shape: [1, "T", 16] },
    b: { dtype: "f32", shape: [1, "T", 16] },
    y: { dtype: "f32", shape: [1, "T", 4] },
    logits: { dtype: "f32", shape: [1, "T", 16] },
  },
  nodes: [
    { op: "embedding", ins: ["table", "ids"], outs: ["e"], attrs: {} },
    { op: "rms_norm", ins: ["e", "norm"], outs: ["n"], attrs: {} },
    { op: "mul", ins: ["n", "scalar"], outs: ["s"], attrs: {} },
    { op: "add", ins: ["s", "bias"], outs: ["b"], attrs: {} },
    { op: "linear", ins: ["b", "w"], outs: ["y"], attrs: {} },
    { op: "linear", ins: ["b", "table"], outs: ["logits"], attrs: {} },
  ],
});

const toy = (): IrDeclaration => parseIrDeclarationValue(toyDocument());

/** `e` → `layer_norm(e, lnw, lnb)` → `linear(·, w, wb)` → `add(table, ·)` (a block without states). */
const blockDeclaration = (): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "layer_norm", "linear"] },
    symbols: ["T"],
    inputs: [{ name: "e", dtype: "f32", shape: [1, "T", 16] }],
    outputs: ["y"],
    initializers: { lnw: {}, lnb: {}, w: {}, wb: {}, table: {} },
    values: {
      lnw: { dtype: "f32", shape: [16] },
      lnb: { dtype: "f32", shape: [16] },
      w: { dtype: "f32", shape: [16, 16] },
      wb: { dtype: "f32", shape: [16] },
      table: { dtype: "f32", shape: [1, 1, 16] },
      l: { dtype: "f32", shape: [1, "T", 16] },
      h: { dtype: "f32", shape: [1, "T", 16] },
      y: { dtype: "f32", shape: [1, "T", 16] },
    },
    nodes: [
      {
        op: "layer_norm",
        ins: ["e", "lnw", "lnb"],
        outs: ["l"],
        attrs: { eps: 1e-6, normalized_shape: [16] },
      },
      { op: "linear", ins: ["l", "w", "wb"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["table", "h"], outs: ["y"], attrs: {} },
    ],
  } satisfies DeclarationJson);

/**
 * `conv1d`, `conv2d` → `group_norm`, `conv3d`, each with weight and bias, and a table normalised by
 * `layer_norm` (a relative-position table).
 */
const convolutionDeclaration = (): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["conv1d", "conv2d", "conv3d", "group_norm", "layer_norm"] },
    symbols: [],
    inputs: [
      { name: "x1", dtype: "f32", shape: [1, 2, 8] },
      { name: "x2", dtype: "f32", shape: [1, 2, 8, 8] },
      { name: "x3", dtype: "f32", shape: [1, 2, 2, 8, 8] },
    ],
    outputs: ["y1", "g", "y3", "r"],
    initializers: {
      "c1.weight": {},
      "c1.bias": {},
      "c2.weight": {},
      "c2.bias": {},
      "c3.weight": {},
      "c3.bias": {},
      "gn.weight": {},
      "gn.bias": {},
      rel: {},
      "ln.weight": {},
      "ln.bias": {},
    },
    values: {
      "c1.weight": { dtype: "f32", shape: [4, 2, 3] },
      "c1.bias": { dtype: "f32", shape: [4] },
      "c2.weight": { dtype: "f32", shape: [4, 2, 3, 3] },
      "c2.bias": { dtype: "f32", shape: [4] },
      "c3.weight": { dtype: "f32", shape: [4, 2, 1, 3, 3] },
      "c3.bias": { dtype: "f32", shape: [4] },
      "gn.weight": { dtype: "f32", shape: [4] },
      "gn.bias": { dtype: "f32", shape: [4] },
      rel: { dtype: "f32", shape: [6, 4] },
      "ln.weight": { dtype: "f32", shape: [4] },
      "ln.bias": { dtype: "f32", shape: [4] },
      y1: { dtype: "f32", shape: [1, 4, 6] },
      y2: { dtype: "f32", shape: [1, 4, 6, 6] },
      g: { dtype: "f32", shape: [1, 4, 6, 6] },
      y3: { dtype: "f32", shape: [1, 4, 2, 6, 6] },
      r: { dtype: "f32", shape: [6, 4] },
    },
    nodes: [
      {
        op: "conv1d",
        ins: ["x1", "c1.weight", "c1.bias"],
        outs: ["y1"],
        attrs: { dilation: 1, groups: 1, padding: 0, stride: 1 },
      },
      {
        op: "conv2d",
        ins: ["x2", "c2.weight", "c2.bias"],
        outs: ["y2"],
        attrs: { dilation: [1, 1], groups: 1, padding: [0, 0], stride: [1, 1] },
      },
      {
        op: "group_norm",
        ins: ["y2", "gn.weight", "gn.bias"],
        outs: ["g"],
        attrs: { eps: 1e-6, num_groups: 2 },
      },
      {
        op: "conv3d",
        ins: ["x3", "c3.weight", "c3.bias"],
        outs: ["y3"],
        attrs: { dilation: [1, 1, 1], groups: 1, padding: [0, 0, 0], stride: [1, 1, 1] },
      },
      {
        op: "layer_norm",
        ins: ["rel", "ln.weight", "ln.bias"],
        outs: ["r"],
        attrs: { eps: 1e-7, normalized_shape: [4] },
      },
    ],
  } satisfies DeclarationJson);

/** The toy with one extra node reading `initializer` at `op` slot `slot`. */
const toyWithReader = (initializer: string, op: string, slot: number): IrDeclaration => {
  const document = toyDocument();
  const ins = slot === 0 ? [initializer, "b"] : ["b", initializer];
  document.nodes.push({ op, ins, outs: ["extra"], attrs: {} });
  document.values.extra = { dtype: "f32", shape: [1, "T", 16] };
  document.outputs.push("extra");
  document.requires.ops = [...new Set([...document.requires.ops, op])].sort();
  return parseIrDeclarationValue(document);
};

/** The toy as a graph container (its `bias` in the const region), opened. */
const toyContainer = async (): Promise<OpenedContainer> => {
  const declaration = toy();
  const { bytes } = await writeGraphContainer({
    graphs: { model: declaration },
    consts: [
      {
        graph: "model",
        initializer: "bias",
        bytes: new Uint8Array(new Float32Array(16).buffer),
        encoding: { codec: "f32" },
      },
    ],
    weights: [],
    assets: [],
    provenance: { license: "none" },
  });
  return await openContainer({ kind: "bytes", bytes });
};

const TOY_STORAGE: ReadonlyMap<string, StoredEncoding> = new Map<string, StoredEncoding>([
  ["table", { codec: "int4-sym-g", groupSize: 16 }],
  ["norm", { codec: "f32" }],
  ["scalar", { codec: "f32" }],
  ["w", { codec: "int4-sym-g", groupSize: 16 }],
]);

describe("the generator's bytes are pinned per codec and role", () => {
  it("an int4-sym-g linear weight [4, 64], group 32", async () => {
    assertEquals(
      await digest(generateTensor(INT4_LINEAR, SEED)),
      "0752599cec64ec919c95437a5ac25b1f5c7d563a540010cbacec0cf2426e7ca6",
    );
  });

  for (const spec of INT4_GROUPS) {
    it(`an int4-sym-g linear weight [${spec.shape.join(", ")}], group ${spec.encoding.groupSize}`, async () => {
      assertEquals(await digest(generateTensor(spec, SEED)), spec.digest);
    });
  }

  it("an int2-off linear weight [4, 32]", async () => {
    assertEquals(
      await digest(generateTensor(INT2_LINEAR, SEED)),
      "fde1cc0a7ce5c9cb95e03d6ff782ccfde43b684368bf1197eaa828e4660e9f2e",
    );
  });

  it("an int8-sym linear weight [4, 18]", async () => {
    assertEquals(
      await digest(generateTensor(INT8_LINEAR, SEED)),
      "f0fe43cea782368eed63b4ad7aaffb0e5a02b0de8e4334a4b4b3e065f861be9c",
    );
  });

  it("an f32 linear weight [4, 8]", async () => {
    assertEquals(
      await digest(generateTensor(F32_LINEAR, SEED)),
      "4b0622c11dd784b220c47fa366b15c2cdd6f53a015c6b3ee5934f3d20e93c63a",
    );
  });

  it("an f32 norm weight [16]", async () => {
    assertEquals(
      await digest(generateTensor(F32_NORM, SEED)),
      "28dd75ea51ce6bd4fd24b36218da670c378bff43990977641317e53c50947151",
    );
  });

  it("an f32 operand [3]", async () => {
    assertEquals(
      await digest(generateTensor(F32_OPERAND, SEED)),
      "7722c980e3782d77b0547683b24d03d5934e346a852ac6224f553cbac216e9f9",
    );
  });

  it("an f32 bias [8]", async () => {
    assertEquals(
      await digest(generateTensor(F32_BIAS, SEED)),
      "eb42ea5855074bdfaca52e6dc965e55981ca850ff591c1bf51b6f59285d9effe",
    );
  });

  it("an f32 additive operand [1, 2, 4]", async () => {
    assertEquals(
      await digest(generateTensor(F32_ADDITIVE, SEED)),
      "a27e04cc10f8d84e12048e9d9c4567bf5c18db07bc218f91f7548a72f7329a18",
    );
  });

  it("an int8-sym convolution weight [4, 2, 1, 3, 3]", async () => {
    assertEquals(
      await digest(generateTensor(INT8_CONVOLUTION, SEED)),
      "e56ab7c27edcf6158fc347acba75a3ca62a1a346e8b71f5536ee1c2ec060dbb2",
    );
  });

  it("an f32 convolution weight [4, 2, 3, 3]", async () => {
    assertEquals(
      await digest(generateTensor(F32_CONVOLUTION, SEED)),
      "681b988b31b70c6202cc3e0738fa23e273549ebcb4e869bfeae861856d9c85dc",
    );
  });

  it("an int8-sym embedding table [8, 6] no linear reads", async () => {
    assertEquals(
      await digest(generateTensor(INT8_EMBEDDING, SEED)),
      "f2b480143c5f2157bf86e8cc21c302fcb091b68e3a7a7048ba5593665ca55322",
    );
  });

  it("an f32 table that is normalised [4, 8]", async () => {
    assertEquals(
      await digest(generateTensor(F32_NORMALIZED, SEED)),
      "ba66973e177c0610548c7a85dff45fa910e3854b281b9744101a414094208b77",
    );
  });
});

describe("the generator's values", () => {
  it("an int4 linear weight stores q in [-7, 7] and every group scale 1/(sqrt(fanIn) · sqrt(56/3))", () => {
    const tensor = generateTensor(INT4_LINEAR, SEED);
    for (const byte of tensor.bytes) {
      assert((byte & 0x0f) !== 0 && byte >> 4 !== 0, "a nibble holds u = 0 (q = -8)");
    }
    assertEquals(tensor.scale?.byteLength, 4 * 2 * 4);
    const expected = Math.fround(1 / (Math.sqrt(64) * Math.sqrt(56 / 3)));
    assert(f32Values(tensor.scale ?? new Uint8Array(0)).every((value) => value === expected));
  });

  it("an int2-off linear weight holds all four codes and every row scale 1/(sqrt(fanIn) · sqrt(5/4))", () => {
    const tensor = generateTensor({ ...INT2_LINEAR, shape: [64, 32] }, SEED);
    const seen = new Set<number>();
    for (const byte of tensor.bytes) {
      for (let lane = 0; lane < 4; lane += 1) seen.add((byte >> (lane * 2)) & 3);
    }
    assertEquals(
      [...seen].sort((a, b) => a - b),
      [0, 1, 2, 3],
    );
    assertEquals(tensor.scale?.byteLength, 64 * 4);
    const expected = Math.fround(1 / (Math.sqrt(32) * Math.sqrt(5 / 4)));
    assert(f32Values(tensor.scale ?? new Uint8Array(0)).every((value) => value === expected));
  });

  it("an int8-sym linear weight stores q in [-127, 127] and every row scale 1/(sqrt(fanIn) · sqrt((255² − 1)/12))", () => {
    const tensor = generateTensor({ ...INT8_LINEAR, shape: [256, 18] }, SEED);
    assertEquals(tensor.bytes.byteLength, 256 * 18);
    const levels = new Int8Array(tensor.bytes.buffer);
    assert(
      levels.every((q) => q !== -128),
      "a byte holds -128",
    );
    assert(levels.includes(-127) && levels.includes(127), "the extreme levels never occur");
    assertEquals(tensor.scale?.byteLength, 256 * 4);
    const expected = Math.fround(1 / (Math.sqrt(18) * Math.sqrt((255 * 255 - 1) / 12)));
    assert(f32Values(tensor.scale ?? new Uint8Array(0)).every((value) => value === expected));
  });

  it("an f32 linear weight lies in ±sqrt(3/fanIn)", () => {
    const bound = Math.sqrt(3 / 8);
    const values = f32Values(generateTensor(F32_LINEAR, SEED).bytes);
    assert(values.every((value) => Math.abs(value) <= bound * (1 + 1e-6)));
  });

  it("a norm weight lies in 1 ± 0.1 and an operand in [-1, 1)", () => {
    assert(f32Values(generateTensor(F32_NORM, SEED).bytes).every((v) => Math.abs(v - 1) <= 0.1));
    assert(f32Values(generateTensor(F32_OPERAND, SEED).bytes).every((v) => v >= -1 && v < 1));
  });

  it("a bias and an additive operand lie in [-0.1, 0.1)", () => {
    for (const spec of [F32_BIAS, F32_ADDITIVE]) {
      const values = f32Values(generateTensor({ ...spec, shape: [4096] }, SEED).bytes);
      assert(values.every((v) => v >= -0.1 && v < 0.1));
      assert(values.some((v) => v < -0.09) && values.some((v) => v > 0.09));
    }
  });

  it("a convolution weight has the bytes of the linear weight [out, fan-in] of the same name", () => {
    for (const spec of [INT8_CONVOLUTION, F32_CONVOLUTION]) {
      const convolution = generateTensor(spec, SEED);
      const linear = generateTensor(
        { ...spec, shape: [4, 18], role: { kind: "linear", fanIn: 18 } },
        SEED,
      );
      assertEquals(convolution.bytes, linear.bytes);
      assertEquals(convolution.scale, linear.scale);
    }
  });

  it("an embedding table stores q in [-127, 127] and every row scale 1/sqrt((255² − 1)/12) (fan-in 1)", () => {
    const tensor = generateTensor({ ...INT8_EMBEDDING, shape: [64, 6] }, SEED);
    assert(new Int8Array(tensor.bytes.buffer).every((q) => q !== -128));
    assertEquals(tensor.scale?.byteLength, 64 * 4);
    const expected = Math.fround(1 / Math.sqrt((255 * 255 - 1) / 12));
    assert(f32Values(tensor.scale ?? new Uint8Array(0)).every((value) => value === expected));
  });

  it("a table that is normalised lies in [-1, 1)", () => {
    const values = f32Values(generateTensor({ ...F32_NORMALIZED, shape: [4096] }, SEED).bytes);
    assert(values.every((v) => v >= -1 && v < 1));
    assert(values.some((v) => v < -0.9) && values.some((v) => v > 0.9));
  });

  it("the bytes follow the seed and the initializer's name", async () => {
    const base = await digest(generateTensor(F32_NORM, SEED));
    assertEquals(await digest(generateTensor(F32_NORM, SEED)), base);
    assertNotEquals(await digest(generateTensor(F32_NORM, SEED + 1)), base);
    assertNotEquals(
      await digest(generateTensor({ ...F32_NORM, name: "other.weight" }, SEED)),
      base,
    );
  });

  it("moving one level changes element 0 by one level and nothing else", () => {
    const tensor = generateTensor(INT4_LINEAR, SEED);
    const moved = perturbOneLevel(INT4_LINEAR.name, tensor);
    assertEquals(Math.abs((moved.bytes[0] & 0x0f) - (tensor.bytes[0] & 0x0f)), 1);
    assertEquals(moved.bytes[0] >> 4, tensor.bytes[0] >> 4);
    assertEquals(moved.bytes.subarray(1), tensor.bytes.subarray(1));
    assertEquals(moved.scale, tensor.scale);
  });

  it("moving one level of an int2-off weight changes element 0 by one level and nothing else", () => {
    const tensor = generateTensor(INT2_LINEAR, SEED);
    const moved = perturbOneLevel(INT2_LINEAR.name, tensor);
    assertEquals(Math.abs((moved.bytes[0] & 0x03) - (tensor.bytes[0] & 0x03)), 1);
    assertEquals(moved.bytes[0] >> 2, tensor.bytes[0] >> 2);
    assertEquals(moved.bytes.subarray(1), tensor.bytes.subarray(1));
    assertEquals(moved.scale, tensor.scale);
  });

  it("moving one level of an int8-sym weight changes element 0 by one level and nothing else", () => {
    const tensor = generateTensor(INT8_LINEAR, SEED);
    const moved = perturbOneLevel(INT8_LINEAR.name, tensor);
    const level = (bytes: Uint8Array<ArrayBuffer>): number => (bytes[0] << 24) >> 24;
    assertEquals(Math.abs(level(moved.bytes) - level(tensor.bytes)), 1);
    assertEquals(moved.bytes.subarray(1), tensor.bytes.subarray(1));
    assertEquals(moved.scale, tensor.scale);
    // At the top level it moves down, never to -128.
    const top = { ...tensor, bytes: Uint8Array.of(127, 0, 0, 0) };
    assertEquals(level(perturbOneLevel(INT8_LINEAR.name, top).bytes), 126);
  });

  it("moving one level of an f32 weight moves element 0 one unit in the last place, away from zero", () => {
    const tensor = generateTensor(F32_LINEAR, SEED);
    const moved = perturbOneLevel(F32_LINEAR.name, tensor);
    const before = f32Values(tensor.bytes);
    const after = f32Values(moved.bytes);
    assertEquals(new Uint32Array(after.buffer)[0] - new Uint32Array(before.buffer)[0], 1);
    assert(Math.abs(after[0]) > Math.abs(before[0]));
    assertEquals(after.subarray(1), before.subarray(1));
    // At the largest finite magnitude it moves down, never to an infinity.
    const largest = {
      ...tensor,
      bytes: new Uint8Array(Float32Array.of(-3.4028234663852886e38).buffer),
    };
    const down = f32Values(perturbOneLevel(F32_LINEAR.name, largest).bytes)[0];
    assert(Number.isFinite(down) && down > -3.4028234663852886e38);
  });

  it("moving one level of a codec without the rule throws", () => {
    const tensor: GeneratedTensor = { encoding: { codec: "f16" }, bytes: new Uint8Array(8) };
    assertThrows(
      () => perturbOneLevel("pin.half.weight", tensor),
      CorpusWeightError,
      "perturbing one level is defined for int4-sym-g, int2-off, int8-sym, f32, not 'f16'",
    );
  });
});

describe("an initializer's role comes from its consumers", () => {
  it("linear slot 1 is a linear weight whose fan-in is its row length", () => {
    assertEquals(initializerRole(toy(), "w"), { kind: "linear", fanIn: 16 });
  });

  it("a table read by embedding and by linear (tied) takes the linear's rule", () => {
    assertEquals(initializerRole(toy(), "table"), { kind: "linear", fanIn: 16 });
  });

  it("rms_norm slot 1 is a norm weight, mul slot 1 an operand", () => {
    assertEquals(initializerRole(toy(), "norm"), { kind: "norm" });
    assertEquals(initializerRole(toy(), "scalar"), { kind: "operand" });
  });

  it("layer_norm slot 1 is a norm weight; linear and layer_norm slot 2 are biases", () => {
    assertEquals(initializerRole(blockDeclaration(), "lnw"), { kind: "norm" });
    assertEquals(initializerRole(blockDeclaration(), "lnb"), { kind: "bias" });
    assertEquals(initializerRole(blockDeclaration(), "wb"), { kind: "bias" });
  });

  it("add (either slot) is an additive operand", () => {
    assertEquals(initializerRole(blockDeclaration(), "table"), { kind: "additive" });
    assertEquals(initializerRole(toy(), "bias"), { kind: "additive" });
  });

  it("a consumer without a rule throws with the initializer and what reads it", () => {
    assertThrows(
      () => initializerRole(toyWithReader("norm", "sub", 1), "norm"),
      CorpusWeightError,
      "initializer 'norm' is read by [rms_norm#1, sub#1]: no generation rule for [sub#1]",
    );
  });

  it("consumers that ask for different rules throw", () => {
    assertThrows(
      () => initializerRole(toyWithReader("norm", "mul", 1), "norm"),
      CorpusWeightError,
      "initializer 'norm' is read by [rms_norm#1, mul#1]: these consumers ask for different value rules",
    );
  });

  it("an embedding table no linear reads is an embedding table", () => {
    const document = toyDocument();
    document.nodes.pop();
    document.outputs = ["y"];
    delete document.values.logits;
    assertEquals(initializerRole(parseIrDeclarationValue(document), "table"), {
      kind: "embedding",
    });
  });

  it("conv1d, conv2d and conv3d slot 1 are convolution weights whose fan-in is every dimension after the first", () => {
    assertEquals(initializerRole(convolutionDeclaration(), "c1.weight"), {
      kind: "convolution",
      fanIn: 2 * 3,
    });
    assertEquals(initializerRole(convolutionDeclaration(), "c2.weight"), {
      kind: "convolution",
      fanIn: 2 * 3 * 3,
    });
    assertEquals(initializerRole(convolutionDeclaration(), "c3.weight"), {
      kind: "convolution",
      fanIn: 2 * 1 * 3 * 3,
    });
  });

  it("convolution and group_norm slot 2 are biases; group_norm slot 1 is a norm weight", () => {
    for (const name of ["c1.bias", "c2.bias", "c3.bias", "gn.bias"]) {
      assertEquals(initializerRole(convolutionDeclaration(), name), { kind: "bias" });
    }
    assertEquals(initializerRole(convolutionDeclaration(), "gn.weight"), { kind: "norm" });
  });

  it("an initializer read as the data input of layer_norm is a table that is normalised", () => {
    assertEquals(initializerRole(convolutionDeclaration(), "rel"), { kind: "normalized" });
  });
});

describe("generation fails loudly outside its rules", () => {
  it("a role stored in a codec without a rule", () => {
    assertThrows(
      () => generateTensor({ ...F32_NORM, encoding: { codec: "int4-sym-g", groupSize: 16 } }, SEED),
      CorpusWeightError,
      "initializer 'pin.norm.weight': no generation rule for a norm weight stored in codec 'int4-sym-g'",
    );
    assertThrows(
      () => generateTensor({ ...INT4_LINEAR, encoding: { codec: "f16" } }, SEED),
      CorpusWeightError,
      "no generation rule for a linear weight stored in codec 'f16'",
    );
  });

  it("a model-side weight missing from the storage table", async () => {
    const container = await toyContainer();
    const storage = new Map(TOY_STORAGE);
    storage.delete("w");
    assertThrows(
      () => generateWeights(container.graphs.model, storage, SEED),
      CorpusWeightError,
      "initializer 'w' is a model-side weight without a stored encoding",
    );
  });

  it("a const named in the storage table", async () => {
    const container = await toyContainer();
    const storage = new Map(TOY_STORAGE).set("bias", { codec: "f32" });
    assertThrows(
      () => generateWeights(container.graphs.model, storage, SEED),
      CorpusWeightError,
      "initializer 'bias' is supplied by the const region",
    );
  });

  it("a storage entry the graph does not declare", async () => {
    const container = await toyContainer();
    const storage = new Map(TOY_STORAGE).set("ghost", { codec: "f32" });
    assertThrows(
      () => generateWeights(container.graphs.model, storage, SEED),
      CorpusWeightError,
      "the storage table names [ghost]",
    );
  });

  it("generates every model-side weight and never a const", async () => {
    const container = await toyContainer();
    const weights = generateWeights(container.graphs.model, TOY_STORAGE, SEED);
    assertEquals([...weights.keys()].sort(), ["norm", "scalar", "table", "w"]);
  });
});

describe("the entries' fixtures", () => {
  for (const entry of CORPUS_ENTRIES) {
    it(`${entry.name}: a graph container of at most 2 MiB whose storage table names exactly its model-side weights`, async () => {
      const fixture = await openCorpusFixture(entry);
      assert(fixture.graphBytes <= 2 * 1024 * 1024, `${fixture.graphBytes} bytes`);
      const { declaration, supplies } = fixture.bound;
      const modelSide = Object.keys(declaration.initializers).filter(
        (name) => !declaration.initializers[name].shared && supplies.get(name)?.origin !== "const",
      );
      assertEquals([...fixture.storage.keys()].sort(), modelSide.sort());
      for (const name of modelSide) initializerRole(declaration, name);
      const codec = fixture.storage.get(entry.perturbed)?.codec;
      assert(
        codec !== undefined && PERTURBABLE_CODECS.includes(codec),
        `the perturbed initializer '${entry.perturbed}' is stored in ${codec ?? "no codec"}, which perturbOneLevel does not cover`,
      );
    });
  }
});

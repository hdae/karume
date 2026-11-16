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

  it("moving one level of a codec without the rule throws", () => {
    assertThrows(
      () => perturbOneLevel(INT8_LINEAR.name, generateTensor(INT8_LINEAR, SEED)),
      CorpusWeightError,
      "perturbing one level is defined for int4-sym-g and int2-off, not 'int8-sym'",
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

  it("an embedding table no linear reads has no rule", () => {
    const document = toyDocument();
    document.nodes.pop();
    document.outputs = ["y"];
    delete document.values.logits;
    assertThrows(
      () => initializerRole(parseIrDeclarationValue(document), "table"),
      CorpusWeightError,
      "initializer 'table' is read only as an embedding table",
    );
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

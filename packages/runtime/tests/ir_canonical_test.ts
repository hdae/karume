/**
 * The IR v2 declaration parser and canonical serialization (docs/ir-v2.md "Canonical
 * serialization") — the same graph becomes the same bytes.
 */

import { readTextFileSync } from "@karume/test-support/fs";
import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { sortedObject } from "../src/format/container/json.ts";
import {
  canonicalIrDocument,
  canonicalIrJson,
  IrError,
  parseIrDeclaration,
} from "../src/format/ir.ts";
import { writeGraphContainer } from "./helpers/container-write.ts";

type Json = Record<string, unknown>;

const baseGraph = (): Json => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { "enc.w": {}, b: {} },
  values: {
    "enc.w": { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "enc.w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

const canonical = (graph: Json): string =>
  canonicalIrJson(parseIrDeclaration(JSON.stringify(graph)));

describe("parseIrDeclaration", () => {
  it("accepts the minimal v2 graph, with initializers only {} or {shared:true}", () => {
    const graph = parseIrDeclaration(JSON.stringify(baseGraph()));
    assertEquals(graph.version, 2);
    assertEquals(graph.initializers, { "enc.w": { shared: false }, b: { shared: false } });
    const shared = baseGraph();
    (shared.initializers as Json)["b"] = { shared: true };
    assertEquals(parseIrDeclaration(JSON.stringify(shared)).initializers["b"], { shared: true });
  });

  it("rejects the v1 form (version 1, tensor / storage), shared:false and unknown keys", () => {
    const v1 = { ...baseGraph(), version: 1 };
    assertThrows(() => parseIrDeclaration(JSON.stringify(v1)), IrError, "version");
    const withStorage = baseGraph();
    (withStorage.initializers as Json)["b"] = { tensor: "b", storage: { dtype: "f32" } };
    assertThrows(() => parseIrDeclaration(JSON.stringify(withStorage)), IrError, "unknown key");
    const sharedFalse = baseGraph();
    (sharedFalse.initializers as Json)["b"] = { shared: false };
    assertThrows(
      () => parseIrDeclaration(JSON.stringify(sharedFalse)),
      IrError,
      "cannot be written",
    );
  });
});

describe("canonicalIrJson", () => {
  it("keys in schema order, name-keyed maps and sets in code point order, empty states not written", () => {
    const graph = baseGraph();
    // Given deliberately in reverse and scrambled order.
    const shuffled: Json = {
      nodes: graph.nodes,
      values: {
        y: (graph.values as Json)["y"],
        b: (graph.values as Json)["b"],
        h: (graph.values as Json)["h"],
        "enc.w": (graph.values as Json)["enc.w"],
      },
      initializers: { "enc.w": {}, b: {} },
      outputs: ["y"],
      inputs: graph.inputs,
      symbols: ["T"],
      requires: { ops: ["matmul", "add"] },
      version: 2,
      format: "karume-ir",
    };
    const text = canonical(shuffled);
    assertEquals(
      text,
      '{"format":"karume-ir","version":2,"requires":{"ops":["add","matmul"]},"symbols":["T"],' +
        '"inputs":[{"name":"x","dtype":"f32","shape":["T",4]}],"outputs":["y"],' +
        '"initializers":{"b":{},"enc.w":{}},' +
        '"values":{"b":{"dtype":"f32","shape":[3]},"enc.w":{"dtype":"f32","shape":[4,3]},"h":{"dtype":"f32","shape":["T",3]},"y":{"dtype":"f32","shape":["T",3]}},' +
        '"nodes":[{"op":"matmul","ins":["x","enc.w"],"outs":["h"],"attrs":{}},{"op":"add","ins":["h","b"],"outs":["y"],"attrs":{}}]}',
    );
    assertEquals(text.includes('"states"'), false);
  });

  it("numbers are spelled by ECMAScript Number::toString (integers without a decimal point, 1e-6 not as an exponent, 1e-7 and 1e21 as exponents)", () => {
    const graph = baseGraph();
    (graph.nodes as Json[])[1].attrs = {
      eps: 0.000001,
      tiny: 1e-7,
      theta: 10000.0,
      big: 1e21,
      bigInt: 1e20,
      negZero: -0,
      half: 0.5,
      list: [1.0, 2.5, 1e-12],
    };
    const text = canonical(graph);
    assertEquals(
      text.slice(text.indexOf('"attrs":{"big"')),
      '"attrs":{"big":1e+21,"bigInt":100000000000000000000,"eps":0.000001,"half":0.5,"list":[1,2.5,1e-12],"negZero":0,"theta":10000,"tiny":1e-7}}]}',
    );
  });

  it("attrs keys are in code point order recursively (array order is kept)", () => {
    const graph = baseGraph();
    (graph.nodes as Json[])[0].attrs = { z: 1, a: { y: [3, 1, 2], b: true }, é: "e", B: 0 };
    const text = canonical(graph);
    // ASCII uppercase < lowercase < non-ASCII (code point order).
    assertEquals(text.includes('"attrs":{"B":0,"a":{"b":true,"y":[3,1,2]},"z":1,"é":"e"}'), true);
  });

  it('names made only of digits are in code point order too ("10" < "2"), giving the same bytes as the Python writer', () => {
    // JS objects enumerate integer-like keys first in ascending order, so assigning them in sorted
    // order and passing the object to JSON.stringify brings back the order "0","2","10". The
    // expected value is a fixture shared with the Python writer (test_container.py in
    // tools/exporter).
    const expected = readTextFileSync(
      new URL("./fixtures/ir/canonical-name-order.txt", import.meta.url),
    );
    assertEquals(canonicalIrJson(parseIrDeclaration(expected)), expected);
  });

  it("parse → canonical → parse → canonical is a fixed point (idempotent)", () => {
    const first = canonical(baseGraph());
    assertEquals(canonical(JSON.parse(first)), first);
  });

  it("spelling a canonical document with JSON.stringify fails (no silently different JSON)", () => {
    const document = canonicalIrDocument(parseIrDeclaration(JSON.stringify(baseGraph())));
    assertThrows(() => JSON.stringify(document), ContainerFormatError, "canonicalJsonText");
  });

  it("state slots, the states field of nodes and external appear only when set", () => {
    const graph = baseGraph();
    graph.symbols = ["T", "C"];
    graph.states = {
      kv: { dtype: "f32", shape: [1, 2, "C", 4], external: true },
      aa: { dtype: "f32", shape: [1, 2, "C", 4] },
    };
    (graph.nodes as Json[])[0].states = { v: "kv", k: "aa" };
    const text = canonical(graph);
    assertEquals(
      text.includes(
        '"states":{"aa":{"dtype":"f32","shape":[1,2,"C",4]},"kv":{"dtype":"f32","shape":[1,2,"C",4],"external":true}}',
      ),
      true,
    );
    assertEquals(text.includes('"attrs":{},"states":{"k":"aa","v":"kv"}}'), true);
  });
});

describe('canonical serialization of the container descriptor (docs/container-v1.md §0, ir-v2.md "Canonical serialization")', () => {
  it("const bindings are ordered by (graph, initializer) in code point order (not UTF-16 unit order)", async () => {
    // In code point order U+FFFF comes first; in UTF-16 unit order (the surrogate 0xD83D is smaller
    // than 0xFFFF) U+1F600 comes first — the smallest pair on which the two orders disagree.
    const emoji = "\u{1F600}";
    const last = "￿";
    const graph = parseIrDeclaration(
      JSON.stringify({
        format: "karume-ir",
        version: 2,
        requires: { ops: ["add"] },
        symbols: [],
        inputs: [{ name: "x", dtype: "f32", shape: [4] }],
        outputs: ["y"],
        initializers: { [emoji]: {}, [last]: {} },
        values: {
          [emoji]: { dtype: "f32", shape: [4] },
          [last]: { dtype: "f32", shape: [4] },
          h: { dtype: "f32", shape: [4] },
          y: { dtype: "f32", shape: [4] },
        },
        nodes: [
          { op: "add", ins: ["x", emoji], outs: ["h"], attrs: {} },
          { op: "add", ins: ["h", last], outs: ["y"], attrs: {} },
        ],
      }),
    );
    const constOf = (initializer: string) => ({
      graph: "g",
      initializer,
      bytes: new Uint8Array(new ArrayBuffer(16)),
      encoding: { codec: "f32" as const },
    });
    const { descriptorBytes } = await writeGraphContainer({
      graphs: { g: graph },
      consts: [constOf(emoji), constOf(last)],
      weights: [],
      assets: [],
      provenance: { license: "apache-2.0" },
    });
    const text = new TextDecoder().decode(descriptorBytes);
    const constants = text.slice(text.indexOf('"constants"'));
    assertEquals(
      constants.indexOf(`"initializer":"${last}"`) < constants.indexOf(`"initializer":"${emoji}"`),
      true,
    );
  });

  it("the key __proto__ is rejected on the writer side too (no writing a name the reader rejects)", () => {
    assertThrows(() => sortedObject({ ["__proto__"]: 1 }), ContainerFormatError, "__proto__");
  });
});

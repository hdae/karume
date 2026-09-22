/**
 * IR v2 の宣言パーサと正準直列化（docs/ir-v2.md「正準直列化」）— 同じグラフは同じバイト列になる。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { canonicalIrJson, IrError, parseIrDeclaration } from "../src/format/ir.ts";

type Json = Record<string, unknown>;

const baseGraph = (): Json => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: { "enc.w": {}, "b": {} },
  values: {
    "enc.w": { dtype: "f32", shape: [4, 3] },
    "b": { dtype: "f32", shape: [3] },
    "h": { dtype: "f32", shape: ["T", 3] },
    "y": { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "enc.w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

const canonical = (graph: Json): string =>
  canonicalIrJson(parseIrDeclaration(JSON.stringify(graph)));

describe("parseIrDeclaration", () => {
  it("v2 の最小グラフを受理し、initializer は {} か {shared:true} だけ", () => {
    const graph = parseIrDeclaration(JSON.stringify(baseGraph()));
    assertEquals(graph.version, 2);
    assertEquals(graph.initializers, { "enc.w": { shared: false }, "b": { shared: false } });
    const shared = baseGraph();
    (shared.initializers as Json)["b"] = { shared: true };
    assertEquals(parseIrDeclaration(JSON.stringify(shared)).initializers["b"], { shared: true });
  });

  it("v1 の形（version 1・tensor / storage）と shared:false・未知キーを拒否する", () => {
    const v1 = { ...baseGraph(), version: 1 };
    assertThrows(() => parseIrDeclaration(JSON.stringify(v1)), IrError, "version");
    const withStorage = baseGraph();
    (withStorage.initializers as Json)["b"] = { tensor: "b", storage: { dtype: "f32" } };
    assertThrows(() => parseIrDeclaration(JSON.stringify(withStorage)), IrError, "未知のキー");
    const sharedFalse = baseGraph();
    (sharedFalse.initializers as Json)["b"] = { shared: false };
    assertThrows(() => parseIrDeclaration(JSON.stringify(sharedFalse)), IrError, "書けない");
  });
});

describe("canonicalIrJson", () => {
  it("キーはスキーマ順、名前キーの map と集合は code point 順、空の states は書かない", () => {
    const graph = baseGraph();
    // わざと逆順・散らした順で与える。
    const shuffled: Json = {
      nodes: graph.nodes,
      values: {
        "y": (graph.values as Json)["y"],
        "b": (graph.values as Json)["b"],
        "h": (graph.values as Json)["h"],
        "enc.w": (graph.values as Json)["enc.w"],
      },
      initializers: { "enc.w": {}, "b": {} },
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

  it("数値の綴りは ECMAScript Number::toString（整数は小数点なし・1e-6 は指数にしない・1e-7 と 1e21 は指数）", () => {
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

  it("attrs のキーは再帰的に code point 順（配列の順序は保つ）", () => {
    const graph = baseGraph();
    (graph.nodes as Json[])[0].attrs = { z: 1, a: { y: [3, 1, 2], b: true }, "é": "e", "B": 0 };
    const text = canonical(graph);
    // ASCII 大文字 < 小文字 < 非 ASCII（code point 順）。
    assertEquals(text.includes('"attrs":{"B":0,"a":{"b":true,"y":[3,1,2]},"z":1,"é":"e"}'), true);
  });

  it("parse → canonical → parse → canonical は不動点（冪等）", () => {
    const first = canonical(baseGraph());
    assertEquals(canonical(JSON.parse(first)), first);
  });

  it("states スロットとノードの states 欄・external は書くときだけ現れる", () => {
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

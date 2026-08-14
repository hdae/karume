import { assertEquals, assertThrows } from "@std/assert";
import { IrError, type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import { baseGraph, type GraphJson } from "./helpers/format.ts";

const parseMutated = (mutate: (graph: GraphJson) => void): IrGraph => {
  const graph = baseGraph();
  mutate(graph);
  return parseIrGraph(JSON.stringify(graph));
};

/** 検証規則ごとに「正常系から 1 点だけ壊す」ことで、規則が実際に効いていることを見る。 */
const assertRejects = (hint: string, mutate: (graph: GraphJson) => void): void => {
  assertThrows(() => parseMutated(mutate), IrError, undefined, hint);
};

Deno.test("parseIrGraph: 最小の正常系グラフを受理する", () => {
  const graph = parseIrGraph(JSON.stringify(baseGraph()));
  assertEquals(graph.format, "karume-ir");
  assertEquals(graph.version, 1);
  assertEquals(graph.symbols, ["T"]);
  assertEquals(graph.inputs, [{ name: "x", dtype: "f32", shape: ["T", 4] }]);
  assertEquals(graph.outputs, ["y"]);
  assertEquals(graph.requires.ops, ["matmul", "add"]);
  assertEquals(graph.nodes.map((node) => node.op), ["matmul", "add"]);
  assertEquals(graph.initializers["w"], { tensor: "enc.w", storage: { dtype: "f32" } });
  assertEquals(graph.values["h"], { dtype: "f32", shape: ["T", 3] });
});

Deno.test("parseIrGraph: format / version の固定値", () => {
  assertRejects("format 不一致", (g) => {
    g.format = "karume";
  });
  assertRejects("version 不一致", (g) => {
    g.version = 2;
  });
});

Deno.test("parseIrGraph: トップレベルのキー集合", () => {
  assertThrows(
    () => parseIrGraph(JSON.stringify({ ...baseGraph(), extra: 1 })),
    IrError,
    "未知のキー",
  );
  const { requires: _dropped, ...withoutRequires } = baseGraph();
  assertThrows(() => parseIrGraph(JSON.stringify(withoutRequires)), IrError, "必須キー");
  assertThrows(
    () => parseIrGraph(JSON.stringify({ ...baseGraph(), nodes: {} })),
    IrError,
    "配列でない",
  );
});

Deno.test("parseIrGraph: requires.ops は使用 op 集合と一致する", () => {
  assertRejects("宣言漏れ", (g) => {
    g.requires.ops = ["matmul"];
  });
  assertRejects("余剰", (g) => {
    g.requires.ops = ["matmul", "add", "gelu"];
  });
  assertRejects("重複", (g) => {
    g.requires.ops = ["matmul", "add", "add"];
  });
});

Deno.test("parseIrGraph: SSA 単一代入", () => {
  assertRejects("ノード出力の二重定義", (g) => {
    g.nodes[1].outs = ["h"];
  });
  assertRejects("入力名の重複", (g) => {
    g.inputs.push({ name: "x", dtype: "f32", shape: ["T", 4] });
  });
  assertRejects("initializer と入力の衝突", (g) => {
    g.initializers["x"] = { tensor: "enc.b", storage: { dtype: "f32" } };
    g.values["x"] = { dtype: "f32", shape: [3] };
  });
});

Deno.test("parseIrGraph: トポロジカル順（前方参照拒否）", () => {
  assertRejects("ノード順の逆転", (g) => {
    g.nodes.reverse();
  });
  assertRejects("未定義の ins", (g) => {
    g.nodes[0].ins = ["x", "unknown"];
  });
  assertRejects("未定義の outputs", (g) => {
    g.outputs = ["z"];
  });
  assertRejects("outputs の重複", (g) => {
    g.outputs = ["y", "y"];
  });
});

Deno.test("parseIrGraph: 宣言の完全性", () => {
  assertRejects("ノード出力が values に無い", (g) => {
    delete g.values["y"];
  });
  assertRejects("initializer が values に無い", (g) => {
    delete g.values["w"];
  });
  assertRejects("入力の二重宣言", (g) => {
    g.values["x"] = { dtype: "f32", shape: ["T", 4] };
  });
  assertRejects("孤立した values 宣言", (g) => {
    g.values["ghost"] = { dtype: "f32", shape: [1] };
  });
  assertRejects("initializer の記号次元", (g) => {
    g.values["w"] = { dtype: "f32", shape: ["T", 3] };
  });
  assertRejects("initializer の意味論 dtype が bool", (g) => {
    g.values["w"] = { dtype: "bool", shape: [4, 3] };
  });
  // 意味論と格納の交差（i32 は生の int32 1 通りのみ — ADR 0010）
  assertRejects("意味論 i32 を f32 格納で宣言", (g) => {
    g.values["w"] = { dtype: "i32", shape: [4, 3] };
  });
  assertRejects("意味論 f32 を i32 格納で宣言", (g) => {
    g.initializers["w"].storage = { dtype: "i32" };
  });
});

// ADR 0010: 記号依存定数の焼き込み先。意味論 i32 × 格納 i32 の 1 組だけが valid。
Deno.test("parseIrGraph: 意味論 i32 の initializer は生の int32 格納と組む", () => {
  const graph = parseMutated((g) => {
    g.values["w"] = { dtype: "i32", shape: [4, 3] };
    g.initializers["w"].storage = { dtype: "i32" };
    // add(matmul(x,w)) の dtype 契約は parse 層では見ない（契約表の層）ので宣言だけ差し替える
  });
  assertEquals(graph.values["w"].dtype, "i32");
  assertEquals(graph.initializers["w"].storage.dtype, "i32");
});

// 格納 dtype は意味論 f32 の符号化なので、f16/bf16/i8 格納は宣言として valid のまま
// （実行可否は assertRuntimeSupport の層）。パーサへ規則を寄せた際に仕様を狭めていないことの固定。
Deno.test("parseIrGraph: 意味論 f32 の initializer は非 f32 格納でも受理する", () => {
  for (const dtype of ["f16", "bf16", "i8"] as const) {
    const graph = parseMutated((g) => {
      // i8 は scale の宣言が必須（ADR 0019）— 他の格納 dtype には付けられない
      g.initializers["w"].storage = dtype === "i8" ? { dtype, scale: "enc.w.scale" } : { dtype };
    });
    assertEquals(graph.initializers["w"].storage.dtype, dtype);
  }
});

Deno.test("parseIrGraph: dtype 語彙", () => {
  assertRejects("意味論 dtype に f16", (g) => {
    g.values["h"].dtype = "f16";
  });
  assertRejects("意味論 dtype に i64", (g) => {
    g.inputs[0].dtype = "i64";
  });
  assertRejects("格納 dtype に i4", (g) => {
    g.initializers["w"].storage = { dtype: "i4" };
  });
});

Deno.test("parseIrGraph: 量子化格納は宣言として受理する（実行可否は別の層）", () => {
  const graph = parseMutated((g) => {
    g.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale", group_size: 32 };
  });
  assertEquals(graph.initializers["w"].storage, {
    dtype: "i8",
    scale: "enc.w.scale",
    groupSize: 32,
  });
});

Deno.test("parseIrGraph: storage 記述子の整合", () => {
  assertRejects("非量子化 dtype の scale", (g) => {
    g.initializers["w"].storage = { dtype: "f32", scale: "enc.w.scale" };
  });
  assertRejects("非量子化 dtype の group_size", (g) => {
    g.initializers["b"].storage = { dtype: "f16", group_size: 32 };
  });
  assertRejects("storage の未知キー", (g) => {
    g.initializers["w"].storage = { dtype: "f32", zero_point: "z" };
  });
  assertRejects("group_size が 0", (g) => {
    g.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale", group_size: 0 };
  });
  // MUST: i8 の scale は必須（ADR 0019）。既定 1.0 で補完すると、scale を書き忘れたモデルが
  // 「量子化前の 1/127 倍の重み」で静かに走る。
  assertRejects("i8 の scale 欠落", (g) => {
    g.initializers["w"].storage = { dtype: "i8" };
  });
});

Deno.test("parseIrGraph: シンボルの宣言と束縛可能性", () => {
  assertRejects("未宣言シンボルの使用", (g) => {
    g.inputs[0].shape = ["S", 4];
  });
  assertRejects("入力 shape の次元位置に一度も現れない", (g) => {
    g.inputs[0].shape = [8, 4];
  });
  assertRejects("symbols の重複", (g) => {
    g.symbols = ["T", "T"];
  });
  assertRejects("シンボル名が不正", (g) => {
    g.symbols = ["T", "2T"];
  });
  // 派生形だけの入力は**受理する**（実寸から解が一意 — ADR 0057）。母音検出 CRNN は
  // 先頭 conv の stride 2 のせいで `2T` でしか長さ軸を宣言できない。
  const derived = parseMutated((g) => {
    g.inputs[0].shape = ["2T", 4];
  });
  assertEquals(derived.inputs[0].shape, ["2T", 4]);
});

Deno.test("parseIrGraph: shape 要素", () => {
  assertRejects("非正準な次元式", (g) => {
    g.inputs[0].shape = ["1T", 4];
  });
  assertRejects("負の次元", (g) => {
    g.values["b"].shape = [-3];
  });
  assertRejects("小数の次元", (g) => {
    g.values["b"].shape = [1.5];
  });
  const boolDim = JSON.stringify(baseGraph()).replace('"shape":[3]', '"shape":[true]');
  assertThrows(() => parseIrGraph(boolDim), IrError, "数値でも文字列でもない");
});

Deno.test("parseIrGraph: ノードの構造", () => {
  assertRejects("outs が空", (g) => {
    g.nodes[0].outs = [];
  });
  assertRejects("attrs がオブジェクトでない", (g) => {
    g.nodes[0].attrs = 5;
  });
  assertRejects("op が空文字列", (g) => {
    g.nodes[0].op = "";
  });
});

// "__proto__" を名前に持つグラフ。フィクスチャは生の JSON 文字列から組む MUST — JS の
// オブジェクトリテラルに `"__proto__":` を書くと own key ではなく [[Prototype]] 指定になり、
// JSON.parse が作る own property と別物になってテストが検査対象を外す。
const protoNameGraph = `{
  "format": "karume-ir",
  "version": 1,
  "requires": { "ops": ["matmul"] },
  "symbols": ["T"],
  "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
  "outputs": ["y"],
  "initializers": { "__proto__": { "tensor": "enc.w", "storage": { "dtype": "f32" } } },
  "values": {
    "__proto__": { "dtype": "f32", "shape": [4, 3] },
    "y": { "dtype": "f32", "shape": ["T", 3] }
  },
  "nodes": [{ "op": "matmul", "ins": ["x", "__proto__"], "outs": ["y"], "attrs": {} }]
}`;

Deno.test("parseIrGraph: '__proto__' という名前の宣言を own property として保全する", () => {
  const graph = parseIrGraph(protoNameGraph);

  // 実行系によっては `o["__proto__"] = v` でも own property が作られるため、hasOwn だけでは
  // 素の `{}` への退行を検出できない。器が null プロトタイプであること（＝ 受理集合に
  // エンジン差を持ち込まないこと）を併せて固定する。
  assertEquals(Object.getPrototypeOf(graph.values), null);
  assertEquals(Object.getPrototypeOf(graph.initializers), null);
  assertEquals(Object.hasOwn(graph.values, "__proto__"), true);
  assertEquals(Object.hasOwn(graph.initializers, "__proto__"), true);
  assertEquals(graph.values["__proto__"], { dtype: "f32", shape: [4, 3] });
  assertEquals(graph.initializers["__proto__"], { tensor: "enc.w", storage: { dtype: "f32" } });
  // 宣言検査を素通りしたのではなく、ノードからの参照込みで通っていること。
  assertEquals(graph.nodes[0].ins, ["x", "__proto__"]);
});

Deno.test("parseIrGraph: '__proto__' という名前でも孤立宣言は黙って消えず拒否される", () => {
  const orphan = `{
    "format": "karume-ir",
    "version": 1,
    "requires": { "ops": ["matmul"] },
    "symbols": ["T"],
    "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
    "outputs": ["y"],
    "initializers": { "w": { "tensor": "enc.w", "storage": { "dtype": "f32" } } },
    "values": {
      "w": { "dtype": "f32", "shape": [4, 3] },
      "y": { "dtype": "f32", "shape": ["T", 3] },
      "__proto__": { "dtype": "f32", "shape": [1] }
    },
    "nodes": [{ "op": "matmul", "ins": ["x", "w"], "outs": ["y"], "attrs": {} }]
  }`;

  assertThrows(() => parseIrGraph(orphan), IrError, "どのノードでも定義されない宣言");
});

Deno.test("parseIrGraph: JSON の受理集合", () => {
  assertThrows(() => parseIrGraph("{"), IrError, "解析できない");
  assertThrows(
    () => parseIrGraph('{"format":"karume-ir","version":NaN}'),
    IrError,
    "解析できない",
  );
  // 1e999 は構文としては有効だが Infinity へ丸まる — 受理集合をブラウザ JSON.parse に
  // 揃える契約のため拒否する。
  const infinite = JSON.stringify(baseGraph()).replace('"shape":[4,3]', '"shape":[4,1e999]');
  assertThrows(() => parseIrGraph(infinite), IrError, "有限");
});

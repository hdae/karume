import { assertEquals, assertThrows } from "@std/assert";
import { type IrDeclaration, IrError, parseIrDeclaration } from "../src/format/ir.ts";
import {
  baseDeclaration,
  type DeclarationJson,
  withStateReaders,
} from "./helpers/model-fixture.ts";

const parseMutated = (mutate: (graph: DeclarationJson) => void): IrDeclaration => {
  const graph = baseDeclaration();
  mutate(graph);
  return parseIrDeclaration(JSON.stringify(graph));
};

/**
 * 検証規則ごとに「正常系から 1 点だけ壊す」ことで、規則が実際に効いていることを見る。
 *
 * MUST: 期待文言（その規則に固有の診断）まで見る — `IrError` の型だけでは、狙った規則が死んでも
 * 同じ入力が別の規則で落ちれば緑のままになる。
 */
const assertRejects = (
  hint: string,
  expected: string,
  mutate: (graph: DeclarationJson) => void,
): void => {
  assertThrows(() => parseMutated(mutate), IrError, expected, hint);
};

// IR v2 の宣言は**計算契約**だけを持つ（格納は容器の束縛表 — docs/ir-v2.md）。initializer 名が
// そのまま実体の鍵なので、v1 にあった「実体のテンソルキーへの改名」はもう起きない。
Deno.test("parseIrDeclaration: 最小の正常系グラフを受理する", () => {
  const graph = parseIrDeclaration(JSON.stringify(baseDeclaration()));
  assertEquals(graph.format, "karume-ir");
  assertEquals(graph.version, 2);
  assertEquals(graph.symbols, ["T"]);
  assertEquals(graph.inputs, [{ name: "x", dtype: "f32", shape: ["T", 4] }]);
  assertEquals(graph.outputs, ["y"]);
  assertEquals(graph.requires.ops, ["add", "matmul"]);
  assertEquals(graph.nodes.map((node) => node.op), ["matmul", "add"]);
  assertEquals(Object.keys(graph.initializers), ["w", "b"]);
  assertEquals(graph.initializers["w"], { shared: false });
  assertEquals(graph.values["w"], { dtype: "f32", shape: [4, 3] });
  assertEquals(graph.nodes[0].ins, ["x", "w"]);
  assertEquals(graph.values["h"], { dtype: "f32", shape: ["T", 3] });
});

Deno.test("parseIrDeclaration: format / version の固定値", () => {
  assertRejects("format 不一致", "graph.format が 'karume-ir' でない", (g) => {
    g.format = "karume";
  });
  assertRejects("version 不一致", "graph.version が 2 でない", (g) => {
    g.version = 1;
  });
});

Deno.test("parseIrDeclaration: トップレベルのキー集合", () => {
  assertThrows(
    () => parseIrDeclaration(JSON.stringify({ ...baseDeclaration(), extra: 1 })),
    IrError,
    "未知のキー",
  );
  const { requires: _dropped, ...withoutRequires } = baseDeclaration();
  assertThrows(() => parseIrDeclaration(JSON.stringify(withoutRequires)), IrError, "必須キー");
  assertThrows(
    () => parseIrDeclaration(JSON.stringify({ ...baseDeclaration(), nodes: {} })),
    IrError,
    "配列でない",
  );
});

Deno.test("parseIrDeclaration: requires.ops は使用 op 集合と一致する", () => {
  assertRejects("宣言漏れ", "宣言漏れ [add]", (g) => {
    g.requires.ops = ["matmul"];
  });
  assertRejects("余剰", "余剰 [gelu]", (g) => {
    g.requires.ops = ["matmul", "add", "gelu"];
  });
  assertRejects("重複", "graph.requires.ops: 'add' が重複している", (g) => {
    g.requires.ops = ["matmul", "add", "add"];
  });
});

Deno.test("parseIrDeclaration: SSA 単一代入", () => {
  assertRejects(
    "ノード出力の二重定義",
    "graph.nodes[1] (add): 値 'h' が二重に定義されている",
    (g) => {
      g.nodes[1].outs = ["h"];
    },
  );
  assertRejects("入力名の重複", "graph.inputs: 値 'x' が二重に定義されている", (g) => {
    g.inputs.push({ name: "x", dtype: "f32", shape: ["T", 4] });
  });
  assertRejects(
    "initializer と入力の衝突",
    "graph.initializers: 値 'x' が二重に定義されている",
    (g) => {
      g.initializers["x"] = {};
      g.values["x"] = { dtype: "f32", shape: [3] };
    },
  );
});

Deno.test("parseIrDeclaration: トポロジカル順（前方参照拒否）", () => {
  assertRejects("ノード順の逆転", "graph.nodes[0] (add): 入力 'h' が未定義（前方参照", (g) => {
    g.nodes.reverse();
  });
  assertRejects("未定義の ins", "入力 'unknown' が未定義", (g) => {
    g.nodes[0].ins = ["x", "unknown"];
  });
  assertRejects("未定義の outputs", "graph.outputs: 'z' が未定義", (g) => {
    g.outputs = ["z"];
  });
  assertRejects("outputs の重複", "graph.outputs: 'y' が重複している", (g) => {
    g.outputs = ["y", "y"];
  });
});

Deno.test("parseIrDeclaration: 宣言の完全性", () => {
  assertRejects("ノード出力が values に無い", "ノード出力 'y' の dtype/shape 宣言が無い", (g) => {
    delete g.values["y"];
  });
  assertRejects(
    "initializer が values に無い",
    "graph.initializers['w']: values に dtype/shape 宣言が無い",
    (g) => {
      delete g.values["w"];
    },
  );
  assertRejects("入力の二重宣言", "入力は inputs[] で宣言済み（二重宣言）", (g) => {
    g.values["x"] = { dtype: "f32", shape: ["T", 4] };
  });
  assertRejects(
    "孤立した values 宣言",
    "graph.values['ghost']: どのノードでも定義されない宣言",
    (g) => {
      g.values["ghost"] = { dtype: "f32", shape: [1] };
    },
  );
  assertRejects("initializer の記号次元", "initializer の shape に記号次元は使えない", (g) => {
    g.values["w"] = { dtype: "f32", shape: ["T", 3] };
  });
  assertRejects(
    "initializer の意味論 dtype が bool",
    "initializer の意味論 dtype 'bool' は語彙外",
    (g) => {
      g.values["w"] = { dtype: "bool", shape: [4, 3] };
    },
  );
});

/**
 * IR v2 の initializer 宣言に書けるのは `shared` だけ（実体との対応は容器の束縛表が
 * `(グラフ名, initializer 名)` で持つ — docs/ir-v2.md）。旧配布形の `tensor` / `storage` を
 * そのまま書いた形は未知のキーとして落ちる。
 */
Deno.test("parseIrDeclaration: initializer 宣言のキー集合は `shared` だけ", () => {
  // 実体を持つ宣言は空オブジェクト、借り物は `{ shared: true }` の 2 形きり。
  assertEquals(parseMutated(() => {}).initializers["w"], { shared: false });
  const borrowed = parseMutated((g) => {
    g.initializers["w"] = { shared: true };
  });
  assertEquals(borrowed.initializers["w"], { shared: true });

  for (const key of ["tensor", "storage", "scale", "group_size"]) {
    assertThrows(
      () =>
        parseMutated((g) => {
          Object.assign(g.initializers["w"], { [key]: "enc.w" });
        }),
      IrError,
      `未知のキー '${key}'`,
    );
  }
  // MUST: `false` は欄の不存在と同じ宣言なので綴れない（正準直列化が 1 つに決まらない）。
  assertThrows(
    () =>
      parseMutated((g) => {
        Object.assign(g.initializers["w"], { shared: false });
      }),
    IrError,
    "借り物のときだけ true",
  );
});

/**
 * 意味論 dtype と格納（codec）の交差（i32 は生の int32 1 通りのみ — ADR 0010）は**合流層**
 * （`format/container/bind.ts` の `allowedLayouts`）の担当になった。宣言だけを見るこの層では
 * `values` の意味論 dtype が i32 でも f32 でも受理する。
 */
Deno.test("parseIrDeclaration: 意味論 i32 の initializer を宣言として受理する", () => {
  const graph = parseMutated((g) => {
    g.values["w"] = { dtype: "i32", shape: [4, 3] };
  });
  assertEquals(graph.values["w"].dtype, "i32");
  assertEquals(graph.initializers["w"], { shared: false });
});

Deno.test("parseIrDeclaration: dtype 語彙", () => {
  assertRejects(
    "意味論 dtype に f16",
    "graph.values['h'].dtype: 意味論 dtype 'f16' は語彙外",
    (g) => {
      g.values["h"].dtype = "f16";
    },
  );
  assertRejects(
    "意味論 dtype に i64",
    "graph.inputs[0].dtype: 意味論 dtype 'i64' は語彙外",
    (g) => {
      g.inputs[0].dtype = "i64";
    },
  );
});
Deno.test("parseIrDeclaration: シンボルの宣言と束縛可能性", () => {
  assertRejects("未宣言シンボルの使用", "シンボル 'S' が graph.symbols で宣言されていない", (g) => {
    g.inputs[0].shape = ["S", 4];
  });
  assertRejects(
    "入力 shape の次元位置に一度も現れない",
    "'T' が入力 shape / states shape の次元位置に現れない",
    (g) => {
      g.inputs[0].shape = [8, 4];
    },
  );
  assertRejects("symbols の重複", "graph.symbols: 'T' が重複している", (g) => {
    g.symbols = ["T", "T"];
  });
  assertRejects("シンボル名が不正", "シンボル名 '2T' が不正", (g) => {
    g.symbols = ["T", "2T"];
  });
  // 派生形だけの入力は**受理する**（実寸から解が一意 — ADR 0057）。母音検出 CRNN は
  // 先頭 conv の stride 2 のせいで `2T` でしか長さ軸を宣言できない。
  const derived = parseMutated((g) => {
    g.inputs[0].shape = ["2T", 4];
  });
  assertEquals(derived.inputs[0].shape, ["2T", 4]);
});

Deno.test("parseIrDeclaration: shape 要素", () => {
  assertRejects("非正準な次元式", "次元式 '1T' が正準文法", (g) => {
    g.inputs[0].shape = ["1T", 4];
  });
  assertRejects("負の次元", "次元 -3 が非負整数でない", (g) => {
    g.values["b"].shape = [-3];
  });
  assertRejects("小数の次元", "次元 1.5 が非負整数でない", (g) => {
    g.values["b"].shape = [1.5];
  });
  const boolDim = JSON.stringify(baseDeclaration()).replace('"shape":[3]', '"shape":[true]');
  assertThrows(() => parseIrDeclaration(boolDim), IrError, "数値でも文字列でもない");
});

Deno.test("parseIrDeclaration: ノードの構造", () => {
  assertRejects("attrs がオブジェクトでない", "graph.nodes[0].attrs: オブジェクトでない", (g) => {
    g.nodes[0].attrs = 5;
  });
  assertRejects("op が空文字列", "graph.nodes[0].op: 空でない文字列でない", (g) => {
    g.nodes[0].op = "";
  });
  assertRejects("outs の要素が空文字列", "graph.nodes[0].outs[0]: 空でない文字列でない", (g) => {
    g.nodes[0].outs = [""];
  });
});

// ADR 0067 決定 5: `outs` の本数に意味を与えるのは**契約層**（0 本を許すのは契約が effect を
// 宣言する op だけ）。パーサは本数を見ない — ここで見るのは「見ていないこと」そのもので、
// 非 effect op の `outs: []` が落ちるのは**別の規則**（出力の宣言が孤立する）による。
Deno.test("parseIrDeclaration: outs の本数はパーサの担当ではない", () => {
  // 出力宣言ごと消せばパーサは通る（本数の執行は tests/ops_contract_test.ts が固定する）
  const graph = parseMutated((g) => {
    g.nodes[1].outs = [];
    delete g.values["y"];
    g.outputs = ["h"];
  });
  assertEquals(graph.nodes[1].outs, []);
  // 宣言だけ残せば「孤立した values 宣言」で落ちる（出力数違反としてではない）
  assertThrows(
    () =>
      parseMutated((g) => {
        g.nodes[1].outs = [];
        g.outputs = ["h"];
      }),
    IrError,
    "どのノードでも定義されない宣言",
  );
});

// "__proto__" を名前に持つグラフ。フィクスチャは生の JSON 文字列から組む MUST — JS の
// オブジェクトリテラルに `"__proto__":` を書くと own key ではなく [[Prototype]] 指定になり、
// JSON.parse が作る own property と別物になってテストが検査対象を外す。
//
// initializer 名がそのまま実体の鍵なので改名は起きないが、合流（`mergedGraph`）は initializer を
// 格納確定後の**新しい器へ詰め直す**ので、宣言側の器（パーサ）と合流後の器の**両方**が
// null プロトタイプでないと own property が黙って消える。
const protoNameGraph = `{
  "format": "karume-ir",
  "version": 2,
  "requires": { "ops": ["matmul"] },
  "symbols": ["T"],
  "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
  "outputs": ["y"],
  "initializers": { "__proto__": {} },
  "values": {
    "__proto__": { "dtype": "f32", "shape": [4, 3] },
    "y": { "dtype": "f32", "shape": ["T", 3] }
  },
  "nodes": [{ "op": "matmul", "ins": ["x", "__proto__"], "outs": ["y"], "attrs": {} }]
}`;

Deno.test("parseIrDeclaration: '__proto__' という名前の宣言を own property として保全する", () => {
  const graph = parseIrDeclaration(protoNameGraph);

  // 実行系によっては `o["__proto__"] = v` でも own property が作られるため、hasOwn だけでは
  // 素の `{}` への退行を検出できない。器が null プロトタイプであること（＝ 受理集合に
  // エンジン差を持ち込まないこと）を併せて固定する。
  assertEquals(Object.getPrototypeOf(graph.values), null);
  assertEquals(Object.getPrototypeOf(graph.initializers), null);
  assertEquals(Object.hasOwn(graph.values, "__proto__"), true);
  assertEquals(Object.hasOwn(graph.initializers, "__proto__"), true);
  assertEquals(graph.values["__proto__"], { dtype: "f32", shape: [4, 3] });
  assertEquals(graph.initializers["__proto__"], { shared: false });
  // 宣言検査を素通りしたのではなく、ノードからの参照込みで通っていること。
  assertEquals(graph.nodes[0].ins, ["x", "__proto__"]);
});

Deno.test("parseIrDeclaration: '__proto__' という名前でも孤立宣言は黙って消えず拒否される", () => {
  const orphan = `{
    "format": "karume-ir",
    "version": 2,
    "requires": { "ops": ["matmul"] },
    "symbols": ["T"],
    "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
    "outputs": ["y"],
    "initializers": { "w": {} },
    "values": {
      "w": { "dtype": "f32", "shape": [4, 3] },
      "y": { "dtype": "f32", "shape": ["T", 3] },
      "__proto__": { "dtype": "f32", "shape": [1] }
    },
    "nodes": [{ "op": "matmul", "ins": ["x", "w"], "outs": ["y"], "attrs": {} }]
  }`;

  assertThrows(() => parseIrDeclaration(orphan), IrError, "どのノードでも定義されない宣言");
});

// ADR 0066 決定 2: 生成 1 本ぶんの可変 state を置く名前付きスロット。参照側の欄（ADR 0067 の
// states 欄 / state_append）が入ったので、**宣言だけのグラフは参照完全性で落ちる** —
// 宣言そのものを見るテストは withStateReaders で参照側を用意する。
Deno.test("parseIrDeclaration: states 節を持たないグラフは空のスロット集合になる", () => {
  const graph = parseIrDeclaration(JSON.stringify(baseDeclaration()));

  assertEquals(Object.keys(graph.states), []);
  // 名前がパース入力由来の器は null プロトタイプ（values / initializers と同じ MUST）。
  assertEquals(Object.getPrototypeOf(graph.states), null);
});

Deno.test("parseIrDeclaration: state スロットの宣言を受理する", () => {
  const empty = parseMutated((g) => {
    g.states = {};
  });
  assertEquals(Object.keys(empty.states), []);

  // 容量の違うスロットが混在する形（sliding 層 / full 層は別スロット — 層 × 均一 KV の
  // 前提を作らない）。
  const slots = parseMutated((g) => {
    g.states = {
      "layer0.k": { dtype: "f32", shape: [1, 2, 512, 128] },
      "layer0.v": { dtype: "f32", shape: [1, 2, 512, 128] },
      "layer1.k": { dtype: "f32", shape: [1, 2, 131072, 128] },
    };
    withStateReaders(g);
  });
  assertEquals(Object.keys(slots.states), ["layer0.k", "layer0.v", "layer1.k"]);
  assertEquals(slots.states["layer0.k"], {
    dtype: "f32",
    shape: [1, 2, 512, 128],
    external: false,
  });
  assertEquals(slots.states["layer1.k"].shape, [1, 2, 131072, 128]);

  // 記号次元は Session の symbols で解決する（入力にも現れる記号なので、束縛点は従来どおり
  // 入力 shape の次元位置 — states 専用記号は別テストが持つ）。
  const symbolic = parseMutated((g) => {
    g.states = { cache: { dtype: "f32", shape: ["T", 4] } };
    withStateReaders(g);
  });
  assertEquals(symbolic.states["cache"].shape, ["T", 4]);

  // rank 1..4 の両端（容量軸 1 本だけの汎用スロットも valid）。
  const ranks = parseMutated((g) => {
    g.states = {
      flat: { dtype: "f32", shape: [8] },
      full: { dtype: "f32", shape: [1, 2, 3, 4] },
    };
    withStateReaders(g);
  });
  assertEquals(ranks.states["flat"].shape, [8]);
  assertEquals(ranks.states["full"].shape, [1, 2, 3, 4]);
});

/**
 * states 節だけを差し替えて拒否を見る。**診断文言まで見る MUST** — 節そのものが未対応へ退行すると
 * 「未知のキー 'states'」で落ちるので、`IrError` だけの検査では規則が効いている証拠にならない。
 */
const assertStatesReject = (states: unknown, includes: string): void => {
  assertThrows(
    () =>
      parseMutated((graph) => {
        (graph as { states: unknown }).states = states;
      }),
    IrError,
    includes,
  );
};

Deno.test("parseIrDeclaration: state スロットの dtype は f32 のみ", () => {
  // f16 は席の予約だけ（ADR 0066 追記 5）— 「語彙外」ではなく「未対応」として落ちる。
  assertStatesReject({ cache: { dtype: "f16", shape: [4] } }, "は未対応");
  for (const dtype of ["i32", "bool", "bf16", "i8"]) {
    assertStatesReject({ cache: { dtype, shape: [4] } }, "は語彙外（f32）");
  }
});

Deno.test("parseIrDeclaration: state スロットの shape は容量込みの具体形", () => {
  assertStatesReject({ cache: { dtype: "f32", shape: [1, 1, 1, 1, 1] } }, "rank 5 は 1..4 の外");
  assertStatesReject({ cache: { dtype: "f32", shape: [] } }, "rank 0 は 1..4 の外");
  assertStatesReject({ cache: { dtype: "f32", shape: [1, 0, 4] } }, "正整数でない");
  assertStatesReject({ cache: { dtype: "f32", shape: [-4] } }, "非負整数でない");
  assertStatesReject({ cache: { dtype: "f32", shape: ["S", 4] } }, "宣言されていない");
  assertStatesReject({ cache: { dtype: "f32", shape: ["1T", 4] } }, "正準文法");
});

/**
 * 次元 1 個だけを**生の JSON テキスト**で差し込む（`JSON.stringify` は `1.0` を `1` に畳むので、
 * 整数値 float の受理はオブジェクト経由では観測できない）。
 */
const parseRawDim = (
  raw: string,
  mutate: (graph: DeclarationJson, dim: string) => void,
): IrDeclaration => {
  const graph = baseDeclaration();
  mutate(graph, "__DIM__");
  return parseIrDeclaration(JSON.stringify(graph).replaceAll('"__DIM__"', raw));
};

// JSON の `4.0` / `4e0` は JSON.parse が単一の number にするので整数として通る。Python 側
// （tools/exporter/src/karume/verify.py の _parse_shape）は int / float が別型なので、この形が
// **両側で同じ受理集合**であることを固定しないと「ランタイムは読めるが exporter の検証が落ちる」
// 乖離になる（鏡像は tools/exporter/tests/test_verify.py の TestIntegralFloatDimensions）。
Deno.test("parseIrDeclaration: JSON の整数値 float 次元を受理し、非整数は拒否する", () => {
  for (const raw of ["4.0", "4e0", "0.4e1"]) {
    const values = parseRawDim(raw, (g, dim) => {
      g.values["w"].shape = [dim];
    });
    assertEquals(values.values["w"].shape, [4], raw);
    const states = parseRawDim(raw, (g, dim) => {
      g.states = { cache: { dtype: "f32", shape: [1, dim] } };
      withStateReaders(g);
    });
    assertEquals(states.states["cache"].shape, [1, 4], raw);
  }
  for (const raw of ["1.5", "15e-1"]) {
    assertThrows(
      () =>
        parseRawDim(raw, (g, dim) => {
          g.values["w"].shape = [dim];
        }),
      IrError,
      "非負整数でない",
    );
    assertThrows(
      () =>
        parseRawDim(raw, (g, dim) => {
          g.states = { cache: { dtype: "f32", shape: [1, dim] } };
          withStateReaders(g);
        }),
      IrError,
      "非負整数でない",
    );
  }
  // safe range 超過は Python 側（> 2^53−1）と同じ点で落ちる。
  assertThrows(
    () =>
      parseRawDim("1e300", (g, dim) => {
        g.values["w"].shape = [dim];
      }),
    IrError,
    "非負整数でない",
  );
});

/**
 * 空の値名 / initializer 名は states のスロット名と同じ理由で拒否する（参照側の欄 —
 * inputs[].name / ins / outs / outputs — はどれも空でない文字列だけを受理するので、通すと
 * 「宣言はできるが原理的に参照できない」declaration になる）。
 *
 * MUST: 空名 value **単独**なら孤立宣言検査が塞ぐが、空名 initializer と**対で**書かれると
 * 定義集合に入って孤立宣言検査も逆参照検査も満たす — 参照不能な実テンソル 1 本が配布形に
 * 居座る。この組が門の唯一の検出対象。
 */
Deno.test("parseIrDeclaration: 値名 / initializer 名の空文字列を拒否する", () => {
  assertThrows(
    () =>
      parseMutated((g) => {
        g.values[""] = { dtype: "f32", shape: [4, 3] };
        g.initializers[""] = {};
      }),
    IrError,
    "graph.values の値名: 空でない文字列でない",
  );
  // initializer 側だけが空の形も落ちる（名前検査は宣言の両側に掛かる）
  assertThrows(
    () =>
      parseMutated((g) => {
        g.initializers[""] = {};
      }),
    IrError,
    "graph.initializers の initializer 名: 空でない文字列でない",
  );
  // 対: 名前が空でなければ同じ組は受理される（= 落としているのは空名だけで、宣言完全性の
  // 2 検査はどちらもこの組を通す — それがこの門が要る理由そのもの）
  const graph = parseMutated((g) => {
    g.values["ghost"] = { dtype: "f32", shape: [4, 3] };
    g.initializers["ghost"] = {};
  });
  assertEquals(graph.initializers["ghost"], { shared: false });
});

Deno.test("parseIrDeclaration: states の構造", () => {
  assertStatesReject([], "graph.states: オブジェクトでない");
  assertStatesReject({ cache: 4 }, "graph.states['cache']: オブジェクトでない");
  assertStatesReject({ cache: { dtype: "f32", shape: 4 } }, "shape: 配列でない");
  assertStatesReject({ cache: { shape: [4] } }, "必須キー 'dtype' が無い");
  assertStatesReject({ cache: { dtype: "f32", shape: [4], window: 512 } }, "未知のキー 'window'");
  // 空のスロット名は参照側の欄（ADR 0067）が受理しない = 参照できない宣言になる。
  assertStatesReject({ "": { dtype: "f32", shape: [4] } }, "スロット名: 空でない文字列でない");
});

// スロット名は値名前空間と別（ins / outs で参照されない）だが、同名は「スロット名の欄に値名を
// 書いた / その逆」を検出できなくするだけなので拒否する。
Deno.test("parseIrDeclaration: state スロット名は値名と衝突できない", () => {
  // 入力 / initializer / 中間値 / グラフ出力の 4 役すべて。
  for (const name of ["x", "w", "h", "y"]) {
    assertStatesReject(
      { [name]: { dtype: "f32", shape: [4] } },
      `graph.states['${name}']: 値名と同名`,
    );
  }
});

Deno.test("parseIrDeclaration: '__proto__' という名前の state スロットを own property として保全する", () => {
  const graph = parseIrDeclaration(`{
    "format": "karume-ir",
    "version": 2,
    "requires": { "ops": ["matmul", "state_append"] },
    "symbols": ["T"],
    "inputs": [{ "name": "x", "dtype": "f32", "shape": ["T", 4] }],
    "outputs": ["y"],
    "initializers": { "w": {} },
    "values": {
      "w": { "dtype": "f32", "shape": [4, 3] },
      "y": { "dtype": "f32", "shape": ["T", 3] }
    },
    "states": { "__proto__": { "dtype": "f32", "shape": [4] } },
    "nodes": [
      { "op": "matmul", "ins": ["x", "w"], "outs": ["y"], "attrs": {} },
      {
        "op": "state_append",
        "ins": ["x"],
        "outs": [],
        "attrs": {},
        "states": { "slot": "__proto__" }
      }
    ]
  }`);

  assertEquals(Object.getPrototypeOf(graph.states), null);
  assertEquals(Object.hasOwn(graph.states, "__proto__"), true);
  assertEquals(graph.states["__proto__"], { dtype: "f32", shape: [4], external: false });
  // 参照側の欄も同じ理由で null プロトタイプ（素の `{}` では代入が [[Prototype]] 設定に
  // 化けて欄が黙って消え、参照完全性検査も契約のキー集合検査も素通りする形が作れる）。
  const referring = graph.nodes[1].states;
  assertEquals(Object.getPrototypeOf(referring), null);
  assertEquals(referring["slot"], "__proto__");
});

/**
 * state を読む層（states 形 attention）と書く層（`state_append`）が 1 本ずつ並んだ最小形
 * （ADR 0067 決定 4 / 5・発行順は決定 5b の「読者 → append」）。**パーサ層の**テストなので
 * shape も契約も見られない — ここで固定するのは欄の構造と参照の完全性だけ。
 */
const statefulGraph = (): DeclarationJson => {
  const graph = baseDeclaration();
  graph.states = {
    "kv.k": { dtype: "f32", shape: [1, 2, 512, 8] },
    "kv.v": { dtype: "f32", shape: [1, 2, 512, 8] },
  };
  graph.requires.ops.push("attention", "state_append");
  graph.values["att"] = { dtype: "f32", shape: ["T", 3] };
  graph.nodes.push(
    {
      op: "attention",
      ins: ["h", "h", "h"],
      outs: ["att"],
      attrs: { scale: 0.5, window: 512 },
      states: { k: "kv.k", v: "kv.v" },
    },
    { op: "state_append", ins: ["h"], outs: [], attrs: { window: 512 }, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["h"], outs: [], attrs: { window: 512 }, states: { slot: "kv.v" } },
  );
  return graph;
};

const parseStateful = (mutate: (graph: DeclarationJson) => void = () => {}): IrDeclaration => {
  const graph = statefulGraph();
  mutate(graph);
  return parseIrDeclaration(JSON.stringify(graph));
};

// ADR 0067 決定 4: ノードは `ins` / `outs` と**別の欄**でスロットを名前参照する。
Deno.test("parseIrDeclaration: ノードの states 欄を読む", () => {
  const graph = parseStateful();

  assertEquals(graph.nodes.map((node) => node.op), [
    "matmul",
    "add",
    "attention",
    "state_append",
    "state_append",
  ]);
  // 欄を持たないノードは空表（常在欄 — 消費側が `?? {}` を書かずに済む）
  assertEquals(Object.keys(graph.nodes[0].states), []);
  assertEquals(Object.entries(graph.nodes[2].states), [["k", "kv.k"], ["v", "kv.v"]]);
  assertEquals(graph.nodes[3].states["slot"], "kv.k");
  // 0 出力ノードはパーサを通る（本数の執行点は契約層 — ADR 0067 決定 5）
  assertEquals(graph.nodes[3].outs, []);
});

Deno.test("parseIrDeclaration: states 欄は宣言済みスロットしか参照できない", () => {
  assertThrows(
    () =>
      parseStateful((g) => {
        g.nodes[3].states = { slot: "kv.missing" };
      }),
    IrError,
    "state スロット 'kv.missing' が graph.states で宣言されていない",
  );
  // 値名を書いた取り違え（スロット名前空間と値名前空間は別 — 値名は参照できない）
  assertThrows(
    () =>
      parseStateful((g) => {
        g.nodes[3].states = { slot: "h" };
      }),
    IrError,
    "graph.states で宣言されていない",
  );
});

Deno.test("parseIrDeclaration: 誰も参照しない state スロットを拒否する", () => {
  assertThrows(
    () =>
      parseStateful((g) => {
        g.states!["kv.orphan"] = { dtype: "f32", shape: [1, 2, 512, 8] };
      }),
    IrError,
    "graph.states['kv.orphan']: どのノードからも参照されない宣言",
  );
  // 参照が 1 本でもあれば足りる（読者だけ / 書き手だけの層はどちらも実在する —
  // KV 共有層は append を持たない）
  const readerOnly = parseStateful((g) => {
    g.nodes.splice(3, 2);
    g.requires.ops = ["matmul", "add", "attention"];
  });
  assertEquals(Object.keys(readerOnly.states), ["kv.k", "kv.v"]);
});

Deno.test("parseIrDeclaration: states 欄の構造", () => {
  const reject = (states: unknown, includes: string): void => {
    assertThrows(
      () =>
        parseStateful((g) => {
          (g.nodes[3] as { states: unknown }).states = states;
        }),
      IrError,
      includes,
    );
  };
  reject([], "graph.nodes[3].states: オブジェクトでない");
  reject({ slot: 4 }, "graph.nodes[3].states['slot']: 空でない文字列でない");
  reject({ slot: "" }, "graph.nodes[3].states['slot']: 空でない文字列でない");
  reject({ "": "kv.k" }, "graph.nodes[3].states のキー: 空でない文字列でない");
});

// ADR 0066 追記 7: 束縛点は 2 つ（入力 shape / states shape）だが、**効く範囲は非対称**。
Deno.test("parseIrDeclaration: states 専用記号は宣言できるが値 shape には現れない", () => {
  // 容量記号 C は入力のどこにも現れない（context 生成時に決まる）— それでも宣言できる
  const graph = parseStateful((g) => {
    g.symbols.push("C");
    g.states!["kv.k"].shape = [1, 2, "C", 8];
    g.states!["kv.v"].shape = [1, 2, "C", 8];
  });
  assertEquals(graph.states["kv.k"].shape, [1, 2, "C", 8]);

  // 値 shape に現れたら fail loudly（通常値の解決に効くのは入力由来の束縛だけなので、
  // 実行時に必ず束縛不能になる — 宣言の時点で落とす）
  assertThrows(
    () =>
      parseStateful((g) => {
        g.symbols.push("C");
        g.states!["kv.k"].shape = [1, 2, "C", 8];
        g.states!["kv.v"].shape = [1, 2, "C", 8];
        g.values["att"].shape = ["C", 3];
      }),
    IrError,
    "states 専用記号 'C' が値 shape に現れる",
  );

  // 入力にも states にも現れない記号は従来どおり束縛が取れない
  assertThrows(
    () =>
      parseStateful((g) => {
        g.symbols.push("C");
      }),
    IrError,
    "'C' が入力 shape / states shape の次元位置に現れない",
  );

  // 入力**と** states の両方に現れる記号は「states 専用」ではない（値 shape に現れてよい）
  const shared = parseStateful((g) => {
    g.states!["kv.k"].shape = [1, 2, "T", 8];
    g.states!["kv.v"].shape = [1, 2, "T", 8];
  });
  assertEquals(shared.values["att"].shape, ["T", 3]);
});

Deno.test("parseIrDeclaration: JSON の受理集合", () => {
  assertThrows(() => parseIrDeclaration("{"), IrError, "解析できない");
  assertThrows(
    () => parseIrDeclaration('{"format":"karume-ir","version":NaN}'),
    IrError,
    "解析できない",
  );
  // 1e999 は構文としては有効だが Infinity へ丸まる — 受理集合をブラウザ JSON.parse に
  // 揃える契約のため拒否する。
  const infinite = JSON.stringify(baseDeclaration()).replace('"shape":[4,3]', '"shape":[4,1e999]');
  assertThrows(() => parseIrDeclaration(infinite), IrError, "有限");
});

/**
 * ランタイム対応表との突合門（`assertRuntimeSupport`）— `PreparedModel` が**重みの block を
 * 1 つも取る前に**通す capability 門（ADR 0070 決定 5 の 2 段境界）。
 *
 * ここで固定するのは「非対応は**全件列挙して** 1 回で落とす」という設計主張そのもの:
 * 件数（重複除去の単位 = 宣言）・軸（op / 意味論 dtype / attrs / 格納）・スロット別の受理集合が
 * 崩れると、モデル作者は対応表を埋めるたびに次の 1 本が現れる形に戻る。
 *
 * グラフは**合流後**（`IrGraph`）でなければならないので、メモリ内容器の供給面から
 * `mergedGraph` で組む（`krm` 経路と同じ 1 本 — 供給元は主題ではないので容器を書かない）。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { mergedGraph } from "../src/format/container/bind.ts";
import type { CodecLayout } from "../src/format/container/codecs.ts";
import type { IrDtype, IrGraph } from "../src/format/ir.ts";
import { RUNTIME_SUPPORT } from "../src/ops.ts";
import {
  assertRuntimeSupport,
  type OpSupport,
  type RuntimeSupport,
  RuntimeSupportError,
} from "../src/ops/support.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  baseDeclaration,
  baseTensors,
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
} from "./helpers/model-fixture.ts";

/** M0 と同形（f32 のみ・attrs 無し・二項・単一出力）の最小対応表。 */
const f32Only: OpSupport = {
  dtypes: new Set<IrDtype>(["f32"]),
  slotDtypes: [new Set<IrDtype>(["f32"]), new Set<IrDtype>(["f32"])],
  outDtypes: [new Set<IrDtype>(["f32"])],
  attrKeys: new Set<string>(),
};

const M0_SUPPORT: RuntimeSupport = {
  ops: new Map([["matmul", f32Only], ["add", f32Only]]),
  storage: new Set<CodecLayout>(["f32"]),
  io: new Set<IrDtype>(["f32"]),
};

/** 宣言 + 供給 → 合流後のグラフ（`assertRuntimeSupport` が受ける形）。 */
const graphOf = (
  declaration: DeclarationJson,
  tensors: readonly TensorInput[] = baseTensors(),
): IrGraph => mergedGraph(memoryModel(declaration, tensors).graphs[GRAPH_NAME], GRAPH_NAME);

/** 重みを持たない宣言（gather / topk のようにグラフ側だけを問うケース）。 */
const weightless = (): DeclarationJson => {
  const graph = baseDeclaration();
  graph.initializers = {};
  return graph;
};

describe("assertRuntimeSupport: op の軸", () => {
  it("非対応 op を列挙して落とす", () => {
    const graph = baseDeclaration();
    graph.requires.ops = ["gelu", "matmul", "tanh"];
    graph.nodes[1] = { op: "gelu", ins: ["h"], outs: ["y"], attrs: {} };
    graph.nodes.push({ op: "tanh", ins: ["y"], outs: ["z"], attrs: {} });
    graph.values["z"] = { dtype: "f32", shape: ["T", 3] };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "非対応 op (2)",
    );
    assertEquals(error.message.includes("gelu, tanh"), true, error.message);
  });

  // op 名だけの突合は「対応表にはあるのに実行時に落ちる」を作る（recon §3-9）。
  it("対応 op でも実行できない意味論 dtype を宣言ごとに列挙する", () => {
    const graph = baseDeclaration();
    graph.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
    graph.values["h"] = { dtype: "bool", shape: ["T", 3] };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      // 件数は「直すべき宣言の本数」— h は matmul の outs と add の ins の 2 箇所に現れるが 1 件
      "非対応 意味論 dtype (2)",
    );
    assertEquals(error.message.includes("値 'x': i32"), true, error.message);
    assertEquals(error.message.includes("値 'h': bool"), true, error.message);
  });

  // エイリアス入力（同じ値を 2 回取るノード）で件数が水増しされないこと。
  it("同一宣言の dtype 違反を重複列挙しない", () => {
    const graph = baseDeclaration();
    graph.values["h"] = { dtype: "bool", shape: ["T", 3] };
    graph.values["y"] = { dtype: "f32", shape: ["T", 3] };
    graph.nodes[1] = { op: "add", ins: ["h", "h"], outs: ["y"], attrs: {} };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "非対応 意味論 dtype (1)",
    );
    assertEquals(error.message.includes("値 'h': bool"), true, error.message);
  });

  // ノード起点の突合だけでは、どのノードも消費しない入力の dtype 違反が門を素通りする
  // （実行器は全 graph.inputs を転送するので、転送層の制約は使用の有無と無関係に実在する）。
  it("どのノードも使わない入力の dtype 違反も列挙する", () => {
    const graph = baseDeclaration();
    graph.inputs.push({ name: "z", dtype: "i32", shape: ["T"] });

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "非対応 意味論 dtype (1)",
    );
    assertEquals(error.message.includes("値 'z': i32"), true, error.message);
  });
});

describe("assertRuntimeSupport: スロット別の受理集合", () => {
  // スロット別 dtype 契約（gather / embedding / masked_fill）は、受理集合の**和**で突き合わせると
  // 「値と添字を逆に渡した形」がどちらも和に入るため列挙門を素通りする。契約検査（plan.ts）まで
  // 落ちて 1 件ずつ止まると、「非対応は全件列挙して一度に見せる」という門の意図が壊れる。
  it("スロットを取り違えた perSlot op を列挙する", () => {
    const graph = weightless();
    graph.requires = { ops: ["gather"] };
    // 値 f32 と添字 i32 を**逆に**渡した形。和（{f32, i32}）だけの突合では両方通ってしまう。
    graph.inputs = [
      { name: "src", dtype: "i32", shape: ["T", 4] },
      { name: "idx", dtype: "f32", shape: ["T", 3] },
    ];
    graph.outputs = ["y"];
    graph.values = { y: { dtype: "f32", shape: ["T", 3] } };
    graph.nodes = [{ op: "gather", ins: ["src", "idx"], outs: ["y"], attrs: {} }];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph, []), RUNTIME_SUPPORT),
      RuntimeSupportError,
      "非対応 意味論 dtype (2)",
    );
    assertEquals(error.message.includes("値 'src': i32"), true, error.message);
    assertEquals(error.message.includes("値 'idx': f32"), true, error.message);
  });

  // 出力もスロット 0（値の側）と同型でなければ実行できない — 和で見ると gather の i32 出力が通る。
  it("perSlot op の出力 dtype も値の側の受理集合で見る", () => {
    const graph = weightless();
    graph.requires = { ops: ["gather"] };
    graph.inputs = [
      { name: "src", dtype: "f32", shape: ["T", 4] },
      { name: "idx", dtype: "i32", shape: ["T", 3] },
    ];
    graph.outputs = ["y"];
    graph.values = { y: { dtype: "i32", shape: ["T", 3] } };
    graph.nodes = [{ op: "gather", ins: ["src", "idx"], outs: ["y"], attrs: {} }];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph, []), RUNTIME_SUPPORT),
      RuntimeSupportError,
      "非対応 意味論 dtype (1)",
    );
    assertEquals(error.message.includes("値 'y': i32"), true, error.message);
  });

  // 多出力 op（topk）の出力宣言を slot 間で**入れ替えた**形。全出力を slot 0 の受理集合で見る
  // 退行（= 出力 slot 別の列を潰した実装）だと、値の側の f32 と添字の側の i32 がどちらも「slot 0 の
  // 集合」に照らして判定され、片方しか列挙されない（あるいは両方素通りする）。
  // MUST: tools/exporter/tests/test_verify.py の
  // `test_swapped_output_slots_of_a_multi_output_op_are_enumerated` と**同形**に保つ（同じ退行を
  // 両側で検出できることがこの門の対称性）。
  it("多出力 op の出力 slot を取り違えた形を両宣言とも列挙する", () => {
    const graph = weightless();
    graph.requires = { ops: ["topk"] };
    graph.inputs = [{ name: "x", dtype: "f32", shape: ["T", 4] }];
    graph.outputs = ["v", "i"];
    // slot 0 は値（f32）・slot 1 は添字（i32）なので、この 2 本は**どちらも**非対応
    graph.values = {
      v: { dtype: "i32", shape: ["T", 2] },
      i: { dtype: "f32", shape: ["T", 2] },
    };
    graph.nodes = [{ op: "topk", ins: ["x"], outs: ["v", "i"], attrs: { k: 2 } }];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph, []), RUNTIME_SUPPORT),
      RuntimeSupportError,
      "非対応 意味論 dtype (2)",
    );
    assertEquals(error.message.includes("値 'v': i32"), true, error.message);
    assertEquals(error.message.includes("値 'i': f32"), true, error.message);
  });
});

describe("assertRuntimeSupport: attrs と格納の軸", () => {
  it("対応 op に付いた未実装 attrs をノードごとに列挙する", () => {
    const graph = baseDeclaration();
    graph.nodes[1].attrs = { alpha: 1, approximate: "tanh" };

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(graph), M0_SUPPORT),
      RuntimeSupportError,
      "未実装 attrs (1)",
    );
    assertEquals(error.message.includes("nodes[1] (add): alpha, approximate"), true, error.message);
  });

  it("非対応の格納を initializer 名つきで列挙する", () => {
    // 意味論 dtype は f32 のまま、格納 codec だけを f16 / bf16 にする（展開経路は layout で見る）。
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: new Uint8Array(new ArrayBuffer(24)),
        encoding: { codec: "f16" },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: new Uint8Array(new ArrayBuffer(6)),
        encoding: { codec: "bf16" },
      },
    ];

    const error = assertThrows(
      () => assertRuntimeSupport(graphOf(baseDeclaration(), tensors), M0_SUPPORT),
      RuntimeSupportError,
      "capability 不足",
    );
    assertEquals(error.message.includes("非対応 格納 'bf16' (1): b"), true, error.message);
    assertEquals(error.message.includes("非対応 格納 'f16' (1): w"), true, error.message);
  });

  it("対応表に載っている格納なら通る（門が恒真でないことの対照）", () => {
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: new Uint8Array(new ArrayBuffer(24)),
        encoding: { codec: "f16" },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: f32Bytes([1, 2, 3]),
        encoding: { codec: "f32" },
      },
    ];
    assertRuntimeSupport(graphOf(baseDeclaration(), tensors), {
      ...M0_SUPPORT,
      storage: new Set<CodecLayout>(["f32", "f16"]),
    });
  });
});

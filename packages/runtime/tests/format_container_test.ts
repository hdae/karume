import { assertEquals, assertThrows } from "@std/assert";
import {
  assertRuntimeSupport,
  ContainerError,
  IR_METADATA_KEY,
  openModel,
  type OpSupport,
  type RuntimeSupport,
} from "../src/format/container.ts";
import { type IrDtype, IrError, type IrStorageDtype } from "../src/format/ir.ts";
import { RUNTIME_SUPPORT } from "../src/ops.ts";
import {
  baseGraph,
  baseModelBuffer,
  buildSafetensors,
  f32Bytes,
  type GraphJson,
  type TensorSpec,
} from "./helpers/format.ts";

/** M0 と同形（f32 のみ・attrs 無し・二項・単一出力）の最小対応表。 */
const f32Only: OpSupport = {
  dtypes: new Set<IrDtype>(["f32"]),
  slotDtypes: [new Set<IrDtype>(["f32"]), new Set<IrDtype>(["f32"])],
  outDtypes: [new Set<IrDtype>(["f32"])],
  attrKeys: new Set<string>(),
};

const M0_SUPPORT: RuntimeSupport = {
  ops: new Map([["matmul", f32Only], ["add", f32Only]]),
  storage: new Set<IrStorageDtype>(["f32"]),
  io: new Set<IrDtype>(["f32"]),
};

const i8Bytes = (length: number): Uint8Array<ArrayBuffer> => new Uint8Array(length);

/**
 * group 量子化格納（i4 + group 形 scale — ADR 0069）の最小モデル。重みは `[4,64]` で
 * group_size 32 なので、scale の group 形は `[4,2]`（keepdim broadcast 形 `[4,1]` とは別物 —
 * 形の分岐が効いていないと片方が素通りする）。
 */
const i4Model = (
  mutate: (parts: { graph: GraphJson; tensors: TensorSpec[] }) => void = () => {},
): ArrayBuffer => {
  const graph = baseGraph();
  graph.values["w"] = { dtype: "f32", shape: [4, 64] };
  graph.initializers["w"].storage = { dtype: "i4", scale: "enc.w.scale", group_size: 32 };
  const tensors: TensorSpec[] = [
    { name: "enc.w", dtype: "I4", shape: [4, 64], data: new Uint8Array(128) },
    { name: "enc.w.scale", dtype: "F32", shape: [4, 2], data: f32Bytes(new Array(8).fill(1)) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
  ];
  mutate({ graph, tensors });
  return baseModelBuffer(graph, tensors);
};

Deno.test("openModel: 正常系は graph と safetensors を結合して開ける", () => {
  const model = openModel(baseModelBuffer());
  assertEquals(model.graph.nodes.length, 2);
  assertEquals(model.file.tensors.get("enc.w")?.shape, [4, 3]);
  assertEquals(model.graph.initializers["w"].tensor, "enc.w");
  assertRuntimeSupport(model.graph, M0_SUPPORT);
});

Deno.test("openModel: グラフ JSON が無いファイルを拒否する", () => {
  const buffer = buildSafetensors([
    { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0)) },
  ]);
  assertThrows(() => openModel(buffer), ContainerError, IR_METADATA_KEY);
});

Deno.test("openModel: initializer の参照先テンソルが無いものを拒否する", () => {
  const graph = baseGraph();
  graph.initializers["w"].tensor = "missing.w";
  assertThrows(
    () =>
      openModel(baseModelBuffer(graph, [
        { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
      ])),
    ContainerError,
    "がファイルに無い",
  );

  // 実体を残したまま宣言だけ改名した形は「欠け」と「余剰」の両方が立つ。shard 進行検証では
  // 余剰は shard 単体で決まり、欠けは全 shard 読了まで決まらない（後続 shard で来るため）ので、
  // 帰属は余剰が先になる — どちらの形も拒否されることは変わらない（ADR 0070 決定 1）。
  assertThrows(
    () => openModel(baseModelBuffer(graph)),
    ContainerError,
    "参照されないテンソル (1): enc.w",
  );
});

Deno.test("openModel: storage.dtype と safetensors dtype の不一致を拒否する", () => {
  const buffer = baseModelBuffer(baseGraph(), [
    { name: "enc.w", dtype: "BF16", shape: [4, 3], data: new Uint8Array(24) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
  ]);
  assertThrows(() => openModel(buffer), ContainerError, "F32 が必要");
});

Deno.test("openModel: 宣言 shape と実テンソル shape の不一致を拒否する", () => {
  const graph = baseGraph();
  graph.values["w"] = { dtype: "f32", shape: [3, 4] };
  assertThrows(() => openModel(baseModelBuffer(graph)), ContainerError, "宣言 shape");
});

// 規則の正本はパーサ（ir.ts）— openModel 経由でも同じ拒否に到達することだけを固定する。
Deno.test("openModel: 意味論と格納が交差した initializer を拒否する", () => {
  const graph = baseGraph();
  graph.values["w"] = { dtype: "i32", shape: [4, 3] };
  assertThrows(() => openModel(baseModelBuffer(graph)), IrError, "組めない");
});

Deno.test("openModel: scale テンソルの欠落を拒否する", () => {
  const graph = baseGraph();
  graph.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale" };
  const buffer = baseModelBuffer(graph, [
    { name: "enc.w", dtype: "I8", shape: [4, 3], data: i8Bytes(12) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
  ]);
  assertThrows(() => openModel(buffer), ContainerError, "scale テンソル");
});

// 逆向き（実体 → 宣言）の検査。宣言側の走査だけでは「使われなくなった重みが配布形に残って
// いる」形が素通りし、黙って太った配布形がロード時に検出されないまま公開されうる。
Deno.test("openModel: どの initializer からも参照されない余剰テンソルを全件列挙して拒否する", () => {
  const buffer = baseModelBuffer(baseGraph(), [
    { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0.5)) },
    { name: "enc.w_old", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(1)) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
    { name: "enc.dead", dtype: "F32", shape: [2], data: f32Bytes([1, 2]) },
  ]);
  const error = assertThrows(
    () => openModel(buffer),
    ContainerError,
    "参照されないテンソル (2)",
  );
  // 列挙は名前順（1 件ずつ落とすと、削る側が何本余っているのか分からない）
  assertEquals(error.message.includes("enc.dead, enc.w_old"), true, error.message);
});

// scale は IR の値ではなく safetensors の**素のテンソル**なので、参照集合は initializer の
// tensor だけでは足りない（storage.scale を数え落とすと量子化モデルが全て余剰判定で落ちる）。
Deno.test("openModel: 量子化 initializer の scale テンソルは余剰扱いしない", () => {
  const graph = baseGraph();
  graph.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale" };
  const buffer = baseModelBuffer(graph, [
    { name: "enc.w", dtype: "I8", shape: [4, 3], data: i8Bytes(12) },
    { name: "enc.w.scale", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
  ]);
  assertEquals(openModel(buffer).file.tensors.size, 3);
});

// ADR 0069 決定 3: group 量子化の scale は「重みと同 rank・最終次元だけ group 数」で、
// per-channel の keepdim broadcast 形とは受理集合が交わらない別分岐。
Deno.test("openModel: 格納 i4 は group 形の scale を受理する", () => {
  const model = openModel(i4Model());
  assertEquals(model.graph.initializers["w"].storage.groupSize, 32);
  assertEquals(model.file.tensors.get("enc.w")?.byteLength, 128);
  assertEquals(model.file.tensors.get("enc.w.scale")?.shape, [4, 2]);
});

Deno.test("openModel: 格納 i4 の scale が group 形でないものを拒否する", () => {
  // rank 違い（group 数だけの 1 次元）
  assertThrows(
    () =>
      openModel(i4Model(({ tensors }) => {
        tensors[1] = {
          name: "enc.w.scale",
          dtype: "F32",
          shape: [8],
          data: f32Bytes(new Array(8).fill(1)),
        };
      })),
    ContainerError,
    "rank",
  );
  // group 数違い（per-channel の keepdim broadcast 形 — i8 なら通る形）
  assertThrows(
    () =>
      openModel(i4Model(({ tensors }) => {
        tensors[1] = {
          name: "enc.w.scale",
          dtype: "F32",
          shape: [4, 1],
          data: f32Bytes([1, 1, 1, 1]),
        };
      })),
    ContainerError,
    "group 形",
  );
  // F32 以外（別 dtype のビット列として読むと全 group が桁違いの値になる）
  assertThrows(
    () =>
      openModel(i4Model(({ tensors }) => {
        tensors[1] = {
          name: "enc.w.scale",
          dtype: "F16",
          shape: [4, 2],
          data: new Uint8Array(16),
        };
      })),
    ContainerError,
    "F32 が必要",
  );
});

// ADR 0069 決定 3 の一般化（波 J-5b）: group scale は **rank 非依存の rank 2**
// `[shape[0], (numel / shape[0]) / g]`。rank 2 の重みでは従来の「同 rank・最終次元だけ
// group 数」と同値なので、既存資産の検査結果は 1 件も変わらない（上の [4,64] → [4,2] が
// その回帰）。conv1d の `[Cout,Cin,K]` → `[Cout, (Cin·K)/g]` が唯一の新形。
Deno.test("openModel: rank 3 の i4 重み（conv1d）は rank 2 の group scale を受理する", () => {
  const conv = (
    mutate: (parts: { tensors: TensorSpec[] }) => void = () => {},
  ): ArrayBuffer => {
    const graph = baseGraph();
    // 重み `[4,8,2]`（行長 16 = g16 が 1 つ）。消費側 op は結合検証の対象外なので、
    // 見るのは格納の宣言と実テンソルの突合だけ。
    graph.values["w"] = { dtype: "f32", shape: [4, 8, 2] };
    graph.initializers["w"].storage = { dtype: "i4", scale: "enc.w.scale", group_size: 16 };
    const tensors: TensorSpec[] = [
      { name: "enc.w", dtype: "I4", shape: [4, 8, 2], data: new Uint8Array(32) },
      { name: "enc.w.scale", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
      { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
    ];
    mutate({ tensors });
    return baseModelBuffer(graph, tensors);
  };
  assertEquals(openModel(conv()).file.tensors.get("enc.w.scale")?.shape, [4, 1]);
  // 重みと同 rank（旧規則の形）は落ちる — group 形は rank に依らず rank 2
  assertThrows(
    () =>
      openModel(conv(({ tensors }) => {
        tensors[1] = {
          name: "enc.w.scale",
          dtype: "F32",
          shape: [4, 8, 1],
          data: f32Bytes(new Array(32).fill(1)),
        };
      })),
    ContainerError,
    "rank",
  );
  // 行長の割り方が違う形（`K / g` を取った `[4, 2]` 等）も落ちる
  assertThrows(
    () =>
      openModel(conv(({ tensors }) => {
        tensors[1] = {
          name: "enc.w.scale",
          dtype: "F32",
          shape: [4, 2],
          data: f32Bytes(new Array(8).fill(1)),
        };
      })),
    ContainerError,
    "group 形",
  );
});

Deno.test("assertRuntimeSupport: 非対応 op を列挙して落とす", () => {
  const graph = baseGraph();
  graph.requires.ops = ["matmul", "gelu", "tanh"];
  graph.nodes[1] = { op: "gelu", ins: ["h"], outs: ["y"], attrs: {} };
  graph.nodes.push({ op: "tanh", ins: ["y"], outs: ["z"], attrs: {} });
  graph.values["z"] = { dtype: "f32", shape: ["T", 3] };
  const model = openModel(baseModelBuffer(graph));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, M0_SUPPORT),
    ContainerError,
    "非対応 op (2)",
  );
  assertEquals(error.message.includes("gelu, tanh"), true, error.message);
});

// op 名だけの突合は「対応表にはあるのに実行時に落ちる」を作る（recon §3-9）。
Deno.test("assertRuntimeSupport: 対応 op でも実行できない意味論 dtype を宣言ごとに列挙する", () => {
  const graph = baseGraph();
  graph.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
  graph.values["h"] = { dtype: "bool", shape: ["T", 3] };
  const model = openModel(baseModelBuffer(graph));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, M0_SUPPORT),
    ContainerError,
    // 件数は「直すべき宣言の本数」— h は matmul の outs と add の ins の 2 箇所に現れるが 1 件
    "非対応 意味論 dtype (2)",
  );
  assertEquals(error.message.includes("値 'x': i32"), true, error.message);
  assertEquals(error.message.includes("値 'h': bool"), true, error.message);
});

// エイリアス入力（同じ値を 2 回取るノード）で件数が水増しされないこと。
Deno.test("assertRuntimeSupport: 同一宣言の dtype 違反を重複列挙しない", () => {
  const graph = baseGraph();
  graph.values["h"] = { dtype: "bool", shape: ["T", 3] };
  graph.values["y"] = { dtype: "f32", shape: ["T", 3] };
  graph.nodes[1] = { op: "add", ins: ["h", "h"], outs: ["y"], attrs: {} };
  const model = openModel(baseModelBuffer(graph));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, M0_SUPPORT),
    ContainerError,
    "非対応 意味論 dtype (1)",
  );
  assertEquals(error.message.includes("値 'h': bool"), true, error.message);
});

// ノード起点の突合だけでは、どのノードも消費しない入力の dtype 違反が門を素通りする
// （実行器は全 graph.inputs を転送するので、転送層の制約は使用の有無と無関係に実在する）。
Deno.test("assertRuntimeSupport: どのノードも使わない入力の dtype 違反も列挙する", () => {
  const graph = baseGraph();
  graph.inputs.push({ name: "z", dtype: "i32", shape: ["T"] });
  const model = openModel(baseModelBuffer(graph));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, M0_SUPPORT),
    ContainerError,
    "非対応 意味論 dtype (1)",
  );
  assertEquals(error.message.includes("値 'z': i32"), true, error.message);
});

// スロット別 dtype 契約（gather / embedding / masked_fill）は、受理集合の**和**で突き合わせると
// 「値と添字を逆に渡した形」がどちらも和に入るため列挙門を素通りする。契約検査（plan.ts）まで
// 落ちて 1 件ずつ止まると、「非対応は全件列挙して一度に見せる」という門の意図が壊れる。
Deno.test("assertRuntimeSupport: スロットを取り違えた perSlot op を列挙する", () => {
  const graph = baseGraph();
  graph.requires = { ops: ["gather"] };
  // 値 f32 と添字 i32 を**逆に**渡した形。和（{f32, i32}）だけの突合では両方通ってしまう。
  graph.inputs = [
    { name: "src", dtype: "i32", shape: ["T", 4] },
    { name: "idx", dtype: "f32", shape: ["T", 3] },
  ];
  graph.outputs = ["y"];
  graph.initializers = {};
  graph.values = { y: { dtype: "f32", shape: ["T", 3] } };
  graph.nodes = [{ op: "gather", ins: ["src", "idx"], outs: ["y"], attrs: {} }];
  const model = openModel(baseModelBuffer(graph, []));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, RUNTIME_SUPPORT),
    ContainerError,
    "非対応 意味論 dtype (2)",
  );
  assertEquals(error.message.includes("値 'src': i32"), true, error.message);
  assertEquals(error.message.includes("値 'idx': f32"), true, error.message);
});

// 出力もスロット 0（値の側）と同型でなければ実行できない — 和で見ると gather の i32 出力が通る。
Deno.test("assertRuntimeSupport: perSlot op の出力 dtype も値の側の受理集合で見る", () => {
  const graph = baseGraph();
  graph.requires = { ops: ["gather"] };
  graph.inputs = [
    { name: "src", dtype: "f32", shape: ["T", 4] },
    { name: "idx", dtype: "i32", shape: ["T", 3] },
  ];
  graph.outputs = ["y"];
  graph.initializers = {};
  graph.values = { y: { dtype: "i32", shape: ["T", 3] } };
  graph.nodes = [{ op: "gather", ins: ["src", "idx"], outs: ["y"], attrs: {} }];
  const model = openModel(baseModelBuffer(graph, []));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, RUNTIME_SUPPORT),
    ContainerError,
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
Deno.test("assertRuntimeSupport: 多出力 op の出力 slot を取り違えた形を両宣言とも列挙する", () => {
  const graph = baseGraph();
  graph.requires = { ops: ["topk"] };
  graph.inputs = [{ name: "x", dtype: "f32", shape: ["T", 4] }];
  graph.outputs = ["v", "i"];
  graph.initializers = {};
  // slot 0 は値（f32）・slot 1 は添字（i32）なので、この 2 本は**どちらも**非対応
  graph.values = {
    v: { dtype: "i32", shape: ["T", 2] },
    i: { dtype: "f32", shape: ["T", 2] },
  };
  graph.nodes = [{ op: "topk", ins: ["x"], outs: ["v", "i"], attrs: { k: 2 } }];
  const model = openModel(baseModelBuffer(graph, []));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, RUNTIME_SUPPORT),
    ContainerError,
    "非対応 意味論 dtype (2)",
  );
  assertEquals(error.message.includes("値 'v': i32"), true, error.message);
  assertEquals(error.message.includes("値 'i': f32"), true, error.message);
});

Deno.test("assertRuntimeSupport: 対応 op に付いた未実装 attrs をノードごとに列挙する", () => {
  const graph = baseGraph();
  graph.nodes[1].attrs = { alpha: 1, approximate: "tanh" };
  const model = openModel(baseModelBuffer(graph));

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, M0_SUPPORT),
    ContainerError,
    "未実装 attrs (1)",
  );
  assertEquals(error.message.includes("nodes[1] (add): alpha, approximate"), true, error.message);
});

Deno.test("assertRuntimeSupport: 非対応の格納 dtype を initializer 名つきで列挙する", () => {
  const graph = baseGraph();
  graph.initializers["w"].storage = { dtype: "f16" };
  graph.initializers["b"].storage = { dtype: "bf16" };
  const buffer = baseModelBuffer(graph, [
    { name: "enc.w", dtype: "F16", shape: [4, 3], data: new Uint8Array(24) },
    { name: "enc.b", dtype: "BF16", shape: [3], data: new Uint8Array(6) },
  ]);
  const model = openModel(buffer);

  const error = assertThrows(
    () => assertRuntimeSupport(model.graph, M0_SUPPORT),
    ContainerError,
    "capability 不足",
  );
  assertEquals(error.message.includes("'bf16' (1): b"), true, error.message);
  assertEquals(error.message.includes("'f16' (1): w"), true, error.message);
});

// group 量子化を受理する格納は i4 だけ（ADR 0069 決定 2）。他の格納 dtype に付いた group_size は
// 実行経路が無く、黙って無視すると group ごとの scale を per-channel として読む沈黙誤値になる。
Deno.test("assertRuntimeSupport: group_size は i4 だけが通り、他の格納 dtype では落ちる", () => {
  assertRuntimeSupport(openModel(i4Model()).graph, {
    ...M0_SUPPORT,
    storage: new Set<IrStorageDtype>(["f32", "i4"]),
  });

  const graph = baseGraph();
  graph.initializers["w"].storage = { dtype: "i8", scale: "enc.w.scale", group_size: 32 };
  const model = openModel(baseModelBuffer(graph, [
    { name: "enc.w", dtype: "I8", shape: [4, 3], data: i8Bytes(12) },
    { name: "enc.w.scale", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
  ]));
  const error = assertThrows(
    () =>
      assertRuntimeSupport(model.graph, {
        ...M0_SUPPORT,
        storage: new Set<IrStorageDtype>(["f32", "i8"]),
      }),
    ContainerError,
    "非対応 group 量子化 (1): w",
  );
  assertEquals(error.message.includes("i4 のみ"), true, error.message);
});

// ADR 0010: 生の int32 格納は safetensors 側の I32 と 1 対 1（f32 の符号化語彙とは別系統）。
Deno.test("openModel: 意味論 i32 の initializer は I32 テンソルと突合される", () => {
  const graph = baseGraph();
  graph.requires = { ops: ["sym_prefix_slice"] };
  graph.inputs = [{ name: "x", dtype: "f32", shape: ["T"] }];
  graph.outputs = ["y"];
  graph.initializers = { table: { tensor: "enc.table", storage: { dtype: "i32" } } };
  graph.values = {
    table: { dtype: "i32", shape: [4, 3] },
    y: { dtype: "i32", shape: ["T", 3] },
  };
  graph.nodes = [{
    op: "sym_prefix_slice",
    ins: ["table"],
    outs: ["y"],
    attrs: { sym: "T", slices: [{ dim: 0, coeff: 1, offset: 0 }] },
  }];
  const i32Tensor = { name: "enc.table", dtype: "I32", shape: [4, 3], data: new Uint8Array(48) };
  assertEquals(openModel(baseModelBuffer(graph, [i32Tensor])).graph.values["table"].dtype, "i32");

  // 格納 dtype と safetensors の実 dtype が食い違う形は受理しない（要素は同じ 4 バイト）
  assertThrows(
    () =>
      openModel(
        baseModelBuffer(graph, [{ ...i32Tensor, dtype: "F32", data: f32Bytes(new Array(12)) }]),
      ),
    ContainerError,
    "格納 dtype",
  );
});

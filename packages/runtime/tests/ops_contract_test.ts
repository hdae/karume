import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import type { OpSupport } from "../src/format/container.ts";
import type { IrNode } from "../src/format/ir.ts";
import {
  arityFits,
  assertDtype,
  assertNodeContract,
  assertSlotDtype,
  attrKeysOf,
  BINARY_OPS,
  broadcastShapes,
  capabilities,
  castTargetDtype,
  computeOutputShape,
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  cumsumDim,
  describeArity,
  IO_DTYPES,
  layerNormAttrs,
  maskedFillValue,
  OP_CONTRACTS,
  OpContractError,
  optionalAttrKeysOf,
  outputCountOf,
  permuteDims,
  REDUCE_OPS,
  resolveNodeDtypes,
  resolveOpContract,
  rmsNormEps,
  RUNTIME_SUPPORT,
  SCALAR_PARAM_ATTRS,
  scalarParamValues,
  softmaxDim,
  UNARY_OPS,
  WEIGHT_CHANNEL_AXES,
  WEIGHT_SLOTS,
} from "../src/ops.ts";

const node = (
  op: string,
  ins: readonly string[],
  attrs: Record<string, unknown> = {},
  states: Record<string, string> = {},
): IrNode => ({
  op,
  ins: [...ins],
  outs: ["y"],
  attrs,
  states,
});

// MUST: doc は読まない（書式依存の抽出突合は脆く、恒真化の温床にもなる）。ここは**期待値
// リテラル**で op 集合を固定するだけで、docs/ir-v1.md の一覧との同期は op 追加時の人手仕事
// （契約 1 セット — ops.ts / ops.py / shapes.py / fixtures / CPU 参照 / golden / ir-v1.md）。
Deno.test("契約表の op 集合が期待値リテラル 60 本と一致する", () => {
  assertEquals(UNARY_OPS.length, 19);
  assertEquals(BINARY_OPS.length, 6);
  // argmax は reduce 族に**入らない**（attrs も出力 dtype も rank の扱いも別 — ADR 0068 決定 2）
  assertEquals(REDUCE_OPS.length, 3);
  assertEquals(OP_CONTRACTS.size, 60);
  assertEquals([...OP_CONTRACTS.keys()].sort(), [
    "abs",
    "add",
    "amax",
    "amin",
    "argmax",
    "attention",
    "bitwise_and",
    "bitwise_not",
    "bmm",
    "cast",
    "cat",
    "clamp",
    "clamp_min",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "cumsum",
    "deform_conv2d",
    "div",
    "embedding",
    "exp",
    "expand",
    "flip",
    "gather",
    "ge",
    "ge_scalar",
    "gelu",
    "gelu_tanh",
    "gru_scan",
    "gru_scan_reverse",
    "gt_scalar",
    "layer_norm",
    "le_scalar",
    "leaky_relu",
    "linear",
    "log",
    "log1p",
    "masked_fill",
    "matmul",
    "mul",
    "neg",
    "pad",
    "permute",
    "relu",
    "reshape",
    "rms_norm",
    "safe_softmax",
    "sigmoid",
    "sin",
    "slice",
    "softmax",
    "sqrt",
    "state_append",
    "sub",
    "sum",
    "sym_prefix_slice",
    "tanh",
    "topk",
    "upsample_bilinear2d",
    "where",
  ]);
});

// 重みスロット表とチャネル軸表は対（ADR 0019）。片方だけ増えると、新しいスロットの i8 が
// 「軸 0 の scale」として黙って実行される（shape 検査は軸 0 と軸 1 の区別を付けられない形が
// 作れる）。
Deno.test("WEIGHT_CHANNEL_AXES は WEIGHT_SLOTS と同じ op を覆う", () => {
  assertEquals([...WEIGHT_CHANNEL_AXES.keys()].sort(), [...WEIGHT_SLOTS.keys()].sort());
  // 期待値リテラルで固定する（表を辿り直すと構造が壊れない限り絶対に落ちない検査になる）
  assertEquals([...WEIGHT_CHANNEL_AXES].sort(), [
    ["conv1d", 0],
    ["conv2d", 0],
    ["conv_transpose1d", 1],
    ["embedding", 0],
    ["linear", 0],
  ]);
});

Deno.test("ランタイム対応表と capability 照会は契約表から導かれる", () => {
  assertEquals(RUNTIME_SUPPORT.ops.size, OP_CONTRACTS.size);
  // 生の int32 格納は記号依存定数の焼き込み先として実行対象（ADR 0010）。f16（ADR 0018）と
  // i8（ADR 0019）は実行経路が入った（適格な重みは圧縮のまま常駐・適格外は CPU 展開）。
  // bf16 だけが宣言として valid なまま実行できない。
  assertEquals([...RUNTIME_SUPPORT.storage], ["f32", "f16", "i8", "i32"]);
  // 転送層の軸（未使用のグラフ入力にも効く）— i32 / bool も転送できる（ADR 0009）
  assertEquals([...RUNTIME_SUPPORT.io].sort(), ["bool", "f32", "i32"]);
  assertEquals([...IO_DTYPES].sort(), ["bool", "f32", "i32"]);
  assertEquals(capabilities().ops, [...OP_CONTRACTS.keys()].sort());
  assertEquals(capabilities().storage, ["f16", "f32", "i32", "i8"]);
  // 射影の中身は**期待値リテラル**で固定する。契約表を辿り直して同じ式で突き合わせると、
  // 導出元が同一オブジェクトなので構造が壊れない限り絶対に落ちない検査になる。
  const support = (op: string): OpSupport => {
    const found = RUNTIME_SUPPORT.ops.get(op);
    assert(found !== undefined, `対応表に op '${op}' が無い`);
    return found;
  };
  const slots = (op: string): string[][] =>
    support(op).slotDtypes.map((accept) => [...accept].sort());

  // スロット別契約の 3 本（値の側と添字 / 条件で受理集合が違う）
  assertEquals(slots("gather"), [["f32"], ["i32"]]);
  assertEquals(slots("embedding"), [["f32"], ["i32"]]);
  assertEquals(slots("masked_fill"), [["f32"], ["bool"]]);
  // uniform 契約はアリティぶん同じ集合が並ぶ（消費側がスロット番号で引けること）
  assertEquals(slots("mul"), [["f32", "i32"], ["f32", "i32"]]);
  assertEquals(slots("relu"), [["f32"]]);
  assertEquals(slots("conv1d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv1d").attrKeys].sort(), [
    "dilation",
    "groups",
    "padding",
    "stride",
  ]);
  assertEquals(slots("conv2d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv2d").attrKeys].sort(), [
    "dilation",
    "groups",
    "padding",
    "stride",
  ]);
  // rms_norm はアリティ 2（bias が無い）— 射影のスロット数がそのまま契約面になる
  assertEquals(slots("rms_norm"), [["f32"], ["f32"]]);
  assertEquals([...support("rms_norm").attrKeys], ["eps"]);
  assertEquals([...support("clamp_min").attrKeys], ["min"]);
  assertEquals(slots("conv_transpose1d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv_transpose1d").attrKeys].sort(), ["padding", "stride"]);
  assertEquals([...support("cast").attrKeys], ["to"]);
  assertEquals([...support("relu").attrKeys], []);

  // 射影どうしの関係（和 = スロット別の合併 / スロット数 = アリティ）は契約表を辿り直さずに
  // 検査できる真の不変条件なので、全 op に対して回す。
  for (const [name, contract] of OP_CONTRACTS) {
    const found = support(name);
    // スロット数は「受理しうる入力の最大本数」= 省略可能な末尾入力（attention の mask）を
    // 含む（可変アリティの cat だけは下限ぶん — 全スロットが同じ受理集合なので列挙門の
    // 和の見方と結論が一致する）。
    assertEquals(
      found.slotDtypes.length,
      contract.maxArity ?? contract.arity,
      `${name}: スロット数 = 受理しうる最大アリティ`,
    );
    const union = new Set(found.slotDtypes.flatMap((accept) => [...accept]));
    assertEquals([...union].sort(), [...found.dtypes].sort(), `${name}: 和 = スロットの合併`);
  }
});

// MUST: 一括解禁しない（ADR 0007 の語彙 allowlist 凍結）。i32 / bool を持つのは実測グラフの
// mask 経路に出る形（mul / sub）と、その経路のために新設した op（cast / bitwise_not）だけ。
Deno.test("dtype の解禁は op ごとで、実測に出ない組み合わせは f32 のまま", () => {
  const dtypes = (op: string): readonly string[] => [...resolveOpContract(op).dtypes].sort();
  assertEquals(dtypes("mul"), ["f32", "i32"]);
  assertEquals(dtypes("sub"), ["f32", "i32"]);
  assertEquals(dtypes("bitwise_not"), ["bool"]);
  assertEquals(dtypes("cast"), ["bool", "f32", "i32"]);
  // レイアウト op（ADR 0011）も実測どおり: reshape は要素に触れないので全語彙、expand は
  // gather 添字（i32）と conv の bool マスクだけ、permute は f32 の attention 整形だけ。
  assertEquals(dtypes("reshape"), ["bool", "f32", "i32"]);
  // 波3 で f32 を追加（相対位置埋め込みの 4D 化 — recon §2）。strided コピー族は dtype
  // パラメトリックなのでカーネルは共用のまま。
  assertEquals(dtypes("expand"), ["bool", "f32", "i32"]);
  assertEquals(dtypes("permute"), ["f32"]);
  // 波3 の数理 op: 比較 4 本と where は f32 を読み、bitwise_and は bool 専業、
  // sum だけが f32 と bool の 2 形を持つ（出力は f32 / i32）。
  assertEquals(dtypes("ge"), ["f32"]);
  assertEquals(dtypes("ge_scalar"), ["f32"]);
  assertEquals(dtypes("bitwise_and"), ["bool"]);
  assertEquals(dtypes("sum"), ["bool", "f32"]);
  assertEquals(dtypes("where"), ["bool", "f32"]);
  // 融合 op（ADR 0012）は値が f32 専業で、添字（embedding）と条件（masked_fill）だけが
  // スロット別に整数 / bool を受ける。
  assertEquals(dtypes("embedding"), ["f32", "i32"]);
  assertEquals(dtypes("masked_fill"), ["bool", "f32"]);
  for (
    const op of [
      "add",
      "div",
      "relu",
      "gelu",
      "gelu_tanh",
      "sin",
      "matmul",
      "bmm",
      "amax",
      "amin",
      "log1p",
      "clamp",
      "clamp_min",
      "leaky_relu",
      "cumsum",
      "permute",
      "linear",
      "layer_norm",
      "rms_norm",
      "softmax",
      "safe_softmax",
      "conv1d",
      "conv2d",
      "conv_transpose1d",
    ]
  ) {
    assertEquals(dtypes(op), ["f32"], op);
  }
  assertThrows(() => assertDtype(resolveOpContract("amax"), "bool", "t"), OpContractError);
  assertThrows(() => assertDtype(resolveOpContract("permute"), "i32", "t"), OpContractError);
  assertThrows(() => assertDtype(resolveOpContract("relu"), "i32", "t"), OpContractError);
  assertThrows(() => assertDtype(resolveOpContract("add"), "i32", "t"), OpContractError);
  assertThrows(() => assertDtype(resolveOpContract("bitwise_not"), "f32", "t"), OpContractError);
});

// ADR 0012: gather は入力スロットごとに受理集合が違う最初の op（src=f32 / index=i32）。
// MUST: 「受理集合の和」だけを見る検査（assertDtype）と、スロット単位の検査を混同しない。
Deno.test("gather はスロット別 dtype 契約を持ち、和との突合より狭く判定する", () => {
  const gather = resolveOpContract("gather");
  assertEquals(gather.slotDtypes.kind, "perSlot");
  // 和は capability 射影に出る（対応表は「この op が触れる dtype」を運ぶ）
  assertEquals([...gather.dtypes], ["f32", "i32"]);
  assertEquals([...(RUNTIME_SUPPORT.ops.get("gather")?.dtypes ?? [])], ["f32", "i32"]);
  // スロット単位ではそれぞれ 1 種類しか受けない
  assertSlotDtype(gather, 0, "f32", "t");
  assertSlotDtype(gather, 1, "i32", "t");
  assertThrows(() => assertSlotDtype(gather, 0, "i32", "t"), OpContractError);
  assertThrows(() => assertSlotDtype(gather, 1, "f32", "t"), OpContractError);
  assertThrows(() => assertSlotDtype(gather, 1, "bool", "t"), OpContractError);
  // 存在しないスロット（アリティ違反は assertNodeContract が先に落とすが、契約自体も拒否する）
  assertThrows(() => assertSlotDtype(gather, 2, "f32", "t"), OpContractError);

  // uniform 契約ではスロット別検査は和との突合と同義（既存 op の判定は変わらない）
  const mul = resolveOpContract("mul");
  assertEquals(mul.slotDtypes.kind, "uniform");
  assertSlotDtype(mul, 1, "i32", "t");
  assertThrows(() => assertSlotDtype(mul, 1, "bool", "t"), OpContractError);
});

Deno.test("スロット別契約の出力はスロット 0 と同型で、混合は uniform の op だけが拒否する", () => {
  const resolve = (
    op: string,
    inputDtypes: readonly ("f32" | "i32" | "bool")[],
    declared: "f32" | "i32" | "bool",
  ) =>
    resolveNodeDtypes(
      resolveOpContract(op),
      node(op, inputDtypes.map((_, index) => `x${index}`)),
      inputDtypes,
      [declared],
      "t",
    )[0];

  // gather は f32 と i32 が混ざるのが正しい形（uniform の「混在は拒否」を適用しない）
  assertEquals(resolve("gather", ["f32", "i32"], "f32"), "f32");
  // スロットの取り違え（src と index を逆に渡した形）は落ちる
  assertThrows(() => resolve("gather", ["i32", "f32"], "f32"), OpContractError);
  // 出力の宣言がスロット 0 と違えば落ちる
  assertThrows(() => resolve("gather", ["f32", "i32"], "i32"), OpContractError);
  // uniform 側の混在拒否は温存（perSlot に潰していない）
  assertThrows(() => resolve("mul", ["i32", "f32"], "i32"), OpContractError);
  assertEquals(resolve("bmm", ["f32", "f32"], "f32"), "f32");
});

// attrs を持つ op は 30 本。それ以外は attrs 空のままで、非空 attrs は fail loudly。
Deno.test("attrs を持つ op は契約表が列挙する 30 本だけで、他は attrs 空", () => {
  const withAttrs = [...OP_CONTRACTS]
    .filter(([, contract]) => attrKeysOf(contract).length > 0)
    .map(([name]) => name)
    .sort();
  assertEquals(withAttrs, [
    "amax",
    "amin",
    "attention",
    "cast",
    "cat",
    "clamp",
    "clamp_min",
    "conv1d",
    "conv2d",
    "conv_transpose1d",
    "cumsum",
    "deform_conv2d",
    "embedding",
    "flip",
    "ge_scalar",
    "gt_scalar",
    "layer_norm",
    "le_scalar",
    "leaky_relu",
    "masked_fill",
    "pad",
    "permute",
    "rms_norm",
    "safe_softmax",
    "slice",
    "softmax",
    "sum",
    "sym_prefix_slice",
    "topk",
    "upsample_bilinear2d",
  ]);
  assertThrows(() => assertNodeContract(node("relu", ["a"], { alpha: 1 }), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("relu", ["a", "b"]), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("add", ["a"]), "t"), OpContractError);
  assertThrows(
    () => assertNodeContract({ ...node("relu", ["a"]), outs: [] }, "t"),
    OpContractError,
  );
  // 融合 op のアリティは固定（bias / affine 無しの形は語彙に無い — ADR 0012）
  assertThrows(() => assertNodeContract(node("linear", ["x", "w"]), "t"), OpContractError);
  assertThrows(
    () =>
      assertNodeContract(
        node("conv1d", ["x", "w"], { stride: 1, padding: 1, dilation: 1, groups: 1 }),
        "t",
      ),
    OpContractError,
  );
  // bias 無し conv はエクスポータのゼロ bias 合成で正規化される（ADR 0015）— 契約側は
  // アリティ 3 固定のまま
  assertThrows(
    () => assertNodeContract(node("conv_transpose1d", ["x", "w"], { stride: 2, padding: 0 }), "t"),
    OpContractError,
  );
  // rms_norm はアリティ 2 固定（weight 無しはエクスポータの ones 合成で正規化 — ADR 0017）
  assertThrows(
    () => assertNodeContract(node("rms_norm", ["x"], { eps: 1e-6 }), "t"),
    OpContractError,
  );
  assertThrows(
    () => assertNodeContract(node("rms_norm", ["x", "w", "b"], { eps: 1e-6 }), "t"),
    OpContractError,
  );
  assertThrows(
    () =>
      assertNodeContract(
        node("conv2d", ["x", "w"], {
          stride: [1, 1],
          padding: [0, 0],
          dilation: [1, 1],
          groups: 1,
        }),
        "t",
      ),
    OpContractError,
  );
  // NOTE: conv2d は ADR 0017 で契約表へ入った（M1-P2 まで「保存リストにあるがカーネルが
  // 無い op」の代表だった）。語彙にすら無い op の代表は conv_transpose2d が引き継ぐ。
  assertThrows(() => resolveOpContract("conv_transpose2d"), OpContractError);
});

// deform_conv2d は **DCNv2 専業・stride/dilation/groups/offset_groups は 1 固定**（第 1' 層・
// ADR 0055）。どちらも値の検査ではなく**欄とスロットの不存在**で表しているので、それらが
// 生えていないことを契約側から固定する。
Deno.test("deform_conv2d の attrs は padding 1 本だけ・アリティ 5 固定", () => {
  assertEquals(attrKeysOf(resolveOpContract("deform_conv2d")), ["padding"]);
  assertEquals(
    assertNodeContract(
      node("deform_conv2d", ["x", "w", "off", "mask", "b"], { padding: [1, 0] }),
      "t",
    ).kind,
    "deformConv2d",
  );
  // stride / dilation / groups / offset_groups の欄は無い = 宣言外キーとして落ちる
  for (
    const extra of [
      { stride: [1, 1] },
      { dilation: [1, 1] },
      { groups: 1 },
      { offset_groups: 1 },
    ]
  ) {
    assertThrows(
      () =>
        assertNodeContract(
          node("deform_conv2d", ["x", "w", "off", "mask", "b"], { padding: [1, 0], ...extra }),
          "t",
        ),
      OpContractError,
    );
  }
  // padding は宣言必須（既定値補完なし）
  assertThrows(
    () => assertNodeContract(node("deform_conv2d", ["x", "w", "off", "mask", "b"]), "t"),
    OpContractError,
  );
  // mask を落とした DCNv1 も、余分なスロットも受理しない（アリティが絞りそのもの）
  for (
    const ins of [
      ["x", "w", "off", "b"],
      ["x", "w", "off", "mask", "b", "extra"],
    ]
  ) {
    assertThrows(
      () => assertNodeContract(node("deform_conv2d", ins, { padding: [1, 0] }), "t"),
      OpContractError,
    );
  }
});

// upsample_bilinear2d は **align_corners=True 専業**（第 1 層）。「True 以外を受理しない」を
// 値の検査ではなく**欄の不存在**で表しているので、その欄が生えていないことを契約側から固定
// する（ADR 0023 決定 4 の規律 — 欄ができた瞬間に False も表現できてしまう）。
Deno.test("upsample_bilinear2d の attrs は output_size 1 本だけ（align_corners の欄が無い）", () => {
  assertEquals(attrKeysOf(resolveOpContract("upsample_bilinear2d")), ["output_size"]);
  assertEquals(
    assertNodeContract(node("upsample_bilinear2d", ["x"], { output_size: [7, 9] }), "t").kind,
    "upsampleBilinear2d",
  );
  // align_corners / mode / scale_factor の欄は無い = 宣言外キーとして落ちる
  for (const extra of [{ align_corners: true }, { mode: "bilinear" }, { scale_factor: [2, 2] }]) {
    assertThrows(
      () =>
        assertNodeContract(
          node("upsample_bilinear2d", ["x"], { output_size: [7, 9], ...extra }),
          "t",
        ),
      OpContractError,
    );
  }
  // 出力空間の宣言は必須（既定値補完なし）
  assertThrows(
    () => assertNodeContract(node("upsample_bilinear2d", ["x"]), "t"),
    OpContractError,
  );
  // アリティ 1 固定（重みも bias も持たない）
  assertThrows(
    () => assertNodeContract(node("upsample_bilinear2d", ["x", "w"], { output_size: [7, 9] }), "t"),
    OpContractError,
  );
});

// ADR 0012: attrs スキーマは「宣言キーは全て必須・宣言外は fail loudly・値も検査する」。
Deno.test("attrs スキーマは必須キーの欠落と契約外の値を拒否する（cast の to）", () => {
  assertEquals(attrKeysOf(resolveOpContract("cast")), ["to"]);
  assertEquals(assertNodeContract(node("cast", ["a"], { to: "i32" }), "t").kind, "cast");
  // 必須キーの欠落（M0 の「未知キーだけ見る」検査ではここが素通りした）
  assertThrows(() => assertNodeContract(node("cast", ["a"]), "t"), OpContractError);
  // 値域外・型違い
  assertThrows(() => assertNodeContract(node("cast", ["a"], { to: "i64" }), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("cast", ["a"], { to: 3 }), "t"), OpContractError);
  // 宣言外のキー
  assertThrows(
    () => assertNodeContract(node("cast", ["a"], { to: "f32", from: "i32" }), "t"),
    OpContractError,
  );
  // MUST: スキーマ照合は Object.hasOwn のみ — prototype 由来の名前は既知キーにならない
  assertThrows(
    () => assertNodeContract(node("cast", ["a"], { to: "f32", toString: "x" }), "t"),
    OpContractError,
  );
  assertEquals(castTargetDtype({ to: "bool" }, "t"), "bool");
  assertThrows(() => castTargetDtype({}, "t"), OpContractError);
});

// ADR 0011: permute の dims は非負の全単射（負の軸表記はエクスポータ境界で正規化する）。
Deno.test("permute の dims は非負・重複無しの並べ替え表だけを受理する", () => {
  assertEquals(attrKeysOf(resolveOpContract("permute")), ["dims"]);
  assertEquals(
    assertNodeContract(node("permute", ["a"], { dims: [0, 2, 1] }), "t").kind,
    "permute",
  );
  assertEquals(permuteDims({ dims: [1, 0] }, "t"), [1, 0]);
  assertThrows(() => permuteDims({}, "t"), OpContractError);
  const bad = (dims: unknown) =>
    assertThrows(() => assertNodeContract(node("permute", ["a"], { dims }), "t"), OpContractError);
  bad([0, -1]); // 負の軸表記は受理しない
  bad([0, 0]); // 全単射でない
  bad([0, 1.5]);
  bad([]);
  bad([0, "1"]);
  bad("0,1");
  // reshape / expand は attrs を取らない（目標形は出力の宣言 shape）
  assertThrows(
    () => assertNodeContract(node("reshape", ["a"], { shape: [2, 3] }), "t"),
    OpContractError,
  );
  assertThrows(
    () => assertNodeContract(node("expand", ["a"], { shape: [2, 3] }), "t"),
    OpContractError,
  );
});

// MUST: 出力 dtype は宣言を鵜呑みにせず契約から導く（cast の to と values{} の宣言が
// 食い違ったグラフは、readback で別の TypedArray として読まれる沈黙誤値になる）。
Deno.test("出力 dtype は契約から導かれ、宣言との食い違いを拒否する", () => {
  const resolve = (
    op: string,
    ins: readonly string[],
    inputDtypes: readonly ("f32" | "i32" | "bool")[],
    declared: "f32" | "i32" | "bool",
    attrs: Record<string, unknown> = {},
  ) =>
    resolveNodeDtypes(resolveOpContract(op), node(op, ins, attrs), inputDtypes, [declared], "t")[0];

  assertEquals(resolve("mul", ["a", "b"], ["i32", "i32"], "i32"), "i32");
  assertEquals(resolve("cast", ["a"], ["i32"], "bool", { to: "bool" }), "bool");
  assertEquals(resolve("bitwise_not", ["a"], ["bool"], "bool"), "bool");
  // 宣言が契約と食い違う
  assertThrows(() => resolve("mul", ["a", "b"], ["i32", "i32"], "f32"), OpContractError);
  assertThrows(
    () => resolve("cast", ["a"], ["f32"], "f32", { to: "i32" }),
    OpContractError,
  );
  // 混合型の elementwise は語彙に無い
  assertThrows(() => resolve("mul", ["a", "b"], ["i32", "f32"], "i32"), OpContractError);
  // 入力 dtype が契約外
  assertThrows(() => resolve("add", ["a", "b"], ["i32", "i32"], "i32"), OpContractError);

  // 波3: 入力と出力で dtype が違う 3 系統（契約表の写像から導く）
  assertEquals(resolve("ge", ["a", "b"], ["f32", "f32"], "bool"), "bool");
  assertEquals(resolve("ge_scalar", ["a"], ["f32"], "bool", { value: 0 }), "bool");
  assertEquals(resolve("sum", ["a"], ["f32"], "f32"), "f32");
  // bool の sum は**個数**なので i32（f32 で数えると 2^24 で静かに丸まる）
  assertEquals(resolve("sum", ["a"], ["bool"], "i32"), "i32");
  assertThrows(() => resolve("sum", ["a"], ["bool"], "f32"), OpContractError);
  assertThrows(() => resolve("sum", ["a"], ["bool"], "bool"), OpContractError);
  assertThrows(() => resolve("ge", ["a", "b"], ["f32", "f32"], "f32"), OpContractError);
  // where は条件が先頭スロットでも出力は**値の側**（写像 bool → f32）
  assertEquals(resolve("where", ["c", "a", "b"], ["bool", "f32", "f32"], "f32"), "f32");
  assertThrows(
    () => resolve("where", ["c", "a", "b"], ["bool", "f32", "f32"], "bool"),
    OpContractError,
  );
  // スロットの取り違え（条件と値を逆に渡した形）
  assertThrows(
    () => resolve("where", ["c", "a", "b"], ["f32", "f32", "bool"], "f32"),
    OpContractError,
  );
  assertEquals(resolve("bitwise_and", ["a", "b"], ["bool", "bool"], "bool"), "bool");
  assertThrows(() => resolve("bitwise_and", ["a", "b"], ["f32", "f32"], "f32"), OpContractError);
});

// MUST: 写像の定義域はスロット 0 の受理集合と完全一致（部分写像は「出力が決まらない dtype」
// の穴になる）。恒等が既定であることも同じ不変条件で担保される。
Deno.test("出力 dtype 写像の定義域がスロット 0 の受理集合と一致する", () => {
  for (const [name, contract] of OP_CONTRACTS) {
    const slots = contract.slotDtypes;
    const domain = slots.kind === "uniform" ? slots.accept : slots.slots[0];
    // 定義域の一致は**出力 slot ごと**に要求する（列のどれか 1 本だけ穴があっても落とす）
    contract.outputDtypes.forEach((slot, index) => {
      assertEquals(
        [...slot.keys()].sort(),
        [...domain].sort(),
        `${name}: 出力 ${index} の dtype 写像の定義域`,
      );
    });
  }
  // 恒等でないのは実測に出た 5 系統だけ（比較 4 本 / bool の sum → i32 / where → 値の側 /
  // argmax → 添字 i32 / topk → slot 1 が添字 i32〈slot 0 の値は恒等〉）
  const nonIdentity = [...OP_CONTRACTS]
    .filter(([name, contract]) =>
      name !== "cast" &&
      contract.outputDtypes.some((slot) => [...slot].some(([from, to]) => from !== to))
    )
    .map(([name]) => name)
    .sort();
  assertEquals(nonIdentity, [
    "argmax",
    "ge",
    "ge_scalar",
    "gt_scalar",
    "le_scalar",
    "sum",
    "topk",
    "where",
  ]);
});

// 波3 の数理 op。attrs の値域とキーを跨ぐ不変条件（clamp の min <= max）を固定する。
Deno.test("clamp / leaky_relu / 比較の attrs スキーマが値域まで検査する", () => {
  assertEquals([...attrKeysOf(resolveOpContract("clamp"))].sort(), ["max", "min"]);
  assertEquals(attrKeysOf(resolveOpContract("leaky_relu")), ["negative_slope"]);
  assertEquals(attrKeysOf(resolveOpContract("ge_scalar")), ["value"]);
  assertEquals(
    assertNodeContract(node("clamp", ["a"], { min: -1, max: 1 }), "t").kind,
    "unary",
  );
  // 必須キーの欠落（既定値補完はしない — ADR 0015 / 0012）
  assertThrows(() => assertNodeContract(node("clamp", ["a"], { min: -1 }), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("leaky_relu", ["a"], {}), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("ge_scalar", ["a"], {}), "t"), OpContractError);
  // 値域外
  assertThrows(
    () => assertNodeContract(node("clamp", ["a"], { min: "0", max: 1 }), "t"),
    OpContractError,
  );
  assertThrows(
    () => assertNodeContract(node("leaky_relu", ["a"], { negative_slope: null }), "t"),
    OpContractError,
  );

  // MUST: min <= max は**キーを跨ぐ**ので attrs スキーマでは表せない。shape 計算（全ノードが
  // 通る唯一の共通経路）で落ちること、そこまでの経路で params の値も同じ検査を通ることを固定。
  const clamp = resolveOpContract("clamp");
  assertEquals(scalarParamValues(clamp, { min: -1, max: 2 }, "t"), [-1, 2]);
  assertEquals(scalarParamValues(clamp, { min: 2, max: 2 }, "t"), [2, 2]);
  assertThrows(() => scalarParamValues(clamp, { min: 2, max: 1 }, "t"), OpContractError);
  assertThrows(
    () => computeOutputShape(clamp, [[2, 3]], "t", { attrs: { min: 2, max: 1 } }),
    OpContractError,
  );
  // スカラを持たない op は空（params の末尾に何も載らない）
  assertEquals(scalarParamValues(resolveOpContract("relu"), {}, "t"), []);
  // 比較 3 本と leaky_relu は 1 本ずつ
  assertEquals(scalarParamValues(resolveOpContract("gt_scalar"), { value: 1.5 }, "t"), [1.5]);
  assertEquals(
    scalarParamValues(resolveOpContract("leaky_relu"), { negative_slope: 0.1 }, "t"),
    [0.1],
  );
});

// exporter 側の対称形（`tools/exporter/tests/test_shapes.py` の
// `the_params_table_and_the_attrs_schema_cover_the_same_unary_ops`）。
Deno.test("スカラ params の並び表と単項 attrs スキーマが同じ op を覆う", () => {
  // attrs スキーマ（契約表の単項 op）と params のレイアウト（SCALAR_PARAM_ATTRS）は別々に
  // 書かれた 2 表で、片方だけに op を足しても**例外は出ない**: params 表に無い側は
  // `scalarParamValues` が空リストを返して attr の値が黙ってカーネルへ届かず、attrs スキーマに
  // 無い側は宣言そのものが契約外 attrs 扱いになる。
  const unaryWithAttrs = [...OP_CONTRACTS]
    .filter(([, contract]) => contract.kind === "unary" && attrKeysOf(contract).length > 0)
    .map(([name]) => name)
    .sort();
  assertEquals([...SCALAR_PARAM_ATTRS.keys()].sort(), unaryWithAttrs);
});

// cumsum は softmax と同じ絞り方（attrs.dim を持ち、最終次元だけを受理する）。
Deno.test("cumsum は最終次元固定で、shape を素通しにする", () => {
  const contract = resolveOpContract("cumsum");
  assertEquals(attrKeysOf(contract), ["dim"]);
  assertEquals(cumsumDim({ dim: 2 }, "t"), 2);
  assertThrows(() => cumsumDim({}, "t"), OpContractError);
  assertThrows(() => cumsumDim({ dim: -1 }, "t"), OpContractError);
  const shape = (ins: readonly number[], dim: number) =>
    computeOutputShape(contract, [ins], "t", { attrs: { dim } })[0];
  assertEquals(shape([2, 5], 1), [2, 5]);
  assertEquals(shape([1, 3, 4], 2), [1, 3, 4]);
  // 長さ 0 の軸は素通し（前縁和の identity は 0 — amax / softmax と違って定義できる）
  assertEquals(shape([3, 0], 1), [3, 0]);
  assertThrows(() => shape([2, 5], 0), OpContractError);
  assertThrows(() => shape([], 0), OpContractError);
});

// argmax は reduce 族と**別の契約**（ADR 0068 決定 2）。絞りを「値の検査」ではなく
// **欄の不存在**で表しているので、欄が生えていないことを契約側から固定する。
Deno.test("argmax は attrs 空・最終次元固定・rank 保存で、出力は i32", () => {
  const contract = resolveOpContract("argmax");
  // MUST: `dim` の欄が無い = 他の軸は語彙に無い / `keepdim` の欄が無い = rank 保存の 1 形だけ
  assertEquals(attrKeysOf(contract), []);
  assertEquals(contract.arity, 1);
  assertEquals(contract.kind, "argmax");
  // 出力は添字なので i32（1 本・写像 f32 → i32）
  assertEquals(contract.outputDtypes.length, 1);
  assertEquals(resolveNodeDtypes(contract, node("argmax", ["x"]), ["f32"], ["i32"], "t"), ["i32"]);
  // 宣言が f32 のグラフは拒否する（readback が別の TypedArray として読む沈黙誤値になる）
  assertThrows(
    () => resolveNodeDtypes(contract, node("argmax", ["x"]), ["f32"], ["f32"], "t"),
    OpContractError,
  );
  // i32 / bool 入力は語彙に無い（f32 専業 — 実測は logits だけ）
  assertThrows(() => assertDtype(contract, "i32", "t"), OpContractError);
  const shape = (ins: readonly number[]) => computeOutputShape(contract, [ins], "t")[0];
  assertEquals(shape([7]), [1]);
  assertEquals(shape([6, 10]), [6, 1]);
  assertEquals(shape([2, 3, 4]), [2, 3, 1]);
  // 長さ 0 なのが最終次元でなければ受理（出力が空になるだけ）
  assertEquals(shape([0, 3]), [0, 1]);
  // 契約外 attrs は fail loudly（attrs 空契約 — ADR 0012）
  assertThrows(
    () => assertNodeContract(node("argmax", ["x"], { dim: 1 }), "t"),
    OpContractError,
  );
  assertThrows(() => shape([]), OpContractError);
  // 長さ 0 の最終次元に「最大値の添字」は無い（amax / amin と同じ絞り）
  assertThrows(() => shape([3, 0]), OpContractError);
});

// topk は**唯一の多出力 op**（ADR 0068 決定 3）。受理領域（1 ≤ k ≤ 最終次元）と 2 本の出力の
// dtype / shape が 1 本の契約から出ること、`largest` / `sorted` の欄が生えていないことを固定する。
Deno.test("topk は attrs k 宣言必須・出力 2 本（値 f32 + 添字 i32）で、受理領域は 1 ≤ k ≤ 最終次元", () => {
  const contract = resolveOpContract("topk");
  // MUST: 欄は `k` だけ（`dim` が無い = 最終次元固定 / `largest` `sorted` が無い = 降順の
  // 最大側 1 形だけ）
  assertEquals(attrKeysOf(contract), ["k"]);
  assertEquals(contract.arity, 1);
  assertEquals(contract.kind, "topk");
  // 出力数は写像の列長そのもの（slot 0 = 値の恒等・slot 1 = 添字の i32）
  assertEquals(contract.outputDtypes.length, 2);
  assertEquals(outputCountOf(contract), 2);
  assertEquals(
    resolveNodeDtypes(contract, node("topk", ["x"], { k: 2 }), ["f32"], ["f32", "i32"], "t"),
    ["f32", "i32"],
  );
  // slot を入れ替えた宣言は落とす（どちらも 4 バイト要素・同形なので、通せば readback が
  // 値と添字を入れ替えたまま黙って成功する）
  assertThrows(
    () => resolveNodeDtypes(contract, node("topk", ["x"], { k: 2 }), ["f32"], ["i32", "f32"], "t"),
    OpContractError,
  );
  // 出力 1 本の宣言も落とす（本数は契約が持つ）
  assertThrows(
    () => resolveNodeDtypes(contract, node("topk", ["x"], { k: 2 }), ["f32"], ["f32"], "t"),
    OpContractError,
  );
  assertThrows(
    () => assertNodeContract({ ...node("topk", ["x"], { k: 2 }), outs: ["v"] }, "t"),
    OpContractError,
    "出力数が 1（契約は 2）",
  );
  // 出力 2 本のノード（`node` ヘルパは単一出力形なので、topk はここで組み直す）
  const topkNode = (attrs: Record<string, unknown>): IrNode => ({
    ...node("topk", ["x"], attrs),
    outs: ["v", "i"],
  });
  assertEquals(assertNodeContract(topkNode({ k: 2 }), "t").kind, "topk");
  // i32 / bool 入力は語彙に無い（f32 専業 — 実測は logits だけ）
  assertThrows(() => assertDtype(contract, "i32", "t"), OpContractError);
  const shapes = (ins: readonly number[], k: unknown) =>
    computeOutputShape(contract, [ins], "t", { attrs: { k } });
  // 2 本とも [..., k]（rank 保存・先行次元は素通し）
  assertEquals(shapes([6, 10], 3), [[6, 3], [6, 3]]);
  assertEquals(shapes([4], 4), [[4], [4]]);
  assertEquals(shapes([2, 3, 5], 1), [[2, 3, 1], [2, 3, 1]]);
  // 受理領域の外は全て fail loudly（縮退しない）
  assertThrows(() => shapes([6, 10], 11), OpContractError, "最終次元");
  assertThrows(() => shapes([6, 10], 0), OpContractError);
  assertThrows(() => shapes([6, 10], -1), OpContractError);
  assertThrows(() => shapes([6, 10], 1.5), OpContractError);
  // 記号 k（次元式の文字列）は静的形状の前提に載らない
  assertThrows(() => shapes([6, 10], "T"), OpContractError);
  // k は宣言必須（欄が無い = 既定値補完しない）
  assertThrows(() => computeOutputShape(contract, [[6, 10]], "t"), OpContractError);
  assertThrows(() => assertNodeContract(topkNode({}), "t"), OpContractError, "必須 attr");
  // 契約外 attrs（torch の largest / sorted）は fail loudly
  assertThrows(
    () => assertNodeContract(topkNode({ k: 2, sorted: true }), "t"),
    OpContractError,
    "契約外 attrs",
  );
  assertThrows(() => shapes([], 1), OpContractError);
  // 長さ 0 の最終次元は k >= 1 との突合でそのまま落ちる
  assertThrows(() => shapes([3, 0], 1), OpContractError);
});

Deno.test("where は三者を右詰め broadcast し、出力は値の側と同形になる", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("where"), ins, "t")[0];
  assertEquals(shape([[1, 4], [3, 4], [3, 1]]), [3, 4]);
  // 条件が値より低い rank（spline の inside 判定の形）
  assertEquals(shape([[5], [2, 3, 5], [1]]), [2, 3, 5]);
  // 条件だけが広い形も torch と同じく通る（出力は 3 者の broadcast）
  assertEquals(shape([[2, 3], [3], [3]]), [2, 3]);
  assertThrows(() => shape([[2], [3], [1]]), OpContractError);
  assertThrows(() => shape([[2], [2]]), OpContractError);
});

Deno.test("broadcast は torch 準拠の右詰めで、長さ 0 の次元も max で潰さない", () => {
  assertEquals(broadcastShapes([2, 3, 4], [3, 1], "t"), [2, 3, 4]);
  assertEquals(broadcastShapes([3, 1], [2, 3, 4], "t"), [2, 3, 4]);
  assertEquals(broadcastShapes([5], [], "t"), [5]);
  assertEquals(broadcastShapes([], [], "t"), []);
  // MUST: max(0, 1) = 1 ではなく 0（torch の broadcast 規則）
  assertEquals(broadcastShapes([0], [1], "t"), [0]);
  assertThrows(() => broadcastShapes([2, 3], [4, 3], "t"), OpContractError);
  assertThrows(() => broadcastShapes([0], [5], "t"), OpContractError);
});

Deno.test("出力 shape 計算が op ごとの契約どおりに決まる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs?: Readonly<Record<string, unknown>>,
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];
  assertEquals(shape("relu", [[2, 3]]), [2, 3]);
  assertEquals(shape("add", [[4, 1], [1, 5]]), [4, 5]);
  assertEquals(shape("matmul", [[7, 5], [5, 3]]), [7, 3]);
  assertEquals(shape("sum", [[6, 10]], { dim: 1 }), [6]);
  assertEquals(shape("amax", [[2, 3, 4]], { dim: 2 }), [2, 3]);
  assertEquals(shape("sum", [[4]], { dim: 0 }), []);
  // 最終次元以外の軸（permute を挟まずチャネル軸を畳む形 — VAE の L2 正規化）
  assertEquals(shape("sum", [[2, 384, 5, 7]], { dim: 1 }), [2, 5, 7]);
  assertEquals(shape("amin", [[2, 3, 4]], { dim: 0 }), [3, 4]);
  // cast / bitwise_not は要素ごとなので shape は素通し
  assertEquals(shape("cast", [[2, 3, 4]]), [2, 3, 4]);
  assertEquals(shape("bitwise_not", [[5]]), [5]);
});

// ADR 0012: bmm は rank-3 専業（rank-2 は matmul）、gather は最終次元固定。
// MUST: B / M / K / N を全て違う長さで確かめる — 軸の取り違えは正方形では見えない。
Deno.test("bmm は rank-3 のバッチ一致・縮約一致だけを受理する", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("bmm"), ins, "t")[0];
  assertEquals(shape([[2, 6, 3], [2, 3, 5]]), [2, 6, 5]);
  assertEquals(shape([[7, 1, 4], [7, 4, 9]]), [7, 1, 9]);
  // rank-2 は matmul の担当（兼用にしない）
  assertThrows(() => shape([[6, 3], [3, 5]]), OpContractError);
  assertThrows(() => shape([[2, 6, 3], [3, 5]]), OpContractError);
  assertThrows(() => shape([[2, 2, 6, 3], [2, 2, 3, 5]]), OpContractError);
  // バッチ次元の不一致（broadcast も stride 0 も語彙に無い）
  assertThrows(() => shape([[2, 6, 3], [3, 3, 5]]), OpContractError);
  assertThrows(() => shape([[1, 6, 3], [2, 3, 5]]), OpContractError);
  // 縮約次元の不一致（K と N / M の取り違え）
  assertThrows(() => shape([[2, 6, 3], [2, 5, 3]]), OpContractError);
});

Deno.test("gather は先行次元一致を要求し、出力は index と同形になる", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("gather"), ins, "t")[0];
  // 実測形（src f32[16,T,512] / index i32[16,T,T]）と同型 — 最終次元だけが違う
  assertEquals(shape([[4, 6, 9], [4, 6, 6]]), [4, 6, 6]);
  // 出力の最終次元は src より長くてもよい（同じ添字を何度引いてもよい）
  assertEquals(shape([[2, 3], [2, 7]]), [2, 7]);
  assertEquals(shape([[5], [3]]), [3]);
  // 先行次元の不一致（torch の一般 gather より狭い契約）
  assertThrows(() => shape([[4, 6, 9], [4, 5, 6]]), OpContractError);
  assertThrows(() => shape([[4, 6, 9], [3, 6, 6]]), OpContractError);
  // rank 不一致・スカラ
  assertThrows(() => shape([[4, 6, 9], [6, 6]]), OpContractError);
  assertThrows(() => shape([[], []]), OpContractError);
});

// ADR 0010: sym_prefix_slice は attrs（+ 束縛）から出力 shape を計算する — 宣言は照合される
// 側で、reshape / expand のような「宣言が目標形」ではない。
Deno.test("sym_prefix_slice は attrs と束縛から prefix 長を決める", () => {
  const shape = (
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown>,
    bindings?: Record<string, number>,
  ) => computeOutputShape(resolveOpContract("sym_prefix_slice"), ins, "t", { attrs, bindings })[0];
  const slice = (dim: number, coeff = 1, offset = 0) => ({ dim, coeff, offset });

  assertEquals([...attrKeysOf(resolveOpContract("sym_prefix_slice"))].sort(), ["slices", "sym"]);
  // 相対位置バケット表の実測形（2 軸とも記号）
  assertEquals(
    shape([[512, 512]], { sym: "T", slices: [slice(0), slice(1)] }, { T: 7 }),
    [7, 7],
  );
  // 縮めない軸は Tmax 形のまま残る
  assertEquals(shape([[512, 64]], { sym: "T", slices: [slice(0)] }, { T: 7 }), [7, 64]);
  // coeff / offset 付き（次元言語 coeff·sym+offset をそのまま使う）
  assertEquals(shape([[64]], { sym: "T", slices: [slice(0, 2, 3)] }, { T: 5 }), [13]);

  // Tmax 超過（定数バッファの範囲外読み出しになる）
  assertThrows(
    () => shape([[8, 8]], { sym: "T", slices: [slice(0)] }, { T: 9 }),
    OpContractError,
  );
  // 入力 rank の外の dim / 束縛が渡っていない形
  assertThrows(
    () => shape([[8, 8]], { sym: "T", slices: [slice(2)] }, { T: 4 }),
    OpContractError,
  );
  assertThrows(() => shape([[8, 8]], { sym: "T", slices: [slice(0)] }), OpContractError);
  // MUST: 束縛の参照は Object.hasOwn のみ（prototype 由来の名前で NaN 長にしない）
  assertThrows(
    () => shape([[8, 8]], { sym: "toString", slices: [slice(0)] }, { T: 4 }),
    OpContractError,
  );

  // attrs スキーマ: 同じ dim を 2 度 / 係数 0 / 負のオフセット / 未知キー / 空
  const bad = (attrs: Record<string, unknown>) =>
    assertThrows(
      () => assertNodeContract(node("sym_prefix_slice", ["a"], attrs), "t"),
      OpContractError,
    );
  bad({ sym: "T", slices: [slice(0), slice(0)] });
  bad({ sym: "T", slices: [{ dim: 0, coeff: 0, offset: 0 }] });
  bad({ sym: "T", slices: [{ dim: 0, coeff: 1, offset: -1 }] });
  bad({ sym: "T", slices: [{ dim: 0, coeff: 1, offset: 0, extra: 1 }] });
  bad({ sym: "T", slices: [{ dim: 0, coeff: 1 }] });
  bad({ sym: "T", slices: [] });
  bad({ sym: "1T", slices: [slice(0)] });
  bad({ slices: [slice(0)] });
  assertEquals(
    assertNodeContract(node("sym_prefix_slice", ["a"], { sym: "T", slices: [slice(0)] }), "t").kind,
    "symPrefixSlice",
  );
});

// ADR 0011: reshape / expand は「出力の宣言 shape が目標形」で、入力からは導けない。
Deno.test("reshape / expand は宣言 shape を目標形に取り、permute は dims で決まる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    context: { declared?: readonly number[]; attrs?: Record<string, unknown> },
  ) => computeOutputShape(resolveOpContract(op), ins, "t", context)[0];

  assertEquals(shape("reshape", [[2, 3, 4]], { declared: [6, 4] }), [6, 4]);
  assertEquals(shape("reshape", [[6]], { declared: [1, 6, 1] }), [1, 6, 1]);
  assertEquals(shape("expand", [[1, 6, 6]], { declared: [16, 6, 6] }), [16, 6, 6]);
  assertEquals(shape("expand", [[4]], { declared: [3, 4] }), [3, 4], "rank が増える形");
  assertEquals(shape("permute", [[2, 3, 4]], { attrs: { dims: [1, 2, 0] } }), [3, 4, 2]);

  // 要素数が合わない reshape（実バッファの大きさと宣言 shape が食い違う）
  assertThrows(() => shape("reshape", [[2, 3]], { declared: [4, 2] }), OpContractError);
  // 長さ 1 でない次元の拡張・rank 下げ
  assertThrows(() => shape("expand", [[2, 3]], { declared: [4, 3] }), OpContractError);
  assertThrows(() => shape("expand", [[2, 3]], { declared: [3] }), OpContractError);
  // dims と入力 rank の不一致 / 範囲外
  assertThrows(() => shape("permute", [[2, 3]], { attrs: { dims: [0, 1, 2] } }), OpContractError);
  assertThrows(() => shape("permute", [[2, 3]], { attrs: { dims: [0, 2] } }), OpContractError);
  // MUST: 目標形を渡さずに黙って推測しない
  assertThrows(() => shape("reshape", [[2, 3]], {}), OpContractError);
  assertThrows(() => shape("expand", [[2, 3]], {}), OpContractError);
  assertThrows(() => shape("permute", [[2, 3]], {}), OpContractError);
});

// ADR 0012: 融合 op の attrs スキーマ（許容キー + 型 + 値域）。
Deno.test("融合 op の attrs スキーマが値域まで検査する", () => {
  assertEquals(attrKeysOf(resolveOpContract("linear")), []);
  assertEquals([...attrKeysOf(resolveOpContract("layer_norm"))].sort(), [
    "eps",
    "normalized_shape",
  ]);
  assertEquals(attrKeysOf(resolveOpContract("softmax")), ["dim"]);
  assertEquals(attrKeysOf(resolveOpContract("safe_softmax")), ["dim"]);
  assertEquals(attrKeysOf(resolveOpContract("embedding")), ["padding_idx"]);
  assertEquals(attrKeysOf(resolveOpContract("masked_fill")), ["value"]);
  assertEquals([...attrKeysOf(resolveOpContract("conv1d"))].sort(), [
    "dilation",
    "groups",
    "padding",
    "stride",
  ]);
  assertEquals([...attrKeysOf(resolveOpContract("conv_transpose1d"))].sort(), [
    "padding",
    "stride",
  ]);

  const accept = (op: string, ins: readonly string[], attrs: Record<string, unknown>) =>
    assertNodeContract(node(op, ins, attrs), "t");
  const reject = (op: string, ins: readonly string[], attrs: Record<string, unknown>) =>
    assertThrows(() => assertNodeContract(node(op, ins, attrs), "t"), OpContractError);

  // layer_norm — 最終次元のみ（長さ 1 の normalized_shape）/ eps は有限の正数
  assertEquals(
    accept("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-7 }).kind,
    "layerNorm",
  );
  assertEquals(layerNormAttrs({ normalized_shape: [8], eps: 1e-7 }, "t"), {
    normalizedShape: [8],
    eps: 1e-7,
  });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [4, 8], eps: 1e-7 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [], eps: 1e-7 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [0], eps: 1e-7 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: 8, eps: 1e-7 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 0 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: -1e-7 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8] });

  // rms_norm — attrs は eps だけ（正規化長の正本は weight — ADR 0017）
  assertEquals(accept("rms_norm", ["x", "w"], { eps: 1e-6 }).kind, "rmsNorm");
  assertEquals(rmsNormEps({ eps: 1e-6 }, "t"), 1e-6);
  reject("rms_norm", ["x", "w"], { eps: 0 });
  reject("rms_norm", ["x", "w"], { eps: -1e-6 });
  reject("rms_norm", ["x", "w"], {});
  // MUST: normalized_shape の欄は無い（layer_norm から欄ごと写した IR を通さない）
  reject("rms_norm", ["x", "w"], { eps: 1e-6, normalized_shape: [8] });

  // clamp_min — attrs は min だけ（max を持ち込むと両側 clamp と区別が消える）
  assertEquals(accept("clamp_min", ["x"], { min: 1e-12 }).kind, "unary");
  assertEquals(scalarParamValues(resolveOpContract("clamp_min"), { min: -1.5 }, "t"), [-1.5]);
  reject("clamp_min", ["x"], {});
  reject("clamp_min", ["x"], { min: "0" });
  reject("clamp_min", ["x"], { min: 0, max: 1 });

  // softmax — 非負の軸番号のみ（負の軸表記はエクスポータ境界で正規化する）
  assertEquals(accept("softmax", ["x"], { dim: 0 }).kind, "softmax");
  assertEquals(softmaxDim({ dim: 3 }, "t"), 3);
  reject("softmax", ["x"], { dim: -1 });
  reject("softmax", ["x"], { dim: 1.5 });
  reject("softmax", ["x"], {});

  // safe_softmax — attrs は softmax と同一（ADR 0044。kind だけが違う）
  assertEquals(accept("safe_softmax", ["x"], { dim: 0 }).kind, "safeSoftmax");
  reject("safe_softmax", ["x"], { dim: -1 });
  reject("safe_softmax", ["x"], {});

  // embedding — padding_idx は受理して不活性（-1 が torch の「未指定」番兵）
  assertEquals(accept("embedding", ["w", "i"], { padding_idx: -1 }).kind, "embedding");
  assertEquals(accept("embedding", ["w", "i"], { padding_idx: 0 }).kind, "embedding");
  reject("embedding", ["w", "i"], { padding_idx: -2 });
  reject("embedding", ["w", "i"], { padding_idx: null });
  reject("embedding", ["w", "i"], {});

  // masked_fill — 有限の f32 スカラのみ
  assertEquals(accept("masked_fill", ["x", "m"], { value: 0 }).kind, "maskedFill");
  assertEquals(maskedFillValue({ value: -1.5 }, "t"), -1.5);
  reject("masked_fill", ["x", "m"], { value: "0" });
  reject("masked_fill", ["x", "m"], {});

  // conv1d — stride / dilation / groups は正整数・padding は非負整数（ADR 0015）
  const convAttrs = { stride: 1, padding: 1, dilation: 1, groups: 1 };
  assertEquals(accept("conv1d", ["x", "w", "b"], convAttrs).kind, "conv1d");
  assertEquals(conv1dAttrs({ stride: 2, padding: 0, dilation: 3, groups: 4 }, "t"), {
    stride: 2,
    padding: 0,
    dilation: 3,
    groups: 4,
  });
  reject("conv1d", ["x", "w", "b"], { ...convAttrs, stride: 0 });
  reject("conv1d", ["x", "w", "b"], { ...convAttrs, padding: -1 });
  reject("conv1d", ["x", "w", "b"], { ...convAttrs, dilation: 0 });
  reject("conv1d", ["x", "w", "b"], { ...convAttrs, groups: 0 });
  reject("conv1d", ["x", "w", "b"], { ...convAttrs, output_padding: 0 });
  // MUST: 宣言済み attrs の既定値補完はしない（欠けたら fail loudly — ADR 0012 / 0015）。
  // dilation / groups を省略可能にすると depthwise の IR が黙って通常畳み込みになる。
  reject("conv1d", ["x", "w", "b"], { stride: 1, padding: 1 });
  reject("conv1d", ["x", "w", "b"], { stride: 1, padding: 1, dilation: 1 });
  reject("conv1d", ["x", "w", "b"], { stride: 1, padding: 1, groups: 1 });

  // conv2d — 空間 3 つは [H, W] の 2 成分（スカラ表記は境界で正規化する規約 — ADR 0017）
  const conv2Attrs = {
    stride: [1, 1],
    padding: [0, 0],
    dilation: [1, 1],
    groups: 1,
  };
  assertEquals(accept("conv2d", ["x", "w", "b"], conv2Attrs).kind, "conv2d");
  assertEquals(
    conv2dAttrs({ stride: [2, 3], padding: [1, 0], dilation: [1, 2], groups: 4 }, "t"),
    { stride: [2, 3], padding: [1, 0], dilation: [1, 2], groups: 4 },
  );
  // MUST: 軸ごとに独立に見る（片側だけ 0 の stride を素通しするとカーネルがハングする）
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, stride: [0, 1] });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, stride: [1, 0] });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, dilation: [1, 0] });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, padding: [-1, 0] });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, groups: 0 });
  // 長さ 2 以外・スカラ表記は受理しない（同じ畳み込みに 2 通りの IR を作らない）
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, stride: 1 });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, stride: [1] });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, stride: [1, 1, 1] });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, groups: [1, 1] });
  // 宣言済み attrs の既定値補完はしない（4 つとも必須）
  reject("conv2d", ["x", "w", "b"], { stride: [1, 1], padding: [0, 0], dilation: [1, 1] });
  reject("conv2d", ["x", "w", "b"], { stride: [1, 1], padding: [0, 0], groups: 1 });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, output_padding: [0, 0] });

  // conv_transpose1d — attrs は stride / padding のみ（output_padding / dilation / groups の
  // 欄を作らないことが「実測外の値を黙って既定値で実行する」経路を潰している）
  assertEquals(
    accept("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: 0 }).kind,
    "convTranspose1d",
  );
  assertEquals(convTranspose1dAttrs({ stride: 8, padding: 4 }, "t"), { stride: 8, padding: 4 });
  // MUST: stride 0 は契約検査で止める（カーネル側でハングする — recon §4）
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 0, padding: 0 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: -1 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: 0, output_padding: 0 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2 });
});

/**
 * ADR 0012: masked_fill の実測埋め値 −3.4028234663852886e+38（f32 の最小有限値）が
 * **JSON 往復で ulp 不変**であることを固定する。IR の attrs は JSON の数値リテラルとして
 * 運ばれるので、往復で 1 ulp でも動くと GPU に載る埋め値が torch と食い違う。
 */
Deno.test("masked_fill の埋め値 −3.4e38 は JSON 往復と f32 丸めで ulp が動かない", () => {
  const value = -3.4028234663852886e+38;
  // f32 の最小有限値そのもの（f32 へ丸めても値が変わらない = ulp 不変）
  assertEquals(Math.fround(value), value);
  assertEquals(new Float32Array([value])[0], value);
  // 1 ulp でも下は f32 で有限に収まらない（= これが最小有限値である根拠）
  assertEquals(Math.fround(-Number.MAX_VALUE), Number.NEGATIVE_INFINITY);
  // JSON 往復（グラフ JSON はこの経路で運ばれる）
  const roundTrip = JSON.parse(JSON.stringify({ value })).value;
  assertEquals(roundTrip, value);
  assertEquals(Object.is(roundTrip, value), true);
  assertEquals(maskedFillValue({ value: roundTrip }, "t"), value);
  // 契約検査も往復後の値を受理する
  assertEquals(
    assertNodeContract(node("masked_fill", ["x", "m"], { value: roundTrip }), "t").name,
    "masked_fill",
  );
  // 非有限は IR v1 の規則どおり受理しない
  assertThrows(() => maskedFillValue({ value: Number.NEGATIVE_INFINITY }, "t"), OpContractError);
  assertThrows(() => maskedFillValue({ value: Number.NaN }, "t"), OpContractError);
});

Deno.test("融合 op の出力 shape が契約どおりに決まる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown> = {},
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];

  // linear — 先行次元はそのまま、最終次元が out へ差し替わる
  assertEquals(shape("linear", [[5, 7], [3, 7], [3]]), [5, 3]);
  assertEquals(shape("linear", [[2, 4, 6], [5, 6], [5]]), [2, 4, 5]);
  assertEquals(shape("linear", [[7], [3, 7], [3]]), [3]);
  assertThrows(() => shape("linear", [[5, 7], [3, 6], [3]]), OpContractError);
  assertThrows(() => shape("linear", [[5, 7], [3, 7], [4]]), OpContractError);
  assertThrows(() => shape("linear", [[5, 7], [2, 3, 7], [3]]), OpContractError);

  // layer_norm — normalized_shape は入力の最終次元と一致し、affine も同形
  const ln = { normalized_shape: [8], eps: 1e-5 };
  assertEquals(shape("layer_norm", [[3, 8], [8], [8]], ln), [3, 8]);
  assertEquals(shape("layer_norm", [[2, 3, 8], [8], [8]], ln), [2, 3, 8]);
  assertThrows(() => shape("layer_norm", [[3, 4], [8], [8]], ln), OpContractError);
  assertThrows(() => shape("layer_norm", [[3, 8], [4], [8]], ln), OpContractError);
  assertThrows(() => shape("layer_norm", [[3, 8], [8], [3, 8]], ln), OpContractError);

  // softmax — 最終次元のみ（一般 dim は語彙に無い）
  assertEquals(shape("softmax", [[4, 9]], { dim: 1 }), [4, 9]);
  assertEquals(shape("softmax", [[2, 3, 4]], { dim: 2 }), [2, 3, 4]);
  assertThrows(() => shape("softmax", [[2, 3, 4]], { dim: 1 }), OpContractError);
  assertThrows(() => shape("softmax", [[2, 3, 4]], { dim: 3 }), OpContractError);
  assertThrows(() => shape("softmax", [[3, 0]], { dim: 1 }), OpContractError);

  // safe_softmax — shape 規則は softmax と同一
  assertEquals(shape("safe_softmax", [[4, 9]], { dim: 1 }), [4, 9]);
  assertThrows(() => shape("safe_softmax", [[2, 3, 4]], { dim: 1 }), OpContractError);
  assertThrows(() => shape("safe_softmax", [[3, 0]], { dim: 1 }), OpContractError);

  // embedding — 出力は index の形に hidden を足したもの
  assertEquals(shape("embedding", [[7, 4], [2, 3]]), [2, 3, 4]);
  assertEquals(shape("embedding", [[7, 4], [5]]), [5, 4]);
  assertThrows(() => shape("embedding", [[7], [5]]), OpContractError);
  assertThrows(() => shape("embedding", [[7, 4], []]), OpContractError);

  // masked_fill — 出力は常に x と同形（mask は右詰め broadcast で読むだけ）
  const fill = { value: 0 };
  assertEquals(shape("masked_fill", [[1, 3, 4, 5], [1, 1, 4, 5]], fill), [1, 3, 4, 5]);
  assertEquals(shape("masked_fill", [[3, 4, 5], [5]], fill), [3, 4, 5]);
  // MUST: mask が x を広げる形は拒否（broadcastShapes をそのまま使うと通ってしまう）
  assertThrows(() => shape("masked_fill", [[1], [4]], fill), OpContractError);
  assertThrows(() => shape("masked_fill", [[3, 4], [2, 3, 4]], fill), OpContractError);
  assertThrows(() => shape("masked_fill", [[3, 4], [3, 5]], fill), OpContractError);

  // conv1d — 出力長は floor((L + 2P − D·(K−1) − 1) / S) + 1（ADR 0015 の dilation 一般形）
  const conv = { stride: 1, padding: 1, dilation: 1, groups: 1 };
  assertEquals(shape("conv1d", [[2, 3, 9], [4, 3, 3], [4]], conv), [2, 4, 9]);
  assertEquals(
    shape("conv1d", [[1, 2, 11], [3, 2, 4], [3]], { ...conv, stride: 2, padding: 0 }),
    [1, 3, 4],
  );
  assertEquals(shape("conv1d", [[1, 1, 3], [2, 1, 3], [2]], { ...conv, padding: 2 }), [1, 2, 5]);
  // dilation の一般形（DDSConv: depthwise g=C・k=5・d=3・p=6 で長さが保たれる）
  assertEquals(
    shape("conv1d", [[1, 6, 17], [6, 1, 5], [6]], { ...conv, padding: 6, dilation: 3, groups: 6 }),
    [1, 6, 17],
  );
  // dilation は出力長を縮める（p=0 / d=3 / k=3 なら L − 6）
  assertEquals(
    shape("conv1d", [[1, 2, 10], [3, 2, 3], [3]], {
      ...conv,
      padding: 0,
      dilation: 3,
    }),
    [1, 3, 4],
  );
  // groups は Cin / Cout の両方を割り切る必要がある（重みの第 2 軸は Cin/groups）
  assertEquals(
    shape("conv1d", [[1, 6, 11], [9, 2, 3], [9]], { ...conv, groups: 3 }),
    [1, 9, 11],
  );
  assertThrows(
    () => shape("conv1d", [[1, 6, 11], [9, 2, 3], [9]], { ...conv, groups: 4 }),
    OpContractError,
  );
  assertThrows(
    () => shape("conv1d", [[1, 6, 11], [8, 2, 3], [8]], { ...conv, groups: 3 }),
    OpContractError,
  );
  // 重みの第 2 軸が Cin/groups でない（groups=1 の完全一致も同じ検査で落ちる）
  assertThrows(() => shape("conv1d", [[2, 3, 9], [4, 2, 3], [4]], conv), OpContractError);
  assertThrows(
    () => shape("conv1d", [[1, 6, 11], [6, 6, 3], [6]], { ...conv, groups: 6 }),
    OpContractError,
  );
  assertThrows(() => shape("conv1d", [[2, 3, 9], [4, 3, 3], [3]], conv), OpContractError);
  assertThrows(() => shape("conv1d", [[3, 9], [4, 3, 3], [4]], conv), OpContractError);
  // padding 込みでもカーネル長に足りない（dilation で張りが伸びた形も同じ門で落ちる）
  assertThrows(() => shape("conv1d", [[1, 1, 2], [1, 1, 5], [1]], conv), OpContractError);
  assertThrows(
    () => shape("conv1d", [[1, 1, 6], [1, 1, 3], [1]], { ...conv, padding: 0, dilation: 4 }),
    OpContractError,
  );

  // conv_transpose1d — 重みは [Cin, Cout, K]・出力長は L·stride（2P == K − S の形のみ）
  assertEquals(
    shape("conv_transpose1d", [[1, 5, 7], [5, 3, 2], [3]], { stride: 2, padding: 0 }),
    [1, 3, 14],
  );
  assertEquals(
    shape("conv_transpose1d", [[1, 3, 5], [3, 2, 16], [2]], { stride: 8, padding: 4 }),
    [1, 2, 40],
  );
  // MUST: 重みを [Cout, Cin, K] と読む取り違えは**非対称チャネル**でしか赤くならない
  assertThrows(
    () => shape("conv_transpose1d", [[1, 5, 7], [3, 5, 2], [5]], { stride: 2, padding: 0 }),
    OpContractError,
  );
  assertThrows(
    () => shape("conv_transpose1d", [[1, 5, 7], [5, 3, 2], [5]], { stride: 2, padding: 0 }),
    OpContractError,
  );
  // 2P ≠ K − S の一般形は fail loudly（出力長が L·stride にならない — ADR 0015）
  assertThrows(
    () => shape("conv_transpose1d", [[1, 5, 7], [5, 3, 3], [3]], { stride: 2, padding: 0 }),
    OpContractError,
  );
  assertThrows(
    () => shape("conv_transpose1d", [[1, 5, 7], [5, 3, 2], [3]], { stride: 2, padding: 1 }),
    OpContractError,
  );
  assertThrows(
    () => shape("conv_transpose1d", [[5, 7], [5, 3, 2], [3]], { stride: 2, padding: 0 }),
    OpContractError,
  );
});

Deno.test("契約に合わない shape は全て fail loudly", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs?: Readonly<Record<string, unknown>>,
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];
  assertThrows(() => shape("matmul", [[7, 5], [4, 3]]), OpContractError);
  assertThrows(() => shape("matmul", [[2, 3, 4], [4, 3]]), OpContractError);
  assertThrows(() => shape("sum", [[]], { dim: 0 }), OpContractError);
  // 空軸の amax は identity が定義できない（sum は 0 なので通る）
  assertThrows(() => shape("amax", [[3, 0]], { dim: 1 }), OpContractError);
  assertEquals(shape("sum", [[3, 0]], { dim: 1 }), [3]);
  // attrs.dim は宣言必須（既定値補完をしない）で、rank 外は fail loudly
  assertThrows(() => shape("sum", [[6, 10]]), OpContractError);
  assertThrows(() => shape("sum", [[6, 10]], { dim: 2 }), OpContractError);
  assertThrows(() => shape("relu", [[2], [2]]), OpContractError);
});

// ADR 0011: strided コピー族（permute / expand / sym_prefix_slice / masked_fill）の rank 上限は
// **契約層**で見る。codegen（stridedParams）まで落とすと、利用者には「契約検査は通ったのに
// 実行段で内部エラー」として出て、どの op のどの値が rank 超過なのか診断に残らない。
Deno.test("strided コピー族の rank 上限（1..4）は契約層で落ちる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    context: {
      declared?: readonly number[];
      attrs?: Record<string, unknown>;
      bindings?: Record<string, number>;
    } = {},
  ) => computeOutputShape(resolveOpContract(op), ins, "t", context)[0];
  const rank4 = [2, 2, 2, 2];
  const rank5 = [2, 2, 2, 2, 2];

  // 上限ちょうど（rank 4）は通る — 上限を下げる誤りもここで赤くなる
  assertEquals(shape("permute", [rank4], { attrs: { dims: [3, 0, 2, 1] } }), rank4);
  assertEquals(shape("expand", [[1, 2, 2, 2]], { declared: rank4 }), rank4);
  assertEquals(
    shape("sym_prefix_slice", [[4, 2, 2, 2]], {
      attrs: { sym: "T", slices: [{ dim: 0, coeff: 1, offset: 0 }] },
      bindings: { T: 2 },
    }),
    rank4,
  );
  assertEquals(shape("masked_fill", [rank4, [2, 2]], { attrs: { value: 0 } }), rank4);

  // rank 5 は strided カーネルの params（rank 4 固定）に載らない
  assertThrows(
    () => shape("permute", [rank5], { attrs: { dims: [0, 1, 2, 3, 4] } }),
    OpContractError,
    "strided",
  );
  assertThrows(
    () => shape("expand", [[1, 1, 1, 1, 1]], { declared: rank5 }),
    OpContractError,
    "strided",
  );
  assertThrows(
    () =>
      shape("sym_prefix_slice", [rank5], {
        attrs: { sym: "T", slices: [{ dim: 0, coeff: 1, offset: 0 }] },
        bindings: { T: 1 },
      }),
    OpContractError,
    "strided",
  );
  assertThrows(
    () => shape("masked_fill", [rank5, [2]], { attrs: { value: 0 } }),
    OpContractError,
    "strided",
  );

  // rank 0（スカラ）も同じ理由で載らない — mask だけ rank 0 の形は契約を素通りしていた
  assertThrows(() => shape("permute", [[]], { attrs: { dims: [0] } }), OpContractError, "strided");
  assertThrows(
    () => shape("masked_fill", [[2, 3], []], { attrs: { value: 0 } }),
    OpContractError,
    "strided",
  );
});

/**
 * 融合 attention の**省略可能な第 4 入力 = 加算 mask**（ADR 0023 改訂）。
 *
 * カーネルは mask を `[1,1,M,N]` としてバッチ base 抜きの平坦添字で読む（B·H 全体へ
 * broadcast）ので、そこから外れた形は**値が静かに壊れる**（例外は出ない）。契約層が
 * 唯一の検出器なので、受理と拒否の全面をここで固定する。
 */
Deno.test("attention の mask は f32・[1,1,M,N] ちょうどだけを受理する", () => {
  const contract = resolveOpContract("attention");
  const q = [2, 3, 5, 4];
  const k = [2, 3, 7, 4];
  const attrs = { scale: 0.5 };
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(contract, ins, "t", { attrs })[0];

  // アリティは 3 か 4（可変ではない — 「何本でも」は cat だけ）
  assertEquals(contract.arity, 3);
  assertEquals(contract.maxArity, 4);
  assertEquals(contract.variadic, undefined);
  assertEquals(describeArity(contract), "3 か 4");
  assertEquals(arityFits(contract, 3), true);
  assertEquals(arityFits(contract, 4), true);
  assertEquals(arityFits(contract, 2), false);
  assertEquals(arityFits(contract, 5), false);
  assertEquals(
    assertNodeContract(node("attention", ["q", "k", "v", "m"], attrs), "t").kind,
    "attention",
  );
  assertThrows(
    () => assertNodeContract(node("attention", ["q", "k", "v", "m", "m2"], attrs), "t"),
    OpContractError,
    "3 か 4",
  );

  // 受理: mask 無し / [1,1,M,N]（M ≠ N の非正方も）
  assertEquals(shape([q, k, k]), q);
  assertEquals(shape([q, k, k, [1, 1, 5, 7]]), q);

  // 拒否: B / H が 1 でない（head 別・バッチ別マスクは語彙に無い）
  assertThrows(() => shape([q, k, k, [2, 1, 5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [1, 3, 5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [2, 3, 5, 7]]), OpContractError, "[1,1,M,N]");
  // 拒否: rank ≠ 4
  assertThrows(() => shape([q, k, k, [5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [1, 5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [1, 1, 1, 5, 7]]), OpContractError, "[1,1,M,N]");
  // 拒否: M / N の不一致（M と N を取り違えた形は正方 mask でだけ素通りするので両方見る）
  assertThrows(() => shape([q, k, k, [1, 1, 4, 7]]), OpContractError, "M / N");
  assertThrows(() => shape([q, k, k, [1, 1, 5, 5]]), OpContractError, "M / N");
  assertThrows(() => shape([q, k, k, [1, 1, 7, 5]]), OpContractError, "M / N");

  // 拒否: dtype（uniform 契約なので mask も f32 — bool マスクは masked_fill の語彙）
  const maskNode = node("attention", ["q", "k", "v", "m"], attrs);
  assertEquals(
    resolveNodeDtypes(contract, maskNode, ["f32", "f32", "f32", "f32"], ["f32"], "t"),
    ["f32"],
  );
  for (const dtype of ["bool", "i32"] as const) {
    assertThrows(
      () => resolveNodeDtypes(contract, maskNode, ["f32", "f32", "f32", dtype], ["f32"], "t"),
      OpContractError,
    );
  }
});

// ---- states 欄と effect op（ADR 0067 決定 4 / 5） --------------------------

/** `state_append` の最小ノード（出力 0 本・states 欄 `{ slot }` ちょうど）。 */
const appendNode = (
  attrs: Record<string, unknown> = {},
  states: Record<string, string> = { slot: "kv.k" },
): IrNode => ({ op: "state_append", ins: ["chunk"], outs: [], attrs, states });

Deno.test("state_append は出力 0 本の effect op で、states 欄が必須", () => {
  const contract = resolveOpContract("state_append");
  assertEquals(contract.kind, "stateAppend");
  assertEquals(contract.arity, 1);
  // 出力数は dtype 写像の列長そのもの（空列 = 0 本の宣言 — ADR 0067 決定 5）
  assertEquals(outputCountOf(contract), 0);
  assertEquals(contract.outputDtypes, []);
  assertEquals(attrKeysOf(contract), []);

  assertStrictEquals(assertNodeContract(appendNode(), "t"), contract);
  // 出力を書いた形は「値を定義しない op」の契約に反する
  assertThrows(
    () => assertNodeContract({ ...appendNode(), outs: ["y"] }, "t"),
    OpContractError,
    "出力数が 1（契約は 0）",
  );
  // states 欄そのものが必須（required: true）
  assertThrows(
    () => assertNodeContract({ ...appendNode(), states: {} }, "t"),
    OpContractError,
    "states 欄が無い",
  );
  // キー集合はちょうど { slot }
  assertThrows(
    () => assertNodeContract(appendNode({}, { slot: "kv.k", extra: "kv.v" }), "t"),
    OpContractError,
    "states 欄のキーが",
  );
  assertThrows(
    () => assertNodeContract(appendNode({}, { k: "kv.k" }), "t"),
    OpContractError,
    "states 欄のキーが",
  );
});

// 0 本を許すのは契約が effect を宣言する op **だけ**（パーサは本数に意味を与えない）。
Deno.test("effect でない op の outs 空は契約層が落とす", () => {
  for (const [op, ins] of [["relu", ["a"]], ["add", ["a", "b"]], ["topk", ["a"]]] as const) {
    assertThrows(
      () => assertNodeContract({ ...node(op, ins, op === "topk" ? { k: 2 } : {}), outs: [] }, "t"),
      OpContractError,
      "出力数が 0",
    );
  }
});

Deno.test("states 欄を持たない op に states を書けない", () => {
  assertThrows(
    () => assertNodeContract(node("relu", ["a"], {}, { slot: "kv.k" }), "t"),
    OpContractError,
    "states 欄を持たない",
  );
  // topk（多出力 op）も同じ — states 欄は attention / state_append の 2 本だけの契約面
  assertThrows(
    () =>
      assertNodeContract(
        { ...node("topk", ["a"], { k: 2 }, { slot: "kv.k" }), outs: ["v", "i"] },
        "t",
      ),
    OpContractError,
    "states 欄を持たない",
  );
});

Deno.test("attention の states 形は欄の有無で判別され、従来形の受理集合は動かない", () => {
  const contract = resolveOpContract("attention");
  const states = { k: "kv.k", v: "kv.v" };
  // 従来形（欄なし）は 3 本でも 4 本でも通る — 1 バイトも動かさない
  assertStrictEquals(
    assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }), "t"),
    contract,
  );
  assertNodeContract(node("attention", ["q", "k", "v", "m"], { scale: 0.5 }), "t");
  // states 形は 3 本ちょうど（mask は causal 固定なので取らない — ADR 0067 決定 4）
  assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }, states), "t");
  assertThrows(
    () => assertNodeContract(node("attention", ["q", "k", "v", "m"], { scale: 0.5 }, states), "t"),
    OpContractError,
    "states 形は入力 3 本ちょうど",
  );
  // キー集合はちょうど { k, v }
  assertThrows(
    () =>
      assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }, { k: "kv.k" }), "t"),
    OpContractError,
    "states 欄のキーが",
  );
  // k と v に同じスロットを書いた取り違えは shape では捕まらない（k/v スロットは同形）
  assertThrows(
    () =>
      assertNodeContract(
        node("attention", ["q", "k", "v"], { scale: 0.5 }, { k: "kv.k", v: "kv.k" }),
        "t",
      ),
    OpContractError,
    "同じスロット 'kv.k' を複数の欄から参照",
  );
});

Deno.test("省略可能 attr `window` は states 欄を持つノードでのみ宣言できる", () => {
  const states = { k: "kv.k", v: "kv.v" };
  // 必須 attrs には出ない（宣言必須の欄と混ざらない）
  assertEquals(attrKeysOf(resolveOpContract("attention")), ["scale"]);
  assertEquals(optionalAttrKeysOf(resolveOpContract("attention")), ["window"]);
  assertEquals(optionalAttrKeysOf(resolveOpContract("state_append")), ["window"]);
  assertEquals(optionalAttrKeysOf(resolveOpContract("relu")), []);

  // states 形では宣言でき、省略もできる（欄の不存在 = 全 context）
  assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5, window: 512 }, states), "t");
  assertNodeContract(appendNode({ window: 512 }), "t");
  assertNodeContract(appendNode({}), "t");
  // 従来形に書くと「誰も読まない attr」になるので拒否する（受理集合の対称性）
  assertThrows(
    () => assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5, window: 512 }), "t"),
    OpContractError,
    "states 欄を持つノードでのみ宣言できる",
  );
  // 値域は必須 attr と同じ厳しさ（0・負・非整数は落ちる）
  for (const window of [0, -1, 1.5, "512", null, true]) {
    assertThrows(() => assertNodeContract(appendNode({ window }), "t"), OpContractError);
  }
  // capability 射影には必須と省略可能の**和**が載る（列挙門が states 形を拒否しないため）
  assertEquals(
    [...RUNTIME_SUPPORT.ops.get("attention")?.attrKeys ?? []].sort(),
    ["scale", "window"],
  );
  assertEquals([...RUNTIME_SUPPORT.ops.get("state_append")?.attrKeys ?? []], ["window"]);
  // 契約外 attrs の判定は従来どおり（省略可能の受理は window 1 本だけを広げる）
  assertThrows(
    () => assertNodeContract(appendNode({ windows: 512 }), "t"),
    OpContractError,
    "契約外 attrs [windows]",
  );
});

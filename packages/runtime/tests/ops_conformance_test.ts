// op 契約の適合表テスト（TS 側）。
//
// 正本は実装ではなく tests/fixtures/op-contracts.json で、Python 側（エクスポータの
// tools/exporter/tests/test_ops_conformance.py）は**同じ表**に対して同じことを確かめる。
// 表をコピーせず読み込むのは、コピーが増えた瞬間に「片側だけ通る契約」が生まれるため
// （tests/fixtures/dim-grammar.json と同じ規律）。
//
// ここが見るのは「両実装で沈黙のうちに割れうる契約面」だけ:
// op 名の全集合 / アリティ / スロット dtype / attrs キー集合 / attrs の値域 / 出力数と
// 出力 slot 別の dtype 写像（ADR 0068 決定 1）/ 出力 shape 規則（strided コピー族の rank
// 上限を含む）/ 低精度格納の適格スロット（ADR 0018）。
//
// The second half of this file covers what the table cannot express: the node-level checks
// (assertNodeContract / resolveNodeDtypes), the attrs extractors, the diagnostics an error must
// name, the capability projection, and literal pins that live outside the table. Cases the table
// already checks row for row are not repeated there.

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import { STRIDED_RANK } from "../src/codegen/strided.ts";
import { evalDim, parseDim } from "../src/format/dims.ts";
import type { IrDim, IrNode } from "../src/format/ir.ts";
import type { OpSupport } from "../src/ops/support.ts";
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
  conv3dAttrs,
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
  stateWindow,
  UNARY_OPS,
  WEIGHT_CHANNEL_AXES,
  WEIGHT_SLOTS,
} from "../src/ops.ts";

type DtypeSpec =
  | { readonly kind: "uniform"; readonly accept: readonly string[] }
  | { readonly kind: "perSlot"; readonly slots: readonly (readonly string[])[] };

type OpEntry = {
  readonly op: string;
  readonly arity: number;
  /** 入力数が可変か（true のとき arity は下限 — cat のみ）。 */
  readonly variadic: boolean;
  /** 末尾に省略可能な入力を持つ op の上限（attention の mask のみ。無ければ undefined）。 */
  readonly maxArity: number | undefined;
  readonly dtypes: DtypeSpec;
  /**
   * **出力 slot 別**の「スロット 0 の入力 dtype → その出力の dtype」写像の列（ADR 0068 決定 1）。
   * 恒等な単一出力 op では表に無い = undefined（= 長さ 1 の恒等列）。
   */
  readonly outDtypes: readonly Readonly<Record<string, string>>[] | undefined;
  readonly attrs: readonly string[];
  /** 省略可能な attrs（ADR 0067 の `window`。持たない op では空）。 */
  readonly optionalAttrs: readonly string[];
  /** states 欄の契約（ADR 0067 決定 4 / 5。欄を持てない op では undefined）。 */
  readonly states: { readonly keys: readonly string[]; readonly required: boolean } | undefined;
  /** 低精度格納が適格になる重みスロット（ADR 0018。持たない op では undefined）。 */
  readonly weightSlot: number | undefined;
  /** i8 の per-channel scale が乗る重みの軸（ADR 0019。持たない op では undefined）。 */
  readonly channelAxis: number | undefined;
};

type AttrEntry = {
  readonly op: string;
  readonly attr: string;
  readonly accept: readonly unknown[];
  readonly reject: readonly unknown[];
};

type ShapeCase = {
  readonly op: string;
  readonly ins: readonly (readonly IrDim[])[];
  readonly attrs?: Readonly<Record<string, unknown>>;
  readonly declared?: readonly IrDim[];
  readonly bindings: Readonly<Record<string, number>>;
  /** ノードの states 欄（契約が固定するキー → スロット名。ADR 0067 決定 4）。 */
  readonly states?: Readonly<Record<string, string>>;
  /** 参照するスロットの容量込み具体形（スロット名 → shape）。 */
  readonly stateShapes?: Readonly<Record<string, readonly IrDim[]>>;
  /** **出力 slot 別**の shape の列（ADR 0068 決定 1 — `state_append` だけ空列）。 */
  readonly outs?: readonly (readonly IrDim[])[];
  readonly throws: boolean;
  readonly why: string;
};

type Fixture = {
  readonly stridedRankMax: number;
  readonly ops: readonly OpEntry[];
  readonly attrValues: readonly AttrEntry[];
  readonly shapes: readonly ShapeCase[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const field = (raw: Record<string, unknown>, key: string): unknown => {
  if (!Object.hasOwn(raw, key)) throw new Error(`fixture: キー '${key}' が無い`);
  return raw[key];
};

const array = (value: unknown, where: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`fixture: ${where} が配列でない`);
  return value;
};

const strings = (value: unknown, where: string): string[] =>
  array(value, where).map((item) => {
    if (typeof item !== "string") throw new Error(`fixture: ${where} は文字列の列`);
    return item;
  });

const dims = (value: unknown, where: string): IrDim[] =>
  array(value, where).map((dim) => {
    if (typeof dim === "number" || typeof dim === "string") return dim;
    throw new Error(`fixture: ${where} の次元が数値でも次元式でもない`);
  });

const asDtypeSpec = (value: unknown): DtypeSpec => {
  if (!isRecord(value)) throw new Error("fixture: dtypes がオブジェクトでない");
  const kind = field(value, "kind");
  if (kind === "uniform") {
    return { kind, accept: strings(field(value, "accept"), "dtypes.accept") };
  }
  if (kind === "perSlot") {
    return {
      kind,
      slots: array(field(value, "slots"), "dtypes.slots").map((slot, index) =>
        strings(slot, `dtypes.slots[${index}]`),
      ),
    };
  }
  throw new Error(`fixture: dtypes.kind が uniform / perSlot でない: ${JSON.stringify(kind)}`);
};

const asOpEntry = (raw: unknown): OpEntry => {
  if (!isRecord(raw)) throw new Error("fixture: ops の要素がオブジェクトでない");
  const arity = field(raw, "arity");
  if (typeof arity !== "number") throw new Error("fixture: arity が数値でない");
  const outDtypes = raw["out_dtypes"];
  if (outDtypes !== undefined && !Array.isArray(outDtypes)) {
    throw new Error("fixture: out_dtypes が出力 slot 別の列でない");
  }
  const variadic = raw["variadic"];
  if (variadic !== undefined && variadic !== true) {
    throw new Error("fixture: variadic は true か省略のみ");
  }
  const maxArity = raw["max_arity"];
  if (maxArity !== undefined && typeof maxArity !== "number") {
    throw new Error("fixture: max_arity が数値でない");
  }
  // 「何本でも」と「決まった 1 本が増える」は別の契約面（併記は表の作り自体の誤り）
  if (maxArity !== undefined && variadic === true) {
    throw new Error("fixture: variadic と max_arity は併記しない");
  }
  const weightSlot = raw["weight_slot"];
  if (weightSlot !== undefined && typeof weightSlot !== "number") {
    throw new Error("fixture: weight_slot が数値でない");
  }
  const channelAxis = raw["channel_axis"];
  if (channelAxis !== undefined && typeof channelAxis !== "number") {
    throw new Error("fixture: channel_axis が数値でない");
  }
  const states = raw["states"];
  if (states !== undefined && !isRecord(states)) throw new Error("fixture: states が表でない");
  const required = states === undefined ? undefined : states["required"];
  if (states !== undefined && typeof required !== "boolean") {
    throw new Error("fixture: states.required が真偽値でない");
  }
  return {
    op: String(field(raw, "op")),
    arity,
    variadic: variadic === true,
    maxArity,
    dtypes: asDtypeSpec(field(raw, "dtypes")),
    outDtypes: outDtypes?.map((slot, index) => {
      if (!isRecord(slot)) throw new Error(`fixture: out_dtypes[${index}] が表でない`);
      return Object.fromEntries(
        Object.entries(slot).map(([from, to]) => {
          if (typeof to !== "string") {
            throw new Error(`fixture: out_dtypes[${index}][${from}] が文字列でない`);
          }
          return [from, to];
        }),
      );
    }),
    attrs: strings(field(raw, "attrs"), "attrs"),
    optionalAttrs:
      raw["optional_attrs"] === undefined ? [] : strings(raw["optional_attrs"], "optional_attrs"),
    states:
      states === undefined || typeof required !== "boolean"
        ? undefined
        : {
            keys: strings(field(states, "keys"), "states.keys"),
            required,
          },
    weightSlot,
    channelAxis,
  };
};

const asAttrEntry = (raw: unknown): AttrEntry => {
  if (!isRecord(raw)) throw new Error("fixture: attr_values の要素がオブジェクトでない");
  return {
    op: String(field(raw, "op")),
    attr: String(field(raw, "attr")),
    accept: array(field(raw, "accept"), "accept"),
    reject: array(field(raw, "reject"), "reject"),
  };
};

const asShapeCase = (raw: unknown): ShapeCase => {
  if (!isRecord(raw)) throw new Error("fixture: shapes の要素がオブジェクトでない");
  const op = String(field(raw, "op"));
  const attrs = raw["attrs"];
  const declared = raw["declared"];
  const outs = raw["outs"];
  const bindings: Record<string, number> = {};
  if (raw["bindings"] !== undefined) {
    if (!isRecord(raw["bindings"])) throw new Error(`fixture: ${op} の bindings が表でない`);
    for (const [sym, bound] of Object.entries(raw["bindings"])) {
      if (typeof bound !== "number") throw new Error(`fixture: 束縛 ${sym} が数値でない`);
      bindings[sym] = bound;
    }
  }
  const throws = raw["throws"] === true;
  if (throws === (outs !== undefined)) {
    throw new Error(
      `fixture: shapes ケースは outs か throws のどちらか一方 ${JSON.stringify(raw)}`,
    );
  }
  if (attrs !== undefined && !isRecord(attrs)) {
    throw new Error(`fixture: ${op} の attrs がオブジェクトでない`);
  }
  const states = raw["states"];
  if (states !== undefined && !isRecord(states)) {
    throw new Error(`fixture: ${op} の states が表でない`);
  }
  const stateShapes = raw["state_shapes"];
  if (stateShapes !== undefined && !isRecord(stateShapes)) {
    throw new Error(`fixture: ${op} の state_shapes が表でない`);
  }
  return {
    op,
    ins: array(field(raw, "ins"), `${op}.ins`).map((shape, index) =>
      dims(shape, `${op}.ins[${index}]`),
    ),
    attrs: attrs === undefined ? undefined : attrs,
    declared: declared === undefined ? undefined : dims(declared, `${op}.declared`),
    bindings,
    states:
      states === undefined
        ? undefined
        : Object.fromEntries(
            Object.entries(states).map(([key, slot]) => {
              if (typeof slot !== "string") {
                throw new Error(`fixture: ${op}.states[${key}] が文字列でない`);
              }
              return [key, slot];
            }),
          ),
    stateShapes:
      stateShapes === undefined
        ? undefined
        : Object.fromEntries(
            Object.entries(stateShapes).map(([slot, shape]) => [
              slot,
              dims(shape, `${op}.state_shapes['${slot}']`),
            ]),
          ),
    outs:
      outs === undefined
        ? undefined
        : array(outs, `${op}.outs`).map((shape, index) => dims(shape, `${op}.outs[${index}]`)),
    throws,
    why: typeof raw["why"] === "string" ? raw["why"] : "",
  };
};

const loadFixture = async (): Promise<Fixture> => {
  const text = await readTextFile(new URL("./fixtures/op-contracts.json", import.meta.url));
  const raw: unknown = JSON.parse(text);
  if (!isRecord(raw)) throw new Error("fixture: オブジェクトでない");
  const max = field(raw, "strided_rank_max");
  if (typeof max !== "number") throw new Error("fixture: strided_rank_max が数値でない");
  return {
    stridedRankMax: max,
    ops: array(field(raw, "ops"), "ops").map(asOpEntry),
    attrValues: array(field(raw, "attr_values"), "attr_values").map(asAttrEntry),
    shapes: array(field(raw, "shapes"), "shapes").map(asShapeCase),
  };
};

/** 適合表の次元言語を束縛で数値へ解決する（TS の shape 計算は束縛後の数値だけを扱う）。 */
const resolveDims = (
  shape: readonly IrDim[],
  bindings: Readonly<Record<string, number>>,
): number[] =>
  shape.map((dim) => (typeof dim === "number" ? dim : evalDim(parseDim(dim), bindings)));

const label = (testCase: ShapeCase): string =>
  `${testCase.op}(${testCase.ins.map((shape) => `[${shape.join(",")}]`).join(", ")})`;

test("適合表の op 集合が契約表と完全一致する", async () => {
  const fixture = await loadFixture();
  assertEquals(fixture.ops.map((entry) => entry.op).sort(), [...OP_CONTRACTS.keys()].sort());
  // 表の中で op が重複していない（重複すると片方の宣言が黙って無視される）
  assertEquals(new Set(fixture.ops.map((entry) => entry.op)).size, fixture.ops.length);
});

test("適合表の strided rank 上限が実装の定数と一致する", async () => {
  const fixture = await loadFixture();
  assertEquals(fixture.stridedRankMax, STRIDED_RANK);
});

test("適合表のアリティ / スロット dtype / attrs キーが契約表と一致する", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    const contract = resolveOpContract(entry.op);
    assertEquals(contract.arity, entry.arity, `${entry.op}: アリティ`);
    // 可変アリティは「入力何本まで受理するか」という契約面そのもの（片側だけ可変にすると
    // エクスポータが書ける本数とランタイムが受理する本数が割れる）。省略可能な末尾入力
    // （attention の mask）の上限も同じ理由で表に載る。
    assertEquals(contract.variadic === true, entry.variadic, `${entry.op}: 可変アリティ`);
    assertEquals(contract.maxArity, entry.maxArity, `${entry.op}: 省略可能入力の上限`);
    assertEquals(
      [...attrKeysOf(contract)].sort(),
      [...entry.attrs].sort(),
      `${entry.op}: attrs キー集合`,
    );
    // 省略可能 attrs（ADR 0067 の `window`）と states 欄の契約も片側だけ動くと
    // 「エクスポータが書ける形をランタイムが拒否する」に直結するので表で固定する。
    assertEquals(
      [...optionalAttrKeysOf(contract)].sort(),
      [...entry.optionalAttrs].sort(),
      `${entry.op}: 省略可能 attrs キー集合`,
    );
    assertEquals(
      contract.states === undefined
        ? undefined
        : {
            keys: [...contract.states.keys].sort(),
            required: contract.states.required,
          },
      entry.states === undefined
        ? undefined
        : { keys: [...entry.states.keys].sort(), required: entry.states.required },
      `${entry.op}: states 欄の契約`,
    );
    const slots = contract.slotDtypes;
    assertEquals(slots.kind, entry.dtypes.kind, `${entry.op}: dtype 契約の種別`);
    if (slots.kind === "uniform" && entry.dtypes.kind === "uniform") {
      assertEquals(
        [...slots.accept].sort(),
        [...entry.dtypes.accept].sort(),
        `${entry.op}: 受理 dtype`,
      );
      continue;
    }
    if (slots.kind === "perSlot" && entry.dtypes.kind === "perSlot") {
      assertEquals(
        slots.slots.map((accept) => [...accept].sort()),
        entry.dtypes.slots.map((accept) => [...accept].sort()),
        `${entry.op}: スロット別受理 dtype`,
      );
    }
  }
});

// 出力 dtype の導出は「対応表では実行可、実行で別の TypedArray として読まれる」を作る軸。
// 比較（f32 → bool）・bool の sum（→ i32）・where（bool → f32）が両側で揃っていることを、
// 表の側から固定する（cast は attrs.to で決まるので対象外）。
test("適合表の出力 dtype 写像が契約表と一致する", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    if (entry.op === "cast") {
      assertEquals(entry.outDtypes, undefined, "cast は out_dtypes を持たない（attrs.to が正本）");
      continue;
    }
    const contract = resolveOpContract(entry.op);
    const slots = contract.slotDtypes;
    const domain = slots.kind === "uniform" ? slots.accept : slots.slots[0];
    // 省略された op は「出力 1 本・スロット 0 の受理集合上の恒等写像」が期待値。
    const expected = (entry.outDtypes ?? [{}]).map((slot) =>
      Object.fromEntries([...domain].sort().map((dtype) => [dtype, slot[dtype] ?? dtype])),
    );
    assertEquals(
      contract.outputDtypes.map((slot) =>
        Object.fromEntries([...slot].sort(([a], [b]) => a.localeCompare(b))),
      ),
      expected,
      `${entry.op}: 出力 slot 別の dtype 写像`,
    );
    // 出力数は写像の列長そのもの（表と実装で本数が割れると、2 本目の出力の dtype が
    // 誰にも照合されない状態になる）。
    assertEquals(outputCountOf(contract), expected.length, `${entry.op}: 契約が宣言する出力数`);
  }
});

// 低精度格納の適格判定（ADR 0018）は TS 側 WEIGHT_SLOTS と Python 側の鏡像で決まり、
// 割れると「エクスポータが f16 で書いた重みをランタイムが適格外と見なして CPU 展開する」
// （= VRAM 削減が黙って消える）か、その逆の「f32 golden と対応しない丸め」が起きる。
// どちらも例外にならないので、表の側から両実装を突き合わせて固定する。
test("適合表の weight_slot が WEIGHT_SLOTS と一致する", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    assertEquals(WEIGHT_SLOTS.get(entry.op), entry.weightSlot, `${entry.op}: 重みスロット`);
  }
  // 表に載っていない op が実装側にだけある状態も落とす（op 集合の一致は別テストが見るが、
  // WEIGHT_SLOTS は op 集合の部分集合なので本数でもう一度突き合わせる）。
  assertEquals(
    fixture.ops.filter((entry) => entry.weightSlot !== undefined).length,
    WEIGHT_SLOTS.size,
    "weight_slot を持つ op の本数",
  );
});

// per-channel scale の軸（ADR 0019）。割れると「エクスポータが軸 0 で作った scale を
// カーネルが軸 1 として引く」形になり、**例外は出ずに値だけが壊れる**（conv_transpose1d の
// [Cin,Cout,K] は Cin == Cout のとき shape 検査も通ってしまう）。
test("適合表の channel_axis が WEIGHT_CHANNEL_AXES と一致する", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    assertEquals(
      WEIGHT_CHANNEL_AXES.get(entry.op),
      entry.channelAxis,
      `${entry.op}: per-channel 軸`,
    );
    // 表の側でも「重みスロットを持つ op はチャネル軸も持つ」を要求する（片方だけの行を
    // 書けてしまうと、実装側で対の関係が崩れても表が受理してしまう）。
    assertEquals(
      entry.channelAxis === undefined,
      entry.weightSlot === undefined,
      `${entry.op}: weight_slot と channel_axis は過不足なく同時に載る`,
    );
  }
  assertEquals(
    fixture.ops.filter((entry) => entry.channelAxis !== undefined).length,
    WEIGHT_CHANNEL_AXES.size,
    "channel_axis を持つ op の本数",
  );
});

test("適合表の attrs 値域を契約表の検査関数がそのとおりに判定する", async () => {
  const fixture = await loadFixture();
  // 必須と省略可能を同じ土俵で見る（値域の正本は 1 本 — 省略可能だからといって値域が
  // 緩んでよい理由は無い）。
  const declared = new Map(
    fixture.ops.map((entry) => [entry.op, [...entry.attrs, ...entry.optionalAttrs]]),
  );
  for (const entry of fixture.attrValues) {
    const contract = resolveOpContract(entry.op);
    assert(
      declared.get(entry.op)?.includes(entry.attr) === true,
      `${entry.op}: attr '${entry.attr}' が ops 節に無い`,
    );
    const check = contract.attrs[entry.attr] ?? contract.optionalAttrs?.[entry.attr];
    assert(check !== undefined, `${entry.op}: attr '${entry.attr}' の検査関数が契約表に無い`);
    // MUST: accept / reject が空のエントリは検査 0 回で緑になる（恒真）— 表の側で落とす
    // （Python 側 tests/test_ops_conformance.py の同じ門と対）。
    assert(
      entry.accept.length > 0 && entry.reject.length > 0,
      `${entry.op}.${entry.attr}: accept / reject のどちらかが空（検査 0 回で緑になる）`,
    );
    for (const value of entry.accept) {
      check(value, `${entry.op}.${entry.attr}`);
    }
    for (const value of entry.reject) {
      assertThrows(
        () => check(value, `${entry.op}.${entry.attr}`),
        OpContractError,
        undefined,
        `${entry.op}.${entry.attr} = ${JSON.stringify(value)} は受理してはならない`,
      );
    }
  }
  // 値域を持つ attr は全て表に載っている（載せ忘れた attr は値域が無検証のまま残る）
  const covered = new Set(fixture.attrValues.map((entry) => `${entry.op}.${entry.attr}`));
  for (const entry of fixture.ops) {
    for (const attr of [...entry.attrs, ...entry.optionalAttrs]) {
      assert(covered.has(`${entry.op}.${attr}`), `${entry.op}.${attr} の値域が表に無い`);
    }
  }
});

test("適合表の出力 shape 規則を computeOutputShape がそのとおりに計算する", async () => {
  const fixture = await loadFixture();
  for (const testCase of fixture.shapes) {
    const contract = resolveOpContract(testCase.op);
    const ins = testCase.ins.map((shape) => resolveDims(shape, testCase.bindings));
    const context = {
      declared:
        testCase.declared === undefined
          ? undefined
          : resolveDims(testCase.declared, testCase.bindings),
      attrs: testCase.attrs,
      bindings: testCase.bindings,
      states: testCase.states,
      stateShapes:
        testCase.stateShapes === undefined
          ? undefined
          : new Map(
              Object.entries(testCase.stateShapes).map(([slot, shape]) => [
                slot,
                resolveDims(shape, testCase.bindings),
              ]),
            ),
    };
    if (testCase.throws) {
      assertThrows(
        () => computeOutputShape(contract, ins, "t", context),
        OpContractError,
        undefined,
        `${label(testCase)} は受理してはならない（${testCase.why}）`,
      );
      continue;
    }
    const computed = computeOutputShape(contract, ins, "t", context);
    assertEquals(
      computed,
      (testCase.outs ?? []).map((shape) => resolveDims(shape, testCase.bindings)),
      label(testCase),
    );
    // shape 列の長さが契約の宣言出力数と一致する（表の outs と ops 節の out_dtypes 列が
    // 別々に育つと、shape だけ 2 本・dtype だけ 1 本という契約が書けてしまう）。
    assertEquals(computed.length, outputCountOf(contract), `${label(testCase)}: 出力数`);
  }
});

test("適合表の shape 節が全 op を受理ケースで踏む", async () => {
  const fixture = await loadFixture();
  const accepted = new Set(
    fixture.shapes.filter((testCase) => !testCase.throws).map((testCase) => testCase.op),
  );
  assertEquals(
    [...accepted].sort(),
    fixture.ops.map((entry) => entry.op).sort(),
    "shape 規則が表で踏まれていない op がある",
  );
});

// ---- Node-level contract checks (cases the table cannot express) ------------

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
// リテラル**で op 集合を固定するだけで、docs/ir-v2.md の一覧との同期は op 追加時の人手仕事
// （契約 1 セット — ops.ts / ops.py / shapes.py / fixtures / CPU 参照 / golden / ir-v2.md）。
test("契約表の op 集合が期待値リテラル 63 本と一致する", () => {
  // The op names themselves are pinned by the ops section of fixtures/op-contracts.json; the
  // family lists below are not in the table.
  assertEquals(UNARY_OPS.length, 19);
  assertEquals(BINARY_OPS.length, 6);
  // argmax は reduce 族に**入らない**（attrs も出力 dtype も rank の扱いも別 — ADR 0068 決定 2）
  assertEquals(REDUCE_OPS.length, 3);
  assertEquals(OP_CONTRACTS.size, 63);
});

test("ランタイム対応表と capability 照会は契約表から導かれる", () => {
  assertEquals(RUNTIME_SUPPORT.ops.size, OP_CONTRACTS.size);
  // 生の int32 格納は記号依存定数の焼き込み先として実行対象（ADR 0010）。f16（ADR 0018）・
  // i8（ADR 0019）・i4（ADR 0069 — 適格は linear / embedding 限定）は実行経路が入った（適格な重みは
  // 圧縮のまま常駐・適格外は CPU 展開）。bf16（ADR 0131）は f16 と同じ経路で実行できる。
  assertEquals([...RUNTIME_SUPPORT.storage], ["f32", "f16", "bf16", "i8", "i4", "i2", "i32"]);
  // 転送層の軸（未使用のグラフ入力にも効く）— i32 / bool も転送できる（ADR 0009）
  assertEquals([...RUNTIME_SUPPORT.io].sort(), ["bool", "f32", "i32"]);
  assertEquals([...IO_DTYPES].sort(), ["bool", "f32", "i32"]);
  assertEquals(capabilities().ops, [...OP_CONTRACTS.keys()].sort());
  assertEquals(capabilities().storage, ["bf16", "f16", "f32", "i2", "i32", "i4", "i8"]);
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
  assertEquals(slots("mul"), [
    ["f32", "i32"],
    ["f32", "i32"],
  ]);
  assertEquals(slots("relu"), [["f32"]]);
  assertEquals(slots("conv1d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv1d").attrKeys].sort(), ["dilation", "groups", "padding", "stride"]);
  assertEquals(slots("conv2d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv2d").attrKeys].sort(), ["dilation", "groups", "padding", "stride"]);
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
test("dtype の解禁は op ごとで、実測に出ない組み合わせは f32 のまま", () => {
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
  for (const op of [
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
    "conv3d",
    "conv_transpose1d",
  ]) {
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
test("gather はスロット別 dtype 契約を持ち、和との突合より狭く判定する", () => {
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

test("スロット別契約の出力はスロット 0 と同型で、混合は uniform の op だけが拒否する", () => {
  const resolve = (
    op: string,
    inputDtypes: readonly ("f32" | "i32" | "bool")[],
    declared: "f32" | "i32" | "bool",
  ) =>
    resolveNodeDtypes(
      resolveOpContract(op),
      node(
        op,
        inputDtypes.map((_, index) => `x${index}`),
      ),
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

// attrs を持つ op は 31 本。それ以外は attrs 空のままで、非空 attrs は fail loudly。
test("attrs を持つ op は契約表が列挙する 32 本だけで、他は attrs 空", () => {
  // The set of ops that carry attrs is pinned per op by the attrs column of the table; this test
  // keeps the node-level rejections.
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
test("deform_conv2d の attrs は padding 1 本だけ・アリティ 5 固定", () => {
  assertEquals(
    assertNodeContract(
      node("deform_conv2d", ["x", "w", "off", "mask", "b"], { padding: [1, 0] }),
      "t",
    ).kind,
    "deformConv2d",
  );
  // stride / dilation / groups / offset_groups の欄は無い = 宣言外キーとして落ちる
  for (const extra of [
    { stride: [1, 1] },
    { dilation: [1, 1] },
    { groups: 1 },
    { offset_groups: 1 },
  ]) {
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
  for (const ins of [
    ["x", "w", "off", "b"],
    ["x", "w", "off", "mask", "b", "extra"],
  ]) {
    assertThrows(
      () => assertNodeContract(node("deform_conv2d", ins, { padding: [1, 0] }), "t"),
      OpContractError,
    );
  }
});

// upsample_bilinear2d は **align_corners=True 専業**（第 1 層）。「True 以外を受理しない」を
// 値の検査ではなく**欄の不存在**で表しているので、その欄が生えていないことを契約側から固定
// する（ADR 0023 決定 4 の規律 — 欄ができた瞬間に False も表現できてしまう）。
test("upsample_bilinear2d の attrs は output_size 1 本だけ（align_corners の欄が無い）", () => {
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
  assertThrows(() => assertNodeContract(node("upsample_bilinear2d", ["x"]), "t"), OpContractError);
  // アリティ 1 固定（重みも bias も持たない）
  assertThrows(
    () => assertNodeContract(node("upsample_bilinear2d", ["x", "w"], { output_size: [7, 9] }), "t"),
    OpContractError,
  );
});

// ADR 0012: attrs スキーマは「宣言キーは全て必須・宣言外は fail loudly・値も検査する」。
test("attrs スキーマは必須キーの欠落と契約外の値を拒否する（cast の to）", () => {
  assertEquals(assertNodeContract(node("cast", ["a"], { to: "i32" }), "t").kind, "cast");
  // 必須キーの欠落（M0 の「未知キーだけ見る」検査ではここが素通りした）
  assertThrows(() => assertNodeContract(node("cast", ["a"]), "t"), OpContractError);
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
test("permute の dims は非負・重複無しの並べ替え表だけを受理する", () => {
  assertEquals(
    assertNodeContract(node("permute", ["a"], { dims: [0, 2, 1] }), "t").kind,
    "permute",
  );
  assertEquals(permuteDims({ dims: [1, 0] }, "t"), [1, 0]);
  assertThrows(() => permuteDims({}, "t"), OpContractError);
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
test("出力 dtype は契約から導かれ、宣言との食い違いを拒否する", () => {
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
  assertThrows(() => resolve("cast", ["a"], ["f32"], "f32", { to: "i32" }), OpContractError);
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
test("出力 dtype 写像の定義域がスロット 0 の受理集合と一致する", () => {
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
});

// 波3 の数理 op。attrs の値域とキーを跨ぐ不変条件（clamp の min <= max）を固定する。
test("clamp / leaky_relu / 比較の attrs スキーマが値域まで検査する", () => {
  assertEquals(assertNodeContract(node("clamp", ["a"], { min: -1, max: 1 }), "t").kind, "unary");
  // 必須キーの欠落（既定値補完はしない — ADR 0015 / 0012）
  assertThrows(() => assertNodeContract(node("clamp", ["a"], { min: -1 }), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("leaky_relu", ["a"], {}), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("ge_scalar", ["a"], {}), "t"), OpContractError);

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
test("スカラ params の並び表と単項 attrs スキーマが同じ op を覆う", () => {
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
test("cumsum は最終次元固定で、shape を素通しにする", () => {
  const contract = resolveOpContract("cumsum");
  assertEquals(cumsumDim({ dim: 2 }, "t"), 2);
  assertThrows(() => cumsumDim({}, "t"), OpContractError);
  assertThrows(() => cumsumDim({ dim: -1 }, "t"), OpContractError);
  const shape = (ins: readonly number[], dim: number) =>
    computeOutputShape(contract, [ins], "t", { attrs: { dim } })[0];
  assertEquals(shape([1, 3, 4], 2), [1, 3, 4]);
  // 長さ 0 の軸は素通し（前縁和の identity は 0 — amax / softmax と違って定義できる）
  assertEquals(shape([3, 0], 1), [3, 0]);
});

// argmax は reduce 族と**別の契約**（ADR 0068 決定 2）。絞りを「値の検査」ではなく
// **欄の不存在**で表しているので、欄が生えていないことを契約側から固定する。
test("argmax は attrs 空・最終次元固定・rank 保存で、出力は i32", () => {
  const contract = resolveOpContract("argmax");
  assertEquals(contract.kind, "argmax");
  assertEquals(resolveNodeDtypes(contract, node("argmax", ["x"]), ["f32"], ["i32"], "t"), ["i32"]);
  // 宣言が f32 のグラフは拒否する（readback が別の TypedArray として読む沈黙誤値になる）
  assertThrows(
    () => resolveNodeDtypes(contract, node("argmax", ["x"]), ["f32"], ["f32"], "t"),
    OpContractError,
  );
  // i32 / bool 入力は語彙に無い（f32 専業 — 実測は logits だけ）
  assertThrows(() => assertDtype(contract, "i32", "t"), OpContractError);
  const shape = (ins: readonly number[]) => computeOutputShape(contract, [ins], "t")[0];
  assertEquals(shape([2, 3, 4]), [2, 3, 1]);
  // 契約外 attrs は fail loudly（attrs 空契約 — ADR 0012）
  assertThrows(() => assertNodeContract(node("argmax", ["x"], { dim: 1 }), "t"), OpContractError);
});

// topk は**唯一の多出力 op**（ADR 0068 決定 3）。受理領域（1 ≤ k ≤ 最終次元）と 2 本の出力の
// dtype / shape が 1 本の契約から出ること、`largest` / `sorted` の欄が生えていないことを固定する。
test("topk は attrs k 宣言必須・出力 2 本（値 f32 + 添字 i32）で、受理領域は 1 ≤ k ≤ 最終次元", () => {
  const contract = resolveOpContract("topk");
  assertEquals(contract.kind, "topk");
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
    "output count is 1 (contract: 2)",
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
  assertEquals(shapes([2, 3, 5], 1), [
    [2, 3, 1],
    [2, 3, 1],
  ]);
  // 受理領域の外は全て fail loudly（縮退しない）
  assertThrows(() => shapes([6, 10], 11), OpContractError, "last dimension");
  assertThrows(() => shapes([6, 10], -1), OpContractError);
  assertThrows(() => shapes([6, 10], 1.5), OpContractError);
  // k は宣言必須（欄が無い = 既定値補完しない）
  assertThrows(() => computeOutputShape(contract, [[6, 10]], "t"), OpContractError);
  assertThrows(() => assertNodeContract(topkNode({}), "t"), OpContractError, "required attr");
  // 契約外 attrs（torch の largest / sorted）は fail loudly
  assertThrows(
    () => assertNodeContract(topkNode({ k: 2, sorted: true }), "t"),
    OpContractError,
    "attrs outside the contract",
  );
});

test("where は三者を右詰め broadcast し、出力は値の側と同形になる", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("where"), ins, "t")[0];
  // 条件が値より低い rank（spline の inside 判定の形）
  assertEquals(shape([[5], [2, 3, 5], [1]]), [2, 3, 5]);
  // 条件だけが広い形も torch と同じく通る（出力は 3 者の broadcast）
  assertEquals(shape([[2, 3], [3], [3]]), [2, 3]);
});

test("broadcast は torch 準拠の右詰めで、長さ 0 の次元も max で潰さない", () => {
  assertEquals(broadcastShapes([2, 3, 4], [3, 1], "t"), [2, 3, 4]);
  assertEquals(broadcastShapes([3, 1], [2, 3, 4], "t"), [2, 3, 4]);
  assertEquals(broadcastShapes([5], [], "t"), [5]);
  assertEquals(broadcastShapes([], [], "t"), []);
  // MUST: max(0, 1) = 1 ではなく 0（torch の broadcast 規則）
  assertEquals(broadcastShapes([0], [1], "t"), [0]);
  assertThrows(() => broadcastShapes([2, 3], [4, 3], "t"), OpContractError);
  assertThrows(() => broadcastShapes([0], [5], "t"), OpContractError);
});

test("出力 shape 計算が op ごとの契約どおりに決まる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs?: Readonly<Record<string, unknown>>,
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];
  assertEquals(shape("sum", [[4]], { dim: 0 }), []);
  assertEquals(shape("amin", [[2, 3, 4]], { dim: 0 }), [3, 4]);
  // cast / bitwise_not は要素ごとなので shape は素通し
  assertEquals(shape("cast", [[2, 3, 4]]), [2, 3, 4]);
});

// ADR 0012: bmm は rank-3 専業（rank-2 は matmul）、gather は最終次元固定。
// MUST: B / M / K / N を全て違う長さで確かめる — 軸の取り違えは正方形では見えない。
test("bmm は rank-3 のバッチ一致・縮約一致だけを受理する", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("bmm"), ins, "t")[0];
  assertEquals(
    shape([
      [7, 1, 4],
      [7, 4, 9],
    ]),
    [7, 1, 9],
  );
  // rank-2 は matmul の担当（兼用にしない）
  assertThrows(
    () =>
      shape([
        [2, 6, 3],
        [3, 5],
      ]),
    OpContractError,
  );
  assertThrows(
    () =>
      shape([
        [2, 2, 6, 3],
        [2, 2, 3, 5],
      ]),
    OpContractError,
  );
  // バッチ次元の不一致（broadcast も stride 0 も語彙に無い）
  assertThrows(
    () =>
      shape([
        [2, 6, 3],
        [3, 3, 5],
      ]),
    OpContractError,
  );
});

test("gather は先行次元一致を要求し、出力は index と同形になる", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("gather"), ins, "t")[0];
  // 出力の最終次元は src より長くてもよい（同じ添字を何度引いてもよい）
  assertEquals(
    shape([
      [2, 3],
      [2, 7],
    ]),
    [2, 7],
  );
  assertEquals(shape([[5], [3]]), [3]);
  // 先行次元の不一致（torch の一般 gather より狭い契約）
  assertThrows(
    () =>
      shape([
        [4, 6, 9],
        [3, 6, 6],
      ]),
    OpContractError,
  );
  // rank 不一致・スカラ
  assertThrows(() => shape([[], []]), OpContractError);
});

// ADR 0010: sym_prefix_slice は attrs（+ 束縛）から出力 shape を計算する — 宣言は照合される
// 側で、reshape / expand のような「宣言が目標形」ではない。
test("sym_prefix_slice は attrs と束縛から prefix 長を決める", () => {
  const shape = (
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown>,
    bindings?: Record<string, number>,
  ) => computeOutputShape(resolveOpContract("sym_prefix_slice"), ins, "t", { attrs, bindings })[0];
  const slice = (dim: number, coeff = 1, offset = 0) => ({ dim, coeff, offset });

  // Tmax 超過（定数バッファの範囲外読み出しになる）
  assertThrows(() => shape([[8, 8]], { sym: "T", slices: [slice(0)] }, { T: 9 }), OpContractError);
  // 入力 rank の外の dim / 束縛が渡っていない形
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
  bad({ sym: "1T", slices: [slice(0)] });
  bad({ slices: [slice(0)] });
  assertEquals(
    assertNodeContract(node("sym_prefix_slice", ["a"], { sym: "T", slices: [slice(0)] }), "t").kind,
    "symPrefixSlice",
  );
});

// ADR 0011: reshape / expand は「出力の宣言 shape が目標形」で、入力からは導けない。
test("reshape / expand は宣言 shape を目標形に取り、permute は dims で決まる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    context: { declared?: readonly number[]; attrs?: Record<string, unknown> },
  ) => computeOutputShape(resolveOpContract(op), ins, "t", context)[0];

  assertEquals(shape("reshape", [[6]], { declared: [1, 6, 1] }), [1, 6, 1]);

  // 長さ 1 でない次元の拡張・rank 下げ
  assertThrows(() => shape("expand", [[2, 3]], { declared: [3] }), OpContractError);
  // dims と入力 rank の不一致 / 範囲外
  assertThrows(() => shape("permute", [[2, 3]], { attrs: { dims: [0, 2] } }), OpContractError);
  // MUST: 目標形を渡さずに黙って推測しない
  assertThrows(() => shape("expand", [[2, 3]], {}), OpContractError);
  assertThrows(() => shape("permute", [[2, 3]], {}), OpContractError);
});

// ADR 0012: 融合 op の attrs スキーマ（許容キー + 型 + 値域）。
test("融合 op の attrs スキーマが値域まで検査する", () => {
  // Key sets and the value domains of each attr are table rows; this test keeps the node-level
  // checks (missing / undeclared keys), the extractors, and values the table does not list.
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
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8] });
  // f32 へ丸めると 0 になる eps は「0 を許さない」理由が GPU 側だけで復活する（GPU は params の
  // f32 語で運び、CPU 参照は f64 のまま計算する）— 全ゼロ行で GPU が NaN・CPU が 0 に分岐する
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-50 });
  // f32 の subnormal へ丸まる eps も拒否（WGSL は subnormal 保存を要求しない — FTZ 実装で
  // 0 に潰れ、上と同じ分岐が GPU 側だけで復活する）。境界 = 最小正規化数 2^-126
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-45 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-40 });
  assertEquals(
    accept("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1.1754943508222875e-38 })
      .kind,
    "layerNorm",
  );

  // rms_norm — attrs は eps だけ（正規化長の正本は weight — ADR 0017）
  assertEquals(accept("rms_norm", ["x", "w"], { eps: 1e-6 }).kind, "rmsNorm");
  assertEquals(rmsNormEps({ eps: 1e-6 }, "t"), 1e-6);
  reject("rms_norm", ["x", "w"], { eps: 1e-50 });
  reject("rms_norm", ["x", "w"], {});
  // MUST: normalized_shape の欄は無い（layer_norm から欄ごと写した IR を通さない）
  reject("rms_norm", ["x", "w"], { eps: 1e-6, normalized_shape: [8] });

  // clamp_min — attrs は min だけ（max を持ち込むと両側 clamp と区別が消える）
  assertEquals(accept("clamp_min", ["x"], { min: 1e-12 }).kind, "unary");
  assertEquals(scalarParamValues(resolveOpContract("clamp_min"), { min: -1.5 }, "t"), [-1.5]);
  reject("clamp_min", ["x"], {});
  reject("clamp_min", ["x"], { min: 0, max: 1 });

  // softmax — 非負の軸番号のみ（負の軸表記はエクスポータ境界で正規化する）
  assertEquals(accept("softmax", ["x"], { dim: 0 }).kind, "softmax");
  assertEquals(softmaxDim({ dim: 3 }, "t"), 3);
  reject("softmax", ["x"], {});

  // safe_softmax — attrs は softmax と同一（ADR 0044。kind だけが違う）
  assertEquals(accept("safe_softmax", ["x"], { dim: 0 }).kind, "safeSoftmax");
  reject("safe_softmax", ["x"], {});

  // embedding — padding_idx は受理して不活性（-1 が torch の「未指定」番兵）
  assertEquals(accept("embedding", ["w", "i"], { padding_idx: -1 }).kind, "embedding");
  reject("embedding", ["w", "i"], {});

  // masked_fill — 有限の f32 スカラのみ
  assertEquals(accept("masked_fill", ["x", "m"], { value: 0 }).kind, "maskedFill");
  assertEquals(maskedFillValue({ value: -1.5 }, "t"), -1.5);
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
  assertEquals(conv2dAttrs({ stride: [2, 3], padding: [1, 0], dilation: [1, 2], groups: 4 }, "t"), {
    stride: [2, 3],
    padding: [1, 0],
    dilation: [1, 2],
    groups: 4,
  });
  // 宣言済み attrs の既定値補完はしない（4 つとも必須）
  reject("conv2d", ["x", "w", "b"], { stride: [1, 1], padding: [0, 0], dilation: [1, 1] });
  reject("conv2d", ["x", "w", "b"], { stride: [1, 1], padding: [0, 0], groups: 1 });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, output_padding: [0, 0] });

  // conv3d — 空間 3 つは [T, H, W] の 3 成分（スカラ表記・2 成分は受理しない — ADR 0118 決定 1）
  const conv3Attrs = {
    stride: [1, 1, 1],
    padding: [0, 0, 0],
    dilation: [1, 1, 1],
    groups: 1,
  };
  assertEquals(accept("conv3d", ["x", "w", "b"], conv3Attrs).kind, "conv3d");
  assertEquals(
    conv3dAttrs({ stride: [1, 2, 3], padding: [0, 1, 2], dilation: [2, 1, 3], groups: 4 }, "t"),
    { stride: [1, 2, 3], padding: [0, 1, 2], dilation: [2, 1, 3], groups: 4 },
  );
  // MUST: 3 軸を独立に見る（どの 1 成分だけが 0 でも落ちる — stride 0 はカーネルのハング）
  reject("conv3d", ["x", "w", "b"], { ...conv3Attrs, padding: [0, -1, 0] });
  // 長さ 3 以外・スカラ表記は受理しない（[H, W] の 2 成分は時間軸の書き落としと区別できない）
  reject("conv3d", ["x", "w", "b"], { ...conv3Attrs, dilation: [1, 1, 1, 1] });
  // 宣言済み attrs の既定値補完はしない（4 つとも必須）
  reject("conv3d", ["x", "w", "b"], { stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1] });
  reject("conv3d", ["x", "w", "b"], { stride: [1, 1, 1], padding: [0, 0, 0], groups: 1 });
  // アリティ 3 固定（bias 無しはエクスポータのゼロ bias 合成で正規化 — conv 族と同じ）
  reject("conv3d", ["x", "w"], conv3Attrs);

  // conv_transpose1d — attrs は stride / padding のみ（output_padding / dilation / groups の
  // 欄を作らないことが「実測外の値を黙って既定値で実行する」経路を潰している）
  assertEquals(
    accept("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: 0 }).kind,
    "convTranspose1d",
  );
  assertEquals(convTranspose1dAttrs({ stride: 8, padding: 4 }, "t"), { stride: 8, padding: 4 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: 0, output_padding: 0 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2 });
});

/**
 * ADR 0012: masked_fill の実測埋め値 −3.4028234663852886e+38（f32 の最小有限値）が
 * **JSON 往復で ulp 不変**であることを固定する。IR の attrs は JSON の数値リテラルとして
 * 運ばれるので、往復で 1 ulp でも動くと GPU に載る埋め値が torch と食い違う。
 */
test("masked_fill の埋め値 −3.4e38 は JSON 往復と f32 丸めで ulp が動かない", () => {
  const value = -3.4028234663852886e38;
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

test("融合 op の出力 shape が契約どおりに決まる", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown> = {},
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];

  // linear — 先行次元はそのまま、最終次元が out へ差し替わる
  assertEquals(shape("linear", [[2, 4, 6], [5, 6], [5]]), [2, 4, 5]);

  // layer_norm — normalized_shape は入力の最終次元と一致し、affine も同形
  const ln = { normalized_shape: [8], eps: 1e-5 };
  assertEquals(shape("layer_norm", [[2, 3, 8], [8], [8]], ln), [2, 3, 8]);
  assertThrows(() => shape("layer_norm", [[3, 8], [8], [3, 8]], ln), OpContractError);

  // softmax — 最終次元のみ（一般 dim は語彙に無い）
  assertEquals(shape("softmax", [[4, 9]], { dim: 1 }), [4, 9]);
  assertThrows(() => shape("softmax", [[2, 3, 4]], { dim: 3 }), OpContractError);

  // safe_softmax — shape 規則は softmax と同一
  assertEquals(shape("safe_softmax", [[4, 9]], { dim: 1 }), [4, 9]);

  // embedding — 出力は index の形に hidden を足したもの
  assertEquals(shape("embedding", [[7, 4], [5]]), [5, 4]);

  // masked_fill — 出力は常に x と同形（mask は右詰め broadcast で読むだけ）
  const fill = { value: 0 };
  // MUST: mask が x を広げる形は拒否（broadcastShapes をそのまま使うと通ってしまう）
  assertThrows(
    () =>
      shape(
        "masked_fill",
        [
          [3, 4],
          [2, 3, 4],
        ],
        fill,
      ),
    OpContractError,
  );
  assertThrows(
    () =>
      shape(
        "masked_fill",
        [
          [3, 4],
          [3, 5],
        ],
        fill,
      ),
    OpContractError,
  );

  // conv1d — 出力長は floor((L + 2P − D·(K−1) − 1) / S) + 1（ADR 0015 の dilation 一般形）
  const conv = { stride: 1, padding: 1, dilation: 1, groups: 1 };
  assertEquals(shape("conv1d", [[1, 1, 3], [2, 1, 3], [2]], { ...conv, padding: 2 }), [1, 2, 5]);
  // dilation の一般形（DDSConv: depthwise g=C・k=5・d=3・p=6 で長さが保たれる）
  assertEquals(
    shape("conv1d", [[1, 6, 17], [6, 1, 5], [6]], { ...conv, padding: 6, dilation: 3, groups: 6 }),
    [1, 6, 17],
  );
  // groups は Cin / Cout の両方を割り切る必要がある（重みの第 2 軸は Cin/groups）
  assertThrows(
    () => shape("conv1d", [[1, 6, 11], [8, 2, 3], [8]], { ...conv, groups: 3 }),
    OpContractError,
  );
  assertThrows(() => shape("conv1d", [[3, 9], [4, 3, 3], [4]], conv), OpContractError);

  // conv_transpose1d — 重みは [Cin, Cout, K]・出力長は L·stride（2P == K − S の形のみ）
  assertEquals(
    shape("conv_transpose1d", [[1, 3, 5], [3, 2, 16], [2]], { stride: 8, padding: 4 }),
    [1, 2, 40],
  );
});

// ADR 0011: strided コピー族（permute / expand / sym_prefix_slice / masked_fill）の rank 上限は
// **契約層**で見る。codegen（stridedParams）まで落とすと、利用者には「契約検査は通ったのに
// 実行段で内部エラー」として出て、どの op のどの値が rank 超過なのか診断に残らない。
test("strided コピー族の rank 上限（1..4）は契約層で落ちる", () => {
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
 * **params 層が持つ門は契約層にも同じだけある**（カーネル側 doc が「契約検査と二重」と
 * 書いている門は特に）。片側にしか無いと受理集合が層で割れ、`computeOutputShape` を通る
 * CPU 参照だけが実行できる形（GPU は params で例外）が作れる。eps の f32 上振れに至っては
 * 例外すら出ず、GPU（`1/sqrt(x + Inf) = 0` で bias 一色）と参照（有限 f64 で正規化）が
 * 黙って別の値を出す。
 *
 * MUST: 診断は**原因の値**を名乗る（層が割れていた頃の症状は「契約は通ったのに実行段で
 * 内部エラー」で、どの attr / どの軸が域外なのかが利用者に残らなかった）。
 */
test("params 層の値域門は契約層にも揃っている（f32 域 / カーネル長 / 幾何 / 隠れ幅）", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown> = {},
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];
  const reject = (op: string, ins: readonly string[], attrs: Record<string, unknown>) =>
    assertThrows(() => assertNodeContract(node(op, ins, attrs), "t"), OpContractError, "1e+39");

  // eps は **f32 として**有限（下振れ 1e-50 の対で、上振れ 1e39 も落とす）
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e39 });
  reject("rms_norm", ["x", "w"], { eps: 1e39 });
  assertThrows(
    () => shape("layer_norm", [[3, 8], [8], [8]], { normalized_shape: [8], eps: 1e39 }),
    OpContractError,
    "infinite",
  );
  assertThrows(() => shape("rms_norm", [[3, 8], [8]], { eps: 1e39 }), OpContractError, "infinite");
  // 正常な eps はどちらも通る（門を広げすぎていないことの対）
  assertEquals(
    shape("layer_norm", [[3, 8], [8], [8]], { normalized_shape: [8], eps: 1e-5 }),
    [3, 8],
  );
  assertEquals(shape("rms_norm", [[3, 8], [8]], { eps: 1e-5 }), [3, 8]);

  // params の f32 語へ載るスカラ attr も同じ門（「有限の f32 スカラ」の宣言どおり）
  reject("attention", ["q", "k", "v"], { scale: 1e39 });
  reject("masked_fill", ["x", "m"], { value: 1e39 });
  reject("clamp", ["x"], { min: -1e39, max: 1 });
  reject("clamp", ["x"], { min: 0, max: 1e39 });
  reject("clamp_min", ["x"], { min: 1e39 });
  reject("leaky_relu", ["x"], { negative_slope: 1e39 });
  reject("ge_scalar", ["x"], { value: 1e39 });
  // f32 へ普通に丸まる値・f32 の最大有限値ちょうどは通る（適用時に丸める規約で両側が一致する）
  assertEquals(
    assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.1 }), "t").kind,
    "attention",
  );
  assertEquals(
    assertNodeContract(node("clamp", ["x"], { min: -3.4028234663852886e38, max: 0.1 }), "t").kind,
    "unary",
  );

  // conv1d / conv2d の K = 0 は**出力長式より前**に落とす（`D·(K−1)` が負に転ぶと、出力が
  // 入力より長い形が導出される — params 側は正整数として落とすので契約層だけが受理していた）
  const conv1 = { stride: 1, padding: 0, dilation: 1, groups: 1 };
  assertThrows(
    () => shape("conv1d", [[1, 1, 4], [1, 1, 0], [1]], conv1),
    OpContractError,
    "kernel length K must be a positive integer",
  );
  assertEquals(shape("conv1d", [[1, 1, 4], [1, 1, 1], [1]], conv1), [1, 1, 4]);
  const conv2 = { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 };
  assertThrows(
    () => shape("conv2d", [[1, 1, 4, 4], [1, 1, 0, 1], [1]], conv2),
    OpContractError,
    "kernel length Kh must be a positive integer",
  );
  assertThrows(
    () => shape("conv2d", [[1, 1, 4, 4], [1, 1, 1, 0], [1]], conv2),
    OpContractError,
    "kernel length Kw must be a positive integer",
  );
  assertEquals(shape("conv2d", [[1, 1, 4, 4], [1, 1, 1, 1], [1]], conv2), [1, 1, 4, 4]);

  // deform_conv2d — 静的な幾何は params の positive 集合と同じ集合（空間長 0 は padding 次第で
  // 正の出力形まで導出でき、全タップ範囲外 = bias 一色を参照が有効値として返していた）
  assertThrows(
    () =>
      shape("deform_conv2d", [[1, 1, 0, 4], [1, 1, 1, 1], [1, 2, 2, 4], [1, 1, 2, 4], [1]], {
        padding: [1, 0],
      }),
    OpContractError,
    "height_in must be a positive integer",
  );
  assertThrows(
    () =>
      shape("deform_conv2d", [[1, 1, 2, 4], [1, 1, 0, 1], [1, 0, 5, 4], [1, 0, 5, 4], [1]], {
        padding: [1, 0],
      }),
    OpContractError,
    "kernel_h must be a positive integer",
  );
  assertEquals(
    shape("deform_conv2d", [[1, 2, 4, 4], [3, 2, 3, 3], [1, 18, 4, 4], [1, 9, 4, 4], [3]], {
      padding: [1, 1],
    }),
    [1, 3, 4, 4],
  );

  // gru_scan — H の域（1 〜 256）は docs/limitations.md が公開している by-design 制約で、
  // params だけが見ていた（H=0 / H=257 は契約を通り、参照だけが実行できていた）
  assertThrows(
    () => shape("gru_scan", [[1, 1, 0], [1, 0], [0, 0], [0]]),
    OpContractError,
    "hidden width H 0",
  );
  assertThrows(
    () => shape("gru_scan", [[1, 1, 771], [1, 257], [771, 257], [771]]),
    OpContractError,
    "hidden width H 257",
  );
  assertThrows(
    () => shape("gru_scan_reverse", [[1, 1, 771], [1, 257], [771, 257], [771]]),
    OpContractError,
    "hidden width H 257",
  );
  // 上限ちょうどは通る（上限を下げる誤りもここで赤くなる）
  assertEquals(shape("gru_scan", [[1, 1, 768], [1, 256], [768, 256], [768]]), [1, 1, 256]);
});

/**
 * 融合 attention の**省略可能な第 4 入力 = 加算 mask**（ADR 0023 改訂）。
 *
 * カーネルは mask を `[1,1,M,N]` としてバッチ base 抜きの平坦添字で読む（B·H 全体へ
 * broadcast）ので、そこから外れた形は**値が静かに壊れる**（例外は出ない）。契約層が
 * 唯一の検出器なので、受理と拒否の全面をここで固定する。
 */
test("attention の mask は f32・[1,1,M,N] ちょうどだけを受理する", () => {
  const contract = resolveOpContract("attention");
  const q = [2, 3, 5, 4];
  const k = [2, 3, 7, 4];
  const attrs = { scale: 0.5 };
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(contract, ins, "t", { attrs })[0];

  // アリティは 3 か 4（可変ではない — 「何本でも」は cat だけ）
  assertEquals(describeArity(contract), "3 or 4");
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
    "3 or 4",
  );

  // 受理: mask 無し / [1,1,M,N]（M ≠ N の非正方も）
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
  assertEquals(resolveNodeDtypes(contract, maskNode, ["f32", "f32", "f32", "f32"], ["f32"], "t"), [
    "f32",
  ]);
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

test("state_append は出力 0 本の effect op で、states 欄が必須", () => {
  const contract = resolveOpContract("state_append");
  assertEquals(contract.kind, "stateAppend");

  assertStrictEquals(assertNodeContract(appendNode(), "t"), contract);
  // 出力を書いた形は「値を定義しない op」の契約に反する
  assertThrows(
    () => assertNodeContract({ ...appendNode(), outs: ["y"] }, "t"),
    OpContractError,
    "output count is 1 (contract: 0)",
  );
  // states 欄そのものが必須（required: true）
  assertThrows(
    () => assertNodeContract({ ...appendNode(), states: {} }, "t"),
    OpContractError,
    "states field is missing",
  );
  // キー集合はちょうど { slot }
  assertThrows(
    () => assertNodeContract(appendNode({}, { slot: "kv.k", extra: "kv.v" }), "t"),
    OpContractError,
    "states field keys are",
  );
  assertThrows(
    () => assertNodeContract(appendNode({}, { k: "kv.k" }), "t"),
    OpContractError,
    "states field keys are",
  );
});

// 0 本を許すのは契約が effect を宣言する op **だけ**（パーサは本数に意味を与えない）。
test("effect でない op の outs 空は契約層が落とす", () => {
  for (const [op, ins] of [
    ["relu", ["a"]],
    ["add", ["a", "b"]],
    ["topk", ["a"]],
  ] as const) {
    assertThrows(
      () => assertNodeContract({ ...node(op, ins, op === "topk" ? { k: 2 } : {}), outs: [] }, "t"),
      OpContractError,
      "output count is 0",
    );
  }
});

test("states 欄を持たない op に states を書けない", () => {
  assertThrows(
    () => assertNodeContract(node("relu", ["a"], {}, { slot: "kv.k" }), "t"),
    OpContractError,
    "has no states field",
  );
  // topk（多出力 op）も同じ — states 欄は attention / state_append の 2 本だけの契約面
  assertThrows(
    () =>
      assertNodeContract(
        { ...node("topk", ["a"], { k: 2 }, { slot: "kv.k" }), outs: ["v", "i"] },
        "t",
      ),
    OpContractError,
    "has no states field",
  );
});

test("attention は B = 0 / M = 0 の退化した q を契約層で名指しして落とす（W-R8-1）", () => {
  const contract = resolveOpContract("attention");
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(contract, ins, "t", { attrs: { scale: 0.5 } })[0];
  // 対照: 正常形は出力 [B,H,M,D] を返す
  assertEquals(
    shape([
      [1, 2, 4, 8],
      [1, 2, 4, 8],
      [1, 2, 4, 8],
    ]),
    [1, 2, 4, 8],
  );
  // B = 0 は「完全一致」を満たすので素通りしていた — 実行相の行ブロック planner が attention を
  // 名乗らない診断で落ちる前に、契約層で名指しする
  assertThrows(
    () =>
      shape([
        [0, 2, 4, 8],
        [0, 2, 4, 8],
        [0, 2, 4, 8],
      ]),
    OpContractError,
    "attention B",
  );
  // 従来形の M = 0（states 形は元から拒否していた）
  assertThrows(
    () =>
      shape([
        [1, 2, 0, 8],
        [1, 2, 4, 8],
        [1, 2, 4, 8],
      ]),
    OpContractError,
    "M = 0",
  );
});

test("attention の states 形は欄の有無で判別され、従来形の受理集合は動かない", () => {
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
    "states form takes exactly 3 inputs",
  );
  // キー集合はちょうど { k, v }
  assertThrows(
    () =>
      assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }, { k: "kv.k" }), "t"),
    OpContractError,
    "states field keys are",
  );
  // k と v に同じスロットを書いた取り違えは shape では捕まらない（k/v スロットは同形）
  assertThrows(
    () =>
      assertNodeContract(
        node("attention", ["q", "k", "v"], { scale: 0.5 }, { k: "kv.k", v: "kv.k" }),
        "t",
      ),
    OpContractError,
    "the same slot 'kv.k' from more than one field",
  );
});

test("省略可能 attr `window` は states 欄を持つノードでのみ宣言できる", () => {
  const states = { k: "kv.k", v: "kv.v" };

  // states 形では宣言でき、省略もできる（欄の不存在 = 全 context）
  assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5, window: 512 }, states), "t");
  assertNodeContract(appendNode({ window: 512 }), "t");
  assertNodeContract(appendNode({}), "t");
  // 従来形に書くと「誰も読まない attr」になるので拒否する（受理集合の対称性）
  assertThrows(
    () => assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5, window: 512 }), "t"),
    OpContractError,
    "can be declared only on a node with a states field",
  );
  // MUST: JSON の `512.0` / `5.12e2` は受理する（JSON.parse は整数値の float を単一の number へ
  // 潰すので TS 側にはそもそも区別が無い）。Python の `_assert_integer_attr` はこの受理集合へ
  // 合わせて整数値 float を int へ正規化する — 片側だけが拒むと「ランタイムが読める graph を
  // エクスポータの検証だけが読めない」乖離になるので、両側の受理集合をここで明示しておく。
  for (const text of ["512.0", "5.12e2"]) {
    const window: unknown = JSON.parse(text);
    assertEquals(window, 512, `${text}: JSON.parse が整数へ潰していない`);
    assertNodeContract(appendNode({ window }), "t");
    assertEquals(stateWindow({ window }, "t"), 512);
  }
  // capability 射影には必須と省略可能の**和**が載る（列挙門が states 形を拒否しないため）
  assertEquals([...(RUNTIME_SUPPORT.ops.get("attention")?.attrKeys ?? [])].sort(), [
    "readonly",
    "scale",
    "window",
  ]);
  assertEquals([...(RUNTIME_SUPPORT.ops.get("state_append")?.attrKeys ?? [])], ["window"]);
  // 契約外 attrs の判定は従来どおり（省略可能の受理は window 1 本だけを広げる）
  assertThrows(
    () => assertNodeContract(appendNode({ windows: 512 }), "t"),
    OpContractError,
    "attrs outside the contract [windows]",
  );
});

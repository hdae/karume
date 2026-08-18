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

import { assert, assertEquals, assertThrows } from "@std/assert";
import { STRIDED_RANK } from "../src/codegen/strided.ts";
import { evalDim, parseDim } from "../src/format/dims.ts";
import type { IrDim } from "../src/format/ir.ts";
import {
  attrKeysOf,
  computeOutputShape,
  OP_CONTRACTS,
  OpContractError,
  optionalAttrKeysOf,
  outputCountOf,
  resolveOpContract,
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
        strings(slot, `dtypes.slots[${index}]`)
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
    optionalAttrs: raw["optional_attrs"] === undefined
      ? []
      : strings(raw["optional_attrs"], "optional_attrs"),
    states: states === undefined || typeof required !== "boolean" ? undefined : {
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
      dims(shape, `${op}.ins[${index}]`)
    ),
    attrs: attrs === undefined ? undefined : attrs,
    declared: declared === undefined ? undefined : dims(declared, `${op}.declared`),
    bindings,
    states: states === undefined ? undefined : Object.fromEntries(
      Object.entries(states).map(([key, slot]) => {
        if (typeof slot !== "string") {
          throw new Error(`fixture: ${op}.states[${key}] が文字列でない`);
        }
        return [key, slot];
      }),
    ),
    stateShapes: stateShapes === undefined ? undefined : Object.fromEntries(
      Object.entries(stateShapes).map((
        [slot, shape],
      ) => [slot, dims(shape, `${op}.state_shapes['${slot}']`)]),
    ),
    outs: outs === undefined
      ? undefined
      : array(outs, `${op}.outs`).map((shape, index) => dims(shape, `${op}.outs[${index}]`)),
    throws,
    why: typeof raw["why"] === "string" ? raw["why"] : "",
  };
};

const loadFixture = async (): Promise<Fixture> => {
  const text = await Deno.readTextFile(new URL("./fixtures/op-contracts.json", import.meta.url));
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

Deno.test("適合表の op 集合が契約表と完全一致する", async () => {
  const fixture = await loadFixture();
  assertEquals(
    fixture.ops.map((entry) => entry.op).sort(),
    [...OP_CONTRACTS.keys()].sort(),
  );
  // 表の中で op が重複していない（重複すると片方の宣言が黙って無視される）
  assertEquals(new Set(fixture.ops.map((entry) => entry.op)).size, fixture.ops.length);
});

Deno.test("適合表の strided rank 上限が実装の定数と一致する", async () => {
  const fixture = await loadFixture();
  assertEquals(fixture.stridedRankMax, STRIDED_RANK);
});

Deno.test("適合表のアリティ / スロット dtype / attrs キーが契約表と一致する", async () => {
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
      contract.states === undefined ? undefined : {
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
Deno.test("適合表の出力 dtype 写像が契約表と一致する", async () => {
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
      Object.fromEntries([...domain].sort().map((dtype) => [dtype, slot[dtype] ?? dtype]))
    );
    assertEquals(
      contract.outputDtypes.map((slot) =>
        Object.fromEntries([...slot].sort(([a], [b]) => a.localeCompare(b)))
      ),
      expected,
      `${entry.op}: 出力 slot 別の dtype 写像`,
    );
    // 出力数は写像の列長そのもの（表と実装で本数が割れると、2 本目の出力の dtype が
    // 誰にも照合されない状態になる）。
    assertEquals(
      outputCountOf(contract),
      expected.length,
      `${entry.op}: 契約が宣言する出力数`,
    );
  }
});

// 低精度格納の適格判定（ADR 0018）は TS 側 WEIGHT_SLOTS と Python 側の鏡像で決まり、
// 割れると「エクスポータが f16 で書いた重みをランタイムが適格外と見なして CPU 展開する」
// （= VRAM 削減が黙って消える）か、その逆の「f32 golden と対応しない丸め」が起きる。
// どちらも例外にならないので、表の側から両実装を突き合わせて固定する。
Deno.test("適合表の weight_slot が WEIGHT_SLOTS と一致する", async () => {
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
Deno.test("適合表の channel_axis が WEIGHT_CHANNEL_AXES と一致する", async () => {
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

Deno.test("適合表の attrs 値域を契約表の検査関数がそのとおりに判定する", async () => {
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

Deno.test("適合表の出力 shape 規則を computeOutputShape がそのとおりに計算する", async () => {
  const fixture = await loadFixture();
  for (const testCase of fixture.shapes) {
    const contract = resolveOpContract(testCase.op);
    const ins = testCase.ins.map((shape) => resolveDims(shape, testCase.bindings));
    const context = {
      declared: testCase.declared === undefined
        ? undefined
        : resolveDims(testCase.declared, testCase.bindings),
      attrs: testCase.attrs,
      bindings: testCase.bindings,
      states: testCase.states,
      stateShapes: testCase.stateShapes === undefined ? undefined : new Map(
        Object.entries(testCase.stateShapes).map((
          [slot, shape],
        ) => [slot, resolveDims(shape, testCase.bindings)]),
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

Deno.test("適合表の shape 節が全 op を受理ケースで踏む", async () => {
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

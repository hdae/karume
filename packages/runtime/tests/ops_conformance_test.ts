// Conformance-table tests of the op contracts (TS side).
//
// The source of truth is not the implementation but tests/fixtures/op-contracts.json, and the
// Python side (the exporter's tools/exporter/tests/test_ops_conformance.py) checks the same things
// against **the same table**. The table is loaded rather than copied because the moment a copy
// appears, "a contract that passes on one side only" is born (the same discipline as
// tests/fixtures/dim-grammar.json).
//
// What this file looks at is only "the contract surfaces that can silently split between the two
// implementations": the full set of op names / arity / slot dtypes / attrs key sets / attrs value
// ranges / output count and the per-output-slot dtype mapping (ADR 0068 decision 1) / output shape
// rules (including the rank limit of the strided copy family) / the slots eligible for
// low-precision storage (ADR 0018).
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
  groupNormAttrs,
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
  /** Whether the input count is variable (when true, arity is the lower bound — cat only). */
  readonly variadic: boolean;
  /**
   * The upper bound for an op with trailing optional inputs (only attention's mask; undefined
   * otherwise).
   */
  readonly maxArity: number | undefined;
  readonly dtypes: DtypeSpec;
  /**
   * The list of **per-output-slot** mappings "slot 0 input dtype → that output's dtype" (ADR 0068
   * decision 1). Absent from the table for an identity single-output op = undefined (= an identity
   * list of length 1).
   */
  readonly outDtypes: readonly Readonly<Record<string, string>>[] | undefined;
  readonly attrs: readonly string[];
  /** Optional attrs (`window` of ADR 0067; empty for ops without them). */
  readonly optionalAttrs: readonly string[];
  /**
   * The contract of the states field (ADR 0067 decisions 4 / 5; undefined for ops that cannot have
   * the field).
   */
  readonly states: { readonly keys: readonly string[]; readonly required: boolean } | undefined;
  /**
   * The weight slot eligible for low-precision storage (ADR 0018; undefined for ops without one).
   */
  readonly weightSlot: number | undefined;
  /**
   * The weight axis the i8 per-channel scale applies to (ADR 0019; undefined for ops without one).
   */
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
  /** The node's states field (key fixed by the contract → slot name; ADR 0067 decision 4). */
  readonly states?: Readonly<Record<string, string>>;
  /** The concrete shapes, capacity included, of the referenced slots (slot name → shape). */
  readonly stateShapes?: Readonly<Record<string, readonly IrDim[]>>;
  /**
   * The list of **per-output-slot** shapes (ADR 0068 decision 1 — an empty list only for
   * `state_append`).
   */
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
  if (!Object.hasOwn(raw, key)) throw new Error(`fixture: missing key '${key}'`);
  return raw[key];
};

const array = (value: unknown, where: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`fixture: ${where} is not an array`);
  return value;
};

const strings = (value: unknown, where: string): string[] =>
  array(value, where).map((item) => {
    if (typeof item !== "string") throw new Error(`fixture: ${where} must be a list of strings`);
    return item;
  });

const dims = (value: unknown, where: string): IrDim[] =>
  array(value, where).map((dim) => {
    if (typeof dim === "number" || typeof dim === "string") return dim;
    throw new Error(
      `fixture: a dimension of ${where} is neither a number nor a dimension expression`,
    );
  });

const asDtypeSpec = (value: unknown): DtypeSpec => {
  if (!isRecord(value)) throw new Error("fixture: dtypes is not an object");
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
  throw new Error(`fixture: dtypes.kind is not uniform / perSlot: ${JSON.stringify(kind)}`);
};

const asOpEntry = (raw: unknown): OpEntry => {
  if (!isRecord(raw)) throw new Error("fixture: an element of ops is not an object");
  const arity = field(raw, "arity");
  if (typeof arity !== "number") throw new Error("fixture: arity is not a number");
  const outDtypes = raw["out_dtypes"];
  if (outDtypes !== undefined && !Array.isArray(outDtypes)) {
    throw new Error("fixture: out_dtypes is not a per-output-slot list");
  }
  const variadic = raw["variadic"];
  if (variadic !== undefined && variadic !== true) {
    throw new Error("fixture: variadic must be true or omitted");
  }
  const maxArity = raw["max_arity"];
  if (maxArity !== undefined && typeof maxArity !== "number") {
    throw new Error("fixture: max_arity is not a number");
  }
  // "Any number" and "one specific extra input" are different contract surfaces (listing both is a
  // mistake in how the table is built)
  if (maxArity !== undefined && variadic === true) {
    throw new Error("fixture: variadic and max_arity are not listed together");
  }
  const weightSlot = raw["weight_slot"];
  if (weightSlot !== undefined && typeof weightSlot !== "number") {
    throw new Error("fixture: weight_slot is not a number");
  }
  const channelAxis = raw["channel_axis"];
  if (channelAxis !== undefined && typeof channelAxis !== "number") {
    throw new Error("fixture: channel_axis is not a number");
  }
  const states = raw["states"];
  if (states !== undefined && !isRecord(states)) throw new Error("fixture: states is not a table");
  const required = states === undefined ? undefined : states["required"];
  if (states !== undefined && typeof required !== "boolean") {
    throw new Error("fixture: states.required is not a boolean");
  }
  return {
    op: String(field(raw, "op")),
    arity,
    variadic: variadic === true,
    maxArity,
    dtypes: asDtypeSpec(field(raw, "dtypes")),
    outDtypes: outDtypes?.map((slot, index) => {
      if (!isRecord(slot)) throw new Error(`fixture: out_dtypes[${index}] is not a table`);
      return Object.fromEntries(
        Object.entries(slot).map(([from, to]) => {
          if (typeof to !== "string") {
            throw new Error(`fixture: out_dtypes[${index}][${from}] is not a string`);
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
  if (!isRecord(raw)) throw new Error("fixture: an element of attr_values is not an object");
  return {
    op: String(field(raw, "op")),
    attr: String(field(raw, "attr")),
    accept: array(field(raw, "accept"), "accept"),
    reject: array(field(raw, "reject"), "reject"),
  };
};

const asShapeCase = (raw: unknown): ShapeCase => {
  if (!isRecord(raw)) throw new Error("fixture: an element of shapes is not an object");
  const op = String(field(raw, "op"));
  const attrs = raw["attrs"];
  const declared = raw["declared"];
  const outs = raw["outs"];
  const bindings: Record<string, number> = {};
  if (raw["bindings"] !== undefined) {
    if (!isRecord(raw["bindings"]))
      throw new Error(`fixture: the bindings of ${op} are not a table`);
    for (const [sym, bound] of Object.entries(raw["bindings"])) {
      if (typeof bound !== "number") throw new Error(`fixture: binding ${sym} is not a number`);
      bindings[sym] = bound;
    }
  }
  const throws = raw["throws"] === true;
  if (throws === (outs !== undefined)) {
    throw new Error(
      `fixture: a shapes case has exactly one of outs or throws ${JSON.stringify(raw)}`,
    );
  }
  if (attrs !== undefined && !isRecord(attrs)) {
    throw new Error(`fixture: the attrs of ${op} are not an object`);
  }
  const states = raw["states"];
  if (states !== undefined && !isRecord(states)) {
    throw new Error(`fixture: the states of ${op} are not a table`);
  }
  const stateShapes = raw["state_shapes"];
  if (stateShapes !== undefined && !isRecord(stateShapes)) {
    throw new Error(`fixture: the state_shapes of ${op} are not a table`);
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
                throw new Error(`fixture: ${op}.states[${key}] is not a string`);
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
  if (!isRecord(raw)) throw new Error("fixture: not an object");
  const max = field(raw, "strided_rank_max");
  if (typeof max !== "number") throw new Error("fixture: strided_rank_max is not a number");
  return {
    stridedRankMax: max,
    ops: array(field(raw, "ops"), "ops").map(asOpEntry),
    attrValues: array(field(raw, "attr_values"), "attr_values").map(asAttrEntry),
    shapes: array(field(raw, "shapes"), "shapes").map(asShapeCase),
  };
};

/**
 * Resolves the conformance table's dimension language to numbers through the bindings (TS shape
 * computation handles only bound numbers).
 */
const resolveDims = (
  shape: readonly IrDim[],
  bindings: Readonly<Record<string, number>>,
): number[] =>
  shape.map((dim) => (typeof dim === "number" ? dim : evalDim(parseDim(dim), bindings)));

const label = (testCase: ShapeCase): string =>
  `${testCase.op}(${testCase.ins.map((shape) => `[${shape.join(",")}]`).join(", ")})`;

test("the op set of the conformance table matches the contract table exactly", async () => {
  const fixture = await loadFixture();
  assertEquals(fixture.ops.map((entry) => entry.op).sort(), [...OP_CONTRACTS.keys()].sort());
  // No op is duplicated in the table (a duplicate would make one of the declarations silently
  // ignored)
  assertEquals(new Set(fixture.ops.map((entry) => entry.op)).size, fixture.ops.length);
});

test("the strided rank limit of the conformance table matches the implementation's constant", async () => {
  const fixture = await loadFixture();
  assertEquals(fixture.stridedRankMax, STRIDED_RANK);
});

test("the arity / slot dtypes / attrs keys of the conformance table match the contract table", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    const contract = resolveOpContract(entry.op);
    assertEquals(contract.arity, entry.arity, `${entry.op}: arity`);
    // Variable arity is itself a contract surface, "how many inputs are accepted" (making it
    // variable on one side only would split the count the exporter can write from the count the
    // runtime accepts). The upper bound of optional trailing inputs (attention's mask) is in the
    // table for the same reason.
    assertEquals(contract.variadic === true, entry.variadic, `${entry.op}: variable arity`);
    assertEquals(contract.maxArity, entry.maxArity, `${entry.op}: upper bound of optional inputs`);
    assertEquals(
      [...attrKeysOf(contract)].sort(),
      [...entry.attrs].sort(),
      `${entry.op}: attrs key set`,
    );
    // The optional attrs (`window` of ADR 0067) and the contract of the states field also lead
    // straight to "the runtime rejects a shape the exporter can write" if only one side moves, so
    // the table pins them.
    assertEquals(
      [...optionalAttrKeysOf(contract)].sort(),
      [...entry.optionalAttrs].sort(),
      `${entry.op}: optional attrs key set`,
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
      `${entry.op}: contract of the states field`,
    );
    const slots = contract.slotDtypes;
    assertEquals(slots.kind, entry.dtypes.kind, `${entry.op}: kind of the dtype contract`);
    if (slots.kind === "uniform" && entry.dtypes.kind === "uniform") {
      assertEquals(
        [...slots.accept].sort(),
        [...entry.dtypes.accept].sort(),
        `${entry.op}: accepted dtypes`,
      );
      continue;
    }
    if (slots.kind === "perSlot" && entry.dtypes.kind === "perSlot") {
      assertEquals(
        slots.slots.map((accept) => [...accept].sort()),
        entry.dtypes.slots.map((accept) => [...accept].sort()),
        `${entry.op}: per-slot accepted dtypes`,
      );
    }
  }
});

// Deriving the output dtype is the axis that produces "runnable according to the support table, but
// read as a different TypedArray at run time". This pins from the table side that comparisons
// (f32 → bool), the sum of bool (→ i32) and where (bool → f32) agree on both sides (cast is
// excluded because attrs.to decides it).
test("the output dtype mapping of the conformance table matches the contract table", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    if (entry.op === "cast") {
      assertEquals(
        entry.outDtypes,
        undefined,
        "cast has no out_dtypes (attrs.to is the source of truth)",
      );
      continue;
    }
    const contract = resolveOpContract(entry.op);
    const slots = contract.slotDtypes;
    const domain = slots.kind === "uniform" ? slots.accept : slots.slots[0];
    // For an omitted op the expectation is "one output, identity mapping over slot 0's accepted
    // set".
    const expected = (entry.outDtypes ?? [{}]).map((slot) =>
      Object.fromEntries([...domain].sort().map((dtype) => [dtype, slot[dtype] ?? dtype])),
    );
    assertEquals(
      contract.outputDtypes.map((slot) =>
        Object.fromEntries([...slot].sort(([a], [b]) => a.localeCompare(b))),
      ),
      expected,
      `${entry.op}: per-output-slot dtype mapping`,
    );
    // The output count is the length of the mapping list itself (if the table and the
    // implementation disagreed on the count, the dtype of the second output would be checked by no
    // one).
    assertEquals(
      outputCountOf(contract),
      expected.length,
      `${entry.op}: output count the contract declares`,
    );
  }
});

// The eligibility of low-precision storage (ADR 0018) is decided by WEIGHT_SLOTS on the TS side
// and its mirror on the Python side. If they split, either "the runtime treats a weight the
// exporter wrote as f16 as ineligible and expands it on the CPU" (= the VRAM saving silently
// disappears) or the reverse, "rounding that does not correspond to the f32 golden", happens.
// Neither raises an exception, so both implementations are matched from the table side.
test("weight_slot of the conformance table matches WEIGHT_SLOTS", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    assertEquals(WEIGHT_SLOTS.get(entry.op), entry.weightSlot, `${entry.op}: weight slot`);
  }
  // Also fail when an op exists only on the implementation side and not in the table (another test
  // checks that the op sets match, but WEIGHT_SLOTS is a subset of the op set, so the count is
  // matched once more).
  assertEquals(
    fixture.ops.filter((entry) => entry.weightSlot !== undefined).length,
    WEIGHT_SLOTS.size,
    "number of ops with a weight_slot",
  );
});

// The axis of the per-channel scale (ADR 0019). If they split, "the kernel reads a scale the
// exporter built on axis 0 as axis 1", and **only the values break, with no exception** (the
// [Cin,Cout,K] of conv_transpose1d even passes the shape check when Cin == Cout).
test("channel_axis of the conformance table matches WEIGHT_CHANNEL_AXES", async () => {
  const fixture = await loadFixture();
  for (const entry of fixture.ops) {
    assertEquals(
      WEIGHT_CHANNEL_AXES.get(entry.op),
      entry.channelAxis,
      `${entry.op}: per-channel axis`,
    );
    // The table side also requires "an op with a weight slot also has a channel axis" (if a row
    // with only one of them could be written, the table would accept it even after the pairing
    // broke on the implementation side).
    assertEquals(
      entry.channelAxis === undefined,
      entry.weightSlot === undefined,
      `${entry.op}: weight_slot and channel_axis appear together, neither without the other`,
    );
  }
  assertEquals(
    fixture.ops.filter((entry) => entry.channelAxis !== undefined).length,
    WEIGHT_CHANNEL_AXES.size,
    "number of ops with a channel_axis",
  );
});

test("the contract table's check functions judge the attrs value ranges of the conformance table as listed", async () => {
  const fixture = await loadFixture();
  // Required and optional attrs are judged on the same footing (there is one source of truth for
  // the value range — being optional is no reason for the range to be looser).
  const declared = new Map(
    fixture.ops.map((entry) => [entry.op, [...entry.attrs, ...entry.optionalAttrs]]),
  );
  for (const entry of fixture.attrValues) {
    const contract = resolveOpContract(entry.op);
    assert(
      declared.get(entry.op)?.includes(entry.attr) === true,
      `${entry.op}: attr '${entry.attr}' is not in the ops section`,
    );
    const check = contract.attrs[entry.attr] ?? contract.optionalAttrs?.[entry.attr];
    assert(
      check !== undefined,
      `${entry.op}: the contract table has no check function for attr '${entry.attr}'`,
    );
    // MUST: an entry with an empty accept / reject passes with zero checks (a tautology) — reject
    // it on the table side (the counterpart of the same check on the Python side,
    // tests/test_ops_conformance.py).
    assert(
      entry.accept.length > 0 && entry.reject.length > 0,
      `${entry.op}.${entry.attr}: accept or reject is empty (it would pass with zero checks)`,
    );
    for (const value of entry.accept) {
      check(value, `${entry.op}.${entry.attr}`);
    }
    for (const value of entry.reject) {
      assertThrows(
        () => check(value, `${entry.op}.${entry.attr}`),
        OpContractError,
        undefined,
        `${entry.op}.${entry.attr} = ${JSON.stringify(value)} must not be accepted`,
      );
    }
  }
  // Every attr with a value range is in the table (an attr left out would keep its range unchecked)
  const covered = new Set(fixture.attrValues.map((entry) => `${entry.op}.${entry.attr}`));
  for (const entry of fixture.ops) {
    for (const attr of [...entry.attrs, ...entry.optionalAttrs]) {
      assert(
        covered.has(`${entry.op}.${attr}`),
        `the value range of ${entry.op}.${attr} is not in the table`,
      );
    }
  }
});

test("computeOutputShape computes the output shape rules of the conformance table as listed", async () => {
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
        `${label(testCase)} must not be accepted (${testCase.why})`,
      );
      continue;
    }
    const computed = computeOutputShape(contract, ins, "t", context);
    assertEquals(
      computed,
      (testCase.outs ?? []).map((shape) => resolveDims(shape, testCase.bindings)),
      label(testCase),
    );
    // The length of the shape list matches the contract's declared output count (if the table's
    // outs and the out_dtypes list of the ops section grew separately, a contract with two shapes
    // but one dtype could be written).
    assertEquals(computed.length, outputCountOf(contract), `${label(testCase)}: output count`);
  }
});

test("the shapes section of the conformance table exercises every op with an accepted case", async () => {
  const fixture = await loadFixture();
  const accepted = new Set(
    fixture.shapes.filter((testCase) => !testCase.throws).map((testCase) => testCase.op),
  );
  assertEquals(
    [...accepted].sort(),
    fixture.ops.map((entry) => entry.op).sort(),
    "some op's shape rule is not exercised by the table",
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

// MUST: the doc is not read (extraction that depends on formatting is fragile and breeds
// tautologies). This only pins the op set with **an expected literal**; keeping it in sync with
// the list in docs/ir-v2.md is manual work when adding an op (one contract set — ops.ts / ops.py /
// shapes.py / fixtures / CPU reference / golden / ir-v2.md).
test("the contract table holds exactly the 64 ops of the expected literal", () => {
  // The op names themselves are pinned by the ops section of fixtures/op-contracts.json; the
  // family lists below are not in the table.
  assertEquals(UNARY_OPS.length, 19);
  assertEquals(BINARY_OPS.length, 6);
  // argmax is **not** in the reduce family (attrs, output dtype and rank handling all differ — ADR
  // 0068 decision 2)
  assertEquals(REDUCE_OPS.length, 3);
  assertEquals(OP_CONTRACTS.size, 64);
});

test("the runtime support table and the capability query are derived from the contract table", () => {
  assertEquals(RUNTIME_SUPPORT.ops.size, OP_CONTRACTS.size);
  // Raw int32 storage is executable as the target for baking symbol-dependent constants (ADR
  // 0010). f16 (ADR 0018), i8 (ADR 0019) and i4 (ADR 0069 — eligible only for linear / embedding)
  // have execution paths (eligible weights stay resident compressed; ineligible ones are expanded
  // on the CPU). bf16 (ADR 0131) runs on the same path as f16.
  assertEquals([...RUNTIME_SUPPORT.storage], ["f32", "f16", "bf16", "i8", "i4", "i2", "i32"]);
  // The transfer-layer axis (applies to unused graph inputs too) — i32 / bool can be transferred as
  // well (ADR 0009)
  assertEquals([...RUNTIME_SUPPORT.io].sort(), ["bool", "f32", "i32"]);
  assertEquals([...IO_DTYPES].sort(), ["bool", "f32", "i32"]);
  assertEquals(capabilities().ops, [...OP_CONTRACTS.keys()].sort());
  assertEquals(capabilities().storage, ["bf16", "f16", "f32", "i2", "i32", "i4", "i8"]);
  // The contents of the projection are pinned with **expected literals**. Walking the contract
  // table again and matching with the same formula would be a check that never fails unless the
  // structure breaks, since the source is the same object.
  const support = (op: string): OpSupport => {
    const found = RUNTIME_SUPPORT.ops.get(op);
    assert(found !== undefined, `the support table has no op '${op}'`);
    return found;
  };
  const slots = (op: string): string[][] =>
    support(op).slotDtypes.map((accept) => [...accept].sort());

  // The three per-slot contracts (the accepted set differs for the values side and the indices /
  // condition)
  assertEquals(slots("gather"), [["f32"], ["i32"]]);
  assertEquals(slots("embedding"), [["f32"], ["i32"]]);
  assertEquals(slots("masked_fill"), [["f32"], ["bool"]]);
  // A uniform contract lists the same set once per arity (so consumers can look it up by slot
  // number)
  assertEquals(slots("mul"), [
    ["f32", "i32"],
    ["f32", "i32"],
  ]);
  assertEquals(slots("relu"), [["f32"]]);
  assertEquals(slots("conv1d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv1d").attrKeys].sort(), ["dilation", "groups", "padding", "stride"]);
  assertEquals(slots("conv2d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv2d").attrKeys].sort(), ["dilation", "groups", "padding", "stride"]);
  // rms_norm has arity 2 (no bias) — the projection's slot count is itself a contract surface
  assertEquals(slots("rms_norm"), [["f32"], ["f32"]]);
  assertEquals([...support("rms_norm").attrKeys], ["eps"]);
  assertEquals([...support("clamp_min").attrKeys], ["min"]);
  assertEquals(slots("conv_transpose1d"), [["f32"], ["f32"], ["f32"]]);
  assertEquals([...support("conv_transpose1d").attrKeys].sort(), ["padding", "stride"]);
  assertEquals([...support("cast").attrKeys], ["to"]);
  assertEquals([...support("relu").attrKeys], []);

  // The relations between projections (union = merge of the per-slot sets / slot count = arity) are
  // true invariants that can be checked without walking the contract table again, so they run for
  // every op.
  for (const [name, contract] of OP_CONTRACTS) {
    const found = support(name);
    // The slot count is "the maximum number of inputs that can be accepted" = it includes the
    // optional trailing input (attention's mask) (only the variable-arity cat counts its lower
    // bound — every slot has the same accepted set, so the conclusion matches the union view of the
    // enumerating check).
    assertEquals(
      found.slotDtypes.length,
      contract.maxArity ?? contract.arity,
      `${name}: slot count = maximum acceptable arity`,
    );
    const union = new Set(found.slotDtypes.flatMap((accept) => [...accept]));
    assertEquals(
      [...union].sort(),
      [...found.dtypes].sort(),
      `${name}: union = merge of the slots`,
    );
  }
});

// MUST: no blanket unlocking (the vocabulary allowlist freeze of ADR 0007). Only the shapes that
// appear on the mask path of measured graphs (mul / sub) and the ops added for that path (cast /
// bitwise_not) have i32 / bool.
test("dtypes are unlocked per op, and combinations that do not appear in measurements stay f32", () => {
  const dtypes = (op: string): readonly string[] => [...resolveOpContract(op).dtypes].sort();
  assertEquals(dtypes("mul"), ["f32", "i32"]);
  assertEquals(dtypes("sub"), ["f32", "i32"]);
  assertEquals(dtypes("bitwise_not"), ["bool"]);
  assertEquals(dtypes("cast"), ["bool", "f32", "i32"]);
  // The layout ops (ADR 0011) also follow the measurements: reshape does not touch elements so it
  // takes the whole vocabulary, expand only gather indices (i32) and conv's bool mask, permute only
  // f32 attention reshaping.
  assertEquals(dtypes("reshape"), ["bool", "f32", "i32"]);
  // f32 was added in wave 3 (making the relative-position embedding 4D — recon §2). The strided
  // copy family is dtype-parametric, so the kernel stays shared.
  assertEquals(dtypes("expand"), ["bool", "f32", "i32"]);
  assertEquals(dtypes("permute"), ["f32"]);
  // The math ops of wave 3: the four comparisons and where read f32, bitwise_and is bool only,
  // and only sum has two forms, f32 and bool (the output is f32 / i32).
  assertEquals(dtypes("ge"), ["f32"]);
  assertEquals(dtypes("ge_scalar"), ["f32"]);
  assertEquals(dtypes("bitwise_and"), ["bool"]);
  assertEquals(dtypes("sum"), ["bool", "f32"]);
  assertEquals(dtypes("where"), ["bool", "f32"]);
  // The fused ops (ADR 0012) are f32-only for values; only the indices (embedding) and the
  // condition (masked_fill) take integer / bool per slot.
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

// ADR 0012: gather is the first op whose accepted set differs per input slot (src=f32 / index=i32).
// MUST: do not confuse the check that looks only at "the union of the accepted sets" (assertDtype)
// with the per-slot check.
test("gather has a per-slot dtype contract and judges more narrowly than matching against the union", () => {
  const gather = resolveOpContract("gather");
  assertEquals(gather.slotDtypes.kind, "perSlot");
  // The union appears in the capability projection (the support table carries "the dtypes this op
  // touches")
  assertEquals([...gather.dtypes], ["f32", "i32"]);
  assertEquals([...(RUNTIME_SUPPORT.ops.get("gather")?.dtypes ?? [])], ["f32", "i32"]);
  // Per slot, each accepts only one type
  assertSlotDtype(gather, 0, "f32", "t");
  assertSlotDtype(gather, 1, "i32", "t");
  assertThrows(() => assertSlotDtype(gather, 0, "i32", "t"), OpContractError);
  assertThrows(() => assertSlotDtype(gather, 1, "f32", "t"), OpContractError);
  assertThrows(() => assertSlotDtype(gather, 1, "bool", "t"), OpContractError);
  // A slot that does not exist (assertNodeContract fails an arity violation first, but the contract
  // itself also rejects it)
  assertThrows(() => assertSlotDtype(gather, 2, "f32", "t"), OpContractError);

  // For a uniform contract the per-slot check is the same as matching against the union (existing
  // ops are judged the same)
  const mul = resolveOpContract("mul");
  assertEquals(mul.slotDtypes.kind, "uniform");
  assertSlotDtype(mul, 1, "i32", "t");
  assertThrows(() => assertSlotDtype(mul, 1, "bool", "t"), OpContractError);
});

test("the output of a per-slot contract has the same type as slot 0, and only uniform ops reject mixing", () => {
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

  // For gather a mix of f32 and i32 is the correct form (the uniform "reject mixing" does not
  // apply)
  assertEquals(resolve("gather", ["f32", "i32"], "f32"), "f32");
  // Mixed-up slots (src and index passed the wrong way round) fail
  assertThrows(() => resolve("gather", ["i32", "f32"], "f32"), OpContractError);
  // An output declaration that differs from slot 0 fails
  assertThrows(() => resolve("gather", ["f32", "i32"], "i32"), OpContractError);
  // The uniform side keeps rejecting mixes (not collapsed into perSlot)
  assertThrows(() => resolve("mul", ["i32", "f32"], "i32"), OpContractError);
  assertEquals(resolve("bmm", ["f32", "f32"], "f32"), "f32");
});

// 31 ops have attrs. All others keep empty attrs, and non-empty attrs fail loudly.
test("only the 32 ops the contract table lists have attrs; the others have empty attrs", () => {
  // The set of ops that carry attrs is pinned per op by the attrs column of the table; this test
  // keeps the node-level rejections.
  assertThrows(() => assertNodeContract(node("relu", ["a"], { alpha: 1 }), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("relu", ["a", "b"]), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("add", ["a"]), "t"), OpContractError);
  assertThrows(
    () => assertNodeContract({ ...node("relu", ["a"]), outs: [] }, "t"),
    OpContractError,
  );
  // The arity of fused ops is fixed (forms without bias / affine are not in the vocabulary — ADR
  // 0012)
  assertThrows(() => assertNodeContract(node("linear", ["x", "w"]), "t"), OpContractError);
  assertThrows(
    () =>
      assertNodeContract(
        node("conv1d", ["x", "w"], { stride: 1, padding: 1, dilation: 1, groups: 1 }),
        "t",
      ),
    OpContractError,
  );
  // A conv without bias is normalised by the exporter synthesizing a zero bias (ADR 0015) — the
  // contract side keeps the arity fixed at 3
  assertThrows(
    () => assertNodeContract(node("conv_transpose1d", ["x", "w"], { stride: 2, padding: 0 }), "t"),
    OpContractError,
  );
  // rms_norm has arity fixed at 2 (a missing weight is normalised by the exporter synthesizing ones
  // — ADR 0017)
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
  // NOTE: conv2d entered the contract table with ADR 0017 (until M1-P2 it was the representative of
  // "an op in the saved list but without a kernel"). conv_transpose2d takes over as the
  // representative of an op not even in the vocabulary.
  assertThrows(() => resolveOpContract("conv_transpose2d"), OpContractError);
});

// deform_conv2d is **DCNv2 only, with stride/dilation/groups/offset_groups fixed at 1** (layer 1',
// ADR 0055). Both are expressed not by value checks but by **the absence of the fields and
// slots**, so this pins from the contract side that they have not appeared.
test("deform_conv2d has a single attr, padding, and arity fixed at 5", () => {
  assertEquals(
    assertNodeContract(
      node("deform_conv2d", ["x", "w", "off", "mask", "b"], { padding: [1, 0] }),
      "t",
    ).kind,
    "deformConv2d",
  );
  // There are no stride / dilation / groups / offset_groups fields = they fail as undeclared keys
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
  // padding must be declared (no default filled in)
  assertThrows(
    () => assertNodeContract(node("deform_conv2d", ["x", "w", "off", "mask", "b"]), "t"),
    OpContractError,
  );
  // Neither DCNv1 without the mask nor an extra slot is accepted (the arity is itself the
  // restriction)
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

// upsample_bilinear2d is **align_corners=True only** (layer 1). "Accepts nothing but True" is
// expressed not by a value check but by **the absence of the field**, so this pins from the
// contract side that the field has not appeared (the discipline of ADR 0023 decision 4 — the moment
// the field exists, False becomes expressible too).
test("upsample_bilinear2d has a single attr, output_size (no align_corners field)", () => {
  assertEquals(
    assertNodeContract(node("upsample_bilinear2d", ["x"], { output_size: [7, 9] }), "t").kind,
    "upsampleBilinear2d",
  );
  // There are no align_corners / mode / scale_factor fields = they fail as undeclared keys
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
  // Declaring the output space is required (no default filled in)
  assertThrows(() => assertNodeContract(node("upsample_bilinear2d", ["x"]), "t"), OpContractError);
  // Arity fixed at 1 (neither weights nor bias)
  assertThrows(
    () => assertNodeContract(node("upsample_bilinear2d", ["x", "w"], { output_size: [7, 9] }), "t"),
    OpContractError,
  );
});

// ADR 0012: the attrs schema is "every declared key is required, undeclared keys fail loudly, and
// values are checked too".
test("the attrs schema rejects a missing required key and a value outside the contract (cast's to)", () => {
  assertEquals(assertNodeContract(node("cast", ["a"], { to: "i32" }), "t").kind, "cast");
  // A missing required key (M0's "look only at unknown keys" check let this through)
  assertThrows(() => assertNodeContract(node("cast", ["a"]), "t"), OpContractError);
  // An undeclared key
  assertThrows(
    () => assertNodeContract(node("cast", ["a"], { to: "f32", from: "i32" }), "t"),
    OpContractError,
  );
  // MUST: schema matching uses Object.hasOwn only — names from the prototype do not become known
  // keys
  assertThrows(
    () => assertNodeContract(node("cast", ["a"], { to: "f32", toString: "x" }), "t"),
    OpContractError,
  );
  assertEquals(castTargetDtype({ to: "bool" }, "t"), "bool");
  assertThrows(() => castTargetDtype({}, "t"), OpContractError);
});

// ADR 0011: permute's dims is a non-negative bijection (negative axis notation is normalised at the
// exporter boundary).
test("permute's dims accepts only a permutation table that is non-negative and has no duplicates", () => {
  assertEquals(
    assertNodeContract(node("permute", ["a"], { dims: [0, 2, 1] }), "t").kind,
    "permute",
  );
  assertEquals(permuteDims({ dims: [1, 0] }, "t"), [1, 0]);
  assertThrows(() => permuteDims({}, "t"), OpContractError);
  // reshape / expand take no attrs (the target shape is the output's declared shape)
  assertThrows(
    () => assertNodeContract(node("reshape", ["a"], { shape: [2, 3] }), "t"),
    OpContractError,
  );
  assertThrows(
    () => assertNodeContract(node("expand", ["a"], { shape: [2, 3] }), "t"),
    OpContractError,
  );
});

// MUST: the output dtype is derived from the contract instead of trusting the declaration (a graph
// whose cast to and values{} declaration disagree gives a silent wrong value, read as a different
// TypedArray on readback).
test("the output dtype is derived from the contract and a disagreement with the declaration is rejected", () => {
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
  // The declaration disagrees with the contract
  assertThrows(() => resolve("mul", ["a", "b"], ["i32", "i32"], "f32"), OpContractError);
  assertThrows(() => resolve("cast", ["a"], ["f32"], "f32", { to: "i32" }), OpContractError);
  // Mixed-type elementwise is not in the vocabulary
  assertThrows(() => resolve("mul", ["a", "b"], ["i32", "f32"], "i32"), OpContractError);
  // An input dtype outside the contract
  assertThrows(() => resolve("add", ["a", "b"], ["i32", "i32"], "i32"), OpContractError);

  // Wave 3: three groups whose dtype differs between input and output (derived from the contract
  // table's mapping)
  assertEquals(resolve("ge", ["a", "b"], ["f32", "f32"], "bool"), "bool");
  assertEquals(resolve("ge_scalar", ["a"], ["f32"], "bool", { value: 0 }), "bool");
  assertEquals(resolve("sum", ["a"], ["f32"], "f32"), "f32");
  // The sum of bool is a **count**, so it is i32 (counting in f32 quietly rounds at 2^24)
  assertEquals(resolve("sum", ["a"], ["bool"], "i32"), "i32");
  assertThrows(() => resolve("sum", ["a"], ["bool"], "f32"), OpContractError);
  assertThrows(() => resolve("sum", ["a"], ["bool"], "bool"), OpContractError);
  assertThrows(() => resolve("ge", ["a", "b"], ["f32", "f32"], "f32"), OpContractError);
  // For where the condition is the first slot, but the output is **the values side** (mapping bool
  // → f32)
  assertEquals(resolve("where", ["c", "a", "b"], ["bool", "f32", "f32"], "f32"), "f32");
  assertThrows(
    () => resolve("where", ["c", "a", "b"], ["bool", "f32", "f32"], "bool"),
    OpContractError,
  );
  // Mixed-up slots (condition and values passed the wrong way round)
  assertThrows(
    () => resolve("where", ["c", "a", "b"], ["f32", "f32", "bool"], "f32"),
    OpContractError,
  );
  assertEquals(resolve("bitwise_and", ["a", "b"], ["bool", "bool"], "bool"), "bool");
  assertThrows(() => resolve("bitwise_and", ["a", "b"], ["f32", "f32"], "f32"), OpContractError);
});

// MUST: the domain of the mapping matches slot 0's accepted set exactly (a partial mapping is a
// hole of "a dtype whose output is undefined"). That identity is the default is guaranteed by the
// same invariant.
test("the domain of the output dtype mapping matches slot 0's accepted set", () => {
  for (const [name, contract] of OP_CONTRACTS) {
    const slots = contract.slotDtypes;
    const domain = slots.kind === "uniform" ? slots.accept : slots.slots[0];
    // The domain match is required **per output slot** (fail even if only one list in it has a
    // hole)
    contract.outputDtypes.forEach((slot, index) => {
      assertEquals(
        [...slot.keys()].sort(),
        [...domain].sort(),
        `${name}: domain of the dtype mapping of output ${index}`,
      );
    });
  }
});

// The math ops of wave 3. Pins the attrs value ranges and an invariant across keys (clamp's min <=
// max).
test("the attrs schemas of clamp / leaky_relu / comparisons check down to the value range", () => {
  assertEquals(assertNodeContract(node("clamp", ["a"], { min: -1, max: 1 }), "t").kind, "unary");
  // A missing required key (no default filled in — ADR 0015 / 0012)
  assertThrows(() => assertNodeContract(node("clamp", ["a"], { min: -1 }), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("leaky_relu", ["a"], {}), "t"), OpContractError);
  assertThrows(() => assertNodeContract(node("ge_scalar", ["a"], {}), "t"), OpContractError);

  // MUST: min <= max **spans keys**, so the attrs schema cannot express it. This pins that it fails
  // in shape computation (the only common path every node goes through) and that the params values
  // pass the same check on the way there.
  const clamp = resolveOpContract("clamp");
  assertEquals(scalarParamValues(clamp, { min: -1, max: 2 }, "t"), [-1, 2]);
  assertEquals(scalarParamValues(clamp, { min: 2, max: 2 }, "t"), [2, 2]);
  assertThrows(() => scalarParamValues(clamp, { min: 2, max: 1 }, "t"), OpContractError);
  assertThrows(
    () => computeOutputShape(clamp, [[2, 3]], "t", { attrs: { min: 2, max: 1 } }),
    OpContractError,
  );
  // An op without scalars is empty (nothing goes at the end of params)
  assertEquals(scalarParamValues(resolveOpContract("relu"), {}, "t"), []);
  // The three comparisons and leaky_relu have one each
  assertEquals(scalarParamValues(resolveOpContract("gt_scalar"), { value: 1.5 }, "t"), [1.5]);
  assertEquals(
    scalarParamValues(resolveOpContract("leaky_relu"), { negative_slope: 0.1 }, "t"),
    [0.1],
  );
});

// The symmetric counterpart on the exporter side
// (`the_params_table_and_the_attrs_schema_cover_the_same_unary_ops` in
// `tools/exporter/tests/test_shapes.py`).
test("the scalar params order table and the unary attrs schema cover the same ops", () => {
  // The attrs schema (the contract table's unary ops) and the params layout (SCALAR_PARAM_ATTRS)
  // are two tables written separately, and adding an op to only one of them **raises no
  // exception**: on the side missing from the params table, `scalarParamValues` returns an empty
  // list and the attr's value silently never reaches the kernel; on the side missing from the attrs
  // schema, the declaration itself is treated as attrs outside the contract.
  const unaryWithAttrs = [...OP_CONTRACTS]
    .filter(([, contract]) => contract.kind === "unary" && attrKeysOf(contract).length > 0)
    .map(([name]) => name)
    .sort();
  assertEquals([...SCALAR_PARAM_ATTRS.keys()].sort(), unaryWithAttrs);
});

// cumsum is restricted the same way as softmax (it has attrs.dim and accepts only the last
// dimension).
test("cumsum is fixed to the last dimension and passes the shape through", () => {
  const contract = resolveOpContract("cumsum");
  assertEquals(cumsumDim({ dim: 2 }, "t"), 2);
  assertThrows(() => cumsumDim({}, "t"), OpContractError);
  assertThrows(() => cumsumDim({ dim: -1 }, "t"), OpContractError);
  const shape = (ins: readonly number[], dim: number) =>
    computeOutputShape(contract, [ins], "t", { attrs: { dim } })[0];
  assertEquals(shape([1, 3, 4], 2), [1, 3, 4]);
  // A zero-length axis passes through (the identity of a prefix sum is 0 — unlike amax / softmax,
  // it is defined)
  assertEquals(shape([3, 0], 1), [3, 0]);
});

// argmax is **a separate contract** from the reduce family (ADR 0068 decision 2). The restriction
// is expressed not by "value checks" but by **the absence of fields**, so this pins from the
// contract side that no field has appeared.
test("argmax has empty attrs, is fixed to the last dimension, keeps the rank, and outputs i32", () => {
  const contract = resolveOpContract("argmax");
  assertEquals(contract.kind, "argmax");
  assertEquals(resolveNodeDtypes(contract, node("argmax", ["x"]), ["f32"], ["i32"], "t"), ["i32"]);
  // A graph that declares f32 is rejected (readback would read it as a different TypedArray, a
  // silent wrong value)
  assertThrows(
    () => resolveNodeDtypes(contract, node("argmax", ["x"]), ["f32"], ["f32"], "t"),
    OpContractError,
  );
  // i32 / bool inputs are not in the vocabulary (f32 only — the measured case is only logits)
  assertThrows(() => assertDtype(contract, "i32", "t"), OpContractError);
  const shape = (ins: readonly number[]) => computeOutputShape(contract, [ins], "t")[0];
  assertEquals(shape([2, 3, 4]), [2, 3, 1]);
  // attrs outside the contract fail loudly (empty-attrs contract — ADR 0012)
  assertThrows(() => assertNodeContract(node("argmax", ["x"], { dim: 1 }), "t"), OpContractError);
});

// topk is **the only multi-output op** (ADR 0068 decision 3). Pins that the accepted region (1 ≤ k
// ≤ last dimension) and the dtype / shape of both outputs come from one contract, and that no
// `largest` / `sorted` field has appeared.
test("topk requires attrs k, has two outputs (values f32 + indices i32), and accepts 1 ≤ k ≤ last dimension", () => {
  const contract = resolveOpContract("topk");
  assertEquals(contract.kind, "topk");
  assertEquals(
    resolveNodeDtypes(contract, node("topk", ["x"], { k: 2 }), ["f32"], ["f32", "i32"], "t"),
    ["f32", "i32"],
  );
  // A declaration with the slots swapped fails (both have 4-byte elements and the same shape, so
  // letting it through would make readback silently succeed with values and indices swapped)
  assertThrows(
    () => resolveNodeDtypes(contract, node("topk", ["x"], { k: 2 }), ["f32"], ["i32", "f32"], "t"),
    OpContractError,
  );
  // A declaration with a single output also fails (the contract owns the count)
  assertThrows(
    () => resolveNodeDtypes(contract, node("topk", ["x"], { k: 2 }), ["f32"], ["f32"], "t"),
    OpContractError,
  );
  assertThrows(
    () => assertNodeContract({ ...node("topk", ["x"], { k: 2 }), outs: ["v"] }, "t"),
    OpContractError,
    "output count is 1 (contract: 2)",
  );
  // A node with two outputs (the `node` helper builds single-output nodes, so topk is rebuilt here)
  const topkNode = (attrs: Record<string, unknown>): IrNode => ({
    ...node("topk", ["x"], attrs),
    outs: ["v", "i"],
  });
  assertEquals(assertNodeContract(topkNode({ k: 2 }), "t").kind, "topk");
  // i32 / bool inputs are not in the vocabulary (f32 only — the measured case is only logits)
  assertThrows(() => assertDtype(contract, "i32", "t"), OpContractError);
  const shapes = (ins: readonly number[], k: unknown) =>
    computeOutputShape(contract, [ins], "t", { attrs: { k } });
  // Both are [..., k] (rank kept, leading dimensions passed through)
  assertEquals(shapes([2, 3, 5], 1), [
    [2, 3, 1],
    [2, 3, 1],
  ]);
  // Everything outside the accepted region fails loudly (no degrading)
  assertThrows(() => shapes([6, 10], 11), OpContractError, "last dimension");
  assertThrows(() => shapes([6, 10], -1), OpContractError);
  assertThrows(() => shapes([6, 10], 1.5), OpContractError);
  // k must be declared (no field = no default filled in)
  assertThrows(() => computeOutputShape(contract, [[6, 10]], "t"), OpContractError);
  assertThrows(() => assertNodeContract(topkNode({}), "t"), OpContractError, "required attr");
  // attrs outside the contract (torch's largest / sorted) fail loudly
  assertThrows(
    () => assertNodeContract(topkNode({ k: 2, sorted: true }), "t"),
    OpContractError,
    "attrs outside the contract",
  );
});

test("where broadcasts the three right-aligned, and the output has the same shape as the values side", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("where"), ins, "t")[0];
  // The condition has a lower rank than the values (the shape of spline's inside test)
  assertEquals(shape([[5], [2, 3, 5], [1]]), [2, 3, 5]);
  // A shape where only the condition is wider also passes, as in torch (the output is the broadcast
  // of the three)
  assertEquals(shape([[2, 3], [3], [3]]), [2, 3]);
});

test("broadcast is right-aligned as in torch, and a zero-length dimension is not collapsed by max", () => {
  assertEquals(broadcastShapes([2, 3, 4], [3, 1], "t"), [2, 3, 4]);
  assertEquals(broadcastShapes([3, 1], [2, 3, 4], "t"), [2, 3, 4]);
  assertEquals(broadcastShapes([5], [], "t"), [5]);
  assertEquals(broadcastShapes([], [], "t"), []);
  // MUST: max(0, 1) is 0, not 1 (torch's broadcast rule)
  assertEquals(broadcastShapes([0], [1], "t"), [0]);
  assertThrows(() => broadcastShapes([2, 3], [4, 3], "t"), OpContractError);
  assertThrows(() => broadcastShapes([0], [5], "t"), OpContractError);
});

test("output shape computation follows each op's contract", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs?: Readonly<Record<string, unknown>>,
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];
  assertEquals(shape("sum", [[4]], { dim: 0 }), []);
  assertEquals(shape("amin", [[2, 3, 4]], { dim: 0 }), [3, 4]);
  // cast / bitwise_not are elementwise, so the shape passes through
  assertEquals(shape("cast", [[2, 3, 4]]), [2, 3, 4]);
});

// ADR 0012: bmm is rank-3 only (rank-2 is matmul), and gather is fixed to the last dimension.
// MUST: check with B / M / K / N all of different lengths — a mixed-up axis is invisible in a
// square.
test("bmm accepts only rank-3 with matching batch and matching contraction", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("bmm"), ins, "t")[0];
  assertEquals(
    shape([
      [7, 1, 4],
      [7, 4, 9],
    ]),
    [7, 1, 9],
  );
  // rank-2 belongs to matmul (no dual use)
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
  // Mismatched batch dimensions (neither broadcast nor stride 0 is in the vocabulary)
  assertThrows(
    () =>
      shape([
        [2, 6, 3],
        [3, 3, 5],
      ]),
    OpContractError,
  );
});

test("gather requires matching leading dimensions, and the output has the same shape as index", () => {
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(resolveOpContract("gather"), ins, "t")[0];
  // The output's last dimension may be longer than src's (the same index may be looked up many
  // times)
  assertEquals(
    shape([
      [2, 3],
      [2, 7],
    ]),
    [2, 7],
  );
  assertEquals(shape([[5], [3]]), [3]);
  // Mismatched leading dimensions (a narrower contract than torch's general gather)
  assertThrows(
    () =>
      shape([
        [4, 6, 9],
        [3, 6, 6],
      ]),
    OpContractError,
  );
  // Mismatched rank, scalars
  assertThrows(() => shape([[], []]), OpContractError);
});

// ADR 0010: sym_prefix_slice computes the output shape from attrs (+ bindings) — the declaration is
// what gets checked, not "the declaration is the target shape" as for reshape / expand.
test("sym_prefix_slice decides the prefix length from attrs and bindings", () => {
  const shape = (
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown>,
    bindings?: Record<string, number>,
  ) => computeOutputShape(resolveOpContract("sym_prefix_slice"), ins, "t", { attrs, bindings })[0];
  const slice = (dim: number, coeff = 1, offset = 0) => ({ dim, coeff, offset });

  // Exceeding Tmax (it would read outside the constant buffer)
  assertThrows(() => shape([[8, 8]], { sym: "T", slices: [slice(0)] }, { T: 9 }), OpContractError);
  // A dim outside the input rank / a form where the binding is not passed
  assertThrows(() => shape([[8, 8]], { sym: "T", slices: [slice(0)] }), OpContractError);
  // MUST: bindings are looked up with Object.hasOwn only (a name from the prototype must not give a
  // NaN length)
  assertThrows(
    () => shape([[8, 8]], { sym: "toString", slices: [slice(0)] }, { T: 4 }),
    OpContractError,
  );

  // attrs schema: the same dim twice / coefficient 0 / negative offset / unknown key / empty
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

// ADR 0011: for reshape / expand, "the output's declared shape is the target shape", which cannot
// be derived from the input.
test("reshape / expand take the declared shape as the target shape, and permute is decided by dims", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    context: { declared?: readonly number[]; attrs?: Record<string, unknown> },
  ) => computeOutputShape(resolveOpContract(op), ins, "t", context)[0];

  assertEquals(shape("reshape", [[6]], { declared: [1, 6, 1] }), [1, 6, 1]);

  // Expanding a dimension whose length is not 1, lowering the rank
  assertThrows(() => shape("expand", [[2, 3]], { declared: [3] }), OpContractError);
  // dims that do not match the input rank / are out of range
  assertThrows(() => shape("permute", [[2, 3]], { attrs: { dims: [0, 2] } }), OpContractError);
  // MUST: do not silently guess when no target shape is passed
  assertThrows(() => shape("expand", [[2, 3]], {}), OpContractError);
  assertThrows(() => shape("permute", [[2, 3]], {}), OpContractError);
});

test("group_norm attrs: num_groups and eps are required, eps as an f32 normal (node level)", () => {
  // The value ranges are table rows (fixtures/op-contracts.json); this keeps the node-level checks
  // and the eps values that only the TS side rejects (they round to 0 / a subnormal in f32).
  const accept = (attrs: Record<string, unknown>) =>
    assertNodeContract(node("group_norm", ["x", "w", "b"], attrs), "t");
  const reject = (attrs: Record<string, unknown>, ins: readonly string[] = ["x", "w", "b"]) =>
    assertThrows(() => assertNodeContract(node("group_norm", ins, attrs), "t"), OpContractError);
  assertEquals(accept({ num_groups: 32, eps: 1e-6 }).kind, "groupNorm");
  assertEquals(groupNormAttrs({ num_groups: 32, eps: 1e-6 }, "t"), { groups: 32, eps: 1e-6 });
  reject({ num_groups: 32 });
  reject({ eps: 1e-6 });
  reject({ num_groups: 32, eps: 1e-50 });
  reject({ num_groups: 32, eps: 1e-40 });
  reject({ num_groups: 32, eps: 1e6, extra: 1 });
  // arity fixed at 3 (the affine is always present — the exporter synthesizes ones / zeros)
  reject({ num_groups: 32, eps: 1e-6 }, ["x", "w"]);
});

// ADR 0012: the attrs schemas of the fused ops (allowed keys + types + value ranges).
test("the attrs schemas of the fused ops check down to the value range", () => {
  // Key sets and the value domains of each attr are table rows; this test keeps the node-level
  // checks (missing / undeclared keys), the extractors, and values the table does not list.
  const accept = (op: string, ins: readonly string[], attrs: Record<string, unknown>) =>
    assertNodeContract(node(op, ins, attrs), "t");
  const reject = (op: string, ins: readonly string[], attrs: Record<string, unknown>) =>
    assertThrows(() => assertNodeContract(node(op, ins, attrs), "t"), OpContractError);

  // layer_norm — last dimension only (normalized_shape of length 1) / eps is a finite positive
  // number
  assertEquals(
    accept("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-7 }).kind,
    "layerNorm",
  );
  assertEquals(layerNormAttrs({ normalized_shape: [8], eps: 1e-7 }, "t"), {
    normalizedShape: [8],
    eps: 1e-7,
  });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8] });
  // An eps that rounds to 0 in f32 brings back the reason for "0 is not allowed" on the GPU side
  // only (the GPU carries it as an f32 word of params, the CPU reference computes in f64) — on an
  // all-zero row the GPU gives NaN and the CPU 0
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-50 });
  // An eps that rounds to an f32 subnormal is rejected too (WGSL does not require keeping
  // subnormals — an FTZ implementation flushes it to 0, and the same split as above comes back on
  // the GPU side only). The boundary = the smallest normal number 2^-126
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-45 });
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e-40 });
  assertEquals(
    accept("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1.1754943508222875e-38 })
      .kind,
    "layerNorm",
  );

  // rms_norm — attrs are only eps (the source of truth for the normalised length is weight — ADR
  // 0017)
  assertEquals(accept("rms_norm", ["x", "w"], { eps: 1e-6 }).kind, "rmsNorm");
  assertEquals(rmsNormEps({ eps: 1e-6 }, "t"), 1e-6);
  reject("rms_norm", ["x", "w"], { eps: 1e-50 });
  reject("rms_norm", ["x", "w"], {});
  // MUST: there is no normalized_shape field (an IR that copied the field over from layer_norm must
  // not pass)
  reject("rms_norm", ["x", "w"], { eps: 1e-6, normalized_shape: [8] });

  // clamp_min — attrs are only min (bringing in max would erase the distinction from two-sided
  // clamp)
  assertEquals(accept("clamp_min", ["x"], { min: 1e-12 }).kind, "unary");
  assertEquals(scalarParamValues(resolveOpContract("clamp_min"), { min: -1.5 }, "t"), [-1.5]);
  reject("clamp_min", ["x"], {});
  reject("clamp_min", ["x"], { min: 0, max: 1 });

  // softmax — non-negative axis numbers only (negative axis notation is normalised at the exporter
  // boundary)
  assertEquals(accept("softmax", ["x"], { dim: 0 }).kind, "softmax");
  assertEquals(softmaxDim({ dim: 3 }, "t"), 3);
  reject("softmax", ["x"], {});

  // safe_softmax — attrs identical to softmax (ADR 0044; only the kind differs)
  assertEquals(accept("safe_softmax", ["x"], { dim: 0 }).kind, "safeSoftmax");
  reject("safe_softmax", ["x"], {});

  // embedding — padding_idx is accepted and inert (-1 is torch's "unspecified" sentinel)
  assertEquals(accept("embedding", ["w", "i"], { padding_idx: -1 }).kind, "embedding");
  reject("embedding", ["w", "i"], {});

  // masked_fill — finite f32 scalars only
  assertEquals(accept("masked_fill", ["x", "m"], { value: 0 }).kind, "maskedFill");
  assertEquals(maskedFillValue({ value: -1.5 }, "t"), -1.5);
  reject("masked_fill", ["x", "m"], {});

  // conv1d — stride / dilation / groups are positive integers, padding a non-negative integer (ADR
  // 0015)
  const convAttrs = { stride: 1, padding: 1, dilation: 1, groups: 1 };
  assertEquals(accept("conv1d", ["x", "w", "b"], convAttrs).kind, "conv1d");
  assertEquals(conv1dAttrs({ stride: 2, padding: 0, dilation: 3, groups: 4 }, "t"), {
    stride: 2,
    padding: 0,
    dilation: 3,
    groups: 4,
  });
  reject("conv1d", ["x", "w", "b"], { ...convAttrs, output_padding: 0 });
  // MUST: no default is filled in for declared attrs (a missing one fails loudly — ADR 0012 /
  // 0015). Making dilation / groups optional would silently turn a depthwise IR into a regular
  // convolution.
  reject("conv1d", ["x", "w", "b"], { stride: 1, padding: 1 });
  reject("conv1d", ["x", "w", "b"], { stride: 1, padding: 1, dilation: 1 });
  reject("conv1d", ["x", "w", "b"], { stride: 1, padding: 1, groups: 1 });

  // conv2d — the three spatial attrs are 2-component [H, W] (the convention that scalar notation is
  // normalised at the boundary — ADR 0017)
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
  // No default is filled in for declared attrs (all four are required)
  reject("conv2d", ["x", "w", "b"], { stride: [1, 1], padding: [0, 0], dilation: [1, 1] });
  reject("conv2d", ["x", "w", "b"], { stride: [1, 1], padding: [0, 0], groups: 1 });
  reject("conv2d", ["x", "w", "b"], { ...conv2Attrs, output_padding: [0, 0] });

  // conv3d — the three spatial attrs are 3-component [T, H, W] (scalar notation and 2 components
  // are not accepted — ADR 0118 decision 1)
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
  // MUST: the three axes are checked independently (it fails if any single component is 0 — stride
  // 0 hangs the kernel)
  reject("conv3d", ["x", "w", "b"], { ...conv3Attrs, padding: [0, -1, 0] });
  // Lengths other than 3 and scalar notation are not accepted (2-component [H, W] cannot be told
  // apart from a forgotten time axis)
  reject("conv3d", ["x", "w", "b"], { ...conv3Attrs, dilation: [1, 1, 1, 1] });
  // No default is filled in for declared attrs (all four are required)
  reject("conv3d", ["x", "w", "b"], { stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1] });
  reject("conv3d", ["x", "w", "b"], { stride: [1, 1, 1], padding: [0, 0, 0], groups: 1 });
  // Arity fixed at 3 (a missing bias is normalised by the exporter synthesizing a zero bias — same
  // as the conv family)
  reject("conv3d", ["x", "w"], conv3Attrs);

  // conv_transpose1d — attrs are only stride / padding (not creating output_padding / dilation /
  // groups fields closes off the path of "silently running unmeasured values with defaults")
  assertEquals(
    accept("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: 0 }).kind,
    "convTranspose1d",
  );
  assertEquals(convTranspose1dAttrs({ stride: 8, padding: 4 }, "t"), { stride: 8, padding: 4 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2, padding: 0, output_padding: 0 });
  reject("conv_transpose1d", ["x", "w", "b"], { stride: 2 });
});

/**
 * ADR 0012: pins that masked_fill's measured fill value −3.4028234663852886e+38 (the smallest
 * finite f32) **keeps its ulp through a JSON round trip**. IR attrs travel as JSON number literals,
 * so if the round trip moved it by even 1 ulp, the fill value on the GPU would disagree with torch.
 */
test("the masked_fill fill value −3.4e38 keeps its ulp through a JSON round trip and f32 rounding", () => {
  const value = -3.4028234663852886e38;
  // It is the smallest finite f32 itself (rounding to f32 does not change the value = ulp
  // unchanged)
  assertEquals(Math.fround(value), value);
  assertEquals(new Float32Array([value])[0], value);
  // Even 1 ulp below does not fit as a finite f32 (= the basis for this being the smallest finite
  // value)
  assertEquals(Math.fround(-Number.MAX_VALUE), Number.NEGATIVE_INFINITY);
  // JSON round trip (graph JSON travels this path)
  const roundTrip = JSON.parse(JSON.stringify({ value })).value;
  assertEquals(roundTrip, value);
  assertEquals(Object.is(roundTrip, value), true);
  assertEquals(maskedFillValue({ value: roundTrip }, "t"), value);
  // The contract check also accepts the value after the round trip
  assertEquals(
    assertNodeContract(node("masked_fill", ["x", "m"], { value: roundTrip }), "t").name,
    "masked_fill",
  );
  // Non-finite values are not accepted, per the IR v1 rule
  assertThrows(() => maskedFillValue({ value: Number.NEGATIVE_INFINITY }, "t"), OpContractError);
  assertThrows(() => maskedFillValue({ value: Number.NaN }, "t"), OpContractError);
});

test("the output shapes of the fused ops follow the contract", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown> = {},
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];

  // linear — leading dimensions as is, the last dimension replaced by out
  assertEquals(shape("linear", [[2, 4, 6], [5, 6], [5]]), [2, 4, 5]);

  // layer_norm — normalized_shape matches the input's last dimension, and affine has the same shape
  const ln = { normalized_shape: [8], eps: 1e-5 };
  assertEquals(shape("layer_norm", [[2, 3, 8], [8], [8]], ln), [2, 3, 8]);
  assertThrows(() => shape("layer_norm", [[3, 8], [8], [3, 8]], ln), OpContractError);

  // softmax — last dimension only (a general dim is not in the vocabulary)
  assertEquals(shape("softmax", [[4, 9]], { dim: 1 }), [4, 9]);
  assertThrows(() => shape("softmax", [[2, 3, 4]], { dim: 3 }), OpContractError);

  // safe_softmax — the shape rule is identical to softmax
  assertEquals(shape("safe_softmax", [[4, 9]], { dim: 1 }), [4, 9]);

  // embedding — the output is the index shape with hidden appended
  assertEquals(shape("embedding", [[7, 4], [5]]), [5, 4]);

  // masked_fill — the output always has the same shape as x (mask is only read through a
  // right-aligned broadcast)
  const fill = { value: 0 };
  // MUST: a mask that widens x is rejected (using broadcastShapes as is would let it through)
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

  // conv1d — the output length is floor((L + 2P − D·(K−1) − 1) / S) + 1 (the general dilation form
  // of ADR 0015)
  const conv = { stride: 1, padding: 1, dilation: 1, groups: 1 };
  assertEquals(shape("conv1d", [[1, 1, 3], [2, 1, 3], [2]], { ...conv, padding: 2 }), [1, 2, 5]);
  // The general dilation form (DDSConv: depthwise g=C, k=5, d=3, p=6 keeps the length)
  assertEquals(
    shape("conv1d", [[1, 6, 17], [6, 1, 5], [6]], { ...conv, padding: 6, dilation: 3, groups: 6 }),
    [1, 6, 17],
  );
  // groups must divide both Cin and Cout (the weight's second axis is Cin/groups)
  assertThrows(
    () => shape("conv1d", [[1, 6, 11], [8, 2, 3], [8]], { ...conv, groups: 3 }),
    OpContractError,
  );
  assertThrows(() => shape("conv1d", [[3, 9], [4, 3, 3], [4]], conv), OpContractError);

  // conv_transpose1d — the weight is [Cin, Cout, K], the output length is L·stride (only the form
  // 2P == K − S)
  assertEquals(
    shape("conv_transpose1d", [[1, 3, 5], [3, 2, 16], [2]], { stride: 8, padding: 4 }),
    [1, 2, 40],
  );
});

// ADR 0011: the rank limit of the strided copy family (permute / expand / sym_prefix_slice /
// masked_fill) is checked at **the contract layer**. Letting it fall through to codegen
// (stridedParams) would show users "the contract check passed but an internal error at run
// time", and the diagnosis would not record which value of which op exceeded the rank.
test("the rank limit (1..4) of the strided copy family fails at the contract layer", () => {
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

  // Exactly the limit (rank 4) passes — a mistake that lowers the limit turns red here too
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

  // rank 5 does not fit in the strided kernel's params (fixed at rank 4)
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

  // rank 0 (a scalar) does not fit either, for the same reason — a form with only the mask at rank
  // 0 used to slip through the contract
  assertThrows(() => shape("permute", [[]], { attrs: { dims: [0] } }), OpContractError, "strided");
  assertThrows(
    () => shape("masked_fill", [[2, 3], []], { attrs: { value: 0 } }),
    OpContractError,
    "strided",
  );
});

/**
 * **Every check the params layer has, the contract layer has too** (especially the checks the
 * kernel-side doc describes as "duplicated with the contract check"). If a check existed on one
 * side only, the accepted sets would split across layers, and a shape could be built that only the
 * CPU reference going through `computeOutputShape` can run (the GPU throws in params). For an eps
 * that overflows f32 there is not even an exception: the GPU (`1/sqrt(x + Inf) = 0`, all bias) and
 * the reference (normalised in finite f64) silently give different values.
 *
 * MUST: the diagnosis names **the value that caused it** (when the layers were split, the symptom
 * was "the contract passed but an internal error at run time", and users were not told which attr /
 * axis was out of range).
 */
test("the params layer's value-range checks are matched at the contract layer (f32 range / kernel length / geometry / hidden width)", () => {
  const shape = (
    op: string,
    ins: readonly (readonly number[])[],
    attrs: Record<string, unknown> = {},
  ) => computeOutputShape(resolveOpContract(op), ins, "t", { attrs })[0];
  const reject = (op: string, ins: readonly string[], attrs: Record<string, unknown>) =>
    assertThrows(() => assertNodeContract(node(op, ins, attrs), "t"), OpContractError, "1e+39");

  // eps is finite **as f32** (paired with the underflow 1e-50, the overflow 1e39 also fails)
  reject("layer_norm", ["x", "w", "b"], { normalized_shape: [8], eps: 1e39 });
  reject("rms_norm", ["x", "w"], { eps: 1e39 });
  assertThrows(
    () => shape("layer_norm", [[3, 8], [8], [8]], { normalized_shape: [8], eps: 1e39 }),
    OpContractError,
    "infinite",
  );
  assertThrows(() => shape("rms_norm", [[3, 8], [8]], { eps: 1e39 }), OpContractError, "infinite");
  // A normal eps passes for both (the counterpart showing the check is not too wide)
  assertEquals(
    shape("layer_norm", [[3, 8], [8], [8]], { normalized_shape: [8], eps: 1e-5 }),
    [3, 8],
  );
  assertEquals(shape("rms_norm", [[3, 8], [8]], { eps: 1e-5 }), [3, 8]);

  // Scalar attrs that go into params' f32 words go through the same check (as the declaration "a
  // finite f32 scalar" says)
  reject("attention", ["q", "k", "v"], { scale: 1e39 });
  reject("masked_fill", ["x", "m"], { value: 1e39 });
  reject("clamp", ["x"], { min: -1e39, max: 1 });
  reject("clamp", ["x"], { min: 0, max: 1e39 });
  reject("clamp_min", ["x"], { min: 1e39 });
  reject("leaky_relu", ["x"], { negative_slope: 1e39 });
  reject("ge_scalar", ["x"], { value: 1e39 });
  // A value that rounds normally to f32 and exactly the largest finite f32 pass (both sides agree
  // through the convention of rounding on application)
  assertEquals(
    assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.1 }), "t").kind,
    "attention",
  );
  assertEquals(
    assertNodeContract(node("clamp", ["x"], { min: -3.4028234663852886e38, max: 0.1 }), "t").kind,
    "unary",
  );

  // K = 0 of conv1d / conv2d fails **before the output-length formula** (if `D·(K−1)` turned
  // negative, an output longer than the input would be derived — the params side rejects it as a
  // positive integer, so only the contract layer accepted it)
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

  // deform_conv2d — the static geometry uses the same set as the params' positive set (a spatial
  // length of 0 could derive even a positive output shape depending on padding, and the reference
  // returned all-taps-out-of-range = all bias as a valid value)
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

  // gru_scan — the range of H (1 to 256) is a by-design constraint published in
  // docs/limitations.md, and only params checked it (H=0 / H=257 passed the contract, and only the
  // reference could run them)
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
  // Exactly the limit passes (a mistake that lowers the limit turns red here too)
  assertEquals(shape("gru_scan", [[1, 1, 768], [1, 256], [768, 256], [768]]), [1, 1, 256]);
});

/**
 * The **optional fourth input of the fused attention = an additive mask** (ADR 0023 revision).
 *
 * The kernel reads the mask as `[1,1,M,N]` with a flat index without the batch base (broadcast over
 * all of B·H), so any other shape **quietly breaks the values** (no exception). The contract layer
 * is the only detector, so the whole surface of acceptance and rejection is pinned here.
 */
test("attention's mask accepts only f32 of exactly [1,1,M,N]", () => {
  const contract = resolveOpContract("attention");
  const q = [2, 3, 5, 4];
  const k = [2, 3, 7, 4];
  const attrs = { scale: 0.5 };
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(contract, ins, "t", { attrs })[0];

  // The arity is 3 or 4 (not variable — "any number" is cat only)
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

  // Accepted: no mask / [1,1,M,N] (including non-square M ≠ N)
  assertThrows(() => shape([q, k, k, [2, 1, 5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [1, 3, 5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [2, 3, 5, 7]]), OpContractError, "[1,1,M,N]");
  // Rejected: rank ≠ 4
  assertThrows(() => shape([q, k, k, [5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [1, 5, 7]]), OpContractError, "[1,1,M,N]");
  assertThrows(() => shape([q, k, k, [1, 1, 1, 5, 7]]), OpContractError, "[1,1,M,N]");
  // Rejected: mismatched M / N (a form with M and N mixed up slips through only with a square mask,
  // so both are checked)
  assertThrows(() => shape([q, k, k, [1, 1, 4, 7]]), OpContractError, "M / N");
  assertThrows(() => shape([q, k, k, [1, 1, 5, 5]]), OpContractError, "M / N");
  assertThrows(() => shape([q, k, k, [1, 1, 7, 5]]), OpContractError, "M / N");

  // Rejected: dtype (a uniform contract, so the mask is f32 too — a bool mask is masked_fill's
  // vocabulary)
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

// ---- The states field and effect ops (ADR 0067 decisions 4 / 5) -----------------

/** The minimal `state_append` node (no outputs, a states field of exactly `{ slot }`). */
const appendNode = (
  attrs: Record<string, unknown> = {},
  states: Record<string, string> = { slot: "kv.k" },
): IrNode => ({ op: "state_append", ins: ["chunk"], outs: [], attrs, states });

test("state_append is an effect op with no outputs, and the states field is required", () => {
  const contract = resolveOpContract("state_append");
  assertEquals(contract.kind, "stateAppend");

  assertStrictEquals(assertNodeContract(appendNode(), "t"), contract);
  // Writing outputs violates the contract of "an op that defines no value"
  assertThrows(
    () => assertNodeContract({ ...appendNode(), outs: ["y"] }, "t"),
    OpContractError,
    "output count is 1 (contract: 0)",
  );
  // The states field itself is required (required: true)
  assertThrows(
    () => assertNodeContract({ ...appendNode(), states: {} }, "t"),
    OpContractError,
    "states field is missing",
  );
  // The key set is exactly { slot }
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

// Zero outputs are allowed **only** for ops whose contract declares an effect (the parser gives the
// count no meaning).
test("the contract layer rejects empty outs for an op that is not an effect", () => {
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

test("states cannot be written on an op without a states field", () => {
  assertThrows(
    () => assertNodeContract(node("relu", ["a"], {}, { slot: "kv.k" }), "t"),
    OpContractError,
    "has no states field",
  );
  // Same for topk (a multi-output op) — the states field is a contract surface of only attention /
  // state_append
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

test("attention names and rejects a degenerate q with B = 0 / M = 0 at the contract layer (W-R8-1)", () => {
  const contract = resolveOpContract("attention");
  const shape = (ins: readonly (readonly number[])[]) =>
    computeOutputShape(contract, ins, "t", { attrs: { scale: 0.5 } })[0];
  // Control: the normal shape returns the output [B,H,M,D]
  assertEquals(
    shape([
      [1, 2, 4, 8],
      [1, 2, 4, 8],
      [1, 2, 4, 8],
    ]),
    [1, 2, 4, 8],
  );
  // B = 0 satisfies "exact match", so it slipped through — name it at the contract layer before the
  // run phase's row-block planner fails with a diagnosis that does not mention attention
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
  // M = 0 of the conventional form (the states form already rejected it)
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

test("attention's states form is told apart by the presence of the field, and the conventional form's accepted set does not move", () => {
  const contract = resolveOpContract("attention");
  const states = { k: "kv.k", v: "kv.v" };
  // The conventional form (no field) passes with 3 or 4 inputs — not a single byte moves
  assertStrictEquals(
    assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }), "t"),
    contract,
  );
  assertNodeContract(node("attention", ["q", "k", "v", "m"], { scale: 0.5 }), "t");
  // The states form has exactly 3 inputs (the mask is fixed to causal, so it takes none — ADR 0067
  // decision 4)
  assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }, states), "t");
  assertThrows(
    () => assertNodeContract(node("attention", ["q", "k", "v", "m"], { scale: 0.5 }, states), "t"),
    OpContractError,
    "states form takes exactly 3 inputs",
  );
  // The key set is exactly { k, v }
  assertThrows(
    () =>
      assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5 }, { k: "kv.k" }), "t"),
    OpContractError,
    "states field keys are",
  );
  // Writing the same slot for k and v is a mix-up the shape does not catch (the k/v slots have the
  // same shape)
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

test("the optional attr `window` can be declared only on nodes with a states field", () => {
  const states = { k: "kv.k", v: "kv.v" };

  // In the states form it can be declared and also omitted (no field = the whole context)
  assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5, window: 512 }, states), "t");
  assertNodeContract(appendNode({ window: 512 }), "t");
  assertNodeContract(appendNode({}), "t");
  // Writing it on the conventional form would make "an attr no one reads", so it is rejected
  // (symmetry of the accepted sets)
  assertThrows(
    () => assertNodeContract(node("attention", ["q", "k", "v"], { scale: 0.5, window: 512 }), "t"),
    OpContractError,
    "can be declared only on a node with a states field",
  );
  // MUST: JSON `512.0` / `5.12e2` are accepted (JSON.parse collapses an integer-valued float into a
  // single number, so the TS side has no distinction at all). Python's `_assert_integer_attr`
  // normalises integer-valued floats to int to match this accepted set — if only one side rejected
  // them, it would be the divergence "only the exporter's validation cannot read a graph the
  // runtime can read", so both sides' accepted sets are spelled out here.
  for (const text of ["512.0", "5.12e2"]) {
    const window: unknown = JSON.parse(text);
    assertEquals(window, 512, `${text}: JSON.parse did not collapse it to an integer`);
    assertNodeContract(appendNode({ window }), "t");
    assertEquals(stateWindow({ window }, "t"), 512);
  }
  // The capability projection carries the **union** of required and optional (so that the
  // enumerating check does not reject the states form)
  assertEquals([...(RUNTIME_SUPPORT.ops.get("attention")?.attrKeys ?? [])].sort(), [
    "readonly",
    "scale",
    "window",
  ]);
  assertEquals([...(RUNTIME_SUPPORT.ops.get("state_append")?.attrKeys ?? [])], ["window"]);
  // attrs outside the contract are judged as before (accepting optional attrs widens only window)
  assertThrows(
    () => assertNodeContract(appendNode({ windows: 512 }), "t"),
    OpContractError,
    "attrs outside the contract [windows]",
  );
});

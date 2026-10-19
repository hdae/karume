// Graph JSON types and structural validation of IR v2 (docs/ir-v2.md).
// Validation is consolidated here: checking declarations against supplies belongs to
// container/bind.ts and the dimension grammar to dims.ts; this file checks only the rules determined
// by the graph alone (declarations, SSA, topological order, vocabulary).

import type { CodecName } from "./container/codecs.ts";
import {
  canonicalJsonText,
  canonicalJsonValue,
  sortedByCodePoints,
  sortedObject,
} from "./container/json.ts";
import { isSymbolName, parseDim, tryParseDim } from "./dims.ts";

/**
 * Semantic dtype. Computation always happens in this type (f16/bf16/i8 are storage-only concepts).
 */
export type IrDtype = "f32" | "i32" | "bool";

/** A non-negative integer, or the canonical spelling of `coeff·sym+offset` (dims.ts). */
export type IrDim = number | string;

type IrValueInfo = {
  readonly dtype: IrDtype;
  readonly shape: readonly IrDim[];
};

/**
 * Named state slot (ADR 0066 decision 2). The shape is **a concrete shape including the
 * capacity**, and unlike `values` a slot is not a value — it never becomes an ordinary graph input /
 * output, and the GenerationContext owns the allocation and lifetime of its backing data.
 */
type IrStateSlot = {
  readonly dtype: IrStateDtype;
  readonly shape: readonly IrDim[];
  /**
   * **A borrowed slot** (ADR 0096, addendum of 2026-09-08 on stage 2, first item). When true, the
   * backing data lives not in this context but in **the lender context** (the target's KV the
   * drafter reads), and this context allocates not a single byte. It can have no writer
   * (`state_append`), and its only readers are readonly attention (enforced by
   * `assertStateOwnership` in `runtime/plan.ts`).
   *
   * MUST: an always-present field (false when omitted). The same reason as `IrNode.states` — with
   * `?:` every consumer would have to write `?? false`, and forgetting it in one place becomes a
   * silent wrong value that "allocates the borrowed slot itself and reads an empty past".
   */
  readonly external: boolean;
};

type IrInput = {
  readonly name: string;
  readonly dtype: IrDtype;
  readonly shape: readonly IrDim[];
};

/**
 * Storage after the merge (docs/ir-v2.md "Values and types", container-v1 §6). `codec` is the
 * ledger's registered name, and the decode path is looked up with `codecLayout(codec)` (`ternary`
 * takes the same i2 path as `int2-off`). `groupSize` / `rowAxis` exist only for quantized codecs
 * (`groupSize` = row length for per-channel). The companion scale's data travels with the payload
 * in the supply plan (the result of merging with the binding table), so it is not here.
 */
export type IrStorage = {
  readonly codec: CodecName;
  readonly groupSize?: number;
  readonly rowAxis?: 0 | 1;
};

/**
 * An initializer of the graph the runtime executes. **The name is the key of the data** (the
 * exporter's FQN / `const.<hash>`), and the container's binding table holds the mapping to the data
 * by `(graph name, initializer name)`.
 *
 * `shared` declares borrowing the weight of a lender Session (ADR 0096, addendum of 2026-09-08 on
 * stage 2, first item) — the name equals the lender's initializer name, and it has no storage (the
 * source of truth is the lender's resident weight — no double bookkeeping).
 */
export type IrInitializer =
  | { readonly shared: true; readonly storage?: undefined }
  | { readonly shared?: undefined; readonly storage: IrStorage };

export type IrNode = {
  readonly op: string;
  readonly ins: readonly string[];
  readonly outs: readonly string[];
  readonly attrs: Readonly<Record<string, unknown>>;
  /**
   * Name references to state slots (ADR 0067 decision 4 — **a field separate from** `ins` /
   * `outs`). The keys are fixed words set by the op contract (`k` / `v` of `attention`, `slot` of
   * `state_append`), and the values are slot names declared in `graph.states`. A node without the
   * field has an empty table.
   *
   * MUST: an always-present field (an empty table when omitted). With `?:` every consumer would have
   * to write `?? {}`, and forgetting it in even one place becomes a silent error of "running as the
   * old form, ignoring the states field".
   */
  readonly states: Readonly<Record<string, string>>;
};

/** Sections other than initializers (shared by the declaration and the merged form). */
type IrGraphBase = {
  readonly format: typeof IR_FORMAT;
  readonly version: number;
  readonly requires: { readonly ops: readonly string[] };
  readonly symbols: readonly string[];
  readonly inputs: readonly IrInput[];
  readonly outputs: readonly string[];
  readonly values: Readonly<Record<string, IrValueInfo>>;
  /** State slot declarations (empty when omitted — graphs without them are unaffected). */
  readonly states: Readonly<Record<string, IrStateSlot>>;
  readonly nodes: readonly IrNode[];
};

/**
 * Initializer declaration of IR v2 (docs/ir-v2.md). It has no storage (codec / scale / group) — the
 * container's binding table holds the mapping to the data by `(graph name, initializer name)`.
 * `shared` declares borrowing the weight of a lender Session (the name MUST equal the lender's
 * initializer name).
 */
export type IrDeclaredInitializer = { readonly shared: boolean };

/**
 * An IR v2 graph = **a computation contract** (no storage declarations). The contents of
 * `graphs[name]` in the container's graph descriptor; merging with the binding table
 * (`format/container/bind.ts`) turns it into an {@link IrGraph}.
 */
export type IrDeclaration = IrGraphBase & {
  readonly initializers: Readonly<Record<string, IrDeclaredInitializer>>;
};

/** The graph the runtime executes (with the storage of every initializer settled). */
export type IrGraph = IrGraphBase & {
  readonly initializers: Readonly<Record<string, IrInitializer>>;
};

export class IrError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IrError";
  }
}

const IR_FORMAT = "karume-ir";
/** IR v2 (docs/ir-v2.md). */
export const IR_VERSION = 2;

const TOP_LEVEL_KEYS = [
  "format",
  "version",
  "requires",
  "symbols",
  "inputs",
  "outputs",
  "initializers",
  "values",
  "nodes",
] as const;

/**
 * Optional top-level sections. That graphs without `states` (= every existing model) are unaffected
 * is the flip side of ADR 0066 decision 2's "the exporter is unaffected until the first model that
 * emits states".
 */
const OPTIONAL_TOP_LEVEL_KEYS = ["states"] as const;

/**
 * The whole vocabulary of semantic dtypes (the accepted set as declarations — whether one can run
 * is held by the contract table, src/ops.ts).
 */
export const SEMANTIC_DTYPES = ["f32", "i32", "bool"] as const;

/** dtype vocabulary of state slots. f32 only for now (ADR 0066 decision 2). */
const STATE_DTYPES = ["f32"] as const;
type IrStateDtype = (typeof STATE_DTYPES)[number];

/**
 * State slot dtypes that are only reserved (ADR 0066, addendum of 2026-08-17, item 5). f16
 * storage changes the numerics contract and so needs an opt-in in the style of ADR 0058 — until
 * that exists, they fail as **unsupported**, not as "outside the vocabulary" (leaving a diagnostic
 * that says the form is coming later).
 */
const RESERVED_STATE_DTYPES = ["f16"] as const;

/**
 * Upper bound of a state slot's rank (the "fixed rank (rank ≤ 4)" of ADR 0066 decision 2). The same
 * number as the strided kernels' limit (codegen/strided.ts), but for a different reason, so the
 * constant is not shared.
 */
const MAX_STATE_RANK = 4;

/**
 * Semantic dtypes an initializer may declare (docs/ir-v2.md "Values and types").
 *
 * MUST: bool initializers are not in the vocabulary (none in measurements, and 1-byte storage does
 * not fit the transfers, which assume 4 bytes). The pairs of semantic dtype and storage codec (f32
 * takes its encodings / i32 takes raw int32 only — ADR 0010) belong to the merge layer
 * (`format/container/bind.ts`) — allowing cross pairings would read an initializer declared `i32` as
 * f16 bits, a silent wrong value.
 */
const INITIALIZER_DTYPES: ReadonlySet<IrDtype> = new Set<IrDtype>(["f32", "i32"]);

export const isSemanticDtype = (value: unknown): value is IrDtype =>
  SEMANTIC_DTYPES.some((dtype) => dtype === value);

const isStateDtype = (value: string): value is IrStateDtype =>
  STATE_DTYPES.some((dtype) => dtype === value);

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asPlainObject = (value: unknown, where: string): Record<string, unknown> => {
  if (!isPlainObject(value)) throw new IrError(`${where}: is not an object`);
  return value;
};

/**
 * Every element of an array field. A declaration built on the host ({@link parseIrDeclarationValue})
 * can contain sparse arrays, but `map` / `some` / `reduce` skip holes, so a hole slips past the
 * checks and later becomes an undefined dimension or name. Holes fail here with the field's path
 * (JSON.parse never produces holes).
 */
const asArray = (value: unknown, where: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new IrError(`${where}: is not an array`);
  const list: readonly unknown[] = value;
  for (let index = 0; index < list.length; index += 1) {
    if (!Object.hasOwn(list, index))
      throw new IrError(`${where}[${index}] is missing (a hole in a sparse array)`);
  }
  return list;
};

const asNonEmptyString = (value: unknown, where: string): string => {
  if (typeof value !== "string" || value.length === 0) {
    throw new IrError(`${where}: is not a non-empty string`);
  }
  return value;
};

/**
 * A string array without duplicates (a field treated as a set — duplicates would make the
 * set-equality check ambiguous).
 */
const asUniqueStrings = (value: unknown, where: string): string[] => {
  const list = asArray(value, where).map((item, index) =>
    asNonEmptyString(item, `${where}[${index}]`),
  );
  const seen = new Set<string>();
  for (const item of list) {
    if (seen.has(item)) throw new IrError(`${where}: '${item}' is duplicated`);
    seen.add(item);
  }
  return list;
};

/**
 * Unreleased, so no forward-compatibility channel — unknown keys fail loudly instead of being
 * ignored silently.
 */
const checkKeys = (
  obj: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[],
  where: string,
): void => {
  for (const key of required) {
    if (!Object.hasOwn(obj, key)) throw new IrError(`${where}: required key '${key}' is missing`);
  }
  for (const key of Object.keys(obj)) {
    if (!required.includes(key) && !optional.includes(key)) {
      throw new IrError(`${where}: unknown key '${key}'`);
    }
  }
};

const asSemanticDtype = (value: unknown, where: string): IrDtype => {
  const dtype = asNonEmptyString(value, where);
  if (!isSemanticDtype(dtype)) {
    throw new IrError(
      `${where}: semantic dtype '${dtype}' is outside the vocabulary (${SEMANTIC_DTYPES.join(" / ")})`,
    );
  }
  return dtype;
};

const parseShape = (value: unknown, symbols: ReadonlySet<string>, where: string): IrDim[] =>
  asArray(value, where).map((dim, index) => {
    const at = `${where}[${index}]`;
    if (typeof dim === "number") {
      if (!Number.isSafeInteger(dim) || dim < 0) {
        throw new IrError(`${at}: dimension ${dim} is not a non-negative integer`);
      }
      return dim;
    }
    if (typeof dim !== "string")
      throw new IrError(`${at}: dimension is neither a number nor a string`);
    const expr = tryParseDim(dim);
    if (expr === undefined) {
      throw new IrError(
        `${at}: dimension expression '${dim}' does not match the canonical grammar coeff·sym+offset`,
      );
    }
    if (!symbols.has(expr.sym)) {
      throw new IrError(`${at}: symbol '${expr.sym}' is not declared in graph.symbols`);
    }
    return dim;
  });

const asStateDtype = (value: unknown, where: string): IrStateDtype => {
  const dtype = asNonEmptyString(value, where);
  if (RESERVED_STATE_DTYPES.some((reserved) => reserved === dtype)) {
    throw new IrError(
      `${where}: state slot dtype '${dtype}' is not supported (a slot reserved by ADR 0066 addendum 5 — ` +
        `needs a numerics-contract opt-in)`,
    );
  }
  if (!isStateDtype(dtype)) {
    throw new IrError(
      `${where}: state slot dtype '${dtype}' is outside the vocabulary (${STATE_DTYPES.join(" / ")})`,
    );
  }
  return dtype;
};

/**
 * Declaration of one state slot (ADR 0066 decision 2).
 *
 * MUST: the shape is **a concrete shape including the capacity**, so numeric dimensions are positive
 * integers (unlike the non-negative ones of `values` — a slot of capacity 0 has no backing data to
 * bind). The rank is 1..{@link MAX_STATE_RANK} (rank 0 has no capacity axis and cannot be
 * "including the capacity"). A symbolic dimension is fine if declared in `symbols`, and **a states
 * shape is a binding point too** (the capacity chosen by `createGenerationContext` — ADR 0066,
 * addendum of 2026-08-17, item 7; checkSymbolBindability).
 */
const parseStateSlot = (
  value: unknown,
  symbols: ReadonlySet<string>,
  where: string,
): IrStateSlot => {
  const obj = asPlainObject(value, where);
  checkKeys(obj, ["dtype", "shape"], ["external"], where);
  const dtype = asStateDtype(obj["dtype"], `${where}.dtype`);
  const shape = parseShape(obj["shape"], symbols, `${where}.shape`);
  // MUST: only `true` may be written. `false` is the same declaration as "the field is absent", so
  // allowing both spellings would make "forgot to write external" and "explicitly turned external
  // off" indistinguishable to the reader.
  if (Object.hasOwn(obj, "external") && obj["external"] !== true) {
    throw new IrError(
      `${where}.external: ${JSON.stringify(obj["external"])} cannot be written` +
        " (the absence of the field means 'allocate it yourself' — write true only when it is borrowed)",
    );
  }
  const external = Object.hasOwn(obj, "external");
  if (shape.length < 1 || shape.length > MAX_STATE_RANK) {
    throw new IrError(
      `${where}.shape: rank ${shape.length} is outside 1..${MAX_STATE_RANK} (a concrete fixed-rank shape including capacity MUST)`,
    );
  }
  shape.forEach((dim, index) => {
    if (typeof dim === "number" && dim < 1) {
      throw new IrError(
        `${where}.shape[${index}]: dimension ${dim} is not a positive integer (no capacity can be allocated)`,
      );
    }
  });
  return { dtype, shape, external };
};

/**
 * The `states` field of a node (ADR 0067 decision 4). Only **three points determined by the graph
 * alone** are checked here: it is a plain object, its keys and values are non-empty strings, and its
 * values are slot names declared in `graph.states`.
 *
 * MUST: a reference to an undeclared slot fails loudly. Letting it through, a node that "reads a
 * name without backing data" gets past Session construction and fails only at execution, never
 * allocated or bound (a rule of the same layer as rejecting forward references on the value side).
 * MUST: the container object has a null prototype. Keys come from the parse input, and on a plain
 * `{}` assigning `"__proto__"` turns into setting [[Prototype]] and the field silently disappears —
 * if another node's reference covers the vanished one, a form can be built that slips past both the
 * reference-completeness check and the contract's key-set check.
 *
 * NOTE: the key set itself (exactly `{k,v}` / exactly `{slot}`) is the op contract's job
 * (`assertNodeContract`) — the parser does not know the op vocabulary.
 */
const parseNodeStates = (
  obj: Record<string, unknown>,
  slots: Readonly<Record<string, IrStateSlot>>,
  where: string,
): Record<string, string> => {
  const states: Record<string, string> = Object.create(null);
  if (!Object.hasOwn(obj, "states")) return states;
  for (const [key, value] of Object.entries(asPlainObject(obj["states"], `${where}.states`))) {
    asNonEmptyString(key, `${where}.states key`);
    const slot = asNonEmptyString(value, `${where}.states['${key}']`);
    if (!Object.hasOwn(slots, slot)) {
      throw new IrError(
        `${where}.states['${key}']: state slot '${slot}' is not declared in graph.states`,
      );
    }
    states[key] = slot;
  }
  return states;
};

/**
 * NaN / Infinity are not JSON literals, but an exponent like `1e999` rounds to Infinity.
 * To keep the contract of aligning the accepted set with the browser's JSON.parse (docs/ir-v2.md),
 * a graph containing non-finite numbers MUST NOT be accepted.
 */
const parseJson = (json: string): unknown => {
  try {
    return JSON.parse(json, (key: string, value: unknown) => {
      if (typeof value === "number" && !Number.isFinite(value)) {
        throw new IrError(`'${key}' in the graph JSON is not a finite number`);
      }
      return value;
    });
  } catch (cause) {
    if (cause instanceof IrError) throw cause;
    throw new IrError(`Cannot parse the graph JSON: ${String(cause)}`);
  }
};

/**
 * The core of graph JSON parsing. Only how initializer values are read can be swapped through
 * `parseInitializer` (the current reader is {@link parseIrDeclaration} = the single kind of IR v2
 * declaration without storage). The rules determined by the graph alone (declarations, SSA,
 * topological order, vocabulary, states) all go through this one function.
 */
const parseIrDocument = <I>(
  document: unknown,
  expectedVersion: number,
  parseInitializer: (obj: Record<string, unknown>, where: string) => I,
): IrGraphBase & { readonly initializers: Readonly<Record<string, I>> } => {
  const root = asPlainObject(document, "graph");
  checkKeys(root, TOP_LEVEL_KEYS, OPTIONAL_TOP_LEVEL_KEYS, "graph");

  const format = root["format"];
  if (format !== IR_FORMAT) {
    throw new IrError(`graph.format is not '${IR_FORMAT}': ${JSON.stringify(format)}`);
  }
  const version = root["version"];
  if (version !== expectedVersion) {
    throw new IrError(`graph.version is not ${expectedVersion}: ${JSON.stringify(version)}`);
  }

  const requires = asPlainObject(root["requires"], "graph.requires");
  checkKeys(requires, ["ops"], [], "graph.requires");
  const requiredOps = asUniqueStrings(requires["ops"], "graph.requires.ops");

  const symbols = asUniqueStrings(root["symbols"], "graph.symbols");
  for (const symbol of symbols) {
    if (!isSymbolName(symbol)) throw new IrError(`graph.symbols: invalid symbol name '${symbol}'`);
  }
  const symbolSet = new Set(symbols);

  const inputs: IrInput[] = asArray(root["inputs"], "graph.inputs").map((raw, index) => {
    const where = `graph.inputs[${index}]`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["name", "dtype", "shape"], [], where);
    return {
      name: asNonEmptyString(obj["name"], `${where}.name`),
      dtype: asSemanticDtype(obj["dtype"], `${where}.dtype`),
      shape: parseShape(obj["shape"], symbolSet, `${where}.shape`),
    };
  });

  const outputs = asUniqueStrings(root["outputs"], "graph.outputs");

  // MUST: a container object that accumulates keys from the parse input has a null prototype. On a
  // plain `{}`, when the name is "__proto__" the assignment turns into setting [[Prototype]], no own
  // property is created, the declaration silently disappears and the graph is accepted. Deno
  // disabled this setter by default, so it did not reproduce there, but it happens on Node and in
  // browsers (the target runtimes).
  const values: Record<string, IrValueInfo> = Object.create(null);
  for (const [name, raw] of Object.entries(asPlainObject(root["values"], "graph.values"))) {
    // Empty value names are rejected for the same reason as states slot names — the referencing
    // sides (inputs[].name / ins / outs / outputs) all accept only non-empty strings, so letting one
    // through creates "a value that can be declared but in principle never referenced". The
    // orphan-declaration check is bypassed when it is written together with an empty-named
    // initializer.
    asNonEmptyString(name, "value name in graph.values");
    const where = `graph.values['${name}']`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["dtype", "shape"], [], where);
    values[name] = {
      dtype: asSemanticDtype(obj["dtype"], `${where}.dtype`),
      shape: parseShape(obj["shape"], symbolSet, `${where}.shape`),
    };
  }

  // MUST: a null prototype, for the same reason as values (see the comment above).
  const initializers: Record<string, I> = Object.create(null);
  for (const [name, raw] of Object.entries(
    asPlainObject(root["initializers"], "graph.initializers"),
  )) {
    // Empty initializer names are rejected likewise (see the comment on values — an empty-named
    // initializer written together with an empty-named value satisfies both sides of the
    // declaration-completeness check, and one unreferenceable real tensor stays in the
    // distribution).
    asNonEmptyString(name, "initializer name in graph.initializers");
    const where = `graph.initializers['${name}']`;
    initializers[name] = parseInitializer(asPlainObject(raw, where), where);
  }

  // MUST: a null prototype, for the same reason as values (see the comment above). Omission is
  // treated as an empty slot set (graphs without the section are unaffected — ADR 0066 decision 2).
  const states: Record<string, IrStateSlot> = Object.create(null);
  if (Object.hasOwn(root, "states")) {
    for (const [name, raw] of Object.entries(asPlainObject(root["states"], "graph.states"))) {
      // Empty slot names are rejected — the referencing field (ADR 0067) accepts only non-empty
      // strings, so letting one through creates "a slot that can be declared but in principle never
      // referenced" (for values, the orphan-declaration check closes the same hole).
      asNonEmptyString(name, "slot name in graph.states");
      states[name] = parseStateSlot(raw, symbolSet, `graph.states['${name}']`);
    }
  }

  const nodes: IrNode[] = asArray(root["nodes"], "graph.nodes").map((raw, index) => {
    const where = `graph.nodes[${index}]`;
    const obj = asPlainObject(raw, where);
    checkKeys(obj, ["op", "ins", "outs", "attrs"], ["states"], where);
    // NOTE: the number of `outs` is not checked here (zero = an effect op that defines no value is
    // in the vocabulary — ADR 0067 decision 5). "Only ops whose contract declares an effect may have
    // zero" is enforced in the contract layer (the output-count check of assertNodeContract); the
    // parser gives the count no meaning.
    return {
      op: asNonEmptyString(obj["op"], `${where}.op`),
      ins: asArray(obj["ins"], `${where}.ins`).map((input, i) =>
        asNonEmptyString(input, `${where}.ins[${i}]`),
      ),
      outs: asArray(obj["outs"], `${where}.outs`).map((out, i) =>
        asNonEmptyString(out, `${where}.outs[${i}]`),
      ),
      attrs: asPlainObject(obj["attrs"], `${where}.attrs`),
      states: parseNodeStates(obj, states, where),
    };
  });

  checkSymbolBindability(symbols, inputs, states, values);
  const defined = checkDefinitions(inputs, Object.keys(initializers), nodes, outputs);
  checkDeclarations(inputs, Object.keys(initializers), values, nodes, defined);
  checkStateSlots(states, values, defined, nodes);
  checkRequiredOps(requiredOps, nodes);

  return {
    format: IR_FORMAT,
    version: expectedVersion,
    requires: { ops: requiredOps },
    symbols,
    inputs,
    outputs,
    initializers,
    values,
    states,
    nodes,
  };
};

/**
 * Graph JSON of **IR v2** (docs/ir-v2.md) — **a computation contract** without storage
 * declarations. The container's binding table holds the mapping to the data, and the merge
 * (`format/container/bind.ts`) turns it into an {@link IrGraph}.
 *
 * `initializers[name]` has only two forms: `{}` (has data) or `{ "shared": true }` (borrows the
 * weight of a lender Session — the name MUST equal the lender's initializer name).
 */
export const parseIrDeclaration = (json: string): IrDeclaration =>
  parseIrDeclarationValue(parseJson(json));

/**
 * Reads IR v2 from an already JSON.parsed value (the caller — the descriptor reader — has done the
 * non-finite-number and depth checks).
 */
export const parseIrDeclarationValue = (document: unknown): IrDeclaration =>
  parseIrDocument(document, IR_VERSION, (obj, where): IrDeclaredInitializer => {
    checkKeys(obj, [], ["shared"], where);
    if (!Object.hasOwn(obj, "shared")) return { shared: false };
    // MUST: only `true` may be written (the same rule as `states[].external` — `false` is the same
    // declaration as the field's absence, so allowing both spellings would leave canonical
    // serialization without a single answer).
    if (obj["shared"] !== true) {
      throw new IrError(
        `${where}.shared: ${JSON.stringify(
          obj["shared"],
        )} cannot be written (write true only when it is borrowed)`,
      );
    }
    return { shared: true };
  });

/**
 * **Canonical serialization** of IR v2 (docs/ir-v2.md "Canonical serialization") — the same graph
 * becomes the same bytes (the condition for deciding `krg` identity by content hash — ADR 0108
 * decision 4).
 *
 * - Fixed-schema objects use the key order of the spec's examples; maps keyed by name
 *   (`initializers` / `values` / `states` / a node's `attrs` and `states`) and sets whose order has
 *   no meaning (`requires.ops` / `symbols`) use code point order.
 * - The spelling of numbers is canonical as `JSON.stringify` (= ECMAScript `Number::toString`)
 *   writes it.
 * - Optional sections (`states` / a node's `states` / `external`) are not written when empty or
 *   false.
 *
 * MUST: the return value is for `canonicalJsonText` only — do not spell it with `JSON.stringify`.
 * Name maps are represented as `SortedEntries` (entry lists in code point order), so a plain
 * `JSON.stringify` cannot write the canonical form (`toJSON` of `SortedEntries` throws).
 */
export const canonicalIrDocument = (graph: IrDeclaration): Record<string, unknown> => ({
  format: IR_FORMAT,
  version: IR_VERSION,
  requires: { ops: sortedByCodePoints(graph.requires.ops) },
  symbols: sortedByCodePoints(graph.symbols),
  inputs: graph.inputs.map((input) => ({
    name: input.name,
    dtype: input.dtype,
    shape: [...input.shape],
  })),
  outputs: [...graph.outputs],
  initializers: sortedObject(graph.initializers, (init) => (init.shared ? { shared: true } : {})),
  values: sortedObject(graph.values, (value) => ({ dtype: value.dtype, shape: [...value.shape] })),
  ...(Object.keys(graph.states).length === 0
    ? {}
    : {
        states: sortedObject(graph.states, (slot) => ({
          dtype: slot.dtype,
          shape: [...slot.shape],
          ...(slot.external ? { external: true } : {}),
        })),
      }),
  nodes: graph.nodes.map((node) => ({
    op: node.op,
    ins: [...node.ins],
    outs: [...node.outs],
    attrs: canonicalJsonValue(node.attrs),
    ...(Object.keys(node.states).length === 0 ? {} : { states: sortedObject(node.states) }),
  })),
});

/** The canonically serialized graph JSON string. */
export const canonicalIrJson = (graph: IrDeclaration): string =>
  canonicalJsonText(canonicalIrDocument(graph));

/**
 * Collects the symbol names that appear in a shape (occurrences in dimension positions only — no
 * inference from element counts).
 */
const symbolsIn = (shape: readonly IrDim[], into: Set<string>): void => {
  for (const dim of shape) {
    if (typeof dim !== "string") continue;
    into.add(parseDim(dim).sym);
  }
};

/**
 * A declared symbol MUST **have a binding point**. There are two binding points (ADR 0066,
 * addendum of 2026-08-17, item 7):
 *
 * 1. **A dimension position of an input shape** (solved from the actual size per run —
 *    `bindSymbols` in `runtime/plan.ts`). Derived forms (`2T` / `T+8`) are fine — a linear
 *    expression with one symbol per dimension has a unique solution from the actual size
 *    (ADR 0057). While the bare form was required, a graph with a stride-2 conv on its time axis
 *    (the vowel-detection CRNN) could not declare a symbolic length.
 * 2. **A dimension position of a states shape** (the KV capacity chosen by
 *    `createGenerationContext(spec.bindings)` — a value the user picks when creating the context,
 *    so baking it as an export-time constant contradicts ADR 0066 decision 3 (static physical
 *    storage)).
 *
 * MUST: **a states-only symbol (one that appears only in states) must not appear in a value shape**
 * (item 7). Only input-derived bindings resolve ordinary value shapes, so if it appears, binding
 * always fails at run time — it fails at declaration time instead.
 */
const checkSymbolBindability = (
  symbols: readonly string[],
  inputs: readonly IrInput[],
  states: Readonly<Record<string, IrStateSlot>>,
  values: Readonly<Record<string, IrValueInfo>>,
): void => {
  const fromInputs = new Set<string>();
  for (const input of inputs) symbolsIn(input.shape, fromInputs);
  const fromStates = new Set<string>();
  for (const slot of Object.values(states)) symbolsIn(slot.shape, fromStates);
  for (const symbol of symbols) {
    if (!fromInputs.has(symbol) && !fromStates.has(symbol)) {
      throw new IrError(
        `graph.symbols: '${symbol}' does not appear at a dimension position of an input shape / states shape — it cannot be bound`,
      );
    }
  }
  for (const [name, value] of Object.entries(values)) {
    const used = new Set<string>();
    symbolsIn(value.shape, used);
    for (const symbol of used) {
      if (!fromStates.has(symbol) || fromInputs.has(symbol)) continue;
      throw new IrError(
        `graph.values['${name}']: states-only symbol '${symbol}' appears in a value shape` +
          ` (only bindings from inputs resolve value shapes — ADR 0066 addendum 7)`,
      );
    }
  }
};

/**
 * SSA single assignment + topological order (forward references rejected) + a check that outputs
 * are defined.
 */
const checkDefinitions = (
  inputs: readonly IrInput[],
  initializerNames: readonly string[],
  nodes: readonly IrNode[],
  outputs: readonly string[],
): ReadonlySet<string> => {
  const defined = new Set<string>();
  const define = (name: string, where: string): void => {
    if (defined.has(name)) {
      throw new IrError(
        `${where}: value '${name}' is defined twice (violates SSA single assignment)`,
      );
    }
    defined.add(name);
  };
  for (const input of inputs) define(input.name, "graph.inputs");
  for (const name of initializerNames) define(name, "graph.initializers");
  nodes.forEach((node, index) => {
    const where = `graph.nodes[${index}] (${node.op})`;
    for (const ref of node.ins) {
      // nodes MUST be stored in topological order — a forward reference would make execution
      // order depend on implicit dependency analysis, so it fails here.
      if (!defined.has(ref)) {
        throw new IrError(
          `${where}: input '${ref}' is undefined (forward reference or undeclared)`,
        );
      }
    }
    for (const out of node.outs) define(out, where);
  });
  for (const output of outputs) {
    if (!defined.has(output)) throw new IrError(`graph.outputs: '${output}' is undefined`);
  }
  return defined;
};

/**
 * Declaration completeness: inputs are declared by inputs[], initializers and node outputs by
 * values{}, each exactly once. Orphan declarations (values nobody defines) fail loudly too.
 */
const checkDeclarations = (
  inputs: readonly IrInput[],
  initializerNames: readonly string[],
  values: Readonly<Record<string, IrValueInfo>>,
  nodes: readonly IrNode[],
  defined: ReadonlySet<string>,
): void => {
  const inputNames = new Set(inputs.map((input) => input.name));
  for (const name of Object.keys(values)) {
    if (inputNames.has(name)) {
      throw new IrError(
        `graph.values['${name}']: inputs are declared in inputs[] (double declaration)`,
      );
    }
    if (!defined.has(name)) {
      throw new IrError(`graph.values['${name}']: declaration not defined by any node`);
    }
  }
  for (const name of initializerNames) {
    if (!Object.hasOwn(values, name)) {
      throw new IrError(`graph.initializers['${name}']: no dtype/shape declaration in values`);
    }
    // MUST: spec rules determined by the graph alone are checked in one place, the parser —
    // writing them per load path splits the rules. Pairs of semantic dtype and storage codec belong
    // to the merge layer (`format/container/bind.ts`), and whether a storage can run belongs to the
    // check against the runtime's support table (`ops/support.ts`).
    if (!INITIALIZER_DTYPES.has(values[name].dtype)) {
      throw new IrError(
        `graph.values['${name}']: initializer semantic dtype '${
          values[name].dtype
        }' is outside the vocabulary (f32 / i32 only)`,
      );
    }
    // An initializer must be settled before binding, or it cannot be checked against the data's
    // shape.
    if (values[name].shape.some((dim) => typeof dim !== "number")) {
      throw new IrError(
        `graph.values['${name}']: symbolic dimensions cannot be used in an initializer shape`,
      );
    }
  }
  for (const node of nodes) {
    for (const out of node.outs) {
      if (!Object.hasOwn(values, out)) {
        throw new IrError(`graph.values: no dtype/shape declaration for node output '${out}'`);
      }
    }
  }
};

/**
 * Checks of state slot names. A slot is not a value (it is not referenced by `ins` / `outs`; nodes
 * reference it by name through a separate field — ADR 0066 decision 1, 0067 decision 4), so it is
 * **separate from the value namespace**, but **identical names are rejected**: the same name in
 * separate namespaces only makes "a value name written in a field meant for a slot name / or the
 * reverse" undetectable, and adds no expressiveness. The same practice as 1 block ≤ 1 reference
 * (the container's rule ①).
 *
 * **Reference completeness** (ADR 0067 decisions 4 / 5): a declared slot MUST be referenced from the
 * `states` field of at least one node. A slot nobody references is the same hole as an orphan
 * declaration in values: capacity the GenerationContext allocates but nobody reads (tens of MiB for
 * KV) stays behind silently.
 */
const checkStateSlots = (
  states: Readonly<Record<string, IrStateSlot>>,
  values: Readonly<Record<string, IrValueInfo>>,
  defined: ReadonlySet<string>,
  nodes: readonly IrNode[],
): void => {
  for (const name of Object.keys(states)) {
    if (defined.has(name) || Object.hasOwn(values, name)) {
      throw new IrError(
        `graph.states['${name}']: same name as a value (state slots are a separate namespace from values — the mix-up is rejected)`,
      );
    }
  }
  const referenced = new Set<string>();
  for (const node of nodes) {
    for (const slot of Object.values(node.states)) referenced.add(slot);
  }
  for (const name of Object.keys(states)) {
    if (!referenced.has(name)) {
      throw new IrError(`graph.states['${name}']: declaration not referenced by any node`);
    }
  }
};

/** requires.ops ≡ the set of ops actually used in nodes (the premise of the runtime check). */
const checkRequiredOps = (requiredOps: readonly string[], nodes: readonly IrNode[]): void => {
  const used = new Set(nodes.map((node) => node.op));
  const declared = new Set(requiredOps);
  const missing = [...used].filter((op) => !declared.has(op)).sort();
  const extra = [...declared].filter((op) => !used.has(op)).sort();
  if (missing.length > 0 || extra.length > 0) {
    throw new IrError(
      `graph.requires.ops does not match the set of ops used: undeclared [${missing.join(", ")}] / surplus [${extra.join(
        ", ",
      )}]`,
    );
  }
};

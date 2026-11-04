import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { DimError } from "../src/format/dims.ts";
import { OpContractError } from "../src/ops.ts";
import {
  bindSymbols,
  countUses,
  eligibleCompressedInitializers,
  ExecutionError,
  i2EligibleInitializers,
  i4EligibleInitializers,
  planGraph,
  resolveShape,
  statesOnlySymbols,
  validateGraphContracts,
  weightChannelAxes,
} from "../src/runtime/plan.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";

/** x: [T,4] → matmul(w[4,3]) → h[T,3] → add(b[3]) → y[T,3] */
const linearGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["matmul", "add"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

/**
 * Declaration → merged graph (storage follows the declared dtype; contents are not inspected —
 * helpers/merged-graph.ts).
 */
const parse = mergeGraph;

test("symbol bindings are taken from dimension positions in input shapes", () => {
  const graph = parse(linearGraph());
  assertEquals(bindSymbols(graph, { x: [7, 4] }), { T: 7 });
  assertEquals(bindSymbols(graph, { x: [0, 4] }), { T: 0 });
});

/**
 * A graph whose input time axis has only the derived form `2T` (the shape of the vowel-detection
 * CRNN — ADR 0057).
 */
const derivedGraph = (): DeclarationJson => {
  const graph = linearGraph();
  graph.inputs = [{ name: "x", dtype: "f32", shape: ["2T", 4] }];
  graph.values.h = { dtype: "f32", shape: ["2T", 3] };
  graph.values.y = { dtype: "f32", shape: ["2T", 3] };
  return graph;
};

test("symbols are solved even from inputs with only derived dimensions (the solution is unique — ADR 0057)", () => {
  const graph = parse(derivedGraph());
  assertEquals(bindSymbols(graph, { x: [284, 4] }), { T: 142 });
  assertEquals(bindSymbols(graph, { x: [0, 4] }), { T: 0 });
  // An explicit seed still takes precedence as before, and fails if it disagrees with the solution.
  assertEquals(bindSymbols(graph, { x: [284, 4] }, { T: 142 }), { T: 142 });
  assertThrows(() => bindSymbols(graph, { x: [284, 4] }, { T: 284 }), ExecutionError);
});

test("an actual size that does not have the declared form fails loudly (no rounding)", () => {
  // The vowel-detection CRNN's "the number of 10ms frames is even" fails here (it is not silently
  // rounded to 142 or 143).
  const graph = parse(derivedGraph());
  const error = assertThrows(
    () => bindSymbols(graph, { x: [285, 4] }),
    ExecutionError,
    "does not have the declared form '2T'",
  );
  assert(error.message.includes("actual 285"), error.message);
});

test("every mismatch between bindings, actual shapes and declarations fails loudly", () => {
  const graph = parse(linearGraph());
  // Mismatched numeric dimension
  assertThrows(() => bindSymbols(graph, { x: [7, 5] }), ExecutionError);
  // Mismatched rank
  assertThrows(() => bindSymbols(graph, { x: [7] }), ExecutionError);
  // Missing or extra input
  assertThrows(() => bindSymbols(graph, {}), ExecutionError);
  assertThrows(() => bindSymbols(graph, { x: [7, 4], z: [1] }), ExecutionError);
  // A dimension that is not a non-negative integer
  assertThrows(() => bindSymbols(graph, { x: [-1, 4] }), ExecutionError);
  // An explicit binding conflicts with a binding derived from an input shape
  assertThrows(() => bindSymbols(graph, { x: [7, 4] }, { T: 8 }), ExecutionError);
  // An explicit binding of a symbol not in the declaration
  assertThrows(() => bindSymbols(graph, { x: [7, 4] }, { S: 8 }), ExecutionError);
});

test("an explicit binding is accepted if it matches the input shape", () => {
  const graph = parse(linearGraph());
  assertEquals(bindSymbols(graph, { x: [7, 4] }, { T: 7 }), { T: 7 });
});

test("whether a binding exists is checked with Object.hasOwn (names from the prototype do not count as bound)", () => {
  const source = linearGraph();
  source.symbols = ["toString"];
  source.inputs = [{ name: "x", dtype: "f32", shape: ["toString", 4] }];
  source.values.h = { dtype: "f32", shape: ["toString", 3] };
  source.values.y = { dtype: "f32", shape: ["toString", 3] };
  const graph = parse(source);
  assertEquals(bindSymbols(graph, { x: [6, 4] }), { toString: 6 });
});

test("bindSymbols preserves a binding for a symbol named '__proto__' as an own property", () => {
  const source = linearGraph();
  source.symbols = ["__proto__"];
  source.inputs = [{ name: "x", dtype: "f32", shape: ["__proto__", 4] }];
  source.values.h = { dtype: "f32", shape: ["__proto__", 3] };
  source.values.y = { dtype: "f32", shape: ["__proto__", 3] };
  const graph = parse(source);

  const bindings = bindSymbols(graph, { x: [6, 4] });
  // Some engines create an own property even with `o["__proto__"] = v`, so hasOwn alone cannot
  // detect a regression to a plain `{}`. That the container has a null prototype (= no engine
  // differences brought into the accepted set) is pinned as well.
  assertEquals(Object.getPrototypeOf(bindings), null);
  assertEquals(Object.hasOwn(bindings, "__proto__"), true);
  assertEquals(bindings["__proto__"], 6);
  // The binding survives to later stages (the shape resolves to a number)
  assertEquals(resolveShape(graph.values["y"].shape, bindings), [6, 3]);

  // The same for the explicit binding (seed). MUST: build it with a computed key — a literal
  // `__proto__:` is a [[Prototype]] specification, not an own key, and would not appear in
  // Object.entries, escaping the check.
  const seeded = bindSymbols(graph, { x: [6, 4] }, { ["__proto__"]: 6 });
  assertEquals(Object.getPrototypeOf(seeded), null);
  assertEquals(seeded["__proto__"], 6);
  assertThrows(() => bindSymbols(graph, { x: [6, 4] }, { ["__proto__"]: 5 }), ExecutionError);
});

test("a bare form and a derived form together still give one binding (a mismatch fails loudly)", () => {
  const source = linearGraph();
  // Inputs are declared only in inputs[] (placing them in values{} is a double declaration and the
  // IR fails)
  source.inputs = [
    { name: "x", dtype: "f32", shape: ["T", 4] },
    { name: "m", dtype: "f32", shape: ["2T+1"] },
  ];
  const graph = parse(source);
  assertEquals(bindSymbols(graph, { x: [3, 4], m: [7] }), { T: 3 });
  // The derived side points to a different T (9 → T=4): fails as a conflict.
  assertThrows(() => bindSymbols(graph, { x: [3, 4], m: [9] }), ExecutionError);
  // The derived side does not have the form `2T+1` at all (6 cannot be an odd length).
  assertThrows(() => bindSymbols(graph, { x: [3, 4], m: [6] }), ExecutionError);
});

test("planGraph resolves every value's shape and checks node outputs against the contracts", () => {
  const graph = parse(linearGraph());
  const plan = planGraph(graph, { T: 5 });
  assertEquals(plan.shapes.get("x"), [5, 4]);
  assertEquals(plan.shapes.get("w"), [4, 3]);
  assertEquals(plan.shapes.get("h"), [5, 3]);
  assertEquals(plan.nodes.length, 2);
  assertEquals(plan.nodes[0].outputs, [{ name: "h", shape: [5, 3], dtype: "f32" }]);
  assertEquals(plan.nodes[1].contract.kind, "binary");
});

test("a mismatch between declared and computed shapes, and unbound symbols, fail loudly", () => {
  const source = linearGraph();
  source.values.h = { dtype: "f32", shape: ["T", 4] };
  assertThrows(() => planGraph(parse(source), { T: 5 }), ExecutionError);
  assertThrows(() => planGraph(parse(linearGraph()), {}), DimError);
  assertThrows(() => resolveShape(["T"], {}), DimError);
  assertEquals(resolveShape([2, "T+1"], { T: 3 }), [2, 4]);
});

/**
 * A shape with more than 2^53 elements is rejected at planning time. Letting it through would make
 * the plain product in `numel` return a rounded value, and the allocation byte counts and dispatch
 * counts would be built from that rounded value (reading and writing outside the declared range
 * without a single exception).
 */
test("planGraph fails loudly on a shape whose element count exceeds the safe integer range (no rounding)", () => {
  const graph = parse(linearGraph());
  // T·4 = 2^54 (only one binding is large; each dimension stays a safe integer).
  assertThrows(
    () => planGraph(graph, { T: 2 ** 52 }),
    ExecutionError,
    "Value 'x': the element count of shape [4503599627370496,4] exceeds the safe integer range",
  );
  // Control: on the same graph, a binding whose product fits in the safe integer range passes
  // (2^50·4 = 2^52).
  assertEquals(planGraph(graph, { T: 2 ** 50 }).shapes.get("x"), [2 ** 50, 4]);
});

test("useCounts is the exact total count over node.ins (taking the same value twice gives 2)", () => {
  const source = linearGraph();
  source.requires.ops = ["matmul", "add", "mul"];
  source.nodes.push({ op: "mul", ins: ["y", "y"], outs: ["z"], attrs: {} });
  source.values.z = { dtype: "f32", shape: ["T", 3] };
  source.outputs = ["z"];
  const counts = countUses(parse(source));
  assertEquals(counts.get("y"), 2);
  assertEquals(counts.get("x"), 1);
  assertEquals(counts.get("z"), undefined);
});

test("the contract check rejects unsupported ops, non-empty attrs and non-f32 dtypes at build time", () => {
  assertEquals(validateGraphContracts(parse(linearGraph())), undefined);

  const foreignOp = linearGraph();
  foreignOp.requires.ops = ["matmul", "softmax"];
  foreignOp.nodes[1] = { op: "softmax", ins: ["h"], outs: ["y"], attrs: {} };
  assertThrows(() => validateGraphContracts(parse(foreignOp)), OpContractError);

  const withAttrs = linearGraph();
  withAttrs.nodes[1].attrs = { alpha: 1 };
  assertThrows(() => validateGraphContracts(parse(withAttrs)), OpContractError);

  const intInput = linearGraph();
  intInput.inputs = [{ name: "x", dtype: "i32", shape: ["T", 4] }];
  assertThrows(() => validateGraphContracts(parse(intInput)), OpContractError);
});

/** table[6,5] (an i32 constant of the Tmax shape) → sym_prefix_slice → y[T,T]. */
const prefixSliceGraph = (overrides: Partial<DeclarationJson> = {}): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["sym_prefix_slice"] },
  symbols: ["T"],
  inputs: [{ name: "bind", dtype: "f32", shape: ["T"] }],
  outputs: ["y"],
  initializers: { table: {} },
  values: {
    table: { dtype: "i32", shape: [6, 5] },
    y: { dtype: "i32", shape: ["T", "T"] },
  },
  nodes: [
    {
      op: "sym_prefix_slice",
      ins: ["table"],
      outs: ["y"],
      attrs: {
        sym: "T",
        slices: [
          { dim: 0, coeff: 1, offset: 0 },
          { dim: 1, coeff: 1, offset: 0 },
        ],
      },
    },
  ],
  ...overrides,
});

// ADR 0010: sym_prefix_slice's contracts that "cannot be decided without looking at the whole
// graph" are rejected before binding.
test("sym_prefix_slice's graph-context contracts fail at Session build", () => {
  validateGraphContracts(parse(prefixSliceGraph()));

  // sym is not in graph.symbols (no binding can be taken at run time, so the prefix length is
  // undetermined)
  const foreignSym = prefixSliceGraph();
  foreignSym.nodes[0].attrs = {
    sym: "U",
    slices: [{ dim: 0, coeff: 1, offset: 0 }],
  };
  assertThrows(() => validateGraphContracts(parse(foreignSym)), ExecutionError);

  // dim is outside the input rank
  const badDim = prefixSliceGraph();
  badDim.nodes[0].attrs = { sym: "T", slices: [{ dim: 2, coeff: 1, offset: 0 }] };
  assertThrows(() => validateGraphContracts(parse(badDim)), ExecutionError);

  // MUST: an input of symbolic shape cannot be told apart from the numbers after binding (it would
  // match on a run with T = Tmax). It can only be detected here, where the declared form is
  // visible.
  const symbolicSource = prefixSliceGraph();
  symbolicSource.initializers = {};
  symbolicSource.inputs = [
    { name: "bind", dtype: "f32", shape: ["T"] },
    { name: "table", dtype: "i32", shape: ["T", 5] },
  ];
  delete symbolicSource.values["table"];
  assertThrows(() => validateGraphContracts(parse(symbolicSource)), ExecutionError);
});

// The prefix length is computed from attrs (+ bindings) and checked against the declaration (unlike
// reshape, where the declaration is the target shape).
test("sym_prefix_slice's output shape is computed from the bindings and checked against the declaration", () => {
  const graph = parse(prefixSliceGraph());
  const plan = planGraph(graph, bindSymbols(graph, { bind: [3] }));
  assertEquals(plan.nodes[0].outputs[0].shape, [3, 3]);

  // Exceeding Tmax (a read outside the constant buffer) fails at binding time
  assertThrows(() => planGraph(graph, bindSymbols(graph, { bind: [6] })), OpContractError);
});

// ADR 0014: the target axis of slice / flip is **static at the declaration level** (slicing a
// symbolic axis is sym_prefix_slice's job). Only cat's concatenation axis was relaxed by ADR 0046
// to "a linear sum of the same symbol". Neither can be told apart in the numeric shapes after
// binding (different symbols can resolve to the same value), so they can only be detected by
// validateGraphContracts, which sees the declared form.
const layoutAxisGraph = (
  op: string,
  attrs: Record<string, unknown>,
  ins: readonly { name: string; shape: (number | string)[] }[],
  outShape: (number | string)[],
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [op] },
  // Only input dimensions that appear in bare form ('T' and the like) are binding sources (a symbol
  // with only derived forms cannot be declared)
  symbols: [...new Set(ins.flatMap((spec) => spec.shape.filter((dim) => typeof dim === "string")))],
  inputs: ins.map((spec) => ({ name: spec.name, dtype: "f32", shape: [...spec.shape] })),
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [...outShape] } },
  nodes: [{ op, ins: ins.map((spec) => spec.name), outs: ["y"], attrs }],
});

test("symbolic axes of slice / flip fail at Session build (static axes only)", () => {
  // Accepted: the target axis is static; other axes being symbolic is fine
  validateGraphContracts(
    parse(
      layoutAxisGraph(
        "slice",
        { dim: 1, start: 1, end: 3 },
        [
          {
            name: "x",
            shape: ["T", 4],
          },
        ],
        ["T", 2],
      ),
    ),
  );
  validateGraphContracts(
    parse(layoutAxisGraph("flip", { dim: 1 }, [{ name: "x", shape: ["T", 4] }], ["T", 4])),
  );

  // Rejected: the target axis itself is symbolic
  assertThrows(
    () =>
      validateGraphContracts(
        parse(
          layoutAxisGraph(
            "slice",
            { dim: 0, start: 0, end: 2 },
            [
              {
                name: "x",
                shape: ["T", 4],
              },
            ],
            [2, 4],
          ),
        ),
      ),
    ExecutionError,
  );
  assertThrows(
    () =>
      validateGraphContracts(
        parse(layoutAxisGraph("flip", { dim: 0 }, [{ name: "x", shape: ["T", 4] }], ["T", 4])),
      ),
    ExecutionError,
  );
});

// ADR 0046 revised ADR 0014's "the concatenation axis is static". DiT's joint attention is
// `cat([self | context], dim=1)` with a symbolic axis length on the self side (`S+1519`) — one
// symbol + a constant fits the dimension language `coeff·sym+offset` as is, so the old rule's
// reason for rejection (the sum does not fit) no longer holds. The only remaining rejection is
// **mixing different symbols**.
test("cat's concatenation axis accepts up to a linear sum of the same symbol and fails on mixed symbols", () => {
  // Accepted: the concatenation axis is static (other axes being symbolic is fine)
  validateGraphContracts(
    parse(
      layoutAxisGraph(
        "cat",
        { dim: 1 },
        [
          { name: "a", shape: ["T", 2] },
          { name: "b", shape: ["T", 3] },
        ],
        ["T", 5],
      ),
    ),
  );
  // Accepted: symbol + constant (same type as `S+1519`)
  validateGraphContracts(
    parse(
      layoutAxisGraph(
        "cat",
        { dim: 0 },
        [
          { name: "a", shape: ["T", 2] },
          { name: "b", shape: [3, 2] },
        ],
        ["T+3", 2],
      ),
    ),
  );
  // Accepted: the same symbol together (the coefficients add up)
  validateGraphContracts(
    parse(
      layoutAxisGraph(
        "cat",
        { dim: 0 },
        [
          { name: "a", shape: ["T", 2] },
          { name: "b", shape: ["T", 2] },
        ],
        ["2T", 2],
      ),
    ),
  );

  // Rejected: mixing different symbols (the sum does not fit the grammar of one symbol per
  // dimension)
  assertThrows(
    () =>
      validateGraphContracts(
        parse(
          layoutAxisGraph(
            "cat",
            { dim: 0 },
            [
              { name: "a", shape: ["T", 2] },
              { name: "b", shape: ["U", 2] },
            ],
            ["T", 2],
          ),
        ),
      ),
    ExecutionError,
    "different symbols",
  );
});

// cat's arity has a **lower bound** of 2 (variadic — ADR 0014). A single input is an identity copy
// and is not in the vocabulary.
test("cat takes 2 or more inputs, and 3 or more also pass the contract check", () => {
  validateGraphContracts(
    parse(
      layoutAxisGraph(
        "cat",
        { dim: 1 },
        [
          { name: "a", shape: [2, 1] },
          { name: "b", shape: [2, 2] },
          { name: "c", shape: [2, 3] },
        ],
        [2, 6],
      ),
    ),
  );
  assertThrows(
    () =>
      validateGraphContracts(
        parse(layoutAxisGraph("cat", { dim: 1 }, [{ name: "a", shape: [2, 1] }], [2, 1])),
      ),
    OpContractError,
  );
});

// ---- State-referencing nodes (ADR 0067 decisions 4 / 5 / 5b) ----------------------------

/**
 * One attention reading the `kv` slot and one `state_append` writing it (the issue order is
 * decision 5b's "all readers → append"). `extra` replaces only the node list to build mutations for
 * the order checks.
 */
const stateGraph = (
  nodes?: DeclarationJson["nodes"],
  states?: DeclarationJson["states"],
): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 8, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 2, "M", 4] },
    { name: "v", dtype: "f32", shape: [1, 2, "M", 4] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 8, "M", 4] } },
  states: states ?? {
    "kv.k": { dtype: "f32", shape: [1, 2, "C", 4] },
    "kv.v": { dtype: "f32", shape: [1, 2, "C", 4] },
  },
  nodes: nodes ?? [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5 },
      states: { k: "kv.k", v: "kv.v" },
    },
    { op: "state_append", ins: ["k"], outs: [], attrs: {}, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: {}, states: { slot: "kv.v" } },
  ],
});

/**
 * The capacity symbol `C` does not appear in the inputs (its binding point is
 * createGenerationContext — ADR 0066 addendum 7).
 */
const stateShapes = (capacity = 16): ReadonlyMap<string, readonly number[]> =>
  new Map([
    ["kv.k", [1, 2, capacity, 4]],
    ["kv.v", [1, 2, capacity, 4]],
  ]);

test("a state-referencing graph passes the contract check and can be planned when slot shapes are passed", () => {
  const graph = parse(stateGraph());
  validateGraphContracts(graph);

  const plan = planGraph(graph, { M: 3, C: 16 }, stateShapes());
  assertEquals(
    plan.nodes[0].outputs.map((out) => out.name),
    ["o"],
  );
  assertEquals(plan.nodes[0].outputs[0].shape, [1, 8, 3, 4]);
  // An effect op has an empty output list (`outs` having 0 entries shows up in the plan as is)
  assertEquals(plan.nodes[1].outputs, []);
  assertEquals(plan.nodes[2].outputs, []);
});

test("planning without slot shapes fails loudly on state-referencing nodes", () => {
  const graph = parse(stateGraph());
  const error = assertThrows(
    () => planGraph(graph, { M: 3, C: 16 }),
    OpContractError,
    "GenerationContext",
  );
  assert(error.message.includes("kv.k"), error.message);
});

// ADR 0067 decision 5b: state references draw no data edges, so the order contract is the nodes
// array order itself.
test("at most one state_append per step to the same slot", () => {
  const nodes = stateGraph().nodes;
  assertThrows(
    () =>
      validateGraphContracts(
        parse(
          stateGraph([
            nodes[0],
            nodes[1],
            { op: "state_append", ins: ["v"], outs: [], attrs: {}, states: { slot: "kv.k" } },
            nodes[2],
          ]),
        ),
      ),
    ExecutionError,
    "state_append appears 2 times",
  );
});

// A slot with zero appends is the form "reads uninitialized (zero) rows as the past and advances
// only pastLength", where the error shows only in values (it can only be stopped here, before
// execution). KV sharing (ADR 0067 decision 5 — several readers share one append per slot) must be
// compatible with this check.
test("a slot that state is read from needs exactly one terminating state_append", () => {
  const base = stateGraph();
  const reader = base.nodes[0];
  const error = assertThrows(
    () =>
      validateGraphContracts(
        parse({
          ...base,
          requires: { ops: ["attention"] },
          nodes: [reader],
        }),
      ),
    ExecutionError,
    "no state_append at all",
  );
  assert(error.message.includes("kv.k"), error.message);

  // The form where two readers share the same slot (KV-sharing layers) passes if each slot has one
  // terminating append
  validateGraphContracts(
    parse({
      ...base,
      outputs: ["o", "o2"],
      values: { ...base.values, o2: { dtype: "f32", shape: [1, 8, "M", 4] } },
      nodes: [reader, { ...reader, outs: ["o2"] }, base.nodes[1], base.nodes[2]],
    }),
  );
});

test("a reader of a slot cannot come after that slot's state_append", () => {
  const nodes = stateGraph().nodes;
  const error = assertThrows(
    () => validateGraphContracts(parse(stateGraph([nodes[1], nodes[2], nodes[0]]))),
    ExecutionError,
    "comes the reader",
  );
  // "A reader after the append" concerns only **the order of nodes touching that slot** (nodes of
  // other slots in between do not change the order)
  assert(error.message.includes("kv.k") || error.message.includes("kv.v"), error.message);
});

test("nodes touching the same slot agree on whether window exists and on its value", () => {
  const nodes = stateGraph().nodes;
  const windowed = (window: number | undefined): DeclarationJson["nodes"] => [
    {
      ...nodes[0],
      attrs: window === undefined ? { scale: 0.5 } : { scale: 0.5, window },
    },
    { op: "state_append", ins: ["k"], outs: [], attrs: { window: 8 }, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: { window: 8 }, states: { slot: "kv.v" } },
  ];
  // A form where only the reader lacks a window declaration (= scans the full context) is a silent
  // misread
  assertThrows(
    () => validateGraphContracts(parse(stateGraph(windowed(undefined)))),
    ExecutionError,
    "attrs.window disagrees",
  );
  // Likewise for a different value
  assertThrows(
    () => validateGraphContracts(parse(stateGraph(windowed(4)))),
    ExecutionError,
    "attrs.window disagrees",
  );
  // Passes when they agree
  validateGraphContracts(parse(stateGraph(windowed(8))));
});

test("a sliding window cannot exceed the slot capacity (on both the read and write sides)", () => {
  const nodes = stateGraph().nodes;
  const windowed: DeclarationJson["nodes"] = [
    { ...nodes[0], attrs: { scale: 0.5, window: 32 } },
    { op: "state_append", ins: ["k"], outs: [], attrs: { window: 32 }, states: { slot: "kv.k" } },
    { op: "state_append", ins: ["v"], outs: [], attrs: { window: 32 }, states: { slot: "kv.v" } },
  ];
  const graph = parse(stateGraph(windowed));
  validateGraphContracts(graph);
  // Passes with capacity 32 and fails with 16 (the declaration alone does not decide it = the
  // context decides the capacity)
  planGraph(graph, { M: 3, C: 32 }, stateShapes(32));
  assertThrows(
    () => planGraph(graph, { M: 3, C: 16 }, stateShapes(16)),
    OpContractError,
    "exceeds the slot capacity 16",
  );
});

// ADR 0066 addendum 7's "division of where it applies": the binding point of states-only symbols
// (the KV capacity `C`) is createGenerationContext alone, and run / enqueue's bindSymbols neither
// requires nor accepts them.
test("bindSymbols does not require states-only symbols and rejects them if they arrive in the seed", () => {
  const graph = parse(stateGraph());
  assertEquals([...statesOnlySymbols(graph)], ["C"]);
  // Bindings close over input shapes alone (no C does not mean "could not bind")
  const inputs = { q: [1, 8, 3, 4], k: [1, 2, 3, 4], v: [1, 2, 3, 4] };
  const bindings = bindSymbols(graph, inputs);
  assertEquals(bindings, { M: 3 });
  assertEquals(Object.hasOwn(bindings, "C"), false);
  // Passing it in the seed fails loudly (double bookkeeping with the capacity at context creation —
  // a mismatch would not show in values; only the plan cache key would quietly split)
  assertThrows(() => bindSymbols(graph, inputs, { C: 16 }), ExecutionError, "states-only symbol");
  // A symbol that also appears in the inputs works as before (a form that **also** appears in
  // states is not "only")
  assertThrows(
    () => bindSymbols(graph, { q: [1, 8, 3, 4], k: [1, 2, 5, 4], v: [1, 2, 3, 4] }),
    ExecutionError,
    "conflicting bindings",
  );
  // For a graph without states it is the empty set (the behavior does not change by a single byte)
  assertEquals(statesOnlySymbols(parse(linearGraph())).size, 0);
});

// ADR 0097: for i2, only linear / embedding weights can run while still packed. If another weight
// slot (conv1d and the like) also consumes the same initializer, that kernel reads the packed 2-bit
// bytes as f32, so it must leave the eligible set and fall back to the expanded seat.
test("i2's eligible set is weights consumed only by linear / embedding, and sharing with conv1d excludes it", () => {
  const source: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear", "embedding", "conv1d"] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: [2, 4] },
      { name: "ids", dtype: "i32", shape: [2] },
      { name: "s", dtype: "f32", shape: [1, 4, 8] },
    ],
    outputs: ["hl", "he", "hc", "hs", "hsc"],
    initializers: { wl: {}, we: {}, wc: {}, ws: {}, bl: {}, bc: {} },
    values: {
      wl: { dtype: "f32", shape: [3, 4] },
      we: { dtype: "f32", shape: [5, 4] },
      wc: { dtype: "f32", shape: [3, 4, 1] },
      ws: { dtype: "f32", shape: [3, 4] },
      bl: { dtype: "f32", shape: [3] },
      bc: { dtype: "f32", shape: [3] },
      hl: { dtype: "f32", shape: [2, 3] },
      he: { dtype: "f32", shape: [2, 4] },
      hc: { dtype: "f32", shape: [1, 3, 8] },
      hs: { dtype: "f32", shape: [2, 3] },
      hsc: { dtype: "f32", shape: [1, 3, 8] },
    },
    nodes: [
      { op: "linear", ins: ["x", "wl", "bl"], outs: ["hl"], attrs: {} },
      { op: "embedding", ins: ["we", "ids"], outs: ["he"], attrs: { padding_idx: -1 } },
      { op: "conv1d", ins: ["s", "wc", "bc"], outs: ["hc"], attrs: {} },
      // The predicate looks only at the op consuming the weight slot (not the shape contract), so
      // the shared form is built simply by also passing the linear weight to conv1d's weight slot.
      { op: "linear", ins: ["x", "ws", "bl"], outs: ["hs"], attrs: {} },
      { op: "conv1d", ins: ["s", "ws", "bc"], outs: ["hsc"], attrs: {} },
    ],
  };
  assertEquals([...i2EligibleInitializers(parse(source))].sort(), ["we", "wl"]);
});

/**
 * One unbatched conv3d node (x[3,4,6,6] * w[6,3/groups,3,3,3] → y[6,2,6,6] — ADR 0118 decision 1).
 */
const conv3dGraph = (groups: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv3d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [3, 4, 6, 6] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [6, 3 / groups, 3, 3, 3] },
    b: { dtype: "f32", shape: [6] },
    y: { dtype: "f32", shape: [6, 2, 6, 6] },
  },
  nodes: [
    {
      op: "conv3d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: [1, 1, 1], padding: [0, 1, 1], dilation: [1, 1, 1], groups },
    },
  ],
});

// The semantics (contract, CPU reference) accept groups in general, but the subset implemented on
// the GPU is groups == 1 only (ADR 0118 decision 1). MUST: reject at Session build (= planning
// time) — do not pass the contract check and carry it into the derivation phase of the first run.
test("conv3d with groups > 1 fails loudly at planning time (the semantics pass; the GPU subset rejects it)", () => {
  const grouped = parse(conv3dGraph(3));
  // The contract and shapes (semantics) accept groups in general
  assertEquals(planGraph(grouped, {}).nodes[0].outputs[0].shape, [6, 2, 6, 6]);
  assertThrows(
    () => validateGraphContracts(grouped),
    ExecutionError,
    "conv3d groups 3 cannot run on the GPU",
  );
  // groups == 1 passes
  validateGraphContracts(parse(conv3dGraph(1)));
});

// conv3d's weights are eligible for f16 / i8 (weight slot 1, channel axis 0 — the output channels),
// and bias is not eligible. There is no i4 expansion path (the supplement to ADR 0069 decision 5
// goes only as far as conv1d).
test("conv3d's weights [Cout,Cin,Kt,Kh,Kw] are eligible for compressed storage on axis 0 and not for i4 / i2", () => {
  const graph = parse(conv3dGraph(1));
  assertEquals([...eligibleCompressedInitializers(graph)], ["w"]);
  assertEquals(weightChannelAxes(graph).get("w"), 0);
  assertEquals([...i4EligibleInitializers(graph)], []);
  assertEquals([...i2EligibleInitializers(graph)], []);
});

/**
 * Execution plan (pure functions): symbol binding → shape resolution of every value → per-node
 * output-shape evaluation and comparison with the declarations. It touches no GPU, so it can be
 * verified in environments without an adapter.
 *
 * Static shapes (ADR 0004): every node's output shape is fixed before a single dispatch is issued.
 * Only graphs that pass here proceed to the executor's encode step.
 */

import { evalDim, parseDim, solveDim } from "../format/dims.ts";
import type { IrDim, IrDtype, IrGraph, IrNode } from "../format/ir.ts";
import {
  assertNodeContract,
  catDim,
  computeOutputShape,
  CONV1D_OP,
  conv1dAttrs,
  conv3dAttrs,
  EMBEDDING_OP,
  flipDim,
  IO_DTYPES,
  LINEAR_OP,
  type OpContract,
  resolveNodeDtypes,
  sliceAttrs,
  STATE_APPEND_OP,
  stateReadonly,
  stateWindow,
  symPrefixSliceAttrs,
  WEIGHT_CHANNEL_AXES,
  WEIGHT_SLOTS,
} from "../ops.ts";

/**
 * A contract, binding or declaration was violated at run time (always thrown — never approximated
 * silently).
 */
export class ExecutionError extends Error {
  override readonly name = "ExecutionError";
}

/**
 * Spells an out-of-type value in an input-boundary diagnostic (a number as its value, a string as a
 * JSON quote, anything else as its `typeof` type name).
 *
 * MUST: do not call user conversions (`toString` / `Symbol.toPrimitive` / `toJSON`). Spelling with
 * `String(x)` or a template literal lets the user's exception escape instead of `ExecutionError`
 * for objects whose conversion throws, and for a symbol the template literal itself throws
 * `TypeError`.
 */
export const describeInputValue = (value: unknown): string =>
  typeof value === "number"
    ? String(value)
    : typeof value === "string"
      ? JSON.stringify(value)
      : typeof value;

/** Symbolic dimension → concrete value at run time. */
export type SymbolBindings = Readonly<Record<string, number>>;

/**
 * The plan of one node output (listed in {@link NodePlan.outputs} in **ascending output slot
 * order**).
 */
export type NodeOutputPlan = {
  readonly name: string;
  readonly shape: readonly number[];
  /**
   * The output's semantic dtype (derived from the contract and checked against the declaration —
   * {@link resolveNodeDtypes}).
   */
  readonly dtype: IrDtype;
};

export type NodePlan = {
  readonly node: IrNode;
  readonly contract: OpContract;
  readonly inputShapes: readonly (readonly number[])[];
  /**
   * The inputs' semantic dtypes (from the declarations). The kernel's element type follows from
   * these.
   */
  readonly inputDtypes: readonly IrDtype[];
  /**
   * The plan in ascending output slot order (same order and length as `node.outs` — ADR 0068
   * decision 1).
   *
   * MUST: do not hold names, shapes and dtypes in separate lists. Three parallel arrays cannot
   * express the invariant "the lengths match" in the type, and reading with a shifted slot slips
   * past the shape check.
   */
  readonly outputs: readonly NodeOutputPlan[];
};

export type GraphPlan = {
  readonly bindings: SymbolBindings;
  /** Resolved shapes of the inputs, initializers and every node output. */
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly nodes: readonly NodePlan[];
};

/** Concretises a declared shape with the bindings. dims.ts fails an unbound symbol with DimError. */
export const resolveShape = (shape: readonly IrDim[], bindings: SymbolBindings): number[] =>
  shape.map((dim) => (typeof dim === "number" ? dim : evalDim(parseDim(dim), bindings)));

/**
 * Value name → declaration (inputs in inputs[], everything else in values{} — the IR guarantees
 * exactly one place).
 */
const declarationOf = (
  graph: IrGraph,
  name: string,
): { readonly dtype: IrDtype; readonly shape: readonly IrDim[] } => {
  const input = graph.inputs.find((spec) => spec.name === name);
  if (input !== undefined) return { dtype: input.dtype, shape: input.shape };
  if (!Object.hasOwn(graph.values, name)) {
    throw new ExecutionError(`Value '${name}' has no declaration`);
  }
  const value = graph.values[name];
  return { dtype: value.dtype, shape: value.shape };
};

/**
 * Collects a node's input and output dtypes from the declarations, checks them against the contract
 * and returns **the dtypes in output slot order**.
 */
const nodeDtypes = (
  graph: IrGraph,
  node: IrNode,
  contract: OpContract,
  where: string,
): { readonly inputs: readonly IrDtype[]; readonly outputs: readonly IrDtype[] } => {
  const inputs = node.ins.map((name) => declarationOf(graph, name).dtype);
  const declaredOutputs = node.outs.map((name) => declarationOf(graph, name).dtype);
  return {
    inputs,
    outputs: resolveNodeDtypes(contract, node, inputs, declaredOutputs, where),
  };
};

/**
 * Binding-independent contract checks (once, at Session construction). Checks arity, attrs and the
 * dtype rules.
 * MUST: fail unsupported dtypes here — kernels only have the element types enabled per op, and
 * dispatching a value of another type yields wrong values from reinterpreted bits, with no exception.
 */
export const validateGraphContracts = (graph: IrGraph): void => {
  graph.nodes.forEach((node, index) => {
    const where = `nodes[${index}] (${node.op})`;
    const contract = assertNodeContract(node, where);
    nodeDtypes(graph, node, contract, where);
    if (contract.kind === "symPrefixSlice") assertSymPrefixSlice(graph, node, where);
    if (contract.kind === "slice" || contract.kind === "flip") {
      assertStaticLayoutAxis(graph, node, contract.kind, where);
    }
    if (contract.kind === "cat") assertCatAxis(graph, node, where);
    if (contract.kind === "conv3d") assertConv3dSubset(node, where);
  });
  assertStateOwnership(graph);
  assertStateOrder(graph);
  for (const spec of graph.inputs) {
    if (!IO_DTYPES.includes(spec.dtype)) {
      throw new ExecutionError(
        `Input '${spec.name}': semantic dtype '${spec.dtype}' cannot be transferred (${IO_DTYPES.join(
          " / ",
        )} only)`,
      );
    }
  }
};

/**
 * The check for conv3d's **implemented GPU subset** (ADR 0118 decision 1 — ADR 0064 axis A):
 * `groups == 1` only.
 *
 * The semantics (contract and CPU reference) accept general groups, but the GPU has only the
 * implicit GEMM (groups == 1). MUST: fail at Session construction (= plan time) — deferring to the
 * derivation phase lets a Session that was built without ever running fail only at its first run
 * (the culprit op and attrs are the same either way, but there is no reason to delay a fact that is
 * known when the asset is opened).
 * MUST: do not fill `groups` with a default (`conv3dAttrs` fails a missing one — ADR 0015).
 */
const assertConv3dSubset = (node: IrNode, where: string): void => {
  const { groups } = conv3dAttrs(node.attrs, where);
  if (groups !== 1) {
    throw new ExecutionError(
      `${where}: conv3d groups ${groups} cannot run on the GPU (the implemented subset is the groups == 1` +
        " implicit GEMM only — ADR 0118 decision 1; the CPU reference holds the semantics)",
    );
  }
};

/**
 * Ownership rules of borrowed slots (`states[].external` — ADR 0096, addendum of 2026-09-08 on
 * stage 2). They depend on neither bindings nor the GPU, so **four points** are checked once at
 * Session construction:
 *
 * 1. **Zero** `state_append` to an external slot (the backing belongs to the lending context and
 *    only the lender writes). Letting a write through makes the borrower overwrite past KV the
 *    lender has committed
 * 2. The readers of an external slot are **readonly attention only** (the form that adds this
 *    step's k/v assumes "reading rows it wrote itself", so on a slot without a writer it always
 *    reads uninitialised rows)
 * 3. A slot referenced by a readonly reader is **external only** (the dual of 2 — reading an own
 *    slot readonly without an append would be a back door around the missing-writer check
 *    ({@link assertStateOrder}))
 * 4. If any slot is external, **every slot is external** (the stage 2 borrowing context has no own
 *    slots, so a mix would mean "the allocating side and the borrowing side live in one context")
 *
 * MUST: fail loudly. A violation of any of the four shows up not as an exception but as **different
 * values** (an uninitialised past / overwritten KV), so it can only be stopped here, before
 * execution.
 */
const assertStateOwnership = (graph: IrGraph): void => {
  const names = Object.keys(graph.states);
  const external = names.filter((name) => graph.states[name].external);
  graph.nodes.forEach((node, index) => {
    const slots = Object.values(node.states);
    if (slots.length === 0) return;
    const where = `nodes[${index}] (${node.op})`;
    const readonly = stateReadonly(node.attrs, where);
    for (const slot of slots) {
      const isExternal = graph.states[slot].external;
      if (readonly && !isExternal) {
        throw new ExecutionError(
          `${where}: a readonly attention reads the non-external state slot '${slot}'` +
            " (readonly is only for slots of the context borrowed from — ADR 0096 stage 2 §1.1)",
        );
      }
      if (readonly || !isExternal) continue;
      throw new ExecutionError(
        node.op === STATE_APPEND_OP
          ? `State slot '${slot}': external, yet ${STATE_APPEND_OP} (${where}) is present` +
              " (only the lender may write to a borrowed slot — ADR 0096 stage 2 §1.1)"
          : `${where}: external state slot '${slot}' is read by the non-readonly ${node.op}` +
              " (the only readers of an external slot are attentions with attrs.readonly — ADR 0096 stage 2 §1.1)",
      );
    }
  });
  if (external.length === 0 || external.length === names.length) return;
  throw new ExecutionError(
    `State slots mix external and non-external slots (external: [${external.join(
      ", ",
    )}] / own: [${names
      .filter((name) => !graph.states[name].external)
      .join(", ")}]) — a borrowing context has no slots of its own (ADR 0096 stage 2 §1.1)`,
  );
};

/** The record of one node that touched a slot ({@link assertStateOrder}). */
type StateTouch = {
  readonly index: number;
  readonly op: string;
  readonly appends: boolean;
  readonly window: number | undefined;
};

/**
 * Order of state effects (ADR 0067 decision 5b, item 2). State references **draw no tensor data
 * edges**, so the DAG's topological order does not fix the order; the contract is the `nodes`
 * **array order** itself. It does not depend on bindings, so three points per slot are checked
 * once at Session construction:
 *
 * 1. **Exactly one** `state_append` per slot. Two or more write twice in one step: the ring
 *    position formula advances twice and the past that readers see changes mid-step. Zero means
 *    nobody writes the slot while `advance` still moves the logical length: readers read
 *    uninitialised (zero) rows as the past and every step from the second on silently returns wrong
 *    values. KV-sharing layers (ADR 0067 decision 5's "layers without a `state_append` node")
 *    **share one append per slot among several readers**, so they are compatible with this per-slot
 *    check — only a slot with no writer at all is rejected
 * 2. If an append exists, it is **the last node touching that slot** (a reader after the append
 *    "reads this step's k/v twice, as the past" — closes the high findings of review rounds 3 / 4)
 * 3. Every node touching the same slot agrees on `window` **in presence and value** (logical col →
 *    physical row MUST use the same formula for reads and writes — ADR 0067 decision 4. A different
 *    formula on the read side alone is a silent misread)
 *
 * MUST: fail loudly. In all three, an "order / declaration error" shows up not as an exception but
 * as **different values**, so it can only be stopped here, before execution.
 */
const assertStateOrder = (graph: IrGraph): void => {
  const externalSlots = new Set(
    Object.keys(graph.states).filter((name) => graph.states[name].external),
  );
  const touches = new Map<string, StateTouch[]>();
  graph.nodes.forEach((node, index) => {
    const slots = Object.values(node.states);
    if (slots.length === 0) return;
    // assertNodeContract has already checked the attrs' ranges (this only reads them again).
    const window = stateWindow(node.attrs, `nodes[${index}] (${node.op})`);
    for (const slot of slots) {
      const list = touches.get(slot) ?? [];
      list.push({ index, op: node.op, appends: node.op === STATE_APPEND_OP, window });
      touches.set(slot, list);
    }
  });
  for (const [slot, list] of touches) {
    const appends = list.filter((touch) => touch.appends);
    // The writer of a borrowed (external) slot lives on the lending context's side. Zero appends is
    // **valid**, so the three branches below (exactly one / terminal) are skipped and only window
    // agreement is checked (the same read/write formula must hold across lender and borrower — a
    // wider borrower window reads outside the window).
    if (externalSlots.has(slot)) {
      assertSameWindow(slot, list);
      continue;
    }
    if (appends.length > 1) {
      throw new ExecutionError(
        `State slot '${slot}': ${STATE_APPEND_OP} appears ${appends.length} times (nodes[${appends
          .map((touch) => touch.index)
          .join("], nodes[")}]) — at most once per step (ADR 0067 decision 5b)`,
      );
    }
    if (appends.length === 0) {
      throw new ExecutionError(
        `State slot '${slot}': no ${STATE_APPEND_OP} at all (readers nodes[${list
          .map((touch) => touch.index)
          .join(
            "], nodes[",
          )}] only) — a slot with readers MUST have exactly one terminal ${STATE_APPEND_OP}` +
          ` (ADR 0067 decision 5b; without a writer the readers read an uninitialised past and only the logical length advances)`,
      );
    }
    // Here there is exactly one append (the two branches above failed zero and two or more).
    const last = list[list.length - 1];
    if (!last.appends) {
      throw new ExecutionError(
        `State slot '${slot}': after ${STATE_APPEND_OP} (nodes[${
          appends[0].index
        }]) comes the reader nodes[${last.index}] (${last.op})` +
          ` (the append MUST be the last node touching the slot — ADR 0067 decision 5b)`,
      );
    }
    assertSameWindow(slot, list);
  }
};

/**
 * Every node touching the same slot agrees on `window` **in presence and value**
 * (point 3 of {@link assertStateOrder} — logical col → physical row MUST use the same formula for
 * reads and writes).
 */
const assertSameWindow = (slot: string, list: readonly StateTouch[]): void => {
  const first = list[0];
  const mismatch = list.find((touch) => touch.window !== first.window);
  if (mismatch === undefined) return;
  const show = (touch: StateTouch): string =>
    `nodes[${touch.index}] (${touch.op}): ${touch.window ?? "undeclared"}`;
  throw new ExecutionError(
    `State slot '${slot}': attrs.window disagrees (${show(first)} / ${show(mismatch)})` +
      ` — the logical col → physical row mapping MUST use the same formula for reads and writes (ADR 0067 decision 4)`,
  );
};

/**
 * The contract of sym_prefix_slice that **needs the graph context** (ADR 0010). The contract table
 * has already checked the attrs' form; this checks only the three points that "cannot be decided
 * without seeing the whole graph".
 *
 * 1. `sym` is in `graph.symbols` (otherwise no binding exists at run time and the prefix length is
 *    undetermined)
 * 2. `dim` is inside the input rank
 * 3. The input's declared shape is **a static shape without symbols** (= the Tmax shape). If it were
 *    symbolic, the read stride would be built from a value that shrinks on every run, and T < Tmax
 *    always reads different rows
 *
 * MUST: fail before binding. Point 3 cannot be told apart from the numeric shapes after binding (on
 * a run with T = Tmax the numbers match), so it can only be detected here, where the declared form
 * is visible.
 */
const assertSymPrefixSlice = (graph: IrGraph, node: IrNode, where: string): void => {
  const { sym, slices } = symPrefixSliceAttrs(node.attrs, where);
  if (!graph.symbols.includes(sym)) {
    throw new ExecutionError(
      `${where}: sym_prefix_slice sym '${sym}' is not in graph.symbols [${graph.symbols.join(
        ", ",
      )}]`,
    );
  }
  const source = declarationOf(graph, node.ins[0]).shape;
  if (source.some((dim) => typeof dim !== "number")) {
    throw new ExecutionError(
      `${where}: sym_prefix_slice input '${node.ins[0]}' has a symbolic dimension in its declared shape [${source.join(
        ",",
      )}] (the input must be a static shape baked at Tmax)`,
    );
  }
  for (const slice of slices) {
    if (slice.dim >= source.length) {
      throw new ExecutionError(
        `${where}: sym_prefix_slice dim ${slice.dim} is outside the input rank ${source.length}`,
      );
    }
  }
};

/**
 * Checks that the target axis of slice / flip is **static at the declaration level** (ADR 0014).
 *
 * MUST: fail before binding. The numeric shapes after binding cannot tell whether an axis is
 * symbolic (on a run with T = the actual length the numbers match), so it can only be detected
 * here, where the declared form is visible — the same layer where assertSymPrefixSlice checks "the
 * input has the Tmax shape".
 *
 * Reasons per axis:
 * - `slice` — slicing a symbolic axis belongs to `sym_prefix_slice`; this op is static only
 *   (the split ADR 0014 decided so that the two do not overlap).
 * - `flip` — every measured use is a static axis (flow's 192ch / sdp's 2ch). Reversing a dynamic
 *   axis could be written in the kernel, but the vocabulary is not widened until a measured need
 *   appears (the same narrowing as softmax's dim).
 *
 * Only `cat`'s concatenation axis was relaxed, by ADR 0046 ({@link assertCatAxis}). The exporter's
 * shape layer has the same rule (the accepted set is kept equal on both sides).
 */
const assertStaticLayoutAxis = (
  graph: IrGraph,
  node: IrNode,
  kind: "slice" | "flip",
  where: string,
): void => {
  const dim = kind === "slice" ? sliceAttrs(node.attrs, where).dim : flipDim(node.attrs, where);
  for (const name of node.ins) {
    const shape = declarationOf(graph, name).shape;
    if (dim < shape.length && typeof shape[dim] !== "number") {
      throw new ExecutionError(
        `${where}: ${node.op} axis ${dim} of input '${name}' is symbolic in the declaration [${shape.join(
          ",",
        )}] (slicing a symbolic axis is the job of sym_prefix_slice)`,
      );
    }
  }
};

/**
 * cat's concatenation axis is a constant or a linear expression of **one** symbol (ADR 0046
 * revises ADR 0014).
 *
 * The sum `Σ(coeff_i·sym + offset_i)` fits the dimension language `coeff·sym+offset` as is when the
 * symbol is the same (`S`+1519 → `S+1519`, `S`+`S` → `2S`). Only **a mix of different symbols**
 * cannot be expressed.
 *
 * MUST: fail before binding. After binding, different symbols can resolve to the same value in the
 * numeric shapes, so it can only be detected here, where the declared form is visible (the same
 * layer as the static-axis check of slice / flip).
 * planGraph checks with the bound numbers that the output axis length equals the sum — this side
 * only stops "forms that cannot be expressed from the declarations alone".
 */
const assertCatAxis = (graph: IrGraph, node: IrNode, where: string): void => {
  const dim = catDim(node.attrs, where);
  let sym: string | undefined;
  for (const name of node.ins) {
    const shape = declarationOf(graph, name).shape;
    // A rank mismatch (dim outside) is checked by computeOutputShape after binding (not this
    // check's job).
    if (dim >= shape.length) continue;
    const extent = shape[dim];
    if (typeof extent === "number") continue;
    const parsed = parseDim(extent);
    if (sym !== undefined && sym !== parsed.sym) {
      throw new ExecutionError(
        `${where}: ${node.op} concatenation axis ${dim} mixes different symbols (${sym} and ${parsed.sym})` +
          ` (input '${name}' declared [${shape.join(",")}])` +
          " — the sum does not fit the dimension language coeff·sym+offset",
      );
    }
    sym = parsed.sym;
  }
};

/**
 * Value name → declared dtype (inputs, initializers and every node output). It does not depend on
 * bindings, so one lookup at Session construction suffices. It decides the TypedArray of uploads and
 * readbacks.
 */
export const declaredDtypes = (graph: IrGraph): ReadonlyMap<string, IrDtype> => {
  const dtypes = new Map<string, IrDtype>();
  for (const spec of graph.inputs) dtypes.set(spec.name, spec.dtype);
  for (const [name, value] of Object.entries(graph.values)) dtypes.set(name, value.dtype);
  return dtypes;
};

/**
 * Initializer consumed in a weight slot → **channel axis** of its per-channel scale (ADR 0019).
 *
 * An eligible initializer ({@link eligibleCompressedInitializers}) is consumed only in weight slots,
 * so the consuming op determines the axis uniquely. The same weight consumed by two ops with
 * different axes (e.g. linear and conv_transpose1d) splits the meaning of the scale, so it fails
 * loudly.
 *
 * MUST: look the axis up in {@link WEIGHT_CHANNEL_AXES}, the pair of {@link WEIGHT_SLOTS}. Guessing
 * the axis from kernel-side constants lets a shifted flat scale index slip past the shape check.
 */
export const weightChannelAxes = (graph: IrGraph): ReadonlyMap<string, number> => {
  const axes = new Map<string, number>();
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    if (weightSlot === undefined) continue;
    const name = node.ins[weightSlot];
    if (name === undefined || !Object.hasOwn(graph.initializers, name)) continue;
    const axis = WEIGHT_CHANNEL_AXES.get(node.op);
    if (axis === undefined) {
      throw new ExecutionError(
        `Op '${node.op}' has no channel axis definition for per-channel scales`,
      );
    }
    const known = axes.get(name);
    if (known !== undefined && known !== axis) {
      throw new ExecutionError(
        `Initializer '${name}': consumers disagree on the channel axis (${known} and ${axis})`,
      );
    }
    axes.set(name, axis);
  }
  return axes;
};

/**
 * Initializers that **can** stay GPU-resident in compressed storage (the eligibility of ADR 0018 /
 * 0019 — shared by f16 and i8. The decision depends only on the graph structure and does not look
 * at the storage dtype, so there is no reason to split it).
 *
 * Eligible = "the initializer is consumed **only** in weight slots of ops listed in
 * {@link WEIGHT_SLOTS}". Depends on neither bindings nor the GPU.
 *
 * MUST: one consumption outside a weight slot makes it ineligible. In a graph that uses the same
 * tensor both as a linear weight and as an elementwise operand, the latter's kernel reads it as f32,
 * so uploading it compressed reinterprets the bits (no exception is raised).
 * MUST: an initializer with zero consumers is ineligible too. Counting bytes that execution never
 * uses as "GPU-resident compressed" makes the diagnostics (ADR 0006) drift from reality.
 * MUST: an initializer listed in `graph.outputs` is ineligible too (the IR allows writing an
 * initializer name directly as a graph output). Readback copies from the weight buffer assuming the
 * declared dtype's semantic f32 (4 bytes / element), so if it stayed resident compressed the copy
 * overruns the real buffer for most sizes and fails validation (misleading — it does not point at
 * the cause), while for tiny sizes it fits within the zero padding and silently returns
 * reinterpreted bits.
 */
export const eligibleCompressedInitializers = (graph: IrGraph): ReadonlySet<string> => {
  const initializers = new Set(Object.keys(graph.initializers));
  const eligible = new Set<string>();
  const disqualified = new Set<string>(graph.outputs);
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    node.ins.forEach((name, slot) => {
      if (!initializers.has(name)) return;
      if (slot === weightSlot) eligible.add(name);
      else disqualified.add(name);
    });
  }
  for (const name of disqualified) eligible.delete(name);
  return eligible;
};

/**
 * Ops that **have an i4 decode path** (ADR 0069 decision 5 and its embedding / conv1d addenda 6 /
 * 7). conv1d is eligible **only with groups == 1**, and that narrowing needs the attrs, so
 * {@link i4Executable} does it. The mirror of the exporter's `karume/emit.py: I4_WEIGHT_OPS` +
 * `_has_i4_kernel` — **both the set's contents and the narrowing's structure are paired** (adding to
 * the set on one side only would turn the two constants that claim to mirror each other into
 * different things, and a future addendum would update only one side).
 *
 * MUST: do not include conv2d / conv3d / conv_transpose1d (they have no decode path at all — the
 * codegen side also rejects them: the conv2d / conv3d checks in src/kernels/gemm.ts and the direct
 * kernel check in src/kernels/conv1d.ts).
 */
const I4_WEIGHT_OPS: ReadonlySet<string> = new Set([LINEAR_OP, EMBEDDING_OP, CONV1D_OP]);

/**
 * Whether this node's weight slot has an i4 decode path (the predicate of ADR 0069 decision 5 and
 * its addenda).
 *
 * For conv1d, **only the implicit GEMM with `groups == 1`** has a decode path (the group scale while
 * filling the A tile — `fillAConv` in gemm.ts). `groups > 1` goes to the direct kernel, which has no
 * decode path (src/kernels/conv1d.ts). conv_transpose1d is excluded because it would need a pack in
 * the transposed layout (the maintainer's ruling of 2026-08-20).
 *
 * MUST: **do not fill `groups` with a default** (`conv1dAttrs` fails a missing one — ADR 0015: once
 * the field exists, only the absence of default filling guarantees it). If this silently assumed 1,
 * a depthwise weight would stay resident as i4 and the direct kernel would read the packed bytes as
 * f32.
 */
const i4Executable = (node: IrNode): boolean =>
  I4_WEIGHT_OPS.has(node.op) &&
  (node.op !== CONV1D_OP || conv1dAttrs(node.attrs, `nodes (${node.op})`).groups === 1);

/** INT2 can be executed packed only for linear / embedding weights (ADR 0097). */
export const i2EligibleInitializers = (graph: IrGraph): ReadonlySet<string> => {
  const executable = new Set<string>();
  const other = new Set<string>();
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    if (weightSlot === undefined) continue;
    const name = node.ins[weightSlot];
    if (name === undefined || !Object.hasOwn(graph.initializers, name)) continue;
    (node.op === LINEAR_OP || node.op === EMBEDDING_OP ? executable : other).add(name);
  }
  for (const name of other) executable.delete(name);
  return executable;
};

/**
 * Initializers whose weight-slot consumers are **only** ops satisfying {@link i4Executable} (the
 * narrowing of the i4 eligible set — ADR 0069 decision 5. The mirror of the exporter's
 * `karume/emit.py: i4_eligible_initializers`).
 *
 * MUST: use it as an **intersection** with {@link eligibleCompressedInitializers} — this only checks
 * "is it also consumed, among weight slots, by an op without a decode path (conv2d / conv3d /
 * conv_transpose1d / conv1d with groups > 1)". Keeping a shared weight resident makes that op's
 * kernel read the packed bytes as f32 (no exception is raised).
 */
export const i4EligibleInitializers = (graph: IrGraph): ReadonlySet<string> => {
  const executable = new Set<string>();
  const other = new Set<string>();
  for (const node of graph.nodes) {
    const weightSlot = WEIGHT_SLOTS.get(node.op);
    if (weightSlot === undefined) continue;
    const name = node.ins[weightSlot];
    if (name === undefined || !Object.hasOwn(graph.initializers, name)) continue;
    (i4Executable(node) ? executable : other).add(name);
  }
  for (const name of other) executable.delete(name);
  return executable;
};

/**
 * Value name → number of uses in the graph. MUST: an exact count over `node.ins` with repeats (a
 * value taken twice counts 2).
 *
 * NOTE: multi-output nodes (ADR 0068 decision 1) do not change the counting at all — only the
 * **consuming side** is counted, regardless of whether the defining side has one output or several.
 * The `uses` of each output just looks up "the use count of that value name" in this table
 * (`#buildStep` in recipe-builder).
 */
export const countUses = (graph: IrGraph): ReadonlyMap<string, number> => {
  const counts = new Map<string, number>();
  for (const node of graph.nodes) {
    for (const name of node.ins) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return counts;
};

const assertExtent = (size: number, where: string): number => {
  if (!Number.isSafeInteger(size) || size < 0) {
    throw new ExecutionError(`${where}: dimension ${String(size)} is not a non-negative integer`);
  }
  return size;
};

/**
 * The set of **states-only symbols** (symbols that appear only in states shapes — the KV capacity
 * `C` and the like).
 *
 * Their binding point is `createGenerationContext(spec.bindings)` (binding point 2 of ADR 0066
 * addendum 7), and they cannot in principle be solved from run / enqueue inputs. In a valid IR
 * "symbols not solvable from inputs = symbols appearing in states" holds (checkSymbolBindability in
 * format/ir.ts requires an appearance in inputs ∪ states and rejects states-only symbols in value
 * shapes), so the test only needs "does not appear at a dimension position of an input shape".
 */
export const statesOnlySymbols = (graph: IrGraph): ReadonlySet<string> => {
  const fromInputs = new Set<string>();
  for (const spec of graph.inputs) {
    for (const dim of spec.shape) {
      if (typeof dim !== "string") continue;
      fromInputs.add(parseDim(dim).sym);
    }
  }
  return new Set(graph.symbols.filter((sym) => !fromInputs.has(sym)));
};

/**
 * Checks that the context's capacity bindings and the run's resolved bindings agree on **symbols
 * that appear in both** (run-time enforcement of the "division of reach" in ADR 0066 addendum 7).
 *
 * A symbol has two binding points: input shapes (the run — {@link bindSymbols}) and `spec.bindings`
 * (the context). A symbol that appears in **both** states and inputs is decided independently at
 * the two places, so running with them split lets a divergence such as "allocated capacity C=4,
 * plan and dispatch C=3" stand with no exception or warning (state reads and writes fall outside
 * the declared shape = a silent misread whose out-of-bounds accesses robust buffer access discards).
 *
 * MUST: fail before encoding (a failure at this step does not reach the state, so it does not
 * poison).
 * NOTE: symbols that appear on only one side are out of scope — states-only symbols cannot be given
 * in the run's bindings ({@link bindSymbols} rejects them), and the context does not know symbols
 * that appear only in value shapes.
 */
export const assertGenerationBindings = (
  contextBindings: SymbolBindings,
  runBindings: SymbolBindings,
): void => {
  for (const [sym, value] of Object.entries(contextBindings)) {
    if (!Object.hasOwn(runBindings, sym)) continue;
    if (runBindings[sym] !== value) {
      throw new ExecutionError(
        `Symbol '${sym}' differs between the GenerationContext binding ${value} and the run's resolved value ` +
          `${runBindings[sym]} (a symbol that appears in both states and inputs must have the same value at` +
          " both binding points — ADR 0066 addendum 7)",
      );
    }
  }
};

/**
 * Binds the symbols from the actual input shapes.
 *
 * MUST: take bindings from **dimension positions of input shapes** (no back-solving from element
 * counts — with several symbols mixed the solution is not unique and a wrong binding passes
 * quietly). Derived forms with one symbol per dimension (`2T` / `T+8` and the like) have a unique
 * solution per position, so they are binding sources ({@link solveDim} — ADR 0057).
 * MUST: **do not require states-only symbols, and do not accept them in seed either** (the
 * division of reach of ADR 0066 addendum 7 — only input-derived bindings resolve value shapes; the
 * context holds the source of truth for capacity). Accepting them would keep two books ("the
 * capacity at context creation" and "the value passed to the run"), and a disagreement would not
 * show in the values but would quietly split the plan cache key (the same plan derived twice under
 * different keys).
 * MUST: an actual size that does not divide fails loudly. Rounding it would let a run that passes an
 * odd length to a declared `2T` go all the way through as "an input whose last element alone means
 * something different".
 * MUST: test whether a binding exists only with `Object.hasOwn`. `bindings[sym] !== undefined` lets
 * `toString` and the like from Object.prototype through, and later arithmetic silently turns into
 * NaN.
 *
 * @param deferredInputs inputs without a shape (resident tensors — they carry only bytes and a size,
 *   so the host has no shape). They **are not binding sources and are not checked**: `planGraph`
 *   resolves their declared shapes from bindings fixed by other inputs or `seed`, and the executor
 *   matches the backing's size against the resolved shape. A symbol that stays undetermined fails
 *   loudly as before.
 */
export const bindSymbols = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  seed: SymbolBindings = {},
  deferredInputs?: ReadonlySet<string>,
): SymbolBindings => {
  const symbols = new Set(graph.symbols);
  // MUST: the container that accumulates symbol names has a null prototype. The symbol grammar
  // (`[A-Za-z_][A-Za-z0-9_]*` in dims.ts) matches "__proto__", and on a plain `{}` the assignment
  // turns into setting [[Prototype]] without creating an own property, so a symbol that was bound
  // fails the hasOwn check right after as "could not be bound" (the accepted set splits by engine).
  // Deno disables this setter, so it did not reproduce in the Deno-era tests; Node and browsers
  // have it.
  const bindings: Record<string, number> = Object.create(null);
  const statesOnly = statesOnlySymbols(graph);
  for (const [sym, value] of Object.entries(seed)) {
    if (!symbols.has(sym)) {
      throw new ExecutionError(
        `Binding '${sym}' is not in the graph's symbols [${graph.symbols.join(", ")}]`,
      );
    }
    if (statesOnly.has(sym)) {
      throw new ExecutionError(
        `Binding '${sym}' is a states-only symbol — its binding point is createGenerationContext(spec.bindings)` +
          " (ADR 0066 addendum 7); it cannot be given in the run / enqueue bindings",
      );
    }
    bindings[sym] = assertExtent(value, `Binding '${sym}'`);
  }

  const declared = new Set(graph.inputs.map((spec) => spec.name));
  // MUST: deferred inputs also go through the "is it a graph input" check (letting them through
  // would silently ignore a resident tensor with a misspelt name, and the real input would fail as
  // "not passed").
  for (const name of [...Object.keys(inputShapes), ...(deferredInputs ?? [])]) {
    if (!declared.has(name)) throw new ExecutionError(`'${name}' is not a graph input`);
  }

  for (const spec of graph.inputs) {
    if (deferredInputs?.has(spec.name) === true) continue;
    if (!Object.hasOwn(inputShapes, spec.name)) {
      throw new ExecutionError(`Input '${spec.name}' was not passed`);
    }
    const actual = inputShapes[spec.name];
    if (actual.length !== spec.shape.length) {
      throw new ExecutionError(
        `Input '${spec.name}': rank ${actual.length} differs from the declaration [${spec.shape.join(",")}]`,
      );
    }
    spec.shape.forEach((dim, index) => {
      const where = `Input '${spec.name}' dimension ${index}`;
      const size = assertExtent(actual[index], where);
      if (typeof dim === "number") return;
      const expr = parseDim(dim);
      const solved = solveDim(expr, size);
      if (solved === undefined) {
        throw new ExecutionError(
          `${where}: actual ${size} does not have the declared form '${dim}'` +
            ` (needs a length divisible by ${expr.coeff} and at least ${expr.offset})`,
        );
      }
      if (!Object.hasOwn(bindings, expr.sym)) {
        bindings[expr.sym] = solved;
        return;
      }
      if (bindings[expr.sym] !== solved) {
        throw new ExecutionError(
          `${where}: conflicting bindings for symbol '${expr.sym}' (${bindings[expr.sym]} and ${solved})`,
        );
      }
    });
  }

  for (const sym of graph.symbols) {
    // States-only symbols are legitimately unsolvable from inputs (bound at context creation — the
    // MUST above).
    if (statesOnly.has(sym)) continue;
    if (!Object.hasOwn(bindings, sym)) {
      throw new ExecutionError(`Symbol '${sym}' could not be bound from the input shapes`);
    }
  }

  // Second pass: re-evaluate the declared shapes, numeric dimensions and derived forms included,
  // and require a full match with the actual inputs.
  for (const spec of graph.inputs) {
    if (deferredInputs?.has(spec.name) === true) continue;
    const resolved = resolveShape(spec.shape, bindings);
    const actual = inputShapes[spec.name];
    if (resolved.some((dim, index) => dim !== actual[index])) {
      throw new ExecutionError(
        `Input '${spec.name}': shape [${actual.join(",")}] does not match the declaration [${spec.shape.join(
          ",",
        )}] (after binding [${resolved.join(",")}])`,
      );
    }
  }
  return bindings;
};

/**
 * Distributes the bound shapes to every value, computes each node's output shape from its contract
 * and checks it against the declaration.
 *
 * @param stateShapes binding-resolved state slot shapes (slot name → concrete shape including
 *   capacity). The GenerationContext that allocates the backing (ADR 0066 decision 1) passes them —
 *   in a plan without them, state-referencing nodes fail loudly (they are not silently computed in
 *   the old form).
 */
export const planGraph = (
  graph: IrGraph,
  bindings: SymbolBindings,
  stateShapes?: ReadonlyMap<string, readonly number[]>,
): GraphPlan => {
  const shapes = new Map<string, readonly number[]>();
  for (const spec of graph.inputs) shapes.set(spec.name, resolveShape(spec.shape, bindings));
  for (const [name, value] of Object.entries(graph.values)) {
    shapes.set(name, resolveShape(value.shape, bindings));
  }
  // MUST: check once, at this point where every value is in one table, that the element count of
  // each resolved shape fits a safe integer (an **incremental** check of the same form as
  // `elementCount` in format/safetensors.ts — checking the product at the end only sees the value
  // after an intermediate overflow has dropped the low digits). `numel` returns the plain product,
  // so beyond 2^53 both the allocation size and the dispatch count become rounded values, ending in
  // reads and writes outside the declared range with no exception.
  for (const [name, shape] of shapes) {
    let count = 1;
    for (const dim of shape) {
      count *= dim;
      if (!Number.isSafeInteger(count)) {
        throw new ExecutionError(
          `Value '${name}': the element count of shape [${shape.join(",")}] exceeds the safe integer range`,
        );
      }
    }
  }

  const nodes = graph.nodes.map((node, index): NodePlan => {
    const where = `nodes[${index}] (${node.op})`;
    const contract = assertNodeContract(node, where);
    const inputShapes = node.ins.map((name) => {
      const shape = shapes.get(name);
      if (shape === undefined) {
        throw new ExecutionError(`${where}: shape of input '${name}' is unresolved`);
      }
      return shape;
    });
    const declaredShapes = node.outs.map((name) => {
      const shape = shapes.get(name);
      if (shape === undefined) {
        throw new ExecutionError(`${where}: output '${name}' has no declaration`);
      }
      return shape;
    });
    // For reshape / expand "the declared shape is the target shape"; permute needs attrs (ADR 0011).
    // Passing the declaration does not skip the check below — only ops that use it as the target
    // shape match trivially; other ops still fail on a mismatch between computed and declared shape.
    // NOTE: the two ops that require a target shape (reshape / expand) have a single output, so
    // {@link ShapeContext} takes only one declared shape (slot 0). Multi-output ops derive their
    // shapes from the inputs.
    const computed = computeOutputShape(contract, inputShapes, where, {
      declared: declaredShapes[0],
      attrs: node.attrs,
      bindings,
      states: node.states,
      stateShapes,
    });
    // MUST: check the list lengths here. If the output count the contract declares (the length of
    // `node.outs` — checked by assertNodeContract) and the length of the list the shape computation
    // returned diverge, the comparison below touches undefined and throws TypeError without saying
    // which slot of which op is missing.
    if (computed.length !== declaredShapes.length) {
      throw new ExecutionError(
        `${where}: output shape computation returned ${computed.length} shapes (${declaredShapes.length} declared)`,
      );
    }
    declaredShapes.forEach((declaredShape, slot) => {
      const shape = computed[slot];
      if (
        declaredShape.length !== shape.length ||
        declaredShape.some((dim, i) => dim !== shape[i])
      ) {
        throw new ExecutionError(
          `${where}: output '${node.outs[slot]}': computed shape [${shape.join(",")}] does not match the declaration [${declaredShape.join(
            ",",
          )}]`,
        );
      }
    });
    const dtypes = nodeDtypes(graph, node, contract, where);
    return {
      node,
      contract,
      inputShapes,
      inputDtypes: dtypes.inputs,
      outputs: node.outs.map((name, slot): NodeOutputPlan => ({
        name,
        shape: computed[slot],
        dtype: dtypes.outputs[slot],
      })),
    };
  });

  return { bindings, shapes, nodes };
};

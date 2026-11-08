/**
 * The runtime support table (executable ops / storage / transfer dtypes) and the check that matches
 * a graph against it.
 *
 * The table's actual values live in the contract table (`RUNTIME_SUPPORT` in
 * {@link "./contracts.ts"}); this module holds **only the types and the matching procedure**. Rules
 * decided by the graph alone (declarations, SSA, vocabulary) belong to `format/ir.ts`, and matching
 * declared shapes against the supply belongs to `format/container/bind.ts`, so this layer looks at
 * one thing only: "can this runtime execute it".
 */

import { type CodecLayout, codecLayout } from "../format/container/codecs.ts";
import type { IrDtype, IrGraph } from "../format/ir.ts";

/** Missing capability (an op / semantic dtype / attrs / storage that cannot be executed). */
export class RuntimeSupportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RuntimeSupportError";
  }
}

/** Per-op conditions for execution. The op name alone cannot express dtype and attrs differences. */
export type OpSupport = {
  /**
   * The **union** of the semantic dtypes that can be executed (differs per op — the contract table
   * in src/ops/contracts.ts is the source of truth). For ops with a per-slot contract this union is
   * wider than what is actually accepted, so it is not used to match inputs.
   */
  readonly dtypes: ReadonlySet<IrDtype>;
  /**
   * The accepted set **per input slot** (ordered, one per contract arity). Under a uniform contract
   * every slot has the same set.
   *
   * MUST: match inputs against this field. Looking only at the union lets a slot mix-up in gather /
   * embedding / masked_fill (value and index passed the wrong way round, as in `gather(index, src)`)
   * slip through the enumerating check as "both dtypes are in the union" — it is then stopped one
   * case at a time only by the contract check (plan.ts), which breaks the intent of "list every
   * unsupported item".
   */
  readonly slotDtypes: readonly ReadonlySet<IrDtype>[];
  /**
   * The semantic dtypes that can appear **per output slot** (the range of the contract table's
   * output dtype mapping — ordered, one per output the contract declares. ADR 0068 decision 1).
   *
   * MUST: do not substitute the accepted set of input slot 0. Comparisons (f32 → bool), `sum` of
   * bool (→ i32) and `where` (bool → f32) have different input and output dtypes, so matching
   * against slot 0 makes **a correct graph fail the enumerating check**. Conversely, for identity
   * ops the two agree, so a dedicated field does not change any existing verdict.
   */
  readonly outDtypes: readonly ReadonlySet<IrDtype>[];
  /**
   * The implemented attr keys (the union of the keys of the required attrs and the optional attrs
   * (`optionalAttrs`) declared by the contract table).
   */
  readonly attrKeys: ReadonlySet<string>;
};

/** The table of the ops (including dtype / attrs) and storage dtypes the runtime can execute. */
export type RuntimeSupport = {
  readonly ops: ReadonlyMap<string, OpSupport>;
  /**
   * The storage dtypes that can be executed.
   *
   * NOTE: f16 is "kept compressed and GPU-resident for an eligible weight slot, otherwise decoded
   * to f32 on the CPU at load time" (ADR 0018); it can be executed on either path, so it is in this
   * set. The eligibility verdict does not change whether it can execute, only how VRAM is used, so
   * there is no branch here (keeping an eligible 0 MB from being silent is the job of the Session's
   * diagnostics — ADR 0006).
   */
  readonly storage: ReadonlySet<CodecLayout>;
  /**
   * Storage layouts executable **only by named consumers**: layout → (op name → its weight slot).
   * An initializer in such a layout is supported when it has at least one consumer and every
   * consumption is one of these (op, slot) pairs — `i4f16` (`int4-sym-g-f16`), whose f16 steps
   * only `expert_linear` reads (ADR 0145 decision 3).
   *
   * MUST: a layout listed here is not also in {@link storage}. There it would be executable for
   * every initializer, and any other consumer would read its f16 step words as f32 (a silent wrong
   * value with a plausible byte count). Absent = no such layout.
   */
  readonly opBoundStorage?: ReadonlyMap<CodecLayout, ReadonlyMap<string, number>>;
  /**
   * The semantic dtypes that can be transferred as graph inputs.
   *
   * MUST: this axis is needed separately from the op-based matching. The executor transfers every
   * entry of `graph.inputs`, including inputs that no node consumes, so the transfer layer's
   * constraints exist regardless of "whether a node uses that input". Without this axis, only the
   * dtype violation of an unused input would slip through the enumerating check.
   */
  readonly io: ReadonlySet<IrDtype>;
};

/**
 * The declared dtype of a value a node touches. The IR parser already guarantees that ins / outs
 * are declared in exactly one place, inputs[] or values{} (checkDefinitions / checkDeclarations).
 */
const declaredDtype = (graph: IrGraph, name: string): IrDtype => {
  const input = graph.inputs.find((spec) => spec.name === name);
  return input !== undefined ? input.dtype : graph.values[name].dtype;
};

/**
 * Whether `name` has at least one consumer and every consumption is a listed (op, slot) pair (the
 * test of {@link RuntimeSupport.opBoundStorage}).
 */
const onlyReadBy = (
  graph: IrGraph,
  name: string,
  readers: ReadonlyMap<string, number>,
): boolean => {
  let uses = 0;
  for (const node of graph.nodes) {
    for (const [slot, input] of node.ins.entries()) {
      if (input !== name) continue;
      if (readers.get(node.op) !== slot) return false;
      uses += 1;
    }
  }
  return uses > 0;
};

/**
 * Matches the graph against the runtime support table.
 *
 * MUST: look not only at the op name but **at the semantic dtypes and attrs too**. Matching by name
 * alone produces "in the support table, yet fails at execution" (the shape named by the recon in
 * docs/research, §3-9 — observed as a real bug in the prior experiment).
 * Unsupported items are **all enumerated** and fail loudly (failing one at a time leaves whoever
 * fills the support table unable to tell how many are missing).
 *
 * NOTE: plan.ts's `validateGraphContracts` looks at the same rules, but at a different layer — this
 * one is "the check that lists missing capabilities to the model author at once", that one is "the
 * GPU-independent contract check run every time". Both derive from the contract table of
 * src/ops.ts, so the rules cannot diverge.
 */
export const assertRuntimeSupport = (graph: IrGraph, support: RuntimeSupport): void => {
  const missingOps = new Set<string>();
  // MUST: dedupe dtype violations **per declaration (value name)**. An aliased input (`add(h,h)`),
  // or a value that appears both in the defining node's outs and in a consuming node's ins, hits the
  // same declaration several times; collecting naively makes the count N larger than "the number of
  // declarations to fix" and weakens its meaning as an enumeration measure.
  const badDtypes = new Map<string, IrDtype>();
  const badAttrs: string[] = [];
  // The transfer-layer axis. Checked before the node-based matching (declaration order = inputs first).
  for (const spec of graph.inputs) {
    if (!support.io.has(spec.dtype)) badDtypes.set(spec.name, spec.dtype);
  }
  graph.nodes.forEach((node, index) => {
    const op = support.ops.get(node.op);
    if (op === undefined) {
      missingOps.add(node.op);
      return;
    }
    const where = `nodes[${index}] (${node.op})`;
    node.ins.forEach((name, slot) => {
      // More inputs than the contract (an arity violation) is the contract check's job. This is the
      // enumerating check, so inputs without a matching slot are looked at via the union to catch as
      // many as possible.
      const accept = op.slotDtypes[slot] ?? op.dtypes;
      const dtype = declaredDtype(graph, name);
      if (!accept.has(dtype)) badDtypes.set(name, dtype);
    });
    node.outs.forEach((name, slot) => {
      // Outputs are checked against the range of the contract table's mapping **per output slot**
      // (cast is decided by attrs.to, so the whole vocabulary). Substituting the input-side set
      // makes correct graphs fail for ops that change the dtype, such as comparisons or sum of bool.
      // More outputs than the contract (an output-count violation) is, as on the input side, the
      // contract check's job; the surplus is looked at via the union of all slots to catch as many
      // as possible.
      const accept =
        op.outDtypes[slot] ?? new Set(op.outDtypes.flatMap((slotAccept) => [...slotAccept]));
      const dtype = declaredDtype(graph, name);
      if (!accept.has(dtype)) badDtypes.set(name, dtype);
    });
    const unknown = Object.keys(node.attrs)
      .filter((key) => !op.attrKeys.has(key))
      .sort();
    if (unknown.length > 0) badAttrs.push(`${where}: ${unknown.join(", ")}`);
  });

  // Storage is checked by decode path (layout) — `ternary` is the same i2 path as `int2-off`, so it
  // needs no support of its own.
  // A shared initializer has no storage (whether it can execute is decided by the lender's resident
  // weight — checked when the borrower is built).
  const missingStorage = new Map<CodecLayout, string[]>();
  for (const [name, initializer] of Object.entries(graph.initializers)) {
    if (initializer.shared !== undefined) continue;
    const layout = codecLayout(initializer.storage.codec);
    if (support.storage.has(layout)) continue;
    const readers = support.opBoundStorage?.get(layout);
    if (readers !== undefined && onlyReadBy(graph, name, readers)) continue;
    const users = missingStorage.get(layout) ?? [];
    users.push(name);
    missingStorage.set(layout, users);
  }

  if (
    missingOps.size === 0 &&
    badDtypes.size === 0 &&
    badAttrs.length === 0 &&
    missingStorage.size === 0
  )
    return;

  const diagnostics: string[] = [];
  if (missingOps.size > 0) {
    diagnostics.push(`unsupported op (${missingOps.size}): ${[...missingOps].sort().join(", ")}`);
  }
  if (badDtypes.size > 0) {
    const listed = [...badDtypes].map(([name, dtype]) => `value '${name}': ${dtype}`);
    diagnostics.push(`unsupported semantic dtype (${badDtypes.size}): ${listed.join(", ")}`);
  }
  if (badAttrs.length > 0) {
    diagnostics.push(`unimplemented attrs (${badAttrs.length}): ${badAttrs.join(", ")}`);
  }
  for (const [layout, users] of [...missingStorage].sort((a, b) => a[0].localeCompare(b[0]))) {
    diagnostics.push(
      `unsupported storage '${layout}' (${users.length}): ${users.sort().join(", ")}`,
    );
  }
  throw new RuntimeSupportError(`Missing runtime capabilities — ${diagnostics.join(" / ")}`);
};

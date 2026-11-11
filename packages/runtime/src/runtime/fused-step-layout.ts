/**
 * The bookkeeping of one execution step that needs no device: its output specs, its release list
 * and, for a fused step, the schedule of its temporaries and what every binding of every dispatch
 * points at.
 *
 * Two callers read it: the derivation phase (`#buildStep` / `#buildFused` in recipe-builder.ts),
 * which replays it into `StepRecipeBuilder` and adds only what needs the device (pipelines, params,
 * resident buffers, workgroup counts), and the memory estimate, which maps it to a
 * `TransientStepSpec` (transient-plan.ts).
 * MUST: one implementation for both. A hand copy on the estimate's side counts a slot table
 * different from the one execution allocates, with no exception — the copy drifts the first time a
 * rule uses a form it did not copy.
 * MUST: pure functions (no GPU object, no async, no module-level state). What a binding reads or
 * writes comes in as an argument ({@link StorageRoles} — execution takes them from the pipeline
 * cache, the estimate parses them from the dispatch's WGSL); this module never generates WGSL.
 */

import type { StorageRoles } from "../gpu/pipeline-cache.ts";
import { numel } from "../ops.ts";
import type { ExecStep, FusedDispatch, FusedOperand, FusedStep, FusedTemp } from "./fusion.ts";
import { ExecutionError } from "./plan.ts";

/** Where the bookkeeping of an output's lifetime is read (the plan's use counts and graph outputs). */
export type OutputBookkeeping = {
  /** Future consumption count per value name (the exact count over `node.ins`). */
  readonly useCounts: ReadonlyMap<string, number>;
  /** The graph outputs (alive to the end, so they stay readable for readback). */
  readonly outputNames: ReadonlySet<string>;
};

/**
 * The allocation spec and bookkeeping of one step output. `Source` is what an alias points at: the
 * derivation phase passes its bind-surface source (the shape of `StepOutput` in recipe.ts), the
 * estimate a value name.
 */
export type StepOutputSpec<Source> = (
  | { readonly kind: "alloc"; readonly byteLength: number }
  | { readonly kind: "alias"; readonly source: Source }
) & {
  readonly name: string;
  readonly uses: number;
  readonly pinned: boolean;
};

/**
 * The output specs of one execution step, in ascending output slot (a plain node: the order and
 * length of `node.outs`; a fused step: its single output).
 *
 * An alias that keeps the element order makes the input buffer the output's object as is (ADR 0011)
 * and allocates nothing; planGraph has already checked that the element counts match, so the buffer
 * behind it always covers the declared shape. Every other output is allocated at `numel × 4` bytes.
 *
 * @param aliasSource What an aliasing output points at: the head of the bind surface. MUST: the ops
 *   that can alias (reshape / expand / permute) have a single output and alias their first input,
 *   so one source covers every alias (aliasing a multi-output op would need a source per slot).
 */
export const stepOutputs = <Source>(
  step: ExecStep,
  bookkeeping: OutputBookkeeping,
  aliasSource: Source,
): readonly StepOutputSpec<Source>[] => {
  const slots: readonly { readonly name: string; readonly shape: readonly number[] }[] =
    step.kind === "node" ? step.plan.outputs : [{ name: step.outputName, shape: step.outputShape }];
  const aliasesInput = step.kind === "node" && step.aliasesInput;
  return slots.map(({ name, shape }): StepOutputSpec<Source> => {
    const lifetime = {
      name,
      uses: bookkeeping.useCounts.get(name) ?? 0,
      pinned: bookkeeping.outputNames.has(name),
    };
    return aliasesInput
      ? { ...lifetime, kind: "alias", source: aliasSource }
      : { ...lifetime, kind: "alloc", byteLength: numel(shape) * 4 };
  });
};

/**
 * The release list of one execution step: the inputs it consumes, one entry per use (a plain node:
 * `node.ins`; a fused step: the total-use list of its external inputs, {@link FusedStep.ins}).
 *
 * MUST: not the bind surface. A fused step binds each input once ({@link FusedStep.binds}) but
 * releases it once per use the folded nodes made; releasing by the bind surface drops a count per
 * repeated use, and the buffer is reassigned while a later node still reads it.
 */
export const stepReleases = (step: ExecStep): readonly string[] =>
  step.kind === "node" ? step.plan.node.ins : step.ins;

/**
 * What one binding of a fused dispatch points at (binding `slot + 1`; binding 0 is the params).
 *
 * - `bind` — an external input, by its index into {@link FusedStep.binds} and its value name.
 * - `weightScale` — the resident scale of the weight at that bind index (`name` is undefined when
 *   the index is outside the binds; the scale's lookup then fails).
 * - `temp` — a step temporary, by its index into {@link FusedStepLayout.temps} (allocation order,
 *   the order the recipe records them in — never the rule's declared id).
 * - `output` — the step's single output.
 * - `unresolved` — an operand that points at nothing: a bind index outside the binds, or a
 *   temporary that is not allocated by this dispatch. `reason` says which, for the caller to throw at
 *   the point it resolves the binding.
 */
export type FusedSlot =
  | { readonly kind: "bind"; readonly index: number; readonly name: string }
  | { readonly kind: "weightScale"; readonly index: number; readonly name: string | undefined }
  | { readonly kind: "temp"; readonly temp: number }
  | { readonly kind: "output" }
  | { readonly kind: "unresolved"; readonly operand: "bind" | "temp"; readonly reason: string };

/** The schedule and bindings of one dispatch of a fused step (index-aligned with its dispatches). */
export type FusedDispatchLayout = {
  /** Temporaries allocated right before this dispatch (indices into {@link FusedStepLayout.temps}, ascending). */
  readonly allocate: readonly number[];
  /** What binding `slot + 1` points at, one entry per slot. */
  readonly operands: readonly FusedSlot[];
  /** Temporaries released right after this dispatch (indices into {@link FusedStepLayout.temps}), in release order. */
  readonly release: readonly number[];
};

/** The device-free layout of a fused step ({@link fusedStepLayout}). */
export type FusedStepLayout = {
  /**
   * The step's temporaries in **allocation order**: by `allocBefore`, and by declared id within one
   * boundary. The recipe records them in this order (`StepRecipeBuilder.allocTemp` numbers them as
   * they come), so an index here is the temporary's id in the recipe.
   */
  readonly temps: readonly FusedTemp[];
  readonly dispatches: readonly FusedDispatchLayout[];
};

/**
 * The operand list of one dispatch: its declared {@link FusedDispatch.operands}, or by default every
 * bind in declaration order, then the output (the shape shared by single-dispatch rules).
 */
const dispatchOperands = (step: FusedStep, dispatch: FusedDispatch): readonly FusedOperand[] =>
  dispatch.operands ?? [
    ...step.binds.map((_, index): FusedOperand => ({ kind: "bind", index })),
    { kind: "output" },
  ];

/**
 * The layout of a fused step: its temporaries in allocation order and, per dispatch, which of them
 * are allocated before it and released after it, and what each of its bindings points at.
 *
 * Before dispatch `i` the temporaries whose `allocBefore` is `i` are allocated in ascending declared
 * id; after it the ones whose `releaseAfter` is `i` are released in descending declared id. A
 * temporary operand resolves to a temporary allocated at or before its dispatch; one bound after its
 * release still resolves here, and the recipe's declaration check (`validateStepRecipe`) refuses it.
 * MUST: an operand that points at nothing is returned as `unresolved`, not thrown. The derivation
 * phase throws it only where it resolves that binding, after the dispatches before it have fetched
 * their pipelines and written their params — the order its failures have always come in.
 */
export const fusedStepLayout = (step: FusedStep): FusedStepLayout => {
  const temps: FusedTemp[] = [];
  // Declared id → index in `temps` (a hole while the temporary is not allocated yet).
  const allocated: (number | undefined)[] = [];
  const slotOf = (operand: FusedOperand): FusedSlot => {
    if (operand.kind === "output") return { kind: "output" };
    if (operand.kind === "bind") {
      const name: string | undefined = step.binds[operand.index];
      return name === undefined
        ? {
            kind: "unresolved",
            operand: "bind",
            reason: `bind index ${operand.index} is outside the ${step.binds.length} declared binds`,
          }
        : { kind: "bind", index: operand.index, name };
    }
    if (operand.kind === "weightScale") {
      const name: string | undefined = step.binds[operand.index];
      return { kind: "weightScale", index: operand.index, name };
    }
    const temp = allocated[operand.id];
    return temp === undefined
      ? { kind: "unresolved", operand: "temp", reason: `unallocated temporary ${operand.id}` }
      : { kind: "temp", temp };
  };
  const dispatches = step.dispatches.map((dispatch, index): FusedDispatchLayout => {
    const allocate: number[] = [];
    for (const [id, temp] of step.temps.entries()) {
      if (temp.allocBefore !== index) continue;
      allocated[id] = temps.length;
      allocate.push(temps.length);
      temps.push(temp);
    }
    const operands = dispatchOperands(step, dispatch).map(slotOf);
    const release: number[] = [];
    for (let id = step.temps.length - 1; id >= 0; id -= 1) {
      if (step.temps[id].releaseAfter !== index) continue;
      const temp = allocated[id];
      // The declaration's entry point refuses such a lifetime (`allocBefore ≤ releaseAfter`), so
      // only a step built by hand reaches this.
      if (temp === undefined) {
        throw new ExecutionError(
          `fusion rule '${step.rule}': temporary ${id} is released after dispatch ${index}, before it is allocated`,
        );
      }
      release.push(temp);
    }
    return { allocate, operands, release };
  });
  return { temps, dispatches };
};

/**
 * Checks that an external rule's dispatch writes only its step's output and temporaries: a binding
 * its WGSL declares `read_write` binds the output or a temporary of the step.
 *
 * MUST: the runtime owns every other buffer such a kernel could bind. An input is read again by later
 * nodes; a resident weight or scale is shared by every Session that leases it; a params buffer is
 * shared by content with every dispatch that has the same params. A kernel that wrote any of them
 * would corrupt values outside its step silently, so the refusal names the rule and the binding.
 */
export const assertExternalWrites = (
  step: FusedStep,
  index: number,
  operands: readonly FusedSlot[],
  roles: StorageRoles,
): void => {
  const refuse = (binding: number, what: string): never => {
    throw new ExecutionError(
      `fusion rule '${step.rule}': dispatch ${index} ('${step.dispatches[index].key}') declares binding ${binding}` +
        ` read_write, but it binds ${what} (an external step writes only its output and its temporaries)`,
    );
  };
  if (roles.writes.has(0)) refuse(0, "the params");
  for (const [slot, operand] of operands.entries()) {
    const writable =
      operand.kind === "output" ||
      operand.kind === "temp" ||
      (operand.kind === "unresolved" && operand.operand === "temp");
    if (!roles.writes.has(slot + 1) || writable) continue;
    // An external step binds no weightScale (refused when the step is planned), so it is an input.
    const name =
      operand.kind === "bind" || operand.kind === "weightScale" ? operand.name : undefined;
    refuse(slot + 1, `the input '${String(name)}'`);
  }
};

/**
 * Checks that an external step writes its output: some dispatch binds it to a binding its WGSL
 * declares `read_write`.
 *
 * MUST: a step whose every dispatch binds the output `read` leaves the buffer as the pool handed it
 * over, and its consumers read stale data silently. The plan checks only that some dispatch binds
 * it; what a binding writes is known only from the WGSL roles (a built-in rule writes its output by
 * construction).
 *
 * @param roles The roles of each dispatch, index-aligned with `layout.dispatches`.
 */
export const assertExternalOutputWritten = (
  step: FusedStep,
  layout: FusedStepLayout,
  roles: readonly StorageRoles[],
): void => {
  const written = layout.dispatches.some((dispatch, index) =>
    dispatch.operands.some(
      (operand, slot) => operand.kind === "output" && roles[index].writes.has(slot + 1),
    ),
  );
  if (written) return;
  throw new ExecutionError(
    `fusion rule '${step.rule}': no dispatch declares the binding of the step's output` +
      ` '${step.outputName}' read_write (its buffer would never be written, and its consumers would read` +
      " stale data)",
  );
};

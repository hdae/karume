/**
 * The corpus driver: runs a graph on a plan knowing only its declaration.
 *
 * The plan says what the declaration cannot: which symbol is the rows of a step, the values of the
 * other input symbols, and, for a graph with state slots, the capacity symbol's value per run. The
 * inputs of each step are drawn from what consumes them — an index read by `embedding` below the
 * table's row count, f32 uniform in [-1, 1), bool all ones — and an input outside those rules
 * throws with its name. Dimension expressions are evaluated with the runtime's own dimension
 * grammar. A graph with state slots runs its steps through one generation context (the first step
 * is the prefill chunk, later steps append); a graph without runs each step directly.
 */

import { evalDim, parseDim } from "../../src/format/dims.ts";
import type { IrDeclaration, IrDim } from "../../src/format/ir.ts";
import type { FusionCounts, Session, Tensor } from "../../src/runtime/executor.ts";
import { fillWords, randomStream, signedUnit } from "./random.ts";

/** One run of a plan: the rows of each step, and the capacity of the state slots. */
export type CorpusRun = {
  readonly name: string;
  /** Rows per step: the first is the prefill chunk; on a graph with states, later ones append. */
  readonly steps: readonly number[];
  /** The capacity symbol's value (graphs with symbolic state slots only). */
  readonly capacity?: number;
};

/** What a corpus entry tells the driver beyond the declaration. */
export type CorpusPlan = {
  /** The symbol whose value is the rows of a step. */
  readonly rowSymbol: string;
  /** Every other input symbol's value (the same in every step). */
  readonly bindings: Readonly<Record<string, number>>;
  /** The states-only symbol that sizes the state slots (graphs with symbolic state slots only). */
  readonly capacitySymbol?: string;
  readonly runs: readonly CorpusRun[];
};

/** One output of one step, as words (a comparison of words compares bits). */
export type StepOutput = {
  readonly name: string;
  readonly dtype: Tensor["dtype"];
  readonly shape: readonly number[];
  readonly words: Uint32Array<ArrayBuffer>;
};

/** One step's outputs (declaration order) and what the Session dispatched for it. */
export type StepResult = {
  readonly rows: number;
  readonly outputs: readonly StepOutput[];
  readonly pipelines: readonly { readonly key: string; readonly dispatchCount: number }[];
  readonly fusions: FusionCounts;
};

const symbolsOf = (shapes: Iterable<readonly IrDim[]>): Set<string> => {
  const found = new Set<string>();
  for (const shape of shapes) {
    for (const dim of shape) if (typeof dim === "string") found.add(parseDim(dim).sym);
  }
  return found;
};

/** Fails loudly when the plan does not bind what the declaration needs. */
export const assertPlanFits = (declaration: IrDeclaration, plan: CorpusPlan): void => {
  const inInputs = symbolsOf(declaration.inputs.map((input) => input.shape));
  const inStates = symbolsOf(Object.values(declaration.states).map((slot) => slot.shape));
  if (!inInputs.has(plan.rowSymbol)) {
    throw new Error(
      `the plan's row symbol '${plan.rowSymbol}' sizes no input of the graph (input symbols: ${[...inInputs].join(", ")})`,
    );
  }
  for (const symbol of Object.keys(plan.bindings)) {
    if (!inInputs.has(symbol) || symbol === plan.rowSymbol) {
      throw new Error(
        `the plan binds '${symbol}', which is not an input symbol other than the row symbol: bind input symbols only`,
      );
    }
  }
  for (const symbol of inInputs) {
    if (symbol !== plan.rowSymbol && !Object.hasOwn(plan.bindings, symbol)) {
      throw new Error(`input symbol '${symbol}' has no value in the plan: give it one in bindings`);
    }
  }
  const capacitySymbols = [...inStates].filter((symbol) => !inInputs.has(symbol));
  const expected = capacitySymbols.length === 0 ? undefined : capacitySymbols.join(", ");
  if (plan.capacitySymbol !== expected) {
    throw new Error(
      `the plan's capacity symbol is ${plan.capacitySymbol ?? "absent"}, but the graph's states-only symbols are ${expected ?? "none"}`,
    );
  }
  for (const run of plan.runs) {
    if (
      run.steps.length === 0 ||
      run.steps.some((rows) => !Number.isSafeInteger(rows) || rows < 1)
    ) {
      throw new Error(`run '${run.name}': every step needs a positive whole number of rows`);
    }
    if ((plan.capacitySymbol === undefined) !== (run.capacity === undefined)) {
      throw new Error(
        `run '${run.name}': a capacity is given exactly when the graph has a capacity symbol`,
      );
    }
  }
};

const numelOf = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

const evaluate = (shape: readonly IrDim[], bindings: Readonly<Record<string, number>>): number[] =>
  shape.map((dim) => (typeof dim === "number" ? dim : evalDim(parseDim(dim), bindings)));

/**
 * The inputs of one step. `label` names the stream of each input's values (with the input's name
 * appended), so every step of every run draws its own.
 */
export const stepInputs = (
  declaration: IrDeclaration,
  bindings: Readonly<Record<string, number>>,
  seed: number,
  label: string,
): Record<string, Tensor> => {
  const inputs: Record<string, Tensor> = {};
  for (const input of declaration.inputs) {
    const shape = evaluate(input.shape, bindings);
    const count = numelOf(shape);
    if (input.dtype === "bool") {
      inputs[input.name] = { dtype: "bool", shape, data: new Uint32Array(count).fill(1) };
      continue;
    }
    const words = new Uint32Array(count);
    fillWords(randomStream(seed, `${label}/${input.name}`), words);
    if (input.dtype === "f32") {
      const data = new Float32Array(count);
      for (let i = 0; i < count; i += 1) data[i] = signedUnit(words[i]);
      inputs[input.name] = { dtype: "f32", shape, data };
      continue;
    }
    // i32: only an index read by `embedding` (slot 1) has a rule — below the table's row count.
    let rows = Number.POSITIVE_INFINITY;
    for (const node of declaration.nodes) {
      node.ins.forEach((name, slot) => {
        if (name !== input.name) return;
        if (node.op !== "embedding" || slot !== 1) {
          throw new Error(
            `input '${input.name}' (i32) is read by ${node.op} slot ${slot}: the driver draws i32 inputs only as embedding indices`,
          );
        }
        const table = declaration.values[node.ins[0]];
        rows = Math.min(rows, evaluate(table.shape.slice(0, 1), bindings)[0]);
      });
    }
    if (!Number.isFinite(rows) || rows < 1) {
      throw new Error(
        `input '${input.name}' (i32) indexes no table with rows (it is read by no embedding, or the table is empty)`,
      );
    }
    const data = new Int32Array(count);
    for (let i = 0; i < count; i += 1) data[i] = Math.floor((words[i] / 2 ** 32) * rows);
    inputs[input.name] = { dtype: "i32", shape, data };
  }
  return inputs;
};

const copyOutputs = (
  declaration: IrDeclaration,
  outputs: Readonly<Record<string, Tensor>>,
): StepOutput[] =>
  declaration.outputs.map((name) => {
    const tensor = Object.hasOwn(outputs, name) ? outputs[name] : undefined;
    if (tensor === undefined) throw new Error(`the run returned no output '${name}'`);
    const { data } = tensor;
    const words = new Uint32Array(
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
    );
    return { name, dtype: tensor.dtype, shape: [...tensor.shape], words };
  });

const census = (session: Session): Pick<StepResult, "pipelines" | "fusions"> => {
  const diagnostics = session.diagnostics();
  const pipelines = diagnostics.lastRunPipelines;
  const fusions = diagnostics.lastRunFusions;
  if (pipelines === undefined || fusions === undefined) {
    throw new Error("the Session reported no dispatch census for the step it just ran");
  }
  return { pipelines: [...pipelines], fusions };
};

/** Runs one run of the plan and returns every step's outputs and census. */
export const driveRun = async (
  session: Session,
  declaration: IrDeclaration,
  plan: CorpusPlan,
  run: CorpusRun,
  seed: number,
): Promise<StepResult[]> => {
  assertPlanFits(declaration, plan);
  const bindingsOf = (rows: number): Record<string, number> => ({
    ...plan.bindings,
    [plan.rowSymbol]: rows,
  });
  const labelOf = (step: number): string => `${run.name}/${step}`;
  const results: StepResult[] = [];
  if (Object.keys(declaration.states).length === 0) {
    for (const [step, rows] of run.steps.entries()) {
      const bindings = bindingsOf(rows);
      const outputs = await session.run(
        stepInputs(declaration, bindings, seed, labelOf(step)),
        bindings,
      );
      results.push({ rows, outputs: copyOutputs(declaration, outputs), ...census(session) });
    }
    return results;
  }
  const chunkLength = Math.max(...run.steps);
  const buckets = [...new Set(run.steps)].filter((rows) => rows > 1 && rows < chunkLength);
  const context = await session.createGenerationContext({
    chunkLength,
    ...(buckets.length === 0 ? {} : { chunkBuckets: buckets.sort((a, b) => a - b) }),
    ...(plan.capacitySymbol === undefined || run.capacity === undefined
      ? {}
      : { bindings: { [plan.capacitySymbol]: run.capacity } }),
  });
  try {
    for (const [step, rows] of run.steps.entries()) {
      const inputs = stepInputs(declaration, bindingsOf(rows), seed, labelOf(step));
      const outputs = await session.run(inputs, undefined, { context, queryLength: rows });
      results.push({ rows, outputs: copyOutputs(declaration, outputs), ...census(session) });
    }
  } finally {
    await context.dispose();
  }
  return results;
};

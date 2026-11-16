/**
 * The corpus driver: runs a graph on a plan knowing only its declaration.
 *
 * The plan says what the declaration cannot: the values every step binds to the input symbols (the
 * rows of a row symbol, or every input symbol by name — several, or none on a graph whose shapes
 * are all static), the values of the other input symbols, and, for a graph with state slots, the
 * capacity symbol's value per run. The inputs of each step are drawn from what consumes them — an
 * i32 index read by `embedding` or `gather` (through layout ops) below the length of the axis it
 * indexes, an i32 mask all ones, f32 uniform in [-1, 1), bool all ones — and an input outside those
 * rules throws with its name. Dimension expressions are evaluated with the runtime's own dimension
 * grammar. A graph with state slots runs its steps through one generation context (the first step
 * is the prefill chunk, later steps append); a graph without runs each step directly.
 */

import { evalDim, parseDim } from "../../src/format/dims.ts";
import type { IrDeclaration, IrDim } from "../../src/format/ir.ts";
import type { FusionCounts, Session, Tensor } from "../../src/runtime/executor.ts";
import { fillWords, randomStream, signedUnit } from "./random.ts";

/**
 * One step of a run: the rows of the plan's row symbol, or — on a plan without a row symbol — the
 * value of every input symbol the plan's `bindings` leave, by name (`{}` on a graph whose input
 * shapes are all static).
 */
export type CorpusStep = number | Readonly<Record<string, number>>;

/** One run of a plan: its steps, and the capacity of the state slots. */
export type CorpusRun = {
  readonly name: string;
  /** The steps: the first is the prefill chunk; on a graph with states, later ones append. */
  readonly steps: readonly CorpusStep[];
  /** The capacity symbol's value (graphs with symbolic state slots only). */
  readonly capacity?: number;
};

/** What a corpus entry tells the driver beyond the declaration. */
export type CorpusPlan = {
  /**
   * The symbol whose value is the rows of a step (each step is then a number). Absent when every
   * step names its symbols' values; a graph with state slots needs it (a step appends rows).
   */
  readonly rowSymbol?: string;
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
  readonly step: CorpusStep;
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

const isPositiveWhole = (value: number): boolean => Number.isSafeInteger(value) && value >= 1;

/** Fails loudly when the plan does not bind what the declaration needs. */
export const assertPlanFits = (declaration: IrDeclaration, plan: CorpusPlan): void => {
  const inInputs = symbolsOf(declaration.inputs.map((input) => input.shape));
  const inStates = symbolsOf(Object.values(declaration.states).map((slot) => slot.shape));
  const { rowSymbol } = plan;
  if (rowSymbol !== undefined && !inInputs.has(rowSymbol)) {
    throw new Error(
      `the plan's row symbol '${rowSymbol}' sizes no input of the graph (input symbols: ${[...inInputs].join(", ")})`,
    );
  }
  if (rowSymbol === undefined && Object.keys(declaration.states).length > 0) {
    throw new Error(
      "the graph has state slots, so the plan needs a row symbol (each step appends that many rows)",
    );
  }
  for (const symbol of Object.keys(plan.bindings)) {
    if (!inInputs.has(symbol) || symbol === rowSymbol) {
      throw new Error(
        `the plan binds '${symbol}', which is not an input symbol other than the row symbol: bind input symbols only`,
      );
    }
  }
  if (rowSymbol !== undefined) {
    for (const symbol of inInputs) {
      if (symbol !== rowSymbol && !Object.hasOwn(plan.bindings, symbol)) {
        throw new Error(
          `input symbol '${symbol}' has no value in the plan: give it one in bindings`,
        );
      }
    }
  }
  // Without a row symbol, each step binds by name exactly the input symbols `bindings` leave.
  const left = [...inInputs].filter((symbol) => !Object.hasOwn(plan.bindings, symbol)).sort();
  for (const run of plan.runs) {
    for (const [index, step] of run.steps.entries()) {
      const where = `run '${run.name}' step ${index}`;
      if (rowSymbol !== undefined) {
        if (typeof step !== "number") {
          throw new Error(
            `${where} binds symbols by name, but the plan has the row symbol '${rowSymbol}': give the step's rows as a number`,
          );
        }
        continue;
      }
      if (typeof step === "number") {
        throw new Error(
          `${where} is a number of rows, but the plan has no row symbol: give the step the value of every input symbol by name ({} when there is none)`,
        );
      }
      const named = Object.keys(step).sort();
      if (named.join(",") !== left.join(",")) {
        throw new Error(
          `${where} binds [${named.join(", ")}], but the input symbols the plan's bindings leave are [${left.join(", ")}]: bind exactly those`,
        );
      }
      if (named.some((symbol) => !isPositiveWhole(step[symbol]))) {
        throw new Error(`${where}: every symbol needs a positive whole number`);
      }
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
      run.steps.some((step) => typeof step === "number" && !isPositiveWhole(step))
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
    // i32: an index, drawn below the length it indexes, or a mask, all ones.
    const role = i32InputRole(declaration, input.name, bindings);
    if (role.kind === "mask") {
      inputs[input.name] = { dtype: "i32", shape, data: new Int32Array(count).fill(1) };
      continue;
    }
    const { length } = role;
    if (!Number.isFinite(length) || length < 1) {
      throw new Error(
        `input '${input.name}' (i32) indexes no table with rows (it is read by no embedding or gather, or the indexed axis is empty)`,
      );
    }
    const data = new Int32Array(count);
    for (let i = 0; i < count; i += 1) data[i] = Math.floor((words[i] / 2 ** 32) * length);
    inputs[input.name] = { dtype: "i32", shape, data };
  }
  return inputs;
};

/** The declared shape of a value or a graph input. */
const shapeOf = (declaration: IrDeclaration, name: string): readonly IrDim[] => {
  const shape = Object.hasOwn(declaration.values, name)
    ? declaration.values[name].shape
    : declaration.inputs.find((input) => input.name === name)?.shape;
  if (shape === undefined)
    throw new Error(`'${name}' is neither a value nor an input of the graph`);
  return shape;
};

/** Ops that move an i32 value without changing it: an index read through them is still an index. */
const LAYOUT_OPS: ReadonlySet<string> = new Set(["reshape", "expand", "permute"]);

/** Every (node, slot) that reads `name`. */
const readersOf = (
  declaration: IrDeclaration,
  name: string,
): { readonly node: IrDeclaration["nodes"][number]; readonly slot: number }[] =>
  declaration.nodes.flatMap((node) =>
    node.ins.flatMap((input, slot) => (input === name ? [{ node, slot }] : [])),
  );

/**
 * What an i32 input is to the graph, from what consumes it:
 *
 * - an index — every reader, through layout ops, is `embedding` slot 1 (the table's rows, axis 0)
 *   or `gather` slot 1 (the source's last axis, the one gather indexes); its length is the
 *   shortest axis it indexes;
 * - a mask — every node that turns its values (through layout ops and integer arithmetic, the nodes
 *   whose outputs stay i32) into another dtype is a `cast`, and at least one casts to bool: the
 *   graph reads it as truth values (all ones = every position valid, like a bool input).
 *
 * Anything else throws with the input's name and the first reader outside the index rule.
 */
const i32InputRole = (
  declaration: IrDeclaration,
  name: string,
  bindings: Readonly<Record<string, number>>,
): { readonly kind: "index"; readonly length: number } | { readonly kind: "mask" } => {
  let length = Number.POSITIVE_INFINITY;
  let outside: string | undefined;
  const frontier = [name];
  for (let value = frontier.pop(); value !== undefined; value = frontier.pop()) {
    for (const { node, slot } of readersOf(declaration, value)) {
      if (LAYOUT_OPS.has(node.op) && slot === 0) {
        frontier.push(...node.outs);
      } else if ((node.op === "embedding" || node.op === "gather") && slot === 1) {
        const source = shapeOf(declaration, node.ins[0]);
        const axis = node.op === "embedding" ? source.slice(0, 1) : source.slice(-1);
        length = Math.min(length, evaluate(axis, bindings)[0]);
      } else {
        outside ??= `${node.op} slot ${slot}`;
      }
    }
  }
  if (outside === undefined) return { kind: "index", length };
  if (isMask(declaration, name)) return { kind: "mask" };
  throw new Error(
    `input '${name}' (i32) is read by ${outside}: the driver draws i32 inputs only as embedding indices, gather indices or masks (a value cast to bool)`,
  );
};

/** Whether every exit of `name`'s integer values is a `cast`, at least one of them to bool. */
const isMask = (declaration: IrDeclaration, name: string): boolean => {
  let toBool = false;
  const seen = new Set<string>();
  const frontier = [name];
  for (let value = frontier.pop(); value !== undefined; value = frontier.pop()) {
    if (seen.has(value)) continue;
    seen.add(value);
    for (const { node } of readersOf(declaration, value)) {
      if (
        node.outs.length > 0 &&
        node.outs.every((out) => declaration.values[out]?.dtype === "i32")
      ) {
        frontier.push(...node.outs);
        continue;
      }
      if (node.op !== "cast") return false;
      if (node.attrs["to"] === "bool") toBool = true;
    }
  }
  return toBool;
};

/** The values a step binds to the input symbols, with the plan's fixed bindings. */
export const stepBindings = (plan: CorpusPlan, step: CorpusStep): Record<string, number> => {
  if (typeof step !== "number") return { ...plan.bindings, ...step };
  if (plan.rowSymbol === undefined) {
    throw new Error(
      `a step of ${step} rows needs the plan's row symbol: name the symbol, or bind the step's symbols by name`,
    );
  }
  return { ...plan.bindings, [plan.rowSymbol]: step };
};

/** A step in the census: `rows <n>`, or its symbols' values by name (empty when it binds none). */
export const describeStep = (step: CorpusStep): string =>
  typeof step === "number"
    ? `rows ${step}`
    : Object.keys(step)
        .sort()
        .map((symbol) => `${symbol} ${step[symbol]}`)
        .join(" ");

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
  const labelOf = (index: number): string => `${run.name}/${index}`;
  const results: StepResult[] = [];
  if (Object.keys(declaration.states).length === 0) {
    for (const [index, step] of run.steps.entries()) {
      const bindings = stepBindings(plan, step);
      const outputs = await session.run(
        stepInputs(declaration, bindings, seed, labelOf(index)),
        bindings,
      );
      results.push({ step, outputs: copyOutputs(declaration, outputs), ...census(session) });
    }
    return results;
  }
  // A graph with state slots has a row symbol (assertPlanFits), so every step is a row count.
  const rowSteps = run.steps.map((step) => {
    if (typeof step !== "number") throw new Error(`run '${run.name}': a step is not a row count`);
    return step;
  });
  const chunkLength = Math.max(...rowSteps);
  const buckets = [...new Set(rowSteps)].filter((rows) => rows > 1 && rows < chunkLength);
  const context = await session.createGenerationContext({
    chunkLength,
    ...(buckets.length === 0 ? {} : { chunkBuckets: buckets.sort((a, b) => a - b) }),
    ...(plan.capacitySymbol === undefined || run.capacity === undefined
      ? {}
      : { bindings: { [plan.capacitySymbol]: run.capacity } }),
  });
  try {
    for (const [index, rows] of rowSteps.entries()) {
      const inputs = stepInputs(declaration, stepBindings(plan, rows), seed, labelOf(index));
      const outputs = await session.run(inputs, undefined, { context, queryLength: rows });
      results.push({ step: rows, outputs: copyOutputs(declaration, outputs), ...census(session) });
    }
  } finally {
    await context.dispose();
  }
  return results;
};

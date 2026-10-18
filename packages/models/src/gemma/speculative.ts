/**
 * The **internal API** that runs Gemma 4's MTP drafter once (ADR
 * [0096](../../../../docs/decisions/0096-speculative-decoding.md) decisions 1–3, stage 2 §5).
 *
 * Three parts — {@link openDrafterContext} opens the borrowing context, {@link draftOnce} takes one
 * cycle's draft, and {@link openGemma4DraftFace} bundles the two into one `DraftFace`. The
 * speculation loop (`src/generation/sequence.ts`) sees only the bundled surface (the generation
 * layer knows neither gemma4 nor Session — ADR 0083).
 *
 * MUST: not exposed on `mod.ts` / the `./gemma` subpath (ADR 0008's thin surface). A
 * GenerationContext is an internal per-sequence object not on the public surface (ADR 0083 decision
 * 3), so exposing these two would line up "functions taking values the user cannot hold".
 *
 * ## Preconditions for running the drafter (wiring with the lender = the target)
 *
 * - The drafter Session **borrows** one embedding table of the lender Session (`sharedWeights` —
 *   not a single byte is copied). The binding is done by construction in `pipeline.ts`; this file
 *   only runs it.
 * - The borrowing context binds the lender context's KV slots (l13 sliding / l14 full) **by name**.
 *   The draft's query is at logical position `P` (= `context.pastLength`) — the token the lender
 *   committed last is the frontier not yet in the KV and sits at that position. The one RoPE row
 *   the host passes is at the same position.
 * - Serialization with the lender's run / commit is the runtime's job (a borrower run takes the
 *   lender's run lease). **A draft is rejected while the lender has an uncommitted run** — the
 *   draft is taken after the commit.
 *
 * MUST: zero module side effects (no execution at import, no global mutable state — CLAUDE.md).
 */

import type { GenerationContext, PreparedModel, Session, Tensor } from "@karume/runtime";

import type { DraftCycle, DraftFace } from "../generation/speculation.ts";
import {
  GEMMA4_ROPE_LAYER_TYPES,
  GEMMA4_ROPE_PARTS,
  gemma4RopeInputName,
  gemma4RopeInputNames,
  gemma4RopeInputs,
  type Gemma4RopeSpec,
} from "./rope.ts";

/**
 * Number of draft tokens the drafter emits per cycle (= the number of MTP head steps).
 *
 * **Only assets baked with this value** are accepted (the number of graph exits is the step count
 * itself). The number `k` the speculation loop uses per cycle can be chosen within this range
 * (`1..GEMMA4_DRAFT_STEPS` — it shrinks further at the end of the budget), but the asset's step
 * count is fixed.
 */
export const GEMMA4_DRAFT_STEPS = 3;

/**
 * The graph declaration this layer reads (`PreparedModel["graph"]` = the IR itself).
 *
 * A subset of the generation layer's `GenerationGraph` (`../generation/program.ts`) is not enough —
 * it reads `external` of states and `shared` of initializers, fields only the IR proper has.
 */
type DrafterGraph = PreparedModel["graph"];

/**
 * Input names of the drafter graph (canonical in `tools/export-recipes/gemma4/export_drafter.py`).
 */
const TOKEN_INPUT = "token";
const HIDDEN_INPUT = "hidden";

/**
 * Input names of the drafter graph (**the order is the contract** — the committed token / the
 * hidden of the row that produced it / the four RoPE inputs).
 *
 * The order is checked, not only the names, to detect whether the baking side (the argument order
 * of `DrafterWrapper.forward`) and the reading side claim the same list. An asset whose order alone
 * is off passes silently as long as inputs are looked up by name.
 */
const DRAFTER_INPUTS: readonly string[] = [TOKEN_INPUT, HIDDEN_INPUT, ...gemma4RopeInputNames()];

/**
 * What running one drafter needs.
 *
 * MUST: a Session alone is not enough — neither the exit **names** (step order) nor the RoPE
 * formula can be obtained from a Session (`Session` does not expose its graph), so what admission
 * settled is bundled here and carried around. Holding them in separate fields would make "reading
 * with another asset's exit names" expressible.
 */
export type Gemma4Drafter = {
  readonly session: Session;
  /**
   * The graph declaration of the drafter container (**for estimates only** —
   * `Gemma4Pipeline.estimateSessionMemory` derives the drafter's resident weights and the borrowing
   * context's state from it).
   *
   * MUST: hold only the graph, not the `PreparedModel` (the same MUST as `Gemma4State.graph` — do
   * not keep holding the full bytes).
   */
  readonly graph: DrafterGraph;
  /** Names of the three draft exits (**step order** = the graph's declaration order). */
  readonly outputs: readonly string[];
  /**
   * The declaration that builds one position's cos / sin row (the lender's `pipelineConfig.rope`).
   */
  readonly rope: Gemma4RopeSpec;
  /** Width of the input `hidden` (= the width of the lender's final-norm hidden). */
  readonly hiddenSize: number;
};

/**
 * The material {@link admitGemma4Drafter} settles (what is decided before building the Session).
 */
export type Gemma4DrafterAdmission = {
  readonly outputs: readonly string[];
  readonly hiddenSize: number;
  /**
   * The mapping passed to `sharedWeights` of `createSession` — **borrower's initializer name →
   * lender's initializer name**. The value side is looked up with `lender.exportWeight(...)`.
   */
  readonly sharedWeights: Readonly<Record<string, string>>;
};

/**
 * Shape comparison (symbolic dimensions are compared by spelling — lender and borrower MUST use the
 * same symbol names).
 */
const sameShape = (
  left: readonly (number | string)[],
  right: readonly (number | string)[],
): boolean => left.length === right.length && left.every((dim, index) => dim === right[index]);

const showShape = (shape: readonly (number | string)[]): string => `[${shape.join(",")}]`;

/**
 * Checks the six inputs of the drafter graph (names, order, dtype, shape).
 *
 * The width of `hidden` is passed **from the lender** (the drafter's own declaration is not taken
 * as correct — an asset paired with a target of another generation fails here). The widths of the
 * four RoPE inputs are checked against `pipelineConfig.rope.<layer type>.headDim` (mixing up
 * sliding 256 and full 512 is a mistake that does not fail until the first run where the host
 * passes the table — the same reason as `assertRopeInputShapes` in `./admission.ts`).
 */
const assertDrafterInputs = (
  where: string,
  graph: DrafterGraph,
  rope: Gemma4RopeSpec,
  hiddenSize: number,
): void => {
  const names = graph.inputs.map((input) => input.name);
  if (names.length !== DRAFTER_INPUTS.length || !names.every((n, i) => n === DRAFTER_INPUTS[i])) {
    throw new Error(
      `${where}: drafter グラフの入力が [${names.join(", ")}]` +
        `（順序も含めて [${DRAFTER_INPUTS.join(", ")}] であること）`,
    );
  }
  const expected = new Map<string, { dtype: string; shape: readonly (number | string)[] }>([
    [TOKEN_INPUT, { dtype: "i32", shape: [1, 1] }],
    [HIDDEN_INPUT, { dtype: "f32", shape: [1, hiddenSize] }],
  ]);
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    for (const part of GEMMA4_ROPE_PARTS) {
      expected.set(gemma4RopeInputName(layerType, part), {
        dtype: "f32",
        shape: [1, 1, rope[layerType].headDim],
      });
    }
  }
  for (const input of graph.inputs) {
    const want = expected.get(input.name);
    if (want === undefined) throw new Error(`${where}: 想定外の drafter 入力 '${input.name}'`);
    if (input.dtype !== want.dtype || !sameShape(input.shape, want.shape)) {
      throw new Error(
        `${where}: drafter 入力 '${input.name}' が ${input.dtype} ${showShape(input.shape)}` +
          `（${want.dtype} ${showShape(want.shape)} が要る）`,
      );
    }
  }
};

/** The three draft exits (step order, i32 `[1,1,1]`). */
const assertDrafterOutputs = (where: string, graph: DrafterGraph): readonly string[] => {
  if (graph.outputs.length !== GEMMA4_DRAFT_STEPS) {
    throw new Error(
      `${where}: drafter グラフの出口が ${graph.outputs.length} 本` +
        `（配布形の drafter は ${GEMMA4_DRAFT_STEPS} 段で焼かれている — 段数を変えるのは再 export）`,
    );
  }
  for (const name of graph.outputs) {
    const value = graph.values[name];
    if (value === undefined) throw new Error(`${where}: drafter 出口 '${name}' の値情報が無い`);
    if (value.dtype !== "i32" || !sameShape(value.shape, [1, 1, 1])) {
      throw new Error(
        `${where}: drafter 出口 '${name}' が ${value.dtype} ${showShape(value.shape)}` +
          `（i32 [1,1,1] の draft token が要る）`,
      );
    }
  }
  return [...graph.outputs];
};

/**
 * Checks that the borrowed slots have **the same names and shapes as the lender's** (the borrowing
 * context binds by name, so a different spelling makes the runtime fail, but by then two Sessions
 * have already been built).
 *
 * MUST: every slot is external. With even one slot of its own, the borrower allocates and reads "an
 * empty past" by itself (values come out but carry nothing = a silent wrong value).
 */
const assertDrafterStates = (where: string, graph: DrafterGraph, target: DrafterGraph): void => {
  const names = Object.keys(graph.states);
  if (names.length === 0) {
    throw new Error(`${where}: drafter グラフに state スロットが無い（target の KV を読まない）`);
  }
  for (const name of names) {
    const slot = graph.states[name];
    if (!slot.external) {
      throw new Error(
        `${where}: drafter の state スロット '${name}' が external でない` +
          `（借り物の実体は貸し手 context にある — 自前確保は空の過去を読む）`,
      );
    }
    const lender = target.states[name];
    if (lender === undefined) {
      throw new Error(
        `${where}: drafter が読む external スロット '${name}' が target グラフに無い` +
          `（target のスロット: ${Object.keys(target.states).join(" / ")}）`,
      );
    }
    if (slot.dtype !== lender.dtype || !sameShape(slot.shape, lender.shape)) {
      throw new Error(
        `${where}: external スロット '${name}' が ${slot.dtype} ${showShape(slot.shape)}` +
          `（target 側は ${lender.dtype} ${showShape(lender.shape)}）`,
      );
    }
  }
};

/**
 * Resolves the shared initializers (declarations without bytes) to the lender's initializer names.
 *
 * The borrower's name is **the same as the lender's initializer name** (docs/ir-v2.md "Shared
 * initializers (`shared`)" — the name is the key of the data, so the mapping is the identity). If
 * the lender graph has no data under that name, fail loudly — letting it through would only make
 * `createSession` fail with "weights missing for a shared declaration", leaving no record of
 * **which table failed to be borrowed**. Storage compatibility (whether the lender's codec can run
 * in the borrower's consuming slot) is checked by the check at borrower Session construction (the
 * runtime's `resolveSharedWeights`).
 */
const resolveSharedWeights = (
  where: string,
  graph: DrafterGraph,
  target: DrafterGraph,
): Readonly<Record<string, string>> => {
  let shared: Record<string, string> = {};
  for (const name of Object.keys(graph.initializers)) {
    if (graph.initializers[name].shared === undefined) continue;
    const lender = target.initializers[name];
    if (lender === undefined || lender.shared !== undefined) {
      throw new Error(
        `${where}: drafter の共有 initializer '${name}' に対応する実体が target グラフに無い` +
          `（借り手の名前は貸し手の initializer 名と同じであること）`,
      );
    }
    shared = { ...shared, [name]: name };
  }
  if (Object.keys(shared).length === 0) {
    throw new Error(
      `${where}: drafter グラフに共有 initializer が 1 本も無い` +
        `（target の埋め込み表を借りる宣言が焼かれていない）`,
    );
  }
  return shared;
};

/**
 * The check of whether the drafter container is "an MTP head that can pair with this target"
 * (**before fetching the weight parts**).
 *
 * It checks five things — (1) the six inputs (including order) (2) the three exits (3) every slot
 * is external with the lender's names and shapes (4) the shared initializers resolve to the
 * lender's initializers (5) the only symbol is the lender's capacity symbol. Each exists to reject
 * "a drafter paired with a target of another generation" before GB-scale weights are downloaded.
 *
 * NOTE: `requiredLimits` is not checked — the drafter is in **the same quant preset** as the target
 * (the manifest's `quants.<name>.weights` points at both), so there is only one declaration
 * (checking it before download is the job of the admission closure of
 * `Gemma4Pipeline.fromPretrained`).
 */
export const admitGemma4Drafter = (
  where: string,
  graph: DrafterGraph,
  target: {
    readonly graph: DrafterGraph;
    readonly rope: Gemma4RopeSpec;
    readonly hiddenSize: number;
    readonly capacitySymbol: string;
  },
): Gemma4DrafterAdmission => {
  assertDrafterInputs(where, graph, target.rope, target.hiddenSize);
  const outputs = assertDrafterOutputs(where, graph);
  assertDrafterStates(where, graph, target.graph);
  const sharedWeights = resolveSharedWeights(where, graph, target.graph);
  if (graph.symbols.length !== 1 || graph.symbols[0] !== target.capacitySymbol) {
    throw new Error(
      `${where}: drafter グラフの記号が [${graph.symbols.join(", ")}]` +
        `（貸し手から継承する容量記号 '${target.capacitySymbol}' 1 本であること）`,
    );
  }
  return { outputs, hiddenSize: target.hiddenSize, sharedWeights };
};

/**
 * Opens the drafter's context that borrows the lender's (the target's) generation context.
 *
 * The borrower does not allocate a single byte of its own KV (it binds the lender's slot buffers).
 * `chunkLength` is exactly 1 and `bindings` are inherited from the lender (both are runtime
 * checks).
 *
 * MUST: **close the borrower first**. While the borrower is alive, the lender context's `dispose()`
 * fails loudly (the borrower's bind groups hold the lender's slots).
 */
export const openDrafterContext = (
  drafter: Gemma4Drafter,
  target: GenerationContext,
): Promise<GenerationContext> =>
  drafter.session.createGenerationContext({ chunkLength: 1, borrow: target });

/**
 * One cycle's input (`P = context.pastLength` = the number of rows in the KV).
 *
 * The pair is "the last committed token and the hidden of **the row that produced it**" — the same
 * pairing as the drafter's own recursion between steps (a token and the hidden of the step that
 * produced it go to the next step), and the torch drafter's golden was also taken with this
 * pairing.
 *
 * MUST: it is the generation layer's `DraftCycle` (`../generation/speculation.ts`) **itself**. With
 * a copied type, the type check would keep passing when the meaning of a field changed in only one
 * of them.
 */
export type Gemma4DraftCycle = DraftCycle;

/**
 * Opens the `DraftFace` for one lender context (the hook the speculation loop — `speculative.open`
 * of `createGenerationSequence` — calls once when the sequence is created).
 *
 * It bundles three things — the step count (the number of exits baked into the distribution),
 * {@link draftOnce}, and the return of the borrowing context. The generation layer knows neither
 * gemma4 nor Session, so this closure owns the borrowing context's lifetime (the `dispose` order =
 * borrower → lender is the sequence side's MUST).
 */
export const openGemma4DraftFace = async (
  drafter: Gemma4Drafter,
  target: GenerationContext,
): Promise<DraftFace> => {
  const borrowed = await openDrafterContext(drafter, target);
  return {
    steps: drafter.outputs.length,
    draft: (cycle: DraftCycle): Promise<Int32Array<ArrayBuffer>> =>
      draftOnce(drafter, borrowed, cycle),
    dispose: (): Promise<void> => borrowed.dispose(),
  };
};

/**
 * Takes one cycle's draft (emits k = 3 steps' tokens in one run).
 *
 * A borrower run takes the lender's run lease, so it is serialized with the lender's run / commit /
 * rewind. **It is rejected while the lender has an uncommitted run** (the draft comes after the
 * commit). The borrower does not move the logical length at all (`state_append` count 0 = not a
 * single row is written to the KV), so there is nothing to roll back when the draft is rejected.
 *
 * MUST: `hidden` is **the row that produced `token`** (position `P−1`) — passing another row still
 * matches the shape, so there is no exception, and only the draft quality quietly drops (all this
 * layer can detect is the width).
 */
export const draftOnce = async (
  drafter: Gemma4Drafter,
  borrowed: GenerationContext,
  cycle: Gemma4DraftCycle,
): Promise<Int32Array<ArrayBuffer>> => {
  const where = "gemma4 draftOnce";
  const { token, hidden, position } = cycle;
  if (!Number.isSafeInteger(token) || token < 0) {
    throw new Error(`${where}: 直前 token ${token} が非負整数でない`);
  }
  if (!Number.isSafeInteger(position) || position < 0) {
    throw new Error(`${where}: 位置 ${position} が非負整数でない`);
  }
  if (hidden.length !== drafter.hiddenSize) {
    throw new Error(`${where}: hidden の要素数 ${hidden.length} が幅 ${drafter.hiddenSize} と違う`);
  }
  const inputs: Record<string, Tensor> = {
    [TOKEN_INPUT]: { dtype: "i32", shape: [1, 1], data: Int32Array.of(token) },
    [HIDDEN_INPUT]: { dtype: "f32", shape: [1, drafter.hiddenSize], data: hidden },
    // The one row at position P (no PLE needed — the drafter reads neither the main table nor the
    // per-layer table).
    ...gemma4RopeInputs(drafter.rope, [position]),
  };
  const outputs = await drafter.session.run(inputs, {}, { context: borrowed, queryLength: 1 });
  const draft = new Int32Array(new ArrayBuffer(drafter.outputs.length * 4));
  drafter.outputs.forEach((name, step) => {
    const tensor = outputs[name];
    if (tensor === undefined) throw new Error(`${where}: 出口 '${name}' が run の結果に無い`);
    if (tensor.dtype !== "i32" || tensor.data.length !== 1) {
      throw new Error(
        `${where}: 出口 '${name}' が ${tensor.dtype} ${tensor.data.length} 要素` +
          `（i32 1 要素の draft token が要る）`,
      );
    }
    draft[step] = tensor.data[0];
  });
  return draft;
};

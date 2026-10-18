/**
 * The **static wiring** of generation (`GenerationProgram` of ADR 0083 decision 1). It is
 * **pipeline-independent shared code**, so it lives in `src/generation/` next to `greedy.ts` /
 * `sampler.ts`.
 *
 * ## Why split into two layers
 *
 * The current `GreedySpec` (`greedy.ts`) holds the static wiring (graph input / output names, chunk
 * length, position limit) and the request (prompt, number of tokens) in one type. A multi-turn
 * conversation is "issuing different requests many times on the same wiring", so without this split
 * the surface would **rewrite the whole wiring every time**. The program is immutable, and only
 * {@link GenerationSequence} (`sequence.ts`) owns a mutable lifetime.
 *
 * The wiring type is split in two again — every field the generation loop reads is in
 * {@link GenerationWiring} (internal), and only the numbers a consumer reads when running
 * `sequence()` are in {@link GenerationProgram} (public, frozen).
 *
 * ## MUST: verify the whole wiring at setup
 *
 * A mixed-up name raises **no exception and no warning** (`readToken` in `greedy.ts` checks dtype
 * and shape for the same reason). If an output name points at another output, "a plausible token
 * sequence" comes out as long as the shape matches. If one graph input goes to a run unwired, the
 * diagnosis appears far from the real cause. So {@link createGenerationProgram} checks all of the
 * following **against the graph**:
 *
 * - that the names exist (2 inputs + the derived input names + 2 outputs)
 * - shapes and dtypes (`[1,M]` i32, `[R]` i32, `[1,R,V]` f32, `[1,R,H]` f32)
 * - **complete coverage of the graph inputs** (no input left unwired by the program, no extra
 *   names)
 * - symbols (exactly one symbol, the capacity symbol, is not determined by input shapes)
 *
 * MUST: the program holds no mutable state (ADR 0083 decision 1).
 */

import { assertChunkBuckets, type RunInputs } from "@karume/runtime";

/**
 * The graph surface the program uses to verify its wiring (a subset of `PreparedModel["graph"]`).
 *
 * MUST: do not require `PreparedModel["graph"]` itself. The whole IR (nodes / initializers /
 * states) is not needed to verify the wiring, and requiring it would force unit tests to **build a
 * real IR container** (the same reason `GreedySession` in `greedy.ts` is a narrow interface). That
 * the real `IrGraph` satisfies this surface is pinned by a type check (on the test side).
 */
export type GenerationGraph = {
  readonly symbols: readonly string[];
  readonly inputs: readonly {
    readonly name: string;
    readonly dtype: string;
    readonly shape: readonly (number | string)[];
  }[];
  readonly outputs: readonly string[];
  readonly values: Readonly<
    Record<string, { readonly dtype: string; readonly shape: readonly (number | string)[] }>
  >;
};

/**
 * Run-time knobs that {@link DerivedRunInputs.derive} receives (not static wiring but what applies
 * to **that one run**).
 *
 * MUST: the contract is **best-effort** — nothing breaks if an implementation ignores `signal`
 * (ignoring it only makes the abort take effect "after the current derived inputs are finished").
 * The generation loop checks `signal` itself after `derive`'s `await` and stops issuing the run, so
 * the correctness of the abort is not delegated to this slot.
 */
export type DeriveInputsOptions = {
  /**
   * This generation's abort (`GenerationRequest.signal` passed down as is).
   *
   * In a distribution whose derived inputs come from GB-scale lazy loads (gemma4's PLE — assets of
   * the `model` container, ADR 0085), not looking at it gives the shape "pressing stop does not
   * return until one block (≤ 32 MiB) has been read".
   */
  readonly signal?: AbortSignal;
};

/**
 * The slot for host-derived per-chunk inputs (ADR 0083 decision 1's "how to build model-specific
 * inputs is knowledge of the models side" — gemma4's PLE `per_layer_inputs[1,M,35,256]` is one
 * implementation of it, ADR 0085).
 *
 * The precedents are SBV2's relative-position tables / Anima's raw rope tables; in both "the host
 * builds the graph input". This is **the slot that plugs that producer into the program**, so the
 * generation loop only passes the token id sequence.
 */
export type DerivedRunInputs = {
  /**
   * Names of the graph inputs this slot supplies.
   *
   * MUST: not optional. Having the names declared is the only way to check "complete coverage of
   * the graph inputs" at setup (a function's return value is unknown until it is called), and a
   * mismatch between the declaration and the actual return value fails at run time (the contract of
   * {@link DerivedRunInputs.derive}).
   */
  readonly names: readonly string[];
  /**
   * The token id sequence and the **absolute position sequence** of one physical chunk → extra
   * inputs.
   *
   * Both cover **the physical row count** (prefill including pad rows, decode one row), with the
   * same 0 as `input_ids` in pad rows — to give the same values as when the lookup ran inside the
   * graph (the PLE `gather` doc and the value contract of ADR 0066 addendum 6). Positions follow the
   * same convention, with 0 in pad rows.
   *
   * `positions` is passed because position-dependent host-derived inputs (gemma4's RoPE cos / sin
   * — `gemma/rope.ts`) use this slot. Passing `position_ids` **as a graph input** no longer exists,
   * so this is the only destination of positions.
   *
   * MUST: **a pure-function slot** (the same value for the same `(ids, positions)`, independent of
   * call order). The keys of the return value match {@link DerivedRunInputs.names} exactly.
   * `options` carries only circumstances that do not affect the values (abort), so it is compatible
   * with this MUST.
   */
  readonly derive: (
    ids: readonly number[],
    positions: readonly number[],
    options?: DeriveInputsOptions,
  ) => Promise<RunInputs>;
};

/** Specification for {@link createGenerationProgram} (with the graph — all verification is here). */
export type GenerationProgramSpec = {
  /** The graph to check the wiring against (`PreparedModel.graph` can be passed as is). */
  readonly graph: GenerationGraph;
  /** Name of the graph input that receives the token id sequence (`[1,M]` i32). */
  readonly inputIds: string;
  /**
   * Name of the graph input that receives the **sequence** of row indices to select (`[R]` i32 —
   * the row selection of ADR 0068 decision 4).
   *
   * R is a symbol and is exactly the number of rows selected in that run (this input's element
   * count is the only source that binds R). Ordinary prefill / decode has R=1 (the one last valid
   * row), and the values and token sequence are identical to before. Baking R as the fixed number
   * 1 would keep speculation's verify (a run that checks the draft rows at once) from running on
   * the same graph.
   */
  readonly lastRow: string;
  /** Graph output name of the selected rows' logits (`[1,R,V]` f32 — ADR 0083 decision 6). */
  readonly logits: string;
  /**
   * Name of the graph output with the selected rows' **final-norm hidden** (`[1,R,H]` f32).
   *
   * MUST: not optional. The drafter takes "the hidden of the row the target actually placed" as
   * input, so a distribution without this cannot build speculation. Making it optional gives the
   * shape "a program can be built even for a graph without hidden", and the gap does not fail until
   * the step that connects the drafter (by which point the distribution has to be rebuilt).
   */
  readonly hidden: string;
  /** Rows of a fixed-length prefill chunk (ADR 0066 decision 4 — a plan-time context constant). */
  readonly chunkLength: number;
  /**
   * Physical chunk row counts allowed as prefill shapes **in addition to** `chunkLength` (ADR 0066
   * decision 4 / addendum 10, buckets). Omitted / empty = none added (one prefill shape + the
   * decode shape, as before).
   *
   * Padding a short prompt to `chunkLength` rows wastes the pad rows' work outright (row-local
   * linear / pointwise / norm scale with the physical row count). Declaring buckets lets each chunk
   * choose the smallest bucket ≥ `queryLength` as its physical row count (`physicalChunkRows` in
   * `sequence.ts`).
   *
   * MUST: checking the acceptance set (integers ≥ 2 and < `chunkLength`, strictly increasing) is
   * left to runtime's single `assertChunkBuckets` — copying the same rule here would let the set the
   * context allows and the set the caller chooses from grow apart.
   */
  readonly chunkBuckets?: readonly number[];
  /**
   * **Exclusive upper bound** of the absolute positions that can be looked up (positions are
   * `0..maxPosition-1` — the position limit the model declares).
   *
   * MUST: not optional (the same reason as `maxPosition` in `greedy.ts` — a position beyond the
   * limit raises no exception, and attention at untrained positions is folded into "a plausible
   * token id").
   */
  readonly maxPosition: number;
  /**
   * The **default** capacity of the full slots (`pastLength + queryLength ≤ capacity` — ADR 0067
   * decision 4, check (4)).
   *
   * MUST: not optional. The runtime also rejects an overflow, but only **right before encoding the
   * run**, where the fact "the conversation does not fit", **which the host has to decide on**, is
   * buried in a generic message (ADR 0083 decision 10). The sequence checks with this value
   * **before** the run and fails with a dedicated type.
   *
   * NOTE: the capacity actually used can be chosen per sequence (= context) (`capacity` of
   * `createGenerationSequence`). The program holds the **default**; the context's physical
   * allocation is decided by that sequence's value.
   */
  readonly capacity: number;
  /** Vocabulary size (the last axis of the logits exit — cross-checked against the graph). */
  readonly vocabSize: number;
  /** The stop token set (ADR 0083 decision 8 — empty means no EOS stop). */
  readonly stopTokens: readonly number[];
  /**
   * The **symbol name** of the full-slot capacity (the spelling used at the binding point of
   * `createGenerationContext`).
   *
   * MUST: hold no binding **value**. The capacity can be chosen per sequence, so also holding the
   * value on the wiring side would be an independently updated duplicate ("which is the real
   * capacity, the program's `capacity` or `bindings`?") (CLAUDE.md's ban on derived state). The
   * symbol is the asset's spelling and immutable; the value is a run-time knob.
   *
   * MUST: never point at a symbol determined by input shapes (checked by
   * {@link createGenerationProgram}) — such a symbol is determined by the run's inputs, so it
   * splits from the context-side binding and the run rejects it.
   */
  readonly capacitySymbol: string;
  /** Host-derived per-chunk inputs (omitted for distributions without them). */
  readonly derivedInputs?: DerivedRunInputs;
  /**
   * Names of the graph inputs that **the Session side plugs in as resident inputs** (gemma4's PLE
   * GPU residency — ADR 0085 addendum (2026-09-19), GPU residency).
   *
   * Only names are declared; no values are built (the producer is the side that issues the run =
   * the `Session` wrapper). The slot is still needed to check "complete coverage of the graph
   * inputs" at setup; without listing them here, **inputs the host does not build** fail as "not
   * wired".
   *
   * MUST: do not overlap with {@link DerivedRunInputs.names} (two producers claiming the same
   * input). An overlap fails in {@link assertInputCoverage}.
   */
  readonly residentInputs?: readonly string[];
};

/**
 * The verified static wiring (only {@link createGenerationProgram} builds it). An **internal
 * type**, not on the public surface (the public surface is {@link GenerationProgram}).
 *
 * MUST: when adding a field, add its verification to {@link createGenerationProgram} at the same
 * time — an unverified wiring field silently breaks this type's reason to exist, "verify the whole
 * wiring at setup".
 */
export type GenerationWiring = Omit<GenerationProgramSpec, "graph" | "chunkBuckets"> & {
  /**
   * The verified prefill buckets (**not optional** — a spec without a declaration is folded into
   * an empty array).
   *
   * Carrying it around as optional would make the side that picks physical row counts
   * (`sequence.ts`) write `?? []` every time, and missing one of them would raise no exception, just
   * "buckets silently have no effect".
   */
  readonly chunkBuckets: readonly number[];
  /**
   * The last axis H of the hidden exit (a value **derived** from the declared shape of
   * {@link GenerationProgramSpec.hidden}).
   *
   * MUST: the caller does not declare it (unlike `vocabSize`, there is nothing on the asset side to
   * check it against). The graph is the only source, so accepting a declaration would make "the
   * wiring's H" and "the graph's H" an independently updated duplicate.
   */
  readonly hiddenSize: number;
  /**
   * Name of the row-count symbol R (**derived** from the declared shape `[R]` of the `last_row`
   * input). In a run, the element count of `last_row` binds it, but an estimate without a run
   * (`bindings` of `estimateGraphMemory`) has no input shapes, so R = 1 is passed explicitly under
   * this name.
   */
  readonly rowSymbol: string;
};

/**
 * The **read surface** of the verified static wiring (public — `Gemma4Pipeline.program`).
 *
 * It exposes only "the numbers you need to read when running `sequence()` yourself". Graph input /
 * output names, the capacity symbol and `derivedInputs` are **internal wiring**
 * ({@link GenerationWiring}); exposing them would mean (1) consumers reading them have no use for
 * them (the Session they are wired to is not on the public surface), and (2) replacing `derive` or
 * altering symbols becomes writable from the public surface — being verified is exactly what
 * `GenerationProgram` means, so a writable path breaks that meaning.
 *
 * MUST: {@link generationProgramFace} returns it frozen (`stopTokens` is a frozen copy).
 */
export type GenerationProgram = {
  /** Rows of a fixed-length prefill chunk (ADR 0066 decision 4 — the value this pipeline uses). */
  readonly chunkLength: number;
  /**
   * Physical chunk row counts used as prefill shapes in addition to `chunkLength` (strictly
   * increasing; empty = none).
   *
   * A short chunk takes the smallest of these ≥ `queryLength` as its physical row count
   * (`chunkLength` if none). It is the only value from which a side running `sequence()` itself can
   * read "how many rows this prompt length lands on".
   */
  readonly chunkBuckets: readonly number[];
  /** Exclusive upper bound of absolute positions (`0..maxPosition-1` — the model's declaration). */
  readonly maxPosition: number;
  /**
   * The **default** capacity of the full slots (`pastLength + queryLength ≤ capacity`).
   *
   * The capacity actually used can be chosen per sequence (`GenerationSequence.capacity` is the
   * value **for that conversation**).
   */
  readonly capacity: number;
  /** Vocabulary size (it decides the range of `prompt` token ids). */
  readonly vocabSize: number;
  /** The stop token set (ADR 0083 decision 8 — empty means no EOS stop). */
  readonly stopTokens: readonly number[];
};

/**
 * Internal wiring → public read surface (**frozen**).
 *
 * MUST: make `stopTokens` a frozen **copy**. Exposing the same array would let a consumer's
 * `sort()` / `length = 0` rewrite the generation loop's stop set itself (silent degradation without
 * an exception, showing up as "generation that does not stop at EOS").
 */
export const generationProgramFace = (wiring: GenerationWiring): GenerationProgram =>
  Object.freeze({
    chunkLength: wiring.chunkLength,
    // A frozen copy for the same reason as `stopTokens` (a consumer's `sort()` / `length = 0` must
    // not rewrite how physical row counts are chosen).
    chunkBuckets: Object.freeze([...wiring.chunkBuckets]),
    maxPosition: wiring.maxPosition,
    capacity: wiring.capacity,
    vocabSize: wiring.vocabSize,
    stopTokens: Object.freeze([...wiring.stopTokens]),
  });

const assertPositiveInteger = (value: number, where: string): void => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${where} ${value} が 1 以上の整数でない`);
  }
};

/** Renders a shape for a diagnostic (symbolic dimensions are printed as spelled). */
const showShape = (shape: readonly (number | string)[]): string => `[${shape.join(",")}]`;

const findInput = (graph: GenerationGraph, name: string, role: string) => {
  const spec = graph.inputs.find((input) => input.name === name);
  if (spec === undefined) {
    throw new Error(
      `${role} '${name}' がグラフ入力に無い（実在するのは ` +
        `${graph.inputs.map((input) => input.name).join(" / ")}）`,
    );
  }
  return spec;
};

const assertDtype = (actual: string, expected: string, where: string): void => {
  if (actual !== expected) throw new Error(`${where} の dtype が ${actual}（${expected} でない）`);
};

/**
 * Checks that it is a `[1, M]` i32 input (M a symbol).
 *
 * MUST: check that the second dimension is a **symbol** too. With a fixed number the prefill shape
 * (`M = chunkLength`) and the decode shape (`M = 1`) cannot both run on the same graph (ADR 0066
 * decision 4), and the mismatch would surface as a diagnosis far from the real cause: "the shape
 * does not match on the first decode".
 */
const assertRowInput = (graph: GenerationGraph, name: string, role: string): string => {
  const spec = findInput(graph, name, role);
  assertDtype(spec.dtype, "i32", `${role} '${name}'`);
  const lengthSymbol = spec.shape[1];
  if (spec.shape.length !== 2 || spec.shape[0] !== 1 || typeof lengthSymbol !== "string") {
    throw new Error(
      `${role} '${name}' の shape ${showShape(spec.shape)} が [1,<記号>] でない` +
        `（prefill 形と decode 形を同じグラフで回せない）`,
    );
  }
  return lengthSymbol;
};

/**
 * Checks that it is a row-selection input (`[R]` i32) and returns its **symbol name R**.
 *
 * MUST: check that the dimension is a **symbol** too. With the fixed number 1, only "1 run = 1 row"
 * is baked in, and speculation's verify (a run that checks the draft rows together) cannot run on
 * the same graph. An R=1 run (ordinary prefill / decode) folds into the same shape as before with a
 * one-element input.
 *
 * NOTE: R appearing in an input shape is also a premise of {@link assertSymbols} — the row count is
 * determined by the run's input (the length of this sequence) and, unlike the capacity symbol,
 * needs no context binding.
 */
const assertLastRowInput = (graph: GenerationGraph, name: string): string => {
  const spec = findInput(graph, name, "last_row 入力");
  assertDtype(spec.dtype, "i32", `last_row 入力 '${name}'`);
  const rowSymbol = spec.shape[0];
  if (spec.shape.length !== 1 || typeof rowSymbol !== "string") {
    throw new Error(
      `last_row 入力 '${name}' の shape ${showShape(spec.shape)} が [<記号>] でない` +
        `（R=1 の prefill / decode と R>1 の verify を同じグラフで回せない）`,
    );
  }
  return rowSymbol;
};

/**
 * Looks up the value info of a name listed among the graph **outputs**.
 *
 * MUST: check **that it is listed among the graph outputs** too. A name that exists only as a node
 * output is not returned by the run, which turns into a run-time exception far from the real cause:
 * "output '…' is missing".
 */
const findOutputValue = (graph: GenerationGraph, name: string, role: string) => {
  if (!graph.outputs.includes(name)) {
    throw new Error(
      `${role} '${name}' がグラフ出力に無い（実在するのは ${graph.outputs.join(" / ")}）`,
    );
  }
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`${role} '${name}' の値情報がグラフに無い`);
  }
  return graph.values[name];
};

/**
 * Checks that it is the logits exit (`[1,R,V]` f32).
 *
 * MUST: check that the second dimension is **the same symbol as `last_row`** too. With a different
 * symbol "the rows selected" and "the rows returned" are decided separately, and the run passes as
 * long as the shapes match (verify passes `R` rows and only 1 comes back, with no exception).
 */
const assertLogitsOutput = (
  graph: GenerationGraph,
  name: string,
  vocabSize: number,
  rowSymbol: string,
): void => {
  const info = findOutputValue(graph, name, "logits 出口");
  assertDtype(info.dtype, "f32", `logits 出口 '${name}'`);
  if (
    info.shape.length !== 3 ||
    info.shape[0] !== 1 ||
    info.shape[1] !== rowSymbol ||
    info.shape[2] !== vocabSize
  ) {
    throw new Error(
      `logits 出口 '${name}' の shape ${showShape(info.shape)} が ` +
        `[1,${rowSymbol},${vocabSize}] でない` +
        `（選んだ**行**だけの出口であること — ADR 0083 決定 6）`,
    );
  }
};

/**
 * Checks that it is the hidden exit (`[1,R,H]` f32) and returns H.
 *
 * No declaration like `vocabSize` is accepted because there is nothing on the asset side to check H
 * against (the vocabulary size is the reference for cross-checking the tokenizer and the PLE index,
 * but only the graph has the hidden width). So this checks only "a fixed positive-integer
 * dimension" — left as a symbol, the width would vary per run and could not connect to the
 * drafter's weights.
 */
const assertHiddenOutput = (graph: GenerationGraph, name: string, rowSymbol: string): number => {
  const info = findOutputValue(graph, name, "hidden 出口");
  assertDtype(info.dtype, "f32", `hidden 出口 '${name}'`);
  const hiddenSize = info.shape[2];
  if (
    info.shape.length !== 3 ||
    info.shape[0] !== 1 ||
    info.shape[1] !== rowSymbol ||
    typeof hiddenSize !== "number"
  ) {
    throw new Error(
      `hidden 出口 '${name}' の shape ${showShape(info.shape)} が [1,${rowSymbol},<H>] でない` +
        `（選んだ行の最終 norm 後 hidden であること）`,
    );
  }
  assertPositiveInteger(hiddenSize, `hidden 出口 '${name}' の H`);
  return hiddenSize;
};

/**
 * Checks that the graph inputs are wired exactly, nothing missing and nothing extra.
 *
 * MUST: check both directions. A **missing** one makes the run fail with "no buffer" (far from the
 * real cause), and an **extra** one (a declared name that is not in `derivedInputs.names`) creates
 * an input ignored on every run.
 */
const assertInputCoverage = (graph: GenerationGraph, wired: readonly string[]): void => {
  const declared = new Set(wired);
  if (declared.size !== wired.length) {
    throw new Error(`結線した入力名に重複がある（${wired.join(" / ")}）`);
  }
  const actual = new Set(graph.inputs.map((input) => input.name));
  const missing = [...actual].filter((name) => !declared.has(name));
  if (missing.length > 0) {
    throw new Error(
      `グラフ入力 ${missing.join(" / ")} が結線されていない` +
        `（ホスト由来の入力は derivedInputs で供給する）`,
    );
  }
  const extra = wired.filter((name) => !actual.has(name));
  if (extra.length > 0) {
    throw new Error(`結線した ${extra.join(" / ")} がグラフ入力に無い`);
  }
};

/**
 * Checks that every symbol is determined.
 *
 * Symbols in input shapes are determined by the run's inputs; for the rest (the states' capacity
 * symbol) the binding of `createGenerationContext` is the only source (the MUST of
 * `resolveBindings` — states are not a binding source). A symbol missed by both would go unnoticed
 * until context creation, so it fails here.
 */
const assertSymbols = (graph: GenerationGraph, capacitySymbol: string): void => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) if (typeof dim === "string") fromInputs.add(dim);
  }
  if (!graph.symbols.includes(capacitySymbol)) {
    throw new Error(
      `容量記号 ${capacitySymbol} がグラフの symbols [${graph.symbols.join(", ")}] に無い`,
    );
  }
  // Choosing an input-derived symbol as the capacity symbol makes the run's inputs and the
  // context's binding decide the same symbol with different values (the runtime rejects the split)
  // — it is a spelling mix-up, so fail when the wiring is built.
  if (fromInputs.has(capacitySymbol)) {
    throw new Error(
      `容量記号 ${capacitySymbol} は入力 shape から決まる記号である` +
        `（state スロットの容量記号は入力に現れない 1 本 — ADR 0066 追記 7）`,
    );
  }
  const unresolved = graph.symbols.filter(
    (symbol) => !fromInputs.has(symbol) && symbol !== capacitySymbol,
  );
  if (unresolved.length > 0) {
    throw new Error(
      `記号 ${unresolved.join(" / ")} が入力 shape からも容量記号からも決まらない` +
        `（state スロットの容量記号は 1 本だけ — ADR 0066 追記 7）`,
    );
  }
};

/**
 * Settles the static wiring against the graph (**the only entry** — no program that bypasses
 * verification can be built).
 *
 * MUST: fail before touching the GPU (the arguments are only the graph and numbers). A wiring error
 * after a 3.7GiB load and one right after `prepareContainer` differ in diagnostic value.
 */
export const createGenerationProgram = (spec: GenerationProgramSpec): GenerationWiring => {
  const { graph } = spec;
  assertPositiveInteger(spec.chunkLength, "chunkLength");
  // The acceptance set's source of truth is the runtime (`createGenerationContext` rejects with
  // the same function). Calling it here is to "fail before the context is built"; this holds no
  // rule of its own.
  assertChunkBuckets(spec.chunkBuckets, spec.chunkLength);
  assertPositiveInteger(spec.maxPosition, "maxPosition");
  assertPositiveInteger(spec.capacity, "capacity");
  assertPositiveInteger(spec.vocabSize, "vocabSize");
  spec.stopTokens.forEach((token, index) => {
    if (!Number.isSafeInteger(token) || token < 0 || token >= spec.vocabSize) {
      throw new Error(`stopTokens[${index}] ${token} が語彙 0..${spec.vocabSize - 1} の外`);
    }
  });

  const lengthSymbol = assertRowInput(graph, spec.inputIds, "token id 入力");
  const rowSymbol = assertLastRowInput(graph, spec.lastRow);
  // MUST: the input length (physical rows) and the selected row count are different symbols.
  // With the same symbol each shape check passes, but the first prefill run (physical rows =
  // chunk, selected rows = 1) binds one symbol to two values and is rejected — decode (both 1)
  // passes by chance, so detecting the miswiring is delayed until after the GPU build.
  if (lengthSymbol === rowSymbol) {
    throw new Error(
      `token id 入力 '${spec.inputIds}' の長さと last_row 入力 '${spec.lastRow}' の行数が` +
        `同じ記号 ${rowSymbol} を使っている（入力長と選択行数は run ごとに別の値をとる）`,
    );
  }
  // Wiring the same name to both exits passes both shape checks only when V = H
  // (the drafter eats logits as hidden = a mix-up without an exception).
  if (spec.logits === spec.hidden) {
    throw new Error(`logits 出口と hidden 出口が同じ名前 '${spec.logits}' を指している`);
  }
  assertLogitsOutput(graph, spec.logits, spec.vocabSize, rowSymbol);
  const hiddenSize = assertHiddenOutput(graph, spec.hidden, rowSymbol);
  assertInputCoverage(graph, [
    spec.inputIds,
    spec.lastRow,
    ...(spec.derivedInputs?.names ?? []),
    ...(spec.residentInputs ?? []),
  ]);
  assertSymbols(graph, spec.capacitySymbol);

  return {
    inputIds: spec.inputIds,
    lastRow: spec.lastRow,
    logits: spec.logits,
    hidden: spec.hidden,
    hiddenSize,
    rowSymbol,
    chunkLength: spec.chunkLength,
    // A frozen copy: the wiring is an immutable object, so if the caller's array were rewritten
    // later, "the set the context allows" and "the set physical row counts are chosen from" would
    // split during execution.
    chunkBuckets: Object.freeze([...(spec.chunkBuckets ?? [])]),
    maxPosition: spec.maxPosition,
    capacity: spec.capacity,
    vocabSize: spec.vocabSize,
    stopTokens: [...spec.stopTokens],
    capacitySymbol: spec.capacitySymbol,
    ...(spec.derivedInputs === undefined ? {} : { derivedInputs: spec.derivedInputs }),
    ...(spec.residentInputs === undefined ? {} : { residentInputs: [...spec.residentInputs] }),
  };
};

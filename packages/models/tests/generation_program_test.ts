// Setup validation tests of the static wiring (`src/generation/program.ts`). Needs neither a GPU nor real assets.
//
// What is pinned here is the contract of ADR 0083 decision 1 itself, "a program is an immutable object that
// validates **all of its wiring** at setup". A mixed-up name or a mismatched shape **raises no exception** at run
// time, or raises one far from the real cause (grabbing another output of a matching shape silently returns a
// plausible token sequence), so a field that escapes validation becomes an entry point for silent degradation.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import type { PreparedModel } from "@karume/runtime";
import {
  createGenerationProgram,
  type GenerationGraph,
  generationProgramFace,
  type GenerationProgramSpec,
} from "../src/generation/program.ts";

const VOCAB = 64;
const HIDDEN_SIZE = 8;
const IDS = "input_ids";
const LAST_ROW = "last_row";
const LOGITS = "logits";
const HIDDEN = "hidden";
const DERIVED = "per_layer_inputs";

type GraphInput = GenerationGraph["inputs"][number];

/**
 * The product graph (a shrunk version of gemma4's real shape) — only the presence of derived inputs is selectable.
 *
 * Row selection is `[R]` (a symbol), and the second dimension of the 2 outputs is **the same R** (R=1 is ordinary
 * prefill / decode; R>1 is the speculative verify).
 */
const graphOf = (options: { readonly derived?: boolean } = {}): GenerationGraph => ({
  symbols: ["C", "M", "R"],
  inputs: [
    { name: IDS, dtype: "i32", shape: [1, "M"] },
    ...(options.derived === false
      ? []
      : [{ name: DERIVED, dtype: "f32", shape: [1, "M", 2, 3] } satisfies GraphInput]),
    { name: LAST_ROW, dtype: "i32", shape: ["R"] },
  ],
  outputs: [LOGITS, HIDDEN],
  values: {
    [LOGITS]: { dtype: "f32", shape: [1, "R", VOCAB] },
    [HIDDEN]: { dtype: "f32", shape: [1, "R", HIDDEN_SIZE] },
  },
});

/** A graph with exactly one of `graphOf()`'s `values` replaced (a prop for creating shape regressions). */
const graphWithValue = (
  name: string,
  info: { readonly dtype: string; readonly shape: readonly (number | string)[] },
): GenerationGraph => ({
  ...graphOf(),
  values: { ...graphOf().values, [name]: info },
});

const specOf = (override: Partial<GenerationProgramSpec> = {}): GenerationProgramSpec => ({
  graph: graphOf(),
  inputIds: IDS,
  lastRow: LAST_ROW,
  logits: LOGITS,
  hidden: HIDDEN,
  chunkLength: 4,
  maxPosition: 128,
  capacity: 64,
  vocabSize: VOCAB,
  stopTokens: [7],
  capacitySymbol: "C",
  derivedInputs: { names: [DERIVED], derive: () => Promise.resolve({}) },
  ...override,
});

test("createGenerationProgram: passes the product shape's wiring as is and does not keep the graph", () => {
  const program = createGenerationProgram(specOf());
  assertEquals(program.inputIds, IDS);
  assertEquals(program.logits, LOGITS);
  assertEquals(program.hidden, HIDDEN);
  // H is derived from the graph (not accepted as a declaration = not held twice).
  assertEquals(program.hiddenSize, HIDDEN_SIZE);
  assertEquals(program.chunkLength, 4);
  assertEquals(program.capacity, 64);
  assertEquals(program.stopTokens, [7]);
  // The graph used for validation is not kept in the program (it holds only the immutable wiring — ADR 0083 decision 1).
  assertEquals(Object.hasOwn(program, "graph"), false);
});

test("createGenerationProgram: stopTokens is copied (detached from the caller's array)", () => {
  const stopTokens = [1, 7];
  const program = createGenerationProgram(specOf({ stopTokens }));
  stopTokens.push(9);
  assertEquals(program.stopTokens, [1, 7]);
});

test("createGenerationProgram: chunkBuckets is copied, and omission folds to an empty array", () => {
  const chunkBuckets = [2, 3];
  const program = createGenerationProgram(specOf({ chunkBuckets }));
  assertEquals(program.chunkBuckets, [2, 3]);
  // Detached from the caller's array (a bucket added later would be "a physical row count the context does not allow").
  chunkBuckets.push(3);
  assertEquals(program.chunkBuckets, [2, 3]);
  // Omitted = nothing added. Carrying `undefined` around would not turn red when a selecting side forgets `?? []`.
  assertEquals(createGenerationProgram(specOf()).chunkBuckets, []);
});

test("createGenerationProgram: the accepted set of chunkBuckets passes through the runtime's check as is", () => {
  // The rules (2 or more and below chunkLength, strictly ascending) are not copied to the models side, so what is
  // pinned here is "it fails before the context is created" itself (copied rules would drift from the context as they grow).
  const cases: readonly (readonly [readonly number[], string])[] = [
    // 1 is the decode shape itself / chunkLength or more is outside the queryLength ≤ chunkLength contract.
    [[1], "chunkBuckets[0] 1 is not an integer in 2..3"],
    [[4], "chunkBuckets[0] 4 is not an integer in 2..3"],
    [[2.5], "chunkBuckets[0] 2.5 is not an integer in 2..3"],
    // Duplicates and descending order make it impossible to pick "the smallest bucket ≥ queryLength" by a linear scan.
    [[2, 2], "chunkBuckets[1] 2 is not greater than the previous 2"],
    [[3, 2], "chunkBuckets[1] 2 is not greater than the previous 3"],
  ];
  for (const [chunkBuckets, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf({ chunkBuckets })), Error, message);
  }
});

test("createGenerationProgram: suppressTokens is copied, frozen, and folded to an empty array when omitted", () => {
  const suppressTokens = [3, 9];
  const program = createGenerationProgram(specOf({ suppressTokens }));
  assertEquals(program.suppressTokens, [3, 9]);
  // Cut off from the caller's array (the selection reads it every step).
  suppressTokens.push(1);
  assertEquals(program.suppressTokens, [3, 9]);
  assert(Object.isFrozen(program.suppressTokens));
  // Omitted = none. Carried as `undefined`, a missed `?? []` on the selection side would raise
  // nothing and leave the mask silently off.
  assertEquals(createGenerationProgram(specOf()).suppressTokens, []);
  // Not on the public read surface (the generation surface applies it; nothing to read).
  assertEquals(Object.hasOwn(generationProgramFace(program), "suppressTokens"), false);
});

test("createGenerationProgram: an invalid suppressed set fails loudly", () => {
  const cases: readonly (readonly [readonly number[], string])[] = [
    // An id outside the vocabulary masks nothing — "a suppression that does nothing".
    [[VOCAB], `suppressTokens[0] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`],
    [[-1], "suppressTokens[0] -1"],
    [[2, 1.5], "suppressTokens[1] 1.5"],
    [[4, 4], "suppressTokens: token 4 appears twice"],
  ];
  for (const [suppressTokens, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf({ suppressTokens })), Error, message);
  }
});

test("createGenerationProgram: a graph without derived inputs passes with derivedInputs omitted", () => {
  const program = createGenerationProgram(
    specOf({ graph: graphOf({ derived: false }), derivedInputs: undefined }),
  );
  assertEquals(program.derivedInputs, undefined);
});

test("createGenerationProgram: the accepted set of numbers", () => {
  const cases: readonly (readonly [Partial<GenerationProgramSpec>, string])[] = [
    [{ chunkLength: 0 }, "chunkLength 0"],
    [{ chunkLength: 2.5 }, "chunkLength 2.5"],
    [{ maxPosition: 0 }, "maxPosition 0"],
    [{ capacity: -1 }, "capacity -1"],
    [{ vocabSize: 0 }, "vocabSize 0"],
    // A stop token outside the vocabulary silently adds "a stop condition that can never hold".
    [{ stopTokens: [VOCAB] }, `stopTokens[0] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`],
    [{ stopTokens: [-1] }, "stopTokens[0] -1"],
    [{ stopTokens: [0, 1.5] }, "stopTokens[1] 1.5"],
  ];
  for (const [override, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf(override)), Error, message);
  }
});

test("createGenerationProgram: a wrong input name / dtype / shape fails loudly", () => {
  const cases: readonly (readonly [string, Partial<GenerationProgramSpec>, string])[] = [
    ["no token id input", { inputIds: "tokens" }, "token id input 'tokens' is not a graph input"],
    ["no last_row", { lastRow: "row" }, "last_row input 'row' is not a graph input"],
    [
      "token id input is f32",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === IDS ? { ...input, dtype: "f32" } : input,
          ),
        },
      },
      "token id input 'input_ids' has dtype f32",
    ],
    [
      // A fixed M cannot run the prefill shape (M=chunkLength) and the decode shape (M=1) in the same graph.
      "M is a fixed number",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === IDS ? { ...input, shape: [1, 4] } : input,
          ),
        },
      },
      "has shape [1,4], not [1,<symbol>]",
    ],
    [
      "last_row is not 1-dimensional",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === LAST_ROW ? { ...input, shape: [1, 1] } : input,
          ),
        },
      },
      "last_row input 'last_row' has shape [1,1], not [<symbol>]",
    ],
    [
      // An old distribution (row selection fixed at 1 row) — the R rows of verify cannot run in the same graph.
      "last_row is the fixed [1]",
      {
        graph: {
          ...graphOf(),
          inputs: graphOf().inputs.map((input) =>
            input.name === LAST_ROW ? { ...input, shape: [1] } : input,
          ),
        },
      },
      "last_row input 'last_row' has shape [1], not [<symbol>]",
    ],
  ];
  for (const [name, override, message] of cases) {
    assertThrows(() => createGenerationProgram(specOf(override)), Error, message, name);
  }
});

test("createGenerationProgram: a miswiring that uses the same symbol for the input length and the selected row count fails at build time", () => {
  // `input_ids:[1,M]` and `last_row:[M]` both pass their individual shape checks, but the first run of prefill (4
  // physical rows, 1 selected row) binds M to both 4 and 1 and the runtime rejects it. Decode (1 row) passes by
  // chance, so detection is delayed until the first prefill after the GPU build.
  const graph: GenerationGraph = {
    symbols: ["C", "M"],
    inputs: [
      { name: IDS, dtype: "i32", shape: [1, "M"] },
      { name: DERIVED, dtype: "f32", shape: [1, "M", 2, 3] },
      { name: LAST_ROW, dtype: "i32", shape: ["M"] },
    ],
    outputs: [LOGITS, HIDDEN],
    values: {
      [LOGITS]: { dtype: "f32", shape: [1, "M", VOCAB] },
      [HIDDEN]: { dtype: "f32", shape: [1, "M", HIDDEN_SIZE] },
    },
  };
  assertThrows(() => createGenerationProgram(specOf({ graph })), Error, "same symbol M");
});

test("createGenerationProgram: checks the logits output's existence / shape / vocabulary size", () => {
  assertThrows(
    () => createGenerationProgram(specOf({ logits: "scores" })),
    Error,
    "logits output 'scores' is not a graph output",
  );
  // A name that only exists as a node output is not returned from run (checks that it is a graph output).
  assertThrows(
    () => createGenerationProgram(specOf({ graph: { ...graphOf(), outputs: [HIDDEN] } })),
    Error,
    "logits output 'logits' is not a graph output",
  );
  // A regression to all-row logits (`[1,M,V]`) = not a row-selection output. M is a different symbol from the row
  // selection, so "the selected row count" and "the returned row count" would be decided separately.
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(LOGITS, { dtype: "f32", shape: [1, "M", VOCAB] }) }),
      ),
    Error,
    "not [1,R,64]",
  );
  // An old distribution (fixed to the single last row) — with R=1 runs the values are the same, but the R rows of verify are not exported.
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(LOGITS, { dtype: "f32", shape: [1, 1, VOCAB] }) }),
      ),
    Error,
    "not [1,R,64]",
  );
  // A vocabulary size mismatch (assets of a different generation combined with the program).
  assertThrows(
    () => createGenerationProgram(specOf({ vocabSize: 32, stopTokens: [7] })),
    Error,
    "not [1,R,32]",
  );
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(LOGITS, { dtype: "i32", shape: [1, "R", VOCAB] }) }),
      ),
    Error,
    "logits output 'logits' has dtype i32",
  );
});

test("createGenerationProgram: checks the hidden output's existence / shape / H", () => {
  // An old distribution with a single output fails as "hidden is not a graph output" (there is no compatibility branch).
  assertThrows(
    () => createGenerationProgram(specOf({ graph: { ...graphOf(), outputs: [LOGITS] } })),
    Error,
    "hidden output 'hidden' is not a graph output",
  );
  assertThrows(
    () => createGenerationProgram(specOf({ hidden: "states" })),
    Error,
    "hidden output 'states' is not a graph output",
  );
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(HIDDEN, { dtype: "i32", shape: [1, "R", HIDDEN_SIZE] }) }),
      ),
    Error,
    "hidden output 'hidden' has dtype i32",
  );
  // Rows with a different symbol from logits = the 2 outputs return different row sets.
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(HIDDEN, { dtype: "f32", shape: [1, "M", HIDDEN_SIZE] }) }),
      ),
    Error,
    "not [1,R,<H>]",
  );
  // If H stayed a symbol, the width would change per run (it would not connect to the drafter's weights).
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(HIDDEN, { dtype: "f32", shape: [1, "R", "H"] }) }),
      ),
    Error,
    "not [1,R,<H>]",
  );
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({ graph: graphWithValue(HIDDEN, { dtype: "f32", shape: [1, "R", 0] }) }),
      ),
    Error,
    "hidden output 'hidden' H 0 is not a positive integer",
  );
  // The same name wired to both outputs (passes both shape checks only when V = H).
  assertThrows(
    () => createGenerationProgram(specOf({ hidden: LOGITS })),
    Error,
    "The logits output and the hidden output point at the same name 'logits'",
  );
});

test("createGenerationProgram: checks the coverage of the graph inputs in both directions", () => {
  // Missing: building the program without supplying per_layer_inputs from the host side.
  assertThrows(
    () => createGenerationProgram(specOf({ derivedInputs: undefined })),
    Error,
    `Graph inputs ${DERIVED} are not wired`,
  );
  // Extra: a declared name is not in the graph (an input ignored on every run).
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: graphOf({ derived: false }),
          derivedInputs: { names: ["ple"], derive: () => Promise.resolve({}) },
        }),
      ),
    Error,
    "The wired ple are not graph inputs",
  );
  // Duplicate: the same name wired twice (one is silently overwritten).
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          graph: graphOf({ derived: false }),
          derivedInputs: { names: [IDS], derive: () => Promise.resolve({}) },
        }),
      ),
    Error,
    "The wired input names contain duplicates",
  );
});

test("createGenerationProgram: resident inputs also count toward coverage (inputs whose producer is the Session side)", () => {
  // The value is produced by the Session wrapper, so it is not in derivedInputs (the PLE GPU-resident option).
  const wiring = createGenerationProgram(
    specOf({ derivedInputs: undefined, residentInputs: [DERIVED] }),
  );
  assertEquals(wiring.residentInputs, [DERIVED]);
  assertEquals(wiring.derivedInputs, undefined);
  // The same input claimed by 2 producers fails as a duplicate.
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          residentInputs: [DERIVED],
          derivedInputs: { names: [DERIVED], derive: () => Promise.resolve({}) },
        }),
      ),
    Error,
    "The wired input names contain duplicates",
  );
});

test("createGenerationProgram: every symbol is determined by either an input shape or the capacity symbol", () => {
  // C (the capacity symbol of the state slots) does not appear in the input shapes = the capacity symbol is its only source.
  assertThrows(
    () => createGenerationProgram(specOf({ capacitySymbol: "K" })),
    Error,
    "Capacity symbol K is not in the graph's symbols [C, M, R]",
  );
  // Choosing a symbol determined by an input shape as the capacity symbol splits the run's binding from the context's.
  assertThrows(
    () => createGenerationProgram(specOf({ capacitySymbol: "M" })),
    Error,
    "Capacity symbol M is a symbol determined by an input shape",
  );
  // The same for R (the row count is determined by the element count of the last_row input = no context binding needed).
  assertThrows(
    () => createGenerationProgram(specOf({ capacitySymbol: "R" })),
    Error,
    "Capacity symbol R is a symbol determined by an input shape",
  );
  // A shape with one capacity symbol too few (assets whose states have 2 symbols).
  assertThrows(
    () =>
      createGenerationProgram(specOf({ graph: { ...graphOf(), symbols: ["C", "D", "M", "R"] } })),
    Error,
    "Symbols D are determined neither by an input shape nor by the capacity symbol",
  );
  // M / R are determined by the input shapes, so they need no capacity symbol (C alone passes).
  assertEquals(createGenerationProgram(specOf()).capacitySymbol, "C");
});

test("GenerationGraph: a real IrGraph satisfies this surface (detects spelling drift)", () => {
  // A type-check-only check (does nothing at run time). `GenerationGraph` **mirrors** `PreparedModel["graph"]`, so if
  // the spelling of the IR's inputs / outputs / values / symbols changes on the runtime side, this one line becomes a
  // compile error.
  const asGenerationGraph = (graph: PreparedModel["graph"]): GenerationGraph => graph;
  assertEquals(typeof asGenerationGraph, "function");
});

// ---- The public view (`generationProgramFace`) ----------------------------------
//
// `GenerationProgram` is a value exposed on the package's public surface as `gemma.Pipeline.program`, so what is
// protected is the consumer side. The only check exercising it was one e2e with real assets + a real GPU, so a change
// that removed `Object.freeze` / turned `[...wiring.stopTokens]` back into `wiring.stopTokens` stayed all green in
// environments without assets (CI and most development machines).

test("generationProgramFace: exposes only 6 numeric fields (no internal wiring)", () => {
  const wiring = createGenerationProgram(specOf({ stopTokens: [1, 7], chunkBuckets: [2] }));
  const face = generationProgramFace(wiring);
  assertEquals(Object.keys(face).sort(), [
    "capacity",
    "chunkBuckets",
    "chunkLength",
    "maxPosition",
    "stopTokens",
    "vocabSize",
  ]);
  assertEquals(face, {
    chunkLength: 4,
    chunkBuckets: [2],
    maxPosition: 128,
    capacity: 64,
    vocabSize: VOCAB,
    stopTokens: [1, 7],
  });
});

test("generationProgramFace: returns a frozen copy (consumer writes do not reach the stop set)", () => {
  const wiring = createGenerationProgram(specOf({ stopTokens: [1, 7], chunkBuckets: [2, 3] }));
  const face = generationProgramFace(wiring);
  assertEquals(Object.isFrozen(face), true, "face is not frozen");
  assertEquals(Object.isFrozen(face.stopTokens), true, "stopTokens is not frozen");
  // The buckets are treated the same (reordering them would change how the physical row count is chosen).
  assertEquals(Object.isFrozen(face.chunkBuckets), true, "chunkBuckets is not frozen");
  assertThrows(() => {
    (face.chunkBuckets as number[]).reverse();
  }, TypeError);
  assert(face.chunkBuckets !== wiring.chunkBuckets, "exposes the wiring's array as is");
  // ESM is always strict mode, so a write to a frozen array is not silently dropped but raises a TypeError.
  assertThrows(() => {
    (face.stopTokens as number[]).length = 0;
  }, TypeError);
  assertThrows(() => {
    (face.stopTokens as number[]).sort((a, b) => a - b);
  }, TypeError);
  // They are separate objects (keeping only the freeze and dropping the copy would freeze the wiring's array too, as a side effect).
  assert(face.stopTokens !== wiring.stopTokens, "exposes the wiring's array as is");
  assertEquals(Object.isFrozen(wiring.stopTokens), false, "freezes the wiring side too");
  assertEquals([...wiring.stopTokens], [1, 7], "the wiring side's stop set is intact");
});

test("createGenerationProgram: derivedInputs.positionsOnly is a declaration, so only a boolean passes", () => {
  const derive = (): Promise<Record<string, never>> => Promise.resolve({});
  for (const positionsOnly of [true, false, undefined]) {
    const program = createGenerationProgram(
      specOf({
        derivedInputs: {
          names: [DERIVED],
          derive,
          ...(positionsOnly === undefined ? {} : { positionsOnly }),
        },
      }),
    );
    assertEquals(program.derivedInputs?.positionsOnly, positionsOnly);
  }
  // A truthy value from outside the type would otherwise admit the run-ahead on a slot that reads ids.
  assertThrows(
    () =>
      createGenerationProgram(
        specOf({
          derivedInputs: { names: [DERIVED], derive, positionsOnly: "yes" as unknown as boolean },
        }),
      ),
    Error,
    "positionsOnly",
  );
});

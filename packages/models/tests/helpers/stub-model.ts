// The minimal {@link GraphOwner} that drives the build check (graph declarations against
// `pipelineConfig`) without a GPU or real assets.
//
// The check reads only the graph's **declarations** (symbols / inputs / outputs / values), never
// tensor data or the node list. What is built here holds only those declarations; it does not go
// through IR structural validation or the container merge — going through them would make a
// "broken declaration" fail there first, and the check itself could not be reached.
//
// MUST: the valid declaration is not built here; each test **writes one valid shape and then
// breaks exactly one point** (the same convention as `helpers/safetensors.ts`). A shared default
// would absorb the breakage.

import type { GraphOwner } from "../../src/hub/components.ts";

/**
 * A declarable dimension (a symbolic dimension is a string in the canonical notation of
 * `format/dims.ts`).
 */
export type StubDim = number | string;

/** A declarable dtype (f32 when omitted). */
type StubDtype = GraphOwner["graph"]["inputs"][number]["dtype"];

export type StubGraph = {
  readonly symbols?: readonly string[];
  readonly inputs: readonly {
    readonly name: string;
    readonly shape: readonly StubDim[];
    readonly dtype?: StubDtype;
  }[];
  readonly outputs: readonly string[];
  /** Value name → declared shape (the check looks only at the one value `outputs[0]`). */
  readonly values: Readonly<Record<string, readonly StubDim[]>>;
};

export const stubModel = (graph: StubGraph): GraphOwner => ({
  graph: {
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: [...(graph.symbols ?? [])],
    inputs: graph.inputs.map((input) => ({
      name: input.name,
      dtype: input.dtype ?? "f32",
      shape: [...input.shape],
    })),
    outputs: [...graph.outputs],
    initializers: {},
    values: Object.fromEntries(
      Object.entries(graph.values).map(([name, shape]) => [
        name,
        { dtype: "f32" as const, shape: [...shape] },
      ]),
    ),
    states: {},
    nodes: [],
  },
});

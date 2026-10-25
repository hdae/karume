/**
 * The Wan DiT's self-attention q / k prologue as the exporter writes it — `rms_norm` over the token
 * row `[1,S,H·D]`, a reshape to `[1,S,H,D]`, a reshape to the pairs `[1,S,H·D/2,2]` and the 8 nodes
 * of the pair-form RoPE — for the fusion rule `rmsNormRope` (GPU-less matcher tests and the GPU
 * Session test share it). Each option breaks one point of q's chain; the defaults are the measured
 * spelling.
 */

import type { DeclarationJson } from "./model-fixture.ts";

/** The values of q's chain an option can expose. */
export type RmsNormRopeValue = "normalised" | "view" | "pairs";

export type RmsNormRopeGraphOptions = {
  /** H (default 2). */
  readonly heads?: number;
  /** D (default 128). */
  readonly headDim?: number;
  /**
   * `"wan"` (default): the q and k chains interleaved in the Wan IR order (`rms_norm` q, `rms_norm`
   * k, the reshapes of q / k / v, q's pair reshape and RoPE, k's pair reshape and RoPE), v's reshape
   * read by a `neg` at the end. `"single"`: q's chain alone.
   */
  readonly form?: "wan" | "single";
  /** `add(n, bias)` between q's rms_norm and its first reshape. */
  readonly bias?: boolean;
  /** q's rms_norm normalises each head: its input is reshaped to `[1,S,H,D]` first, weight `[D]`. */
  readonly perHead?: boolean;
  /** q's RoPE is the half split on `[1,S,H,D]` (no pair reshape). */
  readonly halfSplit?: boolean;
  /**
   * A zero-dispatch reshape of each chain's normalised value right after its rms_norm (q's and, in
   * the Wan form, k's): the values and the dispatches stay those of the plain path, but no chain
   * matches (the unfused reference on the same backend).
   */
  readonly interpose?: boolean;
  /** q's value read once more outside the window (a `neg`, as a graph output). */
  readonly extraConsumer?: RmsNormRopeValue;
  /** q's value is a graph output. */
  readonly graphOutput?: RmsNormRopeValue;
  /** A `neg` of an unrelated input between q's first reshape and its pair reshape. */
  readonly foreignGap?: boolean;
  /** cos is a reshape of a table input inside q's window (the chain reads a node in its window). */
  readonly tableInWindow?: boolean;
};

const EPS = 1e-6;

export const rmsNormRopeGraph = (options: RmsNormRopeGraphOptions = {}): DeclarationJson => {
  const heads = options.heads ?? 2;
  const headDim = options.headDim ?? 128;
  const dim = heads * headDim;
  const row = [1, "S", dim];
  const full = [1, "S", heads, headDim];
  const pairs = [1, "S", dim / 2, 2];
  const half = [1, "S", dim / 2, 1];
  const halfHead = [1, "S", heads, headDim / 2];
  const table = [1, "S", 1, headDim];
  const wan = (options.form ?? "wan") === "wan";
  const values: DeclarationJson["values"] = {};
  const inputs: DeclarationJson["inputs"] = [];
  const outputs: string[] = [];
  const nodes: DeclarationJson["nodes"] = [];
  const value = (name: string, shape: (number | string)[]): string => {
    values[name] = { dtype: "f32", shape };
    return name;
  };
  const input = (name: string, shape: (number | string)[]): string => {
    inputs.push({ name, dtype: "f32", shape });
    return name;
  };
  const node = (op: string, ins: string[], out: string, attrs: unknown = {}): void => {
    nodes.push({ op, ins, outs: [out], attrs });
  };
  /** One chain's rms_norm (its input and weight as graph inputs); returns the normalised value. */
  const normalise = (p: string, perHead: boolean): string => {
    const x = input(`${p}_x`, row);
    if (perHead) {
      node("reshape", [x], value(`${p}_x4`, full));
      node("rms_norm", [`${p}_x4`, input(`${p}_w`, [headDim])], value(`${p}_n`, full), {
        eps: EPS,
      });
      return `${p}_n`;
    }
    node("rms_norm", [x, input(`${p}_w`, [dim])], value(`${p}_n`, row), { eps: EPS });
    return `${p}_n`;
  };
  /** The 8 pair-form RoPE nodes on `x` and `pairsName`. */
  const pairRope = (p: string, x: string, pairsName: string, cos: string): void => {
    node("slice", [pairsName], value(`${p}_odd`, half), { dim: 3, start: 1, end: 2 });
    node("neg", [`${p}_odd`], value(`${p}_negative`, half));
    node("slice", [pairsName], value(`${p}_even`, half), { dim: 3, start: 0, end: 1 });
    node("cat", [`${p}_negative`, `${p}_even`], value(`${p}_rotated`, pairs), { dim: 3 });
    node("reshape", [`${p}_rotated`], value(`${p}_rot`, full));
    node("mul", [x, cos], value(`${p}_direct`, full));
    node("mul", [`${p}_rot`, "sin"], value(`${p}_cross`, full));
    node("add", [`${p}_direct`, `${p}_cross`], value(`${p}_y`, full));
  };
  /** The 7 half-split RoPE nodes (slice-first) on `x`. */
  const halfRope = (p: string, x: string, cos: string): void => {
    node("slice", [x], value(`${p}_first`, halfHead), { dim: 3, start: 0, end: headDim / 2 });
    node("slice", [x], value(`${p}_second`, halfHead), {
      dim: 3,
      start: headDim / 2,
      end: headDim,
    });
    node("neg", [`${p}_second`], value(`${p}_negative`, halfHead));
    node("cat", [`${p}_negative`, `${p}_first`], value(`${p}_rotated`, full), { dim: 3 });
    node("mul", [x, cos], value(`${p}_direct`, full));
    node("mul", [`${p}_rotated`, "sin"], value(`${p}_cross`, full));
    node("add", [`${p}_direct`, `${p}_cross`], value(`${p}_y`, full));
  };

  // q's head (with the options that sit right after it), then k's.
  let qNormalised = normalise("q", options.perHead === true);
  if (options.bias) {
    node("add", [qNormalised, input("q_bias", [dim])], value("q_nb", row));
    qNormalised = "q_nb";
  }
  if (options.interpose) {
    node("reshape", [qNormalised], value("q_n_alias", options.perHead ? full : row));
    qNormalised = "q_n_alias";
  }
  const cos = "cos";
  if (options.tableInWindow) {
    node("reshape", [input("cos_source", [1, "S", headDim])], value(cos, table));
  } else {
    input(cos, table);
  }
  input("sin", table);
  let kNormalised = "k_n";
  if (wan) normalise("k", false);
  if (wan && options.interpose) {
    node("reshape", [kNormalised], value("k_n_alias", row));
    kNormalised = "k_n_alias";
  }
  node("reshape", [qNormalised], value("q_v", full));
  if (wan) {
    node("reshape", [kNormalised], value("k_v", full));
    node("reshape", [input("v_x", row)], value("v4", full));
  }
  if (options.foreignGap) node("neg", [input("z", row)], value("z_neg", row));
  if (options.halfSplit) {
    halfRope("q", "q_v", cos);
  } else {
    node("reshape", ["q_v"], value("q_pairs", pairs));
    pairRope("q", "q_v", "q_pairs", cos);
  }
  outputs.push("q_y");
  if (wan) {
    node("reshape", ["k_v"], value("k_pairs", pairs));
    pairRope("k", "k_v", "k_pairs", cos);
    node("neg", ["v4"], value("v_out", full));
    outputs.push("k_y", "v_out");
  }
  if (options.foreignGap) outputs.push("z_neg");
  const exposed: Readonly<Record<RmsNormRopeValue, string>> = {
    normalised: "q_n",
    view: "q_v",
    pairs: "q_pairs",
  };
  if (options.extraConsumer !== undefined) {
    const source = exposed[options.extraConsumer];
    node("neg", [source], value("q_extra", values[source].shape));
    outputs.push("q_extra");
  }
  if (options.graphOutput !== undefined) outputs.push(exposed[options.graphOutput]);
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((step) => step.op))].sort() },
    symbols: ["S"],
    inputs,
    outputs,
    initializers: {},
    values,
    nodes,
  };
};

/** The graph's input shapes with `S` bound to `sequence`. */
export const rmsNormRopeInputShapes = (
  graph: DeclarationJson,
  sequence: number,
): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((extent) => (extent === "S" ? sequence : Number(extent))),
    ]),
  );

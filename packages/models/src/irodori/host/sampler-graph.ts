/**
 * The 2 GPU-side graphs of the Euler sampler (IR v2 declarations **built on the host** — they do not
 * go through the exporter).
 *
 * Once the DiT loop is bundled into 1 batch (H-5), only the CFG combination and the Euler update
 * would remain on the host and force "readback → re-upload on every forward". These 2 operations
 * are nothing but elementwise add / sub / mul, so instead of adding them to the distribution,
 * **small IR built on the spot** runs in separate Sessions, and the latent stays on the GPU as a
 * resident tensor.
 *
 * ## MUST: the built declarations go through `parseIrDeclarationValue`
 *
 * This goes through neither the exporter nor the container reader, so without it not one of the
 * rules decided by the graph alone (SSA, forward references, topological order, `requires.ops`
 * matching the ops actually used, unknown keys) is checked. The merge layer (declaration × supply)
 * and `prepareContainer` (capability + op contracts) do not look at that kind of breakage, so an
 * editing mistake would not fail until the Session is built or run.
 *
 * MUST NOT: do not add the optional `states` to the literals below. Leaving it out keeps the
 * literals from typing as {@link IrDeclaration} (`states` is missing), so only the return value of
 * `parseIrDeclarationValue` can be this function's return value. An edit that removes the wrapper
 * fails the type check (`vp check`).
 *
 * ## MUST: association and argument order match the TS source of truth (`sampler.ts`) op by op
 *
 * f32 addition is not associative, so a different queuing order changes the last digits. The
 * source of truth is {@link combineCfg} (folds `f32(value + f32(scale * f32(base − velocity)))` in
 * variant order) and {@link eulerStep} (`f32(x + f32(deltaT * velocity))`), and the node lists
 * below are copies of them (`sub` → `mul` → `add` / `mul` → `add`). The host-side functions are
 * kept as the numeric source of truth.
 *
 * ## NOTE: no symbolic dimensions (S is fixed per generation)
 *
 * These are called after S is decided, so the graphs are built with concrete dimensions. A resident
 * tensor has no shape (only bytes and a size), so it cannot be the binding source of a symbol, and
 * a symbol would force passing the binding on every enqueue. With concrete dimensions the bindings
 * stay empty, and the runtime matches the real sizes against the declared shapes.
 */

import { type IrDeclaration, parseIrDeclarationValue } from "@karume/runtime";

/** Shape of one latent (`[1, S, latentDim]`, the same as the DiT's `x_t` / velocity field). */
const latentShape = (frames: number, latentDim: number): readonly number[] => [
  1,
  frames,
  latentDim,
];

/** Input names (shared with the caller in one place). */
export const COMBINE_INPUTS = {
  /** The combination so far (cond itself for k = 0). */
  accumulator: "acc_in",
  /** Velocity field of cond (the base of the difference is **always** cond). */
  cond: "cond",
  /** The uncond velocity field of that variant. */
  variant: "v_k",
  /** Strength of that variant (shape `[1]` — right-aligned broadcast applies it everywhere). */
  scale: "s",
} as const;

/** Output name of {@link combineGraph}. */
export const COMBINE_OUTPUT = "acc_out";

export const EULER_INPUTS = {
  /** The current latent. */
  x: "x",
  /**
   * Velocity field (the combined one on a step with CFG, cond itself on a step without).
   */
  velocity: "v",
  /** Step size `t_next − t` (shape `[1]`, negative). */
  deltaT: "dt",
} as const;

/** Output name of {@link eulerGraph}. */
export const EULER_OUTPUT = "x_next";

/**
 * CFG combination for one variant, `acc_out = acc_in + s·(cond − v_k)` (3 nodes).
 *
 * Variants go through this graph one at a time — the same queuing as the source of truth
 * {@link combineCfg}, which folds one variant at a time in variant order. The strength `s` is an
 * input of shape `[1]`, so the graph is not rebuilt per value.
 */
export const combineGraph = (frames: number, latentDim: number): IrDeclaration => {
  const rows = latentShape(frames, latentDim);
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["sub", "mul", "add"] },
    symbols: [],
    inputs: [
      { name: COMBINE_INPUTS.accumulator, dtype: "f32", shape: rows },
      { name: COMBINE_INPUTS.cond, dtype: "f32", shape: rows },
      { name: COMBINE_INPUTS.variant, dtype: "f32", shape: rows },
      { name: COMBINE_INPUTS.scale, dtype: "f32", shape: [1] },
    ],
    outputs: [COMBINE_OUTPUT],
    initializers: {},
    values: {
      diff: { dtype: "f32", shape: rows },
      scaled: { dtype: "f32", shape: rows },
      [COMBINE_OUTPUT]: { dtype: "f32", shape: rows },
    },
    nodes: [
      { op: "sub", ins: [COMBINE_INPUTS.cond, COMBINE_INPUTS.variant], outs: ["diff"], attrs: {} },
      { op: "mul", ins: [COMBINE_INPUTS.scale, "diff"], outs: ["scaled"], attrs: {} },
      {
        op: "add",
        ins: [COMBINE_INPUTS.accumulator, "scaled"],
        outs: [COMBINE_OUTPUT],
        attrs: {},
      },
    ],
  });
};

/** Euler update `x_next = x + dt·v` (2 nodes — same order as the canonical {@link eulerStep}). */
export const eulerGraph = (frames: number, latentDim: number): IrDeclaration => {
  const rows = latentShape(frames, latentDim);
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["mul", "add"] },
    symbols: [],
    inputs: [
      { name: EULER_INPUTS.x, dtype: "f32", shape: rows },
      { name: EULER_INPUTS.velocity, dtype: "f32", shape: rows },
      { name: EULER_INPUTS.deltaT, dtype: "f32", shape: [1] },
    ],
    outputs: [EULER_OUTPUT],
    initializers: {},
    values: {
      step: { dtype: "f32", shape: rows },
      [EULER_OUTPUT]: { dtype: "f32", shape: rows },
    },
    nodes: [
      { op: "mul", ins: [EULER_INPUTS.deltaT, EULER_INPUTS.velocity], outs: ["step"], attrs: {} },
      { op: "add", ins: [EULER_INPUTS.x, "step"], outs: [EULER_OUTPUT], attrs: {} },
    ],
  });
};

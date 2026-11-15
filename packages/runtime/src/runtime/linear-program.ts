/**
 * A linear strategy's dispatches as data.
 *
 * Most strategies of `linear` are still built by a function that talks to the device (it fetches the
 * pipeline, writes the uniform, declares the dispatch). A {@link LinearProgram} says the same thing
 * without a device: which kernels run, with which uniform, over which operands and on which grid,
 * and which temporaries live between them. The recipe builder turns it into dispatches with one
 * generic routine (`emitLinearProgram` in recipe-builders/linear.ts), and anything that needs to know
 * a strategy's WGSL or temporaries before running — the memory estimate, the verification cache —
 * can read the same value (ADR 0152 decisions 6 and 8).
 *
 * The split-K linear is the first strategy written this way; the others move here one at a time.
 *
 * NOTE: device-free (kernel modules only). Not part of the public surface of mod.ts (ADR 0008).
 */

import {
  LINEAR_SPLITK_COMBINE_GROUP,
  LINEAR_SPLITK_TILE,
  linearSplitKCombineKey,
  linearSplitKCombineWgsl,
  linearSplitKParams,
  linearSplitKPartialKey,
  linearSplitKPartialsBytes,
  linearSplitKPartialWgsl,
  type LinearSplitKSlices,
} from "../kernels/linear-splitk.ts";

/** What a binding points at: the linear's operands, or a temporary of the program by its index. */
export type ProgramOperand = "x" | "weight" | "bias" | "out" | { readonly temp: number };

/** One axis of a tiled grid: `ceil(items / tile)` workgroups, failing loudly above the device limit. */
export type TiledAxis = { readonly items: number; readonly tile: number };

export type ProgramGrid =
  /** One workgroup = one tile on each axis; nothing falls back, so exceeding the limit is an error. */
  | { readonly kind: "tiled"; readonly axes: readonly [TiledAxis, TiledAxis, TiledAxis] }
  /** A grid-stride kernel over `items` elements with `workgroupSize` threads per workgroup. */
  | { readonly kind: "gridStride"; readonly items: number; readonly workgroupSize: number };

export type ProgramDispatch = {
  readonly key: string;
  /** The WGSL of the key (generated on demand — the pipeline cache holds it after the first use). */
  readonly wgsl: () => string;
  /** The uniform at binding 0. */
  readonly params: Uint32Array<ArrayBuffer>;
  /** The operands at bindings 1, 2, … in order. */
  readonly operands: readonly ProgramOperand[];
  readonly grid: ProgramGrid;
};

export type LinearProgram = {
  /** The head of a failure's message (`linear split-K [x] × [weight]`). */
  readonly where: string;
  /** The byte length of each temporary. A temporary lives from the first dispatch to the last. */
  readonly temps: readonly number[];
  readonly dispatches: readonly ProgramDispatch[];
};

/**
 * The split-K linear (src/kernels/linear-splitk.ts): the partial sums of every K slice into one
 * temporary, then their sum and the bias into the output.
 */
export const linearSplitKProgram = (
  where: string,
  m: number,
  n: number,
  k: number,
  slices: LinearSplitKSlices,
): LinearProgram => {
  const params = linearSplitKParams(m, n, k, slices);
  return {
    where,
    temps: [linearSplitKPartialsBytes(m, n, slices)],
    dispatches: [
      {
        key: linearSplitKPartialKey(),
        wgsl: linearSplitKPartialWgsl,
        params,
        operands: ["x", "weight", { temp: 0 }],
        grid: {
          kind: "tiled",
          axes: [
            { items: n, tile: LINEAR_SPLITK_TILE },
            { items: m, tile: LINEAR_SPLITK_TILE },
            { items: slices, tile: 1 },
          ],
        },
      },
      {
        key: linearSplitKCombineKey(),
        wgsl: linearSplitKCombineWgsl,
        params,
        operands: [{ temp: 0 }, "bias", "out"],
        grid: { kind: "gridStride", items: m * n, workgroupSize: LINEAR_SPLITK_COMBINE_GROUP },
      },
    ],
  };
};

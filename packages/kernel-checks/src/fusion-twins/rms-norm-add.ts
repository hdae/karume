/**
 * The rms_norm + add fusion (`fuseRmsNormAdd`, the rule rmsNormAdd — the runtime's
 * src/runtime/fusion.ts) against the same graph run without the option (the norm also an output, so
 * nothing fuses) on the same backend, over widths, both addition orders, row counts and value
 * patterns (finite, cancelling to ±0, special words, signed zeros): y and the later consumer of x
 * u32-identical with NaN by class, rmsNormAdd at 1 and 0, the fused side in 2 dispatches.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  fill,
  type FilledTensor,
  openModelBytes,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { rmsNormAddGraph } from "../../../runtime/tests/helpers/rms-norm-add-graph.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import {
  type CheckOptions,
  type CheckPart,
  type ExpectedCases,
  scopedGpuWork,
  wordsOf,
} from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

export const RMS_NORM_ADD_DIMS: readonly number[] = [256, 1536, 2560, 2816, 3840];
export const RMS_NORM_ADD_ROWS: readonly number[] = [1, 4, 8, 40, 64];
/** Whether the norm is the first addend (`norm + residual`) or the second. */
export const RMS_NORM_ADD_ORDERS: readonly boolean[] = [false, true];

export type RmsNormAddPattern = "finite" | "cancel" | "special" | "zeros";

export const RMS_NORM_ADD_PATTERNS: readonly RmsNormAddPattern[] = [
  "finite",
  "cancel",
  "special",
  "zeros",
];

/**
 * x, w and r of a pattern. `cancel` starts from the finite values: its residual is the negated norm,
 * which only the plain side's `n` output gives ({@link checkRmsNormAddTwins} writes it).
 */
export const rmsNormAddInputs = (
  rows: number,
  dim: number,
  pattern: RmsNormAddPattern,
): { readonly x: FilledTensor; readonly w: FilledTensor; readonly r: FilledTensor } => {
  const x = fill([rows, dim], (i) => Math.sin((i % dim) * 0.713 + 0.19) * 4);
  const w = fill([dim], (i) => 0.2 + Math.cos(i * 0.17));
  const r = fill([rows, dim], (i) => Math.cos((i % dim) * 0.531) * 3);
  if (pattern === "special") {
    wordsOf(x.data).set([
      0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x7f800000, 0xff800000, 0x7fc00123,
    ]);
  }
  if (pattern === "zeros") {
    wordsOf(x.data).set(
      Uint32Array.from({ length: rows * dim }, (_, i) => (i % 2 ? 0x80000000 : 0)),
    );
  }
  return { x, w, r };
};

const PART = "fusion-rms-norm-add";

/** The cases of {@link checkRmsNormAddTwins}: 5 widths × 2 orders × 5 row counts × 4 patterns. */
export const RMS_NORM_ADD_TWIN_CASES = { [PART]: 200 } as const satisfies ExpectedCases;

/**
 * Every width, order, row count and pattern: the Session with `fuseRmsNormAdd` against the one
 * without it (checkFusionTwin, NaN by class) on y and the later consumer of x (`copy`), rmsNormAdd
 * at 1 and 0, the fused side in 2 dispatches.
 */
export const checkRmsNormAddTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, RMS_NORM_ADD_TWIN_CASES[PART], options);
  for (const dim of RMS_NORM_ADD_DIMS) {
    for (const normFirst of RMS_NORM_ADD_ORDERS) {
      for (const rows of RMS_NORM_ADD_ROWS) {
        sink.next();
        const graph = { rows, dim, normFirst };
        const where = `dim=${dim} / normFirst=${normFirst}`;
        await withTwinSessions(
          gpu,
          `RMS→add ${where} rows=${rows}`,
          {
            model: await openModelBytes(rmsNormAddGraph(graph), []),
            options: { fuseRmsNormAdd: true },
          },
          { model: await openModelBytes(rmsNormAddGraph({ ...graph, normOutput: true }), []) },
          async ({ fused, plain }) => {
            for (const pattern of RMS_NORM_ADD_PATTERNS) {
              sink.next();
              const { x, w, r } = rmsNormAddInputs(rows, dim, pattern);
              const label = `${where}: ${rows}/${pattern}`;
              if (pattern === "cancel") {
                const { n } = await scopedGpuWork(gpu, `${label}: norm`, () =>
                  plain.run({ x, w, r }),
                );
                for (let i = 0; i < n.data.length; i++) r.data[i] = -n.data[i];
              }
              // y and the late x consumer (`copy`); the reference also outputs the norm.
              sink.add(
                await checkFusionTwin(gpu, {
                  label,
                  fused,
                  plain,
                  feeds: { x, w, r },
                  counter: "rmsNormAdd",
                  hits: 1,
                  nan: "class",
                  outputs: ["y", "copy"],
                  dispatches: { fused: 2 },
                }),
              );
            }
          },
        );
      }
    }
  }
  return {
    id: PART,
    title: "rmsNormAdd: fused rms_norm + add vs unfused (widths, orders, row counts, patterns)",
    cases: sink.cases,
    requirements: [],
  };
};

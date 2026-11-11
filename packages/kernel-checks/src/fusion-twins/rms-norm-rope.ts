/**
 * The fusion rule rmsNormRope (the runtime's src/runtime/fusion-rules/rms-norm-rope.ts) in a
 * Session: the Wan DiT's interleaved q / k prologues — rms_norm → two reshapes → the pair-form RoPE,
 * with k's nodes deferred behind q's fused step — against the same graph with a zero-dispatch
 * reshape after each rms_norm, which breaks the match and leaves the plain path (rms_norm, then the
 * pair RoPE folded alone by the rope rule) on the same backend. u32 identity with NaN by payload, two
 * fusions, 3 dispatches against 5.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  fill,
  type FilledTensor,
  openModelBytes,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { rmsNormRopeGraph } from "../../../runtime/tests/helpers/rms-norm-rope-graph.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases, wordsOf } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

/**
 * The graph inputs: q / k / v rows, the two norm weights and the tables, with special values in
 * q's first row (±0, a subnormal, 2e19, ±inf, a NaN payload) and an all-zero second row of k.
 */
export const rmsNormRopeInputs = (
  heads: number,
  headDim: number,
  sequence: number,
): Readonly<Record<string, FilledTensor>> => {
  const dim = heads * headDim;
  const row = [1, sequence, dim];
  const table = [1, sequence, 1, headDim];
  const q = fill(row, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  wordsOf(q.data).set([
    0x00000000, 0x80000000, 0x00000001, 0x5f0ac723, 0x7f800000, 0xff800000, 0x7fc01234,
  ]);
  const k = fill(row, (i) => Math.cos(i * 0.29) * 2.3 - i / 1531);
  if (sequence > 1) k.data.fill(0, dim, 2 * dim);
  return {
    q_x: q,
    k_x: k,
    v_x: fill(row, (i) => Math.sin(i * 0.11) - 0.5),
    q_w: fill([dim], (i) => 0.2 + Math.cos(i * 0.17)),
    k_w: fill([dim], (i) => 1.1 - Math.sin(i * 0.13)),
    cos: fill(table, (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill(table, (i) => Math.sin(i * 0.23 - 0.2)),
  };
};

/** The compared outputs: q's and k's prologue and v's `neg`. */
export const RMS_NORM_ROPE_OUTPUTS = ["q_y", "k_y", "v_out"] as const;

/**
 * (heads, headDim, sequence): Wan2.2 (24 × 128), Wan2.1 (12 × 128), and a row of 384 (its second
 * chunk half full).
 */
export const RMS_NORM_ROPE_SHAPES: readonly (readonly [number, number, number])[] = [
  [24, 128, 7],
  [12, 128, 5],
  [3, 128, 1],
];

const PART = "fusion-rms-norm-rope";

/** The cases of {@link checkRmsNormRopeTwins}: one per shape. */
export const RMS_NORM_ROPE_TWIN_CASES = { [PART]: 3 } as const satisfies ExpectedCases;

/**
 * Every shape: the fused Session against the interposed one (checkFusionTwin, NaN by payload) on
 * q_y, k_y and v_out, rmsNormRope at 2 and 0, 3 dispatches against 5.
 */
export const checkRmsNormRopeTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, RMS_NORM_ROPE_TWIN_CASES[PART], options);
  for (const [heads, headDim, sequence] of RMS_NORM_ROPE_SHAPES) {
    sink.next();
    const label = `H=${heads} D=${headDim} S=${sequence}`;
    await withTwinSessions(
      gpu,
      `rmsNormRope ${label}`,
      { model: await openModelBytes(rmsNormRopeGraph({ heads, headDim }), []) },
      { model: await openModelBytes(rmsNormRopeGraph({ heads, headDim, interpose: true }), []) },
      async ({ fused, plain }) => {
        sink.add(
          await checkFusionTwin(gpu, {
            label,
            fused,
            plain,
            feeds: rmsNormRopeInputs(heads, headDim, sequence),
            counter: "rmsNormRope",
            hits: 2,
            nan: "payload",
            outputs: RMS_NORM_ROPE_OUTPUTS,
            // q and k one dispatch each, plus v's neg; the plain path runs both rms_norms and both
            // pair RoPEs (the reshapes are zero-dispatch aliases).
            dispatches: { fused: 3, plain: 5 },
          }),
        );
      },
    );
  }
  return {
    id: PART,
    title: "rmsNormRope: fused vs rms_norm → pair RoPE (Wan q / k prologues)",
    cases: sink.cases,
    requirements: [],
  };
};

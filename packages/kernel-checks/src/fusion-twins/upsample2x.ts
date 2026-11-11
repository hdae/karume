/**
 * The strict peephole for the VAE's nearest-exact x2 (fusion rule upsample2x — the runtime's
 * src/runtime/fusion.ts) against the exporter's two expands on the same backend: replacing them with
 * a private 1-pass copy still duplicates every word (u32 identity with NaN by class against the f32
 * primitives), in 1 dispatch against 2.
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  openModelBytes,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases, wordsOf } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

/** B>1 / C>1, odd H/W, and 2*3*5*7=210 is not a multiple of the workgroup width 256. */
export const UPSAMPLE2X_SHAPE = [2, 3, 5, 7] as const;
const FIRST_RESHAPE = [30, 7, 1] as const;

/** Builds the real exporter's reshape→expand→reshape→reshape→expand→reshape verbatim. */
export const upsample2xGraph = (interpose: boolean, scaleWidth = 2): DeclarationJson => {
  const wide = [FIRST_RESHAPE[0], FIRST_RESHAPE[1], scaleWidth];
  const wideReshape = [6, 5, 7 * scaleWidth];
  const tallReshape = [6, 5, 1, 7 * scaleWidth];
  const tall = [6, 5, 2, 7 * scaleWidth];
  const out = [2, 3, 10, 7 * scaleWidth];
  const values: DeclarationJson["values"] = {
    first_reshape: { dtype: "f32", shape: [...FIRST_RESHAPE] },
    wide: { dtype: "f32", shape: wide },
    wide_reshape: { dtype: "f32", shape: wideReshape },
    tall_reshape: { dtype: "f32", shape: tallReshape },
    tall: { dtype: "f32", shape: tall },
    y: { dtype: "f32", shape: out },
  };
  if (interpose) values.first_alias = { dtype: "f32", shape: [...FIRST_RESHAPE] };
  const nodes: DeclarationJson["nodes"] = [
    { op: "reshape", ins: ["x"], outs: ["first_reshape"], attrs: {} },
  ];
  if (interpose) {
    // Inserts only a same-shape alias: neither the values nor the physical dispatch count change,
    // only the exact matcher stops matching.
    nodes.push({ op: "reshape", ins: ["first_reshape"], outs: ["first_alias"], attrs: {} });
  }
  nodes.push(
    { op: "expand", ins: [interpose ? "first_alias" : "first_reshape"], outs: ["wide"], attrs: {} },
    { op: "reshape", ins: ["wide"], outs: ["wide_reshape"], attrs: {} },
    { op: "reshape", ins: ["wide_reshape"], outs: ["tall_reshape"], attrs: {} },
    { op: "expand", ins: ["tall_reshape"], outs: ["tall"], attrs: {} },
    { op: "reshape", ins: ["tall"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["reshape", "expand"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [...UPSAMPLE2X_SHAPE] }],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * The copy fusion does no arithmetic, so every bit pattern of x must be preserved: ±0, subnormals,
 * ±Inf and NaN payloads at its head.
 */
export const upsample2xInput = (): FilledTensor => {
  const x = fill(UPSAMPLE2X_SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  wordsOf(x.data).set([
    0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000,
    0x7fc01234, 0xff801234,
  ]);
  return x;
};

const PART = "fusion-upsample2x";

/** The cases of {@link checkUpsample2xTwins}: one. */
export const UPSAMPLE2X_TWIN_CASES = { [PART]: 1 } as const satisfies ExpectedCases;

/**
 * The fused Session against the interposed one (checkFusionTwin, NaN by class — the f32 primitives
 * may canonicalize a NaN), upsample2x at 1 and 0, the copy in 1 pass against expand x2.
 */
export const checkUpsample2xTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, UPSAMPLE2X_TWIN_CASES[PART], options);
  sink.next();
  await withTwinSessions(
    gpu,
    "upsample x2",
    { model: await openModelBytes(upsample2xGraph(false), []) },
    { model: await openModelBytes(upsample2xGraph(true), []) },
    async ({ fused, plain }) => {
      sink.add(
        await checkFusionTwin(gpu, {
          label: "fused vs primitive",
          fused,
          plain,
          feeds: { x: upsample2xInput() },
          counter: "upsample2x",
          hits: 1,
          nan: "class",
          // The fused copy in 1 pass against expand x2.
          dispatches: { fused: 1, plain: 2 },
        }),
      );
    },
  );
  return {
    id: PART,
    title: "upsample2x: fused copy vs expand → expand",
    cases: sink.cases,
    requirements: [],
  };
};

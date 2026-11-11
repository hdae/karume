/**
 * Every built-in fusion against its unfused plan (ADR 0147 decision 9, layer 1): the twin cases of
 * the runtime's fusion tests, one part per rule in a fixed order — the attention permute rule in two
 * parts, the routes any device takes and the online routes in the sg32 form. A part the device
 * cannot run comes back with its reason in `notRun` (the linear prologue without dp4a, the sg32
 * routes on a device whose online attention takes another form).
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  type CheckOptions,
  type CheckPart,
  type ExpectedCases,
  throwIfAborted,
} from "../result.ts";
import { ADALN_TWIN_CASES, checkAdalnTwins } from "./adaln.ts";
import {
  ATTENTION_PERMUTE_TWIN_CASES,
  checkAttentionPermuteSg32Twins,
  checkAttentionPermuteTwins,
} from "./attention-permute.ts";
import { checkGatedResidualTwins, GATED_RESIDUAL_TWIN_CASES } from "./gated-residual.ts";
import {
  checkGeluMulStaticQuantizeTwins,
  GELU_MUL_STATIC_QUANTIZE_TWIN_CASES,
} from "./gelu-mul-static-quantize.ts";
import { checkLinearPrologueTwins, LINEAR_PROLOGUE_TWIN_CASES } from "./linear-prologue.ts";
import {
  checkLinearStaticQuantizeTwins,
  LINEAR_STATIC_QUANTIZE_TWIN_CASES,
} from "./linear-static-quantize.ts";
import { checkRmsNormAddTwins, RMS_NORM_ADD_TWIN_CASES } from "./rms-norm-add.ts";
import { checkRmsNormRopeTwins, RMS_NORM_ROPE_TWIN_CASES } from "./rms-norm-rope.ts";
import { checkRopeTwins, ROPE_TWIN_CASES } from "./rope.ts";
import {
  checkRowBlockAttentionTwins,
  ROW_BLOCK_ATTENTION_TWIN_CASES,
} from "./row-block-attention.ts";
import { checkSiluTwins, SILU_TWIN_CASES } from "./silu.ts";
import { checkUpsample2xTwins, UPSAMPLE2X_TWIN_CASES } from "./upsample2x.ts";

type PartCheck = (gpu: GpuContext, options: CheckOptions) => Promise<CheckPart>;

/** The parts of {@link checkFusionTwins}, in the order it runs them. */
const PARTS: readonly PartCheck[] = [
  checkGatedResidualTwins,
  checkSiluTwins,
  checkAdalnTwins,
  checkRopeTwins,
  checkUpsample2xTwins,
  checkRmsNormRopeTwins,
  checkLinearPrologueTwins,
  checkAttentionPermuteTwins,
  checkAttentionPermuteSg32Twins,
  checkRowBlockAttentionTwins,
  checkRmsNormAddTwins,
  checkGeluMulStaticQuantizeTwins,
  checkLinearStaticQuantizeTwins,
];

/** The cases each part of {@link checkFusionTwins} returns, in its order. */
export const FUSION_TWIN_CASES = {
  ...GATED_RESIDUAL_TWIN_CASES,
  ...SILU_TWIN_CASES,
  ...ADALN_TWIN_CASES,
  ...ROPE_TWIN_CASES,
  ...UPSAMPLE2X_TWIN_CASES,
  ...RMS_NORM_ROPE_TWIN_CASES,
  ...LINEAR_PROLOGUE_TWIN_CASES,
  ...ATTENTION_PERMUTE_TWIN_CASES,
  ...ROW_BLOCK_ATTENTION_TWIN_CASES,
  ...RMS_NORM_ADD_TWIN_CASES,
  ...GELU_MUL_STATIC_QUANTIZE_TWIN_CASES,
  ...LINEAR_STATIC_QUANTIZE_TWIN_CASES,
} as const satisfies ExpectedCases;

/** Runs every part, one after another (each case under the error scopes of its check). */
export const checkFusionTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart[]> => {
  const parts: CheckPart[] = [];
  for (const check of PARTS) {
    throwIfAborted(options);
    parts.push(await check(gpu, options));
  }
  return parts;
};

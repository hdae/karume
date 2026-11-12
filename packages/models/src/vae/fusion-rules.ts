/**
 * The fusion rules of the VAE decoder graphs Wan (2.1 and 2.2) and anima share the norm of: the
 * rule `models/vae_rms_silu` (`./vae-rms-silu.ts` — adopted on 2026-11-12, perf-ledger V-10), then
 * the rules the caller gives through the pipeline's experimental `fusionRules`. One list for the
 * Sessions of Wan's VAE decoder chunk graphs (`../wan/tile-decode.ts`: the 16-latent pair, and with
 * `vaeTile` 32 its lenders and the 32-latent borrowers) and of anima's VAE decoder, and for anima's
 * estimate of that graph (`../anima/pipeline.ts`). Wan2.2's VAE encoder, the DiTs and the text
 * encoders do not get the rule.
 *
 * The rule is on by default with no option to turn it off (docs/kernel-rules.md 4(a): a rule
 * u32-identical to the unfused plan takes no boolean key). Its evidence is the fused-against-unfused
 * u32 check (packages/e2e/tests/e2e_vae_rms_silu_rule_test.ts) and its hits on the real graphs
 * (the Wan2.1, Wan2.2 and anima fusion-count tests of packages/e2e). A caller that gives the same
 * rule again is refused by the Session (a rule given twice).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 *
 * DECIDED: docs/decisions/0148-external-fusion-rules.md
 */

import type { PipelineFusionRules } from "../session/fusion-rules.ts";
import { VAE_RMS_SILU_RULE } from "./vae-rms-silu.ts";

/** The rule list of a VAE decoder graph: `models/vae_rms_silu` first, then `callerRules` in order. */
export const vaeDecoderFusionRules = (
  callerRules: PipelineFusionRules,
): NonNullable<PipelineFusionRules> => [VAE_RMS_SILU_RULE, ...(callerRules ?? [])];

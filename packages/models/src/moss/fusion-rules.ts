/**
 * The fusion rules of the MOSS-Image image graphs: the family's own rule `models/ew_chain`
 * (`./ew-chain.ts` — adopted on 2026-11-12, perf-ledger F-13), then the rules the caller gives through
 * the experimental `fusionRules` of the pipeline. One list for the Sessions of `dit_text`, `dit`,
 * `dit_skip22` and the three VAE decoder graphs and for the headroom estimates of the same graphs
 * (`./dit-stage.ts`, `./vae-stage.ts`); the Qwen3.5 text encoder's Session gets none.
 *
 * The rule is on by default with no option to turn it off (docs/kernel-rules.md 4(a): a rule
 * u32-identical to the unfused plan takes no boolean key). Its evidence is the fused-against-unfused
 * u32 check (packages/e2e/tests/e2e_ew_chain_rule_test.ts) and its hits on the real graphs
 * (packages/e2e/tests/e2e_moss_fusion_counts_test.ts). A caller that gives the same rule again is
 * refused by the Session (a rule given twice).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 *
 * DECIDED: docs/decisions/0148-external-fusion-rules.md
 */

import type { PipelineFusionRules } from "../session/fusion-rules.ts";
import { EW_CHAIN_RULE } from "./ew-chain.ts";

/** The rule list of a MOSS-Image image graph: `models/ew_chain` first, then `callerRules` in order. */
export const mossFusionRules = (
  callerRules: PipelineFusionRules,
): NonNullable<PipelineFusionRules> => [EW_CHAIN_RULE, ...(callerRules ?? [])];

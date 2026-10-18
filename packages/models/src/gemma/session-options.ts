/** Overrides Gemma's quant execution settings with the caller's explicit options (ADR 0104). */
import type { SessionSpec } from "@karume/hub";
import type { SessionOptions } from "@karume/runtime";

import {
  assertSessionOverrides,
  type FamilySessionPolicy,
  resolveSessionOptions,
} from "../session/options.ts";

type GemmaSessionOptions = Pick<
  SessionOptions,
  | "linearGemvReduce"
  | "stateAttentionReduce"
  | "fuseRmsNormAdd"
  | "fuseLinearStaticQuantize"
  | "packedStaticQuantize"
>;

/**
 * The keys Gemma accepts (both in the manifest declaration and as explicit options). Precedence,
 * value ranges, combinations and the classification of the thrown type live in
 * {@link resolveSessionOptions}, shared by all families; this only decides the accepted set.
 *
 * The linear / attention execution forms (`linearCompute` / `attentionCompute` /
 * `attentionScoreStorage`) are not accepted because Gemma's distribution declares only five fields —
 * parallel GEMV, the reduction form of states-form attention, and fusion — and any other declaration
 * is an unverified combination (ADR 0104).
 *
 * `stateAttentionReduce` follows the same rule as the other fields (ADR 0104 addendum 2026-09-26): a
 * quant preset with no declaration (the reference `i4`) and `fromAssets`, which has no manifest, run
 * on the runtime's reference path `"sequential"`. The family keeps no code default — one would take
 * even presets with an empty `session` off the reference path.
 *
 * NOTE: `export` is because the equivalence test runs every quant of the mirror through the same
 * table (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const GEMMA_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: true,
  stateAttentionReduce: true,
  fuseRmsNormAdd: true,
  fuseLinearStaticQuantize: true,
  packedStaticQuantize: true,
};

/**
 * The guard that looks **only at the caller's explicit options** (input error = `ModelInputError`).
 *
 * MUST: `fromAssets` is a surface with no quant declaration, so it passes this guard **before
 * opening a single byte of the assets** (otherwise the same mistaken option would travel down to
 * Session construction, turn into the runtime's `ExecutionError`, and split the classification from
 * `fromPretrained`). It does not look at combinations checked against a quant declaration, so it
 * decides the same as against the default `{}`.
 *
 * NOTE: `export` is because the two entry points pass the same guard (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */
export const assertGemmaSessionOverrides = (overrides: GemmaSessionOptions, where: string): void =>
  assertSessionOverrides(GEMMA_SESSION_POLICY, overrides, where);

// DECIDED: docs/decisions/0104-gemma-fast-quant.md
export const resolveGemmaSessionOptions = (
  quant: SessionSpec,
  overrides: GemmaSessionOptions,
  where: string,
): GemmaSessionOptions => resolveSessionOptions(GEMMA_SESSION_POLICY, quant, overrides, where);

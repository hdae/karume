/** Overrides Gemma's quant execution settings with the caller's explicit options (ADR 0104). */
import type { SessionSpec } from "@karume/hub";
import { DEFAULT_SUBMIT_POLICY, type SessionOptions, type SubmitPolicy } from "@karume/runtime";

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
  convCompute: false,
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

/** The dispatch cap per submit of Gemma's default submission policy ({@link gemmaSubmitPolicy}). */
export const GEMMA_MAX_CHUNK_SIZE = 128;

/**
 * The submission policy of Gemma's Sessions (perf-ledger H-40): the caller's own policy, otherwise
 * the runtime default with at most {@link GEMMA_MAX_CHUNK_SIZE} dispatches per submit. A decode token
 * is about 900 short dispatches; sent as one submit, the GPU waits for the whole host encode and
 * Dawn's command recording, while 128-dispatch chunks let it start on the first one. The commands
 * and their order do not change, so the output is bit-identical. Measured 2026-10-25 (gemma4 E2B
 * `i4-fast`): Node 10.1 → 7.4 ms per token, Chromium-Linux 10.0 → 8.0, Chrome on the RTX 5070 Ti
 * 11.73 → 10.03; prefill at short prompts and speculative decoding were not slower.
 *
 * Derived from `DEFAULT_SUBMIT_POLICY` at each call, so a change of the runtime default still
 * reaches this family (only the cap differs). It is a host policy, not a quant knob, so it is not in
 * {@link GEMMA_SESSION_POLICY}.
 */
export const gemmaSubmitPolicy = (requested: SubmitPolicy | undefined): SubmitPolicy =>
  requested ?? { ...DEFAULT_SUBMIT_POLICY, maxChunkSize: GEMMA_MAX_CHUNK_SIZE };

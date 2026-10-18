/**
 * **Route decision** of fused attention (requested options × shape → the route each stage actually
 * runs) — the single source of truth for that decision.
 *
 * a8 (`attentionCompute`) and s16 (`attentionScoreStorage`) fall back silently to the f32 route only
 * for the stages whose shape conditions are not met (ADR 0030 decision 5 / ADR 0058 decision 3).
 * Recipe derivation (`buildAttention` in recipe-builders/attention.ts) and the tests that run a
 * family's acceptance set call the **same function** — writing the decision twice lets one copy move
 * alone, and "eligible in the tests, fallen back at run time" happens silently.
 *
 * NOTE: the `export` is for recipe derivation and the tests (not part of the public surface of
 * mod.ts — ADR 0008).
 */

import { attentionPvI8a8Eligible, attentionQkI8a8Eligible } from "../kernels/attention-i8a8.ts";
import { attentionScoreUsesF16, type ScoreStorage } from "../kernels/score-storage.ts";
import type { ComputePrecision } from "./session-types.ts";

/** Route of a stage (①QK / ③PV). `dense` is the f32 / f16 GEMM skeleton (`attentionCompute` f32 / f16). */
export type AttentionStageRoute = "i8a8" | "dense";

/** The routes one fused attention actually runs (② row statistics always compute in f32; only the storage form of S moves). */
export type AttentionRoute = {
  readonly qk: AttentionStageRoute;
  readonly pv: AttentionStageRoute;
  readonly score: ScoreStorage;
};

/** The options that decide the route (the Session's resolved values — filling omitted values is session-build.ts's job). */
export type AttentionSeats = {
  readonly attentionCompute: ComputePrecision;
  readonly attentionScoreStorage: ScoreStorage;
};

/**
 * Decides, from the requested options and the shape (`depth` = D, `cols` = key length N), the route
 * each stage actually runs.
 *
 * - i8a8: MUST: eligibility is decided **independently per stage**. ① packs q / k along D and ③
 *   packs P̃ / Vᵀ along N, so the conditions differ too (shapes that meet neither and shapes that
 *   meet only one both exist).
 * - Storage form of S: `pack2x16float` holds 2 elements per word, so only shapes where the writer
 *   ①QK takes the v4 route (`D % 4 == 0 && N % 4 == 0`) are eligible. MUST: the storage form of S
 *   switches in all 3 kernels (①②③) **at once** — writer and readers must agree, so mixing per
 *   stage cannot happen (a separate matter from the per-stage independence of i8a8: whether ① is
 *   i8a8 or f32, the storage form of S is decided as one).
 *
 * MUST: the rejections of mask × a8 and GQA × a8 look at the **requested mode**
 * (`attentionCompute === "a8"`), not at this return value (the MUST of buildAttention — looking at
 * the value after eligibility lets exactly the shapes that fall back slip past the rejection).
 */
export const attentionRoute = (
  seats: AttentionSeats,
  depth: number,
  cols: number,
): AttentionRoute => {
  const i8a8 = seats.attentionCompute === "a8";
  return {
    qk: i8a8 && attentionQkI8a8Eligible(depth) ? "i8a8" : "dense",
    pv: i8a8 && attentionPvI8a8Eligible(cols) ? "i8a8" : "dense",
    score:
      seats.attentionScoreStorage === "f16" && attentionScoreUsesF16(depth, cols) ? "f16" : "f32",
  };
};

/**
 * **Route decision** of fused attention (requested options × shape → the route each stage actually
 * runs) — the single source of truth for that decision.
 *
 * a8 (`attentionCompute`) and s16 (`attentionScoreStorage`) fall back silently to the f32 route only
 * for the stages whose shape conditions are not met (ADR 0030 decision 5 / ADR 0058 decision 3).
 * The online kernel (`attentionScoreStorage: "none"` — ADR 0132) falls back to the three dispatches
 * of a8 with S in f32 when the shape or the device does not admit it.
 * Recipe derivation (`buildAttention` in recipe-builders/attention.ts) and the tests that run a
 * family's acceptance set call the **same function** — writing the decision twice lets one copy move
 * alone, and "eligible in the tests, fallen back at run time" happens silently.
 *
 * NOTE: the `export` is for recipe derivation and the tests (not part of the public surface of
 * mod.ts — ADR 0008).
 */

import { attentionPvI8a8Eligible, attentionQkI8a8Eligible } from "../kernels/attention-i8a8.ts";
import {
  type AttentionOnlineReduce,
  attentionOnlineI8a8Eligible,
  attentionOnlineI8a8WorkgroupStorageBytes,
} from "../kernels/attention-online-i8a8.ts";
import {
  attentionScoreUsesF16,
  type MaterialisedScoreStorage,
  type ScoreStorage,
} from "../kernels/score-storage.ts";
import type { ComputePrecision } from "./session-types.ts";

/** Route of a stage (①QK / ③PV). `dense` is the f32 / f16 GEMM skeleton (`attentionCompute` f32 / f16). */
export type AttentionStageRoute = "i8a8" | "dense";

/**
 * The routes one fused attention actually runs: either the three dispatches (② row statistics always
 * compute in f32; only the storage form of S moves) or the online kernel (`online` set — S is never
 * stored, ①③ are its i8a8 quantisation; ADR 0132). `online` doubles as the discriminant, so `score`
 * of the three-dispatch form can never be `"none"`.
 */
export type AttentionRoute =
  | {
      readonly qk: AttentionStageRoute;
      readonly pv: AttentionStageRoute;
      readonly score: MaterialisedScoreStorage;
      readonly online?: undefined;
    }
  | {
      readonly qk: "i8a8";
      readonly pv: "i8a8";
      readonly score: "none";
      /** The reduction form of the online kernel (bit-identical either way — ADR 0132 decision 5). */
      readonly online: AttentionOnlineReduce;
    };

/**
 * The device facts the online kernel's route needs (ADR 0132 decision 5). Fixed once per Session
 * (session-build.ts) and passed explicitly to the estimate (`EstimateOptions.attentionDevice`).
 */
export type AttentionDeviceFacts = {
  /** `subgroups` and `subgroup-size-control` enabled on the device and WGSL `subgroup_id` present. */
  readonly subgroups: boolean;
  /** The device's granted `maxComputeWorkgroupStorageSize`. */
  readonly maxComputeWorkgroupStorageSize: number;
};

/**
 * Reads {@link AttentionDeviceFacts} from a device context (a structural parameter so that this
 * module stays free of GPU imports — the estimate imports it).
 * MUST: the subgroup fact is the three conditions the `sg32` form compiles under (the features the
 * device **enabled**, not those the adapter merely advertises — the same reading as the
 * `rmsNormReduce: "subgroup32"` check of session-build.ts) **and** the measured linear lane
 * mapping (`GpuContext.subgroupLanesLinear` — the form addresses its tiles by
 * `local_invocation_index`, ADR 0132 decision 5).
 */
export const attentionDeviceFacts = (gpu: {
  readonly features: ReadonlySet<string>;
  readonly wgslLanguageFeatures: ReadonlySet<string>;
  readonly subgroupLanesLinear: boolean;
  readonly limits: { readonly maxComputeWorkgroupStorageSize: number };
}): AttentionDeviceFacts => ({
  subgroups:
    gpu.features.has("subgroups") &&
    gpu.features.has("subgroup-size-control") &&
    gpu.wgslLanguageFeatures.has("subgroup_id") &&
    gpu.subgroupLanesLinear,
  maxComputeWorkgroupStorageSize: gpu.limits.maxComputeWorkgroupStorageSize,
});

/** The options that decide the route (the Session's resolved values — filling omitted values is session-build.ts's job). */
export type AttentionSeats = {
  readonly attentionCompute: ComputePrecision;
  readonly attentionScoreStorage: ScoreStorage;
};

/** The online kernel's reduction form the device holds (`undefined` = neither fits — the route falls back). */
const onlineReduce = (
  depth: number,
  device: AttentionDeviceFacts,
): AttentionOnlineReduce | undefined => {
  const fits = (reduce: AttentionOnlineReduce): boolean =>
    attentionOnlineI8a8WorkgroupStorageBytes(depth, reduce) <=
    device.maxComputeWorkgroupStorageSize;
  if (device.subgroups && fits("sg32")) return "sg32";
  return fits("wg") ? "wg" : undefined;
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
 * - Online kernel (`attentionScoreStorage: "none"` × `attentionCompute: "a8"` — ADR 0132): chosen
 *   when the shape is eligible (`D ∈ {64, 128}`, any `N ≥ 1` — `N % 4 != 0` runs with a padded Vᵀ,
 *   ADR 0132 addendum 2026-10-27, while ③PV's i8a8 above keeps `N % 4 == 0`) and the device
 *   holds one of its reduction forms — `sg32` when the device has subgroups and its tiles fit,
 *   otherwise `wg` when the workgroup-storage limit holds its scratch too. Otherwise the route falls
 *   back to the three dispatches of a8 with S in f32 (never `"none"` — the three kernels cannot
 *   store it).
 *
 * MUST: the rejections of mask × a8 and GQA × a8 look at the **requested mode**
 * (`attentionCompute === "a8"`), not at this return value (the MUST of buildAttention — looking at
 * the value after eligibility lets exactly the shapes that fall back slip past the rejection).
 */
export const attentionRoute = (
  seats: AttentionSeats,
  depth: number,
  cols: number,
  device: AttentionDeviceFacts,
): AttentionRoute => {
  const i8a8 = seats.attentionCompute === "a8";
  const online =
    seats.attentionScoreStorage === "none" && i8a8 && attentionOnlineI8a8Eligible(depth, cols)
      ? onlineReduce(depth, device)
      : undefined;
  if (online !== undefined) return { qk: "i8a8", pv: "i8a8", score: "none", online };
  return {
    qk: i8a8 && attentionQkI8a8Eligible(depth) ? "i8a8" : "dense",
    pv: i8a8 && attentionPvI8a8Eligible(cols) ? "i8a8" : "dense",
    score:
      seats.attentionScoreStorage === "f16" && attentionScoreUsesF16(depth, cols) ? "f16" : "f32",
  };
};

/**
 * The device facts to pass to {@link attentionRoute} on a seat whose route never reads them: a device
 * with neither subgroups nor workgroup storage.
 *
 * MUST: only on a seat other than `attentionCompute` `"a8"` × `attentionScoreStorage` `"none"`.
 * `attentionRoute` reads the device facts only on that seat (to choose the online kernel's reduction
 * form), so elsewhere this stand-in cannot change the route. On that seat the route depends on the
 * device, so a caller without the real facts fails loudly instead of passing this.
 */
export const UNREAD_ATTENTION_DEVICE: AttentionDeviceFacts = {
  subgroups: false,
  maxComputeWorkgroupStorageSize: 0,
};

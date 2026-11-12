/**
 * The op families of the geometry sweep (`GeometryProfileFamily` of src/kernels/geometry-profile.ts —
 * DECIDED: ADR 0117 addendum 2026-11-12, "per op family"): which family each sweep op and each
 * profile field belongs to, and what a family's matching keys are taken over (its cases and, for
 * `gemm`, the row-count bucket bounds). A table's case-set version and kernel fingerprint, and a
 * sweep record's `caseSets` and `defaultKernels`, are one value per family, so adding one family's
 * cases moves only that family's keys.
 *
 * MUST: nothing is computed at import time (tables of constants and functions only).
 */
import {
  GEOMETRY_PROFILE_FAMILIES,
  type GeometryProfileFamily,
} from "../kernels/geometry-profile.ts";
import { PROFILE_GEMM_ROWS_BOUNDS, type SweepOp } from "./cases.ts";
import type { GemmRowsSlot, ProfileSlot } from "./derive.ts";

/**
 * One value per op family, for the families present (the shape of the per-family matching keys: the
 * sweep record's `caseSets` / `defaultKernels`, `sweepCaseSetIds` and `sweepDefaultKernelsIds`).
 */
export type GeometryProfileFamilyIds = { readonly [Family in GeometryProfileFamily]?: string };

/**
 * The family of each sweep op. MUST: a new sweep op is classified here (`satisfies` requires every
 * op — a forgotten op is not silently left out of every family's keys).
 */
const SWEEP_OP_FAMILIES = {
  linear: "gemm",
  matmul: "gemm",
  bmm: "gemm",
  attention: "attention",
  conv2d: "conv2d",
  "i8a8-linear": "i8a8-linear",
  "i8a8-attention": "i8a8-attention",
  "i8a8-conv3d": "i8a8-conv3d",
} as const satisfies Record<SweepOp, GeometryProfileFamily>;

/** The family a sweep op belongs to. */
export const sweepOpFamily = (op: SweepOp): GeometryProfileFamily => SWEEP_OP_FAMILIES[op];

/**
 * The family of each profile field (`gemmRows` stands for every row-count bucket). MUST: a new field
 * is classified here (`satisfies` requires every slot of `ProfileSlot`).
 */
const PROFILE_FIELD_FAMILIES = {
  gemmRows: "gemm",
  "attention.qk": "attention",
  "attention.pv": "attention",
  "conv2d.rows64": "conv2d",
  "conv2d.rows32": "conv2d",
  "i8a8.linear": "i8a8-linear",
  "i8a8.attentionQk": "i8a8-attention",
  "i8a8.attentionPv": "i8a8-attention",
  "i8a8.conv3d": "i8a8-conv3d",
} as const satisfies Record<"gemmRows" | Exclude<ProfileSlot, GemmRowsSlot>, GeometryProfileFamily>;

const isGemmRowsSlot = (slot: ProfileSlot): slot is GemmRowsSlot => slot.startsWith("gemmRows ");

/** The family a profile field (= class of the derivation's verdicts) belongs to. */
export const profileSlotFamily = (slot: ProfileSlot): GeometryProfileFamily =>
  isGemmRowsSlot(slot) ? PROFILE_FIELD_FAMILIES.gemmRows : PROFILE_FIELD_FAMILIES[slot];

/** The entries of `entries` (cases or rows) that belong to `family`, in their order. */
export const familyEntries = <T extends { readonly op: SweepOp }>(
  entries: readonly T[],
  family: GeometryProfileFamily,
): T[] => entries.filter((entry) => sweepOpFamily(entry.op) === family);

/** The families that have at least one entry of `entries` (cases or rows), in the fixed order. */
export const familiesOf = (
  entries: readonly { readonly op: SweepOp }[],
): readonly GeometryProfileFamily[] =>
  GEOMETRY_PROFILE_FAMILIES.filter((family) =>
    entries.some((entry) => sweepOpFamily(entry.op) === family),
  );

/**
 * The bounds a family's case-set version hashes besides its cases: the row-count bucket bounds
 * (`PROFILE_GEMM_ROWS_BOUNDS`) for `gemm`, none for the other five. A family that gets bounds later
 * adds them here; the hashed form already has the entry, so the record and table formats do not
 * change.
 */
export const familyBounds = (family: GeometryProfileFamily): readonly number[] | undefined =>
  family === "gemm" ? PROFILE_GEMM_ROWS_BOUNDS : undefined;

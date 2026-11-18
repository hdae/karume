/**
 * The **case table** of the GEMV sweep (pure data): the shapes the GEMV family of `linear` serves in
 * the models — weight storage, i4 group length, `n` and `k` — each with where it comes from and the
 * row counts to measure.
 *
 * The shapes are those of the default geometry profile's GEMV choices (`DEFAULT_GEOMETRY_PROFILE`
 * `.linearGemv` in src/kernels/geometry-profile.ts): every parallel row and every per-shape
 * exception of the sequential variant has a case here (a test pins it). The table is its own data
 * rather than a reading of the profile, so that a shape can be measured before it has a row — as the
 * bf16 linears of regular Gemma 4 are, which no profile names yet. It also holds every other
 * integer-storage linear a decode token or a draft step of the Gemma 4 mirrors runs at M = 1 (the
 * output heads of the QAT lines, the MTP drafters), whether or not the default profile gives it a
 * row. The sources repeat the comments of the profile's rows; the Gemma 4 E2B rows have none, so
 * their roles are those the E4B rows name when they copy "the E2B row with the same role", and the
 * census of docs/research/2026-09-07-gemv-rows-k21.md (the 12288-wide MLP layers). The shapes
 * without a profile row are read from the graphs of the distribution mirrors.
 *
 * The row counts are the parallel family's range M = 1, 4, 8 and the row-block form's 16 and 64
 * ({@link GEMV_SWEEP_ROW_COUNTS}). A float-stored weight (bf16 / f16 / f32) has no row-block form —
 * past one row its reference kernel is the GEMM — so its cases stop at the parallel family's range
 * ({@link GEMV_SWEEP_FLOAT_ROW_COUNTS}).
 */

import { isFloatStorage } from "../kernels/linear-gemv.ts";

/**
 * The weight storages the GEMV sweep measures: the storages of the profile's GEMV rows, bf16 — the
 * storage regular Gemma 4 ships in —, and the other float storages the parallel GEMV takes, f16 and
 * f32 (the table has no case of those two yet).
 */
export type GemvSweepStorage = "i2" | "i4" | "i8" | "bf16" | "f16" | "f32";

/** One shape of the GEMV sweep. */
export type GemvCase = {
  /** `<storage>[g<group>]-n<n>-k<k>` (unique in the table). */
  readonly id: string;
  readonly storage: GemvSweepStorage;
  /** The i4 group length (i4 only). */
  readonly group?: number;
  /** Output columns (the weight's rows). */
  readonly n: number;
  /** The reduction length. */
  readonly k: number;
  /** The row counts M to measure, ascending. */
  readonly rows: readonly number[];
  /** The model, its weight storage and the role of the linear (several roles share some shapes). */
  readonly source: string;
};

/**
 * The row counts every case measures: 1, 4 and 8 (the parallel GEMV's range, which shares one
 * summation order across them) and 16 and 64 (the row-block form up to the GEMV family's last row
 * count).
 */
export const GEMV_SWEEP_ROW_COUNTS: readonly number[] = [1, 4, 8, 16, 64];

/**
 * The row counts a case of a float storage measures: 1, 4 and 8, the parallel GEMV's range. Beyond
 * it such a weight runs the GEMM under every level, which the geometry sweep measures.
 */
export const GEMV_SWEEP_FLOAT_ROW_COUNTS: readonly number[] = [1, 4, 8];

/** A case of the table, its id derived from its shape. */
const shape = (
  storage: GemvSweepStorage,
  group: number | undefined,
  n: number,
  k: number,
  source: string,
): GemvCase => ({
  id: `${storage}${group === undefined ? "" : `g${group}`}-n${n}-k${k}`,
  storage,
  ...(group === undefined ? {} : { group }),
  n,
  k,
  rows: isFloatStorage(storage) ? GEMV_SWEEP_FLOAT_ROW_COUNTS : GEMV_SWEEP_ROW_COUNTS,
  source,
});

const E2B = "Gemma 4 E2B, i4 g32";
const E2B_QAT = "Gemma 4 E2B QAT";
const E4B = "Gemma 4 E4B, i4 g32";
const E4B_QAT = "Gemma 4 E4B QAT";
const E2B_BF16 = "Gemma 4 E2B, bf16";
const E4B_BF16 = "Gemma 4 E4B, bf16";
const G12B = "Gemma 4 12B q4_0 QAT, i4 g32";
const E2B_DRAFTER = "Gemma 4 E2B q4_0 QAT drafter, i4 g32";
const E2B_DRAFTER_I8 = "Gemma 4 E2B drafter, i8";
const G12B_DRAFTER = "Gemma 4 12B q4_0 QAT drafter, i4 g32";
/** The drafter shapes the E2B and the E4B q4_0 QAT drafters share (their inner width is 256). */
const E_DRAFTERS = "Gemma 4 E2B / E4B q4_0 QAT drafter, i4 g32";
const G26B = "Gemma 4 26B A4B q4_0 QAT, i4 g32";

/**
 * The case table. The integer-storage cases come first, grouped by model in the order of the default
 * profile's rows: within a group the profile's rows first, then the group's shapes that have none
 * (the output heads, the MTP drafter). The bf16 cases of regular Gemma 4 follow.
 */
export const GEMV_CASES: readonly GemvCase[] = [
  // Gemma 4 E2B (regular, i4 g32 and the i8 output head).
  shape("i4", 32, 8960, 1536, `${E2B}: per_layer_model_projection (35 layers × 256)`),
  shape("i4", 32, 2048, 1536, `${E2B}: q (sliding)`),
  shape("i4", 32, 256, 1536, `${E2B}: k / v (sliding), per_layer_input_gate`),
  shape("i4", 32, 1536, 2048, `${E2B}: o (sliding)`),
  shape("i4", 32, 6144, 1536, `${E2B}: gate / up`),
  shape("i4", 32, 1536, 6144, `${E2B}: down`),
  shape("i4", 32, 1536, 256, `${E2B}: per_layer_projection`),
  shape("i4", 32, 4096, 1536, `${E2B}: q (full)`),
  shape("i4", 32, 512, 1536, `${E2B}: k / v (full)`),
  shape("i4", 32, 1536, 4096, `${E2B}: o (full)`),
  shape("i4", 32, 12288, 1536, `${E2B}: gate / up of the 12288-wide MLP layers`),
  shape("i4", 32, 1536, 12288, `${E2B}: down of the 12288-wide MLP layers`),
  shape(
    "i8",
    undefined,
    262144,
    1536,
    "Gemma 4 E2B, i8: lm_head (tied embedding); also the sequential variant's per-shape exception",
  ),
  shape("i4", 32, 262144, 1536, "Gemma 4 E2B q4_0 QAT, i4 g32: lm_head (tied embedding)"),
  // The MTP drafter of the q4_0 QAT E2B (one run drafts three tokens; post_projection 1536×256 is
  // the per_layer_projection case above).
  shape("i4", 32, 256, 3072, `${E2B_DRAFTER}: pre_projection`),
  shape("i4", 32, 1024, 256, `${E_DRAFTERS}: q (sliding)`),
  shape("i4", 32, 256, 1024, `${E_DRAFTERS}: o (sliding)`),
  shape("i4", 32, 2048, 256, `${E_DRAFTERS}: gate / up, q (full)`),
  shape("i4", 32, 256, 2048, `${E_DRAFTERS}: down, o (full)`),
  shape("i4", 32, 262144, 256, `${E_DRAFTERS}: lm_head`),
  // The MTP drafter of the regular E2B (i8; post_projection 1536×256 is the QAT case below).
  shape("i8", undefined, 256, 3072, `${E2B_DRAFTER_I8}: pre_projection`),
  shape("i8", undefined, 1024, 256, `${E2B_DRAFTER_I8}: q (sliding)`),
  shape("i8", undefined, 256, 1024, `${E2B_DRAFTER_I8}: o (sliding)`),
  shape("i8", undefined, 2048, 256, `${E2B_DRAFTER_I8}: gate / up, q (full)`),
  shape("i8", undefined, 256, 2048, `${E2B_DRAFTER_I8}: down, o (full)`),
  shape("i8", undefined, 262144, 256, `${E2B_DRAFTER_I8}: lm_head`),
  // Gemma 4 E2B QAT.
  shape("i4", 512, 2048, 1536, `${E2B_QAT}, i4 g512: q (sliding)`),
  shape("i4", 512, 256, 1536, `${E2B_QAT}, i4 g512: k / v (sliding)`),
  shape("i4", 2048, 1536, 2048, `${E2B_QAT}, i4 g2048: o (sliding)`),
  shape("i4", 512, 6144, 1536, `${E2B_QAT}, i4 g512: gate / up`),
  shape("i4", 2048, 1536, 6144, `${E2B_QAT}, i4 g2048: down`),
  shape("i8", undefined, 256, 1536, `${E2B_QAT}, i8: per_layer_input_gate`),
  shape("i8", undefined, 1536, 256, `${E2B_QAT}, i8: per_layer_projection`),
  shape("i4", 512, 4096, 1536, `${E2B_QAT}, i4 g512: q (full)`),
  shape("i4", 512, 512, 1536, `${E2B_QAT}, i4 g512: k / v (full)`),
  shape("i4", 4096, 1536, 4096, `${E2B_QAT}, i4 g4096: o (full)`),
  shape("i2", undefined, 12288, 1536, `${E2B_QAT}, i2: gate / up of the 12288-wide MLP layers`),
  shape("i2", undefined, 1536, 12288, `${E2B_QAT}, i2: down of the 12288-wide MLP layers`),
  shape("i2", undefined, 262144, 1536, `${E2B_QAT}, i2: lm_head (tied embedding)`),
  // Gemma 4 E4B (regular) — the profile's rows copy the lanes of the E2B row with the same role.
  shape("i4", 32, 10752, 2560, `${E4B}: per_layer_model_projection (42 layers × 256)`),
  shape("i4", 32, 2048, 2560, `${E4B}: q (sliding)`),
  shape("i4", 32, 512, 2560, `${E4B}: k / v (sliding)`),
  shape("i4", 32, 256, 2560, `${E4B}: per_layer_input_gate`),
  shape("i4", 32, 2560, 2048, `${E4B}: o (sliding)`),
  shape("i4", 32, 10240, 2560, `${E4B}: gate / up`),
  shape("i4", 32, 2560, 10240, `${E4B}: down`),
  shape("i4", 32, 2560, 256, `${E4B}: per_layer_projection`),
  shape("i4", 32, 4096, 2560, `${E4B}: q (full)`),
  shape("i4", 32, 1024, 2560, `${E4B}: k / v (full)`),
  shape("i4", 32, 2560, 4096, `${E4B}: o (full)`),
  shape("i8", undefined, 262144, 2560, "Gemma 4 E4B, i8: lm_head (tied embedding)"),
  shape("i4", 32, 262144, 2560, "Gemma 4 E4B q4_0 QAT, i4 g32: lm_head (tied embedding)"),
  // The MTP drafter of the q4_0 QAT E4B: its other shapes are the E2B drafter's above, and
  // post_projection 2560×256 is the per_layer_projection case above.
  shape("i4", 32, 256, 5120, "Gemma 4 E4B q4_0 QAT drafter, i4 g32: pre_projection"),
  // Gemma 4 E4B QAT.
  shape("i4", 512, 2048, 2560, `${E4B_QAT}, i4 g512: q (sliding)`),
  shape("i4", 512, 512, 2560, `${E4B_QAT}, i4 g512: k / v (sliding)`),
  shape("i4", 2048, 2560, 2048, `${E4B_QAT}, i4 g2048: o (sliding)`),
  shape("i4", 512, 10240, 2560, `${E4B_QAT}, i4 g512: gate / up`),
  shape("i4", 2048, 2560, 10240, `${E4B_QAT}, i4 g2048: down`),
  shape("i8", undefined, 256, 2560, `${E4B_QAT}, i8: per_layer_input_gate`),
  shape("i8", undefined, 2560, 256, `${E4B_QAT}, i8: per_layer_projection`),
  shape("i4", 512, 4096, 2560, `${E4B_QAT}, i4 g512: q (full)`),
  shape("i4", 512, 1024, 2560, `${E4B_QAT}, i4 g512: k / v (full)`),
  shape("i4", 4096, 2560, 4096, `${E4B_QAT}, i4 g4096: o (full)`),
  shape("i2", undefined, 262144, 2560, `${E4B_QAT}, i2: lm_head (tied embedding)`),
  // Gemma 4 12B q4_0 QAT.
  shape("i4", 32, 15360, 3840, `${G12B}: gate / up`),
  shape("i4", 32, 8192, 3840, `${G12B}: q (full)`),
  shape("i4", 32, 4096, 3840, `${G12B}: q (sliding)`),
  shape("i4", 32, 2048, 3840, `${G12B}: k / v (sliding)`),
  shape("i4", 32, 512, 3840, `${G12B}: k (full)`),
  shape("i4", 32, 3840, 4096, `${G12B}: o (sliding)`),
  shape("i4", 32, 3840, 8192, `${G12B}: o (full)`),
  shape("i4", 32, 3840, 15360, `${G12B}: down`),
  shape("i4", 32, 262144, 3840, `${G12B}: lm_head (tied embedding)`),
  // The MTP drafter of the q4_0 QAT 12B (inner width 1024).
  shape("i4", 32, 1024, 7680, `${G12B_DRAFTER}: pre_projection`),
  shape("i4", 32, 4096, 1024, `${G12B_DRAFTER}: q (sliding)`),
  shape("i4", 32, 1024, 4096, `${G12B_DRAFTER}: o (sliding)`),
  shape("i4", 32, 8192, 1024, `${G12B_DRAFTER}: gate / up, q (full)`),
  shape("i4", 32, 1024, 8192, `${G12B_DRAFTER}: down, o (full)`),
  shape("i4", 32, 3840, 1024, `${G12B_DRAFTER}: post_projection`),
  shape("i4", 32, 262144, 1024, `${G12B_DRAFTER}: lm_head`),
  // Gemma 4 26B A4B q4_0 QAT — the dense linears of the mixture of experts.
  shape("i4", 32, 8192, 2816, `${G26B}: q (full)`),
  shape("i4", 32, 4096, 2816, `${G26B}: q (sliding)`),
  shape("i4", 32, 2048, 2816, `${G26B}: k / v (sliding)`),
  shape("i4", 32, 2112, 2816, `${G26B}: gate / up`),
  shape("i4", 32, 1024, 2816, `${G26B}: k (full)`),
  shape("i4", 32, 128, 2816, `${G26B}: router`),
  shape("i4", 32, 2816, 4096, `${G26B}: o (sliding)`),
  shape("i4", 32, 2816, 8192, `${G26B}: o (full)`),
  shape("i4", 32, 2816, 2112, `${G26B}: down`),
  shape("i4", 32, 262144, 2816, `${G26B}: lm_head (tied embedding)`),
  // Gemma 4 E2B (regular) as it ships: every linear in bf16 (ADR 0128). The shapes are the i4 g32
  // cases' and the output head's.
  shape("bf16", undefined, 8960, 1536, `${E2B_BF16}: per_layer_model_projection (35 layers × 256)`),
  shape("bf16", undefined, 2048, 1536, `${E2B_BF16}: q (sliding)`),
  shape("bf16", undefined, 256, 1536, `${E2B_BF16}: k / v (sliding), per_layer_input_gate`),
  shape("bf16", undefined, 1536, 2048, `${E2B_BF16}: o (sliding)`),
  shape("bf16", undefined, 6144, 1536, `${E2B_BF16}: gate / up`),
  shape("bf16", undefined, 1536, 6144, `${E2B_BF16}: down`),
  shape("bf16", undefined, 1536, 256, `${E2B_BF16}: per_layer_projection`),
  shape("bf16", undefined, 4096, 1536, `${E2B_BF16}: q (full)`),
  shape("bf16", undefined, 512, 1536, `${E2B_BF16}: k / v (full)`),
  shape("bf16", undefined, 1536, 4096, `${E2B_BF16}: o (full)`),
  shape("bf16", undefined, 12288, 1536, `${E2B_BF16}: gate / up of the 12288-wide MLP layers`),
  shape("bf16", undefined, 1536, 12288, `${E2B_BF16}: down of the 12288-wide MLP layers`),
  shape("bf16", undefined, 262144, 1536, `${E2B_BF16}: lm_head (tied embedding)`),
  // Gemma 4 E4B (regular), bf16.
  shape(
    "bf16",
    undefined,
    10752,
    2560,
    `${E4B_BF16}: per_layer_model_projection (42 layers × 256)`,
  ),
  shape("bf16", undefined, 2048, 2560, `${E4B_BF16}: q (sliding)`),
  shape("bf16", undefined, 512, 2560, `${E4B_BF16}: k / v (sliding)`),
  shape("bf16", undefined, 256, 2560, `${E4B_BF16}: per_layer_input_gate`),
  shape("bf16", undefined, 2560, 2048, `${E4B_BF16}: o (sliding)`),
  shape("bf16", undefined, 10240, 2560, `${E4B_BF16}: gate / up`),
  shape("bf16", undefined, 2560, 10240, `${E4B_BF16}: down`),
  shape("bf16", undefined, 2560, 256, `${E4B_BF16}: per_layer_projection`),
  shape("bf16", undefined, 4096, 2560, `${E4B_BF16}: q (full)`),
  shape("bf16", undefined, 1024, 2560, `${E4B_BF16}: k / v (full)`),
  shape("bf16", undefined, 2560, 4096, `${E4B_BF16}: o (full)`),
  shape("bf16", undefined, 262144, 2560, `${E4B_BF16}: lm_head (tied embedding)`),
];

/** The case of an id, or `undefined`. */
export const gemvCaseOf = (
  id: string,
  cases: readonly GemvCase[] = GEMV_CASES,
): GemvCase | undefined => cases.find((gemvCase) => gemvCase.id === id);

/**
 * Every case of the table, in its order: the ids `GemvSweepOptions.cases` takes. A new array of new
 * objects on every call, so nothing a caller does reaches the table or its fingerprint.
 */
export const gemvSweepCaseEntries = (): GemvCase[] =>
  GEMV_CASES.map((gemvCase) => ({ ...gemvCase, rows: [...gemvCase.rows] }));

/**
 * GEMM geometry **profiles** (static tables of shape × adapter — DECIDED: ADR 0115).
 *
 * The best geometry (which thread handles which output) reverses between GPUs — a geometry 1.4–1.7×
 * faster on Apple M2 becomes 0.72–0.98× on Arc B570 (docs/research/2026-09-27-k70-metal-per-op.md
 * §9). With one common default one of them loses, so the selection is widened from "a pure function
 * of the shape" to "a pure function of shape × profile", and one profile is chosen in a
 * deterministic order from the adapter's (vendor, architecture, description) (a table without
 * `match` is not auto-selected and is used only by injection).
 *
 * MUST: a profile is **plain data** (no side effects, nothing read from outside at run time). The
 * results measured by explicit tuning (sweeps of tools/geometry-sweep (removed in 0.15); the sweep
 * now runs through `@karume/runtime/tune`, ADR 0117) are embedded in the source as generated files
 * (`./geometry-profiles/`). Run-time autotuning (measuring and re-choosing during execution) stays
 * banned (the MUST of the ADR 0022 addendum) — on the same machine and the same shape, the
 * geometry, key and WGSL are the same every time.
 * MUST: the geometry may change only the work assignment; the numeric contract (K reduction order,
 * literal form of the multiply-add) does not move (src/kernels/gemm-geometry.ts /
 * src/kernels/i8a8-geometry.ts). The geometry goes into the pipeline key, so a different profile
 * means a different key, and the device-lifetime pipeline cache cannot mix them up.
 *
 * The scope (the paths that read the profile) is only the row-count buckets of matmul / bmm /
 * linear (or, in a table that carries it, the grid-fill rule that replaces them for those three —
 * {@link GeometryProfile.gemmGrid}), ①QK / ③PV of fused attention f32, the implicit GEMM of conv2d, the i8a8 linear (row-count
 * buckets of its own — ADR 0115 addendum decision 14) / fused attention ①QK / ③PV, and the w8a8
 * implicit GEMM of conv3d (one geometry for every conv3d shape — ADR 0115 addendum 2026-10-26). The
 * `gemmRows` bucket table is shared by the 3 paths matmul / bmm /
 * linear (the same GEMM skeleton — the same meaning of geometry and the same K reduction order; only
 * the B-side fill differs). The sweep has cases for all 3 paths, and the generator adopts only
 * geometries that are faster on all of them (the `profile` command of tools/geometry-sweep (removed
 * in 0.15); now `deriveGeometryProfile` of `@karume/runtime/tune`). States-shaped attention (①ₜ /
 * ③ₜ), the row blocks of decomposed attention and the implicit GEMM of conv1d keep the default
 * selection (outside the sweep — a path that was not measured is not moved by a table).
 * One field is a kernel variant rather than a geometry: the key-axis form of the online attention's
 * quantised Vᵀ (`attentionOnline`), read through the attention route only. Another holds the GEMV
 * choices of linear (`linearGemv` — which sequential variant, how many rows per row block, which
 * shapes take the parallel GEMV and how): GEMV sweep records decide it (`deriveLinearGemvChoices` of
 * `@karume/runtime/tune`), and a table without the field runs with the default profile's
 * ({@link resolveGeometryProfile}).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_ROWS_BUCKETS,
  GEMM_TILE,
  type GemmGeometry,
  gemmTileM,
  gemmTileN,
} from "./gemm-geometry.ts";
import { GEMM_MTILE_SMALL, gemmMTileGeometry, gemmWorkgroupStorageBytes } from "./gemm.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8WorkgroupStorageBytes,
} from "./i8a8-geometry.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "./geometry-profiles/index.ts";
import {
  assertLinearGemvVariant,
  isFloatStorage,
  LINEAR_GEMV_PARALLEL_ROWS,
  type LinearGemvChoices,
  type LinearGemvParallelLanes,
  linearGemvParallelRowsFits,
} from "./linear-gemv.ts";
import type { WeightStorage } from "./weight-storage.ts";

/**
 * One row-count bucket rule. The first rule with `rows <= maxRows` is used ({@link gemmRowsGeometry}).
 * The rules are in strictly ascending `maxRows`, and the last is `Number.POSITIVE_INFINITY` (every
 * row count hits some rule).
 */
export type GemmRowsRule = { readonly maxRows: number; readonly geometry: GemmGeometry };

/**
 * One row-count bucket rule of the i8a8 linear (`i8a8.linear` — ADR 0115 addendum decision 14): the
 * same form and the same rule order as {@link GemmRowsRule} — the first rule with `rows <= maxRows`
 * is used ({@link i8a8LinearGeometry}), strictly ascending `maxRows`, the last
 * `Number.POSITIVE_INFINITY`.
 */
export type I8a8RowsRule = { readonly maxRows: number; readonly geometry: I8a8Geometry };

/**
 * The grid-fill rule of the GEMM of linear / matmul / bmm ({@link GeometryProfile.gemmGrid}). The
 * tile is a step of `ladder` chosen from the shape's dispatch grid ({@link gemmGridStep}):
 *
 * - the first tile (the largest) when `k ≥ keepLargest.minK`, its grid has at least
 *   `keepLargest.minWorkgroups` workgroups and its padding is at most `maxPadding`;
 * - otherwise the first of the other tiles whose grid has at least `minWorkgroups` workgroups and
 *   whose padding is at most `maxPadding`;
 * - otherwise the last tile.
 *
 * A grid's workgroup count is `ceil(n / tileN) · ceil(m / tileM) · batch` and its padding the
 * area the tiles cover over the output's, `(ceil(n / tileN) · tileN · ceil(m / tileM) · tileM) /
 * (m · n)` ({@link gemmGridWorkgroups}, {@link gemmGridPadding}). The two workgroup counts are the
 * device's (a table derives them from its device's grid sweep records); the K floor and the padding
 * bound are facts of the shapes the rule was fitted on, written into the field with the ladder so
 * that the field states the whole rule it runs.
 *
 * MUST: the ladder is in strictly decreasing tile area, and every tile passes the geometry check —
 * a ladder out of order would skip its smaller tiles silently ({@link assertGeometryProfile}).
 * SPEC: docs/spec/geometry-profiles.md#the-grid-fill-rule-of-the-gemm
 */
export type GemmGridRule = {
  /** The tiles, largest first. A tile may carry the skeleton form (not the counter decode). */
  readonly ladder: readonly GemmGeometry[];
  /** When the largest tile is kept: its grid's least workgroup count, and the least K. */
  readonly keepLargest: { readonly minWorkgroups: number; readonly minK: number };
  /** The least workgroup count of a smaller tile's grid. */
  readonly minWorkgroups: number;
  /** The largest padding a tile is taken at (tile area over output area, at least 1). */
  readonly maxPadding: number;
};

/** The shape of one GEMM dispatch: `m × k` by `k × n`, `batch` matrices on the grid's z axis. */
export type GemmShape = {
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly batch: number;
};

/**
 * The op families of the geometry sweep: the unit a table's evidence stands on, and the unit its
 * matching keys are taken per (DECIDED: ADR 0117 addendum 2026-11-12, "per op family"). Each family
 * owns the profile fields its sweep cases decide — `gemm` the row-count buckets `gemmRows` (sweep ops
 * linear / matmul / bmm), `attention` `attention.qk` / `attention.pv`, `conv2d` `conv2d.rows64` /
 * `conv2d.rows32`, `i8a8-linear` `i8a8.linear`, `i8a8-attention` `i8a8.attentionQk` /
 * `i8a8.attentionPv`, `i8a8-conv3d` `i8a8.conv3d`. The maps from sweep op and from profile field to
 * family live in `src/tune/families.ts` (kernels do not import tune). The union is derived from
 * {@link GEOMETRY_PROFILE_FAMILIES}, so a family cannot be added to the type without also entering
 * the list every per-family loop walks.
 */
export type GeometryProfileFamily = (typeof GEOMETRY_PROFILE_FAMILIES)[number];

/**
 * Every family in the fixed order — the order of the fields they decide, the order matching checks
 * them in and the order the generator writes `provenance.families` in. The source of
 * {@link GeometryProfileFamily}.
 *
 * NOTE: `./geometry-profiles/index.ts` must not read it: this module imports that one as a value,
 * and that one builds its frozen list at evaluation, so reading this constant there would depend on
 * the import order.
 */
export const GEOMETRY_PROFILE_FAMILIES = [
  "gemm",
  "attention",
  "conv2d",
  "i8a8-linear",
  "i8a8-attention",
  "i8a8-conv3d",
] as const;

/**
 * The provenance of one op family of a table generated from sweeps (DECIDED: ADR 0117 decision 4 as
 * amended by its addendum 2026-11-12): the sweep records that have a row of the family, and the two
 * matching keys taken over the family's cases only.
 */
export type GeometryProfileFamilyProvenance = {
  /**
   * Paths of the sweep records that have a row of this family (joined with `", "` in the order
   * given to the generator).
   */
  readonly sweep: string;
  /** SHA-256 (hex) of the bytes of the same records. */
  readonly sha256: string;
  /** Dates of the same sweeps. */
  readonly date: string;
  /** Candidate sets of the same sweeps (`quick` / `quick+` / `full`). */
  readonly candidateSet: string;
  /**
   * The browsers that ran the same sweeps (the records' `userAgent` — records from the Deno CLI
   * carry `Deno/<version>`), deduplicated in order of appearance (one element when all are the
   * same). They are not joined because a browser's userAgent itself contains `", "`
   * (`(KHTML, like Gecko)`) and joined values could not be split apart. A reference field not used
   * for matching (`geometryProfileMismatch`); material for an application to compare with the
   * current browser and suggest a re-sweep (DECIDED: ADR 0117 addendum 2026-10-02).
   * Not written when a record without `userAgent` is among the same records (an empty array is not
   * written either).
   */
  readonly userAgent?: readonly string[];
  /**
   * Kernel fingerprint over this family's sweep cases (hash of the kernels the table and the
   * default geometry build on those shapes, the WGSL counted without comments and whitespace — hex).
   */
  readonly kernels: string;
  /**
   * Case-set version of this family (hash of its sweep cases and, for `gemm` and `i8a8-linear`,
   * the row-count bucket bounds — hex).
   */
  readonly caseSet: string;
};

/**
 * One swept adapter as its records name it: the 4 fields of `GPUAdapterInfo` (an empty string stays
 * a value).
 */
export type GeometryProfileAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/**
 * The adapters a generated table was derived from, in one of two forms. A measured table holds
 * `adapter`: the one adapter every record agrees with on all four fields. A shared table holds
 * `adapters`: two or more adapters of one vendor and architecture, each with a description of its
 * own ({@link sharedAdaptersViolation}), in the order their records were given — it is matched by
 * vendor and architecture and predicts for the adapters of that vendor and architecture that no
 * table names by description.
 * SPEC: docs/spec/geometry-profiles.md#shared-tables
 */
export type GeometryProfileAdapters =
  | { readonly adapter: GeometryProfileAdapter; readonly adapters?: undefined }
  | { readonly adapters: readonly GeometryProfileAdapter[]; readonly adapter?: undefined };

/**
 * Why a shared table's `provenance.adapters` is not one (undefined when it is), as the rest of a
 * sentence about the list: fewer than two adapters, an adapter without a description, or two
 * adapters of one description. A shared derivation tells devices apart by their descriptions and
 * needs two of them, so such a list names fewer than two devices. The injection check, the JSON
 * reader and matching all read the list by this rule.
 * SPEC: docs/spec/geometry-profiles.md#shared-tables
 */
export const sharedAdaptersViolation = (
  adapters: readonly GeometryProfileAdapter[],
): string | undefined => {
  if (adapters.length < 2) {
    return `names ${adapters.length} adapter${adapters.length === 1 ? "" : "s"} (a shared table is derived from two or more)`;
  }
  const seen = new Map<string, number>();
  for (const [index, { description }] of adapters.entries()) {
    if (description === "") {
      return `[${index}] has no description (a shared table tells its devices apart by their descriptions)`;
    }
    const twin = seen.get(description);
    if (twin !== undefined) {
      return `[${twin}] and [${index}] are one device (both ${JSON.stringify(description)} — a shared table tells its devices apart by their descriptions and names each once)`;
    }
    seen.set(description, index);
  }
  return undefined;
};

/**
 * A GEMM geometry profile = a static table of tile geometries per op × bucket (DECIDED: ADR 0115).
 * The embedded tables (`./geometry-profiles/`) are chosen by the adapter's (vendor, architecture,
 * description), and users can inject, with `acquireGpu({ geometryProfile })`, a table made from
 * their own sweep or an embedded table (`BUILTIN_GEOMETRY_PROFILES` on the public surface — ADR 0115
 * addendum decision 7).
 */
export type GeometryProfile = {
  /**
   * Name of the profile (`"default"` / `"apple-metal-3"` and so on — matches the generated file
   * name).
   */
  readonly id: string;
  /**
   * Which adapter it applies to (exact string match with `vendor` / `architecture` / `description`
   * of `GPUAdapterInfo`). All fields undefined (`{}`) is the form of the default profile
   * ({@link DEFAULT_GEOMETRY_PROFILE}) only.
   *
   * Omitted = not auto-selected (injection only = opt-in — ADR 0115 addendum decision 7). A table
   * that must not be applied silently to another machine reporting the same vendor / architecture
   * (in Chrome, Apple M2 and M5 both report `apple` / `metal-3`) is embedded without `match`, and an
   * application looks it up by id and injects it.
   */
  readonly match?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly description?: string;
  };
  /**
   * linear / matmul / bmm (row-count buckets). Not read by those ops in a table that carries
   * {@link gemmGrid}.
   */
  readonly gemmRows: readonly GemmRowsRule[];
  /**
   * ①QK / ③PV of fused attention f32. S is materialised between the two stages, so each stage may
   * have its own geometry.
   */
  readonly attention: { readonly qk: GemmGeometry; readonly pv: GemmGeometry };
  /**
   * The implicit GEMM of conv2d. A geometry per m-tile **class** (64 rows / 32 rows as returned by
   * `conv2dIgemmMTile`); the class is decided by that predicate as before. The canonical source of
   * the actual tile edge is the geometry, so a geometry with tileM = 64 may be assigned to the
   * 32-row class (dispatch is derived from the geometry).
   */
  readonly conv2d: { readonly rows64: GemmGeometry; readonly rows32: GemmGeometry };
  readonly i8a8: {
    /**
     * The i8a8 linear (w8a8 / w4a8 — `linearCompute: "a8"`) by buckets of the row count M (ADR 0115
     * addendum decision 14). The default holds one rule; a generated table holds one rule per bucket
     * of the generator's bounds.
     */
    readonly linear: readonly I8a8RowsRule[];
    readonly attentionQk: I8a8Geometry;
    readonly attentionPv: I8a8Geometry;
    /**
     * The w8a8 implicit GEMM of conv3d (`convCompute: "a8"` — ADR 0133). One geometry for every
     * conv3d shape (no shape buckets — ADR 0115 addendum 2026-10-26).
     */
    readonly conv3d: I8a8Geometry;
  };
  /**
   * The key-axis form of the online int8 attention's quantised Vᵀ (perf-ledger K-106 — ADR 0132
   * addendum 2026-11-12), a kernel-variant choice rather than a geometry. Absent (every table but one
   * measured to gain from it) = the form `attentionOnlineKeyWords` gives N (`whole` at `N % 4 == 0`,
   * `padded` otherwise), so every key, WGSL and estimate of a table without the field is unchanged.
   * `{ keyWords: "tiled" }` = the producer writes Vᵀ in the kernel's key order at every N. Both forms
   * are bit-identical by measured behaviour (`checkAttentionOnlineTiled` of packages/kernel-checks).
   *
   * MUST: read only through the attention route (`attentionDeviceFacts` → `attentionRoute`), the one
   * decision point the producer, the kernel, its uniform and the size of the quantised Vᵀ share.
   */
  readonly attentionOnline?: { readonly keyWords: "tiled" };
  /**
   * The GEMV choices of linear (1 ≤ M ≤ 64 over i2 / i4 / i8 weights; over f16 / bf16 / f32
   * weights M = 1, and M ≤ 8 on the parallel GEMV — src/kernels/linear-gemv.ts): the sequential variant and its per-shape exceptions, the
   * row-block thread target and the shape rows of the parallel GEMV with their lanes and packed /
   * staged flags. Kernel-variant choices rather than geometries, like {@link attentionOnline}. GEMV
   * sweep records decide them (`deriveLinearGemvChoices` of `@karume/runtime/tune`).
   *
   * Absent = the default profile's ({@link DEFAULT_GEOMETRY_PROFILE}), resolved in one place, where
   * a Session selects its profile ({@link resolveGeometryProfile}): every reader sees a
   * {@link ResolvedGeometryProfile}, so a table without the field keeps every key, WGSL and plan it
   * had.
   * SPEC: docs/spec/geometry-profiles.md#the-gemv-choices-of-linear
   */
  readonly linearGemv?: LinearGemvChoices;
  /**
   * The grid-fill rule of linear / matmul / bmm ({@link GemmGridRule}): the tile is chosen from the
   * workgroup count of the dispatch grid instead of the row count alone. Present, it replaces
   * {@link gemmRows} for those three ops — a walk down its ladder ({@link gemmShapeGeometry}) —, and
   * `gemmRows` is not read by them. Absent (the default profile and every table without the field),
   * `gemmRows` applies as it always has, so every key, WGSL and dispatch is unchanged.
   * Grid sweep records decide it (`deriveGemmGrid` of `@karume/runtime/tune`).
   * SPEC: docs/spec/geometry-profiles.md#the-grid-fill-rule-of-the-gemm
   */
  readonly gemmGrid?: GemmGridRule;
  /**
   * Provenance and matching material held only by profiles generated from a sweep (DECIDED: ADR 0117
   * decision 4, per op family since its addendum 2026-11-12). Only the generator
   * (`deriveGeometryProfile` of `@karume/runtime/tune`) writes it, and `geometryProfileMismatch` of
   * `./tune` does the matching. The runtime (`assertGeometryProfile`, the injection entry point of
   * `acquireGpu`) checks only the shape (the field types) and does not use it for matching.
   */
  readonly provenance?: GeometryProfileAdapters & {
    /**
     * Per op family ({@link GeometryProfileFamily}), the records and matching keys of that family.
     * A family that no record has a case of has no entry: its fields are the default (matching
     * checks that they are).
     */
    readonly families: {
      readonly [Family in GeometryProfileFamily]?: GeometryProfileFamilyProvenance;
    };
    /**
     * The attention-bench records the derivation read for {@link GeometryProfile.attentionOnline}
     * (written whenever records were given, whether or not they adopted the field — ADR 0117
     * addendum 2026-11-12): their paths, SHA-256 and dates joined with `", "` in the order given,
     * the id of the bench's shape set they covered (the `attentionBenchShapeSetId` of the runtime
     * that derived the table — `geometryProfileMismatch` compares it) and, exactly when the table
     * carries `attentionOnline`, the fingerprint of the online attention's kernels the field chooses
     * between (`kernels` — compared by matching as well).
     */
    readonly attentionBench?: {
      readonly records: string;
      readonly sha256: string;
      readonly date: string;
      readonly shapes: string;
      readonly kernels?: string;
    };
  };
};

/**
 * The GEMV choices of the default profile ({@link GeometryProfile.linearGemv}): the values measured
 * on the reference device (RTX 3080 Ti). This table is their one place — the kernel
 * (src/kernels/linear-gemv.ts) has no default of its own and reads them as an argument — and every
 * table without the field runs with them ({@link resolveGeometryProfile}).
 */
const DEFAULT_LINEAR_GEMV: LinearGemvChoices = {
  sequential: {
    /**
     * The default variant. Census-weighted best in **the sweep of the 12 real shapes + 4 remainder
     * shapes of gemma4 E2B decode on an RTX 3080 Ti** (`c32 u4`, ×8.45 over the default — ADR 0082 /
     * docs/research/2026-08-30-gemma4-decode-wallclock.md §7).
     * MUST: a change of the default goes together with re-measuring the bit-identity check
     * (tests/gpu_linear_gemv_test.ts).
     */
    variant: { cols: 32, unroll: 4 },
    // The rows are bf16 linears of regular Gemma 4 E2B / E4B: a bf16 word holds 8 elements
    // where an int4 word holds 32, so the general read-ahead of 4 words keeps a quarter of the bytes
    // in flight. They are the exceptions that the derivation adopts from GEMV sweep records of the
    // RTX 3080 Ti (docs/spec/geometry-profiles.md, "Deriving the GEMV choices": every word the general
    // variant's, and at least ×1.10 over it at M = 1 in every steady group, from two records or
    // more — ×1.17 to ×1.89 per dispatch in three to five records each, perf-ledger K-112) and
    // that no measured device is slower with: the RTX 5070 Ti in Chrome reads them ×1.015 to ×1.495
    // over the general variant (2026-11-18). The Apple M2's records, taken since, time the general
    // variant against 12 of them and read it ×1.01 to ×1.26 faster on 11; `apple-metal-3` runs
    // another variant on 17 of the 22 (docs/limitations.md, the linear GEMV family).
    // A row a measured device is slower with is not added to the default; it lives in the table of
    // the device that gains from it (`nvidia-ampere` carries the int4 output heads of the q4_0 QAT
    // E2B / E4B models and bf16 10240 × 2560, which the RTX 5070 Ti reads ×0.89 to ×0.95, and the
    // bf16 output head 262144 × 2560; `nvidia-blackwell` carries 42 int4 and int8 rows of its own,
    // and another variant on two of the bf16 rows, at least ×1.12 and ×1.20 faster in its records;
    // `apple-metal-3` carries 64 int4 and int8 rows of its own).
    // A row names a storage and a shape (no group length) and applies at M = 1: a row block takes
    // the general variant.
    // MUST: a row added or changed, here or in a registered table, goes together with
    // tests/gpu_linear_gemv_variant_test.ts, which compares every row's words with the general
    // variant's on the row's own shape.
    shapes: [
      { storage: "bf16", n: 8960, k: 1536, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 2048, k: 1536, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 1536, k: 2048, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 6144, k: 1536, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
      { storage: "bf16", n: 4096, k: 1536, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 512, k: 1536, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 12288, k: 1536, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 1536, k: 12288, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 10752, k: 2560, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 2048, k: 2560, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 512, k: 2560, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 2560, k: 2048, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 2560, k: 10240, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 4096, k: 2560, variant: { cols: 16, unroll: 8 } },
      { storage: "bf16", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
      { storage: "bf16", n: 2560, k: 4096, variant: { cols: 16, unroll: 8 } },
    ],
  },
  /**
   * In the RTX 3080 Ti sweep, shapes below this value were faster buying parallelism with y tiles,
   * and shapes above it were faster reducing weight re-reads with `rows` (the mechanism in the
   * module doc of src/kernels/linear-gemv.ts).
   * NOTE: **a value baked from the saturation point of the reference device (RTX 3080 Ti, 80 SMs)**,
   * not a portable optimum. On GPUs whose saturation point is an order of magnitude smaller
   * (integrated GPUs, Apple M series) it buys too many y tiles and the weight re-reads exceed the
   * optimum. For other devices the value is replaced with `SessionOptions.linearGemvRowsThreadTarget`
   * (a **static** knob fixed at Session creation = the limit that flows into the 5th argument of
   * `linearGemvRowsForShape`; the selection stays a pure function). Nothing chooses it
   * automatically by looking at the device (the ban on run-time autotuning (ADR 0022) —
   * docs/limitations.md).
   * MUST: a change goes together with a re-measured sweep (the check in tests/gpu_linear_gemv_test.ts
   * does not pin the value — it only checks that the selection is a pure function of (storage, m, n,
   * target)).
   */
  rowsThreadTarget: 16384,
  // DECIDED: only measured shapes are eligible for the parallel GEMV (linear at the level `reorder`).
  // Not an automatic selection by GPU name.
  // docs/decisions/0098-linear-gemv-parallel.md
  //
  // A row's `rowBlock` (the row-blocked kernel at M = 2..8) is set from two sweep records of the
  // RTX 3080 Ti at M = 4 and 8 (2026-11-18, perf-ledger K-108). The rule of a row block
  // (docs/spec/geometry-profiles.md, "Deriving the GEMV choices"): against the row without one,
  // slower in no group and at least ×1.10 faster in every group of M = 4 or of M = 8, the baseline
  // steady in both records. Every row set here is ×1.10 faster in every group of both row counts.
  // A row of the Gemma 4 q4_0 QAT shapes without one was slower in a group, had a record whose
  // baseline was unsteady, or — E2B's per_layer_projection, ×1.05 at M = 4 and ×1.28 at M = 8 with
  // 2 rows — meets the rule only since it stopped asking ×1.10 at both row counts, and waits for the
  // next sweep. The mobile QAT rows and the i8 heads were not measured.
  parallelShapes: [
    { storage: "i4", n: 8960, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
    { storage: "i4", n: 2048, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    { storage: "i4", n: 256, k: 1536, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
    {
      storage: "i4",
      n: 1536,
      k: 2048,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    { storage: "i4", n: 6144, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
    {
      storage: "i4",
      n: 1536,
      k: 6144,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 2,
    },
    { storage: "i4", n: 1536, k: 256, group: 32, lanes: 4, packedActivations: false },
    { storage: "i4", n: 4096, k: 1536, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
    { storage: "i4", n: 512, k: 1536, group: 32, lanes: 32, packedActivations: false },
    {
      storage: "i4",
      n: 1536,
      k: 4096,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    {
      storage: "i4",
      n: 12288,
      k: 1536,
      group: 32,
      lanes: 4,
      packedActivations: false,
      rowBlock: 4,
    },
    {
      storage: "i4",
      n: 1536,
      k: 12288,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 2,
    },
    { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false },
    { storage: "i4", n: 2048, k: 1536, group: 512, lanes: 4, packedActivations: false },
    { storage: "i4", n: 256, k: 1536, group: 512, lanes: 32, packedActivations: false },
    { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
    { storage: "i4", n: 6144, k: 1536, group: 512, lanes: 4, packedActivations: false },
    { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
    { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
    { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
    { storage: "i4", n: 4096, k: 1536, group: 512, lanes: 4, packedActivations: false },
    { storage: "i4", n: 512, k: 1536, group: 512, lanes: 32, packedActivations: false },
    { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
    { storage: "i2", n: 12288, k: 1536, lanes: 2, packedActivations: false },
    { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
    // ---- E4B (added 2026-09-26) — **the lanes and packedActivations of the rows below are not
    // measured**: they are guesses copied from the E2B row with the same role, to be revisited with
    // E4B's flag-bench (per-key census and GPU time) (ADR 0098 addendum 2026-09-26). A `rowBlock`
    // on one of them is measured at the row's own shape, against the same row without one (the
    // comment above `parallelShapes`). The shapes are the real ones, read
    // for regular E4B from the upstream config (hidden 2560, intermediate 10240, kv 2, 42 layers, PLE
    // 256) and for QAT E4B from the container descriptor of the distribution mirror.
    // E4B regular i4 g32: per_layer_model_projection (42×256) — lanes from E2B 8960×1536 (unmeasured)
    {
      storage: "i4",
      n: 10752,
      k: 2560,
      group: 32,
      lanes: 4,
      packedActivations: false,
      rowBlock: 4,
    },
    // E4B regular i4 g32: q (sliding) — lanes from E2B 2048×1536 (unmeasured)
    { storage: "i4", n: 2048, k: 2560, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    // E4B regular i4 g32: k / v (sliding) — lanes from E2B 256×1536 (unmeasured)
    { storage: "i4", n: 512, k: 2560, group: 32, lanes: 32, packedActivations: false },
    // E4B regular i4 g32: per_layer_input_gate — lanes from E2B 256×1536 (unmeasured)
    { storage: "i4", n: 256, k: 2560, group: 32, lanes: 32, packedActivations: false },
    // E4B regular i4 g32: o (sliding) — lanes from E2B 1536×2048 (unmeasured)
    {
      storage: "i4",
      n: 2560,
      k: 2048,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    // E4B regular i4 g32: gate / up — lanes from E2B 6144×1536 (unmeasured)
    {
      storage: "i4",
      n: 10240,
      k: 2560,
      group: 32,
      lanes: 4,
      packedActivations: false,
      rowBlock: 4,
    },
    // E4B regular i4 g32: down — lanes from E2B 1536×6144 (unmeasured)
    {
      storage: "i4",
      n: 2560,
      k: 10240,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 2,
    },
    // E4B regular i4 g32: per_layer_projection — lanes from E2B 1536×256 (unmeasured)
    { storage: "i4", n: 2560, k: 256, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    // E4B regular i4 g32: q (full) — lanes from E2B 4096×1536 (unmeasured)
    { storage: "i4", n: 4096, k: 2560, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    // E4B regular i4 g32: k / v (full) — lanes from E2B 512×1536 (unmeasured)
    {
      storage: "i4",
      n: 1024,
      k: 2560,
      group: 32,
      lanes: 32,
      packedActivations: false,
      rowBlock: 2,
    },
    // E4B regular i4 g32: o (full) — lanes from E2B 1536×4096 (unmeasured)
    {
      storage: "i4",
      n: 2560,
      k: 4096,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 2,
    },
    // E4B regular i8: lm_head (tied embedding) — lanes from E2B 262144×1536 (unmeasured)
    { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false },
    // E4B QAT i4 g512: q (sliding) — lanes from E2B QAT 2048×1536 g512 (unmeasured)
    { storage: "i4", n: 2048, k: 2560, group: 512, lanes: 4, packedActivations: false },
    // E4B QAT i4 g512: k / v (sliding) — lanes from E2B QAT 256×1536 g512 (unmeasured)
    { storage: "i4", n: 512, k: 2560, group: 512, lanes: 32, packedActivations: false },
    // E4B QAT i4 g2048: o (sliding) — lanes / packed from E2B QAT 1536×2048 g2048 (unmeasured)
    { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
    // E4B QAT i4 g512: gate / up — lanes from E2B QAT 6144×1536 g512 (unmeasured)
    { storage: "i4", n: 10240, k: 2560, group: 512, lanes: 4, packedActivations: false },
    // E4B QAT i4 g2048: down — lanes / packed from E2B QAT 1536×6144 g2048 (unmeasured)
    { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
    // E4B QAT i8: per_layer_input_gate — lanes from E2B QAT 256×1536 i8 (unmeasured)
    { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
    // E4B QAT i8: per_layer_projection — lanes from E2B QAT 1536×256 i8 (unmeasured)
    { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
    // E4B QAT i4 g512: q (full) — lanes from E2B QAT 4096×1536 g512 (unmeasured)
    { storage: "i4", n: 4096, k: 2560, group: 512, lanes: 4, packedActivations: false },
    // E4B QAT i4 g512: k / v (full) — lanes from E2B QAT 512×1536 g512 (unmeasured)
    { storage: "i4", n: 1024, k: 2560, group: 512, lanes: 32, packedActivations: false },
    // E4B QAT i4 g4096: o (full) — lanes / packed from E2B QAT 1536×4096 g4096 (unmeasured)
    { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
    // ---- q4_0 QAT E2B / E4B MTP drafters (added 2026-11-18) — the rows the GEMV derivation adopts
    // from GEMV sweep records of the RTX 3080 Ti: at least ×1.10 over the sequential GEMV at M = 1 in
    // two steady records and not slower at M = 4 and 8 (docs/research/2026-11-18-gemma4-gemv-census.md
    // §4). The drafters' other shapes keep the sequential GEMV: they gain less, lose at M = 4 or 8, or
    // have fewer than two steady records.
    // E2B drafter: pre_projection — parallel 32 (×5.9 at M = 1 in three records)
    { storage: "i4", n: 256, k: 3072, group: 32, lanes: 32, packedActivations: false },
    // E2B / E4B drafter: o (sliding) — parallel 32 (×3.3 in three records)
    { storage: "i4", n: 256, k: 1024, group: 32, lanes: 32, packedActivations: false },
    // ---- 12B q4_0 QAT (added 2026-11-06) — lanes measured as whole-model decode on the RTX 3080 Ti
    // (ADR 0098 addendum 2026-11-06): 4 beats 16 and 32 on the rows that widen, staged 32 ties 16 and
    // beats plain 32 on the rows that narrow, and the output head takes 16 (32 does not build).
    // gate / up
    { storage: "i4", n: 15360, k: 3840, group: 32, lanes: 4, packedActivations: false },
    // q (full), q (sliding), k / v (sliding)
    { storage: "i4", n: 8192, k: 3840, group: 32, lanes: 4, packedActivations: false },
    { storage: "i4", n: 4096, k: 3840, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    { storage: "i4", n: 2048, k: 3840, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    // k (full)
    { storage: "i4", n: 512, k: 3840, group: 32, lanes: 32, packedActivations: false, rowBlock: 2 },
    // o (sliding), o (full), down
    {
      storage: "i4",
      n: 3840,
      k: 4096,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    {
      storage: "i4",
      n: 3840,
      k: 8192,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 2,
    },
    {
      storage: "i4",
      n: 3840,
      k: 15360,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    // lm_head (tied embedding)
    { storage: "i4", n: 262144, k: 3840, group: 32, lanes: 16, packedActivations: false },
    // ---- 12B q4_0 QAT MTP drafter (added 2026-11-18) — derived as the E2B / E4B drafter rows above.
    // pre_projection — staged 32 (×5.1–5.3 in three records)
    {
      storage: "i4",
      n: 1024,
      k: 7680,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    // o (sliding) — staged 32 (×4.8 and ×10.3 in two records)
    // NOTE: a row is matched by shape, and this one is not the drafter's alone: the 22 FFN-down
    // linears of the SBV2 text encoder's int4 graph are int4 g32 1024×4096 too. The sbv2 pipeline
    // does not take `numerics`, so they reach this row only in a Session created directly with
    // linear at `reorder` and at most 8 rows; the row was not measured in that model.
    {
      storage: "i4",
      n: 1024,
      k: 4096,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    // down, o (full) — staged 32 (×5.2–5.7 in two records)
    {
      storage: "i4",
      n: 1024,
      k: 8192,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    // ---- 26B A4B q4_0 QAT (added 2026-11-10) — the dense linears of the mixture of experts (the
    // routed experts run on `expert_linear`, which has no parallel form). Lanes measured per kernel
    // as whole-model decode on the RTX 3080 Ti (ADR 0098 addendum 2026-11-10): on the rows that
    // widen 4, 16 and 32 tie and 2 and 8 lose, so they take 12B's 4; the router and the full k take
    // 32; the rows that narrow take staged 32 (staged 16 ties, unstaged 32 and plain 4 lose); and
    // the output head takes 4, which beats 8 and 16 here (32 does not build).
    // q (full), q (sliding), k / v (sliding), gate / up
    { storage: "i4", n: 8192, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 4 },
    { storage: "i4", n: 4096, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    { storage: "i4", n: 2048, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    { storage: "i4", n: 2112, k: 2816, group: 32, lanes: 4, packedActivations: false, rowBlock: 2 },
    // k (full), router
    { storage: "i4", n: 1024, k: 2816, group: 32, lanes: 32, packedActivations: false },
    { storage: "i4", n: 128, k: 2816, group: 32, lanes: 32, packedActivations: false },
    // o (sliding), o (full), down
    {
      storage: "i4",
      n: 2816,
      k: 4096,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    {
      storage: "i4",
      n: 2816,
      k: 8192,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    {
      storage: "i4",
      n: 2816,
      k: 2112,
      group: 32,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
      rowBlock: 2,
    },
    // lm_head (tied embedding)
    { storage: "i4", n: 262144, k: 2816, group: 32, lanes: 4, packedActivations: false },
    // ---- regular Gemma 4 E2B / E4B, bf16 — the bf16 linears that no measured device is slower with
    // on the parallel GEMV (docs/spec/geometry-profiles.md, "The GEMV choices of linear"). The
    // evidence is GEMV sweep records of the RTX 3080 Ti (Node, Vulkan; four records of 2026-11-19),
    // the RTX 5070 Ti (Chrome, D3D12; two of 2026-11-18) and an Apple M2 (Chrome, Metal; two of
    // 2026-11-19), each with the full candidate set. Read conservatively per device (its fastest
    // baseline over its records against the row's slowest time), a row's kernel is slower than the
    // default's choice — the sequential GEMV at 1 row, the GEMM at 4 and 8 — on none of the three;
    // its smallest gain over the devices and row counts is ×1.09 to ×3.37. The other bf16 linears
    // keep the default's choice: 8960 × 1536, 10752 × 2560 and 10240 × 2560 are slower than the GEMM
    // at 8 rows on the RTX 3080 Ti (×0.69 to ×0.89), and 1536 × 2048, 12288 × 1536, 2560 × 256 and
    // the two output heads have no candidate measured at all three row counts on all three devices.
    // A float-stored row carries no row block: the row-blocked kernel is built for integer storage.
    // E2B
    { storage: "bf16", n: 2048, k: 1536, lanes: 16, packedActivations: false },
    { storage: "bf16", n: 256, k: 1536, lanes: 32, packedActivations: false },
    { storage: "bf16", n: 6144, k: 1536, lanes: 8, packedActivations: false },
    { storage: "bf16", n: 1536, k: 6144, lanes: 16, packedActivations: false },
    { storage: "bf16", n: 1536, k: 256, lanes: 8, packedActivations: false },
    { storage: "bf16", n: 4096, k: 1536, lanes: 8, packedActivations: false },
    { storage: "bf16", n: 512, k: 1536, lanes: 16, packedActivations: false },
    {
      storage: "bf16",
      n: 1536,
      k: 4096,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    { storage: "bf16", n: 1536, k: 12288, lanes: 16, packedActivations: false },
    // E4B
    {
      storage: "bf16",
      n: 2048,
      k: 2560,
      lanes: 32,
      packedActivations: false,
      stagedActivations: true,
    },
    { storage: "bf16", n: 512, k: 2560, lanes: 32, packedActivations: false },
    { storage: "bf16", n: 256, k: 2560, lanes: 32, packedActivations: false },
    { storage: "bf16", n: 2560, k: 2048, lanes: 8, packedActivations: false },
    { storage: "bf16", n: 2560, k: 10240, lanes: 16, packedActivations: false },
    { storage: "bf16", n: 4096, k: 2560, lanes: 8, packedActivations: false },
    { storage: "bf16", n: 1024, k: 2560, lanes: 16, packedActivations: false },
    { storage: "bf16", n: 2560, k: 4096, lanes: 16, packedActivations: false },
  ],
};

/**
 * The default profile (the table used by an adapter that no generated table matches).
 *
 * MUST: the geometries are built **by referencing** the existing selection functions (not copied —
 * copying would put the default in 2 places, and when only one is fixed, the Session path and the
 * shape-pure-function path would choose different geometries). The GEMV choices have no such
 * function: {@link DEFAULT_LINEAR_GEMV} is their one place. On a machine where this table is chosen,
 * key, WGSL and dispatch do not differ by a single byte from before profiles were introduced (the
 * codegen snapshots and tests/geometry_profile_test.ts are the detectors).
 */
export const DEFAULT_GEOMETRY_PROFILE: ResolvedGeometryProfile = {
  id: "default",
  match: {},
  gemmRows: GEMM_ROWS_BUCKETS,
  attention: { qk: defaultGemmGeometry(), pv: defaultGemmGeometry() },
  conv2d: {
    rows64: gemmMTileGeometry(GEMM_TILE),
    rows32: gemmMTileGeometry(GEMM_MTILE_SMALL),
  },
  i8a8: {
    linear: [{ maxRows: Number.POSITIVE_INFINITY, geometry: defaultI8a8Geometry("linear") }],
    attentionQk: defaultI8a8Geometry("attention_qk"),
    attentionPv: defaultI8a8Geometry("attention_pv"),
    conv3d: defaultI8a8Geometry("conv3d"),
  },
  linearGemv: DEFAULT_LINEAR_GEMV,
};

/**
 * A profile as every reader sees it: the optional fields that fall back to the default profile's
 * values are resolved ({@link GeometryProfile.linearGemv}).
 */
export type ResolvedGeometryProfile = GeometryProfile & { readonly linearGemv: LinearGemvChoices };

/**
 * Resolves a selected or injected table to the profile a Session runs with: a table without
 * `linearGemv` takes the default profile's GEMV choices; every other field is the table's own.
 *
 * MUST: called in one place, where a Session selects its profile (`sessionGeometryProfile` in
 * src/runtime/geometry-selection.ts). The selection and the embedded list return the tables as they
 * are, so a table read back or compared by identity is the table that was registered.
 */
export const resolveGeometryProfile = (profile: GeometryProfile): ResolvedGeometryProfile => ({
  ...profile,
  linearGemv: profile.linearGemv ?? DEFAULT_LINEAR_GEMV,
});

/**
 * Guard on the row-count bucket rules of one field (`field` — `gemmRows` or `i8a8.linear`, the one
 * rule-order check both share; `assertGeometry` checks each rule's geometry). **Rules not in
 * ascending order leave later rules dead, and rules whose last entry is not Infinity leave no rule
 * for large M** — both silently turn the selection into something else, so fail loudly.
 */
const assertRowsRules = <Geometry>(
  rules: readonly { readonly maxRows: number; readonly geometry: Geometry }[],
  field: string,
  where: string,
  assertGeometry: (geometry: Geometry, where: string) => void,
): void => {
  // A table read with JSON.parse and a cast can hold the field in another shape (a single
  // geometry); fail with the guard's error rather than a TypeError
  if (!Array.isArray(rules)) throw new CodegenError(`${where}: ${field} is not an array`);
  if (rules.length === 0) {
    throw new CodegenError(`${where}: ${field} is empty (no geometry matches any row count)`);
  }
  rules.forEach((rule, index) => {
    const last = index === rules.length - 1;
    if (last) {
      if (rule.maxRows !== Number.POSITIVE_INFINITY) {
        throw new CodegenError(
          `${where}: the last maxRows of ${field} must be Infinity (${rule.maxRows} — no rule covers a larger M)`,
        );
      }
    } else {
      assertU32Params(`${where}: ${field}[${index}]`, { maxRows: rule.maxRows });
      const next = rules[index + 1].maxRows;
      if (!(rule.maxRows < next)) {
        throw new CodegenError(
          `${where}: maxRows of ${field} is not strictly increasing ([${index}] ${rule.maxRows} → [${
            index + 1
          }] ${next} — no row count is left for the later rule)`,
        );
      }
    }
    assertGeometry(rule.geometry, `${where}: ${field}[${index}]`);
  });
};

const assertGemmRowsRules = (rules: readonly GemmRowsRule[], where: string): void =>
  assertRowsRules(rules, "gemmRows", where, assertGemmGeometry);

const assertI8a8LinearRules = (rules: readonly I8a8RowsRule[], where: string): void =>
  assertRowsRules(rules, "i8a8.linear", where, assertI8a8Geometry);

/**
 * The rule of a checked list that covers `rows` (the first with `rows <= maxRows` — a checked list
 * ends in Infinity, so a u32 row count always hits one).
 */
const ruleFor = <Rule extends { readonly maxRows: number }>(
  rules: readonly Rule[],
  rows: number,
): Rule => rules.find((rule) => rows <= rule.maxRows) ?? rules[rules.length - 1];

/**
 * Guard that a value is an object whose fields can be read (rejects arrays, null and primitives
 * with the field path).
 */
const recordOf = (value: unknown, where: string, path: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new CodegenError(`${where}: ${path} is not an object`);
  }
  return { ...value };
};

const stringOf = (value: unknown, where: string, path: string): string => {
  if (typeof value === "string") return value;
  throw new CodegenError(
    `${where}: ${path} is not a string (${value === undefined ? "missing" : typeof value})`,
  );
};

const optionalStringOf = (value: unknown, where: string, path: string): string | undefined =>
  value === undefined ? undefined : stringOf(value, where, path);

/**
 * All elements of an array field (rejects holes with the field path). `map` / `forEach` skip the
 * holes of a sparse array, so a hole would pass the guard and become a TypeError outside it (in the
 * selection's `find` or an index read) — a table built by a JS caller (`new Array(n)` and so on) is
 * rejected as a missing field with the guard's {@link CodegenError}.
 */
const denseEntriesOf = (
  values: readonly unknown[],
  where: string,
  path: string,
): readonly unknown[] => {
  const entries: unknown[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (!(index in values)) {
      throw new CodegenError(`${where}: ${path}[${index}] is missing (a hole in a sparse array)`);
    }
    entries.push(values[index]);
  }
  return entries;
};

/**
 * Guard that a geometry field is a number. The range (positive integer) and the divisibility
 * conditions belong to {@link assertGemmGeometry} / {@link assertI8a8Geometry} (a non-number is
 * rejected with the same wording).
 */
const numberOf = (value: unknown, where: string, path: string, name: string): number => {
  if (typeof value === "number") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry ${name} must be a positive integer (${String(value)})`,
  );
};

/**
 * Guard on the optional skeleton form of an f32 GEMM geometry (perf-ledger K-88 — the same
 * wording as {@link assertGemmGeometry}, which checks it again at generation).
 */
const formOf = (value: unknown, where: string, path: string): NonNullable<GemmGeometry["form"]> => {
  if (value === "tb-un" || value === "tas-tb-un") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry form must be "tb-un", "tas-tb-un" or absent (${JSON.stringify(value)})`,
  );
};

/** Guard on the optional im2col decode of a conv2d geometry (perf-ledger K-88). */
const decodeOf = (
  value: unknown,
  where: string,
  path: string,
): NonNullable<GemmGeometry["decode"]> => {
  if (value === "counter") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry decode must be "counter" or absent (${JSON.stringify(value)})`,
  );
};

const gemmGeometryOf = (value: unknown, where: string, path: string): GemmGeometry => {
  const { regM, regN, wgX, wgY, form, decode } = recordOf(value, where, path);
  // Absent stays absent (a table without the fields keeps its keys and WGSL byte for byte)
  return {
    regM: numberOf(regM, where, path, "regM"),
    regN: numberOf(regN, where, path, "regN"),
    wgX: numberOf(wgX, where, path, "wgX"),
    wgY: numberOf(wgY, where, path, "wgY"),
    ...(form === undefined ? {} : { form: formOf(form, where, path) }),
    ...(decode === undefined ? {} : { decode: decodeOf(decode, where, path) }),
  };
};

/**
 * Which optional fields of an f32 GEMM geometry a profile field may carry (perf-ledger K-88): the
 * skeleton form where a kernel takes it (the row-count buckets of linear / matmul / bmm, conv2d),
 * the counter decode in conv2d only.
 *
 * MUST: refuse them elsewhere. The fused attention kernels keep the original form whatever the
 * geometry carries, and every op but conv2d refuses the decode at generation — a table carrying
 * them there would either be silently not what it says or fail only when that op is first built.
 */
const assertGemmFieldsIn = (
  geometry: GemmGeometry,
  where: string,
  path: string,
  allowed: { readonly form: boolean; readonly decode: boolean },
): void => {
  if (geometry.form !== undefined && !allowed.form) {
    throw new CodegenError(
      `${where}: ${path}: the skeleton form is taken only by linear / matmul / bmm and conv2d (form ${JSON.stringify(
        geometry.form,
      )} — the fused attention kernels keep the original form)`,
    );
  }
  if (geometry.decode !== undefined && !allowed.decode) {
    throw new CodegenError(
      `${where}: ${path}: the counter decode is conv2d's alone (decode ${JSON.stringify(
        geometry.decode,
      )})`,
    );
  }
};

/**
 * Guard on the optional skeleton form of an i8a8 geometry (perf-ledger K-85): `"vec4"` is the only
 * value (the same wording as {@link assertI8a8Geometry}, which checks it again at generation).
 */
const skeletonOf = (
  value: unknown,
  where: string,
  path: string,
): NonNullable<I8a8Geometry["skeleton"]> => {
  if (value === "vec4") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry skeleton must be "vec4" or absent (${JSON.stringify(value)})`,
  );
};

const i8a8GeometryOf = (value: unknown, where: string, path: string): I8a8Geometry => {
  const { tileK, skeleton } = recordOf(value, where, path);
  const geometry = {
    ...gemmGeometryOf(value, where, path),
    tileK: numberOf(tileK, where, path, "tileK"),
  };
  // Absent stays absent (a table without the field keeps its keys and WGSL byte for byte)
  return skeleton === undefined
    ? geometry
    : { ...geometry, skeleton: skeletonOf(skeleton, where, path) };
};

/**
 * The row-count rules of one field (`gemmRows` / `i8a8.linear`) read from an unknown value: an
 * array without holes of `{ maxRows: number, geometry }` (`geometryOf` reads each geometry at its
 * path). The order and the trailing Infinity are {@link assertRowsRules}' job.
 */
const rulesOf = <Geometry>(
  value: unknown,
  where: string,
  field: string,
  geometryOf: (value: unknown, path: string) => Geometry,
): { readonly maxRows: number; readonly geometry: Geometry }[] => {
  if (!Array.isArray(value)) throw new CodegenError(`${where}: ${field} is not an array`);
  return denseEntriesOf(value, where, field).map((entry, index) => {
    const rule = recordOf(entry, where, `${field}[${index}]`);
    if (typeof rule.maxRows !== "number") {
      throw new CodegenError(`${where}: maxRows of ${field}[${index}] is not a number`);
    }
    return {
      maxRows: rule.maxRows,
      geometry: geometryOf(rule.geometry, `${field}[${index}].geometry`),
    };
  });
};

/** Guard that a count of the grid-fill rule is a positive u32 integer. */
const assertGridCount = (value: unknown, where: string, path: string): number => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new CodegenError(
      `${where}: gemmGrid.${path} must be a positive integer (${String(value)})`,
    );
  }
  return value;
};

/**
 * The semantic check of a grid-fill rule ({@link GemmGridRule}): two tiles or more in strictly
 * decreasing tile area, each passing {@link assertGemmGeometry} and carrying no counter decode
 * (conv2d's alone), positive integer counts, and a padding bound that is a finite number of at least
 * 1 (below 1 no tile fits). Fails loudly: a ladder out of order leaves its later tiles unreachable
 * by the walk, and a missing count would turn every comparison false.
 */
const assertGemmGridRule = (rule: GemmGridRule, where: string): void => {
  const { ladder } = rule;
  if (!Array.isArray(ladder)) throw new CodegenError(`${where}: gemmGrid.ladder is not an array`);
  if (ladder.length < 2) {
    throw new CodegenError(
      `${where}: gemmGrid.ladder has ${ladder.length} tile${ladder.length === 1 ? "" : "s"} (the rule walks down two or more)`,
    );
  }
  ladder.forEach((geometry, index) => {
    const path = `gemmGrid.ladder[${index}]`;
    assertGemmFieldsIn(geometry, where, path, { form: true, decode: false });
    assertGemmGeometry(geometry, `${where}: ${path}`);
    if (index === 0) return;
    const previous = ladder[index - 1];
    const area = gemmTileM(geometry) * gemmTileN(geometry);
    const previousArea = gemmTileM(previous) * gemmTileN(previous);
    if (!(area < previousArea)) {
      throw new CodegenError(
        `${where}: gemmGrid.ladder is not in strictly decreasing tile area ([${index - 1}] ${gemmTileM(previous)}x${gemmTileN(previous)} → [${index}] ${gemmTileM(geometry)}x${gemmTileN(geometry)} — the walk would never reach a later tile)`,
      );
    }
  });
  assertGridCount(rule.keepLargest.minWorkgroups, where, "keepLargest.minWorkgroups");
  assertGridCount(rule.keepLargest.minK, where, "keepLargest.minK");
  assertGridCount(rule.minWorkgroups, where, "minWorkgroups");
  if (!Number.isFinite(rule.maxPadding) || rule.maxPadding < 1) {
    throw new CodegenError(
      `${where}: gemmGrid.maxPadding must be a finite number of at least 1 (${String(rule.maxPadding)} — no tile covers less than the output)`,
    );
  }
};

/**
 * The grid-fill rule read from an unknown value (the shape guard of {@link assertGeometryProfile}):
 * an object of `ladder` (an array without holes of geometries), `keepLargest` (`minWorkgroups`,
 * `minK`), `minWorkgroups` and `maxPadding`. Its meaning is {@link assertGemmGridRule}'s job.
 */
const gemmGridOf = (value: unknown, where: string): GemmGridRule => {
  const fields = recordOf(value, where, "gemmGrid");
  if (!Array.isArray(fields.ladder)) {
    throw new CodegenError(`${where}: gemmGrid.ladder is not an array`);
  }
  const ladder = denseEntriesOf(fields.ladder, where, "gemmGrid.ladder").map((entry, index) =>
    gemmGeometryOf(entry, where, `gemmGrid.ladder[${index}]`),
  );
  const keepLargest = recordOf(fields.keepLargest, where, "gemmGrid.keepLargest");
  if (typeof fields.maxPadding !== "number") {
    throw new CodegenError(
      `${where}: gemmGrid.maxPadding is not a number (${fields.maxPadding === undefined ? "missing" : typeof fields.maxPadding})`,
    );
  }
  return {
    ladder,
    keepLargest: {
      minWorkgroups: assertGridCount(keepLargest.minWorkgroups, where, "keepLargest.minWorkgroups"),
      minK: assertGridCount(keepLargest.minK, where, "keepLargest.minK"),
    },
    minWorkgroups: assertGridCount(fields.minWorkgroups, where, "minWorkgroups"),
    maxPadding: fields.maxPadding,
  };
};

/**
 * The shape of `provenance` (field types, and a shared table's adapter list by
 * {@link sharedAdaptersViolation}). Matching its content is the job of `geometryProfileMismatch` in
 * `@karume/runtime/tune`, and the runtime does not read it — it is checked here because a value
 * that passes the guard is typed as {@link GeometryProfile}, so a value that differs from the type
 * declaration must not pass.
 */
const assertProvenanceShape = (value: unknown, where: string): void => {
  const provenance = recordOf(value, where, "provenance");
  const adapterShape = (entry: unknown, path: string): GeometryProfileAdapter => {
    const adapter = recordOf(entry, where, path);
    return {
      vendor: stringOf(adapter.vendor, where, `${path}.vendor`),
      architecture: stringOf(adapter.architecture, where, `${path}.architecture`),
      device: stringOf(adapter.device, where, `${path}.device`),
      description: stringOf(adapter.description, where, `${path}.description`),
    };
  };
  // The two forms of GeometryProfileAdapters: a measured table's one adapter, or a shared table's
  // list — never both and never neither
  if (provenance.adapters === undefined) {
    adapterShape(provenance.adapter, "provenance.adapter");
  } else {
    if (provenance.adapter !== undefined) {
      throw new CodegenError(
        `${where}: provenance has both adapter and adapters (a table is measured on one adapter or shared by several, not both)`,
      );
    }
    if (!Array.isArray(provenance.adapters)) {
      throw new CodegenError(`${where}: provenance.adapters is not an array`);
    }
    const adapters = denseEntriesOf(provenance.adapters, where, "provenance.adapters").map(
      (entry, index) => adapterShape(entry, `provenance.adapters[${index}]`),
    );
    const violation = sharedAdaptersViolation(adapters);
    if (violation !== undefined) {
      throw new CodegenError(`${where}: provenance.adapters ${violation}`);
    }
  }
  const families = recordOf(provenance.families, where, "provenance.families");
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    if (families[family] !== undefined) {
      assertFamilyProvenanceShape(families[family], where, `provenance.families.${family}`);
    }
  }
  if (provenance.attentionBench !== undefined) {
    assertAttentionBenchShape(provenance.attentionBench, where);
  }
};

/** The shape of one family's provenance (field types only, as {@link assertProvenanceShape}). */
const assertFamilyProvenanceShape = (value: unknown, where: string, path: string): void => {
  const entry = recordOf(value, where, path);
  for (const key of ["sweep", "sha256", "date", "candidateSet", "kernels", "caseSet"] as const) {
    stringOf(entry[key], where, `${path}.${key}`);
  }
  if (entry.userAgent === undefined) return;
  if (!Array.isArray(entry.userAgent)) {
    throw new CodegenError(`${where}: ${path}.userAgent is not an array`);
  }
  const agents = denseEntriesOf(entry.userAgent, where, `${path}.userAgent`);
  agents.forEach((agent, index) => stringOf(agent, where, `${path}.userAgent[${index}]`));
};

/** The shape of `provenance.attentionBench` (field types only, as {@link assertProvenanceShape}). */
const assertAttentionBenchShape = (value: unknown, where: string): void => {
  const bench = recordOf(value, where, "provenance.attentionBench");
  for (const key of ["records", "sha256", "date", "shapes"] as const) {
    stringOf(bench[key], where, `provenance.attentionBench.${key}`);
  }
  optionalStringOf(bench.kernels, where, "provenance.attentionBench.kernels");
};

/**
 * Guard on the optional {@link GeometryProfile.attentionOnline}: an object whose `keyWords` is
 * `"tiled"` (the one form a table can choose — the others are N's own).
 */
const assertAttentionOnlineShape = (value: unknown, where: string): void => {
  const { keyWords } = recordOf(value, where, "attentionOnline");
  if (keyWords !== "tiled") {
    throw new CodegenError(
      `${where}: attentionOnline.keyWords must be "tiled" (${JSON.stringify(keyWords)} — omit the field for the form N gives)`,
    );
  }
};

/** The storages a per-shape exception of the sequential variant may name (M = 1 takes every one). */
const SEQUENTIAL_GEMV_STORAGES: readonly WeightStorage[] = ["f32", "f16", "bf16", "i8", "i4", "i2"];

/** The storages of the parallel GEMV. */
const PARALLEL_GEMV_STORAGES: readonly WeightStorage[] = ["i2", "i4", "i8", "bf16", "f16", "f32"];

/** The lane counts the parallel GEMV generates. */
const PARALLEL_GEMV_LANES: readonly LinearGemvParallelLanes[] = [2, 4, 8, 16, 32];

const assertPositiveInteger = (value: unknown, where: string, path: string): void => {
  if (typeof value === "number" && Number.isSafeInteger(value) && value >= 1) return;
  throw new CodegenError(`${where}: ${path} must be a positive integer (${String(value)})`);
};

const assertOneOf = (
  values: readonly (string | number)[],
  value: unknown,
  where: string,
  path: string,
): void => {
  if (values.some((candidate) => candidate === value)) return;
  throw new CodegenError(
    `${where}: ${path} must be one of ${values.join(" / ")} (${JSON.stringify(value)})`,
  );
};

/** The rows of an array field (rejects a non-array and holes with the field path). */
const rowsOf = (value: unknown, where: string, path: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new CodegenError(`${where}: ${path} is not an array`);
  return denseEntriesOf(value, where, path);
};

/** A sequential variant: its field types here, its ranges by the kernel's own guard. */
const assertGemvVariantShape = (value: unknown, where: string, path: string): void => {
  const { cols, unroll } = recordOf(value, where, path);
  if (typeof cols !== "number" || typeof unroll !== "number") {
    throw new CodegenError(`${where}: ${path}: cols and unroll must be numbers`);
  }
  assertLinearGemvVariant({ cols, unroll }, `${where}: ${path}`);
};

const assertGemvShapeVariantShape = (value: unknown, where: string, path: string): void => {
  const { storage, n, k, variant } = recordOf(value, where, path);
  assertOneOf(SEQUENTIAL_GEMV_STORAGES, storage, where, `${path}.storage`);
  assertPositiveInteger(n, where, `${path}.n`);
  assertPositiveInteger(k, where, `${path}.k`);
  assertGemvVariantShape(variant, where, `${path}.variant`);
};

const assertGemvParallelShapeShape = (value: unknown, where: string, path: string): void => {
  const { storage, n, k, group, lanes, packedActivations, stagedActivations, rowBlock } = recordOf(
    value,
    where,
    path,
  );
  assertOneOf(PARALLEL_GEMV_STORAGES, storage, where, `${path}.storage`);
  assertPositiveInteger(n, where, `${path}.n`);
  assertPositiveInteger(k, where, `${path}.k`);
  if (group !== undefined) assertPositiveInteger(group, where, `${path}.group`);
  assertOneOf(PARALLEL_GEMV_LANES, lanes, where, `${path}.lanes`);
  if (typeof packedActivations !== "boolean") {
    throw new CodegenError(`${where}: ${path}.packedActivations is not a boolean`);
  }
  if (stagedActivations !== undefined && stagedActivations !== true) {
    throw new CodegenError(
      `${where}: ${path}.stagedActivations must be true or absent (${JSON.stringify(
        stagedActivations,
      )})`,
    );
  }
  // A float-storage row that named a group would match no linear (the lookup compares the group),
  // and one marked packed would name a kernel that does not exist: both are a broken table.
  const named = PARALLEL_GEMV_STORAGES.find((one) => one === storage);
  if (named !== undefined && isFloatStorage(named) && (group !== undefined || packedActivations)) {
    throw new CodegenError(
      `${where}: ${path}: a row of ${String(storage)} storage takes no group and no packedActivations`,
    );
  }
  if (rowBlock === undefined) return;
  assertOneOf(LINEAR_GEMV_PARALLEL_ROWS, rowBlock, where, `${path}.rowBlock`);
  if (packedActivations) {
    throw new CodegenError(
      `${where}: ${path}.rowBlock is set on a row that takes packed activations (the row-blocked kernel reads f32 activations)`,
    );
  }
  // The row-blocked kernel is the integer storages' alone: a float-stored row runs the kernel of
  // M = 1 at every row count it serves, and a row block on it would name a kernel that does not exist.
  if (named !== undefined && isFloatStorage(named)) {
    throw new CodegenError(
      `${where}: ${path}.rowBlock is set on a row of ${named} storage (the row-blocked kernel exists for i2 / i4 / i8 storage)`,
    );
  }
  // The storage and the lanes passed the checks above; the kernel's own predicate decides the fit.
  const rowLanes = PARALLEL_GEMV_LANES.find((one) => one === lanes);
  if (
    named === undefined ||
    rowLanes === undefined ||
    typeof rowBlock !== "number" ||
    !linearGemvParallelRowsFits(named, rowLanes, rowBlock)
  ) {
    throw new CodegenError(
      `${where}: ${path}.rowBlock ${JSON.stringify(rowBlock)} has no kernel for ${JSON.stringify(storage)} at ${JSON.stringify(lanes)} lanes (its staged activations exceed the workgroup storage)`,
    );
  }
};

/**
 * Guard on the optional {@link GeometryProfile.linearGemv}: the field types, and the ranges the
 * kernels and the lookup need — a variant the kernel would refuse, a thread target that is not a
 * positive integer (a NaN would silently keep every row block at its ceiling), a lane count the
 * parallel GEMV does not generate.
 */
const assertLinearGemvShape = (value: unknown, where: string): void => {
  const { sequential, rowsThreadTarget, parallelShapes } = recordOf(value, where, "linearGemv");
  const { variant, shapes } = recordOf(sequential, where, "linearGemv.sequential");
  assertGemvVariantShape(variant, where, "linearGemv.sequential.variant");
  rowsOf(shapes, where, "linearGemv.sequential.shapes").forEach((row, index) =>
    assertGemvShapeVariantShape(row, where, `linearGemv.sequential.shapes[${index}]`),
  );
  assertPositiveInteger(rowsThreadTarget, where, "linearGemv.rowsThreadTarget");
  rowsOf(parallelShapes, where, "linearGemv.parallelShapes").forEach((row, index) =>
    assertGemvParallelShapeShape(row, where, `linearGemv.parallelShapes[${index}]`),
  );
};

/**
 * Guard on one profile (shape of match, the rules, divisibility conditions of the geometry in every
 * field).
 *
 * The geometry also passes the same guards at generation time (`gemmWgsl` / the i8a8 generators);
 * it is rejected here first so that a broken profile is not left unnoticed **until it hits that
 * op** (checked once at Session construction).
 * NOTE: the w4a8 `groupSize % tileK` depends on the asset's group length, so it cannot be checked
 * here (the job of `assertI8a8GroupGeometry` at generation time).
 *
 * It takes unknown and is a type predicate that narrows a passing value to {@link GeometryProfile}
 * (`acquireGpu` also takes values built by JS callers or from stored data, so a missing field or a
 * wrong type is rejected with this guard's {@link CodegenError} rather than a TypeError). It checks
 * the type of every field (including `provenance`) but does not reject unknown fields (that is the
 * job of `parseGeometryProfileJson`, which reads stored data). It does not check device limits (the
 * job of {@link assertGeometryProfileWithinLimits} — that needs an adapter).
 */
export const assertGeometryProfile: (profile: unknown) => asserts profile is GeometryProfile = (
  profile,
) => {
  const fields = recordOf(profile, "Geometry profile", "table");
  const { id } = fields;
  if (typeof id !== "string") {
    throw new CodegenError(`Geometry profile: id is not a string (${typeof id})`);
  }
  if (id.length === 0) {
    throw new CodegenError(
      "Geometry profile: id is empty (diagnostics could not name the selected table)",
    );
  }
  const where = `Geometry profile '${id}'`;
  if (fields.match !== undefined) {
    const match = recordOf(fields.match, where, "match");
    const vendor = optionalStringOf(match.vendor, where, "match.vendor");
    const architecture = optionalStringOf(match.architecture, where, "match.architecture");
    const description = optionalStringOf(match.description, where, "match.description");
    if (vendor === "" || architecture === "" || description === "") {
      throw new CodegenError(
        `${where}: match vendor / architecture / description must not be an empty string (omit a field that is not specified)`,
      );
    }
    if (architecture !== undefined && vendor === undefined) {
      throw new CodegenError(`${where}: match architecture must be given together with vendor`);
    }
    // description is the model name, a field only for splitting within a vendor / architecture. If
    // it were allowed alone, a rule could hit a machine of another vendor with the same name (or
    // one with an empty vendor).
    if (description !== undefined && (vendor === undefined || architecture === undefined)) {
      throw new CodegenError(
        `${where}: match description must be given together with both vendor and architecture`,
      );
    }
  }
  assertGemmRowsRules(
    rulesOf(fields.gemmRows, where, "gemmRows", (value, path) => {
      const geometry = gemmGeometryOf(value, where, path);
      assertGemmFieldsIn(geometry, where, path, { form: true, decode: false });
      return geometry;
    }),
    where,
  );
  const attention = recordOf(fields.attention, where, "attention");
  const conv2d = recordOf(fields.conv2d, where, "conv2d");
  for (const [name, value, allowed] of [
    ["attention.qk", attention.qk, false],
    ["attention.pv", attention.pv, false],
    ["conv2d.rows64", conv2d.rows64, true],
    ["conv2d.rows32", conv2d.rows32, true],
  ] as const) {
    const geometry = gemmGeometryOf(value, where, name);
    assertGemmFieldsIn(geometry, where, name, { form: allowed, decode: allowed });
    assertGemmGeometry(geometry, `${where}: ${name}`);
  }
  const i8a8 = recordOf(fields.i8a8, where, "i8a8");
  assertI8a8LinearRules(
    rulesOf(i8a8.linear, where, "i8a8.linear", (value, path) => i8a8GeometryOf(value, where, path)),
    where,
  );
  for (const [name, geometry] of [
    ["i8a8.attentionQk", i8a8.attentionQk],
    ["i8a8.attentionPv", i8a8.attentionPv],
    ["i8a8.conv3d", i8a8.conv3d],
  ] as const) {
    assertI8a8Geometry(i8a8GeometryOf(geometry, where, name), `${where}: ${name}`);
  }
  if (fields.attentionOnline !== undefined)
    assertAttentionOnlineShape(fields.attentionOnline, where);
  if (fields.linearGemv !== undefined) assertLinearGemvShape(fields.linearGemv, where);
  if (fields.gemmGrid !== undefined) assertGemmGridRule(gemmGridOf(fields.gemmGrid, where), where);
  if (fields.provenance !== undefined) assertProvenanceShape(fields.provenance, where);
};

/**
 * The device limits the limit guard checks (the 4 fields of `GPUSupportedLimits` decided by the
 * geometry alone — {@link assertGeometryProfileWithinLimits}). The requiredLimits planned by
 * `acquireGpu` (RequiredLimits) can be passed as is.
 */
export type WorkgroupLimits = {
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeWorkgroupSizeY: number;
  readonly maxComputeWorkgroupStorageSize: number;
};

/**
 * Spelling of the geometry in the over-limit message (names the geometry by its values — the field
 * name alone is not enough to fix a table brought in from outside).
 */
const geometryLabel = (geometry: GemmGeometry | I8a8Geometry): string =>
  `regM=${geometry.regM} regN=${geometry.regN} wgX=${geometry.wgX} wgY=${geometry.wgY}${
    "tileK" in geometry ? ` tileK=${geometry.tileK}` : ""
  }`;

/**
 * Guard that the geometry of every field of a profile fits the device's workgroup limits (takes only
 * tables that passed the shape guard — the order is {@link assertGeometryProfile} → this). It names
 * every field that exceeds a limit and rejects with a {@link CodegenError}.
 *
 * The limits checked are the same 4 as WebGPU's compute pipeline validation: each component of
 * `@workgroup_size(wgX, wgY)` (`maxComputeWorkgroupSizeX` / `Y`), the thread count wgX · wgY
 * (`maxComputeInvocationsPerWorkgroup`), and the workgroup storage of the shared tiles
 * (`maxComputeWorkgroupStorageSize` — {@link gemmWorkgroupStorageBytes} /
 * {@link i8a8WorkgroupStorageBytes}).
 * MUST: an injected table passes here before the device is created. A pipeline with an exceeding
 * geometry fails validation, which goes unnoticed until that op is hit (Session construction, the
 * first run) — the same discipline as "do not create a device with a broken table".
 * GEMM shared memory is counted for the f32 variant (twice f16): a table applies to every variant
 * without choosing the compute variant, so a table where the larger one does not fit fails on some
 * path.
 * NOTE: the dispatch count limit (`maxComputeWorkgroupsPerDimension`) is not checked. It is decided
 * by the geometry together with the **run-time shape** (`ceil(M / tileM)` and so on) and cannot be
 * judged from the table alone — that is the job of `tiledWorkgroups`, which rejects with
 * `DispatchLimitError` when the dispatch is built (src/codegen/dispatch.ts). This guard checks only
 * the shape of one workgroup, which does not depend on the shape of the data.
 * NOTE: tests pin that the embedded tables and the default table pass under WebGPU's default limits
 * (256 / 256 / 256 / 16,384 B — every device provides at least these), so the auto-selection path
 * does not go through here.
 */
export const assertGeometryProfileWithinLimits = (
  profile: GeometryProfile,
  limits: WorkgroupLimits,
): void => {
  const violations: string[] = [];
  const check = (
    name: string,
    geometry: GemmGeometry | I8a8Geometry,
    storageBytes: number,
  ): void => {
    const exceeds = (what: string, value: number, limit: keyof WorkgroupLimits): void => {
      if (value > limits[limit]) {
        violations.push(
          `${name} (${geometryLabel(geometry)}): ${what} ${value} exceeds ${limit} ${
            limits[limit]
          }`,
        );
      }
    };
    exceeds(
      "thread count wgX × wgY",
      geometry.wgX * geometry.wgY,
      "maxComputeInvocationsPerWorkgroup",
    );
    exceeds("wgX", geometry.wgX, "maxComputeWorkgroupSizeX");
    exceeds("wgY", geometry.wgY, "maxComputeWorkgroupSizeY");
    exceeds("shared memory (bytes)", storageBytes, "maxComputeWorkgroupStorageSize");
  };
  const gemm = (name: string, geometry: GemmGeometry): void =>
    check(name, geometry, gemmWorkgroupStorageBytes(geometry, "f32"));
  const i8a8 = (name: string, geometry: I8a8Geometry): void =>
    check(name, geometry, i8a8WorkgroupStorageBytes(geometry));
  profile.gemmRows.forEach((rule, index) => gemm(`gemmRows[${index}]`, rule.geometry));
  profile.gemmGrid?.ladder.forEach((geometry, index) =>
    gemm(`gemmGrid.ladder[${index}]`, geometry),
  );
  gemm("attention.qk", profile.attention.qk);
  gemm("attention.pv", profile.attention.pv);
  gemm("conv2d.rows64", profile.conv2d.rows64);
  gemm("conv2d.rows32", profile.conv2d.rows32);
  profile.i8a8.linear.forEach((rule, index) => i8a8(`i8a8.linear[${index}]`, rule.geometry));
  i8a8("i8a8.attentionQk", profile.i8a8.attentionQk);
  i8a8("i8a8.attentionPv", profile.i8a8.attentionPv);
  i8a8("i8a8.conv3d", profile.i8a8.conv3d);
  if (violations.length > 0) {
    throw new CodegenError(
      `Geometry profile '${profile.id}': geometry exceeds the device limits (${violations.join(" / ")})`,
    );
  }
};

/**
 * The adapter fields used for matching in the selection (`GpuContext.adapterInfo` can be passed as
 * is). `description` is `GPUAdapterInfo.description` as is, and an empty string is treated the same
 * as absent (Chrome without the flag returns an empty one).
 */
type AdapterIdentity = {
  readonly vendor: string;
  readonly architecture: string;
  readonly description?: string;
};

/**
 * Chooses one profile for the adapter (called once at Session construction).
 *
 * Selection order (deterministic): ① vendor, architecture and description of `match` all match →
 * ② description unset, vendor and architecture match → ③ architecture unset, only vendor matches →
 * ④ {@link DEFAULT_GEOMETRY_PROFILE}. Matching is exact string equality (no normalisation, no prefix
 * or substring matching). If the adapter's description is empty or absent, ① never matches. A table
 * without `match` is not matched (injection only), but it still goes through the guard and the
 * duplicate-id check (so a broken table is not left in the list unnoticed).
 *
 * MUST: a list in which 2 tables can hit the same rank fails loudly — silently choosing one by list
 * order would let an edit of the list alone change the selection. **The list itself is rejected,
 * not only for the adapter that hits it** (2 tables with the same `match` collide only on a machine
 * with that `match` — on other machines' CI it would stay invisible).
 * MUST: do not choose the nearest neighbour by similarity (closeness of GPU names and so on). That
 * would break the correspondence between the reference sha rows per environment key (ADR 0106) and
 * "which geometry that machine ran with".
 */
export const selectGeometryProfile = (
  adapter: AdapterIdentity,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): GeometryProfile => {
  const ids = new Set([DEFAULT_GEOMETRY_PROFILE.id]);
  const matches = new Set<string>();
  for (const profile of profiles) {
    assertGeometryProfile(profile);
    if (ids.has(profile.id)) {
      throw new CodegenError(
        `Geometry profile '${profile.id}': duplicate id (diagnostics could not name the table)`,
      );
    }
    ids.add(profile.id);
    if (profile.match === undefined) continue;
    const { vendor, architecture, description } = profile.match;
    if (vendor === undefined) {
      throw new CodegenError(
        `Geometry profile '${profile.id}': match is empty (only the default profile matches every adapter — ` +
          "an injection-only table omits match)",
      );
    }
    // The JSON array form cannot collide on a separator (`a/b` + `c` is not confused with `a` + `b/c`).
    const match = JSON.stringify([vendor, architecture ?? null, description ?? null]);
    if (matches.has(match)) {
      throw new CodegenError(
        `Geometry profile '${profile.id}': match (vendor=${vendor} / architecture=${
          architecture ?? "(unspecified)"
        } / description=${
          description ?? "(unspecified)"
        }) is the same as another profile's (two tables would match at the same rank)`,
      );
    }
    matches.add(match);
  }
  // A table without match has `profile.match?.vendor` undefined, which never equals the string
  // adapter.vendor.
  const described = adapter.description === "" ? undefined : adapter.description;
  const withDescription =
    described === undefined
      ? undefined
      : profiles.find(
          (profile) =>
            profile.match?.vendor === adapter.vendor &&
            profile.match.architecture === adapter.architecture &&
            profile.match.description === described,
        );
  if (withDescription !== undefined) return withDescription;
  const exact = profiles.find(
    (profile) =>
      profile.match?.vendor === adapter.vendor &&
      profile.match.architecture !== undefined &&
      profile.match.architecture === adapter.architecture &&
      profile.match.description === undefined,
  );
  if (exact !== undefined) return exact;
  const vendorOnly = profiles.find(
    (profile) =>
      profile.match?.vendor === adapter.vendor && profile.match.architecture === undefined,
  );
  return vendorOnly ?? DEFAULT_GEOMETRY_PROFILE;
};

/**
 * Looks up the geometry from the row count M (the 3 paths matmul / bmm / linear).
 *
 * MUST: the derivation phase passes **the return value of this one call** to all three of key, WGSL
 * and dispatch (if one of them got a different geometry, the cached WGSL and the dispatch count would
 * not match, and output tiles would be missing without any exception).
 */
export const gemmRowsGeometry = (profile: GeometryProfile, rows: number): GemmGeometry => {
  assertU32Params("Geometry selection", { "row count M": rows });
  assertGemmRowsRules(profile.gemmRows, `Geometry profile '${profile.id}'`);
  return ruleFor(profile.gemmRows, rows).geometry;
};

/** The workgroup count of a tile's dispatch grid on a shape: `ceil(n / tileN) · ceil(m / tileM) · batch`. */
export const gemmGridWorkgroups = (geometry: GemmGeometry, shape: GemmShape): number =>
  Math.ceil(shape.n / gemmTileN(geometry)) * Math.ceil(shape.m / gemmTileM(geometry)) * shape.batch;

/**
 * A tile's padding on a shape: the area its grid covers over the output's,
 * `(ceil(n / tileN) · tileN · ceil(m / tileM) · tileM) / (m · n)` (1 when the tiles divide the
 * output).
 */
export const gemmGridPadding = (geometry: GemmGeometry, shape: GemmShape): number => {
  const tileM = gemmTileM(geometry);
  const tileN = gemmTileN(geometry);
  return (
    (Math.ceil(shape.n / tileN) * tileN * Math.ceil(shape.m / tileM) * tileM) / (shape.m * shape.n)
  );
};

/**
 * Whether a tile's padding is within the rule's bound. Compared as products of integers (exact
 * below 2^53), so a padding of exactly the bound is in whatever the division would round to.
 */
const withinPadding = (geometry: GemmGeometry, shape: GemmShape, maxPadding: number): boolean => {
  const tileM = gemmTileM(geometry);
  const tileN = gemmTileN(geometry);
  const covered = Math.ceil(shape.n / tileN) * tileN * Math.ceil(shape.m / tileM) * tileM;
  return covered <= maxPadding * shape.m * shape.n;
};

/**
 * The step of the grid-fill rule's ladder a shape takes (the walk of {@link GemmGridRule}): 0 when
 * the largest tile is kept, else the first smaller tile whose grid is full enough and whose padding
 * is within the bound, else the last. Pure; the rule and the shape are checked by the caller
 * ({@link gemmShapeGeometry}).
 */
export const gemmGridStep = (rule: GemmGridRule, shape: GemmShape): number => {
  const [largest] = rule.ladder;
  if (
    shape.k >= rule.keepLargest.minK &&
    gemmGridWorkgroups(largest, shape) >= rule.keepLargest.minWorkgroups &&
    withinPadding(largest, shape, rule.maxPadding)
  ) {
    return 0;
  }
  for (let step = 1; step < rule.ladder.length; step += 1) {
    const tile = rule.ladder[step];
    if (
      gemmGridWorkgroups(tile, shape) >= rule.minWorkgroups &&
      withinPadding(tile, shape, rule.maxPadding)
    ) {
      return step;
    }
  }
  return rule.ladder.length - 1;
};

/**
 * Looks up the tile geometry of one GEMM of linear / matmul / bmm from its shape (the three call
 * sites: the GEMM and the split-K partial of src/runtime/linear-program.ts, `buildMatmul` and
 * `buildBmm` of src/runtime/recipe-builders/linear.ts). With the profile's grid-fill rule
 * ({@link GeometryProfile.gemmGrid}) the tile is the ladder's step for the shape
 * ({@link gemmGridStep}); without it, the `gemmRows` rule of the row count M
 * ({@link gemmRowsGeometry} — exactly the lookup before the rule existed, so a table without the
 * field keeps every key and byte of WGSL). bmm's M is the row count of one matrix and its batch
 * the grid's z axis; linear and matmul have a batch of 1.
 *
 * MUST: the derivation phase passes **the return value of this one call** to the key, the WGSL and
 * the dispatch alike (the same discipline as {@link gemmRowsGeometry}).
 * SPEC: docs/spec/geometry-profiles.md#the-grid-fill-rule-of-the-gemm
 */
export const gemmShapeGeometry = (profile: GeometryProfile, shape: GemmShape): GemmGeometry => {
  const rule = profile.gemmGrid;
  if (rule === undefined) return gemmRowsGeometry(profile, shape.m);
  const where = `Geometry profile '${profile.id}'`;
  assertU32Params("Geometry selection", {
    "row count M": shape.m,
    "column count N": shape.n,
    "reduction length K": shape.k,
    batch: shape.batch,
  });
  for (const [name, value] of [
    ["M", shape.m],
    ["N", shape.n],
    ["K", shape.k],
    ["batch", shape.batch],
  ] as const) {
    if (value < 1) {
      throw new CodegenError(
        `Geometry selection: ${name} is ${value} (a GEMM without outputs or terms has no grid to fill)`,
      );
    }
  }
  assertGemmGridRule(rule, where);
  return rule.ladder[gemmGridStep(rule, shape)];
};

/**
 * Looks up the i8a8 linear's geometry from the row count M (`i8a8.linear` — ADR 0115 addendum
 * decision 14), by the same rule order as {@link gemmRowsGeometry}.
 *
 * MUST: one call per linear, and its return value (after the w4a8 fallback of
 * src/runtime/linear-program.ts) feeds the key, the WGSL, the dispatch and
 * `linearI8a8UsesVecFill` alike — the sweep's case plan and the fingerprint select by the case's M
 * through this same function.
 */
export const i8a8LinearGeometry = (profile: GeometryProfile, rows: number): I8a8Geometry => {
  assertU32Params("Geometry selection", { "row count M": rows });
  assertI8a8LinearRules(profile.i8a8.linear, `Geometry profile '${profile.id}'`);
  return ruleFor(profile.i8a8.linear, rows).geometry;
};

/**
 * Looks up the profile's geometry from the conv2d m-tile class (64 / 32 as returned by
 * `conv2dIgemmMTile`).
 *
 * MUST: there are only 2 classes. Any other m tile means the predicate and the table disagree (a
 * value was added to only one of them), so fail loudly — rounding to the nearer one would silently
 * apply another class's geometry to a class that was not measured.
 */
export const conv2dProfileGeometry = (profile: GeometryProfile, mTile: number): GemmGeometry => {
  if (mTile === GEMM_TILE) return profile.conv2d.rows64;
  if (mTile === GEMM_MTILE_SMALL) return profile.conv2d.rows32;
  throw new CodegenError(
    `Geometry profile '${profile.id}': no class matches conv2d m tile ${mTile} (only ${GEMM_TILE} / ${GEMM_MTILE_SMALL})`,
  );
};

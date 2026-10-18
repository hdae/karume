/**
 * The **tile-geometry parameter surface** shared by the f32 / f16 GEMM skeleton (the matmul / bmm /
 * linear, ①QK / ③PV of fused attention, ①ₜ / ③ₜ of states-shaped attention and the implicit GEMM of
 * conv1d / conv2d / conv3d held by src/kernels/gemm.ts).
 *
 * ```
 * tileM = regM · wgY   (rows of output one workgroup holds — only the implicit GEMM of
 *                       conv1d / conv2d / conv3d takes a variant of 32)
 * tileN = regN · wgX   (columns                          )
 * threads = wgX · wgY  (one thread holds regM×regN outputs)
 * ```
 *
 * ## Why parameterising is allowed (relation to the numeric contract)
 *
 * MUST: the geometry may decide only **which thread handles which output**. The K reduction order
 * per output element (outer `t` ascending, inner `kk` ascending) and the literal form of the
 * multiply-add (`acc = acc + a * b`) do not change with the geometry, and that is the numeric
 * contract of ADR 0022 decision 3 itself (unlike the i32 reduction of i8a8, f32 addition is
 * order-dependent, so touching anything other than the work assignment moves the values).
 * MUST: the geometry **goes into the pipeline key** ({@link gemmGeometryKeyPart}). Without it the
 * determinism "same key → byte-identical WGSL" breaks.
 * MUST: even where the concepts overlap, do not share the implementation with the i8a8 side
 * (src/kernels/i8a8-geometry.ts). There the geometry is free precisely because integer reduction is
 * order-independent; here only the work assignment of the f32 reduction is free — the numeric
 * contracts differ, so being independent is itself an invariant.
 *
 * ## Where the default comes from
 *
 * {@link defaultGemmGeometry} is the only selection point. In end-to-end measurement on an RTX 3080
 * Ti (ABBA palindrome with the cool-down rule — docs/research/2026-08-10-f32-geometry-probe.md),
 * against `r8×4 wg16×8`, the best of the 64×64 sweep, `M128N128 r8×8 wg16×16` with tile axis 128
 * (256 threads, 16KB shared) gave a further ×1.17 on f16-1024 (w8a8-s16-1024 neutral). What
 * mattered, in order, was "first the register block (regM·regN = 32 → 64), second the tile edge",
 * the same as in the i8a8 sweep, but the best tile shape **does not match** i8a8's (M128N64) — the
 * answer on geometry comes only from measurement per path.
 * With every geometry the PNG / WAV sha256 checks are bit-identical (an empirical claim — ADR 0022
 * addendum).
 *
 * ## Shape buckets
 *
 * The default is a shape that assumes M ≥ 128, so only for skinny M (M < 128) does
 * {@link gemmGeometryForRows} return another geometry. **The selection is a pure function of the
 * static shape** (run-time autotuning is banned by ADR 0022), and the return value goes into the key
 * as is, so "same key → byte-identical WGSL" is kept.
 *
 * NOTE: the reach of "the only selection point" above is **inside this skeleton** (which geometry
 * cuts the tiles). linear has another family that does not use this skeleton at all for M=1 × i4
 * (the GEMV of {@link "./linear-gemv.ts"} — ADR 0082), and family selection lives in one place,
 * `buildLinear` in `src/runtime/recipe-builders/linear.ts`. With two stages, both stages remain pure
 * functions and both still go into the key (ADR 0082 decision 2).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/**
 * The base row count of the conv2d m-tile heuristic (`conv2dIgemmMTile`). Different from the tileM
 * (128) of the default geometry — conv2d has few output channels, so it keeps the m tile at 64 / 32.
 *
 * MUST: **the canonical source of the actual tile edges is the geometry** ({@link gemmTileM} /
 * {@link gemmTileN}). This constant is vocabulary of the heuristic; if generation or dispatch read
 * it directly as an edge, there would be 2 values that can disagree with the geometry.
 */
export const GEMM_TILE = 64;

/**
 * K tile width. It is **not geometry but numeric contract** (ADR 0022's MUST), so it is not part of
 * {@link GemmGeometry} — changing it moves the step of the additions per output element.
 */
export const GEMM_TILE_K = 16;

/**
 * Components of a vec4. The granularity of how the shared B tile is grouped and of quad reads and
 * writes of inputs and outputs — **an axis separate from the geometry**.
 */
export const GEMM_QUAD = 4;

/** Quads per K tile (used for the thread assignment of the A / W tile fill). */
export const GEMM_K_QUADS = GEMM_TILE_K / GEMM_QUAD;

/**
 * Generation parameters. The output tile edges are **derived** as `regM · wgY` × `regN · wgX`
 * (holding the tile edges independently gives 2 values that can disagree — the same discipline as
 * the i8a8 side).
 */
export type GemmGeometry = {
  /** Rows of output one thread holds. */
  readonly regM: number;
  /**
   * Columns of output one thread holds (a multiple of {@link GEMM_QUAD}, because `acc` is grouped
   * into vec4).
   */
  readonly regN: number;
  /** x edge of the workgroup (column direction). */
  readonly wgX: number;
  /** y edge of the workgroup (row direction). */
  readonly wgY: number;
};

/** Rows of the output tile. The dispatch's y is `ceil(m / tileM)`. */
export const gemmTileM = (geometry: GemmGeometry): number => geometry.regM * geometry.wgY;

/** Columns of the output tile. The dispatch's x is `ceil(n / tileN)`. */
export const gemmTileN = (geometry: GemmGeometry): number => geometry.regN * geometry.wgX;

/** Threads per workgroup. */
export const gemmThreads = (geometry: GemmGeometry): number => geometry.wgX * geometry.wgY;

/** Column quads one thread holds (the column-side count of `acc{row}_{column quad}`). */
export const gemmQuadsPerThread = (geometry: GemmGeometry): number => geometry.regN / GEMM_QUAD;

/** Column quads of the output tile (the column stride of the shared B tile). */
export const gemmColumnQuads = (geometry: GemmGeometry): number => gemmTileN(geometry) / GEMM_QUAD;

/**
 * When filling a tile laid out as `[row][k quad]` (A / linear's W / ①QK's k / conv2d's weight), the
 * number of rows one slot advances. One thread takes {@link gemmRowSlots} entries (on the W side
 * {@link gemmColumnSlots}) from its own row with this step.
 */
export const gemmRowFillStride = (geometry: GemmGeometry): number =>
  gemmThreads(geometry) / GEMM_K_QUADS;

/** Entries of the A tile ({@link gemmTileM} rows) one thread fills. */
export const gemmRowSlots = (geometry: GemmGeometry): number =>
  gemmTileM(geometry) / gemmRowFillStride(geometry);

/**
 * Entries one thread handles in the W / k tile ({@link gemmTileN}, since output channels = columns
 * are read as rows).
 */
export const gemmColumnSlots = (geometry: GemmGeometry): number =>
  gemmTileN(geometry) / gemmRowFillStride(geometry);

/**
 * When filling a dense B tile laid out as `[k row][column quad]`, the number of k rows one slot
 * advances.
 */
export const gemmQuadFillStride = (geometry: GemmGeometry): number =>
  gemmThreads(geometry) / gemmColumnQuads(geometry);

/** Entries of the dense B tile (K {@link GEMM_TILE_K} rows) one thread fills. */
export const gemmQuadSlots = (geometry: GemmGeometry): number =>
  GEMM_TILE_K / gemmQuadFillStride(geometry);

/**
 * Rejects an inconsistent geometry at generation time. **A combination that does not divide evenly
 * leaves holes in the shared tile** (words the fill does not reach enter the inner product as 0 =
 * a wrong value with no exception), so failing loudly is the only guard.
 */
export const assertGemmGeometry = (geometry: GemmGeometry, where: string): void => {
  const { regM, regN, wgX, wgY } = geometry;
  for (const [name, value] of [
    ["regM", regM],
    ["regN", regN],
    ["wgX", wgX],
    ["wgY", wgY],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new CodegenError(`${where}: 幾何の ${name} は正整数（${value}）`);
    }
  }
  // If regN is a multiple of the quad, so is tileN, so the `n4` conversion of the v4 path follows here.
  if (regN % GEMM_QUAD !== 0) {
    throw new CodegenError(`${where}: regN は ${GEMM_QUAD} の倍数（acc は vec4 束ね・${regN}）`);
  }
  const threads = gemmThreads(geometry);
  for (const [name, unit] of [
    ["K quad", GEMM_K_QUADS],
    ["列 quad", gemmColumnQuads(geometry)],
  ] as const) {
    if (threads % unit !== 0) {
      throw new CodegenError(
        `${where}: スレッド数 ${threads} が ${name} 数 ${unit} で割り切れない（充填の担当が組めない）`,
      );
    }
  }
  const tileM = gemmTileM(geometry);
  const tileN = gemmTileN(geometry);
  const rowStride = gemmRowFillStride(geometry);
  if (tileM % rowStride !== 0 || tileN % rowStride !== 0) {
    throw new CodegenError(
      `${where}: タイル辺 ${tileM}x${tileN} が行充填ストライド ${rowStride} で割り切れない`,
    );
  }
  const quadStride = gemmQuadFillStride(geometry);
  if (GEMM_TILE_K % quadStride !== 0) {
    throw new CodegenError(
      `${where}: K タイル ${GEMM_TILE_K} が quad 充填ストライド ${quadStride} で割り切れない`,
    );
  }
};

/**
 * The geometry discriminator of the pipeline key (appended after the tile-edge fragment
 * `reg{tileM}x{tileN}`).
 *
 * It adds what is needed because **the tile edges alone do not determine the geometry** (64×64 can
 * be built as `r4×4 wg16×16` or as `r8×4 wg16×8`, so a key of edges alone would collide with a
 * different kernel). wgY is not included — it is fixed by `tileM / regM`, so once the edges and regM
 * are in, there is nothing left to discriminate. The implicit GEMM of conv2d carries the workgroup
 * shape in its own key (`igemm{tileM}x{tileN}…:wg{x}x{y}`), so it has the same discriminating power
 * without this fragment.
 */
const gemmGeometryKeyPart = (geometry: GemmGeometry): string =>
  `r${geometry.regM}x${geometry.regN}w${geometry.wgX}`;

/**
 * The "tile edges + geometry" key fragment (`reg128x128r8x8w16`).
 *
 * The **only spelling** shared by `gemmKeyPart` in gemm.ts (which appends the v4 bit) and the key of
 * states-shaped ①ₜ (`stateQkTiledKey` in src/kernels/state-attention.ts). Copied into 2 places, only
 * one of the keys would lose its discriminating power when a geometry parameter is added.
 */
export const gemmGeometryTileKeyPart = (geometry: GemmGeometry): string =>
  `reg${gemmTileM(geometry)}x${gemmTileN(geometry)}${gemmGeometryKeyPart(geometry)}`;

/** Geometry note for generated comments (in a form that fits on one WGSL header line). */
export const gemmGeometryNote = (geometry: GemmGeometry): string =>
  `レジスタ ${gemmTileM(geometry)}x${gemmTileN(
    geometry,
  )} タイル / 1 スレッド ${geometry.regM}x${geometry.regN} / wg ${geometry.wgX}x${geometry.wgY}`;

/**
 * The **only pure function** that decides the default geometry. One point common to every op, with
 * no per-op defaults (if they become necessary, this can be raised to the same shape as the i8a8
 * side (`defaultI8a8Geometry` in {@link "./i8a8-geometry.ts"}), so no machinery that is not needed
 * now is added).
 *
 * `M128N128 r8×8 wg16×16` comes from the measurement at the top of the module.
 * MUST: a change of the default goes together with re-measuring the PNG / WAV checks (ADR 0022
 * addendum — bit identity is an empirical claim).
 */
export const defaultGemmGeometry = (): GemmGeometry => ({ regM: 8, regN: 8, wgX: 16, wgY: 16 });

/**
 * Candidates for the small-M buckets (the table of {@link gemmGeometryForRows}).
 *
 * ## Why split on M
 *
 * The default `M128N128` has one workgroup hold 128 rows × 128 columns, so **for M < 128 only one
 * row tile is launched**. The dispatch count is just `ceil(n / 128)`, and on the measured shapes of
 * EmbeddingGemma (bare T=4) `M=4, N ∈ {256, 768, 1152, 3072}` there are 2–24 workgroups = not enough
 * to fill the SMs (measured: 170 dispatches in 41.3ms, 242.6µs each — two orders of magnitude above
 * what the weight traffic would require).
 *
 * With one row tile, the weight `[n,k]` is read **exactly once with every geometry** (no overlap
 * across tiles), so the only lever the geometry has is "how parallel the same traffic can flow".
 * The thread count of the whole grid is `n · wgY / regN` and the workgroup count is `n / tileN`, so
 * the direction is **keep tileM just above M to gain wgY, and shrink tileN to add workgroups**.
 *
 * ## Why they divide evenly (the 5 conditions of {@link assertGemmGeometry})
 *
 * | Geometry | tileM×tileN | threads | rowFillStride | quadFillStride | Shared |
 * | --- | --- | --- | --- | --- | --- |
 * | M16N16 `r1×4 wg4×16` | 16×16 | 64 | 16 | 4 | 2,048 B |
 * | default M128N128 `r8×8 wg16×16` | 128×128 | 256 | 64 | 8 | 16,384 B |
 *
 * They satisfy `regN % 4 == 0`, `threads % 4` (K quads) `== 0`, `threads % (tileN/4)` (column
 * quads) `== 0`, `tileM % rowFillStride == 0 && tileN % rowFillStride == 0` and
 * `16 % quadFillStride == 0`. **`tileM ≥ threads / 4` is the effective upper bound** (the row fill
 * stride cannot exceed the tile edge): with tileM=16, threads ≤ 64; with tileM=64, threads ≤ 256.
 *
 * ## Where the values come from (RTX 3080 Ti, sweep of 2026-08-11 — docs/research/2026-08-11-skinny-m-geometry.md)
 *
 * ABBA paired measurement of 11 candidates × the real linear shapes of EmbeddingGemma ×
 * M ∈ {1,4,64,318,512} (bit identity confirmed at the same time for every candidate and every
 * shape). Results weighted by the shape mix of a real run:
 *
 * - For M = 1 / 4 / 64 alike, **M16N16 gives ×3.0–3.2 over the default** and beat
 *   `M64N32 r4×4 wg8×16` (×2.5), which had been placed as the second step, **even at M=64** —
 *   adding workgroups, even by splitting into 4 row tiles, wins over placing tileM just above M to
 *   avoid re-reading the weights. So the M ≤ 64 bucket is **the single step M16N16** (the conclusion
 *   at the time of this 3-point sweep — the 65–512 bucket was added later, see the item below).
 * - `M16N8` (threads 32) ties with M16N16 within less than 1% (slightly ahead at all 3 points, but
 *   equal under the rule). The tied candidate is not taken; the implemented and tested M16N16 is
 *   kept.
 * - For M = 318 / 512, `M64N32 r4×4 wg8×16` is best at ×1.67 / ×1.28 (M16N16 drops to ×1.35 —
 *   re-reading 20–32 row tiles starts to matter). The 65–512 bucket was first deferred because of
 *   its spill-over into Anima / SBV2, but was **adopted together with the end-to-end A/B of both
 *   models (ABBA, including the PNG/WAV checks)** (phase ① of 2026-08-11 — measurements in
 *   docs/research/2026-08-11-skinny-m-geometry.md §4).
 * - M ≥ 513 stays at the default. The measured points of the sweep go up to 512, and for the DiT
 *   region (S = 1024 / 4096) the default was chosen by the tile-axis sweep of 2026-08-10 — a range
 *   without measurements is not filled by interpolation.
 */
const GEOMETRY_M16N16: GemmGeometry = { regM: 1, regN: 4, wgX: 4, wgY: 16 };

const GEOMETRY_M64N32: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };

/**
 * The table of {@link gemmGeometryForRows} itself (the first rule with `rows <= maxRows` applies;
 * `maxRows` ascending; the last one has no upper bound). Where the boundaries 64 / 512 and each
 * step come from: {@link gemmGeometryForRows} below and the doc of the candidate table above.
 *
 * MUST: `gemmRows` of the default profile (`DEFAULT_GEOMETRY_PROFILE` in
 * src/kernels/geometry-profile.ts) **references this array as is**. Copying the boundaries or
 * geometries over there would put the default in 2 places, and when only one is fixed, the profile
 * path (matmul / bmm / linear) and the paths that read this table directly (states-shaped ①ₜ / ③ₜ,
 * the row blocks of decomposed attention) would choose different geometries for the same M.
 */
export const GEMM_ROWS_BUCKETS: readonly {
  readonly maxRows: number;
  readonly geometry: GemmGeometry;
}[] = [
  { maxRows: 64, geometry: GEOMETRY_M16N16 },
  { maxRows: 512, geometry: GEOMETRY_M64N32 },
  { maxRows: Number.POSITIVE_INFINITY, geometry: defaultGemmGeometry() },
];

/**
 * A **static table** that chooses the geometry from the row count M (only for the 3 paths matmul /
 * bmm / linear).
 *
 * MUST: a pure function = a function of plan-time shapes only. Run-time autotuning (measuring and
 * re-choosing) is banned by ADR 0022 — the f32 reduction is free only in its work assignment, and if
 * the choice varied per run, "same key → byte-identical WGSL", the bit identity of the PNG / WAV
 * checks and the meaning of the key would all break at once.
 * MUST: **for M ≥ 513 return the default as is**. This boundary keeps the measured choice for the DiT
 * (S = 1024 / 4096) (tile-axis sweep of 2026-08-10) in place, and the range outside the sweep's
 * measured points (M ≤ 512) is not filled by interpolation.
 * MUST: fused attention (①QK / ③PV) does not go through this table. Its default was chosen by
 * measurement on Anima, so switching it on M (= query length) on its own would remove the premise
 * of that measurement.
 *
 * The bucket boundaries 64 / 512 are the measured boundaries of the sweep (M16N16 is best up to
 * M=64, M64N32 at M=318/512, above that unmeasured = default). The 65–512 step was adopted after the
 * end-to-end A/B of Anima / SBV2 (ABBA, including the checks) confirmed no regression (research doc
 * §4).
 *
 * NOTE: this is **the table of the default profile**. A Session's matmul / bmm / linear read the
 * per-adapter profile (`gemmRowsGeometry` in src/kernels/geometry-profile.ts), and the default
 * profile holds this table ({@link GEMM_ROWS_BUCKETS}) as is. The selection is "a static table of
 * shape × adapter", and the ban on run-time autotuning is unchanged (DECIDED: ADR 0115).
 */
export const gemmGeometryForRows = (rows: number): GemmGeometry => {
  assertU32Params("幾何の選択", { "行数 M": rows });
  // The last step has no upper bound (Infinity), so a u32 row count always hits one of the steps.
  const bucket =
    GEMM_ROWS_BUCKETS.find((rule) => rows <= rule.maxRows) ??
    GEMM_ROWS_BUCKETS[GEMM_ROWS_BUCKETS.length - 1];
  return bucket.geometry;
};

/**
 * Initialisation of the f32 accumulator. It uses **named variables `acc{row}_{column quad}` rather
 * than one array** because the dynamic index of `acc[i]` requires an addressable function-local
 * region, which does not stay in registers and spills to local memory (pronounced on Metal).
 * Unrolled, the K tile width, the ascending `t` / `kk` and the addition order per output element do
 * not change (the reductions, independent per output, are only reassigned to names).
 *
 * `init` is the initial-value expression for one row (default 0). Only the implicit GEMM of conv1d /
 * conv2d inserts the bias of row = output channel, **bias-first** (ADR 0024's MUST).
 */
export const gemmAccumulatorInit = (
  geometry: GemmGeometry,
  init: (row: number) => string = () => "vec4<f32>(0.0)",
): string =>
  Array.from({ length: geometry.regM }, (_, row) =>
    Array.from(
      { length: gemmQuadsPerThread(geometry) },
      (_, quad) => `  var acc${row}_${quad} = ${init(row)};`,
    ).join("\n"),
  ).join("\n");

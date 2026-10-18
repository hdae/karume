/**
 * The **tile-geometry parameter surface** shared by the i8a8 GEMM family (linear and ①QK / ③PV of
 * fused attention).
 *
 * The 3 kernels do not share a generator (their differences are scattered in places that fragment
 * insertion cannot absorb — the docstring of src/kernels/linear-i8a8.ts), but **the geometry
 * arithmetic and the inner-product loop live only here, in one place**. Copying the geometry into
 * constants in each file would let the index arithmetic of `sa` / `sb` (pack stride, thread
 * assignment of the fill) diverge three ways, so that one of them breaks silently.
 *
 * ```
 * tileM = regM · wgY   (rows of output one workgroup holds)
 * tileN = regN · wgX   (columns                          )
 * threads = wgX · wgY  (one thread holds regM×regN outputs)
 * ```
 *
 * ## Why parameterising is allowed (relation to the numeric contract)
 *
 * i8a8 reduction is **exact i32 addition and order-independent**, so changing "which thread handles
 * which output" does not move the returned integers by a single bit (on an RTX 3080 Ti, more than 100
 * variants were compared byte by byte against the production kernel with 0 mismatches —
 * docs/research/2026-08-10-kernel-variant-sweep.md).
 * MUST: this freedom **belongs to the i8a8 path only**. For the f32 / f16 GEMM
 * (src/kernels/gemm.ts) the K reduction order is the numeric contract itself (ADR 0022), and the same
 * reasoning does not apply.
 * MUST: the geometry **goes into the pipeline key** ({@link i8a8GeometryKeyPart}). Without it the
 * determinism "same key → byte-identical WGSL" breaks (the same vocabulary as the v4 flag of
 * `gemmUsesVec4`).
 *
 * ## Where the default comes from
 *
 * {@link defaultI8a8Geometry} is the only selection point. In measurement (same research, §2 /
 * §3.2) linear and ①QK are won by `M128N64 r8×8 wg8×16 K16` (×1.39–1.47 on 3 shapes / QK ×1.92),
 * and only ③PV by `M64N128 r8×8 wg16×8 K16` (the shape where N = D = 128 fits in one tile — ×1.74).
 * The bottleneck is LDS bandwidth × issue; what matters is **first the register block
 * (regM·regN)**, second threads ≤ 128, third tileK.
 */

import { CodegenError } from "../codegen/errors.ts";
import { workgroupVariableBytes } from "./workgroup-storage.ts";

/** Packing i8 payload by 4 (one u32 word = 4 elements). Exactly the granularity of dp4a. */
export const I8A8_PACK = 4;

/**
 * The unit a geometry is chosen for. **Matches the op field of the key** (so that a reader of
 * diagnostics can map one to the other).
 */
type I8a8GemmOp = "linear" | "attention_qk" | "attention_pv";

/**
 * Generation parameters. The output tile edges are **derived** as `regM · wgY` × `regN · wgX`
 * (holding the tile edges independently would let 3 values disagree — derived, they cannot).
 */
export type I8a8Geometry = {
  /** Rows of output one thread holds. */
  readonly regM: number;
  /** Columns of output one thread holds (a multiple of 4, because `acc` is grouped into vec4). */
  readonly regN: number;
  /** x edge of the workgroup (column direction). */
  readonly wgX: number;
  /** y edge of the workgroup (row direction). */
  readonly wgY: number;
  /** K tile width (in elements — the pack count is `tileK / 4`). */
  readonly tileK: number;
};

/** Rows of the output tile. The dispatch's y is `ceil(m / tileM)`. */
export const i8a8TileM = (geometry: I8a8Geometry): number => geometry.regM * geometry.wgY;

/** Columns of the output tile. The dispatch's x is `ceil(n / tileN)`. */
export const i8a8TileN = (geometry: I8a8Geometry): number => geometry.regN * geometry.wgX;

/** Threads per workgroup. */
export const i8a8Threads = (geometry: I8a8Geometry): number => geometry.wgX * geometry.wgY;

/** Packs per K tile (the pack stride of the shared tile). */
export const i8a8KPacks = (geometry: I8a8Geometry): number => geometry.tileK / I8A8_PACK;

/**
 * The assignment stride of the shared-tile fill. One thread fills {@link i8a8ASlots} entries (on the
 * B side {@link i8a8BSlots}) with this stride, starting at row `tid / kPacks`.
 */
export const i8a8FillStride = (geometry: I8a8Geometry): number =>
  i8a8Threads(geometry) / i8a8KPacks(geometry);

/** Entries of the A (activation / q / P̃) tile one thread fills. */
export const i8a8ASlots = (geometry: I8a8Geometry): number =>
  i8a8TileM(geometry) / i8a8FillStride(geometry);

/** Entries of the B (weight / k / Vᵀ) tile one thread fills. */
export const i8a8BSlots = (geometry: I8a8Geometry): number =>
  i8a8TileN(geometry) / i8a8FillStride(geometry);

/** Bytes of one u32 word (the element type of the shared tiles sa / sb). */
const U32_BYTES = 4;

/**
 * Bytes of workgroup storage used by the shared tiles sa (`kPacks · tileM` words) + sb
 * (`kPacks · tileN` words) (the amount WebGPU compares with `maxComputeWorkgroupStorageSize` —
 * {@link workgroupVariableBytes}). linear and ①QK / ③PV of fused attention all have the same 2
 * declarations, so it does not depend on the op.
 * MUST: the same expression as the generator's `var<workgroup>` declarations
 * (tests/geometry_profile_test.ts compares it with the total counted from the WGSL declarations) —
 * if they disagree, the limit check (geometry-profile.ts) lets a table that exceeds the limit
 * through.
 */
export const i8a8WorkgroupStorageBytes = (geometry: I8a8Geometry): number =>
  workgroupVariableBytes(i8a8KPacks(geometry) * i8a8TileM(geometry) * U32_BYTES) +
  workgroupVariableBytes(i8a8KPacks(geometry) * i8a8TileN(geometry) * U32_BYTES);

/**
 * Rejects an inconsistent geometry at generation time. **A combination that does not divide evenly
 * leaves holes in the shared tile** (words the fill does not reach enter the inner product as 0 =
 * a wrong value with no exception), so failing loudly is the only guard.
 */
export const assertI8a8Geometry = (geometry: I8a8Geometry, where: string): void => {
  const { regM, regN, wgX, wgY, tileK } = geometry;
  for (const [name, value] of [
    ["regM", regM],
    ["regN", regN],
    ["wgX", wgX],
    ["wgY", wgY],
    ["tileK", tileK],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new CodegenError(`${where}: 幾何の ${name} は正整数（${value}）`);
    }
  }
  if (regN % I8A8_PACK !== 0) {
    throw new CodegenError(`${where}: regN は ${I8A8_PACK} の倍数（acc は vec4 束ね・${regN}）`);
  }
  if (tileK % I8A8_PACK !== 0) {
    throw new CodegenError(`${where}: tileK は ${I8A8_PACK} の倍数（i8 の 4 詰め・${tileK}）`);
  }
  const threads = i8a8Threads(geometry);
  const kPacks = i8a8KPacks(geometry);
  if (threads % kPacks !== 0) {
    throw new CodegenError(
      `${where}: スレッド数 ${threads} が K パック数 ${kPacks} で割り切れない（充填の担当が組めない）`,
    );
  }
  const stride = i8a8FillStride(geometry);
  if (i8a8TileM(geometry) % stride !== 0 || i8a8TileN(geometry) % stride !== 0) {
    throw new CodegenError(
      `${where}: タイル辺 ${i8a8TileM(geometry)}x${i8a8TileN(
        geometry,
      )} が充填ストライド ${stride} で割り切れない`,
    );
  }
};

/**
 * Consistency guard between the i4 group length and the K tile (a guard of **the w4a8 variant
 * only** — {@link "./linear-i8a8.ts"}).
 *
 * MUST: `groupSize % tileK == 0`. If one K tile crosses a group boundary, the integer inner product
 * of that tile mixes the weights of 2 groups, and the flush applies only one of the scales. Shape
 * and type stay consistent while only the values go wrong — a **silent wrong value** — so failing
 * loudly at generation time is the only guard.
 *
 * NOTE: the reverse (tileK larger than the group) is rejected by the same divisibility condition.
 */
export const assertI8a8GroupGeometry = (
  geometry: I8a8Geometry,
  groupSize: number,
  where: string,
): void => {
  if (groupSize % geometry.tileK !== 0) {
    throw new CodegenError(
      `${where}: group_size ${groupSize} が K タイル ${geometry.tileK} の倍数でない` +
        "（1 枚の K タイルが group 境界を跨ぐと scale が混ざる）",
    );
  }
};

/**
 * The geometry discriminator of the pipeline key. It keeps the style of `reg64x64r8x4w16v4`
 * (`gemmKeyPart` of the f32 skeleton, {@link "./gemm.ts"}) and adds what is needed now that **the
 * tile edges alone no longer determine the geometry** (`M128N64` can be built as `r8×8 wg8×16` or as
 * `r16×8 wg8×8`, so a key of tile edges alone would collide with a different kernel). The position
 * and spelling of the v4 flag stay at the end as before.
 */
export const i8a8GeometryKeyPart = (geometry: I8a8Geometry, v4: boolean): string =>
  `tile${i8a8TileM(geometry)}x${i8a8TileN(geometry)}r${geometry.regM}x${geometry.regN}` +
  `w${geometry.wgX}x${geometry.wgY}k${geometry.tileK}${v4 ? "v4" : ""}`;

/** Default for linear / ①QK (the only variant all 3 shapes picked as their best in measurement). */
const GEOMETRY_M128N64: I8a8Geometry = { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 };

/** Default for ③PV (the shape that covers N = D = 128 with one tile wins for both self and cross). */
const GEOMETRY_M64N128: I8a8Geometry = { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 };

/**
 * The **only pure function** that decides the default geometry.
 *
 * For now it is a constant per op, with no shape buckets — only on L2 (4096×8192×2048) does
 * `r16×8 wg8×8` win by a narrow margin (~6%), but adding one boundary doubles the key count and the
 * verification surface. The per-adapter tables made by manual tuning (sweeps) are held not by this
 * function but by the profile (the `i8a8` field of src/kernels/geometry-profile.ts — DECIDED: ADR
 * 0115). This function is the source of the default profile's values; the Session's derivation phase
 * only receives the geometry from the profile and passes it on.
 */
export const defaultI8a8Geometry = (op: I8a8GemmOp): I8a8Geometry =>
  op === "attention_pv" ? GEOMETRY_M64N128 : GEOMETRY_M128N64;

/**
 * Initialisation of the i32 accumulator. It uses **named variables `acc{row}_{column quad}` rather
 * than one array** because the dynamic index of `acc[i]` requires an addressable function-local
 * region, which does not stay in registers and spills to local memory (pronounced on Metal).
 * Unrolled, the reduction is still exact i32 addition and independent per output, so **the returned
 * integers do not change by a single bit**.
 */
export const i8a8AccumulatorInit = (geometry: I8a8Geometry): string => {
  const quads = geometry.regN / I8A8_PACK;
  return Array.from({ length: geometry.regM }, (_, row) =>
    Array.from({ length: quads }, (_, quad) => `  var acc${row}_${quad} = vec4<i32>(0);`).join(
      "\n",
    ),
  ).join("\n");
};

/**
 * The inner-product loop inside a K tile. It reads `regN` words of B and `regM` words of A from the
 * shared tiles (`sa` = `[pack][row]` / `sb` = `[pack][column]`) and runs `regM · regN` integer inner
 * products.
 *
 * MUST: the 3 kernels use this one loop (copying the geometry's index arithmetic would let one of
 * them get the pack stride wrong). The addition order per output is "K tiles ascending, 4 words of a
 * pack together"; whatever the geometry, it is exact i32 addition, so the result is
 * order-independent.
 */
export const i8a8InnerProductLoop = (geometry: I8a8Geometry): string => {
  const { regM, regN } = geometry;
  const quads = regN / I8A8_PACK;
  const kPacks = i8a8KPacks(geometry);
  const at = (offset: number): string => (offset === 0 ? "" : ` + ${offset}u`);
  const bLoads = Array.from(
    { length: regN },
    (_, col) => `      let b${col} = sb[bcol${at(col)}];`,
  ).join("\n");
  const rows = Array.from({ length: regM }, (_, row) => {
    const updates = Array.from({ length: quads }, (_, quad) => {
      const lanes = Array.from(
        { length: I8A8_PACK },
        (_, lane) => `idot(a${row}, b${quad * I8A8_PACK + lane})`,
      ).join(", ");
      return `      acc${row}_${quad} = acc${row}_${quad} + vec4<i32>(${lanes});`;
    }).join("\n");
    return `      let a${row} = sa[arow_at${at(row)}];\n${updates}`;
  }).join("\n");
  return `    // ${regN + regM} shared loads (${regN} words of B + ${regM} words of A) give ${
    regM * regN
  } integer inner products = ${regM * regN * I8A8_PACK} MACs.
    // The reduction is exact i32 addition, so it **does not depend on order** (unlike the f32
    // skeleton, the addition order is not part of the numeric contract)
    for (var p = 0u; p < ${kPacks}u; p = p + 1u) {
      let bcol = p * ${i8a8TileN(geometry)}u + lid.x * ${regN}u;
${bLoads}
      let arow_at = p * ${i8a8TileM(geometry)}u + lid.y * ${regM}u;
${rows}
    }`;
};

/** Geometry note for generated comments (in a form that fits on one header line). */
export const i8a8GeometryNote = (geometry: I8a8Geometry): string =>
  `タイル ${i8a8TileM(geometry)}x${i8a8TileN(
    geometry,
  )} / 1 スレッド ${geometry.regM}x${geometry.regN} / wg ${geometry.wgX}x${geometry.wgY} / K ${geometry.tileK}`;

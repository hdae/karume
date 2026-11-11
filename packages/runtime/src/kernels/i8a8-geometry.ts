/**
 * The **tile-geometry parameter surface** shared by the i8a8 GEMM family (linear, ①QK / ③PV of
 * fused attention, and the w8a8 implicit GEMM of conv3d).
 *
 * The kernels do not share a generator (their differences are scattered in places that fragment
 * insertion cannot absorb — the docstring of src/kernels/linear-i8a8.ts), but **the geometry
 * arithmetic and the inner-product loop live only here, in one place**. Copying the geometry into
 * constants in each file would let the index arithmetic of `sa` / `sb` (pack stride, thread
 * assignment of the fill) diverge between the kernels, so that one of them breaks silently.
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
type I8a8GemmOp = "linear" | "attention_qk" | "attention_pv" | "conv3d";

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
  /**
   * The skeleton form (perf-ledger K-85). Absent = the original form: a pack loop that runs at run
   * time, and a fill that loads one u32 (4 i8) of global memory per item. `"vec4"` = the pack loop
   * emitted unrolled (every pack offset a constant) and, in a consumer that implements it and on a
   * shape that allows it, a fill that loads one `vec4<u32>` (16 i8) of global memory per item
   * ({@link i8a8VecQuads}). Both forms return the same integers; the choice is speed only.
   * MUST: it is in the key ({@link i8a8GeometryKeyPart}) — it changes the WGSL.
   */
  readonly skeleton?: "vec4";
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
 * {@link workgroupVariableBytes}). linear, ①QK / ③PV of fused attention and conv3d all have the same 2
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
      throw new CodegenError(`${where}: geometry ${name} must be a positive integer (${value})`);
    }
  }
  if (regN % I8A8_PACK !== 0) {
    throw new CodegenError(
      `${where}: regN must be a multiple of ${I8A8_PACK} (acc is bundled as vec4, ${regN})`,
    );
  }
  if (tileK % I8A8_PACK !== 0) {
    throw new CodegenError(
      `${where}: tileK must be a multiple of ${I8A8_PACK} (four i8 packed per word, ${tileK})`,
    );
  }
  const threads = i8a8Threads(geometry);
  const kPacks = i8a8KPacks(geometry);
  if (threads % kPacks !== 0) {
    throw new CodegenError(
      `${where}: thread count ${threads} is not divisible by the K pack count ${kPacks} (the fill assignment cannot be built)`,
    );
  }
  const stride = i8a8FillStride(geometry);
  if (i8a8TileM(geometry) % stride !== 0 || i8a8TileN(geometry) % stride !== 0) {
    throw new CodegenError(
      `${where}: tile edges ${i8a8TileM(geometry)}x${i8a8TileN(
        geometry,
      )} are not divisible by the fill stride ${stride}`,
    );
  }
  const skeleton: unknown = geometry.skeleton;
  if (skeleton === undefined) return;
  if (skeleton !== "vec4") {
    throw new CodegenError(
      `${where}: geometry skeleton must be "vec4" or absent (${JSON.stringify(skeleton)})`,
    );
  }
  // The vec4 fill takes 4 packs per item: a K tile is whole vec4s, and the threads divide over them.
  if (kPacks % I8A8_PACK !== 0) {
    throw new CodegenError(
      `${where}: the vec4 skeleton needs tileK to be a multiple of ${I8A8_PACK * I8A8_PACK} (${tileK})`,
    );
  }
  if (threads % (kPacks / I8A8_PACK) !== 0) {
    throw new CodegenError(
      `${where}: thread count ${threads} is not divisible by the K quad count ${
        kPacks / I8A8_PACK
      } (the vec4 fill assignment cannot be built)`,
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
      `${where}: group_size ${groupSize} is not a multiple of the K tile ${geometry.tileK}` +
        " (a K tile that crosses a group boundary mixes scales)",
    );
  }
};

/**
 * The geometry discriminator of the pipeline key. It keeps the style of `reg64x64r8x4w16v4`
 * (`gemmKeyPart` of the f32 skeleton, {@link "./gemm.ts"}) and adds what is needed now that **the
 * tile edges alone no longer determine the geometry** (`M128N64` can be built as `r8×8 wg8×16` or as
 * `r16×8 wg8×8`, so a key of tile edges alone would collide with a different kernel). The position
 * and spelling of the v4 flag stay at the end as before. The vec4 skeleton adds `s` after the K
 * tile (`k32sv4`); a geometry without it keeps its key byte for byte.
 */
export const i8a8GeometryKeyPart = (geometry: I8a8Geometry, v4: boolean): string =>
  `tile${i8a8TileM(geometry)}x${i8a8TileN(geometry)}r${geometry.regM}x${geometry.regN}` +
  `w${geometry.wgX}x${geometry.wgY}k${geometry.tileK}${geometry.skeleton === undefined ? "" : "s"}${
    v4 ? "v4" : ""
  }`;

/**
 * Default for linear / ①QK (the only variant all 3 shapes picked as their best in measurement) and
 * for conv3d (the value conv3d ran with before the profile had a field for it — ADR 0133 decision 5,
 * ADR 0115 addendum 2026-10-26).
 */
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
 * MUST: the kernels use this one loop (copying the geometry's index arithmetic would let one of
 * them get the pack stride wrong). The addition order per output is "K tiles ascending, 4 words of a
 * pack together"; whatever the geometry, it is exact i32 addition, so the result is
 * order-independent.
 *
 * With the vec4 skeleton the loop over the packs is emitted unrolled: one block per pack, every
 * pack offset a constant. The body and the addition order per output are the loop's.
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
  if (geometry.skeleton === "vec4") {
    return Array.from(
      { length: kPacks },
      (_, pack) => `    {
      let bcol = ${pack * i8a8TileN(geometry)}u + lid.x * ${regN}u;
${bLoads}
      let arow_at = ${pack * i8a8TileM(geometry)}u + lid.y * ${regM}u;
${rows}
    }`,
    ).join("\n");
  }
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
  `tile ${i8a8TileM(geometry)}x${i8a8TileN(
    geometry,
  )} / ${geometry.regM}x${geometry.regN} per thread / wg ${geometry.wgX}x${geometry.wgY} / K ${geometry.tileK}${
    geometry.skeleton === undefined ? "" : " / vec4 skeleton"
  }`;

// ---------------------------------------------------------------------------------------------
// The vec4 fill of the vec4 skeleton (perf-ledger K-85), shared by every consumer that has one.
//
// One fill item = one row (or column) x 4 consecutive packs = 16 i8 = one `vec4<u32>` of global
// memory. A K tile has Q = kPacks / 4 items per row. A thread takes the items tid, tid + T, …
// (T = threads): T % Q == 0 ({@link assertI8a8Geometry}), so its pack quad is tid % Q in every slot
// and its row advances by T / Q per slot.
//
// MUST: the shared tiles stay `array<u32>` and each loaded vec4 is stored as four whole-word
// stores with static components (`.x .y .z .w`). No workgroup vec4 is written through a dynamic
// component (the MUST of gemm.ts on shared tile stores).
// ---------------------------------------------------------------------------------------------

/** Pack quads per row per K tile (vec4 fill). */
export const i8a8VecQuads = (geometry: I8a8Geometry): number => i8a8KPacks(geometry) / I8A8_PACK;

/** The row step between two slots of one thread (vec4 fill). */
export const i8a8VecStride = (geometry: I8a8Geometry): number =>
  i8a8Threads(geometry) / i8a8VecQuads(geometry);

/** Slots one thread fills for a tile edge of `edge` rows (vec4 fill); the last may be partial. */
export const i8a8VecSlots = (geometry: I8a8Geometry, edge: number): number =>
  Math.ceil(edge / i8a8VecStride(geometry));

/** The row offset of each slot (slot · stride). */
export const i8a8VecSlotOffsets = (geometry: I8a8Geometry, edge: number): number[] =>
  Array.from({ length: i8a8VecSlots(geometry, edge) }, (_, slot) => slot * i8a8VecStride(geometry));

/** Whether slot `slot` of an edge of `edge` rows reaches past the tile (it needs a row guard). */
const i8a8VecSlotPartial = (geometry: I8a8Geometry, edge: number, slot: number): boolean =>
  (slot + 1) * i8a8VecStride(geometry) > edge;

const VEC_LANES = ["x", "y", "z", "w"] as const;

/** The thread's assignment, invariant over the K tiles: its quad, its first row, its tile offset. */
export const i8a8VecAssign = (
  geometry: I8a8Geometry,
  edge: number,
  names: { readonly quad: string; readonly row: string; readonly at: string },
): string => {
  const quads = i8a8VecQuads(geometry);
  return `  let ${names.quad} = tid % ${quads}u;
  let ${names.row} = tid / ${quads}u;
  let ${names.at} = ${names.quad} * ${I8A8_PACK * edge}u + ${names.row};`;
};

/**
 * One slot of the fill: `value` (a `vec4<u32>` already loaded, or zero) stored as four words of
 * the `[pack][row]` tile. A partial slot stores only inside the tile.
 */
export const i8a8VecStoreSlot = (
  geometry: I8a8Geometry,
  edge: number,
  slot: number,
  names: {
    readonly tile: string;
    readonly at: string;
    readonly row: string;
    readonly value: string;
  },
  indent: string,
): string => {
  const offset = slot * i8a8VecStride(geometry);
  const plus = (extra: number): string => (extra === 0 ? "" : ` + ${extra}u`);
  const stores = VEC_LANES.map(
    (lane, pack) =>
      `${indent}${names.tile}[${names.at}${plus(offset + pack * edge)}] = ${names.value}.${lane};`,
  ).join("\n");
  return i8a8VecSlotPartial(geometry, edge, slot)
    ? `${indent}if (${names.row} + ${offset}u < ${edge}u) {\n${stores.replace(/^/gm, "  ")}\n${indent}}`
    : stores;
};

/** The in-tile row condition a partial slot's load needs (` && …`), or "". */
export const i8a8VecLoadGuard = (
  geometry: I8a8Geometry,
  edge: number,
  slot: number,
  row: string,
): string =>
  i8a8VecSlotPartial(geometry, edge, slot)
    ? ` && ${row} + ${slot * i8a8VecStride(geometry)}u < ${edge}u`
    : "";

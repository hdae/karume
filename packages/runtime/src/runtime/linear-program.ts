/**
 * A linear strategy's dispatches as data.
 *
 * Every strategy of `linear` is a {@link LinearProgram}: which kernels run, with which uniform, over
 * which bindings and on which grid, and which temporaries live between them — said without a device.
 * {@link linearProgramFor} maps the strategy the selection chose (src/runtime/linear-strategies.ts)
 * and the linear's facts to its program, through one constructor per kernel family below. The recipe
 * builder turns a program into dispatches with one generic routine (`emitLinearProgram` in
 * recipe-builders/linear.ts): it fetches each pipeline from the Session's cache, writes the uniforms,
 * binds the weight's scale buffer where its storage has one and checks the grids against the device
 * limit. Anything that needs to know a strategy's WGSL or temporaries before running — the memory
 * estimate, the verification cache — can read the same value (ADR 0152 decisions 6 and 8); neither
 * reads it yet.
 *
 * NOTE: device-free (kernel modules only). Not part of the public surface of mod.ts (ADR 0008).
 */

import type { GemmCompute } from "../kernels/gemm.ts";
import { gemmTileM, gemmTileN } from "../kernels/gemm-geometry.ts";
import {
  gemmRowsGeometry,
  type GeometryProfile,
  i8a8LinearGeometry,
} from "../kernels/geometry-profile.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8TileM,
  i8a8TileN,
} from "../kernels/i8a8-geometry.ts";
import { LINEAR_SCALE_BINDING, linearKey, linearParams, linearWgsl } from "../kernels/linear.ts";
import {
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvRowsKey,
  linearGemvRowsWgsl,
  linearGemvSubgroupKey,
  linearGemvSubgroupWgsl,
  linearGemvWgsl,
} from "../kernels/linear-gemv.ts";
import {
  integerGemvKey,
  integerGemvParams,
  type IntegerGemvVariant,
  integerGemvWgsl,
  integerGemvWorkgroups,
} from "../kernels/linear-gemv-integer.ts";
import {
  planesGemvKey,
  planesGemvParams,
  type PlanesGemvVariant,
  planesGemvWgsl,
  planesGemvWorkgroups,
} from "../kernels/linear-gemv-planes.ts";
import {
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8TempBytes,
  linearI8a8UsesVec4,
  linearI8a8UsesVecFill,
  linearI8a8Wgsl,
} from "../kernels/linear-i8a8.ts";
import {
  LINEAR_SPLITK_COMBINE_GROUP,
  LINEAR_SPLITK_TILE,
  linearSplitKCombineKey,
  linearSplitKCombineWgsl,
  linearSplitKParams,
  linearSplitKPartialKey,
  linearSplitKPartialsBytes,
  linearSplitKPartialWgsl,
  type LinearSplitKSlices,
} from "../kernels/linear-splitk.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsPrologueKey,
  quantizeRowsPrologueWgsl,
  quantizeRowsWgsl,
} from "../kernels/quantize-rows.ts";
import type { WeightStorage } from "../kernels/weight-storage.ts";
import type { LinearPrologue } from "./fusion-rule.ts";
import type { LinearChoice, LinearGemvChoice } from "./linear-strategies.ts";
import { ExecutionError } from "./plan.ts";
import type { I8a8Dot } from "./session-types.ts";

/**
 * What a binding points at: the linear's operands, the scale buffer of its weight, or a temporary of
 * the program by its index.
 *
 * `weightScale` is bound only where the weight's resident storage has a scale buffer (i2 / i4 / i8);
 * a float storage (f32 / f16 / bf16) has none, and the binding is then left out — the WGSL declares
 * `wscale` on the same condition.
 */
export type ProgramOperand =
  | "x"
  | "weight"
  | "bias"
  | "out"
  | "weightScale"
  | { readonly temp: number };

/** One entry of a dispatch's bind group after the uniform. */
export type ProgramBinding = { readonly binding: number; readonly operand: ProgramOperand };

/** One axis of a tiled grid: `ceil(items / tile)` workgroups, failing loudly above the device limit. */
export type TiledAxis = { readonly items: number; readonly tile: number };

export type ProgramGrid =
  /** One workgroup = one tile on each axis; nothing falls back, so exceeding the limit is an error. */
  | { readonly kind: "tiled"; readonly axes: readonly [TiledAxis, TiledAxis, TiledAxis] }
  /**
   * A grid-stride kernel over `items` items, `workgroupSize` of them per workgroup and pass (the
   * split-K combine: elements, one per thread; the i8a8 quantization: rows).
   */
  | { readonly kind: "gridStride"; readonly items: number; readonly workgroupSize: number };

export type ProgramDispatch = {
  readonly key: string;
  /** The WGSL of the key (generated on demand — the pipeline cache holds it after the first use). */
  readonly wgsl: () => string;
  /** The uniform at binding 0. */
  readonly params: Uint32Array<ArrayBuffer>;
  /** The bindings after the uniform, in bind-group order. MUST: each number is the WGSL's. */
  readonly bindings: readonly ProgramBinding[];
  readonly grid: ProgramGrid;
};

export type LinearProgram = {
  /** The head of a failure's message (`linear split-K [x] × [weight]`). */
  readonly where: string;
  /**
   * The byte length of each temporary. A temporary lives from the first dispatch to the last:
   * allocated in this order before the first, released in the reverse order after the last.
   */
  readonly temps: readonly number[];
  readonly dispatches: readonly ProgramDispatch[];
};

/** What a linear's program is made from besides its strategy. */
export type LinearProgramFacts = {
  /** The shape of the activation (`[…, k]`) — for the heads of failure messages. */
  readonly x: readonly number[];
  /** The shape of the weight (`[n, k]`) — for the heads of failure messages. */
  readonly weight: readonly number[];
  /** The flattened row count. */
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** The resident storage of the weight. */
  readonly weightStorage: WeightStorage;
  /** The i4 group length of the weight (`undefined` for any other storage). */
  readonly groupSize: number | undefined;
  /**
   * The producer SRQ's scale of the packed-activation pairing the fusion pass declared for this
   * linear (ADR 0105), `undefined` without one.
   */
  readonly packedScale: number | undefined;
  /** The producer folded into step ① of the i8a8 path (perf-ledger F-4), if any. */
  readonly prologue: LinearPrologue | undefined;
  /** The Session's geometry profile (src/kernels/geometry-profile.ts). */
  readonly geometryProfile: GeometryProfile;
  /** The Session's integer dot-product variant of the i8a8 path. */
  readonly linearI8a8Dot: I8a8Dot;
};

/**
 * The bindings of the single-dispatch linear kernels (the GEMM skeleton and the GEMV family): 1 x /
 * 2 weight / 3 bias / 4 out, then the weight's scale at {@link LINEAR_SCALE_BINDING} where its storage
 * has one. A linear always binds its bias (arity 3).
 */
const LINEAR_BINDINGS: readonly ProgramBinding[] = [
  { binding: 1, operand: "x" },
  { binding: 2, operand: "weight" },
  { binding: 3, operand: "bias" },
  { binding: 4, operand: "out" },
  { binding: LINEAR_SCALE_BINDING, operand: "weightScale" },
];

/** One workgroup on the axis (the z axis of the kernels whose grid is two-dimensional). */
const SINGLE: TiledAxis = { items: 1, tile: 1 };

/**
 * The program of one linear: the strategy {@link LinearChoice} names, made from the linear's facts.
 * The selection (`selectLinearStrategy`) decides which strategy; this function only builds it.
 */
export const linearProgramFor = (
  choice: LinearChoice,
  facts: LinearProgramFacts,
): LinearProgram => {
  const { x, weight, m, n, k } = facts;
  const operands = `[${x.join(",")}] × [${weight.join(",")}]`;
  switch (choice.id) {
    case "integer-gemv":
      return linearIntegerGemvProgram(
        `linear gemv integer ${operands}`,
        choice.variant,
        m,
        n,
        k,
        packedScaleOf(facts.packedScale, choice.id),
      );
    case "i8a8":
      return linearI8a8Program(
        `linear i8a8 ${operands}`,
        m,
        n,
        k,
        facts.weightStorage,
        facts.groupSize,
        facts.prologue,
        facts.geometryProfile,
        facts.linearI8a8Dot,
      );
    case "planes-gemv":
      return linearPlanesGemvProgram(`linear gemv planes ${operands}`, choice.variant, m, n, k);
    case "gemm-splitk":
      return linearSplitKProgram(`linear split-K ${operands}`, m, n, k, choice.slices);
    case "gemm-f32":
    case "gemm-f16":
      return linearGemmProgram(
        `linear ${operands}`,
        m,
        n,
        k,
        facts.weightStorage,
        choice.v4,
        choice.compute,
        choice.groupSize,
        facts.geometryProfile,
      );
    default:
      return linearGemvProgram(
        `linear gemv ${operands}`,
        choice,
        m,
        n,
        k,
        choice.id === "gemv-parallel-packed"
          ? packedScaleOf(facts.packedScale, choice.id)
          : undefined,
      );
  }
};

/**
 * The producer SRQ's scale of a strategy that reads packed activations. The selection takes such a
 * strategy only for a linear the pairing was declared for, so a missing pairing is a broken plan.
 */
const packedScaleOf = (packedScale: number | undefined, strategy: string): number => {
  if (packedScale === undefined) {
    throw new ExecutionError(
      `linear: the strategy ${strategy} was chosen without a packed pairing`,
    );
  }
  return packedScale;
};

/**
 * The default GEMM skeleton (`gemm-f32`, and `gemm-f16` under f16 compute — src/kernels/linear.ts):
 * the 2D GEMM `[m,k] × [k,n] + bias`, reading the weight in its transposed `[n,k]` layout as is (no
 * transposing copy). One workgroup = one output tile, so exceeding the limit fails loudly.
 */
const linearGemmProgram = (
  where: string,
  m: number,
  n: number,
  k: number,
  weightStorage: WeightStorage,
  v4: boolean,
  compute: GemmCompute,
  groupSize: number | undefined,
  profile: GeometryProfile,
): LinearProgram => {
  // MUST: the tile geometry is the bucket of the flattened row count m (the adapter's profile —
  // src/kernels/geometry-profile.ts). Pass the geometry taken once **unchanged** to key, WGSL and
  // dispatch.
  const geometry = gemmRowsGeometry(profile, m);
  return {
    where,
    temps: [],
    dispatches: [
      {
        key: linearKey(weightStorage, v4, compute, m, groupSize, geometry),
        wgsl: () => linearWgsl(weightStorage, v4, compute, m, groupSize, geometry),
        params: linearParams(m, n, k),
        bindings: LINEAR_BINDINGS,
        grid: {
          kind: "tiled",
          axes: [
            { items: n, tile: gemmTileN(geometry) },
            { items: m, tile: gemmTileM(geometry) },
            SINGLE,
          ],
        },
      },
    ],
  };
};

/**
 * linear's **GEMV family** (1 ≤ M ≤ LINEAR_GEMV_MAX_ROWS × i2 / i4 / i8 weights, and M=1 ×
 * f16 / bf16 / f32 weights — ADR 0082 (f16 / f32 in addenda 6 and 7); i2 is ADR 0097).
 *
 * Bindings, uniform and output storage are the same as the default path; only "which thread owns
 * which output" changes (one thread = one output column (× row block for M ≥ 2), no shared tile
 * and no barrier). The exception is the parallel GEMV that linear at the level `reorder` resolves
 * to (`linearGemvReduce: parallel` — ADR 0098), which splits K for the eligible shapes.
 * The K reduction order per output element stays sequential in ascending k, so it is
 * **bit-identical** (the numeric contract of src/kernels/linear-gemv.ts).
 * Which member runs (the M = 1 column form, the row-block form whose row count is part of the key,
 * the parallel forms) was decided by `selectLinearStrategy`; this function turns that choice into
 * the key, the WGSL, the uniform and the grid.
 * MUST: `groupSize` is set only for i4 (i8 has no groups — the kernel checks the pairing).
 * MUST: one thread, one output (column × row block), so the grid is
 * `[ceil(n / cols), ceil(m / rows), 1]`.
 * It is not grid-stride, so exceeding the limit fails loudly (the same discipline as the default
 * path).
 */
const linearGemvProgram = (
  where: string,
  /** The member of the family `selectLinearStrategy` chose, with its lanes and row form. */
  choice: LinearGemvChoice,
  m: number,
  n: number,
  k: number,
  /** The producer SRQ's scale (only for the packed form — ADR 0105). */
  packedScale: number | undefined,
): LinearProgram => {
  const { storage, groupSize, lanes, variant, rowsVariant } = choice;
  const subgroup = choice.id === "gemv-parallel-subgroup32";
  const staged = choice.id === "gemv-parallel-staged";
  const rows = rowsVariant?.rows ?? 1;
  const key =
    lanes !== undefined
      ? subgroup
        ? linearGemvSubgroupKey(storage, groupSize, lanes)
        : packedScale !== undefined
          ? linearGemvParallelPackedKey(storage, groupSize, lanes)
          : staged
            ? linearGemvParallelStagedKey(storage, groupSize, lanes)
            : linearGemvParallelKey(storage, groupSize, lanes)
      : rowsVariant === undefined
        ? linearGemvKey(storage, groupSize, variant)
        : linearGemvRowsKey(storage, groupSize, rowsVariant);
  const wgsl = (): string =>
    lanes !== undefined
      ? subgroup
        ? linearGemvSubgroupWgsl(storage, groupSize, lanes)
        : packedScale !== undefined
          ? linearGemvParallelPackedWgsl(storage, groupSize, lanes)
          : staged
            ? linearGemvParallelStagedWgsl(storage, groupSize, lanes)
            : linearGemvParallelWgsl(storage, groupSize, lanes)
      : rowsVariant === undefined
        ? linearGemvWgsl(storage, groupSize, variant)
        : linearGemvRowsWgsl(storage, groupSize, rowsVariant);
  return {
    where,
    temps: [],
    dispatches: [
      {
        key,
        wgsl,
        // The uniform is the same three words as the default path (no separate binding layout). The
        // divisibility check lives in one place on the family side. Only the packed variant appends
        // one word, the activation scale, to Dims.
        params:
          packedScale === undefined
            ? linearGemvParams(storage, m, n, k, groupSize)
            : linearGemvParallelPackedParams(storage, m, n, k, packedScale, groupSize),
        bindings: LINEAR_BINDINGS,
        grid: {
          kind: "tiled",
          axes: [
            { items: n, tile: lanes === undefined ? variant.cols : 128 / lanes },
            { items: m, tile: rows },
            SINGLE,
          ],
        },
      },
    ],
  };
};

/**
 * linear's **integer GEMV** (`SessionOptions.integerGemv` — src/kernels/linear-gemv-integer.ts): an
 * integer GEMV linear whose activation arrives as packed int8 codes. The output stays f32 (the
 * linear→SRQ fusion builds the SRQ-epilogue form in its own rule).
 *
 * The bindings keep the linear's numbering, as the parallel GEMV's do (1 x = the packed codes / 2 w =
 * the weight units / 3 bias / 4 out / 5 the weight scales, per channel and for i4 per group); a
 * linear always binds its bias (arity 3).
 * The activation scale is the producer SRQ's, carried by the pairing (`packed.scale`).
 * MUST: built only for the selection's `integer-gemv` (the activation is packed and the variant came
 * from `linearIntegerGemvVariant`). The grid is `(workgroups(n), m)`, one workgroup per tile, so
 * exceeding the limit fails loudly.
 */
const linearIntegerGemvProgram = (
  where: string,
  variant: IntegerGemvVariant,
  m: number,
  n: number,
  k: number,
  xScale: number,
): LinearProgram => ({
  where,
  temps: [],
  dispatches: [
    {
      key: integerGemvKey(variant, false),
      wgsl: () => integerGemvWgsl(variant, false),
      params: integerGemvParams(variant, m, n, k, xScale),
      bindings: LINEAR_BINDINGS,
      grid: {
        kind: "tiled",
        axes: [
          { items: integerGemvWorkgroups(n, variant), tile: 1 },
          { items: m, tile: 1 },
          SINGLE,
        ],
      },
    },
  ],
});

/**
 * linear's **byte-plane GEMV** (`REORDER_STRATEGIES.linearGemvPlanes` —
 * src/kernels/linear-gemv-planes.ts): an i2 linear of a measured shape whose activation arrives as
 * f32.
 *
 * The bindings keep the linear's numbering (1 x / 2 w = the i2 words / 3 bias / 4 out / 5 the
 * per-channel weight scale); a linear always binds its bias (arity 3).
 * MUST: built only for the selection's `planes-gemv` (the activation is f32 and the variant came from
 * `linearPlanesGemvVariant`). The grid is `(workgroups(n), m)`, one workgroup per tile, so exceeding
 * the limit fails loudly.
 */
const linearPlanesGemvProgram = (
  where: string,
  variant: PlanesGemvVariant,
  m: number,
  n: number,
  k: number,
): LinearProgram => ({
  where,
  temps: [],
  dispatches: [
    {
      key: planesGemvKey(variant),
      wgsl: () => planesGemvWgsl(variant),
      params: planesGemvParams(m, n, k),
      bindings: LINEAR_BINDINGS,
      grid: {
        kind: "tiled",
        axes: [{ items: planesGemvWorkgroups(n, variant), tile: 1 }, { items: m, tile: 1 }, SINGLE],
      },
    },
  ],
});

/**
 * The kernel of the i8a8 path's step ①: the plain per-token quantization (one row = one workgroup,
 * or the small-row geometry), or — when the fusion pass folded a producer into this linear — the
 * prologue form, which keeps one row per workgroup in registers (perf-ledger F-4).
 */
const quantizeStep = (
  k: number,
  prologue: LinearPrologue | undefined,
): { readonly key: string; readonly wgsl: () => string; readonly rowsPerGroup: number } => {
  if (prologue !== undefined) {
    return {
      key: quantizeRowsPrologueKey(prologue.op, k),
      wgsl: () => quantizeRowsPrologueWgsl(prologue.op, k),
      rowsPerGroup: 1,
    };
  }
  const geometry = quantizeRowsGeometry(k);
  return {
    key: quantizeRowsKey(geometry),
    wgsl: () => quantizeRowsWgsl(geometry),
    rowsPerGroup: geometry.rowsPerGroup,
  };
};

/**
 * The geometry of the i8a8 path's GEMM (②) for a linear of `m` rows: the profile's `i8a8.linear`
 * rule for `m` (`i8a8LinearGeometry`), or — for i4 resident weights whose group length is not a
 * multiple of that rule's K tile — the default geometry `defaultI8a8Geometry("linear")`, whose K
 * tile 16 divides every i4 group length (a power of two of at least 16). The w4a8 kernel refuses a
 * K tile that crosses a group boundary (`assertI8a8GroupGeometry`), and a profile field applies to
 * i8 and i4 weights alike, so a K64 rule would otherwise make w4a8 at group 32 unbuildable (ADR 0115
 * addendum decision 14).
 *
 * MUST: called once per linear, and the one value it returns feeds the key, the WGSL, the dispatch
 * grid and `linearI8a8UsesVecFill` — a second lookup could disagree with the first and build a grid
 * that does not match the WGSL.
 */
export const linearI8a8Geometry = (
  profile: GeometryProfile,
  m: number,
  weightStorage: WeightStorage,
  groupSize: number | undefined,
): I8a8Geometry => {
  const selected = i8a8LinearGeometry(profile, m);
  return weightStorage === "i4" && groupSize !== undefined && groupSize % selected.tileK !== 0
    ? defaultI8a8Geometry("linear")
    : selected;
};

/**
 * linear's **w8a8 / w4a8 variant** (opt-in — `SessionOptions.linearCompute`).
 * **One node = two dispatches** (the same "one node in several dispatches" treatment as fused
 * attention):
 *
 * ① `quantize_rows` (activations to per-token i8, grid-stride over rows) → ② the integer
 * dot-product GEMM (one workgroup = one output tile, so exceeding the limit fails loudly). ① is the
 * same whatever the weights' storage form. With a {@link LinearPrologue} (the producer folded by the
 * fusion pass — perf-ledger F-4), ① quantizes `op(x)` of the re-pointed input instead (same bindings
 * and params, and the same values as "elementwise op → quantize_rows").
 *
 * MUST: the temporaries (`xq` / `xs`) live inside the node: allocated before ① and released after
 * ②. That closes the plan's reference counting, and even on a failure path `arena.destroy()`
 * collects the whole region.
 * MUST: the i32 overflow guard fails loudly. Letting it pass would garble values, sign included,
 * through i32 wrap-around.
 * The guard's axis depends on the storage form — for i8 it is **k** (the whole reduction is one
 * i32), for i4 **the group length** (flushing is per group, so only one group's worth sits in an
 * i32).
 */
const linearI8a8Program = (
  where: string,
  m: number,
  n: number,
  k: number,
  weightStorage: WeightStorage,
  /** For i4 the group length affects both the key and the WGSL (the baked-in shift) (ADR 0069). */
  groupSize: number | undefined,
  prologue: LinearPrologue | undefined,
  profile: GeometryProfile,
  dot: I8a8Dot,
): LinearProgram => {
  if (groupSize === undefined && k > LINEAR_I8A8_MAX_K) {
    throw new ExecutionError(
      `${where}: k=${k} exceeds the i32 reduction guard ${LINEAR_I8A8_MAX_K} of the i8a8 path` +
        " (set linearCompute to 'f32', or take this linear off i8 residency)",
    );
  }
  // The quantized activations `xq` (four i8 packed per word) and the per-token scale `xs`.
  // Temporary regions closed inside the node (the bytes from the function the estimate shares).
  const bytes = linearI8a8TempBytes(m, k);
  const xq = { temp: 0 } as const;
  const xs = { temp: 1 } as const;

  // ① Per-token quantization of the activations (grid-stride over rows). With a prologue the same
  // quantization of `op(x)`, `x` being the folded producer's input; the bindings and params are the
  // plain form's.
  const quantize = quantizeStep(k, prologue);
  const quantizeParams = quantizeRowsParams(m, k);

  // ② The integer dot-product GEMM. The tile geometry is the adapter's profile's `i8a8.linear` rule
  // for m (src/kernels/geometry-profile.ts), with the w4a8 group fallback — taken once and passed
  // unchanged to key, WGSL, grid and the fill form; it is part of the key, so "same key →
  // byte-identical WGSL" holds.
  const v4 = linearI8a8UsesVec4(n);
  const geometry = linearI8a8Geometry(profile, m, weightStorage, groupSize);
  const dp4a = dot === "dp4a";
  // MUST: look up both key and wgsl with **the actual resident form** (`weightStorage`). Fixing it
  // to i8 would hand out i4's group scales as per-channel scales, a silent wrong value (the numeric
  // contracts differ — ADR 0076).
  // The fill form of the vec4 skeleton (a geometry without the skeleton: always the word fill).
  const vecFill = linearI8a8UsesVecFill(geometry, weightStorage, k);
  const key = linearI8a8Key(v4, dp4a, geometry, weightStorage, groupSize, vecFill);
  return {
    where,
    temps: [bytes.xq, bytes.xs],
    dispatches: [
      {
        key: quantize.key,
        wgsl: quantize.wgsl,
        params: quantizeParams,
        bindings: [
          { binding: 1, operand: "x" },
          { binding: 2, operand: xq },
          { binding: 3, operand: xs },
        ],
        grid: { kind: "gridStride", items: m, workgroupSize: quantize.rowsPerGroup },
      },
      {
        key,
        wgsl: () => linearI8a8Wgsl(v4, dp4a, geometry, weightStorage, groupSize, vecFill),
        params: linearI8a8Params(m, n, k, groupSize),
        bindings: [
          { binding: 1, operand: xq },
          { binding: 2, operand: "weight" },
          { binding: 3, operand: "bias" },
          { binding: 4, operand: "out" },
          // Only **whether** it is bound depends on the resident form (i8 / i4) — the buffer itself
          // is looked up by initializer name, so it is the same for i8 / i4. The **interpretation**
          // of the scale (per-channel or group) is decided by the discriminator of `linearI8a8Wgsl` /
          // `linearI8a8Key` above, which is the real guard against silent wrong values.
          { binding: LINEAR_SCALE_BINDING, operand: "weightScale" },
          { binding: LINEAR_ACT_SCALE_BINDING, operand: xs },
        ],
        grid: {
          kind: "tiled",
          axes: [
            { items: n, tile: i8a8TileN(geometry) },
            { items: m, tile: i8a8TileM(geometry) },
            SINGLE,
          ],
        },
      },
    ],
  };
};

/**
 * The split-K linear (src/kernels/linear-splitk.ts): the partial sums of every K slice into one
 * temporary, then their sum and the bias into the output.
 */
const linearSplitKProgram = (
  where: string,
  m: number,
  n: number,
  k: number,
  slices: LinearSplitKSlices,
): LinearProgram => {
  const params = linearSplitKParams(m, n, k, slices);
  return {
    where,
    temps: [linearSplitKPartialsBytes(m, n, slices)],
    dispatches: [
      {
        key: linearSplitKPartialKey(),
        wgsl: linearSplitKPartialWgsl,
        params,
        bindings: [
          { binding: 1, operand: "x" },
          { binding: 2, operand: "weight" },
          { binding: 3, operand: { temp: 0 } },
        ],
        grid: {
          kind: "tiled",
          axes: [
            { items: n, tile: LINEAR_SPLITK_TILE },
            { items: m, tile: LINEAR_SPLITK_TILE },
            { items: slices, tile: 1 },
          ],
        },
      },
      {
        key: linearSplitKCombineKey(),
        wgsl: linearSplitKCombineWgsl,
        params,
        bindings: [
          { binding: 1, operand: { temp: 0 } },
          { binding: 2, operand: "bias" },
          { binding: 3, operand: "out" },
        ],
        grid: { kind: "gridStride", items: m * n, workgroupSize: LINEAR_SPLITK_COMBINE_GROUP },
      },
    ],
  };
};

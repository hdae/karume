/**
 * The implicit GEMM kernel of conv3d (unbatched
 * `x[Cin,T,H,W] * W[Cout,Cin/groups,Kt,Kh,Kw] + b[Cout]`, f32 — ADR 0118 decision 1).
 *
 * | Path          | Key                                                        | Condition   |
 * | ------------- | ---------------------------------------------------------- | ----------- |
 * | implicit GEMM | `conv3d:v1:f32:igemm{tileM}x{tileN}{v4}:wg{x}x{y}{:w…}`    | groups == 1 |
 *
 * Solves `C[Cout, N] = W[Cout, K] × Xcol[K, N]` by sharing fragments of the GEMM skeleton
 * (src/kernels/gemm.ts). `M = Cout`, `N = Tout·Hout·Wout` (**the time axis is folded into N**),
 * `K = Cin·Kt·Kh·Kw`. Flattening the weight `[Cout, Cin, Kt, Kh, Kw]` is the A tile `[M, K]` as
 * is, and the row-major output `[Cout][Tout·Hout·Wout]` is the unbatched output tensor as is (zero
 * reshapes — the same argument as ADR 0024 decision 1).
 *
 * The basis of bit identity is inherited from ADR 0024 decision 3 unchanged: the reduction runs
 * over flat K in ascending order (= fourfold ascending `(ic, kt, kh, kw)`), K tiles of 16, the bias
 * as the initial `acc`, out-of-range x as 0. With Kt = 1 this order becomes the same sequence as
 * conv2d's `(ic, kh, kw)`, so "conv3d with Kt = 1 ≡ conv2d per frame" matches in Uint32 (identity
 * check ① of tests/gpu_conv3d_parity_test.ts).
 *
 * ## Implemented subset (ADR 0064 axis A — the CPU reference holds the full semantics)
 *
 * - **groups == 1 only**. groups > 1 fails loudly at Session construction (src/runtime/plan.ts)
 *   and is rejected again in the params layer ({@link conv3dIgemmParams}). There is no direct
 *   kernel — one is added when demand appears (depthwise and the like) (the generalisation
 *   condition of ADR 0118 decision 1).
 * - Weight storage is f32 / f16 / i8 (the A-tile fill is shared with 1D / 2D, so i8's per-channel
 *   scale is read per row = output channel — ADR 0024 decision 6). The generator entry point
 *   (gemm.ts) rejects i4 / i2.
 * - **No geometry profile is consulted**. It runs on the default geometry
 *   (`gemmMTileGeometry(64 / 32)`) (ADR 0118 decision 1 — the same treatment as conv1d. The conv2d
 *   column's geometry is not applied to 3D without sweep cases).
 *
 * MUST: the dispatch is `[ceil(N/tileN), ceil(M/tileM), 1]`, and shapes whose N tile count exceeds
 * the limit raise `DispatchLimitError` (no fallback via grid-stride — a missing tile would be a
 * silent wrong value).
 */

import { tiledWorkgroups } from "../codegen/dispatch.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { assertConvAxisExtent, assertConvFlatExtents } from "./conv-bounds.ts";
import { gemmMTileGeometry, gemmWgsl, LINEAR_SCALE_BINDING } from "./gemm.ts";
import { GEMM_TILE, gemmTileM, gemmTileN } from "./gemm-geometry.ts";
import { weightKeyPart, type WeightStorage } from "./weight-storage.ts";

/**
 * Scale binding of the i8 variant (the number after the output — paired with the executor's bind
 * entries). The binding layout is the same as the implicit GEMM of linear / conv1d / conv2d (0
 * dims / 1 x / 2 weight / 3 bias / 4 output).
 */
export const CONV3D_SCALE_BINDING = LINEAR_SCALE_BINDING;

/** Geometry of conv3d (the one input type of the params function). */
export type Conv3dDims = {
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly timeIn: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly timeOut: number;
  readonly heightOut: number;
  readonly widthOut: number;
  readonly kernelT: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly strideT: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly paddingT: number;
  readonly paddingH: number;
  readonly paddingW: number;
  readonly dilationT: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
};

/**
 * Whether the implicit GEMM uses v4 (vec4 reads and writes).
 *
 * MUST: this is the only place that decides. The three conditions have the same reasons as conv2d
 * (`conv2dUsesVec4`); the innermost axis is W:
 * - `kFlat % 4 == 0` … the A-side quad read (f16's `dequant4` / i8's `unpack4xI8` rely on the flat
 *   index being a multiple of 4). **Not `Cin % 4`**.
 * - `widthOut % 4 == 0` … the B-side column quad **does not cross an output row (or frame)**.
 *   **`N % 4 == 0` is not enough** (even when Hout·Wout is a multiple of 4, a quad splits over two
 *   rows unless Wout is a multiple of 4). The `n % 4 == 0` the store side requires follows from
 *   this condition.
 * - `strideW === 1` … the x addresses of the four columns are contiguous.
 */
export const conv3dUsesVec4 = (kFlat: number, widthOut: number, strideW: number): boolean =>
  kFlat % 4 === 0 && widthOut % 4 === 0 && strideW === 1;

/**
 * Pipeline key of the implicit GEMM. The v4 flag is a shape → one-bit mapping (determinism is
 * preserved — the same vocabulary as ADR 0022 decision 2). The m-tile selection predicate is
 * shared with `conv2dIgemmMTile` (src/kernels/conv2d.ts) (it is a function of M = Cout only and
 * does not depend on the dimensionality).
 *
 * MUST: the key's geometry is derived from **the same resolution point** as generation
 * (`gemmMTileGeometry`). Embedding mTile directly would make only the key keep naming the old
 * edges when the geometry is swapped, colliding with a different WGSL.
 */
export const conv3dIgemmKey = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => {
  const geometry = gemmMTileGeometry(mTile);
  return `conv3d:v1:f32:igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}${
    v4 ? "v4" : ""
  }:wg${geometry.wgX}x${geometry.wgY}${weightKeyPart(weight)}`;
};

export const conv3dIgemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
): string => gemmWgsl({ op: "conv3d", v4, weight, mTile });

/**
 * The implicit GEMM's dispatch `[ceil(N/tileN), ceil(M/tileM), 1]` (unbatched — z is 1).
 *
 * MUST: the tile edges are derived from **the same resolution point** as generation and the key
 * (`gemmMTileGeometry(mTile)`) — passing edges around as constants means that when they disagree
 * with the geometry, `ceil(dim / constant)` falls short of the real tile count and output tiles go
 * missing (wrong values without an exception).
 * MUST: exceeding the limit raises `DispatchLimitError` ({@link tiledWorkgroups}). The one
 * workgroup = one output tile skeleton cannot fall back to grid-stride. With the default tileN of
 * 128, N ≤ 8,388,480 (on a machine with a limit of 65,535).
 */
export const conv3dIgemmWorkgroups = (
  dims: Conv3dDims,
  mTile: number,
  limit: number,
  where: string,
): readonly [number, number, number] => {
  const geometry = gemmMTileGeometry(mTile);
  return [
    tiledWorkgroups(
      dims.timeOut * dims.heightOut * dims.widthOut,
      gemmTileN(geometry),
      limit,
      where,
    ),
    tiledWorkgroups(dims.channelsOut, gemmTileM(geometry), limit, where),
    1,
  ];
};

/**
 * The implicit GEMM's uniform Dims (`{m, n, k}` + 18 geometry words = 21 words, rounded up to 24
 * words = 96 bytes — allocating in 16-byte units is a project convention (the fusion pass fixes
 * the same form in ADR 0040), not a requirement of the specification).
 *
 * `m = Cout` / `n = Tout·Hout·Wout` / `k = Cin·Kt·Kh·Kw`.
 * MUST: the order pairs with `CONV3D_DIMS_EXTRA` in gemm.ts.
 * MUST: stride / dilation / extents are positive integers (stride 0 divides by zero in the
 * output-length formula and gives wrong values where every output reads the same window; a zero
 * edge never runs the reduction and the output becomes all bias — redundant with the contract
 * check, but it closes the direct kernel-call path too).
 * MUST: only for `groups == 1` (the reduction band differs per group, so one m tile cannot share
 * one B tile). groups > 1 fails loudly (the implemented subset of ADR 0118 decision 1).
 * MUST: per axis `input length + 2·padding ≤ 2^31 − 1` (WGSL coordinate arithmetic is i32 — the
 * reason is in the check itself).
 */
export const conv3dIgemmParams = (dims: Conv3dDims): Uint32Array<ArrayBuffer> => {
  // The names pair with the WGSL Dims field names. The order is the uniform's word order as is.
  const geometry = {
    channels_in: dims.channelsIn,
    time_in: dims.timeIn,
    height_in: dims.heightIn,
    width_in: dims.widthIn,
    height_out: dims.heightOut,
    width_out: dims.widthOut,
    kernel_t: dims.kernelT,
    kernel_h: dims.kernelH,
    kernel_w: dims.kernelW,
    stride_t: dims.strideT,
    stride_h: dims.strideH,
    stride_w: dims.strideW,
    padding_t: dims.paddingT,
    padding_h: dims.paddingH,
    padding_w: dims.paddingW,
    dilation_t: dims.dilationT,
    dilation_h: dims.dilationH,
    dilation_w: dims.dilationW,
  };
  assertU32Params("conv3d igemm params", {
    ...geometry,
    channels_out: dims.channelsOut,
    time_out: dims.timeOut,
    groups: dims.groups,
  });
  const positive: readonly (readonly [string, number])[] = [
    ["channels_in", dims.channelsIn],
    ["channels_out", dims.channelsOut],
    ["time_in", dims.timeIn],
    ["height_in", dims.heightIn],
    ["width_in", dims.widthIn],
    ["time_out", dims.timeOut],
    ["height_out", dims.heightOut],
    ["width_out", dims.widthOut],
    ["kernel_t", dims.kernelT],
    ["kernel_h", dims.kernelH],
    ["kernel_w", dims.kernelW],
    ["stride_t", dims.strideT],
    ["stride_h", dims.strideH],
    ["stride_w", dims.strideW],
    ["dilation_t", dims.dilationT],
    ["dilation_h", dims.dilationH],
    ["dilation_w", dims.dilationW],
    ["groups", dims.groups],
  ];
  for (const [name, value] of positive) {
    if (value < 1)
      throw new CodegenError(`conv3d igemm params: ${name} must be a positive integer (${value})`);
  }
  if (dims.groups !== 1) {
    throw new CodegenError(
      `conv3d igemm params: groups must be 1 (${dims.groups}) — there is no GPU execution path for groups > 1 (ADR 0118 decision 1)`,
    );
  }
  const n = dims.timeOut * dims.heightOut * dims.widthOut;
  const k = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
  assertU32Params("conv3d igemm params", { n, k });
  // Unbatched, so the output is Cout·n and the weight Cout·k (groups == 1).
  assertConvFlatExtents("conv3d igemm params", {
    input: dims.channelsIn * dims.timeIn * dims.heightIn * dims.widthIn,
    output: dims.channelsOut * n,
    weight: dims.channelsOut * k,
  });
  // MUST: per axis `input length + 2·padding ≤ 2^31 − 1` (WGSL coordinate arithmetic is i32 — the
  // proof is in the leaf function's doc).
  assertConvAxisExtent("conv3d igemm params", "t", dims.timeIn, dims.paddingT);
  assertConvAxisExtent("conv3d igemm params", "h", dims.heightIn, dims.paddingH);
  assertConvAxisExtent("conv3d igemm params", "w", dims.widthIn, dims.paddingW);
  const params = new Uint32Array(24);
  params[0] = dims.channelsOut;
  params[1] = n;
  params[2] = k;
  Object.values(geometry).forEach((value, index) => {
    params[index + 3] = value;
  });
  return params;
};

/**
 * The two kernels of conv2d (`x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout]`, f32).
 *
 * | Path                | Key                                                                    | Condition   |
 * | ------------------- | ---------------------------------------------------------------------- | ----------- |
 * | implicit GEMM       | `conv2d:v3:f32:igemm{tileM}x{tileN}{v4}:wg{x}x{y}{:w…}{:form}{:dec}`   | groups == 1 |
 * | direct convolution  | `conv2d:v1:f32:direct:wg256{:w…}`                                      | groups > 1  |
 *
 * `{:form}` (`:tb-un` / `:tas-tb-un`, the geometry's skeleton form) and `{:dec}` (`:dec`, the
 * geometry's counter decode) are optional parts: they appear only in the v4 kernel and only when
 * the geometry carries them, so a geometry without them, and the scalar kernel, keep the key
 * without them ({@link conv2dIgemmKey}).
 *
 * The tile edges and workgroup shape in the implicit GEMM key are **derived from the geometry**
 * ({@link conv2dIgemmKey} — the edges `gemmMTileGeometry` returns for the m-tile 64 / 32 variants.
 * The concrete values are not copied here so that the doc does not keep naming the old edges when
 * the geometry is swapped).
 *
 * The implicit GEMM (ADR 0024) solves `C[Cout, N] = W[Cout, K] × Xcol[K, N]` by sharing fragments
 * of the GEMM skeleton (src/kernels/gemm.ts). **Its reduction order matches the direct convolution
 * exactly**, so the output is bit-identical (the only exception = signed zero — the doc of
 * {@link conv2dIgemmKey}). The direct convolution (ADR 0017) is one thread = one output element; it
 * handles groups > 1 and doubles as **the permanent differential oracle**
 * (tests/gpu_conv2d_parity_test.ts).
 *
 * ## Narrow contract (ADR 0007: kept on the runtime capability side, not in the IR)
 *
 * - **stride / padding / dilation have two components (H/W) and groups is a scalar; all four are
 *   required in attrs** (no default filling — CONV2D_ATTRS in src/ops/attrs.ts). The same
 *   discipline as conv1d: the moment omission is allowed, there is a path where "an IR with
 *   asymmetric padding silently runs as symmetric".
 * - **bias is always present (arity fixed at 3)**. A conv without bias is normalised to arity 3 by
 *   the exporter's **zero-bias synthesis** (ADR 0015) — no arity branch in the kernel.
 *
 * MUST: **the weight is `[Cout, Cin/groups, Kh, Kw]`**. A mix-up (`[Cin/groups, Cout, …]` / `Kh`
 * and `Kw` swapped) can produce a shape with the right element count that slips through the shape
 * check, so the index is built in the order `oc` → `ic_rel` → `kh` → `kw`, and the tests pin it
 * with asymmetric shapes (Cin ≠ Cout / Kh ≠ Kw / stride and padding with H≠W) (the lesson of
 * conv_transpose1d).
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535) (the VAE decoder at 512px has one layer of 128×512×512 = 33.55 million elements).
 *
 * The reduction runs sequentially in ascending `(ic, kh, kw)` order, so the same input always gets
 * the same rounding (determinism). The padding region is zero-filled, so it is **skipped rather
 * than added** (adding 0 changes the rounding sequence, and dropping the out-of-range branch lets
 * the index go negative).
 *
 * groups has the same form as conv1d (only the input channel band of the group that output
 * channel `oc` belongs to is reduced, and the weight's second axis is indexed by the relative
 * number within the group).
 *
 * The weight has storage variants (`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019). They differ only
 * in the one read line inside the reduction and in i8's per-channel scale
 * (src/kernels/weight-storage.ts). conv2d has no i4 execution path (ADR 0069 decision 5) — the
 * direct kernel rejects it in {@link assertDirectWeight}, and the implicit GEMM in gemm.ts's op
 * branch at the generator entry point.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { assertConvAxisExtent, assertConvFlatExtents } from "./conv-bounds.ts";
import {
  conv2dCounterDecode,
  GEMM_MTILE_SMALL,
  gemmMTileGeometry,
  gemmTakesForm,
  gemmWgsl,
} from "./gemm.ts";
import { GEMM_TILE, type GemmGeometry, gemmTileM, gemmTileN } from "./gemm-geometry.ts";
import {
  WEIGHT_SCALE_VAR,
  weightArrayType,
  weightKeyPart,
  weightLoaderWgsl,
  weightNote,
  weightRead,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

export const CONV2D_WORKGROUP_SIZE = 256;

/**
 * Scale binding of the i8 variant (the number after the output — used together with the executor's
 * bind entries). The direct kernel and the implicit GEMM have the same binding layout, so the
 * number is shared too (a codegen test pins that it equals gemm.ts's `LINEAR_SCALE_BINDING`).
 */
export const CONV2D_SCALE_BINDING = 5;

/**
 * The direct kernel **does not accept i4** (ADR 0069 decision 5 — the i4 execution path is only
 * the implicit GEMM of linear / embedding / conv1d; conv2d has none, not even on the implicit GEMM
 * side).
 *
 * MUST: the eligibility check (`i4EligibleInitializers` in plan.ts) drops conv2d from the i4
 * eligible set, so this is normally unreachable, but **the path of calling the kernel directly is
 * closed too** (generation alone would succeed, producing invalid WGSL in which
 * `dequant(i, wscale_v)` references an undeclared identifier without a group scale binding — the
 * same fail-loudly style as the conv1d direct kernel).
 */
const assertDirectWeight = (weight: WeightStorage): void => {
  if (weight === "i2") {
    throw new CodegenError("convolution: i2 weight storage is not supported (ADR 0097)");
  }
  if (weight === "i4") {
    throw new CodegenError(
      "conv2d direct kernel: i4 weight storage is not supported — the only i4 execution paths are linear / embedding / the implicit GEMM of conv1d(groups==1) (ADR 0069 decision 5)",
    );
  }
};

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const conv2dKey = (weight: WeightStorage): string => {
  assertDirectWeight(weight);
  return `conv2d:v1:f32:direct:wg${CONV2D_WORKGROUP_SIZE}${weightKeyPart(weight)}`;
};

/**
 * Whether the implicit GEMM uses v4 (vec4 reads and writes).
 *
 * MUST: this is the only place that decides. Each of the three conditions is needed for its own
 * reason:
 * - `kFlat % 4 == 0` … the A-side quad read (f16's `dequant4` / i8's `unpack4xI8` rely on **the
 *   flat index being a multiple of 4**, and the row start = `arow · kFlat` satisfies it).
 *   **Not `Cin % 4`** (a 3×3 with Cin=2 has kFlat=18).
 * - `widthOut % 4 == 0` … the B-side column quad **does not cross an output row**. **`N % 4 == 0`
 *   is not enough** (B=1, Hout=Wout=2 has N=4, so N%4==0, but the quad splits over two rows). In
 *   the measured shapes Wout is always a multiple of 4, so this mix-up **never shows on real
 *   models** — the Hout=Wout=2 cases of tests/gpu_ops_test.ts and tests/gpu_conv2d_parity_test.ts
 *   are the only detector. The `n % 4 == 0` the store side requires follows from this condition
 *   (n = B·Hout·Wout).
 * - `strideW === 1` … the x addresses of the four columns are contiguous.
 */
export const conv2dUsesVec4 = (kFlat: number, widthOut: number, strideW: number): boolean =>
  kFlat % 4 === 0 && widthOut % 4 === 0 && strideW === 1;

/**
 * Pipeline key of the implicit GEMM (ADR 0024). A separate line from the direct kernel's `v1`; the
 * v4 flag is a shape → one-bit mapping (determinism is preserved — the same vocabulary as ADR 0022
 * decision 2).
 *
 * NOTE: the output is **bit-identical** to the direct kernel (the reduction order matches exactly).
 * The only exception is signed zero: adding a padding `+0.0` where the partial sum is exactly
 * `−0.0` turns it into `+0.0` (the direct kernel does not add padding, so `−0.0` remains). It is
 * unreachable unless the bias is 0.
 *
 * `geometry` is an explicit geometry (`GemmSpec` in src/kernels/gemm.ts — the profile's value in a
 * Session, a sweep candidate in measurement); when given, it takes precedence over `mTile`. The
 * edges and the workgroup shape determine `regM = tileM / wgY` and `regN = tileN / wgX`, so the
 * geometry can be told apart with the same spelling. MUST: pass **the same value** to
 * {@link conv2dIgemmWgsl}.
 */
export const conv2dIgemmKey = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
  geometry?: GemmGeometry,
): string => {
  // MUST: the key's geometry is derived from **the same resolution point** as generation (explicit
  // → `gemmMTileGeometry`, in that order). Embedding mTile directly would make only the key keep
  // naming the old edges when the geometry is swapped, colliding with a different WGSL.
  const resolved = geometry ?? gemmMTileGeometry(mTile);
  // MUST: the skeleton form appends `:tb-un` / `:tas-tb-un` and the counter decode `:dec`, each
  // only where the kernel takes it (the v4 kernel; the scalar kernel keeps the original text
  // whatever the geometry carries); a geometry without them keeps its key byte for byte.
  return `conv2d:v3:f32:igemm${gemmTileM(resolved)}x${gemmTileN(resolved)}${
    v4 ? "v4" : ""
  }:wg${resolved.wgX}x${resolved.wgY}${weightKeyPart(weight)}${
    gemmTakesForm("conv2d", v4) && resolved.form !== undefined ? `:${resolved.form}` : ""
  }${conv2dCounterDecode(resolved, v4) ? ":dec" : ""}`;
};

export const conv2dIgemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
  geometry?: GemmGeometry,
): string => gemmWgsl({ op: "conv2d", v4, weight, mTile, geometry });

/**
 * Choice of the m-tile row count (the 32-row m-tile variant of the ADR 0024 addendum).
 *
 * The 32-row tile is chosen only when `M % 64` is 1–32. Only in this band is "the last 64-row tile
 * at most half filled", and splitting it into two 32-row tiles removes the waste entirely (the
 * census's Cout=96 goes from `2·64/96 = 1.33×` → `3·32/96 = 1.00×`; Cout=3 from 21.33× → 10.67×).
 *
 * MUST: the boundary is `<= 32`. With `< 32`, exactly M%64 == 32 (= the census's Cout=96, **the
 * main target, 32.4% of the MACs of the 1024 series**) stays at 64 rows. Shapes with
 * `M % 64 == 0` (Cout=192/384/1152) have zero waste at 64 rows, and splitting into 32 rows only
 * reads the B tile (the implicit gather of x) twice. The 33–64 band also has **the same tile
 * quantization waste** (one 64-row tile's worth either way), so the 64-row tile, with fewer B-tile
 * reads, is chosen.
 */
export const conv2dIgemmMTile = (channelsOut: number): number => {
  const remainder = channelsOut % GEMM_TILE;
  return remainder >= 1 && remainder <= GEMM_MTILE_SMALL ? GEMM_MTILE_SMALL : GEMM_TILE;
};

export const conv2dWgsl = (weight: WeightStorage): string => {
  assertDirectWeight(weight);
  return `// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout], f32${weightNote(
    weight,
  )}, direct convolution)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  height_in: u32,
  width_in: u32,
  height_out: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_h: u32,
  stride_w: u32,
  padding_h: u32,
  padding_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("w", weight, CONV2D_SCALE_BINDING)}
@compute @workgroup_size(${CONV2D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${CONV2D_WORKGROUP_SIZE}u;
  // The contract check (src/ops.ts) guarantees groups divides Cin / Cout — the division is exact.
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.height_out * dims.width_out;
    let chunk = dims.channels_out * plane;
    let b = i / chunk;
    let rest = i % chunk;
    let oc = rest / plane;
    let pixel = rest % plane;
    let oy = pixel / dims.width_out;
    let ox = pixel % dims.width_out;
    // the start position on the input side is signed (it goes negative by the padding)
    let origin_y = i32(oy * dims.stride_h) - i32(dims.padding_h);
    let origin_x = i32(ox * dims.stride_w) - i32(dims.padding_w);
    // the weight's second axis is Cin/groups — indexed by the relative number within the group
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;${weightScaleWgsl(weight, "oc", "    ")}
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.height_in * dims.width_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel_h * dims.kernel_w;
      for (var kh = 0u; kh < dims.kernel_h; kh = kh + 1u) {
        let iy = origin_y + i32(kh * dims.dilation_h);
        // the padding region is zero-filled — skipped rather than added
        if (iy < 0 || u32(iy) >= dims.height_in) {
          continue;
        }
        let row_base = x_base + u32(iy) * dims.width_in;
        let w_row = w_base + kh * dims.kernel_w;
        for (var kw = 0u; kw < dims.kernel_w; kw = kw + 1u) {
          let ix = origin_x + i32(kw * dims.dilation_w);
          if (ix >= 0 && u32(ix) < dims.width_in) {
            acc = acc + x[row_base + u32(ix)] * ${weightRead(
              "w",
              weight,
              "w_row + kw",
              WEIGHT_SCALE_VAR,
            )};
          }
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;
};

/** Geometry of conv2d (the one input type shared by the two params functions). */
export type Conv2dDims = {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly heightOut: number;
  readonly widthOut: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly paddingH: number;
  readonly paddingW: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
};

/**
 * Contract check of the geometry (shared by both kernels). Returns the values in the order of the
 * direct kernel's Dims.
 *
 * MUST: reject stride 0. stride is used only to multiply coordinates (not as a loop increment), so
 * with 0 the output-length formula (shapes.ts) divides by zero and the kernel gives wrong values
 * where every output reads the same window (no exception) — redundant with the contract check, but
 * it closes the direct kernel-call path too (the same double check as conv_transpose1d).
 */
const checkConv2dDims = (dims: Conv2dDims): readonly number[] => {
  // The names pair with the WGSL Dims field names (from the one after `n`). The order is the
  // uniform's word order as is.
  const dimensions = {
    batch: dims.batch,
    channels_in: dims.channelsIn,
    channels_out: dims.channelsOut,
    height_in: dims.heightIn,
    width_in: dims.widthIn,
    height_out: dims.heightOut,
    width_out: dims.widthOut,
    kernel_h: dims.kernelH,
    kernel_w: dims.kernelW,
    stride_h: dims.strideH,
    stride_w: dims.strideW,
    padding_h: dims.paddingH,
    padding_w: dims.paddingW,
    dilation_h: dims.dilationH,
    dilation_w: dims.dilationW,
    groups: dims.groups,
  };
  assertU32Params("conv2d params", dimensions);
  const values = Object.values(dimensions);
  const positive: readonly (readonly [string, number])[] = [
    // MUST: the tensor extents are positive integers too. A zero edge never runs the reduction and
    // the output becomes all bias without an exception (on the GPU, binding a zero-element buffer
    // fails validation, but a direct kernel call does not get that far) — pairs with deform_conv2d
    // having the same set.
    ["batch", dims.batch],
    ["channels_in", dims.channelsIn],
    ["channels_out", dims.channelsOut],
    ["height_in", dims.heightIn],
    ["width_in", dims.widthIn],
    ["height_out", dims.heightOut],
    ["width_out", dims.widthOut],
    ["stride_h", dims.strideH],
    ["stride_w", dims.strideW],
    ["kernel_h", dims.kernelH],
    ["kernel_w", dims.kernelW],
    ["dilation_h", dims.dilationH],
    ["dilation_w", dims.dilationW],
    ["groups", dims.groups],
  ];
  for (const [name, value] of positive) {
    if (value < 1)
      throw new CodegenError(`conv2d params: ${name} must be a positive integer (${value})`);
  }
  // MUST: per axis `input length + 2·padding ≤ 2^31 − 1` (WGSL coordinate arithmetic is i32 — kept
  // in this check, which both pass, so the direct and the implicit GEMM versions accept the same
  // set).
  assertConvAxisExtent("conv2d params", "h", dims.heightIn, dims.paddingH);
  assertConvAxisExtent("conv2d params", "w", dims.widthIn, dims.paddingW);
  // MUST: divisibility by groups is checked in the params layer too (the shader's division
  // truncates, and the input channel band read shifts silently).
  if (dims.channelsIn % dims.groups !== 0 || dims.channelsOut % dims.groups !== 0) {
    throw new CodegenError(
      `conv2d params: groups ${dims.groups} does not divide Cin ${dims.channelsIn} / Cout ${dims.channelsOut}`,
    );
  }
  // The weight's element count uses Cin/groups, so it comes after the divisibility check.
  assertConvFlatExtents("conv2d params", {
    input: dims.batch * dims.channelsIn * dims.heightIn * dims.widthIn,
    output: dims.batch * dims.channelsOut * dims.heightOut * dims.widthOut,
    weight: dims.channelsOut * (dims.channelsIn / dims.groups) * dims.kernelH * dims.kernelW,
  });
  return values;
};

/**
 * The direct kernel's uniform Dims. 17 words rounded up to a 16-byte unit: 20 words (80 bytes)
 * are allocated (allocating in 16-byte units is a project convention (the fusion pass fixes the
 * same form in ADR 0040), not a requirement of the specification).
 */
export const conv2dParams = (dims: Conv2dDims): Uint32Array<ArrayBuffer> => {
  const values = checkConv2dDims(dims);
  const n = dims.batch * dims.channelsOut * dims.heightOut * dims.widthOut;
  assertU32Params("conv2d params", { n });
  const params = new Uint32Array(20);
  params[0] = n;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  return params;
};

/**
 * The implicit GEMM's uniform Dims (`{m, n, k}` + 12 geometry words = 15 words, so 16 words = 64
 * bytes are allocated).
 *
 * `m = Cout` / `n = Hout·Wout` (**the output plane of one batch** — the batch is the dispatch's z
 * axis) / `k = Cin·Kh·Kw`.
 * MUST: the order pairs with `CONV2D_DIMS_EXTRA` in gemm.ts. Only for `groups == 1`; anything else
 * fails loudly (**the reduction band differs per group, so a 64-row m tile cannot share one B
 * tile** — groups > 1 goes to the direct kernel).
 */
export const conv2dIgemmParams = (dims: Conv2dDims): Uint32Array<ArrayBuffer> => {
  checkConv2dDims(dims);
  if (dims.groups !== 1) {
    throw new CodegenError(`conv2d igemm params: groups must be 1 (${dims.groups})`);
  }
  const n = dims.heightOut * dims.widthOut;
  const k = dims.channelsIn * dims.kernelH * dims.kernelW;
  assertU32Params("conv2d igemm params", { n, k });
  const params = new Uint32Array(16);
  params[0] = dims.channelsOut;
  params[1] = n;
  params[2] = k;
  [
    dims.channelsIn,
    dims.heightIn,
    dims.widthIn,
    dims.widthOut,
    dims.kernelH,
    dims.kernelW,
    dims.strideH,
    dims.strideW,
    dims.paddingH,
    dims.paddingW,
    dims.dilationH,
    dims.dilationW,
  ].forEach((value, index) => {
    params[index + 3] = value;
  });
  return params;
};

/**
 * The two kernels of conv1d (`x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout]`, f32).
 *
 * | Path          | Key                                                         | Condition   |
 * | ------------- | ----------------------------------------------------------- | ----------- |
 * | implicit GEMM | `conv1d:v4:f32:igemm{tileM}x{tileN}{v4}:wg{x}x{y}{:w…}{g…}` | groups == 1 |
 * | direct        | `conv1d:v2:f32:direct:wg256{:w…}`                           | groups > 1  |
 *
 * The tile edges and workgroup shape in the implicit GEMM key are **derived from the geometry**
 * ({@link conv1dIgemmKey} — the same discipline as conv2d: copying the concrete values would make
 * only the doc and the key keep naming the old edges when the geometry is swapped).
 *
 * The implicit GEMM (the 1D version of ADR 0024) solves `C[Cout, N] = W[Cout, K] × Xcol[K, N]` by
 * sharing fragments of the GEMM skeleton (src/kernels/gemm.ts). **Its reduction order matches the
 * direct convolution exactly**, so the output is bit-identical (the only exception = signed zero —
 * the doc of {@link conv1dIgemmKey}). The direct convolution (ADR 0017) is one thread = one output
 * element; it handles groups > 1 and doubles as **the permanent differential oracle**
 * (tests/gpu_conv1d_parity_test.ts).
 *
 * ## Narrow contract (ADR 0007: kept on the runtime capability side, not in the IR)
 *
 * - **groups / dilation are attrs (ADR 0015)**. The property "there is no path that silently runs
 *   a value other than 1 as 1", once guaranteed by "no field = fixed at 1", is guaranteed after the
 *   fields were added only by **not filling in defaults** (CONV1D_ATTRS in src/ops/attrs.ts).
 * - **bias is always present (arity fixed at 3)**. A conv without bias is normalised to arity 3 by
 *   the exporter **synthesising a zero bias** (ADR 0015) — no arity branch in the kernel.
 * - `B` / `Cin` / `Cout` / `K` / stride / padding / dilation / groups are **ordinary loop bounds**
 *   carried in params, so they are not pinned to measured values. Pinning would only reduce the
 *   shapes that can run without adding any power to detect errors (errors come from mixed-up axes,
 *   which tests with asymmetric shapes catch).
 *
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535) (measured: outputs of 1024×T elements).
 *
 * The reduction runs sequentially in ascending `(ic, k)` order, so the same input always gets the
 * same rounding (determinism). The padding region is zero-filled, so it is **skipped rather than
 * added** (adding 0 changes the rounding sequence, and dropping the out-of-range branch lets the
 * index go negative).
 *
 * groups is implemented as "reduce only the input channel band `[g·Cin/groups, (g+1)·Cin/groups)`
 * of the group g = oc / (Cout/groups) that output channel oc belongs to". The weight's second axis
 * is **Cin/groups, not Cin**, so the w index is also built from the relative number within the
 * band (leaving it at Cin is a silent wrong value that skips weights in depthwise convolutions).
 *
 * The weight has storage variants (`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019). They differ only
 * in the one read line inside the reduction and in i8's per-channel scale (loop-invariant per `oc`,
 * so hoisted out of the reduction) (src/kernels/weight-storage.ts).
 *
 * **Only the implicit GEMM has `w=i4`** (the conv1d addendum of ADR 0069 decision 5, wave J-5b).
 * The group scale depends on k = `Cin·K` and cannot be hoisted per `oc`, so the A-tile fill reads
 * it per quad (`fillAConv` in gemm.ts) — the direct kernel (groups > 1) has no decode path, and the
 * generator entry points reject it ({@link conv1dKey} / {@link conv1dWgsl}). The i4 variant bakes
 * the group length into the WGSL, so the key carries a `g` part (`:wi4g32` — the same style as
 * linear).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { assertConvAxisExtent, assertConvFlatExtents } from "./conv-bounds.ts";
import { gemmMTileGeometry, gemmWgsl } from "./gemm.ts";
import { GEMM_TILE, gemmTileM, gemmTileN } from "./gemm-geometry.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  WEIGHT_SCALE_VAR,
  weightArrayType,
  weightKeyPart,
  weightLoaderWgsl,
  weightNote,
  weightRead,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

export const CONV1D_WORKGROUP_SIZE = 256;

/** i8 / i4 scale binding (the number after the output — paired with the executor bind entries). */
export const CONV1D_SCALE_BINDING = 5;

/**
 * The direct kernel (groups > 1) **does not accept i4** (the conv1d addendum of ADR 0069 decision
 * 5 is limited to the implicit GEMM = groups == 1).
 *
 * MUST: the eligibility check (`i4EligibleInitializers` in plan.ts) drops conv1d with `groups > 1`
 * from the i4 eligible set, so this is normally unreachable, but **the path of calling the kernel
 * directly is closed too** (generation alone would succeed, producing invalid WGSL in which
 * `dequant(i, scale)` references an undeclared identifier without a group scale index — the
 * existing fail-loudly style).
 */
const assertDirectWeight = (weight: WeightStorage): void => {
  if (weight === "i2") {
    throw new CodegenError("畳み込み: 重み i2 格納は未対応（ADR 0097）");
  }
  if (weight === "i4") {
    throw new CodegenError(
      "conv1d 直接カーネル: 重み i4 格納は未対応 — i4 の conv1d は groups == 1 の implicit GEMM 限定（ADR 0069 決定 5）",
    );
  }
};

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const conv1dKey = (weight: WeightStorage): string => {
  assertDirectWeight(weight);
  return `conv1d:v2:f32:direct:wg${CONV1D_WORKGROUP_SIZE}${weightKeyPart(weight)}`;
};

/**
 * Whether the implicit GEMM uses v4 (vec4 reads and writes).
 *
 * MUST: this is the only place that decides. The three conditions answer two reasons:
 * - `kFlat % 4 == 0` … the A-side quad read (f16's `dequant4` / i8's `unpack4xI8` rely on **the
 *   flat index being a multiple of 4**, and the row start = `arow · kFlat` satisfies it).
 *   **Not `Cin % 4`** (K=7 with Cin=1 has kFlat=7).
 * - `lengthOut % 4 == 0 && stride == 1` … the B-side column quad lands on four contiguous elements
 *   of x, and the store side's quad write has no remainder. In 1D the output plane is one row, so
 *   `n = Lout`, and the reason conv2d distinguishes `Wout % 4` from `N % 4` (a quad crossing an
 *   output row) disappears here.
 */
export const conv1dUsesVec4 = (kFlat: number, lengthOut: number, stride: number): boolean =>
  kFlat % 4 === 0 && lengthOut % 4 === 0 && stride === 1;

/**
 * Pipeline key of the implicit GEMM (the 1D version of ADR 0024). A separate line from the direct
 * kernel's `v2`; the v4 flag is a shape → one-bit mapping (determinism is preserved — the same
 * vocabulary as ADR 0022 decision 2).
 *
 * NOTE: the output is **bit-identical** to the direct kernel (the reduction order matches exactly).
 * The only exception is signed zero: adding a padding `+0.0` where the partial sum is exactly
 * `−0.0` turns it into `+0.0` (the direct kernel does not add padding, so `−0.0` remains). It is
 * unreachable unless the bias is 0 (the mechanism is pinned by the negative case of
 * tests/gpu_conv1d_parity_test.ts).
 *
 * NOTE: the m-tile selection predicate is shared with `conv2dIgemmMTile` (src/kernels/conv2d.ts).
 * **It is a function of M = Cout only** (waste = `ceil(M/tile)·tile / M`), so it does not depend on
 * the dimensionality; copying it for 1D would scatter the same boundary over two places.
 */
export const conv1dIgemmKey = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
  groupSize?: number,
): string => {
  i4GroupShift("conv1d igemm", weight, groupSize);
  // MUST: the key's geometry is derived from **the same resolution point** as generation
  // (`gemmMTileGeometry`). Embedding mTile directly would make only the key keep naming the old
  // edges when the geometry is swapped, colliding with a different WGSL.
  const geometry = gemmMTileGeometry(mTile);
  return `conv1d:v4:f32:igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}${
    v4 ? "v4" : ""
  }:wg${geometry.wgX}x${geometry.wgY}${weightKeyPart(weight)}${i4GroupKeyPart(groupSize)}`;
};

/**
 * `groupSize` is the i4 group length (required for i4 — the conv1d addendum of ADR 0069 decision
 * 5). The shift is baked into the WGSL, so it **goes into the key too** (`:wi4g32` — the same style
 * as linear).
 */
export const conv1dIgemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  mTile: number = GEMM_TILE,
  groupSize?: number,
): string =>
  gemmWgsl({
    op: "conv1d",
    v4,
    weight,
    mTile,
    weightGroupShift: i4GroupShift("conv1d igemm", weight, groupSize),
  });

export const conv1dWgsl = (weight: WeightStorage): string => {
  assertDirectWeight(weight);
  return `// karume conv1d (x[B,Cin,L] * W[Cout,Cin/groups,K] + b[Cout], f32${weightNote(
    weight,
  )}, direct convolution)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  length_in: u32,
  length_out: u32,
  kernel: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  groups: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("w", weight, CONV1D_SCALE_BINDING)}
@compute @workgroup_size(${CONV1D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${CONV1D_WORKGROUP_SIZE}u;
  // The contract check (src/ops.ts) guarantees groups divides Cin / Cout — the division is exact.
  let in_per_group = dims.channels_in / dims.groups;
  let out_per_group = dims.channels_out / dims.groups;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // the start position on the input side is signed (it goes negative by the padding)
    let origin = i32(ox * dims.stride) - i32(dims.padding);
    // the weight's second axis is Cin/groups — indexed by the relative number within the group
    let group_index = oc / out_per_group;
    let ic_base = group_index * in_per_group;${weightScaleWgsl(weight, "oc", "    ")}
    var acc = bias[oc];
    for (var ic_rel = 0u; ic_rel < in_per_group; ic_rel = ic_rel + 1u) {
      let x_base = (b * dims.channels_in + ic_base + ic_rel) * dims.length_in;
      let w_base = (oc * in_per_group + ic_rel) * dims.kernel;
      for (var k = 0u; k < dims.kernel; k = k + 1u) {
        let ix = origin + i32(k * dims.dilation);
        // the padding region is zero-filled — skipped rather than added
        if (ix >= 0 && u32(ix) < dims.length_in) {
          acc = acc + x[x_base + u32(ix)] * ${weightRead(
            "w",
            weight,
            "w_base + k",
            WEIGHT_SCALE_VAR,
          )};
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;
};

/** Geometry of conv1d (the one input type shared by the two params functions). */
export type Conv1dDims = {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly lengthIn: number;
  readonly lengthOut: number;
  readonly kernel: number;
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
};

/**
 * Contract check of the geometry (shared by both kernels). Returns the values in the order of the
 * direct kernel's Dims.
 *
 * MUST: reject stride 0. stride is used only to multiply coordinates (not as a loop increment), so
 * with 0 the output-length formula (shapes.ts) divides by zero and the kernel gives wrong values
 * where every output reads the same window (no exception) — redundant with the contract check, but
 * it closes the direct kernel-call path too.
 */
const checkConv1dDims = (dims: Conv1dDims): readonly number[] => {
  // The names pair with the WGSL Dims field names (from the one after `n`). The order is the
  // uniform's word order as is.
  const dimensions = {
    batch: dims.batch,
    channels_in: dims.channelsIn,
    channels_out: dims.channelsOut,
    length_in: dims.lengthIn,
    length_out: dims.lengthOut,
    kernel: dims.kernel,
    stride: dims.stride,
    padding: dims.padding,
    dilation: dims.dilation,
    groups: dims.groups,
  };
  assertU32Params("conv1d params", dimensions);
  const values = Object.values(dimensions);
  if (dims.stride < 1 || dims.kernel < 1 || dims.dilation < 1 || dims.groups < 1) {
    throw new CodegenError(
      `conv1d params: stride / kernel / dilation / groups は正整数（${dims.stride}, ${dims.kernel}, ${dims.dilation}, ${dims.groups}）`,
    );
  }
  // MUST: the tensor extents are positive integers too. A zero edge never runs the reduction and
  // the output becomes all bias without an exception (pairs with deform_conv2d, which has the
  // same set).
  const positive: readonly (readonly [string, number])[] = [
    ["batch", dims.batch],
    ["channels_in", dims.channelsIn],
    ["channels_out", dims.channelsOut],
    ["length_in", dims.lengthIn],
    ["length_out", dims.lengthOut],
  ];
  for (const [name, value] of positive) {
    if (value < 1) throw new CodegenError(`conv1d params: ${name} は正整数（${value}）`);
  }
  // MUST: `input length + 2·padding ≤ 2^31 − 1` (WGSL coordinate arithmetic is i32 — kept in this
  // check, which both pass, so the direct and the implicit GEMM versions accept the same set).
  assertConvAxisExtent("conv1d params", "l", dims.lengthIn, dims.padding);
  // MUST: divisibility by groups is checked in the params layer too (the shader's division
  // truncates, and the input channel band read shifts silently — redundant with the contract
  // check, but it closes the direct kernel-call path too).
  if (dims.channelsIn % dims.groups !== 0 || dims.channelsOut % dims.groups !== 0) {
    throw new CodegenError(
      `conv1d params: groups ${dims.groups} が Cin ${dims.channelsIn} / Cout ${dims.channelsOut} を割り切らない`,
    );
  }
  // The weight's element count uses Cin/groups, so it comes after the divisibility check.
  assertConvFlatExtents("conv1d params", {
    input: dims.batch * dims.channelsIn * dims.lengthIn,
    output: dims.batch * dims.channelsOut * dims.lengthOut,
    weight: dims.channelsOut * (dims.channelsIn / dims.groups) * dims.kernel,
  });
  return values;
};

/**
 * The direct kernel's uniform Dims. 11 words rounded up to a 16-byte unit: 12 words (48 bytes) are
 * allocated (allocating in 16-byte units is a project convention (the fusion pass fixes the same
 * form in ADR 0040), not a requirement of the specification).
 */
export const conv1dParams = (dims: Conv1dDims): Uint32Array<ArrayBuffer> => {
  const values = checkConv1dDims(dims);
  const n = dims.batch * dims.channelsOut * dims.lengthOut;
  assertU32Params("conv1d params", { n });
  const params = new Uint32Array(12);
  params[0] = n;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  return params;
};

/**
 * The implicit GEMM's uniform Dims (`{m, n, k}` + 6 geometry words = 9 words, so 12 words = 48
 * bytes are allocated).
 *
 * `m = Cout` / `n = Lout` (**the output plane of one batch** — the batch is the dispatch's z axis)
 * / `k = Cin·K`.
 * MUST: the order pairs with `CONV1D_DIMS_EXTRA` in gemm.ts. Only for `groups == 1`; anything else
 * fails loudly (**the reduction band differs per group, so one m tile cannot share one B tile** —
 * groups > 1 goes to the direct kernel).
 */
export const conv1dIgemmParams = (dims: Conv1dDims): Uint32Array<ArrayBuffer> => {
  checkConv1dDims(dims);
  if (dims.groups !== 1) {
    throw new CodegenError(`conv1d igemm params: groups は 1 専用（${dims.groups}）`);
  }
  const k = dims.channelsIn * dims.kernel;
  assertU32Params("conv1d igemm params", { k });
  const params = new Uint32Array(12);
  params[0] = dims.channelsOut;
  params[1] = dims.lengthOut;
  params[2] = k;
  [dims.channelsIn, dims.lengthIn, dims.kernel, dims.stride, dims.padding, dims.dilation].forEach(
    (value, index) => {
      params[index + 3] = value;
    },
  );
  return params;
};

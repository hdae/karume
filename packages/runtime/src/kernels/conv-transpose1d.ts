/**
 * The fixed kernel of conv_transpose1d (`x[B,Cin,L] ⊛ᵀ W[Cin,Cout,K] + b[Cout]`, f32).
 * A **gather form** with one thread = one output element (required by the full-write invariant —
 * below). The inside of the reduction uses **residue grouping** (perf-ledger K-10 — it walks only
 * the valid taps, over an O(1) range).
 *
 * ## Why the gather form (the full-write invariant — ADR 0014)
 *
 * The straightforward implementation of a transposed convolution is a scatter that "**spreads** one
 * input point over several output points", which zero-initialises the output buffer and then does
 * read-modify-write. Karume's buffers are recycled (ADR 0004) and not zero-initialised, and the
 * additions from several threads would race. Here it is turned inside out into **one output element
 * = the sum of the inputs that contribute to it**, so one thread **writes exactly once** to one
 * output element.
 *
 * The pairs `(ix, k)` that contribute to output coordinate `ox` satisfy
 * `ox = ix·stride − padding + k`, and only when `t = ox + padding − k` **is divisible by stride**
 * and `0 <= t/stride < L`. A `k` that does not divide means "there is no input there" = zero
 * contribution, so like the padding region it **does not appear in the addition** (adding 0
 * changes the rounding sequence).
 *
 * ## residue grouping (perf-ledger K-10)
 *
 * The divisibility test is equivalent to `k ≡ shifted (mod stride)` (`shifted = ox + padding`), so
 * decomposing `shifted = q·stride + r` **once per output element** lets the valid taps be written
 * as **an enumeration** `k = r + j·stride` / `ix = q − j`. The range of `j` closes in O(1) with
 * three inequalities (`ix <= L−1` / `ix >= 0` / `k <= K−1`), so the scan over every `k` and the
 * remainder test disappear (in dacvae's measured shape, stride 8 / K 16, 7/8 of the taps were
 * tested invalid).
 *
 * ## Narrow contract
 *
 * - The weight is **`[Cin, Cout, K]`** (the transpose of conv1d's `[Cout, Cin, K]`). A mix-up can
 *   produce a shape with the right element count (Cin == Cout) that slips through the shape check,
 *   so the tests pin it with **asymmetric channel counts** (ADR 0015 / recon §4).
 * - **stride >= 1 MUST**. The prototype's implementation did not advance its loop at stride 0 and
 *   **hung the GPU** (no exception). This gather-form kernel does not use stride as a loop bound,
 *   but residue splitting divides by zero once per element in `shifted % stride`, so
 *   {@link convTranspose1dParams} blocks it (a double check with the contract check).
 * - bias is always present (arity fixed at 3). A conv without bias is normalised by the exporter's
 *   zero-bias synthesis.
 * - `output_padding` / `dilation` / `groups` are not in attrs = fixed at 0 / 1 / 1, as measured.
 *
 * MUST: grid-stride loop (the ups of dec grow the output to 512×T·8 elements).
 * The reduction runs sequentially in ascending `(ic, k)` order, so the same input always gets the
 * same rounding (determinism).
 *
 * The weight has storage variants (`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019). They differ only
 * in the one read line inside the reduction and in i8's per-channel scale
 * (src/kernels/weight-storage.ts).
 * MUST: the **channel axis of the i8 scale is 1** (the second axis of the weight `[Cin, Cout, K]` =
 * the output channel). The other four kernels use axis 0; only this one is shifted by the
 * transposed layout.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { assertConvFlatExtents, assertI32Coordinates } from "./conv-bounds.ts";
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

export const CONV_TRANSPOSE1D_WORKGROUP_SIZE = 256;

/** i8 scale binding (the number after the output — paired with the executor bind entries). */
export const CONV_TRANSPOSE1D_SCALE_BINDING = 5;

/**
 * This kernel **does not accept i4** (ADR 0069 decision 5 — the i4 execution path is only the
 * implicit GEMM of linear / embedding / conv1d(groups == 1); conv_transpose1d has no decode path).
 *
 * MUST: the eligibility check (`i4EligibleInitializers` in plan.ts) drops conv_transpose1d from the
 * i4 eligible set, so this is normally unreachable, but **the path of calling the kernel directly
 * is closed too** (generation alone would succeed, producing invalid WGSL in which
 * `dequant(i, wscale_v)` references an undeclared identifier without a group scale binding — the
 * same fail-loudly style as the conv1d direct kernel).
 */
const assertGatherWeight = (weight: WeightStorage): void => {
  if (weight === "i2") {
    throw new CodegenError("畳み込み: 重み i2 格納は未対応（ADR 0097）");
  }
  if (weight === "i4") {
    throw new CodegenError(
      "conv_transpose1d: 重み i4 格納は未対応 — i4 の実行経路は linear / embedding / conv1d(groups==1) の implicit GEMM だけ（ADR 0069 決定 5）",
    );
  }
};

export const convTranspose1dKey = (weight: WeightStorage): string => {
  assertGatherWeight(weight);
  return `conv_transpose1d:v2:f32:gather:wg${CONV_TRANSPOSE1D_WORKGROUP_SIZE}${weightKeyPart(
    weight,
  )}`;
};

export const convTranspose1dWgsl = (weight: WeightStorage): string => {
  assertGatherWeight(weight);
  return `// karume conv_transpose1d (x[B,Cin,L] * W[Cin,Cout,K] + b[Cout], f32${weightNote(
    weight,
  )}, gather form)
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
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("w", weight, CONV_TRANSPOSE1D_SCALE_BINDING)}
@compute @workgroup_size(${CONV_TRANSPOSE1D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${CONV_TRANSPOSE1D_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < dims.n) {
    let plane = dims.channels_out * dims.length_out;
    let b = i / plane;
    let rest = i % plane;
    let oc = rest / dims.length_out;
    let ox = rest % dims.length_out;
    // ox = ix*stride - padding + k  =>  ix*stride = ox + padding - k
    // the valid taps are only k ≡ shifted (mod stride) — decompose into quotient and remainder
    // once per output element
    let shifted = ox + dims.padding;
    let q = shifted / dims.stride;
    let r = shifted % dims.stride;
    // the valid taps are k = r + j*stride / ix = q - j. The range of j closes with three
    // inequalities:
    // ix <= L-1 → j >= q+1-L / ix >= 0 → j <= q / k <= K-1 → j <= (K-1-r)/stride
    let j_start = max(0, i32(q) + 1 - i32(dims.length_in));
    // r > K-1 (reachable only when stride > K) has zero valid taps. Avoids the u32 borrow of K-1-r
    // and turns it into an empty loop with j_end = -1 < j_start (j_start >= 0)
    var j_end = -1;
    if (r < dims.kernel) {
      j_end = min(i32((dims.kernel - 1u - r) / dims.stride), i32(q));
    }${weightScaleWgsl(weight, "oc", "    ")}
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * dims.length_in;
      // the weight is [Cin, Cout, K] — the transpose of conv1d (the first axis is the input
      // channel)
      let w_base = (ic * dims.channels_out + oc) * dims.kernel;
      // MUST: ascending j = ascending k. The tap set and the ascending (ic, k) reduction order are
      // identical to the version that scans every k: the same f32 multiply-adds apply to the same
      // operands in the same order = bit-identical (reordering changes the rounding)
      for (var j = j_start; j <= j_end; j = j + 1) {
        let ix = q - u32(j);
        let k = r + u32(j) * dims.stride;
        acc = acc + x[x_base + ix] * ${weightRead("w", weight, "w_base + k", WEIGHT_SCALE_VAR)};
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;
};

/**
 * The uniform Dims. 9 words rounded up to a 16-byte unit: 12 words (48 bytes) are allocated
 * (allocating in 16-byte units is a project convention (the fusion pass fixes the same form in
 * ADR 0040), not a requirement of the specification).
 */
export const convTranspose1dParams = (dims: {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly lengthIn: number;
  readonly lengthOut: number;
  readonly kernel: number;
  readonly stride: number;
  readonly padding: number;
}): Uint32Array<ArrayBuffer> => {
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
  };
  assertU32Params("conv_transpose1d params", dimensions);
  const values = Object.values(dimensions);
  // MUST: with stride 0, WGSL's `% dims.stride` divides by zero (the result is undefined, and
  // depending on the implementation it becomes a silent wrong value or a hang without an exception
  // — the precedent of recon §4).
  if (dims.stride < 1 || dims.kernel < 1) {
    throw new CodegenError(
      `conv_transpose1d params: stride / kernel は正整数（${dims.stride}, ${dims.kernel}）`,
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
    if (value < 1) throw new CodegenError(`conv_transpose1d params: ${name} は正整数（${value}）`);
  }
  // MUST: only shapes whose tap-range arithmetic fits in i32. The WGSL computes
  // `shifted = ox + padding` in u32 and reinterprets `q = shifted / stride`, `length_in` and
  // `(kernel − 1 − r) / stride` as i32 to close the range of j, so if any of them wraps, the range
  // becomes empty or a different interval, and values that dropped contributions or read other
  // inputs are returned without an exception (e.g. kernel > 2^31 makes the end negative and only
  // the bias remains). With `length_out + padding ≤ 2^31 − 1`, shifted and q (and `q + 1`) fit in
  // i32, and the two conditions on kernel and length_in make the remaining reinterpretations fit.
  assertI32Coordinates("conv_transpose1d params", {
    kernel: dims.kernel,
    length_in: dims.lengthIn,
    "length_out + padding": dims.lengthOut + dims.padding,
  });
  assertConvFlatExtents("conv_transpose1d params", {
    input: dims.batch * dims.channelsIn * dims.lengthIn,
    output: dims.batch * dims.channelsOut * dims.lengthOut,
    weight: dims.channelsIn * dims.channelsOut * dims.kernel,
  });
  const n = dims.batch * dims.channelsOut * dims.lengthOut;
  assertU32Params("conv_transpose1d params", { n });
  const params = new Uint32Array(12);
  params[0] = n;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  return params;
};

// The check that a convolution's GPU coordinates fit in i32 (leaf functions shared by the params
// of conv1d / conv2d / conv3d / conv_transpose1d).
//
// `assertU32Params` checks that each value carried in the uniform fits in u32, but not the results
// the WGSL multiplies and adds. The conv WGSL builds input coordinates in i32, so a true coordinate
// outside the i32 range wraps and passes the out-of-range test as another, existing pixel (a wrong
// value without an exception). Writing the test per kernel splits the accepted sets of the direct
// and the implicit GEMM versions and lets one side be left unfixed (there is a precedent: only
// conv3d had it), so it lives here, in one place.

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/** The largest i32 (2^31 − 1). */
const I32_MAX = 0x7fff_ffff;

/**
 * Checks that the values entering WGSL's i32 coordinate arithmetic (and the derived values that
 * bound them) fit in the non-negative range of i32.
 *
 * MUST: fail loudly when exceeded. WGSL's `i32(u32)` reinterprets the bits, so it wraps to a
 * negative value without rounding or saturating.
 *
 * @param where The head of the failure message (e.g. `"conv1d params"`).
 * @param values Name → value. A name is a field name or a derived expression (e.g. the input
 * length + 2·padding of axis w, as {@link assertConvAxisExtent} names it).
 */
export const assertI32Coordinates = (
  where: string,
  values: Readonly<Record<string, number>>,
): void => {
  for (const [name, value] of Object.entries(values)) {
    if (value > I32_MAX) {
      throw new CodegenError(
        `${where}: ${name} ${value} が i32 の上限 2^31 − 1 を超える` +
          "（GPU の座標演算は i32 — 折り返した座標は例外を出さずに誤値を返す）",
      );
    }
  }
};

/**
 * The coordinate range of one axis of a forward convolution (conv1d / conv2d / conv3d).
 *
 * MUST: `input length + 2·padding ≤ 2^31 − 1`. The WGSL builds the input coordinate as the 32-bit
 * integer `i32(output coordinate·stride) + i32(k·dilation) − i32(padding)`. The output-length
 * formula (shapes.ts) implies `(out − 1)·stride ≤ L + 2p − d(K − 1) − 1` and
 * `k·dilation ≤ d(K − 1) ≤ L + 2p − 1`, so the coordinate lies in `[−p, L + p − 1]`, and this one
 * condition keeps every intermediate value in i32. The CPU reference's semantics do not change
 * (a subset for GPU execution — ADR 0064 axis A).
 *
 * The caller checks each value's u32 range first with `assertU32Params` in
 * src/codegen/params.ts.
 */
export const assertConvAxisExtent = (
  where: string,
  axis: string,
  length: number,
  padding: number,
): void => {
  assertI32Coordinates(where, { [`軸 ${axis} の入力長 + 2·padding`]: length + 2 * padding });
};

/**
 * The check on the flat indices of input, output and weight (each tensor's element count ≤
 * 2^32 − 1).
 *
 * MUST: the WGSL builds the indices of the three tensors as u32 flat indices (e.g.
 * `((c·T + t)·H + y)·W + x`, `(oc·Cin + ic)·K + k`), so with more than 2^32 elements the upper bits
 * drop and another position is read or written without an exception. The u32 checks on each
 * uniform value and on m / n / k do not stop it (products with the batch or channels are not in the
 * uniform — the planning-step check is also against the safe integer 2^53, not u32).
 *
 * @param extents Element counts. The output is the whole including the batch (the batch iterated
 * on the dispatch's z axis also enters the flat index).
 */
export const assertConvFlatExtents = (
  where: string,
  extents: { readonly input: number; readonly output: number; readonly weight: number },
): void => {
  assertU32Params(where, {
    入力の要素数: extents.input,
    出力の要素数: extents.output,
    重みの要素数: extents.weight,
  });
};

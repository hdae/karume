/**
 * Convolution params reject shapes "whose indices and coordinates do not fit WGSL's 32-bit
 * integers" (no GPU needed).
 *
 * Even when every uniform value fits in a u32, the products and sums on the WGSL side (the i32
 * input coordinates) can wrap around. A wrapped coordinate passes the "out of range reads 0" test
 * as another, existing pixel and yields a wrong value with no exception. The direct and the
 * implicit-GEMM variants share the same coordinate expression, so a GPU parity test cannot detect
 * this; the accepted set of params is pinned here instead.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { type Conv1dDims, conv1dIgemmParams, conv1dParams } from "../src/kernels/conv1d.ts";
import { type Conv2dDims, conv2dIgemmParams, conv2dParams } from "../src/kernels/conv2d.ts";
import { convTranspose1dParams } from "../src/kernels/conv-transpose1d.ts";
import { type Conv3dDims, conv3dIgemmParams } from "../src/kernels/conv3d.ts";

const U32_MAX = 0xffff_ffff;
// Boundary shape: length L, K 1, stride 1, dilation 1, with L + 2·padding = 2^31 − 1 + (L − 1).
// The output length follows the formula in shapes.ts (K = 1 and stride 1 give L + 2·padding).
const EDGE_PADDING = 2 ** 30 - 1;

const conv1dParamBuilders = [
  ["conv1dParams", conv1dParams],
  ["conv1dIgemmParams", conv1dIgemmParams],
] as const;

const conv2dParamBuilders = [
  ["conv2dParams", conv2dParams],
  ["conv2dIgemmParams", conv2dIgemmParams],
] as const;

const unit1d: Conv1dDims = {
  batch: 1,
  channelsIn: 1,
  channelsOut: 1,
  lengthIn: 1,
  lengthOut: 1,
  kernel: 1,
  stride: 1,
  padding: 0,
  dilation: 1,
  groups: 1,
};

const unit2d: Conv2dDims = {
  batch: 1,
  channelsIn: 1,
  channelsOut: 1,
  heightIn: 1,
  widthIn: 1,
  heightOut: 1,
  widthOut: 1,
  kernelH: 1,
  kernelW: 1,
  strideH: 1,
  strideW: 1,
  paddingH: 0,
  paddingW: 0,
  dilationH: 1,
  dilationW: 1,
  groups: 1,
};

describe("conv1d params: shapes whose input coordinates do not fit in i32", () => {
  it("rejects the shape with padding and dilation at the u32 maximum (CPU reference [14,21], GPU [29,43])", () => {
    // With input length 2, K 3 and stride 1 the output-length formula holds with
    // Lout = 2 + 2·(2^32 − 1) − (2^32 − 1)·2 = 2, but WGSL's i32(k·dilation) − i32(padding) wraps
    // and a read that should be an out-of-range 0 picks up an existing pixel.
    const dims: Conv1dDims = {
      ...unit1d,
      lengthIn: 2,
      lengthOut: 2,
      kernel: 3,
      padding: U32_MAX,
      dilation: U32_MAX,
    };
    for (const [name, build] of conv1dParamBuilders) {
      assertThrows(() => build(dims), CodegenError, "the i32 maximum", name);
    }
  });

  it("accepts input length + 2·padding = 2^31 − 1 and rejects 2^31 (same accepted set in both kernels)", () => {
    const withLength = (length: number): Conv1dDims => ({
      ...unit1d,
      lengthIn: length,
      lengthOut: length + 2 * EDGE_PADDING,
      padding: EDGE_PADDING,
    });
    for (const [name, build] of conv1dParamBuilders) {
      assertEquals(build(withLength(1)).length, 12, `${name} boundary`);
      assertThrows(
        () => build(withLength(2)),
        CodegenError,
        "the i32 maximum",
        `${name} boundary + 1`,
      );
    }
  });
});

describe("conv2d params: shapes whose input coordinates do not fit in i32", () => {
  it("rejects the shape with padding and dilation at the u32 maximum on both the H and the W axis", () => {
    const huge = (axis: "h" | "w"): Conv2dDims =>
      axis === "h"
        ? {
            ...unit2d,
            heightIn: 2,
            heightOut: 2,
            kernelH: 3,
            paddingH: U32_MAX,
            dilationH: U32_MAX,
          }
        : { ...unit2d, widthIn: 2, widthOut: 2, kernelW: 3, paddingW: U32_MAX, dilationW: U32_MAX };
    for (const [name, build] of conv2dParamBuilders) {
      for (const axis of ["h", "w"] as const) {
        assertThrows(() => build(huge(axis)), CodegenError, "the i32 maximum", `${name} ${axis}`);
      }
    }
  });

  it("accepts input length + 2·padding = 2^31 − 1 and rejects 2^31 (on both the H and the W axis)", () => {
    const withLength = (axis: "h" | "w", length: number): Conv2dDims =>
      axis === "h"
        ? {
            ...unit2d,
            heightIn: length,
            heightOut: length + 2 * EDGE_PADDING,
            paddingH: EDGE_PADDING,
          }
        : {
            ...unit2d,
            widthIn: length,
            widthOut: length + 2 * EDGE_PADDING,
            paddingW: EDGE_PADDING,
          };
    for (const [name, build] of conv2dParamBuilders) {
      for (const axis of ["h", "w"] as const) {
        build(withLength(axis, 1));
        assertThrows(
          () => build(withLength(axis, 2)),
          CodegenError,
          "the i32 maximum",
          `${name} ${axis} boundary + 1`,
        );
      }
    }
  });
});

describe("conv_transpose1d params: shapes whose valid tap range does not fit in i32", () => {
  // The contract (shapes.ts) is 2·padding == K − stride and output length = input length·stride.
  const transpose = (lengthIn: number, kernel: number, stride: number, padding: number) => ({
    batch: 1,
    channelsIn: 1,
    channelsOut: 1,
    lengthIn,
    lengthOut: lengthIn * stride,
    kernel,
    stride,
    padding,
  });

  it("rejects kernel > 2^31 (the tap end wraps negative and only the bias comes back instead of the expected 6)", () => {
    assertThrows(
      () => convTranspose1dParams(transpose(1, 2 ** 31 + 1, 1, 2 ** 30)),
      CodegenError,
      "the i32 maximum",
    );
  });

  it("accepts kernel = 2^31 − 1 and rejects 2^31", () => {
    assertEquals(convTranspose1dParams(transpose(1, 2 ** 31 - 1, 1, 2 ** 30 - 1)).length, 12);
    assertThrows(
      () => convTranspose1dParams(transpose(1, 2 ** 31, 2, 2 ** 30 - 1)),
      CodegenError,
      "the i32 maximum",
    );
  });

  it("rejects shapes where output coordinate + padding exceeds u32 (the last output reads input 0)", () => {
    // With 3·stride = 2^32 − 1 the output length fits in u32, but WGSL's ox + padding wraps to 0 at
    // 2^32.
    const stride = (2 ** 32 - 1) / 3;
    assertThrows(
      () => convTranspose1dParams(transpose(3, stride + 4, stride, 2)),
      CodegenError,
      "the i32 maximum",
    );
  });
});

describe("convolution params: shapes whose flat input, output or weight index does not fit in u32", () => {
  // 2^32 + 2^16 elements (no tensor data is allocated — only the accepted set of params is
  // checked).
  const OVER = 65537;
  const PLANE = 65536;

  it("conv3d: rejects Cin·T·H·W over 2^32 (the first index of the last channel wraps to 0)", () => {
    const unit3d: Conv3dDims = {
      channelsIn: 1,
      channelsOut: 1,
      timeIn: 1,
      heightIn: 1,
      widthIn: 1,
      timeOut: 1,
      heightOut: 1,
      widthOut: 1,
      kernelT: 1,
      kernelH: 1,
      kernelW: 1,
      strideT: 1,
      strideH: 1,
      strideW: 1,
      paddingT: 0,
      paddingH: 0,
      paddingW: 0,
      dilationT: 1,
      dilationH: 1,
      dilationW: 1,
      groups: 1,
    };
    const input = { ...unit3d, channelsIn: OVER, widthIn: PLANE, widthOut: PLANE };
    assertThrows(() => conv3dIgemmParams(input), CodegenError, "input element count");
    // 2^32 − 2^16 is accepted (the check sits at the edge of the u32 range)
    assertEquals(conv3dIgemmParams({ ...input, channelsIn: PLANE - 1 }).length, 24);
    const output = { ...unit3d, channelsOut: OVER, widthIn: PLANE, widthOut: PLANE };
    assertThrows(() => conv3dIgemmParams(output), CodegenError, "output element count");
    const weight = { ...unit3d, channelsOut: OVER, channelsIn: PLANE };
    assertThrows(() => conv3dIgemmParams(weight), CodegenError, "weight element count");
  });

  it("conv1d: rejects any of input, output (batch included) or weight over 2^32 in both the direct and the implicit-GEMM variant", () => {
    const cases: readonly (readonly [string, Conv1dDims])[] = [
      ["input element count", { ...unit1d, channelsIn: OVER, lengthIn: PLANE, lengthOut: PLANE }],
      // To keep the input small, padding stretches the output length (K 1, stride 1: Lout = 2 +
      // 2·32767)
      [
        "output element count",
        { ...unit1d, batch: OVER, lengthIn: 2, padding: 32767, lengthOut: PLANE },
      ],
      ["weight element count", { ...unit1d, channelsOut: OVER, channelsIn: PLANE }],
    ];
    for (const [name, build] of conv1dParamBuilders) {
      for (const [label, dims] of cases) {
        assertThrows(() => build(dims), CodegenError, label, `${name} ${label}`);
      }
    }
  });

  it("conv2d: rejects any of input, output (batch included) or weight over 2^32 in both the direct and the implicit-GEMM variant", () => {
    const cases: readonly (readonly [string, Conv2dDims])[] = [
      ["input element count", { ...unit2d, channelsIn: OVER, widthIn: PLANE, widthOut: PLANE }],
      [
        "output element count",
        { ...unit2d, batch: OVER, widthIn: 2, paddingW: 32767, widthOut: PLANE },
      ],
      ["weight element count", { ...unit2d, channelsOut: OVER, channelsIn: PLANE }],
    ];
    for (const [name, build] of conv2dParamBuilders) {
      for (const [label, dims] of cases) {
        assertThrows(() => build(dims), CodegenError, label, `${name} ${label}`);
      }
    }
  });

  it("conv_transpose1d: rejects input or weight over 2^32", () => {
    const base = {
      batch: 1,
      channelsIn: 1,
      channelsOut: 1,
      lengthIn: 1,
      lengthOut: 1,
      kernel: 1,
      stride: 1,
      padding: 0,
    };
    assertThrows(
      () => convTranspose1dParams({ ...base, channelsIn: OVER, lengthIn: PLANE, lengthOut: PLANE }),
      CodegenError,
      "input element count",
    );
    assertThrows(
      () => convTranspose1dParams({ ...base, channelsIn: OVER, channelsOut: PLANE }),
      CodegenError,
      "weight element count",
    );
  });
});

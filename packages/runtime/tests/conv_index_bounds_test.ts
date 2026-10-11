/**
 * 畳み込みの params が「WGSL の 32 bit 整数の添字・座標で表せない形」を拒否すること（GPU 不要）。
 *
 * uniform の各値が u32 に収まっても、WGSL 側で掛けた・足した結果（i32 の入力座標）は
 * 折り返しうる。折り返した座標は「範囲外なら 0」の判定を実在する別の画素として通り抜け、
 * 例外の出ない誤値になる。direct 版と暗黙 GEMM 版は同じ座標式を持つので GPU の parity
 * テストでは検出できず、params の受理集合をここで固定する。
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { CodegenError } from "../src/codegen/errors.ts";
import { type Conv1dDims, conv1dIgemmParams, conv1dParams } from "../src/kernels/conv1d.ts";
import { type Conv2dDims, conv2dIgemmParams, conv2dParams } from "../src/kernels/conv2d.ts";

const U32_MAX = 0xffff_ffff;
// 境界の形: 長さ L・K 1・stride 1・dilation 1 で、L + 2·padding = 2^31 − 1 + (L − 1)。
// 出力長は shapes.ts の式どおり（K = 1・stride 1 なので L + 2·padding）に揃える。
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

describe("conv1d の params: 入力座標が i32 に収まらない形", () => {
  it("padding と dilation が u32 の最大の形（CPU 参照 [14,21] に対し GPU は [29,43]）を拒否する", () => {
    // 入力長 2・K 3・stride 1 で出力長の式は Lout = 2 + 2·(2^32 − 1) − (2^32 − 1)·2 = 2 で成り立つが、
    // WGSL の i32(k·dilation) − i32(padding) が折り返して範囲外 0 のはずの読みが実在の画素を拾う。
    const dims: Conv1dDims = {
      ...unit1d,
      lengthIn: 2,
      lengthOut: 2,
      kernel: 3,
      padding: U32_MAX,
      dilation: U32_MAX,
    };
    for (const [name, build] of conv1dParamBuilders) {
      assertThrows(() => build(dims), CodegenError, "i32 の上限", name);
    }
  });

  it("入力長 + 2·padding = 2^31 − 1 は受理し、2^31 は拒否する（両カーネルで同じ受理集合）", () => {
    const withLength = (length: number): Conv1dDims => ({
      ...unit1d,
      lengthIn: length,
      lengthOut: length + 2 * EDGE_PADDING,
      padding: EDGE_PADDING,
    });
    for (const [name, build] of conv1dParamBuilders) {
      assertEquals(build(withLength(1)).length, 12, `${name} 境界`);
      assertThrows(() => build(withLength(2)), CodegenError, "i32 の上限", `${name} 境界 + 1`);
    }
  });
});

describe("conv2d の params: 入力座標が i32 に収まらない形", () => {
  it("padding と dilation が u32 の最大の形を H 軸・W 軸のどちらでも拒否する", () => {
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
        assertThrows(() => build(huge(axis)), CodegenError, "i32 の上限", `${name} ${axis}`);
      }
    }
  });

  it("入力長 + 2·padding = 2^31 − 1 は受理し、2^31 は拒否する（H 軸・W 軸とも）", () => {
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
          "i32 の上限",
          `${name} ${axis} 境界 + 1`,
        );
      }
    }
  });
});

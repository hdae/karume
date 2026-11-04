/**
 * The params of the GEMM family and of deform_conv2d reject shapes whose u32 flat indices wrap
 * (no GPU).
 *
 * Every uniform value fitting in u32 is not enough: the WGSL multiplies them into flat indices
 * (`abase + arow·k + col`, `wid.z · m · k`, `(b·Cin + ic)·H·W + …`), and a product past 2^32 − 1
 * drops its upper bits and reads or writes another element without an exception. No GPU test sees
 * it (the buffers would be tens of GiB), so the accepted sets are pinned here, each at its edge:
 * an element count of 2^32 − 1 passes and 2^32 fails.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { bmmParams, bmmRowWindowParams } from "../src/kernels/bmm.ts";
import { deformConv2dParams } from "../src/kernels/deform-conv2d.ts";
import { linearParams } from "../src/kernels/linear.ts";
import { matmulParams } from "../src/kernels/matmul.ts";

/** 65535 · 65537 = 2^32 − 1, the largest element count a u32 flat index can address. */
const EDGE_LOW = 65_535;
const EDGE_HIGH = 65_537;
/** 65536 · 65536 = 2^32, one past it. */
const OVER = 65_536;

describe("matmulParams / linearParams: the operands of one matrix", () => {
  it("rejects matmul's A when M·K = 2^32 and accepts M·K = 2^32 − 1", () => {
    assertThrows(() => matmulParams(OVER, 1, OVER), CodegenError, "m * k");
    assertEquals([...matmulParams(EDGE_LOW, 1, EDGE_HIGH)], [EDGE_LOW, 1, EDGE_HIGH, 0]);
  });

  it("rejects matmul's B when K·N = 2^32 and accepts K·N = 2^32 − 1", () => {
    assertThrows(() => matmulParams(1, OVER, OVER), CodegenError, "k * n");
    assertEquals([...matmulParams(1, EDGE_LOW, EDGE_HIGH)], [1, EDGE_LOW, EDGE_HIGH, 0]);
  });

  it("rejects matmul's C when M·N = 2^32 and accepts M·N = 2^32 − 1", () => {
    assertThrows(() => matmulParams(OVER, OVER, 1), CodegenError, "m * n");
    assertEquals([...matmulParams(EDGE_LOW, EDGE_HIGH, 1)], [EDGE_LOW, EDGE_HIGH, 1, 0]);
  });

  it("rejects linear's activation when m·k = 2^32 although the weight n·k fits", () => {
    assertThrows(() => linearParams(OVER, 1, OVER), CodegenError, "m * k");
    assertEquals([...linearParams(EDGE_LOW, 1, EDGE_HIGH)], [EDGE_LOW, 1, EDGE_HIGH, 0]);
  });
});

describe("bmmParams: the flat indices multiplied by the batch", () => {
  // One matrix is 256 × 256 = 2^16 elements, so 65536 batches reach 2^32 exactly.
  it("rejects batch·M·K = 2^32 although one matrix's M·K is only 2^16", () => {
    assertThrows(() => bmmParams(256, 1, 256, OVER), CodegenError, "A elements");
    assertEquals([...bmmParams(256, 1, 256, OVER - 1)], [256, 1, 256, 0]);
  });

  it("rejects batch·K·N = 2^32", () => {
    assertThrows(() => bmmParams(1, 256, 256, OVER), CodegenError, "B elements");
    assertEquals([...bmmParams(1, 256, 256, OVER - 1)], [1, 256, 256, 0]);
  });

  it("rejects batch·M·N = 2^32 (the output side)", () => {
    assertThrows(() => bmmParams(256, 256, 1, OVER), CodegenError, "C elements");
    assertEquals([...bmmParams(256, 256, 1, OVER - 1)], [256, 256, 1, 0]);
  });

  it("rejects a batch that is not a u32 integer", () => {
    assertThrows(() => bmmParams(1, 1, 1, 1.5), CodegenError, "batch");
  });
});

describe("bmmRowWindowParams: the windowed side strides by the full M", () => {
  // A one-row block of a 65536-row matrix with K = 65536: the windowed side is 2^32 elements,
  // the block itself only 2^16.
  it("rejects the A window when batch·rows_full·K = 2^32, and accepts the same numbers on the C side", () => {
    assertThrows(() => bmmRowWindowParams(1, 1, OVER, 0, OVER, "a", 1), CodegenError, "A elements");
    assertEquals(bmmRowWindowParams(1, 1, OVER, 0, OVER, "c", 1).length, 8);
  });

  it("rejects the C window when batch·rows_full·N = 2^32, and accepts the same numbers on the A side", () => {
    assertThrows(() => bmmRowWindowParams(1, OVER, 1, 0, OVER, "c", 1), CodegenError, "C elements");
    assertEquals(bmmRowWindowParams(1, OVER, 1, 0, OVER, "a", 1).length, 8);
  });

  it("multiplies the batch into the windowed side", () => {
    assertThrows(
      () => bmmRowWindowParams(1, 1, 256, 0, 256, "a", OVER),
      CodegenError,
      "A elements",
    );
    assertEquals(bmmRowWindowParams(1, 1, 256, 0, 256, "a", OVER - 1).length, 8);
  });
});

describe("deformConv2dParams: input / weight / offset", () => {
  const unit = {
    batch: 1,
    channelsIn: 1,
    channelsOut: 1,
    heightIn: 1,
    widthIn: 1,
    heightOut: 1,
    widthOut: 1,
    kernelH: 1,
    kernelW: 1,
    paddingH: 0,
    paddingW: 0,
  };

  it("rejects an input of B·Cin·H·W = 2^32 and accepts 2^32 − 1", () => {
    assertThrows(
      () => deformConv2dParams({ ...unit, channelsIn: OVER, heightIn: OVER }),
      CodegenError,
      "input element count",
    );
    assertEquals(
      deformConv2dParams({ ...unit, channelsIn: EDGE_LOW, heightIn: EDGE_HIGH }).length,
      16,
    );
  });

  it("rejects a weight of Cout·Cin·Kh·Kw = 2^32 and accepts 2^32 − 1", () => {
    assertThrows(
      () => deformConv2dParams({ ...unit, channelsIn: OVER, kernelH: OVER }),
      CodegenError,
      "weight element count",
    );
    assertEquals(
      deformConv2dParams({ ...unit, channelsIn: EDGE_LOW, kernelH: EDGE_HIGH }).length,
      16,
    );
  });

  it("rejects an offset of B·2·Kh·Kw·Hout·Wout = 2^32 although the output is only 2^31", () => {
    // Output = Hout·Wout = 2^31 (fits); offset = 2·2^31 = 2^32.
    assertThrows(
      () => deformConv2dParams({ ...unit, heightOut: OVER, widthOut: OVER / 2 }),
      CodegenError,
      "offset element count",
    );
    // One output row fewer: offset 2·(2^31 − 2^15) fits.
    assertEquals(
      deformConv2dParams({ ...unit, heightOut: OVER - 1, widthOut: OVER / 2 }).length,
      16,
    );
  });
});

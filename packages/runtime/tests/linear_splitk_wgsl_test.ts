/**
 * The split-K partial's WGSL and key (`src/kernels/linear-splitk.ts`, the `splitSlices` option of
 * src/kernels/gemm.ts), without a device: it is the linear GEMM of the same input with the K tile
 * loop cut to one slice's run and the epilogue storing partials without the bias — the fills and the
 * inner-product loop are the GEMM's, byte for byte, on every tile geometry the linear GEMM takes;
 * its key tells it and its slice count from the plain GEMM; a request it does not build fails at
 * the generator. The snapshots (`tests/fixtures/wgsl/linear_splitk_*.wgsl`) pin the bytes.
 */

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import { gemmWgsl } from "../src/kernels/gemm.ts";
import { linearKey, linearWgsl } from "../src/kernels/linear.ts";
import {
  linearSplitKGemmKey,
  linearSplitKGemmWgsl,
  linearSplitKParams,
} from "../src/kernels/linear-splitk.ts";
import { linearGemmGeometries } from "./helpers/geometry.ts";

const PLAIN_LOOP = "  for (var t = 0u; t < tiles; t = t + 1u) {\n";
const SPLIT_LOOP = "  for (var t = t_first; t < t_end; t = t + 1u) {\n";
/** The end of the K tile loop: its second barrier, then the loop's brace. */
const LOOP_END = "    workgroupBarrier();\n  }\n";

/** The K tile loop's body (the fills and the inner-product loop) after the loop header. */
const loopBody = (wgsl: string, header: string): string => {
  const start = wgsl.indexOf(header);
  assert(start >= 0 && wgsl.indexOf(header, start + 1) < 0, "the loop header is not unique");
  const end = wgsl.indexOf(LOOP_END, start);
  assert(end > start, "the loop end is missing");
  return wgsl.slice(start + header.length, end + LOOP_END.length);
};

/** Every (geometry, vec4) pair a linear GEMM is built at. */
const VARIANTS = linearGemmGeometries().flatMap((geometry) =>
  [false, true].map((v4) => ({ geometry, v4 })),
);

describe("the split-K partial's WGSL", () => {
  it("runs the linear GEMM's fills and inner-product loop byte for byte on every tile geometry, scalar and vec4", () => {
    assert(VARIANTS.length >= 20, `${VARIANTS.length} variants`);
    for (const { geometry, v4 } of VARIANTS) {
      const where = linearKey("f32", v4, "f32", undefined, undefined, geometry);
      const plain = linearWgsl("f32", v4, "f32", undefined, undefined, geometry);
      for (const slices of [1, 2, 16]) {
        const split = linearSplitKGemmWgsl("f32", v4, slices, undefined, undefined, geometry);
        assertEquals(loopBody(split, SPLIT_LOOP), loopBody(plain, PLAIN_LOOP), where);
        assertEquals(split.includes(PLAIN_LOOP), false, where);
      }
    }
  });

  it("deals the K tiles out by the baked slice count, from wid.z and the uniform only", () => {
    const split = linearSplitKGemmWgsl("f32", true, 8);
    assert(split.includes("  let slice_tiles = tiles / 8u;\n  let slice_extra = tiles % 8u;\n"));
    assert(split.includes("  let t_first = wid.z * slice_tiles + min(wid.z, slice_extra);\n"));
    assert(
      split.includes(
        "  let t_end = t_first + slice_tiles + select(0u, 1u, wid.z < slice_extra);\n",
      ),
    );
    // The accumulators start from 0 in every slice (the bias is the combine's).
    assert(split.includes("  var acc0_0 = vec4<f32>(0.0);"));
  });

  it("stores the accumulators at the slice's base of partials, without the bias, and declares no bias", () => {
    for (const v4 of [false, true]) {
      const split = linearSplitKGemmWgsl("f32", v4, 4);
      assertEquals(split.includes("bias["), false, `v4=${v4}`);
      assertEquals(split.includes("biasv"), false, `v4=${v4}`);
      assertEquals(split.includes("bias: array"), false, `v4=${v4}`);
      assertEquals(split.includes("out["), false, `v4=${v4}`);
      assert(
        split.includes("@group(0) @binding(4) var<storage, read_write> partials: array<"),
        `v4=${v4}`,
      );
      assertEquals(split.includes("@binding(3)"), false, `v4=${v4}`);
      assert(split.includes(`  let pbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};\n`));
      assert(
        v4
          ? split.includes("      partials[pbase + orow0 * n4 + ocq0] = acc0_0;")
          : split.includes("    let obase = pbase + orow0 * dims.n;"),
        `v4=${v4}`,
      );
    }
  });

  it("generates the same WGSL for the same input, and the plain linear does not move", () => {
    for (const { geometry, v4 } of VARIANTS) {
      assertEquals(
        linearSplitKGemmWgsl("f32", v4, 4, undefined, undefined, geometry),
        linearSplitKGemmWgsl("f32", v4, 4, undefined, undefined, geometry),
      );
      // The option omitted is the plain kernel (the snapshots pin its bytes).
      assertEquals(
        gemmWgsl({ op: "linear", v4, weight: "f32", geometry }),
        linearWgsl("f32", v4, "f32", undefined, undefined, geometry),
      );
    }
  });
});

describe("the split-K partial's key", () => {
  it("is apart from the plain linear GEMM's, carries the geometry, form and storage parts, and one key per slice count", () => {
    for (const { geometry, v4 } of VARIANTS) {
      const plain = linearKey("f32", v4, "f32", undefined, undefined, geometry);
      const tail = plain.slice("linear:v2:f32:".length);
      const keys = [1, 2, 4, 8, 16].map((slices) =>
        linearSplitKGemmKey("f32", v4, slices, undefined, undefined, geometry),
      );
      for (const [index, key] of keys.entries()) {
        assertEquals(key, `linear_splitk_gemm:v1:f32:${tail}:s${[1, 2, 4, 8, 16][index]}`);
        assertNotEquals(key, plain);
      }
      assertEquals(new Set(keys).size, keys.length);
    }
  });
});

describe("the split-K partial refuses what it does not build", () => {
  it("fails loudly for f16 compute, for a storage other than f32, and for a slice count that is not a positive integer", () => {
    assertThrows(
      () => gemmWgsl({ op: "linear", v4: true, weight: "f32", compute: "f16", splitSlices: 4 }),
      CodegenError,
      "linear split-K: only f32 weight storage under f32 compute is built",
    );
    assertThrows(
      () => linearSplitKGemmWgsl("bf16", true, 4),
      CodegenError,
      "linear split-K: only f32 weight storage under f32 compute is built",
    );
    for (const slices of [0, -2, 1.5, Number.NaN]) {
      assertThrows(
        () => linearSplitKGemmWgsl("f32", true, slices),
        CodegenError,
        "linear split-K: the slice count must be a positive integer",
      );
    }
  });

  it("fails loudly on params whose indices would wrap in u32", () => {
    assertEquals(Array.from(linearSplitKParams(3, 5, 512, 2)), [3, 5, 512, 2]);
    // The activation index m · k, the weight index n · k and the partials' slices · m · n.
    assertThrows(() => linearSplitKParams(65536, 4, 65536, 2), CodegenError, "m * k");
    assertThrows(() => linearSplitKParams(4, 65536, 65536, 2), CodegenError, "n * k");
    assertThrows(() => linearSplitKParams(65536, 32768, 16, 2), CodegenError, "partials");
    assertThrows(() => linearSplitKParams(4, 4, 512, 0), CodegenError, "0 slices");
  });
});

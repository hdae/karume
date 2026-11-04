/**
 * The CPU references of `state_append` and readonly attention check their dimensions for integers
 * before the range comparison (no GPU), as the states-form entry does.
 *
 * Every comparison with NaN is false, so a bare `dim < 1` lets NaN through, and a fraction passes
 * whenever the products of the dimensions happen to be integers. Either then returns another
 * number (or fails far away on an array length) instead of naming a shape error.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ReferenceOpError } from "../src/reference/ops.ts";
import {
  referenceStateAppend,
  referenceStateAttentionReadonly,
} from "../src/reference/state-attention.ts";

describe("referenceStateAppend: dimensions", () => {
  const valid = {
    kvPlanes: 1,
    chunkRows: 2,
    depth: 2,
    capacity: 4,
    window: 0,
    past: 0,
    query: 2,
    x: new Float32Array([1, 2, 3, 4]),
    slot: new Float32Array(8),
  };

  it("writes the first Q rows of a valid shape", () => {
    assertEquals([...referenceStateAppend(valid).data], [1, 2, 3, 4, 0, 0, 0, 0]);
  });

  for (const name of ["kvPlanes", "chunkRows", "depth", "capacity"] as const) {
    for (const bad of [Number.NaN, 1.5]) {
      it(`rejects ${name} = ${bad} as a shape error`, () => {
        assertThrows(
          () => referenceStateAppend({ ...valid, [name]: bad }),
          ReferenceOpError,
          "state_append shape is not all positive integers",
        );
      });
    }
  }
});

describe("referenceStateAttentionReadonly: dimensions", () => {
  const valid = {
    batch: 1,
    heads: 1,
    kvHeads: 1,
    depth: 2,
    capacity: 2,
    window: 0,
    past: 1,
    q: new Float32Array([1, 0]),
    slotK: new Float32Array([1, 0, 0, 0]),
    slotV: new Float32Array([3, 4, 0, 0]),
    scale: 1,
  };

  it("reads the past columns of a valid shape", () => {
    // One live column: its softmax weight is 1, so the output is that column's v.
    assertEquals([...referenceStateAttentionReadonly(valid).data], [3, 4]);
  });

  for (const name of ["batch", "heads", "kvHeads", "depth", "capacity"] as const) {
    for (const bad of [Number.NaN, 1.5]) {
      it(`rejects ${name} = ${bad} as a shape error`, () => {
        assertThrows(
          () => referenceStateAttentionReadonly({ ...valid, [name]: bad }),
          ReferenceOpError,
          "readonly shape is not all positive integers",
        );
      });
    }
  }
});

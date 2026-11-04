// Expected bit patterns from the official Torch CPU build. Both the table-driven form and the
// independent CPU reference are compared against the same oracle.
// The real-GPU comparison is gpu_static_quantize_test.ts (it shares the same oracle through
// helpers).
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { assertNodeContract } from "../src/ops.ts";
import { referenceStaticQuantize, refTensor } from "../src/reference/ops.ts";
import { bits, cases } from "./helpers/static-quantize-oracle.ts";

test("static_quantize's independent CPU reference is bit-identical to the official 24,416 boundary and special-value inputs", async () => {
  const fixture = await cases();
  assertEquals(fixture.length, 30);
  assertEquals(
    fixture.reduce((sum, c) => sum + c.x.length, 0),
    24416,
  );
  for (const c of fixture) {
    const before = bits(c.x).slice();
    const result = referenceStaticQuantize(refTensor([c.x.length], c.x), { scale: c.scale });
    assert(result.dtype === "f32");
    assertEquals(bits(result.data), c.y, `scale=${c.scale}`);
    assertEquals(bits(c.x), before, "does not modify the input");
  }
});

test("static_quantize does not round scale implicitly and rejects invalid attributes and dtypes", () => {
  for (const scale of [-1, NaN, Infinity, -Infinity, 0.1, 1e-50, 1e39, undefined, true, "1"]) {
    assertThrows(() =>
      assertNodeContract(
        {
          op: "static_quantize",
          ins: ["x"],
          outs: ["y"],
          attrs: { scale },
          states: {},
        },
        "test",
      ),
    );
  }
  assertThrows(() => referenceStaticQuantize(refTensor([1], Int32Array.of(1)), { scale: 1 }));
  for (const scale of [-1, NaN, Infinity, 0.1, 1e-50, 1e39]) {
    assertThrows(() => staticQuantizeParams(1, scale));
  }
  for (const count of [-1, 0.5, 2 ** 32]) assertThrows(() => staticQuantizeParams(count, 1));
  const payload = Uint32Array.of(0x80000000, 0x7fa00001, 0xffa00002);
  const identity = referenceStaticQuantize(refTensor([3], new Float32Array(payload.buffer)), {
    scale: -0,
  });
  assert(identity.dtype === "f32");
  assertEquals(bits(identity.data), payload);
});

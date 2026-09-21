// 公式 Torch CPU の期待ビット列。表方式と独立した CPU 参照の両方を同じ oracle に突き合わせる。
// 実 GPU 突合は gpu_static_quantize_test.ts（同じ oracle を helpers 経由で共有する）。
import { assert, assertEquals, assertThrows } from "@std/assert";
import { staticQuantizeParams } from "../src/kernels/static-quantize.ts";
import { assertNodeContract } from "../src/ops.ts";
import { referenceStaticQuantize, refTensor } from "../src/reference/ops.ts";
import { bits, cases } from "./helpers/static-quantize-oracle.ts";

Deno.test("static_quantize の独立 CPU 参照は公式の境界・特殊値 24,416 入力とビット一致する", async () => {
  const fixture = await cases();
  assertEquals(fixture.length, 30);
  assertEquals(fixture.reduce((sum, c) => sum + c.x.length, 0), 24416);
  for (const c of fixture) {
    const before = bits(c.x).slice();
    const result = referenceStaticQuantize(refTensor([c.x.length], c.x), { scale: c.scale });
    assert(result.dtype === "f32");
    assertEquals(bits(result.data), c.y, `scale=${c.scale}`);
    assertEquals(bits(c.x), before, "入力を変更しない");
  }
});

Deno.test("static_quantize は scale を暗黙に丸めず不正属性・dtype を拒否する", () => {
  for (const scale of [-1, NaN, Infinity, -Infinity, 0.1, 1e-50, 1e39, undefined, true, "1"]) {
    assertThrows(() =>
      assertNodeContract({
        op: "static_quantize",
        ins: ["x"],
        outs: ["y"],
        attrs: { scale },
        states: {},
      }, "test")
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

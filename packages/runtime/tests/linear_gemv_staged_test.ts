import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  linearGemvPackedEligible,
  linearGemvParallelKey,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvStagedEligible,
} from "../src/kernels/linear-gemv.ts";

describe("linearGemvStagedEligible (K-90)", () => {
  it("marks the lanes-32 rows with K ≥ 2048 that measured faster, and the E4B rows with the same role", () => {
    for (const [n, k] of [
      [1536, 2048],
      [1536, 4096],
      [1536, 6144],
      [1536, 12288],
      [2560, 2048],
      [2560, 4096],
      [2560, 10240],
    ]) {
      for (const m of [1, 4, 8])
        assertEquals(linearGemvStagedEligible("i4", m, n, k, 32), 32, `${n}×${k} M${m}`);
    }
  });

  it("leaves the rows that measured slower or were not measured on the plain variant", () => {
    // E2B k / v (K = 1536, ×0.86 staged) and the lanes-4 rows (×0.86–1.06)
    assertEquals(linearGemvStagedEligible("i4", 1, 256, 1536, 32), undefined);
    assertEquals(linearGemvStagedEligible("i4", 1, 12288, 1536, 32), undefined);
    assertEquals(linearGemvStagedEligible("i4", 1, 2048, 1536, 32), undefined);
    // E4B k / v copy the E2B k / v row
    assertEquals(linearGemvStagedEligible("i4", 1, 512, 2560, 32), undefined);
  });

  it("never marks a packed-activation row (ADR 0105) or a prefill M", () => {
    assert(linearGemvPackedEligible("i4", 1, 1536, 2048, 2048) !== undefined);
    assertEquals(linearGemvStagedEligible("i4", 1, 1536, 2048, 2048), undefined);
    assertEquals(linearGemvStagedEligible("i4", 9, 1536, 2048, 32), undefined);
  });
});

describe("the staged-activation parallel GEMV's key and WGSL", () => {
  it("appends a fragment to the parallel key, so storage and lane discriminators stay where they are", () => {
    assertEquals(
      linearGemvParallelStagedKey("i4", 32, 32),
      `${linearGemvParallelKey("i4", 32, 32)}:staged-x`,
    );
  });

  it("keeps the plain variant's multiply-add lines and reads every activation quad from workgroup memory", () => {
    const plain = linearGemvParallelWgsl("i4", 32, 32);
    const staged = linearGemvParallelStagedWgsl("i4", 32, 32);
    const macs = (code: string) => code.split("\n").filter((line) => line.includes("acc = fma("));
    assertEquals(macs(staged), macs(plain));
    assertEquals(staged.match(/= sx\[/g)?.length, 8);
    assert(!/= x\[xq/.test(staged));
    assertStringIncludes(staged, "var<workgroup> sx: array<vec4<f32>, 256u>;");
    assertEquals(
      staged.match(/workgroupBarrier\(\)/g)?.length,
      (plain.match(/workgroupBarrier\(\)/g)?.length ?? 0) + 2,
    );
  });
});

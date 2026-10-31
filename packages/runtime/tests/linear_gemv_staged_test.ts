import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  type LinearGemvParallelLanes,
  linearGemvPackedEligible,
  linearGemvPackedStagedEligible,
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvParallelPackedStagedKey,
  linearGemvParallelPackedStagedWgsl,
  linearGemvParallelPackedWgsl,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvStagedEligible,
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
  linearGemvStaticQuantizePackedStagedKey,
  linearGemvStaticQuantizePackedStagedWgsl,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeStagedKey,
  linearGemvStaticQuantizeStagedWgsl,
  linearGemvStaticQuantizeWgsl,
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

describe("linearGemvPackedStagedEligible", () => {
  it("never stages a shape that does not take packed activations", () => {
    assertEquals(linearGemvPackedEligible("i4", 1, 1536, 2048, 32), undefined);
    assertEquals(linearGemvPackedStagedEligible("i4", 1, 1536, 2048, 32), undefined);
    assertEquals(linearGemvPackedStagedEligible("i4", 9, 1536, 2048, 2048), undefined);
  });

  it("leaves every packed row on the unstaged packed body until the row is measured", () => {
    for (const [storage, n, k, group] of [
      ["i4", 1536, 2048, 2048],
      ["i4", 1536, 6144, 2048],
      ["i4", 1536, 4096, 4096],
      ["i2", 1536, 12288, undefined],
      ["i4", 2560, 10240, 2048],
    ] as const) {
      assert(linearGemvPackedEligible(storage, 1, n, k, group) !== undefined);
      assertEquals(linearGemvPackedStagedEligible(storage, 1, n, k, group), undefined);
    }
  });
});

type Body = {
  readonly name: string;
  readonly unstaged: (
    storage: "i2" | "i4" | "i8",
    group: number | undefined,
    lanes: LinearGemvParallelLanes,
  ) => string;
  readonly staged: (
    storage: "i2" | "i4" | "i8",
    group: number | undefined,
    lanes: LinearGemvParallelLanes,
  ) => string;
  readonly packed: boolean;
};

const BODIES: readonly Body[] = [
  {
    name: "SRQ-fused",
    unstaged: linearGemvStaticQuantizeWgsl,
    staged: linearGemvStaticQuantizeStagedWgsl,
    packed: false,
  },
  {
    name: "packed",
    unstaged: linearGemvParallelPackedWgsl,
    staged: linearGemvParallelPackedStagedWgsl,
    packed: true,
  },
  {
    name: "SRQ-fused packed",
    unstaged: linearGemvStaticQuantizePackedWgsl,
    staged: linearGemvStaticQuantizePackedStagedWgsl,
    packed: true,
  },
];

describe("the staged forms of the SRQ-fused and packed parallel GEMVs", () => {
  it("append the staged fragment last, so the SRQ and packed discriminators stay where they are", () => {
    assertEquals(
      linearGemvStaticQuantizeStagedKey("i4", 512, 32),
      `${linearGemvStaticQuantizeKey("i4", 512, 32)}:staged-x`,
    );
    assertEquals(
      linearGemvParallelPackedStagedKey("i2", undefined, 32),
      `${linearGemvParallelPackedKey("i2", undefined, 32)}:staged-x`,
    );
    assertEquals(
      linearGemvStaticQuantizePackedStagedKey("i4", 2048, 32),
      `${linearGemvStaticQuantizePackedKey("i4", 2048, 32)}:staged-x`,
    );
  });

  for (const body of BODIES) {
    it(`keep the ${body.name} body's multiply-adds, restoration and store, reading the activation only from workgroup memory`, () => {
      for (const [storage, group] of [
        ["i2", undefined],
        ["i4", 512],
        ["i8", undefined],
      ] as const) {
        for (const lanes of [2, 4, 32] satisfies LinearGemvParallelLanes[]) {
          const where = `${storage} l${lanes}`;
          const unstaged = body.unstaged(storage, group, lanes);
          const staged = body.staged(storage, group, lanes);
          const kept = (code: string) =>
            code.split("\n").filter((line) => /acc = fma\(|unpack4xI8\(xp|out\[wg\.y/.test(line));
          assertEquals(kept(staged), kept(unstaged), where);
          assert(!/= x\[xq/.test(staged), where);
          assertStringIncludes(
            staged,
            `var<workgroup> sx: array<vec4<${body.packed ? "u32" : "f32"}>`,
            where,
          );
        }
      }
    });
  }
});

// The check of geometryProfileMismatch, the matching of a saved table (src/tune/mismatch.ts — ADR
// 0117 decision 5). The public functions come from the public surface (../tune.ts), the unexported
// fingerprint (geometryProfileKernelsId) from src. No GPU is used (the fingerprint is derived from
// codegen alone).
//
// What this pins: ① undefined when everything matches ② a missing provenance and a mismatch of each
// of the adapter's 4 fields (the empty string is a value too), the fingerprint and the case-set
// version are each returned with wording that names that field ③ the matching order (only the first
// mismatch is returned) ④ it does not throw (a table whose geometry cannot build the kernels of the
// sweep shapes is also returned as a message).

import { platformObject } from "@karume/test-support/webgpu";
import { assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { geometryProfileMismatch, sweepCaseSetId } from "../tune.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { geometryProfileKernelsId } from "../src/tune/fingerprint.ts";

type Provenance = NonNullable<GeometryProfile["provenance"]>;
type Adapter = Provenance["adapter"];

const ADAPTER: Adapter = {
  vendor: "apple",
  architecture: "metal-3",
  device: "0x0000",
  description: "Apple M2",
};

/**
 * A table whose matching inputs are the current runtime's values (`overrides` replaces provenance
 * fields).
 */
const stored = (
  table: GeometryProfile = DEFAULT_GEOMETRY_PROFILE,
  overrides: Partial<Provenance> = {},
): GeometryProfile & { readonly provenance: Provenance } => ({
  ...table,
  id: "stored",
  provenance: {
    sweep: "sweep.json",
    sha256: "sha",
    date: "2026-10-02T00:00:00.000Z",
    candidateSet: "quick+",
    adapter: ADAPTER,
    kernels: geometryProfileKernelsId(table),
    caseSet: sweepCaseSetId(),
    ...overrides,
  },
});

/** Asserts that matching returns a mismatch (fails on undefined). */
const mismatchOf = (profile: GeometryProfile, adapter: Adapter): string => {
  const message = geometryProfileMismatch(profile, adapter);
  if (message === undefined) throw new Error("judged as a match");
  return message;
};

describe("geometryProfileMismatch", () => {
  it("is undefined when the adapter's 4 fields, the kernel fingerprint and the case-set version all match", () => {
    assertEquals(geometryProfileMismatch(stored(), ADAPTER), undefined);
    // Other GPUAdapterInfo fields (the subgroup sizes and so on) are ignored —
    // GpuContext.adapterInfo can be passed as it is
    const info: GPUAdapterInfo = platformObject<GPUAdapterInfo>({
      ...ADAPTER,
      subgroupMinSize: 4,
      subgroupMaxSize: 64,
      isFallbackAdapter: false,
    });
    assertEquals(geometryProfileMismatch(stored(), info), undefined);
  });

  it("reports that a table without provenance (a hand-written table) has nothing to match against", () => {
    const message = mismatchOf(DEFAULT_GEOMETRY_PROFILE, ADAPTER);
    assertStringIncludes(message, "'default' has no provenance");
    assertStringIncludes(message, "nothing to match against");
  });

  for (const field of ["vendor", "architecture", "device", "description"] as const) {
    it(`names the field and both values when the adapter's ${field} differs (the empty string compares as a value)`, () => {
      const blank = mismatchOf(stored(), { ...ADAPTER, [field]: "" });
      assertStringIncludes(blank, `adapter ${field} differs`);
      assertStringIncludes(blank, `table ${JSON.stringify(ADAPTER[field])} / this adapter ""`);
      // Also a mismatch when the table's side is empty (a table made in Chrome without the flag)
      // and the adapter returns a value (the flag is on)
      const filled = mismatchOf(
        stored(DEFAULT_GEOMETRY_PROFILE, { adapter: { ...ADAPTER, [field]: "" } }),
        ADAPTER,
      );
      assertStringIncludes(filled, `adapter ${field} differs`);
      // Case is not normalised either (exact string match)
      assertStringIncludes(
        mismatchOf(stored(), { ...ADAPTER, [field]: ADAPTER[field].toUpperCase() + "!" }),
        `adapter ${field} differs`,
      );
    });
  }

  it("fields that are both empty match (made in Chrome without the flag, matched in the same Chrome)", () => {
    const blank = { ...ADAPTER, device: "", description: "" };
    assertEquals(
      geometryProfileMismatch(stored(DEFAULT_GEOMETRY_PROFILE, { adapter: blank }), blank),
      undefined,
    );
  });

  it("reports a kernel fingerprint that differs from the current runtime's (also when the table's geometry was edited after saving)", () => {
    const message = mismatchOf(
      stored(DEFAULT_GEOMETRY_PROFILE, { kernels: "0123456789abcdef" }),
      ADAPTER,
    );
    assertStringIncludes(message, "kernel fingerprint differs");
    assertStringIncludes(message, "table 0123456789abcdef");
    assertStringIncludes(
      message,
      `current runtime ${geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE)}`,
    );
    const edited: GeometryProfile = {
      ...stored(),
      attention: {
        ...DEFAULT_GEOMETRY_PROFILE.attention,
        qk: { regM: 4, regN: 4, wgX: 8, wgY: 16 },
      },
    };
    assertStringIncludes(mismatchOf(edited, ADAPTER), "kernel fingerprint differs");
  });

  it("reports a case-set version that differs from the current runtime's", () => {
    const message = mismatchOf(
      stored(DEFAULT_GEOMETRY_PROFILE, { caseSet: "fedcba9876543210" }),
      ADAPTER,
    );
    assertStringIncludes(message, "case-set version differs");
    assertStringIncludes(message, `table fedcba9876543210 / current runtime ${sweepCaseSetId()}`);
  });

  it("returns only the first mismatch (in the order adapter's 4 fields → fingerprint → case-set version)", () => {
    const all = stored(DEFAULT_GEOMETRY_PROFILE, {
      kernels: "0123456789abcdef",
      caseSet: "fedcba9876543210",
    });
    assertStringIncludes(mismatchOf(all, { ...ADAPTER, device: "0x1234" }), "adapter device");
    assertStringIncludes(mismatchOf(all, ADAPTER), "kernel fingerprint differs");
    assertStringIncludes(
      mismatchOf(
        { ...all, provenance: { ...all.provenance, kernels: geometryProfileKernelsId(all) } },
        ADAPTER,
      ),
      "case-set version differs",
    );
  });

  it("does not match the candidate set or match", () => {
    assertEquals(
      geometryProfileMismatch(
        {
          ...stored(DEFAULT_GEOMETRY_PROFILE, { candidateSet: "quick" }),
          match: { vendor: "intel" },
        },
        ADAPTER,
      ),
      undefined,
    );
  });

  it("does not match provenance.userAgent (an informational field naming the sweeping browser) — a match whether it differs or is absent", () => {
    for (const overrides of [
      { userAgent: ["Mozilla/5.0 (Macintosh) Chrome/153.0.0.0"] },
      { userAgent: ["Deno/2.9.6"] },
      {},
    ]) {
      assertEquals(
        geometryProfileMismatch(stored(DEFAULT_GEOMETRY_PROFILE, overrides), ADAPTER),
        undefined,
        JSON.stringify(overrides),
      );
    }
  });

  it("does not throw: returns the reason for a table whose geometry cannot build the sweep shapes' kernels (dispatch count over 65535)", () => {
    // It passes the check (assertGeometryProfile), but with tileN 4 the N = 262144 of conv2d 512²
    // becomes 65536 workgroups
    const narrow: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      conv2d: { ...DEFAULT_GEOMETRY_PROFILE.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
    };
    const message = mismatchOf({ ...narrow, provenance: stored().provenance }, ADAPTER);
    assertStringIncludes(message, "cannot build the kernels");
    assertStringIncludes(message, "conv2d-c96-512x512");
  });
});

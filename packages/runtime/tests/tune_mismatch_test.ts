// The check of geometryProfileMismatch, the matching of a saved table (src/tune/mismatch.ts — ADR
// 0117 decision 5, per op family since its addendum 2026-11-12). The public functions come from the
// public surface (../tune.ts), the unexported fingerprint (familyKernelsId) and the JSON writer of
// a saved table (infinityJson) from src. No GPU is used (the fingerprint is derived from codegen
// alone).
//
// What this pins: ① undefined when everything matches ② a missing provenance and a mismatch of each
// of the adapter's 4 fields (the empty string is a value too), a family's fingerprint and its
// case-set version are each returned with wording that names that field and the family ③ the
// matching order (only the first mismatch is returned; families in the fixed order) ④ a family
// without an entry matches only when its fields are the default ⑤ it does not throw (a table whose
// geometry cannot build the kernels of the sweep shapes, with or without the family's entry, and a
// table saved in the earlier shape without per-family entries are also returned as a message) ⑥
// every built-in table matches its own adapter.

import { platformObject } from "@karume/test-support/webgpu";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { geometryProfileMismatch, sweepCaseSetIds } from "../tune.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  GEOMETRY_PROFILE_FAMILIES,
  type GeometryProfile,
  type GeometryProfileFamily,
  type GeometryProfileFamilyProvenance,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { infinityJson } from "../src/tune/derive.ts";
import { familyKernelsId } from "../src/tune/fingerprint.ts";

type Provenance = NonNullable<GeometryProfile["provenance"]>;
type Adapter = Provenance["adapter"];

const ADAPTER: Adapter = {
  vendor: "apple",
  architecture: "metal-3",
  device: "0x0000",
  description: "Apple M2",
};

/**
 * One family's entry whose matching inputs are the current runtime's values for `table`
 * (`overrides` replaces fields).
 */
const entryOf = (
  table: GeometryProfile,
  family: GeometryProfileFamily,
  overrides: Partial<GeometryProfileFamilyProvenance> = {},
): GeometryProfileFamilyProvenance => ({
  sweep: "sweep.json",
  sha256: "sha",
  date: "2026-10-02T00:00:00.000Z",
  candidateSet: "quick+",
  kernels: familyKernelsId(table, family),
  caseSet: sweepCaseSetIds()[family] ?? "",
  ...overrides,
});

/**
 * A table whose matching inputs are the current runtime's values: an entry for every family
 * (`entries` replaces fields of the named families' entries), the adapter `adapter`.
 */
const stored = (
  table: GeometryProfile = DEFAULT_GEOMETRY_PROFILE,
  entries: {
    readonly [Family in GeometryProfileFamily]?: Partial<GeometryProfileFamilyProvenance>;
  } = {},
  adapter: Adapter = ADAPTER,
): GeometryProfile & { readonly provenance: Provenance } => {
  const families: { [Family in GeometryProfileFamily]?: GeometryProfileFamilyProvenance } = {};
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    families[family] = entryOf(table, family, entries[family]);
  }
  return { ...table, id: "stored", provenance: { adapter, families } };
};

/** Asserts that matching returns a mismatch (fails on undefined). */
const mismatchOf = (profile: GeometryProfile, adapter: Adapter): string => {
  const message = geometryProfileMismatch(profile, adapter);
  if (message === undefined) throw new Error("judged as a match");
  return message;
};

describe("geometryProfileMismatch", () => {
  it("is undefined when the adapter's 4 fields and every family's kernel fingerprint and case-set version match", () => {
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
        stored(DEFAULT_GEOMETRY_PROFILE, {}, { ...ADAPTER, [field]: "" }),
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
      geometryProfileMismatch(stored(DEFAULT_GEOMETRY_PROFILE, {}, blank), blank),
      undefined,
    );
  });

  it("reports a family's kernel fingerprint that differs from the current runtime's, naming the family (also when the table's geometry was edited after saving)", () => {
    const message = mismatchOf(
      stored(DEFAULT_GEOMETRY_PROFILE, { gemm: { kernels: "0123456789abcdef" } }),
      ADAPTER,
    );
    assertStringIncludes(message, "family gemm: kernel fingerprint differs");
    assertStringIncludes(message, "table 0123456789abcdef");
    assertStringIncludes(
      message,
      `current runtime ${familyKernelsId(DEFAULT_GEOMETRY_PROFILE, "gemm")}`,
    );
    assertStringIncludes(message, "sweep this family again");
    const edited: GeometryProfile = {
      ...stored(),
      attention: {
        ...DEFAULT_GEOMETRY_PROFILE.attention,
        qk: { regM: 4, regN: 4, wgX: 8, wgY: 16 },
      },
    };
    assertStringIncludes(
      mismatchOf(edited, ADAPTER),
      "family attention: kernel fingerprint differs",
    );
  });

  it("reports a family's case-set version that differs from the current runtime's, naming the family", () => {
    const message = mismatchOf(
      stored(DEFAULT_GEOMETRY_PROFILE, { "i8a8-conv3d": { caseSet: "fedcba9876543210" } }),
      ADAPTER,
    );
    assertStringIncludes(message, "family i8a8-conv3d: case-set version differs");
    assertStringIncludes(
      message,
      `table fedcba9876543210 / current runtime ${sweepCaseSetIds()["i8a8-conv3d"] ?? ""}`,
    );
  });

  it("returns only the first mismatch (in the order adapter's 4 fields → per family in the fixed order: fingerprint → case-set version)", () => {
    const all = stored(DEFAULT_GEOMETRY_PROFILE, {
      gemm: { kernels: "0123456789abcdef", caseSet: "fedcba9876543210" },
      conv2d: { kernels: "0123456789abcdef" },
    });
    assertStringIncludes(mismatchOf(all, { ...ADAPTER, device: "0x1234" }), "adapter device");
    assertStringIncludes(mismatchOf(all, ADAPTER), "family gemm: kernel fingerprint differs");
    const kernelsFixed = stored(DEFAULT_GEOMETRY_PROFILE, {
      gemm: { caseSet: "fedcba9876543210" },
      conv2d: { kernels: "0123456789abcdef" },
    });
    assertStringIncludes(
      mismatchOf(kernelsFixed, ADAPTER),
      "family gemm: case-set version differs",
    );
    // gemm matching, the next family in the order is reported
    const gemmFixed = stored(DEFAULT_GEOMETRY_PROFILE, { conv2d: { kernels: "0123456789abcdef" } });
    assertStringIncludes(
      mismatchOf(gemmFixed, ADAPTER),
      "family conv2d: kernel fingerprint differs",
    );
  });

  it("matches a family without an entry only when its fields build the default table's kernels, and names the family otherwise", () => {
    const { conv2d: _conv2d, ...others } = stored().provenance.families;
    const withoutConv2d: GeometryProfile = {
      ...stored(),
      provenance: { adapter: ADAPTER, families: others },
    };
    // The default's conv2d fields: a match (default by absence)
    assertEquals(geometryProfileMismatch(withoutConv2d, ADAPTER), undefined);
    // A conv2d geometry that is not the default, with no entry to stand for it: a mismatch
    const edited: GeometryProfile = {
      ...withoutConv2d,
      conv2d: {
        ...DEFAULT_GEOMETRY_PROFILE.conv2d,
        rows64: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
      },
    };
    const message = mismatchOf(edited, ADAPTER);
    assertStringIncludes(message, "family conv2d");
    assertStringIncludes(
      message,
      "no provenance for family conv2d but its geometry is not the default",
    );
    // Other families' entries are not read for it: the same conv2d field with an entry of its own
    // matches
    assertEquals(
      geometryProfileMismatch(
        stored({ ...DEFAULT_GEOMETRY_PROFILE, conv2d: edited.conv2d }),
        ADAPTER,
      ),
      undefined,
    );
  });

  it("does not match the candidate set or match", () => {
    assertEquals(
      geometryProfileMismatch(
        {
          ...stored(DEFAULT_GEOMETRY_PROFILE, { gemm: { candidateSet: "quick" } }),
          match: { vendor: "intel" },
        },
        ADAPTER,
      ),
      undefined,
    );
  });

  it("does not match a family's userAgent (an informational field naming the sweeping browser) — a match whether it differs or is absent", () => {
    for (const overrides of [
      { userAgent: ["Mozilla/5.0 (Macintosh) Chrome/153.0.0.0"] },
      { userAgent: ["Deno/2.9.6"] },
      {},
    ]) {
      assertEquals(
        geometryProfileMismatch(stored(DEFAULT_GEOMETRY_PROFILE, { gemm: overrides }), ADAPTER),
        undefined,
        JSON.stringify(overrides),
      );
    }
  });

  it("does not throw: returns the reason for a table whose geometry cannot build the sweep shapes' kernels (dispatch count over 65535), naming the family", () => {
    // It passes the check (assertGeometryProfile), but with tileN 4 the N = 262144 of conv2d 512²
    // becomes 65536 workgroups
    const narrow: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      conv2d: { ...DEFAULT_GEOMETRY_PROFILE.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
    };
    const message = mismatchOf({ ...narrow, provenance: stored().provenance }, ADAPTER);
    assertStringIncludes(message, "family conv2d");
    assertStringIncludes(message, "cannot build the kernels");
    assertStringIncludes(message, "conv2d-c96-512x512");
  });

  it("returns the cannot-build reason, naming the family, for a family without an entry whose geometry cannot build the sweep shapes' kernels", () => {
    const { conv2d: _conv2d, ...others } = stored().provenance.families;
    const narrow: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      id: "stored",
      conv2d: { ...DEFAULT_GEOMETRY_PROFILE.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
      provenance: { adapter: ADAPTER, families: others },
    };
    const message = mismatchOf(narrow, ADAPTER);
    assertStringIncludes(message, "Table 'stored', family conv2d");
    assertStringIncludes(message, "cannot build the kernels");
    assertStringIncludes(message, "conv2d-c96-512x512");
  });

  it("does not throw on a table saved in the earlier shape (provenance without per-family entries): returns that it has none", () => {
    const { families: _families, ...earlier } = stored().provenance;
    const text = infinityJson({
      ...DEFAULT_GEOMETRY_PROFILE,
      id: "stored",
      provenance: { ...earlier, sweep: "sweep.json", kernels: "0123456789abcdef" },
    });
    // What an application holds after JSON.parse and a cast of a table saved by an earlier runtime
    const saved: GeometryProfile = JSON.parse(text);
    const message = mismatchOf(saved, ADAPTER);
    assertStringIncludes(message, "Table 'stored': its provenance has no per-family entries");
    assertStringIncludes(message, "saved by an earlier runtime; sweep again");
    // The adapter is still compared first
    assertStringIncludes(mismatchOf(saved, { ...ADAPTER, device: "0x1234" }), "adapter device");
  });
});

describe("geometryProfileMismatch on the built-in tables", () => {
  for (const profile of BUILTIN_GEOMETRY_PROFILES) {
    it(`${profile.id} matches its own adapter on the current runtime`, () => {
      const { provenance } = profile;
      assert(provenance !== undefined, `${profile.id} has no provenance`);
      assertEquals(geometryProfileMismatch(profile, provenance.adapter), undefined);
    });
  }
});

// The check of the shared derivation (deriveGeometryProfile with `shared`, deriveSharedProfile of
// src/tune/derive.ts — SPEC: docs/spec/geometry-profiles.md#shared-tables): one table from the sweep
// records of several adapters of one vendor and architecture, matched without a description. No GPU
// is used — the sweep records are small synthetic JSON (helpers/sweep-records.ts).
//
// What this pins: (1) the rule — a geometry is taken only when it passes the measured table's rule
// on every device, on that device's own cases against that device's own default, and among those
// the largest geometric mean of the per-device geometric means (devices weigh equally, not cases);
// a class one device has no case of keeps the default and names the device; (2) the verdict lines
// show each device's ratio; (3) the provenance names every device; (4) a device is told apart by
// its description (a record without one, one description with two device ids and one description
// alone are refused); (5) the refusals (a description, attention-bench records) and that the
// measured derivation still refuses mixed adapters; (6) a shared table as the base of a
// regeneration; (7) matching and the JSON reader of the several-adapter form.

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  deriveGeometryProfile,
  geometryProfileJson,
  geometryProfileMismatch,
  type GeometrySweepInput,
  parseGeometryProfileJson,
} from "../tune.ts";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  assertGeometryProfile,
  type GeometryProfile,
  type GeometryProfileAdapter,
} from "../src/kernels/geometry-profile.ts";
import { caseShape, SWEEP_CASES } from "../src/tune/cases.ts";
import { infinityJson } from "../src/tune/derive.ts";
import { familyEntries } from "../src/tune/families.ts";
import { gemmCandidate, quickConv2dCandidates } from "../src/tune/geometries.ts";
import { casePlan } from "../src/tune/harness.ts";
import {
  A,
  B,
  BIG,
  caseRows,
  familyDefaultRows,
  linearCase,
  report,
  row,
} from "./helpers/sweep-records.ts";
import { benchRecord } from "./helpers/attention-bench-records.ts";

const M5: GeometryProfileAdapter = {
  vendor: "apple",
  architecture: "metal-3",
  device: "0x0000",
  description: "Apple M5",
};
const M2: GeometryProfileAdapter = { ...M5, description: "Apple M2" };

const SHARED = { id: "apple-shared", vendor: "apple", architecture: "metal-3", shared: true };

const input = (
  path: string,
  rows: readonly Record<string, unknown>[],
  adapter: GeometryProfileAdapter,
): GeometrySweepInput => ({ report: report(rows, adapter), path, sha256: `sha-${path}` });

/** The gemmRows rule of the bucket above 512 rows (where linear-m1024 and the like fall). */
const largestRows = (profile: GeometryProfile) => profile.gemmRows.at(-1)?.geometry;

/** The verdict line of a field and the lines under it. */
const linesOf = (verdicts: readonly string[], slot: string): string[] => {
  const start = verdicts.findIndex((line) => line.startsWith(`- ${slot} (`));
  assert(start >= 0, `no verdict line of ${slot}`);
  const end = verdicts.findIndex((line, index) => index > start && line.startsWith("- "));
  return verdicts.slice(start, end < 0 ? undefined : end);
};

describe("deriveGeometryProfile with shared: the rule", () => {
  it("does not adopt a geometry faster on one device and slower on the other, and the verdict shows each device's ratio", () => {
    const { profile, verdicts } = deriveGeometryProfile(
      [
        input("m5.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]), M5),
        input("m2.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 0.9 }]]), M2),
      ],
      SHARED,
    );
    assertEquals(largestRows(profile), BIG);
    const lines = linesOf(verdicts, "gemmRows > 512");
    assertStringIncludes(lines[0], "Apple M5 1 cases, Apple M2 1 cases");
    assertStringIncludes(lines[0], "keeps the default");
    assertStringIncludes(lines[0], "on every device");
    assertEquals(
      lines.includes(
        `  - ${gemmCandidate(A).name}: Apple M5: ×1.300 (×1.300–×1.300); Apple M2: linear-m1024: ×0.900 < ×1.050`,
      ),
      true,
      lines.join("\n"),
    );
  });

  it("adopts a geometry eligible on every device, by the geometric mean of the per-device geometric means — devices weigh equally, not cases", () => {
    // The M5 holds three cases of the class, the M2 one. Per device, A is ×1.10 on the M5 and ×2.0
    // on the M2 (√(1.1 · 2.0) ≈ ×1.483) and B ×1.5 and ×1.2 (√(1.5 · 1.2) ≈ ×1.342): A. Pooled over
    // the four cases, B would win (×1.419 against ×1.277), so the choice tells the rules apart.
    const m5 = [1024, 2048, 4096].flatMap((m) =>
      caseRows(linearCase(m), BIG, [
        [A, { speedup: 1.1 }],
        [B, { speedup: 1.5 }],
      ]),
    );
    const m2 = caseRows(linearCase(1024), BIG, [
      [A, { speedup: 2.0 }],
      [B, { speedup: 1.2 }],
    ]);
    const { profile, verdicts } = deriveGeometryProfile(
      [input("m5.json", m5, M5), input("m2.json", m2, M2)],
      SHARED,
    );
    assertEquals(largestRows(profile), A);
    const lines = linesOf(verdicts, "gemmRows > 512");
    assertEquals(
      lines[0],
      `- gemmRows > 512 (linear / matmul / bmm row count > 512; Apple M5 3 cases, Apple M2 1 cases): adopted ${gemmCandidate(A).name} ×1.483 over the devices (Apple M5 ×1.100 (×1.100–×1.100), Apple M2 ×2.000 (×2.000–×2.000))`,
    );
    assertEquals(
      lines.includes(
        `  - ${gemmCandidate(B).name}: geometric mean ×1.342 (below the adopted ×1.483) over the devices — Apple M5: ×1.500 (×1.500–×1.500); Apple M2: ×1.200 (×1.200–×1.200)`,
      ),
      true,
      lines.join("\n"),
    );
  });

  it("holds a device to every case its records have of the class: one slow case on one device rules a geometry out", () => {
    const m5 = [1024, 2048].flatMap((m) => caseRows(linearCase(m), BIG, [[A, { speedup: 1.4 }]]));
    const m2 = [
      ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.4 }]]),
      ...caseRows(linearCase(2048), BIG, [[A, { speedup: 1.02 }]]),
    ];
    const { profile, verdicts } = deriveGeometryProfile(
      [input("m5.json", m5, M5), input("m2.json", m2, M2)],
      SHARED,
    );
    assertEquals(largestRows(profile), BIG);
    assertStringIncludes(
      linesOf(verdicts, "gemmRows > 512").join("\n"),
      "Apple M2: linear-m2048: ×1.020 < ×1.050",
    );
  });

  it("judges a geometry one device did not measure as not measured there", () => {
    const { profile, verdicts } = deriveGeometryProfile(
      [
        input("m5.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]), M5),
        input("m2.json", caseRows(linearCase(1024), BIG, []), M2),
      ],
      SHARED,
    );
    assertEquals(largestRows(profile), BIG);
    assertStringIncludes(
      linesOf(verdicts, "gemmRows > 512").join("\n"),
      "Apple M2: not measured in linear-m1024",
    );
  });

  it("keeps the default for a class one device has no case of, and names that device", () => {
    const { profile, verdicts } = deriveGeometryProfile(
      [
        input(
          "m5.json",
          [
            ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]),
            ...caseRows(linearCase(128), A, [[B, { speedup: 1.3 }]]),
          ],
          M5,
        ),
        input("m2.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]), M2),
      ],
      SHARED,
    );
    assertEquals(largestRows(profile), A);
    const rows128 = profile.gemmRows.find((rule) => rule.maxRows === 128);
    assertEquals(rows128?.geometry, A);
    assertStringIncludes(
      linesOf(verdicts, "gemmRows 65〜128")[0],
      "keeps the default reg64x32r4x4w8 (no case of this class in the records of Apple M2 (a shared table takes a geometry only for a class every device measured))",
    );
  });

  it("combines the records of one device as one adapter's, and names every device in provenance.adapters in the order their first records were given", () => {
    const { profile } = deriveGeometryProfile(
      [
        input("m2-a.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]), M2),
        input("m5.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]), M5),
        input("m2-b.json", caseRows(linearCase(2048), BIG, [[A, { speedup: 1.25 }]]), M2),
      ],
      SHARED,
    );
    assertEquals(largestRows(profile), A);
    assertEquals(profile.match, { vendor: "apple", architecture: "metal-3" });
    assertEquals(profile.provenance?.adapters, [M2, M5]);
    assertEquals(profile.provenance?.adapter, undefined);
    assertEquals(profile.provenance?.families.gemm?.sweep, "m2-a.json, m5.json, m2-b.json");
    assertEquals(profile.attentionOnline, undefined);
    assertEquals(profile.linearGemv, undefined);
  });
});

describe("deriveGeometryProfile with shared: a device is told apart by its description", () => {
  const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]);

  it("refuses the records of one description only (two records of one device)", () => {
    assertThrows(
      () =>
        deriveGeometryProfile(
          [input("m5.json", rows, M5), input("m5-again.json", rows, M5)],
          SHARED,
        ),
      Error,
      "A shared derivation needs the records of at least two devices (told apart by their descriptions); every record given is of Apple M5 — derive a measured table of one adapter instead",
    );
  });

  it("refuses a record whose adapter reports no description: one machine with and without Chrome's flag is not two devices", () => {
    // Chrome reports the same M5 without its developer-features flag with an empty device and
    // description: by the four fields it would be a second adapter
    assertThrows(
      () =>
        deriveGeometryProfile(
          [
            input("m5.json", rows, M5),
            input("m5-no-flag.json", rows, { ...M5, device: "", description: "" }),
          ],
          SHARED,
        ),
      Error,
      "m5-no-flag.json: the adapter reports no description, so its records cannot be told apart from another device's (a shared derivation tells devices apart by their descriptions) — sweep where the adapter reports its description (in Chrome, with the WebGPU developer-features flag: chrome://flags/#enable-webgpu-developer-features)",
    );
  });

  it("refuses records of one description with another device id: they are not of one adapter", () => {
    assertThrows(
      () =>
        deriveGeometryProfile(
          [
            input("m5.json", rows, M5),
            input("m2.json", rows, M2),
            input("m5-other.json", rows, { ...M5, device: "0x1111" }),
          ],
          SHARED,
        ),
      Error,
      'm5-other.json: adapter device "0x1111" differs from m5.json\'s "0x0000" under the same description "Apple M5" (a shared derivation tells devices apart by their descriptions, so the records of one description must be of one adapter)',
    );
  });
});

describe("deriveGeometryProfile with shared: the refusals", () => {
  const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]);
  const both = [input("m5.json", rows, M5), input("m2.json", rows, M2)];

  it("refuses a description, a missing architecture, an injection-only target and records of another architecture", () => {
    assertThrows(
      () => deriveGeometryProfile(both, { ...SHARED, description: "Apple M5" }),
      Error,
      "A shared table is matched without a description",
    );
    assertThrows(
      () => deriveGeometryProfile(both, { id: "apple-shared", vendor: "apple", shared: true }),
      Error,
      "A shared derivation needs both vendor and architecture",
    );
    assertThrows(
      () => deriveGeometryProfile(both, { id: "apple-shared", optIn: true, shared: true }),
      Error,
      "it cannot be an injection-only table",
    );
    assertThrows(
      () =>
        deriveGeometryProfile(
          [input("m5.json", rows, M5), input("m1.json", rows, { ...M2, architecture: "metal-2" })],
          SHARED,
        ),
      Error,
      "m1.json: adapter apple / metal-2 / Apple M2 does not match the requested apple / metal-3",
    );
  });

  it("refuses attention-bench records: the online attention's key axis is not derived for a shared table", () => {
    assertThrows(
      () =>
        deriveGeometryProfile(both, {
          ...SHARED,
          attentionBench: [
            { report: benchRecord({ adapter: M5 }), path: "bench.json", sha256: "b" },
          ],
        }),
      Error,
      "A shared table carries tile geometries only",
    );
  });

  it("leaves the measured derivation as it was: records of two adapters without shared are refused by the four fields", () => {
    assertThrows(
      () => deriveGeometryProfile(both, { id: "apple", vendor: "apple", architecture: "metal-3" }),
      Error,
      `m2.json: adapter description "Apple M2" differs from m5.json's "Apple M5" (only sweeps that agree on all four fields are combined into one table)`,
    );
  });
});

describe("deriveGeometryProfile with shared and a base", () => {
  const gemm = [
    input("m5-gemm.json", familyDefaultRows("gemm"), M5),
    input("m2-gemm.json", familyDefaultRows("gemm"), M2),
  ];
  const conv2d = [
    input("m5-conv2d.json", familyDefaultRows("conv2d"), M5),
    input("m2-conv2d.json", familyDefaultRows("conv2d"), M2),
  ];
  const base = (): GeometryProfile => deriveGeometryProfile(gemm, SHARED).profile;

  it("carries the families the records do not cover from a shared table of the same adapters, given in another order", () => {
    const from = base();
    const { profile, verdicts } = deriveGeometryProfile([...conv2d].reverse(), {
      ...SHARED,
      base: from,
    });
    assertEquals(profile.provenance?.families.gemm, from.provenance?.families.gemm);
    assertEquals(profile.provenance?.families.conv2d?.sweep, "m2-conv2d.json, m5-conv2d.json");
    assertEquals(profile.provenance?.adapters, [M2, M5]);
    assertEquals(
      verdicts.some((line) =>
        line.startsWith("- gemmRows ≤ 16 (linear / matmul / bmm row count ≤ 16): carried over"),
      ),
      true,
    );
  });

  it("refuses a shared base of other adapters, a measured base, a shared base without shared, and a covered family one device sweeps in part", () => {
    const from = base();
    const m4 = { ...M5, description: "Apple M4" };
    assertThrows(
      () =>
        deriveGeometryProfile(
          [conv2d[0], input("m4-conv2d.json", familyDefaultRows("conv2d"), m4)],
          {
            ...SHARED,
            base: from,
          },
        ),
      Error,
      "are not the sweep records' (apple / metal-3 / 0x0000 / Apple M5; apple / metal-3 / 0x0000 / Apple M4) — a shared table is carried over only onto records of the same adapters",
    );
    const measured = deriveGeometryProfile([gemm[0]], {
      id: "apple-m5",
      vendor: "apple",
      architecture: "metal-3",
      description: "Apple M5",
    }).profile;
    assertThrows(
      () => deriveGeometryProfile(conv2d, { ...SHARED, base: measured }),
      Error,
      "Base table 'apple-m5' is a measured table (one adapter): a shared derivation does not take it as its base",
    );
    assertThrows(
      () =>
        deriveGeometryProfile([conv2d[0]], {
          id: "apple-shared",
          vendor: "apple",
          architecture: "metal-3",
          description: "Apple M5",
          base: from,
        }),
      Error,
      "Base table 'apple-shared' is a shared table (derived from 2 adapters): it is regenerated by a shared derivation",
    );
    assertThrows(
      () =>
        deriveGeometryProfile(
          [conv2d[0], input("m2-part.json", familyDefaultRows("conv2d").slice(0, 1), M2)],
          { ...SHARED, base: from },
        ),
      Error,
      "The records of Apple M2 cover family conv2d only in part",
    );
  });

  it("derives a family one device has no row of where the base holds the default for it, and refuses it where the base holds another geometry", () => {
    // Every case of conv2d with the default and the first other quick candidate at ×1.2 (identical)
    const fasterConv2d = familyEntries(SWEEP_CASES, "conv2d").flatMap((sweepCase) => {
      const ref = { caseId: sweepCase.id, op: sweepCase.op, shape: caseShape(sweepCase) };
      const { defaultCandidate } = casePlan(sweepCase, 65535, false);
      const faster = quickConv2dCandidates().find(({ name }) => name !== defaultCandidate.name);
      assert(faster !== undefined);
      return [
        row(ref, defaultCandidate, true),
        row(ref, faster, false, { speedup: 1.2, identical: true }),
      ];
    });
    const m2Gemm = input("m2-gemm-again.json", familyDefaultRows("gemm"), M2);
    // The base's conv2d is the default: the M2's records without conv2d keep it so
    const plain = deriveGeometryProfile([conv2d[0], m2Gemm], { ...SHARED, base: base() });
    assertEquals(plain.profile.conv2d, base().conv2d);
    assertStringIncludes(
      linesOf(plain.verdicts, "conv2d.rows64")[0],
      "no case of this class in the records of Apple M2",
    );
    // A base whose conv2d is adopted on both devices would silently lose it
    const adopted = deriveGeometryProfile(
      [
        input("m5-conv2d-fast.json", fasterConv2d, M5),
        input("m2-conv2d-fast.json", fasterConv2d, M2),
      ],
      SHARED,
    ).profile;
    assert(JSON.stringify(adopted.conv2d) !== JSON.stringify(base().conv2d));
    assertThrows(
      () => deriveGeometryProfile([conv2d[0], m2Gemm], { ...SHARED, base: adopted }),
      Error,
      "The records of Apple M2 have no row of family conv2d, and the base table 'apple-shared' holds other than the default there",
    );
  });
});

describe("a shared table's provenance where it is read", () => {
  const shared = (): GeometryProfile =>
    deriveGeometryProfile(
      [
        input("m5.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]), M5),
        input("m2.json", caseRows(linearCase(1024), BIG, [[A, { speedup: 1.3 }]]), M2),
      ],
      SHARED,
    ).profile;

  it("geometryProfileMismatch: matches each of its adapters and one of their vendor and architecture it was not measured on, and no other architecture", () => {
    const profile = shared();
    for (const adapter of [M5, M2, { ...M5, device: "", description: "" }]) {
      assertEquals(geometryProfileMismatch(profile, adapter), undefined, adapter.description);
    }
    assertStringIncludes(
      geometryProfileMismatch(profile, { ...M5, architecture: "metal-2" }) ?? "",
      'Table \'apple-shared\': adapter architecture differs (the shared table\'s "metal-3" / this adapter "metal-2"',
    );
  });

  it("geometryProfileMismatch: a list of one adapter is a mismatch, not a weaker check of a measured table", () => {
    const profile = shared();
    const { provenance } = profile;
    assert(provenance?.adapters !== undefined);
    const one: GeometryProfile = {
      ...profile,
      provenance: { ...provenance, adapters: [M5] },
    };
    assertStringIncludes(
      geometryProfileMismatch(one, M5) ?? "",
      "a shared table names two or more adapters",
    );
  });

  it("geometryProfileMismatch: a list that repeats one device or holds an adapter without a description is a mismatch", () => {
    const profile = shared();
    const { provenance } = profile;
    assert(provenance?.adapters !== undefined);
    const withAdapters = (adapters: readonly GeometryProfileAdapter[]): GeometryProfile => ({
      ...profile,
      provenance: { ...provenance, adapters },
    });
    assertEquals(
      geometryProfileMismatch(withAdapters([M5, { ...M5, device: "0x1111" }]), M5),
      `Table 'apple-shared': its provenance.adapters [0] and [1] are one device (both "Apple M5" — a shared table tells its devices apart by their descriptions and names each once)`,
    );
    assertEquals(
      geometryProfileMismatch(withAdapters([M5, { ...M2, device: "", description: "" }]), M5),
      "Table 'apple-shared': its provenance.adapters [1] has no description (a shared table tells its devices apart by their descriptions)",
    );
  });

  it("the JSON reader gives a shared table back, and refuses one adapter in the list and both forms at once", () => {
    const profile = shared();
    assertEquals(parseGeometryProfileJson(geometryProfileJson(profile)), profile);
    const { provenance } = profile;
    assert(provenance?.adapters !== undefined);
    assertThrows(
      () =>
        parseGeometryProfileJson(
          infinityJson({ ...profile, provenance: { ...provenance, adapters: [M5] } }),
        ),
      Error,
      "provenance.adapters names 1 adapter (a shared table is derived from two or more)",
    );
    assertThrows(
      () =>
        parseGeometryProfileJson(
          infinityJson({ ...profile, provenance: { ...provenance, adapter: M5 } }),
        ),
      Error,
      "provenance has both adapter and adapters",
    );
  });

  it("the JSON reader refuses an empty list, a list that repeats one device and an adapter without a description", () => {
    const profile = shared();
    const { provenance } = profile;
    assert(provenance?.adapters !== undefined);
    const read = (adapters: readonly GeometryProfileAdapter[]) => () =>
      parseGeometryProfileJson(
        infinityJson({ ...profile, provenance: { ...provenance, adapters } }),
      );
    assertThrows(
      read([]),
      Error,
      "provenance.adapters names 0 adapters (a shared table is derived from two or more)",
    );
    assertThrows(
      read([M5, M2, M5]),
      Error,
      'provenance.adapters [0] and [2] are one device (both "Apple M5"',
    );
    assertThrows(
      read([{ ...M5, description: "" }, M2]),
      Error,
      "provenance.adapters [0] has no description",
    );
  });

  it("the injection check refuses a list of fewer than two adapters, one that repeats one device and an adapter without a description", () => {
    const profile = shared();
    const { provenance } = profile;
    assert(provenance?.adapters !== undefined);
    const inject = (adapters: readonly GeometryProfileAdapter[]) => () =>
      assertGeometryProfile({ ...profile, provenance: { ...provenance, adapters } });
    for (const [adapters, message] of [
      [[], "provenance.adapters names 0 adapters (a shared table is derived from two or more)"],
      [[M5], "provenance.adapters names 1 adapter (a shared table is derived from two or more)"],
      [
        [M5, { ...M5, device: "0x1111" }],
        'provenance.adapters [0] and [1] are one device (both "Apple M5" — a shared table tells its devices apart by their descriptions and names each once)',
      ],
      [
        [M5, { ...M2, device: "", description: "" }],
        "provenance.adapters [1] has no description (a shared table tells its devices apart by their descriptions)",
      ],
    ] as const) {
      assertThrows(inject(adapters), CodegenError, message);
    }
  });

  it("the injection check refuses both forms at once and a list that is not one", () => {
    const profile = shared();
    const { provenance } = profile;
    assert(provenance?.adapters !== undefined);
    assertThrows(
      () => assertGeometryProfile({ ...profile, provenance: { ...provenance, adapter: M5 } }),
      CodegenError,
      "provenance has both adapter and adapters",
    );
    assertThrows(
      () => assertGeometryProfile({ ...profile, provenance: { ...provenance, adapters: M5 } }),
      CodegenError,
      "provenance.adapters is not an array",
    );
    assertGeometryProfile(profile);
  });
});

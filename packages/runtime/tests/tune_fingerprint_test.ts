// The matching keys of a saved table (src/tune/fingerprint.ts — ADR 0117 decision 4): the kernel
// fingerprint (geometryProfileKernelsId) and the case-set version (sweepCaseSetId). The public
// function (sweepCaseSetId) comes from the public surface (../tune.ts); the fingerprint, which is not
// public (geometryProfileMismatch derives it internally — ADR 0117 decision 2), and the injectable
// bodies and tool functions come from src. No GPU.
//
// Pinned: (1) the hash is FNV-1a 64 bit itself; (2) the fingerprint moves with each of the table's
// 15 fields, the default geometry, both dp4a variants and a dispatch's params and workgroups, and not
// with id / match / provenance or the WGSL's comments and whitespace (ADR 0117 addendum 2026-10-19);
// (3) a dispatch that cannot be built throws naming the case, the field and the geometry; (4) the
// embedded tables' provenance equals the current runtime's values (a change to a GEMM kernel or to
// the cases re-bakes the tables); (5) the case-set version moves with added or removed cases, the
// dispatch-deciding fields and the bounds, and not with informational fields or key order; (6) a
// record's `candidateKernels` (the fingerprint of the measured candidates) moves on a runtime that
// changed only a candidate's WGSL (`defaultKernels` does not), skips failed rows and builds with the
// record's dp4a variant.

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertMatch,
  assertNotEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { sweepCaseSetId } from "../tune.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { APPLE_METAL_3 } from "../src/kernels/geometry-profiles/apple-metal-3.ts";
import { PROFILE_GEMM_ROWS_BOUNDS, SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import {
  candidateKernelsId,
  canonicalJson,
  type CasePlanner,
  caseSetId,
  fnv1a64Hex,
  geometryProfileKernelsId,
  KernelsIdError,
  type MeasuredRow,
  profileKernelsId,
  sweepCandidateKernelsId,
  wgslTokenText,
} from "../src/tune/fingerprint.ts";
import {
  type GeometryCandidate,
  i8a8Candidate,
  quickConv2dCandidates,
  quickGemmCandidates,
  quickI8a8Candidates,
} from "../src/tune/geometries.ts";
import type { I8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { linearI8a8Key, linearI8a8UsesVec4, linearI8a8Wgsl } from "../src/kernels/linear-i8a8.ts";
import { candidatesFor, type CasePlan, casePlan } from "../src/tune/harness.ts";

type Launch = ReturnType<CasePlan["launch"]>;

/**
 * A naive FNV-1a 64-bit implementation (BigInt — an oracle independent of the fingerprint's
 * implementation).
 */
const referenceFnv1a64 = (text: string): string => {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(text)) {
    hash = ((hash ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return hash.toString(16).padStart(16, "0");
};

describe("fnv1a64Hex", () => {
  it("matches the published FNV-1a 64-bit values", () => {
    assertEquals(fnv1a64Hex(""), "cbf29ce484222325");
    assertEquals(fnv1a64Hex("a"), "af63dc4c8601ec8c");
    assertEquals(fnv1a64Hex("foobar"), "85944171f73967e8");
  });

  it("matches the naive BigInt implementation for multi-byte characters and long strings too (a hash of the UTF-8 bytes)", () => {
    for (const text of [
      "幾何プロファイル",
      "\u0000￿\u{1f600}",
      "x".repeat(10_000),
      JSON.stringify(SWEEP_CASES),
    ]) {
      assertEquals(fnv1a64Hex(text), referenceFnv1a64(text), text.slice(0, 16));
    }
  });
});

/**
 * Replaces the geometry of one field with the first quick candidate of the same family that differs
 * from the current value.
 */
const another = <T extends GeometryCandidate["geometry"]>(
  current: T,
  pool: readonly { readonly geometry: T }[],
): T => {
  const found = pool.find(({ geometry }) => JSON.stringify(geometry) !== JSON.stringify(current));
  if (found === undefined) throw new Error("no candidate to replace it with");
  return found.geometry;
};

/** One table per field of the 15, with only that field changed (field name → table). */
const singleFieldVariants = (base: GeometryProfile): ReadonlyMap<string, GeometryProfile> => {
  const gemm = quickGemmCandidates();
  const conv2d = quickConv2dCandidates();
  const i8a8 = quickI8a8Candidates();
  const variants = new Map<string, GeometryProfile>();
  base.gemmRows.forEach((rule, index) => {
    variants.set(`gemmRows[${index}]`, {
      ...base,
      gemmRows: base.gemmRows.map((entry, at) =>
        at === index ? { ...entry, geometry: another(rule.geometry, gemm) } : entry,
      ),
    });
  });
  const { attention, conv2d: conv, i8a8: integer } = base;
  variants.set("attention.qk", {
    ...base,
    attention: { ...attention, qk: another(attention.qk, gemm) },
  });
  variants.set("attention.pv", {
    ...base,
    attention: { ...attention, pv: another(attention.pv, gemm) },
  });
  variants.set("conv2d.rows64", {
    ...base,
    conv2d: { ...conv, rows64: another(conv.rows64, conv2d) },
  });
  variants.set("conv2d.rows32", {
    ...base,
    conv2d: { ...conv, rows32: another(conv.rows32, conv2d) },
  });
  variants.set("i8a8.linear", {
    ...base,
    i8a8: { ...integer, linear: another(integer.linear, i8a8) },
  });
  variants.set("i8a8.attentionQk", {
    ...base,
    i8a8: { ...integer, attentionQk: another(integer.attentionQk, i8a8) },
  });
  variants.set("i8a8.attentionPv", {
    ...base,
    i8a8: { ...integer, attentionPv: another(integer.attentionPv, i8a8) },
  });
  variants.set("i8a8.conv3d", {
    ...base,
    i8a8: { ...integer, conv3d: another(integer.conv3d, i8a8) },
  });
  return variants;
};

describe("geometryProfileKernelsId: the kernel fingerprint", () => {
  it("gives the same value (16 hex digits) for the same table, independent of copies and field order", () => {
    const value = geometryProfileKernelsId(APPLE_METAL_3);
    assertMatch(value, /^[0-9a-f]{16}$/);
    assertEquals(geometryProfileKernelsId(APPLE_METAL_3), value);
    assertEquals(geometryProfileKernelsId(structuredClone(APPLE_METAL_3)), value);
    const { i8a8, conv2d, attention, gemmRows, id } = APPLE_METAL_3;
    assertEquals(geometryProfileKernelsId({ i8a8, conv2d, attention, gemmRows, id }), value);
  });

  it("id, match and provenance do not affect the fingerprint (only the table's geometry does)", () => {
    const value = geometryProfileKernelsId(APPLE_METAL_3);
    const { match: _match, provenance: _provenance, ...bare } = APPLE_METAL_3;
    assertEquals(geometryProfileKernelsId({ ...bare, id: "other" }), value);
    assertEquals(
      geometryProfileKernelsId({
        ...APPLE_METAL_3,
        match: { vendor: "intel" },
        provenance: {
          sweep: "x.json",
          sha256: "0",
          date: "2000-01-01",
          candidateSet: "quick",
          adapter: { vendor: "x", architecture: "y", device: "z", description: "w" },
          kernels: "0000000000000000",
          caseSet: "0000000000000000",
        },
      }),
      value,
    );
  });

  it("changing any one of the table's 15 fields changes the value, and the 15 values differ from each other", () => {
    const base = geometryProfileKernelsId(APPLE_METAL_3);
    const variants = singleFieldVariants(APPLE_METAL_3);
    assertEquals(variants.size, 15);
    const values = new Set([base]);
    for (const [field, profile] of variants) {
      const value = geometryProfileKernelsId(profile);
      assertNotEquals(value, base, field);
      values.add(value);
    }
    assertEquals(values.size, 16);
  });

  it("a table with different gemmRows segment boundaries (the default's 3 segments) is also derivable, assigning geometries to cases with the runtime's selection function", () => {
    const value = geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE);
    assertMatch(value, /^[0-9a-f]{16}$/);
    // The default table is not the hash of an empty input (both the table's and the default's
    // kernels are counted)
    assertNotEquals(value, fnv1a64Hex(""));
    // The 3-segment default table and a table refining the same geometries into 7 segments assign
    // the same geometry to every case, so they have the same fingerprint
    const refined: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      gemmRows: PROFILE_GEMM_ROWS_BOUNDS.map((maxRows) => ({
        maxRows,
        geometry: (
          DEFAULT_GEOMETRY_PROFILE.gemmRows.find((rule) => maxRows <= rule.maxRows) ??
          DEFAULT_GEOMETRY_PROFILE.gemmRows[DEFAULT_GEOMETRY_PROFILE.gemmRows.length - 1]
        ).geometry,
      })),
    };
    assertEquals(geometryProfileKernelsId(refined), value);
  });

  it("changes when the default geometry (the case plan's defaultCandidate) changes — the kernels the ratios are based on are in the fingerprint too", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    assertEquals(base, geometryProfileKernelsId(APPLE_METAL_3));
    // A case plan with the default geometry of one case replaced by another candidate (mimics a
    // runtime whose default geometry changed)
    const target = SWEEP_CASES.find((sweepCase) => sweepCase.op === "linear");
    assert(target !== undefined);
    const shifted: CasePlanner = (sweepCase, limit, dp4a) => {
      const plan = casePlan(sweepCase, limit, dp4a);
      if (sweepCase !== target) return plan;
      const candidate = quickGemmCandidates().find(
        ({ name }) => name !== plan.defaultCandidate.name,
      );
      assert(candidate !== undefined);
      return { ...plan, defaultCandidate: candidate };
    };
    assertNotEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, shifted), base);
  });

  it("i8a8 cases count both dp4a variants (the value changes when either WGSL changes)", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    for (const variant of [false, true]) {
      const marked: CasePlanner = (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        if (dp4a !== variant) return plan;
        return {
          ...plan,
          launch: (candidate) => {
            const launch = plan.launch(candidate);
            return { ...launch, wgsl: `${launch.wgsl}\nfn changed() {}` };
          },
        };
      };
      const isI8a8 = (sweepCase: SweepCase): boolean =>
        sweepCase.op === "i8a8-linear" ||
        sweepCase.op === "i8a8-attention" ||
        sweepCase.op === "i8a8-conv3d";
      assertNotEquals(
        profileKernelsId(APPLE_METAL_3, SWEEP_CASES.filter(isI8a8), marked),
        profileKernelsId(APPLE_METAL_3, SWEEP_CASES.filter(isI8a8), casePlan),
        `dp4a ${variant}`,
      );
      assertNotEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, marked), base);
    }
  });

  it("the conv3d w8a8 cases count both dp4a variants and read the table's i8a8.conv3d (not another i8a8 field)", () => {
    const conv3dCases = SWEEP_CASES.filter((sweepCase) => sweepCase.op === "i8a8-conv3d");
    assert(conv3dCases.length > 0);
    const plain = profileKernelsId(APPLE_METAL_3, conv3dCases, casePlan);
    for (const variant of [false, true]) {
      const marked: CasePlanner = (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        if (dp4a !== variant) return plan;
        return {
          ...plan,
          launch: (candidate) => {
            const launch = plan.launch(candidate);
            return { ...launch, wgsl: `${launch.wgsl}\nfn changed() {}` };
          },
        };
      };
      assertNotEquals(
        profileKernelsId(APPLE_METAL_3, conv3dCases, marked),
        plain,
        `dp4a ${variant}`,
      );
    }
    const { i8a8 } = APPLE_METAL_3;
    const conv3d = quickI8a8Candidates().find(
      ({ geometry }) =>
        ![i8a8.linear, i8a8.attentionQk, i8a8.attentionPv, i8a8.conv3d].some(
          (field) => JSON.stringify(field) === JSON.stringify(geometry),
        ),
    );
    assert(conv3d !== undefined);
    // Only conv3d moves: the conv3d cases see it, the other cases do not
    const moved = { ...APPLE_METAL_3, i8a8: { ...i8a8, conv3d: conv3d.geometry } };
    assertNotEquals(profileKernelsId(moved, conv3dCases, casePlan), plain);
    const others = SWEEP_CASES.filter((sweepCase) => sweepCase.op !== "i8a8-conv3d");
    assertEquals(
      profileKernelsId(moved, others, casePlan),
      profileKernelsId(APPLE_METAL_3, others, casePlan),
    );
  });

  it("changes when only the dispatch params or only the workgroups change (a different kernel even with the same key and WGSL)", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    // A case plan with one field of every case's launch replaced (mimics a runtime that differs
    // only in that field)
    const altered =
      (change: (launch: Launch) => Launch): CasePlanner =>
      (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        return { ...plan, launch: (candidate) => change(plan.launch(candidate)) };
      };
    const params = altered((launch) => ({
      ...launch,
      params: Uint32Array.from(launch.params, (value, index) => (index === 0 ? value + 1 : value)),
    }));
    const workgroups = altered((launch) => ({
      ...launch,
      workgroups: [launch.workgroups[0] + 1, launch.workgroups[1], launch.workgroups[2]],
    }));
    const paramsValue = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, params);
    const workgroupsValue = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, workgroups);
    assertNotEquals(paramsValue, base, "params");
    assertNotEquals(workgroupsValue, base, "workgroups");
    assertNotEquals(paramsValue, workgroupsValue);
  });

  it("does not change for a runtime whose WGSL differs only in comments and whitespace (the same tokens are the same kernel — ADR 0117 addendum 2026-10-19)", () => {
    const base = profileKernelsId(APPLE_METAL_3, SWEEP_CASES, casePlan);
    const rewritten =
      (change: (wgsl: string) => string): CasePlanner =>
      (sweepCase, limit, dp4a) => {
        const plan = casePlan(sweepCase, limit, dp4a);
        return {
          ...plan,
          launch: (candidate) => {
            const launch = plan.launch(candidate);
            return { ...launch, wgsl: change(launch.wgsl) };
          },
        };
      };
    const commented = rewritten(
      (wgsl) => `// 翻訳したコメント /* 入れ子 /* の */ 注記 */\n${wgsl}\n// changed`,
    );
    const stripped = rewritten((wgsl) => wgsl.replace(/\/\/[^\n]*/g, ""));
    const reindented = rewritten((wgsl) => wgsl.replace(/\n[ \t]+/g, "\n\t\t"));
    assertEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, commented), base, "comments added");
    assertEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, stripped), base, "comments removed");
    assertEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, reindented), base, "whitespace");
    // A change of even one token is a different kernel
    const token = rewritten((wgsl) => `${wgsl}\nfn changed() {}`);
    assertNotEquals(profileKernelsId(APPLE_METAL_3, SWEEP_CASES, token), base);
  });

  it("throws on a dispatch that cannot be built (over 65535), naming the case, the table field, the geometry and why 65535 is fixed", () => {
    // It passes the check (assertGeometryProfile), but with tileN 4 the N = 262144 of conv2d 512²
    // becomes 65536 workgroups
    const narrow: GeometryProfile = {
      ...APPLE_METAL_3,
      id: "narrow",
      conv2d: { ...APPLE_METAL_3.conv2d, rows32: { regM: 1, regN: 4, wgX: 1, wgY: 16 } },
    };
    const error = assertThrows(() => geometryProfileKernelsId(narrow), KernelsIdError);
    assertInstanceOf(error.cause, DispatchLimitError);
    for (const part of [
      "conv2d-c96-512x512",
      "table 'narrow', conv2d.rows32",
      '{"regM":1,"regN":4,"wgX":1,"wgY":16}',
      "65535",
      "WebGPU guarantees on every device",
    ]) {
      assertStringIncludes(error.message, part);
    }
  });

  it("reads neither the GPU nor the clock (the same value in an environment where navigator.gpu, performance.now and Date.now throw)", () => {
    const expected = geometryProfileKernelsId(APPLE_METAL_3);
    const forbidden = (what: string) => (): never => {
      throw new Error(`read ${what}`);
    };
    // oxlint-disable-next-line typescript/unbound-method -- kept only to reassign in finally; never called
    const originalNow = performance.now;
    const originalDateNow = Date.now;
    Object.defineProperty(navigator, "gpu", {
      get: forbidden("navigator.gpu"),
      configurable: true,
    });
    performance.now = forbidden("performance.now");
    Date.now = forbidden("Date.now");
    try {
      assertEquals(geometryProfileKernelsId(APPLE_METAL_3), expected);
    } finally {
      Reflect.deleteProperty(navigator, "gpu");
      performance.now = originalNow;
      Date.now = originalDateNow;
    }
  });
});

describe("candidateKernelsId: the kernel fingerprint of the measured candidates (the record's candidateKernels — ADR 0117 decision 8)", () => {
  const caseOf = (op: SweepCase["op"]): SweepCase => {
    const found = SWEEP_CASES.find((sweepCase) => sweepCase.op === op);
    assert(found !== undefined, op);
    return found;
  };
  /**
   * Rows measuring a case with every candidate geometry (the fields of the writer's row shape that
   * the fingerprint reads).
   */
  const rowsOf = (sweepCase: SweepCase, candidates: readonly GeometryCandidate[]): MeasuredRow[] =>
    candidates.map((candidate) => ({ caseId: sweepCase.id, geometryParams: candidate.geometry }));
  const LINEAR = caseOf("linear");
  const I8A8 = caseOf("i8a8-linear");
  const CONV = caseOf("conv2d");
  const F32_ROWS = [
    ...rowsOf(LINEAR, quickGemmCandidates()),
    ...rowsOf(CONV, quickConv2dCandidates()),
  ];
  const ROWS = [...F32_ROWS, ...rowsOf(I8A8, quickI8a8Candidates())];

  it("is deterministic (16 hex digits), and the writer's and reader's entry points give the value derived with the current runtime's case plan", () => {
    const value = sweepCandidateKernelsId(ROWS, false);
    assertMatch(value, /^[0-9a-f]{16}$/);
    assertEquals(sweepCandidateKernelsId(structuredClone(ROWS), false), value);
    assertEquals(candidateKernelsId(ROWS, false, SWEEP_CASES, casePlan), value);
  });

  it("changes for a runtime that changed only the candidates' WGSL and not the default (the default's fingerprint defaultKernels does not change)", () => {
    // A case plan that adds one line only to the WGSL of dispatches whose geometry differs from the
    // default (mimics a runtime that fixed only the candidates' kernels)
    const candidatesOnly: CasePlanner = (sweepCase, limit, dp4a) => {
      const plan = casePlan(sweepCase, limit, dp4a);
      const defaultGeometry = JSON.stringify(plan.defaultCandidate.geometry);
      return {
        ...plan,
        launch: (candidate) => {
          const launch = plan.launch(candidate);
          return JSON.stringify(candidate.geometry) === defaultGeometry
            ? launch
            : { ...launch, wgsl: `${launch.wgsl}\nfn changed() {}` };
        },
      };
    };
    // The fingerprint of the ratios' basis does not see this change (if records were matched on
    // defaultKernels alone, stale measurements of the candidates would pass)
    assertEquals(
      profileKernelsId(DEFAULT_GEOMETRY_PROFILE, SWEEP_CASES, candidatesOnly),
      geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
    );
    assertNotEquals(
      candidateKernelsId(ROWS, false, SWEEP_CASES, candidatesOnly),
      sweepCandidateKernelsId(ROWS, false),
    );
  });

  it("does not count failed rows (they are not adoption inputs), and changes when one measured row differs", () => {
    const value = sweepCandidateKernelsId(ROWS, false);
    const failed: MeasuredRow = {
      caseId: LINEAR.id,
      geometryParams: quickGemmCandidates()[1].geometry,
      error: "device lost",
    };
    assertEquals(sweepCandidateKernelsId([...ROWS, failed], false), value);
    assertNotEquals(sweepCandidateKernelsId(ROWS.slice(1), false), value);
  });

  it("builds i8a8 rows with the record's dp4a variant, while f32 skeleton rows do not depend on dp4a", () => {
    const i8a8Rows = rowsOf(I8A8, quickI8a8Candidates());
    assertNotEquals(
      sweepCandidateKernelsId(i8a8Rows, true),
      sweepCandidateKernelsId(i8a8Rows, false),
    );
    assertEquals(sweepCandidateKernelsId(F32_ROWS, true), sweepCandidateKernelsId(F32_ROWS, false));
  });

  it("throws on a row of a case missing from the current runtime's shape table, naming the case", () => {
    assertThrows(
      () =>
        sweepCandidateKernelsId(
          [{ caseId: "linear-m3", geometryParams: quickGemmCandidates()[0].geometry }],
          false,
        ),
      KernelsIdError,
      "Case linear-m3 of a measured row is not in the current runtime's shape table",
    );
  });

  it("is also derivable from rows measuring every case with every full candidate (buildable within 65535 — the writer does not throw after the sweep)", () => {
    const rows = SWEEP_CASES.flatMap((sweepCase) =>
      rowsOf(sweepCase, [
        casePlan(sweepCase, 65535, false).defaultCandidate,
        ...candidatesFor(sweepCase, "full"),
      ]),
    );
    for (const dp4a of [false, true]) {
      assertMatch(sweepCandidateKernelsId(rows, dp4a), /^[0-9a-f]{16}$/);
    }
  });
});

describe("the provenance of the built-in tables matches the current runtime's values", () => {
  // On failure: a GEMM kernel (key, params, WGSL tokens, dispatch count — comments and whitespace do
  // not count), the default geometry, the sweep cases or the gemmRows bounds changed. Re-derive every
  // embedded table with the current runtime (ADR 0117 decision 4: the regeneration makes the machine
  // say that the kernels changed, and the per-profile GPU test gpu_geometry_profile_test.ts re-checks
  // output identity on the new kernels). A record without caseSet / defaultKernels (the 2026-09/10
  // Chrome records of nvidia-blackwell and apple-metal-3) is still accepted, so those tables re-derive
  // with unchanged values. A record that carries them (nvidia-ampere's) is rejected once they move:
  // that table needs a new sweep, which can re-pick other fields (the 2026-10-26 conv3d re-sweep also
  // moved i8a8.attentionQk — docs/research/2026-10-26-conv3d-geometry-sweep.md §4).
  for (const profile of BUILTIN_GEOMETRY_PROFILES) {
    it(`${profile.id}: provenance.kernels = the current runtime's fingerprint, provenance.caseSet = the current case-set version`, () => {
      const provenance = profile.provenance;
      assert(
        provenance !== undefined,
        `${profile.id}: built-in tables are generated from a sweep (provenance is required)`,
      );
      assertEquals(
        provenance.kernels,
        geometryProfileKernelsId(profile),
        `${profile.id}: the kernel fingerprint differs from the current runtime's — regenerate with the command in the header comment`,
      );
      assertEquals(
        provenance.caseSet,
        sweepCaseSetId(),
        `${profile.id}: the case-set version differs from the current runtime's — regenerate with the command in the header comment`,
      );
    });
  }
});

describe("sweepCaseSetId: the case-set version", () => {
  it("is deterministic (16 hex digits), a value derived from the current cases and boundaries", () => {
    const value = sweepCaseSetId();
    assertMatch(value, /^[0-9a-f]{16}$/);
    assertEquals(sweepCaseSetId(), value);
    assertEquals(caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS), value);
  });

  it("changes when a case is added, a dispatch-deciding field (shape, op, attr) changes or a boundary changes", () => {
    const base = caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS);
    const [first] = SWEEP_CASES;
    assertNotEquals(
      caseSetId([...SWEEP_CASES, { ...first, id: `${first.id}-copy` }], PROFILE_GEMM_ROWS_BOUNDS),
      base,
    );
    /** The version with only the case at index `at` replaced. */
    const replaced = (at: number, sweepCase: SweepCase): string =>
      caseSetId(SWEEP_CASES.with(at, sweepCase), PROFILE_GEMM_ROWS_BOUNDS);
    const indexOf = (op: SweepCase["op"]): number => {
      const index = SWEEP_CASES.findIndex((sweepCase) => sweepCase.op === op);
      assert(index >= 0, op);
      return index;
    };
    const linear = indexOf("linear");
    const attention = indexOf("attention");
    const conv2d = indexOf("conv2d");
    const linearCase = SWEEP_CASES[linear];
    const attentionCase = SWEEP_CASES[attention];
    const conv2dCase = SWEEP_CASES[conv2d];
    assert(linearCase.op === "linear");
    assert(attentionCase.op === "attention");
    assert(conv2dCase.op === "conv2d");
    assertNotEquals(replaced(linear, { ...linearCase, m: linearCase.m + 1 }), base, "m");
    assertNotEquals(replaced(linear, { ...linearCase, op: "i8a8-linear" }), base, "op");
    assertNotEquals(replaced(attention, { ...attentionCase, d: 64 }), base, "d");
    assertNotEquals(replaced(attention, { ...attentionCase, scale: 0.5 }), base, "scale");
    assertNotEquals(replaced(attention, { ...attentionCase, score: "f16" }), base, "score");
    assertNotEquals(replaced(conv2d, { ...conv2dCase, width: 64 }), base, "width");
    const conv3d = indexOf("i8a8-conv3d");
    const conv3dCase = SWEEP_CASES[conv3d];
    assert(conv3dCase.op === "i8a8-conv3d");
    for (const field of ["timeIn", "paddingH", "kernelT", "channelsOut"] as const) {
      const dims = { ...conv3dCase.dims, [field]: conv3dCase.dims[field] + 1 };
      assertNotEquals(replaced(conv3d, { ...conv3dCase, dims }), base, `dims.${field}`);
    }
    assertNotEquals(caseSetId(SWEEP_CASES, [16, 32, 64, 512, Number.POSITIVE_INFINITY]), base);
    // The trailing Infinity counts as a value too (plain JSON.stringify drops it to null and does
    // not tell it apart from other values)
    assertNotEquals(caseSetId(SWEEP_CASES, [...PROFILE_GEMM_ROWS_BOUNDS.slice(0, -1), 1024]), base);
  });

  it("does not change when only informational fields such as descriptions (source, censusCount, mirrorOf) change", () => {
    const base = caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS);
    const described = SWEEP_CASES.map((sweepCase): SweepCase =>
      sweepCase.op === "matmul"
        ? { ...sweepCase, source: `${sweepCase.source} (edited)`, mirrorOf: "other" }
        : {
            ...sweepCase,
            source: `${sweepCase.source} (edited)`,
            censusCount: sweepCase.censusCount + 1,
          },
    );
    assertEquals(caseSetId(described, PROFILE_GEMM_ROWS_BOUNDS), base);
  });

  it("throws on a case with an unclassified field (a field not yet decided for the version is not dropped silently)", () => {
    const [first] = SWEEP_CASES;
    const unknown = { ...first, note: "x" };
    assertThrows(
      () => caseSetId([unknown, ...SWEEP_CASES.slice(1)], PROFILE_GEMM_ROWS_BOUNDS),
      Error,
      "field note has no classification",
    );
  });

  it("does not depend on the order of a case's keys (canonical JSON)", () => {
    const reordered = SWEEP_CASES.map((sweepCase) =>
      Object.fromEntries(Object.entries(sweepCase).reverse()),
    );
    assertEquals(canonicalJson(reordered), canonicalJson(SWEEP_CASES));
  });
});

describe("wgslTokenText: the WGSL the fingerprint counts", () => {
  it("drops line comments and nested block comments, and folds runs of whitespace into one", () => {
    assertEquals(
      wgslTokenText(
        "// head\nfn f() -> u32 { /* a /* nested */ b */ return 1u; } // tail\n\n  let x = 2u;",
      ),
      "fn f() -> u32 { return 1u; } let x = 2u;",
    );
  });

  it("an unclosed block comment runs to the end, and tokens outside comments stay as they are", () => {
    assertEquals(wgslTokenText("let a = 1u; /* open"), "let a = 1u;");
    assertEquals(wgslTokenText("let url = 1u; // https://example"), "let url = 1u;");
    assertEquals(wgslTokenText("a/b // c\nd*e"), "a/b d*e");
  });
});

describe("canonicalJson", () => {
  it("sorts keys, removes whitespace, drops undefined fields and writes Infinity as 1e999", () => {
    assertEquals(
      canonicalJson({ b: [1, Number.POSITIVE_INFINITY, "x"], a: { d: undefined, c: null } }),
      '{"a":{"c":null},"b":[1,1e999,"x"]}',
    );
  });
});

describe("the vec4 skeleton of an i8a8 geometry in the fingerprint and the sweep's dispatch (perf-ledger K-85)", () => {
  const word: I8a8Geometry = { ...DEFAULT_GEOMETRY_PROFILE.i8a8.linear, tileK: 32 };
  const skeleton: I8a8Geometry = { ...word, skeleton: "vec4" };

  it("a table whose i8a8.linear carries the skeleton fingerprints differently from the same geometry without it", () => {
    const withWord = {
      ...DEFAULT_GEOMETRY_PROFILE,
      i8a8: { ...DEFAULT_GEOMETRY_PROFILE.i8a8, linear: word },
    };
    const withSkeleton = { ...withWord, i8a8: { ...withWord.i8a8, linear: skeleton } };
    assertNotEquals(geometryProfileKernelsId(withSkeleton), geometryProfileKernelsId(withWord));
    assertNotEquals(
      geometryProfileKernelsId(withWord),
      geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
    );
  });

  it("the sweep builds an i8a8 linear skeleton candidate with the fill form a Session takes (:fill16 at k % 16 == 0), and the same geometry without it as before", () => {
    const sweepCase = SWEEP_CASES.find((entry) => entry.op === "i8a8-linear");
    assert(sweepCase !== undefined && sweepCase.op === "i8a8-linear" && sweepCase.k % 16 === 0);
    const v4 = linearI8a8UsesVec4(sweepCase.n);
    for (const dp4a of [false, true]) {
      const plan = casePlan(sweepCase, 65535, dp4a);
      const vec = plan.launch(i8a8Candidate(skeleton));
      assertEquals(vec.key, linearI8a8Key(v4, dp4a, skeleton, "i8", undefined, true));
      assertEquals(vec.wgsl, linearI8a8Wgsl(v4, dp4a, skeleton, "i8", undefined, true));
      assertStringIncludes(vec.key, "k32s");
      assertEquals(vec.key.endsWith(":fill16"), true);
      const plain = plan.launch(i8a8Candidate(word));
      assertEquals(plain.key, linearI8a8Key(v4, dp4a, word));
      assertEquals(plain.wgsl, linearI8a8Wgsl(v4, dp4a, word));
    }
  });

  it("a measured row with the skeleton fingerprints differently from the same row without it", () => {
    const sweepCase = SWEEP_CASES.find((entry) => entry.op === "i8a8-linear");
    assert(sweepCase !== undefined);
    const row = (geometryParams: I8a8Geometry): MeasuredRow => ({
      caseId: sweepCase.id,
      geometryParams,
    });
    assertNotEquals(
      sweepCandidateKernelsId([row(skeleton)], true),
      sweepCandidateKernelsId([row(word)], true),
    );
  });
});

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmGeometryTileKeyPart,
  gemmThreads,
  gemmTileM,
  gemmTileN,
} from "../src/kernels/gemm-geometry.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8WorkgroupStorageBytes,
} from "../src/kernels/i8a8-geometry.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { conv2dIgemmKey } from "../src/kernels/conv2d.ts";
import {
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import {
  CANDIDATE_SETS,
  conv2dCandidate,
  conv2dCandidates,
  conv2dCandidatesIn,
  gemmCandidate,
  gemmCandidates,
  gemmCandidatesIn,
  i8a8Candidate,
  i8a8Candidates,
  i8a8CandidatesIn,
  i8a8LinearCandidates,
  i8a8LinearCandidatesIn,
  EVALUATION_SETS,
  k88Conv2dCandidates,
  k88GemmCandidates,
  quickConv2dCandidates,
  quickGemmCandidates,
  quickI8a8Candidates,
  quickPlusConv2dCandidates,
  quickPlusGemmCandidates,
  quickPlusI8a8Candidates,
} from "../src/tune/geometries.ts";
import { SWEEP_CASES } from "../src/tune/cases.ts";
import { candidatesFor, selectedCandidates } from "../src/tune/harness.ts";

const names = (items: readonly { readonly name: string }[]): string[] =>
  items.map((item) => item.name);

const gemmName = (geometry: GemmGeometry): string => gemmGeometryTileKeyPart(geometry);

describe("candidates of the f32 skeleton", () => {
  it("include the 3 production geometries (the default, mid-M and small-M buckets)", () => {
    const listed = names(gemmCandidates());
    for (const rows of [4096, 512, 64]) {
      assert(listed.includes(gemmName(gemmGeometryForRows(rows))), `M=${rows}`);
    }
    assert(listed.includes(gemmName(defaultGemmGeometry())));
  });

  it("every candidate passes the runtime's check and stays within threads ≤ 256 and tile edges ≤ 128", () => {
    for (const { name, geometry } of gemmCandidates()) {
      assertGemmGeometry(geometry, name);
      assert(gemmThreads(geometry) <= 256, name);
      assert(gemmTileM(geometry) <= 128 && gemmTileN(geometry) <= 128, name);
      assertEquals(name, gemmName(geometry));
    }
  });

  it("have no duplicate names (= key fragments)", () => {
    const listed = names(gemmCandidates());
    assertEquals(new Set(listed).size, listed.length);
  });

  it("quick is a subset of full and includes the default", () => {
    const full = new Set(names(gemmCandidates()));
    for (const name of names(quickGemmCandidates())) assert(full.has(name), name);
    assert(names(quickGemmCandidates()).includes(gemmName(defaultGemmGeometry())));
  });
});

describe("conv2d candidates", () => {
  it("n tiles are only 64 / 128, and both production geometries with m tiles of 64 / 32 are included", () => {
    const candidates = conv2dCandidates();
    for (const { name, geometry } of candidates) {
      assert([64, 128].includes(gemmTileN(geometry)), name);
    }
    const listed = names(candidates);
    for (const mTile of [64, 32]) {
      assert(listed.includes(conv2dCandidate(gemmMTileGeometry(mTile)).name), `mTile ${mTile}`);
    }
  });

  it("names are spelled like fragments of the production pipeline key (contained in the key without v4) and have no duplicates", () => {
    const candidates = conv2dCandidates();
    for (const { name, geometry } of candidates) {
      const key = conv2dIgemmKey("f16", false, 64, geometry);
      assert(key.includes(`:${name}:`), `${key} does not contain ${name}`);
    }
    assertEquals(new Set(names(candidates)).size, candidates.length);
  });

  it("quick is a subset of full", () => {
    const full = new Set(names(conv2dCandidates()));
    for (const name of names(quickConv2dCandidates())) assert(full.has(name), name);
  });
});

describe("i8a8 candidates", () => {
  it("include the linear / ①QK default and the ③PV default, and every candidate passes the runtime's check", () => {
    const candidates = i8a8Candidates();
    const listed = names(candidates);
    for (const op of ["linear", "attention_qk", "attention_pv"] as const) {
      assert(listed.includes(i8a8GeometryKeyPart(defaultI8a8Geometry(op), false)), op);
    }
    for (const { name, geometry } of candidates) {
      assertI8a8Geometry(geometry, name);
      assertEquals(name, i8a8GeometryKeyPart(geometry, false));
    }
    assertEquals(new Set(listed).size, listed.length);
  });

  it("quick is a subset of full", () => {
    const full = new Set(names(i8a8Candidates()));
    for (const name of names(quickI8a8Candidates())) assert(full.has(name), name);
  });
});

describe("the i8a8 linear's full grid (ADR 0115 addendum decision 14)", () => {
  const linear = i8a8LinearCandidates();
  const listed = names(linear);
  /** Whether a geometry passes the runtime's check and fits 16,384 B (the grid's filter). */
  const allowed = (geometry: I8a8Geometry): boolean => {
    if (i8a8WorkgroupStorageBytes(geometry) > 16384) return false;
    try {
      assertI8a8Geometry(geometry, "grid");
      return true;
    } catch {
      return false;
    }
  };

  it("is the old full i8a8 grid first, then the skeleton grid regM 8 / 12 / 16 × regN 4 / 8 × wgX 8 / 16 × wgY 4 / 8 / 16 × tileK 16 / 32 / 64, without and with the skeleton, deduplicated, within the checks and 16,384 B", () => {
    assertEquals(listed.slice(0, i8a8Candidates().length), names(i8a8Candidates()));
    assertEquals(new Set(listed).size, listed.length);
    assertEquals(linear.length, 230);
    assertEquals(linear.filter(({ geometry }) => geometry.skeleton === "vec4").length, 103);
    assertEquals(linear.filter(({ geometry }) => geometry.tileK === 64).length, 62);
    for (const { name, geometry } of linear) {
      assert(allowed(geometry), name);
      assertEquals(name, i8a8GeometryKeyPart(geometry, false));
    }
    // Every point of the skeleton grid the checks and the limit allow is in it (nothing else dropped)
    for (const regM of [8, 12, 16]) {
      for (const regN of [4, 8]) {
        for (const wgX of [8, 16]) {
          for (const wgY of [4, 8, 16]) {
            for (const tileK of [16, 32, 64]) {
              for (const skeleton of [undefined, "vec4" as const]) {
                const geometry: I8a8Geometry = {
                  regM,
                  regN,
                  wgX,
                  wgY,
                  tileK,
                  ...(skeleton === undefined ? {} : { skeleton }),
                };
                assertEquals(
                  listed.includes(i8a8GeometryKeyPart(geometry, false)),
                  allowed(geometry),
                  JSON.stringify(geometry),
                );
              }
            }
          }
        }
      }
    }
    // The geometries the K-85 evaluation picked on both GPUs are in it
    for (const name of [
      "tile64x64r8x4w16x8k64s",
      "tile32x64r8x4w16x4k64s",
      "tile128x64r16x4w16x8k64s",
      "tile128x64r8x8w8x16k64s",
      "tile32x32r8x4w8x4k64s",
      "tile96x128r12x8w16x8k64s",
    ]) {
      assert(listed.includes(name), name);
    }
  });

  it("is what the i8a8 linear sweeps under full; its quick+ adds the tables' linear-only forms; quick and the i8a8 attention and conv3d keep the i8a8 family's set", () => {
    assertEquals(names(i8a8LinearCandidatesIn("full")), listed);
    assertEquals(names(i8a8LinearCandidatesIn("quick")), names(i8a8CandidatesIn("quick")));
    // quick+: the linear measures every table geometry; the attention and conv3d leave out the
    // linear-only forms (a table's skeleton / K64 i8a8.linear rule)
    const familyForm = ({ geometry }: { readonly geometry: I8a8Geometry }) =>
      !("skeleton" in geometry) && geometry.tileK <= 32;
    assertEquals(names(i8a8LinearCandidatesIn("quick+")), names(quickPlusI8a8Candidates()));
    assertEquals(
      names(i8a8CandidatesIn("quick+")),
      names(quickPlusI8a8Candidates().filter(familyForm)),
    );
    assertEquals(names(i8a8CandidatesIn("full")), names(i8a8Candidates()));
    for (const op of ["i8a8-linear", "i8a8-attention", "i8a8-conv3d"] as const) {
      const sweepCase = SWEEP_CASES.find((entry) => entry.op === op);
      assert(sweepCase !== undefined, op);
      assertEquals(
        names(candidatesFor(sweepCase, "full")),
        op === "i8a8-linear" ? listed : names(i8a8Candidates()),
        op,
      );
      assertEquals(
        names(candidatesFor(sweepCase, "quick+")),
        op === "i8a8-linear"
          ? names(quickPlusI8a8Candidates())
          : names(quickPlusI8a8Candidates().filter(familyForm)),
        op,
      );
    }
    // The i8a8 family's sets (the attention's and conv3d's) hold no skeleton and no K64 tile
    for (const set of ["quick", "quick+", "full"] as const) {
      for (const { name, geometry } of i8a8CandidatesIn(set)) {
        assert(!("skeleton" in geometry) && geometry.tileK <= 32, `${set}: ${name}`);
      }
    }
  });
});

describe("selectedCandidates: the candidates one case measures under the geometries option (ADR 0117 addendum 2026-11-12)", () => {
  const sweepCase = SWEEP_CASES.find((entry) => entry.op === "i8a8-linear");
  assert(sweepCase !== undefined);

  it("is every candidate of the set when absent, and the named ones in the order given otherwise", () => {
    assertEquals(
      names(selectedCandidates(sweepCase, "full", undefined)),
      names(candidatesFor(sweepCase, "full")),
    );
    const [first, second, third] = candidatesFor(sweepCase, "full").slice(-3);
    const picked = selectedCandidates(sweepCase, "full", [third.name, first.name, second.name]);
    assertEquals(names(picked), [third.name, first.name, second.name]);
    assertEquals(picked[0].geometry, third.geometry);
  });

  it("throws on a name that is not a candidate of the case in the set", () => {
    assertThrows(
      () => selectedCandidates(sweepCase, "quick", ["tile64x64r8x4w16x8k64s"]),
      Error,
      `Geometry tile64x64r8x4w16x8k64s is not a candidate of case ${sweepCase.id} in candidate set quick`,
    );
  });
});

describe("candidate set quick+ (quick ∪ the geometries adopted by the registered profiles)", () => {
  const includesAll = (listed: readonly string[], expected: readonly string[], label: string) => {
    const set = new Set(listed);
    for (const name of expected) assert(set.has(name), `${label}: ${name} is missing`);
  };

  it("includes quick (f32, conv2d, i8a8)", () => {
    includesAll(names(quickPlusGemmCandidates()), names(quickGemmCandidates()), "f32");
    includesAll(names(quickPlusConv2dCandidates()), names(quickConv2dCandidates()), "conv2d");
    includesAll(names(quickPlusI8a8Candidates()), names(quickI8a8Candidates()), "i8a8");
  });

  it("includes every geometry of every registered profile in the set of its family", () => {
    assert(BUILTIN_GEOMETRY_PROFILES.length > 0);
    for (const profile of BUILTIN_GEOMETRY_PROFILES) {
      includesAll(
        names(quickPlusGemmCandidates()),
        [
          ...profile.gemmRows.map((rule) => gemmCandidate(rule.geometry).name),
          gemmCandidate(profile.attention.qk).name,
          gemmCandidate(profile.attention.pv).name,
        ],
        `${profile.id} f32`,
      );
      includesAll(
        names(quickPlusConv2dCandidates()),
        [conv2dCandidate(profile.conv2d.rows64).name, conv2dCandidate(profile.conv2d.rows32).name],
        `${profile.id} conv2d`,
      );
      includesAll(
        names(quickPlusI8a8Candidates()),
        [
          ...profile.i8a8.linear.map((rule) => i8a8Candidate(rule.geometry).name),
          i8a8Candidate(profile.i8a8.attentionQk).name,
          i8a8Candidate(profile.i8a8.attentionPv).name,
          i8a8Candidate(profile.i8a8.conv3d).name,
        ],
        `${profile.id} i8a8`,
      );
    }
  });

  it("i8a8 quick+ adds every i8a8.linear rule and each of the other three i8a8 fields of a profile it is given (distinct, all outside quick)", () => {
    // Five distinct geometries from the full grid that quick does not hold, one per rule or field —
    // so a rule or field missing from quick+ cannot be hidden by quick or by another carrying the
    // same value.
    const quick = new Set(names(quickI8a8Candidates()));
    const outside = i8a8Candidates().filter(({ name }) => !quick.has(name));
    assert(outside.length >= 5, "the full grid holds at least five geometries outside quick");
    const [smallLinear, linear, attentionQk, attentionPv, conv3d] = outside;
    const profile: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      id: "synthetic-i8a8",
      i8a8: {
        linear: [
          { maxRows: 256, geometry: smallLinear.geometry },
          { maxRows: Number.POSITIVE_INFINITY, geometry: linear.geometry },
        ],
        attentionQk: attentionQk.geometry,
        attentionPv: attentionPv.geometry,
        conv3d: conv3d.geometry,
      },
    };
    const listed = new Set(names(quickPlusI8a8Candidates([profile])));
    for (const [field, candidate] of Object.entries({
      "linear[0]": smallLinear,
      "linear[1]": linear,
      attentionQk,
      attentionPv,
      conv3d,
    })) {
      assert(
        listed.has(candidate.name),
        `i8a8.${field} (${candidate.name}) is missing from quick+`,
      );
    }
  });

  it("conv2d includes the f32 quick+ geometries with tileN 64 / 128", () => {
    const expected = quickPlusGemmCandidates()
      .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
      .map(({ geometry }) => conv2dCandidate(geometry).name);
    includesAll(names(quickPlusConv2dCandidates()), expected, "conv2d");
  });

  it("has no duplicate names, and every candidate passes the runtime's check", () => {
    for (const candidates of [quickPlusGemmCandidates(), quickPlusConv2dCandidates()]) {
      assertEquals(new Set(names(candidates)).size, candidates.length);
      for (const { name, geometry } of candidates) assertGemmGeometry(geometry, name);
    }
    const i8a8 = quickPlusI8a8Candidates();
    assertEquals(new Set(names(i8a8)).size, i8a8.length);
    for (const { name, geometry } of i8a8) assertI8a8Geometry(geometry, name);
  });

  it("the named sets match the per-family set functions (quick / quick+ / full)", () => {
    assertEquals(names(gemmCandidatesIn("quick")), names(quickGemmCandidates()));
    assertEquals(names(gemmCandidatesIn("quick+")), names(quickPlusGemmCandidates()));
    assertEquals(names(gemmCandidatesIn("full")), names(gemmCandidates()));
    assertEquals(names(conv2dCandidatesIn("quick")), names(quickConv2dCandidates()));
    assertEquals(names(conv2dCandidatesIn("quick+")), names(quickPlusConv2dCandidates()));
    assertEquals(names(conv2dCandidatesIn("full")), names(conv2dCandidates()));
    assertEquals(names(i8a8CandidatesIn("quick")), names(quickI8a8Candidates()));
    assertEquals(
      names(i8a8CandidatesIn("quick+")),
      names(
        quickPlusI8a8Candidates().filter(
          ({ geometry }) => !("skeleton" in geometry) && geometry.tileK <= 32,
        ),
      ),
    );
    assertEquals(names(i8a8CandidatesIn("full")), names(i8a8Candidates()));
  });
});

describe("the candidate sets", () => {
  it("are quick+, quick, full and the evaluation set k88 (the evaluation set k85 is retired — its cases are the sweep's own and its grid is the i8a8 linear's full)", () => {
    assertEquals(CANDIDATE_SETS, ["quick+", "quick", "full", "k88"]);
    assertEquals(EVALUATION_SETS, ["k88"]);
  });
});

describe("candidate set k88 (perf-ledger K-88 — the f32 skeleton forms and the conv2d counter decode)", () => {
  const PROFILES = [DEFAULT_GEOMETRY_PROFILE, ...BUILTIN_GEOMETRY_PROFILES];
  const distinct = (geometries: readonly GemmGeometry[]): GemmGeometry[] =>
    geometries.filter(
      (geometry, index) =>
        geometries.findIndex((other) => JSON.stringify(other) === JSON.stringify(geometry)) ===
        index,
    );

  it("a linear / matmul case: the gemmRows geometry of the default and every registered table for its M, each without the form and with each form, the default first", () => {
    for (const rows of [64, 512, 768, 4096]) {
      const bases = distinct(PROFILES.map((profile) => gemmRowsGeometry(profile, rows)));
      const candidates = k88GemmCandidates(rows);
      assertEquals(
        names(candidates),
        bases.flatMap((base) => [
          gemmName(base),
          `${gemmName(base)}:tb-un`,
          `${gemmName(base)}:tas-tb-un`,
        ]),
        `M ${rows}`,
      );
      assertEquals(candidates[0]?.geometry, gemmRowsGeometry(DEFAULT_GEOMETRY_PROFILE, rows));
      for (const { name, geometry } of candidates) assertGemmGeometry(geometry, name);
    }
  });

  it("a conv2d case: the geometry of its m-tile class in the default and every registered table, each with no field, each form, the decode alone and the decode with each form", () => {
    for (const channels of [1024, 96]) {
      const mTile = channels % 64 === 0 ? 64 : 32;
      const bases = distinct(PROFILES.map((profile) => conv2dProfileGeometry(profile, mTile)));
      const candidates = k88Conv2dCandidates(channels);
      assertEquals(
        names(candidates),
        bases.flatMap((base) => {
          const name = names([conv2dCandidate(base)])[0];
          return [
            name,
            `${name}:tb-un`,
            `${name}:tas-tb-un`,
            `${name}:dec`,
            `${name}:tb-un:dec`,
            `${name}:tas-tb-un:dec`,
          ];
        }),
        `Cout ${channels}`,
      );
      // The names are the key's spelling of the fields
      for (const { name, geometry } of candidates) {
        assertEquals(
          conv2dIgemmKey("f32", true, mTile, geometry).includes(
            name.replace(/^igemm(\w+):/, "igemm$1v4:"),
          ),
          true,
          name,
        );
      }
    }
  });

  it("leaves quick / quick+ / full as they were (no form, no decode), and gives no candidates outside a case", () => {
    for (const set of ["quick", "quick+", "full"] as const) {
      for (const { name, geometry } of [...gemmCandidatesIn(set), ...conv2dCandidatesIn(set)]) {
        assert(!("form" in geometry) && !("decode" in geometry), `${set}: ${name}`);
      }
    }
    assertThrows(() => gemmCandidatesIn("k88"), Error, "has no f32 skeleton candidates");
    assertThrows(() => conv2dCandidatesIn("k88"), Error, "has no conv2d candidates");
    assertThrows(() => i8a8CandidatesIn("k88"), Error, "has no i8a8 candidates");
    // A geometry without the fields keeps its name
    assertEquals(gemmCandidate(defaultGemmGeometry()).name, "reg128x128r8x8w16");
    assertEquals(conv2dCandidate(gemmMTileGeometry(64)).name, "igemm64x128:wg16x8");
  });
});

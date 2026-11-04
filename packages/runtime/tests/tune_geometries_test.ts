import { assert, assertEquals } from "@std/assert";
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
  i8a8GeometryKeyPart,
} from "../src/kernels/i8a8-geometry.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { conv2dIgemmKey } from "../src/kernels/conv2d.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import {
  conv2dCandidate,
  conv2dCandidates,
  conv2dCandidatesIn,
  gemmCandidate,
  gemmCandidates,
  gemmCandidatesIn,
  i8a8Candidate,
  i8a8Candidates,
  i8a8CandidatesIn,
  quickConv2dCandidates,
  quickGemmCandidates,
  quickI8a8Candidates,
  quickPlusConv2dCandidates,
  quickPlusGemmCandidates,
  quickPlusI8a8Candidates,
} from "../src/tune/geometries.ts";

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
          i8a8Candidate(profile.i8a8.linear).name,
          i8a8Candidate(profile.i8a8.attentionQk).name,
          i8a8Candidate(profile.i8a8.attentionPv).name,
          i8a8Candidate(profile.i8a8.conv3d).name,
        ],
        `${profile.id} i8a8`,
      );
    }
  });

  it("i8a8 quick+ adds each of the four i8a8 fields of a profile it is given (distinct, all outside quick)", () => {
    // Four distinct geometries from the full grid that quick does not hold, one per field — so a
    // field missing from quick+ cannot be hidden by quick or by another field carrying the same value.
    const quick = new Set(names(quickI8a8Candidates()));
    const outside = i8a8Candidates().filter(({ name }) => !quick.has(name));
    assert(outside.length >= 4, "the full grid holds at least four geometries outside quick");
    const [linear, attentionQk, attentionPv, conv3d] = outside;
    const profile: GeometryProfile = {
      ...DEFAULT_GEOMETRY_PROFILE,
      id: "synthetic-i8a8",
      i8a8: {
        linear: linear.geometry,
        attentionQk: attentionQk.geometry,
        attentionPv: attentionPv.geometry,
        conv3d: conv3d.geometry,
      },
    };
    const listed = new Set(names(quickPlusI8a8Candidates([profile])));
    for (const [field, candidate] of Object.entries({ linear, attentionQk, attentionPv, conv3d })) {
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
    assertEquals(names(i8a8CandidatesIn("quick+")), names(quickPlusI8a8Candidates()));
    assertEquals(names(i8a8CandidatesIn("full")), names(i8a8Candidates()));
  });
});

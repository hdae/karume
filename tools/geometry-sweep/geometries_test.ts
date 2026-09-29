import { assert, assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmGeometryTileKeyPart,
  gemmThreads,
  gemmTileM,
  gemmTileN,
} from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  i8a8GeometryKeyPart,
} from "../../packages/runtime/src/kernels/i8a8-geometry.ts";
import { gemmMTileGeometry } from "../../packages/runtime/src/kernels/gemm.ts";
import { conv2dIgemmKey } from "../../packages/runtime/src/kernels/conv2d.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../../packages/runtime/src/kernels/geometry-profiles/index.ts";
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
} from "./geometries.ts";

const names = (items: readonly { readonly name: string }[]): string[] =>
  items.map((item) => item.name);

const gemmName = (geometry: GemmGeometry): string => gemmGeometryTileKeyPart(geometry);

describe("f32 骨格の候補", () => {
  it("本番の 3 幾何（既定・中 M・小 M バケット）を含む", () => {
    const listed = names(gemmCandidates());
    for (const rows of [4096, 512, 64]) {
      assert(listed.includes(gemmName(gemmGeometryForRows(rows))), `M=${rows}`);
    }
    assert(listed.includes(gemmName(defaultGemmGeometry())));
  });

  it("全候補が runtime の門を通り、threads ≤ 256・タイル辺 ≤ 128 に収まる", () => {
    for (const { name, geometry } of gemmCandidates()) {
      assertGemmGeometry(geometry, name);
      assert(gemmThreads(geometry) <= 256, name);
      assert(gemmTileM(geometry) <= 128 && gemmTileN(geometry) <= 128, name);
      assertEquals(name, gemmName(geometry));
    }
  });

  it("名前（= キー断片）に重複が無い", () => {
    const listed = names(gemmCandidates());
    assertEquals(new Set(listed).size, listed.length);
  });

  it("quick は full の部分集合で、既定を含む", () => {
    const full = new Set(names(gemmCandidates()));
    for (const name of names(quickGemmCandidates())) assert(full.has(name), name);
    assert(names(quickGemmCandidates()).includes(gemmName(defaultGemmGeometry())));
  });
});

describe("conv2d の候補", () => {
  it("n タイルは 64 / 128 だけで、m タイル 64 / 32 の本番幾何を両方含む", () => {
    const candidates = conv2dCandidates();
    for (const { name, geometry } of candidates) {
      assert([64, 128].includes(gemmTileN(geometry)), name);
    }
    const listed = names(candidates);
    for (const mTile of [64, 32]) {
      assert(listed.includes(conv2dCandidate(gemmMTileGeometry(mTile)).name), `mTile ${mTile}`);
    }
  });

  it("名前は本番のパイプラインキーの断片と同じ綴り（v4 無しのキーに含まれる）で、重複が無い", () => {
    const candidates = conv2dCandidates();
    for (const { name, geometry } of candidates) {
      const key = conv2dIgemmKey("f16", false, 64, geometry);
      assert(key.includes(`:${name}:`), `${key} に ${name} が無い`);
    }
    assertEquals(new Set(names(candidates)).size, candidates.length);
  });

  it("quick は full の部分集合", () => {
    const full = new Set(names(conv2dCandidates()));
    for (const name of names(quickConv2dCandidates())) assert(full.has(name), name);
  });
});

describe("i8a8 の候補", () => {
  it("linear / ①QK の既定と ③PV の既定を含み、全候補が runtime の門を通る", () => {
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

  it("quick は full の部分集合", () => {
    const full = new Set(names(i8a8Candidates()));
    for (const name of names(quickI8a8Candidates())) assert(full.has(name), name);
  });
});

describe("候補集合 quick+（quick ∪ 登録済みプロファイルの採用幾何）", () => {
  const includesAll = (listed: readonly string[], expected: readonly string[], label: string) => {
    const set = new Set(listed);
    for (const name of expected) assert(set.has(name), `${label}: ${name} が無い`);
  };

  it("quick を含む（f32・conv2d・i8a8）", () => {
    includesAll(names(quickPlusGemmCandidates()), names(quickGemmCandidates()), "f32");
    includesAll(names(quickPlusConv2dCandidates()), names(quickConv2dCandidates()), "conv2d");
    includesAll(names(quickPlusI8a8Candidates()), names(quickI8a8Candidates()), "i8a8");
  });

  it("登録済みの全プロファイルの全幾何を、その族の集合に含む", () => {
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
        ],
        `${profile.id} i8a8`,
      );
    }
  });

  it("conv2d は f32 の quick+ のうち tileN 64 / 128 のものを含む", () => {
    const expected = quickPlusGemmCandidates()
      .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
      .map(({ geometry }) => conv2dCandidate(geometry).name);
    includesAll(names(quickPlusConv2dCandidates()), expected, "conv2d");
  });

  it("名前に重複が無く、全候補が runtime の門を通る", () => {
    for (const candidates of [quickPlusGemmCandidates(), quickPlusConv2dCandidates()]) {
      assertEquals(new Set(names(candidates)).size, candidates.length);
      for (const { name, geometry } of candidates) assertGemmGeometry(geometry, name);
    }
    const i8a8 = quickPlusI8a8Candidates();
    assertEquals(new Set(names(i8a8)).size, i8a8.length);
    for (const { name, geometry } of i8a8) assertI8a8Geometry(geometry, name);
  });

  it("集合の名前は族ごとの集合の関数と一致する（quick / quick+ / full）", () => {
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

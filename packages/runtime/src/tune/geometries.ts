/**
 * Geometry candidates for the sweep (pure functions — perf-ledger K-70).
 *
 * A candidate is "the Cartesian product of a grid filtered by the runtime's checks
 * (`assertGemmGeometry` / `assertI8a8Geometry`)"; a combination that fails the checks leaves holes
 * in the fill of the shared tile, so it is dropped before measuring (measuring it would give wrong
 * values). Display names use the same spelling as the pipeline-key fragments
 * (`gemmGeometryTileKeyPart` / `i8a8GeometryKeyPart` / for conv2d the `igemm…:wg…` of
 * `conv2dIgemmKey`, all without v4), so they can be read against the key list of the diagnostics.
 *
 * The range caps (`threads ≤ 256`, tile edge ≤ 128) put the default geometry (M128N128 / 256
 * threads) at the ceiling: it stays inside WebGPU's default limits (workgroup of 256 invocations,
 * 16,384 B of shared memory), and the f32 skeleton's shared `64·(tileM + tileN)` B is exactly
 * 16,384 B at 128 + 128.
 *
 * There are three candidate sets ({@link CandidateSet}): `quick` (the default + 4–5 shapes),
 * `quick+` (quick ∪ the geometries adopted by the registered profiles — the default) and `full`
 * (the whole grid).
 */
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryTileKeyPart,
  gemmThreads,
  gemmTileM,
  gemmTileN,
} from "../kernels/gemm-geometry.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
} from "../kernels/i8a8-geometry.ts";
import { CodegenError } from "../codegen/errors.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../kernels/geometry-profiles/index.ts";

/** One candidate for the f32 skeleton (linear / fused attention / conv2d's implicit GEMM). */
export type GemmCandidate = {
  readonly family: "gemm";
  /**
   * `reg{tileM}x{tileN}r{regM}x{regN}w{wgX}` for linear / fused attention, and
   * `igemm{tileM}x{tileN}:wg{wgX}x{wgY}` for conv2d (both the same spelling as the op's key
   * fragment, without v4).
   */
  readonly name: string;
  readonly geometry: GemmGeometry;
};

/** One candidate for the i8a8 family (linear / fused attention's ①QK and ③PV). */
export type I8a8Candidate = {
  readonly family: "i8a8";
  /**
   * `tile{tileM}x{tileN}r{regM}x{regN}w{wgX}x{wgY}k{tileK}` (same spelling as the key fragment,
   * without v4).
   */
  readonly name: string;
  readonly geometry: I8a8Geometry;
};

export type GeometryCandidate = GemmCandidate | I8a8Candidate;

const MAX_THREADS = 256;
const MAX_TILE = 128;

export const gemmCandidate = (geometry: GemmGeometry): GemmCandidate => ({
  family: "gemm",
  name: gemmGeometryTileKeyPart(geometry),
  geometry,
});

/**
 * A candidate for conv2d's implicit GEMM (the name is the fragment
 * `igemm{tileM}x{tileN}{v4}:wg{wgX}x{wgY}` of `conv2dIgemmKey` in src/kernels/conv2d.ts without v4
 * — the edges and the workgroup shape fix `regM = tileM / wgY` and `regN = tileN / wgX`, so it maps
 * one-to-one to the geometry).
 */
export const conv2dCandidate = (geometry: GemmGeometry): GemmCandidate => ({
  family: "gemm",
  name: `igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}:wg${geometry.wgX}x${geometry.wgY}`,
  geometry,
});

export const i8a8Candidate = (geometry: I8a8Geometry): I8a8Candidate => ({
  family: "i8a8",
  name: i8a8GeometryKeyPart(geometry, false),
  geometry,
});

/**
 * Whether it passes the check (a combination that fails is dropped from the candidates — only
 * CodegenError is expected as the exception type).
 */
const passes = (check: () => void): boolean => {
  try {
    check();
    return true;
  } catch (error) {
    if (error instanceof CodegenError) return false;
    throw error;
  }
};

/** Removes duplicates by name (keeps the first occurrence — preserves the caller's order). */
const unique = <T extends { readonly name: string }>(items: readonly T[]): T[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.name)) return false;
    seen.add(item.name);
    return true;
  });
};

/**
 * All candidates for the f32 skeleton (regM ∈ {1,2,4,8}, regN ∈ {4,8}, wgX / wgY ∈ {4,8,16},
 * filtered by the checks and the range). The default `{8,8,16,16}`, the medium-M bucket
 * `{4,4,8,16}` and the small-M bucket `{1,4,4,16}` are in this grid (tune_geometries_test.ts pins
 * it).
 */
export const gemmCandidates = (): GemmCandidate[] => {
  const found: GemmCandidate[] = [];
  for (const regM of [1, 2, 4, 8]) {
    for (const regN of [4, 8]) {
      for (const wgX of [4, 8, 16]) {
        for (const wgY of [4, 8, 16]) {
          const geometry: GemmGeometry = { regM, regN, wgX, wgY };
          if (
            gemmThreads(geometry) <= MAX_THREADS &&
            gemmTileM(geometry) <= MAX_TILE &&
            gemmTileN(geometry) <= MAX_TILE &&
            passes(() => assertGemmGeometry(geometry, "geometry-sweep"))
          ) {
            found.push(gemmCandidate(geometry));
          }
        }
      }
    }
  }
  return unique(found);
};

/**
 * All candidates for the i8a8 family (regM / regN ∈ {4,8}, wgX ∈ {8,16}, wgY ∈ {4,8,16},
 * tileK ∈ {16,32}, filtered by the checks). The default of linear / ①QK `{8,8,8,16,16}` and the
 * default of ③PV `{8,8,16,8,16}` are in this grid.
 */
export const i8a8Candidates = (): I8a8Candidate[] => {
  const found: I8a8Candidate[] = [];
  for (const regM of [4, 8]) {
    for (const regN of [4, 8]) {
      for (const wgX of [8, 16]) {
        for (const wgY of [4, 8, 16]) {
          for (const tileK of [16, 32]) {
            const geometry: I8a8Geometry = { regM, regN, wgX, wgY, tileK };
            if (passes(() => assertI8a8Geometry(geometry, "geometry-sweep"))) {
              found.push(i8a8Candidate(geometry));
            }
          }
        }
      }
    }
  }
  return unique(found);
};

/**
 * Candidates for conv2d's implicit GEMM = the f32 skeleton candidates with tileN ∈ {64, 128} (the
 * production n tile 128 and its half — the m tile spans the grid as is, including the production
 * values 32 / 64).
 */
export const conv2dCandidates = (): GemmCandidate[] =>
  gemmCandidates()
    .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
    .map(({ geometry }) => conv2dCandidate(geometry));

/**
 * The f32 set of `--quick` (CLI flag, removed in 0.15) (the default + 64×64/256, 64×64/128,
 * 64×32/128, 32×32/64). 64×64/128 is `r8×4 wg16×8`, the best of the 64×64 sweep of 2026-08-10;
 * 64×32/128 is the medium-M bucket itself.
 */
export const QUICK_GEMM: readonly GemmGeometry[] = [
  defaultGemmGeometry(),
  { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  { regM: 8, regN: 4, wgX: 16, wgY: 8 },
  { regM: 4, regN: 4, wgX: 8, wgY: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 8 },
];

/**
 * The i8a8 set of `--quick` (CLI flag, removed in 0.15) (the two defaults + the counterparts of the
 * same four shapes as f32, tileK 16).
 */
export const QUICK_I8A8: readonly I8a8Geometry[] = [
  defaultI8a8Geometry("linear"),
  defaultI8a8Geometry("attention_pv"),
  { regM: 4, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
  { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 16, tileK: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 8, tileK: 16 },
];

export const quickGemmCandidates = (): GemmCandidate[] => unique(QUICK_GEMM.map(gemmCandidate));

export const quickI8a8Candidates = (): I8a8Candidate[] => unique(QUICK_I8A8.map(i8a8Candidate));

/** The conv2d set of `--quick` (CLI flag, removed in 0.15) (f32's quick with tileN ∈ {64, 128}). */
export const quickConv2dCandidates = (): GemmCandidate[] =>
  quickGemmCandidates()
    .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
    .map(({ geometry }) => conv2dCandidate(geometry));

/**
 * The f32 set of `quick+` = quick ∪ every `gemmRows` rule and `attention.qk` / `pv` of the
 * registered profiles (`BUILTIN_GEOMETRY_PROFILES`). The geometries of the registered tables are
 * re-measured every time to read, next to the default and within quick's time, "whether that table
 * is fast on this machine too" (the generator only considers geometries measured in every case of a
 * field, so if a table's adopted geometry is missing from the set, a sweep that rebuilds the table
 * regresses that field to the default).
 */
export const quickPlusGemmCandidates = (): GemmCandidate[] =>
  unique([
    ...quickGemmCandidates(),
    ...BUILTIN_GEOMETRY_PROFILES.flatMap((profile) => [
      ...profile.gemmRows.map((rule) => gemmCandidate(rule.geometry)),
      gemmCandidate(profile.attention.qk),
      gemmCandidate(profile.attention.pv),
    ]),
  ]);

/**
 * The conv2d set of `quick+` = f32's quick+ with tileN ∈ {64, 128} ∪ `conv2d.rows64` / `rows32` of
 * the registered profiles (a table's conv2d geometries are added regardless of the tileN
 * condition).
 */
export const quickPlusConv2dCandidates = (): GemmCandidate[] =>
  unique([
    ...quickPlusGemmCandidates()
      .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
      .map(({ geometry }) => conv2dCandidate(geometry)),
    ...BUILTIN_GEOMETRY_PROFILES.flatMap((profile) => [
      conv2dCandidate(profile.conv2d.rows64),
      conv2dCandidate(profile.conv2d.rows32),
    ]),
  ]);

/** The i8a8 set of `quick+` = quick ∪ the three `i8a8` fields of the registered profiles. */
export const quickPlusI8a8Candidates = (): I8a8Candidate[] =>
  unique([
    ...quickI8a8Candidates(),
    ...BUILTIN_GEOMETRY_PROFILES.flatMap((profile) => [
      i8a8Candidate(profile.i8a8.linear),
      i8a8Candidate(profile.i8a8.attentionQk),
      i8a8Candidate(profile.i8a8.attentionPv),
    ]),
  ]);

/**
 * The vocabulary of candidate sets (the CLI's `--set` and the page's select — both removed in
 * 0.15 — and the JSON's `settings.candidateSet`).
 */
export const CANDIDATE_SETS = ["quick+", "quick", "full"] as const;

/** A candidate set (`quick` / `quick+` / `full` — {@link CANDIDATE_SETS}). */
export type CandidateSet = (typeof CANDIDATE_SETS)[number];

/**
 * The default when no set is given (`quick+` — measures up to the registered tables' geometries
 * within quick's time).
 */
export const DEFAULT_CANDIDATE_SET: CandidateSet = "quick+";

export const isCandidateSet = (value: string): value is CandidateSet =>
  (CANDIDATE_SETS as readonly string[]).includes(value);

/** The set for the f32 skeleton (linear / matmul / bmm / fused attention). */
export const gemmCandidatesIn = (set: CandidateSet): GemmCandidate[] =>
  set === "quick"
    ? quickGemmCandidates()
    : set === "quick+"
      ? quickPlusGemmCandidates()
      : gemmCandidates();

/** The set for conv2d's implicit GEMM. */
export const conv2dCandidatesIn = (set: CandidateSet): GemmCandidate[] =>
  set === "quick"
    ? quickConv2dCandidates()
    : set === "quick+"
      ? quickPlusConv2dCandidates()
      : conv2dCandidates();

/** The set for the i8a8 family. */
export const i8a8CandidatesIn = (set: CandidateSet): I8a8Candidate[] =>
  set === "quick"
    ? quickI8a8Candidates()
    : set === "quick+"
      ? quickPlusI8a8Candidates()
      : i8a8Candidates();

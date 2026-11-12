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
 * There are three candidate sets a table is derived from ({@link CandidateSet}): `quick` (the
 * default + 4–5 shapes), `quick+` (quick ∪ the geometries adopted by the registered profiles — the
 * default) and `full` (the whole grid — for the i8a8 linear a grid of its own with the vec4
 * skeleton and the K64 tiles, {@link i8a8LinearCandidates}). One more, `k88`, is an evaluation set
 * ({@link EVALUATION_SETS}): the f32 skeleton forms with the conv2d counter decode (perf-ledger
 * K-88), on cases of its own, outside the sweep's case set, so it changes neither the other sets
 * nor any fingerprint.
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
  i8a8WorkgroupStorageBytes,
} from "../kernels/i8a8-geometry.ts";
import { CodegenError } from "../codegen/errors.ts";
import { conv2dIgemmMTile } from "../kernels/conv2d.ts";
import {
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
} from "../kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../kernels/geometry-profiles/index.ts";

/** One candidate for the f32 skeleton (linear / fused attention / conv2d's implicit GEMM). */
export type GemmCandidate = {
  readonly family: "gemm";
  /**
   * `reg{tileM}x{tileN}r{regM}x{regN}w{wgX}` for linear / fused attention, and
   * `igemm{tileM}x{tileN}:wg{wgX}x{wgY}` for conv2d (both the same spelling as the op's key
   * fragment, without v4), followed by the geometry's skeleton form (`:tb-un` / `:tas-tb-un`) and,
   * for conv2d, its counter decode (`:dec`) when it carries them (perf-ledger K-88).
   */
  readonly name: string;
  readonly geometry: GemmGeometry;
};

/** One candidate for the i8a8 family (linear / fused attention's ①QK and ③PV / conv3d w8a8). */
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

/** The name part of the optional skeleton form (the key's spelling — empty without it). */
const formPart = (geometry: GemmGeometry): string =>
  geometry.form === undefined ? "" : `:${geometry.form}`;

export const gemmCandidate = (geometry: GemmGeometry): GemmCandidate => ({
  family: "gemm",
  name: `${gemmGeometryTileKeyPart(geometry)}${formPart(geometry)}`,
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
  name: `igemm${gemmTileM(geometry)}x${gemmTileN(geometry)}:wg${geometry.wgX}x${geometry.wgY}${formPart(
    geometry,
  )}${geometry.decode === undefined ? "" : ":dec"}`,
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

/**
 * The i8a8 set of `quick+` = quick ∪ the `i8a8` fields of the registered profiles (every rule of
 * `i8a8.linear`, then ①QK, ③PV and conv3d — `profiles` is the embedded tables unless a test passes
 * its own list).
 */
export const quickPlusI8a8Candidates = (
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): I8a8Candidate[] =>
  unique([
    ...quickI8a8Candidates(),
    ...profiles.flatMap((profile) => [
      ...profile.i8a8.linear.map((rule) => i8a8Candidate(rule.geometry)),
      i8a8Candidate(profile.i8a8.attentionQk),
      i8a8Candidate(profile.i8a8.attentionPv),
      i8a8Candidate(profile.i8a8.conv3d),
    ]),
  ]);

/**
 * The workgroup storage every WebGPU device provides (the default of
 * `maxComputeWorkgroupStorageSize`, 16,384 B). The skeleton grid and the i8a8 linear's `full`
 * keep to it as an embedded table must:
 * the automatic selection path does not run the limit check (geometry-profile.ts).
 */
const DEFAULT_WORKGROUP_STORAGE = 16384;

/**
 * Whether a geometry fits the workgroup storage every device provides
 * ({@link DEFAULT_WORKGROUP_STORAGE}).
 */
const fitsEveryDevice = (geometry: I8a8Geometry): boolean =>
  i8a8WorkgroupStorageBytes(geometry) <= DEFAULT_WORKGROUP_STORAGE;

/**
 * The skeleton grid of the i8a8 linear (perf-ledger K-85): regM ∈ {8, 12, 16} × regN ∈ {4, 8} ×
 * wgX ∈ {8, 16} × wgY ∈ {4, 8, 16} × tileK ∈ {16, 32, 64}, each without and then with the vec4
 * skeleton, filtered by the runtime's check and {@link DEFAULT_WORKGROUP_STORAGE}.
 */
const skeletonGridCandidates = (): I8a8Candidate[] => {
  const grid: I8a8Candidate[] = [];
  for (const regM of [8, 12, 16]) {
    for (const regN of [4, 8]) {
      for (const wgX of [8, 16]) {
        for (const wgY of [4, 8, 16]) {
          for (const tileK of [16, 32, 64]) {
            const word: I8a8Geometry = { regM, regN, wgX, wgY, tileK };
            for (const geometry of [word, { ...word, skeleton: "vec4" as const }]) {
              if (
                fitsEveryDevice(geometry) &&
                passes(() => assertI8a8Geometry(geometry, "geometry-sweep"))
              ) {
                grid.push(i8a8Candidate(geometry));
              }
            }
          }
        }
      }
    }
  }
  return grid;
};

/**
 * The i8a8 linear's candidates in `full` (ADR 0115 addendum decision 14): the i8a8 grid of
 * {@link i8a8Candidates}, then the skeleton grid (the K64 tiles and the vec4 skeleton included —
 * {@link skeletonGridCandidates}), deduplicated by name and filtered to
 * {@link DEFAULT_WORKGROUP_STORAGE} as an embedded table must be. The i8a8 attention and conv3d keep
 * {@link i8a8Candidates}.
 *
 * NOTE: the i8a8 linear's sweep cases are i8 weights. A K64 geometry is refused for w4a8 at group
 * 32 (`assertI8a8GroupGeometry`); the recipe builder then runs the default geometry instead.
 */
export const i8a8LinearCandidates = (): I8a8Candidate[] =>
  unique([...i8a8Candidates(), ...skeletonGridCandidates()]).filter(({ geometry }) =>
    fitsEveryDevice(geometry),
  );

/** The skeleton forms of the f32 GEMM (perf-ledger K-88), in the order the `k88` set measures them. */
const K88_FORMS = ["tb-un", "tas-tb-un"] as const;

/**
 * The geometries of a `k88` case: each base geometry (`bases` — deduplicated, in order) without
 * the fields, then with each skeleton form, and for conv2d (`decode`) the counter decode alone and
 * with each form.
 */
const k88Variants = (bases: readonly GemmGeometry[], decode: boolean): GemmGeometry[] => {
  const seen = new Set<string>();
  return bases
    .filter((base) => {
      const key = JSON.stringify(base);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .flatMap((base) => [
      base,
      ...K88_FORMS.map((form) => ({ ...base, form })),
      ...(decode
        ? [
            { ...base, decode: "counter" as const },
            ...K88_FORMS.map((form) => ({ ...base, form, decode: "counter" as const })),
          ]
        : []),
    ]);
};

/**
 * The f32 candidates of a `k88` linear / matmul / bmm case of `rows` rows (perf-ledger K-88): the
 * `gemmRows` geometry the default profile and every registered table (`profiles` — the adapter's
 * table among them) give that row count, each without the field and with each skeleton form. The
 * sweep puts the default geometry at the front.
 */
export const k88GemmCandidates = (
  rows: number,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): GemmCandidate[] =>
  unique(
    k88Variants(
      [DEFAULT_GEOMETRY_PROFILE, ...profiles].map((profile) => gemmRowsGeometry(profile, rows)),
      false,
    ).map(gemmCandidate),
  );

/**
 * The conv2d candidates of a `k88` case with `channelsOut` output channels (perf-ledger K-88): the
 * `conv2d` geometry of that m-tile class in the default profile and every registered table, each
 * without the fields, with each skeleton form, with the counter decode alone and with the decode
 * and each form.
 */
export const k88Conv2dCandidates = (
  channelsOut: number,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): GemmCandidate[] => {
  const mTile = conv2dIgemmMTile(channelsOut);
  return unique(
    k88Variants(
      [DEFAULT_GEOMETRY_PROFILE, ...profiles].map((profile) =>
        conv2dProfileGeometry(profile, mTile),
      ),
      true,
    ).map(conv2dCandidate),
  );
};

/**
 * The vocabulary of candidate sets (the CLI's `--set` and the page's select — both removed in
 * 0.15 — and the JSON's `settings.candidateSet`).
 */
export const CANDIDATE_SETS = ["quick+", "quick", "full", "k88"] as const;

/** A candidate set (`quick` / `quick+` / `full` / `k88` — {@link CANDIDATE_SETS}). */
export type CandidateSet = (typeof CANDIDATE_SETS)[number];

/**
 * The evaluation sets: a kernel change measured on cases of its own (cases.ts's `sweepCasesFor`),
 * outside the case set a table is derived from. Their records carry the case-set version and the
 * default's fingerprint of their own cases, and the generator derives no table from them (derive.ts).
 */
export const EVALUATION_SETS: readonly CandidateSet[] = ["k88"];

/** The perf-ledger row an evaluation set measures (`k88` → `K-88`). */
export const evaluationSetLedgerId = (set: CandidateSet): string => `K-${set.replace(/^k/, "")}`;

/**
 * The default when no set is given (`quick+` — measures up to the registered tables' geometries
 * within quick's time).
 */
export const DEFAULT_CANDIDATE_SET: CandidateSet = "quick+";

export const isCandidateSet = (value: string): value is CandidateSet =>
  (CANDIDATE_SETS as readonly string[]).includes(value);

/**
 * An evaluation set has no candidates for a family its cases do not have (`k88` gives its f32
 * candidates per case — {@link k88GemmCandidates}, {@link k88Conv2dCandidates}).
 */
const noCandidates = (set: CandidateSet, family: string, reason: string): never => {
  throw new Error(`Candidate set ${set} has no ${family} candidates (${reason})`);
};

const K88_PER_CASE =
  "they depend on the case's row count or m-tile class — candidatesFor gives them per case";

/** The set for the f32 skeleton (linear / matmul / bmm / fused attention). */
export const gemmCandidatesIn = (set: CandidateSet): GemmCandidate[] => {
  switch (set) {
    case "quick":
      return quickGemmCandidates();
    case "quick+":
      return quickPlusGemmCandidates();
    case "full":
      return gemmCandidates();
    case "k88":
      return noCandidates(set, "f32 skeleton", K88_PER_CASE);
  }
};

/** The set for conv2d's implicit GEMM. */
export const conv2dCandidatesIn = (set: CandidateSet): GemmCandidate[] => {
  switch (set) {
    case "quick":
      return quickConv2dCandidates();
    case "quick+":
      return quickPlusConv2dCandidates();
    case "full":
      return conv2dCandidates();
    case "k88":
      return noCandidates(set, "conv2d", K88_PER_CASE);
  }
};

/**
 * The set for the i8a8 linear: `full` is its own grid ({@link i8a8LinearCandidates}), every other
 * set the i8a8 family's ({@link i8a8CandidatesIn}).
 */
export const i8a8LinearCandidatesIn = (set: CandidateSet): I8a8Candidate[] =>
  set === "full" ? i8a8LinearCandidates() : i8a8CandidatesIn(set);

/** The set for the i8a8 family (the i8a8 attention and conv3d — the linear's `full` differs). */
export const i8a8CandidatesIn = (set: CandidateSet): I8a8Candidate[] => {
  switch (set) {
    case "quick":
      return quickI8a8Candidates();
    case "quick+":
      return quickPlusI8a8Candidates();
    case "full":
      return i8a8Candidates();
    case "k88":
      return noCandidates(set, "i8a8", "it measures the f32 skeleton and conv2d");
  }
};

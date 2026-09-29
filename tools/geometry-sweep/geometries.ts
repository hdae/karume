/**
 * 掃引する幾何の候補（純関数 — perf-ledger K-70）。
 *
 * 候補は「格子の直積を runtime の門（`assertGemmGeometry` / `assertI8a8Geometry`）で濾したもの」で、
 * 門を通らない組は共有タイルの充填に穴が空く形なので、測る前に落とす（測っても誤値になる）。
 * 表示名はパイプラインキーの断片と同じ綴り（`gemmGeometryTileKeyPart` / `i8a8GeometryKeyPart` /
 * conv2d は `conv2dIgemmKey` の `igemm…:wg…`・いずれも v4 抜き）で、診断のキー一覧と突き合わせて読める。
 *
 * 範囲の上限（`threads ≤ 256`・タイル辺 ≤ 128）は既定幾何（M128N128 / 256 スレッド）を天井に置く
 * もの: WebGPU の既定上限（workgroup 256 invocation・共有 16,384 B）の内側に収まり、f32 骨格の
 * 共有 `64·(tileM + tileN)` B は 128 + 128 でちょうど 16,384 B になる。
 *
 * 候補集合は 3 段（{@link CandidateSet}）: `quick`（既定 + 4〜5 形）・`quick+`（quick ∪ 登録済み
 * プロファイルの採用幾何 — 既定）・`full`（格子全体）。
 */
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  type GemmGeometry,
  gemmGeometryTileKeyPart,
  gemmThreads,
  gemmTileM,
  gemmTileN,
} from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
} from "../../packages/runtime/src/kernels/i8a8-geometry.ts";
import { CodegenError } from "../../packages/runtime/src/codegen/errors.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../../packages/runtime/src/kernels/geometry-profiles/index.ts";

/** f32 骨格（linear / 融合 attention / conv2d の implicit GEMM）の候補 1 つ。 */
export type GemmCandidate = {
  readonly family: "gemm";
  /**
   * linear / 融合 attention は `reg{tileM}x{tileN}r{regM}x{regN}w{wgX}`、conv2d は
   * `igemm{tileM}x{tileN}:wg{wgX}x{wgY}`（どちらも op のキー断片と同じ綴り・v4 無し）。
   */
  readonly name: string;
  readonly geometry: GemmGeometry;
};

/** i8a8 族（linear / 融合 attention の ①QK・③PV）の候補 1 つ。 */
export type I8a8Candidate = {
  readonly family: "i8a8";
  /** `tile{tileM}x{tileN}r{regM}x{regN}w{wgX}x{wgY}k{tileK}`（キー断片と同じ綴り・v4 無し）。 */
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
 * conv2d の implicit GEMM の候補（名前は src/kernels/conv2d.ts の `conv2dIgemmKey` の断片
 * `igemm{tileM}x{tileN}{v4}:wg{wgX}x{wgY}` から v4 を抜いた綴り — 辺と workgroup 形で
 * `regM = tileM / wgY`・`regN = tileN / wgX` が決まるので幾何と 1 対 1）。
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

/** 門を通るか（通らない組は候補から落とす — 例外の種類は CodegenError だけを期待する）。 */
const passes = (check: () => void): boolean => {
  try {
    check();
    return true;
  } catch (error) {
    if (error instanceof CodegenError) return false;
    throw error;
  }
};

/** 名前で重複を除く（先に現れたものを残す — 呼び手が並べた順を保つ）。 */
const unique = <T extends { readonly name: string }>(items: readonly T[]): T[] => {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.name)) return false;
    seen.add(item.name);
    return true;
  });
};

/**
 * f32 骨格の全候補（regM ∈ {1,2,4,8}・regN ∈ {4,8}・wgX / wgY ∈ {4,8,16} を門と範囲で濾す）。
 * 既定 `{8,8,16,16}`・中 M バケット `{4,4,8,16}`・小 M バケット `{1,4,4,16}` はこの格子に入る
 * （geometries_test.ts が固定する）。
 */
export const gemmCandidates = (): GemmCandidate[] => {
  const found: GemmCandidate[] = [];
  for (const regM of [1, 2, 4, 8]) {
    for (const regN of [4, 8]) {
      for (const wgX of [4, 8, 16]) {
        for (const wgY of [4, 8, 16]) {
          const geometry: GemmGeometry = { regM, regN, wgX, wgY };
          if (
            gemmThreads(geometry) <= MAX_THREADS && gemmTileM(geometry) <= MAX_TILE &&
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
 * i8a8 族の全候補（regM / regN ∈ {4,8}・wgX ∈ {8,16}・wgY ∈ {4,8,16}・tileK ∈ {16,32} を門で濾す）。
 * linear / ①QK の既定 `{8,8,8,16,16}` と ③PV の既定 `{8,8,16,8,16}` はこの格子に入る。
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
 * conv2d の implicit GEMM の候補 = f32 骨格の候補のうち tileN ∈ {64, 128}（本番の n タイル 128 と
 * その半分 — m タイルは 32 / 64 の本番値を含めて格子のまま振る）。
 */
export const conv2dCandidates = (): GemmCandidate[] =>
  gemmCandidates()
    .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
    .map(({ geometry }) => conv2dCandidate(geometry));

/**
 * `--quick` の f32 集合（既定 + 64×64/256・64×64/128・64×32/128・32×32/64）。64×64/128 は
 * 2026-08-10 の 64×64 掃引の最良 `r8×4 wg16×8`、64×32/128 は中 M バケットそのもの。
 */
export const QUICK_GEMM: readonly GemmGeometry[] = [
  defaultGemmGeometry(),
  { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  { regM: 8, regN: 4, wgX: 16, wgY: 8 },
  { regM: 4, regN: 4, wgX: 8, wgY: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 8 },
];

/** `--quick` の i8a8 集合（2 つの既定 + f32 と同じ 4 形の相当品・tileK 16）。 */
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

/** `--quick` の conv2d 集合（f32 の quick のうち tileN ∈ {64, 128}）。 */
export const quickConv2dCandidates = (): GemmCandidate[] =>
  quickGemmCandidates()
    .filter(({ geometry }) => [64, 128].includes(gemmTileN(geometry)))
    .map(({ geometry }) => conv2dCandidate(geometry));

/**
 * `quick+` の f32 集合 = quick ∪ 登録済みプロファイル（`BUILTIN_GEOMETRY_PROFILES`）の `gemmRows` の
 * 全規則と `attention.qk` / `pv`。登録済みの表の幾何を毎回測り直すのは、既定と並べて「その表が
 * この機でも速いか」を quick の時間で読むため（生成器は欄の全ケースで測った幾何しか候補にしない
 * ので、表の採用幾何が集合に無いと、表を作り直す掃引でその欄が既定へ後退する）。
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
 * `quick+` の conv2d 集合 = f32 の quick+ のうち tileN ∈ {64, 128} ∪ 登録済みプロファイルの
 * `conv2d.rows64` / `rows32`（表の conv2d 幾何は tileN の条件に依らず足す）。
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

/** `quick+` の i8a8 集合 = quick ∪ 登録済みプロファイルの `i8a8` の 3 欄。 */
export const quickPlusI8a8Candidates = (): I8a8Candidate[] =>
  unique([
    ...quickI8a8Candidates(),
    ...BUILTIN_GEOMETRY_PROFILES.flatMap((profile) => [
      i8a8Candidate(profile.i8a8.linear),
      i8a8Candidate(profile.i8a8.attentionQk),
      i8a8Candidate(profile.i8a8.attentionPv),
    ]),
  ]);

/** 候補集合の語彙（CLI の `--set`・ページの select・JSON の `settings.candidateSet`）。 */
export const CANDIDATE_SETS = ["quick+", "quick", "full"] as const;

export type CandidateSet = typeof CANDIDATE_SETS[number];

/** 集合を指定しないときの既定（`quick+` — 登録済みの表の幾何まで quick の時間で測る）。 */
export const DEFAULT_CANDIDATE_SET: CandidateSet = "quick+";

export const isCandidateSet = (value: string): value is CandidateSet =>
  (CANDIDATE_SETS as readonly string[]).includes(value);

/** f32 骨格（linear / matmul / bmm / 融合 attention）の集合。 */
export const gemmCandidatesIn = (set: CandidateSet): GemmCandidate[] =>
  set === "quick"
    ? quickGemmCandidates()
    : set === "quick+"
    ? quickPlusGemmCandidates()
    : gemmCandidates();

/** conv2d の implicit GEMM の集合。 */
export const conv2dCandidatesIn = (set: CandidateSet): GemmCandidate[] =>
  set === "quick"
    ? quickConv2dCandidates()
    : set === "quick+"
    ? quickPlusConv2dCandidates()
    : conv2dCandidates();

/** i8a8 族の集合。 */
export const i8a8CandidatesIn = (set: CandidateSet): I8a8Candidate[] =>
  set === "quick"
    ? quickI8a8Candidates()
    : set === "quick+"
    ? quickPlusI8a8Candidates()
    : i8a8Candidates();

/**
 * The exploration of the GEMV sweep's `explore` set (device-free): which cases get the whole grid,
 * and which kernels every other case measures at M = 1.
 *
 * The `full` set times every candidate of every case. On one device the winners are regular — a
 * few sequential variants and parallel rows win almost every shape, and which one wins follows the
 * storage, the words of a weight row and n — while no two devices share them. So a device's own
 * measurement cannot be skipped, but a few of its shapes predict the rest:
 * 1. the **sentinels** ({@link gemvSentinels}) — a few cases spread over the table's shapes — are
 *    measured with every candidate;
 * 2. every other case measures, at M = 1, the references, the winners of its nearest sentinel of
 *    the same storage ({@link gemvExploreSelection}: the fastest sequential variant, and the fastest
 *    plain or staged parallel row with every other one the derivation could adopt there, whose
 *    checks held) and their neighbours on the candidate grid; a case for which no sentinel of its
 *    storage has winners measures every candidate. A neighbour is what finds the winner when the
 *    prediction is one step off; without them the prediction alone loses it on most shapes of a
 *    device whose winners vary.
 * What M = 4 and 8 measure of a case follows from its own M = 1 group
 * (gemv-candidates.ts's `gemvExploreCarriedFromM1`).
 *
 * A record of the set holds, at M = 1, rows of the kernels it measured and no row for the others,
 * and at M = 4 and 8 a skipped row for each reorder candidate M = 1 does not carry; so the
 * derivation reads it as it reads any record that holds part of the candidates: an unmeasured
 * candidate has no evidence in it (a skipped row is none either), and the references are measured
 * in every group.
 */

import {
  GEMV_COLS,
  GEMV_LANES,
  GEMV_MIN_SPEEDUP,
  GEMV_UNROLLS,
  type GemvCandidateClass,
  gemvParallelName,
  gemvProfileChoice,
  gemvSequentialName,
} from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase, type GemvSweepStorage } from "./gemv-cases.ts";

/**
 * How many cases of a table are sentinels. Replayed on the GEMV sweep records of two devices
 * (2026-11-19, two runs each, the derivation's decisions against those of the full records): 6, 8,
 * 12 and 16 sentinels leave 18, 14, 8 and 5 of 214 decisions different on the Apple M2 and 1, 0, 0
 * and 0 of 116 on the RTX 5070 Ti. Twelve is where the M2's count stops falling fast; each sentinel
 * costs the whole grid of one case.
 */
export const GEMV_SENTINEL_COUNT = 12;

/** The weight elements one 32-bit word of a storage holds. */
const ELEMENTS_PER_WORD: Readonly<Record<GemvSweepStorage, number>> = {
  i2: 16,
  i4: 8,
  i8: 4,
  bf16: 2,
  f16: 2,
  f32: 1,
};

/** Any two storages are further apart than any two shapes of one storage. */
const STORAGE_DISTANCE = 100;

/** Two i4 group lengths (or a group length and none) of otherwise equal shapes. */
const GROUP_DISTANCE = 0.5;

/**
 * How far two cases are apart as shapes: the octaves between the words of their weight rows and
 * between their column counts — the two sizes a kernel's loop and grid follow —, the storage (which
 * decides the decode, so a winner does not carry across it) and the i4 group length.
 */
export const gemvCaseDistance = (a: GemvCase, b: GemvCase): number =>
  (a.storage === b.storage ? 0 : STORAGE_DISTANCE) +
  (a.group === b.group ? 0 : GROUP_DISTANCE) +
  Math.abs(
    Math.log2(a.k / ELEMENTS_PER_WORD[a.storage]) - Math.log2(b.k / ELEMENTS_PER_WORD[b.storage]),
  ) +
  Math.abs(Math.log2(a.n) - Math.log2(b.n));

/**
 * The sentinels of a list of cases, by id, in the order chosen: first the case with the smallest
 * total distance to the others, then, again and again, the case farthest from every one chosen (a
 * tie goes to the earlier case of the list). So the choice depends on the list alone: the same
 * table gives the same sentinels on every device and in every run, which is what lets two records
 * of one device hold the same rows. A list of at most `count` cases is all sentinels.
 */
export const gemvSentinels = (
  cases: readonly GemvCase[],
  count: number = GEMV_SENTINEL_COUNT,
): string[] => {
  if (!Number.isInteger(count) || count < 1) {
    throw new Error(`The sentinel count must be an integer of at least 1 (${count})`);
  }
  if (cases.length <= count) return cases.map(({ id }) => id);
  const total = (one: GemvCase): number =>
    cases.reduce((sum, other) => sum + gemvCaseDistance(one, other), 0);
  let centre = cases[0];
  if (centre === undefined) return [];
  let smallest = total(centre);
  for (const one of cases.slice(1)) {
    const sum = total(one);
    if (sum < smallest) {
      centre = one;
      smallest = sum;
    }
  }
  const chosen: GemvCase[] = [centre];
  while (chosen.length < count) {
    let farthest: GemvCase | undefined;
    let reach = -1;
    for (const one of cases) {
      const nearest = Math.min(...chosen.map((sentinel) => gemvCaseDistance(sentinel, one)));
      if (nearest > reach) {
        reach = nearest;
        farthest = one;
      }
    }
    // Every case left is a chosen one (two cases of one shape): nothing more to add
    if (farthest === undefined || reach === 0) break;
    chosen.push(farthest);
  }
  return chosen.map(({ id }) => id);
};

/**
 * The cases of an `explore` run in measuring order: the sentinels of the whole case table
 * ({@link gemvSentinels} of `GEMV_CASES`, whatever the run is narrowed to) first, then the others,
 * each part in the order given — the table's for a list taken from it. So every case that is no
 * sentinel is predicted from every sentinel the run measures.
 */
export const gemvExploreOrder = <T extends Pick<GemvCase, "id">>(cases: readonly T[]): T[] => {
  const sentinels = gemvSentinels(GEMV_CASES);
  return [
    ...cases.filter(({ id }) => sentinels.includes(id)),
    ...cases.filter(({ id }) => !sentinels.includes(id)),
  ];
};

/** A row of a sentinel's M = 1 group as the exploration reads it (a record's row has these fields). */
export type GemvExploreRow = {
  readonly candidate: string;
  readonly class: GemvCandidateClass;
  readonly isBaseline?: boolean;
  readonly perDispatch?: number;
  readonly identicalToBaseline?: boolean;
  readonly withinBand?: boolean;
  readonly nonFiniteOutputs?: number;
  readonly screened?: true;
  readonly skipped?: true;
  readonly error?: string;
};

/** A measured sentinel: the case and the rows of its M = 1 group. */
export type GemvExploreEvidence = {
  readonly gemvCase: GemvCase;
  readonly rows: readonly GemvExploreRow[];
};

/**
 * A row with a time whose check held: only such a row can be adopted, so only such a row predicts.
 * A time that is no finite number (absent, `null` from a JSON round trip, NaN) is none.
 */
const held = (row: GemvExploreRow): row is GemvExploreRow & { readonly perDispatch: number } => {
  if (!Number.isFinite(row.perDispatch) || row.screened === true || row.skipped === true) {
    return false;
  }
  if (row.error !== undefined) return false;
  return row.class === "identical"
    ? row.isBaseline === true || row.identicalToBaseline === true
    : row.withinBand === true && (row.nonFiniteOutputs ?? 0) === 0;
};

/** The sequential variants of the grid, by name. */
const SEQUENTIAL_GRID: ReadonlyMap<string, readonly [number, number]> = new Map(
  GEMV_COLS.flatMap((cols, column) =>
    GEMV_UNROLLS.map((unroll, row) => [gemvSequentialName(cols, unroll), [column, row]] as const),
  ),
);

/** The plain and staged parallel rows (no row block), by name. */
const PARALLEL_GRID: ReadonlyMap<string, readonly ["parallel" | "staged", number]> = new Map(
  (["parallel", "staged"] as const).flatMap((form) =>
    GEMV_LANES.map((lanes, index) => [gemvParallelName(form, lanes), [form, index]] as const),
  ),
);

/**
 * The neighbours of a candidate on its grid: a sequential variant's four (one step of the columns
 * per workgroup or of the words read ahead), a parallel row's three (one step of the lanes, and the
 * other of plain and staged at its lanes). Another name has none.
 */
export const gemvGridNeighbours = (name: string): string[] => {
  const sequential = SEQUENTIAL_GRID.get(name);
  if (sequential !== undefined) {
    const [column, row] = sequential;
    return [
      [column - 1, row],
      [column + 1, row],
      [column, row - 1],
      [column, row + 1],
    ].flatMap(([nextColumn, nextRow]) => {
      const cols = nextColumn === undefined ? undefined : GEMV_COLS[nextColumn];
      const unroll = nextRow === undefined ? undefined : GEMV_UNROLLS[nextRow];
      return cols === undefined || unroll === undefined ? [] : [gemvSequentialName(cols, unroll)];
    });
  }
  const parallel = PARALLEL_GRID.get(name);
  if (parallel === undefined) return [];
  const [form, index] = parallel;
  const lanes = GEMV_LANES[index];
  if (lanes === undefined) return [];
  return [
    ...[GEMV_LANES[index - 1], GEMV_LANES[index + 1]].flatMap((other) =>
      other === undefined ? [] : [gemvParallelName(form, other)],
    ),
    gemvParallelName(form === "parallel" ? "staged" : "parallel", lanes),
  ];
};

/** The rows of `rows` that `keep` takes and whose check held, fastest first (equals in their order). */
const heldRows = (
  rows: readonly GemvExploreRow[],
  keep: (row: GemvExploreRow) => boolean,
): (GemvExploreRow & { readonly perDispatch: number })[] =>
  rows
    .filter(
      (row): row is GemvExploreRow & { readonly perDispatch: number } => keep(row) && held(row),
    )
    .toSorted((a, b) => a.perDispatch - b.perDispatch);

/**
 * The parallel rows a sentinel predicts, fastest first (none when it has no plain or staged row whose
 * check held): its fastest plain or staged row, and every other one the derivation could adopt in
 * this group — at least {@link GEMV_MIN_SPEEDUP} faster than the class's reference there, the
 * default's choice at `reorder` at M = 1 (only the fastest when that reference has no time). Rows a
 * few percent apart are equally the sentinel's winners as far as its one group tells, and the
 * case's own winner may lie next to any of them on the grid: on an RTX 3080 Ti (the two `full`
 * records of 2026-11-26/27) staged-l32 led staged-l16 by 1 to 2 % at the nearest i4 sentinel, and
 * three i4 cases whose full records adopt staged-l8 — staged-l16's neighbour, two lane steps from
 * staged-l32 — were measured without it in both explore runs.
 */
const parallelWinners = (sentinel: GemvExploreEvidence): string[] => {
  const rows = heldRows(sentinel.rows, (row) => PARALLEL_GRID.has(row.candidate));
  const [first, ...others] = rows;
  if (first === undefined) return [];
  const referenceName = gemvProfileChoice(sentinel.gemvCase, 1, "reorder").name;
  const reference = sentinel.rows.find((row) => row.candidate === referenceName);
  const time = reference?.error === undefined ? reference?.perDispatch : undefined;
  if (time === undefined || !Number.isFinite(time)) return [first.candidate];
  return [
    first.candidate,
    ...others
      .filter((row) => time / row.perDispatch >= GEMV_MIN_SPEEDUP)
      .map(({ candidate }) => candidate),
  ];
};

/**
 * The winners of the nearest sentinel of the case's storage that has any (the first of equally near
 * ones). A sentinel of another storage does not predict: the storage decides the decode, so its
 * winners say nothing of the case's. The i4 group length may differ.
 */
const predicted = (
  gemvCase: GemvCase,
  sentinels: readonly GemvExploreEvidence[],
  winnersOf: (sentinel: GemvExploreEvidence) => readonly string[],
): readonly string[] => {
  let winners: readonly string[] = [];
  let nearest = Number.POSITIVE_INFINITY;
  for (const sentinel of sentinels) {
    if (sentinel.gemvCase.storage !== gemvCase.storage) continue;
    const distance = gemvCaseDistance(gemvCase, sentinel.gemvCase);
    if (distance >= nearest) continue;
    const found = winnersOf(sentinel);
    if (found.length === 0) continue;
    winners = found;
    nearest = distance;
  }
  return winners;
};

/**
 * The candidates a case that is no sentinel measures at M = 1, by name (the sweep measures the
 * references whatever the list holds): the fastest sequential variant of the nearest sentinel of the
 * case's storage that has one; the fastest plain or staged parallel row of the nearest sentinel of
 * that storage that has one, with every other plain or staged row of that sentinel at least
 * {@link GEMV_MIN_SPEEDUP} faster than the default's choice at `reorder` there
 * ({@link parallelWinners}); and the grid neighbours of each. A subgroup row does not predict (a
 * table holds no such row), and neither does a sentinel of another storage.
 *
 * `undefined` — measure every candidate — when no sentinel given of the case's storage has a winner
 * of one of the two classes: a prediction for half the kernels would leave the other half unmeasured
 * without saying so. A name that is no candidate of the case (a grid over the device's limit) selects nothing.
 */
export const gemvExploreSelection = (
  gemvCase: GemvCase,
  sentinels: readonly GemvExploreEvidence[],
): string[] | undefined => {
  const sequential = predicted(gemvCase, sentinels, ({ rows }) =>
    heldRows(rows, (row) => SEQUENTIAL_GRID.has(row.candidate))
      .slice(0, 1)
      .map(({ candidate }) => candidate),
  );
  const parallel = predicted(gemvCase, sentinels, parallelWinners);
  if (sequential.length === 0 || parallel.length === 0) return undefined;
  return [
    ...new Set(
      [...sequential, ...parallel].flatMap((winner) => [winner, ...gemvGridNeighbours(winner)]),
    ),
  ];
};

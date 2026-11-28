/**
 * The summary of a sweep record of the `grid` set (the grid-fill rule's ladder timed in interleaved
 * rounds — ADR 0157): per case, the default row's time and, for every other ladder tile, its speed
 * against the default as the median over the rounds of the default's round ÷ the tile's round (> 1:
 * the tile is faster), with the spread of those ratios (largest ÷ smallest), and whether every tile
 * wrote the default's words. The rows of a grid case share their reps and their rounds are aligned
 * (round i of every row was timed in the same palindrome), so a round's ratio compares passes taken
 * under the same clocks. Pure.
 *
 * What the derivation reads of such a record (its integers, gaps and margins) is
 * `deriveGemmGrid`'s; this is what a person reads after the run.
 */

import type { GeometrySweepReport, GeometrySweepRow } from "@karume/runtime/tune";

import { joined, ratioText } from "../per-case-sweep.ts";

/** One tile of a grid case against the case's default row. */
export type GridTile = {
  readonly geometry: string;
  /** The median over the rounds of the default's round ÷ this tile's (> 1: faster). */
  readonly speedup?: number;
  /** The largest ÷ the smallest of those per-round ratios (absent with fewer than two). */
  readonly spread?: number;
  readonly identical?: boolean;
  readonly error?: string;
};

export type GridCase = {
  readonly caseId: string;
  readonly censusCount: number;
  readonly defaultGeometry?: string;
  readonly tiles: readonly GridTile[];
};

const median = (values: readonly number[]): number => {
  const sorted = values.toSorted((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/** The per-round ratios default ÷ tile over the rounds both rows timed (a round of 0 is left out). */
const roundRatios = (reference: GeometrySweepRow, row: GeometrySweepRow): number[] => {
  const ours = row.rounds ?? [];
  const theirs = reference.rounds ?? [];
  const ratios: number[] = [];
  for (let round = 0; round < Math.min(ours.length, theirs.length); round += 1) {
    if (ours[round] > 0 && theirs[round] > 0) ratios.push(theirs[round] / ours[round]);
  }
  return ratios;
};

/** The grid summary of a record's rows, case by case in the record's order. */
export const gridSummary = (report: GeometrySweepReport): readonly GridCase[] => {
  const caseIds = [...new Set(report.rows.map((row) => row.caseId))];
  return caseIds.map((caseId) => {
    const rows = report.rows.filter((row) => row.caseId === caseId);
    const reference = rows.find((row) => row.isDefault);
    const tiles = rows
      .filter((row) => !row.isDefault)
      .map((row): GridTile => {
        if (row.error !== undefined) return { geometry: row.geometry, error: row.error };
        const ratios = reference === undefined ? [] : roundRatios(reference, row);
        return {
          geometry: row.geometry,
          ...(ratios.length === 0 ? {} : { speedup: median(ratios) }),
          ...(ratios.length < 2 ? {} : { spread: Math.max(...ratios) / Math.min(...ratios) }),
          ...(row.identicalToDefault === undefined ? {} : { identical: row.identicalToDefault }),
        };
      });
    return {
      caseId,
      censusCount: rows[0]?.censusCount ?? 0,
      ...(reference === undefined ? {} : { defaultGeometry: reference.geometry }),
      tiles,
    };
  });
};

/** The grid summary as lines: a head, then one line per case. */
export const gridLines = (report: GeometrySweepReport): string[] => [
  "Grid set: each ladder tile against the case's default tile, the median over the interleaved rounds of the default's round ÷ the tile's (above ×1 the tile is faster) and the spread of those ratios",
  ...gridSummary(report).map(
    (entry) =>
      `${entry.caseId} (×${entry.censusCount}): ${joined([
        entry.defaultGeometry === undefined ? "no default row" : `default ${entry.defaultGeometry}`,
        ...entry.tiles.map((tile) =>
          tile.error !== undefined
            ? `${tile.geometry} failed`
            : `${tile.geometry} ${ratioText(tile.speedup)}${tile.spread === undefined ? "" : ` (spread ×${tile.spread.toFixed(3)})`}${tile.identical === false ? " NOT IDENTICAL" : ""}`,
        ),
      ])}`,
  ),
];

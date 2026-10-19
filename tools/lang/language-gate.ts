/**
 * The language gate (ADR 0130, step 2-9h): every tracked text file that carries Japanese must be
 * covered by `ALLOW_LIST`, within its `maxLines` where one is set, and every allowance must still
 * cover at least one such file (a stale allowance is a defect too — the list ratchets down).
 *
 * `packages/runtime/tests/verify_language_test.ts` runs this in the core lane. From the shell:
 *   node tools/lang/language-gate.ts [--root <dir>]
 *       prints the offenders and exits 1 when the gate is red
 */

import { ALLOW_LIST, type Allowance, allowanceOf } from "./language-allow-list.ts";
import { type ScanEntry, scanFiles, trackedFiles } from "./scan-japanese.ts";

export type GateReport = {
  /** Files with Japanese that no allowance covers. */
  readonly uncovered: readonly ScanEntry[];
  /** Files whose Japanese lines exceed the allowance's `maxLines`. */
  readonly overBudget: readonly (ScanEntry & { readonly maxLines: number })[];
  /** Allowances that cover no file with Japanese any more. */
  readonly stale: readonly Allowance[];
};

export const gateReport = (
  entries: readonly ScanEntry[],
  list: readonly Allowance[] = ALLOW_LIST,
): GateReport => {
  const uncovered: ScanEntry[] = [];
  const overBudget: (ScanEntry & { maxLines: number })[] = [];
  const used = new Set<Allowance>();
  for (const entry of entries) {
    const allowance = allowanceOf(entry.file, list);
    if (allowance === undefined) {
      uncovered.push(entry);
      continue;
    }
    used.add(allowance);
    if (allowance.maxLines !== undefined && entry.lines > allowance.maxLines) {
      overBudget.push({ ...entry, maxLines: allowance.maxLines });
    }
  }
  const stale = list.filter((allowance) => !used.has(allowance));
  return { uncovered, overBudget, stale };
};

export const isGreen = (report: GateReport): boolean =>
  report.uncovered.length === 0 && report.overBudget.length === 0 && report.stale.length === 0;

/** The report as lines for a test failure or the shell. */
export const describeReport = (report: GateReport): string[] => [
  ...report.uncovered.map(
    (entry) =>
      `${entry.file}: ${entry.lines} lines with Japanese and no allowance (translate, or add an entry with its reason)`,
  ),
  ...report.overBudget.map(
    (entry) => `${entry.file}: ${entry.lines} lines with Japanese, allowed ${entry.maxLines}`,
  ),
  ...report.stale.map(
    (allowance) =>
      `${allowance.path}: the allowance covers no file with Japanese any more (remove it)`,
  ),
];

/** Scans the tracked files under `root` and applies the allow-list. */
export const runGate = (root: string): GateReport =>
  gateReport(scanFiles(root, trackedFiles(root)));

const main = (argv: readonly string[]): void => {
  const at = argv.indexOf("--root");
  const root = at >= 0 ? (argv[at + 1] ?? process.cwd()) : process.cwd();
  const report = runGate(root);
  for (const line of describeReport(report)) console.log(line);
  console.log(
    `language gate: ${report.uncovered.length} uncovered, ${report.overBudget.length} over budget, ${report.stale.length} stale allowances`,
  );
  if (!isGreen(report)) process.exitCode = 1;
};

if (import.meta.main) {
  main(process.argv.slice(2));
}

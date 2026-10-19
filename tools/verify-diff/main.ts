/**
 * verify-diff — a read-only tool that lines up the verification results taken on several machines
 * and prints where they diverge, in one command.
 *
 *     node tools/verify-diff/main.ts
 *     node tools/verify-diff/main.ts --family anima --family golden
 *     node tools/verify-diff/main.ts --date 2026-09-21
 *     node tools/verify-diff/main.ts --root ~/verify-from-another-box --json
 *
 * It reads only `outputs/verify/<environment key>/<date>_<family>/results.json` (the results directories
 * of ADR 0106 decision 5) and writes nothing. A second machine's results are assumed to be copied
 * **by hand** into `outputs/verify/<environment key>/` (the results directories are split so that
 * untracked results are not brought under tracking, so there is no sync mechanism; results kept
 * elsewhere are pointed at with `--root`).
 *
 * Per environment it takes one results directory, the one with the **newest date** (with `--date`, only
 * that date). The date of a results directory is **UTC** (the date part of the run's start time —
 * packages/runtime/tests/helpers/results.ts), so a run started before 09:00 JST lands in the
 * previous day's entry. `--date` also takes a UTC date.
 *
 * The exit code is **0 even with differences** — this is not a check but a tool that points at
 * where the results diverge. Cross-device identity of sha256 is not guaranteed in the first place
 * (docs/limitations.md), so an artifact sha mismatch is also only listed as a difference. It exits
 * with 1 only when a document cannot be read or has a different schema, and never prints a silent
 * "no differences".
 */

import type { Dirent } from "node:fs";
import { readdir, readFile } from "node:fs/promises";

import {
  buildDiff,
  type Cell,
  type Comparison,
  type Difference,
  type DiffReport,
  type FamilyMatrix,
  type LoadedResults,
  type Measurement,
  parseResultsDocument,
} from "./diff.ts";

/** The default root (`outputs/verify/`) — resolved from import.meta.url, as in results.ts. */
const DEFAULT_ROOT = new URL("../../outputs/verify/", import.meta.url);

/** Options that take a value. */
const OPTIONS: ReadonlySet<string> = new Set(["root", "family", "date"]);
/**
 * Options that take no value (one form added to the `--key value` pairs of opbench, a measurement
 * tool removed in bb35fe2a).
 */
const FLAGS: ReadonlySet<string> = new Set(["json"]);

const USAGE = `Usage: node tools/verify-diff/main.ts [options]
  --root <dir>         root of the results (default = outputs/verify/ of this repository)
  --family <name>      only this family (repeatable; default = every family under the root)
  --date YYYY-MM-DD    only the results directories of this date (UTC) (default = the newest date per environment)
  --json               print the matrix and the warnings as JSON (default = Markdown)`;

export type Args = {
  readonly values: ReadonlyMap<string, readonly string[]>;
  readonly flags: ReadonlySet<string>;
};

/**
 * Reads a sequence of `--key value` (the convention of opbench, since removed — a broken pair, a
 * missing value or an unknown key fails).
 *
 * Only `--json`, which takes no value, is consumed as one word. A scan that only counts pairs would
 * shift every later pair by one word after a single flag and read it with another meaning, so flags
 * are sorted out first against the known set.
 */
export const parseArgs = (argv: readonly string[]): Args => {
  const values = new Map<string, string[]>();
  const flags = new Set<string>();
  for (let at = 0; at < argv.length;) {
    const key = argv[at];
    if (!key.startsWith("--")) throw new Error(`argument ${key} is not a '--key value' pair`);
    const name = key.slice(2);
    if (FLAGS.has(name)) {
      flags.add(name);
      at += 1;
      continue;
    }
    // Do not drop an unknown option silently (a mistyped `--families` would quietly fall back to
    // every family).
    if (!OPTIONS.has(name)) throw new Error(`unknown option --${name}.\n${USAGE}`);
    const value = argv[at + 1];
    if (value === undefined) throw new Error(`argument ${key} is not a '--key value' pair`);
    if (value.startsWith("--")) {
      throw new Error(`argument ${key} has no value ('${value}' is an option)`);
    }
    values.set(name, [...(values.get(name) ?? []), value]);
    at += 2;
  }
  return { values, flags };
};

/** Reads a key that can be given only once. */
export const single = (args: Args, name: string): string | undefined => {
  const values = args.values.get(name);
  if (values === undefined) return undefined;
  if (values.length > 1) throw new Error(`--${name} can be given only once`);
  return values[0];
};

/**
 * The directory name of a results directory (`<date>_<family>`). The document is the source of truth for
 * the family name, so only the date is taken here.
 */
const SEAT = /^(\d{4}-\d{2}-\d{2})_.+$/;

const DATE = /^\d{4}-\d{2}-\d{2}$/;

/**
 * Turns an OS path into a directory URL (a local copy: the shared helper it was copied from,
 * `tools/_shared/assets.ts`, pulled in the IR parser with it and was removed in bb35fe2a).
 *
 * MUST: `encodeURIComponent` each segment before joining (the same convention as {@link childUrl}).
 * Passing the path to `new URL` as is parses `#` / `?` / `%` as URL syntax, so given the real
 * directory `x%41y` it silently reads `xAy` (showing results from another place as the root that was
 * typed).
 */
const directoryUrl = (path: string): URL => {
  const absolute = path.startsWith("/") ? path : `${process.cwd()}/${path}`;
  const encoded = absolute.split("/").map(encodeURIComponent).join("/");
  return new URL(`file://${encoded.endsWith("/") ? encoded : `${encoded}/`}`);
};

/** The URL of a child directory (so that a `#` / `?` in the name does not point elsewhere). */
const childUrl = (parent: URL, name: string): URL =>
  new URL(`${encodeURIComponent(name)}/`, parent);

/** The failure node:fs throws when a path does not exist (`ENOENT`). */
const isNotFound = (cause: unknown): boolean =>
  cause instanceof Error && "code" in cause && cause.code === "ENOENT";

/**
 * Walks the root and reads every `results.json`.
 *
 * Throws when a directory named like a results directory (`<date>_<family>`) has no `results.json` (a
 * results directory holds an "in progress" `results.json` from the moment it is created, so a missing one
 * means a broken results directory). Directories and files not spelled like a results directory are not result
 * entries and are skipped silently.
 */
const loadRoot = async (root: URL, label: string): Promise<LoadedResults[]> => {
  const loaded: LoadedResults[] = [];
  let environments: Dirent[];
  try {
    environments = await readdir(root, { withFileTypes: true });
  } catch (cause) {
    if (isNotFound(cause)) {
      throw new Error(`the results root does not exist: ${label} (point --root at the results)`);
    }
    throw cause;
  }
  for (const environment of environments) {
    if (!environment.isDirectory()) continue;
    const environmentDir = childUrl(root, environment.name);
    for (const seat of await readdir(environmentDir, { withFileTypes: true })) {
      const matched = SEAT.exec(seat.name);
      if (!seat.isDirectory() || matched === null) continue;
      const path = `${label}/${environment.name}/${seat.name}/results.json`;
      const file = new URL("results.json", childUrl(environmentDir, seat.name));
      let text: string;
      try {
        text = await readFile(file, "utf8");
      } catch (cause) {
        if (isNotFound(cause)) {
          throw new Error(`${path}: the results directory has no results.json (broken)`);
        }
        throw cause;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch (cause) {
        const reason = cause instanceof Error ? cause.message : String(cause);
        throw new Error(`${path}: cannot be read as JSON (${reason})`);
      }
      loaded.push({ path, date: matched[1], document: parseResultsDocument(parsed, path) });
    }
  }
  return loaded;
};

/** A string for a table cell (`|` breaks the table, so it is escaped). */
const md = (text: string): string => text.replaceAll("|", "\\|");

/**
 * Bands and observed values are orders of magnitude apart, so switch between exponential and fixed
 * notation by magnitude.
 */
const num = (value: number | null): string => {
  if (value === null) return "null";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude < 1e-3 || magnitude >= 1e4) return value.toExponential(2);
  return String(Number(value.toPrecision(4)));
};

const shortSha = (sha: string): string => sha.slice(0, 7);

const describeDifference = (difference: Difference): string => {
  const values = Object.entries(difference.values)
    .map(
      ([environment, value]) =>
        `${environment}=${difference.kind === "actual" ? shortSha(value) : value}`,
    )
    .join(" / ");
  if (difference.kind === "status") {
    return `\`${difference.caseId}\`: status differs — ${values}`;
  }
  if (difference.kind === "actual") {
    // Cross-device bit identity is not guaranteed (ADR 0106), so a divergence is not made red.
    return `\`${difference.caseId}\`: artifact sha differs (not guaranteed across devices; listed as a difference only) — ${values}`;
  }
  return `\`${difference.caseId}\`: no result on ${(difference.missing ?? []).join(
    " / ",
  )} — ${values}`;
};

/**
 * Observed tolerance checks (the ratio to the band can be derived only when rtol=0 — with rtol the
 * band moves per element).
 */
const measurementRows = (
  environments: readonly string[],
  cells: Readonly<Record<string, Cell>>,
): string[] => {
  const rows: string[] = [];
  const outputs = new Set<string>();
  for (const environment of environments) {
    const cell = Object.hasOwn(cells, environment) ? cells[environment] : undefined;
    for (const measurement of cell?.measurements ?? []) outputs.add(measurement.output);
  }
  for (const output of [...outputs].sort()) {
    for (const environment of environments) {
      const cell = Object.hasOwn(cells, environment) ? cells[environment] : undefined;
      const found: Measurement | undefined = cell?.measurements?.find(
        (one) => one.output === output,
      );
      if (found === undefined) continue;
      const { tolerance, maxAbs } = found;
      const margin =
        tolerance.rtol === 0 && maxAbs !== null && tolerance.atol !== 0
          ? num(maxAbs / tolerance.atol)
          : "-";
      rows.push(
        `| ${md(output)} | ${md(environment)} | ${num(maxAbs)} | ${num(found.maxRel)} | ` +
          `atol=${num(tolerance.atol)} rtol=${num(tolerance.rtol)} | ${margin} | ${found.stage} |`,
      );
    }
  }
  return rows;
};

/**
 * Observations of the self A/B check (ADR 0110 decision 5). The band is declared, so it is only
 * listed and no ratio to it is printed (outside the band the check has already ended red — this
 * table shows how much relRMS moves between environments).
 */
const comparisonRows = (
  environments: readonly string[],
  cells: Readonly<Record<string, Cell>>,
): string[] => {
  const rows: string[] = [];
  const outputs = new Set<string>();
  for (const environment of environments) {
    const cell = Object.hasOwn(cells, environment) ? cells[environment] : undefined;
    for (const comparison of cell?.comparisons ?? []) outputs.add(comparison.output);
  }
  for (const output of [...outputs].sort()) {
    for (const environment of environments) {
      const cell = Object.hasOwn(cells, environment) ? cells[environment] : undefined;
      const found: Comparison | undefined = cell?.comparisons?.find((one) => one.output === output);
      if (found === undefined) continue;
      const band =
        found.band === undefined
          ? "not derived"
          : `${num(found.band.floor)}〜${num(found.band.ceiling)}`;
      rows.push(
        `| ${md(output)} | ${md(environment)} | ${md(found.reference)} → ${md(
          found.practical,
        )} | ` + `${num(found.relRms)} | ${num(found.maxAbs)} | ${band} |`,
      );
    }
  }
  return rows;
};

const renderFamily = (matrix: FamilyMatrix): string[] => {
  const lines: string[] = [`## ${matrix.family}`, ""];
  for (const one of matrix.selected) {
    const checkout =
      one.checkout === null
        ? "no checkout"
        : `${shortSha(one.checkout.sha)}${one.checkout.dirty ? ", dirty" : ""}`;
    lines.push(`- ${one.environment}: ${one.date}（${checkout}）`);
  }
  if (matrix.absent.length > 0) {
    lines.push(`- Environments without results for this family: ${matrix.absent.join(" / ")}`);
  }
  lines.push("");
  lines.push(`| Case | ${matrix.environments.map(md).join(" | ")} |`);
  lines.push(`| --- | ${matrix.environments.map(() => "---").join(" | ")} |`);
  for (const row of matrix.rows) {
    const cells = matrix.environments.map((environment) =>
      Object.hasOwn(row.cells, environment) ? row.cells[environment].status : "—",
    );
    lines.push(`| ${md(row.id)} | ${cells.join(" | ")} |`);
  }
  lines.push("");
  if (matrix.warnings.length > 0) {
    lines.push("Warnings:", "");
    for (const warning of matrix.warnings) lines.push(`- ${warning}`);
    lines.push("");
  }
  if (matrix.differences.length === 0) {
    lines.push("No differences", "");
  } else {
    lines.push("Differences:", "");
    for (const difference of matrix.differences) lines.push(`- ${describeDifference(difference)}`);
    lines.push("");
  }
  for (const row of matrix.rows) {
    const rows = measurementRows(matrix.environments, row.cells);
    if (rows.length === 0) continue;
    lines.push(`### Measurements of ${row.id}`, "");
    lines.push("| Output | Environment | maxAbs | maxRel | Band | maxAbs/atol | Stage |");
    lines.push("| --- | --- | --- | --- | --- | --- | --- |");
    lines.push(...rows, "");
  }
  for (const row of matrix.rows) {
    const rows = comparisonRows(matrix.environments, row.cells);
    if (rows.length === 0) continue;
    lines.push(`### A/B of ${row.id}`, "");
    lines.push(
      "| Observation point | Environment | Reference → practical | relRMS | maxAbs | Band (relRMS) |",
    );
    lines.push("| --- | --- | --- | --- | --- | --- |");
    lines.push(...rows, "");
  }
  return lines;
};

const renderMarkdown = (report: DiffReport, label: string): string => {
  const lines: string[] = ["# verify-diff", "", `Root: ${label}`, ""];
  for (const warning of report.warnings) lines.push(`- Warning: ${warning}`);
  if (report.warnings.length > 0) lines.push("");
  if (report.families.length === 0) {
    lines.push(
      "No results to compare (the root has no results directories, or the filters match none)",
      "",
    );
  }
  for (const matrix of report.families) lines.push(...renderFamily(matrix));
  return lines.join("\n");
};

const run = async (argv: readonly string[]): Promise<void> => {
  const args = parseArgs(argv);
  const rootArg = single(args, "root");
  const root = rootArg === undefined ? DEFAULT_ROOT : directoryUrl(rootArg);
  // The root printed in messages keeps the spelling the person typed (only the default is turned
  // back from a URL into a plain path).
  const label = (rootArg ?? decodeURIComponent(DEFAULT_ROOT.pathname)).replace(/\/+$/, "");
  const date = single(args, "date");
  if (date !== undefined && !DATE.test(date)) {
    throw new Error(`--date must be YYYY-MM-DD ('${date}')`);
  }
  const families = args.values.get("family");
  const report = buildDiff(await loadRoot(root, label), {
    ...(families === undefined ? {} : { families }),
    ...(date === undefined ? {} : { date }),
  });
  console.log(
    args.flags.has("json") ? JSON.stringify(report, undefined, 2) : renderMarkdown(report, label),
  );
};

if (import.meta.main) {
  try {
    await run(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    // process.exit() is not used: it cuts off stderr queued on an asynchronous pipe (macOS / Windows).
    process.exitCode = 1;
  }
}

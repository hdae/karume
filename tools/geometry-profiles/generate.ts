/**
 * The generator of the runtime's embedded geometry profile tables
 * (`packages/runtime/src/kernels/geometry-profiles/<id>.ts`). From the repository root:
 *
 *   node tools/geometry-profiles/generate.ts <profile id> [--records=<path>[,<path>…]]
 *     [--attention-bench=<path>[,…]] [--gemv-records=<path>[,<path>…]]
 *     [--vendor=<v> [--architecture=<a> [--description=<d>]]] [--min-speedup=<x>] [--dry-run]
 *   node tools/geometry-profiles/generate.ts <profile id> --gemv-plan [--gemv-records=<path>[,…]]
 *     [--dry-run]
 *
 * It always works as "the registered table as base + new records for some op families": a
 * registered id takes that table as `base` of `deriveGeometryProfile`, so the families the records
 * do not cover are carried over unchanged, and its adapter match is the table's own. An id that is
 * not registered is a new adapter's first table: the match options are required and there is no
 * base. It never re-derives a whole table from its historical records and never lifts an old record
 * format. GEMV sweep records (`--gemv-records`) derive the table's GEMV choices of linear
 * (`linearGemv`, with `deriveLinearGemvChoices`) onto the table; without them a base's field is
 * carried. A registered table can take GEMV sweep records alone; a first table needs `--records`.
 *
 * `--gemv-plan` derives nothing: it writes the workbench batch `inputs/workbench/batches/
 * gemv-plan-<profile id>.json` that measures, twice, what the GEMV sweep records at hand lack for the
 * GEMV derivation to be complete under the current runtime (plan.ts) — the whole `full` set when
 * no record is given.
 *
 * Exit codes: 0 on success, 2 for a command line it cannot act on, 1 for anything else (the
 * derivation's error message on stderr, unchanged).
 *
 * SPEC: docs/spec/geometry-profiles.md#where-a-table-comes-from
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { BUILTIN_GEOMETRY_PROFILES, type GeometryProfile } from "@karume/runtime";
import {
  deriveGeometryProfile,
  deriveLinearGemvChoices,
  type GeometryProfileDerivation,
  type GeometryProfileTarget,
  geometryProfileJson,
  type LinearGemvDerivation,
} from "@karume/runtime/tune";
import { differenceLine, profileDifferences } from "./differences.ts";
import { formatTypeScript, REPOSITORY_ROOT } from "./format.ts";
import { profileHeader } from "./header.ts";
import { profileModule } from "./literal.ts";
import { gemvPlan, planBatchText, UNTRACKED_BATCHES_DIRECTORY } from "./plan.ts";
import {
  type LoadedRecord,
  loadRecord,
  RecordError,
  type RecordKind,
  repositoryRelativePath,
  UsageError,
} from "./records.ts";
import { INDEX_PATH, PROFILE_TEST_PATH, registrationEdits } from "./registration.ts";

/** Where the tables live, repository-relative. */
export const PROFILES_DIRECTORY = "packages/runtime/src/kernels/geometry-profiles";

/**
 * The runtime's default adoption threshold (`DEFAULT_MIN_SPEEDUP` of the derivation, which the
 * public surface does not export). The command always passes a threshold explicitly, so the header
 * states the value the derivation actually used.
 */
export const DEFAULT_MIN_SPEEDUP = 1.05;

const USAGE =
  "usage: node tools/geometry-profiles/generate.ts <profile id> [--records=<path>[,<path>…]] " +
  "[--attention-bench=<path>[,…]] [--gemv-records=<path>[,<path>…]] " +
  "[--vendor=<v> [--architecture=<a> [--description=<d>]]] [--min-speedup=<x>] [--dry-run]\n" +
  "       node tools/geometry-profiles/generate.ts <profile id> --gemv-plan " +
  "[--gemv-records=<path>[,<path>…]] [--dry-run]";

/** A table id is also its file name: kebab-case, as every registered id is. */
const PROFILE_ID = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export type AdapterMatch = {
  readonly vendor: string;
  readonly architecture?: string;
  readonly description?: string;
};

export type Request = {
  readonly id: string;
  readonly records: readonly string[];
  readonly attentionBench: readonly string[];
  readonly gemvRecords: readonly string[];
  readonly match?: AdapterMatch;
  readonly minSpeedup: number;
  readonly dryRun: boolean;
  /** Write the batch of what the GEMV records lack instead of deriving (`--gemv-plan`). */
  readonly gemvPlan: boolean;
};

const VALUE_OPTIONS = [
  "records",
  "attention-bench",
  "gemv-records",
  "vendor",
  "architecture",
  "description",
  "min-speedup",
] as const;
type ValueOption = (typeof VALUE_OPTIONS)[number];

const isValueOption = (name: string): name is ValueOption =>
  VALUE_OPTIONS.some((option) => option === name);

const pathList = (value: string | undefined): string[] =>
  value === undefined ? [] : value.split(",").filter((path) => path !== "");

/** The command line as a request, or a {@link UsageError} naming what it cannot act on. */
export const parseCommandLine = (argv: readonly string[]): Request => {
  const values = new Map<ValueOption, string>();
  const positional: string[] = [];
  let dryRun = false;
  let gemvPlan = false;
  for (const argument of argv) {
    if (argument === "--dry-run") {
      dryRun = true;
      continue;
    }
    if (argument === "--gemv-plan") {
      gemvPlan = true;
      continue;
    }
    if (!argument.startsWith("--")) {
      positional.push(argument);
      continue;
    }
    const equals = argument.indexOf("=");
    const name = argument.slice(2, equals < 0 ? undefined : equals);
    if (!isValueOption(name) || equals < 0) {
      throw new UsageError(`Unknown option or option without a value: ${argument}`);
    }
    if (values.has(name)) throw new UsageError(`--${name} is given twice`);
    values.set(name, argument.slice(equals + 1));
  }
  if (positional.length !== 1) {
    throw new UsageError(`Expected one profile id, got ${positional.length}`);
  }
  const [id] = positional;
  if (!PROFILE_ID.test(id)) {
    throw new UsageError(`The profile id ${JSON.stringify(id)} is not kebab-case`);
  }
  const records = pathList(values.get("records"));
  const gemvRecords = pathList(values.get("gemv-records"));
  if (gemvPlan) {
    const derivationOnly = VALUE_OPTIONS.filter(
      (option) => option !== "gemv-records" && values.has(option),
    );
    if (derivationOnly.length > 0) {
      throw new UsageError(
        `--gemv-plan reads GEMV sweep records only (drop ${derivationOnly.map((option) => `--${option}`).join(", ")})`,
      );
    }
  } else if (records.length === 0 && gemvRecords.length === 0) {
    throw new UsageError("Neither --records nor --gemv-records names a record");
  }
  const vendor = values.get("vendor");
  const architecture = values.get("architecture");
  const description = values.get("description");
  if (vendor === undefined && (architecture !== undefined || description !== undefined)) {
    throw new UsageError("--architecture and --description need --vendor");
  }
  const speedupText = values.get("min-speedup");
  const minSpeedup = speedupText === undefined ? DEFAULT_MIN_SPEEDUP : Number(speedupText);
  if (speedupText === "" || !Number.isFinite(minSpeedup)) {
    throw new UsageError(`--min-speedup is not a number (${JSON.stringify(speedupText)})`);
  }
  return {
    id,
    records,
    attentionBench: pathList(values.get("attention-bench")),
    gemvRecords,
    ...(vendor === undefined
      ? {}
      : {
          match: {
            vendor,
            ...(architecture === undefined ? {} : { architecture }),
            ...(description === undefined ? {} : { description }),
          },
        }),
    minSpeedup,
    dryRun,
    gemvPlan,
  };
};

/** The derivation's adapter target from a table's (or the command line's) match. */
const targetOf = (match: GeometryProfile["match"]): GeometryProfileTarget => {
  if (match === undefined) return { optIn: true };
  if (match.vendor === undefined) {
    throw new Error("A match without a vendor is the default table's, which has no records");
  }
  return {
    vendor: match.vendor,
    ...(match.architecture === undefined ? {} : { architecture: match.architecture }),
    ...(match.description === undefined ? {} : { description: match.description }),
  };
};

/**
 * The generator's derivation: `deriveGeometryProfile` over the loaded records, with the registered
 * table as `base` (and its match) when there is one, else the command line's match. Its errors pass
 * through unchanged.
 */
export const deriveTable = (input: {
  readonly id: string;
  readonly match?: AdapterMatch;
  readonly minSpeedup: number;
  readonly records: readonly LoadedRecord[];
  readonly attentionBench: readonly LoadedRecord[];
  readonly base?: GeometryProfile;
}): GeometryProfileDerivation =>
  deriveGeometryProfile(
    input.records.map(({ report, path, sha256 }) => ({ report, path, sha256 })),
    {
      ...targetOf(input.base === undefined ? input.match : input.base.match),
      id: input.id,
      minSpeedup: input.minSpeedup,
      ...(input.attentionBench.length === 0
        ? {}
        : {
            attentionBench: input.attentionBench.map(({ report, path, sha256 }) => ({
              report,
              path,
              sha256,
            })),
          }),
      ...(input.base === undefined ? {} : { base: input.base }),
    },
  );

/**
 * `table` with `linearGemv` set to the given field, or without it when the field is absent. The
 * field goes where the profile's type places it, before the provenance, so the printer writes the
 * fields in one order.
 */
export const withLinearGemv = (
  table: GeometryProfile,
  linearGemv: GeometryProfile["linearGemv"],
): GeometryProfile => {
  const { linearGemv: _replaced, provenance, ...fields } = table;
  return {
    ...fields,
    ...(linearGemv === undefined ? {} : { linearGemv }),
    ...(provenance === undefined ? {} : { provenance }),
  };
};

/**
 * The generator's GEMV step: `deriveLinearGemvChoices` over the loaded GEMV sweep records for
 * `table` (the tile geometry's derivation, or the registered table when only GEMV records are
 * given), and the table with the derived field in place. Its errors pass through unchanged.
 */
export const deriveGemvTable = (
  table: GeometryProfile,
  records: readonly LoadedRecord[],
): { readonly profile: GeometryProfile; readonly derivation: LinearGemvDerivation } => {
  const derivation = deriveLinearGemvChoices(
    records.map(({ report, path, sha256 }) => ({ report, path, sha256 })),
    { table },
  );
  return { profile: withLinearGemv(table, derivation.linearGemv), derivation };
};

export type CommandEnvironment = {
  /** Where record paths resolve and the table and registration files are. */
  readonly repositoryRoot: string;
  /** The registered tables (a registered id is regenerated on its table as base). */
  readonly registered: readonly GeometryProfile[];
  /** The last commit that touched a repository-relative path. */
  readonly lastCommit: (path: string) => string;
  readonly stdout: (text: string) => void;
  readonly stderr: (text: string) => void;
};

const gitLastCommit =
  (repositoryRoot: string) =>
  (path: string): string => {
    const commit = execFileSync("git", ["log", "-1", "--format=%H", "--", path], {
      cwd: repositoryRoot,
      encoding: "utf8",
    }).trim();
    if (commit === "") throw new Error(`${path}: no commit touched the file`);
    return commit;
  };

export const defaultEnvironment = (): CommandEnvironment => ({
  repositoryRoot: REPOSITORY_ROOT,
  registered: BUILTIN_GEOMETRY_PROFILES,
  lastCommit: gitLastCommit(REPOSITORY_ROOT),
  stdout: (text) => process.stdout.write(`${text}\n`),
  stderr: (text) => process.stderr.write(`${text}\n`),
});

const readRecords = async (
  repositoryRoot: string,
  given: readonly string[],
  kind: RecordKind,
): Promise<LoadedRecord[]> => {
  const paths = given.map((path) => repositoryRelativePath(repositoryRoot, path));
  const records: LoadedRecord[] = [];
  for (const path of paths) {
    let bytes: Uint8Array;
    try {
      bytes = await readFile(join(repositoryRoot, path));
    } catch (cause) {
      const code =
        typeof cause === "object" && cause !== null && "code" in cause ? String(cause.code) : "";
      throw new RecordError(`${path}: cannot read the file (${code})`, { cause });
    }
    records.push(loadRecord(bytes, path, kind));
  }
  return records;
};

/**
 * `--gemv-plan`: the plan of what the GEMV records lack (plan.ts), printed, and its batch written
 * under the workbench's untracked batches (with `--dry-run`, printed instead). Nothing is written
 * when nothing is missing.
 */
const writePlan = async (
  request: Request,
  base: GeometryProfile | undefined,
  environment: CommandEnvironment,
): Promise<void> => {
  const { repositoryRoot, stdout } = environment;
  const records = await readRecords(repositoryRoot, request.gemvRecords, "gemv-records");
  const plan = gemvPlan({
    profileId: request.id,
    records,
    ...(base === undefined ? {} : { table: base }),
  });
  for (const line of plan.lines) stdout(line);
  if (plan.batch === undefined) {
    stdout("No batch written: there is nothing to measure.");
    return;
  }
  const path = `${UNTRACKED_BATCHES_DIRECTORY}/${plan.batchId}.json`;
  const text = planBatchText(plan.batch);
  if (request.dryRun) {
    stdout(`Dry run: nothing written. The batch ${path}:`);
    stdout(text.trimEnd());
    return;
  }
  await mkdir(join(repositoryRoot, UNTRACKED_BATCHES_DIRECTORY), { recursive: true });
  await writeFile(join(repositoryRoot, path), text);
  stdout(
    `Wrote ${path} (${plan.batch.steps.length} steps): the workbench lists it as the untracked batch ${plan.batchId}; under Node, node apps/workbench/run.ts --batch=${path}`,
  );
};

const run = async (argv: readonly string[], environment: CommandEnvironment): Promise<void> => {
  const { repositoryRoot, stdout } = environment;
  const request = parseCommandLine(argv);
  const base = environment.registered.find((profile) => profile.id === request.id);
  if (request.gemvPlan) {
    await writePlan(request, base, environment);
    return;
  }
  if (base !== undefined && request.match !== undefined) {
    throw new UsageError(
      `'${request.id}' is registered: its adapter match is the table's own (drop --vendor, --architecture and --description)`,
    );
  }
  if (base === undefined && request.match === undefined) {
    throw new UsageError(
      `'${request.id}' is not registered: a first table needs its adapter match (--vendor, with --architecture and --description as needed)`,
    );
  }
  if (request.records.length === 0 && (base === undefined || request.attentionBench.length > 0)) {
    throw new UsageError(
      base === undefined
        ? `'${request.id}' is not registered: a first table needs --records (the tile geometry)`
        : "--attention-bench needs --records (the derivation of the tile geometry reads them together)",
    );
  }
  const records = await readRecords(repositoryRoot, request.records, "records");
  const attentionBench = await readRecords(
    repositoryRoot,
    request.attentionBench,
    "attention-bench",
  );
  const gemvRecords = await readRecords(repositoryRoot, request.gemvRecords, "gemv-records");
  const geometry =
    records.length === 0
      ? undefined
      : deriveTable({
          id: request.id,
          ...(request.match === undefined ? {} : { match: request.match }),
          minSpeedup: request.minSpeedup,
          records,
          attentionBench,
          ...(base === undefined ? {} : { base }),
        });
  const tiled = geometry?.profile ?? base;
  if (tiled === undefined) throw new Error("No table to derive the GEMV choices for");
  const gemv = gemvRecords.length === 0 ? undefined : deriveGemvTable(tiled, gemvRecords);
  const profile = gemv?.profile ?? tiled;
  const verdicts = [...(geometry?.verdicts ?? []), ...(gemv?.derivation.verdicts ?? [])];
  const tablePath = `${PROFILES_DIRECTORY}/${request.id}.ts`;
  if (request.dryRun) {
    stdout(`Dry run: nothing written. The derived profile of '${request.id}':`);
    stdout(geometryProfileJson(profile));
    stdout("Verdicts:");
    for (const line of verdicts) stdout(line);
    if (gemv !== undefined) {
      stdout(
        gemv.derivation.linearGemv === undefined
          ? "linearGemv: absent from the table (every GEMV choice stays the default's)"
          : "linearGemv: written to the table (the default's choices with the adopted ones in place)",
      );
    }
    if (base !== undefined) {
      // A side without linearGemv runs with the default's choices: compare what each table runs with
      const runsWith = (table: GeometryProfile): GeometryProfile =>
        gemv === undefined
          ? table
          : withLinearGemv(table, table.linearGemv ?? gemv.derivation.defaults);
      const { fields, provenance } = profileDifferences(runsWith(base), runsWith(profile));
      stdout(
        `Fields that differ from the registered table (${tablePath}${
          gemv === undefined ? "" : "; linearGemv as each table runs with it"
        }):`,
      );
      stdout(
        fields.length === 0 ? "  none" : fields.map((d) => `  ${differenceLine(d)}`).join("\n"),
      );
      stdout("Provenance fields that differ:");
      stdout(
        provenance.length === 0
          ? "  none"
          : provenance.map((d) => `  ${differenceLine(d)}`).join("\n"),
      );
    }
    return;
  }
  const header = profileHeader({
    profile,
    verdicts,
    records,
    attentionBench,
    ...(gemv === undefined
      ? {}
      : {
          gemv: {
            records: gemvRecords,
            minSpeedup: gemv.derivation.rule.minSpeedup,
            referenceRepeat: gemv.derivation.rule.referenceRepeat,
          },
        }),
    minSpeedup: request.minSpeedup,
    ...(base === undefined
      ? {}
      : { base: { commit: environment.lastCommit(tablePath), path: tablePath } }),
  });
  // Worked out before anything is written: a registration place that cannot be found stops the run
  // with the repository unchanged.
  const registration =
    base === undefined
      ? registrationEdits(
          request.id,
          await readFile(join(repositoryRoot, INDEX_PATH), "utf8"),
          await readFile(join(repositoryRoot, PROFILE_TEST_PATH), "utf8"),
        )
      : [];
  const text = formatTypeScript(`${header}${profileModule(profile)}`, tablePath);
  await writeFile(join(repositoryRoot, tablePath), text);
  stdout(`Wrote ${tablePath}`);
  stdout("Verdicts:");
  for (const line of verdicts) stdout(line);
  for (const line of registration) stdout(line);
};

/** Runs the command and returns its exit code. */
export const main = async (
  argv: readonly string[],
  environment: CommandEnvironment = defaultEnvironment(),
): Promise<number> => {
  try {
    await run(argv, environment);
    return 0;
  } catch (error) {
    if (error instanceof UsageError) {
      environment.stderr(error.message);
      environment.stderr(USAGE);
      return 2;
    }
    environment.stderr(error instanceof Error ? error.message : String(error));
    return 1;
  }
};

if (import.meta.main) {
  process.exitCode = await main(process.argv.slice(2));
}

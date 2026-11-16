/**
 * Loading the records the generator derives from: where a record path may point, and what a record
 * file may hold. Pure over a path and the file's bytes (the command reads the file).
 *
 * A record is a workbench record (format `karume-workbench/1`, unwrapped to its `data`) or a bare
 * runtime record (a sweep or attention-bench record written before the workbench existed). Whether
 * the runtime record is usable — its format version, adapter, case sets, an aborted sweep — is the
 * derivation's call, not this module's: it passes the record on unread.
 */

import { createHash } from "node:crypto";
import { isAbsolute, relative, resolve, sep } from "node:path";

/** A command line the generator cannot act on (exit code 2). */
export class UsageError extends Error {
  override name = "UsageError";
}

/** A record file that is neither of the two accepted shapes, or a workbench run without data. */
export class RecordError extends Error {
  override name = "RecordError";
}

/** Which option a record was given to: sweep records or attention-bench records. */
export type RecordKind = "records" | "attention-bench";

/** How a workbench run was made, as its envelope records it. */
export type WorkbenchOrigin = {
  readonly kind: "workbench";
  readonly host: string;
  readonly userAgent?: string;
  readonly backend?: { readonly name: string; readonly source?: string };
  readonly checkout: { readonly commit: string; readonly dirty: boolean };
};

export type RecordOrigin = WorkbenchOrigin | { readonly kind: "bare" };

export type LoadedRecord = {
  /** Repository-relative, with `/` (the spelling the table's provenance cites). */
  readonly path: string;
  /** SHA-256 of the file's bytes as stored (for a workbench record, of the envelope file). */
  readonly sha256: string;
  /** The runtime record, unread (the derivation checks it). */
  readonly report: unknown;
  readonly origin: RecordOrigin;
};

const WORKBENCH_FORMAT = "karume-workbench/1";

/** The workbench entry and the bare format prefix each option takes. */
const EXPECTED: Readonly<Record<RecordKind, { readonly entry: string; readonly format: string }>> =
  {
    records: { entry: "geometry-sweep", format: "karume-geometry-sweep/" },
    "attention-bench": { entry: "attention-bench", format: "karume-attention-bench/" },
  };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * The repository-relative spelling of a record path given on the command line (relative paths are
 * taken from the repository root). The provenance cites this spelling, so a path outside the
 * repository could not be cited and is refused.
 */
export const repositoryRelativePath = (repositoryRoot: string, given: string): string => {
  const absolute = isAbsolute(given) ? resolve(given) : resolve(repositoryRoot, given);
  const path = relative(repositoryRoot, absolute);
  if (path === "" || path === ".." || path.startsWith(`..${sep}`) || isAbsolute(path)) {
    throw new UsageError(`${given}: the path is outside the repository`);
  }
  return path.split(sep).join("/");
};

export const sha256Hex = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex");

const workbenchOrigin = (envelope: Record<string, unknown>, path: string): WorkbenchOrigin => {
  const { environment, checkout } = envelope;
  if (!isRecord(environment) || typeof environment.host !== "string") {
    throw new RecordError(`${path}: the workbench record has no environment.host`);
  }
  if (
    !isRecord(checkout) ||
    typeof checkout.commit !== "string" ||
    typeof checkout.dirty !== "boolean"
  ) {
    throw new RecordError(`${path}: the workbench record has no checkout (commit and dirty)`);
  }
  const { userAgent, backend } = environment;
  const backendName = isRecord(backend) ? backend.name : undefined;
  const backendSource = isRecord(backend) ? backend.source : undefined;
  return {
    kind: "workbench",
    host: environment.host,
    ...(typeof userAgent === "string" ? { userAgent } : {}),
    ...(typeof backendName === "string"
      ? {
          backend: {
            name: backendName,
            ...(typeof backendSource === "string" ? { source: backendSource } : {}),
          },
        }
      : {}),
    checkout: { commit: checkout.commit, dirty: checkout.dirty },
  };
};

/**
 * Reads one record file: a workbench record of the entry the option takes is unwrapped to its
 * `data`, a bare runtime record of the option's kind is passed as it is, and anything else throws a
 * {@link RecordError} that names the file.
 */
export const loadRecord = (bytes: Uint8Array, path: string, kind: RecordKind): LoadedRecord => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (cause) {
    throw new RecordError(`${path}: not a JSON file`, { cause });
  }
  const expected = EXPECTED[kind];
  const sha256 = sha256Hex(bytes);
  const format = isRecord(parsed) ? parsed.format : undefined;
  if (isRecord(parsed) && format === WORKBENCH_FORMAT) {
    const entry = isRecord(parsed.entry) ? parsed.entry.id : undefined;
    if (entry !== expected.entry) {
      throw new RecordError(
        `${path}: a workbench record of entry ${JSON.stringify(entry)}, but --${kind} takes records of the ${expected.entry} entry`,
      );
    }
    if (parsed.data === undefined) {
      throw new RecordError(
        `${path}: the workbench record has no data (outcome ${JSON.stringify(parsed.outcome)})`,
      );
    }
    return { path, sha256, report: parsed.data, origin: workbenchOrigin(parsed, path) };
  }
  if (typeof format === "string" && format.startsWith(expected.format)) {
    return { path, sha256, report: parsed, origin: { kind: "bare" } };
  }
  throw new RecordError(
    `${path}: neither a workbench record of the ${expected.entry} entry nor a bare ${expected.format}* record (format ${JSON.stringify(format)})`,
  );
};

/**
 * The contract of a workbench entry: one module under `src/entries/` that declares its parameters
 * and runs. An entry holds no UI code; the browser shell and the Node run (`run.ts`) render and drive
 * every entry the same way, and the same module runs in both.
 *
 * The parameters and their parsing are `@karume/sweep-app-support/parameters` (the benchmark app
 * writes the sweep entries' records with the same declarations); the result an entry resolves to
 * and its verdict are the record's (`@karume/sweep-app-support/record`).
 *
 * SPEC: docs/spec/workbench.md#an-entry
 */

import type { DistributionSource, HubRepoRef } from "@karume/hub";
import type { SourceKind } from "@karume/local-distributions/browser";
import type { MountName } from "@karume/local-distributions/mounts";
import type { AcquireGpuOptions } from "@karume/runtime";
import type { Parameter, ParameterValues } from "@karume/sweep-app-support/parameters";
import type { EntryResult } from "@karume/sweep-app-support/record";

/** The part of the runtime's `acquireGpu` options the host decides: `{}` in a browser, Dawn's GPU under Node. */
export type GpuOptions = Pick<AcquireGpuOptions, "gpu">;

/**
 * How a run read a distribution: `cached` (the hub's CacheStorage over the dev server's `/hf/`
 * route), `direct` (plain fetch from the dev server's `/models/<mount>/`) or `directory` (the
 * mirror's directory, under Node).
 */
export type DistributionRead = SourceKind | "directory";

/** What a record carries about a distribution a run read. */
export type DistributionFacts = {
  /** The mount's directory relative to the repository root (`models/karume-moss-image`). */
  readonly directory: string;
  /** The sha256 of the mount's `karume.json` as it was read. */
  readonly manifestSha256: string;
  readonly read: DistributionRead;
};

export type OpenedDistribution = {
  /** What `loadManifest` and a family's `fromPretrained` take. */
  readonly ref: HubRepoRef | DistributionSource;
  readonly facts: DistributionFacts;
};

/**
 * Opens a local distribution: `source` is how a browser reads it (`cached` or `direct`); under
 * Node the mirror's directory is read whatever it says. A referenced repository is read from the
 * mount that holds it. Throws `MountUnavailableError` (`@karume/local-distributions/browser`), which
 * names the mount, when its mirror is missing or, for `cached`, when the mount has no HF-compatible
 * coordinates.
 */
export type OpenDistribution = (
  mount: MountName,
  source: SourceKind,
) => Promise<OpenedDistribution>;

/** A file read by ranges (what `@karume/models`' `fromSingleFile` takes). */
export type RangedFile = {
  readonly size: number;
  /** Bytes `[offset, offset + length)` of the file. */
  read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
};

/**
 * How a run read an input file: `direct` (ranged requests to the dev server's `/inputs/` route,
 * in a browser) or `directory` (the file itself, under Node).
 */
export type InputRead = "direct" | "directory";

/** What a record carries about an input file a run read. */
export type InputFacts = {
  /** Relative to the repository root (`inputs/anima/…`). */
  readonly path: string;
  readonly size: number;
  readonly read: InputRead;
};

export type OpenedInput = {
  readonly file: RangedFile;
  readonly facts: InputFacts;
};

/**
 * Opens a file under the repository's `inputs/` (outside git): `path` is relative to `inputs/`,
 * `/`-separated, without `.` or `..` segments. Under Node the file is read from disk; in a
 * browser by ranged requests to the dev server's `/inputs/` route. Throws naming the path when the
 * file cannot be opened.
 */
export type OpenInput = (path: string) => Promise<OpenedInput>;

export type RunContext = {
  readonly parameters: ParameterValues;
  /** Spread into `acquireGpu`. The entry acquires and releases its own GPU context. */
  readonly gpuOptions: GpuOptions;
  /**
   * Aborted by Cancel (or SIGINT under Node). An entry MAY resolve after the abort with what it has;
   * the record's outcome is then `cancelled` either way.
   */
  readonly signal: AbortSignal;
  /** Reports one status line (the latest replaces the previous one). */
  readonly progress: (text: string) => void;
  /** For an entry that reads a local distribution (a file under `src/entries/local/`). */
  readonly openDistribution: OpenDistribution;
  /** For a local entry that reads a file under `inputs/` (a user's weights file). */
  readonly openInput: OpenInput;
  /**
   * The backend's name as the host knows it — what the record's `environment.backend` holds — for
   * an entry whose own record names the backend. Absent when a browser run declared none.
   */
  readonly backend?: string;
};

export type Entry = {
  /** Kebab-case (the record's `ENTRY_ID`), equal to the entry's file name without `.ts`. */
  readonly id: string;
  readonly title: string;
  /** One or two sentences. */
  readonly description: string;
  readonly parameters: readonly Parameter[];
  readonly run: (context: RunContext) => Promise<EntryResult>;
};

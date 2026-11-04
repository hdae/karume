/**
 * Storage of verification results (`outputs/verify/<environment key>/<date>_<series>/`).
 *
 * Reference values (tracked, one row per environment) and results (untracked, growing with every
 * run) are different in nature, so they live in different places. This is the **safe-to-delete**
 * side — `rm -rf outputs/verify` can always be rebuilt (the same convention as the outputs root of
 * docs/assets-layout.md, and a different root from `bench` (benchmarks and evaluation)).
 *
 * Only two kinds of things go here:
 *
 * - `results.json` — the outcome of each case (match / mismatch / reference value written) and the
 *   environment and checkout that produced it. It is written first as "running" with empty `cases`
 *   when the directory is created, and rewritten whole on every `record`. Even a run that fails
 *   before reaching a record never leaves the previous run's outcomes in place (the worst case is a
 *   directory whose artifacts were replaced by this run's bytes while the previous PASS remains).
 * - Artifacts (PNG / WAV) — **every time, success or failure**. Keeping them only on a mismatch
 *   would leave no artifact of the matching run at hand, and no A/B the next time it breaks.
 *
 * Running the same series twice on the same day **keeps the last run** (directories are split down
 * to the date, not per run — splitting per run would leave a person counting which one is the
 * latest).
 *
 * The directory's date is **UTC** (the date part of `toISOString()` at the time the run started).
 * It can differ from the local date — on a JST machine, runs before 09:00 land in the previous
 * day's directory.
 */

import type { Tolerance } from "../../src/reference/allclose.ts";
import { execFile as execFileCallback } from "node:child_process";
import { promisify } from "node:util";
import { isNotFound, mkdirSync, writeTextFile, writeTextFileSync } from "@karume/test-support/fs";
import { ENVIRONMENT, type Environment } from "./environment.ts";

const execFile = promisify(execFileCallback);

/** Root of the results (`outputs/verify/`). */
const DEFAULT_ROOT = new URL("../../../../outputs/verify/", import.meta.url);

/** Repository root (the working directory `git` runs in). */
const REPO_ROOT = new URL("../../../../", import.meta.url);

/** Outcome of one case. */
export type ResultStatus = "pass" | "fail" | "written" | "rewritten";

/**
 * One measurement of a tolerance check (the cases of the golden series record one **per output**).
 *
 * A tolerance check shows numbers only when it fails, so **the difference of a passing run** is
 * kept nowhere. Keeping "how large the difference is on this machine" on every run means the
 * material for deciding whether to loosen a band is at hand before anything breaks (the band's
 * values, the check and the assert have nothing to do with this record).
 *
 * MUST: no derived values (a ratio to the band and the like) — whatever can be derived is derived
 * by the reader (`tools/verify-diff`). Two representations of the same number in two places would
 * allow a state where only one of them was fixed.
 * NOTE: non-finite values (NaN / ±Inf) become `null` through `JSON.stringify` (the reader accepts
 * null).
 */
export type Measurement = {
  /** Output name in the graph. */
  readonly output: string;
  readonly maxAbs: number;
  readonly maxRel: number;
  /** The band used for acceptance. */
  readonly tolerance: Tolerance;
  /** The stage that accepted it (on fail, the last stage measured). */
  readonly stage: "karume" | "spec";
};

/**
 * Band on relative RMS (a declaration, not a row per environment key). Used by the self A/B check
 * (ADR 0110 decision 5) and by the collapse ceiling of the DeBERTa w8a8 mirror check (ADR 0026
 * addendum 2026-10-07).
 */
export type ComparisonBand = {
  /** The metric the band applies to (only relative RMS for now). */
  readonly metric: "relRms";
  /** Lower bound. 0 means no floor (the mirror check applies only the collapse ceiling). */
  readonly floor: number;
  readonly ceiling: number;
};

/**
 * One measurement compared by relative RMS. Recorded by the self A/B check (the difference at the
 * same observation point between the reference partner with an empty `session` and the practical
 * preset, on the same machine with the same weights) and by the collapse ceiling of the mirror
 * check (the difference between the torch mirror golden and the same output of the path that ran).
 *
 * As with {@link Measurement}, the difference of a passing run is kept too (the material for
 * deriving a band is needed before anything breaks).
 *
 * MUST: no derived values (a ratio to the band and the like) — the same reason as the MUST of
 * {@link Measurement}.
 * NOTE: non-finite values (NaN / ±Inf) become `null` through `JSON.stringify` (the reader accepts
 * null). When `band` is `undefined` (not yet derived, or an observation that is only recorded) the
 * field is not written at all.
 */
export type Comparison = {
  /** Observation point (`step1-latent` / an output name in the graph and the like). */
  readonly output: string;
  /**
   * The reference side. For the A/B check the reference partner (the quant name with an empty
   * `session`); for the mirror check the golden's series name (`io-i8a8.<case>`).
   */
  readonly reference: string;
  /**
   * The executed side. For the A/B check the practical preset (the quant name with a `session`);
   * for the mirror check the path that ran (`a8`).
   */
  readonly practical: string;
  /** ‖p − r‖₂ / ‖r‖₂. */
  readonly relRms: number;
  readonly maxAbs: number;
  /** The band used for the check. */
  readonly band: ComparisonBand | undefined;
};

/** One entry recorded in the `cases` of `results.json`. */
export type ResultEntry = {
  readonly id: string;
  readonly status: ResultStatus;
  /** The reference value compared against (absent on a run that wrote the reference value). */
  readonly expected?: string;
  /** Measured sha256 (absent for cases that take no sha). */
  readonly actual?: string;
  /** File name of the artifact (relative to {@link Results.dir}). */
  readonly artifact?: string;
  /**
   * Elapsed time of one case. **What it covers depends on each series' implementation** (in a
   * series that builds one Session and runs several cases, construction and release fall into no
   * case). Not for comparing times across series — the field is for comparing runs of the same
   * series.
   */
  readonly elapsedMs: number;
  readonly note?: string;
  /**
   * Measurements of the tolerance checks. Cases run through {@link runRecordedCase} always have it
   * (an empty array when the body threw before measuring). Only cases that call
   * {@link Results.record} directly lack the field.
   */
  readonly measurements?: readonly Measurement[];
  /**
   * Relative-RMS measurements (the self A/B check — ADR 0110 decision 5; the collapse ceiling of
   * the mirror check — ADR 0026 addendum 2026-10-07). Cases run through {@link runRecordedCase}
   * have it **only when they recorded some** (the outcome shape of cases that record none is
   * unchanged).
   */
  readonly comparisons?: readonly Comparison[];
};

/** Where the results of one series go. */
export type Results = {
  readonly dir: URL;
  /** Location of one artifact (the directory is created on the first call). */
  artifact(name: string): URL;
  /** Records one entry and rewrites `results.json`. */
  record(entry: ResultEntry): Promise<void>;
};

type Checkout = { readonly sha: string; readonly dirty: boolean };

/**
 * One `git` output. **`undefined` where `git` is missing** (results are still kept — not knowing
 * the checkout is a different thing from not keeping results). Anything but NotFound propagates.
 */
const git = async (...args: string[]): Promise<string | undefined> => {
  try {
    const { stdout } = await execFile("git", args, { cwd: REPO_ROOT, encoding: "utf8" });
    return stdout;
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    // Non-zero exit (e.g. not a git checkout) rejects with the numeric exit code.
    if (
      typeof cause === "object" &&
      cause !== null &&
      "code" in cause &&
      typeof cause.code === "number"
    ) {
      return undefined;
    }
    throw cause;
  }
};

const readCheckout = async (): Promise<Checkout | undefined> => {
  const sha = await git("rev-parse", "HEAD");
  if (sha === undefined) return undefined;
  const status = await git("status", "--porcelain");
  if (status === undefined) return undefined;
  return { sha: sha.trim(), dirty: status.trim() !== "" };
};

/** Fixed once per process (every case of one run reports the same time and the same checkout). */
const STARTED_AT = new Date().toISOString();
/** Date of the directory (UTC — see the module doc). */
const TODAY = STARTED_AT.slice(0, 10);
const CHECKOUT: Checkout | undefined = await readCheckout();

/**
 * Opens the results location of one series.
 *
 * MUST: both choosing and creating the location are **deferred until use** — this module is loaded
 * on machines without a GPU too (the tests SKIP through ignore), so requiring the environment key
 * when opening would make "a file that should SKIP" fail during module evaluation.
 */
export const openResults = (
  family: string,
  options: { readonly root?: URL; readonly environment?: Environment } = {},
): Results => {
  const environment = options.environment ?? ENVIRONMENT;
  const root = options.root ?? DEFAULT_ROOT;
  const cases: ResultEntry[] = [];
  let created = false;
  const directory = (): URL => {
    const { key } = environment;
    if (key === undefined) {
      throw new Error(
        `this environment has no environment key (no GPU adapter was acquired), so the results of ${family} ` +
          "have no directory to be written to",
      );
    }
    return new URL(`${key}/${TODAY}_${family}/`, root);
  };
  const documentText = (): string =>
    `${JSON.stringify(
      {
        schema: 1,
        family,
        environment,
        checkout: CHECKOUT ?? null,
        startedAt: STARTED_AT,
        cases,
      },
      undefined,
      2,
    )}\n`;
  // The location is created **on use** (no empty directory is left on a machine where everything
  // SKIPs for lack of assets).
  const ensureDir = (): URL => {
    const dir = directory();
    if (!created) {
      mkdirSync(dir, { recursive: true });
      // Write "running" first (empty `cases`, this run's startedAt). Even if an exception exits
      // before any record, the previous outcomes in the same directory are never read together with
      // this run's artifacts.
      writeTextFileSync(new URL("results.json", dir), documentText());
      created = true;
    }
    return dir;
  };
  return {
    get dir(): URL {
      return directory();
    },
    artifact: (name: string): URL => {
      // A file: URL resolves backslashes as separators too, so they are rejected like `/`.
      if (/[/\\]/.test(name)) {
        throw new Error(`the artifact name must not contain a directory separator: '${name}'`);
      }
      // `%` is decoded during resolution (`%2e%2e` is the parent directory), so the recorded name
      // and the written location would diverge. `?` and `#` diverge the same way, since what
      // follows them moves into the query / fragment and drops out of the path (`a?b.wav` is
      // written to `a`).
      if (name === "" || name === "." || name === ".." || /[%?#]/.test(name)) {
        throw new Error(
          `the artifact name does not point to one file directly under the results directory: '${name}'`,
        );
      }
      const dir = ensureDir();
      const resolved = new URL(name, dir);
      // MUST: check the resolved result (a check of the name's spelling copies the URL resolution
      // rules and can miss cases).
      const rest = resolved.href.startsWith(dir.href) ? resolved.href.slice(dir.href.length) : "";
      // A `?` / `#` in the rest of the href means part of the name sits outside the path (in the
      // query / fragment).
      if (rest === "" || /[/?#]/.test(rest)) {
        throw new Error(
          `the artifact name does not point to one file directly under the results directory: '${name}'`,
        );
      }
      return resolved;
    },
    record: async (entry: ResultEntry): Promise<void> => {
      const dir = ensureDir();
      cases.push(entry);
      await writeTextFile(new URL("results.json", dir), documentText());
    },
  };
};

/**
 * Keeps the outcome of a failed run (**a recording failure never replaces the original
 * verification exception**).
 *
 * It is called only inside a catch that holds a verification exception. Awaiting
 * {@link Results.record} bare there would let an I/O failure (the directory cannot be created or
 * written) propagate as is, replacing the caller's original exception — the only diagnostic that
 * says what broke — with an I/O exception. The recording failure is reported through
 * `console.error`, and the caller rethrows the original exception unchanged.
 *
 * NOTE: the `record` of the success side does not go through this (if it cannot write, failing
 * loudly is right, and there is no diagnostic to lose).
 */
export const recordFailure = async (results: Results, entry: ResultEntry): Promise<void> => {
  try {
    await results.record(entry);
  } catch (cause) {
    console.error(
      `[karume] could not record the outcome of case '${entry.id}' in results.json: ` +
        `${cause instanceof Error ? cause.message : String(cause)}`,
    );
  }
};

/** The recording handle for one case, passed to the body of {@link runRecordedCase}. */
export type RecordedCase = {
  /**
   * Measurements per output (kept on passing runs too — not used for the check). The body records
   * them and they go into the outcome.
   */
  readonly measurements: Measurement[];
  /**
   * Measurements of the self A/B check (the body records them; they go into the outcome only when
   * there is at least one).
   */
  readonly comparisons: Comparison[];
};

/**
 * The outcome when the body returns without an exception. A body that returns nothing is `pass`.
 *
 * `fail` is returned by cases that "fail after measuring every output" — the caller asserts after
 * the outcome (with a note) is kept. Cases that double as the sha check carry the reference-value
 * outcome (including `written` / `rewritten`) and `expected` / `actual` / `artifact` (a copy of
 * `referenceEntryFields` in `reference.ts`).
 */
export type CaseVerdict = {
  readonly status: ResultStatus;
  readonly expected?: string;
  readonly actual?: string;
  readonly artifact?: string;
  readonly note?: string;
};

/** The identity of a case {@link runRecordedCase} records, and how a failed run is handled. */
export type RecordedCaseSpec = {
  /** `cases[].id` in `results.json`. */
  readonly id: string;
  /** Note of a run that threw (omitted = no note). */
  readonly failureNote?: (cause: unknown) => string;
  /**
   * Called before recording, only when the body threw. The hook through which a series running
   * several cases in one test learns "this failure was kept in the case's entry" — not called when
   * the success-side record failed to write and threw (then no outcome of the case was kept).
   */
  readonly onFailure?: () => void;
};

/**
 * Runs one case and always keeps its outcome in `results.json` (the shared frame of every case of
 * the real-weight goldens).
 *
 * - The body returned: its outcome (default `pass`) is recorded with {@link Results.record}. If it
 *   cannot be written, it fails loudly (the same reason as the NOTE of {@link recordFailure}).
 * - The body threw: `fail` is recorded with {@link recordFailure}, then **the same exception is
 *   rethrown**. Exiting without an outcome would leave the outcome of the same day's previous run
 *   in this directory.
 *
 * On both paths the fields follow the declaration order of {@link ResultEntry} (id / status /
 * expected / actual / artifact / elapsedMs / note / measurements / comparisons), and the elapsed
 * time runs from the call to just before recording. `expected` / `actual` / `artifact` appear only
 * when the returned outcome has them, `comparisons` only when the body recorded at least one.
 */
export const runRecordedCase = async (
  results: Results,
  recordedCase: RecordedCaseSpec,
  body: (recorded: RecordedCase) => Promise<CaseVerdict | undefined>,
): Promise<void> => {
  const { id, failureNote, onFailure } = recordedCase;
  const startedAt = performance.now();
  const measurements: Measurement[] = [];
  const comparisons: Comparison[] = [];
  const elapsedMs = (): number => Math.round(performance.now() - startedAt);
  const compared = (): { comparisons?: readonly Comparison[] } =>
    comparisons.length === 0 ? {} : { comparisons };
  let verdict: CaseVerdict | undefined;
  try {
    verdict = await body({ measurements, comparisons });
  } catch (cause) {
    onFailure?.();
    await recordFailure(results, {
      id,
      status: "fail",
      elapsedMs: elapsedMs(),
      ...(failureNote === undefined ? {} : { note: failureNote(cause) }),
      measurements,
      ...compared(),
    });
    throw cause;
  }
  await results.record({
    id,
    status: verdict?.status ?? "pass",
    ...(verdict?.expected === undefined ? {} : { expected: verdict.expected }),
    ...(verdict?.actual === undefined ? {} : { actual: verdict.actual }),
    ...(verdict?.artifact === undefined ? {} : { artifact: verdict.artifact }),
    elapsedMs: elapsedMs(),
    ...(verdict?.note === undefined ? {} : { note: verdict.note }),
    measurements,
    ...compared(),
  });
};

/**
 * sha256 reference values **per environment** (tracked JSON), and their lookup.
 *
 * ## Why rows per environment rather than constants
 *
 * What the reference check claims is "the numbers have not moved by a single bit on this machine",
 * not bit identity across devices (docs/limitations.md). With reference values as constants in the
 * test sources, every check would turn red the moment the GPU changed, and there would be no
 * telling which red is "a regression" and which is "just a different machine". With rows split by
 * {@link Environment.key}, red always means "a regression on that machine".
 *
 * ## Three modes (environment variable `KARUME_REFERENCE`)
 *
 * | Mode | Case without a row | Case with a row |
 * | --- | --- | --- |
 * | unset | explicit SKIP ({@link References.warnMissing} says how to add it) | compare (red on mismatch) |
 * | `write` | **add** the measurement, green | compare (**no overwrite**) |
 * | `rewrite` | add the measurement, green | **overwrite** with it, green (prints old → new) |
 *
 * MUST: a mismatch is never absorbed by a tolerance. `rewrite` is an operation only for "when you
 * can say up front what changed", and it **never touches the rows of other environments** (another
 * machine's reference values are not dragged along).
 *
 * MUST: "SKIP because there is no reference" is never a silent green. {@link registerReferenceGate}
 * adds one failing test in the same form as the guard of ADR 0005 (with an opt-out). The check
 * counts only **the cases the caller registered**, and requires that **all** of them have a row for
 * the current environment, or that a case without one has an explicit held row
 * ({@link referenceGatePasses}).
 *
 * ## Caller patterns (e2e tests with a sha check)
 *
 * - Taking the sha as **an additional check inside an existing test**: call {@link settleOrObserve}
 *   **after** the original checks (tolerance comparison, classification, exact match of `.lab` and
 *   so on) **have passed** (no row is created from an output that broke the original checks) →
 *   record the returned fields ({@link ReferenceOutcome.fields}) in the results → **after
 *   recording**, fail the comparison's `fail` with {@link referenceMismatchMessage} (throwing first
 *   would keep the measured sha out of the results). A case without a row in compare mode goes
 *   through the same path, and its outcome has exactly one shape — write the artifact, take the
 *   sha, record `status: "pass"` + `actual` + `artifact` (no `expected`). There is no comparison,
 *   but the measured sha stays in the results, so `tools/verify-diff` can compare `actual` across
 *   environments (how to create the row is printed by {@link References.warnMissing} at
 *   registration).
 * - **The test itself is the sha check**: register it with
 *   `ignore: !RUNNABLE || references.lacksReference(id)`, and the body records
 *   {@link settleReference} → {@link referenceEntryFields} in the same order as above before
 *   failing (a case without a row never reaches the body).
 *
 * Both call {@link References.warnMissing} and {@link registerReferenceGate} at the end of the
 * file. An artifact that is an f32 tensor is built with {@link f32ArtifactBytes} (one safetensors
 * file, no metadata).
 */

import { readTextFileSync, writeFile, writeTextFileSync } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { fail } from "@std/assert";
import { ENVIRONMENT, type Environment } from "./environment.ts";
import type { ResultEntry, Results } from "./results.ts";
import { buildSafetensors } from "./safetensors.ts";

/** Mode that creates reference values. Unset (= compare only) is `undefined`. */
export type ReferenceMode = "write" | "rewrite";

/**
 * Format version of the fixture (an unreadable version throws — never silently treated as empty).
 */
const SCHEMA = 1;
/** Kind of value the fixture holds (only sha256 for now). */
const KIND = "sha256";

/** Reads the value of `KARUME_REFERENCE`. An unknown spelling throws (never silently "unset"). */
export const parseReferenceMode = (raw: string | undefined): ReferenceMode | undefined => {
  if (raw === undefined) return undefined;
  if (raw === "write" || raw === "rewrite") return raw;
  throw new Error(
    `KARUME_REFERENCE is 'write' or 'rewrite' (received: '${raw}'). ` +
      "When unset, only the comparison with existing reference values runs",
  );
};

/** Reference mode of this run (fixed once, at module evaluation). */
export const REFERENCE_MODE: ReferenceMode | undefined = parseReferenceMode(
  process.env.KARUME_REFERENCE,
);

/**
 * Opt-out that explicitly allows "SKIP with no reference value at all for this environment"
 * (the same shape as `ALLOW_NO_GPU` in `helpers/gpu.ts`).
 */
export const ALLOW_NO_REFERENCE: boolean = process.env.KARUME_ALLOW_NO_REFERENCE === "1";

/** Outcome of the comparison for one case. */
export type ReferenceCheck =
  /** Matched the current environment's row. */
  | { readonly status: "pass"; readonly expected: string }
  /** Differed from the current environment's row (the caller fails with a diagnostic). */
  | { readonly status: "fail"; readonly expected: string }
  /** There was no row, so the measurement was added (`write` / `rewrite`). */
  | { readonly status: "written" }
  /** The row was overwritten with the measurement (`rewrite`). */
  | { readonly status: "rewritten"; readonly previous: string };

/**
 * Maps the comparison outcome to the `expected` field of the results JSON. A run that **created**
 * the reference value has nothing to compare against, so it has none (a rewrite run gets the old
 * value — the results keep what moved from what to what).
 */
export const expectedOf = (check: ReferenceCheck): string | undefined => {
  if (check.status === "written") return undefined;
  return check.status === "rewritten" ? check.previous : check.expected;
};

/**
 * Logs one line for a run that created / rewrote a reference value (compare-only runs say nothing).
 */
export const announceCheck = (caseId: string, check: ReferenceCheck, actual: string): void => {
  if (check.status === "written") {
    console.log(`[reference] ${caseId}: wrote ${actual} as this environment's reference value`);
  } else if (check.status === "rewritten") {
    console.log(`[reference] ${caseId}: rewrote the reference value ${check.previous} → ${actual}`);
  }
};

/** Read / write handle of one reference-value fixture. */
export type References = {
  /** Reference mode of this run. */
  readonly mode: ReferenceMode | undefined;
  /**
   * Environment key the rows are looked up by (`undefined` without a GPU adapter — used to look up
   * held rows).
   */
  readonly environmentKey: string | undefined;
  /** The current environment's row (`undefined` if none). */
  lookup(caseId: string): string | undefined;
  /** No reference value and not a creating mode (= the case is explicitly SKIPped). */
  lacksReference(caseId: string): boolean;
  /**
   * Writes back the current environment's row (writes on every call whatever the mode — the
   * decision is {@link check}'s).
   */
  record(caseId: string, sha256: string): "written" | "rewritten";
  /** Compares according to the mode, writing the row if needed. */
  check(caseId: string, sha256: string): ReferenceCheck;
  /** Warns once about cases without a reference value (called at registration). */
  warnMissing(caseIds: readonly string[]): void;
  /** Location of the fixture (shown in diagnostics). */
  readonly fixtureUrl: URL;
};

type ReferenceDocument = {
  readonly schema: number;
  readonly kind: string;
  readonly cases: Record<string, Record<string, string>>;
};

/** Reads the fixture. Missing, broken or a different version all throw (never treated as empty). */
const readDocument = (fixtureUrl: URL): ReferenceDocument => {
  let text: string;
  try {
    text = readTextFileSync(fixtureUrl);
  } catch (cause) {
    throw new Error(`the reference-value fixture cannot be read: ${fixtureUrl.pathname}`, {
      cause,
    });
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new Error(`the reference-value fixture is broken as JSON: ${fixtureUrl.pathname}`, {
      cause,
    });
  }
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error(`the reference-value fixture is not an object: ${fixtureUrl.pathname}`);
  }
  const document = parsed as Partial<ReferenceDocument>;
  if (document.schema !== SCHEMA) {
    throw new Error(
      `the schema of the reference-value fixture is ${String(document.schema)} (expected ${SCHEMA}): ` +
        fixtureUrl.pathname,
    );
  }
  if (document.kind !== KIND) {
    throw new Error(
      `the kind of the reference-value fixture is '${String(document.kind)}' (expected '${KIND}'): ` +
        fixtureUrl.pathname,
    );
  }
  const { cases } = document;
  if (typeof cases !== "object" || cases === null) {
    throw new Error(`the reference-value fixture has no cases: ${fixtureUrl.pathname}`);
  }
  for (const caseId of Object.keys(cases)) {
    const rows = cases[caseId];
    if (typeof rows !== "object" || rows === null) {
      throw new Error(`case '${caseId}' of the reference-value fixture is not a table of rows`);
    }
    for (const key of Object.keys(rows)) {
      if (typeof rows[key] !== "string") {
        throw new Error(
          `case '${caseId}' / environment '${key}' of the reference-value fixture is not a string`,
        );
      }
    }
  }
  return { schema: SCHEMA, kind: KIND, cases };
};

/**
 * Makes the output stable (case names and environment keys in lexicographic order, 2 spaces,
 * trailing newline). A diff showing only "the rows that actually changed" is a hard requirement for
 * a file in which the rows of several environments live together.
 */
const serialize = (cases: Record<string, Record<string, string>>): string => {
  const sorted: Record<string, Record<string, string>> = {};
  for (const caseId of Object.keys(cases).sort()) {
    const rows = cases[caseId];
    const row: Record<string, string> = {};
    for (const key of Object.keys(rows).sort()) row[key] = rows[key];
    sorted[caseId] = row;
  }
  return `${JSON.stringify({ schema: SCHEMA, kind: KIND, cases: sorted }, undefined, 2)}\n`;
};

/** Injection points of {@link openReferences} (for unit tests — runtime calls pass nothing). */
export type OpenReferencesOptions = {
  /** Environment the reference values are looked up for (default: this run's environment). */
  readonly environment?: Environment;
  /**
   * Reference mode of this run.
   *
   * MUST: the environment variable `KARUME_REFERENCE` is read **only when the field is omitted
   * altogether**. Taking it as a default parameter would turn "an `undefined` passed explicitly to
   * mean compare-only" into the variable's write / rewrite, and in a run with
   * `KARUME_REFERENCE=write` the unit tests would run in write mode.
   */
  readonly mode?: ReferenceMode | undefined;
};

/**
 * Opens a reference-value fixture.
 *
 * `environment` / `mode` can be injected for unit tests (so the mode branches can be checked
 * without a real GPU or environment variables). Runtime calls omit the second argument altogether.
 */
export const openReferences = (
  fixtureUrl: URL,
  options: OpenReferencesOptions = {},
): References => {
  const environment = options.environment ?? ENVIRONMENT;
  const mode = Object.hasOwn(options, "mode") ? options.mode : REFERENCE_MODE;
  const { cases } = readDocument(fixtureUrl);
  const { key } = environment;
  const requireKey = (): string => {
    if (key === undefined) {
      throw new Error(
        `this environment has no environment key (no GPU adapter was acquired), so no row of ${fixtureUrl.pathname} ` +
          "can be written",
      );
    }
    return key;
  };
  const lookup = (caseId: string): string | undefined =>
    key === undefined ? undefined : cases[caseId]?.[key];
  /**
   * Reads the current environment's row **from disk** (the only input to the decision).
   *
   * MUST: the comparison is decided against the latest row on disk, not against the copy taken at
   * open time. With two handles open for the same environment key, deciding on the copy would treat
   * a row written by the other handle as "missing", and `write` would silently overwrite it
   * (breaking the MUST of the table at the top: "`write` does not overwrite a case with a row").
   */
  const onDiskLookup = (caseId: string): string | undefined =>
    key === undefined ? undefined : readDocument(fixtureUrl).cases[caseId]?.[key];
  const record = (caseId: string, sha256: string): "written" | "rewritten" => {
    const current = requireKey();
    // MUST: re-read right before writing. Writing back the copy taken at open time would drop
    // **other environments' rows** another handle added in between ("never touches the rows of
    // other environments" holds because of this re-read).
    const onDisk = readDocument(fixtureUrl).cases;
    // The old value comes from the same re-read too (taking it from the copy would report a row
    // another handle added as "was missing").
    const previous = onDisk[caseId]?.[current];
    onDisk[caseId] = { ...onDisk[caseId], [current]: sha256 };
    writeTextFileSync(fixtureUrl, serialize(onDisk));
    cases[caseId] = { ...cases[caseId], [current]: sha256 };
    return previous === undefined ? "written" : "rewritten";
  };
  return {
    mode,
    environmentKey: key,
    fixtureUrl,
    lookup,
    lacksReference: (caseId: string): boolean => mode === undefined && lookup(caseId) === undefined,
    record,
    check: (caseId: string, sha256: string): ReferenceCheck => {
      const expected = onDiskLookup(caseId);
      if (expected === undefined) {
        if (mode === undefined) {
          throw new Error(
            `case '${caseId}' has no reference value for this environment (${key ?? "no GPU"}). ` +
              "To create it, run the same lane with KARUME_REFERENCE=write",
          );
        }
        record(caseId, sha256);
        return { status: "written" };
      }
      if (mode === "rewrite") {
        record(caseId, sha256);
        return { status: "rewritten", previous: expected };
      }
      return { status: sha256 === expected ? "pass" : "fail", expected };
    },
    warnMissing: (caseIds: readonly string[]): void => {
      if (mode !== undefined) return;
      const missing = caseIds.filter((caseId) => lookup(caseId) === undefined);
      if (missing.length === 0) return;
      console.warn(
        `[karume] cases without a reference value for this environment (${key ?? "no GPU"}): ${missing.join(" / ")}. ` +
          "They are explicitly SKIPped. To create them, run the same lane with KARUME_REFERENCE=write " +
          `(rows live in: ${fixtureUrl.pathname})`,
      );
    },
  };
};

/**
 * Held rows: case ID → environment key → reason (the same shape as `HELD_SERIES` of ADR 0106
 * addendum decision 8).
 *
 * The place that tells the reference check about cases that "are not run on that machine, so have
 * no reference row for the current environment". As with decision 8, a row may be added only for
 * cases that **crash the whole process** when run on that machine, or whose model's resident weights
 * exceed that machine's GPU so the load fails with `GpuOutOfMemoryError` (ADR 0106 addendum decision
 * 10) — cases whose numbers do not match stay red and get fixed. The reason states the evidence and
 * the condition for lifting it; once it is resolved, the held row is removed.
 */
export type ReferenceHolds = Readonly<Record<string, Readonly<Record<string, string>>>>;

/**
 * Inputs to the reference check's decision ({@link referenceGateFindings} — the failure message
 * names cases from these).
 */
export type ReferenceGateFindings = {
  /**
   * Registered cases with neither a row nor a held row for the current environment (the ones to
   * create rows for with `KARUME_REFERENCE=write`).
   */
  readonly missing: readonly string[];
  /**
   * Registered cases held for the current environment that also have a row (the held row is stale —
   * remove it if resolved).
   */
  readonly staleHolds: readonly string[];
};

/** Reason of the current environment's held row (`undefined` if none). */
const heldReasonOf = (
  holds: ReferenceHolds,
  caseId: string,
  environmentKey: string | undefined,
): string | undefined => {
  if (environmentKey === undefined || !Object.hasOwn(holds, caseId)) return undefined;
  const byEnvironment = holds[caseId];
  return Object.hasOwn(byEnvironment, environmentKey) ? byEnvironment[environmentKey] : undefined;
};

/** Sorts registered cases into "no row" and "held yet has a row" (pure function). */
export const referenceGateFindings = (
  references: Pick<References, "lookup" | "environmentKey">,
  caseIds: readonly string[],
  holds: ReferenceHolds = {},
): ReferenceGateFindings => {
  const missing: string[] = [];
  const staleHolds: string[] = [];
  for (const caseId of caseIds) {
    const hasRow = references.lookup(caseId) !== undefined;
    const held = heldReasonOf(holds, caseId, references.environmentKey) !== undefined;
    if (!hasRow && !held) missing.push(caseId);
    if (hasRow && held) staleHolds.push(caseId);
  }
  return { missing, staleHolds };
};

/**
 * The green condition of the reference check (pure function — the regression tests poke it
 * directly).
 *
 * It counts only **the cases this run registered**. Counting across the whole fixture would let
 * orphan rows left by renamed or removed cases (rows nobody compares against any more) hide the
 * SKIP of live cases in green. Opt-in cases (enabled by an environment variable) are left out of
 * the registration by the caller on runs where they are disabled.
 *
 * MUST: **every** registered case has a row for the current environment, or a case without one has
 * a held row for the current environment. With "green if any one has a row", a forgotten row for a
 * newly added case would pass green as a SKIP with a warning (the lane's green would read as
 * verified for a case that has no row at all and was never verified).
 * MUST: a case with both a held row and a row for the current environment is red (the held row is
 * stale — keeping it means the caller keeps the case stopped and never compares against the row it
 * has).
 * Zero registered cases is red (a runnable check with nothing to check is a registration the caller
 * forgot). A creating mode naturally has no rows, so it is always green.
 */
export const referenceGatePasses = (
  references: Pick<References, "mode" | "lookup" | "environmentKey">,
  caseIds: readonly string[],
  holds: ReferenceHolds = {},
): boolean => {
  if (references.mode !== undefined) return true;
  if (caseIds.length === 0) return false;
  const { missing, staleHolds } = referenceGateFindings(references, caseIds, holds);
  return missing.length === 0 && staleHolds.length === 0;
};

/**
 * The reference check (one per sha file). A guard that keeps "some case has no reference value for
 * this environment" from becoming **a silent green**, with an opt-out like the GPU guard of
 * ADR 0005.
 *
 * `caseIds` is every case ID the file registered (the same set as passed to `warnMissing`).
 * `holds` are the held rows of cases not run in the current environment ({@link ReferenceHolds}).
 */
export const registerReferenceGate = (
  references: References,
  options: {
    readonly runnable: boolean;
    readonly caseIds: readonly string[];
    readonly holds?: ReferenceHolds;
  },
): void => {
  const environment = references.environmentKey ?? "no GPU";
  const holds = options.holds ?? {};
  // Cases stopped by a held row announce themselves once at registration (ADR 0106 addendum
  // decision 8 — no silent SKIP).
  const heldHere = options.caseIds.flatMap((caseId) => {
    const reason = heldReasonOf(holds, caseId, references.environmentKey);
    return reason === undefined ? [] : [`${caseId}（${reason}）`];
  });
  if (options.runnable && heldHere.length > 0) {
    console.warn(
      `[karume] cases without a reference value in this environment (${environment}) because of a held row: ` +
        heldHere.join(" / "),
    );
  }
  test.skipIf(ALLOW_NO_REFERENCE || !options.runnable)(
    `reference check: every registered case has a reference value for this environment (${environment})`,
    () => {
      if (referenceGatePasses(references, options.caseIds, holds)) return;
      const { missing, staleHolds } = referenceGateFindings(references, options.caseIds, holds);
      const lines = [
        `the reference-value check of this environment (${environment}) is red (${options.caseIds.length} registered cases, ` +
          `rows live in: ${references.fixtureUrl.pathname}).`,
      ];
      if (options.caseIds.length === 0) {
        lines.push(
          "The runnable check has 0 registered cases (the caller forgot to register them).",
        );
      }
      if (missing.length > 0) {
        lines.push(
          `${missing.length} cases without a row: ${missing.join(" / ")}. ` +
            "These sha checks were explicitly SKIPped — treated as FAIL as in ADR 0005 (it would make " +
            "something unverified read as verified). To create this machine's rows, run the same lane " +
            "with KARUME_REFERENCE=write (existing rows are not overwritten). Only for cases that " +
            "crash the whole process on that machine, add a held row (ReferenceHolds) with the reason and the condition for lifting it.",
        );
      }
      if (staleHolds.length > 0) {
        lines.push(
          `cases with both a held row and a row for the current environment: ${staleHolds.join(" / ")}. ` +
            "If the reason for the hold is resolved, remove the held row.",
        );
      }
      lines.push("To pass deliberately without reference values, set KARUME_ALLOW_NO_REFERENCE=1.");
      fail(lines.join("\n"));
    },
  );
};

/**
 * sha256 of bytes (64 lowercase hex digits).
 *
 * NOTE: `src/format/container` has a function of the same name, but this one is the yardstick of
 * the verification, so it calls Web Crypto directly instead of leaning on the implementation under
 * test.
 */
export const sha256Hex = async (bytes: Uint8Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** Outcome of {@link settleReference} (the caller records it in the results). */
export type ReferenceSettlement = {
  /** sha256 of the artifact file's bytes (the value of the results' `actual` field). */
  readonly sha256: string;
  readonly check: ReferenceCheck;
  /** Location of the artifact (inside the results directory). */
  readonly artifactUrl: URL;
};

/**
 * Writes the artifact into the results directory and compares its sha256 with the reference value
 * (the shared tail of the sha checks).
 *
 * - The artifact is written **before the comparison**, on success or failure (keeping it only on a
 *   mismatch would leave no artifact of the matching run at hand, and no A/B the next time it
 *   breaks).
 * - The sha is taken from the same bytes that were written (the results' `actual` always matches
 *   the artifact's sha).
 *
 * It does not record into the results — the caller `record`s through {@link referenceEntryFields}
 * or puts it into the outcome of `runRecordedCase`. It does not throw on a mismatch either
 * (throwing before recording would keep the measured sha out of the results). For a case without a
 * row in compare mode, {@link References.check} throws — callers that take the sha as an additional
 * check use {@link settleOrObserve} instead (which fixes the single outcome shape of a case without
 * a row).
 */
export const settleReference = async (
  references: References,
  results: Pick<Results, "artifact">,
  settled: {
    /** Case ID of the reference value. */
    readonly id: string;
    /** File name of the artifact (relative to the results directory). */
    readonly artifact: string;
    readonly bytes: Uint8Array<ArrayBuffer>;
  },
): Promise<ReferenceSettlement> => {
  const artifactUrl = results.artifact(settled.artifact);
  await writeFile(artifactUrl, settled.bytes);
  const sha256 = await sha256Hex(settled.bytes);
  const check = references.check(settled.id, sha256);
  announceCheck(settled.id, check, sha256);
  return { sha256, check, artifactUrl };
};

/**
 * Maps the outcome to the fields of one result entry (in the order `status` / `expected` /
 * `actual` / `artifact`).
 *
 * `expected` follows {@link expectedOf} (a creating run has no such field; a rewrite run has the
 * old value).
 */
export const referenceEntryFields = (
  settlement: ReferenceSettlement,
  artifact: string,
): Pick<ResultEntry, "status" | "expected" | "actual" | "artifact"> => {
  const expected = expectedOf(settlement.check);
  return {
    status: settlement.check.status,
    ...(expected === undefined ? {} : { expected }),
    actual: settlement.sha256,
    artifact,
  };
};

/** Outcome of {@link settleOrObserve}. */
export type ReferenceOutcome = {
  /**
   * Fields of one result entry (in the order `status` / `expected` / `actual` / `artifact` — spread
   * and recorded as is).
   */
  readonly fields: Pick<ResultEntry, "status" | "expected" | "actual" | "artifact">;
  /**
   * The comparison outcome. `undefined` when there was no row in compare mode and the comparison
   * was skipped.
   */
  readonly settlement: ReferenceSettlement | undefined;
};

/**
 * The tail of the sha taken as an additional check (the first of the module doc's "caller
 * patterns").
 *
 * - A row exists, or a creating mode: the outcome of {@link settleReference} (fields from
 *   {@link referenceEntryFields}).
 * - No row in compare mode: only the comparison is skipped. The artifact is written the same way,
 *   and the sha is taken from the same bytes. The outcome is `status: "pass"` (it is called after
 *   the original checks passed) + `actual` + `artifact`, without `expected` (nothing was compared
 *   against).
 *
 * The branch is not left to each caller because the result shape of a case without a row would
 * split per series (with some series recording the measured sha and others recording nothing, the
 * cross-environment `actual` comparison would work only for some series).
 */
export const settleOrObserve = async (
  references: References,
  results: Pick<Results, "artifact">,
  settled: {
    /** Case ID of the reference value. */
    readonly id: string;
    /** File name of the artifact (relative to the results directory). */
    readonly artifact: string;
    readonly bytes: Uint8Array<ArrayBuffer>;
  },
): Promise<ReferenceOutcome> => {
  if (references.lacksReference(settled.id)) {
    await writeFile(results.artifact(settled.artifact), settled.bytes);
    return {
      fields: {
        status: "pass",
        actual: await sha256Hex(settled.bytes),
        artifact: settled.artifact,
      },
      settlement: undefined,
    };
  }
  const settlement = await settleReference(references, results, settled);
  return { fields: referenceEntryFields(settlement, settled.artifact), settlement };
};

/**
 * Turns one f32 tensor into the bytes of an artifact (one safetensors file). The tensor name is the
 * graph output name the caller passes, the dtype is `F32`, and the shape is the output's.
 *
 * MUST: no metadata — values that change per run (time, checkout) would move the sha even with the
 * same numbers, and the reference row would capture "when it ran" instead of "a numeric
 * regression".
 * MUST: a mismatch between the shape's element count and the data length throws (an artifact cut
 * out wrongly is never pinned silently).
 */
export const f32ArtifactBytes = (
  name: string,
  shape: readonly number[],
  data: Float32Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> => {
  const count = shape.reduce((product, dim) => product * dim, 1);
  if (count !== data.length) {
    throw new Error(
      `artifact '${name}': shape [${shape.join(
        ",",
      )}] has ${count} elements, which disagrees with the data length ${data.length}`,
    );
  }
  return new Uint8Array(
    buildSafetensors([
      {
        name,
        dtype: "F32",
        shape,
        data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      },
    ]),
  );
};

/**
 * The diagnostic message for a mismatch with the reference value.
 *
 * MUST: no falling back to a tolerance here. The reference is only a sha256 (the reference bytes
 * are not kept), so the position of the first difference cannot be reported in principle — instead
 * the locations of the artifact and the fixture are listed side by side so a person can compare
 * them. Passing an outcome that is not a mismatch is a caller error, so it throws.
 */
export const referenceMismatchMessage = (
  label: string,
  settlement: ReferenceSettlement,
  references: Pick<References, "fixtureUrl">,
): string => {
  const { check } = settlement;
  if (check.status !== "fail") {
    throw new Error(
      `${label}: the outcome is '${check.status}', so no mismatch diagnostic can be built`,
    );
  }
  return (
    `${label}: the artifact's sha256 does not match the reference\n` +
    `  expected ${check.expected}\n  actual ${settlement.sha256}\n` +
    `  artifact ${settlement.artifactUrl.pathname} (the reference is only a sha256, not the bytes, so the position of the first difference cannot be reported)\n` +
    `  reference values ${references.fixtureUrl.pathname}\n` +
    "  do not fall back to a tolerance — if the change is intended, be able to say what changed, then " +
    "re-record with KARUME_REFERENCE=rewrite"
  );
};

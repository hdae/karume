/**
 * The two matching keys, among those that decide whether a saved table may be applied, that are
 * derived from the runtime (DECIDED: ADR 0117 decision 4) — the **kernel fingerprint**
 * ({@link geometryProfileKernelsId}) and the **case-set version** ({@link sweepCaseSetId}).
 *
 * Both are synchronous pure functions that read neither the GPU nor the clock, and the same runtime
 * always gives the same value (codegen determinism — same key → byte-identical WGSL). The runtime's
 * version string is not used because the runtime has no version constant, a version would discard
 * tables even on releases that do not touch the GEMM kernels, and on unreleased checkouts it stays
 * the same while the kernels change. The fingerprint changes only when the kernels the table points
 * to or the default kernels change.
 *
 * The matching keys on the sweep record's side (ADR 0117 decision 8) are derived in the same style:
 * `defaultKernels` for the ratio baseline ({@link geometryProfileKernelsId} applied to the default
 * table) and `candidateKernels` for the measured candidates ({@link sweepCandidateKernelsId}).
 * `defaultKernels` protects only the ratio baseline; the speed and output match of an adopted
 * candidate come from the candidate's own kernels — a record without the field guarantees only the
 * default baseline.
 *
 * The hash is FNV-1a 64 bit ({@link fnv1a64Hex}). It is synchronous to keep the pure matching
 * function and the injection entry point's callback (ADR 0117 decision 6) synchronous
 * (`crypto.subtle` is async only). Only accidental collisions are considered; adversarial forgery
 * is out of scope (the table is the user's own stored data).
 *
 * MUST: nothing is computed at import time (functions, not constants — every module is free of side
 * effects, ADR 0117 decision 1).
 */
import { CodegenError, DispatchLimitError } from "../codegen/errors.ts";
import { conv2dIgemmMTile } from "../kernels/conv2d.ts";
import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import type { I8a8Geometry } from "../kernels/i8a8-geometry.ts";
import {
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
} from "../kernels/geometry-profile.ts";
import {
  ALL_SWEEP_CASES,
  PROFILE_GEMM_ROWS_BOUNDS,
  SWEEP_CASES,
  type SweepCase,
  sweepCasesFor,
} from "./cases.ts";
import {
  type CandidateSet,
  conv2dCandidate,
  gemmCandidate,
  type GeometryCandidate,
  i8a8Candidate,
} from "./geometries.ts";
import { type CasePlan, casePlan } from "./harness.ts";

/** The upper / lower 32 bits of the FNV-1a 64-bit offset basis (0xcbf29ce484222325). */
const FNV_OFFSET_HIGH = 0xcbf29ce4;
const FNV_OFFSET_LOW = 0x84222325;
/**
 * The low part of the FNV 64-bit prime 2^40 + 0x1b3 (the 2^40 part is added as an 8-bit left
 * shift).
 */
const FNV_PRIME_LOW = 0x1b3;
const TWO_POW_32 = 0x1_0000_0000;

/**
 * FNV-1a 64 bit of the string's UTF-8 bytes (16 hex digits).
 *
 * The 64-bit state is held in two 32-bit words (BigInt allocates heavily per byte — the fingerprint
 * input is about 1 MB). The product `h · (2^40 + 0x1b3)` mod 2^64 is built as the sum of the carry
 * of low word × 0x1b3 (< 2^41, exact in a double), high word × 0x1b3, and the low word × 2^40 (an
 * 8-bit left shift into the high word).
 */
export const fnv1a64Hex = (text: string): string => {
  const bytes = new TextEncoder().encode(text);
  let high = FNV_OFFSET_HIGH;
  let low = FNV_OFFSET_LOW;
  // An index loop because for-of over a Uint8Array is about 2x slower on V8 (Deno 2.9.6) (11 → 5 ms
  // for about 1 MB)
  for (let index = 0; index < bytes.length; index += 1) {
    low = (low ^ bytes[index]) >>> 0;
    const product = low * FNV_PRIME_LOW;
    high = (Math.imul(high, FNV_PRIME_LOW) + Math.floor(product / TWO_POW_32) + (low << 8)) >>> 0;
    low = product >>> 0;
  }
  return `${high.toString(16).padStart(8, "0")}${low.toString(16).padStart(8, "0")}`;
};

/**
 * The spelling of `Infinity` written in JSON (`JSON.parse` reads an out-of-range number as Infinity
 * — plain `JSON.stringify` writes null). The table's JSON (`geometryProfileJson` in derive.ts) and
 * the canonical JSON ({@link canonicalJson}) use this one — with separate spellings, changing only
 * one would make the saved table and the derivation of the matching keys disagree.
 */
export const INFINITY_JSON = "1e999";

/**
 * Canonical JSON (keys sorted by UTF-16 code unit, no whitespace, undefined fields dropped).
 * `Infinity` is written as {@link INFINITY_JSON} (plain `JSON.stringify` writes null, which would
 * spell the same even if the last boundary changed to another value).
 */
export const canonicalJson = (value: unknown): string => {
  if (typeof value === "number") {
    if (Number.isFinite(value)) return JSON.stringify(value);
    if (value === Number.POSITIVE_INFINITY) return INFINITY_JSON;
    if (value === Number.NEGATIVE_INFINITY) return `-${INFINITY_JSON}`;
    throw new Error("Canonical JSON: NaN cannot be written");
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const entries: [string, unknown][] = Object.entries(value);
    return `{${entries
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
      .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
      .join(",")}}`;
  }
  throw new Error(`Canonical JSON: a value of type ${typeof value} cannot be written`);
};

/**
 * The fingerprint's dispatch-count limit = `maxComputeWorkgroupsPerDimension` that WebGPU
 * guarantees on every device (default value 65535). The workgroup counts do not depend on the limit
 * (it only throws when exceeded), which keeps the fingerprint independent of the device.
 */
const FINGERPRINT_WORKGROUP_LIMIT = 65535;

/**
 * Deriving the fingerprint could not build a case's dispatch (it fails a codegen check, or the
 * dispatch count exceeds {@link FINGERPRINT_WORKGROUP_LIMIT}). The message names the case, the
 * table's field and the geometry; `cause` is the codegen exception. Matching (mismatch.ts) returns
 * it as a mismatch message; the generator (derive.ts) throws without building a table.
 */
export class KernelsIdError extends Error {
  override readonly name = "KernelsIdError";
}

/**
 * The note added when the dispatch-count limit is hit (why a geometry that passed the sweep under
 * the device's limit can still fail).
 */
const LIMIT_NOTE =
  `to keep the fingerprint independent of the device, dispatches are built with the default ` +
  `maxComputeWorkgroupsPerDimension ${FINGERPRINT_WORKGROUP_LIMIT} that WebGPU guarantees on every device — ` +
  "a table that exceeds this count yields no fingerprint, even if its geometry passed the sweep on a device with a larger limit";

/**
 * Turns a codegen exception from `build` into a {@link KernelsIdError} that names `where` (other
 * exceptions pass through).
 */
const naming = <T>(where: () => string, build: () => T): T => {
  try {
    return build();
  } catch (cause) {
    if (cause instanceof DispatchLimitError) {
      throw new KernelsIdError(`${where()}: ${cause.message} (${LIMIT_NOTE})`, { cause });
    }
    if (cause instanceof CodegenError) {
      throw new KernelsIdError(`${where()}: ${cause.message}`, { cause });
    }
    throw cause;
  }
};

/** The field the table applies to a case (for naming in messages) and its geometry. */
type ProfileSlot = { readonly field: string; readonly candidate: GeometryCandidate };

/**
 * The geometry the table applies to a case (the same selection functions as the Session path —
 * `gemmRowsGeometry(profile, m)` in src/runtime/recipe-builders/linear.ts, `conv2dProfileGeometry`
 * and `i8a8.conv3d` in conv.ts, the fields in attention.ts). bmm's M is the row count of one matrix
 * (as in buildBmm).
 */
const profileSlot = (profile: GeometryProfile, sweepCase: SweepCase): ProfileSlot => {
  switch (sweepCase.op) {
    case "linear":
    case "matmul":
    case "bmm":
      return {
        field: `gemmRows bucket for M ${sweepCase.m}`,
        candidate: gemmCandidate(gemmRowsGeometry(profile, sweepCase.m)),
      };
    case "attention":
      return {
        field: `attention.${sweepCase.stage}`,
        candidate: gemmCandidate(
          sweepCase.stage === "qk" ? profile.attention.qk : profile.attention.pv,
        ),
      };
    case "conv2d": {
      const mTile = conv2dIgemmMTile(sweepCase.channelsOut);
      return {
        field: `conv2d.rows${mTile}`,
        candidate: conv2dCandidate(conv2dProfileGeometry(profile, mTile)),
      };
    }
    case "i8a8-linear":
      return { field: "i8a8.linear", candidate: i8a8Candidate(profile.i8a8.linear) };
    case "i8a8-attention":
      return sweepCase.stage === "qk"
        ? { field: "i8a8.attentionQk", candidate: i8a8Candidate(profile.i8a8.attentionQk) }
        : { field: "i8a8.attentionPv", candidate: i8a8Candidate(profile.i8a8.attentionPv) };
    case "i8a8-conv3d":
      return { field: "i8a8.conv3d", candidate: i8a8Candidate(profile.i8a8.conv3d) };
  }
};

/**
 * The dp4a variants (WGSL language feature `packed_4x8_integer_dot_product`). i8a8's WGSL splits in
 * two on this feature (the numerics are identical — `dp4aAvailable` in linear-i8a8.ts), and which
 * one the device applying the table uses cannot be decided from the adapter information, so i8a8
 * cases put both variants into the fingerprint. The WGSL of other ops does not depend on dp4a.
 */
const dp4aVariants = (sweepCase: SweepCase): readonly boolean[] =>
  sweepCase.op === "i8a8-linear" ||
  sweepCase.op === "i8a8-attention" ||
  sweepCase.op === "i8a8-conv3d"
    ? [false, true]
    : [false];

/** The sweep's case plan (harness.ts's `casePlan` — tests inject a replacement for the default). */
export type CasePlanner = (sweepCase: SweepCase, limit: number, dp4a: boolean) => CasePlan;

/**
 * The WGSL reduced to its tokens: comments removed (`//` to the end of the line, `/* … *\/` with
 * nesting — WGSL block comments nest and WGSL has no string literals, so every `//` and `/*` opens
 * a comment) and runs of whitespace collapsed to one space. Two shaders with the same token text
 * compile to the same program, so the fingerprint hashes this form (DECIDED: ADR 0117 addendum
 * 2026-10-19): editing a comment or re-wrapping a line changes no kernel and moves no fingerprint.
 */
export const wgslTokenText = (wgsl: string): string => {
  let out = "";
  let i = 0;
  while (i < wgsl.length) {
    if (wgsl.startsWith("//", i)) {
      const newline = wgsl.indexOf("\n", i);
      i = newline < 0 ? wgsl.length : newline;
      continue;
    }
    if (wgsl.startsWith("/*", i)) {
      let depth = 1;
      i += 2;
      while (i < wgsl.length && depth > 0) {
        if (wgsl.startsWith("/*", i)) {
          depth += 1;
          i += 2;
        } else if (wgsl.startsWith("*/", i)) {
          depth -= 1;
          i += 2;
        } else i += 1;
      }
      out += " ";
      continue;
    }
    out += wgsl[i];
    i += 1;
  }
  return out.replace(/\s+/g, " ").trim();
};

/**
 * One fingerprint line of one dispatch (JSON of pipeline key, params, the WGSL's token text and
 * workgroups — contains no newline).
 */
const launchLine = (launch: ReturnType<CasePlan["launch"]>): string =>
  JSON.stringify([launch.key, [...launch.params], wgslTokenText(launch.wgsl), launch.workgroups]);

/**
 * The body of the kernel fingerprint ({@link geometryProfileKernelsId} — a form into which the
 * cases and the case plan can be injected).
 *
 * For each case (in the order of `cases`) × dp4a variant (false → true), builds the dispatches of
 * the table's geometry and of the default geometry (the case plan's `defaultCandidate` = the
 * field's fallback — the value the generator compares with the sweep's default row), turns pipeline
 * key, params, the WGSL's token text ({@link wgslTokenText}) and workgroups into one line of JSON,
 * and hashes the lines joined with newlines (JSON contains no newline, so the join is unambiguous).
 *
 * MUST: dispatches are built with the same case plan as the sweep — the kernels measured by the
 * sweep and the kernels the fingerprint points to are never built by separate paths. `id`, `match`
 * and `provenance` are not read (only the table's geometries affect the fingerprint).
 *
 * A dispatch that cannot be built (a codegen check, the dispatch-count limit) throws a
 * {@link KernelsIdError} that names the case, the field and the geometry.
 */
export const profileKernelsId = (
  profile: GeometryProfile,
  cases: readonly SweepCase[],
  planner: CasePlanner,
): string => {
  const lines: string[] = [];
  for (const sweepCase of cases) {
    const slot = naming(
      () => `Table '${profile.id}', field for case ${sweepCase.id}`,
      () => profileSlot(profile, sweepCase),
    );
    for (const dp4a of dp4aVariants(sweepCase)) {
      const plan = planner(sweepCase, FINGERPRINT_WORKGROUP_LIMIT, dp4a);
      const roles = [
        { field: `table '${profile.id}', ${slot.field}`, candidate: slot.candidate },
        {
          field: "default geometry (defaultCandidate of the case plan)",
          candidate: plan.defaultCandidate,
        },
      ];
      for (const { field, candidate } of roles) {
        const launch = naming(
          () =>
            `Case ${sweepCase.id}, ${field}: cannot build the dispatch of geometry ${candidate.name} ${JSON.stringify(
              candidate.geometry,
            )}`,
          () => plan.launch(candidate),
        );
        lines.push(launchLine(launch));
      }
    }
  }
  return fnv1a64Hex(lines.join("\n"));
};

/**
 * The table's kernel fingerprint (`provenance.kernels` — ADR 0117 decision 4): the hash (16 hex
 * digits) of the kernels that the table's geometry and the default geometry build for the shapes of
 * every sweep case. Reads neither the GPU nor the clock.
 *
 * The value changes when a GEMM kernel (key, params, WGSL tokens, dispatch count) or the default
 * geometry changes. Changes outside key, params, WGSL and workgroups (the order of recipe bindings
 * and the like) and changes to the WGSL's comments and whitespace do not show in the fingerprint —
 * the runtime's own checks (codegen snapshots, GPU tests with the default geometry) cover the
 * former, and the latter change no kernel. A table with a broken geometry (it fails a codegen check,
 * or the dispatch count exceeds 65535) throws {@link KernelsIdError}.
 */
export const geometryProfileKernelsId = (profile: GeometryProfile): string =>
  profileKernelsId(profile, SWEEP_CASES, casePlan);

/**
 * The fields of a sweep record row that the measured-candidate fingerprint
 * ({@link candidateKernelsId}) reads (a shape satisfied by both report.ts's `SweepRow` and the rows
 * the generator read).
 */
export type MeasuredRow = {
  readonly caseId: string;
  readonly geometryParams: GemmGeometry | I8a8Geometry;
  readonly error?: string;
};

/**
 * The body of the measured-candidate kernel fingerprint ({@link sweepCandidateKernelsId} — a form
 * into which the cases and the case plan can be injected).
 *
 * In the order of the record's rows, builds the (case, geometry) of every non-failed row (a row
 * without `error` — failed rows are no adoption material) from the case plan with the record's dp4a
 * variant, and hashes the same one-line JSON as {@link profileKernelsId} joined with newlines. The
 * dispatch-count limit is the same {@link FINGERPRINT_WORKGROUP_LIMIT} (kept independent of the
 * device).
 *
 * The input is **the record's rows** rather than "the current candidate set" because what is to be
 * matched is "the kernels that record measured": `quick+` candidates include the geometries of the
 * registered tables, so deriving from the current set would change the value as soon as one table
 * is registered and reject even records whose kernels did not change (including regeneration from
 * the record that produced that table). Derived from the rows, sweeps narrowed by ops / cases and
 * aborted sweeps can be matched as they are.
 *
 * MUST: dispatches are built with the same case plan as the sweep (the same reason as the MUST of
 * {@link profileKernelsId}). The writer (sweep.ts) and the reader (derive.ts) go through this
 * single function — built separately, the values would disagree even on the same runtime.
 */
export const candidateKernelsId = (
  rows: readonly MeasuredRow[],
  dp4a: boolean,
  cases: readonly SweepCase[],
  planner: CasePlanner,
): string => {
  const byId = new Map(cases.map((sweepCase) => [sweepCase.id, sweepCase]));
  const plans = new Map<string, CasePlan>();
  const lines: string[] = [];
  for (const row of rows) {
    if (row.error !== undefined) continue;
    const sweepCase = byId.get(row.caseId);
    if (sweepCase === undefined) {
      throw new KernelsIdError(
        `Case ${row.caseId} of a measured row is not in the current runtime's shape table`,
      );
    }
    const plan =
      plans.get(sweepCase.id) ??
      naming(
        () => `Cannot build the case plan of case ${sweepCase.id}`,
        () => planner(sweepCase, FINGERPRINT_WORKGROUP_LIMIT, dp4a),
      );
    plans.set(sweepCase.id, plan);
    // The family is decided by the geometry's shape (whether it has tileK) and the case's op (the
    // name only matters for naming in messages)
    const candidate =
      "tileK" in row.geometryParams
        ? i8a8Candidate(row.geometryParams)
        : sweepCase.op === "conv2d"
          ? conv2dCandidate(row.geometryParams)
          : gemmCandidate(row.geometryParams);
    const launch = naming(
      () =>
        `Case ${sweepCase.id}: cannot build the dispatch of measured geometry ${candidate.name} ${JSON.stringify(
          candidate.geometry,
        )}`,
      () => plan.launch(candidate),
    );
    lines.push(launchLine(launch));
  }
  return fnv1a64Hex(lines.join("\n"));
};

/**
 * The kernel fingerprint of the candidates the sweep measured (the record's `candidateKernels` —
 * ADR 0117 decision 8): the hash (16 hex digits) of the kernels that the (case, geometry) of the
 * record's non-failed rows build with the record's dp4a variant. Reads neither the GPU nor the
 * clock. The writer records it at the end of the sweep, and the generator re-derives it with the
 * current runtime and rejects the record on a difference — no table is built from old measurements
 * (speed ratios, output match) on a runtime where only the candidates' WGSL changed (the table's
 * `provenance.kernels` is derived with the current runtime at generation, so matching could not
 * detect it either).
 */
export const sweepCandidateKernelsId = (rows: readonly MeasuredRow[], dp4a: boolean): string =>
  candidateKernelsId(rows, dp4a, ALL_SWEEP_CASES, casePlan);

/**
 * The names of fields any case type has (the union of keyof over each member of the case-type
 * union).
 */
type SweepCaseField = SweepCase extends infer Case
  ? Case extends SweepCase
    ? keyof Case
    : never
  : never;

/**
 * Per case field, whether it goes into the case-set version (`true` = a field that decides the
 * dispatch). The included fields are those harness.ts's `casePlan` reads — `id` (the case's name,
 * which ties the sweep record's rows to the table's verdicts), `op`, the shape (`m` / `n` / `k` /
 * `batch`, attention's `stage` / `batchHeads` / `d`, conv2d's `channelsIn` / `channelsOut` /
 * `height` / `width`, conv3d's `dims` — every field of the kernel's dims) and the attrs that affect
 * the dispatch (attention's `scale` goes into params,
 * `score` into the WGSL as the storage format of S, the weight storage `weight` / `groupSize` of
 * the `k88` cases into the key and the WGSL). The excluded fields are information for humans
 * (`source` = description, `censusCount` = census count, `censusGroup` = what it counts,
 * `mirrorOf` = the mirrored case) — fixing a description does not make every saved table mismatch.
 *
 * MUST: when a field is added to a case type, decide here whether it goes in or out (`satisfies`
 * requires every field to be classified — a forgotten field is not silently left out of the
 * version).
 */
const CASE_SET_FIELDS = {
  id: true,
  op: true,
  m: true,
  n: true,
  k: true,
  batch: true,
  stage: true,
  batchHeads: true,
  d: true,
  scale: true,
  score: true,
  channelsIn: true,
  channelsOut: true,
  height: true,
  width: true,
  dims: true,
  weight: true,
  groupSize: true,
  censusCount: false,
  censusGroup: false,
  source: false,
  mirrorOf: false,
} as const satisfies Record<SweepCaseField, boolean>;

const isSweepCaseField = (field: string): field is SweepCaseField =>
  Object.hasOwn(CASE_SET_FIELDS, field);

/**
 * Only the fields of a case that go into the case-set version ({@link CASE_SET_FIELDS}). An
 * unclassified field throws.
 */
const caseSetFields = (sweepCase: SweepCase): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(sweepCase).filter(([field]) => {
      if (!isSweepCaseField(field)) {
        throw new Error(
          `Case ${sweepCase.id}: field ${field} has no classification (CASE_SET_FIELDS)`,
        );
      }
      return CASE_SET_FIELDS[field];
    }),
  );

/**
 * The body of the case-set version ({@link sweepCaseSetId} — a form into which the cases and the
 * boundaries can be injected).
 */
export const caseSetId = (cases: readonly SweepCase[], bounds: readonly number[]): string =>
  fnv1a64Hex(canonicalJson({ cases: cases.map(caseSetFields), bounds }));

/**
 * The case-set version (`provenance.caseSet` — ADR 0117 decision 4): the hash (16 hex digits) of
 * the canonical JSON of the dispatch-deciding fields ({@link CASE_SET_FIELDS}) of the sweep cases
 * (`SWEEP_CASES`) and the gemmRows bucket boundaries (`PROFILE_GEMM_ROWS_BOUNDS`).
 *
 * No hand-bumped version number (no way to add a case and forget the bump). When cases are added,
 * removed or reordered, or a dispatch-deciding field or a boundary changes, the value changes and
 * saved tables mismatch at matching (re-sweep). Changes only to informational fields such as
 * descriptions do not change it.
 */
export const sweepCaseSetId = (): string => caseSetId(SWEEP_CASES, PROFILE_GEMM_ROWS_BOUNDS);

/**
 * The case-set version a sweep of `set` records (`caseSet`): {@link sweepCaseSetId} for the
 * table-deriving sets, and the same hash over its own case table for an evaluation set
 * (`sweepCasesFor` in cases.ts), so that its record never claims the sweep's case set.
 */
export const candidateSetCaseSetId = (set: CandidateSet): string =>
  caseSetId(sweepCasesFor(set), PROFILE_GEMM_ROWS_BOUNDS);

/**
 * The fingerprint of the ratio baseline a sweep of `set` records (`defaultKernels`): the default
 * table over the case table of `set` ({@link geometryProfileKernelsId} of the default table for the
 * table-deriving sets).
 */
export const candidateSetDefaultKernelsId = (set: CandidateSet): string =>
  profileKernelsId(DEFAULT_GEOMETRY_PROFILE, sweepCasesFor(set), casePlan);

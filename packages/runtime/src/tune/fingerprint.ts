/**
 * The two matching keys, among those that decide whether a saved table may be applied, that are
 * derived from the runtime (DECIDED: ADR 0117 decision 4) — the **kernel fingerprint**
 * ({@link geometryProfileKernelsIds}) and the **case-set version** ({@link sweepCaseSetIds}) — one
 * value **per op family** (`GeometryProfileFamily`, src/tune/families.ts — ADR 0117 addendum
 * 2026-11-12), each taken over that family's sweep cases only, so adding one family's cases moves
 * only that family's keys. The online attention's key-axis form (`attentionOnline`) is decided from
 * attention-bench records, not sweep cases; its kernels have a key of their own
 * ({@link attentionBenchKernelsId}).
 *
 * All are synchronous pure functions that read neither the GPU nor the clock, and the same runtime
 * always gives the same value (codegen determinism — same key → byte-identical WGSL). The runtime's
 * version string is not used because the runtime has no version constant, a version would discard
 * tables even on releases that do not touch the GEMM kernels, and on unreleased checkouts it stays
 * the same while the kernels change. The fingerprint changes only when the kernels the table points
 * to or the default kernels change.
 *
 * The matching keys on the sweep record's side (ADR 0117 decision 8) are derived in the same style:
 * `defaultKernels` for the ratio baseline (the per-family fingerprint of the default table) and
 * `candidateKernels` for the measured candidates ({@link sweepCandidateKernelsId}).
 * `defaultKernels` protects only the ratio baseline; the speed and output match of an adopted
 * candidate come from the candidate's own kernels — a record without `candidateKernels` guarantees
 * only the default baseline.
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
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import {
  ATTENTION_ONLINE_REDUCES,
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineKeyWords,
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
  attentionOnlineKeyWords,
} from "../kernels/attention-online-i8a8.ts";
import { conv2dIgemmMTile } from "../kernels/conv2d.ts";
import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import type { I8a8Geometry } from "../kernels/i8a8-geometry.ts";
import {
  conv2dProfileGeometry,
  DEFAULT_GEOMETRY_PROFILE,
  gemmRowsGeometry,
  type GeometryProfile,
  type GeometryProfileFamily,
  i8a8LinearGeometry,
} from "../kernels/geometry-profile.ts";
import { ALL_SWEEP_CASES, SWEEP_CASES, type SweepCase, sweepCasesFor } from "./cases.ts";
import {
  familiesOf,
  familyBounds,
  familyEntries,
  type GeometryProfileFamilyIds,
  sweepOpFamily,
} from "./families.ts";
import {
  type CandidateSet,
  conv2dCandidate,
  gemmCandidate,
  type GeometryCandidate,
  i8a8Candidate,
} from "./geometries.ts";
import { type CasePlan, casePlan } from "./harness.ts";
import { ATTENTION_BENCH_SHAPES, type AttentionBenchShape } from "./attention-shapes.ts";
import {
  attentionValueQuantization,
  planAttentionOnlineWindows,
} from "../runtime/recipe-builders/attention.ts";

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
 * `gemmRowsGeometry(profile, m)` and `i8a8LinearGeometry(profile, m)` in
 * src/runtime/recipe-builders/linear.ts, `conv2dProfileGeometry` and `i8a8.conv3d` in conv.ts, the
 * fields in attention.ts). bmm's M is the row count of one matrix (as in buildBmm). The i8a8 linear
 * cases are i8, so the builder's w4a8 group fallback never applies to them.
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
      return {
        field: `i8a8.linear bucket for M ${sweepCase.m}`,
        candidate: i8a8Candidate(i8a8LinearGeometry(profile, sweepCase.m)),
      };
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

/** The JSON line of one launch built outside the case plan (the same fields as {@link launchLine}). */
const rawLaunchLine = (
  key: string,
  params: ArrayLike<number>,
  wgsl: string,
  workgroups: readonly [number, number, number],
): string => JSON.stringify([key, Array.from(params), wgslTokenText(wgsl), workgroups]);

/**
 * The lines of the online attention's kernel fingerprint ({@link attentionBenchKernelsId} — ADR 0117
 * addendum 2026-11-12): per shape of the bench's set, for the form N gives and the tiled form, the
 * Vᵀ producer and the kernel of the first window in each dp4a variant × reduction form (the layout
 * the bench times, `bhsd`) — the kernels the field `attentionOnline` chooses between, so a runtime
 * that changes either moves the key of a table that chose.
 * MUST: built with the production generator calls (`attentionValueQuantization`, the windows of
 * `planAttentionOnlineWindows`) the bench and the derivation's key check use.
 */
const attentionOnlineLines = (shapes: readonly AttentionBenchShape[]): string[] => {
  const lines: string[] = [];
  for (const { id, heads, queries, keys, depth, scale } of shapes) {
    const where = `Attention bench shape ${id}`;
    const windows = planAttentionOnlineWindows(queries, { batch: heads, cols: keys, depth });
    const forms: readonly AttentionOnlineKeyWords[] = [attentionOnlineKeyWords(keys), "tiled"];
    for (const keyWords of forms) {
      const producer = naming(
        () => `${where}, the ${keyWords} Vᵀ producer`,
        () => attentionValueQuantization(heads * depth, keys, keyWords),
      );
      lines.push(
        rawLaunchLine(producer.key, producer.params, producer.wgsl, [
          gridStrideWorkgroups(
            heads * depth,
            producer.geometry.rowsPerGroup,
            FINGERPRINT_WORKGROUP_LIMIT,
          ),
          1,
          1,
        ]),
      );
      for (const dp4a of [false, true]) {
        for (const reduce of ATTENTION_ONLINE_REDUCES) {
          const line = naming(
            () => `${where}, the ${keyWords} kernel (${reduce}, dp4a ${String(dp4a)})`,
            () =>
              rawLaunchLine(
                attentionOnlineI8a8Key(dp4a, depth, reduce, "bhsd", keyWords),
                attentionOnlineI8a8Params(queries, keys, 0, scale, keyWords),
                attentionOnlineI8a8Wgsl(dp4a, depth, reduce, "bhsd", keyWords),
                [
                  tiledWorkgroups(
                    windows[0].rows,
                    ATTENTION_ONLINE_ROWS,
                    FINGERPRINT_WORKGROUP_LIMIT,
                    where,
                  ),
                  tiledWorkgroups(heads, 1, FINGERPRINT_WORKGROUP_LIMIT, where),
                  1,
                ],
              ),
          );
          lines.push(line);
        }
      }
    }
  }
  return lines;
};

/**
 * The body of the kernel fingerprint over the cases given (one family's cases for the per-family
 * keys — {@link profileKernelsIds}; a form into which the cases and the case plan can be injected).
 *
 * For each case (in the order of `cases`) × dp4a variant (false → true), builds the dispatches of
 * the table's geometry and of the default geometry (the case plan's `defaultCandidate` = the
 * field's fallback — the value the generator compares with the sweep's default row), turns pipeline
 * key, params, the WGSL's token text ({@link wgslTokenText}) and workgroups into one line of JSON,
 * and hashes the lines joined with newlines (JSON contains no newline, so the join is unambiguous).
 *
 * MUST: dispatches are built with the same case plan as the sweep — the kernels measured by the
 * sweep and the kernels the fingerprint points to are never built by separate paths. `id`, `match`,
 * `attentionOnline` and `provenance` are not read (only the geometries of the cases' fields affect
 * the fingerprint — the online attention's kernels have their own key, {@link attentionBenchKernelsId}).
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
 * The kernel fingerprint per op family over `cases` ({@link profileKernelsId} over each family's
 * cases, for every family that has a case in `cases` — in the fixed family order).
 */
export const profileKernelsIds = (
  profile: GeometryProfile,
  cases: readonly SweepCase[],
  planner: CasePlanner,
): GeometryProfileFamilyIds => {
  const ids: { [Family in GeometryProfileFamily]?: string } = {};
  for (const family of familiesOf(cases)) {
    ids[family] = profileKernelsId(profile, familyEntries(cases, family), planner);
  }
  return ids;
};

/**
 * The table's kernel fingerprint of one op family (`provenance.families[family].kernels` — ADR 0117
 * decision 4, per family since its addendum 2026-11-12): the hash (16 hex digits) of the kernels
 * that the table's geometry and the default geometry build for the shapes of the family's sweep
 * cases. Reads neither the GPU nor the clock, and only the family's own fields.
 *
 * The value changes when a GEMM kernel (key, params, WGSL tokens, dispatch count) of the family or
 * its default geometry changes. Changes outside key, params, WGSL and workgroups (the order of
 * recipe bindings and the like) and changes to the WGSL's comments and whitespace do not show in the
 * fingerprint — the runtime's own checks (codegen snapshots, GPU tests with the default geometry)
 * cover the former, and the latter change no kernel. A table with a broken geometry in the family (it
 * fails a codegen check, or the dispatch count exceeds 65535) throws {@link KernelsIdError}.
 */
export const familyKernelsId = (profile: GeometryProfile, family: GeometryProfileFamily): string =>
  profileKernelsId(profile, familyEntries(SWEEP_CASES, family), casePlan);

/**
 * The table's kernel fingerprints of every op family ({@link familyKernelsId} for each — every family
 * has cases in `SWEEP_CASES`).
 */
export const geometryProfileKernelsIds = (profile: GeometryProfile): GeometryProfileFamilyIds =>
  profileKernelsIds(profile, SWEEP_CASES, casePlan);

/**
 * The fingerprint of the online attention's kernels (`provenance.attentionBench.kernels` — ADR 0117
 * addendum 2026-11-12): the hash (16 hex digits) of {@link attentionOnlineLines} over
 * `ATTENTION_BENCH_SHAPES`, the kernels and Vᵀ producers `attentionOnline` chooses between. It does
 * not depend on any table; a table carrying the field records it, and matching compares it. A
 * dispatch that cannot be built throws {@link KernelsIdError}.
 */
export const attentionBenchKernelsId = (): string =>
  fnv1a64Hex(attentionOnlineLines(ATTENTION_BENCH_SHAPES).join("\n"));

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
 * family fingerprints are derived with the current runtime at generation, so matching could not
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
 * The body of the case-set version over the cases given (one family's cases and that family's
 * bounds for the per-family keys — {@link caseSetIds}; a form into which the cases and the bounds
 * can be injected). Absent bounds leave the entry out of the hashed JSON.
 */
export const caseSetId = (cases: readonly SweepCase[], bounds?: readonly number[]): string =>
  fnv1a64Hex(canonicalJson({ cases: cases.map(caseSetFields), bounds }));

/**
 * The case-set version per op family over `cases`: for every family that has a case in `cases`
 * (in the fixed family order), {@link caseSetId} over the family's cases and its bounds
 * (`familyBounds` — the row-count bucket bounds for `gemm` only).
 */
export const caseSetIds = (cases: readonly SweepCase[]): GeometryProfileFamilyIds => {
  const ids: { [Family in GeometryProfileFamily]?: string } = {};
  for (const family of familiesOf(cases)) {
    ids[family] = caseSetId(familyEntries(cases, family), familyBounds(family));
  }
  return ids;
};

/**
 * The case-set version of one op family (`provenance.families[family].caseSet`): {@link caseSetId}
 * over the family's cases of `SWEEP_CASES` and the family's bounds.
 */
export const familyCaseSetId = (family: GeometryProfileFamily): string =>
  caseSetId(familyEntries(SWEEP_CASES, family), familyBounds(family));

/**
 * The case-set versions (`provenance.families[family].caseSet` — ADR 0117 decision 4, per op family
 * since its addendum 2026-11-12): per family, the hash (16 hex digits) of the canonical JSON of the
 * dispatch-deciding fields ({@link CASE_SET_FIELDS}) of the family's sweep cases (`SWEEP_CASES`)
 * and, for `gemm`, the gemmRows bucket boundaries (`PROFILE_GEMM_ROWS_BOUNDS`). Given a candidate
 * set, the versions a sweep of every case of that set records ({@link candidateSetCaseSetIds} — the
 * same values for the table-deriving sets, the families of its own table for an evaluation set).
 *
 * No hand-bumped version number (no way to add a case and forget the bump). When a family's cases
 * are added, removed or reordered, or a dispatch-deciding field or a bound of the family changes,
 * that family's value changes and the saved tables' entries of that family mismatch at matching
 * (re-sweep of that family). Changes only to informational fields such as descriptions do not change
 * it.
 */
export const sweepCaseSetIds = (set?: CandidateSet): GeometryProfileFamilyIds =>
  set === undefined ? caseSetIds(SWEEP_CASES) : candidateSetCaseSetIds(set);

/**
 * The kernel fingerprints of the default table a sweep of every case of `set` records as its ratio
 * baseline (`defaultKernels` — ADR 0117 decision 8, per op family since its addendum 2026-11-12): per
 * family of the set's case table, the default table's fingerprint over the family's cases
 * ({@link candidateSetDefaultKernelsIds}; without `set`, over `SWEEP_CASES`).
 */
export const sweepDefaultKernelsIds = (set?: CandidateSet): GeometryProfileFamilyIds =>
  set === undefined
    ? geometryProfileKernelsIds(DEFAULT_GEOMETRY_PROFILE)
    : candidateSetDefaultKernelsIds(set);

/**
 * The id of the attention bench's shape set (`provenance.attentionBench.shapes` — ADR 0117
 * addendum 2026-11-12): the hash (16 hex digits) of the canonical JSON of `ATTENTION_BENCH_SHAPES`
 * (every field — id, heads, queries, keys, depth and scale decide what was measured). Records given
 * to the derivation must cover this set, and matching compares a table's value with it.
 */
export const attentionBenchShapeSetId = (): string =>
  fnv1a64Hex(canonicalJson(ATTENTION_BENCH_SHAPES));

/**
 * The case table of `set` (`sweepCasesFor` in cases.ts) restricted to the families of `families`
 * (the whole table when absent).
 */
const setCases = (
  set: CandidateSet,
  families: readonly GeometryProfileFamily[] | undefined,
): readonly SweepCase[] => {
  const cases = sweepCasesFor(set);
  return families === undefined
    ? cases
    : cases.filter((sweepCase) => families.includes(sweepOpFamily(sweepCase.op)));
};

/**
 * The case-set versions a sweep of `set` records (`caseSets`) for the families of `families`
 * (default: every family of the set's case table) that the set's case table has: each hashed over
 * the set's own case table restricted to the family — {@link sweepCaseSetIds} for the
 * table-deriving sets, and an evaluation set's own cases otherwise, so that its record never claims
 * the sweep's case set.
 */
export const candidateSetCaseSetIds = (
  set: CandidateSet,
  families?: readonly GeometryProfileFamily[],
): GeometryProfileFamilyIds => caseSetIds(setCases(set, families));

/**
 * The fingerprints of the ratio baseline a sweep of `set` records (`defaultKernels`) for the
 * families of `families` (default: every family of the set's case table) that the set's case table
 * has: the default table over the set's case table restricted to each family
 * ({@link sweepDefaultKernelsIds} for the table-deriving sets).
 */
export const candidateSetDefaultKernelsIds = (
  set: CandidateSet,
  families?: readonly GeometryProfileFamily[],
): GeometryProfileFamilyIds =>
  profileKernelsIds(DEFAULT_GEOMETRY_PROFILE, setCases(set, families), casePlan);

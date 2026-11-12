/**
 * The `unknown` boundary that reads the JSON of a saved table (DECIDED: ADR 0117 decision 7). The
 * writer is `geometryProfileJson` (derive.ts — it writes the trailing `maxRows` Infinity as
 * `1e999`).
 *
 * Unknown fields are rejected at every level. A table is a value that is injected and affects
 * execution; silently reading a table of a different shape version lets a misreading of a field's
 * meaning through — the right move for the application is to discard an unreadable table and sweep
 * again (ADR 0117 decision 10). The policy is the opposite of the sweep record's (an append-only
 * log — a reader uses only the fields it reads).
 */
import { CodegenError } from "../codegen/errors.ts";
import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import {
  assertGeometryProfile,
  GEOMETRY_PROFILE_FAMILIES,
  type GemmRowsRule,
  type GeometryProfile,
  type GeometryProfileFamily,
  type GeometryProfileFamilyProvenance,
} from "../kernels/geometry-profile.ts";
import type { I8a8Geometry } from "../kernels/i8a8-geometry.ts";

/**
 * The JSON of a saved table cannot be read (the message names the field's path —
 * {@link parseGeometryProfileJson}).
 */
export class GeometryProfileParseError extends Error {
  override readonly name = "GeometryProfileParseError";
}

type Match = NonNullable<GeometryProfile["match"]>;
type Provenance = NonNullable<GeometryProfile["provenance"]>;

/** The path spelling of the table itself (the top level). */
const ROOT = "top level";

const fail = (path: string, reason: string): never => {
  throw new GeometryProfileParseError(`Geometry profile JSON: ${path} ${reason}`);
};

const show = (value: unknown): string => {
  if (value === undefined) return "missing";
  // JSON.stringify spells Infinity as null, so numbers are printed with String
  return typeof value === "number" ? String(value) : JSON.stringify(value);
};

const child = (path: string, key: string): string => (path === ROOT ? key : `${path}.${key}`);

/**
 * Checks that the value at `path` is an object, its fields fit in `required` ∪ `optional`, and
 * every `required` field is present.
 */
const readRecord = (
  value: unknown,
  path: string,
  required: readonly string[],
  optional: readonly string[] = [],
): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return fail(path, `is not an object (${show(value)})`);
  }
  const fields: Record<string, unknown> = { ...value };
  const unknown = Object.keys(fields).filter(
    (key) => !required.includes(key) && !optional.includes(key),
  );
  if (unknown.length > 0) {
    return fail(
      path,
      `has unknown fields ${unknown.join(", ")} (a table of another format version is not read — sweep again to make it)`,
    );
  }
  const missing = required.filter((key) => fields[key] === undefined);
  if (missing.length > 0) return fail(path, `is missing fields ${missing.join(", ")}`);
  return fields;
};

const readString = (value: unknown, path: string): string =>
  typeof value === "string" ? value : fail(path, `is not a string (${show(value)})`);

/**
 * A family entry's `userAgent` (an array of strings — order of appearance; deduplication is the writer's
 * job). An empty array is a form the writer never writes, so it is rejected.
 */
const readUserAgents = (value: unknown, path: string): string[] => {
  if (!Array.isArray(value)) return fail(path, `is not an array (${show(value)})`);
  const agents: readonly unknown[] = value;
  if (agents.length === 0)
    return fail(path, "is an empty array (a table without userAgent omits the field entirely)");
  return agents.map((agent, index) => readString(agent, `${path}[${index}]`));
};

const readInteger = (value: unknown, path: string): number =>
  typeof value === "number" && Number.isSafeInteger(value)
    ? value
    : fail(path, `is not an integer (${show(value)})`);

const GEMM_FIELDS = ["regM", "regN", "wgX", "wgY"] as const;

/** The optional skeleton form of an f32 GEMM geometry (perf-ledger K-88). */
const readForm = (value: unknown, path: string): NonNullable<GemmGeometry["form"]> =>
  value === "tb-un" || value === "tas-tb-un"
    ? value
    : fail(path, `is not "tb-un" or "tas-tb-un" (${show(value)})`);

/** The optional im2col decode of a conv2d geometry (perf-ledger K-88 — `"counter"` only). */
const readDecode = (value: unknown, path: string): NonNullable<GemmGeometry["decode"]> =>
  value === "counter" ? value : fail(path, `is not "counter" (${show(value)})`);

/**
 * An f32 GEMM geometry. `optional` names the K-88 fields the slot may carry (the row-count buckets
 * the skeleton form, conv2d the form and the decode, the fused attention none — anywhere else they
 * are unknown fields).
 */
const readGemmGeometry = (
  value: unknown,
  path: string,
  optional: readonly ("form" | "decode")[] = [],
): GemmGeometry => {
  const fields = readRecord(value, path, GEMM_FIELDS, optional);
  return {
    regM: readInteger(fields.regM, child(path, "regM")),
    regN: readInteger(fields.regN, child(path, "regN")),
    wgX: readInteger(fields.wgX, child(path, "wgX")),
    wgY: readInteger(fields.wgY, child(path, "wgY")),
    ...(fields.form === undefined ? {} : { form: readForm(fields.form, child(path, "form")) }),
    ...(fields.decode === undefined
      ? {}
      : { decode: readDecode(fields.decode, child(path, "decode")) }),
  };
};

/** The optional skeleton form of an i8a8 geometry (perf-ledger K-85 — `"vec4"` is the only value). */
const readSkeleton = (value: unknown, path: string): NonNullable<I8a8Geometry["skeleton"]> =>
  value === "vec4" ? value : fail(path, `is not "vec4" (${show(value)})`);

const readI8a8Geometry = (value: unknown, path: string): I8a8Geometry => {
  const fields = readRecord(value, path, [...GEMM_FIELDS, "tileK"], ["skeleton"]);
  return {
    regM: readInteger(fields.regM, child(path, "regM")),
    regN: readInteger(fields.regN, child(path, "regN")),
    wgX: readInteger(fields.wgX, child(path, "wgX")),
    wgY: readInteger(fields.wgY, child(path, "wgY")),
    tileK: readInteger(fields.tileK, child(path, "tileK")),
    ...(fields.skeleton === undefined
      ? {}
      : { skeleton: readSkeleton(fields.skeleton, child(path, "skeleton")) }),
  };
};

/**
 * `match` (all three fields optional — the combination rules are {@link assertGeometryProfile}'s
 * job).
 */
const readMatch = (value: unknown, path: string): Match => {
  const fields = readRecord(value, path, [], ["vendor", "architecture", "description"]);
  const optional = (key: keyof Match): string | undefined =>
    fields[key] === undefined ? undefined : readString(fields[key], child(path, key));
  const vendor = optional("vendor");
  const architecture = optional("architecture");
  const description = optional("description");
  return {
    ...(vendor === undefined ? {} : { vendor }),
    ...(architecture === undefined ? {} : { architecture }),
    ...(description === undefined ? {} : { description }),
  };
};

/**
 * `gemmRows`. The trailing `maxRows` is Infinity (`1e999` in JSON — `JSON.parse` reads it as
 * Infinity). The range and ascending order of the other entries are {@link assertGeometryProfile}'s
 * job.
 */
const readGemmRows = (value: unknown, path: string): GemmRowsRule[] => {
  if (!Array.isArray(value)) return fail(path, `is not an array (${show(value)})`);
  const rules: readonly unknown[] = value;
  return rules.map((entry, index) => {
    const where = `${path}[${index}]`;
    const fields = readRecord(entry, where, ["maxRows", "geometry"]);
    const maxRows = fields.maxRows;
    const last = index === rules.length - 1;
    if (last && maxRows === null) {
      return fail(
        child(where, "maxRows"),
        "is null (a lost Infinity — plain JSON.stringify turns Infinity into null; write the table " +
          "with geometryProfileJson)",
      );
    }
    if (typeof maxRows !== "number") {
      return fail(child(where, "maxRows"), `is not a number (${show(maxRows)})`);
    }
    if (last && maxRows !== Number.POSITIVE_INFINITY) {
      return fail(
        child(where, "maxRows"),
        `is not Infinity (${maxRows} — the last rule is 1e999, which covers every row count)`,
      );
    }
    return {
      maxRows,
      geometry: readGemmGeometry(fields.geometry, child(where, "geometry"), ["form"]),
    };
  });
};

/** `attentionOnline` (`{ keyWords: "tiled" }` — the only form a table chooses). */
const readAttentionOnline = (
  value: unknown,
  path: string,
): NonNullable<GeometryProfile["attentionOnline"]> => {
  const fields = readRecord(value, path, ["keyWords"]);
  return fields.keyWords === "tiled"
    ? { keyWords: "tiled" }
    : fail(child(path, "keyWords"), `is not "tiled" (${show(fields.keyWords)})`);
};

const readProvenance = (value: unknown, path: string): Provenance => {
  const fields = readRecord(value, path, ["adapter", "families"], ["attentionBench"]);
  const adapterPath = child(path, "adapter");
  const adapter = readRecord(fields.adapter, adapterPath, [
    "vendor",
    "architecture",
    "device",
    "description",
  ]);
  return {
    adapter: {
      vendor: readString(adapter.vendor, child(adapterPath, "vendor")),
      architecture: readString(adapter.architecture, child(adapterPath, "architecture")),
      device: readString(adapter.device, child(adapterPath, "device")),
      description: readString(adapter.description, child(adapterPath, "description")),
    },
    families: readFamilies(fields.families, child(path, "families")),
    ...(fields.attentionBench === undefined
      ? {}
      : {
          attentionBench: readAttentionBench(fields.attentionBench, child(path, "attentionBench")),
        }),
  };
};

/**
 * `provenance.families` (one entry per op family that has one — ADR 0117 addendum 2026-11-12; a key
 * that is not a family is an unknown field).
 */
const readFamilies = (value: unknown, path: string): Provenance["families"] => {
  const fields = readRecord(value, path, [], GEOMETRY_PROFILE_FAMILIES);
  const families: { [Family in GeometryProfileFamily]?: GeometryProfileFamilyProvenance } = {};
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    if (fields[family] !== undefined) {
      families[family] = readFamilyProvenance(fields[family], child(path, family));
    }
  }
  return families;
};

/** One family's provenance (six strings and the optional `userAgent`). */
const readFamilyProvenance = (value: unknown, path: string): GeometryProfileFamilyProvenance => {
  // userAgent is an optional reference field (not used for matching — ADR 0117 addendum
  // 2026-10-02). Unknown fields are still rejected
  const fields = readRecord(
    value,
    path,
    ["sweep", "sha256", "date", "candidateSet", "kernels", "caseSet"],
    ["userAgent"],
  );
  return {
    sweep: readString(fields.sweep, child(path, "sweep")),
    sha256: readString(fields.sha256, child(path, "sha256")),
    date: readString(fields.date, child(path, "date")),
    candidateSet: readString(fields.candidateSet, child(path, "candidateSet")),
    ...(fields.userAgent === undefined
      ? {}
      : { userAgent: readUserAgents(fields.userAgent, child(path, "userAgent")) }),
    kernels: readString(fields.kernels, child(path, "kernels")),
    caseSet: readString(fields.caseSet, child(path, "caseSet")),
  };
};

/**
 * `provenance.attentionBench` (four strings and the optional `kernels` — ADR 0117 addenda
 * 2026-11-12).
 */
const readAttentionBench = (
  value: unknown,
  path: string,
): NonNullable<Provenance["attentionBench"]> => {
  const fields = readRecord(value, path, ["records", "sha256", "date", "shapes"], ["kernels"]);
  return {
    records: readString(fields.records, child(path, "records")),
    sha256: readString(fields.sha256, child(path, "sha256")),
    date: readString(fields.date, child(path, "date")),
    shapes: readString(fields.shapes, child(path, "shapes")),
    ...(fields.kernels === undefined
      ? {}
      : { kernels: readString(fields.kernels, child(path, "kernels")) }),
  };
};

/**
 * Reads the JSON of a saved table (the string from `geometryProfileJson`) and turns it back into a
 * table that can be passed to `acquireGpu({ geometryProfile })`. Failures are
 * {@link GeometryProfileParseError} (the message names the field's path).
 *
 * Checks: it parses as JSON; the types of all fields (the `id` and `provenance` fields are strings,
 * geometries have integer fields only); unknown fields are rejected at every level; the trailing
 * `maxRows` of `gemmRows` is Infinity (the restoration of `1e999` — a trailing `null` is named as
 * a lost Infinity); finally `assertGeometryProfile`, the same check the runtime runs at injection.
 */
export const parseGeometryProfileJson = (text: string): GeometryProfile => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (cause) {
    throw new GeometryProfileParseError(
      `Geometry profile JSON: cannot be parsed as JSON (${
        cause instanceof Error ? cause.message : String(cause)
      })`,
      { cause },
    );
  }
  const fields = readRecord(
    parsed,
    ROOT,
    ["id", "gemmRows", "attention", "conv2d", "i8a8"],
    ["match", "attentionOnline", "provenance"],
  );
  const attention = readRecord(fields.attention, "attention", ["qk", "pv"]);
  const conv2d = readRecord(fields.conv2d, "conv2d", ["rows64", "rows32"]);
  const i8a8 = readRecord(fields.i8a8, "i8a8", ["linear", "attentionQk", "attentionPv", "conv3d"]);
  const profile: GeometryProfile = {
    id: readString(fields.id, "id"),
    ...(fields.match === undefined ? {} : { match: readMatch(fields.match, "match") }),
    gemmRows: readGemmRows(fields.gemmRows, "gemmRows"),
    attention: {
      qk: readGemmGeometry(attention.qk, "attention.qk"),
      pv: readGemmGeometry(attention.pv, "attention.pv"),
    },
    conv2d: {
      rows64: readGemmGeometry(conv2d.rows64, "conv2d.rows64", ["form", "decode"]),
      rows32: readGemmGeometry(conv2d.rows32, "conv2d.rows32", ["form", "decode"]),
    },
    i8a8: {
      linear: readI8a8Geometry(i8a8.linear, "i8a8.linear"),
      attentionQk: readI8a8Geometry(i8a8.attentionQk, "i8a8.attentionQk"),
      attentionPv: readI8a8Geometry(i8a8.attentionPv, "i8a8.attentionPv"),
      conv3d: readI8a8Geometry(i8a8.conv3d, "i8a8.conv3d"),
    },
    ...(fields.attentionOnline === undefined
      ? {}
      : { attentionOnline: readAttentionOnline(fields.attentionOnline, "attentionOnline") }),
    ...(fields.provenance === undefined
      ? {}
      : { provenance: readProvenance(fields.provenance, "provenance") }),
  };
  try {
    // The same single check the runtime runs at injection (ascending rule lists, the divisibility
    // of geometries, the combination of match)
    assertGeometryProfile(profile);
  } catch (cause) {
    if (cause instanceof CodegenError) {
      throw new GeometryProfileParseError(`Geometry profile JSON: ${cause.message}`, { cause });
    }
    throw cause;
  }
  return profile;
};

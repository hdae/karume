/**
 * GEMM geometry **profiles** (static tables of shape × adapter — DECIDED: ADR 0115).
 *
 * The best geometry (which thread handles which output) reverses between GPUs — a geometry 1.4–1.7×
 * faster on Apple M2 becomes 0.72–0.98× on Arc B570 (docs/research/2026-09-27-k70-metal-per-op.md
 * §9). With one common default one of them loses, so the selection is widened from "a pure function
 * of the shape" to "a pure function of shape × profile", and one profile is chosen in a
 * deterministic order from the adapter's (vendor, architecture, description) (a table without
 * `match` is not auto-selected and is used only by injection).
 *
 * MUST: a profile is **plain data** (no side effects, nothing read from outside at run time). The
 * results measured by explicit tuning (sweeps of tools/geometry-sweep (removed in 0.15); the sweep
 * now runs through `@karume/runtime/tune`, ADR 0117) are embedded in the source as generated files
 * (`./geometry-profiles/`). Run-time autotuning (measuring and re-choosing during execution) stays
 * banned (the MUST of the ADR 0022 addendum) — on the same machine and the same shape, the
 * geometry, key and WGSL are the same every time.
 * MUST: the geometry may change only the work assignment; the numeric contract (K reduction order,
 * literal form of the multiply-add) does not move (src/kernels/gemm-geometry.ts /
 * src/kernels/i8a8-geometry.ts). The geometry goes into the pipeline key, so a different profile
 * means a different key, and the device-lifetime pipeline cache cannot mix them up.
 *
 * The scope (the paths that read the profile) is only the row-count buckets of matmul / bmm /
 * linear, ①QK / ③PV of fused attention f32, the implicit GEMM of conv2d, the i8a8 linear / fused
 * attention ①QK / ③PV, and the w8a8 implicit GEMM of conv3d (one geometry for every conv3d shape —
 * ADR 0115 addendum 2026-10-26). The row-count bucket table is shared by the 3 paths matmul / bmm /
 * linear (the same GEMM skeleton — the same meaning of geometry and the same K reduction order; only
 * the B-side fill differs). The sweep has cases for all 3 paths, and the generator adopts only
 * geometries that are faster on all of them (the `profile` command of tools/geometry-sweep (removed
 * in 0.15); now `deriveGeometryProfile` of `@karume/runtime/tune`). States-shaped attention (①ₜ /
 * ③ₜ), the row blocks of decomposed attention and the implicit GEMM of conv1d keep the default
 * selection (outside the sweep — a path that was not measured is not moved by a table).
 * One field is a kernel variant rather than a geometry: the key-axis form of the online attention's
 * quantised Vᵀ (`attentionOnline`), read through the attention route only.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_ROWS_BUCKETS,
  GEMM_TILE,
  type GemmGeometry,
} from "./gemm-geometry.ts";
import { GEMM_MTILE_SMALL, gemmMTileGeometry, gemmWorkgroupStorageBytes } from "./gemm.ts";
import {
  assertI8a8Geometry,
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8WorkgroupStorageBytes,
} from "./i8a8-geometry.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "./geometry-profiles/index.ts";

/**
 * One row-count bucket rule. The first rule with `rows <= maxRows` is used ({@link gemmRowsGeometry}).
 * The rules are in strictly ascending `maxRows`, and the last is `Number.POSITIVE_INFINITY` (every
 * row count hits some rule).
 */
export type GemmRowsRule = { readonly maxRows: number; readonly geometry: GemmGeometry };

/**
 * The op families of the geometry sweep: the unit a table's evidence stands on, and the unit its
 * matching keys are taken per (DECIDED: ADR 0117 addendum 2026-11-12, "per op family"). Each family
 * owns the profile fields its sweep cases decide — `gemm` the row-count buckets `gemmRows` (sweep ops
 * linear / matmul / bmm), `attention` `attention.qk` / `attention.pv`, `conv2d` `conv2d.rows64` /
 * `conv2d.rows32`, `i8a8-linear` `i8a8.linear`, `i8a8-attention` `i8a8.attentionQk` /
 * `i8a8.attentionPv`, `i8a8-conv3d` `i8a8.conv3d`. The maps from sweep op and from profile field to
 * family live in `src/tune/families.ts` (kernels do not import tune). The union is derived from
 * {@link GEOMETRY_PROFILE_FAMILIES}, so a family cannot be added to the type without also entering
 * the list every per-family loop walks.
 */
export type GeometryProfileFamily = (typeof GEOMETRY_PROFILE_FAMILIES)[number];

/**
 * Every family in the fixed order — the order of the fields they decide, the order matching checks
 * them in and the order the generator writes `provenance.families` in. The source of
 * {@link GeometryProfileFamily}.
 *
 * NOTE: `./geometry-profiles/index.ts` must not read it: this module imports that one as a value,
 * and that one builds its frozen list at evaluation, so reading this constant there would depend on
 * the import order.
 */
export const GEOMETRY_PROFILE_FAMILIES = [
  "gemm",
  "attention",
  "conv2d",
  "i8a8-linear",
  "i8a8-attention",
  "i8a8-conv3d",
] as const;

/**
 * The provenance of one op family of a table generated from sweeps (DECIDED: ADR 0117 decision 4 as
 * amended by its addendum 2026-11-12): the sweep records that have a row of the family, and the two
 * matching keys taken over the family's cases only.
 */
export type GeometryProfileFamilyProvenance = {
  /**
   * Paths of the sweep records that have a row of this family (joined with `", "` in the order
   * given to the generator).
   */
  readonly sweep: string;
  /** SHA-256 (hex) of the bytes of the same records. */
  readonly sha256: string;
  /** Dates of the same sweeps. */
  readonly date: string;
  /** Candidate sets of the same sweeps (`quick` / `quick+` / `full`). */
  readonly candidateSet: string;
  /**
   * The browsers that ran the same sweeps (the records' `userAgent` — records from the Deno CLI
   * carry `Deno/<version>`), deduplicated in order of appearance (one element when all are the
   * same). They are not joined because a browser's userAgent itself contains `", "`
   * (`(KHTML, like Gecko)`) and joined values could not be split apart. A reference field not used
   * for matching (`geometryProfileMismatch`); material for an application to compare with the
   * current browser and suggest a re-sweep (DECIDED: ADR 0117 addendum 2026-10-02).
   * Not written when a record without `userAgent` is among the same records (an empty array is not
   * written either).
   */
  readonly userAgent?: readonly string[];
  /**
   * Kernel fingerprint over this family's sweep cases (hash of the kernels the table and the
   * default geometry build on those shapes, the WGSL counted without comments and whitespace — hex).
   */
  readonly kernels: string;
  /**
   * Case-set version of this family (hash of its sweep cases and, for `gemm`, the row-count bucket
   * bounds — hex).
   */
  readonly caseSet: string;
};

/**
 * A GEMM geometry profile = a static table of tile geometries per op × bucket (DECIDED: ADR 0115).
 * The embedded tables (`./geometry-profiles/`) are chosen by the adapter's (vendor, architecture,
 * description), and users can inject, with `acquireGpu({ geometryProfile })`, a table made from
 * their own sweep or an embedded table (`BUILTIN_GEOMETRY_PROFILES` on the public surface — ADR 0115
 * addendum decision 7).
 */
export type GeometryProfile = {
  /**
   * Name of the profile (`"default"` / `"apple-metal-3"` and so on — matches the generated file
   * name).
   */
  readonly id: string;
  /**
   * Which adapter it applies to (exact string match with `vendor` / `architecture` / `description`
   * of `GPUAdapterInfo`). All fields undefined (`{}`) is the form of the default profile
   * ({@link DEFAULT_GEOMETRY_PROFILE}) only.
   *
   * Omitted = not auto-selected (injection only = opt-in — ADR 0115 addendum decision 7). A table
   * that must not be applied silently to another machine reporting the same vendor / architecture
   * (in Chrome, Apple M2 and M5 both report `apple` / `metal-3`) is embedded without `match`, and an
   * application looks it up by id and injects it.
   */
  readonly match?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly description?: string;
  };
  /** linear / matmul / bmm (row-count buckets). */
  readonly gemmRows: readonly GemmRowsRule[];
  /**
   * ①QK / ③PV of fused attention f32. S is materialised between the two stages, so each stage may
   * have its own geometry.
   */
  readonly attention: { readonly qk: GemmGeometry; readonly pv: GemmGeometry };
  /**
   * The implicit GEMM of conv2d. A geometry per m-tile **class** (64 rows / 32 rows as returned by
   * `conv2dIgemmMTile`); the class is decided by that predicate as before. The canonical source of
   * the actual tile edge is the geometry, so a geometry with tileM = 64 may be assigned to the
   * 32-row class (dispatch is derived from the geometry).
   */
  readonly conv2d: { readonly rows64: GemmGeometry; readonly rows32: GemmGeometry };
  readonly i8a8: {
    readonly linear: I8a8Geometry;
    readonly attentionQk: I8a8Geometry;
    readonly attentionPv: I8a8Geometry;
    /**
     * The w8a8 implicit GEMM of conv3d (`convCompute: "a8"` — ADR 0133). One geometry for every
     * conv3d shape (no shape buckets — ADR 0115 addendum 2026-10-26).
     */
    readonly conv3d: I8a8Geometry;
  };
  /**
   * The key-axis form of the online int8 attention's quantised Vᵀ (perf-ledger K-106 — ADR 0132
   * addendum 2026-11-12), a kernel-variant choice rather than a geometry. Absent (every table but one
   * measured to gain from it) = the form `attentionOnlineKeyWords` gives N (`whole` at `N % 4 == 0`,
   * `padded` otherwise), so every key, WGSL and estimate of a table without the field is unchanged.
   * `{ keyWords: "tiled" }` = the producer writes Vᵀ in the kernel's key order at every N. Both forms
   * are bit-identical by measured behaviour (`checkAttentionOnlineTiled` of packages/kernel-checks).
   *
   * MUST: read only through the attention route (`attentionDeviceFacts` → `attentionRoute`), the one
   * decision point the producer, the kernel, its uniform and the size of the quantised Vᵀ share.
   */
  readonly attentionOnline?: { readonly keyWords: "tiled" };
  /**
   * Provenance and matching material held only by profiles generated from a sweep (DECIDED: ADR 0117
   * decision 4, per op family since its addendum 2026-11-12). Only the generator
   * (`deriveGeometryProfile` of `@karume/runtime/tune`) writes it, and `geometryProfileMismatch` of
   * `./tune` does the matching. The runtime (`assertGeometryProfile`, the injection entry point of
   * `acquireGpu`) checks only the shape (the field types) and does not use it for matching.
   */
  readonly provenance?: {
    /**
     * The swept adapter (the 4 fields of `GPUAdapterInfo` — an empty string stays a value;
     * identical across all records).
     */
    readonly adapter: {
      readonly vendor: string;
      readonly architecture: string;
      readonly device: string;
      readonly description: string;
    };
    /**
     * Per op family ({@link GeometryProfileFamily}), the records and matching keys of that family.
     * A family that no record has a case of has no entry: its fields are the default (matching
     * checks that they are).
     */
    readonly families: {
      readonly [Family in GeometryProfileFamily]?: GeometryProfileFamilyProvenance;
    };
    /**
     * The attention-bench records the derivation read for {@link GeometryProfile.attentionOnline}
     * (written whenever records were given, whether or not they adopted the field — ADR 0117
     * addendum 2026-11-12): their paths, SHA-256 and dates joined with `", "` in the order given,
     * the id of the bench's shape set they covered (the `attentionBenchShapeSetId` of the runtime
     * that derived the table — `geometryProfileMismatch` compares it) and, exactly when the table
     * carries `attentionOnline`, the fingerprint of the online attention's kernels the field chooses
     * between (`kernels` — compared by matching as well).
     */
    readonly attentionBench?: {
      readonly records: string;
      readonly sha256: string;
      readonly date: string;
      readonly shapes: string;
      readonly kernels?: string;
    };
  };
};

/**
 * The default profile (the table used by an adapter that no generated table matches).
 *
 * MUST: the values are built **by referencing** the existing selection functions (not copied —
 * copying would put the default in 2 places, and when only one is fixed, the Session path and the
 * shape-pure-function path would choose different geometries). On a machine where this table is
 * chosen, key, WGSL and dispatch do not differ by a single byte from before profiles were introduced
 * (the codegen snapshots and tests/geometry_profile_test.ts are the detectors).
 */
export const DEFAULT_GEOMETRY_PROFILE: GeometryProfile = {
  id: "default",
  match: {},
  gemmRows: GEMM_ROWS_BUCKETS,
  attention: { qk: defaultGemmGeometry(), pv: defaultGemmGeometry() },
  conv2d: {
    rows64: gemmMTileGeometry(GEMM_TILE),
    rows32: gemmMTileGeometry(GEMM_MTILE_SMALL),
  },
  i8a8: {
    linear: defaultI8a8Geometry("linear"),
    attentionQk: defaultI8a8Geometry("attention_qk"),
    attentionPv: defaultI8a8Geometry("attention_pv"),
    conv3d: defaultI8a8Geometry("conv3d"),
  },
};

/**
 * Guard on the row-count bucket rules. **Rules not in ascending order leave later rules dead, and
 * rules whose last entry is not Infinity leave no rule for large M** — both silently turn the
 * selection into something else, so fail loudly.
 */
const assertGemmRowsRules = (rules: readonly GemmRowsRule[], where: string): void => {
  if (rules.length === 0) {
    throw new CodegenError(`${where}: gemmRows is empty (no geometry matches any row count)`);
  }
  rules.forEach((rule, index) => {
    const last = index === rules.length - 1;
    if (last) {
      if (rule.maxRows !== Number.POSITIVE_INFINITY) {
        throw new CodegenError(
          `${where}: the last maxRows of gemmRows must be Infinity (${rule.maxRows} — no rule covers a larger M)`,
        );
      }
    } else {
      assertU32Params(`${where}: gemmRows[${index}]`, { maxRows: rule.maxRows });
      const next = rules[index + 1].maxRows;
      if (!(rule.maxRows < next)) {
        throw new CodegenError(
          `${where}: maxRows of gemmRows is not strictly increasing ([${index}] ${rule.maxRows} → [${
            index + 1
          }] ${next} — no row count is left for the later rule)`,
        );
      }
    }
    assertGemmGeometry(rule.geometry, `${where}: gemmRows[${index}]`);
  });
};

/**
 * Guard that a value is an object whose fields can be read (rejects arrays, null and primitives
 * with the field path).
 */
const recordOf = (value: unknown, where: string, path: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new CodegenError(`${where}: ${path} is not an object`);
  }
  return { ...value };
};

const stringOf = (value: unknown, where: string, path: string): string => {
  if (typeof value === "string") return value;
  throw new CodegenError(
    `${where}: ${path} is not a string (${value === undefined ? "missing" : typeof value})`,
  );
};

const optionalStringOf = (value: unknown, where: string, path: string): string | undefined =>
  value === undefined ? undefined : stringOf(value, where, path);

/**
 * All elements of an array field (rejects holes with the field path). `map` / `forEach` skip the
 * holes of a sparse array, so a hole would pass the guard and become a TypeError outside it (in the
 * selection's `find` or an index read) — a table built by a JS caller (`new Array(n)` and so on) is
 * rejected as a missing field with the guard's {@link CodegenError}.
 */
const denseEntriesOf = (
  values: readonly unknown[],
  where: string,
  path: string,
): readonly unknown[] => {
  const entries: unknown[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (!(index in values)) {
      throw new CodegenError(`${where}: ${path}[${index}] is missing (a hole in a sparse array)`);
    }
    entries.push(values[index]);
  }
  return entries;
};

/**
 * Guard that a geometry field is a number. The range (positive integer) and the divisibility
 * conditions belong to {@link assertGemmGeometry} / {@link assertI8a8Geometry} (a non-number is
 * rejected with the same wording).
 */
const numberOf = (value: unknown, where: string, path: string, name: string): number => {
  if (typeof value === "number") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry ${name} must be a positive integer (${String(value)})`,
  );
};

/**
 * Guard on the optional skeleton form of an f32 GEMM geometry (perf-ledger K-88 — the same
 * wording as {@link assertGemmGeometry}, which checks it again at generation).
 */
const formOf = (value: unknown, where: string, path: string): NonNullable<GemmGeometry["form"]> => {
  if (value === "tb-un" || value === "tas-tb-un") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry form must be "tb-un", "tas-tb-un" or absent (${JSON.stringify(value)})`,
  );
};

/** Guard on the optional im2col decode of a conv2d geometry (perf-ledger K-88). */
const decodeOf = (
  value: unknown,
  where: string,
  path: string,
): NonNullable<GemmGeometry["decode"]> => {
  if (value === "counter") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry decode must be "counter" or absent (${JSON.stringify(value)})`,
  );
};

const gemmGeometryOf = (value: unknown, where: string, path: string): GemmGeometry => {
  const { regM, regN, wgX, wgY, form, decode } = recordOf(value, where, path);
  // Absent stays absent (a table without the fields keeps its keys and WGSL byte for byte)
  return {
    regM: numberOf(regM, where, path, "regM"),
    regN: numberOf(regN, where, path, "regN"),
    wgX: numberOf(wgX, where, path, "wgX"),
    wgY: numberOf(wgY, where, path, "wgY"),
    ...(form === undefined ? {} : { form: formOf(form, where, path) }),
    ...(decode === undefined ? {} : { decode: decodeOf(decode, where, path) }),
  };
};

/**
 * Which optional fields of an f32 GEMM geometry a profile field may carry (perf-ledger K-88): the
 * skeleton form where a kernel takes it (the row-count buckets of linear / matmul / bmm, conv2d),
 * the counter decode in conv2d only.
 *
 * MUST: refuse them elsewhere. The fused attention kernels keep the original form whatever the
 * geometry carries, and every op but conv2d refuses the decode at generation — a table carrying
 * them there would either be silently not what it says or fail only when that op is first built.
 */
const assertGemmFieldsIn = (
  geometry: GemmGeometry,
  where: string,
  path: string,
  allowed: { readonly form: boolean; readonly decode: boolean },
): void => {
  if (geometry.form !== undefined && !allowed.form) {
    throw new CodegenError(
      `${where}: ${path}: the skeleton form is taken only by linear / matmul / bmm and conv2d (form ${JSON.stringify(
        geometry.form,
      )} — the fused attention kernels keep the original form)`,
    );
  }
  if (geometry.decode !== undefined && !allowed.decode) {
    throw new CodegenError(
      `${where}: ${path}: the counter decode is conv2d's alone (decode ${JSON.stringify(
        geometry.decode,
      )})`,
    );
  }
};

/**
 * Guard on the optional skeleton form of an i8a8 geometry (perf-ledger K-85): `"vec4"` is the only
 * value (the same wording as {@link assertI8a8Geometry}, which checks it again at generation).
 */
const skeletonOf = (
  value: unknown,
  where: string,
  path: string,
): NonNullable<I8a8Geometry["skeleton"]> => {
  if (value === "vec4") return value;
  throw new CodegenError(
    `${where}: ${path}: geometry skeleton must be "vec4" or absent (${JSON.stringify(value)})`,
  );
};

const i8a8GeometryOf = (value: unknown, where: string, path: string): I8a8Geometry => {
  const { tileK, skeleton } = recordOf(value, where, path);
  const geometry = {
    ...gemmGeometryOf(value, where, path),
    tileK: numberOf(tileK, where, path, "tileK"),
  };
  // Absent stays absent (a table without the field keeps its keys and WGSL byte for byte)
  return skeleton === undefined
    ? geometry
    : { ...geometry, skeleton: skeletonOf(skeleton, where, path) };
};

/**
 * The shape of `provenance` (field types only). Matching its content is the job of
 * `geometryProfileMismatch` in `@karume/runtime/tune`, and the runtime does not read it — it is
 * checked here because a value that passes the guard is typed as {@link GeometryProfile}, so a value
 * that differs from the type declaration must not pass.
 */
const assertProvenanceShape = (value: unknown, where: string): void => {
  const provenance = recordOf(value, where, "provenance");
  const adapter = recordOf(provenance.adapter, where, "provenance.adapter");
  for (const key of ["vendor", "architecture", "device", "description"] as const) {
    stringOf(adapter[key], where, `provenance.adapter.${key}`);
  }
  const families = recordOf(provenance.families, where, "provenance.families");
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    if (families[family] !== undefined) {
      assertFamilyProvenanceShape(families[family], where, `provenance.families.${family}`);
    }
  }
  if (provenance.attentionBench !== undefined) {
    assertAttentionBenchShape(provenance.attentionBench, where);
  }
};

/** The shape of one family's provenance (field types only, as {@link assertProvenanceShape}). */
const assertFamilyProvenanceShape = (value: unknown, where: string, path: string): void => {
  const entry = recordOf(value, where, path);
  for (const key of ["sweep", "sha256", "date", "candidateSet", "kernels", "caseSet"] as const) {
    stringOf(entry[key], where, `${path}.${key}`);
  }
  if (entry.userAgent === undefined) return;
  if (!Array.isArray(entry.userAgent)) {
    throw new CodegenError(`${where}: ${path}.userAgent is not an array`);
  }
  const agents = denseEntriesOf(entry.userAgent, where, `${path}.userAgent`);
  agents.forEach((agent, index) => stringOf(agent, where, `${path}.userAgent[${index}]`));
};

/** The shape of `provenance.attentionBench` (field types only, as {@link assertProvenanceShape}). */
const assertAttentionBenchShape = (value: unknown, where: string): void => {
  const bench = recordOf(value, where, "provenance.attentionBench");
  for (const key of ["records", "sha256", "date", "shapes"] as const) {
    stringOf(bench[key], where, `provenance.attentionBench.${key}`);
  }
  optionalStringOf(bench.kernels, where, "provenance.attentionBench.kernels");
};

/**
 * Guard on the optional {@link GeometryProfile.attentionOnline}: an object whose `keyWords` is
 * `"tiled"` (the one form a table can choose — the others are N's own).
 */
const assertAttentionOnlineShape = (value: unknown, where: string): void => {
  const { keyWords } = recordOf(value, where, "attentionOnline");
  if (keyWords !== "tiled") {
    throw new CodegenError(
      `${where}: attentionOnline.keyWords must be "tiled" (${JSON.stringify(keyWords)} — omit the field for the form N gives)`,
    );
  }
};

/**
 * Guard on one profile (shape of match, the rules, divisibility conditions of the geometry in every
 * field).
 *
 * The geometry also passes the same guards at generation time (`gemmWgsl` / the i8a8 generators);
 * it is rejected here first so that a broken profile is not left unnoticed **until it hits that
 * op** (checked once at Session construction).
 * NOTE: the w4a8 `groupSize % tileK` depends on the asset's group length, so it cannot be checked
 * here (the job of `assertI8a8GroupGeometry` at generation time).
 *
 * It takes unknown and is a type predicate that narrows a passing value to {@link GeometryProfile}
 * (`acquireGpu` also takes values built by JS callers or from stored data, so a missing field or a
 * wrong type is rejected with this guard's {@link CodegenError} rather than a TypeError). It checks
 * the type of every field (including `provenance`) but does not reject unknown fields (that is the
 * job of `parseGeometryProfileJson`, which reads stored data). It does not check device limits (the
 * job of {@link assertGeometryProfileWithinLimits} — that needs an adapter).
 */
export const assertGeometryProfile: (profile: unknown) => asserts profile is GeometryProfile = (
  profile,
) => {
  const fields = recordOf(profile, "Geometry profile", "table");
  const { id } = fields;
  if (typeof id !== "string") {
    throw new CodegenError(`Geometry profile: id is not a string (${typeof id})`);
  }
  if (id.length === 0) {
    throw new CodegenError(
      "Geometry profile: id is empty (diagnostics could not name the selected table)",
    );
  }
  const where = `Geometry profile '${id}'`;
  if (fields.match !== undefined) {
    const match = recordOf(fields.match, where, "match");
    const vendor = optionalStringOf(match.vendor, where, "match.vendor");
    const architecture = optionalStringOf(match.architecture, where, "match.architecture");
    const description = optionalStringOf(match.description, where, "match.description");
    if (vendor === "" || architecture === "" || description === "") {
      throw new CodegenError(
        `${where}: match vendor / architecture / description must not be an empty string (omit a field that is not specified)`,
      );
    }
    if (architecture !== undefined && vendor === undefined) {
      throw new CodegenError(`${where}: match architecture must be given together with vendor`);
    }
    // description is the model name, a field only for splitting within a vendor / architecture. If
    // it were allowed alone, a rule could hit a machine of another vendor with the same name (or
    // one with an empty vendor).
    if (description !== undefined && (vendor === undefined || architecture === undefined)) {
      throw new CodegenError(
        `${where}: match description must be given together with both vendor and architecture`,
      );
    }
  }
  if (!Array.isArray(fields.gemmRows)) {
    throw new CodegenError(`${where}: gemmRows is not an array`);
  }
  const rows = denseEntriesOf(fields.gemmRows, where, "gemmRows");
  assertGemmRowsRules(
    rows.map((entry, index): GemmRowsRule => {
      const rule = recordOf(entry, where, `gemmRows[${index}]`);
      if (typeof rule.maxRows !== "number") {
        throw new CodegenError(`${where}: maxRows of gemmRows[${index}] is not a number`);
      }
      const path = `gemmRows[${index}].geometry`;
      const geometry = gemmGeometryOf(rule.geometry, where, path);
      assertGemmFieldsIn(geometry, where, path, { form: true, decode: false });
      return { maxRows: rule.maxRows, geometry };
    }),
    where,
  );
  const attention = recordOf(fields.attention, where, "attention");
  const conv2d = recordOf(fields.conv2d, where, "conv2d");
  for (const [name, value, allowed] of [
    ["attention.qk", attention.qk, false],
    ["attention.pv", attention.pv, false],
    ["conv2d.rows64", conv2d.rows64, true],
    ["conv2d.rows32", conv2d.rows32, true],
  ] as const) {
    const geometry = gemmGeometryOf(value, where, name);
    assertGemmFieldsIn(geometry, where, name, { form: allowed, decode: allowed });
    assertGemmGeometry(geometry, `${where}: ${name}`);
  }
  const i8a8 = recordOf(fields.i8a8, where, "i8a8");
  for (const [name, geometry] of [
    ["i8a8.linear", i8a8.linear],
    ["i8a8.attentionQk", i8a8.attentionQk],
    ["i8a8.attentionPv", i8a8.attentionPv],
    ["i8a8.conv3d", i8a8.conv3d],
  ] as const) {
    assertI8a8Geometry(i8a8GeometryOf(geometry, where, name), `${where}: ${name}`);
  }
  if (fields.attentionOnline !== undefined)
    assertAttentionOnlineShape(fields.attentionOnline, where);
  if (fields.provenance !== undefined) assertProvenanceShape(fields.provenance, where);
};

/**
 * The device limits the limit guard checks (the 4 fields of `GPUSupportedLimits` decided by the
 * geometry alone — {@link assertGeometryProfileWithinLimits}). The requiredLimits planned by
 * `acquireGpu` (RequiredLimits) can be passed as is.
 */
export type WorkgroupLimits = {
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeWorkgroupSizeY: number;
  readonly maxComputeWorkgroupStorageSize: number;
};

/**
 * Spelling of the geometry in the over-limit message (names the geometry by its values — the field
 * name alone is not enough to fix a table brought in from outside).
 */
const geometryLabel = (geometry: GemmGeometry | I8a8Geometry): string =>
  `regM=${geometry.regM} regN=${geometry.regN} wgX=${geometry.wgX} wgY=${geometry.wgY}${
    "tileK" in geometry ? ` tileK=${geometry.tileK}` : ""
  }`;

/**
 * Guard that the geometry of every field of a profile fits the device's workgroup limits (takes only
 * tables that passed the shape guard — the order is {@link assertGeometryProfile} → this). It names
 * every field that exceeds a limit and rejects with a {@link CodegenError}.
 *
 * The limits checked are the same 4 as WebGPU's compute pipeline validation: each component of
 * `@workgroup_size(wgX, wgY)` (`maxComputeWorkgroupSizeX` / `Y`), the thread count wgX · wgY
 * (`maxComputeInvocationsPerWorkgroup`), and the workgroup storage of the shared tiles
 * (`maxComputeWorkgroupStorageSize` — {@link gemmWorkgroupStorageBytes} /
 * {@link i8a8WorkgroupStorageBytes}).
 * MUST: an injected table passes here before the device is created. A pipeline with an exceeding
 * geometry fails validation, which goes unnoticed until that op is hit (Session construction, the
 * first run) — the same discipline as "do not create a device with a broken table".
 * GEMM shared memory is counted for the f32 variant (twice f16): a table applies to every variant
 * without choosing the compute variant, so a table where the larger one does not fit fails on some
 * path.
 * NOTE: the dispatch count limit (`maxComputeWorkgroupsPerDimension`) is not checked. It is decided
 * by the geometry together with the **run-time shape** (`ceil(M / tileM)` and so on) and cannot be
 * judged from the table alone — that is the job of `tiledWorkgroups`, which rejects with
 * `DispatchLimitError` when the dispatch is built (src/codegen/dispatch.ts). This guard checks only
 * the shape of one workgroup, which does not depend on the shape of the data.
 * NOTE: tests pin that the embedded tables and the default table pass under WebGPU's default limits
 * (256 / 256 / 256 / 16,384 B — every device provides at least these), so the auto-selection path
 * does not go through here.
 */
export const assertGeometryProfileWithinLimits = (
  profile: GeometryProfile,
  limits: WorkgroupLimits,
): void => {
  const violations: string[] = [];
  const check = (
    name: string,
    geometry: GemmGeometry | I8a8Geometry,
    storageBytes: number,
  ): void => {
    const exceeds = (what: string, value: number, limit: keyof WorkgroupLimits): void => {
      if (value > limits[limit]) {
        violations.push(
          `${name} (${geometryLabel(geometry)}): ${what} ${value} exceeds ${limit} ${
            limits[limit]
          }`,
        );
      }
    };
    exceeds(
      "thread count wgX × wgY",
      geometry.wgX * geometry.wgY,
      "maxComputeInvocationsPerWorkgroup",
    );
    exceeds("wgX", geometry.wgX, "maxComputeWorkgroupSizeX");
    exceeds("wgY", geometry.wgY, "maxComputeWorkgroupSizeY");
    exceeds("shared memory (bytes)", storageBytes, "maxComputeWorkgroupStorageSize");
  };
  const gemm = (name: string, geometry: GemmGeometry): void =>
    check(name, geometry, gemmWorkgroupStorageBytes(geometry, "f32"));
  const i8a8 = (name: string, geometry: I8a8Geometry): void =>
    check(name, geometry, i8a8WorkgroupStorageBytes(geometry));
  profile.gemmRows.forEach((rule, index) => gemm(`gemmRows[${index}]`, rule.geometry));
  gemm("attention.qk", profile.attention.qk);
  gemm("attention.pv", profile.attention.pv);
  gemm("conv2d.rows64", profile.conv2d.rows64);
  gemm("conv2d.rows32", profile.conv2d.rows32);
  i8a8("i8a8.linear", profile.i8a8.linear);
  i8a8("i8a8.attentionQk", profile.i8a8.attentionQk);
  i8a8("i8a8.attentionPv", profile.i8a8.attentionPv);
  i8a8("i8a8.conv3d", profile.i8a8.conv3d);
  if (violations.length > 0) {
    throw new CodegenError(
      `Geometry profile '${profile.id}': geometry exceeds the device limits (${violations.join(" / ")})`,
    );
  }
};

/**
 * The adapter fields used for matching in the selection (`GpuContext.adapterInfo` can be passed as
 * is). `description` is `GPUAdapterInfo.description` as is, and an empty string is treated the same
 * as absent (Chrome without the flag returns an empty one).
 */
type AdapterIdentity = {
  readonly vendor: string;
  readonly architecture: string;
  readonly description?: string;
};

/**
 * Chooses one profile for the adapter (called once at Session construction).
 *
 * Selection order (deterministic): ① vendor, architecture and description of `match` all match →
 * ② description unset, vendor and architecture match → ③ architecture unset, only vendor matches →
 * ④ {@link DEFAULT_GEOMETRY_PROFILE}. Matching is exact string equality (no normalisation, no prefix
 * or substring matching). If the adapter's description is empty or absent, ① never matches. A table
 * without `match` is not matched (injection only), but it still goes through the guard and the
 * duplicate-id check (so a broken table is not left in the list unnoticed).
 *
 * MUST: a list in which 2 tables can hit the same rank fails loudly — silently choosing one by list
 * order would let an edit of the list alone change the selection. **The list itself is rejected,
 * not only for the adapter that hits it** (2 tables with the same `match` collide only on a machine
 * with that `match` — on other machines' CI it would stay invisible).
 * MUST: do not choose the nearest neighbour by similarity (closeness of GPU names and so on). That
 * would break the correspondence between the reference sha rows per environment key (ADR 0106) and
 * "which geometry that machine ran with".
 */
export const selectGeometryProfile = (
  adapter: AdapterIdentity,
  profiles: readonly GeometryProfile[] = BUILTIN_GEOMETRY_PROFILES,
): GeometryProfile => {
  const ids = new Set([DEFAULT_GEOMETRY_PROFILE.id]);
  const matches = new Set<string>();
  for (const profile of profiles) {
    assertGeometryProfile(profile);
    if (ids.has(profile.id)) {
      throw new CodegenError(
        `Geometry profile '${profile.id}': duplicate id (diagnostics could not name the table)`,
      );
    }
    ids.add(profile.id);
    if (profile.match === undefined) continue;
    const { vendor, architecture, description } = profile.match;
    if (vendor === undefined) {
      throw new CodegenError(
        `Geometry profile '${profile.id}': match is empty (only the default profile matches every adapter — ` +
          "an injection-only table omits match)",
      );
    }
    // The JSON array form cannot collide on a separator (`a/b` + `c` is not confused with `a` + `b/c`).
    const match = JSON.stringify([vendor, architecture ?? null, description ?? null]);
    if (matches.has(match)) {
      throw new CodegenError(
        `Geometry profile '${profile.id}': match (vendor=${vendor} / architecture=${
          architecture ?? "(unspecified)"
        } / description=${
          description ?? "(unspecified)"
        }) is the same as another profile's (two tables would match at the same rank)`,
      );
    }
    matches.add(match);
  }
  // A table without match has `profile.match?.vendor` undefined, which never equals the string
  // adapter.vendor.
  const described = adapter.description === "" ? undefined : adapter.description;
  const withDescription =
    described === undefined
      ? undefined
      : profiles.find(
          (profile) =>
            profile.match?.vendor === adapter.vendor &&
            profile.match.architecture === adapter.architecture &&
            profile.match.description === described,
        );
  if (withDescription !== undefined) return withDescription;
  const exact = profiles.find(
    (profile) =>
      profile.match?.vendor === adapter.vendor &&
      profile.match.architecture !== undefined &&
      profile.match.architecture === adapter.architecture &&
      profile.match.description === undefined,
  );
  if (exact !== undefined) return exact;
  const vendorOnly = profiles.find(
    (profile) =>
      profile.match?.vendor === adapter.vendor && profile.match.architecture === undefined,
  );
  return vendorOnly ?? DEFAULT_GEOMETRY_PROFILE;
};

/**
 * Looks up the geometry from the row count M (the 3 paths matmul / bmm / linear).
 *
 * MUST: the derivation phase passes **the return value of this one call** to all three of key, WGSL
 * and dispatch (if one of them got a different geometry, the cached WGSL and the dispatch count would
 * not match, and output tiles would be missing without any exception).
 */
export const gemmRowsGeometry = (profile: GeometryProfile, rows: number): GemmGeometry => {
  assertU32Params("Geometry selection", { "row count M": rows });
  assertGemmRowsRules(profile.gemmRows, `Geometry profile '${profile.id}'`);
  const rules = profile.gemmRows;
  // Rules that passed the guard end with Infinity, so a u32 row count always hits one of them.
  return (rules.find((rule) => rows <= rule.maxRows) ?? rules[rules.length - 1]).geometry;
};

/**
 * Looks up the profile's geometry from the conv2d m-tile class (64 / 32 as returned by
 * `conv2dIgemmMTile`).
 *
 * MUST: there are only 2 classes. Any other m tile means the predicate and the table disagree (a
 * value was added to only one of them), so fail loudly — rounding to the nearer one would silently
 * apply another class's geometry to a class that was not measured.
 */
export const conv2dProfileGeometry = (profile: GeometryProfile, mTile: number): GemmGeometry => {
  if (mTile === GEMM_TILE) return profile.conv2d.rows64;
  if (mTile === GEMM_MTILE_SMALL) return profile.conv2d.rows32;
  throw new CodegenError(
    `Geometry profile '${profile.id}': no class matches conv2d m tile ${mTile} (only ${GEMM_TILE} / ${GEMM_MTILE_SMALL})`,
  );
};

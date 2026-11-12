/**
 * Matching: whether a saved table may be applied (DECIDED: ADR 0117 decision 5, per op family since
 * its addendum 2026-11-12).
 *
 * The runtime does not match at injection (ADR 0115 addendum decision 6 — keeps the use of applying
 * another machine's table for an A/B). Matching is exposed as a pure function; the application
 * decides whether to call it and what to do on a mismatch (run with the default, re-sweep, tell the
 * user) — the same style as `sessionOptionsViolation`.
 */
import {
  DEFAULT_GEOMETRY_PROFILE,
  GEOMETRY_PROFILE_FAMILIES,
  type GeometryProfile,
  type GeometryProfileFamily,
} from "../kernels/geometry-profile.ts";
import {
  attentionBenchKernelsId,
  attentionBenchShapeSetId,
  familyCaseSetId,
  familyKernelsId,
  KernelsIdError,
} from "./fingerprint.ts";

const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

/**
 * A key derived by the current runtime, or the reason it cannot be derived: a geometry or a kernel
 * that fails the current codegen checks or the dispatch-count limit (a table that would fail
 * Session construction if applied — returned as a mismatch instead of thrown; other exceptions are
 * implementation errors and propagate as is).
 */
const derived = (derive: () => string): { readonly id: string } | { readonly error: string } => {
  try {
    return { id: derive() };
  } catch (cause) {
    if (cause instanceof KernelsIdError) return { error: cause.message };
    throw cause;
  }
};

/** The first mismatch of one family (undefined when it matches). */
const familyMismatch = (
  profile: GeometryProfile & { readonly provenance: NonNullable<GeometryProfile["provenance"]> },
  family: GeometryProfileFamily,
): string | undefined => {
  const where = `Table '${profile.id}', family ${family}`;
  const kernels = derived(() => familyKernelsId(profile, family));
  if ("error" in kernels) {
    return `${where}: its geometry cannot build the kernels of the sweep shapes (${kernels.error})`;
  }
  const entry = profile.provenance.families[family];
  if (entry === undefined) {
    const defaults = familyKernelsId(DEFAULT_GEOMETRY_PROFILE, family);
    return kernels.id === defaults
      ? undefined
      : `${where}: no provenance for family ${family} but its geometry is not the default (fingerprint ${kernels.id} / the default table's ${defaults} — a table edited by hand, one whose provenance lost the family's entry, or the runtime's default geometry of the family changed since the table was written)`;
  }
  if (kernels.id !== entry.kernels) {
    return (
      `${where}: kernel fingerprint differs (table ${entry.kernels} / current runtime ${kernels.id}` +
      " — a runtime update changed the kernels of the family's fields or of its default geometry; sweep this family again)"
    );
  }
  const caseSet = familyCaseSetId(family);
  if (caseSet !== entry.caseSet) {
    return (
      `${where}: case-set version differs (table ${entry.caseSet} / current runtime ${caseSet}` +
      " — a runtime update changed the family's sweep cases or bounds; sweep this family again)"
    );
  }
  return undefined;
};

/**
 * The first mismatch of the online attention's evidence (undefined when it matches): a table
 * carrying `attentionOnline` needs `provenance.attentionBench` with `kernels`, and a table without
 * it a block without `kernels` (the key is present exactly when the table carries the field);
 * `kernels`, when present, and `shapes` are compared with the current runtime's.
 */
const attentionBenchMismatch = (
  profile: GeometryProfile & { readonly provenance: NonNullable<GeometryProfile["provenance"]> },
): string | undefined => {
  const { attentionBench } = profile.provenance;
  const where = `Table '${profile.id}'`;
  if (attentionBench === undefined) {
    return profile.attentionOnline === undefined
      ? undefined
      : `${where}: it carries attentionOnline but no provenance.attentionBench (nothing to match the online attention's form against)`;
  }
  if (profile.attentionOnline !== undefined && attentionBench.kernels === undefined) {
    return `${where}: it carries attentionOnline but provenance.attentionBench has no kernels (nothing to match the online attention's kernels against)`;
  }
  if (profile.attentionOnline === undefined && attentionBench.kernels !== undefined) {
    return `${where}: provenance.attentionBench names the online attention's kernels but the table chooses no form (no attentionOnline — the block does not belong to this table's fields)`;
  }
  if (attentionBench.kernels !== undefined) {
    const kernels = derived(attentionBenchKernelsId);
    if ("error" in kernels) {
      return `${where}: the online attention's kernels cannot be built at the attention bench's shapes (${kernels.error})`;
    }
    if (kernels.id !== attentionBench.kernels) {
      return (
        `${where}: the online attention's kernel fingerprint differs (table ${attentionBench.kernels} / current runtime ${kernels.id}` +
        " — a runtime update changed the kernels its online attention form was chosen between; measure the attention bench again)"
      );
    }
  }
  const shapes = attentionBenchShapeSetId();
  if (shapes !== attentionBench.shapes) {
    return (
      `${where}: the attention bench's shape set differs (table ${attentionBench.shapes} / current runtime ${shapes}` +
      " — a runtime update changed the shapes its online attention form was measured on)"
    );
  }
  return undefined;
};

/**
 * Whether the table `profile` may be applied to this adapter and the current runtime. Returns the
 * message of the first mismatch, or `undefined` when everything matches. Does not throw.
 *
 * What is compared (all by exact string equality, in this order):
 * 1. `provenance` is present (a table without it = a hand-written table → nothing to match
 *    against).
 * 2. The four adapter fields (`vendor` / `architecture` / `device` / `description` — empty strings
 *    are compared as values too).
 * 3. `provenance.families` is an object (a table saved in the earlier shape, without per-family
 *    entries, is returned as a mismatch — not thrown).
 * 4. Per op family, in the fixed family order (`GEOMETRY_PROFILE_FAMILIES`): with an entry in
 *    `provenance.families`, the family's kernel fingerprint (re-derived from the table by the
 *    current runtime — the WGSL counted without comments and whitespace, ADR 0117 addendum
 *    2026-10-19), then its case-set version; without an entry, that the family's fingerprint equals
 *    the default table's (its fields are the default). The message names the family.
 * 5. The online attention's evidence (`provenance.attentionBench` — ADR 0117 addenda 2026-11-12): a
 *    table carrying `attentionOnline` has the block with `kernels`, a table without it has no
 *    `kernels` in the block; `kernels`, when present, equals the current runtime's
 *    `attentionBenchKernelsId`, and `shapes` its `attentionBenchShapeSetId`.
 *
 * Not compared: `candidateSet` (whether a quick+ table suffices is the application's call), `match`
 * (injection does not look at `match`), browser and driver versions (no information is available —
 * an update may silently slow things down, so re-sweeping is an explicit action of the user).
 *
 * Limit: Chrome without flags returns empty `device` and `description`, so Apple M2 and M5 have the
 * same four fields. This function cannot detect a table carried between those two machines.
 *
 * Deriving the fingerprints needs no GPU, but it runs codegen over every sweep case (about 1 MB of
 * WGSL), and over a family's cases once more with the default table for a family without an entry.
 * A table whose geometry cannot build the kernels of the sweep shapes (it fails a codegen check, or
 * the dispatch count exceeds the limit) returns that reason, naming the family.
 */
export const geometryProfileMismatch = (
  profile: GeometryProfile,
  // Requires only the adapter fields that are compared (`GPUAdapterInfo` and
  // `GpuContext.adapterInfo` can be passed as is)
  adapterInfo: Pick<GPUAdapterInfo, "vendor" | "architecture" | "device" | "description">,
): string | undefined => {
  const { provenance } = profile;
  if (provenance === undefined) {
    return `Table '${profile.id}' has no provenance (nothing to match against — a hand-written table or one not made from sweeps)`;
  }
  for (const field of ADAPTER_FIELDS) {
    if (provenance.adapter[field] !== adapterInfo[field]) {
      return `Table '${profile.id}': adapter ${field} differs (table ${JSON.stringify(
        provenance.adapter[field],
      )} / this adapter ${JSON.stringify(adapterInfo[field])} — a table made on another machine or another adapter)`;
    }
  }
  // The type says `families` is an object, but a table saved by an earlier runtime (the single
  // top-level keys) and read back with JSON.parse and a cast has none; reading its entries would
  // throw a TypeError, and this function promises a message instead
  const families: unknown = provenance.families;
  if (typeof families !== "object" || families === null) {
    return `Table '${profile.id}': its provenance has no per-family entries (provenance.families is ${JSON.stringify(families)} — a table saved by an earlier runtime; sweep again)`;
  }
  const stored = { ...profile, provenance };
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    const mismatch = familyMismatch(stored, family);
    if (mismatch !== undefined) return mismatch;
  }
  return attentionBenchMismatch(stored);
};

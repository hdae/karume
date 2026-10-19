/**
 * Matching: whether a saved table may be applied (DECIDED: ADR 0117 decision 5).
 *
 * The runtime does not match at injection (ADR 0115 addendum decision 6 — keeps the use of applying
 * another machine's table for an A/B). Matching is exposed as a pure function; the application
 * decides whether to call it and what to do on a mismatch (run with the default, re-sweep, tell the
 * user) — the same style as `sessionOptionsViolation`.
 */
import type { GeometryProfile } from "../kernels/geometry-profile.ts";
import { geometryProfileKernelsId, KernelsIdError, sweepCaseSetId } from "./fingerprint.ts";

const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

/**
 * Whether the table `profile` may be applied to this adapter and the current runtime. Returns the
 * message of the first mismatch, or `undefined` when everything matches. Does not throw.
 *
 * What is compared (all by exact string equality, in this order):
 * 1. `provenance` is present (a table without it = a hand-written table → nothing to match
 *    against).
 * 2. The four adapter fields (`vendor` / `architecture` / `device` / `description` — empty strings
 *    are compared as values too).
 * 3. The kernel fingerprint (the value re-derived from the table by the current runtime vs
 *    `provenance.kernels` — the WGSL counted without comments and whitespace, ADR 0117 addendum
 *    2026-10-19).
 * 4. The case-set version (the current runtime's value vs `provenance.caseSet`).
 *
 * Not compared: `candidateSet` (whether a quick+ table suffices is the application's call), `match`
 * (injection does not look at `match`), browser and driver versions (no information is available —
 * an update may silently slow things down, so re-sweeping is an explicit action of the user).
 *
 * Limit: Chrome without flags returns empty `device` and `description`, so Apple M2 and M5 have the
 * same four fields. This function cannot detect a table carried between those two machines.
 *
 * Deriving the fingerprint needs no GPU, but it runs codegen over every sweep case (about 1 MB of
 * WGSL). A table whose geometry cannot build the kernels of the sweep shapes (it fails a codegen
 * check, or the dispatch count exceeds the limit) returns that reason.
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
  let kernels: string;
  try {
    kernels = geometryProfileKernelsId(profile);
  } catch (cause) {
    // The table's geometry fails the current codegen checks or the dispatch-count limit — a table
    // that would fail Session construction if applied, so return it as a mismatch instead of
    // throwing (other exceptions are implementation errors and propagate as is)
    if (cause instanceof KernelsIdError) {
      return `Table '${profile.id}': its geometry cannot build the kernels of the sweep shapes (${cause.message})`;
    }
    throw cause;
  }
  if (kernels !== provenance.kernels) {
    return (
      `Table '${profile.id}': kernel fingerprint differs (table ${provenance.kernels} / current runtime ${kernels}` +
      " — a runtime update changed the kernels of the table or of the default geometry)"
    );
  }
  const caseSet = sweepCaseSetId();
  if (caseSet !== provenance.caseSet) {
    return (
      `Table '${profile.id}': case-set version differs (table ${provenance.caseSet} / current runtime ${caseSet}` +
      " — a runtime update changed the sweep cases or the gemmRows bucket bounds)"
    );
  }
  return undefined;
};

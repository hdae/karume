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
 *    `provenance.kernels`).
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
    return `表 '${profile.id}' に provenance が無い（照合の材料が無い — 手書きの表か、掃引から作っていない表）`;
  }
  for (const field of ADAPTER_FIELDS) {
    if (provenance.adapter[field] !== adapterInfo[field]) {
      return `表 '${profile.id}' の adapter の ${field} が違う（表 ${JSON.stringify(
        provenance.adapter[field],
      )} / この adapter ${JSON.stringify(adapterInfo[field])} — 別の機か別の adapter で作った表）`;
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
      return `表 '${profile.id}' の幾何から掃引の shape のカーネルを組めない（${cause.message}）`;
    }
    throw cause;
  }
  if (kernels !== provenance.kernels) {
    return (
      `表 '${profile.id}' のカーネルの指紋が違う（表 ${provenance.kernels} / 今の runtime ${kernels}` +
      " — runtime の更新で表か既定の幾何のカーネルが変わった）"
    );
  }
  const caseSet = sweepCaseSetId();
  if (caseSet !== provenance.caseSet) {
    return (
      `表 '${profile.id}' のケース集合の版が違う（表 ${provenance.caseSet} / 今の runtime ${caseSet}` +
      " — runtime の更新で掃引のケースか gemmRows の段の境界が変わった）"
    );
  }
  return undefined;
};

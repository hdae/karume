/**
 * In-app tuning of the GEMM tile geometry (`@karume/runtime/tune`).
 *
 * An application measures the tile geometries on the user's own device, derives a geometry profile
 * from the measurements, stores it, and passes it to the next `acquireGpu({ geometryProfile })`.
 * This is the only part of the package that measures anything: the runtime never measures on the
 * `acquireGpu` / `Session` path, and an injected profile stays fixed for the device's lifetime.
 *
 * - {@link runGeometrySweep} acquires a **dedicated** GPU device, sweeps the candidate geometries
 *   case by case, and destroys the device when it returns. It never takes a `GpuContext`: do not run
 *   a sweep while inference runs on the same GPU — the measurements would be meaningless and the two
 *   would compete for the GPU. The runtime cannot see other devices, so this is the caller's
 *   responsibility.
 * - The sweep's cases come from the op census of Anima (a DiT image model): the profile's fields
 *   are decided by Anima's shapes, also for applications that run other models.
 * - {@link deriveGeometryProfile} turns one or more sweep records into a profile. A geometry is
 *   adopted for a field only if, in every case of that field, its output matched the default
 *   geometry's output bit for bit and it was at least ×1.05 faster. The evidence is limited to the
 *   sweep's cases on that device; outputs of an in-app profile are not covered by the repository's
 *   end-to-end reference hashes.
 * - {@link geometryProfileJson} serializes a profile for storage (`Infinity` is written as `1e999`,
 *   which `JSON.parse` reads back as `Infinity`); {@link parseGeometryProfileJson} reads it back,
 *   rejecting unknown fields at every level and anything the injection gate would reject, with a
 *   {@link GeometryProfileParseError} that names the field.
 * - {@link geometryProfileMismatch} tells whether a stored profile may be used with this adapter
 *   and this runtime: it compares the four adapter fields the profile was swept on, the kernel
 *   fingerprint (derived again from the profile by this runtime) and the case-set id
 *   ({@link sweepCaseSetId}), and returns the first mismatch as a message. On a mismatch, run with
 *   the default table (or another one) and sweep again. It cannot detect a browser or driver
 *   update, and Chrome without developer flags reports an empty `device` and `description`, so an
 *   Apple M2 and an M5 look the same to it.
 *
 * To apply a stored profile, pass a callback as `acquireGpu({ geometryProfile })`: the runtime calls
 * it once, after it has obtained the adapter and before it creates the device, with that adapter's
 * information. Inside it, parse the stored text, call {@link geometryProfileMismatch}, and return the
 * profile, or `undefined` to fall back to automatic selection. The callback must be a synchronous
 * pure function: read the stored text (from IndexedDB, a file, ...) before calling `acquireGpu`,
 * because an adapter may expire while it waits, and a device requested from an expired adapter is
 * lost from the start. A callback that returns a Promise is rejected. The runtime itself does not
 * compare an injected profile with the adapter. An exception thrown inside the callback, including
 * the {@link GeometryProfileParseError} of a broken stored profile, propagates as is and fails
 * `acquireGpu`; the right move for a broken stored profile is to catch that error inside the
 * callback, return `undefined` (the default selection), and — after `acquireGpu` has returned, so
 * the callback itself stays pure — discard the stored text and sweep again.
 *
 * What stays with the application: deciding with {@link geometryProfileMismatch} whether a stored
 * profile may still be used, never sweeping while inference runs on the same GPU, and recording
 * which profile each run used (outputs reproduce only under the same profile). Doing "start with the
 * default table, sweep in the background, re-acquire with the new table" automatically amounts to
 * auto-tuning at the application level; the runtime still keeps a profile fixed for a device's
 * lifetime and measures nothing on the `acquireGpu` / `Session` path. A profile's
 * `provenance.userAgent` lists the browsers (or `Deno/<version>`) its sweeps ran under, one string
 * per distinct browser; it is not part of the match, so an application can compare it with the
 * current `navigator.userAgent` and suggest a new sweep after a browser update.
 *
 * @module
 */

/**
 * Sweeps the tile geometries on a dedicated device and returns the sweep record
 * (`karume-geometry-sweep/2`). Uses GPU timestamps when the adapter lists `timestamp-query`.
 */
export { runGeometrySweep } from "./src/tune/sweep.ts";
export type { GeometrySweepOptions, GeometrySweepProgress } from "./src/tune/sweep.ts";
/** The candidate set (`"quick"`, `"quick+"` or `"full"`) and the op families of a sweep. */
export type { CandidateSet as GeometrySweepCandidateSet } from "./src/tune/geometries.ts";
export type { SweepOp as GeometrySweepOp } from "./src/tune/cases.ts";
/** The sweep record and its parts. */
export type {
  CaseSummary as GeometrySweepCaseSummary,
  Report as GeometrySweepReport,
  ReportAdapter as GeometrySweepAdapter,
  SweepRow as GeometrySweepRow,
  TimingUnit as GeometrySweepTimingUnit,
} from "./src/tune/report.ts";

/**
 * Derives a geometry profile from sweep records, and serializes a profile as JSON for storage and
 * injection.
 */
export { deriveGeometryProfile, geometryProfileJson } from "./src/tune/derive.ts";
export type {
  DeriveGeometryProfileOptions,
  GeometryProfileDerivation,
  GeometrySweepInput,
  ProfileTarget as GeometryProfileTarget,
} from "./src/tune/derive.ts";

/**
 * Reads a stored profile (the text of {@link geometryProfileJson}) back into a `GeometryProfile`,
 * or throws {@link GeometryProfileParseError}.
 */
export { GeometryProfileParseError, parseGeometryProfileJson } from "./src/tune/profile-json.ts";

/**
 * Whether a stored profile may be used with this adapter and this runtime (`undefined` when it
 * may, otherwise the first mismatch as a message). Never throws for a mismatch.
 */
export { geometryProfileMismatch } from "./src/tune/mismatch.ts";

/**
 * The id of the sweep's case set (`provenance.caseSet`), derived from this runtime without a GPU as
 * a 16-digit hexadecimal string.
 */
export { sweepCaseSetId } from "./src/tune/fingerprint.ts";

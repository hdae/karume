/**
 * In-app tuning and measurement (`@karume/runtime/tune`): the GEMM tile geometry, the session A/B
 * of whole runs, and the public u32 identity comparison.
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
 * - The sweep's cases come from the op census of the models the profile serves: Anima (a DiT image
 *   model) for the GEMM, attention and conv2d fields, and the VAE decoder of Wan2.2 TI2V-5B for
 *   `i8a8.conv3d`. The profile's fields are decided by those shapes, also for applications that run
 *   other models.
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
 * {@link runAttentionRouteBench} is a separate diagnostic, not part of the profile flow: it times
 * the two a8 routes of fused attention (S stored in f16 over three dispatches, and the online
 * kernel) at given shapes with the kernels, row blocks and variants production would use on the
 * caller's device, and returns a JSON-serializable record. It chooses nothing and changes no
 * profile. The same rule applies: do not run it while inference runs on the same GPU.
 *
 * {@link runSessionAb} is the wall-clock A/B of whole runs, the measurement a fusion rule (built-in
 * or external) is accepted or dropped on: two or more arms — Session configurations built with
 * {@link sessionArm}, or a caller's own pipeline — run the same work in alternating order (one
 * untimed warm-up round, then arms forward and reverse in every round: A B B A), the wall clock
 * spans each run alone (its readback included; opening, closing, the arm's counts and the identity
 * comparison outside; with `openPerVisit` an untimed run of the fresh Session precedes the timed
 * one), every run must report the counts (fusion hits, dispatches) of its arm's first run, and every
 * run's outputs are compared as u32 words with a copy of the first arm's (the base's) first outputs
 * through {@link wordIdentity}. It exists because per-dispatch GPU time overstates what removing a
 * small dispatch saves in an untimed run. It shares nothing with the geometry sweep: no case list,
 * no fingerprint, a record of its own (`karume-session-ab/1`). {@link sessionAbStatistics} and
 * {@link judgeSessionAb} are pure functions of a record (a stored record can be judged again): per
 * arm the median of the paired round differences, the gain over the base's median, the noise floor
 * (how far each arm moved against itself within a round; conservative on purpose, it does not shrink
 * with more rounds), the interval the median difference can lie in (read from the differences
 * themselves; when it holds 0 the rounds do not agree on which arm is faster) and one verdict — `"adopt"`, `"reject"` or `"inconclusive"` — with identity required by default
 * (a class E rule). A verdict speaks for the record's device, backend and work only;
 * `"inconclusive"` does not mean the difference is zero. Never run it while other work uses the same
 * GPU: the times would measure the contention.
 *
 * {@link wordIdentity} is the public u32 identity comparison (NaN by class or by payload), the one
 * comparison loop behind the A/B and the fused-vs-unfused checks.
 *
 * @module
 */

/**
 * Sweeps the tile geometries on a dedicated device and returns the sweep record
 * (`karume-geometry-sweep/2`). Uses GPU timestamps when the adapter lists `timestamp-query`.
 */
export { runGeometrySweep } from "./src/tune/sweep.ts";
export type { GeometrySweepOptions, GeometrySweepProgress } from "./src/tune/sweep.ts";
/**
 * The candidate set (`"quick"`, `"quick+"`, `"full"`, or an evaluation set on cases of its own,
 * from whose record no table is derived: `"k85"` — the vec4 skeleton of the i8a8 linear — and
 * `"k88"` — the f32 GEMM skeleton forms and the conv2d counter decode) and the op families of a
 * sweep.
 */
export type { CandidateSet as GeometrySweepCandidateSet } from "./src/tune/geometries.ts";
export type { SweepOp as GeometrySweepOp } from "./src/tune/cases.ts";
/**
 * Every case of the sweep's shape table as its id and op family, in op-family order: the ids
 * `GeometrySweepOptions.cases` takes and the cases {@link sweepCaseSetId} covers. Given the
 * evaluation set `"k85"`, the cases of its own table instead, each with its `censusCount` (nodes in
 * one Wan2.2 TI2V-5B 33f DiT forward, 0 for a guard shape of another model); given `"k88"`, its
 * cases with their `censusCount` and the unit it counts in, `censusGroup` (one gemma4 E2B prefill
 * chunk, one Wan2.2 VAE next chunk, … — absent with a count of 0 for a shape of no model). A new
 * array on every call.
 */
export { sweepCaseEntries as geometrySweepCases } from "./src/tune/cases.ts";
export type { SweepCaseEntry as GeometrySweepCase } from "./src/tune/cases.ts";
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

/**
 * The kernel fingerprint of the candidates a sweep record measured (the record's
 * `candidateKernels`), derived from its rows and its `dp4a` without a GPU. An application that
 * merges the records of several `runGeometrySweep` calls (one per case, with idle time between
 * them) recomputes it over the merged rows: each record fingerprints only its own rows, and
 * {@link deriveGeometryProfile} rejects a record whose value differs from the one it derives.
 */
export { sweepCandidateKernelsId } from "./src/tune/fingerprint.ts";

/**
 * Times the s16 and online routes of a8 fused attention at given shapes (default: Wan2.2 TI2V-5B's
 * self- and cross-attention) on the caller's device and returns the record
 * (`karume-attention-bench/2`). Uses GPU timestamps when the device was acquired with
 * `gpuTiming: true`. `onlineTiled` adds a measure-first candidate (perf-ledger K-106).
 */
export {
  ATTENTION_BENCH_FORMAT,
  runAttentionRouteBench,
  WAN22_ATTENTION_SHAPES,
} from "./src/tune/attention-bench.ts";
export type {
  AttentionBenchOptions,
  AttentionBenchProgress,
  AttentionBenchReport,
  AttentionBenchRoute,
  AttentionBenchShape,
  AttentionBenchShapeResult,
  AttentionBenchStage,
  AttentionBenchVariants,
} from "./src/tune/attention-bench.ts";

/**
 * The wall-clock A/B of whole runs (`karume-session-ab/1`): the runner, the arm built from a
 * prepared model, the record and its parts.
 */
export { runSessionAb } from "./src/tune/session-ab.ts";
export type {
  SessionAbAdapter,
  SessionAbArm,
  SessionAbArmRecord,
  SessionAbEnvironment,
  SessionAbIdentity,
  SessionAbMismatch,
  SessionAbOptions,
  SessionAbOutput,
  SessionAbRecord,
  SessionAbReport,
  SessionAbRun,
  SessionAbVisitIdentity,
} from "./src/tune/session-ab.ts";
export { sessionArm } from "./src/tune/session-arm.ts";
export type { SessionArmOptions } from "./src/tune/session-arm.ts";

/** The statistics and the verdict of a session A/B record, pure functions of the record. */
export { judgeSessionAb, sessionAbStatistics } from "./src/tune/session-ab-judge.ts";
export type {
  JudgeSessionAbOptions,
  SessionAbComparison,
  SessionAbJudgement,
  SessionAbVerdict,
} from "./src/tune/session-ab-judge.ts";

/** The u32 identity comparison of two word arrays, NaN by a declared policy. */
export { wordIdentity } from "./src/tune/word-identity.ts";
export type { NanPolicy, WordIdentity, WordMismatch } from "./src/tune/word-identity.ts";

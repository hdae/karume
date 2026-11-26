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
 *   model) for the GEMM, attention and conv2d fields, the DiTs of Anima and Wan2.2 TI2V-5B with a
 *   small-M guard of Irodori for `i8a8.linear` (by row-count buckets of its own), and the VAE
 *   decoder of Wan2.2 TI2V-5B for `i8a8.conv3d`. The profile's fields are decided by those shapes,
 *   also for applications that run other models.
 * - {@link deriveGeometryProfile} turns one or more sweep records into a profile. A geometry is
 *   adopted for a field only if, in every case of that field, its output matched the default
 *   geometry's output bit for bit and it was at least ×1.05 faster. The evidence is limited to the
 *   sweep's cases on that device; outputs of an in-app profile are not covered by the repository's
 *   end-to-end reference hashes. The fields fall into six op families (`gemm` — the row-count
 *   buckets of linear / matmul / bmm —, `attention`, `conv2d`, `i8a8-linear`, `i8a8-attention`,
 *   `i8a8-conv3d`), and the profile's provenance and matching keys are kept per family: given an
 *   existing profile as `base`, the families the new records do not cover are carried over from it
 *   unchanged, so sweeping one family does not require sweeping the others again.
 * - {@link geometryProfileJson} serializes a profile for storage (`Infinity` is written as `1e999`,
 *   which `JSON.parse` reads back as `Infinity`); {@link parseGeometryProfileJson} reads it back,
 *   rejecting unknown fields at every level and anything the injection gate would reject, with a
 *   {@link GeometryProfileParseError} that names the field.
 * - {@link geometryProfileMismatch} tells whether a stored profile may be used with this adapter
 *   and this runtime: it compares the four adapter fields the profile was swept on and, per op
 *   family, the kernel fingerprint (derived again from the profile by this runtime) and the case-set
 *   id ({@link sweepCaseSetIds}) — a family the profile has no provenance for must be the default —
 *   and returns the first mismatch as a message, naming the family. On a mismatch, run with the
 *   default table (or another one) and sweep again (that family is enough, derived with the stored
 *   profile as `base`). It cannot detect a browser or driver update, and Chrome without developer
 *   flags reports an empty `device` and `description`, so an Apple M2 and an M5 look the same to it.
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
 * lifetime and measures nothing on the `acquireGpu` / `Session` path. Each op family entry of a
 * profile's provenance has a `userAgent` that lists the browsers (or `Deno/<version>`) its sweeps ran
 * under, one string per distinct browser; it is not part of the match, so an application can compare
 * it with the current `navigator.userAgent` and suggest a new sweep after a browser update.
 *
 * {@link runAttentionRouteBench} is a separate diagnostic, not part of the profile flow: it times
 * the two a8 routes of fused attention (S stored in f16 over three dispatches, and the online
 * kernel) at given shapes with the kernels, row blocks and variants production would use on the
 * caller's device, and returns a JSON-serializable record. It chooses nothing and changes no
 * profile. The same rule applies: do not run it while inference runs on the same GPU.
 *
 * {@link runGemvSweep} measures the GEMV choices of `linear` (the sequential variant, the row-block
 * height, the parallel GEMV's lane count, staged or subgroup) on a dedicated device, at the shapes of
 * its case table (the default profile's GEMV rows and the other integer-storage linears a Gemma 4
 * decode token or draft step runs), and returns its own record (`karume-gemv-sweep/4`). After a warm
 * start, every kernel of a row count gets a quick look, and only the references and the kernels near
 * enough to matter are timed and checked: a screened kernel's row holds its quick time only. At M = 4
 * and 8 a reorder candidate that the same run's M = 1 group of the case already leaves unadoptable is
 * not measured at all: its row is marked skipped. Each
 * timed candidate's output is checked in the sweep: a candidate of the baseline's summation order
 * must write the baseline's words, a reordering one must stay inside the band of `linear` against
 * an f64 oracle on the host. The same rule applies: never alongside inference on the same GPU.
 * {@link deriveLinearGemvChoices} turns any number of its records of one adapter into a profile's
 * `linearGemv`, reading them group by group (a record's case at a row count): a group is evidence
 * when its rows are of the current runtime (each row and each case carries its own fingerprint) and
 * it timed the kernel the current default runs for the candidate's class, whatever default the record
 * was made under. That kernel's time is its fastest over the records, read only when a second record
 * repeats it within ×1.10. A candidate replaces the default profile's choice for a shape only when
 * groups of two or more records are evidence for it, it is at least ×1.10 faster than that choice in
 * every one, and its check held. The identical class
 * writes the sequential variant's per-shape exceptions, which change no number of any Session; the
 * reorder class writes the parallel rows, which change the numbers of Sessions at `reorder`, inside
 * the band, and need evidence at every row count a row serves (M = 1 to 8: the `full` set). Groups
 * that are not current are reported, not refused. {@link planGemvMeasurements} says what the records
 * lack for that derivation to be complete.
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
 * (`karume-geometry-sweep/4`). Uses GPU timestamps when the adapter lists `timestamp-query`. After
 * a case's warm start every geometry gets a quick look, and only the default geometry and the
 * candidates whose quick time is within the margin of the default's are timed in full
 * (`GeometrySweepOptions.quickLookMargin`; 0 times every geometry).
 */
export { runGeometrySweep } from "./src/tune/sweep.ts";
export type { GeometrySweepOptions, GeometrySweepProgress } from "./src/tune/sweep.ts";
/**
 * The candidate set (`"quick"`, `"quick+"`, `"full"`, or an evaluation set on cases of its own,
 * from whose record no table is derived: `"k88"` — the f32 GEMM skeleton forms and the conv2d
 * counter decode) and the op families of a sweep. In `"quick+"` and `"full"` a linear / matmul /
 * bmm case measures every tile also with each skeleton form, and a conv2d case also with the
 * counter decode and with both, so a table derived from their records can adopt them; `"quick"`
 * measures the tiles alone.
 */
export type { CandidateSet as GeometrySweepCandidateSet } from "./src/tune/geometries.ts";
export type { SweepOp as GeometrySweepOp } from "./src/tune/cases.ts";
/**
 * Every case of the sweep's shape table as its id and op family, in op-family order: the ids
 * `GeometrySweepOptions.cases` takes and the cases {@link sweepCaseSetIds} covers. Given the
 * evaluation set `"k88"`, the cases of its own table instead, with their `censusCount` and the unit
 * it counts in, `censusGroup` (one gemma4 E2B prefill chunk, one Wan2.2 VAE next chunk, … — absent
 * with a count of 0 for a shape of no model). A new array on every call.
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
 * Derives a geometry profile from sweep records (and, for the online attention's key-axis form,
 * attention-bench records — `DeriveGeometryProfileOptions.attentionBench`), and serializes a profile
 * as JSON for storage and injection.
 */
export { deriveGeometryProfile, geometryProfileJson } from "./src/tune/derive.ts";
export type {
  AttentionBenchInput,
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
 * The ids of the sweep's case set per op family (`provenance.families[family].caseSet`, and the
 * sweep record's `caseSets`), and the fingerprints of the default table per op family (the record's
 * `defaultKernels`), derived from this runtime without a GPU as 16-digit hexadecimal strings. Given
 * a candidate set, the values a sweep of every case of that set records (an evaluation set's own
 * cases). A record holds the entries of the families it planned cases of, and
 * {@link deriveGeometryProfile} requires them for every family it has rows of.
 */
export { sweepCaseSetIds, sweepDefaultKernelsIds } from "./src/tune/fingerprint.ts";
export type { GeometryProfileFamilyIds } from "./src/tune/families.ts";

/**
 * The kernel fingerprint of the candidates a sweep record measured (the record's
 * `candidateKernels`), derived from its rows and its `dp4a` without a GPU. An application that
 * merges the records of several `runGeometrySweep` calls (one per case, with idle time between
 * them) recomputes it over the merged rows: each record fingerprints only its own rows, and
 * {@link deriveGeometryProfile} rejects a record whose value differs from the one it derives. Such
 * an application also takes the union of the records' `caseSets` and `defaultKernels`: each record
 * holds the entries of the op families of its own cases.
 */
export { sweepCandidateKernelsId } from "./src/tune/fingerprint.ts";

/**
 * Measures the GEMV choices of `linear` on a dedicated device and returns the record
 * (`karume-gemv-sweep/4`): per case (a weight storage, i4 group and `[n, k]` the GEMV family serves
 * in the models) and row count, the baseline — what a Session at the level `exact` runs with the
 * default profile's GEMV choices; for a float-stored case past one row that is the GEMM, built with
 * the tile geometry of the profile the sweep's device resolves to, whose id the record names — and
 * the candidates in two classes: `identical` (the sequential M = 1 variants and
 * the row-block heights, whose output words must be the baseline's) and `reorder` (the parallel GEMV
 * at each lane count, plain, staged and — where the device has the features — subgroup, held to the
 * band of `linear` against an f64 oracle on the host). After a warm start on the baseline, every
 * kernel of a row count gets a quick look; the references and the kernels near enough to matter are
 * timed in interleaved rounds, and the spread of the baseline's rounds says how steady the device
 * was. At M = 4 and 8 the reorder candidates that the same run's M = 1 group of the case leaves
 * unadoptable are skipped (a row without a measurement). The options narrow it by case, row count and candidate (the references are measured either
 * way). The `explore` set measures a short run: the whole grid on a few sentinel cases, first, and
 * on every other case the candidates the sentinels predict. Before the screening every kernel's
 * check is read, and only a kernel whose check held, and that is no subgroup form (no table row
 * holds one), sets its class's bar. It derives no table and changes no choice.
 */
export { runGemvSweep } from "./src/tune/gemv-sweep.ts";
/**
 * The quick look's default margin of both sweeps (`GeometrySweepOptions.quickLookMargin`,
 * `GemvSweepOptions.quickLookMargin`): ×1.25; ×1.10 for the GEMV sweep's `explore` set
 * ({@link EXPLORE_QUICK_LOOK_MARGIN}). {@link gemvSweepQuickLookMargin} is the margin a GEMV sweep
 * of a set runs with.
 */
export { EXPLORE_QUICK_LOOK_MARGIN, QUICK_LOOK_MARGIN } from "./src/tune/measurement.ts";
export { gemvSweepQuickLookMargin } from "./src/tune/gemv-sweep.ts";
/**
 * The GEMV sweep's options and progress: besides its status lines, a `plan` event when a case
 * starts (its row counts and their kernels) and a `step` event per unit of a row count's
 * measurement (a phase and its done / total), for a two-level progress display.
 */
export type {
  GemvSweepOptions,
  GemvSweepPhase,
  GemvSweepPlannedRowCount,
  GemvSweepProgress,
  GemvSweepStep,
} from "./src/tune/gemv-sweep.ts";
/**
 * The `explore` set's exploration: the sentinels of a case table (measured with every candidate,
 * first), the candidates another case measures at M = 1 (the winners of its nearest sentinels of
 * the same storage and their grid neighbours), the distance between two cases' shapes, the grid neighbours of a candidate, and
 * the measuring order of a run. {@link gemvSweepExploreEvidence} reads the sentinels a record
 * measured, which a later sweep takes as `GemvSweepOptions.sentinelEvidence`.
 */
export {
  GEMV_SENTINEL_COUNT,
  gemvCaseDistance,
  gemvExploreOrder,
  gemvExploreSelection,
  gemvGridNeighbours,
  gemvSentinels,
} from "./src/tune/gemv-explore.ts";
export type { GemvExploreEvidence, GemvExploreRow } from "./src/tune/gemv-explore.ts";
export { gemvSweepExploreEvidence } from "./src/tune/gemv-sweep.ts";
/**
 * The GEMV sweep's candidate set: `"quick"` (M = 1, a small identical grid), `"full"` (M = 1, 4
 * and 8 — every row count a derivation rule reads —, the whole grid), `"row-blocks"` (M = 16 and
 * 64, which no rule reads yet) or `"explore"` (the row counts of `"full"`: the whole grid on the
 * sentinels, the predicted candidates on the other cases).
 */
export type { GemvCandidateSet as GemvSweepCandidateSet } from "./src/tune/gemv-candidates.ts";
/** Every case of the GEMV sweep's table (the ids `GemvSweepOptions.cases` takes). A new array on every call. */
export { gemvSweepCaseEntries as gemvSweepCases } from "./src/tune/gemv-cases.ts";
/**
 * Every candidate name of the GEMV sweep's table, at every row count of every set (the names
 * `GemvSweepOptions.candidates` takes). A new array on every call.
 */
export { gemvSweepCandidateNames } from "./src/tune/gemv-candidates.ts";
export type { GemvCase as GemvSweepCase } from "./src/tune/gemv-cases.ts";
/** The GEMV sweep record and its parts. */
export type {
  GemvSweepCaseEntry,
  GemvSweepCaseSummary,
  GemvSweepChoice,
  GemvSweepFamilyChoice,
  GemvSweepGates,
  GemvSweepGemmChoice,
  GemvSweepReport,
  GemvSweepRow,
} from "./src/tune/gemv-report.ts";
/**
 * Derives the GEMV choices of linear (`GeometryProfile.linearGemv`) for a table from GEMV sweep
 * records of its adapter, group by group: per shape and class a decision (adopted or kept, the gains
 * per group that is evidence, why it was kept), the groups whose case is not the current runtime's,
 * the lines that state them, and the records used with how much of each is current and how many of
 * its current groups have no usable reference at `exact` and at `reorder`. The field is absent
 * from the result when every choice stays the default's. A record of another format or adapter, a
 * wall-clock or quantized one, the same record twice and no record at all are refused.
 *
 * {@link planGemvMeasurements} reads the same records and says, per case and row count of the
 * `full` set, the candidates the derivation reads there that fewer than two records could be
 * evidence for — what a sweep must measure for the derivation to be complete.
 */
export { deriveLinearGemvChoices, planGemvMeasurements } from "./src/tune/gemv-derive.ts";
export type {
  DeriveLinearGemvOptions,
  GemvMeasurementPlanning,
  GemvMissingMeasurement,
  GemvSweepInput,
  LinearGemvDecision,
  LinearGemvDerivation,
  LinearGemvGain,
  LinearGemvRecordGroups,
  LinearGemvRowGain,
  LinearGemvStaleGroup,
  PlanGemvMeasurementsOptions,
} from "./src/tune/gemv-derive.ts";

/**
 * Times the s16 and online routes of a8 fused attention at given shapes (default:
 * {@link ATTENTION_BENCH_SHAPES}) on the caller's device and returns the record
 * (`karume-attention-bench/2`). Uses GPU timestamps when the device was acquired with
 * `gpuTiming: true`. `onlineTiled` adds the tiled key axis against the form N gives (perf-ledger
 * K-106); such records are the second input of {@link deriveGeometryProfile}.
 */
export { ATTENTION_BENCH_FORMAT, runAttentionRouteBench } from "./src/tune/attention-bench.ts";
/**
 * The bench's shapes: Wan2.2 TI2V-5B's self- and cross-attention, and the bench's default set,
 * which attention-bench records given to {@link deriveGeometryProfile} must cover.
 */
export { ATTENTION_BENCH_SHAPES, WAN22_ATTENTION_SHAPES } from "./src/tune/attention-shapes.ts";
export type { AttentionBenchShape } from "./src/tune/attention-shapes.ts";
export type {
  AttentionBenchOptions,
  AttentionBenchProgress,
  AttentionBenchReport,
  AttentionBenchRoute,
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

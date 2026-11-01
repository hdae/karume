/**
 * Anima DiT residency (ADR 0112) — the pure decision functions and the state machine of the one
 * resident slot.
 *
 * The default (`"per-stage"`) opens and closes a Session per stage as before. The opt-in
 * `"transformer"` keeps the DiT Session across generate calls and removes the weight reload,
 * re-derivation and backing rebuild from the second generate on (ADR 0112 decision 1).
 *
 * Only "when to keep, when to release, when to evict" lives here; GPU work is called through
 * injected hooks (`open` / `dispose` / `settleRelease`), so GPU-less unit tests can pin the state
 * transitions with fault injection (`tests/anima_residency_test.ts`).
 *
 * ## State
 *
 * | Field        | Meaning                                                                        |
 * | ------------ | ------------------------------------------------------------------------------ |
 * | `held`       | The resident DiT (undefined if none)                                           |
 * | `downgraded` | Has evicted (OOM / low headroom): downgraded for the pipeline's remaining life |
 *
 * MUST: a downgrade is not undone. Undoing it would make a machine short of VRAM hit the OOM again
 * (or probe the headroom) every time before falling to the slow path (paying the eviction cost per
 * generate).
 * MUST: a decision not to keep the DiT resident is always reported (`residency` event). Falling to
 * the slow path silently leaves the user unable to notice that the opt-in has no effect. The only
 * exceptions are a generate in the default per-stage operation unrelated to residency (no event is
 * added to the default behaviour — existing subscribers' event sequences do not change) and folding
 * a DiT created in this generate after a stage failure (a failure before it enters the slot; the
 * residency state does not change — the next generate tries residency again, so nothing has fallen
 * to the slow path).
 *
 * ## The two lines of eviction
 *
 * 1. **Pre-emptive** ({@link TransformerResidency.ensureHeadroom}): while a resident DiT exists,
 *    the amount the next stage needs is measured with a headroom probe, and if it does not fit, the
 *    DiT is evicted before the stage is opened. There are two measuring points in a generate:
 *    - before the text stages (a generate with a carried-over DiT) — the larger of the text_encoder
 *      / text_conditioner needs ({@link generateMemoryNeed}).
 *    - after the DiT stage and before the VAE stage (when a resident DiT exists at that point) —
 *      the vae_decoder need. This point covers the first generate (the DiT is created in this
 *      generate and put in the slot, and the VAE stacks on it) and a generate whose carried-over
 *      DiT ran a new resolution (the plan's backing grows inside the DiT stage, so the headroom
 *      measured before the text stages is stale by the VAE). The VAE is not included in the
 *      measurement before the text stages because evicting later is cheaper (the DiT stage has
 *      already benefited from residency).
 * 2. **Reactive** ({@link TransformerResidency.runStage} / `runTransformer`): when a stage throws
 *    `GpuOutOfMemoryError`, evict and re-run (the second line — when the pre-emptive estimate is
 *    off).
 *
 * An eviction on either line sets the same downgrade (this pipeline does not keep the DiT resident
 * afterwards — the "a downgrade is not undone" MUST above applies to pre-emptive eviction as is).
 *
 * WHY pre-emption is the main line: in WebGPU only `createBuffer` / `createTexture` return OOM
 * without losing the device; an OOM in the staging of a weight upload (`queue.writeBuffer`)
 * invalidates the device (wgpu / Dawn — the head of the runtime's `gpu/headroom.ts`). The reactive
 * line can be too late by the time it hits that OOM.
 */

import { type AdmissionReport, GpuOutOfMemoryError } from "@karume/runtime";
import { ModelInputError } from "../errors.ts";
import { HEADROOM_MARGIN_BYTES } from "../session/headroom-margin.ts";

/**
 * The accepted spellings (the source of truth for the vocabulary — the type is derived from it).
 *
 * NOTE: only the type is on the public surface, not this list (asymmetric with
 * `SAMPLER_TYPES`) — the public surface stays thin (ADR 0008), so the list is added as the
 * same frozen list once a consumer that enumerates the choices (CLI / UI) appears.
 */
const ANIMA_RESIDENCIES = Object.freeze(["per-stage", "transformer"] as const);

/**
 * Whether the DiT Session is kept after a generate.
 *
 * - `"per-stage"` (default): open and close per stage. The VRAM peak is the largest single stage.
 * - `"transformer"`: keep the DiT **after this generate too**. The text / VAE stages stack on the
 *   resident DiT.
 */
export type Residency = (typeof ANIMA_RESIDENCIES)[number];

/** Value when omitted (the old behaviour — the default VRAM assumption does not change). */
export const DEFAULT_ANIMA_RESIDENCY: Residency = "per-stage";

/** Action of the `residency` event. */
export type ResidencyAction =
  /** The DiT is kept after this generate too. */
  | "retained"
  /**
   * The DiT was released (residency was not requested — at the end of the DiT stage, or after a
   * generate that failed before the DiT stage / requested but already downgraded).
   */
  | "released"
  /**
   * The resident DiT was discarded midway (evicted on OOM / the stage of a carried-over DiT
   * failed).
   */
  | "evicted";

/** Reason of the `residency` event. */
export type ResidencyReason =
  /** As the effective residency says (request ?? pipeline default). */
  | "request"
  /**
   * Residency was requested, but this pipeline is downgraded by an eviction (`out-of-memory` /
   * `headroom`).
   */
  | "downgraded"
  /** Evicted because another stage or the run of a carried-over DiT threw `GpuOutOfMemoryError`. */
  | "out-of-memory"
  /**
   * The headroom probe before opening a stage found that the next stage's need does not fit on top
   * of the resident DiT, so it was evicted first (before hitting an OOM —
   * {@link TransformerResidency.ensureHeadroom}). There are two measuring points: before the text
   * stages (the need of the two text stages) and after the DiT stage, before the VAE stage (the
   * VAE's need).
   */
  | "headroom"
  /**
   * The DiT stage using a carried-over resident DiT failed (including an abort by `onEvent`
   * throwing in the denoise loop), so the possibly broken Session was discarded.
   */
  | "failure";

/** Notification the state machine emits (the pipeline maps it to a `residency` event). */
export type ResidencyNotice = {
  readonly action: ResidencyAction;
  readonly reason: ResidencyReason;
};

/**
 * Checks the spelling of `residency` (the entry for both the construction option and the request).
 *
 * MUST: an unknown spelling fails instead of falling back to the default — if a misspelled
 * `"transformers"` silently became per-stage operation, the user could notice that residency has no
 * effect only from the speed. The fix is on the caller's side, so it is input-caused (ADR 0107).
 */
export const assertAnimaResidency = (value: unknown, where: string): Residency => {
  const accepted =
    typeof value === "string"
      ? ANIMA_RESIDENCIES.find((candidate) => candidate === value)
      : undefined;
  if (accepted === undefined) {
    throw new ModelInputError(
      `${where}: expected ${ANIMA_RESIDENCIES.map((name) => `'${name}'`).join(" / ")}` +
        ` (actual ${typeof value === "string" ? `'${value}'` : String(value)})`,
    );
  }
  return accepted;
};

/** Where the DiT stage's Session comes from. */
export type TransformerSource =
  /** Use the resident DiT carried over from an earlier generate (no reload). */
  | "resident"
  /**
   * Create a new one and put it in the resident slot if the stage succeeds (it enters the slot
   * after the stage — a failure inside the stage is the same as per-stage operation).
   */
  | "build-resident"
  /** Create and fold per stage (not put in the resident slot). */
  | "per-stage";

/** Input of the decision (the facts at the **start** of the DiT stage). */
export type ResidencyFacts = {
  /** Effective residency of this generate (request ?? pipeline default). */
  readonly effective: Residency;
  /** Whether already downgraded by an eviction (OOM / not enough headroom). */
  readonly downgraded: boolean;
  /**
   * Whether a DiT is in the resident slot (= a DiT carried over from an earlier generate exists).
   */
  readonly held: boolean;
};

/**
 * Where to take the DiT stage's Session from.
 *
 * MUST: if a resident DiT already exists, use it whatever the effective value (no reload — the
 * effective value means "keep it **after** this generate", not "use it in this generate").
 */
export const transformerSource = (facts: ResidencyFacts): TransformerSource => {
  if (facts.held) return "resident";
  return facts.effective === "transformer" && !facts.downgraded ? "build-resident" : "per-stage";
};

/**
 * Outcome of the DiT stage (an OOM of a carried-over DiT takes another path —
 * {@link TransformerResidency} evicts and re-runs).
 */
export type TransformerOutcome = "success" | "failure";

/** Decision on what to do with the DiT in the slot. */
export type TransformerDecision = {
  /**
   * `retain` = put in the slot and keep / `release` = release and fold / `evict` = discard midway
   * and fold (the GPU side is the same as `release`; only the reported reason differs).
   */
  readonly action: "retain" | "release" | "evict";
  /** Event to emit (none in the default per-stage operation unrelated to residency). */
  readonly notice?: ResidencyNotice;
};

/**
 * Decides, when leaving the DiT stage, what to do with the DiT the stage used (`facts` are the
 * facts at the **start** of the stage — the source is derived with {@link transformerSource}). The
 * state machine applies this `action` as is (the single source of truth).
 *
 * - Failure: a carried-over DiT (`resident`) is discarded and reported (a broken Session is not
 *   carried into the next generate — a Session after an OOM or device loss is not necessarily
 *   usable). A DiT created in this generate is the same as a per-stage failure: folded without a
 *   report (a failure before it enters the slot — from the user's point of view residency has not
 *   changed).
 * - Success: kept if the effective value is `transformer` and not downgraded. When not kept, the
 *   reason is reported. Only per-stage operation unrelated to residency is not reported (the
 *   default event sequence does not change).
 *
 * MUST: an unreachable combination throws instead of falling back to a default value (silently
 * returning "release without a report" would leave a value that contradicts the MUST that a
 * decision not to keep the DiT is always reported). A downgraded state never has a DiT in the slot
 * — the downgrade is set only when an eviction empties the slot, and nothing enters the slot after
 * it.
 */
export const decideAfterTransformer = (
  facts: ResidencyFacts,
  outcome: TransformerOutcome,
): TransformerDecision => {
  if (facts.held && facts.downgraded) {
    throw new Error(
      "anima: a DiT is in the resident slot although residency is already downgraded (unreachable state — the state machine is broken)",
    );
  }
  const source = transformerSource(facts);
  if (outcome === "failure") {
    return source === "resident"
      ? { action: "evict", notice: { action: "evicted", reason: "failure" } }
      : { action: "release" };
  }
  switch (source) {
    case "build-resident":
      return { action: "retain", notice: { action: "retained", reason: "request" } };
    case "resident":
      // A carried-over DiT used with effective per-stage reports its release (the observation point
      // of the usage that makes the last of a series per-stage to release it).
      return facts.effective === "transformer"
        ? { action: "retain", notice: { action: "retained", reason: "request" } }
        : { action: "release", notice: { action: "released", reason: "request" } };
    case "per-stage":
      // Empty slot and effective transformer, yet per-stage operation = downgraded (the
      // contrapositive of transformerSource).
      return facts.effective === "transformer"
        ? { action: "release", notice: { action: "released", reason: "downgraded" } }
        : { action: "release" };
  }
};

/**
 * Decides what to do with the DiT in the slot when a generate fails outside the DiT stage (called
 * only when the slot holds a DiT).
 *
 * - Effective `per-stage`: release and report `released` / `request` — the instruction "do not keep
 *   it after this generate" takes effect even when the generate fails (otherwise VRAM thought
 *   released stays until the next generate or dispose).
 * - Effective `transformer`: keep it without a report (if the DiT stage was left, `retained` has
 *   been reported; before the DiT stage the residency state has not changed).
 */
export const decideAfterFailedGenerate = (effective: Residency): TransformerDecision =>
  effective === "per-stage"
    ? { action: "release", notice: { action: "released", reason: "request" } }
    : { action: "retain" };

/**
 * Bytes one stage needs on top of the resident DiT = estimated peak + staging of the largest part +
 * margin.
 *
 * - `report.peakAccountedBytes`: weights + the bound of the retained set (`ModelComponent.estimate`
 *   — the bindings and device limits are the same as the stage's actual arguments).
 * - `maxPartBytes`: the staging alive at the same time during construction (the WHY of
 *   `ModelComponent.maxPartBytes`).
 * - {@link HEADROOM_MARGIN_BYTES}: the extra for what is outside the accounting.
 *
 * NOTE: `export` is for the tests that pin the arithmetic without a GPU (not on the public
 * surface).
 */
export const stageMemoryNeed = (
  report: Pick<AdmissionReport, "peakAccountedBytes">,
  maxPartBytes: number,
): number => report.peakAccountedBytes + maxPartBytes + HEADROOM_MARGIN_BYTES;

/**
 * Bytes a group of consecutive stages (text_encoder / text_conditioner for the measurement before
 * the text stages) needs on top of the resident DiT = the **maximum** of the per-stage needs.
 *
 * WHY the maximum, not the sum: stages run one at a time, and each stage's Session is folded before
 * the next stage is opened (the finally of the pipeline's `withSession`). Only the resident DiT
 * (constant during the text stages — already in place at the probe) and one stage coexist, so if
 * the heaviest stage fits, the rest fit. Across the DiT stage the resident DiT's backing can grow,
 * so the VAE stage is not in this group and is measured separately after the DiT stage (module doc,
 * "The two lines of eviction").
 *
 * NOTE: `export` is for the tests that pin the arithmetic without a GPU (not on the public
 * surface).
 */
export const generateMemoryNeed = (needs: readonly number[]): number => Math.max(0, ...needs);

/**
 * Whether a failure calls for an OOM eviction (there is something to evict = a DiT in the resident
 * slot, and the failure is `GpuOutOfMemoryError`).
 *
 * MUST: decide by type (do not look at the message) — only running out of memory, caught by the
 * errorScope, is a failure eviction can fix; re-running a validation error or device loss after an
 * eviction buries the real cause under the second failure.
 */
export const evictsForMemory = (error: unknown, held: boolean): boolean =>
  held && error instanceof GpuOutOfMemoryError;

/**
 * Hooks the state machine asks of the GPU side (in production, thin wrappers of Session and
 * GpuContext).
 */
export type ResidencyHooks<T> = {
  /** Disposes the resource that was kept resident. */
  readonly dispose: (resource: T) => Promise<void> | void;
  /**
   * After disposing on an eviction (OOM / not enough headroom), waits for the release to reach the
   * device. On Intel / wgpu the release by `destroy()` is deferred to the next device poll
   * (docs/known-issues.md, Intel Arc B570 section), so re-running without waiting can hit the same
   * OOM (the grounds for the shape of the wait and the unverified range are in the doc of the
   * pipeline-side implementation). The wait also surfaces the loss of a device already invalidated
   * by a fatal OOM (the pipeline-side implementation issues an empty submit before waiting — a
   * validated call), so a dead device becomes a loss exception here, before the next stage.
   * Pre-emptive eviction ({@link TransformerResidency.ensureHeadroom}) has no re-run — after the
   * wait the next stage is opened as is.
   */
  readonly settleRelease: () => Promise<void>;
};

/** Receiver of notifications (mapped to the pipeline's `onEvent`). */
export type ResidencyNotify = (notice: ResidencyNotice) => Promise<void>;

/** Input of one DiT stage. */
export type TransformerStage<T, R> = {
  readonly effective: Residency;
  /** Creates the DiT Session. */
  readonly open: () => Promise<T>;
  /** The stage body (the denoise loop). A re-run calls it again from the start. */
  readonly body: (resource: T) => Promise<R>;
  readonly notify: ResidencyNotify;
};

/**
 * State machine of the one resident slot (DiT). The pipeline holds one and calls it only from
 * inside the serialisation chain of `generate` / `dispose` (concurrent calls are not expected — the
 * chain orders them one at a time).
 *
 * What to do with the slot is decided by pure functions ({@link decideAfterTransformer} /
 * {@link decideAfterFailedGenerate}); this only applies their `action` and emits notifications. The
 * exception is eviction (reactive on OOM {@link evictsForMemory} / pre-emptive on insufficient
 * headroom {@link TransformerResidency.ensureHeadroom}), which lives here because it reacts to GPU
 * facts rather than being a decision.
 */
export class TransformerResidency<T> {
  readonly #hooks: ResidencyHooks<T>;
  #held: T | undefined;
  #downgraded = false;

  constructor(hooks: ResidencyHooks<T>) {
    this.#hooks = hooks;
  }

  /** Whether a resident DiT exists (for diagnostics and tests). */
  get holding(): boolean {
    return this.#held !== undefined;
  }

  /** Whether downgraded by an eviction (OOM / not enough headroom) (for diagnostics and tests). */
  get downgraded(): boolean {
    return this.#downgraded;
  }

  /**
   * Runs one stage other than the DiT. If the stage throws `GpuOutOfMemoryError` while a resident
   * DiT exists, the resident DiT is evicted and **the stage is re-run once** (`attempt` is called
   * again — from Session construction). A failure of the re-run is thrown as is (nothing is left to
   * evict).
   */
  async runStage<R>(attempt: () => Promise<R>, notify: ResidencyNotify): Promise<R> {
    try {
      return await attempt();
    } catch (error) {
      if (!evictsForMemory(error, this.holding)) throw error;
      await this.#evict("out-of-memory", notify, { error });
      return await attempt();
    }
  }

  /**
   * Runs the DiT stage (the Session's source is {@link transformerSource}; on leaving, the `action`
   * of {@link decideAfterTransformer} is applied).
   *
   * If the run of a carried-over DiT throws `GpuOutOfMemoryError`, it is evicted and **the stage is
   * re-run once from the start** in per-stage operation. The stage body rebuilds the denoise state,
   * so the output does not change (same seed, same inputs, same graph).
   *
   * A DiT created in this generate does not enter the slot until the stage succeeds and `retain` is
   * decided. A failure of its run is not evicted even on OOM (no downgrade or notification; it is
   * folded and the original error thrown). WHY: the pipeline then holds only this DiT as GPU
   * resources (the text stages are already folded), the same VRAM layout as per-stage operation —
   * the OOM is not caused by residency, so downgrading and reloading the GB-scale weights would
   * only retry the same layout. An OOM in constructing the DiT itself is likewise not evicted
   * (nothing to evict).
   */
  async runTransformer<R>(stage: TransformerStage<T, R>): Promise<R> {
    const facts = this.#facts(stage.effective);
    const carried = this.#held;
    const resource = carried ?? (await stage.open());
    let result: R;
    try {
      result = await stage.body(resource);
    } catch (error) {
      if (evictsForMemory(error, carried !== undefined)) {
        await this.#evict("out-of-memory", stage.notify, { error });
        // The eviction emptied the slot and set the downgrade, so the re-run is per-stage
        // operation, which does not evict (nothing to evict = structurally only one re-run).
        return await this.runTransformer(stage);
      }
      throw await this.#settleFailure(
        error,
        decideAfterTransformer(facts, "failure"),
        resource,
        stage.notify,
      );
    }
    const decision = decideAfterTransformer(facts, "success");
    await this.#apply(decision.action, resource);
    if (decision.notice !== undefined) await stage.notify(decision.notice);
    return result;
  }

  /**
   * Called when a generate fails outside the DiT stage (the failure path of the pipeline's
   * `#generate` — a failure inside the DiT stage is settled in {@link runTransformer}). If the slot
   * holds a DiT, {@link decideAfterFailedGenerate} is applied. Returns the exception to throw (the
   * MUST of {@link #settleFailure}).
   */
  async releaseIfRequested(
    error: unknown,
    effective: Residency,
    notify: ResidencyNotify,
  ): Promise<unknown> {
    const held = this.#held;
    if (held === undefined) return error;
    return await this.#settleFailure(error, decideAfterFailedGenerate(effective), held, notify);
  }

  /**
   * Pre-emptive eviction (line 1 of the module doc's "The two lines of eviction"): `probe` measures
   * whether `need` bytes fit on top of the resident DiT, and if not: downgrade → dispose the
   * resident DiT → wait for the release → report `evicted` / `headroom`. The pipeline calls it
   * before opening the next stage (before the text stages / before the VAE stage).
   *
   * - No resident DiT: nothing happens (`probe` is not called either — nothing to evict, so nothing
   *   to measure).
   * - `probe` returns true: nothing happens (residency continues).
   * - `probe` returns false: evict. If the clean-up (dispose, release wait) fails, report and then
   *   throw an `AggregateError` listing the clean-up failures — the only difference from the OOM
   *   eviction is that no original failure leads the list.
   * - `probe` throws: thrown as is (no downgrade or disposal; the resident DiT is kept). WHY: a
   *   validation error (a bug) or device loss is not "does not fit" — handling it by eviction would
   *   bury the real cause. What happens to the DiT in the slot is decided by the generate's failure
   *   path ({@link releaseIfRequested}).
   */
  async ensureHeadroom(
    need: number,
    probe: (bytes: number) => Promise<boolean>,
    notify: ResidencyNotify,
  ): Promise<void> {
    if (!this.holding) return;
    if (await probe(need)) return;
    await this.#evict("headroom", notify, undefined);
  }

  /**
   * Releases the resident DiT (from the pipeline's `dispose` — called inside the chain **before**
   * the GPU is destroyed; flush-before-destroy). Nothing happens without a resident DiT.
   */
  async dispose(): Promise<void> {
    await this.#release();
  }

  #facts(effective: Residency): ResidencyFacts {
    return { effective, downgraded: this.#downgraded, held: this.holding };
  }

  /**
   * Applies the decision's `action` to the DiT (`retain` = put in the slot — a carried-over one is
   * already there / `release`, `evict` = take out of the slot and dispose).
   *
   * MUST: empty the slot **before** disposing (so the next generate does not pick up a Session
   * whose disposal failed).
   */
  async #apply(action: TransformerDecision["action"], resource: T): Promise<void> {
    if (action === "retain") {
      this.#held = resource;
      return;
    }
    if (this.#held === resource) this.#held = undefined;
    await this.#hooks.dispose(resource);
  }

  /**
   * Clean-up after a failure: apply the decision, then report, and return the exception to throw.
   *
   * MUST: do not overwrite the original failure. Failures of the clean-up (dispose, notification)
   * are listed in an `AggregateError`. When a notification subscriber rethrows **the same exception
   * as the original failure** (the abort pattern that calls `signal.throwIfAborted()` every time),
   * it is not a new failure and is not listed — listing it would stop the consumer from recognising
   * its own abort with `error === signal.reason`.
   */
  async #settleFailure(
    error: unknown,
    decision: TransformerDecision,
    resource: T,
    notify: ResidencyNotify,
  ): Promise<unknown> {
    const notice = decision.notice;
    return await withCleanupFailures(error, [
      () => this.#apply(decision.action, resource),
      ...(notice === undefined ? [] : [() => notify(notice)]),
    ]);
  }

  /**
   * Eviction (reactive on OOM / pre-emptive on insufficient headroom — one shared implementation):
   * downgrade → dispose the resident DiT → wait for the release → report `evicted` / `reason`.
   * `cause` is the original failure of a reactive eviction (undefined for a pre-emptive one).
   *
   * MUST: set the downgrade first (even if disposal fails, this pipeline does not try residency
   * again).
   * MUST: empty the slot **before** disposing (so the next generate does not pick up a Session
   * whose disposal failed).
   * MUST: even if the clean-up (dispose, release wait) fails, report `evicted` before throwing —
   * the downgrade is set and the slot is empty, so without the report the user would not notice the
   * loss of residency until the next generate. A failure of that notification is listed too
   * (`AggregateError` — led by the original failure if there is one). If the clean-up passes, a
   * throw from the notification is thrown as is (the means of aborting).
   */
  async #evict(
    reason: "out-of-memory" | "headroom",
    notify: ResidencyNotify,
    cause: { readonly error: unknown } | undefined,
  ): Promise<void> {
    this.#downgraded = true;
    // Without an original failure (pre-emptive) there is no "rethrow of the original failure"
    // either — compare against a unique marker.
    const original = cause === undefined ? NO_ORIGINAL_FAILURE : cause.error;
    const failures = await cleanupFailures(original, [
      () => this.#release(),
      () => this.#hooks.settleRelease(),
    ]);
    const notice: ResidencyNotice = { action: "evicted", reason };
    if (failures.length === 0) {
      await notify(notice);
      return;
    }
    failures.push(...(await cleanupFailures(original, [() => notify(notice)])));
    throw new AggregateError(
      cause === undefined ? failures : [cause.error, ...failures],
      CLEANUP_FAILED,
    );
  }

  async #release(): Promise<void> {
    const held = this.#held;
    this.#held = undefined;
    if (held !== undefined) await this.#hooks.dispose(held);
  }
}

/**
 * Marker passed to {@link cleanupFailures} for an eviction without an original failure
 * (pre-emptive); matches no exception.
 */
const NO_ORIGINAL_FAILURE: unique symbol = Symbol("no original failure");

/**
 * Message of the `AggregateError` listing clean-up failures (used for both the per-stage DiT and
 * the resident DiT).
 */
const CLEANUP_FAILED = "anima: DiT clean-up failed";

/**
 * Runs every clean-up and collects the failures. A rethrow of the same exception as the original
 * failure is not listed (the MUST of `#settleFailure` in {@link TransformerResidency}).
 */
const cleanupFailures = async (
  error: unknown,
  cleanups: readonly (() => Promise<void> | void)[],
): Promise<unknown[]> => {
  const failures: unknown[] = [];
  for (const cleanup of cleanups) {
    try {
      await cleanup();
    } catch (cleanupError) {
      if (cleanupError !== error) failures.push(cleanupError);
    }
  }
  return failures;
};

/**
 * Runs every clean-up and returns an exception led by the original failure (the original failure
 * itself if every clean-up passes).
 */
const withCleanupFailures = async (
  error: unknown,
  cleanups: readonly (() => Promise<void> | void)[],
): Promise<unknown> => {
  const failures = await cleanupFailures(error, cleanups);
  return failures.length === 0 ? error : new AggregateError([error, ...failures], CLEANUP_FAILED);
};

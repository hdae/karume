/**
 * The generation object that lives for one conversation (`GenerationSequence` of ADR 0083
 * decisions 1 to 5). It is **pipeline-independent shared code**, so it lives in `src/generation/`
 * next to `program.ts` / `sampler.ts`.
 *
 * ## Mutable state is `context` and `pendingToken` (+ speculation's hidden copy and uncommitted rows)
 *
 * MUST: the mutable state of **progress** the sequence holds is **only the two, `context` and
 * `pendingToken`** (ADR 0083 decision 1 — the hidden copy, the counters and `frontierRows` that
 * speculation adds are not records of progress: the two paragraphs below). It keeps no counter such
 * as position / totalLength and reads `context.pastLength` right before building a run — the
 * context advances the logical length when a run succeeds (ADR 0066 decision 6's **ban on double
 * bookkeeping**). The `position` on events is also read on the spot from `context.pastLength`
 * **after** that run, and not stored.
 *
 * The speculative path (ADR 0096 stage 3) adds to this **a copy of the hidden passed to the
 * drafter** (one row — a readback is valid only until the next run and cannot be derived) and the
 * counters (`GenerationStop.speculation`). The record of progress is still only
 * `context.pastLength`; verify is a deferred run (the logical length is held), and
 * `context.commit(rows)` advances **up to the frontier of the delivered tokens** (the "speculative
 * path" section below).
 *
 * Only the row count passed to that `commit` (`frontierRows`) becomes sequence-scoped mutable
 * state — how far into the pending run the target consumed cannot be derived from the context (the
 * context knows only the rows written, `queryLength`), and the `used` getter has to add it to the
 * logical length during delivery. It is not a progress counter (it returns to 0 right after the
 * commit, and progress remains only in `context.pastLength`).
 *
 * ## `pendingToken` — the uncommitted frontier of at most one token
 *
 * A `GenerationContext` always holds an uncommitted frontier of at most one token (after K
 * generated tokens `pastLength = T + K − 1`, because decode runs only `maxNewTokens − 1` times). So
 * the next turn **prefills with `pendingToken` prepended to the new prompt** (ADR 0083 decision 4).
 * Dropping the prepend makes "the last token of the previous assistant turn disappear from the
 * history" — silent degradation without an exception, and the check on this path is the pass line
 * of stage 3.
 *
 * MUST: update `pendingToken` **right after choosing** (before the yield). `break` / `return()` only
 * enter `finally`, from where there is no way to reconstruct "how far it got". Written at the
 * moment of choosing, the value is correct whichever yield the interruption happens at.
 *
 * MUST: `rewind` is not used — contexts with sliding slots are always rejected (ADR 0066 addendum
 * 2). Editing (rewriting an earlier utterance) and changing the capacity are properly "a new
 * context + token transcript replay". Branching is {@link GenerationSequence.fork} (section below).
 *
 * ## Branching (fork — the ADR 0066 / 0083 addenda of 2026-10-11, ruling 8)
 *
 * `fork()` returns an independent sequence that continues the same conversation. It copies all
 * mutable state of progress — the context (the runtime's `GenerationContext.fork` = a copy of every
 * byte of every slot and of the logical length) and `pendingToken` — plus the self-tuning gate's
 * observed state ({@link SpeculationGate.fork}). The drafter face is reopened on the copied context
 * with `speculative.open` (the borrower goes with the context — sharing the original borrower would
 * make the branch's drafts read the original KV). The RNG is not copied (each `generate` builds it
 * from the seed). `frontierRows` is 0 outside `generate`, so it needs no copy.
 *
 * MUST: the three things copied (frontier, gate, context) are captured at **the same point** inside
 * the serialization chain's slot. The slot is taken after any in-flight `generate`, so that point
 * is "the settled values up to the most recently completed generation".
 *
 * ## An interruption closes in "the state advanced by the completed runs"
 *
 * Whether closed by `break` / `return()` / `AbortSignal`, the conversation has advanced **by the
 * successful runs**. An interruption before any token was received (= during prefill) leaves the
 * prompt partly in the conversation, and the `chunk` of the `prefill` event gives the number of
 * committed chunks. Whether to send the rest or discard the sequence is the caller's decision
 * (managing the conversation is the host's job — the same line as decision 10).
 *
 * ## `AbortSignal` is the way to cancel
 *
 * It is checked at stage boundaries (right before each run), and `signal.reason` is thrown **as is,
 * unwrapped** (ADR 0083 decision 5 — the precedent is `anima.PipelineOptions.signal`).
 *
 * ## The speculative path (ADR 0096 stage 3 — when `options.speculative` is given, at any temperature)
 *
 * 1 cycle = draft (a borrower run, `k' = min(k, remaining budget − 1)` drafts) → verify (the
 * lender's deferred run, the `k'+1` rows `[b, d₁..d_k']`) → acceptance (synchronous, matching
 * prefix, **the enumeration stops at a stop token** — `speculation.ts`) → the confirmed sequence is
 * delivered one by one (`pendingToken` / `generated` / `history` are updated before the yield — the
 * existing MUST) → **`settleCommit()` commits up to the frontier** (rows consumed = the number of
 * tokens made frontier).
 *
 * MUST: call `settleCommit()` both right after the delivery loop and in the generator's `finally`
 * (it commits once if something is pending). A consumer's `break` / `return()` commits "up to the
 * delivered tokens" in finally, and an exception in the synchronous span between verify returning
 * and delivery becomes `commit(0)` (the frontier `b` stays out of the KV; rejected rows are
 * overwritten by the next run) — both close to "conversation = the sequence the consumer received +
 * one frontier", the same shape as non-speculative decoding (leaving it pending would make the
 * context accept only dispose).
 * MUST: no `await` between acceptance and the first yield (the readback subarray is valid until the
 * next run — the lending Session can be shared with other sequences). The hidden for the next cycle
 * is copied.
 * MUST: `k + 1 ≤ context.slidingSlack` (the borrower reads up to the oldest column of the sliding
 * ring — the runtime's check is `queryLength ≤ slidingSlack`, and `k` is kept inside it).
 *
 * ## The self-tuning gate (ADR 0096 stage 4-B (4) — the default of `policy: "auto"`)
 *
 * What speculation gains depends on the task and the host, and some combinations lose (measurements
 * in the doc of `speculation-gate.ts`). The gate measures the cycle wall and the plain-step wall
 * itself and, while losing, returns `"plain"` at the head of a cycle. That cycle goes through **the
 * existing `drafted === 0` path** as is (verify shape M=1, R=1, deferred, the one row `[b]`) — it
 * does not branch to the non-speculative decode path because all three of these are needed:
 * (1) it goes through `readHidden`, so the drafter input (hidden) stays connected for returning to
 * speculate (the decode path reads only logits); (2) it is deferred, so if the draw throws,
 * `commit(0)` keeps the frontier from being fed twice; (3) M=1, R=1 is identical to decode down to
 * the PreparedPlan key, so the GPU cost is the same.
 *
 * Gate-induced plain steps **are not counted in the speculation counters** (they do not pollute the
 * denominators of `cycles` / `draftRuns` / `drafted` / `accepted` / `delivered` /
 * `acceptedHistogram`). They are counted in {@link GenerationSpeculation.plainSteps} and appear in
 * the observation hook as `{ kind: "decode", step }` (no branch added to the public type = consumers
 * with exhaustive switches do not break). The forced plain at the end of the budget (`k' = 0`) is
 * counted as verify as before, and is **neither submitted to the gate nor observed**.
 *
 * MUST: two kinds of cycles are not mixed into the wall observations — the **first cycle** of each
 * turn (it includes PLE gather cold misses) and **the forced plain at the end of the budget** (a
 * cycle whose shape the caller's budget decided, not a matter of how well speculation does). For
 * those two, `SpeculationGate.skip()` is called instead of an observation (only the exploration
 * period advances — even a cycle that is not mixed in runs one piece of GPU work).
 *
 * ## The decode run-ahead (`options.greedyAhead` — where the per-step greedy selection would run)
 *
 * One call runs `n = min(depth, remaining)` decode steps in one GPU batch (step k at position
 * `pastLength + k`, fed step k−1's on-GPU selection) and leaves the `n` rows pending, like a verify.
 * The tokens are then handed out one by one with **the per-step loop's bookkeeping** (`pendingToken`
 * / `generated` / `history` updated before each yield, the stop check, the yield position), and
 * `settleCommit()` commits the rows the per-step loop would have committed by then: up to the first
 * stop token, the token at which a consumer closed, or the draw that threw. The rows after it were
 * computed but are never committed (the next run overwrites them — the deferred contract). The
 * abort boundary is the batch; a token of a batch already run is checked synchronously before it is
 * handed out. A failed batch commits nothing (the frontier stays `pendingToken`, as for a failed
 * per-step run).
 */

import { closeableGenerator } from "../concurrency/closeable-generator.ts";
import { ModelInputError } from "../errors.ts";
import type {
  GenerationContext,
  GenerationContextSpec,
  RunInputs,
  RunOutputs,
  SymbolBindings,
  Tensor,
} from "@karume/runtime";
import { settleAbort } from "../concurrency/abort.ts";
import { createOperationChain } from "../concurrency/serial.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { planPrefillChunks } from "./greedy.ts";
import { createSampler, isStopToken, type SamplerSpec } from "./sampler.ts";
import type { GenerationWiring } from "./program.ts";
import {
  acceptDrafts,
  createSpeculationTally,
  type DraftFace,
  planDraftLength,
  type SpeculationTally,
  takeDrafts,
  verifyRowIndices,
} from "./speculation.ts";
import {
  createSpeculationGate,
  type SpeculationDecision,
  type SpeculationGate,
  type SpeculationGateOptions,
} from "./speculation-gate.ts";

/** The limit {@link GenerationCapacityError} hit (both mean "it no longer fits"). */
export type GenerationCapacityConstraint = "capacity" | "maxPosition";

/**
 * The actual values {@link GenerationCapacityError} carries (the observation **at the moment**
 * `assertBudget` fails).
 *
 * MUST: all are values derived on the spot (not built from a stored counter = the ban on double
 * bookkeeping).
 * MUST: do not make callers parse the message. Passing the numbers truncation needs as fields is
 * the point of having a dedicated type.
 */
export type GenerationCapacityDetail = {
  /** The limit hit (`capacity` = state capacity / `maxPosition` = rows of the position table). */
  readonly constraint: GenerationCapacityConstraint;
  /**
   * The conversation's logical length when checked (`GenerationSequence.used` minus the uncommitted
   * frontier).
   */
  readonly pastLength: number;
  /**
   * The number of tokens this turn feeds — **after prepending `pendingToken`**.
   *
   * It can be one more than the `prompt` the caller passed (prepended when the previous turn
   * produced a token). Compute truncation with this value (the same one token is prepended on the
   * next turn too).
   */
  readonly promptLength: number;
  /**
   * `maxNewTokens` of the failed request.
   *
   * A request without one fails only when its prompt alone does not fit, and reports 1 (the smallest
   * turn there is). `ChatSession`'s up-front check of a turn without one reports the reserve it
   * could not keep (`ChatSessionOptions.reserveNewTokens`).
   */
  readonly requestedNewTokens: number;
  /** The value of the limit `constraint` points at. */
  readonly limit: number;
  /**
   * The largest `maxNewTokens` that fits **now** with this `promptLength`.
   *
   * 0 or less means the prompt alone does not fit (a negative value keeps "by how many tokens it
   * overflows" — a guide for truncation). Both constraints fold into
   * `K ≤ limit − pastLength − promptLength + 1` (`capacity` is an inclusive limit and `maxPosition`
   * an exclusive one, so the two formulas of `assertBudget` take exactly the same shape).
   */
  readonly maxNewTokens: number;
};

/**
 * The capacity was exceeded (= this conversation no longer fits) — ADR 0083 decision 10.
 *
 * MUST: fail with a dedicated type. The runtime also rejects `pastLength + queryLength ≤ capacity`
 * violations, but with a generic message right before encoding the run, and the caller would have
 * to tell "truncating makes it pass" from "the wiring is broken" by reading the text. **Truncating
 * the conversation is the host's job** (limitations), so only that one case, which the host decides
 * on, is separated by type.
 *
 * Exceeding the position-table limit (`maxPosition`) fails with the same type — for the caller it
 * is the same "no longer fits", and the remedy (drop old turns / build a new context) is the same.
 * Which one was hit, and the values needed to compute truncation from it, are carried by the fields
 * of {@link GenerationCapacityDetail}.
 */
export class GenerationCapacityError extends ModelInputError {
  readonly constraint: GenerationCapacityConstraint;
  readonly pastLength: number;
  readonly promptLength: number;
  readonly requestedNewTokens: number;
  readonly limit: number;
  readonly maxNewTokens: number;

  constructor(message: string, detail: GenerationCapacityDetail) {
    super(message);
    this.name = "GenerationCapacityError";
    this.constraint = detail.constraint;
    this.pastLength = detail.pastLength;
    this.promptLength = detail.promptLength;
    this.requestedNewTokens = detail.requestedNewTokens;
    this.limit = detail.limit;
    this.maxNewTokens = detail.maxNewTokens;
  }
}

/** Events during generation (ADR 0083 decision 2). */
export type GenerationEvent =
  | {
      readonly kind: "token";
      /** The chosen token id. */
      readonly id: number;
      /**
       * The absolute position where this token sits in the conversation (non-speculative: = the
       * `context.pastLength` right after; speculative: one verify confirms several tokens, so
       * `pastLength + 1 + i` before the commit).
       */
      readonly position: number;
    }
  | {
      readonly kind: "prefill";
      /**
       * Number of **committed** prefill chunks (1-based — `chunk / chunks` is the progress as is).
       */
      readonly chunk: number;
      readonly chunks: number;
    };

/**
 * The stop reason (returned by {@link GenerationStream.done}).
 *
 * Two branches close on a stop token, and both carry **the stop token itself** — `eos` is the set
 * the distribution declared (`GenerationWiring.stopTokens`), and `stop-token` is the set only that
 * request added ({@link GenerationRequest.stopTokens}). An id in both closes as `eos` (its meaning
 * as the distribution's terminator wins). A stop token is not emitted as a `token` event (it is a
 * terminator, not body text; in chat it is a formatting token such as `<turn|>`), but it stays in
 * the conversation = it is the `pendingToken` prepended to the next turn's prefill.
 *
 * `closed` is when the consumer closed with `break` / `return()` (`aborted` is via `AbortSignal`).
 *
 * The two limit branches are told apart by the request, not by the numbers, so a UI can say which
 * one happened: `max-tokens` is a request **with** `maxNewTokens` that drew all of it (even when it
 * equals the room the conversation had), and `capacity` is a request **without** one that filled the
 * conversation (the context is full).
 */
export type GenerationStop = {
  /**
   * The number of tokens this turn **generated** (the prompt is not included).
   *
   * MUST: a stop token (`eos` / `stop-token`) counts as one. Non-speculatively one draw = one run,
   * so this number equals the runs spent on generation, and `tok/s` can be written without
   * re-encoding (the purpose of this field). With speculation one verify run confirms up to `k+1`
   * tokens, so read the run count as `speculation.cycles + speculation.draftRuns` (`+ plainSteps`
   * for gated turns). For the body-only count (= the number of `token` events), subtract 1 for the
   * two branches that carry a stop token (`"eos"` / `"stop-token"`).
   *
   * For `max-tokens` it equals the request's `maxNewTokens`; for `capacity` the limit the turn
   * resolved (the doc of {@link GenerationRequest.maxNewTokens}); for `closed` / `aborted` it is the
   * number produced before the cut (both mean "the conversation advanced by the successful runs" —
   * the same line as the section above. Speculation also commits only up to the delivered tokens,
   * so this number = the generated tokens that entered the conversation).
   */
  readonly tokens: number;
  /**
   * Speculation counters (present only on turns of a sequence with `options.speculative` — absent
   * altogether on a non-speculative sequence).
   */
  readonly speculation?: GenerationSpeculation;
  /**
   * Decode run-ahead counters (present only on turns of a sequence with `options.greedyAhead` —
   * absent altogether otherwise).
   */
  readonly decodeAhead?: GenerationDecodeAhead;
} & (
  | { readonly reason: "eos"; readonly token: number }
  | { readonly reason: "stop-token"; readonly token: number }
  | { readonly reason: "max-tokens" }
  | { readonly reason: "capacity" }
  | { readonly reason: "aborted" }
  | { readonly reason: "closed" }
);

/**
 * Decode run-ahead counters ({@link GenerationStop.decodeAhead}) — which decode path the turn took.
 *
 * Per-token observation (`onRun`) is not part of it: the run-ahead runs only where the per-step
 * greedy selection would, and that selection already requires a sequence without `onRun` (the
 * diagnostics of one run cannot be read from a batch of several), so these counters are the only
 * place the path shows.
 */
export type GenerationDecodeAhead = {
  /** The depth the sequence was built with (the most decode steps one batch runs). */
  readonly depth: number;
  /**
   * Batches the turn ran (each pays one fence and one readback for up to `depth` tokens). `0` = the
   * turn's decode did not take the run-ahead: its sampler was not plain greedy (temperature > 0,
   * repetition penalty, logit bias), or the turn ended at the token the prefill produced.
   */
  readonly batches: number;
  /**
   * Decode steps the batches ran. It can exceed the decode tokens the turn delivered: a batch that
   * stops at a stop token has already computed the steps after it (their rows are left uncommitted
   * and overwritten by the next run).
   */
  readonly steps: number;
};

/**
 * Speculation counters ({@link GenerationStop.speculation}).
 *
 * **The source of truth for confirmed tokens per cycle (token/cycle) is `delivered / cycles`**. An
 * ordinary cycle confirms `1 + a` tokens (even on rejection the frontier always advances by one),
 * but a cycle whose accepted draft was a stop token stops the enumeration there and confirms only
 * `a` — the old `(accepted + cycles) / cycles` overstates the numerator by 1 per such cycle. A
 * `k' = 0` cycle at the end of the budget (no draft taken) also counts in `cycles` and lands in
 * `acceptedHistogram[0]` — for acceptances per draft use `accepted / draftRuns`.
 */
export type GenerationSpeculation = {
  /** Number of verify runs (= number of cycles). */
  readonly cycles: number;
  /** Number of draft runs (a `k' = 0` cycle takes no draft). */
  readonly draftRuns: number;
  /** Total drafts produced (Σ k'). */
  readonly drafted: number;
  /**
   * Total drafts accepted and delivered (Σ a — for a cycle cut at a stop token, the count after the
   * cut).
   */
  readonly accepted: number;
  /**
   * Total tokens the cycles confirmed (Σ `confirmed.length` — the numerator of token/cycle).
   *
   * A cycle's share is added all at once when acceptance is decided (the same place as
   * `accepted`), so on a turn whose delivery closed mid-cycle through `break` or an interruption,
   * it is more than the consumer received.
   */
  readonly delivered: number;
  /** Index = that cycle's accepted count `a` (length `k+1`; a `k' = 0` cycle goes to index 0). */
  readonly acceptedHistogram: readonly number[];
  /**
   * Number of steps the self-tuning gate ran in the decode shape (M=1) (**for this turn**).
   *
   * MUST: **absent altogether** on a `policy: "always"` (no gate) turn — `0` and "no gate" are
   * different (0 is the observation "the gate never dropped", while absence is no observation).
   * These steps go into none of the fields above, so the turn's run count is
   * `cycles + draftRuns + plainSteps`.
   */
  readonly plainSteps?: number;
  /**
   * Number of times the gate switched speculate ↔ plain (**for this turn**; single exploration
   * trials are not counted).
   *
   * The gate itself lives as long as the sequence (the moving average is not discarded per turn),
   * but like the other fields this one counts the turn — it is the difference from the running
   * total at the start of the turn. A caller that needs the total for the whole conversation adds
   * them up (the total can be built from per-turn values, not the other way round).
   */
  readonly switches?: number;
};

/**
 * The self-tuning gate's state (`gate` of {@link GenerationRunPhase} — present only with
 * `policy: "auto"`).
 *
 * MUST: it is read **after observing** that run (in a cycle where the observation flips the mode,
 * the value after the flip appears) — "what happened after this run's wall went in" is what the
 * reader needs.
 */
export type GenerationGateTrace = {
  /**
   * The steady mode (`SpeculationGate.mode`).
   *
   * While exploration tries the other side, it still names **the steady side** (exploration burst
   * cycles appear as verify with `mode: "plain"`, and `W1` probes as decode with
   * `mode: "speculate"`).
   */
  readonly mode: SpeculationDecision;
  /** Switches within this turn (counted the same way as {@link GenerationSpeculation.switches}). */
  readonly switches: number;
  /**
   * Whether the gate put this run into its wall observations.
   *
   * `false` is a cycle the gate `skip()`ped (the first cycle of each turn and the forced plain at
   * the end of the budget — the two "not mixed in" kinds of the module doc). The wall is present but
   * not used for the decision.
   */
  readonly measured: boolean;
};

/**
 * One observation per run ({@link GenerationSequenceOptions.onRun}).
 *
 * All numbers are **1-based**. `chunk` / `chunks` of `prefill` are the same numbers as the
 * `prefill` of `GenerationEvent`, `step` of `decode` is the number of that turn's decode run, and
 * `cycle` of `draft` / `verify` is the speculation cycle number (the draft and the verify of one
 * cycle carry the same number). `verify.rows` is that run's valid row count `k'+1`, `accepted` the
 * number of accepted drafts `a`, and `delivered` the tokens that cycle confirmed (less than
 * `accepted + 1` only for a cycle whose enumeration stopped at a stop token — the same quantity as
 * {@link GenerationSpeculation.delivered}).
 *
 * A plain step the self-tuning gate drops to on a speculative turn also appears as `decode` (`step`
 * is that turn's running number of plain steps = counted the same way as
 * {@link GenerationSpeculation.plainSteps}) — because the run's shape is decode itself (M=1, R=1),
 * and adding a branch would break consumers with exhaustive switches over the public type.
 *
 * ## `wallMs` and `gate` (the two fields for reading the time breakdown)
 *
 * `wallMs` is the wall ms of one run: a speculative cycle is measured **from its head (before the
 * draft is issued) to right after the acceptance check**, and a non-speculative decode **from the
 * step's head (before the derived inputs) to right after the draw** (the same span as a speculative
 * cycle — including the derived inputs on only one side would put a systematic difference into
 * comparisons across modes). Both are taken **before the delivering yield**, so the consumer's speed
 * is excluded (it is the same value as the wall the gate uses for its own decisions — mixing it in
 * would make slower consumers turn speculation off). It is always present, with or without a gate.
 *
 * `gate` is present only on `policy: "auto"` turns, and the pair of `kind` and `gate.mode` gives
 * the phase of that run:
 *
 * | kind × gate.mode | phase |
 * | --- | --- |
 * | verify × speculate | steady speculation (after returning from plain if `switches` ≥ 1) |
 * | verify × plain | plain-side exploration burst |
 * | decode × speculate | `W1` probe while speculating |
 * | decode × plain | steady plain (a step the gate dropped to) |
 */
export type GenerationRunPhase =
  | { readonly kind: "prefill"; readonly chunk: number; readonly chunks: number }
  | {
      readonly kind: "decode";
      readonly step: number;
      readonly wallMs?: number;
      readonly gate?: GenerationGateTrace;
    }
  | { readonly kind: "draft"; readonly cycle: number }
  | {
      readonly kind: "verify";
      readonly cycle: number;
      readonly rows: number;
      readonly accepted: number;
      readonly delivered?: number;
      readonly wallMs?: number;
      readonly gate?: GenerationGateTrace;
    };

/**
 * One generation request (ADR 0083 decision 1).
 *
 * MUST: {@link GenerationSequence.generate} **copies the contents at issue** (ADR 0083 addendum
 * 2026-09-02). Rewriting this object, the `prompt` / `stopTokens` arrays or the `sampler` spec after
 * issue has no effect on the running generation — to affect the next turn, pass it as the next
 * request.
 */
export type GenerationRequest = {
  /**
   * This turn's token sequence. In multi-turn use pass **only the new turn's tokens** (the past is
   * in the context's KV). If the previous turn produced a token, `pendingToken` is prepended, so a
   * turn that "only continues generating" can pass an empty array.
   */
  readonly prompt: readonly number[];
  /**
   * Upper limit of tokens to draw (1 or more). Stop tokens are **included** in this count (not
   * emitted as `token` events but counted as one — the same counting as `tokens` of
   * {@link GenerationStop}).
   *
   * Omitted = until a stop token or the limit, and such a turn closes as `capacity` when it reaches
   * the limit. The limit is resolved once, right before the budget check, as the largest request
   * that check accepts (`min(capacity, maxPosition) − pastLength − prompt length + 1`, `pendingToken`
   * included in the prompt): the budget is not relaxed, and a prompt that alone does not fit fails
   * with {@link GenerationCapacityError} before any run.
   */
  readonly maxNewTokens?: number;
  /**
   * **Additional** stop tokens for this turn only (checked as the **union** with the set the
   * distribution declared).
   *
   * The distribution's EOS (`GenerationWiring.stopTokens`) always applies, so list here only "ids
   * this request wants to stop on" (stopping on a chat formatting token, cutting at the opening
   * marker of a tool call, and so on). That it stopped this way can be told from `stop-token` of
   * {@link GenerationStop}.
   *
   * MUST: out-of-vocabulary and duplicate ids fail loudly (ignoring them silently would leave "a
   * stop condition that does nothing" without an exception, noticeable only as output that keeps
   * growing).
   */
  readonly stopTokens?: readonly number[];
  /**
   * The sampling spec (omitted = temperature 0 = greedy — this layer's default per ADR 0083
   * decision 7).
   */
  readonly sampler?: SamplerSpec;
  /** Abort (checked at stage boundaries; `signal.reason` is thrown as is). */
  readonly signal?: AbortSignal;
};

/**
 * **Range** checks of a request (`maxNewTokens` / `stopTokens` / the sampler spec) — **the single
 * source of truth for the check**.
 *
 * MUST: do not keep two copies of the same expressions. Both the synchronous section of
 * {@link GenerationSequence.generate} and the high-level surfaces (`gemma.Pipeline.chat` /
 * `gemma.ChatSession.send`) **at issue** call this. The high-level surfaces call `generate` in an
 * async generator body, so without calling this themselves an invalid request would "not fail until
 * the first `next()`" (the diagnosis moves away from the issuer).
 *
 * NOTE: two things are not checked here — the `prompt` token ids must be checked on the array
 * **after the copy** (copying is `generate`'s job), and the budget (capacity, position limit) is
 * not settled until this request's turn comes (`assertBudget`).
 */
export const assertGenerationRequestValues = (
  vocabSize: number,
  request: Pick<GenerationRequest, "maxNewTokens" | "stopTokens" | "sampler">,
): void => {
  // Only `undefined` means "omitted": `null` and `NaN` are mistakes, not a request for the limit.
  const maxNewTokens = request.maxNewTokens;
  if (maxNewTokens !== undefined && (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1)) {
    throw new ModelInputError(`maxNewTokens ${maxNewTokens} is not a positive integer`);
  }
  // A stop token is "an id that does not appear in the output", so even an out-of-vocabulary one
  // lets generation **run to completion normally** (that id is just never drawn). To avoid quietly
  // leaving a stop condition that does nothing, the same range check as the program-side set
  // (`createGenerationProgram`) runs here too.
  const declared = new Set<number>();
  (request.stopTokens ?? []).forEach((token, index) => {
    if (!Number.isSafeInteger(token) || token < 0 || token >= vocabSize) {
      throw new ModelInputError(
        `stopTokens[${index}] ${token} is outside the vocabulary 0..${vocabSize - 1}`,
      );
    }
    // A duplicate cannot mean more than "the same condition written twice" = a sign of a caller
    // mix-up.
    if (declared.has(token))
      throw new ModelInputError(`Token ${token} appears twice in stopTokens`);
    declared.add(token);
  });
  // The sampler spec also fails here. `createSampler` only checks, copies and initializes `Randu`,
  // touching no external state, so building one for the check and discarding it is fine (the one
  // that actually runs is built by `generate`).
  createSampler(request.sampler);
};

/**
 * The token sequence itself (drained with `for await`) + the stop reason.
 *
 * MUST: `done` is a **secondary** notification path. Failures (a failed run, capacity overflow) are
 * thrown primarily by the iterable, and `done` only rejects with the same exception. It is caught
 * once internally for callers that never drain it, so not reading `done` does not cause an
 * unhandled rejection.
 * MUST: `done` settles **before** the end of iteration (the last `next()` returning `done: true`)
 * — the side wrapping the sequence (the close of gemma4's chat) relies on reading `done` right after
 * the sequence runs out without waiting. If settling slipped after the end, that `await` would stall
 * together with the consumer's `for await`.
 *
 * NOTE: calling the iterator's `return()` **without ever starting** iteration also settles `done`
 * (`closed`, `tokens: 0`). An async generator body does not run until the first `next()`, so an
 * end before the start is picked up and reported outside the body
 * (`concurrency/closeable-generator.ts`). No run is issued and no serialization slot is taken, so
 * other generations and `dispose` are not blocked.
 */
export type GenerationStream = AsyncIterable<GenerationEvent> & {
  readonly done: Promise<GenerationStop>;
};

export type GenerationSequence = {
  /**
   * The full-slot capacity this conversation can use (the value chosen at creation — it does not
   * change afterwards).
   *
   * This is the limit to use when computing yourself whether the next turn fits, not
   * `program.capacity` (the distribution's default) (the two differ for a sequence created with
   * `capacity`).
   */
  readonly capacity: number;
  /**
   * The number of logical positions this conversation already occupies (= where the next turn's
   * `prompt` starts piling up).
   *
   * MUST: a **derived value** (`context.pastLength` + uncommitted verify rows + 1 frontier — ADR
   * 0066 decision 6's ban on double bookkeeping). No independent counter is kept, so reading it
   * during a run reflects **the tokens that have entered the conversation at that moment** (it grows
   * as generation proceeds — read right after a `token` event, `used === position + 1`). Read between
   * generations it is "the settled value up to the most recently completed generation", and
   * truncation is decided there.
   *
   * The next turn fits when `used + prompt.length + maxNewTokens - 1 ≤ capacity` and
   * `used + prompt.length + maxNewTokens - 2 < program.maxPosition` (the actual values on overflow
   * are carried by {@link GenerationCapacityError}).
   */
  readonly used: number;
  /**
   * Generates one turn. Until the return value is drained (or `break`), this sequence's next
   * `generate` / `dispose` does not start (the serialization of ADR 0083 decision 2).
   *
   * Acceptance-set checks (`maxNewTokens` / the token ids of the prompt and `stopTokens` / the
   * sampler spec) and the **lifetime** check (a disposed sequence cannot generate) fail
   * **synchronously**. The budget check (position table, capacity) waits until this request's turn
   * comes, because at issue it is not yet known how far preceding generations will advance the
   * conversation (no decision from a stored counter = the ban on double bookkeeping).
   *
   * MUST: **copy the request at issue** (copies of `prompt` / `stopTokens`, binding of
   * `maxNewTokens` / `signal`, a snapshot of the sampler spec). The body does not run until the
   * first `next()`, so without copying, "the checked values" and "the values actually fed" can
   * differ (adding an out-of-vocabulary id to `prompt` after the check, raising `maxNewTokens`
   * mid-run, emptying `stopTokens`, swapping `logitBias` — none of these raises an exception).
   */
  generate(request: GenerationRequest): GenerationStream;
  /**
   * Returns an **independent** sequence continuing the same conversation (branching a
   * conversation — regeneration or trying an alternative).
   *
   * The caller owns the returned sequence and disposes it separately (no parent-child dependency on
   * the original; the original can be disposed first). The KV is copied in full at the same
   * capacity, so each branch adds the state's bytes (capacity × bytes per token).
   *
   * On a disposed sequence it throws **synchronously** (as `generate` does). The body is queued on
   * the same chain as `generate` / `dispose`, so it runs after an in-flight generation — if a stream
   * is being drained, drain it or close it with `break` before awaiting (fork does not return until
   * it is closed).
   *
   * Whether the token sequence after branching matches "the same operations without fork" follows
   * the determinism of the numerics path (no speculation and `"always"` match; with the self-tuning
   * gate only when the device's check is green — the Determinism section of the ADR 0066 addendum
   * of 2026-10-11).
   */
  fork(): Promise<GenerationSequence>;
  /**
   * Releases the context (queued on the same chain as `generate`, so it runs after an in-flight
   * generation).
   */
  dispose(): Promise<void>;
};

/**
 * The surface the generation loop requires of a context.
 *
 * It does not use `GenerationDisposable` from `greedy.ts` because it needs `pastLength` — that one
 * has the shape "the host does not read the logical length" (the prompt is the only starting
 * point), so it deliberately offers only lifetime release.
 */
export type GenerationContextFace = {
  /** The conversation's logical length (the only source of positions for this sequence). */
  readonly pastLength: number;
  /**
   * The pending deferred run (`undefined` if none) — read by the speculative path's
   * `settleCommit()`.
   */
  readonly pendingCommit: { readonly pastLength: number; readonly queryLength: number } | undefined;
  /**
   * The sliding ring's slack (`undefined` without sliding slots) — the `k + 1 ≤ slidingSlack`
   * check.
   */
  readonly slidingSlack: number | undefined;
  /**
   * Confirms the first `rows` rows of the pending deferred run (`0 ≤ rows ≤ queryLength`,
   * synchronous).
   */
  commit(rows: number): void;
  dispose(): Promise<void>;
};

/**
 * The surface of a context that can branch ({@link GenerationContextFace} + a `fork` returning its
 * own type).
 *
 * `fork`'s return is bound by the type parameter `C` (F-bound) so that the copied context is passed
 * to the same Session's `run` — collapsing it to `GenerationContextFace` would let passing a context
 * of another type to a real Session pass the type check (the same reason {@link GenerationSession}
 * threads the context through a type parameter). TS type literals cannot use the `this` type
 * (TS2526), so the self-reference is written with a type parameter.
 */
export type ForkableContextFace<C> = GenerationContextFace & {
  /**
   * An independent context with a copy of every byte of every slot and of the logical length (the
   * runtime's `GenerationContext.fork`).
   */
  fork(): Promise<C>;
};

/**
 * The surface the generation loop requires of a Session (narrow interface — DI lets a fake stand
 * in).
 *
 * MUST: not `Pick<Session, …>` (the same reason as `GreedySession` in `greedy.ts` —
 * `GenerationContext` is a nominal type with `#` private fields that a GPU-free fake cannot
 * satisfy). The context type is threaded through as a type parameter to pin in the types that "only
 * the object create returned goes back into run"; collapsing it to `{ dispose }` would (through
 * method bivariance) let passing another context to a real Session pass the type check.
 *
 * MUST: that the real `Session` satisfies this surface is pinned by a type check on the test side
 * (detects spelling drift).
 */
export type GenerationSession<C extends GenerationContextFace = GenerationContext> = {
  createGenerationContext(spec: GenerationContextSpec): Promise<C>;
  run(
    inputs: RunInputs,
    bindings: SymbolBindings | undefined,
    generation: {
      readonly context: C;
      readonly queryLength: number;
      /**
       * `"deferred"` = hold without advancing the logical length (speculation's verify — `commit`
       * once the accepted count is known).
       */
      readonly commit?: "deferred";
    },
  ): Promise<RunOutputs>;
};

/** The speculation specification ({@link GenerationSequenceOptions.speculative}). */
export type GenerationSpeculativeOptions<C extends GenerationContextFace> = {
  /**
   * Opens the drafter face on the lending context (once at sequence creation; on failure the
   * sequence closes the lending context before rethrowing). The close order is **drafter → lending
   * context** (owned by the sequence).
   */
  open(context: C): Promise<DraftFace>;
  /** Drafts used per cycle (default `DraftFace.steps`, `1 ≤ k ≤ steps`). */
  readonly k?: number;
  /**
   * When to speculate (default `"auto"`).
   *
   * - `"auto"` … with the self-tuning gate (the "self-tuning gate" section of the module doc). In
   *   losing contexts it drops to the decode shape (M=1) cycle by cycle.
   * - `"always"` … always speculate (no gate is built; the wall clock is still read for `wallMs`,
   *   see `now`). An option for A/B comparison, acceptance checks and measurement; `plainSteps` /
   *   `switches` of `GenerationSpeculation` do not appear either.
   */
  readonly policy?: "auto" | "always";
  /**
   * The gate's knobs (effective only with `policy: "auto"` — omitted = the defaults of
   * `speculation-gate.ts`).
   */
  readonly gate?: SpeculationGateOptions;
  /**
   * The wall clock (default `performance.now` — tests plug in a fake clock).
   *
   * It is read for the gate's observations and the observation hook's `wallMs`
   * ({@link GenerationRunPhase}), and even with `policy: "always"` it is called twice per cycle
   * (at the head of the cycle and right after the acceptance check).
   */
  readonly now?: () => number;
};

export type GreedyRunResult = { readonly value: number; readonly index: number };
export type GenerationGreedyRun<C extends GenerationContextFace> = (
  inputs: RunInputs,
  generation: { readonly context: C; readonly queryLength: number },
) => Promise<GreedyRunResult>;

/** One call of {@link GenerationGreedyAhead.run}. */
export type GenerationGreedyAheadRequest<C extends GenerationContextFace> = {
  /** The token step 0 feeds (the frontier). Step `k > 0` feeds step `k − 1`'s selection. */
  readonly token: number;
  /**
   * Per step, every run input **except the token ids** (the last-row selector and the derived
   * inputs of the step's position `context.pastLength + k`). The length is the step count.
   */
  readonly steps: readonly RunInputs[];
  readonly context: C;
};

/**
 * The decode run-ahead (internal capability — {@link GenerationSequenceOptions.greedyAhead}):
 * several greedy decode steps in **one GPU batch**, each step fed the previous step's on-GPU
 * selection, the selections read back together at the batch's single fence. It removes the host
 * round trip between tokens (the fence, the readback, the host writing the next token) from all but
 * one step in `depth`.
 *
 * MUST: the steps are **deferred** runs chained on the context (`GenerationRun.chain` of the
 * runtime) — on return `context.pendingCommit` holds exactly `steps.length` rows starting at the
 * `pastLength` of the call, and the caller commits the rows it keeps with `context.commit(rows)`.
 * MUST: `1 ≤ steps.length ≤ depth`; the results are the steps' selections in order.
 */
export type GenerationGreedyAhead<C extends GenerationContextFace> = {
  /** The most steps one batch runs (`≥ 2`). */
  readonly depth: number;
  run(request: GenerationGreedyAheadRequest<C>): Promise<readonly GreedyRunResult[]>;
};

export type GenerationSequenceOptions<C extends GenerationContextFace> = {
  /**
   * Internal capability that returns the maximum of the raw logits. Used only for decode at
   * temperature 0, without processing, non-speculative, without observation and with an empty
   * {@link GenerationWiring.suppressTokens} (the raw maximum can be a suppressed id).
   */
  readonly greedy?: GenerationGreedyRun<C>;
  /**
   * The decode run-ahead (internal capability). It replaces the per-step decode exactly where
   * {@link GenerationSequenceOptions.greedy} would be used — temperature 0 without processing,
   * non-speculative, without observation — and the token ids it yields are **identical** to the
   * per-step decode's (the same kernels on the same inputs; only the batching changes). A turn that
   * does not meet the condition decodes per step as before ({@link GenerationStop.decodeAhead}
   * tells which path ran).
   *
   * MUST (checked at sequence creation, failing loudly): `depth` is an integer ≥ 2 within the
   * context's sliding slack and `chunkLength` (one batch's rows stay uncommitted until the host
   * reads them back); no `speculative` (a verify is not a greedy step); the derived inputs are
   * absent or declared `positionsOnly` (a step is derived before its token exists).
   */
  readonly greedyAhead?: GenerationGreedyAhead<C>;
  readonly session: GenerationSession<C>;
  /** The verified static wiring (the return value of `createGenerationProgram`). */
  readonly program: GenerationWiring;
  /** Speculative decoding (ADR 0096 stage 3). Without it, the usual 1 token = 1 run. */
  readonly speculative?: GenerationSpeculativeOptions<C>;
  /**
   * Observation hook called once per run in the **synchronous section** after that run's outputs
   * are read (for verify, **before** the confirmed tokens are delivered and committed — across the
   * delivering yield, a run of another sequence sharing the lending Session can slip in). Only in
   * this synchronous section can diagnostics (`lastRun*` of `Session.diagnostics()`) be read as the
   * values of "that run". Nothing is called without it.
   */
  readonly onRun?: (phase: GenerationRunPhase) => void;
  /**
   * The full-slot capacity this conversation uses (omitted = {@link GenerationWiring.capacity} = the
   * distribution's default).
   *
   * **A knob at sequence creation** — the capacity is the physical allocation of the state slots
   * itself (12,288 B/token over the full layers for gemma4 E2B), so the trade between "long
   * conversation" and "VRAM" is decided here once. It cannot change afterwards because the
   * context's physical allocation is done at creation; to change it, build another sequence (the KV
   * cannot be carried over — the same line as ADR 0066 addendum 2's rejection of every `rewind`).
   *
   * MUST: `program.chunkLength ≤ capacity ≤ program.maxPosition` (fails loudly at creation).
   */
  readonly capacity?: number;
};

/** One i32 input tensor (both the token id sequence and the absolute positions are `[1, rows]`). */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/**
 * Chooses the **physical row count** (the row count of the declared shape = `M`) this chunk runs
 * with.
 *
 * - One valid row runs in the **decode shape (M=1)**. It adds no plan, and resuming after an
 *   interruption becomes **the same run** as "a run that was never interrupted" (re-feeding
 *   `pendingToken` is always one row, so this is the key to multi-turn bit identity).
 * - Otherwise the **smallest** value in `chunkBuckets` that is ≥ `queryLength`; `chunkLength` if
 *   none.
 *
 * Why buckets: padding a short prompt to `chunkLength` (distribution default 768) rows piles up
 * work for row-local ops (linear / pointwise / norm) in proportion to the physical row count, even
 * though pad rows contribute nothing to the output or the KV. The measured wall of one 32-token
 * utterance + one generated token went from 385 → 106 ms (linear −73%) (ADR 0066 addendum 10 —
 * re-measured in `docs/research/2026-09-07-codex-perf-review-followup.md` §5); chat user utterances
 * are short while the context's `chunkLength` cannot change midway, so multi-turn use pays this pad
 * on every turn. Long prompts, on the other hand, are fastest at 768 in one go, so "lowering the
 * default" is not taken; instead **several allowed physical row counts are kept and one is chosen
 * per chunk** (ADR 0066 decision 4 / addendum 10, buckets).
 *
 * MUST: take the first match assuming ascending order (the order is not checked here — the check
 * belongs to runtime's `assertChunkBuckets`, which `createGenerationProgram` calls).
 *
 * The argument is narrowed to two fields because the estimate (`gemma4EstimateGenerations`) chooses
 * row counts by the same rule for a **hypothetical** chunk length and ladder rather than the
 * pipeline's wiring (the rule is not kept twice for real runs and estimates).
 */
export const physicalChunkRows = (
  queryLength: number,
  program: Pick<GenerationWiring, "chunkLength" | "chunkBuckets">,
): number =>
  queryLength === 1
    ? 1
    : (program.chunkBuckets.find((rows) => rows >= queryLength) ?? program.chunkLength);

/**
 * The row-selection input (`[R]` i32 — ADR 0068 decision 4).
 *
 * What is passed is **a sequence of row indices** (its element count binds that run's R). This
 * path always passes one (prefill = the last valid row / decode = row 0), so the shape folds to
 * `[1]` as before.
 */
const lastRowInput = (rows: readonly number[]): Tensor => ({
  dtype: "i32",
  shape: [rows.length],
  data: Int32Array.from(rows),
});

/** Read surface of the selected rows' logits (returned by {@link readLogits} — a view per row). */
type LogitsRows = {
  /** Rows returned (= R, the number of indices passed to `last_row`). */
  readonly rows: number;
  /**
   * Raw logits of row `index` (a view of `vocabSize` elements — not copied).
   *
   * MUST: what is returned is a subarray of the output buffer, valid **only until the next run**
   * (the readback buffer is rebuilt per run — the caller copies it to keep it).
   */
  readonly row: (index: number) => Float32Array<ArrayBuffer>;
};

/**
 * Reads the raw data of the selected rows' output `[1,R,width]` (`V` for logits, `H` for hidden).
 *
 * The program's setup already verified the name and declared shape, but it is checked here too
 * because this is the only line of detection when holding "a Session built from **a different
 * graph** than the one the program verified" (catching another output with a matching shape gives
 * another token sequence with no exception and no warning).
 *
 * MUST: match the row count against **the number of indices passed** (`expectedRows`). R is a
 * symbol in the graph, so "asked for 1 row, R rows returned" is also valid as a declaration, and
 * this is the only check.
 */
const readRows = (
  outputs: RunOutputs,
  name: string,
  width: number,
  where: string,
  expectedRows: number,
): LogitsRows => {
  if (!Object.hasOwn(outputs, name)) {
    throw new Error(`${where}: graph output '${name}' is missing`);
  }
  const tensor = outputs[name];
  if (tensor.dtype !== "f32") {
    throw new Error(`${where}: '${name}' is not f32 (${tensor.dtype})`);
  }
  const shape = tensor.shape;
  if (shape.length !== 3 || shape[0] !== 1 || shape[1] !== expectedRows || shape[2] !== width) {
    throw new Error(
      `${where}: '${name}' has shape [${shape.join(",")}], not [1,${expectedRows},${width}]`,
    );
  }
  const data = tensor.data;
  return {
    rows: expectedRows,
    row: (index: number): Float32Array<ArrayBuffer> => {
      if (!Number.isSafeInteger(index) || index < 0 || index >= expectedRows) {
        throw new Error(`${where}: row ${index} is outside 0..${expectedRows - 1}`);
      }
      return data.subarray(index * width, (index + 1) * width);
    },
  };
};

/** The selected rows' logits `[1,R,V]`. */
const readLogits = (
  outputs: RunOutputs,
  program: GenerationWiring,
  where: string,
  expectedRows: number,
): LogitsRows => readRows(outputs, program.logits, program.vocabSize, where, expectedRows);

/**
 * The selected rows' final-norm hidden `[1,R,H]` (read only by the speculative path — the drafter's
 * input).
 */
const readHidden = (
  outputs: RunOutputs,
  program: GenerationWiring,
  where: string,
  expectedRows: number,
): LogitsRows => readRows(outputs, program.hidden, program.hiddenSize, where, expectedRows);

/**
 * Checks the limits this turn will hit **before** the run (ADR 0083 decision 10).
 *
 * - Positions: prefill touches `past .. past+T-1`, decode `past+T .. past+T+K-2`.
 * - The maximum of `pastLength + queryLength` is `past+T+K-1` at the last decode (`past+T` when
 *   K=1).
 */
const assertBudget = (
  program: GenerationWiring,
  capacity: number,
  pastLength: number,
  promptLength: number,
  maxNewTokens: number,
): void => {
  // "The maxNewTokens that fits now" folds into the same shape for both constraints
  // ({@link GenerationCapacityDetail}).
  const detail = (
    constraint: GenerationCapacityConstraint,
    limit: number,
  ): GenerationCapacityDetail => ({
    constraint,
    pastLength,
    promptLength,
    requestedNewTokens: maxNewTokens,
    limit,
    maxNewTokens: limit - pastLength - promptLength + 1,
  });
  const lastPosition = pastLength + promptLength + maxNewTokens - 2;
  if (lastPosition >= program.maxPosition) {
    throw new GenerationCapacityError(
      `The conversation leaves the position table: existing ${pastLength} + prompt ${promptLength} + ` +
        `maxNewTokens ${maxNewTokens} reaches the last position ${lastPosition}` +
        ` (this asset can look up 0..${program.maxPosition - 1} — truncating the conversation is the host's job)`,
      detail("maxPosition", program.maxPosition),
    );
  }
  const peak = pastLength + promptLength + maxNewTokens - 1;
  if (peak > capacity) {
    throw new GenerationCapacityError(
      `The conversation exceeds the state capacity: existing ${pastLength} + prompt ${promptLength} + ` +
        `maxNewTokens ${maxNewTokens} needs ${peak} rows` +
        ` (capacity ${capacity} — truncating the conversation is the host's job)`,
      detail("capacity", capacity),
    );
  }
};

/** Whether it is the abort exception (`signal.reason` is thrown unwrapped, so identity tells). */
const isAbortOf = (error: unknown, signal: AbortSignal | undefined): boolean =>
  signal !== undefined && signal.aborted && error === signal.reason;

/**
 * The acceptance set of the speculation specification (at sequence creation, synchronous).
 *
 * - `1 ≤ k ≤ steps` (the distribution's drafter is baked with `steps` steps).
 * - `k + 1 ≤ slidingSlack`: verify writes `k+1` rows deferred, and the condition under which
 *   rejected rows do not clobber the sliding ring's live window is the runtime's check
 *   `queryLength ≤ slidingSlack` (the borrower reads up to the oldest column `P−W`). Without
 *   checking here, the runtime check would fail at the first verify, after a GB-scale load.
 * - `k + 1 ≤ chunkLength` and "a bucket of at least `k+1` rows exists": otherwise
 *   `physicalChunkRows` falls back to `chunkLength` rows (distribution default 768), and one verify
 *   becomes 768 rows without an exception.
 */
const assertSpeculativeSetup = (
  k: number,
  steps: number,
  slidingSlack: number | undefined,
  program: GenerationWiring,
): void => {
  // The step count is the value the borrower face reports (the steps the distribution's drafter
  // was baked with). A non-integer or ≤ 0 value **slips through** the `k` check (omitting `k` gives
  // `k = steps` and every NaN comparison is false; passing `k` makes `k > steps` false), so it is
  // checked before `k`.
  if (!Number.isSafeInteger(steps) || steps < 1) {
    throw new Error(`Drafter step count ${steps} is not a positive integer`);
  }
  if (!Number.isSafeInteger(k) || k < 1 || k > steps) {
    throw new Error(`speculative.k ${k} is outside 1..${steps} (the drafter's step count)`);
  }
  if (slidingSlack !== undefined && k + 1 > slidingSlack) {
    throw new Error(
      `speculative.k ${k} does not fit in the sliding ring's slack ${slidingSlack} (k + 1 ≤ slack MUST hold)`,
    );
  }
  if (k + 1 > program.chunkLength) {
    throw new Error(
      `The verify of speculative.k ${k} (${k + 1} rows) exceeds chunkLength ${program.chunkLength}`,
    );
  }
  if (!program.chunkBuckets.some((rows) => rows >= k + 1)) {
    throw new Error(
      `No bucket holds the ${k + 1} verify rows (chunkBuckets [${program.chunkBuckets.join(
        ", ",
      )}])`,
    );
  }
};

/**
 * The setup check of the decode run-ahead ({@link GenerationSequenceOptions.greedyAhead}) — at
 * sequence creation, once the context (and with it the sliding slack) exists.
 *
 * - `depth ≤ slidingSlack` and `≤ chunkLength`: one batch leaves up to `depth` rows uncommitted
 *   until the host reads the batch back, and the runtime bounds a chain of deferred runs by the
 *   same two values. Failing here names the knob; the runtime would refuse the first batch with
 *   the generic bound.
 * - No speculation: a speculative turn verifies drafts, it does not take greedy steps, and the two
 *   would contend for the one pending commit.
 * - `positionsOnly` derived inputs: a step's inputs are derived before its token exists (the token
 *   is still on the GPU), so a slot that reads ids would be fed a stand-in.
 */
const assertGreedyAheadSetup = (
  depth: number,
  speculative: boolean,
  slidingSlack: number | undefined,
  program: GenerationWiring,
): void => {
  if (!Number.isSafeInteger(depth) || depth < 2) {
    throw new Error(
      `greedyAhead.depth ${depth} is not an integer ≥ 2 (1 step per batch is the per-step decode)`,
    );
  }
  if (speculative) {
    throw new Error(
      "greedyAhead and speculative cannot be combined (a verify is not a greedy step, and both hold the context's one pending commit)",
    );
  }
  if (slidingSlack !== undefined && depth > slidingSlack) {
    throw new Error(
      `greedyAhead.depth ${depth} exceeds the sliding ring's slack ${slidingSlack}` +
        " (a batch's rows stay uncommitted until it is read back, and rejected rows must not overwrite live past KV)",
    );
  }
  if (depth > program.chunkLength) {
    throw new Error(`greedyAhead.depth ${depth} exceeds chunkLength ${program.chunkLength}`);
  }
  if (program.derivedInputs !== undefined && program.derivedInputs.positionsOnly !== true) {
    throw new Error(
      "greedyAhead needs derived inputs declared positionsOnly (a step's inputs are derived before its" +
        " token exists — it is still on the GPU)",
    );
  }
};

/**
 * A copy of an output row (a readback subarray lasts until the next run — the hidden kept for the
 * next cycle is copied).
 */
const copyRow = (row: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const copy = new Float32Array(new ArrayBuffer(row.byteLength));
  copy.set(row);
  return copy;
};

/**
 * Builds the sequence for one conversation (the context is allocated here, and this object owns
 * its lifetime from then on).
 *
 * MUST: do not expose the `GenerationContext` (ADR 0083 decision 3) — hiding "the uncommitted
 * frontier of at most one token" from consumers is the key that makes decision 4 work.
 *
 * Allocation (the capacity check and context creation) is here; assembly (drafter, gate and the
 * generation loop) is {@link assembleSequence} — assembly is shared with branching
 * ({@link GenerationSequence.fork}).
 */
export const createGenerationSequence = async <C extends ForkableContextFace<C>>(
  options: GenerationSequenceOptions<C>,
): Promise<GenerationSequence> => {
  const { session, program } = options;
  const capacity = options.capacity ?? program.capacity;
  // capacity is a caller knob that the public entries (`sequence({ capacity })` / `chat`) pass
  // through, so all three conditions are input-caused (ADR 0107 decision 2). The type matches the
  // same conditions and wording in `chat-session.ts` — split, the same value would turn into 400 /
  // 500 depending on the entry.
  if (!Number.isSafeInteger(capacity) || capacity < 1) {
    throw new ModelInputError(`capacity ${capacity} is not a positive integer`);
  }
  // The capacity relations are checked before the context is allocated (the same two expressions
  // `gemma.parsePipelineConfig` checks against the declaration, applied to the run-time knob too).
  if (capacity < program.chunkLength) {
    throw new ModelInputError(
      `capacity ${capacity} is below chunkLength ${program.chunkLength} (not even one chunk fits)`,
    );
  }
  if (capacity > program.maxPosition) {
    throw new ModelInputError(
      `capacity ${capacity} exceeds maxPosition ${program.maxPosition}` +
        ` (a conversation filling the capacity would look up positions beyond the model's position limit)`,
    );
  }
  // MUST: the capacity symbol is bound only at context creation (ADR 0066 addendum 7 — not passed to
  // the run's bindings).
  const context = await session.createGenerationContext({
    bindings: { [program.capacitySymbol]: capacity },
    chunkLength: program.chunkLength,
    // Allowed physical row counts can be declared only at context creation (the set the pre-run
    // check reads is folded here). An empty array = none added, so wiring without a declaration
    // keeps the usual two shapes.
    chunkBuckets: program.chunkBuckets,
  });
  return await assembleSequence(options, capacity, context, undefined);
};

/**
 * State a branch carries over to the new sequence (captured by {@link GenerationSequence.fork}
 * inside the chain's slot).
 */
type BranchState = {
  /** The original uncommitted frontier (undefined if none). */
  readonly pendingToken: number | undefined;
  /** A copy of the original gate (undefined if the original had no gate). */
  readonly gate: SpeculationGate | undefined;
};

/**
 * Assembles a sequence from an allocated context (the one function shared by
 * {@link createGenerationSequence} and {@link GenerationSequence.fork}). Without `inherited` it is
 * a new conversation (the gate is built from the knobs); with it, a branch (frontier and gate are
 * carried over).
 *
 * MUST: if assembly fails, close the context before rethrowing (the context of a failed assembly is
 * unreachable from the caller — leaked, not even Session.dispose would notice).
 */
const assembleSequence = async <C extends ForkableContextFace<C>>(
  options: GenerationSequenceOptions<C>,
  capacity: number,
  context: C,
  inherited: BranchState | undefined,
): Promise<GenerationSequence> => {
  const { session, program } = options;
  /**
   * The drafter face (only with a speculation spec — the borrower opens right after the context and
   * closes first).
   */
  let face: DraftFace | undefined;
  let k = 0;
  /**
   * The self-tuning gate (only with `policy: "auto"`).
   *
   * MUST: its lifetime is the sequence (kept across turns) — the moving averages of wall and
   * accepted count do not converge within one turn, and measuring again per conversation would
   * repeat "speculate only for the first few cycles" on every turn.
   * MUST: a forked sequence receives a copy of the observed state (`inherited.gate` — a branch
   * continues the same conversation on the same host, so it does not measure again).
   */
  let gate: SpeculationGate | undefined = inherited?.gate;
  /**
   * The wall clock (read by the gate's observations and the hook's `wallMs` — default
   * `performance.now`).
   */
  const now = options.speculative?.now ?? ((): number => performance.now());
  const ahead = options.greedyAhead;
  if (ahead !== undefined) {
    try {
      assertGreedyAheadSetup(
        ahead.depth,
        options.speculative !== undefined,
        context.slidingSlack,
        program,
      );
    } catch (error) {
      // The context of a failed assembly is unreachable from the caller — close it first (the same
      // rule as the speculative block below; it runs first, so no drafter is open yet).
      try {
        await context.dispose();
      } catch (cleanup) {
        throw new AggregateError(
          [error, cleanup],
          "greedyAhead was refused, and closing the context failed too",
        );
      }
      throw error;
    }
  }
  if (options.speculative !== undefined) {
    try {
      // The knob check runs **before** the borrower is opened (an invalid knob must not wait until
      // after a GB-scale load to fail). A branch (`inherited` set) receives a gate copied with knobs
      // checked at the original creation, so it skips this.
      if (inherited === undefined && (options.speculative.policy ?? "auto") === "auto") {
        gate = createSpeculationGate(options.speculative.gate);
      } else if (inherited === undefined && options.speculative.gate !== undefined) {
        // Passing knobs where there is no gate is "a knob that does nothing" — not passed silently
        // (cross-cutting invariant).
        throw new Error(
          `speculative: gate knobs ${JSON.stringify(
            options.speculative.gate,
          )} were passed with policy "always"` + `, but "always" has no gate`,
        );
      }
      face = await options.speculative.open(context);
      k = options.speculative.k ?? face.steps;
      assertSpeculativeSetup(k, face.steps, context.slidingSlack, program);
    } catch (error) {
      // Close borrower → lender, then rethrow (a borrower that failed to open does not exist). A
      // leaked lending context goes unnoticed by Session.dispose (it does not count live
      // contexts). If the clean-up fails too, carry both (only the original failure would make
      // "the context was not returned" silent).
      try {
        await disposeSteps([() => face?.dispose(), () => context.dispose()]);
      } catch (cleanup) {
        throw new AggregateError(
          [error, cleanup],
          "speculative.open failed, and cleaning up the lending context failed too",
        );
      }
      throw error;
    }
  }

  // Serialization per "one generate" (ADR 0083 decision 2) — no home-made lock.
  const chain = createOperationChain();
  // A branch inherits the original uncommitted frontier (prepended to the next turn's prefill —
  // decision 4).
  let pendingToken: number | undefined = inherited?.pendingToken;
  /**
   * The rows of the pending verify made frontier (= rows the target consumed; non-zero only during
   * speculative delivery and during the hand-out of a run-ahead batch, whose rows are pending the
   * same way). {@link settleCommit} commits the pending verify up to here.
   *
   * MUST: kept in sequence scope (the `used` getter reads it). During delivery `context.pastLength`
   * stays at the start of the cycle, so without adding this **delivered tokens would not appear in
   * the logical length**. It does not survive across generates because `settleCommit()` resets it
   * to 0 right after the commit and every exit of delivery (stop / after the delivery loop /
   * `finally`) goes through it. If `context.commit` itself throws it is not reset, but that happens
   * only when the context is poisoned / disposed, after which reading `pastLength` fails too (the
   * leftover value is never observed).
   */
  let frontierRows = 0;
  let disposal: Promise<void> | undefined;

  /**
   * Takes a slot in the serialization chain and releases it with the returned function.
   *
   * The slot is taken **when the body starts running** (an async generator body does not run until
   * the first `next()`). Taking it at issue would let an iterable discarded without draining hold
   * the chain forever, dragging `dispose` along. The order is "whoever starts draining first".
   */
  const acquire = async (): Promise<() => void> => {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    await new Promise<void>((admitted) => {
      void chain(() => {
        admitted();
        return held;
      });
    });
    return release;
  };

  /**
   * Host-derived extra inputs (checked every time against the declared names, nothing missing and
   * nothing extra).
   *
   * `signal` is passed down because some distributions take their derived inputs from GB-scale
   * lazy loads (gemma4's PLE blocks) — it is best-effort, so even an implementation that ignores it
   * is closed by the check before the run.
   */
  const deriveInputs = async (
    ids: Int32Array<ArrayBuffer>,
    positions: Int32Array<ArrayBuffer>,
    signal: AbortSignal | undefined,
  ): Promise<RunInputs> => {
    const derived = program.derivedInputs;
    if (derived === undefined) return {};
    const extra = await derived.derive(
      [...ids],
      [...positions],
      signal === undefined ? {} : { signal },
    );
    const keys = Object.keys(extra);
    const missing = derived.names.filter((name) => !Object.hasOwn(extra, name));
    const surplus = keys.filter((name) => !derived.names.includes(name));
    if (missing.length > 0 || surplus.length > 0) {
      throw new Error(
        `derivedInputs disagree with the declaration (missing: ${missing.join(" / ") || "none"} / ` +
          `extra: ${surplus.join(" / ") || "none"})`,
      );
    }
    return extra;
  };

  const generate = (request: GenerationRequest): GenerationStream => {
    // MUST: the lifetime check is **synchronous** too (ADR 0083 decision 3 — the context is not
    // exposed, so the caller has no other way to find out it was disposed). Delayed, it would not
    // fail until the first iteration, and with the generic wording of `GenerationContext`, so the
    // real cause (the caller disposed it) could not be read.
    if (disposal !== undefined) {
      throw new Error("GenerationSequence: cannot generate after dispose");
    }
    // MUST: **copy** the request here (the MUST of {@link GenerationSequence.generate}). An async
    // generator body does not run until the first `next()`, so reading `request` without copying
    // would make the acceptance check see "the values at issue" and the run "the values when
    // draining started" — a request that passed the check would run as something else.
    const prompt = [...request.prompt];
    const requestStopTokens = request.stopTokens === undefined ? [] : [...request.stopTokens];
    const maxNewTokens = request.maxNewTokens;
    const signal = request.signal;
    // The acceptance set fails synchronously (before entering the GPU or the queue). It checks the
    // copied values.
    assertGenerationRequestValues(program.vocabSize, {
      maxNewTokens,
      stopTokens: requestStopTokens,
      sampler: request.sampler,
    });
    prompt.forEach((id, index) => {
      // Writing to an `Int32Array` **silently** truncates non-integers and wraps out-of-range values
      // (2^32+1 → 1 = another valid token id), so fail at the entry. Out of vocabulary means an
      // out-of-range embedding gather = NaN contamination of the whole row, so check it here too.
      if (!Number.isSafeInteger(id) || id < 0 || id >= program.vocabSize) {
        throw new ModelInputError(
          `prompt[${index}] ${id} is outside the vocabulary 0..${program.vocabSize - 1}`,
        );
      }
    });
    // One sampler per generation (it carries RNG state across steps). The spec check and the spec
    // snapshot (copying `logitBias` down to its elements) also happen inside `createSampler`.
    // The distribution's suppressed set rides along, so every selection below — the first token,
    // decode and each speculative acceptance row — masks it; a non-empty set also withdraws
    // `sampler.greedy`, which turns the GPU selection below off.
    const sampler = createSampler(request.sampler, program.suppressTokens);
    const greedy =
      face === undefined && options.onRun === undefined && sampler.greedy !== undefined
        ? options.greedy
        : undefined;
    /**
     * The run-ahead takes the decode exactly where the per-step greedy selection would run (the same
     * three conditions — {@link GenerationSequenceOptions.greedyAhead}).
     */
    const runAhead =
      face === undefined && options.onRun === undefined && sampler.greedy !== undefined
        ? ahead
        : undefined;
    /** This turn's run-ahead counters (only on a sequence built with `greedyAhead`). */
    const aheadTally =
      ahead === undefined ? undefined : { depth: ahead.depth, batches: 0, steps: 0 };
    const pickGreedy = (result: GreedyRunResult): number => {
      if (
        !Number.isSafeInteger(result.index) ||
        result.index < 0 ||
        result.index >= program.vocabSize
      ) {
        throw new Error(
          `Greedy index ${result.index} is outside the vocabulary 0..${program.vocabSize - 1}`,
        );
      }
      if (sampler.greedy === undefined) throw new Error("No greedy selection capability");
      return sampler.greedy(result.value, result.index);
    };
    // With a drafter, speculation runs **regardless of temperature**. Acceptance calls
    // `sampler.next` once per row with the same logits, the same history and in the same order as
    // non-speculative decode (once per confirmed token), so the RNG consumption and the token
    // sequence match non-speculative decoding exactly — at temperature > 0, "accept when the drawn
    // token equals the draft" is speculative sampling with a one-hot draft (acceptance probability =
    // the draft's probability under the target distribution).
    const tally: SpeculationTally | undefined =
      face === undefined ? undefined : createSpeculationTally(k);
    /** Steps the gate ran in the decode shape (for this turn — not in the speculation counters). */
    let plainSteps = 0;
    /**
     * The gate's running total of switches at the start of the turn (the difference is
     * {@link GenerationSpeculation.switches}).
     *
     * MUST: re-read it **after** taking the chain's slot (a preceding turn can move the gate between
     * issue and the body starting). It is initialized with the value at issue so that a turn closed
     * before taking the slot (an abort while queued) also yields a difference of 0.
     */
    let switchesAtStart = gate?.switches ?? 0;
    const withSpeculation = (stopped: GenerationStop): GenerationStop =>
      tally === undefined
        ? stopped
        : {
            ...stopped,
            speculation: {
              ...tally,
              acceptedHistogram: [...tally.acceptedHistogram],
              // A turn without a gate gets no such fields (0 and "no gate" are different).
              ...(gate === undefined
                ? {}
                : { plainSteps, switches: gate.switches - switchesAtStart }),
            },
          };
    /** The turn's stop with every counter the sequence keeps (speculation, run-ahead). */
    const withCounters = (stopped: GenerationStop): GenerationStop =>
      aheadTally === undefined
        ? withSpeculation(stopped)
        : { ...withSpeculation(stopped), decodeAhead: { ...aheadTally } };
    /** Confirms the pending verify up to the frontier (does nothing if nothing is pending). */
    const settleCommit = (): void => {
      if (context.pendingCommit === undefined) return;
      context.commit(frontierRows);
      frontierRows = 0;
    };
    const onRun = options.onRun;

    /** The stop of a turn that drew its whole limit (the doc of {@link GenerationStop}). */
    const limitStop = (tokens: number): GenerationStop =>
      maxNewTokens === undefined
        ? { reason: "capacity", tokens }
        : { reason: "max-tokens", tokens };

    /**
     * The stop check (the **union** of the distribution's set and the request's set).
     *
     * The order matters only for the reason — an id in both closes as `eos`, as the distribution's
     * terminator (the doc of {@link GenerationStop}).
     */
    const stopFor = (token: number, tokens: number): GenerationStop | undefined => {
      if (isStopToken(token, program.stopTokens)) return { reason: "eos", token, tokens };
      if (isStopToken(token, requestStopTokens)) return { reason: "stop-token", token, tokens };
      return undefined;
    };

    let settle!: (stop: GenerationStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<GenerationStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // A secondary notification path, so no unhandled rejection even if unread (the primary path is
    // the iterable's throw).
    done.catch(() => {});

    const events = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
      let stop: GenerationStop | undefined;
      let failure: { readonly error: unknown } | undefined;
      let release: (() => void) | undefined;
      /** Tokens drawn (a stop token counts as one — the doc of `GenerationStop.tokens`). */
      let generated = 0;
      try {
        release = await acquire();
        // An abort that arrived while queued closes here (a long preceding generation means a long
        // wait). A synchronous check suffices because the wait itself is an `await` = the abort
        // task has already been delivered.
        switchesAtStart = gate?.switches ?? 0;
        signal?.throwIfAborted();

        const past = context.pastLength;
        // Multi-turn prepend (ADR 0083 decision 4) — the uncommitted frontier goes in front of the
        // new prompt.
        const promptIds = pendingToken === undefined ? prompt : [pendingToken, ...prompt];
        if (promptIds.length === 0) {
          // The prompt is the public request itself, and the only remedy is "put tokens in" (ADR
          // 0107 decision 2). Same type as this function throws for out-of-vocabulary ids.
          throw new ModelInputError(
            "prompt is empty (there is no pendingToken from the previous turn either, so there is no token to feed)",
          );
        }
        // A request without `maxNewTokens` resolves to the largest one `assertBudget` accepts (both
        // constraints fold into one shape — {@link GenerationCapacityDetail.maxNewTokens}). Below 1
        // the prompt alone does not fit, and checking 1 throws the same error before any run.
        const tokenLimit =
          maxNewTokens ??
          Math.max(1, Math.min(capacity, program.maxPosition) - past - promptIds.length + 1);
        assertBudget(program, capacity, past, promptIds.length, tokenLimit);

        const chunks = planPrefillChunks(promptIds.length, program.chunkLength);
        // "The token sequence so far" the repetition penalty looks at (the same shape as HF applying
        // it to the whole `input_ids`). **Only this turn's tokens**, past turns excluded (the
        // sequence holds neither a progress counter nor a transcript of the whole conversation —
        // the "mutable state" section of the module doc).
        const history = [...promptIds];

        let logits: LogitsRows | undefined;
        /**
         * Hidden of the last valid row of the last chunk (drafter input of the first speculative
         * cycle — a copy).
         */
        let hidden: Float32Array<ArrayBuffer> | undefined;
        for (const [index, chunk] of chunks.entries()) {
          await settleAbort(signal);
          // The physical row count is decided from the valid row count (decode shape / bucket /
          // chunkLength — reasons in {@link physicalChunkRows}). How chunks are **split** stays
          // `chunkLength`; only the rows the last chunk is carried on change.
          const rows = physicalChunkRows(chunk.queryLength, program);
          const ids = new Int32Array(rows);
          const positions = new Int32Array(rows);
          // MUST: pad rows stay 0 (the value contract of ADR 0066 addendum 6).
          const base = context.pastLength;
          for (let row = 0; row < chunk.queryLength; row += 1) {
            ids[row] = promptIds[chunk.position + row];
            positions[row] = base + row;
          }
          const extra = await deriveInputs(ids, positions, signal);
          // MUST: check again after the derived inputs' `await` (the "stage boundary" of ADR 0083
          // decision 5 is **right before issuing** the run). Skipping it lets a whole run proceed
          // after the abort arrived — the first chunk is always a cold miss that reads GB-scale
          // blocks, so "stop right after sending" always hits this window.
          signal?.throwIfAborted();
          const outputs = await session.run(
            {
              [program.inputIds]: i32Row(rows, ids),
              // Select the one last valid row (R=1 — this path does not speculate).
              [program.lastRow]: lastRowInput([chunk.queryLength - 1]),
              ...extra,
            },
            undefined,
            { context, queryLength: chunk.queryLength },
          );
          // Once the first chunk has run, the frontier is in the KV (= it must not be prepended
          // again).
          if (index === 0) pendingToken = undefined;
          const where = `prefill@${chunk.position}`;
          logits = readLogits(outputs, program, where, 1);
          if (tally !== undefined) hidden = copyRow(readHidden(outputs, program, where, 1).row(0));
          onRun?.({ kind: "prefill", chunk: index + 1, chunks: chunks.length });
          yield { kind: "prefill", chunk: index + 1, chunks: chunks.length };
        }
        if (logits === undefined) throw new Error("prefill never ran");

        // Generation starts from the last valid row of the last chunk (the one row `last_row`
        // selected = row 0).
        let token = sampler.next(logits.row(0), history);
        generated += 1;
        history.push(token);
        pendingToken = token;
        const stopped = stopFor(token, generated);
        if (stopped !== undefined) {
          stop = stopped;
          return;
        }
        yield { kind: "token", id: token, position: context.pastLength };

        if (tally !== undefined && face !== undefined) {
          // ---- The speculative path (the "speculative path" section of the module doc, ADR 0096
          // stage 3)
          if (hidden === undefined) throw new Error("prefill did not produce hidden");
          const isStop = (id: number): boolean =>
            isStopToken(id, program.stopTokens) || isStopToken(id, requestStopTokens);
          /**
           * Number of the counted cycle (a gate-induced plain step does not consume a cycle
           * number).
           */
          let cycle = 0;
          /** Cycles run in this turn (the index used to leave the first one out of observation). */
          let turnCycles = 0;
          while (generated < tokenLimit) {
            await settleAbort(signal);
            // k' = min(k, remaining − 1): at most k'+1 are confirmed, so the budget is not exceeded.
            // k' = 0 is one decode row.
            const budget = planDraftLength(k, tokenLimit - generated);
            // Only cycles that can take drafts are submitted to the gate (the forced plain at the end
            // of the budget is not about how well speculation does, so it enters neither the
            // decision nor the observations). MUST: `decide()` once per cycle.
            const gated = budget >= 1 && gate?.decide() === "plain";
            const drafted = gated ? 0 : budget;
            if (gated) plainSteps += 1;
            else cycle += 1;
            /**
             * Whether this cycle goes into the wall observations (the two "not mixed in" kinds of
             * the module doc).
             */
            const measured = gate !== undefined && turnCycles >= 1 && budget >= 1;
            turnCycles += 1;
            // The wall is taken at the same two points every cycle (head → right after the
            // acceptance check). The points do not depend on whether it is observed, so that
            // fake-clock tests do not merely trace the implementation's branches. Nor do they depend
            // on the gate's presence — the hook's `wallMs` carries the same value
            // (`GenerationRunPhase`).
            const startedAt = now();
            let observerMs = 0;
            let drafts: number[] = [];
            if (drafted >= 1) {
              // The drafter produces d₁.. from (frontier b, the hidden of the row that produced b,
              // the position P of b).
              const raw = await face.draft({ token, hidden, position: context.pastLength });
              // A run that returned is reported (even a draft that fails the range check
              // completed its run).
              tally.draftRuns += 1;
              tally.drafted += drafted;
              if (onRun !== undefined) {
                // Diagnostics are delivered at once, tied to the run just finished. Only that cost
                // is excluded from the break-even measurement.
                const observerStartedAt = now();
                onRun({ kind: "draft", cycle });
                observerMs = now() - observerStartedAt;
              }
              drafts = takeDrafts(raw, drafted, program.vocabSize);
            }
            // verify = the k'+1 rows [b, d₁..d_k'] at positions P.. (pad rows stay 0). The physical
            // row count M stays fixed at the `k+1`-row shape even when k' shrinks (like R, it adds
            // no PreparedPlan shape — only `k' = 0` uses the decode shape M=1).
            const queryLength = drafted + 1;
            const rows = physicalChunkRows(drafted === 0 ? 1 : k + 1, program);
            const ids = new Int32Array(rows);
            const positions = new Int32Array(rows);
            const base = context.pastLength;
            ids[0] = token;
            positions[0] = base;
            for (let index = 0; index < drafted; index += 1) {
              ids[index + 1] = drafts[index];
              positions[index + 1] = base + 1 + index;
            }
            const extra = await deriveInputs(ids, positions, signal);
            signal?.throwIfAborted();
            const rowIndices = verifyRowIndices(drafted, k);
            const where = gated ? `plain@${plainSteps}` : `verify@${cycle}`;
            const outputs = await session.run(
              {
                [program.inputIds]: i32Row(rows, ids),
                [program.lastRow]: lastRowInput(rowIndices),
                ...extra,
              },
              undefined,
              // MUST: deferred — the logical length advances to the frontier once the accepted count
              // is known (settleCommit).
              { context, queryLength, commit: "deferred" },
            );
            // ---- Synchronous from here to the first yield (an exception is folded by finally's
            // settleCommit as commit(0)).
            frontierRows = 0;
            const logits = readLogits(outputs, program, where, rowIndices.length);
            const hiddenRows = readHidden(outputs, program, where, rowIndices.length);
            const { accepted, confirmed } = acceptDrafts(
              logits.row,
              drafts,
              sampler,
              history,
              isStop,
            );
            // The gate's measure is taken right after the acceptance check, before delivery, and
            // excludes the draft's diagnostics cost. Mixing in the speed of consumers or observers
            // would change the gate's decision for the same inference.
            // MUST: the `wallMs` put on the hook is this same variable (measuring it separately
            // would make the wall the gate saw and the wall in the breakdown disagree, and "why it
            // flipped" could no longer be traced from the breakdown).
            const wall = now() - startedAt - observerMs;
            if (gate !== undefined) {
              // A cycle that is not mixed in calls `skip()` instead of an observation (the MUST of
              // the gate's calling convention) — reporting nothing would stop the exploration
              // period, and the first `W1` sample would slip from the turn's second cycle to "the
              // first cycle mixed into the observations".
              if (!measured) gate.skip();
              else if (gated) gate.observePlain(wall);
              else gate.observeCycle(wall, confirmed.length);
            }
            // The gate's state is read **after the observation** (the MUST of
            // {@link GenerationGateTrace}).
            const gateTrace: GenerationGateTrace | undefined =
              gate === undefined
                ? undefined
                : {
                    mode: gate.mode,
                    switches: gate.switches - switchesAtStart,
                    measured,
                  };
            // The drafter input for the next cycle = row a, which produced the new frontier b'
            // (copied — it is gone after the next run). A gate-induced plain step is also one row of
            // the verify shape, so the hidden stays connected here.
            const nextHidden = copyRow(hiddenRows.row(accepted));
            // Only speculative cycles are counted (the gate's plain steps are counted by
            // `plainSteps`).
            if (!gated) {
              tally.cycles += 1;
              tally.accepted += accepted;
              tally.acceptedHistogram[accepted] += 1;
              tally.delivered += confirmed.length;
            }
            // The verify observation happens **in this synchronous section** (across the delivering
            // yield, a run of another sequence sharing the lending Session can slip in). The commit
            // comes after delivery (up to the frontier).
            onRun?.(
              gated
                ? {
                    kind: "decode",
                    step: plainSteps,
                    wallMs: wall,
                    ...(gateTrace === undefined ? {} : { gate: gateTrace }),
                  }
                : {
                    kind: "verify",
                    cycle,
                    rows: queryLength,
                    accepted,
                    delivered: confirmed.length,
                    wallMs: wall,
                    ...(gateTrace === undefined ? {} : { gate: gateTrace }),
                  },
            );
            // `confirmed` is exactly the delivery sequence (acceptance enumerates nothing past a
            // stop token).
            for (let index = 0; index < confirmed.length; index += 1) {
              const id = confirmed[index];
              // MUST: update the frontier before the yield (when `break` enters finally, everything
              // up to the frontier is committed = the conversation is the sequence the consumer
              // received + one frontier).
              token = id;
              generated += 1;
              history.push(id);
              pendingToken = id;
              frontierRows = index + 1;
              const stopped = stopFor(id, generated);
              if (stopped !== undefined) {
                stop = stopped;
                settleCommit();
                return;
              }
              yield { kind: "token", id, position: context.pastLength + 1 + index };
            }
            settleCommit();
            hidden = nextHidden;
          }
          stop = limitStop(generated);
          return;
        }

        if (runAhead !== undefined && aheadTally !== undefined) {
          // ---- The run-ahead path: the per-step decode below, `n` steps per batch. A batch feeds
          // `token` to step 0 and each later step the previous step's selection on the GPU, leaves
          // the `n` rows pending, and the walk below hands the tokens out exactly as the per-step
          // loop would — then commits the rows that loop would have committed (`settleCommit`, up to
          // `frontierRows`). Its decode runs `tokenLimit - 1` steps too, so the last token stays
          // uncommitted in `pendingToken` (decision 4).
          let step = 0;
          while (step + 1 < tokenLimit) {
            // The stage boundary is the batch (one fence per batch); a token of a batch already run
            // is checked synchronously before it is handed out (below).
            await settleAbort(signal);
            const count = Math.min(runAhead.depth, tokenLimit - 1 - step);
            const base = context.pastLength;
            const steps: RunInputs[] = [];
            for (let k = 0; k < count; k += 1) {
              // The derived inputs are positions-only (checked at creation): step k > 0's token is
              // still on the GPU, so the pad value 0 stands in for it and is never read.
              const extra = await deriveInputs(
                Int32Array.of(k === 0 ? token : 0),
                Int32Array.of(base + k),
                signal,
              );
              steps.push({ [program.lastRow]: lastRowInput([0]), ...extra });
            }
            signal?.throwIfAborted();
            const selected = await runAhead.run({ token, steps, context });
            aheadTally.batches += 1;
            aheadTally.steps += count;
            if (selected.length !== count) {
              throw new Error(
                `greedyAhead returned ${selected.length} selections for a batch of ${count} steps`,
              );
            }
            // ---- From here every exit commits through `settleCommit` (here, at a stop, or in
            // `finally` for a throw / an interruption).
            frontierRows = 0;
            for (let k = 0; k < count; k += 1) {
              // An interruption that arrived while the previous token was handed out closes here,
              // where the per-step loop would have closed before its next run (rows 0..k−1 kept).
              if (k > 0) signal?.throwIfAborted();
              // MUST: row k is consumed (the frontier went into the KV) **before** the draw — the
              // same point at which the per-step loop's run had committed it, so a throwing draw
              // (NaN logits) keeps the row, and the old frontier is not fed twice.
              frontierRows = k + 1;
              pendingToken = undefined;
              token = pickGreedy(selected[k]);
              generated += 1;
              history.push(token);
              pendingToken = token;
              step += 1;
              const stopped = stopFor(token, generated);
              if (stopped !== undefined) {
                stop = stopped;
                settleCommit();
                return;
              }
              yield { kind: "token", id: token, position: context.pastLength + frontierRows };
            }
            settleCommit();
          }
          stop = limitStop(generated);
          return;
        }

        // Decode has the shape "placing `g_i` at position P yields `g_{i+1}`". It runs
        // `tokenLimit - 1` times, and the last token stays uncommitted in `pendingToken`
        // (decision 4).
        for (let step = 0; step + 1 < tokenLimit; step += 1) {
          await settleAbort(signal);
          // The wall runs from the step's head (before the derived inputs) to right after the draw
          // — the same span as a speculative cycle, taken before the delivering yield, so the
          // consumer's speed is excluded. Only the hook's `wallMs` reads this value.
          const startedAt = now();
          const ids = Int32Array.of(token);
          const positions = Int32Array.of(context.pastLength);
          const extra = await deriveInputs(ids, positions, signal);
          // Check again right before issuing the run, for the same reason as on the prefill side
          // (hitting it in decode delivers one extra token to the consumer).
          signal?.throwIfAborted();
          const inputs: RunInputs = {
            [program.inputIds]: i32Row(1, ids),
            [program.lastRow]: lastRowInput([0]),
            ...extra,
          };
          // Prefill keeps the normal run, avoiding the cost of allocating a large backing from the
          // first turn.
          const result =
            greedy === undefined
              ? {
                  kind: "logits" as const,
                  outputs: await session.run(inputs, undefined, { context, queryLength: 1 }),
                }
              : {
                  kind: "greedy" as const,
                  selected: await greedy(inputs, { context, queryLength: 1 }),
                };
          // MUST: once the run has gone through, the frontier is in the KV (= it must not be
          // prepended again) — for the same reason as `index === 0` on the prefill side, clear it
          // **before** the draw. Placed after the draw, a throw from `sampler.next` (NaN logits,
          // say) would leave the old frontier, which would be prepended to the next turn's prompt
          // and **put the same token at two positions** (silent degradation without an exception).
          pendingToken = undefined;
          token =
            result.kind === "greedy"
              ? pickGreedy(result.selected)
              : sampler.next(
                  readLogits(result.outputs, program, `decode@${step}`, 1).row(0),
                  history,
                );
          onRun?.({ kind: "decode", step: step + 1, wallMs: now() - startedAt });
          generated += 1;
          history.push(token);
          pendingToken = token;
          const stopped = stopFor(token, generated);
          if (stopped !== undefined) {
            stop = stopped;
            return;
          }
          yield { kind: "token", id: token, position: context.pastLength };
        }
        stop = limitStop(generated);
      } catch (error) {
        if (isAbortOf(error, signal)) stop = { reason: "aborted", tokens: generated };
        else failure = { error };
        // MUST: throw it as is, unwrapped (ADR 0083 decision 5 — the consumer can identify its own
        // abort with `error === controller.signal.reason`).
        throw error;
      } finally {
        try {
          // MUST: commit the pending verify up to the frontier (whether we came via `break` /
          // `return()` / an exception — leaving it pending makes the context accept only dispose).
          // `commit` itself is an in-range synchronous call; a throw here means broken bookkeeping
          // (not swallowed silently — but the chain's slot is still released).
          settleCommit();
        } catch (cleanup) {
          // Do not overwrite the original failure (a context poisoned by device loss, for example)
          // — a pending commit cannot be read on a poisoned context, and dispose discards it all.
          // A commit failing without an earlier failure is broken bookkeeping; it cannot be thrown
          // from finally (no-unsafe-finally), so it travels the secondary path (`done`'s reject).
          failure ??= { error: cleanup };
        } finally {
          if (failure !== undefined) fail(failure.error);
          // `stop` is still empty in `finally` only via `break` / `return()`.
          else settle(withCounters(stop ?? { reason: "closed", tokens: generated }));
          release?.();
        }
      }
    };

    const iterable = closeableGenerator(events(), (failure) => {
      if (failure !== undefined) fail(failure.error);
      else settle(withCounters({ reason: "closed", tokens: 0 }));
    });
    return { [Symbol.asyncIterator]: () => iterable, done };
  };

  return {
    capacity,
    // MUST: derive it in the getter every time (`context.pastLength`, `frontierRows` and
    // `pendingToken` are the only sources). During speculative delivery the verify stays pending
    // (`pastLength` stays at the start of the cycle — and a run-ahead batch likewise until its
    // tokens are handed out), so the uncommitted rows up to the frontier
    // are added — this makes **`used === the delivered token's position + 1` hold even during
    // delivery**, speculative or not (without it, delivered tokens would be missing mid-cycle and
    // the next turn's budget calculation would be lax by up to k).
    get used(): number {
      return context.pastLength + frontierRows + (pendingToken === undefined ? 0 : 1);
    },
    generate,
    fork(): Promise<GenerationSequence> {
      // MUST: the lifetime check is **synchronous** (as in `generate` — the context is not exposed,
      // so the caller has no other way to find out it was disposed).
      if (disposal !== undefined) {
        throw new Error("GenerationSequence: cannot fork after dispose");
      }
      return chain(async () => {
        // MUST: the three things copied (frontier, gate, context) are captured at the same point
        // inside the chain's slot (inside the slot is between generates = `frontierRows` is 0 and
        // nothing deferred is pending). `context.fork()` captures the logical length and bytes in
        // the synchronous section where it is called, so it is called next to the frontier and gate
        // copies without an intervening await.
        const inherited: BranchState = { pendingToken, gate: gate?.fork() };
        const forked = await context.fork();
        // If assembly (reopening the drafter) fails, assembleSequence closes the copied context
        // before rethrowing (the original is untouched).
        return await assembleSequence(options, capacity, forked, inherited);
      });
    },
    dispose(): Promise<void> {
      // MUST: return the same completion on the second and later calls (returning earlier would let
      // the caller grab the window before disposal).
      // MUST: borrower (the drafter face) → lending context. Failures are collected and every step
      // runs (if a rejected earlier step kept later steps from running, "a context that can never
      // be returned" would remain).
      disposal ??= chain(() => disposeSteps([() => face?.dispose(), () => context.dispose()]));
      return disposal;
    },
  };
};

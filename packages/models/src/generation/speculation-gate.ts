/**
 * The **self-tuning gate** of speculation (ADR 0096 stage 4-B (4)) — it measures at run time
 * "whether speculation is beating decode", and in contexts where it loses it drops to one step of
 * the decode shape (M=1). It is a pure-function module that is **pipeline-independent,
 * GPU-independent and owns neither a clock nor an RNG**, so it lives in `src/generation/` next to
 * `speculation.ts`.
 *
 * ## Why measure at run time (no fixed threshold can be baked in)
 *
 * What speculation gains depends on the task and the host (measured in
 * `docs/research/2026-09-09-mtp-stage4.md` — RTX 3080 Ti: extraction 1.81× / summary 1.41× /
 * dialogue 1.18× / free text **0.96×**; Apple M2: extraction 1.27× / summary 1.03× / dialogue
 * 0.93× / free text **0.76×**). The break-even accepted count `A* = cycle wall ÷ decode wall` also
 * moves by 0.5 or more with the host (RTX 1.72 to 1.88, M2 2.17 to 2.6), so it can be baked into
 * neither the distribution nor a constant. The gate measures only three quantities and switches to
 * whichever side the ratio
 *
 * ```text
 * (ΣWc / Σdelivered) / W1   … speculation's wall per token ÷ decode's wall per token
 * ```
 *
 * crosses 1 towards (below 1, speculation is faster). `Wc` = the wall of a speculative cycle,
 * `delivered` = the tokens that cycle confirmed, `W1` = the wall of a plain step (M=1); all are
 * values the caller measured.
 *
 * ## The estimator is **block aggregation** (no moving average over the accepted count)
 *
 * The denominator `delivered` scatters widely per cycle — the measured distribution of the accepted
 * count (RTX dialogue) has mean 2.03 and **standard deviation 0.88** (43%). One sample or the mean of
 * a few cannot decide "speculation is losing". So the speculate side groups `window` (default 16)
 * cycles into **non-overlapping blocks** and evaluates `ΣWc / Σdelivered` once each time a block
 * fills (the block is emptied after the evaluation). It switches to plain only when
 * `msPerTokenSpec > W1 × leave` holds for `confirm` (default 2) blocks **in a row**.
 *
 * **The early leave on a strong loss is off by default** ({@link SpeculationGateOptions.earlyLeave}
 * — only when it is passed, the gate switches as soon as one block's ratio exceeds
 * `leave + earlyLeave`, without waiting for `confirm` in a row). Acceptance is not steady within a
 * turn; even on a winning task (dialogue, ratio 0.90) stretches of about 16 cycles with low
 * acceptance occur. The early leave saves only "the cost until leaving", one block = once per
 * sequence lifetime, while the harm of a false trigger appears on every turn of a winning task (with
 * a reused sequence, 3 of 7 turns left by mistake and auto fell to always's 0.96× — with the early
 * leave stopped (v2's full set of knobs; the burst-side knobs act only on the plain side, so the
 * difference is attributable to the early leave) it was 0 / 7 at 1.00×,
 * `docs/research/2026-09-09-mtp-stage4.md` §6.5 table 11″). So it is not taken by default.
 *
 * **EWMA is not used because of its seeding defect**. `ewma(undefined, sample) = sample`, so an
 * EWMA of the accepted count is **seeded by the first sample** — if the first observed cycle accepts
 * 0 (1 delivered), `A = 1` and the ratio = `Wc / 1 / W1 ≈ 1.7 > leave` leaves for plain at once.
 * When the plain-side exploration is a single cycle it draws the same noise and cannot come back;
 * only the backoff grows. This is the shape observed in a real run, `200 token / 7 cycle / plain
 * step 190 / switch 1` — it dropped to plain at cycle 7, did not come back for 190 steps, and took
 * 9,262 ms against 7,066 ms for `always` (no gate) (**31% slower**). Block aggregation does not
 * decide until a block fills, so this seeding has no room to happen.
 *
 * ## The dead band is **asymmetric** (it guards only against "large losses")
 *
 * The gate leaves for plain at ratio > {@link SpeculationGateOptions.leave} (default 1.01) and
 * returns to speculate at ratio < {@link SpeculationGateOptions.enter} (default 0.97). With a
 * symmetric 5% dead band, the largest loss to guard against, RTX free text (ratio 1.037), would fall
 * inside the band and **never trigger**. What the gate actually rescues are losses of the M2 free
 * text −24% / M2 dialogue −7% class, and the acceptance criteria are read at that granularity.
 *
 * ## Back-of-the-envelope basis (why the default knobs have these values)
 *
 * Taking the per-cycle standard deviation of the accepted count as 0.88 (the measured RTX free-text
 * distribution — dialogue scatters more at 1.14):
 *
 * - **Dialogue class (ratio 0.85, A 2.03)**: the sd of `A` over a 16-cycle block ≈ 0.22, so the sd
 *   of the ratio ≈ 0.09. The probability that one block exceeds 1.01 ≈ 5%, two blocks in a row ≈
 *   0.3%. A 200-token turn is ≈ 6 blocks, so **a false leave happens in ≈ 1.5% of turns**. With the
 *   one-sample EWMA seeding (above) it would leave almost certainly under the same conditions; this
 *   is where `confirm = 2` pays off.
 * - **M2 free text (ratio 1.31)**: the probability that one block falls below 1.01 ≈ 0.1%. So it
 *   reliably leaves after 2 blocks = 32 cycles (≈ 3.8 s). That is a cost paid once per sequence
 *   lifetime (halving this once is what the early leave does; the reason it is off by default is in
 *   the section above).
 * - **Returning**: a burst of `burst` (default 8) cycles has a ratio sd ≈ 0.13. For M2 free text
 *   the probability of falling below 0.97 ≈ 0.5%, so it does not come back by mistake. The burst
 *   cost the losing side pays is +227 ms per burst for M2 free text, and it fades in long
 *   conversations as the interval grows from 16 to 512.
 * - **Early leave (when `earlyLeave` = 0.15 is passed)**: the block ratio sd of the dialogue class
 *   (ratio 0.85) is the 0.09 above, so the probability of exceeding 1.16 is ≈ 0.03% on paper.
 *   **This estimate does not match real runs**, which is why it is off by default: when acceptance
 *   is not steady within a turn, even a winning task has stretches where one block exceeds 1.16 (the
 *   3 / 7 turns of the section above). A small probability on paper is no basis.
 * - **Early burst abort (`burstMin` = 4, `burstAbort` = 0.15)**: at ratio 0.85 the probability that
 *   the 4-cycle ratio (sd ≈ 0.19) exceeds 1.16 is ≈ 5%. This false trigger can only happen in "a
 *   burst after a false leave", and its only cost is that returning is delayed by one burst (this
 *   is why, unlike the early leave, it stays on by default — the harm is bounded by one exploration
 *   and contexts that are winning are not touched). On the effective side, at ratio 1.27 (M2 free
 *   text class) the probability that the 4-cycle ratio exceeds 1.16 is ≈ 0.72, so the mean burst
 *   length in losing contexts shrinks from 8 cycles to **≈ 4.8 cycles** (the value from running the
 *   gate itself on the measured distribution; 70% of the aborts happen at the 4th cycle).
 *
 * ## Exploration (a "way back" even for the losing side)
 *
 * - **While plain**, speculation is tried for `burst` cycles **in a row** with geometric backoff
 *   (16 → 32 → 64 → … → at most 512 steps). Single-cycle exploration is buried in the sd 0.88 above
 *   and cannot decide (the second reason the EWMA design could not come back). If the burst's
 *   `ΣWc / Σdelivered < W1 × enter`, the gate returns to speculate and the interval goes back to
 *   `exploreBase`. On a miss the interval doubles.
 * - **A burst whose miss is certain is not run to the end**. If at any observation from the
 *   `burstMin`-th (default 4) on the ratio exceeds `leave + burstAbort` (default 1.16), the burst is
 *   folded as "a miss" without running the remaining cycles ({@link SpeculationGateOptions.burstMin}
 *   / {@link SpeculationGateOptions.burstAbort}). This is the main exploration cost losing contexts
 *   pay, and unlike shrinking `burst` itself **it does not lower the decision accuracy on the side
 *   that can return** (a hit is still decided on the aggregate of a full burst).
 * - **While speculating**, plain is measured once every {@link SpeculationGateOptions.exploreBase}
 *   observations (the freshness of `W1` is itself the material of the "leave" decision). The default
 *   is 16, the same as the decision block, and the cost is one token at plain price per 16 cycles
 *   (about 1% against always in measured walls). This plain step **does not count towards the
 *   block's cycles** (only the aggregate of speculative cycles makes up a block).
 * - **The first `W1` sample is at the second decide** (right after one cycle). Running 16 cycles
 *   without it loses 2% up front on M2 free text. `W1` is the wall of a plain step and is stable on a
 *   host, so for this alone EWMA (factor {@link SpeculationGateOptions.alpha}) is fine.
 *
 * ## Calling convention
 *
 * MUST: call {@link SpeculationGate.decide} once per cycle and report that cycle's observation once
 * with {@link SpeculationGate.observeCycle} / {@link SpeculationGate.observePlain}. `decide()` is
 * **a read that moves no state at all** (mode switches and `switches` increments happen on the
 * observation side), so reading it twice in one cycle does not split the decision.
 *
 * MUST: for a cycle the caller decided "not to mix into the wall observations" (the first cycle of
 * each turn and the forced plain at the end of the budget), call {@link SpeculationGate.skip} once
 * instead of an observation. Moving on to the next cycle without calling anything makes the gate
 * decide as if that cycle never happened, so the exploration period stops, and "the first `W1`
 * sample is at the second decide" above is delayed to "the first cycle mixed into the
 * observations".
 */

import { ModelInputError } from "../errors.ts";

/** Whether the next cycle runs speculatively or as one step of the decode shape (M=1). */
export type SpeculationDecision = "speculate" | "plain";

/**
 * The gate's knobs (all have defaults — **the defaults suffice**).
 *
 * They are on the public surface as a static pipeline knob (`gate` of
 * `gemma.PipelineOptions.speculative`), but the option is meant for measurement and acceptance
 * (the same treatment as `linearGemvRowsThreadTarget` — it exists only to run A/B of the knobs with
 * the same script).
 */
export type SpeculationGateOptions = {
  /** EWMA factor of `W1` (the wall of a plain step) (default 0.2, `0 < alpha ≤ 1`). */
  readonly alpha?: number;
  /** Cycles per decision block on the speculate side (default 16, an integer of 2 or more). */
  readonly window?: number;
  /** "Losing blocks" in a row needed to leave for plain (default 2, an integer of 1 or more). */
  readonly confirm?: number;
  /** Cycles per exploration burst on the plain side (default 8, an integer of 1 or more). */
  readonly burst?: number;
  /**
   * The smallest cycle count at which a burst can be aborted early (default `min(4, burst)`, an
   * integer with `1 ≤ burstMin ≤ burst`). If at any observation from here up to the `burst - 1`-th
   * the ratio exceeds `leave + burstAbort`, the burst is folded as "a missed burst" without running
   * the rest.
   */
  readonly burstMin?: number;
  /** Base exploration interval (default 16, in cycles / steps). */
  readonly exploreBase?: number;
  /** Upper limit of the plain-side backoff (default 512). */
  readonly exploreMax?: number;
  /** Condition to return to speculate, `ΣWc/Σdelivered < W1 × enter` (default 0.97, `enter < 1`). */
  readonly enter?: number;
  /** Condition to leave for plain, `ΣWc/Σdelivered > W1 × leave` (default 1.01, `1 < leave`). */
  readonly leave?: number;
  /**
   * Margin of the speculate side's **early leave** (**unset = off**; when given, a positive finite
   * number). When passed, the gate switches to plain as soon as one block's ratio exceeds
   * `leave + earlyLeave`, without waiting for `confirm` in a row. It has no default because in real
   * runs false triggers appear on every turn even for winning tasks, while the gain is "one block
   * until leaving" = once per sequence lifetime (module doc).
   */
  readonly earlyLeave?: number;
  /**
   * Margin of the "strong loss" that folds a plain-side exploration burst (default 0.15, a positive
   * finite number). A burst whose ratio exceeds `leave + burstAbort` at an observation from the
   * `burstMin`-th on is folded as "a miss" without running the remaining cycles. It has no effect on
   * the speculate side's decision (`leave` / `confirm`).
   */
  readonly burstAbort?: number;
};

export type SpeculationGate = {
  /**
   * Which one runs next (a deterministic **read** of the observations and internal counters only —
   * the clock is received by observe*).
   *
   * MUST: call once per cycle. It moves no state at all, so reading it any number of times in one
   * cycle gives the same value, and the decision need not be carried over (mode switches and
   * `switches` increments happen on the observation side, when a block / burst fills).
   */
  decide(): SpeculationDecision;
  /**
   * Observation of a speculative cycle (wall ms and the tokens that cycle delivered =
   * `confirmed.length`).
   *
   * While speculating it goes into the block, while plain into the exploration burst. The decision
   * moves only when they fill, and when a burst shows a strong loss from the `burstMin`-th cycle on
   * (that burst is aborted).
   */
  observeCycle(wallMs: number, delivered: number): void;
  /** Observation of a plain step (M=1) (wall ms) — it moves only the EWMA of `W1`. */
  observePlain(wallMs: number): void;
  /**
   * A cycle that reports no observation (a cycle the caller decided "not to mix into the wall
   * observations" — including cycles whose shape was fixed without consulting the decision).
   *
   * Only the exploration period (what remains until the next exploration) moves; neither the mode,
   * `W1` nor the block move. Even a cycle that is not mixed in runs one piece of GPU work, so
   * advancing the period by it is correct.
   */
  skip(): void;
  /** Number of speculate ↔ plain switches (exploration bursts themselves are not counted). */
  readonly switches: number;
  /** The current steady mode (it does not change while an exploration burst runs the other side). */
  readonly mode: SpeculationDecision;
  /**
   * Returns an **independent** gate with a copy of the observed state (mode, `W1`, exploration
   * period and interval, switch count, the group being aggregated, the count of consecutive losses)
   * (the knobs are the same checked values). Used by conversation branching
   * (`GenerationSequence.fork` — ADR 0083 addendum 2026-10-11): a branch continues the same
   * conversation on the same host, so it decides from where it was without measuring again (the
   * same reason the gate lives as long as the sequence).
   *
   * MUST: the two do not share state after the copy (an observation on one does not move the other
   * — sharing would let one branch's walls flip the other's mode).
   */
  fork(): SpeculationGate;
};

/**
 * The gate's observed state (everything {@link SpeculationGate.fork} copies — the knobs excluded).
 */
type GateState = {
  readonly mode: SpeculationDecision;
  readonly plainWall: number | undefined;
  readonly sinceProbe: number;
  readonly backoff: number;
  readonly switches: number;
  readonly batchCycles: number;
  readonly batchWall: number;
  readonly batchDelivered: number;
  readonly overRuns: number;
};

/**
 * Check for a positive finite number (ratio and factor knobs).
 *
 * Every knob is a value the caller passes through `speculative.gate`, so a rejection is
 * input-caused (ADR 0107 decision 2 — the only remedy is "fix the value passed").
 */
const assertPositive = (name: string, value: number): void => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new ModelInputError(`speculation gate: ${name} ${value} is not a positive finite number`);
  }
};

/** Check for a cycle count — a fractional cycle count cannot mean "every so many cycles". */
const assertInterval = (name: string, value: number, least: number): void => {
  if (!Number.isSafeInteger(value) || value < least) {
    throw new ModelInputError(`speculation gate: ${name} ${value} is not an integer ≥ ${least}`);
  }
};

/**
 * Checked knobs (after {@link createSpeculationGate} checked their ranges — a branch reuses them).
 */
type GateKnobs = {
  readonly alpha: number;
  readonly window: number;
  readonly confirm: number;
  readonly burst: number;
  readonly burstMin: number;
  readonly exploreBase: number;
  readonly exploreMax: number;
  readonly enter: number;
  readonly leave: number;
  readonly earlyLeave: number | undefined;
  readonly burstAbort: number;
};

/**
 * Builds a gate from checked knobs and an observed state (the one function shared by
 * {@link createSpeculationGate} and {@link SpeculationGate.fork}).
 *
 * MUST: knob ranges are not checked here (callers pass only checked values — a branch passes the
 * values the original gate checked as they are).
 */
const openGate = (knobs: GateKnobs, initial: GateState): SpeculationGate => {
  const {
    alpha,
    window,
    confirm,
    burst,
    burstMin,
    exploreBase,
    exploreMax,
    enter,
    leave,
    earlyLeave,
    burstAbort,
  } = knobs;
  let mode: SpeculationDecision = initial.mode;
  /** EWMA of the plain step's wall ms (`undefined` until observed). */
  let plainWall: number | undefined = initial.plainWall;
  /**
   * Observations in the current mode (back to 0 on a mode switch and when a plain burst completes).
   */
  let sinceProbe = initial.sinceProbe;
  /** Plain-side exploration interval (doubles on every miss, back to `exploreBase` on a hit). */
  let backoff = initial.backoff;
  let switches = initial.switches;
  /**
   * The group being aggregated (a block of `window` cycles while speculating, an exploration burst
   * of `burst` cycles while plain).
   *
   * One set of variables can hold both because a mode switch always empties it (a block and a burst
   * are never alive at the same time).
   */
  let batchCycles = initial.batchCycles;
  let batchWall = initial.batchWall;
  let batchDelivered = initial.batchDelivered;
  /** Blocks in a row that exceeded `leave` (leaves when it reaches `confirm`). */
  let overRuns = initial.overRuns;

  const ewma = (value: number | undefined, sample: number): number =>
    value === undefined ? sample : value + alpha * (sample - value);

  const resetBatch = (): void => {
    batchCycles = 0;
    batchWall = 0;
    batchDelivered = 0;
  };

  /**
   * Ratio of the group's wall ms per token (`ΣWc / Σdelivered`) to `W1`.
   *
   * At least one token is confirmed per cycle, so `delivered` is never 0, but it is checked before
   * dividing (switching while `W1` is unobserved would be a switch without basis).
   */
  const batchRatio = (): number | undefined =>
    batchDelivered <= 0 || plainWall === undefined || plainWall <= 0
      ? undefined
      : batchWall / batchDelivered / plainWall;

  /**
   * Switches to plain (when blocks lost `confirm` times in a row / when, with `earlyLeave` set, one
   * block was a strong loss).
   */
  const leaveSpeculation = (): void => {
    mode = "plain";
    switches += 1;
    backoff = exploreBase;
    sinceProbe = 0;
    overRuns = 0;
    resetBatch();
  };

  /** Folds a missed burst (full without falling below `enter` / a strong loss midway). */
  const missBurst = (): void => {
    resetBatch();
    // Push the next exploration further out and restart counting the interval.
    sinceProbe = 0;
    backoff = Math.min(backoff * 2, exploreMax);
  };

  /** Returns to speculate (when an exploration burst fell below `enter`). */
  const enterSpeculation = (): void => {
    mode = "speculate";
    switches += 1;
    backoff = exploreBase;
    sinceProbe = 0;
    overRuns = 0;
    resetBatch();
  };

  return {
    /**
     * Decides from the current state (a read only — the MUST of {@link SpeculationGate.decide}).
     *
     * The plain side: "continue during a burst / start a burst once the interval has run". The
     * speculate side: "if `W1` is unobserved, one step right after one cycle / afterwards one step
     * every `exploreBase`".
     */
    decide: (): SpeculationDecision => {
      if (mode === "plain") {
        // `batchCycles > 0` = an exploration burst is in progress (back to 0 when it fills and on
        // an abort from `burstMin` on). This read alone makes an abort return to plain at once.
        return batchCycles > 0 || sinceProbe >= backoff ? "speculate" : "plain";
      }
      // If `W1` has never been measured, insert a single plain step right after one cycle.
      if (plainWall === undefined) return sinceProbe >= 1 ? "plain" : "speculate";
      return sinceProbe % exploreBase === exploreBase - 1 ? "plain" : "speculate";
    },
    observeCycle: (wallMs: number, delivered: number): void => {
      batchCycles += 1;
      batchWall += wallMs;
      batchDelivered += delivered;
      if (mode === "plain") {
        // An exploration burst while plain — no "return" decision and no period restart until it
        // fills. From the `burstMin`-th cycle on, though, it watches for a strong loss only and
        // folds a burst whose miss is certain.
        if (batchCycles < burst) {
          if (batchCycles < burstMin) return;
          const probe = batchRatio();
          if (probe !== undefined && probe > leave + burstAbort) missBurst();
          return;
        }
        const measured = batchRatio();
        if (measured !== undefined && measured < enter) {
          enterSpeculation();
          return;
        }
        missBurst();
        return;
      }
      sinceProbe += 1;
      if (batchCycles < window) return;
      // The decision for one full block (with `W1` unobserved there is no decision = the run is
      // broken).
      const measured = batchRatio();
      // Only when the early leave is passed, a strong loss does not wait for `confirm` in a row
      // (one block switches).
      if (earlyLeave !== undefined && measured !== undefined && measured > leave + earlyLeave) {
        leaveSpeculation();
        return;
      }
      overRuns = measured !== undefined && measured > leave ? overRuns + 1 : 0;
      if (overRuns >= confirm) leaveSpeculation();
      else resetBatch();
    },
    observePlain: (wallMs: number): void => {
      plainWall = ewma(plainWall, wallMs);
      sinceProbe += 1;
    },
    skip: (): void => {
      sinceProbe += 1;
    },
    get switches(): number {
      return switches;
    },
    get mode(): SpeculationDecision {
      return mode;
    },
    fork: (): SpeculationGate =>
      openGate(knobs, {
        mode,
        plainWall,
        sinceProbe,
        backoff,
        switches,
        batchCycles,
        batchWall,
        batchDelivered,
        overRuns,
      }),
  };
};

export const createSpeculationGate = (options: SpeculationGateOptions = {}): SpeculationGate => {
  const alpha = options.alpha ?? 0.2;
  const window = options.window ?? 16;
  const confirm = options.confirm ?? 2;
  const burst = options.burst ?? 8;
  // The default is capped by `burst` because with a shrunk `burst` (a 1-cycle burst, say) "no room
  // to abort" is the correct shape, and there is no reason for the check to reject a default
  // combination.
  const burstMin = options.burstMin ?? Math.min(4, burst);
  const exploreBase = options.exploreBase ?? 16;
  const exploreMax = options.exploreMax ?? 512;
  const enter = options.enter ?? 0.97;
  const leave = options.leave ?? 1.01;
  // The early leave is **unset = off** (having no value is itself how "do not switch" is spelled).
  const earlyLeave = options.earlyLeave;
  const burstAbort = options.burstAbort ?? 0.15;
  assertPositive("alpha", alpha);
  // An EWMA with α > 1 puts negative weight on the past (it stops being an average), so the upper
  // end is closed too.
  if (alpha > 1) throw new ModelInputError(`speculation gate: alpha ${alpha} is outside 0 < α ≤ 1`);
  // A 1-cycle block turns "block aggregation" back into a one-sample decision (the seeding defect
  // itself).
  assertInterval("window", window, 2);
  assertInterval("confirm", confirm, 1);
  assertInterval("burst", burst, 1);
  assertInterval("burstMin", burstMin, 1);
  // `burstMin > burst` is a dead knob ("the abort never fires"), so it is not passed silently.
  if (burstMin > burst) {
    throw new ModelInputError(`speculation gate: burstMin ${burstMin} exceeds burst ${burst}`);
  }
  assertInterval("exploreBase", exploreBase, 1);
  assertInterval("exploreMax", exploreMax, exploreBase);
  assertPositive("enter", enter);
  assertPositive("leave", leave);
  if (earlyLeave !== undefined) assertPositive("earlyLeave", earlyLeave);
  assertPositive("burstAbort", burstAbort);
  // The dead band straddles 1 (with `enter ≥ 1` it returns while losing, with `leave ≤ 1` it leaves
  // even while winning).
  if (!(enter < 1 && 1 < leave)) {
    throw new Error(`speculation gate: enter ${enter} < 1 < leave ${leave} does not hold`);
  }

  return openGate(
    {
      alpha,
      window,
      confirm,
      burst,
      burstMin,
      exploreBase,
      exploreMax,
      enter,
      leave,
      earlyLeave,
      burstAbort,
    },
    {
      mode: "speculate",
      plainWall: undefined,
      sinceProbe: 0,
      backoff: exploreBase,
      switches: 0,
      batchCycles: 0,
      batchWall: 0,
      batchDelivered: 0,
      overRuns: 0,
    },
  );
};

// Behaviour tests of the speculation self-accounting gate (ADR 0096 step 4-B ④). It is a pure function, so it needs
// neither a GPU nor a clock (the wall times are passed by the caller = the very constants this test passes).
//
// Four things are pinned here:
//
// 1. "In a losing context it falls back to the decode shape, in a winning one it stays speculative" — the decision
//    lands on the correct side for **the measured numbers** (`docs/research/2026-09-09-mtp-stage4.md` §2 table 1 /
//    §5.1) (T8). The threshold moves between 1.7 and 2.6 per host, so it cannot be hard-coded; the gate has to
//    measure it itself.
// 2. **It does not decide on a single sample** (T2 / T3). The accepted count scatters with an sd of 0.88 per cycle
//    (43% of the mean of 2.03), so deciding on the seeded moving average (`ewma(undefined, sample) = sample`) would
//    drop a conversation whose first cycle accepted 0 straight into plain. That is the defect observed in a real
//    run: `200 tokens / 7 cycles / plain step 190 / 1 switch`, 9,262 ms against always's 7,066 ms. The estimator is
//    the sum of `window` non-overlapping blocks, and only `confirm` consecutive losses justify leaving.
// 3. **The "strong loss" knobs are separate for the speculate side and the plain side** (T2′ / T4″). The speculate
//    side's early leave (`earlyLeave`) is **unset = off**, so by default even a ratio of 1.30 waits for `confirm`
//    consecutive blocks (in real runs, even winning tasks mix in stretches where one block exceeds 1.16, and in a
//    warm conversation 3 of 7 turns left by mistake). The plain side's abort (`burstAbort`, default 0.15) stays — a
//    false trigger can only happen in "the burst after a mistaken leave", and its cost is one exploration.
// 4. **A branch (`fork`) is an independent gate that copies the observation state** (T10 — ADR 0083 addendum
//    2026-10-11). Without copying, a branched conversation would repeat the speculation of the first few cycles;
//    with sharing, one branch's wall times would flip the other's mode.

import { describe, it, test } from "vite-plus/test";
import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import {
  createSpeculationGate,
  type SpeculationDecision,
  type SpeculationGate,
} from "../src/generation/speculation-gate.ts";

/** The wall times of one cycle (the observation passed to the gate — a pair of a speculative cycle and a plain step). */
type Walls = {
  /** Wall time of the speculative cycle, in ms. */
  readonly cycle: number;
  /** The number of tokens that cycle confirmed. */
  readonly delivered: number;
  /** Wall time of the plain step (M=1), in ms. */
  readonly plain: number;
};

/**
 * Runs the gate for one cycle (returns the observation matching the decision, and returns that decision).
 *
 * A direct copy of the caller's contract (one decide → one observation).
 */
const step = (gate: SpeculationGate, walls: Walls): SpeculationDecision => {
  const decision = gate.decide();
  if (decision === "speculate") gate.observeCycle(walls.cycle, walls.delivered);
  else gate.observePlain(walls.plain);
  return decision;
};

/** The record of one step (the decision, and the steady mode **before and after** that step). */
type Observed = {
  readonly decision: SpeculationDecision;
  /** The mode when the decision was made (whether this step is on the exploration side is read from this). */
  readonly before: SpeculationDecision;
  /** The mode after the observation was returned (a switch happens on an observation). */
  readonly mode: SpeculationDecision;
};

const trace = (gate: SpeculationGate, walls: Walls, count: number): Observed[] =>
  Array.from({ length: count }, () => {
    const before = gate.mode;
    const decision = step(gate, walls);
    return { decision, before, mode: gate.mode };
  });

const run = (gate: SpeculationGate, walls: Walls, count: number): SpeculationDecision[] =>
  trace(gate, walls, count).map((observed) => observed.decision);

/** The indices in a decision sequence where `plain` appears. */
const plainAt = (decisions: readonly SpeculationDecision[]): number[] =>
  decisions.flatMap((decision, index) => (decision === "plain" ? [index] : []));

/** The number of speculate decisions in a sequence (= the cycles accumulated into blocks / bursts). */
const speculateCount = (decisions: readonly SpeculationDecision[]): number =>
  decisions.filter((decision) => decision === "speculate").length;

/**
 * The list of lengths of consecutive steps of the same kind (`kind` of `"plain"` gives the exploration intervals;
 * `"speculate"` gives the exploration burst lengths).
 *
 * Only **steps whose mode at decision time is plain** are considered. This keeps out the single `W1` re-measurement
 * inserted in speculate mode, and keeps the step at the moment of the flip (decision speculate, plain after the
 * observation) from being counted as a burst.
 */
const plainModeRuns = (steps: readonly Observed[], kind: SpeculationDecision): number[] => {
  const lengths: number[] = [];
  let current = 0;
  for (const observed of steps) {
    if (observed.before === "plain" && observed.decision === kind) current += 1;
    else if (current > 0) {
      lengths.push(current);
      current = 0;
    }
  }
  return lengths;
};

/** Wall times where speculation clearly wins (ratio (100/4)/50 = 0.5). */
const WINNING: Walls = { cycle: 100, delivered: 4, plain: 50 };
/** Wall times where speculation clearly loses (ratio (200/2)/50 = 2.0 — the plain wall time is 50, the same as WINNING). */
const LOSING: Walls = { cycle: 200, delivered: 2, plain: 50 };
/**
 * Wall times that lose slightly (ratio (110/2)/50 = 1.10) — above `leave` (1.01) but below the "strong loss" threshold
 * `leave + 0.15` (1.16 — the default of `burstAbort`; the setups that pass `earlyLeave` use the same value).
 */
const MILD: Walls = { cycle: 110, delivered: 2, plain: 50 };
/** Wall times that lose strongly (ratio (130/2)/50 = 1.30 — above the 1.16 above). */
const STRONG_LOSS: Walls = { cycle: 130, delivered: 2, plain: 50 };

test("gate T1 start: it starts in speculate, and takes W1 exactly once on the second decision", () => {
  // Until `W1` (the decode wall time per token) is measured, "should it leave" cannot be decided. Running 16 cycles
  // without measuring it would lose 2% up front on M2 free text (0.76×).
  const gate = createSpeculationGate();
  assertEquals(gate.mode, "speculate", "initial mode");
  assertEquals(gate.switches, 0);
  assertEquals(run(gate, WINNING, 6), [
    "speculate",
    "plain",
    "speculate",
    "speculate",
    "speculate",
    "speculate",
  ]);
  // A single plain step is a re-measurement of `W1`, not a mode switch.
  assertEquals(gate.mode, "speculate");
  assertEquals(gate.switches, 0, "counts the `W1` re-measurement as a switch");
});

test("gate T2 blocks: no decision until a block fills, and it leaves only after 2 consecutive losing blocks", () => {
  const gate = createSpeculationGate();
  // The default window is 16. The `W1` re-measurements fall on decisions 1 / 15, so 17 decisions give exactly 15
  // speculative cycles = the block has not filled yet. Even with losing wall times it does not flip here.
  const first = run(gate, MILD, 17);
  assertEquals(speculateCount(first), 15, "the premise that the block has not filled is broken");
  assertEquals(gate.mode, "speculate", "leaves before the block fills");
  assertEquals(gate.switches, 0);

  // The 16th fills block 1 (ratio 1.10 > leave 1.01), but `confirm` is 2, so it does not leave yet (early leave is
  // off by default, so the size of the ratio does not change how long it waits — T2′).
  run(gate, MILD, 1);
  assertEquals(gate.mode, "speculate", "leaves after a single losing block");
  assertEquals(gate.switches, 0);

  // The next 17 (decisions 18..34, with one re-measurement at decision 31) give 16 speculative cycles = block 2
  // fills. It stays speculate until the last observation, and flips on that observation.
  const second = trace(gate, MILD, 17);
  assertEquals(speculateCount(second.map((observed) => observed.decision)), 16);
  assertEquals(
    second.slice(0, -1).map((observed) => observed.mode),
    Array.from({ length: 16 }, () => "speculate"),
    "flips before block 2 fills",
  );
  assertEquals(gate.mode, "plain");
  assertEquals(gate.switches, 1);
});

test("gate T2′ early leave: off by default (2 blocks even at a ratio of 1.30); only the earlyLeave setup flips after 1", () => {
  // The default of off rests on real runs, not on desk probabilities. Acceptance is not steady within a turn, and
  // even winning tasks (dialogue, ratio 0.90) mix in stretches of about 16 cycles where acceptance drops, so in a
  // conversation that reused its sequence, 3 of 7 turns left by mistake (research §6.5 table 11″). The gain is
  // "one block until leaving", once per sequence lifetime, and the harm hits every turn of a winning task, so it is
  // not taken by default.
  const patient = createSpeculationGate();
  const steps = trace(patient, STRONG_LOSS, 18);
  assertEquals(speculateCount(steps.map((observed) => observed.decision)), 16);
  assertEquals(patient.mode, "speculate", "flips after one block by default (early leave is off)");
  assertEquals(patient.switches, 0);
  // It first flips at the 2nd block (17 decisions give 16 speculative cycles — the same count as T2).
  run(patient, STRONG_LOSS, 17);
  assertEquals(patient.mode, "plain", "does not flip after 2 consecutive losing blocks");
  assertEquals(patient.switches, 1);

  // A setup that passes the knob flips after 1 block (= early leave takes effect through the `earlyLeave` threshold alone).
  const eager = createSpeculationGate({ earlyLeave: 0.15 });
  const eagerSteps = trace(eager, STRONG_LOSS, 18);
  assertEquals(speculateCount(eagerSteps.map((observed) => observed.decision)), 16);
  assertEquals(
    eagerSteps.slice(0, -1).map((observed) => observed.mode),
    Array.from({ length: 17 }, () => "speculate"),
    "flips before the block fills (early leave is also a decision on one filled block)",
  );
  assertEquals(eager.mode, "plain", "does not flip after 1 block in the earlyLeave 0.15 setup");
  assertEquals(eager.switches, 1);

  // Counterpart (in the same setup, a ratio of 1.10 does not reach `leave + earlyLeave` = 1.16, so it waits for `confirm`).
  const mild = createSpeculationGate({ earlyLeave: 0.15 });
  run(mild, MILD, 18);
  assertEquals(mild.mode, "speculate", "leaves early on a loss that does not reach 1.16");
  assertEquals(mild.switches, 0);

  // Counterpart 2 (miswiring): the speculate side's threshold is `earlyLeave`, not `burstAbort`.
  // A ratio of 1.30 exceeds `leave + burstAbort` = 1.16 but does not reach `leave + earlyLeave` = 1.31.
  const wired = createSpeculationGate({ earlyLeave: 0.3, burstAbort: 0.15 });
  run(wired, STRONG_LOSS, 18);
  assertEquals(wired.mode, "speculate", "the speculate side reads the burstAbort threshold");
  assertEquals(wired.switches, 0);
});

/** Only the first cycle delivers 1 (accepts 0) — the shape that seeded the old EWMA design with a single sample. */
const SEED_FIRST: Walls = { cycle: 100, delivered: 1, plain: 40 };
/** The steady state afterwards (ratio (100/3)/40 = 0.83 = speculation is winning). */
const SEED_REST: Walls = { cycle: 100, delivered: 3, plain: 40 };

test("gate T3 seeding: even if the first cycle accepts 0, it does not leave in a winning context", () => {
  // A design that keeps `A` as a moving average gets `ewma(undefined, 1) = 1`, and at the next decision the ratio is
  // `(100/1)/40 = 2.5 > leave` → **it drops to plain on the 3rd move** (exactly the defect observed in a real run).
  // Block aggregation looks at the sum of 16, so `1600 / (1 + 15×3) = 34.8 ms/token` < `40 × 1.01`, and the
  // decision correctly comes out on the "winning" side.
  const gate = createSpeculationGate();
  step(gate, SEED_FIRST);
  run(gate, SEED_REST, 59);
  assertEquals(gate.mode, "speculate", "leaves on single-sample seeding");
  assertEquals(gate.switches, 0);

  // Counterpart (fault injection): with the same seeding, it leaves if the steady state really loses (a ratio of
  // (100/2)/40 = 1.25 lasts for `confirm` = 2 blocks). Without this, the green above would pass even for "a gate that never leaves".
  const losing = createSpeculationGate();
  step(losing, SEED_FIRST);
  run(losing, { cycle: 100, delivered: 2, plain: 40 }, 59);
  assertEquals(losing.mode, "plain", "does not leave even with really losing wall times");
  assertEquals(losing.switches, 1);
});

test("gate T4 exploration: the plain side measures in bursts, and the interval doubles on every miss", () => {
  // A single-cycle exploration is buried in the accepted count's sd of 0.88 and cannot decide (why the old design
  // could not return from plain). It measures in bursts and backs off on a miss (with a fixed interval, the loss of
  // exploration itself would be 0.950× on M2 free text, paying back in exploration the loss the gate saved).
  const gate = createSpeculationGate({ window: 2, confirm: 1 });
  const steps = trace(gate, LOSING, 2000);
  assertEquals(
    plainModeRuns(steps, "plain").slice(0, 7),
    [16, 32, 64, 128, 256, 512, 512],
    `the exploration interval is not a geometric backoff: ${plainModeRuns(steps, "plain").join(",")}`,
  );
  // A ratio of 2.0 exceeds `leave + burstAbort` = 1.16, so a burst whose miss is settled is folded at `burstMin` = 4
  // (T4″).
  assertEquals(
    plainModeRuns(steps, "speculate").slice(0, 6),
    [4, 4, 4, 4, 4, 4],
    `the exploration burst is not folded at 4 cycles: ${plainModeRuns(steps, "speculate").join(",")}`,
  );
  assertEquals(gate.mode, "plain", "returns despite an exploration that keeps missing");
  assertEquals(gate.switches, 1, "counts exploration bursts as switches");
});

test("gate T4′ return: it returns when a burst's aggregate falls below enter, and the interval goes back to 16", () => {
  const gate = createSpeculationGate({ window: 2, confirm: 1 });
  run(gate, LOSING, 3);
  assertEquals(
    gate.mode,
    "plain",
    "the premise of dropping to plain on losing wall times is broken",
  );
  assertEquals(gate.switches, 1);

  // The task changes and speculation starts winning → after the interval of 16 plain steps, it measures with a burst
  // of 8 cycles (a hitting burst runs until it fills = the abort only acts on the losing side).
  const steps = trace(gate, WINNING, 24);
  assertEquals(
    steps.map((observed) => observed.decision),
    [...Array.from({ length: 16 }, () => "plain"), ...Array.from({ length: 8 }, () => "speculate")],
  );
  assertEquals(
    steps.slice(16, 23).map((observed) => observed.mode),
    Array.from({ length: 7 }, () => "plain"),
    "returns before the burst fills (it must not return on a single hitting cycle)",
  );
  assertEquals(gate.mode, "speculate", "does not return on a hitting burst");
  assertEquals(gate.switches, 2);

  // Making it lose again drops it to plain, and exploration restarts **from 16** (the grown interval is not carried over).
  const again = trace(gate, LOSING, 40);
  assertEquals(gate.mode, "plain", "does not drop to plain after returning to a loss");
  assertEquals(gate.switches, 3);
  assertEquals(
    plainModeRuns(again, "plain")[0],
    16,
    `the exploration interval after returning is not 16: ${plainModeRuns(again, "plain").join(",")}`,
  );
});

test("gate T4″ abort: a burst with a strong loss (burstAbort) is folded at the burstMin-th cycle", () => {
  // This is the main exploration cost a losing context pays (bursts of 8 cycles × intervals of 16 → 32 → …). The
  // hit decision is still made on a filled burst, so unlike shrinking `burst` itself, the return accuracy does not drop.
  const gate = createSpeculationGate({ window: 2, confirm: 1 });
  run(gate, LOSING, 3);
  assertEquals(
    gate.mode,
    "plain",
    "the premise of dropping to plain on losing wall times is broken",
  );
  assertEquals(gate.switches, 1);

  // After the interval of 16 plain steps, a burst → at the 4th observation the ratio is 1.30 > 1.16, so it is folded,
  // and from then on it stays plain until the doubled interval (32 steps).
  const steps = trace(gate, STRONG_LOSS, 53);
  assertEquals(
    plainModeRuns(steps, "plain"),
    [16, 32],
    `the interval after the aborted burst did not double: ${plainModeRuns(steps, "plain").join(",")}`,
  );
  assertEquals(
    plainModeRuns(steps, "speculate"),
    [4],
    `the burst is not folded at the 4th: ${plainModeRuns(steps, "speculate").join(",")}`,
  );
  assertEquals(gate.mode, "plain");
  assertEquals(gate.switches, 1, "counts the aborted burst as a switch");

  // Counterpart 1 (the abort position is decided by `burstMin` — with the same wall times it does not fold before the 5th).
  const later = createSpeculationGate({ window: 2, confirm: 1, burstMin: 5 });
  run(later, LOSING, 3);
  assertEquals(
    plainModeRuns(trace(later, STRONG_LOSS, 22), "speculate"),
    [5],
    "the burst is folded at 4 even with a raised burstMin",
  );

  // Counterpart 2 (a burst with a ratio of 1.10, above `leave` but not a strong loss, runs all 8 cycles).
  const full = createSpeculationGate({ window: 2, confirm: 1 });
  run(full, LOSING, 3);
  const mild = trace(full, MILD, 25);
  assertEquals(
    plainModeRuns(mild, "speculate"),
    [8],
    `aborts a weakly losing burst: ${plainModeRuns(mild, "speculate").join(",")}`,
  );
  assertEquals(full.mode, "plain", "returns to speculate on a burst with a ratio of 1.10");

  // Counterpart 3 (the threshold is owned by `burstAbort` — raising it lets a burst with the same ratio of 1.30 run to
  // the end. It is also the counterpart showing it is a different knob from the speculate side's `earlyLeave`).
  const lenient = createSpeculationGate({ window: 2, confirm: 1, burstAbort: 0.5 });
  run(lenient, LOSING, 3);
  const kept = trace(lenient, STRONG_LOSS, 25);
  assertEquals(
    plainModeRuns(kept, "speculate"),
    [8],
    `aborts the burst even with a raised burstAbort: ${plainModeRuns(kept, "speculate").join(",")}`,
  );
  assertEquals(lenient.mode, "plain", "returns to speculate on a burst with a ratio of 1.30");
});

test("gate T5 steady state: measures plain once per 16 observations, and that step is not counted in a block", () => {
  // A stale `W1` leaves the basis of "should it leave" stale. The interval is 16, the same as the decision block, and
  // the cost is one token at plain price per 16 cycles (about 1% relative to always) — this is paid.
  const gate = createSpeculationGate();
  const decisions = run(gate, WINNING, 40);
  assertEquals(plainAt(decisions), [1, 15, 31]);
  assertEquals(gate.mode, "speculate");
  assertEquals(gate.switches, 0);

  // Blocks count **only speculative cycles**. With a window of 4 and losing wall times, one `W1` plain step is
  // inserted, so the block only fills at the 4th speculative cycle = the 5th decision. An implementation that also
  // counted plain steps in blocks would flip one move early (at the 4th decision).
  const counted = createSpeculationGate({ window: 4, confirm: 1 });
  const steps = trace(counted, LOSING, 5);
  assertEquals(
    steps.map((observed) => observed.mode),
    ["speculate", "speculate", "speculate", "speculate", "plain"],
  );
  assertEquals(counted.switches, 1);
});

describe("gate T6 skip: only the period advances; neither W1 nor the blocks move", () => {
  it("the period advances (the first `W1` sample comes that much earlier)", () => {
    // A cycle the caller decided "not to mix into the wall-time observations" (the first cycle of each turn) — it reads
    // the decision but returns `skip()` instead of an observation. Even a cycle that is not mixed in runs one GPU job.
    const gate = createSpeculationGate();
    assertEquals(gate.decide(), "speculate");
    gate.skip();
    assertEquals(gate.decide(), "plain", "the period did not advance for the skipped cycle");
    gate.observePlain(WINNING.plain);
    assertEquals(gate.mode, "speculate");
    assertEquals(gate.switches, 0);
  });

  it("is not counted in a block", () => {
    const gate = createSpeculationGate({ window: 2, confirm: 1 });
    assertEquals(gate.decide(), "speculate");
    gate.skip();
    assertEquals(gate.decide(), "plain");
    gate.observePlain(LOSING.plain);
    assertEquals(step(gate, LOSING), "speculate", "block 1");
    assertEquals(gate.mode, "speculate", "skip is counted in a block");
    assertEquals(step(gate, LOSING), "speculate", "block 2 = it fills");
    assertEquals(gate.mode, "plain");
    assertEquals(gate.switches, 1);
  });
});

describe("gate T7 idempotence: decide does not move the state (a switch happens on an observation)", () => {
  // `decide()` is a pure read. Two directions are pinned here: repeated reads in the same state do not advance the
  // state, and the read right after the state moved (an observation / `skip`) is **always recomputed**. Without the
  // latter, an implementation that memoizes the decision and discards it only on an observation (= the decision of a
  // `skip`ped cycle comes out one move stale) would stay green.
  it("start (nothing measured yet)", () => {
    const gate = createSpeculationGate();
    assertEquals(
      [gate.decide(), gate.decide(), gate.decide()],
      ["speculate", "speculate", "speculate"],
    );
    assertEquals(gate.switches, 0);
  });

  it("however often it is read right after a flip, there is one switch and the exploration period does not shift", () => {
    const gate = createSpeculationGate({ window: 2, confirm: 1 });
    // It flips to plain on the **observation** of the 3rd move (the observation, not the decision, moves the state).
    run(gate, LOSING, 3);
    assertEquals(gate.mode, "plain");
    assertEquals(gate.switches, 1);
    assertEquals([gate.decide(), gate.decide(), gate.decide()], ["plain", "plain", "plain"]);
    assertEquals(gate.switches, 1, "merely reading the decision advanced the switches");

    // The number of reads does not affect the period, so the exploration burst starts exactly at the 17th move.
    assertEquals(
      run(gate, LOSING, 17),
      [...Array.from({ length: 16 }, () => "plain"), "speculate"],
      "repeated reads advance the exploration period",
    );
    assertEquals(gate.switches, 1);
    assertEquals(gate.mode, "plain");
  });

  it("returning an observation recomputes the next decision", () => {
    const gate = createSpeculationGate();
    // The 16th move = the steady `W1` re-measurement.
    run(gate, WINNING, 15);
    assertEquals(gate.decide(), "plain");
    assertEquals(gate.decide(), "plain", "the decision moved before the observation was returned");
    gate.observePlain(WINNING.plain);
    assertEquals(
      gate.decide(),
      "speculate",
      "the decision did not update after the observation was returned",
    );
  });

  it("the decision of a skipped cycle is not carried over either (the period advances by 1)", () => {
    const gate = createSpeculationGate();
    assertEquals(gate.decide(), "speculate");
    gate.skip();
    // The period advanced by 1, so next is the first `W1` sample (the second decision = plain). An implementation that
    // carries the decision over would emit a one-move-stale speculate here.
    assertEquals(gate.decide(), "plain", "the decision of the skipped cycle is carried over");
    gate.observePlain(WINNING.plain);
    // `skip` does not move the blocks = there is no speculative observation yet, so no decision comes out.
    assertEquals(gate.mode, "speculate");
    assertEquals(gate.switches, 0);
  });
});

/** One measured condition (`docs/research/2026-09-09-mtp-stage4.md` §2 table 1 = RTX / §5.1 = M2, greedy). */
type Condition = Walls & {
  readonly name: string;
  /** The ratio `(Wc/A)/W1` derived from the table (below 1 means speculation is faster). */
  readonly ratio: number;
  /** The mode it should be in after 2 blocks (the default setup = early leave off). */
  readonly mode: SpeculationDecision;
  /**
   * The number of blocks needed to flip to plain in the setup that passes `earlyLeave: 0.15` (only conditions whose
   * `mode` is plain have it).
   *
   * 1 = a condition whose ratio exceeds `leave + earlyLeave` = 1.16, so it leaves early; 2 = a condition above `leave`
   * but below 1.16, so it waits for `confirm` consecutive blocks. **In the default setup every condition is 2**
   * (early leave is off).
   */
  readonly eagerBlocks?: 1 | 2;
};

const CONDITIONS: readonly Condition[] = [
  {
    name: "RTX extraction",
    cycle: 53.7,
    delivered: 3.39,
    plain: 29.22,
    ratio: 0.542,
    mode: "speculate",
  },
  {
    name: "RTX summarization",
    cycle: 55.8,
    delivered: 2.65,
    plain: 29.67,
    ratio: 0.71,
    mode: "speculate",
  },
  {
    name: "RTX dialogue",
    cycle: 46.8,
    delivered: 2.03,
    plain: 27.13,
    ratio: 0.85,
    mode: "speculate",
  },
  {
    name: "RTX free text",
    cycle: 46.5,
    delivered: 1.66,
    plain: 27.02,
    ratio: 1.037,
    mode: "plain",
    eagerBlocks: 2,
  },
  {
    name: "M2 extraction",
    cycle: 172.4,
    delivered: 3.39,
    plain: 66.03,
    ratio: 0.77,
    mode: "speculate",
  },
  {
    name: "M2 summarization",
    cycle: 170.7,
    delivered: 2.65,
    plain: 66.39,
    ratio: 0.97,
    mode: "speculate",
  },
  {
    name: "M2 dialogue",
    cycle: 123.0,
    delivered: 2.03,
    plain: 56.14,
    ratio: 1.079,
    mode: "plain",
    eagerBlocks: 2,
  },
  {
    name: "M2 free text",
    cycle: 118.2,
    delivered: 1.66,
    plain: 54.45,
    ratio: 1.308,
    mode: "plain",
    eagerBlocks: 1,
  },
];

describe("gate T8 measurements: of the 8 conditions, only the 3 losing ones drop to plain", () => {
  for (const condition of CONDITIONS) {
    it(`${condition.name} (ratio ${condition.ratio})`, () => {
      // The constants are the same as the research table's (it turns red the moment the table is miscopied).
      assertAlmostEquals(
        condition.cycle / condition.delivered / condition.plain,
        condition.ratio,
        0.001,
        `${condition.name}: the ratio from the wall-time constants differs from the table's value`,
      );
      const gate = createSpeculationGate();
      // 18 decisions = 16 speculative cycles = exactly 1 block (the only plain steps are the `W1` re-measurements, the
      // 2 at decisions 1 / 15 ≈ 1/16). Early leave is off by default, so **no condition flips here**.
      const decisions = run(gate, condition, 18);
      assertEquals(plainAt(decisions), [1, 15], `${condition.name}: number of re-measurements`);
      assertEquals(
        gate.mode,
        "speculate",
        `${condition.name}: mode after 1 block (early leave is off by default)`,
      );

      // The next 17 decisions (one re-measurement at decision 31) fill the 2nd block.
      run(gate, condition, 17);
      assertEquals(gate.mode, condition.mode, `${condition.name}: mode after 2 blocks`);
      assertEquals(
        gate.switches,
        condition.mode === "plain" ? 1 : 0,
        `${condition.name}: number of switches`,
      );

      // In the setup that passes early leave, only the conditions above a ratio of 1.16 flip after 1 block (the 5
      // winning conditions flip in neither setup — early leave only moves "the cost until leaving").
      const eager = createSpeculationGate({ earlyLeave: 0.15 });
      run(eager, condition, 18);
      assertEquals(
        eager.mode,
        condition.eagerBlocks === 1 ? "plain" : "speculate",
        `${condition.name}: mode after 1 block in the earlyLeave 0.15 setup`,
      );
      run(eager, condition, 17);
      assertEquals(
        eager.mode,
        condition.mode,
        `${condition.name}: mode after 2 blocks in the earlyLeave 0.15 setup`,
      );
    });
  }

  it("the dead band (M2 summarization's 0.970) crosses neither enter nor leave", () => {
    // 0.970 is just above the return-side threshold of 0.97 and does not satisfy `< enter` = it stays as is. A
    // difference of 3–4% is buried in the noise of the accepted count (the sd per cycle is 43% of the mean), so
    // not moving is correct.
    const gate = createSpeculationGate();
    const summarize = CONDITIONS[5];
    run(gate, summarize, 120);
    assertEquals(gate.mode, "speculate");
    assertEquals(gate.switches, 0);

    // At the same ratio it does not return when seen from the plain side either (the dead band acts from both sides).
    // The `W1` EWMA approaches from 54.45 to 66.39 **from below**, so the return threshold `W1 × 0.97` does not exceed
    // 64.40 (the burst measures 170.7 / 2.65 = 64.42 ms/token).
    const stuck = createSpeculationGate();
    run(stuck, CONDITIONS[7], 60);
    assertEquals(stuck.mode, "plain", "the premise of dropping on M2 free text is broken");
    run(stuck, summarize, 600);
    assertEquals(stuck.mode, "plain", "returns to speculate at a ratio inside the dead band");
    assertEquals(stuck.switches, 1);
  });
});

describe("gate T9 checks: knob ranges fail loudly at creation", () => {
  it("alpha is 0 < α ≤ 1", () => {
    assertThrows(
      () => createSpeculationGate({ alpha: 0 }),
      Error,
      "alpha 0 is not a positive finite number",
    );
    assertThrows(
      () => createSpeculationGate({ alpha: Number.NaN }),
      Error,
      "alpha NaN is not a positive finite number",
    );
    assertThrows(
      () => createSpeculationGate({ alpha: 1.5 }),
      Error,
      "alpha 1.5 is outside 0 < α ≤ 1",
    );
  });

  it("a block is 2 cycles or more (1 would return to single-sample decisions)", () => {
    assertThrows(
      () => createSpeculationGate({ window: 1 }),
      Error,
      "window 1 is not an integer ≥ 2",
    );
    assertThrows(
      () => createSpeculationGate({ window: 16.5 }),
      Error,
      "window 16.5 is not an integer ≥ 2",
    );
  });

  it("the consecutive count and the burst are integers of 1 or more", () => {
    assertThrows(
      () => createSpeculationGate({ confirm: 0 }),
      Error,
      "confirm 0 is not an integer ≥ 1",
    );
    assertThrows(() => createSpeculationGate({ burst: 0 }), Error, "burst 0 is not an integer ≥ 1");
  });

  it("the abort minimum is an integer from 1 to burst", () => {
    assertThrows(
      () => createSpeculationGate({ burstMin: 0 }),
      Error,
      "burstMin 0 is not an integer ≥ 1",
    );
    assertThrows(
      () => createSpeculationGate({ burstMin: 4.5 }),
      Error,
      "burstMin 4.5 is not an integer ≥ 1",
    );
    // `burstMin > burst` is a dead knob that never fires (the default `burst` is 8).
    assertThrows(() => createSpeculationGate({ burstMin: 9 }), Error, "burstMin 9 exceeds burst 8");
    // The default `burstMin` is capped by `burst` (a setup with a burst of 1 cycle does not fail the check).
    assertEquals(createSpeculationGate({ burst: 1 }).mode, "speculate");
  });

  it("both strong-loss margins are positive finite numbers (early leave unset = off)", () => {
    for (const [name, value] of [
      ["earlyLeave", 0],
      ["earlyLeave", -1],
      ["earlyLeave", Number.POSITIVE_INFINITY],
      ["burstAbort", 0],
      ["burstAbort", -1],
      ["burstAbort", Number.POSITIVE_INFINITY],
    ] as const) {
      assertThrows(
        () => createSpeculationGate({ [name]: value }),
        Error,
        `${name} ${value} is not a positive finite number`,
      );
    }
    // An unset `earlyLeave` means "off" and is not subject to the check (not to be confused with passing `0` — 0 would
    // mean "early leave at threshold = `leave`", so it is rejected).
    assertEquals(createSpeculationGate({}).mode, "speculate");
    assertEquals(createSpeculationGate({ earlyLeave: undefined }).mode, "speculate");
  });

  it("the exploration interval is an integer of 1 or more, and its cap is at least the base interval", () => {
    assertThrows(
      () => createSpeculationGate({ exploreBase: 0 }),
      Error,
      "exploreBase 0 is not an integer ≥ 1",
    );
    assertThrows(
      () => createSpeculationGate({ exploreBase: 1.5 }),
      Error,
      "exploreBase 1.5 is not an integer ≥ 1",
    );
    assertThrows(
      () => createSpeculationGate({ exploreBase: 8, exploreMax: 4 }),
      Error,
      "exploreMax 4 is not an integer ≥ 8",
    );
  });

  it("the dead band straddles 1 (enter < 1 < leave)", () => {
    assertThrows(
      () => createSpeculationGate({ enter: 1 }),
      Error,
      "enter 1 < 1 < leave 1.01 does not hold",
    );
    assertThrows(
      () => createSpeculationGate({ leave: 0.99 }),
      Error,
      "enter 0.97 < 1 < leave 0.99 does not hold",
    );
    assertThrows(
      () => createSpeculationGate({ enter: -0.5 }),
      Error,
      "enter -0.5 is not a positive finite number",
    );
  });

  it("the edge values pass (the counterpart showing the check is not too wide)", () => {
    const gate = createSpeculationGate({
      alpha: 1,
      window: 2,
      confirm: 1,
      burst: 1,
      burstMin: 1,
      exploreBase: 1,
      exploreMax: 1,
      enter: 0.5,
      leave: 2,
      earlyLeave: 0.5,
      burstAbort: 0.5,
    });
    assert(gate.mode === "speculate");
  });
});

test("gate T9 checks: a rejected knob is caught as the cross-family ModelInputError", () => {
  // The knob bundle is passed by the caller as `speculative.gate`, and both the ranges and the ordering are fixed by
  // "fixing the value passed" = caused by the input per ADR 0107 decision 2. This pins by type that it feeds the host's 400 / 500 branch.
  const rejected = [
    { alpha: 0 },
    { alpha: 1.5 },
    { window: 1 },
    { confirm: 0 },
    { burst: 0 },
    { burstMin: 9 },
    { exploreBase: 8, exploreMax: 4 },
    { earlyLeave: 0 },
    { burstAbort: -1 },
    { enter: -0.5 },
  ];
  for (const options of rejected) {
    const error = assertThrows(() => createSpeculationGate(options));
    assert(
      error instanceof ModelInputError,
      `the rejection of ${JSON.stringify(options)} is not thrown with the input-caused type`,
    );
  }
});

describe("gate T10 branching: fork returns an independent gate that copies the observation state", () => {
  it("given the same observations, the original and the branch have the same decisions and mode transitions", () => {
    const gate = createSpeculationGate();
    // After 17, `W1` is measured and block 1 is just about to fill (the same position as T2).
    run(gate, MILD, 17);
    const forked = gate.fork();
    assertEquals([forked.mode, forked.switches], [gate.mode, gate.switches]);
    const fromGate = trace(gate, MILD, 18);
    const fromFork = trace(forked, MILD, 18);
    assertEquals(
      fromFork,
      fromGate,
      "the branch's decisions and mode transitions differ from the original's",
    );
    // It is the observation state that was copied: flipping within the next 18 happens only if the first 17 observations were inherited.
    assertEquals([gate.mode, forked.mode], ["plain", "plain"]);
    assertEquals([gate.switches, forked.switches], [1, 1]);
    // Positive control: a gate without observations does not flip within the same 18 (if it did, the agreement above would check nothing).
    const fresh = createSpeculationGate();
    run(fresh, MILD, 18);
    assertEquals(
      fresh.mode,
      "speculate",
      "a gate without observations flipped within 18 (the check is vacuous)",
    );
  });

  it("giving observations to only one side does not move the other (both directions)", () => {
    // The baseline is a twin with the same history (the behaviour had it not branched).
    const knobs = { window: 2, confirm: 1 };
    const gate = createSpeculationGate(knobs);
    const twin = createSpeculationGate(knobs);
    run(gate, WINNING, 5);
    run(twin, WINNING, 5);
    const forked = gate.fork();
    const forkedTwin = createSpeculationGate(knobs);
    run(forkedTwin, WINNING, 5);

    // Only the branch is made to lose → the branch flips, and the original behaves the same as the twin.
    run(forked, LOSING, 40);
    assertEquals(
      forked.mode,
      "plain",
      "positive control: the branch does not flip (the observations have no effect)",
    );
    assertEquals([gate.mode, gate.switches], [twin.mode, twin.switches]);
    assertEquals(
      trace(gate, WINNING, 20),
      trace(twin, WINNING, 20),
      "the observation on the branch moved the original",
    );

    // The other direction: making only the original lose does not move the branch (its continuation).
    const second = forkedTwin.fork();
    const secondTwin = createSpeculationGate(knobs);
    run(secondTwin, WINNING, 5);
    run(forkedTwin, LOSING, 40);
    assertEquals(forkedTwin.mode, "plain");
    assertEquals(
      trace(second, WINNING, 20),
      trace(secondTwin, WINNING, 20),
      "the observation on the original moved the branch",
    );
  });
});

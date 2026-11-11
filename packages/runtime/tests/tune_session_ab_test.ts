// The wall-clock A/B of whole runs (`runSessionAb`, `sessionAbStatistics` and `judgeSessionAb` of
// `@karume/runtime/tune`) on synthetic arms and a fake clock. No GPU. The arms are scripted: each
// run advances the fake clock by the next duration of the arm's script and returns the outputs the
// script gives for that visit, and every open, run and close is logged.

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  judgeSessionAb,
  runSessionAb,
  type SessionAbArm,
  type SessionAbEnvironment,
  type SessionAbRecord,
  type SessionAbRun,
  sessionAbStatistics,
} from "../tune.ts";

const ENVIRONMENT: SessionAbEnvironment = {
  adapter: { vendor: "test", architecture: "synthetic", device: "", description: "" },
  userAgent: "test-agent/1",
  backend: "none",
  date: "2026-01-01T00:00:00.000Z",
};

/** What an open or a close costs on the fake clock: enough to show up in any time it leaks into. */
const OPEN_CLOSE_MS = 1000;

type FakeClock = { now: number };

type Script = {
  /** The duration of each run in visit order: the two warm-up visits first, then two per round. */
  readonly durations: readonly number[];
  /** The outputs of the n-th visit (0-based, warm-up included). Default: {@link baseOutputs}. */
  readonly outputs?: (visit: number) => SessionAbRun["outputs"];
  /** The visit (0-based) whose run throws. */
  readonly throwAt?: number;
  /** Whether `open` throws. */
  readonly openThrows?: boolean;
};

const baseOutputs = (): SessionAbRun["outputs"] => ({
  y: Float32Array.of(1, 2, 3, 4),
  ids: Int32Array.of(7, 8),
});

/** An arm that follows `script` on `clock` and logs `open <name>`, `run <name>`, `close <name>`. */
const scriptedArm = (
  name: string,
  script: Script,
  clock: FakeClock,
  log: string[],
): SessionAbArm => {
  let visit = 0;
  return {
    name,
    open: () => {
      log.push(`open ${name}`);
      clock.now += OPEN_CLOSE_MS;
      if (script.openThrows === true) return Promise.reject(new Error(`open ${name} failed`));
      return Promise.resolve();
    },
    run: () => {
      log.push(`run ${name}`);
      const current = visit;
      visit += 1;
      if (current === script.throwAt) return Promise.reject(new Error(`run ${name} failed`));
      clock.now += script.durations[current] ?? 1;
      return Promise.resolve({
        outputs: (script.outputs ?? baseOutputs)(current),
        fusions: { "test/rule": name === "base" ? 0 : 2 },
        dispatches: name === "base" ? 6 : 4,
      });
    },
    close: () => {
      log.push(`close ${name}`);
      clock.now += OPEN_CLOSE_MS;
      return Promise.resolve();
    },
  };
};

/** Durations in visit order from per-round pairs, after a warm-up of two 50 ms runs. */
const durations = (rounds: readonly (readonly [number, number])[]): number[] => [
  50,
  50,
  ...rounds.flat(),
];

type Arms = Readonly<Record<string, Script>>;

const run = async (
  arms: Arms,
  rounds: number,
  openPerVisit = false,
): Promise<{ readonly record: SessionAbRecord; readonly log: string[] }> => {
  const clock: FakeClock = { now: 0 };
  const log: string[] = [];
  const record = await runSessionAb({
    arms: Object.entries(arms).map(([name, script]) => scriptedArm(name, script, clock, log)),
    rounds,
    environment: ENVIRONMENT,
    openPerVisit,
    clock: () => clock.now,
  });
  return { record, log };
};

// The base of the verdict cases: round means 101, 101, 102 (median 101); |visit 1 − visit 2| is
// 2, 0, 2 (median 2).
const BASE_TIMES = [
  [100, 102],
  [101, 101],
  [103, 101],
] as const;

describe("runSessionAb", () => {
  describe("visit order", () => {
    it("runs an untimed warm-up round, then every round forward and reverse (A B B A), between one open and one close per arm", async () => {
      const { log } = await run(
        { base: { durations: durations(BASE_TIMES) }, fused: { durations: durations(BASE_TIMES) } },
        3,
      );
      const round = ["run base", "run fused", "run fused", "run base"];
      assertEquals(log, [
        "open base",
        "open fused",
        ...round, // warm-up
        ...round,
        ...round,
        ...round,
        "close fused",
        "close base",
      ]);
    });

    it("visits three arms as A B C C B A", async () => {
      const script = { durations: durations(BASE_TIMES) };
      const { log } = await run({ a: script, b: script, c: script }, 3);
      const round = ["run a", "run b", "run c", "run c", "run b", "run a"];
      assertEquals(log, [
        "open a",
        "open b",
        "open c",
        ...round,
        ...round,
        ...round,
        ...round,
        "close c",
        "close b",
        "close a",
      ]);
    });

    it("opens and closes around every visit with openPerVisit", async () => {
      const { log } = await run(
        { base: { durations: durations(BASE_TIMES) }, fused: { durations: durations(BASE_TIMES) } },
        3,
        true,
      );
      const visit = (name: string): string[] => [`open ${name}`, `run ${name}`, `close ${name}`];
      const round = [...visit("base"), ...visit("fused"), ...visit("fused"), ...visit("base")];
      assertEquals(log, [...round, ...round, ...round, ...round]);
    });
  });

  describe("timed spans", () => {
    const fusedTimes = [
      [95, 97],
      [96, 96],
      [98, 96],
    ] as const;
    for (const openPerVisit of [false, true]) {
      it(`records each run's own duration and nothing of open or close (openPerVisit ${openPerVisit})`, async () => {
        const { record } = await run(
          {
            base: { durations: durations(BASE_TIMES) },
            fused: { durations: durations(fusedTimes) },
          },
          3,
          openPerVisit,
        );
        assertEquals(record.openPerVisit, openPerVisit);
        assertEquals(
          record.arms.map((arm) => arm.timesMs),
          [BASE_TIMES, fusedTimes],
        );
      });
    }
  });

  describe("the record", () => {
    it("is plain JSON data: the environment as given, rounds, and per arm its reported counts, raw times and identity", async () => {
      const { record } = await run(
        { base: { durations: durations(BASE_TIMES) }, fused: { durations: durations(BASE_TIMES) } },
        3,
      );
      assertEquals(JSON.parse(JSON.stringify(record)), record);
      assertEquals(record.format, "karume-session-ab/1");
      assertEquals(record.environment, ENVIRONMENT);
      assertEquals(record.rounds, 3);
      assertEquals(
        record.arms.map(({ name, fusions, dispatches }) => ({ name, fusions, dispatches })),
        [
          { name: "base", fusions: { "test/rule": 0 }, dispatches: 6 },
          { name: "fused", fusions: { "test/rule": 2 }, dispatches: 4 },
        ],
      );
      // Two outputs of 4 + 2 words; 8 visits per arm (the warm-up round's included).
      for (const arm of record.arms) {
        assertEquals(arm.identity, {
          identical: true,
          visits: 8,
          worst: { round: 0, visit: 1, words: 6, failures: 0 },
        });
      }
    });
  });

  describe("closing", () => {
    it("closes every opened arm when a run throws, and the run's error propagates", async () => {
      const clock: FakeClock = { now: 0 };
      const log: string[] = [];
      const arms = [
        scriptedArm("base", { durations: durations(BASE_TIMES) }, clock, log),
        // Visit 4 = round 2, visit 1.
        scriptedArm("fused", { durations: durations(BASE_TIMES), throwAt: 4 }, clock, log),
      ];
      await assertRejects(
        () => runSessionAb({ arms, rounds: 3, environment: ENVIRONMENT, clock: () => clock.now }),
        Error,
        "run fused failed",
      );
      assertEquals(log.slice(-3), ["run fused", "close fused", "close base"]);
    });

    it("closes the arm whose run throws with openPerVisit", async () => {
      const clock: FakeClock = { now: 0 };
      const log: string[] = [];
      const arms = [
        scriptedArm("base", { durations: durations(BASE_TIMES), throwAt: 1 }, clock, log),
        scriptedArm("fused", { durations: durations(BASE_TIMES) }, clock, log),
      ];
      await assertRejects(
        () =>
          runSessionAb({
            arms,
            rounds: 3,
            environment: ENVIRONMENT,
            openPerVisit: true,
            clock: () => clock.now,
          }),
        Error,
        "run base failed",
      );
      assertEquals(log.slice(-3), ["open base", "run base", "close base"]);
    });

    it("closes the arms already opened when an open throws", async () => {
      const clock: FakeClock = { now: 0 };
      const log: string[] = [];
      const arms = [
        scriptedArm("base", { durations: durations(BASE_TIMES) }, clock, log),
        scriptedArm("fused", { durations: durations(BASE_TIMES), openThrows: true }, clock, log),
      ];
      await assertRejects(
        () => runSessionAb({ arms, rounds: 3, environment: ENVIRONMENT, clock: () => clock.now }),
        Error,
        "open fused failed",
      );
      assertEquals(log, ["open base", "open fused", "close base"]);
    });
  });

  describe("identity", () => {
    const QUIET_NAN = 0x7fc00000;
    const PAYLOAD_NAN = 0x7fc01234;

    const floatsOf = (...words: number[]): Float32Array<ArrayBuffer> =>
      new Float32Array(Uint32Array.from(words).buffer);

    /** The A/B of a base and one arm whose outputs at visit `at` (0-based) are `outputs`. */
    const identityOf = async (
      outputs: SessionAbRun["outputs"],
      at: number,
      base: SessionAbRun["outputs"] = baseOutputs(),
    ): Promise<SessionAbRecord> =>
      (
        await run(
          {
            base: { durations: durations(BASE_TIMES), outputs: () => base },
            fused: {
              durations: durations(BASE_TIMES),
              outputs: (visit) => (visit === at ? outputs : base),
            },
          },
          3,
        )
      ).record;

    it("finds a differing word in any visit and keeps the worst visit with its first failing word", async () => {
      const differing = { ...baseOutputs(), y: Float32Array.of(1, 2, 3.5, 4) };
      // Visit 5 = round 2, visit 2.
      const record = await identityOf(differing, 5);
      assertEquals(record.arms[1].identity, {
        identical: false,
        visits: 8,
        worst: {
          round: 2,
          visit: 2,
          words: 6,
          failures: 1,
          first: { output: "y", cause: "word", index: 2, expected: 0x40400000, actual: 0x40600000 },
        },
      });
      assertEquals(record.arms[0].identity.identical, true);
    });

    it("passes a NaN against a NaN of another payload in an f32 output", async () => {
      const base = { y: floatsOf(QUIET_NAN, 0x3f800000) };
      const record = await identityOf({ y: floatsOf(PAYLOAD_NAN, 0x3f800000) }, 3, base);
      assertEquals(record.arms[1].identity.identical, true);
    });

    it("compares an integer output word for word (no NaN class for integers)", async () => {
      const base = { ids: Int32Array.of(QUIET_NAN) };
      const record = await identityOf({ ids: Int32Array.of(PAYLOAD_NAN) }, 3, base);
      assertEquals(record.arms[1].identity.worst.failures, 1);
    });

    it("fails a missing output with all its words", async () => {
      const record = await identityOf({ y: baseOutputs().y }, 2);
      assertEquals(record.arms[1].identity.worst, {
        round: 1,
        visit: 1,
        words: 6,
        failures: 2,
        first: { output: "ids", cause: "missing", index: 0, expected: 7, actual: 0 },
      });
    });

    it("fails an output of another length", async () => {
      const record = await identityOf({ ...baseOutputs(), y: Float32Array.of(1, 2, 3) }, 2);
      assertEquals(record.arms[1].identity.worst.first, {
        output: "y",
        cause: "length",
        index: 3,
        expected: 0x40800000,
        actual: 0,
      });
      assertEquals(record.arms[1].identity.worst.failures, 1);
    });

    it("fails an output the base does not return", async () => {
      const record = await identityOf({ ...baseOutputs(), z: Float32Array.of(0) }, 2);
      assertEquals(record.arms[1].identity.worst.first, {
        output: "z",
        cause: "extra",
        index: 0,
        expected: 0,
        actual: 0,
      });
    });

    it("rejects an arm that differs whatever its times, and judges it on time when identity is not required", async () => {
      // The arm's times are those of the adopt case below.
      const record = (
        await run(
          {
            base: { durations: durations(BASE_TIMES) },
            fused: {
              durations: durations([
                [95, 97],
                [96, 96],
                [98, 96],
              ]),
              outputs: (visit) =>
                visit === 3 ? { ...baseOutputs(), ids: Int32Array.of(7, 9) } : baseOutputs(),
            },
          },
          3,
        )
      ).record;
      const [judged] = judgeSessionAb(record, { minGain: 0.01 });
      assertEquals(judged.verdict, "reject");
      assertEquals(judged.identical, false);
      assertStringIncludes(judged.reason, 'arm "fused" differs from the base');
      assertStringIncludes(judged.reason, "output ids[1]");
      const [unchecked] = judgeSessionAb(record, { minGain: 0.01, requireIdentity: false });
      assertEquals(unchecked.verdict, "adopt");
    });

    it("rejects every arm when the base's own runs differ", async () => {
      const record = (
        await run(
          {
            base: {
              durations: durations(BASE_TIMES),
              outputs: (visit) =>
                visit === 6 ? { ...baseOutputs(), ids: Int32Array.of(0, 8) } : baseOutputs(),
            },
            fused: { durations: durations(BASE_TIMES) },
          },
          3,
        )
      ).record;
      assertEquals(record.arms[0].identity.identical, false);
      const [judged] = judgeSessionAb(record, { minGain: 0 });
      assertEquals(judged.verdict, "reject");
      assertStringIncludes(judged.reason, 'the base arm "base" differs from its own first outputs');
    });
  });

  describe("refusals", () => {
    const arm = (name: string): SessionAbArm =>
      scriptedArm(name, { durations: [] }, { now: 0 }, []);
    const refuse = (
      arms: readonly SessionAbArm[],
      rounds: number,
      message: string,
    ): Promise<void> =>
      assertRejects(
        () => runSessionAb({ arms, rounds, environment: ENVIRONMENT }),
        RangeError,
        message,
      ).then(() => undefined);

    it("refuses fewer than two arms", async () => {
      await refuse([arm("base")], 3, "at least two arms");
    });

    it("refuses a name used twice", async () => {
      await refuse([arm("base"), arm("base")], 3, '"base" is used twice');
    });

    it("refuses rounds that are not an integer of at least 3", async () => {
      for (const rounds of [2, 3.5, Number.NaN]) {
        await refuse(
          [arm("base"), arm("fused")],
          rounds,
          "rounds must be an integer of at least 3",
        );
      }
    });
  });
});

describe("judgeSessionAb and sessionAbStatistics", () => {
  const recordOf = async (
    fusedTimes: readonly (readonly [number, number])[],
    baseTimes: readonly (readonly [number, number])[] = BASE_TIMES,
  ): Promise<SessionAbRecord> =>
    (
      await run(
        { base: { durations: durations(baseTimes) }, fused: { durations: durations(fusedTimes) } },
        baseTimes.length,
      )
    ).record;

  it("adopts an arm faster than the noise floor by at least minGain", async () => {
    // fused round means 96, 96, 97 → deltas −5, −5, −5 (median −5); its spreads 2, 0, 2 (median 2);
    // floor max(2, 2) = 2; |−5| > 2; gain 5 / 101 ≈ 4.95% ≥ 1%.
    const record = await recordOf([
      [95, 97],
      [96, 96],
      [98, 96],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 0.01 });
    assertEquals(judged, {
      arm: "fused",
      base: "base",
      deltasMs: [-5, -5, -5],
      medianDeltaMs: -5,
      baseMedianMs: 101,
      gain: 5 / 101,
      noiseFloorMs: 2,
      verdict: "adopt",
      identical: true,
      reason: judged.reason,
    });
    assertStringIncludes(judged.reason, "median delta -5.000 ms, noise floor 2.000 ms");
  });

  it("calls a difference inside the noise floor inconclusive", async () => {
    // fused round means 102, 102, 103 → deltas 1, 1, 1 (median 1); its spreads 2, 0, 2 (median 2);
    // floor 2; |1| ≤ 2.
    const record = await recordOf([
      [101, 103],
      [102, 102],
      [104, 102],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 0.01 });
    assertEquals(
      { verdict: judged.verdict, medianDeltaMs: judged.medianDeltaMs, floor: judged.noiseFloorMs },
      { verdict: "inconclusive", medianDeltaMs: 1, floor: 2 },
    );
    assertStringIncludes(judged.reason, "inside the noise floor");
  });

  it("takes the larger of the two arms' spreads as the noise floor", async () => {
    // fused round means 98, 98, 99 → deltas −3 (median −3); its spreads 8, 6, 8 (median 8);
    // floor max(2, 8) = 8; |−3| ≤ 8.
    const record = await recordOf([
      [94, 102],
      [95, 101],
      [95, 103],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 0.01 });
    assertEquals(
      [judged.verdict, judged.medianDeltaMs, judged.noiseFloorMs],
      ["inconclusive", -3, 8],
    );
  });

  it("rejects an arm slower than the noise floor", async () => {
    // fused round means 111, 111, 112 → deltas 10 (median 10); floor 2; gain −10 / 101.
    const record = await recordOf([
      [110, 112],
      [111, 111],
      [113, 111],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 0 });
    assertEquals([judged.verdict, judged.medianDeltaMs, judged.gain], ["reject", 10, -10 / 101]);
    assertStringIncludes(judged.reason, "slower than the base");
  });

  it("rejects an arm faster than the noise floor by less than minGain", async () => {
    // The adopt case's numbers (gain 5 / 101 ≈ 4.95%) against a minimum of 10%.
    const record = await recordOf([
      [95, 97],
      [96, 96],
      [98, 96],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 0.1 });
    assertEquals(judged.verdict, "reject");
    assertStringIncludes(judged.reason, "less than the minimum gain 10.00%");
  });

  it("takes the mean of the two middle values as the median of an even count", async () => {
    // Four rounds. base round means 100, 100, 100, 100 (median 100), spreads 0.
    // fused round means 99, 97, 95, 93 → deltas −1, −3, −5, −7 (median (−3 − 5) / 2 = −4);
    // spreads 2, 2, 2, 6 (median 2).
    const record = await recordOf(
      [
        [98, 100],
        [96, 98],
        [94, 96],
        [90, 96],
      ],
      [
        [100, 100],
        [100, 100],
        [100, 100],
        [100, 100],
      ],
    );
    const [stats] = sessionAbStatistics(record);
    assertEquals(stats, {
      arm: "fused",
      base: "base",
      deltasMs: [-1, -3, -5, -7],
      medianDeltaMs: -4,
      baseMedianMs: 100,
      gain: 0.04,
      noiseFloorMs: 2,
    });
  });

  it("judges every non-base arm against the base, in order", async () => {
    const { record } = await run(
      {
        base: { durations: durations(BASE_TIMES) },
        fast: {
          durations: durations([
            [95, 97],
            [96, 96],
            [98, 96],
          ]),
        },
        same: { durations: durations(BASE_TIMES) },
      },
      3,
    );
    assertEquals(
      judgeSessionAb(record, { minGain: 0.01 }).map(({ arm, verdict }) => [arm, verdict]),
      [
        ["fast", "adopt"],
        ["same", "inconclusive"],
      ],
    );
  });

  it("gives the same verdicts on a record parsed back from JSON", async () => {
    const record = await recordOf([
      [95, 97],
      [96, 96],
      [98, 96],
    ]);
    const parsed: SessionAbRecord = JSON.parse(JSON.stringify(record));
    for (const minGain of [0, 0.01, 0.1]) {
      assertEquals(judgeSessionAb(parsed, { minGain }), judgeSessionAb(record, { minGain }));
    }
  });

  describe("refusals", () => {
    it("refuses a minGain that is not a finite non-negative number", async () => {
      const record = await recordOf(BASE_TIMES);
      for (const minGain of [-0.01, Number.NaN, Number.POSITIVE_INFINITY]) {
        assertThrows(
          () => judgeSessionAb(record, { minGain }),
          RangeError,
          "minGain must be a finite non-negative number",
        );
      }
    });

    it("refuses a record of another format", async () => {
      const record = await recordOf(BASE_TIMES);
      const other: SessionAbRecord = JSON.parse(
        JSON.stringify({ ...record, format: "karume-geometry-sweep/2" }),
      );
      assertThrows(() => judgeSessionAb(other, { minGain: 0 }), RangeError, "format");
      assertThrows(() => sessionAbStatistics(other), RangeError, "format");
    });

    it("refuses a record whose times do not cover every round", async () => {
      const record = await recordOf(BASE_TIMES);
      const short: SessionAbRecord = {
        ...record,
        arms: [record.arms[0], { ...record.arms[1], timesMs: record.arms[1].timesMs.slice(1) }],
      };
      assertThrows(() => sessionAbStatistics(short), RangeError, 'arm "fused"');
    });
  });
});

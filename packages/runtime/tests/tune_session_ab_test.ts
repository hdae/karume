// The wall-clock A/B of whole runs (`runSessionAb`, `sessionAbStatistics` and `judgeSessionAb` of
// `@karume/runtime/tune`) on synthetic arms and a fake clock. No GPU. The arms are scripted: each
// run advances the fake clock by the next duration of the arm's script and returns the outputs the
// script gives for that run, and every open, run, report and close is logged. Opening, closing and
// reporting advance the fake clock too, so any of them inside a timed span changes a time.

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  judgeSessionAb,
  runSessionAb,
  type SessionAbArm,
  type SessionAbEnvironment,
  type SessionAbRecord,
  type SessionAbReport,
  type SessionAbRun,
  sessionAbStatistics,
} from "../tune.ts";

const ENVIRONMENT: SessionAbEnvironment = {
  adapter: { vendor: "test", architecture: "synthetic", device: "", description: "" },
  userAgent: "test-agent/1",
  backend: "none",
  date: "2026-01-01T00:00:00.000Z",
};

/**
 * What an open, a close or a report costs on the fake clock: enough to show up in any time it leaks
 * into.
 */
const OPEN_CLOSE_MS = 1000;

/** The duration of the untimed first run of a visit with openPerVisit (see {@link withUntimed}). */
const UNTIMED_MS = 7;

type FakeClock = { now: number };

type Script = {
  /**
   * The duration of each run in run order: the two warm-up visits first, then two per round (with
   * openPerVisit, two runs per visit: {@link withUntimed}).
   */
  readonly durations: readonly number[];
  /** The outputs of the n-th run (0-based, warm-up included). Default: {@link baseOutputs}. */
  readonly outputs?: (run: number) => SessionAbRun["outputs"];
  /** The counts reported after the n-th run (0-based). Default: {@link defaultReport}. */
  readonly report?: (run: number) => SessionAbReport;
  /** The run (0-based) that throws. */
  readonly throwAt?: number;
  /** Whether `open` throws. */
  readonly openThrows?: boolean;
  /** Whether `close` throws. */
  readonly closeThrows?: boolean;
};

const baseOutputs = (): SessionAbRun["outputs"] => ({
  y: Float32Array.of(1, 2, 3, 4),
  ids: Int32Array.of(7, 8),
});

const defaultReport = (name: string): SessionAbReport => ({
  fusions: { "test/rule": name === "base" ? 0 : 2 },
  dispatches: name === "base" ? 6 : 4,
});

/**
 * An arm that follows `script` on `clock` and logs `open <name>`, `run <name>`, `report <name>` and
 * `close <name>`.
 */
const scriptedArm = (
  name: string,
  script: Script,
  clock: FakeClock,
  log: string[],
): SessionAbArm => {
  let runs = 0;
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
      const current = runs;
      runs += 1;
      if (current === script.throwAt) return Promise.reject(new Error(`run ${name} failed`));
      clock.now += script.durations[current] ?? 1;
      return Promise.resolve({ outputs: (script.outputs ?? baseOutputs)(current) });
    },
    report: () => {
      log.push(`report ${name}`);
      clock.now += OPEN_CLOSE_MS;
      return Promise.resolve(script.report?.(runs - 1) ?? defaultReport(name));
    },
    close: () => {
      log.push(`close ${name}`);
      clock.now += OPEN_CLOSE_MS;
      if (script.closeThrows === true) return Promise.reject(new Error(`close ${name} failed`));
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

/** Durations in run order with openPerVisit: an untimed run before every visit's duration. */
const withUntimed = (visits: readonly number[]): number[] =>
  visits.flatMap((duration) => [UNTIMED_MS, duration]);

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

/** One run of an arm as the log shows it: the run, then its report. */
const runOf = (name: string): string[] => [`run ${name}`, `report ${name}`];

describe("runSessionAb", () => {
  describe("visit order", () => {
    it("runs an untimed warm-up round, then every round forward and reverse (A B B A), between one open and one close per arm", async () => {
      const { log } = await run(
        { base: { durations: durations(BASE_TIMES) }, fused: { durations: durations(BASE_TIMES) } },
        3,
      );
      const round = [...runOf("base"), ...runOf("fused"), ...runOf("fused"), ...runOf("base")];
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
      const round = [
        ...runOf("a"),
        ...runOf("b"),
        ...runOf("c"),
        ...runOf("c"),
        ...runOf("b"),
        ...runOf("a"),
      ];
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

    it("opens and closes around every visit with openPerVisit, with an untimed run before the timed one", async () => {
      const { log } = await run(
        {
          base: { durations: withUntimed(durations(BASE_TIMES)) },
          fused: { durations: withUntimed(durations(BASE_TIMES)) },
        },
        3,
        true,
      );
      const visit = (name: string): string[] => [
        `open ${name}`,
        ...runOf(name),
        ...runOf(name),
        `close ${name}`,
      ];
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
      it(`records each timed run's own duration and nothing of open, close, report or an untimed run (openPerVisit ${openPerVisit})`, async () => {
        const order = openPerVisit ? withUntimed : (visits: number[]) => visits;
        const { record } = await run(
          {
            base: { durations: order(durations(BASE_TIMES)) },
            fused: { durations: order(durations(fusedTimes)) },
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

    it("keeps reading the outputs (the copy and the identity comparison) out of the timed span", async () => {
      const clock: FakeClock = { now: 0 };
      // Every read of `y` costs time on the fake clock, as work on the outputs would.
      const costlyOutputs = (): SessionAbRun["outputs"] => ({
        get y() {
          clock.now += OPEN_CLOSE_MS;
          return Float32Array.of(1, 2);
        },
      });
      const record = await runSessionAb({
        arms: [
          scriptedArm(
            "base",
            { durations: durations(BASE_TIMES), outputs: costlyOutputs },
            clock,
            [],
          ),
          scriptedArm(
            "fused",
            { durations: durations(fusedTimes), outputs: costlyOutputs },
            clock,
            [],
          ),
        ],
        rounds: 3,
        environment: ENVIRONMENT,
        clock: () => clock.now,
      });
      assertEquals(
        record.arms.map((arm) => arm.timesMs),
        [BASE_TIMES, fusedTimes],
      );
      assertEquals(record.arms[1].identity.worst.failures, 0);
    });
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
      // Two outputs of 4 + 2 words; 8 runs per arm (the warm-up round's included).
      for (const arm of record.arms) {
        assertEquals(arm.identity, {
          runs: 8,
          worst: { round: 0, visit: 1, run: 1, words: 6, failures: 0 },
        });
      }
    });

    it("copies the adapter's four fields by name, also from accessors on a prototype (a browser's GPUAdapterInfo)", async () => {
      class AdapterInfo {
        get vendor(): string {
          return "vendor-a";
        }
        get architecture(): string {
          return "arch-b";
        }
        get device(): string {
          return "device-c";
        }
        get description(): string {
          return "description-d";
        }
      }
      const clock: FakeClock = { now: 0 };
      const script = { durations: durations(BASE_TIMES) };
      const record = await runSessionAb({
        arms: [scriptedArm("base", script, clock, []), scriptedArm("fused", script, clock, [])],
        rounds: 3,
        environment: { ...ENVIRONMENT, adapter: new AdapterInfo() },
        clock: () => clock.now,
      });
      assertEquals(JSON.parse(JSON.stringify(record.environment.adapter)), {
        vendor: "vendor-a",
        architecture: "arch-b",
        device: "device-c",
        description: "description-d",
      });
    });
  });

  describe("counts", () => {
    it("refuses an arm whose run reports other counts than its first run, and closes every arm", async () => {
      const log: string[] = [];
      const clock: FakeClock = { now: 0 };
      const arms = [
        scriptedArm("base", { durations: durations(BASE_TIMES) }, clock, log),
        scriptedArm(
          "fused",
          {
            durations: durations(BASE_TIMES),
            // Run 4 = round 2, visit 1.
            report: (run) =>
              run === 4 ? { ...defaultReport("fused"), dispatches: 5 } : defaultReport("fused"),
          },
          clock,
          log,
        ),
      ];
      await assertRejects(
        () => runSessionAb({ arms, rounds: 3, environment: ENVIRONMENT, clock: () => clock.now }),
        RangeError,
        'arm "fused" reported {"fusions":{"test/rule":2},"dispatches":5} in round 2, visit 1, run 1',
      );
      assertEquals(log.slice(-2), ["close fused", "close base"]);
    });

    it("refuses a fusion count that changes between runs", async () => {
      const report = (run: number): SessionAbReport => ({
        fusions: run === 0 ? { "test/rule": 2 } : { "test/rule": 2, "test/other": 1 },
        dispatches: 4,
      });
      await assertRejects(
        () =>
          run(
            {
              base: { durations: durations(BASE_TIMES) },
              fused: { durations: durations(BASE_TIMES), report },
            },
            3,
          ),
        RangeError,
        'arm "fused" reported',
      );
    });

    it("compares the untimed run's counts too with openPerVisit", async () => {
      // Run 2 = the warm-up's visit 2, its untimed run.
      const report = (run: number): SessionAbReport =>
        run === 2 ? { ...defaultReport("fused"), dispatches: 9 } : defaultReport("fused");
      await assertRejects(
        () =>
          run(
            {
              base: { durations: withUntimed(durations(BASE_TIMES)) },
              fused: { durations: withUntimed(durations(BASE_TIMES)), report },
            },
            3,
            true,
          ),
        RangeError,
        "in round 0, visit 2, run 1",
      );
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
      // Run 1 = the timed run of the base's first visit.
      const arms = [
        scriptedArm(
          "base",
          { durations: withUntimed(durations(BASE_TIMES)), throwAt: 1 },
          clock,
          log,
        ),
        scriptedArm("fused", { durations: withUntimed(durations(BASE_TIMES)) }, clock, log),
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
      assertEquals(log, ["open base", "run base", "report base", "run base", "close base"]);
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

    it("attempts every close when closes throw after a complete A/B, and the first close error propagates", async () => {
      const clock: FakeClock = { now: 0 };
      const log: string[] = [];
      const arms = [
        scriptedArm("base", { durations: durations(BASE_TIMES), closeThrows: true }, clock, log),
        scriptedArm("fused", { durations: durations(BASE_TIMES), closeThrows: true }, clock, log),
      ];
      await assertRejects(
        () => runSessionAb({ arms, rounds: 3, environment: ENVIRONMENT, clock: () => clock.now }),
        Error,
        "close fused failed",
      );
      assertEquals(log.slice(-2), ["close fused", "close base"]);
    });

    it("lets the run's error win over close errors, and still attempts every close", async () => {
      const clock: FakeClock = { now: 0 };
      const log: string[] = [];
      const arms = [
        scriptedArm("base", { durations: durations(BASE_TIMES), closeThrows: true }, clock, log),
        scriptedArm(
          "fused",
          { durations: durations(BASE_TIMES), throwAt: 4, closeThrows: true },
          clock,
          log,
        ),
      ];
      await assertRejects(
        () => runSessionAb({ arms, rounds: 3, environment: ENVIRONMENT, clock: () => clock.now }),
        Error,
        "run fused failed",
      );
      assertEquals(log.slice(-3), ["run fused", "close fused", "close base"]);
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
        runs: 8,
        worst: {
          round: 2,
          visit: 2,
          run: 1,
          words: 6,
          failures: 1,
          first: { output: "y", cause: "word", index: 2, expected: 0x40400000, actual: 0x40600000 },
        },
      });
      assertEquals(record.arms[0].identity.worst.failures, 0);
    });

    it("passes a NaN against a NaN of another payload in an f32 output", async () => {
      const base = { y: floatsOf(QUIET_NAN, 0x3f800000) };
      const record = await identityOf({ y: floatsOf(PAYLOAD_NAN, 0x3f800000) }, 3, base);
      assertEquals(record.arms[1].identity.worst.failures, 0);
    });

    it("fails an output of another array type whole, even with the same bits", async () => {
      const base = { y: Float32Array.of(1, 2) };
      const sameBits = new Int32Array(Float32Array.of(1, 2).buffer);
      const record = await identityOf({ y: sameBits }, 3, base);
      assertEquals(record.arms[1].identity.worst, {
        round: 1,
        visit: 2,
        run: 1,
        words: 2,
        failures: 2,
        first: { output: "y", cause: "type", index: 0, expected: 0x3f800000, actual: 0x3f800000 },
      });
    });

    it("compares with a copy of the base's first outputs, so a base that rewrites one array shows as not deterministic", async () => {
      const shared = Float32Array.of(0, 5);
      const record = (
        await run(
          {
            base: {
              durations: durations(BASE_TIMES),
              outputs: (index) => {
                shared[0] = index;
                return { y: shared };
              },
            },
            fused: {
              durations: durations(BASE_TIMES),
              outputs: () => ({ y: Float32Array.of(0, 5) }),
            },
          },
          3,
        )
      ).record;
      assertEquals(record.arms[0].identity.worst, {
        round: 0,
        visit: 2,
        run: 1,
        words: 2,
        failures: 1,
        first: { output: "y", cause: "word", index: 0, expected: 0, actual: 0x3f800000 },
      });
      assertEquals(record.arms[1].identity.worst.failures, 0);
    });

    it("compares the untimed run of a visit too with openPerVisit", async () => {
      const differing = { ...baseOutputs(), ids: Int32Array.of(7, 9) };
      // Run 2 = the warm-up's visit 2, its untimed run.
      const record = (
        await run(
          {
            base: { durations: withUntimed(durations(BASE_TIMES)) },
            fused: {
              durations: withUntimed(durations(BASE_TIMES)),
              outputs: (index) => (index === 2 ? differing : baseOutputs()),
            },
          },
          3,
          true,
        )
      ).record;
      assertEquals(record.arms[1].identity.runs, 16);
      assertEquals(
        [record.arms[1].identity.worst.round, record.arms[1].identity.worst.visit],
        [0, 2],
      );
      assertEquals(record.arms[1].identity.worst.run, 1);
    });

    it("refuses a base whose first run returns no output, or outputs with no word", async () => {
      const empties: SessionAbRun["outputs"][] = [
        {},
        { y: new Float32Array(0), ids: new Int32Array(0) },
      ];
      for (const empty of empties) {
        await assertRejects(
          () =>
            run(
              {
                base: { durations: durations(BASE_TIMES), outputs: () => empty },
                fused: { durations: durations(BASE_TIMES), outputs: () => empty },
              },
              3,
            ),
          RangeError,
          'the base arm "base" returned no output word on its first run',
        );
      }
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
        run: 1,
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
      assertEquals(record.arms[0].identity.worst.failures, 1);
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

    it("refuses a timed span the clock reads as not finite or negative, in the judge's words", async () => {
      for (const bad of [-1, Number.NaN]) {
        const fused = durations([
          [bad, 100],
          [100, 100],
          [100, 100],
        ]);
        await assertRejects(
          () => run({ base: { durations: durations(BASE_TIMES) }, fused: { durations: fused } }, 3),
          RangeError,
          'arm "fused" does not hold two finite non-negative times for each of the 3 rounds' +
            ` (round 1, visit 1: ${bad} ms)`,
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
    // floor max(2, 2) = 2; |−5| > 2; three rounds: the interval is [min, max] = [−5, −5] at
    // 1 − 2 / 8 = 75% coverage, and it does not hold 0; gain 5 / 101 ≈ 4.95% ≥ 1%.
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
      deltaLowMs: -5,
      deltaHighMs: -5,
      intervalCoverage: 0.75,
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
    // spreads 2, 2, 2, 6 (median 2); four rounds: the interval is [−7, −1] at 1 − 2 / 16 = 87.5%.
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
      deltaLowMs: -7,
      deltaHighMs: -1,
      intervalCoverage: 0.875,
    });
  });

  it("takes the base's visit spread when it is the larger one", async () => {
    // base round means 103, 104, 103 (median 103), spreads 6, 6, 6; fused round means 100, 101, 100
    // → deltas −3, −3, −3, its spreads 0; floor max(6, 0) = 6; |−3| ≤ 6.
    const record = await recordOf(
      [
        [100, 100],
        [101, 101],
        [100, 100],
      ],
      [
        [100, 106],
        [101, 107],
        [100, 106],
      ],
    );
    const [judged] = judgeSessionAb(record, { minGain: 0.01 });
    assertEquals(
      [judged.verdict, judged.medianDeltaMs, judged.noiseFloorMs],
      ["inconclusive", -3, 6],
    );
  });

  /** `rounds` rounds of a base at 100 ms in every visit (noise floor 0 when the arm's visits agree). */
  const flatBase = (rounds: number): readonly (readonly [number, number])[] =>
    Array.from({ length: rounds }, () => [100, 100] as const);
  /** An arm whose two visits of a round agree, at `100 + delta` ms per round. */
  const agreeing = (deltas: readonly number[]): readonly (readonly [number, number])[] =>
    deltas.map((delta) => [100 + delta, 100 + delta] as const);

  it("calls it inconclusive when the rounds do not agree on which arm is faster, whatever the noise floor", async () => {
    // Five rounds, every arm's two visits agree (floor 0). deltas −3, −2, −1, 2, 3: median −1
    // (gain 1%), but the interval [min, max] = [−3, 3] (1 − 2 / 32 = 93.75%) holds 0.
    const record = await recordOf(agreeing([-3, -2, -1, 2, 3]), flatBase(5));
    const [judged] = judgeSessionAb(record, { minGain: 0.005 });
    assertEquals(
      {
        verdict: judged.verdict,
        median: judged.medianDeltaMs,
        floor: judged.noiseFloorMs,
        interval: [judged.deltaLowMs, judged.deltaHighMs, judged.intervalCoverage],
      },
      { verdict: "inconclusive", median: -1, floor: 0, interval: [-3, 3, 0.9375] },
    );
    assertStringIncludes(judged.reason, "the rounds do not agree on which arm is faster");
  });

  it("is not fooled by rounds that tie: three rounds at −1 and two at +4 are inconclusive", async () => {
    // deltas −1, −1, −1, 4, 4: median −1, and the scatter around the median is 0 for most rounds
    // (the reading a spread around the median would give), yet two rounds of five say the arm is
    // slower: the interval [−1, 4] holds 0.
    const record = await recordOf(agreeing([-1, -1, -1, 4, 4]), flatBase(5));
    const [judged] = judgeSessionAb(record, { minGain: 0.005 });
    assertEquals(
      [judged.verdict, judged.medianDeltaMs, judged.deltaLowMs, judged.deltaHighMs],
      ["inconclusive", -1, -1, 4],
    );
  });

  it("drops one outlier round of eight from the interval, and not two", async () => {
    // Eight rounds: P(B ≤ 1) = 9 / 256, so the second smallest and second largest keep
    // 1 − 18 / 256 = 92.97% coverage. Seven rounds at −2 and one at +50: interval [−2, −2], adopt
    // (gain 2%). With two rounds at +50 the interval is [−2, 50] and holds 0.
    const one = await recordOf(agreeing([-2, -2, 50, -2, -2, -2, -2, -2]), flatBase(8));
    const [adopted] = judgeSessionAb(one, { minGain: 0.01 });
    assertEquals(
      [adopted.verdict, adopted.deltaLowMs, adopted.deltaHighMs, adopted.intervalCoverage],
      ["adopt", -2, -2, 1 - 18 / 256],
    );
    const two = await recordOf(agreeing([-2, -2, 50, -2, -2, 50, -2, -2]), flatBase(8));
    const [undecided] = judgeSessionAb(two, { minGain: 0.01 });
    assertEquals(
      [undecided.verdict, undecided.medianDeltaMs, undecided.deltaLowMs, undecided.deltaHighMs],
      ["inconclusive", -2, -2, 50],
    );
  });

  it("gives the same verdict as rounds are added to an arm that is faster in every round", async () => {
    // The verdict must not depend on the parity of the round count: −2 in every round is adopt at
    // 3, 4, 5, 6, 7 and 8 rounds alike.
    for (const rounds of [3, 4, 5, 6, 7, 8]) {
      const record = await recordOf(
        agreeing(Array.from({ length: rounds }, () => -2)),
        flatBase(rounds),
      );
      assertEquals(
        judgeSessionAb(record, { minGain: 0.01 })[0].verdict,
        "adopt",
        `${rounds} rounds`,
      );
    }
  });

  it("refuses, by name, a stored record whose arms are not arms, and more rounds than the statistics take", async () => {
    const record = await recordOf(agreeing([-2, -2, -2]), flatBase(3));
    for (const arms of [undefined, "base", [{ name: "base" }, { name: "fused" }], [5, 6]]) {
      assertThrows(
        () => sessionAbStatistics({ ...record, arms } as unknown as SessionAbRecord),
        RangeError,
        "the record's arms must be an array of arms",
        JSON.stringify(arms),
      );
    }
    const times = Array.from({ length: 1001 }, () => [100, 100] as const);
    assertThrows(
      () =>
        sessionAbStatistics({
          ...record,
          rounds: 1001,
          arms: record.arms.map((arm) => ({ ...arm, timesMs: times })),
        }),
      RangeError,
      "at most 1000 rounds",
    );
  });

  it("calls a median difference equal to the noise floor inconclusive", async () => {
    // fused round means 99, 99, 100 → deltas −2, −2, −2; floor max(2, 2) = 2; |−2| = 2.
    const record = await recordOf([
      [98, 100],
      [99, 99],
      [101, 99],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 0.01 });
    assertEquals(
      [judged.verdict, judged.medianDeltaMs, judged.noiseFloorMs],
      ["inconclusive", -2, 2],
    );
  });

  it("adopts a gain equal to minGain", async () => {
    // The adopt case's numbers: gain exactly 5 / 101.
    const record = await recordOf([
      [95, 97],
      [96, 96],
      [98, 96],
    ]);
    const [judged] = judgeSessionAb(record, { minGain: 5 / 101 });
    assertEquals([judged.verdict, judged.gain], ["adopt", 5 / 101]);
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

    it("refuses times that are not finite and non-negative in a stored record", async () => {
      const record = await recordOf(BASE_TIMES);
      for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
        const edited: SessionAbRecord = {
          ...record,
          arms: [
            record.arms[0],
            { ...record.arms[1], timesMs: [[bad, 100], ...record.arms[1].timesMs.slice(1)] },
          ],
        };
        assertThrows(
          () => judgeSessionAb(edited, { minGain: 0 }),
          RangeError,
          'arm "fused" does not hold two finite non-negative times for each of the 3 rounds',
        );
      }
    });

    it("refuses a base whose median round time is not positive", async () => {
      const record = await recordOf(BASE_TIMES, [
        [0, 0],
        [0, 0],
        [0, 0],
      ]);
      assertThrows(
        () => sessionAbStatistics(record),
        RangeError,
        'the base arm "base" has a median round time of 0 ms',
      );
    });

    it("refuses an identity block that does not hold what the judge reads, by name", async () => {
      const record = await recordOf(BASE_TIMES);
      const { worst } = record.arms[1].identity;
      const malformed: unknown[] = [
        undefined,
        { runs: 8 },
        { runs: 0, worst },
        { runs: 8, worst: { ...worst, failures: -1 } },
        { runs: 8, worst: { ...worst, visit: 3 } },
        { runs: 8, worst: { ...worst, round: 4 } },
        // Failures without the first mismatch, and a first mismatch of an unknown cause.
        { runs: 8, worst: { ...worst, failures: 3 } },
        {
          runs: 8,
          worst: {
            ...worst,
            failures: 1,
            first: { output: "y", cause: "other", index: 0, expected: 0, actual: 1 },
          },
        },
      ];
      for (const identity of malformed) {
        const edited: SessionAbRecord = JSON.parse(
          JSON.stringify({ ...record, arms: [record.arms[0], { ...record.arms[1], identity }] }),
        );
        assertThrows(
          () => judgeSessionAb(edited, { minGain: 0 }),
          RangeError,
          'arm "fused" does not hold a valid identity block',
        );
      }
    });

    it("reads identity from the worst run of a stored record", async () => {
      const record = await recordOf([
        [95, 97],
        [96, 96],
        [98, 96],
      ]);
      const failing: SessionAbRecord = {
        ...record,
        arms: [
          record.arms[0],
          {
            ...record.arms[1],
            identity: {
              runs: 8,
              worst: {
                round: 3,
                visit: 1,
                run: 1,
                words: 6,
                failures: 2,
                first: { output: "ids", cause: "word", index: 1, expected: 8, actual: 9 },
              },
            },
          },
        ],
      };
      const [judged] = judgeSessionAb(failing, { minGain: 0.01 });
      assertEquals([judged.verdict, judged.identical], ["reject", false]);
      assertStringIncludes(judged.reason, "2 of 6 words fail in round 3, visit 1, run 1");
    });
  });
});

import { runSessionAb, type SessionAbArm } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { ParameterError } from "../src/entry.ts";
import { type ArmStep, presetOf, runPositions, steppedArm } from "../src/session-ab/shared.ts";

import { gemmaManifestText, manifestOf } from "./helpers/session-ab-manifests.ts";

/** One model with the presets i4 and i4-fast (the default). */
const MANIFEST = manifestOf(gemmaManifestText(["e2b"]));

const ENVIRONMENT = {
  adapter: { vendor: "test", architecture: "", device: "", description: "" },
  userAgent: "test",
  backend: "other",
  date: "2026-11-11T00:00:00.000Z",
};

/**
 * Runs the real runner with fake arms and lists every run as the runner made it: the arm, and
 * whether the runner read its clock right before it (the clock spans exactly the run it times; the
 * warm-up round's times are read and then not kept).
 */
const observedRuns = async (
  arms: number,
  rounds: number,
  openPerVisit: boolean,
): Promise<readonly { readonly arm: number; readonly clocked: boolean }[]> => {
  const runs: { arm: number; clocked: boolean }[] = [];
  let clockOpen = false;
  let tick = 0;
  const fakes: SessionAbArm[] = Array.from({ length: arms }, (_, arm) => ({
    name: `arm${arm}`,
    open: () => Promise.resolve(),
    run: () => {
      runs.push({ arm, clocked: clockOpen });
      return Promise.resolve({ outputs: { tokens: Int32Array.of(1, 2, 3) } });
    },
    close: () => Promise.resolve(),
  }));
  await runSessionAb({
    arms: fakes,
    rounds,
    openPerVisit,
    environment: ENVIRONMENT,
    clock: () => {
      clockOpen = !clockOpen;
      tick += 1;
      return tick;
    },
  });
  return runs;
};

describe("the runner's visiting order as an entry lists it", () => {
  for (const openPerVisit of [true, false]) {
    it(`matches runSessionAb run for run (openPerVisit: ${openPerVisit})`, async () => {
      for (const [arms, rounds] of [
        [2, 3],
        [2, 4],
        [3, 3],
      ] as const) {
        const where = `${arms} arms, ${rounds} rounds`;
        const clockedRun = openPerVisit ? 2 : 1;
        const observed = await observedRuns(arms, rounds, openPerVisit);
        const listed = runPositions(arms, rounds, openPerVisit);
        expect(
          listed.map(({ arm, run }) => ({ arm, clocked: run === clockedRun })),
          where,
        ).toEqual(observed);
        // Timed = clocked and kept: the warm-up round's clocked runs are not kept.
        for (const { round, run, timed } of listed) {
          expect(timed, where).toBe(round > 0 && run === clockedRun);
        }
      }
    });
  }

  it("has (rounds + 1) × 2 visits per arm, two runs each when opened per visit, A B B A in a round", () => {
    const positions = runPositions(2, 4, true);
    expect(positions).toHaveLength(5 * 2 * 2 * 2);
    expect(positions.filter(({ timed }) => timed)).toHaveLength(4 * 2 * 2);
    expect(
      positions.filter(({ round, run }) => round === 1 && run === 2).map(({ arm }) => arm),
    ).toEqual([0, 1, 1, 0]);
    expect(runPositions(2, 3, false).filter(({ timed }) => timed)).toHaveLength(3 * 2 * 2);
  });
});

describe("the arms' steps of a Session target (steppedArm)", () => {
  /** Runs the real runner over two fake arms wrapped by steppedArm, logging what happens when. */
  const observe = async (openPerVisit: boolean) => {
    const log: { readonly entry: string; readonly timed: boolean }[] = [];
    let clockOpen = false;
    let tick = 0;
    const fake = (name: string, dispatches: number): SessionAbArm => ({
      name,
      open: () => {
        log.push({ entry: `${name}:open`, timed: clockOpen });
        return Promise.resolve();
      },
      run: () => {
        log.push({ entry: `${name}:run`, timed: clockOpen });
        return Promise.resolve({ outputs: { y: Float32Array.of(1, 2) } });
      },
      report: () => Promise.resolve({ fusions: { silu: 1 }, dispatches }),
      close: () => {
        log.push({ entry: `${name}:close`, timed: clockOpen });
        return Promise.resolve();
      },
    });
    const onStep = (step: ArmStep): void => {
      log.push({ entry: `${step.arm}:step:${step.step}`, timed: clockOpen });
    };
    const record = await runSessionAb({
      arms: [steppedArm(fake("base", 10), onStep), steppedArm(fake("rule", 7), onStep)],
      rounds: 3,
      openPerVisit,
      environment: {
        adapter: { vendor: "test", architecture: "", device: "", description: "" },
        userAgent: "test",
        backend: "other",
        date: "2026-11-11T00:00:00.000Z",
      },
      clock: () => {
        clockOpen = !clockOpen;
        tick += 1;
        return tick;
      },
    });
    return { log, record };
  };

  for (const openPerVisit of [true, false]) {
    it(`are reported outside the timed spans, around each open, run and close (openPerVisit: ${openPerVisit})`, async () => {
      const { log } = await observe(openPerVisit);
      expect(log.filter(({ timed }) => timed).map(({ entry }) => entry.split(":")[1])).toEqual(
        log.filter(({ timed }) => timed).map(() => "run"),
      );
      expect(log.some(({ timed }) => timed)).toBe(true);
      const steps = log.map(({ entry }) => entry);
      // Every open is preceded by its step, every run followed by its `ran`, every close preceded.
      steps.forEach((entry, index) => {
        const [arm, what] = entry.split(":");
        if (what === "open") expect(steps[index - 1]).toBe(`${arm}:step:open`);
        if (what === "run") expect(steps[index + 1]).toBe(`${arm}:step:ran`);
        if (what === "close") expect(steps[index - 1]).toBe(`${arm}:step:close`);
      });
    });
  }

  it("pass the arm's counts through to the runner's record", async () => {
    const { record } = await observe(true);
    expect(
      record.arms.map(({ name, fusions, dispatches }) => ({ name, fusions, dispatches })),
    ).toEqual([
      { name: "base", fusions: { silu: 1 }, dispatches: 10 },
      { name: "rule", fusions: { silu: 1 }, dispatches: 7 },
    ]);
  });
});

describe("the preset a session A/B loads", () => {
  it("is the model's default preset when none is named, and the named one when the model has it", () => {
    expect(presetOf(MANIFEST, "e2b", "").name).toBe("i4-fast");
    expect(presetOf(MANIFEST, "e2b", "i4").name).toBe("i4");
  });

  it("refuses a name that is not a preset of the model, listing its presets", () => {
    expect(() => presetOf(MANIFEST, "e2b", "f16")).toThrow(ParameterError);
    expect(() => presetOf(MANIFEST, "e2b", "f16")).toThrow(
      "Parameter preset: 'f16' is not a preset of model 'e2b'; its presets are i4, i4-fast",
    );
  });
});

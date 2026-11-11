import { runSessionAb, type SessionAbArm } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { runPositions } from "../src/models/session-ab/run.ts";

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

describe("the runner's visiting order as the page lists it", () => {
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

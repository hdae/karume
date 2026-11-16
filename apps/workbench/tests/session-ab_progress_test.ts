import { runSessionAb, type SessionAbArm } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { steppedArm } from "../src/session-ab/moss.ts";
import { armStepLine, armStepper, fetchLine, probeStepLine } from "../src/session-ab/progress.ts";
import { runPositions } from "../src/session-ab/run.ts";

const ENVIRONMENT = {
  adapter: { vendor: "test", architecture: "", device: "", description: "" },
  userAgent: "test",
  backend: "other",
  date: "2026-11-11T00:00:00.000Z",
};

describe("the session A/B status lines", () => {
  const positions = runPositions(2, 3, true);

  it("say where the runner is and how many runs have ended", () => {
    expect(armStepLine({ arm: "base", step: "open" }, positions, 0, "Session")).toBe(
      "opening the Session · warm-up round · visit 1 · base (untimed) · 0 / 32 runs",
    );
    expect(armStepLine({ arm: "base", step: "ran" }, positions, 10, "Session")).toBe(
      "ran · round 1 · visit 1 · base (timed) · 10 / 32 runs",
    );
    expect(armStepLine({ arm: "rule", step: "close" }, positions, 12, "pipeline")).toBe(
      "closing the pipeline of rule · 12 / 32 runs",
    );
  });

  it("say which probe step runs, and the bytes of a fetch in whole MiB", () => {
    expect(probeStepLine({ arm: "rule", step: "run" }, 1, 2)).toBe(
      "probing the counts · one untimed generation of rule · 1 / 2 probes",
    );
    const fetched = {
      phase: "downloading",
      path: "sft/dit/model-00002-of-00002.krm",
      loaded: 3.4 * 2 ** 20,
      total: 2 ** 30,
      fileLoaded: 3.4 * 2 ** 20,
      fileTotal: 2 ** 30,
    } as const;
    expect(fetchLine("dit", fetched)).toBe("fetching the dit component · 3 / 1024 MiB");
  });
});

describe("the arm stepper", () => {
  /**
   * Runs the real runner over two fake arms wrapped by steppedArm with the stepper, and logs every
   * run, report and yield with whether the runner's clock was open at that moment.
   */
  const observe = async (openPerVisit: boolean) => {
    const log: { readonly entry: string; readonly timed: boolean }[] = [];
    let clockOpen = false;
    let tick = 0;
    const fake = (name: string): SessionAbArm => ({
      name,
      open: () => Promise.resolve(),
      run: () => {
        log.push({ entry: `${name}:run`, timed: clockOpen });
        return Promise.resolve({ outputs: { y: Float32Array.of(1, 2) } });
      },
      report: () => Promise.resolve({ fusions: {}, dispatches: 1 }),
      close: () => Promise.resolve(),
    });
    const positions = runPositions(2, 3, openPerVisit);
    const onStep = armStepper({
      positions,
      opens: "Session",
      signal: new AbortController().signal,
      progress: (text) => log.push({ entry: `progress:${text}`, timed: clockOpen }),
      yieldTurn: () => {
        log.push({ entry: "yield", timed: clockOpen });
        return Promise.resolve();
      },
    });
    await runSessionAb({
      arms: [steppedArm(fake("base"), onStep), steppedArm(fake("rule"), onStep)],
      rounds: 3,
      openPerVisit,
      environment: ENVIRONMENT,
      clock: () => {
        clockOpen = !clockOpen;
        tick += 1;
        return tick;
      },
    });
    return { log, positions };
  };

  for (const openPerVisit of [true, false]) {
    it(`reports outside the timed spans and yields once after every report (openPerVisit: ${openPerVisit})`, async () => {
      const { log, positions } = await observe(openPerVisit);
      const timed = log.filter(({ timed: inSpan }) => inSpan).map(({ entry }) => entry);
      expect(timed.length).toBeGreaterThan(0);
      expect(timed.every((entry) => entry.endsWith(":run"))).toBe(true);
      log.forEach(({ entry }, index) => {
        if (entry.startsWith("progress:")) expect(log[index + 1]?.entry).toBe("yield");
      });
      // Every run ends in one "ran" line; the last one counts every run.
      const ran = log.filter(({ entry }) => entry.startsWith("progress:ran"));
      expect(ran).toHaveLength(positions.length);
      expect(ran.at(-1)?.entry).toMatch(
        new RegExp(`· ${positions.length} / ${positions.length} runs$`),
      );
    });
  }

  it("stops the A/B at the next open after a cancel", async () => {
    const controller = new AbortController();
    const onStep = armStepper({
      positions: runPositions(2, 3, true),
      opens: "Session",
      signal: controller.signal,
      progress: () => undefined,
      yieldTurn: () => Promise.resolve(),
    });
    await onStep({ arm: "base", step: "open" });
    controller.abort(new Error("cancelled"));
    await onStep({ arm: "base", step: "ran" });
    await expect(onStep({ arm: "rule", step: "open" })).rejects.toThrow("cancelled");
  });
});

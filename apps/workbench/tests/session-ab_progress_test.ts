import type { SessionAbArm } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import { armStepLine, armStepper, fetchLine, probeStepLine } from "../src/session-ab/progress.ts";
import { runPositions, steppedArm } from "../src/session-ab/shared.ts";

import { spanRunner } from "./helpers/session-ab-runner.ts";

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
   * run, report and yield. (That steppedArm reports outside the timed spans is the shared module's
   * test.)
   */
  const observe = async (openPerVisit: boolean) => {
    const log: string[] = [];
    const fake = (name: string): SessionAbArm => ({
      name,
      open: () => Promise.resolve(),
      run: () => {
        log.push(`${name}:run`);
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
      progress: (text) => log.push(`progress:${text}`),
      yieldTurn: () => {
        log.push("yield");
        return Promise.resolve();
      },
    });
    await spanRunner().run(
      [steppedArm(fake("base"), onStep), steppedArm(fake("rule"), onStep)],
      3,
      openPerVisit,
    );
    return { log, positions };
  };

  for (const openPerVisit of [true, false]) {
    it(`yields once after every report, and reports every run once, counted (openPerVisit: ${openPerVisit})`, async () => {
      const { log, positions } = await observe(openPerVisit);
      log.forEach((entry, index) => {
        if (entry.startsWith("progress:")) expect(log[index + 1]).toBe("yield");
      });
      // Every run ends in one "ran" line; the last one counts every run.
      const ran = log.filter((entry) => entry.startsWith("progress:ran"));
      expect(ran).toHaveLength(positions.length);
      expect(ran.at(-1)).toMatch(new RegExp(`· ${positions.length} / ${positions.length} runs$`));
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

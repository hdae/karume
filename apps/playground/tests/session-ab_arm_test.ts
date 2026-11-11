import { readFileSync } from "node:fs";

import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import type {
  ExpertCacheReport,
  GenerationEvent,
  GenerationStop,
  RunPhase,
} from "@karume/models/gemma";
import type { SessionOptions } from "@karume/runtime";
import { runSessionAb } from "@karume/runtime/tune";
import { describe, expect, it } from "vite-plus/test";

import {
  type ArmPipeline,
  type ArmRequest,
  type ArmSequence,
  type CountsDiagnostics,
  type DiagnosticsCallback,
  pipelineArm,
} from "../src/models/session-ab/run.ts";

const ENVIRONMENT = {
  adapter: { vendor: "test", architecture: "", device: "", description: "" },
  userAgent: "test",
  backend: "other",
  date: "2026-11-11T00:00:00.000Z",
};

const RULE_KEY = "ext:candidate:gelu_mul:v1:gelu-other:f32:wg128";

/**
 * The diagnostics of one scripted run: its phase in the counts, so a test can tell which run was
 * counted (`dispatches` = 1000 + the step or cycle), and the rule's key next to a built-in one.
 */
const indexOf = (phase: RunPhase): number => {
  switch (phase.kind) {
    case "prefill":
      return phase.chunk;
    case "decode":
      return phase.step;
    case "draft":
    case "verify":
      return phase.cycle;
  }
};

const diagnosticsOf = (phase: RunPhase, rules: boolean): CountsDiagnostics => {
  const kindBase = { prefill: 0, decode: 1000, verify: 2000, draft: 3000 }[phase.kind];
  return {
    lastRunFusions: rules ? { [GELU_MUL_RULE_NAME]: 60 } : {},
    lastRunPipelines: [
      { key: "linear:f32", dispatchCount: kindBase + indexOf(phase) },
      ...(rules ? [{ key: RULE_KEY, dispatchCount: 60 }] : []),
    ],
  };
};

const cacheReport = (slots: readonly number[], depth?: number): ExpertCacheReport => ({
  runs: 0,
  missedRuns: 0,
  loads: 0,
  loadedExperts: 0,
  evictedExperts: 0,
  hostExperts: 0,
  containerExperts: 0,
  bytesRead: 0,
  bytesUploaded: 0,
  loadMs: 0,
  readMs: 0,
  uploadMs: 0,
  hostHeldExperts: 0,
  hostHeldBytes: 0,
  slots,
  committedRuns: 0,
  retries: 0,
  ...(depth === undefined
    ? {}
    : {
        speculation: {
          depth,
          cycles: 0,
          draftRuns: 0,
          acceptedDrafts: 0,
          verifyRetries: 0,
          partialCommits: 0,
        },
      }),
});

/** A plain decode: one prefill chunk, then one decode run per token after the first. */
const decodePhases = (tokens: number): readonly RunPhase[] => [
  { kind: "prefill", chunk: 1, chunks: 1 },
  ...Array.from({ length: tokens - 1 }, (_, step): RunPhase => ({
    kind: "decode",
    step: step + 1,
  })),
];

/** A self-speculating decode: the prefill, then verify runs only (depth 2: three rows a cycle). */
const verifyPhases = (cycles: number): readonly RunPhase[] => [
  { kind: "prefill", chunk: 1, chunks: 1 },
  ...Array.from({ length: cycles }, (_, cycle): RunPhase => ({
    kind: "verify",
    cycle: cycle + 1,
    rows: 3,
    accepted: 2,
  })),
];

type Rig = {
  /** Everything the fake pipeline and the arm's callbacks did, in order. */
  readonly log: string[];
  /** The options each build received. */
  readonly builds: { readonly fusionRules?: unknown; readonly onRunDiagnostics?: unknown }[];
  /** The log entries made while the runner's clock was open (inside a timed span). */
  readonly timed: string[];
  /** The runner's clock: every read opens or closes the span. */
  readonly clock: () => number;
  readonly record: (entry: string) => void;
};

const newRig = (): Rig => {
  const log: string[] = [];
  const timed: string[] = [];
  let open = false;
  let tick = 0;
  return {
    log,
    timed,
    builds: [],
    clock: () => {
      open = !open;
      tick += 1;
      return tick;
    },
    record: (entry) => {
      log.push(entry);
      if (open) timed.push(entry);
    },
  };
};

type FakeOptions = {
  readonly phases: readonly RunPhase[];
  readonly report?: ExpertCacheReport;
  readonly generateThrows?: boolean;
  readonly disposeThrows?: boolean;
};

/** A fake pipeline build: a sequence whose generation emits the scripted phases and tokens. */
const fakeBuild = (where: Rig, name: string, fake: FakeOptions) => {
  return (options: {
    readonly fusionRules?: SessionOptions["fusionRules"];
    readonly onRunDiagnostics?: DiagnosticsCallback;
  }): Promise<ArmPipeline> => {
    where.builds.push(options);
    where.record(`${name}:build`);
    const rules = options.fusionRules !== undefined;
    const sequence = (): ArmSequence => ({
      generate: (request) => {
        where.record(`${name}:generate`);
        if (fake.generateThrows === true) throw new Error("the generation failed");
        const events = async function* (): AsyncGenerator<GenerationEvent> {
          let position = request.prompt.length;
          for (const phase of fake.phases) {
            options.onRunDiagnostics?.(diagnosticsOf(phase, rules), phase);
            if (phase.kind === "draft") continue;
            yield { kind: "token", id: 7, position };
            position += 1;
          }
        };
        const done: Promise<GenerationStop> = Promise.resolve({
          tokens: fake.phases.length,
          reason: "max-tokens",
        });
        return Object.assign(events(), { done });
      },
      dispose: () => {
        where.record(`${name}:sequence.dispose`);
        return Promise.resolve();
      },
    });
    return Promise.resolve({
      sequence: () => {
        where.record(`${name}:sequence`);
        return Promise.resolve(sequence());
      },
      expertCacheReport: () => fake.report,
      dispose: () => {
        where.record(`${name}:dispose`);
        return fake.disposeThrows === true
          ? Promise.reject(new Error("the dispose failed"))
          : Promise.resolve();
      },
    });
  };
};

const REQUEST: ArmRequest = {
  tokens: 4,
  sampler: { temperature: 0, seed: 0 },
  capacity: 1024,
  readCounts: true,
};

/** The page's two arms: `base` without rules, `rule` with the candidate rule. */
const ARMS: readonly {
  readonly name: string;
  readonly fusionRules?: SessionOptions["fusionRules"];
}[] = [{ name: "base" }, { name: "rule", fusionRules: [GELU_MUL_RULE] }];

const armsOf = (where: Rig, request: ArmRequest, fake: FakeOptions) =>
  ARMS.map(({ name, fusionRules }) =>
    pipelineArm({
      name,
      ...(fusionRules === undefined ? {} : { fusionRules }),
      request,
      build: fakeBuild(where, name, fake),
      prompt: () => {
        where.record(`${name}:prompt`);
        return [1, 2, 3];
      },
      signal: new AbortController().signal,
      onStep: ({ step }) => where.record(`${name}:step:${step}`),
    }),
  );

const run = (where: Rig, arms: ReturnType<typeof armsOf>, openPerVisit: boolean) =>
  runSessionAb({ arms, rounds: 3, openPerVisit, environment: ENVIRONMENT, clock: where.clock });

describe("the session A/B page's arm on a Gemma 4 pipeline", () => {
  describe("the options each arm builds with", () => {
    it("passes the rules to the rule arm alone, and the callback only when counts are read", async () => {
      for (const readCounts of [true, false]) {
        const where = newRig();
        const arms = armsOf(where, { ...REQUEST, readCounts }, { phases: decodePhases(4) });
        await run(where, arms, false);
        const [base, rule] = where.builds;
        expect(where.builds).toHaveLength(2);
        expect(Object.hasOwn(base, "fusionRules")).toBe(false);
        expect(rule.fusionRules).toEqual([GELU_MUL_RULE]);
        for (const options of where.builds) {
          expect(Object.hasOwn(options, "onRunDiagnostics"), `readCounts ${readCounts}`).toBe(
            readCounts,
          );
        }
      }
    });
  });

  describe("the timed span", () => {
    for (const openPerVisit of [false, true]) {
      it(`holds the generation alone: the sequence, the prompt and the dispose are outside it (openPerVisit: ${openPerVisit})`, async () => {
        const where = newRig();
        await run(where, armsOf(where, REQUEST, { phases: decodePhases(4) }), openPerVisit);
        // Two clocked runs per arm per round (the warm-up round's too), one generation each.
        expect(where.timed).toHaveLength(2 * 2 * 4);
        expect(new Set(where.timed)).toEqual(new Set(["base:generate", "rule:generate"]));
        // The control: what is kept out of the span did happen.
        for (const entry of ["sequence", "prompt", "sequence.dispose", "dispose"]) {
          expect(where.log, entry).toContain(`rule:${entry}`);
        }
      });
    }

    it("generates every run in a sequence of its own, created before the run and disposed after it", async () => {
      const where = newRig();
      await run(where, armsOf(where, REQUEST, { phases: decodePhases(4) }), false);
      const base = where.log.filter((entry) => entry.startsWith("base:"));
      const generations = base.filter((entry) => entry === "base:generate").length;
      expect(generations).toBe(8);
      // One sequence ready at open, then one more after every run; every used one disposed.
      expect(base.filter((entry) => entry === "base:sequence")).toHaveLength(generations + 1);
      expect(base.filter((entry) => entry === "base:sequence.dispose")).toHaveLength(generations);
      const first = base.indexOf("base:generate");
      expect(base.slice(0, first)).toEqual([
        "base:step:open",
        "base:build",
        "base:prompt",
        "base:sequence",
      ]);
    });
  });

  describe("the counts", () => {
    it("are the first decode run's, and the rule's dispatch keys come from it", async () => {
      const where = newRig();
      const arms = armsOf(where, REQUEST, { phases: decodePhases(4) });
      const record = await run(where, arms, true);
      const [base, rule] = record.arms;
      // decode step 1 = 1000 + 1, plus the rule's 60.
      expect([base.dispatches, rule.dispatches]).toEqual([1001, 1061]);
      expect(rule.fusions).toEqual({ [GELU_MUL_RULE_NAME]: 60 });
      expect(arms.map((arm) => arm.facts().ruleKeys)).toEqual([[], [RULE_KEY]]);
    });

    it("under self-speculation are the first verify run's (no decode run happens)", async () => {
      const where = newRig();
      const request: ArmRequest = { ...REQUEST, expertCache: { slots: 64, depth: 2 } };
      const record = await run(
        where,
        armsOf(where, request, { phases: verifyPhases(2), report: cacheReport([64, 64], 2) }),
        true,
      );
      expect(record.arms.map(({ dispatches }) => dispatches)).toEqual([2001, 2061]);
    });

    it("refuse a run without the counted run: a verify-only decode without a depth set", async () => {
      const where = newRig();
      await expect(
        run(where, armsOf(where, REQUEST, { phases: verifyPhases(2) }), true),
      ).rejects.toThrow("Arm base: the run had no decode run to count");
    });

    it("are not read when the request reads none: the record has no counts and no keys", async () => {
      const where = newRig();
      const arms = armsOf(where, { ...REQUEST, readCounts: false }, { phases: decodePhases(4) });
      const record = await run(where, arms, false);
      for (const arm of record.arms) {
        expect(Object.hasOwn(arm, "fusions") || Object.hasOwn(arm, "dispatches")).toBe(false);
      }
      expect(arms.map((arm) => arm.facts())).toEqual([{}, {}]);
    });
  });

  describe("the expert cache as resolved", () => {
    it("is the pipeline's slots per layer, the host tier as passed and the depth", async () => {
      const where = newRig();
      const request: ArmRequest = {
        ...REQUEST,
        // One count for every layer, as typed; the pipeline resolves it per layer.
        expertCache: { slots: 46, hostBytes: 2 ** 30, depth: 2 },
      };
      const arms = armsOf(where, request, {
        phases: verifyPhases(2),
        report: cacheReport([40, 52], 2),
      });
      await run(where, arms, true);
      expect(arms[0].facts().expertCache).toEqual({
        slots: [40, 52],
        hostBytes: 2 ** 30,
        depth: 2,
      });
    });
  });

  describe("close", () => {
    it("disposes the pipeline when a run throws", async () => {
      const where = newRig();
      await expect(
        run(where, armsOf(where, REQUEST, { phases: decodePhases(4), generateThrows: true }), true),
      ).rejects.toThrow("the generation failed");
      expect(where.log.slice(-2)).toEqual(["base:step:close", "base:dispose"]);
    });

    it("lets the dispose's error propagate", async () => {
      const where = newRig();
      const [base] = armsOf(where, REQUEST, { phases: decodePhases(4), disposeThrows: true });
      await base.open();
      await expect(base.close()).rejects.toThrow("the dispose failed");
      expect(where.log.slice(-2)).toEqual(["base:step:close", "base:dispose"]);
    });
  });

  describe("its twin in the opt-in Node A/B", () => {
    /** The text between the twin markers, `export` aside. */
    const twinOf = (path: string): string => {
      const text = readFileSync(new URL(path, import.meta.url), "utf8");
      const begin = text.indexOf("// ---- The arm builder: twin begin");
      const endMarker = "// ---- The arm builder: twin end ----";
      const end = text.indexOf(endMarker);
      expect(begin, path).toBeGreaterThanOrEqual(0);
      expect(end, path).toBeGreaterThan(begin);
      return text.slice(begin, end + endMarker.length).replace(/^export /gm, "");
    };

    it("is the same text as the page's (a change to one is made to the other)", () => {
      const page = twinOf("../src/models/session-ab/run.ts");
      expect(page).toContain("const pipelineArm = <P extends ArmPipeline>");
      expect(twinOf("../../../packages/e2e/tests/e2e_gemma4_session_ab_test.ts")).toBe(page);
    });
  });
});

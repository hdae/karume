/**
 * The generator's plan of GEMV measurements (`--gemv-plan`): what the GEMV sweep records at hand lack
 * for the GEMV derivation to be complete, written as a workbench batch that the workbench's own batch
 * reader accepts (its `resolveBatch` against its `gemv-sweep` entry) and whose steps measure exactly
 * the missing candidates, twice; with no record at hand, the whole `full` set, twice. Each case runs
 * on a temporary repository; the GEMV records are built from the runtime's own measurement plans
 * (packages/runtime/tests/helpers/gemv-sweep-records.ts). No GPU.
 *
 * Faults injected: staged-l8's kernel fingerprint replaced at M = 4 in one of two records (the plan
 * lists that candidate alone for that case and row count); `--records` beside `--gemv-plan` (a usage
 * error).
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { BUILTIN_GEOMETRY_PROFILES } from "@karume/runtime";
import { gemvSweepCases } from "@karume/runtime/tune";
import { makeTempDir, remove } from "@karume/test-support/fs";
import { afterEach, beforeEach, describe, it } from "vite-plus/test";
import {
  BATCH_FORMAT as WORKBENCH_BATCH_FORMAT,
  resolveBatch,
  UNTRACKED_BATCHES_DIRECTORY as WORKBENCH_UNTRACKED,
} from "../../../apps/workbench/src/batch.ts";
import gemvSweepEntry from "../../../apps/workbench/src/entries/gemv-sweep.ts";
import {
  type GemvCaseSpec,
  gemvRecord,
} from "../../../packages/runtime/tests/helpers/gemv-sweep-records.ts";
import { type CommandEnvironment, main } from "../generate.ts";
import {
  BATCH_FORMAT,
  gemvPlan,
  PLAN_PASSES,
  planBatchId,
  planSteps,
  UNTRACKED_BATCHES_DIRECTORY,
} from "../plan.ts";
import { loadRecord } from "../records.ts";
import { bytesOf, workbenchRecord } from "./helpers/records.ts";

const GEMV_A = "outputs/workbench/node-test-gpu/gemv-sweep_a.json";
const GEMV_B = "outputs/workbench/node-test-gpu/gemv-sweep_b.json";
const BATCH = `${UNTRACKED_BATCHES_DIRECTORY}/${planBatchId("test-gpu")}.json`;

/** E2B's per_layer_model_projection (its default's choice at reorder is parallel-l4). */
const PLE = "i4g32-n8960-k1536";
const COMPLETE: GemvCaseSpec = { caseId: PLE, rows: [1, 4, 8] };
const STALE: GemvCaseSpec = {
  ...COMPLETE,
  fields: { 4: { "staged-l8": { kernel: "0000000000000000" } } },
};

let root = "";
let stdout: string[] = [];
let stderr: string[] = [];

const environment = (): CommandEnvironment => ({
  repositoryRoot: root,
  registered: BUILTIN_GEOMETRY_PROFILES,
  lastCommit: () => "fedcba9876543210fedcba9876543210fedcba98",
  stdout: (text) => stdout.push(text),
  stderr: (text) => stderr.push(text),
});

const write = (path: string, spec: GemvCaseSpec, date: string): void => {
  mkdirSync(join(root, path, ".."), { recursive: true });
  writeFileSync(join(root, path), bytesOf(workbenchRecord("gemv-sweep", gemvRecord(date, [spec]))));
};

const loaded = (path: string) => loadRecord(readFileSync(join(root, path)), path, "gemv-records");

/** The batch file as the workbench reads it, and its resolution. */
const resolved = () => {
  const text = readFileSync(join(root, BATCH), "utf8");
  return { text, resolution: resolveBatch(planBatchId("test-gpu"), text, [gemvSweepEntry]) };
};

beforeEach(async () => {
  root = await makeTempDir({ prefix: "karume-geometry-profiles-plan-" });
  stdout = [];
  stderr = [];
  write(GEMV_A, COMPLETE, "2026-11-17T11:00:00.000Z");
  write(GEMV_B, STALE, "2026-11-17T12:00:00.000Z");
});

afterEach(async () => {
  await remove(root, { recursive: true });
});

describe("the plan of what the GEMV records lack", () => {
  it("writes the batch where, and in the format, the workbench reads untracked batches", () => {
    assertEquals(UNTRACKED_BATCHES_DIRECTORY, WORKBENCH_UNTRACKED);
    assertEquals(BATCH_FORMAT, WORKBENCH_BATCH_FORMAT);
  });

  it("lists exactly the candidate one record holds no current row of, and every candidate of the cases no record holds (fault: staged-l8's kernel replaced at M = 4)", () => {
    const plan = gemvPlan({ profileId: "test-gpu", records: [loaded(GEMV_A), loaded(GEMV_B)] });
    assertEquals(
      plan.missing.filter(({ caseId }) => caseId === PLE),
      [{ caseId: PLE, m: 4, candidates: ["staged-l8"] }],
    );
    const others = new Set(plan.missing.map(({ caseId }) => caseId));
    assertEquals(others.size, gemvSweepCases().length);
    // The exact step for the stale row, once per pass
    const exact = plan.batch?.steps.filter(({ parameters }) => parameters.cases?.includes(PLE));
    assertEquals(
      exact?.map(({ parameters }) => parameters),
      Array.from({ length: PLAN_PASSES }, () => ({
        candidateSet: "full" as const,
        cases: [PLE],
        rows: ["4"],
        candidates: ["staged-l8"],
      })),
    );
    // Two passes of the same steps
    const steps = plan.batch?.steps ?? [];
    assertEquals(steps.length % PLAN_PASSES, 0);
    const pass = steps.length / PLAN_PASSES;
    assertEquals(steps.slice(pass), steps.slice(0, pass));
    assertEquals(planSteps(plan.missing), steps.slice(0, pass));
  });

  it("keeps out of the batch what only a steady record fills, and lists it apart (fault: PLE at M = 4 and 8 counted as filled by the two records of no class)", () => {
    // A steady record of PLE and 19 more cases at M = 1 (passes of 100 ms): PLE's decisions are the
    // rule of one steady record's, which reads no record of no class
    const STEADY = "outputs/workbench/node-test-gpu/gemv-sweep_steady.json";
    const padding = gemvSweepCases()
      .filter(({ id }) => id !== PLE)
      .slice(0, 19)
      .map(({ id }) => ({ caseId: id }));
    writeFileSync(
      join(root, STEADY),
      bytesOf(
        workbenchRecord(
          "gemv-sweep",
          gemvRecord("2026-11-17T10:00:00.000Z", [{ caseId: PLE }, ...padding], {
            passes: { reps: 100_000 },
          }),
        ),
      ),
    );
    const plan = gemvPlan({
      profileId: "test-gpu",
      records: [loaded(STEADY), loaded(GEMV_A), loaded(GEMV_B)],
    });
    assertEquals(
      plan.missing.filter(({ caseId }) => caseId === PLE),
      [],
    );
    assertEquals(
      plan.missingSteady.filter(({ caseId }) => caseId === PLE).map(({ m }) => m),
      [4, 8],
    );
    assertEquals(
      plan.batch?.steps.some(({ parameters }) => parameters.cases?.includes(PLE)),
      false,
    );
    const text = plan.lines.join("\n");
    assertStringIncludes(text, "Only a steady record fills");
    assertStringIncludes(text, `  - ${PLE} M=4: `);
  });

  it("is written by the command as a batch the workbench's batch reader accepts, each step measuring the cases, row counts and candidates it names", async () => {
    const code = await main(
      ["test-gpu", "--gemv-plan", `--gemv-records=${GEMV_A},${GEMV_B}`],
      environment(),
    );
    assertEquals(code, 0, stderr.join("\n"));
    assertStringIncludes(stdout.join("\n"), `Wrote ${BATCH}`);
    const { resolution } = resolved();
    assert(resolution.kind === "resolved", JSON.stringify(resolution));
    const exact = resolution.steps.filter(({ parameters }) => {
      const cases = parameters.cases;
      return Array.isArray(cases) && cases.includes(PLE);
    });
    assertEquals(
      exact.map(({ entry, parameters }) => [
        entry.id,
        parameters.candidateSet,
        parameters.cases,
        parameters.rows,
        parameters.candidates,
      ]),
      Array.from({ length: PLAN_PASSES }, () => [
        "gemv-sweep",
        "full",
        [PLE],
        ["4"],
        ["staged-l8"],
      ]),
    );
  });

  it("with no record at hand writes the whole measurement: the full set, twice", async () => {
    const code = await main(["test-gpu", "--gemv-plan"], environment());
    assertEquals(code, 0, stderr.join("\n"));
    const { resolution } = resolved();
    assert(resolution.kind === "resolved", JSON.stringify(resolution));
    assertEquals(resolution.steps.length, PLAN_PASSES);
    for (const step of resolution.steps) {
      assertEquals(step.given, { candidateSet: "full" });
      assertEquals([step.parameters.cases, step.parameters.candidates], [[], []]);
    }
  });

  it("in a dry run prints the batch and writes nothing", async () => {
    const code = await main(
      ["test-gpu", "--gemv-plan", "--dry-run", `--gemv-records=${GEMV_A},${GEMV_B}`],
      environment(),
    );
    assertEquals(code, 0, stderr.join("\n"));
    const printed = stdout.join("\n");
    assertStringIncludes(printed, `Dry run: nothing written. The batch ${BATCH}:`);
    assertStringIncludes(printed, `"format": "${BATCH_FORMAT}"`);
    assertEquals(readdirSync(root).includes("inputs"), false);
  });

  it("refuses the derivation's options beside it (fault: --records with --gemv-plan)", async () => {
    const code = await main(["test-gpu", "--gemv-plan", `--records=${GEMV_A}`], environment());
    assertEquals(code, 2);
    assertStringIncludes(
      stderr[0] ?? "",
      "--gemv-plan reads GEMV sweep records only (drop --records)",
    );
  });
});

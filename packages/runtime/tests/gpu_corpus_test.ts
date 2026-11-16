// The runtime's regression corpus on a real GPU: each entry's graph (a real model cut to a few
// layers — tests/corpus/entries.ts) runs on weights generated from its seed, through a driver that
// knows only the declaration. No real weights, no models or hub code, no read of models/.
//
// Per entry and per run of its plan:
//
// 1. fusion identity — `exact` and `exact` with every fusion rule disabled give the same output
//    words at every step, and the comparison is not vacuous (the fused side's rule counters are
//    above 0 at every step, the plain side's are 0);
// 2. determinism — two Sessions of the same settings give the same words, at `exact` and at
//    `reorder`;
// 3. the comparison can fail — weights one quantization level apart in one element give other
//    words;
// 4. the entry exercises `reorder` — at least one step dispatches another kernel key set than at
//    `exact`;
// 5. outputs are finite and vary;
// 6. sha256 rows per environment (ADR 0106 — helpers/reference.ts): per level, one row for the
//    output words of every step and one for the census (kernel keys with their dispatch counts and
//    the fusion counters, per step), the artifacts kept the way that mechanism keeps them;
// 7. the distance between `exact` and `reorder` per step goes to the results as information. It is
//    never asserted: an end-to-end tolerance cannot tell a defect from a reordering, and the sha256
//    rows are never replaced by one.
//
// The weights are generated once per entry; Sessions are opened one after another and disposed.
//
// SPEC: docs/spec/verification.md#what-is-checked

import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  createSessionFromContainer,
  DISABLED_FUSION_RULES,
  type SessionOptions,
} from "../src/runtime/executor.ts";
import { FUSION_RULES } from "../src/runtime/fusion.ts";
import { type CorpusRun, driveRun, type StepOutput, type StepResult } from "./corpus/driver.ts";
import { CORPUS_ENTRIES, type CorpusEntry } from "./corpus/entries.ts";
import { openCorpusFixture } from "./corpus/fixture.ts";
import { constSideOf, corpusSupply } from "./corpus/supply.ts";
import { generateWeights, perturbOneLevel } from "./corpus/weights.ts";
import { verifyRunningGpu } from "./helpers/environment.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  openReferences,
  referenceEntryFields,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleReference,
} from "./helpers/reference.ts";
import { openResults, runRecordedCase } from "./helpers/results.ts";
import { buildSafetensors } from "./helpers/safetensors.ts";

/** The corpus's sha256 rows, one per environment key. */
const references = openReferences(new URL("./fixtures/references/corpus.json", import.meta.url));
/** Artifacts and outcomes (`outputs/verify/<environment key>/<date>_corpus/` — safe to delete). */
const results = openResults("corpus");

type Level = "exact" | "reorder";

type Side = "exact" | "exactAgain" | "exactUnfused" | "reorder" | "reorderAgain" | "exactPerturbed";

/** The Sessions every run is driven through, in the order they are opened. */
const SIDES: readonly {
  readonly side: Side;
  readonly weights: "base" | "perturbed";
  readonly options: SessionOptions;
}[] = [
  { side: "exact", weights: "base", options: { numerics: { level: "exact" } } },
  { side: "exactAgain", weights: "base", options: { numerics: { level: "exact" } } },
  {
    side: "exactUnfused",
    weights: "base",
    options: { numerics: { level: "exact" }, [DISABLED_FUSION_RULES]: "all" },
  },
  { side: "reorder", weights: "base", options: { numerics: { level: "reorder" } } },
  { side: "reorderAgain", weights: "base", options: { numerics: { level: "reorder" } } },
  { side: "exactPerturbed", weights: "perturbed", options: { numerics: { level: "exact" } } },
];

/** Every side's steps, per run name. */
type EntryOutcome = ReadonlyMap<Side, ReadonlyMap<string, readonly StepResult[]>>;

const RULE_NAMES = FUSION_RULES.map((rule) => rule.name);

/** The rule counters of one step (the annotation counters are not rules and are left out). */
const ruleCounters = (step: StepResult): readonly (readonly [string, number])[] =>
  RULE_NAMES.map((name) => [name, step.fusions[name]] as const);

/** Where two sides' steps first differ in their words (`undefined` when every word is equal). */
const firstDifference = (
  a: readonly StepResult[],
  b: readonly StepResult[],
): string | undefined => {
  if (a.length !== b.length) return `${a.length} steps against ${b.length}`;
  for (const [index, step] of a.entries()) {
    for (const [at, output] of step.outputs.entries()) {
      const other = b[index].outputs[at];
      if (other.words.length !== output.words.length) {
        return `step ${index} output ${output.name}: ${output.words.length} words against ${other.words.length}`;
      }
      let differing = 0;
      for (let i = 0; i < output.words.length; i += 1) {
        if (output.words[i] !== other.words[i]) differing += 1;
      }
      if (differing > 0) {
        return `step ${index} output ${output.name}: ${differing} of ${output.words.length} words differ`;
      }
    }
  }
  return undefined;
};

const f32Of = (output: StepOutput): Float32Array<ArrayBuffer> =>
  new Float32Array(output.words.buffer, output.words.byteOffset, output.words.length);

const SAFETENSORS_DTYPES = { f32: "F32", i32: "I32", bool: "U32" } as const;

/** The artifact of a run's outputs: every step's output words, in step and declaration order. */
const outputsArtifact = (steps: readonly StepResult[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(
    buildSafetensors(
      steps.flatMap((step, index) =>
        step.outputs.map((output) => ({
          name: `step${index}.${output.name}`,
          dtype: SAFETENSORS_DTYPES[output.dtype],
          shape: output.shape,
          data: new Uint8Array(
            output.words.buffer,
            output.words.byteOffset,
            output.words.byteLength,
          ),
        })),
      ),
    ),
  );

/**
 * The census of a run as canonical text: per step its rows, every kernel key with its dispatch
 * count, and every fusion counter that is not 0 (a rule added later at 0 does not move the row).
 */
const censusText = (steps: readonly StepResult[]): string =>
  steps
    .map((step, index) => {
      const kernels = [...step.pipelines]
        .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))
        .map((row) => `  kernel ${row.key} ${row.dispatchCount}`);
      const fusions = Object.entries(step.fusions)
        .filter(([, count]) => count !== undefined && count !== 0)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([name, count]) => `  fusion ${name} ${String(count)}`);
      return [`step ${index} rows ${step.rows}`, ...kernels, ...fusions].join("\n");
    })
    .join("\n")
    .concat("\n");

/** max |exact − reorder| and the largest relative difference, per output of one step. */
const distanceOf = (exact: StepResult, reorder: StepResult): string =>
  exact.outputs
    .flatMap((output, at) => (output.dtype === "f32" ? [{ output, at }] : []))
    .map(({ output, at }) => {
      const a = f32Of(output);
      const b = f32Of(reorder.outputs[at]);
      let scale = 0;
      for (const value of a) scale = Math.max(scale, Math.abs(value));
      // Relative to max(|exact|, 1e-3 · max |exact|): elements near 0 do not dominate.
      const floor = scale * 1e-3;
      let maxAbs = 0;
      let maxRel = 0;
      for (let i = 0; i < a.length; i += 1) {
        const difference = Math.abs(a[i] - b[i]);
        maxAbs = Math.max(maxAbs, difference);
        maxRel = Math.max(maxRel, difference / Math.max(Math.abs(a[i]), floor));
      }
      return `${output.name} maxAbs ${maxAbs.toExponential(2)} maxRel ${maxRel.toExponential(2)}`;
    })
    .join(", ");

const rowId = (
  entry: CorpusEntry,
  run: CorpusRun,
  level: Level,
  kind: "outputs" | "census",
): string => `${entry.name}/${run.name}/${level}/${kind}`;

/** Settles one sha256 row, recording the outcome before a mismatch fails. */
const settleRow = async (
  id: string,
  artifact: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<void> => {
  let settlement: ReferenceSettlement | undefined;
  await runRecordedCase(
    results,
    { id, failureNote: (cause) => (cause instanceof Error ? cause.message : String(cause)) },
    async () => {
      settlement = await settleReference(references, results, { id, artifact, bytes });
      return referenceEntryFields(settlement, artifact);
    },
  );
  if (settlement?.check.status === "fail") {
    throw new Error(referenceMismatchMessage(id, settlement, references));
  }
};

const CASE_IDS: string[] = [];

for (const entry of CORPUS_ENTRIES) {
  describe.skipIf(!GPU_AVAILABLE)(`corpus entry ${entry.name} (real GPU)`, () => {
    let gpu: GpuContext | undefined;
    let outcome: EntryOutcome | undefined;
    const outcomeOf = (side: Side, run: CorpusRun): readonly StepResult[] => {
      const steps = outcome?.get(side)?.get(run.name);
      if (steps === undefined) throw new Error(`no steps of side ${side} for run ${run.name}`);
      return steps;
    };

    beforeAll(async () => {
      gpu = verifyRunningGpu(await acquireGpu());
      const fixture = await openCorpusFixture(entry);
      const declaration = fixture.bound.declaration;
      let start = performance.now();
      const weights = generateWeights(fixture.bound, fixture.storage, entry.seed);
      const generationMs = performance.now() - start;
      const original = weights.get(entry.perturbed);
      if (original === undefined) {
        throw new Error(
          `entry ${entry.name}: the perturbed initializer '${entry.perturbed}' is not generated`,
        );
      }
      const perturbed = new Map(weights).set(
        entry.perturbed,
        perturbOneLevel(entry.perturbed, original),
      );
      const consts = constSideOf(fixture.container);
      const supplies = {
        base: corpusSupply({ graphName: entry.graph, declaration, consts, weights }),
        perturbed: corpusSupply({
          graphName: entry.graph,
          declaration,
          consts,
          weights: perturbed,
        }),
      };
      const sides = new Map<Side, Map<string, readonly StepResult[]>>();
      const construction: string[] = [];
      for (const { side, weights: which, options } of SIDES) {
        start = performance.now();
        const session = await createSessionFromContainer(
          gpu,
          supplies[which],
          entry.graph,
          options,
        );
        construction.push(`${side} ${(performance.now() - start).toFixed(0)} ms`);
        try {
          const runs = new Map<string, readonly StepResult[]>();
          for (const run of entry.plan.runs) {
            runs.set(run.name, await driveRun(session, declaration, entry.plan, run, entry.seed));
          }
          sides.set(side, runs);
        } finally {
          await session.dispose();
        }
      }
      outcome = sides;
      const note =
        `fixture ${fixture.graphBytes} bytes; generated ${weights.size} weights in ${generationMs.toFixed(0)} ms; ` +
        `Session construction: ${construction.join(", ")}`;
      console.log(`[corpus] ${entry.name}: ${note}`);
      await results.record({ id: `${entry.name}/timing`, status: "pass", elapsedMs: 0, note });
    });

    afterAll(() => {
      gpu?.destroy();
    });

    for (const run of entry.plan.runs) {
      describe(`run ${run.name} (steps ${run.steps.join(", ")})`, () => {
        it("fusion identity: exact with every fusion rule disabled gives the same words at every step", () => {
          const fused = outcomeOf("exact", run);
          const plain = outcomeOf("exactUnfused", run);
          assertEquals(firstDifference(fused, plain), undefined);
          for (const [index, step] of fused.entries()) {
            const fired = ruleCounters(step).reduce((total, [, count]) => total + count, 0);
            assert(fired > 0, `step ${index}: no fusion rule fired on the fused side`);
            const plainFired = ruleCounters(plain[index]).filter(([, count]) => count !== 0);
            assertEquals(plainFired, [], `step ${index}: a rule fired on the unfused side`);
          }
        });

        it("determinism: two Sessions at exact give the same words", () => {
          assertEquals(
            firstDifference(outcomeOf("exact", run), outcomeOf("exactAgain", run)),
            undefined,
          );
        });

        it("determinism: two Sessions at reorder give the same words", () => {
          assertEquals(
            firstDifference(outcomeOf("reorder", run), outcomeOf("reorderAgain", run)),
            undefined,
          );
        });

        it("the comparison can fail: one weight one quantization level away gives other words", () => {
          assert(
            firstDifference(outcomeOf("exact", run), outcomeOf("exactPerturbed", run)) !==
              undefined,
            `moving '${entry.perturbed}' element 0 by one level changed no output word`,
          );
        });

        it("reorder selects something: some step dispatches another kernel key set than exact", () => {
          const keys = (step: StepResult): string =>
            [...new Set(step.pipelines.map((row) => row.key))].sort().join("\n");
          const exact = outcomeOf("exact", run);
          const reorder = outcomeOf("reorder", run);
          assert(
            exact.some((step, index) => keys(step) !== keys(reorder[index])),
            "every step dispatched the same kernel keys at exact and at reorder",
          );
        });

        it("outputs are finite and vary at both levels", () => {
          for (const side of ["exact", "reorder"] as const) {
            for (const [index, step] of outcomeOf(side, run).entries()) {
              for (const output of step.outputs) {
                if (output.dtype !== "f32") continue;
                const values = f32Of(output);
                const where = `${side} step ${index} output ${output.name}`;
                assert(values.every(Number.isFinite), `${where}: a value is NaN or infinite`);
                assert(
                  values.some((value) => value !== values[0]),
                  `${where}: every value is equal`,
                );
              }
            }
          }
        });

        for (const level of ["exact", "reorder"] as const) {
          const outputsId = rowId(entry, run, level, "outputs");
          const censusId = rowId(entry, run, level, "census");
          CASE_IDS.push(outputsId, censusId);
          const stem = `${entry.name}.${run.name}.${level}`;
          it.skipIf(references.lacksReference(outputsId))(
            `${level}: the output words of every step equal this machine's sha256 row`,
            () =>
              settleRow(
                outputsId,
                `${stem}.outputs.safetensors`,
                outputsArtifact(outcomeOf(level, run)),
              ),
          );
          it.skipIf(references.lacksReference(censusId))(
            `${level}: the census of every step equals this machine's sha256 row`,
            () =>
              settleRow(
                censusId,
                `${stem}.census.txt`,
                new TextEncoder().encode(censusText(outcomeOf(level, run))),
              ),
          );
        }

        it("the distance between exact and reorder is recorded, not asserted", async () => {
          const exact = outcomeOf("exact", run);
          const reorder = outcomeOf("reorder", run);
          const note = exact
            .map((step, index) => `step ${index}: ${distanceOf(step, reorder[index])}`)
            .join("; ");
          console.log(`[corpus] ${entry.name} ${run.name} exact against reorder: ${note}`);
          await results.record({
            id: `${entry.name}/${run.name}/exact-against-reorder`,
            status: "pass",
            elapsedMs: 0,
            note,
          });
        });
      });
    }
  });
}

if (GPU_AVAILABLE) references.warnMissing(CASE_IDS);
// A missing row of this environment is not a silent green (ADR 0005's rule).
registerReferenceGate(references, { runnable: GPU_AVAILABLE, caseIds: CASE_IDS });

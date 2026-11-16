// The runtime's regression corpus on a real GPU: each entry's graph (a real model cut to a few
// layers, or whole — tests/corpus/entries.ts) runs on weights generated from its seed, through a
// driver that knows only the declaration. No real weights, no models or hub code, no read of
// models/.
//
// Each entry lists the Session settings it is checked under (entries.ts); a setting made of a level
// alone promises the bits of its unfused plan, one that names an approximation promises nothing
// against another plan. Per entry and per run of its plan:
//
// 1. fusion identity, for every setting made of a level alone — the setting and the same setting
//    with every fusion rule disabled give the same output words at every step, and the comparison
//    is not vacuous (the fused side's rule counters are above 0 at every step, the plain side's
//    are 0). An entry whose graph no rule matches says so, and the check is that none fires;
// 2. determinism, for every setting — two Sessions give the same words;
// 3. the comparison can fail — at `exact`, weights one quantization level apart in one element
//    give other words;
// 4. every setting exercises something — no two settings dispatch the same kernel key sets at every
//    step;
// 5. outputs are finite and vary, for every setting;
// 6. sha256 rows per environment (ADR 0106 — helpers/reference.ts): per setting, one row for the
//    output words of every step and one for the census (kernel keys with their dispatch counts and
//    the fusion counters, per step), the artifacts kept the way that mechanism keeps them (an
//    entry with very large outputs keeps the sha256 of each output instead of its words);
// 7. the distance of every other setting from `exact` per step goes to the results as information.
//    It is never asserted: an end-to-end tolerance cannot tell a defect from a reordering, and the
//    sha256 rows are never replaced by one.
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
import {
  type CorpusRun,
  describeStep,
  driveRun,
  type StepOutput,
  type StepResult,
} from "./corpus/driver.ts";
import { CORPUS_ENTRIES, type CorpusEntry, type CorpusSetting } from "./corpus/entries.ts";
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
  sha256Hex,
} from "./helpers/reference.ts";
import { openResults, runRecordedCase } from "./helpers/results.ts";
import { buildSafetensors } from "./helpers/safetensors.ts";

/** The corpus's sha256 rows, one per environment key. */
const references = openReferences(new URL("./fixtures/references/corpus.json", import.meta.url));
/** Artifacts and outcomes (`outputs/verify/<environment key>/<date>_corpus/` — safe to delete). */
const results = openResults("corpus");

/** The setting the perturbed weights run under and the distances are taken from. */
const EXACT = "exact";

/** One Session every run is driven through: a setting, on the base or the perturbed weights. */
type Side = {
  /** The setting's name, `<name>/again`, `<name>/unfused` or `exact/perturbed`. */
  readonly id: string;
  readonly weights: "base" | "perturbed";
  readonly options: SessionOptions;
};

const againOf = (setting: CorpusSetting): string => `${setting.name}/again`;
const unfusedOf = (setting: CorpusSetting): string => `${setting.name}/unfused`;
const PERTURBED = `${EXACT}/perturbed`;

/** The entry's `exact` setting (every entry has one). */
const exactOf = (entry: CorpusEntry): CorpusSetting => {
  const exact = entry.settings.find((setting) => setting.name === EXACT);
  if (exact === undefined) {
    throw new Error(
      `entry ${entry.name} has no setting named '${EXACT}': the perturbed weights and the distances are taken at exact, so add it`,
    );
  }
  return exact;
};

/**
 * The Sessions every run of `entry` is driven through, in the order they are opened: per setting,
 * two Sessions, and a third with every fusion rule disabled when the setting is a level alone (and
 * the entry does not say that no rule matches its graph); then the perturbed weights at `exact`.
 */
const sidesOf = (entry: CorpusEntry): readonly Side[] => {
  const names = entry.settings.map((setting) => setting.name);
  if (new Set(names).size !== names.length) {
    throw new Error(
      `entry ${entry.name} repeats a setting name in [${names.join(", ")}]: give each setting its own name`,
    );
  }
  return [
    ...entry.settings.flatMap((setting): Side[] => [
      { id: setting.name, weights: "base", options: setting.options },
      { id: againOf(setting), weights: "base", options: setting.options },
      ...(setting.kind === "level" && entry.fusion !== "none"
        ? [
            {
              id: unfusedOf(setting),
              weights: "base",
              options: { ...setting.options, [DISABLED_FUSION_RULES]: "all" },
            } satisfies Side,
          ]
        : []),
    ]),
    { id: PERTURBED, weights: "perturbed", options: exactOf(entry).options },
  ];
};

/** Every side's steps (by side id), per run name. */
type EntryOutcome = ReadonlyMap<string, ReadonlyMap<string, readonly StepResult[]>>;

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
      const label = describeStep(step.step);
      return [`step ${index}${label === "" ? "" : ` ${label}`}`, ...kernels, ...fusions].join("\n");
    })
    .join("\n")
    .concat("\n");

/** max |exact − other| and the largest relative difference, per output of one step. */
const distanceOf = (exact: StepResult, other: StepResult): string =>
  exact.outputs
    .flatMap((output, at) => (output.dtype === "f32" ? [{ output, at }] : []))
    .map(({ output, at }) => {
      const a = f32Of(output);
      const b = f32Of(other.outputs[at]);
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

/**
 * The artifact of a run's outputs for an entry whose outputs are too large to keep per run: one
 * line per output with the sha256 of its words, in step and declaration order.
 */
const outputsDigests = async (steps: readonly StepResult[]): Promise<Uint8Array<ArrayBuffer>> => {
  const lines: string[] = [];
  for (const [index, step] of steps.entries()) {
    for (const output of step.outputs) {
      const digest = await sha256Hex(
        new Uint8Array(output.words.buffer, output.words.byteOffset, output.words.byteLength),
      );
      lines.push(
        `step ${index} ${output.name} ${output.dtype} [${output.shape.join(",")}] ${digest}`,
      );
    }
  }
  return new TextEncoder().encode(`${lines.join("\n")}\n`);
};

const rowId = (
  entry: CorpusEntry,
  run: CorpusRun,
  setting: CorpusSetting,
  kind: "outputs" | "census",
): string => `${entry.name}/${run.name}/${setting.name}/${kind}`;

/** The set of kernel keys one step dispatched, as canonical text. */
const keySetOf = (step: StepResult): string =>
  [...new Set(step.pipelines.map((row) => row.key))].sort().join("\n");

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
  const sides = sidesOf(entry);
  const exact = exactOf(entry);
  describe.skipIf(!GPU_AVAILABLE)(`corpus entry ${entry.name} (real GPU)`, () => {
    let gpu: GpuContext | undefined;
    let outcome: EntryOutcome | undefined;
    const outcomeOf = (side: string, run: CorpusRun): readonly StepResult[] => {
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
      const driven = new Map<string, Map<string, readonly StepResult[]>>();
      const construction: string[] = [];
      for (const { id, weights: which, options } of sides) {
        start = performance.now();
        const session = await createSessionFromContainer(
          gpu,
          supplies[which],
          entry.graph,
          options,
        );
        construction.push(`${id} ${(performance.now() - start).toFixed(0)} ms`);
        try {
          const runs = new Map<string, readonly StepResult[]>();
          for (const run of entry.plan.runs) {
            runs.set(run.name, await driveRun(session, declaration, entry.plan, run, entry.seed));
          }
          driven.set(id, runs);
        } finally {
          await session.dispose();
        }
      }
      outcome = driven;
      const note =
        `fixture ${fixture.graphBytes} bytes; generated ${weights.size} weights in ${generationMs.toFixed(0)} ms; ` +
        `Session construction: ${construction.join(", ")}`;
      console.log(`[corpus] ${entry.name}: ${note}`);
      await results.record({ id: `${entry.name}/timing`, status: "pass", elapsedMs: 0, note });
    });

    afterAll(() => {
      gpu?.destroy();
      // Every side's output words stay here until the entry's cases are done (hundreds of MiB for
      // a whole graph that returns its caches): release them before the next entry runs.
      outcome = undefined;
    });

    for (const run of entry.plan.runs) {
      const steps = run.steps.map((step) =>
        typeof step === "number" ? String(step) : `{${describeStep(step)}}`,
      );
      describe(`run ${run.name} (steps ${steps.join(", ")})`, () => {
        for (const setting of entry.settings) {
          if (setting.kind !== "level") continue;
          if (entry.fusion === "none") {
            it(`no fusion rule fires at ${setting.name} (the entry says none matches its graph)`, () => {
              for (const [index, step] of outcomeOf(setting.name, run).entries()) {
                const fired = ruleCounters(step).filter(([, count]) => count !== 0);
                assertEquals(
                  fired,
                  [],
                  `step ${index}: a rule fired — drop the entry's fusion: "none" so that the unfused plan is compared`,
                );
              }
            });
            continue;
          }
          it(`fusion identity: ${setting.name} with every fusion rule disabled gives the same words at every step`, () => {
            const fused = outcomeOf(setting.name, run);
            const plain = outcomeOf(unfusedOf(setting), run);
            assertEquals(firstDifference(fused, plain), undefined);
            for (const [index, step] of fused.entries()) {
              const fired = ruleCounters(step).reduce((total, [, count]) => total + count, 0);
              assert(fired > 0, `step ${index}: no fusion rule fired on the fused side`);
              const plainFired = ruleCounters(plain[index]).filter(([, count]) => count !== 0);
              assertEquals(plainFired, [], `step ${index}: a rule fired on the unfused side`);
            }
          });
        }

        for (const setting of entry.settings) {
          it(`determinism: two Sessions at ${setting.name} give the same words`, () => {
            assertEquals(
              firstDifference(outcomeOf(setting.name, run), outcomeOf(againOf(setting), run)),
              undefined,
            );
          });
        }

        it("the comparison can fail: one weight one quantization level away gives other words", () => {
          assert(
            firstDifference(outcomeOf(exact.name, run), outcomeOf(PERTURBED, run)) !== undefined,
            `moving '${entry.perturbed}' element 0 by one level changed no output word`,
          );
        });

        it("every setting exercises something: no two settings dispatch the same kernel key sets at every step", () => {
          const alike: string[] = [];
          entry.settings.forEach((a, at) => {
            for (const b of entry.settings.slice(at + 1)) {
              const stepsA = outcomeOf(a.name, run);
              const stepsB = outcomeOf(b.name, run);
              if (stepsA.every((step, index) => keySetOf(step) === keySetOf(stepsB[index]))) {
                alike.push(`${a.name} and ${b.name}`);
              }
            }
          });
          assertEquals(alike, [], "these settings dispatched the same kernel keys at every step");
        });

        it("outputs are finite and vary under every setting", () => {
          for (const setting of entry.settings) {
            for (const [index, step] of outcomeOf(setting.name, run).entries()) {
              for (const output of step.outputs) {
                if (output.dtype !== "f32") continue;
                const values = f32Of(output);
                const where = `${setting.name} step ${index} output ${output.name}`;
                assert(values.every(Number.isFinite), `${where}: a value is NaN or infinite`);
                assert(
                  values.some((value) => value !== values[0]),
                  `${where}: every value is equal`,
                );
              }
            }
          }
        });

        for (const setting of entry.settings) {
          const outputsId = rowId(entry, run, setting, "outputs");
          const censusId = rowId(entry, run, setting, "census");
          CASE_IDS.push(outputsId, censusId);
          const stem = `${entry.name}.${run.name}.${setting.name}`;
          it.skipIf(references.lacksReference(outputsId))(
            `${setting.name}: the output words of every step equal this machine's sha256 row`,
            async () =>
              entry.outputsArtifact === "digests"
                ? settleRow(
                    outputsId,
                    `${stem}.outputs.txt`,
                    await outputsDigests(outcomeOf(setting.name, run)),
                  )
                : settleRow(
                    outputsId,
                    `${stem}.outputs.safetensors`,
                    outputsArtifact(outcomeOf(setting.name, run)),
                  ),
          );
          it.skipIf(references.lacksReference(censusId))(
            `${setting.name}: the census of every step equals this machine's sha256 row`,
            () =>
              settleRow(
                censusId,
                `${stem}.census.txt`,
                new TextEncoder().encode(censusText(outcomeOf(setting.name, run))),
              ),
          );
        }

        for (const setting of entry.settings) {
          if (setting.name === exact.name) continue;
          it(`the distance between exact and ${setting.name} is recorded, not asserted`, async () => {
            const reference = outcomeOf(exact.name, run);
            const other = outcomeOf(setting.name, run);
            const note = reference
              .map((step, index) => `step ${index}: ${distanceOf(step, other[index])}`)
              .join("; ");
            console.log(
              `[corpus] ${entry.name} ${run.name} exact against ${setting.name}: ${note}`,
            );
            await results.record({
              id: `${entry.name}/${run.name}/exact-against-${setting.name}`,
              status: "pass",
              elapsedMs: 0,
              note,
            });
          });
        }
      });
    }
  });
}

if (GPU_AVAILABLE) references.warnMissing(CASE_IDS);
// A missing row of this environment is not a silent green (ADR 0005's rule).
registerReferenceGate(references, { runnable: GPU_AVAILABLE, caseIds: CASE_IDS });

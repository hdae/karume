/**
 * The fused-vs-unfused identity check of one graph: two open Sessions — the side that fuses and the
 * plain side that does not — run the same feeds, and their outputs are compared as u32 words (class
 * E of ADR 0147 decision 2: u32 equality with the unfused plan, NaN read by a declared policy). The
 * diagnostics show that each side ran the plan it stands for: the fused side's fusion counter shows
 * the expected hits and the plain side's shows none, and, when given, each side's dispatches.
 *
 * The caller builds both Sessions, in whichever way gives the plain plan of the same values: a twin
 * graph with a zero-dispatch reshape interposed (the exact matcher stops matching), an opt-in present
 * on one side only, a forced split, or the test-only `DISABLED_FUSION_RULES` key on the plain side
 * (which takes an external rule's name on a Session that was given the rule).
 *
 * MUST: the plain side's counter is a criterion of its own. Two sides that both fuse would compare a
 * kernel with itself and pass with nothing checked.
 */

import type { GpuContext } from "../../runtime/src/gpu/device.ts";
import type {
  FusionCounts,
  RunInputs,
  RunOutputs,
  Session,
} from "../../runtime/src/runtime/executor.ts";
import type { NanPolicy } from "../../runtime/src/tune/word-identity.ts";
import {
  type CheckCase,
  compareCount,
  compareWords,
  compareWordsNanClass,
  type Criterion,
  scopedGpuWork,
  wordsOf,
} from "./result.ts";

/** What {@link checkFusionTwin} compares. */
export type FusionTwin = {
  /** The case's label. */
  readonly label: string;
  /** The side that fuses. */
  readonly fused: Session;
  /** The side that runs the same values unfused. */
  readonly plain: Session;
  /** The inputs both sides run. */
  readonly feeds: RunInputs;
  /**
   * The counter of `diagnostics().lastRunFusions` that shows the fusion — any key of the record, a
   * rule's or an annotation counter. An external rule (`SessionOptions.fusionRules`) counts under its
   * `<namespace>/<rule>` name; give both sides the rule and disable it on the plain side by that name.
   */
  readonly counter: keyof FusionCounts;
  /** The count the fused side's counter MUST show over its run. */
  readonly hits: number;
  /** How a NaN word compares (`"payload"` for words a kernel only moves). */
  readonly nan: NanPolicy;
  /**
   * The outputs to compare. Omitted: every output, which MUST be the same set on both sides. A name
   * that either side does not return throws.
   */
  readonly outputs?: readonly string[];
  /**
   * The dispatches each given side MUST record over its run (increments of
   * `diagnostics().submit.dispatchCount`). An omitted side is not compared.
   */
  readonly dispatches?: { readonly fused?: number; readonly plain?: number };
};

type SideRun = {
  readonly outputs: RunOutputs;
  readonly dispatches: number;
  readonly hits: number | undefined;
};

const runSide = async (
  gpu: GpuContext,
  session: Session,
  twin: FusionTwin,
  side: "fused" | "plain",
): Promise<SideRun> => {
  const before = session.diagnostics().submit.dispatchCount;
  const outputs = await scopedGpuWork(gpu, `${twin.label}: ${side} run`, () =>
    session.run(twin.feeds),
  );
  const diagnostics = session.diagnostics();
  return {
    outputs,
    dispatches: diagnostics.submit.dispatchCount - before,
    hits: diagnostics.lastRunFusions?.[twin.counter],
  };
};

/** The compared output names; a misnamed or mismatched output set is the caller's error. */
const comparedOutputs = (twin: FusionTwin, fused: RunOutputs, plain: RunOutputs): string[] => {
  if (twin.outputs === undefined) {
    const names = Object.keys(fused);
    const plainNames = Object.keys(plain);
    if (names.length !== plainNames.length || names.some((name) => !Object.hasOwn(plain, name))) {
      throw new Error(
        `${twin.label}: the two sides return different outputs (fused ${names.join(", ")}; plain` +
          ` ${plainNames.join(", ")}) — name the outputs to compare`,
      );
    }
    return names;
  }
  const missing = twin.outputs.filter(
    (name) => !Object.hasOwn(fused, name) || !Object.hasOwn(plain, name),
  );
  if (missing.length > 0) {
    throw new Error(`${twin.label}: output(s) ${missing.join(", ")} not returned by both sides`);
  }
  return [...twin.outputs];
};

/**
 * Runs the fused side, then the plain side, on `twin.feeds` (each under the error scopes of
 * {@link scopedGpuWork}) and returns one case: a word-identity criterion per compared output (the
 * plain side's words are the expected ones), the fused side's counter against `twin.hits`, the plain
 * side's counter against 0, and the dispatches of each side given in `twin.dispatches`.
 */
export const checkFusionTwin = async (gpu: GpuContext, twin: FusionTwin): Promise<CheckCase> => {
  const fused = await runSide(gpu, twin.fused, twin, "fused");
  const plain = await runSide(gpu, twin.plain, twin, "plain");
  const compare = twin.nan === "class" ? compareWordsNanClass : compareWords;
  const criteria: Criterion[] = comparedOutputs(twin, fused.outputs, plain.outputs).map((name) =>
    compare(
      `output ${name} (NaN by ${twin.nan})`,
      wordsOf(fused.outputs[name].data),
      wordsOf(plain.outputs[name].data),
    ),
  );
  criteria.push(
    compareCount(`fused ${twin.counter}`, fused.hits, twin.hits),
    compareCount(`plain ${twin.counter}`, plain.hits, 0),
  );
  const expected = twin.dispatches;
  if (expected?.fused !== undefined) {
    criteria.push(compareCount("fused dispatches", fused.dispatches, expected.fused));
  }
  if (expected?.plain !== undefined) {
    criteria.push(compareCount("plain dispatches", plain.dispatches, expected.plain));
  }
  return { label: twin.label, criteria };
};

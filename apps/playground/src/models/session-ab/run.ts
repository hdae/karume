/**
 * The session A/B page's arms and the order the runner visits them in.
 *
 * An arm (`SessionAbArm` of `@karume/runtime/tune`) here is one Gemma 4 pipeline configuration:
 * open builds the pipeline with the arm's options (the candidate rule for the `rule` arm), a run is
 * one generation of the request's tokens from the fixed prompt with the fixed sampler — its output
 * is the generated token ids, so the A/B's identity is token identity — the report gives the counts
 * of the run's first one-row run (when the request reads them), and close disposes the pipeline.
 *
 * {@link runPositions} is the runner's visiting order as a list, so the page can say where a run is
 * (`runSessionAb` reports no progress of its own); `vp test` drives the real runner with fake arms
 * against it.
 */

import { chatPrompt, type RunPhase } from "@karume/models/gemma";
import type { SessionDiagnostics, SessionOptions } from "@karume/runtime";
import type { SessionAbArm, SessionAbReport, SessionAbRun } from "@karume/runtime/tune";

import type { GemmaPipeline } from "@/models/gemma4/use-gemma-pipeline";

import type { SessionAbRequest } from "./request";

/** Where one run sits in the runner's order. */
export type RunPosition = {
  /** 0 = the untimed warm-up round, 1 to `rounds` the timed rounds. */
  readonly round: number;
  /** 1 = the forward pass over the arms, 2 = the reverse pass. */
  readonly visit: 1 | 2;
  readonly arm: number;
  /** 1 or, with `openPerVisit`, 2 (the fresh pipeline's untimed first run comes first). */
  readonly run: 1 | 2;
  /** Whether the runner times this run. */
  readonly timed: boolean;
};

/**
 * Every run in the runner's order: per round the arms forward then in reverse (A B B A), one run
 * per visit, or two with `openPerVisit` (an untimed one on the fresh pipeline, then the timed one).
 */
export const runPositions = (
  arms: number,
  rounds: number,
  openPerVisit: boolean,
): readonly RunPosition[] => {
  const order = Array.from({ length: arms }, (_, arm) => arm);
  const runs: (1 | 2)[] = openPerVisit ? [1, 2] : [1];
  const positions: RunPosition[] = [];
  for (let round = 0; round <= rounds; round += 1) {
    for (const [visit, arm] of [
      ...order.map((arm) => [1, arm] as const),
      ...order.toReversed().map((arm) => [2, arm] as const),
    ]) {
      for (const run of runs) {
        positions.push({ round, visit, arm, run, timed: round > 0 && run === runs.length });
      }
    }
  }
  return positions;
};

/**
 * What one arm did, as the page shows it: `open` before a pipeline is built, `ran` after a run
 * returned (from the report, outside the timed span), `close` before a pipeline is disposed.
 */
export type ArmStep = { readonly arm: string; readonly step: "open" | "ran" | "close" };

/** The diagnostics callback a pipeline is built with. */
export type DiagnosticsCallback = (diagnostics: SessionDiagnostics, phase: RunPhase) => void;

/** The counts of a run, read from its first one-row run (decode, or a draft under self-speculation). */
const countsOf = (diagnostics: SessionDiagnostics): SessionAbReport => {
  const fusions: Record<string, number> = {};
  for (const [name, count] of Object.entries(diagnostics.lastRunFusions ?? {})) {
    if (count !== undefined) fusions[name] = count;
  }
  return {
    fusions,
    dispatches: (diagnostics.lastRunPipelines ?? []).reduce(
      (sum, { dispatchCount }) => sum + dispatchCount,
      0,
    ),
  };
};

/**
 * One arm on a Gemma 4 pipeline. `build` loads the pipeline with the arm's rules and, when the
 * request reads counts, the diagnostics callback; `afterClose` runs after the pipeline is disposed
 * (the page lets the device free the pipeline's memory there). The signal is checked before every
 * open and passed to every generation. The report is always there, so the runner calls it after
 * every run, outside the timed span: it says the run ended (`onStep`) and gives the counts, or none
 * when the request does not read them (the record then has no counts). Nothing runs inside the
 * timed span but the generation.
 */
export const pipelineArm = (input: {
  readonly name: string;
  readonly fusionRules?: SessionOptions["fusionRules"];
  readonly request: Pick<
    SessionAbRequest,
    "prompt" | "tokens" | "sampler" | "capacity" | "readCounts"
  >;
  readonly build: (options: {
    readonly fusionRules?: SessionOptions["fusionRules"];
    readonly onRunDiagnostics?: DiagnosticsCallback;
  }) => Promise<GemmaPipeline>;
  readonly afterClose: () => Promise<void>;
  readonly signal: AbortSignal;
  readonly onStep: (step: ArmStep) => void;
}): SessionAbArm => {
  const { name, fusionRules, request, signal, onStep } = input;
  let pipeline: GemmaPipeline | undefined;
  let counts: SessionAbReport | undefined;
  const onRunDiagnostics: DiagnosticsCallback = (diagnostics, phase) => {
    if (counts !== undefined || (phase.kind !== "decode" && phase.kind !== "draft")) return;
    counts = countsOf(diagnostics);
  };
  const opened = (): GemmaPipeline => {
    if (pipeline === undefined) throw new Error(`Arm ${name}: used before open`);
    return pipeline;
  };
  const run = async (): Promise<SessionAbRun> => {
    counts = undefined;
    const current = opened();
    const sequence = await current.sequence({ capacity: request.capacity });
    try {
      const prompt = chatPrompt(current.tokenizer, [{ role: "user", content: request.prompt }]);
      const ids: number[] = [];
      const stream = sequence.generate({
        prompt,
        maxNewTokens: request.tokens,
        sampler: request.sampler,
        signal,
      });
      for await (const event of stream) if (event.kind === "token") ids.push(event.id);
      await stream.done;
      return { outputs: { tokens: Int32Array.from(ids) } };
    } finally {
      await sequence.dispose();
    }
  };
  return {
    name,
    open: async () => {
      signal.throwIfAborted();
      onStep({ arm: name, step: "open" });
      pipeline = await input.build({
        ...(fusionRules === undefined ? {} : { fusionRules }),
        ...(request.readCounts ? { onRunDiagnostics } : {}),
      });
    },
    run,
    report: () => {
      onStep({ arm: name, step: "ran" });
      if (!request.readCounts) return Promise.resolve({});
      return counts === undefined
        ? Promise.reject(new Error(`Arm ${name}: the run had no one-row run to count`))
        : Promise.resolve(counts);
    },
    close: async () => {
      onStep({ arm: name, step: "close" });
      const current = pipeline;
      pipeline = undefined;
      await current?.dispose();
      await input.afterClose();
    },
  };
};

/**
 * One empty submit and its fence, run after an arm's pipeline is disposed. Dawn frees the memory of
 * destroyed buffers only once a submit made after the destroy has completed, so on the page's device
 * a disposed pipeline stays allocated until the next submit, and the next visit's build of a large
 * model fails out of memory before its first submit (observed 2026-11-11 with Gemma 4 26B under
 * Dawn on Vulkan). Outside the timed span; the pipeline itself is not changed.
 */
export const releaseFence = async (device: GPUDevice): Promise<void> => {
  device.queue.submit([device.createCommandEncoder().finish()]);
  await device.queue.onSubmittedWorkDone();
};

/**
 * The arms of an EmbeddingGemma preset A/B and the probe before it. An arm is one preset's
 * pipeline, built when the runner opens it and kept until it closes; a run is one `embed` of the
 * cut text, nothing else (no signal, so the pipeline takes no turn of the event loop before the
 * run; no diagnostics callback, so the run is the production path). Outside the timed span the
 * arm's report compares the run's vector with the arm's own first one, so the record says whether
 * each preset gave the same bits every time — the runner's identity holds against the base's first
 * vector, which the other preset differs from by design.
 *
 * The probe is a pipeline of its own, built with the diagnostics callback, timed while it loads and
 * while it runs the text once, then disposed: what the diagnostics say about the load, the resident
 * weights and the plan backing. Its pipeline is a narrow stand-in type, so `vp test` drives both
 * with fakes.
 */

import type { SessionDiagnostics } from "@karume/runtime";
import type { SessionAbArm } from "@karume/runtime/tune";

import type { CutText } from "./text.ts";

/** The part of `embeddinggemma.Pipeline` the arms and the probe use. */
export type EmbeddingPipeline = {
  embed(text: string, options?: { readonly prompt?: string }): Promise<Float32Array<ArrayBuffer>>;
  tokenize(text: string, options?: { readonly prompt?: string }): Int32Array<ArrayBuffer>;
  dispose(): Promise<void>;
};

/** The name of the one output an arm returns to the runner. */
export const EMBEDDING_OUTPUT = "embedding";

/** The prompt option of a cut text (none without a prompt). */
export const promptOptions = (text: Pick<CutText, "prompt">): { readonly prompt?: string } =>
  text.prompt === undefined ? {} : { prompt: text.prompt };

/** What one arm learned while the runner drove it. */
export type PresetArmFacts = {
  /** The wall time of each open (building the pipeline: `fromPretrained`), in order. */
  readonly openMs: readonly number[];
  /** The runs compared with the arm's own first vector (the first included). */
  readonly runs: number;
  /** The runs whose vector differs from the first in at least one word. */
  readonly differingRuns: number;
  /** The arm's first vector, the one the others are compared with. */
  readonly first?: Float32Array<ArrayBuffer>;
};

/** An arm of the A/B and what it learned so far. */
export type PresetArm = SessionAbArm & { readonly facts: () => PresetArmFacts };

/** Whether two vectors hold the same 32-bit words (NaN compared by its bits). */
const sameWords = (a: Float32Array<ArrayBuffer>, b: Float32Array<ArrayBuffer>): boolean => {
  if (a.length !== b.length) return false;
  const left = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const right = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return left.every((word, index) => word === right[index]);
};

/**
 * One preset's arm: `open` builds the pipeline (`build`, timed by `clock`), `run` embeds the text,
 * `report` checks the vector against the arm's first and then `signal` (a cancel takes effect
 * between two runs, outside the timed span), `close` disposes the pipeline.
 */
export const presetArm = (input: {
  readonly name: string;
  readonly text: CutText;
  readonly build: () => Promise<EmbeddingPipeline>;
  readonly signal: AbortSignal;
  readonly clock?: () => number;
}): PresetArm => {
  const { name, text, build, signal, clock = () => performance.now() } = input;
  const options = promptOptions(text);
  const openMs: number[] = [];
  let pipeline: EmbeddingPipeline | undefined;
  let last: Float32Array<ArrayBuffer> | undefined;
  let first: Float32Array<ArrayBuffer> | undefined;
  let runs = 0;
  let differingRuns = 0;
  return {
    name,
    open: async () => {
      signal.throwIfAborted();
      const start = clock();
      pipeline = await build();
      openMs.push(clock() - start);
    },
    run: async () => {
      if (pipeline === undefined) throw new Error(`The arm ${name} runs before it is open`);
      last = await pipeline.embed(text.text, options);
      return { outputs: { [EMBEDDING_OUTPUT]: last } };
    },
    report: async () => {
      if (last === undefined) throw new Error(`The arm ${name} reports before a run`);
      runs += 1;
      if (first === undefined) first = last.slice();
      else if (!sameWords(last, first)) differingRuns += 1;
      signal.throwIfAborted();
      return {};
    },
    close: async () => {
      const open = pipeline;
      pipeline = undefined;
      await open?.dispose();
    },
    facts: () => ({
      openMs: [...openMs],
      runs,
      differingRuns,
      ...(first === undefined ? {} : { first }),
    }),
  };
};

/**
 * What a probe of one preset read: the times around it and the diagnostics of its second run (the
 * runtime builds a plan backing on a run that finds its prepared plan, so the first run has none).
 */
export type PresetProbe = {
  readonly preset: string;
  /** The wall time of `fromPretrained`: the manifest, part 0, admission, reading the weight parts and uploading them. */
  readonly loadMs: number;
  /** The wall time of the first `embed` (the plan is derived and the pipelines are made there). */
  readonly firstRunMs: number;
  /** The wall time of the second `embed` (the prepared plan, and the plan backing built). */
  readonly secondRunMs: number;
  /** How the Session's build spent its time (`SessionBuildStats`, host clock). */
  readonly build: SessionDiagnostics["buildStats"];
  /** The weight arena (it also holds the params cache's buffers): `allocatedBytes` is what it made. */
  readonly weights: SessionDiagnostics["weights"];
  /** The low-precision weights resident as they are stored, and those decoded to f32 on the host. */
  readonly storage: SessionDiagnostics["storage"];
  /** The plan backing after the second run (the intermediates of the text's length). */
  readonly planBacking: SessionDiagnostics["planBacking"];
  /** The dispatches the second run planned (every pipeline key's count summed). */
  readonly dispatches: number;
  readonly geometryProfile: string;
};

/** The facts of one probe from its three times and the diagnostics of its second run. */
export const presetProbe = (input: {
  readonly preset: string;
  readonly loadMs: number;
  readonly firstRunMs: number;
  readonly secondRunMs: number;
  readonly diagnostics: SessionDiagnostics;
}): PresetProbe => {
  const { diagnostics } = input;
  const pipelines = diagnostics.lastRunPipelines;
  if (pipelines === undefined) {
    throw new Error(`The probe of ${input.preset} read diagnostics without a planned run`);
  }
  return {
    preset: input.preset,
    loadMs: input.loadMs,
    firstRunMs: input.firstRunMs,
    secondRunMs: input.secondRunMs,
    build: { ...diagnostics.buildStats },
    weights: { ...diagnostics.weights },
    storage: { ...diagnostics.storage },
    planBacking: { ...diagnostics.planBacking },
    dispatches: pipelines.reduce((total, { dispatchCount }) => total + dispatchCount, 0),
    geometryProfile: diagnostics.geometryProfile,
  };
};

/**
 * Probes one preset: builds its pipeline with `build`, which hands it the diagnostics sink, asks
 * `textOf` for the text on that pipeline (untimed: the cut tokenizes on the host), embeds it twice,
 * reads what the second run's diagnostics say, and disposes it. Times by `clock`.
 */
export const probePreset = async (input: {
  readonly preset: string;
  readonly build: (
    onRunDiagnostics: (diagnostics: SessionDiagnostics) => void,
  ) => Promise<EmbeddingPipeline>;
  readonly textOf: (pipeline: EmbeddingPipeline) => CutText;
  readonly clock?: () => number;
}): Promise<{ readonly probe: PresetProbe; readonly text: CutText }> => {
  const { preset, build, textOf, clock = () => performance.now() } = input;
  const seen: SessionDiagnostics[] = [];
  const loadStart = clock();
  const pipeline = await build((diagnostics) => {
    seen.push(diagnostics);
  });
  try {
    const loadMs = clock() - loadStart;
    const text = textOf(pipeline);
    const timedEmbed = async (): Promise<number> => {
      const start = clock();
      await pipeline.embed(text.text, promptOptions(text));
      return clock() - start;
    };
    const firstRunMs = await timedEmbed();
    const secondRunMs = await timedEmbed();
    const [, second] = seen;
    if (seen.length !== 2 || second === undefined) {
      throw new Error(
        `The probe of ${preset} received ${seen.length} diagnostics for its two runs`,
      );
    }
    return {
      probe: presetProbe({ preset, loadMs, firstRunMs, secondRunMs, diagnostics: second }),
      text,
    };
  } finally {
    await pipeline.dispose();
  }
};

/**
 * The request, the `data` and the summary of an EmbeddingGemma preset A/B (`embeddinggemma-preset-ab`):
 * two presets of one mirror timed against each other on one text through the public pipeline. The
 * runtime's `karume-session-ab/1` record is kept as it is, with the judgement of
 * `judgeSessionAb` without identity (the presets store the weights differently, so their vectors
 * differ by design), each arm's median embed time, the host's GCs inside each of its timed visits,
 * each preset's probe and what the arms learned. Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";
import type { SessionAbJudgement, SessionAbRecord } from "@karume/runtime/tune";

import { type DistributionFacts, type EntryResult, ParameterError } from "../entry.ts";
import { modelPreset } from "../model-preset.ts";
import { MODELS_VERSION } from "../session-ab/data.ts";
import { runPositions } from "../session-ab/shared.ts";
import { judgementLine } from "../session-ab/summary.ts";

import type { PresetArmFacts, PresetProbe, RunSpan } from "./arm.ts";
import { type GcEntry, type GcReading, type VisitGc, visitGc } from "./gc.ts";
import { type CutText, LONGEST_CASE_TOKENS } from "./text.ts";

/** The format of the entry's `data`. */
export const PRESET_AB_FORMAT = "karume-embeddinggemma-preset-ab/2";

/** The mount the entry reads: the local mirror of EmbeddingGemma 2. */
export const PRESET_AB_MOUNT = "embeddinggemma-2";

/** What one run measures. */
export type PresetAbRequest = {
  readonly mount: typeof PRESET_AB_MOUNT;
  /** The manifest's default model (the family has one). */
  readonly model: string;
  /** The base arm's preset (the judge's base). */
  readonly base: string;
  /** The compared arm's preset. */
  readonly compared: string;
  /** The text's length in token ids, `<bos>` and `<eos>` included. */
  readonly tokens: number;
  readonly rounds: number;
  readonly minGain: number;
};

/**
 * The request of one run: `base` (empty: the model's default preset) and `compared` must be two
 * different presets of the manifest's default model, and `tokens` at most the longest golden case.
 * Anything else is a {@link ParameterError} that names the parameter.
 */
export const presetAbRequest = (
  manifest: Manifest,
  values: {
    readonly base: string;
    readonly compared: string;
    readonly tokens: number;
    readonly rounds: number;
    readonly minGain: number;
  },
): PresetAbRequest => {
  const model = manifest.defaultModel;
  const presetNamed = (parameter: string, chosen: string | undefined): string => {
    const preset = modelPreset(manifest, model, chosen);
    if (preset.ok) return preset.name;
    throw new ParameterError(
      parameter,
      `'${preset.chosen}' is not a preset of model '${model}'; its presets are ${preset.presets.join(", ")}`,
    );
  };
  const base = presetNamed("base", values.base === "" ? undefined : values.base);
  if (values.compared === "") {
    throw new ParameterError("compared", "names no preset (the compared arm needs one)");
  }
  const compared = presetNamed("compared", values.compared);
  if (compared === base) {
    throw new ParameterError("compared", `'${compared}' is the base preset too`);
  }
  if (values.tokens > LONGEST_CASE_TOKENS) {
    throw new ParameterError(
      "tokens",
      `${values.tokens} is longer than the longest golden case (${LONGEST_CASE_TOKENS} tokens)`,
    );
  }
  return {
    mount: PRESET_AB_MOUNT,
    model,
    base,
    compared,
    tokens: values.tokens,
    rounds: values.rounds,
    minGain: values.minGain,
  };
};

const median = (values: readonly number[]): number => {
  const sorted = values.toSorted((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

/** The median over the rounds of the mean of the round's two visits (the judge's `baseMedianMs` for the base). */
export const medianRoundMs = (times: readonly (readonly [number, number])[]): number =>
  median(times.map(([first, second]) => (first + second) / 2));

/**
 * The fastest timed visit, kept beside the median: a median alone can land inside a stretch of
 * slow runs (what the development machine showed is in
 * docs/research/2026-11-22-embeddinggemma2-quant-screen.md).
 */
export const minimumMs = (times: readonly (readonly [number, number])[]): number =>
  Math.min(...times.flat());

/** The cosine of two vectors of one width. */
export const cosine = (a: Float32Array<ArrayBuffer>, b: Float32Array<ArrayBuffer>): number => {
  if (a.length !== b.length)
    throw new Error(`The vectors' widths differ (${a.length}, ${b.length})`);
  let dot = 0;
  let left = 0;
  let right = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
    left += a[index] * a[index];
    right += b[index] * b[index];
  }
  return dot / Math.sqrt(left * right);
};

/**
 * The host's GCs in an arm's timed visits: `visits[r][v]` is round `r + 1`, visit `v + 1`, as the
 * record's `timesMs`; or that the host cannot report GCs.
 */
export type ArmGc =
  | { readonly observable: false }
  | { readonly observable: true; readonly visits: readonly (readonly [VisitGc, VisitGc])[] };

/**
 * The GCs of each timed visit of the arm at `index` of `record`, from the arm's run spans (every
 * run in the runner's order, the untimed ones included). A span count that is not the runner's
 * runs of that arm is an error.
 */
export const armGc = (input: {
  readonly name: string;
  readonly index: number;
  readonly spans: readonly RunSpan[];
  readonly record: Pick<SessionAbRecord, "rounds" | "openPerVisit" | "arms">;
  readonly entries: readonly GcEntry[];
}): readonly (readonly [VisitGc, VisitGc])[] => {
  const { name, index, spans, record, entries } = input;
  const positions = runPositions(record.arms.length, record.rounds, record.openPerVisit).filter(
    ({ arm }) => arm === index,
  );
  if (spans.length !== positions.length) {
    throw new Error(
      `The arm ${name} noted ${spans.length} runs; the record's ${record.rounds} rounds and the warm-up round make ${positions.length}`,
    );
  }
  const timed = positions.flatMap((position, run) =>
    position.timed ? [{ ...position, gc: visitGc(spans[run], entries) }] : [],
  );
  return Array.from({ length: record.rounds }, (_, round) => {
    const of = (visit: 1 | 2): VisitGc => {
      const found = timed.find((run) => run.round === round + 1 && run.visit === visit);
      if (found === undefined) {
        throw new Error(`The arm ${name} has no timed run in round ${round + 1}, visit ${visit}`);
      }
      return found.gc;
    };
    return [of(1), of(2)] as const;
  });
};

/** One arm in the data: its name, its median embed time, the host's GCs and what it learned. */
export type PresetAbArm = {
  readonly name: string;
  /** {@link medianRoundMs} of the arm's timed runs. */
  readonly medianMs: number;
  /** {@link minimumMs} of the arm's timed runs. */
  readonly minMs: number;
  /** The host's GCs that began inside each timed visit's `embed`. */
  readonly gc: ArmGc;
  /** Each open's wall time (`fromPretrained`); the probe of the preset had read its files once before. */
  readonly openMs: readonly number[];
  /** The runs compared with the arm's own first vector, and how many differed from it. */
  readonly runs: number;
  readonly differingRuns: number;
};

/** The `data` of a run. */
export type PresetAbData = {
  readonly format: typeof PRESET_AB_FORMAT;
  readonly request: PresetAbRequest;
  readonly distribution: DistributionFacts;
  /** The text without its body (the golden case and the code points reproduce it). */
  readonly text: Omit<CutText, "text">;
  /** The probes, base first. */
  readonly probes: readonly [PresetProbe, PresetProbe];
  /** The arms, base first. */
  readonly arms: readonly [PresetAbArm, PresetAbArm];
  /** The cosine of the compared arm's first vector against the base's. */
  readonly cosine: number;
  readonly modelsVersion: string;
  /** The runtime's record (`karume-session-ab/1`), unchanged. */
  readonly record: SessionAbRecord;
  /** `judgeSessionAb(record, { minGain, requireIdentity: false })` of the compared arm. */
  readonly judgement: SessionAbJudgement;
};

/**
 * The data of one finished run. Refuses a record whose arms or round count are not the request's,
 * probes not of the request's presets, an arm without a first vector, and an arm whose run spans
 * are not the record's runs.
 */
export const presetAbData = (input: {
  readonly request: PresetAbRequest;
  readonly distribution: DistributionFacts;
  readonly text: CutText;
  readonly probes: readonly [PresetProbe, PresetProbe];
  readonly arms: readonly [PresetArmFacts, PresetArmFacts];
  readonly record: SessionAbRecord;
  readonly judgements: readonly SessionAbJudgement[];
  /** What the GC observer saw over the whole A/B. */
  readonly gc: GcReading;
}): PresetAbData => {
  const { request, record, probes, arms, gc } = input;
  const names = [request.base, request.compared];
  const recorded = record.arms.map(({ name }) => name);
  if (recorded.join(",") !== names.join(",") || record.rounds !== request.rounds) {
    throw new Error(
      `The A/B record (${recorded.join(", ")}; ${record.rounds} rounds) is not the request's (${names.join(", ")}; ${request.rounds} rounds)`,
    );
  }
  const probed = probes.map(({ preset }) => preset);
  if (probed.join(",") !== names.join(",")) {
    throw new Error(
      `The probes (${probed.join(", ")}) are not the request's presets (${names.join(", ")})`,
    );
  }
  const [judgement] = input.judgements;
  if (input.judgements.length !== 1 || judgement.arm !== request.compared) {
    throw new Error(`The judgements are not the one of ${request.compared}`);
  }
  const [baseFirst, comparedFirst] = arms.map((arm, index) => {
    if (arm.first === undefined) throw new Error(`The arm ${names[index]} has no first vector`);
    return arm.first;
  });
  const { text: _body, ...text } = input.text;
  const arm = (index: 0 | 1): PresetAbArm => ({
    name: names[index],
    medianMs: medianRoundMs(record.arms[index].timesMs),
    minMs: minimumMs(record.arms[index].timesMs),
    gc: gc.observable
      ? {
          observable: true,
          visits: armGc({
            name: names[index],
            index,
            spans: arms[index].runSpans,
            record,
            entries: gc.entries,
          }),
        }
      : { observable: false },
    openMs: arms[index].openMs,
    runs: arms[index].runs,
    differingRuns: arms[index].differingRuns,
  });
  return {
    format: PRESET_AB_FORMAT,
    request,
    distribution: input.distribution,
    text,
    probes,
    arms: [arm(0), arm(1)],
    cosine: cosine(baseFirst, comparedFirst),
    modelsVersion: MODELS_VERSION,
    record,
    judgement,
  };
};

const MIB = 2 ** 20;
const mib = (bytes: number): string => `${(bytes / MIB).toFixed(1)} MiB`;
const ms = (value: number): string => `${value.toFixed(1)} ms`;

const probeLine = (probe: PresetProbe): string => {
  const { build } = probe;
  return (
    `${probe.preset}: load ${ms(probe.loadMs)} (supply wait ${ms(build.supplyWaitMs)}, decode ${ms(build.decodeMs)},` +
    ` buffer create ${ms(build.bufferCreateMs)}, write issue ${ms(build.writeBufferIssueMs)},` +
    ` upload fence ${ms(build.uploadFenceMs)}; ${mib(build.uploadedBytes)} uploaded) · weights` +
    ` ${mib(probe.weights.allocatedBytes)} (low-precision as stored ${mib(probe.storage.residentCompressedBytes)},` +
    ` decoded on the host ${mib(probe.storage.hostExpandedBytes)}) · plan backing` +
    ` ${mib(probe.planBacking.residentBytes)} · ${probe.dispatches} dispatches · first run ${ms(probe.firstRunMs)}, second ${ms(probe.secondRunMs)}`
  );
};

/** An arm's timed visits that held a major GC, and the range of those visits' summed pauses. */
const majorGcText = (
  arm: PresetAbArm,
  visits: readonly (readonly [VisitGc, VisitGc])[],
): string => {
  const all = visits.flat();
  const pauses = all.filter(({ major }) => major > 0).map(({ majorPauseMs }) => majorPauseMs);
  const counted = `${arm.name} ${pauses.length} of ${all.length} visits`;
  if (pauses.length === 0) return counted;
  const low = Math.min(...pauses);
  const high = Math.max(...pauses);
  return `${counted} (${low === high ? ms(low) : `${ms(low)} to ${ms(high)}`} of pause each)`;
};

const gcLine = (arms: readonly PresetAbArm[]): string => {
  const texts = arms.map((arm) =>
    arm.gc.observable ? majorGcText(arm, arm.gc.visits) : undefined,
  );
  if (texts.some((text) => text === undefined)) {
    return "Timed visits holding a major GC of the host: not observable (the host reports no gc performance entries)";
  }
  return `Timed visits holding a major GC of the host: ${texts.join(" · ")}`;
};

const selfText = (arm: PresetAbArm): string =>
  arm.differingRuns === 0
    ? `${arm.name} identical (${arm.runs} runs)`
    : `${arm.name} ${arm.differingRuns} of ${arm.runs} runs differ`;

/** The result of a run: no verdict (it only measures), the summary lines and the data. */
export const presetAbResult = (data: PresetAbData): EntryResult => {
  const { request, text, arms, judgement } = data;
  const [base, compared] = arms;
  return {
    summary: [
      `${request.tokens} tokens: a prefix of ${text.case} (${text.codePoints} code points, prompt ${text.prompt ?? "none"}), ${request.rounds} rounds`,
      `Median embed (each round's two visits averaged): ${base.name} ${ms(base.medianMs)} · ${compared.name} ${ms(compared.medianMs)}; fastest visit ${base.name} ${ms(base.minMs)} · ${compared.name} ${ms(compared.minMs)}`,
      `${judgementLine(judgement)} · median difference ${ms(judgement.medianDeltaMs)} · noise floor ${ms(judgement.noiseFloorMs)}`,
      gcLine(arms),
      ...data.probes.map(probeLine),
      `Arm loads (fromPretrained after the probe read the files): ${arms.map((arm) => `${arm.name} ${arm.openMs.map(ms).join(", ")}`).join(" · ")}`,
      `Each arm against its own first vector: ${arms.map(selfText).join("; ")}`,
      `Cosine of ${compared.name} against ${base.name}: ${data.cosine.toFixed(7)}`,
    ],
    data,
  };
};

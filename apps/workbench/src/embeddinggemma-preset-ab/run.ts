/**
 * The run of an EmbeddingGemma preset A/B (`embeddinggemma-preset-ab`): the request read from the
 * mirror's manifest, the device acquired, each preset probed (the base's probe cuts the text, the
 * compared one checks that its pipeline reads the same ids), the two arms timed and judged while
 * an observer collects the host's GCs, the device released. The device and its pipelines come
 * through {@link AcquirePresetAbDevice} and the observer through an {@link ObserveGc}, so
 * `vp test` drives the whole run with fakes; the entry passes `acquireGpu` and
 * `embeddinggemma.Pipeline.fromPretrained`, and the observer is the global `PerformanceObserver`'s.
 */

import { loadManifest } from "@karume/hub";
import type { SessionDiagnostics } from "@karume/runtime";
import {
  judgeSessionAb,
  runSessionAb,
  type SessionAbAdapter,
  type SessionAbRecord,
} from "@karume/runtime/tune";

import {
  type EntryResult,
  type GpuOptions,
  integerValue,
  numberValue,
  type OpenedDistribution,
  type RunContext,
  textValue,
} from "../entry.ts";
import { armStepper } from "../session-ab/progress.ts";
import { runnerEnvironment, runPositions, steppedArm } from "../session-ab/shared.ts";

import {
  type EmbeddingPipeline,
  type PresetArm,
  presetArm,
  probePreset,
  promptOptions,
} from "./arm.ts";
import { PRESET_AB_MOUNT, presetAbData, presetAbRequest, presetAbResult } from "./data.ts";
import { type GcObserver, type GcReading, globalPerformanceObserver, observeGc } from "./gc.ts";
import { type CutText, cutText, GOLDEN_CASES, sourceCase } from "./text.ts";

/** The mirror has the direct route only (no published repository behind `/hf/`). */
const SOURCE = "direct";

/** The device the arms share, and how a pipeline of one preset is built on it. */
export type PresetAbDevice = {
  readonly adapterInfo: SessionAbAdapter;
  /** `embeddinggemma.Pipeline.fromPretrained` of one preset on this device. */
  readonly pipeline: (
    ref: OpenedDistribution["ref"],
    options: {
      readonly model: string;
      readonly quant: string;
      readonly signal: AbortSignal;
      readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void;
    },
  ) => Promise<EmbeddingPipeline>;
  readonly destroy: () => void;
};

/** Acquires the device with the host's options (`acquireGpu` in the entry). */
export type AcquirePresetAbDevice = (options: GpuOptions) => Promise<PresetAbDevice>;

/** Starts observing the host's GCs ({@link observeGc} unless a test passes a fake). */
export type ObserveGc = () => GcObserver;

const sameIds = (a: Int32Array<ArrayBuffer>, b: Int32Array<ArrayBuffer>): boolean =>
  a.length === b.length && a.every((id, index) => id === b[index]);

/**
 * One run of the entry over `context`, the device from `acquire`; the host's GCs from `observe`,
 * made before the A/B's runner starts and disconnected when it is done.
 */
export const runPresetAb = async (
  context: RunContext,
  acquire: AcquirePresetAbDevice,
  observe: ObserveGc = () => observeGc(globalPerformanceObserver()),
): Promise<EntryResult> => {
  const { parameters, gpuOptions, signal, progress, openDistribution, backend } = context;
  const opened = await openDistribution(PRESET_AB_MOUNT, SOURCE);
  const loaded = await loadManifest(opened.ref, { signal });
  const request = presetAbRequest(loaded.manifest, {
    base: textValue(parameters, "base"),
    compared: textValue(parameters, "compared"),
    tokens: integerValue(parameters, "tokens"),
    rounds: integerValue(parameters, "rounds"),
    minGain: numberValue(parameters, "minGain"),
  });
  const source = sourceCase(GOLDEN_CASES, request.tokens);
  const device = await acquire({ ...gpuOptions });
  try {
    const pipelineOf = (
      quant: string,
      onRunDiagnostics?: (diagnostics: SessionDiagnostics) => void,
    ): Promise<EmbeddingPipeline> =>
      device.pipeline(opened.ref, {
        model: request.model,
        quant,
        signal,
        ...(onRunDiagnostics === undefined ? {} : { onRunDiagnostics }),
      });
    // The base's probe cuts the text; the compared one checks that its pipeline runs the same ids.
    const idsOf = (pipeline: EmbeddingPipeline, cut: CutText): Int32Array<ArrayBuffer> =>
      pipeline.tokenize(cut.text, promptOptions(cut));
    let baseIds = new Int32Array(0);
    progress(
      `probing ${request.base}: loading, cutting ${source.name} to ${request.tokens} tokens, two embeds`,
    );
    const baseProbe = await probePreset({
      preset: request.base,
      build: (sink) => pipelineOf(request.base, sink),
      textOf: (pipeline) => {
        const cut = cutText(
          source,
          request.tokens,
          (prefix) => pipeline.tokenize(prefix, promptOptions(source)).length,
        );
        baseIds = idsOf(pipeline, cut);
        return cut;
      },
    });
    const { text } = baseProbe;
    signal.throwIfAborted();
    progress(`probing ${request.compared}: loading, two embeds`);
    const comparedProbe = await probePreset({
      preset: request.compared,
      build: (sink) => pipelineOf(request.compared, sink),
      textOf: (pipeline) => {
        if (!sameIds(idsOf(pipeline, text), baseIds)) {
          throw new Error(
            `The ${request.compared} pipeline reads the text as other token ids than the ${request.base} pipeline`,
          );
        }
        return text;
      },
    });
    const onStep = armStepper({
      positions: runPositions(2, request.rounds, false),
      opens: "pipeline",
      signal,
      progress,
    });
    const arms: readonly [PresetArm, PresetArm] = [
      presetArm({ name: request.base, text, build: () => pipelineOf(request.base), signal }),
      presetArm({
        name: request.compared,
        text,
        build: () => pipelineOf(request.compared),
        signal,
      }),
    ];
    const observer = observe();
    let record: SessionAbRecord;
    let gc: GcReading;
    try {
      record = await runSessionAb({
        arms: arms.map((arm) => steppedArm(arm, onStep)),
        rounds: request.rounds,
        environment: runnerEnvironment(device.adapterInfo, backend),
      });
      gc = await observer.read();
    } finally {
      observer.disconnect();
    }
    const judgements = judgeSessionAb(record, {
      minGain: request.minGain,
      requireIdentity: false,
    });
    return presetAbResult(
      presetAbData({
        request,
        distribution: opened.facts,
        text,
        probes: [baseProbe.probe, comparedProbe.probe],
        arms: [arms[0].facts(), arms[1].facts()],
        record,
        judgements,
        gc,
      }),
    );
  } finally {
    device.destroy();
  }
};

/**
 * The Irodori-TTS page's form: the two distributions, the knobs, the numbers the page reads from the
 * manifest's `pipelineConfig`, and the pure step from the form to a generation request. The pipeline
 * validates every request itself (seed range, codec tile, reference length and rate — a
 * `ModelInputError`); the form only parses what it must turn into numbers.
 */

import type { LinearCompute, Manifest } from "@karume/hub";
import type { GenerateRequest } from "@karume/models/irodori";

import type { Choice } from "@/components/choice-select";

import { DEFAULT_CAPTION, DEFAULT_TEXT } from "./samples";

/** The two distributions (separate repos upstream, so separate mounts); v4.1 is the default. */
export type IrodoriMount = "irodori-v4.1" | "irodori-v4";

export const DEFAULT_MOUNT: IrodoriMount = "irodori-v4.1";

export const MODEL_CHOICES: readonly Choice<IrodoriMount>[] = [
  { value: "irodori-v4.1", label: "v4.1-small (default) — retrained duration predictor" },
  { value: "irodori-v4", label: "v4-small" },
];

/**
 * `linearCompute` of `fromPretrained` (the linears of `dit` / `dit_context` only): `preset` leaves it
 * to the preset's `session` declaration. `"f16"` is not offered: it needs a device acquired with
 * `shaderF16: true`, which this page does not request.
 */
export type LinearChoice = "preset" | "f32" | "a8";

export const LINEAR_CHOICES: readonly Choice<LinearChoice>[] = [
  { value: "preset", label: "The preset's declaration" },
  { value: "f32", label: "f32 — dequantized weights, f32 math" },
  { value: "a8", label: "a8 — per-token int8 activations, integer dot product (int weights only)" },
];

/** Voice design (a caption) or voice cloning (a reference audio) — exclusive, as upstream. */
export type VoiceMode = "caption" | "reference";

export type IrodoriForm = {
  readonly mount: IrodoriMount;
  /** A quant of the mount's manifest; "" until the manifest is known (then its default). */
  readonly quant: string;
  readonly linearCompute: LinearChoice;
  readonly text: string;
  readonly voice: VoiceMode;
  readonly caption: string;
  readonly seed: string;
  /** Empty = the duration predictor decides the length. */
  readonly durationSeconds: string;
  /** Empty = the pipeline's default tile. */
  readonly codecTileFrames: string;
  /**
   * Subscribe to the generation events: per-stage seconds, the step counter and Cancel, at the cost
   * of the host DiT loop. Off = no `onEvent`, so a generation cannot be observed or cancelled once it
   * runs; its DiT loop is resident (the fastest path) unless GPU timing forces the host loop.
   */
  readonly stageEvents: boolean;
};

export const initialForm = (): IrodoriForm => ({
  mount: DEFAULT_MOUNT,
  quant: "",
  linearCompute: "preset",
  text: DEFAULT_TEXT,
  voice: "caption",
  caption: DEFAULT_CAPTION,
  seed: "0",
  durationSeconds: "",
  codecTileFrames: "",
  stageEvents: true,
});

/**
 * The numbers of the model's `pipelineConfig` the page needs before a pipeline exists: the rate to
 * decode a reference audio to, the length bounds, the token limits and the fixed step count for the
 * hints. The pipeline owns the full schema; this reads only these fields and fails loudly when one
 * is missing.
 */
export type IrodoriConfigView = {
  readonly sampleRate: number;
  /** Tokens of the text, BOS included; the pipeline cuts the rest without an error (upstream's). */
  readonly maxTextLen: number;
  /** Tokens of the caption, BOS included; cut the same way. */
  readonly maxCaptionLen: number;
  readonly hopLength: number;
  readonly speakerPatchSize: number;
  readonly speakerRows: number;
  readonly codecHaloFrames: number;
  readonly minSeconds: number;
  readonly maxSeconds: number;
  readonly steps: number;
};

const CONFIG_FIELDS = [
  "sampleRate",
  "maxTextLen",
  "maxCaptionLen",
  "hopLength",
  "speakerPatchSize",
  "speakerRows",
  "codecHaloFrames",
  "minSeconds",
  "maxSeconds",
  "steps",
] as const satisfies readonly (keyof IrodoriConfigView)[];

export const readIrodoriConfig = (
  manifest: Manifest,
  model: string = manifest.defaultModel,
): IrodoriConfigView => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const config = entry.pipelineConfig;
  const read = (field: (typeof CONFIG_FIELDS)[number]): number => {
    const value = Object.hasOwn(config, field) ? config[field] : undefined;
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`pipelineConfig.${field} of model '${model}' is not a finite number`);
    }
    return value;
  };
  return {
    sampleRate: read("sampleRate"),
    maxTextLen: read("maxTextLen"),
    maxCaptionLen: read("maxCaptionLen"),
    hopLength: read("hopLength"),
    speakerPatchSize: read("speakerPatchSize"),
    speakerRows: read("speakerRows"),
    codecHaloFrames: read("codecHaloFrames"),
    minSeconds: read("minSeconds"),
    maxSeconds: read("maxSeconds"),
    steps: read("steps"),
  };
};

/** What a preset declares that the page shows next to the linear-compute choice. */
export type PresetFacts = {
  /** The `session.linearCompute` of the quant (undefined = the runtime's f32 path). */
  readonly declaredLinearCompute: LinearCompute | undefined;
  /**
   * The weights label of `dit` (`f32`, `i8`, `i4`…). A display label: the storage itself is the
   * container's codec, so the page uses it only for a hint, never to refuse a combination.
   */
  readonly ditWeights: string | undefined;
};

export const readPresetFacts = (
  manifest: Manifest,
  model: string = manifest.defaultModel,
): Readonly<Record<string, PresetFacts>> => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  return Object.fromEntries(
    Object.entries(entry.quants).map(([name, quant]) => [
      name,
      {
        declaredLinearCompute: quant.session.linearCompute,
        ditWeights: Object.hasOwn(quant.weights, "dit") ? quant.weights["dit"] : undefined,
      },
    ]),
  );
};

/**
 * The `linearCompute` the DiT Sessions are asked for: the page's override, else the preset's
 * declaration, else `f32` (the runtime's path without a declaration). Asked for, not applied: the
 * runtime runs `a8` only on int8 / int4 weights and sends float weights to the f32 path without an
 * error, so the record also counts the linear kernels that ran (`linearDispatchesOf`).
 */
export const resolveLinearCompute = (
  choice: LinearChoice,
  declared: LinearCompute | undefined,
): LinearCompute => (choice === "preset" ? (declared ?? "f32") : choice);

/**
 * A note when `a8` is asked for but the preset's DiT weights are labelled float: the runtime then
 * runs the f32 path, so the choice changes nothing. Undefined otherwise.
 */
export const linearComputeNote = (
  resolved: LinearCompute,
  ditWeights: string | undefined,
): string | undefined =>
  resolved === "a8" && (ditWeights === "f32" || ditWeights === "f16")
    ? `The preset's DiT weights are labelled '${ditWeights}': a8 runs only on int8 / int4 weights, so the DiT linears will run the f32 path (the runtime does not refuse it).`
    : undefined;

/**
 * The reference lengths the pipeline documents, for the hints: the minimum (two speaker tokens after
 * patching — `(2 × speakerPatchSize − 1) × hopLength + 1` samples) and the maximum it reads (the rest
 * is cut off — `(speakerRows − 1) × speakerPatchSize × hopLength` samples). The pipeline enforces
 * both; the page only shows them.
 */
export const referenceLimits = (
  config: IrodoriConfigView,
): { readonly minSamples: number; readonly maxSamples: number } => ({
  minSamples: (2 * config.speakerPatchSize - 1) * config.hopLength + 1,
  maxSamples: (config.speakerRows - 1) * config.speakerPatchSize * config.hopLength,
});

/** A form value the page cannot turn into a request (shown next to Run; nothing runs). */
export class FormError extends Error {
  override name = "FormError";
}

/** A decoded reference audio as the request takes it (mono f32 at the distribution's rate). */
export type ReferenceSamples = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
};

/** The request without `onEvent` (the page adds it when stage events are on). */
export type IrodoriRequestParts = Omit<GenerateRequest, "onEvent">;

const parseOptionalNumber = (text: string, what: string): number | undefined => {
  const trimmed = text.trim();
  if (trimmed === "") return undefined;
  const value = Number(trimmed);
  if (!Number.isFinite(value)) throw new FormError(`${what} '${text}' is not a number`);
  return value;
};

const parseOptionalInteger = (text: string, what: string): number | undefined => {
  const value = parseOptionalNumber(text, what);
  if (value !== undefined && !Number.isInteger(value)) {
    throw new FormError(`${what} '${text}' is not an integer`);
  }
  return value;
};

/**
 * The generation request of the form. Only the active voice tab is sent: a caption in caption mode
 * (an empty one = no voice design, the API's zero caption condition), the reference in reference
 * mode. The text is sent as typed (the pipeline normalizes it).
 */
export const buildRequest = (
  form: IrodoriForm,
  reference: ReferenceSamples | undefined,
): IrodoriRequestParts => {
  if (form.text.trim() === "") throw new FormError("Enter the text to speak");
  const seed = parseOptionalInteger(form.seed, "Seed");
  if (seed === undefined) throw new FormError("Enter a seed");
  const durationSeconds = parseOptionalNumber(form.durationSeconds, "Duration");
  const codecTileFrames = parseOptionalInteger(form.codecTileFrames, "Codec tile");
  let voice: Pick<GenerateRequest, "caption" | "speaker">;
  if (form.voice === "caption") {
    voice = form.caption.trim() === "" ? {} : { caption: form.caption };
  } else {
    if (reference === undefined) throw new FormError("Choose a reference audio file");
    voice = { speaker: { audio: { data: reference.data, sampleRate: reference.sampleRate } } };
  }
  return {
    text: form.text,
    ...voice,
    seed,
    ...(durationSeconds === undefined ? {} : { durationSeconds }),
    ...(codecTileFrames === undefined ? {} : { codecTileFrames }),
  };
};

/**
 * What a pipeline was built for. The page keeps the last pipeline between runs and reuses it while
 * every field is the same object or value (GPU and source are stable objects — a re-acquired GPU or a
 * new source is a new object).
 */
export type PipelineKey<Gpu, Source> = {
  readonly gpu: Gpu;
  readonly source: Source;
  readonly mount: IrodoriMount;
  readonly model: string;
  readonly quant: string;
  readonly linearCompute: LinearChoice;
};

const KEY_FIELDS = [
  "gpu",
  "source",
  "mount",
  "model",
  "quant",
  "linearCompute",
] as const satisfies readonly (keyof PipelineKey<unknown, unknown>)[];

/** The fields in which two keys differ, in {@link KEY_FIELDS} order (empty = the same key). */
export const changedKeyFields = <Gpu, Source>(
  left: PipelineKey<Gpu, Source>,
  right: PipelineKey<Gpu, Source>,
): readonly (keyof PipelineKey<Gpu, Source>)[] =>
  KEY_FIELDS.filter((field) => left[field] !== right[field]);

export const samePipelineKey = <Gpu, Source>(
  left: PipelineKey<Gpu, Source>,
  right: PipelineKey<Gpu, Source>,
): boolean => changedKeyFields(left, right).length === 0;

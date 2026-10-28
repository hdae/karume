/**
 * The Style-Bert-VITS2 page's form and the pure step from it to the arguments of
 * `Sbv2Pipeline.generate`. Acceptance belongs to the pipeline: the page refuses only what it cannot
 * turn into a request at all (empty text, a seed that is not a number) and passes every number on as
 * written, so an out-of-range value comes back as the pipeline's own `Sbv2InputError`.
 */

import type { Sbv2GenerateOptions, Sbv2Mora, Sbv2Utterance } from "@karume/models/sbv2";

import type { Sbv2Defaults, Sbv2ModelConfig } from "./config";

/** The knobs a slider sets (all of `Sbv2GenerateOptions` that are numbers, except the seed). */
export type Sbv2Knob = "styleWeight" | "sdpRatio" | "noiseScale" | "noiseScaleW" | "lengthScale";

export type Sbv2Form = {
  readonly text: string;
  readonly speaker: string;
  readonly style: string;
  readonly styleWeight: number;
  readonly sdpRatio: number;
  readonly noiseScale: number;
  readonly noiseScaleW: number;
  readonly lengthScale: number;
  /** As typed (parsed by {@link buildGenerateOptions}). */
  readonly seed: string;
};

/** The slider of each knob: label, range and what it does. The pipeline accepts any finite value. */
export const KNOBS: readonly {
  readonly key: Sbv2Knob;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly hint: string;
}[] = [
  {
    key: "styleWeight",
    label: "Style weight",
    min: 0,
    max: 2,
    step: 0.05,
    hint: "1 = the chosen style itself, 0 = the mean style, above 1 exaggerates it.",
  },
  {
    key: "lengthScale",
    label: "Length scale",
    min: 0.5,
    max: 2,
    step: 0.05,
    hint: "Duration scale: larger is slower (1 = the model's pace).",
  },
  {
    key: "sdpRatio",
    label: "SDP ratio",
    min: 0,
    max: 1,
    step: 0.05,
    hint: "Mix of the stochastic and the deterministic duration predictor (logw = sdp·r + dp·(1−r)).",
  },
  {
    key: "noiseScale",
    label: "Noise scale",
    min: 0,
    max: 1.5,
    step: 0.05,
    hint: "Noise of the latent z_p (expressiveness of the voice).",
  },
  {
    key: "noiseScaleW",
    label: "Noise scale W",
    min: 0,
    max: 1.5,
    step: 0.05,
    hint: "Noise of the stochastic duration predictor (variation of the timing).",
  },
];

/** The knobs and seed of the distribution's defaults (seed 0 = the pipeline's own default). */
export const defaultKnobs = (defaults: Sbv2Defaults): Omit<Sbv2Form, "text"> => ({
  speaker: defaults.speaker,
  style: defaults.style,
  styleWeight: defaults.styleWeight,
  sdpRatio: defaults.sdpRatio,
  noiseScale: defaults.noiseScale,
  noiseScaleW: defaults.noiseScaleW,
  lengthScale: defaults.lengthScale,
  seed: "0",
});

/**
 * The form after switching to a model with `config`: the speaker becomes that model's default (a
 * jvnv model holds its own speaker only), the style stays when the model has it, everything else is
 * kept.
 */
export const formForModel = (form: Sbv2Form, config: Sbv2ModelConfig): Sbv2Form => ({
  ...form,
  speaker: config.speakers.includes(form.speaker) ? form.speaker : config.defaults.speaker,
  style: config.styles.includes(form.style) ? form.style : config.defaults.style,
});

/** The request of one run: the text to analyse and the options of `generate`. */
export type Sbv2Request = {
  readonly text: string;
  readonly options: Required<Sbv2GenerateOptions>;
};

/**
 * Form → request. Throws a `RangeError` naming the field for what cannot become a request (the
 * pipeline's `Sbv2InputError` covers everything else, the seed's range included).
 */
export const buildRequest = (form: Sbv2Form): Sbv2Request => {
  if (form.text.trim() === "") throw new RangeError("The text is empty");
  const seedText = form.seed.trim();
  const seed = Number(seedText);
  if (seedText === "" || Number.isNaN(seed)) {
    throw new RangeError(`Seed must be a number (got '${form.seed}')`);
  }
  return {
    text: form.text,
    options: {
      speaker: form.speaker,
      style: form.style,
      styleWeight: form.styleWeight,
      sdpRatio: form.sdpRatio,
      noiseScale: form.noiseScale,
      noiseScaleW: form.noiseScaleW,
      lengthScale: form.lengthScale,
      seed,
    },
  };
};

/** A run of the reading: consecutive moras of one tone, or a punctuation mark (no tone). */
export type ReadingSegment =
  | { readonly kind: "moras"; readonly kana: string; readonly high: boolean }
  | { readonly kind: "punctuation"; readonly text: string };

const pushMora = (segments: ReadingSegment[], mora: Sbv2Mora): void => {
  const high = mora.tone === 1;
  const last = segments.at(-1);
  if (last?.kind === "moras" && last.high === high) {
    segments[segments.length - 1] = { kind: "moras", kana: last.kana + mora.kana, high };
  } else {
    segments.push({ kind: "moras", kana: mora.kana, high });
  }
};

/**
 * The reading of an utterance for display: moras grouped by tone (the page draws the high ones with
 * an overline — the usual pitch-accent notation) and the punctuation where the model reads it.
 */
export const readingSegments = (utterance: Sbv2Utterance): readonly ReadingSegment[] => {
  const segments: ReadingSegment[] = utterance.leadingPunctuations.map((text) => ({
    kind: "punctuation",
    text,
  }));
  for (const mora of utterance.moras) {
    pushMora(segments, mora);
    for (const text of mora.punctuations ?? []) segments.push({ kind: "punctuation", text });
  }
  return segments;
};

/**
 * The reading as one line of text, high moras in brackets (`ko[nnichiwa],` in kana) — the same information
 * as {@link readingSegments}, for the record and for comparing two runs' analyses at a glance.
 */
export const readingText = (utterance: Sbv2Utterance): string =>
  readingSegments(utterance)
    .map((segment) =>
      segment.kind === "punctuation"
        ? segment.text
        : segment.high
          ? `[${segment.kana}]`
          : segment.kana,
    )
    .join("");

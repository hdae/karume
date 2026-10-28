/**
 * The Wan2.2 page's form: choices, defaults, and the pure step from the form to a generation
 * request. The accepted values mirror `Ti2vPipeline` (1280×704 / 704×1280, 4n+1 frames from 5 to
 * 121, `vaeTile` 16 or 32); the pipeline still validates every request itself.
 */

import { type Manifest, resolveSelection } from "@karume/hub";
import type { Prompt, Ti2vGenerateRequest } from "@karume/models/wan";

export type TextEncoderPath = "precomputed" | "gpu";
export type SizeChoice = "1280x704" | "704x1280";
export type VaeTile = 16 | 32;

export const SIZES: readonly { readonly value: SizeChoice; readonly label: string }[] = [
  { value: "1280x704", label: "1280×704 (landscape)" },
  { value: "704x1280", label: "704×1280 (portrait)" },
];

/** 4n+1 from 5 to 121. */
export const FRAME_CHOICES: readonly number[] = Array.from({ length: 30 }, (_, n) => 4 * n + 5);

/** The manifest component of the GPU text encoder (a cross-repository reference to umT5). */
const TEXT_ENCODER = "text_encoder";

export type WanForm = {
  readonly quant: string;
  readonly vaeTile: VaeTile;
  readonly size: SizeChoice;
  readonly frames: number;
  readonly steps: string;
  readonly seed: string;
  /** Empty = the manifest's default. */
  readonly shift: string;
  readonly textEncoder: TextEncoderPath;
  /** `precomputed`: the name of a positive prompt row of the text embedding asset. */
  readonly promptName: string;
  /** `gpu`: free text. */
  readonly prompt: string;
  /** `gpu`: free text; empty = the official default negative prompt. */
  readonly negativePrompt: string;
};

export const initialForm = (defaultQuant: string): WanForm => ({
  quant: defaultQuant,
  vaeTile: 16,
  size: "1280x704",
  frames: 33,
  steps: "50",
  seed: "42",
  shift: "",
  textEncoder: "precomputed",
  promptName: "",
  prompt: "",
  negativePrompt: "",
});

/** The quick check: 2 steps at 17 frames (enough to run every stage, VAE tiles included). */
export const quickCheck = (form: WanForm): WanForm => ({ ...form, steps: "2", frames: 17 });

/** What the page needs from the manifest. */
export type WanManifestSummary = {
  readonly quants: readonly {
    readonly name: string;
    readonly label: string;
    readonly description: string;
  }[];
  readonly defaultQuant: string;
  readonly defaultShift: number | undefined;
  readonly defaultSteps: number | undefined;
  /** The repo the text encoder references (`crossRepo` key of the umT5 mount), if any. */
  readonly textEncoderRepo: string | undefined;
};

const numberAt = (value: unknown, ...keys: readonly string[]): number | undefined => {
  let current = value;
  for (const key of keys) {
    if (typeof current !== "object" || current === null || !Object.hasOwn(current, key)) {
      return undefined;
    }
    current = Reflect.get(current, key);
  }
  return typeof current === "number" ? current : undefined;
};

export const summarizeManifest = (manifest: Manifest): WanManifestSummary => {
  const model = manifest.models[manifest.defaultModel];
  if (model === undefined) {
    throw new Error(`defaultModel '${manifest.defaultModel}' is not in the manifest's models`);
  }
  return {
    quants: Object.entries(model.quants).map(([name, quant]) => ({
      name,
      label: quant.label ?? name,
      description: quant.description ?? "",
    })),
    defaultQuant: model.defaultQuant,
    defaultShift: numberAt(model.pipelineConfig, "scheduler", "shift"),
    defaultSteps: numberAt(model.pipelineConfig, "defaults", "steps"),
    textEncoderRepo: resolveSelection(manifest).containers[TEXT_ENCODER]?.parts[0]?.repo,
  };
};

const parseInteger = (text: string, what: string, min: number, max: number): number => {
  const value = Number(text.trim());
  if (text.trim() === "" || !Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${what} must be an integer from ${min} to ${max} (got '${text}')`);
  }
  return value;
};

/** The request parts the form decides (the page adds `onEvent` / `signal`). */
export type WanRequestParts = Omit<Ti2vGenerateRequest, "onEvent" | "signal"> & {
  readonly steps: number;
  readonly seed: number;
  readonly frames: number;
  readonly width: number;
  readonly height: number;
  readonly vaeTile: VaeTile;
};

/**
 * Form → request. Throws a `RangeError` naming the field for input the page can reject before
 * touching the GPU (the pipeline's `ModelInputError`s still cover the rest).
 */
export const buildRequest = (form: WanForm, prompts: readonly Prompt[]): WanRequestParts => {
  const [width, height] = form.size === "1280x704" ? [1280, 704] : [704, 1280];
  const shiftText = form.shift.trim();
  const shift = Number(shiftText);
  if (shiftText !== "" && !(Number.isFinite(shift) && shift > 0)) {
    throw new RangeError(`Shift must be a positive number (got '${form.shift}')`);
  }
  let prompt: string;
  let negativePrompt: string | undefined;
  if (form.textEncoder === "precomputed") {
    const row = prompts.find(({ name, role }) => name === form.promptName && role === "positive");
    if (row === undefined) throw new RangeError("Choose a prompt of the text embedding asset");
    prompt = row.prompt;
  } else {
    if (form.prompt.trim() === "") throw new RangeError("The prompt is empty");
    prompt = form.prompt;
    if (form.negativePrompt.trim() !== "") negativePrompt = form.negativePrompt;
  }
  return {
    prompt,
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    seed: parseInteger(form.seed, "Seed", 0, Number.MAX_SAFE_INTEGER),
    steps: parseInteger(form.steps, "Steps", 1, 1000),
    ...(shiftText === "" ? {} : { shift }),
    frames: form.frames,
    width,
    height,
    vaeTile: form.vaeTile,
  };
};

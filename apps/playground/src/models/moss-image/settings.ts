/**
 * The MOSS-Image page's form: the manifest's request defaults (`pipelineConfig`), and the pure step
 * from the form to a generation request. The pipeline still validates every request itself
 * (`ModelInputError` — the area cap of the untiled VAE, the token bound, the σ schedule); the checks
 * here only reject before touching the GPU what the page can tell is wrong, naming the field.
 */

import type { Manifest } from "@karume/hub";
import type { GenerateRequest } from "@karume/models/moss";

import type { MountName } from "@/mounts";

/** The one mount the page reads (`models/karume-moss-image`, on the cached or the direct route). */
export const MOSS_MOUNT = "moss-image" satisfies MountName;

/** What a model's manifest entry declares as the defaults of a request (`pipelineConfig`). */
export type MossDefaults = {
  readonly width?: number;
  readonly height?: number;
  readonly steps?: number;
  readonly guidance?: number;
  /** `scheduler.shift`. */
  readonly shift?: number;
  /** Used only while guidance ≠ 1 (upstream's default is the empty string). */
  readonly negativePrompt?: string;
};

const valueAt = (value: unknown, keys: readonly string[]): unknown => {
  let current = value;
  for (const key of keys) {
    if (typeof current !== "object" || current === null || !Object.hasOwn(current, key)) {
      return undefined;
    }
    current = Reflect.get(current, key);
  }
  return current;
};

const numberAt = (value: unknown, ...keys: readonly string[]): number | undefined => {
  const found = valueAt(value, keys);
  return typeof found === "number" ? found : undefined;
};

/**
 * The defaults of `model` as the manifest spells them (for the placeholders and the record; the
 * pipeline parses and checks `pipelineConfig` itself).
 */
export const modelDefaults = (manifest: Manifest, model: string): MossDefaults => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const config = entry.pipelineConfig;
  const width = numberAt(config, "defaults", "width");
  const height = numberAt(config, "defaults", "height");
  const steps = numberAt(config, "defaults", "steps");
  const guidance = numberAt(config, "defaults", "guidance");
  const shift = numberAt(config, "scheduler", "shift");
  const negativePrompt = valueAt(config, ["defaults", "negativePrompt"]);
  return {
    ...(width === undefined ? {} : { width }),
    ...(height === undefined ? {} : { height }),
    ...(steps === undefined ? {} : { steps }),
    ...(guidance === undefined ? {} : { guidance }),
    ...(shift === undefined ? {} : { shift }),
    ...(typeof negativePrompt === "string" ? { negativePrompt } : {}),
  };
};

/** Quick choices for the size fields (each side a multiple of 16; the smaller ones run faster). */
export const SIZE_CHOICES: readonly { readonly width: number; readonly height: number }[] = [
  { width: 1024, height: 1024 },
  { width: 1152, height: 896 },
  { width: 896, height: 1152 },
  { width: 768, height: 768 },
  { width: 512, height: 512 },
];

export type MossForm = {
  readonly quant: string;
  readonly prompt: string;
  /** Empty = the manifest's default; not sent at guidance 1 (no negative branch runs). */
  readonly negativePrompt: string;
  /** Empty = the manifest's default. */
  readonly width: string;
  /** Empty = the manifest's default. */
  readonly height: string;
  /** Empty = the manifest's default. */
  readonly steps: string;
  /** Empty = the manifest's default. */
  readonly guidance: string;
  /** Empty = the manifest's default. */
  readonly shift: string;
  readonly seed: string;
};

export const initialForm = (prompt: string): MossForm => ({
  quant: "",
  prompt,
  negativePrompt: "",
  width: "",
  height: "",
  steps: "",
  guidance: "",
  shift: "",
  seed: "42",
});

/** The request the form decides: the prompt form with a seed (no latents, overflow refused). */
export type MossRequestParts = Omit<GenerateRequest, "latents" | "overflow"> & {
  readonly seed: number;
};

const parseInteger = (text: string, what: string, min: number): number => {
  const value = Number(text.trim());
  if (text.trim() === "" || !Number.isSafeInteger(value) || value < min) {
    throw new RangeError(`${what} must be an integer ≥ ${min} (got '${text}')`);
  }
  return value;
};

const parseSide = (text: string, what: string): number => {
  const value = Number(text.trim());
  if (text.trim() === "" || !Number.isSafeInteger(value) || value < 16 || value % 16 !== 0) {
    throw new RangeError(`${what} must be a positive multiple of 16 (got '${text}')`);
  }
  return value;
};

/** A finite number; `undefined` for an empty field. */
const parseNumber = (text: string, what: string): number | undefined => {
  if (text.trim() === "") return undefined;
  const value = Number(text.trim());
  if (!Number.isFinite(value))
    throw new RangeError(`${what} must be a finite number (got '${text}')`);
  return value;
};

/**
 * The guidance a request runs with: the form's value, else the manifest's default. `undefined` when
 * neither is known (the form's text is not a number — {@link buildRequest} names that).
 */
export const effectiveGuidance = (form: MossForm, defaults: MossDefaults): number | undefined => {
  const text = form.guidance.trim();
  if (text === "") return defaults.guidance;
  const value = Number(text);
  return Number.isFinite(value) ? value : undefined;
};

/**
 * Whether the negative prompt takes part: only when the guidance is not 1 (at 1 the pipeline runs no
 * negative branch and refuses a negative prompt).
 */
export const usesNegativePrompt = (form: MossForm, defaults: MossDefaults): boolean => {
  const guidance = effectiveGuidance(form, defaults);
  return guidance !== undefined && guidance !== 1;
};

/**
 * Form → request. Only the fields the form sets are sent; an empty field leaves the manifest's
 * default to the pipeline. Throws a `RangeError` naming the field for input the page can reject
 * before touching the GPU.
 */
export const buildRequest = (form: MossForm, defaults: MossDefaults): MossRequestParts => {
  if (form.prompt.trim() === "") throw new RangeError("The prompt is empty");
  const guidance = parseNumber(form.guidance, "Guidance");
  if (guidance !== undefined && guidance < 0) {
    throw new RangeError(`Guidance must be ≥ 0 (got '${form.guidance}')`);
  }
  const shift = parseNumber(form.shift, "Shift");
  if (shift !== undefined && shift <= 0) {
    throw new RangeError(`Shift must be > 0 (got '${form.shift}')`);
  }
  const negativePrompt =
    usesNegativePrompt(form, defaults) && form.negativePrompt.trim() !== ""
      ? form.negativePrompt
      : undefined;
  return {
    prompt: form.prompt,
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    ...(form.width.trim() === "" ? {} : { width: parseSide(form.width, "Width") }),
    ...(form.height.trim() === "" ? {} : { height: parseSide(form.height, "Height") }),
    ...(form.steps.trim() === "" ? {} : { steps: parseInteger(form.steps, "Steps", 1) }),
    ...(guidance === undefined ? {} : { guidance }),
    ...(shift === undefined ? {} : { shift }),
    seed: parseInteger(form.seed, "Seed", 0),
  };
};

/** The values a request runs with once the manifest's defaults fill the fields it leaves out. */
export type EffectiveRequest = {
  readonly width: number | null;
  readonly height: number | null;
  readonly steps: number | null;
  readonly guidance: number | null;
  readonly shift: number | null;
  /** `null` when no negative branch runs (guidance 1). */
  readonly negativePrompt: string | null;
};

export const effectiveRequest = (
  parts: MossRequestParts,
  defaults: MossDefaults,
): EffectiveRequest => {
  const guidance = parts.guidance ?? defaults.guidance;
  return {
    width: parts.width ?? defaults.width ?? null,
    height: parts.height ?? defaults.height ?? null,
    steps: parts.steps ?? defaults.steps ?? null,
    guidance: guidance ?? null,
    shift: parts.shift ?? defaults.shift ?? null,
    negativePrompt:
      guidance === undefined || guidance === 1
        ? null
        : (parts.negativePrompt ?? defaults.negativePrompt ?? null),
  };
};

/** A fresh seed (a non-negative safe integer, as the pipeline requires). */
export const randomSeed = (): number => Math.floor(Math.random() * 2 ** 32);

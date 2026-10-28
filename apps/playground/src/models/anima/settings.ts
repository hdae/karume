/**
 * The Anima page's form: the model choice over the two distributions, the manifest's defaults of a
 * model, and the pure step from the form to a generation request. The pipeline still validates every
 * request itself (`ModelInputError`); the checks here only reject before touching the GPU what the
 * page can tell is wrong, naming the field.
 */

import { type Manifest, resolveSelection } from "@karume/hub";
import {
  type AnimaGenerateRequest,
  type AnimaResidency,
  type AnimaSamplerType,
  type ImageSize,
  parseResolution,
} from "@karume/models/anima";

import type { Choice } from "@/components/choice-select";
import { modelChoices } from "@/lib/presets";

/** The two mounts the page reads: the official models and the third-party fine-tunes. */
export const ANIMA_MOUNTS = ["anima", "anima-extra"] as const;
export type AnimaMount = (typeof ANIMA_MOUNTS)[number];

export const MOUNT_LABELS: Readonly<Record<AnimaMount, string>> = {
  anima: "Official",
  "anima-extra": "Fine-tune",
};

/** One model of one mount (the select's value is {@link modelKey} of it). */
export type ModelRef = { readonly mount: AnimaMount; readonly model: string };

export const modelKey = (ref: ModelRef): string => `${ref.mount}/${ref.model}`;

/** The inverse of {@link modelKey}; `undefined` for a key of no known mount. */
export const parseModelKey = (key: string): ModelRef | undefined => {
  const slash = key.indexOf("/");
  if (slash < 0) return undefined;
  const mount = ANIMA_MOUNTS.find((name) => name === key.slice(0, slash));
  return mount === undefined ? undefined : { mount, model: key.slice(slash + 1) };
};

export type AnimaManifests = Partial<Record<AnimaMount, Manifest>>;

/**
 * The models of every loaded manifest in mount order, each mount's `defaultModel` marked
 * "(default)" and prefixed with the mount's label.
 */
export const animaModelChoices = (manifests: AnimaManifests): readonly Choice<string>[] =>
  ANIMA_MOUNTS.flatMap((mount) => {
    const manifest = manifests[mount];
    if (manifest === undefined) return [];
    return modelChoices(manifest).map(({ value, label }) => ({
      value: modelKey({ mount, model: value }),
      label: `${MOUNT_LABELS[mount]} · ${label}`,
    }));
  });

/** The page's default model: the official `defaultModel`, else the fine-tunes' (whichever loaded). */
export const defaultModelRef = (manifests: AnimaManifests): ModelRef | undefined => {
  for (const mount of ANIMA_MOUNTS) {
    const manifest = manifests[mount];
    if (manifest !== undefined) return { mount, model: manifest.defaultModel };
  }
  return undefined;
};

/**
 * The mount that holds each repository a manifest may reference across repositories: the fine-tunes
 * take the text encoder, the VAE and the tokenizers from the official repository. Keyed by the public
 * repo name (`ANIMA_SOURCES["anima"].repo` — the name the `anima` mount is served under).
 */
export type RepoHolders = Readonly<Record<string, AnimaMount>>;

/**
 * The `crossRepo` map of the direct route for one selection: every repository its containers and
 * assets reference, mapped to the mount that holds it. Throws when a referenced repository has no
 * mount (the direct route would fail on the first part it cannot read).
 */
export const crossRepoMounts = (
  manifest: Manifest,
  model: string,
  quant: string,
  holders: RepoHolders,
): Readonly<Record<string, AnimaMount>> => {
  const selection = resolveSelection(manifest, { model, quant });
  const files = [
    ...Object.values(selection.containers).flatMap(({ parts }) => parts),
    ...Object.values(selection.assets),
  ];
  const map: Record<string, AnimaMount> = {};
  for (const { repo } of files) {
    if (repo === undefined || Object.hasOwn(map, repo)) continue;
    if (!Object.hasOwn(holders, repo)) {
      throw new Error(
        `Model '${model}' (${quant}) references ${repo}, which no mount of this page holds`,
      );
    }
    map[repo] = holders[repo];
  }
  return map;
};

/** What a model's manifest entry declares as the defaults of a request (`pipelineConfig`). */
export type ModelDefaults = {
  readonly steps?: number;
  readonly guidanceScale?: number;
  readonly resolution?: ImageSize;
  readonly negativePrompt?: string;
  /** `scheduler.type`; absent = the pipeline's own default. */
  readonly sampler?: string;
  readonly shift?: number;
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

const stringAt = (value: unknown, ...keys: readonly string[]): string | undefined => {
  const found = valueAt(value, keys);
  return typeof found === "string" ? found : undefined;
};

/**
 * The defaults of `model` as the manifest spells them (for the placeholders and the record; the
 * pipeline parses and checks `pipelineConfig` itself).
 */
export const modelDefaults = (manifest: Manifest, model: string): ModelDefaults => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const config = entry.pipelineConfig;
  const width = numberAt(config, "defaults", "resolution", "width");
  const height = numberAt(config, "defaults", "resolution", "height");
  const steps = numberAt(config, "defaults", "steps");
  const guidanceScale = numberAt(config, "defaults", "guidanceScale");
  const negativePrompt = stringAt(config, "defaults", "negativePrompt");
  const sampler = stringAt(config, "scheduler", "type");
  const shift = numberAt(config, "scheduler", "shift");
  return {
    ...(steps === undefined ? {} : { steps }),
    ...(guidanceScale === undefined ? {} : { guidanceScale }),
    ...(width === undefined || height === undefined ? {} : { resolution: { width, height } }),
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    ...(sampler === undefined ? {} : { sampler }),
    ...(shift === undefined ? {} : { shift }),
  };
};

/** Quick choices for the resolution field (every one inside the pipeline's accepted set). */
export const RESOLUTION_CHOICES: readonly { readonly value: string; readonly label: string }[] = [
  { value: "1024x1024", label: "1024×1024" },
  { value: "1152x896", label: "1152×896" },
  { value: "896x1152", label: "896×1152" },
  { value: "1216x832", label: "1216×832" },
  { value: "832x1216", label: "832×1216" },
  { value: "1344x768", label: "1344×768" },
  { value: "768x1344", label: "768×1344" },
  { value: "512x512", label: "512×512" },
];

export type AnimaForm = {
  /** {@link modelKey} of the chosen model; empty until a manifest has loaded. */
  readonly model: string;
  readonly quant: string;
  readonly prompt: string;
  /** Empty = the manifest's default; not sent at guidance 1 (the uncond side is not computed). */
  readonly negativePrompt: string;
  /** Empty = the manifest's default. */
  readonly guidanceScale: string;
  /** Empty = the manifest's default. */
  readonly steps: string;
  /** `WxH` or the square shorthand; empty = the manifest's default. */
  readonly resolution: string;
  /** Empty = the manifest's `scheduler.type`. */
  readonly sampler: "" | AnimaSamplerType;
  readonly seed: string;
  readonly residency: AnimaResidency;
};

export const initialForm = (prompt: string): AnimaForm => ({
  model: "",
  quant: "",
  prompt,
  negativePrompt: "",
  guidanceScale: "",
  steps: "",
  resolution: "",
  sampler: "",
  seed: "42",
  residency: "per-stage",
});

/** The request parts the form decides (the page adds `onEvent`). */
export type AnimaRequestParts = Omit<AnimaGenerateRequest, "onEvent"> & {
  readonly seed: number;
  readonly residency: AnimaResidency;
};

const parseInteger = (text: string, what: string, min: number, max: number): number => {
  const value = Number(text.trim());
  if (text.trim() === "" || !Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${what} must be an integer from ${min} to ${max} (got '${text}')`);
  }
  return value;
};

/**
 * The guidance a request runs with: the form's value, else the manifest's default. `undefined` when
 * neither is known (the form's text is not a number — {@link buildRequest} names that).
 */
export const effectiveGuidance = (form: AnimaForm, defaults: ModelDefaults): number | undefined => {
  const text = form.guidanceScale.trim();
  if (text === "") return defaults.guidanceScale;
  const value = Number(text);
  return Number.isFinite(value) ? value : undefined;
};

/**
 * Whether the negative prompt takes part: only when the guidance is not 1 (at 1 the pipeline never
 * computes the uncond side and refuses a negative prompt).
 */
export const usesNegativePrompt = (form: AnimaForm, defaults: ModelDefaults): boolean => {
  const guidance = effectiveGuidance(form, defaults);
  return guidance !== undefined && guidance !== 1;
};

/**
 * Form → request. Only the fields the form sets are sent; an empty field leaves the manifest's
 * default to the pipeline. Throws a `RangeError` (or the pipeline's `ModelInputError` for the
 * resolution spelling) naming the field for input the page can reject before touching the GPU.
 */
export const buildRequest = (form: AnimaForm, defaults: ModelDefaults): AnimaRequestParts => {
  if (form.prompt.trim() === "") throw new RangeError("The prompt is empty");
  const guidanceText = form.guidanceScale.trim();
  const guidanceScale = Number(guidanceText);
  if (guidanceText !== "" && !Number.isFinite(guidanceScale)) {
    throw new RangeError(`Guidance must be a finite number (got '${form.guidanceScale}')`);
  }
  let negativePrompt: string | undefined;
  if (usesNegativePrompt(form, defaults)) {
    if (form.negativePrompt.trim() !== "") {
      negativePrompt = form.negativePrompt;
    } else if (defaults.negativePrompt === undefined) {
      throw new RangeError(
        "This model has no default negative prompt: enter one (guidance other than 1 computes the uncond side)",
      );
    }
  }
  const resolutionText = form.resolution.trim();
  return {
    prompt: form.prompt,
    ...(negativePrompt === undefined ? {} : { negativePrompt }),
    ...(guidanceText === "" ? {} : { guidanceScale }),
    ...(form.steps.trim() === "" ? {} : { steps: parseInteger(form.steps, "Steps", 2, 1000) }),
    ...(resolutionText === "" ? {} : { resolution: parseResolution(resolutionText) }),
    ...(form.sampler === "" ? {} : { sampler: form.sampler }),
    seed: parseInteger(form.seed, "Seed", 0, Number.MAX_SAFE_INTEGER),
    residency: form.residency,
  };
};

/** The values a request runs with once the manifest's defaults fill the fields it leaves out. */
export type EffectiveRequest = {
  readonly steps: number | null;
  readonly guidanceScale: number | null;
  /** `WxH`, both sides spelled out. */
  readonly resolution: string | null;
  /** `null` = the manifest declares none (the pipeline's own default applies). */
  readonly sampler: string | null;
  /** `null` when the uncond side is not computed (guidance 1). */
  readonly negativePrompt: string | null;
};

const spellSize = (size: ImageSize): string => `${size.width}x${size.height}`;

export const effectiveRequest = (
  parts: AnimaRequestParts,
  defaults: ModelDefaults,
): EffectiveRequest => {
  const guidanceScale = parts.guidanceScale ?? defaults.guidanceScale;
  const resolution = parts.resolution ?? defaults.resolution;
  return {
    steps: parts.steps ?? defaults.steps ?? null,
    guidanceScale: guidanceScale ?? null,
    resolution: resolution === undefined ? null : spellSize(resolution),
    sampler: parts.sampler ?? defaults.sampler ?? null,
    negativePrompt:
      guidanceScale === undefined || guidanceScale === 1
        ? null
        : (parts.negativePrompt ?? defaults.negativePrompt ?? null),
  };
};

/** The resolution as the field's placeholder shows the manifest's default. */
export const resolutionText = (size: ImageSize | undefined): string =>
  size === undefined ? "default" : `default (${spellSize(size)})`;

/** A fresh seed (a non-negative safe integer, as the pipeline requires). */
export const randomSeed = (): number => Math.floor(Math.random() * 2 ** 32);

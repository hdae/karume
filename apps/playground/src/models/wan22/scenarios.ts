/**
 * Fixed run scenarios of the Wan2.2 page: complete settings (form and source) for the measurements
 * that are compared across runs and machines, so a record never differs from another by a setting
 * typed by hand. "Scenario", not "preset": on this page a preset is a quant of the manifest.
 */

import type { Prompt } from "@karume/models/wan";

import type { VaeTile, WanForm } from "./settings";

export type ScenarioId = "quick-check" | "bench-33f" | "bench-33f-tile32" | "bench-121f";

export type Scenario = {
  readonly id: ScenarioId;
  readonly label: string;
  readonly hint: string;
  readonly frames: number;
  readonly steps: number;
  readonly vaeTile: VaeTile;
};

/** What every scenario shares (besides the manifest's default quant and the source). */
const SCENARIO_PROMPT_NAME = "boxing-cats";
const SCENARIO_SEED = "42";
const SCENARIO_SIZE = "1280x704";

export const SCENARIOS: readonly Scenario[] = [
  {
    id: "quick-check",
    label: "Quick check",
    hint: "2 steps at 17 frames: runs every stage, VAE tiles included, in little time.",
    frames: 17,
    steps: 2,
    vaeTile: 16,
  },
  {
    id: "bench-33f",
    label: "33 frames, 50 steps (tile 16)",
    hint: "The comparison point with the earlier records.",
    frames: 33,
    steps: 50,
    vaeTile: 16,
  },
  {
    id: "bench-33f-tile32",
    label: "33 frames, 50 steps (tile 32)",
    hint: "As the 33-frame bench, decoded in 32-latent VAE tiles.",
    frames: 33,
    steps: 50,
    vaeTile: 32,
  },
  {
    id: "bench-121f",
    label: "121 frames, 50 steps (tile 32)",
    hint: "The longest video the pipeline accepts, decoded in 32-latent VAE tiles.",
    frames: 121,
    steps: 50,
    vaeTile: 32,
  },
];

/** The page's facts a scenario depends on. */
export type ScenarioContext = {
  readonly defaultQuant: string;
  /** The text embedding asset's rows (the asset is the model's: the same for every quant and path). */
  readonly prompts: readonly Prompt[];
  /** Whether the dev server serves the distribution on the HF-compatible route (`/hf/`). */
  readonly cachedSource: boolean;
  /** Whether the umT5 distribution the GPU text encoder needs is there. */
  readonly gpuTextEncoder: boolean;
};

/** A scenario that cannot be applied on this page (the message says why). */
export class ScenarioUnavailableError extends Error {
  override readonly name = "ScenarioUnavailableError";
}

export type ScenarioSettings = {
  readonly form: WanForm;
  readonly source: "cached";
};

/** Scenario → the complete form and source. Throws `ScenarioUnavailableError` when it cannot apply. */
export const scenarioSettings = (
  scenario: Scenario,
  context: ScenarioContext,
): ScenarioSettings => {
  if (!context.cachedSource) {
    throw new ScenarioUnavailableError(
      `'${scenario.label}' reads through the cache, but the dev server serves no HF-compatible route (/hf/) for this distribution`,
    );
  }
  if (!context.gpuTextEncoder) {
    throw new ScenarioUnavailableError(
      `'${scenario.label}' runs the GPU text encoder, but the umT5 distribution is missing`,
    );
  }
  const row = context.prompts.find(
    ({ name, role }) => name === SCENARIO_PROMPT_NAME && role === "positive",
  );
  if (row === undefined) {
    throw new ScenarioUnavailableError(
      context.prompts.length === 0
        ? `'${scenario.label}' needs the prompt list of the text embedding asset, which is not known until a pipeline has opened`
        : `'${scenario.label}' needs the positive prompt '${SCENARIO_PROMPT_NAME}', which the text embedding asset does not have`,
    );
  }
  return {
    form: {
      quant: context.defaultQuant,
      vaeTile: scenario.vaeTile,
      size: SCENARIO_SIZE,
      frames: scenario.frames,
      steps: String(scenario.steps),
      seed: SCENARIO_SEED,
      // Empty = the manifest's default shift and the official default negative prompt.
      shift: "",
      textEncoder: "gpu",
      // Unused on the GPU path; kept on the same row so switching to `precomputed` stays on it.
      promptName: row.name,
      prompt: row.prompt,
      negativePrompt: "",
    },
    source: "cached",
  };
};

/**
 * The form fields a generation reads on the scenarios' (GPU) path. `promptName` is left out: that
 * path reads the free-text prompt only.
 */
const COMPARED_FIELDS = [
  "quant",
  "vaeTile",
  "size",
  "frames",
  "steps",
  "seed",
  "shift",
  "textEncoder",
  "prompt",
  "negativePrompt",
] as const satisfies readonly (keyof WanForm)[];

/** The scenario the form and source equal, or `"custom"` (also when no scenario can apply). */
export const matchingScenario = (
  form: WanForm,
  source: "cached" | "direct",
  context: ScenarioContext,
): ScenarioId | "custom" => {
  for (const scenario of SCENARIOS) {
    let settings: ScenarioSettings;
    try {
      settings = scenarioSettings(scenario, context);
    } catch (error) {
      if (error instanceof ScenarioUnavailableError) continue;
      throw error;
    }
    if (
      settings.source === source &&
      COMPARED_FIELDS.every((field) => settings.form[field] === form[field])
    ) {
      return scenario.id;
    }
  }
  return "custom";
};

/**
 * What the page reads from one model's `pipelineConfig` in an EmbeddingGemma manifest: the prompt
 * names with their prefixes (the prompt select of each row), the output dimensions (the dimension
 * select) and the token limit (a hint). The distribution owns these values — the page derives every
 * list from here and hard-codes none. `embeddinggemma.Pipeline` validates the whole block again
 * when it is built; this reader only refuses a block it cannot show (fail loudly, never guess) and
 * leaves the fields it does not show (`placeholderTokens`) to the pipeline.
 */

import type { Manifest } from "@karume/hub";

import type { Choice } from "@/components/choice-select";

/** One prompt the model declares: its name and the prefix the pipeline puts before the text. */
export type DeclaredPrompt = { readonly name: string; readonly prefix: string };

export type EmbeddingModelConfig = {
  /** In the manifest's order (upstream's). */
  readonly prompts: readonly DeclaredPrompt[];
  /** Descending; the first is the full width and the pipeline's default. */
  readonly dimensions: readonly [number, ...number[]];
  /** The longest token sequence one call accepts (`<bos>` and `<eos>` included). */
  readonly maxTokens: number;
};

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** An own field of `block` (never an inherited one such as `constructor`), or `undefined`. */
const ownField = (block: Readonly<Record<string, unknown>>, key: string): unknown =>
  Object.hasOwn(block, key) ? block[key] : undefined;

const positiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value > 0;

/** `value` as a non-empty, strictly descending list of positive integers, else `undefined`. */
const descendingDimensions = (value: unknown): readonly [number, ...number[]] | undefined => {
  if (!Array.isArray(value)) return undefined;
  const items: readonly unknown[] = value;
  const [first, ...rest] = items;
  if (!positiveInteger(first) || !rest.every(positiveInteger)) return undefined;
  const list: [number, ...number[]] = [first, ...rest];
  return list.every((dimension, index) => index === 0 || dimension < (list[index - 1] ?? 0))
    ? list
    : undefined;
};

/** The page's view of `pipelineConfig` of `model` (throws naming the field that is off). */
export const readEmbeddingConfig = (manifest: Manifest, model: string): EmbeddingModelConfig => {
  const entry = Object.hasOwn(manifest.models, model) ? manifest.models[model] : undefined;
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const where = `models.${model}.pipelineConfig`;
  const config = entry.pipelineConfig;

  const promptBlock = ownField(config, "prompts");
  if (!isRecord(promptBlock)) throw new Error(`${where}.prompts is not an object`);
  const prompts = Object.entries(promptBlock).map(([name, prefix]) => {
    if (typeof prefix !== "string") {
      throw new Error(`${where}.prompts.${name} is not a string (got ${JSON.stringify(prefix)})`);
    }
    return { name, prefix };
  });

  const dimensions = descendingDimensions(ownField(config, "dimensions"));
  if (dimensions === undefined) {
    throw new Error(
      `${where}.dimensions is not a non-empty descending list of positive integers (got ${JSON.stringify(ownField(config, "dimensions"))})`,
    );
  }

  const maxTokens = ownField(config, "maxTokens");
  if (!positiveInteger(maxTokens)) {
    throw new Error(
      `${where}.maxTokens is not a positive integer (got ${JSON.stringify(maxTokens)})`,
    );
  }
  return { prompts, dimensions, maxTokens };
};

/** The value of the "no prompt" choice (a prompt's value is `prompt:<name>`, so the two never meet). */
const NO_PROMPT = "none";

/** The select value of a row's prompt (`undefined` = no prompt). */
export const promptValue = (prompt: string | undefined): string =>
  prompt === undefined ? NO_PROMPT : `prompt:${prompt}`;

/** The prompt a select value names (`undefined` = no prompt). */
export const promptOfValue = (value: string): string | undefined =>
  value.startsWith("prompt:") ? value.slice("prompt:".length) : undefined;

/**
 * The prompt select of a row: "none", then every declared prompt (`query — task: search result |
 * query: `). A `current` prompt the model does not declare is listed too, marked, so the select
 * shows what the run will send — and the pipeline's refusal of it reaches the row as is.
 */
export const promptChoices = (
  config: EmbeddingModelConfig,
  current: string | undefined,
): readonly Choice<string>[] => [
  { value: NO_PROMPT, label: "none (no prefix)" },
  ...config.prompts.map(({ name, prefix }) => ({
    value: promptValue(name),
    label: `${name} — ${JSON.stringify(prefix)}`,
  })),
  ...(current === undefined || config.prompts.some(({ name }) => name === current)
    ? []
    : [{ value: promptValue(current), label: `${current} (not declared by this model)` }]),
];

/** The dimension select: every declared width, the first marked as the full width (the default). */
export const dimensionChoices = (config: EmbeddingModelConfig): readonly Choice<string>[] =>
  config.dimensions.map((dimension, index) => ({
    value: String(dimension),
    label: index === 0 ? `${dimension} (full, default)` : String(dimension),
  }));

/** The width a run asks for: the chosen one where the model declares it, else the full width. */
export const effectiveDimension = (
  config: EmbeddingModelConfig,
  chosen: number | undefined,
): number =>
  chosen !== undefined && config.dimensions.includes(chosen) ? chosen : config.dimensions[0];

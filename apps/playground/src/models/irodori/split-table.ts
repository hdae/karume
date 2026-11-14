/**
 * Whether the chosen distribution's tokenizer asset carries upstream's rare-token split table
 * (ADR 0149): the fact the pipeline's `splitRareTokens` default and its refusal of `true` rest on.
 * The page reads it before a pipeline exists, so it offers the option only where the table exists.
 */

import { type DistributionSource, fetchAssets, type HubRepoRef, loadManifest } from "@karume/hub";

/** The manifest's asset key of the text tokenizer (the key the pipeline reads). */
const TOKENIZER_ASSET = "tokenizer";

/**
 * Whether a parsed tokenizer asset carries the split table. For every asset the pipeline accepts,
 * that is the top-level `rareTokenSplit` key: the Unigram asset of the small models (no `format`
 * key) has it from the distribution revision that added it, and the Gemma asset of v4-large
 * (`karume-gemma-tokenizer/1`) never has it (its reader refuses unknown keys, and the pipeline's Gemma
 * tokenizer has no split). An asset the pipeline would refuse fails at the build, not here.
 */
export const hasRareTokenSplitTable = (asset: unknown): boolean => {
  if (typeof asset !== "object" || asset === null || Array.isArray(asset)) {
    throw new Error("The tokenizer asset is not a JSON object");
  }
  return Object.hasOwn(asset, "rareTokenSplit");
};

/**
 * Reads the tokenizer asset of `ref`'s default model through the hub and answers
 * {@link hasRareTokenSplitTable}. `ref` is what the pipeline build gets (`pretrainedRef`), so the
 * read follows the same cross-repo references, and on the cached route it fills the same
 * CacheStorage the build then reads the asset from.
 */
export const readSplitTable = async (
  ref: HubRepoRef | DistributionSource,
  signal?: AbortSignal,
): Promise<boolean> => {
  const options = signal === undefined ? {} : { signal };
  const loaded = await loadManifest(ref, options);
  const { manifest } = loaded;
  const model = manifest.defaultModel;
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const file = Object.hasOwn(entry.assets, TOKENIZER_ASSET)
    ? entry.assets[TOKENIZER_ASSET]
    : undefined;
  if (file === undefined) {
    throw new Error(`Model '${model}' of the manifest has no '${TOKENIZER_ASSET}' asset`);
  }
  // fetchAssets returns the table with the keys it was given.
  const bytes = (await fetchAssets(loaded, { [TOKENIZER_ASSET]: file }, options))[TOKENIZER_ASSET];
  const asset: unknown = JSON.parse(new TextDecoder().decode(bytes));
  return hasRareTokenSplitTable(asset);
};

/**
 * The preset a local entry runs: one quant of a model as its manifest declares it, found by name or
 * as the model's default. The Gemma entries and the session A/B's Session targets look presets up
 * here and word their own refusals.
 */

import type { Manifest, Quant } from "@karume/hub";

/** A preset of a model: its name, the manifest's declaration of it and whether it is the default. */
export type FoundPreset = {
  readonly name: string;
  readonly declaration: Quant;
  readonly isDefault: boolean;
};

export type PresetLookup =
  | ({ readonly ok: true } & FoundPreset)
  | {
      readonly ok: false;
      /** The name looked up. */
      readonly chosen: string;
      /** The model's presets, in the manifest's order. */
      readonly presets: readonly string[];
    };

/**
 * The preset `chosen` of `model` (`undefined`: the model's default), or the presets it has when it
 * has none of that name. Throws for a model the manifest does not have: a caller checks the model
 * first.
 */
export const modelPreset = (
  manifest: Manifest,
  model: string,
  chosen: string | undefined,
): PresetLookup => {
  const entry = Object.hasOwn(manifest.models, model) ? manifest.models[model] : undefined;
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const name = chosen ?? entry.defaultQuant;
  const declaration = Object.hasOwn(entry.quants, name) ? entry.quants[name] : undefined;
  return declaration === undefined
    ? { ok: false, chosen: name, presets: Object.keys(entry.quants) }
    : { ok: true, name, declaration, isDefault: name === entry.defaultQuant };
};

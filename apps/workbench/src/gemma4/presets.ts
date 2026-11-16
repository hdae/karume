/**
 * The presets a manifest offers for a model: each a quant of the manifest (`quant` of
 * `fromPretrained`) with its label and description, and the model's default. Copied from the
 * Playground's `apps/playground/src/lib/presets.ts` (`manifestPresets`), which keeps its own: the
 * workbench and the Playground share no module. Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";

/** One preset of a model: its name, the manifest's label and one-line description. */
export type Preset = {
  readonly name: string;
  readonly label: string;
  readonly description: string;
};

/** The presets of `model` (the manifest's `defaultModel` when omitted) and its default preset. */
export const manifestPresets = (
  manifest: Manifest,
  model: string = manifest.defaultModel,
): { readonly presets: readonly Preset[]; readonly defaultQuant: string } => {
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  return {
    presets: Object.entries(entry.quants).map(([name, quant]) => ({
      name,
      label: quant.label ?? name,
      description: quant.description ?? "",
    })),
    defaultQuant: entry.defaultQuant,
  };
};

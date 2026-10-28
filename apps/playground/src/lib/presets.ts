/**
 * The models and presets a manifest offers, as the page's selects show them. A preset is a quant of
 * the manifest (`quant` of `fromPretrained`); a model is one of `models` (`model` of
 * `fromPretrained`). Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";

import type { Choice } from "@/components/choice-select";

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

/** `name — label`, the default marked "(default)" (the Wan2.2 page's spelling). */
export const presetChoices = (
  presets: readonly Preset[],
  defaultQuant: string,
): readonly Choice<string>[] =>
  presets.map(({ name, label }) => ({
    value: name,
    label: `${name} — ${label}${name === defaultQuant ? " (default)" : ""}`,
  }));

/** The manifest's models in its order, `defaultModel` marked "(default)". */
export const modelChoices = (manifest: Manifest): readonly Choice<string>[] =>
  Object.keys(manifest.models).map((name) => ({
    value: name,
    label: `${name}${name === manifest.defaultModel ? " (default)" : ""}`,
  }));

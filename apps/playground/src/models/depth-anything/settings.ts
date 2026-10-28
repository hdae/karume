/**
 * The Depth Anything page's form and what it needs before a run: the chosen model and preset, the
 * model's input size from the manifest, and why Run is disabled. Pure, so `vp test` covers it.
 */

import type { Manifest } from "@karume/hub";

/** The model and preset of the form; empty until the manifest's defaults are known. */
export type DepthForm = { readonly model: string; readonly quant: string };

/**
 * The form after the manifest is known: an empty or unknown model becomes `defaultModel`, an empty
 * or unknown preset becomes that model's `defaultQuant`. A valid choice is kept.
 */
export const settleForm = (form: DepthForm, manifest: Manifest): DepthForm => {
  const model = Object.hasOwn(manifest.models, form.model) ? form.model : manifest.defaultModel;
  const entry = manifest.models[model];
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const quant = Object.hasOwn(entry.quants, form.quant) ? form.quant : entry.defaultQuant;
  return model === form.model && quant === form.quant ? form : { model, quant };
};

/** The square size the graph takes (`pipelineConfig.imageWidth` / `imageHeight` of the model). */
export type Size = { readonly width: number; readonly height: number };

const positiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value > 0;

/**
 * The model's input size from its `pipelineConfig`, `undefined` when the manifest does not state it
 * as positive integers (the pipeline validates the config itself; the page only shows the size).
 */
export const modelInputSize = (
  pipelineConfig: Readonly<Record<string, unknown>>,
): Size | undefined => {
  const { imageWidth, imageHeight } = pipelineConfig;
  return positiveInteger(imageWidth) && positiveInteger(imageHeight)
    ? { width: imageWidth, height: imageHeight }
    : undefined;
};

/** What Run waits for, in the order the page gets them. */
export type RunReadiness = {
  readonly gpuReady: boolean;
  readonly manifestReady: boolean;
  readonly formSettled: boolean;
  readonly hasImage: boolean;
  readonly running: boolean;
};

/** Why Run is disabled now (shown next to the button), or `undefined` when it can run. */
export const runBlocker = (readiness: RunReadiness): string | undefined => {
  if (readiness.running) return "A run is in progress.";
  if (!readiness.manifestReady) return "The manifest is not loaded.";
  if (!readiness.formSettled) return "Choose a model and a preset.";
  if (!readiness.gpuReady) return "The GPU is not ready.";
  if (!readiness.hasImage) return "Choose an image.";
  return undefined;
};

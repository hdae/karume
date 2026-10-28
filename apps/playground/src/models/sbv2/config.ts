/**
 * What the page reads from one model's `pipelineConfig` in the SBV2 manifest: the speaker and style
 * names (the selects), the distribution's defaults (the form's starting values) and the limits
 * `maxTokens` / `maxFrames` (the hints on the text and on a rejected input). The distribution owns these sets — the page derives every list from here and
 * hard-codes none. `Pipeline` validates the same block again when it is built; this reader only
 * refuses a block it cannot show (fail loudly, never guess).
 */

import type { Manifest } from "@karume/hub";

/** The distribution's recommended values for the knobs `generate` fills when they are omitted. */
export type Sbv2Defaults = {
  readonly speaker: string;
  readonly style: string;
  readonly styleWeight: number;
  readonly sdpRatio: number;
  readonly noiseScale: number;
  readonly noiseScaleW: number;
  readonly lengthScale: number;
};

export type Sbv2ModelConfig = {
  /** Speaker names in table-row order (a jvnv model holds one speaker). */
  readonly speakers: readonly string[];
  /** Style names in table-row order. */
  readonly styles: readonly string[];
  readonly defaults: Sbv2Defaults;
  /** The largest token / phoneme count one synthesis accepts. */
  readonly maxTokens: number;
  /**
   * The largest total frame count one synthesis accepts (the predicted durations, so it grows with
   * the text and with `lengthScale`; checked after the front graph has run).
   */
  readonly maxFrames: number;
};

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** An own field of `block` (never an inherited one such as `constructor`), or `undefined`. */
const ownField = (block: Readonly<Record<string, unknown>>, key: string): unknown =>
  Object.hasOwn(block, key) ? block[key] : undefined;

/** The names of a `name → table row` block, ordered by row. */
const rowNames = (value: unknown, where: string): readonly string[] => {
  if (!isRecord(value)) throw new Error(`${where} is not an object`);
  const entries = Object.entries(value).map(([name, row]) => {
    if (typeof row !== "number" || !Number.isInteger(row) || row < 0) {
      throw new Error(`${where}.${name} is not a table row (got ${JSON.stringify(row)})`);
    }
    return [name, row] as const;
  });
  if (entries.length === 0) throw new Error(`${where} is empty`);
  return entries.toSorted(([, left], [, right]) => left - right).map(([name]) => name);
};

const numberField = (block: Readonly<Record<string, unknown>>, key: string, where: string) => {
  const value = ownField(block, key);
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${where}.${key} is not a finite number (got ${JSON.stringify(value)})`);
  }
  return value;
};

const nameField = (
  block: Readonly<Record<string, unknown>>,
  key: string,
  names: readonly string[],
  where: string,
): string => {
  const value = ownField(block, key);
  if (typeof value !== "string" || !names.includes(value)) {
    throw new Error(`${where}.${key} ${JSON.stringify(value)} is not one of ${names.join(" / ")}`);
  }
  return value;
};

/** The page's view of `pipelineConfig` of `model` (throws naming the field that is off). */
export const readSbv2Config = (manifest: Manifest, model: string): Sbv2ModelConfig => {
  const entry = Object.hasOwn(manifest.models, model) ? manifest.models[model] : undefined;
  if (entry === undefined) throw new Error(`The manifest has no model '${model}'`);
  const where = `models.${model}.pipelineConfig`;
  const config = entry.pipelineConfig;
  const speakers = rowNames(ownField(config, "speakers"), `${where}.speakers`);
  const styles = rowNames(ownField(config, "styles"), `${where}.styles`);
  const maxTokens = numberField(config, "maxTokens", where);
  const maxFrames = numberField(config, "maxFrames", where);
  const defaultsBlock = ownField(config, "defaults");
  if (!isRecord(defaultsBlock)) throw new Error(`${where}.defaults is not an object`);
  const at = `${where}.defaults`;
  return {
    speakers,
    styles,
    maxTokens,
    maxFrames,
    defaults: {
      speaker: nameField(defaultsBlock, "speaker", speakers, at),
      style: nameField(defaultsBlock, "style", styles, at),
      styleWeight: numberField(defaultsBlock, "styleWeight", at),
      sdpRatio: numberField(defaultsBlock, "sdpRatio", at),
      noiseScale: numberField(defaultsBlock, "noiseScale", at),
      noiseScaleW: numberField(defaultsBlock, "noiseScaleW", at),
      lengthScale: numberField(defaultsBlock, "lengthScale", at),
    },
  };
};

/**
 * The model select: each model with the speakers it holds (`F1 — jvnv-F1-jp`), the manifest's
 * `defaultModel` marked "(default)". In jvnv one model is one speaker, so choosing the model is
 * choosing the voice.
 */
export const speakerModelChoices = (
  manifest: Manifest,
): readonly { readonly value: string; readonly label: string }[] =>
  Object.keys(manifest.models).map((model) => {
    const speakers = readSbv2Config(manifest, model).speakers.join(", ");
    return {
      value: model,
      label: `${model} — ${speakers}${model === manifest.defaultModel ? " (default)" : ""}`,
    };
  });

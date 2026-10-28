/**
 * Reader of Wan2.1's text embedding asset (stage 1 = precomputed embeddings — ADR 0118 decision 4).
 *
 * The asset is one safetensors file written by the recipe `tools/export-recipes/wan/text_embeds.py`:
 *
 * - Tensors: one per prompt (name = prompt name, `F32`, `[L_valid, width]` — umT5's output rows cut
 *   to the valid length; `width` is the last dimension of the DiT's `encoder_hidden_states`).
 * - Metadata: JSON under the single key {@link WAN_TEXT_EMBEDS_METADATA_KEY} (`prompts` holds each
 *   prompt's name, role, original text, normalized string and token count; the other fields
 *   (provenance, version) are history and are not read here).
 *
 * The accepted set is the metadata's prompt texts themselves: `generate({ prompt })` accepts only a
 * row whose original text or normalized string matches exactly, and anything outside the set fails
 * loudly (decision 4 — stage 2 (umT5 i8) only widens the set and does not break the API).
 *
 * MUST: modules are side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import { parseSafetensors } from "@karume/runtime";

/** The only metadata key (the recipe's `METADATA_KEY`). */
export const WAN_TEXT_EMBEDS_METADATA_KEY = "karume.wan.text_embeds";

/** The prompt's role (`negative` is the default for CFG's uncond side — the official `sample_neg_prompt`). */
export type PromptRole = "positive" | "negative";

/** One row of the accepted set (the asset's metadata — without the embedding values). */
export type Prompt = {
  /** The asset's tensor name (such as `boxing-cats`). */
  readonly name: string;
  readonly role: PromptRole;
  /** The source's original text (verbatim, including line breaks and full-width punctuation). */
  readonly prompt: string;
  /** The string after upstream's `prompt_clean` (ftfy → HTML unescape → whitespace collapsing). */
  readonly normalized: string;
};

/** The embedding of one prompt (row-major `[tokens, width]`, f32). */
export type WanTextEmbedding = Prompt & {
  /** The valid length `L_valid` (the length over which the tokenizer's mask is 1). */
  readonly tokens: number;
  readonly data: Float32Array<ArrayBuffer>;
};

/** The whole embedding asset (checked). */
export type WanTextEmbeds = {
  /** The width of one row (umT5's output dimension). */
  readonly width: number;
  /** In the metadata's order. */
  readonly entries: readonly WanTextEmbedding[];
};

/** The embedding asset breaks this module's conventions (an asset inconsistency — not an input error). */
export class WanTextEmbedsError extends Error {
  override readonly name = "WanTextEmbedsError";
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const textField = (entry: Record<string, unknown>, key: string, where: string): string => {
  const value = entry[key];
  if (typeof value !== "string") throw new WanTextEmbedsError(`${where}: '${key}' is not a string`);
  return value;
};

/**
 * Reads and checks the embedding asset (safetensors).
 *
 * MUST: check the whole correspondence between metadata and tensors (the name sets match, the row
 * counts match the metadata's token counts, the width is the same on every row). If any of it is
 * off, "generating with another prompt's embedding" passes silently.
 * MUST: reject an asset where an original text / normalized string hits two rows (which embedding to
 * use would be undecided).
 */
export const parseWanTextEmbeds = (buffer: ArrayBuffer): WanTextEmbeds => {
  const file = parseSafetensors(buffer);
  const keys = [...file.metadata.keys()];
  if (keys.length !== 1 || keys[0] !== WAN_TEXT_EMBEDS_METADATA_KEY) {
    throw new WanTextEmbedsError(
      `Metadata keys [${keys.join(", ")}] are not the single key '${WAN_TEXT_EMBEDS_METADATA_KEY}'`,
    );
  }
  let meta: unknown;
  try {
    meta = JSON.parse(file.metadata.get(WAN_TEXT_EMBEDS_METADATA_KEY) as string);
  } catch (cause) {
    throw new WanTextEmbedsError("Metadata cannot be read as JSON", { cause });
  }
  if (!isRecord(meta) || !Array.isArray(meta.prompts) || meta.prompts.length === 0) {
    throw new WanTextEmbedsError("Metadata has no 'prompts' array (with at least 1 row)");
  }
  const names = new Set<string>();
  const texts = new Map<string, string>();
  let width: number | undefined;
  const entries = meta.prompts.map((raw, index): WanTextEmbedding => {
    const where = `prompts[${index}]`;
    if (!isRecord(raw)) throw new WanTextEmbedsError(`${where} is not an object`);
    const name = textField(raw, "name", where);
    const role = textField(raw, "role", where);
    if (role !== "positive" && role !== "negative") {
      throw new WanTextEmbedsError(`${where}: role '${role}' is neither positive nor negative`);
    }
    const prompt = textField(raw, "prompt", where);
    const normalized = textField(raw, "normalized", where);
    const tokens = raw.tokens;
    if (typeof tokens !== "number" || !Number.isInteger(tokens) || tokens < 1) {
      throw new WanTextEmbedsError(
        `${where}: 'tokens' is not a positive integer (${String(tokens)})`,
      );
    }
    if (names.has(name)) throw new WanTextEmbedsError(`${where}: duplicate name '${name}'`);
    names.add(name);
    for (const text of new Set([prompt, normalized])) {
      const owner = texts.get(text);
      if (owner !== undefined) {
        throw new WanTextEmbedsError(
          `${where}: the text of '${name}' is the same as '${owner}' (the row is ambiguous)`,
        );
      }
      texts.set(text, name);
    }
    const view = file.tensors.get(name);
    if (view === undefined) throw new WanTextEmbedsError(`${where}: tensor '${name}' is missing`);
    if (view.dtype !== "F32" || view.shape.length !== 2 || view.shape[0] !== tokens) {
      throw new WanTextEmbedsError(
        `Tensor '${name}' is not F32 [${tokens}, width] (${view.dtype} [${view.shape.join(",")}])`,
      );
    }
    width ??= view.shape[1];
    if (view.shape[1] !== width || width < 1) {
      throw new WanTextEmbedsError(
        `Tensor '${name}' width ${view.shape[1]} differs from the other rows' ${width}`,
      );
    }
    const data = new Float32Array(file.buffer, view.byteOffset, tokens * width);
    // MUST: check finiteness once at load time (not again in the padding of every generation). The
    // GPU path rejects non-finite umT5 output immediately, so without this check a broken embedding
    // would fail only after the DiT is loaded and run, as non-finite latents, blaming the wrong
    // stage.
    for (let at = 0; at < data.length; at += 1) {
      if (!Number.isFinite(data[at])) {
        throw new WanTextEmbedsError(
          `Tensor '${name}' row ${Math.floor(at / width)}, column ${at % width} is non-finite` +
            ` (${data[at]})`,
        );
      }
    }
    return { name, role, prompt, normalized, tokens, data };
  });
  const extra = [...file.tensors.keys()].filter((name) => !names.has(name));
  if (extra.length > 0) {
    throw new WanTextEmbedsError(`Tensors [${extra.join(", ")}] are not in the metadata`);
  }
  return { width: width as number, entries };
};

/** The row whose original text or normalized string matches exactly (undefined when none). */
export const findWanTextEmbedding = (
  embeds: WanTextEmbeds,
  text: string,
): WanTextEmbedding | undefined =>
  embeds.entries.find((entry) => entry.prompt === text || entry.normalized === text);

/**
 * Zero-pads past the valid length into the DiT's `encoder_hidden_states [1, rows, width]` (the same
 * as `new_zeros` in upstream's `_get_t5_prompt_embeds` — the host pads. Decision 4).
 */
export const padWanTextEmbedding = (
  entry: WanTextEmbedding,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (entry.data.length !== entry.tokens * width) {
    throw new WanTextEmbedsError(
      `'${entry.name}' width differs from the DiT context width ${width}`,
    );
  }
  if (entry.tokens > rows) {
    throw new WanTextEmbedsError(
      `'${entry.name}' valid length ${entry.tokens} exceeds the context row count ${rows}`,
    );
  }
  const padded = new Float32Array(rows * width);
  padded.set(entry.data);
  return padded;
};

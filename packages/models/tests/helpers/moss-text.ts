/**
 * What the MOSS-Image prompt-form tests share: the Qwen3.5 text encoder and its tokenizer as the
 * distribution ships them, the opened text-encoder state, and the fixed prompts of upstream's
 * text-embedding asset with the token ids upstream fed its encoder (the asset's own metadata —
 * `tools/export-recipes/moss/text_embeds.py`).
 *
 * Where the assets come from (ADR 0139): the `text_encoder` container and the `tokenizer` asset are
 * read from the mirror `models/karume-moss-image/` through its manifest (model `sft`); the
 * text-embedding asset is a golden (`./moss-dit.ts`'s {@link MOSS_TEXT_EMBEDS}).
 */

import { parseSafetensors } from "@karume/runtime";
import type { ModelComponent } from "../../src/hub/components.ts";
import { mossTextEncoderContract, type MossTextEncoderState } from "../../src/moss/text-stage.ts";
import type { MossTextEmbeddings } from "../../src/moss/request.ts";
import { parseQwen3_5TokenizerAsset, Qwen3_5Tokenizer } from "../../src/qwen3_5/tokenizer.ts";
import { readMirrorAsset } from "./mirror.ts";
import {
  assertTestAssetsPresent,
  componentOf,
  floatsOf,
  MOSS_MIRROR,
  MOSS_TEXT_EMBEDS,
  openMossMirrorContainer,
  readBuffer,
} from "./moss-dit.ts";

/**
 * The series name of the text encoder. MUST stay verbatim: the sha256 case IDs of the encoder rows
 * spell it (ADR 0139 decision 7 — `qwen3_5.json`, `e2e_moss_prompt_test.ts`).
 */
export const MOSS_TEXT_SERIES_NAME = "qwen3.5-9b-text-moss-image-sft-i8-dyn";
const COMPONENT = "text_encoder";

/** The tokenizer over the mirror's `tokenizer` asset (`sft/tokenizer/tokenizer.json`). */
export const openMossTokenizer = async (): Promise<Qwen3_5Tokenizer> =>
  new Qwen3_5Tokenizer(
    parseQwen3_5TokenizerAsset(
      JSON.parse(
        new TextDecoder().decode(await readMirrorAsset(MOSS_MIRROR, "tokenizer", { model: "sft" })),
      ),
    ),
  );

/**
 * Opens the mirror's text encoder (the descriptors only — the weights are read when a Session is
 * built) and settles its contract as a generation's state carries it.
 */
export const openMossTextEncoder = async (): Promise<
  MossTextEncoderState & { readonly component: ModelComponent }
> => {
  const component = componentOf(await openMossMirrorContainer(COMPONENT), COMPONENT);
  const tokenizer = await openMossTokenizer();
  return { component, tokenizer, contract: mossTextEncoderContract(component, tokenizer) };
};

/** One fixed prompt of upstream's asset: its text, the ids upstream encoded and its rows. */
export type MossUpstreamPrompt = {
  readonly name: string;
  readonly role: "positive" | "negative";
  readonly text: string;
  readonly ids: readonly number[];
  readonly embeddings: MossTextEmbeddings;
};

const METADATA_KEY = "karume.moss.text_embeds";

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * The fixed prompts of upstream's text-embedding asset, in the asset's order (fails loudly on a
 * metadata layout other than `moss.text_embeds` writes, and on an asset absent from a complete fetch
 * of the pinned test-asset dataset — `testAssetsRunnable`).
 */
export const readMossUpstreamPrompts = async (): Promise<readonly MossUpstreamPrompt[]> => {
  assertTestAssetsPresent([MOSS_TEXT_EMBEDS]);
  const file = parseSafetensors(await readBuffer(MOSS_TEXT_EMBEDS));
  const raw = file.metadata.get(METADATA_KEY);
  if (raw === undefined) throw new Error(`the text-embedding asset has no '${METADATA_KEY}'`);
  const metadata: unknown = JSON.parse(raw);
  const prompts = isRecord(metadata) ? metadata["prompts"] : undefined;
  if (!Array.isArray(prompts)) throw new Error(`'${METADATA_KEY}' holds no prompt list`);
  return prompts.map((entry: unknown, index) => {
    const where = `${METADATA_KEY}.prompts[${index}]`;
    if (!isRecord(entry)) throw new Error(`${where} is not an object`);
    const { name, role, prompt, token_ids: ids } = entry;
    if (typeof name !== "string" || typeof prompt !== "string") {
      throw new Error(`${where} has no name or prompt string`);
    }
    if (role !== "positive" && role !== "negative") throw new Error(`${where}: role ${role}`);
    if (!Array.isArray(ids) || !ids.every((id) => Number.isSafeInteger(id))) {
      throw new Error(`${where}: token_ids are not integers`);
    }
    const view = file.tensors.get(name);
    if (view === undefined) throw new Error(`the text-embedding asset has no tensor '${name}'`);
    return {
      name,
      role,
      text: prompt,
      ids: ids.map(Number),
      embeddings: { data: floatsOf(file, name, "text embeds").slice(), shape: view.shape },
    };
  });
};

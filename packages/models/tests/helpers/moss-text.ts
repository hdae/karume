/**
 * What the MOSS-Image prompt-form tests share: where the Qwen3.5 text series and its tokenizer asset
 * live, the commands that write them, the opened text-encoder state, and the fixed prompts of
 * upstream's text-embedding asset with the token ids upstream fed its encoder (the asset's own
 * metadata — `tools/export-recipes/moss/text_embeds.py`).
 *
 * The caller opens the series itself: there is no manifest or distribution for the text encoder yet
 * (ADR 0138 decision 12), as for the DiT and VAE series (`./moss-dit.ts`, `./moss-vae.ts`).
 */

import { readTextFile } from "@karume/test-support/fs";
import { parseSafetensors } from "@karume/runtime";
import {
  modelPresent,
  openSeriesContainer,
} from "../../../runtime/tests/helpers/container-files.ts";
import type { ModelComponent } from "../../src/hub/components.ts";
import { mossTextEncoderContract, type MossTextEncoderState } from "../../src/moss/text-stage.ts";
import type { MossTextEmbeddings } from "../../src/moss/request.ts";
import { parseQwen3_5TokenizerAsset, Qwen3_5Tokenizer } from "../../src/qwen3_5/tokenizer.ts";
import { componentOf, filePresent, floatsOf, MOSS_TEXT_EMBEDS, readBuffer } from "./moss-dit.ts";
import { seriesGraph } from "./series-graphs.ts";

export const MOSS_TEXT_SERIES_NAME = "qwen3.5-9b-text-moss-image-sft-i8-dyn";
const COMPONENT = "text_encoder";
const OUTPUTS = new URL("../../../../outputs/", import.meta.url);
const TEXT_MODEL = new URL(`series/${MOSS_TEXT_SERIES_NAME}/${COMPONENT}/model.krm`, OUTPUTS);
export const MOSS_TEXT_TOKENIZER = new URL("series/qwen3_5-tokenizer/tokenizer.json", OUTPUTS);

const QWEN3_5_RECIPE =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' --inexact python -m";
export const MOSS_GENERATE_TEXT_SERIES = `${QWEN3_5_RECIPE} qwen3_5.export write --storage i8`;
export const MOSS_GENERATE_TOKENIZER = `${QWEN3_5_RECIPE} qwen3_5.tokenizer`;

/** The text tower's container and the tokenizer asset are both written. */
export const mossTextSeriesPresent = (): boolean =>
  modelPresent(TEXT_MODEL) && filePresent(MOSS_TEXT_TOKENIZER);

/** The tokenizer over the recipe's asset. */
export const openMossTokenizer = async (): Promise<Qwen3_5Tokenizer> =>
  new Qwen3_5Tokenizer(
    parseQwen3_5TokenizerAsset(JSON.parse(await readTextFile(MOSS_TEXT_TOKENIZER))),
  );

/**
 * Opens the text series (the descriptors only — the weights are read when a Session is built) and
 * settles its contract as a generation's state carries it.
 */
export const openMossTextEncoder = async (): Promise<
  MossTextEncoderState & { readonly component: ModelComponent }
> => {
  const component = componentOf(
    await openSeriesContainer(TEXT_MODEL),
    seriesGraph(MOSS_TEXT_SERIES_NAME, COMPONENT),
  );
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
 * metadata layout other than `moss.text_embeds` writes).
 */
export const readMossUpstreamPrompts = async (): Promise<readonly MossUpstreamPrompt[]> => {
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

/**
 * The distribution's tokenizer asset (parsed JSON) → the text tokenizer, chosen by the asset's
 * format.
 *
 * - **no `format` key** — the Unigram asset of the Small models (`./tokenizer.ts`)
 * - **`"karume-gemma-tokenizer/1"`** — the shared Gemma asset (`src/gemma/text/asset.ts`), written
 *   for v4-Large, whose text tokenizer is the Gemma BPE tokenizer of T5Gemma 2
 *   (`./gemma-tokenizer.ts`)
 *
 * MUST: any other `format` fails loudly. The recipe decides the kind from upstream's
 * `tokenizer.json` itself (`tokenizer_kind` of `tools/export-recipes/irodori/tokenizer_ref.py`) and
 * writes one of the two; reading an unknown format as either one would tokenize with the wrong
 * model, a silent wrong value.
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

import {
  GEMMA_TOKENIZER_ASSET_FORMAT,
  interpretGemmaTokenizerAsset,
} from "../../gemma/text/asset.ts";
import { asRecord } from "../../text/asset-gates.ts";
import { IrodoriGemmaTokenizer } from "./gemma-tokenizer.ts";
import type { IrodoriTextTokenizer } from "./text-tokenizer.ts";
import { IrodoriTokenizer, parseIrodoriTokenizerAsset } from "./tokenizer.ts";

/** Builds the text tokenizer the asset declares (`label` names the asset in the Unigram errors). */
export const parseIrodoriTextTokenizer = (
  raw: unknown,
  label: string = "tokenizer",
): IrodoriTextTokenizer => {
  const record = asRecord(raw, label);
  if (!Object.hasOwn(record, "format")) {
    return new IrodoriTokenizer(parseIrodoriTokenizerAsset(record, label));
  }
  const format = record["format"];
  if (format === GEMMA_TOKENIZER_ASSET_FORMAT) {
    return new IrodoriGemmaTokenizer(interpretGemmaTokenizerAsset(raw));
  }
  throw new Error(
    `${label}.format: unknown tokenizer asset format ${JSON.stringify(format)} (irodori reads the` +
      ` Unigram asset, which has no format key, or ${JSON.stringify(GEMMA_TOKENIZER_ASSET_FORMAT)})`,
  );
};

// Builds the Wan2.1 prompt layer (`WanPromptEncoder`) from the parity fixture
// `wan-text/parity.json` (written by the recipe `tools/export-recipes/wan/umt5_tokenizer.py` from
// transformers 5.14.1 and ftfy 6.3.1).
//
// The fixture's vocabulary is a **subset** (substrings of the fragments of every case), so only the
// strings of the fixture's cases can be looked up — the 4 fixed prompts (exactly `FIXED_PROMPTS` of
// the recipe `wan/prompts.py`) and the boundary and random cases. The vehicle for the pipeline's
// host tests to run the entry of the GPU path and the text stage without reading the 8 MB asset.
//
// The tables are built the same way as in `wan_text_tokenizer_test.ts` (that file is the source of
// truth for parity; this is only a reader).

import { readTextFile } from "@karume/test-support/fs";
import { parseCodeRanges } from "../../src/text/code-ranges.ts";
import type { T5Tables, T5VocabEntry } from "../../src/text/t5-tokenizer.ts";
import { parsePromptCleanTables } from "../../src/wan/text/prompt-clean.ts";
import { WanPromptEncoder } from "../../src/wan/text/tokenizer.ts";

/**
 * One case of the fixture (when accepted, the upstream preprocessing output and the id sequence;
 * when rejected, the reason).
 */
export type WanParityCase = {
  readonly id: string;
  readonly why: string;
  readonly text: string;
  readonly cleaned?: string;
  readonly ids?: readonly number[];
  readonly reject?: string;
};

type WanParityFixture = {
  readonly maxLength: number;
  readonly t5: {
    readonly addedTokens: [string, number][];
    readonly unkId: number;
    readonly eosId: number;
    readonly minScore: number;
    readonly maxTokenLength: number;
    readonly space: unknown;
    readonly vocab: [string, number, number][];
  };
  readonly promptClean: unknown;
  readonly cases: readonly WanParityCase[];
};

const fixture = JSON.parse(
  await readTextFile(new URL("../fixtures/wan-text/parity.json", import.meta.url)),
) as WanParityFixture;

/** Every case of the fixture (in generation order). */
export const wanParityCases: readonly WanParityCase[] = fixture.cases;

/**
 * Looks up a fixture case by id (fails loudly when missing — a case removed by regeneration does
 * not pass silently).
 */
export const wanParityCase = (id: string): WanParityCase => {
  const found = fixture.cases.find((entry) => entry.id === id);
  if (found === undefined) throw new Error(`wan-text/parity.json has no case '${id}'`);
  return found;
};

/**
 * The prompt layer built from the fixture's tables (`maxLength` is the fixture's value = upstream's
 * 512).
 */
export const wanParityEncoder = (): WanPromptEncoder => {
  const t5: T5Tables = {
    vocab: new Map<string, T5VocabEntry>(
      fixture.t5.vocab.map(([token, id, score]) => [token, { id, score }]),
    ),
    minScore: fixture.t5.minScore,
    maxTokenLength: fixture.t5.maxTokenLength,
    unkId: fixture.t5.unkId,
    eosId: fixture.t5.eosId,
    addedTokens: new Map(fixture.t5.addedTokens),
    space: parseCodeRanges(fixture.t5.space, "space"),
    maxLength: fixture.maxLength,
  };
  return new WanPromptEncoder({
    t5,
    promptClean: parsePromptCleanTables(fixture.promptClean, "promptClean"),
  });
};

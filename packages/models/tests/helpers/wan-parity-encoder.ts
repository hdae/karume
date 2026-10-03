// Wan2.1 のプロンプト層（`WanPromptEncoder`）を、パリティ用フィクスチャ `wan-text/parity.json`（recipe
// `tools/export-recipes/wan/umt5_tokenizer.py` が transformers 5.14.1 と ftfy 6.3.1 から焼く）から組む読み口。
//
// フィクスチャの語彙は**部分集合**（全ケースの断片の部分文字列）なので、引けるのはフィクスチャのケースの
// 文字列だけ — 固定 4 本（recipe `wan/prompts.py` の `FIXED_PROMPTS` そのもの）と境界・乱択のケース。
// パイプラインのホストテストが、資産 8 MB を読まずに GPU 経路の入口と text 段を回すための器。
//
// 表の組み方は `wan_text_tokenizer_test.ts` と同じ（あちらがパリティの正本で、ここは読み口だけ）。

import { parseCodeRanges } from "../../src/text/code-ranges.ts";
import type { T5Tables, T5VocabEntry } from "../../src/text/t5-tokenizer.ts";
import { parsePromptCleanTables } from "../../src/wan/text/prompt-clean.ts";
import { WanPromptEncoder } from "../../src/wan/text/tokenizer.ts";

/** フィクスチャのケース 1 本（受理なら上流の前処理の出力と id 列、拒否なら理由）。 */
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
  await Deno.readTextFile(new URL("../fixtures/wan-text/parity.json", import.meta.url)),
) as WanParityFixture;

/** フィクスチャの全ケース（生成の順）。 */
export const wanParityCases: readonly WanParityCase[] = fixture.cases;

/** フィクスチャのケースを id で引く（無ければ fail loudly — 再生成でケースが消えたことを黙って通さない）。 */
export const wanParityCase = (id: string): WanParityCase => {
  const found = fixture.cases.find((entry) => entry.id === id);
  if (found === undefined) throw new Error(`wan-text/parity.json にケース '${id}' が無い`);
  return found;
};

/** フィクスチャの表で組んだプロンプト層（`maxLength` はフィクスチャの値 = 上流の 512）。 */
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

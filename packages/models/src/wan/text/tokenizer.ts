/**
 * Wan2.1 のプロンプト層 — 前処理（`prompt_clean` の鏡像）→ umT5 のトークナイザ → 受理集合の検査
 * （ADR 0119 決定 1・2・4）。
 *
 * 経路の正本は pin した transformers 5.14.1 の `AutoTokenizer`（Wan の `tokenizer/`）で、形は T5 の
 * 共通層（`src/text/t5-tokenizer.ts` — 正規化なし・WhitespaceSplit → Metaspace → Unigram →
 * 末尾 `</s>`）そのもの。Wan が上流と違えるのは「黙って id を作らない」所だけで、どれも
 * `ModelInputError`（入力起因）で拒む:
 *
 * - **語彙外の文字**（決定 1）— 未知の id は transformers 5（id 2 = `<s>`）と `tokenizer.json`・
 *   transformers 4.x（id 3 = `<unk>`）で割れる。
 * - **本文中の追加語彙**（`</s>`・`<extra_id_*>` など）と**空白の直後の `▁`（U+2581）** —
 *   transformers 5 は空白で先に割り、`tokenizer.json` は空白を `▁` に置き換えてから割るので、
 *   この 2 つの形では id 列が版で割れる（`a </s> b` は `[289, 1, 748, 1]` と
 *   `[289, 273, 1, 748, 1]`）。決定 1 の「どの版の参照とも同じ id 列」を守るため拒む
 *   （recipe `wan/umt5_tokenizer.py` の docstring に実測）。
 * - **512 トークン超**（決定 4）— 上流は黙って切り詰める。
 * - **1 トークン**（空・空白だけで `</s>` だけ）— `torch.export` の 0 / 1 特殊化を避けた下限 2。
 *
 * 受理した入力の id 列は、transformers 5.14.1 と `tokenizer.json` の 2 経路のどちらとも一致する
 * （recipe が fixture の生成時に確かめ、`wan_text_tokenizer_test.ts` が両方と突き合わせる）。
 *
 * 資産（系列 `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json` — 配布形では Wan の manifest の資産
 * `umt5_tokenizer`）は
 * 語彙・スコア・追加語彙・空白集合と前処理の表（`promptClean`）を 1 本に持つ。同じ recipe の
 * 1 回の実行で焼くので、トークナイザと前処理が別々に古びない。I/O は持たない（取得は hub の責務）。
 */

import { ModelInputError } from "../../errors.ts";
import { splitAddedTokens } from "../../text/added-tokens.ts";
import { asRecord, asString } from "../../text/asset-gates.ts";
import {
  METASPACE,
  parseT5Tables,
  type T5Tables,
  T5UnigramTokenizer,
} from "../../text/t5-tokenizer.ts";
import { cleanPrompt, parsePromptCleanTables, type PromptCleanTables } from "./prompt-clean.ts";

/** 資産の形式の版（recipe の `ASSET_FORMAT`）。 */
export const WAN_TOKENIZER_FORMAT = "karume-wan-umt5-tokenizer/1";

/**
 * 受理集合の下限。umT5 のグラフの有効長 L は `torch.export` の 0 / 1 特殊化を避けて 2 以上
 * （ADR 0119 決定 4 — anima の `PROMPT_MIN_TOKENS` と同じ理由）。
 */
export const WAN_PROMPT_MIN_TOKENS = 2;

/** Wan のトークナイザ資産を読んだもの。 */
export type WanTokenizerAssets = {
  readonly t5: T5Tables;
  readonly promptClean: PromptCleanTables;
};

/** 空白の直後の Metaspace（2 経路で断片化が割れる形）。 */
const SPACE_METASPACE = " " + METASPACE;

/** プロンプト 1 本を umT5 の id 列にする。 */
export class WanPromptEncoder {
  readonly #promptClean: PromptCleanTables;
  readonly #t5: T5UnigramTokenizer;
  readonly #added: string[];
  readonly #maxLength: number;

  constructor(assets: WanTokenizerAssets) {
    this.#promptClean = assets.promptClean;
    this.#t5 = new T5UnigramTokenizer(assets.t5, { unknown: "reject", overflow: "reject" });
    this.#added = [...assets.t5.addedTokens.keys()];
    this.#maxLength = assets.t5.maxLength;
  }

  /** id 列の最大長（`</s>` を含む — 上流の `max_sequence_length`）。 */
  get maxLength(): number {
    return this.#maxLength;
  }

  /** 前処理だけ（上流 `prompt_clean` の出力と同じ文字列）。 */
  clean(prompt: string, label: string = "プロンプト"): string {
    return cleanPrompt(this.#promptClean, prompt, label);
  }

  /**
   * 前処理 → 符号化 → 受理集合の検査。i32 の id 列（末尾は `</s>`・長さは 2〜`maxLength`）。
   *
   * 拒否の順は recipe の fixture の生成と同じ: 前処理 → 追加語彙 → 空白の直後の `▁` →
   * 語彙外 / 上限超え（先頭から積んで先に当たった方）→ 下限割れ。
   */
  encode(prompt: string, label: string = "プロンプト"): Int32Array<ArrayBuffer> {
    const cleaned = this.clean(prompt, label);
    for (const chunk of splitAddedTokens(cleaned, this.#added)) {
      if (chunk.added) {
        throw new ModelInputError(
          `${label}: 特殊トークン ${JSON.stringify(chunk.text)} を含む — 本文中の特殊トークンは` +
            "上流の版で id 列が割れるので受けない（その綴りを除く）",
        );
      }
    }
    if (cleaned.includes(SPACE_METASPACE)) {
      throw new ModelInputError(
        `${label}: 空白の直後に ${METASPACE}（U+2581）がある — SentencePiece の空白の印で、` +
          "上流の版で断片化が割れるので受けない（空白を詰めるか ▁ を除く）",
      );
    }
    const ids = this.#t5.encode(cleaned, label);
    if (ids.length < WAN_PROMPT_MIN_TOKENS) {
      throw new ModelInputError(
        `${label}: id 列が ${ids.length} トークン（最低 ${WAN_PROMPT_MIN_TOKENS}）— 空文字や` +
          "空白だけのプロンプトは受けない。1 語以上入れる",
      );
    }
    return Int32Array.from(ids);
  }
}

/**
 * 資産 JSON（`JSON.parse` 済み）を表に変換する。外部境界なので構造を検査してから使う
 * （`fromPretrained(ref)` は任意の repo を指せる — `src/text/asset-gates.ts` の前提）。
 */
export const parseWanTokenizerAsset = (
  raw: unknown,
  label: string = "tokenizer",
): WanTokenizerAssets => {
  const obj = asRecord(raw, label);
  const format = asString(obj["format"], `${label}.format`);
  if (format !== WAN_TOKENIZER_FORMAT) {
    // 知らない版を黙って読まない（欄の意味が変わっていても例外にならず id 列だけが変わる）。
    throw new Error(`${label}: 形式 ${format} を読めない（${WAN_TOKENIZER_FORMAT} だけを読む）`);
  }
  return {
    t5: parseT5Tables(obj, label),
    promptClean: parsePromptCleanTables(obj["promptClean"], `${label}.promptClean`),
  };
};

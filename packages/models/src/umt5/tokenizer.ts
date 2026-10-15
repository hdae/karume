/**
 * umT5-XXL のトークナイザ — 符号化と受理集合の検査（ADR 0119 決定 1・4 の規則を、Wan の前処理なしで使う形）。
 *
 * 経路の正本は pin した transformers 5.14.1 の `AutoTokenizer`（本家 `google/umt5-xxl` の `tokenizer.json` と
 * `tokenizer_config.json`）で、形は T5 の共通層（`src/text/t5-tokenizer.ts` — 正規化なし・WhitespaceSplit →
 * Metaspace → Unigram → 末尾 `</s>`）そのもの。上流と違えるのは「黙って id を作らない」所だけで、どれも
 * `ModelInputError`（入力起因）で拒む。拒否の順（recipe `wan/umt5_xxl_tokenizer.py` の fixture の生成と同じ）:
 *
 * 1. **本文中の追加語彙**（`</s>`・`<extra_id_*>` など）と 2. **空白の直後の `▁`（U+2581）** —
 *    transformers 5 は空白で先に割り、`tokenizer.json` は空白を `▁` に置き換えてから割るので、この 2 つの形では
 *    id 列が版で割れる（`a </s> b` は `[289, 1, 748, 1]` と `[289, 273, 1, 748, 1]`）。
 * 3. **U+0020 以外の空白と末尾の U+0020** — transformers 5 は区切りとして捨て、`tokenizer.json` は文字として
 *    残す（`a\tb` は `[289, 748, 1]` と `[289, 3, 485, 1]`・`a ` は `[289, 1]` と `[289, 273, 1]`）。空白の集合は
 *    資産の `space`（WhitespaceSplit の空白集合）。先頭の U+0020 と語の間の連続した U+0020 は割れない
 *    （`tokenizer.json` の normalizer が畳む）ので受ける。trim や畳み込みは足さない（足すと第 2 の前処理になる）。
 * 4. **語彙外の文字**（決定 1）と **512 トークン超**（決定 4 — 上流は黙って切り詰める）— 先頭から積んで先に
 *    当たった方。語彙外の id は transformers 5（id 2 = `<s>`）と `tokenizer.json`・transformers 4.x（id 3 =
 *    `<unk>`）で割れる。
 * 5. **2 トークン未満**（空文字で `</s>` だけ）— グラフの有効長の下限（`./relative-position.ts` の
 *    {@link UMT5_MIN_TOKENS}）。
 *
 * 受理した入力の id 列は、transformers 5.14.1 と `tokenizer.json` の 2 経路のどちらとも一致する（recipe が
 * fixture の生成時に確かめ、`umt5_tokenizer_test.ts` が両方と突き合わせる）。
 *
 * Wan（`src/wan/text/tokenizer.ts` の `WanPromptEncoder`）は上流の前処理 `prompt_clean` の鏡像を通した文字列を
 * これに渡す。前処理は空白を U+0020 1 個へ畳んで両端を落とすので、3 の拒否は Wan の経路では起きない（Wan の
 * 資産の空白の表は畳み込みの集合と同じ — `umt5_tokenizer_test.ts` が見る）。
 *
 * 資産（形式 {@link UMT5_TOKENIZER_FORMAT} — 配布形 `karume-umt5-xxl` のモデル単位の資産 `tokenizer`）は語彙・
 * スコア・追加語彙・空白集合・最大長を 1 本に持つ（書き手は recipe `wan/umt5_xxl_tokenizer.py`）。I/O は持たない
 * （取得は hub の責務）。
 *
 * MUST: モジュール副作用ゼロ（CLAUDE.md）。
 */

import { ModelInputError } from "../errors.ts";
import { type AddedTokenMatcher, createAddedTokenMatcher } from "../text/added-tokens.ts";
import { asRecord, asString } from "../text/asset-gates.ts";
import { assertEncodableText } from "../text/code-points.ts";
import { type CodeRanges, inCodeRanges } from "../text/code-ranges.ts";
import {
  METASPACE,
  parseT5Tables,
  type T5Tables,
  T5UnigramTokenizer,
} from "../text/t5-tokenizer.ts";
import { UMT5_MIN_TOKENS } from "./relative-position.ts";

/** 資産の形式の版（recipe `wan/umt5_xxl_tokenizer.py` の `ASSET_FORMAT`）。 */
export const UMT5_TOKENIZER_FORMAT = "karume-umt5-tokenizer/1";

/** 受理する唯一の空白（U+0020）。 */
const SPACE = " ";

/** 空白の直後の Metaspace（2 経路で断片化が割れる形）。 */
const SPACE_METASPACE = SPACE + METASPACE;

/** コードポイントの表示（`U+0009` — 拒否の文言で直す文字を名指しする）。 */
const codePointLabel = (cp: number): string =>
  `U+${cp.toString(16).toUpperCase().padStart(4, "0")}`;

/** 文字列 1 本を umT5 の id 列にする（受理集合の外は {@link ModelInputError}）。 */
export class Umt5Tokenizer {
  readonly #t5: T5UnigramTokenizer;
  readonly #added: AddedTokenMatcher;
  readonly #space: CodeRanges;
  readonly #maxLength: number;

  constructor(tables: T5Tables) {
    this.#t5 = new T5UnigramTokenizer(tables, { unknown: "reject", overflow: "reject" });
    this.#added = createAddedTokenMatcher([...tables.addedTokens.keys()]);
    this.#space = tables.space;
    this.#maxLength = tables.maxLength;
  }

  /** id 列の最大長（`</s>` を含む — 上流の `max_length`）。 */
  get maxLength(): number {
    return this.#maxLength;
  }

  /**
   * 符号化 → 受理集合の検査。i32 の id 列（末尾は `</s>`・長さは {@link UMT5_MIN_TOKENS}〜`maxLength`）。
   * 拒否の順はモジュール doc の 1〜5。`label` は拒否の文言の主語。
   */
  tokenize(text: string, label: string = "text"): Int32Array<ArrayBuffer> {
    assertEncodableText(text, label);
    for (const chunk of this.#added.split(text)) {
      if (chunk.added) {
        throw new ModelInputError(
          `${label}: 特殊トークン ${JSON.stringify(chunk.text)} を含む — 本文中の特殊トークンは` +
            "上流の版で id 列が割れるので受けない（その綴りを除く）",
        );
      }
    }
    if (text.includes(SPACE_METASPACE)) {
      throw new ModelInputError(
        `${label}: 空白の直後に ${METASPACE}（U+2581）がある — SentencePiece の空白の印で、` +
          "上流の版で断片化が割れるので受けない（空白を詰めるか ▁ を除く）",
      );
    }
    this.#assertPlainSpaces(text, label);
    const ids = this.#t5.encode(text, label);
    if (ids.length < UMT5_MIN_TOKENS) {
      throw new ModelInputError(
        `${label}: id 列が ${ids.length} トークン（最低 ${UMT5_MIN_TOKENS}）— 空文字や` +
          "空白だけのプロンプトは受けない。1 語以上入れる",
      );
    }
    return Int32Array.from(ids);
  }

  /** U+0020 以外の空白と末尾の U+0020 を拒む（モジュール doc の 3）。 */
  #assertPlainSpaces(text: string, label: string): void {
    for (const ch of text) {
      const cp = ch.codePointAt(0) as number;
      if (ch !== SPACE && inCodeRanges(this.#space, cp)) {
        const found = codePointLabel(cp);
        throw new ModelInputError(
          `${label}: U+0020 以外の空白 ${found} を含む — 前処理をしないトークナイザでは` +
            "上流の版で id 列が割れるので受けない（U+0020 に置き換える）",
        );
      }
    }
    if (text.endsWith(SPACE)) {
      throw new ModelInputError(
        `${label}: 末尾が空白（U+0020）— 前処理をしないトークナイザでは上流の版で id 列が割れるので` +
          "受けない（末尾の空白を除く）",
      );
    }
  }
}

/**
 * 資産 JSON（`JSON.parse` 済み — 形式 {@link UMT5_TOKENIZER_FORMAT}）を T5 の表に変換する。外部境界なので
 * 構造を検査してから使う（`fromPretrained(ref)` は任意の repo を指せる — `src/text/asset-gates.ts` の前提）。
 *
 * 出所の欄（`source` / `versions`）は読まない — 書き手の門（recipe `wan/umt5_distribution.py`）が配布形を
 * 組む時に見る欄で、実行時の id 列は表だけで決まる（Wan の資産の読み口と同じ扱い）。
 */
export const parseUmt5TokenizerAsset = (raw: unknown, label: string = "tokenizer"): T5Tables => {
  const obj = asRecord(raw, label);
  const format = asString(obj["format"], `${label}.format`);
  if (format !== UMT5_TOKENIZER_FORMAT) {
    // 知らない版を黙って読まない（欄の意味が変わっていても例外にならず id 列だけが変わる）。
    throw new Error(`${label}: 形式 ${format} を読めない（${UMT5_TOKENIZER_FORMAT} だけを読む）`);
  }
  return parseT5Tables(obj, label);
};

/**
 * Wan2.1 のプロンプト層 — 前処理（`prompt_clean` の鏡像）→ umT5 のトークナイザ（受理集合の検査を含む）
 * （ADR 0119 決定 1・2・4）。
 *
 * トークナイザと受理集合の検査は単体の umT5 と共有の 1 本（`src/umt5/tokenizer.ts` の `Umt5Tokenizer` — 経路の
 * 正本・拒否の規則と順はそちらのモジュール doc）で、Wan はその前に上流の前処理を通すだけ。上流と違えるのは
 * 「黙って id を作らない」所で、どれも `ModelInputError`（入力起因）で拒む: 前処理が拒む入力
 * （`PromptCleanError` — `./prompt-clean.ts`）・語彙外の文字（決定 1）・本文中の追加語彙と空白の直後の `▁`
 * （U+2581 — 版で id 列が割れる形）・512 トークン超（決定 4）・1 トークン（空・空白だけで `</s>` だけ）。
 * 共有の検査の「U+0020 以外の空白と末尾の U+0020」は、前処理が空白を U+0020 1 個へ畳んで両端を落とすので
 * この経路では起きない。
 *
 * 受理した入力の id 列は、transformers 5.14.1 と `tokenizer.json` の 2 経路のどちらとも一致する
 * （recipe が fixture の生成時に確かめ、`wan_text_tokenizer_test.ts` が両方と突き合わせる）。
 *
 * 資産（系列 `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json` — 配布形では Wan の manifest の資産
 * `umt5_tokenizer`）は
 * 語彙・スコア・追加語彙・空白集合と前処理の表（`promptClean`）を 1 本に持つ。同じ recipe の
 * 1 回の実行で焼くので、トークナイザと前処理が別々に古びない。I/O は持たない（取得は hub の責務）。
 */

import { asRecord, asString } from "../../text/asset-gates.ts";
import { parseT5Tables, type T5Tables } from "../../text/t5-tokenizer.ts";
import { UMT5_MIN_TOKENS } from "../../umt5/relative-position.ts";
import { Umt5Tokenizer } from "../../umt5/tokenizer.ts";
import { cleanPrompt, parsePromptCleanTables, type PromptCleanTables } from "./prompt-clean.ts";

/** 資産の形式の版（recipe の `ASSET_FORMAT`）。 */
export const WAN_TOKENIZER_FORMAT = "karume-wan-umt5-tokenizer/1";

/**
 * 受理集合の下限（umT5 のグラフの有効長の下限 `UMT5_MIN_TOKENS` の別名 — ADR 0119 決定 4。値の正本は
 * `src/umt5/relative-position.ts`）。
 */
export const WAN_PROMPT_MIN_TOKENS = UMT5_MIN_TOKENS;

/** Wan のトークナイザ資産を読んだもの。 */
export type WanTokenizerAssets = {
  readonly t5: T5Tables;
  readonly promptClean: PromptCleanTables;
};

/** プロンプト 1 本を umT5 の id 列にする。 */
export class WanPromptEncoder {
  readonly #promptClean: PromptCleanTables;
  readonly #tokenizer: Umt5Tokenizer;

  constructor(assets: WanTokenizerAssets) {
    this.#promptClean = assets.promptClean;
    this.#tokenizer = new Umt5Tokenizer(assets.t5);
  }

  /** id 列の最大長（`</s>` を含む — 上流の `max_sequence_length`）。 */
  get maxLength(): number {
    return this.#tokenizer.maxLength;
  }

  /** 前処理だけ（上流 `prompt_clean` の出力と同じ文字列）。 */
  clean(prompt: string, label: string = "プロンプト"): string {
    return cleanPrompt(this.#promptClean, prompt, label);
  }

  /**
   * 前処理 → 符号化 → 受理集合の検査。i32 の id 列（末尾は `</s>`・長さは 2〜`maxLength`）。
   *
   * 拒否の順は recipe の fixture の生成と同じ: 前処理 → 追加語彙 → 空白の直後の `▁` →
   * 語彙外 / 上限超え（先頭から積んで先に当たった方）→ 下限割れ（前処理の後は共有の 1 本 —
   * `Umt5Tokenizer.tokenize`）。
   */
  encode(prompt: string, label: string = "プロンプト"): Int32Array<ArrayBuffer> {
    return this.#tokenizer.tokenize(this.clean(prompt, label), label);
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

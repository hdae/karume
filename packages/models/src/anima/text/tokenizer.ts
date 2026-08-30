/**
 * Anima のプロンプト層 — 2 つのトークナイザ資産の解釈と、プロンプト 1 本の符号化。
 *
 * Anima のテキスト層は**同じ文字列を 2 本のトークナイザに独立に通す**:
 *
 *   ① Qwen2 BPE（manifest の `tokenizer`）    → text_encoder（Qwen3）→ source_hidden_states
 *   ② T5 Unigram（manifest の `tokenizer_2`） → text_conditioner     → target_input_ids
 *
 * 配布形: 語彙・merges・正規化表は exporter が焼いた JSON（生の `tokenizer.json` 計 13.8MB
 * ではなく、実行に要る情報だけを抜いた計 4.6MB の形）。**取得は hub の責務**なので、ここは
 * バイト列を受けるだけで I/O を持たない（ブラウザでも Deno でも同じ経路）。
 */

import {
  asFiniteNumber,
  asPositiveInteger,
  assertCodePoint,
  assertUniqueLines,
  setUnique,
} from "../../text/asset-gates.ts";
import { parseCodeRanges } from "./code-ranges.ts";
import {
  type Qwen2Assets,
  type Qwen2CaseFold,
  type Qwen2CharClasses,
  Qwen2Tokenizer,
} from "./qwen2-tokenizer.ts";
import { parseSpmTables } from "./spm-normalizer.ts";
import { type T5Assets, T5Tokenizer, type T5VocabEntry } from "./t5-tokenizer.ts";

/** プロンプト 1 本の符号化結果（グラフ入力は i32 — ADR 0009）。 */
type AnimaPromptIds = {
  readonly qwenIds: Int32Array<ArrayBuffer>;
  readonly t5Ids: Int32Array<ArrayBuffer>;
};

/**
 * 受理集合の下限。エクスポータの `Dim("Tsrc", min=2)` / `Dim("Ttgt", min=2)` そのもの
 * （`torch.export` の 0/1 特殊化を避けるための下限で、1 トークンの入力は**グラフに食わせ
 * られない**）。空文字・空白だけのプロンプトは T5 が `</s>` だけの長さ 1 になり、1 文字の
 * プロンプトは Qwen2 が 1 トークンになる — どちらもここで落とす。
 */
export const PROMPT_MIN_TOKENS = 2;

/**
 * プロンプト長が受理集合に入っていることを検査する（違反は fail loudly）。
 *
 * 上限側は「切り詰めが効いていること」の検査でもある: T5 は `</s>` を足す前に切り詰めるので、
 * 順序を取り違えると `maxLength + 1` 個になる（shape エラーにならず conditioner の
 * 512 パディング検査まで沈黙する）。
 */
export const assertPromptTokenLengths = (
  label: string,
  qwenLength: number,
  t5Length: number,
  maxLength: number,
): void => {
  for (const [which, length] of [["Qwen2", qwenLength], ["T5", t5Length]] as const) {
    if (length < PROMPT_MIN_TOKENS) {
      throw new Error(
        `${label}の ${which} id 列が ${length} トークン（最低 ${PROMPT_MIN_TOKENS}）— ` +
          `空文字や空白だけのプロンプトは受理集合 Dim(min=${PROMPT_MIN_TOKENS}) の外。` +
          "1 語以上入れる。",
      );
    }
    if (length > maxLength) {
      throw new Error(`${label}の ${which} id 列が ${length} トークン（上限 ${maxLength}）`);
    }
  }
};

/** Anima のテキスト層が要する 2 つのトークナイザ。 */
export class AnimaTokenizers {
  readonly #qwen2: Qwen2Tokenizer;
  readonly #t5: T5Tokenizer;
  readonly #maxLength: number;

  constructor(qwen2Assets: Qwen2Assets, t5Assets: T5Assets) {
    if (qwen2Assets.maxLength !== t5Assets.maxLength) {
      // 片方だけ古い資産を掴んでいる状態。切り詰め長が違うと長いプロンプトで id 列の
      // 長さだけが食い違い、conditioner の 512 パディング検査まで表面化しない。
      throw new Error(
        `トークナイザ資産の maxLength が食い違う（Qwen2 ${qwen2Assets.maxLength} / ` +
          `T5 ${t5Assets.maxLength}）`,
      );
    }
    this.#qwen2 = new Qwen2Tokenizer(qwen2Assets);
    this.#t5 = new T5Tokenizer(t5Assets);
    this.#maxLength = qwen2Assets.maxLength;
  }

  /** 正本の呼び出しに合わせた切り詰め長（512）。 */
  get maxLength(): number {
    return this.#maxLength;
  }

  /**
   * プロンプト 1 本を両トークナイザに通す。長さが受理集合を外れたらここで落とす
   * （`label` はその際のメッセージに出す — 正 / ネガティブのどちらかが判る形にする）。
   */
  encode(prompt: string, label: string = "プロンプト"): AnimaPromptIds {
    const qwenIds = Int32Array.from(this.#qwen2.encode(prompt));
    const t5Ids = Int32Array.from(this.#t5.encode(prompt));
    assertPromptTokenLengths(label, qwenIds.length, t5Ids.length, this.#maxLength);
    return { qwenIds, t5Ids };
  }
}

const asRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null) throw new Error(`${label}: オブジェクトでない`);
  return value as Record<string, unknown>;
};

const asString = (value: unknown, label: string): string => {
  if (typeof value !== "string") throw new Error(`${label}: 文字列でない`);
  return value;
};

const asNumber = (value: unknown, label: string): number => {
  if (typeof value !== "number") throw new Error(`${label}: 数値でない`);
  return value;
};

/** i32 の上限（id は最終的に `Int32Array` へ書かれる）。 */
const MAX_ID = 2147483647;

/**
 * トークン id として受ける数値。
 *
 * MUST: `typeof number` だけで通さない。id は `encode` の結果が `Int32Array` へ写されるので、
 * 非整数は**黙って切り捨てられ**、i32 の範囲外は wrap する — どちらも「別のトークンを指す」
 * 沈黙誤値になり、グラフ側の embedding gather まで表面化しない。
 */
const asId = (value: unknown, label: string): number => {
  const id = asNumber(value, label);
  if (!Number.isInteger(id) || id < 0 || id > MAX_ID) {
    throw new Error(`${label}: トークン id が 0..${MAX_ID} の整数でない（${id}）`);
  }
  return id;
};

/** 語彙表の行を指す id（{@link asId} に加えて語彙の行数未満であること）。 */
const asVocabId = (value: unknown, label: string, vocabSize: number): number => {
  const id = asId(value, label);
  if (id >= vocabSize) {
    throw new Error(`${label}: トークン id ${id} が語彙の行数 ${vocabSize} 以上`);
  }
  return id;
};

const parseAddedTokens = (raw: unknown, label: string): Map<string, number> => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 配列でない`);
  const out = new Map<string, number>();
  for (const entry of raw) {
    if (!Array.isArray(entry) || entry.length !== 2) {
      throw new Error(`${label}: [文字列, id] でない`);
    }
    // NOTE: 語彙の行数では縛らない — Qwen2 の追加トークンは語彙表の**外**へ採番される
    // （実資産で 151643..151668 / 語彙 151643 行）。ここで見るのは i32 として健全なことだけ。
    setUnique(out, asString(entry[0], label), asId(entry[1], label), label);
  }
  return out;
};

const parseClasses = (raw: unknown, label: string): Qwen2CharClasses => {
  const obj = asRecord(raw, label);
  return {
    letter: parseCodeRanges(obj["letter"], `${label}.letter`),
    number: parseCodeRanges(obj["number"], `${label}.number`),
    space: parseCodeRanges(obj["space"], `${label}.space`),
  };
};

/** `(?i:)` の同一視表（`[cp, 接尾辞側の文字]` の対）。 */
const parseCaseFold = (raw: unknown, label: string): Qwen2CaseFold => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 配列でない`);
  const out = new Map<number, number>();
  for (const entry of raw) {
    if (
      !Array.isArray(entry) || entry.length !== 2 ||
      typeof entry[0] !== "number" || typeof entry[1] !== "number"
    ) {
      throw new Error(`${label}: [cp, cp] でない`);
    }
    assertCodePoint(entry[0], label);
    assertCodePoint(entry[1], label);
    setUnique(out, entry[0], entry[1], label);
  }
  return out;
};

/**
 * 行区切りの表を読み、行数が資産の申告と合っていることを確かめる。
 *
 * MUST: 件数を突き合わせる。`mergesText` が欠けても BPE は「見つからないので結合しない」
 * だけで**落ちない** — 分割が変わって id 列が静かに別物になる（`vocabText` 側は語彙に無い
 * トークンで落ちるが、こちらは沈黙誤値）。
 */
const parseLines = (
  raw: unknown,
  count: unknown,
  label: string,
  countLabel: string,
): string[] => {
  const lines = asString(raw, label).split("\n");
  const want = asNumber(count, countLabel);
  if (lines.length !== want) {
    throw new Error(
      `${label}: 行数 ${lines.length} が ${countLabel} の ${want} と違う（資産が壊れている）`,
    );
  }
  return lines;
};

/** Qwen2 の資産 JSON を資産表に変換する（外部境界なので構造を検査してから使う）。 */
const parseQwen2Asset = (raw: unknown, label: string = "tokenizer"): Qwen2Assets => {
  const obj = asRecord(raw, label);
  // 行番号 0-origin = id / rank。byte-level 語彙は改行も空白も含まない（emit 時に検査済み）。
  const vocabLines = parseLines(
    obj["vocabText"],
    obj["vocabCount"],
    `${label}.vocabText`,
    `${label}.vocabCount`,
  );
  assertUniqueLines(vocabLines, `${label}.vocabText`);
  const vocab = new Map<string, number>();
  for (const [id, token] of vocabLines.entries()) vocab.set(token, id);
  const mergesLines = parseLines(
    obj["mergesText"],
    obj["mergesCount"],
    `${label}.mergesText`,
    `${label}.mergesCount`,
  );
  assertUniqueLines(mergesLines, `${label}.mergesText`);
  const merges = new Map<string, number>();
  for (const [rank, line] of mergesLines.entries()) merges.set(line, rank);
  return {
    vocab,
    merges,
    addedTokens: parseAddedTokens(obj["addedTokens"], `${label}.addedTokens`),
    classes: parseClasses(obj["classes"], `${label}.classes`),
    caseFold: parseCaseFold(obj["caseFold"], `${label}.caseFold`),
    nfcSegments: parseCodeRanges(obj["nfcSegments"], `${label}.nfcSegments`),
    maxLength: asPositiveInteger(obj["maxLength"], `${label}.maxLength`),
  };
};

/** T5 の資産 JSON を資産表に変換する。 */
const parseT5Asset = (raw: unknown, label: string = "tokenizer_2"): T5Assets => {
  const obj = asRecord(raw, label);
  const tokens = asString(obj["vocabText"], `${label}.vocabText`).split("\n");
  const rawScores = obj["scores"];
  if (!Array.isArray(rawScores) || rawScores.length !== tokens.length) {
    throw new Error(`${label}: scores の長さが語彙数 ${tokens.length} と合わない`);
  }
  assertUniqueLines(tokens, `${label}.vocabText`);
  const vocab = new Map<string, T5VocabEntry>();
  let minScore = Number.POSITIVE_INFINITY;
  let maxTokenLength = 0;
  for (const [id, token] of tokens.entries()) {
    const score = asFiniteNumber(rawScores[id], `${label}.scores[${id}]`);
    vocab.set(token, { id, score });
    minScore = Math.min(minScore, score);
    maxTokenLength = Math.max(maxTokenLength, Array.from(token).length);
  }
  return {
    vocab,
    minScore,
    maxTokenLength,
    unkId: asVocabId(obj["unkId"], `${label}.unkId`, tokens.length),
    eosId: asVocabId(obj["eosId"], `${label}.eosId`, tokens.length),
    addedTokens: parseAddedTokens(obj["addedTokens"], `${label}.addedTokens`),
    space: parseCodeRanges(obj["space"], `${label}.space`),
    normalizer: parseSpmTables(obj["normalizer"], `${label}.normalizer`),
    maxLength: asPositiveInteger(obj["maxLength"], `${label}.maxLength`),
  };
};

const decodeJson = (bytes: Uint8Array, label: string): unknown => {
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error(`${label}: UTF-8 として読めない`, { cause });
  }
  try {
    return JSON.parse(text);
  } catch (cause) {
    throw new Error(`${label}: JSON として読めない`, { cause });
  }
};

/**
 * 取得済みバイト列（manifest の `tokenizer` / `tokenizer_2`）からトークナイザ 2 本を組む。
 * I/O を持たない — 取得は hub の責務。
 */
export const createTokenizers = (
  qwen2Bytes: Uint8Array,
  t5Bytes: Uint8Array,
): AnimaTokenizers =>
  new AnimaTokenizers(
    parseQwen2Asset(decodeJson(qwen2Bytes, "tokenizer")),
    parseT5Asset(decodeJson(t5Bytes, "tokenizer_2")),
  );

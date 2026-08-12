/**
 * Irodori-TTS v4 のトークナイザ（Unigram + byte_fallback）。text / caption の両方が同じ 1 本を通る。
 *
 * 正本の経路（`tokenizer/tokenizer.json`）は
 *   AddedVocabulary（18 特殊トークンを leftmost-longest で切り出す）
 *   → normalizer は **null**（正規化は上流のパイプライン段 `normalize.ts` が担う）
 *   → Metaspace{replacement=▁, prepend_scheme=never, split=false}
 *      = U+0020 を ▁ に置換するだけ（**先頭に ▁ を足さない / ▁ で断片に割らない**）
 *   → Unigram（Viterbi・unk_id=0・**byte_fallback あり**・fuse_unk）
 *   → post_processor は特殊トークンを足さない（BOS は下の {@link IrodoriTokenizer.encodePadded}
 *      = 上流 `PretrainedTextTokenizer.batch_encode` 相当が前置する）
 *
 * Unigram 本体（Viterbi・同点処理・`fuse_unk`・バイト展開）はファミリ非依存なので
 * `src/text/unigram.ts` が持つ。ここが持つのは Irodori 固有の前段（AddedVocabulary・Metaspace）と
 * 後段（BOS 前置・切り詰め・右 pad・マスク）だけ。
 */

import { splitAddedTokens } from "../../text/added-tokens.ts";
import { toCodePoints } from "../../text/code-points.ts";
import { type UnigramModel, unigramTokenize, type UnigramVocabEntry } from "../../text/unigram.ts";

/** Metaspace の置換文字（U+2581）。 */
const METASPACE = "▁";

export type IrodoriTokenizerAssets = {
  readonly vocab: ReadonlyMap<string, UnigramVocabEntry>;
  /** 語彙**全体**の最小スコア。部分集合を渡す場合も全体の値を渡す（未知ノードの重み）。 */
  readonly minScore: number;
  /** 語彙**全体**の最長トークンのコードポイント数（前方一致の探索幅）。 */
  readonly maxTokenLength: number;
  readonly unkId: number;
  /** `<0x00>` の id（未知の断片は UTF-8 バイト列 → `byteBaseId + バイト値` へ展開される）。 */
  readonly byteBaseId: number;
  readonly bosId: number;
  readonly padId: number;
  /** 追加語彙（特殊トークン）。正規化の**前**に切り出される。 */
  readonly addedTokens: ReadonlyMap<string, number>;
};

/** 上流 `batch_encode` の最終形（グラフ入力の dtype — i32 と bool = u32・ADR 0009）。 */
export type IrodoriPaddedIds = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

export class IrodoriTokenizer {
  readonly #assets: IrodoriTokenizerAssets;
  readonly #added: string[];

  constructor(assets: IrodoriTokenizerAssets) {
    this.#assets = assets;
    this.#added = [...assets.addedTokens.keys()];
  }

  /**
   * 正規化済みテキスト → 素の id 列（特殊トークンを足さない）。
   *
   * MUST: 入力は `normalize.ts` を通した後の文字列。正本の `tokenizer.json` は normalizer が
   * null なので、ここで正規化すると二重掛けになる（`…` の縮約などが 2 回走る）。
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const chunk of splitAddedTokens(text, this.#added)) {
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      // Metaspace は `split=false` なので断片に割らない — 全文を 1 つの格子で解く。
      const piece = chunk.text.replaceAll(" ", METASPACE);
      // IrodoriTokenizerAssets は UnigramModel の面をそのまま満たす。断片が全文なので id 列も
      // 長くなりうる — 引数展開（`push(...ids)`）にするとスタック上限に当たる。
      for (const id of unigramTokenize(this.#assets satisfies UnigramModel, toCodePoints(piece))) {
        ids.push(id);
      }
    }
    return ids;
  }

  /**
   * 上流 `PretrainedTextTokenizer.batch_encode([text], max_length)` と同じ最終形:
   * BOS 前置 + 右詰め pad + 有効長マスク。
   *
   * MUST: 本体の予算は `maxLength - 1`（BOS の 1 個ぶんを空ける）。素朴に `maxLength` で切ると
   * 全体が 1 個溢れる — 書き込みに {@link Int32Array.prototype.set} を使うのはそれを黙って
   * 捨てさせないため（添字代入は範囲外を無視するので、溢れが沈黙で吸われる）。
   */
  encodePadded(text: string, maxLength: number): IrodoriPaddedIds {
    if (!Number.isInteger(maxLength) || maxLength <= 0) {
      throw new Error(`maxLength は 1 以上の整数（受け取った値: ${maxLength}）`);
    }
    const body = this.encode(text).slice(0, maxLength - 1);
    const ids = new Int32Array(maxLength).fill(this.#assets.padId);
    const mask = new Uint32Array(maxLength);
    ids[0] = this.#assets.bosId;
    mask[0] = 1;
    ids.set(body, 1);
    mask.fill(1, 1, body.length + 1);
    return { ids, mask };
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

const parseAddedTokens = (raw: unknown, label: string): Map<string, number> => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 配列でない`);
  const out = new Map<string, number>();
  for (const entry of raw) {
    if (!Array.isArray(entry) || entry.length !== 2) {
      throw new Error(`${label}: [文字列, id] でない`);
    }
    out.set(asString(entry[0], label), asNumber(entry[1], label));
  }
  return out;
};

/**
 * 資産 JSON（exporter の `tokenizer.json`）を資産表へ。外部境界なので構造を検査してから使う。
 *
 * MUST: 語彙の行数と `scores` の本数を突き合わせる。ずれたまま通すと id が総ずれするが、
 * Viterbi は落ちない（語彙に無い断片はバイト展開へ逃げる）ので**沈黙誤値**になる。
 */
export const parseIrodoriTokenizerAsset = (
  raw: unknown,
  label: string = "tokenizer",
): IrodoriTokenizerAssets => {
  const obj = asRecord(raw, label);
  // 行番号 0-origin = id。語彙に改行・タブ・生空白を含むトークンが無いことは emit 時に検査済み。
  const tokens = asString(obj["vocabText"], `${label}.vocabText`).split("\n");
  const rawScores = obj["scores"];
  if (!Array.isArray(rawScores) || rawScores.length !== tokens.length) {
    throw new Error(`${label}: scores の長さが語彙の行数 ${tokens.length} と合わない`);
  }
  const vocab = new Map<string, UnigramVocabEntry>();
  let minScore = Number.POSITIVE_INFINITY;
  let maxTokenLength = 0;
  for (const [id, token] of tokens.entries()) {
    const score = asNumber(rawScores[id], `${label}.scores[${id}]`);
    vocab.set(token, { id, score });
    minScore = Math.min(minScore, score);
    maxTokenLength = Math.max(maxTokenLength, toCodePoints(token).length);
  }
  return {
    vocab,
    minScore,
    maxTokenLength,
    unkId: asNumber(obj["unkId"], `${label}.unkId`),
    byteBaseId: asNumber(obj["byteBaseId"], `${label}.byteBaseId`),
    bosId: asNumber(obj["bosId"], `${label}.bosId`),
    padId: asNumber(obj["padId"], `${label}.padId`),
    addedTokens: parseAddedTokens(obj["addedTokens"], `${label}.addedTokens`),
  };
};

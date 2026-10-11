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

import { type AddedTokenMatcher, createAddedTokenMatcher } from "../../text/added-tokens.ts";
import {
  asFiniteNumber,
  asId,
  asRecord,
  assertUniqueLines,
  asString,
  asVocabId,
  parseAddedTokens,
} from "../../text/asset-gates.ts";
import { assertEncodableText, toCodePoints } from "../../text/code-points.ts";
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
type IrodoriPaddedIds = {
  readonly ids: Int32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

export class IrodoriTokenizer {
  readonly #assets: IrodoriTokenizerAssets;
  readonly #added: AddedTokenMatcher;

  constructor(assets: IrodoriTokenizerAssets) {
    this.#assets = assets;
    this.#added = createAddedTokenMatcher([...assets.addedTokens.keys()]);
  }

  /**
   * 正規化済みテキスト → 素の id 列（特殊トークンを足さない）。
   *
   * MUST: 入力は `normalize.ts` を通した後の文字列。正本の `tokenizer.json` は normalizer が
   * null なので、ここで正規化すると二重掛けになる（`…` の縮約などが 2 回走る）。
   */
  encode(text: string): number[] {
    // 本文そのものの受理検査は格子を解く前に通す（ADR 0107 決定 2 の入力起因）。下の
    // 資産解析（`parseIrodoriTokenizerAssets`）も同じ条件を通るが、そちらの破れは資産の
    // 齟齬なので素の `Error` のまま残す。
    assertEncodableText(text, "irodori の本文");
    const ids: number[] = [];
    for (const chunk of this.#added.split(text)) {
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

/**
 * byte fallback の先頭 id（{@link asId} に加えて 256 本連番全体が語彙内であること）。
 *
 * MUST: 単独 id 用の {@link asVocabId}（`< vocabSize`）では足りない — `byteBaseId` は
 * 単独の id ではなく未知断片の展開先 256 本連番の先頭で、消費側（`unigram.ts` の
 * `base + byte`、byte = 0..255）は `byteBaseId + 255` まで発行する。ここを見逃すと
 * 未知断片が現れた瞬間に語彙外 id が出て、embedding gather の OOB（非有限 — ADR 0061）まで
 * 表面化しない沈黙誤値になる。
 *
 * MUST: 256 行の綴りも照合する（`<0x00>`..`<0xFF>` が順に並ぶこと = 配布契約）。消費側は
 * 綴りではなく連番の算術で id を求めるので、行が入れ替わった資産は HF（名前で id を引く）と
 * 別の id を黙って出す。exporter 側（`tools/export-recipes/irodori/tokenizer_ref.py`）の
 * 連番検査と対で、読み込み側でも同じ契約を落とす。
 */
const asByteBaseId = (value: unknown, label: string, tokens: readonly string[]): number => {
  const id = asId(value, label);
  if (id + 255 >= tokens.length) {
    throw new Error(
      `${label}: バイト 256 本（${id}..${id + 255}）が語彙の行数 ${tokens.length} に収まらない`,
    );
  }
  for (let byte = 0; byte < 256; byte += 1) {
    const want = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
    if (tokens[id + byte] !== want) {
      throw new Error(
        `${label}: 行 ${id + byte} が ${want} でない（${JSON.stringify(tokens[id + byte])}）`,
      );
    }
  }
  return id;
};

/**
 * 追加トークン（語彙の前段で丸ごと切り出す綴り → id）を読み、語彙表と突き合わせる。
 *
 * MUST: id は語彙の行数未満。共通の {@link parseAddedTokens} は i32 の値域しか見ない（anima の
 * Qwen2 が語彙表の外へ採番する形を正当に持つため — 共通側は狭めない）が、Irodori の追加
 * トークンは全て語彙表の行で、encode は写像の値をそのままグラフ入力へ積む。範囲外の id を
 * 通すと embedding gather の OOB（非有限 — ADR 0061）まで表面化しない。
 * MUST: 語彙に同じ綴りの行があれば id はその行番号と一致する。食い違うと、追加トークンとして
 * 切り出した綴りだけが語彙表と別の行を指す沈黙誤値になる（形も値域も合ったまま通る）。
 */
const parseIrodoriAddedTokens = (
  raw: unknown,
  label: string,
  vocab: ReadonlyMap<string, UnigramVocabEntry>,
): Map<string, number> => {
  const added = parseAddedTokens(raw, label);
  for (const [token, id] of added) {
    if (id >= vocab.size) {
      throw new Error(
        `${label}: ${JSON.stringify(token)} の id ${id} が語彙の行数 ${vocab.size} 以上`,
      );
    }
    const row = vocab.get(token)?.id;
    if (row !== undefined && row !== id) {
      throw new Error(
        `${label}: ${JSON.stringify(token)} の id ${id} が語彙の行 ${row} と食い違う`,
      );
    }
  }
  return added;
};

/**
 * 資産 JSON（exporter の `tokenizer.json`）を資産表へ。外部境界なので構造を検査してから使う。
 *
 * MUST: 語彙の行数と `scores` の本数を突き合わせる。ずれたまま通すと id が総ずれするが、
 * Viterbi は落ちない（語彙に無い断片はバイト展開へ逃げる）ので**沈黙誤値**になる。
 * MUST: 行の一意性も見る（{@link assertUniqueLines} — 行数の突合では重複を捕まえられない）。
 * MUST: 単独の特殊 id は整数かつ語彙の行数未満（{@link asVocabId} — 理由は同 doc）。
 * MUST: `byteBaseId` は単独 id ではないので {@link asVocabId} では足りない —
 * 256 本連番の全体が語彙内であること（{@link asByteBaseId} — 理由は同 doc）。
 * MUST: 追加トークンも語彙表と突き合わせる（{@link parseIrodoriAddedTokens} — 理由は同 doc）。
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
  assertUniqueLines(tokens, `${label}.vocabText`);
  const vocab = new Map<string, UnigramVocabEntry>();
  let minScore = Number.POSITIVE_INFINITY;
  let maxTokenLength = 0;
  for (const [id, token] of tokens.entries()) {
    const score = asFiniteNumber(rawScores[id], `${label}.scores[${id}]`);
    vocab.set(token, { id, score });
    minScore = Math.min(minScore, score);
    maxTokenLength = Math.max(maxTokenLength, toCodePoints(token).length);
  }
  return {
    vocab,
    minScore,
    maxTokenLength,
    unkId: asVocabId(obj["unkId"], `${label}.unkId`, tokens.length),
    byteBaseId: asByteBaseId(obj["byteBaseId"], `${label}.byteBaseId`, tokens),
    bosId: asVocabId(obj["bosId"], `${label}.bosId`, tokens.length),
    padId: asVocabId(obj["padId"], `${label}.padId`, tokens.length),
    addedTokens: parseIrodoriAddedTokens(obj["addedTokens"], `${label}.addedTokens`, vocab),
  };
};

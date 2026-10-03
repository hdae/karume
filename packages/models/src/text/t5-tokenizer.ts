/**
 * T5 系（SentencePiece の Unigram を `tokenizers` で組んだもの）のトークナイザ — 家族横断。
 *
 * 正本の経路（anima の T5 と Wan の umT5 の transformers 5.14.1 が同じ形）:
 *
 *   AddedVocabulary（特殊トークンを leftmost-longest で切り出す）
 *   → 正規化（家族ごと — anima は Precompiled・umT5 は無し）
 *   → WhitespaceSplit → Metaspace（replacement=▁ / prepend_scheme=always / split）
 *   → Unigram（Viterbi・byte_fallback なし・fuse_unk）
 *   → TemplateProcessing（末尾に `</s>`）
 *
 * 表（語彙・追加語彙・空白集合）は資産から受け、家族で割れる 3 点は {@link T5Policy} で受ける:
 * 正規化の有無、語彙外の扱い（unk へ融合 / 拒む）、上限超えの扱い（切り詰め / 拒む）。
 * anima は上流どおり「unk・切り詰め」、Wan の umT5 は版に依らない id 列のために「拒む・拒む」
 * （ADR 0119 決定 1・決定 4）。
 *
 * Unigram の本体（Viterbi・同点処理・融合）は `unigram.ts`、追加語彙の切り出しは
 * `added-tokens.ts` が持つ。
 */

import { ModelInputError } from "../errors.ts";
import { splitAddedTokens } from "./added-tokens.ts";
import {
  asFiniteNumber,
  asPositiveInteger,
  asRecord,
  assertUniqueLines,
  asString,
  asVocabId,
  parseAddedTokens,
} from "./asset-gates.ts";
import { toCodePoints } from "./code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "./code-ranges.ts";
import {
  type UnigramModel,
  unigramSegments,
  unigramTokenize,
  type UnigramVocabEntry,
} from "./unigram.ts";

/** Metaspace の置換文字（U+2581）。 */
export const METASPACE = "▁";

export type T5VocabEntry = UnigramVocabEntry;

/** T5 系トークナイザの表（資産由来）。 */
export type T5Tables = {
  readonly vocab: ReadonlyMap<string, T5VocabEntry>;
  /** 語彙**全体**の最小スコア。部分集合を渡す場合も全体の値を渡す（未知ノードの重み）。 */
  readonly minScore: number;
  /** 語彙**全体**の最長トークンのコードポイント数（前方一致の探索幅）。 */
  readonly maxTokenLength: number;
  readonly unkId: number;
  readonly eosId: number;
  readonly addedTokens: ReadonlyMap<string, number>;
  /** WhitespaceSplit の空白集合（エクスポータが正本に 1 文字ずつ聞いて畳んだ表）。 */
  readonly space: CodeRanges;
  /** `</s>` を含む id 列の最大長（上流の `max_length`）。 */
  readonly maxLength: number;
};

/** 家族ごとに上流の呼び方が割れる 3 点。 */
export type T5Policy = {
  /** 追加語彙を切り出した後の各断片に掛ける正規化（上流の normalizer）。無ければ掛けない。 */
  readonly normalize?: (text: string) => string;
  /** 語彙外（Unigram の未知ノード）: `"unk"` = unk_id 1 個へ融合 / `"reject"` = 拒む。 */
  readonly unknown: "unk" | "reject";
  /** `maxLength` を超える入力: `"truncate"` = `</s>` の分を空けて切る / `"reject"` = 拒む。 */
  readonly overflow: "truncate" | "reject";
};

/**
 * WhitespaceSplit → Metaspace。空白は捨て、各断片の先頭に ▁ を付ける。
 *
 * MUST: `split=true` は MergedWithNext — 区切りの ▁ は**次の**断片の先頭に付く。
 * 「前の断片の末尾」にすると分割が変わる。
 */
export const t5PreTokenize = (text: string, space: CodeRanges): string[] => {
  const out: string[] = [];
  for (const word of splitOnSpace(text, space)) {
    let piece = word.replaceAll(" ", METASPACE);
    if (!piece.startsWith(METASPACE)) piece = METASPACE + piece;
    let start = 0;
    for (let idx = 1; idx < piece.length; idx++) {
      if (piece[idx] === METASPACE) {
        out.push(piece.slice(start, idx));
        start = idx;
      }
    }
    out.push(piece.slice(start));
  }
  return out.filter((piece) => piece !== "");
};

const splitOnSpace = (text: string, space: CodeRanges): string[] => {
  const out: string[] = [];
  let buffer = "";
  for (const ch of text) {
    if (inCodeRanges(space, ch.codePointAt(0) as number)) {
      if (buffer !== "") {
        out.push(buffer);
        buffer = "";
      }
    } else {
      buffer += ch;
    }
  }
  if (buffer !== "") out.push(buffer);
  return out;
};

/** 未知の片の表示（語彙外の拒否のメッセージ用 — 人が直す箇所を探せる形）。 */
const describeText = (text: string): string =>
  `${JSON.stringify(text)}（${
    toCodePoints(text).map((cp) => `U+${cp.toString(16).toUpperCase().padStart(4, "0")}`)
      .join(" ")
  }）`;

export class T5UnigramTokenizer {
  readonly #tables: T5Tables;
  readonly #policy: T5Policy;
  readonly #added: string[];

  constructor(tables: T5Tables, policy: T5Policy) {
    this.#tables = tables;
    this.#policy = policy;
    this.#added = [...tables.addedTokens.keys()];
  }

  /** 1 断片を id 列へ（語彙外は方針どおり unk へ融合するか拒む）。 */
  #tokenize(piece: string, label: string): number[] {
    // T5Tables は UnigramModel の面をそのまま満たす（byteBaseId を持たない = byte_fallback なし）。
    const model = this.#tables satisfies UnigramModel;
    const cps = toCodePoints(piece);
    if (this.#policy.unknown === "unk") return unigramTokenize(model, cps);
    const ids: number[] = [];
    for (const segment of unigramSegments(model, cps)) {
      if (segment.entry === undefined) {
        throw new ModelInputError(
          `${label}: 語彙に無い文字列 ${describeText(segment.text)} を含む — 未知トークンへ` +
            "黙って潰さない。その文字を除くか言い換える",
        );
      }
      ids.push(segment.entry.id);
    }
    return ids;
  }

  /**
   * `tokenizer([text], max_length=maxLength, truncation=True)` と同じ id 列（上限超えは方針どおり）。
   *
   * MUST: 切り詰めは `</s>` の分を空けてから（正本の truncation は post_processor の**前**）。
   * `slice(0, maxLength)` の後に足すと `maxLength + 1` 個になる。結果として id 列は常に `</s>` で
   * 終わり、長さは必ず 1 以上。
   *
   * `label` は拒否のメッセージに出す（正 / ネガティブのどちらかが判る形にする）。
   */
  encode(text: string, label: string = "入力"): number[] {
    const budget = this.#tables.maxLength - 1;
    const reject = this.#policy.overflow === "reject";
    const ids: number[] = [];
    const overflow = (): ModelInputError =>
      new ModelInputError(
        `${label}: トークン数が上限 ${this.#tables.maxLength}（末尾の </s> を含む）を超える — ` +
          "黙って切り詰めない。プロンプトを短くする",
      );
    /**
     * 次の id を積む前に呼ぶ（追加語彙も断片も必ず 1 個以上の id を出す）。予算に達していれば、
     * 切り詰めなら false（以降を捨てる）・拒むなら投げる。
     */
    const hasRoom = (): boolean => {
      if (ids.length < budget) return true;
      if (reject) throw overflow();
      return false;
    };
    // 先頭から順に積むだけなので、予算に達した後の chunk / 断片を捨てても id 列は変わらない
    // （切り詰めの後ろを符号化する費用を入力長に比例させない — 拒む方針でも同じ所で止まる）。
    for (const chunk of splitAddedTokens(text, this.#added)) {
      if (chunk.added) {
        if (!hasRoom()) break;
        ids.push(this.#tables.addedTokens.get(chunk.text) as number);
        continue;
      }
      // NOTE: 追加語彙でない chunk は空白だけなら id を出さないので、ここでは拒まない
      // （拒む方針の判定は id を出す断片の手前と末尾で行う）。
      if (!reject && ids.length >= budget) break;
      const normalized = this.#policy.normalize?.(chunk.text) ?? chunk.text;
      for (const piece of t5PreTokenize(normalized, this.#tables.space)) {
        if (!hasRoom()) break;
        // 長い断片でも引数展開しない（`push(...)` は V8 の引数上限で RangeError になる）。
        for (const id of this.#tokenize(piece, label)) ids.push(id);
      }
    }
    // 最後の断片が予算をまたいで積んだ分（拒む方針ではここで落とす）。
    if (reject && ids.length > budget) throw overflow();
    return [...ids.slice(0, budget), this.#tables.eosId];
  }
}

/**
 * T5 系の資産 JSON（`vocabText` / `scores` / `unkId` / `eosId` / `addedTokens` / `space` /
 * `maxLength`）を表に変換する（外部境界なので構造を検査してから使う）。
 *
 * MUST: `minScore` / `maxTokenLength` は**語彙全体**から導く（未知ノードの重みと探索幅）。
 */
export const parseT5Tables = (raw: unknown, label: string): T5Tables => {
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
    maxLength: asPositiveInteger(obj["maxLength"], `${label}.maxLength`),
  };
};

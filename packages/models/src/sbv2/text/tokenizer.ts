/**
 * DeBERTa 文字トークナイザ（transformers `BertJapaneseTokenizer`、word=basic / subword=character）
 * の移植。対象は `outputs/series/deberta/` と同一の
 * `ku-nlp/deberta-v2-large-japanese-char-wwm`。
 *
 * 参照実装の経路は `tokenization_bert_japanese.py` の BasicTokenizer → CharacterTokenizer で、
 * 実効手順は:
 *
 * 1. `BasicTokenizer._clean_text`: cp==0 / cp==0xFFFD / Unicode category C* を除去、
 *    空白（" " \t \n \r と Zs）を半角スペースへ。**この判定表は資産に焼く** — TS 側で
 *    Unicode 分類を再実装すると ICU 版差が静かな不一致になる（判定の正本は Python）。
 * 2. NFC 正規化。
 * 3. 空白分割（空白は捨てる）。
 * 4. 各トークンを NFKC 正規化 → 1 コードポイントずつに分割。
 * 5. vocab 参照（無ければ `[UNK]`）。`encode` は `[CLS] + ids + [SEP]`。
 *
 * NOTE: 参照実装は 3 と 4 の間に `_run_split_on_punc`（句読点を独立トークンへ切り出す）を
 * 挟むが、直後の CharacterTokenizer が全トークンを 1 文字ずつに割るため出力文字列は
 * 変わらない（NFKC の合成は結合文字と基底文字の間でしか起きず、句読点境界を跨がない）。
 * よって移植しない。
 *
 * 「1 文字」は Unicode コードポイント単位。トークン数は word2ph の長さと直結するので、
 * ここの分割規則が参照実装とずれると BERT 特徴が音素へ誤配置される（音は出るが崩れる）。
 * その齟齬は `sbv2_demo.py reference` が「同じ bert_text から同じ input_ids が出るか」で
 * 実データ突合する。
 */

/** `clean_text` の除去・スペース化コードポイント範囲（両端含む閉区間の昇順リスト）。 */
export type CleanRanges = {
  /** 除去する（出力しない）コードポイント範囲。 */
  readonly removed: readonly (readonly [number, number])[];
  /** 半角スペースへ置換するコードポイント範囲。 */
  readonly spaced: readonly (readonly [number, number])[];
};

/** `[CLS]` / `[SEP]` / `[UNK]` の語彙 ID。 */
type SpecialTokens = {
  readonly clsId: number;
  readonly sepId: number;
  readonly unkId: number;
};

/** 昇順の閉区間リストに対する二分探索で cp が含まれるか判定する。 */
const inRanges = (ranges: readonly (readonly [number, number])[], cp: number): boolean => {
  let low = 0;
  let high = ranges.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const [start, end] = ranges[mid];
    if (cp < start) high = mid - 1;
    else if (cp > end) low = mid + 1;
    else return true;
  }
  return false;
};

export class DebertaTokenizer {
  readonly #vocab: ReadonlyMap<string, number>;
  readonly #clean: CleanRanges;
  readonly #special: SpecialTokens;

  constructor(vocab: ReadonlyMap<string, number>, clean: CleanRanges, special: SpecialTokens) {
    this.#vocab = vocab;
    this.#clean = clean;
    this.#special = special;
  }

  /**
   * `vocab.txt` 相当（1 行 = 1 トークン、行番号 0-origin = ID）から構築する。
   *
   * MUST: 特殊 ID が語彙表の行を指すことを検査する。範囲外の ID は id 列へそのまま乗り、
   * グラフの embedding gather が範囲外添字を引く（GPU では NaN 汚染）まで表面化しない。
   */
  static fromVocabText(
    vocabText: string,
    clean: CleanRanges,
    special: SpecialTokens,
  ): DebertaTokenizer {
    const vocab = new Map<string, number>();
    // CRLF 混在で split("\n") だけにすると全トークン末尾に "\r" が残り、全 lookup が
    // [UNK] に落ちる（エラーは出ず BERT 特徴だけ静かに壊れる）。
    const lines = vocabText.split(/\r?\n/);
    let size = 0;
    for (const [id, line] of lines.entries()) {
      // 末尾改行由来の空行はトークンではない。行内に空トークンは存在しない。
      if (line === "" && id === lines.length - 1) break;
      vocab.set(line, id);
      size = id + 1;
    }
    for (const [name, id] of Object.entries(special)) {
      if (id >= size) throw new Error(`${name} ${id} が語彙の行数 ${size} 以上`);
    }
    return new DebertaTokenizer(vocab, clean, special);
  }

  /**
   * clean_text + NFC + 空白分割 + NFKC を適用し、トークン（1 コードポイント）配列にする。
   * CLS / SEP は含まない — word2ph 用の語トークン数はこれで数える。
   */
  tokenize(text: string): string[] {
    let cleaned = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (cp === undefined) continue;
      if (inRanges(this.#clean.removed, cp)) continue;
      cleaned += inRanges(this.#clean.spaced, cp) ? " " : ch;
    }
    // NFC は参照実装の BasicTokenizer 側、NFKC は CharacterTokenizer 側。間に空白分割が
    // 入るため 1 回にまとめられない（NFKC がスペースを生む文字で分割結果が変わる）。
    const tokens: string[] = [];
    for (const word of cleaned.normalize("NFC").split(/\s+/)) {
      if (word === "") continue;
      for (const ch of word.normalize("NFKC")) tokens.push(ch);
    }
    return tokens;
  }

  /** text を input_ids へ（`[CLS]` + 各文字 ID + `[SEP]`）。参照 `tokenizer(text)` と一致する。 */
  encode(text: string): number[] {
    const ids = [this.#special.clsId];
    for (const token of this.tokenize(text)) {
      ids.push(this.#vocab.get(token) ?? this.#special.unkId);
    }
    ids.push(this.#special.sepId);
    return ids;
  }
}

/**
 * Qwen2 トークナイザ（byte-level BPE）。Anima の text_encoder（Qwen3）の入力を作る。
 *
 * 正本の経路（`tokenizer/tokenizer.json`）は
 *   AddedVocabulary（26 特殊トークンを leftmost-longest で切り出す）
 *   → NFC → Split（正規表現・Isolated）→ ByteLevel（add_prefix_space=false）
 *   → BPE（ignore_merges=false / unk なし）→ TemplateProcessing（**特殊トークン追加なし**）
 *
 * MUST: chat template を通さない。正本は素の `tokenizer([text], …)` を呼ぶ。
 * `apply_chat_template` を通すと `<|im_start|>` 系が前置され id 列が丸ごと別物になる。
 *
 * **正規表現の `\p{L}` / `\p{N}` / `\s` は使わない** — 判定の正本は Rust 側の Unicode 表で、
 * JS エンジンの ICU とずれた瞬間に pre-token の切れ目が変わり、id 列が静かに別物になる。
 * 分類はエクスポータが正本に 1 文字ずつ聞いて畳んだ閉区間表を引く（`code-ranges.ts`）。
 *
 * MUST: **NFC も同じ規律**。`String.prototype.normalize("NFC")` は正本ではない — 正本
 * （Rust の `unicode-normalization`）は Unicode 表が古く、実測で 123 コードポイントについて
 * ICU / Python と出力が割れる（120 cp は結合クラスを 0 と見なすので並べ替えない・3 cp は
 * 新しい合成をしない）。素の `normalize("NFC")` に委ねると、その文字を含むプロンプトだけが
 * 例外も警告も出さずに別の id 列になる。焼いた分節表で切ってから素の NFC を掛ける
 * （{@link normalizeNfc}）。
 */

import { type CodeRanges, inCodeRanges, toCodePoints } from "./code-ranges.ts";

/** pre_tokenizer 正規表現が使う文字クラス（閉区間表）。 */
export type Qwen2CharClasses = {
  /** `\p{L}` */
  readonly letter: CodeRanges;
  /** `\p{N}` */
  readonly number: CodeRanges;
  /** `\s` */
  readonly space: CodeRanges;
};

/**
 * `(?i:'s|'t|…)` が同一視するコードポイント → 接尾辞側の文字（どちらも ASCII 小文字）。
 *
 * MUST: `toLowerCase()` で代用しない。**正本の `(?i:)` は Unicode の simple case folding**
 * で、実測では U+017F（ſ）が `s` と同一視される（`it'ſs` は `'ſ` + `s` に切れる）。ASCII
 * の大小反転でも `toLowerCase()`（ſ → ſ）でも取りこぼし、pre-token の切れ目が静かに変わる。
 * 自明な組（cp 自身）は入っていない。
 */
export type Qwen2CaseFold = ReadonlyMap<number, number>;

export type Qwen2Assets = {
  /** トークン文字列 → id。 */
  readonly vocab: ReadonlyMap<string, number>;
  /** `"左 右"` → rank（byte-level 語彙は空白を含まないので区切りに使える）。 */
  readonly merges: ReadonlyMap<string, number>;
  /** 追加語彙（特殊トークン）。正規化の**前**に切り出される。 */
  readonly addedTokens: ReadonlyMap<string, number>;
  readonly classes: Qwen2CharClasses;
  readonly caseFold: Qwen2CaseFold;
  /** 正本の NFC と `String.prototype.normalize` が割れるコードポイント（{@link normalizeNfc}）。 */
  readonly nfcSegments: CodeRanges;
  /** 正本の呼び出しに合わせた切り詰め長（512）。 */
  readonly maxLength: number;
};

/** `(?i:'s|'t|'re|'ve|'m|'ll|'d)` の接尾辞（正規表現と同じ順）。 */
const APOSTROPHE_SUFFIXES = ["s", "t", "re", "ve", "m", "ll", "d"] as const;

/**
 * GPT-2 由来の byte → 可視文字表。Unicode 版に依存しない純アルゴリズムなので組み立ててよい
 * （表を焼く必要が無い唯一の箇所）。
 *
 * MUST: モジュールスコープの `const` に持たない（横断不変条件「全モジュール副作用ゼロ」）。
 * 組み立ては {@link Qwen2Tokenizer} のコンストラクタで 1 インスタンス 1 回だけ走る。
 */
const buildByteEncoder = (): readonly string[] => {
  // 可視のまま置ける範囲。それ以外は U+0100 から順に割り当てる。
  const isDirect = (byte: number): boolean =>
    (byte >= 0x21 && byte <= 0x7e) ||
    (byte >= 0xa1 && byte <= 0xac) ||
    (byte >= 0xae && byte <= 0xff);
  let extra = 0;
  return Array.from({ length: 256 }, (_, byte) => {
    if (isDirect(byte)) return String.fromCodePoint(byte);
    const mapped = String.fromCodePoint(0x100 + extra);
    extra += 1;
    return mapped;
  });
};

/**
 * 正本（Rust の `unicode-normalization`）と同じ NFC。
 *
 * 分節 cp で文字列を切り、各節だけを `String.prototype.normalize("NFC")` に掛けて連結する。
 * MUST: 分節 cp 自身は**どちらの節にも入れない** — 正本はその cp を starter として扱い、
 * 前後の並べ替えにも合成にも関わらせないので、素通しがその挙動そのものになる。
 * 分節表は正本への探り針で焼いたもので、同値は emit のたびに全コードポイント × 11 文脈 +
 * 乱択で検査される。
 */
export const normalizeNfc = (text: string, segments: CodeRanges): string => {
  const out: string[] = [];
  let buffer = "";
  for (const ch of text) {
    if (inCodeRanges(segments, ch.codePointAt(0) as number)) {
      out.push(buffer.normalize("NFC"), ch);
      buffer = "";
    } else {
      buffer += ch;
    }
  }
  out.push(buffer.normalize("NFC"));
  return out.join("");
};

/**
 * 追加語彙を leftmost-longest で切り出す（正本の `AddedVocabulary` = AhoCorasick の
 * LeftmostLongest と同じ規則）。位置が同じなら長い方、位置が違えば左が勝つ。
 *
 * MUST: 長い順に並べて先頭一致で採るだけでは足りない — **各位置で全候補中の最長**を採る。
 * 切り出された断片は正規化も pre-token 化も通さず id を直接出す。
 */
export const splitAddedTokens = (
  text: string,
  added: readonly string[],
): { text: string; added: boolean }[] => {
  if (added.some((token) => token === "")) {
    // 空文字は全位置で一致して走査が進まない（無限ループ）。`qwen2PreTokenize` が
    // 「走査が進まない」を落としているのと同じ理由でここも落とす。
    throw new Error("追加語彙に空文字がある — 切り出しが進まない");
  }
  const out: { text: string; added: boolean }[] = [];
  let buffer = "";
  let i = 0;
  while (i < text.length) {
    let hit: string | undefined;
    for (const token of added) {
      if (text.startsWith(token, i) && (hit === undefined || token.length > hit.length)) {
        hit = token;
      }
    }
    if (hit === undefined) {
      buffer += text[i];
      i += 1;
      continue;
    }
    if (buffer !== "") {
      out.push({ text: buffer, added: false });
      buffer = "";
    }
    out.push({ text: hit, added: true });
    i += hit.length;
  }
  if (buffer !== "") out.push({ text: buffer, added: false });
  return out;
};

/**
 * Split 正規表現の手書き走査。選択肢は上から順に、最初に一致したものが勝つ:
 *   ① `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
 *   ② `[^\r\n\p{L}\p{N}]?\p{L}+`
 *   ③ `\p{N}`
 *   ④ ` ?[^\s\p{L}\p{N}]+[\r\n]*`
 *   ⑤ `\s*[\r\n]+`
 *   ⑥ `\s+(?!\S)`
 *   ⑦ `\s+`
 * どの選択肢も空一致しないので、位置ごとに 1 つ選んで進めば正本の find_iter と同値になる。
 * 同値そのものは `verify_qwen_pre_tokenize`（全コードポイント × 11 文脈）が実測で固定する。
 */
export const qwen2PreTokenize = (
  cps: readonly number[],
  classes: Qwen2CharClasses,
  caseFold: Qwen2CaseFold,
): string[] => {
  const n = cps.length;
  const isLetter = (k: number): boolean => inCodeRanges(classes.letter, cps[k]);
  const isNumber = (k: number): boolean => inCodeRanges(classes.number, cps[k]);
  const isSpace = (k: number): boolean => inCodeRanges(classes.space, cps[k]);
  const isOther = (k: number): boolean => !isSpace(k) && !isLetter(k) && !isNumber(k);
  const isNewline = (k: number): boolean => cps[k] === 0x0d || cps[k] === 0x0a;

  const out: string[] = [];
  let i = 0;
  while (i < n) {
    let end = -1;
    // ① 短縮形。大小無視は焼いた表で引く（`toLowerCase()` は正本ではない）。
    if (cps[i] === 0x27) {
      for (const suffix of APOSTROPHE_SUFFIXES) {
        if (matchesFolded(cps, i + 1, suffix, caseFold)) {
          end = i + 1 + suffix.length;
          break;
        }
      }
    }
    // ② 任意の 1 文字（CR/LF・字・数以外）+ 字の連続。任意部を先に食い、駄目なら戻る。
    if (end < 0) {
      for (const skip of [1, 0]) {
        if (skip === 1 && (isNewline(i) || isLetter(i) || isNumber(i))) continue;
        let k = i + skip;
        const start = k;
        while (k < n && isLetter(k)) k++;
        if (k > start) {
          end = k;
          break;
        }
      }
    }
    // ③ 数字 1 文字（`+` が無いので必ず 1 文字ずつ切れる）
    if (end < 0 && isNumber(i)) end = i + 1;
    // ④ 任意の半角空白 1 文字 + その他の連続 + CR/LF の連続
    if (end < 0) {
      for (const skip of [1, 0]) {
        if (skip === 1 && cps[i] !== 0x20) continue;
        let k = i + skip;
        const start = k;
        while (k < n && isOther(k)) k++;
        if (k > start) {
          while (k < n && isNewline(k)) k++;
          end = k;
          break;
        }
      }
    }
    // ⑤ 空白の連続のうち、末尾が CR/LF になる最長のもの
    if (end < 0 && isSpace(i)) {
      let k = i;
      while (k < n && isSpace(k)) k++;
      let last = -1;
      for (let m = i; m < k; m++) if (isNewline(m)) last = m;
      if (last >= 0) end = last + 1;
    }
    // ⑥ 空白の連続（非空白が続くなら最後の 1 文字を残す）/ ⑦ 空白の連続
    if (end < 0 && isSpace(i)) {
      let k = i;
      while (k < n && isSpace(k)) k++;
      end = k === n ? k : Math.max(i + 1, k - 1);
    }
    if (end <= i) {
      throw new Error(`pre-token 走査が進まない（位置 ${i}, U+${cps[i].toString(16)}）`);
    }
    // 長い pre-token でも引数展開しない（`fromCodePoint(...)` はスタック上限に当たる）。
    let piece = "";
    for (let k = i; k < end; k++) piece += String.fromCodePoint(cps[k]);
    out.push(piece);
    i = end;
  }
  return out;
};

/** `cps[at:]` が接尾辞に大小無視で一致するか（同一視は焼いた表だけで判定する）。 */
const matchesFolded = (
  cps: readonly number[],
  at: number,
  suffix: string,
  caseFold: Qwen2CaseFold,
): boolean => {
  if (at + suffix.length > cps.length) return false;
  for (let k = 0; k < suffix.length; k++) {
    const cp = cps[at + k];
    const want = suffix.codePointAt(k);
    if (cp !== want && caseFold.get(cp) !== want) return false;
  }
  return true;
};

export class Qwen2Tokenizer {
  readonly #assets: Qwen2Assets;
  readonly #added: string[];
  readonly #byteEncoder: readonly string[];
  readonly #encoder = new TextEncoder();

  constructor(assets: Qwen2Assets) {
    this.#assets = assets;
    this.#added = [...assets.addedTokens.keys()];
    this.#byteEncoder = buildByteEncoder();
  }

  /** 1 つの pre-token（byte-level 符号化済み）を BPE で分割する。 */
  #bpe(piece: string): string[] {
    const symbols = Array.from(piece);
    for (;;) {
      let bestRank = Number.POSITIVE_INFINITY;
      let bestAt = -1;
      for (let idx = 0; idx + 1 < symbols.length; idx++) {
        const rank = this.#assets.merges.get(`${symbols[idx]} ${symbols[idx + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = idx;
        }
      }
      if (bestAt < 0) return symbols;
      symbols.splice(bestAt, 2, symbols[bestAt] + symbols[bestAt + 1]);
    }
  }

  /**
   * `tokenizer([text], padding="longest", max_length=512, truncation=True)` と同じ id 列。
   * post_processor が特殊トークンを足さないので、切り詰めは単純な先頭切り。
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const chunk of splitAddedTokens(text, this.#added)) {
      if (chunk.added) {
        ids.push(this.#assets.addedTokens.get(chunk.text) as number);
        continue;
      }
      const cps = toCodePoints(normalizeNfc(chunk.text, this.#assets.nfcSegments));
      for (const piece of qwen2PreTokenize(cps, this.#assets.classes, this.#assets.caseFold)) {
        // ByteLevel（add_prefix_space=false）— **先頭スペースを足さない**。
        let encoded = "";
        for (const byte of this.#encoder.encode(piece)) encoded += this.#byteEncoder[byte];
        for (const token of this.#bpe(encoded)) {
          const id = this.#assets.vocab.get(token);
          // byte-level なので語彙に無いトークンは原理的に出ない。出たら表が壊れている。
          if (id === undefined) {
            throw new Error(`Qwen2 語彙に無いトークン ${JSON.stringify(token)}`);
          }
          ids.push(id);
        }
      }
    }
    return ids.slice(0, this.#assets.maxLength);
  }
}

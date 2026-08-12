/**
 * Irodori-TTS v4 の `normalize_text` 移植（生テキスト → トークナイザに渡す文字列）。
 *
 * 正本は上流 `irodori_tts/text_normalization.py` で、5 段を**この順に**通す:
 *   ① SIMPLE_REPLACE_MAP の逐次 `replace`
 *   ② REGEX_REPLACE_MAP の逐次 `sub`
 *   ③ `strip_outer_brackets`（対応する外側括弧を、全体を囲んでいる間だけ剥がし続ける）
 *   ④ NFKC
 *   ⑤ `...` → `…` / `..` → `…`
 *
 * MUST: 順序そのものが観測できる（例: `①` は ④ の前に消えるので `1` にならない / 全角 `；`
 * は ② の削除規則をすり抜けて ④ で `;` になる / `....` は ⑤ の前段が先に食うので `….`）。
 * 段を入れ替えると例外も警告も出さずに別の文字列になるので、golden 33 ケースが順序を固定する。
 *
 * NFKC は `String.prototype.normalize("NFKC")` に委ねる（正本も Python の `unicodedata` で、
 * 互換分解は Unicode 標準そのもの）。両者の Unicode 版がずれると黙って結果が変わるため、
 * パリティ門が**全コードポイント**を掃引して差分表と突き合わせる。
 */

/**
 * ① 逐次 `replace`（正本の dict の順）。目に見えない / 紛らわしい文字は `\uXXXX` で書く —
 * 編集の途中で消えたり、○ ◯ 〇 のように見分けの付かない別字が混ざる。
 */
const SIMPLE_REPLACEMENTS: readonly (readonly [string, string])[] = [
  ["\t", ""],
  ["[n]", ""],
  // 正本の `r"\[n\]"` は正規表現ではなく**リテラル**（`str.replace` に渡している）。
  ["\\[n\\]", ""],
  ["\u3000", ""], // 全角空白。④ の前なので半角化ではなく削除される
  ["\uFF1F", "?"], // ？
  ["\uFF01", "!"], // ！
  ["\u2665", "\u2661"], // ♥ → ♡
  ["\u25CF", "\u25CB"], // ● → ○
  ["\u25EF", "\u25CB"], // ◯ → ○
  ["\u3007", "\u25CB"], // 〇（漢数字ゼロ）→ ○
];

/**
 * ② 逐次 `sub`（正本の dict の順）。
 *
 * NOTE: `g` 付き正規表現をモジュールスコープに置いているが、`String.prototype.replace` は
 * 開始時に `lastIndex` を 0 に戻すので呼び出し間で状態を持たない（横断不変条件「全モジュール
 * 副作用ゼロ」の意味での可変状態にはならない）。
 */
const REGEX_REPLACEMENTS: readonly (readonly [RegExp, string])[] = [
  // ; ▼ ♀ ♂ 《 》 ≪ ≫ ①〜⑥
  [/[;\u25BC\u2640\u2642\u300A\u300B\u226A\u226B\u2460\u2461\u2462\u2463\u2464\u2465]/gu, ""],
  // ダッシュ・罫線類（U+2010〜U+2015 は範囲）
  [/[\u02D7\u2010-\u2015\u2043\u2212\u23AF\u23E4\u2500\u2501\u2E3A\u2E3B]/gu, ""],
  [/[\uFF5E\u301C]/gu, "\u30FC"], // ～ 〜 → ー（長音）
  [/\u2026{3,}/gu, "\u2026\u2026"], // … 3 つ以上のときだけ 2 つへ縮約
];

/** ③ が剥がす括弧の対（開き → 閉じ）。 */
const BRACKET_PAIRS: readonly (readonly [string, string])[] = [
  ["\u300C", "\u300D"], // 「」
  ["\u300E", "\u300F"], // 『』
  ["\uFF08", "\uFF09"], // （）
  ["\u3010", "\u3011"], // 【】
  ["(", ")"],
];

/**
 * ③ 対応する外側括弧を剥がす。剥がすのは**全体を囲んでいる**ときだけで、剥がせた場合は
 * もう一度先頭から試す（`「（あ）」` は 2 回剥がれる）。
 *
 * MUST: 走査はコードポイント単位（正本は Python の str = コードポイント列）。UTF-16 単位で
 * 数えると、BMP 外の文字を含む入力で長さと添字がずれる。
 */
const stripOuterBrackets = (text: string): string => {
  let cps = [...text];
  for (;;) {
    if (cps.length < 2) break;
    const pair = BRACKET_PAIRS.find(([open]) => open === cps[0]);
    if (pair === undefined || cps[cps.length - 1] !== pair[1]) break;
    const [open, close] = pair;
    let depth = 0;
    let enclosingAll = true;
    for (const [index, ch] of cps.entries()) {
      if (ch === open) depth += 1;
      else if (ch === close) depth -= 1;
      // 途中で対応が閉じたら、外側の 1 対は全体を囲んでいない（`「あ」と「い」`）。
      if (depth === 0 && index < cps.length - 1) {
        enclosingAll = false;
        break;
      }
    }
    if (!enclosingAll || depth !== 0) break;
    cps = cps.slice(1, -1);
  }
  return cps.join("");
};

/** 生テキスト → 正規化済みテキスト（正本 `normalize_text` と同値）。 */
export const normalizeText = (text: string): string => {
  let out = text;
  for (const [from, to] of SIMPLE_REPLACEMENTS) out = out.replaceAll(from, to);
  for (const [pattern, to] of REGEX_REPLACEMENTS) out = out.replace(pattern, to);
  out = stripOuterBrackets(out);
  out = out.normalize("NFKC");
  // MUST: `...` を先に食う（`....` は `….` になる — 逆順だと `……` になって別物）。
  return out.replaceAll("...", "…").replaceAll("..", "…");
};

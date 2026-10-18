/**
 * Port of Irodori-TTS v4's `normalize_text` (raw text → the string passed to the tokenizer).
 *
 * The source of truth is upstream's `irodori_tts/text_normalization.py`, which applies 5 steps
 * **in this order**:
 *   (1) sequential `replace` of SIMPLE_REPLACE_MAP
 *   (2) sequential `sub` of REGEX_REPLACE_MAP
 *   (3) `strip_outer_brackets` (keeps stripping a matching outer bracket pair while it encloses the
 *       whole text)
 *   (4) NFKC
 *   (5) `...` → `…` / `..` → `…`
 *
 * MUST: the order itself is observable (e.g. `①` disappears before (4), so it never becomes `1` /
 * a fullwidth `；` slips past the deletion rule of (2) and becomes `;` in (4) / `....` becomes `….`
 * because the first replacement of (5) consumes it first). Swapping steps yields a different string
 * without any exception or warning, so the 33 golden cases pin the order.
 *
 * NFKC is delegated to `String.prototype.normalize("NFKC")` (the source of truth is Python's
 * `unicodedata`, and compatibility decomposition is the Unicode standard itself). If the Unicode
 * versions of the two drift, the result changes silently, so the parity check sweeps **every code
 * point** and compares it against the difference table.
 */

/**
 * (1) Sequential `replace` (in the order of the canonical dict). Invisible / confusable characters
 * are written as `\uXXXX` — otherwise they vanish during editing, or indistinguishable different
 * characters get mixed in (such as U+25CB / U+25EF / U+3007, three different circles).
 */
const SIMPLE_REPLACEMENTS: readonly (readonly [string, string])[] = [
  ["\t", ""],
  ["[n]", ""],
  // The canonical `r"\[n\]"` is not a regular expression but a **literal** (it is passed to
  // `str.replace`).
  ["\\[n\\]", ""],
  ["\u3000", ""], // ideographic space. Before (4), so it is deleted rather than turned halfwidth
  ["\uFF1F", "?"], // fullwidth ？
  ["\uFF01", "!"], // fullwidth ！
  ["\u2665", "\u2661"], // ♥ → ♡
  ["\u25CF", "\u25CB"], // ● → ○
  ["\u25EF", "\u25CB"], // ◯ → ○
  ["\u3007", "\u25CB"], // U+3007 ideographic number zero → ○
];

/**
 * (2) Sequential `sub` (in the order of the canonical dict).
 *
 * NOTE: regular expressions with `g` sit at module scope, but `String.prototype.replace` resets
 * `lastIndex` to 0 when it starts, so no state carries between calls (it is not mutable state in
 * the sense of the cross-cutting invariant "every module is free of side effects").
 */
const REGEX_REPLACEMENTS: readonly (readonly [RegExp, string])[] = [
  // ; ▼ ♀ ♂ 《 》 ≪ ≫ ①〜⑥
  [/[;\u25BC\u2640\u2642\u300A\u300B\u226A\u226B\u2460\u2461\u2462\u2463\u2464\u2465]/gu, ""],
  // dashes and box-drawing lines (U+2010 to U+2015 as a range)
  [/[\u02D7\u2010-\u2015\u2043\u2212\u23AF\u23E4\u2500\u2501\u2E3A\u2E3B]/gu, ""],
  [/[\uFF5E\u301C]/gu, "\u30FC"], // ～ 〜 → U+30FC (prolonged sound mark)
  [/\u2026{3,}/gu, "\u2026\u2026"], // 3 or more … collapse to 2, only then
];

/** The bracket pairs that (3) strips (open → close). */
const BRACKET_PAIRS: readonly (readonly [string, string])[] = [
  ["\u300C", "\u300D"], // 「」
  ["\u300E", "\u300F"], // 『』
  ["\uFF08", "\uFF09"], // （）
  ["\u3010", "\u3011"], // 【】
  ["(", ")"],
];

/**
 * (3) Strips matching outer brackets. It strips only while the pair **encloses the whole text**, and
 * after a successful strip it tries again from the start (`「（a）」` is stripped twice).
 *
 * MUST: the scan goes by code points (the source of truth is a Python str = a code-point
 * sequence). Counting UTF-16 units shifts the length and the indices for input with characters
 * outside the BMP.
 *
 * MUST: it runs in time linear in the input length. Copying the canonical form as is (scan and
 * slice the whole remainder for every layer stripped) is quadratic for n nested layers, and since
 * the token-count truncation comes after normalization, input of pathological length arrives
 * untouched. So "the matching that counts only brackets of the same kind" is computed once, with
 * one stack per kind: the canonical "the depth counted from the outer opening bracket does not
 * return to 0 before the end, and is exactly 0 at the end" is equivalent to the partner of opening
 * bracket `k` being exactly the end `n−1−k` (the opening brackets before `k` are lower in the
 * stack, so the characters outside the window do not change the match of `k`). Each layer's test
 * depends only on its own window, so the layers are tested from the outside in, stopping at the
 * first one that fails, and the text is sliced once at the end.
 */
const stripOuterBrackets = (text: string): string => {
  const cps = Array.from(text);
  const n = cps.length;
  // Position of an opening bracket → position of the matching closing bracket, counting only
  // brackets of the same kind.
  const matchOf = new Map<number, number>();
  const stacks = BRACKET_PAIRS.map(() => [] as number[]);
  for (const [index, ch] of cps.entries()) {
    for (const [kind, [open, close]] of BRACKET_PAIRS.entries()) {
      if (ch === open) {
        stacks[kind].push(index);
      } else if (ch === close) {
        const opened = stacks[kind].pop();
        if (opened !== undefined) matchOf.set(opened, index);
      }
    }
  }
  let layer = 0;
  // Try only while at least 2 characters remain (the canonical `len(text) >= 2`).
  while (n - 2 * layer >= 2) {
    const last = n - 1 - layer;
    const pair = BRACKET_PAIRS.find(([open]) => open === cps[layer]);
    // If the match closes midway (`「a」と「b」`) or never closes (`((a)`), the outer pair does not
    // enclose the whole text.
    if (pair === undefined || cps[last] !== pair[1] || matchOf.get(layer) !== last) break;
    layer += 1;
  }
  return cps.slice(layer, n - layer).join("");
};

/** Raw text → normalized text (equivalent to the canonical `normalize_text`). */
export const normalizeText = (text: string): string => {
  let out = text;
  for (const [from, to] of SIMPLE_REPLACEMENTS) out = out.replaceAll(from, to);
  for (const [pattern, to] of REGEX_REPLACEMENTS) out = out.replace(pattern, to);
  out = stripOuterBrackets(out);
  out = out.normalize("NFKC");
  // MUST: consume `...` first (`....` becomes `….` — in the reverse order it would become `……`, a
  // different string).
  return out.replaceAll("...", "…").replaceAll("..", "…");
};

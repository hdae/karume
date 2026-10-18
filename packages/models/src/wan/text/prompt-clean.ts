/**
 * A mirror of Wan2.1's preprocessing `prompt_clean` (diffusers 0.39.0) — ADR 0119 decision 2
 * (option b').
 *
 * Upstream is `ftfy.fix_text → html.unescape × 2 → strip → \s+ to one U+0020 → strip`. ftfy 6.3.1's
 * `fix_text` splits the input "right after a newline" and "every 1,000,000 code points", and runs
 * the following round on each segment until a fixed point:
 *
 *   unescape_html → fix_encoding (heuristic) → fix_c1_controls → fix_latin_ligatures →
 *   fix_character_width → uncurl_quotes → fix_line_breaks → fix_surrogates →
 *   remove_terminal_escapes → remove_control_chars → NFC
 *
 * This module has only the deterministic steps (the four table lookups, line breaks, terminal
 * escapes, NFC, whitespace collapsing). Steps without tables and the heuristics are pinned to no-ops
 * by **rejecting the inputs they act on** — for an input that is not rejected, upstream's step does
 * nothing, so the remaining steps alone give upstream's string (no silent approximation). Rejected
 * inputs ({@link PromptCleanRejectReason}):
 *
 * - `"c1"` — C1 control characters (U+0080 to 009F). `fix_c1_controls` re-reads C1 as
 *   Windows-1252. Neither the table outputs nor NFC produce C1, so one check at the entry suffices.
 * - `"entity"` — sequences that can become HTML character references. Rejected if ftfy's
 *   `HTML_ENTITY_RE` (`&#?[0-9A-Za-z]{1,24};`) matches at the start of a round, or if
 *   `html.unescape`'s `_charref` (`&` followed by `#` + digit, `#x` + hex digit, or a character
 *   other than `\t \n \f space < & # ;`) matches after `fix_text`. Without a match, ftfy's
 *   `unescape_html` and both `html.unescape` calls do nothing. There is no name table, so sequences
 *   that do not become an entity, like `R&D`, are rejected too (`R & D` with spaces passes).
 * - `"mojibake"` — "not ASCII, and ftfy's `BADNESS_RE` matches" at the start of a round. Without a
 *   match `fix_encoding` does nothing (`ftfy/__init__.py:482`). The recipe translates `BADNESS_RE`
 *   from its syntax tree into a JS regular expression made only of `\u{…}` literals and bakes it into
 *   the asset (so the meaning of `\w` / `\s` / `.` cannot split between Python and JS —
 *   `badness_terms` in `tools/export-recipes/wan/prompt_clean.py`).
 * - `"unassigned"` — code points unassigned in UCD 16.0.0. Not an ftfy step but a guard against NFC
 *   version differences: NFC is left to the JS engine's `normalize`, and for assigned characters
 *   Unicode's stability policy gives the same result across versions, but a newer engine can give a
 *   character unassigned in Python (UCD 16.0.0) a decomposition or combining class.
 *
 * Lone surrogates are rejected at the entry by `assertEncodableText` (`fix_surrogates` has no effect
 * on a Python str made from a valid JS string passed as UTF-8).
 *
 * The tables ({@link PromptCleanTables}) are only looked up from the asset the recipe bakes from
 * ftfy 6.3.1, `re` / `regex` and `unicodedata` 16.0.0; Unicode classification is not reimplemented in
 * TS. The order and shape of the steps copy the recipe's mirror `prompt_clean.clean`; the recipe
 * checks the mirror against upstream exhaustively + by random sampling, and
 * `wan_text_prompt_clean_test.ts` pins TS to those expectations.
 */

import { ModelInputError } from "../../errors.ts";
import { asRecord, assertCodePoint, asString, setUnique } from "../../text/asset-gates.ts";
import { assertEncodableText, toCodePoints } from "../../text/code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "../../text/code-ranges.ts";
import { sha256HexSync } from "../../text/sha256.ts";

/** The reason the preprocessing does not accept an input (ADR 0119 decision 2). */
export type PromptCleanRejectReason = "unassigned" | "c1" | "entity" | "mojibake";

/**
 * An input the preprocessing does not accept (an input error — derived from
 * {@link ModelInputError}).
 *
 * It is a separate type so that it carries {@link PromptCleanRejectReason} in a field (a UI can show
 * the matching fix, and tests can compare down to the reason with upstream's expectations).
 */
export class PromptCleanError extends ModelInputError {
  readonly reason: PromptCleanRejectReason;

  constructor(reason: PromptCleanRejectReason, message: string) {
    super(message);
    this.name = "PromptCleanError";
    this.reason = reason;
  }
}

/** The preprocessing tables the recipe baked (the asset's `promptClean`). */
export type PromptCleanTables = {
  /** The mapping of `fix_latin_ligatures` (`ﬁ` → `fi` etc.). */
  readonly ligatures: ReadonlyMap<number, string>;
  /** The mapping of `fix_character_width` (standard forms of full-width / half-width forms, U+3000 → U+0020). */
  readonly width: ReadonlyMap<number, string>;
  /** The characters `uncurl_quotes` maps to `'` / `"`. */
  readonly singleQuotes: CodeRanges;
  readonly doubleQuotes: CodeRanges;
  /** The characters `remove_control_chars` removes. */
  readonly controlChars: CodeRanges;
  /** `\d` of `remove_terminal_escapes` (Python's Unicode Nd — not only ASCII digits). */
  readonly ansiDigits: CodeRanges;
  /** The characters assigned in UCD 16.0.0. */
  readonly assigned: CodeRanges;
  /** The characters `str.strip()` removes (`str.isspace`). */
  readonly stripSpace: CodeRanges;
  /** `\s` of `regex` in `whitespace_clean`. */
  readonly collapseSpace: CodeRanges;
  /** The JS translation of `BADNESS_RE` (`u` flag, no `g` / `y` = stateless). */
  readonly badness: RegExp;
};

/**
 * The sha256 of the UTF-8 of the asset's `badness` (the JS translation of ftfy 6.3.1's `BADNESS_RE`
 * — the recipe's `emit_js`).
 *
 * MUST: match the expression against this value before running it. The asset is untrusted external
 * input (`fromPretrained` can point at any repo — `src/text/asset-gates.ts`), and accepting any
 * syntactically valid expression would let one like `(a+)+$` make backtracking grow exponentially in
 * the preprocessing (including the check of the default negative at construction) and stall the
 * event loop. The translation is a single one pinned to a version (ADR 0119 decision 2), so there is
 * no reason to accept arbitrary expressions. When the ftfy version or the translation changes,
 * update this value together with rebuilding the recipe output (it equals the parity fixture's
 * value).
 */
const FTFY_BADNESS_SHA256 = "7da149647fdaad8c293f44706c67b5649a6f76593cbcf9f23d9613680466312e";

/** ftfy's `max_decode_length` (the maximum number of code points per segment). */
const MAX_DECODE_LENGTH = 1_000_000;

/**
 * The upper limit of rounds to the fixed point. With upstream's tables it stops in 2 rounds (observed
 * in the recipe's random sampling — `maxRounds`). Not reaching it happens only with broken tables, so
 * it fails with a plain Error, not an input error (it never loops forever).
 */
const MAX_FIX_ROUNDS = 8;

const C1_FIRST = 0x80;
const C1_LAST = 0x9f;

/** ftfy's `HTML_ENTITY_RE` (verbatim from `ftfy/chardata.py:82`). */
const FTFY_ENTITY = /&#?[0-9A-Za-z]{1,24};/;

/**
 * A form equivalent to **whether** `html.unescape`'s `_charref` (Python 3.14
 * `html/__init__.py:118-120`) matches. `#[0-9]+;?` and the like are always satisfied once the first
 * character matches, so the first character is enough to test for existence (the recipe's pytest
 * confirms the equivalence on every code point).
 */
const HTML_CHARREF = /&(?:#[0-9]|#[xX][0-9a-fA-F]|[^\t\n\f <&#;])/u;

const ESC = 0x1b;
const LEFT_BRACKET = 0x5b;
const SEMICOLON = 0x3b;

const codePointLabel = (cp: number): string =>
  `U+${cp.toString(16).toUpperCase().padStart(4, "0")}`;

const isAscii = (text: string): boolean => {
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) > 0x7f) return false;
  }
  return true;
};

const isAsciiLetter = (cp: number): boolean =>
  (cp >= 0x41 && cp <= 0x5a) || (cp >= 0x61 && cp <= 0x7a);

/**
 * The segments of `fix_text` (split right after a newline; one segment is at most
 * {@link MAX_DECODE_LENGTH} code points).
 */
const splitSegments = (text: string): string[] => {
  const out: string[] = [];
  let start = 0;
  let count = 0;
  let index = 0;
  while (index < text.length) {
    const cp = text.codePointAt(index) as number;
    index += cp > 0xffff ? 2 : 1;
    count += 1;
    if (cp === 0x0a || count === MAX_DECODE_LENGTH) {
      out.push(text.slice(start, index));
      start = index;
      count = 0;
    }
  }
  if (start < text.length) out.push(text.slice(start));
  return out;
};

const translate = (text: string, map: ReadonlyMap<number, string>): string => {
  let out = "";
  for (const ch of text) out += map.get(ch.codePointAt(0) as number) ?? ch;
  return out;
};

/** `uncurl_quotes` (the two sets are disjoint and the replacements lie outside them — checked on the recipe's tables). */
const uncurlQuotes = (tables: PromptCleanTables, text: string): string => {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0) as number;
    out += inCodeRanges(tables.doubleQuotes, cp)
      ? '"'
      : inCodeRanges(tables.singleQuotes, cp)
        ? "'"
        : ch;
  }
  return out;
};

/** `fix_line_breaks` (verbatim from `ftfy/fixes.py` — the same replacement order too). */
const fixLineBreaks = (text: string): string =>
  text
    .replaceAll("\r\n", "\n")
    .replaceAll("\r", "\n")
    .replaceAll(" ", "\n")
    .replaceAll(" ", "\n")
    .replaceAll("\u0085", "\n");

/**
 * `remove_terminal_escapes` (removes `ESC [ (\d|;)* letter`). Like `re.sub` it tries from the left
 * and advances one character when there is no match (no overlap). Digits, `;` and letters are
 * disjoint, so backing up after a greedy read finds no other match.
 */
const removeTerminalEscapes = (tables: PromptCleanTables, text: string): string => {
  if (!text.includes("\x1b")) return text;
  const cps = toCodePoints(text);
  let out = "";
  let i = 0;
  while (i < cps.length) {
    if (cps[i] === ESC && cps[i + 1] === LEFT_BRACKET) {
      let j = i + 2;
      while (j < cps.length && (cps[j] === SEMICOLON || inCodeRanges(tables.ansiDigits, cps[j]))) {
        j += 1;
      }
      if (j < cps.length && isAsciiLetter(cps[j])) {
        i = j + 1;
        continue;
      }
    }
    out += String.fromCodePoint(cps[i]);
    i += 1;
  }
  return out;
};

const removeControlChars = (tables: PromptCleanTables, text: string): string => {
  let out = "";
  for (const ch of text) {
    if (!inCodeRanges(tables.controlChars, ch.codePointAt(0) as number)) out += ch;
  }
  return out;
};

/** `str.strip()` (removes `isspace` characters at both ends). */
const strip = (tables: PromptCleanTables, text: string): string => {
  const cps = toCodePoints(text);
  let start = 0;
  let end = cps.length;
  while (start < end && inCodeRanges(tables.stripSpace, cps[start])) start += 1;
  while (end > start && inCodeRanges(tables.stripSpace, cps[end - 1])) end -= 1;
  // No argument spreading (a long input would hit V8's argument limit with a RangeError).
  let out = "";
  for (let k = start; k < end; k++) out += String.fromCodePoint(cps[k]);
  return out;
};

/** `regex.sub(r"\s+", " ", text)`。 */
const collapse = (tables: PromptCleanTables, text: string): string => {
  let out = "";
  let inSpace = false;
  for (const ch of text) {
    if (inCodeRanges(tables.collapseSpace, ch.codePointAt(0) as number)) {
      if (!inSpace) out += " ";
      inSpace = true;
    } else {
      out += ch;
      inSpace = false;
    }
  }
  return out;
};

const rejectEntity = (label: string, found: string): PromptCleanError =>
  new PromptCleanError(
    "entity",
    `${label}: HTML の文字参照になりうる並び ${JSON.stringify(found)} を含む — 上流（ftfy / ` +
      "html.unescape）は文字参照を解除するが、ここは解除の表を持たないので受けない。" +
      "`&` の直後に空白を入れる（例 `R&D` → `R & D`）か、参照を元の文字に直してから渡す",
  );

/**
 * Runs one segment to the fixed point (ftfy's round without the steps made no-ops by
 * rejection).
 */
const fixSegment = (tables: PromptCleanTables, segment: string, label: string): string => {
  let text = segment;
  for (let round = 0; round < MAX_FIX_ROUNDS; round++) {
    const before = text;
    // unescape_html: reject if a sequence matches (otherwise a no-op).
    const entity = FTFY_ENTITY.exec(text);
    if (entity !== null) throw rejectEntity(label, entity[0]);
    // fix_encoding: reject if not ASCII and BADNESS_RE matches (otherwise a no-op).
    if (!isAscii(text)) {
      const bad = tables.badness.exec(text);
      if (bad !== null) {
        throw new PromptCleanError(
          "mojibake",
          `${label}: 文字化け（mojibake）に見える並び ${JSON.stringify(bad[0])} を含む — ` +
            "上流（ftfy）は推定で修復するが、その推定は移植していないので受けない。" +
            "元の文字に直してから渡す",
        );
      }
    }
    // fix_c1_controls / fix_surrogates: rejected at the entry, so no-ops.
    text = translate(text, tables.ligatures);
    text = translate(text, tables.width);
    text = uncurlQuotes(tables, text);
    text = fixLineBreaks(text);
    text = removeTerminalEscapes(tables, text);
    text = removeControlChars(tables, text);
    text = text.normalize("NFC");
    if (text === before) return text;
  }
  throw new Error(
    `前処理が ${MAX_FIX_ROUNDS} 周で不動点に届かない — 前処理の表（資産の promptClean）が壊れている`,
  );
};

/**
 * Returns the same string as upstream's `prompt_clean`. An input that is not accepted throws
 * {@link PromptCleanError} (only a lone surrogate throws `assertEncodableText`'s
 * {@link ModelInputError}).
 *
 * `label` appears in the rejection message (in a form that tells positive from negative).
 */
export const cleanPrompt = (
  tables: PromptCleanTables,
  text: string,
  label: string = "プロンプト",
): string => {
  assertEncodableText(text, label);
  for (const ch of text) {
    const cp = ch.codePointAt(0) as number;
    if (!inCodeRanges(tables.assigned, cp)) {
      throw new PromptCleanError(
        "unassigned",
        `${label}: Unicode 16.0.0 で未割り当てのコードポイント ${codePointLabel(cp)} を含む — ` +
          "正規化（NFC）の結果が実行環境の Unicode の版で割れうるので受けない。その文字を外して渡す",
      );
    }
    if (cp >= C1_FIRST && cp <= C1_LAST) {
      throw new PromptCleanError(
        "c1",
        `${label}: C1 制御文字 ${codePointLabel(cp)} を含む — 上流（ftfy）は文字化けの手掛かりと` +
          "して別の文字へ読み直すが、その推定は移植していないので受けない。その文字を外して渡す",
      );
    }
  }
  let fixed = "";
  for (const segment of splitSegments(text)) fixed += fixSegment(tables, segment, label);
  // html.unescape × 2: reject if a sequence matches (otherwise both calls are no-ops).
  const charref = HTML_CHARREF.exec(fixed);
  if (charref !== null) throw rejectEntity(label, charref[0]);
  return strip(tables, collapse(tables, strip(tables, fixed)));
};

/**
 * Turns a list of `[cp, replacement]` pairs into a map (checks the key range, duplicates and
 * surrogates in the replacement).
 */
const parseCharMap = (raw: unknown, label: string): Map<number, string> => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 配列でない`);
  const out = new Map<number, string>();
  for (const entry of raw) {
    if (
      !Array.isArray(entry) ||
      entry.length !== 2 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "string"
    ) {
      throw new Error(`${label}: [cp, 文字列] でない`);
    }
    assertCodePoint(entry[0], label);
    // A lone surrogate in the replacement would split NFC and the later steps from upstream (a
    // Python str).
    toCodePoints(entry[1]);
    setUnique(out, entry[0], entry[1], label);
  }
  return out;
};

/**
 * Converts the asset's `promptClean` into tables (an external boundary, so the structure is checked
 * before use).
 *
 * MUST: range tables go through the range check of `parseCodeRanges` (a broken table raises no
 * exception but becomes "silently a different character classification", and only the
 * preprocessing output splits from upstream).
 */
export const parsePromptCleanTables = (raw: unknown, label: string): PromptCleanTables => {
  const obj = asRecord(raw, label);
  const source = asString(obj["badness"], `${label}.badness`);
  const digest = sha256HexSync(new TextEncoder().encode(source));
  if (digest !== FTFY_BADNESS_SHA256) {
    throw new Error(
      `${label}.badness: 正規表現が ftfy 6.3.1 の BADNESS_RE の固定の翻訳と違う（sha256 ${digest}・` +
        `期待 ${FTFY_BADNESS_SHA256}）— 資産の任意の式は実行しない`,
    );
  }
  let badness: RegExp;
  // The sha256 match above narrows the accepted expression to one fixed expression, whose
  // construction does not fail on the engines at hand. The catch stays because the fixed expression
  // uses `\u{…}` code point escapes (which need the u flag), so construction can fail on an engine
  // that does not understand the u flag — then it fails in the asset's context rather than as a
  // bare SyntaxError.
  try {
    // No `g` / `y` — `exec` / `test` keep no lastIndex and search from the start on every call.
    badness = new RegExp(source, "u");
  } catch (cause) {
    throw new Error(`${label}.badness: 正規表現として読めない`, { cause });
  }
  return {
    ligatures: parseCharMap(obj["ligatures"], `${label}.ligatures`),
    width: parseCharMap(obj["width"], `${label}.width`),
    singleQuotes: parseCodeRanges(obj["singleQuotes"], `${label}.singleQuotes`),
    doubleQuotes: parseCodeRanges(obj["doubleQuotes"], `${label}.doubleQuotes`),
    controlChars: parseCodeRanges(obj["controlChars"], `${label}.controlChars`),
    ansiDigits: parseCodeRanges(obj["ansiDigits"], `${label}.ansiDigits`),
    assigned: parseCodeRanges(obj["assigned"], `${label}.assigned`),
    stripSpace: parseCodeRanges(obj["stripSpace"], `${label}.stripSpace`),
    collapseSpace: parseCodeRanges(obj["collapseSpace"], `${label}.collapseSpace`),
    badness,
  };
};

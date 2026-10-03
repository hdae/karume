/**
 * Wan2.1 の前処理 `prompt_clean`（diffusers 0.39.0）の鏡像 — ADR 0119 決定 2（案 b'）。
 *
 * 上流は `ftfy.fix_text → html.unescape × 2 → strip → \s+ を U+0020 1 個へ → strip`。ftfy 6.3.1 の
 * `fix_text` は入力を「改行の直後」と「1,000,000 コードポイントごと」で区切り、区切りごとに
 * 次の 1 周を不動点まで回す:
 *
 *   unescape_html → fix_encoding（ヒューリスティック）→ fix_c1_controls → fix_latin_ligatures →
 *   fix_character_width → uncurl_quotes → fix_line_breaks → fix_surrogates →
 *   remove_terminal_escapes → remove_control_chars → NFC
 *
 * ここが持つのは決定的な処理だけ（表引きの 4 本・改行・端末エスケープ・NFC・空白の畳み込み）。
 * 表を持たない処理とヒューリスティックは、**それが効く入力を拒む**ことで no-op に固定する —
 * 拒まなかった入力では上流のその処理が何もしないので、残りだけで上流と同じ文字列になる
 * （黙って近似しない）。拒む入力（{@link PromptCleanRejectReason}）:
 *
 * - `"c1"` — C1 制御文字（U+0080〜009F）。`fix_c1_controls` は C1 を Windows-1252 として読み直す。
 *   表の出力と NFC は C1 を作らないので、入口で 1 度見れば足りる。
 * - `"entity"` — HTML の文字参照になりうる並び。各周の先頭で ftfy の `HTML_ENTITY_RE`
 *   （`&#?[0-9A-Za-z]{1,24};`）が当たるか、`fix_text` の後で `html.unescape` の `_charref`
 *   （`&` の直後が `#` + 数字・`#x` + 16 進・`\t \n \f 空白 < & # ;` 以外の文字）が当たれば拒む。
 *   当たらなければ ftfy の `unescape_html` と 2 回の `html.unescape` はどれも何もしない。名前表は
 *   持たないので、`R&D` のように entity にならない並びも拒む（`R & D` と空白を挟めば通る）。
 * - `"mojibake"` — 各周の先頭で「ASCII でなく、ftfy の `BADNESS_RE` が当たる」。当たらなければ
 *   `fix_encoding` は何もしない（`ftfy/__init__.py:482`）。`BADNESS_RE` は recipe が構文木から
 *   `\u{…}` の字面だけの JS 正規表現へ訳して資産に焼く（`\w` / `\s` / `.` の意味が Python と JS で
 *   割れないように — `tools/export-recipes/wan/prompt_clean.py` の `badness_terms`）。
 * - `"unassigned"` — UCD 16.0.0 で未割り当てのコードポイント。ftfy の処理ではなく NFC の版差の
 *   防壁: NFC は JS エンジンの `normalize` に任せ、割り当て済みの文字なら Unicode の安定性方針で
 *   版をまたいで同じ結果になるが、Python（UCD 16.0.0）で未割り当ての文字は新しいエンジンが
 *   分解・結合クラスを持ちうる。
 *
 * 孤立サロゲートは入口の `assertEncodableText` が拒む（`fix_surrogates` は JS の正当な文字列を
 * UTF-8 で渡した Python の str には効かない）。
 *
 * 表（{@link PromptCleanTables}）は recipe が ftfy 6.3.1・`re` / `regex`・`unicodedata` 16.0.0 から
 * 焼いた資産を引くだけで、Unicode の判定を TS で再実装しない。処理の順と形は recipe の鏡像
 * `prompt_clean.clean` の写経で、鏡像と上流の一致は recipe が網羅 + 乱択で検査し、TS とその
 * 期待の一致は `wan_text_prompt_clean_*_test.ts` が縛る。
 */

import { ModelInputError } from "../../errors.ts";
import { asRecord, assertCodePoint, asString, setUnique } from "../../text/asset-gates.ts";
import { assertEncodableText, toCodePoints } from "../../text/code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "../../text/code-ranges.ts";

/** 前処理が受けない入力の理由（ADR 0119 決定 2）。 */
export type PromptCleanRejectReason = "unassigned" | "c1" | "entity" | "mojibake";

/**
 * 前処理が受けない入力（入力起因 — {@link ModelInputError} の派生）。
 *
 * 型を分けるのは {@link PromptCleanRejectReason} を欄で運ぶため（UI が直し方を出し分けられる・
 * テストが上流の期待と理由まで突き合わせられる）。
 */
export class PromptCleanError extends ModelInputError {
  readonly reason: PromptCleanRejectReason;

  constructor(reason: PromptCleanRejectReason, message: string) {
    super(message);
    this.name = "PromptCleanError";
    this.reason = reason;
  }
}

/** recipe が焼いた前処理の表（資産の `promptClean`）。 */
export type PromptCleanTables = {
  /** `fix_latin_ligatures` の写像（`ﬁ` → `fi` など）。 */
  readonly ligatures: ReadonlyMap<number, string>;
  /** `fix_character_width` の写像（全角・半角形の標準形・U+3000 → U+0020）。 */
  readonly width: ReadonlyMap<number, string>;
  /** `uncurl_quotes` が `'` / `"` へ寄せる文字。 */
  readonly singleQuotes: CodeRanges;
  readonly doubleQuotes: CodeRanges;
  /** `remove_control_chars` が消す文字。 */
  readonly controlChars: CodeRanges;
  /** `remove_terminal_escapes` の `\d`（Python の Unicode の Nd — ASCII の数字だけではない）。 */
  readonly ansiDigits: CodeRanges;
  /** UCD 16.0.0 で割り当て済みの文字。 */
  readonly assigned: CodeRanges;
  /** `str.strip()` が落とす文字（`str.isspace`）。 */
  readonly stripSpace: CodeRanges;
  /** `whitespace_clean` の `regex` の `\s`。 */
  readonly collapseSpace: CodeRanges;
  /** `BADNESS_RE` の JS 訳（`u` フラグ・`g` / `y` なし = 状態を持たない）。 */
  readonly badness: RegExp;
};

/** ftfy の `max_decode_length`（区切りの最大コードポイント数）。 */
const MAX_DECODE_LENGTH = 1_000_000;

/**
 * 不動点の周回の上限。上流の表では 2 周で止まる（recipe の乱択の観測 — `maxRounds`）。届かない
 * のは表が壊れているときだけなので、入力起因ではない素の Error で落とす（無限に回さない）。
 */
const MAX_FIX_ROUNDS = 8;

const C1_FIRST = 0x80;
const C1_LAST = 0x9f;

/** ftfy の `HTML_ENTITY_RE`（`ftfy/chardata.py:82` の逐語）。 */
const FTFY_ENTITY = /&#?[0-9A-Za-z]{1,24};/;

/**
 * `html.unescape` の `_charref`（Python 3.14 `html/__init__.py:118-120`）が**当たるかどうか**と
 * 同値な形。`#[0-9]+;?` などは 1 文字目が当たれば必ず満たせるので、存在判定には 1 文字目だけで
 * よい（同値は recipe の pytest が全コードポイントで確かめる）。
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

/** `fix_text` の区切り（改行の直後で切る・1 区切りは最大 {@link MAX_DECODE_LENGTH} コードポイント）。 */
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

/** `uncurl_quotes`（2 つの集合は交わらず、置換先は集合の外 — recipe の表の検査）。 */
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

/** `fix_line_breaks`（`ftfy/fixes.py` の逐語 — 置換の順も同じ）。 */
const fixLineBreaks = (text: string): string =>
  text.replaceAll("\r\n", "\n").replaceAll("\r", "\n").replaceAll(" ", "\n")
    .replaceAll(" ", "\n").replaceAll("\u0085", "\n");

/**
 * `remove_terminal_escapes`（`ESC [ (\d|;)* 英字` を消す）。`re.sub` と同じく左から試し、
 * 当たらなければ 1 文字進む（重ならない）。数字と `;` と英字は交わらないので、貪欲に読んだ後に
 * 戻っても別の一致は無い。
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

/** `str.strip()`（両端の `isspace` を落とす）。 */
const strip = (tables: PromptCleanTables, text: string): string => {
  const cps = toCodePoints(text);
  let start = 0;
  let end = cps.length;
  while (start < end && inCodeRanges(tables.stripSpace, cps[start])) start += 1;
  while (end > start && inCodeRanges(tables.stripSpace, cps[end - 1])) end -= 1;
  // 引数展開しない（長い入力で V8 の引数上限の RangeError になる）。
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

/** 1 区切りを不動点まで回す（ftfy の 1 周から、拒んで no-op にした処理を除いたもの）。 */
const fixSegment = (tables: PromptCleanTables, segment: string, label: string): string => {
  let text = segment;
  for (let round = 0; round < MAX_FIX_ROUNDS; round++) {
    const before = text;
    // unescape_html: 当たる並びがあれば拒む（無ければ no-op）。
    const entity = FTFY_ENTITY.exec(text);
    if (entity !== null) throw rejectEntity(label, entity[0]);
    // fix_encoding: ASCII でなく BADNESS_RE が当たれば拒む（無ければ no-op）。
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
    // fix_c1_controls / fix_surrogates: 入口で拒んだので no-op。
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
 * 上流の `prompt_clean` と同じ文字列を返す。受けない入力は {@link PromptCleanError}
 * （孤立サロゲートだけは `assertEncodableText` の {@link ModelInputError}）。
 *
 * `label` は拒否のメッセージに出す（正 / ネガティブのどちらかが判る形にする）。
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
  // html.unescape × 2: 当たる並びがあれば拒む（無ければ 2 回とも no-op）。
  const charref = HTML_CHARREF.exec(fixed);
  if (charref !== null) throw rejectEntity(label, charref[0]);
  return strip(tables, collapse(tables, strip(tables, fixed)));
};

/** `[cp, 置換文字列]` の対の列を写像へ（鍵の域と重複・置換文字列のサロゲートまで見る）。 */
const parseCharMap = (raw: unknown, label: string): Map<number, string> => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 配列でない`);
  const out = new Map<number, string>();
  for (const entry of raw) {
    if (
      !Array.isArray(entry) || entry.length !== 2 ||
      typeof entry[0] !== "number" || typeof entry[1] !== "string"
    ) {
      throw new Error(`${label}: [cp, 文字列] でない`);
    }
    assertCodePoint(entry[0], label);
    // 置換文字列が孤立サロゲートを持つと、NFC と以降の処理が上流（Python の str）と割れる。
    toCodePoints(entry[1]);
    setUnique(out, entry[0], entry[1], label);
  }
  return out;
};

/**
 * 資産の `promptClean` を表に変換する（外部境界なので構造を検査してから使う）。
 *
 * MUST: 区間表は `parseCodeRanges` の値域門を通す（壊れた表は例外にならず「静かに別の文字分類」
 * になり、前処理の出力だけが上流と割れる）。
 */
export const parsePromptCleanTables = (raw: unknown, label: string): PromptCleanTables => {
  const obj = asRecord(raw, label);
  const source = asString(obj["badness"], `${label}.badness`);
  let badness: RegExp;
  try {
    // `g` / `y` を付けない — `exec` / `test` が lastIndex を持たず、呼ぶたびに先頭から探す。
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

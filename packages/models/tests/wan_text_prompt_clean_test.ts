// Wan2.1 の前処理（src/wan/text/prompt-clean.ts — 上流 prompt_clean の鏡像）の振る舞いと乱択の
// パリティ（ADR 0119 決定 2）。GPU も実資産も要らない。
//
// 乱択の期待はフィクスチャ `wan-text/prompt-clean-sweep.json` の digest（recipe が上流の
// `prompt_clean` / `BADNESS_RE` から採った値）。ここは recipe と同じ PRNG・池・手順で同じ列を作り、
// 決着（受理なら出力・拒否なら理由）の digest を突き合わせる。調査 §3.3 の池と本数以上:
// 安全な池（`&` と BADNESS_RE の字面の文字を含まない）200,000 本・全池（字面の 449 文字と `&` を
// 混ぜる）50,000 本・entity の門を叩く池 20,000 本。
//
// The all-codepoint sweeps (BADNESS_RE `\w` / `\s` / `.`, single-character prompt_clean, NFC
// version drift) are the describe block at the end of this file.

import { describe, test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { inCodeRanges } from "../src/text/code-ranges.ts";
import { cleanPrompt, PromptCleanError } from "../src/wan/text/prompt-clean.ts";
import {
  allCodePoints,
  chunkedDigests,
  digestBits,
  expandRanges,
  fuzzStrings,
  loadPromptCleanTables,
  loadSweepFixture,
  mismatchedIndices,
  recordOf,
  sweepBlocks,
  xorshift32,
} from "./helpers/wan-prompt-clean-fixture.ts";

const fixture = await loadSweepFixture();
const tables = await loadPromptCleanTables();
const clean = (text: string): string => cleanPrompt(tables, text);

const rejectReason = (text: string): string => {
  try {
    clean(text);
  } catch (error) {
    if (error instanceof PromptCleanError) return error.reason;
    throw error;
  }
  throw new Error(`拒否されなかった: ${JSON.stringify(text)}`);
};

// ---- 決定的な処理（期待は recipe の上流 prompt_clean で確かめた値）-------------

test("表引き: 全角・曲がった引用符・合字・端末エスケープ・改行が上流と同じ文字列になる", () => {
  assertEquals(clean("色调艳丽，过曝"), "色调艳丽,过曝", "公式 negative の全角読点");
  assertEquals(clean("“quoted” it’s"), '"quoted" it\'s');
  // `ŉ` は合字表で `ʼn`（U+02BC）になり、同じ周の uncurl_quotes で `'n` になる（表の順が効く）。
  assertEquals(clean("ŉ"), "'n");
  assertEquals(clean("ﬁne"), "fine");
  assertEquals(clean("\x1b[31mred\x1b[0m"), "red");
  assertEquals(clean("a\r\nb\rc d"), "a b c d");
  assertEquals(clean("ＡＢＣ　ｶﾀｶﾅ"), "ABC カタカナ");
  assertEquals(clean("a\x07b﻿c"), "abc");
  assertEquals(clean("x​y"), "x​y", "ゼロ幅空白は空白ではない");
});

test("端末エスケープの数字は Python の `\\d`（Unicode の Nd・面をまたぐ）", () => {
  // ASCII の数字だけで判定すると ESC と `[` を消し残し、制御文字の削除で `[` が残る。
  assertEquals(clean("\x1b[١mred"), "red", "アラビア・インド数字");
  assertEquals(clean("\x1b[\u{1D7CE}mred"), "red", "数学用の太字数字（U+1D7CE）");
  // 英字で閉じない並びは消さず、ESC だけが制御文字として消える。
  assertEquals(clean("\x1b[1;2xok\x1b["), "ok[");
});

test("区切り: 1,000,000 コードポイントで切れた所では NFC が合成しない（ftfy の区切りどおり）", () => {
  // 上流（実測）: 空白 999,999 個 + e + U+0301 は 1 区切り目が e で終わり、U+0301 が次の区切りに
  // 落ちるので合成されない。1 個少なければ同じ区切りに入って é になる。
  assertEquals(clean(" ".repeat(999_999) + "é"), "é");
  assertEquals(clean(" ".repeat(999_998) + "é"), "é");
});

test("各周の先頭で門を見る: NFC や全角の表が作った並びも拒む", () => {
  // 1 周目の入力には当たらないが、NFC が à À を作った 2 周目で BADNESS_RE が当たる
  // （上流は fix_encoding が試して戻すだけ = 黙って通すが、ここは推定を移植しないので拒む）。
  assertEquals(rejectReason("àÀ"), "mojibake");
  // 全角の `＆` が 1 周目で `&` になり、2 周目に ftfy の unescape_html が `&amp;` を解く形。
  assertEquals(rejectReason("＆amp;"), "entity");
});

test("entity の門: `&` の直後が文字なら拒み、空白・改行なら受ける", () => {
  assertEquals(rejectReason("salt &amp; pepper"), "entity");
  assertEquals(rejectReason("R&D"), "entity", "entity にならない並びも拒む側へ倒す");
  assertEquals(rejectReason("x &#65 y"), "entity", "セミコロン無しの数値参照（上流は A に解く）");
  assertEquals(clean("salt & pepper"), "salt & pepper");
  assertEquals(clean("a &\rb"), "a & b", "CR は改行になってから html.unescape を通る");
  assertEquals(clean("a &　b"), "a & b");
});

test("拒否: mojibake・C1・未割り当ては PromptCleanError（入力起因）で理由を運ぶ", () => {
  assertEquals(rejectReason("cafÃ©"), "mojibake");
  assertEquals(rejectReason("a\x85b"), "c1");
  assertEquals(rejectReason("a͸b"), "unassigned");
  const error = assertThrows(() => cleanPrompt(tables, "a\x85b", "ネガティブ"), ModelInputError);
  assert(error instanceof PromptCleanError);
  assert(error.message.includes("ネガティブ"), error.message);
  assert(error.message.includes("U+0085"), error.message);
});

test("孤立サロゲートは前処理の前に拒む（PromptCleanError ではなく ModelInputError）", () => {
  const error = assertThrows(() => clean("a\ud800b"), ModelInputError, "サロゲート");
  assert(!(error instanceof PromptCleanError));
});

test("壊れた表で不動点に届かなければ素の Error で落ちる（無限に回さない）", () => {
  // 全角 A ↔ A の循環（上流の表には無い — 資産の破損だけが作る形）。
  const cyclic = {
    ...tables,
    width: new Map([
      [0x41, "Ａ"],
      [0xff21, "A"],
    ]),
  };
  const error = assertThrows(() => cleanPrompt(cyclic, "A"), Error, "不動点");
  assert(!(error instanceof ModelInputError), "資産の破損は入力起因ではない");
});

// ---- 乱択（recipe と同じ列を再生して digest を突き合わせる）--------------------

test("PRNG: xorshift32 が recipe と同じ列を出す", () => {
  const next = xorshift32(fixture.prng.seed);
  assertEquals(Array.from({ length: fixture.prng.first.length }, next), fixture.prng.first);
});

for (const run of fixture.clean) {
  test(`乱択 [${run.name}] ${run.count} 本の決着が上流 prompt_clean と一致`, async () => {
    const pool = expandRanges(fixture.pools[run.pool]);
    const records: string[] = [];
    for (const text of fuzzStrings(pool, run.seed, run.count, run.maxLength)) {
      records.push(recordOf(tables, text));
    }
    const outcomes: Record<string, number> = {};
    for (const item of records) {
      const key = item.startsWith("A") ? "accepted" : item.slice(1);
      outcomes[key] = (outcomes[key] ?? 0) + 1;
    }
    assertEquals(outcomes, run.outcomes, "受理 / 拒否の内訳");
    const chunks = await chunkedDigests(records, run.chunk);
    assertEquals(mismatchedIndices(chunks, run.chunks), [], "上流と割れた塊（chunk 番号）");
  });
}

test("安全な池は全件受理（拒否の門が普通のプロンプトを落とさない）", () => {
  const safe = fixture.clean.find((run) => run.name === "safe");
  assertEquals(safe?.outcomes, { accepted: safe?.count ?? -1 });
});

for (const check of fixture.badness.filter((entry) => entry.template === undefined)) {
  test(`BADNESS_RE の JS 訳 [乱択 ${check.name}] が上流の is_bad と全件一致`, async () => {
    const pool = expandRanges(fixture.pools[check.pool as string]);
    let bits = "";
    for (const text of fuzzStrings(
      pool,
      check.seed as number,
      check.count as number,
      check.maxLength as number,
    )) {
      bits += tables.badness.test(text) ? "1" : "0";
    }
    assertEquals(Array.from(bits).filter((bit) => bit === "1").length, check.hits, "当たった本数");
    assertEquals(await digestBits(bits), check.digest);
  });
}

// ---- 全コードポイントの掃引 ----------------------------------------------------
//
// Wan2.1 の前処理（src/wan/text/prompt-clean.ts）の全コードポイント掃引（ADR 0119 決定 2）。
// GPU も実資産も要らない。期待はフィクスチャ `wan-text/prompt-clean-sweep.json` の digest
// （recipe が上流から採った値 — 4,096 コードポイントの塊ごと）。
//
//  ・BADNESS_RE の JS 訳: Unicode に依る構文（`\w`・`\s`・`.`・`[^…]`・`^`）を叩く文脈に全コード
//    ポイントを入れ、上流の `is_bad` と当たり外れが全件一致する
//  ・前処理の 1 文字: `{}` と `a{}b` に全コードポイントを入れ、決着（受理なら出力・拒否なら理由）が
//    上流の prompt_clean と一致する（表・空白集合・門の取りこぼしを網羅で拾う）
//  ・NFC の版差: Python（UCD 16.0.0）で割り当て済みの全コードポイントを 8 文脈に入れ、JS エンジンの
//    `normalize("NFC")` が `unicodedata.normalize` と一致する。外れたら、その塊の文字を前処理の表で
//    拒む / 焼く対応が要る（ADR 0119 決定 2 — 実行エンジンの Unicode の版が変わったときの門）

describe("all-codepoint sweeps", () => {
  const everyCodePoint = allCodePoints();

  /**
   * `template` の `{}` を**全部** 1 文字に置き換える（Python の `str.replace` と同じ — JS の
   * `replace` は最初の 1 個だけで、置換文字列の `$` も特別扱いするので使わない）。
   */
  const fill = (template: string, cp: number): string =>
    template.split("{}").join(String.fromCodePoint(cp));

  /** 塊の先頭コードポイント（外れた塊をメッセージで名指しする）。 */
  const blockStarts = (indices: readonly number[]): string[] =>
    indices.map(
      (index) => `U+${(index * fixture.block).toString(16).toUpperCase().padStart(4, "0")}`,
    );

  for (const check of fixture.badness.filter((entry) => entry.template !== undefined)) {
    test(`BADNESS_RE の JS 訳 [掃引 ${check.name}: ${check.template}] が is_bad と全件一致`, async () => {
      const template = check.template as string;
      let bits = "";
      for (const cp of everyCodePoint) {
        bits += tables.badness.test(fill(template, cp)) ? "1" : "0";
      }
      assertEquals(
        Array.from(bits).filter((bit) => bit === "1").length,
        check.hits,
        "当たった本数",
      );
      assertEquals(await digestBits(bits), check.digest);
    });
  }

  for (const sweep of fixture.cleanSweeps) {
    test(`前処理の掃引 [${sweep.name}: ${sweep.template}] が上流 prompt_clean と全件一致`, async () => {
      const blocks = await sweepBlocks(
        everyCodePoint,
        (cp) => recordOf(tables, fill(sweep.template, cp)),
        fixture.block,
      );
      assertEquals(blockStarts(mismatchedIndices(blocks, sweep.blocks)), [], "上流と割れた塊");
    });
  }

  const assigned = everyCodePoint.filter((cp) => inCodeRanges(tables.assigned, cp));

  for (const sweep of fixture.nfcSweeps) {
    test(`NFC の掃引 [${sweep.name}] が Python（UCD 16.0.0）と全件一致`, async () => {
      const blocks = await sweepBlocks(
        assigned,
        (cp) => fill(sweep.template, cp).normalize("NFC"),
        fixture.block,
      );
      assertEquals(blockStarts(mismatchedIndices(blocks, sweep.blocks)), [], "Python と割れた塊");
    });
  }
});

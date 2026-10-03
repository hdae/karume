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

import { assertEquals } from "@std/assert";
import { inCodeRanges } from "../src/text/code-ranges.ts";
import {
  allCodePoints,
  digestBits,
  loadPromptCleanTables,
  loadSweepFixture,
  mismatchedIndices,
  recordOf,
  sweepBlocks,
} from "./helpers/wan-prompt-clean-fixture.ts";

const fixture = await loadSweepFixture();
const tables = await loadPromptCleanTables();
const everyCodePoint = allCodePoints();

/**
 * `template` の `{}` を**全部** 1 文字に置き換える（Python の `str.replace` と同じ — JS の
 * `replace` は最初の 1 個だけで、置換文字列の `$` も特別扱いするので使わない）。
 */
const fill = (template: string, cp: number): string =>
  template.split("{}").join(String.fromCodePoint(cp));

/** 塊の先頭コードポイント（外れた塊をメッセージで名指しする）。 */
const blockStarts = (indices: readonly number[]): string[] =>
  indices.map((index) =>
    `U+${(index * fixture.block).toString(16).toUpperCase().padStart(4, "0")}`
  );

for (const check of fixture.badness.filter((entry) => entry.template !== undefined)) {
  Deno.test(`BADNESS_RE の JS 訳 [掃引 ${check.name}: ${check.template}] が is_bad と全件一致`, async () => {
    const template = check.template as string;
    let bits = "";
    for (const cp of everyCodePoint) {
      bits += tables.badness.test(fill(template, cp)) ? "1" : "0";
    }
    assertEquals([...bits].filter((bit) => bit === "1").length, check.hits, "当たった本数");
    assertEquals(await digestBits(bits), check.digest);
  });
}

for (const sweep of fixture.cleanSweeps) {
  Deno.test(`前処理の掃引 [${sweep.name}: ${sweep.template}] が上流 prompt_clean と全件一致`, async () => {
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
  Deno.test(`NFC の掃引 [${sweep.name}] が Python（UCD 16.0.0）と全件一致`, async () => {
    const blocks = await sweepBlocks(
      assigned,
      (cp) => fill(sweep.template, cp).normalize("NFC"),
      fixture.block,
    );
    assertEquals(blockStarts(mismatchedIndices(blocks, sweep.blocks)), [], "Python と割れた塊");
  });
}

// umT5-XXL の単体のトークナイザ（src/umt5/tokenizer.ts と共通層 src/text/t5-tokenizer.ts）の Python 正本との
// パリティ検証。GPU も実資産も要らない（実資産は在れば全語彙でも回す）。
//
// フィクスチャ `umt5/tokenizer-cases.json` は recipe `tools/export-recipes/wan/umt5_xxl_tokenizer.py` が
// 本家 `google/umt5-xxl` の pin から生成する。中身は
//  ・ケースごとの期待（境界・多言語 + 乱択）。受理なら id 列、拒否なら理由。**id 列の正本は transformers 5.14.1 の
//    AutoTokenizer** で、受理したケースは tokenizer.json の経路の id 列も持つ（2 経路の一致は生成時に確認済み —
//    ここでも両方と突き合わせる）。前処理はしない — text をそのまま渡す
//  ・その再現に要る語彙の**部分集合**（全ケースの断片の部分文字列 — 格子が全語彙のときと同じになる）
//
// 縛るのは振る舞い 1 つ: 「同じ文字列を入れたら、上流と同じ id 列が出るか、同じ理由で拒む」。併せて、資産の
// 読み口の拒否と、Wan（前処理の後に同じトークナイザを通す）で空白の拒否が起きないことを見る。

import { assert, assertEquals, assertInstanceOf, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "../src/text/code-ranges.ts";
import type { T5Tables, T5VocabEntry } from "../src/text/t5-tokenizer.ts";
import { UMT5_MAX_LENGTH, UMT5_MIN_TOKENS } from "../src/umt5/relative-position.ts";
import {
  parseUmt5TokenizerAsset,
  UMT5_TOKENIZER_FORMAT,
  Umt5Tokenizer,
} from "../src/umt5/tokenizer.ts";
import { PromptCleanError } from "../src/wan/text/prompt-clean.ts";
import { WAN_TOKENIZER_FORMAT } from "../src/wan/text/tokenizer.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { wanParityEncoder } from "./helpers/wan-parity-encoder.ts";

type FixtureCase = {
  readonly id: string;
  readonly why: string;
  readonly text: string;
  readonly ids?: number[];
  readonly tokenizerJsonIds?: number[];
  readonly reject?: string;
};

type Fixture = {
  readonly assetFormat: string;
  readonly maxLength: number;
  readonly minTokens: number;
  readonly t5: {
    readonly addedTokens: [string, number][];
    readonly unkId: number;
    readonly eosId: number;
    readonly minScore: number;
    readonly maxTokenLength: number;
    readonly space: unknown;
    readonly vocab: [string, number, number][];
  };
  readonly cases: FixtureCase[];
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("./fixtures/umt5/tokenizer-cases.json", import.meta.url)),
) as Fixture;

const space: CodeRanges = parseCodeRanges(fixture.t5.space, "space");
const t5Tables: T5Tables = {
  vocab: new Map<string, T5VocabEntry>(
    fixture.t5.vocab.map(([token, id, score]) => [token, { id, score }]),
  ),
  // MUST: 部分集合から導かない — 未知ノードのスコアと探索幅は語彙**全体**から決まる。
  minScore: fixture.t5.minScore,
  maxTokenLength: fixture.t5.maxTokenLength,
  unkId: fixture.t5.unkId,
  eosId: fixture.t5.eosId,
  addedTokens: new Map(fixture.t5.addedTokens),
  space,
  maxLength: fixture.maxLength,
};
const tokenizer = new Umt5Tokenizer(t5Tables);

/** 拒否の理由 → メッセージの目印（理由ごとに他の理由の文言に現れない綴り）。 */
const REJECT_MARKERS: Record<string, string> = {
  "oov": "語彙に無い文字列",
  "too-long": `上限 ${UMT5_MAX_LENGTH}`,
  "too-short": `最低 ${UMT5_MIN_TOKENS}`,
  "added-token": "特殊トークン",
  "metaspace": "U+2581",
  "whitespace": "U+0020",
};

/** recipe が焼いた実資産（全語彙 — 在れば部分集合の格子に依らない突合も回す）。 */
const SERIES_ASSET = new URL(
  "../../../outputs/series/umt5-xxl-tokenizer/tokenizer.json",
  import.meta.url,
);
const seriesAsset = await readTextIfPresent(SERIES_ASSET);
if (seriesAsset === undefined) {
  console.warn(
    `[karume] ${SERIES_ASSET.pathname} が無いため umT5 の実資産での突合を SKIP する。生成: ` +
      "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_xxl_tokenizer",
  );
}

const caseById = (id: string): FixtureCase => {
  const found = fixture.cases.find((entry) => entry.id === id);
  if (found === undefined) throw new Error(`フィクスチャにケース ${id} が無い`);
  return found;
};

/** 拒否の期待（理由の目印を含む `ModelInputError`）。 */
const assertRejected = (
  run: () => unknown,
  reason: string,
  message?: string,
): ModelInputError => {
  const marker = REJECT_MARKERS[reason];
  assert(marker !== undefined, `未知の拒否の理由 ${reason}`);
  return assertThrows(run, ModelInputError, marker, message);
};

// ---- フィクスチャ本体 -------------------------------------------------------

Deno.test("フィクスチャが空でなく、資産の形式・上下限・全ての決着を持つ", () => {
  // 1 本目に置く。読み違い / 生成失敗で cases が空になると、以降のループが 0 回になって
  // 「緑だが何も検証していない」状態が黙って成立する。
  assert(fixture.cases.length >= 60, `ケース数 ${fixture.cases.length} が少なすぎる`);
  assertEquals(fixture.assetFormat, UMT5_TOKENIZER_FORMAT);
  assertEquals(fixture.maxLength, UMT5_MAX_LENGTH);
  assertEquals(fixture.minTokens, UMT5_MIN_TOKENS);
  assertEquals(fixture.t5.addedTokens.length, 304, "umT5 の追加語彙は 304 個");
  const reasons = new Set(fixture.cases.map((entry) => entry.reject ?? "accepted"));
  for (const reason of ["accepted", ...Object.keys(REJECT_MARKERS)]) {
    assert(reasons.has(reason), `決着 ${reason} のケースが無い`);
  }
});

for (const testCase of fixture.cases) {
  Deno.test(`パリティ [${testCase.id}] ${testCase.why}`, () => {
    if (testCase.ids !== undefined) {
      const ids = tokenizer.tokenize(testCase.text);
      assertInstanceOf(ids, Int32Array);
      assertEquals([...ids], testCase.ids, "transformers 5.14.1 の id 列");
      assertEquals([...ids], testCase.tokenizerJsonIds, "tokenizer.json の経路の id 列");
      return;
    }
    assertRejected(() => tokenizer.tokenize(testCase.text), testCase.reject as string);
  });
}

// ---- 境界（フィクスチャのケースが実際に叩いているもの）-----------------------

Deno.test("上限: 512 トークンちょうどは受理・513 は拒む（黙って切り詰めない）", () => {
  const max = tokenizer.tokenize(caseById("max-512").text);
  assertEquals(max.length, UMT5_MAX_LENGTH);
  assertEquals(max.at(-1), fixture.t5.eosId, "末尾は </s>");
  assertRejected(() => tokenizer.tokenize(caseById("over-513").text), "too-long");
});

Deno.test("下限: 2 トークンは受理・空文字（</s> だけ）は拒む", () => {
  assertEquals(tokenizer.tokenize("cat").length, UMT5_MIN_TOKENS);
  assertRejected(() => tokenizer.tokenize(""), "too-short");
});

Deno.test("空白: 先頭と語の間の U+0020 は受け、末尾の U+0020 と U+0020 以外の空白は拒む", () => {
  assertEquals(
    [...tokenizer.tokenize("  a cat")],
    caseById("leading-spaces").ids,
    "先頭の U+0020",
  );
  assertEquals(
    [...tokenizer.tokenize("a   cat  sits")],
    caseById("inner-spaces").ids,
    "語の間の連続した U+0020",
  );
  assertRejected(() => tokenizer.tokenize("a cat "), "whitespace", "末尾の U+0020");
  assertRejected(() => tokenizer.tokenize("   "), "whitespace", "空白だけ");
  // 拒否の文言は直す文字をコードポイントで名指しする（目に見えない文字を人が探せる形）。
  assertThrows(() => tokenizer.tokenize("猫　犬"), ModelInputError, "U+3000");
  assertThrows(() => tokenizer.tokenize("a\tcat"), ModelInputError, "U+0009");
});

Deno.test("拒否の順: 追加語彙 → 空白の直後の ▁ → 空白 → 語彙外（recipe の生成と同じ順）", () => {
  // 1 本に 2 つの違反を持たせ、先の理由で落ちることを見る（順が入れ替わると理由だけが変わる）。
  assertRejected(() => tokenizer.tokenize("a </s> ▁b\t"), "added-token");
  assertRejected(() => tokenizer.tokenize("a ▁b\tcat"), "metaspace");
  assertRejected(() => tokenizer.tokenize("a\t\u{2000b}"), "whitespace");
});

Deno.test("拒否の文言に呼び手のラベルが出る・孤立サロゲートは入力起因で拒む", () => {
  assertThrows(() => tokenizer.tokenize("\u{2000b}", "negative"), ModelInputError, "negative:");
  assertThrows(() => tokenizer.tokenize("a\ud800b"), ModelInputError, "サロゲート");
});

// ---- Wan の経路（前処理の後に同じトークナイザを通す）---------------------------

/** Wan のパリティ用フィクスチャの空白集合（Wan の資産の `space` — 前処理の後に通すトークナイザの表）。 */
const wanSpace: CodeRanges = parseCodeRanges(
  (JSON.parse(
    await Deno.readTextFile(new URL("./fixtures/wan-text/parity.json", import.meta.url)),
  ) as { readonly t5: { readonly space: unknown } }).t5.space,
  "wan.space",
);

Deno.test("Wan: 前処理の後の文字列には U+0020 以外の空白も末尾の空白も残らない（空白の拒否は起きない）", () => {
  // `Umt5Tokenizer` の空白の拒否は前処理を持たない経路のためのもので、Wan は `prompt_clean` の鏡像を通してから
  // 同じトークナイザへ渡す。前処理が資産の空白集合の全ての文字を U+0020 1 個へ畳んで両端を落とすなら、Wan の
  // 受理集合は共有の前と同じ（Wan の資産の空白の表で全文字を当てる）。
  const encoder = wanParityEncoder();
  const plainOnly = (text: string): boolean =>
    [...text].every((ch) => ch === " " || !inCodeRanges(wanSpace, ch.codePointAt(0) as number));
  let checked = 0;
  for (const [start, end] of wanSpace) {
    for (let cp = start; cp <= end; cp += 1) {
      const ch = String.fromCodePoint(cp);
      for (const text of [`a${ch}b`, `a${ch}`, `${ch}a`, `a ${ch} `, `${ch}`]) {
        let cleaned: string;
        try {
          cleaned = encoder.clean(text);
        } catch (error) {
          // 前処理が拒む文字（C1 の U+0085 など）はトークナイザまで届かない。
          assertInstanceOf(error, PromptCleanError);
          continue;
        }
        assert(plainOnly(cleaned), `${JSON.stringify(text)} → ${JSON.stringify(cleaned)}`);
        assert(!cleaned.endsWith(" "), `${JSON.stringify(text)} → ${JSON.stringify(cleaned)}`);
        checked += 1;
      }
    }
  }
  assert(checked >= 100, `検査した入力が少なすぎる（${checked}）`);
});

// ---- 資産の読み取り（外部境界）-----------------------------------------------

/** 最小の資産 JSON（語彙 6 行）。 */
const minimalAsset = (patch: Record<string, unknown> = {}): Record<string, unknown> => ({
  format: UMT5_TOKENIZER_FORMAT,
  maxLength: 512,
  unkId: 2,
  eosId: 1,
  addedTokens: [["<pad>", 0], ["</s>", 1], ["<s>", 2], ["<unk>", 3]],
  space: [[0x09, 0x0d], [0x20, 0x20]],
  vocabText: "<pad>\n</s>\n<s>\n<unk>\n▁cat\n▁",
  scores: [0, 0, 0, 0, -1, -2],
  ...patch,
});

Deno.test("parseUmt5TokenizerAsset: 資産 JSON から組んだトークナイザが符号化できる", () => {
  const tiny = new Umt5Tokenizer(parseUmt5TokenizerAsset(minimalAsset()));
  assertEquals(tiny.maxLength, 512);
  assertEquals([...tiny.tokenize("cat cat")], [4, 4, 1]);
  assertThrows(() => tiny.tokenize("dog"), ModelInputError, "語彙に無い文字列");
  assertThrows(() => tiny.tokenize("cat\tcat"), ModelInputError, "U+0009");
});

Deno.test("parseUmt5TokenizerAsset: 知らない形式の版は読まない（Wan の資産の形式も）", () => {
  for (const format of ["karume-umt5-tokenizer/2", WAN_TOKENIZER_FORMAT]) {
    assertThrows(
      () => parseUmt5TokenizerAsset(minimalAsset({ format })),
      Error,
      `形式 ${format} を読めない`,
    );
  }
  assertThrows(
    () => parseUmt5TokenizerAsset(minimalAsset({ format: undefined })),
    Error,
    "tokenizer.format",
  );
  assertThrows(() => parseUmt5TokenizerAsset("tokenizer"), Error, "オブジェクトでない");
});

Deno.test("parseUmt5TokenizerAsset: 壊れた表は落とす（入力起因ではない素の Error）", () => {
  const rejected: readonly [string, Record<string, unknown>, string][] = [
    ["スコアの本数違い", { scores: [0, 0, 0, 0, -1] }, "scores の長さ"],
    [
      "語彙の重複行",
      { vocabText: "<pad>\n</s>\n<s>\n<unk>\n▁cat\n▁cat" },
      "重複は配布の破損",
    ],
    ["非有限のスコア", { scores: [0, 0, 0, 0, -1, Number.NaN] }, "有限"],
    ["空白の表の重なり", { space: [[0x09, 0x20], [0x20, 0x20]] }, "重なる"],
    ["最大長が無い", { maxLength: undefined }, "maxLength"],
    ["eos が語彙の外", { eosId: 6 }, "eosId"],
  ];
  for (const [label, patch, message] of rejected) {
    const error = assertThrows(
      () => parseUmt5TokenizerAsset(minimalAsset(patch)),
      Error,
      message,
      label,
    );
    assert(!(error instanceof ModelInputError), `資産の齟齬を入力起因にしない: ${label}`);
  }
});

// ---- 実資産（全語彙）での突合 -------------------------------------------------

Deno.test({
  name: "実資産: 全語彙の資産から組んだトークナイザもフィクスチャの全ケースと同じ決着になる",
  ignore: seriesAsset === undefined,
  fn: () => {
    // フィクスチャの語彙は部分集合（断片の部分文字列）なので、資産の読み取り（25.6 万行）と全語彙の格子でも
    // 同じ id 列 / 拒否になることをここで見る。
    const full = new Umt5Tokenizer(parseUmt5TokenizerAsset(JSON.parse(seriesAsset as string)));
    assertEquals(full.maxLength, UMT5_MAX_LENGTH);
    for (const testCase of fixture.cases) {
      if (testCase.ids !== undefined) {
        assertEquals([...full.tokenize(testCase.text)], testCase.ids, testCase.id);
      } else {
        assertRejected(() => full.tokenize(testCase.text), testCase.reject as string, testCase.id);
      }
    }
  },
});

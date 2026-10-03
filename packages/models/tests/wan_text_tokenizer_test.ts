// Wan2.1 のプロンプト層（src/wan/text/ と共通層 src/text/t5-tokenizer.ts）の Python 正本との
// パリティ検証（ADR 0119 段 10a — 決定 1・2・4）。GPU も実資産も要らない。
//
// フィクスチャ `wan-text/parity.json` は recipe `tools/export-recipes/wan/umt5_tokenizer.py` が
// 生成する。中身は
//  ・ケースごとの期待（固定 4 本 + 境界 + 乱択）。受理なら上流 prompt_clean の出力と id 列、拒否なら
//    理由。**id 列の正本は transformers 5.14.1 の AutoTokenizer** で、受理したケースは tokenizer.json の
//    経路の id 列も持つ（2 経路の一致は生成時に確認済み — ここでも両方と突き合わせる）
//  ・その再現に要る語彙の**部分集合**（全ケースの断片の部分文字列 — 格子が全語彙のときと同じになる）
//  ・前処理の表（promptClean）の**全体**
//
// 縛るのは振る舞い 1 つ: 「同じ文字列を入れたら、上流と同じ id 列が出るか、同じ理由で拒む」。

import { assert, assertEquals, assertInstanceOf, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { parseCodeRanges } from "../src/text/code-ranges.ts";
import { type T5Tables, T5UnigramTokenizer, type T5VocabEntry } from "../src/text/t5-tokenizer.ts";
import { parsePromptCleanTables, PromptCleanError } from "../src/wan/text/prompt-clean.ts";
import {
  parseWanTokenizerAsset,
  WAN_PROMPT_MIN_TOKENS,
  WAN_TOKENIZER_FORMAT,
  WanPromptEncoder,
} from "../src/wan/text/tokenizer.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";

type FixtureCase = {
  readonly id: string;
  readonly why: string;
  readonly text: string;
  readonly cleaned?: string;
  readonly ids?: number[];
  readonly tokenizerJsonIds?: number[];
  readonly reject?: string;
};

type Fixture = {
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
  readonly promptClean: unknown;
  readonly cases: FixtureCase[];
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("./fixtures/wan-text/parity.json", import.meta.url)),
) as Fixture;

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
  space: parseCodeRanges(fixture.t5.space, "space"),
  maxLength: fixture.maxLength,
};
const promptClean = parsePromptCleanTables(fixture.promptClean, "promptClean");
const encoder = new WanPromptEncoder({ t5: t5Tables, promptClean });

/** トークナイザ側の拒否の理由 → メッセージの目印（前処理の拒否は `PromptCleanError.reason`）。 */
const TOKENIZER_REJECTS: Record<string, string> = {
  "oov": "語彙に無い文字列",
  "too-long": "上限 512",
  "too-short": `最低 ${WAN_PROMPT_MIN_TOKENS}`,
  "added-token": "特殊トークン",
  "metaspace": "U+2581",
};
const PROMPT_CLEAN_REJECTS = new Set(["unassigned", "c1", "entity", "mojibake"]);

/** recipe が焼いた実資産（全語彙 — 在れば部分集合の格子に依らない突合も回す）。 */
const seriesAsset = await readTextIfPresent(
  new URL("../../../outputs/series/wan2.1-umt5-tokenizer/tokenizer.json", import.meta.url),
);
if (seriesAsset === undefined) {
  console.warn(
    "[karume] outputs/series/wan2.1-umt5-tokenizer/tokenizer.json が無いため Wan の実資産での " +
      "突合を SKIP する。生成: cd tools/export-recipes && uv run --group wan --inexact python -m " +
      "wan.umt5_tokenizer",
  );
}

const caseById = (id: string): FixtureCase => {
  const found = fixture.cases.find((entry) => entry.id === id);
  if (found === undefined) throw new Error(`フィクスチャにケース ${id} が無い`);
  return found;
};

// ---- フィクスチャ本体 -------------------------------------------------------

Deno.test("フィクスチャが空でない（取り違えで全ケース素通しになっていない）", () => {
  // 1 本目に置く。読み違い / 生成失敗で cases が空になると、以降のループが 0 回になって
  // 「緑だが何も検証していない」状態が黙って成立する。
  assert(fixture.cases.length >= 200, `ケース数 ${fixture.cases.length} が少なすぎる`);
  assertEquals(fixture.maxLength, 512);
  assertEquals(fixture.minTokens, WAN_PROMPT_MIN_TOKENS);
  assertEquals(fixture.t5.addedTokens.length, 304, "umT5 の追加語彙は 304 個");
  const reasons = new Set(fixture.cases.map((entry) => entry.reject ?? "accepted"));
  for (const reason of ["accepted", ...PROMPT_CLEAN_REJECTS, ...Object.keys(TOKENIZER_REJECTS)]) {
    assert(reasons.has(reason), `決着 ${reason} のケースが無い`);
  }
});

for (const testCase of fixture.cases) {
  Deno.test(`パリティ [${testCase.id}] ${testCase.why}`, () => {
    if (testCase.cleaned !== undefined) {
      assertEquals(encoder.clean(testCase.text), testCase.cleaned, "前処理の出力");
    }
    if (testCase.ids !== undefined) {
      const ids = encoder.encode(testCase.text);
      assertInstanceOf(ids, Int32Array);
      assertEquals([...ids], testCase.ids, "transformers 5.14.1 の id 列");
      assertEquals([...ids], testCase.tokenizerJsonIds, "tokenizer.json の経路の id 列");
      return;
    }
    const reason = testCase.reject as string;
    const error = assertThrows(() => encoder.encode(testCase.text), ModelInputError);
    if (PROMPT_CLEAN_REJECTS.has(reason)) {
      assertInstanceOf(error, PromptCleanError);
      assertEquals(error.reason, reason);
    } else {
      assert(!(error instanceof PromptCleanError), `前処理で落ちた: ${error.message}`);
      assert(
        error.message.includes(TOKENIZER_REJECTS[reason]),
        `理由 ${reason} の目印が無い: ${error.message}`,
      );
    }
  });
}

// ---- 固定プロンプトと境界（フィクスチャのケースが実際に叩いているもの）--------

Deno.test("固定 4 本は長さ 28 / 118 / 50 / 126 で受理され、2 経路で同じ id 列", () => {
  const lengths = ["boxing-cats", "ferret", "cat-dog-baking", "negative"].map((name) => {
    const fixed = caseById(`fixed-${name}`);
    assertEquals(fixed.ids, fixed.tokenizerJsonIds);
    return encoder.encode(fixed.text).length;
  });
  assertEquals(lengths, [28, 118, 50, 126]);
});

Deno.test("公式 negative の全角読点 `，` は `,` になる（fix_character_width）", () => {
  const negative = caseById("fixed-negative");
  assert(negative.text.includes("，"));
  const cleaned = encoder.clean(negative.text);
  assert(!cleaned.includes("，") && cleaned.includes(","), cleaned);
});

Deno.test("上限: 512 トークンちょうどは受理・513 は拒む（黙って切り詰めない）", () => {
  const max = encoder.encode(caseById("max-512").text);
  assertEquals(max.length, 512);
  assertEquals(max.at(-1), fixture.t5.eosId, "末尾は </s>");
  assertThrows(() => encoder.encode(caseById("over-513").text), ModelInputError, "上限 512");
});

Deno.test("下限: 2 トークンは受理・空文字と空白だけ（</s> だけ）は拒む", () => {
  assertEquals(encoder.encode("cat").length, WAN_PROMPT_MIN_TOKENS);
  for (const text of ["", " \n\t　 "]) {
    assertThrows(() => encoder.encode(text), ModelInputError, `最低 ${WAN_PROMPT_MIN_TOKENS}`);
  }
});

Deno.test("拒否のメッセージに正 / ネガティブのラベルが出る", () => {
  assertThrows(() => encoder.encode("𠀋", "ネガティブプロンプト"), ModelInputError, "ネガティブ");
  assertThrows(
    () => encoder.encode("cafÃ©", "ネガティブプロンプト"),
    PromptCleanError,
    "ネガティブ",
  );
});

Deno.test("孤立サロゲートは前処理の前に入力起因で拒む", () => {
  assertThrows(() => encoder.encode("a\ud800b"), ModelInputError, "サロゲート");
});

// ---- 共通層の方針（合成の小語彙で規則そのものを見る）-------------------------

/** 語彙 `▁` / `a` / `b` だけの T5 表（unk = 2・追加語彙 `<s>` も id 2 — umT5 と同じ重なり）。 */
const tinyTables = (maxLength: number): T5Tables => ({
  vocab: new Map<string, T5VocabEntry>([
    ["▁", { id: 3, score: -1 }],
    ["a", { id: 4, score: -1 }],
    ["b", { id: 5, score: -1 }],
  ]),
  minScore: -1,
  maxTokenLength: 1,
  unkId: 2,
  eosId: 1,
  addedTokens: new Map([["</s>", 1], ["<s>", 2]]),
  space: [[0x20, 0x20]],
  maxLength,
});

Deno.test("語彙外: unk の方針は unk 1 個へ融合・拒む方針は ModelInputError", () => {
  const unk = new T5UnigramTokenizer(tinyTables(512), { unknown: "unk", overflow: "truncate" });
  assertEquals(unk.encode("axyb"), [3, 4, 2, 5, 1], "▁ a <unk> b </s>");
  const reject = new T5UnigramTokenizer(tinyTables(512), { unknown: "reject", overflow: "reject" });
  assertThrows(() => reject.encode("axyb", "検査"), ModelInputError, '"xy"');
});

Deno.test("語彙外の判定は id ではなく格子の未知ノード（unk_id と同じ id の追加語彙を通す）", () => {
  // umT5 の unk_id 2 は `<s>` の id でもある。id 列に 2 があるかで語彙外を判定すると、本文中の
  // `<s>`（追加語彙）まで拒むか、逆に未知を見逃す。
  const reject = new T5UnigramTokenizer(tinyTables(512), { unknown: "reject", overflow: "reject" });
  assertEquals(reject.encode("a<s>b"), [3, 4, 2, 3, 5, 1]);
});

Deno.test("上限: 拒む方針は `</s>` を含めてちょうど maxLength まで受け、1 つ超えたら投げる", () => {
  const reject = new T5UnigramTokenizer(tinyTables(5), { unknown: "reject", overflow: "reject" });
  // `▁a` は ▁ + a の 2 id。"a a" = 4 id + </s> = 5（ちょうど上限）。
  assertEquals(reject.encode("a a"), [3, 4, 3, 4, 1]);
  assertThrows(() => reject.encode("a a b"), ModelInputError, "上限 5");
  // 最後の断片が予算をまたぐ形（4 + 2 = 6 id）も拒む。
  assertThrows(() => reject.encode("a aa"), ModelInputError, "上限 5");
  // 追加語彙は 1 id を出すので、予算ちょうどの後ろに来れば拒む。
  assertThrows(() => reject.encode("a a</s>"), ModelInputError, "上限 5");
});

Deno.test("上限: 予算ちょうどの後ろが空白だけの chunk なら id は増えないので拒まない", () => {
  // `a a`（4 id）+ `</s>`（1 id）で予算 5 ちょうど。後ろの " " は追加語彙でない chunk だが
  // 断片を出さない — chunk の手前で拒むと、受理すべき入力を落とす。
  const reject = new T5UnigramTokenizer(tinyTables(6), { unknown: "reject", overflow: "reject" });
  assertEquals(reject.encode("a a</s> "), [3, 4, 3, 4, 1, 1]);
});

Deno.test("上限: 切り詰める方針は `</s>` の分を空けて切る（anima の上流どおり）", () => {
  const truncate = new T5UnigramTokenizer(tinyTables(5), { unknown: "unk", overflow: "truncate" });
  assertEquals(truncate.encode("a a b b"), [3, 4, 3, 4, 1]);
});

// ---- 資産の読み取り（外部境界）-----------------------------------------------

/** 最小の資産 JSON（語彙 6 行 + フィクスチャの前処理の表）。 */
const minimalAsset = (patch: Record<string, unknown> = {}): Record<string, unknown> => ({
  format: WAN_TOKENIZER_FORMAT,
  maxLength: 512,
  unkId: 2,
  eosId: 1,
  addedTokens: [["<pad>", 0], ["</s>", 1], ["<s>", 2], ["<unk>", 3]],
  space: [[0x20, 0x20]],
  vocabText: "<pad>\n</s>\n<s>\n<unk>\n▁cat\n▁",
  scores: [0, 0, 0, 0, -1, -2],
  promptClean: fixture.promptClean,
  ...patch,
});

Deno.test("parseWanTokenizerAsset: 資産 JSON から組んだ encoder が符号化できる", () => {
  const assets = parseWanTokenizerAsset(minimalAsset());
  const tiny = new WanPromptEncoder(assets);
  assertEquals(tiny.maxLength, 512);
  assertEquals([...tiny.encode("cat cat")], [4, 4, 1]);
  assertThrows(() => tiny.encode("dog"), ModelInputError, "語彙に無い文字列");
});

Deno.test("parseWanTokenizerAsset: 知らない形式の版は読まない", () => {
  assertThrows(
    () => parseWanTokenizerAsset(minimalAsset({ format: "karume-wan-umt5-tokenizer/2" })),
    Error,
    "形式",
  );
});

Deno.test("parseWanTokenizerAsset: 壊れた前処理の表は落とす（黙って別の分類にしない）", () => {
  const clean = fixture.promptClean as Record<string, unknown>;
  const broken = (patch: Record<string, unknown>): Record<string, unknown> =>
    minimalAsset({ promptClean: { ...clean, ...patch } });
  assertThrows(() => parseWanTokenizerAsset(broken({ badness: "[" })), Error, "正規表現");
  assertThrows(
    () => parseWanTokenizerAsset(broken({ assigned: [[10, 20], [15, 30]] })),
    Error,
    "重なる",
  );
  assertThrows(
    () => parseWanTokenizerAsset(broken({ width: [[0xff21, "A"], [0xff21, "B"]] })),
    Error,
    "重複",
  );
  assertThrows(
    () => parseWanTokenizerAsset(broken({ ligatures: [[0xfb01, "\ud800"]] })),
    Error,
    "サロゲート",
  );
  assertThrows(
    () => parseWanTokenizerAsset(broken({ width: [[-1, "A"]] })),
    Error,
    "コードポイント",
  );
});

Deno.test("parseWanTokenizerAsset: 語彙の値域門は T5 の共通層と同じ（重複行・非有限スコア）", () => {
  assertThrows(
    () =>
      parseWanTokenizerAsset(
        minimalAsset({
          vocabText: "<pad>\n</s>\n<s>\n<unk>\n▁cat\n▁cat",
          scores: [0, 0, 0, 0, -1, -2],
        }),
      ),
    Error,
    "重複は配布の破損",
  );
  assertThrows(
    () => parseWanTokenizerAsset(minimalAsset({ scores: [0, 0, 0, 0, -1, Number.NaN] })),
    Error,
    "有限",
  );
});

// ---- 実資産（全語彙）での突合 -------------------------------------------------

Deno.test({
  name: "実資産: 全語彙の資産から組んだ encoder もフィクスチャの全ケースと同じ決着になる",
  ignore: seriesAsset === undefined,
  fn: () => {
    // フィクスチャの語彙は部分集合（断片の部分文字列）なので、資産の読み取り（25.6 万行）と
    // 全語彙の格子でも同じ id 列 / 拒否になることをここで見る。
    const full = new WanPromptEncoder(parseWanTokenizerAsset(JSON.parse(seriesAsset as string)));
    for (const testCase of fixture.cases) {
      if (testCase.ids !== undefined) {
        assertEquals([...full.encode(testCase.text)], testCase.ids, testCase.id);
      } else {
        assertThrows(() => full.encode(testCase.text), ModelInputError, undefined, testCase.id);
      }
    }
  },
});

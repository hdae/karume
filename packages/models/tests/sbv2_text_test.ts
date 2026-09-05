// SBV2 のテキスト層（`src/sbv2/text/` の純関数）の挙動テスト。
//
// この層が壊れると **shape は合ったまま音だけが崩れる** — 音素 ID が 1 つずれても front は
// 通り、BERT 特徴が別の音素へ配られても波形は出る。数値の正は golden E2E が担保できない
// 領域なので、ここでは「不変条件が実際に破れを捕まえること」を故障注入で実証する。
//
// 資産（outputs/misc/sbv2-demo/）に依存するのは末尾の 1 本だけで、それ以外は資産なしで走る。

import { assert, assertEquals, assertFalse, assertThrows } from "@std/assert";
import { Sbv2InputError } from "../src/sbv2/errors.ts";
import { buildSbv2ModelInput } from "../src/sbv2/text/model-input.ts";
import {
  addBlankWord2ph,
  intersperse,
  type JpExtraRules,
  parseJpExtraRules,
  phonesToIds,
  phonesTonesToModelIds,
} from "../src/sbv2/text/symbols.ts";
import { DebertaTokenizer } from "../src/sbv2/text/tokenizer.ts";
import { buildBaseWord2ph, distributePhone } from "../src/sbv2/text/word2ph.ts";
import { bertHiddenOutput, tileBertToPhoneLevel } from "../src/sbv2/text/bert-tile.ts";
import { toBertText } from "../src/sbv2/text/bert-text.ts";

/**
 * テスト用の JP-Extra 規則。**本物の値ではない**（本物は `symbols.json` 資産が正 — 末尾の
 * テストがそれを読む）。ここは規則の形だけを踏むための最小構成で、tone 基点と言語 ID には
 * わざと 0 でない値を置く（0 にすると「加算していない実装」と区別できない）。
 */
const TEST_RULES_JSON = {
  symbols: ["_", "a", "k", "o", "N", ".", ","],
  pad: "_",
  punctuations: [".", ","],
  toneStart: 6,
  languageId: 1,
  numTones: 12,
  numLanguages: 3,
  addBlank: true,
  blankId: 0,
  samplingRate: 44100,
  hopLength: 512,
  bertHiddenFromEnd: 3,
  bertRelPos: { positionBuckets: 256, maxPosition: 512 },
  defaults: { sdpRatio: 0.2, noiseScale: 0.6, noiseScaleW: 0.8, lengthScale: 1 },
};

const testRules = (): JpExtraRules => parseJpExtraRules(TEST_RULES_JSON, "テスト規則");

/** 1 コードポイント = 1 トークンの最小トークナイザ（clean 表は空 = 何も落とさない）。 */
const tinyTokenizer = (tokens: readonly string[]): DebertaTokenizer =>
  DebertaTokenizer.fromVocabText(
    ["[PAD]", "[CLS]", "[SEP]", "[UNK]", ...tokens].join("\n"),
    { removed: [], spaced: [] },
    { clsId: 1, sepId: 2, unkId: 3 },
  );

Deno.test("intersperse: 要素間と両端に blank を挟んで 2n+1 にする", () => {
  assertEquals(intersperse([5, 7, 9], 0), [0, 5, 0, 7, 0, 9, 0]);
  // 空列でも「両端」ぶんの 1 要素は残る（参照実装 commons.intersperse と同じ）。
  assertEquals(intersperse([], 0), [0]);
  // blank が 0 以外でも同じ位置に入る（0 決め打ちの実装を弾く）。
  assertEquals(intersperse([5], 4), [4, 5, 4]);
});

Deno.test("addBlankWord2ph: 全要素を 2 倍し先頭にだけ 1 を足す", () => {
  const base = [1, 2, 3, 1];
  assertEquals(addBlankWord2ph(base), [3, 4, 6, 2]);
  // 不変条件: sum(word2ph) が add_blank 後の音素長 2n+1 に一致する。
  const phoneCount = base.reduce((sum, count) => sum + count, 0);
  const total = addBlankWord2ph(base).reduce((sum, count) => sum + count, 0);
  assertEquals(total, 2 * phoneCount + 1);
});

Deno.test("addBlankWord2ph: 故障注入 — 先頭の +1 を落とすと音素長と合わなくなる", () => {
  // 上のテストが恒真でないことの実証。`[0] += 1` を落とした実装は sum が 1 少なくなり、
  // 下流（tileBertToPhoneLevel）で「音素数と違う」と落ちる。
  const base = [1, 2, 3, 1];
  const broken = base.map((count) => count * 2); // +1 を忘れた実装
  const phoneCount = base.reduce((sum, count) => sum + count, 0);
  assert(broken.reduce((sum, count) => sum + count, 0) !== 2 * phoneCount + 1);
});

Deno.test("addBlankWord2ph: 空の base は落とす（両端の番兵が欠落）", () => {
  assertThrows(() => addBlankWord2ph([]), Error, "空");
});

Deno.test("phonesToIds: 記号表の添字を返し、未知の音素は落とす", () => {
  const rules = testRules();
  assertEquals(phonesToIds(rules, ["_", "k", "o", "."]), [0, 2, 3, 5]);
  // 呼び手の発話だけで到達する失敗なので入力起因（400 相当）— 素の Error では 500 に化ける。
  assertThrows(() => phonesToIds(rules, ["ky"]), Sbv2InputError, "記号表に無い音素");
});

Deno.test("phonesTonesToModelIds: tone 基点の加算・言語 ID の配布・add_blank が同時に効く", () => {
  const rules = testRules();
  const ids = phonesTonesToModelIds(rules, ["_", "k", "a", "_"], [0, 1, 1, 0]);
  assertEquals(ids.phoneIds, [0, 0, 0, 2, 0, 1, 0, 0, 0]);
  // tone は +6（blank 位置は blankId のまま — 加算しない）。
  assertEquals(ids.toneIds, [0, 6, 0, 7, 0, 7, 0, 6, 0]);
  // MUST: JP-Extra でも language は全 0 ではない（実音素位置は languageId）。
  assertEquals(ids.languageIds, [0, 1, 0, 1, 0, 1, 0, 1, 0]);
});

Deno.test("phonesTonesToModelIds: 範囲外の tone / 長さ不一致は落とす", () => {
  const rules = testRules();
  assertThrows(() => phonesTonesToModelIds(rules, ["_"], [0, 0]), Error, "長さが不一致");
  // 12 トーンのモデルに tone=7 を渡すと 6+7=13 で範囲外。黙って別の行を引かせない。
  assertThrows(() => phonesTonesToModelIds(rules, ["_"], [7]), Error, "範囲外");
});

Deno.test("parseJpExtraRules: 壊れた資産を黙って使わない", () => {
  assertThrows(() => parseJpExtraRules({ ...TEST_RULES_JSON, symbols: [] }, "x"), Error, "空");
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, symbols: ["_", "_"] }, "x"),
    Error,
    "重複",
  );
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, addBlank: false }, "x"),
    Error,
    "add_blank",
  );
  assertThrows(() => parseJpExtraRules({ ...TEST_RULES_JSON, pad: "?" }, "x"), Error, "pad");
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, toneStart: 1.5 }, "x"),
    Error,
    "整数でない",
  );
});

Deno.test("parseJpExtraRules: 実行定数の値域を構築時に見る（整数だけでは通さない）", () => {
  // 決め手は samplingRate — `Sbv2Pipeline.generate` の `sampleRate` として**そのまま公開結果
  // へ出る**唯一の欄なので、負値や 0 はどこでも例外にならず「正常に見える壊れた WAV」に
  // なる。残りは遅れて fail loudly するだけだが、門をここへ揃える。
  for (
    const key of ["samplingRate", "hopLength", "bertHiddenFromEnd", "numTones", "numLanguages"]
  ) {
    for (const bad of [0, -1, 1.5]) {
      assertThrows(
        () => parseJpExtraRules({ ...TEST_RULES_JSON, [key]: bad }, "x"),
        Error,
        `x.${key}: 正の安全整数でない（${bad}）`,
      );
    }
  }
  // languageId は language 埋め込みの行を指すので、numLanguages 未満まで見る（0 は正当）。
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, languageId: TEST_RULES_JSON.numLanguages }, "x"),
    Error,
    "x.languageId: language 埋め込みの範囲外",
  );
  assertThrows(
    () => parseJpExtraRules({ ...TEST_RULES_JSON, languageId: -1 }, "x"),
    Error,
    "x.languageId: language 埋め込みの範囲外",
  );
  assertEquals(parseJpExtraRules({ ...TEST_RULES_JSON, languageId: 0 }, "x").languageId, 0);
});

Deno.test("distributePhone: 音素を文字へ均等分配する（余りは左から）", () => {
  assertEquals(distributePhone(5, 2), [3, 2]);
  assertEquals(distributePhone(1, 3), [1, 0, 0]);
  assertEquals(distributePhone(0, 2), [0, 0]);
  assertEquals(distributePhone(6, 3), [2, 2, 2]);
  assertThrows(() => distributePhone(3, 0), Error, "1 以上の整数でない");
});

Deno.test("buildBaseWord2ph: 語ごとの分配を並べ、両端に番兵 1 を置く", () => {
  const tokenizer = tinyTokenizer(["こ", "ん", "に", "ち", "は", "、"]);
  const words = [
    { surface: "こんにちは", phones: ["k", "o", "N", "n", "i", "ch", "i", "w", "a"] },
    { surface: "、", phones: [","] },
  ];
  // 音素は 9 + 1 = 10、両端 PAD 込みで given_phone 長は 12。
  const word2ph = buildBaseWord2ph(words, tokenizer, 12);
  assertEquals(word2ph, [1, 2, 2, 2, 2, 1, 1, 1]);
  assertEquals(word2ph.length, tokenizer.encode("こんにちは、").length);
});

Deno.test("buildBaseWord2ph: 故障注入 — given_phone 長が 1 ずれると落ちる", () => {
  // この不変条件が無いと、音素列と word2ph が別々にずれたまま front まで到達し、
  // BERT 特徴が 1 音素ぶん回転して配られる（音は出るが崩れる）。
  const tokenizer = tinyTokenizer(["こ", "ん"]);
  const words = [{ surface: "こん", phones: ["k", "o", "N"] }];
  assertEquals(buildBaseWord2ph(words, tokenizer, 5), [1, 2, 1, 1]);
  assertThrows(() => buildBaseWord2ph(words, tokenizer, 6), Error, "sum(word2ph)");
});

Deno.test("buildBaseWord2ph: 0 トークンへ正規化される surface は入力起因で落とす", () => {
  // 空白だけの surface は呼び手が words を直せば直る = Sbv2InputError（400 相当）。
  const tokenizer = tinyTokenizer(["あ"]);
  assertThrows(
    () => buildBaseWord2ph([{ surface: " ", phones: ["a"] }], tokenizer, 3),
    Sbv2InputError,
    "0 トークンに正規化された",
  );
  // 対: sum(word2ph) の不一致は内部不変条件の破れなので素の Error のまま残す
  // （errors.ts が名指しで 500 側に置いている経路 — 両方 400 にすると分類が消える）。
  const sumError = assertThrows(
    () => buildBaseWord2ph([{ surface: "あ", phones: ["a"] }], tokenizer, 9),
    Error,
    "sum(word2ph)",
  );
  assertFalse(sumError instanceof Sbv2InputError);
});

Deno.test("buildSbv2ModelInput: input_ids 長と word2ph 長の食い違いは入力起因で落とす", () => {
  // 記号語の surface を「正規形と別のトークン数」に書き換えた発話でだけ到達する経路。
  // 呼び手が words を解析どおりに戻せば直るので 400 側。
  const rules = testRules();
  const tokenizer = tinyTokenizer([".", "あ"]);
  const utterance = {
    leadingPunctuations: ["."],
    moras: [],
    words: [{ surface: "あ あ", phones: ["."] }],
  };
  // surface は 2 トークン・正規形 "." は 1 トークンなので word2ph 長 4 と input_ids 長 3 が割れる。
  assertThrows(
    () => buildSbv2ModelInput(utterance, tokenizer, rules),
    Sbv2InputError,
    "input_ids 長",
  );
});

Deno.test("DebertaTokenizer: NFKC で 1 文字が複数トークンへ割れる（… → ...）", () => {
  const tokenizer = tinyTokenizer([".", "あ"]);
  assertEquals(tokenizer.tokenize("…"), [".", ".", "."]);
  assertEquals(tokenizer.tokenize("あ…"), ["あ", ".", ".", "."]);
  assertEquals(tokenizer.encode("あ"), [1, 5, 2]);
  // vocab に無い文字は [UNK]（握り潰しではなく既知の縮退）。
  assertEquals(tokenizer.encode("ん"), [1, 3, 2]);
});

Deno.test("DebertaTokenizer: clean 表の除去・スペース化が効く", () => {
  // U+0000 を除去、U+200B（ZWSP）を空白へ。**スペース化に NBSP を使わない**のは、後段の
  // 分割 `/\s+/` が NBSP を最初から空白として拾うため — spaced 規則を丸ごと落としても
  // 同じ答えが出てしまい、門が恒真になる（ZWSP は `/\s/` が拾わない）。
  const vocabText = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "a", "b"].join("\n");
  const special = { clsId: 1, sepId: 2, unkId: 3 };
  const tokenizer = DebertaTokenizer.fromVocabText(
    vocabText,
    { removed: [[0, 0]], spaced: [[0x200b, 0x200b]] },
    special,
  );
  assertEquals(tokenizer.tokenize("a\u0000b"), ["a", "b"]);
  assertEquals(tokenizer.tokenize("a\u200bb"), ["a", "b"]);
  // 対: clean 表が空なら同じ入力が割れない（上の 2 本が表そのものを踏んでいることの実証）。
  const bare = DebertaTokenizer.fromVocabText(vocabText, { removed: [], spaced: [] }, special);
  assertEquals(bare.tokenize("a\u0000b"), ["a", "\u0000", "b"]);
  assertEquals(bare.tokenize("a\u200bb"), ["a", "\u200b", "b"]);
});

Deno.test("DebertaTokenizer: CRLF 混じりの語彙でも末尾 CR が残らない", () => {
  // \r が残ると全 lookup が [UNK] に落ちる（エラーは出ず BERT 特徴だけ静かに壊れる）。
  const tokenizer = DebertaTokenizer.fromVocabText(
    "[PAD]\r\n[CLS]\r\n[SEP]\r\n[UNK]\r\nあ\r\n",
    { removed: [], spaced: [] },
    { clsId: 1, sepId: 2, unkId: 3 },
  );
  assertEquals(tokenizer.encode("あ"), [1, 4, 2]);
});

Deno.test("DebertaTokenizer: 語彙の重複行は落とす（後勝ちで別の行を引かせない）", () => {
  // 行番号 = ID なので、重複行は先の ID を引けなくする。例外は出ず BERT 特徴だけが
  // 「合法な別の行」に変わるため、id 列の長さ検査でも word2ph の突合でも捕まらない。
  assertThrows(
    () =>
      DebertaTokenizer.fromVocabText(
        "[PAD]\n[CLS]\n[SEP]\n[UNK]\nあ\nあ",
        { removed: [], spaced: [] },
        { clsId: 1, sepId: 2, unkId: 3 },
      ),
    Error,
    "vocabText: 行 5 が行 4 と同じトークン",
  );
});

Deno.test("toBertText: 記号語だけ正規形へ写して連結する", () => {
  const punctuations = new Set([",", "."]);
  const words = [
    { surface: "これ", phones: ["k", "o", "r", "e"] },
    { surface: "。", phones: ["."] },
  ];
  assertEquals(toBertText(words, punctuations), "これ.");
  // 記号でない語（phones が 1 個でも正規形でない）は surface のまま。
  assertEquals(toBertText([{ surface: "ん", phones: ["N"] }], punctuations), "ん");
});

Deno.test("bertHiddenOutput: 末尾からの相対で選ぶ（層数が違っても同じ規則）", () => {
  const full = Array.from({ length: 25 }, (_, index) => `h${index}`);
  assertEquals(bertHiddenOutput(full, 3), "h22");
  assertEquals(bertHiddenOutput(["h0", "h1", "h2"], 3), "h0");
  assertThrows(() => bertHiddenOutput(["h0", "h1"], 3), Error, "3 本以上");
  assertThrows(() => bertHiddenOutput(full, 0), Error, "1 以上");
});

Deno.test("tileBertToPhoneLevel: トークンを word2ph 回だけ複製し、転置して [dim, P] にする", () => {
  // 2 トークン × dim 3。値はトークンと次元の両方で違う（転置の取り違えを踏むため
  // dim ≠ P かつ非対称にする — 対称な入力では軸を入れ替えても同じ答えが出てしまう）。
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  const { data, dim, columns } = tileBertToPhoneLevel(hidden, 2, [1, 2]);
  assertEquals(columns, 3);
  assertEquals(dim, 3);
  // 期待: 列は [t0, t1, t1]、行は dim。row-major で [dim=3][P=3]。
  assertEquals([...data], [1, 10, 10, 2, 20, 20, 3, 30, 30]);
});

Deno.test("tileBertToPhoneLevel: 故障注入 — word2ph の配分が変わると値の並びも変わる", () => {
  // 「合計さえ合えば通る」実装（配分を無視して均等に割る等）を弾くための対。
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  const a = tileBertToPhoneLevel(hidden, 2, [1, 2]);
  const b = tileBertToPhoneLevel(hidden, 2, [2, 1]);
  assertEquals(a.columns, b.columns, "合計は同じ（= 長さでは検出できない形）");
  assert([...a.data].some((value, index) => value !== b.data[index]), "配分を変えても同じ値");
});

Deno.test("tileBertToPhoneLevel: トークン数と word2ph 長の不一致は落とす", () => {
  const hidden = new Float32Array([1, 2, 3, 10, 20, 30]);
  assertThrows(() => tileBertToPhoneLevel(hidden, 2, [1, 2, 1]), Error, "word2ph 長");
  assertThrows(() => tileBertToPhoneLevel(hidden, 4, [1, 1, 1, 1]), Error, "割り切れない");
});

// ---- 実資産（あれば）--------------------------------------------------------

const SYMBOLS_PATH = new URL("../../../outputs/misc/sbv2-demo/symbols.json", import.meta.url);
const assetAvailable = (() => {
  try {
    return Deno.statSync(SYMBOLS_PATH).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
})();

if (!assetAvailable) {
  console.warn(
    `[karume] ${SYMBOLS_PATH.pathname} が無いためデモ資産の検査を SKIP する。` +
      "生成: cd tools/exporter && uv run --group sbv2 python sbv2_demo.py assets",
  );
}

Deno.test({
  name: "デモ資産: symbols.json が JP-Extra の規則として読める",
  ignore: !assetAvailable,
  fn: async () => {
    const rules = parseJpExtraRules(
      JSON.parse(await Deno.readTextFile(SYMBOLS_PATH)),
      "symbols.json",
    );
    // 値そのものは資産が正（TS 側に期待値表を作らない — それをやると資産の意味が消える）。
    // ここで固定するのは「規則として成立していること」だけ。
    assertEquals(rules.symbols[0], rules.pad, "記号表の先頭が PAD");
    assert(rules.toneStart + 1 < rules.numTones, "tone 基点 + 最大 tone(1) が範囲内");
    assert(rules.languageId < rules.numLanguages, "言語 ID が範囲内");
    for (const punctuation of rules.punctuations) {
      assert(rules.symbolToId.has(punctuation), `句読点 ${punctuation} が記号表に無い`);
    }
    assert(rules.hopLength > 0 && rules.samplingRate > 0);
  },
});

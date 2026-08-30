// SBV2 の発話入力（`toSbv2Utterance` の変換と、合成入口の門）の挙動テスト。
//
// この層が壊れると **shape は合ったまま音だけが変わる** — 核 → トーンの展開が 1 段ずれても
// front は通り、モーラを差し替えた発話も音素数さえ合えば走る。どちらも例外は出ない。
// ここでは「変換が上流の規則どおりであること」と「壊れた発話が門で止まること」を分けて押さえる。
//
// 変換と門は karume の純関数なので辞書なしで走る。呼び手側の解析（`@hdae/yomi`）を通した
// 経路は、実辞書があるときだけ走らせる（0.6.0 以降、解析はパッケージの外側にある）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { analyzeWithWords, JtdDictionary, OverlayDictionary, type OverlayEntry } from "@hdae/yomi";
import { Sbv2InputError } from "../src/sbv2/errors.ts";
import { assertWordPhones, buildSbv2ModelInput } from "../src/sbv2/text/model-input.ts";
import { toSbv2PhoneTone } from "../src/sbv2/text/phone-tone.ts";
import {
  type Sbv2Mora,
  type Sbv2Phrases,
  type Sbv2Utterance,
  type Sbv2Word,
  toSbv2Utterance,
} from "../src/sbv2/text/utterance.ts";
import { type JpExtraRules, parseJpExtraRules } from "../src/sbv2/text/symbols.ts";
import { DebertaTokenizer } from "../src/sbv2/text/tokenizer.ts";

/**
 * テスト用の JP-Extra 規則。**本物の値ではない**（本物は `symbols.json` 資産が正）。
 * tone 基点は 0 でない値を置く（0 だと「加算していない実装」と区別できない）。
 */
const TEST_RULES_JSON = {
  symbols: ["_", "a", "e", "k", "m", "o", "r", "t", "u", ".", ","],
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

const TEXT = "香留芽";
const textTokenizer = (): DebertaTokenizer => tinyTokenizer(["香", "留", "芽", "。"]);

// ---- 核 → トーンの展開（辞書なし）------------------------------------------

/** 「カルメ」1 句ぶんの解析結果（呼び手側の解析器が返す形を手で組んだもの）。 */
const karumePhrases = (accentNucleus: number, punctuations: string[] = []): Sbv2Phrases => ({
  result: {
    leadingPunctuations: [],
    accentPhrases: [{
      moras: [
        { kana: "カ", consonant: "k", vowel: "a" },
        { kana: "ル", consonant: "r", vowel: "u" },
        { kana: "メ", consonant: "m", vowel: "e" },
      ],
      accentNucleus,
      punctuations,
    }],
  },
  // 語アライメントは記号を 1 要素ずつ持つ（解析器の語ビューと同じ形）。
  words: [
    { surface: TEXT, phones: ["k", "a", "r", "u", "m", "e"] },
    ...punctuations.map((punct) => ({ surface: punct, phones: [punct] })),
  ],
});

const tonesOf = (utterance: Sbv2Utterance): number[] => utterance.moras.map((mora) => mora.tone);

Deno.test("toSbv2Utterance: 核をモーラごとの 2 値トーンへ展開する（平板 / 頭高 / 中高 / 尾高）", () => {
  // 上流 SBV2 の g2p と同じ規則。1 段でもずれると「別のアクセントで合成される」沈黙誤値。
  assertEquals(tonesOf(toSbv2Utterance(karumePhrases(0))), [0, 1, 1], "平板");
  assertEquals(tonesOf(toSbv2Utterance(karumePhrases(1))), [1, 0, 0], "頭高");
  assertEquals(tonesOf(toSbv2Utterance(karumePhrases(2))), [0, 1, 0], "中高");
  assertEquals(tonesOf(toSbv2Utterance(karumePhrases(3))), [0, 1, 1], "尾高");
});

Deno.test("toSbv2Utterance: 核の上端はモーラ数へ丸める（往復不能を作らない）", () => {
  // 範囲外核は辞書差・修正辞書由来で普通に出る。落とすと「解析どおりの発話が通らない」に
  // なるので、上端だけは丸める（丸めてもトーン列は変わらない = 音は同じ）。
  assertEquals(
    tonesOf(toSbv2Utterance(karumePhrases(9))),
    tonesOf(toSbv2Utterance(karumePhrases(3))),
  );
});

Deno.test("toSbv2Utterance: 負の核・非整数の核は落とす（下端は丸めない）", () => {
  // 黙って平板へ倒すと、指定したものと別のアクセントで合成される。
  assertThrows(() => toSbv2Utterance(karumePhrases(-1)), Sbv2InputError, "accentNucleus = -1");
  assertThrows(() => toSbv2Utterance(karumePhrases(1.5)), Sbv2InputError, "accentNucleus = 1.5");
});

Deno.test("toSbv2Utterance: 句の記号は句末尾モーラへ、句より前の記号は先頭欄へ", () => {
  const withPunct = toSbv2Utterance(karumePhrases(1, ["."]));
  assertEquals(withPunct.moras[2].punctuations, ["."], "句末尾モーラに付いていない");
  assertEquals(withPunct.moras[0].punctuations, undefined, "無関係なモーラに欄が生えた");

  // 文頭の記号はフラットな音素列からは復元できない情報（構造で運ぶ理由そのもの）。
  const leading = karumePhrases(1);
  const withLeading = toSbv2Utterance({
    result: { ...leading.result, leadingPunctuations: ["…"] },
    words: [{ surface: "…", phones: ["…"] }, ...leading.words],
  });
  assertEquals(withLeading.leadingPunctuations, ["…"]);
});

// ---- 複数句（句境界・辞書なし）----------------------------------------------
//
// 1 句フィクスチャでは「発話の末尾モーラ」と「句の末尾モーラ」が同一なので、記号の付け替えを
// 見分ける検出力が構造的にゼロになる。句を跨ぐ添字ずれ・記号の流出・モーラを持たない句は
// ここでしか捕まらない（辞書経路のテストは資産のある環境でしか走らない）。

/** 「カルメ」+「トク」の 2 句（`words` は句の音素列どおりに手で組む — 記号は 1 要素 1 語）。 */
const twoPhrases = (
  nuclei: readonly [number, number],
  punctuations: readonly [string[], string[]] = [[], []],
): Sbv2Phrases => ({
  result: {
    leadingPunctuations: [],
    accentPhrases: [
      {
        moras: [
          { kana: "カ", consonant: "k", vowel: "a" },
          { kana: "ル", consonant: "r", vowel: "u" },
          { kana: "メ", consonant: "m", vowel: "e" },
        ],
        accentNucleus: nuclei[0],
        punctuations: punctuations[0],
      },
      {
        moras: [
          { kana: "ト", consonant: "t", vowel: "o" },
          { kana: "ク", consonant: "k", vowel: "u" },
        ],
        accentNucleus: nuclei[1],
        punctuations: punctuations[1],
      },
    ],
  },
  words: [
    { surface: TEXT, phones: ["k", "a", "r", "u", "m", "e"] },
    ...punctuations[0].map((punct) => ({ surface: punct, phones: [punct] })),
    { surface: "特", phones: ["t", "o", "k", "u"] },
    ...punctuations[1].map((punct) => ({ surface: punct, phones: [punct] })),
  ],
});

Deno.test("toSbv2Utterance: 核は句ごとに独立して展開される（句 2 が句 1 の核に汚されない）", () => {
  // `tones[at]` の `at` を発話全体の添字にしてしまう退行は、1 句のテストでは一度も踏めない。
  assertEquals(tonesOf(toSbv2Utterance(twoPhrases([1, 0]))), [1, 0, 0, 0, 1]);
  assertEquals(tonesOf(toSbv2Utterance(twoPhrases([0, 0]))), [0, 1, 1, 0, 1]);
  // 展開は「句 1 の展開 ++ 句 2 の展開」に厳密一致する（1 句版の答えを連結した形）。
  assertEquals(
    tonesOf(toSbv2Utterance(twoPhrases([1, 2]))),
    [...tonesOf(toSbv2Utterance(karumePhrases(1))), 0, 1],
  );
  // 句を跨いでも音素列と語アライメントは一致する。
  const utterance = toSbv2Utterance(twoPhrases([1, 0]));
  assertWordPhones(toSbv2PhoneTone(utterance, "_").phones, utterance.words);
});

Deno.test("toSbv2Utterance: 記号は**その句の**末尾モーラへ付く（中間句の記号が末尾句へ流れない）", () => {
  const utterance = toSbv2Utterance(twoPhrases([1, 0], [[","], ["."]]));
  assertEquals(utterance.moras.length, 5, "句を跨いでモーラが落ちている");
  assertEquals(utterance.moras[2].punctuations, [","], "中間句の記号が句 1 の末尾モーラに無い");
  assertEquals(utterance.moras[4].punctuations, ["."], "末尾句の記号が句 2 の末尾モーラに無い");
  assertEquals(utterance.moras[3].punctuations, undefined, "句 2 の先頭モーラに欄が生えた");
  assertEquals(utterance.leadingPunctuations, []);
  // 位置は音素列にも出る（記号が末尾句へ流れれば並びが変わる）。
  assertEquals(
    toSbv2PhoneTone(utterance, "_").phones,
    ["_", "k", "a", "r", "u", "m", "e", ",", "t", "o", "k", "u", ".", "_"],
  );
  assertWordPhones(toSbv2PhoneTone(utterance, "_").phones, utterance.words);
});

Deno.test("toSbv2Utterance: モーラを持たない句の記号は直前モーラへ寄る（先頭なら先頭欄へ）", () => {
  // `utterance.ts` の doc が明示している 2 分岐。1 句フィクスチャは必ず 3 モーラを持つので、
  // どちらの分岐にも到達しない。
  const emptyPhrase = { moras: [], accentNucleus: 0, punctuations: ["…"] };
  const karume = karumePhrases(0);

  const trailing = toSbv2Utterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [...karume.result.accentPhrases, emptyPhrase],
    },
    words: [...karume.words, { surface: "…", phones: ["…"] }],
  });
  assertEquals(trailing.moras.length, 3, "モーラの無い句がモーラを生やした");
  assertEquals(trailing.moras[2].punctuations, ["…"], "直前モーラへ寄っていない");
  assertEquals(trailing.leadingPunctuations, []);
  assertWordPhones(toSbv2PhoneTone(trailing, "_").phones, trailing.words);

  // 先頭に置くと寄せ先のモーラが無い → `leadingPunctuations`（構造でしか運べない情報）。
  const leading = toSbv2Utterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [emptyPhrase, ...karume.result.accentPhrases],
    },
    words: [{ surface: "…", phones: ["…"] }, ...karume.words],
  });
  assertEquals(leading.leadingPunctuations, ["…"]);
  for (const [index, mora] of leading.moras.entries()) {
    assertEquals(mora.punctuations, undefined, `moras[${index}] に欄が生えた`);
  }
  assertWordPhones(toSbv2PhoneTone(leading, "_").phones, leading.words);
});

Deno.test("toSbv2Utterance: 促音は音素へ畳む（`Sbv2Mora.vowel` は音素そのもの）", () => {
  // 上流の解析は促音を "cl" で表すが、語アライメントの音素は "q"。畳まずに運ぶと、
  // 音素列と words の突合がその 1 個だけで割れる（= 促音を含む入力が全部落ちる）。
  const utterance = toSbv2Utterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [{
        moras: [
          { kana: "カ", consonant: "k", vowel: "a" },
          { kana: "ッ", vowel: "cl" },
          { kana: "ト", consonant: "t", vowel: "o" },
        ],
        accentNucleus: 1,
        punctuations: [],
      }],
    },
    words: [{ surface: "カット", phones: ["k", "a", "q", "t", "o"] }],
  });
  assertEquals(utterance.moras[1].vowel, "q");
  assertEquals(toSbv2PhoneTone(utterance, "_").phones, ["_", "k", "a", "q", "t", "o", "_"]);
  // 突合が通る（畳んでいない実装はここで落ちる）。
  assertWordPhones(toSbv2PhoneTone(utterance, "_").phones, utterance.words);
});

Deno.test("toSbv2Utterance: 子音なしモーラは consonant 欄そのものを出さない（撥音・促音・母音単独）", () => {
  // `Sbv2Mora.consonant` は optional property であって `string | undefined` ではない。
  // 欄が生えると「渡せるのに効かない欄を作らない」責務が破れる（撥音 N は通常経路）。
  const utterance = toSbv2Utterance({
    result: {
      leadingPunctuations: [],
      accentPhrases: [{
        moras: [
          { kana: "カ", consonant: "k", vowel: "a" },
          { kana: "ン", vowel: "N" },
        ],
        accentNucleus: 1,
        punctuations: [],
      }],
    },
    words: [{ surface: "缶", phones: ["k", "a", "N"] }],
  });
  assertEquals(Object.keys(utterance.moras[0]).sort(), ["consonant", "kana", "tone", "vowel"]);
  assertEquals(Object.keys(utterance.moras[1]).sort(), ["kana", "tone", "vowel"]);
  // 子音なしモーラは音素 1 個ぶんしか出さない（`_` + k a N + `_`）。
  assertEquals(toSbv2PhoneTone(utterance, "_").phones, ["_", "k", "a", "N", "_"]);
});

// ---- 音素・トーン列の梱包（辞書なし）----------------------------------------

Deno.test("toSbv2PhoneTone: 両端 PAD・記号 tone 0・モーラ内は同一トーン", () => {
  const utterance = toSbv2Utterance(karumePhrases(1, ["."]));
  const { phones, tones } = toSbv2PhoneTone(utterance, "_");
  assertEquals(phones, ["_", "k", "a", "r", "u", "m", "e", ".", "_"]);
  // 頭高: 第 1 モーラの子音・母音が 1、以降 0。記号と PAD は 0。
  assertEquals(tones, [0, 1, 1, 0, 0, 0, 0, 0, 0]);
});

Deno.test("toSbv2PhoneTone: 0/1 以外の tone は落とす（JS からの呼び出しを型で守れない）", () => {
  // 2 も -1 も toneStart を足せば記号表の範囲に収まる（`phonesTonesToModelIds` は素通り）。
  const utterance = toSbv2Utterance(karumePhrases(1));
  const broken = (tone: number): Sbv2Utterance => ({
    ...utterance,
    moras: utterance.moras.map((mora, index) => index === 1 ? { ...mora, tone } as Sbv2Mora : mora),
  });
  assertThrows(() => toSbv2PhoneTone(broken(2), "_"), Sbv2InputError, "moras[1]（ル）の tone = 2");
  assertThrows(() => toSbv2PhoneTone(broken(-1), "_"), Sbv2InputError, "tone = -1");
  assertThrows(() => toSbv2PhoneTone(broken(0.5), "_"), Sbv2InputError, "tone = 0.5");
});

// ---- 合成入口の門（moras と words の突合）-----------------------------------

Deno.test("assertWordPhones: 長さが合っていても内容が違えば落とす", () => {
  const words: Sbv2Word[] = [{ surface: TEXT, phones: ["k", "a", "r", "u", "m", "e"] }];
  const phones = ["_", "k", "a", "r", "u", "m", "e", "_"];
  assertWordPhones(phones, words);
  assertThrows(
    () => assertWordPhones(["_", "k", "a", "r", "u", "m", "o", "_"], words),
    Sbv2InputError,
    "音素[5]",
  );
  assertThrows(
    () => assertWordPhones(phones.slice(0, 7), words),
    Sbv2InputError,
    "音素数 5",
  );
});

Deno.test("tone の直編集は合成入力の tone ID まで届く（音素側は動かない）", () => {
  const rules = testRules();
  const tokenizer = textTokenizer();
  const base = toSbv2Utterance(karumePhrases(1));
  const baseInput = buildSbv2ModelInput(base, tokenizer, rules);

  // 反転させる（解析が何を返しても全位置が変わる = 「無視している実装」を必ず捕まえる）。
  const flipped: Sbv2Utterance = {
    ...base,
    moras: base.moras.map((mora) => ({ ...mora, tone: (1 - mora.tone) as 0 | 1 })),
  };
  const edited = buildSbv2ModelInput(flipped, tokenizer, rules);

  assertEquals(edited.tones, [0, 0, 0, 1, 1, 1, 1, 0], "モーラの tone が音素へ配られていない");
  assertEquals(edited.phones, baseInput.phones, "音素列が tone 編集で動いた");
  assertEquals(edited.ids.phoneIds, baseInput.ids.phoneIds, "音素 ID が動いた");
  assertEquals(edited.bertText, baseInput.bertText, "BERT 入力はトーンを読まない");
  assertEquals(edited.baseWord2ph, baseInput.baseWord2ph, "word2ph はトーンを読まない");
  assert(
    edited.ids.toneIds.some((id, index) => id !== baseInput.ids.toneIds[index]),
    "トーン ID が編集を反映していない",
  );
  // add_blank 後の実音素位置は奇数添字（`intersperse` の規約）。
  for (const [index, tone] of edited.tones.entries()) {
    assertEquals(edited.ids.toneIds[2 * index + 1], TEST_RULES_JSON.toneStart + tone);
    assertEquals(edited.ids.toneIds[2 * index], TEST_RULES_JSON.blankId, "blank 位置が汚れた");
  }
});

Deno.test("音素を変える編集・記号の増減・words の取り違えは入口で落ちる", () => {
  const rules = testRules();
  const tokenizer = textTokenizer();
  const base = toSbv2Utterance(karumePhrases(1));
  const withMoras = (moras: readonly Sbv2Mora[]): Sbv2Utterance => ({ ...base, moras });

  // ① モーラの読み替え（音素数は変わらない = 長さ検査だけなら素通りする）。
  assertThrows(
    () =>
      buildSbv2ModelInput(
        withMoras([
          base.moras[0],
          { kana: "ロ", consonant: "r", vowel: "o", tone: base.moras[1].tone },
          base.moras[2],
        ]),
        tokenizer,
        rules,
      ),
    Sbv2InputError,
    "音素[3]",
  );

  // ② モーラを 1 つ落とす（音素数が変わる）。
  assertThrows(
    () => buildSbv2ModelInput(withMoras(base.moras.slice(0, 2)), tokenizer, rules),
    Sbv2InputError,
    "音素数 4",
  );

  // ③ 記号を足す（疑問形の上げを差し込もうとした場合）。
  assertThrows(
    () =>
      buildSbv2ModelInput(
        withMoras([base.moras[0], base.moras[1], { ...base.moras[2], punctuations: ["."] }]),
        tokenizer,
        rules,
      ),
    Sbv2InputError,
    "音素数 7",
  );

  // ④ 別の解析から採った words を混ぜた（= moras と words が同一解析でない）。
  assertThrows(
    () =>
      buildSbv2ModelInput(
        { ...base, words: toSbv2Utterance(karumePhrases(1, ["."])).words },
        tokenizer,
        rules,
      ),
    Sbv2InputError,
    "音素数",
  );
});

// ---- 呼び手側の解析経路（実辞書があれば）------------------------------------
//
// 0.6.0 で解析はパッケージの外へ出た。ここで押さえるのは「解析器の返り値がそのまま
// `toSbv2Utterance` に渡せること」と「呼び手側の修正辞書が最終 ID まで届くこと」。

const DICT_DIR = new URL("../../../outputs/misc/yomi/", import.meta.url);

/** `outputs/misc/yomi/` に置かれた JTD1 辞書（版が上がると綴りが変わるので拡張子で拾う）。 */
const dictUrl = ((): URL | undefined => {
  let found: string | undefined;
  try {
    for (const entry of Deno.readDirSync(DICT_DIR)) {
      if (!entry.isFile || !entry.name.endsWith(".jtd")) continue;
      if (found === undefined || entry.name < found) found = entry.name;
    }
  } catch (cause) {
    if (!(cause instanceof Deno.errors.NotFound)) throw cause;
  }
  return found === undefined ? undefined : new URL(found, DICT_DIR);
})();

if (dictUrl === undefined) {
  console.warn(
    `[karume] ${DICT_DIR.pathname} に *.jtd が無いため解析経路の検査を SKIP する。` +
      "取得: HF dataset hdae/yomi-dict の naist-jdic.jtd（gzip を解いて置く）",
  );
}

/** 19MB を読むのは資産があるときだけ（この節の全テストが同じ 1 本を使い回す）。 */
const dictionary = dictUrl === undefined
  ? undefined
  : JtdDictionary.load(Deno.readFileSync(dictUrl).buffer);

/** ignore の外で undefined を潰す（`ignore: !available` が守るので実際には来ない）。 */
const jtd = (): JtdDictionary => {
  if (dictionary === undefined) throw new Error("辞書が無いのに辞書テストが走った");
  return dictionary;
};

const entry = (reading: string, accentType: number): OverlayEntry => ({
  surface: TEXT,
  reading,
  accentType,
});

Deno.test({
  name: "解析器の返り値はそのまま発話になる（修正辞書は呼び手側で効かせる）",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const base = buildSbv2ModelInput(
      toSbv2Utterance(analyzeWithWords(dict, TEXT)),
      tokenizer,
      rules,
    );

    // 読み: 本辞書の読みを固定値で書かない（辞書の版が上がると変わる — 検証したいのは
    // 「呼び手側の overlay が指定どおりの音素列にする」ことで、本辞書がどう読むかではない）。
    const head = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const fixed = buildSbv2ModelInput(
      toSbv2Utterance(analyzeWithWords(dict, TEXT, head)),
      tokenizer,
      rules,
    );
    assertEquals(fixed.phones, ["_", "k", "a", "r", "u", "m", "e", "_"]);
    assert(base.phones.join(" ") !== fixed.phones.join(" "), "overlay 無しと同じ読みになっている");

    // アクセント: 同じ読みで型だけ変えると、音素列は同一のままトーンだけが動く。
    const flat = new OverlayDictionary(dict, [entry("カルメ", 0)]);
    const heiban = buildSbv2ModelInput(
      toSbv2Utterance(analyzeWithWords(dict, TEXT, flat)),
      tokenizer,
      rules,
    );
    assertEquals(heiban.phones, fixed.phones);
    assertEquals(fixed.tones, [0, 1, 1, 0, 0, 0, 0, 0], "頭高: 第 1 モーラだけ高い");
    assertEquals(heiban.tones, [0, 0, 0, 1, 1, 1, 1, 0], "平板: 第 1 モーラの後で上がる");

    // 差は front へ渡る ID まで届く（tone 基点 6 を足した値・blank 位置は blankId）。
    assertEquals(fixed.ids.phoneIds, heiban.ids.phoneIds, "音素 ID は型で変わらない");
    assert(
      fixed.ids.toneIds.some((id, index) => id !== heiban.ids.toneIds[index]),
      "トーン ID が型の違いを反映していない",
    );
    // word2ph も同一解析から出ている（1 語に畳まれた overlay の語割りを踏む）。
    assertEquals(
      fixed.baseWord2ph.reduce((sum, count) => sum + count, 0),
      fixed.phones.length,
      "sum(word2ph) が音素数と違う",
    );
  },
});

Deno.test({
  name: "解析どおりの発話は門を素通りし、tone だけを直した発話も通る",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const overlay = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const utterance = toSbv2Utterance(analyzeWithWords(dict, TEXT, overlay));
    assertEquals(utterance.moras.length, 3, "1 句 3 モーラに畳まれていない");
    assertEquals(utterance.moras.map((mora) => mora.tone), [1, 0, 0], "頭高になっていない");

    // 核を平板へ直したのと同じ編集（モーラの tone を直接置く）。
    const flattened: Sbv2Utterance = {
      ...utterance,
      moras: utterance.moras.map((mora, index) => ({ ...mora, tone: index === 0 ? 0 : 1 })),
    };
    const edited = buildSbv2ModelInput(flattened, tokenizer, rules);
    const head = buildSbv2ModelInput(utterance, tokenizer, rules);
    assertEquals(edited.tones, [0, 0, 0, 1, 1, 1, 1, 0], "平板のトーン列になっていない");
    assertEquals(edited.phones, head.phones, "音素列が動いた");
    assertEquals(edited.baseWord2ph, head.baseWord2ph, "word2ph はトーンを読まない");
    assertEquals(edited.bertText, head.bertText, "BERT 入力はトーンを読まない");

    // 別の解析（overlay 無し）から採った words を混ぜると入口で落ちる。
    assertThrows(
      () =>
        buildSbv2ModelInput(
          { ...utterance, words: toSbv2Utterance(analyzeWithWords(dict, TEXT)).words },
          tokenizer,
          rules,
        ),
      Sbv2InputError,
    );
  },
});

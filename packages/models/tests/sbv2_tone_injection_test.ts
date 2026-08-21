// SBV2 のトーン注入席（修正辞書 `overlay` + `given_tone`）の挙動テスト。
//
// この席が壊れると **shape は合ったまま音だけが変わる** — 修正辞書が効かなければ元の誤読が
// そのまま合成され、`given_tone` が黙って無視されればユーザーが直したはずのアクセントが
// 元に戻る。どちらも例外は出ず、front も voice も通る。ここでは「差し替えが実際に最終 ID
// まで届くこと」と「壊れた指定が門で止まること」を分けて押さえる。
//
// 値域・長さの門（`resolveGivenTone`）と置き換え規則（`overlayFor`）は辞書なしで走る。
// 読み・アクセントの変化は実辞書が要るので、資産があるときだけ走らせる。

import { assert, assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import {
  type FrontendResult,
  JtdDictionary,
  OverlayDictionary,
  type OverlayEntry,
} from "@hdae/yomi";
import { analyzeSbv2Text, resolveGivenTone } from "../src/sbv2/text/analyze.ts";
import { Sbv2InputError } from "../src/sbv2/errors.ts";
import { toSbv2PhoneTone } from "../src/sbv2/text/phone-tone.ts";
import {
  assertProsodyPhones,
  assertProsodyShape,
  toSbv2Prosody,
} from "../src/sbv2/text/prosody.ts";
import { type JpExtraRules, parseJpExtraRules } from "../src/sbv2/text/symbols.ts";
import { DebertaTokenizer } from "../src/sbv2/text/tokenizer.ts";
import { overlayFor } from "../src/sbv2/pipeline.ts";

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

// ---- 辞書なしで走る門 --------------------------------------------------------

Deno.test("resolveGivenTone: 指定が無ければ解析のトーンをそのまま通す", () => {
  const analyzed = [0, 1, 1, 0];
  assertEquals(resolveGivenTone(analyzed, undefined), analyzed);
});

Deno.test("入力起因の失敗は Sbv2InputError で飛ぶ（HTTP なら 400 と 500 を分けられる）", () => {
  // 型を分ける目的は分岐で、内部不変条件の破れ（素の Error）と混ざらないことがその条件。
  assertThrows(() => resolveGivenTone([0, 1], [0]), Sbv2InputError);
  assertThrows(() => resolveGivenTone([0, 1], [0, 2]), Sbv2InputError);
});

Deno.test("resolveGivenTone: 長さ不一致は期待と実際を添えて落とす", () => {
  // 黙って切り詰め・0 埋めをすると、音素とトーンが 1 個ずれたまま front まで届く。
  assertThrows(() => resolveGivenTone([0, 1, 0], [0, 1]), Error, "givenTone の長さ 2");
  assertThrows(() => resolveGivenTone([0, 1, 0], [0, 1, 0, 1]), Error, "音素数 3");
});

Deno.test("resolveGivenTone: 0/1 以外は落とす（toneStart 加算後では捕まらない値）", () => {
  // 2 も -1 も 6 を足せば 0..11 の範囲に収まるので、`phonesTonesToModelIds` の範囲検査は
  // 素通りする。生値で見る門がここに要る理由そのもの。
  assertThrows(() => resolveGivenTone([0, 0], [0, 2]), Error, "givenTone[1] = 2");
  assertThrows(() => resolveGivenTone([0, 0], [-1, 0]), Error, "givenTone[0] = -1");
  assertThrows(() => resolveGivenTone([0], [1.5]), Error, "givenTone[0] = 1.5");
});

// ---- 実辞書（あれば）--------------------------------------------------------

const DICT_DIR = new URL("../../../outputs/yomi/", import.meta.url);

/** `outputs/yomi/` に置かれた JTD1 辞書（版が上がると綴りが変わるので拡張子で拾う）。 */
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
    `[karume] ${DICT_DIR.pathname} に *.jtd が無いため修正辞書の検査を SKIP する。` +
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

/** 修正辞書で読みを固定する対象。本辞書では 1 字ずつ別の語に割れる（下で実測する）。 */
const TEXT = "香留芽";
const textTokenizer = (): DebertaTokenizer => tinyTokenizer(["香", "留", "芽"]);

const entry = (reading: string, accentType: number): OverlayEntry => ({
  surface: TEXT,
  reading,
  accentType,
});

Deno.test({
  name: "overlay: 読みが差し替わり、アクセント型がトーン列に出る",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const base = analyzeSbv2Text(dict, TEXT, tokenizer, rules);

    // 読み: 本辞書の読みを固定値で書かない（辞書の版が上がると変わる — 検証したいのは
    // 「overlay が指定どおりの音素列にする」ことで、本辞書がどう読むかではない）。
    const head = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const fixed = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay: head });
    assertEquals(fixed.phones, ["_", "k", "a", "r", "u", "m", "e", "_"]);
    assert(base.phones.join(" ") !== fixed.phones.join(" "), "overlay 無しと同じ読みになっている");

    // アクセント: 同じ読みで型だけ変えると、音素列は同一のままトーンだけが動く
    // （読みの違いに引きずられていないことの実証）。
    const flat = new OverlayDictionary(dict, [entry("カルメ", 0)]);
    const heiban = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay: flat });
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
  name: "givenTone: 解析のトーンを置き換え、front の tone ID に反映される",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const base = analyzeSbv2Text(dict, TEXT, tokenizer, rules);
    // 反転させる（解析が何を返しても必ず全位置が変わる = 「無視している実装」を必ず捕まえる）。
    const flipped = base.tones.map((tone) => 1 - tone);

    const injected = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { givenTone: flipped });
    assertEquals(injected.tones, flipped);
    assertEquals(injected.phones, base.phones, "音素列はトーン指定で変わらない");
    assertEquals(injected.ids.phoneIds, base.ids.phoneIds, "音素 ID はトーン指定で変わらない");
    assertEquals(injected.bertText, base.bertText, "BERT 入力はトーンを読まない");
    assertEquals(injected.baseWord2ph, base.baseWord2ph, "word2ph はトーンを読まない");

    // add_blank 後の実音素位置は奇数添字（`intersperse` の規約）。
    for (const [index, tone] of flipped.entries()) {
      assertEquals(injected.ids.toneIds[2 * index + 1], TEST_RULES_JSON.toneStart + tone);
      assertEquals(injected.ids.toneIds[2 * index], TEST_RULES_JSON.blankId, "blank 位置が汚れた");
    }
  },
});

Deno.test({
  name: "overlayFor: 要求側の overlay は既定席を置き換える（合成しない）",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const cache: {
      defaultOverlay: readonly OverlayEntry[];
      defaultOverlayResolved: OverlayDictionary | undefined;
    } = { defaultOverlay: [entry("カルメ", 1)], defaultOverlayResolved: undefined };

    const standing = overlayFor(cache, dict, undefined);
    assert(standing !== undefined, "既定席が解決されていない");
    const once = overlayFor(cache, dict, [entry("カオルメ", 0)]);
    assert(once !== undefined, "要求側が解決されていない");

    // 置き換わっている（合成なら既定席の 1 件も残る）。
    assertEquals(once.entries.map((resolved) => resolved.reading), ["カオルメ"]);
    assertEquals(
      analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay: once }).phones,
      ["_", "k", "a", "o", "r", "u", "m", "e", "_"],
    );

    // 既定席は無傷のまま次の解析へ戻る（1 回きりの指定が居座らない）。
    // MUST: 同一性で見る（deep-equal だと解決し直した別インスタンスでも通ってしまう）。
    const after = overlayFor(cache, dict, undefined);
    assertStrictEquals(after, standing, "既定席が解決し直された（キャッシュが効いていない）");
    assertEquals(
      analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay: after }).phones,
      ["_", "k", "a", "r", "u", "m", "e", "_"],
    );
  },
});

// ---- 下書き（句 / モーラ構造）の往復 ----------------------------------------

/** 「カルメ」1 句ぶんの解析結果（辞書なしで門を叩くための最小の形）。 */
const karumeResult = (accentNucleus: number, punctuations: string[] = []): FrontendResult => ({
  normalizedText: "カルメ",
  leadingPunctuations: [],
  accentPhrases: [{
    moras: [
      { kana: "カ", consonant: "k", vowel: "a" },
      { kana: "ル", consonant: "r", vowel: "u" },
      { kana: "メ", consonant: "m", vowel: "e" },
    ],
    accentNucleus,
    pauseAfter: "none",
    punctuations,
  }],
});

Deno.test("toSbv2Prosody: SBV2 が読む欄だけを出す（渡せるのに効かない欄を作らない）", () => {
  const prosody = toSbv2Prosody(karumeResult(1, ["."]));
  // 解析結果に居た normalizedText / pauseAfter / devoiced は落ちている。
  assertEquals(Object.keys(prosody).sort(), ["leadingPunctuations", "phrases"]);
  assertEquals(Object.keys(prosody.phrases[0]).sort(), ["accentNucleus", "moras", "punctuations"]);
  assertEquals(Object.keys(prosody.phrases[0].moras[0]).sort(), ["consonant", "kana", "vowel"]);
  assertEquals(prosody.phrases[0].punctuations, ["."]);
});

Deno.test("toSbv2Prosody: 範囲外の核は正規形へ丸めて出す（往復不能を作らない）", () => {
  // yomi の moraTones は範囲外核を尾高相当へ黙ってクランプするので、生値のまま下書きに載せると
  // 「解析どおりの下書きを戻したのに範囲検査で落ちる」入力が作れてしまう。
  const clamped = toSbv2Prosody(karumeResult(9));
  assertEquals(clamped.phrases[0].accentNucleus, 3, "モーラ数へ丸めていない");
  // 丸めてもトーン列は変わらない（音は同じまま、往復だけが通るようになる）。
  assertEquals(
    toSbv2PhoneTone(clamped, "_").tones,
    toSbv2PhoneTone(toSbv2Prosody(karumeResult(3)), "_").tones,
  );
});

Deno.test("下書きは文頭の記号を保つ（フラットな音素列からは復元できない情報）", () => {
  // `{ phones, tones }` だけでは leadingPunctuations の個数も句境界も割り戻せない
  // （文頭に記号がある入力で詰む）— 構造で返す理由そのもの。
  const result: FrontendResult = { ...karumeResult(1), leadingPunctuations: ["…"] };
  const prosody = toSbv2Prosody(result);
  assertEquals(prosody.leadingPunctuations, ["…"]);
  assertEquals(toSbv2PhoneTone(prosody, "_").phones, [
    "_",
    "…",
    "k",
    "a",
    "r",
    "u",
    "m",
    "e",
    "_",
  ]);
  assertEquals(toSbv2PhoneTone(prosody, "_").tones, [0, 0, 1, 1, 0, 0, 0, 0, 0]);
});

Deno.test("assertProsodyShape: 核の範囲外は落とす（moraTones は黙ってクランプする）", () => {
  const shifted = (accentNucleus: number) => ({
    ...toSbv2Prosody(karumeResult(1)),
    phrases: [{ ...toSbv2Prosody(karumeResult(1)).phrases[0], accentNucleus }],
  });
  assertThrows(() => assertProsodyShape(shifted(4)), Sbv2InputError, "0..3 の範囲外");
  assertThrows(() => assertProsodyShape(shifted(-1)), Sbv2InputError, "0..3 の範囲外");
  assertThrows(() => assertProsodyShape(shifted(1.5)), Sbv2InputError, "0..3 の範囲外");
  // 平板 0 と尾高 3 は正当。
  assertProsodyShape(shifted(0));
  assertProsodyShape(shifted(3));
});

Deno.test("assertProsodyPhones: 長さが合っていても内容が違えば落とす", () => {
  // 梱包規則を外部で再実装してズレたとき、長さ検査だけでは素通りする（音だけ崩れる）。
  const analyzed = ["_", "k", "a", "r", "u", "m", "e", "_"];
  assertProsodyPhones(analyzed, [...analyzed]);
  assertThrows(
    () => assertProsodyPhones(analyzed, ["_", "k", "a", "r", "u", "m", "o", "_"]),
    Sbv2InputError,
    "音素[6]",
  );
  assertThrows(
    () => assertProsodyPhones(analyzed, analyzed.slice(0, 7)),
    Sbv2InputError,
    "音素数 7",
  );
});

Deno.test({
  name: "prosody: 解析どおりの下書きを戻すと恒等（無指定と同じ ID 列が出る）",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const base = analyzeSbv2Text(dict, TEXT, tokenizer, rules);

    const roundTrip = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { prosody: base.prosody });
    assertEquals(roundTrip.phones, base.phones);
    assertEquals(roundTrip.tones, base.tones);
    assertEquals(roundTrip.ids.phoneIds, base.ids.phoneIds);
    assertEquals(roundTrip.ids.toneIds, base.ids.toneIds);
    assertEquals(roundTrip.baseWord2ph, base.baseWord2ph);
    assertEquals(roundTrip.bertText, base.bertText);
  },
});

Deno.test({
  name: "prosody: 核だけ直すとトーンだけが動く（overlay で型を変えたのと同じ列になる）",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    // 読みを固定して 1 句 3 モーラにする（頭高）。
    const overlay = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const head = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay });
    assertEquals(head.prosody.phrases.length, 1, "1 句に畳まれていない");
    assertEquals(head.prosody.phrases[0].accentNucleus, 1, "頭高になっていない");

    // 核を平板へ直して戻す。
    const flattened = {
      ...head.prosody,
      phrases: [{ ...head.prosody.phrases[0], accentNucleus: 0 }],
    };
    const edited = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay, prosody: flattened });

    assertEquals(edited.tones, [0, 0, 0, 1, 1, 1, 1, 0], "平板のトーン列になっていない");
    assertEquals(edited.phones, head.phones, "音素列が動いた");
    assertEquals(edited.baseWord2ph, head.baseWord2ph, "word2ph はトーンを読まない");
    assertEquals(edited.bertText, head.bertText, "BERT 入力はトーンを読まない");
    assertEquals(edited.ids.phoneIds, head.ids.phoneIds, "音素 ID が動いた");
    assert(
      edited.ids.toneIds.some((id, index) => id !== head.ids.toneIds[index]),
      "トーン ID が核の違いを反映していない",
    );
  },
});

Deno.test({
  name: "prosody: 音素数が変わる編集・読み替え・text の取り違えは落とす",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const overlay = new OverlayDictionary(dict, [entry("カルメ", 1)]);
    const head = analyzeSbv2Text(dict, TEXT, tokenizer, rules, { overlay });
    const phrase = head.prosody.phrases[0];
    const withPhrase = (moras: typeof phrase.moras, punctuations = phrase.punctuations) => ({
      ...head.prosody,
      phrases: [{ ...phrase, moras, punctuations }],
    });

    // ① モーラの読み替え（音素数は変わらない = 長さ検査だけなら素通りする）。
    assertThrows(
      () =>
        analyzeSbv2Text(dict, TEXT, tokenizer, rules, {
          overlay,
          prosody: withPhrase([
            phrase.moras[0],
            { kana: "ロ", consonant: "r", vowel: "o" },
            phrase.moras[2],
          ]),
        }),
      Sbv2InputError,
      "音素[4]",
    );

    // ② モーラを 1 つ落とす（音素数が変わる）。
    assertThrows(
      () =>
        analyzeSbv2Text(dict, TEXT, tokenizer, rules, {
          overlay,
          prosody: withPhrase(phrase.moras.slice(0, 2)),
        }),
      Sbv2InputError,
      "音素数が変わる編集",
    );

    // ③ 記号を足す（疑問形の上げを句へ差し込もうとした場合）。
    assertThrows(
      () =>
        analyzeSbv2Text(dict, TEXT, tokenizer, rules, {
          overlay,
          prosody: withPhrase(phrase.moras, ["."]),
        }),
      Sbv2InputError,
      "音素数が変わる編集",
    );

    // ④ overlay を渡し忘れた（= 別の解析で採った下書きを戻した）。
    assertThrows(
      () => analyzeSbv2Text(dict, TEXT, tokenizer, rules, { prosody: head.prosody }),
      Sbv2InputError,
    );
  },
});

Deno.test({
  name: "prosody と givenTone の同時指定は落とす（優先規則を作らない）",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const rules = testRules();
    const tokenizer = textTokenizer();
    const base = analyzeSbv2Text(dict, TEXT, tokenizer, rules);
    assertThrows(
      () =>
        analyzeSbv2Text(dict, TEXT, tokenizer, rules, {
          prosody: base.prosody,
          givenTone: base.tones,
        }),
      Sbv2InputError,
      "同時に指定できない",
    );
  },
});

Deno.test({
  name: "overlayFor: 解決済み OverlayDictionary はそのまま使う（毎回作り直さない）",
  ignore: dictUrl === undefined,
  fn: () => {
    const dict = jtd();
    const cache: {
      defaultOverlay: readonly OverlayEntry[] | OverlayDictionary | undefined;
      defaultOverlayResolved: OverlayDictionary | undefined;
    } = { defaultOverlay: undefined, defaultOverlayResolved: undefined };
    const resolved = new OverlayDictionary(dict, [entry("カルメ", 1)]);

    // 要求側: 解決し直さない（数千語の辞書を毎合成で解決しないための席）。
    assertStrictEquals(overlayFor(cache, dict, resolved), resolved);
    // 既定席: 解決済みで置いても同じものが返り続ける。
    cache.defaultOverlay = resolved;
    assertStrictEquals(overlayFor(cache, dict, undefined), resolved);
    assertStrictEquals(overlayFor(cache, dict, undefined), resolved);
  },
});

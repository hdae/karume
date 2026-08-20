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
import { JtdDictionary, OverlayDictionary, type OverlayEntry } from "@hdae/yomi";
import { analyzeSbv2Text, resolveGivenTone } from "../src/sbv2/text/analyze.ts";
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
      overlayEntries: readonly OverlayEntry[];
      overlay: OverlayDictionary | undefined;
    } = { overlayEntries: [entry("カルメ", 1)], overlay: undefined };

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

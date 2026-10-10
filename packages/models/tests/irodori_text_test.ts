// Irodori テキスト層（src/irodori/text/）の Python 正本とのパリティ検証。
//
// フィクスチャ `irodori-text/parity.json` は exporter（tools/exporter/irodori_tokenizer.py）が
// 生成する。中身は
//  ・`normalize_text` の `{raw, normalized}` 33 ケース（各置換規則を最低 1 回発火させたもの）
//  ・トークナイザの `{raw, normalized, ids}` 14 ケースと `batch_encode` の最終形 3 ケース。
//    **正本は tokenizers と transformers の 2 経路**で、emit 時に両者の一致まで実測してある
//  ・caption 経路（上流は strip のみ）と text 経路（normalize）の id 列を同じ生テキストから
//    2 通り採った 6 ケース
//  ・その再現に要る語彙の**部分集合**（102,400 本を commit しない。格子が全語彙と同じである
//    ことは exporter 側が語彙全体を逆向きに走査して実測する）
//  ・NFKC が恒等でない単一コードポイントの写像表**全体**
//
// 実資産（outputs/ 配下）にも GPU にも依存しないので常時走る。
//
// 縛るのは振る舞い: 「同じ生テキストを入れたら Python と同じ id 列が出る」。表の持ち方や
// 内部の分割の仕方は縛らない。

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { packCaptionIds, packIds } from "../src/irodori/host/pack.ts";
import { normalizeText } from "../src/irodori/text/normalize.ts";
import {
  IrodoriTokenizer,
  type IrodoriTokenizerAssets,
  parseIrodoriTokenizerAsset,
} from "../src/irodori/text/tokenizer.ts";

type EncodeCase = {
  readonly name: string;
  readonly why: string;
  readonly raw: string;
  readonly normalized: string;
  readonly ids: number[];
};

type BatchCase = {
  readonly name: string;
  readonly why: string;
  readonly raw: string;
  readonly normalized: string;
  readonly maxLength: number;
  readonly idsPadded: number[];
  readonly mask: boolean[];
};

type CaptionCase = {
  readonly name: string;
  readonly why: string;
  readonly raw: string;
  readonly stripped: string;
  readonly normalized: string;
  readonly maxLength: number;
  readonly idsPacked: number[];
  readonly normalizedIdsPacked: number[];
  readonly normalizeSensitive: boolean;
};

type NormalizeCase = {
  readonly name: string;
  readonly why: string;
  readonly raw: string;
  readonly normalized: string;
};

type Fixture = {
  readonly asset: {
    readonly vocab: [string, number, number][];
    readonly minScore: number;
    readonly maxTokenLength: number;
    readonly addedTokens: [string, number][];
    readonly bosId: number;
    readonly padId: number;
    readonly unkId: number;
    readonly byteBaseId: number;
  };
  readonly encode: {
    readonly cases: EncodeCase[];
    readonly batch: BatchCase[];
    readonly caption: CaptionCase[];
  };
  readonly normalize: { readonly cases: NormalizeCase[] };
  readonly nfkcDiff: Record<string, string>;
};

const FIXTURE_PATH = new URL("./fixtures/irodori-text/parity.json", import.meta.url);
const fixture = JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture;

const assets: IrodoriTokenizerAssets = {
  vocab: new Map(fixture.asset.vocab.map(([token, id, score]) => [token, { id, score }])),
  // MUST: 部分集合から導かない — 未知ノードのスコアと探索幅は語彙**全体**から決まる。
  minScore: fixture.asset.minScore,
  maxTokenLength: fixture.asset.maxTokenLength,
  unkId: fixture.asset.unkId,
  byteBaseId: fixture.asset.byteBaseId,
  bosId: fixture.asset.bosId,
  padId: fixture.asset.padId,
  addedTokens: new Map(fixture.asset.addedTokens),
};

const tokenizer = new IrodoriTokenizer(assets);

// ---- フィクスチャ本体 -------------------------------------------------------

Deno.test("フィクスチャが空でない（取り違えで全ケース素通しになっていない）", () => {
  // 1 本目に置く。読み違い / 生成失敗でケースが空になると、以降のループが 0 回になって
  // 「緑だが何も検証していない」状態が黙って成立する。
  assertEquals(fixture.encode.cases.length, 14);
  assertEquals(fixture.encode.batch.length, 3);
  assertEquals(fixture.normalize.cases.length, 33);
  assert(fixture.asset.vocab.length > 256, `語彙の部分集合が ${fixture.asset.vocab.length} 本`);
  assert(Object.keys(fixture.nfkcDiff).length > 4000, "NFKC 差分表が薄い");
  // MUST: 未知ノードの重みは語彙**全体**の最小スコアから決まる。部分集合の最小に一致したら、
  // フィクスチャか組み立てのどちらかが全体から採れていない。golden の id 列はこの差を検出
  // できない（未知区間に競合する既知トークンが無いので経路が変わらない — 故障注入で実測）
  // ので、値そのものをここで縛る。
  const subsetMin = Math.min(...fixture.asset.vocab.map(([, , score]) => score));
  assert(
    assets.minScore < subsetMin,
    `minScore ${assets.minScore} が部分集合の最小 ${subsetMin} と同じ`,
  );
});

// ---- ① 正規化（トークナイザの手前で切り分ける）------------------------------

for (const testCase of fixture.normalize.cases) {
  Deno.test(`正規化パリティ [${testCase.name}] ${testCase.why}`, () => {
    assertEquals(normalizeText(testCase.raw), testCase.normalized);
  });
}

// ---- ② 符号化（生テキスト → id 列）------------------------------------------

for (const testCase of fixture.encode.cases) {
  Deno.test(`符号化パリティ [${testCase.name}] ${testCase.why}`, () => {
    // 中間（正規化後の文字列）でも突き合わせる — 全経路だけを見ると、正規化と符号化の
    // 取り違えが打ち消し合った場合に気づけない。
    assertEquals(normalizeText(testCase.raw), testCase.normalized, "正規化後の文字列");
    assertEquals(tokenizer.encode(testCase.normalized), testCase.ids, "正規化済みからの id 列");
    assertEquals(tokenizer.encode(normalizeText(testCase.raw)), testCase.ids, "生からの全経路");
  });
}

Deno.test("符号化: byte_fallback は UTF-8 バイト列そのもの（語彙を見ずに再現できる）", () => {
  // c ケース（語彙外絵文字）の期待値を、フィクスチャの id 列とは独立に UTF-8 から組む。
  // `byteBaseId` の取り違えはここで落ちる。
  const byteCase = fixture.encode.cases.find((entry) => entry.name === "emoji-byte-fallback");
  assert(byteCase !== undefined, "c ケースがフィクスチャに無い");
  const bytes = new TextEncoder().encode(byteCase.raw);
  assertEquals(byteCase.ids, [...bytes].map((byte) => fixture.asset.byteBaseId + byte));
});

Deno.test("資産: バイトトークン 256 本の id が byteBaseId + バイト値で連番になっている", () => {
  // byte_fallback の id 算術（`unigram.ts` の展開）が成立する前提そのもの。
  for (let byte = 0; byte < 256; byte++) {
    const token = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
    const entry = assets.vocab.get(token);
    assert(entry !== undefined, `${token} が語彙の部分集合に無い`);
    assertEquals(entry.id, fixture.asset.byteBaseId + byte, `${token} の id`);
  }
});

// ---- ③ batch_encode の最終形（BOS 前置 + 切り詰め + 右 pad + マスク）---------

for (const testCase of fixture.encode.batch) {
  Deno.test(`組み立てパリティ [${testCase.name}] ${testCase.why}`, () => {
    const padded = tokenizer.encodePadded(testCase.normalized, testCase.maxLength);
    assertEquals([...padded.ids], testCase.idsPadded, "idsPadded");
    assertEquals([...padded.mask], testCase.mask.map(Number), "mask");
  });
}

Deno.test("組み立て: truncation が実際に発火し、BOS が残って全長は max_length になる", () => {
  const truncated = fixture.encode.batch.find((entry) => entry.name === "batch-truncated");
  assert(truncated !== undefined, "truncation ケースがフィクスチャに無い");
  // 本体の予算を `max_length` と取り違える故障は、このケースの id 列パリティが落とす
  // （発火しないケースだけなら素通りする）。
  assert(
    tokenizer.encode(truncated.normalized).length + 1 > truncated.maxLength,
    "truncation が発火していない",
  );
  assertEquals(truncated.idsPadded.length, truncated.maxLength);
  assertEquals(truncated.idsPadded[0], fixture.asset.bosId);
  assertEquals(truncated.mask.filter(Boolean).length, truncated.maxLength);
});

Deno.test("組み立て: 短い入力は pad で右詰めされ、マスクが有効長を指す", () => {
  const short = fixture.encode.batch.find((entry) => entry.name === "batch-short");
  assert(short !== undefined, "短いケースがフィクスチャに無い");
  const used = tokenizer.encode(short.normalized).length + 1;
  assertEquals(short.mask.filter(Boolean).length, used);
  assertEquals(
    short.idsPadded.slice(used),
    Array(short.maxLength - used).fill(fixture.asset.padId),
  );
});

// ---- ③' caption 経路（上流は strip のみ・normalize は text 専用）--------------
//
// 上流 `inference_runtime._synthesize` が caption に掛けるのは `str(...).strip()` だけで、
// `normalize_text` は text 専用。ここが揃っていないと conditioning が例外も警告も無く別物に
// なる（外側括弧の剥がし・NFKC・記号削除のぶん）ので、**同じ生テキストを 2 経路へ通した
// id 列**をフィクスチャに持たせ、両方を突き合わせる。

for (const testCase of fixture.encode.caption) {
  Deno.test(`caption パリティ [${testCase.name}] ${testCase.why}`, () => {
    assertEquals(
      [...packCaptionIds(tokenizer, testCase.raw, testCase.maxLength)],
      testCase.idsPacked,
      "caption 経路（strip のみ）",
    );
    assertEquals(
      [...packIds(tokenizer, testCase.raw, testCase.maxLength, "text")],
      testCase.normalizedIdsPacked,
      "text 経路（normalize + strip）",
    );
    assertEquals(normalizeText(testCase.raw).trim(), testCase.normalized, "正規化後の文字列");
  });
}

Deno.test("caption: 2 経路の id 列が実際に割れる（門が恒真でない証拠）", () => {
  // ケースが空だとループが 0 回になって「緑だが何も検証していない」状態が黙って成立する。
  assertEquals(fixture.encode.caption.length, 6);
  const sensitive = fixture.encode.caption.filter((entry) => entry.normalizeSensitive);
  assert(sensitive.length >= 5, `正規化に感受するケースが ${sensitive.length} 件しかない`);
  for (const entry of sensitive) {
    assertNotEquals(entry.idsPacked, entry.normalizedIdsPacked, `${entry.name} が 2 経路で同じ`);
  }
  // 公式 Voice Design 文は正規化に無感 = 既存 golden（caption 系）はこの修正で動かない側。
  const insensitive = fixture.encode.caption.filter((entry) => !entry.normalizeSensitive);
  assertEquals(insensitive.map((entry) => entry.name), ["caption-official"]);
  for (const entry of insensitive) {
    assertEquals(entry.idsPacked, entry.normalizedIdsPacked);
  }
});

// ---- ④ NFKC（正本 = Python の unicodedata / こちら = JS エンジンの ICU）-------

Deno.test("NFKC: サロゲートを除く全コードポイントで正本の差分表と完全一致する", () => {
  // 両方向を見る: 表にあるのに JS が変えない（= 正本の方が新しい）ケースと、表に無いのに
  // JS が変える（= JS の方が新しい）ケースのどちらも、正規化結果を静かに割る。
  // 1 件ずつ assert すると 1.1M 回の呼び出しになるので、突合はループ内で行い件数だけ集める。
  const mismatches: string[] = [];
  for (let cp = 0; cp < 0x110000; cp++) {
    if (cp >= 0xd800 && cp <= 0xdfff) continue;
    const source = String.fromCodePoint(cp);
    const want = fixture.nfkcDiff[String(cp)] ?? source;
    const got = source.normalize("NFKC");
    if (got !== want && mismatches.length < 8) {
      mismatches.push(`U+${cp.toString(16).toUpperCase()}: 表=${want} JS=${got}`);
    }
  }
  assertEquals(mismatches, [], "NFKC が正本と割れるコードポイントがある");
});

// ---- ⑤ 資産パーサ（外部境界）------------------------------------------------

/** バイト値 → byte_fallback の綴り（`<0x00>`..`<0xFF>`）。 */
const byteToken = (byte: number): string =>
  `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;

/** バイト 256 本の綴り（`byteBaseId` から連番で並べる行）。 */
const BYTE_TOKENS: readonly string[] = Array.from({ length: 256 }, (_, byte) => byteToken(byte));

/**
 * `byteBaseId` は 256 本連番の先頭なので、成功パスの検査には `id + 255 < 語彙の行数` を
 * 満たす語彙が要る（G4-02）。本物のトークンの直後へバイト 256 本の行を足し、その先頭を
 * `byteBaseId` として返す（配布物と同じく、連番の行は `<0xNN>` の綴りで並ぶ）。
 * バイト行はスコア 0（本物の最小スコアより大きい）に揃え、`minScore` を狂わせない。
 */
const padVocabForByteBase = (
  realTokens: readonly string[],
  realScores: readonly number[],
): { vocabText: string; scores: number[]; byteBaseId: number } => ({
  vocabText: [...realTokens, ...BYTE_TOKENS].join("\n"),
  scores: [...realScores, ...Array(BYTE_TOKENS.length).fill(0)],
  byteBaseId: realTokens.length,
});

Deno.test("資産パーサ: 素の JSON から資産表を組み、id / スコアを行番号で対応付ける", () => {
  const { vocabText, scores, byteBaseId } = padVocabForByteBase(
    ["<unk>", "<s>", "<pad>", "▁こんにちは", "世界"],
    [0, 0, 0, -1.5, -2.5],
  );
  const parsed = parseIrodoriTokenizerAsset({
    vocabText,
    scores,
    addedTokens: [["<unk>", 0], ["<s>", 1], ["<pad>", 2]],
    bosId: 1,
    padId: 2,
    unkId: 0,
    byteBaseId,
  });
  assertEquals(parsed.vocab.get("世界"), { id: 4, score: -2.5 });
  assertEquals(parsed.minScore, -2.5);
  // 探索幅はコードポイント数（`▁こんにちは` = 6 — UTF-16 単位でもバイト数でもない）。
  assertEquals(parsed.maxTokenLength, 6);
  assertEquals(parsed.bosId, 1);
  assertEquals([...parsed.addedTokens.keys()], ["<unk>", "<s>", "<pad>"]);
});

Deno.test("資産パーサ: 行数不一致と必須キー欠落は fail loudly", () => {
  const { vocabText, scores, byteBaseId } = padVocabForByteBase(
    ["<unk>", "<s>", "<pad>"],
    [0, 0, 0],
  );
  const base = {
    vocabText,
    scores,
    addedTokens: [["<unk>", 0]],
    bosId: 1,
    padId: 2,
    unkId: 0,
    // 本物は 3 行（バイト 256 本の行はその後ろ — 上の doc 参照）。
    byteBaseId,
  };
  // スコアが 1 本欠けても Viterbi は落ちない（語彙に無い断片はバイト展開へ逃げる）ので、
  // ここで落とさないと id が総ずれしたまま沈黙で流れる。
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, scores: [0, 0] }),
    Error,
    "scores の長さ",
  );
  for (
    const key of ["vocabText", "scores", "addedTokens", "bosId", "padId", "unkId", "byteBaseId"]
  ) {
    const broken: Record<string, unknown> = { ...base };
    delete broken[key];
    assertThrows(
      () => parseIrodoriTokenizerAsset(broken),
      Error,
      key === "scores" ? "scores" : key,
    );
  }
  assertThrows(() => parseIrodoriTokenizerAsset(undefined), Error, "オブジェクトでない");
});

Deno.test("資産パーサ: 特殊 id は整数かつ語彙の行数未満（Int32Array で沈黙切り捨てになる前に落とす）", () => {
  const { vocabText, scores, byteBaseId } = padVocabForByteBase(
    ["<unk>", "<s>", "<pad>"],
    [0, 0, 0],
  );
  const VOCAB_SIZE = vocabText.split("\n").length;
  const base = {
    vocabText,
    scores,
    addedTokens: [["<unk>", 0]],
    bosId: 1,
    padId: 2,
    unkId: 0,
    byteBaseId,
  };
  // 非整数は `Int32Array` 代入で切り捨てられ、範囲外は wrap する — どちらも「別のトークンを
  // 指す」沈黙誤値で、グラフの embedding gather まで表面化しない。
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, unkId: 1.5 }),
    Error,
    "tokenizer.unkId: トークン id が 0..2147483647 の整数でない（1.5）",
  );
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, byteBaseId: -1 }),
    Error,
    "tokenizer.byteBaseId: トークン id が 0..2147483647 の整数でない（-1）",
  );
  // 語彙の行数ちょうど = 1 つ外側（バイト行込みで VOCAB_SIZE 行）。
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, bosId: VOCAB_SIZE }),
    Error,
    `tokenizer.bosId: トークン id ${VOCAB_SIZE} が語彙の行数 ${VOCAB_SIZE} 以上`,
  );
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, addedTokens: [["<unk>", 0.5]] }),
    Error,
    "トークン id が 0..2147483647 の整数でない（0.5）",
  );
});

Deno.test("資産パーサ: byteBaseId はバイト 256 本の連番全体が語彙内にあることを要求する（G4-02）", () => {
  const VOCAB_SIZE = 300;
  const base = {
    vocabText: [...Array.from({ length: VOCAB_SIZE - 256 }, (_, i) => `t${i}`), ...BYTE_TOKENS]
      .join("\n"),
    scores: Array.from({ length: VOCAB_SIZE }, () => 0),
    addedTokens: [],
    bosId: 0,
    padId: 0,
    unkId: 0,
  };
  // 負例: byteBaseId=299 だと 299..554 が語彙 300 行に収まらない。
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, byteBaseId: 299 }),
    Error,
    "バイト 256 本",
  );
  // 正例: byteBaseId = 300 − 256 = 44 はちょうど収まる（44..299 の 256 本が末尾まで）。
  const parsed = parseIrodoriTokenizerAsset({ ...base, byteBaseId: 44 });
  assertEquals(parsed.byteBaseId, 44);
});

Deno.test("資産パーサ: byteBaseId からの 256 行は <0x00>..<0xFF> の綴りが順に並ぶ", () => {
  // 未知文字は `byteBaseId + バイト値` へ展開されるので、行が入れ替わった資産では HF（名前で
  // id を引く）と別の id が出る（"A" = 0x41 が `<0x42>` の行の id になる）。
  const VOCAB_SIZE = 300;
  const base = {
    scores: Array.from({ length: VOCAB_SIZE }, () => 0),
    addedTokens: [],
    bosId: 0,
    padId: 0,
    unkId: 0,
    byteBaseId: 44,
  };
  const words = Array.from({ length: VOCAB_SIZE - 256 }, (_, i) => `t${i}`);
  const swapped = [...BYTE_TOKENS];
  [swapped[0x41], swapped[0x42]] = [swapped[0x42], swapped[0x41]];
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, vocabText: [...words, ...swapped].join("\n") }),
    Error,
    `tokenizer.byteBaseId: 行 ${44 + 0x41} が <0x41> でない`,
  );
  // 1 つずれた位置（先頭が語彙の単語）も同じく落ちる。
  assertThrows(
    () =>
      parseIrodoriTokenizerAsset({
        ...base,
        vocabText: [...words, ...BYTE_TOKENS].join("\n"),
        byteBaseId: 43,
      }),
    Error,
    "tokenizer.byteBaseId: 行 43 が <0x00> でない",
  );
});

Deno.test("資産パーサ: 語彙の重複行・追加語彙の重複鍵は落とす（`Map` の後勝ちを禁じる）", () => {
  const VOCAB_SIZE = 300;
  const base = {
    vocabText: [...Array.from({ length: VOCAB_SIZE - 256 }, (_, i) => `t${i}`), ...BYTE_TOKENS]
      .join("\n"),
    scores: Array.from({ length: VOCAB_SIZE }, () => 0),
    addedTokens: [["t0", 0]],
    bosId: 0,
    padId: 0,
    unkId: 0,
    byteBaseId: 44,
  };
  // 正例（門が恒真でないことの対）。
  assertEquals(parseIrodoriTokenizerAsset(base).vocab.size, VOCAB_SIZE);

  // 行番号 = id なので、重複行は先の id を引けなくする。行数の突合（scores の本数）は
  // 重複で変わらないため、そちらでは 1 件も捕まらない。
  const duplicated = [
    ...Array.from({ length: VOCAB_SIZE - 256 }, (_, i) => `t${i}`),
    ...BYTE_TOKENS.slice(0, -1),
    "t0",
  ];
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, vocabText: duplicated.join("\n") }),
    Error,
    `tokenizer.vocabText: 行 ${VOCAB_SIZE - 1} が行 0 と同じトークン`,
  );
  // 追加語彙は正規化の**前**に切り出されるので、後勝ちは切り出し先 id を静かに変える。
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, addedTokens: [["t0", 0], ["t0", 1]] }),
    Error,
    "tokenizer.addedTokens: 鍵 ",
  );
});

Deno.test("資産パーサ: scores は有限（minScore が汚れると未知ノードの重みが潰れる）", () => {
  const VOCAB_SIZE = 300;
  const scores = Array.from({ length: VOCAB_SIZE }, () => 0);
  const base = {
    vocabText: [...Array.from({ length: VOCAB_SIZE - 256 }, (_, i) => `t${i}`), ...BYTE_TOKENS]
      .join("\n"),
    scores,
    addedTokens: [],
    bosId: 0,
    padId: 0,
    unkId: 0,
    byteBaseId: 44,
  };
  // NaN は `Math.min` を通して minScore を汚し、±Infinity は全経路を同値へ潰す。どちらも
  // 例外にならず「別の分割」になるだけ（本数の突合では捕まらない）。
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    assertThrows(
      () =>
        parseIrodoriTokenizerAsset({
          ...base,
          scores: scores.map((value, index) => (index === 7 ? bad : value)),
        }),
      Error,
      "tokenizer.scores[7]: 有限の数値でない",
    );
  }
});

// ---- 境界（golden のケースが実際に叩いているもの）----------------------------

Deno.test("対にならないサロゲートは受け付けない（正本に無い入力を黙って通さない）", () => {
  // 生成要求の本文として通るので入力起因（ADR 0107 決定 2 — 文字を直せば通る）。受理条件の
  // 正本は `toCodePoints` 1 本のままなので、所有者の診断は `cause` に残る。
  const error = assertThrows(() => tokenizer.encode("a\ud800b"), ModelInputError, "サロゲート");
  assert(error.cause instanceof Error);
  assert(!(error.cause instanceof ModelInputError), "cause は共有ヘルパの素の Error");
  // 逆側: 対になっているサロゲート（U+1F600 など）は通る。
  assert(tokenizer.encode("a\ud83d\ude00b").length > 0);
});

Deno.test("追加語彙は格子の**前**に切り出される（語彙側の分割に負けない）", () => {
  // 実資産の 18 本は語彙にも同じ綴り・同じ id で載っているので、g ケースの id 列では
  // この規則を捕まえられない（切り出しを外しても同じ列になると故障注入で実測）。合成資産で
  // 「格子に流すと別の列になる」形を作って直接叩く。
  const vocab = new Map([
    ["<|x|>", { id: 5, score: -20 }],
    ["<|", { id: 6, score: -1 }],
    ["x", { id: 7, score: -1 }],
    ["|>", { id: 8, score: -1 }],
  ]);
  const base = {
    vocab,
    minScore: -20,
    maxTokenLength: 5,
    unkId: 0,
    byteBaseId: 100,
    bosId: 1,
    padId: 2,
  };
  const added = new IrodoriTokenizer({ ...base, addedTokens: new Map([["<|x|>", 99]]) });
  assertEquals(added.encode("<|x|>"), [99], "追加語彙の id が直接出る");
  // 追加語彙から外すと格子が勝って 3 分割になる（= 上の検査が恒真でない証拠）。
  const plain = new IrodoriTokenizer({ ...base, addedTokens: new Map() });
  assertEquals(plain.encode("<|x|>"), [6, 7, 8], "格子はスコアの和で 3 分割を選ぶ");
});

Deno.test("Metaspace: 先頭に ▁ を足さない（prepend_scheme=never）", () => {
  // i ケース（先頭が半角空白）は置換で ▁ が付く。足す実装だと ▁▁ になって別の id 列になる。
  const leading = fixture.encode.cases.find((entry) => entry.name === "leading-space");
  const withSpaces = fixture.encode.cases.find((entry) => entry.name === "with-spaces");
  assert(leading !== undefined && withSpaces !== undefined, "i / h ケースがフィクスチャに無い");
  assert(leading.normalized.startsWith(" "), "i ケースが先頭空白でない");
  // 先頭に空白が無い h ケースの先頭 id は、▁ 付きトークンの id にはならない。
  const first = withSpaces.ids[0];
  const head = [...assets.vocab.entries()].find(([, entry]) => entry.id === first);
  assert(head !== undefined && !head[0].startsWith("▁"), `先頭トークン ${head?.[0]} に ▁ が付いた`);
});

// ---- 外側括弧の剥がし（③）の意味論と計算量 ------------------------------------

Deno.test("外側括弧の剥がし: 混在・途中で閉じる括弧の意味論", () => {
  // パリティのフィクスチャ（brackets-*）に無い形を、Python 正本 `strip_outer_brackets` と同じ
  // 規則（全体を囲む同種の対だけを剥がし、剥がせたら先頭から試し直す）で固定する。
  // NFKC は剥がしの後に掛かるので、残った全角括弧は ASCII になる。
  const cases: readonly (readonly [string, string])[] = [
    ["「「あ」」", "あ"],
    ["（「（あ）」）", "あ"],
    ["「「あ」と「い」」", "「あ」と「い」"],
    ["「あ」「い」", "「あ」「い」"],
    ["「（あ」）", "「(あ」)"],
    ["((a)", "((a)"],
    ["(a))", "(a))"],
    ["(「a)」", "(「a)」"],
    ["「」", ""],
    ["「「」」", ""],
    ["「", "「"],
    ["「😀」", "😀"],
    ["『（あ）い』", "(あ)い"],
  ];
  for (const [raw, normalized] of cases) assertEquals(normalizeText(raw), normalized, raw);
});

Deno.test("外側括弧の剥がし: 深い入れ子でも入力長に線形の時間で剥がす", () => {
  // 1 段剥がすたびに残り全体を走査して切り出す形は入れ子 n 段で二次時間になり、
  // n = 32,000（約 64 KB）で数秒メインスレッドを占有する。token 数の切り詰めは正規化の後なので、
  // 長い入力がそのまま届く。
  const depth = 32_000;
  const started = performance.now();
  const normalized = normalizeText("「".repeat(depth) + "a" + "」".repeat(depth));
  const elapsed = performance.now() - started;
  assertEquals(normalized, "a");
  assert(elapsed < 1_000, `入れ子 ${depth} 段の剥がしに ${elapsed.toFixed(0)} ms かかった`);
});

Deno.test("外側括弧の剥がし: 正本の逐次の形と乱択の入力で一致する", () => {
  // 期待側は正本 `strip_outer_brackets` の形をそのまま写したもの（1 段ずつ全体を走査して
  // 剥がす — 二次時間だが短い入力なら問題ない）。実装は対応表で線形に求めるので、2 つの形が
  // 括弧の混在・途中で閉じる・閉じない入力で割れないことを乱択で見る。
  const pairs: readonly (readonly [string, string])[] = [
    ["「", "」"],
    ["『", "』"],
    ["（", "）"],
    ["【", "】"],
    ["(", ")"],
  ];
  const reference = (text: string): string => {
    let cps = [...text];
    while (cps.length >= 2) {
      const pair = pairs.find(([open]) => open === cps[0]);
      if (pair === undefined || cps[cps.length - 1] !== pair[1]) break;
      let depth = 0;
      let enclosing = true;
      for (const [index, ch] of cps.entries()) {
        if (ch === pair[0]) depth += 1;
        else if (ch === pair[1]) depth -= 1;
        if (depth === 0 && index < cps.length - 1) {
          enclosing = false;
          break;
        }
      }
      if (!enclosing || depth !== 0) break;
      cps = cps.slice(1, -1);
    }
    return cps.join("");
  };
  const alphabet = ["「", "」", "（", "）", "(", ")", "a"];
  let state = 12345;
  const nextIndex = (bound: number): number => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state % bound;
  };
  let stripped = 0;
  for (let sample = 0; sample < 5_000; sample += 1) {
    const length = nextIndex(13);
    const raw = Array.from({ length }, () => alphabet[nextIndex(alphabet.length)]).join("");
    const expected = reference(raw);
    if (expected !== raw) stripped += 1;
    assertEquals(normalizeText(raw), expected.normalize("NFKC"), JSON.stringify(raw));
  }
  // 乱択が剥がす側を実際に踏んでいること（全部が素通しだと比較が恒真になる）。
  assert(stripped >= 50, `剥がれる入力が ${stripped} 件しかない`);
});

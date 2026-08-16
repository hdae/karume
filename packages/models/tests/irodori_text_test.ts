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

Deno.test("資産パーサ: 素の JSON から資産表を組み、id / スコアを行番号で対応付ける", () => {
  const parsed = parseIrodoriTokenizerAsset({
    vocabText: "<unk>\n<s>\n<pad>\n▁こんにちは\n世界",
    scores: [0, 0, 0, -1.5, -2.5],
    addedTokens: [["<unk>", 0], ["<s>", 1], ["<pad>", 2]],
    bosId: 1,
    padId: 2,
    unkId: 0,
    byteBaseId: 3,
  });
  assertEquals(parsed.vocab.get("世界"), { id: 4, score: -2.5 });
  assertEquals(parsed.minScore, -2.5);
  // 探索幅はコードポイント数（`▁こんにちは` = 6 — UTF-16 単位でもバイト数でもない）。
  assertEquals(parsed.maxTokenLength, 6);
  assertEquals(parsed.bosId, 1);
  assertEquals([...parsed.addedTokens.keys()], ["<unk>", "<s>", "<pad>"]);
});

Deno.test("資産パーサ: 行数不一致と必須キー欠落は fail loudly", () => {
  const base = {
    vocabText: "<unk>\n<s>\n<pad>",
    scores: [0, 0, 0],
    addedTokens: [["<unk>", 0]],
    bosId: 1,
    padId: 2,
    unkId: 0,
    // 3 行の語彙なので特殊 id は 0..2（範囲そのものは下の専用テストが見る）。
    byteBaseId: 2,
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
  const base = {
    vocabText: "<unk>\n<s>\n<pad>",
    scores: [0, 0, 0],
    addedTokens: [["<unk>", 0]],
    bosId: 1,
    padId: 2,
    unkId: 0,
    byteBaseId: 0,
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
  // 語彙の行数ちょうど = 1 つ外側。
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, bosId: 3 }),
    Error,
    "tokenizer.bosId: トークン id 3 が語彙の行数 3 以上",
  );
  assertThrows(
    () => parseIrodoriTokenizerAsset({ ...base, addedTokens: [["<unk>", 0.5]] }),
    Error,
    "トークン id が 0..2147483647 の整数でない（0.5）",
  );
});

// ---- 境界（golden のケースが実際に叩いているもの）----------------------------

Deno.test("対にならないサロゲートは受け付けない（正本に無い入力を黙って通さない）", () => {
  assertThrows(() => tokenizer.encode("a\ud800b"), Error, "サロゲート");
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

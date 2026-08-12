// Anima プロンプト層（src/anima/text/）の Python 正本とのパリティ検証。
//
// フィクスチャ `anima-text/parity.json` は exporter が生成する。中身は
//  ・全ケースぶんの参照 id 列（Qwen2 / T5）と T5 正規化後の文字列。**正本は transformers の
//    AutoTokenizer**（パイプライン正本と同じ呼び方）
//  ・その再現に要る語彙・merges の**部分集合**（ライセンス物を丸ごと commit しない）
//  ・正規化表・文字クラス表・NFC 分節表の**全体**（畳み込みそのものが検証対象なので削らない）
//  ・NFC の `[入力, 正本の出力]` 対（素の `normalize("NFC")` では再現できない実測値）
//
// 主に縛るのは振る舞い 1 つ: 「同じ文字列を入れたら Python と同じ id 列が出る」。表の持ち方は
// 縛らない（実装を変えても意味が変わらなければ緑のまま）。**例外は数本の直叩き**で、
// そこは id 列に出ない規則（故障注入で実測）を分割・正規化そのもので固定している。
//
// NOTE: フィクスチャの置き場は暫定で `packages/runtime/tests/fixtures/`（exporter のテストが
// 参照している）。models 側へ移すのは別タスク — ここは相対参照で読む。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseCodeRanges } from "../src/anima/text/code-ranges.ts";
import {
  normalizeNfc,
  type Qwen2Assets,
  qwen2PreTokenize,
  Qwen2Tokenizer,
} from "../src/anima/text/qwen2-tokenizer.ts";
import { normalizeSpm, parseSpmTables } from "../src/anima/text/spm-normalizer.ts";
import {
  type T5Assets,
  t5PreTokenize,
  T5Tokenizer,
  type T5VocabEntry,
} from "../src/anima/text/t5-tokenizer.ts";
import {
  AnimaTokenizers,
  assertPromptTokenLengths,
  createTokenizers,
  PROMPT_MIN_TOKENS,
} from "../src/anima/text/tokenizer.ts";
import { splitAddedTokens } from "../src/text/added-tokens.ts";
import { toCodePoints } from "../src/text/code-points.ts";

type FixtureCase = {
  readonly id: string;
  readonly why: string;
  readonly text: string;
  readonly qwenIds: number[];
  readonly t5Ids: number[];
  readonly t5Normalized: string;
};

type Fixture = {
  readonly maxLength: number;
  readonly qwen: {
    readonly addedTokens: [string, number][];
    readonly classes: { letter: unknown; number: unknown; space: unknown };
    readonly caseFold: [number, number][];
    readonly nfcSegments: unknown;
    readonly nfcCases: [string, string][];
    readonly vocab: Record<string, number>;
    readonly merges: [string, string, number][];
  };
  readonly t5: {
    readonly addedTokens: [string, number][];
    readonly unkId: number;
    readonly eosId: number;
    readonly minScore: number;
    readonly maxTokenLength: number;
    readonly space: unknown;
    readonly normalizer: unknown;
    readonly vocab: [string, number, number][];
  };
  readonly cases: FixtureCase[];
};

const FIXTURE_PATH = new URL(
  "../../runtime/tests/fixtures/anima-text/parity.json",
  import.meta.url,
);
const fixture = JSON.parse(await Deno.readTextFile(FIXTURE_PATH)) as Fixture;

const qwenAssets: Qwen2Assets = {
  vocab: new Map(Object.entries(fixture.qwen.vocab)),
  merges: new Map(fixture.qwen.merges.map(([left, right, rank]) => [`${left} ${right}`, rank])),
  addedTokens: new Map(fixture.qwen.addedTokens),
  caseFold: new Map(fixture.qwen.caseFold),
  nfcSegments: parseCodeRanges(fixture.qwen.nfcSegments, "nfcSegments"),
  classes: {
    letter: parseCodeRanges(fixture.qwen.classes.letter, "letter"),
    number: parseCodeRanges(fixture.qwen.classes.number, "number"),
    space: parseCodeRanges(fixture.qwen.classes.space, "space"),
  },
  maxLength: fixture.maxLength,
};

const t5Assets: T5Assets = {
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
  normalizer: parseSpmTables(fixture.t5.normalizer, "normalizer"),
  maxLength: fixture.maxLength,
};

const qwen = new Qwen2Tokenizer(qwenAssets);
const t5 = new T5Tokenizer(t5Assets);

const caseById = (id: string): FixtureCase => {
  const found = fixture.cases.find((entry) => entry.id === id);
  if (found === undefined) throw new Error(`フィクスチャにケース ${id} が無い`);
  return found;
};

// ---- フィクスチャ本体 -------------------------------------------------------

Deno.test("フィクスチャが空でない（取り違えで全ケース素通しになっていない）", () => {
  // 1 本目に置く。読み違い / 生成失敗で cases が空になると、以降のループが 0 回になって
  // 「緑だが何も検証していない」状態が黙って成立する。
  assert(fixture.cases.length >= 28, `ケース数 ${fixture.cases.length} が少なすぎる`);
  assertEquals(fixture.maxLength, 512);
  assert(Object.keys(fixture.qwen.vocab).length > 0, "Qwen2 語彙の部分集合が空");
  assert(fixture.t5.vocab.length > 0, "T5 語彙の部分集合が空");
});

for (const testCase of fixture.cases) {
  Deno.test(`パリティ [${testCase.id}] ${testCase.why}`, () => {
    assertEquals(qwen.encode(testCase.text), testCase.qwenIds, "Qwen2 id 列");
    assertEquals(t5.encode(testCase.text), testCase.t5Ids, "T5 id 列");
  });
}

Deno.test("T5 の正規化結果そのものが正本と一致する（id 列の手前で切り分ける）", () => {
  for (const testCase of fixture.cases) {
    assertEquals(
      normalizeSpm(t5Assets.normalizer, testCase.text),
      testCase.t5Normalized,
      `[${testCase.id}] の正規化後文字列`,
    );
  }
});

// ---- NFC（正本と JS エンジンが割れる領域）-----------------------------------

Deno.test("NFC: 正本と割れる入力を分節表で再現する（素の normalize では作れない値）", () => {
  // フィクスチャの各対は `tokenizers.normalizers.NFC` の実測値。①分節 cp ごとに 1 本
  // （素の NFC では必ず外れる文脈）②乱択 — ②は分節の外側で ICU が Python とずれた場合の網。
  assert(
    fixture.qwen.nfcCases.length >= 200,
    `NFC ケースが ${fixture.qwen.nfcCases.length} 件しかない`,
  );
  for (const [text, want] of fixture.qwen.nfcCases) {
    assertEquals(
      normalizeNfc(text, qwenAssets.nfcSegments),
      want,
      `NFC [${[...text].map((ch) => (ch.codePointAt(0) as number).toString(16)).join(" ")}]`,
    );
  }
});

Deno.test("NFC: 分節表を 1 区間落とすと素の normalize と同じ（= 別 id 列）に戻る", () => {
  // 実測（`tokenizers` の NFC に同じ文字列を投げた結果）: 正本は U+089A を結合クラス 0 と
  // 見なすので**並べ替えない**。ICU（V8）は ccc 220 < 230 で並べ替える。
  const text = "\u0818\u089A";
  assertEquals(normalizeNfc(text, qwenAssets.nfcSegments), text, "分節表あり = 正本");
  assertEquals(text.normalize("NFC"), "\u089A\u0818", "素の NFC は並べ替える");
  const dropped = qwenAssets.nfcSegments.filter(([start, end]) =>
    !(start <= 0x089a && 0x089a <= end)
  );
  assertEquals(normalizeNfc(text, dropped), "\u089A\u0818", "表から落ちると正本と割れる");
});

// ---- 境界（フィクスチャのケースが実際に叩いているもの）----------------------

Deno.test("512 を超える入力は切り詰められ、T5 は必ず `</s>` で終わる", () => {
  const long = caseById("long");
  assertEquals(long.qwenIds.length, fixture.maxLength);
  const ids = t5.encode(long.text);
  // 切り詰めが `</s>` の**前**に効く（後に足すと 513 個になる）。
  assertEquals(ids.length, fixture.maxLength);
  assertEquals(ids.at(-1), fixture.t5.eosId);
});

Deno.test("対にならないサロゲートは受け付けない（正本に無い入力を黙って通さない）", () => {
  assertThrows(() => toCodePoints("a\ud800b"), Error, "サロゲート");
  assertThrows(() => qwen.encode("a\ud800b"), Error, "サロゲート");
  assertThrows(() => t5.encode("a\ud800b"), Error, "サロゲート");
});

Deno.test("追加語彙は leftmost-longest（途中まで一致する文字列は切り出さない）", () => {
  const added = ["<|endoftext|>", "<|im_start|>"];
  assertEquals(splitAddedTokens("a<|endoftext|>b", added), [
    { text: "a", added: false },
    { text: "<|endoftext|>", added: true },
    { text: "b", added: false },
  ]);
  // 途中まで一致（`<|endoftext|x`）は 1 件も切り出さない。
  assertEquals(splitAddedTokens("<|endoftext|x", added), [
    { text: "<|endoftext|x", added: false },
  ]);
  // 位置が同じなら長い方が勝つ（短い候補を先に置いても結果が変わらない）。
  assertEquals(splitAddedTokens("<|a|>", ["<|a|>", "<|a"]), [{ text: "<|a|>", added: true }]);
  // 空文字の追加語彙は全位置で一致して走査が進まない（無限ループの芽）。
  assertThrows(() => splitAddedTokens("abc", ["", "<|a"]), Error, "空文字");
});

Deno.test("区間表: 昇順・非重複でない表は受け付けない（黙って別の分類にしない）", () => {
  // `inCodeRanges` は二分探索なので、順序が崩れた表は例外にならず判定だけが変わる。
  assertEquals(parseCodeRanges([[1, 2], [4, 9]], "表"), [[1, 2], [4, 9]]);
  assertThrows(() => parseCodeRanges([[5, 1]], "表"), Error, "始端が終端より大きい");
  assertThrows(() => parseCodeRanges([[4, 6], [1, 2]], "表"), Error, "昇順でない");
  assertThrows(() => parseCodeRanges([[1, 5], [5, 9]], "表"), Error, "重なる");
  assertThrows(() => parseCodeRanges([[1, 2], [4]], "表"), Error, "数値対でない");
});

Deno.test("正規化: 結合文字クラスタは最短の一致接頭辞だけを出して残りを捨てる", () => {
  const normalize = (text: string): string => normalizeSpm(t5Assets.normalizer, text);
  // A + U+0301 は規則があるので U+00C1。さらに U+0301 を足しても U+00C1 のまま
  // （3 文字目は捨てられる）。**合成済み文字を書かない** — 入力は必ず分解形で綴る。
  assertEquals(normalize("A\u0301"), "\u00C1");
  assertEquals(normalize("A\u0301\u0301"), "\u00C1");
  // 3cp 規則 U+1EA4 ではなく 2cp 接頭辞 U+00C2 が勝つ（「最長一致」で実装するとここで外れる）。
  assertEquals(normalize("A\u0302\u0301"), "\u00C2");
});

Deno.test("正規化: クラスタが 6 バイト以上になると丸ごと置換の経路に入らない", () => {
  // A(1) + U+0302(2) + U+0301(2) + U+0301(2) = 7 バイト → 1 文字ずつ（何も合成されない）。
  assertEquals(
    normalizeSpm(t5Assets.normalizer, "A\u0302\u0301\u0301"),
    "A\u0302\u0301\u0301",
  );
});

Deno.test("正規化: クラスタがちょうど 6 バイトでも丸ごと置換の経路に入らない（境界そのもの）", () => {
  // U+00A0(2) + U+0300(2) + U+0300(2) = **ちょうど 6 バイト**。`<` を `<=` に書き換えると
  // 「U+00A0 → 空白」の 1cp 規則が丸ごと置換として発火し、結合文字 2 個が黙って消える。
  // 既存の直叩き 2 本は 5 バイトと 7 バイトしか持たず、この 1 文字の書き換えを素通しする
  // （28 ケースのパリティも同様 — 故障注入で実測）。
  assertEquals(
    normalizeSpm(t5Assets.normalizer, "\u00A0\u0300\u0300"),
    " \u0300\u0300",
  );
});

Deno.test("正規化: 制御文字の直後ではクラスタが切れる（結合文字を巻き込まない）", () => {
  const normalize = (text: string): string => normalizeSpm(t5Assets.normalizer, text);
  // U+0009 は空白へ写る。直後の U+0301 は別クラスタなのでそのまま残る。
  assertEquals(normalize("\t\u0301"), " \u0301");
  // CR LF は 1 クラスタ（GB3）なので空白 1 つに畳まれる。
  assertEquals(normalize("\r\n"), " ");
  assertEquals(normalize("\n\n"), "  ");
});

// ---- フィクスチャのケースが**到達しない**規則（合成入力で直接固定する）--------
//
// 下の 4 本は、故障注入で「フィクスチャの id 列だけでは検出できない」と実測した規則
// （① `(?i:)` の同一視 ② `\p{N}` の 1 文字切り ③ Viterbi の同点処理 ④ Metaspace の
// MergedWithNext）。①②は分割が変わっても byte-level BPE が結合し直して同じ id 列になり、
// ③④は実プロンプトでは踏まない。どれも正本に合わせてある規則なので、合成入力で直接叩かないと
// 書き換えが静かに通る。

Deno.test("pre-token: `(?i:)` の同一視は焼いた表で引く（ASCII 相当ではない）", () => {
  // 実測（`tokenizers` の Split に同じ文字列を投げた結果）: 正本は `it'ſs` を
  // `it` / `'ſ` / `s` に切る — Rust の `(?i:)` は simple case folding で U+017F ≡ s。
  // **id 列では捕まらない**（byte-level BPE が結合し直して同じ列になる）ので、分割そのものを
  // 見る。ASCII の大小反転や `toLowerCase()` はここで `'ſs` の 1 断片になって落ちる。
  const split = (text: string): string[] =>
    qwen2PreTokenize(toCodePoints(text), qwenAssets.classes, qwenAssets.caseFold);
  assertEquals(split("it'ſs"), ["it", "'ſ", "s"]);
  assertEquals(split("it'S"), ["it", "'S"]);
  assertEquals(split("it's"), ["it", "'s"]);
  // 表そのものにも U+017F → s が入っていること（落とすと上の期待値が作れない）。
  assertEquals(qwenAssets.caseFold.get(0x017f), "s".codePointAt(0));
});

Deno.test("pre-token: `\\p{N}` は 1 文字ずつ切れる（`+` が無い）", () => {
  // **id 列では捕まらない**（Qwen2 の merges に数字結合が無いので、貪欲化しても実資産で
  // 同じ列になると故障注入で実測）。分割そのものを見ないと書き換えが静かに通る。
  const split = (text: string): string[] =>
    qwen2PreTokenize(toCodePoints(text), qwenAssets.classes, qwenAssets.caseFold);
  assertEquals(split("2024"), ["2", "0", "2", "4"]);
  assertEquals(split("1.5x"), ["1", ".", "5", "x"]);
});

Deno.test("Viterbi: 同点なら長い断片が勝つ（比較は厳密な `>`）", () => {
  // a(-1) + b(-1) と ab(-2) がちょうど同点。正本（tokenizers の Unigram）を同じ語彙で
  // 実測すると `ab` 1 個になる（同じ位置で終わるノードは開始位置の昇順に積まれ、同点なら
  // 先に積まれた = 長い方が残る）。`>=` にすると a, b の 2 個へ黙って変わる。
  const tie = new T5Tokenizer({
    vocab: new Map<string, T5VocabEntry>([
      ["▁", { id: 3, score: -1 }],
      ["a", { id: 4, score: -1 }],
      ["b", { id: 5, score: -1 }],
      ["ab", { id: 6, score: -2 }],
    ]),
    minScore: -2,
    maxTokenLength: 2,
    unkId: 2,
    eosId: 1,
    addedTokens: new Map(),
    space: [[0x20, 0x20]],
    normalizer: parseSpmTables(
      { single: [], multi: [], extend: [], breakAfter: [], prepend: [] },
      "空",
    ),
    maxLength: 512,
  });
  assertEquals(tie.encode("ab"), [3, 6, 1], "▁ + ab + </s>");
});

Deno.test("Metaspace: 区切りの ▁ は次の断片の先頭に付く（MergedWithNext）", () => {
  const space: readonly (readonly [number, number])[] = [[0x20, 0x20]];
  // 先頭には必ず ▁ が付き（prepend_scheme=always）、入力中の ▁ は**その位置で切って次へ**。
  assertEquals(t5PreTokenize("under▁score", space), ["▁under", "▁score"]);
  assertEquals(t5PreTokenize("abc", space), ["▁abc"]);
  assertEquals(t5PreTokenize("a b", space), ["▁a", "▁b"]);
  // 連続する ▁ は空断片を作らない。
  assertEquals(t5PreTokenize("a▁▁b", space), ["▁a", "▁", "▁b"]);
});

// ---- 受理集合（Dim(min=2)）--------------------------------------------------

Deno.test("受理集合: 空文字・空白だけのプロンプトは fail loudly", () => {
  // T5 は空文字でも `</s>` だけの長さ 1 になり Dim("Ttgt", min=2) を外れる。
  assertEquals(t5.encode(""), [fixture.t5.eosId]);
  assertEquals(caseById("only_space").t5Ids.length, 1);
  assertThrows(
    () => assertPromptTokenLengths("ネガティブプロンプト", 0, 1, fixture.maxLength),
    Error,
    "ネガティブプロンプト",
  );
});

Deno.test("受理集合: 1 文字のプロンプトは Qwen2 側が 1 トークンで外れる", () => {
  const single = caseById("single_char");
  assertEquals(single.qwenIds.length, 1);
  // T5 側は `▁a` + `</s>` などで 2 以上になり得るので、**両側を見ないと**捕まらない。
  assert(single.t5Ids.length >= PROMPT_MIN_TOKENS);
  assertThrows(
    () =>
      assertPromptTokenLengths(
        "プロンプト",
        single.qwenIds.length,
        single.t5Ids.length,
        fixture.maxLength,
      ),
    Error,
    "Qwen2",
  );
});

Deno.test("受理集合: 通常のタグ列は受理し、上限超過は拒否する", () => {
  const tags = caseById("tags");
  assertPromptTokenLengths("プロンプト", tags.qwenIds.length, tags.t5Ids.length, fixture.maxLength);
  assertThrows(
    () => assertPromptTokenLengths("プロンプト", fixture.maxLength + 1, 4, fixture.maxLength),
    Error,
    "上限",
  );
});

// ---- 2 本組の入口（AnimaTokenizers / createTokenizers）------------------------

Deno.test("AnimaTokenizers: 片方だけ古い資産（maxLength の食い違い）を落とす", () => {
  // 切り詰め長が違うと長いプロンプトで id 列の長さだけが食い違い、conditioner の
  // 512 パディング検査まで表面化しない。
  assertThrows(
    () => new AnimaTokenizers(qwenAssets, { ...t5Assets, maxLength: 256 }),
    Error,
    "maxLength が食い違う",
  );
});

Deno.test("AnimaTokenizers: 正 / ネガティブのどちらで落ちたかがメッセージに出る", () => {
  const tokenizers = new AnimaTokenizers(qwenAssets, t5Assets);
  assertEquals(tokenizers.maxLength, fixture.maxLength);
  const tags = caseById("tags");
  const ids = tokenizers.encode(tags.text);
  assertEquals([...ids.qwenIds], tags.qwenIds);
  assertEquals([...ids.t5Ids], tags.t5Ids);
  assertThrows(() => tokenizers.encode("", "ネガティブプロンプト"), Error, "ネガティブプロンプト");
});

Deno.test("createTokenizers: バイト列（manifest の tokenizer / tokenizer_2）から組む", () => {
  // 実資産の JSON 形は「行区切りの語彙 + 件数」なので、**件数の食い違いを落とす**ところまでが
  // 入口の契約（merges が欠けても BPE は落ちず、id 列だけが静かに別物になる）。
  const encode = (value: unknown): Uint8Array => new TextEncoder().encode(JSON.stringify(value));
  const qwen2Json = {
    vocabText: "a\nb\nab",
    vocabCount: 3,
    mergesText: "a b",
    mergesCount: 1,
    addedTokens: [],
    classes: { letter: [[0x61, 0x7a]], number: [], space: [[0x20, 0x20]] },
    caseFold: [],
    nfcSegments: [],
    maxLength: 8,
  };
  const t5Json = {
    vocabText: "<pad>\n</s>\n<unk>\n▁\na\nb",
    scores: [0, 0, 0, -1, -1, -2],
    unkId: 2,
    eosId: 1,
    addedTokens: [],
    space: [[0x20, 0x20]],
    normalizer: { single: [], multi: [], extend: [], breakAfter: [], prepend: [] },
    maxLength: 8,
  };
  const tokenizers = createTokenizers(encode(qwen2Json), encode(t5Json));
  assertEquals(tokenizers.maxLength, 8);
  // `a b` の merge が rank 0 で効くので `aba` は `ab` + `a` に割れる（vocab は a=0 / b=1 / ab=2）。
  // **2 トークン以上**にするのは受理集合 Dim(min=2) の内側に入れるため。
  assertEquals([...tokenizers.encode("aba").qwenIds], [2, 0], "merges で ab に結合される");

  assertThrows(
    () => createTokenizers(encode({ ...qwen2Json, mergesCount: 2 }), encode(t5Json)),
    Error,
    "行数",
  );
  assertThrows(
    () => createTokenizers(new TextEncoder().encode("{"), encode(t5Json)),
    Error,
    "JSON として読めない",
  );
});

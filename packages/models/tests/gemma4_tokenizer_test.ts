// Gemma 系トークナイザ（src/gemma/text/）の上流パリティ検証。
//
// フィクスチャ `gemma-text/{gemma4,embeddinggemma}-parity.json` は export recipe
// （`tools/export-recipes/{gemma4,embeddinggemma}/tokenizer.py`）が生成する。期待値は
// **上流の `tokenizers.Tokenizer` を独立に呼んだ**もので、compile 台本の畳み込みも Python 側の
// 前処理も通っていない（ADR 0084 決定 7 — 共有すると parity が恒真化する。ADR 0048 追記の実例）。
//
// 中身は
//  ・符号化ケースの `{text, ids, idsWithSpecials}`（英・日本語長文・byte fallback・特殊トークン）
//  ・復号ケースの `{ids, text, textSkipSpecials}`（符号化の往復 + 直接叩く byte run）
//  ・その再現に要る語彙と併合規則の**部分集合**（262,144 行 / 514,906 本を commit しない。
//    部分集合が足りなければ期待値と食い違って落ちるので、絞り方の誤りは危険側に倒れない）
//
// **同じ電池を 2 資産に流す**のが眼目（ADR 0084 決定 6 = 1 実装 2 資産）。実装は共用でも資産は
// 共用できない: merges はビット同一でも語彙は 6,206 スロット違い、追加語彙は 24 本 vs 6,415 本、
// post_processor も別（gemma4 = 素通し / EmbeddingGemma = `<bos>` … `<eos>`）。
//
// 実資産が在れば**全語彙**でも同じ id 列になることを併せて見る（部分集合と full の食い違いを
// 塞ぐ門）。The gemma4 asset is the distribution's tokenizer (`models/karume-gemma4/`, read through
// the manifest — ADR 0139 decision 1); EmbeddingGemma has no published distribution yet, so its
// full-vocabulary cases read the series (`outputs/series/embeddinggemma-300m-tokenizer/`) as
// implementation-time checks (decision 6 — KARUME_SERIES=1). 無い環境では SKIP する。GPU は使わない。
//
// NOTE: EmbeddingGemma gets a published distribution (ADR 0139 Rulings D2, staging step 7); its
// full-vocabulary cases then read that distribution's tokenizer like gemma4's and return to the
// default lane.

import { readTextFile } from "@karume/test-support/fs";
import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { createBpeModel } from "../src/text/bpe.ts";
import { type GemmaTokenizerAssets, parseGemmaTokenizerAsset } from "../src/gemma/text/asset.ts";
import { Tokenizer } from "../src/gemma/text/tokenizer.ts";
import { readFileIfPresent } from "./helpers/read-if-present.ts";
import { mirrorHasModel, mirrorPresent, mirrorRoot, readMirrorAsset } from "./helpers/mirror.ts";
import {
  GEMMA4_12B_IN_MIRROR,
  GEMMA4_QAT_MIRROR,
  GEMMA4_12B_TOKENIZER_SERIES,
  readGemma4_12bTokenizer,
} from "./helpers/gemma-mirror.ts";
import { SERIES_CHECKS, SERIES_CHECKS_OPT_IN } from "./helpers/series-checks.ts";

type EncodeCase = {
  readonly name: string;
  readonly why: string;
  readonly text: string;
  readonly ids: number[];
  readonly idsWithSpecials: number[];
};

type DecodeCase = {
  readonly name: string;
  readonly why: string;
  readonly ids: number[];
  readonly text: string;
  readonly textSkipSpecials: string;
};

type Fixture = {
  readonly spec: {
    readonly normalizer: string;
    readonly preTokenizer: string;
    readonly decoder: string;
    readonly postProcessor: "none" | "bos-eos";
  };
  readonly asset: {
    readonly vocab: [number, string][];
    readonly merges: [number, number, number][];
    readonly byteIds: number[];
    readonly addedTokens: [string, number][];
    readonly specialIds: number[];
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;
  };
  readonly encode: EncodeCase[];
  readonly decode: DecodeCase[];
};

/** フィクスチャの部分集合から資産表を直接組む（JSON パーサの門は別テストが見る）。 */
const assetsOf = (fixture: Fixture): GemmaTokenizerAssets => ({
  spec: { postProcessor: fixture.spec.postProcessor },
  model: createBpeModel({
    vocab: fixture.asset.vocab.map(([id, token]) => [id, token] as const),
    merges: fixture.asset.merges.map(([left, right, rank]) => [left, right, rank] as const),
    byteIds: fixture.asset.byteIds,
  }),
  addedTokens: new Map(fixture.asset.addedTokens),
  specialIds: new Set(fixture.asset.specialIds),
  unkId: fixture.asset.unkId,
  bosId: fixture.asset.bosId,
  eosId: fixture.asset.eosId,
});

const readFixture = async (name: string): Promise<Fixture> =>
  JSON.parse(
    await readTextFile(new URL(`fixtures/gemma-text/${name}-parity.json`, import.meta.url)),
  ) as Fixture;

/** 系列側の実資産（無ければ undefined = compile していない）。 */
const readSeriesAsset = async (series: string): Promise<Uint8Array<ArrayBuffer> | undefined> =>
  await readFileIfPresent(
    new URL(`../../../outputs/series/${series}/tokenizer.json`, import.meta.url),
  );

const GEMMA4_MIRROR = mirrorRoot("karume-gemma4");

/**
 * Where a family's full-vocabulary asset comes from: `read` gives `undefined` when it is absent
 * (the real-asset cases then SKIP), `absent` is the warning naming what is missing (none when the
 * cases are opt-in and not requested), and `suffix` marks implementation-time cases.
 */
type RealAsset = {
  readonly read: () => Promise<Uint8Array<ArrayBuffer> | undefined>;
  readonly absent?: string;
  readonly suffix: string;
};

const GEMMA4_ASSET: RealAsset = {
  // E2B's tokenizer. Every model has its own since Google's glitch-token fixes (E2B drops 267 BPE
  // merges, E4B 253 — {@link GEMMA4_E4B_ASSET}; 12B 239 — {@link GEMMA4_12B_ASSET}); the
  // vocabulary is the same. e2b is named so the selection is fixed.
  read: async () =>
    mirrorPresent(GEMMA4_MIRROR)
      ? await readMirrorAsset(GEMMA4_MIRROR, "tokenizer", { model: "e2b" })
      : undefined,
  absent:
    `[karume] the distribution mirror ${GEMMA4_MIRROR.pathname} is absent (or not karume/5), so ` +
    "the gemma4 full-vocabulary tokenizer cases SKIP. Assemble: cd tools/export-recipes && uv run " +
    "python dist.py --pipeline gemma4",
  suffix: "",
};

const GEMMA4_E4B_ASSET: RealAsset = {
  read: async () =>
    mirrorHasModel(GEMMA4_MIRROR, "e4b")
      ? await readMirrorAsset(GEMMA4_MIRROR, "tokenizer", { model: "e4b" })
      : undefined,
  absent:
    `[karume] the distribution mirror ${GEMMA4_MIRROR.pathname} has no model e4b, so the gemma4 ` +
    "E4B full-vocabulary tokenizer cases SKIP. Assemble: cd tools/export-recipes && uv run python " +
    "dist.py --pipeline gemma4 --model e2b --model e4b --out ../../models/karume-gemma4",
  suffix: "",
};

const GEMMA4_12B_ASSET: RealAsset = {
  // The distribution's asset when the mirror carries model 12b; before the mirror is assembled with
  // it, the series is read as an implementation-time check (KARUME_SERIES=1 — ADR 0139 decision 6).
  read: readGemma4_12bTokenizer,
  absent:
    `[karume] the distribution mirror ${GEMMA4_QAT_MIRROR.pathname} has no model 12b` +
    (SERIES_CHECKS
      ? ` and outputs/series/${GEMMA4_12B_TOKENIZER_SERIES}/tokenizer.json is absent`
      : "") +
    ", so the gemma4 12B full-vocabulary tokenizer cases SKIP. Assemble: cd tools/export-recipes" +
    " && uv run python dist.py --pipeline gemma4-qat",
  suffix: GEMMA4_12B_IN_MIRROR ? "" : SERIES_CHECKS_OPT_IN,
};

const EMBEDDINGGEMMA_SERIES = "embeddinggemma-300m-tokenizer";
const EMBEDDINGGEMMA_ASSET: RealAsset = {
  // Without KARUME_SERIES=1 the series is not read at all.
  read: async () => (SERIES_CHECKS ? await readSeriesAsset(EMBEDDINGGEMMA_SERIES) : undefined),
  ...(SERIES_CHECKS
    ? {
        absent:
          `[karume] outputs/series/${EMBEDDINGGEMMA_SERIES}/tokenizer.json が無いため ` +
          "EmbeddingGemma の実資産突合を SKIP する。生成: cd tools/export-recipes && uv run " +
          "python -m embeddinggemma.tokenizer",
      }
    : {}),
  suffix: SERIES_CHECKS_OPT_IN,
};

/** 決定的な擬似乱数（chunk 分割の抽選を毎回同じにする）。 */
const nextRandom = (state: number): { value: number; state: number } => {
  // Math.imul keeps the product exact modulo 2^32; a plain multiplication passes 2^53 and loses
  // the low bits (the sequence then falls into a cycle of about 10^4 draws).
  const next = (Math.imul(state, 1103515245) + 12345) & 0x7fffffff;
  return { value: next / 0x7fffffff, state: next };
};

/** 列の分け方を列挙する（短ければ全 partition・長ければ決まった本数を抽選）。 */
const partitionsOf = (length: number, samples: number): number[][][] => {
  const build = (cuts: readonly boolean[]): number[][] => {
    const chunks: number[][] = [[]];
    for (let index = 0; index < length; index++) {
      if (index > 0 && cuts[index - 1]) chunks.push([]);
      chunks[chunks.length - 1].push(index);
    }
    return chunks;
  };
  if (length <= 12) {
    const out: number[][][] = [];
    for (let mask = 0; mask < 1 << Math.max(0, length - 1); mask++) {
      out.push(build(Array.from({ length: length - 1 }, (_, bit) => ((mask >> bit) & 1) === 1)));
    }
    return out;
  }
  let state = 20260831;
  return Array.from({ length: samples }, () => {
    const cuts = Array.from({ length: length - 1 }, () => {
      const drawn = nextRandom(state);
      state = drawn.state;
      return drawn.value < 0.3;
    });
    return build(cuts);
  });
};

/** 資産 JSON の門を叩くための最小形（実資産と同じ欄構成・語彙は 256 + 3 本）。 */
const minimalAsset = (): Record<string, unknown> => {
  const byteToken = (byte: number): string =>
    `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
  const vocab = [
    "<unk>",
    "<bos>",
    "<eos>",
    ...Array.from({ length: 256 }, (_, byte) => byteToken(byte)),
    "a",
    "aa",
  ];
  return {
    format: "karume-gemma-tokenizer/1",
    spec: {
      normalizer: "replace-space-with-metaspace",
      preTokenizer: "split-space-merged-with-previous",
      decoder: "metaspace-byte-fallback-fuse",
      postProcessor: "none",
    },
    vocab,
    mergesText: `${vocab.indexOf("a")} ${vocab.indexOf("a")}`,
    mergesCount: 1,
    byteIds: Array.from({ length: 256 }, (_, byte) => vocab.indexOf(byteToken(byte))),
    addedTokens: [
      ["<unk>", 0],
      ["<bos>", 1],
      ["<eos>", 2],
    ],
    specialIds: [0, 1, 2],
    unkId: 0,
    bosId: 1,
    eosId: 2,
  };
};

const parseAsset = (asset: Record<string, unknown>): void => {
  parseGemmaTokenizerAsset(new TextEncoder().encode(JSON.stringify(asset)));
};

describe("資産 JSON の門: 信用できない外部入力として扱う", () => {
  it("最小形は通る", () => {
    parseAsset(minimalAsset());
  });

  it("知らない format 版は読まない", () => {
    assertThrows(() => parseAsset({ ...minimalAsset(), format: "karume-gemma-tokenizer/2" }));
  });

  it("出所記録 source は読み流す（書き手が必ず綴る欄）", () => {
    parseAsset({ ...minimalAsset(), source: { path: "synthetic" } });
  });

  it("未知キーは落ちる（同版内の欄の追加・綴り違いを黙って読み流さない）", () => {
    assertThrows(
      () => parseAsset({ ...minimalAsset(), mergeCount: 1 }),
      Error,
      "unknown key 'mergeCount'",
    );
    const asset = minimalAsset();
    asset["spec"] = { ...(asset["spec"] as Record<string, unknown>), extra: "x" };
    assertThrows(() => parseAsset(asset), Error, "tokenizer.spec: unknown key 'extra'");
  });

  it("欄が欠ければ「無い」で落ちる", () => {
    const { mergesCount: _, ...withoutCount } = minimalAsset();
    assertThrows(() => parseAsset(withoutCount), Error, "tokenizer.mergesCount: missing");
    const asset = minimalAsset();
    const { decoder: __, ...specWithoutDecoder } = asset["spec"] as Record<string, unknown>;
    asset["spec"] = specWithoutDecoder;
    assertThrows(() => parseAsset(asset), Error, "tokenizer.spec.decoder: missing");
  });

  it("宣言された構成が 1 欄でも違えば落ちる", () => {
    for (const field of ["normalizer", "preTokenizer", "decoder"]) {
      const asset = minimalAsset();
      asset["spec"] = { ...(asset["spec"] as Record<string, unknown>), [field]: "別物" };
      assertThrows(() => parseAsset(asset), Error, field);
    }
  });

  it("未知の post_processor は落ちる", () => {
    const asset = minimalAsset();
    asset["spec"] = { ...(asset["spec"] as Record<string, unknown>), postProcessor: "bos-only" };
    assertThrows(() => parseAsset(asset), Error, "postProcessor");
  });

  it("merges の行数が申告と違えば落ちる", () => {
    // 欠けても BPE は「見つからないので結合しない」だけで落ちない（分割だけが静かに変わる）。
    assertThrows(() => parseAsset({ ...minimalAsset(), mergesCount: 2 }), Error, "line count");
  });

  it("特殊トークンが追加語彙の外なら落ちる", () => {
    assertThrows(
      () => parseAsset({ ...minimalAsset(), specialIds: [0, 1, 2, 5] }),
      Error,
      "added vocabulary",
    );
  });

  it("追加語彙の id 重複は落ちる（綴り→id と id→綴りが食い違う）", () => {
    // 同じ id を 2 つの綴りが指すと、encode は両方をその id にし、decode は片方しか返せない。
    // 両方とも語彙の外（300）に置く — 語彙内の id だと綴り不一致の検査が先に落とし、id 重複の
    // 検査を外しても赤にならない。
    assertThrows(
      () =>
        parseAsset({
          ...minimalAsset(),
          addedTokens: [
            ["<unk>", 0],
            ["<bos>", 1],
            ["<eos>", 2],
            ["<x>", 300],
            ["<y>", 300],
          ],
        }),
      Error,
      "tokenizer.addedTokens: id 300 is assigned to two spellings",
    );
  });

  it("語彙内の id を綴りの違う追加語彙が使えば落ちる", () => {
    // `["ZZ", 259]`（語彙では "a"）を受けると encode("a") が 259 を返し、decode(259) が "ZZ" になる。
    assertThrows(
      () =>
        parseAsset({
          ...minimalAsset(),
          addedTokens: [
            ["<unk>", 0],
            ["<bos>", 1],
            ["<eos>", 2],
            ["ZZ", 259],
          ],
        }),
      Error,
      "id 259",
    );
  });

  it("語彙の外への採番は従来どおり受ける（EmbeddingGemma の形）", () => {
    parseAsset({
      ...minimalAsset(),
      addedTokens: [
        ["<unk>", 0],
        ["<bos>", 1],
        ["<eos>", 2],
        ["<image_soft_token>", 261],
      ],
    });
  });

  it("`<bos>` / `<eos>` / `<unk>` が語彙の外なら落ちる", () => {
    assertThrows(
      () => parseAsset({ ...minimalAsset(), bosId: 9999 }),
      Error,
      "is not in the vocabulary",
    );
  });

  it("JSON でないバイト列は落ちる", () => {
    assertThrows(
      () => parseGemmaTokenizerAsset(new TextEncoder().encode("{")),
      Error,
      "cannot be read as JSON",
    );
  });
});

for (const { family, fixtureName, real } of [
  { family: "gemma4", fixtureName: "gemma4", real: GEMMA4_ASSET },
  { family: "gemma4 E4B", fixtureName: "gemma4-e4b", real: GEMMA4_E4B_ASSET },
  { family: "gemma4 12B", fixtureName: "gemma4-12b", real: GEMMA4_12B_ASSET },
  { family: "EmbeddingGemma", fixtureName: "embeddinggemma", real: EMBEDDINGGEMMA_ASSET },
]) {
  const fixture = await readFixture(fixtureName);
  const tokenizer = new Tokenizer(assetsOf(fixture));

  describe(`${family} 符号化パリティ: 上流 tokenizers と id 列がビット一致`, () => {
    for (const testCase of fixture.encode) {
      it(`${testCase.name} — ${testCase.why}`, () => {
        assertEquals(tokenizer.encode(testCase.text), testCase.ids, "add_special_tokens=False");
        assertEquals(
          tokenizer.encodeWithSpecialTokens(testCase.text),
          testCase.idsWithSpecials,
          "add_special_tokens=True",
        );
      });
    }
  });

  describe(`${family} 復号パリティ: 上流 tokenizers と文字列がビット一致`, () => {
    for (const testCase of fixture.decode) {
      it(`${testCase.name} — ${testCase.why}`, () => {
        assertEquals(
          tokenizer.decode(testCase.ids, { skipSpecialTokens: false }),
          testCase.text,
          "skip_special_tokens=False",
        );
        assertEquals(
          tokenizer.decode(testCase.ids),
          testCase.textSkipSpecials,
          "skip_special_tokens=True（既定）",
        );
      });
    }
  });

  test(`${family} 逐次復号: どう区切って push しても一括復号と一致`, () => {
    // 突き合わせ先は**上流が返した文字列**（自前の一括復号ではない）— 逐次と一括は同じ
    // 状態機械を通るので、自前同士の比較では何も落とせない。
    for (const testCase of fixture.decode) {
      for (const partition of partitionsOf(testCase.ids.length, 32)) {
        const detokenizer = tokenizer.createDetokenizer({ skipSpecialTokens: false });
        let out = "";
        for (const chunk of partition) {
          for (const index of chunk) out += detokenizer.push(testCase.ids[index]);
        }
        out += detokenizer.finish();
        assertEquals(out, testCase.text, `${testCase.name}: 区切り ${partition.length} 個`);
      }
    }
  });

  test(`${family} 資産の宣言: compile が確かめた構成と一致`, () => {
    assertEquals(fixture.spec.normalizer, "replace-space-with-metaspace");
    assertEquals(fixture.spec.preTokenizer, "split-space-merged-with-previous");
    assertEquals(fixture.spec.decoder, "metaspace-byte-fallback-fuse");
    assertEquals(tokenizer.postProcessor, fixture.spec.postProcessor);
  });

  test(`${family} 本文の門: 対にならないサロゲートは入力の誤り（ModelInputError）`, () => {
    // 呼び手が本文を直せば解消する誤りなので、内部障害（素の Error）に見せない（ADR 0107）。
    // chat の描画も encode を通るので、ここ 1 箇所で chat まで覆う。
    for (const text of ["\uD800", "a\uDC00b"]) {
      assertThrows(() => tokenizer.encode(text), ModelInputError, "surrogate");
      assertThrows(() => tokenizer.encodeWithSpecialTokens(text), ModelInputError, "surrogate");
    }
  });

  const fullBytes = await real.read();
  // 実資産 1 本の解釈は 0.6 秒級（併合 514,906 本の組み立て）。門ごとに組み直さない。
  let fullTokenizer: Tokenizer | undefined;
  const full = (): Tokenizer =>
    (fullTokenizer ??= new Tokenizer(
      parseGemmaTokenizerAsset(fullBytes as Uint8Array<ArrayBuffer>),
    ));

  if (fullBytes === undefined && real.absent !== undefined) console.warn(real.absent);

  test.skipIf(fullBytes === undefined)(
    `${family} 実資産突合: 全語彙でもフィクスチャと同じ id 列（部分集合の穴を塞ぐ）${real.suffix}`,
    () => {
      const tokenizerFull = full();
      for (const testCase of fixture.encode) {
        assertEquals(tokenizerFull.encode(testCase.text), testCase.ids, testCase.name);
        assertEquals(
          tokenizerFull.encodeWithSpecialTokens(testCase.text),
          testCase.idsWithSpecials,
          testCase.name,
        );
      }
      for (const testCase of fixture.decode) {
        assertEquals(
          tokenizerFull.decode(testCase.ids, { skipSpecialTokens: false }),
          testCase.text,
          testCase.name,
        );
      }
    },
  );

  test.skipIf(fullBytes === undefined)(
    `${family} 実資産の日本語長文: 併合キューが実用時間で終わる${real.suffix}`,
    () => {
      // 正規化後の断片は入力全長 1 本になるので、50,000 文字は記号 50,000 個の 1 断片。
      // 全走査の実装なら併合（約 20,000 回）ごとに列を舐めて 10^8〜10^9 回規模の探索になる
      // （数十秒〜）。merge queue の実測は 20ms 前後で、予算はその 250 倍を採ってある。
      const source = fixture.encode.find((row) => row.name === "japanese-long") as EncodeCase;
      const long = source.text.repeat(Math.ceil(50_000 / source.text.length));
      const tokenizerFull = full();

      const started = performance.now();
      const ids = tokenizerFull.encode(long);
      const elapsed = performance.now() - started;

      // 1 文字 1 トークンより短くなる（併合が実際に効いている）ことまで見る。
      if (ids.length === 0 || ids.length >= Array.from(long).length) {
        throw new Error(
          `併合が効いていない（${ids.length} トークン / ${Array.from(long).length} 文字）`,
        );
      }
      const BUDGET_MS = 5_000;
      if (elapsed > BUDGET_MS) {
        throw new Error(
          `${Array.from(long).length} 文字の符号化に ${elapsed.toFixed(0)}ms（予算 ${BUDGET_MS}ms）`,
        );
      }
    },
  );
}

// The reason E4B and 12B have tokenizer series of their own: their tokenizer.json drops other BPE
// merges than E2B's, so E2B's asset gives other ids for some inputs. The 12B fixture's
// `dropped-merge*` cases hit merges E2B has and 12B dropped, the E4B fixture's
// `e2b-dropped-merge*` cases merges E4B has and E2B dropped; on E2B's full asset they must tokenize
// differently, or the cases (and the series split they justify) prove nothing.
const e2bFull = await GEMMA4_ASSET.read();
for (const { model, prefix } of [
  { model: "12b", prefix: "dropped-merge" },
  { model: "e4b", prefix: "e2b-dropped-merge" },
]) {
  test.skipIf(e2bFull === undefined)(
    `gemma4 ${model.toUpperCase()} fixture: the ${prefix} cases tokenize differently with the E2B asset`,
    async () => {
      const fixture = await readFixture(`gemma4-${model}`);
      const e2b = new Tokenizer(parseGemmaTokenizerAsset(e2bFull as Uint8Array<ArrayBuffer>));
      const dropped = fixture.encode.filter((row) => row.name.startsWith(prefix));
      assert(dropped.length > 0, `the ${model} fixture has no ${prefix} case`);
      for (const testCase of dropped) {
        assertNotEquals(e2b.encode(testCase.text), testCase.ids, testCase.name);
      }
    },
  );
}

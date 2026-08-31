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
// 実資産（`outputs/series/*-tokenizer/`）が在れば**全語彙**でも同じ id 列になることを併せて
// 見る（部分集合と full の食い違いを塞ぐ門）。無い環境では SKIP する。GPU は使わない。

import { assertEquals, assertThrows } from "@std/assert";
import { createBpeModel } from "../src/text/bpe.ts";
import { type GemmaTokenizerAssets, parseGemmaTokenizerAsset } from "../src/gemma/text/asset.ts";
import { GemmaTokenizer } from "../src/gemma/text/tokenizer.ts";

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
    await Deno.readTextFile(new URL(`fixtures/gemma-text/${name}-parity.json`, import.meta.url)),
  ) as Fixture;

/** 系列側の実資産（無ければ undefined = compile していない）。 */
const readSeriesAsset = async (series: string): Promise<Uint8Array | undefined> =>
  await Deno.readFile(
    new URL(`../../../outputs/series/${series}/tokenizer.json`, import.meta.url),
  ).catch(() => undefined);

/** 決定的な擬似乱数（chunk 分割の抽選を毎回同じにする）。 */
const nextRandom = (state: number): { value: number; state: number } => {
  const next = (state * 1103515245 + 12345) & 0x7fffffff;
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
      out.push(build(Array.from({ length: length - 1 }, (_, bit) => (mask >> bit & 1) === 1)));
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
    addedTokens: [["<unk>", 0], ["<bos>", 1], ["<eos>", 2]],
    specialIds: [0, 1, 2],
    unkId: 0,
    bosId: 1,
    eosId: 2,
  };
};

const parseAsset = (asset: Record<string, unknown>): void => {
  parseGemmaTokenizerAsset(new TextEncoder().encode(JSON.stringify(asset)));
};

Deno.test("資産 JSON の門: 信用できない外部入力として扱う", async (t) => {
  await t.step("最小形は通る", () => {
    parseAsset(minimalAsset());
  });

  await t.step("知らない format 版は読まない", () => {
    assertThrows(() => parseAsset({ ...minimalAsset(), format: "karume-gemma-tokenizer/2" }));
  });

  await t.step("宣言された構成が 1 欄でも違えば落ちる", () => {
    for (const field of ["normalizer", "preTokenizer", "decoder"]) {
      const asset = minimalAsset();
      asset["spec"] = { ...(asset["spec"] as Record<string, unknown>), [field]: "別物" };
      assertThrows(() => parseAsset(asset), Error, field);
    }
  });

  await t.step("未知の post_processor は落ちる", () => {
    const asset = minimalAsset();
    asset["spec"] = { ...(asset["spec"] as Record<string, unknown>), postProcessor: "bos-only" };
    assertThrows(() => parseAsset(asset), Error, "postProcessor");
  });

  await t.step("merges の行数が申告と違えば落ちる", () => {
    // 欠けても BPE は「見つからないので結合しない」だけで落ちない（分割だけが静かに変わる）。
    assertThrows(() => parseAsset({ ...minimalAsset(), mergesCount: 2 }), Error, "行数");
  });

  await t.step("特殊トークンが追加語彙の外なら落ちる", () => {
    assertThrows(
      () => parseAsset({ ...minimalAsset(), specialIds: [0, 1, 2, 5] }),
      Error,
      "追加語彙",
    );
  });

  await t.step("`<bos>` / `<eos>` / `<unk>` が語彙の外なら落ちる", () => {
    assertThrows(() => parseAsset({ ...minimalAsset(), bosId: 9999 }), Error, "語彙に無い");
  });

  await t.step("JSON でないバイト列は落ちる", () => {
    assertThrows(
      () => parseGemmaTokenizerAsset(new TextEncoder().encode("{")),
      Error,
      "JSON として読めない",
    );
  });
});

for (
  const { family, fixtureName, series } of [
    { family: "gemma4", fixtureName: "gemma4", series: "gemma4-e2b-tokenizer" },
    {
      family: "EmbeddingGemma",
      fixtureName: "embeddinggemma",
      series: "embeddinggemma-300m-tokenizer",
    },
  ]
) {
  const fixture = await readFixture(fixtureName);
  const tokenizer = new GemmaTokenizer(assetsOf(fixture));

  Deno.test(`${family} 符号化パリティ: 上流 tokenizers と id 列がビット一致`, async (t) => {
    for (const testCase of fixture.encode) {
      await t.step(`${testCase.name} — ${testCase.why}`, () => {
        assertEquals(tokenizer.encode(testCase.text), testCase.ids, "add_special_tokens=False");
        assertEquals(
          tokenizer.encodeWithSpecialTokens(testCase.text),
          testCase.idsWithSpecials,
          "add_special_tokens=True",
        );
      });
    }
  });

  Deno.test(`${family} 復号パリティ: 上流 tokenizers と文字列がビット一致`, async (t) => {
    for (const testCase of fixture.decode) {
      await t.step(`${testCase.name} — ${testCase.why}`, () => {
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

  Deno.test(`${family} 逐次復号: どう区切って push しても一括復号と一致`, () => {
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

  Deno.test(`${family} 資産の宣言: compile が確かめた構成と一致`, () => {
    assertEquals(fixture.spec.normalizer, "replace-space-with-metaspace");
    assertEquals(fixture.spec.preTokenizer, "split-space-merged-with-previous");
    assertEquals(fixture.spec.decoder, "metaspace-byte-fallback-fuse");
    assertEquals(tokenizer.postProcessor, fixture.spec.postProcessor);
  });

  const seriesBytes = await readSeriesAsset(series);
  // 実資産 1 本の解釈は 0.6 秒級（併合 514,906 本の組み立て）。門ごとに組み直さない。
  let fullTokenizer: GemmaTokenizer | undefined;
  const full = (): GemmaTokenizer =>
    fullTokenizer ??= new GemmaTokenizer(parseGemmaTokenizerAsset(seriesBytes as Uint8Array));

  if (seriesBytes === undefined) {
    console.warn(
      `[karume] outputs/series/${series}/tokenizer.json が無いため ${family} の実資産突合を ` +
        `SKIP する。生成: cd tools/export-recipes && uv run python -m ` +
        `${fixtureName === "gemma4" ? "gemma4" : "embeddinggemma"}.tokenizer`,
    );
  }

  Deno.test({
    name: `${family} 実資産突合: 全語彙でもフィクスチャと同じ id 列（部分集合の穴を塞ぐ）`,
    ignore: seriesBytes === undefined,
    fn: () => {
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
  });

  Deno.test({
    name: `${family} 実資産の日本語長文: 併合キューが実用時間で終わる`,
    ignore: seriesBytes === undefined,
    fn: () => {
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
      if (ids.length === 0 || ids.length >= [...long].length) {
        throw new Error(`併合が効いていない（${ids.length} トークン / ${[...long].length} 文字）`);
      }
      const BUDGET_MS = 5_000;
      if (elapsed > BUDGET_MS) {
        throw new Error(
          `${[...long].length} 文字の符号化に ${elapsed.toFixed(0)}ms（予算 ${BUDGET_MS}ms）`,
        );
      }
    },
  });
}

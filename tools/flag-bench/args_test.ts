/**
 * flag-bench の引数解析の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る点:
 *
 * 1. **既定値と基準** — 省略したノブは既定値で埋まり、`--set` は与えた順に並ぶ（1 本目が基準）
 * 2. **fail loudly** — 未知のオプション・未知の上書きキー・型の違う値・重複・値の欠けを落とす
 * 3. **`--set` の綴り** — `=` の最初の 1 個で割る・ラベルの綴り・空の `{}` は上書き無し
 * 4. **subgroup の要求** — subgroup 変種を指定した set だけが device に subgroup を要る
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assert, assertEquals, assertFalse, assertThrows } from "@std/assert";
import {
  DEFAULT_CAPACITY,
  DEFAULT_NEW_TOKENS,
  DEFAULT_ROUNDS,
  needsSubgroups,
  parseArgs,
  parseOverrides,
  parseSet,
} from "./args.ts";

const BASE = [
  "--source",
  "models/karume-gemma4-qat",
  "--family",
  "qat",
  "--model",
  "e4b",
  "--quant",
  "i4",
  "--out",
  "out.jsonl",
];

Deno.test("parseArgs: 必須だけを与えたとき", async (t) => {
  await t.step("省略したノブは既定値で埋まり、計測は OFF", () => {
    const args = parseArgs([...BASE, "--set", "ref={}"]);
    assertEquals(args.rounds, DEFAULT_ROUNDS);
    assertEquals(args.newTokens, DEFAULT_NEW_TOKENS);
    assertEquals(args.capacity, DEFAULT_CAPACITY);
    assertFalse(args.gpuTiming);
    assertEquals(args.family, "qat");
    assertEquals(args.model, "e4b");
  });

  await t.step("--set は与えた順に並び、1 本目が基準になる", () => {
    const args = parseArgs([
      ...BASE,
      "--set",
      'ref={"stateAttentionReduce":"sequential"}',
      "--set",
      'par={"linearGemvReduce":"parallel"}',
      "--gpu-timing",
    ]);
    assertEquals(args.sets.map((set) => set.label), ["ref", "par"]);
    assertEquals(args.sets[0].overrides, { stateAttentionReduce: "sequential" });
    assert(args.gpuTiming);
  });
});

Deno.test("parseArgs: 条件を偽る指定は落とす", async (t) => {
  await t.step("未知のオプション", () => {
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--round", "3"]), Error, "未知");
  });
  await t.step("--set が 1 本も無い", () => {
    assertThrows(() => parseArgs(BASE), Error, "--set");
  });
  await t.step("必須の欠け", () => {
    assertThrows(() => parseArgs(["--set", "a={}", "--out", "x.jsonl"]), Error, "必須");
  });
  await t.step("--set 以外の重複（後勝ちにしない）", () => {
    assertThrows(
      () => parseArgs([...BASE, "--set", "a={}", "--rounds", "2", "--rounds", "3"]),
      Error,
      "2 回",
    );
  });
  await t.step("ラベルの重複", () => {
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--set", "a={}"]), Error, "重複");
  });
  await t.step("値の位置に次のフラグ（値として食わない）", () => {
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--rounds", "--gpu-timing"]), Error);
  });
  await t.step("family / model の綴り違い", () => {
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--family", "gemma4"]), Error);
    assertThrows(
      () =>
        parseArgs([
          "--source",
          "s",
          "--family",
          "normal",
          "--model",
          "e8b",
          "--quant",
          "i4",
          "--out",
          "o",
          "--set",
          "a={}",
        ]),
      Error,
      "e8b",
    );
  });
  await t.step("decode の分母が無い --new-tokens 1 と、正でない整数", () => {
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--new-tokens", "1"]), Error, "2 以上");
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--rounds", "0"]), Error);
    assertThrows(() => parseArgs([...BASE, "--set", "a={}", "--capacity", "1.5"]), Error);
  });
});

Deno.test("parseSet: <label>=<json>", async (t) => {
  await t.step("空の {} は上書き無し", () => {
    assertEquals(parseSet("ref={}"), { label: "ref", overrides: {} });
  });
  await t.step("= の最初の 1 個で割る（ラベルに = は入らない）", () => {
    assertThrows(() => parseSet('a={"x":"="}'), Error, "未知のキー");
    assertThrows(() => parseSet("noequals"), Error, "<label>=<json>");
  });
  await t.step("ラベルの綴り（空・空白・先頭記号は受けない）", () => {
    assertThrows(() => parseSet("={}"), Error, "ラベル");
    assertThrows(() => parseSet("a b={}"), Error, "ラベル");
    assertThrows(() => parseSet("-a={}"), Error, "ラベル");
    assertEquals(parseSet("fuse.rms+add_1={}").label, "fuse.rms+add_1");
  });
  await t.step("JSON として読めない", () => {
    assertThrows(() => parseSet("a={linearGemvReduce:parallel}"), Error, "JSON");
  });
});

Deno.test("parseOverrides: 受理キーと型", async (t) => {
  await t.step("全キーを型どおりに写す", () => {
    // JSON.parse の出力と同じ unknown 境界から入れる。
    const raw: unknown = {
      linearGemvReduce: "parallel",
      fuseRmsNormAdd: true,
      fuseLinearStaticQuantize: false,
      packedStaticQuantize: true,
      stateAttentionReduce: "parallel-fused",
      rmsNormReduce: "workgroup",
      submitPolicy: { timeBudgetMs: 100, initialChunkSize: 16, minChunkSize: 1, maxChunkSize: 768 },
      planBackingBudgetBytes: 0,
      linearGemvRowsThreadTarget: 4096,
      chunkLength: 64,
      chunkBuckets: [4, 8, 16, 32],
      pleResidency: "gpu",
      maxResidentPleBytes: 1024,
    };
    assertEquals(parseOverrides(raw, "t"), raw);
  });
  await t.step("未知のキー（打ち間違い）は落とす", () => {
    assertThrows(() => parseOverrides({ linearGemvReduction: "parallel" }, "t"), Error, "未知");
    assertThrows(() => parseOverrides({ speculative: {} }, "t"), Error, "未知");
    assertThrows(() => parseOverrides({ gpu: {} }, "t"), Error, "未知");
  });
  await t.step("型の違う値は落とす", () => {
    assertThrows(() => parseOverrides({ fuseRmsNormAdd: "true" }, "t"), Error, "boolean");
    assertThrows(() => parseOverrides({ linearGemvReduce: "fast" }, "t"), Error, "既知");
    assertThrows(() => parseOverrides({ stateAttentionReduce: 1 }, "t"), Error);
    assertThrows(() => parseOverrides({ chunkLength: 64.5 }, "t"), Error, "整数");
    assertThrows(() => parseOverrides({ chunkBuckets: [4, "8"] }, "t"), Error, "[1]");
    assertThrows(() => parseOverrides({ pleResidency: "disk" }, "t"), Error);
    assertThrows(() => parseOverrides([], "t"), Error, "オブジェクト");
  });
  await t.step("submitPolicy は 4 欄が揃った形だけ", () => {
    assertThrows(
      () => parseOverrides({ submitPolicy: { timeBudgetMs: 100, initialChunkSize: 16 } }, "t"),
      Error,
    );
    assertThrows(
      () =>
        parseOverrides({
          submitPolicy: {
            timeBudgetMs: 100,
            initialChunkSize: 16,
            minChunkSize: 1,
            maxChunkSize: 768,
            extra: 1,
          },
        }, "t"),
      Error,
      "未知の欄",
    );
  });
});

Deno.test("needsSubgroups: device に subgroup が要る set", async (t) => {
  await t.step("subgroup 変種を指定したときだけ true", () => {
    assert(needsSubgroups({ linearGemvReduce: "parallel-subgroup32" }));
    assert(needsSubgroups({ rmsNormReduce: "subgroup32" }));
    assertFalse(needsSubgroups({ linearGemvReduce: "parallel", rmsNormReduce: "workgroup" }));
    assertFalse(needsSubgroups({}));
  });
});

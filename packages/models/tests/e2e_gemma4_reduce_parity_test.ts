// 実重み Gemma 4 E2B の **③PV 縮約順 A/B**（K-12・ADR 0058 決定 4 ②のモデル面）。
//
// [limitations](../../../docs/limitations.md) は「`Gemma4Pipeline` の既定は KV 並列縮約
// （`"parallel"`）で、runtime の参照経路 `"sequential"` とは縮約順が違うが、**gemma4 の golden は
// 両者で同一**」と書く。K-12 を既定へ昇格した根拠がこの主張なので、機械で守る門をここに置く。
//
// 既存の門はこれを担保しない: カーネル単位の A/B 帯（`gpu_state_attention_parallel_test.ts`）は
// 帯の内側であることしか見ず、census（`e2e_gemma4_pretrained_test.ts`）は**キーの有無**しか見ず、
// 2 本の golden（greedy = sequential 経由 / chat = parallel 経由）は別々の期待値で別々に緑になる
// だけで「両者が同一」は導けない。
//
// MUST: この門は census へ相乗りさせない。census は `timestamp-query` を要求するので Metal では
// 常に SKIP になり、等式の門まで Mac で永久に走らなくなる（W-G7-4 / Pass2）。ここは計測を
// 要求しないので、SKIP 条件は**資産と GPU の有無だけ**である。
//
// ## 資産と経路
//
// 配布形ミラー `models/karume-gemma4-e2b/` を `denoDirectory` で直読みする（HTTP も永続キャッシュ
// も要らない — ADR 0086）。ミラーが無い環境では**明示 SKIP**。

import { assert, assertEquals } from "@std/assert";
import { MANIFEST_FILENAME } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import { Gemma4Pipeline } from "../gemma.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { allResidentPleBytesOfMirror } from "./helpers/ple-budget.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4-e2b/", import.meta.url);

/** SKIP 時にそのまま貼れる組み立てコマンド。 */
const ASSEMBLE_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline gemma4";

/**
 * 比べる 2 つの縮約順。**両方を明示的に渡す**（省略時の既定が `"parallel"` であることは census の
 * 担当で、ここが見るのは「2 つの経路が同じ列を出す」ことだけ）。
 */
const REDUCERS = ["parallel", "sequential"] as const;

/**
 * 検収ケース。prompt は chat フィクスチャの token 列（描画の正本は `gemma_chat_test.ts`）を
 * そのまま使う — この門は描画でも復号でもなく**縮約順**だけを問うので、tokenizer を挟まずに
 * token 列 in → token 列 out で回す。日本語ケースを混ぜるのは decode step を伸ばすため。
 */
const CASES = [
  { fixture: "single-user", maxNewTokens: 24 },
  { fixture: "japanese", maxNewTokens: 24 },
] as const;

type ChatFixture = {
  readonly chat: { readonly name: string; readonly ids: number[] }[];
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
) as ChatFixture;

const promptOf = (name: string): readonly number[] => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `フィクスチャに chat ケース '${name}' が無い`);
  return found.ids;
};

const AVAILABLE = ((): boolean => {
  try {
    return Deno.statSync(new URL(MANIFEST_FILENAME, MIRROR_DIR)).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
})();

if (!AVAILABLE) {
  console.warn(
    `[karume] 配布形ミラー models/karume-gemma4-e2b/ が無いため ③PV 縮約順の A/B を SKIP する。` +
      `組み立て: ${ASSEMBLE_COMMAND}`,
  );
}

/** 1 会話ぶんの生成結果（`token` イベントの id 列と停止理由）。 */
type Run = {
  readonly ids: number[];
  readonly stop: { readonly reason: string; readonly tokens: number };
};

/**
 * 1 ケースを 1 本の sequence で回す。**温度 0** なので抽選の揺れは無く、列が動いたら原因は
 * 縮約順しかない。
 */
const runCase = async (
  pipeline: Gemma4Pipeline,
  prompt: readonly number[],
  maxNewTokens: number,
): Promise<Run> => {
  const sequence = await pipeline.sequence();
  try {
    const stream = sequence.generate({ prompt, maxNewTokens, sampler: { temperature: 0 } });
    const ids: number[] = [];
    for await (const event of stream) {
      if (event.kind === "token") ids.push(event.id);
    }
    const stop = await stream.done;
    return { ids, stop: { reason: stop.reason, tokens: stop.tokens } };
  } finally {
    await sequence.dispose();
  }
};

Deno.test({
  name: "gemma4: ③PV の縮約順（parallel / sequential）で token 列が 1 個も動かない（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const maxResidentPleBytes = allResidentPleBytesOfMirror(MIRROR_DIR);
    const runs: Run[][] = [];
    for (const reduce of REDUCERS) {
      await t.step(`stateAttentionReduce = ${reduce}`, async () => {
        const pipeline = await Gemma4Pipeline.fromPretrained(denoDirectory(MIRROR_DIR), {
          maxResidentPleBytes,
          stateAttentionReduce: reduce,
        });
        try {
          const collected: Run[] = [];
          for (const { fixture: name, maxNewTokens } of CASES) {
            const started = performance.now();
            const run = await runCase(pipeline, promptOf(name), maxNewTokens);
            collected.push(run);
            console.log(
              `[e2e] gemma4 ${reduce} ${name}: ${run.ids.length} token / ` +
                `${JSON.stringify(run.stop)} / ${(performance.now() - started).toFixed(0)}ms`,
            );
          }
          runs.push(collected);
        } finally {
          await pipeline.dispose();
        }
      });
    }

    const [parallel, sequential] = runs;
    assertEquals(runs.length, REDUCERS.length, "両方の縮約順で走っていない");
    for (const [index, { fixture: name }] of CASES.entries()) {
      // 陽性対照 — 「どちらも 0 個だった」形で等式が成立するのを塞ぐ。
      assert(parallel[index].ids.length > 0, `${name}: 1 token も生成していない`);
      assertEquals(parallel[index].ids, sequential[index].ids, `${name}: 縮約順で割れた token 列`);
      assertEquals(parallel[index].stop, sequential[index].stop, `${name}: 縮約順で割れた停止理由`);
    }
  },
});

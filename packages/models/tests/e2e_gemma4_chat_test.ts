// 実重み Gemma 4 E2B の **文字列 in → 文字列 out** の検収門 — 段 4 の合格線。
//
// 検収するのは `src/gemma/pipeline.ts`（`Gemma4Pipeline`）と `src/gemma/text/chat.ts`
// （`gemma4ChatPrompt`）の結線で、門は 5 本:
//
// ① **完走**: `fromAssets` → `chat([...])` が実重みで回り、温度 0（この層の既定）の出力が
//    固定した greedy golden と文字単位で一致する。golden は**この経路自身で採った**もので、
//    モデルの正しさは証明しない — 証明するのは「結線を変えても出力が動かない」ことである
//    （数値の正は `e2e_gemma4_product_test.ts` の交差 parity が持つ）
// ② **chat の id 列**が HF `apply_chat_template` のフィクスチャと一致する。`gemma_chat_test.ts`
//    は部分集合と実資産で同じことを見るが、こちらは**パイプラインが握っている資産**で見る
//    （別の tokenizer を掴んでいれば描画は合ったまま id 列だけが違う）
// ③ **逐次と一括の一致**: streaming の片を連結したものが、同じ会話を低レベル面（`sequence()`）
//    で回して得た token 列の一括 decode と一致する。逐次復号が byte run を取りこぼしたり
//    「finish() まで全部溜める偽 streaming」になっても①は緑のままなので、この門が要る
// ④ **停止集合**が上流の `generation_config.json` の宣言（`[1, 106, 50]`）と一致する
// ⑤ **射程外は GPU に触る前に落ちる**（tools / 未知 role — `chat` は同期に throw する）
//
// ## 資産
//
// `outputs/series/gemma4-e2b-product/`（製品グラフ + PLE sidecar）と
// `outputs/series/gemma4-e2b-tokenizer/tokenizer.json`。どちらもリポジトリ管理外で、無い環境
// では**明示 SKIP** する。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { Gemma4Pipeline } from "../src/gemma/pipeline.ts";
import { type Gemma4ChatMessage, gemma4ChatPrompt } from "../src/gemma/text/chat.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const PRODUCT_ROOT = new URL("../../../outputs/series/gemma4-e2b-product/", import.meta.url);
const TOKENIZER_ASSET = new URL(
  "../../../outputs/series/gemma4-e2b-tokenizer/tokenizer.json",
  import.meta.url,
);
const PLE_INDEX_FILE = "ple.json";
const MODEL_SHARD = /^model-\d+-of-\d+\.safetensors$/;

/** SKIP 時にそのまま貼れる生成コマンド。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_product" +
  "（tokenizer 資産は … python -m gemma4.tokenizer）";

/** 実行条件は既存の gemma4 検収門と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 32;
const CAPACITY = 640;
const MAX_POSITION = 1024;
/** PLE は 3 shard 全部を常駐させる（範囲をまたぐ会話で 758MB の読み直しを起こさない）。 */
const RESIDENT_PLE_SHARDS = 3;

/** gemma-4-E2B-it の `generation_config.json` の `eos_token_id`（ADR 0083 決定 8）。 */
const STOP_TOKENS = [1, 106, 50];

/**
 * 検収ケース。`prompt` はフィクスチャ（`gemma4-chat.json`）のケース名で、`expected` は
 * **この経路で採った温度 0 の出力**である（2026-08-31・RTX 3080 Ti）。
 *
 * MUST: 期待値を「実行結果で上書きして緑にする」ことをしない — 割れたら、まず
 * `e2e_gemma4_product_test.ts` の交差 parity（torch との突合）が緑かどうかを見る。あちらが
 * 緑でここだけ割れるなら、動いたのは結線（chat の描画 / 逐次復号 / sampler の既定）である。
 */
const CASES = [
  {
    fixture: "single-user",
    maxNewTokens: 24,
    expected: "The capital of France is **Paris**.",
    // `<turn|>`（106）で自分から turn を閉じる = 停止集合が実出力に効いている証拠。
    stop: { reason: "eos", token: 106 },
  },
  {
    fixture: "japanese",
    maxNewTokens: 24,
    // 多ターン（system + user + assistant + user）の 4 通目への応答。
    expected: "2023年時点で、およそ1,400万人です。",
    stop: { reason: "eos", token: 106 },
  },
] as const;

type ChatFixture = {
  readonly stopTokens: number[];
  readonly chat: {
    readonly name: string;
    readonly messages: Gemma4ChatMessage[];
    readonly ids: number[];
  }[];
};

const fixture = JSON.parse(
  await Deno.readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
) as ChatFixture;

const caseOf = (name: string) => {
  const found = fixture.chat.find((row) => row.name === name);
  assert(found !== undefined, `フィクスチャに chat ケース '${name}' が無い`);
  return found;
};

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const shardFiles = (): string[] => {
  try {
    return [...Deno.readDirSync(PRODUCT_ROOT)]
      .map((entry) => entry.name)
      .filter((name) => MODEL_SHARD.test(name))
      .sort();
  } catch {
    return [];
  }
};

const MODEL_SHARDS = shardFiles();
const AVAILABLE = MODEL_SHARDS.length > 0 &&
  exists(new URL(PLE_INDEX_FILE, PRODUCT_ROOT)) && exists(TOKENIZER_ASSET);

if (!AVAILABLE) {
  console.warn(
    `[karume] 製品系列 / tokenizer 資産が無いため Gemma 4 E2B chat 検収を SKIP する。` +
      `生成: ${GENERATE_COMMAND}`,
  );
}

/**
 * ファイル 1 本を `ArrayBuffer` として読む。
 * MUST: view が buffer 全体を覆っているなら slice しない（PLE sidecar は 1 本 758MB 級）。
 */
const readBuffer = async (root: URL, file: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(file, root));
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const openPipeline = async (): Promise<Gemma4Pipeline> => {
  const model: Uint8Array<ArrayBuffer>[] = [];
  for (const file of MODEL_SHARDS) model.push(new Uint8Array(await readBuffer(PRODUCT_ROOT, file)));
  return await Gemma4Pipeline.fromAssets({
    config: { chunkLength: CHUNK_LENGTH, maxPosition: MAX_POSITION, capacity: CAPACITY },
    model,
    tokenizer: await Deno.readFile(TOKENIZER_ASSET),
    pleIndex: await Deno.readFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT)),
    readPleShard: (file) => readBuffer(PRODUCT_ROOT, file),
  }, { residentPleShards: RESIDENT_PLE_SHARDS });
};

Deno.test({
  name: "Gemma 4 E2B chat 検収: 文字列 in → 文字列 out・逐次と一括の一致（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const pipeline = await openPipeline();
    try {
      await t.step("① 温度 0 の chat が実重みで完走し、固定した greedy golden と一致", async () => {
        for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
          const { messages } = caseOf(name);
          const started = performance.now();
          const stream = pipeline.chat(messages, { maxNewTokens });
          const parts: string[] = [];
          for await (const chunk of stream) parts.push(chunk);
          const stopped = await stream.done;

          console.log(
            `[e2e] gemma4 chat ${name}: ${JSON.stringify(parts.join(""))} / ` +
              `${JSON.stringify(stopped)} / ${(performance.now() - started).toFixed(0)}ms`,
          );
          assertEquals(parts.join(""), expected, `${name}: 温度 0 の出力`);
          assertEquals(stopped, stop, `${name}: 停止理由`);
          // 片は「確定したぶんだけ」なので空文字は流れない（偽の 1 反復を作らない）。
          assertEquals(parts.filter((part) => part === "").length, 0, `${name}: 空の片`);
        }
      });

      await t.step("② パイプラインが握る資産の chat id 列がフィクスチャと一致", () => {
        for (const { fixture: name } of CASES) {
          const { messages, ids } = caseOf(name);
          assertEquals(gemma4ChatPrompt(pipeline.tokenizer, messages), ids, name);
        }
      });

      await t.step("③ streaming の片の連結 = 同じ会話の token 列の一括 decode", async () => {
        for (const { fixture: name, maxNewTokens, expected, stop } of CASES) {
          const { ids } = caseOf(name);
          // 低レベル面は 1 会話 = 1 sequence（context を跨がせない）。
          const sequence = await pipeline.sequence();
          try {
            const stream = sequence.generate({ prompt: ids, maxNewTokens });
            const tokens: number[] = [];
            for await (const event of stream) if (event.kind === "token") tokens.push(event.id);
            assertEquals(await stream.done, stop, `${name}: 低レベル面の停止理由`);
            assertEquals(
              pipeline.tokenizer.decode(tokens),
              expected,
              `${name}: 一括 decode（逐次が byte run を取りこぼしていれば割れる）`,
            );
          } finally {
            await sequence.dispose();
          }
        }
      });

      await t.step("④ 停止集合は上流の generation_config.json の宣言と一致", () => {
        const ascending = (ids: readonly number[]): number[] => [...ids].sort((a, b) => a - b);
        assertEquals(ascending(pipeline.program.stopTokens), ascending(STOP_TOKENS), "program");
        assertEquals(ascending(fixture.stopTokens), ascending(STOP_TOKENS), "フィクスチャ");
      });

      await t.step("⑤ 射程外の会話は GPU に触る前に落ちる（同期の throw）", () => {
        assertThrows(
          () =>
            pipeline.chat(
              [{ role: "user", content: "hi", tools: [] }] as unknown as Gemma4ChatMessage[],
              { maxNewTokens: 4 },
            ),
          Error,
          "射程外の欄",
        );
        assertThrows(
          () =>
            pipeline.chat(
              [{ role: "tool", content: "…" }] as unknown as Gemma4ChatMessage[],
              { maxNewTokens: 4 },
            ),
          Error,
          "role",
        );
      });
    } finally {
      await pipeline.dispose();
    }

    await t.step("dispose 済みのパイプラインは生成を受けない", async () => {
      assertThrows(
        () => pipeline.chat([{ role: "user", content: "hi" }], { maxNewTokens: 4 }),
        Error,
        "dispose 済み",
      );
      await assertRejects(() => pipeline.sequence(), Error, "dispose 済み");
    });
  },
});

// 実重み Gemma 4 E2B の**製品グラフ + ホスト PLE + GenerationSequence** の検収門 — 段 3 の合格線。
//
// 検収するのは ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) 決定 1〜5 の
// 生成面（`src/generation/program.ts` / `sequence.ts`）で、門は 3 本:
//
// ① **1 ターン目の parity**: 温度 0（この層の既定）の sequence で回した 1 ターンが、logits
//    opt-in 系列の `greedy.<case>.safetensors`（torch full re-forward の期待列）と 3 ケース ×
//    K=16 で厳密一致する。段 2 の門（`e2e_gemma4_sampler_test.ts`）が**その場で書いたループ**で
//    採った同じ列を、**sequence 経由**で採り直す門である — 2 層化（program / sequence）と
//    ホスト由来入力の席を挟んだことで列が動かないことの証明。
// ② **多ターンで token を 1 個も落とさない**（決定 4 の `pendingToken` 連結 prefill）: 1 ターン目を
//    途中で `break` し、続きを 2 ターン目として出した列を連結すると、**中断せずに 1 本で流した
//    参照走（= ①の golden）と厳密一致**する。落ちても例外は出ず「直前 assistant の最後の
//    1 token が履歴から消える」だけなので、この直接門でしか捕まらない。
//    NOTE: 一致が**厳密**になるのは、有効行 1 本の chunk を decode 形（M=1）で流すため
//    （`sequence.ts` の該当 MUST）— 再開の 1 run が中断しなかった走りの decode run と同一になる。
// ③ **EOS 集合での実停止**（決定 8）: 停止 token を宣言した program では、実 GPU の出力に対して
//    ケースごとに固定した添字（段 2 の門と同じ 2 / 3 / 2）で止まり、停止 token 自体は `token`
//    イベントに出ず `done` が運ぶ。
//
// NOTE: golden の出所（`reference.json` が束ねた digest）を突き合わせる門は
// `e2e_gemma4_product_test.ts` の④が同じ 3 本に対して持っている — 同じ資産に 2 つ置かない。
//
// ## 資産
//
// `outputs/series/gemma4-e2b-product/`（コンテナ + PLE sidecar）と
// `outputs/series/gemma4-e2b-decode/`（期待列の正本）。どちらもリポジトリ管理外で、無い環境では
// **明示 SKIP** する。

import { assert, assertEquals } from "@std/assert";
import { acquireGpu, parseSafetensors, prepareModel, type SafetensorsFile } from "@karume/runtime";
import { createGemma4Ple, parseGemma4PleIndex } from "../src/gemma/ple.ts";
import { createGenerationProgram, type GenerationProgram } from "../src/generation/program.ts";
import {
  createGenerationSequence,
  type GenerationEvent,
  type GenerationSequence,
} from "../src/generation/sequence.ts";
import {
  modelPresent,
  readShard,
  resolveShards,
  streamShards,
} from "../../runtime/tests/helpers/shard-files.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const PRODUCT_ROOT = new URL("../../../outputs/series/gemma4-e2b-product/", import.meta.url);
const GOLDEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-decode/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const PLE_INDEX_FILE = "ple.json";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_product" +
  "（期待列の正本は … python -m gemma4.export_decode）";

/**
 * ケースごとの期待列（正典は `export_decode.GREEDY_CASES`）と、EOS 集合が最初に成立する添字。
 *
 * MUST: 停止添字は手で固定する（段 2 の門と同文）— golden から導くと「集合が痩せても添字も
 * 一緒に動く」形になり、③が恒真化する。
 */
const EXPECTED_CASES = [
  { name: "capital-en", firstStop: 2 },
  { name: "capital-ja", firstStop: 3 },
  { name: "context-en", firstStop: 2 },
] as const;
const GREEDY_STEPS = 16;
/** ②で 1 ターン目を打ち切る token 数（残りは 2 ターン目が出す）。 */
const BREAK_AFTER = 5;

/** gemma-4-E2B-it の `generation_config.json` の `eos_token_id`（ADR 0083 決定 8）。 */
const STOP_TOKENS = [1, 106, 50] as const;

/** 実行条件は既存の門と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 32;
const CAPACITY_SYMBOL = "C";
const CAPACITY = 640;
const MAX_POSITION = 1024;

/** グラフ入力の名前（正本は `export_product` の定数）。 */
const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";
const PER_LAYER_INPUTS = "per_layer_inputs";
const LAST_ROW = "last_row";

const VOCAB = 262144;

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, PRODUCT_ROOT));
const SIDECAR_PRESENT = exists(new URL(PLE_INDEX_FILE, PRODUCT_ROOT));
const GOLDENS_PRESENT = EXPECTED_CASES.every(({ name }) =>
  exists(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT))
);
const AVAILABLE = MODEL_PRESENT && SIDECAR_PRESENT && GOLDENS_PRESENT;

if (!MODEL_PRESENT) {
  console.warn(
    `[karume] 製品系列（${PRODUCT_ROOT.pathname}）が無いため Gemma 4 E2B sequence 検収を ` +
      `SKIP する。生成: ${GENERATE_COMMAND}`,
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

const goldenI32 = (file: SafetensorsFile, name: string): Int32Array<ArrayBuffer> => {
  const view = file.tensors.get(name);
  assert(view !== undefined, `golden に '${name}' が無い`);
  assertEquals(view.dtype, "I32", `golden '${name}' の格納 dtype`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const readCase = async (
  name: string,
): Promise<{ readonly prompt: number[]; readonly expected: number[] }> => {
  const golden = parseSafetensors(
    await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${name}${SUFFIX}`),
  );
  return {
    prompt: [...goldenI32(golden, "prompt")],
    expected: [...goldenI32(golden, "expected")],
  };
};

const tokenIds = (events: readonly GenerationEvent[]): number[] =>
  events.filter((event) => event.kind === "token").map((event) => event.id);

Deno.test({
  name: "Gemma 4 E2B sequence 検収: 1 ターン parity・多ターン連結・EOS 停止（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const shards = resolveShards(new URL(MODEL_FILE, PRODUCT_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));

    const index = parseGemma4PleIndex(
      JSON.parse(await Deno.readTextFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT))),
    );
    const ple = createGemma4Ple({
      index,
      readShard: (file) => readBuffer(PRODUCT_ROOT, file),
      vocabSize: VOCAB,
      residentShards: index.shards.length,
    });

    /** 静的配線（停止集合だけを変えて 2 本作る — 他は同じ資産の同じ結線）。 */
    const programOf = (stopTokens: readonly number[]): GenerationProgram =>
      createGenerationProgram({
        graph: parsed.graph,
        inputIds: INPUT_IDS,
        positionIds: POSITION_IDS,
        lastRow: LAST_ROW,
        logits: parsed.graph.outputs[0],
        chunkLength: CHUNK_LENGTH,
        maxPosition: MAX_POSITION,
        capacity: CAPACITY,
        vocabSize: VOCAB,
        stopTokens,
        bindings: { [CAPACITY_SYMBOL]: CAPACITY },
        // ホスト由来の per-chunk 入力の席に PLE gather を差す（ADR 0085 — `ple.ts` は無改変）。
        derivedInputs: {
          names: [PER_LAYER_INPUTS],
          derive: async (ids) => ({ [PER_LAYER_INPUTS]: await ple.gather(ids) }),
        },
      });

    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    /** 1 会話 = 1 sequence（context を抱えたままにしないので、使い終わりで返す）。 */
    const withSequence = async <T>(
      stopTokens: readonly number[],
      body: (sequence: GenerationSequence) => Promise<T>,
    ): Promise<T> => {
      const sequence = await createGenerationSequence({
        session,
        program: programOf(stopTokens),
      });
      try {
        return await body(sequence);
      } finally {
        await sequence.dispose();
      }
    };

    try {
      await t.step("① 温度 0 の sequence が 3 ケース × 16 step で期待列と一致", async () => {
        for (const { name } of EXPECTED_CASES) {
          const { prompt, expected } = await readCase(name);
          assertEquals(expected.length, GREEDY_STEPS, `${name}: golden の step 数`);

          const started = performance.now();
          const { events, stop } = await withSequence([], async (sequence) => {
            const stream = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
            const collected: GenerationEvent[] = [];
            for await (const event of stream) collected.push(event);
            return { events: collected, stop: await stream.done };
          });

          assertEquals(tokenIds(events), expected, `${name}: sequence 経由の生成 token 列`);
          assertEquals(stop, { reason: "max-tokens" }, `${name}: 停止理由`);
          // prefill イベントは chunk の割り方どおり（T=10 級なので 1 本）。
          const prefills = events.filter((event) => event.kind === "prefill");
          assertEquals(
            prefills.length,
            Math.ceil(prompt.length / CHUNK_LENGTH),
            `${name}: prefill イベント数`,
          );
          // 位置は prompt の続きから 1 ずつ（sequence は counter を持たず context から導出する）。
          assertEquals(
            events.filter((event) => event.kind === "token").map((event) => event.position),
            Array.from({ length: GREEDY_STEPS }, (_unused, step) => prompt.length + step),
            `${name}: token イベントの絶対位置`,
          );
          console.log(
            `[e2e] gemma4 sequence ${name}: T=${prompt.length} + K=${GREEDY_STEPS} step / ` +
              `${(performance.now() - started).toFixed(0)}ms`,
          );
        }
      });

      await t.step(
        "② break → 続きの 2 ターン目が参照走と厳密一致（token を 1 個も落とさない）",
        async () => {
          const { name } = EXPECTED_CASES[0];
          const { prompt, expected } = await readCase(name);

          const generated = await withSequence([], async (sequence) => {
            // 1 ターン目: 途中で break（`return()` 経由で finally へ入る経路）。
            const first = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
            const head: number[] = [];
            for await (const event of first) {
              if (event.kind !== "token") continue;
              head.push(event.id);
              if (head.length === BREAK_AFTER) break;
            }
            assertEquals(await first.done, { reason: "closed" }, "1 ターン目の停止理由");
            assertEquals(head, expected.slice(0, BREAK_AFTER), "1 ターン目の token 列");

            // 2 ターン目: 新しい token を足さずに続きだけ（pendingToken が先頭へ連結される）。
            const second = sequence.generate({
              prompt: [],
              maxNewTokens: GREEDY_STEPS - BREAK_AFTER,
            });
            const tail: number[] = [];
            for await (const event of second) {
              if (event.kind === "token") tail.push(event.id);
            }
            assertEquals(await second.done, { reason: "max-tokens" }, "2 ターン目の停止理由");
            return [...head, ...tail];
          });

          assertEquals(
            generated,
            expected,
            `${name}: 中断 → 再開の連結列（1 本で流した参照走と厳密一致すること — ` +
              `割れたら pendingToken が落ちているか、再開の run 形が違う）`,
          );
          console.log(
            `[e2e] gemma4 sequence 多ターン: ${BREAK_AFTER} + ${
              GREEDY_STEPS - BREAK_AFTER
            } token が ` +
              `1 本走りと一致`,
          );
        },
      );

      await t.step("③ EOS 集合を宣言した program は実出力で止まる", async () => {
        for (const { name, firstStop } of EXPECTED_CASES) {
          const { prompt, expected } = await readCase(name);
          const { events, stop } = await withSequence(STOP_TOKENS, async (sequence) => {
            const stream = sequence.generate({ prompt, maxNewTokens: GREEDY_STEPS });
            const collected: GenerationEvent[] = [];
            for await (const event of stream) collected.push(event);
            return { events: collected, stop: await stream.done };
          });

          // 停止 token 自体は `token` イベントに出さない（本文ではなく終端記号）。
          assertEquals(
            tokenIds(events),
            expected.slice(0, firstStop),
            `${name}: 停止までに出た token 列`,
          );
          assertEquals(
            stop,
            { reason: "eos", token: expected[firstStop] },
            `${name}: EOS 集合 [${STOP_TOKENS.join(",")}] での停止`,
          );
          console.log(`[e2e] gemma4 sequence ${name}: EOS 停止 @${firstStop}`);
        }
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

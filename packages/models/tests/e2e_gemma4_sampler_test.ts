// 実重み Gemma 4 E2B の**製品グラフ + ホスト sampler** の検収門 — 段 2 の合格線。
//
// 検収するのは ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) 決定 7〜8 の
// ホスト側 sampling（`src/generation/sampler.ts`）で、門は 2 本:
//
// ① **温度 0 の parity**: `createSampler()`（既定 = 温度 0 の greedy 縮退）で回した生成ループが、
//    logits opt-in 系列の `greedy.<case>.safetensors`（torch full re-forward の期待列）と
//    3 ケース × K=16 で厳密一致する。段 1b の交差 parity（`e2e_gemma4_product_test.ts` ③）が
//    **その場で書いた argmax** で採った同じ列を、**製品面の sampler 経由**で採り直す門である
//    — sampler を挟んだことで列が動かない（= 温度 0 が argmax の縮退形である）ことの証明で、
//    割れたら加工の順序・tie-break・非有限の扱いのどれかが argmax とずれている。
//    併せて `topK: 1` が V = 262,144 の実 logits でも greedy と一致することを 1 ケースで見る
//    （単体テストの語彙は 4〜4,096 なので、実寸の候補選択を踏むのはここだけ）。
// ② **EOS 集合の停止判定**（決定 8）: 実 GPU が出した token 列に対し `isStopToken` が
//    `[1, 106, 50]`（gemma-4-E2B-it の `generation_config.json`）で最初に成立する位置が、
//    ケースごとに固定した添字と一致する。**ループへの結線は段 3** なので、ここで見るのは
//    「実出力に当てたときにどこで止まるか」だけ。集合が痩せる（`[1]` だけになる等）と添字が
//    後ろへずれて落ちる。
//
// ホスト側の生成ループはこの門が自前で組む（`src/generation/greedy.ts` はグラフ側 argmax 出力を
// 読む形で、最終行 logits 出口の製品グラフには通せない）。chunk の割り方だけは
// `planPrefillChunks` を共有する。
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
import {
  acquireGpu,
  parseSafetensors,
  type PreparedModel,
  prepareModel,
  type SafetensorsFile,
  type Tensor,
} from "@karume/runtime";
import {
  createGemma4Ple,
  type Gemma4Ple,
  gemma4PleShardBytes,
  parseGemma4PleIndex,
} from "../src/gemma/ple.ts";
import { gemma4RopeInputs, type Gemma4RopeSpec } from "../src/gemma/rope.ts";
import { planPrefillChunks } from "../src/generation/greedy.ts";
import { createSampler, isStopToken, type SamplerSpec } from "../src/generation/sampler.ts";
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
 * MUST: 停止添字は手で固定する — golden から導くと「集合が痩せても添字も一緒に動く」形になり、
 * ②が恒真化する。
 */
const EXPECTED_CASES = [
  { name: "capital-en", firstStop: 2 },
  { name: "capital-ja", firstStop: 3 },
  { name: "context-en", firstStop: 2 },
] as const;
const GREEDY_STEPS = 16;

/** gemma-4-E2B-it の `generation_config.json` の `eos_token_id`（ADR 0083 決定 8）。 */
const STOP_TOKENS = [1, 106, 50] as const;

/** 実行条件は既存の門と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 768;
const CAPACITY_SYMBOL = "C";
const CAPACITY = 4096;
const MAX_POSITION = 131072;

/** RoPE のパラメータ（配布形の宣言と同じ値 — 表は資産に無く cos / sin はホストが作る）。 */
const ROPE: Gemma4RopeSpec = {
  sliding_attention: { theta: 10000, headDim: 256, rotaryDim: 256 },
  full_attention: { theta: 1000000, headDim: 512, rotaryDim: 128 },
};

/** グラフ入力の名前（正本は `export_product` の定数 — `position_ids` はもう無い）。 */
const INPUT_IDS = "input_ids";
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
    `[karume] 製品系列（${PRODUCT_ROOT.pathname}）が無いため Gemma 4 E2B sampler 検収を ` +
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

/** i32 の入力テンソル 1 本（token id 列も絶対位置列も `[1, rows]`）。 */
const i32Row = (rows: number, data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, rows],
  data,
});

/** token-only 形と同じ行選択入力（`[1]` の i32）。 */
const lastRowInput = (row: number): Tensor => ({
  dtype: "i32",
  shape: [1],
  data: Int32Array.of(row),
});

/**
 * 最終行 logits `[1,1,V]` の生データ。
 *
 * MUST: dtype と shape を検査する（出力名の取り違えは例外も警告も出ないまま別の列を返す）。
 * 検査を通すと union が `f32` に絞れるので、sampler へ `Float32Array` をそのまま渡せる。
 */
const lastRowLogits = (tensor: Tensor, where: string): Float32Array<ArrayBuffer> => {
  if (tensor.dtype !== "f32") {
    throw new Error(`${where}: logits の dtype が ${tensor.dtype}（f32 でない）`);
  }
  assertEquals(tensor.shape, [1, 1, VOCAB], `${where}: logits の shape`);
  return tensor.data;
};

/**
 * ホスト PLE gather + **ホスト sampler** の生成ループ。
 *
 * `history` には prompt を含む「それまでの token 列」を渡す（repetition penalty の契約どおりの
 * 呼び方 — 温度 0 の指定では読まれないが、実運用と同じ形で通す）。
 */
const generate = async (
  session: Awaited<ReturnType<PreparedModel["createSession"]>>,
  logitsName: string,
  ple: Gemma4Ple,
  spec: SamplerSpec,
  prompt: readonly number[],
  maxNewTokens: number,
): Promise<number[]> => {
  const chunks = planPrefillChunks(prompt.length, CHUNK_LENGTH);
  const lastPosition = prompt.length + maxNewTokens - 2;
  assert(lastPosition < MAX_POSITION, `最終位置 ${lastPosition} がモデルの位置上限の外`);
  assert(prompt.length + maxNewTokens <= CAPACITY, `T + K が容量 ${CAPACITY} を超える`);

  const sampler = createSampler(spec);
  const history = [...prompt];
  const context = await session.createGenerationContext({
    bindings: { [CAPACITY_SYMBOL]: CAPACITY },
    chunkLength: CHUNK_LENGTH,
  });
  try {
    let token = 0;
    for (const chunk of chunks) {
      const ids = new Int32Array(CHUNK_LENGTH);
      const positions = new Int32Array(CHUNK_LENGTH);
      for (let row = 0; row < chunk.queryLength; row += 1) {
        ids[row] = prompt[chunk.position + row];
        positions[row] = chunk.position + row;
      }
      const outputs = await session.run(
        {
          [INPUT_IDS]: i32Row(CHUNK_LENGTH, ids),
          [PER_LAYER_INPUTS]: await ple.gather([...ids]),
          ...gemma4RopeInputs(ROPE, positions),
          [LAST_ROW]: lastRowInput(chunk.queryLength - 1),
        },
        undefined,
        { context, queryLength: chunk.queryLength },
      );
      token = sampler.next(
        lastRowLogits(outputs[logitsName], `prefill@${chunk.position}`),
        history,
      );
    }

    const generated = [token];
    history.push(token);
    for (let index = 0; index + 1 < maxNewTokens; index += 1) {
      const current = generated[index];
      const outputs = await session.run(
        {
          [INPUT_IDS]: i32Row(1, Int32Array.of(current)),
          [PER_LAYER_INPUTS]: await ple.gather([current]),
          ...gemma4RopeInputs(ROPE, [prompt.length + index]),
          [LAST_ROW]: lastRowInput(0),
        },
        undefined,
        { context, queryLength: 1 },
      );
      const next = sampler.next(lastRowLogits(outputs[logitsName], `decode@${index}`), history);
      generated.push(next);
      history.push(next);
    }
    return generated;
  } finally {
    // MUST: 途中で落ちても返す（KV 容量ぶんの VRAM を抱えたままにしない）。
    await context.dispose();
  }
};

Deno.test({
  name: "Gemma 4 E2B sampler 検収: 温度 0 の parity と EOS 集合の停止判定（実 GPU）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const shards = resolveShards(new URL(MODEL_FILE, PRODUCT_ROOT));
    const parsed = prepareModel(await readShard(shards[0]));
    const logitsName = parsed.graph.outputs[0];

    const index = parseGemma4PleIndex(
      JSON.parse(await Deno.readTextFile(new URL(PLE_INDEX_FILE, PRODUCT_ROOT))),
    );
    const ple = createGemma4Ple({
      index,
      readShard: (file) => readBuffer(PRODUCT_ROOT, file),
      vocabSize: VOCAB,
      // 全 shard 常駐（生成の往復で読み直さない）= sidecar 全量ぶんの予算。
      maxResidentBytes: index.shards.reduce(
        (sum, shard) => sum + gemma4PleShardBytes(index, shard),
        0,
      ),
    });

    const gpu = await acquireGpu();
    const session = await parsed.createSession(gpu, streamShards(shards.slice(1)));
    try {
      await t.step("① 温度 0 の sampler で 3 ケース × 16 step が期待列と厳密一致", async () => {
        for (const { name, firstStop } of EXPECTED_CASES) {
          const golden = parseSafetensors(
            await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${name}${SUFFIX}`),
          );
          const prompt = [...goldenI32(golden, "prompt")];
          const expected = [...goldenI32(golden, "expected")];
          assertEquals(expected.length, GREEDY_STEPS, `${name}: golden の step 数`);

          const started = performance.now();
          // 指定なし = 温度 0（この層の既定 — ADR 0083 決定 7）。
          const generated = await generate(session, logitsName, ple, {}, prompt, GREEDY_STEPS);
          assertEquals(
            generated,
            expected,
            `${name}: sampler 経由の生成 token 列（対 logits opt-in 系列 golden）`,
          );

          // ② 停止判定を実出力へ当てる（結線は段 3 — ここは判定だけ）。
          const stopAt = generated.findIndex((token) => isStopToken(token, STOP_TOKENS));
          assertEquals(
            stopAt,
            firstStop,
            `${name}: EOS 集合 [${STOP_TOKENS.join(",")}] が最初に成立する添字`,
          );
          console.log(
            `[e2e] gemma4 sampler ${name}: T=${prompt.length} + K=${GREEDY_STEPS} step / ` +
              `停止 @${stopAt} / ${(performance.now() - started).toFixed(0)}ms`,
          );
        }
      });

      await t.step("① topK = 1 も実寸の語彙（V = 262,144）で greedy と一致する", async () => {
        const { name } = EXPECTED_CASES[0];
        const golden = parseSafetensors(
          await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${name}${SUFFIX}`),
        );
        const generated = await generate(
          session,
          logitsName,
          ple,
          // 候補が最上位 1 件へ縮退するので、seed を与えても列は動かない。
          { temperature: 1, topK: 1, seed: 20_260_831 },
          [...goldenI32(golden, "prompt")],
          GREEDY_STEPS,
        );
        assertEquals(
          generated,
          [...goldenI32(golden, "expected")],
          `${name}: topK 1 / 温度 1 の生成 token 列（greedy と同じであること）`,
        );
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

// 実重み Gemma 4 E2B の **token-only 既定出口**（ADR 0068 決定 4）の実 GPU 検収門。
//
// `e2e_gemma4_greedy_test.ts`（logits opt-in 形 — `[logits, token]` の 2 出力）が生成ループと
// KV 機構の本体を受け持つのに対し、こちらは**出口の差し替えだけ**を検収する:
//
// ① 形の前提（入力 3 本 = input_ids / position_ids / **last_row**・出力 1 本 = argmax token・
//    states 30 スロットは opt-in 形と同一）
// ② **系列間交差 parity**: token-only 系列の greedy が logits opt-in 系列の
//    `greedy.<case>.safetensors`（torch full re-forward の期待列）と 3 ケース × K=16 で
//    厳密一致する。同じ重み・同じ丸め・同じ手術で出口だけが違うので、列が割れたら
//    行選択（last_row 配線）か 1 行 lm_head の側の誤り。
//
// 期待列の正本を**再計算しない**のがこの門の設計 — logits opt-in 系列の golden をそのまま
// 流用することで、「両系列が同じ列を吐く」という交差検証そのものが門になる（torch 参照の
// 再実走〈数十分〉を払わず、しかも独立性は落ちない — 期待値の出所は full re-forward のまま）。
//
// 資産 2 系列を両方要求する: `outputs/series/gemma4-e2b-decode-token/`（model 本体 —
// `tools/export-recipes/gemma4/export_token.py`）と `outputs/series/gemma4-e2b-decode/`
// （greedy golden — `export_decode.py`）。どちらかが無い環境では**明示 SKIP** する。

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  createSession,
  type KarumeModel,
  openModel,
  parseSafetensors,
  type SafetensorsFile,
} from "@karume/runtime";
import { generateGreedy } from "../src/generation/greedy.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const TOKEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-decode-token/", import.meta.url);
const GOLDEN_ROOT = new URL("../../../outputs/series/gemma4-e2b-decode/", import.meta.url);
const MODEL_FILE = "model.safetensors";
const GREEDY_PREFIX = "greedy.";
const SUFFIX = ".safetensors";

/** SKIP 時にそのまま貼れる生成コマンド。 */
const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --with 'transformers==5.14.1' python -m gemma4.export_token" +
  "（golden 側は … python -m gemma4.export_decode）";

/** 期待列の正本 = logits opt-in 系列の greedy golden（正典は export_decode.GREEDY_CASES）。 */
const EXPECTED_CASES = ["capital-en", "capital-ja", "context-en"] as const;
const GREEDY_STEPS = 16;

/** 実行条件は e2e_gemma4_greedy_test.ts と同値（同じ資産世代の裁定をそのまま使う）。 */
const CHUNK_LENGTH = 32;
const CAPACITY_SYMBOL = "C";
const CAPACITY = 640;
const MAX_POSITION = 1024;

const INPUT_IDS = "input_ids";
const POSITION_IDS = "position_ids";
/** token-only 形の行選択入力（正本は export_decode.TOKEN_ONLY_LAST_ROW）。 */
const LAST_ROW = "last_row";

const exists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const AVAILABLE = exists(new URL(MODEL_FILE, TOKEN_ROOT)) &&
  EXPECTED_CASES.every((name) => exists(new URL(`${GREEDY_PREFIX}${name}${SUFFIX}`, GOLDEN_ROOT)));

if (!AVAILABLE) {
  console.warn(
    `[karume] token-only 系列（${TOKEN_ROOT.pathname}）か greedy golden（${GOLDEN_ROOT.pathname}）` +
      `が無いため Gemma 4 E2B token-only 検収を SKIP する。生成: ${GENERATE_COMMAND}`,
  );
}

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

/**
 * ① 形の前提。opt-in 形との差分（入力 +1 本・出力 −1 本・argmax 直結）だけをここで見る —
 * states / 層種別 / 格納の本体検査は書き手側（`export_decode.assert_ir_form_decode` の
 * `token_only=True`）と opt-in 門が持つ。
 */
const assertTokenOnlyForm = (parsed: KarumeModel): void => {
  const graph = parsed.graph;
  assertEquals(
    graph.inputs.map((spec) => spec.name),
    [INPUT_IDS, POSITION_IDS, LAST_ROW],
    "グラフ入力（token-only は last_row が増える）",
  );
  assertEquals(graph.outputs.length, 1, "graph.outputs の本数（token 1 本 — logits は出さない）");
  const producer = new Map<string, string>();
  for (const node of graph.nodes) {
    for (const out of node.outs) producer.set(out, node.op);
  }
  assertEquals(producer.get(graph.outputs[0]), "argmax", "出力 0 の供給元（argmax 直結）");
  assertEquals(
    Object.keys(graph.states).length,
    30,
    "states スロットの本数（opt-in 形と同一の 30 本）",
  );
};

Deno.test({
  name: "Gemma 4 E2B token-only 検収: 系列間交差 parity（実 GPU / opt-in 系列の期待列）",
  ignore: !AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const parsed = openModel(await readBuffer(TOKEN_ROOT, MODEL_FILE));
    const tokenName = parsed.graph.outputs[0];

    await t.step("① 形の前提: token-only 出口である", () => {
      assertTokenOnlyForm(parsed);
    });

    const gpu = await acquireGpu();
    const session = await createSession(gpu, parsed);
    try {
      await t.step("② 3 ケース × 16 step が opt-in 系列の期待列と厳密一致", async () => {
        for (const caseName of EXPECTED_CASES) {
          const golden = parseSafetensors(
            await readBuffer(GOLDEN_ROOT, `${GREEDY_PREFIX}${caseName}${SUFFIX}`),
          );
          const prompt = [...goldenI32(golden, "prompt")];
          const expected = [...goldenI32(golden, "expected")];
          assertEquals(expected.length, GREEDY_STEPS, `${caseName}: golden の step 数`);

          const started = performance.now();
          const generated = await generateGreedy({
            session,
            inputIds: INPUT_IDS,
            positionIds: POSITION_IDS,
            token: tokenName,
            lastRow: LAST_ROW,
            chunkLength: CHUNK_LENGTH,
            maxPosition: MAX_POSITION,
            bindings: { [CAPACITY_SYMBOL]: CAPACITY },
            prompt,
            maxNewTokens: GREEDY_STEPS,
          });
          assertEquals(generated, expected, `${caseName}: 生成 token 列（対 opt-in 系列 golden）`);
          console.log(
            `[e2e] gemma4 token-only ${caseName}: T=${prompt.length} + K=${GREEDY_STEPS} step / ` +
              `${(performance.now() - started).toFixed(0)}ms`,
          );
        }
      });
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
});

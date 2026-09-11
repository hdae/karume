/** 全行 logits を持つ実験グラフの、EOS 停止つき逐次 greedy 生成。 */
import type { PreparedModel, RunOutputs, Tensor } from "../../packages/runtime/mod.ts";
import {
  type GreedySession,
  planPrefillChunks,
} from "../../packages/models/src/generation/greedy.ts";
import { llmProfile } from "./llm-source.ts";
import type { LlmFamily } from "./llm-tokenizer.ts";

export const LLM_CAPACITY = 128;
export const LLM_CHUNK_LENGTH = 64;
export type LlmGraph = {
  readonly token: string;
  readonly vocabSize: number;
  readonly maxPosition: number;
};

export const inspectLlmGraph = (family: LlmFamily, graph: PreparedModel["graph"]): LlmGraph => {
  const profile = llmProfile(family);
  const same = (actual: unknown, expected: unknown): boolean =>
    JSON.stringify(actual) === JSON.stringify(expected);
  if (
    !same(graph.inputs, [{ name: "input_ids", dtype: "i32", shape: [1, "M"] }, {
      name: "position_ids",
      dtype: "i32",
      shape: [1, "M"],
    }]) || graph.outputs.length !== 2
  ) throw new Error("input_ids / position_ids → 全行 logits / token の実験グラフが必要です");
  const [logits, token] = graph.outputs;
  if (
    !same(graph.values[logits], { dtype: "f32", shape: [1, "M", profile.vocabSize] }) ||
    !same(graph.values[token], { dtype: "i32", shape: [1, "M", 1] })
  ) throw new Error(`${profile.name} の出力宣言と一致しません`);
  const states = Object.values(graph.states ?? {});
  if (
    states.length !== profile.layers * 2 ||
    states.some((state) =>
      state.dtype !== "f32" || state.external || !same(state.shape, [1, profile.kvHeads, "C", 128])
    )
  ) throw new Error(`${profile.name} の KV キャッシュ宣言と一致しません`);
  const tables = Object.entries(graph.initializers).filter(([, init]) =>
    typeof init.tensor === "string" && /rotary_emb\.(cos|sin)_table$/.test(init.tensor)
  ).map(([name]) => graph.values[name]);
  const maxPosition = tables[0]?.shape[0];
  if (
    tables.length !== 2 || typeof maxPosition !== "number" || !Number.isSafeInteger(maxPosition) ||
    maxPosition < LLM_CAPACITY ||
    tables.some((table) => !same(table, { dtype: "f32", shape: [maxPosition, 128] }))
  ) throw new Error("RoPE の位置表が対応する形ではありません");
  return { token, vocabSize: profile.vocabSize, maxPosition };
};

export const checkLlmRequest = (
  prompt: readonly number[],
  maxNewTokens: number,
  graph: LlmGraph,
): void => {
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1) {
    throw new Error("--max-new-tokens は 1 以上の整数が必要です");
  }
  if (
    prompt.length === 0 ||
    prompt.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= graph.vocabSize)
  ) throw new Error("入力 token が空または語彙の範囲外です");
  const positions = prompt.length + maxNewTokens - 1;
  if (positions > Math.min(LLM_CAPACITY, graph.maxPosition)) {
    throw new Error(
      `入力 ${prompt.length} token + 生成上限 ${maxNewTokens} token は、この実験モデルの容量 ${LLM_CAPACITY} を超えます。入力または --max-new-tokens を短くしてください。`,
    );
  }
};

const row = (data: Int32Array<ArrayBuffer>): Tensor => ({
  dtype: "i32",
  shape: [1, data.length],
  data,
});
const readToken = (outputs: RunOutputs, graph: LlmGraph, rows: number, at: number): number => {
  const output = outputs[graph.token];
  if (
    output === undefined || output.dtype !== "i32" || output.shape.length !== 3 ||
    output.shape[0] !== 1 || output.shape[1] !== rows || output.shape[2] !== 1
  ) throw new Error("生成 token の形が不正です");
  const token = output.data[at];
  if (!Number.isSafeInteger(token) || token < 0 || token >= graph.vocabSize) {
    throw new Error(`生成 token ${token} が語彙の範囲外です`);
  }
  return token;
};

/** EOS も 1 回返す。呼び手はその ID を記録できるが本文へは復号しない。 */
export async function* streamLlm<
  C extends { readonly pastLength: number; dispose(): Promise<void> },
>(
  session: GreedySession<C>,
  graph: LlmGraph,
  prompt: readonly number[],
  maxNewTokens: number,
  stopTokens: readonly number[],
  signal?: AbortSignal,
): AsyncGenerator<number> {
  checkLlmRequest(prompt, maxNewTokens, graph);
  signal?.throwIfAborted();
  const context = await session.createGenerationContext({
    bindings: { C: LLM_CAPACITY },
    chunkLength: LLM_CHUNK_LENGTH,
  });
  // 解放も失敗した場合は SuppressedError に両方を残す（runMain が展開する）。
  await using _release = { [Symbol.asyncDispose]: () => context.dispose() };
  let token = 0;
  for (const chunk of planPrefillChunks(prompt.length, LLM_CHUNK_LENGTH)) {
    signal?.throwIfAborted();
    const ids = new Int32Array(LLM_CHUNK_LENGTH);
    const positions = new Int32Array(LLM_CHUNK_LENGTH);
    for (let at = 0; at < chunk.queryLength; at++) {
      ids[at] = prompt[chunk.position + at];
      positions[at] = chunk.position + at;
    }
    const outputs = await session.run(
      { input_ids: row(ids), position_ids: row(positions) },
      undefined,
      { context, queryLength: chunk.queryLength },
    );
    token = readToken(outputs, graph, LLM_CHUNK_LENGTH, chunk.queryLength - 1);
  }
  for (let step = 0; step < maxNewTokens; step++) {
    signal?.throwIfAborted();
    yield token;
    if (stopTokens.includes(token) || step + 1 === maxNewTokens) return;
    signal?.throwIfAborted();
    const outputs = await session.run(
      {
        input_ids: row(Int32Array.of(token)),
        position_ids: row(Int32Array.of(context.pastLength)),
      },
      undefined,
      { context, queryLength: 1 },
    );
    token = readToken(outputs, graph, 1, 0);
  }
}

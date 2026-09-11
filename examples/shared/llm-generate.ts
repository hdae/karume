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

export class LlmCapacityError extends Error {}

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
    throw new LlmCapacityError(
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

export type LlmContext = { readonly pastLength: number; dispose(): Promise<void> };
export type LlmPrefill = {
  readonly chunk: number;
  readonly chunks: number;
  readonly reusedTokens: number;
};

/**
 * 実験 CLI の会話キャッシュ。公式テンプレートを描き直した全 prompt と、実際に commit 済みの
 * token 列の一致を条件に継ぐ。Qwen の過去 thinking の除去や再符号化で列が変われば作り直す。
 * rewind は使わない（ADR 0083 決定 4）。利用は CLI の直列ターンに限定する。
 */
export class LlmSequence<C extends LlmContext> {
  readonly #session: GreedySession<C>;
  readonly #graph: LlmGraph;
  #cache: { context: C; tokens: readonly number[] } | undefined;
  #busy = false;
  #closed = false;

  constructor(session: GreedySession<C>, graph: LlmGraph) {
    this.#session = session;
    this.#graph = graph;
  }

  async reset(): Promise<void> {
    if (this.#busy) throw new Error("生成中は会話を reset / dispose できません");
    const cache = this.#cache;
    this.#cache = undefined;
    await cache?.context.dispose();
  }

  async dispose(): Promise<void> {
    if (this.#busy) throw new Error("生成中は会話を dispose できません");
    this.#closed = true;
    await this.reset();
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  /** EOS を含む ID 列。中断・生成上限・例外では未閉鎖のキャッシュを返却する。 */
  async *stream(
    prompt: readonly number[],
    maxNewTokens: number,
    stopTokens: readonly number[],
    signal?: AbortSignal,
    onPrefill?: (progress: LlmPrefill) => void,
  ): AsyncGenerator<number> {
    if (this.#closed) throw new Error("dispose 済みの会話では生成できません");
    if (this.#busy) throw new Error("同じ会話で並行生成はできません");
    checkLlmRequest(prompt, maxNewTokens, this.#graph);
    signal?.throwIfAborted();
    this.#busy = true;
    try {
      let cache = this.#cache;
      this.#cache = undefined;
      if (
        cache !== undefined &&
        (cache.tokens.length >= prompt.length ||
          !cache.tokens.every((token, at) => token === prompt[at]))
      ) {
        await cache.context.dispose();
        cache = undefined;
      }
      const context = cache?.context ?? await this.#session.createGenerationContext({
        bindings: { C: LLM_CAPACITY },
        chunkLength: LLM_CHUNK_LENGTH,
      });
      let retained = false;
      // 本体と解放の両方が失敗したら SuppressedError に残す。
      await using _release = {
        [Symbol.asyncDispose]: async (): Promise<void> => {
          if (!retained) await context.dispose();
        },
      };
      const reusedTokens = context.pastLength;
      if (reusedTokens !== (cache?.tokens.length ?? 0)) {
        throw new Error("会話 token 列と KV キャッシュの長さが一致しません");
      }
      const chunks = planPrefillChunks(prompt.length - reusedTokens, LLM_CHUNK_LENGTH);
      let token = 0;
      for (const [index, chunk] of chunks.entries()) {
        signal?.throwIfAborted();
        const ids = new Int32Array(LLM_CHUNK_LENGTH);
        const positions = new Int32Array(LLM_CHUNK_LENGTH);
        const past = context.pastLength;
        for (let at = 0; at < chunk.queryLength; at++) {
          ids[at] = prompt[past + at];
          positions[at] = past + at;
        }
        const outputs = await this.#session.run(
          { input_ids: row(ids), position_ids: row(positions) },
          undefined,
          { context, queryLength: chunk.queryLength },
        );
        token = readToken(outputs, this.#graph, LLM_CHUNK_LENGTH, chunk.queryLength - 1);
        onPrefill?.({ chunk: index + 1, chunks: chunks.length, reusedTokens });
      }
      const generated: number[] = [];
      for (let step = 0; step < maxNewTokens; step++) {
        signal?.throwIfAborted();
        generated.push(token);
        yield token;
        signal?.throwIfAborted();
        if (stopTokens.includes(token)) {
          // 最後に配送した EOS は未 commit。次ターンの全 prompt に含まれ、差分の先頭から入る。
          this.#cache = { context, tokens: [...prompt, ...generated.slice(0, -1)] };
          retained = true;
          return;
        }
        if (step + 1 === maxNewTokens) return;
        const outputs = await this.#session.run(
          {
            input_ids: row(Int32Array.of(token)),
            position_ids: row(Int32Array.of(context.pastLength)),
          },
          undefined,
          { context, queryLength: 1 },
        );
        token = readToken(outputs, this.#graph, 1, 0);
      }
    } finally {
      this.#busy = false;
    }
  }
}

/** 単発生成は同じループを使い、読み終わった時点で EOS 後のキャッシュも返す。 */
export async function* streamLlm<C extends LlmContext>(
  session: GreedySession<C>,
  graph: LlmGraph,
  prompt: readonly number[],
  maxNewTokens: number,
  stopTokens: readonly number[],
  signal?: AbortSignal,
): AsyncGenerator<number> {
  await using sequence = new LlmSequence(session, graph);
  yield* sequence.stream(prompt, maxNewTokens, stopTokens, signal);
}

/** 実験 CLI の履歴管理。公開 pipeline を追加せず、対になった発話と実行用キャッシュを束ねる。 */
import {
  checkLlmRequest,
  LLM_CAPACITY,
  type LlmContext,
  type LlmGraph,
  type LlmPrefill,
  LlmSequence,
} from "./llm-generate.ts";
import type { LlmTokenizer, LlmTurn } from "./llm-tokenizer.ts";

export type LlmReply = {
  readonly promptTokens: readonly number[];
  readonly tokens: readonly number[];
  readonly text: string;
  readonly stop: "eos" | "length" | "aborted";
  readonly reusedTokens: number;
  readonly droppedTurns: number;
};

/** 超過時は古い user / assistant の対を外す。system と今回の質問は必ず保持する。 */
export const prepareLlmChat = (
  tokenizer: LlmTokenizer,
  graph: LlmGraph,
  turns: readonly LlmTurn[],
  prompt: string,
  maxNewTokens: number,
  system?: string,
): { turns: readonly LlmTurn[]; ids: number[]; droppedTurns: number } => {
  if (prompt.length === 0) throw new Error("入力文が空です");
  let droppedTurns = 0;
  let kept = turns.slice();
  let ids = tokenizer.chat(prompt, system, kept);
  while (
    ids.length + maxNewTokens - 1 > Math.min(LLM_CAPACITY, graph.maxPosition) && kept.length > 0
  ) {
    kept = kept.slice(1);
    droppedTurns++;
    ids = tokenizer.chat(prompt, system, kept);
  }
  checkLlmRequest(ids, maxNewTokens, graph);
  return { turns: kept, ids, droppedTurns };
};

/** send / reset / dispose は CLI が直列に呼ぶ。描画済みの部分回答も次ターンの履歴に残す。 */
export class LlmChat<C extends LlmContext> {
  readonly #sequence: LlmSequence<C>;
  readonly #tokenizer: LlmTokenizer;
  readonly #graph: LlmGraph;
  readonly #maxNewTokens: number;
  readonly #system: string | undefined;
  #turns: readonly LlmTurn[] = [];
  #busy = false;

  constructor(
    sequence: LlmSequence<C>,
    tokenizer: LlmTokenizer,
    graph: LlmGraph,
    maxNewTokens: number,
    system?: string,
  ) {
    this.#sequence = sequence;
    this.#tokenizer = tokenizer;
    this.#graph = graph;
    this.#maxNewTokens = maxNewTokens;
    this.#system = system;
  }

  get turns(): readonly LlmTurn[] {
    return this.#turns.map((turn) => ({ ...turn }));
  }

  async reset(): Promise<void> {
    if (this.#busy) throw new Error("生成中は会話を reset できません");
    await this.#sequence.reset();
    this.#turns = [];
  }

  async send(
    prompt: string,
    onText: (text: string) => void,
    options: {
      readonly signal?: AbortSignal;
      readonly onPrefill?: (progress: LlmPrefill) => void;
      readonly onOverflow?: (droppedTurns: number) => void;
    } = {},
  ): Promise<LlmReply> {
    if (this.#busy) throw new Error("同じ会話で並行生成はできません");
    const { signal } = options;
    signal?.throwIfAborted();
    const plan = prepareLlmChat(
      this.#tokenizer,
      this.#graph,
      this.#turns,
      prompt,
      this.#maxNewTokens,
      this.#system,
    );
    const tokens: number[] = [];
    let text = "";
    let stop: LlmReply["stop"] = "length";
    let reusedTokens = 0;
    const decoder = this.#tokenizer.decoder();
    const write = (chunk: string): void => {
      if (chunk === "") return;
      onText(chunk);
      text += chunk;
    };
    this.#busy = true;
    try {
      if (plan.droppedTurns > 0) options.onOverflow?.(plan.droppedTurns);
      try {
        for await (
          const token of this.#sequence.stream(
            plan.ids,
            this.#maxNewTokens,
            this.#tokenizer.stopTokens,
            signal,
            (progress) => {
              reusedTokens = progress.reusedTokens;
              options.onPrefill?.(progress);
            },
          )
        ) {
          tokens.push(token);
          if (this.#tokenizer.stopTokens.includes(token)) stop = "eos";
          else write(decoder.push(token));
        }
      } catch (error) {
        if (!signal?.aborted || error !== signal.reason) throw error;
        stop = "aborted";
      }
      write(decoder.finish());
      return {
        promptTokens: plan.ids,
        tokens,
        text,
        stop,
        reusedTokens,
        droppedTurns: plan.droppedTurns,
      };
    } finally {
      // 未出力で中断した質問は残さない。EOS の空回答は完了した発話として保持する。
      if (text !== "" || stop === "eos") {
        this.#turns = [...plan.turns, { user: prompt, assistant: text }];
      }
      this.#busy = false;
    }
  }
}

/** stdin を行で読む。パイプの末尾行や UTF-8 が chunk を跨ぐ場合も同じ操作にする。 */
export async function* readLlmLines(
  input: ReadableStream<Uint8Array<ArrayBuffer>>,
): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of input) {
    buffer += decoder.decode(chunk, { stream: true });
    for (;;) {
      const end = buffer.indexOf("\n");
      if (end < 0) break;
      yield buffer.slice(0, end).replace(/\r$/, "");
      buffer = buffer.slice(end + 1);
    }
  }
  buffer += decoder.decode();
  if (buffer !== "") yield buffer.replace(/\r$/, "");
}

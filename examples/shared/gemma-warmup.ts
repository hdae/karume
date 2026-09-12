/** Gemma デモの主要な prefill 形と decode を暖機し、生成状態は全て解放する。 */
import type { Gemma4ChatSessionHost, SamplerSpec } from "../../packages/models/gemma.ts";

export const warmupGemma = async (
  host: Gemma4ChatSessionHost,
  capacity: number,
  sampler?: SamplerSpec,
): Promise<number> => {
  const bos = host.tokenizer.addedTokenId("<bos>");
  if (bos === undefined) throw new Error("Gemma tokenizer に <bos> がありません");
  let tokens = 0;
  // 最大 chunk の prefill と短い decode を別 sequence で行う。最小容量でも両者が収まる。
  for (const [length, count] of [[host.program.chunkLength, 1], [1, Math.min(4, capacity)]]) {
    const sequence = await host.sequence({ capacity });
    await using _release = { [Symbol.asyncDispose]: () => sequence.dispose() };
    const stream = sequence.generate({
      prompt: Array.from({ length }, () => bos),
      maxNewTokens: count,
      ...(sampler === undefined ? {} : { sampler }),
    });
    for await (const event of stream) if (event.kind === "token") tokens += 1;
    await stream.done;
  }
  return tokens;
};

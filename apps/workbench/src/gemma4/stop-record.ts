/**
 * How a Gemma 4 chat stopped, flattened for JSON. Copied from the Playground's chat page record
 * (`apps/playground/src/models/gemma4/chat-record.ts`, `stopRecordOf`), which keeps its own: the
 * workbench and the Playground share no module. Pure, so `vp test` covers it.
 */

import type { ChatStop, GenerationDecodeAhead, GenerationSpeculation } from "@karume/models/gemma";

/** How a chat stopped (the stop of the chat's stream, flattened for JSON). */
export type StopRecord = {
  readonly reason: ChatStop["reason"];
  /** Generated tokens, a stop token counted as one. */
  readonly tokens: number;
  /** The stop token (`eos` / `stop-token`). */
  readonly token?: number;
  readonly stopString?: string;
  /** Present only on chats of a pipeline with the MTP drafter. */
  readonly speculation?: GenerationSpeculation;
  /**
   * Present only on chats of a pipeline built with `decodeAhead` ≥ 2: the depth, the batches the
   * chat ran (0 = it decoded one step per batch) and the steps they computed.
   */
  readonly decodeAhead?: GenerationDecodeAhead;
};

export const stopRecordOf = (stop: ChatStop): StopRecord => ({
  reason: stop.reason,
  tokens: stop.tokens,
  ...("token" in stop ? { token: stop.token } : {}),
  ...(stop.reason === "stop-string" ? { stopString: stop.stopString } : {}),
  ...(stop.speculation === undefined ? {} : { speculation: stop.speculation }),
  ...(stop.decodeAhead === undefined ? {} : { decodeAhead: stop.decodeAhead }),
});

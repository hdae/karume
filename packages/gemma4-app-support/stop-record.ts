/**
 * `@karume/gemma4-app-support/stop-record` — how a Gemma 4 generation stopped (the stop of its text
 * stream), flattened for JSON: the stop of a turn in the playground's chat record and of a chat in
 * the workbench's decode bench. Pure, so `vp test` covers it.
 */

import type {
  ChatStop,
  GenerationCandidateCounts,
  GenerationDecodeAhead,
  GenerationSpeculation,
} from "@karume/models/gemma";

/** How a generation stopped (the stop of its stream, flattened for JSON). */
export type StopRecord = {
  readonly reason: ChatStop["reason"];
  /** Generated tokens, a stop token counted as one. */
  readonly tokens: number;
  /** The stop token (`eos` / `stop-token`). */
  readonly token?: number;
  readonly stopString?: string;
  /** Present only on generations of a pipeline with the MTP drafter. */
  readonly speculation?: GenerationSpeculation;
  /**
   * Present only on generations of a pipeline built with `decodeAhead` ≥ 2: the depth, the batches
   * the generation ran (0 = it decoded one step per batch) and the steps they computed.
   */
  readonly decodeAhead?: GenerationDecodeAhead;
  /**
   * Present on generations that sample on a pipeline that has the candidate selection: the leading
   * logits a sampled decode step read back in place of the whole row (0 = the generation did not
   * use the selection), the steps drawn from them, and the steps that read the whole row after all.
   */
  readonly candidateSelection?: GenerationCandidateCounts;
};

export const stopRecordOf = (stop: ChatStop): StopRecord => ({
  reason: stop.reason,
  tokens: stop.tokens,
  ...("token" in stop ? { token: stop.token } : {}),
  ...(stop.reason === "stop-string" ? { stopString: stop.stopString } : {}),
  ...(stop.speculation === undefined ? {} : { speculation: stop.speculation }),
  ...(stop.decodeAhead === undefined ? {} : { decodeAhead: stop.decodeAhead }),
  ...(stop.candidateSelection === undefined ? {} : { candidateSelection: stop.candidateSelection }),
});

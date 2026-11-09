/**
 * The Gemma 4 chat record (`format: "karume-playground-gemma4/3"`): the environment, the
 * distribution and source, the loaded model and preset, the chat settings, every turn with its
 * request, stop, statistics and generated token ids, and the Session diagnostics when the pipeline was
 * loaded with them. Pure (the
 * environment is passed in), so `vp test` covers it.
 */

import type {
  ChatJsonObject,
  ChatStop,
  ExpertCacheReport,
  GenerationDecodeAhead,
  GenerationSpeculation,
  SamplerSpec,
} from "@karume/models/gemma";

import type { RecordEnvironment } from "@/lib/environment";
import type { ComponentRecord, SourceRecord } from "@/lib/record";

import { type ChatSettings, entryPointOf, type GemmaEntryPoint, type GemmaMount } from "./settings";
import type { TurnTiming } from "./turns";

/**
 * The version of the saved record; bump it when a field changes meaning or goes away.
 *
 * Version 3: `distribution.mount` `gemma4-qat` names the q4_0 QAT line (read by `gemma.Pipeline`),
 * and the mobile QAT line, which version 2 recorded as mount `gemma4-qat` with pipeline
 * `gemma4Qat.Pipeline`, is mount `gemma4-qat-mobile` with pipeline `gemma4QatMobile.Pipeline`.
 *
 * Version 4: `request.maxNewTokens` is absent on a turn sent without a limit, and such a turn can
 * stop with the reason `capacity`.
 */
export const RECORD_FORMAT = "karume-playground-gemma4/4";

/** How a turn stopped (the stop of the turn's stream, flattened for JSON). */
export type StopRecord = {
  readonly reason: ChatStop["reason"];
  /** Generated tokens, a stop token counted as one. */
  readonly tokens: number;
  /** The stop token (`eos` / `stop-token`). */
  readonly token?: number;
  readonly stopString?: string;
  /** Present only on turns of a pipeline with the MTP drafter. */
  readonly speculation?: GenerationSpeculation;
  /**
   * Present only on turns of a pipeline built with `decodeAhead` ≥ 2: the depth, the batches the
   * turn ran (0 = it decoded one step per batch) and the steps they computed.
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

/** Runs reported by `onRunDiagnostics` during the turn, by phase (only with diagnostics). */
export type RunCounts = {
  readonly prefill: number;
  readonly decode: number;
  readonly draft: number;
  readonly verify: number;
};

export const NO_RUNS: RunCounts = { prefill: 0, decode: 0, draft: 0, verify: 0 };

/** One tool result a turn sent (`sendToolResults`). */
export type ToolResultRecord = {
  readonly toolCallId?: string;
  readonly name: string;
  readonly content: string;
};

/** One tool call a turn ended on (`reply.message.tool_calls`). */
export type ToolCallRecord = {
  readonly id?: string;
  readonly name: string;
  readonly arguments: ChatJsonObject;
};

/** One image a turn sent. */
export type ImageRecord = {
  readonly width: number;
  readonly height: number;
  readonly softTokens: number;
};

/**
 * One generation of the session: a user message, or — in a chat with tools — a continuation after
 * the results of the calls the previous one ended on (one row per `send` / `sendToolResults`, so the
 * statistics stay those of one prefill and one decode).
 */
export type TurnRecord = {
  /** 1-based, in the order sent (failed turns included). */
  readonly index: number;
  /** The user's message (a continuation repeats the message whose model turn it continues). */
  readonly userText: string;
  /**
   * Present only on a message with images: each one's size and soft tokens, in the order sent. The
   * pixels are not recorded: a record stays small and shareable.
   */
  readonly images?: readonly ImageRecord[];
  /** Present only on a continuation: the tool results it sent, in order. */
  readonly toolResults?: readonly ToolResultRecord[];
  /** Present only on a turn that ended on tool calls: the calls, with their session-assigned ids. */
  readonly toolCalls?: readonly ToolCallRecord[];
  /** The streamed text (empty for a turn that produced none). */
  readonly replyText: string;
  /** As passed: `maxNewTokens` is absent on a turn sent without a limit. */
  readonly request: { readonly maxNewTokens?: number; readonly sampler: SamplerSpec };
  readonly outcome:
    | { readonly kind: "stopped"; readonly stop: StopRecord }
    | { readonly kind: "failed"; readonly error: string };
  readonly prefill: {
    /** The prompt the session sent (absent when the turn failed before generating). */
    readonly promptTokens?: number;
    /** Positions the carried KV already held (0 = the history was rendered from the start). */
    readonly carriedPositions?: number;
    /** Prefill chunks committed. */
    readonly chunks: number;
  };
  readonly timing: TurnTiming;
  /** Every generated token id except a stop token (`onToken`), in order. */
  readonly generatedTokenIds: readonly number[];
  /** History entries the overflow policy dropped before this turn was sent. */
  readonly droppedEntries: number;
  /**
   * Whether the session kept the turn (its message, or its tool results, and the reply) in its
   * history. False when it treated the turn as never having happened: no text and no EOS (a Stop
   * during prefill, a failure).
   */
  readonly inHistory: boolean;
  /** Positions the session holds after the turn; absent when it released the sequence. */
  readonly kvUsedAfter?: number;
  /** Present only when the pipeline was loaded with diagnostics. */
  readonly runs?: RunCounts;
};

/** The Session diagnostics folded over the chat's runs: the target Session, and the drafter's. */
export type DiagnosticsRecord = {
  readonly target?: ComponentRecord;
  readonly drafter?: ComponentRecord;
};

export type GemmaRecord = RecordEnvironment & {
  readonly format: typeof RECORD_FORMAT;
  readonly page: "gemma4";
  readonly distribution: {
    readonly mount: GemmaMount;
    readonly directory: string;
    readonly manifestSha256: string;
  };
  readonly source: SourceRecord;
  /** The entry point: `Pipeline` (gemma4, gemma4-qat) or `gemma4QatMobile.Pipeline` (gemma4-qat-mobile). */
  readonly pipeline: GemmaEntryPoint;
  readonly model: string;
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  /** Whether the MTP drafter was built (`speculative` of `fromPretrained`). */
  readonly speculative: boolean;
  /** Present only when the full-layer KV was stored in f16 (`fullStateStorage`). */
  readonly fullStateStorage?: "f16";
  /** Present only when the image embedder was built (`vision` of `fromPretrained`). */
  readonly vision?: true;
  /**
   * Present only when the pipeline was built with the exact expert cache (`expertCache`), with its
   * host tier's budget when it had one (`hostBytes`; `Number.MAX_SAFE_INTEGER` = every expert).
   */
  readonly expertCache?: { readonly slots: number; readonly hostBytes?: number };
  /**
   * Present only with {@link expertCache}: `Pipeline.expertCacheReport()` after the latest turn
   * (cumulative over the pipeline's lifetime, so a "Load again" starts it over).
   */
  readonly expertCacheReport?: ExpertCacheReport;
  readonly loadSeconds: number;
  /**
   * Whether the pipeline was loaded with `onRunDiagnostics`. With it the pipeline skips its on-GPU
   * greedy selection (a temperature-0 turn reads the whole logits row back and picks on the CPU), so
   * the decode rates of such a record are not those of an app without diagnostics.
   */
  readonly diagnosticsCollected: boolean;
  readonly chat: ChatSettings;
  readonly turns: readonly TurnRecord[];
  /** Present only when {@link diagnosticsCollected}. */
  readonly diagnostics?: DiagnosticsRecord;
};

export const buildGemmaRecord = (input: {
  readonly environment: RecordEnvironment;
  readonly mount: GemmaMount;
  readonly directory: string;
  readonly manifestSha256: string;
  readonly source: SourceRecord;
  readonly model: string;
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  readonly speculative: boolean;
  readonly fullStateStorage?: "f16";
  readonly vision?: true;
  readonly expertCache?: { readonly slots: number; readonly hostBytes?: number };
  readonly expertCacheReport?: ExpertCacheReport;
  readonly loadSeconds: number;
  readonly chat: ChatSettings;
  readonly turns: readonly TurnRecord[];
  /** `undefined` = the pipeline was loaded without diagnostics. */
  readonly diagnostics: DiagnosticsRecord | undefined;
}): GemmaRecord => ({
  format: RECORD_FORMAT,
  page: "gemma4",
  ...input.environment,
  distribution: {
    mount: input.mount,
    directory: input.directory,
    manifestSha256: input.manifestSha256,
  },
  source: input.source,
  pipeline: entryPointOf(input.mount),
  model: input.model,
  preset: input.preset,
  speculative: input.speculative,
  ...(input.fullStateStorage === undefined ? {} : { fullStateStorage: input.fullStateStorage }),
  ...(input.vision === undefined ? {} : { vision: input.vision }),
  ...(input.expertCache === undefined ? {} : { expertCache: input.expertCache }),
  ...(input.expertCacheReport === undefined ? {} : { expertCacheReport: input.expertCacheReport }),
  loadSeconds: input.loadSeconds,
  diagnosticsCollected: input.diagnostics !== undefined,
  chat: input.chat,
  turns: input.turns,
  ...(input.diagnostics === undefined ? {} : { diagnostics: input.diagnostics }),
});

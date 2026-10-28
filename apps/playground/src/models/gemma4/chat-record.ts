/**
 * The Gemma 4 chat record (`format: "karume-playground-gemma4/1"`): the environment, the
 * distribution and source, the loaded model and preset, the chat settings, every turn with its
 * request, stop, statistics and generated token ids, and the Session diagnostics when the pipeline was
 * loaded with them. Pure (the
 * environment is passed in), so `vp test` covers it.
 */

import type { ChatStop, GenerationSpeculation, SamplerSpec } from "@karume/models/gemma";

import type { RecordEnvironment } from "@/lib/environment";
import type { ComponentRecord, SourceRecord } from "@/lib/record";

import type { ChatSettings, GemmaMount } from "./settings";
import type { TurnTiming } from "./turns";

/** The version of the saved record; bump it when a field changes meaning or goes away. */
export const RECORD_FORMAT = "karume-playground-gemma4/1";

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
};

export const stopRecordOf = (stop: ChatStop): StopRecord => ({
  reason: stop.reason,
  tokens: stop.tokens,
  ...("token" in stop ? { token: stop.token } : {}),
  ...(stop.reason === "stop-string" ? { stopString: stop.stopString } : {}),
  ...(stop.speculation === undefined ? {} : { speculation: stop.speculation }),
});

/** Runs reported by `onRunDiagnostics` during the turn, by phase (only with diagnostics). */
export type RunCounts = {
  readonly prefill: number;
  readonly decode: number;
  readonly draft: number;
  readonly verify: number;
};

export const NO_RUNS: RunCounts = { prefill: 0, decode: 0, draft: 0, verify: 0 };

export type TurnRecord = {
  /** 1-based, in the order sent (failed turns included). */
  readonly index: number;
  readonly userText: string;
  /** The streamed text (empty for a turn that produced none). */
  readonly replyText: string;
  readonly request: { readonly maxNewTokens: number; readonly sampler: SamplerSpec };
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
   * Whether the session kept the turn (its message and reply) in its history. False when it treated
   * the turn as never having happened: no text and no EOS (a Stop during prefill, a failure).
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
  /** The entry point: `Pipeline` (gemma4) or `gemma4Qat.Pipeline` (gemma4-qat). */
  readonly pipeline: "gemma.Pipeline" | "gemma4Qat.Pipeline";
  readonly model: string;
  readonly preset: { readonly quant: string; readonly label: string; readonly isDefault: boolean };
  /** Whether the MTP drafter was built (`speculative` of `fromPretrained`). */
  readonly speculative: boolean;
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
  pipeline: input.mount === "gemma4" ? "gemma.Pipeline" : "gemma4Qat.Pipeline",
  model: input.model,
  preset: input.preset,
  speculative: input.speculative,
  loadSeconds: input.loadSeconds,
  diagnosticsCollected: input.diagnostics !== undefined,
  chat: input.chat,
  turns: input.turns,
  ...(input.diagnostics === undefined ? {} : { diagnostics: input.diagnostics }),
});

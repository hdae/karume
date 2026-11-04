/**
 * `@karume/models/gemma4-qat-mobile` — E2B / E4B of the fixed mobile QAT.
 *
 * Experimental. The generation and conversation APIs are shared with Gemma, but it accepts only its
 * own distribution, with fixed INT2/INT4/INT8 and SRQ. MTP and broad quality acceptance are not
 * supported.
 *
 * The distribution is `karume-gemma4-qat-mobile` (renamed from `karume-gemma4-qat` on 2026-11-04 —
 * that name now holds the q4_0 QAT line, which `@karume/models/gemma` reads). It has no published
 * revision under the new name yet, so this subpath has no `SOURCES` table (ADR 0073 decision 1 — a
 * family without a public repository has none); pass a `{ repo, revision }` or a local source.
 * DECIDED: [ADR 0097](../../docs/decisions/0097-gemma4-qat-integration.md).
 */
export { Pipeline } from "./src/gemma/qat-pipeline.ts";
export type { FromPretrainedOptions, PipelineOptions } from "./src/gemma/qat.ts";
export { ropeInputs } from "./src/gemma/qat.ts";
export {
  dropOldestTurns,
  CHUNK_BUCKETS,
  chatPrompt,
  ChatSession,
  chatTurn,
  GenerationCapacityError,
  ModelInputError,
  parsePipelineConfig,
} from "./gemma.ts";
export type {
  Assets,
  ChatDialect,
  ChatMessage,
  ChatOptions,
  ChatOverflow,
  ChatOverflowPolicy,
  ChatSessionHost,
  ChatSessionOptions,
  ChatStop,
  ChatStream,
  ChatTurnOptions,
  DefaultSampler,
  EstimateOptions,
  PipelineConfig,
  PleResidency,
  PrefillProgress,
  RopeLayerSpec,
  RopeLayerType,
  RopeSpec,
  RunPhase,
  SequenceOptions,
  GenerationEvent,
  GenerationStop,
  SamplerSpec,
} from "./gemma.ts";

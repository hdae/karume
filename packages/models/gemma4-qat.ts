/**
 * `@karume/models/gemma4-qat` — E2B / E4B of the fixed mobile QAT.
 *
 * Experimental. The generation and conversation APIs are shared with Gemma, but it accepts only its
 * own distribution, with fixed INT2/INT4/INT8 and SRQ. MTP and broad quality acceptance are not
 * supported.
 * DECIDED: [ADR 0097](../../docs/decisions/0097-gemma4-qat-integration.md).
 */
export { Gemma4QatPipeline } from "./src/gemma/pipeline.ts";
export type {
  Gemma4QatFromPretrainedOptions,
  Gemma4QatPipelineOptions,
} from "./src/gemma/pipeline.ts";
export { gemma4QatRopeInputs } from "./src/gemma/rope.ts";
export {
  dropOldestTurns,
  GEMMA4_CHUNK_BUCKETS,
  gemma4ChatPrompt,
  Gemma4ChatSession,
  gemma4ChatTurn,
  GenerationCapacityError,
  ModelInputError,
  parseGemma4PipelineConfig,
} from "./gemma.ts";
export type {
  Gemma4Assets,
  Gemma4ChatMessage,
  Gemma4ChatOptions,
  Gemma4ChatOverflow,
  Gemma4ChatOverflowPolicy,
  Gemma4ChatSessionHost,
  Gemma4ChatSessionOptions,
  Gemma4ChatStop,
  Gemma4ChatStream,
  Gemma4ChatTurnOptions,
  Gemma4DefaultSampler,
  Gemma4EstimateOptions,
  Gemma4PipelineConfig,
  Gemma4PleResidency,
  Gemma4PrefillProgress,
  Gemma4RopeLayerSpec,
  Gemma4RopeLayerType,
  Gemma4RopeSpec,
  Gemma4RunPhase,
  Gemma4SequenceOptions,
  GenerationEvent,
  GenerationStop,
  SamplerSpec,
} from "./gemma.ts";

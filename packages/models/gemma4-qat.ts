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

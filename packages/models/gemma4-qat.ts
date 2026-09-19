/**
 * `@karume/models/gemma4-qat` — 固定 mobile QAT の E2B / E4B。
 *
 * 実験段階。通常生成・会話の API は Gemma と共通だが、固定 INT2/INT4/INT8 と
 * SRQ を持つ別配布形だけを受ける。MTP と広い品質検収は未対応。
 * DECIDED: [ADR 0097](../../docs/decisions/0097-gemma4-qat-integration.md)。
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
  Gemma4PleReadOptions,
  Gemma4PleResidency,
  Gemma4PleShardSource,
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

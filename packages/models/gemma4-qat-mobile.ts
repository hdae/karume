/**
 * `@karume/models/gemma4-qat-mobile` — E2B / E4B of the fixed mobile QAT.
 *
 * Experimental. The generation and conversation APIs are shared with Gemma, but it accepts only its
 * own distribution, with fixed INT2/INT4/INT8 and SRQ. MTP and broad quality acceptance are not
 * supported.
 *
 * The distribution is `karume-gemma4-qat-mobile` (renamed from `karume-gemma4-qat` on 2026-11-04 —
 * that name now holds the q4_0 QAT line, which `@karume/models/gemma` reads).
 * DECIDED: [ADR 0097](../../docs/decisions/0097-gemma4-qat-integration.md).
 */
export { Pipeline } from "./src/gemma/qat-pipeline.ts";
export type { FromPretrainedOptions, PipelineOptions } from "./src/gemma/qat.ts";
export { ropeInputs } from "./src/gemma/qat.ts";
/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092):
 *
 * - `SOURCES["gemma4-qat-mobile"]` = `hdae/karume-gemma4-qat-mobile` (E2B and E4B; the manifest's
 *   default is E2B, E4B is chosen with `{ model: "e4b" }`)
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass the entry
 * as is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility yourself,
 * write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { QAT_MOBILE_SOURCES as SOURCES } from "./src/gemma/config.ts";
export {
  dropOldestTurns,
  CHUNK_BUCKETS,
  chatPrompt,
  ChatSession,
  chatToolResults,
  chatTurn,
  GenerationCapacityError,
  ModelInputError,
  parseChatResponse,
  parsePipelineConfig,
} from "./gemma.ts";
export type {
  Assets,
  ChatAssistantMessage,
  ChatDialect,
  ChatJsonObject,
  ChatJsonValue,
  ChatMessage,
  ChatOptions,
  ChatOverflow,
  ChatOverflowPolicy,
  ChatReply,
  ChatResponse,
  ChatSessionHost,
  ChatSessionOptions,
  ChatSessionStream,
  ChatStop,
  ChatStream,
  ChatTemplateOptions,
  ChatTextMessage,
  ChatTool,
  ChatToolCall,
  ChatToolMessage,
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
  ParsedToolCall,
  SamplerSpec,
} from "./gemma.ts";

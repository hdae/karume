/**
 * `@karume/models/gemma` — the subpath surface of the Gemma family (gemma4 = text → text
 * generation).
 *
 * ADR 0008: this is an **explicitly designed thin surface**; internal modules are not re-exported
 * wholesale. The surface follows the user's stories — build ({@link Pipeline.fromPretrained} /
 * {@link Pipeline.fromAssets}) / converse (`chat`) / drive it yourself (`sequence` +
 * {@link chatPrompt} + decoding with the tokenizer) / release (`dispose`).
 *
 * The generation vocabulary (`GenerationEvent` / `GenerationStop` / `SamplerSpec` /
 * `GenerationCapacityError`) is **independent of any pipeline**, but for now only this family uses
 * it, so it is exported from this surface (and through the barrel as `gemma.*`). Functions that
 * **create** a `GenerationProgram` / `GenerationSequence` are not exported — the pipeline is the
 * only entry point, and no surface lets one be assembled around the static wiring.
 *
 * MUST: every module is free of side effects (no execution at import, no global mutable state —
 * CLAUDE.md).
 */

export { Pipeline } from "./src/gemma/pipeline.ts";
/**
 * The default of {@link PipelineOptions.chunkBuckets} when omitted (only the steps below
 * `chunkLength` are used).
 */
export { CHUNK_BUCKETS } from "./src/gemma/pipeline.ts";
export type {
  Assets,
  ChatOptions,
  /**
   * The generation shape `estimateSessionMemory` estimates (capacity / chunk length — both run-time
   * knobs).
   */
  EstimateOptions,
  FromPretrainedOptions,
  PipelineOptions,
  /** The specification of `sequence()` (the capacity this conversation allocates). */
  SequenceOptions,
} from "./src/gemma/pipeline.ts";
export type {
  /**
   * The stop reason of `chat` (the sequence layer's reasons + `stop-string`, which only this layer
   * can decide).
   */
  ChatStop,
  ChatStream,
  /** One prefill progress report (`chunk / chunks` — received by `onPrefill`). */
  PrefillProgress,
  /**
   * Which run one diagnostics report belongs to (a prefill chunk / a decode step — received by
   * `onRunDiagnostics`).
   */
  RunPhase,
} from "./src/gemma/chat-turn.ts";
/**
 * The knobs of the self-tuning gate (`gate` of {@link PipelineOptions.speculative} — **static
 * knobs for measurement and acceptance**; the defaults are enough). They are on the public surface
 * because a caller running an A/B of the knobs (default vs another threshold) with the same script
 * needs them to build {@link PipelineOptions}.
 */
export type { SpeculationGateOptions } from "./src/generation/speculation-gate.ts";
/**
 * Where PLE lives (the values of {@link PipelineOptions.pleResidency} — default `"host"`).
 *
 * `"gpu"` is the opt-in value that uploads the quantized bytes to the GPU once at load and builds
 * each run's `per_layer_inputs` with a gather on the GPU (ADR 0085 addendum 2026-09-19, GPU
 * residency).
 */
export type { PleResidency } from "./src/gemma/ple-gpu.ts";
/**
 * The middle layer that carries a multi-turn conversation (between `chat` and `sequence` — ADR 0083
 * addendum 2026-09-02).
 *
 * `chat` is one sequence per turn and re-renders the past turns every time (prefill grows O(n²) as
 * the conversation grows); `sequence` hands token ids and the KV lifetime to the caller.
 * {@link ChatSession} covers only the gap between them — it carries the KV over and truncates
 * a turn that does not fit the capacity **before sending it** with
 * {@link ChatSessionOptions.onOverflow} (default {@link dropOldestTurns}).
 *
 * With {@link ChatSessionOptions.tools} a turn can end on a tool call (`reply.end === "tool-call"`);
 * {@link ChatSession.sendToolResults} continues the same model turn with the results.
 */
export { dropOldestTurns, ChatSession } from "./src/gemma/chat-session.ts";
export type {
  ChatOverflow,
  ChatOverflowPolicy,
  /** How one session turn ended — `"turn"` / `"tool-call"` / `"cut-off"`, with the parsed message. */
  ChatReply,
  /**
   * The surface the session reads from the pipeline ({@link Pipeline} satisfies it as is).
   */
  ChatSessionHost,
  ChatSessionOptions,
  /** The stream of one session turn (`ChatStream` + `reply`). */
  ChatSessionStream,
  ChatTurnOptions,
} from "./src/gemma/chat-session.ts";
/**
 * The static wiring declared by the distribution (`pipelineConfig` of `karume.json` — ADR 0038 §1).
 */
export type { DefaultSampler, PipelineConfig } from "./src/gemma/config.ts";
/**
 * **The table of sources verified with this package version** (one table per family; keys are the
 * Hugging Face repository name without `karume-`, values are pinned commit SHAs — ADR 0073 / 0092).
 * One entry per repository, several models in each (choose a non-default one with `{ model }`):
 *
 * - `SOURCES["gemma4"]` = `hdae/karume-gemma4` — the regular line (E2B / E4B)
 * - `SOURCES["gemma4-qat"]` = `hdae/karume-gemma4-qat` — the q4_0 QAT line (E2B / E4B / 12B, each
 *   with its QAT MTP drafter)
 *
 * **An opt-in for callers who want to follow the package version automatically** — pass an entry as
 * is as the first argument of {@link Pipeline.fromPretrained}. To pin reproducibility
 * yourself, write your own `{ repo, revision }` instead of using this table (`fromPretrained` has no
 * default).
 */
export { SOURCES } from "./src/gemma/config.ts";
/**
 * A raw declaration (`unknown`) → a validated {@link PipelineConfig} (failures fail loudly).
 *
 * **An entry point for consumers that use `fromAssets`**. `fromPretrained` runs it internally, so
 * there is no need to call it there; it is needed by callers that read `karume.json` themselves and
 * pass it to {@link Pipeline.fromAssets} (what they have is the hub's
 * `ModelEntry.pipelineConfig` = `Record<string, unknown>`, so without this entry point they could
 * only cast with `as` or keep their own copy of the three numbers).
 */
export { parsePipelineConfig } from "./src/gemma/config.ts";
/**
 * The RoPE parameters declared by the distribution (`pipelineConfig.rope` — two layer types).
 *
 * The distribution has no table; the host computes cos / sin for one chunk at a time. The source of
 * truth for the formula is `src/gemma/rope.ts` (the exporter side mirrors it).
 */
export type { RopeLayerSpec, RopeLayerType, RopeSpec } from "./src/gemma/rope.ts";
/**
 * Positions → the four RoPE graph inputs (`[1, rows, headDim]` f32 of
 * `rope_{sliding,full}_attention_{cos,sin}`).
 *
 * `Pipeline` feeds these into the derived-input slots internally, so there is no need to call
 * it there. It is needed by callers that **run the production graph themselves** (below the
 * low-level surface — acceptance and measurement paths that issue runs on a real Session); without
 * this entry point they would have to copy the formula (and a copy only compares the same formula
 * written twice).
 */
export { ropeInputs } from "./src/gemma/rope.ts";

/**
 * Conversation → token ids (including `<bos>`, ending with the generation prompt), with the
 * template options (function declarations, thinking — ADR 0084 decision 5 and its addendum of
 * 2026-11-04). Messages, tool calls, tool results and declarations take the HF / OpenAI shape;
 * what the template cannot take (content parts, the legacy `tool_responses`, ambiguous results)
 * fails loudly.
 *
 * `chat` runs this internally, so it is needed only when driving the low-level surface
 * (`sequence`) yourself.
 */
export { chatPrompt } from "./src/gemma/text/chat.ts";
/**
 * One new message → **the delta tokens appended as the continuation of the conversation** (the entry
 * point for the second and later turns when running multiple turns with `sequence` — past turns are
 * in the context's KV, so they are not re-rendered).
 *
 * MUST: do not include the `<turn|>` that closes the previous turn (the `pendingToken` of
 * `GenerationSequence` prepends it — the turn-local contract is defined in the doc of
 * `renderGemma4ChatTurn`, and the check in `gemma4_chat_test.ts` verifies it holds).
 */
export { chatTurn } from "./src/gemma/text/chat.ts";
/**
 * Tool results → **the delta tokens that continue the model turn after a tool call** (generation
 * stopped at `<|tool_response>`, which the sequence holds as the pending token — the counterpart of
 * {@link chatTurn} for a turn that ended on a call).
 */
export { chatToolResults } from "./src/gemma/text/chat.ts";
export type {
  ChatAssistantMessage,
  ChatJsonObject,
  ChatJsonValue,
  ChatMessage,
  ChatRole,
  /** The template options (`tools` / `enableThinking` / `preserveThinking`). */
  ChatTemplateOptions,
  ChatTextMessage,
  /** One function declaration (the OpenAI shape). */
  ChatTool,
  /** One tool call of an assistant message (arguments as an object). */
  ChatToolCall,
  /** One tool result (role `tool`). */
  ChatToolMessage,
} from "./src/gemma/text/chat.ts";
/**
 * Generated token ids (with their stop token) + the prompt they followed → **the parsed model
 * turn** (`content` / `thinking` / `tool_calls`) — upstream `parse_response`. A turn that is cut,
 * or malformed, fails loudly.
 */
export { parseChatResponse } from "./src/gemma/text/chat-response.ts";
export type { ChatResponse, ParsedToolCall } from "./src/gemma/text/chat-response.ts";
/**
 * The chat template a model follows ({@link PipelineConfig.chatDialect} — `"gemma4"` for E2B / E4B,
 * `"gemma4-unified"` for 12B). {@link chatPrompt} / {@link chatTurn} read it from the tokenizer.
 */
export type { ChatDialect } from "./src/gemma/text/chat-dialect.ts";

/**
 * The tokenizer built from the assets (`Pipeline.tokenizer`). It holds the entry points that
 * turn token ids received on the low-level surface back into strings (`decode` /
 * `createDetokenizer`).
 *
 * Not public as a value — the entry point is `Pipeline.fromAssets`, and no surface lets a
 * half-built tokenizer be created around the asset checks (ADR 0008).
 */
export type { DecodeOptions, Tokenizer } from "./src/gemma/text/tokenizer.ts";
export type { StreamingDetokenizer } from "./src/text/detokenizer.ts";

/**
 * The generation vocabulary (ADR 0083). The types needed to drive `sequence()` yourself, and the
 * exception that tells "the conversation does not fit" apart from other failures (decision 10 — **on
 * the low-level surface** truncating the conversation is the host's responsibility. The high-level
 * surface {@link ChatSession} has an injectable default policy — the same ADR's addendum
 * 2026-09-02).
 */
export { GenerationCapacityError } from "./src/generation/sequence.ts";
export type {
  GenerationCapacityConstraint,
  GenerationCapacityDetail,
  /**
   * The decode run-ahead counters (`GenerationStop.decodeAhead` — the configured depth, the batches
   * the turn ran and the steps they computed. Present only on turns of a pipeline built with
   * `decodeAhead` ≥ 2).
   */
  GenerationDecodeAhead,
  GenerationEvent,
  /**
   * The self-tuning gate state carried by one diagnostics report (`GenerationRunPhase.gate` — the
   * steady mode, the number of switches within the turn, and whether the run was included in the
   * wall-time observation).
   */
  GenerationGateTrace,
  GenerationRequest,
  /**
   * One diagnostics report per run (what the {@link RunPhase} received by `onRunDiagnostics`
   * is — a prefill chunk / a decode step / a speculative draft-verify cycle, plus the run's wall time
   * and the gate state).
   */
  GenerationRunPhase,
  GenerationSequence,
  /**
   * The speculation tally (`GenerationStop.speculation` — cycle count, draft run count, accepted
   * count, and a histogram of accepted counts. Present only on turns of a pipeline with a drafter).
   */
  GenerationSpeculation,
  GenerationStop,
  GenerationStream,
} from "./src/generation/sequence.ts";
export type { GenerationProgram } from "./src/generation/program.ts";
export type { SamplerSpec } from "./src/generation/sampler.ts";

/**
 * A failure caused by the input (the request itself cannot be accepted = fixing the input makes it
 * pass — the range of sampler values, `maxNewTokens`; {@link GenerationCapacityError}, raised when
 * the conversation does not fit the capacity, is a subclass of it). **One type across all
 * families**, so a host serving several families can tell 400 from 500 with this alone. Broken
 * internal invariants, asset mismatches and GPU capacity are thrown as plain `Error`s.
 */
export { ModelInputError } from "./src/errors.ts";

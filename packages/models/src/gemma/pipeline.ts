/**
 * `Pipeline` — the one surface for **string in → string out** (stage 4 of the generation API
 * wave).
 *
 * It ties together four things, each of which already has its source of truth elsewhere:
 *
 * 1. the Session of the product graph (PLE moved out + last-row logits exit) — the container
 *    `model.krm` written directly by `tools/export-recipes/gemma4/export_product.py`
 * 2. the host PLE gather (`src/gemma/ple.ts` — ADR
 *    [0085](../../../../docs/decisions/0085-ple-host-gather.md)), plugged into the
 *    {@link GenerationWiring.derivedInputs} field — for the models with PLE; gemma-4-12B-it has
 *    none, and admission settles which kind a distribution is ({@link admitGemma4Ple})
 * 3. the compiled tokenizer (`src/gemma/text/`) and the chat format (`src/gemma/text/chat.ts`)
 * 4. the generation loop (`src/generation/` — the program / sequence / sampler of ADR
 *    [0083](../../../../docs/decisions/0083-generation-api-surface.md))
 *
 * What this adds is **only the wiring and the cross-check of the id spaces**; it holds not a single
 * number or vocabulary of its own.
 *
 * ## Two entry points (the distribution and bytes in hand)
 *
 * {@link Pipeline.fromPretrained} builds from the HF distribution (the `pipelineConfig` of
 * `karume.json` declares the static wiring — the writer's source of truth is
 * `tools/export-recipes/gemma4/distribution.py`), and {@link Pipeline.fromAssets} builds from
 * bytes already fetched. **No defaults are placed here** — chunk length, capacity and position limit
 * all move with the asset generation, so nothing silently uses a stale number.
 *
 * MUST: PLE is **not made resident in full** even from the distribution (ADR 0085 decision 3). Both
 * the index and the values are assets of the `model` container (ADR 0109 decision 4), and only the
 * touched blocks are range-read from the persistent cache.
 *
 * ## MUST: cross-check the id spaces (ADR 0085 decision 5)
 *
 * {@link admitGemma4} and {@link buildGemma4Program} compare the ids the tokenizer can produce, the
 * vocab rows of the main embedding and the rows of the PLE index. A mismatch here reads **not out of
 * bounds but "a valid row of another token"** (silent corruption without an exception), so this is
 * the only place for the fail-loudly check.
 *
 * ## MUST: every module is side-effect free
 *
 * No import-time execution and no global mutable state (CLAUDE.md).
 */

import {
  assertGemmaSessionOverrides,
  gemmaSubmitPolicy,
  resolveGemmaSessionOptions,
} from "./session-options.ts";
import { fitCapacityCandidates, pickFitCapacity } from "./fit-capacity.ts";
import { assertGemma4ExpertCacheOptions, resolveGemma4ExpertCache } from "./expert-cache.ts";
import { createGemmaGreedyOutput, type GemmaGreedyOutput } from "./greedy-output.ts";
import { admitGemma4Qat, assertGemma4QatPle, ropeInputs as qatRopeInputs } from "./qat.ts";
import { closeableGenerator } from "../concurrency/closeable-generator.ts";
import {
  acquireGpu,
  type AdmissionReport,
  estimateGraphMemory,
  ExecutionError,
  type EstimateOptions as GraphEstimateOptions,
  fitsHeadroom,
  type GenerationContext,
  type GpuContext,
  planWeightResidency,
  type ResidentTensor,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type SharedWeight,
  type StateAttentionReduce,
} from "@karume/runtime";
import {
  type DistributionSource,
  type FetchAssetsOptions,
  type GpuFeaturesSpec,
  type HubRepoRef,
  loadManifest,
  resolveSelection,
} from "@karume/hub";

import { createOperationChain } from "../concurrency/serial.ts";
import { ModelInputError } from "../errors.ts";
import {
  assetComponentOpener,
  type ComponentOpener,
  type GraphOwner,
  loadContainerComponents,
} from "../hub/components.ts";
import {
  type FromPretrainedComponentOptions,
  type FromPretrainedHubOptions,
  hubLoadOptions,
} from "../hub/load-options.ts";
import { readAssetBuffer } from "../hub/asset-readers.ts";
import { toManifestSource } from "../hub/repo-ref.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import { settleReleasedMemory } from "../session/settle-released-memory.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsBeforeDownload,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../session/gpu-features.ts";
import {
  GEMMA4_PIPELINE_NAME,
  type DefaultSampler,
  type PipelineConfig,
  parsePipelineConfig,
  type PipelineVisionConfig,
} from "./config.ts";
import {
  createGenerationProgram,
  type GenerationProgram,
  generationProgramFace,
  type GenerationSoftInputs,
  type GenerationWiring,
} from "../generation/program.ts";
import {
  assertGenerationRequestValues,
  createGenerationSequence,
  type GenerationRequest,
  type GenerationSequence,
  type GenerationSoftBlock,
  type GenerationSpeculativeOptions,
  type GenerationStream,
  physicalChunkRows,
} from "../generation/sequence.ts";
import {
  createExpertCacheSession,
  type ExpertCacheReport,
  type ExpertCacheSession,
} from "../generation/expert-cache-session.ts";
import type { DraftFace } from "../generation/speculation.ts";
import {
  createSpeculationGate,
  type SpeculationGateOptions,
} from "../generation/speculation-gate.ts";
import { type SamplerSpec, snapshotSpec } from "../generation/sampler.ts";
import { createGemma4Ple, type Gemma4Ple } from "./ple.ts";
import { type Gemma4PleIndex, readGemma4PleIndex } from "./ple-index.ts";
import {
  assertGemma4PleResidency,
  createGemma4PleResident,
  type PleResidency,
  type Gemma4PleResident,
} from "./ple-gpu.ts";
import { gemma4RopeInputNames, ropeInputs } from "./rope.ts";
import { GEMMA4_DRAFT_STEPS, type Gemma4Drafter, openGemma4DraftFace } from "./speculative.ts";
import { createStopStringFilter } from "../text/detokenizer.ts";
import { parseGemmaTokenizerAsset } from "./text/asset.ts";
import { Tokenizer } from "./text/tokenizer.ts";
import { assertRgb8, type Rgb8Image } from "../image/preprocess.ts";
import { type ChatMessage, chatImagePrompt, chatPrompt, gemma4StopTokens } from "./text/chat.ts";
import { type ImageEmbedding, type ImageRows, imageRows, runImageEmbedder } from "./vision.ts";
import { DEFAULT_CHAT_DIALECT } from "./text/chat-dialect.ts";
import {
  admitGemma4,
  admitGemma4Ple,
  assertChunkLength,
  assertGemma4Vision,
  assertGemma4VisionChunkLength,
  assertGemma4ChunkBuckets,
  GEMMA4_PER_LAYER_INPUTS,
  GEMMA4_SOFT_EMBEDS,
  GEMMA4_SOFT_MASK,
  type Gemma4Admission,
  gemma4ManifestConfig,
  gemmaEntryName,
  type GemmaFamily,
  type GemmaFamilyAdmission,
} from "./admission.ts";
import {
  chatStreamOf,
  completeChatTurn,
  decodeChatChunks,
  type ChatStop,
  type ChatStream,
  type PrefillProgress,
  type RunPhase,
  runDiagnosticsHook,
} from "./chat-turn.ts";

/**
 * Graph input names (source of truth: the constants of `export_product.py`).
 *
 * NOTE: `position_ids` is gone — the four RoPE cos / sin inputs (`./rope.ts` builds both the names
 * and the values) became the only inputs that carry the position. The position is passed as "the
 * table itself", not as "an index into a table".
 */
const INPUT_IDS = "input_ids";
const PER_LAYER_INPUTS = GEMMA4_PER_LAYER_INPUTS;
const LAST_ROW = "last_row";

/**
 * The names that appear in the manifest's weights / assets tables (the convention names of ADR 0041
 * §3).
 */
const MODEL = "model";
/**
 * The weights role added only when speculation (the MTP drafter) is used (ADR 0096 stage 2 §5).
 *
 * MUST: by default it is **not put in the table of components to fetch** (the `weights` of
 * `resolveSelection` narrows to `model` only). Putting it there would download the drafter's
 * descriptor and weight parts even for a load that does not use speculation.
 */
const DRAFTER = "drafter";
/**
 * The weights role of the image embedder (ADR 0144). The same MUST as {@link DRAFTER}: not a byte of
 * it is fetched unless {@link PipelineOptions.vision} asks for it.
 */
const VISION = "vision";
const TOKENIZER_ASSET = "tokenizer";

/**
 * The input for building from fetched assets (**one generation of the product series**).
 *
 * PLE is no longer an asset the caller prepares — both the index and the values are inside the
 * `model` container (ADR 0109 decision 4), so this surface too gets everything down to PLE just by
 * passing the parts of `model`. A PLE-free model (gemma-4-12B-it) has neither in its container and
 * no per-layer input in its graph; the same fields build it.
 */
export type Assets = {
  readonly config: PipelineConfig;
  /**
   * The **parts** of the product graph's container (`krm`) (in index order from part 0, including
   * zero-length parts).
   *
   * MUST: the index is the part id itself. Skipping a zero-length part shifts every later part up by
   * one, and they are read as different parts.
   */
  readonly model: readonly Uint8Array<ArrayBuffer>[];
  /** The bytes of the compiled tokenizer asset (ADR 0084 decision 1). */
  readonly tokenizer: Uint8Array<ArrayBuffer>;
};

export type PipelineOptions = {
  /**
   * Shares an existing GPU. **The side that passed it keeps ownership**, so
   * {@link Pipeline.dispose} does not destroy it. When omitted, the pipeline calls
   * `acquireGpu` internally and destroys it in `dispose()`.
   */
  readonly gpu?: GpuContext;
  /**
   * The **host RAM limit (bytes)** for PLE's resident whole blocks (LRU — ADR 0085 decision 3).
   *
   * Default: two of the largest blocks (`defaultGemma4PleResidentBytes` in `ple-index.ts`). `0`
   * keeps neither whole blocks nor quantized rows. The budget changes neither values nor token
   * sequences.
   *
   * NOTE: taken in bytes, not as a count — the block width changes with the asset generation, so
   * "N blocks" would mean different RAM per generation (ADR 0085 addendum 2026-09-02).
   * NOTE: the budget bounds whole blocks only. On top of it sits a row cache of at most 256 rows
   * (256 rows of regular E2B: 4,587,520 B with its one preset `bf16`; q4_0 QAT E2B 1,433,600 B;
   * mobile QAT E2B 1,182,720 B), kept only while the budget is above 0, so the host RAM PLE holds is
   * at most the budget plus that headroom (ADR 0085 addendum 2026-10-27). Tightening the budget increases block re-reads (ADR 0085).
   *
   * MUST: refused (fails loudly) for a model without PLE (gemma-4-12B-it) — there is nothing to
   * budget ({@link assertGemma4PleFreeOptions}).
   */
  readonly maxResidentPleBytes?: number;
  /**
   * Where PLE is placed (default `"host"` = as it is today; ADR 0085 addendum (GPU residency)).
   *
   * `"gpu"` uploads PLE's stored values (the quantized bytes, or the bf16 values of an
   * original-weights preset) to the GPU **once at load** and builds each run's `per_layer_inputs`
   * with an on-GPU gather (the host dequantization and writeBuffer disappear — about 0.25 ms/token in
   * decode). Prefill and decode take the same path.
   *
   * MUST: it is a single binding, so on a device where the byte size of values does not fit
   * `maxStorageBufferBindingSize` / `maxBufferSize` it **fails loudly** (it does not silently fall
   * back to the host path). The amount is set by the distribution and the preset: the regular
   * Gemma 4 line's one preset `bf16` needs 4.37 GiB (E2B — 4,697,620,480 B) and 5.25 GiB
   * (E4B — 5,637,144,576 B), more than a 2 GiB binding limit holds; the q4_0 QAT line (i4) needs
   * 1.09 GiB (E2B) and 1.31 GiB (E4B); the mobile QAT line 1.09 GiB (E2B, i4) and 672 MiB (E4B, i2).
   *
   * MUST: it **cannot be combined** with the host-side row cache
   * ({@link PipelineOptions.maxResidentPleBytes}) (under GPU residency that budget would never
   * be drawn on). Combining it with speculative decoding ({@link PipelineOptions.speculative})
   * or with GPU timing diagnostics (`acquireGpu({ gpuTiming: true })`) is not supported either; each
   * is refused by failing loudly.
   *
   * MUST: refused (fails loudly, even `"host"`) for a model without PLE (gemma-4-12B-it) — there is
   * nothing to place ({@link assertGemma4PleFreeOptions}).
   */
  readonly pleResidency?: PleResidency;
  /**
   * The **decode run-ahead** depth (default `1` = off: one greedy step per GPU batch, as today).
   *
   * With `d ≥ 2`, a temperature-0 turn without processing runs up to `d` decode steps in one GPU
   * batch: each step feeds the token the previous step selected **on the GPU**, and the `d`
   * selections are read back together at the batch's single fence. It removes the host round trip
   * between tokens (the fence, the readback, writing the next token) from all but one step in `d`;
   * the cost is that a batch which meets a stop token has already computed up to `d − 1` steps past
   * it (left uncommitted, never delivered). The generated token ids are **identical** to `d = 1`
   * (the same kernels on the same inputs). Turns whose sampler is not plain greedy, and pipelines
   * built with {@link PipelineOptions.onRunDiagnostics}, decode one step at a time as before (the
   * turn's `done` carries `decodeAhead` with the batches it ran).
   *
   * MUST: an integer ≥ 1 and **at most the KV ring's sliding slack** (the rows one batch leaves
   * uncommitted — 8 for the published Gemma 4 graphs). The slack is a property of the conversation's
   * context, so a larger value fails loudly when the first conversation is opened. `d ≥ 2` cannot be
   * combined with {@link PipelineOptions.speculative} (refused before anything is fetched), and a
   * model with PLE needs `pleResidency: "gpu"` (with the host PLE, every step's per-layer inputs are
   * gathered on the host from the step's token, which the run-ahead keeps on the GPU — refused once
   * admission has read the model, before any weight part is fetched). A model without PLE
   * (gemma-4-12B-it) needs nothing more.
   */
  readonly decodeAhead?: number;
  /**
   * **The exact expert cache** of a mixture-of-experts model (Gemma 4 26B A4B — default absent =
   * every expert resident, as before).
   *
   * With it, each layer keeps only `slots` of its experts on the GPU, and the missing ones are read
   * from the distribution when a run's routing picks them. The output is bit-identical to full
   * residency: a run that missed an expert is never committed or sampled from — its rows are dropped,
   * the experts loaded, and the same run issued again, so the cost of a miss is a re-run, not an
   * approximation ({@link Pipeline.expertCacheReport} counts both). A cached distribution must stay
   * in its cache while the pipeline lives (the reads come from it).
   *
   * Prefill chunks are at most `min(slidingSlack, ⌊slots / k⌋, the distribution's chunkLength)` rows
   * (k = the routing's top-k — 8 rows at slots 64 on 26B), which is the default
   * {@link PipelineOptions.chunkLength}; a larger one fails loudly. Long prompts get slow.
   *
   * `hostBytes` adds a **host tier**: a budget in bytes of expert bytes held in RAM in front of the
   * distribution (the runtime's `SessionOptions.expertCache.hostBytes`). The Session fills it while
   * it reads the weights, and a miss the tier holds is uploaded from RAM instead of being read from
   * the distribution again; the least recently used experts over the budget make room for the ones
   * read later. `Number.MAX_SAFE_INTEGER` (any budget at or above every expert's bytes — about
   * 12.85 GB on 26B) holds every expert, so no miss reads the distribution after the load. Absent =
   * no tier: every miss reads the distribution. It changes where the bytes come from, not the
   * output. The tier is host memory: the estimate does not count it, and `dispose` releases it.
   *
   * `depth` turns decode into **self-speculation over the cache** (ADR 0146 decision 8): each cycle
   * drafts up to `depth` tokens one row at a time on the cache as it is (a missing expert contributes
   * 0 and nothing waits), loads what the drafts picked, verifies the frontier and the drafts in one
   * run of `depth + 1` rows, and keeps only the rows before the first one that picked a missing
   * expert. Every token is drawn from such an exact row with the sampler's own random sequence, so
   * the output stays that of full residency at every temperature; the drafts only decide how many
   * tokens a cycle confirms. Absent = one run per decode token, re-run after a miss. Prefill is the
   * same either way.
   *
   * MUST: `slots` is an integer from 2 × k (a chunk has at least 2 rows) to the experts of a layer,
   * on a model whose graph has `expert_linear`; `hostBytes`, when present, is a positive safe integer
   * (both fail loudly before any weight part is fetched; a `hostBytes` below one expert fails at the
   * Session's construction). `depth`, when present, is a positive integer with `depth + 1` at most
   * `⌊slots / k⌋` (a verify's picks fit the slots), the sliding slack and the chunk rows (one verify
   * is one deferred run) — 7 at slots 64 on 26B; a larger one fails loudly with the bound, before any
   * weight part is fetched. Refused (fails loudly, before any weight part is fetched) with
   * `decodeAhead ≥ 2`, {@link PipelineOptions.speculative}, `pleResidency: "gpu"` and
   * {@link PipelineOptions.vision} — each runs the target in a batch or an immediate run, which a run
   * under the cache cannot be. Decode runs on the host sampler (no GPU argmax).
   *
   * DECIDED: docs/decisions/0146-exact-expert-cache.md
   */
  readonly expertCache?: {
    readonly slots: number;
    readonly hostBytes?: number;
    readonly depth?: number;
  };
  /**
   * The row count of a fixed-length prefill chunk (default: the distribution's declaration
   * {@link PipelineConfig.chunkLength}).
   *
   * Raising it lowers the number of prefill runs (and the number of fence waits in proportion), and
   * raises the transient buffers and the attention score matrix per run. The graph's chunk rows are
   * symbolic, so it can be chosen without rebuilding the assets.
   *
   * MUST: at least 2, at most the distribution's declaration
   * {@link PipelineConfig.maxChunkLength}, and at most `maxPosition` (this pipeline's check). 1
   * is a value reserved for the decode shape, and there is no path that runs it as a prefill shape (a
   * chunk with one valid row is run by `GenerationSequence` in the decode shape). The upper limit is
   * taken from the declaration because the trace range of the symbol `M` cannot be read from the
   * assets (the IR's `symbols` is only a list of names) — before the declaration existed, an
   * out-of-range value ran without an exception (measured 2026-09-03).
   * `chunkLength ≤ capacity` against each sequence's capacity is checked by
   * `createGenerationSequence`.
   */
  readonly chunkLength?: number;
  /**
   * Physical chunk row counts used as prefill shapes **in addition to** `chunkLength` (default: the
   * entries of {@link CHUNK_BUCKETS} below `chunkLength`; `[]` disables them).
   *
   * Padding a short prompt to `chunkLength` (768 by distribution default) rows makes the pad rows
   * contribute nothing to the output or the KV, yet the row-local ops (linear / pointwise / norm)
   * still do work proportional to the physical row count. With buckets, such a chunk runs on only the
   * smallest bucket of at least `queryLength` rows. For a long prompt a single 768 chunk is fastest,
   * so instead of lowering the default, **more execution shapes are added**.
   *
   * MUST: integers of at least 2 and below `chunkLength` in **strictly ascending** order (the source
   * of truth of the accepted set is the runtime's `assertChunkBuckets` — this layer only adds "which
   * entry point's option" to the message).
   *
   * NOTE: each one adds a steady PreparedPlan (one execution shape = one plan with its own key — the
   * LRU of ADR 0042 decision 2). The default 6 + the prefill shape + the decode shape = 8 shapes **per
   * capacity**, which fits the PreparedPlan LRU bound (the runtime's `PREPARED_PLAN_CAPACITY` —
   * widened to 12 in the verify wave) — the key includes the resolved slot capacity (ADR 0066
   * decision 3), so alternating sequences of different capacities multiply the shapes by the number
   * of capacities.
   * Adding more makes the oldest drop every time inside the generation loop, and decode silently
   * falls back to re-derivation (no exception — the observation point is
   * `SessionDiagnostics.lastRunPrepared`).
   *
   * NOTE: in a multi-chunk turn only the last chunk has a different M, so the slot backing
   * (capacity 1) is rebuilt one more time (2 rebuilds per turn, prefill → decode, become 3).
   */
  readonly chunkBuckets?: readonly number[];
  /**
   * The observation callback that receives the diagnostics of each run (the same shape as the other
   * families). When per-op GPU time (`lastRunTiming`) is needed, pass `acquireGpu({ gpuTiming: true
   * })` as `gpu` (ADR 0021 — no measurement by default).
   *
   * It is called **once per run** (per chunk in prefill, per step in decode, and per cycle one draft
   * + one verify with speculation), in the **synchronous section** after that run's outputs have been
   * read (the diagnostics' `lastRun*` hold "that run's" values only in this section). What the run
   * was is carried by the second argument {@link RunPhase} — do not infer it from the count (in
   * a turn with a multi-chunk prefill, the second and later prefills would be mistaken for decode).
   * The first token right after prefill is only sampled from the last chunk's logits and comes with
   * no run, so there is no call there.
   *
   * The Session whose diagnostics are read depends on the phase — `draft` is the borrower (the
   * drafter Session), everything else the lender (the target Session).
   *
   * For a non-speculative turn that ran to the end without interruption, the number of notifications
   * is `GenerationStop.tokens − 1 + the number of prefill chunks` (`tokens` counts a stop token as
   * 1, and the subtracted 1 is "the first sampling that comes with no run"). The last decode run that
   * drew a stop token ends without yielding a `token`, but one notification still arrives for that
   * run. A turn closed by the consumer's `break` / `return()` / an interruption delivers only the runs
   * completed up to that point. For a speculative turn the count is
   * `number of prefill chunks + speculation.cycles + speculation.draftRuns + speculation.plainSteps`
   * (`plainSteps` = steps the self-tuning gate ran in the decode shape — absent altogether for a
   * `speculative: "always"` turn).
   *
   * NOTE: unlike the other families, no component name is passed — there is only one graph, so the
   * name would always be the same single value (an argument the receiver cannot branch on is not
   * passed). The same holds when a drafter is present; which Session's diagnostics they are is read
   * from `phase.kind`.
   *
   * Exceptions from the callback are not caught (fail loudly — the whole turn fails).
   */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics, phase: RunPhase) => void;
  /**
   * The reduction form of states-form attention ①QK / ③PV. Resolved in the order explicit value →
   * the selected quant's session → the runtime's reference path `"sequential"` (the same rule as the
   * other quant execution knobs — ADR 0104 addendum 2026-09-26). fromAssets has no quant, so it is
   * sequential when unspecified. The family keeps no code default.
   *
   * `"parallel"` is the variant that splits the KV-length direction over 16 lanes (perf-ledger K-12),
   * removing decode attention's growth in proportion to the KV length (82 → 41 ms/token at P=16K).
   * The reduction order differs, so it is not bit-identical to `"sequential"` (the runtime's reference
   * path), but gemma4's greedy / chat goldens are identical under both (ADR 0058 decision 6, ADR 0067
   * addendum 2026-09-03). `"sequential"` is the way back for parity comparisons or when "suspecting
   * an order-dependent difference".
   * `"parallel-fused"` is an optional setting that fuses the row statistics and PV for few rows and a
   * short column limit (ADR 0102). Its addition order is the same as parallel, and shapes outside its
   * target use the parallel path.
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /**
   * The addition order of the quantized GEMV. Resolved in the order explicit value → the selected
   * quant's session → sequential. fromAssets has no quant, so it is sequential when unspecified. No
   * automatic selection by model configuration. The runtime's SessionOptions is the source of truth
   * for the meaning and the shapes it applies to. Passed to both target and drafter (ADR 0098).
   * Especially for the mobile QAT line (its activations are re-quantized), the generated sequence
   * may change.
   */
  readonly linearGemvReduce?: SessionOptions["linearGemvReduce"];
  /**
   * The optional RMS→add fusion. The runtime setting of the same name is the source of truth for its
   * meaning and scope (ADR 0099). Resolved in the order explicit value → the selected quant's session
   * → the runtime default, and passed to both target and drafter (ADR 0104). fromAssets has no quant,
   * so it is the runtime default when unspecified.
   */
  readonly fuseRmsNormAdd?: SessionOptions["fuseRmsNormAdd"];
  /**
   * The optional linear→fixed-SRQ fusion. The runtime setting of the same name is the source of
   * truth for its meaning and scope (ADR 0103). Rejected unless the effective `linearGemvReduce` is
   * `"parallel"`. Resolved in the order explicit value → the selected quant's session → the runtime
   * default, and passed to both target and drafter (ADR 0104). fromAssets has no quant, so it is the
   * runtime default when unspecified.
   */
  readonly fuseLinearStaticQuantize?: SessionOptions["fuseLinearStaticQuantize"];
  /**
   * The optional gelu_tanh→mul→fixed-SRQ fusion (the mobile QAT per-layer-input gate). The runtime
   * setting of the same name is the source of truth for its meaning and scope (perf-ledger K-48). Resolved
   * in the order explicit value → the selected quant's session → the runtime default, and passed to
   * both target and drafter. fromAssets has no quant, so it is the runtime default when unspecified.
   */
  readonly fuseGeluMulStaticQuantize?: SessionOptions["fuseGeluMulStaticQuantize"];
  /**
   * Passes the fixed-SRQ activations to the parallel GEMV as packed int8. The runtime setting of the
   * same name is the source of truth for its meaning and scope (ADR 0105). Rejected unless the
   * effective `linearGemvReduce` is `"parallel"`. The manifest vocabulary has no field for it, so it
   * enters **only through the caller's explicit option**, and is passed to both target and drafter.
   */
  readonly packedStaticQuantize?: SessionOptions["packedStaticQuantize"];
  /**
   * Runs the eligible decode linears (i2 and i4 at the measured mobile QAT E2B shapes — ADR 0140) on
   * the integer GEMV. The runtime setting of the same name is
   * the source of truth for its meaning and scope. Rejected unless the effective
   * `packedStaticQuantize` is `true`. Resolved in the order explicit value → the selected quant's
   * session → the runtime default, and passed to both target and drafter. Only the mobile QAT graphs
   * have linears at those shapes; fromAssets has no quant, so it is the runtime default when
   * unspecified.
   */
  readonly integerGemv?: SessionOptions["integerGemv"];
  /**
   * Runs the eligible i2 linears (the mobile QAT E2B lm_head, at its measured shape) on the
   * byte-plane GEMV. The runtime setting of the same name is the source of truth for its meaning and scope; it
   * needs no other option. Resolved in the order explicit value → the selected quant's session → the
   * runtime default, and passed to both target and drafter. Only the mobile QAT graphs have i2
   * linears at that shape; fromAssets has no quant, so it is the runtime default when unspecified.
   */
  readonly linearGemvPlanes?: SessionOptions["linearGemvPlanes"];
  /**
   * The optional RMS reduction. subgroup32 requires a supporting GPU and changes the addition order
   * from the reference (ADR 0100).
   */
  readonly rmsNormReduce?: SessionOptions["rmsNormReduce"];
  /**
   * The storage of the full state slots (`"f16"` halves them; default `"f32"`). The runtime setting
   * of the same name is the source of truth for its meaning, and it needs no re-export or GPU feature.
   * Not a quant setting: only this explicit value turns it on. Passed to both target and drafter, and
   * {@link Pipeline.estimateSessionMemory} and {@link Pipeline.fitCapacity} count the smaller slots.
   *
   * It changes the numbers: earlier runs are read rounded to f16 (the first run is unchanged). On a
   * pipeline with a drafter, a turn MUST say `speculative: "always"` or `false` (the self-tuning gate
   * is refused, see {@link speculativeSetup}), so `speculative.gate` is refused here too.
   * An invalid value is refused before anything is fetched.
   *
   * DECIDED: docs/decisions/0143-f16-full-state-storage.md
   */
  readonly fullStateStorage?: SessionOptions["fullStateStorage"];
  /**
   * The GPU submission policy. The same value goes to target / drafter; when omitted,
   * {@link gemmaSubmitPolicy} (the runtime default with at most 128 dispatches per submit — H-40).
   */
  readonly submitPolicy?: SessionOptions["submitPolicy"];
  /**
   * The byte budget for slot backings (the bundle of intermediate buffers per run shape) kept at the
   * same time. The runtime's `SessionOptions.planBackingBudgetBytes` is the source of truth for the
   * meaning, default and value range, and this pipeline passes it straight through (the same value
   * applies to both `createSession` and {@link Pipeline.estimateSessionMemory}).
   * Generation switches between the prefill bucket shapes and the decode shape every turn, so with
   * the default (256 MiB) both are kept and the rebuild on every switch disappears.
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * The limit per row-block temporary of attention (bytes; unset by default = the device's binding
   * limit alone decides the split). The runtime's `SessionOptions.attentionRowBlockMaxBytes` is the
   * source of truth for the meaning and value range (ADR 0124), and this pipeline passes it straight
   * through (the same value applies to the target Session, the drafter Session and
   * {@link Pipeline.estimateSessionMemory}).
   * It is a resource policy: the token sequences stay unchanged (row blocks only change the
   * assignment of rows), and it trades prefill speed for VRAM. The prefill score matrix is heads ×
   * chunk rows × capacity × 4 bytes (1.5 GiB at 8 heads, 768 rows and capacity 65,536), and a value
   * below the binding limit splits it into smaller blocks at the cost of more dispatches per prefill.
   */
  readonly attentionRowBlockMaxBytes?: SessionOptions["attentionRowBlockMaxBytes"];
  /**
   * The parallelism target of the row-block gemv (the linear GEMV family, M ≥ 2). The runtime's
   * `SessionOptions.linearGemvRowsThreadTarget` is the source of truth for the meaning, default
   * (16384 = the saturation point of the reference device) and value range, and this pipeline passes
   * it straight through (the same value applies to both the target Session and the drafter Session).
   * Lowering it on a GPU with a smaller saturation point makes the M=4 verify use taller row blocks
   * (fewer weight re-reads).
   */
  readonly linearGemvRowsThreadTarget?: number;
  /**
   * **Builds an MTP drafter alongside** for speculative decoding (ADR 0096 — not built when omitted).
   *
   * When given, the distribution's `drafter` weights are fetched as well, and one drafter Session is
   * set up that borrows one embedding table of the target Session (no bytes are copied). **Unless it
   * is given, not a single byte of the drafter's parts is downloaded** — it is out of the selection
   * itself (the MUST of {@link DRAFTER}).
   *
   * `chat` / `sequence` of a pipeline with a drafter **speculate by default** (one verify run commits
   * up to `k+1` tokens). The knobs to turn it off per turn / per conversation are
   * {@link ChatOptions.speculative} and {@link SequenceOptions.speculative}.
   * Speculation runs **regardless of temperature**, and the emitted token sequence is identical to
   * the non-speculative one (acceptance draws once per row, with the same logits and the same history
   * as the sampling the non-speculative decode does at the same position — ADR 0096 decision 7).
   *
   * MUST: `k` is **1..{@link GEMMA4_DRAFT_STEPS}** (the distribution's drafter graph is built with 3
   * steps, and the number of exits is the upper limit). Out of range fails loudly.
   */
  readonly speculative?: {
    /**
     * The number of drafts used per cycle (default {@link GEMMA4_DRAFT_STEPS} = the distribution's
     * step count).
     */
    readonly k?: number;
    /**
     * The self-tuning gate's knobs (**static knobs for measurement and acceptance** — the
     * defaults suffice. Treated like {@link PipelineOptions.linearGemvRowsThreadTarget}, and
     * passed straight through to `GenerationSpeculativeOptions.gate`).
     *
     * They apply **only to turns that have a gate** (`speculative: true` / unspecified =
     * `policy: "auto"`). A `speculative: "always"` turn on the same pipeline builds no gate, so none
     * of these knobs reaches it (the `gate` field does not appear at all) — an A/B script that
     * alternates the three modes on one pipeline (`tools/mtp-bench`, removed in 0.15) had that shape.
     *
     * MUST: the knobs' check runs at the argument-check step of `fromPretrained` (build one
     * {@link createSpeculationGate} and discard it) — an invalid value must not survive until after
     * GB-scale weights have been read.
     */
    readonly gate?: SpeculationGateOptions;
  };
  /**
   * **Builds the image embedder alongside** (ADR 0144 — not built when omitted or `false`).
   *
   * When `true`, the distribution's `vision` weights are fetched as well and one more Session is set
   * up; {@link Pipeline.embedImage} and image parts in {@link Pipeline.chat} / `ChatSession` work
   * on it. **Unless it is given, not a byte of those weights is downloaded**, as for the drafter.
   *
   * A distribution without the role, a declaration without `pipelineConfig.vision` or a product graph
   * without `soft_embeds` fails loudly before any weight part is fetched. Only
   * {@link Pipeline.fromPretrained} accepts it: {@link Assets} holds one product graph.
   */
  readonly vision?: boolean;
};

/**
 * The default prefill buckets of the gemma4 pipeline ({@link PipelineOptions.chunkBuckets}) —
 * a power-of-two ladder.
 *
 * The list was settled by measurement (2026-09-07, RTX 3080 Ti / Vulkan, ABBA over 5 ladders × 5
 * prompt lengths). It helps short messages: the wall time of one 20-token prompt + 1 generated token
 * goes from 434 ms with `chunkLength` fixed at 768 to 129 ms with this ladder (−70%).
 *
 * **512 is left out because it measured slower than 768** (550 ms vs 411 ms for a 310-row prompt).
 * The GEMM geometry switches to `M64N32` for `M ≤ 512` and to `reg128x128` above
 * (`gemm-geometry.ts`), so the small-tile shape of 512 rows loses to the large-tile shape of 768
 * rows. 256 is also `M64N32`, but wins with 305 ms vs 410 ms for a 130-row prompt, so it stays in
 * the ladder.
 *
 * **4 / 8 are the buckets for speculative verify**. Verify runs "the k drafted rows + the 1 row
 * committed just before" in one run (R = k+1 rows), which fits 4 rows for k=3 and 8 rows for k ≤ 7
 * (the upper limit of k itself is the sliding ring's slack = the upper limit of rows that can be
 * rejected, 8, which the context exposes as `slidingSlack`). Without these buckets a 4-row verify
 * would be padded to 32 rows, paying back through padding part of the work speculation saved. The
 * buckets also help ordinary generation (follow-up messages of 4 tokens or fewer).
 *
 * NOTE: the default 6 + the prefill shape + the decode shape = **8 shapes per capacity** stay steady
 * (the PreparedPlan count of {@link PipelineOptions.chunkBuckets}).
 *
 * MUST: frozen — this array is shared at module scope, and a consumer reordering it would change how
 * every pipeline built afterwards chooses its physical row count (`chunkBuckets` assumes ascending
 * order and takes the first match).
 */
export const CHUNK_BUCKETS: readonly number[] = /* @__PURE__ */ Object.freeze([
  4, 8, 32, 64, 128, 256,
]);

/**
 * Trims the default buckets to those that fit the chosen `chunkLength`.
 *
 * `chunkLength` is a runtime knob ({@link PipelineOptions.chunkLength}), so passing the default
 * ladder as is would make an option such as `chunkLength: 64` fail on "a conflict between two
 * defaults". **Only the default** is trimmed; buckets the caller gave explicitly are never dropped
 * and fail loudly instead (silently discarding them would leave "a bucket that was declared but has
 * no effect" without an exception).
 */
const defaultChunkBuckets = (chunkLength: number): readonly number[] =>
  CHUNK_BUCKETS.filter((rows) => rows < chunkLength);

/**
 * What {@link Pipeline.fromPretrained} accepts in addition (the selection axes + knobs passed
 * through to the fetch layer).
 *
 * NOTE: the description of the fetch layer's knobs, including that `headers` / `fetch` / `caches` /
 * `onRetry` are **for HTTP sources only**, lives in one place, {@link FromPretrainedHubOptions}.
 */
export type FromPretrainedOptions = PipelineOptions &
  FromPretrainedHubOptions &
  FromPretrainedComponentOptions & {
    /** The manifest's model name (default `defaultModel`). */
    readonly model?: string;
    /** The quant name (default: that model's `defaultQuant`). */
    readonly quant?: string;
    /**
     * Aborts the fetch (not passed to construction — `chat` / `sequence` are interrupted by each
     * request's `signal`).
     */
    readonly signal?: AbortSignal;
  };

/**
 * The options of {@link Pipeline.sequence} (the only knobs that apply to one conversation's
 * lifetime).
 */
export type SequenceOptions = {
  /**
   * The capacity of the full slots this conversation allocates (default: the distribution's default
   * — {@link PipelineConfig.capacity}).
   *
   * MUST: `chunkLength ≤ capacity ≤ maxPosition` (`createGenerationSequence` fails loudly). `"fit"` is
   * {@link Pipeline.fitCapacity}'s result.
   */
  readonly capacity?: number | "fit";
  /**
   * Whether this conversation uses speculative decoding (default = `true` if the pipeline has a
   * drafter).
   *
   * - `true` (default) … speculation **with the self-tuning gate**. While speculation is losing
   *   to decode, it drops to the decode shape (M=1) per cycle, and returns when it looks like winning
   *   (`src/generation/speculation-gate.ts`).
   * - `"always"` … always speculate (no gate). The option for A/B comparison, acceptance checks and
   *   measurement.
   * - `false` … "keep the drafter loaded but run this conversation the old way, 1 token = 1 run".
   *
   * On a pipeline without a drafter only `false` / unspecified pass (`true` / `"always"` fail loudly
   * — the option is not a value to be silently ignored; **there was never a choice to begin with**).
   *
   * NOTE: the gate chooses between the decode shape (M=1) and the verify shape (M=4) by **wall
   * clock**, but the e2e test (`e2e_gemma4_speculative_test.ts`) pins their identity (u32 match of
   * the logits, and sequence match between gated and always) with a check per device, so the
   * wall-clock choice does not change the output (ADR 0096 addendum 2026-09-26).
   * `"always"` and `stateAttentionReduce: "sequential"` remain as options for A/B and acceptance (on a
   * device where the check is red the difference is real, so choose these two there —
   * `docs/limitations.md`).
   */
  readonly speculative?: boolean | "always";
};

/**
 * The options of {@link Pipeline.estimateSessionMemory} (the shape of the generation to
 * estimate).
 */
export type EstimateOptions = {
  /** The capacity to estimate (default: this pipeline's default). */
  readonly capacity?: number;
  /**
   * The chunk length to estimate (default: the value this pipeline uses).
   *
   * The bucket shapes in the estimate are only those of this pipeline's
   * {@link PipelineOptions.chunkBuckets} below this value. Even when it is larger than the
   * pipeline's value, the buckets passed stay this pipeline's ladder; they are not widened to the
   * ladder a pipeline rebuilt with that chunk length would have.
   */
  readonly chunkLength?: number;
};

/**
 * One turn's chat request.
 *
 * MUST: {@link Pipeline.chat} **copies** the contents **at issue** (ADR 0083 addendum
 * 2026-09-02) — rewriting this object or the `sampler` options after the returned stream has started
 * draining has no effect on the running turn.
 */
export type ChatOptions = {
  /**
   * The upper limit of tokens to sample (at least 1). Stop tokens **count** toward it (the same
   * accounting as `GenerationRequest.maxNewTokens`).
   *
   * Omitted = until a stop token or until this turn's sequence is full (`done` then reports
   * `capacity`). `chat` has no history to keep room for, so nothing is reserved; a prompt that alone
   * does not fit fails with `GenerationCapacityError`.
   */
  readonly maxNewTokens?: number;
  /**
   * Additional stop **tokens** for this turn only (unioned with the distribution's EOS set —
   * `GenerationRequest.stopTokens`). Out-of-vocabulary ids and duplicates fail loudly.
   */
  readonly stopTokens?: readonly number[];
  /**
   * Stop **strings** for this turn only (generation stops as soon as any of them appears in **the
   * decoded text**).
   *
   * The check happens after decoding (in this layer), so it stops even when a string spans a token
   * boundary. The matched stop string itself and what follows **are not streamed**, so the pieces
   * received with `for await`, concatenated, end "just before the stop string" (`done` carries
   * `stop-string` and its spelling). Empty strings and duplicates fail loudly.
   *
   * NOTE: for the check, only **the tail that is a prefix of a stop string** is held back from the
   * output even when committed (`createStopStringFilter` in `src/text/detokenizer.ts`). It is
   * streamed in one go once it stops being a prefix, so a turn that did not stop loses not a single
   * character of output.
   *
   * NOTE: the low-level surface ({@link Pipeline.sequence}) has no such knob —
   * `GenerationSequence` handles only token ids, so when running it yourself, write stops as tokens
   * (`docs/limitations.md`).
   */
  readonly stopStrings?: readonly string[];
  /** Sampling options (default {@link PipelineConfig.sampler}, and failing that greedy). */
  readonly sampler?: SamplerSpec;
  /**
   * The capacity this turn's sequence allocates (default: the distribution's default —
   * {@link PipelineConfig.capacity}).
   *
   * `chat` is one turn = one sequence, so this is "how much KV to take for this turn". To carry a
   * long conversation, use `ChatSession` (capacity per session). `"fit"` is
   * {@link Pipeline.fitCapacity}'s result.
   */
  readonly capacity?: number | "fit";
  /**
   * Prefill progress (each time one chunk is committed).
   *
   * For a long prompt the silent time until the first character is prefill itself, and this is the
   * only entry point that reports progress (`chat` streams only the decoded **text** — ADR 0084
   * decision 4).
   *
   * Exceptions from the callback are not caught (fail loudly — the whole turn fails).
   */
  readonly onPrefill?: (progress: PrefillProgress) => void;
  /**
   * Notification of generated tokens excluding stop tokens. Called synchronously, before decoding
   * and before stop-string holding. Includes special tokens that emit no text. Exceptions propagate
   * to the turn. The function is copied at issue.
   */
  readonly onToken?: (id: number) => void;
  /**
   * Interruption (checked at step boundaries; throws `signal.reason` as is — ADR 0083 decision 5).
   */
  readonly signal?: AbortSignal;
  /**
   * Whether this turn uses speculative decoding (default = `true` = gated if the pipeline has a
   * drafter — the values mean the same as {@link SequenceOptions.speculative}).
   *
   * Whether to speculate does not depend on the sampler options (even at temperature > 0 the token
   * sequence is identical to the non-speculative one). This layer only passes the option down; the
   * generation surface assembles the cycles.
   */
  readonly speculative?: boolean | "always";
};

/**
 * The internal state of {@link Pipeline} (not exposed on the public surface — exported only for the
 * mobile QAT subclass in `./qat-pipeline.ts`).
 */
export type Gemma4State = {
  /**
   * The name of the entry point used for construction ({@link gemmaEntryName}). The prefix of
   * messages thrown after construction comes from here.
   *
   * MUST: hold the **name**, not the family itself (no more family branching from this layer on).
   */
  readonly entry: string;
  readonly greedyOutput?: GemmaGreedyOutput;
  /**
   * The re-run face of the target Session (only with {@link PipelineOptions.expertCache}) — the
   * session every conversation runs through; there is no {@link Gemma4State.greedyOutput} then.
   */
  readonly expertCache?: ExpertCacheSession<GenerationContext>;
  readonly gpu: GpuContext;
  readonly ownsGpu: boolean;
  readonly session: Session;
  /**
   * The immutable setting used to build the Session. The same value goes to the intermediate memory
   * estimate (unspecified = no field = the estimate also reads the runtime default).
   */
  readonly stateAttentionReduce?: StateAttentionReduce;
  /** Likewise for the storage of the full slots; {@link speculativeSetup} reads it too. */
  readonly fullStateStorage?: SessionOptions["fullStateStorage"];
  /**
   * Likewise for the expert cache (its slots and host tier — present exactly when
   * {@link Gemma4State.expertCache} is).
   */
  readonly expertCacheOptions?: SessionOptions["expertCache"];
  /**
   * The product graph's declaration (input of `estimateSessionMemory` — the estimator of ADR 0070
   * decision 5 is a pure function of `graph + residency plan`).
   *
   * MUST: hold only the graph, not the `PreparedModel` (the same MUST as `hub/components.ts` — do not
   * keep holding the whole bytes).
   */
  readonly graph: GraphOwner["graph"];
  /**
   * The internal wiring the generation loop reads (the object passed to
   * `createGenerationSequence`).
   */
  readonly wiring: GenerationWiring;
  /**
   * The public read-out ({@link Pipeline.program}).
   *
   * A field so that the frozen narrow surface is not built on every getter call; it is derived on
   * the spot from the one {@link Gemma4State.wiring} that {@link buildGemma4Program} returned (not a
   * field updated by a separate path).
   */
  readonly program: GenerationProgram;
  /**
   * PLE's host-side cache (**what {@link Pipeline.dispose} releases**).
   *
   * MUST: the field exists only for dispose — only the closure of `wiring.derivedInputs.derive`
   * draws on it. Both come from one return of {@link buildGemma4Program}, so "only one of them
   * replaced" cannot be written, and without a release point the residency
   * ({@link PipelineOptions.maxResidentPleBytes}) would remain for the life of the process.
   *
   * Under GPU residency ({@link PipelineOptions.pleResidency} = `"gpu"`) it is **not built** —
   * the host-side row cache is never drawn on, so building only the field would leave "an LRU that is
   * never used".
   */
  readonly ple?: Gemma4Ple;
  /**
   * PLE's GPU residency (only when {@link PipelineOptions.pleResidency} = `"gpu"`).
   *
   * MUST: dispose **after the target Session** (the target's bind groups hold this resident tensor —
   * the runtime refuses, by failing loudly, a release while baked references remain).
   */
  readonly pleGpu?: Gemma4PleResident;
  /** The id a run-ahead step feeds, on a model without PLE (the PLE's resident-id form owns it otherwise). */
  readonly aheadIds?: ResidentTensor;
  readonly tokenizer: Tokenizer;
  readonly config: PipelineConfig;
  /**
   * The speculation drafter (only when {@link PipelineOptions.speculative} was given).
   *
   * MUST: dispose **before the lender (the target Session)** (the runtime refuses, by failing
   * loudly, the lender's `dispose()` while a borrower is alive — the borrower's bind groups hold the
   * lender's weights).
   */
  readonly drafter?: Gemma4Drafter;
  /**
   * The number of drafts used per cycle (only when {@link PipelineOptions.speculative} was
   * given — default {@link GEMMA4_DRAFT_STEPS}).
   *
   * MUST: the field exists so that the two consumers of speculation (generation's `speculative.k`
   * and the verify scenario of `k+1` rows in {@link Pipeline.estimateSessionMemory}) see **the
   * same value** — if they diverged, "a run of a different shape from the estimated one" would run.
   */
  readonly speculativeK?: number;
  /**
   * The self-tuning gate's knobs (only when the `gate` of {@link PipelineOptions.speculative}
   * was given — default: the defaults of `speculation-gate.ts`).
   *
   * MUST: the value check is already done at the pipeline-building step ({@link assertSpeculative}) —
   * this field only exists "to pass through on every generation", and nothing past here interprets it.
   */
  readonly speculativeGate?: SpeculationGateOptions;
  /** The image embedder (only when {@link PipelineOptions.vision} was given). */
  readonly vision?: Gemma4Vision;
  /**
   * The slot-backing budget passed to the Session ({@link PipelineOptions.planBackingBudgetBytes};
   * the runtime default when unspecified).
   *
   * MUST: the field exists so that {@link Pipeline.estimateSessionMemory} passes **the same
   * value** to the estimate — if the held value and the estimate's premise diverged, the reported peak
   * would name a different budget from the actual held set.
   */
  readonly planBackingBudgetBytes?: number;
  /**
   * The row-block limit passed to the Session ({@link PipelineOptions.attentionRowBlockMaxBytes};
   * only the binding limit when unspecified).
   *
   * MUST: the field exists so that {@link Pipeline.estimateSessionMemory} passes **the same
   * value** to the estimate — if the two diverged, the reported peak would count a different block
   * split from the one the Session runs.
   */
  readonly attentionRowBlockMaxBytes?: SessionOptions["attentionRowBlockMaxBytes"];
  /** The per-run observation callback ({@link PipelineOptions.onRunDiagnostics}). */
  readonly onRunDiagnostics?: (diagnostics: SessionDiagnostics, phase: RunPhase) => void;
};

/** The image embedder of one pipeline. */
type Gemma4Vision = {
  readonly session: Session;
  /** For the estimate only (the same reason as the drafter's). */
  readonly graph: GraphOwner["graph"];
  /** The embedder graph's one output. */
  readonly output: string;
  /** Its row symbol ({@link Pipeline.estimateSessionMemory} binds it to the budget). */
  readonly symbol: string;
  readonly config: PipelineVisionConfig;
};

/**
 * Of the material that passed admission, the index the family check settled and the manifest's
 * assets (the two surfaces prepare them by different paths, and the interpretation is gathered in
 * one place).
 */
type Gemma4SidecarAssets = {
  readonly tokenizer: Uint8Array<ArrayBuffer>;
  /**
   * The **parsed** PLE index — absent for a PLE-free model (admission settled that the graph and the
   * container agree on it — {@link admitGemma4Ple}).
   *
   * MUST: parsed only once per surface (`ple_index` is not opened twice). Matching the index against
   * the container's assets (`assertGemma4PleAssets` in `./ple-index.ts`) also needs the index, so that
   * one parse is lifted into the admission step.
   */
  readonly pleIndex?: Gemma4PleIndex;
};

/**
 * How the per-layer inputs reach the graph: not at all (`"none"` — a PLE-free model), through the
 * host gather (`"host"`) or through the GPU-resident gather (`"gpu"`). Settled once per build
 * ({@link gemma4PlePlan}) so that "PLE-free" and "where PLE lives" cannot disagree.
 */
type Gemma4PlePlan =
  | { readonly residency: "none" }
  | { readonly residency: PleResidency; readonly index: Gemma4PleIndex };

/**
 * Turns fetched bytes into an ArrayBuffer that can be passed to `openContainer` (the body of the
 * check is {@link readAssetBuffer}, shared by all families — a product graph part is 256 MiB-class,
 * so it is not copied).
 */
const assetBuffer = (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer(where, "weights / assets", assets, key);

const assetBytes = (
  where: string,
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): Uint8Array<ArrayBuffer> => {
  assetBuffer(where, assets, key);
  return assets[key];
};

/**
 * The check of {@link PipelineOptions.speculative} (fails synchronously **before fetching a
 * single byte of the assets**) — returns the resolved `k` when it passes.
 *
 * MUST: `1 ≤ k ≤ {@link GEMMA4_DRAFT_STEPS}`. The distribution's drafter graph is built with 3 steps
 * and the number of exits is the upper limit; below it is "speculation that takes no draft at all" =
 * a meaningless option. Accepting an out-of-range value would silently let "taking a different
 * number of drafts than declared" through (the generation surface's `assertSpeculativeSetup` checks
 * the same relation, but it fails only **after** the GB-scale load).
 *
 * The check of the gate's knobs (`gate`) runs at the same step — build one
 * {@link createSpeculationGate} and discard it. The gate itself is the source of truth for the value
 * ranges, so the same check is not implemented twice.
 */
const assertSpeculative = (
  where: string,
  speculative: NonNullable<PipelineOptions["speculative"]>,
): number => {
  const k = speculative.k ?? GEMMA4_DRAFT_STEPS;
  if (!Number.isSafeInteger(k) || k < 1 || k > GEMMA4_DRAFT_STEPS) {
    // `k` violates the value range of a caller option, so it is an input error (ADR 0107 decision
    // 2). Its type matches the session / PLE checks in the same early block. Matching against the
    // graph declaration (the generation surface's `assertSpeculativeSetup`) is an asset mismatch, so
    // that stays a plain `Error`.
    throw new ModelInputError(
      `${where}: speculative.k ${k} is outside 1..${GEMMA4_DRAFT_STEPS}` +
        ` (the distribution's drafter is baked with ${GEMMA4_DRAFT_STEPS} steps — the number of outputs is the upper limit)`,
    );
  }
  if (speculative.gate !== undefined) createSpeculationGate(speculative.gate);
  return k;
};

/**
 * Settles where PLE is placed (fails synchronously **before reading a single byte of the assets** —
 * ADR 0085 addendum).
 *
 * MUST: knobs that cannot be combined are refused here. Each is a mistake of the form "the option is
 * not a value to be silently ignored"; letting it through would leave "GPU residency that is not
 * faster" or "a budget that has no effect" with neither an exception nor a warning (the same style
 * as the missing drafter in {@link speculativeSetup}).
 *
 * NOTE: `export` is for the test that calls the check directly (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */
export const resolveGemma4PleResidency = (
  where: string,
  options: PipelineOptions,
): PleResidency => {
  if (options.pleResidency === undefined) return "host";
  const residency = assertGemma4PleResidency(where, options.pleResidency);
  if (residency === "host") return residency;
  // Both are violations of combinations decided **only** by the caller's options, so they are input
  // errors (ADR 0107 decision 2).
  if (options.maxResidentPleBytes !== undefined) {
    throw new ModelInputError(
      `${where}: maxResidentPleBytes has no effect with pleResidency: "gpu"` +
        ` (GPU residency never looks up the host-side row cache)`,
    );
  }
  if (options.speculative !== undefined) {
    throw new ModelInputError(
      `${where}: pleResidency: "gpu" and speculative cannot be combined` +
        ` (the drafter's per-layer inputs are outside the scope of GPU residency — ADR 0085 addendum, GPU residency)`,
    );
  }
  return residency;
};

/**
 * Refuses the PLE placement options for a PLE-free model (gemma-4-12B-it) — called as soon as
 * admission has settled that the model has no PLE ({@link admitGemma4Ple}), before any weight part
 * is fetched.
 *
 * MUST: refuse even `pleResidency: "host"`. Each option is "where PLE lives" or "how much of it to
 * keep"; for a model without PLE neither has anything to act on, and accepting them would let a
 * caller believe a budget or a placement was applied (the same style as {@link resolveGemma4PleResidency}).
 * Input errors (ADR 0107 decision 2) — dropping the option makes the call pass.
 *
 * NOTE: `export` is for the test that calls the check directly (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */
export const assertGemma4PleFreeOptions = (where: string, options: PipelineOptions): void => {
  const given = (["pleResidency", "maxResidentPleBytes"] as const).filter(
    (key) => options[key] !== undefined,
  );
  if (given.length === 0) return;
  throw new ModelInputError(
    `${where}: ${given.join(" / ")} cannot be used with this model` +
      ` (its graph has no '${PER_LAYER_INPUTS}' input and its container no PLE assets — there is no PLE to place)`,
  );
};

/**
 * Settles the {@link Gemma4PlePlan} of one build from the admitted PLE index (absent = PLE-free) and
 * the placement options.
 */
const gemma4PlePlan = (
  where: string,
  options: PipelineOptions,
  index: Gemma4PleIndex | undefined,
): Gemma4PlePlan => {
  if (index === undefined) {
    assertGemma4PleFreeOptions(where, options);
    return { residency: "none" };
  }
  return { residency: resolveGemma4PleResidency(where, options), index };
};

/** Whether {@link PipelineOptions.vision} asks for the embedder (a non-boolean fails loudly). */
const wantsGemma4Vision = (where: string, options: PipelineOptions): boolean => {
  const vision: unknown = options.vision;
  if (vision !== undefined && typeof vision !== "boolean") {
    throw new ModelInputError(`${where}: vision ${JSON.stringify(vision)} is not a boolean`);
  }
  return vision === true;
};

/**
 * The checks of {@link PipelineOptions.decodeAhead} that the options alone decide (the integer and
 * `speculative`), returning the depth — run **before reading a single byte of the assets**, like
 * {@link resolveGemma4PleResidency}; the model-dependent ones follow in {@link resolveGemma4DecodeAhead}.
 */
const assertGemma4DecodeAheadOptions = (where: string, options: PipelineOptions): number => {
  const depth = options.decodeAhead ?? 1;
  if (!Number.isSafeInteger(depth) || depth < 1) {
    throw new ModelInputError(
      `${where}: decodeAhead ${JSON.stringify(depth)} is not an integer ≥ 1 (1 = one decode step per batch)`,
    );
  }
  if (depth === 1) return depth;
  if (options.speculative !== undefined) {
    throw new ModelInputError(
      `${where}: decodeAhead ${depth} and speculative cannot be combined` +
        " (a speculative turn verifies drafts instead of taking greedy steps)",
    );
  }
  return depth;
};

/**
 * The check of {@link PipelineOptions.fullStateStorage}, run **before reading a single byte of the
 * assets**.
 *
 * The option is not on the manifest's session vocabulary, so `resolveSessionOptions` never sees it,
 * and without this it would first fail at Session construction, after the GB-scale load (and `null`
 * would silently read as the default there).
 *
 * The gate's knobs are refused here because every gate turn is refused under f16
 * ({@link speculativeSetup}), so they could never be read (the option is not a value to be silently
 * ignored). Which policy a turn uses is the turn's option, so the turn itself is refused only when it
 * starts.
 *
 * NOTE: `export` is for the test that calls the check directly (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */
export const assertGemma4FullStateStorage = (where: string, options: PipelineOptions): void => {
  const storage = options.fullStateStorage;
  if (storage === undefined) return;
  if (storage !== "f32" && storage !== "f16") {
    throw new ModelInputError(
      `${where}: fullStateStorage ${
        typeof storage === "string" ? JSON.stringify(storage) : typeof storage
      } is not supported (accepted: 'f32' / 'f16')`,
    );
  }
  if (storage === "f16" && options.speculative?.gate !== undefined) {
    throw new ModelInputError(
      `${where}: speculative.gate has no effect with fullStateStorage: "f16"` +
        ` (the self-tuning gate is refused there; turns run with speculative: "always" or false)`,
    );
  }
};

/**
 * Settles the decode run-ahead depth ({@link PipelineOptions.decodeAhead} — the option checks of
 * {@link assertGemma4DecodeAheadOptions}, then the model's: called once admission has settled
 * whether the model has PLE, before any weight part is fetched).
 *
 * Every refusal here is decided by the caller's options and the admitted model — whether it has
 * PLE — so each is an input error (ADR 0107 decision 2). The upper bound — the context's sliding slack — exists only once a conversation's context does, so the
 * generation surface checks it there (`assertGreedyAheadSetup`).
 *
 * MUST: a depth ≥ 2 that cannot run fails loudly rather than silently decoding per step — "the
 * run-ahead did not make it faster" would otherwise remain with neither an exception nor a warning.
 *
 * NOTE: `export` is for the test that calls the check directly (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */
export const resolveGemma4DecodeAhead = (
  where: string,
  options: PipelineOptions,
  model: { readonly ple: boolean },
): number => {
  const depth = assertGemma4DecodeAheadOptions(where, options);
  if (depth === 1 || !model.ple) return depth;
  if (resolveGemma4PleResidency(where, options) !== "gpu") {
    throw new ModelInputError(
      `${where}: decodeAhead ${depth} needs pleResidency: "gpu"` +
        " (with the host PLE every step's per-layer inputs are gathered on the host from the step's token," +
        " and the run-ahead keeps that token on the GPU)",
    );
  }
  return depth;
};

/**
 * If this conversation speculates, assembles the DI set passed to the generation surface (the
 * resolution of {@link SequenceOptions.speculative} / {@link ChatOptions.speculative} —
 * the one path shared by chat and sequence).
 *
 * The default is "speculate if there is a drafter", and only `false` explicitly cancels it. The
 * sampler options are not looked at (speculation runs regardless of temperature, and the token
 * sequence is identical to the non-speculative one — ADR 0096 decision 7). `"always"` is the option
 * that builds no self-tuning gate (A/B, acceptance checks, measurement), passed straight down to
 * the generation surface's `policy`.
 *
 * The pipeline's gate knobs (the `gate` of {@link PipelineOptions.speculative}) go down **only
 * to `"auto"` turns** (`"always"` builds no gate, so nothing would read them). Alternating `false` /
 * `"always"` / `true` on one pipeline was the shape of the A/B script (`tools/mtp-bench`, removed in
 * 0.15), so the combination itself is not a mistake — the range where the knobs apply is stated in
 * the doc of {@link PipelineOptions.speculative} (the bench README that also stated it
 * (`config.gate`) was removed with the tool).
 */
export const speculativeSetup = (
  state: Pick<Gemma4State, "drafter" | "speculativeK" | "speculativeGate" | "fullStateStorage">,
  enabled: boolean | "always" | undefined,
): GenerationSpeculativeOptions<GenerationContext> | undefined => {
  const drafter = state.drafter;
  if (enabled === false) return undefined;
  if (drafter === undefined) {
    // Unspecified (undefined) means "speculate if there is a drafter"; an explicit option requires a
    // drafter — silently running non-speculatively would leave no way to read the ignoring from the
    // result either (the absence of the `speculation` field).
    if (enabled !== undefined) {
      throw new Error(
        `speculative: ${JSON.stringify(enabled)} was passed, but this pipeline was opened without` +
          ` a drafter`,
      );
    }
    return undefined;
  }
  // MUST: under f16 storage the gate is refused, because it would make the output depend on timing
  // (ADR 0110 decision 6). Verify rows j >= 1 read earlier draft rows from `ins` in f32, while decode
  // reads them from the slot in f16, so the two shapes the gate switches between on wall time no
  // longer give the same bits. A pipeline without a drafter never gets here: it has no gate to refuse.
  if (enabled !== "always" && state.fullStateStorage === "f16") {
    throw new ModelInputError(
      `speculative: ${JSON.stringify(enabled)} selects the self-tuning gate, which fullStateStorage: "f16"` +
        ` refuses (it would make the output depend on timing): pass speculative: "always" or false`,
    );
  }
  // The gate's knobs are not passed to `"always"` (there is no gate, so nothing reads them) — not
  // adding the field at all is how "this turn has no gate" is spelled.
  const gate = enabled === "always" ? undefined : state.speculativeGate;
  return {
    // The borrower context is opened once when the sequence is created, and the sequence's dispose
    // closes it **before the lender**.
    open: (context: GenerationContext): Promise<DraftFace> => openGemma4DraftFace(drafter, context),
    k: state.speculativeK ?? GEMMA4_DRAFT_STEPS,
    policy: enabled === "always" ? "always" : "auto",
    ...(gate === undefined ? {} : { gate }),
  };
};

/**
 * The two `generation` fields of `estimateGraphMemory` that {@link gemma4EstimateGenerations}
 * returns.
 */
export type Gemma4EstimateGenerations = {
  /** The options passed to the estimate of the main model (the lender). */
  readonly target: NonNullable<GraphEstimateOptions["generation"]>;
  /**
   * The options passed to the estimate of the drafter (the borrower context) (not read for a
   * pipeline without a drafter).
   */
  readonly borrower: NonNullable<GraphEstimateOptions["generation"]>;
};

/**
 * Builds the `generation` options (chunk length, buckets, additional scenarios) that
 * {@link Pipeline.estimateSessionMemory} passes to `estimateGraphMemory`, for both the main
 * model and the drafter.
 *
 * It is a separate function because this wiring cannot be observed from the estimate values of the
 * real assets — in gemma4 E2B's default configuration the bucket shapes do not dominate the peak, so
 * removing a line of the wiring does not move `peakAccountedBytes`.
 *
 * NOTE: `export` is for the test that calls the wiring directly without a GPU (not exposed on
 * `mod.ts` / the subpath surface — ADR 0008).
 */
export const gemma4EstimateGenerations = (
  wiring: GenerationWiring,
  shape: {
    readonly capacity: number;
    readonly chunkLength: number;
    /**
     * The row count `k+1` that speculative verify runs in one run (omitted for a pipeline without a
     * drafter).
     */
    readonly verifyQueryLength?: number;
  },
): Gemma4EstimateGenerations => {
  const bindings = { [wiring.capacitySymbol]: shape.capacity };
  const verifyQueryLength = shape.verifyQueryLength;
  // MUST: pass the bucket shapes the sequence runs (otherwise, at a block-count boundary where a
  // bucket shape is larger than the prefill shape, the peak comes out too small — ADR 0095 addendum
  // 2026-10-10).
  const chunkBuckets = wiring.chunkBuckets.filter((bucketRows) => bucketRows < shape.chunkLength);
  return {
    target: {
      chunkLength: shape.chunkLength,
      bindings,
      chunkBuckets,
      // Speculative verify runs `k+1` rows in one run (neither prefill nor decode).
      // R is k+1; the physical row count M is the same as the shape the sequence runs (the row count
      // rounded to a bucket — naming `chunkLength: k+1` would estimate a shape smaller than the real
      // run for k < 3).
      // MUST: the rounding is done against the context **being estimated** (the assumed chunk length
      // and the narrowed ladder), not the pipeline's wiring. When k+1 exceeds the ladder's largest
      // bucket, it rounds to the chunk length, and the pipeline's chunk length is not in the assumed
      // context's allowed set {1, ladder, assumed chunk length}, so the estimate would throw.
      ...(verifyQueryLength === undefined
        ? {}
        : {
            scenarios: [
              {
                name: "verify",
                chunkLength: physicalChunkRows(verifyQueryLength, {
                  chunkLength: shape.chunkLength,
                  chunkBuckets,
                }),
                bindings: { [wiring.rowSymbol]: verifyQueryLength },
              },
            ],
          }),
    },
    // The borrower context has `chunkLength` exactly 1 and inherits the capacity symbol from the
    // lender. No buckets are passed — the borrower runs only the M = 1 shape, and its accepted set
    // (at least 2 and below `chunkLength`) is empty.
    borrower: { chunkLength: 1, bindings },
  };
};

/** The soft inputs a product graph declares ({@link GenerationProgramSpec.softInputs}). */
const softInputsOf = (
  graph: GraphOwner["graph"],
): { readonly softInputs: GenerationSoftInputs } | Record<string, never> => {
  const inputOf = (name: string) => graph.inputs.find((input) => input.name === name);
  const embeds = inputOf(GEMMA4_SOFT_EMBEDS);
  if (embeds === undefined && inputOf(GEMMA4_SOFT_MASK) === undefined) return {};
  const hidden = embeds?.shape[2];
  return {
    softInputs: {
      embeds: GEMMA4_SOFT_EMBEDS,
      mask: GEMMA4_SOFT_MASK,
      // A symbolic width reads as not an integer, which the program's check refuses by name.
      hidden: typeof hidden === "number" ? hidden : Number.NaN,
    },
  };
};

/**
 * Builds the static wiring from the material that passed admission + the assets (shared by
 * `fromAssets` and `fromPretrained`).
 *
 * This runs the whole cross-check of the id spaces (ADR 0085 decision 5) — ① the ids the tokenizer
 * can produce ② the vocab rows of the main embedding ③ the rows of the PLE index (a PLE-free model
 * has no ③).
 *
 * It is also where the distribution's declarations reach the generation surface: the chat dialect
 * goes into the tokenizer and the suppressed set into the wiring, and both surfaces build their
 * sequences from exactly these two objects.
 *
 * NOTE: `export` is for the GPU-free test of that hand-over (`tests/gemma4_config_test.ts` — a
 * Session needs a real device, so the whole construction cannot run without one). It is not on the
 * `mod.ts` / subpath surface (ADR 0008).
 */
export const buildGemma4Program = (
  admitted: Gemma4Admission,
  open: ComponentOpener,
  tokenizerBytes: Uint8Array<ArrayBuffer>,
  options: PipelineOptions,
  admission: GemmaFamilyAdmission,
  plePlan: Gemma4PlePlan,
): {
  readonly wiring: GenerationWiring;
  readonly tokenizer: Tokenizer;
  readonly ple?: Gemma4Ple;
} => {
  const { config, vocabSize, capacitySymbol } = admitted;
  const entry = gemmaEntryName(admission.family);
  // The supplier is **the same one** admission looked at (`open` only looks up the opened component).
  const model = open(MODEL);
  if (admission.family === "gemma4-qat") {
    // The mobile QAT models always carry PLE (`admitGemma4Qat` refuses a graph without the input),
    // so reaching here without an index is a broken invariant, not a model to run.
    if (plePlan.residency === "none") {
      throw new Error(
        `${entry}: internal inconsistency: a mobile QAT model was admitted without PLE`,
      );
    }
    // The model name is the value admission settled, used as is (not re-derived from the layer count).
    assertGemma4QatPle(admission.model, model.graph, plePlan.index);
  }
  const familyRopeInputs = admission.family === "gemma4-qat" ? qatRopeInputs : ropeInputs;
  // The chat dialect rides on the tokenizer, so `chatPrompt` / `chatTurn` render the template of
  // the model this tokenizer serves (`Tokenizer.chatDialect`).
  const tokenizer = new Tokenizer(
    parseGemmaTokenizerAsset(tokenizerBytes),
    config.chatDialect ?? DEFAULT_CHAT_DIALECT,
  );
  // ① the ids the tokenizer can produce and ② the vocab rows of the main embedding.
  if (tokenizer.maxTokenId >= vocabSize) {
    throw new Error(
      `${entry}: the tokenizer's largest token id ${tokenizer.maxTokenId} is outside` +
        ` the main embedding's ${vocabSize} vocab rows (a combination baked with a different vocabulary)`,
    );
  }
  // ③ the rows of the PLE index (`createGemma4Ple` owns this check — the same check is not
  // implemented twice). Under GPU residency no host gather is built, and `createGemma4PleResident`
  // runs the same check. A PLE-free model builds neither.
  const ple =
    plePlan.residency === "host"
      ? createGemma4Ple({
          index: plePlan.index,
          openBlock: (asset) => model.asset(asset),
          vocabSize,
          ...(options.maxResidentPleBytes === undefined
            ? {}
            : { maxResidentBytes: options.maxResidentPleBytes }),
        })
      : undefined;

  const chunkLength = assertChunkLength(options.chunkLength ?? config.chunkLength, config, entry);
  const wiring = createGenerationProgram({
    graph: model.graph,
    inputIds: INPUT_IDS,
    lastRow: LAST_ROW,
    // The exits are taken by order ({@link GRAPH_OUTPUTS} — admission has already checked the count).
    logits: model.graph.outputs[0],
    hidden: model.graph.outputs[1],
    chunkLength,
    chunkBuckets: assertGemma4ChunkBuckets(
      options.chunkBuckets ?? defaultChunkBuckets(chunkLength),
      chunkLength,
      entry,
    ),
    maxPosition: config.maxPosition,
    capacity: config.capacity,
    vocabSize,
    // The stop set is derived from the added vocabulary of the tokenizer asset (ADR 0083 decision 8
    // / 0084 decision 5 — it comes from the same digest set as the chat format).
    stopTokens: gemma4StopTokens(tokenizer),
    // The distribution's suppressed set (gemma-4-12B-it's `<image|>` / `<audio|>`), masked before
    // every host selection; absent for E2B / E4B, whose selection is unchanged.
    ...(config.suppressTokens === undefined ? {} : { suppressTokens: config.suppressTokens }),
    capacitySymbol,
    // The PLE gather and the RoPE cos / sin are plugged into the host-derived per-chunk input field
    // (ADR 0085 / this wave). `options` goes straight down to PLE — so that reading a 32 MiB-class
    // block does not become a section the interruption cannot reach (rope is a synchronous
    // computation and opens no interruption window).
    // Under GPU residency the producer of `per_layer_inputs` moves to the Session side (a resident
    // input), so it is removed from the host-derived field and its coverage is declared only through
    // `residentInputs`. A PLE-free model has no such input: the field supplies only RoPE.
    derivedInputs:
      ple === undefined
        ? {
            names: gemma4RopeInputNames(),
            derive: (_ids, positions) => Promise.resolve(familyRopeInputs(config.rope, positions)),
            // Only RoPE is left on the host, and it reads positions alone (the run-ahead derives a
            // step before its token exists — `decodeAhead`).
            positionsOnly: true,
          }
        : {
            names: [PER_LAYER_INPUTS, ...gemma4RopeInputNames()],
            derive: async (ids, positions, deriveOptions) => ({
              [PER_LAYER_INPUTS]: await ple.gather(ids, deriveOptions),
              ...familyRopeInputs(config.rope, positions),
            }),
          },
    ...(plePlan.residency === "gpu" ? { residentInputs: [PER_LAYER_INPUTS] } : {}),
    // MUST: follows the graph, not the `vision` option. A graph with these inputs needs them bound
    // on every run, so a text-only pipeline over it binds zeros and false like any other run. A graph
    // with only one of the two fails in the program's check, by name.
    ...softInputsOf(model.graph),
  });
  return { wiring, tokenizer, ...(ple === undefined ? {} : { ple }) };
};

/**
 * One image's rows → vectors on {@link Gemma4State.vision}. The caller holds the pipeline's
 * serialization slot, and a pipeline without the embedder never reaches here.
 */
const embedRows = (state: Gemma4State, rows: ImageRows): Promise<ImageEmbedding> => {
  const { vision } = state;
  if (vision === undefined) throw new Error(`${state.entry}: no image embedder is built`);
  return runImageEmbedder(vision.session, rows, vision.output, vision.config.hiddenSize);
};

/**
 * The gemma4 chat pipeline (one product graph + host PLE + tokenizer).
 *
 * The only construction entry points are {@link Pipeline.fromPretrained} (fetching from HF) and
 * {@link Pipeline.fromAssets} (bytes already fetched) — the constructor is private so that no
 * half-built state that bypasses the asset checks can be created (ADR 0008).
 *
 * Exported only for the mobile QAT subclass in `./qat-pipeline.ts`; no entry re-exports it.
 */
export class GemmaPipeline {
  readonly #state: Gemma4State;
  /**
   * The serialization chain of chat and dispose (one Session is not pushed by two conversations at
   * once).
   */
  readonly #chain = createOperationChain();
  /**
   * The objects {@link Pipeline.sequence} handed out (closes the gap of a missed dispose).
   */
  readonly #handed = new Set<GenerationSequence>();
  /** The one dispose. **Not undefined means "disposed"** (no separate derived state). */
  #disposal: Promise<void> | undefined;

  protected constructor(state: Gemma4State) {
    this.#state = state;
  }

  /** Assembly for the shared factory. Checks the family's contract and returns the state. */
  protected static async loadPretrained(
    family: GemmaFamily,
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Gemma4State> {
    const entry = gemmaEntryName(family);
    const where = `${entry}.fromPretrained`;
    if (family === "gemma4-qat" && options.speculative !== undefined) {
      throw new Error(
        `${where}: MTP is not supported for the mobile QAT line (it ships no drafter; the q4_0` +
          " QAT line's drafters run through gemma.Pipeline)",
      );
    }
    const vision = wantsGemma4Vision(where, options);
    if (family === "gemma4-qat" && vision) {
      throw new Error(
        `${where}: vision is not supported for the mobile QAT line (it ships no image embedder; ` +
          "12B of the q4_0 QAT line runs through gemma.Pipeline)",
      );
    }
    if (options.speculative !== undefined) {
      assertSpeculative(where, options.speculative);
    }
    // MUST: the placement option is checked **before reading a single byte of the assets** (`#build`
    // goes through the same function too, but that is after the GB-scale load).
    resolveGemma4PleResidency(where, options);
    assertGemma4DecodeAheadOptions(where, options);
    assertGemma4FullStateStorage(where, options);
    assertGemma4ExpertCacheOptions(where, options);
    // Both families share this load body, so each names its own table in the omitted-ref hint.
    const source = toManifestSource(
      ref,
      where,
      family === "gemma4"
        ? 'SOURCES["gemma4"] (@karume/models/gemma)'
        : 'SOURCES["gemma4-qat-mobile"] (@karume/models/gemma4-qat-mobile)',
    );
    const hubOptions: FetchAssetsOptions = hubLoadOptions(options);
    const loaded = await loadManifest(source, hubOptions);
    const choice = {
      ...(options.model === undefined ? {} : { model: options.model }),
      ...(options.quant === undefined ? {} : { quant: options.quant }),
    };
    // MUST: narrow the weights to fetch by role (`ResolveOptions.weights`). If the drafter stayed in
    // the table for a load that does not use speculation, its descriptor and weight parts would be
    // downloaded too.
    const componentKeys = [
      MODEL,
      ...(options.speculative === undefined ? [] : [DRAFTER]),
      ...(vision ? [VISION] : []),
    ];
    const selection = resolveSelection(loaded.manifest, {
      ...choice,
      weights: componentKeys,
    });
    const { admitted, assets, open } = await loadContainerComponents(
      where,
      loaded,
      selection,
      componentKeys,
      // The family check runs at the admission step (before fetching the weight parts —
      // `src/hub/components.ts`).
      async (open) => {
        const { config, quantName, quant } = gemma4ManifestConfig(loaded.manifest, choice, family);
        // Unsupported declarations are never ignored in order to run anyway. They are rejected before fetching the
        // weight parts.
        const quantSession = resolveGemmaSessionOptions(
          quant.session,
          options,
          `${where}: quant '${quantName}'`,
        );
        const quantWhere = `${entry}: quant '${quantName}'`;
        const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, quantSession);
        const model = open(MODEL);
        const admitted = admitGemma4(
          model,
          config,
          options.speculative === undefined ? undefined : open(DRAFTER),
          entry,
        );
        const visionAdmission = vision
          ? assertGemma4Vision(model, open(VISION), config, admitted.vocabSize, entry)
          : undefined;
        if (visionAdmission !== undefined) {
          assertGemma4VisionChunkLength(
            options.chunkLength ?? config.chunkLength,
            config,
            visionAdmission.config,
            entry,
          );
        }
        // MUST: **carry, not discard,** the model name settled from the graph (the PLE check does not
        // re-derive it from the layer count).
        const admission: GemmaFamilyAdmission =
          family === "gemma4-qat"
            ? {
                family,
                model: admitGemma4Qat(model.graph, choice.model ?? loaded.manifest.defaultModel),
              }
            : { family };
        // Whether the model has PLE is settled from the declarations (graph + container), and a
        // PLE-free model refuses the PLE placement options here — before any weight part.
        const hasPle = admitGemma4Ple(model, entry);
        if (!hasPle) assertGemma4PleFreeOptions(where, options);
        // The cache's graph checks (expert_linear, the slots, the chunk rows) — before any weight part.
        resolveGemma4ExpertCache(where, model.graph, options, config.chunkLength);
        // MUST: a shared GPU's missing features fail at this step (unlike a device acquired here,
        // they cannot be requested, and this is the only check that knows before downloading the
        // weights — the later check is the same one in `session/gpu-features.ts`).
        if (options.gpu !== undefined) {
          assertGpuFeaturesGranted(gpuFeatures, options.gpu, quantWhere);
        }
        // The `requiredLimits` the distribution declared are checked here **before fetching the
        // weight parts** (ADR 0089 decision 5 — the limits of a shared GPU, or the adapter's measured
        // values on the path that acquires its own).
        // Unlike the other families this sits in the closure because {@link admitGemma4} takes no
        // construction options (it is decided by the graph alone).
        await assertRequiredLimitsBeforeDownload(quant.requiredLimits, options.gpu, quantWhere);
        // The PLE index is **an asset of the container** (ADR 0109 decision 4), so it can be read and
        // fully matched at this step — enumerating every entry: whether each block the index points
        // to is in the container, and whether its role and length are as declared.
        //
        // MUST: placed **after every check** that can fail on the declaration alone. This is the only
        // step that touches a component's bytes (one index of a few KB — not a single `values` /
        // `scales` block is read). "Not fetching a single byte of weights" holds because of the
        // writer's convention that whole-read assets **do not share a part with weight blocks**
        // (ADR 0109 decision 4 / container-v1 §4.2), so fetching the index's part does not reach the
        // weight parts.
        const pleIndex = hasPle ? await readGemma4PleIndex(where, model) : undefined;
        return {
          ...admitted,
          ...(pleIndex === undefined ? {} : { pleIndex }),
          ...(visionAdmission === undefined ? {} : { vision: visionAdmission }),
          quantSession,
          admission,
          requested: { gpuFeatures, where: quantWhere },
        };
      },
      {
        ...hubOptions,
        ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
        ...(options.components === undefined ? {} : { components: options.components }),
      },
    );
    return await GemmaPipeline.#build(
      admitted.admission,
      admitted,
      admitted.requested,
      open,
      {
        tokenizer: assetBytes(where, assets, TOKENIZER_ASSET),
        ...(admitted.pleIndex === undefined ? {} : { pleIndex: admitted.pleIndex }),
      },
      {
        ...options,
        ...admitted.quantSession,
      },
    );
  }

  /** Assembly for the shared factory. Checks the family's contract and returns the state. */
  protected static async loadAssets(
    family: GemmaFamily,
    input: Assets,
    options: PipelineOptions = {},
  ): Promise<Gemma4State> {
    const where = `${gemmaEntryName(family)}.fromAssets`;
    // MUST: this surface has no place for a drafter ({@link Assets} holds one product graph).
    // Silently building without speculation would be "specified but no effect", so it is refused by
    // failing loudly.
    if (options.speculative !== undefined) {
      throw new Error(
        `${where}: speculative is not accepted` +
          ` (gemma.Assets has no drafter container — build speculation from fromPretrained)`,
      );
    }
    // The same for the embedder: "specified but no effect" is refused.
    if (wantsGemma4Vision(where, options)) {
      throw new Error(
        `${where}: vision is not accepted` +
          ` (gemma.Assets has no vision container — build image input from fromPretrained)`,
      );
    }
    // The placement option is checked before opening a single byte of the assets (the same position
    // as in `loadPretrained`).
    resolveGemma4PleResidency(where, options);
    assertGemma4DecodeAheadOptions(where, options);
    assertGemma4FullStateStorage(where, options);
    assertGemma4ExpertCacheOptions(where, options);
    // MUST: the explicit quant execution knobs are checked **here** too. This surface has no manifest
    // and cannot go through `resolveGemmaSessionOptions`; without this the same mistaken option would
    // travel down to Session construction and turn into the runtime's `ExecutionError` = the
    // classification would split per entry point (ADR 0107 decision 2).
    assertGemmaSessionOverrides(options, where);
    const config = parsePipelineConfig(input.config);
    if (input.model.length === 0) {
      throw new Error(
        `${where}: the product graph's part list is empty (list them in index order from part 0)`,
      );
    }
    const parts = Object.fromEntries(
      input.model.map((bytes, index): readonly [string, Uint8Array<ArrayBuffer>] => [
        `${MODEL}[${index}]`,
        bytes,
      ]),
    );
    const open = await assetComponentOpener(where, parts, (key) => assetBuffer(where, parts, key), [
      MODEL,
    ]);
    // NOTE: this surface has no `requiredLimits` check — {@link Assets} has no manifest (only
    // bytes and `config`), so the declaration itself cannot be reached. The check against actual sizes
    // is done by `assertWeightsWithinLimits` at Session construction (ADR 0089 decision 1).
    const model = open(MODEL);
    const admitted = admitGemma4(model, config, undefined, gemmaEntryName(family));
    // The model name is determined from the graph and the PLE configuration (this surface has no
    // manifest).
    const admission: GemmaFamilyAdmission =
      family === "gemma4-qat" ? { family, model: admitGemma4Qat(model.graph) } : { family };
    // The PLE-free decision and its option check, in the same order as the fetch surface. The PLE
    // index is a container asset, so it goes through **the same one** check as the fetch surface
    // (`readGemma4PleIndex` checks the consistency of the index and the blocks too).
    const hasPle = admitGemma4Ple(model, gemmaEntryName(family));
    if (!hasPle) assertGemma4PleFreeOptions(where, options);
    const pleIndex = hasPle ? await readGemma4PleIndex(where, model) : undefined;
    // This surface has no manifest, so there is no declaration — only the features the effective
    // settings need are requested.
    const requested = { gpuFeatures: sessionGpuFeatures(undefined, options), where };
    return await GemmaPipeline.#build(
      admission,
      admitted,
      requested,
      open,
      {
        tokenizer: input.tokenizer,
        ...(pleIndex === undefined ? {} : { pleIndex }),
      },
      options,
    );
  }

  /**
   * Builds from the material that passed admission + the assets (the one path shared by the two
   * surfaces).
   *
   * MUST: the assets are parsed **before going to get the GPU** (the same order as the other
   * families) — in a different order, on an environment without a GPU the failure would turn into a
   * different exception, and "what was wrong" would not reach the reader.
   *
   * The Session is kept for the same reason as siglip2: there is nothing to make room for (no other
   * huge graph that cannot be loaded at the same time) — rebuilding it per conversation would
   * re-upload 1.5 GiB of weights every time.
   *
   * When speculation is requested, one drafter Session is also set up **here** (not per conversation
   * — the ruling of ADR 0096 stage 2, "the binding point is the context"). The target comes first, and
   * the drafter borrows its embedding table.
   *
   * @param requested The features to request (the quant declaration ∪ the features the effective
   *   settings need — `sessionGpuFeatures`) and the subject of its diagnostics. The drafter Session
   *   is set up on the same device, so nothing is requested separately for the drafter.
   */
  static async #build(
    admission: GemmaFamilyAdmission,
    admitted: Gemma4Admission,
    requested: { readonly gpuFeatures: GpuFeaturesSpec | undefined; readonly where: string },
    open: ComponentOpener,
    assets: Gemma4SidecarAssets,
    options: PipelineOptions,
  ): Promise<Gemma4State> {
    const entry = gemmaEntryName(admission.family);
    const plePlan = gemma4PlePlan(entry, options, assets.pleIndex);
    const decodeAhead = resolveGemma4DecodeAhead(entry, options, {
      ple: plePlan.residency !== "none",
    });
    const expertCache = resolveGemma4ExpertCache(
      entry,
      open(MODEL).graph,
      options,
      admitted.config.chunkLength,
    );
    const { wiring, tokenizer, ple } = buildGemma4Program(
      admitted,
      open,
      assets.tokenizer,
      // Under the cache the chunk rows are the settled ones (a chunk's picks fit in the slots).
      expertCache === undefined ? options : { ...options, chunkLength: expertCache.chunkLength },
      admission,
      plePlan,
    );
    // The speculative `k` is resolved **once, here**, so that the two consumers (generation and the
    // estimate) see the same value.
    const speculativeK =
      options.speculative === undefined
        ? undefined
        : assertSpeculative("gemma.Pipeline", options.speculative);
    const gpu =
      options.gpu ??
      (await acquireGpu({
        ...toAcquireGpuOptions(requested.gpuFeatures),
        subgroups:
          options.rmsNormReduce === "subgroup32" ||
          options.linearGemvReduce === "parallel-subgroup32",
      }));
    const ownsGpu = options.gpu === undefined;
    // The quant execution knobs (including the ③PV reduction form) pass only the composed values
    // (explicit > quant declaration); unspecified ones are not passed at all (copying defaults here
    // would make this family alone run with stale values when the runtime's default moves — the same
    // for the budget). The drafter Session gets the same knobs.
    const sessionOptions = {
      ...(options.fuseRmsNormAdd === undefined ? {} : { fuseRmsNormAdd: options.fuseRmsNormAdd }),
      ...(options.fuseLinearStaticQuantize === undefined
        ? {}
        : { fuseLinearStaticQuantize: options.fuseLinearStaticQuantize }),
      ...(options.fuseGeluMulStaticQuantize === undefined
        ? {}
        : { fuseGeluMulStaticQuantize: options.fuseGeluMulStaticQuantize }),
      ...(options.packedStaticQuantize === undefined
        ? {}
        : { packedStaticQuantize: options.packedStaticQuantize }),
      ...(options.integerGemv === undefined ? {} : { integerGemv: options.integerGemv }),
      ...(options.linearGemvPlanes === undefined
        ? {}
        : { linearGemvPlanes: options.linearGemvPlanes }),
      ...(options.rmsNormReduce === undefined ? {} : { rmsNormReduce: options.rmsNormReduce }),
      ...(options.fullStateStorage === undefined
        ? {}
        : { fullStateStorage: options.fullStateStorage }),
      submitPolicy: gemmaSubmitPolicy(options.submitPolicy),
      ...(options.stateAttentionReduce === undefined
        ? {}
        : { stateAttentionReduce: options.stateAttentionReduce }),
      ...(options.linearGemvReduce === undefined
        ? {}
        : { linearGemvReduce: options.linearGemvReduce }),
      ...(options.planBackingBudgetBytes === undefined
        ? {}
        : { planBackingBudgetBytes: options.planBackingBudgetBytes }),
      ...(options.attentionRowBlockMaxBytes === undefined
        ? {}
        : { attentionRowBlockMaxBytes: options.attentionRowBlockMaxBytes }),
      ...(options.linearGemvRowsThreadTarget === undefined
        ? {}
        : { linearGemvRowsThreadTarget: options.linearGemvRowsThreadTarget }),
    };
    // MUST: cannot be combined with GPU timing diagnostics (`beginBatch` cannot open on a device
    // with measurement enabled — the mechanism of enqueuing the gather in the same batch as the target
    // does not work at all).
    if (plePlan.residency === "gpu" && gpu.gpuTimingEnabled) {
      throw new Error(
        `${entry}: pleResidency: "gpu" cannot be used on a device with gpuTiming enabled` +
          ` (the on-GPU gather is queued into a batch, but a batch cannot be opened while timing — ADR 0021)`,
      );
    }
    // The supplier is **the same one** admission looked at (the earlier and later steps do not hold
    // different components).
    const model = open(MODEL);
    let session: Session | undefined;
    let pleGpu: Gemma4PleResident | undefined;
    let aheadIds: ResidentTensor | undefined;
    let visionSession: Session | undefined;
    try {
      // MUST: for a device acquired here, this is the only feature check (a shared GPU goes through
      // the same one at the admission step too). It is inside the try so that, on failure, the
      // device acquired internally is given back.
      assertGpuFeaturesGranted(requested.gpuFeatures, gpu, requested.where);
      // A shortfall of the binding limit fails **before uploading 1.5 GiB of weights** (the same
      // ordering as graph-first).
      pleGpu =
        plePlan.residency === "gpu"
          ? await createGemma4PleResident({
              gpu,
              index: plePlan.index,
              openBlock: (asset) => model.asset(asset),
              vocabSize: admitted.vocabSize,
              inputName: PER_LAYER_INPUTS,
              idsName: INPUT_IDS,
              // The physical row counts that can run (decode 1 + prefill buckets + chunk length) — the
              // upper bound of the resident output.
              rows: [1, ...wiring.chunkBuckets, wiring.chunkLength],
              entry,
              // The run-ahead's steps feed token ids that stay on the GPU (`decodeAhead`).
              residentIds: decodeAhead >= 2,
            })
          : undefined;
      aheadIds =
        decodeAhead >= 2 && pleGpu === undefined
          ? await gpu.createResident(4, "greedy-ahead-ids")
          : undefined;
      // The cache goes to the target alone (the drafter and the embedder are refused with it).
      const expertCacheOptions: SessionOptions["expertCache"] =
        expertCache === undefined
          ? undefined
          : {
              slots: expertCache.slots,
              ...(expertCache.hostBytes === undefined ? {} : { hostBytes: expertCache.hostBytes }),
            };
      session = await model.createSession(
        gpu,
        expertCacheOptions === undefined
          ? sessionOptions
          : { ...sessionOptions, expertCache: expertCacheOptions },
      );
      const cacheSession =
        expertCache === undefined
          ? undefined
          : createExpertCacheSession(
              session,
              expertCache.depth === undefined ? {} : { depth: expertCache.depth },
            );
      // Under GPU residency, runs that do not use greedy (prefill, decode with diagnostics) also go
      // through the wrapper — it is the only place to plug in `per_layer_inputs`. Whether greedy
      // itself is used is decided by the generation surface from the presence of `onRun`, so it is
      // independent of whether the wrapper exists.
      // A distribution with a suppressed set masks it inside the GPU selection (`Sampler.greedy`).
      // MUST: none under the expert cache — the GPU selection enqueues the target into a batch, which
      // a Session with the cache refuses (its runs go through the re-run face and the host sampler).
      const greedyOutput =
        cacheSession === undefined &&
        (pleGpu !== undefined ||
          decodeAhead >= 2 ||
          (!gpu.gpuTimingEnabled && options.onRunDiagnostics === undefined))
          ? createGemmaGreedyOutput(
              gpu,
              session,
              wiring.logits,
              wiring.vocabSize,
              pleGpu,
              decodeAhead,
              {
                suppressTokens: wiring.suppressTokens,
                ...(aheadIds === undefined
                  ? {}
                  : { aheadIds: { name: INPUT_IDS, resident: aheadIds } }),
              },
            )
          : undefined;
      // Stateless and on bf16 weights, which need no device feature (they are expanded inside the
      // kernel — ADR 0131), so of the Session options only the host policies apply to it.
      visionSession =
        admitted.vision === undefined
          ? undefined
          : await open(VISION).createSession(gpu, {
              submitPolicy: sessionOptions.submitPolicy,
              ...(options.planBackingBudgetBytes === undefined
                ? {}
                : { planBackingBudgetBytes: options.planBackingBudgetBytes }),
              ...(options.linearGemvRowsThreadTarget === undefined
                ? {}
                : { linearGemvRowsThreadTarget: options.linearGemvRowsThreadTarget }),
            });
      const drafter = await GemmaPipeline.#buildDrafter(
        admitted,
        open,
        session,
        gpu,
        sessionOptions,
      );
      return {
        entry,
        gpu,
        ownsGpu,
        session,
        ...(cacheSession === undefined ? {} : { expertCache: cacheSession }),
        ...(expertCacheOptions === undefined ? {} : { expertCacheOptions }),
        ...(sessionOptions.stateAttentionReduce === undefined
          ? {}
          : { stateAttentionReduce: sessionOptions.stateAttentionReduce }),
        ...(sessionOptions.fullStateStorage === undefined
          ? {}
          : { fullStateStorage: sessionOptions.fullStateStorage }),
        ...(greedyOutput === undefined ? {} : { greedyOutput }),
        graph: model.graph,
        wiring,
        program: generationProgramFace(wiring),
        ...(ple === undefined ? {} : { ple }),
        ...(pleGpu === undefined ? {} : { pleGpu }),
        ...(aheadIds === undefined ? {} : { aheadIds }),
        tokenizer,
        config: admitted.config,
        ...(drafter === undefined ? {} : { drafter }),
        ...(visionSession === undefined || admitted.vision === undefined
          ? {}
          : {
              vision: {
                session: visionSession,
                graph: open(VISION).graph,
                output: admitted.vision.output,
                symbol: admitted.vision.symbol,
                config: admitted.vision.config,
              },
            }),
        ...(speculativeK === undefined ? {} : { speculativeK }),
        // The gate's knobs pass straight through with the already checked (`assertSpeculative`) value.
        ...(options.speculative?.gate === undefined
          ? {}
          : { speculativeGate: options.speculative.gate }),
        ...(options.planBackingBudgetBytes === undefined
          ? {}
          : { planBackingBudgetBytes: options.planBackingBudgetBytes }),
        ...(options.attentionRowBlockMaxBytes === undefined
          ? {}
          : { attentionRowBlockMaxBytes: options.attentionRowBlockMaxBytes }),
        ...(options.onRunDiagnostics === undefined
          ? {}
          : { onRunDiagnostics: options.onRunDiagnostics }),
      };
    } catch (error) {
      // If construction fails nobody else can release these, so they are given back here. The order
      // is borrower (the drafter was not set up = the borrow count is already back) → lender → PLE's
      // GPU residency → the GPU acquired internally.
      await disposeSteps([
        () => visionSession?.dispose(),
        () => {
          if (session !== undefined) return session.dispose();
        },
        () => pleGpu?.dispose(),
        () => aheadIds?.dispose(),
        () => {
          if (ownsGpu) gpu.destroy();
        },
      ]).catch(() => {});
      throw error;
    }
  }

  /**
   * Sets up one drafter Session (only when speculation is requested).
   *
   * It borrows the embedding table the target Session has already put on the GPU; the mapping from
   * the borrower's initializer name to the lender's initializer name was settled by admission
   * (`./speculative.ts`). Not a single byte is copied.
   */
  static async #buildDrafter(
    admitted: Gemma4Admission,
    open: ComponentOpener,
    target: Session,
    gpu: GpuContext,
    sessionOptions: SessionOptions,
  ): Promise<Gemma4Drafter | undefined> {
    const { drafter } = admitted;
    if (drafter === undefined) return undefined;
    const component = open(DRAFTER);
    let sharedWeights: Record<string, SharedWeight> = {};
    for (const borrower of Object.keys(drafter.admission.sharedWeights)) {
      sharedWeights = {
        ...sharedWeights,
        [borrower]: target.exportWeight(drafter.admission.sharedWeights[borrower]),
      };
    }
    return {
      session: await component.createSession(gpu, {
        ...sessionOptions,
        sharedWeights,
      }),
      // For the estimate only (`estimateSessionMemory` derives the drafter's resident weights from it).
      graph: component.graph,
      outputs: drafter.admission.outputs,
      rope: admitted.config.rope,
      hiddenSize: drafter.admission.hiddenSize,
    };
  }

  /**
   * Runs one conversation turn and streams **committed text pieces** (the streaming decode of ADR
   * 0084 decision 4).
   *
   * Rendering and encoding the conversation is `chatPrompt` (`chatImagePrompt` on a pipeline opened
   * with {@link PipelineOptions.vision}, where a user message can hold images; out-of-scope input
   * fails loudly here, **synchronously**), and generation is one `GenerationSequence`; whether
   * drained, `break`-ed or interrupted, the sequence is returned. An image is read at issue and
   * embedded when the stream starts draining. **One turn = one sequence**, so past turns do not
   * remain — to run a multi-turn conversation yourself, use {@link Pipeline.sequence}.
   *
   * If incremental display is not needed, {@link ChatStream.text} gives one string (mixing it
   * with iteration throws synchronously — a stream is consumed in only one way).
   *
   * Stop conditions have two layers and can be added per request (the distribution's EOS set always
   * applies): {@link ChatOptions.stopTokens} is checked by the sequence layer (token ids), and
   * {@link ChatOptions.stopStrings} by this layer (the decoded text).
   *
   * A pipeline with a drafter loaded **speculates by default** (it can be turned off per turn with
   * {@link ChatOptions.speculative}). A speculating turn carries the accounting in the
   * `speculation` of `done`. It speculates even with the distribution's recommended sampler
   * (temperature 1), and the text is the same sequence as without speculation. The default comes
   * with the self-tuning gate, so in a context where speculation loses it drops to the decode
   * shape partway (`speculation.plainSteps`).
   *
   * When called concurrently, calls **wait and run in order** (one Session is not pushed by two
   * conversations at once).
   */
  chat(messages: readonly ChatMessage[], options: ChatOptions): ChatStream {
    if (this.#disposal !== undefined) {
      throw new Error(`${this.#state.entry}: cannot generate after dispose`);
    }
    // The accepted set fails synchronously (before entering the GPU or the queue).
    const visionConfig = this.#state.vision?.config;
    const rendered =
      visionConfig === undefined
        ? { prompt: chatPrompt(this.#state.tokenizer, messages), images: [] }
        : chatImagePrompt(this.#state.tokenizer, messages, visionConfig);
    const { prompt } = rendered;
    // The images are read here too, for the rule of the MUST below: the rows are built at issue, so
    // nothing written into the pixels afterwards reaches this turn.
    const imageBlocks = rendered.images.map(({ image, start }) => ({
      start,
      rows: imageRows(image, visionConfig?.softTokenBudget),
    }));
    const chosen = options.sampler ?? this.#state.config.sampler;
    // MUST: the request is **copied at issue** (ADR 0083 addendum 2026-09-02). The body (an async
    // generator) does not run until the first `next()`, so without reading `options` through here,
    // `maxNewTokens` / `signal` would be "the values at the time draining started" — an option
    // rewritten between issue and consumption would silently take effect.
    // The arrays and the sampler options are copied **here** as well. The receivers' copies
    // (`GenerationSequence.generate` / `createSampler`) happen in the body of the async generator =
    // the first `next()`, so relying on them alone would silently apply a stop set or sampling
    // options rewritten in the window "from issue until draining starts". The copy point is the one
    // {@link snapshotSpec} in `sampler.ts`, shared (the same as `ChatSession.send`).
    const sampler = chosen === undefined ? undefined : snapshotSpec(chosen);
    const request: GenerationRequest = {
      prompt,
      ...(options.maxNewTokens === undefined ? {} : { maxNewTokens: options.maxNewTokens }),
      ...(options.stopTokens === undefined ? {} : { stopTokens: [...options.stopTokens] }),
      ...(sampler === undefined ? {} : { sampler }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    };
    // MUST: the value ranges of the accepted set are checked **here** as well (the source of truth of
    // the check is the one in `sequence.ts` — the formulas are not copied). `generate` is called in
    // the body of the async generator below, so without this a request such as `maxNewTokens: 0`
    // would "not fail until draining starts" = the diagnostic would appear only far from the issuer.
    assertGenerationRequestValues(this.#state.wiring.vocabSize, request);
    // The stop-string state machine is also built here (the option check and copy happen inside it =
    // the accepted set fails synchronously). With no stop strings it passes everything through, so
    // the path is not split in two.
    const stopStrings = createStopStringFilter(options.stopStrings ?? []);
    const capacityOption = options.capacity;
    const fitCapacity = (): Promise<number> => this.fitCapacity();
    const onPrefill = options.onPrefill;
    const onToken = options.onToken;
    // The speculation DI and the observation hook are also decided **at issue** (the body does not run
    // until the first `next()`).
    const speculative = speculativeSetup(this.#state, options.speculative);
    const onRun = runDiagnosticsHook(this.#state);

    let settle!: (stop: ChatStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<ChatStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // A secondary notification path, so not an unhandled rejection even when nobody reads it.
    done.catch(() => {});

    const state = this.#state;
    const acquire = this.#acquire.bind(this);
    const disposed = (): boolean => this.#disposal !== undefined;
    const chunks = async function* (): AsyncGenerator<string, void, undefined> {
      let release: (() => void) | undefined;
      let sequence: GenerationSequence | undefined;
      let stream: GenerationStream | undefined;
      let failure: { readonly error: unknown } | undefined;
      /**
       * The matched stop string (this layer's stop reason — once set, consuming the sequence stops
       * there too).
       */
      let matched: string | undefined;
      try {
        // MUST: check again at the top of the body. The body of an async generator does not run until
        // the first `next()`, so the check at issue alone misses "issue → dispose → start draining".
        // Even past that the runtime would not accept it (a disposed Session), but it would fail with
        // a **runtime message** far from the real cause, so this aligns it to **the same pipeline
        // message as at issue**.
        if (disposed()) {
          throw new Error(`${state.entry}: cannot generate after dispose`);
        }
        release = await acquire();
        const capacity = capacityOption === "fit" ? await fitCapacity() : capacityOption;
        const softBlocks: GenerationSoftBlock[] = [];
        for (const { start, rows } of imageBlocks) {
          // The embedder's stage boundary: a long list of images stops between two of them. An abort
          // here is left to the sequence, which settles the turn as aborted before its first run, as it
          // does for a turn without images.
          if (request.signal?.aborted === true) break;
          softBlocks.push({ start, rows: (await embedRows(state, rows)).rows });
        }
        sequence = await createGenerationSequence({
          session: state.expertCache ?? state.greedyOutput?.session ?? state.session,
          ...(state.greedyOutput === undefined
            ? {}
            : { greedy: state.greedyOutput.greedy, greedyMasksSuppressed: true }),
          ...(state.greedyOutput?.ahead === undefined
            ? {}
            : { greedyAhead: state.greedyOutput.ahead }),
          ...(state.expertCache?.speculation === undefined
            ? {}
            : { selfSpeculation: state.expertCache.speculation }),
          program: state.wiring,
          ...(capacity === undefined ? {} : { capacity }),
          ...(speculative === undefined ? {} : { speculative }),
          ...(onRun === undefined ? {} : { onRun }),
        });
        stream = sequence.generate(
          softBlocks.length === 0 || request.signal?.aborted === true
            ? request
            : { ...request, softBlocks },
        );
        matched = yield* decodeChatChunks(
          stream,
          state.tokenizer.createDetokenizer(),
          stopStrings,
          onPrefill,
          onToken,
        );
      } catch (error) {
        failure = { error };
        // MUST: throw as is, unwrapped (ADR 0083 decision 5 — the consumer can identify its own
        // interruption with `error === signal.reason`).
        throw error;
      } finally {
        await completeChatTurn({
          where: "gemma.Pipeline.chat",
          stream,
          matched,
          failure,
          settle,
          fail,
          release,
          cleanup: async () => {
            await sequence?.dispose();
          },
        });
      }
    };

    return chatStreamOf(
      closeableGenerator(chunks(), (failure) => {
        if (failure !== undefined) fail(failure.error);
        else settle({ reason: "closed", tokens: 0 });
      }),
      done,
    );
  }

  /**
   * One image → its vectors: the rows {@link GenerationRequest.softBlocks} takes, for the prompt
   * `chatImagePrompt` / `chatImageTurn` renders. {@link Pipeline.chat} and `ChatSession` call it for
   * themselves, so only a caller of {@link Pipeline.sequence} needs it.
   *
   * The pixels are read at the call (preprocessed in its synchronous part). The run itself waits for
   * the pipeline's serialization chain, as {@link Pipeline.chat} does, so an embedding issued while a
   * `chat` stream is open starts when that stream closes.
   *
   * MUST: the pipeline was opened with {@link PipelineOptions.vision} (`ModelInputError` otherwise).
   */
  async embedImage(image: Rgb8Image): Promise<ImageEmbedding> {
    const state = this.#state;
    if (this.#disposal !== undefined) {
      throw new Error(`${state.entry}: cannot embed an image after dispose`);
    }
    if (state.vision === undefined) {
      throw new ModelInputError(
        `${state.entry}: embedImage needs a pipeline opened with vision: true`,
      );
    }
    assertRgb8(image);
    const rows = imageRows(image, state.vision.config.softTokenBudget);
    const release = await this.#acquire();
    try {
      if (this.#disposal !== undefined) {
        throw new Error(`${state.entry}: cannot embed an image after dispose`);
      }
      return await embedRows(state, rows);
    } finally {
      release();
    }
  }

  /**
   * The low-level surface — run a multi-turn conversation yourself (the `GenerationSequence` of ADR
   * 0083 decisions 1–4).
   *
   * `prompt` is a token id sequence, so the caller renders the conversation (the `./gemma` subpath) —
   * the first turn with {@link chatPrompt} (the whole, including `<bos>`), the second and later
   * with `chatTurn` (**only that turn's delta**; the `<turn|>` that closes the previous turn is
   * prepended by `GenerationSequence`'s `pendingToken`, so it is not included — ADR 0083 decision 4).
   * **The caller calls `dispose()` on the returned object** — returning it when the conversation ends
   * is right, and if it is missed {@link Pipeline.dispose} collects it (so that the Session is
   * not closed with a live context).
   *
   * MUST: no serialization with {@link Pipeline.chat} (another conversation is another context,
   * so the runtime accepts it). Running them at the same time keeps two KVs resident.
   *
   * `capacity` is the capacity this conversation allocates (default: the distribution's default). The
   * KV's physical allocation happens here, so this is the one place to decide "do not take a large
   * capacity for a short conversation / take only what a long conversation needs".
   *
   * A pipeline with a drafter loaded **speculates by default** (it can be turned off per conversation
   * with {@link SequenceOptions.speculative}). The borrower context is bound to this sequence's
   * lifetime, and `dispose()` closes it before the lender. The self-tuning gate's state also lives
   * as long as the sequence (it keeps measuring across turns).
   */
  async sequence(options: SequenceOptions = {}): Promise<GenerationSequence> {
    if (this.#disposal !== undefined) {
      throw new Error(`${this.#state.entry}: cannot create a sequence after dispose`);
    }
    const state = this.#state;
    const speculative = speculativeSetup(state, options.speculative);
    const onRun = runDiagnosticsHook(state);
    // Tracked from the call on: a dispose issued while this allocates waits for it (#inFlight).
    return await this.#track(
      (async (): Promise<GenerationSequence> => {
        const capacity = options.capacity === "fit" ? await this.fitCapacity() : options.capacity;
        const inner = await createGenerationSequence({
          session: state.expertCache ?? state.greedyOutput?.session ?? state.session,
          ...(state.greedyOutput === undefined
            ? {}
            : { greedy: state.greedyOutput.greedy, greedyMasksSuppressed: true }),
          ...(state.greedyOutput?.ahead === undefined
            ? {}
            : { greedyAhead: state.greedyOutput.ahead }),
          ...(state.expertCache?.speculation === undefined
            ? {}
            : { selfSpeculation: state.expertCache.speculation }),
          program: state.wiring,
          ...(capacity === undefined ? {} : { capacity }),
          ...(speculative === undefined ? {} : { speculative }),
          ...(onRun === undefined ? {} : { onRun }),
        });
        return await this.#hand(inner);
      })(),
    );
  }

  /**
   * Hand-outs still in flight (a `sequence()` allocating, a branch being copied, or the
   * `inner.dispose()` that {@link Pipeline.#hand} runs when disposal started meanwhile).
   *
   * MUST: `dispose` waits for every one of these before closing the Sessions. Such an inner sequence
   * borrows the target Session (and the drafter's) but is not in `#handed` yet, so without the wait
   * its dispose raced the steps of the dispose chain (drafter Session → target Session) and was
   * queued first only because of how many awaits each side happened to take — reversed, the target
   * Session's dispose fails loudly ("still lent out") and the borrower is closed after its lender.
   * Entries are added synchronously at the call (`sequence()` / `fork()` check `#disposal` first, so
   * none is added after the chain body starts).
   */
  readonly #inFlight = new Set<Promise<unknown>>();

  /** Tracks one hand-out in {@link Pipeline.#inFlight} until it settles. */
  #track<T>(handing: Promise<T>): Promise<T> {
    this.#inFlight.add(handing);
    const untrack = (): void => {
      this.#inFlight.delete(handing);
    };
    handing.then(untrack, untrack);
    return handing;
  }

  /**
   * Hands a sequence out in a thin tracking wrapper (the one path shared by
   * {@link Pipeline.sequence} and its branches (`GenerationSequence.fork`)).
   *
   * A thin wrapper rather than the object itself is handed out so that `GenerationSequence` gets no
   * field that would let it know about the pipeline (the generation surface stays pipeline-independent
   * to the end — ADR 0083). The wrapper holds **only this tracking** (the observation callback went
   * down into the sequence as the `onRun` hook, so `generate` can pass straight through). Branches are
   * handed out in the same wrapper, so `dispose` collects them too.
   *
   * MUST: check once more after `await` (an object disposed in the middle of allocation (or of the
   * copy, for a branch) is **closed by neither path**, because the body of the `dispose` chain has
   * already walked `#handed` — this re-check is the only thing that closes the window. The same shape
   * as `#adopt` in the runtime's `executor.ts`).
   */
  async #hand(inner: GenerationSequence): Promise<GenerationSequence> {
    const state = this.#state;
    if (this.#disposal !== undefined) {
      await inner.dispose();
      throw new Error(`${state.entry}: cannot create a sequence after dispose`);
    }
    // A sequence returned properly is removed from tracking (otherwise, with a multi-turn UI that
    // creates and closes one per conversation, the Set grows monotonically and `dispose` awaits every
    // already-destroyed object once more).
    const handed: GenerationSequence = {
      capacity: inner.capacity,
      // A derived value, so the wrapper also passes it through with a getter (copying the value would
      // freeze it at "the value at hand-out time").
      get used(): number {
        return inner.used;
      },
      generate: (request): GenerationStream => inner.generate(request),
      fork: (): Promise<GenerationSequence> => {
        // MUST: not async — the sequence side's contract is to throw the disposed rejection
        // **synchronously**, and wrapping it in async would turn it into a reject. After the
        // pipeline's dispose it likewise fails synchronously.
        if (this.#disposal !== undefined) {
          throw new Error(`${state.entry}: cannot create a sequence after dispose`);
        }
        return this.#track(inner.fork().then((forked) => this.#hand(forked)));
      },
      dispose: async (): Promise<void> => {
        await inner.dispose();
        // A failed dispose is not removed (the context has not been returned, so it stays on the side
        // `dispose` collects).
        this.#handed.delete(handed);
      },
    };
    this.#handed.add(handed);
    return handed;
  }

  /**
   * Estimates the GPU memory needed to run one conversation on this pipeline (the estimator of ADR
   * 0070 decision 5).
   *
   * A surface for **after loading and before creating a sequence**. Capacity and chunk length are
   * runtime knobs, so the requirement is settled only "after the caller has decided the shape of the
   * generation" (the same reading as the deferral of ADR 0089 addendum 2026-09-02 — not wired into
   * the load surface).
   *
   * MUST: it returns **only the demand-side totals by category**, and neither compares against free
   * memory nor decides feasibility (same, decision 5). The final check remains the out-of-memory
   * errorScope; this estimate is an advance diagnostic.
   *
   * NOTE: each shape of {@link PipelineOptions.chunkBuckets} enters the max of
   * `peakAccountedBytes` but is not listed in `scenarios` (ADR 0095 addendum 2026-10-10). The prefill
   * shape of the largest `M` is not used as the representative because the row-block split of
   * attention creates, at block-count boundaries, "a shape with fewer rows having larger transients",
   * so the intermediates are not monotonic in the physical row count `M`. When the `chunkLength` to
   * estimate differs from the pipeline's value, only this pipeline's buckets below that
   * `chunkLength` are passed (the accepted set of buckets is below `chunkLength` — the same rule as
   * trimming the default ladder). The amount by which bucket-shape backings **stay resident at the
   * same time** as other shapes is bounded from above by
   * {@link PipelineOptions.planBackingBudgetBytes} as one term of `peakAccountedBytes` (ADR
   * 0095 decision 4).
   *
   * The auxiliary Session and resident inputs / outputs of small-output decode are counted separately
   * in `auxiliaryBytes` and also added to the peak. For a pipeline whose small output was disabled by
   * diagnostics this field is omitted (0).
   *
   * NOTE: `AdmissionReport` is a runtime type, and `@karume/models` does not re-export it (the thin
   * surface of ADR 0008 — consumers that read the estimate use the runtime type as is).
   *
   * ## Speculation (a pipeline given {@link PipelineOptions.speculative})
   *
   * Three things are added (ADR 0096 stage 3 §3.3):
   *
   * 1. **The verify scenario** — the shape that runs `k+1` rows in one run (listed in `scenarios` as
   *    `"verify"`). The row-selection symbol `R` is also bound to `k+1` (readback needs as many rows
   *    as were selected).
   * 2. **The drafter's resident weights** — added to each field of `resident.weights`. The embedding
   *    table borrowed from the lender is excluded by the runtime's residency planner as a `shared`
   *    entry, so it is not counted twice.
   * 3. **The borrower context's state** — borrowed slots allocate not a single byte, so the only
   *    increase is the 8-byte logical-length uniform.
   *
   * One drafter run (inputs / outputs and intermediates) is put on the `unaccounted` side — a draft
   * run never runs at the same time as a verify run, and the requirement of that shape is a small
   * constant that depends on neither `k` nor `capacity`.
   *
   * With {@link PipelineOptions.expertCache} `depth`, item 1 alone is added, with `depth + 1` rows
   * (its drafts run in the decode shape on the target itself).
   */
  estimateSessionMemory(options: EstimateOptions = {}): AdmissionReport {
    const { wiring, graph, gpu, drafter, vision, entry, expertCacheOptions } = this.#state;
    const capacity = options.capacity ?? wiring.capacity;
    const chunkLength = assertChunkLength(
      options.chunkLength ?? wiring.chunkLength,
      this.#state.config,
      entry,
    );
    // MUST: the integer condition and the ordering are reported as **separate diagnostics** (the same
    // three as `createGenerationSequence` / `ChatSession`). Folded into one, `capacity: NaN` and
    // `1.5` would both be diagnosed as "below", and the same option would look as if it failed for a
    // different reason per entry point.
    if (!Number.isSafeInteger(capacity) || capacity < 1) {
      throw new ModelInputError(`${entry}: capacity ${capacity} is not a positive integer`);
    }
    if (capacity < chunkLength) {
      throw new ModelInputError(
        `${entry}: capacity ${capacity} is below chunkLength ${chunkLength} (not even one chunk fits)`,
      );
    }
    if (capacity > wiring.maxPosition) {
      throw new ModelInputError(
        `${entry}: capacity ${capacity} exceeds maxPosition ${wiring.maxPosition}`,
      );
    }
    // MUST: pass the same budgets as were passed to the Session (if only one side fell back to the
    // default, the reported peak would name a different budget from the actual held set, or count a
    // different row-block split). Unspecified is not passed at all and left to the runtime default.
    const budget = {
      ...(this.#state.planBackingBudgetBytes === undefined
        ? {}
        : { planBackingBudgetBytes: this.#state.planBackingBudgetBytes }),
      ...(this.#state.attentionRowBlockMaxBytes === undefined
        ? {}
        : { attentionRowBlockMaxBytes: this.#state.attentionRowBlockMaxBytes }),
    };
    // MUST: pass it (the in-node transients of states-form attention have a row-block count decided by
    // this limit alone, so omitting it makes the estimator fail loudly — it is not filled with a
    // default).
    const maxStorageBufferBindingSize = gpu.limits.maxStorageBufferBindingSize;
    // Both limits on a single state slot are checked (the same check as the real construction in
    // `GenerationContext.create` — passing only one would let through the estimate a capacity that
    // "shows a number at startup and fails on the first turn").
    const maxBufferSize = gpu.limits.maxBufferSize;
    const generations = gemma4EstimateGenerations(wiring, {
      capacity,
      chunkLength,
      ...(drafter === undefined
        ? {}
        : { verifyQueryLength: (this.#state.speculativeK ?? GEMMA4_DRAFT_STEPS) + 1 }),
      // Self-speculation over the expert cache verifies `depth + 1` rows the same way (a drafter is
      // refused with the cache, so at most one of the two is present).
      ...(this.#state.expertCache?.speculation === undefined
        ? {}
        : { verifyQueryLength: this.#state.expertCache.speculation.depth + 1 }),
    });
    const baseTarget = estimateGraphMemory(graph, planWeightResidency(graph), {
      // In a run the row-count symbol R is bound by the element count of `last_row` (from the input
      // shape). The estimate has no inputs, so R = 1 (the ordinary prefill / decode shape) is given
      // explicitly.
      bindings: { [wiring.rowSymbol]: 1 },
      generation: generations.target,
      maxStorageBufferBindingSize,
      maxBufferSize,
      ...(this.#state.stateAttentionReduce === undefined
        ? {}
        : { stateAttentionReduce: this.#state.stateAttentionReduce }),
      ...(this.#state.fullStateStorage === undefined
        ? {}
        : { fullStateStorage: this.#state.fullStateStorage }),
      // The expert pools and slot tables the Session holds: the very option it was built with (its
      // host tier is named among the uncounted).
      ...(expertCacheOptions === undefined ? {} : { expertCache: expertCacheOptions }),
      ...budget,
    });
    // The auxiliary Session of small-output decode, and the share of the GPU-resident PLE option (ADR
    // 0085 addendum). A pipeline that has neither omits the field altogether (it does not claim 0).
    const auxiliary =
      (this.#state.greedyOutput?.extraBytes ?? 0) + (this.#state.pleGpu?.extraBytes ?? 0);
    const auxiliaryBytes = auxiliary === 0 ? undefined : auxiliary;
    const target = {
      ...baseTarget,
      ...(auxiliaryBytes === undefined ? {} : { auxiliaryBytes }),
      peakAccountedBytes: baseTarget.peakAccountedBytes + (auxiliaryBytes ?? 0),
    };
    if (drafter === undefined && vision === undefined) return target;
    // The borrower's share (resident weights and the context's state) is derived by **running the
    // drafter graph through the same estimator** (the formula is not rebuilt in this layer — excluding
    // borrowed slots and shared initializers is exactly the branching of the runtime's residency
    // planner and `GenerationContext.create`).
    const borrower =
      drafter === undefined
        ? undefined
        : estimateGraphMemory(drafter.graph, planWeightResidency(drafter.graph), {
            generation: generations.borrower,
            maxStorageBufferBindingSize,
            maxBufferSize,
            ...budget,
          });
    // The embedder goes through the same estimator, at the most rows one image can have (its row
    // symbol is bound by the inputs, which an estimate does not have).
    const embedder =
      vision === undefined
        ? undefined
        : estimateGraphMemory(vision.graph, planWeightResidency(vision.graph), {
            bindings: { [vision.symbol]: vision.config.softTokenBudget },
            maxStorageBufferBindingSize,
            maxBufferSize,
            ...budget,
          });
    const companions = [borrower, embedder].filter((report) => report !== undefined);
    const sumOf = (read: (report: AdmissionReport) => number): number =>
      read(target) + companions.reduce((total, report) => total + read(report), 0);
    const weights = {
      compressedBytes: sumOf((report) => report.resident.weights.compressedBytes),
      uncompressedBytes: sumOf((report) => report.resident.weights.uncompressedBytes),
      expandedBytes: sumOf((report) => report.resident.weights.expandedBytes),
      totalBytes: sumOf((report) => report.resident.weights.totalBytes),
    };
    const borrowerBytes =
      borrower === undefined
        ? 0
        : borrower.resident.weights.totalBytes + borrower.resident.stateBytes;
    return {
      resident: {
        weights,
        stateBytes: sumOf((report) => report.resident.stateBytes),
      },
      // The per-shape requirement stays the lender's three shapes (prefill / decode / verify) — a
      // drafter run never runs at the same time as a lender run (the borrower's run takes the lender's
      // run lease).
      scenarios: target.scenarios,
      planBackingBudgetBytes: target.planBackingBudgetBytes,
      ...(auxiliaryBytes === undefined ? {} : { auxiliaryBytes }),
      // Residency is held by the Sessions at once, so it is the sum. The upper limit of the held set
      // (budget vs the largest scenario) applies on the lender's side as is. The embedder's whole
      // peak counts, not only its weights: its Session keeps the slot backing of its last image.
      peakAccountedBytes:
        target.peakAccountedBytes + borrowerBytes + (embedder?.peakAccountedBytes ?? 0),
      unaccounted: [
        ...target.unaccounted,
        ...(borrower === undefined
          ? []
          : [
              "The inputs, outputs and intermediates of one drafter run (the borrowing Session's slot backing — a draft run" +
                " never runs at the same time as a verify run, and the requirement depends on neither k nor capacity)",
            ]),
      ],
    };
  }

  /**
   * The largest capacity this device can hold right now: the position limit, then the powers of two
   * below it, down to the distribution's default, taking the first whose estimated need plus a
   * margin passes a trial allocation (`fitsHeadroom` — ADR 0070 decision 5).
   *
   * The value is a fact of this moment, not a reservation: other processes move it, and it differs
   * between machines. It does not move the generated tokens. A larger capacity costs VRAM from the
   * first token, so take it only for a conversation that needs it. Fails loudly when not even the
   * default fits. For a `ChatSession`, pass the result as its `capacity`.
   */
  async fitCapacity(): Promise<number> {
    if (this.#disposal !== undefined) {
      throw new Error(`${this.#state.entry}: cannot fit a capacity after dispose`);
    }
    const { gpu, wiring, entry } = this.#state;
    // The weights are on the GPU already; the rest of the peak is what a conversation adds.
    const needBytes = (capacity: number): number | undefined => {
      try {
        const report = this.estimateSessionMemory({ capacity });
        return report.peakAccountedBytes - report.resident.weights.totalBytes;
      } catch (cause) {
        // A slot over a buffer limit: this candidate cannot exist on this device.
        if (cause instanceof ExecutionError) return undefined;
        throw cause;
      }
    };
    await settleReleasedMemory(gpu);
    const picked = await pickFitCapacity(
      fitCapacityCandidates(wiring.capacity, wiring.maxPosition),
      needBytes,
      (bytes) => fitsHeadroom(gpu, bytes),
    );
    if (picked === undefined) {
      throw new Error(
        `${entry}: capacity "fit" found nothing this device can allocate now, not even the default ${wiring.capacity}`,
      );
    }
    return picked;
  }

  /**
   * The counters of the expert cache over this pipeline's lifetime (`undefined` without
   * {@link PipelineOptions.expertCache}): the runtime's runs, missed runs, loads and the bytes and
   * time they took, with the slots, the runs committed and the re-runs a miss cost (ADR 0146).
   */
  expertCacheReport(): ExpertCacheReport | undefined {
    return this.#state.expertCache?.report();
  }

  /**
   * The read-out of the static wiring (read the chunk length, position limit, capacity, vocabulary
   * size and stop set when running with `sequence()`). It is a **frozen** value, and `stopTokens` is a
   * frozen copy too.
   *
   * Only numbers come out; the names of graph inputs / outputs and `derivedInputs` do not (the doc of
   * `GenerationProgram` — the Session they are wired to is not on the public surface, so reading them
   * would be useless, and a writable entry point would break the type's meaning of "verified").
   */
  get program(): GenerationProgram {
    return this.#state.program;
  }

  /**
   * The tokenizer built from the assets (needed to render and decode chat — a member of the same
   * digest set).
   */
  get tokenizer(): Tokenizer {
    return this.#state.tokenizer;
  }

  /**
   * The image input's declaration when this pipeline was opened with
   * {@link PipelineOptions.vision} (`undefined` otherwise, whatever the distribution declares):
   * what `chatImagePrompt` renders with, and how a `ChatSession` knows it can take image parts.
   */
  get vision(): PipelineVisionConfig | undefined {
    return this.#state.vision?.config;
  }

  /**
   * The **default** sampler the distribution declared (ADR 0083 decision 7 — `undefined` = greedy
   * when there is no declaration).
   *
   * MUST: the name does not mean "the sampler this generation is using now".
   * {@link Pipeline.chat} uses it only when the request omits `sampler`; if the request passes
   * one, that wins. When running the low-level surface ({@link Pipeline.sequence}) yourself,
   * **pass it yourself** to `generate`'s `sampler` — `GenerationSequence` does not know the
   * distribution, so without it it runs with the lower layer's default (temperature 0) (the parity
   * check takes that path).
   */
  get defaultSampler(): DefaultSampler | undefined {
    return this.#state.config.sampler;
  }

  /**
   * Releases. Closes the handed-out sequences first, then **the drafter Session (if any) → the target
   * Session**, destroys **only the GPU acquired internally**, and finally returns PLE's host cache.
   *
   * MUST: destroy after in-flight generation has completed (flush-before-destroy) — putting the
   * destruction on the chain too decides the order of waiting and destroying in one place. Later
   * calls return the same completion.
   *
   * MUST: PLE is released as well. Unlike GPU residency it is **host RAM**
   * ({@link PipelineOptions.maxResidentPleBytes} worth of blocks = by default two of the
   * largest blocks, plus the row cache of at most 256 rows), so without a release point "keeping one
   * disposed handle" alone would keep that RAM for the life of the process.
   *
   * MUST: even when one step throws, **proceed to the remaining steps**. `#disposal` keeps one promise
   * including failures (a second call returns the same rejection = there is no retry), so stopping at
   * the first failure would mean the Session, the GPU and PLE's host RAM are **never** released. One
   * failure is carried as is, two or more in an `AggregateError` (which step failed is not lost).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#chain(async () => {
      // MUST: before walking #handed — a hand-out in flight is in neither set yet, and its inner
      // sequence borrows the Sessions closed below (#inFlight). Each one either joins #handed or
      // disposes its inner itself (#hand's re-check), so after this wait #handed is complete.
      await Promise.allSettled([...this.#inFlight]);
      const handed = [...this.#handed];
      this.#handed.clear();
      await disposeSteps([
        ...handed.map((sequence) => () => sequence.dispose()),
        // MUST: close the borrower (the drafter Session) before the lender — the runtime refuses, by
        // failing loudly, the lender's `dispose()` while a borrower is alive (the borrower's bind
        // groups hold the lender's weight buffers). In the reverse order this one step always fails;
        // the remaining steps still go through, but every report would say "still lent out".
        () => this.#state.greedyOutput?.dispose(),
        () => this.#state.drafter?.session.dispose(),
        () => this.#state.vision?.session.dispose(),
        () => this.#state.session.dispose(),
        // MUST: after the target Session (released once the baked references of the resident input
        // have been returned).
        () => this.#state.pleGpu?.dispose(),
        () => this.#state.aheadIds?.dispose(),
        () => {
          if (this.#state.ownsGpu) this.#state.gpu.destroy();
        },
        // Ordered after the GPU (running generation has already been closed, so nothing draws on it
        // any more).
        () => this.#state.ple?.dispose(),
      ]);
    });
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  /**
   * Takes a place on the serialization chain, and gives it up with the returned function.
   *
   * The place is taken **when the body starts running** (the body of an async generator does not run
   * until the first `next()`). Taking it at issue would let a stream discarded without draining hold
   * the chain forever, dragging `dispose` along.
   */
  #acquire(): Promise<() => void> {
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    return new Promise<() => void>((admitted) => {
      void this.#chain(() => {
        admitted(release);
        return held;
      });
    });
  }
}

/** Regular Gemma 4. The entry point that keeps the existing fetch, numerics and MTP contracts. */
export class Pipeline extends GemmaPipeline {
  private constructor(state: Gemma4State) {
    super(state);
  }
  /**
   * Fetches from the distribution and builds (`loadManifest` → `resolveSelection` → fetch **only each
   * component's descriptor (part 0)** for the family admission → warm the weight parts → fetch the
   * tokenizer → build). Blocks are read in part order at the moment the Session is built, and for PLE
   * **only the touched blocks** are range-read from the persistent cache (ADR 0109 / 0085 decision 3).
   *
   * To replace a component with the same role from another repository, use
   * {@link FromPretrainedComponentOptions.components} (the roles are `model`, `drafter` and
   * `vision`).
   *
   * **`ref` is required** (there is no default source — the MUST of `src/hub/repo-ref.ts`). The source
   * the package release verified is `"gemma4"` (the regular line) or `"gemma4-qat"` (the q4_0 QAT
   * line) of {@link SOURCES} (`./config.ts`) — to pin reproducibility yourself, the caller gives
   * `{ repo, revision }` explicitly. A string `ref` reads as
   * `{ repo }` (= follows `main`).
   *
   * A distribution in hand is passed as a **source handle** (`localDirectory`, `denoDirectory` of
   * `@karume/hub/deno`, or `nodeDirectory` of `@karume/hub/node`). It does not go through the check of
   * HF's `owner/name` spelling, nor through the network or CacheStorage (the HTTP-only knobs of
   * {@link FromPretrainedOptions} have no effect).
   */
  static async fromPretrained(
    ref: string | HubRepoRef | DistributionSource,
    options: FromPretrainedOptions = {},
  ): Promise<Pipeline> {
    return new Pipeline(await super.loadPretrained(GEMMA4_PIPELINE_NAME, ref, options));
  }
  /**
   * Builds from fetched assets. Interpreting the assets, matching them against the graph and
   * cross-checking the id spaces are all done here, and it returns with **one Session of the product
   * graph set up**.
   *
   * `model` takes the container's (`krm`) **parts** (in index order from part 0, including zero-length
   * parts), and opens them as parts without concatenating the bytes (the receiving implementation is
   * {@link assetComponentOpener} in `src/hub/components.ts`, shared by all families). PLE and its index
   * are both inside the container, so the assets passed to this surface are only the parts and the
   * tokenizer (ADR 0109 decision 4).
   *
   * MUST: `config` goes through **the same check** as {@link fromPretrained}
   * ({@link parsePipelineConfig}). TS types check neither unknown keys nor value ranges, so
   * without the check a declaration such as `temperature: -1` would first fail at the first `chat`
   * after reading 3.7 GiB. The declaration is checked **before opening a single one of the byte
   * arrays**.
   */
  static async fromAssets(input: Assets, options: PipelineOptions = {}): Promise<Pipeline> {
    return new Pipeline(await super.loadAssets(GEMMA4_PIPELINE_NAME, input, options));
  }
}

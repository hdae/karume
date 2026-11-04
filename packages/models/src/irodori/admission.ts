/**
 * irodori's **admission** — the checks of whether the distribution's declarations and the product
 * graphs can run on this implementation, and the material those checks settle. The fetch keys and
 * the asset readers live here too (they are what the checks read first).
 *
 * The description of the 10 steps (what runs in which order) is canonical in the doc at the top of
 * {@link "./pipeline.ts"}; this module holds only the checks passed **before** entering the steps.
 * It holds no Session, no GPU and no ownership of assets (construction and ownership are in
 * `./pipeline.ts`), so the checks can be called directly without a real GPU or real assets
 * (`tests/irodori_admission_test.ts`).
 *
 * MUST: every module is free of side effects (no execution at import time, no global mutable state
 * — CLAUDE.md).
 */

import type { SessionOptions } from "@karume/runtime";
import type { GpuFeaturesSpec, Manifest, ModelEntry, Quant } from "@karume/hub";

import {
  IRODORI_PIPELINE_MAJOR,
  IRODORI_PIPELINE_NAME,
  type IrodoriPipelineConfig,
  parseIrodoriPipelineConfig,
} from "./config.ts";
import type { Assets, PipelineOptions } from "./pipeline.ts";
import { settleAbort } from "../concurrency/abort.ts";
import {
  assertGpuFeaturesGranted,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
} from "../session/gpu-features.ts";
import { type FamilySessionPolicy, resolveSessionOptions } from "../session/options.ts";
import { assetComponentOpener, type ComponentOpener, type GraphOwner } from "../hub/components.ts";
import { readAssetBuffer, readAssetJson } from "../hub/asset-readers.ts";
import { assertGraphInputDim } from "../hub/graph-gates.ts";

/** Names in the manifest's weights / assets tables (the conventional names of ADR 0041 §3). */
export const BACKBONE = "backbone";
export const TEXT_PROJ = "text_proj";
export const CAPTION_PROJ = "caption_proj";
export const SPEAKER = "speaker";
export const DURATION = "duration";
export const DIT = "dit";
/**
 * Context K/V projection (the graph that computes, once per generation and ahead of time, the
 * context K / V that each block of `dit` builds from the 3 conditions — DECIDED: ADR 0114). Its
 * outputs become the inputs of `dit` with the same names, as they are.
 */
export const DIT_CONTEXT = "dit_context";
export const CODEC_DECODER = "codec_decoder";
export const CODEC_ENCODER = "codec_encoder";
export const TOKENIZER = "tokenizer";

/**
 * The weights components of this family (also the role names that component replacement accepts;
 * in declaration order).
 */
export const COMPONENT_KEYS = [
  BACKBONE,
  TEXT_PROJ,
  CAPTION_PROJ,
  SPEAKER,
  DURATION,
  DIT,
  DIT_CONTEXT,
  CODEC_DECODER,
  CODEC_ENCODER,
] as const;

/**
 * Turns fetched bytes into an ArrayBuffer that can be passed to `openContainer` (the check itself
 * is {@link readAssetBuffer}).
 */
const assetBuffer = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): ArrayBuffer => readAssetBuffer("irodori", "weights / assets", assets, key);

/**
 * Component supplier of the full-asset surface (`fromAssets`) (the receiving implementation is
 * shared by the 8 families — {@link assetComponentOpener}). A component key is either one entry
 * for a single-form `krm` (`dit`) or the part list of the split form (`dit[0]` / `dit[1]` / …).
 */
export const assetOpener = (assets: Assets["assets"]): Promise<ComponentOpener> =>
  assetComponentOpener("irodori", assets, (key) => assetBuffer(assets, key), COMPONENT_KEYS);

/**
 * Reads an asset JSON (the decode / parse checks are {@link readAssetJson}).
 *
 * NOTE: `export` is for the tests that call the check directly (reaching here through `fromAssets`
 * needs 9 real IR containers). Not exposed on `mod.ts` / the subpath surface (ADR 0008).
 */
export const assetJson = (
  assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>,
  key: string,
): unknown => readAssetJson("irodori", "weights / assets", assets, key);

/**
 * Checks that the **static** dimension of one axis of a graph input matches the declaration in
 * `pipelineConfig`.
 *
 * MUST: do not remove this check. Even when the declared length or width of a condition state
 * disagrees with the host's value, the right padding and the row arithmetic still go through (the
 * shapes match), and a result that **read the condition at another position** comes out silently.
 */
const assertStaticDim = (
  model: GraphOwner,
  inputName: string,
  axis: number,
  expected: number,
  where: string,
): void => assertGraphInputDim("irodori", model, inputName, axis, expected, where);

/**
 * Checks that one axis of a graph **output** is a derived dimension "symbol × coefficient" whose
 * coefficient matches the declaration.
 *
 * MUST: do not remove this check. Even when the decoder's output ratio (1 latent frame →
 * `hopLength` samples) is off, the output is a "plausibly long waveform", and only the cut to the
 * requested seconds and the tail trim point at another position.
 */
const assertOutputScale = (
  model: GraphOwner,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`irodori: graph output '${name}' is not declared (${where})`);
  }
  const symbol = model.graph.symbols[0];
  // The canonical spelling is `coeff·sym`, and **a coefficient of 1 is omitted** (`formatDim` in
  // `format/dims.ts`). Without matching that spelling, this check would wrongly reject a correct
  // graph with a ratio of 1.
  const canonical = `${expected === 1 ? "" : expected}${symbol}`;
  const dim = value.shape[axis];
  if (symbol === undefined || dim !== canonical) {
    throw new Error(
      `irodori: ${where} — graph output '${name}' has axis ${axis} = ${String(dim)}; ` +
        `pipelineConfig expects '${canonical}'`,
    );
  }
};

/**
 * Checks that one axis of a graph **output** is the declared **static** dimension.
 *
 * MUST: do not remove this check. Even when the encoder's latent width disagrees with `latentDim`
 * of `pipelineConfig`, the later patch goes through as long as "width × patchSize" divides, and a
 * result that **read other channels as speaker features** comes out silently.
 */
const assertOutputDim = (
  model: GraphOwner,
  axis: number,
  expected: number,
  where: string,
): void => {
  const name = model.graph.outputs[0];
  const value = model.graph.values[name];
  if (value === undefined) {
    throw new Error(`irodori: graph output '${name}' is not declared (${where})`);
  }
  const dim = value.shape[axis];
  if (dim !== expected) {
    throw new Error(
      `irodori: ${where} — graph output '${name}' has axis ${axis} = ${String(dim)}; ` +
        `pipelineConfig expects ${expected}`,
    );
  }
};

/** Inputs `dit` takes from the host per forward (the rest are all outputs of `dit_context`). */
const DIT_STEP_INPUTS = ["x_t", "t_embed", "mask"] as const;

/**
 * Checks the wiring from the outputs of `dit_context` to the inputs of `dit`, and derives the sizes
 * of the resident tensors (DECIDED: ADR 0114 — the context K / V run once per generation, and the
 * outputs go to `dit` as resident tensors).
 *
 * It checks 4 points:
 * - `dit_context` has no symbolic dimension (a resident tensor needs its size at allocation, so a
 *   shape that depends on S cannot be placed)
 * - the outputs are an even number, at least 2 (the K / V pair of each block — the block count is
 *   derived from the declaration, not baked in)
 * - the set of `dit` input names = {@link DIT_STEP_INPUTS} ∪ the output names of `dit_context`
 * - each output has a static shape and matches the same-named input of `dit` in dtype and shape
 *
 * MUST: do not remove this check. A resident input is checked **only for its size** (the doc of
 * `RunInput` in the runtime), so even with a mismatched shape or dtype, as long as the byte count
 * matches, the loop runs silently on K / V read as a different layout. A name mismatch also fails
 * at run time, but only **after** GB-class weights have been downloaded.
 */
const admitDitContext = (dit: GraphOwner, ditContext: GraphOwner): readonly DitContextOutput[] => {
  const symbols = ditContext.graph.symbols;
  if (symbols.length !== 0) {
    throw new Error(
      `irodori: dit_context has symbolic dimensions ([${symbols.join(", ")}]) — its outputs must have` +
        " a static shape for one generation",
    );
  }
  const outputs = ditContext.graph.outputs;
  if (outputs.length < 2 || outputs.length % 2 !== 0) {
    throw new Error(
      `irodori: dit_context has ${outputs.length} outputs (one K / V pair per block = an even count` +
        " of at least 2 is required)",
    );
  }
  const ditInputs = new Map(dit.graph.inputs.map((input) => [input.name, input] as const));
  const expected: readonly string[] = [...DIT_STEP_INPUTS, ...outputs];
  const missing = expected.filter((name) => !ditInputs.has(name));
  const extra = [...ditInputs.keys()].filter((name) => !expected.includes(name));
  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      `irodori: the dit inputs do not match {${DIT_STEP_INPUTS.join(", ")}} ∪ the dit_context outputs` +
        ` (missing from dit: [${missing.join(", ")}] / only in dit: [${extra.join(", ")}])`,
    );
  }
  return outputs.map((name) => {
    const value = ditContext.graph.values[name];
    if (value === undefined)
      throw new Error(`irodori: dit_context output '${name}' is not declared`);
    // Always present given the set equality above (checked here only to narrow the type).
    const input = ditInputs.get(name);
    if (input === undefined) throw new Error(`irodori: dit input '${name}' does not exist`);
    const dims = value.shape.filter((dim): dim is number => typeof dim === "number");
    const sameShape =
      value.shape.length === input.shape.length &&
      value.shape.every((dim, axis) => dim === input.shape[axis]);
    if (dims.length !== value.shape.length || value.dtype !== input.dtype || !sameShape) {
      const produced = `${value.dtype} [${value.shape.join(", ")}]`;
      const consumed = `${input.dtype} [${input.shape.join(", ")}]`;
      throw new Error(
        `irodori: dit_context output '${name}' (${produced}) does not match the dit input of the same name (${consumed})` +
          " (the same static shape and the same dtype are required)",
      );
    }
    // Elements are 4 bytes for every dtype (ADR 0009).
    return { name, byteLength: dims.reduce((product, dim) => product * dim, 4) };
  });
};

/**
 * The execution knobs irodori accepts (both from the manifest's quant declaration and from explicit
 * options). They apply only to the Sessions of `dit` and `dit_context`
 * ({@link IrodoriAdmission.ditSessionOptions}). Precedence, value ranges, combinations and the
 * classification of thrown types are owned by `resolveSessionOptions`, shared by every family;
 * this only decides the acceptance set.
 *
 * `attentionCompute` / `attentionScoreStorage` are not accepted because the attention of `dit`
 * takes a runtime bool mask input, so it does not fit the fused-attention contract and is exported
 * on a decomposed path that does not preserve SDPA (matmul + safe_softmax) — the graph has no place
 * where the fused-attention knobs would apply (the `dit` boundary section of
 * `tools/export-recipes/irodori/export.py`). The 4 fields of parallel GEMV and fusion are not
 * accepted because the distributions do not declare them and the combinations have not been
 * checked against irodori's reference values.
 *
 * NOTE: `export` is so that the equivalence test runs every quant of the mirror through the same
 * table (not exposed on `mod.ts` / the subpath surface — ADR 0008).
 */
export const IRODORI_SESSION_POLICY: FamilySessionPolicy = {
  linearCompute: true,
  convCompute: false,
  attentionCompute: false,
  attentionScoreStorage: false,
  linearGemvReduce: false,
  stateAttentionReduce: false,
  fuseRmsNormAdd: false,
  fuseLinearStaticQuantize: false,
  fuseGeluMulStaticQuantize: false,
  packedStaticQuantize: false,
  integerGemv: false,
  linearGemvPlanes: false,
};

/**
 * One output of `dit_context` = one same-named input of `dit` (context K / V — DECIDED: ADR 0114).
 * The name, and the byte count taken for the resident tensor (element count of the static shape × 4
 * — the semantic dtypes of ADR 0009 are all 4 bytes).
 */
export type DitContextOutput = {
  readonly name: string;
  readonly byteLength: number;
};

/** The material that family admission ({@link admitIrodori}) settles. */
export type IrodoriAdmission = {
  readonly config: IrodoriPipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
  /**
   * Name of the symbolic dimension S of `dit` (admission is the only place that derives it).
   *
   * MUST: do not look it up again from `graph.symbols` at run time — if the place that confirmed
   * "there is one symbol" and the place that uses it drift apart, the check runs on some paths and
   * not on others (CLAUDE.md's "derive a derived value once from the source of truth").
   */
  readonly ditSymbol: string;
  /**
   * Output list of `dit_context` (in declaration order) — the resident-tensor allocation of both
   * paths reads only this.
   *
   * MUST: do not derive it again from the declaration at run time (same reason as
   * {@link IrodoriAdmission.ditSymbol} — the place that confirmed "the outputs and the inputs of
   * `dit` correspond 1 to 1 by name and shape" and the place that uses the byte counts stay
   * together).
   */
  readonly ditContextOutputs: readonly DitContextOutput[];
  /**
   * The effective options of `dit` and `dit_context`, combined from the explicit options and the
   * quant declaration ({@link IRODORI_SESSION_POLICY}). Before the split, the condition-side
   * linears ran inside `dit` with the same options, so `dit_context` gets the same object too
   * (running it with other options moves the numbers).
   */
  readonly ditSessionOptions: SessionOptions;
  /**
   * The quant declaration plus the features the effective options need (both the request and the
   * check read this).
   */
  readonly gpuFeatures: GpuFeaturesSpec | undefined;
};

/**
 * Checks whether this manifest and these graphs can run as irodori (the family admission hook of
 * `hub/components.ts` — on the fetch surface it is called **before a single byte of a weight part
 * is fetched**).
 *
 * MUST: the family checks are gathered in this one function. Spread over later steps, on the fetch
 * surface they would fail only **after** GB-class weights are downloaded (the implementation would
 * be narrower than the text of ADR 0070 decision 5).
 * MUST: manifest contract violations and **the matching against the graphs** fail **before the GPU
 * is acquired**. If the order slips, on an environment without a GPU the failure turns into a
 * different exception and the reader never learns "what was wrong".
 *
 * NOTE: the check of {@link PipelineOptions.signal} sits only at **step boundaries**, where
 * it yields to the event loop once before checking ({@link settleAbort}) — an abort that arrives
 * during synchronous parsing takes effect at the next boundary (likewise when `options.gpu` is
 * given). The matching against the graphs (the `assertStaticDim` group) only reads the declarations
 * of the opened containers, so it does not split a boundary.
 * NOTE: the parsing of an asset (tokenizer) cannot be placed here — at admission time only the
 * asset's **declaration** is at hand, not its bytes (waiting for them would lose the position
 * before the weight prefetch), so it stays in `buildIrodoriState` of {@link "./pipeline.ts"}.
 */
export const admitIrodori = async (
  manifest: Manifest,
  open: ComponentOpener,
  options: PipelineOptions,
): Promise<IrodoriAdmission> => {
  // Abort checks sit at **step boundaries**. The entry is the first one: when called already
  // aborted, return without touching the manifest or the assets.
  options.signal?.throwIfAborted();
  const modelName = options.model ?? manifest.defaultModel;
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `irodori.Pipeline: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== IRODORI_PIPELINE_NAME) {
    throw new Error(
      `irodori.Pipeline: the manifest pipeline is '${name}/${major}'` +
        ` ('${IRODORI_PIPELINE_NAME}/${IRODORI_PIPELINE_MAJOR}' is required)`,
    );
  }
  if (major !== IRODORI_PIPELINE_MAJOR) {
    // The only check that stops the silent degradation of "old implementation × new repository"
    // (ADR 0038 §6).
    throw new Error(
      `irodori.Pipeline: pipeline '${name}/${major}' has an unsupported major` +
        ` (this implementation reads ${IRODORI_PIPELINE_NAME}/${IRODORI_PIPELINE_MAJOR})`,
    );
  }
  const config = parseIrodoriPipelineConfig(entry.pipelineConfig);

  const quantName = options.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `irodori.Pipeline: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  const quant = entry.quants[quantName];
  // Explicit > quant declaration > runtime default (one function shared by every family). Both an
  // unsupported declaration and a wrong explicit option fail before the components are opened (the
  // decision depends only on the manifest and the caller's options, so it does not wait for the
  // containers).
  const ditSessionOptions = resolveSessionOptions(
    IRODORI_SESSION_POLICY,
    quant.session,
    options,
    `irodori.Pipeline: quant '${quantName}'`,
  );
  const gpuFeatures = sessionGpuFeatures(quant.gpuFeatures, ditSessionOptions);

  // Checks here that all 9 components can be opened (for a role that is not open, the supplier
  // fails loudly). Abort boundaries sit between the lookups.
  for (const key of COMPONENT_KEYS) {
    await settleAbort(options.signal);
    open(key);
  }
  // Matching against the graphs comes before the GPU (the ordering MUST of the docstring). The
  // matching applies to the 6 below.
  const speaker = open(SPEAKER);
  const duration = open(DURATION);
  const dit = open(DIT);
  const ditContext = open(DIT_CONTEXT);
  const codecDecoder = open(CODEC_DECODER);
  const codecEncoder = open(CODEC_ENCODER);

  // Matching the graph declarations against pipelineConfig (every number the host formulas read
  // comes from the config).
  assertStaticDim(dit, "x_t", 2, config.latentDim, "latentDim");
  assertStaticDim(dit, "t_embed", 1, config.timestepEmbedDim, "timestepEmbedDim");
  // The 3 conditions are taken by `dit_context` (`dit` takes the K / V — ADR 0114).
  assertStaticDim(ditContext, "text_state", 1, config.maxTextLen, "maxTextLen");
  assertStaticDim(ditContext, "text_state", 2, config.textDim, "textDim");
  assertStaticDim(ditContext, "speaker_state", 1, config.speakerRows, "speakerRows");
  assertStaticDim(ditContext, "speaker_state", 2, config.speakerDim, "speakerDim");
  assertStaticDim(ditContext, "caption_state", 1, config.maxCaptionLen, "maxCaptionLen");
  assertStaticDim(ditContext, "caption_state", 2, config.captionDim, "captionDim");
  assertStaticDim(duration, "text_state", 2, config.textDim, "textDim");
  assertStaticDim(duration, "speaker_vec", 1, config.speakerDim, "speakerDim");
  assertStaticDim(duration, "caption_vec", 1, config.captionDim, "captionDim");
  // The patch width of the reference latent (latentDim × speakerPatchSize) matches the input width
  // of speaker.
  assertStaticDim(
    speaker,
    "latent",
    2,
    config.latentDim * config.speakerPatchSize,
    "latentDim × speakerPatchSize",
  );
  // The codec decoder decodes 1 latent frame into hopLength samples. Both the input width and **the
  // coefficient of the output's derived dimension** are checked (an asset where only the
  // coefficient is off passes with matching shapes, and only the sample positions of the cut and
  // the tail trim quietly point somewhere else).
  assertStaticDim(codecDecoder, "latent", 2, config.latentDim, "latentDim");
  assertOutputScale(codecDecoder, 2, config.hopLength, "hopLength");
  // The encoder goes the other way (a `[1,T,hopLength]` waveform → `[1,T,latentDim]`). If the input
  // frame width is not `hopLength`, the waveform the host laid out is read **shifted by one frame
  // at a time**.
  assertStaticDim(codecEncoder, "wav", 2, config.hopLength, "hopLength");
  assertOutputDim(codecEncoder, 2, config.latentDim, "latentDim");
  // `dit` has exactly one symbolic dimension, S (the resident path passes the binding under this
  // name on every enqueue). This is part of the MUST that gathers the family checks in this one
  // function; placed at run time, two things happen: (1) it fails only after GB-class weights are
  // downloaded (2) it does not run on the host path (a `gpuTiming` device / an `onEvent`
  // subscriber), so the same distribution fails with different messages depending on the
  // observation path.
  const ditSymbols = dit.graph.symbols;
  if (ditSymbols.length !== 1) {
    throw new Error(
      `irodori: dit does not have exactly one symbolic dimension ([${ditSymbols.join(", ")}])`,
    );
  }
  const ditSymbol = ditSymbols[0];
  const ditContextOutputs = admitDitContext(dit, ditContext);

  // MUST: a missing capability of a shared GPU (feature / device limit) fails here — unlike the
  // case where the pipeline acquires its own, there is no reason to wait for `acquireGpu`, and this
  // is the only family check that knows before the weights are downloaded (the mapping between
  // request and check is the one in `session/gpu-features.ts`, and the later checks call the same
  // functions).
  if (options.gpu !== undefined) {
    assertGpuFeaturesGranted(gpuFeatures, options.gpu, `irodori.Pipeline: quant '${quantName}'`);
    assertRequiredLimitsSatisfied(
      quant.requiredLimits,
      options.gpu.limits,
      `irodori.Pipeline: quant '${quantName}'`,
    );
  }

  return {
    config,
    quantName,
    quant,
    ditSymbol,
    ditContextOutputs,
    ditSessionOptions,
    gpuFeatures,
  };
};

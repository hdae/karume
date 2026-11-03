/**
 * gemma4 **admission** — the checks that decide whether the distribution's declarations, the
 * product graph and the runtime options can run on this implementation, and the material those
 * checks settle.
 *
 * MUST: the family's checks are gathered here. Scattered into {@link Pipeline}, on the fetch
 * surface they would fail only **after** GB-scale weights were downloaded — each function's doc
 * states at which point (before acquiring the GPU / before reading assets / before fetching weight
 * parts) it is called.
 *
 * This module only decides the acceptance set; it holds no Session, GPU or asset reader
 * (construction and ownership live in `./pipeline.ts`). As a result the checks can be called
 * directly without a real GPU or real assets (`tests/gemma4_config_test.ts`).
 *
 * MUST: zero module side effects (no execution at import, no global mutable state — CLAUDE.md).
 */

import { assertChunkBuckets } from "@karume/runtime";
import type { Manifest, ModelEntry, Quant } from "@karume/hub";

import { ModelInputError } from "../errors.ts";
import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import { assertSuppressTokens, type GenerationGraph } from "../generation/program.ts";
import { GEMMA4_PIPELINE_MAJOR, type PipelineConfig, parsePipelineConfig } from "./config.ts";
import { assertGemma4QatModel, type Gemma4QatModel } from "./qat.ts";
import { declaredGemma4PleAssets } from "./ple-index.ts";
import { GEMMA4_ROPE_LAYER_TYPES, GEMMA4_ROPE_PARTS, gemma4RopeInputName } from "./rope.ts";
import { admitGemma4Drafter, type Gemma4DrafterAdmission } from "./speculative.ts";

export type GemmaFamily = "gemma4" | "gemma4-qat";

/**
 * The entry name of each family (the exception prefix and the first half of `where` come from
 * **this one place**).
 *
 * MUST: messages thrown by the shared base also carry the family's name — a user who called
 * `gemma4Qat.Pipeline.fromPretrained` and receives `gemma.Pipeline: ...` cannot tell which entry it
 * is about.
 */
export const gemmaEntryName = (family: GemmaFamily): string =>
  family === "gemma4" ? "gemma.Pipeline" : "gemma4Qat.Pipeline";

/**
 * The family, and the model name only QAT has (the value {@link admitGemma4Qat} **settles from the
 * graph**).
 *
 * MUST: do not re-derive the model name later (if the discrimination rule splits into two
 * implementations, one keeps using the old mapping when models are added). If `Gemma4QatModel`
 * gains a field, the type check reports the gap.
 */
export type GemmaFamilyAdmission =
  | { readonly family: "gemma4" }
  | { readonly family: "gemma4-qat"; readonly model: Gemma4QatModel };

/**
 * Number of exits of the product graph (**the order is the contract** — output 0 = the logits of
 * the selected rows `[1,R,V]`, output 1 = the final-norm hidden of the same rows `[1,R,H]`).
 *
 * Looked up by order, not by name (`vocabSizeOf` / {@link buildGemma4Program}) — exit names are the
 * baker's internal names and may move between distributions.
 */
const GRAPH_OUTPUTS = 2;

/**
 * The graph input the host PLE gather feeds (source of truth: the constants of `export_product.py`).
 * A product graph either has it (E2B / E4B and the QAT models) or not (gemma-4-12B-it, which has no
 * per-layer inputs).
 */
export const GEMMA4_PER_LAYER_INPUTS = "per_layer_inputs";

/**
 * The material settled by the family admission (the check passed before acquiring the GPU, and on
 * the fetch surface before the weight prefetch).
 *
 * NOTE: component objects are not carried here — the supplier (`ComponentOpener`) is passed to both
 * the earlier and the later stage as **the same one**, so copying an opened component into the
 * material would make "the later stage holds a different component from the one the earlier stage
 * saw" expressible (all the more since component replacement arrived).
 * NOTE: the PLE loader is not carried either — the closure of `wiring.derivedInputs.derive` holds
 * the only reference, and two fields would make "only one of them was replaced" expressible.
 */
export type Gemma4Admission = {
  readonly config: PipelineConfig;
  /**
   * Vocabulary size of the last-row logits exit (the basis of the id-space cross-check — ADR 0085
   * decision 5).
   */
  readonly vocabSize: number;
  /** Capacity symbol of the full slots (the binding point of `createGenerationContext`). */
  readonly capacitySymbol: string;
  /** The drafter cross-check result, settled only when speculation is requested. */
  readonly drafter?: {
    readonly admission: Gemma4DrafterAdmission;
  };
};

/**
 * Reads the vocabulary size of the selected-row logits exit from the graph (`[1, R, V]` — ADR 0083
 * decision 6).
 *
 * MUST: do not make the caller declare it. V is the main embedding's row count itself, and if a
 * declaration disagreed, the cross-check with the PLE index (ADR 0085 decision 5) would pass
 * against **the wrong basis**. `createGenerationProgram` checks the shape again with the same
 * value.
 *
 * MUST: **exactly two** exits (output 0 = logits, output 1 = final-norm hidden). The order is the
 * IR contract; not looking them up by name keeps this independent of the distribution's spelling
 * (the same approach as `capacitySymbolOf`). An old distribution with one exit fails **here** — a
 * compatibility branch would leave "speculation silently cannot be built on an asset without
 * hidden".
 */
const vocabSizeOf = (graph: GenerationGraph, entry: string): number => {
  if (graph.outputs.length !== GRAPH_OUTPUTS) {
    throw new Error(
      `${entry}: the graph has ${graph.outputs.length} outputs` +
        ` (the product graph has ${GRAPH_OUTPUTS} outputs, logits + hidden — ADR 0083 decision 6)`,
    );
  }
  const name = graph.outputs[0];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`${entry}: graph output '${name}' has no value info`);
  }
  const shape = graph.values[name].shape;
  const vocab = shape[2];
  if (shape.length !== 3 || typeof vocab !== "number") {
    throw new Error(`${entry}: graph output '${name}' has shape [${shape.join(",")}], not [1,R,V]`);
  }
  return vocab;
};

/**
 * Reads the width of the final-norm hidden exit from the graph (`[1, R, H]` — output 1, ADR 0083
 * decision 6).
 *
 * Called only for speculation (the drafter's input `hidden` MUST match this width). The count and
 * order were already checked by {@link vocabSizeOf}.
 */
const hiddenSizeOf = (graph: GenerationGraph, entry: string): number => {
  const name = graph.outputs[1];
  if (!Object.hasOwn(graph.values, name)) {
    throw new Error(`${entry}: graph output '${name}' has no value info`);
  }
  const shape = graph.values[name].shape;
  const hidden = shape[2];
  if (shape.length !== 3 || typeof hidden !== "number") {
    throw new Error(`${entry}: graph output '${name}' has shape [${shape.join(",")}], not [1,R,H]`);
  }
  return hidden;
};

/**
 * Reads the capacity symbol of the full slots from the graph.
 *
 * Exactly one symbol should be "not determined by the input shapes" (the chunk-length symbol comes
 * from the second dimension of `input_ids`; the capacity symbol appears only in states), and it is
 * passed to the binding point of `createGenerationContext` (ADR 0066 addendum of 2026-08-17,
 * item 7). Its spelling is not held as a constant, to avoid leaving **a symbol that is silently
 * never bound** when the asset's spelling changes.
 */
const capacitySymbolOf = (graph: GenerationGraph, entry: string): string => {
  const fromInputs = new Set<string>();
  for (const input of graph.inputs) {
    for (const dim of input.shape) {
      if (typeof dim === "string") fromInputs.add(dim);
    }
  }
  const free = graph.symbols.filter((symbol) => !fromInputs.has(symbol));
  if (free.length !== 1) {
    throw new Error(
      `${entry}: ${free.length} symbols are not determined by the input shapes` +
        ` ([${free.join(", ")}] — there must be exactly one, the capacity symbol of the full slots)`,
    );
  }
  return free[0];
};

/**
 * Cross-checks the declared shape (`[1, M, headDim]`) of the four RoPE derived inputs against
 * `pipelineConfig.rope.<layer type>.headDim`.
 *
 * MUST: wiring that can be checked at setup is checked at setup. `createGenerationProgram` checks
 * only **name coverage** of the derived inputs, so a width mismatch (mixing up layer types =
 * reading sliding 256 for full 512 or vice versa — the mistake the exporter's `rope.py`
 * acknowledges in its `head_dim` / `global_head_dim` branch) does not fail until the first `run`
 * receives the table the host passes — **after** a 3.7 GiB load, and with the message "element
 * count does not match the shape". The baking side's mirror is `assert_rope_inputs` in
 * `export_decode.py`.
 *
 * NOTE: an internal function, but exported because at this granularity the declaration cross-check
 * can be pinned without a real GPU or real assets (`tests/gemma4_config_test.ts` — the same
 * approach as siglip2's `assertStaticDim`).
 */
export const assertRopeInputShapes = (
  graph: GenerationGraph,
  config: PipelineConfig,
  // NOTE: the default is regular Gemma's name for the same reason as {@link assertChunkLength}
  // (checks that call this directly have no entry). The real path ({@link admitGemma4}) always
  // passes the family's name.
  entry: string = gemmaEntryName("gemma4"),
): void => {
  for (const layerType of GEMMA4_ROPE_LAYER_TYPES) {
    const { headDim } = config.rope[layerType];
    for (const part of GEMMA4_ROPE_PARTS) {
      const name = gemma4RopeInputName(layerType, part);
      const input = graph.inputs.find((entry) => entry.name === name);
      if (input === undefined) {
        throw new Error(
          `${entry}: graph input '${name}' is missing (RoPE is not an asset the host supplies)`,
        );
      }
      if (input.shape.length !== 3 || input.shape[2] !== headDim) {
        throw new Error(
          `${entry}: graph input '${name}' has shape [${input.shape.join(",")}], which disagrees` +
            ` with pipelineConfig.rope.${layerType}.headDim ${headDim}` +
            ` ([1, M, ${headDim}] is needed)`,
        );
      }
    }
  }
};

/**
 * Checks whether this product graph can run as gemma4 (**before a single byte of the weight parts
 * is fetched**).
 *
 * MUST: the family's checks are gathered in this one function (the same discipline as other
 * families' `admit*` — called at the {@link FamilyAdmission} slot of `hub/components.ts`).
 * Scattered into later stages, on the fetch surface they would fail only **after** GB-scale weights
 * were downloaded.
 *
 * NOTE: tokenizer parsing cannot live here — at admission time the manifest's `assets` have not
 * been fetched yet (fetching them would lose the position before the weight prefetch), so it stays
 * in {@link buildGemma4Program} (the same split as anima's `#admit`). Only the PLE index is a
 * **container asset**, so it can be read at the same point as this check (`readGemma4PleIndex` in
 * `./ple-index.ts`).
 *
 * NOTE: there is no check of `config` **alone** here — **both** entries pass it through
 * {@link parsePipelineConfig} before calling this (that is the canonical place for the checks
 * of ranges, relations and unknown keys; the same check is not implemented twice). What this
 * function checks is only the cross-check **of the declaration against the graph**, and
 * {@link assertRopeInputShapes} is that one (the graph first becomes available at this point).
 */
export const admitGemma4 = (
  component: GraphOwner,
  config: PipelineConfig,
  drafter: GraphOwner | undefined,
  // MUST: the family's entry name ({@link gemmaEntryName}) — both families go through this one
  // function, so if the messages of the checks below carried a fixed name, they would reach QAT
  // users as being about a different entry.
  entry: string,
): Gemma4Admission => {
  const { graph } = component;
  assertRopeInputShapes(graph, config, entry);
  const vocabSize = vocabSizeOf(graph, entry);
  // The suppressed set's vocabulary bound — the first point the vocabulary is known (the parser
  // checked the rest of the declaration). The same check runs again when the wiring is built; this
  // one fails before any weight part is fetched.
  assertSuppressTokens(
    config.suppressTokens ?? [],
    vocabSize,
    `${entry}: pipelineConfig.suppressTokens`,
  );
  const capacitySymbol = capacitySymbolOf(graph, entry);
  return {
    config,
    vocabSize,
    capacitySymbol,
    // The drafter's check lives in `./speculative.ts` (the names of the borrowed slots and shared
    // initializers are speculation knowledge, a separate function from the target's check). The
    // target's material is passed **as settled**.
    ...(drafter === undefined
      ? {}
      : {
          drafter: {
            admission: admitGemma4Drafter(entry, drafter.graph, {
              graph,
              rope: config.rope,
              hiddenSize: hiddenSizeOf(graph, entry),
              capacitySymbol,
            }),
          },
        }),
  };
};

/**
 * Settles whether this product graph carries PLE (per-layer inputs), from **both** of its
 * declarations — the graph input {@link GEMMA4_PER_LAYER_INPUTS} and the container's PLE assets
 * (`declaredGemma4PleAssets` in `./ple-index.ts`). No byte is read, so it runs at the admission
 * step (before any weight part is fetched).
 *
 * MUST: the two agree, or this fails loudly. A graph that takes per-layer inputs from a container
 * without PLE assets cannot be fed; PLE assets next to a graph without the input would be shipped and
 * never read. Either way one side was baked for a different model, and choosing one of them would
 * hide that.
 *
 * Returns `true` for a model with PLE (the caller then reads and checks the index —
 * `readGemma4PleIndex`), `false` for a PLE-free model.
 */
export const admitGemma4Ple = (
  component: GraphOwner & Pick<ModelComponent, "assets">,
  entry: string,
): boolean => {
  const inGraph = component.graph.inputs.some((input) => input.name === GEMMA4_PER_LAYER_INPUTS);
  const assets = declaredGemma4PleAssets(component);
  if (inGraph && assets.length === 0) {
    throw new Error(
      `${entry}: the graph takes '${GEMMA4_PER_LAYER_INPUTS}' but the container declares no PLE assets` +
        ` (a graph and a container baked for different models — the per-layer inputs cannot be built)`,
    );
  }
  if (!inGraph && assets.length > 0) {
    throw new Error(
      `${entry}: the container declares PLE assets (${assets.join(" / ")}) but the graph has no` +
        ` '${GEMMA4_PER_LAYER_INPUTS}' input (a graph and a container baked for different models)`,
    );
  }
  return inGraph;
};

/**
 * Checks and returns the runtime option `chunkLength` (the check of
 * {@link PipelineOptions.chunkLength}).
 *
 * MUST: at least 2 (the graph's chunk symbol was baked with prefill shapes of at least 2, and a
 * one-row chunk runs as the decode shape), at most the `maxChunkLength` the distribution declares,
 * and at most `maxPosition`. The declaration (`parsePipelineConfig`) checks the same
 * relations against the defaults, so what this checks is **the value the caller overrode**.
 *
 * MUST: the `maxChunkLength` check cannot be dropped — the trace range of the symbol `M` is not
 * kept in the asset (the IR's `symbols` is only a list of names), so the declaration is the only
 * source of "the chunk row counts this asset accepts". Before this check existed, passing
 * `chunkLength: 1024` to an asset with a limit of 768 ran without an exception (measured
 * 2026-09-03) — running outside the guarantee is made to fail loudly (cross-cutting invariant).
 *
 * NOTE: the relation to the capacity (`chunkLength ≤ capacity`) is not checked here — the capacity
 * is chosen per sequence, so the only place where both are known is `createGenerationSequence` (the
 * same formula is not kept in two places).
 *
 * NOTE: `export` is for tests that call the check directly (treated like
 * {@link assertRopeInputShapes} — the real paths are `fromAssets` / `fromPretrained` /
 * `estimateSessionMemory`, all of which pass only valid values). Not exposed on `mod.ts` / the
 * subpath surface (ADR 0008).
 */
export const assertChunkLength = (
  chunkLength: number,
  config: PipelineConfig,
  // NOTE: the default is regular Gemma's name because **checks that call this directly** have no
  // entry. The real paths (`buildGemma4Program` / `estimateSessionMemory`) always pass the family's
  // name.
  entry: string = gemmaEntryName("gemma4"),
): number => {
  if (!Number.isSafeInteger(chunkLength) || chunkLength < 2) {
    throw new ModelInputError(`${entry}: chunkLength ${chunkLength} is not an integer ≥ 2`);
  }
  if (chunkLength > config.maxChunkLength) {
    throw new ModelInputError(
      `${entry}: chunkLength ${chunkLength} exceeds the distribution's declared maxChunkLength` +
        ` ${config.maxChunkLength} (outside the trace range the symbol M was baked with)`,
    );
  }
  if (chunkLength > config.maxPosition) {
    throw new ModelInputError(
      `${entry}: chunkLength ${chunkLength} exceeds maxPosition ${config.maxPosition}`,
    );
  }
  return chunkLength;
};

/**
 * Checks and returns the runtime option `chunkBuckets` (the check of
 * {@link PipelineOptions.chunkBuckets}).
 *
 * MUST: the rules of the acceptance set (at least 2, below `chunkLength`, strictly ascending) are
 * **not copied** — the canonical one is the runtime's `assertChunkBuckets`, and it is run here
 * first so that a setting it rejects does not get as far as context creation. All this layer adds
 * is the entry name, so that a caller who passed it to `PipelineOptions` can read "which of
 * my settings failed" (the same approach as `assertChunkLength`).
 *
 * NOTE: no `maxChunkLength` check is needed (buckets are below `chunkLength`, and that
 * `chunkLength` itself has passed the check of {@link assertChunkLength}).
 */
export const assertGemma4ChunkBuckets = (
  chunkBuckets: readonly number[],
  chunkLength: number,
  // NOTE: the reason for the default is the same as {@link assertChunkLength} (for checks that call
  // this directly).
  entry: string = gemmaEntryName("gemma4"),
): readonly number[] => {
  try {
    assertChunkBuckets(chunkBuckets, chunkLength);
  } catch (cause) {
    throw new ModelInputError(
      `${entry}: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
  }
  return chunkBuckets;
};

/**
 * Checks whether this manifest can run as gemma4 (**before touching the GPU or any weight part**).
 *
 * MUST: an unknown major fails loudly (ADR 0038 §1 — the only check that stops the silent
 * degradation of "old implementation × new repository"). Whether `quant` exists is checked before
 * fetching (`resolveSelection` checks the same, but this one fails with the list of available
 * names).
 *
 * MUST: **return the selected `Quant` instead of discarding it** — the pre-download check of
 * `requiredLimits` (ADR 0089 decision 5) is run by the caller (the admission closure of
 * {@link Pipeline.fromPretrained}). Checking only that the name exists and dropping the
 * content here would go back to downloading 3.7 GiB without anyone reading the declared GPU
 * prerequisites.
 */
export const gemma4ManifestConfig = (
  manifest: Manifest,
  selection: { readonly model?: string; readonly quant?: string },
  family: GemmaFamily,
): {
  readonly config: PipelineConfig;
  readonly quantName: string;
  readonly quant: Quant;
} => {
  const where = gemmaEntryName(family);
  const modelName = selection.model ?? manifest.defaultModel;
  if (family === "gemma4-qat") assertGemma4QatModel(modelName);
  if (!Object.hasOwn(manifest.models, modelName)) {
    throw new Error(
      `${where}: model '${modelName}' is not in the manifest` +
        ` (available: ${manifest.available.models.join(" / ")})`,
    );
  }
  const entry: ModelEntry = manifest.models[modelName];
  const { name, major } = entry.pipeline;
  if (name !== family) {
    throw new Error(
      `${where}: the manifest's pipeline is '${name}/${major}'` +
        ` ('${family}/${GEMMA4_PIPELINE_MAJOR}' is needed)`,
    );
  }
  if (major !== GEMMA4_PIPELINE_MAJOR) {
    throw new Error(
      `${where}: unsupported major of pipeline '${name}/${major}'` +
        ` (this implementation reads ${family}/${GEMMA4_PIPELINE_MAJOR})`,
    );
  }
  const quantName = selection.quant ?? entry.defaultQuant;
  if (!Object.hasOwn(entry.quants, quantName)) {
    throw new Error(
      `${where}: quant '${quantName}' is not in the manifest` +
        ` (available: ${entry.available.quants.join(" / ")})`,
    );
  }
  return {
    config: parsePipelineConfig(entry.pipelineConfig),
    quantName,
    quant: entry.quants[quantName],
  };
};

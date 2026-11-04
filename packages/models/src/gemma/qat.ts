/**
 * What the QAT family has of its own: the option types and the RoPE inputs of its entry
 * (`@karume/models/gemma4-qat-mobile`), and the structural checks — separately from regular Gemma's entry,
 * it requires fixed storage and SRQ.
 *
 * The public declarations carry the names users import (`PipelineOptions`, `FromPretrainedOptions`,
 * `ropeInputs`), so they live here rather than next to regular Gemma's declarations of the same
 * names in `pipeline.ts` / `rope.ts`. Regular Gemma's types are referenced through the namespace
 * `gemma` (not import aliases), because JSR prints a type reference as written: the docs then read
 * `gemma.PipelineOptions`, the barrel's spelling, instead of an alias users cannot import.
 */
import { codecLayout, type Tensor } from "@karume/runtime";
import type { GraphOwner } from "../hub/components.ts";
import type { Gemma4PleIndex } from "./ple-index.ts";
import type * as gemma from "./pipeline.ts";
import { buildRopeInputs, type RopeSpec } from "./rope.ts";

type Graph = GraphOwner["graph"];
export type Gemma4QatModel = "e2b" | "e4b";

/** The initial implementation of fixed mobile QAT does not accept a drafter. */
export type PipelineOptions = Omit<gemma.PipelineOptions, "speculative"> & {
  readonly speculative?: never;
};
export type FromPretrainedOptions = Omit<gemma.FromPretrainedOptions, "speculative" | "model"> & {
  readonly speculative?: never;
  readonly model?: "e2b" | "e4b";
};

/**
 * Per-step f32 rounding, for fixed QAT only. Builds the four RoPE inputs of a physical chunk
 * including pad.
 * DECIDED: [ADR 0097 addendum 5](../../../../docs/decisions/0097-gemma4-qat-integration.md#addendum-5-2026-09-11--fixed-qat-recipe-and-how-numeric-comparisons-are-treated).
 */
export const ropeInputs = (spec: RopeSpec, positions: ArrayLike<number>): Record<string, Tensor> =>
  buildRopeInputs(spec, positions, true);

/**
 * Model → PLE storage dtype (the canonical source on the TS side).
 *
 * MUST: do not re-derive it from another key such as the layer count — if the discrimination rule
 * splits into two implementations, one keeps using the old mapping when a third model is added.
 * Adding a field to `Gemma4QatModel` makes the type check report the gap. The primary source is
 * upstream's `QuantizedEmbedding.num_bits` (the recipe's `ple.py` builds `i{bits}` from it), and
 * this is the one place that spells the expected value.
 */
const PLE_STORAGE: Record<Gemma4QatModel, "i2" | "i4"> = { e2b: "i4", e4b: "i2" };

/** PLE dimension per layer (shape[3] of `per_layer_inputs`). */
const PLE_DIM = 256;

export const assertGemma4QatModel = (model: string): Gemma4QatModel => {
  if (model !== "e2b" && model !== "e4b") {
    throw new Error(`gemma4QatMobile.Pipeline: unsupported model '${model}'`);
  }
  return model;
};

export const admitGemma4Qat = (graph: Graph, selected?: string): Gemma4QatModel => {
  const shape = graph.inputs.find((input) => input.name === "per_layer_inputs")?.shape;
  if (shape === undefined) {
    throw new Error("gemma4QatMobile.Pipeline: graph input per_layer_inputs is missing");
  }
  const model = shape[2] === 35 ? "e2b" : shape[2] === 42 ? "e4b" : undefined;
  if (model === undefined) {
    throw new Error(
      `gemma4QatMobile.Pipeline: the layer count ${shape[2]} of per_layer_inputs is not E2B 35 / E4B 42`,
    );
  }
  if (shape[3] !== PLE_DIM) {
    throw new Error(
      `gemma4QatMobile.Pipeline: the PLE dimension ${shape[3]} of per_layer_inputs is not ${PLE_DIM}`,
    );
  }
  const hidden = graph.values[graph.outputs[1]]?.shape[2];
  const expectedHidden = model === "e2b" ? 1536 : 2560;
  if (hidden !== expectedHidden) {
    throw new Error(
      `gemma4QatMobile.Pipeline: hidden width ${hidden} is not ${model}'s ${expectedHidden}`,
    );
  }
  if (selected !== undefined && assertGemma4QatModel(selected) !== model) {
    throw new Error(
      "gemma4QatMobile.Pipeline: the model name and the graph's configuration differ",
    );
  }
  const producers = new Map(
    graph.nodes.flatMap((node) => node.outs.map((out) => [out, node] as const)),
  );
  const consumers = new Map<string, Graph["nodes"][number][]>();
  for (const node of graph.nodes) {
    for (const input of node.ins) {
      const list = consumers.get(input) ?? [];
      list.push(node);
      consumers.set(input, list);
    }
  }
  const embeddings = graph.nodes.filter(
    (node) => node.op === "embedding" && node.ins[1] === "input_ids",
  );
  if (embeddings.length !== 1) {
    throw new Error("gemma4QatMobile.Pipeline: there is not exactly one token embedding");
  }
  const tokenWeight = embeddings[0].ins[0];
  const tokenStorage = graph.initializers[tokenWeight]?.storage;
  if (tokenStorage === undefined || codecLayout(tokenStorage.codec) !== "i2") {
    throw new Error("gemma4QatMobile.Pipeline: the token embedding needs fixed INT2");
  }
  let heads = 0,
    ordinary = 0;
  const storages = new Set<string>();
  for (const node of graph.nodes) {
    if (node.op !== "linear") continue;
    const weight = graph.initializers[node.ins[1]];
    // initializer name = the key of the data (FQN — docs/ir-v2.md).
    if (
      weight?.storage?.codec === "f32" &&
      node.ins[1] === "model.model.per_layer_model_projection.weight"
    ) {
      ordinary++;
      continue;
    }
    if (weight === undefined) {
      throw new Error(
        `gemma4QatMobile.Pipeline: linear weight '${node.ins[1]}' is not an initializer`,
      );
    }
    if (weight.shared !== undefined) {
      throw new Error(
        `gemma4QatMobile.Pipeline: linear weight '${node.ins[1]}' is a shared initializer (QAT does not accept it)`,
      );
    }
    const layout = codecLayout(weight.storage.codec);
    if (!["i2", "i4", "i8"].includes(layout)) {
      throw new Error("gemma4QatMobile.Pipeline: linear needs fixed INT2/INT4/INT8");
    }
    storages.add(layout);
    if (node.ins[1] === tokenWeight) {
      // Only the shared head does not require SRQ before and after. The official checkpoint's
      // lm_head has SRQ scales of 0 on both input and output (uncalibrated = identity), and the
      // recipe does not insert the identity SRQ into the IR (ADR 0097). The form with SRQ inserted
      // is accepted too (it is the identity, so the numbers are the same).
      heads++;
      continue;
    }
    const before = producers.get(node.ins[0]);
    const after = consumers.get(node.outs[0]);
    if (
      before?.op !== "static_quantize" ||
      after?.length !== 1 ||
      after[0].op !== "static_quantize"
    ) {
      throw new Error("gemma4QatMobile.Pipeline: linear needs fixed SRQ before and after it");
    }
  }
  if (heads !== 1) {
    throw new Error(
      `gemma4QatMobile.Pipeline: ${heads} heads share the token embedding (exactly 1 is needed)`,
    );
  }
  if (ordinary !== 1) {
    throw new Error(
      `gemma4QatMobile.Pipeline: per_layer_model_projection has ${ordinary} f32 linears (exactly 1 is needed)`,
    );
  }
  // i2 is always satisfied by the shared head (the head uses the same i2 initializer as the token
  // embedding, and `heads !== 1` already requires exactly one), so it is left out of the coverage
  // condition. The coexistence of i4 and i8 is an assumption based on the measured configuration of
  // the official E2B / E4B (ADR 0097 addendum 7).
  if (!["i4", "i8"].every((dtype) => storages.has(dtype))) {
    throw new Error(
      `gemma4QatMobile.Pipeline: the storage of the quantized linears does not include both fixed INT4 and INT8` +
        ` (${[...storages].sort().join(" / ")})`,
    );
  }
  return model;
};

export const assertGemma4QatPle = (
  model: Gemma4QatModel,
  graph: Graph,
  index: Gemma4PleIndex,
): void => {
  if (index.storage !== PLE_STORAGE[model]) {
    throw new Error(
      `gemma4QatMobile.Pipeline: PLE storage '${index.storage}' differs from ${model}'s` +
        ` '${PLE_STORAGE[model]}'`,
    );
  }
  // The canonical layer count and dimension are the graph's `per_layer_inputs` (the tokens
  // cross-check belongs to `createGemma4Ple` — the same check is not implemented twice).
  const shape = graph.inputs.find((input) => input.name === "per_layer_inputs")?.shape;
  if (index.layers !== shape?.[2]) {
    throw new Error(
      `gemma4QatMobile.Pipeline: PLE layer count ${index.layers} differs from the graph's per_layer_inputs` +
        ` ${shape?.[2]}`,
    );
  }
  if (index.dim !== shape[3]) {
    throw new Error(
      `gemma4QatMobile.Pipeline: PLE dimension ${index.dim} differs from the graph's per_layer_inputs` +
        ` ${shape[3]}`,
    );
  }
};

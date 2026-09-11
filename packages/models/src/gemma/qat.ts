/** QAT family の構造門。通常 Gemma の入口とは別に固定格納と SRQ を要求する。 */
import type { ModelComponent } from "../hub/components.ts";
import type { Gemma4PleIndex } from "./ple.ts";

type Graph = ModelComponent["graph"];
export type Gemma4QatModel = "e2b" | "e4b";

export const assertGemma4QatModel = (model: string): Gemma4QatModel => {
  if (model !== "e2b" && model !== "e4b") {
    throw new Error(`Gemma4QatPipeline: 未対応 model '${model}'`);
  }
  return model;
};

export const admitGemma4Qat = (
  graph: Graph,
  selected?: string,
): Gemma4QatModel => {
  const shape = graph.inputs.find((input) => input.name === "per_layer_inputs")
    ?.shape;
  const model = shape?.[2] === 35 ? "e2b" : shape?.[2] === 42 ? "e4b" : undefined;
  const hidden = graph.values[graph.outputs[1]]?.shape[2];
  if (
    model === undefined || shape?.[3] !== 256 ||
    hidden !== (model === "e2b" ? 1536 : 2560)
  ) {
    throw new Error(
      "Gemma4QatPipeline: E2B/E4B の層数・PLE次元・hidden幅が必要",
    );
  }
  if (selected !== undefined && assertGemma4QatModel(selected) !== model) {
    throw new Error("Gemma4QatPipeline: model 名とグラフの構成が違う");
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
  const embeddings = graph.nodes.filter((node) =>
    node.op === "embedding" && node.ins[1] === "input_ids"
  );
  if (embeddings.length !== 1) {
    throw new Error("Gemma4QatPipeline: token embedding が1本でない");
  }
  const tokenWeight = embeddings[0].ins[0];
  if (graph.initializers[tokenWeight]?.storage.dtype !== "i2") {
    throw new Error("Gemma4QatPipeline: token embedding は固定 INT2 が必要");
  }
  let heads = 0, ordinary = 0;
  const storages = new Set<string>();
  for (const node of graph.nodes) {
    if (node.op !== "linear") continue;
    const weight = graph.initializers[node.ins[1]];
    if (
      weight?.storage.dtype === "f32" &&
      weight.tensor === "model.model.per_layer_model_projection.weight"
    ) {
      ordinary++;
      continue;
    }
    if (
      weight === undefined || weight.shared !== undefined ||
      !["i2", "i4", "i8"].includes(weight.storage.dtype)
    ) {
      throw new Error("Gemma4QatPipeline: linear は固定 INT2/INT4/INT8 が必要");
    }
    storages.add(weight.storage.dtype);
    const before = producers.get(node.ins[0]);
    const after = consumers.get(node.outs[0]);
    if (
      before?.op !== "static_quantize" || after?.length !== 1 ||
      after[0].op !== "static_quantize"
    ) {
      throw new Error("Gemma4QatPipeline: linear の前後に固定 SRQ が必要");
    }
    if (node.ins[1] === tokenWeight) heads++;
  }
  if (
    heads !== 1 || ordinary !== 1 ||
    !["i2", "i4", "i8"].every((dtype) => storages.has(dtype))
  ) {
    throw new Error(
      "Gemma4QatPipeline: 固定混成格納・共有 head・projection の構成が違う",
    );
  }
  return model;
};

export const assertGemma4QatPle = (
  graph: Graph,
  index: Gemma4PleIndex,
): void => {
  const layers = graph.inputs.find((input) => input.name === "per_layer_inputs")
    ?.shape[2];
  if (index.storage !== (layers === 35 ? "i4" : "i2")) {
    throw new Error(
      "Gemma4QatPipeline: PLE は E2B が INT4、E4B が INT2 であること",
    );
  }
};

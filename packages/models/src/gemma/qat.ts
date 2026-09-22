/** QAT family の構造門。通常 Gemma の入口とは別に固定格納と SRQ を要求する。 */
import { codecLayout } from "@karume/runtime";
import type { GraphOwner } from "../hub/components.ts";
import type { Gemma4PleIndex } from "./ple-index.ts";

type Graph = GraphOwner["graph"];
export type Gemma4QatModel = "e2b" | "e4b";

/**
 * モデル → PLE の格納型（TS 側の正本）。
 *
 * MUST: 層数など別の鍵から導き直さない — 判別規則が 2 実装に割れると、3 つ目のモデルが
 * 増えたとき片方だけが古い写像を使い続ける。`Gemma4QatModel` に欄を足せば型検査が欠落を
 * 教える。一次情報は上流 `QuantizedEmbedding.num_bits`（recipe の `ple.py` が `i{bits}` を
 * 作る）で、ここはその期待値の綴り 1 箇所である。
 */
const PLE_STORAGE: Record<Gemma4QatModel, "i2" | "i4"> = { e2b: "i4", e4b: "i2" };

/** PLE の 1 層あたりの次元（`per_layer_inputs` の shape[3]）。 */
const PLE_DIM = 256;

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
  if (shape === undefined) {
    throw new Error("Gemma4QatPipeline: グラフ入力 per_layer_inputs が無い");
  }
  const model = shape[2] === 35 ? "e2b" : shape[2] === 42 ? "e4b" : undefined;
  if (model === undefined) {
    throw new Error(
      `Gemma4QatPipeline: per_layer_inputs の層数 ${shape[2]} が E2B 35 / E4B 42 でない`,
    );
  }
  if (shape[3] !== PLE_DIM) {
    throw new Error(
      `Gemma4QatPipeline: per_layer_inputs の PLE 次元 ${shape[3]} が ${PLE_DIM} でない`,
    );
  }
  const hidden = graph.values[graph.outputs[1]]?.shape[2];
  const expectedHidden = model === "e2b" ? 1536 : 2560;
  if (hidden !== expectedHidden) {
    throw new Error(
      `Gemma4QatPipeline: hidden 幅 ${hidden} が ${model} の ${expectedHidden} でない`,
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
  const tokenStorage = graph.initializers[tokenWeight]?.storage;
  if (tokenStorage === undefined || codecLayout(tokenStorage.codec) !== "i2") {
    throw new Error("Gemma4QatPipeline: token embedding は固定 INT2 が必要");
  }
  let heads = 0, ordinary = 0;
  const storages = new Set<string>();
  for (const node of graph.nodes) {
    if (node.op !== "linear") continue;
    const weight = graph.initializers[node.ins[1]];
    // initializer 名 = 実体の鍵（FQN — docs/ir-v2.md）。
    if (
      weight?.storage?.codec === "f32" &&
      node.ins[1] === "model.model.per_layer_model_projection.weight"
    ) {
      ordinary++;
      continue;
    }
    if (weight === undefined) {
      throw new Error(
        `Gemma4QatPipeline: linear の重み '${node.ins[1]}' が initializer でない`,
      );
    }
    if (weight.shared !== undefined) {
      throw new Error(
        `Gemma4QatPipeline: linear の重み '${node.ins[1]}' は共有 initializer（QAT は受けない）`,
      );
    }
    const layout = codecLayout(weight.storage.codec);
    if (!["i2", "i4", "i8"].includes(layout)) {
      throw new Error("Gemma4QatPipeline: linear は固定 INT2/INT4/INT8 が必要");
    }
    storages.add(layout);
    if (node.ins[1] === tokenWeight) {
      // 共有 head だけ前後の SRQ を要求しない。公式 checkpoint の lm_head は SRQ の scale が
      // 入出力とも 0（未較正 = 恒等）で、recipe は恒等 SRQ を IR に挟まない（ADR 0097）。
      // SRQ を挟んだ形も受ける（恒等なので数値は同じ）。
      heads++;
      continue;
    }
    const before = producers.get(node.ins[0]);
    const after = consumers.get(node.outs[0]);
    if (
      before?.op !== "static_quantize" || after?.length !== 1 ||
      after[0].op !== "static_quantize"
    ) {
      throw new Error("Gemma4QatPipeline: linear の前後に固定 SRQ が必要");
    }
  }
  if (heads !== 1) {
    throw new Error(
      `Gemma4QatPipeline: token embedding を共有する head が ${heads} 本（1本が必要）`,
    );
  }
  if (ordinary !== 1) {
    throw new Error(
      `Gemma4QatPipeline: per_layer_model_projection の f32 linear が ${ordinary} 本（1本が必要）`,
    );
  }
  // i2 は共有 head が必ず満たす（head は token embedding と同じ i2 initializer を使い、
  // `heads !== 1` が既に 1 本を要求している）ので網羅条件から外す。i4 と i8 の同時存在は
  // 公式 E2B / E4B の実測構成に基づく仮定（ADR 0097 追記 7）。
  if (!["i4", "i8"].every((dtype) => storages.has(dtype))) {
    throw new Error(
      `Gemma4QatPipeline: 量子化 linear の格納が固定 INT4 と INT8 を揃えていない` +
        `（${[...storages].sort().join(" / ")}）`,
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
      `Gemma4QatPipeline: PLE の格納 '${index.storage}' が ${model} の` +
        ` '${PLE_STORAGE[model]}' と違う`,
    );
  }
  // 層数と次元の正本はグラフの `per_layer_inputs`（tokens の突合は `createGemma4Ple` が持つ —
  // 同じ検査を 2 実装持たない）。
  const shape = graph.inputs.find((input) => input.name === "per_layer_inputs")
    ?.shape;
  if (index.layers !== shape?.[2]) {
    throw new Error(
      `Gemma4QatPipeline: PLE の層数 ${index.layers} がグラフの per_layer_inputs` +
        ` ${shape?.[2]} と違う`,
    );
  }
  if (index.dim !== shape[3]) {
    throw new Error(
      `Gemma4QatPipeline: PLE の次元 ${index.dim} がグラフの per_layer_inputs` +
        ` ${shape[3]} と違う`,
    );
  }
};

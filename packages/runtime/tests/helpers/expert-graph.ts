// The routed expert graph of the expert-cache tests (ADR 0146): `layers` mixture-of-experts blocks of
// ADR 0145's form, chained on one width, each routed by its own topk:
//
//   scores = linear(x, router)  →  topk(scores, k) = (picked, index)
//   h = expert_linear(x, index, gate_up)        (x shared by the k picks)
//   m = gelu_tanh(h[..., :F]) · h[..., F:]
//   y = expert_linear(m, index, down)           (one row per pick)
//   x' = sum(y · picked[..., None], dim 1)
//
// The expert weights are written as f32, int4-sym-g (f32 steps) or int4-sym-g-f16 (f16 steps), so
// one helper serves the unit tests (planning, ranges, refusals) and the GPU tests (bit identity
// against full residency).

import type { EncodingInput, TensorInput } from "./container-write.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./f16.ts";
import { quantizeI4 } from "./i4.ts";
import { type DeclarationJson, f32Bytes, GRAPH_NAME } from "./model-fixture.ts";

export type ExpertStorage = "f32" | "i4" | "i4f16";

/** The group length of the i4 codecs. */
export const EXPERT_GROUP = 32;

export type ExpertGraphShape = {
  readonly experts: number;
  readonly picks: number;
  /** The width of every layer's input and output (gate_up's `in`, down's `out`). */
  readonly width: number;
  /** The expert FFN width F (gate_up is `[E, 2F, width]`, down `[E, width, F]`). */
  readonly ffn: number;
  readonly layers: number;
};

/** A deterministic value in [-scale, scale) (no random numbers — a failure must reproduce). */
export const wave = (i: number, seed: number, scale: number): number =>
  (((Math.imul(i + seed, 0x9e3779b9) >>> 8) / 2 ** 24) * 2 - 1) * scale;

export const waveValues = (count: number, seed: number, scale: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: count }, (_, i) => wave(i, seed, scale));

/** The names of layer `layer`'s values. */
export const expertLayerNames = (layer: number) => ({
  router: `router_w${layer}`,
  bias: `router_b${layer}`,
  gateUp: `gate_up${layer}`,
  down: `down${layer}`,
  index: `index${layer}`,
});

/**
 * The values, initializers and nodes of the expert layers from `input` to `output` (`[T, width]`
 * each; `T` is the token symbol — the chunk rows `M` of a states-form graph).
 */
export const expertLayers = (
  shape: ExpertGraphShape,
  input: string,
  output: string,
  T = "T",
): Pick<DeclarationJson, "values" | "nodes" | "initializers"> => {
  const { experts: E, picks: K, width: W, ffn: F } = shape;
  const values: DeclarationJson["values"] = {};
  const nodes: DeclarationJson["nodes"] = [];
  const initializers: DeclarationJson["initializers"] = {};
  for (let layer = 0; layer < shape.layers; layer += 1) {
    const names = expertLayerNames(layer);
    const from = layer === 0 ? input : `s${layer - 1}`;
    const to = layer === shape.layers - 1 ? output : `s${layer}`;
    const v = (name: string) => `${name}${layer}`;
    for (const name of [names.router, names.bias, names.gateUp, names.down])
      initializers[name] = {};
    Object.assign(values, {
      [names.router]: { dtype: "f32", shape: [E, W] },
      [names.bias]: { dtype: "f32", shape: [E] },
      [names.gateUp]: { dtype: "f32", shape: [E, 2 * F, W] },
      [names.down]: { dtype: "f32", shape: [E, W, F] },
      [v("scores")]: { dtype: "f32", shape: [T, E] },
      [v("picked")]: { dtype: "f32", shape: [T, K] },
      [names.index]: { dtype: "i32", shape: [T, K] },
      [v("h")]: { dtype: "f32", shape: [T, K, 2 * F] },
      [v("gate")]: { dtype: "f32", shape: [T, K, F] },
      [v("up")]: { dtype: "f32", shape: [T, K, F] },
      [v("act")]: { dtype: "f32", shape: [T, K, F] },
      [v("m")]: { dtype: "f32", shape: [T, K, F] },
      [v("y")]: { dtype: "f32", shape: [T, K, W] },
      [v("w3")]: { dtype: "f32", shape: [T, K, 1] },
      [v("yw")]: { dtype: "f32", shape: [T, K, W] },
      [to]: { dtype: "f32", shape: [T, W] },
    });
    nodes.push(
      { op: "linear", ins: [from, names.router, names.bias], outs: [v("scores")], attrs: {} },
      { op: "topk", ins: [v("scores")], outs: [v("picked"), names.index], attrs: { k: K } },
      { op: "expert_linear", ins: [from, names.index, names.gateUp], outs: [v("h")], attrs: {} },
      { op: "slice", ins: [v("h")], outs: [v("gate")], attrs: { dim: 2, start: 0, end: F } },
      { op: "slice", ins: [v("h")], outs: [v("up")], attrs: { dim: 2, start: F, end: 2 * F } },
      { op: "gelu_tanh", ins: [v("gate")], outs: [v("act")], attrs: {} },
      { op: "mul", ins: [v("act"), v("up")], outs: [v("m")], attrs: {} },
      { op: "expert_linear", ins: [v("m"), names.index, names.down], outs: [v("y")], attrs: {} },
      { op: "reshape", ins: [v("picked")], outs: [v("w3")], attrs: {} },
      { op: "mul", ins: [v("y"), v("w3")], outs: [v("yw")], attrs: {} },
      { op: "sum", ins: [v("yw")], outs: [to], attrs: { dim: 1 } },
    );
  }
  return { values, nodes, initializers };
};

/** The ops of {@link expertLayers}, in code point order (the order a `krm` writes sets in). */
export const EXPERT_OPS = [
  "expert_linear",
  "gelu_tanh",
  "linear",
  "mul",
  "reshape",
  "slice",
  "sum",
  "topk",
];

/** The whole graph: input `x` `[T, width]`, output `out` `[T, width]`. */
export const expertGraph = (shape: ExpertGraphShape): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: [...EXPERT_OPS] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", shape.width] }],
  outputs: ["out"],
  ...expertLayers(shape, "x", "out"),
});

/** The stored bytes and encoding of one expert weight. */
export const storedExpertWeight = (
  storage: ExpertStorage,
  data: Float32Array<ArrayBuffer>,
  shape: readonly number[],
): { readonly bytes: Uint8Array<ArrayBuffer>; readonly encoding: EncodingInput } => {
  if (storage === "f32") return { bytes: f32Bytes(data), encoding: { codec: "f32" } };
  const quantized = quantizeI4(data, shape, EXPERT_GROUP);
  if (storage === "i4") {
    return {
      bytes: quantized.bytes,
      encoding: {
        codec: "int4-sym-g",
        groupSize: EXPERT_GROUP,
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    };
  }
  // The f16 step table: each step rounded once, two per u32 word (step 2i in the low half).
  const bits = Array.from(quantized.scale, (step) => f32ToF16Bits(step));
  return {
    bytes: quantized.bytes,
    encoding: {
      codec: "int4-sym-g-f16",
      groupSize: EXPERT_GROUP,
      scale: { bytes: f16BytesFromBits(bits.length % 2 === 0 ? bits : [...bits, 0]), dtype: "f16" },
    },
  };
};

/** The weights of {@link expertLayers}: f32 routers, the expert weights in `storage`. */
export const expertTensors = (
  shape: ExpertGraphShape,
  storage: ExpertStorage,
): readonly TensorInput[] => {
  const { experts: E, width: W, ffn: F } = shape;
  return Array.from({ length: shape.layers }, (_, layer) => {
    const names = expertLayerNames(layer);
    const seed = 100 * layer;
    const tensor = (initializer: string, stored: Omit<TensorInput, "graph" | "initializer">) => ({
      graph: GRAPH_NAME,
      initializer,
      ...stored,
    });
    return [
      tensor(names.router, {
        bytes: f32Bytes(waveValues(E * W, seed + 3, 0.3)),
        encoding: { codec: "f32" },
      }),
      tensor(names.bias, {
        bytes: f32Bytes(waveValues(E, seed + 5, 0.2)),
        encoding: { codec: "f32" },
      }),
      tensor(
        names.gateUp,
        storedExpertWeight(storage, waveValues(E * 2 * F * W, seed + 7, 0.15), [E, 2 * F, W]),
      ),
      tensor(
        names.down,
        storedExpertWeight(storage, waveValues(E * W * F, seed + 11, 0.15), [E, W, F]),
      ),
    ];
  }).flat();
};

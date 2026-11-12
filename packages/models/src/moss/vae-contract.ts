/**
 * The contract of the MOSS-Image VAE series — the three decoder graphs and the asset
 * `latent_batch_norm` (ADR 0138 decision 8, stage 3; recipe
 * `tools/export-recipes/moss/export_vae.py`).
 *
 * | Graph              | Symbols | Input → output                         |
 * | ------------------ | ------- | -------------------------------------- |
 * | `vae_decoder_pre`  | h, w    | `z [1, Cz, h, w]` → `[1, C, h, w]`     |
 * | `vae_decoder_attn` | S       | `tokens [1, C, S]` → `[1, C, S]`       |
 * | `vae_decoder_post` | h, w    | `hidden [1, C, h, w]` → `[1, 3, k·h, k·w]` |
 *
 * (h, w = the VAE latent's height and width, S = h·w, Cz = 32 latent channels, C = 384 the mid
 * block's width, k = 8 the spatial scale.) `[1, C, h, w]` and `[1, C, S]` are the same bytes — the
 * stage hands them over by reinterpreting resident tensors (`./vae-stage.ts`).
 *
 * ## The area limit (ADR 0138 decision 8 — per device)
 *
 * The decoder runs untiled, so every intermediate grows with the area. The largest one must fit
 * one storage binding: the contract reads every value declared in the three graphs, requires each
 * symbolic one to be proportional to the area (`K·h·w` in pre / post — written with the
 * coefficients of the canonical `coeff·sym` dimensions — or `K·S` in attn) and keeps the largest
 * `K` as bytes per latent cell ({@link MossVaeContract.bytesPerCell}; the FLUX.2 small decoder's
 * `[1, 192, 8h, 8w]` f32 — 49,152 B per cell, 768 B per pixel). {@link mossVaeAreaLimit} divides a
 * device's binding limit by it (about 2.80M pixels at 2 GiB); the acceptance check refuses a larger
 * area with an error that names the limit (`./request.ts`). A value of another form (an offset,
 * one spatial symbol alone) fails the admission loudly: the limit would not be the area's.
 *
 * MUST: the checks run in the family admission, before the weight parts are fetched (the DiT rule
 * — `./dit-contract.ts`).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import { parseSafetensors } from "@karume/runtime";

import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import { MOSS_PIXELS_PER_CELL } from "./request.ts";

/** The graph names (= the series part directories and the components). */
export const MOSS_VAE_DECODER_PRE = "vae_decoder_pre";
export const MOSS_VAE_DECODER_ATTN = "vae_decoder_attn";
export const MOSS_VAE_DECODER_POST = "vae_decoder_post";

/** The asset of `vae_decoder_pre` and its role (the BatchNorm restoration's `mean` / `std`). */
export const MOSS_LATENT_BATCH_NORM = "latent_batch_norm";
export const MOSS_LATENT_BATCH_NORM_ROLE = "latent-batch-norm";

/** The RGB channels of the decoded image. */
const IMAGE_CHANNELS = 3;

/** Bytes per element (every semantic dtype is 4 bytes — ADR 0009). */
const BYTES_PER_ELEMENT = 4;

/** The three graphs (what the contract reads). */
export type MossVaeGraphs = {
  readonly pre: GraphOwner & Pick<ModelComponent, "assets">;
  readonly attn: GraphOwner;
  readonly post: GraphOwner;
};

/** One graph's single input and output names. */
type Io = { readonly input: string; readonly output: string };

/** What the declarations settle ({@link mossVaeGraphContract}). */
export type MossVaeGraphContract = {
  readonly pre: Io & { readonly height: string; readonly width: string };
  readonly attn: Io & { readonly sequence: string };
  readonly post: Io & { readonly height: string; readonly width: string };
  /** Cz — the VAE latent channels (32; the DiT's 128 = Cz × 2×2). */
  readonly latentChannels: number;
  /** C — the mid block's width (384). */
  readonly hiddenChannels: number;
  /** k — image pixels per VAE latent cell along each side (8). */
  readonly spatialScale: number;
  /** The bytes of the largest area-proportional value per VAE latent cell (module doc). */
  readonly bytesPerCell: number;
};

/** The BatchNorm restoration's two vectors (`latents · std + mean` per DiT channel). */
export type MossLatentBatchNorm = {
  readonly mean: Float32Array<ArrayBuffer>;
  readonly std: Float32Array<ArrayBuffer>;
};

/** The contract with the asset matched ({@link mossVaeContract}). */
export type MossVaeContract = MossVaeGraphContract & {
  readonly batchNorm: MossLatentBatchNorm;
  /** The DiT-space channels the BatchNorm covers (Cz · 2·2 = 128). */
  readonly ditChannels: number;
};

type Dims = readonly (number | string)[];

const sameDims = (left: Dims, right: Dims): boolean =>
  left.length === right.length && left.every((dim, axis) => dim === right[axis]);

/** The canonical `coeff·sym` dimension without offset (`"8h"`, `"h"`), or undefined. */
const scaledSymbol = (dim: number | string): { coeff: number; sym: string } | undefined => {
  if (typeof dim !== "string") return undefined;
  const match = /^(\d*)([A-Za-z_][A-Za-z0-9_]*)$/.exec(dim);
  if (match === null) return undefined;
  return { coeff: match[1] === "" ? 1 : Number(match[1]), sym: match[2] };
};

const scaledDim = (scale: number, symbol: string): string =>
  scale === 1 ? symbol : `${scale}${symbol}`;

/** The graph's one f32 input and one f32 output. */
const soleIo = (owner: string, part: string, model: GraphOwner) => {
  const { inputs, outputs, values } = model.graph;
  if (inputs.length !== 1 || outputs.length !== 1) {
    throw new Error(
      `${owner}: ${part} has ${inputs.length} inputs → ${outputs.length} outputs (only 1 → 1 is built)`,
    );
  }
  const [input] = inputs;
  const [output] = outputs;
  if (!Object.hasOwn(values, output)) {
    throw new Error(`${owner}: ${part} output '${output}' is not declared`);
  }
  for (const [what, dtype] of [
    [`input '${input.name}'`, input.dtype],
    [`output '${output}'`, values[output].dtype],
  ] as const) {
    if (dtype !== "f32") throw new Error(`${owner}: ${part} ${what} is ${dtype}, not f32`);
  }
  return { input: input.name, inputShape: input.shape, output, outputShape: values[output].shape };
};

const assertSymbols = (owner: string, part: string, model: GraphOwner, expected: Dims) => {
  const declared = [...model.graph.symbols].sort();
  const wanted = [...expected].map(String).sort();
  if (!sameDims(declared, wanted)) {
    throw new Error(
      `${owner}: ${part} declares symbols [${model.graph.symbols.join(", ")}] (expected [${expected.join(", ")}])`,
    );
  }
};

const assertShape = (owner: string, where: string, actual: Dims, expected: Dims): void => {
  if (!sameDims(actual, expected)) {
    throw new Error(
      `${owner}: ${where} has shape [${actual.join(", ")}] (expected [${expected.join(", ")}])`,
    );
  }
};

const staticAt = (owner: string, where: string, dims: Dims, axis: number): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") throw new Error(`${owner}: ${where} axis ${axis} is not static`);
  return dim;
};

const symbolAt = (owner: string, where: string, dims: Dims, axis: number): string => {
  const dim = dims.at(axis);
  if (typeof dim !== "string") throw new Error(`${owner}: ${where} axis ${axis} is not a symbol`);
  return dim;
};

/**
 * Elements per area unit of every symbolic value of one graph (module doc "The area limit"): the
 * product of the static extents and of the symbolic coefficients, where the symbolic dimensions
 * must be exactly one of each `symbols` (no offset). Returns the largest.
 */
const largestPerArea = (
  owner: string,
  part: string,
  model: GraphOwner,
  symbols: readonly string[],
): number => {
  const { values, inputs } = model.graph;
  const shapes: [string, Dims][] = [
    ...inputs.map((input): [string, Dims] => [input.name, input.shape]),
    ...Object.entries(values).map(([name, info]): [string, Dims] => [name, info.shape]),
  ];
  let largest = 0;
  for (const [name, shape] of shapes) {
    if (shape.every((dim) => typeof dim === "number")) continue;
    const seen: string[] = [];
    let elements = 1;
    for (const dim of shape) {
      if (typeof dim === "number") {
        elements *= dim;
        continue;
      }
      const scaled = scaledSymbol(dim);
      if (scaled === undefined || !symbols.includes(scaled.sym)) {
        throw new Error(
          `${owner}: ${part} value '${name}' has dimension '${dim}' (not coeff·symbol)`,
        );
      }
      seen.push(scaled.sym);
      elements *= scaled.coeff;
    }
    if (!sameDims([...seen].sort(), [...symbols].sort())) {
      throw new Error(
        `${owner}: ${part} value '${name}' [${shape.join(", ")}] does not grow with the area (symbols [${symbols.join(", ")}] once each)`,
      );
    }
    largest = Math.max(largest, elements);
  }
  return largest;
};

/** Checks the three graph declarations and the asset declaration (module doc). */
export const mossVaeGraphContract = (
  graphs: MossVaeGraphs,
  owner: string,
): MossVaeGraphContract => {
  const pre = soleIo(owner, MOSS_VAE_DECODER_PRE, graphs.pre);
  const preIn = `${MOSS_VAE_DECODER_PRE} input '${pre.input}'`;
  const latentChannels = staticAt(owner, preIn, pre.inputShape, 1);
  const height = symbolAt(owner, preIn, pre.inputShape, 2);
  const width = symbolAt(owner, preIn, pre.inputShape, 3);
  assertSymbols(owner, MOSS_VAE_DECODER_PRE, graphs.pre, [height, width]);
  assertShape(owner, preIn, pre.inputShape, [1, latentChannels, height, width]);
  const preOut = `${MOSS_VAE_DECODER_PRE} output '${pre.output}'`;
  const hiddenChannels = staticAt(owner, preOut, pre.outputShape, 1);
  assertShape(owner, preOut, pre.outputShape, [1, hiddenChannels, height, width]);

  const attn = soleIo(owner, MOSS_VAE_DECODER_ATTN, graphs.attn);
  const attnIn = `${MOSS_VAE_DECODER_ATTN} input '${attn.input}'`;
  const sequence = symbolAt(owner, attnIn, attn.inputShape, 2);
  assertSymbols(owner, MOSS_VAE_DECODER_ATTN, graphs.attn, [sequence]);
  assertShape(owner, attnIn, attn.inputShape, [1, hiddenChannels, sequence]);
  assertShape(owner, `${MOSS_VAE_DECODER_ATTN} output '${attn.output}'`, attn.outputShape, [
    1,
    hiddenChannels,
    sequence,
  ]);

  const post = soleIo(owner, MOSS_VAE_DECODER_POST, graphs.post);
  const postIn = `${MOSS_VAE_DECODER_POST} input '${post.input}'`;
  const postHeight = symbolAt(owner, postIn, post.inputShape, 2);
  const postWidth = symbolAt(owner, postIn, post.inputShape, 3);
  assertSymbols(owner, MOSS_VAE_DECODER_POST, graphs.post, [postHeight, postWidth]);
  assertShape(owner, postIn, post.inputShape, [1, hiddenChannels, postHeight, postWidth]);
  const postOut = `${MOSS_VAE_DECODER_POST} output '${post.output}'`;
  const scaled = scaledSymbol(post.outputShape.at(2) ?? 0);
  const spatialScale = scaled?.coeff ?? 0;
  if (spatialScale !== MOSS_PIXELS_PER_CELL / 2) {
    throw new Error(
      `${owner}: ${postOut} scales the latent by ${spatialScale} (the DiT's ${MOSS_PIXELS_PER_CELL}-px cell is 2×2 VAE cells of ${MOSS_PIXELS_PER_CELL / 2} px)`,
    );
  }
  assertShape(owner, postOut, post.outputShape, [
    1,
    IMAGE_CHANNELS,
    scaledDim(spatialScale, postHeight),
    scaledDim(spatialScale, postWidth),
  ]);

  if (graphs.pre.assets[MOSS_LATENT_BATCH_NORM] !== MOSS_LATENT_BATCH_NORM_ROLE) {
    throw new Error(
      `${owner}: the ${MOSS_VAE_DECODER_PRE} container has no asset '${MOSS_LATENT_BATCH_NORM}' of role '${MOSS_LATENT_BATCH_NORM_ROLE}'`,
    );
  }
  const perCell = Math.max(
    largestPerArea(owner, MOSS_VAE_DECODER_PRE, graphs.pre, [height, width]),
    largestPerArea(owner, MOSS_VAE_DECODER_ATTN, graphs.attn, [sequence]),
    largestPerArea(owner, MOSS_VAE_DECODER_POST, graphs.post, [postHeight, postWidth]),
  );
  return {
    pre: { input: pre.input, output: pre.output, height, width },
    attn: { input: attn.input, output: attn.output, sequence },
    post: { input: post.input, output: post.output, height: postHeight, width: postWidth },
    latentChannels,
    hiddenChannels,
    spatialScale,
    bytesPerCell: perCell * BYTES_PER_ELEMENT,
  };
};

/** Parses the `latent_batch_norm` asset (safetensors `mean` / `std`, both f32 `[C]`). */
export const parseMossLatentBatchNorm = (buffer: ArrayBuffer): MossLatentBatchNorm => {
  const file = parseSafetensors(buffer);
  const names = [...file.tensors.keys()].sort();
  if (names.join(",") !== "mean,std") {
    throw new Error(`${MOSS_LATENT_BATCH_NORM} holds [${names.join(", ")}] (mean, std)`);
  }
  const vector = (name: string): Float32Array<ArrayBuffer> => {
    const view = file.tensors.get(name);
    if (view === undefined || view.dtype !== "F32" || view.shape.length !== 1) {
      throw new Error(`${MOSS_LATENT_BATCH_NORM} '${name}' is not an f32 vector`);
    }
    // A copy (the parsed file is a view of the asset bytes).
    return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4).slice();
  };
  const mean = vector("mean");
  const std = vector("std");
  if (mean.length !== std.length) {
    throw new Error(`${MOSS_LATENT_BATCH_NORM}: ${mean.length} means and ${std.length} stds`);
  }
  return { mean, std };
};

/** Matches the asset to the graph contract (the DiT-space channels are Cz × 2 × 2). */
export const mossVaeContract = (
  graph: MossVaeGraphContract,
  batchNorm: MossLatentBatchNorm,
  owner: string,
): MossVaeContract => {
  const ditChannels = graph.latentChannels * 4;
  if (batchNorm.mean.length !== ditChannels) {
    throw new Error(
      `${owner}: ${MOSS_LATENT_BATCH_NORM} covers ${batchNorm.mean.length} channels (the DiT's ${ditChannels} = ${graph.latentChannels} × 2×2)`,
    );
  }
  const broken = batchNorm.std.findIndex((value) => !(value > 0 && Number.isFinite(value)));
  if (broken !== -1 || batchNorm.mean.some((value) => !Number.isFinite(value))) {
    throw new Error(`${owner}: ${MOSS_LATENT_BATCH_NORM} holds a non-finite or non-positive value`);
  }
  return { ...graph, batchNorm, ditChannels };
};

/** Reads and parses the asset of `vae_decoder_pre`. */
export const readMossLatentBatchNorm = async (
  pre: ModelComponent,
): Promise<MossLatentBatchNorm> => {
  const reader = pre.asset(MOSS_LATENT_BATCH_NORM);
  const bytes = await reader.read(0, reader.length);
  // A copy: the reader's bytes may be a view of a whole part (the `AssetReader` contract).
  return parseMossLatentBatchNorm(bytes.slice().buffer);
};

/** The device limits the area limit and the estimates take. */
export type MossVaeDeviceLimits = {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
  /**
   * Not read by the area limit. The VAE stage's estimates require it (`mossVaeStageNeeds`): they
   * plan with the family's rule list (`./fusion-rules.ts`), and a rule's matcher may read it
   * (`fusionRulesEstimateOptions`).
   */
  readonly maxComputeWorkgroupsPerDimension?: number;
};

/** The largest accepted area on a device (module doc "The area limit"). */
export type MossAreaLimit = {
  /** The largest accepted image area in pixels (a multiple of the 8×8 VAE cell). */
  readonly maxPixels: number;
  /** The largest decoder tensor's bytes per image pixel. */
  readonly bytesPerPixel: number;
  /** The binding the largest tensor must fit (min of the binding and the buffer limits). */
  readonly bindingBytes: number;
};

/** The area limit of `contract` on a device of `limits`. */
export const mossVaeAreaLimit = (
  contract: Pick<MossVaeGraphContract, "bytesPerCell" | "spatialScale">,
  limits: MossVaeDeviceLimits,
): MossAreaLimit => {
  const bindingBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  const cellPixels = contract.spatialScale * contract.spatialScale;
  return {
    maxPixels: Math.floor(bindingBytes / contract.bytesPerCell) * cellPixels,
    bytesPerPixel: contract.bytesPerCell / cellPixels,
    bindingBytes,
  };
};

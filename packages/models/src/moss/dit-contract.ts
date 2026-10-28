/**
 * The contract of the MOSS-Image DiT series — the three graphs `dit`, `dit_skip22` and `dit_text`
 * and the two assets of `dit` (ADR NNNN decisions 2–4 and 11, stage 2).
 *
 * | Graph        | Symbol | Inputs                                                                                         | Output       |
 * | ------------ | ------ | ---------------------------------------------------------------------------------------------- | ------------ |
 * | `dit`        | S      | `latent_tokens [1,S,C]`, `text_context [1,S,D]`, `is_image [1,S,1]` bool, `rope_cos` / `rope_sin [1,1,S,R]`, `timesteps_proj [1,P]` | `[1,S,C]` |
 * | `dit_skip22` | S      | the same as `dit` (block 22 left out; every weight borrowed from `dit`)                          | `[1,S,C]`    |
 * | `dit_text`   | T      | `text_embeds [1,T,4096]`, `rope_cos` / `rope_sin [1,1,T,R]`                                      | `[1,T,D]`    |
 *
 * (C = 128 latent channels, D = 3072, R = 128 = the RoPE width of `rope_base`, P = 256 = the width
 * of `training_grid`'s sinusoid.) The `dit` container carries the assets `rope_base` (role
 * `rope-base` — `./dit-rope.ts`) and `training_grid` (role `training-grid` — `./dit-timestep.ts`).
 *
 * MUST: the checks run in the family admission, before the weight parts are fetched — the host
 * builds its inputs with its own constants, so a graph exported with other dimensions would otherwise
 * run the host side to the end and fail only in a Session's shape check after uploading 4.7 GiB of
 * weights (the Wan rule — `../wan/dit-loop.ts`). The declarations alone decide the graph checks
 * ({@link mossDitGraphContract}); the base tables are matched once their bytes are read
 * ({@link mossDitContract}).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { GraphOwner, ModelComponent } from "../hub/components.ts";
import { type MossRopeBase, mossRopeWidth, parseMossRopeBase } from "./dit-rope.ts";
import { type MossTrainingGrid, parseMossTrainingGrid } from "./dit-timestep.ts";
import { MOSS_MAX_SIDE_CELLS, MOSS_MAX_TEXT_TOKENS, MOSS_TEXT_WIDTH } from "./request.ts";

/** The graph names (= the series part directories and the components). */
export const MOSS_DIT = "dit";
export const MOSS_DIT_SKIP22 = "dit_skip22";
export const MOSS_DIT_TEXT = "dit_text";

/** The `dit` container's assets and their roles. */
export const MOSS_ROPE_BASE = "rope_base";
export const MOSS_ROPE_BASE_ROLE = "rope-base";
export const MOSS_TRAINING_GRID = "training_grid";
export const MOSS_TRAINING_GRID_ROLE = "training-grid";

/** The input names (the forward argument names of the recipe's `dit_patch.py`). */
export const MOSS_DIT_INPUTS = {
  latentTokens: "latent_tokens",
  textContext: "text_context",
  isImage: "is_image",
  ropeCos: "rope_cos",
  ropeSin: "rope_sin",
  timestepsProj: "timesteps_proj",
} as const;
export const MOSS_DIT_TEXT_INPUTS = {
  textEmbeds: "text_embeds",
  ropeCos: "rope_cos",
  ropeSin: "rope_sin",
} as const;

/** The three graphs (what the contract reads — a graph declaration and, for `dit`, its assets). */
export type MossDitGraphs = {
  readonly dit: GraphOwner & Pick<ModelComponent, "assets">;
  readonly ditSkip22: GraphOwner;
  readonly ditText: GraphOwner;
};

/** What the declarations settle (before any asset byte is read). */
export type MossDitGraphContract = {
  /** C — the latent channels (128). */
  readonly latentChannels: number;
  /** D — the DiT width (3072; `text_context` and the `dit_text` output). */
  readonly width: number;
  /** R — the RoPE table width of all three graphs (128). */
  readonly ropeWidth: number;
  /** P — the width of `timesteps_proj` (256). */
  readonly projWidth: number;
  /** The output name per graph. */
  readonly outputs: {
    readonly dit: string;
    readonly ditSkip22: string;
    readonly ditText: string;
  };
  /** The initializers `dit_skip22` borrows from `dit` (by the same name — ADR 0096 decision 2). */
  readonly borrowed: readonly string[];
};

/** The contract with the base tables matched ({@link mossDitContract}). */
export type MossDitContract = MossDitGraphContract & {
  readonly ropeBase: MossRopeBase;
  readonly trainingGrid: MossTrainingGrid;
};

type Dims = readonly (number | string)[];

const sameDims = (left: Dims, right: Dims): boolean =>
  left.length === right.length && left.every((dim, axis) => dim === right[axis]);

const inputOf = (owner: string, graph: string, model: GraphOwner, name: string) => {
  const spec = model.graph.inputs.find((input) => input.name === name);
  if (spec === undefined) throw new Error(`${owner}: ${graph} input '${name}' is missing`);
  return spec;
};

const staticDim = (owner: string, where: string, dims: Dims, axis: number): number => {
  const dim = dims.at(axis);
  if (typeof dim !== "number") throw new Error(`${owner}: ${where} axis ${axis} is not static`);
  return dim;
};

const assertInput = (
  owner: string,
  graph: string,
  model: GraphOwner,
  name: string,
  dtype: "f32" | "bool",
  expected: Dims,
): void => {
  const spec = inputOf(owner, graph, model, name);
  if (spec.dtype !== dtype) {
    throw new Error(`${owner}: ${graph} input '${name}' is ${spec.dtype}, not ${dtype}`);
  }
  if (!sameDims(spec.shape, expected)) {
    throw new Error(
      `${owner}: ${graph} input '${name}' has shape [${spec.shape.join(", ")}]` +
        ` (the host builds [${expected.join(", ")}])`,
    );
  }
};

/** The one symbol of a graph (exactly one — ADR 0034). */
const soleSymbol = (owner: string, graph: string, model: GraphOwner): string => {
  const { symbols } = model.graph;
  if (symbols.length !== 1) {
    throw new Error(`${owner}: ${graph} declares symbols [${symbols.join(", ")}] (exactly one)`);
  }
  return symbols[0];
};

/** The input names match exactly (the host builds only these). */
const assertInputNames = (
  owner: string,
  graph: string,
  model: GraphOwner,
  expected: readonly string[],
): void => {
  const declared = model.graph.inputs.map((input) => input.name);
  if (
    declared.length !== expected.length ||
    [...declared].sort().some((name, index) => name !== [...expected].sort()[index])
  ) {
    throw new Error(
      `${owner}: ${graph} inputs are [${declared.join(", ")}] (expected [${expected.join(", ")}])`,
    );
  }
};

/** The one output of a graph and its shape. */
const soleOutput = (owner: string, graph: string, model: GraphOwner, expected: Dims): string => {
  const { outputs, values } = model.graph;
  if (outputs.length !== 1) {
    throw new Error(`${owner}: ${graph} has ${outputs.length} outputs (exactly one)`);
  }
  const [output] = outputs;
  if (!Object.hasOwn(values, output)) {
    throw new Error(`${owner}: ${graph} output '${output}' is not declared`);
  }
  if (!sameDims(values[output].shape, expected)) {
    throw new Error(
      `${owner}: ${graph} output '${output}' has shape [${values[output].shape.join(", ")}]` +
        ` (expected [${expected.join(", ")}])`,
    );
  }
  return output;
};

/**
 * The S-form inputs of `dit` (and `dit_skip22`) — returns `(C, D, R, P)` after matching every
 * input to the shape the host builds with the graph's own symbol.
 */
const ditInputs = (owner: string, graph: string, model: GraphOwner) => {
  const symbol = soleSymbol(owner, graph, model);
  const names = MOSS_DIT_INPUTS;
  assertInputNames(owner, graph, model, Object.values(names));
  const tokens = inputOf(owner, graph, model, names.latentTokens).shape;
  const latentChannels = staticDim(owner, `${graph} '${names.latentTokens}'`, tokens, 2);
  const context = inputOf(owner, graph, model, names.textContext).shape;
  const width = staticDim(owner, `${graph} '${names.textContext}'`, context, 2);
  const rope = inputOf(owner, graph, model, names.ropeCos).shape;
  const ropeWidth = staticDim(owner, `${graph} '${names.ropeCos}'`, rope, 3);
  const proj = inputOf(owner, graph, model, names.timestepsProj).shape;
  const projWidth = staticDim(owner, `${graph} '${names.timestepsProj}'`, proj, 1);
  assertInput(owner, graph, model, names.latentTokens, "f32", [1, symbol, latentChannels]);
  assertInput(owner, graph, model, names.textContext, "f32", [1, symbol, width]);
  assertInput(owner, graph, model, names.isImage, "bool", [1, symbol, 1]);
  assertInput(owner, graph, model, names.ropeCos, "f32", [1, 1, symbol, ropeWidth]);
  assertInput(owner, graph, model, names.ropeSin, "f32", [1, 1, symbol, ropeWidth]);
  assertInput(owner, graph, model, names.timestepsProj, "f32", [1, projWidth]);
  const output = soleOutput(owner, graph, model, [1, symbol, latentChannels]);
  return { symbol, latentChannels, width, ropeWidth, projWidth, output };
};

/**
 * The initializers `dit_skip22` borrows (IR v2: a shared initializer's name is the lender's —
 * ADR 0096 decision 2).
 *
 * MUST: every initializer of the borrower is shared (ADR NNNN decision 4 — the graph holds no weight
 * bytes), and each names a lender initializer that is not itself shared. The runtime's own checks
 * (shape, dtype, residency class, device, "every shared declaration, no more, no fewer") run when the
 * borrower's Session is built.
 */
const borrowedInitializers = (owner: string, borrower: GraphOwner, lender: GraphOwner) => {
  const names = Object.keys(borrower.graph.initializers);
  const owned = names.filter((name) => borrower.graph.initializers[name].shared !== true);
  if (owned.length > 0) {
    throw new Error(
      `${owner}: ${MOSS_DIT_SKIP22} owns ${owned.length} initializers (${owned.slice(0, 3).join(", ")}…)` +
        ` — it borrows every weight of ${MOSS_DIT}`,
    );
  }
  const unbacked = names.filter(
    (name) =>
      !Object.hasOwn(lender.graph.initializers, name) ||
      lender.graph.initializers[name].shared === true,
  );
  if (unbacked.length > 0) {
    throw new Error(
      `${owner}: ${unbacked.length} initializers of ${MOSS_DIT_SKIP22} have no weight of the same name` +
        ` in ${MOSS_DIT} (${unbacked.slice(0, 3).join(", ")}…)`,
    );
  }
  if (names.length === 0) throw new Error(`${owner}: ${MOSS_DIT_SKIP22} declares no initializer`);
  return names;
};

const assertAsset = (owner: string, dit: MossDitGraphs["dit"], name: string, role: string) => {
  if (!Object.hasOwn(dit.assets, name)) {
    throw new Error(`${owner}: the ${MOSS_DIT} container has no asset '${name}'`);
  }
  if (dit.assets[name] !== role) {
    throw new Error(
      `${owner}: asset '${name}' has role '${dit.assets[name]}' (expected '${role}')`,
    );
  }
};

/**
 * Checks the three graph declarations and the asset declarations (module doc) — the part of the
 * contract that needs no asset byte.
 */
export const mossDitGraphContract = (
  graphs: MossDitGraphs,
  owner: string,
): MossDitGraphContract => {
  const dit = ditInputs(owner, MOSS_DIT, graphs.dit);
  const skip = ditInputs(owner, MOSS_DIT_SKIP22, graphs.ditSkip22);
  for (const [name, mine, theirs] of [
    ["latent channels", skip.latentChannels, dit.latentChannels],
    ["width", skip.width, dit.width],
    ["RoPE width", skip.ropeWidth, dit.ropeWidth],
    ["timesteps_proj width", skip.projWidth, dit.projWidth],
  ] as const) {
    if (mine !== theirs) {
      throw new Error(
        `${owner}: ${MOSS_DIT_SKIP22} ${name} ${mine} differs from ${MOSS_DIT}'s ${theirs}`,
      );
    }
  }
  const text = MOSS_DIT_TEXT_INPUTS;
  const symbol = soleSymbol(owner, MOSS_DIT_TEXT, graphs.ditText);
  assertInputNames(owner, MOSS_DIT_TEXT, graphs.ditText, Object.values(text));
  assertInput(owner, MOSS_DIT_TEXT, graphs.ditText, text.textEmbeds, "f32", [
    1,
    symbol,
    MOSS_TEXT_WIDTH,
  ]);
  for (const name of [text.ropeCos, text.ropeSin]) {
    assertInput(owner, MOSS_DIT_TEXT, graphs.ditText, name, "f32", [1, 1, symbol, dit.ropeWidth]);
  }
  const textOutput = soleOutput(owner, MOSS_DIT_TEXT, graphs.ditText, [1, symbol, dit.width]);
  assertAsset(owner, graphs.dit, MOSS_ROPE_BASE, MOSS_ROPE_BASE_ROLE);
  assertAsset(owner, graphs.dit, MOSS_TRAINING_GRID, MOSS_TRAINING_GRID_ROLE);
  return {
    latentChannels: dit.latentChannels,
    width: dit.width,
    ropeWidth: dit.ropeWidth,
    projWidth: dit.projWidth,
    outputs: { dit: dit.output, ditSkip22: skip.output, ditText: textOutput },
    borrowed: borrowedInitializers(owner, graphs.ditSkip22, graphs.dit),
  };
};

/**
 * Matches the base tables to the graph contract: the RoPE width, the `timesteps_proj` width, and
 * base tables that cover exactly the accepted limits (`./request.ts` — 4,096 cells per side and
 * 4,096 text tokens; a table of other extent would let the acceptance check pass a request the
 * tables cannot build, or reject one they can).
 */
export const mossDitContract = (
  graph: MossDitGraphContract,
  ropeBase: MossRopeBase,
  trainingGrid: MossTrainingGrid,
  owner: string,
): MossDitContract => {
  if (mossRopeWidth(ropeBase) !== graph.ropeWidth) {
    throw new Error(
      `${owner}: ${MOSS_ROPE_BASE} builds rows of ${mossRopeWidth(ropeBase)} (the graphs take ${graph.ropeWidth})`,
    );
  }
  if (ropeBase.maxSide !== MOSS_MAX_SIDE_CELLS || ropeBase.maxText !== MOSS_MAX_TEXT_TOKENS) {
    throw new Error(
      `${owner}: ${MOSS_ROPE_BASE} covers ${ropeBase.maxSide} cells per side and ${ropeBase.maxText} text tokens` +
        ` (the accepted ${MOSS_MAX_SIDE_CELLS} and ${MOSS_MAX_TEXT_TOKENS})`,
    );
  }
  if (trainingGrid.width !== graph.projWidth) {
    throw new Error(
      `${owner}: ${MOSS_TRAINING_GRID} rows are ${trainingGrid.width} wide (timesteps_proj takes ${graph.projWidth})`,
    );
  }
  return { ...graph, ropeBase, trainingGrid };
};

const assetBuffer = async (component: ModelComponent, name: string): Promise<ArrayBuffer> => {
  const reader = component.asset(name);
  const bytes = await reader.read(0, reader.length);
  // A copy: the reader's bytes may be a view of a whole part (the `AssetReader` contract).
  return bytes.slice().buffer;
};

/** Reads and parses the `dit` container's two assets. */
export const readMossDitAssets = async (
  dit: ModelComponent,
): Promise<{ readonly ropeBase: MossRopeBase; readonly trainingGrid: MossTrainingGrid }> => ({
  ropeBase: parseMossRopeBase(await assetBuffer(dit, MOSS_ROPE_BASE)),
  trainingGrid: parseMossTrainingGrid(await assetBuffer(dit, MOSS_TRAINING_GRID)),
});

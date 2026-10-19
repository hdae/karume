/**
 * The host side that runs Wan2.1's video VAE as **two chunk graphs** (`vae_decoder_first` /
 * `vae_decoder_next`) with a resident causal cache (ADR 0118 decision 2 and the addendum of
 * 2026-10-02). It covers up to the chunk sequence of one tile — tile planning, blending, pasting and
 * the clamp are the step 5 layer.
 *
 * ## Graph contract (the form written by recipe `tools/export-recipes/wan/vae_patch.py`)
 *
 * - Inputs: one latent frame `latent [C,1,t,t]` (the de-normalization `z / (1/std) + mean` is on the
 *   host — `latents.ts`) and the caches `cache_NN [Cin,2,h,w]` (NN in upstream's `feat_idx` order).
 * - Outputs: the pre-clamp frames (first `[Cs,1,8t,8t]`, next `[Cs,4,8t,8t]` — Cs is the exit channel
 *   count, read from the declaration ({@link WanVaeChunkLayout.sampleChannels}; RGB's 3 for Wan2.1))
 *   and the updated caches (**in the same order as the inputs** — output 1+k is the update of cache
 *   input k).
 * - first has no cache for upsample3d's `time_conv` (it does not run in the first chunk — upstream's
 *   `'Rep'`). That cache stays at the tile-head zeros and is read by the first run of next.
 *
 * The tile edge `t` and the cache shapes are not literals here; they are derived from the opened
 * asset's graph declarations ({@link wanVaeChunkLayout} — so a different tile edge needs only a new
 * distribution. Decision 2).
 *
 * ## One tile = one batch, read back only the frames (the procedure of the 2026-10-02 addendum)
 *
 * `Session.run` / `enqueueRead` read back every graph output (30–32 caches, about 0.56–0.58 GiB), so
 * they MUST NOT be used. The procedure is `beginBatch` → first with `enqueue({ copyOutputs })` → next
 * (chunk count − 1) times → `finishAndRead({ frame residents })`. The cache outputs are copied to the
 * same residents with `copyOutputs` (the residents bound as inputs are also the copy targets — the
 * copies are queued in FIFO after all dispatches of the run, so read-then-overwrite order is kept. ADR
 * 0054 decision 3). The frame residents are held separately per chunk (copies are queued in the span's
 * FIFO and the read-back runs only once at settlement, so copying to the same resident would keep only
 * the last chunk).
 *
 * ## Zeroing the caches (at the tile head)
 *
 * The cache inputs of first (being zero is what causal padding means) and the two of `time_conv`
 * (first does not write them, the first run of next reads them) **all** need to be zero — there is
 * nothing to narrow down. The runtime's `ResidentTensor` has no entry point for writing zeros on the
 * GPU side (`clearBuffer`), so zeros are written from the host with `ResidentTensor.write`
 * (`queue.writeBuffer`). With tile 32 that is 0.58 GiB of host writes per tile, and on the B570 (ReBAR)
 * the staging sits in the VRAM heap (the form ADR 0118 decision 2 wanted to avoid). Whether to add a
 * GPU-side zeroing entry point is a decision about the runtime's public surface (open).
 */

import type { BatchScope, GpuContext, ResidentTensor, Session, Tensor } from "@karume/runtime";
import type { GraphOwner } from "../hub/components.ts";

/** The latent input name (the recipe's forward argument name becomes the IR input name as is). */
export const WAN_VAE_LATENT_INPUT = "latent";

/** Cache input names (`cache_` + two digits of upstream's `feat_idx`). */
const CACHE_INPUT = /^cache_\d{2}$/;

/** The number of frames a cache carries over (upstream's `CACHE_T`). */
const CACHE_FRAMES = 2;

/** Frames produced by the first / next chunks (two temporal upsamples = 4×; the first gives 1). */
const FIRST_FRAMES = 1;
const NEXT_FRAMES = 4;

/** Bytes per element (every semantic dtype is 4 bytes — ADR 0009). */
const BYTES_PER_ELEMENT = 4;

/** A chunk graph declaration deviates from the contract (this module's doc). */
export class WanVaeChunkError extends Error {
  override readonly name = "WanVaeChunkError";
}

/** The inputs / outputs of one chunk graph (checked). */
export type WanVaeChunkGraph = {
  /** Cache input names (in graph input order). */
  readonly caches: readonly string[];
  /** The frame output name. */
  readonly frameOutput: string;
  /** Output names of the updated caches (the update of `caches[k]` is `cacheOutputs[k]`). */
  readonly cacheOutputs: readonly string[];
  /** The frame shape `[Cs, T, 8t, 8t]` (T is 1 for first, 4 for next). */
  readonly frameShape: readonly number[];
};

/** The contract of the two chunk graphs (values derived from the opened asset's graph declarations). */
export type WanVaeChunkLayout = {
  /** The latent channel count (axis 0 of `latent`). */
  readonly latentChannels: number;
  /** The edge of the latent tile (axes 2 / 3 of `latent`). */
  readonly tile: number;
  /** The edge of the output frame (`8t`). */
  readonly sampleTile: number;
  /**
   * The exit channel count (axis 0 of the frame output — the same for first and next). RGB's 3 for a
   * generation without unpatchify, `3·p²` for a generation that outputs patchify space (matched
   * against the descriptor by `assertWanVaeMatchesGeneration` in `tile-decode.ts`).
   */
  readonly sampleChannels: number;
  /** All caches (in next's input order, name → shape). */
  readonly cacheShapes: ReadonlyMap<string, readonly number[]>;
  readonly first: WanVaeChunkGraph;
  readonly next: WanVaeChunkGraph;
};

const staticShape = (dims: readonly (number | string)[], where: string): readonly number[] =>
  dims.map((dim, axis) => {
    if (typeof dim !== "number") {
      throw new WanVaeChunkError(
        `${where} axis ${axis} is not a static dimension (${String(dim)})`,
      );
    }
    return dim;
  });

const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((dim, axis) => dim === b[axis]);

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** The static shape of a graph value (an undeclared name fails loudly — looked up by `Object.hasOwn`). */
const valueShape = (owner: GraphOwner, name: string, where: string): readonly number[] => {
  if (!Object.hasOwn(owner.graph.values, name)) {
    throw new WanVaeChunkError(`${where}: value '${name}' is not declared`);
  }
  return staticShape(owner.graph.values[name].shape, `${where}'s '${name}'`);
};

/**
 * The declared dtype is f32. With shape and name alone, the bytes of an i32 frame read back would be
 * read as a Float32Array (integer 1 becomes a finite value of about 1.4e-45), pass the finiteness
 * check too, and an almost black video would come out silently. Not the same thing as the storage
 * dtype of quantized weights (this is the semantic dtype of inputs / outputs — ADR 0009).
 */
const assertF32Declared = (dtype: string, where: string, what: string): void => {
  if (dtype !== "f32") {
    throw new WanVaeChunkError(`${where}: ${what} has dtype ${dtype}, not f32`);
  }
};

/**
 * Checks one graph declaration and extracts its I/O contract (also returns the latent shape — every
 * I/O dtype is f32).
 */
const chunkGraph = (
  owner: GraphOwner,
  where: string,
  frames: number,
): { readonly graph: WanVaeChunkGraph; readonly latent: readonly number[] } => {
  const { inputs, outputs } = owner.graph;
  const [latentSpec, ...cacheSpecs] = inputs;
  if (latentSpec?.name !== WAN_VAE_LATENT_INPUT) {
    throw new WanVaeChunkError(`${where}: the first input is not '${WAN_VAE_LATENT_INPUT}'`);
  }
  assertF32Declared(latentSpec.dtype, where, `latent input '${WAN_VAE_LATENT_INPUT}'`);
  const latent = staticShape(latentSpec.shape, `${where}'s '${WAN_VAE_LATENT_INPUT}'`);
  const [channels, latentFrames, rows, cols] = latent;
  if (latent.length !== 4 || latentFrames !== 1 || rows !== cols || channels < 1) {
    throw new WanVaeChunkError(`${where}: the latent must be [C,1,t,t] (${latent.join(",")})`);
  }
  if (outputs.length !== inputs.length) {
    throw new WanVaeChunkError(
      `${where}: output count ${outputs.length} differs from input count ${inputs.length} (frame + cache)`,
    );
  }
  if (cacheSpecs.length === 0) {
    // Only the causal cache carries state between chunks (with zero caches chunks cannot be chained).
    throw new WanVaeChunkError(
      `${where}: no cache inputs (by convention, cache_NN chains the chunks)`,
    );
  }
  const [frameOutput, ...cacheOutputs] = outputs;
  const frameShape = valueShape(owner, frameOutput, `${where} frame output`);
  assertF32Declared(owner.graph.values[frameOutput].dtype, where, `frame output '${frameOutput}'`);
  const [sampleChannels, , side] = frameShape;
  if (
    !sameShape(frameShape, [sampleChannels, frames, side, side]) ||
    sampleChannels < 1 ||
    side % rows !== 0
  ) {
    throw new WanVaeChunkError(
      `${where}: frame shape [${frameShape.join(",")}] is not [C,${frames},8t,8t]`,
    );
  }
  cacheSpecs.forEach((spec, index) => {
    if (!CACHE_INPUT.test(spec.name)) {
      throw new WanVaeChunkError(`${where}: input '${spec.name}' is not spelled cache_NN`);
    }
    assertF32Declared(spec.dtype, where, `cache input '${spec.name}'`);
    const shape = staticShape(spec.shape, `${where}'s '${spec.name}'`);
    if (shape.length !== 4 || shape[1] !== CACHE_FRAMES) {
      throw new WanVaeChunkError(
        `${where}: '${spec.name}' must be [Cin,2,h,w] (${shape.join(",")})`,
      );
    }
    const output = cacheOutputs[index];
    const updated = valueShape(owner, output, `${where} output`);
    assertF32Declared(owner.graph.values[output].dtype, where, `updated cache output '${output}'`);
    if (!sameShape(shape, updated)) {
      throw new WanVaeChunkError(
        `${where}: output ${
          index + 1
        } ('${output}') shape differs from '${spec.name}' (by convention, outputs follow the input order)`,
      );
    }
  });
  return {
    graph: { caches: cacheSpecs.map((spec) => spec.name), frameOutput, cacheOutputs, frameShape },
    latent,
  };
};

/**
 * Derives and checks the chunk-sequence contract from the opened asset's first / next graph
 * declarations.
 *
 * MUST: the shapes are taken from the asset here (no tile edge or cache shape as a literal on the host
 * — decision 2). Checks that the two latent shapes and frame edges agree, and that first's caches are
 * a subsequence of next's caches (same names, same shapes, same order) (caches outside first are read
 * as zero by the first run of next).
 */
export const wanVaeChunkLayout = (first: GraphOwner, next: GraphOwner): WanVaeChunkLayout => {
  const firstGraph = chunkGraph(first, "vae_decoder_first", FIRST_FRAMES);
  const nextGraph = chunkGraph(next, "vae_decoder_next", NEXT_FRAMES);
  if (!sameShape(firstGraph.latent, nextGraph.latent)) {
    throw new WanVaeChunkError(
      `The latent shapes of the 2 graphs differ (first [${firstGraph.latent.join(",")}], next [${nextGraph.latent.join(
        ",",
      )}])`,
    );
  }
  const [sampleChannels, , sampleTile] = nextGraph.graph.frameShape;
  if (firstGraph.graph.frameShape[2] !== sampleTile) {
    throw new WanVaeChunkError("The frame side differs between first and next");
  }
  if (firstGraph.graph.frameShape[0] !== sampleChannels) {
    throw new WanVaeChunkError(
      `The frame channel count differs between first and next (first ${
        firstGraph.graph.frameShape[0]
      }, next ${sampleChannels})`,
    );
  }
  const cacheShapes = new Map<string, readonly number[]>(
    next.graph.inputs.slice(1).map((spec) => [spec.name, staticShape(spec.shape, spec.name)]),
  );
  let cursor = 0;
  for (const name of firstGraph.graph.caches) {
    const shape = cacheShapes.get(name);
    const firstShape = staticShape(
      first.graph.inputs.find((spec) => spec.name === name)?.shape ?? [],
      name,
    );
    if (shape === undefined || !sameShape(shape, firstShape)) {
      throw new WanVaeChunkError(`'${name}' of first is not in next with the same shape`);
    }
    const position = nextGraph.graph.caches.indexOf(name);
    if (position < cursor) {
      throw new WanVaeChunkError(`The cache order of first differs from next ('${name}')`);
    }
    cursor = position + 1;
  }
  const [latentChannels, , tile] = nextGraph.latent;
  return {
    latentChannels,
    tile,
    sampleTile,
    sampleChannels,
    cacheShapes,
    first: firstGraph.graph,
    next: nextGraph.graph,
  };
};

/** The output frame count for F chunks (`1 + 4(F−1)`). */
export const wanVaeFrameCount = (chunks: number): number =>
  FIRST_FRAMES + NEXT_FRAMES * (chunks - 1);

/**
 * The set of resident tensors the chunk sequence uses (all caches + the frames per chunk). Reused
 * across tiles.
 *
 * MUST: call {@link WanVaeChunkCaches.zero} and `dispose` only after the batches that use these
 * residents have settled (`ResidentTensor.write` can overtake an enqueue already issued — the
 * runtime's contract).
 */
export class WanVaeChunkCaches {
  readonly layout: WanVaeChunkLayout;
  readonly #gpu: GpuContext;
  readonly #caches: ReadonlyMap<string, ResidentTensor>;
  readonly #zeros: Float32Array<ArrayBuffer>;
  readonly #frames: ResidentTensor[] = [];

  private constructor(
    gpu: GpuContext,
    layout: WanVaeChunkLayout,
    caches: ReadonlyMap<string, ResidentTensor>,
    zeros: Float32Array<ArrayBuffer>,
  ) {
    this.#gpu = gpu;
    this.layout = layout;
    this.#caches = caches;
    this.#zeros = zeros;
  }

  /** Allocates the cache residents (a new WebGPU buffer starts as zeros). */
  static async create(gpu: GpuContext, layout: WanVaeChunkLayout): Promise<WanVaeChunkCaches> {
    // MUST: allocate the host zero array before the GPU residents — afterwards, if only the host array
    // failed with a RangeError after the residents (about 0.58 GiB) were allocated, nobody could
    // release the allocated residents (the caller has not received the return value, so it cannot
    // close them). Placed first, the failure point moves before touching the GPU and needs no cleanup.
    const largest = Math.max(...[...layout.cacheShapes.values()].map(elements));
    const zeros = new Float32Array(largest);
    const caches = new Map<string, ResidentTensor>();
    try {
      for (const [name, shape] of layout.cacheShapes) {
        caches.set(name, await gpu.createResident(elements(shape) * BYTES_PER_ELEMENT, name));
      }
    } catch (cause) {
      for (const resident of caches.values()) resident.dispose();
      throw cause;
    }
    return new WanVaeChunkCaches(gpu, layout, caches, zeros);
  }

  /** The total bytes of the cache residents. */
  get cacheBytes(): number {
    return [...this.#caches.values()].reduce((sum, resident) => sum + resident.byteLength, 0);
  }

  /** The total bytes of the frame residents (those allocated so far). */
  get frameBytes(): number {
    return this.#frames.reduce((sum, resident) => sum + resident.byteLength, 0);
  }

  /** Writes zeros to every cache (at the tile head — why nothing can be narrowed: the module doc). */
  zero(): void {
    for (const resident of this.#caches.values()) {
      resident.write(this.#zeros.subarray(0, resident.byteLength / BYTES_PER_ELEMENT));
    }
  }

  /** The cache inputs of that graph (name → resident). */
  inputs(graph: WanVaeChunkGraph): Record<string, ResidentTensor> {
    return Object.fromEntries(graph.caches.map((name) => [name, this.#cache(name)]));
  }

  /** The `copyOutputs` of that graph (updated caches → the same residents, frames → `frame`). */
  copyOutputs(graph: WanVaeChunkGraph, frame: ResidentTensor): Record<string, ResidentTensor> {
    const copies: Record<string, ResidentTensor> = { [graph.frameOutput]: frame };
    graph.caches.forEach((name, index) => {
      copies[graph.cacheOutputs[index]] = this.#cache(name);
    });
    return copies;
  }

  /** The frame residents of chunks 0..count−1 (allocated if missing — chunk 0 has first's shape). */
  async frames(count: number): Promise<readonly ResidentTensor[]> {
    while (this.#frames.length < count) {
      const index = this.#frames.length;
      const shape = index === 0 ? this.layout.first.frameShape : this.layout.next.frameShape;
      this.#frames.push(
        await this.#gpu.createResident(elements(shape) * BYTES_PER_ELEMENT, `frame ${index}`),
      );
    }
    return this.#frames.slice(0, count);
  }

  /**
   * Returns every resident (may be called before the Sessions — the runtime refuses if a baked-in
   * reference remains).
   */
  dispose(): void {
    for (const resident of [...this.#caches.values(), ...this.#frames]) resident.dispose();
  }

  #cache(name: string): ResidentTensor {
    const resident = this.#caches.get(name);
    if (resident === undefined) throw new WanVaeChunkError(`Cache '${name}' is not resident`);
    return resident;
  }
}

/** The chunk count F of a latent `[C,F,t,t]` (fails loudly if the length does not fit). */
export const wanVaeChunkCount = (
  layout: WanVaeChunkLayout,
  latents: Float32Array<ArrayBuffer>,
): number => {
  const plane = layout.latentChannels * layout.tile * layout.tile;
  const chunks = latents.length / plane;
  if (!Number.isInteger(chunks) || chunks < 1) {
    throw new WanVaeChunkError(
      `Latent element count ${latents.length} does not match [${layout.latentChannels},F,${layout.tile},${layout.tile}]`,
    );
  }
  return chunks;
};

/** Cuts the single frame `[C,1,t,t]` of chunk `index` out of the latent `[C,F,t,t]` (a copy). */
export const wanVaeLatentChunk = (
  layout: WanVaeChunkLayout,
  latents: Float32Array<ArrayBuffer>,
  index: number,
): Tensor => {
  const chunks = wanVaeChunkCount(layout, latents);
  // Letting an out-of-range index through would read the neighbouring channel's plane or past the
  // end (an empty subarray) and silently return values mixed with zeros.
  if (!Number.isInteger(index) || index < 0 || index >= chunks) {
    throw new RangeError(`chunk ${index} is outside [0, ${chunks})`);
  }
  const area = layout.tile * layout.tile;
  const data = new Float32Array(layout.latentChannels * area);
  for (let channel = 0; channel < layout.latentChannels; channel += 1) {
    const from = (channel * chunks + index) * area;
    data.set(latents.subarray(from, from + area), channel * area);
  }
  return { dtype: "f32", shape: [layout.latentChannels, 1, layout.tile, layout.tile], data };
};

/**
 * Queues one chunk on the batch (the caches are bound as resident inputs; the updated caches and the
 * frame go through `copyOutputs`).
 *
 * `graph` and `session` are passed as a pair (the first chunk takes first's declaration and Session).
 */
export const enqueueWanVaeChunk = (
  batch: BatchScope,
  session: Session,
  graph: WanVaeChunkGraph,
  caches: WanVaeChunkCaches,
  latent: Tensor,
  frame: ResidentTensor,
): Promise<void> =>
  session.enqueue(
    { [WAN_VAE_LATENT_INPUT]: latent, ...caches.inputs(graph) },
    { batch, copyOutputs: caches.copyOutputs(graph, frame) },
  );

/**
 * Concatenates the chunks' frames (`[Cs,T_k,s,s]` in chunk order) along the time axis into
 * `[Cs, 1 + 4(F−1), s, s]` (Cs is {@link WanVaeChunkLayout.sampleChannels}).
 */
export const concatWanVaeFrames = (
  layout: WanVaeChunkLayout,
  chunks: readonly ArrayBuffer[],
): Float32Array<ArrayBuffer> => {
  const { sampleChannels } = layout;
  const plane = layout.sampleTile * layout.sampleTile;
  const total = wanVaeFrameCount(chunks.length);
  const out = new Float32Array(sampleChannels * total * plane);
  let offset = 0;
  chunks.forEach((buffer, index) => {
    const frames = index === 0 ? FIRST_FRAMES : NEXT_FRAMES;
    const values = new Float32Array(buffer);
    if (values.length !== sampleChannels * frames * plane) {
      throw new WanVaeChunkError(
        `chunk ${index} frame element count ${values.length} does not match`,
      );
    }
    for (let channel = 0; channel < sampleChannels; channel += 1) {
      const from = channel * frames * plane;
      out.set(values.subarray(from, from + frames * plane), (channel * total + offset) * plane);
    }
    offset += frames;
  });
  return out;
};

/**
 * Runs the chunk sequence of one tile (one tile = one batch, read back only the frames — module doc).
 *
 * `latents` is the de-normalized `[C,F,t,t]`. Returns the **pre-clamp** `[Cs, 1 + 4(F−1), 8t, 8t]`.
 * Writes zeros to the caches at the tile head ({@link WanVaeChunkCaches.zero}).
 *
 * MUST: do not issue another batch or run on the same device concurrently (the batch holds the
 * device's span lock).
 */
export const decodeWanVaeTile = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  latents: Float32Array<ArrayBuffer>,
): Promise<Float32Array<ArrayBuffer>> => {
  const { layout } = caches;
  const chunks = wanVaeChunkCount(layout, latents);
  const frames = await caches.frames(chunks);
  caches.zero();
  const batch = await gpu.beginBatch();
  try {
    for (let index = 0; index < chunks; index += 1) {
      const head = index === 0;
      await enqueueWanVaeChunk(
        batch,
        head ? sessions.first : sessions.next,
        head ? layout.first : layout.next,
        caches,
        wanVaeLatentChunk(layout, latents, index),
        frames[index],
      );
    }
  } catch (cause) {
    // MUST: close the span and return the lock (otherwise the next batch / run on the same device waits
    // forever). The main failure is rethrown to the caller (the settlement on the finish side is a copy
    // of the same failure, so it is discarded).
    await batch.finish().catch(() => undefined);
    throw cause;
  }
  const names = frames.map((_, index) => `frame${index}`);
  const read = await batch.finishAndRead(
    Object.fromEntries(frames.map((resident, index) => [names[index], resident])),
  );
  return concatWanVaeFrames(
    layout,
    names.map((name) => read[name]),
  );
};

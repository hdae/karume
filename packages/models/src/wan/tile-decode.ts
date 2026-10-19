/**
 * Wan's VAE stage — the **always-tiled** decode of the two chunk graphs (first / next) (ADR 0118
 * decision 2 — `vae-tiles.ts`), and the family admission checks of the VAE graph declarations ×
 * the generation descriptor ({@link assertWanVaeMatchesGeneration}, {@link assertWanVaeTilesCover}).
 *
 * The stage order and Session lifetimes (text → DiT → VAE, each created and closed one stage at a
 * time) are defined by the doc at the top of `./pipeline.ts`; this module only runs the one VAE stage.
 * Shared by the Wan2.1 / 2.2 classes (ADR 0121 decision 10).
 * Per-generation values (accepted set, de-normalization statistics, unpatchify factor) are received
 * as arguments from the generation descriptor (`./descriptor.ts`). The latent and exit channel
 * counts, the tile edge and the graph's input / output spatial ratio are read from the graph
 * declarations (`wanVaeChunkLayout`). `owner` is the message prefix (`"WanPipeline"` for Wan2.1,
 * `"WanTi2vPipeline"` for Wan2.2).
 *
 * NOTE: the public types (`WanGenerateEvent` / `WanRunComponent`) are taken from `./pipeline.ts` with
 * `import type` (type-only references are erased, so there is no import cycle).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type { GpuContext, Session, SessionDiagnostics } from "@karume/runtime";

import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import type { ModelComponent } from "../hub/components.ts";
import type { WanGenerateEvent, WanRunComponent } from "./pipeline.ts";
import type { PlanLayout, WanGenerationKnobs } from "./plan.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";
import { firstNonFinite } from "./graph-io.ts";
import { denormalizeWanLatents } from "./latents.ts";
import { WanVaeChunkCaches, type WanVaeChunkLayout } from "./vae-chunks.ts";
import {
  clampWanVaeFrames,
  decodeWanVaeTiled,
  planWanVaeTiles,
  unpatchifyWanVaeFrames,
  wanVaeMinTileOverlap,
  wanVaeSpatialCompression,
  wanVaeTileCount,
  type WanVaeTilePlan,
} from "./vae-tiles.ts";

/** Component keys (the series' graph names = the weights names of the step 7 manifest). */
export const VAE_DECODER_FIRST = "vae_decoder_first";
export const VAE_DECODER_NEXT = "vae_decoder_next";

/**
 * The channel count of the final output (RGB). The VAE graph's exit is `WAN_RGB_CHANNELS ·
 * vaePatchSize²` (matched by {@link assertWanVaeMatchesGeneration} — for a generation without
 * unpatchify, exit = RGB).
 */
const WAN_RGB_CHANNELS = 3;

/**
 * The graph's input / output spatial ratio (the chunk graph declaration's `sampleTile / tile`). The
 * compression ({@link wanSpatialCompression}) and the overlap ({@link planWanGenerationTiles}) take
 * the ratio from this one function (the derivation of the input is not split in two).
 */
const graphScale = (layout: PlanLayout): number => layout.sampleTile / layout.tile;

/**
 * The spatial compression of a generation request (final output pixels per latent) = the graph's
 * input / output spatial ratio (`layout.sampleTile / layout.tile`) × the generation's unpatchify factor
 * (`generation.vaePatchSize`). Fails loudly unless it is a positive integer
 * (`wanVaeSpatialCompression`).
 *
 * MUST: the latent size (`planWanRequest` in `plan.ts`), the check that tiles cover it
 * ({@link assertWanVaeTilesCover}) and the overlap ({@link planWanGenerationTiles}) use the same single
 * compression (ADR 0121 decision 6 — if even one place divides by the graph's ratio alone, the latent
 * size and the tile plan disagree for a generation with unpatchify).
 *
 * NOTE: `export` is for the shared stages and for tests that pin the checks without a GPU (not on the
 * `mod.ts` / subpath surface — ADR 0008).
 */
export const wanSpatialCompression = (
  layout: PlanLayout,
  generation: WanGenerationDescriptor,
): number => wanVaeSpatialCompression(graphScale(layout), generation.vaePatchSize);

/**
 * The tile plan for a generation request's latent `height × width` (the single production entry
 * point). The overlap is derived by dividing 64 px of output by the spatial compression
 * (`wanVaeMinTileOverlap` — fails loudly if it does not divide evenly).
 *
 * NOTE: `export` is so that the plan check, the admission check and the GPU comparison
 * (`e2e_wan_vae_tiles_test.ts`) go through the production derivation (not on the `mod.ts` / subpath
 * surface — ADR 0008).
 */
export const planWanGenerationTiles = (
  layout: PlanLayout,
  height: number,
  width: number,
  generation: WanGenerationDescriptor,
): WanVaeTilePlan =>
  planWanVaeTiles(
    layout,
    height,
    width,
    wanVaeMinTileOverlap(graphScale(layout), generation.vaePatchSize),
  );

/**
 * Checks that the VAE graph declarations agree with the generation descriptor (a family admission
 * check). The number of de-normalization statistics = the latent channel count; the exit channel
 * count = RGB × `vaePatchSize²`.
 *
 * MUST: call in admission. Even when the number of statistics differs from the latent channel count,
 * de-normalization (`denormalizeWanLatents`) passes silently if the latent's element count is divisible
 * by the number of statistics. An asset whose exit channel count does not match the unpatchify factor
 * would not fail until the shape check after the tiled decode ({@link finishWanVaeFrames}) (after
 * paying for the whole DiT stage).
 *
 * NOTE: `vaePatchSize` could also be derived from the exit channel count, but it is held in the
 * descriptor as an upstream fact and used here for matching (ADR 0121 decision 6). `export` is for
 * tests that pin the checks without a GPU (not on the `mod.ts` / subpath surface — ADR 0008).
 */
export const assertWanVaeMatchesGeneration = (
  layout: Pick<WanVaeChunkLayout, "latentChannels" | "sampleChannels">,
  generation: WanGenerationDescriptor,
  owner: string,
): void => {
  const { mean, std } = generation.latents;
  if (mean.length !== layout.latentChannels || std.length !== layout.latentChannels) {
    throw new Error(
      `${owner}: the denormalization statistics (mean ${mean.length}, std ${std.length}) do not match the VAE latent's ` +
        `${layout.latentChannels} channels`,
    );
  }
  const sampleChannels = WAN_RGB_CHANNELS * generation.vaePatchSize ** 2;
  if (layout.sampleChannels !== sampleChannels) {
    throw new Error(
      `${owner}: VAE output ${layout.sampleChannels} channels differ from RGB ${WAN_RGB_CHANNELS} × ` +
        `unpatchify ${generation.vaePatchSize}² = ${sampleChannels}`,
    );
  }
};

/**
 * Checks that, with the VAE chunk graphs' geometry, every accepted size (`generation.acceptedSizes`)
 * can be covered by tiles (a family admission check — the sizes are finite, so every one is planned).
 *
 * MUST: call in admission. The tile edge can be changed by the distribution alone (ADR 0118 decision
 * 2), so an asset with tiles larger than the latent's short side (for example 64 > 60) or tiles that
 * cannot meet the overlap lower bound (for example 8) still passes the chunk graph check
 * (`wanVaeChunkLayout`). Unless it fails here, it fails for the first time in the VAE stage, after
 * paying for the whole DiT stage.
 *
 * MUST: the overlap ({@link planWanGenerationTiles}) is computed **after** the "is the latent an
 * integer" check, inside the `try` — placed before, an asset whose scale makes the latent non-integer
 * would first show the message that 64 px does not divide evenly.
 */
export const assertWanVaeTilesCover = (
  layout: PlanLayout,
  generation: WanGenerationDescriptor,
  owner: string,
): void => {
  const compression = wanSpatialCompression(layout, generation);
  for (const { width, height } of generation.acceptedSizes) {
    const latentHeight = height / compression;
    const latentWidth = width / compression;
    if (!Number.isInteger(latentHeight) || !Number.isInteger(latentWidth)) {
      throw new Error(
        `${owner}: with spatial compression ${compression} (VAE scale ${layout.sampleTile} / ${layout.tile} × ` +
          `unpatchify ${generation.vaePatchSize}), the latent of ${width}×${height} is not integral`,
      );
    }
    try {
      planWanGenerationTiles(layout, latentHeight, latentWidth, generation);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(
        `${owner}: the VAE tile (latent ${layout.tile}) cannot tile-decode ${width}×${height}` +
          ` (latent ${latentHeight}×${latentWidth}) — ${reason}`,
        { cause: error },
      );
    }
  }
};

/**
 * The built material the VAE stage reads (this stage's part of the pipeline's internal state — taken
 * structurally).
 */
type WanVaeDecodeState = {
  readonly gpu: GpuContext;
  readonly vaeFirst: ModelComponent;
  readonly vaeNext: ModelComponent;
  readonly layout: WanVaeChunkLayout;
  readonly onRunDiagnostics?: (component: WanRunComponent, diagnostics: SessionDiagnostics) => void;
};

/**
 * Turns the pasted frames in patchify space (pre-clamp, `[Cs, F', h, w]`) into the final frames
 * `[3, F, H, W]` (clamped): shape check → unpatchify (the generation's `vaePatchSize`) → finiteness
 * (before the clamp, named in RGB coordinates) → clamp. The order is the same as upstream's `_decode` /
 * `tiled_decode` (clamp after unpatchify).
 *
 * The shape is checked per dimension: the channels `Cs`, height `h` and width `w` are derived from the
 * tile plan (the exit channel count and latent × scale), and the frame count `F'` from the length
 * (fails unless it divides by `Cs·h·w`). The unpatchified shape is compared with
 * `[3, plan.frames, plan.height, plan.width]` (by element count alone, patchify space's
 * `3·p²·(H/p)·(W/p)` equals `3·H·W`, and whether unpatchify happened could not be told apart).
 *
 * MUST: check finiteness **before** the clamp. The clamp, the same as upstream's
 * ({@link clampWanVaeFrames}), turns ±Inf into ±1, so it cannot be detected afterwards (the clamp
 * itself is a copy of upstream and is not changed).
 *
 * NOTE: a generation without unpatchify (`vaePatchSize` 1) clamps `assembled` itself in place and
 * returns it (no copy). `export` is for tests that pin the tail without a GPU (not on the `mod.ts` /
 * subpath surface — ADR 0008).
 */
export const finishWanVaeFrames = (
  assembled: Float32Array<ArrayBuffer>,
  plan: Pick<WanGenerationKnobs, "frames" | "width" | "height" | "tiles">,
  generation: Pick<WanGenerationDescriptor, "vaePatchSize">,
  owner: string,
): Float32Array<ArrayBuffer> => {
  const { tiles } = plan;
  const patchSize = generation.vaePatchSize;
  const channels = tiles.sampleChannels;
  const height = tiles.rows.extent * tiles.scale;
  const width = tiles.cols.extent * tiles.scale;
  const perFrame = channels * height * width;
  if (assembled.length === 0) {
    throw new Error(`${owner}: the VAE output has 0 elements (no frames at all)`);
  }
  const frames = assembled.length / perFrame;
  if (!Number.isInteger(frames)) {
    throw new Error(
      `${owner}: the VAE output's ${assembled.length} elements do not form [${channels}, F, ${height}, ${width}]` +
        ` (not divisible by ${perFrame} elements per frame)`,
    );
  }
  const area = patchSize ** 2;
  if (channels % area !== 0) {
    throw new Error(
      `${owner}: VAE output ${channels} channels are not divisible by the unpatchify patch size ${patchSize}² = ${area}`,
    );
  }
  const patchShape = [channels, frames, height, width] as const;
  const rgbShape = [channels / area, frames, height * patchSize, width * patchSize];
  const expected = [WAN_RGB_CHANNELS, plan.frames, plan.height, plan.width];
  if (rgbShape.some((dim, axis) => dim !== expected[axis])) {
    throw new Error(
      `${owner}: VAE output shape [${patchShape.join(", ")}] (unpatchify ${patchSize} gives ` +
        `[${rgbShape.join(", ")}]) differs from [${expected.join(", ")}]`,
    );
  }
  const video = unpatchifyWanVaeFrames(assembled, patchShape, patchSize);
  const broken = firstNonFinite(video);
  if (broken !== -1) {
    const plane = plan.height * plan.width;
    const pixel = broken % plane;
    const sheet = Math.floor(broken / plane);
    throw new Error(
      `${owner}: VAE output (before clamping) channel ${Math.floor(sheet / plan.frames)}, ` +
        `frame ${sheet % plan.frames}, pixel (x=${pixel % plan.width}, ` +
        `y=${Math.floor(pixel / plan.width)}) is non-finite (${video[broken]})`,
    );
  }
  clampWanVaeFrames(video);
  return video;
};

/**
 * The VAE stage (creates the Sessions of the two chunk graphs and the resident caches, runs the tiled
 * decode and closes them). The final shape check, unpatchify, finiteness and clamp are in
 * {@link finishWanVaeFrames}.
 */
export const decodeWanVaeStage = async (
  state: WanVaeDecodeState,
  plan: WanGenerationKnobs,
  latents: Float32Array<ArrayBuffer>,
  generation: WanGenerationDescriptor,
  emit: (event: WanGenerateEvent) => Promise<void>,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<Float32Array<ArrayBuffer>> => {
  const { layout } = state;
  const tilePlan = plan.tiles;
  const tiles = wanVaeTileCount(tilePlan);
  const denormalized = denormalizeWanLatents(latents, generation.latents);
  const observe = state.onRunDiagnostics;

  await emit({ kind: "stage", component: "vae_decoder", at: "start" });
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let frames: Float32Array<ArrayBuffer>;
  let failure: { readonly error: unknown } | undefined;
  try {
    // The VAE does not take the quant's session (`{}` — the same contract as anima's `withStage`.
    // `sessionOptions` are the effective settings the family admission derived from the quant
    // declaration through the acceptance table, passed to the DiT Session).
    first = await state.vaeFirst.createSession(state.gpu, {});
    next = await state.vaeNext.createSession(state.gpu, {});
    caches = await WanVaeChunkCaches.create(state.gpu, layout);
    // Before the first tile (a cancellation that arrived during the start notification or while
    // allocating resources takes effect here — as in the text / encoder / DiT stages, checked after
    // acquiring resources and before the first GPU work. The check between tiles is the callback
    // below).
    await settleAbort(signal);
    const sessions = { first, next };
    frames = await decodeWanVaeTiled(
      state.gpu,
      sessions,
      caches,
      tilePlan,
      denormalized,
      async (tile) => {
        observe?.("vae_decoder_first", sessions.first.diagnostics());
        observe?.("vae_decoder_next", sessions.next.diagnostics());
        await emit({ kind: "vae-tile", tile, tiles });
        // Between tiles (one tile = one batch, indivisible — cancellation takes effect before the
        // next tile).
        await settleAbort(signal);
      },
    );
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: close Sessions → residents in that order (the same as the step 4 / 5 e2e). A failure to
    // close does not overwrite the main failure.
    const opened = { first, next, caches };
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => opened.first?.dispose(),
      () => opened.next?.dispose(),
      () => opened.caches?.dispose(),
    ]);
  }
  const video = finishWanVaeFrames(frames, plan, generation, owner);
  await emit({ kind: "stage", component: "vae_decoder", at: "end" });
  return video;
};

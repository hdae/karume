/**
 * What the e2e of the Wan video VAE's chunk sequence (2.1's `e2e_wan_vae_tiles_test.ts`, 2.2's
 * `e2e_wan_ti2v_vae_tiles_test.ts`) share: the fixture reader, a chunk loop with a fault-injection
 * entry, the VRAM breakdown and bit equality.
 *
 * The product path is `decodeWanVaeTile` of `src/wan/vae-chunks.ts`. The loop here is built from
 * the same parts, and each e2e checks that without a fault it matches the product path bit for bit
 * as Uint32 (the guarantee that the injected results represent the product path). Both generations
 * use the same loop so that the injection's shape is not grown per generation with one side
 * silently drifting.
 */

import { readFile } from "@karume/test-support/fs";
import { assert } from "@std/assert";
import {
  type GpuContext,
  parseSafetensors,
  type ResidentTensor,
  type Session,
  type SessionDiagnostics,
} from "@karume/runtime";
import {
  concatWanVaeFrames,
  enqueueWanVaeChunk,
  type WanVaeChunkCaches,
  wanVaeChunkCount,
  wanVaeLatentChunk,
} from "../../src/wan/vae-chunks.ts";

/**
 * The chunk-sequence fixture (the denormalized latent `[z,F,t,t]`, upstream's output before the
 * clamp, and the file's metadata — the writer is `tools/export-recipes/wan/export_vae.py`).
 */
export type Fixture = {
  readonly latents: Float32Array<ArrayBuffer>;
  readonly frames: Float32Array<ArrayBuffer>;
  readonly latentShape: readonly number[];
  readonly frameShape: readonly number[];
  readonly metadata: ReadonlyMap<string, string>;
};

export const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await readFile(url);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string): Float32Array<ArrayBuffer> => {
    const tensor = file.tensors.get(key);
    assert(tensor !== undefined && tensor.dtype === "F32", `${url.pathname}: '${key}' is not F32`);
    return new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4);
  };
  const frames = file.tensors.get("frames");
  assert(frames !== undefined, `${url.pathname}: 'frames' is missing`);
  const latents = file.tensors.get("latents");
  assert(latents !== undefined, `${url.pathname}: 'latents' is missing`);
  return {
    latents: view("latents"),
    frames: view("frames"),
    latentShape: latents.shape,
    frameShape: frames.shape,
    metadata: file.metadata,
  };
};

/** The kind of injected fault (none = the same as the product path). */
export type Fault =
  | { readonly kind: "none" }
  | { readonly kind: "skip-zero" }
  | { readonly kind: "drop-cache"; readonly cache: string }
  | { readonly kind: "first-for-all" }
  | { readonly kind: "swap-cache-frames" };

/**
 * A chunk loop built from the same parts as the product path (`decodeWanVaeTile`) (the
 * fault-injection entry).
 *
 * Only `swap-cache-frames` closes the batch after the first chunk, swaps the cache's 2 frames on
 * the host, then runs the rest in another batch (a batch boundary does not change values — the
 * no-fault equality check looks at the single-batch form).
 */
export const decodeWithFault = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  latents: Float32Array<ArrayBuffer>,
  fault: Fault,
): Promise<Float32Array<ArrayBuffer>> => {
  const { layout } = caches;
  const chunks = wanVaeChunkCount(layout, latents);
  const frames = await caches.frames(chunks);
  if (fault.kind !== "skip-zero") caches.zero();
  const read: Record<string, ArrayBuffer> = {};
  const runRange = async (from: number, to: number): Promise<void> => {
    const batch = await gpu.beginBatch();
    try {
      for (let index = from; index < to; index += 1) {
        const head = index === 0 || fault.kind === "first-for-all";
        const graph = head ? layout.first : layout.next;
        const session = head ? sessions.first : sessions.next;
        const latent = wanVaeLatentChunk(layout, latents, index);
        if (fault.kind === "drop-cache" && !head) {
          const copyOutputs = caches.copyOutputs(graph, frames[index]);
          const output = graph.cacheOutputs[graph.caches.indexOf(fault.cache)];
          delete copyOutputs[output];
          await session.enqueue({ latent, ...caches.inputs(graph) }, { batch, copyOutputs });
        } else {
          await enqueueWanVaeChunk(batch, session, graph, caches, latent, frames[index]);
        }
      }
    } catch (cause) {
      await batch.finish().catch(() => undefined);
      throw cause;
    }
    const range = frames.slice(from, to);
    Object.assign(
      read,
      await batch.finishAndRead(
        Object.fromEntries(range.map((resident, offset) => [`frame${from + offset}`, resident])),
      ),
    );
  };
  if (fault.kind === "swap-cache-frames") {
    await runRange(0, 1);
    for (const [name, resident] of Object.entries(caches.inputs(layout.next))) {
      const shape = layout.cacheShapes.get(name);
      assert(shape !== undefined, `cache '${name}' has no shape in the layout`);
      await swapCacheFrames(resident, shape[2] * shape[3]);
    }
    await runRange(1, chunks);
  } else {
    await runRange(0, chunks);
  }
  return concatWanVaeFrames(
    layout,
    frames.map((_, index) => read[`frame${index}`]),
  );
};

/**
 * Swaps the order of the 2 frames of a cache `[C,2,h,w]` on the host (fault injection). It is
 * row-major, so channel c's 2 frames are a contiguous `2·plane` (plane = h·w).
 */
const swapCacheFrames = async (resident: ResidentTensor, plane: number): Promise<void> => {
  const values = new Float32Array(await resident.read());
  const swapped = new Float32Array(values.length);
  for (let offset = 0; offset < values.length; offset += 2 * plane) {
    swapped.set(values.subarray(offset + plane, offset + 2 * plane), offset);
    swapped.set(values.subarray(offset, offset + plane), offset + plane);
  }
  resident.write(swapped);
};

/** The VRAM breakdown (from the diagnostics — the sum of the live allocations). */
export const vramBreakdown = (
  first: SessionDiagnostics,
  next: SessionDiagnostics,
  caches: WanVaeChunkCaches,
): Record<string, number> => {
  const session = (diagnostics: SessionDiagnostics): number =>
    diagnostics.weights.allocatedBytes +
    diagnostics.planBacking.residentBytes +
    diagnostics.planBacking.inputBytes;
  const parts = {
    firstWeights: first.weights.allocatedBytes,
    firstBacking: first.planBacking.residentBytes + first.planBacking.inputBytes,
    nextWeights: next.weights.allocatedBytes,
    nextBacking: next.planBacking.residentBytes + next.planBacking.inputBytes,
    caches: caches.cacheBytes,
    frames: caches.frameBytes,
    // finishAndRead's staging (the total of the frames — alive only while it settles).
    readbackStaging: caches.frameBytes,
  };
  return {
    ...parts,
    total: session(first) + session(next) + 2 * caches.frameBytes + caches.cacheBytes,
  };
};

export const bitsEqual = (a: Float32Array, b: Float32Array): boolean => {
  const left = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const right = new Uint32Array(b.buffer, b.byteOffset, b.length);
  return left.length === right.length && left.every((value, index) => value === right[index]);
};

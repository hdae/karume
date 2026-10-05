/**
 * Wan の動画 VAE の chunk 列の e2e（2.1 の `e2e_wan_vae_chunks_test.ts`・2.2 の
 * `e2e_wan_ti2v_vae_chunks_test.ts`）が共有するもの: フィクスチャの読み口・故障注入の口を持つ chunk
 * ループ・VRAM の内訳・ビット一致。
 *
 * 製品の経路は `src/wan/vae-chunks.ts` の `decodeWanVaeTile`。ここのループはそれと同じ部品で組み、
 * 故障なしなら製品の経路と Uint32 で一致することを各 e2e が門にする（注入の結果が製品の経路を
 * 代表している担保）。2 つの世代で同じループを使うのは、注入の形を世代ごとに育てて片方だけ
 * 黙ってずれる形を作らないため。
 */

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
 * chunk 列のフィクスチャ（逆正規化済みの潜在 `[z,F,t,t]` と上流のクランプ前の出力・ファイルのメタ —
 * 書き手は `tools/export-recipes/wan/export_vae.py`）。
 */
export type Fixture = {
  readonly latents: Float32Array<ArrayBuffer>;
  readonly frames: Float32Array<ArrayBuffer>;
  readonly latentShape: readonly number[];
  readonly frameShape: readonly number[];
  readonly metadata: ReadonlyMap<string, string>;
};

export const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await Deno.readFile(url);
  const file = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string): Float32Array<ArrayBuffer> => {
    const tensor = file.tensors.get(key);
    assert(
      tensor !== undefined && tensor.dtype === "F32",
      `${url.pathname}: '${key}' が F32 で無い`,
    );
    return new Float32Array(file.buffer, tensor.byteOffset, tensor.byteLength / 4);
  };
  const frames = file.tensors.get("frames");
  assert(frames !== undefined, `${url.pathname}: 'frames' が無い`);
  const latents = file.tensors.get("latents");
  assert(latents !== undefined, `${url.pathname}: 'latents' が無い`);
  return {
    latents: view("latents"),
    frames: view("frames"),
    latentShape: latents.shape,
    frameShape: frames.shape,
    metadata: file.metadata,
  };
};

/** 故障注入の種類（無し = 製品の経路と同じ）。 */
export type Fault =
  | { readonly kind: "none" }
  | { readonly kind: "skip-zero" }
  | { readonly kind: "drop-cache"; readonly cache: string }
  | { readonly kind: "first-for-all" }
  | { readonly kind: "swap-cache-frames" };

/**
 * 製品の経路（`decodeWanVaeTile`）と同じ部品で組んだ chunk ループ（故障注入の口）。
 *
 * `swap-cache-frames` だけは first の後で区間を閉じ、cache の 2 フレームをホストで入れ替えてから
 * 残りを別の区間で回す（区間の切れ目は値を変えない — 故障なしの一致の門は 1 区間の形で見る）。
 */
export const decodeWithFault = async (
  gpu: GpuContext,
  sessions: { readonly first: Session; readonly next: Session },
  caches: WanVaeChunkCaches,
  latents: Float32Array,
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
          await session.enqueue(
            { latent, ...caches.inputs(graph) },
            { batch, copyOutputs },
          );
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
      assert(shape !== undefined, `cache '${name}' の形が layout に無い`);
      await swapCacheFrames(resident, shape[2] * shape[3]);
    }
    await runRange(1, chunks);
  } else {
    await runRange(0, chunks);
  }
  return concatWanVaeFrames(layout, frames.map((_, index) => read[`frame${index}`]));
};

/**
 * cache `[C,2,h,w]` の 2 フレームの順をホストで入れ替える（故障注入）。行優先なのでチャネル c の
 * 2 フレームは連続した `2·plane`（plane = h·w）。
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

/** VRAM の内訳（診断から — 生きている確保の和）。 */
export const vramBreakdown = (
  first: SessionDiagnostics,
  next: SessionDiagnostics,
  caches: WanVaeChunkCaches,
): Record<string, number> => {
  const session = (diagnostics: SessionDiagnostics): number =>
    diagnostics.weights.allocatedBytes + diagnostics.planBacking.residentBytes +
    diagnostics.planBacking.inputBytes;
  const parts = {
    firstWeights: first.weights.allocatedBytes,
    firstBacking: first.planBacking.residentBytes + first.planBacking.inputBytes,
    nextWeights: next.weights.allocatedBytes,
    nextBacking: next.planBacking.residentBytes + next.planBacking.inputBytes,
    caches: caches.cacheBytes,
    frames: caches.frameBytes,
    // finishAndRead の staging（フレームの合計 — 決着の間だけ生きる）。
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

// The host side of the Wan video VAE's two chunk graphs (`src/wan/vae-chunks.ts`, shared by Wan2.1 and
// Wan2.2): the check that derives the chunk-sequence contract from the graph declarations, cutting the
// latent into chunks, concatenating the frames, and the cache residents with their tile-head zeroing
// (`ResidentTensor.clear` on mock residents, and across two tiles on stub chunk graphs on a real GPU).
//
// The chunk sequence on a real GPU with real assets is checked by `e2e_wan_vae_tiles_test.ts` and
// `e2e_wan_ti2v_vae_tiles_test.ts`. The e2e tests only pass well-formed assets, so the declaration's
// failure shapes (the branches that reject an asset off the contract by name) are checked only here.
// Declarations follow `helpers/stub-model.ts`: write one correct form, then break one point.

import { assert, assertEquals, assertRejects, assertStrictEquals, assertThrows } from "@std/assert";
import { acquireGpu, prepareContainer, type ResidentTensor } from "@karume/runtime";
import { describe, it } from "vite-plus/test";
import {
  concatWanVaeFrames,
  decodeWanVaeTile,
  WanVaeChunkCaches,
  wanVaeChunkCount,
  WanVaeChunkError,
  wanVaeChunkLayout,
  wanVaeFrameCount,
  wanVaeLatentChunk,
} from "../src/wan/vae-chunks.ts";
import type { GraphOwner } from "../src/hub/components.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  openGraphModel,
} from "../../runtime/tests/helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { decodeWithFault } from "./helpers/wan-vae-chunk-loop.ts";

/** The latent tile edge, the latent channel count and the output frame edge (`8t`). */
const TILE = 4;
const CHANNELS = 2;
const SIDE = 8 * TILE;

/** The declaration's dtype (f32 when omitted). */
type Dtype = GraphOwner["graph"]["inputs"][number]["dtype"];

/** One value (input / output) of one chunk graph's declaration. */
type ChunkValue = {
  readonly name: string;
  readonly shape: readonly StubDim[];
  readonly dtype?: Dtype;
};

/**
 * One chunk graph's declaration (input order = latent → caches; output order = frame → updated
 * caches).
 */
type ChunkSpec = {
  readonly inputs: readonly ChunkValue[];
  readonly outputs: readonly ChunkValue[];
};

const owner = (spec: ChunkSpec): GraphOwner => {
  const stub = stubModel({
    inputs: spec.inputs,
    outputs: spec.outputs.map((output) => output.name),
    values: Object.fromEntries(spec.outputs.map((output) => [output.name, output.shape])),
  });
  // `stubModel` declares every value as f32, so the output dtypes are layered on here.
  const values = { ...stub.graph.values };
  for (const output of spec.outputs) {
    if (output.dtype !== undefined) {
      values[output.name] = { ...values[output.name], dtype: output.dtype };
    }
  }
  return { graph: { ...stub.graph, values } };
};

/**
 * The two graphs as the convention says (of the three caches in upstream `feat_idx` order, first does
 * not have time_conv's `cache_01` — the doc of `vae-chunks.ts`). Each cache has a different shape, so
 * a mix-up shows in the shapes.
 */
const FIRST: ChunkSpec = {
  inputs: [
    { name: "latent", shape: [CHANNELS, 1, TILE, TILE] },
    { name: "cache_00", shape: [3, 2, TILE, TILE] },
    { name: "cache_02", shape: [5, 2, 2 * TILE, 2 * TILE] },
  ],
  outputs: [
    { name: "frame", shape: [3, 1, SIDE, SIDE] },
    { name: "cache_00_out", shape: [3, 2, TILE, TILE] },
    { name: "cache_02_out", shape: [5, 2, 2 * TILE, 2 * TILE] },
  ],
};
const NEXT: ChunkSpec = {
  inputs: [
    { name: "latent", shape: [CHANNELS, 1, TILE, TILE] },
    { name: "cache_00", shape: [3, 2, TILE, TILE] },
    { name: "cache_01", shape: [4, 2, TILE, TILE] },
    { name: "cache_02", shape: [5, 2, 2 * TILE, 2 * TILE] },
  ],
  outputs: [
    { name: "frame", shape: [3, 4, SIDE, SIDE] },
    { name: "cache_00_out", shape: [3, 2, TILE, TILE] },
    { name: "cache_01_out", shape: [4, 2, TILE, TILE] },
    { name: "cache_02_out", shape: [5, 2, 2 * TILE, 2 * TILE] },
  ],
};

/** Replaces one point of one declaration (input / output index → new entry; `null` deletes it). */
const patched = (
  spec: ChunkSpec,
  patch: {
    readonly inputs?: Readonly<Record<number, ChunkSpec["inputs"][number] | null>>;
    readonly outputs?: Readonly<Record<number, ChunkSpec["outputs"][number] | null>>;
  },
): ChunkSpec => {
  const apply = <T>(items: readonly T[], changes: Readonly<Record<number, T | null>> = {}) =>
    items.flatMap((item, index) => {
      if (!Object.hasOwn(changes, index)) return [item];
      const change = changes[index];
      return change === null ? [] : [change];
    });
  return {
    inputs: apply(spec.inputs, patch.inputs),
    outputs: apply(spec.outputs, patch.outputs),
  };
};

const layoutOf = (first: ChunkSpec = FIRST, next: ChunkSpec = NEXT) =>
  wanVaeChunkLayout(owner(first), owner(next));

describe("wanVaeChunkLayout (graph declarations → the chunk-sequence contract)", () => {
  it("takes the tile edge, the scale and the cache shapes from the declarations, and pairs cache inputs and outputs in the same order", () => {
    const layout = layoutOf();
    assertEquals([layout.latentChannels, layout.tile, layout.sampleTile], [CHANNELS, TILE, SIDE]);
    assertEquals(
      [...layout.cacheShapes.entries()],
      [
        ["cache_00", [3, 2, TILE, TILE]],
        ["cache_01", [4, 2, TILE, TILE]],
        ["cache_02", [5, 2, 2 * TILE, 2 * TILE]],
      ],
    );
    assertEquals(layout.first.caches, ["cache_00", "cache_02"]);
    assertEquals(layout.first.cacheOutputs, ["cache_00_out", "cache_02_out"]);
    assertEquals(layout.next.caches, ["cache_00", "cache_01", "cache_02"]);
    assertEquals(layout.next.cacheOutputs, ["cache_00_out", "cache_01_out", "cache_02_out"]);
    assertEquals(layout.first.frameShape, [3, 1, SIDE, SIDE]);
    assertEquals(layout.next.frameShape, [3, 4, SIDE, SIDE]);
  });

  it("takes the output channel count from axis 0 of the frame output (does not assume RGB's 3)", () => {
    assertEquals(layoutOf().sampleChannels, 3);
    // A synthetic declaration with 12 output channels (only the frame outputs of both first and next
    // are replaced).
    const wide = layoutOf(
      patched(FIRST, { outputs: { 0: { name: "frame", shape: [12, 1, SIDE, SIDE] } } }),
      patched(NEXT, { outputs: { 0: { name: "frame", shape: [12, 4, SIDE, SIDE] } } }),
    );
    assertEquals(wide.sampleChannels, 12);
  });

  it("throws WanVaeChunkError when the output channel count differs between first and next", () => {
    assertThrows(
      () =>
        layoutOf(
          patched(FIRST, { outputs: { 0: { name: "frame", shape: [12, 1, SIDE, SIDE] } } }),
          NEXT,
        ),
      WanVaeChunkError,
      "The frame channel count differs between first and next (first 12, next 3)",
    );
  });

  it("throws WanVaeChunkError when the declared dtype of any one of the latent, a cache input / output or the frame is not f32", () => {
    // A declaration with correct shapes and only a wrong dtype passes every shape check. Reading an
    // i32 frame as f32 turns the integer 1 into a finite value of about 1.4e-45, which passes the
    // finiteness check too, and an almost black video comes out silently.
    const rejected: readonly [string, ChunkSpec, ChunkSpec, string][] = [
      [
        "the frame output is i32",
        FIRST,
        patched(NEXT, { outputs: { 0: { ...NEXT.outputs[0], dtype: "i32" } } }),
        "vae_decoder_next: frame output 'frame' has dtype i32, not f32",
      ],
      [
        "a cache input is i32",
        patched(FIRST, { inputs: { 2: { ...FIRST.inputs[2], dtype: "i32" } } }),
        NEXT,
        "vae_decoder_first: cache input 'cache_02' has dtype i32, not f32",
      ],
      [
        "an updated cache output is bool",
        FIRST,
        patched(NEXT, { outputs: { 2: { ...NEXT.outputs[2], dtype: "bool" } } }),
        "vae_decoder_next: updated cache output 'cache_01_out' has dtype bool, not f32",
      ],
      [
        "the latent input is i32",
        patched(FIRST, { inputs: { 0: { ...FIRST.inputs[0], dtype: "i32" } } }),
        NEXT,
        "vae_decoder_first: latent input 'latent' has dtype i32, not f32",
      ],
    ];
    for (const [label, first, next, message] of rejected) {
      assertThrows(() => layoutOf(first, next), WanVaeChunkError, message, label);
    }
  });

  it("throws WanVaeChunkError naming which graph and what, when one declaration departs from the contract", () => {
    const latent = (shape: readonly StubDim[]) => ({ name: "latent", shape });
    const rejected: readonly [string, ChunkSpec, ChunkSpec, string][] = [
      [
        "the first input is not the latent",
        patched(FIRST, { inputs: { 0: { name: "z", shape: [CHANNELS, 1, TILE, TILE] } } }),
        NEXT,
        "vae_decoder_first: the first input is not 'latent'",
      ],
      [
        "the latent has a symbolic dimension",
        FIRST,
        patched(NEXT, { inputs: { 0: latent([CHANNELS, 1, "T", TILE]) } }),
        "is not a static dimension",
      ],
      [
        "the latent has 2 frames",
        patched(FIRST, { inputs: { 0: latent([CHANNELS, 2, TILE, TILE]) } }),
        NEXT,
        "the latent must be [C,1,t,t]",
      ],
      [
        "the latent is not square",
        FIRST,
        patched(NEXT, { inputs: { 0: latent([CHANNELS, 1, TILE, 2 * TILE]) } }),
        "the latent must be [C,1,t,t]",
      ],
      [
        "the output count differs from the input count",
        patched(FIRST, { outputs: { 2: null } }),
        NEXT,
        "output count 2 differs from input count 3",
      ],
      [
        "first's frame has 4 frames",
        patched(FIRST, { outputs: { 0: { name: "frame", shape: [3, 4, SIDE, SIDE] } } }),
        NEXT,
        "vae_decoder_first: frame shape",
      ],
      [
        "the frame edge is not a multiple of the tile",
        FIRST,
        patched(NEXT, { outputs: { 0: { name: "frame", shape: [3, 4, SIDE + 1, SIDE + 1] } } }),
        "vae_decoder_next: frame shape",
      ],
      [
        "the cache spelling",
        FIRST,
        patched(NEXT, { inputs: { 2: { name: "cache_1", shape: [4, 2, TILE, TILE] } } }),
        "'cache_1' is not spelled cache_NN",
      ],
      [
        "a cache holds 3 frames",
        FIRST,
        patched(NEXT, { inputs: { 2: { name: "cache_01", shape: [4, 3, TILE, TILE] } } }),
        "'cache_01' must be [Cin,2,h,w]",
      ],
      [
        "an updated cache's shape differs from the input (order mixed up)",
        FIRST,
        patched(NEXT, {
          outputs: {
            1: { name: "cache_01_out", shape: [4, 2, TILE, TILE] },
            2: { name: "cache_00_out", shape: [3, 2, TILE, TILE] },
          },
        }),
        "by convention, outputs follow the input order",
      ],
      [
        "no caches",
        FIRST,
        { inputs: [NEXT.inputs[0]], outputs: [NEXT.outputs[0]] },
        "vae_decoder_next: no cache inputs",
      ],
      [
        "the latent shapes of the two graphs differ",
        patched(FIRST, { inputs: { 0: latent([CHANNELS + 1, 1, TILE, TILE]) } }),
        NEXT,
        "The latent shapes of the 2 graphs differ",
      ],
      [
        "the frame edge differs between first and next",
        patched(FIRST, { outputs: { 0: { name: "frame", shape: [3, 1, 2 * SIDE, 2 * SIDE] } } }),
        NEXT,
        "The frame side differs between first and next",
      ],
      [
        "a cache of first is not in next",
        patched(FIRST, {
          inputs: { 1: { name: "cache_03", shape: [3, 2, TILE, TILE] } },
        }),
        NEXT,
        "'cache_03' of first is not in next with the same shape",
      ],
      [
        "a cache of first has a different shape in next",
        patched(FIRST, {
          inputs: { 1: { name: "cache_00", shape: [6, 2, TILE, TILE] } },
          outputs: { 1: { name: "cache_00_out", shape: [6, 2, TILE, TILE] } },
        }),
        NEXT,
        "'cache_00' of first is not in next with the same shape",
      ],
      [
        "first's cache order differs from next",
        {
          inputs: [FIRST.inputs[0], FIRST.inputs[2], FIRST.inputs[1]],
          outputs: [FIRST.outputs[0], FIRST.outputs[2], FIRST.outputs[1]],
        },
        NEXT,
        "The cache order of first differs from next ('cache_00')",
      ],
    ];
    for (const [label, first, next, message] of rejected) {
      assertThrows(() => layoutOf(first, next), WanVaeChunkError, message, label);
    }
  });
});

describe("cutting latent chunks and concatenating frames", () => {
  /**
   * `[C=2, F=3, t, t]`, value = `c·100 + f·10 + pixel` (a mix-up of channel, chunk or pixel shows in
   * the values).
   */
  const layout = layoutOf();
  const area = TILE * TILE;
  const latents = new Float32Array(CHANNELS * 3 * area).map((_, index) => {
    const plane = Math.floor(index / area);
    return Math.floor(plane / 3) * 100 + (plane % 3) * 10 + (index % area);
  });

  it("takes the chunk count from the latent length and rejects a length that does not divide into [C,F,t,t]", () => {
    assertEquals(wanVaeChunkCount(layout, latents), 3);
    assertThrows(
      () => wanVaeChunkCount(layout, latents.subarray(1)),
      WanVaeChunkError,
      "Latent element count",
    );
    assertThrows(() => wanVaeChunkCount(layout, new Float32Array(0)), WanVaeChunkError);
  });

  it("copies chunk k's single frame [C,1,t,t] from each channel", () => {
    const chunk = wanVaeLatentChunk(layout, latents, 1);
    assertEquals(chunk.shape, [CHANNELS, 1, TILE, TILE]);
    assert(chunk.dtype === "f32");
    const expected = [0, 1].flatMap((channel) =>
      Array.from({ length: area }, (_, pixel) => channel * 100 + 10 + pixel),
    );
    assertEquals([...chunk.data], expected);
  });

  it("throws RangeError when the chunk index is outside [0, chunks) or not an integer (does not silently return a neighbouring plane or zeros)", () => {
    for (const index of [3, -1, 1.5, Number.NaN]) {
      assertThrows(
        () => wanVaeLatentChunk(layout, latents, index),
        RangeError,
        "is outside [0, 3)",
      );
    }
  });

  it("concatenates the frames along the time axis in chunk order into [3, 1 + 4(F−1), s, s]", () => {
    assertEquals([wanVaeFrameCount(1), wanVaeFrameCount(3)], [1, 9]);
    const plane = SIDE * SIDE;
    // Chunk k's value = `k·10 + c` (chunk 0 has 1 frame, later ones 4).
    const chunk = (index: number, frames: number) =>
      new Float32Array(3 * frames * plane).map(
        (_, at) => index * 10 + Math.floor(at / (frames * plane)),
      ).buffer;
    const out = concatWanVaeFrames(layout, [chunk(0, 1), chunk(1, 4), chunk(2, 4)]);
    assertEquals(out.length, 3 * 9 * plane);
    for (let channel = 0; channel < 3; channel += 1) {
      const frames = Array.from({ length: 9 }, (_, frame) => out[(channel * 9 + frame) * plane]);
      assertEquals(
        frames,
        [0, 10, 10, 10, 10, 20, 20, 20, 20].map((value) => value + channel),
      );
    }
    assertThrows(
      () => concatWanVaeFrames(layout, [chunk(0, 1), chunk(1, 1)]),
      WanVaeChunkError,
      "chunk 1 frame element count",
    );
  });

  it("concatenates the declared output channel count (a declaration of 12 copies 12 channels)", () => {
    const channels = 12;
    const wide = layoutOf(
      patched(FIRST, { outputs: { 0: { name: "frame", shape: [channels, 1, SIDE, SIDE] } } }),
      patched(NEXT, { outputs: { 0: { name: "frame", shape: [channels, 4, SIDE, SIDE] } } }),
    );
    const plane = SIDE * SIDE;
    // Chunk k's value = `k·100 + c` (a concatenation that assumes 3 does not copy the planes with
    // c ≥ 3, and its length does not match either).
    const chunk = (index: number, frames: number) =>
      new Float32Array(channels * frames * plane).map(
        (_, at) => index * 100 + Math.floor(at / (frames * plane)),
      ).buffer;
    const out = concatWanVaeFrames(wide, [chunk(0, 1), chunk(1, 4)]);
    assertEquals(out.length, channels * 5 * plane);
    for (let channel = 0; channel < channels; channel += 1) {
      const frames = Array.from({ length: 5 }, (_, frame) => out[(channel * 5 + frame) * plane]);
      assertEquals(
        frames,
        [0, 100, 100, 100, 100].map((value) => value + channel),
      );
    }
  });
});

describe("WanVaeChunkCaches (the cache residents and their tile-head zeroing)", () => {
  /**
   * A GpuContext whose `createResident` hands out mock residents (only `byteLength` / `clear` /
   * `write` / `dispose` — a fake device cannot back a real one, so the boundary casts). Allocation
   * number `failAt` (0-based) rejects with `failure`.
   */
  const mockResidents = (options: { readonly failAt?: number; readonly failure?: Error } = {}) => {
    const gpu = fakeGpuContext(fakeDevice());
    const created: string[] = [];
    const disposed: string[] = [];
    const cleared: string[] = [];
    const written: string[] = [];
    // oxlint-disable-next-line typescript/no-useless-default-assignment -- false positive: GpuContext.createResident declares `label = "resident"`, so `label` is optional for callers; dropping the default types it `string | undefined`
    gpu.createResident = (byteLength: number, label = "resident") => {
      if (created.length === options.failAt) return Promise.reject(options.failure);
      created.push(label);
      return Promise.resolve({
        byteLength,
        clear: () => cleared.push(label),
        write: () => written.push(label),
        dispose: () => disposed.push(label),
      } as unknown as ResidentTensor);
    };
    return { gpu, created, disposed, cleared, written };
  };

  it("a resident allocation failing partway rethrows that failure and leaves no resident behind", async () => {
    const failure = new Error("out of memory on the third cache");
    const { gpu, created, disposed } = mockResidents({ failAt: 2, failure });
    const rejected = await assertRejects(() => WanVaeChunkCaches.create(gpu, layoutOf()), Error);
    assertStrictEquals(rejected, failure, "the allocation failure was not rethrown as is");
    assertEquals(created.length, 2, "the failure did not hit partway (the check would be vacuous)");
    assertEquals(disposed.toSorted(), created.toSorted(), "an allocated resident was left behind");
  });

  it("zero() clears every cache on the GPU, the time_conv cache first never writes included, without a host write", async () => {
    const { gpu, cleared, written } = mockResidents();
    const caches = await WanVaeChunkCaches.create(gpu, layoutOf());
    caches.zero();
    // NEXT's caches: cache_01 is the time_conv cache FIRST does not have (the module doc).
    assertEquals(cleared.toSorted(), ["cache_00", "cache_01", "cache_02"]);
    assertEquals(written, [], "zeroing went through a host write");
  });
});

/**
 * A GPU-cheap stand-in for the two chunk graphs (every value a few floats): the frame and the updated
 * caches depend on the incoming caches, so caches left over from an earlier tile change the output.
 * first: `s = cache_00 + latent`, frame = `s` as `[2,1,1,1]`, `cache_00 ← s + latent`. next: `t =
 * cache_00 + cache_01 + latent`, frame = `t` repeated over 4 frames, `cache_00 ← t + latent`,
 * `cache_01 ← cache_01 + latent` (cache_01 is the time_conv-like cache first never writes).
 */
const CACHE_SHAPE = [1, 2, 1, 1];
const LATENT_SHAPE = [1, 1, 1, 1];
const stubChunk = (head: boolean): DeclarationJson => {
  const value = (shape: number[]) => ({ dtype: "f32", shape });
  const caches = head ? ["cache_00"] : ["cache_00", "cache_01"];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: head ? ["add", "reshape"] : ["add", "reshape", "expand"] },
    symbols: [],
    inputs: [
      { name: "latent", dtype: "f32", shape: LATENT_SHAPE },
      ...caches.map((name) => ({ name, dtype: "f32", shape: CACHE_SHAPE })),
    ],
    outputs: ["frame", ...caches.map((name) => `${name}_out`)],
    initializers: {},
    values: head
      ? {
          s: value(CACHE_SHAPE),
          frame: value([2, 1, 1, 1]),
          cache_00_out: value(CACHE_SHAPE),
        }
      : {
          s: value(CACHE_SHAPE),
          t: value(CACHE_SHAPE),
          r: value([2, 1, 1, 1]),
          frame: value([2, 4, 1, 1]),
          cache_00_out: value(CACHE_SHAPE),
          cache_01_out: value(CACHE_SHAPE),
        },
    nodes: head
      ? [
          { op: "add", ins: ["cache_00", "latent"], outs: ["s"], attrs: {} },
          { op: "reshape", ins: ["s"], outs: ["frame"], attrs: {} },
          { op: "add", ins: ["s", "latent"], outs: ["cache_00_out"], attrs: {} },
        ]
      : [
          { op: "add", ins: ["cache_00", "cache_01"], outs: ["s"], attrs: {} },
          { op: "add", ins: ["s", "latent"], outs: ["t"], attrs: {} },
          { op: "reshape", ins: ["t"], outs: ["r"], attrs: {} },
          { op: "expand", ins: ["r"], outs: ["frame"], attrs: {} },
          { op: "add", ins: ["t", "latent"], outs: ["cache_00_out"], attrs: {} },
          { op: "add", ins: ["cache_01", "latent"], outs: ["cache_01_out"], attrs: {} },
        ],
  };
};

describe.skipIf(!GPU_AVAILABLE)(
  "decodeWanVaeTile across tiles (stub chunk graphs, real GPU)",
  () => {
    it("the second tile on the same caches starts from zeros: bit-identical to fresh caches and to a host write of zeros", async () => {
      const gpu = await acquireGpu();
      const first = prepareContainer(await openGraphModel(stubChunk(true)), GRAPH_NAME);
      const next = prepareContainer(await openGraphModel(stubChunk(false)), GRAPH_NAME);
      const sessions = {
        first: await first.createContainerSession(gpu),
        next: await next.createContainerSession(gpu),
      };
      const layout = wanVaeChunkLayout({ graph: first.graph }, { graph: next.graph });
      const reused = await WanVaeChunkCaches.create(gpu, layout);
      const fresh = await WanVaeChunkCaches.create(gpu, layout);
      // Latents `[1, F=3, 1, 1]`; tile A leaves non-zero caches behind for tile B.
      const tileA = Float32Array.from([1, 2, 3]);
      const tileB = Float32Array.from([5, 7, 11]);
      const bits = (values: Float32Array<ArrayBuffer>): number[] => [
        ...new Uint32Array(values.buffer),
      ];
      try {
        // The reference is a tile on fresh caches. It also goes through `zero()` = `clear()`, so it is
        // independent of clear() only against a clear that does nothing (a new WebGPU buffer starts as
        // zeros). A clear that wrote non-zero values would corrupt the reference and the reused tile
        // alike; the comparison with host-written zeros and no clear (skip-zero) below catches that.
        const reference = bits(await decodeWanVaeTile(gpu, sessions, fresh, tileB));

        await decodeWanVaeTile(gpu, sessions, reused, tileA);
        // Detector check: without zeroing, tile A's caches leak into tile B's output.
        const leaked = bits(
          await decodeWithFault(gpu, sessions, reused, tileB, { kind: "skip-zero" }),
        );
        assert(
          JSON.stringify(leaked) !== JSON.stringify(reference),
          "leftover caches do not change the output (the check is vacuous)",
        );

        await decodeWanVaeTile(gpu, sessions, reused, tileA);
        assertEquals(bits(await decodeWanVaeTile(gpu, sessions, reused, tileB)), reference);

        // The host-write path clear() replaced (`write` of a zero array) gives the same bits.
        await decodeWanVaeTile(gpu, sessions, reused, tileA);
        for (const resident of Object.values(reused.inputs(layout.next))) {
          resident.write(new Float32Array(resident.byteLength / 4));
        }
        const hostZeroed = await decodeWithFault(gpu, sessions, reused, tileB, {
          kind: "skip-zero",
        });
        assertEquals(bits(hostZeroed), reference);
      } finally {
        await sessions.first.dispose();
        await sessions.next.dispose();
        reused.dispose();
        fresh.dispose();
        gpu.destroy();
      }
    });
  },
);

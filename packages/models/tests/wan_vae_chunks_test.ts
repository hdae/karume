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

/** 潜在タイルの辺・潜在のチャネル数・出力フレームの辺（`8t`）。 */
const TILE = 4;
const CHANNELS = 2;
const SIDE = 8 * TILE;

/** 宣言の dtype（省けば f32）。 */
type Dtype = GraphOwner["graph"]["inputs"][number]["dtype"];

/** chunk グラフ 1 本の宣言の 1 本（入力 / 出力）。 */
type ChunkValue = {
  readonly name: string;
  readonly shape: readonly StubDim[];
  readonly dtype?: Dtype;
};

/** chunk グラフ 1 本の宣言（入力の順 = 潜在 → cache・出力の順 = フレーム → 更新後の cache）。 */
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
  // `stubModel` の値の宣言は f32 固定なので、出力の dtype はここで重ねる。
  const values = { ...stub.graph.values };
  for (const output of spec.outputs) {
    if (output.dtype !== undefined) {
      values[output.name] = { ...values[output.name], dtype: output.dtype };
    }
  }
  return { graph: { ...stub.graph, values } };
};

/**
 * 取り決めどおりの 2 本（上流の `feat_idx` 順の cache 3 本のうち、first は time_conv の
 * `cache_01` を持たない — `vae-chunks.ts` の doc）。形は cache ごとに変えてあり、取り違えが形に出る。
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

/** 1 本の宣言の 1 点を差し替える（入力 / 出力の添字 → 新しい欄。`null` は削除）。 */
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

describe("wanVaeChunkLayout（グラフ宣言 → chunk 列の取り決め）", () => {
  it("タイル辺・縮尺・cache の形を宣言から引き、cache の入出力を同じ順で対にする", () => {
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

  it("出口のチャネル数はフレーム出力の軸 0 から引く（RGB の 3 を仮定しない）", () => {
    assertEquals(layoutOf().sampleChannels, 3);
    // 出口 12 チャネルの合成の宣言（first / next の両方のフレーム出力だけを差し替える）。
    const wide = layoutOf(
      patched(FIRST, { outputs: { 0: { name: "frame", shape: [12, 1, SIDE, SIDE] } } }),
      patched(NEXT, { outputs: { 0: { name: "frame", shape: [12, 4, SIDE, SIDE] } } }),
    );
    assertEquals(wide.sampleChannels, 12);
  });

  it("出口のチャネル数が first と next で違えば WanVaeChunkError", () => {
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

  it("潜在・cache の入出力・フレームのどれか 1 本でも宣言の dtype が f32 でなければ WanVaeChunkError", () => {
    // 形は正しく dtype だけが違う宣言は、形の検査を全部通る。i32 のフレームを f32 として読むと、
    // 整数の 1 が約 1.4e-45 の有限値になり、有限性の検査も通ってほぼ黒い動画が黙って出る。
    const rejected: readonly [string, ChunkSpec, ChunkSpec, string][] = [
      [
        "フレーム出力が i32",
        FIRST,
        patched(NEXT, { outputs: { 0: { ...NEXT.outputs[0], dtype: "i32" } } }),
        "vae_decoder_next: frame output 'frame' has dtype i32, not f32",
      ],
      [
        "cache の入力が i32",
        patched(FIRST, { inputs: { 2: { ...FIRST.inputs[2], dtype: "i32" } } }),
        NEXT,
        "vae_decoder_first: cache input 'cache_02' has dtype i32, not f32",
      ],
      [
        "更新後の cache の出力が bool",
        FIRST,
        patched(NEXT, { outputs: { 2: { ...NEXT.outputs[2], dtype: "bool" } } }),
        "vae_decoder_next: updated cache output 'cache_01_out' has dtype bool, not f32",
      ],
      [
        "潜在の入力が i32",
        patched(FIRST, { inputs: { 0: { ...FIRST.inputs[0], dtype: "i32" } } }),
        NEXT,
        "vae_decoder_first: latent input 'latent' has dtype i32, not f32",
      ],
    ];
    for (const [label, first, next, message] of rejected) {
      assertThrows(() => layoutOf(first, next), WanVaeChunkError, message, label);
    }
  });

  it("1 本の宣言が取り決めから外れたら、どちらのグラフの何かを言って WanVaeChunkError", () => {
    const latent = (shape: readonly StubDim[]) => ({ name: "latent", shape });
    const rejected: readonly [string, ChunkSpec, ChunkSpec, string][] = [
      [
        "先頭の入力が潜在でない",
        patched(FIRST, { inputs: { 0: { name: "z", shape: [CHANNELS, 1, TILE, TILE] } } }),
        NEXT,
        "vae_decoder_first: the first input is not 'latent'",
      ],
      [
        "潜在が記号次元",
        FIRST,
        patched(NEXT, { inputs: { 0: latent([CHANNELS, 1, "T", TILE]) } }),
        "is not a static dimension",
      ],
      [
        "潜在が 2 フレーム",
        patched(FIRST, { inputs: { 0: latent([CHANNELS, 2, TILE, TILE]) } }),
        NEXT,
        "the latent must be [C,1,t,t]",
      ],
      [
        "潜在が正方でない",
        FIRST,
        patched(NEXT, { inputs: { 0: latent([CHANNELS, 1, TILE, 2 * TILE]) } }),
        "the latent must be [C,1,t,t]",
      ],
      [
        "出力の本数が入力と違う",
        patched(FIRST, { outputs: { 2: null } }),
        NEXT,
        "output count 2 differs from input count 3",
      ],
      [
        "first のフレームが 4 枚",
        patched(FIRST, { outputs: { 0: { name: "frame", shape: [3, 4, SIDE, SIDE] } } }),
        NEXT,
        "vae_decoder_first: frame shape",
      ],
      [
        "フレームの辺がタイルの倍数でない",
        FIRST,
        patched(NEXT, { outputs: { 0: { name: "frame", shape: [3, 4, SIDE + 1, SIDE + 1] } } }),
        "vae_decoder_next: frame shape",
      ],
      [
        "cache の綴り",
        FIRST,
        patched(NEXT, { inputs: { 2: { name: "cache_1", shape: [4, 2, TILE, TILE] } } }),
        "'cache_1' is not spelled cache_NN",
      ],
      [
        "cache が 3 フレームを持つ",
        FIRST,
        patched(NEXT, { inputs: { 2: { name: "cache_01", shape: [4, 3, TILE, TILE] } } }),
        "'cache_01' must be [Cin,2,h,w]",
      ],
      [
        "更新後の cache の形が入力と違う（順の取り違え）",
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
        "cache が 0 本",
        FIRST,
        { inputs: [NEXT.inputs[0]], outputs: [NEXT.outputs[0]] },
        "vae_decoder_next: no cache inputs",
      ],
      [
        "潜在の形が 2 本で違う",
        patched(FIRST, { inputs: { 0: latent([CHANNELS + 1, 1, TILE, TILE]) } }),
        NEXT,
        "The latent shapes of the 2 graphs differ",
      ],
      [
        "フレームの辺が first と next で違う",
        patched(FIRST, { outputs: { 0: { name: "frame", shape: [3, 1, 2 * SIDE, 2 * SIDE] } } }),
        NEXT,
        "The frame side differs between first and next",
      ],
      [
        "first の cache が next に無い",
        patched(FIRST, {
          inputs: { 1: { name: "cache_03", shape: [3, 2, TILE, TILE] } },
        }),
        NEXT,
        "'cache_03' of first is not in next with the same shape",
      ],
      [
        "first の cache の形が next と違う",
        patched(FIRST, {
          inputs: { 1: { name: "cache_00", shape: [6, 2, TILE, TILE] } },
          outputs: { 1: { name: "cache_00_out", shape: [6, 2, TILE, TILE] } },
        }),
        NEXT,
        "'cache_00' of first is not in next with the same shape",
      ],
      [
        "first の cache の順が next と違う",
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

describe("潜在の chunk の切り出しとフレームの連結", () => {
  /** `[C=2, F=3, t, t]`・値 = `c·100 + f·10 + 画素`（チャネル・chunk・画素の取り違えが値に出る）。 */
  const layout = layoutOf();
  const area = TILE * TILE;
  const latents = new Float32Array(CHANNELS * 3 * area).map((_, index) => {
    const plane = Math.floor(index / area);
    return Math.floor(plane / 3) * 100 + (plane % 3) * 10 + (index % area);
  });

  it("chunk 数は潜在の長さから引き、[C,F,t,t] に割り切れない長さは落とす", () => {
    assertEquals(wanVaeChunkCount(layout, latents), 3);
    assertThrows(
      () => wanVaeChunkCount(layout, latents.subarray(1)),
      WanVaeChunkError,
      "Latent element count",
    );
    assertThrows(() => wanVaeChunkCount(layout, new Float32Array(0)), WanVaeChunkError);
  });

  it("chunk k の 1 フレーム [C,1,t,t] を各チャネルから写す", () => {
    const chunk = wanVaeLatentChunk(layout, latents, 1);
    assertEquals(chunk.shape, [CHANNELS, 1, TILE, TILE]);
    assert(chunk.dtype === "f32");
    const expected = [0, 1].flatMap((channel) =>
      Array.from({ length: area }, (_, pixel) => channel * 100 + 10 + pixel),
    );
    assertEquals([...chunk.data], expected);
  });

  it("chunk の番号が [0, chunks) の外・整数でないなら RangeError（隣の平面やゼロを黙って返さない）", () => {
    for (const index of [3, -1, 1.5, Number.NaN]) {
      assertThrows(
        () => wanVaeLatentChunk(layout, latents, index),
        RangeError,
        "is outside [0, 3)",
      );
    }
  });

  it("フレームを chunk 順に時間軸で連結して [3, 1 + 4(F−1), s, s] にする", () => {
    assertEquals([wanVaeFrameCount(1), wanVaeFrameCount(3)], [1, 9]);
    const plane = SIDE * SIDE;
    // chunk k の値 = `k·10 + c`（chunk 0 は 1 枚・以降は 4 枚）。
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

  it("連結のチャネル数は宣言の出口のチャネル数（12 の宣言なら 12 チャネルを写す）", () => {
    const channels = 12;
    const wide = layoutOf(
      patched(FIRST, { outputs: { 0: { name: "frame", shape: [channels, 1, SIDE, SIDE] } } }),
      patched(NEXT, { outputs: { 0: { name: "frame", shape: [channels, 4, SIDE, SIDE] } } }),
    );
    const plane = SIDE * SIDE;
    // chunk k の値 = `k·100 + c`（3 を仮定した連結は c ≥ 3 の平面を写さず、長さも合わない）。
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

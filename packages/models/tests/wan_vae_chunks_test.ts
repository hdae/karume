// Wan2.1 の動画 VAE の chunk グラフ 2 本のホスト側（`src/wan/vae-chunks.ts` の GPU 不要の部分）—
// グラフ宣言から chunk 列の取り決めを導く門・潜在の chunk の切り出し・フレームの連結。
//
// 実 GPU と実資産での chunk 列の照合は `e2e_wan_vae_chunks_test.ts`。e2e は正常系の資産だけを通すので、
// 宣言の故障形（取り決めから外れた資産を名指しで落とす分岐）はここが唯一の門になる。宣言は
// `helpers/stub-model.ts` の流儀で、正しい形を 1 つ書いてから 1 点だけ壊す。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  concatWanVaeFrames,
  wanVaeChunkCount,
  WanVaeChunkError,
  wanVaeChunkLayout,
  wanVaeFrameCount,
  wanVaeLatentChunk,
} from "../src/wan/vae-chunks.ts";
import { type StubDim, stubModel } from "./helpers/stub-model.ts";

/** 潜在タイルの辺・潜在のチャネル数・出力フレームの辺（`8t`）。 */
const TILE = 4;
const CHANNELS = 2;
const SIDE = 8 * TILE;

/** chunk グラフ 1 本の宣言（入力の順 = 潜在 → cache・出力の順 = フレーム → 更新後の cache）。 */
type ChunkSpec = {
  readonly inputs: readonly { readonly name: string; readonly shape: readonly StubDim[] }[];
  readonly outputs: readonly { readonly name: string; readonly shape: readonly StubDim[] }[];
};

const owner = (spec: ChunkSpec) =>
  stubModel({
    inputs: spec.inputs,
    outputs: spec.outputs.map((output) => output.name),
    values: Object.fromEntries(spec.outputs.map((output) => [output.name, output.shape])),
  });

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
    assertEquals([...layout.cacheShapes.entries()], [
      ["cache_00", [3, 2, TILE, TILE]],
      ["cache_01", [4, 2, TILE, TILE]],
      ["cache_02", [5, 2, 2 * TILE, 2 * TILE]],
    ]);
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
      "フレームのチャネル数が first と next で違う（first 12・next 3）",
    );
  });

  it("1 本の宣言が取り決めから外れたら、どちらのグラフの何かを言って WanVaeChunkError", () => {
    const latent = (shape: readonly StubDim[]) => ({ name: "latent", shape });
    const rejected: readonly [string, ChunkSpec, ChunkSpec, string][] = [
      [
        "先頭の入力が潜在でない",
        patched(FIRST, { inputs: { 0: { name: "z", shape: [CHANNELS, 1, TILE, TILE] } } }),
        NEXT,
        "vae_decoder_first: 先頭の入力が 'latent' でない",
      ],
      [
        "潜在が記号次元",
        FIRST,
        patched(NEXT, { inputs: { 0: latent([CHANNELS, 1, "T", TILE]) } }),
        "静的次元でない",
      ],
      [
        "潜在が 2 フレーム",
        patched(FIRST, { inputs: { 0: latent([CHANNELS, 2, TILE, TILE]) } }),
        NEXT,
        "潜在は [C,1,t,t]",
      ],
      [
        "潜在が正方でない",
        FIRST,
        patched(NEXT, { inputs: { 0: latent([CHANNELS, 1, TILE, 2 * TILE]) } }),
        "潜在は [C,1,t,t]",
      ],
      [
        "出力の本数が入力と違う",
        patched(FIRST, { outputs: { 2: null } }),
        NEXT,
        "出力 2 本が入力 3 本と違う",
      ],
      [
        "first のフレームが 4 枚",
        patched(FIRST, { outputs: { 0: { name: "frame", shape: [3, 4, SIDE, SIDE] } } }),
        NEXT,
        "vae_decoder_first: フレームの形",
      ],
      [
        "フレームの辺がタイルの倍数でない",
        FIRST,
        patched(NEXT, { outputs: { 0: { name: "frame", shape: [3, 4, SIDE + 1, SIDE + 1] } } }),
        "vae_decoder_next: フレームの形",
      ],
      [
        "cache の綴り",
        FIRST,
        patched(NEXT, { inputs: { 2: { name: "cache_1", shape: [4, 2, TILE, TILE] } } }),
        "'cache_1' が cache_NN の綴りでない",
      ],
      [
        "cache が 3 フレームを持つ",
        FIRST,
        patched(NEXT, { inputs: { 2: { name: "cache_01", shape: [4, 3, TILE, TILE] } } }),
        "'cache_01' は [Cin,2,h,w]",
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
        "入力と同じ順の取り決め",
      ],
      [
        "cache が 0 本",
        FIRST,
        { inputs: [NEXT.inputs[0]], outputs: [NEXT.outputs[0]] },
        "vae_decoder_next: cache の入力が 0 本",
      ],
      [
        "潜在の形が 2 本で違う",
        patched(FIRST, { inputs: { 0: latent([CHANNELS + 1, 1, TILE, TILE]) } }),
        NEXT,
        "潜在の形が 2 本で違う",
      ],
      [
        "フレームの辺が first と next で違う",
        patched(FIRST, { outputs: { 0: { name: "frame", shape: [3, 1, 2 * SIDE, 2 * SIDE] } } }),
        NEXT,
        "フレームの辺が first と next で違う",
      ],
      [
        "first の cache が next に無い",
        patched(FIRST, {
          inputs: { 1: { name: "cache_03", shape: [3, 2, TILE, TILE] } },
        }),
        NEXT,
        "first の 'cache_03' が next に同じ形で無い",
      ],
      [
        "first の cache の形が next と違う",
        patched(FIRST, {
          inputs: { 1: { name: "cache_00", shape: [6, 2, TILE, TILE] } },
          outputs: { 1: { name: "cache_00_out", shape: [6, 2, TILE, TILE] } },
        }),
        NEXT,
        "first の 'cache_00' が next に同じ形で無い",
      ],
      [
        "first の cache の順が next と違う",
        {
          inputs: [FIRST.inputs[0], FIRST.inputs[2], FIRST.inputs[1]],
          outputs: [FIRST.outputs[0], FIRST.outputs[2], FIRST.outputs[1]],
        },
        NEXT,
        "first の cache の順が next と違う（'cache_00'）",
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
      "潜在の要素数",
    );
    assertThrows(() => wanVaeChunkCount(layout, new Float32Array(0)), WanVaeChunkError);
  });

  it("chunk k の 1 フレーム [C,1,t,t] を各チャネルから写す", () => {
    const chunk = wanVaeLatentChunk(layout, latents, 1);
    assertEquals(chunk.shape, [CHANNELS, 1, TILE, TILE]);
    assert(chunk.dtype === "f32");
    const expected = [0, 1].flatMap((channel) =>
      Array.from({ length: area }, (_, pixel) => channel * 100 + 10 + pixel)
    );
    assertEquals([...chunk.data], expected);
  });

  it("chunk の番号が [0, chunks) の外・整数でないなら RangeError（隣の平面やゼロを黙って返さない）", () => {
    for (const index of [3, -1, 1.5, Number.NaN]) {
      assertThrows(() => wanVaeLatentChunk(layout, latents, index), RangeError, "[0, 3) の外");
    }
  });

  it("フレームを chunk 順に時間軸で連結して [3, 1 + 4(F−1), s, s] にする", () => {
    assertEquals([wanVaeFrameCount(1), wanVaeFrameCount(3)], [1, 9]);
    const plane = SIDE * SIDE;
    // chunk k の値 = `k·10 + c`（chunk 0 は 1 枚・以降は 4 枚）。
    const chunk = (index: number, frames: number) =>
      new Float32Array(3 * frames * plane).map((_, at) =>
        index * 10 + Math.floor(at / (frames * plane))
      ).buffer;
    const out = concatWanVaeFrames(layout, [chunk(0, 1), chunk(1, 4), chunk(2, 4)]);
    assertEquals(out.length, 3 * 9 * plane);
    for (let channel = 0; channel < 3; channel += 1) {
      const frames = Array.from({ length: 9 }, (_, frame) => out[(channel * 9 + frame) * plane]);
      assertEquals(frames, [0, 10, 10, 10, 10, 20, 20, 20, 20].map((value) => value + channel));
    }
    assertThrows(
      () => concatWanVaeFrames(layout, [chunk(0, 1), chunk(1, 1)]),
      WanVaeChunkError,
      "chunk 1 のフレームの要素数",
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
      new Float32Array(channels * frames * plane).map((_, at) =>
        index * 100 + Math.floor(at / (frames * plane))
      ).buffer;
    const out = concatWanVaeFrames(wide, [chunk(0, 1), chunk(1, 4)]);
    assertEquals(out.length, channels * 5 * plane);
    for (let channel = 0; channel < channels; channel += 1) {
      const frames = Array.from({ length: 5 }, (_, frame) => out[(channel * 5 + frame) * plane]);
      assertEquals(frames, [0, 100, 100, 100, 100].map((value) => value + channel));
    }
  });
});

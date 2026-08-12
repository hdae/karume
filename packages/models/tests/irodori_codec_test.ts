// Irodori の codec ホスト層（タイル幾何 + 末尾トリム）の挙動テスト。GPU は要らない。
//
// タイル幾何は「採用区間が隙間なく S を覆い、halo ぶんの文脈が両側にある（または真の境界）」
// という**構造**の検査で、値のビット一致は実 GPU の門（`e2e_irodori_codec_test.ts`）が見る。
// 末尾トリムは golden 3 ケース（`outputs/series/dacvae-32dim/host/trim.safetensors` — 上流
// `find_flattening_point` の実測が焼かれている）との突合。golden が無い環境は明示 SKIP。

import { assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import { type CodecTile, DEFAULT_CODEC_TILE_FRAMES, planCodecTiles } from "../src/irodori/codec.ts";
import { findFlatteningPoint, trimmedSampleCount } from "../src/irodori/host/trim.ts";

// ---- タイル幾何 -----------------------------------------------------------

/** 実重み v4-small の運用値（`pipelineConfig.codecHaloFrames`）。 */
const HALO = 8;

/**
 * 幾何の不変条件を外側からもう一度確かめる（`planCodecTiles` 内の表明とは別の書き方で）。
 * 返すのは採用区間の合計フレーム数。
 */
const auditTiles = (tiles: readonly CodecTile[], frames: number, halo: number): number => {
  let covered = 0;
  for (const tile of tiles) {
    const takeStart = tile.start + tile.offset;
    assertEquals(takeStart, covered, "採用区間が連続していない");
    const takeEnd = takeStart + tile.take;
    const leftContext = takeStart - tile.start;
    const rightContext = tile.start + tile.length - takeEnd;
    if (tile.start > 0) assertEquals(leftContext >= halo, true, "左の文脈が halo 未満");
    if (tile.start + tile.length < frames) {
      assertEquals(rightContext >= halo, true, "右の文脈が halo 未満");
    }
    covered = takeEnd;
  }
  return covered;
};

Deno.test("planCodecTiles: S がタイル長以下なら 1 枚に縮退する（= 単発 decode）", () => {
  for (const frames of [1, 2, 161, DEFAULT_CODEC_TILE_FRAMES]) {
    const tiles = planCodecTiles(frames, {
      tileFrames: DEFAULT_CODEC_TILE_FRAMES,
      haloFrames: HALO,
    });
    assertEquals(tiles, [{ start: 0, length: frames, offset: 0, take: frames }], `S=${frames}`);
  }
});

Deno.test("planCodecTiles: 採用区間が隙間なく S を覆い、両側に halo ぶんの文脈がある", () => {
  for (const frames of [17, 33, 64, 161, 183, 300, 749, 750]) {
    for (const tileFrames of [17, 24, 64, 100, DEFAULT_CODEC_TILE_FRAMES]) {
      const tiles = planCodecTiles(frames, { tileFrames, haloFrames: HALO });
      assertEquals(auditTiles(tiles, frames, HALO), frames, `S=${frames} tile=${tileFrames}`);
    }
  }
});

Deno.test("planCodecTiles: 複数枚なら decode 長は全タイル同一（prepared plan が効き続ける）", () => {
  const tiles = planCodecTiles(750, { tileFrames: 64, haloFrames: HALO });
  assertEquals(new Set(tiles.map((tile) => tile.length)), new Set([64]));
  // 採用は step = 64 − 2×8 = 48 フレームずつ（末尾だけ端数）。
  assertEquals(tiles.length, Math.ceil(750 / 48));
  assertEquals(tiles[0].offset, 0, "先頭タイルは左 halo を持たない（真の境界）");
  const last = tiles[tiles.length - 1];
  assertEquals(last.start + last.length, 750, "末尾タイルの右端は真の境界へスナップする");
});

Deno.test("planCodecTiles: halo 2 枚ぶん以下のタイル長は落とす（採用が 1 枚も残らない）", () => {
  assertThrows(
    () => planCodecTiles(300, { tileFrames: 16, haloFrames: HALO }),
    Error,
    "halo 2 枚ぶん",
  );
});

Deno.test("planCodecTiles: S が正の整数でなければ落とす", () => {
  assertThrows(() => planCodecTiles(0, { tileFrames: 64, haloFrames: HALO }), Error, "frames 0");
  assertThrows(
    () => planCodecTiles(1.5, { tileFrames: 64, haloFrames: HALO }),
    Error,
    "frames 1.5",
  );
});

// ---- 末尾トリム -----------------------------------------------------------

const LATENT_DIM = 32;
/** 上流 `hop_length`（`pipelineConfig.hopLength`）。 */
const HOP = 1920;

const GOLDEN_DIR = new URL("../../../outputs/series/dacvae-32dim/host/", import.meta.url);
const GOLDEN_COMMAND =
  "cd tools/exporter && uv run --with descript-audiotools --with einops python dacvae_host.py";

/** golden `meta.json` の trim 節（正本は `dacvae_host.py` — ここは読み口）。 */
type TrimMeta = {
  readonly trim: Readonly<Record<string, { readonly frames: number; readonly point: number }>>;
};

const trimBytes = await Deno.readFile(new URL("trim.safetensors", GOLDEN_DIR)).catch(
  () => undefined,
);
const trimMetaText = await Deno.readTextFile(new URL("meta.json", GOLDEN_DIR)).catch(
  () => undefined,
);
const GOLDEN_AVAILABLE = trimBytes !== undefined && trimMetaText !== undefined;
if (!GOLDEN_AVAILABLE) {
  console.warn(
    `[karume] 末尾トリムの golden 突合を SKIP する（${GOLDEN_DIR.pathname} が要る）。` +
      `生成: ${GOLDEN_COMMAND}`,
  );
}

Deno.test({
  name: "findFlatteningPoint: golden 3 ケースが上流 find_flattening_point と同じ位置を返す",
  ignore: !GOLDEN_AVAILABLE,
  fn: () => {
    const bytes = trimBytes as Uint8Array<ArrayBuffer>;
    const file = parseSafetensors(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const meta = JSON.parse(trimMetaText as string) as TrimMeta;
    // 実 z 2 本は「切らない」側、合成 1 本は「ちょうど 121 を返す」側（恒真化の遮断）。
    assertEquals(Object.keys(meta.trim).sort(), ["z-full", "z-no-ref", "z-silent-tail"]);
    for (const [name, expected] of Object.entries(meta.trim)) {
      const spec = file.tensors.get(name);
      if (spec === undefined) throw new Error(`golden trim に '${name}' が無い`);
      const latent = new Float32Array(
        file.buffer.slice(spec.byteOffset, spec.byteOffset + spec.byteLength),
      );
      assertEquals(spec.shape, [1, expected.frames, LATENT_DIM], `${name} の形`);
      assertEquals(
        findFlatteningPoint(latent, expected.frames, LATENT_DIM),
        expected.point,
        `${name}: 上流と違う位置を返した`,
      );
    }
  },
});

Deno.test("findFlatteningPoint: 全部 0 の latent は先頭で平坦（窓が全部 pad と同じ）", () => {
  assertEquals(findFlatteningPoint(new Float32Array(10 * LATENT_DIM), 10, LATENT_DIM), 0);
});

Deno.test("findFlatteningPoint: 値が一定でも 0 近傍でなければ平坦としない（窓は pad ごと見る）", () => {
  // 全要素 1.0。末尾の窓は 0 pad で平均が薄まる（i=9 で mean 0.05 < 0.1）が、実在行と pad の
  // 差が std 0.218 を作るので通らない — 見つからず `frames` が返る。
  const flat = new Float32Array(10 * LATENT_DIM).fill(1);
  assertEquals(findFlatteningPoint(flat, 10, LATENT_DIM), 10);
});

Deno.test("findFlatteningPoint: 末尾を 0 にした位置をそのまま返す（golden 合成ケースの縮小版）", () => {
  const frames = 40;
  const silentFrom = 25;
  const latent = new Float32Array(frames * LATENT_DIM);
  // 平均 0・分散 1 の交互符号（mean のしきい値では落ちず std だけで落ちる形）。
  for (let index = 0; index < silentFrom * LATENT_DIM; index += 1) {
    latent[index] = index % 2 === 0 ? 1 : -1;
  }
  assertEquals(findFlatteningPoint(latent, frames, LATENT_DIM), silentFrom);
});

Deno.test("findFlatteningPoint: 要素数が frames × latentDim と違えば落とす", () => {
  assertThrows(
    () => findFlatteningPoint(new Float32Array(31), 1, LATENT_DIM),
    Error,
    "要素数 31",
  );
});

Deno.test("trimmedSampleCount: 短いほうを採る（上流 max_samples）", () => {
  assertEquals(trimmedSampleCount(161 * HOP, 121, HOP), 121 * HOP, "トリムが効く");
  assertEquals(trimmedSampleCount(161 * HOP, 161, HOP), 161 * HOP, "切らない");
  // 秒指定でフレーム全長より短い切り出しが要求されているときは、そちらが勝つ。
  assertEquals(trimmedSampleCount(48_192, 26, HOP), 48_192);
});

Deno.test("trimmedSampleCount: 先頭から平坦（0）のときだけ切らない（0 サンプルを返さない）", () => {
  assertEquals(trimmedSampleCount(161 * HOP, 0, HOP), 161 * HOP);
});

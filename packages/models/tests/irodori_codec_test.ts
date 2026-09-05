// Irodori の codec ホスト層（タイル幾何 + 末尾トリム）の挙動テスト。GPU は要らない。
//
// タイル幾何は「採用区間が隙間なく S を覆い、halo ぶんの文脈が両側にある（または真の境界）」
// という**構造**の検査で、値のビット一致は実 GPU の門（`e2e_irodori_codec_test.ts`）が見る。
// 末尾トリムは golden 3 ケース（`outputs/series/dacvae-32dim/host/trim.safetensors` — 上流
// `find_flattening_point` の実測が焼かれている）との突合。golden が無い環境は明示 SKIP。

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { parseSafetensors } from "@karume/runtime";
import {
  type CodecTile,
  decodeTiles,
  DEFAULT_CODEC_TILE_FRAMES,
  planCodecTiles,
} from "../src/irodori/codec.ts";
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

// ---- 貼り合わせ（decodeTiles）---------------------------------------------
//
// `decodeTiles` は `run` を注入で受ける（本体の貼り合わせそのものを外から測れるようにするため
// — `codec.ts` のモジュール doc）。実 GPU の門（`e2e_irodori_codec_test.ts`）は decoder 資産と
// golden が要るので、資産の無い機ではここが唯一の門になる。anima 側の同型 `decodeTiled` は
// 同じ手で GPU 無しに縛られている（`anima_tiling_test.ts`）。

const TILE_LATENT_DIM = 2;
const TILE_HOP = 4;

/** latent `[frames, latentDim]`（先頭列がフレーム番号 = 位置が値で読める ramp）。 */
const rampLatent = (frames: number): Float32Array<ArrayBuffer> => {
  const latent = new Float32Array(frames * TILE_LATENT_DIM) as Float32Array<ArrayBuffer>;
  for (let frame = 0; frame < frames; frame += 1) latent[frame * TILE_LATENT_DIM] = frame;
  return latent;
};

/**
 * 平行移動同変な偽 decoder — 出力サンプルが**入力フレームの値だけ**から決まる
 * （`out[i*hop + s] = slice[i*latentDim]*hop + s`）。ramp を食わせると全長 decode の結果が
 * 恒等列になるので、貼り合わせが 1 サンプルでもずれれば値で露見する。
 */
const equivariantDecoder = (slice: Float32Array<ArrayBuffer>, frames: number) => {
  const out = new Float32Array(frames * TILE_HOP);
  for (let index = 0; index < frames; index += 1) {
    for (let sample = 0; sample < TILE_HOP; sample += 1) {
      out[index * TILE_HOP + sample] = slice[index * TILE_LATENT_DIM] * TILE_HOP + sample;
    }
  }
  return Promise.resolve(out);
};

Deno.test("decodeTiles: 採用区間を全長の正しい位置へ貼る（複数枚が単発 decode と一致）", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  assertEquals(tiles.length > 1, true, "1 枚に縮退すると貼り合わせの添字算術を踏まない");

  const waveform = await decodeTiles(
    rampLatent(frames),
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    equivariantDecoder,
  );

  assertEquals(waveform.length, frames * TILE_HOP);
  // offset / take / start のどれかの掛け算が違えば、この恒等列が崩れる。
  assertEquals([...waveform], Array.from({ length: frames * TILE_HOP }, (_, index) => index));
});

Deno.test("decodeTiles: 位置を無視する decoder では一致が破れる（門が恒真でない証拠）", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  // 入力フレームの値ではなく**タイル内の添字**から作る（= 平行移動同変でない）decoder。
  const waveform = await decodeTiles(
    rampLatent(frames),
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    (_slice, tileFrames) =>
      Promise.resolve(
        Float32Array.from({ length: tileFrames * TILE_HOP }, (_, index) => index),
      ),
  );
  assertEquals(
    [...waveform].some((value, index) => value !== index),
    true,
    "位置を無視した decoder でも一致してしまい、この門は何も縛れていない",
  );
});

Deno.test("decodeTiles: run へ渡すのはタイル 1 枚ぶんの写し（view でない）", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  let call = 0;
  await decodeTiles(
    rampLatent(frames),
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    (slice, tileFrames) => {
      const tile = tiles[call];
      call += 1;
      assertEquals(tileFrames, tile.length, "frames 引数がタイルの decode 長と違う");
      assertEquals(slice.length, tile.length * TILE_LATENT_DIM);
      // 写しならバッファ全体をちょうど占める（view のまま渡すと、バッファ全体を占めることを
      // 前提にした受け渡しの契約から外れる）。
      assertEquals(slice.buffer.byteLength, slice.length * 4);
      assertEquals(slice[0], tile.start, "slice の先頭がタイルの開始フレームでない");
      return equivariantDecoder(slice, tileFrames);
    },
  );
  assertEquals(call, tiles.length);
});

Deno.test("decodeTiles: latent の要素数が latentDim で割れなければ落とす", async () => {
  const tiles = planCodecTiles(12, { tileFrames: 8, haloFrames: 2 });
  await assertRejects(
    () =>
      decodeTiles(
        new Float32Array(5) as Float32Array<ArrayBuffer>,
        { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
        equivariantDecoder,
      ),
    Error,
    "latentDim 2 で割れない",
  );
});

Deno.test("decodeTiles: decoder の出力長が合わなければ落とす（期待サンプル数つき）", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: 8, haloFrames: 2 });
  const error = await assertRejects(
    () =>
      decodeTiles(
        rampLatent(frames),
        { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
        (_slice, tileFrames) => Promise.resolve(new Float32Array(tileFrames * TILE_HOP - 1)),
      ),
    Error,
    "decoder の出力が",
  );
  assertStringIncludes(error.message, `${tiles[0].length * TILE_HOP} のはず`);
});

Deno.test("decodeTiles: 1 枚に縮退した計画では単発 decode の写しそのもの", async () => {
  const frames = 12;
  const tiles = planCodecTiles(frames, { tileFrames: DEFAULT_CODEC_TILE_FRAMES, haloFrames: HALO });
  assertEquals(tiles.length, 1);
  const latent = rampLatent(frames);
  const waveform = await decodeTiles(
    latent,
    { latentDim: TILE_LATENT_DIM, hopLength: TILE_HOP, tiles },
    equivariantDecoder,
  );
  const single = await equivariantDecoder(latent, frames);
  // ビット一致（f32 のビット列で比べる — 値の等価では NaN / -0 の取り違えが素通りする）。
  assertEquals(new Uint32Array(waveform.buffer), new Uint32Array(single.buffer));
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

// タイル decode の幾何と貼り合わせ（VAE tiling — ADR 0033）。
//
// 実 GPU / 実重みでの参照突合は E2E（P3 波 2）。ここは GPU も資産も要らない純関数側で、
// **幾何の不変条件**（固定形の decoder が食える配置であること）と**貼り合わせの解析解**を
// 押さえる。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ANIMA_SPATIAL_COMPRESSION } from "../src/anima/dit-tokens.ts";
import {
  MAX_RESOLUTION_SIDE,
  MIN_RESOLUTION_SIDE,
  RESOLUTION_GRANULARITY,
} from "../src/anima/resolution.ts";
import {
  assembleTiles,
  blendExtentAt,
  decodeTiled,
  latentTile,
  MIN_TILE_OVERLAP_LATENT,
  planTileAxis,
  planVaeTiling,
  tileCount,
} from "../src/anima/tiling.ts";

Deno.test("planTileAxis: 1024px（latent 128）は 3 タイル・開始 [0,32,64]・ブレンド 256px", () => {
  // 旧規則（整除スナップ）と同じ配置になる辺。**出荷解像度のバイト不変**がここに乗って
  // いるので、値は literal で凍結する（1824×1248 の枚数変更で巻き添えにしない）。
  const axis = planTileAxis(128, 64, 8);
  assertEquals([...axis.starts], [0, 32, 64]);
  assertEquals([blendExtentAt(axis, 8, 1), blendExtentAt(axis, 8, 2)], [256, 256]);
});

/**
 * 受理集合の全 97 辺（512〜2048px・刻み 16px）の開始位置（latent）— `[辺, starts]`。
 *
 * 下の不変条件ループは丸めの**寄せ方**を決めない（`Math.round` を `Math.floor` に替えても
 * 先頭 0 / 末尾スナップ / 本数 / 重なり下限 / 間隔のばらつき ≤ 1 は全て成立する）。配置が
 * 1 latent ずれても例外は出ず、継ぎ目のブレンド位置が動いて絵が滲むだけなので、**値で**凍結
 * する。e2e の PNG 門も拾えない — 実 GPU が回す 1024 系は span が本数−1 で割り切れて丸めが
 * 恒等になる辺だから。
 *
 * 出所は旧規則（整除スナップ）との全数突合: 本数が変わるのは 24 辺だけで
 * `anima_resolution_test.ts` の `RECOUNTED_SIDES` と一致し、残る 73 辺は開始位置まで旧規則と
 * 同一（= 出荷バイト不変）。`i·span/(本数−1)` がちょうど半分になるのは 1872 / 1904 / 1936 /
 * 1968 / 2000 / 2032px の 6 辺で、そこは 0.5 切り上げに寄る（1872px = `(0,43,85,128,170)`）—
 * Python 側の整数式が偶数丸めへ退行すると (0,42,85,128,170) になり、鏡像の突合
 * （`tools/export-recipes/anima/tests/test_tiling.py` の `MIRRORED_STARTS`）が割れる。
 */
const AXIS_STARTS: readonly (readonly [number, readonly number[]])[] = [
  [512, [0]],
  [528, [0, 2]],
  [544, [0, 4]],
  [560, [0, 6]],
  [576, [0, 8]],
  [592, [0, 10]],
  [608, [0, 12]],
  [624, [0, 14]],
  [640, [0, 16]],
  [656, [0, 18]],
  [672, [0, 20]],
  [688, [0, 22]],
  [704, [0, 24]],
  [720, [0, 26]],
  [736, [0, 28]],
  [752, [0, 30]],
  [768, [0, 32]],
  [784, [0, 34]],
  [800, [0, 36]],
  [816, [0, 38]],
  [832, [0, 40]],
  [848, [0, 42]],
  [864, [0, 44]],
  [880, [0, 46]],
  [896, [0, 48]],
  [912, [0, 50]],
  [928, [0, 52]],
  [944, [0, 54]],
  [960, [0, 56]],
  [976, [0, 29, 58]],
  [992, [0, 30, 60]],
  [1008, [0, 31, 62]],
  [1024, [0, 32, 64]],
  [1040, [0, 33, 66]],
  [1056, [0, 34, 68]],
  [1072, [0, 35, 70]],
  [1088, [0, 36, 72]],
  [1104, [0, 37, 74]],
  [1120, [0, 38, 76]],
  [1136, [0, 39, 78]],
  [1152, [0, 40, 80]],
  [1168, [0, 41, 82]],
  [1184, [0, 42, 84]],
  [1200, [0, 43, 86]],
  [1216, [0, 44, 88]],
  [1232, [0, 45, 90]],
  [1248, [0, 46, 92]],
  [1264, [0, 47, 94]],
  [1280, [0, 48, 96]],
  [1296, [0, 49, 98]],
  [1312, [0, 50, 100]],
  [1328, [0, 51, 102]],
  [1344, [0, 52, 104]],
  [1360, [0, 53, 106]],
  [1376, [0, 54, 108]],
  [1392, [0, 55, 110]],
  [1408, [0, 56, 112]],
  [1424, [0, 38, 76, 114]],
  [1440, [0, 39, 77, 116]],
  [1456, [0, 39, 79, 118]],
  [1472, [0, 40, 80, 120]],
  [1488, [0, 41, 81, 122]],
  [1504, [0, 41, 83, 124]],
  [1520, [0, 42, 84, 126]],
  [1536, [0, 43, 85, 128]],
  [1552, [0, 43, 87, 130]],
  [1568, [0, 44, 88, 132]],
  [1584, [0, 45, 89, 134]],
  [1600, [0, 45, 91, 136]],
  [1616, [0, 46, 92, 138]],
  [1632, [0, 47, 93, 140]],
  [1648, [0, 47, 95, 142]],
  [1664, [0, 48, 96, 144]],
  [1680, [0, 49, 97, 146]],
  [1696, [0, 49, 99, 148]],
  [1712, [0, 50, 100, 150]],
  [1728, [0, 51, 101, 152]],
  [1744, [0, 51, 103, 154]],
  [1760, [0, 52, 104, 156]],
  [1776, [0, 53, 105, 158]],
  [1792, [0, 53, 107, 160]],
  [1808, [0, 54, 108, 162]],
  [1824, [0, 55, 109, 164]],
  [1840, [0, 55, 111, 166]],
  [1856, [0, 56, 112, 168]],
  [1872, [0, 43, 85, 128, 170]],
  [1888, [0, 43, 86, 129, 172]],
  [1904, [0, 44, 87, 131, 174]],
  [1920, [0, 44, 88, 132, 176]],
  [1936, [0, 45, 89, 134, 178]],
  [1952, [0, 45, 90, 135, 180]],
  [1968, [0, 46, 91, 137, 182]],
  [1984, [0, 46, 92, 138, 184]],
  [2000, [0, 47, 93, 140, 186]],
  [2016, [0, 47, 94, 141, 188]],
  [2032, [0, 48, 95, 143, 190]],
  [2048, [0, 48, 96, 144, 192]],
];

Deno.test("planTileAxis: 受理集合の全 97 辺の開始位置を値で凍結する（丸めの寄せ方まで）", () => {
  // 表が受理集合を**覆っている**ことを先に見る。辺の下限 / 天井 / 刻みが動いたとき、表が
  // 黙って部分被覆になるのではなくここで割れる。
  const sides: number[] = [];
  for (
    let side = MIN_RESOLUTION_SIDE;
    side <= MAX_RESOLUTION_SIDE;
    side += RESOLUTION_GRANULARITY
  ) sides.push(side);
  assertEquals(AXIS_STARTS.map(([side]) => side), sides, "表が覆う辺");

  const tile = MIN_RESOLUTION_SIDE / ANIMA_SPATIAL_COMPRESSION;
  for (const [side, starts] of AXIS_STARTS) {
    const axis = planTileAxis(side / ANIMA_SPATIAL_COMPRESSION, tile, MIN_TILE_OVERLAP_LATENT);
    assertEquals([...axis.starts], [...starts], `${side}px の開始位置`);
  }
});

Deno.test("planTileAxis: latent 64（512px）は 1 タイルに縮退しブレンド対が無い", () => {
  const axis = planTileAxis(64, 64, 8);
  assertEquals([...axis.starts], [0]);
  assertThrows(() => blendExtentAt(axis, 8, 1), RangeError, "範囲外");
});

Deno.test("planTileAxis: どの latent 全長でも配置の不変条件が成り立つ", () => {
  // 182 / 228 / 242 は旧規則で本数が跳ねた辺（1456 / 1824 / 1936px）— 撤廃の対象そのもの。
  for (const extent of [64, 80, 96, 112, 128, 160, 182, 192, 228, 242, 256, 384, 512]) {
    const axis = planTileAxis(extent, 64, 8);
    const where = `extent=${extent}`;
    const span = extent - 64;
    assertEquals(axis.starts[0], 0, where);
    // 末端へのスナップ = 固定形の decoder が最後のタイルも食えることの条件。
    assertEquals(axis.starts.at(-1), span, `${where}: 末端へスナップ`);
    // 本数は「重なりの下限だけを制約にした最小」の解析式そのもの。安全側に倒して間隔 1 に
    // する実装も、旧規則の約数探索も、ここで割れる。
    assertEquals(
      axis.starts.length,
      span === 0 ? 1 : Math.ceil(span / (64 - 8)) + 1,
      `${where}: 本数`,
    );
    const gaps = axis.starts.slice(1).map((start, index) => start - axis.starts[index]);
    for (const [index, gap] of gaps.entries()) {
      assert(64 - gap >= 8, `${where}: 対 ${index} の重なり ${64 - gap} が下限 8 未満`);
      assertEquals(blendExtentAt(axis, 8, index + 1), (64 - gap) * 8, `${where}: 対 ${index}`);
    }
    // 丸め等間隔の実体 = 間隔（したがってブレンド幅）の差は高々 1 latent。上流同型の
    // 「固定 stride + 末尾だけスナップ」に退行すると最後の対だけ大きく開いてここで割れる。
    if (gaps.length > 0) {
      assert(
        Math.max(...gaps) - Math.min(...gaps) <= 1,
        `${where}: 間隔のばらつき ${gaps}`,
      );
    }
  }
});

Deno.test("planTileAxis: タイルより短い latent と広すぎる重なりは落とす", () => {
  assertThrows(() => planTileAxis(32, 64, 8), Error, "タイル幅");
  assertThrows(() => planTileAxis(128, 64, 64), Error, "重なり");
});

Deno.test("planVaeTiling: 縮尺は資産の入出力形から割り出し、軸は独立に配置する", () => {
  const geometry = planVaeTiling([1, 16, 128, 64], [1, 16, 64, 64], [1, 3, 512, 512]);
  assertEquals(geometry.scale, 8);
  assertEquals(geometry.channels, 16);
  assertEquals(geometry.rows.starts.length, 3);
  assertEquals(geometry.cols.starts.length, 1, "H = W でなくても軸ごとに独立");
  assertEquals(tileCount(geometry), 3);
  // 入口の受理集合（チャネル不一致 / 軸で揃わない縮尺 / B>1）。
  assertThrows(
    () => planVaeTiling([1, 8, 128, 128], [1, 16, 64, 64], [1, 3, 512, 512]),
    Error,
    "チャネル数",
  );
  assertThrows(
    () => planVaeTiling([1, 16, 128, 128], [1, 16, 64, 64], [1, 3, 512, 256]),
    Error,
    "縮尺",
  );
  assertThrows(
    () => planVaeTiling([2, 16, 128, 128], [1, 16, 64, 64], [1, 3, 512, 512]),
    Error,
    "batch=1",
  );
});

Deno.test("latentTile: 平面ごとに正しい矩形を切り出す（軸とチャネルの取り違え検出）", () => {
  // [1,2,4,4]。値は `channel*100 + y*10 + x` で全要素が識別可能。
  const geometry = planVaeTiling([1, 2, 4, 4], [1, 2, 2, 2], [1, 1, 2, 2], 1);
  assertEquals([...geometry.rows.starts], [0, 1, 2]);
  const latents = new Float32Array(2 * 16);
  for (let channel = 0; channel < 2; channel += 1) {
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) latents[channel * 16 + y * 4 + x] = channel * 100 + y * 10 + x;
    }
  }
  assertEquals([...latentTile(latents, geometry, 0, 0)], [0, 1, 10, 11, 100, 101, 110, 111]);
  assertEquals([...latentTile(latents, geometry, 2, 1)], [21, 22, 31, 32, 121, 122, 131, 132]);
  assertThrows(() => latentTile(latents, geometry, 3, 0), RangeError);
});

Deno.test("assembleTiles: 1 枚（縮退）は decode 出力の素の写し（ビット同一）", () => {
  const geometry = planVaeTiling([1, 2, 8, 8], [1, 2, 8, 8], [1, 3, 8, 8], 2);
  assertEquals(tileCount(geometry), 1);
  const decoded = new Float32Array(3 * 64);
  for (let index = 0; index < decoded.length; index += 1) decoded[index] = Math.sin(index) * 3;
  const assembled = assembleTiles([decoded], geometry);
  assertEquals(
    new Uint32Array(assembled.buffer, assembled.byteOffset, assembled.length),
    new Uint32Array(decoded.buffer, decoded.byteOffset, decoded.length),
  );
});

Deno.test("assembleTiles: 位置依存の decode 出力を線形ランプが解析解どおりに畳む", () => {
  // extent 16 / tile 8 / stride 4（重なり 4）で開始位置は 0,4,8。タイル内の値が行番号
  // `0..7` の傾斜のとき、貼り合わせ後は `0,1,2,3` → 中間は全て 4 → 末尾が `4,5,6,7`。
  // 重なりの値がタイルごとに**違う**ので、ブレンドの向き反転 / stride の off-by-one /
  // 末端タイルのスナップ落とし / 担当領域の取り違えが全てここで割れる（同じ値が重なる
  // 作りだと向きを反転しても緑のまま通る）。
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  assertEquals([...geometry.rows.starts], [0, 4, 8]);
  const rowRamp = new Float32Array(64);
  const colRamp = new Float32Array(64);
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      rowRamp[y * 8 + x] = y;
      colRamp[y * 8 + x] = x;
    }
  }
  const expected = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7];
  const rows = assembleTiles(Array.from({ length: 9 }, () => rowRamp), geometry);
  assertEquals([...Array.from({ length: 16 }, (_, y) => rows[y * 16])], expected, "行方向");
  const cols = assembleTiles(Array.from({ length: 9 }, () => colRamp), geometry);
  assertEquals([...cols.subarray(0, 16)], expected, "列方向");
});

Deno.test("assembleTiles: 末端まで覆う（stride 幅で切り詰めるだけだと欠ける）", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const flat = new Float32Array(64).fill(7);
  const image = assembleTiles(Array.from({ length: 9 }, () => flat), geometry);
  assertEquals(image.length, 16 * 16);
  assert(image.every((value) => value === 7), "覆われていない画素が残っている");
});

Deno.test("assembleTiles: 渡されたタイルを破壊しない（in-place ブレンドは写しの上）", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const tiles = Array.from({ length: 9 }, (_, index) => new Float32Array(64).fill(index + 1));
  const before = tiles.map((tile) => [...tile]);
  assembleTiles(tiles, geometry);
  assertEquals(tiles.map((tile) => [...tile]), before);
});

Deno.test("assembleTiles: タイル枚数と要素数の食い違いを落とす", () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const flat = new Float32Array(64);
  assertThrows(() => assembleTiles([flat], geometry), Error, "幾何の");
  assertThrows(
    () =>
      assembleTiles(
        [...Array.from({ length: 8 }, () => flat), new Float32Array(63)],
        geometry,
      ),
    Error,
    "1 枚目",
  );
});

Deno.test("decodeTiled: 行優先で全タイルを 1 度ずつ回す", async () => {
  const geometry = planVaeTiling([1, 1, 16, 16], [1, 1, 8, 8], [1, 1, 8, 8], 2);
  const visited: string[] = [];
  await decodeTiled(new Float32Array(256), geometry, (_tile, row, col) => {
    visited.push(`${row},${col}`);
    return Promise.resolve(new Float32Array(64));
  });
  assertEquals(visited, ["0,0", "0,1", "0,2", "1,0", "1,1", "1,2", "2,0", "2,1", "2,2"]);
});

Deno.test("decodeTiled: decode の返り値は所有権ごと受け取り、写しを挟まず畳む", async () => {
  const geometry = planVaeTiling([1, 2, 16, 16], [1, 2, 8, 8], [1, 1, 8, 8], 2);
  const colCount = geometry.cols.starts.length;
  // タイルごとに・タイル内でも値が違う（重なりの両側が同値だとブレンドが恒等になり、
  // 写しの有無も畳み方の違いも見えなくなる）。
  const makeTiles = () =>
    Array.from({ length: 9 }, (_, index) => {
      const tile = new Float32Array(64);
      for (let y = 0; y < 8; y += 1) {
        for (let x = 0; x < 8; x += 1) tile[y * 8 + x] = index * 100 + y * 10 + x;
      }
      return tile;
    });
  const handed = makeTiles();
  const pristine = makeTiles();

  const image = await decodeTiled(
    new Float32Array(2 * 256),
    geometry,
    (_tile, row, col) => Promise.resolve(handed[row * colCount + col]),
  );

  // 所有権を取っても畳み方は公開面と 1 ビットも変わらない。
  const expected = assembleTiles(pristine, geometry);
  assertEquals(
    new Uint32Array(image.buffer, image.byteOffset, image.length),
    new Uint32Array(expected.buffer, expected.byteOffset, expected.length),
  );
  // 渡した配列そのものが畳まれた側 = 所有権移転の契約。内部にまた写し層が戻るとここで割れる。
  assert(
    handed.some((tile, index) => tile.some((value, at) => value !== pristine[index][at])),
    "decode の返り値が in-place に畳まれていない（写しが挟まっている）",
  );
});

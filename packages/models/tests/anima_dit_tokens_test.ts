// S 形 DiT のホストグルー（patchify / unpatchify / rope 表）と rope 素表の読み手。
//
// **軸の取り違えは「実測形では対合になって偶然一致する」クラス**なので、ここのケースは patch の
// 高さ / 幅 / チャネル数を全部違う値にして、巡回長 3 以上の並べ替えでしか通らない形にしてある。
// 実重み・実 GPU での門は E2E（P3 波 2）。

import { assertEquals, assertThrows } from "@std/assert";
import {
  ditPatchGeometry,
  patchifyLatents,
  ropeTables,
  tokenCount,
  unpatchifyTokens,
} from "../src/anima/dit-tokens.ts";
import { parseRopeBase, type RopeBase, ropeWidth } from "../src/anima/rope-base.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

/** 非正方 patch（2×3）・チャネル 3 の幾何。C = ph = pw のどれも一致しない。 */
const TOKEN_GEOMETRY = { channels: 3, patchHeight: 2, patchWidth: 3 } as const;

Deno.test("ditPatchGeometry: グラフの入出力幅から patch とチャネル数を割り出す", () => {
  // Anima の実形（tokens 68 = 17·2·2 / 出力 64 = 2·2·16）。
  assertEquals(ditPatchGeometry(68, 64), { channels: 16, patchHeight: 2, patchWidth: 2 });
  // 差 = patch 体積、商 = チャネル数（呼び出し側に literal を置かないための割り出し）。
  assertEquals(ditPatchGeometry(45, 36), { channels: 4, patchHeight: 3, patchWidth: 3 });
});

Deno.test("ditPatchGeometry: 割り出せない幅は落とす（黙って正方 patch を仮定しない）", () => {
  assertThrows(() => ditPatchGeometry(64, 68), Error, "割り出せない");
  assertThrows(() => ditPatchGeometry(70, 64), Error, "割り出せない");
  // 差が平方数でない = pt=1・正方 patch の前提が崩れている。
  assertThrows(() => ditPatchGeometry(72, 64), Error, "平方数");
});

Deno.test("tokenCount: latent を patch で割った格子の積（S）", () => {
  assertEquals(tokenCount([1, 3, 4, 9], TOKEN_GEOMETRY), 2 * 3);
  // 1024px（latent 128・patch 2）は S = 4,096。
  assertEquals(
    tokenCount([1, 16, 128, 128], { channels: 16, patchHeight: 2, patchWidth: 2 }),
    4096,
  );
});

Deno.test("patchifyLatents: 最終次元は (c, ih, iw)・トークン添字は h·W'+w・padding は全ゼロ", () => {
  const { channels, patchHeight, patchWidth } = TOKEN_GEOMETRY;
  const [height, width] = [4, 9];
  const rows = height / patchHeight;
  const cols = width / patchWidth;
  const latents = Float32Array.from({ length: channels * height * width }, (_, i) => i + 1);

  const tokens = patchifyLatents(latents, [1, channels, height, width], TOKEN_GEOMETRY);

  const patchVolume = patchHeight * patchWidth;
  const tokenWidth = (channels + 1) * patchVolume;
  assertEquals(tokens.length, rows * cols * tokenWidth);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        for (let innerH = 0; innerH < patchHeight; innerH += 1) {
          for (let innerW = 0; innerW < patchWidth; innerW += 1) {
            const at = channel * height * width +
              (row * patchHeight + innerH) * width + col * patchWidth + innerW;
            assertEquals(
              tokens[
                (row * cols + col) * tokenWidth + channel * patchVolume +
                innerH * patchWidth + innerW
              ],
              latents[at],
              `c=${channel} h=${row}.${innerH} w=${col}.${innerW}`,
            );
          }
        }
      }
    }
  }
  // padding channel（最後の 1 本）は恒常ゼロ — パイプラインが渡すマスクがゼロだから。
  for (let token = 0; token < rows * cols; token += 1) {
    for (let inner = 0; inner < patchVolume; inner += 1) {
      assertEquals(tokens[token * tokenWidth + channels * patchVolume + inner], 0);
    }
  }
});

Deno.test("unpatchifyTokens: 最終次元は (ih, iw, c) — patchify の逆順ではない", () => {
  const { channels, patchHeight, patchWidth } = TOKEN_GEOMETRY;
  const [height, width] = [4, 9];
  const rows = height / patchHeight;
  const cols = width / patchWidth;
  const tokenWidth = patchHeight * patchWidth * channels;
  const tokens = Float32Array.from({ length: rows * cols * tokenWidth }, (_, i) => i + 1);

  const latents = unpatchifyTokens(tokens, [1, channels, height, width], TOKEN_GEOMETRY);

  assertEquals(latents.length, channels * height * width);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      for (let innerH = 0; innerH < patchHeight; innerH += 1) {
        for (let innerW = 0; innerW < patchWidth; innerW += 1) {
          for (let channel = 0; channel < channels; channel += 1) {
            const at = channel * height * width +
              (row * patchHeight + innerH) * width + col * patchWidth + innerW;
            assertEquals(
              latents[at],
              tokens[
                (row * cols + col) * tokenWidth +
                (innerH * patchWidth + innerW) * channels + channel
              ],
              `c=${channel} h=${row}.${innerH} w=${col}.${innerW}`,
            );
          }
        }
      }
    }
  }
});

Deno.test("patchify / unpatchify: 形の食い違いは落とす（沈黙誤値にしない）", () => {
  const shape = [1, TOKEN_GEOMETRY.channels, 4, 9];
  assertThrows(
    () => patchifyLatents(new Float32Array(4), shape, TOKEN_GEOMETRY),
    Error,
    "要素数",
  );
  assertThrows(
    () => patchifyLatents(new Float32Array(36), [1, 3, 5, 9], TOKEN_GEOMETRY),
    Error,
    "割り切れない",
  );
  assertThrows(
    () => patchifyLatents(new Float32Array(36), [2, 3, 4, 9], TOKEN_GEOMETRY),
    Error,
    "[1,C,H,W] 前提",
  );
  assertThrows(
    () => unpatchifyTokens(new Float32Array(7), shape, TOKEN_GEOMETRY),
    Error,
    "要素数",
  );
});

/** 軸ごとに違う幅・違う値を持つ素表（ブロック順と読み出し位置の取り違えが必ず露見する形）。 */
const syntheticRopeBase = (rows: number): RopeBase => {
  const widths: readonly [number, number, number] = [1, 2, 3];
  const make = (axis: number, kind: number): Float32Array =>
    Float32Array.from(
      { length: rows * widths[axis] },
      (_, i) =>
        (axis + 1) * 1000 + kind * 100 + Math.floor(i / widths[axis]) * 10 + i % widths[axis],
    );
  return {
    rows,
    widths,
    cos: [make(0, 0), make(1, 0), make(2, 0)],
    sin: [make(0, 1), make(1, 1), make(2, 1)],
  };
};

Deno.test("ropeTables: 1 行は [t,h,w,t,h,w]・位置は t=0 / h=行 / w=列", () => {
  const base = syntheticRopeBase(8);
  const geometry = { channels: 2, patchHeight: 2, patchWidth: 2 } as const;
  const [height, width] = [6, 8];
  const rows = height / geometry.patchHeight;
  const cols = width / geometry.patchWidth;

  const { cos, sin } = ropeTables(base, [1, geometry.channels, height, width], geometry);

  const rowWidth = ropeWidth(base);
  assertEquals(rowWidth, 12, "1 行の幅は 2·(1+2+3)");
  assertEquals(cos.length, rows * cols * rowWidth);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const offsets = [0, row * base.widths[1], col * base.widths[2]];
      const expectedCos: number[] = [];
      const expectedSin: number[] = [];
      for (let half = 0; half < 2; half += 1) {
        for (let axis = 0; axis < 3; axis += 1) {
          const span = base.widths[axis];
          for (let index = 0; index < span; index += 1) {
            expectedCos.push(base.cos[axis][offsets[axis] + index]);
            expectedSin.push(base.sin[axis][offsets[axis] + index]);
          }
        }
      }
      const at = (row * cols + col) * rowWidth;
      assertEquals([...cos.slice(at, at + rowWidth)], expectedCos, `cos (${row}, ${col})`);
      assertEquals([...sin.slice(at, at + rowWidth)], expectedSin, `sin (${row}, ${col})`);
    }
  }
});

Deno.test("ropeTables: 素表の行数を超える解像度は落とす（モデル側の位置表の天井）", () => {
  const base = syntheticRopeBase(3);
  const geometry = { channels: 2, patchHeight: 2, patchWidth: 2 } as const;

  assertThrows(() => ropeTables(base, [1, 2, 8, 4], geometry), Error, "天井");
});

const ropeTable = (rows: number, width: number, seed: number) => ({
  dtype: "F32",
  shape: [rows, width],
  data: f32Bytes(Array.from({ length: rows * width }, (_, i) => seed + i)),
});

Deno.test("parseRopeBase: 6 本の素表を軸順（t → h → w）で読む", () => {
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 100) },
    { name: "sin_t", ...ropeTable(4, 1, 200) },
    { name: "cos_h", ...ropeTable(4, 2, 300) },
    { name: "sin_h", ...ropeTable(4, 2, 400) },
    { name: "cos_w", ...ropeTable(4, 3, 500) },
    { name: "sin_w", ...ropeTable(4, 3, 600) },
  ]);
  const base = parseRopeBase(buffer);
  assertEquals(base.rows, 4);
  assertEquals(base.widths, [1, 2, 3]);
  assertEquals(ropeWidth(base), 12);
  // 軸の並びが t → h → w であること（cos_h を w の位置で読む取り違えはここで割れる）。
  assertEquals([...base.cos[0]], [100, 101, 102, 103]);
  assertEquals([...base.cos[1]].slice(0, 2), [300, 301]);
  assertEquals([...base.sin[2]].slice(0, 3), [600, 601, 602]);
});

Deno.test("parseRopeBase: 軸ごとの行数が揃わない素表は落とす（黙って別軸の行を読ませない）", () => {
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 0) },
    { name: "sin_t", ...ropeTable(4, 1, 0) },
    { name: "cos_h", ...ropeTable(4, 2, 0) },
    { name: "sin_h", ...ropeTable(4, 2, 0) },
    // w だけ行数が違う = 「h の行を w の表から読む」取り違えが範囲内に収まる形。
    { name: "cos_w", ...ropeTable(3, 2, 0) },
    { name: "sin_w", ...ropeTable(3, 2, 0) },
  ]);

  assertThrows(() => parseRopeBase(buffer), Error, "行数");
});

Deno.test("parseRopeBase: 想定外のテンソルが混ざった素表は落とす", () => {
  // 余分な 1 本（`karume-anima-extra` のミラーを焼き直すときに踏みうる形）。読み飛ばすと
  // 「素表と別物の表を同じ資産に同居させたまま気づかない」経路ができる。
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 0) },
    { name: "sin_t", ...ropeTable(4, 1, 0) },
    { name: "cos_h", ...ropeTable(4, 2, 0) },
    { name: "sin_h", ...ropeTable(4, 2, 0) },
    { name: "cos_w", ...ropeTable(4, 3, 0) },
    { name: "sin_w", ...ropeTable(4, 3, 0) },
    { name: "cos_x", ...ropeTable(4, 1, 0) },
  ]);

  assertThrows(() => parseRopeBase(buffer), Error, "想定外のテンソル 'cos_x'");
});

Deno.test("parseRopeBase: 同一軸の cos / sin で列幅が違えば落とす（行数一致では捕まらない）", () => {
  // 行数は全軸 4 で揃えたまま、h 軸の対だけ列幅を割る。行数の門は通り抜けるので、対ごとの
  // shape 突合が無いと「cos は 2 列 / sin は 3 列」の表が黙って通る。
  const buffer = buildSafetensors([
    { name: "cos_t", ...ropeTable(4, 1, 0) },
    { name: "sin_t", ...ropeTable(4, 1, 0) },
    { name: "cos_h", ...ropeTable(4, 2, 0) },
    { name: "sin_h", ...ropeTable(4, 3, 0) },
    { name: "cos_w", ...ropeTable(4, 3, 0) },
    { name: "sin_w", ...ropeTable(4, 3, 0) },
  ]);

  assertThrows(() => parseRopeBase(buffer), Error, "cos / sin で shape が違う");
});

Deno.test("parseRopeBase: 欠けた表・非 F32・rank 違いは落とす", () => {
  assertThrows(
    () =>
      parseRopeBase(buildSafetensors([
        { name: "cos_t", ...ropeTable(2, 1, 0) },
        { name: "sin_t", ...ropeTable(2, 1, 0) },
      ])),
    Error,
    "'cos_h' が無い",
  );
  assertThrows(
    () =>
      parseRopeBase(buildSafetensors([
        { name: "cos_t", dtype: "F16", shape: [2, 1], data: new Uint8Array(4) },
      ])),
    Error,
    "F32 が必要",
  );
  assertThrows(
    () =>
      parseRopeBase(buildSafetensors([
        { name: "cos_t", dtype: "F32", shape: [2], data: f32Bytes([0, 0]) },
      ])),
    Error,
    "rank が 2 でない",
  );
});

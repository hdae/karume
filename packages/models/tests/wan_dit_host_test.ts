// Wan2.1 の DiT のホスト関数（S 形グラフの外の 4 段）を、上流の値の fixture と突き合わせる。
// GPU も実重みも要らない純関数のテスト（実 GPU の通しは e2e_wan_dit_test.ts）。
//
// fixture は `fixtures/wan-dit/`（生成: tools/export-recipes/wan/dit_host_fixture.py — 期待値は上流の
// 演算の出力で、式の写しではない）。格子は F'·H'·W' = 21·3·5 で 3 軸とも違う値にしてある — 軸の
// 取り違えは正方の格子では値が一致して隠れる。F' = 21 は 81 フレームの潜在の時間軸（ADR 0118 段 8）。
//
// - patchify / unpatchify / RoPE の表: **ビット一致**（Uint32）。どれもデータ移動だけ。
// - `timesteps_proj`: atol（下の `TIMESTEPS_PROJ_ATOL` の doc に実測）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
  wanTokenWidth,
} from "../src/wan/dit-tokens.ts";
import { parseWanRopeBase, wanRopeTables, wanRopeWidth } from "../src/wan/dit-rope.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { buildSafetensors, f32Bytes, type TensorSpec } from "./helpers/safetensors.ts";

/**
 * `timesteps_proj` の TS 実装と上流（torch CPU f32）の許容差（絶対）。
 *
 * 実測（2026-10-02・timestep 0〜1,000 の全 1,001 通り × 256 = 256,256 要素）: 不一致 14,701 件
 * （5.7%）・**最大絶対差 3.0517e-5**（t = 745・要素 140 = sin 側の周波数 12）。
 *
 * 大きい差の出所は `exp` の 1 ULP だけ: 周波数 128 本のうち 2 本（添字 12 と 106）で torch の
 * `exp`（SLEEF の 1.0 ULP 実装）と `Math.exp` を f32 に丸めた値が 1 ULP 割れる（添字 12 は f64 の真値
 * 0.42169649837 に対し torch 0.42169651389・TS 0.42169648409 — TS が正しい丸め）。角度 `t·freq` が
 * 300 を超えると 1 ULP が 3e-5 になり、その差が sin / cos にそのまま出る。それ以外は sin / cos の
 * 1 ULP（6e-8 級）。Anima の前例 6e-7 より大きいのは timestep が整数 0〜999 で角度が大きいため
 * （Anima の実測は 72 行で、割れる周波数の角度が小さかった）。
 *
 * atol 1.5e-4 は実測最悪の約 4.9 倍。実装の誤りの差は桁違いに大きい（下の故障注入: timestep の
 * 1 ずれ・cos / sin の前後反転は 1e-1〜1e0 級）。fixture の timestep は参照の設定（50 ステップ・
 * shift 3.0）の列に 0 / 1 / 745（全通りの最悪）を足したもの。
 */
const TIMESTEPS_PROJ_ATOL = 1.5e-4;

/** Wan2.1 T2V 1.3B の patch（`(1,2,2)`・潜在 16 チャネル — fixture の `host.json`）。 */
const WAN_GEOMETRY: WanPatchGeometry = {
  channels: 16,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

const FIXTURE_DIR = new URL("./fixtures/wan-dit/", import.meta.url);

type HostMeta = {
  readonly patch_size: readonly [number, number, number];
  readonly latent_shape: readonly [number, number, number, number];
  readonly rope_base_rows: number;
  readonly timesteps: readonly number[];
};

const readBuffer = async (name: string): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(new URL(name, FIXTURE_DIR));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

const meta = JSON.parse(
  await Deno.readTextFile(new URL("host.json", FIXTURE_DIR)),
) as HostMeta;
const host = parseSafetensors(await readBuffer("host.safetensors"));
const ropeBaseBuffer = await readBuffer("rope_base.safetensors");

const viewOf = (file: SafetensorsFile, key: string) => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`fixture に '${key}' が無い`);
  return view;
};

const floats = (file: SafetensorsFile, key: string): Float32Array => {
  const view = viewOf(file, key);
  if (view.dtype !== "F32") throw new Error(`fixture '${key}' が ${view.dtype}`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const shapeOf = (file: SafetensorsFile, key: string): readonly number[] => viewOf(file, key).shape;

/** 最初にビットが割れる要素の添字（一致なら -1）。長さが違えば 0。 */
const firstBitMismatch = (actual: Float32Array, expected: Float32Array): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

const maxAbsDiff = (actual: ArrayLike<number>, expected: ArrayLike<number>): number => {
  let worst = 0;
  for (let index = 0; index < expected.length; index += 1) {
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  return worst;
};

Deno.test("fixture: 上流の patch と潜在の形が TS の前提（Wan2.1 T2V 1.3B）と一致する", () => {
  assertEquals(meta.patch_size, [
    WAN_GEOMETRY.patchFrames,
    WAN_GEOMETRY.patchHeight,
    WAN_GEOMETRY.patchWidth,
  ]);
  assertEquals(meta.latent_shape[0], WAN_GEOMETRY.channels);
  assertEquals([...shapeOf(host, "patchify.latents")], [...meta.latent_shape]);
});

Deno.test("wanTokenGrid: 潜在 [C,F,H,W] を patch で割った格子と S = F'·H'·W'", () => {
  assertEquals(wanTokenGrid([16, 9, 60, 104], WAN_GEOMETRY), {
    frames: 9,
    rows: 30,
    cols: 52,
    count: 14_040,
  });
  assertEquals(wanTokenGrid([16, 21, 60, 104], WAN_GEOMETRY).count, 32_760);
});

Deno.test("wanTokenGrid: patch で割り切れない潜在・チャネル数違い・rank 違いは落とす", () => {
  assertThrows(() => wanTokenGrid([16, 3, 15, 16], WAN_GEOMETRY), Error, "割り切れない");
  assertThrows(() => wanTokenGrid([4, 3, 16, 16], WAN_GEOMETRY), Error, "チャネル数");
  assertThrows(() => wanTokenGrid([1, 16, 3, 16, 16], WAN_GEOMETRY), Error, "[C,F,H,W]");
});

Deno.test("patchifyLatents: 上流の patch 埋め込みの並び（1-hot conv3d）とビット一致する", () => {
  const tokens = patchifyLatents(
    floats(host, "patchify.latents"),
    meta.latent_shape,
    WAN_GEOMETRY,
  );
  const expected = floats(host, "patchify.tokens");
  assertEquals(firstBitMismatch(tokens, expected), -1, "patchify の並びが上流と割れる");
});

Deno.test("unpatchifyTokens: 上流の出口（reshape / permute）とビット一致する", () => {
  const latents = unpatchifyTokens(
    floats(host, "unpatchify.tokens"),
    meta.latent_shape,
    WAN_GEOMETRY,
  );
  const expected = floats(host, "unpatchify.latents");
  assertEquals(firstBitMismatch(latents, expected), -1, "unpatchify の並びが上流と割れる");
});

/** 3 軸と C が全部違う幾何（巡回長 3 以上の並べ替えでしか通らない形）。 */
const ODD_GEOMETRY: WanPatchGeometry = {
  channels: 2,
  patchFrames: 2,
  patchHeight: 3,
  patchWidth: 4,
};
const ODD_LATENT = [2, 4, 6, 8] as const;

Deno.test("patchify / unpatchify: 添字の式（独立オラクル）どおりに並べる", () => {
  const { channels, patchFrames, patchHeight, patchWidth } = ODD_GEOMETRY;
  const [, frames, height, width] = ODD_LATENT;
  const grid = wanTokenGrid(ODD_LATENT, ODD_GEOMETRY);
  const tokenWidth = wanTokenWidth(ODD_GEOMETRY);
  const latents = Float32Array.from({ length: channels * frames * height * width }, (_, i) => i);
  const tokens = patchifyLatents(latents, ODD_LATENT, ODD_GEOMETRY);
  const back = unpatchifyTokens(tokens, ODD_LATENT, ODD_GEOMETRY);
  for (let c = 0; c < channels; c += 1) {
    for (let f = 0; f < frames; f += 1) {
      for (let h = 0; h < height; h += 1) {
        for (let w = 0; w < width; w += 1) {
          const token = (Math.floor(f / patchFrames) * grid.rows + Math.floor(h / patchHeight)) *
              grid.cols + Math.floor(w / patchWidth);
          const inner = ((f % patchFrames) * patchHeight + (h % patchHeight)) * patchWidth +
            (w % patchWidth);
          const value = latents[((c * frames + f) * height + h) * width + w];
          const volume = patchFrames * patchHeight * patchWidth;
          // 入口は (c, pt, ph, pw)。
          assertEquals(tokens[token * tokenWidth + c * volume + inner], value);
          // 出口は (pt, ph, pw, c) — 入口のトークンを出口の並びとして読むと別の画素へ散る。
          assertEquals(
            back[((c * frames + f) * height + h) * width + w],
            tokens[
              token * tokenWidth + inner * channels + c
            ],
          );
        }
      }
    }
  }
});

Deno.test("patchify → 最終次元を (c,v) から (v,c) へ並べ替え → unpatchify は恒等（往復）", () => {
  const { channels } = ODD_GEOMETRY;
  const volume = wanTokenWidth(ODD_GEOMETRY) / channels;
  const [, frames, height, width] = ODD_LATENT;
  const latents = Float32Array.from(
    { length: channels * frames * height * width },
    (_, i) => Math.sin(i) * 3,
  );
  const tokens = patchifyLatents(latents, ODD_LATENT, ODD_GEOMETRY);
  const transposed = new Float32Array(tokens.length);
  for (let token = 0; token < tokens.length / (channels * volume); token += 1) {
    for (let c = 0; c < channels; c += 1) {
      for (let v = 0; v < volume; v += 1) {
        transposed[token * channels * volume + v * channels + c] =
          tokens[token * channels * volume + c * volume + v];
      }
    }
  }
  const back = unpatchifyTokens(transposed, ODD_LATENT, ODD_GEOMETRY);
  assertEquals(firstBitMismatch(back, latents), -1);
  // 並べ替えを省いた往復は恒等にならない（入口と出口の並びが別物であることの裏取り）。
  assert(firstBitMismatch(unpatchifyTokens(tokens, ODD_LATENT, ODD_GEOMETRY), latents) !== -1);
});

Deno.test("unpatchifyTokens: 要素数の食い違いは落とす", () => {
  assertThrows(
    () => unpatchifyTokens(new Float32Array(10), meta.latent_shape, WAN_GEOMETRY),
    Error,
    "要素数",
  );
  assertThrows(
    () => patchifyLatents(new Float32Array(10), meta.latent_shape, WAN_GEOMETRY),
    Error,
    "要素数",
  );
});

Deno.test("wanRopeTables: 素表からの並べ替えが上流 model.rope の表とビット一致する", () => {
  const base = parseWanRopeBase(ropeBaseBuffer);
  assertEquals(base.rows, meta.rope_base_rows);
  assertEquals(base.widths, [22, 21, 21]);
  assertEquals(wanRopeWidth(base), 128);
  const grid = wanTokenGrid(meta.latent_shape, WAN_GEOMETRY);
  // 81 フレームの潜在の時間軸 T' = 21 を覆う — t 軸の位置 0〜20 が全部、上流の表とビットで照合される。
  assertEquals(grid.frames, 21);
  const tables = wanRopeTables(base, grid);
  assertEquals([...shapeOf(host, "rope.cos")], [1, grid.count, 1, 128]);
  assertEquals(firstBitMismatch(tables.cos, floats(host, "rope.cos")), -1, "cos 表が割れる");
  assertEquals(firstBitMismatch(tables.sin, floats(host, "rope.sin")), -1, "sin 表が割れる");
});

Deno.test("wanRopeTables: h と w の位置を取り違えた格子は上流の表と割れる（非正方の格子）", () => {
  const base = parseWanRopeBase(ropeBaseBuffer);
  const grid = wanTokenGrid(meta.latent_shape, WAN_GEOMETRY);
  const swapped = wanRopeTables(base, { ...grid, rows: grid.cols, cols: grid.rows });
  assert(firstBitMismatch(swapped.cos, floats(host, "rope.cos")) !== -1);
});

Deno.test("wanRopeTables: 素表の行数を超える格子は落とす（上流の位置表の天井）", () => {
  const base = parseWanRopeBase(ropeBaseBuffer);
  assertThrows(
    () => wanRopeTables(base, { frames: 1, rows: base.rows + 1, cols: 1, count: base.rows + 1 }),
    Error,
    "天井",
  );
});

/** 正しい形の素表（各軸 2 行・幅 1 / 2 / 3）。異常系は 1 点だけ壊す。 */
const ropeSpecs = (): TensorSpec[] =>
  (["t", "h", "w"] as const).flatMap((axis, index) =>
    ["cos", "sin"].map((kind) => ({
      name: `${kind}_${axis}`,
      dtype: "F32",
      shape: [2, index + 1],
      data: f32Bytes(Array.from({ length: 2 * (index + 1) }, (_, i) => i)),
    }))
  );

Deno.test("parseWanRopeBase: 正しい形は軸ごとの行数・幅を返す", () => {
  const base = parseWanRopeBase(buildSafetensors(ropeSpecs()));
  assertEquals(base.rows, 2);
  assertEquals(base.widths, [1, 2, 3]);
  assertEquals(wanRopeWidth(base), 12);
});

Deno.test("parseWanRopeBase: 想定外のキー・欠け・行数違い・F32 以外は落とす", () => {
  const extra = [...ropeSpecs(), { name: "cos_x", dtype: "F32", shape: [1], data: f32Bytes([0]) }];
  assertThrows(() => parseWanRopeBase(buildSafetensors(extra)), Error, "想定外");
  assertThrows(
    () => parseWanRopeBase(buildSafetensors(ropeSpecs().filter((s) => s.name !== "sin_h"))),
    Error,
    "'sin_h' が無い",
  );
  const rows = ropeSpecs().map((spec) =>
    spec.name.endsWith("_w")
      ? { ...spec, shape: [1, 6], data: f32Bytes(new Array(6).fill(0)) }
      : spec
  );
  assertThrows(() => parseWanRopeBase(buildSafetensors(rows)), Error, "行数");
  const dtype = ropeSpecs().map((spec) =>
    spec.name === "cos_t"
      ? { ...spec, dtype: "I32", data: new Uint8Array(new Int32Array([0, 1]).buffer) }
      : spec
  );
  assertThrows(() => parseWanRopeBase(buildSafetensors(dtype)), Error, "F32");
});

/** fixture の timestep の列と、その上流の `timesteps_proj`（行 = timestep）。 */
const timestepRows = (): { readonly values: Int32Array; readonly proj: Float32Array } => {
  const view = viewOf(host, "timesteps.values");
  if (view.dtype !== "I32") throw new Error(`timesteps.values が ${view.dtype}`);
  return {
    values: new Int32Array(host.buffer, view.byteOffset, view.byteLength / 4),
    proj: floats(host, "timesteps.proj"),
  };
};

Deno.test("timestepsProj: 上流の timesteps_proj と atol 内で一致する（cos 先・sin 後）", () => {
  const { values, proj } = timestepRows();
  const width = shapeOf(host, "timesteps.proj")[1];
  assertEquals(width, 256);
  assertEquals([...values], [...meta.timesteps]);
  let worst = 0;
  values.forEach((timestep, row) => {
    const got = timestepsProj(timestep, width);
    worst = Math.max(worst, maxAbsDiff(got, proj.subarray(row * width, (row + 1) * width)));
  });
  assert(
    worst <= TIMESTEPS_PROJ_ATOL,
    `最大絶対差 ${worst} が atol ${TIMESTEPS_PROJ_ATOL} を超える`,
  );
});

Deno.test("timestepsProj: timestep の 1 ずれと cos / sin の前後反転は atol の外へ出る（故障注入）", () => {
  const { values, proj } = timestepRows();
  const width = 256;
  const half = width / 2;
  const row = values.indexOf(999);
  const expected = proj.subarray(row * width, (row + 1) * width);
  const shifted = maxAbsDiff(timestepsProj(998, width), expected);
  const correct = timestepsProj(999, width);
  const flipped = Float32Array.from(
    { length: width },
    (_, i) => correct[i < half ? half + i : i - half],
  );
  const swappedHalves = maxAbsDiff(flipped, expected);
  assert(shifted > 1e3 * TIMESTEPS_PROJ_ATOL, `1 ずれの差 ${shifted}`);
  assert(swappedHalves > 1e3 * TIMESTEPS_PROJ_ATOL, `前後反転の差 ${swappedHalves}`);
});

Deno.test("timestepsProj: 幅が正の偶数でない・timestep が負や非整数は落とす", () => {
  assertThrows(() => timestepsProj(10, 255), RangeError, "偶数");
  assertThrows(() => timestepsProj(10, 0), RangeError, "偶数");
  assertThrows(() => timestepsProj(-1, 256), RangeError, "整数");
  assertThrows(() => timestepsProj(0.5, 256), RangeError, "整数");
});

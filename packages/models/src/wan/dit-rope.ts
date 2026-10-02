/**
 * Wan2.1 の DiT の 3D RoPE 表のホスト側 — 容器の資産 `rope_base`（軸別の素表）の読み取りと、
 * S 形グラフの入力 `rope_cos` / `rope_sin [1,S,1,head_dim]` への並べ替え（ADR 0118 決定 3）。
 *
 * 素表は `transformer` 容器の**資産**（役割 `rope-base` — ADR 0109 決定 4）で、payload は safetensors
 * 形式。キーは `cos_t` / `sin_t` / `cos_h` / `sin_h` / `cos_w` / `sin_w`（Anima の `rope_base` と同じ
 * 綴り・同じ意味 = 軸 × 位置 × 周波数の表）で、各 `[rows, その軸の周波数の本数]` の F32。値は上流
 * `WanRotaryPosEmbed` の表そのもの（exporter が上流の出力から切り出す — `wan/dit_patch.py`）。
 *
 * ## 表は「計算」ではなく「素表からの並べ替え」
 *
 * MUST: cos / sin を TS で計算し**ない**。上流は float64 で角度と三角関数を計算してから f32 に落とす
 * ので、TS で式を写すと `pow` / 除算 / `cos` の実装差で最終ビットが割れうる。素表を並べ替えるだけなら
 * 上流の表とビット同一になる（Anima の `rope-base.ts` と同じ判断）。
 *
 * ## 並べ方（interleave 形）
 *
 * 1 行（1 トークン）は `[t ブロック, h ブロック, w ブロック]` の連結で、各ブロックは周波数 1 本ごとに
 * 同じ値を 2 つ並べる（上流の `repeat_interleave(2)` — 回転の対が隣接 2 要素 `(x[2i], x[2i+1])`）。
 * Anima の half-split 形（`[t, h, w, t, h, w]`）とは並びが違う。
 */

import { parseSafetensors } from "@karume/runtime";
import type { SafetensorsFile } from "@karume/runtime";
import type { WanTokenGrid } from "./dit-tokens.ts";

/** 素表の軸（順序は t → h → w のブロック順）。 */
const ROPE_AXES = ["t", "h", "w"] as const;

/** F32 のみ受ける（素表は cos / sin の実数表で、他の格納形は上流に存在しない）。 */
const F32_DTYPE = "F32";

/** 軸ごとの cos / sin 素表（行 = 位置・列 = その軸の周波数）。 */
export type WanRopeBase = {
  /** 全軸で共通の行数（= 上流の位置表の天井 `rope_max_seq_len`）。 */
  readonly rows: number;
  /** 軸ごとの周波数の本数（`2 × (t + h + w)` が head_dim）。 */
  readonly widths: readonly [number, number, number];
  /** `[t, h, w]` の順に並べた cos 素表（行優先 `[rows, widths[axis]]`）。 */
  readonly cos: readonly [Float32Array, Float32Array, Float32Array];
  /** 同 sin。 */
  readonly sin: readonly [Float32Array, Float32Array, Float32Array];
};

type RawTable = {
  readonly data: Float32Array;
  readonly shape: readonly [number, number];
};

/** F32・rank 2 の表を 1 本引く（整列は `parseSafetensors` が保証済み — view はゼロコピー）。 */
const tableOf = (file: SafetensorsFile, name: string): RawTable => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`rope 素表に '${name}' が無い`);
  if (view.dtype !== F32_DTYPE) {
    throw new Error(`rope 素表 '${name}': 格納 dtype が ${view.dtype}（F32 が必要）`);
  }
  if (view.shape.length !== 2) throw new Error(`rope 素表 '${name}': rank が 2 でない`);
  const shape: readonly [number, number] = [view.shape[0], view.shape[1]];
  return { data: new Float32Array(file.buffer, view.byteOffset, shape[0] * shape[1]), shape };
};

/**
 * 資産 `rope_base` の payload（safetensors 形式）を読む。
 *
 * MUST: キーがちょうど 6 本であること・行数が全軸で揃っていることを見る。揃っていないと「h の行を
 * w の表から読む」形の取り違えが範囲内に収まって黙って通る。
 */
export const parseWanRopeBase = (buffer: ArrayBuffer): WanRopeBase => {
  const file = parseSafetensors(buffer);
  // MUST: モジュールスコープの const に持たない（横断不変条件「全モジュール副作用ゼロ」）。
  const expectedKeys = new Set(ROPE_AXES.flatMap((axis) => [`cos_${axis}`, `sin_${axis}`]));
  for (const name of file.tensors.keys()) {
    if (!expectedKeys.has(name)) throw new Error(`rope 素表に想定外のテンソル '${name}' がある`);
  }
  const tables = ROPE_AXES.map((axis) => {
    const cos = tableOf(file, `cos_${axis}`);
    const sin = tableOf(file, `sin_${axis}`);
    if (cos.shape[0] !== sin.shape[0] || cos.shape[1] !== sin.shape[1]) {
      throw new Error(`rope 素表 ${axis} の cos / sin で shape が違う`);
    }
    return { axis, cos, sin };
  });
  const rows = tables[0].cos.shape[0];
  for (const { axis, cos } of tables) {
    if (cos.shape[0] !== rows) {
      throw new Error(`rope 素表 ${axis} の行数 ${cos.shape[0]} が他軸の ${rows} と違う`);
    }
  }
  const [t, h, w] = tables;
  return {
    rows,
    widths: [t.cos.shape[1], h.cos.shape[1], w.cos.shape[1]],
    cos: [t.cos.data, h.cos.data, w.cos.data],
    sin: [t.sin.data, h.sin.data, w.sin.data],
  };
};

/** RoPE 表 1 行の幅（= attention の head_dim）。 */
export const wanRopeWidth = (base: WanRopeBase): number =>
  2 * (base.widths[0] + base.widths[1] + base.widths[2]);

/**
 * `rope_cos` / `rope_sin`（`[1,S,1,head_dim]` を平坦化した `S·head_dim`）を素表から組む。
 *
 * トークン `(f, h, w)`（添字 `(f·H' + h)·W' + w`）の行は、t ブロックが位置 f・h ブロックが位置 h・
 * w ブロックが位置 w の素表の行を、周波数 1 本ごとに 2 回ずつ並べたもの（モジュール doc）。
 * MUST: F' / H' / W' のどれかが素表の行数を超える格子は落とす（上流でも位置表の天井を超えて組めない）。
 */
export const wanRopeTables = (
  base: WanRopeBase,
  grid: WanTokenGrid,
): { readonly cos: Float32Array<ArrayBuffer>; readonly sin: Float32Array<ArrayBuffer> } => {
  const positions = [grid.frames, grid.rows, grid.cols];
  if (positions.some((extent) => extent > base.rows)) {
    throw new Error(
      `rope 素表の行数 ${base.rows} では F'=${grid.frames} / H'=${grid.rows} / W'=${grid.cols} を` +
        "組めない（上流の位置表の天井を超えている）",
    );
  }
  const rowWidth = wanRopeWidth(base);
  const cos = new Float32Array(grid.count * rowWidth);
  const sin = new Float32Array(grid.count * rowWidth);
  for (let frame = 0; frame < grid.frames; frame += 1) {
    for (let row = 0; row < grid.rows; row += 1) {
      for (let col = 0; col < grid.cols; col += 1) {
        const token = (frame * grid.rows + row) * grid.cols + col;
        const axisPositions = [frame, row, col];
        let at = token * rowWidth;
        for (let axis = 0; axis < ROPE_AXES.length; axis += 1) {
          const span = base.widths[axis];
          const from = axisPositions[axis] * span;
          for (let index = 0; index < span; index += 1) {
            const cosValue = base.cos[axis][from + index];
            const sinValue = base.sin[axis][from + index];
            cos[at] = cosValue;
            cos[at + 1] = cosValue;
            sin[at] = sinValue;
            sin[at + 1] = sinValue;
            at += 2;
          }
        }
      }
    }
  }
  return { cos, sin };
};

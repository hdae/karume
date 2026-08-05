/**
 * `rope_base.safetensors`（軸別 rope 素表）の読み取り。
 *
 * 素表は IR コンテナではない**素の safetensors**（`transformer` の `extras.rope_base` —
 * ADR 0038 §2）なので `openModel` では開けない。ファイルの解析は `@karume/runtime` 公開面の
 * `parseSafetensors`（厳格リーダ — データ節の被覆・整列・dtype はそちらで検査済み）に委ね、
 * この層は「F32・rank2 の表を軸ごとに引き、行数を突き合わせる」用途特化の検査だけを持つ
 * （DECIDED: 二重実装の解消として runtime 公開面へ載せた — ADR 0008 追記 2026-08-05）。
 *
 * ## rope 表は「計算」ではなく「素表からの並べ替え」
 *
 * MUST: cos / sin を TS で計算し**ない**。`torch` の f32 三角関数は正しく丸めた値と 1 ulp
 * ずれることがあり（実測: 位置 × 周波数 8,192 通りのうち cos 472 件 / sin 231 件）、JS の
 * `Math.cos` は f64 で正しく丸まるので**必ず食い違う**。静的グラフには torch の値が焼かれて
 * いる以上、S 形とのビット同一は「軸ごとの素表を焼いて並べ替えるだけ」でしか成立しない。
 *
 * 素表は**解像度に依らない**（軸 × 位置の表なので、任意の H' / W' を組める）。行数は上流の
 * `seq = arange(max(max_size))` の長さ = モデル側の位置表の天井そのもの。
 */

import { parseSafetensors } from "@karume/runtime";
import type { SafetensorsFile } from "@karume/runtime";

/** 素表のテンソルキー（順序は t → h → w のブロック順）。 */
const ROPE_AXES = ["t", "h", "w"] as const;

/** F32 のみ受ける（素表は cos / sin の実数表で、他の格納形は上流に存在しない）。 */
const F32_DTYPE = "F32";

/** 軸ごとの cos / sin 素表（行 = 位置・列 = その軸のブロック幅）。 */
export type RopeBase = {
  /** 全軸で共通の行数（= モデル側の位置表の天井）。 */
  readonly rows: number;
  /** 軸ごとのブロック幅（`t + h + w` の 2 倍が head_dim）。 */
  readonly widths: readonly [number, number, number];
  /** `[t, h, w]` の順に並べた cos 素表。 */
  readonly cos: readonly Float32Array[];
  /** 同 sin。 */
  readonly sin: readonly Float32Array[];
};

type RawTable = {
  readonly data: Float32Array;
  readonly shape: readonly [number, number];
};

/** F32・rank2 の表を 1 本引く（整列は `parseSafetensors` が保証済み — view はゼロコピー）。 */
const tableOf = (file: SafetensorsFile, name: string): RawTable => {
  const view = file.tensors.get(name);
  if (view === undefined) throw new Error(`rope 素表に '${name}' が無い`);
  if (view.dtype !== F32_DTYPE) {
    throw new Error(`rope 素表 '${name}': 格納 dtype が ${view.dtype}（F32 が必要）`);
  }
  if (view.shape.length !== 2) {
    throw new Error(`rope 素表 '${name}': rank が 2 でない`);
  }
  const shape: readonly [number, number] = [view.shape[0], view.shape[1]];
  return {
    data: new Float32Array(file.buffer, view.byteOffset, shape[0] * shape[1]),
    shape,
  };
};

/**
 * `rope_base.safetensors` を読む。
 *
 * MUST: 行数が全軸で揃っていることを見る。揃っていないと「h の行を w の表から読む」形の
 * 取り違えが範囲内に収まって黙って通る。
 */
export const parseRopeBase = (buffer: ArrayBuffer): RopeBase => {
  const file = parseSafetensors(buffer);
  // MUST: モジュールスコープの const に持たない（横断不変条件「全モジュール副作用ゼロ =
  // import 時実行禁止」— barrel 経由 tree-shaking の成立条件。png.ts の buildCrcTable と同じ
  // 理由）。素表のテンソルは 6 本だけで、呼び出しごとの構築は無視できる。
  const expectedKeys = new Set(ROPE_AXES.flatMap((axis) => [`cos_${axis}`, `sin_${axis}`]));
  for (const name of file.tensors.keys()) {
    if (!expectedKeys.has(name)) throw new Error(`rope 素表に想定外のテンソル '${name}' がある`);
  }
  const cos: Float32Array[] = [];
  const sin: Float32Array[] = [];
  const widths: number[] = [];
  let rows: number | undefined;
  for (const axis of ROPE_AXES) {
    const cosTable = tableOf(file, `cos_${axis}`);
    const sinTable = tableOf(file, `sin_${axis}`);
    if (cosTable.shape[0] !== sinTable.shape[0] || cosTable.shape[1] !== sinTable.shape[1]) {
      throw new Error(`rope 素表 ${axis} の cos / sin で shape が違う`);
    }
    if (rows !== undefined && cosTable.shape[0] !== rows) {
      throw new Error(`rope 素表 ${axis} の行数 ${cosTable.shape[0]} が他軸の ${rows} と違う`);
    }
    rows = cosTable.shape[0];
    widths.push(cosTable.shape[1]);
    cos.push(cosTable.data);
    sin.push(sinTable.data);
  }
  return { rows: rows ?? 0, widths: [widths[0], widths[1], widths[2]], cos, sin };
};

/** rope 表 1 行の幅（= attention の head_dim）。 */
export const ropeWidth = (base: RopeBase): number =>
  2 * (base.widths[0] + base.widths[1] + base.widths[2]);

/** 軸の並び（`ropeTables` が読み出し位置を組むのに使う）。 */
export const ROPE_AXIS_COUNT = ROPE_AXES.length;

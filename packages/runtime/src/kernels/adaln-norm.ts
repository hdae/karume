/**
 * adaptive layer norm（DiT の変調）の融合カーネル。`layer_norm → (1 + scale) → mul → add` の
 * 4 ノードを 1 dispatch へ畳む（融合ルール `adaln` の実体 — src/runtime/fusion.ts）。
 *
 * 置換元の primitive 列は 3 本の中間 f32 を storage へ書いて次 dispatch が読み戻す:
 * `t = layer_norm(x)` → storage、`s = scale + 1` → storage、`p = t · s` → storage、最後に
 * `p + shift`。融合後に素直に 1 式で書くと `t · s + shift` も `t · (scale + 1)` も **fma 縮約の
 * 教科書的な形**で、1 丸め / 2 丸めの違いがそのままビット差になる。そこで `s` と `p` の
 * ビット列を workgroup `u32` へ書いて barrier 後に読み戻し、素の列と同じ位置に明示的な
 * materialization 点を 2 段残す（silu.ts と同じ手筋）。
 *
 * `t` にも storage 境界があるが、**3 段目は要らない**: 後段は乗算で `a·b+c` の形にならず、
 * 縮約が跨げない（WGSL が許すのは縮約であって再結合ではないので `(u·w+b)·s` を
 * `u·w·s + b·s` へ開くこともできない）。`t` の式**内部**の縮約判断は、素の layer_norm と
 * 同一文字列を共有していることで揃う。
 *
 * **この「丸め障壁」は WGSL 仕様の保証ではない** — 仕様は浮動小数の fusion（fma への縮約）を
 * 許すだけで、workgroup memory 往復を最適化障壁として尊重することは要求していない。有限値の
 * ビット一致はバックエンドごとの実 GPU A/B（tests/gpu_adaln_fusion_test.ts）で採用門にした
 * 実測事実であり、**バックエンド更新で PNG sha256 門が割れたら、まずここを疑うこと**
 * （融合を外して primitive 列に戻せば切り分けられる）。WGSL が規定しない NaN payload に
 * ついては分類の一致だけを契約にする。
 *
 * MUST: 正規化の本体（行統計と affine）は素の layer_norm と**同一文字列**を共有する
 * （{@link LAYER_NORM_ROW_STATS_WGSL} / {@link LAYER_NORM_AFFINE_WGSL}）。別々に書くと
 * primitive と融合版で縮約順が割れうる。共有の代償として識別子が固定される — 変調の
 * scale ベクトルを `scale_vec` と呼ぶのは、共有ブロックが `scale` を `1/dim` に使っているため。
 * MUST: barrier は workgroup 一様な制御流の中だけに置く。行ループ（`row = wid.x` の
 * grid-stride）と block ループ（`0..blocks`）はどちらも一様で、layer_norm が既に同じ形の
 * 行ループの中で barrier を回している。出力を `o = lid` の while で回すと lane ごとに
 * 反復数が変わって barrier を置けないので、silu と同じ block ループへ組み替えてある。
 */

import {
  LAYER_NORM_AFFINE_WGSL,
  LAYER_NORM_ROW_STATS_WGSL,
  LAYER_NORM_WORKGROUP_SIZE,
} from "./layer-norm.ts";

/**
 * uniform の Params（rows / dim / eps の 16 バイト）。**素の layer_norm と同一の契約**なので
 * 生成器ごと共有する（eps を f32 のビット列として載せる MUST も込み）。
 */
export { layerNormParams as adalnNormParams } from "./layer-norm.ts";

/**
 * MUST: 素の layer_norm と同じ値。共有する行統計ブロックが `scratch` の長さと木構造縮約の
 * 段数をこの定数から展開しているので、片方だけ動かすと WGSL が壊れる（か、黙って別の
 * 縮約順になる）。
 */
export const ADALN_NORM_WORKGROUP_SIZE = LAYER_NORM_WORKGROUP_SIZE;

export const ADALN_NORM_KEY = `adaln_norm:v1:lastdim:f32:wg${ADALN_NORM_WORKGROUP_SIZE}`;

export const ADALN_NORM_WGSL: string =
  `// karume adaLN (layer_norm -> 1 + scale -> mul -> add, f32, staged 変調)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> scale_vec: array<f32>;
@group(0) @binding(5) var<storage, read> one: array<f32>;
@group(0) @binding(6) var<storage, read> shift: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${ADALN_NORM_WORKGROUP_SIZE}>;
var<workgroup> staged: array<u32, ${ADALN_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${ADALN_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  // NOTE: 切り上げを (dim + 255) / 256 と書かない — dim = u32::MAX で桁溢れする。
  let blocks = dim / ${ADALN_NORM_WORKGROUP_SIZE}u +
    select(0u, 1u, dim % ${ADALN_NORM_WORKGROUP_SIZE}u != 0u);
  var row = wid.x;
  while (row < params.rows) {
${LAYER_NORM_ROW_STATS_WGSL}

    var block = 0u;
    while (block < blocks) {
      let o = block * ${ADALN_NORM_WORKGROUP_SIZE}u + lid;
      // ③ 変調係数 1 + scale の実体化（素の列では add ノードが storage へ書く点）
      if (o < dim) {
        staged[lid] = bitcast<u32>(scale_vec[o] + one[0u]);
      }
      workgroupBarrier();
      // ④ 正規化（素の layer_norm と同一式）→ 乗算。素の mul と同じ位置で丸める。
      if (o < dim) {
        let modulation = bitcast<f32>(staged[lid]);
        staged[lid] = bitcast<u32>((${LAYER_NORM_AFFINE_WGSL}) * modulation);
      }
      workgroupBarrier();
      // ⑤ shift の加算（素の add ノード）
      if (o < dim) {
        out[base + o] = bitcast<f32>(staged[lid]) + shift[o];
      }
      workgroupBarrier();
      block = block + 1u;
    }
    row = row + nwg.x;
  }
}
`;

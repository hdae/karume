/**
 * `mul(gate, x) → add(residual, ·)` の中間 f32 格納境界を保ったゲート付き残差の融合カーネル
 * （融合ルール `gatedResidual` の実体 — src/runtime/fusion.ts）。ゲートは行方向 broadcast
 * （`[1,…,1,dim]`）で、x / residual / 出力は同じ `[…, dim]`。
 *
 * 置換元の primitive 列は mul の結果を storage へ書き、次 dispatch の add が読み戻す。単に
 * `residual + gate * x` と書くと **fma 縮約の教科書的な形**（`a·b + c`）になり、1 丸め / 2 丸めの
 * 違いがそのままビット差になる。そこで積の f32 ビット列を workgroup `u32` へ書いて barrier 後に
 * 読み戻し、素の列と同じ位置に明示的な materialization 点を残す（silu.ts と同じ手筋）。
 *
 * **この「丸め障壁」は WGSL 仕様の保証ではない** — 仕様は浮動小数の fusion（fma への縮約）を
 * 許すだけで、workgroup memory 往復を最適化障壁として尊重することは要求していない。有限値の
 * ビット一致はバックエンドごとの実 GPU A/B（tests/gpu_gated_residual_fusion_test.ts）で採用門に
 * した実測事実であり、**バックエンド更新で PNG sha256 門が割れたら、まずここを疑うこと**
 * （融合を外して primitive 列に戻せば切り分けられる）。WGSL が規定しない NaN payload に
 * ついては分類の一致だけを契約にする。
 *
 * MUST: barrier は workgroup 一様な block loop の中だけに置く（2 本目は、次 block の書き込みを
 * 前 block の全 lane が読み終えてから始めるため）。subgroup / atomics / optional feature を
 * 使わないので WebGPU core の Metal / Dawn / wgpu 経路で使える。
 * MUST: 和の綴りは `residual + 積` 固定（実測の `add` は残差が slot 0 の 1 順序しかない）。積の
 * 綴りだけが 2 順序あるので、そちらはキーと WGSL の両方に残す。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/**
 * 置換元 mul の入力順（`gate-x` = `mul(gate, x)` / `x-gate` = その逆）。有限値では可換だが、
 * NaN payload はバックエンド差がありうる。
 */
export type GatedResidualOrder = "gate-x" | "x-gate";

export const GATED_RESIDUAL_WORKGROUP_SIZE = 256;

const canonicalizeOrder = (order: GatedResidualOrder): GatedResidualOrder => {
  if (order === "gate-x" || order === "x-gate") return order;
  throw new CodegenError(`gated_residual codegen: mul の入力順が不正（${order}）`);
};

export const gatedResidualKey = (order: GatedResidualOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  return `gated_residual:v1:${canonicalOrder}:f32:wg${GATED_RESIDUAL_WORKGROUP_SIZE}`;
};

/**
 * MUST: 積の入力順は WGSL とパイプラインキーの両方に残す。有限値では乗算は可換でも、NaN
 * payload の選ばれ方はバックエンドごとに違いうるので、片方だけで畳むと沈黙で別の列になる。
 * NOTE: block 数の切り上げを `(n + 255) / 256` と書かない — n = u32::MAX で桁溢れする。
 */
export const gatedResidualWgsl = (order: GatedResidualOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  // 置換元 add も mul の出力を別 dispatch で読み直す。レジスタ越しに持ち越さない形で書く。
  const product = canonicalOrder === "gate-x"
    ? "gate_for_mul * x_for_mul"
    : "x_for_mul * gate_for_mul";
  return `// karume gated residual (${canonicalOrder}, f32, staged product, full-write)
struct Params {
  n: u32,
  dim: u32,
  reserved0: u32,
  reserved1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> product_bits: array<u32, ${GATED_RESIDUAL_WORKGROUP_SIZE}>;

@compute @workgroup_size(${GATED_RESIDUAL_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let blocks = params.n / ${GATED_RESIDUAL_WORKGROUP_SIZE}u +
    select(0u, 1u, params.n % ${GATED_RESIDUAL_WORKGROUP_SIZE}u != 0u);
  var block = wid.x;
  while (block < blocks) {
    let i = block * ${GATED_RESIDUAL_WORKGROUP_SIZE}u + lid;
    if (i < params.n) {
      // ゲートは先行軸が全て 1 なので、行内の位置だけで引ける（素の broadcast mul と同じ写像）。
      let gate_for_mul = gate[i % params.dim];
      let x_for_mul = x[i];
      product_bits[lid] = bitcast<u32>(${product});
    }
    workgroupBarrier();
    if (i < params.n) {
      let product_after_store = bitcast<f32>(product_bits[lid]);
      out[i] = residual[i] + product_after_store;
    }
    workgroupBarrier();
    block = block + nwg.x;
  }
}
`;
};

/**
 * 16-byte uniform params。n は f32 x / residual / output の要素数、dim は行幅（= ゲートの要素数）。
 */
export const gatedResidualParams = (n: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("gated_residual params", { n, dim });
  if (dim < 1) throw new CodegenError(`gated_residual params: dim は 1 以上（${dim}）`);
  return new Uint32Array([n, dim, 0, 0]);
};

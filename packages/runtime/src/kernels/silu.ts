/**
 * `sigmoid(x) → mul` の中間 f32 格納境界を保った SiLU 融合カーネル（融合ルール `silu` の
 * 実体 — src/runtime/fusion.ts）。
 *
 * 置換元の primitive 列は sigmoid の結果を storage へ書き、次 dispatch の mul が読み戻す。
 * 単に `x * sigmoid_stable(x)` と書くとこの境界をコンパイラが畳めるため、sigmoid の f32
 * ビット列を workgroup `u32` へ書いて barrier 後に読み戻し、明示的な materialization 点を残す。
 *
 * **この「丸め障壁」は WGSL 仕様の保証ではない** — 仕様は浮動小数の fusion（fma への縮約）を
 * 許すだけで、workgroup memory 往復を最適化障壁として尊重することは要求していない。有限値の
 * ビット一致はバックエンドごとの実 GPU A/B（tests/gpu_silu_fusion_test.ts）で採用門にした
 * 実測事実であり、**バックエンド更新で PNG sha256 門が割れたら、まずここを疑うこと**
 * （融合を外して primitive 列に戻せば切り分けられる）。WGSL が規定しない NaN payload に
 * ついては分類の一致だけを契約にする。
 *
 * MUST: barrier は workgroup 一様な block loop の中だけに置く（2 本目は、次 block の書き込みを
 * 前 block の全 lane が読み終えてから始めるため）。subgroup / atomics / optional feature を
 * 使わないので WebGPU core の Metal / Dawn / wgpu 経路で使える。
 * MUST: sigmoid 本体は elementwise codegen と**同じ文字列**を共有する（別々に書くと
 * primitive と融合版で丸め列が割れうる）。
 */

import { SIGMOID_STABLE_WGSL } from "../codegen/elementwise.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/** 置換元 mul の入力順（有限値では可換だが、NaN payload はバックエンド差がありうる）。 */
export type SiluMulOrder = "x-sigmoid" | "sigmoid-x";

export const SILU_WORKGROUP_SIZE = 256;

const canonicalizeOrder = (order: SiluMulOrder): SiluMulOrder => {
  if (order === "x-sigmoid" || order === "sigmoid-x") return order;
  throw new CodegenError(`silu codegen: mul の入力順が不正（${order}）`);
};

export const siluKey = (order: SiluMulOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  return `silu:v1:${canonicalOrder}:f32:wg${SILU_WORKGROUP_SIZE}`;
};

/**
 * MUST: 入力順は WGSL とパイプラインキーの両方に残す。有限値では乗算は可換でも、NaN payload
 * の選ばれ方はバックエンドごとに違いうるので、片方だけで畳むと沈黙で別の列になる。
 * NOTE: block 数の切り上げを `(n + 255) / 256` と書かない — n = u32::MAX で桁溢れする。
 */
export const siluWgsl = (order: SiluMulOrder): string => {
  const canonicalOrder = canonicalizeOrder(order);
  // 置換元 mul も元入力を別 dispatch で読み直す。レジスタ越しに持ち越さない形で書く。
  const product = canonicalOrder === "x-sigmoid"
    ? "x_for_mul * sigmoid_after_store"
    : "sigmoid_after_store * x_for_mul";
  return `// karume SiLU (${canonicalOrder}, f32, staged sigmoid, full-write)
struct Params {
  n: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> sigmoid_bits: array<u32, ${SILU_WORKGROUP_SIZE}>;

${SIGMOID_STABLE_WGSL}

@compute @workgroup_size(${SILU_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let blocks = params.n / ${SILU_WORKGROUP_SIZE}u +
    select(0u, 1u, params.n % ${SILU_WORKGROUP_SIZE}u != 0u);
  var block = wid.x;
  while (block < blocks) {
    let i = block * ${SILU_WORKGROUP_SIZE}u + lid;
    if (i < params.n) {
      let x_for_sigmoid = x[i];
      sigmoid_bits[lid] = bitcast<u32>(sigmoid_stable(x_for_sigmoid));
    }
    workgroupBarrier();
    if (i < params.n) {
      let x_for_mul = x[i];
      let sigmoid_after_store = bitcast<f32>(sigmoid_bits[lid]);
      out[i] = ${product};
    }
    workgroupBarrier();
    block = block + nwg.x;
  }
}
`;
};

/** 16-byte uniform params。n は f32 input / output の要素数。 */
export const siluParams = (n: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("silu params", { n });
  return new Uint32Array([n, 0, 0, 0]);
};

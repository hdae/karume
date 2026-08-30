/**
 * masked_fill（`out = mask ? value : x`、値 f32 / 条件 bool、埋め値は attrs のスカラ）の
 * 固定カーネル。
 *
 * 出力と `x` は同形・連続で、**mask だけを右詰め broadcast の stride で読む**（実測は
 * `mask [1,1,T,T] × x [1,16,T,T]` と `mask [1,T,1024] × x [1,T,1024]` の 2 形 — recon §2）。
 * stride の組み立ては strided コピー族（ADR 0011）の expand と同一の規則なので
 * `expandSrcStrides` を共有する — 「長さ 1 の軸は stride 0 / 足りない先行軸も stride 0」は
 * 同じ右詰め broadcast の定義そのもので、別々に書けば規則が 2 箇所に割れる。
 *
 * rank は strided 族と同じく {@link STRIDED_RANK} 固定にして 1 パイプラインへ畳む。実 rank が
 * 足りないぶんは呼び出し側が左詰めで 1（dims）と 0（strides）に埋める。
 *
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 * MUST: 埋め値は params の**f32 のビット列**で運ぶ。定数として WGSL に焼くと、値ごとに
 * 別カーネルになってパイプラインキャッシュが埋め値の種類だけ膨らむ（-3.4e38 と 0 が別物に
 * なる）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { STRIDED_RANK } from "../codegen/strided.ts";

export const MASKED_FILL_WORKGROUP_SIZE = 256;

export const MASKED_FILL_KEY =
  `masked_fill:v1:f32:bool:r${STRIDED_RANK}:wg${MASKED_FILL_WORKGROUP_SIZE}`;

/**
 * params のレイアウト（storage, u32 配列）:
 * `[0]=要素数 n, [1..STRIDED_RANK]=出力 dims, 続けて mask の strides を同数, 末尾に埋め値`。
 * uniform ではなく storage で渡すのは、この族に workgroupBarrier が無く一様性解析の制約を
 * 受けないため（elementwise / strided と同じ理由）。
 */
const dimAt = (d: number): number => 1 + d;
const strideAt = (d: number): number => 1 + STRIDED_RANK + d;
const VALUE_AT = 1 + 2 * STRIDED_RANK;

export const MASKED_FILL_WGSL: string = [
  `// karume masked_fill (out = select(x, value, mask), rank ${STRIDED_RANK}, f32 / mask bool)`,
  "@group(0) @binding(0) var<storage, read> params: array<u32>;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read> mask: array<u32>;",
  "@group(0) @binding(3) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${MASKED_FILL_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  "  let n = params[0u];",
  `  let fill = bitcast<f32>(params[${VALUE_AT}u]);`,
  `  let stride = nwg.x * ${MASKED_FILL_WORKGROUP_SIZE}u;`,
  "  var i = gid.x;",
  "  while (i < n) {",
  "    var rem = i;",
  ...Array.from({ length: STRIDED_RANK - 1 }, (_, index) => {
    const d = STRIDED_RANK - 1 - index;
    return `    let c${d} = rem % params[${dimAt(d)}u]; rem = rem / params[${dimAt(d)}u];`;
  }),
  "    let c0 = rem;",
  `    let mask_index = ${
    Array.from({ length: STRIDED_RANK }, (_, d) => `c${d} * params[${strideAt(d)}u]`).join(" + ")
  };`,
  "    // bool の格納は u32 の 0 / 1（ADR 0009）— 真の位置だけを埋め値に差し替える",
  "    out[i] = select(x[i], fill, mask[mask_index] != 0u);",
  "    i = i + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * params バッファの中身を組み立てる。実 rank に足りないぶんは**左詰めで** dims=1 /
 * strides=0 に埋める（strided 族と同じ埋め方 — 余った先行軸の座標は常に 0 になり
 * 読み出し位置に寄与しない）。
 *
 * MUST: 埋め値は f32 へ丸めて載せる。IR の attrs は JSON の f64 なので、丸めずに扱うと
 * CPU 参照（`Math.fround`）と GPU（f32 語）で違う値になりうる。
 */
export const maskedFillParams = (
  outShape: readonly number[],
  maskStrides: readonly number[],
  value: number,
): Uint32Array<ArrayBuffer> => {
  if (outShape.length < 1 || outShape.length > STRIDED_RANK) {
    throw new CodegenError(
      `masked_fill params: 出力の rank ${outShape.length} は 1..${STRIDED_RANK} の外`,
    );
  }
  if (maskStrides.length !== outShape.length) {
    throw new CodegenError(
      `masked_fill params: mask の stride 本数 ${maskStrides.length} が出力 rank ${outShape.length} と違う`,
    );
  }
  // MUST: 有限判定は **f32 として**行う（f64 で有限な `1e39` は f32 語で `+Inf` になり、
  // 「有限の埋め値」を指定したはずの出力へ非有限値が書かれる — 契約層と同じ門）。
  if (!Number.isFinite(Math.fround(value))) {
    throw new CodegenError(`masked_fill params: 埋め値が f32 として有限でない（${value}）`);
  }
  const n = outShape.reduce((count, dim) => count * dim, 1);
  assertU32Params("masked_fill params", {
    ...Object.fromEntries(outShape.map((dim, d) => [`out_dims[${d}]`, dim])),
    ...Object.fromEntries(maskStrides.map((stride, d) => [`mask_strides[${d}]`, stride])),
    n,
  });
  const pad = STRIDED_RANK - outShape.length;
  const params = new Uint32Array(2 + 2 * STRIDED_RANK);
  params[0] = n;
  for (let d = 0; d < STRIDED_RANK; d += 1) {
    params[dimAt(d)] = d < pad ? 1 : outShape[d - pad];
    params[strideAt(d)] = d < pad ? 0 : maskStrides[d - pad];
  }
  new Float32Array(params.buffer)[VALUE_AT] = value;
  return params;
};

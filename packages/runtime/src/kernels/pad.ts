/**
 * pad（最終次元の定数 0 埋め、f32）の固定カーネル。
 *
 * 入力は先行次元を平坦化した `[rows, in_len]` の連続レイアウトで、出力は
 * `[rows, left + in_len + right]`。実測は相対位置注意の value 側 `F.pad(p_attn, [w, w])`
 * のみ（recon §2）で、最終次元・定数 0・左右パディングの 1 形しか無い。
 *
 * MUST: **1 dispatch で出力の全バイトを書く**（full-write 不変条件 — ADR 0014 / 0004）。
 * 範囲内は転写・範囲外は 0 を**書く**形にしてあり、「ゼロ初期化されたバッファへ範囲内だけ
 * copy する」形にはしない。バッファプールの再利用バッファはゼロ初期化されないため、
 * 後者はプールと両立しない（プロトタイプ既知の罠）。
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 */

import { assertU32Params } from "../codegen/params.ts";

export const PAD_WORKGROUP_SIZE = 256;

export const PAD_KEY = `pad:v1:f32:wg${PAD_WORKGROUP_SIZE}`;

export const PAD_WGSL: string = [
  "// karume pad (last dim, constant 0, f32, full-write)",
  "struct Params {",
  "  n: u32,",
  "  out_len: u32,",
  "  in_len: u32,",
  "  left: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${PAD_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  `  let stride = nwg.x * ${PAD_WORKGROUP_SIZE}u;`,
  "  var i = gid.x;",
  "  while (i < params.n) {",
  "    let col = i % params.out_len;",
  "    let row = i / params.out_len;",
  "    // 範囲外にも必ず 0 を書く（未書き込みを 1 要素も残さない）",
  "    var value = 0.0;",
  "    if (col >= params.left && col < params.left + params.in_len) {",
  "      value = x[row * params.in_len + (col - params.left)];",
  "    }",
  "    out[i] = value;",
  "    i = i + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * uniform の Params（行 op 族と同じく 4 語 = 16 バイト）。WGSL の uniform アドレス空間では
 * struct の整列が 16 バイトになる MUST。
 */
export const padParams = (
  rows: number,
  lengthIn: number,
  left: number,
  right: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("pad params", { rows, in_len: lengthIn, left, right });
  const lengthOut = left + lengthIn + right;
  const n = rows * lengthOut;
  assertU32Params("pad params", { out_len: lengthOut, n });
  const params = new Uint32Array(4);
  params[0] = n;
  params[1] = lengthOut;
  params[2] = lengthIn;
  params[3] = left;
  return params;
};

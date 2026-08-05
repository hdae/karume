/**
 * flip（静的軸の添字反転、f32）の固定カーネル。
 *
 * 軸の位置は `[outer, len, inner]` の 3 分割に畳んで運ぶ（`outer` = 反転軸より前の次元の積、
 * `inner` = 後ろの次元の積）。rank を params に載せないので rank 上限が要らず、
 * strided 族の負 stride 化（既存 params と決定性スナップショットの全面改版）も避けられる —
 * ADR 0014 が「新カーネル 1 本」を採った理由そのもの。
 *
 * MUST: 出力の全バイトを書く（full-write 不変条件 — ADR 0014 / 0004）。反転は全単射なので、
 * 1 dispatch = 出力 1 要素の grid-stride がそのまま全域被覆になる。
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 * MUST: 反転は `len - 1 - c`。`len - c` にすると先頭要素が範囲外を読み、shape も要素数も
 * 変わらないので値でしか検出できない（tests/gpu_ops_test.ts が非対称な列で固定する）。
 */

import { CodegenError } from "../codegen/errors.ts";

export const FLIP_WORKGROUP_SIZE = 256;

export const FLIP_KEY = `flip:v1:f32:wg${FLIP_WORKGROUP_SIZE}`;

export const FLIP_WGSL: string = [
  "// karume flip (static axis, f32, full-write)",
  "struct Params {",
  "  n: u32,",
  "  len: u32,",
  "  inner: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<f32>;",
  "",
  `@compute @workgroup_size(${FLIP_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(global_invocation_id) gid: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  `  let stride = nwg.x * ${FLIP_WORKGROUP_SIZE}u;`,
  "  var i = gid.x;",
  "  while (i < params.n) {",
  "    let tail = i % params.inner;",
  "    let rest = i / params.inner;",
  "    let axis = rest % params.len;",
  "    let head = rest / params.len;",
  "    let src = (head * params.len + (params.len - 1u - axis)) * params.inner + tail;",
  "    out[i] = x[src];",
  "    i = i + stride;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * uniform の Params（3 語だが 16 バイト確保する — uniform struct の整列は 16 バイト）。
 * `outer` は載せない（`n = outer · len · inner` から復元は不要で、載せれば同じ事実の二重管理）。
 */
export const flipParams = (
  outer: number,
  length: number,
  inner: number,
): Uint32Array<ArrayBuffer> => {
  for (const [name, value] of Object.entries({ outer, len: length, inner })) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new CodegenError(`flip params: ${name} は非負整数（${value}）`);
    }
  }
  const params = new Uint32Array(4);
  params[0] = outer * length * inner;
  params[1] = length;
  params[2] = inner;
  return params;
};

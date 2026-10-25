/**
 * **Batched tiled transpose** of f32 `[batch, s, c]` → `[batch, c, s]` through a workgroup tile
 * (perf-ledger F-8).
 *
 * The attention's Vᵀ comes from a token-major V (`[B, N, H·D]`, the linear's output) as one transpose
 * per batch: row `(h, d)` of Vᵀ is column `h·D + d` of V. The strided copy family (ADR 0011) reads
 * such a transpose with a stride of `c` floats between neighbouring lanes, which falls off a cliff
 * once a plane outgrows the L2 cache (research 2026-10-25-speed-recon-3080ti §6: 129 GB/s at 27,280
 * tokens on the RTX 3080 Ti). Here a 32 × 32 block is read along `c` and written along `s`, both
 * coalesced.
 *
 * It is a copy: no arithmetic, so the output is bit-identical to any other transpose.
 *
 * MUST: the tile is a scalar f32 array — no vector component is written through a dynamic index
 * (the naga/Metal silent drop of gemm.ts `storeBTransposed`). The row stride 33 keeps the column
 * reads of the store phase on distinct banks.
 * MUST: one workgroup = one 32 × 32 block (not grid-stride), so the caller sizes the grid with
 * `tiledWorkgroups` (fail loudly beyond the device limit). The batch runs along z.
 */

import { assertU32Params } from "../codegen/params.ts";

/** The edge of one block (and the workgroup's x width). */
export const TRANSPOSE_TILED_TILE = 32;

/** The workgroup's y height: each thread moves `TILE / ROWS` elements per phase. */
const TRANSPOSE_TILED_ROWS = 8;

/** The tile's row stride (one float of padding against bank conflicts in the column reads). */
const TILE_STRIDE = TRANSPOSE_TILED_TILE + 1;

export const TRANSPOSE_TILED_KEY = `transpose_tiled:v1:f32:t${TRANSPOSE_TILED_TILE}x${TILE_STRIDE}:wg${TRANSPOSE_TILED_TILE}x${TRANSPOSE_TILED_ROWS}`;

export const TRANSPOSE_TILED_WGSL: string = `// karume transpose_tiled (f32 [batch, s, c] → [batch, c, s] through a ${TRANSPOSE_TILED_TILE}×${TILE_STRIDE} workgroup tile)
struct Params {
  s: u32,
  c: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

var<workgroup> tile: array<f32, ${TRANSPOSE_TILED_TILE * TILE_STRIDE}>;

@compute @workgroup_size(${TRANSPOSE_TILED_TILE}, ${TRANSPOSE_TILED_ROWS})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let c0 = wid.x * ${TRANSPOSE_TILED_TILE}u;
  let s0 = wid.y * ${TRANSPOSE_TILED_TILE}u;
  let plane = wid.z * params.s * params.c;
  // Load: lane x walks c (a coalesced read of one source row), y steps through the block's rows
  for (var r = lid.y; r < ${TRANSPOSE_TILED_TILE}u; r = r + ${TRANSPOSE_TILED_ROWS}u) {
    let s = s0 + r;
    let c = c0 + lid.x;
    if (s < params.s && c < params.c) {
      tile[r * ${TILE_STRIDE}u + lid.x] = src[plane + s * params.c + c];
    }
  }
  workgroupBarrier();
  // Store: lane x walks s (a coalesced write of one destination row)
  for (var r = lid.y; r < ${TRANSPOSE_TILED_TILE}u; r = r + ${TRANSPOSE_TILED_ROWS}u) {
    let c = c0 + r;
    let s = s0 + lid.x;
    if (s < params.s && c < params.c) {
      dst[plane + c * params.s + s] = tile[lid.x * ${TILE_STRIDE}u + r];
    }
  }
}
`;

/**
 * Uniform Params (`{s, c}`). 16 bytes are allocated even though the content is 2 words (the
 * project's 16-byte convention, as in quantize_rows).
 */
export const transposeTiledParams = (s: number, c: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("transpose_tiled params", { s, c, plane: s * c });
  const params = new Uint32Array(4);
  params[0] = s;
  params[1] = c;
  return params;
};

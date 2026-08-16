/**
 * gather（最終次元固定、`out[..., j] = src[..., index[..., j]]`、値 f32 / 添字 i32）の
 * 固定カーネル。
 *
 * 出力・src・index はいずれも連続レイアウトで、先行次元は src と index で一致している
 * （契約 — src/ops.ts）。よって出力の平坦添字 i から `row = i / J` を作れば、読み出しは
 * `src[row * D + index[i]]` の 1 本で足りる（D = src の最終次元、J = 出力の最終次元）。
 *
 * MUST: grid-stride ループ前提。1 次元の dispatch 上限（仕様既定 65535）は実モデルで超える。
 *
 * ## 範囲外添字の扱い（裁定 — ADR 0012 の実装細目）
 *
 * 契約は `0 <= index < D`。違反は**モデル側の誤り**で、実運用の添字は export 時に
 * `clamp(x + 256, 0, 511)` 済みの定数由来（recon §2）なので実行時には起こらない前提だが、
 * 起きたときに**黙って別の要素を返さない**ことを優先する:
 *
 * - **GPU**: 範囲外の要素にだけ NaN（{@link GATHER_OOB_BITS}）を書く。検査は 2 比較 / 要素で
 *   gather はメモリ律速なので実効コストはほぼ 0。WebGPU の境界付きアクセスに任せて無検査に
 *   すると「0 または別の in-bounds 要素」が静かに返り、下流と突合の両方を通過してしまう。
 *   NaN は allclose・golden 突合・下流の全演算に伝播して必ず表に出る。
 * - **例外にしない理由**: カーネルから throw はできず、host へ通知するには run ごとの
 *   デバイス側フォールト旗 + readback（新しい診断チャネル）が要る。これは gather に付随して
 *   入れる機構ではないので、この波では NaN 汚染に留める。
 * - **CPU 参照は厳密**（範囲外で throw）。段 2 の突合ではオラクル側が先に落ちる。
 *
 * MUST: NaN のビット列は params で運ぶ。WGSL には NaN リテラルが無く、`bitcast<f32>(...)` を
 * 定数式で書くと「const-expression が NaN」としてシェーダ生成エラーになりうる実装がある。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const GATHER_WORKGROUP_SIZE = 256;

/** 範囲外添字の出力に書く quiet NaN のビット列（f32）。 */
export const GATHER_OOB_BITS = 0x7fc00000;

export const GATHER_KEY = `gather:v1:f32:i32:lastdim:wg${GATHER_WORKGROUP_SIZE}`;

export const GATHER_WGSL: string =
  `// karume gather (out[..., j] = src[..., index[..., j]], 最終次元固定, f32 / 添字 i32)
struct Dims {
  n: u32,
  cols: u32,
  src_cols: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${GATHER_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${GATHER_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.cols;
    let pick = index[i];
    // 契約外の添字は別の要素を返さず NaN で汚染する（カーネル doc の裁定）
    if (pick < 0 || u32(pick) >= dims.src_cols) {
      out[i] = bitcast<f32>(dims.oob);
    } else {
      out[i] = src[row * dims.src_cols + u32(pick)];
    }
    i = i + stride;
  }
}
`;

/**
 * uniform の Dims（ちょうど 4 語 = 16 バイト。uniform アドレス空間の整列要件を満たす）。
 * `cols` は 0 でもよい（そのとき `n` も 0 でカーネルのループが 1 度も回らない）。
 */
export const gatherParams = (
  count: number,
  cols: number,
  srcCols: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("gather params", { count, cols, srcCols });
  if (cols === 0 && count !== 0) {
    // 0 除算になる組み合わせ（要素数 > 0 なのに列が無い）は shape 契約上ありえない。
    throw new CodegenError(`gather params: 列数 0 で要素数 ${count} の組み合わせは無い`);
  }
  const params = new Uint32Array(4);
  params[0] = count;
  params[1] = cols;
  params[2] = srcCols;
  params[3] = GATHER_OOB_BITS;
  return params;
};

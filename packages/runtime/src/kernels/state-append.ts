/**
 * `state_append`（今 step の k / v を state スロットへ書く単機能 op — ADR 0067 決定 5）の
 * 固定カーネル。1 ノード = 1 dispatch で、値を定義しない **effect op**（出力 0 本）。
 *
 * 入力 `[B,Hkv,M,D]` の**先頭 `queryLength` 行だけ**を、スロット `[B,Hkv,C,D]` の論理行
 * `[P, P+Q)` へ写す。物理行の写像は {@link stateSlotRowWgsl}（sliding は `col % W` のリング）で、
 * **読み側（①QK / ③PV）と同一文字列を共有する MUST**（ADR 0067 決定 4「読み書き同式」— 片方
 * だけ別式にすると ring が一周した後の全読みが黙って別の行を指す）。
 *
 * MUST: **pad 行（`row ≥ Q`）は書かない**（スロットは full-write 対象外 — ADR 0066 追記 6。
 * 残骸は次 step の append が同じ式で上書きし、読者は resident 範囲外を読まない）。書くと
 * 「窓の中に無意味な値が混ざったスロット」になり、次 step の読者がそれを past として食う。
 * MUST: 仕事量は `B·Hkv·Q·D` に比例する（容量 `C` に比例させない — ADR 0066 決定 3）。
 * MUST: sliding で `Q > W` のとき、同じ物理行へ写る論理行のうち**最後の 1 本だけ**が書く
 * （`row + W ≥ Q`）。ring の意味論では最新の行が残るのが正だが、全行を並列に書かせると
 * どちらが勝つかは実装依存 — 沈黙の非決定性になる。full は `dst = P + row` が単射なので
 * 衝突しない（この門は sliding 変種にだけ生成される）。
 *
 * NOTE: full スロットの `P + Q ≤ C`（ADR 0067 決定 4 ④）は **context 側の実行時検査**で、
 * ここでは見ない（`P` は実行時値で params に載らない）。破った場合の書きは範囲外へ落ちる
 * （robustness で捨てられる）ので、検査の欠落は「静かに書かれない」形で出る。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gridStrideWorkgroups } from "../codegen/dispatch.ts";
import { assertU32Params } from "../codegen/params.ts";
import { STATE_LENGTHS_STRUCT, stateSlotRowWgsl, stateVariantKeyPart } from "./state-attention.ts";

export const STATE_APPEND_WORKGROUP_SIZE = 256;

export const stateAppendKey = (sliding: boolean): string =>
  `state_append:v1:f32:wg${STATE_APPEND_WORKGROUP_SIZE}${stateVariantKeyPart(sliding, false)}`;

/**
 * 束縛（**binding 0 = 静的 params・最後 = 論理長** — states 形 3 カーネルと同じ並び）:
 *
 * | binding | 資源                                |
 * | ------- | ----------------------------------- |
 * | 0       | `Params`（uniform）                 |
 * | 1       | `x` `[B,Hkv,M,D]`（読み）           |
 * | 2       | `slot` `[B,Hkv,C,D]`（書き）        |
 * | 3       | `Lengths`（uniform — context 所有） |
 *
 * params の語順（**この表が正本**）: `kv_planes`（= `B·Hkv`）/ `chunk_rows`（= `M`）/
 * `depth`（= `D`）/ `capacity`（= `C`）/ `window`（= `W`・`0` = full）。
 */
export const stateAppendWgsl = (sliding: boolean): string =>
  `// karume state_append (今 step の k / v を state スロットへ, f32${
    sliding ? ", sliding window のリング書込み" : ""
  })
struct Params {
  kv_planes: u32,
  chunk_rows: u32,
  depth: u32,
  capacity: u32,
  window: u32,
}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> slot: array<f32>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

@compute @workgroup_size(${STATE_APPEND_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let past = lengths.past;
  let query = lengths.query;
  // 総仕事は **B·Hkv·Q·D**（pad 行は添字空間に入らない = 書かれない）
  let total = params.kv_planes * query * params.depth;
  let stride = nwg.x * ${STATE_APPEND_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < total) {
    let d = i % params.depth;
    let rest = i / params.depth;
    let row = rest % query;
    let kv_plane = rest / query;
    let src = (kv_plane * params.chunk_rows + row) * params.depth + d;
    let dst = (kv_plane * params.capacity + slot_row(past + row)) * params.depth + d;
${
    sliding
      ? `    // ring が一周する Q > W では同じ物理行へ複数の論理行が写る。**最後の論理行だけ**が
    // 書く（全行を並列に書かせると勝者が実装依存 = 沈黙の非決定性）
    if (row + params.window >= query) {
      slot[dst] = x[src];
    }`
      : "    slot[dst] = x[src];"
  }
    i = i + stride;
  }
}
`;

/** `state_append` の静的幾何（uniform の語順の型側）。 */
export type StateAppendGeometry = {
  /** `B·Hkv`（スロットの平面数 — 入力とスロットで共通）。 */
  readonly kvPlanes: number;
  /** `M`（入力の行ストライド = 物理 chunk 行数）。 */
  readonly chunkRows: number;
  /** `D`。 */
  readonly depth: number;
  /** `C`（スロットの行容量）。 */
  readonly capacity: number;
  /** `W`（`0` = full）。 */
  readonly window: number;
};

/**
 * uniform の Params（5 語 — uniform struct の整列で 32 バイト確保する）。
 *
 * MUST: `W ≤ C`（窓がスロット容量を超える形は ADR 0067 決定 4 ③ の違反）。sliding かどうかは
 * `window > 0`（src/kernels/state-attention.ts の `stateSliding`）で、`W = 0` = full。呼び手は
 * キー選択・WGSL 生成・この params の 3 者を**同じ `window` の値**から導くこと（片方だけ
 * sliding にすると `col % 0u` が実装依存値になる）。
 */
export const stateAppendParams = (
  geometry: StateAppendGeometry,
): Uint32Array<ArrayBuffer> => {
  const { kvPlanes, chunkRows, depth, capacity, window } = geometry;
  assertU32Params("state_append params", {
    kv_planes: kvPlanes,
    chunk_rows: chunkRows,
    depth,
    capacity,
    window,
  });
  if (kvPlanes < 1 || chunkRows < 1 || depth < 1 || capacity < 1) {
    throw new CodegenError(
      `state_append params: kv_planes / chunk_rows / depth / capacity は正整数` +
        `（${kvPlanes} / ${chunkRows} / ${depth} / ${capacity}）`,
    );
  }
  if (window > capacity) {
    throw new CodegenError(
      `state_append params: window ${window} が容量 ${capacity} を超える（ADR 0067 決定 4 ③）`,
    );
  }
  const params = new Uint32Array(8);
  params[0] = kvPlanes;
  params[1] = chunkRows;
  params[2] = depth;
  params[3] = capacity;
  params[4] = window;
  return params;
};

/**
 * workgroup 数 `[⌈B·Hkv·Q·D / 256⌉, 1, 1]`（grid-stride なので上限超過は縮退）。
 *
 * MUST: 容量 `C` を引数に取らない（`C` に比例した dispatch は ADR 0066 決定 3 の不合格形）。
 * `Q` に**比例**し、`P` には依らない（書くのは今 step の Q 行だけ）。
 */
export const stateAppendWorkgroups = (
  geometry: Pick<StateAppendGeometry, "kvPlanes" | "depth">,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  const { kvPlanes, depth } = geometry;
  assertU32Params(where, { kv_planes: kvPlanes, depth, query });
  if (kvPlanes < 1 || depth < 1 || query < 1) {
    throw new CodegenError(
      `${where}: kv_planes / depth / query は正整数（${kvPlanes} / ${depth} / ${query}）`,
    );
  }
  const total = kvPlanes * query * depth;
  assertU32Params(where, { "kv_planes * query * depth": total });
  return [gridStrideWorkgroups(total, STATE_APPEND_WORKGROUP_SIZE, limit), 1, 1];
};

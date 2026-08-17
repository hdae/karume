/**
 * argmax（最終次元・**rank 保存**・出力 i32）の固定カーネル（ADR 0068 決定 2）。
 *
 * 入力は先行次元を平坦化した `[rows, dim]` の連続レイアウトで、出力は行ごとに 1 語
 * （宣言 shape は `[…, 1]` = 最終次元を 1 に潰した固定形。`keepdim` の欄は無い）。
 *
 * 形は行 reduce（src/codegen/reduce.ts）と同型 — 1 行 = 1 workgroup(256) + 256 幅の
 * ビット反転二分木 + **行方向も grid-stride**（1 次元の workgroup 上限 65535 を実モデルが
 * 超える）。新しいのは **(値, index) の対を運ぶ**点だけ。
 *
 * MUST: reduce 族に相乗りしない（別族・別キー）。既存 `amax` の identity は
 * `-F32_MAX`（有限 sentinel）で、本カーネルが要求する **−inf 始まり**と衝突する。
 * reduce 側のキー版数を上げて合流させると既存のスナップショットと WAV / PNG 門が
 * 丸ごと動くので、席を分ける方が正しい。
 *
 * MUST: **タイブレークは最小 index**（torch 準拠 — 実測 2026-08-17: 同値が複数ある行は
 * 先頭が返る）。llama.cpp は GPU 側 = 最大 index / CPU sampler = 最小 index で同一リポ内
 * でも食い違っており（調査 §2）、明文化しないと greedy の再現性が実装差で割れる。
 * 保存の機序は {@link ARGMAX_BEATS_FN}: 「値が大きい方が勝ち・同値なら index が小さい方が
 * 勝ち」は **(値 降順, index 昇順) の辞書式順序での最大元**なので、木の結合の仕方
 * （どちらの子が第 1 引数か）に依らず結果が同じ。
 *
 * MUST: 行 max の初期値は **−inf**（有限 sentinel 禁止）。index の初期値は番兵 `dim`
 * （= 「まだ候補が無い」）で、これで「全要素 −inf の行 → 最小 index = 0」が決定的になる:
 * 値が同値（−inf 対 −inf）でも index 比較で実要素（`< dim`）が必ず番兵に勝つ。dim ≥ 1 は
 * shape 層が保証する（長さ 0 の最終次元は契約で拒否）ので、レーン 0 は必ず 1 要素以上を
 * 走査し、最終結果の index は常に `[0, dim)` に入る。
 *
 * MUST: NaN は**最大として扱う**（torch 準拠 — 実測 2026-08-17: `argmax` は NaN の index を
 * 返し、複数あれば最小 index）。`amax` / `amin` の NaN 伝播（reduce.ts）と同じ規律で、
 * 判定は**ビット列**（{@link IS_NAN_FN}）が担う — ドライバの比較は NaN で全て false に
 * なるので、素の `>` に任せると NaN が黙って負けて「amax は NaN・argmax は別要素」という
 * 族内で食い違う結果になる。
 *
 * MUST: params は uniform で渡す（行ループ内に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある）。
 * MUST: −inf のビット列は params で運ぶ（safe_softmax と同じ理由 — 定数式の
 * `bitcast<f32>(0xff800000u)` を「const-expression が inf」としてシェーダ生成エラーに
 * する実装がありうる）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const ARGMAX_WORKGROUP_SIZE = 256;

export const ARGMAX_KEY = `argmax:v1:f32>i32:lastdim:minindex:wg${ARGMAX_WORKGROUP_SIZE}`;

/** −inf の f32 ビット列（params 3 語目 — WGSL に無限大リテラルが無い）。 */
export const ARGMAX_NEG_INF_BITS = 0xff800000;

/**
 * f32 の NaN を**ビット列**で判定する（符号を落として指数部全 1 + 仮数部非 0）。
 * src/codegen/reduce.ts / elementwise.ts と同じ判定式（浮動小数の比較で判定しない — 比較を
 * 含む式はシェーダコンパイラが `max` イディオムへ畳み、ドライバの `max` が NaN を飲む）。
 */
const IS_NAN_FN = `fn is_nan_bits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}`;

/**
 * `(vb, ib)` が `(va, ia)` に勝つか。順序は **NaN > 有限 > −inf**、同値なら **index が
 * 小さい方**（torch 準拠）。
 *
 * この述語は辞書式順序の厳密比較なので、①どちらの子を第 1 引数にしても勝者が変わらない
 * ②勝者は候補集合の最大元 — 木の簡約が段ごとに最大元を残すことで、走査順や結合順に
 * 依らず「最大値のうち最小 index」が出る。同一の `(値, index)` 対が来たときだけ引き分けで
 * `false`（= 第 1 引数側を残す）になるが、その 2 つは区別できないので決定性は保たれる。
 */
const ARGMAX_BEATS_FN = `fn argmax_beats(vb: f32, ib: u32, va: f32, ia: u32) -> bool {
  let na = is_nan_bits(va);
  let nb = is_nan_bits(vb);
  if (na != nb) {
    return nb;
  }
  if (na) {
    return ib < ia;
  }
  return vb > va || (vb == va && ib < ia);
}`;

export const ARGMAX_WGSL: string = [
  "// karume argmax (last dim, f32>i32, min-index tie-break, -inf identity)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "  neg_inf: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<f32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<i32>;",
  "",
  IS_NAN_FN,
  "",
  ARGMAX_BEATS_FN,
  "",
  `var<workgroup> scratch_value: array<f32, ${ARGMAX_WORKGROUP_SIZE}>;`,
  `var<workgroup> scratch_index: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
  "",
  `@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})`,
  "fn main(",
  "  @builtin(workgroup_id) wid: vec3<u32>,",
  "  @builtin(local_invocation_id) lid3: vec3<u32>,",
  "  @builtin(num_workgroups) nwg: vec3<u32>,",
  ") {",
  "  let lid = lid3.x;",
  "  let dim = params.dim;",
  "  let neg_inf = bitcast<f32>(params.neg_inf);",
  "  var row = wid.x;",
  "  while (row < params.rows) {",
  "    let base = row * dim;",
  "    // identity は値も index も最弱（番兵 dim = 候補なし）— 全 -inf 行でも最初の実要素が",
  "    // index 比較で勝つので、結果は決定的に最小 index になる",
  "    var best = neg_inf;",
  "    var best_at = dim;",
  "    var i = lid;",
  "    while (i < dim) {",
  "      let v = x[base + i];",
  "      if (argmax_beats(v, i, best, best_at)) {",
  "        best = v;",
  "        best_at = i;",
  "      }",
  `      i = i + ${ARGMAX_WORKGROUP_SIZE}u;`,
  "    }",
  "    scratch_value[lid] = best;",
  "    scratch_index[lid] = best_at;",
  "    workgroupBarrier();",
  `    var stride = ${ARGMAX_WORKGROUP_SIZE / 2}u;`,
  "    while (stride > 0u) {",
  "      if (lid < stride) {",
  "        let other = scratch_value[lid + stride];",
  "        let other_at = scratch_index[lid + stride];",
  "        if (argmax_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
  "          scratch_value[lid] = other;",
  "          scratch_index[lid] = other_at;",
  "        }",
  "      }",
  "      workgroupBarrier();",
  "      stride = stride / 2u;",
  "    }",
  "    if (lid == 0u) {",
  "      out[row] = i32(scratch_index[0u]);",
  "    }",
  "    // 次の行が scratch を上書きする前に scratch[0] の読み終わりを揃える",
  "    workgroupBarrier();",
  "    row = row + nwg.x;",
  "  }",
  "}",
  "",
].join("\n");

/**
 * uniform の Params（`{rows, dim, neg_inf}`）。WGSL の uniform アドレス空間では struct の
 * 整列が 16 バイトになるため、3 語ぶんの内容でも 16 バイト確保する MUST。
 *
 * MUST: `dim` は正整数（`dim = 0` は「最大値が無い行」で、番兵 index がそのまま出力へ
 * 漏れる）。契約側も長さ 0 の最終次元を拒否するので、ここは二重の門。
 */
export const argmaxParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("argmax params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`argmax params: dim は正整数（${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  params[2] = ARGMAX_NEG_INF_BITS;
  return params;
};

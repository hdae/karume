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
 * 保存の機序は {@link F32_RANK_WGSL}: 「値が大きい方が勝ち・同値なら index が小さい方が
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
 * 判定は**ビット列**（{@link F32_RANK_WGSL}）が担う — ドライバの比較は NaN で全て false に
 * なるので、素の `>` に任せると NaN が黙って負けて「amax は NaN・argmax は別要素」という
 * 族内で食い違う結果になる。
 *
 * MUST: params は uniform で渡す（行ループ内に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある）。
 * −inf のビット列は既存の params レイアウトで運ぶ。比較・保存とも整数のまま扱い、
 * 非正規数が途中でゼロ化されないようにする（ADR 0068 追記 10）。
 */

import { F32_RANK_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const ARGMAX_WORKGROUP_SIZE = 256;

export const ARGMAX_KEY = `argmax:v2:f32>i32:lastdim:minindex:wg${ARGMAX_WORKGROUP_SIZE}`;

/** −inf の f32 ビット列（params 3 語目）。 */
export const ARGMAX_NEG_INF_BITS = 0xff800000;

export const ARGMAX_WGSL: string = [
  "// karume argmax (last dim, f32>i32, min-index tie-break, -inf identity)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "  neg_inf: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<u32>;",
  "@group(0) @binding(2) var<storage, read_write> out: array<i32>;",
  "",
  F32_RANK_WGSL,
  "",
  `var<workgroup> scratch_value: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
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
  "  let neg_inf = f32_rank_key(params.neg_inf);",
  "  var row = wid.x;",
  "  while (row < params.rows) {",
  "    let base = row * dim;",
  "    // identity は値も index も最弱（番兵 dim = 候補なし）— 全 -inf 行でも最初の実要素が",
  "    // index 比較で勝つので、結果は決定的に最小 index になる",
  "    var best = neg_inf;",
  "    var best_at = dim;",
  "    var i = lid;",
  "    while (i < dim) {",
  "      let v = f32_rank_key(x[base + i]);",
  "      if (rank_key_beats(v, i, best, best_at)) {",
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
  "        if (rank_key_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
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

// ---- 2 相分割（長い行 — MTP 段 4-B ③） --------------------------------------------------
//
// 1 行 = 1 workgroup の形は、行が語彙長（gemma4 の drafter は 262,144）になると 256 スレッドが
// 1,024 要素ずつを逐次で畳む遅延が律速になる（実測 2026-09-09・RTX 3080 Ti: 1 行 0.5〜0.6 ms・
// drafter 3 段で 1.5〜2.1 ms/cycle）。長い行は **2 dispatch** に割る:
//
// - **partial**: 行を {@link ARGMAX_SPLIT_SPAN} 要素の区間に切り、区間ごとに 1 workgroup が
//   区間内の最大元 `(順位キー, index)` を出して一時バッファへ書く（`[rows, groups, 2]` の u32 —
//   順位キーは NaN を同点で最優先にする）。
// - **merge**: 行ごとに 1 workgroup が `groups` 本の部分結果を畳んで添字を書く。
//
// MUST: 結果は 1 dispatch 形と**ビット同一**（辞書式順序 (値 降順, index 昇順) の最大元は
// 結合順に依らないので、区間ごとに最大元を取ってから畳んでも全域の最大元に一致する。部分
// 結果の identity は同じ −inf / 番兵 `dim` で、区間は空にならない〈`groups = ceil(dim / span)`〉
// ので各部分結果の index は `[0, dim)` に入り、全 −inf 行でも merge は最小 index = 0 を返す）。
// MUST: 分割の閾値・区間幅は**形の純関数**（{@link argmaxSplitGroups}）— 同じ形は常に同じ
// 経路・同じキーになる（実行時オートチューン禁止・ADR 0022）。

/** partial 1 workgroup が受け持つ要素数（256 スレッド × 16 要素）。 */
export const ARGMAX_SPLIT_ELEMENTS = 16;
export const ARGMAX_SPLIT_SPAN = ARGMAX_WORKGROUP_SIZE * ARGMAX_SPLIT_ELEMENTS;
/**
 * 2 相に割る最小の行長（= 区間 4 本以上）。これより短い行は 1 dispatch 形のまま（キー・WGSL
 * とも不変 — 既存のスナップショット / ビット同一門はそのまま効く）。
 */
export const ARGMAX_SPLIT_MIN_DIM = ARGMAX_SPLIT_SPAN * 4;

export const ARGMAX_SPLIT_PARTIAL_KEY =
  `argmax:v2:f32>i32:lastdim:minindex:split-partial:wg${ARGMAX_WORKGROUP_SIZE}e${ARGMAX_SPLIT_ELEMENTS}`;
export const ARGMAX_SPLIT_MERGE_KEY =
  `argmax:v2:f32>i32:lastdim:minindex:split-merge:wg${ARGMAX_WORKGROUP_SIZE}`;

/** 行長 `dim` を 2 相に割るときの区間数（割らない形は 0）。 */
export const argmaxSplitGroups = (dim: number): number =>
  dim >= ARGMAX_SPLIT_MIN_DIM ? Math.ceil(dim / ARGMAX_SPLIT_SPAN) : 0;

/**
 * topk(k=1)の分割はWebGPUの既定dispatch上限内に限る。
 * 巨大な行は従来のgrid-stride 1 dispatchへ残し、対応入力を狭めない。
 * https://gpuweb.github.io/gpuweb/#dom-gpusupportedlimits-maxcomputeworkgroupsperdimension
 */
export const topkOneSplitGroups = (dim: number): number => {
  const groups = argmaxSplitGroups(dim);
  return groups <= 65_535 ? groups : 0;
};

/** 部分結果の一時バッファの大きさ（`[rows, groups]` × (値 u32 + index u32)）。 */
export const argmaxSplitPartialBytes = (rows: number, groups: number): number => rows * groups * 8;

/** 木の簡約（scratch の 256 対を 1 対へ）— 1 dispatch 形と同じ骨格。 */
const ARGMAX_TREE_WGSL = [
  `    var stride = ${ARGMAX_WORKGROUP_SIZE / 2}u;`,
  "    while (stride > 0u) {",
  "      if (lid < stride) {",
  "        let other = scratch_value[lid + stride];",
  "        let other_at = scratch_index[lid + stride];",
  "        if (rank_key_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
  "          scratch_value[lid] = other;",
  "          scratch_index[lid] = other_at;",
  "        }",
  "      }",
  "      workgroupBarrier();",
  "      stride = stride / 2u;",
  "    }",
].join("\n");

export const ARGMAX_SPLIT_PARTIAL_WGSL: string = [
  "// karume argmax split partial (last dim, f32>i32, min-index tie-break, -inf identity)",
  "struct Params {",
  "  rows: u32,",
  "  dim: u32,",
  "  groups: u32,",
  "  neg_inf: u32,",
  "}",
  "@group(0) @binding(0) var<uniform> params: Params;",
  "@group(0) @binding(1) var<storage, read> x: array<u32>;",
  "@group(0) @binding(2) var<storage, read_write> partial: array<u32>;",
  "",
  F32_RANK_WGSL,
  "",
  `var<workgroup> scratch_value: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
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
  "  let neg_inf = f32_rank_key(params.neg_inf);",
  "  // x 軸 = 区間（1 workgroup = 1 区間・上限超過は fail loudly）・y 軸 = 行（grid-stride）",
  "  let group = wid.x;",
  `  let start = group * ${ARGMAX_SPLIT_SPAN}u;`,
  `  let end = min(start + ${ARGMAX_SPLIT_SPAN}u, dim);`,
  "  var row = wid.y;",
  "  while (row < params.rows) {",
  "    let base = row * dim;",
  "    var best = neg_inf;",
  "    var best_at = dim;",
  "    var i = start + lid;",
  "    while (i < end) {",
  "      let v = f32_rank_key(x[base + i]);",
  "      if (rank_key_beats(v, i, best, best_at)) {",
  "        best = v;",
  "        best_at = i;",
  "      }",
  `      i = i + ${ARGMAX_WORKGROUP_SIZE}u;`,
  "    }",
  "    scratch_value[lid] = best;",
  "    scratch_index[lid] = best_at;",
  "    workgroupBarrier();",
  ARGMAX_TREE_WGSL,
  "    if (lid == 0u) {",
  "      let at = (row * params.groups + group) * 2u;",
  "      partial[at] = scratch_value[0u];",
  "      partial[at + 1u] = scratch_index[0u];",
  "    }",
  "    workgroupBarrier();",
  "    row = row + nwg.y;",
  "  }",
  "}",
  "",
].join("\n");

const splitMergeWgsl = (valueOutput: boolean): string =>
  [
    valueOutput
      ? "// karume topk k=1 split merge (raw input bits and minimum index)"
      : "// karume argmax split merge (partials [rows, groups, 2] -> i32 index per row)",
    "struct Params {",
    "  rows: u32,",
    "  dim: u32,",
    "  groups: u32,",
    "  neg_inf: u32,",
    "}",
    "@group(0) @binding(0) var<uniform> params: Params;",
    "@group(0) @binding(1) var<storage, read> partial: array<u32>;",
    "@group(0) @binding(2) var<storage, read_write> out: array<i32>;",
    ...(valueOutput
      ? [
        "@group(0) @binding(3) var<storage, read_write> value: array<u32>;",
        "@group(0) @binding(4) var<storage, read> input: array<u32>;",
      ]
      : []),
    "",
    F32_RANK_WGSL,
    "",
    `var<workgroup> scratch_value: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
    `var<workgroup> scratch_index: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;`,
    "",
    `@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})`,
    "fn main(",
    "  @builtin(workgroup_id) wid: vec3<u32>,",
    "  @builtin(local_invocation_id) lid3: vec3<u32>,",
    "  @builtin(num_workgroups) nwg: vec3<u32>,",
    ") {",
    "  let lid = lid3.x;",
    "  let groups = params.groups;",
    "  let neg_inf = f32_rank_key(params.neg_inf);",
    "  var row = wid.x;",
    "  while (row < params.rows) {",
    "    let base = row * groups * 2u;",
    "    var best = neg_inf;",
    "    var best_at = params.dim;",
    "    var g = lid;",
    "    while (g < groups) {",
    "      let v = partial[base + g * 2u];",
    "      let v_at = partial[base + g * 2u + 1u];",
    "      if (rank_key_beats(v, v_at, best, best_at)) {",
    "        best = v;",
    "        best_at = v_at;",
    "      }",
    `      g = g + ${ARGMAX_WORKGROUP_SIZE}u;`,
    "    }",
    "    scratch_value[lid] = best;",
    "    scratch_index[lid] = best_at;",
    "    workgroupBarrier();",
    ARGMAX_TREE_WGSL,
    "    if (lid == 0u) {",
    "      out[row] = i32(scratch_index[0u]);",
    ...(valueOutput ? ["      value[row] = input[row * params.dim + scratch_index[0u]];"] : []),
    "    }",
    "    workgroupBarrier();",
    "    row = row + nwg.x;",
    "  }",
    "}",
    "",
  ].join("\n");

export const ARGMAX_SPLIT_MERGE_WGSL: string = splitMergeWgsl(false);
export const TOPK_ONE_SPLIT_MERGE_KEY =
  `topk:v2:f32+i32:lastdim:desc:minindex:k1:split-merge:wg${ARGMAX_WORKGROUP_SIZE}`;
export const topkOneSplitMergeWgsl = (): string => splitMergeWgsl(true);

/**
 * 2 相形の uniform（`{rows, dim, groups, neg_inf}` — partial / merge で同じ 4 語）。
 *
 * MUST: `groups` は {@link argmaxSplitGroups} の値（1 以上）— 0 で呼ぶのは 1 dispatch 形を
 * 選ぶべき形なので fail loudly。
 */
export const argmaxSplitParams = (
  rows: number,
  dim: number,
  groups: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("argmax split params", { rows, dim, groups });
  if (dim < 1 || groups < 1 || groups !== argmaxSplitGroups(dim)) {
    throw new CodegenError(
      `argmax split params: dim ${dim} / groups ${groups} が分割の形に合わない` +
        `（${argmaxSplitGroups(dim)} 区間が正）`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  params[2] = groups;
  params[3] = ARGMAX_NEG_INF_BITS;
  return params;
};

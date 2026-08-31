/**
 * topk（最終次元・**static-k**・出力 2 本 = 値 f32 の降順 + 添字 i32）の固定カーネル
 * （ADR 0068 決定 3 — ノード多出力の最初の入居者）。
 *
 * 入力は先行次元を平坦化した `[rows, dim]` の連続レイアウトで、出力は行ごとに k 語ずつ
 * （宣言 shape は 2 本とも `[…, k]`）。`k` は**計画時定数**（attrs 宣言必須）なので WGSL に
 * 焼き、params では運ばない（同じ事実を 2 箇所で持たない）。
 *
 * 形は **1 行 = 1 workgroup** の 2 相で、行方向は grid-stride（1 次元の workgroup 上限を実
 * モデルが超える）:
 *
 * 1. **レーン局所 top-k** — レーン `l` は `i ≡ l (mod W)` の要素だけを走査し、自分の
 *    ブロック（`cand_*` の K 語）を**降順のまま**保つ挿入で更新する。末尾（最弱）に勝てない
 *    候補はそこで捨てる。
 * 2. **merge** — k ラウンドの **W 者トーナメント**。各ラウンドで「各レーンの現在の先頭」の
 *    最大元を木で畳み、それが残り集合の最大元（各ブロックが降順だから）。勝った要素の
 *    持ち主だけがカーソルを 1 進める。
 *
 * MUST: **全語彙 argsort を経由しない**（ADR 0068 決定 3）。読み出しは行を 1 回だけで、
 * 仕事量は `dim/W` の走査 + `k·log2(W)` の畳み込み。k 回の行 reduce（マスク付き argmax の
 * 反復）に均すと行を k 回読むことになり、ADR が避けている高コスト側に落ちる。
 *
 * MUST: **scratch は workgroup storage に閉じる**（出力バッファへの同居も一時バッファも
 * 出さない — ADR 0068 決定 3 の「確保仕様が読めなくなる」を構造で回避）。代わりに k の
 * **実装上限**が workgroup storage の device limit から決まる（{@link topkMaxK}）ので、
 * 超過は縮退させず {@link assertTopkK} が上限値つきで fail loudly にする。
 *
 * MUST: タイブレークは **最小 index**（ADR 0068 決定 3）。述語 {@link TOPK_BEATS_FN} は
 * argmax（src/kernels/argmax.ts の `argmax_beats`）と**同一本文**で、(値 降順, index 昇順) の
 * 辞書式順序の厳密比較 — ①どちらの子を第 1 引数にしても勝者が変わらない ②勝者は候補集合の
 * 最大元。よってレーン局所の挿入・木の畳み込み・ラウンド間の順序が全て 1 本の述語で閉じ、
 * 結合順に依らず「値降順・同値なら index 昇順」で出る。k=1 は argmax と同じ答えになる
 * （族間の食い違いを門が突き合わせる）。
 *
 * NOTE: torch の `topk` は**同値要素の順序を保証しない**（実測 2026-08-17 / torch 2.13.0+cpu):
 * `topk([5,5,5,5], 1)` は index **2** を返すのに `argmax([5,5,5,5])` は **0** で、torch 自身が
 * 同一リポ内で食い違っている（k=1 の 7 行中 3 行で不一致）。`[…9…9…9…]` の 700 列では
 * `[650,300,12]` = **降順 index** で返る。したがって ADR 0068 決定 3 の括弧書き「同値要素の
 * 順序も torch と一致」は**実測で成り立たない** — 一致するのは**値の列**（降順・多重度が
 * 同じなので (値降順, index昇順) 実装とビット一致。重複だらけの 200×4 ケースで実測）で、
 * 添字の列は karume が最小 index に**規定する**側にした（argmax と同族・決定的）。
 *
 * MUST: 行 max の初期値は **−inf**、ブロックの identity は `(−inf, 番兵 index = dim)`
 * （有限 sentinel 禁止 — argmax と同じ理由）。実要素は必ず identity に勝つ（値が −inf でも
 * index 比較で `< dim` が勝つ）ので、全 −inf 行でも答えは「最小 index から k 本」に決まる。
 * `1 ≤ k ≤ dim` は契約層（shape 規則）が保証するので、レーン局所リストの和には必ず k 本以上の
 * 実要素が入り（`Σ min(K, count_l) ≥ min(K, Σ count_l) = K`）、identity が出力へ漏れることは
 * 無い。
 *
 * MUST: NaN は**最大として扱う**（argmax と同じ規律。torch も NaN を先頭へ出す — 実測同上）。
 * 判定は**ビット列**（{@link IS_NAN_BITS_WGSL}）— ドライバの比較は NaN で全て false になるので、
 * 素の `>` に任せると NaN が黙って負ける。
 *
 * MUST: params は uniform で渡す（行ループ内に workgroupBarrier があり、ループ条件が
 * workgroup 内で一様である必要がある）。
 * MUST: −inf のビット列は params で運ぶ（safe_softmax / argmax と同じ理由 — 定数式の
 * `bitcast<f32>(0xff800000u)` を「const-expression が inf」としてシェーダ生成エラーにする
 * 実装がありうる）。
 */

import { IS_NAN_BITS_WGSL } from "../codegen/elementwise.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

/**
 * 1 行を畳むレーン数。
 *
 * MUST: 大きくしない。workgroup storage は `8·W·(k+1)` バイト（レーンごとに K 語の
 * (値, index) 対 + トーナメント 1 語）で、W を倍にすると k の実装上限が半分になる
 * （{@link topkMaxK}）。W=32 は WebGPU 既定の 16384 バイトで **k ≤ 63** を通す最大の 2 冪で、
 * top-k sampling の実用域（k ≤ 50 級）を既定の機でも収める側に振った選択。
 */
export const TOPK_WORKGROUP_SIZE = 32;

/** −inf の f32 ビット列（params 3 語目 — WGSL に無限大リテラルが無い）。 */
export const TOPK_NEG_INF_BITS = 0xff800000;

/** 1 候補ぶんの workgroup storage（f32 の値 + u32 の添字）。 */
const TOPK_CANDIDATE_BYTES = 8;

/**
 * `k` を通すのに要る workgroup storage のバイト数 = `8·W·(k+1)`。
 * 内訳はレーンごとの候補ブロック `k` 語（`cand_value` / `cand_index`）と、トーナメントの
 * 先頭 1 語（`head_value` / `head_index`）。
 */
export const topkWorkgroupStorageBytes = (k: number): number =>
  TOPK_CANDIDATE_BYTES * TOPK_WORKGROUP_SIZE * (k + 1);

/**
 * device の workgroup storage 上限から決まる **k の実装上限**（ADR 0068 決定 3）。
 * `8·W·(k+1) ≤ limit` の最大の k で、WebGPU 既定（16384 バイト）では **63**。
 */
export const topkMaxK = (storageLimitBytes: number): number =>
  Math.floor(storageLimitBytes / (TOPK_CANDIDATE_BYTES * TOPK_WORKGROUP_SIZE)) - 1;

/**
 * WebGPU core 既定（`maxComputeWorkgroupStorageSize` = 16384）で保証される k の上限。
 * これを超える k は「手元のアダプタでは通るが既定の機で落ちる」ポータビリティの段差になる
 * （どちらでも黙って縮退はしない — {@link assertTopkK} が上限つきで落とす）。
 */
export const TOPK_CORE_LIMIT_MAX_K = 63;

/**
 * k が実装上限の内側であることを検査する（ADR 0068 決定 3 — **縮退しない**）。
 *
 * MUST: 診断に**上限値そのもの**を載せる。「k が大きすぎる」だけでは利用者が k をいくつまで
 * 下げればよいか分からず、device 依存の値（{@link topkMaxK}）を当てさせることになる。
 * MUST: `1 ≤ k ≤ 最終次元` はこの関数の担当ではない（device に依らない受理領域なので契約層 —
 * src/ops/attrs.ts の `TOPK_ATTRS` と src/ops/shapes.ts の shape 規則が持つ）。
 */
export const assertTopkK = (k: number, storageLimitBytes: number, where: string): void => {
  assertU32Params(`${where} の topk`, { k });
  if (k < 1) {
    throw new CodegenError(`${where}: topk の k は正整数（${k}）`);
  }
  const max = topkMaxK(storageLimitBytes);
  if (k > max) {
    throw new CodegenError(
      `${where}: topk の k=${k} が実装上限 ${max} を超える（workgroup storage ${
        topkWorkgroupStorageBytes(k)
      } バイトが要るが device 上限は ${storageLimitBytes} バイト・1 行 ${TOPK_WORKGROUP_SIZE} レーン）`,
    );
  }
};

/**
 * `(vb, ib)` が `(va, ia)` に勝つか。順序は **NaN > 有限 > −inf**、同値なら **index が
 * 小さい方**（モジュール doc の MUST）。
 *
 * MUST: argmax の `argmax_beats` と**同一の本文**にする（族間で答えが割れないことが k=1 の
 * 突合門の前提）。名前だけ分けるのは、キーが別パイプラインだと分かるようにするため。
 */
const TOPK_BEATS_FN = `fn topk_beats(vb: f32, ib: u32, va: f32, ia: u32) -> bool {
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

/**
 * パイプラインキー。**k を含む**（WGSL が k で変わるので、含めないと最初の k のパイプラインが
 * 別の k の dispatch に配られ、例外なしに別のバイト列が書かれる）。
 */
export const topkKey = (k: number): string =>
  `topk:v1:f32+i32:lastdim:desc:minindex:k${k}:wg${TOPK_WORKGROUP_SIZE}`;

/** k を焼いた WGSL（配列長とラウンド数が k で決まる）。 */
export const topkWgsl = (k: number): string => {
  const w = TOPK_WORKGROUP_SIZE;
  return `// karume topk (last dim, f32 values + i32 indices, descending, min-index tie-break, -inf identity, k=${k})
struct Params {
  rows: u32,
  dim: u32,
  neg_inf: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> indices: array<i32>;

${IS_NAN_BITS_WGSL}

${TOPK_BEATS_FN}

var<workgroup> cand_value: array<f32, ${k * w}>;
var<workgroup> cand_index: array<u32, ${k * w}>;
var<workgroup> head_value: array<f32, ${w}>;
var<workgroup> head_index: array<u32, ${w}>;

@compute @workgroup_size(${w})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let neg_inf = bitcast<f32>(params.neg_inf);
  let block = lid * ${k}u;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    // 相 1: レーン局所 top-k。identity は値も index も最弱（番兵 dim = 候補なし）なので、
    // 実要素は必ず勝つ — 全 -inf 行でも答えが最小 index から k 本に決まる
    for (var s = 0u; s < ${k}u; s = s + 1u) {
      cand_value[block + s] = neg_inf;
      cand_index[block + s] = dim;
    }
    var i = lid;
    while (i < dim) {
      let v = x[base + i];
      // 末尾（最弱）に勝てない候補はここで捨てる。勝つ候補だけが降順を保つ挿入へ進む
      if (topk_beats(v, i, cand_value[block + ${k - 1}u], cand_index[block + ${k - 1}u])) {
        var s = ${k - 1}u;
        while (s > 0u && topk_beats(v, i, cand_value[block + s - 1u], cand_index[block + s - 1u])) {
          cand_value[block + s] = cand_value[block + s - 1u];
          cand_index[block + s] = cand_index[block + s - 1u];
          s = s - 1u;
        }
        cand_value[block + s] = v;
        cand_index[block + s] = i;
      }
      i = i + ${w}u;
    }
    // 相 2: merge。各レーンの先頭同士の最大元 = 残り集合の最大元（各ブロックが降順）なので、
    // k ラウンド回せば値降順・同値なら index 昇順で出る。カーソルは k-1 回しか進まないので
    // block + cursor は常にブロック内
    var cursor = 0u;
    for (var r = 0u; r < ${k}u; r = r + 1u) {
      head_value[lid] = cand_value[block + cursor];
      head_index[lid] = cand_index[block + cursor];
      workgroupBarrier();
      var stride = ${w / 2}u;
      while (stride > 0u) {
        if (lid < stride) {
          let other = head_value[lid + stride];
          let other_at = head_index[lid + stride];
          if (topk_beats(other, other_at, head_value[lid], head_index[lid])) {
            head_value[lid] = other;
            head_index[lid] = other_at;
          }
        }
        workgroupBarrier();
        stride = stride / 2u;
      }
      let won = head_index[0u];
      if (lid == 0u) {
        values[row * ${k}u + r] = head_value[0u];
        indices[row * ${k}u + r] = i32(won);
      }
      // 走査は i ≡ lid (mod ${w}) の分担なので、勝った要素の持ち主は won % ${w} で決まる
      if (won % ${w}u == lid) {
        cursor = cursor + 1u;
      }
      // 次ラウンドが head を上書きする前に won の読み終わりを揃える
      workgroupBarrier();
    }
    row = row + nwg.x;
  }
}
`;
};

/**
 * uniform の Params（`{rows, dim, neg_inf}`）。WGSL の uniform アドレス空間では struct の
 * 整列が 16 バイトになるため、3 語ぶんの内容でも 16 バイト確保する MUST。
 *
 * MUST: `k` を載せない（WGSL に焼いてある — 同じ事実を 2 箇所で持つと、片方だけ更新した
 * dispatch が黙って別の本数を書く）。
 * MUST: `dim` は正整数（`dim = 0` の行に「大きい順の k 本」は無い）。契約側も
 * `1 ≤ k ≤ 最終次元` を要求するので、ここは二重の門。
 */
export const topkParams = (rows: number, dim: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("topk params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`topk params: dim は正整数（${dim}）`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  params[2] = TOPK_NEG_INF_BITS;
  return params;
};

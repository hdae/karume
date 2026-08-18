/**
 * states 形 attention（ADR 0067 決定 4 / 6 / 7）の 3 カーネル。1 ノード = 3 dispatch:
 *
 * | 段 | キー                                     | 役割                                            |
 * | -- | ---------------------------------------- | ----------------------------------------------- |
 * | ①  | `attention_state_qk:v1:f32:wg16x4`       | 論理 col 空間の `S` を**行ブロック窓で実体化**  |
 * | ②  | `attention_state_stats:v1:f32:wg256`     | 行ごとの `m = amax S` と `inv = 1/Σexp(S−m)`    |
 * | ③  | `attention_state_pv:v1:f32:wg16x4`       | `O = P @ V`（`P = exp(S−m)·inv` は**非実体化**）|
 *
 * 既存の融合 attention（src/kernels/attention.ts + GEMM 骨格）とは**別族**で、1 バイトも共有
 * しない。理由は 3 つで、どれも既存側を触らずに済ませるためではなく、意味論が違うため:
 *
 * 1. **K / V の出どころが 2 つ**（論理 col < pastLength は state スロット・以降は今 step の
 *    `ins`）で、GEMM のタイル充填は「1 本の連続バッファ」を前提にしている。
 * 2. **行統計の identity が −inf**（既存 ② は `-F32_MAX` の有限 sentinel）。states 形は padding
 *    行と空 context 行が**正規に**出るため、空行 → 出力 0 を構造で保証する必要がある
 *    （ADR 0067 決定 6 — 有限 sentinel での代用は MUST NOT）。
 * 3. **走査範囲が実行時値**（論理長 uniform から読む live 列数）で、容量 `C` に比例しない
 *    （ADR 0066 決定 3 の仕事量合格条件）。
 *
 * MUST: **ビット同一契約（分解経路との一致）は states 形に適用されない**（ADR 0067 決定 4 は
 * 分解経路を持たない — GQA モデルは SDPA 保存が必須）。代わりに従来どおり**決定性**が掛かる:
 * 同一キー → バイト同一 WGSL・同一入力 → 同一出力（縮約は col 昇順の逐次で固定）。
 *
 * ## 記号（正本 = ADR 0067 決定 4）
 *
 * | 記号     | 意味                                   | 出どころ                          |
 * | -------- | -------------------------------------- | --------------------------------- |
 * | `P`      | pastLength（確定済み KV の論理長）     | **実行時** — 論理長 uniform       |
 * | `Q`      | queryLength（今 step の実 token 数）   | **実行時** — 論理長 uniform       |
 * | `M`      | 物理 chunk 行数（`chunk_rows`）        | 静的（宣言 shape）                |
 * | `W`      | sliding window（`0` = full）           | 静的（attrs `window`）            |
 * | `C`      | スロットの行容量（`capacity`）         | 静的（states 宣言 + 束縛）        |
 * | `r`      | GQA の繰り返し数 `H / Hkv`             | 静的（導出値）                    |
 * | `D`      | head 次元                              | 静的                              |
 *
 * ## 論理 col 空間と live 範囲
 *
 * 論理 col は `[0, P+Q)`。`col < P` は past（**スロット**から読む）・`col ≥ P` は current
 * （**ins** の行 `col − P` から読む）。S が実体化するのは resident 範囲だけで、
 * `colBase = full: 0 / sliding: P − min(P, W−1)`（{@link stateColumnBase}）から
 * `n_live = (P − colBase) + Q` 列（{@link stateLiveColumns}）。
 *
 * S の格納は `[B·H, rows_block, colCap]` 行優先で、`colCap` は**静的**な列ストライド上限。
 * MUST: カーネルは `colCap` を**ストライドとしてだけ**使い、走査は `n_live` で切る（`colCap`
 * まで走らせると容量比例の仕事量になり ADR 0066 決定 3 の合格条件を落とす）。
 * MUST: `[n_live, colCap)` の残骸は**読者が触らない**ことで無害化する（②③ とも `n_live` で
 * 切る）。逆に `[0, n_live)` は述語外でも**必ず −inf を書く**（S は一時バッファで前回の残骸が
 * 居るため — 書かないと ② の amax が残骸を食う）。
 *
 * ## 述語は両側 MUST（ADR 0067 決定 4 の第 5 巡 high）
 *
 * 出力行 `row`（= `row_offset` + 局所行）に対し `col ≤ P + row`（causal）AND sliding のとき
 * `col ≥ max(0, P + row − W + 1)`。**上限だけの実装は禁止** — W=4 で row 1 が 5 個の key を
 * 見る沈黙混入になる。下限は `(P + row) − col < W` の形で持つ（{@link stateWindowFn} —
 * 同値変形: `col ≥ limit−W+1 ⟺ limit−col ≤ W−1 ⟺ limit−col < W`。引き算は causal 上限
 * `col ≤ limit` が**短絡してから**評価されるので u32 で巻き戻らない）。
 *
 * ## 有効行と pad 行（仕事量は Q に比例する — ADR 0066 決定 3 / 追記 1 の訂正式）
 *
 * 物理 chunk は `M` 行だが、有効データは先頭 `Q` 行の compact-prefix（ADR 0066 追記 6）。
 * 行ブロック `[row_offset, row_offset + rows_block)` の**有効行数**は
 * `clamp(Q − row_offset, 0, rows_block)`（{@link stateEffectiveRows}）で、
 *
 * - **①QK / ②stats は有効行だけを覆う**（dispatch 数もカーネルの行範囲も同じ 1 つの式から出る —
 *   ホスト側が幾何の純関数・WGSL 側が {@link stateEffectiveRowsWgsl} の写し）。有効行 0 の
 *   ブロックはホスト算出が 0 を返し、dispatch そのものが積まれない。
 * - **③PV は宣言 shape の全 M 行を書く**（full-write 不変条件は不変）が、**pad 行は live を
 *   1 列も走査せず全 D に厳密 `0.0` を書いて返す**。空行（述語を満たす col が 1 本も無い行）は
 *   **pad 行の部分集合**（valid 行は causal 自己参照 `col = P + row` を必ず含むので非空）なので、
 *   この 0 書きが空行 → 0（ADR 0067 決定 6）を**構造的に包含**し、同時に「非有限な V が
 *   `0 · NaN` で空行出力を NaN 化する」残穴も閉じる。② の空行ガードは**防御として残す**。
 *
 * ## 空行 → 0 の構成（ADR 0067 決定 6）
 *
 * ② の行 max は identity **−inf**。`amax == −inf`（空行）なら `(m, inv) = (0.0, 0.0)` を書き、
 * それ以外は `inv = 1/Σexp(S−amax)`（最大要素が `exp(0)=1` を出すので分母 ≥ 1）。③ の
 * `p = exp(S − m)·inv` は
 *
 * - 空行: `exp(−inf − 0) · 0 = 0 · 0 = 0`（厳密）
 * - 非空行の述語外: `exp(−inf − 有限) · inv = 0 · inv = 0`（厳密）
 *
 * となり、**厳密 0** が構造的に出る。MUST: ② の Σ ループを空行でも回さない（`exp(−inf −
 * (−inf))` = `exp(NaN)` = NaN が分母へ入る）。
 */

import { CodegenError } from "../codegen/errors.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import { assertU32Params } from "../codegen/params.ts";

/** ①QK / ③PV の workgroup 幅（① は列方向・③ は D 方向）。 */
export const STATE_ATTENTION_TILE_X = 16;

/** ①QK / ③PV の workgroup 高さ（両者とも行方向 — 行ブロックの局所行）。 */
export const STATE_ATTENTION_TILE_M = 4;

/** ② の workgroup 幅（1 行 = 1 workgroup の 256 幅ツリー縮約 — 既存 attention_stats と同型）。 */
export const STATE_STATS_WORKGROUP_SIZE = 256;

/** ② の行統計 1 行あたりの語数（`[0]` = 行の最大値 / `[1]` = `1/Σexp(S−m)`）。 */
export const STATE_STATS_STRIDE = 2;

/**
 * −inf の f32 ビット列（params の語で運ぶ）。
 *
 * MUST: WGSL に定数式で書かない（argmax / topk / safe_softmax と同じ理由 — 定数式の
 * `bitcast<f32>(0xff800000u)` を「const-expression が inf」としてシェーダ生成エラーにする
 * 実装がありうる）。
 */
export const STATE_NEG_INF_BITS = 0xff800000;

/**
 * 変種の判別子（**`:sliding` と `:gqa` の 2 ビットだけ**）。
 *
 * MUST: 並び順をここ 1 箇所で固定する（両方立ちうるので、順序が散ると同一構成が 2 通りの
 * キーを持つ — attention.ts の `maskKeyPart` / `gqaKeyPart` と同じ規律）。
 * MUST: `W` / `r` / `C` の**値そのものは載せない**（uniform で運ぶ — 載せると値の種類ぶん
 * パイプラインが増える）。
 */
export const stateVariantKeyPart = (sliding: boolean, gqa: boolean): string =>
  `${sliding ? ":sliding" : ""}${gqa ? ":gqa" : ""}`;

/**
 * sliding 変種かどうかの**唯一の判定**（`window = 0` = attrs 欄の不存在 = 全 context）。
 *
 * MUST: キー選択・WGSL 生成・params 検査の 3 者はこの 1 関数で揃える。片方が `window > 0` で
 * もう片方が別条件だと、full の params（`window = 0`）が sliding の WGSL に入り、
 * `window - 1u` の u32 アンダーフローと `col % 0u` の実装依存値で沈黙誤読になる。
 */
export const stateSliding = (window: number): boolean => window > 0;

export const stateQkKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_qk:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_ATTENTION_TILE_M}${
    stateVariantKeyPart(sliding, gqa)
  }`;

export const stateStatsKey = (sliding: boolean): string =>
  `attention_state_stats:v1:f32:wg${STATE_STATS_WORKGROUP_SIZE}${
    stateVariantKeyPart(sliding, false)
  }`;

export const statePvKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_pv:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_ATTENTION_TILE_M}${
    stateVariantKeyPart(sliding, gqa)
  }`;

/**
 * ①③ が共有する静的 params（**内容アドレスキャッシュ適格** — 毎 step 変わる値を含まない）。
 *
 * 語順（**この表が正本**。ホスト側は {@link stateAttentionParams} 1 本だけが組む）:
 *
 * | 語 | 欄           | 意味                                             | ① | ③ |
 * | -- | ------------ | ------------------------------------------------ | - | - |
 * | 0  | `rows_block` | この dispatch が担当する行数（S の行数）         | ✓ | ✓ |
 * | 1  | `row_offset` | chunk 内の先頭行（グローバル行 = offset + 局所） | ✓ | ✓ |
 * | 2  | `chunk_rows` | `M`（q / ins / O の行ストライド）                | ✓ | ✓ |
 * | 3  | `depth`      | `D`                                              | ✓ | ✓ |
 * | 4  | `kv_repeat`  | `r = H / Hkv`                                    | ✓ | ✓ |
 * | 5  | `window`     | `W`（`0` = full）                                | ✓ | ✓ |
 * | 6  | `capacity`   | `C`（スロットの行容量）                          | ✓ | ✓ |
 * | 7  | `col_cap`    | S の列ストライド                                 | ✓ | ✓ |
 * | 8  | `neg_inf`    | −inf の f32 ビット列                             | ✓ |   |
 * | 9  | `scale`      | 半スケール（**f32**）                            | ✓ |   |
 *
 * MUST: ①③ で同一の struct（= 同一の語順）にする。同じ幾何を 2 つの表で持つと、片方だけ
 * 更新した dispatch が例外なしに別のバッファ領域を読む。③ が読まない 2 語（`neg_inf` /
 * `scale`）ぶんの無駄より、語順が 1 箇所しかない性質が優先する。
 */
const STATE_PARAMS_STRUCT = `struct Params {
  rows_block: u32,
  row_offset: u32,
  chunk_rows: u32,
  depth: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
  col_cap: u32,
  neg_inf: u32,
  scale: f32,
}`;

/**
 * 論理長 uniform（ADR 0066 追記 4 の搬送路 — `GenerationContext` 所有の可変 8 バイト）。
 *
 * MUST: `P` / `Q` を静的 params に載せない（毎 step 値が変わるものを内容アドレスキャッシュへ
 * 載せると「キャッシュ無界成長」と「PreparedPlan ヒット時に更新不能」の両方を踏む）。
 * MUST: **最後の binding** に置く（束縛表の末尾 = context 所有の面、という並びを 4 カーネルで
 * 揃える。src/runtime/generation-context.ts の `lengths` がそのまま入る）。
 */
export const STATE_LENGTHS_STRUCT = `struct Lengths {
  past: u32,
  query: u32,
}`;

/**
 * 論理 col → スロット物理行の写像（**読み書き同式 MUST** — ADR 0067 決定 4）。
 *
 * ①QK / ③PV の**読み**と `state_append` の**書き**がこの 1 文字列を共有する
 * （src/kernels/state-append.ts が import する）。読み側だけ別式にすると、ring が一周した
 * 後の全読みが黙って別の行を指す（例外も NaN も出ない沈黙誤読）。
 */
export const stateSlotRowWgsl = (sliding: boolean): string =>
  `fn slot_row(col: u32) -> u32 {
  return ${sliding ? "col % params.window" : "col"};
}`;

/**
 * live 範囲（S が実体化する論理 col の窓）を決める 2 関数。
 *
 * `column_base` = full: `0` / sliding: `P − min(P, W−1)`（append 前なので row 0 の窓まで
 * 全行 resident — ADR 0067 決定 4）。`live_columns` = `(P − column_base) + Q`。
 */
const stateLiveWgsl = (sliding: boolean): string =>
  `fn column_base(past: u32) -> u32 {
  return ${sliding ? "past - min(past, params.window - 1u)" : "0u"};
}

fn live_columns(past: u32, query: u32) -> u32 {
  return ${sliding ? "min(past, params.window - 1u)" : "past"} + query;
}`;

/**
 * 述語（`limit = P + row`）。causal 上限に sliding の下限を **AND する MUST**。
 *
 * 下限は `limit − col < W`（`col ≥ limit − W + 1` の u32 安全形）。
 *
 * MUST: **引き算は `limit − col` の側**（`col + W > limit` にしない）。WGSL の `&&` は短絡なので
 * 先行する `col <= limit` が真のときしか評価されず、`limit − col` は決して巻き戻らない。逆に
 * `col + W` は **P が u32 の上限近くまで進んだ生成で加算が巻き戻り**、causal 対角（`col = limit`）
 * まで窓外と判定して行が静かに空になる。`limit − W + 1` を直接計算する形も
 * `limit < W−1` でアンダーフローするので採らない。
 */
const stateWindowFn = (sliding: boolean): string =>
  `fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit${sliding ? " && (limit - col) < params.window" : ""};
}`;

/**
 * 行ブロック内の**有効行数**を論理長から出す（{@link stateEffectiveRows} の WGSL 側の写し）。
 *
 * MUST: ホスト（dispatch 数の算出）と WGSL（行範囲の切り方）は**同じ 1 つの式**から出る。
 * ずれると ①② が書く行と ③ が pad と見なす行が食い違い、未書込みの S / stats を読む沈黙誤値に
 * なる（例外も NaN も出ない）。
 * MUST: 引き算はアンダーフローを避けて分岐で切る（`query - row_offset` は u32 で巻き戻る）。
 * NOTE: 3 カーネルとも params の欄名を `rows_block` / `row_offset` で揃えてあるので、この
 * 1 文字列をそのまま共有できる。
 */
const stateEffectiveRowsWgsl = `fn effective_rows(query: u32) -> u32 {
  if (query <= params.row_offset) {
    return 0u;
  }
  return min(params.rows_block, query - params.row_offset);
}`;

/**
 * ①QK の内積（**半スケール契約** — `scale` を q 側と k 側の**両方**へ載せてから積む）。
 *
 * MUST: 生成の実体は 1 箇所（この関数）。K の出どころがスロットと ins の 2 つあるぶん本文は
 * 2 つ生成されるが、式が 2 箇所に書かれていると片方だけ「内積の後に 1 度掛ける」形へ
 * 直された時に、スロット由来の列と ins 由来の列で丸めが変わる。
 */
const stateScoreFn = (name: string, array: string): string =>
  `fn ${name}(q_base: u32, k_base: u32) -> f32 {
  var acc = 0.0;
  for (var d = 0u; d < params.depth; d = d + 1u) {
    acc = acc + (q[q_base + d] * params.scale) * (${array}[k_base + d] * params.scale);
  }
  return acc;
}`;

/**
 * kv 平面の写像（`z = b·H + h` に対し `z / r = b·Hkv + h/r` が整数除算で厳密成立 —
 * ADR 0067 決定 2 と同じ恒等式）。
 *
 * MUST: 非 GQA 変種は除算そのものを生成しない（`r = 1` の経路に整数除算を残さない）。
 * 値域門（{@link assertStateGeometry}）が `r ≥ 1` を保証するので、GQA 変種のゼロ除算は
 * 起こらない。
 */
const kvPlaneWgsl = (gqa: boolean): string => gqa ? "z / params.kv_repeat" : "z";

/**
 * ①QK。1 invocation = S の 1 要素（`(局所行, live 列)`）で、内積は D の逐次ループ。
 *
 * 束縛（**binding 0 = 静的 params・最後 = 論理長**）:
 *
 * | binding | 資源                              |
 * | ------- | --------------------------------- |
 * | 0       | `Params`（uniform）               |
 * | 1       | `q` `[B,H,M,D]`                   |
 * | 2       | `ins_k` `[B,Hkv,M,D]`             |
 * | 3       | `slot_k` `[B,Hkv,C,D]`            |
 * | 4       | `s` `[B·H, rows_block, col_cap]`（書き） |
 * | 5       | `Lengths`（uniform — context 所有）|
 */
export const stateQkWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_qk (states 形の S 実体化, f32${sliding ? ", sliding window" : ""}${
    gqa ? ", GQA" : ""
  })
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> ins_k: array<f32>;
@group(0) @binding(3) var<storage, read> slot_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> s: array<f32>;
@group(0) @binding(5) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateWindowFn(sliding)}

${stateEffectiveRowsWgsl}

${stateScoreFn("score_slot", "slot_k")}

${stateScoreFn("score_ins", "ins_k")}

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_ATTENTION_TILE_M})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let local_row = gid.y;
  let cl = gid.x;
  // 端数タイルの空振り。live より右（[live, col_cap)）は残骸のまま残すのが正で、
  // 読者（②③）が live で切ることと対になっている。行は**有効行まで**（pad 行の S は
  // 誰も読まない — ③ が 0 を書いて返す）ので、仕事量が Q に比例する
  if (local_row >= effective_rows(lengths.query) || cl >= live) {
    return;
  }
  let z = gid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = ${kvPlaneWgsl(gqa)};
  // 述語外は -inf。live 範囲は**述語外でも必ず書く**（書かないと ② が前回の残骸を食う）
  var value = bitcast<f32>(params.neg_inf);
  if (in_window(col, past + row)) {
    if (col < past) {
      // past（col < P）はスロットから。物理行は読み書き同式の slot_row
      value = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth);
    } else {
      // current（col ≥ P）は今 step の ins の行 col − P から
      value = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth);
    }
  }
  s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
}
`;

/**
 * ② 行統計。1 行 = 1 workgroup(256) + 行方向 grid-stride（既存 attention_stats と同じ骨格で、
 * 違うのは **dim が実行時値**（live 列数）・**identity が −inf**・**空行ガード**の 3 点）。
 *
 * 束縛:
 *
 * | binding | 資源                                        |
 * | ------- | ------------------------------------------- |
 * | 0       | `Params`（uniform — 下の 6 語だけの別 struct）|
 * | 1       | `s`（読み）                                 |
 * | 2       | `stats` `[B·H·rows_block, 2]`（書き）       |
 * | 3       | `Lengths`（uniform）                        |
 *
 * MUST: 覆うのは**有効行だけ**（`B·H × effective_rows` 本）。統計の書き先は S と同じ
 * `z · rows_block + 局所行` で、pad 行のぶんは書かれないまま残る（読者が居ない — ③ は pad 行で
 * `stats` を 1 語も読まずに 0 を書く）。
 * MUST: `col_cap` は**ストライド**で、走査は live まで。
 * NOTE: `max` は NaN 伝播を保証しない（softmax / reduce.ts と同じ既知の乖離）。
 */
export const stateStatsWgsl = (sliding: boolean): string =>
  `// karume attention_state_stats (states 形の行統計 m = amax(S) と inv = 1/Σexp(S - m), f32${
    sliding ? ", sliding window" : ""
  })
struct Params {
  batch_heads: u32,
  rows_block: u32,
  row_offset: u32,
  col_cap: u32,
  window: u32,
  neg_inf: u32,
}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@group(0) @binding(3) var<uniform> lengths: Lengths;

${stateLiveWgsl(sliding)}

${stateEffectiveRowsWgsl}

var<workgroup> scratch: array<f32, ${STATE_STATS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${STATE_STATS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let neg_inf = bitcast<f32>(params.neg_inf);
  let live = live_columns(lengths.past, lengths.query);
  // 有効行は各 z 平面の**前詰め** rows 本。total = 0 ならループへ入らないので rows での
  // 除算・剰余は 0 除算にならない（ホストも 0 なら dispatch を積まない）
  let rows = effective_rows(lengths.query);
  let total = params.batch_heads * rows;
  var index = wid.x;
  while (index < total) {
    let row = (index / rows) * params.rows_block + index % rows;
    let base = row * params.col_cap;

    // ① 行の最大値。identity は **-inf**（有限 sentinel は MUST NOT — ADR 0067 決定 6）
    var hi = neg_inf;
    var i = lid;
    while (i < live) {
      hi = max(hi, s[base + i]);
      i = i + ${STATE_STATS_WORKGROUP_SIZE}u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${STATE_STATS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let amax = scratch[0u];
    // scratch の読み終わりを揃えてから ② で上書きする
    workgroupBarrier();

    // ② Σ exp(S - amax)。**空行（amax == -inf）は 1 度も回さない** —
    // exp(-inf - (-inf)) = exp(NaN) = NaN が分母へ入る
    let empty = amax == neg_inf;
    var acc = 0.0;
    if (!empty) {
      var j = lid;
      while (j < live) {
        acc = acc + exp(s[base + j] - amax);
        j = j + ${STATE_STATS_WORKGROUP_SIZE}u;
      }
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = ${STATE_STATS_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    if (lid == 0u) {
      // 空行は (0.0, 0.0)。③ の exp(-inf - 0) * 0 = 0 で出力が**厳密 0** になる
      var m = 0.0;
      var inv = 0.0;
      if (!empty) {
        m = amax;
        inv = 1.0 / scratch[0u];
      }
      stats[row * ${STATE_STATS_STRIDE}u] = m;
      stats[row * ${STATE_STATS_STRIDE}u + 1u] = inv;
    }
    // 次の行が scratch[lid] を上書きする前に scratch[0] の読み終わりを揃える
    workgroupBarrier();
    index = index + nwg.x;
  }
}
`;

/**
 * ③PV。1 invocation = O の 1 要素（`(局所行, D)`）で、縮約は **live 列の昇順逐次**（決定性）。
 *
 * 束縛:
 *
 * | binding | 資源                                |
 * | ------- | ----------------------------------- |
 * | 0       | `Params`（uniform — ① と同一 struct）|
 * | 1       | `s`（読み）                         |
 * | 2       | `stats`（読み）                     |
 * | 3       | `ins_v` `[B,Hkv,M,D]`               |
 * | 4       | `slot_v` `[B,Hkv,C,D]`              |
 * | 5       | `out` `[B,H,M,D]`（書き）           |
 * | 6       | `Lengths`（uniform）                |
 *
 * MUST: 出力は `row_offset` からの `rows_block` 行**全て**を書く（pad 行〈`row ≥ Q`〉も
 * 通常出力としては書かれる — ADR 0066 追記 6 の「不定 = 値が契約上無意味」であって未書込み
 * ではない）。ただし pad 行は **live を 1 列も走査せず厳密 `0.0`** を書く（① ② が有効行しか
 * 覆わないので S / stats が居ない + 仕事量を Q に比例させる + 空行 → 0 を構造的に包含する）。
 */
export const statePvWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_pv (states 形の O = P @ V, f32${sliding ? ", sliding window" : ""}${
    gqa ? ", GQA" : ""
  })
${STATE_PARAMS_STRUCT}
${STATE_LENGTHS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> ins_v: array<f32>;
@group(0) @binding(4) var<storage, read> slot_v: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> lengths: Lengths;

${stateSlotRowWgsl(sliding)}

${stateLiveWgsl(sliding)}

${stateEffectiveRowsWgsl}

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_ATTENTION_TILE_M})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let local_row = gid.y;
  if (d >= params.depth || local_row >= params.rows_block) {
    return;
  }
  let z = gid.z;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // pad 行（row ≥ Q）: live を走査せず**厳密 0** を書いて返す（full-write は保つ = ADR 0066
  // 追記 6 の「値が契約上無意味」を 0 で固定）。空行 ⊂ pad 行なので ADR 0067 決定 6 の
  // 「空行 → 厳密 0」はこの分岐が構造的に包含し、非有限 V による 0·NaN の穴も同時に閉じる
  if (local_row >= effective_rows(lengths.query)) {
    out[at] = 0.0;
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let base_col = column_base(past);
  let kv_plane = ${kvPlaneWgsl(gqa)};
  let s_row = z * params.rows_block + local_row;
  let s_base = s_row * params.col_cap;
  let amax = stats[s_row * ${STATE_STATS_STRIDE}u];
  let inv = stats[s_row * ${STATE_STATS_STRIDE}u + 1u];
  // 縮約は col 昇順の逐次（決定性）。述語外は S が -inf なので p = 0 が**厳密**に出る
  var acc = 0.0;
  for (var cl = 0u; cl < live; cl = cl + 1u) {
    let col = base_col + cl;
    let p = exp(s[s_base + cl] - amax) * inv;
    var value = 0.0;
    if (col < past) {
      value = slot_v[(kv_plane * params.capacity + slot_row(col)) * params.depth + d];
    } else {
      value = ins_v[(kv_plane * params.chunk_rows + (col - past)) * params.depth + d];
    }
    acc = acc + p * value;
  }
  out[at] = acc;
}
`;

/** ①③ が共有する静的幾何（{@link STATE_PARAMS_STRUCT} の語順の型側）。 */
export type StateAttentionGeometry = {
  /** この dispatch が担当する行数（S の行数 — `rows_block`）。 */
  readonly rowsBlock: number;
  /** chunk 内の先頭行。グローバル行 = `rowOffset` + 局所行。 */
  readonly rowOffset: number;
  /** `M`（q / ins / O の行ストライド = 物理 chunk 行数）。 */
  readonly chunkRows: number;
  /** `D`。 */
  readonly depth: number;
  /** `r = H / Hkv`（GQA の繰り返し数）。 */
  readonly kvRepeat: number;
  /** `W`（`0` = full — {@link stateSliding}）。 */
  readonly window: number;
  /** `C`（スロットの行容量）。 */
  readonly capacity: number;
  /** S の列ストライド上限。 */
  readonly colCap: number;
  /** 半スケール（q 側と k 側の両方へ載る値）。 */
  readonly scale: number;
};

/**
 * 静的幾何の値域門。
 *
 * MUST: `kvRepeat ≥ 1`（WGSL の u32 ゼロ除算は trap せず実装依存の値を返すので、`r = 0` は
 * 例外も NaN も出ないまま kv 平面が化ける — attention.ts の `assertKvRepeat` と同型）。
 * MUST: sliding は `1 ≤ W ≤ C`（ADR 0067 決定 4 ③）。`W = 0` で sliding の WGSL を撃つと
 * `window - 1u` がアンダーフローし `col % 0u` が実装依存値になる。
 * MUST: `colCap` は live 列の**静的上限**以上（full は `P + Q ≤ C` の context 側検査から `C`・
 * sliding は `(W−1) + M`）。足りないと ① の書きが範囲外へ落ち（robustness で捨てられ）、
 * ②③ が残骸を読む。
 * MUST: `rowOffset + rowsBlock ≤ M`（行ブロックが chunk からはみ出さない）。
 */
const assertStateGeometry = (where: string, geometry: StateAttentionGeometry): void => {
  const { rowsBlock, rowOffset, chunkRows, depth, kvRepeat, window, capacity, colCap } = geometry;
  assertU32Params(where, {
    rows_block: rowsBlock,
    row_offset: rowOffset,
    chunk_rows: chunkRows,
    depth,
    kv_repeat: kvRepeat,
    window,
    capacity,
    col_cap: colCap,
  });
  if (rowsBlock < 1 || chunkRows < 1 || depth < 1 || capacity < 1 || colCap < 1) {
    throw new CodegenError(
      `${where}: rows_block / chunk_rows / depth / capacity / col_cap は正整数` +
        `（${rowsBlock} / ${chunkRows} / ${depth} / ${capacity} / ${colCap}）`,
    );
  }
  if (kvRepeat < 1) {
    throw new CodegenError(`${where}: kv_repeat は正整数（${kvRepeat}）`);
  }
  if (rowOffset + rowsBlock > chunkRows) {
    throw new CodegenError(
      `${where}: 行ブロック [${rowOffset}, ${
        rowOffset + rowsBlock
      }) が chunk 行数 ${chunkRows} を超える`,
    );
  }
  if (stateSliding(window)) {
    if (window > capacity) {
      throw new CodegenError(
        `${where}: window ${window} が容量 ${capacity} を超える（ADR 0067 決定 4 ③）`,
      );
    }
    const need = window - 1 + chunkRows;
    if (colCap < need) {
      throw new CodegenError(
        `${where}: col_cap ${colCap} が sliding の live 上限 ${need}（= W−1 + M）に足りない`,
      );
    }
  } else if (colCap < capacity) {
    throw new CodegenError(
      `${where}: col_cap ${colCap} が full の live 上限 ${capacity}（= C）に足りない`,
    );
  }
  if (!Number.isFinite(geometry.scale)) {
    throw new CodegenError(`${where}: scale は有限の数値（${geometry.scale}）`);
  }
};

/**
 * ①③ が共有する uniform（{@link STATE_PARAMS_STRUCT} の 10 語 — uniform struct の整列で
 * 48 バイト確保する）。
 */
export const stateAttentionParams = (
  geometry: StateAttentionGeometry,
): Uint32Array<ArrayBuffer> => {
  assertStateGeometry("attention_state params", geometry);
  const params = new Uint32Array(12);
  params[0] = geometry.rowsBlock;
  params[1] = geometry.rowOffset;
  params[2] = geometry.chunkRows;
  params[3] = geometry.depth;
  params[4] = geometry.kvRepeat;
  params[5] = geometry.window;
  params[6] = geometry.capacity;
  params[7] = geometry.colCap;
  params[8] = STATE_NEG_INF_BITS;
  new Float32Array(params.buffer)[9] = geometry.scale;
  return params;
};

/**
 * ② の uniform（`{batch_heads, rows_block, row_offset, col_cap, window, neg_inf}` の 6 語 —
 * uniform struct の整列で 32 バイト確保する）。
 *
 * MUST: `B·H` と `rows_block` を**畳まずに**受ける（① ③ と同じ 2 軸）。畳んだ 1 語では
 * `z · rows_block + 局所行` へ戻せず、有効行の前詰めから S / stats の行を引けない。
 */
export const stateStatsParams = (
  batchHeads: number,
  rowsBlock: number,
  rowOffset: number,
  colCap: number,
  window: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("attention_state_stats params", {
    batch_heads: batchHeads,
    rows_block: rowsBlock,
    row_offset: rowOffset,
    col_cap: colCap,
    window,
  });
  if (batchHeads < 1 || rowsBlock < 1 || colCap < 1) {
    throw new CodegenError(
      `attention_state_stats params: batch_heads / rows_block / col_cap は正整数` +
        `（${batchHeads} / ${rowsBlock} / ${colCap}）`,
    );
  }
  const params = new Uint32Array(8);
  params[0] = batchHeads;
  params[1] = rowsBlock;
  params[2] = rowOffset;
  params[3] = colCap;
  params[4] = window;
  params[5] = STATE_NEG_INF_BITS;
  return params;
};

/**
 * sliding スロットの resident 範囲の先頭（論理 col）。full は常に 0。
 *
 * MUST: ホスト（dispatch 数の算出）と WGSL（{@link stateLiveWgsl}）で**同じ式**にする。
 * ずれると ① が書く範囲と ②③ が読む範囲が食い違い、読者側が広ければ残骸を食う。
 */
export const stateColumnBase = (window: number, past: number): number =>
  stateSliding(window) ? past - Math.min(past, window - 1) : 0;

/** live 列数 `n_live = (P − colBase) + Q`（S が実体化する列数 = ②③ の走査長）。 */
export const stateLiveColumns = (window: number, past: number, query: number): number =>
  past - stateColumnBase(window, past) + query;

/**
 * 行ブロック `[rowOffset, rowOffset + rowsBlock)` の**有効行数**（`row < Q` の局所行の本数）。
 *
 * MUST: ①QK / ②stats の dispatch 数はこの 1 関数から出す（ADR 0066 決定 3 の仕事量合格条件 —
 * 物理 chunk 行数 `M` に比例させると、`Q = 1` の decode が `M` 倍の行を回す）。
 * MUST: WGSL 側（{@link stateEffectiveRowsWgsl}）と**同じ式**。ホストが幾何の純関数で、
 * カーネルはその写し。
 */
export const stateEffectiveRows = (
  rowsBlock: number,
  rowOffset: number,
  query: number,
): number => Math.min(rowsBlock, Math.max(0, query - rowOffset));

/** `queryLength` の値域門（搬送先は u32・`Q ≥ 1` は `GenerationContext` の契約と同じ）。 */
const assertQueryLength = (where: string, query: number): void => {
  assertU32Params(where, { query });
  if (query < 1) {
    throw new CodegenError(`${where}: queryLength は 1 以上（${query}）`);
  }
};

/** 論理長の値域門（搬送先は u32・`Q ≥ 1` は `GenerationContext` の契約と同じ）。 */
const assertLengths = (where: string, past: number, query: number): void => {
  assertU32Params(where, { past });
  assertQueryLength(where, query);
};

/**
 * dispatch 幾何（**論理長から算出する側** — ADR 0066 決定 3 / ADR 0067 決定 4）。
 *
 * MUST: `capacity` / `colCap` を**持たない**。持てば「容量に比例した workgroup 数」を書ける
 * ようになり、131K 容量 × 短系列で 3 桁の無駄という ADR 0066 決定 3 の不合格形へ静かに戻る。
 */
export type StateDispatchGeometry = {
  /** `B·H`（z 軸 — 1 workgroup = 1 (b, h)）。 */
  readonly batchHeads: number;
  readonly rowsBlock: number;
  /** chunk 内の先頭行（有効行数 `clamp(Q − rowOffset, 0, rowsBlock)` の算出に要る）。 */
  readonly rowOffset: number;
  readonly depth: number;
  /** `W`（`0` = full — live 列数が `min(P, W−1) + Q` で頭打ちになる）。 */
  readonly window: number;
};

/** dispatch 幾何の値域門（3 軸とも正整数・`rowOffset` は 0 可）。 */
const assertDispatchGeometry = (where: string, geometry: StateDispatchGeometry): void => {
  const { batchHeads, rowsBlock, rowOffset, depth, window } = geometry;
  assertU32Params(where, {
    batch_heads: batchHeads,
    rows_block: rowsBlock,
    row_offset: rowOffset,
    depth,
    window,
  });
  if (batchHeads < 1 || rowsBlock < 1 || depth < 1) {
    throw new CodegenError(
      `${where}: batchHeads / rowsBlock / depth は正整数（${batchHeads} / ${rowsBlock} / ${depth}）`,
    );
  }
};

/**
 * ①QK の workgroup 数 `[live 列, 有効行, B·H]`。
 *
 * MUST: 列軸は **live 列数**（`colCap` でも `C` でもない）・行軸は**有効行数**
 * （`rowsBlock` でも `M` でもない）。この 2 軸が仕事量合格条件（∝ `Q × (有効 past + Q)`）の
 * 機構そのもので、容量を渡せないよう {@link StateDispatchGeometry} から容量を外してある。
 * MUST: 上限超過は fail loudly（タイル系 — 縮退させると S のタイルが欠落し、②③ が残骸を
 * 読んだまま例外なしに進む）。
 * NOTE: 有効行 0 のブロック（`rowOffset ≥ Q`）は行軸が 0 になり、呼び手は dispatch そのものを
 * 積まない（{@link stateEffectiveRows} の doc）。
 */
export const stateQkWorkgroups = (
  geometry: StateDispatchGeometry,
  past: number,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①QK`, geometry);
  assertLengths(`${where} ①QK`, past, query);
  const live = stateLiveColumns(geometry.window, past, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [
    tiledWorkgroups(live, STATE_ATTENTION_TILE_X, limit, `${where} ①QK`),
    tiledWorkgroups(rows, STATE_ATTENTION_TILE_M, limit, `${where} ①QK`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①QK`),
  ];
};

/**
 * ② の workgroup 数 `[B·H × 有効行, 1, 1]`（行方向 grid-stride なので上限超過は**縮退**）。
 *
 * 覆うのは有効行だけ（pad 行の統計は誰も読まない）。live の走査は行ループの内側なので、
 * 総反復回数は `B·H × 有効行 × live` = 仕事量合格条件どおり `Q × (有効 past + Q)` に比例する。
 */
export const stateStatsWorkgroups = (
  geometry: StateDispatchGeometry,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ②stats`, geometry);
  assertQueryLength(`${where} ②stats`, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [gridStrideWorkgroups(geometry.batchHeads * rows, 1, limit), 1, 1];
};

/**
 * ③PV の workgroup 数 `[D, 行, B·H]`。
 *
 * MUST: 行軸は **`rows_block` 全て**（① ② と違って有効行で切らない — pad 行も書くのが
 * full-write 不変条件）。仕事量が Q に比例するのはカーネル側で、pad 行の invocation は
 * live を 1 列も回さず `0.0` を 1 語書いて返る。
 * MUST: 上限超過は fail loudly（タイル系 — 縮退させると O の一部が未書き込みのまま残り、
 * full-write 不変条件が黙って崩れる）。
 */
export const statePvWorkgroups = (
  geometry: StateDispatchGeometry,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ③PV`, geometry);
  return [
    tiledWorkgroups(geometry.depth, STATE_ATTENTION_TILE_X, limit, `${where} ③PV`),
    tiledWorkgroups(geometry.rowsBlock, STATE_ATTENTION_TILE_M, limit, `${where} ③PV`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ③PV`),
  ];
};

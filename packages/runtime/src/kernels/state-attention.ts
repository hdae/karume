/**
 * states 形 attention（ADR 0067 決定 4 / 6 / 7）の 3 カーネル。1 ノード = 3 dispatch:
 *
 * | 段 | キー                                     | 役割                                            |
 * | -- | ---------------------------------------- | ----------------------------------------------- |
 * | ①  | `attention_state_qk:v1:f32:wg16x4`       | 論理 col 空間の `S` を**行ブロック窓で実体化**  |
 * | ①' | `attention_state_qk:v1:f32:wg16x16:par`  | ① の **D 並列縮約**変種（opt-in — 下記）       |
 * | ①ₜ | `attention_state_qk:v1:f32:reg<幾何>`     | ① の **K 行タイル共有**変種（既定 — 下記）      |
 * | ②  | `attention_state_stats:v2:f32:wg256`     | 行ごとの `m = amax S` と `inv = 1/Σexp(S−m)`    |
 * | ③  | `attention_state_pv:v1:f32:wg16x4`       | `O = P @ V`（`P = exp(S−m)·inv` は**非実体化**）|
 * | ③' | `attention_state_pv:v1:f32:wg16x16:par`  | ③ の **KV 並列縮約**変種（opt-in — 下記）      |
 * | ③ₜ | `attention_state_pv:v1:f32:reg<幾何>`     | ③ の **V 行タイル共有**変種（既定 — 下記）      |
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
 * ## ③' KV 並列縮約変種（`SessionOptions.stateAttentionReduce: "parallel"` — perf-ledger K-12）
 *
 * ③ は 1 invocation が O の 1 要素を live 列の**逐次ループ**で積むため、decode（M=1）では
 * 有効 invocation が `D × B·H`（Gemma 4 E2B の full 層で 4,096）に固定され、KV 長が伸びるほど
 * 1 スレッドの逐次長だけが伸びる（P=16K で attention が decode GPU 時間の 72% —
 * docs/research/2026-09-03-gemma4-context-length-sweep.md）。③' は workgroup を
 * `TILE_X（D 方向）× KV_LANES（KV 方向）` の 2 次元に組み替え、レーン `l` が `cl ≡ l (mod KV_LANES)`
 * の列を昇順に部分累積し、workgroup 共有メモリで**固定順の木縮約**（stride 8 → 4 → 2 → 1）に
 * 畳む。dispatch 数・中間バッファは ③ と同じ（増えるのは workgroup 内のレーンだけ）。
 *
 * MUST: 縮約順が ③ と違うので**ビット同一ではない**（決定性は保つ — 同一入力 → 同一出力）。
 * ADR 0058 の opt-in 席で、既定は ③（参照経路）。検証門は 3 点セット（参照経路の門は無変更・
 * ③' の A/B 帯門 = tests/gpu_state_attention_parallel_test.ts・census 門 =
 * tests/gpu_state_execution_test.ts）。
 * MUST: **③' が選ばれるのは `M < 16` の計画だけ**（適用条件は {@link statePvTiledEligible} の
 * 裏側）。`M ≥ 16` は席に依らず ③ₜ（下記・③ とビット同一）が取る — 席が `"parallel"` でも
 * prefill 計画の値は参照経路のものになる。K-12 の実測で ③' の利得は decode に閉じており
 * （prefill 側は誤差内 — docs/research/2026-09-03-gemma4-chunklength-k12-sweep.md）、
 * traffic を削る ③ₜ を優先しても失うものが無い。
 * MUST: pad 行の分岐は **workgroup 一様**（局所行は `workgroup_id.y` 由来）なので barrier の
 * 手前で返してよいが、`d ≥ D` のレーンは barrier に参加させる（return しない — 走査を空回り
 * させて `0.0` を寄与する）。WGSL の barrier は一様制御流の外に置けない。
 *
 * ## ①' D 並列縮約変種（**席は ③' と同じ** `stateAttentionReduce: "parallel"` — perf-ledger K-14）
 *
 * ① は 1 invocation が S の 1 要素（`(局所行, live 列)`）を持ち、内積が `D` の**逐次ループ**
 * （1 スレッドが D 本の積和）。decode（M=1）では live 列が伸びるほど invocation 数は増えるが、
 * 1 invocation の遅延は D 逐次のまま長く、P=16K で ①QK が decode GPU 時間の 17% を占める
 * （K-12 後の内訳 — docs/research/2026-09-03-gemma4-chunklength-k12-sweep.md）。①' は ③' と
 * **対称**に workgroup を `TILE_X（列方向 = S の cl）× D_LANES（D 方向）` へ組み替え、1 workgroup =
 * 局所行 1 本（`workgroup_id.y`）× `TILE_X` 本の列。レーン `l` が `d = l, l + D_LANES, …` を
 * 昇順に部分累積し、workgroup 共有メモリで**固定順の木縮約**（stride 8 → 4 → 2 → 1）に畳んで
 * レーン 0 が S の 1 語を書く。dispatch 数・束縛・params・中間バッファは ① と同じ
 * （増えるのは workgroup 内のレーンだけ・行軸だけが `⌈有効行 / TILE_M⌉` から「有効行」へ変わる）。
 *
 * MUST: 縮約順が ① と違うので**ビット同一ではない**（決定性は保つ）。**席は ③' と同じ 1 つ**
 * （`"parallel"` を指定すると ①' と ③' が一緒に選ばれる — 2026-09-06 裁定。新しいノブは作らない）。
 * MUST: **①' が選ばれるのは `M = 1` の計画だけ**（適用条件は
 * {@link stateQkParallelEligible}）。prefill 計画（M > 1）は席が `"parallel"` でも ① のまま走る
 * — prefill では ① が既に行 × 列で埋まっており、①' は行タイル幅を落として barrier を積む
 * ぶんだけ遅くなると実測した（2026-09-06 — 詳細は同関数の WHY）。③' の側は `M < 16` の計画
 * だけが取る（`M ≥ 16` は席に依らず ③ₜ）ので、**席 1 つで 2 段の適用範囲が違う**形になる
 * （M=1 の decode では ①' と ③' が揃って選ばれる）。
 * MUST: ① の契約は 1 つも動かさない — 述語（causal + sliding 下限）外は live 範囲内なら
 * **必ず −inf を書く**（② が残骸を食わないため）・`cl ≥ live` の列と pad 行は書かない・
 * scale の掛け方（半スケールを q 側と k 側の両方へ）と −inf のビット（`params.neg_inf`）は ① と同一。
 * MUST: pad 行の分岐は **workgroup 一様**（`workgroup_id.y` 由来）なので barrier の手前で
 * 返してよいが、`cl ≥ live` の列と `d ≥ D` のレーンは return せず**空回りで 0 を寄与する**
 * （③' と同じ理由 — WGSL の barrier は一様制御流の外に置けない）。
 *
 * ## ①ₜ K 行タイル共有変種（**席に依らない既定経路** — perf-ledger K-13）
 *
 * ① は 1 invocation = S の 1 要素なので、K の 1 行を**行の本数ぶん**読み直す。prefill
 * （M = 768）では 14.7K token の chunk で ①QK が chunk GPU 時間の 40% を占める traffic 律速
 * （docs/research/2026-09-06-state-qk-parallel-k14.md の内訳）。①ₜ は融合 attention の
 * `attention_qk` が既に持つ **GEMM 骨格**（共有メモリのタイル・レジスタブロック・M バケット幾何
 * — src/kernels/gemm.ts）に states 用の断片を差した 1 本で、K を行タイルへ 1 度だけ載せて
 * `tileM` 行の q で使い回す。行列の対応は A = この行ブロックの q（`[rows_block × depth]`）・
 * B = Kᵀ（`[depth × live]`・列 = 論理 col の 2 源）・出力 = S（① と同じ添字）。
 *
 * MUST: **① とビット同一**。1 出力要素あたりの加算順は骨格側で `d` 昇順の逐次に固定されており
 * （K タイル 16 昇順 — ADR 0022 決定 3）、① の `stateScoreFn` の逐次と厳密に一致する。半スケールも
 * 1 項の式（`(q·scale) · (k·scale)`）で揃える（A ローダで q 側・B ローダで k 側）。だから席
 * （`stateAttentionReduce`）に依らない**既定経路**で選べる — 縮約順が変わる ①' とは性格が違う。
 * MUST: 実効 `N` は**実行時の live**。骨格の `Dims.n` には静的上界 `col_cap` を入れるが、B ローダは
 * `cl < live` の列しか読まず store も `cl < live` しか書かない（`[live, col_cap)` の残骸は ① と
 * 同じく触らない）。行は `local_row < effective_rows(query)` まで（pad 行は 1 語も書かない）。
 * MUST: 述語は ① と同じ `in_window(col, past + row)`。live 範囲内なら述語外でも **−inf を必ず
 * 書く**（② が残骸を食わないため）。述語外の列の積和は回してよい（読む K 行は範囲内なので安全で、
 * 値は書き出しの `select` で捨てる）。
 * MUST: 適用条件は {@link stateQkTiledEligible}（`M ≥ 16`）。優先順は M=1 かつ席が `"parallel"` →
 * ①' / `M ≥ 16` → ①ₜ / それ以外 → ①（判定は runtime 側 `#buildStateAttention` の 1 箇所）。
 *
 * ## ③ₜ V 行タイル共有変種（**席に依らない既定経路** — perf-ledger K-13 段 2）
 *
 * ③ は 1 invocation = O の 1 要素なので、V の 1 行を**行の本数ぶん**読み直す（①ₜ が K で潰した
 * のと同じ traffic 律速が、prefill では V 側にもそのまま居る）。③ₜ は ①ₜ と同じ GEMM 骨格
 * （src/kernels/gemm.ts）に states 用の断片を差した 1 本で、V を行タイルへ 1 度だけ載せて
 * `tileM` 行の P で使い回す。行列の対応は A = P（`[rows_block × live]`・**S と行統計から充填時に
 * 計算**して実体化しない）・B = V（`[live × depth]`・行 = 論理 col の 2 源）・出力 = O
 * （③ と同じ添字）。
 *
 * MUST: **③ とビット同一**。1 項の式は ③ と同じ `p = exp(S − m) · inv` を A ローダで組み
 * （融合 attention ③PV の `probability` と同じ形）、縮約は骨格側で **col 昇順の逐次**に固定
 * されている（K タイル 16 昇順 — ADR 0022 決定 3）ので、③ の col 昇順逐次と加算順が厳密に
 * 一致する。だから席（`stateAttentionReduce`）に依らない**既定経路**で選べる。
 * MUST: 実効 `K`（縮約長）は**実行時の live**。骨格の `Dims.k` には S の列ストライド `col_cap` を
 * 入れるが、K タイルループの上限は live から作る（`col_cap` で回すと容量比例の仕事量になり
 * ADR 0066 決定 3 の合格条件を落とす）。A ローダは `cl < live` の列しか読まない。
 * MUST: 行は `rows_block` **全て**を書く（③ と同じ full-write 不変条件 — dispatch の行軸も
 * `rowsBlock`）。ただし pad 行（`local_row ≥ 有効行`）は **`acc` ではなく厳密 `0.0`** を書く
 * （③ の「pad 行は live を 1 列も走査せず 0」と同値。A を 0 で埋めるだけでは、非有限な V が
 * 混ざったとき `0 · NaN` で pad 行が NaN 化して ③ と 1 語違う）。
 * MUST: 行タイルが有効行を 1 行も含まないなら **K ループを 1 周も回さない**（③ の pad 行が
 * live を走査しないことの写し — 仕事量が Q に比例する機構）。判定は `workgroup_id.y` 由来で
 * workgroup 一様なので、内側の `workgroupBarrier` の一様性要件を壊さない。
 * MUST: 適用条件は {@link statePvTiledEligible}（`M ≥ 16`）。優先順は `M ≥ 16` → ③ₜ（席に
 * 依らない）/ `M < 16` かつ席が `"parallel"` → ③' / それ以外 → ③（判定は runtime 側
 * `#buildStateAttention` の 1 箇所）。
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

import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../codegen/dispatch.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  gemmGeometryForRows,
  gemmGeometryTileKeyPart,
  gemmTileM,
  gemmTileN,
} from "./gemm-geometry.ts";

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

/**
 * ①' の D 方向レーン数（workgroup = `TILE_X × D_LANES` = 256 スレッド — ③' と同じ
 * ポータビリティの床 `maxComputeInvocationsPerWorkgroup` の仕様既定 256 に収める）。
 */
export const STATE_QK_D_LANES = 16;

/** ①' のキー（`:par` が census 門の目印 — ① と同じ変種ビットを後置）。 */
export const stateQkParallelKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_qk:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_QK_D_LANES}:par${
    stateVariantKeyPart(sliding, gqa)
  }`;

/**
 * ①' の**適用条件** — `M`（物理 chunk 行数）が 1 の計画だけ。席（`stateAttentionReduce`）が
 * `"parallel"` でも、この条件を満たさない計画は ① のまま走る（席の判定は runtime 側
 * `#buildStateAttention` が持ち、ここは計画の形だけを見る純関数）。
 *
 * WHY: ①' が縮めるのは「1 invocation が `D` 本の積和を逐次で回す遅延」で、それが律速なのは
 * 有効 invocation が `live 列 × 1 行` しか無い decode（M=1）だけ。prefill（M=768）では ① が
 * 既に行 × 列で埋まっており、①' は行タイル幅を `TILE_M` = 4 から 1 へ落として 4 倍の
 * workgroup と barrier を積むだけになる（1 workgroup = 1 行なので K 行の行間再利用も失う）。
 * 実測（2026-09-06）でも decode は 6.3 → 3.7〜4.1 ms/token（×1.55〜1.72）で効いた一方、
 * prefill の ①QK は GPU 時間が 1.5〜1.9 倍に伸び、prefill 全体で +30〜60% 逆行した。
 * MUST: 条件は**実測した範囲に留める**（ADR 0082 決定 4 と同じ規律 — 「M が小さければ得だろう」
 * の外挿で `M <= 4` などに広げない。効くと測ったのは M=1 だけ）。③'（PV）はこの門を持たない
 * ことが対称でないように見えるが、③' は prefill でも逆行しないことを 2026-09-03 の実測で
 * 確かめてある（縮約するのが KV 長方向で、prefill でも 1 スレッドの逐次長が伸びるため）。
 */
export const stateQkParallelEligible = (chunkRows: number): boolean => chunkRows === 1;

/**
 * ①ₜ のキー（GEMM 骨格のタイル経路 — perf-ledger K-13）。幾何の綴りは gemm 側と同じ断片
 * （{@link gemmGeometryTileKeyPart}）で、`M` のバケットがそのまま載る。
 *
 * MUST: 幾何を載せる（① の `wg16x4` に当たる位置）。載せないと、`M` バケットの違う 2 つの計画が
 * 同じキーで別の WGSL を要求し、「同一キー → バイト同一 WGSL」が崩れる。
 * MUST: `chunkRows` は生成（gemm.ts の `attentionStateQkTiledWgsl`）へ渡すものと**同じ値**
 * （gemm.ts の `gemmKeyPart` の MUST と同文 — 片方だけ渡し忘れるとキャッシュに載った別幾何の WGSL が
 * dispatch 数と噛み合わずに出力タイルが欠落する）。
 */
export const stateQkTiledKey = (sliding: boolean, gqa: boolean, chunkRows: number): string =>
  `attention_state_qk:v1:f32:${gemmGeometryTileKeyPart(gemmGeometryForRows(chunkRows))}${
    stateVariantKeyPart(sliding, gqa)
  }`;

/**
 * ①ₜ の**適用条件** — `M`（物理 chunk 行数）が最小の行タイル辺 16 以上の計画だけ。
 *
 * WHY: ① は 1 invocation = S の 1 要素で、K の 1 行を**行の本数ぶん**読み直す traffic 律速。
 * ①ₜ は GEMM 骨格の共有メモリタイルへ K を 1 度だけ載せ、その行タイル（`M` バケットの
 * `tileM`）ぶんの q 行で使い回す。読み直しが減るのは行タイルに 2 行以上載るときで、
 * 最小バケット M16N16 の `tileM` が 16 なので、`M < 16` はタイルの大半が空振りして
 * ① と同じ traffic のまま barrier と共有メモリのぶんだけ損になる。
 * MUST: **① とビット同一**（1 出力要素あたりの加算順が d 昇順の逐次で一致 — ADR 0022 決定 3 の
 * 数値契約が骨格側の不変条件）。だから席（`stateAttentionReduce`）に依らない**既定経路**で、
 * 縮約順が変わる ①' とは性格が違う。
 * MUST: `M = 1` の計画では ①' の適用条件（{@link stateQkParallelEligible}）と重ならない
 * （1 < 16）。優先順は runtime 側 `#buildStateAttention` が 1 箇所で持つ。
 */
export const stateQkTiledEligible = (chunkRows: number): boolean => chunkRows >= 16;

// v2: 行 max を nan_max へ（全 NaN 行が空行判定へ化けて stats (0,0) になる穴を塞ぐ —
// ADR 0020。非 NaN 入力ではビット不変）
export const stateStatsKey = (sliding: boolean): string =>
  `attention_state_stats:v2:f32:wg${STATE_STATS_WORKGROUP_SIZE}${
    stateVariantKeyPart(sliding, false)
  }`;

export const statePvKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_pv:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_ATTENTION_TILE_M}${
    stateVariantKeyPart(sliding, gqa)
  }`;

/**
 * ③' の KV 方向レーン数（workgroup = `TILE_X × KV_LANES` = 256 スレッド — ポータビリティの床
 * `maxComputeInvocationsPerWorkgroup` の仕様既定 256 に収める）。
 */
export const STATE_PV_KV_LANES = 16;

/** ③' のキー（`:par` が census 門の目印 — ③ と同じ変種ビットを後置）。 */
export const statePvParallelKey = (sliding: boolean, gqa: boolean): string =>
  `attention_state_pv:v1:f32:wg${STATE_ATTENTION_TILE_X}x${STATE_PV_KV_LANES}:par${
    stateVariantKeyPart(sliding, gqa)
  }`;

/**
 * ③ₜ のキー（GEMM 骨格のタイル経路 — perf-ledger K-13 段 2）。綴りの規律は ①ₜ の
 * {@link stateQkTiledKey} と同文で、幾何断片（{@link gemmGeometryTileKeyPart}）に `M` の
 * バケットがそのまま載る。
 *
 * MUST: 幾何を載せる（③ の `wg16x4` に当たる位置）。載せないと、`M` バケットの違う 2 つの計画が
 * 同じキーで別の WGSL を要求し、「同一キー → バイト同一 WGSL」が崩れる。
 * MUST: `chunkRows` は生成（gemm.ts の `attentionStatePvTiledWgsl`）と dispatch
 * （{@link statePvTiledWorkgroups}）へ渡すものと**同じ値**。
 */
export const statePvTiledKey = (sliding: boolean, gqa: boolean, chunkRows: number): string =>
  `attention_state_pv:v1:f32:${gemmGeometryTileKeyPart(gemmGeometryForRows(chunkRows))}${
    stateVariantKeyPart(sliding, gqa)
  }`;

/**
 * ③ₜ の**適用条件** — `M`（物理 chunk 行数）が最小の行タイル辺 16 以上の計画だけ。
 *
 * WHY: ③ は 1 invocation = O の 1 要素で、V の 1 行を**行の本数ぶん**読み直す traffic 律速
 * （①ₜ が K で潰したのと同じ形が V 側に残っている）。③ₜ は GEMM 骨格の共有メモリタイルへ V を
 * 1 度だけ載せ、その行タイル（`M` バケットの `tileM`）ぶんの P の行で使い回す。読み直しが減るのは
 * 行タイルに 2 行以上載るときで、最小バケット M16N16 の `tileM` が 16 なので、`M < 16` は
 * タイルの大半が空振りして ③ と同じ traffic のまま barrier と共有メモリのぶんだけ損になる。
 * MUST: **③ とビット同一**（1 項の式が `exp(S − m)·inv` で同じ・加算順が col 昇順の逐次で
 * 一致 — ADR 0022 決定 3 の数値契約が骨格側の不変条件）。だから席（`stateAttentionReduce`）に
 * 依らない**既定経路**で、縮約順が変わる ③' とは性格が違う。
 * MUST: しきい値は ①ₜ（{@link stateQkTiledEligible}）と**同じ 16**だが、関数は別に持つ
 * （①ₜ と ③ₜ は別カーネルで、片方だけ適用範囲を動かせる形にしておく）。`M = 1` の decode は
 * どちらの条件も満たさないので、席どおり ①' / ③' が選ばれる。
 */
export const statePvTiledEligible = (chunkRows: number): boolean => chunkRows >= 16;

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
 * 述語・幾何の断片が読む uniform 変数名の既定（① ①' ② ③ ③' と `state_append` の `params`）。
 *
 * ①ₜ（GEMM 骨格のタイル経路）だけは骨格が持つ `dims` を binding 0 に置くので、断片へ
 * {@link STATE_TILED_UNIFORM} を渡して**同じ 1 つの式**から生成する（式を書き写すと、片方だけ
 * 直された時に読み書きが黙ってずれる）。
 */
const STATE_UNIFORM = "params";

/**
 * 論理 col → スロット物理行の写像（**読み書き同式 MUST** — ADR 0067 決定 4）。
 *
 * ①QK / ①ₜ / ③PV の**読み**と `state_append` の**書き**がこの 1 文字列を共有する
 * （src/kernels/state-append.ts が import する）。読み側だけ別式にすると、ring が一周した
 * 後の全読みが黙って別の行を指す（例外も NaN も出ない沈黙誤読）。
 */
export const stateSlotRowWgsl = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn slot_row(col: u32) -> u32 {
  return ${sliding ? `col % ${uniform}.window` : "col"};
}`;

/**
 * live 範囲（S が実体化する論理 col の窓）を決める 2 関数。
 *
 * `column_base` = full: `0` / sliding: `P − min(P, W−1)`（append 前なので row 0 の窓まで
 * 全行 resident — ADR 0067 決定 4）。`live_columns` = `(P − column_base) + Q`。
 */
const stateLiveWgsl = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn column_base(past: u32) -> u32 {
  return ${sliding ? `past - min(past, ${uniform}.window - 1u)` : "0u"};
}

fn live_columns(past: u32, query: u32) -> u32 {
  return ${sliding ? `min(past, ${uniform}.window - 1u)` : "past"} + query;
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
const stateWindowFn = (sliding: boolean, uniform = STATE_UNIFORM): string =>
  `fn in_window(col: u32, limit: u32) -> bool {
  return col <= limit${sliding ? ` && (limit - col) < ${uniform}.window` : ""};
}`;

/**
 * 行ブロック内の**有効行数**を論理長から出す（{@link stateEffectiveRows} の WGSL 側の写し）。
 *
 * MUST: ホスト（dispatch 数の算出）と WGSL（行範囲の切り方）は**同じ 1 つの式**から出る。
 * ずれると ①② が書く行と ③ が pad と見なす行が食い違い、未書込みの S / stats を読む沈黙誤値に
 * なる（例外も NaN も出ない）。
 * MUST: 引き算はアンダーフローを避けて分岐で切る（`query - row_offset` は u32 で巻き戻る）。
 * NOTE: 3 カーネルとも params の欄名を `rows_block` / `row_offset` で揃えてあるので、この
 * 1 文字列をそのまま共有できる。①ₜ だけは行数が骨格の `dims.m` に居る（Dims の先頭 3 語は
 * `m` / `n` / `k` 固定）ので、欄名も引数で受ける。
 */
const stateEffectiveRowsWgsl = (uniform = STATE_UNIFORM, rows = "rows_block"): string =>
  `fn effective_rows(query: u32) -> u32 {
  if (query <= ${uniform}.row_offset) {
    return 0u;
  }
  return min(${uniform}.${rows}, query - ${uniform}.row_offset);
}`;

/**
 * ①QK の内積（**半スケール契約** — `scale` を q 側と k 側の**両方**へ載せてから積む）。
 *
 * MUST: 生成の実体は 1 箇所（この関数）。K の出どころがスロットと ins の 2 つあるぶん本文は
 * 2 つ生成されるが、式が 2 箇所に書かれていると片方だけ「内積の後に 1 度掛ける」形へ
 * 直された時に、スロット由来の列と ins 由来の列で丸めが変わる。
 * MUST: ①' の**部分和**（`lanes` 指定 — `d = lane, lane + lanes, …`）も同じ 1 箇所から出す。
 * 積の式を ① と ①' で別々に書くと、片方だけ半スケールを畳む形へ直された時に A/B 帯の根拠
 * （「同じ積を違う順に足しているだけ」）が黙って崩れる。走査の**開始と刻み**だけが変わる。
 */
const stateScoreFn = (name: string, array: string, lanes?: number): string =>
  `fn ${name}(q_base: u32, k_base: u32${lanes === undefined ? "" : ", lane: u32"}) -> f32 {
  var acc = 0.0;
  for (var d = ${lanes === undefined ? "0u" : "lane"}; d < params.depth; d = d + ${
    lanes === undefined ? "1u" : `${lanes}u`
  }) {
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
const kvPlaneWgsl = (gqa: boolean, uniform = STATE_UNIFORM): string =>
  gqa ? `z / ${uniform}.kv_repeat` : "z";

/**
 * ①ₜ（GEMM 骨格のタイル経路 — src/kernels/gemm.ts の `attentionStateQkTiledWgsl`）が読む
 * uniform 変数名。骨格が binding 0 に `dims: Dims` を置くので、① の `params` とは名前だけが違う。
 */
const STATE_TILED_UNIFORM = "dims";

/**
 * ①ₜ が使う述語・幾何関数の一式（`slot_row` / `column_base` / `live_columns` / `in_window` /
 * `effective_rows`）。
 *
 * MUST: 生成の実体は ① と**同じ 4 つの断片**（この関数は uniform 名と行数の欄名を差し替えて
 * 呼ぶだけ）。①ₜ 用に式を書き写すと、ring 写像・窓の下限・live の切り方のどれかが片方だけ
 * 直された時に**例外も NaN も出ない**まま読み書きがずれる。
 * NOTE: `effective_rows` の行数は骨格の `dims.m`（= `rows_block`）。Dims の先頭 3 語は
 * `m` / `n` / `k` 固定なので、states の欄名をそのまま置けない唯一の欄。
 */
export const stateTiledGeometryWgsl = (sliding: boolean): string =>
  `${stateSlotRowWgsl(sliding, STATE_TILED_UNIFORM)}

${stateLiveWgsl(sliding, STATE_TILED_UNIFORM)}

${stateWindowFn(sliding, STATE_TILED_UNIFORM)}

${stateEffectiveRowsWgsl(STATE_TILED_UNIFORM, "m")}`;

/**
 * ③ₜ が使う述語・幾何関数の一式（`slot_row` / `column_base` / `live_columns` /
 * `effective_rows`）。①ₜ の {@link stateTiledGeometryWgsl} から **`in_window` を落とした**もので、
 * 生成の実体は同じ断片。
 *
 * WHY: ③ 系は述語を自分で評価しない — 述語外の列は ① が S へ −inf を書いており、
 * `p = exp(−inf − m)·inv = 0` が厳密に出ることで落ちる（ADR 0067 決定 6）。使わない
 * `in_window` を生成物へ残すと、読者が「③ₜ も窓を切っている」と誤読する。
 */
export const stateTiledPvGeometryWgsl = (sliding: boolean): string =>
  `${stateSlotRowWgsl(sliding, STATE_TILED_UNIFORM)}

${stateLiveWgsl(sliding, STATE_TILED_UNIFORM)}

${stateEffectiveRowsWgsl(STATE_TILED_UNIFORM, "m")}`;

/** ①ₜ / ③ₜ の kv 平面（{@link kvPlaneWgsl} の uniform 差し替え版 — `z` は `wid.z` の別名）。 */
export const stateTiledKvPlaneWgsl = (gqa: boolean): string =>
  kvPlaneWgsl(gqa, STATE_TILED_UNIFORM);

/**
 * ①ₜ が骨格の `Dims`（`{m, n, k}`）へ足す 7 語。先頭 3 語は
 * **`m` = `rows_block` / `n` = `col_cap` / `k` = `depth`** で、行列としての M / N / K そのもの
 * （実効 N は実行時の live なので `col_cap` は**ストライドの静的上界**にしか使わない）。
 *
 * MUST: 並びは {@link stateQkTiledParams} と対（この 2 つが唯一の対 — gemm.ts 側の
 * `ROW_WINDOW_DIMS_EXTRA` / `GQA_DIMS_EXTRA` と同じ規律）。
 * MUST: ① ③ が共有する {@link STATE_PARAMS_STRUCT} とは**別の並び**にしてよい（①ₜ は束縛の
 * 番号だけを ① と揃えた差し替え可能なカーネルで、params は自分専用の 1 本を持つ）。逆に
 * `STATE_PARAMS_STRUCT` の語順は動かさない — ③ と見積りが同じ表を読む。
 * MUST: 変種（sliding / GQA）で欄を出し入れしない。1 つの struct を全変種で使うことで、
 * ホスト側の params 組み立てが 1 本で済む（欄が変種依存だと、片方の変種だけ語位置がずれた
 * params を書いても例外が出ない）。
 */
export const STATE_QK_TILED_DIMS_EXTRA = `  row_offset: u32,
  chunk_rows: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
  neg_inf: u32,
  scale: f32,
`;

/**
 * ③ₜ が骨格の `Dims`（`{m, n, k}`）へ足す 5 語。先頭 3 語は
 * **`m` = `rows_block` / `n` = `depth` / `k` = `col_cap`** で、行列としての M / N / K
 * （実効 K は実行時の live なので `col_cap` は **S の行ストライドと K の静的上界**にしか使わない）。
 *
 * MUST: 並びは {@link statePvTiledParams} と対（この 2 つが唯一の対）。
 * MUST: ①ₜ の {@link STATE_QK_TILED_DIMS_EXTRA} から `neg_inf` / `scale` を**落とす**
 * （③ 系はどちらも読まない — ① が S へ焼いた −inf を `exp` が 0 にするだけ）。読まない語を
 * 残すと「③ₜ も述語や半スケールを持つ」という誤読が params 側にも生える。
 * MUST: 変種（sliding / GQA）で欄を出し入れしない（①ₜ と同文 — ホスト側の params 組み立てが
 * 1 本で済む）。
 */
export const STATE_PV_TILED_DIMS_EXTRA = `  row_offset: u32,
  chunk_rows: u32,
  kv_repeat: u32,
  window: u32,
  capacity: u32,
`;

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

${stateEffectiveRowsWgsl()}

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
 * ①' D 並列縮約変種（ファイル冒頭「①' D 並列縮約変種」節）。束縛と params は ① と**同一**
 * （差し替え可能 — 呼び手はキーと WGSL と workgroup 算出だけを切り替える）。
 *
 * 1 workgroup = 局所行 1 本 × `TILE_X` 本の列 `cl`。レーン `lane` は `d = lane, lane + D_LANES, …`
 * を昇順に部分累積し、`scratch[lane][x]` に置いてから固定順の木で畳んで**レーン 0 が S を書く**。
 *
 * MUST: 書く条件は ① と同一 — `cl < live` なら述語外でも `-inf` を書き、`cl ≥ live` と pad 行は
 * 1 語も書かない。`cl ≥ live` は workgroup 一様でない（端数タイル）ので、そのレーンは
 * **return せず**内積を空回りして barrier に参加する。
 */
export const stateQkParallelWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_qk (states 形の S 実体化, f32, D 並列縮約${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
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

${stateEffectiveRowsWgsl()}

${stateScoreFn("score_slot", "slot_k", STATE_QK_D_LANES)}

${stateScoreFn("score_ins", "ins_k", STATE_QK_D_LANES)}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_QK_D_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_QK_D_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let local_row = wid.y;
  // pad 行の S は誰も読まない（③ が 0 を書いて返す）ので 1 語も書かずに返る。局所行は
  // workgroup 一様なので barrier の手前で返してよい
  if (local_row >= effective_rows(lengths.query)) {
    return;
  }
  let past = lengths.past;
  let live = live_columns(past, lengths.query);
  let cl = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let lane = lid.y;
  let z = wid.z;
  let col = column_base(past) + cl;
  let row = params.row_offset + local_row;
  let q_base = (z * params.chunk_rows + row) * params.depth;
  let kv_plane = ${kvPlaneWgsl(gqa)};
  // 端数タイル（cl ≥ live）と述語外の列は内積を回さない。**return はしない** — この分岐は
  // workgroup 一様でなく、下の barrier は一様制御流の中だけに置けるため（0.0 を寄与する）
  let inside = cl < live && in_window(col, past + row);
  var acc = 0.0;
  if (inside) {
    if (col < past) {
      // past（col < P）はスロットから。物理行は読み書き同式の slot_row
      acc = score_slot(q_base, (kv_plane * params.capacity + slot_row(col)) * params.depth, lane);
    } else {
      // current（col ≥ P）は今 step の ins の行 col − P から
      acc = score_ins(q_base, (kv_plane * params.chunk_rows + (col - past)) * params.depth, lane);
    }
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // 固定順の木縮約（stride 8 → 4 → 2 → 1）— 決定性の根拠
  var stride = ${STATE_QK_D_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  // 述語外は -inf。live 範囲は**述語外でも必ず書く**（書かないと ② が前回の残骸を食う）
  if (lane == 0u && cl < live) {
    var value = bitcast<f32>(params.neg_inf);
    if (inside) {
      value = scratch[lid.x];
    }
    s[(z * params.rows_block + local_row) * params.col_cap + cl] = value;
  }
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
 * MUST: 行 max は `nan_max`（ビット列 NaN 判定 — ADR 0020）。素の `max` は仕様レベルで
 * NaN を落とし、全 NaN 行が空行（stats (0,0)）へ化ける。nan_max なら m = NaN が ③ へ渡り
 * 出力の NaN 分類が保存される。非 NaN 入力ではビット不変（v2）。
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

${stateEffectiveRowsWgsl()}

${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

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
      hi = nan_max(hi, s[base + i]);
      i = i + ${STATE_STATS_WORKGROUP_SIZE}u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${STATE_STATS_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
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

${stateEffectiveRowsWgsl()}

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

/**
 * ③' KV 並列縮約変種（ファイル冒頭「③' KV 並列縮約変種」節）。束縛と params は ③ と**同一**
 * （差し替え可能 — 呼び手はキーと workgroup 算出だけを切り替える）。
 *
 * 1 workgroup = 局所行 1 本 × `TILE_X` 本の `d`。レーン `lane` は `cl = lane, lane + KV_LANES, …`
 * を昇順に部分累積し、`scratch[lane][x]` に置いてから固定順の木で畳む。
 */
export const statePvParallelWgsl = (sliding: boolean, gqa: boolean): string =>
  `// karume attention_state_pv (states 形の O = P @ V, f32, KV 並列縮約${
    sliding ? ", sliding window" : ""
  }${gqa ? ", GQA" : ""})
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

${stateEffectiveRowsWgsl()}

var<workgroup> scratch: array<f32, ${STATE_ATTENTION_TILE_X * STATE_PV_KV_LANES}>;

@compute @workgroup_size(${STATE_ATTENTION_TILE_X}, ${STATE_PV_KV_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let d = wid.x * ${STATE_ATTENTION_TILE_X}u + lid.x;
  let local_row = wid.y;
  let lane = lid.y;
  let z = wid.z;
  let in_depth = d < params.depth;
  let at = (z * params.chunk_rows + params.row_offset + local_row) * params.depth + d;
  // pad 行（row ≥ Q）は live を走査せず厳密 0（③ と同じ契約）。局所行は workgroup 一様なので
  // barrier の手前で返してよい
  if (local_row >= effective_rows(lengths.query)) {
    if (in_depth) {
      out[at] = 0.0;
    }
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
  // レーンごとの部分和（col 昇順・stride KV_LANES）。d が範囲外のレーンは走査せず 0 を寄与する
  // （barrier に参加させるため return しない）
  var acc = 0.0;
  if (in_depth) {
    for (var cl = lane; cl < live; cl = cl + ${STATE_PV_KV_LANES}u) {
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
  }
  scratch[lane * ${STATE_ATTENTION_TILE_X}u + lid.x] = acc;
  workgroupBarrier();
  // 固定順の木縮約（stride 8 → 4 → 2 → 1）— 決定性の根拠
  var stride = ${STATE_PV_KV_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      let mine = lane * ${STATE_ATTENTION_TILE_X}u + lid.x;
      scratch[mine] = scratch[mine] + scratch[(lane + stride) * ${STATE_ATTENTION_TILE_X}u + lid.x];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u && in_depth) {
    out[at] = scratch[lid.x];
  }
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
  // MUST: 有限判定は **f32 として**行う（attention_qk params と同じ門 — f64 で有限な `1e39` は
  // f32 語で `+Inf` になり、`0 * Inf = NaN` でスコアが黙って壊れる）。
  if (!Number.isFinite(Math.fround(geometry.scale))) {
    throw new CodegenError(`${where}: scale は f32 として有限の数値（${geometry.scale}）`);
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
 * ①ₜ の uniform（{@link STATE_QK_TILED_DIMS_EXTRA} と骨格の `{m, n, k}` を合わせた 10 語 —
 * uniform struct の整列で 48 バイト確保する）。
 *
 * MUST: 幾何の値域門は ① と**同じ 1 本**（{@link assertStateGeometry}）。①ₜ は ① と同じ
 * 静的幾何から出るので、門を別に持つと片方だけ通る形ができる。
 * MUST: 並びは {@link STATE_QK_TILED_DIMS_EXTRA} と対（この 2 つが唯一の対）。
 */
export const stateQkTiledParams = (
  geometry: StateAttentionGeometry,
): Uint32Array<ArrayBuffer> => {
  assertStateGeometry("attention_state_qk (tiled) params", geometry);
  const params = new Uint32Array(12);
  params[0] = geometry.rowsBlock;
  params[1] = geometry.colCap;
  params[2] = geometry.depth;
  params[3] = geometry.rowOffset;
  params[4] = geometry.chunkRows;
  params[5] = geometry.kvRepeat;
  params[6] = geometry.window;
  params[7] = geometry.capacity;
  params[8] = STATE_NEG_INF_BITS;
  new Float32Array(params.buffer)[9] = geometry.scale;
  return params;
};

/**
 * ③ₜ の uniform（{@link STATE_PV_TILED_DIMS_EXTRA} と骨格の `{m, n, k}` を合わせた 8 語 —
 * uniform struct の整列で 32 バイト確保する）。
 *
 * MUST: 幾何の値域門は ③ と**同じ 1 本**（{@link assertStateGeometry}）。③ₜ は ③ と同じ
 * 静的幾何から出るので、門を別に持つと片方だけ通る形ができる。
 * MUST: 並びは {@link STATE_PV_TILED_DIMS_EXTRA} と対（この 2 つが唯一の対）。
 */
export const statePvTiledParams = (
  geometry: StateAttentionGeometry,
): Uint32Array<ArrayBuffer> => {
  assertStateGeometry("attention_state_pv (tiled) params", geometry);
  const params = new Uint32Array(8);
  params[0] = geometry.rowsBlock;
  params[1] = geometry.depth;
  params[2] = geometry.colCap;
  params[3] = geometry.rowOffset;
  params[4] = geometry.chunkRows;
  params[5] = geometry.kvRepeat;
  params[6] = geometry.window;
  params[7] = geometry.capacity;
  return params;
};

/**
 * ② の uniform（`{batch_heads, rows_block, row_offset, col_cap, window, neg_inf}` の 6 語 —
 * uniform struct の整列で 32 バイト確保する）。
 *
 * MUST: `B·H` と `rows_block` を**畳まずに**受ける（① ③ と同じ 2 軸）。畳んだ 1 語では
 * `z · rows_block + 局所行` へ戻せず、有効行の前詰めから S / stats の行を引けない。
 * MUST: `window` も値域門に通す（{@link stateSliding} が要求する「キー選択・WGSL 生成・
 * params 検査の 3 者を揃える」の ② 側）。② は `capacity` / `M` を受けないので上限式
 * `W−1 + M`（{@link assertStateGeometry}）そのものは書けないが、`M ≥ 1` から従う必要条件
 * `W ≤ col_cap` は書ける。これが破れると `live_columns` が `col_cap` を跨いで隣接行の S を
 * 自分の行の amax に混ぜる（例外も NaN も出ない沈黙誤値）。
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
  if (stateSliding(window) && colCap < window) {
    throw new CodegenError(
      `attention_state_stats params: col_cap ${colCap} が sliding の live 上限 ${window}` +
        `（= W−1 + M の最小 M = 1）に足りない`,
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
 * ①' の workgroup 数 `[⌈live / TILE_X⌉, 有効行, B·H]`（行軸は **1 行 = 1 workgroup**）。
 *
 * ① と同じく列軸は **live 列数**・行軸は**有効行数**（`colCap` / `rowsBlock` にすると仕事量
 * 合格条件を落とし、pad 行の S を書いてしまう）。上限超過は fail loudly も ① と同じ。
 * 変わるのは行のタイル幅だけで、D 方向は workgroup 内のレーンに畳まれるので dispatch に出ない。
 */
export const stateQkParallelWorkgroups = (
  geometry: StateDispatchGeometry,
  past: number,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①'QK`, geometry);
  assertLengths(`${where} ①'QK`, past, query);
  const live = stateLiveColumns(geometry.window, past, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [
    tiledWorkgroups(live, STATE_ATTENTION_TILE_X, limit, `${where} ①'QK`),
    tiledWorkgroups(rows, 1, limit, `${where} ①'QK`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①'QK`),
  ];
};

/**
 * ①ₜ の workgroup 数 `[⌈live / tileN⌉, ⌈有効行 / tileM⌉, B·H]`（1 workgroup = 1 出力タイル）。
 *
 * MUST: タイル辺は**幾何から導く**（gemm 骨格の MUST と同文 — 定数で持ち回ると幾何と食い違い
 * うる値が 2 つになり、`ceil(dim / 定数)` が実タイル辺での本数を下回った瞬間に**タイルが欠落
 * して沈黙誤値**になる）。幾何を解決するのはキー・生成・ここの 3 者で、全部 `chunkRows` 1 値の
 * 純関数（{@link gemmGeometryForRows}）を通る。
 * MUST: ① と同じく列軸は **live 列数**・行軸は**有効行数**（`colCap` / `rowsBlock` にすると
 * 仕事量合格条件〈ADR 0066 決定 3〉を落とし、pad 行の S を書いてしまう）。
 * MUST: 上限超過は fail loudly（タイル系 — 縮退させると S のタイルが欠落し、②③ が残骸を
 * 読んだまま例外なしに進む）。
 */
export const stateQkTiledWorkgroups = (
  geometry: StateDispatchGeometry,
  chunkRows: number,
  past: number,
  query: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ①ₜQK`, geometry);
  assertLengths(`${where} ①ₜQK`, past, query);
  const tile = gemmGeometryForRows(chunkRows);
  const live = stateLiveColumns(geometry.window, past, query);
  const rows = stateEffectiveRows(geometry.rowsBlock, geometry.rowOffset, query);
  return [
    tiledWorkgroups(live, gemmTileN(tile), limit, `${where} ①ₜQK`),
    tiledWorkgroups(rows, gemmTileM(tile), limit, `${where} ①ₜQK`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ①ₜQK`),
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

/**
 * ③' の workgroup 数 `[D / TILE_X, rows_block, B·H]`（行軸は **1 行 = 1 workgroup**）。
 *
 * ③ と同じく行軸は `rows_block` 全て（pad 行も full-write）・上限超過は fail loudly。
 */
export const statePvParallelWorkgroups = (
  geometry: StateDispatchGeometry,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ③'PV`, geometry);
  return [
    tiledWorkgroups(geometry.depth, STATE_ATTENTION_TILE_X, limit, `${where} ③'PV`),
    tiledWorkgroups(geometry.rowsBlock, 1, limit, `${where} ③'PV`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ③'PV`),
  ];
};

/**
 * ③ₜ の workgroup 数 `[⌈D / tileN⌉, ⌈rows_block / tileM⌉, B·H]`（1 workgroup = 1 出力タイル）。
 *
 * MUST: タイル辺は**幾何から導く**（①ₜ の {@link stateQkTiledWorkgroups} と同文 — 定数で
 * 持ち回ると幾何と食い違いうる値が 2 つになり、`ceil(dim / 定数)` が実タイル辺での本数を
 * 下回った瞬間に**タイルが欠落して沈黙誤値**になる）。幾何を解決するのはキー・生成・ここの
 * 3 者で、全部 `chunkRows` 1 値の純関数（{@link gemmGeometryForRows}）を通る。
 * MUST: 行軸は **`rows_block` 全て**（① 系と違って有効行で切らない — pad 行も書くのが ③ の
 * full-write 不変条件）。仕事量が Q に比例するのはカーネル側で、有効行を 1 行も含まない
 * 行タイルは K ループを 1 周も回さずに `0.0` を書いて終わる。
 * MUST: 論理長を受け取らない（③ / ③' と同じく出力側の形だけで決まる）。
 * MUST: 上限超過は fail loudly（タイル系 — 縮退させると O の一部が未書き込みのまま残り、
 * full-write 不変条件が黙って崩れる）。
 */
export const statePvTiledWorkgroups = (
  geometry: StateDispatchGeometry,
  chunkRows: number,
  limit: number,
  where: string,
): [number, number, number] => {
  assertDispatchGeometry(`${where} ③ₜPV`, geometry);
  const tile = gemmGeometryForRows(chunkRows);
  return [
    tiledWorkgroups(geometry.depth, gemmTileN(tile), limit, `${where} ③ₜPV`),
    tiledWorkgroups(geometry.rowsBlock, gemmTileM(tile), limit, `${where} ③ₜPV`),
    tiledWorkgroups(geometry.batchHeads, 1, limit, `${where} ③ₜPV`),
  ];
};

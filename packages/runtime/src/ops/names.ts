/** op 契約に反する IR（未対応 op・アリティ違反・契約外 attrs・dtype 違反・shape 不整合）。 */
export class OpContractError extends Error {
  override readonly name = "OpContractError";
}

/**
 * 単項 elementwise。
 *
 * MUST: kind が意味するのは「入力 1 本の elementwise」だけ。attrs を持つ 6 本
 * （clamp / clamp_min / leaky_relu / ge_scalar / le_scalar / gt_scalar）も、出力 dtype が
 * 入力と違う比較 3 本もここに属する — attrs スキーマと出力 dtype は契約表の別の欄が持つので、
 * その差のために kind を割らない（割ると codegen と CPU 参照の分岐が同じ理由で増える）。
 */
export const UNARY_OPS = [
  "neg",
  "abs",
  "exp",
  "log",
  // WGSL に log1p 組込は無い（実装方針は src/codegen/elementwise.ts の LOG1P_FN）
  "log1p",
  "sqrt",
  // 三角関数は sin だけ（ADR 0043 の第 1 層）。定数の RoPE 表はエクスポータが畳むので、
  // 語彙に要るのは**実行時値**を取る形だけ — DACVAE の Snake 活性 `x + (α+1e-9)⁻¹·sin²(αx)` が
  // 初出。`cos` は実測に出るまで足さない（「対称性のための追加をしない」）。
  "sin",
  "tanh",
  "sigmoid",
  "relu",
  "gelu",
  // torch の `approximate="tanh"` 形。erf 形の `gelu` とは値が違うので別 op で表す —
  // attrs 空の契約に近似種別を載せる欄は無く、載せると「同じ op 名で数値が変わる」分岐が
  // 契約の外にできる。
  "gelu_tanh",
  "bitwise_not",
  // attrs のスカラを params 末尾で運ぶ 6 本（{@link SCALAR_PARAM_ATTRS}）
  "clamp",
  // 片側 clamp（ADR 0017）。既存 clamp の attrs を optional 化して兼ねる案は「宣言済み
  // attrs の既定値補完はしない」（ADR 0012）を崩すので採らない — 別 op で表す。
  "clamp_min",
  "leaky_relu",
  "ge_scalar",
  "le_scalar",
  "gt_scalar",
] as const;
export type UnaryOpName = (typeof UNARY_OPS)[number];

/**
 * 二項 elementwise（torch 準拠の右詰め broadcast）。`ge` は f32 × f32 → **bool** で、
 * 出力 dtype だけが入力と違う（導出は {@link OUTPUT_DTYPES}）。
 */
export const BINARY_OPS = ["add", "sub", "mul", "div", "ge", "bitwise_and"] as const;
export type BinaryOpName = (typeof BINARY_OPS)[number];

/**
 * 1 軸の reduce（attrs `dim`・keepdim 無し）。最終次元は行カーネル、それ以外は軸変種で
 * 実行する（設計は docs/research/2026-08-04-vae-axis-reduce-recon.md §5）。
 */
export const REDUCE_OPS = ["sum", "amax", "amin"] as const;
export type ReduceOpName = (typeof REDUCE_OPS)[number];

/**
 * 最終次元の argmax（ADR 0068 決定 2）。入力 f32 → 出力 **i32** で、**rank 保存**
 * （最終次元を 1 に潰す固定形）。
 *
 * MUST: `attrs` を持たない。軸は最終次元固定（`dim` の欄が無いことがそのまま「他の軸は
 * 語彙に無い」の宣言 — gather と同じ絞り方）で、`keepdim` の欄も作らない（rank 保存の
 * 1 形だけを表す）。needs が出たら別 op で足す規律は `gelu` / `gelu_tanh` と同じ。
 * MUST: reduce 族（{@link REDUCE_OPS}）に入れない。attrs も出力 dtype も rank の扱いも
 * 違い、カーネルも別族（identity が −inf・(値, index) 対を運ぶ — src/kernels/argmax.ts）。
 * MUST: タイブレークは**最小 index**・NaN は**最大**・全 −inf 行は **index 0**（いずれも
 * torch 準拠。実装契約の正本は src/kernels/argmax.ts と reference/ops.ts）。
 */
export const ARGMAX_OP = "argmax";

/**
 * 最終次元の top-k（ADR 0068 決定 3 — **ノード多出力の最初の入居者**）。入力 f32 →
 * **出力 2 本**（slot 0 = 値 f32 の**降順**・slot 1 = 添字 i32）で、2 本とも `[…, k]`。
 *
 * MUST: `k` は **attrs で宣言必須**（計画時定数 = static-k）。記号 k・k=0・最終次元超過は
 * 全て fail loudly（受理領域は `1 ≤ k ≤ 最終次元`）。torch の schema は `k` が SymInt だが、
 * 実行時に決まる k は静的形状の前提（ADR 0004）に載らないので受理しない。
 * MUST: 軸は最終次元固定（`dim` の欄が無いことがそのまま「他の軸は語彙に無い」の宣言）で、
 * `largest` / `sorted` の欄も作らない（**降順ソート済みの最大側だけ**が語彙 — 最小側が要れば
 * 別 op、順序無しは要求が出てから）。
 * MUST: タイブレークは**最小 index**（argmax と同族の述語）・NaN は**最大**・全 −inf 行も
 * 最小 index から k 本。**値の列は torch とビット一致**する一方、**添字の列は torch の
 * 未規定部分を karume が規定した**側（torch は同値要素の順序を保証せず、`argmax` と
 * `topk(k=1)` が同一リポ内で食い違う — 実測は src/kernels/topk.ts の NOTE）。
 */
export const TOPK_OP = "topk";

/**
 * 三項 elementwise `out = cond ? a : b`（torch の `where`）。
 *
 * MUST: 3 者とも右詰め broadcast（torch と同じ）。条件スロットが先頭なのは torch の
 * 引数順（`where(cond, a, b)`）に合わせるためで、**出力は条件ではなく値の側と同型**
 * （導出は {@link OUTPUT_DTYPES} の bool → f32）。
 */
export const WHERE_OP = "where";

/**
 * 最終次元の前縁和（`out[…, j] = Σ_{i ≤ j} x[…, i]`、attrs `dim`）。
 *
 * MUST: 軸は**最終次元のみ**受理する（softmax と同じ絞り方 — 行カーネルは縮約軸が連続で
 * あることを前提にしている）。attrs に `dim` を持つのも softmax に倣う: 負の軸表記は
 * エクスポータ境界で正規化され、ランタイムは rank が分かる shape 計算側で最終次元との
 * 一致を再検査する（宣言と実 rank の食い違いをもう 1 枚の門で止める）。
 */
export const CUMSUM_OP = "cumsum";

export const MATMUL_OP = "matmul";
/**
 * バッチ matmul（ADR 0012）。`[B,M,K] × [B,K,N] → [B,M,N]` の **rank-3 のみ**で、rank-2 は
 * {@link MATMUL_OP} の担当。実測（recon §2 の 8 本）は全て rank-3 で、バッチ軸は完全一致
 * （broadcast も stride 0 も出ない）。
 *
 * MUST: matmul と兼用しない。1 つの op が rank で別のカーネルへ分岐すると、契約の
 * 「入力 shape → 出力 shape」が rank に依存する多相になり、shape 検査の抜けが見えなくなる。
 */
export const BMM_OP = "bmm";
/**
 * 最終次元の gather（ADR 0012）。`out[..., j] = src[..., index[..., j]]`。
 *
 * MUST: 軸は**最終次元固定**で attrs を持たない（実測は dim = −1 のみ — softmax と同じ絞り方。
 * 一般 dim は要求実測が出てから広げる）。
 * MUST: **最初**のスロット別 dtype 契約（src=f32 / index=i32 → out=f32）。同型は embedding /
 * masked_fill / where で、一覧の正本は契約表（ops/contracts.ts の `SlotDtypes` doc と
 * `kind: "perSlot"` の実値）— 本数はここに数え上げない（op 追加のたびに動く導出可能な事実で、
 * op 名定数の側に置くと必ず腐る）。
 */
export const GATHER_OP = "gather";
/** 意味論 dtype 変換（attrs `to` が変換先 — ADR 0009）。 */
export const CAST_OP = "cast";

/**
 * レイアウト op（ADR 0011）。
 *
 * - `reshape` — 要素順を変えない（view / squeeze / unsqueeze の正規化先）。**出力の宣言
 *   shape が目標形**で、契約は要素数一致のみ。実行はバッファ別名（コピー無し）。
 * - `permute` — attrs `dims` の軸並べ替え。実体化コピー。
 * - `expand` — **出力の宣言 shape へ右詰め broadcast**。拡張できるのは長さ 1 の次元だけ
 *   （長さ n → m の複製は語彙に無い）。実体化コピー。
 */
export const RESHAPE_OP = "reshape";
export const PERMUTE_OP = "permute";
export const EXPAND_OP = "expand";

/**
 * レイアウト op 第 2 群（ADR 0014）。full-write 不変条件（全カーネルが出力の全バイトを書く）
 * を満たす形だけを語彙に入れる。
 *
 * - `slice` — attrs `dim` / `start` / `end` の**静的軸・静的範囲**の切り出し。実行は既存の
 *   strided **読み**コピー族の流用で、可変点は params の offset 1 語（ADR 0011 の予告どおり）。
 *   記号次元の切り出しは {@link SYM_PREFIX_SLICE_OP} の担当で、こちらは静的専業
 *   （重複させない — 記号軸の slice は plan.ts が宣言 shape を見て落とす）。
 * - `cat` — attrs `dim` の**静的軸**連結。**入力数が可変**（{@link OpContract} の `variadic`）。
 *   実行は strided **書き**コピー族で、入力ごとに出力の部分領域へ書く。単独の dispatch は
 *   部分書きでも、全入力で出力全域を覆うのでノード単位では full-write が成立する。
 * - `pad` — attrs `left` / `right` の**最終次元・定数 0** 埋め。専用カーネル 1 本が出力全域を
 *   書く（ゼロ初期化保証にも noReuse 特例にも依存しない）。
 * - `flip` — attrs `dim` の静的軸の添字反転。専用の極小カーネル。
 */
export const SLICE_OP = "slice";
export const CAT_OP = "cat";
export const PAD_OP = "pad";
export const FLIP_OP = "flip";

/**
 * 記号 prefix スライス（ADR 0010）。エクスポータが記号 T 依存の部分木を **Tmax で実評価**して
 * 焼いた定数から、束縛後の `coeff·sym+offset` 長の**先頭**を切り出す。
 *
 * - 入力は**記号を含まない静的 shape**の値（実際にはエクスポータが焼いた initializer）。
 *   この不変条件は plan.ts の `validateGraphContracts` が見る — 入力側が記号 shape だと
 *   「Tmax 形」という前提が崩れ、読み出し stride が実行ごとに変わる。
 * - 出力 shape は attrs（+ 束縛）から計算し、宣言と照合する（reshape / expand のような
 *   「宣言が目標形」ではない）。
 * - 実行は strided 実体化コピー（offset 0・入力側の連続 stride）— 新カーネルは無い。
 */
export const SYM_PREFIX_SLICE_OP = "sym_prefix_slice";

/**
 * 融合 op（ADR 0012 / ADR 0015 / ADR 0017）。ADR 0007 起点の「分解禁止」リスト（**現行 12 op** —
 * 台帳の正本は docs/op-vocabulary.md。`leaky_relu` で 10・`rms_norm` で 11・
 * `scaled_dot_product_attention` で 12）は全てカーネルを持つ（`conv2d` は Anima の VAE decoder で
 * 実測に出た — ADR 0017）。`rms_norm` は**保存だけでは供給しきれない** 1 本目で、手書き分解形
 * （Qwen3 / DiT）はエクスポータの畳み込みパスが 1 ノードへ合成する（供給ルート 2 系統 —
 * ADR 0016 / 0017）。12 本目の attention だけはエクスポータの**既定の保存リストに載らない**
 * （ターゲット別の opt-in — 理由は op-vocabulary.md）。
 *
 * MUST: 受理制約は IR の語彙ではなく**この契約表 + 明確な診断**で表す（ADR 0007）。
 * 「実測に出た形しか通さない」を IR スキーマ側に彫ると、形を広げるたびに配布形の非互換
 * 改訂になる。
 *
 * - `linear` — `x[…,in] × W[out,in] + b[out]`。**bias は常時あり（arity 3 固定）**で、
 *   実測 16 本すべてが bias 付き（recon §5）。bias 無し形は実測が出るまで fail loudly。
 * - `layer_norm` — attrs `normalized_shape` / `eps`。affine（weight / bias）も常時あり。
 * - `softmax` — attrs `dim`。**最終次元のみ**受理（実測は −1 のみ — gather と同じ絞り方）。
 * - `safe_softmax` — `softmax` と同一契約 + 「行 max が −inf の行は全 0」（ADR 0044）。
 * - `embedding` — `weight f32[V,H]` を `index i32[…]` で行 gather。attrs `padding_idx`。
 * - `masked_fill` — attrs `value`（f32 スカラ）。mask は bool で**右詰め broadcast**。
 * - `conv1d` — attrs `stride` / `padding` / `dilation` / `groups`（ADR 0015 で後 2 者を追加）。
 *   「欄が無い = 1 固定」の設計価値は「欄がある = 宣言必須・既定値補完なし・不整合は
 *   fail loudly」で引き継ぐ。重みは `[Cout, Cin/groups, K]`。
 * - `conv_transpose1d` — attrs `stride` / `padding`。重みは **`[Cin, Cout, K]`**（conv1d と
 *   転置）。受理するのは出力長が `L·stride` になる形（`2·padding == K − stride`）だけで、
 *   一般形は fail loudly（ADR 0015 — 需要が出た時に広げる）。
 * - `conv2d` — attrs `stride` / `padding` / `dilation` / `groups`（空間 3 つは **H/W の
 *   2 成分**）。`x[B,Cin,H,W] * W[Cout,Cin/groups,Kh,Kw] + b[Cout]`（ADR 0017）。
 * - `rms_norm` — attrs `eps`・**アリティ 2**（x, weight）。`y = x · rsqrt(mean(x², 最終次元)
 *   + eps) · weight`。layer_norm と違い bias が無く、平均を引かない（ADR 0017）。
 */
export const LINEAR_OP = "linear";
export const LAYER_NORM_OP = "layer_norm";
/**
 * 融合 attention（ADR 0023）。`out = softmax_lastdim((q·scale) @ (k·scale)ᵀ + mask) @ v`。
 *
 * - **アリティ 3 か 4**（q / k / v + 省略可能な mask）。可変アリティ（`cat`）とは別の機構で、
 *   上限を持つ（{@link ContractBase.maxArity}）— 「何本でも」ではなく「mask 1 本だけ増える」。
 * - 入力は **rank-4 head-first**（`q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]`）で連続。
 *   出力は `[B,H,M,D]`。**D は 3 者とも同じ**（実測の全 attention がそうで、v 側だけ
 *   別の長さを許すと「D を取り違えた IR」が shape 検査を素通りする）。
 * - **GQA / MQA は整除 broadcast**（ADR 0067 決定 1）。`H % Hkv == 0` を満たす形だけを受理し、
 *   `r = H / Hkv` は**導出値**（attrs 欄を作らない）。B の完全一致・k/v 間の Hkv 一致・
 *   D 3 者同一・N=0 拒否は取り違えの検出線としてそのまま残る。
 * - **mask は f32・rank-4・shape はちょうど `[1,1,M,N]`**（加算型 — `S' = S + mask`）で、
 *   B·H の全バッチへ broadcast する。`[B,1,M,N]` / `[1,H,M,N]` / bool / rank≠4 は
 *   **受理しない**（実行時マスクの需要が出た時に広げる — 欄の不存在が「語彙に無い」を構造で
 *   表す規律）。causal / dropout は依然として語彙に無い。
 *
 * MUST: `scale` は **q と k の両方に掛かる（半スケール契約）**。torch の
 * `aten::_scaled_dot_product_attention_math` が `q *= √scale_factor; k *= √scale_factor` と
 * 書く形と同義で、`√` を含む見慣れない値になるのはそのため。**内積の後に 1 度だけ掛ける形
 * （全スケール）に変えてはならない** — 丸め列が変わり、分解経路（mul → permute → bmm →
 * softmax → expand → bmm）とのビット同一が失われる（ADR 0023 の設計の核）。
 */
export const ATTENTION_OP = "attention";
export const RMS_NORM_OP = "rms_norm";
export const SOFTMAX_OP = "softmax";
/**
 * `softmax` + 「**行 max が −inf の行は全 0 を書く**」（ADR 0044）。契約・attrs・shape 規則は
 * `softmax` と同一で、違いはこの 1 行だけ。
 *
 * 出所は torch の SDPA 分解が `softmax` に被せる safe-softmax ガード
 * （`where(¬any(¬eq(src,−inf)), 0, softmax(src))`）で、マスクが実行時値だと「ガードが
 * 発火しない」証明が原理的に立たない（エクスポータの `_drop_safe_softmax_guard`）。
 * スコアが有限なら「全要素 −inf ⇔ 行 max = −inf」なので、この op はガードと**厳密に**
 * 同じ意味論を持つ。
 *
 * MUST: `softmax` と別 op のまま保つ。素の `softmax` に空行の分岐を足すと、全 −inf 行を
 * 契約違反として扱う既存の絞り（ADR 0044 決定 3 — 融合 attention も同じ絞り）が消え、
 * 「NaN で表に出るはずの契約違反」が黙って 0 になる。
 */
export const SAFE_SOFTMAX_OP = "safe_softmax";
export const EMBEDDING_OP = "embedding";
export const MASKED_FILL_OP = "masked_fill";
export const CONV1D_OP = "conv1d";
export const CONV2D_OP = "conv2d";
export const CONV_TRANSPOSE1D_OP = "conv_transpose1d";

/**
 * modulated deformable convolution v2（第 1' 層の原子 — `torchvision::deform_conv2d`。
 * ADR 0055）。**アリティ 5 固定**で `x[B,Cin,H,W]` / `weight[Cout,Cin,Kh,Kw]` /
 * `offset[B,2·Kh·Kw,Hout,Wout]` / `mask[B,Kh·Kw,Hout,Wout]` / `bias[Cout]` を取る。
 *
 * - `offset` のチャネル並びは `(kh, kw)` の入れ子で最内が 2 連、**偶数 = y / 奇数 = x**。
 * - `mask`（modulator）は**双線形補間の後**に掛かる（`(m · v) · w`）。
 * - 境界外は**ゼロ埋め**（中心が `(−1, in)` の外ならタップ全体 0・内側でも範囲外の隅は
 *   その隅だけ 0 — border clamp ではない）。
 *
 * MUST: `stride` / `dilation` / `groups` / `offset_groups` の**欄を作らない**（= 1 固定）。
 * 実測（BiRefNet 一族の 20 箇所）が全て 1 で、欄の不存在がそのまま「その形は語彙に無い」の
 * 宣言になる（conv_transpose1d の `output_padding` と同じ手筋 — ADR 0023 決定 4）。
 * MUST: `mask` はスロットとして必須 = **DCNv2 専業**。`use_mask=False`（DCNv1）を表す欄が
 * 無いことがそのまま fail loudly になる。
 * MUST: 出力空間は x / weight / `padding` から導き、`offset` / `mask` の空間 2 軸とは
 * **突き合わせるだけ**（同じ事実を 2 か所から取ると形の食い違いが素通りする）。
 */
export const DEFORM_CONV2D_OP = "deform_conv2d";

/**
 * NCHW の空間 2 軸の双線形 resample（第 1 層の原子 — `aten.upsample_bilinear2d.vec`）。
 * `x[B,Cin,H,W] → [B,Cin,Hout,Wout]`、attrs は `output_size`（`[Hout, Wout]`）のみ。
 *
 * MUST: **`align_corners = True` 専業**で、attrs に `align_corners` の欄を作らない
 * （欄の不存在が「語彙に無い」を構造で表す規律 — ADR 0023 決定 4）。`False` は座標式
 * （`scale·(i+0.5) − 0.5`）も端の扱いも別物なので、受理すると同じ op 名で数値が変わる。
 * 需要が出たら `gelu` / `gelu_tanh` と同じ手筋で**別 op**として足す。
 * MUST: `mode` の欄も作らない（nearest / bicubic / area / antialias は語彙に無い）。倍率は
 * `output_size` と入力から**実行時に導く** — `scale_factor` を載せる欄も持たない（同じ形に
 * 2 通りの IR ができる）。
 * NOTE: 縮小（`Hout < H`）も同じ op・同じ式で通る（torch も同一 op）。antialias を持たない
 * ぶん 2 タップしか読まないのは torch の仕様どおりで、`area` とは別物。
 */
export const UPSAMPLE_BILINEAR2D_OP = "upsample_bilinear2d";

/**
 * GRU の**隠れ側スキャン**（第 2 層 — ADR 0056）。時間方向の逐次だけを 1 ノードに畳み、
 * **入力側 GEMM は持たない**（呼び手が既存 `linear` で `gi = x·W_ihᵀ + b_ih` を用意する）。
 *
 * **アリティ 4 固定**で `gi[T,N,3H]` / `h0[N,H]` / `w_hh[3H,H]` / `b_hh[3H]` を取り、
 * 出力は `y[T,N,H]`（全ステップの `h`）。3H のゲート並びは **r / z / n** の順。
 *
 * ```
 * r = sigmoid(gh_r + gi_r) / z = sigmoid(gh_z + gi_z) / n = tanh(gi_n + gh_n·r)
 * h' = (h − n)·z + n            （gh = W_hh·h + b_hh）
 * ```
 *
 * MUST: 走査方向は **op 名で分ける**（`gru_scan` / `gru_scan_reverse`）。attrs の bool 変種に
 * しない — attrs に bool を載せる前例が無く（Python の bool は int の派生で検証機構の新設が
 * 要る）、`gelu` / `gelu_tanh` と同じ「attr 変種は別 op」の手筋に揃える（ADR 0056 決定 2）。
 * MUST: 逆方向 op も**出力は順方向の時間順**で書く。`flip` は記号軸を拒否する（ADR 0014 /
 * 0046）ので、走査方向を op の中へ畳むことが記号 T を通す唯一の形。
 * MUST: 多層 / 双方向の欄を作らない。IR v1 の可変アリティは `cat` だけで、`aten.gru` の
 * `Tensor[16]` は構造的に載らない — 層と方向は**ノードを並べて**表す（ADR 0056 決定 7）。
 * MUST: `h_n` を返さない（IR v1 は実質単一出力 — 出力は `y` だけ）。
 * MUST: `has_biases=False` / `batch_first` / `dropout` の欄を作らない。欄の不存在がそのまま
 * 「その形は語彙に無い」の宣言になる（ADR 0023 決定 4）。
 */
export const GRU_SCAN_OPS = ["gru_scan", "gru_scan_reverse"] as const;
export type GruScanOpName = (typeof GRU_SCAN_OPS)[number];

/**
 * 低精度格納が**適格**になる重みスロット（op 名 → 入力スロット番号 — ADR 0018）。
 *
 * 実測でサイズが支配的な 5 スロットだけを載せる。ここに無い消費（bias / norm 系の weight /
 * その他の op）が 1 つでもあれば、その initializer は適格外としてロード時に CPU で f32 展開
 * される。
 *
 * MUST: bias を含めない。プロトタイプは bias の f32 定数が weight を道連れに降格させて
 * f16 の適格を 0MB にした（ADR 0006 が名指しした根治対象）— 「bias は常に f32」は
 * 「bias スロットを適格判定に載せない」ことでしか担保できない。
 */
export const WEIGHT_SLOTS: ReadonlyMap<string, number> = new Map([
  [LINEAR_OP, 1],
  [CONV1D_OP, 1],
  [CONV2D_OP, 1],
  [CONV_TRANSPOSE1D_OP, 1],
  [EMBEDDING_OP, 0],
]);

/**
 * per-channel scale の**チャネル軸**（op 名 → 重みテンソルの軸番号 — ADR 0019）。
 *
 * 出力チャネルの軸で、linear `[out,in]` / conv1d `[Cout,Cin/g,K]` / conv2d `[Cout,Cin/g,Kh,Kw]` /
 * embedding `[V,H]` は 0、**conv_transpose1d だけ `[Cin,Cout,K]` の転置レイアウトで 1**。
 *
 * MUST: キー集合は {@link WEIGHT_SLOTS} と一致させる（tests/ops_contract_test.ts が固定）。
 * 片方だけ増えると、新しい重みスロットの i8 が「軸 0 の scale」として黙って実行される。
 */
export const WEIGHT_CHANNEL_AXES: ReadonlyMap<string, number> = new Map([
  [LINEAR_OP, 0],
  [CONV1D_OP, 0],
  [CONV2D_OP, 0],
  [CONV_TRANSPOSE1D_OP, 1],
  [EMBEDDING_OP, 0],
]);

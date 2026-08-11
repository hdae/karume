/**
 * op 契約テーブル — op ごとに「アリティ / dtype 規則 / attrs スキーマ / 出力 shape 計算」を
 * 1 箇所に持つ。codegen・CPU 参照・executor は全てここを参照する。
 *
 * MUST: ランタイムの capability 宣言（{@link RUNTIME_SUPPORT}）はこの表からのみ導出する。
 * op 名だけの突合は dtype と属性の差を見逃す（対応表に名前だけある op が実行時に落ちる）。
 * MUST: 意味論 dtype の解禁は op ごとに行う（一括解禁しない — ADR 0007 の語彙 allowlist
 * 凍結）。i32 / bool を許すのは実測グラフに現れた形と、その形のために新設した op だけ。
 *
 * 出力 shape の計算は**束縛解決後の数値 shape**に対して行う。記号次元は実行前に必ず具体値へ
 * 解決される（静的形状 — ADR 0004）ため、broadcast 可否のような判定を記号のまま扱う必要が無い。
 */

import { STRIDED_RANK } from "./codegen/strided.ts";
import type { OpSupport, RuntimeSupport } from "./format/container.ts";
import { isSymbolName } from "./format/dims.ts";
import { type IrDtype, type IrNode, isSemanticDtype, SEMANTIC_DTYPES } from "./format/ir.ts";

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

const MATMUL_OP = "matmul";
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
 * MUST: 入力スロットで dtype が違う唯一の op（src=f32 / index=i32 → out=f32）。
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
 * 融合 op（ADR 0012 / ADR 0015 / ADR 0017）。ADR 0007 の「分解禁止 10 op」は全て
 * カーネルを持つ（`conv2d` は Anima の VAE decoder で実測に出た — ADR 0017）。
 * `rms_norm` は保存リストの外から入った 1 本目（Qwen3 / DiT は手書き分解形なので
 * エクスポータの畳み込みパスが作る — ADR 0016）。
 *
 * MUST: 受理制約は IR の語彙ではなく**この契約表 + 明確な診断**で表す（ADR 0007）。
 * 「実測に出た形しか通さない」を IR スキーマ側に彫ると、形を広げるたびに配布形の非互換
 * 改訂になる。
 *
 * - `linear` — `x[…,in] × W[out,in] + b[out]`。**bias は常時あり（arity 3 固定）**で、
 *   実測 16 本すべてが bias 付き（recon §5）。bias 無し形は実測が出るまで fail loudly。
 * - `layer_norm` — attrs `normalized_shape` / `eps`。affine（weight / bias）も常時あり。
 * - `softmax` — attrs `dim`。**最終次元のみ**受理（実測は −1 のみ — gather と同じ絞り方）。
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
 * - 入力は **rank-4 head-first**（`q[B,H,M,D]` / `k[B,H,N,D]` / `v[B,H,N,D]`）で連続。
 *   出力は `[B,H,M,D]`。**D は 3 者とも同じ**（実測の全 attention がそうで、v 側だけ
 *   別の長さを許すと「D を取り違えた IR」が shape 検査を素通りする）。
 * - **mask は f32・rank-4・shape はちょうど `[1,1,M,N]`**（加算型 — `S' = S + mask`）で、
 *   B·H の全バッチへ broadcast する。`[B,1,M,N]` / `[1,H,M,N]` / bool / rank≠4 は
 *   **受理しない**（実行時マスクの需要が出た時に広げる — 欄の不存在が「語彙に無い」を構造で
 *   表す規律）。causal / dropout / GQA は依然として語彙に無い。
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
export const EMBEDDING_OP = "embedding";
export const MASKED_FILL_OP = "masked_fill";
export const CONV1D_OP = "conv1d";
export const CONV2D_OP = "conv2d";
export const CONV_TRANSPOSE1D_OP = "conv_transpose1d";

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

export type OpKind =
  | "unary"
  | "binary"
  | "where"
  | "cumsum"
  | "matmul"
  | "bmm"
  | "gather"
  | "rowReduce"
  | "cast"
  | "reshape"
  | "permute"
  | "expand"
  | "slice"
  | "cat"
  | "pad"
  | "flip"
  | "symPrefixSlice"
  | "linear"
  | "layerNorm"
  | "rmsNorm"
  | "softmax"
  | "attention"
  | "embedding"
  | "maskedFill"
  | "conv1d"
  | "conv2d"
  | "convTranspose1d";

/**
 * attrs スキーマ = attr キー → 値の検査（ADR 0012）。
 *
 * MUST: 宣言したキーは**全て必須**、宣言外のキーは fail loudly。省略可能な attr を持つ op
 * （融合 op の bias 有無など）が出た時点で表現を広げる — 先回りして機構だけ増やさない。
 */
export type AttrSchema = Readonly<Record<string, (value: unknown, where: string) => void>>;

/**
 * 入力スロットの dtype 契約（ADR 0012 の拡張）。
 *
 * - `uniform` — 全スロットが同じ受理集合を持ち、かつ**互いに同型**であることまで要求する
 *   （出力も同型）。elementwise / matmul / bmm / 行 reduce / レイアウトはこれ。
 * - `perSlot` — スロットごとに受理集合が違い、スロット間の同型は要求しない。出力は
 *   **スロット 0（値の側）と同型**。gather（src=f32 / index=i32 → out=f32）が最初の例。
 *
 * MUST: 受理集合が同じ op を `perSlot` で書かない。`uniform` は `mul(f32, i32)` のような
 * 混合を拒否する規則も担っており、perSlot に潰すとその拒否が黙って消える。
 * MUST: 出力 dtype は**スロット 0 の dtype を写像に通して**導く（{@link ContractBase.outputDtypes}）。
 * op ごとに出力元スロットを指定できる欄は作らない — `where(cond, a, b)` のように値の側が
 * 先頭でない op も、写像（bool → f32）で表せる。欄を増やすと「どのスロットから取るか」と
 * 「何に写すか」の 2 つの自由度が同じ事実を二重に持つ。
 */
export type SlotDtypes =
  | { readonly kind: "uniform"; readonly accept: readonly IrDtype[] }
  | { readonly kind: "perSlot"; readonly slots: readonly (readonly IrDtype[])[] };

type ContractBase = {
  /**
   * 受理する**入力の**意味論 dtype の**和**。capability 射影（{@link RUNTIME_SUPPORT}）と
   * 列挙門の診断が使う射影で、スロット別契約の正本は {@link slotDtypes}。
   *
   * MUST: 手書きせず {@link slotDtypes} から導出する（{@link unionDtypes}）— 二重管理すると
   * 「対応表では実行可、契約検査で落ちる」が生える。
   */
  readonly dtypes: readonly IrDtype[];
  /** スロット別の受理集合と出力導出の正本。 */
  readonly slotDtypes: SlotDtypes;
  /**
   * **スロット 0 の入力 dtype → 出力 dtype**。既定は恒等（入力と同型）で、違うのは実測に
   * 出た 3 系統だけ: 比較（f32 → bool）/ bool 入力の `sum`（→ i32 のカウント）/
   * `where`（条件 bool → 値の f32）。
   *
   * MUST: 定義域はスロット 0 の受理集合と**完全一致**させる（{@link slotContract} が恒等で
   * 埋める）。部分写像にすると、スロット検査を通った dtype の出力が決まらない穴ができる。
   * NOTE: `cast` だけはこの写像を使わない（出力は attrs.to — {@link resolveNodeDtypes}）。
   */
  readonly outputDtypes: ReadonlyMap<IrDtype, IrDtype>;
  readonly attrs: AttrSchema;
  /**
   * 入力数が可変の op（現状 `cat` のみ）。true のとき {@link OpContract} の `arity` は
   * **下限**として読む（{@link assertArity}）。
   *
   * MUST: 「アリティ検査を緩める」ためにこの欄を立てない。可変なのは cat の入力本数だけで、
   * 他の op は本数そのものが契約（bias 常時ありのアリティ 3 固定など）。
   * NOTE: capability 射影（{@link slotDtypesOf}）は下限ぶんのスロットしか作らない。uniform
   * 契約では全スロットが同じ受理集合なので、余ったスロットを和で見る列挙門（container.ts）と
   * 結論が一致する。
   */
  readonly variadic?: true;
  /**
   * **末尾に省略可能な入力**を持つ op の入力数の上限（現状 `attention` の mask だけ）。
   * 指定時は `arity` が下限・これが上限で、受理するのはその閉区間の本数だけ。
   *
   * MUST: `variadic` の代わりに使わない。`variadic` は「何本でも」（cat の連結）で、こちらは
   * 「決まったスロットが 1 つ増えるだけ」— 上限を持たない表現に潰すと、余分な入力が
   * 黙って無視される形（カーネルが読まないスロット）を契約が受理してしまう。
   * NOTE: capability 射影（{@link slotDtypesOf}）は**上限ぶん**のスロットを作る（省略可能な
   * スロットも dtype 契約を持つ — uniform なら本体と同じ受理集合）。
   */
  readonly maxArity?: number;
};

/**
 * kind と op 名を判別可能ユニオンで結ぶ。消費側（codegen 選択・CPU 参照）が
 * `op as UnaryOpName` のような取り違えの効かないキャストを書かずに済むようにするため。
 * `arity` は入力の個数（現状の op は全て単一出力）。
 */
export type OpContract =
  | (ContractBase & { readonly kind: "unary"; readonly name: UnaryOpName; readonly arity: 1 })
  | (ContractBase & { readonly kind: "binary"; readonly name: BinaryOpName; readonly arity: 2 })
  | (ContractBase & { readonly kind: "where"; readonly name: typeof WHERE_OP; readonly arity: 3 })
  | (ContractBase & { readonly kind: "cumsum"; readonly name: typeof CUMSUM_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "matmul"; readonly name: "matmul"; readonly arity: 2 })
  | (ContractBase & { readonly kind: "bmm"; readonly name: typeof BMM_OP; readonly arity: 2 })
  | (ContractBase & {
    readonly kind: "gather";
    readonly name: typeof GATHER_OP;
    readonly arity: 2;
  })
  | (ContractBase & { readonly kind: "rowReduce"; readonly name: ReduceOpName; readonly arity: 1 })
  | (ContractBase & { readonly kind: "cast"; readonly name: typeof CAST_OP; readonly arity: 1 })
  | (ContractBase & {
    readonly kind: "reshape";
    readonly name: typeof RESHAPE_OP;
    readonly arity: 1;
  })
  | (ContractBase & {
    readonly kind: "permute";
    readonly name: typeof PERMUTE_OP;
    readonly arity: 1;
  })
  | (ContractBase & {
    readonly kind: "expand";
    readonly name: typeof EXPAND_OP;
    readonly arity: 1;
  })
  | (ContractBase & { readonly kind: "slice"; readonly name: typeof SLICE_OP; readonly arity: 1 })
  // 唯一の可変アリティ op。`arity` は**下限** 2（1 本だけの cat は恒等コピーで、実測にも
  // 出ない — 表現を持たないことがそのまま fail loudly になる）。
  | (ContractBase & {
    readonly kind: "cat";
    readonly name: typeof CAT_OP;
    readonly arity: 2;
    readonly variadic: true;
  })
  | (ContractBase & { readonly kind: "pad"; readonly name: typeof PAD_OP; readonly arity: 1 })
  | (ContractBase & { readonly kind: "flip"; readonly name: typeof FLIP_OP; readonly arity: 1 })
  | (ContractBase & {
    readonly kind: "symPrefixSlice";
    readonly name: typeof SYM_PREFIX_SLICE_OP;
    readonly arity: 1;
  })
  | (ContractBase & { readonly kind: "linear"; readonly name: typeof LINEAR_OP; readonly arity: 3 })
  | (ContractBase & {
    readonly kind: "layerNorm";
    readonly name: typeof LAYER_NORM_OP;
    readonly arity: 3;
  })
  // affine が weight だけなのでアリティ 2（bias の欄を作らない — ADR 0017）。
  | (ContractBase & {
    readonly kind: "rmsNorm";
    readonly name: typeof RMS_NORM_OP;
    readonly arity: 2;
  })
  | (ContractBase & {
    readonly kind: "softmax";
    readonly name: typeof SOFTMAX_OP;
    readonly arity: 1;
  })
  // 融合 attention（ADR 0023）。arity 3〜4（4 本目は省略可能な加算 mask）・attrs `scale`
  // 宣言必須・rank-4 head-first。
  | (ContractBase & {
    readonly kind: "attention";
    readonly name: typeof ATTENTION_OP;
    readonly arity: 3;
    readonly maxArity: 4;
  })
  | (ContractBase & {
    readonly kind: "embedding";
    readonly name: typeof EMBEDDING_OP;
    readonly arity: 2;
  })
  | (ContractBase & {
    readonly kind: "maskedFill";
    readonly name: typeof MASKED_FILL_OP;
    readonly arity: 2;
  })
  | (ContractBase & {
    readonly kind: "conv1d";
    readonly name: typeof CONV1D_OP;
    readonly arity: 3;
  })
  | (ContractBase & {
    readonly kind: "conv2d";
    readonly name: typeof CONV2D_OP;
    readonly arity: 3;
  })
  | (ContractBase & {
    readonly kind: "convTranspose1d";
    readonly name: typeof CONV_TRANSPOSE1D_OP;
    readonly arity: 3;
  });

/** f32 専業の op（実測グラフに i32 / bool 形が現れていない — 対称性のためには解禁しない）。 */
const F32: readonly IrDtype[] = ["f32"];
/**
 * f32 と i32 の両方を実行できる op。根拠は DeBERTa front の実測（recon §3-8）:
 * mask 外積 `mul(i64, i64)` と `1 - attention_mask` の `sub`。いずれも入力値依存で
 * 定数畳み込みできないため実行系に必要。
 */
const F32_I32: readonly IrDtype[] = ["f32", "i32"];
/** bool 専業（否定・論理積は真偽値にしか意味が無い）。 */
const BOOL: readonly IrDtype[] = ["bool"];
/**
 * f32 と bool の両方を実行できる op。根拠は sdp の spline（recon §2 の searchsorted 行）:
 * `sum(inputs[…,None] >= bl, dim=-1)` が **bool の行 sum**（= 真の個数）で、出力は i32
 * （{@link OUTPUT_DTYPES}）。f32 の総和と併存する。
 */
const F32_BOOL: readonly IrDtype[] = ["f32", "bool"];
/** 整数添字専業（gather の index スロット — 実測は i32[16,T,T]）。 */
const I32: readonly IrDtype[] = ["i32"];
/** cast の**入力**側（出力は attrs.to）と、要素値に触れない reshape。 */
const ANY_DTYPE: readonly IrDtype[] = SEMANTIC_DTYPES;

const NO_ATTRS: AttrSchema = {};

/** op ごとの dtype 解禁表。ここに無い op は f32 のまま。 */
const DTYPES: ReadonlyMap<string, readonly IrDtype[]> = new Map([
  ["mul", F32_I32],
  ["sub", F32_I32],
  ["bitwise_not", BOOL],
  // spline の `inside = (x >= -b) & (x <= b)`（recon §2）— bool の論理積。整数の bitwise_and は
  // 意味が違うので解禁しない（bitwise_not と同じ絞り方）。
  ["bitwise_and", BOOL],
  ["sum", F32_BOOL],
  // reshape は要素を 1 つも読み書きしない（別名化のみ）ので全語彙。実測も f32 の view と
  // i32 mask の squeeze/unsqueeze、bool マスクの unsqueeze が全て出る（recon §2）。
  [RESHAPE_OP, ANY_DTYPE],
  // gather 添字（i32）・conv 経路の bool マスクに加え、相対位置埋め込みの 4D 化
  // `ek.expand(...)` / `ev.expand(...)`（f32 — recon §2 の enc_p / flow 行）で f32 も解禁。
  // strided コピー族は dtype パラメトリックなのでカーネルは共用のまま。
  [EXPAND_OP, ANY_DTYPE],
  // 焼いた定数は相対位置バケット表（i32）と位置テーブル（f32）の 2 系統。bool の
  // initializer は語彙に無い（ir.ts の INITIALIZER_STORAGE）ので解禁しない。
  [SYM_PREFIX_SLICE_OP, F32_I32],
]);

const dtypesOf = (name: string): readonly IrDtype[] => DTYPES.get(name) ?? F32;

/**
 * 入力（スロット 0）と出力で dtype が違う op の写像。ここに無い op は恒等（入力と同型）。
 *
 * MUST: 写像の値域は実測に出た形だけ（ADR 0009 の「op ごと実測ベース」）。
 * - 比較 4 本 — spline の `inside` 判定と searchsorted（recon §2）。真偽値なので bool。
 * - `sum` の bool 入力 — `sum(x >= bl, dim=-1)` は**真の個数**なので i32（f32 で数えると
 *   2^24 を超えた時点で静かに丸まる）。f32 入力は従来どおり f32。
 * - `where` — 条件が先頭スロットなので、写像だけが「出力は値の側」を表せる。
 */
const OUTPUT_DTYPES: ReadonlyMap<string, ReadonlyMap<IrDtype, IrDtype>> = new Map([
  ["ge", new Map<IrDtype, IrDtype>([["f32", "bool"]])],
  ["ge_scalar", new Map<IrDtype, IrDtype>([["f32", "bool"]])],
  ["le_scalar", new Map<IrDtype, IrDtype>([["f32", "bool"]])],
  ["gt_scalar", new Map<IrDtype, IrDtype>([["f32", "bool"]])],
  ["sum", new Map<IrDtype, IrDtype>([["f32", "f32"], ["bool", "i32"]])],
  [WHERE_OP, new Map<IrDtype, IrDtype>([["bool", "f32"]])],
]);

/**
 * cast の変換先を検査する。IR の意味論 dtype 語彙そのものを受理する（同型 cast は恒等コピー）。
 *
 * MUST: f32 → i32 は torch 準拠の **truncate（0 方向切り捨て）**、x → bool は **x != 0**。
 * bool の実表現は u32 の 0 / 1（ADR 0009）。丸め規約を契約に明記しないと、GPU 側の
 * `i32(x)`（WGSL も truncate）と CPU 参照が静かにずれる。
 */
const assertCastTarget = (value: unknown, where: string): IrDtype => {
  if (!isSemanticDtype(value)) {
    throw new OpContractError(
      `${where}: cast の変換先が意味論 dtype でない（${SEMANTIC_DTYPES.join(" / ")}）: ${
        JSON.stringify(value)
      }`,
    );
  }
  return value;
};

/** cast ノードの変換先 dtype（attrs の検査は {@link assertNodeContract} が済ませている前提）。 */
export const castTargetDtype = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): IrDtype =>
  assertCastTarget(Object.hasOwn(attrs, "to") ? attrs["to"] : undefined, `${where} の attrs.to`);

const CAST_ATTRS: AttrSchema = {
  to: (value, where) => {
    assertCastTarget(value, where);
  },
};

/**
 * permute の軸並べ替え表を検査する。`dims[d]` = 出力の次元 d が取る入力の次元番号。
 *
 * MUST: 負の軸番号を受理しない。torch の `-1` 表記はエクスポータ境界で正規化する規約で、
 * ランタイム側で両表記を受けると同じ並べ替えに 2 通りの IR ができる。
 * MUST: 重複を拒否する（並べ替えは全単射）。重複を許すと同じ入力軸を 2 度読む「複製」に
 * なり、要素数が合わないまま stride 計算だけが通る。入力 rank との突合は shape 計算側
 * （rank は束縛解決後にしか分からない）。
 */
const assertPermuteDims = (value: unknown, where: string): readonly number[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new OpContractError(
      `${where}: permute の dims が非空の配列でない: ${JSON.stringify(value)}`,
    );
  }
  const dims: number[] = value.map((dim) => {
    if (typeof dim !== "number" || !Number.isSafeInteger(dim) || dim < 0) {
      throw new OpContractError(
        `${where}: permute の dims に非負整数でない要素がある: ${JSON.stringify(value)}`,
      );
    }
    return dim;
  });
  if (new Set(dims).size !== dims.length) {
    throw new OpContractError(`${where}: permute の dims [${dims.join(",")}] に重複がある`);
  }
  return dims;
};

/** permute ノードの軸並べ替え表（attrs の検査は {@link assertNodeContract} が済ませている前提）。 */
export const permuteDims = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): readonly number[] =>
  assertPermuteDims(
    Object.hasOwn(attrs, "dims") ? attrs["dims"] : undefined,
    `${where} の attrs.dims`,
  );

const PERMUTE_ATTRS: AttrSchema = {
  dims: (value, where) => {
    assertPermuteDims(value, where);
  },
};

/** attrs の値を prototype 汚染に触れずに引く（`Object.hasOwn` のみ — 横断の不変条件）。 */
const attrValue = (attrs: Readonly<Record<string, unknown>>, key: string): unknown =>
  Object.hasOwn(attrs, key) ? attrs[key] : undefined;

/**
 * `min` 以上の整数 attr。
 * MUST: `typeof value === "number"` を先に見る（`Number.isSafeInteger` は非数値に false を
 * 返すので通るが、真偽値は数値化されないまま比較を素通りする形が JSON 由来で入りうる）。
 */
const assertIntegerAttr = (value: unknown, where: string, min: number): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min) {
    throw new OpContractError(`${where}: ${min} 以上の整数でない: ${JSON.stringify(value)}`);
  }
  return value;
};

/**
 * slice の切り出し指定（ADR 0014）。`dim` 軸を `[start, end)` に縮める。
 *
 * MUST: 3 つとも**非負整数**（負の軸表記・負の添字表記はエクスポータ境界で正規化する規約 —
 * permute の dims と同じ）。torch の `x[..., -1]` や既定の巨大 end をそのまま持ち込むと、
 * 同じ切り出しに 2 通りの IR ができる。
 * MUST: `start <= end` と `end <= 入力の軸長` は shape 計算側が見る（キーを跨ぐ不変条件と、
 * 入力 shape が要る規則は attrs スキーマでは表せない — clamp の min/max と同じ分担）。
 */
export type SliceAttrs = {
  readonly dim: number;
  readonly start: number;
  readonly end: number;
};

const SLICE_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  start: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  end: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/** slice ノードの attrs（検査は {@link assertNodeContract} が済ませている前提）。 */
export const sliceAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): SliceAttrs => ({
  dim: assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0),
  start: assertIntegerAttr(attrValue(attrs, "start"), `${where} の attrs.start`, 0),
  end: assertIntegerAttr(attrValue(attrs, "end"), `${where} の attrs.end`, 0),
});

/** cat / flip の軸（非負の軸番号。入力 rank との突合は shape 計算側）。 */
const AXIS_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

export const catDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

export const flipDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/**
 * pad の左右パディング幅（**最終次元・定数 0 のみ** — ADR 0014）。
 *
 * MUST: 負の幅（torch の `constant_pad_nd` は負で切り詰めができる）を受理しない。切り詰めは
 * pad ではなく slice の意味で、通すと「同じ形を 2 つの op で書ける」うえに専用カーネルの
 * 出力長計算が負になる。
 * MUST: 埋め値を attrs に持たない。実測は 0 のみで、欄を作らないことが「0 以外を黙って 0 で
 * 実行する」経路を構造的に潰す（conv1d の groups と同じ絞り方）。
 */
export type PadAttrs = {
  readonly left: number;
  readonly right: number;
};

const PAD_ATTRS: AttrSchema = {
  left: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  right: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

export const padAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): PadAttrs => ({
  left: assertIntegerAttr(attrValue(attrs, "left"), `${where} の attrs.left`, 0),
  right: assertIntegerAttr(attrValue(attrs, "right"), `${where} の attrs.right`, 0),
});

/** sym_prefix_slice の 1 軸ぶんの切り出し指定（`dim` を長さ `coeff·sym+offset` に縮める）。 */
export type PrefixSlice = {
  readonly dim: number;
  readonly coeff: number;
  readonly offset: number;
};

const PREFIX_SLICE_KEYS = ["dim", "coeff", "offset"] as const;

/**
 * sym_prefix_slice の `slices` を検査する。
 *
 * MUST: 軸の重複を拒否する。同じ軸に 2 つの指定があると「後勝ち」で片方が黙って消え、
 * 宣言 shape との照合だけが通る形が作れる。
 * MUST: 係数は 1 以上・オフセットは 0 以上（次元言語 `coeff·sym+offset` と同じ値域 —
 * 負を許すと prefix 長が負になり、要素数計算だけが 0 で通る）。
 * MUST: キーは `Object.hasOwn` のみで見る（JSON 由来のオブジェクトは prototype 経由で
 * `toString` 等を「持っている」ように見える — 横断の不変条件）。
 */
const assertPrefixSlices = (value: unknown, where: string): readonly PrefixSlice[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new OpContractError(
      `${where}: sym_prefix_slice の slices が非空の配列でない: ${JSON.stringify(value)}`,
    );
  }
  const slices: PrefixSlice[] = value.map((raw, index) => {
    const at = `${where}[${index}]`;
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new OpContractError(`${at}: オブジェクトでない: ${JSON.stringify(raw)}`);
    }
    const entry: Record<string, unknown> = raw as Record<string, unknown>;
    const unknown = Object.keys(entry).filter((key) =>
      !PREFIX_SLICE_KEYS.some((known) => known === key)
    );
    if (unknown.length > 0) {
      throw new OpContractError(`${at}: 未知のキー [${unknown.sort().join(", ")}]`);
    }
    for (const key of PREFIX_SLICE_KEYS) {
      if (!Object.hasOwn(entry, key)) throw new OpContractError(`${at}: キー '${key}' が無い`);
    }
    return {
      dim: assertIntegerAttr(entry["dim"], `${at}.dim`, 0),
      coeff: assertIntegerAttr(entry["coeff"], `${at}.coeff`, 1),
      offset: assertIntegerAttr(entry["offset"], `${at}.offset`, 0),
    };
  });
  const dims = slices.map((slice) => slice.dim);
  if (new Set(dims).size !== dims.length) {
    throw new OpContractError(`${where}: sym_prefix_slice の slices に同じ dim が 2 度ある`);
  }
  return slices;
};

/** sym_prefix_slice の `sym`（次元言語のシンボル名 — 束縛済みかはグラフ側の検査）。 */
const assertPrefixSym = (value: unknown, where: string): string => {
  if (typeof value !== "string" || !isSymbolName(value)) {
    throw new OpContractError(
      `${where}: sym_prefix_slice の sym がシンボル名でない: ${JSON.stringify(value)}`,
    );
  }
  return value;
};

const SYM_PREFIX_SLICE_ATTRS: AttrSchema = {
  sym: (value, where) => {
    assertPrefixSym(value, where);
  },
  slices: (value, where) => {
    assertPrefixSlices(value, where);
  },
};

/** sym_prefix_slice ノードの attrs（検査は {@link assertNodeContract} が済ませている前提）。 */
export const symPrefixSliceAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly sym: string; readonly slices: readonly PrefixSlice[] } => ({
  sym: assertPrefixSym(attrValue(attrs, "sym"), `${where} の attrs.sym`),
  slices: assertPrefixSlices(attrValue(attrs, "slices"), `${where} の attrs.slices`),
});

/**
 * layer_norm の正規化軸。**長さ 1（= 最終次元）のみ**受理する。
 *
 * MUST: 多軸正規化を「対称性のため」受け入れない（ADR 0007 の語彙 allowlist 凍結）。実測は
 * 全 7 本が `[1024]`（recon §5）で、行カーネルは最終次元の連続並びを前提に組んである。
 * 軸との突合（`x.shape` の末尾と一致するか）は束縛解決後の shape 計算側。
 */
const assertNormalizedShape = (value: unknown, where: string): readonly number[] => {
  if (!Array.isArray(value) || value.length !== 1) {
    throw new OpContractError(
      `${where}: layer_norm の normalized_shape は長さ 1 の配列のみ（最終次元の正規化だけを実行できる）: ${
        JSON.stringify(value)
      }`,
    );
  }
  return [assertIntegerAttr(value[0], `${where}[0]`, 1)];
};

/**
 * 正規化 op（layer_norm / rms_norm）の eps。**有限の正数**のみ（0 を許すと分散 0・
 * 全要素 0 の行で `1/sqrt(0)` が inf になり、「定数行の正規化」が黙って NaN を吐く）。
 */
const assertEps = (value: unknown, where: string, what: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new OpContractError(
      `${where}: ${what} の eps は有限の正数でない: ${JSON.stringify(value)}`,
    );
  }
  return value;
};

/**
 * params の f32 語で運ぶスカラ attr。**有限の f32 スカラ**（IR v1 は非有限値を JSON
 * リテラルでも値レベルでも拒否する）。
 *
 * NOTE: f32 に厳密表現できる値だけに絞りはしない — 適用時に f32 へ丸める規約（GPU は
 * params の f32 語、CPU 参照は `Math.fround`）で両側が一致する。
 */
const assertFiniteAttr = (value: unknown, where: string, what: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OpContractError(`${where}: ${what} は有限の数値でない: ${JSON.stringify(value)}`);
  }
  return value;
};

/**
 * masked_fill の埋め値。実測値 −3.4028234663852886e+38 は f32 の最小有限値ちょうどで、
 * JSON 往復でも f32 丸めでも ulp が動かない（tests/ops_contract_test.ts が固定）。
 */
const assertFillValue = (value: unknown, where: string): number =>
  assertFiniteAttr(value, where, "masked_fill の value");

const LAYER_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "layer_norm");
  },
  normalized_shape: (value, where) => {
    assertNormalizedShape(value, where);
  },
};

/**
 * rms_norm の attrs（ADR 0017）。
 *
 * MUST: `normalized_shape` の欄を作らない。正規化軸は最終次元 1 本に固定で、長さは
 * **weight の長さ**が持つ（layer_norm は attrs と weight で同じ事実を二重に持っていて、
 * shape 計算がその一致を毎回検査している）。欄を作ると同じ二重管理をもう 1 op ぶん増やす。
 */
const RMS_NORM_ATTRS: AttrSchema = {
  eps: (value, where) => {
    assertEps(value, where, "rms_norm");
  },
};

/** rms_norm ノードの eps（検査は {@link assertNodeContract} が済ませている前提）。 */
export const rmsNormEps = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertEps(attrValue(attrs, "eps"), `${where} の attrs.eps`, "rms_norm");

const SOFTMAX_ATTRS: AttrSchema = {
  // 負の軸表記はエクスポータ境界で正規化する規約（permute の dims と同じ）。最終次元との
  // 突合は rank が分かる shape 計算側。
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * attention の attrs（ADR 0023）。
 *
 * MUST: `scale` は**宣言必須で既定値補完をしない**。エクスポータが SDPA の `scale` 引数
 * （省略時 `1/√D`）から `f32(√scale_factor)` を計算して載せる規約で、ランタイム側が
 * 「無ければ 1/√D」を補うと、`scale` を明示した SDPA の IR と補完した IR が同じ形になり
 * 取り違えが値にしか出なくなる（conv1d の dilation / groups と同じ規律 — ADR 0015）。
 * MUST: mask / causal / dropout の欄を作らない。欄が無いこと自体が「その形は語彙に無い」を
 * 構造で表す（pad の埋め値と同じ絞り方）。
 */
const ATTENTION_ATTRS: AttrSchema = {
  scale: (value, where) => {
    assertFiniteAttr(value, where, "attention の scale");
  },
};

/** attention ノードの半スケール（検査は {@link assertNodeContract} が済ませている前提）。 */
export const attentionScale = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number =>
  assertFiniteAttr(attrValue(attrs, "scale"), `${where} の attrs.scale`, "attention の scale");

const EMBEDDING_ATTRS: AttrSchema = {
  /**
   * torch の `padding_idx`。**受理するが forward には効かない**（勾配で padding 行を更新
   * しないための欄で、順伝播は素の行 gather と完全に同じ — torch の `F.embedding` も
   * forward では参照しない）。したがってカーネルにも CPU 参照にも渡さない。
   *
   * MUST: それでも契約表に載せる。attrs を落として無視すると「未知 attr は fail loudly」の
   * 規律に穴が開き、次に forward へ効く欄が来たときに同じ理由で素通りする。
   * 値域は torch 準拠で `-1`（未指定を表す番兵）以上。
   */
  padding_idx: (value, where) => {
    assertIntegerAttr(value, where, -1);
  },
};

const MASKED_FILL_ATTRS: AttrSchema = {
  value: (value, where) => {
    assertFillValue(value, where);
  },
};

const CLAMP_ATTRS: AttrSchema = {
  min: (value, where) => {
    assertFiniteAttr(value, where, "clamp の min");
  },
  max: (value, where) => {
    assertFiniteAttr(value, where, "clamp の max");
  },
};

/**
 * clamp_min の attrs（ADR 0017 — チャネル L2 正規化の `clamp(min=eps)` 30 本）。
 *
 * MUST: `max` の欄を作らない。「欠けた側を f32 の最大有限値で補って clamp へ流す」は
 * 上限を持たない意味論を「上限が飽和するほど大きい」で置き換える近似で、ADR 0017 が
 * 名指しで却下している。欄が無いこと自体が両側必須の clamp との住み分けを構造で表す。
 */
const CLAMP_MIN_ATTRS: AttrSchema = {
  min: (value, where) => {
    assertFiniteAttr(value, where, "clamp_min の min");
  },
};

const LEAKY_RELU_ATTRS: AttrSchema = {
  /**
   * torch の `negative_slope`。**必須で既定値補完はしない**（ADR 0015）— dec は 0.1
   * （ups / ResBlock）と 0.01（最終段・位置引数ごと省略）が混在し、既定に頼ると片方が
   * 黙って誤る。torch 側の既定はエクスポータが読み取って attrs に載せる（境界で明示化する）。
   */
  negative_slope: (value, where) => {
    assertFiniteAttr(value, where, "leaky_relu の negative_slope");
  },
};

/** ge_scalar / le_scalar / gt_scalar の比較相手（有限の f32 スカラ）。 */
const SCALAR_COMPARE_ATTRS: AttrSchema = {
  value: (value, where) => {
    assertFiniteAttr(value, where, "比較 op の value");
  },
};

const CUMSUM_ATTRS: AttrSchema = {
  // 負の軸表記はエクスポータ境界で正規化する規約（softmax の dim と同じ）。最終次元との
  // 突合は rank が分かる shape 計算側。
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * reduce 族（sum / amax / amin）の縮約軸。
 *
 * MUST: **宣言必須**（既定値補完をしない — conv1d の dilation / groups と同じ理由）。
 * 「欄が無い = 最終次元」を許すと、チャネル軸の縮約を書いたつもりの IR が黙って最終次元を
 * 畳んだ別の計算として実行される（形が合ってしまう組み合わせが実在する）。
 * 負の軸表記はエクスポータ境界で正規化する規約で、rank との突合は shape 計算側。
 */
const REDUCE_ATTRS: AttrSchema = {
  dim: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
};

/**
 * conv1d の attrs（ADR 0015 で `dilation` / `groups` を追加）。
 *
 * MUST: 4 つとも**宣言必須**（{@link assertNodeContract} が全キーの存在を要求する）。
 * 「欄が無い = 1 固定」で担保していた「1 以外を黙って 1 で実行する経路が無い」性質は、
 * 欄を作った後は「既定値補完をしない」ことだけが担保している — `dilation` / `groups` を
 * 省略可能にした瞬間に、depthwise の IR が黙って通常畳み込みとして実行される。
 */
const CONV1D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  stride: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/**
 * conv_transpose1d の attrs（ADR 0015）。
 *
 * MUST: `stride >= 1`。stride 0 は「入力 1 点が出力の 1 点にしか寄与しない」形にすら
 * ならず、実装によってはループが進まず **GPU ハング**（例外にならない）— 契約検査と
 * params 検査の両方で遮断する（recon §4）。
 * MUST: `output_padding` / `dilation` / `groups` の欄を作らない。実測は全て 0 / 1 / 1 で、
 * 欄を持たないことが「0 / 1 以外を黙って既定値で実行する」経路を構造的に潰す
 * （pad の埋め値と同じ絞り方）。広げるのは実測が出てから。
 */
const CONV_TRANSPOSE1D_ATTRS: AttrSchema = {
  padding: (value, where) => {
    assertIntegerAttr(value, where, 0);
  },
  stride: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
};

/** layer_norm ノードの attrs（検査は {@link assertNodeContract} が済ませている前提）。 */
export const layerNormAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly normalizedShape: readonly number[]; readonly eps: number } => ({
  normalizedShape: assertNormalizedShape(
    attrValue(attrs, "normalized_shape"),
    `${where} の attrs.normalized_shape`,
  ),
  eps: assertEps(attrValue(attrs, "eps"), `${where} の attrs.eps`, "layer_norm"),
});

/** softmax ノードの縮約軸（非負の軸番号）。 */
export const softmaxDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/** masked_fill ノードの埋め値。 */
export const maskedFillValue = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): number => assertFillValue(attrValue(attrs, "value"), `${where} の attrs.value`);

/** cumsum ノードの累積軸（非負の軸番号）。 */
export const cumsumDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/** reduce 族ノードの縮約軸（非負の軸番号 — 既定値補完はしない）。 */
export const reduceDim = (attrs: Readonly<Record<string, unknown>>, where: string): number =>
  assertIntegerAttr(attrValue(attrs, "dim"), `${where} の attrs.dim`, 0);

/**
 * elementwise カーネルへ **params の末尾で** f32 として渡す attr（並びがそのまま params の
 * レイアウト）。
 *
 * MUST: 値を WGSL に焼かない。焼くと値の種類だけパイプラインが増える（masked_fill の
 * 埋め値と同じ理由 — src/kernels/masked-fill.ts）。
 * MUST: 並びの正本はここ 1 箇所。attrs スキーマのキー順（オブジェクトの挿入順）に頼ると、
 * 宣言の並べ替えが codegen の添字と params の書き込み順を黙ってずらす。
 */
export const SCALAR_PARAM_ATTRS: ReadonlyMap<string, readonly string[]> = new Map([
  ["clamp", ["min", "max"]],
  ["clamp_min", ["min"]],
  ["leaky_relu", ["negative_slope"]],
  ["ge_scalar", ["value"]],
  ["le_scalar", ["value"]],
  ["gt_scalar", ["value"]],
]);

/** op が params 末尾に載せる f32 スカラの本数（codegen が添字を決めるのに使う）。 */
export const scalarParamCount = (op: string): number => SCALAR_PARAM_ATTRS.get(op)?.length ?? 0;

/**
 * スカラ attr を params の並び順で取り出す（検査は {@link assertNodeContract} が済ませている
 * 前提だが、CPU 参照から直接呼ばれる経路もあるので値域はここでも見る）。
 *
 * MUST: clamp の `min <= max` はここでしか見られない（attrs スキーマはキー単位の検査なので、
 * 2 つのキーに跨る不変条件を表せない）。逆転を許すと WGSL 側の分岐順に依存した「片方だけ
 * 効く clamp」になり、CPU 参照との差が入力の位置によって出たり出なかったりする。
 */
export const scalarParamValues = (
  found: OpContract,
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): readonly number[] => {
  const keys = SCALAR_PARAM_ATTRS.get(found.name);
  if (keys === undefined) return [];
  const values = keys.map((key) =>
    assertFiniteAttr(
      attrValue(attrs, key),
      `${where} の attrs.${key}`,
      `op '${found.name}' の ${key}`,
    )
  );
  if (found.name === "clamp" && values[0] > values[1]) {
    throw new OpContractError(`${where}: clamp の min ${values[0]} が max ${values[1]} を超える`);
  }
  return values;
};

/**
 * conv2d の空間 attr（`[H, W]` の 2 成分）。
 *
 * MUST: **長さちょうど 2 の配列**のみ受理する。スカラ表記（torch の `stride=1` が両軸に
 * 効く形）を併せて許すと同じ畳み込みに 2 通りの IR ができ、CSE も適合表の突合も割れる —
 * 正規化はエクスポータ境界の仕事（permute の負の軸表記と同じ分担）。
 * MUST: H と W を別のキーに割らない（`stride_h` / `stride_w`）。2 軸で 3 つの attr なので
 * 6 キーになり、「片方だけ書き忘れた IR」の見え方が「必須キー欠落」から「値が既定に
 * 見える」へ落ちる。組であることを型で表す。
 */
const assertIntPair = (
  value: unknown,
  where: string,
  min: number,
  what: string,
): readonly [number, number] => {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new OpContractError(
      `${where}: ${what} は [H, W] の長さ 2 の配列でない: ${JSON.stringify(value)}`,
    );
  }
  return [
    assertIntegerAttr(value[0], `${where}[0]`, min),
    assertIntegerAttr(value[1], `${where}[1]`, min),
  ];
};

/**
 * conv2d の attrs（ADR 0017）。空間 3 つは H/W の 2 成分、`groups` はスカラ。
 *
 * MUST: 4 つとも**宣言必須・既定値補完なし**（conv1d と同じ規律 — ADR 0015）。
 * depthwise（groups = Cin = Cout）と非対称 stride/padding が実測に出るので、省略を許すと
 * 黙って通常畳み込み・対称パディングとして実行される。
 */
const CONV2D_ATTRS: AttrSchema = {
  dilation: (value, where) => {
    assertIntPair(value, where, 1, "conv2d の dilation");
  },
  groups: (value, where) => {
    assertIntegerAttr(value, where, 1);
  },
  padding: (value, where) => {
    assertIntPair(value, where, 0, "conv2d の padding");
  },
  stride: (value, where) => {
    assertIntPair(value, where, 1, "conv2d の stride");
  },
};

/** conv2d ノードの stride / padding / dilation（H/W の組）と groups。 */
export type Conv2dAttrs = {
  readonly stride: readonly [number, number];
  readonly padding: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly groups: number;
};

export const conv2dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv2dAttrs => ({
  stride: assertIntPair(
    attrValue(attrs, "stride"),
    `${where} の attrs.stride`,
    1,
    "conv2d の stride",
  ),
  padding: assertIntPair(
    attrValue(attrs, "padding"),
    `${where} の attrs.padding`,
    0,
    "conv2d の padding",
  ),
  dilation: assertIntPair(
    attrValue(attrs, "dilation"),
    `${where} の attrs.dilation`,
    1,
    "conv2d の dilation",
  ),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} の attrs.groups`, 1),
});

/** conv1d ノードの stride / padding / dilation / groups。 */
export type Conv1dAttrs = {
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
};

export const conv1dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): Conv1dAttrs => ({
  stride: assertIntegerAttr(attrValue(attrs, "stride"), `${where} の attrs.stride`, 1),
  padding: assertIntegerAttr(attrValue(attrs, "padding"), `${where} の attrs.padding`, 0),
  dilation: assertIntegerAttr(attrValue(attrs, "dilation"), `${where} の attrs.dilation`, 1),
  groups: assertIntegerAttr(attrValue(attrs, "groups"), `${where} の attrs.groups`, 1),
});

/** conv_transpose1d ノードの stride / padding。 */
export const convTranspose1dAttrs = (
  attrs: Readonly<Record<string, unknown>>,
  where: string,
): { readonly stride: number; readonly padding: number } => ({
  stride: assertIntegerAttr(attrValue(attrs, "stride"), `${where} の attrs.stride`, 1),
  padding: assertIntegerAttr(attrValue(attrs, "padding"), `${where} の attrs.padding`, 0),
});

/**
 * スロット別の受理集合を 1 本の和へ畳む。順序は語彙の宣言順（{@link SEMANTIC_DTYPES}）で
 * 固定する — 表示と突合に使う射影なので、スロットの並びで順序が変わってはいけない。
 */
const unionDtypes = (slotDtypes: SlotDtypes): readonly IrDtype[] =>
  slotDtypes.kind === "uniform"
    ? slotDtypes.accept
    : SEMANTIC_DTYPES.filter((dtype) => slotDtypes.slots.some((accept) => accept.includes(dtype)));

const contract = (name: string, attrs: AttrSchema = NO_ATTRS): ContractBase =>
  slotContract(name, { kind: "uniform", accept: dtypesOf(name) }, attrs);

/** スロット 0 の受理集合（uniform は全スロット共通）。出力 dtype 写像の定義域になる。 */
const firstSlotAccept = (slotDtypes: SlotDtypes): readonly IrDtype[] =>
  slotDtypes.kind === "uniform" ? slotDtypes.accept : slotDtypes.slots[0];

const slotContract = (
  name: string,
  slotDtypes: SlotDtypes,
  attrs: AttrSchema = NO_ATTRS,
): ContractBase => {
  const declared = OUTPUT_DTYPES.get(name);
  // 宣言が無い op は恒等写像で埋める（定義域 = スロット 0 の受理集合という不変条件を
  // 構造で満たす — 表の書き忘れが「出力が決まらない dtype」にならない）。
  const outputDtypes = new Map<IrDtype, IrDtype>(
    firstSlotAccept(slotDtypes).map((dtype) => [dtype, declared?.get(dtype) ?? dtype]),
  );
  return { dtypes: unionDtypes(slotDtypes), slotDtypes, outputDtypes, attrs };
};

/** attrs を持つ単項 op のスキーマ（無い op は attrs 空）。 */
const UNARY_ATTRS: ReadonlyMap<string, AttrSchema> = new Map([
  ["clamp", CLAMP_ATTRS],
  ["clamp_min", CLAMP_MIN_ATTRS],
  ["leaky_relu", LEAKY_RELU_ATTRS],
  ["ge_scalar", SCALAR_COMPARE_ATTRS],
  ["le_scalar", SCALAR_COMPARE_ATTRS],
  ["gt_scalar", SCALAR_COMPARE_ATTRS],
]);

export const OP_CONTRACTS: ReadonlyMap<string, OpContract> = new Map<string, OpContract>([
  ...UNARY_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name, UNARY_ATTRS.get(name) ?? NO_ATTRS), kind: "unary", name, arity: 1 },
  ]),
  ...BINARY_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name), kind: "binary", name, arity: 2 },
  ]),
  ...REDUCE_OPS.map((name): [string, OpContract] => [
    name,
    { ...contract(name, REDUCE_ATTRS), kind: "rowReduce", name, arity: 1 },
  ]),
  // 条件 bool と値 f32 のスロット別契約。出力は**値の側**（写像 bool → f32 — 出力元
  // スロットを指す欄は作らない規律）。
  [WHERE_OP, {
    ...slotContract(WHERE_OP, { kind: "perSlot", slots: [BOOL, F32, F32] }),
    kind: "where",
    name: WHERE_OP,
    arity: 3,
  }],
  [CUMSUM_OP, {
    ...contract(CUMSUM_OP, CUMSUM_ATTRS),
    kind: "cumsum",
    name: CUMSUM_OP,
    arity: 1,
  }],
  [MATMUL_OP, { ...contract(MATMUL_OP), kind: "matmul", name: MATMUL_OP, arity: 2 }],
  [BMM_OP, { ...contract(BMM_OP), kind: "bmm", name: BMM_OP, arity: 2 }],
  // 最初のスロット別 dtype 契約: 値 f32 と添字 i32 が混在し、出力は値の側と同型。
  [GATHER_OP, {
    ...slotContract(GATHER_OP, { kind: "perSlot", slots: [F32, I32] }),
    kind: "gather",
    name: GATHER_OP,
    arity: 2,
  }],
  [CAST_OP, {
    ...slotContract(CAST_OP, { kind: "uniform", accept: ANY_DTYPE }, CAST_ATTRS),
    kind: "cast",
    name: CAST_OP,
    arity: 1,
  }],
  [RESHAPE_OP, { ...contract(RESHAPE_OP), kind: "reshape", name: RESHAPE_OP, arity: 1 }],
  [PERMUTE_OP, {
    ...contract(PERMUTE_OP, PERMUTE_ATTRS),
    kind: "permute",
    name: PERMUTE_OP,
    arity: 1,
  }],
  [EXPAND_OP, { ...contract(EXPAND_OP), kind: "expand", name: EXPAND_OP, arity: 1 }],
  // レイアウト第 2 群（ADR 0014）。実測は全て f32（enc_p の m_p/logs_p 分割・coupling の
  // 96/96 分割と cat・相対位置 value 側の pad・flow/sdp の Flip — recon §2）なので、
  // dtype は解禁表に載せない = f32 専業。
  [SLICE_OP, { ...contract(SLICE_OP, SLICE_ATTRS), kind: "slice", name: SLICE_OP, arity: 1 }],
  [CAT_OP, {
    ...contract(CAT_OP, AXIS_ATTRS),
    kind: "cat",
    name: CAT_OP,
    arity: 2,
    variadic: true,
  }],
  [PAD_OP, { ...contract(PAD_OP, PAD_ATTRS), kind: "pad", name: PAD_OP, arity: 1 }],
  [FLIP_OP, { ...contract(FLIP_OP, AXIS_ATTRS), kind: "flip", name: FLIP_OP, arity: 1 }],
  [SYM_PREFIX_SLICE_OP, {
    ...contract(SYM_PREFIX_SLICE_OP, SYM_PREFIX_SLICE_ATTRS),
    kind: "symPrefixSlice",
    name: SYM_PREFIX_SLICE_OP,
    arity: 1,
  }],
  // 融合 op（ADR 0012）。bias / affine を持つ 3 本はアリティ 3 固定 — 実測が bias 常時ありで、
  // 「bias 無し」を表す欄を作らないことがそのまま fail loudly になる。
  [LINEAR_OP, { ...contract(LINEAR_OP), kind: "linear", name: LINEAR_OP, arity: 3 }],
  [LAYER_NORM_OP, {
    ...contract(LAYER_NORM_OP, LAYER_NORM_ATTRS),
    kind: "layerNorm",
    name: LAYER_NORM_OP,
    arity: 3,
  }],
  // bias が無いのでアリティ 2（ADR 0017）。weight 無しの形はエクスポータが ones 合成で
  // アリティ 2 へ正規化する — ゼロ bias 合成（ADR 0015）と同じ手筋。
  [RMS_NORM_OP, {
    ...contract(RMS_NORM_OP, RMS_NORM_ATTRS),
    kind: "rmsNorm",
    name: RMS_NORM_OP,
    arity: 2,
  }],
  [SOFTMAX_OP, {
    ...contract(SOFTMAX_OP, SOFTMAX_ATTRS),
    kind: "softmax",
    name: SOFTMAX_OP,
    arity: 1,
  }],
  // 融合 attention（ADR 0023）。q / k / v と省略可能な mask の 4 本とも f32 で同型
  // （uniform 契約）。mask は加算型なので値の側と同じ dtype で、bool は受理しない。
  [ATTENTION_OP, {
    ...contract(ATTENTION_OP, ATTENTION_ATTRS),
    kind: "attention",
    name: ATTENTION_OP,
    arity: 3,
    maxArity: 4,
  }],
  // 値 f32 と添字 i32 のスロット別契約（gather と同型 — 出力は値の側と同型）。
  [EMBEDDING_OP, {
    ...slotContract(EMBEDDING_OP, { kind: "perSlot", slots: [F32, I32] }, EMBEDDING_ATTRS),
    kind: "embedding",
    name: EMBEDDING_OP,
    arity: 2,
  }],
  // 値 f32 と条件 bool のスロット別契約。出力はスロット 0（値の側）と同型。
  [MASKED_FILL_OP, {
    ...slotContract(MASKED_FILL_OP, { kind: "perSlot", slots: [F32, BOOL] }, MASKED_FILL_ATTRS),
    kind: "maskedFill",
    name: MASKED_FILL_OP,
    arity: 2,
  }],
  [CONV1D_OP, {
    ...contract(CONV1D_OP, CONV1D_ATTRS),
    kind: "conv1d",
    name: CONV1D_OP,
    arity: 3,
  }],
  [CONV2D_OP, {
    ...contract(CONV2D_OP, CONV2D_ATTRS),
    kind: "conv2d",
    name: CONV2D_OP,
    arity: 3,
  }],
  // bias 無し conv はエクスポータのゼロ bias 合成でアリティ 3 に正規化される（ADR 0015）—
  // カーネルにも契約にも arity 分岐を持ち込まない。
  [CONV_TRANSPOSE1D_OP, {
    ...contract(CONV_TRANSPOSE1D_OP, CONV_TRANSPOSE1D_ATTRS),
    kind: "convTranspose1d",
    name: CONV_TRANSPOSE1D_OP,
    arity: 3,
  }],
]);

/** 契約の attrs スキーマが宣言するキー（capability 射影と診断で使う）。 */
export const attrKeysOf = (found: OpContract): readonly string[] => Object.keys(found.attrs);

/**
 * 入力スロット別の受理集合（capability 射影）。uniform 契約は**受理しうるスロット数ぶん
 * 複製**する（省略可能な末尾スロットを含む — {@link ContractBase.maxArity}）。
 * 消費側（列挙門）がスロット番号で引けるようにするため。
 */
export const slotDtypesOf = (found: OpContract): readonly (readonly IrDtype[])[] => {
  const slots = found.slotDtypes;
  return slots.kind === "uniform"
    ? Array.from({ length: found.maxArity ?? found.arity }, () => slots.accept)
    : slots.slots;
};

/**
 * グラフ入力として転送できる意味論 dtype。要素は全型 4 バイトで、bool は u32 の 0/1
 * として運ぶ（ADR 0009）。op の dtype 集合とは別軸 — 実行器はどのノードも消費しない入力も
 * 含めて転送するため。
 */
export const IO_DTYPES: readonly IrDtype[] = SEMANTIC_DTYPES;

/**
 * `assertRuntimeSupport` に渡す対応表。契約表からのみ導出する MUST — 手書きの集合は
 * 表と乖離し、「宣言では対応、実行で落ちる」を作る。op 名だけでなく意味論 dtype と attrs
 * まで運ぶ（名前だけの突合が見逃す差は recon §3-9 の実バグ）。
 */
export const RUNTIME_SUPPORT: RuntimeSupport = {
  ops: new Map<string, OpSupport>(
    [...OP_CONTRACTS].map(([name, found]) => [name, {
      dtypes: new Set(found.dtypes),
      slotDtypes: slotDtypesOf(found).map((accept) => new Set(accept)),
      // cast の出力は attrs.to で決まる（写像は恒等で埋まっているが値域が狭すぎる）ので、
      // 列挙門には語彙全体を渡す。それ以外は写像の値域そのもの。
      outDtypes: found.kind === "cast"
        ? new Set(SEMANTIC_DTYPES)
        : new Set(found.outputDtypes.values()),
      attrKeys: new Set(attrKeysOf(found)),
    }]),
  ),
  // 生の int32 格納（ADR 0010）は記号依存定数の焼き込み先として実行対象。f16（ADR 0018）と
  // i8（ADR 0019）は実行経路が入った（適格な重みスロットは圧縮のまま GPU 常駐・適格外は
  // ロード時に CPU で f32 展開）ので、**どの initializer に付いていても実行できる**。
  // bf16 だけが宣言としては valid で実行できない（capability 不足として列挙で落ちる）。
  storage: new Set(["f32", "f16", "i8", "i32"]),
  io: new Set(IO_DTYPES),
};

/** ランタイムが実行できるものの照会用（公開 API 向けに安定した形へ整えた射影）。 */
export type RuntimeCapabilities = {
  readonly ops: readonly string[];
  readonly storage: readonly string[];
};

export const capabilities = (): RuntimeCapabilities => ({
  ops: [...RUNTIME_SUPPORT.ops.keys()].sort(),
  storage: [...RUNTIME_SUPPORT.storage].sort(),
});

export const numel = (shape: readonly number[]): number =>
  shape.reduce((count, dim) => count * dim, 1);

/**
 * 入力の本数が契約に合うか（可変アリティ op では `arity` を**下限**として読む）。
 *
 * MUST: 判定を呼び出し側に散らさない。「固定なら ===、可変なら >=」を 3 箇所（契約検査 /
 * shape 計算 / CPU 参照）で書くと、cat を足したときのように 1 箇所だけ古い判定が残る。
 * 例外の**型**は層ごとに違う（契約層は OpContractError / CPU 参照は ReferenceOpError）ので、
 * 共有するのは述語と表示だけにする。
 */
export const arityFits = (found: OpContract, count: number): boolean =>
  found.variadic === true
    ? count >= found.arity
    : count >= found.arity && count <= (found.maxArity ?? found.arity);

/** 契約のアリティの表示形（可変なら「N 本以上」・省略可能な末尾があれば「N か M」）。 */
export const describeArity = (found: OpContract): string => {
  if (found.variadic === true) return `${found.arity} 本以上`;
  return found.maxArity === undefined ? `${found.arity}` : `${found.arity} か ${found.maxArity}`;
};

export const assertArity = (
  found: OpContract,
  count: number,
  what: string,
  where: string,
): void => {
  if (arityFits(found, count)) return;
  throw new OpContractError(
    `${where}: op '${found.name}' の${what}が ${count}（契約は ${describeArity(found)}）`,
  );
};

export const resolveOpContract = (op: string): OpContract => {
  const found = OP_CONTRACTS.get(op);
  if (found === undefined) {
    throw new OpContractError(`op '${op}' は契約表に無い`);
  }
  return found;
};

/**
 * ノードが契約に適合することを検査して契約を返す（shape は含まない — 束縛前に決まる規則だけ）。
 *
 * MUST: attrs の照合は `Object.hasOwn` のみで行う。スキーマは素のオブジェクトリテラルなので、
 * IR 由来のキー（`toString` 等）が prototype 経由で「既知キー」に化ける。
 */
export const assertNodeContract = (node: IrNode, where: string): OpContract => {
  const found = resolveOpContract(node.op);
  assertArity(found, node.ins.length, "入力数", where);
  if (node.outs.length !== 1) {
    throw new OpContractError(
      `${where}: op '${node.op}' の出力数が ${node.outs.length}（現状の op は全て単一出力）`,
    );
  }
  const unknown = Object.keys(node.attrs).filter((key) => !Object.hasOwn(found.attrs, key));
  if (unknown.length > 0) {
    throw new OpContractError(
      `${where}: op '${node.op}' の契約外 attrs [${unknown.sort().join(", ")}]`,
    );
  }
  for (const key of attrKeysOf(found)) {
    if (!Object.hasOwn(node.attrs, key)) {
      throw new OpContractError(`${where}: op '${node.op}' の必須 attr '${key}' が無い`);
    }
    found.attrs[key](node.attrs[key], `${where} の attrs.${key}`);
  }
  return found;
};

/**
 * スロット 0 の入力 dtype から出力 dtype を導く（{@link ContractBase.outputDtypes}）。
 *
 * 写像の定義域はスロット 0 の受理集合と一致する（{@link slotContract} が恒等で埋める）ので、
 * スロット検査を通った dtype は必ず引ける — 引けないのはランタイム内部の不変条件破れ。
 */
export const outputDtypeOf = (
  found: OpContract,
  inputDtype: IrDtype,
  where: string,
): IrDtype => {
  const mapped = found.outputDtypes.get(inputDtype);
  if (mapped === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' の出力 dtype 写像に入力 '${inputDtype}' が無い`,
    );
  }
  return mapped;
};

/**
 * 受理集合の**和**との突合。スロットを跨いだ緩い検査なので、スロット別契約の op では
 * {@link assertSlotDtype} を使う（この関数は「その op が触れる dtype か」しか見ない）。
 */
export const assertDtype = (found: OpContract, dtype: IrDtype, where: string): void => {
  if (!found.dtypes.includes(dtype)) {
    throw new OpContractError(
      `${where}: op '${found.name}' は意味論 dtype '${dtype}' を実行できない（対応: ${
        found.dtypes.join(", ")
      }）`,
    );
  }
};

/**
 * 入力スロット単位の dtype 検査。uniform 契約では和との突合と同義（診断文も同じ）で、
 * perSlot 契約でだけスロット番号まで見る。
 */
export const assertSlotDtype = (
  found: OpContract,
  slot: number,
  dtype: IrDtype,
  where: string,
): void => {
  if (found.slotDtypes.kind === "uniform") {
    assertDtype(found, dtype, where);
    return;
  }
  const accept = found.slotDtypes.slots[slot];
  if (accept === undefined) {
    throw new OpContractError(`${where}: op '${found.name}' に入力スロット ${slot} は無い`);
  }
  if (!accept.includes(dtype)) {
    throw new OpContractError(
      `${where}: op '${found.name}' の入力スロット ${slot} は意味論 dtype '${dtype}' を実行できない（対応: ${
        accept.join(", ")
      }）`,
    );
  }
};

/**
 * ノードの意味論 dtype を検査して**出力 dtype を返す**。
 *
 * MUST: 出力 dtype は宣言を鵜呑みにせず契約から導き、宣言と突き合わせる。宣言をそのまま
 * 信じると、`cast` の attrs.to と values{} の宣言が食い違ったグラフが「宣言どおり」に
 * 通ってしまい、readback で別の TypedArray として読まれる沈黙誤値になる。
 * 出力の導出は 2 通りだけ: cast は attrs.to、それ以外は**スロット 0 の dtype を契約の写像に
 * 通す**（{@link ContractBase.outputDtypes} — 既定は恒等）。uniform 契約では加えて
 * スロット間の同型も要求する（混合型の演算は語彙に無い）。
 */
export const resolveNodeDtypes = (
  found: OpContract,
  node: IrNode,
  inputDtypes: readonly IrDtype[],
  declaredOutput: IrDtype,
  where: string,
): IrDtype => {
  inputDtypes.forEach((dtype, index) => {
    assertSlotDtype(found, index, dtype, `${where} の入力 '${node.ins[index]}'`);
  });
  let expected: IrDtype;
  if (found.kind === "cast") {
    expected = castTargetDtype(node.attrs, where);
  } else {
    // スロットごとに受理集合が違う op は、スロット間の同型を要求しない（それが perSlot の
    // 意味）。uniform 契約だけが混在を拒否する。
    if (found.slotDtypes.kind === "uniform") {
      const mixed = inputDtypes.findIndex((dtype) => dtype !== inputDtypes[0]);
      if (mixed >= 0) {
        throw new OpContractError(
          `${where}: op '${found.name}' の入力 dtype が混在（${inputDtypes.join(", ")}）`,
        );
      }
    }
    expected = outputDtypeOf(found, inputDtypes[0], where);
  }
  if (declaredOutput !== expected) {
    throw new OpContractError(
      `${where}: 出力 '${
        node.outs[0]
      }' の宣言 dtype '${declaredOutput}' が契約の '${expected}' と違う`,
    );
  }
  return expected;
};

/**
 * torch 準拠の右詰め broadcast。次元は「一致」または「片方が 1」のみ許す。
 * MUST: 結果を `max(a, b)` で決めない — 0 と 1 の組（`max` なら 1）が torch では 0 になる。
 */
export const broadcastShapes = (
  a: readonly number[],
  b: readonly number[],
  where: string,
): number[] => {
  const rank = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < rank; i += 1) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da === db || db === 1) {
      out.push(da);
    } else if (da === 1) {
      out.push(db);
    } else {
      throw new OpContractError(
        `${where}: shape [${a.join(",")}] と [${b.join(",")}] は右詰め broadcast できない`,
      );
    }
  }
  return out;
};

/**
 * 出力 shape が**入力から導けない** op のための追加入力。
 *
 * reshape / expand は「出力の宣言 shape が目標形」という契約（ADR 0011）なので、宣言を
 * 渡さずには計算できない。permute は attrs の並べ替え表が要る。
 */
export type ShapeContext = {
  /** 束縛解決済みの宣言 shape（reshape / expand で必須）。 */
  readonly declared?: readonly number[];
  /** ノードの attrs（permute / layer_norm / softmax / conv1d / sym_prefix_slice で必須）。 */
  readonly attrs?: Readonly<Record<string, unknown>>;
  /**
   * シンボル束縛（sym_prefix_slice で必須）。prefix 長は `coeff·sym+offset` で、入力 shape
   * からは導けない。
   *
   * MUST: 参照は `Object.hasOwn` のみ（`bindings[sym] !== undefined` は Object.prototype 由来の
   * `toString` 等が素通りして以後の算術が NaN 化する — 横断の不変条件）。
   */
  readonly bindings?: Readonly<Record<string, number>>;
};

/**
 * strided コピー族（permute / expand / sym_prefix_slice / masked_fill）の rank 上限。
 *
 * MUST: **契約層で見る**。カーネルの params は rank {@link STRIDED_RANK} 固定で、超過は
 * codegen 層（`stridedParams`）まで落ちて初めて CodegenError になる — 利用者から見ると
 * 「契約検査は通ったのに実行段で内部エラー」で、どの op のどの入力が悪いのか出ない。
 * ここで落とせば診断が op と入力の名前つきになり、CPU 参照（同じ契約表を通る）と GPU の
 * 受理範囲も揃う。
 */
const assertStridedRank = (rank: number, what: string, where: string): void => {
  if (rank < 1 || rank > STRIDED_RANK) {
    throw new OpContractError(
      `${where}: ${what} の rank ${rank} は 1..${STRIDED_RANK} の外（strided カーネルの上限）`,
    );
  }
};

const requireDeclared = (
  context: ShapeContext,
  found: OpContract,
  where: string,
): readonly number[] => {
  if (context.declared === undefined) {
    throw new OpContractError(
      `${where}: op '${found.name}' の出力 shape は宣言が目標形（ShapeContext.declared が要る）`,
    );
  }
  return context.declared;
};

/** 束縛解決済みの入力 shape から出力 shape を計算する。 */
export const computeOutputShape = (
  found: OpContract,
  inputShapes: readonly (readonly number[])[],
  where: string,
  context: ShapeContext = {},
): number[] => {
  assertArity(found, inputShapes.length, "入力 shape 数", where);
  switch (found.kind) {
    case "unary":
      // MUST: スカラ attr の値域と**キーを跨ぐ不変条件**（clamp の min <= max）をここで見る。
      // 全ノードが必ず通る共通経路はこの計算だけで、attrs スキーマはキー単位の検査しか
      // 表せない（src/runtime/plan.ts の planGraph が全ノードでここを呼ぶ）。
      scalarParamValues(found, context.attrs ?? {}, where);
      return [...inputShapes[0]];
    case "cast":
      return [...inputShapes[0]];
    case "binary":
      return broadcastShapes(inputShapes[0], inputShapes[1], `${where} (${found.name})`);
    case "where": {
      // torch と同じく 3 者を右詰め broadcast する（条件も値と同じ規則で広がる）。
      const [cond, a, b] = inputShapes;
      const label = `${where} (${found.name})`;
      return broadcastShapes(broadcastShapes(cond, a, label), b, label);
    }
    case "cumsum": {
      const shape = inputShapes[0];
      const dim = cumsumDim(context.attrs ?? {}, where);
      // MUST: 最終次元以外は受理しない（softmax と同じ理由 — 行カーネルは縮約軸が連続で
      // あることを前提にしていて、通せば黙って別の軸を畳む）。
      if (shape.length < 1 || dim !== shape.length - 1) {
        throw new OpContractError(
          `${where}: cumsum は最終次元のみ（attrs.dim=${dim} / 入力 [${shape.join(",")}]）`,
        );
      }
      // 長さ 0 の軸は素通し（前縁和の identity は 0 で、空行は空行のまま）。
      return [...shape];
    }
    case "matmul": {
      const [a, b] = inputShapes;
      if (a.length !== 2 || b.length !== 2) {
        throw new OpContractError(
          `${where}: matmul は rank-2 × rank-2 のみ（[${a.join(",")}] × [${b.join(",")}]）`,
        );
      }
      if (a[1] !== b[0]) {
        throw new OpContractError(
          `${where}: matmul の縮約次元が不一致 [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      return [a[0], b[1]];
    }
    case "bmm": {
      const [a, b] = inputShapes;
      // MUST: rank-2 を通さない（matmul の担当）。兼用にすると「バッチ軸を落とした形」が
      // 同じ op 名で通り、B の取り違えが shape 検査を素通りする。
      if (a.length !== 3 || b.length !== 3) {
        throw new OpContractError(
          `${where}: bmm は rank-3 × rank-3 のみ（rank-2 は matmul）: [${a.join(",")}] × [${
            b.join(",")
          }]`,
        );
      }
      if (a[0] !== b[0]) {
        throw new OpContractError(
          `${where}: bmm のバッチ次元が不一致 [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      if (a[2] !== b[1]) {
        throw new OpContractError(
          `${where}: bmm の縮約次元が不一致 [${a.join(",")}] × [${b.join(",")}]`,
        );
      }
      return [a[0], a[1], b[2]];
    }
    case "gather": {
      const [src, index] = inputShapes;
      // 契約は「最終次元固定」— 先行次元は src と index で完全一致し、最終次元だけが自由。
      // torch の一般 gather（他軸の長さが src 以下でよい）より狭いが、実測形はこれで足りる。
      if (src.length === 0 || index.length !== src.length) {
        throw new OpContractError(
          `${where}: gather は src と index が同じ rank（1 以上）: [${src.join(",")}] / [${
            index.join(",")
          }]`,
        );
      }
      const mismatch = src.findIndex((dim, d) => d < src.length - 1 && dim !== index[d]);
      if (mismatch >= 0) {
        throw new OpContractError(
          `${where}: gather の先行次元 ${mismatch} が不一致 [${src.join(",")}] / [${
            index.join(",")
          }]`,
        );
      }
      // 出力は index と同形（値は src から引く）。添字の値域は実行時データ依存なので
      // shape 契約では見ない（方針は src/kernels/gather.ts と reference/ops.ts）。
      return [...index];
    }
    case "rowReduce": {
      const shape = inputShapes[0];
      if (shape.length === 0) {
        throw new OpContractError(
          `${where}: reduce の入力は rank 1 以上（スカラは縮約できない）`,
        );
      }
      const dim = reduceDim(context.attrs ?? {}, where);
      // MUST: 負値・rank 外は fail loudly（負の軸表記の正規化はエクスポータ境界の責務で、
      // ここで `% rank` を補うと「宣言と実 rank の食い違い」を黙って別の軸へ吸収してしまう）。
      if (dim >= shape.length) {
        throw new OpContractError(
          `${where}: op '${found.name}' の attrs.dim=${dim} が rank ${shape.length} の範囲外`,
        );
      }
      // 空軸の amax/amin は identity が定義できない（torch も同様に拒否する）。sum は 0。
      if (shape[dim] === 0 && found.name !== "sum") {
        throw new OpContractError(`${where}: op '${found.name}' は長さ 0 の軸を縮約できない`);
      }
      return [...shape.slice(0, dim), ...shape.slice(dim + 1)];
    }
    case "reshape": {
      const target = requireDeclared(context, found, where);
      const source = inputShapes[0];
      // 契約は要素数一致だけ（要素順は変えない）。ここを緩めると別名化した実バッファの
      // 大きさと宣言 shape が食い違い、readback が範囲外まで読む。
      if (numel(source) !== numel(target)) {
        throw new OpContractError(
          `${where}: reshape の要素数が合わない [${source.join(",")}] → [${target.join(",")}]`,
        );
      }
      return [...target];
    }
    case "expand": {
      const target = requireDeclared(context, found, where);
      const source = inputShapes[0];
      assertStridedRank(source.length, "expand の入力", where);
      assertStridedRank(target.length, "expand の出力", where);
      if (target.length < source.length) {
        throw new OpContractError(
          `${where}: expand は rank を下げられない [${source.join(",")}] → [${target.join(",")}]`,
        );
      }
      // 右詰めで、入力の各次元は「目標と一致」か「長さ 1（stride 0 で複製）」のみ。
      const offset = target.length - source.length;
      source.forEach((extent, index) => {
        if (extent !== 1 && extent !== target[offset + index]) {
          throw new OpContractError(
            `${where}: expand は長さ 1 でない次元 ${index}（${extent}）を ${
              target[offset + index]
            } に拡張できない`,
          );
        }
      });
      return [...target];
    }
    case "slice": {
      const source = inputShapes[0];
      // 実行は strided 読みコピー族の流用（ADR 0014）なので rank 上限も同じ。
      assertStridedRank(source.length, "slice の入力", where);
      const { dim, start, end } = sliceAttrs(context.attrs ?? {}, where);
      if (dim >= source.length) {
        throw new OpContractError(
          `${where}: slice の dim ${dim} が入力 rank ${source.length} の外`,
        );
      }
      // MUST: 範囲外の切り出しを通さない。GPU では例外なしに隣の値（別の行・別のバッファ）を
      // 読む形になり、shape 検査だけが「宣言どおり」で素通りする。
      if (end > source[dim]) {
        throw new OpContractError(
          `${where}: slice の end ${end} が軸 ${dim} の長さ ${source[dim]} を超える`,
        );
      }
      // MUST: キーを跨ぐ不変条件（clamp の min <= max と同じ分担）— attrs スキーマは
      // キー単位の検査しか表せない。逆転を許すと長さが負になり、要素数だけが 0 で通る。
      if (start > end) {
        throw new OpContractError(`${where}: slice の start ${start} が end ${end} を超える`);
      }
      const out = [...source];
      out[dim] = end - start;
      return out;
    }
    case "cat": {
      const dim = catDim(context.attrs ?? {}, where);
      const first = inputShapes[0];
      // 実行は strided 書きコピー族（ADR 0014）— 出力側の stride を params に載せるので、
      // rank 上限は出力（= 入力と同 rank）に効く。
      assertStridedRank(first.length, "cat の入力", where);
      if (dim >= first.length) {
        throw new OpContractError(`${where}: cat の dim ${dim} が入力 rank ${first.length} の外`);
      }
      let total = 0;
      inputShapes.forEach((shape, index) => {
        if (shape.length !== first.length) {
          throw new OpContractError(
            `${where}: cat の入力 ${index} の rank ${shape.length} が入力 0 の ${first.length} と違う`,
          );
        }
        // MUST: 連結軸**以外**は全一致を要求する（torch と同じ）。緩めると出力の一部が
        // どの入力にも書かれないまま残り、full-write 不変条件が破れる。
        shape.forEach((extent, axis) => {
          if (axis !== dim && extent !== first[axis]) {
            throw new OpContractError(
              `${where}: cat の入力 ${index} [${shape.join(",")}] が入力 0 [${
                first.join(",")
              }] と軸 ${axis} で違う（連結軸は ${dim}）`,
            );
          }
        });
        total += shape[dim];
      });
      const out = [...first];
      // 出力の軸長 = 入力の軸長の総和。この規則そのものが「全入力で出力全域を覆う」
      // （full-write — ADR 0014）の担保で、executor 側は書き出し位置の総和をこれと突き合わせる。
      out[dim] = total;
      return out;
    }
    case "pad": {
      const source = inputShapes[0];
      if (source.length < 1) {
        throw new OpContractError(`${where}: pad の入力は rank 1 以上（最終次元が要る）`);
      }
      const { left, right } = padAttrs(context.attrs ?? {}, where);
      const out = [...source];
      out[out.length - 1] = source[source.length - 1] + left + right;
      return out;
    }
    case "flip": {
      const source = inputShapes[0];
      const dim = flipDim(context.attrs ?? {}, where);
      if (source.length < 1 || dim >= source.length) {
        throw new OpContractError(
          `${where}: flip の dim ${dim} が入力 rank ${source.length} の外`,
        );
      }
      // 反転は shape を変えない（恒等 shape 規則）。
      return [...source];
    }
    case "symPrefixSlice": {
      const source = inputShapes[0];
      // 出力 rank は入力と同じ（各軸の先頭を切り出すだけ）なので入力側だけ見れば足りる。
      assertStridedRank(source.length, "sym_prefix_slice の入力", where);
      const { sym, slices } = symPrefixSliceAttrs(context.attrs ?? {}, where);
      const bindings = context.bindings;
      if (bindings === undefined || !Object.hasOwn(bindings, sym)) {
        throw new OpContractError(
          `${where}: sym_prefix_slice の sym '${sym}' が束縛されていない（ShapeContext.bindings）`,
        );
      }
      const bound = bindings[sym];
      const out = [...source];
      for (const slice of slices) {
        if (slice.dim >= source.length) {
          throw new OpContractError(
            `${where}: sym_prefix_slice の dim ${slice.dim} が入力 rank ${source.length} の外`,
          );
        }
        const length = slice.coeff * bound + slice.offset;
        // MUST: 定数側（Tmax 形）を超える prefix を許さない。超えた分は定数バッファの
        // 範囲外読み出しになり、GPU では例外なしに隣の値が出る。
        if (length > source[slice.dim]) {
          throw new OpContractError(
            `${where}: sym_prefix_slice の prefix 長 ${slice.coeff}·${sym}+${slice.offset}=${length} が定数次元 ${
              source[slice.dim]
            } を超える（Tmax 超過）`,
          );
        }
        out[slice.dim] = length;
      }
      return out;
    }
    case "linear": {
      const [x, weight, bias] = inputShapes;
      if (x.length < 1 || weight.length !== 2 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: linear は x[…,in] × W[out,in] + b[out]（rank ≥ 1 / 2 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [out, inFeatures] = weight;
      if (x[x.length - 1] !== inFeatures) {
        throw new OpContractError(
          `${where}: linear の入力特徴数が不一致 [${x.join(",")}] × [${weight.join(",")}]`,
        );
      }
      if (bias[0] !== out) {
        throw new OpContractError(
          `${where}: linear の bias 長 ${bias[0]} が出力特徴数 ${out} と違う`,
        );
      }
      return [...x.slice(0, -1), out];
    }
    case "layerNorm": {
      const [x, weight, bias] = inputShapes;
      const { normalizedShape } = layerNormAttrs(context.attrs ?? {}, where);
      // 契約は「最終次元のみ」（attrs 検査済み）。ここでは実 shape の末尾との一致を見る。
      if (x.length < 1 || x[x.length - 1] !== normalizedShape[0]) {
        throw new OpContractError(
          `${where}: layer_norm の normalized_shape [${normalizedShape.join(",")}] が入力 [${
            x.join(",")
          }] の最終次元と違う`,
        );
      }
      for (const [name, shape] of [["weight", weight], ["bias", bias]] as const) {
        if (shape.length !== 1 || shape[0] !== normalizedShape[0]) {
          throw new OpContractError(
            `${where}: layer_norm の ${name} [${shape.join(",")}] が normalized_shape [${
              normalizedShape.join(",")
            }] と違う`,
          );
        }
      }
      return [...x];
    }
    case "rmsNorm": {
      const [x, weight] = inputShapes;
      // MUST: eps はここでも引く（attrs スキーマを通らない経路 — CPU 参照の直呼び — でも
      // 値域が効くようにする。unary の scalarParamValues と同じ役割）。
      rmsNormEps(context.attrs ?? {}, where);
      if (x.length < 1) {
        throw new OpContractError(`${where}: rms_norm の入力は rank 1 以上（最終次元が要る）`);
      }
      const dim = x[x.length - 1];
      // MUST: 長さ 0 の軸は縮約できない（二乗和 0 / 要素数 0 で mean が 0/0 = NaN になる —
      // softmax と同じ絞り方）。
      if (dim === 0) {
        throw new OpContractError(`${where}: rms_norm は長さ 0 の軸を正規化できない`);
      }
      // 正規化長の正本は **weight の長さ**（attrs に normalized_shape の欄を作らない — ADR 0017）。
      if (weight.length !== 1 || weight[0] !== dim) {
        throw new OpContractError(
          `${where}: rms_norm の weight [${weight.join(",")}] が入力 [${
            x.join(",")
          }] の最終次元長 ${dim} の rank1 でない`,
        );
      }
      return [...x];
    }
    case "conv2d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv2dAttrs(context.attrs ?? {}, where);
      if (x.length !== 4 || weight.length !== 4 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv2d は x[B,Cin,H,W] / W[Cout,Cin/groups,Kh,Kw] / b[Cout]（rank 4 / 4 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn] = x;
      const [channelsOut, weightIn] = weight;
      // MUST: conv1d と同じ規律 — 割り切れない形はカーネルの `Cin/groups` が切り捨てになり、
      // 読む入力チャネル帯が黙ってずれる。
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv2d の groups ${groups} が Cin ${channelsIn} / Cout ${channelsOut} を割り切らない`,
        );
      }
      // MUST: 重みは **[Cout, Cin/groups, Kh, Kw]**。要素数が合う取り違え（[Cin/groups, Cout,
      // Kh, Kw] や Kh/Kw の入れ替え）は shape 検査を素通りしうるので、テストは
      // Cin ≠ Cout・Kh ≠ Kw の非対称形で固定する（conv_transpose1d の教訓 — ADR 0017）。
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv2d の重みは [Cout, Cin/groups, Kh, Kw]（Cin/groups = ${
            channelsIn / groups
          }）のはずが [${weight.join(",")}]（x は [${x.join(",")}] / groups ${groups}）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv2d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // 空間軸は H / W で独立に同じ一般形を適用する（axis は診断の主語）。
      const spatial = (axis: 0 | 1, name: string): number => {
        const length = x[2 + axis];
        const kernel = weight[2 + axis];
        const span = length + 2 * padding[axis] - dilation[axis] * (kernel - 1) - 1;
        if (span < 0) {
          throw new OpContractError(
            `${where}: conv2d の入力 ${name} ${length}（padding ${padding[axis]}）が dilation ${
              dilation[axis]
            } 込みのカーネル張り ${dilation[axis] * (kernel - 1) + 1} に足りない`,
          );
        }
        return Math.floor(span / stride[axis]) + 1;
      };
      return [x[0], channelsOut, spatial(0, "H"), spatial(1, "W")];
    }
    case "softmax": {
      const shape = inputShapes[0];
      const dim = softmaxDim(context.attrs ?? {}, where);
      // MUST: 一般 dim を「そのうち実装する」として受理しない。最終次元以外は行カーネルの
      // 前提（縮約軸が連続）が崩れ、通せば黙って別の軸を畳む。
      if (shape.length < 1 || dim !== shape.length - 1) {
        throw new OpContractError(
          `${where}: softmax は最終次元のみ（attrs.dim=${dim} / 入力 [${shape.join(",")}]）`,
        );
      }
      if (shape[shape.length - 1] === 0) {
        // 空軸の softmax は amax の identity が定義できない（行 reduce と同じ理由）。
        throw new OpContractError(`${where}: softmax は長さ 0 の軸を縮約できない`);
      }
      return [...shape];
    }
    case "attention": {
      const [q, k, v, mask] = inputShapes;
      // MUST: scale はここでも引く（attrs スキーマを通らない CPU 参照の直呼びでも値域を効かせる
      // ため。rms_norm の eps / unary の scalarParamValues と同じ役割）。
      attentionScale(context.attrs ?? {}, where);
      const show = `[${q.join(",")}] / [${k.join(",")}] / [${v.join(",")}]`;
      if (q.length !== 4 || k.length !== 4 || v.length !== 4) {
        throw new OpContractError(
          `${where}: attention は q[B,H,M,D] / k[B,H,N,D] / v[B,H,N,D] の rank-4 のみ: ${show}`,
        );
      }
      // MUST: B と H を**別々に**突き合わせる（積だけを見ると B/H の取り違えが素通りする —
      // カーネルは B·H を 1 本の軸に畳むので、値にも出ない形が作れる）。
      for (const axis of [0, 1] as const) {
        if (q[axis] !== k[axis] || q[axis] !== v[axis]) {
          throw new OpContractError(
            `${where}: attention の軸 ${axis}（${axis === 0 ? "B" : "H"}）が不一致 ${show}`,
          );
        }
      }
      // MUST: D は 3 者とも同じ（v 側だけ別の長さを許すと、取り違えが要素数で捕まらない）。
      if (q[3] !== k[3] || q[3] !== v[3]) {
        throw new OpContractError(`${where}: attention の D（軸 3）が不一致 ${show}`);
      }
      if (k[2] !== v[2]) {
        throw new OpContractError(`${where}: attention の N（k / v の軸 2）が不一致 ${show}`);
      }
      // 空軸の softmax は amax の identity が定義できない（softmax / 行 reduce と同じ理由）。
      if (k[2] === 0) {
        throw new OpContractError(`${where}: attention は長さ 0 の N を縮約できない ${show}`);
      }
      if (mask !== undefined) {
        // MUST: mask は **[1,1,M,N] ちょうど**。B·H へ broadcast する加算項なので、
        // `[B,1,M,N]` / `[1,H,M,N]` のような「一部の軸だけ実体を持つ」形を通すと、
        // カーネル（B·H を 1 軸に畳んで先頭バッチの mask を全バッチへ配る）が黙って
        // 別のバッチの mask を適用する。広げるなら添字算術とセットで契約を改版する。
        const shown = `${show} + mask [${mask.join(",")}]`;
        if (mask.length !== 4 || mask[0] !== 1 || mask[1] !== 1) {
          throw new OpContractError(
            `${where}: attention の mask は [1,1,M,N] の rank-4 のみ（B / H は broadcast 固定）: ${shown}`,
          );
        }
        if (mask[2] !== q[2] || mask[3] !== k[2]) {
          throw new OpContractError(
            `${where}: attention の mask の M / N が q / k と不一致 ${shown}`,
          );
        }
      }
      return [...q];
    }
    case "embedding": {
      const [weight, index] = inputShapes;
      if (weight.length !== 2) {
        throw new OpContractError(
          `${where}: embedding の weight は rank-2 [V,H]: [${weight.join(",")}]`,
        );
      }
      if (index.length < 1) {
        throw new OpContractError(
          `${where}: embedding の index は rank 1 以上（スカラ添字は無い）`,
        );
      }
      // 添字の値域 0 <= index < V は実行時データ依存なので shape 契約では見ない
      // （範囲外の扱いは src/kernels/embedding.ts の裁定 — GPU は NaN 汚染 / CPU 参照は throw）。
      return [...index, weight[1]];
    }
    case "maskedFill": {
      const [x, mask] = inputShapes;
      // 出力は x と同形。mask も右詰め broadcast の stride を組むので rank 1 以上が要る
      // （rank 0 の mask は契約を素通りして codegen 層で落ちる形になっていた）。
      assertStridedRank(x.length, "masked_fill の x", where);
      assertStridedRank(mask.length, "masked_fill の mask", where);
      // MUST: 出力は**常に x と同形**（mask 側は右詰め broadcast で読むだけ）。broadcastShapes を
      // そのまま使うと mask が x を広げる形（mask [4] × x [1]）まで通り、埋め値が本来無い要素へ
      // 漏れる。ここは「mask が x に収まる」ことだけを見る非対称な検査。
      if (mask.length > x.length) {
        throw new OpContractError(
          `${where}: masked_fill の mask rank ${mask.length} が x rank ${x.length} を超える（mask は右詰め broadcast のみ）`,
        );
      }
      const offset = x.length - mask.length;
      mask.forEach((extent, index) => {
        if (extent !== 1 && extent !== x[offset + index]) {
          throw new OpContractError(
            `${where}: masked_fill の mask [${mask.join(",")}] が x [${
              x.join(",")
            }] へ右詰め broadcast できない（次元 ${index}）`,
          );
        }
      });
      return [...x];
    }
    case "conv1d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding, dilation, groups } = conv1dAttrs(context.attrs ?? {}, where);
      if (x.length !== 3 || weight.length !== 3 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv1d は x[B,Cin,L] / W[Cout,Cin/groups,K] / b[Cout]（rank 3 / 3 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn, length] = x;
      const [channelsOut, weightIn, kernel] = weight;
      // MUST: グループ分割は両側で割り切れることが契約（depthwise は groups = Cin = Cout）。
      // 割り切れない形を通すとカーネルの `Cin/groups` が切り捨てになり、読む入力チャネルが
      // 黙ってずれる。
      if (channelsIn % groups !== 0 || channelsOut % groups !== 0) {
        throw new OpContractError(
          `${where}: conv1d の groups ${groups} が Cin ${channelsIn} / Cout ${channelsOut} を割り切らない`,
        );
      }
      if (weightIn !== channelsIn / groups) {
        throw new OpContractError(
          `${where}: conv1d の重みは [Cout, Cin/groups, K]（Cin/groups = ${
            channelsIn / groups
          }）のはずが [${weight.join(",")}]（x は [${x.join(",")}] / groups ${groups}）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv1d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // dilation の一般形。K=1 でも d·(K−1) = 0 なので従来式と一致する。
      const span = length + 2 * padding - dilation * (kernel - 1) - 1;
      if (span < 0) {
        throw new OpContractError(
          `${where}: conv1d の入力長 ${length}（padding ${padding}）が dilation ${dilation} 込みのカーネル張り ${
            dilation * (kernel - 1) + 1
          } に足りない`,
        );
      }
      return [x[0], channelsOut, Math.floor(span / stride) + 1];
    }
    case "convTranspose1d": {
      const [x, weight, bias] = inputShapes;
      const { stride, padding } = convTranspose1dAttrs(context.attrs ?? {}, where);
      if (x.length !== 3 || weight.length !== 3 || bias.length !== 1) {
        throw new OpContractError(
          `${where}: conv_transpose1d は x[B,Cin,L] / W[Cin,Cout,K] / b[Cout]（rank 3 / 3 / 1）: [${
            x.join(",")
          }] / [${weight.join(",")}] / [${bias.join(",")}]`,
        );
      }
      const [, channelsIn, length] = x;
      const [weightIn, channelsOut, kernel] = weight;
      // MUST: 重みは conv1d と**転置**の [Cin, Cout, K]。取り違えても要素数が合う形
      // （Cin == Cout）が作れるので、テストは非対称チャネル数で固定する（ADR 0015）。
      if (weightIn !== channelsIn) {
        throw new OpContractError(
          `${where}: conv_transpose1d の重みは [Cin, Cout, K]（Cin = ${channelsIn}）のはずが [${
            weight.join(",")
          }]（x は [${x.join(",")}]）`,
        );
      }
      if (bias[0] !== channelsOut) {
        throw new OpContractError(
          `${where}: conv_transpose1d の bias 長 ${bias[0]} が出力チャネル ${channelsOut} と違う`,
        );
      }
      // MUST: 受理するのは出力長がちょうど L·stride になる形だけ（`2p == K − s`）。一般形
      // `(L−1)·s − 2p + K` は記号長 L の一次式にはなるが、実測（dec の ups 5 本）が全て
      // この形なので、広げるのは需要が出てから — 黙って一般形を通さない（ADR 0015）。
      if (2 * padding !== kernel - stride) {
        throw new OpContractError(
          `${where}: conv_transpose1d は 2·padding == K − stride の形のみ受理（K ${kernel} / stride ${stride} / padding ${padding} — 出力長が L·stride にならない）`,
        );
      }
      return [x[0], channelsOut, length * stride];
    }
    case "permute": {
      const source = inputShapes[0];
      // 出力 rank は入力と同じ（並べ替えるだけ）なので入力側だけ見れば足りる。
      assertStridedRank(source.length, "permute の入力", where);
      const dims = permuteDims(context.attrs ?? {}, where);
      if (dims.length !== source.length) {
        throw new OpContractError(
          `${where}: permute の dims [${dims.join(",")}] が入力 rank ${source.length} と違う`,
        );
      }
      return dims.map((dim) => {
        if (dim >= source.length) {
          throw new OpContractError(
            `${where}: permute の dims に入力 rank ${source.length} 外の軸 ${dim} がある`,
          );
        }
        return source[dim];
      });
    }
  }
};

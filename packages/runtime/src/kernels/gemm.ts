/**
 * GEMM 族（matmul / bmm / linear + 融合 attention の QK / PV + conv2d の implicit GEMM —
 * ADR 0022 / 0023 / 0024）が共有する **レジスタブロッキング + vec4** の骨格。
 *
 * 1 スレッドが {@link GemmGeometry} の `regM`×`regN` の出力を持ち（`acc{行}_{列 quad}` の
 * 名前付き変数へ codegen 時に静的展開する — {@link gemmAccumulatorInit}）、共有 B タイルは
 * **列方向を vec4 に束ねた** `[k][列 quad]`。内側ループは共有ロード `regM + regN/4` 回
 * （A のスカラ `regM` + B の vec4 `regN/4`）で `regM · regN` MAC を回す — 共有帯域あたりの
 * 演算密度は `regM·regN / (regM + regN/4)` で、1 スレッド 1 出力の 16×16 タイル
 * （2 ロード 1 MAC = 0.5）に対して幾何を上げるほど大きくなる。
 *
 * MUST: **具体値をここに書かない**（幾何・doc・表の 3 箇所に同じ事実が散るとドリフトする）。
 * 既定は {@link defaultGemmGeometry}・出力タイル辺は {@link gemmTileM} / {@link gemmTileN}・
 * K タイル幅は {@link GEMM_TILE_K}（幾何ではなく数値契約）・共有メモリのバイト数は skeleton が
 * 生成時に計算して WGSL ヘッダへ書く（生成物とスナップショット tests/fixtures/wgsl/ が正本）。
 * 幾何の候補表と実測の出どころは src/kernels/gemm-geometry.ts のモジュール doc。
 *
 * MUST: タイル幾何が決めてよいのは**どのスレッドがどの出力を担当するか**だけで、
 * 既定は {@link defaultGemmGeometry} の 1 箇所（src/kernels/gemm-geometry.ts に幾何の
 * 算術・門・キー断片がまとまっている）。共有タイルの容量と 1 workgroup = 1 出力タイルの
 * 対応は幾何によらず不変なので、dispatch 側（executor の `tiledWorkgroups`）は無関係。
 * MUST: dispatch の辺は**幾何から導く**（`tiledWorkgroups(n, gemmTileN(geometry), …)`）。
 * 辺の値を定数で持ち回ると幾何と食い違いうる値が 2 つになり、`ceil(dim / 定数)` が
 * `ceil(dim / 実タイル辺)` を下回った瞬間に**タイルが欠落して沈黙誤値**になる。
 *
 * MUST: 全 op は**この 1 本の骨格を共有する**。内積ループの正本が 1 箇所にあることが
 * 「縮約順序が全 op で同一」という不変条件を機械的に守る唯一の手段で、括り出しを禁じると
 * 同じループを 5 回書き写すことになる。旧 16×16 実装が持っていた「共通化するな」MUST は
 * 「既存の生成バイト列とスナップショットを動かすな」が目的で、3 本とも WGSL を総取り替え
 * してキーを v2 へ改版した時点で保護対象が消えている。融合 attention（ADR 0023）が
 * **分解経路とビット同一**でいられるのも、conv2d の implicit GEMM（ADR 0024）が
 * **直接畳み込みとビット同一**でいられるのも、どちらもここの内積ループをそのまま使うため。
 * MUST: 断片を足すときは**既存 3 op の生成バイト列を 1 バイトも動かさない**（省略時の既定が
 * 従来の文字列を返す形にする）。スナップショット（tests/fixtures/wgsl/）が検出器。
 * MUST: K タイル幅は 16 のまま・外側 `t` も内側 `kk` も昇順。1 出力要素あたりの加算順序が
 * 旧 16×16 カーネルと完全に一致する（既存の数値契約を動かさないための土台）。
 * MUST: m タイルの行数（{@link GEMM_MTILE_SMALL} の 32 行変種 — conv2d のみ）を変えても
 * **1 出力要素の数値経路は変えない**。タイル形が決めるのは「どの workgroup がどの出力を
 * 担当するか」だけで、K 縮約の順序も丸めの並びも `acc` の初期値も共通の骨格が持つ。
 * MUST: 1 workgroup = 1 出力タイルで全域を覆う。grid-stride で縮退できないため dispatch 数の
 * 上限超過は fail loudly（{@link tiledWorkgroups}）— 縮退させるとタイルが欠落し、例外なしに
 * 出力の一部が未書き込み（プール再利用なら前の値）のまま残る。bmm のバッチ軸（z）も同じ。
 *
 * **v4 経路**（{@link gemmUsesVec4}: `k % 4 == 0 && n % 4 == 0`）は入出力を `vec4<f32>` で
 * 読み書きする。条件を満たさない形は**同じレジスタ構造のままスカラ読み書きへ落ちる**変種で、
 * v4 フラグはパイプラインキーに載る。キーが形状の関数になるが決定性は崩れない（形状 →
 * 1 ビットの写像なので「同一キー ⇔ 同一 WGSL バイト列」は保たれる — `elementwiseKey` の
 * `r${rank}` と同じ語彙）。
 */

import { CodegenError } from "../codegen/errors.ts";
import {
  assertGemmGeometry,
  defaultGemmGeometry,
  GEMM_K_QUADS,
  GEMM_QUAD,
  GEMM_TILE_K,
  gemmAccumulatorInit,
  gemmColumnQuads,
  gemmColumnSlots,
  type GemmGeometry,
  gemmGeometryForRows,
  gemmGeometryKeyPart,
  gemmGeometryNote,
  gemmQuadFillStride,
  gemmQuadSlots,
  gemmQuadsPerThread,
  gemmRowFillStride,
  gemmRowSlots,
  gemmThreads,
  gemmTileM,
  gemmTileN,
} from "./gemm-geometry.ts";
import {
  assertScoreStorageSupported,
  scoreArrayType,
  scoreNote,
  scoreQuadLoaderWgsl,
  scoreReadQuad,
  type ScoreStorage,
  scoreStoreQuad,
  scoreStoreWgsl,
} from "./score-storage.ts";
import {
  WEIGHT_SCALE_VAR,
  weightArrayType,
  weightLoaderWgsl,
  weightNote,
  weightRead,
  weightRead4,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

/**
 * conv2d の implicit GEMM だけが持つ **32 行 m タイル**（n タイルは幾何のまま — ADR 0024 隣接）。
 *
 * 動機は M = Cout が 64 の倍数でない層のタイル量子化の無駄（census の Cout=96 は
 * `ceil(96/64)·64/96 = 1.33×`）。**1 スレッドの出力は幾何のまま**で workgroup の y 辺だけを
 * 半分に落とすので、内積ループの演算密度は 64 行版と変わらない（1 スレッドの出力を削って
 * workgroup を保つ形は密度が落ちる）。
 * MUST: 1 出力要素あたりの K 縮約順・丸め列は 64 行版と厳密一致（タイル形は「どの workgroup が
 * どの出力を担当するか」だけを変える）。
 */
export const GEMM_MTILE_SMALL = 32;

/**
 * i8 変種の scale 束縛（出力束縛の次の番号 — executor の bind entries と対）。
 *
 * linear と conv2d の implicit GEMM は束縛配置が同じ（0 dims / 1 活性 / 2 重み / 3 bias /
 * 4 出力）なので、番号は 1 本で足りる。
 */
export const LINEAR_SCALE_BINDING = 5;

/**
 * 融合 attention ①QK の**加算 mask** の束縛（S の次の番号 — executor の bind entries と対）。
 *
 * MUST: mask 変種でだけ束縛する。無い変種で番号を予約すると、束縛されていない binding を
 * 宣言したパイプラインとして validation で落ちる。
 */
export const ATTENTION_QK_MASK_BINDING = 4;

/**
 * m タイルの行数から幾何を解決する（既定幾何の **wgY だけ**を `mTile / regM` へ差し替える）。
 *
 * conv2d の implicit GEMM が持つヒューリスティック（`conv2dIgemmMTile`）は「行数」で答えを
 * 出すので、その 1 値を幾何へ吸収する唯一の点。**生成（{@link gemmWgsl}）もキーも dispatch も
 * ここを通す** — 通さずに mTile を直に読む経路を残すと、キーの幾何と生成物の幾何が食い違う。
 * 割り切れない mTile は {@link assertGemmGeometry} の「wgY は正整数」で落ちる。
 */
export const gemmMTileGeometry = (mTile: number): GemmGeometry => {
  const geometry = defaultGemmGeometry();
  return { ...geometry, wgY: mTile / geometry.regM };
};

/**
 * v4 経路（vec4 の読み書き）を使えるかどうかの判定。
 *
 * dense 側は a が `[m,k]`（k%4）・b が `[k,n]`（n%4）・c が `[m,n]`（n%4）、linear 側は
 * x が `[m,k]`（k%4）・out が `[m,n]`（n%4）・W が `[n,k]`（k%4 で行頭が quad 境界）で、
 * **両方**必要になる。
 *
 * MUST: 判定はここ 1 箇所。`k%4` だけ見て `n%4` を見落とすと、v4 の書き出しが quad 境界を
 * 越えて隣接行を潰す（例外なしの誤値）。
 */
export const gemmUsesVec4 = (k: number, n: number): boolean => k % 4 === 0 && n % 4 === 0;

/**
 * 行数 M から幾何を解決する（matmul / bmm / linear の 3 経路）。**キーと生成の両方がここを通る**
 * ので、片方だけ別の幾何になることが構造的に起こらない。
 *
 * `rows` 省略は「行数を持たない呼び出し」= 融合 attention（幾何は Anima の実測選定で固定 —
 * {@link "./gemm-geometry.ts"} `gemmGeometryForRows` の MUST）と、既定変種だけを見る
 * スナップショット / キー検査。
 */
const rowsGeometry = (rows: number | undefined): GemmGeometry =>
  rows === undefined ? defaultGemmGeometry() : gemmGeometryForRows(rows);

/**
 * パイプラインキーのタイル判別子。v4 フラグは形状から導いた 1 ビット。
 *
 * MUST: タイル辺だけでなく**幾何そのもの**を載せる（{@link gemmGeometryKeyPart}）— 64×64 は
 * 複数の `regM×regN wgX` から組めるので、辺だけのキーは別の生成物へ衝突する。
 * MUST: `rows` は生成（{@link gemmWgsl}）へ渡すものと**同じ値**にする。片方だけ渡し忘れると
 * キーと生成物の幾何が食い違い、キャッシュに載った別幾何の WGSL が dispatch 数と噛み合わずに
 * 出力タイルが欠落する（例外の出ない誤値）。
 */
export const gemmKeyPart = (v4: boolean, rows?: number): string => {
  const geometry = rowsGeometry(rows);
  return `reg${gemmTileM(geometry)}x${gemmTileN(geometry)}${gemmGeometryKeyPart(geometry)}${
    v4 ? "v4" : ""
  }`;
};

/**
 * 内積の**計算**変種（ADR 0028）。`"f16"` は共有タイルを f16 に落とす形で、
 * **重み格納**の f16（ADR 0018 の `wf16`）とは完全に別の軸。
 *
 * 丸め列（ここが数値契約の全て。TS 参照はこの並びを写す）:
 *
 * 1. 共有タイルへ書く直前に `f16()` で 1 度だけ丸める（A / B とも）。`scale` の乗算や
 *    `exp(S−m)·inv` の評価は**丸める前**に f32 で済ませる。
 * 2. 内側ループはタイルから読んだ値を `f32()` へ広げてから MAC する（f16 → f32 は厳密なので
 *    丸めではない）。**MAC ごとに f16↔f32 変換を挟まない** — 変換はレジスタロード 1 回ぶんで、
 *    `regM · regN` MAC あたり `regM + regN/4` 回（B の vec4 + A のスカラ）。
 * 3. 累積は f32（`acc{行}_{列 quad}` はどれも vec4<f32>）。k が大きい経路で f16 累積にすると
 *    桁落ちが値に出る。
 * 4. 出力は f32（唯一の例外が融合 attention の S — ①が f16 で書き②③が f16 で読む形にして
 *    transient を半減させる。この丸めは {@link store} の書き出し 1 箇所）。
 *
 * したがって **f16 変種の値は「入力を f16 に丸めた f32 変種」と厳密に一致する**（縮約順序も
 * 丸め列も f32 変種と同じ）。この性質が単体テストのオラクル（f32 カーネル × 事前丸め入力）を
 * 成立させている。
 */
export type GemmCompute = "f32" | "f16";

/**
 * パイプラインキーの計算変種判別子。
 *
 * MUST: f32 は空文字（既存キーはバイト単位で不変）。MUST: 語は格納 f16 の `:wf16` と**別**に
 * する — 同じ語にすると「重みが f16 で常駐している」と「内積を f16 タイルで回している」が
 * 診断で区別できなくなる（両方同時に起こりうる組み合わせ）。
 */
export const gemmComputeKeyPart = (compute: GemmCompute): string => compute === "f16" ? ":c16" : "";

/**
 * 生成入力。`weight` は重みスロットを持つ op（linear / conv2d）だけが取る（matmul / bmm /
 * attention は活性が f32 固定）。`compute` は f16 計算変種（ADR 0028）を結んだ op
 * （linear / attention_qk / attention_pv）だけが取る — 省略時は `"f32"` で、生成物は
 * 1 バイトも動かない。
 *
 * `attention_qk` / `attention_pv` は融合 attention の ① と ③（ADR 0023）。②（行統計）は
 * GEMM 骨格ではないので src/kernels/attention.ts が別に持つ。`conv2d` は implicit GEMM
 * （ADR 0024）で、**groups == 1 専用**（groups > 1 は直接カーネルが受ける）。
 */
export type GemmSpec =
  | {
    readonly op: "matmul" | "bmm";
    readonly v4: boolean;
    /** 出力の行数 M（幾何のバケット — 省略時は既定幾何）。 */
    readonly rows?: number;
  }
  | {
    readonly op: "attention_qk" | "attention_pv";
    readonly v4: boolean;
    readonly compute?: GemmCompute;
    /** S の格納形（ADR 0023 の ①が書き ③が読むバッファ — 省略時 `"f32"`）。 */
    readonly score?: ScoreStorage;
    /**
     * 加算 mask の有無（①QK だけ・省略時 `false` で生成物は 1 バイトも動かない）。
     * ③PV は mask を見ない（S は既に mask 済み）。
     */
    readonly mask?: boolean;
  }
  | {
    readonly op: "linear";
    readonly v4: boolean;
    readonly weight: WeightStorage;
    readonly compute?: GemmCompute;
    /** 平坦化後の行数 M（幾何のバケット — 省略時は既定幾何）。 */
    readonly rows?: number;
  }
  | {
    readonly op: "conv2d";
    readonly v4: boolean;
    readonly weight: WeightStorage;
    /** m タイルの行数（`conv2dIgemmMTile` が返す 64 行か {@link GEMM_MTILE_SMALL} の 32 行）。 */
    readonly mTile: number;
  };

/**
 * uniform の Dims（3 語 `{m,n,k}`）。uniform アドレス空間の struct は 16 バイト整列になるため、
 * 3 語ぶんの内容でも 16 バイト確保する MUST（不足すると binding が validation で落ちる）。
 *
 * bmm もバッチを uniform で運ばない — バッチは `wid.z` から導くので、載せても WGSL が
 * 一度も読まない死んだフィールドになる。
 */
export const gemmParams = (
  op: GemmSpec["op"],
  m: number,
  n: number,
  k: number,
): Uint32Array<ArrayBuffer> => {
  for (const value of [m, n, k]) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new CodegenError(`${op} params: m/n/k は非負整数（${m}, ${n}, ${k}）`);
    }
  }
  const params = new Uint32Array(4);
  params[0] = m;
  params[1] = n;
  params[2] = k;
  return params;
};

/**
 * バッチ軸（dispatch の z）を持つ op。attention は B·H を 1 本のバッチ軸に畳んだ形。
 *
 * MUST: conv2d もここに入る。出力 NCHW は `[B][Cout][Hout·Wout]` なので、バッチを N 側へ
 * 畳むと `[Cout][B·Hout·Wout]` になって**軸の順序が入れ替わる**（B = 1 でだけ一致するため
 * 実測形では露見しない）。B を z 軸に置けば 1 タイルが `[Cout][Hout·Wout]` の平面に閉じる。
 */
const BATCHED_OPS: ReadonlySet<GemmSpec["op"]> = new Set([
  "bmm",
  "attention_qk",
  "attention_pv",
  "conv2d",
]);

/** バッチのオフセット接頭辞（matmul / linear だけ空）。 */
const batchBase = (op: GemmSpec["op"], name: string): string =>
  BATCHED_OPS.has(op) ? `${name} + ` : "";

/**
 * 共有タイルへ書く 1 要素 / 1 quad の丸め（{@link GemmCompute} の丸め列 1）。
 *
 * MUST: f32 変種では**恒等**（引数の式をそのまま返す）。既存 6 op の生成バイト列がここに
 * 掛かっている。
 * MUST: 丸めるのは共有タイルへの書き込み**だけ**。読んだ直後や MAC ごとに `f16()` を挟むと
 * 丸めが 2 回起きて「入力を f16 に丸めた f32 変種」という等価性が崩れる。
 */
const tileScalar = (compute: GemmCompute, expr: string): string =>
  compute === "f16" ? `f16(${expr})` : expr;

const tileQuad = (compute: GemmCompute, expr: string): string =>
  compute === "f16" ? `vec4<f16>(${expr})` : expr;

/** 共有タイルの要素型（f16 変種はバイト半減 — 期待利得はこの 1 機序に全て乗る）。 */
const tileScalarType = (compute: GemmCompute): string => compute === "f16" ? "f16" : "f32";

const tileQuadType = (compute: GemmCompute): string =>
  compute === "f16" ? "vec4<f16>" : "vec4<f32>";

/** v4 経路が使う quad 数の束縛（スカラ経路では空）。 */
const quadDims = (v4: boolean): string =>
  v4 ? `  let k4 = dims.k / ${GEMM_QUAD}u;\n  let n4 = dims.n / ${GEMM_QUAD}u;\n` : "";

/**
 * 充填スロットの列挙（0..n-1）。1 スレッドが複数の行 / k 行を受け持つ形は幾何の帰結で、
 * どの断片も同じ「スロット番号を接尾辞にした静的展開」で書く。
 */
const slots = (count: number): readonly number[] =>
  Array.from({ length: count }, (_, index) => index);

/** 添字の加算オフセット（0 は省く — 生成物を読める形に保つ）。 */
const at = (offset: number): string => offset === 0 ? "" : ` + ${offset}u`;

/**
 * スロットごとの i8 scale 変数名。スロット 0 は従来名（{@link WEIGHT_SCALE_VAR}）のままで、
 * 1 スレッド 1 チャネルの他カーネルと名前が揃う。
 */
const scaleVar = (slot: number): string =>
  slot === 0 ? WEIGHT_SCALE_VAR : `${WEIGHT_SCALE_VAR}${slot}`;

/**
 * bmm のバッチ base。
 *
 * MUST: v4 では **quad 単位**で数える（`wid.z * m * k4` であって `m * k` ではない）。
 * 要素単位のまま組むと batch ≥ 2 で隣のバッチを読み書きする — 例外なしの誤値。
 */
const batchPrologue = (op: GemmSpec["op"], v4: boolean): string =>
  `  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチのタイル）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすための前提${
    v4 ? "。MUST: base は quad 単位（要素単位で組むと隣のバッチを読む）" : ""
  }
  let abase = wid.z * dims.m * ${v4 ? "k4" : "dims.k"};
${
    op === "attention_qk"
      // k は [B·H, N, D] のまま読む（旧経路の permute(kᵀ) が要らない）。転置読みは行頭を
      // **平坦添字**で作るので、base は v4 でも quad ではなく**要素単位**。
      ? `  let bbase = wid.z * dims.n * dims.k;\n`
      : `  let bbase = wid.z * dims.k * ${v4 ? "n4" : "dims.n"};\n`
  }  let cbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
${op === "attention_pv" ? "  let rbase = wid.z * dims.m;\n" : ""}`;

/**
 * A タイル（`[m,k]` 行優先）の担当と行頭オフセット。どれも K タイルループ不変。
 *
 * 1 スレッドが受け持つ行は {@link gemmRowSlots} 本で、スロット間は
 * {@link gemmRowFillStride} 行ずつ離れる（担当が重ならず、タイルに穴も空かない唯一の割り）。
 */
const prologueA = (geometry: GemmGeometry, op: GemmSpec["op"], v4: boolean): string => {
  const stride = gemmRowFillStride(geometry);
  const kUnit = v4 ? "k4" : "dims.k";
  const rows = slots(gemmRowSlots(geometry)).map((slot) =>
    slot === 0
      ? `  let arow0 = wid.y * ${gemmTileM(geometry)}u + ar;
  let arow_base0 = ${batchBase(op, "abase")}arow0 * ${kUnit};
  let sa_base0 = ar * ${GEMM_TILE_K}u + aq * ${GEMM_QUAD}u;`
      : `  let arow${slot} = arow0 + ${slot * stride}u;
  let arow_base${slot} = arow_base0 + ${slot * stride}u * ${kUnit};
  let sa_base${slot} = sa_base0 + ${slot * stride * GEMM_TILE_K}u;`
  ).join("\n");
  return `  // A タイルの担当（${gemmTileM(geometry)} 行 × ${GEMM_K_QUADS} quad を ${
    gemmThreads(geometry)
  } スレッドで ${gemmRowSlots(geometry)} 巡）
  let ar = tid / ${GEMM_K_QUADS}u;
  let aq = tid % ${GEMM_K_QUADS}u;
${rows}`;
};

/**
 * A タイルの充填。範囲外は 0 で埋める（内積に寄与しないので K タイル端数でも結果は変わらない）。
 * v4 は 1 スレッドが vec4 を 1 回読み、スカラ版は 4 成分を個別に境界検査して詰める。
 *
 * `value` は読んだ 1 要素に掛ける変換式（融合 attention の半スケール / `exp(S−m)·inv`）。
 * 第 2 引数は充填スロット番号 — ③PV が**行ごとの統計**（`row_max{slot}` / `row_inv{slot}`）を
 * 引くために要る（1 スレッドが複数行を埋めるので、統計も行ごとに別の変数になる）。
 * MUST: 変換は**範囲内の要素にだけ**掛ける。範囲外の 0 に `exp(0−m)·inv` を掛けると
 * 0 でない値が内積へ混ざり、端数タイルだけが静かに誤る。
 * MUST: 変換は**成分ごとのスカラ式**で書く（`vec4` へまとめて `exp` を掛けない）。加減乗算と
 * 違って超越関数のベクトル版は実装依存で、分解経路とのビット同一の前提が崩れうる。
 * MUST: `value` 省略時の生成物は従来のバイト列そのまま（スナップショットが検出器）。
 */
const fillA = (
  geometry: GemmGeometry,
  name: string,
  v4: boolean,
  value?: (expr: string, slot: number) => string,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string => {
  const filled = slots(gemmRowSlots(geometry)).map((slot) => {
    const read = (expr: string): string => value === undefined ? expr : value(expr, slot);
    // s16 の quad 読みは**同じ quad 添字**を取る（f32 変種の添字算術を 1 文字も動かさない）
    const quad = scoreReadQuad(name, score, `arow_base${slot} + t * ${GEMM_K_QUADS}u + aq`);
    const load = v4
      ? value === undefined
        ? `    if (arow${slot} < dims.m && ak0 < dims.k) {
      av${slot} = ${quad};
    }`
        : `    if (arow${slot} < dims.m && ak0 < dims.k) {
      let raw${slot} = ${quad};
      av${slot} = vec4<f32>(
        ${["x", "y", "z", "w"].map((lane) => read(`raw${slot}.${lane}`)).join(",\n        ")},
      );
    }`
      : `    if (arow${slot} < dims.m) {
      if (ak0 < dims.k) {
        av${slot}.x = ${read(`${name}[arow_base${slot} + ak0]`)};
      }
      if (ak0 + 1u < dims.k) {
        av${slot}.y = ${read(`${name}[arow_base${slot} + ak0 + 1u]`)};
      }
      if (ak0 + 2u < dims.k) {
        av${slot}.z = ${read(`${name}[arow_base${slot} + ak0 + 2u]`)};
      }
      if (ak0 + 3u < dims.k) {
        av${slot}.w = ${read(`${name}[arow_base${slot} + ak0 + 3u]`)};
      }
    }`;
    return `    var av${slot} = vec4<f32>(0.0);
${load}
    sa[sa_base${slot}] = ${tileScalar(compute, `av${slot}.x`)};
    sa[sa_base${slot} + 1u] = ${tileScalar(compute, `av${slot}.y`)};
    sa[sa_base${slot} + 2u] = ${tileScalar(compute, `av${slot}.z`)};
    sa[sa_base${slot} + 3u] = ${tileScalar(compute, `av${slot}.w`)};`;
  }).join("\n");
  return `    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
    let ak0 = t * ${GEMM_TILE_K}u + aq * ${GEMM_QUAD}u;
${filled}`;
};

/**
 * dense な B タイル（`[k,n]` 行優先 — 列方向がそのまま vec4）の担当。
 *
 * 列 quad はスロット不変で、担当する k 行だけが {@link gemmQuadFillStride} ずつ進む。
 */
const prologueBDense = (geometry: GemmGeometry, v4: boolean): string => {
  const stride = gemmQuadFillStride(geometry);
  const nQuads = gemmColumnQuads(geometry);
  const rows = slots(gemmQuadSlots(geometry)).slice(1)
    .map((slot) => `\n  let bk${slot} = bk0 + ${slot * stride}u;`).join("");
  return `  // B タイルの担当（K ${GEMM_TILE_K} 行 × 列 quad ${nQuads} を ${
    gemmThreads(geometry)
  } スレッドで ${gemmQuadSlots(geometry)} 巡）
  let bk0 = tid / ${nQuads}u;
  let bcq = tid % ${nQuads}u;
  ${
    v4
      ? `let bc4 = wid.x * ${nQuads}u + bcq;`
      : `let bcol = wid.x * ${gemmTileN(geometry)}u + bcq * ${GEMM_QUAD}u;`
  }${rows}`;
};

const fillBDense = (
  geometry: GemmGeometry,
  name: string,
  op: GemmSpec["op"],
  v4: boolean,
  compute: GemmCompute = "f32",
): string =>
  slots(gemmQuadSlots(geometry)).map((slot) =>
    `    let brow${slot} = t * ${GEMM_TILE_K}u + bk${slot};
    var bv4_${slot} = vec4<f32>(0.0);
${
      v4
        ? `    if (brow${slot} < dims.k && bc4 < n4) {
      bv4_${slot} = ${name}[${batchBase(op, "bbase")}brow${slot} * n4 + bc4];
    }`
        : `    if (brow${slot} < dims.k) {
      let brow_base = ${batchBase(op, "bbase")}brow${slot} * dims.n;
      if (bcol < dims.n) {
        bv4_${slot}.x = ${name}[brow_base + bcol];
      }
      if (bcol + 1u < dims.n) {
        bv4_${slot}.y = ${name}[brow_base + bcol + 1u];
      }
      if (bcol + 2u < dims.n) {
        bv4_${slot}.z = ${name}[brow_base + bcol + 2u];
      }
      if (bcol + 3u < dims.n) {
        bv4_${slot}.w = ${name}[brow_base + bcol + 3u];
      }
    }`
    }
    sb[bk${slot} * ${gemmColumnQuads(geometry)}u + bcq] = ${tileQuad(compute, `bv4_${slot}`)};`
  ).join("\n");

/** 共有 B タイルの成分（転置後に `wsl` が指す先）。 */
const TILE_COMPONENTS = ["x", "y", "z", "w"] as const;

/**
 * `wv` の k 連続 4 要素を共有 B タイルへ**転置して**置く（書き込む成分は実行時の `wsl`）。
 *
 * MUST: `sb[i][wsl] = v` と**動的インデックスで成分を書かない**。Deno 2.9.4 / Apple M2
 * （Metal）では `wsl != 0` の書き込みが**黙って捨てられ**、B タイルの 4 要素中 3 要素が 0 の
 * まま内積へ入る。動的成分書き込みを持たない matmul / bmm は無傷で、linear / attention /
 * conv2d だけが壊れる — 相対誤差が 1 を超える「全く違う値」になるのに例外は一切出ない。
 * ポインタ経由（`(*p)[wsl] = v`）も naga が同じ MSL に落とすため同様に壊れるので、
 * **静的成分へ落とす以外に回避手段が無い**（現状 / switch / スカラ配列 / ポインタの 4 通りを
 * 実機で突合して確認）。Linux / Vulkan では動的インデックスでも正しく動くため、この冗長さは
 * Metal のためだけにある。
 *
 * NOTE: `wsl = (tid / 4) % 4` なので simdgroup 内に 4 値が揃い、4 アームとも実行される。
 * コストが乗るのは B タイル充填だけで、頻度が桁違いに高い内積ループの読み出しは `sb` を
 * vec4 のまま一括で読む（共有タイルのレイアウトを変えない選択の理由）。
 */
const storeBTransposed = (geometry: GemmGeometry, compute: GemmCompute, slot: number): string => {
  const arm = (component: string): string =>
    TILE_COMPONENTS
      .map((source, quad) =>
        `        sb[sb_base${slot}${at(quad * gemmColumnQuads(geometry))}].${component} = ${
          tileScalar(compute, `wv${slot}.${source}`)
        };`
      )
      .join("\n");
  const arms = TILE_COMPONENTS
    .map((component, lane) =>
      `      ${lane === TILE_COMPONENTS.length - 1 ? "default" : `case ${lane}u`}: {\n${
        arm(component)
      }\n      }`
    )
    .join("\n");
  return `    switch wsl${slot} {\n${arms}\n    }`;
};

/**
 * linear の B タイル（重み `[n,k]` — 連続方向が k）の担当。
 *
 * 1 スレッドが担当チャネルごとに **k 連続 4 要素**を読み、**共有メモリ側で転置して置く**。
 * 内積ループを dense と共有できるのはこの転置のおかげ。i8 の scale は担当チャネル
 * `wcol{スロット}` のもので、K タイルループ不変なので**スロットごとに 1 度だけ**束ねる
 * （ADR 0019 のループ不変巻き上げ。名前は {@link scaleVar}）。端タイルでは `wcol >= n` に
 * なりうるが WGSL の境界付きアクセスで安全で、読んだ値は `wcol < n` のときしか `sb` に載らない。
 */
const prologueBLinear = (geometry: GemmGeometry, weight: WeightStorage): string => {
  const stride = gemmRowFillStride(geometry);
  const nQuads = gemmColumnQuads(geometry);
  const channels = slots(gemmColumnSlots(geometry)).map((slot) =>
    slot === 0
      ? `  let wc0 = tid / ${GEMM_K_QUADS}u;
  let wq = tid % ${GEMM_K_QUADS}u;
  let wcol0 = wid.x * ${gemmTileN(geometry)}u + wc0;${
        weightScaleWgsl(weight, "wcol0", "  ", scaleVar(0))
      }
  let wrow_base0 = wcol0 * dims.k;
  // 共有メモリ側で転置して置く（列 quad = wc / ${GEMM_QUAD}・成分 = wc % ${GEMM_QUAD}）
  let wsq0 = wc0 / ${GEMM_QUAD}u;
  let wsl0 = wc0 % ${GEMM_QUAD}u;
  let sb_base0 = (wq * ${GEMM_QUAD}u) * ${nQuads}u + wsq0;`
      : `  let wc${slot} = wc0 + ${slot * stride}u;
  let wcol${slot} = wcol0 + ${slot * stride}u;${
        weightScaleWgsl(weight, `wcol${slot}`, "  ", scaleVar(slot))
      }
  let wrow_base${slot} = wcol${slot} * dims.k;
  let wsq${slot} = wc${slot} / ${GEMM_QUAD}u;
  let wsl${slot} = wc${slot} % ${GEMM_QUAD}u;
  let sb_base${slot} = (wq * ${GEMM_QUAD}u) * ${nQuads}u + wsq${slot};`
  ).join("\n");
  return `  // W タイルの担当（${gemmTileN(geometry)} 出力チャネル × ${GEMM_K_QUADS} quad を ${
    gemmThreads(geometry)
  } スレッドで ${gemmColumnSlots(geometry)} 巡）
${channels}`;
};

const fillBLinear = (
  geometry: GemmGeometry,
  name: string,
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
): string => {
  const filled = slots(gemmColumnSlots(geometry)).map((slot) =>
    `    var wv${slot} = vec4<f32>(0.0);
${
      v4
        ? `    if (wcol${slot} < dims.n && wk0 < dims.k) {
      wv${slot} = ${weightRead4(name, weight, `wrow_base${slot} + wk0`, scaleVar(slot))};
    }`
        : `    if (wcol${slot} < dims.n) {
      let wbase = wrow_base${slot} + wk0;
      if (wk0 < dims.k) {
        wv${slot}.x = ${weightRead(name, weight, "wbase", scaleVar(slot))};
      }
      if (wk0 + 1u < dims.k) {
        wv${slot}.y = ${weightRead(name, weight, "wbase + 1u", scaleVar(slot))};
      }
      if (wk0 + 2u < dims.k) {
        wv${slot}.z = ${weightRead(name, weight, "wbase + 2u", scaleVar(slot))};
      }
      if (wk0 + 3u < dims.k) {
        wv${slot}.w = ${weightRead(name, weight, "wbase + 3u", scaleVar(slot))};
      }
    }`
    }
${storeBTransposed(geometry, compute, slot)}`
  ).join("\n");
  return `    let wk0 = t * ${GEMM_TILE_K}u + wq * ${GEMM_QUAD}u;
${filled}`;
};

/**
 * 融合 attention ① の B タイル（k `[B·H, N, D]` — 連続方向が D）の担当。
 *
 * 構造は linear の重み読み（`[n,k]` を k 連続で読み、共有側で転置して置く）と同一で、
 * 違いは **バッチ base が要る**ことと **重み格納の変種を持たない**（活性は常に f32）ことだけ。
 * この共有のおかげで、旧経路の `permute` による kᵀ 実体化がまるごと消える。
 */
const prologueBAttentionQk = (geometry: GemmGeometry): string => {
  const stride = gemmRowFillStride(geometry);
  const nQuads = gemmColumnQuads(geometry);
  const columns = slots(gemmColumnSlots(geometry)).map((slot) =>
    slot === 0
      ? `  let wc0 = tid / ${GEMM_K_QUADS}u;
  let wq = tid % ${GEMM_K_QUADS}u;
  let wcol0 = wid.x * ${gemmTileN(geometry)}u + wc0;
  let krow_base0 = bbase + wcol0 * dims.k;
  let wsq0 = wc0 / ${GEMM_QUAD}u;
  let wsl0 = wc0 % ${GEMM_QUAD}u;
  let sb_base0 = (wq * ${GEMM_QUAD}u) * ${nQuads}u + wsq0;`
      : `  let wc${slot} = wc0 + ${slot * stride}u;
  let wcol${slot} = wcol0 + ${slot * stride}u;
  let krow_base${slot} = krow_base0 + ${slot * stride}u * dims.k;
  let wsq${slot} = wc${slot} / ${GEMM_QUAD}u;
  let wsl${slot} = wc${slot} % ${GEMM_QUAD}u;
  let sb_base${slot} = (wq * ${GEMM_QUAD}u) * ${nQuads}u + wsq${slot};`
  ).join("\n");
  return `  // k タイルの担当（${gemmTileN(geometry)} 列（N）× ${GEMM_K_QUADS} quad を ${
    gemmThreads(geometry)
  } スレッドで ${gemmColumnSlots(geometry)} 巡）。
  // k は [N,D] のまま読み、**共有メモリ側で転置して置く**（linear の重み読みと同じ構造）
${columns}`;
};

const fillBAttentionQk = (
  geometry: GemmGeometry,
  v4: boolean,
  compute: GemmCompute = "f32",
): string => {
  const filled = slots(gemmColumnSlots(geometry)).map((slot) =>
    `    var wv${slot} = vec4<f32>(0.0);
${
      v4
        ? `    if (wcol${slot} < dims.n && wk0 < dims.k) {
      wv${slot} = k[(krow_base${slot} + wk0) >> 2u];
    }`
        : `    if (wcol${slot} < dims.n) {
      if (wk0 < dims.k) {
        wv${slot}.x = k[krow_base${slot} + wk0];
      }
      if (wk0 + 1u < dims.k) {
        wv${slot}.y = k[krow_base${slot} + wk0 + 1u];
      }
      if (wk0 + 2u < dims.k) {
        wv${slot}.z = k[krow_base${slot} + wk0 + 2u];
      }
      if (wk0 + 3u < dims.k) {
        wv${slot}.w = k[krow_base${slot} + wk0 + 3u];
      }
    }`
    }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv${slot} = wv${slot} * dims.scale;
${storeBTransposed(geometry, compute, slot)}`
  ).join("\n");
  return `    let wk0 = t * ${GEMM_TILE_K}u + wq * ${GEMM_QUAD}u;
${filled}`;
};

/**
 * 融合 attention ③ が使う行統計の束縛（K タイルループ不変なので 1 度だけ引く）。
 *
 * `stats` は `[B·H·M, 2]` で `[0]` が行の最大値 `m`・`[1]` が `1/Σexp(S−m)`。②（行統計）が
 * 現行 softmax のパス①②と同一の縮約で作る。
 *
 * MUST: 行の添字は `rbase + arow`（バッチごとの行オフセット）。`arow` だけで引くと
 * B·H ≥ 2 で全バッチが 0 番目のバッチの統計を使い、**B=H=1 のテストでは値に出ない**。
 * 1 スレッドは A タイルを {@link gemmRowSlots} 行ぶん埋めるので、統計もその行ごとに引く。
 */
const prologueAttentionStats = (geometry: GemmGeometry): string => {
  const rows = slots(gemmRowSlots(geometry)).map((slot) =>
    `  let stat_at${slot} = select(0u, (rbase + arow${slot}) * 2u, arow${slot} < dims.m);
  let row_max${slot} = stats[stat_at${slot}];
  let row_inv${slot} = stats[stat_at${slot} + 1u];`
  ).join("\n");
  return `  // 端タイルでは arow >= m がありうるので添字を 0 へ倒す（読んだ値は arow < m の枝でしか
  // 使われない）。範囲外の stats を読むこと自体は WGSL の境界付きアクセスで安全
${rows}`;
};

/**
 * 書き出し。`ceil(m/64) × ceil(n/64)` タイルが出力の全要素をちょうど 1 回ずつ覆う
 * （full-write — ADR 0014）。v4 は `n % 4 == 0` なので quad の端数が出ない。
 *
 * `bias` は常に f32 なのでキャスト無しで vec4 に組める（ADR 0006）。
 *
 * `outF16` は融合 attention ① の S だけが立てる（ADR 0028 の丸め列 4 — ②③ が f16 で読むので
 * transient が半減する）。**bias とは同時に立たない**（bias を持つのは linear / conv2d だけで、
 * どちらも出力は f32）。
 *
 * `score` も融合 attention ① の S だけが立てる（案 γ 波 1 の `pack2x16float` 格納）。
 * MUST: **v4 経路でしか立たない** — スカラ経路の部分書きは同じ u32 語への read-modify-write
 * になる（src/kernels/score-storage.ts）。生成の入口が `!v4` を fail loudly で落とす。
 *
 * `mask` も融合 attention ① だけが立てる（加算型 mask — ADR 0023 改訂）。**書き出しの直前に
 * `S' = fl(acc + mask[m·N + n])` を 1 度足すだけ**で、縮約ループには一切触れない。
 * MUST: mask の添字に**バッチ base を足さない**（契約は `[1,1,M,N]` で B·H 全体へ broadcast
 * する — 足すと 2 バッチ目以降が範囲外／別の行を読む）。
 * MUST: 加算は f32 で行い、`outF16` / `score` の丸めはその**後**に来る（分解経路の
 * `bmm → add` と丸めの位置と回数が一致することがビット同一の根拠）。
 * MUST: `bias` とは同時に立たない（bias を持つのは linear / conv2d だけ）。
 *
 * 行・列 quad とも codegen 時に展開して `acc{行}_{列 quad}` を静的に読む
 * （{@link gemmAccumulatorInit} の理由）。ガードの構造・bias の足し順・書き出す語の並びは
 * 幾何によらず同一。
 */
const store = (
  geometry: GemmGeometry,
  name: string,
  op: GemmSpec["op"],
  v4: boolean,
  bias: boolean,
  outF16 = false,
  score: ScoreStorage = "f32",
  mask = false,
): string => {
  const base = batchBase(op, "cbase");
  const quads = gemmQuadsPerThread(geometry);
  const rows = slots(geometry.regM).map((row) =>
    row === 0
      ? `  let orow0 = wid.y * ${gemmTileM(geometry)}u + lid.y * ${geometry.regM}u;`
      : `  let orow${row} = orow0 + ${row}u;`
  ).join("\n");
  if (v4) {
    const columns = slots(quads).map((quad) =>
      quad === 0
        ? `  let ocq0 = wid.x * ${gemmColumnQuads(geometry)}u + lid.x${
          quads === 1 ? "" : ` * ${quads}u`
        };`
        : `  let ocq${quad} = ocq0 + ${quad}u;`
    ).join("\n");
    const blocks = slots(quads).map((quad) => {
      const biasBind = bias
        ? `
    // n % ${GEMM_QUAD} == 0 かつ ocq${quad} < n4 なので oc + ${
          GEMM_QUAD - 1
        } < n（bias は常に f32 — ADR 0006）
    let oc = ocq${quad} * ${GEMM_QUAD}u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);`
        : "";
      const rowStores = slots(geometry.regM).map((row) => {
        const acc = `acc${row}_${quad}`;
        const summed = mask ? "sv" : acc;
        const value = bias ? `${summed} + biasv` : outF16 ? `vec4<f16>(${summed})` : summed;
        // 加算 mask は行 m・列 n の平坦添字（バッチ base 無し）。quad 単位で数えるのは
        // 出力と同じ `n % 4 == 0` が v4 の条件だから（mask の行頭も quad 境界にある）。
        const maskAdd = mask ? `      let sv = ${acc} + mask[orow${row} * n4 + ocq${quad}];\n` : "";
        return `    if (orow${row} < dims.m) {
${maskAdd}      ${scoreStoreQuad(name, score, `${base}orow${row} * n4 + ocq${quad}`, value)}
    }`;
      }).join("\n");
      return `  if (ocq${quad} < n4) {${biasBind}
${rowStores}
  }`;
    }).join("\n");
    return `${columns}
${rows}
${blocks}`;
  }
  const write = (row: number, col: number): string => {
    const acc = `acc${row}_${Math.floor(col / GEMM_QUAD)}.${TILE_COMPONENTS[col % GEMM_QUAD]}`;
    const summed = mask ? "sv" : acc;
    // 加算 mask は `mbase`（バッチ base を含まない行頭）からの平坦添字で 1 要素だけ読む
    const maskAdd = mask ? `      let sv = ${acc} + mask[mbase + ocol${at(col)}];\n` : "";
    return `    if (ocol${at(col)} < dims.n) {
${maskAdd}      ${name}[obase + ocol${at(col)}] = ${outF16 ? `f16(${summed})` : summed}${
      bias ? ` + bias[ocol${at(col)}]` : ""
    };
    }`;
  };
  const rowStores = slots(geometry.regM).map((row) =>
    `  if (orow${row} < dims.m) {
    let obase = ${base}orow${row} * dims.n;
${mask ? `    let mbase = orow${row} * dims.n;\n` : ""}${
      slots(geometry.regN).map((col) => write(row, col)).join("\n")
    }
  }`
  ).join("\n");
  return `  let ocol = wid.x * ${gemmTileN(geometry)}u + lid.x * ${geometry.regN}u;
${rows}
${rowStores}`;
};

/**
 * K 内側ループ 1 段（共有タイルから `regN/4` 本の vec4 と `regM` 本のスカラを読み、
 * `regM · regN` 個の MAC を回す）。行 昇順・列 quad 昇順に並べるだけなので、
 * **1 出力要素あたりの加算順序は幾何によらず同一**（K タイル 16 昇順の逐次）。
 *
 * MUST: 積和は `acc = acc + a * b` の字面のまま（`fma` を明示しない）。i8a8 の i32 縮約と
 * 違ってここは f32 縮約なので、**backend の fma 縮約判断が変われば最終ビットは動きうる**
 * — PNG sha256 門（f16-1024 を含む 4 本）と WAV 門が検出器。
 */
const accumulatorUpdate = (geometry: GemmGeometry, compute: GemmCompute): string => {
  const { regM } = geometry;
  const quads = gemmQuadsPerThread(geometry);
  const column = (quad: number): string =>
    `sb[kk * ${gemmColumnQuads(geometry)}u + lid.x${quads === 1 ? "" : ` * ${quads}u${at(quad)}`}]`;
  const loads = slots(quads).map((quad) =>
    `      let bv${quad} = ${compute === "f16" ? `vec4<f32>(${column(quad)})` : column(quad)};`
  ).join("\n");
  const row = (index: number): string =>
    compute === "f16"
      ? `f32(sa[(lid.y * ${regM}u + ${index}u) * ${GEMM_TILE_K}u + kk])`
      : `sa[(lid.y * ${regM}u + ${index}u) * ${GEMM_TILE_K}u + kk]`;
  const macs = slots(regM).map((index) =>
    slots(quads).map((quad) =>
      `      acc${index}_${quad} = acc${index}_${quad} + ${row(index)} * bv${quad};`
    ).join("\n")
  ).join("\n");
  return `${loads}\n${macs}`;
};

/**
 * head / Dims 追加欄 / prologue / fillA / fillB / store の断片を差し込んだ 1 本のシェーダ。
 *
 * `dimsExtra` は uniform の 4 語目以降（融合 attention ① の `scale` / conv2d の幾何 13 語）。
 * uniform struct は 16 バイト整列なので 4 語目は既に確保済みで、既存 3 op では空文字
 * （生成物は 1 バイトも動かない）。`accInit` も同様に省略時は従来の 0 初期化そのまま。
 */
const skeleton = (
  geometry: GemmGeometry,
  header: string,
  bindings: string,
  loader: string,
  prologue: string,
  fillTiles: string,
  storeTile: string,
  dimsExtra = "",
  accInit = gemmAccumulatorInit(geometry),
  compute: GemmCompute = "f32",
): string => {
  const tileM = gemmTileM(geometry);
  const nQuads = gemmColumnQuads(geometry);
  return `${header}${
    // MUST: `enable` はモジュール先頭・全ての global 宣言より前の directive（コメントだけが
    // 前に来てよい）。f16 変種のときだけここが差し込む — f32 変種の生成物は 1 バイトも動かない。
    compute === "f16" ? "\nenable f16;" : ""}
struct Dims {
  m: u32,
  n: u32,
  k: u32,
${dimsExtra}}
@group(0) @binding(0) var<uniform> dims: Dims;
${bindings}
${loader}
// 共有 A タイル（${tileM} 行 × K ${GEMM_TILE_K}・スカラ格納）と
// 共有 B タイル（K ${GEMM_TILE_K} × 列 quad ${nQuads}・列方向を vec4 に束ねた形）${
    compute === "f16"
      ? `。f16 変種は共有バイトが半分
// （${tileM * GEMM_TILE_K * 4 + GEMM_TILE_K * nQuads * 16} B → ${
        tileM * GEMM_TILE_K * 2 + GEMM_TILE_K * nQuads * 8
      } B / WG）— 期待利得はこの 1 機序に全て乗る`
      : ""
  }
var<workgroup> sa: array<${tileScalarType(compute)}, ${tileM * GEMM_TILE_K}>;
var<workgroup> sb: array<${tileQuadType(compute)}, ${GEMM_TILE_K * nQuads}>;

@compute @workgroup_size(${geometry.wgX}, ${geometry.wgY})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${geometry.wgX}u + lid.x;
${prologue}
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + ${GEMM_TILE_K - 1}u) / ${GEMM_TILE_K}u;
${accInit}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillTiles}
    workgroupBarrier();
    // 共有ロード ${geometry.regM + gemmQuadsPerThread(geometry)} 回（B の vec4 ${
    gemmQuadsPerThread(geometry)
  } + A のスカラ ${geometry.regM}）で ${geometry.regM * geometry.regN} MAC。
    // 縮約は k 昇順の逐次で、1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と
    // 完全に一致する。${
    compute === "f16"
      ? `
    // MUST: f32 への拡幅は**レジスタロード時に 1 回**（${
        geometry.regM + gemmQuadsPerThread(geometry)
      } 回 / ${geometry.regM * geometry.regN} MAC）。MAC ごとに
    // f32(av * bv) と書くと変換が ${
        geometry.regM * geometry.regN
      } 回に増え、しかも積が f16 精度に落ちる（ADR 0028 の丸め列 2）。
    // f16 → f32 の拡幅は厳密なので、値は「入力を f16 に丸めた f32 変種」と 1 ビットも違わない。`
      : ""
  }
    for (var kk = 0u; kk < ${GEMM_TILE_K}u; kk = kk + 1u) {
${accumulatorUpdate(geometry, compute)}
    }
    workgroupBarrier();
  }
${storeTile}
}
`;
};

const denseWgsl = (geometry: GemmGeometry, op: "matmul" | "bmm", v4: boolean): string => {
  const element = v4 ? "vec4<f32>" : "f32";
  const signature = op === "bmm" ? "a[b,m,k] · b[b,k,n]" : "a[m,k] · b[k,n]";
  return skeleton(
    geometry,
    `// karume ${op} (${signature}, f32, ${gemmGeometryNote(geometry)}${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> a: array<${element}>;
@group(0) @binding(2) var<storage, read> b: array<${element}>;
@group(0) @binding(3) var<storage, read_write> c: array<${element}>;`,
    "",
    `${quadDims(v4)}${op === "bmm" ? batchPrologue(op, v4) : ""}${prologueA(geometry, op, v4)}
${prologueBDense(geometry, v4)}`,
    `${fillA(geometry, "a", v4)}
${fillBDense(geometry, "b", op, v4)}`,
    store(geometry, "c", op, v4, false),
  );
};

/**
 * 融合 attention ①（QK）— `S[b,m,n] = Σ_d (q·scale)[b,m,d] · (k·scale)[b,n,d]`（ADR 0023）。
 *
 * A は q（`[batch,M,D]` 連続）で dense の A 充填そのまま、B は k（`[batch,N,D]` 連続）を
 * linear の重み読みと同じ「k 連続で読んで共有側で転置」構造で読む。**縮約は K 昇順・
 * K タイル 16** で現行 bmm と 1 ビット違わない。
 *
 * `mask` 変種は**書き出しの epilogue で `S' = fl(S + mask[m·N+n])` を足すだけ**（束縛が
 * 1 本増える以外は同一の生成物）。分解経路の `bmm`（S を実体化）→ `add`（mask）と丸めが
 * 起きる位置も回数も同じなので、ビット同一が保たれる。
 */
const attentionQkWgsl = (
  geometry: GemmGeometry,
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
  mask = false,
): string => {
  const f16 = compute === "f16";
  assertScoreStorageSupported("attention_qk", score, v4, f16);
  const element = v4 ? "vec4<f32>" : "f32";
  return skeleton(
    geometry,
    `// karume attention_qk (S[b,m,n] = (q·scale)[b,m,d] · (k·scale)[b,n,d]ᵀ${
      mask ? " + mask[m,n]" : ""
    }, f32${f16 ? ", f16 タイル計算・S も f16" : ""}${scoreNote(score)}, ${
      gemmGeometryNote(geometry)
    }${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> q: array<${element}>;
@group(0) @binding(2) var<storage, read> k: array<${element}>;
@group(0) @binding(3) var<storage, read_write> s: array<${
      scoreArrayType(score, v4 ? tileQuadType(compute) : tileScalarType(compute))
    }>;${
      mask
        ? `
// 加算 mask [1,1,M,N]（B·H の全バッチへ broadcast — 添字にバッチ base は入らない）
@group(0) @binding(${ATTENTION_QK_MASK_BINDING}) var<storage, read> mask: array<${element}>;`
        : ""
    }`,
    scoreStoreWgsl("s", score),
    `${quadDims(v4)}${batchPrologue("attention_qk", v4)}${prologueA(geometry, "attention_qk", v4)}
${prologueBAttentionQk(geometry)}`,
    `${fillA(geometry, "q", v4, (expr) => `${expr} * dims.scale`, compute)}
${fillBAttentionQk(geometry, v4, compute)}`,
    store(geometry, "s", "attention_qk", v4, false, f16, score, mask),
    "  scale: f32,\n",
    gemmAccumulatorInit(geometry),
    compute,
  );
};

/**
 * 融合 attention ③（PV）— `O[b,m,d] = Σ_n P[b,m,n] · v[b,n,d]`、`P = exp(S−m)·inv`（ADR 0023）。
 *
 * 形は bmm そのもの（B = v は `[batch,N,D]` 連続の dense）で、違いは **A タイル充填時に
 * P を作る**ことだけ。P を実体化しないので、旧経路の softmax 出力（+ 恒等 expand）1 枚ぶんの
 * バッファと dispatch が消える。値は現行 softmax のパス③（`exp(x−amax)·inv`）と同じ式・
 * 同じ演算順なので**ビット同一**。
 */
const attentionPvWgsl = (
  geometry: GemmGeometry,
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string => {
  const f16 = compute === "f16";
  assertScoreStorageSupported("attention_pv", score, v4, f16);
  // MUST: S を f32 へ広げてから `exp(S−m)·inv` を評価する（拡幅は厳密なので値は動かない）。
  // f16 のまま `exp` を評価すると指数関数が f16 精度になり、丸め列が 1 段増える。
  const probability = (expr: string, slot: number): string =>
    `exp(${f16 ? `f32(${expr})` : expr} - row_max${slot}) * row_inv${slot}`;
  return skeleton(
    geometry,
    `// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P = exp(S − m)·inv は非実体化, f32${
      f16 ? ", f16 タイル計算・S は f16" : ""
    }${scoreNote(score)}, ${gemmGeometryNote(geometry)}${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> s: array<${
      scoreArrayType(score, v4 ? tileQuadType(compute) : tileScalarType(compute))
    }>;
@group(0) @binding(2) var<storage, read> v: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<${v4 ? "vec4<f32>" : "f32"}>;`,
    scoreQuadLoaderWgsl("s", score),
    `${quadDims(v4)}${batchPrologue("attention_pv", v4)}${prologueA(geometry, "attention_pv", v4)}
${prologueAttentionStats(geometry)}
${prologueBDense(geometry, v4)}`,
    `${fillA(geometry, "s", v4, probability, compute, score)}
${fillBDense(geometry, "v", "attention_pv", v4, compute)}`,
    store(geometry, "o", "attention_pv", v4, false),
    "",
    gemmAccumulatorInit(geometry),
    compute,
  );
};

const linearVariantWgsl = (
  geometry: GemmGeometry,
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
): string => {
  // MUST: i8 重み × f16 計算は組まない（w8a16 — ADR 0028 決定 3）。ALU が 1:1 の機では
  // 速度の案として成立せず、品質の案としては需要が出てから別途裁定する。黙って f32 計算へ
  // 落とすと「i8 資産のときだけ f16 変種が効かない」形の沈黙になるので、生成の入口で落とす。
  if (compute === "f16" && weight === "i8") {
    throw new CodegenError(
      "linear: 重み i8 格納 × f16 計算（w8a16）は未実装 — " +
        "linearCompute を 'f32' か 'i8a8' にするか、重みを f32 / f16 格納で持つこと",
    );
  }
  const element = v4 ? "vec4<f32>" : "f32";
  return skeleton(
    geometry,
    `// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32${weightNote(weight)}${
      compute === "f16" ? ", f16 タイル計算" : ""
    }, ${gemmGeometryNote(geometry)}${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> x: array<${element}>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight, v4)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${element}>;`,
    weightLoaderWgsl("w", weight, LINEAR_SCALE_BINDING, v4),
    `${quadDims(v4)}${prologueA(geometry, "linear", v4)}
${prologueBLinear(geometry, weight)}`,
    `${fillA(geometry, "x", v4, undefined, compute)}
${fillBLinear(geometry, "w", weight, v4, compute)}`,
    store(geometry, "out", "linear", v4, true),
    "",
    gemmAccumulatorInit(geometry),
    compute,
  );
};

/**
 * conv2d の implicit GEMM が uniform に足す幾何 12 語（`{m,n,k}` の後ろ）。
 *
 * `m = Cout` / `n = Hout·Wout` / `k = Cin·Kh·Kw`（groups == 1・バッチは dispatch の z 軸）。
 * MUST: 並びは src/kernels/conv2d.ts の `conv2dIgemmParams` と対。
 * NOTE: `height_out` は載せない（`n / width_out` で導けるうえ WGSL が一度も読まない死んだ
 * フィールドになる — ADR 0022 決定 5 で bmm の `batch` を落としたのと同じ規律）。
 */
const CONV2D_DIMS_EXTRA = `  channels_in: u32,
  height_in: u32,
  width_in: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  stride_h: u32,
  stride_w: u32,
  padding_h: u32,
  padding_w: u32,
  dilation_h: u32,
  dilation_w: u32,
`;

/**
 * conv2d の `acc` 初期値 = **bias-first**（ADR 0024 の MUST ①）。
 *
 * 直接カーネルの `var acc = bias[oc];` をそのまま再現する。GEMM の「store で最後に足す」形に
 * 流用すると `(Σ) + bias` になり、丸めの並びが変わってビット同一が崩れる（本設計で最大の
 * 分岐点）。行 = 出力チャネルなので bias は行ごとのスカラ splat になる。
 * 端タイルでは `bias0 + i >= m` を読みうるが、その行の `acc` は `store` の行ガードで捨てられる
 * （範囲外の storage 読み自体は WGSL の境界付きアクセスで安全）。
 */
const conv2dAccInit = (geometry: GemmGeometry): string =>
  `  let bias0 = wid.y * ${gemmTileM(geometry)}u + lid.y * ${geometry.regM}u;
${gemmAccumulatorInit(geometry, (row) => `vec4<f32>(bias[bias0${at(row)}])`)}`;

/**
 * `Xcol[k][n]` の 1 要素（x の暗黙 gather — im2col を実体化しない）。
 *
 * MUST: 範囲外は **0 を返す**（ADR 0024 の MUST ③）。クランプした添字で読むと実在する別
 * ピクセルが混ざり、例外の出ない誤値になる。直接カーネルの「padding 域は加算せず読み飛ばす」
 * と値が一致するのは `a + 0.0 == a`（`a` が有限）だからで、唯一の例外は符号付きゼロ
 * （部分和がちょうど `−0.0` のときだけ `+0.0` に転ぶ）。
 */
const CONV2D_XCOL_WGSL = `
// Xcol[k][n] = x[b, ic, oy·sh − ph + kh·dh, ox·sw − pw + kw·dw]（範囲外は 0）。
// xc = b·Cin + ic（バッチは dispatch の z 軸なので呼び出し側で足す）
fn xcol(xc: u32, ky: i32, kx: i32, n: u32) -> f32 {
  let iy = i32((n / dims.width_out) * dims.stride_h) + ky;
  let ix = i32((n % dims.width_out) * dims.stride_w) + kx;
  if (iy < 0 || u32(iy) >= dims.height_in || ix < 0 || u32(ix) >= dims.width_in) {
    return 0.0;
  }
  return x[(xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix)];
}
`;

/**
 * conv2d の A タイル（重み `[Cout, K]` — 平坦化すると行優先の `[m,k]` そのもの）の担当。
 *
 * dense の {@link prologueA} と同じ割り当てだが、読みが weight-storage 経由になるぶん
 * **行頭は要素単位**で持つ（{@link weightRead4} が平坦添字を取り、f32 の quad 添字化は
 * 内部で行う）。
 * MUST: i8 の scale は **行 = 出力チャネル**（ADR 0024 の MUST ④）。linear 側の `wcol`
 * （= 列）を持ってくると列 = ピクセルの scale を引く沈黙誤値になり、**m タイルが 2 枚以上
 * ある形**のテストだけが検出器になる。
 */
const prologueAConv2d = (
  geometry: GemmGeometry,
  weight: WeightStorage,
): string => {
  const stride = gemmRowFillStride(geometry);
  const rows = slots(gemmRowSlots(geometry)).map((slot) =>
    slot === 0
      ? `  let arow0 = wid.y * ${gemmTileM(geometry)}u + ar;${
        weightScaleWgsl(weight, "arow0", "  ", scaleVar(0))
      }
  let arow_base0 = arow0 * dims.k;
  let sa_base0 = ar * ${GEMM_TILE_K}u + aq * ${GEMM_QUAD}u;`
      : `  let arow${slot} = arow0 + ${slot * stride}u;${
        weightScaleWgsl(weight, `arow${slot}`, "  ", scaleVar(slot))
      }
  let arow_base${slot} = arow_base0 + ${slot * stride}u * dims.k;
  let sa_base${slot} = sa_base0 + ${slot * stride * GEMM_TILE_K}u;`
  ).join("\n");
  return `  // A タイルの担当（${gemmTileM(geometry)} 行 × ${GEMM_K_QUADS} quad を ${
    gemmThreads(geometry)
  } スレッドで ${gemmRowSlots(geometry)} 巡）
  let ar = tid / ${GEMM_K_QUADS}u;
  let aq = tid % ${GEMM_K_QUADS}u;
${rows}`;
};

/** conv2d の A タイル充填（{@link fillA} と {@link fillBLinear} を合成した形）。 */
const fillAConv2d = (
  geometry: GemmGeometry,
  name: string,
  weight: WeightStorage,
  v4: boolean,
): string => {
  const filled = slots(gemmRowSlots(geometry)).map((slot) =>
    `    var av${slot} = vec4<f32>(0.0);
${
      v4
        ? `    if (arow${slot} < dims.m && ak0 < dims.k) {
      av${slot} = ${weightRead4(name, weight, `arow_base${slot} + ak0`, scaleVar(slot))};
    }`
        : `    if (arow${slot} < dims.m) {
      let abase = arow_base${slot} + ak0;
      if (ak0 < dims.k) {
        av${slot}.x = ${weightRead(name, weight, "abase", scaleVar(slot))};
      }
      if (ak0 + 1u < dims.k) {
        av${slot}.y = ${weightRead(name, weight, "abase + 1u", scaleVar(slot))};
      }
      if (ak0 + 2u < dims.k) {
        av${slot}.z = ${weightRead(name, weight, "abase + 2u", scaleVar(slot))};
      }
      if (ak0 + 3u < dims.k) {
        av${slot}.w = ${weightRead(name, weight, "abase + 3u", scaleVar(slot))};
      }
    }`
    }
    sa[sa_base${slot}] = av${slot}.x;
    sa[sa_base${slot} + 1u] = av${slot}.y;
    sa[sa_base${slot} + 2u] = av${slot}.z;
    sa[sa_base${slot} + 3u] = av${slot}.w;`
  ).join("\n");
  return `    // 範囲外は 0 で埋める。内積に寄与しないので K 端数でも結果は変わらない
    let ak0 = t * ${GEMM_TILE_K}u + aq * ${GEMM_QUAD}u;
${filled}`;
};

/**
 * conv2d の B タイル（`Xcol[k,n]` の暗黙 gather）の担当。割り当ては dense と同じ。
 *
 * MUST: バッチは `wid.z`（{@link BATCHED_OPS} の doc）。`xbase` は x の**チャネル平面**単位、
 * `cbase` は出力要素（v4 では quad）単位で数える。
 */
const prologueBConv2d = (geometry: GemmGeometry, v4: boolean): string => {
  const stride = gemmQuadFillStride(geometry);
  const nQuads = gemmColumnQuads(geometry);
  const rows = slots(gemmQuadSlots(geometry)).slice(1)
    .map((slot) => `\n  let bk${slot} = bk0 + ${slot * stride}u;`).join("");
  return `  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Hout·Wout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
  // B タイルの担当（K ${GEMM_TILE_K} 行 × 列 quad ${nQuads} を ${
    gemmThreads(geometry)
  } スレッドで ${gemmQuadSlots(geometry)} 巡）
  let bk0 = tid / ${nQuads}u;
  let bcq = tid % ${nQuads}u;
  ${
    v4
      ? `let bc4 = wid.x * ${nQuads}u + bcq;`
      : `let bcol = wid.x * ${gemmTileN(geometry)}u + bcq * ${GEMM_QUAD}u;`
  }${rows}
  // K タイルループ不変（平坦 k を (ic, kh, kw) へ割るための刻み）
  let khw = dims.kernel_h * dims.kernel_w;`;
};

/**
 * 平坦 k → `(ic, kh, kw)` の分解と入力座標のオフセット。
 *
 * MUST: 平坦 k の昇順が直接カーネルの `(ic, kh, kw)` 三重昇順と一致することが、
 * K タイル 16 昇順 → ビット同一の土台（ADR 0024）。
 */
const conv2dKDecode = (slot: number): string =>
  `      let ic = brow${slot} / khw;
      let kr = brow${slot} % khw;
      let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);
      let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);`;

/**
 * conv2d の B タイル充填（本設計で唯一の完全新規部分）。
 *
 * MUST: x を読むのは `bc4 < n4`（v4）/ `bcol + j < n`（スカラ）の門の**内側**だけ。
 * MUST: v4 の連続 4 列読みが成立する条件は `Wout % 4 == 0 && stride_w == 1`
 * （`N % 4 == 0` では不十分 — Hout=Wout=2 は N=4 だが quad が出力行をまたぐ）。判定は
 * `conv2dUsesVec4`（src/kernels/conv2d.ts）1 箇所。
 *
 * NOTE（検出限界・実測）: `bc4 < n4` の門を落としても**数値テストは全て緑のまま**になる
 * （故障注入で実測）。WGSL の境界付きアクセスで範囲外読みが安全なうえ、そこで作った `sb` の
 * 列は `store` の `ocq < n4` に阻まれて一度も書き出されないため。検出器は WGSL スナップ
 * ショット（tests/fixtures/wgsl/conv2d_igemm*.wgsl）だけで、これは意図した構造の固定。
 */
const fillBConv2d = (geometry: GemmGeometry, v4: boolean): string =>
  slots(gemmQuadSlots(geometry)).map((slot) =>
    `    let brow${slot} = t * ${GEMM_TILE_K}u + bk${slot};
    var bv4_${slot} = vec4<f32>(0.0);
${
      v4
        ? `    if (brow${slot} < dims.k && bc4 < n4) {
${conv2dKDecode(slot)}
      let xc = xbase + ic;
      // quad の 4 列は同じ出力行の連続 ox（v4 の条件）なので x 側も連続に読める
      let n0 = bc4 * ${GEMM_QUAD}u;
      let iy = i32((n0 / dims.width_out) * dims.stride_h) + ky;
      let ix0 = i32((n0 % dims.width_out) * dims.stride_w) + kx;
      if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
        bv4_${slot} = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 画像端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4_${slot} = vec4<f32>(
          xcol(xc, ky, kx, n0),
          xcol(xc, ky, kx, n0 + 1u),
          xcol(xc, ky, kx, n0 + 2u),
          xcol(xc, ky, kx, n0 + 3u),
        );
      }
    }`
        : `    if (brow${slot} < dims.k) {
${conv2dKDecode(slot)}
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4_${slot}.x = xcol(xc, ky, kx, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4_${slot}.y = xcol(xc, ky, kx, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4_${slot}.z = xcol(xc, ky, kx, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4_${slot}.w = xcol(xc, ky, kx, bcol + 3u);
      }
    }`
    }
    sb[bk${slot} * ${gemmColumnQuads(geometry)}u + bcq] = bv4_${slot};`
  ).join("\n");

/**
 * conv2d の implicit GEMM（ADR 0024）— `C[Cout, N] = W[Cout, K] × Xcol[K, N]`。
 *
 * A は重み（平坦化で `[M,K]` 行優先そのもの・weight-storage 3 変種）、B は x の暗黙 gather、
 * store は 1 バッチぶんの `[Cout][Hout·Wout]` 行優先 = **NCHW の平面そのもの**なので後段の
 * レイアウト変換は要らない（バッチは z 軸 — {@link BATCHED_OPS}）。
 * bias は `acc` の初期値（{@link conv2dAccInit}）。m タイルは 64 行 / 32 行の 2 変種
 * （{@link GEMM_MTILE_SMALL} — 出力の**担当割り**だけが変わり、数値経路は共通）。
 */
const conv2dIgemmWgsl = (
  geometry: GemmGeometry,
  weight: WeightStorage,
  v4: boolean,
): string =>
  skeleton(
    geometry,
    `// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32${
      weightNote(weight)
    }, implicit GEMM ${gemmGeometryNote(geometry)}${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight, v4)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${v4 ? "vec4<f32>" : "f32"}>;`,
    `${weightLoaderWgsl("w", weight, LINEAR_SCALE_BINDING, v4)}${CONV2D_XCOL_WGSL}`,
    `${v4 ? `  let n4 = dims.n / ${GEMM_QUAD}u;\n` : ""}${prologueAConv2d(geometry, weight)}
${prologueBConv2d(geometry, v4)}`,
    `${fillAConv2d(geometry, "w", weight, v4)}
${fillBConv2d(geometry, v4)}`,
    store(geometry, "out", "conv2d", v4, false),
    CONV2D_DIMS_EXTRA,
    conv2dAccInit(geometry),
  );

/** op 別の解決（conv2d = m タイル / 融合 attention = 既定固定 / 残り 3 op = 行数バケット）。 */
const resolveGeometry = (spec: GemmSpec): GemmGeometry => {
  switch (spec.op) {
    case "conv2d":
      return gemmMTileGeometry(spec.mTile);
    case "attention_qk":
    case "attention_pv":
      return defaultGemmGeometry();
    default:
      return rowsGeometry(spec.rows);
  }
};

/**
 * 生成入力 1 つから WGSL 1 本。同じ入力からは常にバイト単位で同じ文字列が出る。
 *
 * タイル幾何を解決する**唯一の点**（conv2d は {@link gemmMTileGeometry}・matmul / bmm / linear は
 * 行数バケット {@link rowsGeometry}・融合 attention は {@link defaultGemmGeometry}）で、門
 * （{@link assertGemmGeometry}）もここ 1 箇所。断片は幾何を受け取って流すだけなので、既定を
 * 差し替えたときの影響がこの関数に閉じる。
 */
export const gemmWgsl = (spec: GemmSpec): string => {
  const geometry = resolveGeometry(spec);
  assertGemmGeometry(geometry, spec.op);
  switch (spec.op) {
    case "linear":
      return linearVariantWgsl(geometry, spec.weight, spec.v4, spec.compute);
    case "conv2d":
      return conv2dIgemmWgsl(geometry, spec.weight, spec.v4);
    case "attention_qk":
      return attentionQkWgsl(geometry, spec.v4, spec.compute, spec.score, spec.mask);
    case "attention_pv":
      return attentionPvWgsl(geometry, spec.v4, spec.compute, spec.score);
    default:
      return denseWgsl(geometry, spec.op, spec.v4);
  }
};

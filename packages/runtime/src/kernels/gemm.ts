/**
 * GEMM 族（matmul / bmm / linear + 融合 attention の QK / PV + conv2d の implicit GEMM —
 * ADR 0022 / 0023 / 0024）が共有する **64×64 レジスタブロッキング + vec4** の骨格。
 *
 * 1 スレッドが 4×4 の出力を持ち（`acc: array<vec4<f32>, 4>` = 16 レジスタ）、共有 B タイルは
 * **列方向を vec4 に束ねた** `[k][列 quad]`。内側ループは共有ロード 5 回（B の vec4 1 +
 * A のスカラ 4）で 16 MAC を回す — 1 スレッド 1 出力の 16×16 タイル（2 ロード 1 MAC）に対し
 * 共有帯域あたりの演算密度が 6.4 倍になる。
 *
 * | 項目 | 値 |
 * | --- | --- |
 * | 出力タイル | 64×64（{@link GEMM_TILE}・dispatch は `ceil(dim / 64)`） |
 * | workgroup | 16×16 = 256 スレッド |
 * | 1 スレッドの出力 | 4×4 |
 * | K タイル | 16（旧 16×16 カーネルと同じ刻み） |
 * | 共有メモリ | `sa` 4,096 B + `sb` 4,096 B = 8,192 B / WG |
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

/** 出力タイルの一辺。dispatch は `ceil(dim / GEMM_TILE)`。 */
export const GEMM_TILE = 64;

/**
 * conv2d の implicit GEMM だけが持つ **32 行 m タイル**（n タイルは 64 のまま — ADR 0024 隣接）。
 *
 * 動機は M = Cout が 64 の倍数でない層のタイル量子化の無駄（census の Cout=96 は
 * `ceil(96/64)·64/96 = 1.33×`）。**1 スレッドの出力は 4×4 のまま**で workgroup を
 * 16×8 = 128 スレッドに落とすので、内積ループの「共有ロード 5 回で 16 MAC」の密度は変わらない
 * （16×16 のまま 1 スレッド 2×4 に落とす形は 3 ロード 8 MAC = 密度 2.67 へ下がる）。
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

/** 1 スレッドが持つ出力の一辺（`acc` は vec4 × REG）。vec4 に束ねる都合で 4 に固定。 */
const REG = 4;
/** workgroup の一辺（= GEMM_TILE / REG）。16×16 = 256 スレッド。パイプラインキーにも載る。 */
export const GEMM_WORKGROUP = GEMM_TILE / REG;
/** 生成文字列で多用する短縮名。 */
const WG = GEMM_WORKGROUP;
/** K タイル幅。旧 16×16 カーネルと同じ刻み（縮約順序を一致させる条件）。 */
const TILE_K = 16;
/** K タイルあたりの quad 数（= TILE_K / REG）。A / W 充填のスレッド割当に使う。 */
const K_QUADS = TILE_K / REG;

/**
 * m タイルに対応する workgroup の y 辺（1 スレッド 4 行なので `mTile / REG`）。
 * x 辺は n タイル 64 に固定なので常に {@link GEMM_WORKGROUP}。
 */
export const gemmWorkgroupY = (mTile: number): number => mTile / REG;

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

/** パイプラインキーのタイル判別子。v4 フラグは形状から導いた 1 ビット。 */
export const gemmKeyPart = (v4: boolean): string => `reg${GEMM_TILE}x${GEMM_TILE}${v4 ? "v4" : ""}`;

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
 *    16 MAC あたり 8 回（B の vec4 1 回 + A のスカラ 4 回）。
 * 3. 累積は f32（`acc: array<vec4<f32>, 4>` のまま）。k が大きい経路で f16 累積にすると
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
  | { readonly op: "matmul" | "bmm"; readonly v4: boolean }
  | {
    readonly op: "attention_qk" | "attention_pv";
    readonly v4: boolean;
    readonly compute?: GemmCompute;
    /** S の格納形（ADR 0023 の ①が書き ③が読むバッファ — 省略時 `"f32"`）。 */
    readonly score?: ScoreStorage;
  }
  | {
    readonly op: "linear";
    readonly v4: boolean;
    readonly weight: WeightStorage;
    readonly compute?: GemmCompute;
  }
  | {
    readonly op: "conv2d";
    readonly v4: boolean;
    readonly weight: WeightStorage;
    /** m タイルの行数（{@link GEMM_TILE} か {@link GEMM_MTILE_SMALL}）。 */
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
  v4 ? `  let k4 = dims.k / ${REG}u;\n  let n4 = dims.n / ${REG}u;\n` : "";

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

/** A タイル（`[m,k]` 行優先）の担当と行頭オフセット。どれも K タイルループ不変。 */
const prologueA = (op: GemmSpec["op"], v4: boolean): string =>
  `  // A タイルの担当（${GEMM_TILE} 行 × ${K_QUADS} quad = ${GEMM_TILE * K_QUADS} スレッド）
  let ar = tid / ${K_QUADS}u;
  let aq = tid % ${K_QUADS}u;
  let arow = wid.y * ${GEMM_TILE}u + ar;
  let arow_base = ${batchBase(op, "abase")}arow * ${v4 ? "k4" : "dims.k"};
  let sa_base = ar * ${TILE_K}u + aq * ${REG}u;`;

/**
 * A タイルの充填。範囲外は 0 で埋める（内積に寄与しないので K タイル端数でも結果は変わらない）。
 * v4 は 1 スレッドが vec4 を 1 回読み、スカラ版は 4 成分を個別に境界検査して詰める。
 *
 * `value` は読んだ 1 要素に掛ける変換式（融合 attention の半スケール / `exp(S−m)·inv`）。
 * MUST: 変換は**範囲内の要素にだけ**掛ける。範囲外の 0 に `exp(0−m)·inv` を掛けると
 * 0 でない値が内積へ混ざり、端数タイルだけが静かに誤る。
 * MUST: 変換は**成分ごとのスカラ式**で書く（`vec4` へまとめて `exp` を掛けない）。加減乗算と
 * 違って超越関数のベクトル版は実装依存で、分解経路とのビット同一の前提が崩れうる。
 * MUST: `value` 省略時の生成物は従来のバイト列そのまま（スナップショットが検出器）。
 */
const fillA = (
  name: string,
  v4: boolean,
  value?: (expr: string) => string,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string => {
  const read = (expr: string): string => value === undefined ? expr : value(expr);
  // s16 の quad 読みは**同じ quad 添字**を取る（f32 変種の添字算術を 1 文字も動かさない）
  const quad = scoreReadQuad(name, score, `arow_base + t * ${K_QUADS}u + aq`);
  const load = v4
    ? value === undefined
      ? `    if (arow < dims.m && ak0 < dims.k) {
      av = ${quad};
    }`
      : `    if (arow < dims.m && ak0 < dims.k) {
      let raw = ${quad};
      av = vec4<f32>(
        ${["x", "y", "z", "w"].map((lane) => read(`raw.${lane}`)).join(",\n        ")},
      );
    }`
    : `    if (arow < dims.m) {
      if (ak0 < dims.k) {
        av.x = ${read(`${name}[arow_base + ak0]`)};
      }
      if (ak0 + 1u < dims.k) {
        av.y = ${read(`${name}[arow_base + ak0 + 1u]`)};
      }
      if (ak0 + 2u < dims.k) {
        av.z = ${read(`${name}[arow_base + ak0 + 2u]`)};
      }
      if (ak0 + 3u < dims.k) {
        av.w = ${read(`${name}[arow_base + ak0 + 3u]`)};
      }
    }`;
  return `    // 範囲外は 0 で埋める。内積に寄与しないので端数 shape でも結果は変わらない
    let ak0 = t * ${TILE_K}u + aq * ${REG}u;
    var av = vec4<f32>(0.0);
${load}
    sa[sa_base] = ${tileScalar(compute, "av.x")};
    sa[sa_base + 1u] = ${tileScalar(compute, "av.y")};
    sa[sa_base + 2u] = ${tileScalar(compute, "av.z")};
    sa[sa_base + 3u] = ${tileScalar(compute, "av.w")};`;
};

/** dense な B タイル（`[k,n]` 行優先 — 列方向がそのまま vec4）の担当。 */
const prologueBDense = (v4: boolean): string =>
  `  // B タイルの担当（K ${TILE_K} 行 × 列 quad ${WG} = ${TILE_K * WG} スレッド）
  let bk = tid / ${WG}u;
  let bcq = tid % ${WG}u;
  ${v4 ? `let bc4 = wid.x * ${WG}u + bcq;` : `let bcol = wid.x * ${GEMM_TILE}u + bcq * ${REG}u;`}`;

const fillBDense = (
  name: string,
  op: GemmSpec["op"],
  v4: boolean,
  compute: GemmCompute = "f32",
): string =>
  `    let brow = t * ${TILE_K}u + bk;
    var bv4 = vec4<f32>(0.0);
${
    v4
      ? `    if (brow < dims.k && bc4 < n4) {
      bv4 = ${name}[${batchBase(op, "bbase")}brow * n4 + bc4];
    }`
      : `    if (brow < dims.k) {
      let brow_base = ${batchBase(op, "bbase")}brow * dims.n;
      if (bcol < dims.n) {
        bv4.x = ${name}[brow_base + bcol];
      }
      if (bcol + 1u < dims.n) {
        bv4.y = ${name}[brow_base + bcol + 1u];
      }
      if (bcol + 2u < dims.n) {
        bv4.z = ${name}[brow_base + bcol + 2u];
      }
      if (bcol + 3u < dims.n) {
        bv4.w = ${name}[brow_base + bcol + 3u];
      }
    }`
  }
    sb[bk * ${WG}u + bcq] = ${tileQuad(compute, "bv4")};`;

/**
 * linear の B タイル（重み `[n,k]` — 連続方向が k）の担当。
 *
 * 1 スレッドが **k 連続 4 要素**を読み、**共有メモリ側で転置して置く**。内積ループを dense と
 * 共有できるのはこの転置のおかげ。i8 の scale は担当チャネル `wcol` のもので、K タイル
 * ループ不変なのでここで 1 度だけ束ねる（ADR 0019 のループ不変巻き上げ）。端タイルでは
 * `wcol >= n` になりうるが WGSL の境界付きアクセスで安全で、読んだ値は `wcol < n` の
 * ときしか `sb` に載らない。
 */
const prologueBLinear = (weight: WeightStorage): string =>
  `  // W タイルの担当（${GEMM_TILE} 出力チャネル × ${K_QUADS} quad = ${
    GEMM_TILE * K_QUADS
  } スレッド）
  let wc = tid / ${K_QUADS}u;
  let wq = tid % ${K_QUADS}u;
  let wcol = wid.x * ${GEMM_TILE}u + wc;${weightScaleWgsl(weight, "wcol", "  ")}
  let wrow_base = wcol * dims.k;
  // 共有メモリ側で転置して置く（列 quad = wc / ${REG}・成分 = wc % ${REG}）
  let wsq = wc / ${REG}u;
  let wsl = wc % ${REG}u;
  let sb_base = (wq * ${REG}u) * ${WG}u + wsq;`;

const fillBLinear = (
  name: string,
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
): string =>
  `    let wk0 = t * ${TILE_K}u + wq * ${REG}u;
    var wv = vec4<f32>(0.0);
${
    v4
      ? `    if (wcol < dims.n && wk0 < dims.k) {
      wv = ${weightRead4(name, weight, "wrow_base + wk0", WEIGHT_SCALE_VAR)};
    }`
      : `    if (wcol < dims.n) {
      let wbase = wrow_base + wk0;
      if (wk0 < dims.k) {
        wv.x = ${weightRead(name, weight, "wbase", WEIGHT_SCALE_VAR)};
      }
      if (wk0 + 1u < dims.k) {
        wv.y = ${weightRead(name, weight, "wbase + 1u", WEIGHT_SCALE_VAR)};
      }
      if (wk0 + 2u < dims.k) {
        wv.z = ${weightRead(name, weight, "wbase + 2u", WEIGHT_SCALE_VAR)};
      }
      if (wk0 + 3u < dims.k) {
        wv.w = ${weightRead(name, weight, "wbase + 3u", WEIGHT_SCALE_VAR)};
      }
    }`
  }
    sb[sb_base][wsl] = ${tileScalar(compute, "wv.x")};
    sb[sb_base + ${WG}u][wsl] = ${tileScalar(compute, "wv.y")};
    sb[sb_base + ${2 * WG}u][wsl] = ${tileScalar(compute, "wv.z")};
    sb[sb_base + ${3 * WG}u][wsl] = ${tileScalar(compute, "wv.w")};`;

/**
 * 融合 attention ① の B タイル（k `[B·H, N, D]` — 連続方向が D）の担当。
 *
 * 構造は linear の重み読み（`[n,k]` を k 連続で読み、共有側で転置して置く）と同一で、
 * 違いは **バッチ base が要る**ことと **重み格納の変種を持たない**（活性は常に f32）ことだけ。
 * この共有のおかげで、旧経路の `permute` による kᵀ 実体化がまるごと消える。
 */
const prologueBAttentionQk = (): string =>
  `  // k タイルの担当（${GEMM_TILE} 列（N）× ${K_QUADS} quad = ${GEMM_TILE * K_QUADS} スレッド）。
  // k は [N,D] のまま読み、**共有メモリ側で転置して置く**（linear の重み読みと同じ構造）
  let wc = tid / ${K_QUADS}u;
  let wq = tid % ${K_QUADS}u;
  let wcol = wid.x * ${GEMM_TILE}u + wc;
  let krow_base = bbase + wcol * dims.k;
  let wsq = wc / ${REG}u;
  let wsl = wc % ${REG}u;
  let sb_base = (wq * ${REG}u) * ${WG}u + wsq;`;

const fillBAttentionQk = (v4: boolean, compute: GemmCompute = "f32"): string =>
  `    let wk0 = t * ${TILE_K}u + wq * ${REG}u;
    var wv = vec4<f32>(0.0);
${
    v4
      ? `    if (wcol < dims.n && wk0 < dims.k) {
      wv = k[(krow_base + wk0) >> 2u];
    }`
      : `    if (wcol < dims.n) {
      if (wk0 < dims.k) {
        wv.x = k[krow_base + wk0];
      }
      if (wk0 + 1u < dims.k) {
        wv.y = k[krow_base + wk0 + 1u];
      }
      if (wk0 + 2u < dims.k) {
        wv.z = k[krow_base + wk0 + 2u];
      }
      if (wk0 + 3u < dims.k) {
        wv.w = k[krow_base + wk0 + 3u];
      }
    }`
  }
    // 半スケール契約（ADR 0023）: scale は q 側と k 側の**両方**へ掛ける。範囲外は 0 のままで
    // 0 · scale = 0 なので端数タイルの結論は変わらない
    wv = wv * dims.scale;
    sb[sb_base][wsl] = ${tileScalar(compute, "wv.x")};
    sb[sb_base + ${WG}u][wsl] = ${tileScalar(compute, "wv.y")};
    sb[sb_base + ${2 * WG}u][wsl] = ${tileScalar(compute, "wv.z")};
    sb[sb_base + ${3 * WG}u][wsl] = ${tileScalar(compute, "wv.w")};`;

/**
 * 融合 attention ③ が使う行統計の束縛（K タイルループ不変なので 1 度だけ引く）。
 *
 * `stats` は `[B·H·M, 2]` で `[0]` が行の最大値 `m`・`[1]` が `1/Σexp(S−m)`。②（行統計）が
 * 現行 softmax のパス①②と同一の縮約で作る。
 *
 * MUST: 行の添字は `rbase + arow`（バッチごとの行オフセット）。`arow` だけで引くと
 * B·H ≥ 2 で全バッチが 0 番目のバッチの統計を使い、**B=H=1 のテストでは値に出ない**。
 */
const prologueAttentionStats = (): string =>
  `  // 端タイルでは arow >= m がありうるので添字を 0 へ倒す（読んだ値は arow < m の枝でしか
  // 使われない）。範囲外の stats を読むこと自体は WGSL の境界付きアクセスで安全
  let stat_at = select(0u, (rbase + arow) * 2u, arow < dims.m);
  let row_max = stats[stat_at];
  let row_inv = stats[stat_at + 1u];`;

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
 */
const store = (
  name: string,
  op: GemmSpec["op"],
  v4: boolean,
  bias: boolean,
  mTile = GEMM_TILE,
  outF16 = false,
  score: ScoreStorage = "f32",
): string => {
  const base = batchBase(op, "cbase");
  const rows = `  let orow0 = wid.y * ${mTile}u + lid.y * ${REG}u;`;
  if (v4) {
    const biasBind = bias
      ? `
    // n % ${REG} == 0 かつ ocq < n4 なので oc + ${REG - 1} < n（bias は常に f32 — ADR 0006）
    let oc = ocq * ${REG}u;
    let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);`
      : "";
    const value = bias ? "acc[i] + biasv" : outF16 ? "vec4<f16>(acc[i])" : "acc[i]";
    return `  let ocq = wid.x * ${WG}u + lid.x;
${rows}
  if (ocq < n4) {${biasBind}
    for (var i = 0u; i < ${REG}u; i = i + 1u) {
      let orow = orow0 + i;
      if (orow < dims.m) {
        ${scoreStoreQuad(name, score, `${base}orow * n4 + ocq`, value)}
      }
    }
  }`;
  }
  const write = (component: string, offset: string): string =>
    `${name}[obase + ocol${offset}] = ${
      outF16 ? `f16(acc[i].${component})` : `acc[i].${component}`
    }${bias ? ` + bias[ocol${offset}]` : ""};`;
  return `  let ocol = wid.x * ${GEMM_TILE}u + lid.x * ${REG}u;
${rows}
  for (var i = 0u; i < ${REG}u; i = i + 1u) {
    let orow = orow0 + i;
    if (orow < dims.m) {
      let obase = ${base}orow * dims.n;
      if (ocol < dims.n) {
        ${write("x", "")}
      }
      if (ocol + 1u < dims.n) {
        ${write("y", " + 1u")}
      }
      if (ocol + 2u < dims.n) {
        ${write("z", " + 2u")}
      }
      if (ocol + 3u < dims.n) {
        ${write("w", " + 3u")}
      }
    }
  }`;
};

/**
 * `acc` の初期値。既定は 0 で、conv2d だけが **bias-first**（ADR 0024 の MUST）で上書きする。
 * MUST: 既定の文字列を動かさない（既存 4 op の生成バイト列がここに掛かっている）。
 */
const ACC_ZERO = `  var acc = array<vec4<f32>, ${REG}>(
    vec4<f32>(0.0),
    vec4<f32>(0.0),
    vec4<f32>(0.0),
    vec4<f32>(0.0),
  );`;

/**
 * head / Dims 追加欄 / prologue / fillA / fillB / store の断片を差し込んだ 1 本のシェーダ。
 *
 * `dimsExtra` は uniform の 4 語目以降（融合 attention ① の `scale` / conv2d の幾何 13 語）。
 * uniform struct は 16 バイト整列なので 4 語目は既に確保済みで、既存 3 op では空文字
 * （生成物は 1 バイトも動かない）。`accInit` も同様に省略時は従来の 0 初期化そのまま。
 */
const skeleton = (
  header: string,
  bindings: string,
  loader: string,
  prologue: string,
  fillTiles: string,
  storeTile: string,
  dimsExtra = "",
  accInit = ACC_ZERO,
  mTile = GEMM_TILE,
  compute: GemmCompute = "f32",
): string =>
  `${header}${
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
// 共有 A タイル（${mTile} 行 × K ${TILE_K}・スカラ格納）と
// 共有 B タイル（K ${TILE_K} × 列 quad ${WG}・列方向を vec4 に束ねた形）${
    compute === "f16"
      ? `。f16 変種は共有バイトが半分
// （${mTile * TILE_K * 4 + TILE_K * WG * 16} B → ${
        mTile * TILE_K * 2 + TILE_K * WG * 8
      } B / WG）— 期待利得はこの 1 機序に全て乗る`
      : ""
  }
var<workgroup> sa: array<${tileScalarType(compute)}, ${mTile * TILE_K}>;
var<workgroup> sb: array<${tileQuadType(compute)}, ${TILE_K * WG}>;

@compute @workgroup_size(${WG}, ${gemmWorkgroupY(mTile)})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.y * ${WG}u + lid.x;
${prologue}
  // ループ条件は uniform（dims は uniform バッファ）— 内側の workgroupBarrier が
  // WGSL の一様性要件を満たすために必要
  let tiles = (dims.k + ${TILE_K - 1}u) / ${TILE_K}u;
${accInit}
  for (var t = 0u; t < tiles; t = t + 1u) {
${fillTiles}
    workgroupBarrier();
    // 共有ロード 5 回（B の vec4 1 + A のスカラ 4）で ${REG * REG} MAC。縮約は k 昇順の逐次で、
    // 1 出力要素あたりの加算順序は 16×16 の 1 スレッド 1 出力と完全に一致する。${
    compute === "f16"
      ? `
    // MUST: f32 への拡幅は**レジスタロード時に 1 回**（${REG + REG} 回 / ${
        REG * REG
      } MAC）。MAC ごとに
    // f32(av * bv) と書くと変換が ${
        REG * REG
      } 回に増え、しかも積が f16 精度に落ちる（ADR 0028 の丸め列 2）。
    // f16 → f32 の拡幅は厳密なので、値は「入力を f16 に丸めた f32 変種」と 1 ビットも違わない。`
      : ""
  }
    for (var kk = 0u; kk < ${TILE_K}u; kk = kk + 1u) {
      let bv = ${
    compute === "f16" ? `vec4<f32>(sb[kk * ${WG}u + lid.x])` : `sb[kk * ${WG}u + lid.x]`
  };
      for (var i = 0u; i < ${REG}u; i = i + 1u) {
        acc[i] = acc[i] + ${
    compute === "f16"
      ? `f32(sa[(lid.y * ${REG}u + i) * ${TILE_K}u + kk])`
      : `sa[(lid.y * ${REG}u + i) * ${TILE_K}u + kk]`
  } * bv;
      }
    }
    workgroupBarrier();
  }
${storeTile}
}
`;

const denseWgsl = (op: "matmul" | "bmm", v4: boolean): string => {
  const element = v4 ? "vec4<f32>" : "f32";
  const signature = op === "bmm" ? "a[b,m,k] · b[b,k,n]" : "a[m,k] · b[k,n]";
  return skeleton(
    `// karume ${op} (${signature}, f32, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${
      v4 ? " + vec4" : ""
    })`,
    `@group(0) @binding(1) var<storage, read> a: array<${element}>;
@group(0) @binding(2) var<storage, read> b: array<${element}>;
@group(0) @binding(3) var<storage, read_write> c: array<${element}>;`,
    "",
    `${quadDims(v4)}${op === "bmm" ? batchPrologue(op, v4) : ""}${prologueA(op, v4)}
${prologueBDense(v4)}`,
    `${fillA("a", v4)}
${fillBDense("b", op, v4)}`,
    store("c", op, v4, false),
  );
};

/**
 * 融合 attention ①（QK）— `S[b,m,n] = Σ_d (q·scale)[b,m,d] · (k·scale)[b,n,d]`（ADR 0023）。
 *
 * A は q（`[batch,M,D]` 連続）で dense の A 充填そのまま、B は k（`[batch,N,D]` 連続）を
 * linear の重み読みと同じ「k 連続で読んで共有側で転置」構造で読む。**縮約は K 昇順・
 * K タイル 16** で現行 bmm と 1 ビット違わない。
 */
const attentionQkWgsl = (
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string => {
  const f16 = compute === "f16";
  assertScoreStorageSupported("attention_qk", score, v4, f16);
  return skeleton(
    `// karume attention_qk (S[b,m,n] = (q·scale)[b,m,d] · (k·scale)[b,n,d]ᵀ, f32${
      f16 ? ", f16 タイル計算・S も f16" : ""
    }${scoreNote(score)}, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> q: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(2) var<storage, read> k: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> s: array<${
      scoreArrayType(score, v4 ? tileQuadType(compute) : tileScalarType(compute))
    }>;`,
    scoreStoreWgsl("s", score),
    `${quadDims(v4)}${batchPrologue("attention_qk", v4)}${prologueA("attention_qk", v4)}
${prologueBAttentionQk()}`,
    `${fillA("q", v4, (expr) => `${expr} * dims.scale`, compute)}
${fillBAttentionQk(v4, compute)}`,
    store("s", "attention_qk", v4, false, GEMM_TILE, f16, score),
    "  scale: f32,\n",
    ACC_ZERO,
    GEMM_TILE,
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
  v4: boolean,
  compute: GemmCompute = "f32",
  score: ScoreStorage = "f32",
): string => {
  const f16 = compute === "f16";
  assertScoreStorageSupported("attention_pv", score, v4, f16);
  // MUST: S を f32 へ広げてから `exp(S−m)·inv` を評価する（拡幅は厳密なので値は動かない）。
  // f16 のまま `exp` を評価すると指数関数が f16 精度になり、丸め列が 1 段増える。
  const probability = (expr: string): string =>
    `exp(${f16 ? `f32(${expr})` : expr} - row_max) * row_inv`;
  return skeleton(
    `// karume attention_pv (O[b,m,d] = P[b,m,n] · v[b,n,d], P = exp(S − m)·inv は非実体化, f32${
      f16 ? ", f16 タイル計算・S は f16" : ""
    }${scoreNote(score)}, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> s: array<${
      scoreArrayType(score, v4 ? tileQuadType(compute) : tileScalarType(compute))
    }>;
@group(0) @binding(2) var<storage, read> v: array<${v4 ? "vec4<f32>" : "f32"}>;
@group(0) @binding(3) var<storage, read> stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<${v4 ? "vec4<f32>" : "f32"}>;`,
    scoreQuadLoaderWgsl("s", score),
    `${quadDims(v4)}${batchPrologue("attention_pv", v4)}${prologueA("attention_pv", v4)}
${prologueAttentionStats()}
${prologueBDense(v4)}`,
    `${fillA("s", v4, probability, compute, score)}
${fillBDense("v", "attention_pv", v4, compute)}`,
    store("o", "attention_pv", v4, false),
    "",
    ACC_ZERO,
    GEMM_TILE,
    compute,
  );
};

const linearVariantWgsl = (
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
    `// karume linear (x[m,k] · wᵀ[k,n] + bias[n], f32${weightNote(weight)}${
      compute === "f16" ? ", f16 タイル計算" : ""
    }, レジスタ ${GEMM_TILE}x${GEMM_TILE} タイル${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> x: array<${element}>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight, v4)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${element}>;`,
    weightLoaderWgsl("w", weight, LINEAR_SCALE_BINDING, v4),
    `${quadDims(v4)}${prologueA("linear", v4)}
${prologueBLinear(weight)}`,
    `${fillA("x", v4, undefined, compute)}
${fillBLinear("w", weight, v4, compute)}`,
    store("out", "linear", v4, true),
    "",
    ACC_ZERO,
    GEMM_TILE,
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
 * 端タイルでは `bias0 + i >= m` を読みうるが、その `acc[i]` は `store` の行ガードで捨てられる
 * （範囲外の storage 読み自体は WGSL の境界付きアクセスで安全）。
 */
const conv2dAccInit = (mTile: number): string =>
  `  let bias0 = wid.y * ${mTile}u + lid.y * ${REG}u;
  var acc = array<vec4<f32>, ${REG}>(
    vec4<f32>(bias[bias0]),
    vec4<f32>(bias[bias0 + 1u]),
    vec4<f32>(bias[bias0 + 2u]),
    vec4<f32>(bias[bias0 + 3u]),
  );`;

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
const prologueAConv2d = (weight: WeightStorage, mTile: number): string =>
  `  // A タイルの担当（${mTile} 行 × ${K_QUADS} quad = ${mTile * K_QUADS} スレッド）
  let ar = tid / ${K_QUADS}u;
  let aq = tid % ${K_QUADS}u;
  let arow = wid.y * ${mTile}u + ar;${weightScaleWgsl(weight, "arow", "  ")}
  let arow_base = arow * dims.k;
  let sa_base = ar * ${TILE_K}u + aq * ${REG}u;`;

/** conv2d の A タイル充填（{@link fillA} と {@link fillBLinear} を合成した形）。 */
const fillAConv2d = (name: string, weight: WeightStorage, v4: boolean): string =>
  `    // 範囲外は 0 で埋める。内積に寄与しないので K 端数でも結果は変わらない
    let ak0 = t * ${TILE_K}u + aq * ${REG}u;
    var av = vec4<f32>(0.0);
${
    v4
      ? `    if (arow < dims.m && ak0 < dims.k) {
      av = ${weightRead4(name, weight, "arow_base + ak0", WEIGHT_SCALE_VAR)};
    }`
      : `    if (arow < dims.m) {
      let abase = arow_base + ak0;
      if (ak0 < dims.k) {
        av.x = ${weightRead(name, weight, "abase", WEIGHT_SCALE_VAR)};
      }
      if (ak0 + 1u < dims.k) {
        av.y = ${weightRead(name, weight, "abase + 1u", WEIGHT_SCALE_VAR)};
      }
      if (ak0 + 2u < dims.k) {
        av.z = ${weightRead(name, weight, "abase + 2u", WEIGHT_SCALE_VAR)};
      }
      if (ak0 + 3u < dims.k) {
        av.w = ${weightRead(name, weight, "abase + 3u", WEIGHT_SCALE_VAR)};
      }
    }`
  }
    sa[sa_base] = av.x;
    sa[sa_base + 1u] = av.y;
    sa[sa_base + 2u] = av.z;
    sa[sa_base + 3u] = av.w;`;

/**
 * conv2d の B タイル（`Xcol[k,n]` の暗黙 gather）の担当。割り当ては dense と同じ。
 *
 * MUST: バッチは `wid.z`（{@link BATCHED_OPS} の doc）。`xbase` は x の**チャネル平面**単位、
 * `cbase` は出力要素（v4 では quad）単位で数える。
 */
const prologueBConv2d = (v4: boolean, mTile: number): string =>
  `  // バッチは workgroup 単位で一様（z 軸 1 つが 1 バッチの出力平面 [Cout, Hout·Wout]）
  let xbase = wid.z * dims.channels_in;
  let cbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};
  // B タイルの担当（K ${TILE_K} 行 × 列 quad ${WG} = ${TILE_K * WG} ${
    mTile === GEMM_TILE ? "スレッド" : `要素 / ${WG * gemmWorkgroupY(mTile)} スレッド`
  }）
  let ${mTile === GEMM_TILE ? "bk" : "bkr"} = tid / ${WG}u;
  let bcq = tid % ${WG}u;
  ${v4 ? `let bc4 = wid.x * ${WG}u + bcq;` : `let bcol = wid.x * ${GEMM_TILE}u + bcq * ${REG}u;`}
  // K タイルループ不変（平坦 k を (ic, kh, kw) へ割るための刻み）
  let khw = dims.kernel_h * dims.kernel_w;`;

/**
 * 平坦 k → `(ic, kh, kw)` の分解と入力座標のオフセット。
 *
 * MUST: 平坦 k の昇順が直接カーネルの `(ic, kh, kw)` 三重昇順と一致することが、
 * K タイル 16 昇順 → ビット同一の土台（ADR 0024）。
 */
const CONV2D_K_DECODE = `      let ic = brow / khw;
      let kr = brow % khw;
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
const fillBConv2d = (v4: boolean, mTile: number): string => {
  const body = fillBConv2dBody(v4);
  if (mTile === GEMM_TILE) return body;
  // 32 行タイルは 128 スレッドしか無いので、sb（K 16 × 列 quad 16 = 256）は 2 パスで埋める。
  // 列（`bcq`）はパス不変で、k 行だけが `bkr` → `bkr + 8` と進む（担当は重ならない）。
  const rows = gemmWorkgroupY(mTile);
  const passes = (TILE_K * WG) / (WG * rows);
  return `    // ${mTile} 行タイルは ${WG * rows} スレッドなので B タイル ${
    TILE_K * WG
  } 要素を ${passes} パスで埋める
    for (var bp = 0u; bp < ${passes}u; bp = bp + 1u) {
      let bk = bp * ${rows}u + bkr;
${indentBlock(body, "  ")}
    }`;
};

/** 生成断片の再字下げ（B タイル充填を `for` の内側へ入れるときだけ使う）。 */
const indentBlock = (text: string, pad: string): string =>
  text.split("\n").map((line) => line === "" ? line : `${pad}${line}`).join("\n");

const fillBConv2dBody = (v4: boolean): string =>
  `    let brow = t * ${TILE_K}u + bk;
    var bv4 = vec4<f32>(0.0);
${
    v4
      ? `    if (brow < dims.k && bc4 < n4) {
${CONV2D_K_DECODE}
      let xc = xbase + ic;
      // quad の 4 列は同じ出力行の連続 ox（v4 の条件）なので x 側も連続に読める
      let n0 = bc4 * ${REG}u;
      let iy = i32((n0 / dims.width_out) * dims.stride_h) + ky;
      let ix0 = i32((n0 % dims.width_out) * dims.stride_w) + kx;
      if (iy >= 0 && u32(iy) < dims.height_in && ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {
        let base = (xc * dims.height_in + u32(iy)) * dims.width_in + u32(ix0);
        bv4 = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
      } else {
        // 画像端と padding 域だけがここに来る（範囲外は 0 — xcol の MUST）
        bv4 = vec4<f32>(
          xcol(xc, ky, kx, n0),
          xcol(xc, ky, kx, n0 + 1u),
          xcol(xc, ky, kx, n0 + 2u),
          xcol(xc, ky, kx, n0 + 3u),
        );
      }
    }`
      : `    if (brow < dims.k) {
${CONV2D_K_DECODE}
      let xc = xbase + ic;
      if (bcol < dims.n) {
        bv4.x = xcol(xc, ky, kx, bcol);
      }
      if (bcol + 1u < dims.n) {
        bv4.y = xcol(xc, ky, kx, bcol + 1u);
      }
      if (bcol + 2u < dims.n) {
        bv4.z = xcol(xc, ky, kx, bcol + 2u);
      }
      if (bcol + 3u < dims.n) {
        bv4.w = xcol(xc, ky, kx, bcol + 3u);
      }
    }`
  }
    sb[bk * ${WG}u + bcq] = bv4;`;

/**
 * conv2d の implicit GEMM（ADR 0024）— `C[Cout, N] = W[Cout, K] × Xcol[K, N]`。
 *
 * A は重み（平坦化で `[M,K]` 行優先そのもの・weight-storage 3 変種）、B は x の暗黙 gather、
 * store は 1 バッチぶんの `[Cout][Hout·Wout]` 行優先 = **NCHW の平面そのもの**なので後段の
 * レイアウト変換は要らない（バッチは z 軸 — {@link BATCHED_OPS}）。
 * bias は `acc` の初期値（{@link conv2dAccInit}）。m タイルは 64 行 / 32 行の 2 変種
 * （{@link GEMM_MTILE_SMALL} — 出力の**担当割り**だけが変わり、数値経路は共通）。
 */
const conv2dIgemmWgsl = (weight: WeightStorage, v4: boolean, mTile: number): string =>
  skeleton(
    `// karume conv2d (x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32${
      weightNote(weight)
    }, implicit GEMM レジスタ ${mTile}x${GEMM_TILE} タイル${v4 ? " + vec4" : ""})`,
    `@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<${weightArrayType(weight, v4)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<${v4 ? "vec4<f32>" : "f32"}>;`,
    `${weightLoaderWgsl("w", weight, LINEAR_SCALE_BINDING, v4)}${CONV2D_XCOL_WGSL}`,
    `${v4 ? `  let n4 = dims.n / ${REG}u;\n` : ""}${prologueAConv2d(weight, mTile)}
${prologueBConv2d(v4, mTile)}`,
    `${fillAConv2d("w", weight, v4)}
${fillBConv2d(v4, mTile)}`,
    store("out", "conv2d", v4, false, mTile),
    CONV2D_DIMS_EXTRA,
    conv2dAccInit(mTile),
    mTile,
  );

/** 生成入力 1 つから WGSL 1 本。同じ入力からは常にバイト単位で同じ文字列が出る。 */
export const gemmWgsl = (spec: GemmSpec): string => {
  switch (spec.op) {
    case "linear":
      return linearVariantWgsl(spec.weight, spec.v4, spec.compute);
    case "conv2d":
      return conv2dIgemmWgsl(spec.weight, spec.v4, spec.mTile);
    case "attention_qk":
      return attentionQkWgsl(spec.v4, spec.compute, spec.score);
    case "attention_pv":
      return attentionPvWgsl(spec.v4, spec.compute, spec.score);
    default:
      return denseWgsl(spec.op, spec.v4);
  }
};

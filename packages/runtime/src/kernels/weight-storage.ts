/**
 * 重み格納の変種（`w=f32` / `w=f16` / `w=i8`）— 融合 5 カーネル（linear / conv1d / conv2d /
 * conv_transpose1d / embedding）が共有する唯一の生成部品（ADR 0018 / 0019）。
 *
 * f16 変種は重みバッファを **`array<u32>`** で束縛し、`unpack2x16float` で 2 要素ずつ展開する
 * （core WGSL・optional feature 依存ゼロ）。添字 `i` の値は `unpack2x16float(w[i >> 1])[i & 1]`
 * で、**平坦添字の偶奇**だけが対を選ぶ。
 *
 * i8 変種も同じく `array<u32>` 束縛で、`unpack4xI8` が 4 要素を 1 語から取り出す。値は
 * `f32(unpack4xI8(w[i >> 2])[i & 3]) * scale[出力チャネル]` — **scale は縮約の外に出さず
 * 要素ごとに掛ける**（ADR 0019: `(Σ x·q)·s` 形は乗算が減る代わりに CPU 展開とのビット一致を
 * 失う）。scale は **weight と対の追加束縛**（{@link weightLoaderWgsl} が宣言を出す）で、
 * 出力チャネルごとの f32 が平坦に並ぶ。
 *
 * MUST: 語と位置の選択は**平坦添字**から作る。行内の相対添字で偶奇（f16）や 4 剰余（i8）を
 * 取ると、行長が 2 / 4 の倍数のときだけ偶然一致して数値が合う（行の先頭が常に語境界に来る
 * ため）。行長が奇数 / 4 の倍数でないテストが唯一の検出器になる
 * （tests/gpu_f16_weights_test.ts と tests/gpu_i8_weights_test.ts の MUST）。
 *
 * MUST: `w=f32` / `w=f16` 変種の生成物は**バイト単位で従来のまま**。キーの格納判別子は f32 で
 * 空文字になり、WGSL の挿入点（{@link weightLoaderWgsl} / {@link weightScaleWgsl}）は
 * f32 / f16 それぞれで従来と同じ文字列を返す。既存のスナップショット
 * （tests/fixtures/wgsl/*.wgsl）とパイプラインキャッシュの同一性がこれに掛かっている。
 *
 * **quad 版**（`quad: true`）は GEMM の v4 経路（src/kernels/gemm.ts）専用の opt-in で、
 * 4 要素をまとめて `vec4<f32>` に展開する。MUST: 既定は `false` — 無条件に足すと上の
 * バイト不変が壊れ、conv1d / conv2d / conv_transpose1d / embedding の生成物が一斉に動く。
 */

/** 重みスロットの格納形。意味論はどれも f32（計算は常に f32 — ADR 0006）。 */
export type WeightStorage = "f32" | "f16" | "i8";

/** 変種の全数（スナップショットと縮退ハーネスの網羅を機械的に回すための列挙）。 */
export const WEIGHT_STORAGES: readonly WeightStorage[] = ["f32", "f16", "i8"];

/**
 * パイプラインキーに載せる格納判別子。
 *
 * MUST: f32 は空文字。既存キーに判別子を足すと、同じカーネルが別キーになってブラウザ側の
 * 暗黙シェーダキャッシュを取り直すうえ、キー固定のテストが一斉に動く。
 */
export const weightKeyPart = (storage: WeightStorage): string =>
  storage === "f32" ? "" : storage === "f16" ? ":wf16" : ":wi8";

/**
 * 重みバッファの WGSL 要素型（f16 は 2 要素・i8 は 4 要素を 1 語に詰めた格納なので u32）。
 *
 * `quad` は f32 だけを `vec4<f32>` へ変える（f16 / i8 は pack 済みで配列型が変わらない）。
 */
export const weightArrayType = (storage: WeightStorage, quad = false): string =>
  storage === "f32" ? (quad ? "vec4<f32>" : "f32") : "u32";

/** WGSL 先頭コメントに足す但し書き（f32 は空 — 既存バイト列を保つ）。 */
export const weightNote = (storage: WeightStorage): string =>
  storage === "f32" ? "" : storage === "f16" ? ", 重み f16 格納" : ", 重み i8 格納";

/**
 * i8 変種で per-channel scale を束ねる局所変数の**既定の**名前。
 *
 * {@link weightScaleWgsl} が縮約の外で 1 度だけ束縛し、{@link weightRead} の第 4 引数として
 * 各カーネルが渡す。カーネルが束縛を忘れると WGSL のコンパイルが「未定義の識別子」で落ちる
 * （沈黙しない）。1 スレッドが複数チャネルを担当する形（GEMM 骨格の充填スロット）では
 * 名前が 1 本では足りないので、{@link weightScaleWgsl} の第 4 引数で別名を渡せる。
 */
export const WEIGHT_SCALE_VAR = "wscale_v";

/**
 * f16 / i8 変種の展開関数（と i8 の scale 束縛宣言）。束縛の直後に置く前提で、**前後 1 行ずつ
 * 空行が入る形**に整えてある（f32 では空文字になり、従来の空行 1 本だけが残る）。
 *
 * `scaleBinding` は i8 のときだけ使う — 出力束縛の**次の番号**を渡す。f32 / f16 は宣言を
 * 出さないので番号は消費されず、既存 2 変種の生成物は 1 バイトも動かない。
 *
 * `quad` は {@link weightRead4} と対の quad 展開（`dequant4`）へ切り替える opt-in で、
 * スカラ版 `dequant` は出さない（v4 経路は quad 読みしか使わない）。
 * MUST: 既定 `false` — 無条件に `dequant4` を足すと既存 4 op の生成物が動く。
 */
export const weightLoaderWgsl = (
  name: string,
  storage: WeightStorage,
  scaleBinding: number,
  quad = false,
): string => {
  if (storage === "f32") return "";
  if (storage === "f16") {
    // MUST: quad 版は**平坦添字 i が 4 の倍数**であることに依存する（語をまたがないので
    // u32 2 語でちょうど 4 要素）。行頭が語境界に来ることへの依存ではないので、ADR 0018 の
    // 偶奇の罠とは別物 — スカラ経路の検出器（行長が 2 の倍数でないテスト）は無傷のまま。
    return quad
      ? `
// f16 格納の quad 展開: 要素 i..i+3 = ${name}[i / 2] と ${name}[i / 2 + 1] の 2 語
fn dequant4(i: u32) -> vec4<f32> {
  let lo = unpack2x16float(${name}[i >> 1u]);
  let hi = unpack2x16float(${name}[(i >> 1u) + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}
`
      : `
// f16 格納の展開: 要素 i = unpack2x16float(${name}[i / 2])[i % 2]（平坦添字の偶奇で対を選ぶ）
fn dequant(i: u32) -> f32 {
  let pair = unpack2x16float(${name}[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}
`;
  }
  // MUST: quad 版でも scale は**成分ごとの f32 乗算**（`vec4 * scalar`）。スカラ経路の
  // `f32(q) * s` と同一の演算・同一の丸めで、ADR 0019 の「scale は縮約の外へ出さない」を保つ。
  return quad
    ? `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// i8 格納の quad 展開: 要素 i..i+3 = f32(unpack4xI8(${name}[i / 4])) · scale（1 語で 4 要素）
// MUST: i は 4 の倍数（v4 経路からのみ呼ぶ）— 語をまたぐと 4 要素が別行に散る
fn dequant4(i: u32, scale: f32) -> vec4<f32> {
  return vec4<f32>(unpack4xI8(${name}[i >> 2u])) * scale;
}
`
    : `
@group(0) @binding(${scaleBinding}) var<storage, read> wscale: array<f32>;

// i8 格納の展開: 要素 i = f32(unpack4xI8(${name}[i / 4])[i % 4]) · scale
// （平坦添字で語と位置を割る。scale は出力チャネルごと — ADR 0019）
fn dequant(i: u32, scale: f32) -> f32 {
  return f32(unpack4xI8(${name}[i >> 2u])[i & 3u]) * scale;
}
`;
};

/**
 * 出力チャネル `channel` の scale を**縮約の外で 1 度だけ**読む束縛（f32 / f16 は空文字）。
 *
 * 行末に置く前提で先頭に改行が入る（f32 / f16 では空文字になり、従来の行がそのまま残る）。
 * `indent` は挿入先の字下げをそのまま渡す。
 *
 * `variable` は束縛する局所変数の名前。MUST: 既定は {@link WEIGHT_SCALE_VAR} — 1 スレッド
 * 1 チャネルの 4 カーネル（conv1d / conv2d 直接 / conv_transpose1d / embedding）の生成物を
 * 1 バイトも動かさないための既定で、スナップショット（tests/fixtures/wgsl/）が検出器。
 * 複数チャネルを担当する側（GEMM 骨格の充填スロット）だけがスロットごとの別名を渡す。
 */
export const weightScaleWgsl = (
  storage: WeightStorage,
  channel: string,
  indent: string,
  variable: string = WEIGHT_SCALE_VAR,
): string =>
  storage === "i8"
    ? `
${indent}// 出力チャネルの scale はループ不変 — 重みの要素ごとに引き直さない（ADR 0019）
${indent}let ${variable} = wscale[${channel}];`
    : "";

/**
 * 重み 1 要素の読み出し式（f32 は直接添字・f16 / i8 は展開関数）。
 *
 * `scale` は**出力チャネルの scale 式**で、i8 変種だけが使う（{@link weightScaleWgsl} が
 * 束縛した {@link WEIGHT_SCALE_VAR} を渡す）。f32 / f16 は無視するので、両変種の生成物は
 * バイト単位で従来のまま。
 */
export const weightRead = (
  name: string,
  storage: WeightStorage,
  index: string,
  scale: string,
): string =>
  storage === "f32"
    ? `${name}[${index}]`
    : storage === "f16"
    ? `dequant(${index})`
    : `dequant(${index}, ${scale})`;

/**
 * 重み **4 要素**（平坦添字 `index`..`index+3`）の読み出し式。
 *
 * MUST: `index` は 4 の倍数（GEMM の v4 経路からのみ呼ぶ）。f32 は `vec4<f32>` 配列の quad
 * 添字、f16 / i8 は {@link weightLoaderWgsl} の quad 版が出す `dequant4` に落ちる。
 */
export const weightRead4 = (
  name: string,
  storage: WeightStorage,
  index: string,
  scale: string,
): string =>
  storage === "f32"
    ? `${name}[(${index}) >> 2u]`
    : storage === "f16"
    ? `dequant4(${index})`
    : `dequant4(${index}, ${scale})`;

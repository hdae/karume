/**
 * attention スコア S の**格納形**（`s=f32` / `s=f16`）— 融合 attention の 3 カーネル
 * （①QK が書き ②行統計 / ③PV が読む — ADR 0023）だけが共有する生成部品。
 * `SessionOptions.attentionScoreStorage` の opt-in で立つ。
 *
 * ## 重み格納（{@link "./weight-storage.ts"}）の鏡像である点と、決定的に違う点
 *
 * f16 変種は S を **`array<u32>`** で束縛し、`unpack2x16float` / `pack2x16float` で 2 要素ずつ
 * 出し入れする（core WGSL・optional feature 依存ゼロ — ADR 0018 の重み f16 と同じ機序）。
 * したがって `attentionCompute: "a8"` が **`shader-f16` を要求しない**という規律（ADR 0030
 * 決定 1）は無傷のまま、i8a8 と S f16 を直交して組める。
 *
 * **違うのは書き側を持つこと**。重み f16 は読み専用だったので `unpack2x16float` が無料だったが、
 * S は ①QK が書く。1 語に 2 要素を詰めると「1 スレッドが偶数境界の 2 要素を同時に書く」制約が
 * 掛かり、これを満たすのは GEMM 骨格の **v4 経路だけ**（1 スレッドが 4 連続列 = 2 語ちょうどを
 * 書く）。スカラ経路は `if (ocol + k < dims.n)` の部分書きなので、同じ u32 語への非アトミックな
 * read-modify-write になる。
 * MUST: したがって **s16 はスカラ経路へ配線しない**。生成の入口（src/kernels/gemm.ts /
 * src/kernels/attention-i8a8.ts）が `!v4` を fail loudly で落とし、適格判定
 * （{@link attentionScoreUsesF16}）が非適格を f32 格納へ縮退させる。
 *
 * ## 丸めは「格納」1 回だけ
 *
 * `pack2x16float` が唯一の丸め点で、読み側の `unpack2x16float` は**厳密**（f16 → f32 は常に
 * 表現可能）。したがって数値契約は
 *
 *     s16 変種の出力 ≡ S をホストで f16 に丸めた f32 変種の出力     … 1 ビットも違わない
 *
 * になる（ADR 0028 の f16 **タイル計算**より契約が強い — あちらは内積の途中に丸めが入る）。
 * MUST: レジスタ上で `f32 → f16 → f32` の往復を書かない（コンパイラに消される — ADR 0028 の
 * 知見）。丸めは必ず格納を経由させる。
 *
 * MUST: `s=f32` 変種の生成物は**バイト単位で従来のまま**。キーの格納判別子は f32 で空文字に
 * なり、各挿入点は f32 で従来と同じ文字列を返す。既存のスナップショット
 * （tests/fixtures/wgsl/*.wgsl）とパイプラインキャッシュの同一性がこれに掛かっている。
 */

import { CodegenError } from "../codegen/errors.ts";

/** S の格納形。意味論はどちらも f32（計算は常に f32 — ADR 0006）。 */
export type ScoreStorage = "f32" | "f16";

/** S 1 要素あたりの格納バイト数（f16 は 2 要素で 1 語）。 */
export const scoreStorageBytes = (storage: ScoreStorage): number => storage === "f16" ? 2 : 4;

/**
 * S を f16 格納にできる形か。
 *
 * **書き手（①QK）が v4 経路を取ることが唯一の条件**で、それが `D % 4 == 0 && N % 4 == 0`
 * になる:
 *
 * - f32 の ①QK は `gemmUsesVec4(k = D, n = N)`（src/kernels/gemm.ts）で v4 を決める — 縮約側
 *   `D % 4` と出力列側 `N % 4` の**両方**が要る。
 * - i8a8 の ①QK は出力列側だけを見る（`N % 4`）が、i8a8 の段の適格条件が既に `D % 4 == 0`
 *   なので、①が i8a8 でも f32 へ縮退していても、条件は同じ 1 本に畳める。
 *
 * 読み側（②行統計 / ③PV）は非適格でも安全だが（読みに RMW は無い）、書き手が f32 で書く以上
 * 意味が無い。**非適格は f32 格納へ沈黙で縮退する**（linear の `k % 4`・ADR 0030 決定 5 と
 * 同じ流儀で、落ちたことは診断のパイプラインキーにだけ出る）。
 */
export const attentionScoreUsesF16 = (depth: number, cols: number): boolean =>
  depth % 4 === 0 && cols % 4 === 0;

/**
 * パイプラインキーに載せる格納判別子。**格納重み `:wf16`（ADR 0018）/ 計算 `:c16`
 * （ADR 0028）と衝突しない第 3 の語**。
 *
 * MUST: f32 は空文字（既存キーは 1 文字も動かない）。MUST: 語は**キーの末尾**に置く — 3 つの
 * 軸が同時に立ちうるので、並び順を 1 箇所で固定しないと同一構成が 2 通りのキーを持つ。
 */
export const scoreKeyPart = (storage: ScoreStorage): string => storage === "f16" ? ":s16" : "";

/** WGSL 先頭コメントに足す但し書き（f32 は空 — 既存バイト列を保つ）。 */
export const scoreNote = (storage: ScoreStorage): string =>
  storage === "f16" ? ", S は f16 格納（pack2x16float）" : "";

/**
 * S バッファの WGSL 要素型。f16 は 2 要素を 1 語に詰めた格納なので **u32 の平坦配列**になる
 * （`f32Type` は f32 変種が使う従来の型をそのまま渡す口 — 変種ごとに `vec4<f32>` / `f32` /
 * `vec4<f16>` / `f16` と違うので、判定をここに集めずに呼び出し側から渡す）。
 */
export const scoreArrayType = (storage: ScoreStorage, f32Type: string): string =>
  storage === "f16" ? "u32" : f32Type;

/**
 * f16 変種の quad 読み出し関数（**quad 添字**を取る — 呼び出し側の添字算術を変えないため）。
 * 束縛の直後に置く前提で、前後 1 行ずつ空行が入る形に整えてある（f32 では空文字）。
 */
export const scoreQuadLoaderWgsl = (name: string, storage: ScoreStorage): string =>
  storage === "f16"
    ? `
// f16 格納の quad 読み: quad q = ${name}[2q] と ${name}[2q + 1] の 2 語（unpack は厳密）
fn score_quad(q: u32) -> vec4<f32> {
  let w = q * 2u;
  let lo = unpack2x16float(${name}[w]);
  let hi = unpack2x16float(${name}[w + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}
`
    : "";

/**
 * f16 変種のスカラ読み出し関数（**平坦要素添字**を取る — 重み f16 の `dequant` と同じ形）。
 *
 * MUST: 語と位置は平坦添字の偶奇で選ぶ。行内の相対添字で偶奇を取ると、行長が 2 の倍数の
 * ときだけ偶然一致する（ADR 0018 の罠と同型）。
 */
export const scoreScalarLoaderWgsl = (name: string, storage: ScoreStorage): string =>
  storage === "f16"
    ? `
// f16 格納の展開: 要素 i = unpack2x16float(${name}[i / 2])[i % 2]（平坦添字の偶奇で対を選ぶ）
fn score_at(i: u32) -> f32 {
  let pair = unpack2x16float(${name}[i >> 1u]);
  return select(pair.x, pair.y, (i & 1u) == 1u);
}
`
    : "";

/**
 * f16 変種の quad 書き出し関数（**quad 添字**を取る）。**丸めはここ 1 箇所**。
 *
 * MUST: 呼ぶのは 1 スレッドが 4 連続列を持つ v4 経路だけ（2 語ちょうどを排他に書く）。
 */
export const scoreStoreWgsl = (name: string, storage: ScoreStorage): string =>
  storage === "f16"
    ? `
// f16 格納の quad 書き: **丸めはこの pack2x16float 1 箇所だけ**（v4 経路が 2 語を排他に持つ）
fn score_store(q: u32, value: vec4<f32>) {
  let w = q * 2u;
  ${name}[w] = pack2x16float(value.xy);
  ${name}[w + 1u] = pack2x16float(value.zw);
}
`
    : "";

/** S の **quad 1 つ**の読み出し式（f32 は直接 quad 添字・f16 は展開関数）。 */
export const scoreReadQuad = (name: string, storage: ScoreStorage, quad: string): string =>
  storage === "f16" ? `score_quad(${quad})` : `${name}[${quad}]`;

/** S の **1 要素**の読み出し式（平坦要素添字）。 */
export const scoreReadAt = (name: string, storage: ScoreStorage, index: string): string =>
  storage === "f16" ? `score_at(${index})` : `${name}[${index}]`;

/** S の **quad 1 つ**の書き出し文（末尾の `;` まで含む）。 */
export const scoreStoreQuad = (
  name: string,
  storage: ScoreStorage,
  quad: string,
  value: string,
): string => storage === "f16" ? `score_store(${quad}, ${value});` : `${name}[${quad}] = ${value};`;

/**
 * s16 変種の生成前提を落とす（黙って f32 経路の WGSL を返さない）。
 *
 * - `!v4`: スカラ経路の部分書きは同じ u32 語への RMW になる（上の MUST）。
 * - `computeF16`: `:c16` は S を `array<f16>` で持つ**別の形**（ADR 0028）。冗長かつ矛盾する
 *   組を黙ってどちらかに解釈すると、丸め列がどちらの契約なのか診断から見えなくなる。
 */
export const assertScoreStorageSupported = (
  op: string,
  storage: ScoreStorage,
  v4: boolean,
  computeF16 = false,
): void => {
  if (storage !== "f16") return;
  if (computeF16) {
    throw new CodegenError(
      `${op}: S の f16 格納（s16）と f16 タイル計算（c16）は同時に指定できない` +
        "（c16 は S を array<f16> で持つ別の形 — attentionScoreStorage か attentionCompute の" +
        "どちらか一方にすること）",
    );
  }
  if (!v4) {
    throw new CodegenError(
      `${op}: S の f16 格納（s16）は v4 経路専用` +
        "（スカラ経路の部分書きは同じ u32 語への read-modify-write になる）",
    );
  }
};

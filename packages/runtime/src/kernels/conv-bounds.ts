// 畳み込みの GPU 座標が i32 に収まることの門（conv1d / conv2d / conv3d / conv_transpose1d の
// params が共有する葉関数）。
//
// `assertU32Params` は uniform に載せる個々の値の u32 収まりを見るが、WGSL が掛けた・足した
// 結果は保証しない。conv の WGSL は入力座標を i32 で組むので、真の座標が i32 の域を出ると
// 折り返し、範囲外の判定を実在する別の画素として通り抜ける（例外の出ない誤値）。判定を
// カーネルごとに書くと direct 版と暗黙 GEMM 版の受理集合が割れ、片側だけ直し忘れる
// （conv3d だけが持っていた前例がある）ため、ここ 1 箇所に置く。

import { CodegenError } from "../codegen/errors.ts";

/** i32 の最大値（2^31 − 1）。 */
const I32_MAX = 0x7fff_ffff;

/**
 * WGSL の i32 座標演算に入る値（と、その上界を与える導出値）が i32 の非負域に収まることを
 * 確かめる。
 *
 * MUST: 超えたら fail loudly。WGSL の `i32(u32)` はビットの再解釈なので、丸めも飽和もせずに
 * 負へ折り返す。
 *
 * @param where 失敗文言の頭（例: `"conv1d params"`）。
 * @param values 名前 → 値。名前は欄名か導出式（例: `"軸 w の入力長 + 2·padding"`）。
 */
export const assertI32Coordinates = (
  where: string,
  values: Readonly<Record<string, number>>,
): void => {
  for (const [name, value] of Object.entries(values)) {
    if (value > I32_MAX) {
      throw new CodegenError(
        `${where}: ${name} ${value} が i32 の上限 2^31 − 1 を超える` +
          "（GPU の座標演算は i32 — 折り返した座標は例外を出さずに誤値を返す）",
      );
    }
  }
};

/**
 * 順方向の畳み込み（conv1d / conv2d / conv3d）の 1 軸の座標域。
 *
 * MUST: `入力長 + 2·padding ≤ 2^31 − 1`。WGSL は入力座標を
 * `i32(出力座標·stride) + i32(k·dilation) − i32(padding)` の 32 ビット整数で組む。
 * 出力長の式（shapes.ts）から `(out − 1)·stride ≤ L + 2p − d(K − 1) − 1` と
 * `k·dilation ≤ d(K − 1) ≤ L + 2p − 1` が従い、座標は `[−p, L + p − 1]` に入るので、この 1 条件で
 * 中間値が全て i32 に収まる。CPU 参照の意味論は変えない（GPU 実行の subset — ADR 0064 軸 A）。
 *
 * 各値の u32 域は呼び手が src/codegen/params.ts の `assertU32Params` で先に確かめておくこと。
 */
export const assertConvAxisExtent = (
  where: string,
  axis: string,
  length: number,
  padding: number,
): void => {
  assertI32Coordinates(where, { [`軸 ${axis} の入力長 + 2·padding`]: length + 2 * padding });
};

// カーネル uniform の u32 域検査（1 本）。
//
// uniform の各欄は WGSL 側で `u32` として読まれる一方、ホストの値は f64 なので、
// `Uint32Array` への代入は 2^32 の剰余を黙って取る。検査をカーネルごとに書くと
// 「safe-integer だけ見る / 上限も見る / 何も見ない」の流儀に割れ、割れた側で超過値が
// narrowing されたまま dispatch まで通る（要素数だけ別物になる沈黙誤値）ため、判定は
// ここ 1 箇所に置く。

import { CodegenError } from "../codegen/errors.ts";

/**
 * uniform に載せる値（および載せる値の導出に使う中間値）が u32 の域に収まることを確かめる。
 *
 * MUST: 非整数・負・`0xffff_ffff` 超は fail loudly。丸めても切り詰めてもいけない —
 * ここで通した値はそのまま WGSL の添字計算になる。
 * MUST: 文言にはパラメタ名と値を入れる（どの欄が溢れたのかが分からないと診断が始まらない）。
 *
 * @param where 失敗文言の頭（例: `"gather params"`）。
 * @param values パラメタ名 → 値。名前は uniform の欄名か、導出式（例: `"4 * n"`）。
 */
export const assertU32Params = (
  where: string,
  values: Readonly<Record<string, number>>,
): void => {
  for (const [name, value] of Object.entries(values)) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
      throw new CodegenError(`${where}: ${name} は u32 の非負整数（${value}）`);
    }
  }
};

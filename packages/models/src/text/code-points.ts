/**
 * 文字列 → コードポイント列。
 *
 * **ファミリ非依存**（Unigram / BPE / 正規化のどの経路も、正本と同じ「コードポイントの列」で
 * 数え始める）なので `src/text/` に置く。ファミリ側のディレクトリに置くと、別ファミリが
 * 使うときに家族間 import（依存の逆流）になる。
 */

import { ModelInputError } from "../errors.ts";

/**
 * 対になっていないサロゲートは**受け付けない** — Python / Rust の str には載らない値で
 * 正本の出力が定義されない。黙って U+FFFD に潰すと id 列が静かに変わるので落とす
 * （JS の文字列型でのみ表現できる入力なので、移植で最も忘れやすい防御）。
 */
export const toCodePoints = (text: string): number[] => {
  const out: number[] = [];
  for (const ch of text) {
    const cp = ch.codePointAt(0) as number;
    if (cp >= 0xd800 && cp <= 0xdfff) {
      throw new Error(`対にならないサロゲート U+${cp.toString(16).toUpperCase()} が入力にある`);
    }
    out.push(cp);
  }
  return out;
};

/**
 * 呼び手の本文が {@link toCodePoints} を通ることを**生成の入口で**確かめる。
 *
 * MUST: 受理条件の正本は {@link toCodePoints} 1 本のまま — ここが上げるのは**送出型だけ**。
 * 一括で置き換えないのは、同じ throw が語彙資産の解析でも出るためで、そちらは資産の齟齬
 * （呼び手が本文を直しても直らない）だから素の `Error` が正しい。呼び手の本文を受ける家族の
 * `encode` だけがこの口を通す（ADR 0107 決定 2）。
 */
export const assertEncodableText = (text: string, label: string): void => {
  try {
    toCodePoints(text);
  } catch (cause) {
    throw new ModelInputError(
      `${label}: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause },
    );
  }
};

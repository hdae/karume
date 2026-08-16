/**
 * DeBERTa 隠れ状態の選択と、word2ph による音素レベルへの tile 展開。
 *
 * 参照実装 `nlp/japanese/bert_feature.py` の
 * `torch.cat(res["hidden_states"][-3:-2], -1)[0]` → `torch.cat([res[i].repeat(w[i],1) ...]).T`
 * に対応する 2 段。IR v1 のグラフ出力は位置で引く（`output.<i>`）ので、選択は
 * **末尾からの相対位置**で行う（層を削った variant でも同じ規則で引ける）。
 */

/**
 * BERT 特徴に使う hidden_states の出力名を、末尾からの位置で選ぶ。
 *
 * MUST: 固定 index にしない。配布形（`sbv2-22layer`）は SBV2 が使う 1 本だけを出すが、検証用の
 * `full-24layer` は埋め込み出力込み 25 本、`dev-2layer` は 3 本で、固定 index はどれかで別の層を
 * 静かに引く。
 */
export const bertHiddenOutput = (outputs: readonly string[], fromEnd: number): string => {
  if (!Number.isInteger(fromEnd) || fromEnd < 1) {
    throw new Error(`hidden_states の末尾からの位置 ${fromEnd} が 1 以上の整数でない`);
  }
  if (outputs.length < fromEnd) {
    throw new Error(`hidden_states が ${outputs.length} 本しかない（${fromEnd} 本以上必要）`);
  }
  return outputs[outputs.length - fromEnd];
};

/**
 * {@link tileBertToPhoneLevel} の結果。
 *
 * MUST: `columns` は**走査が実際に進めた列数**で、`data` の確保サイズ（`dim · sum(word2ph)`）
 * とは別の観測値。両者を突き合わせる門（`pipeline.ts` の `assertTiledBert`）はこの独立性の
 * 上にだけ成立する — `sum(word2ph)` から算出し直した値を入れると、門は恒真に戻る。
 */
export type TiledBert = {
  readonly data: Float32Array<ArrayBuffer>;
  /** BERT 特徴の次元（`hidden` の行長）。 */
  readonly dim: number;
  /** 走査が書いた列数（正しく回れば `sum(word2ph)` = 音素数 P）。 */
  readonly columns: number;
};

/**
 * トークンごと隠れ状態 `[tokenCount, dim]`（行優先）を word2ph で音素レベルへ tile 展開し、
 * 転置して `[dim, P]`（P = sum(word2ph)）にする。
 *
 * @param hidden グラフ出力の生データ（`[1, tokenCount, dim]` の row-major と同じ並び）。
 * @param tokenCount DeBERTa トークン数（CLS / SEP 込み）。
 * @param word2ph add_blank 後の word2ph（長さ === tokenCount が必須）。
 */
export const tileBertToPhoneLevel = (
  hidden: Float32Array,
  tokenCount: number,
  word2ph: readonly number[],
): TiledBert => {
  if (tokenCount <= 0 || hidden.length % tokenCount !== 0) {
    throw new Error(`hidden 長 ${hidden.length} が tokenCount ${tokenCount} で割り切れない`);
  }
  if (word2ph.length !== tokenCount) {
    throw new Error(
      `DeBERTa トークン数 ${tokenCount} が word2ph 長 ${word2ph.length} と違う` +
        "（文字トークナイズと word2ph の齟齬を疑う）",
    );
  }
  const dim = hidden.length / tokenCount;
  const total = word2ph.reduce((sum, count) => sum + count, 0);
  // out[d * total + p] = hidden[srcToken(p) * dim + d]（展開と転置を 1 周で行う）。
  const out = new Float32Array(dim * total);
  let column = 0;
  for (let token = 0; token < tokenCount; token += 1) {
    const source = token * dim;
    for (let repeat = 0; repeat < word2ph[token]; repeat += 1) {
      for (let d = 0; d < dim; d += 1) out[d * total + column] = hidden[source + d];
      column += 1;
    }
  }
  return { data: out, dim, columns: column };
};

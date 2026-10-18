/**
 * Selection of the DeBERTa hidden state, and its tile expansion to phoneme level by word2ph.
 *
 * The 2 steps correspond to the reference implementation `nlp/japanese/bert_feature.py`'s
 * `torch.cat(res["hidden_states"][-3:-2], -1)[0]` → `torch.cat([res[i].repeat(w[i],1) ...]).T`.
 * The IR's graph outputs are looked up by position, so the selection is made by **position relative
 * to the end** (a variant with trimmed layers is looked up by the same rule).
 */

/**
 * Selects the output name of the hidden_states used for BERT features by its position from the end.
 *
 * MUST: do not use a fixed index. The distribution (`sbv2-22layer`) outputs only the one SBV2 uses,
 * but the verification graph `full-24layer` has 25 including the embedding output and `dev-2layer`
 * has 3, so a fixed index silently picks a different layer in one of them.
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
 * The result of {@link tileBertToPhoneLevel}.
 *
 * MUST: `columns` is **the number of columns the walk actually advanced**, an observation separate
 * from the allocation size of `data` (`dim · sum(word2ph)`). The check that compares the two
 * (`assertTiledBert` in `pipeline.ts`) holds only on top of this independence — put in a value
 * recomputed from `sum(word2ph)` and the check becomes a tautology again.
 */
export type TiledBert = {
  readonly data: Float32Array<ArrayBuffer>;
  /** Dimension of the BERT features (the row length of `hidden`). */
  readonly dim: number;
  /** Number of columns the walk wrote (`sum(word2ph)` = phoneme count P when it runs correctly). */
  readonly columns: number;
};

/**
 * Tile-expands the per-token hidden states `[tokenCount, dim]` (row-major) to phoneme level by
 * word2ph and transposes to `[dim, P]` (P = sum(word2ph)).
 *
 * @param hidden raw data of the graph output (same order as row-major `[1, tokenCount, dim]`).
 * @param tokenCount DeBERTa token count (including CLS / SEP).
 * @param word2ph word2ph after add_blank (length === tokenCount is required).
 */
export const tileBertToPhoneLevel = (
  hidden: Float32Array<ArrayBuffer>,
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
  // out[d * total + p] = hidden[srcToken(p) * dim + d] (expansion and transpose in one pass).
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

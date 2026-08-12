/**
 * 系列 → ベクトルの縮約（ADR 0047 決定 4 のホスト残置）。
 *
 * 2 箇所で要る: `speaker` グラフの出力へ前置する平均トークン（上流
 * `_prepend_masked_mean_token`）と、`duration` の `caption_vec`（`caption_norm` 済み系列の
 * masked mean — 上流 `DurationPredictor._caption_vec`）。
 *
 * どちらも **B=1・マスク全 True の経路でしか呼ばない**（参照なし / caption 空はホストが
 * ゼロを置いて短絡する）ので、上流の `sum(x·mask)/clamp_min(sum(mask),1)` は単純平均に落ちる。
 *
 * MUST: f32 の丸めを 1 演算ずつ踏む（列ごとの逐次和）。
 */

const f32 = Math.fround;

/** 行方向の平均 `[width]`。 */
export const rowMean = (
  state: Float32Array<ArrayBuffer>,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  if (state.length !== rows * width) {
    throw new Error(`系列の長さ ${state.length} が ${rows}×${width} と違う`);
  }
  if (rows < 1) throw new Error("系列が空（行平均を作れない）");
  const out = new Float32Array(width);
  for (let column = 0; column < width; column += 1) {
    let sum = 0;
    for (let row = 0; row < rows; row += 1) sum = f32(sum + state[row * width + column]);
    out[column] = f32(sum / rows);
  }
  return out;
};

/**
 * 平均トークンを先頭に前置した `[(rows+1)·width]`。
 *
 * MUST: 前置であって後置ではない — `duration` の `speaker_vec` は `speaker_state[:,0]`
 * （= この平均トークン）を採る契約で、位置がずれると別の行が話者ベクトルになる。
 */
export const prependMeanToken = (
  state: Float32Array<ArrayBuffer>,
  rows: number,
  width: number,
): Float32Array<ArrayBuffer> => {
  const mean = rowMean(state, rows, width);
  const out = new Float32Array((rows + 1) * width);
  out.set(mean, 0);
  out.set(state, width);
  return out;
};

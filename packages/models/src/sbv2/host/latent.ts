/**
 * z_p の構築（front の m_p / logs_p を音素→フレームへ展開し、ノイズを載せる）。
 *
 * 参照実装 `SynthesizerTrnJPExtra.infer` の
 * `m_p = attn · m_pᵀ` → `z_p = m_p + randn_like(m_p)·exp(logs_p)·noise_scale`。
 * attn は単調パスの 0/1 行列なので、行列積は `expandIdx` による gather と同値
 * （`duration.ts` の doc を参照）。
 */

/**
 * `z_p[c][f] = m_p[c][idx[f]] + noise[c][f]·exp(logs_p[c][idx[f]])·noiseScale` を組む。
 *
 * @param mP front の出力 2（`[1, C, P]` の row-major データ）。
 * @param logsP front の出力 3（同形）。
 * @param expandIdx フレーム → 音素の添字（長さ Ty）。
 * @param channels C（front のグラフ出力 shape から採る — ここで決め打ちしない）。
 * @param noise 標準正規列（長さ `C·Ty`）。**利用者が供給する面ではない** — 常に内部の
 * `random.normals` から来る。引数にしてあるのは `examples/sbv2/dump.ts` が同じ列を
 * `zp_noise` として書き出し、参照側（`demo.py`）がそれを読んで突き合わせるため。
 *
 * NOTE: 式は f64 で通して `Float32Array` 代入で 1 度だけ丸める（同ディレクトリの
 * `random.ts` と同じ家風）。参照は f32 逐次なので要素の 4 割が 1 ulp 割れるが、この経路を
 * 測る波形突合の実測 maxAbs 5.16e-5 に対して 3 桁下で、ceil のような離散化も挟まらない。
 */
export const buildZp = (
  mP: Float32Array,
  logsP: Float32Array,
  expandIdx: Int32Array,
  channels: number,
  noise: Float32Array,
  noiseScale: number,
): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(channels) || channels <= 0) {
    throw new Error(`チャネル数 ${channels} が 1 以上の整数でない`);
  }
  if (logsP.length !== mP.length) {
    throw new Error(`m_p(${mP.length}) と logs_p(${logsP.length}) の要素数が違う`);
  }
  const phonemes = mP.length / channels;
  if (!Number.isInteger(phonemes)) {
    throw new Error(`m_p の要素数 ${mP.length} が C=${channels} で割り切れない`);
  }
  const frames = expandIdx.length;
  if (noise.length !== channels * frames) {
    throw new Error(`ノイズ長 ${noise.length} が C·Ty(${channels * frames}) と違う`);
  }
  const zP = new Float32Array(channels * frames);
  for (let c = 0; c < channels; c += 1) {
    const rowSource = c * phonemes;
    const rowTarget = c * frames;
    for (let f = 0; f < frames; f += 1) {
      const index = expandIdx[f];
      if (index < 0 || index >= phonemes) {
        throw new Error(`展開インデックス ${index}（フレーム ${f}）が P=${phonemes} の外`);
      }
      const source = rowSource + index;
      zP[rowTarget + f] = mP[source] + noise[rowTarget + f] * Math.exp(logsP[source]) * noiseScale;
    }
  }
  return zP;
};

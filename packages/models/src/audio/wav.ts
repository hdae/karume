/**
 * f32 モノラル波形 → 16bit PCM の WAV（RIFF）バイト列。
 *
 * **パイプライン非依存の共通処理**（音声生成モデルは総じて最後にこれを通す）なので、
 * ファミリのディレクトリではなく `src/audio/` に置く（`src/image/png.ts` と同じ位置づけ）。
 * ランタイム依存は無い（`DataView` だけ）。
 *
 * パイプラインの出力と、torch 参照（`tools/exporter/sbv2_demo.py`）が出す reference.wav /
 * official.wav は**同じ規則**で書く（クリップ → `floor(x·32767 + 0.5)`）。エンコード規則が
 * 割れると「聴き比べ」に実装差が混ざる。
 */

const HEADER_BYTES = 44;
const BITS_PER_SAMPLE = 16;
const CHANNELS = 1;
const FORMAT_PCM = 1;
/** f32 → i16 のスケール。`Math.round(v · 32767)` が ±32767 に収まる（-32768 は使わない）。 */
const FULL_SCALE = 32767;

export const encodeWav = (
  samples: Float32Array,
  sampleRate: number,
): Uint8Array<ArrayBuffer> => {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`サンプリング周波数 ${sampleRate} が 1 以上の整数でない`);
  }
  const bytesPerSample = BITS_PER_SAMPLE / 8;
  const dataBytes = samples.length * bytesPerSample;
  const out = new Uint8Array(HEADER_BYTES + dataBytes);
  const view = new DataView(out.buffer);
  const ascii = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i += 1) out[offset + i] = text.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, HEADER_BYTES - 8 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // fmt チャンク長（PCM）
  view.setUint16(20, FORMAT_PCM, true);
  view.setUint16(22, CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * CHANNELS * bytesPerSample, true); // byte rate
  view.setUint16(32, CHANNELS * bytesPerSample, true); // block align
  view.setUint16(34, BITS_PER_SAMPLE, true);
  ascii(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clipped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(HEADER_BYTES + i * bytesPerSample, Math.round(clipped * FULL_SCALE), true);
  }
  return out;
};

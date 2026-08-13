/**
 * f32 モノラル波形 ↔ WAV（RIFF）バイト列。
 *
 * **パイプライン非依存の共通処理**（音声生成モデルは総じて最後に WAV を書き、声質の参照には
 * WAV を読む）なので、ファミリのディレクトリではなく `src/audio/` に置く（`src/image/png.ts`
 * と同じ位置づけ）。ランタイム依存は無い（`DataView` だけ）。
 *
 * パイプラインの出力と、torch 参照（`tools/exporter/sbv2_demo.py`）が出す reference.wav /
 * official.wav は**同じ規則**で書く（クリップ → `floor(x·32767 + 0.5)`）。エンコード規則が
 * 割れると「聴き比べ」に実装差が混ざる。
 *
 * ## MUST: 読みと書きでスケールが非対称なのは意図（往復はビット一致しない）
 *
 * {@link decodeWav} の int16 → f32 は **32768 で割り**、{@link encodeWav} の f32 → int16 は
 * **32767 を掛ける**。前者は参照音声の読み手（上流 `soundfile` / `codec.py`）の規約に、後者は聴き比べ
 * 相手の torch 台本に、それぞれ合わせた結果で、どちらも**外部との一致が正**。揃えると
 * 参照音声の LUFS が上流とずれる（相対 3e-5 — 実測は `dacvae_host.py` の
 * `_wav_scale_evidence`）。したがって往復すると 1LSB 級の差が出る（`wav_test.ts` が固定）。
 */

const HEADER_BYTES = 44;
const BITS_PER_SAMPLE = 16;
const CHANNELS = 1;
const FORMAT_PCM = 1;
/** IEEE 754 float の WAVE format tag（`WAVE_FORMAT_IEEE_FLOAT`）。 */
const FORMAT_IEEE_FLOAT = 3;
/** f32 → i16 のスケール。`Math.round(v · 32767)` が ±32767 に収まる（-32768 は使わない）。 */
const FULL_SCALE = 32767;
/** i16 → f32 のスケール。下端 −32768 が厳密に −1.0 へ写る（上流の読み手と同じ規約）。 */
const INT16_DIVISOR = 32768;
/** RIFF のヘッダ欄（チャンク長 / sample rate / byte rate）が取れる最大値。 */
const U32_MAX = 0xffff_ffff;

export const encodeWav = (
  samples: Float32Array,
  sampleRate: number,
): Uint8Array<ArrayBuffer> => {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`サンプリング周波数 ${sampleRate} が 1 以上の整数でない`);
  }
  const bytesPerSample = BITS_PER_SAMPLE / 8;
  const frameBytes = CHANNELS * bytesPerSample;
  // MUST: u32 域を先に見る。`setUint32` は mod 2^32 で黙って巻き戻すので、検査を欠くと
  // 「別の周波数・別の長さを宣言した valid な WAV」が出る。byte rate は sampleRate の 2 倍
  // なので、この 1 本で sampleRate 自身の u32 超過も覆う。
  const byteRate = sampleRate * frameBytes;
  if (byteRate > U32_MAX) {
    throw new RangeError(
      `サンプリング周波数 ${sampleRate} の byte rate ${byteRate} が u32 に収まらない` +
        `（上限 ${Math.floor(U32_MAX / frameBytes)} Hz）`,
    );
  }
  const dataBytes = samples.length * bytesPerSample;
  // RIFF チャンク長は data 長 + 36 を書くので、data 長より先にそちらが溢れる。
  const riffBytes = HEADER_BYTES - 8 + dataBytes;
  if (riffBytes > U32_MAX) {
    throw new RangeError(
      `サンプル数 ${samples.length}（data ${dataBytes} バイト）の RIFF チャンク長 ${riffBytes} が` +
        ` u32 に収まらない（上限 ${
          Math.floor((U32_MAX - (HEADER_BYTES - 8)) / bytesPerSample)
        } サンプル）`,
    );
  }
  const out = new Uint8Array(HEADER_BYTES + dataBytes);
  const view = new DataView(out.buffer);
  const ascii = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i += 1) out[offset + i] = text.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, riffBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true); // fmt チャンク長（PCM）
  view.setUint16(20, FORMAT_PCM, true);
  view.setUint16(22, CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true); // byte rate
  view.setUint16(32, frameBytes, true); // block align
  view.setUint16(34, BITS_PER_SAMPLE, true);
  ascii(36, "data");
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clipped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(HEADER_BYTES + i * bytesPerSample, Math.round(clipped * FULL_SCALE), true);
  }
  return out;
};

/** {@link decodeWav} の結果（モノラル f32 と、ファイルが宣言していたサンプリング周波数）。 */
export type DecodedWav = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
};

/** RIFF の 1 チャンク（`id` は ASCII 4 文字、`offset` はデータ本体の先頭）。 */
type RiffChunk = {
  readonly offset: number;
  readonly length: number;
};

const ascii = (view: DataView, offset: number): string =>
  String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );

/**
 * `fmt ` と `data` のチャンクを拾う。
 *
 * MUST: 知らないチャンク（`LIST` / `fact` / `bext` …）は**読み飛ばす**。実在の WAV は
 * メタデータチャンクを普通に挟むので、44 バイト固定ヘッダを決め打ちすると読めない。
 * チャンクは奇数長のとき 1 バイトの詰めが入る（RIFF の規約）。
 */
const findChunks = (view: DataView): { readonly fmt: RiffChunk; readonly data: RiffChunk } => {
  if (view.byteLength < 12) throw new Error(`decodeWav: ${view.byteLength} バイトしかない`);
  if (ascii(view, 0) !== "RIFF" || ascii(view, 8) !== "WAVE") {
    throw new Error(
      `decodeWav: RIFF/WAVE ヘッダでない（'${ascii(view, 0)}' / '${ascii(view, 8)}'）`,
    );
  }
  let fmt: RiffChunk | undefined;
  let data: RiffChunk | undefined;
  let cursor = 12;
  while (cursor + 8 <= view.byteLength) {
    const id = ascii(view, cursor);
    const length = view.getUint32(cursor + 4, true);
    const offset = cursor + 8;
    if (offset + length > view.byteLength) {
      throw new Error(
        `decodeWav: チャンク '${id}' が ${length} バイトを宣言しているが、` +
          `残りは ${view.byteLength - offset} バイトしかない`,
      );
    }
    if (id === "fmt ") fmt = { offset, length };
    if (id === "data") data = { offset, length };
    cursor = offset + length + (length % 2);
  }
  if (fmt === undefined || data === undefined) {
    throw new Error(
      `decodeWav: ${fmt === undefined ? "'fmt '" : "'data'"} チャンクが無い`,
    );
  }
  return { fmt, data };
};

/**
 * WAV（RIFF）バイト列 → モノラル f32。
 *
 * 受け付けるのは **PCM 16bit**（/32768）と **IEEE float 32bit**（そのまま）だけで、他の
 * format tag / bit depth は fail loudly（黙って近似しない — 横断不変条件）。多チャネルは
 * **チャネル平均**で mono 化する（上流 `codec.py` の `encode_waveform` と同じ）。
 *
 * リサンプルはしない。周波数が要求と違うかどうかは呼び出し側（パイプライン）が判定する。
 */
export const decodeWav = (bytes: Uint8Array): DecodedWav => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const { fmt, data } = findChunks(view);
  if (fmt.length < 16) {
    throw new Error(`decodeWav: 'fmt ' チャンクが ${fmt.length} バイトしかない（16 以上が要る）`);
  }
  const format = view.getUint16(fmt.offset, true);
  const channels = view.getUint16(fmt.offset + 2, true);
  const sampleRate = view.getUint32(fmt.offset + 4, true);
  const bits = view.getUint16(fmt.offset + 14, true);
  if (channels < 1) throw new Error("decodeWav: チャネル数が 0");
  if (sampleRate < 1) throw new Error("decodeWav: サンプリング周波数が 0");
  const pcm16 = format === FORMAT_PCM && bits === 16;
  const float32 = format === FORMAT_IEEE_FLOAT && bits === 32;
  if (!pcm16 && !float32) {
    throw new Error(
      `decodeWav: format ${format} / ${bits}bit に未対応` +
        `（PCM ${FORMAT_PCM} の 16bit と IEEE float ${FORMAT_IEEE_FLOAT} の 32bit だけ）`,
    );
  }
  const bytesPerSample = bits / 8;
  const frameBytes = bytesPerSample * channels;
  if (data.length % frameBytes !== 0) {
    throw new Error(
      `decodeWav: 'data' が ${data.length} バイトで、` +
        `1 フレーム ${frameBytes} バイト（${channels}ch × ${bits}bit）で割り切れない`,
    );
  }
  const frames = data.length / frameBytes;
  const out = new Float32Array(frames) as Float32Array<ArrayBuffer>;
  for (let frame = 0; frame < frames; frame += 1) {
    const base = data.offset + frame * frameBytes;
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const at = base + channel * bytesPerSample;
      sum += pcm16 ? view.getInt16(at, true) / INT16_DIVISOR : view.getFloat32(at, true);
    }
    // mono は割り算を通さない（1 で割ると値は同じだが、意図が「平均」でなく「そのまま」なので）。
    out[frame] = channels === 1 ? sum : sum / channels;
  }
  return { data: out, sampleRate };
};

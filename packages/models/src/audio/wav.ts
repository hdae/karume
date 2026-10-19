/**
 * f32 mono waveform ↔ WAV (RIFF) bytes.
 *
 * This is **pipeline-independent shared processing** (audio generation models generally write a WAV
 * at the end and read a WAV for the voice reference), so it lives in `src/audio/` rather than in a
 * family directory (the same standing as `src/image/png.ts`). No runtime dependencies (`DataView`
 * only).
 *
 * The pipeline output and the reference.wav / official.wav produced by the torch reference
 * (`tools/export-recipes/sbv2/demo.py`) are written by **the same rule** (clip →
 * `floor(x·32767 + 0.5)`). If the encoding rules diverged, implementation differences would mix
 * into "listening comparisons".
 *
 * ## MUST: the asymmetric scale between reading and writing is intended (a round trip is not bit-identical)
 *
 * {@link decodeWav}'s int16 → f32 **divides by 32768**, and {@link encodeWav}'s f32 → int16
 * **multiplies by 32767**. The former follows the convention of the reference-audio reader (upstream
 * `soundfile` / `codec.py`), the latter the torch script it is compared against by ear; in both
 * cases **agreement with the outside is what is correct**. Aligning them would shift the reference
 * audio's LUFS from upstream (relative 3e-5 — measured by `_wav_scale_evidence` in
 * `tools/export-recipes/irodori/dacvae/host.py`). So a round trip differs at the 1 LSB level (pinned
 * by `wav_test.ts`).
 */

import { ModelInputError } from "../errors.ts";

const HEADER_BYTES = 44;
const BITS_PER_SAMPLE = 16;
const CHANNELS = 1;
const FORMAT_PCM = 1;
/** WAVE format tag for IEEE 754 float (`WAVE_FORMAT_IEEE_FLOAT`). */
const FORMAT_IEEE_FLOAT = 3;
/** f32 → i16 scale. `Math.round(v · 32767)` stays within ±32767 (-32768 is not used). */
const FULL_SCALE = 32767;
/** i16 → f32 scale. The bottom end −32768 maps exactly to −1.0 (the same convention as the upstream reader). */
const INT16_DIVISOR = 32768;
/** The largest value the RIFF header fields (chunk length / sample rate / byte rate) can hold. */
const U32_MAX = 0xffff_ffff;

/**
 * NOTE: all 4 checks of this function are {@link ModelInputError}. Both `samples` and `sampleRate`
 * are values the caller passes (it is only exported from the barrel and is not called from inside a
 * pipeline), and the only remedy is "fix the waveform / rate you pass" = what HTTP would call a 400
 * (ADR 0107 decisions 2 and 5).
 */
export const encodeWav = (
  samples: Float32Array<ArrayBufferLike>,
  sampleRate: number,
): Uint8Array<ArrayBuffer> => {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new ModelInputError(`Sample rate ${sampleRate} is not an integer of at least 1`);
  }
  const bytesPerSample = BITS_PER_SAMPLE / 8;
  const frameBytes = CHANNELS * bytesPerSample;
  // MUST: check the u32 range first. `setUint32` wraps silently modulo 2^32, so without the check
  // the output would be "a valid WAV declaring a different rate and a different length". The byte
  // rate is twice sampleRate, so this one check also covers sampleRate itself exceeding u32.
  const byteRate = sampleRate * frameBytes;
  if (byteRate > U32_MAX) {
    throw new ModelInputError(
      `Sample rate ${sampleRate}: byte rate ${byteRate} does not fit in u32` +
        ` (upper bound ${Math.floor(U32_MAX / frameBytes)} Hz)`,
    );
  }
  const dataBytes = samples.length * bytesPerSample;
  // The RIFF chunk length is written as data length + 36, so it overflows before the data length does.
  const riffBytes = HEADER_BYTES - 8 + dataBytes;
  if (riffBytes > U32_MAX) {
    throw new ModelInputError(
      `Sample count ${samples.length} (data ${dataBytes} bytes): the RIFF chunk length ${riffBytes}` +
        ` does not fit in u32 (upper bound ${Math.floor(
          (U32_MAX - (HEADER_BYTES - 8)) / bytesPerSample,
        )} samples)`,
    );
  }
  // MUST: reject non-finite samples here. The clip below uses `Math.max` / `Math.min`, which pass
  // NaN through, and `setInt16` given `Math.round(NaN)` writes 0 (±Infinity sticks to full scale) —
  // neither throws, so numerical breakdown turns into "a valid silent / clipped WAV".
  for (let i = 0; i < samples.length; i += 1) {
    if (!Number.isFinite(samples[i])) {
      throw new ModelInputError(
        `Sample ${i} (${samples[i]}) is non-finite (cannot be written to WAV)`,
      );
    }
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
  view.setUint32(16, 16, true); // fmt chunk length (PCM)
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

/** The result of {@link decodeWav} (mono f32 and the sampling rate the file declared). */
export type DecodedWav = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
};

/** One RIFF chunk (`id` is 4 ASCII characters; `offset` is the start of the data body). */
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
 * Picks up the `fmt ` and `data` chunks.
 *
 * MUST: **skip** unknown chunks (`LIST` / `fact` / `bext` …). Real WAV files routinely include
 * metadata chunks, so assuming a fixed 44-byte header cannot read them. A chunk of odd length is
 * followed by a 1-byte pad (the RIFF convention).
 *
 * MUST: the scan boundary is not the physical length but **the logical end RIFF declares at offset
 * 4** (the field {@link encodeWav} writes). Physical bytes left beyond the declaration are outside the
 * RIFF container and are not read (ignored per the specification, not an approximation).
 * Conversely, a logical end beyond the physical length means a truncated container.
 *
 * MUST: the scan ends **exactly** at the logical end. A fragment of fewer than 8 bytes left before
 * the end means either a broken container or a header read 1 byte off after a writer that omitted
 * the pad of an odd-length chunk (every later chunk boundary is shifted). Ignoring it silently would
 * make this inconsistent in strictness with the other checks that reject duplicate chunks and
 * excessive lengths.
 * NOTE: the only exception is the shape "**the last** chunk has odd length and only its 1-byte pad
 * is missing", where the scan ends 1 byte past the end. It is common among real writers, and there
 * is nothing to read after the pad so nothing shifts, hence it is accepted.
 */
const findChunks = (view: DataView): { readonly fmt: RiffChunk; readonly data: RiffChunk } => {
  if (view.byteLength < 12) {
    throw new ModelInputError(`decodeWav: only ${view.byteLength} bytes`);
  }
  if (ascii(view, 0) !== "RIFF" || ascii(view, 8) !== "WAVE") {
    throw new ModelInputError(
      `decodeWav: not a RIFF/WAVE header ('${ascii(view, 0)}' / '${ascii(view, 8)}')`,
    );
  }
  const riffSize = view.getUint32(4, true);
  const logicalEnd = 8 + riffSize;
  if (logicalEnd > view.byteLength) {
    throw new ModelInputError(
      `decodeWav: RIFF declares ${riffSize} bytes, ` +
        `but only ${view.byteLength - 8} bytes remain`,
    );
  }
  let fmt: RiffChunk | undefined;
  let data: RiffChunk | undefined;
  let cursor = 12;
  while (cursor < logicalEnd) {
    if (cursor + 8 > logicalEnd) {
      throw new ModelInputError(
        `decodeWav: there is a fragment of ${logicalEnd - cursor} bytes before the end of RIFF` +
          " (a chunk header is 8 bytes — the container is broken, or the pad of an odd-length chunk is missing and " +
          "every later boundary is shifted)",
      );
    }
    const id = ascii(view, cursor);
    const length = view.getUint32(cursor + 4, true);
    const offset = cursor + 8;
    if (offset + length > logicalEnd) {
      throw new ModelInputError(
        `decodeWav: chunk '${id}' declares ${length} bytes, ` +
          `but only ${logicalEnd - offset} bytes remain`,
      );
    }
    // MUST: one `fmt ` / `data` each. Overwriting with the last one could silently read a data other
    // than the one the writer intended (skipped unknown chunks may repeat).
    if ((id === "fmt " && fmt !== undefined) || (id === "data" && data !== undefined)) {
      throw new ModelInputError(`decodeWav: chunk '${id}' appears twice`);
    }
    if (id === "fmt ") fmt = { offset, length };
    if (id === "data") data = { offset, length };
    cursor = offset + length + (length % 2);
  }
  if (fmt === undefined || data === undefined) {
    throw new ModelInputError(
      `decodeWav: ${fmt === undefined ? "'fmt '" : "'data'"} chunk is missing`,
    );
  }
  return { fmt, data };
};

/**
 * WAV (RIFF) bytes → mono f32.
 *
 * Only **PCM 16-bit** (/32768) and **IEEE float 32-bit** (as is) are accepted; any other format
 * tag / bit depth fails loudly (no silent approximation — the cross-cutting invariant).
 * Multi-channel input is made mono by **channel averaging** (the same as `encode_waveform` in
 * upstream `codec.py`).
 *
 * No resampling. Whether the rate differs from what is required is decided by the caller (the
 * pipeline).
 *
 * NOTE: every header, chunk and format check, including the 7 in {@link findChunks}, is a
 * {@link ModelInputError}. What is parsed is **the bytes the host passed themselves** (this function
 * too is only exported from the barrel and is not called from inside a pipeline), not an asset
 * inconsistency — the only remedy is "fix the WAV you pass" = what HTTP would call a 400 (ADR 0107
 * decision 2).
 */
export const decodeWav = (bytes: Uint8Array<ArrayBufferLike>): DecodedWav => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const { fmt, data } = findChunks(view);
  if (fmt.length < 16) {
    throw new ModelInputError(
      `decodeWav: 'fmt ' chunk is only ${fmt.length} bytes (at least 16 are required)`,
    );
  }
  const format = view.getUint16(fmt.offset, true);
  const channels = view.getUint16(fmt.offset + 2, true);
  const sampleRate = view.getUint32(fmt.offset + 4, true);
  const bits = view.getUint16(fmt.offset + 14, true);
  if (channels < 1) throw new ModelInputError("decodeWav: channel count is 0");
  if (sampleRate < 1) throw new ModelInputError("decodeWav: sample rate is 0");
  const pcm16 = format === FORMAT_PCM && bits === 16;
  const float32 = format === FORMAT_IEEE_FLOAT && bits === 32;
  if (!pcm16 && !float32) {
    throw new ModelInputError(
      `decodeWav: format ${format} / ${bits}bit is not supported` +
        ` (only PCM ${FORMAT_PCM} 16bit and IEEE float ${FORMAT_IEEE_FLOAT} 32bit)`,
    );
  }
  const bytesPerSample = bits / 8;
  const frameBytes = bytesPerSample * channels;
  // MUST: the frame length is **derived independently** from the 4 fields above, so it is matched
  // against the block align / byte rate that fmt declares. Without this check a header whose
  // declaration and derivation disagree would pass silently, and data would be read with an
  // alignment and rate other than the writer intended.
  const blockAlign = view.getUint16(fmt.offset + 12, true);
  const byteRate = view.getUint32(fmt.offset + 8, true);
  if (blockAlign !== frameBytes) {
    throw new ModelInputError(
      `decodeWav: the declared 'fmt ' block align ${blockAlign} does not match ` +
        `${channels}ch × ${bits}bit = ${frameBytes}`,
    );
  }
  if (byteRate !== sampleRate * blockAlign) {
    throw new ModelInputError(
      `decodeWav: the declared 'fmt ' byte rate ${byteRate} does not match ` +
        `${sampleRate}Hz × block align ${blockAlign} = ${sampleRate * blockAlign}`,
    );
  }
  if (data.length % frameBytes !== 0) {
    throw new ModelInputError(
      `decodeWav: 'data' is ${data.length} bytes, ` +
        `not divisible by one frame of ${frameBytes} bytes (${channels}ch × ${bits}bit)`,
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
    // Mono skips the division (dividing by 1 gives the same value, but the intent is "as is", not "average").
    out[frame] = channels === 1 ? sum : sum / channels;
  }
  return { data: out, sampleRate };
};

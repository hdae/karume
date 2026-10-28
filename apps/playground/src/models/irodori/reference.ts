/**
 * The reference audio of voice cloning: how a chosen file becomes mono f32 at the distribution's rate
 * (the pipeline has no resampler — a rate mismatch is a `ModelInputError`). Pure parts only; the
 * browser decode is `decode-reference.ts`.
 *
 * Two decoders, and the record names which one ran:
 * - `decodeWav` (the models barrel) when the file is a PCM-16 / float-32 WAV already at the rate:
 *   the samples are the file's own, so runs compare across machines and browsers;
 * - the browser's `decodeAudioData` on an `AudioContext` at the rate otherwise (another rate, another
 *   format): the browser resamples, and its resampler is not specified — the same file can give other
 *   samples in another browser. The record keeps the sha256 of the samples that went in.
 */

/** Which decoder produced the samples, and why the browser's when it was the browser's. */
export type ReferenceDecoder =
  | { readonly kind: "decodeWav" }
  | { readonly kind: "browser"; readonly reason: string };

/**
 * The decoder for a file, given what `decodeWav` made of it: its samples at their rate, or the
 * refusal message (not a supported WAV). `decodeWav` wins only when its rate is the one needed.
 */
export const chooseDecoder = (
  wav: { readonly sampleRate: number } | { readonly refused: string },
  sampleRate: number,
): ReferenceDecoder => {
  if ("refused" in wav) {
    return { kind: "browser", reason: `decodeWav refused the file: ${wav.refused}` };
  }
  if (wav.sampleRate !== sampleRate) {
    return {
      kind: "browser",
      reason: `the WAV is ${wav.sampleRate} Hz; the browser resampled it to ${sampleRate} Hz`,
    };
  }
  return { kind: "decodeWav" };
};

/**
 * The channel average with `decodeWav`'s arithmetic (the sum in f64, divided by the channel count,
 * rounded to f32 once; mono as is), so both decoders mix the same channel samples to the same mono.
 */
export const mixToMono = (channels: readonly Float32Array[]): Float32Array<ArrayBuffer> => {
  const [first] = channels;
  if (first === undefined) throw new Error("The decoded audio has no channel");
  for (const channel of channels) {
    if (channel.length !== first.length) {
      throw new Error(`Channel lengths differ (${channel.length} vs ${first.length})`);
    }
  }
  const mono = new Float32Array(first.length);
  for (let frame = 0; frame < mono.length; frame += 1) {
    let sum = 0;
    for (const channel of channels) sum += channel[frame];
    mono[frame] = channels.length === 1 ? sum : sum / channels.length;
  }
  return mono;
};

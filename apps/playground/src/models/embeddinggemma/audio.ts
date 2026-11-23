/**
 * The audio of a row: a chosen file → mono f32 samples at the model's rate (`pipelineConfig.audio.
 * sampleRate`, 16 kHz), the `MonoAudio` an audio part holds. The pipeline has no resampler (another
 * rate is a `ModelInputError`), so the page decodes and resamples with the browser: an
 * `AudioContext` at the model's rate makes `decodeAudioData` resample to it, and the channels are
 * mixed to mono. The browser's resampler is not specified — the same file can give other samples in
 * another browser — so the record keeps the sha256 of the samples that went in. This is page code,
 * not library code: the library takes samples only.
 *
 * Nothing is cut here: an audio over 30 s is kept as it is and refused by the pipeline (its message
 * reaches the row), as upstream's 30 s truncation is not copied.
 */

import type { MonoAudio } from "@karume/models/embeddinggemma";

/** A decoded file: its name and its samples as the pipeline takes them. */
export type LoadedAudio = {
  readonly name: string;
  readonly audio: MonoAudio;
  /** The channels the file decoded to (mixed to mono). */
  readonly channels: number;
};

/**
 * The channel average (the sum in f64, divided by the channel count, rounded to f32 once; mono as
 * is) — the arithmetic of the models barrel's `decodeWav`.
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

/** The duration of an audio in seconds. */
const audioSeconds = (audio: MonoAudio): number => audio.data.length / audio.sampleRate;

/** How a row shows its audio: the name, the duration, the samples and the rate. */
export const audioSummary = (loaded: LoadedAudio): string =>
  `${loaded.name} · ${audioSeconds(loaded.audio).toFixed(2)} s · ${loaded.audio.data.length} samples` +
  ` at ${loaded.audio.sampleRate} Hz mono` +
  (loaded.channels === 1 ? "" : ` (mixed from ${loaded.channels} channels)`);

/** Decodes an audio file in the browser, resampled to `sampleRate` and mixed to mono. */
export const decodeAudioFile = async (file: File, sampleRate: number): Promise<LoadedAudio> => {
  const bytes = await file.arrayBuffer();
  const context = new AudioContext({ sampleRate });
  try {
    // decodeAudioData detaches the buffer it gets; `bytes` is not used after this.
    const decoded = await context.decodeAudioData(bytes);
    const channels = Array.from({ length: decoded.numberOfChannels }, (_, channel) =>
      decoded.getChannelData(channel),
    );
    return {
      name: file.name,
      audio: { data: mixToMono(channels), sampleRate },
      channels: decoded.numberOfChannels,
    };
  } finally {
    await context.close();
  }
};

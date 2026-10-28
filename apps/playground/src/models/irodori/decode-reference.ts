/**
 * The browser side of the reference audio: a chosen file → mono f32 at the distribution's rate, with
 * the decoder that produced it (`reference.ts` holds the rules). `decodeWav` comes from the models
 * barrel, loaded on the first file rather than at page load.
 */

import { ModelInputError } from "@karume/models/irodori";

import { chooseDecoder, mixToMono, type ReferenceDecoder } from "./reference";

export type DecodedReference = {
  readonly data: Float32Array<ArrayBuffer>;
  readonly sampleRate: number;
  readonly decoder: ReferenceDecoder;
};

/** The two decoders (injected by the tests; the page uses {@link browserDecoders}). */
export type ReferenceDecoders = {
  /** The models barrel's `decodeWav`: a `ModelInputError` when the bytes are not a supported WAV. */
  readonly decodeWav: (bytes: Uint8Array<ArrayBuffer>) => {
    readonly data: Float32Array<ArrayBuffer>;
    readonly sampleRate: number;
  };
  /** Decodes any audio the browser knows, resampled to `sampleRate`, mixed to mono. */
  readonly decodeInBrowser: (
    bytes: ArrayBuffer,
    sampleRate: number,
  ) => Promise<Float32Array<ArrayBuffer>>;
};

const decodeInBrowser = async (
  bytes: ArrayBuffer,
  sampleRate: number,
): Promise<Float32Array<ArrayBuffer>> => {
  // An AudioContext at the target rate makes decodeAudioData resample to it.
  const context = new AudioContext({ sampleRate });
  try {
    const decoded = await context.decodeAudioData(bytes);
    return mixToMono(
      Array.from({ length: decoded.numberOfChannels }, (_, channel) =>
        decoded.getChannelData(channel),
      ),
    );
  } finally {
    await context.close();
  }
};

/** The page's decoders: the barrel's `decodeWav` (imported on first use) and the browser's. */
export const browserDecoders = async (): Promise<ReferenceDecoders> => {
  const { decodeWav } = await import("@karume/models");
  return { decodeWav, decodeInBrowser };
};

export const decodeReferenceFile = async (
  file: Blob,
  sampleRate: number,
  decoders: ReferenceDecoders,
): Promise<DecodedReference> => {
  const bytes = await file.arrayBuffer();
  let wav: { readonly data: Float32Array<ArrayBuffer>; readonly sampleRate: number } | undefined;
  let probe: { readonly sampleRate: number } | { readonly refused: string };
  try {
    wav = decoders.decodeWav(new Uint8Array(bytes));
    probe = { sampleRate: wav.sampleRate };
  } catch (error) {
    // Only decodeWav's refusal of the bytes (not a supported WAV) hands the file to the browser.
    if (!(error instanceof ModelInputError)) throw error;
    probe = { refused: error.message };
  }
  const decoder = chooseDecoder(probe, sampleRate);
  if (decoder.kind === "decodeWav" && wav !== undefined) {
    return { data: wav.data, sampleRate, decoder };
  }
  // decodeAudioData detaches the buffer it gets; `bytes` is not used after this.
  return { data: await decoders.decodeInBrowser(bytes, sampleRate), sampleRate, decoder };
};

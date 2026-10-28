import { decodeWav } from "@karume/models";
import { ModelInputError } from "@karume/models/irodori";
import { describe, expect, it } from "vite-plus/test";

import {
  decodeReferenceFile,
  type ReferenceDecoders,
} from "../src/models/irodori/decode-reference.ts";

/** A 16-bit PCM mono WAV of `samples` at `rate`, written by hand. */
const monoWav = (samples: readonly number[], rate: number): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const ascii = (at: number, text: string) => {
    for (let index = 0; index < 4; index += 1) view.setUint8(at + index, text.charCodeAt(index));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, sample, true));
  return bytes;
};

/** The real decodeWav, and a browser decoder that logs its calls and returns `browserSamples`. */
const decoders = (browserSamples = new Float32Array([0.25, 0.5])) => {
  const calls: number[] = [];
  const value: ReferenceDecoders = {
    decodeWav,
    decodeInBrowser: async (_bytes, sampleRate) => {
      calls.push(sampleRate);
      return browserSamples;
    },
  };
  return { value, calls };
};

describe("decodeReferenceFile", () => {
  it("returns decodeWav's own samples for a WAV at the needed rate, without the browser", async () => {
    const wav = monoWav([16384, -8192, 1], 48000);
    const { value, calls } = decoders();
    const decoded = await decodeReferenceFile(new Blob([wav]), 48000, value);
    expect(decoded.decoder).toEqual({ kind: "decodeWav" });
    expect(decoded.sampleRate).toBe(48000);
    expect([...decoded.data]).toEqual([...decodeWav(wav).data]);
    expect(calls).toEqual([]);
  });

  it("hands a WAV at another rate to the browser at the needed rate", async () => {
    const browserSamples = new Float32Array([0.125]);
    const { value, calls } = decoders(browserSamples);
    const decoded = await decodeReferenceFile(new Blob([monoWav([1, 2], 44100)]), 48000, value);
    expect(calls).toEqual([48000]);
    expect(decoded.data).toBe(browserSamples);
    expect(decoded.decoder.kind === "browser" ? decoded.decoder.reason : "").toContain("44100 Hz");
  });

  it("hands bytes decodeWav refuses (not a WAV) to the browser, keeping the refusal", async () => {
    const { value, calls } = decoders();
    const decoded = await decodeReferenceFile(new Blob(["ID3 not a wav"]), 48000, value);
    expect(calls).toEqual([48000]);
    expect(decoded.decoder.kind).toBe("browser");
    expect(decoded.decoder.kind === "browser" ? decoded.decoder.reason : "").toMatch(
      /^decodeWav refused the file: /,
    );
  });

  it("rethrows a decodeWav failure that is not a refusal of the input", async () => {
    const failure = new TypeError("broken invariant");
    const { calls } = decoders();
    const value: ReferenceDecoders = {
      decodeWav: () => {
        throw failure;
      },
      decodeInBrowser: async () => {
        calls.push(0);
        return new Float32Array(0);
      },
    };
    await expect(decodeReferenceFile(new Blob(["x"]), 48000, value)).rejects.toBe(failure);
    expect(calls).toEqual([]);
  });

  it("relies on decodeWav refusing non-WAV bytes with a ModelInputError", () => {
    expect(() => decodeWav(new Uint8Array(4))).toThrow(ModelInputError);
  });
});

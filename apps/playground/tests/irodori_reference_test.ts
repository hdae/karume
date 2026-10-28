import { decodeWav } from "@karume/models";
import { describe, expect, it } from "vite-plus/test";

import { chooseDecoder, mixToMono } from "../src/models/irodori/reference.ts";

describe("chooseDecoder", () => {
  it("reads a WAV at the needed rate with decodeWav", () => {
    expect(chooseDecoder({ sampleRate: 48000 }, 48000)).toEqual({ kind: "decodeWav" });
  });

  it("hands a WAV at another rate to the browser and says why", () => {
    const decoder = chooseDecoder({ sampleRate: 44100 }, 48000);
    expect(decoder.kind).toBe("browser");
    expect(decoder.kind === "browser" ? decoder.reason : "").toContain("44100 Hz");
  });

  it("hands a file decodeWav refused to the browser, keeping the refusal", () => {
    const decoder = chooseDecoder({ refused: "not a RIFF/WAVE header" }, 48000);
    expect(decoder).toEqual({
      kind: "browser",
      reason: "decodeWav refused the file: not a RIFF/WAVE header",
    });
  });
});

/** A 16-bit PCM stereo WAV of `left` / `right` (interleaved), written by hand. */
const stereoWav = (left: readonly number[], right: readonly number[], rate: number) => {
  const frames = left.length;
  const bytes = new Uint8Array(44 + frames * 4);
  const view = new DataView(bytes.buffer);
  const ascii = (at: number, text: string) => {
    for (let index = 0; index < 4; index += 1) view.setUint8(at + index, text.charCodeAt(index));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + frames * 4, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, frames * 4, true);
  for (let frame = 0; frame < frames; frame += 1) {
    view.setInt16(44 + frame * 4, left[frame], true);
    view.setInt16(46 + frame * 4, right[frame], true);
  }
  return bytes;
};

describe("mixToMono", () => {
  it("returns a mono channel as is", () => {
    const channel = new Float32Array([0.1, -0.7, 0.3]);
    expect([...mixToMono([channel])]).toEqual([...channel]);
  });

  it("averages the channels exactly as decodeWav does", () => {
    const left = [1, 3, -32768, 32767, 12345];
    const right = [2, -3, 32767, 32767, -1];
    const decoded = decodeWav(stereoWav(left, right, 48000));
    const mixed = mixToMono([
      new Float32Array(left.map((value) => value / 32768)),
      new Float32Array(right.map((value) => value / 32768)),
    ]);
    expect([...mixed]).toEqual([...decoded.data]);
  });

  it("refuses channels of different lengths and no channel at all", () => {
    expect(() => mixToMono([new Float32Array(2), new Float32Array(3)])).toThrow("lengths differ");
    expect(() => mixToMono([])).toThrow("no channel");
  });
});

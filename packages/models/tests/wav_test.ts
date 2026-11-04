// Behaviour tests of reading and writing WAV (RIFF). They need neither a GPU nor real assets.
//
// `decodeWav` is the only entry point that takes reference audio, and **a mixed-up convention does
// not raise** (an int16 scale off by 1 only yields "a waveform at a slightly different volume").
// So the values themselves are pinned.
//
// When the golden `outputs/test-assets/goldens/dacvae-32dim/host/meta.json` is present, it is also
// compared with its `wavScale` (what the upstream reader measured at both ends of int16); without it
// only that one case SKIPs, or FAILs when the test assets are a complete fetch of the pin (ADR 0139).

import { assert, assertEquals, assertThrows } from "@std/assert";
import { describe, it, test } from "vite-plus/test";
import { decodeWav, encodeWav } from "../src/audio/wav.ts";
import { ModelInputError } from "../src/errors.ts";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { readTextIfPresent } from "./helpers/read-if-present.ts";

/** The golden `meta.json` (written by `tools/export-recipes/irodori/dacvae/host.py`). */
const HOST_GOLDEN = new URL("host/meta.json", goldenDir("dacvae-32dim"));
const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts — the assets gate accepts only a complete fetch; " +
  "--include 'goldens/dacvae-32dim/host/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone";

const goldenText = await readTextIfPresent(HOST_GOLDEN);
// An absent golden SKIPs unless `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139),
// where the dataset at the pin lacks it and the case FAILs.
const GOLDEN_AVAILABLE = goldenText !== undefined || testAssetsFetched();
if (!GOLDEN_AVAILABLE) {
  console.warn(
    `[karume] The wav scale comparison SKIPs: missing ${HOST_GOLDEN.pathname} (${FETCH_HINT})`,
  );
}

const ascii = (bytes: Uint8Array, offset: number, text: string): void => {
  for (let i = 0; i < text.length; i += 1) bytes[offset + i] = text.charCodeAt(i);
};

/**
 * Builds a WAV of any format / bit depth / channel count (the base that the error cases break one
 * point at a time). `payload` is the content of the data chunk as is.
 */
const buildWav = (options: {
  readonly format: number;
  readonly channels: number;
  readonly sampleRate: number;
  readonly bits: number;
  readonly payload: Uint8Array;
  /** An unknown chunk inserted before `fmt ` (for checking that it is skipped). */
  readonly extraChunk?: { readonly id: string; readonly length: number };
}): Uint8Array<ArrayBuffer> => {
  const { format, channels, sampleRate, bits, payload, extraChunk } = options;
  const extraBytes = extraChunk === undefined ? 0 : 8 + extraChunk.length + (extraChunk.length % 2);
  const total = 12 + extraBytes + 8 + 16 + 8 + payload.length;
  const bytes = new Uint8Array(total);
  const view = new DataView(bytes.buffer);
  ascii(bytes, 0, "RIFF");
  view.setUint32(4, total - 8, true);
  ascii(bytes, 8, "WAVE");
  let cursor = 12;
  if (extraChunk !== undefined) {
    ascii(bytes, cursor, extraChunk.id);
    view.setUint32(cursor + 4, extraChunk.length, true);
    cursor += extraBytes;
  }
  ascii(bytes, cursor, "fmt ");
  view.setUint32(cursor + 4, 16, true);
  view.setUint16(cursor + 8, format, true);
  view.setUint16(cursor + 10, channels, true);
  view.setUint32(cursor + 12, sampleRate, true);
  view.setUint32(cursor + 16, (sampleRate * channels * bits) / 8, true);
  view.setUint16(cursor + 20, (channels * bits) / 8, true);
  view.setUint16(cursor + 22, bits, true);
  cursor += 24;
  ascii(bytes, cursor, "data");
  view.setUint32(cursor + 4, payload.length, true);
  bytes.set(payload, cursor + 8);
  return bytes;
};

const int16Payload = (values: readonly number[]): Uint8Array => {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setInt16(index * 2, value, true));
  return bytes;
};

const float32Payload = (values: readonly number[]): Uint8Array => {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return bytes;
};

// ---- int16 scale (the convention itself) ------------------------------------

test("decodeWav: divides int16 by 32768 (the bottom −32768 is exactly −1.0)", () => {
  const probe = [-32768, -32767, -1, 0, 1, 16384, 32767];
  const decoded = decodeWav(
    buildWav({ format: 1, channels: 1, sampleRate: 48000, bits: 16, payload: int16Payload(probe) }),
  );
  assertEquals(decoded.sampleRate, 48000);
  assertEquals(
    Array.from(decoded.data),
    [-1, -0.999969482421875, -3.0517578125e-5, 0, 3.0517578125e-5, 0.5, 0.999969482421875],
    "the int16 → f32 scale is not /32768 (on real audio it differs from /32767 by only 3e-5, " +
      "which the tolerance of a golden comparison hides)",
  );
});

test.skipIf(!GOLDEN_AVAILABLE)(
  "decodeWav: the int16 scale matches what the upstream reader measured in the golden (wavScale)",
  () => {
    if (goldenText === undefined) {
      throw new Error(
        `${HOST_GOLDEN.pathname} absent, but outputs/test-assets/ holds a complete fetch of the ` +
          "pinned revision: the test-asset dataset at the pin lacks it",
      );
    }
    const meta = JSON.parse(goldenText) as {
      readonly wavScale: {
        readonly probeInt16: readonly number[];
        readonly probeFloat: readonly number[];
      };
    };
    const decoded = decodeWav(
      buildWav({
        format: 1,
        channels: 1,
        sampleRate: 48000,
        bits: 16,
        payload: int16Payload(meta.wavScale.probeInt16),
      }),
    );
    assertEquals(
      Array.from(decoded.data),
      [...meta.wavScale.probeFloat],
      "differs from the upstream (soundfile) int16 normalization",
    );
  },
);

// ---- Round trip (pins the asymmetry of the read and write scales itself) --------

test("encodeWav → decodeWav: a round trip is off by about 1 LSB from the asymmetry of ×32767 and /32768", () => {
  const original = Float32Array.of(-1, -0.5, 0, 0.5, 1);
  const decoded = decodeWav(encodeWav(original, 48000));
  assertEquals(decoded.sampleRate, 48000);
  assertEquals(decoded.data.length, original.length);
  // Full scale is written as 32767 and read with 32768, so it **does not come back exactly**. If
  // this is not consistent, the code disagrees with either the listening-comparison partner (the
  // torch script) or the reader of the reference audio.
  assertEquals(decoded.data[0], -0.999969482421875, "round trip of −1.0");
  assertEquals(decoded.data[4], 0.999969482421875, "round trip of +1.0");
  // Only 0 round-trips exactly (0 stays 0 even with an asymmetric scale).
  assertEquals(decoded.data[2], 0);
  // The error stays within one 16-bit LSB (1/32768) over the whole range.
  for (let i = 0; i < original.length; i += 1) {
    const error = Math.abs(decoded.data[i] - original[i]);
    assertEquals(
      error <= 1 / 32768,
      true,
      `element ${i}: the round-trip error ${error} exceeds 1 LSB`,
    );
  }
});

// ---- f32 WAV / stereo --------------------------------------------------

test("decodeWav: IEEE float 32-bit is the identity (no rounding, no division)", () => {
  // Every value is exactly representable in f32 (the point is to see that no rounding happens).
  const values = [-1, -0.3333333432674408, 0, 3.0517578125e-5, 0.75, 2.5];
  const decoded = decodeWav(
    buildWav({
      format: 3,
      channels: 1,
      sampleRate: 44100,
      bits: 32,
      payload: float32Payload(values),
    }),
  );
  assertEquals(decoded.sampleRate, 44100);
  // Out-of-range values (2.5) are not clipped either — normalization is the caller's step.
  assertEquals(Array.from(decoded.data), values);
});

test("decodeWav: downmixes multiple channels to mono by the channel mean", () => {
  // 2ch interleaved (L, R, L, R, …).
  const decoded = decodeWav(
    buildWav({
      format: 3,
      channels: 2,
      sampleRate: 48000,
      bits: 32,
      payload: float32Payload([1, 0, 0.5, -0.5, -1, -1]),
    }),
  );
  assertEquals(Array.from(decoded.data), [0.5, 0, -1]);
});

test("decodeWav: skips unknown chunks (LIST and the like)", () => {
  const decoded = decodeWav(
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([16384]),
      // Odd length — a reader that cannot step over RIFF's 1-byte padding loses the next chunk
      // boundary.
      extraChunk: { id: "LIST", length: 7 },
    }),
  );
  assertEquals(Array.from(decoded.data), [0.5]);
});

test("decodeWav: does not scan physical bytes left over beyond the boundary RIFF declares", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([16384]),
  });
  // Adds a chunk with a bogus length **beyond** the logical end (the RIFF declaration =
  // complete.length). An implementation that stops scanning at the physical length reads this far
  // and fails with "not enough bytes remain".
  const padded = new Uint8Array(complete.length + 8);
  padded.set(complete);
  ascii(padded, complete.length, "junk");
  new DataView(padded.buffer).setUint32(complete.length + 4, 0xffff_ffff, true);
  const decoded = decodeWav(padded);
  assertEquals(Array.from(decoded.data), [0.5]);
});

// ---- fail loudly ---------------------------------------------------------

test("decodeWav: rejects bytes that are not RIFF/WAVE", () => {
  const bytes = new Uint8Array(64);
  ascii(bytes, 0, "RIFX");
  ascii(bytes, 8, "WAVE");
  assertThrows(() => decodeWav(bytes), ModelInputError, "not a RIFF/WAVE header");
  assertThrows(() => decodeWav(new Uint8Array(8)), ModelInputError, "only 8 bytes");
});

test("decodeWav: rejects unsupported formats / bit depths (no silent approximation)", () => {
  // 24-bit PCM.
  assertThrows(
    () =>
      decodeWav(
        buildWav({
          format: 1,
          channels: 1,
          sampleRate: 48000,
          bits: 24,
          payload: new Uint8Array(6),
        }),
      ),
    ModelInputError,
    "format 1 / 24bit is not supported",
  );
  // A-law.
  assertThrows(
    () =>
      decodeWav(
        buildWav({ format: 6, channels: 1, sampleRate: 8000, bits: 8, payload: new Uint8Array(4) }),
      ),
    ModelInputError,
    "format 6 / 8bit is not supported",
  );
  // WAVE_FORMAT_EXTENSIBLE is not accepted, since its kind is unknown without reading SubFormat.
  assertThrows(
    () =>
      decodeWav(
        buildWav({
          format: 0xfffe,
          channels: 1,
          sampleRate: 48000,
          bits: 16,
          payload: new Uint8Array(4),
        }),
      ),
    ModelInputError,
    "is not supported",
  );
});

test("decodeWav: rejects a file without 'data' or whose length disagrees with the declaration", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([1, 2, 3, 4]),
  });
  // Makes only the data chunk's declared length larger than its content (right after the 8 bytes of
  // 'data' + length).
  const truncated = complete.slice(0, complete.length - 2) as Uint8Array<ArrayBuffer>;
  assertThrows(() => decodeWav(truncated), ModelInputError, "bytes remain");
  // A file with only fmt (no data). The RIFF declared size is also set to the 36-byte container —
  // otherwise it fails first at the "the container is truncated" check and never reaches the
  // missing-data check.
  const fmtOnly = complete.slice(0, 36) as Uint8Array<ArrayBuffer>;
  new DataView(fmtOnly.buffer).setUint32(4, 36 - 8, true);
  assertThrows(() => decodeWav(fmtOnly), ModelInputError, "'data' chunk is missing");
});

/**
 * Appends one chunk to the end of the container and updates the RIFF declared size (odd lengths get 1
 * byte of padding).
 */
const appendChunk = (
  wav: Uint8Array<ArrayBuffer>,
  id: string,
  body: Uint8Array,
): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(wav.length + 8 + body.length + (body.length % 2));
  bytes.set(wav);
  ascii(bytes, wav.length, id);
  const view = new DataView(bytes.buffer);
  view.setUint32(wav.length + 4, body.length, true);
  bytes.set(body, wav.length + 8);
  view.setUint32(4, bytes.length - 8, true);
  return bytes;
};

describe("when decodeWav receives a container with two chunks of the same id", () => {
  const single = (): Uint8Array<ArrayBuffer> =>
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([16384]),
    });

  it("rejects two 'data' chunks instead of letting the later one win", () => {
    const twoData = appendChunk(single(), "data", int16Payload([-16384]));
    assertThrows(() => decodeWav(twoData), ModelInputError, "chunk 'data' appears twice");
  });

  it("rejects two 'fmt ' chunks instead of letting the later one win", () => {
    const wav = single();
    // Copies the content of the first `fmt ` (16 bytes from offset 20) as is — a duplicate is a
    // duplicate even with the same content.
    const twoFmt = appendChunk(wav, "fmt ", wav.slice(20, 36));
    assertThrows(() => decodeWav(twoFmt), ModelInputError, "chunk 'fmt ' appears twice");
  });

  it("accepts two of an unknown chunk it skips (LIST)", () => {
    const twoList = appendChunk(
      buildWav({
        format: 1,
        channels: 1,
        sampleRate: 48000,
        bits: 16,
        payload: int16Payload([16384]),
        extraChunk: { id: "LIST", length: 7 },
      }),
      "LIST",
      new Uint8Array(3),
    );
    assertEquals(Array.from(decodeWav(twoList).data), [0.5]);
  });
});

describe("when decodeWav receives a container whose chunk scan cannot end exactly at RIFF's logical end", () => {
  const single = (): Uint8Array<ArrayBuffer> =>
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([16384]),
    });

  /**
   * Appends raw bytes to the end of the container and sets the RIFF declared size to the real length.
   */
  const appendRaw = (
    wav: Uint8Array<ArrayBuffer>,
    tail: Uint8Array<ArrayBuffer>,
  ): Uint8Array<ArrayBuffer> => {
    const bytes = new Uint8Array(wav.length + tail.length);
    bytes.set(wav);
    bytes.set(tail, wav.length);
    new DataView(bytes.buffer).setUint32(4, bytes.length - 8, true);
    return bytes;
  };

  /** An odd-length chunk written without its padding (id + length + content, no padding). */
  const unpaddedChunk = (id: string, body: readonly number[]): Uint8Array<ArrayBuffer> => {
    const bytes = new Uint8Array(8 + body.length);
    ascii(bytes, 0, id);
    new DataView(bytes.buffer).setUint32(4, body.length, true);
    bytes.set(body, 8);
    return bytes;
  };

  it("rejects a header fragment shorter than 8 bytes (a 4-byte JUNK) before the end", () => {
    const fragment = new Uint8Array(4);
    ascii(fragment, 0, "JUNK");
    assertThrows(
      () => decodeWav(appendRaw(single(), fragment)),
      ModelInputError,
      "a fragment of 4 bytes before the end of RIFF",
    );
  });

  it("rejects the form where the header after an odd-length chunk missing its padding is read 1 byte off", () => {
    // With LIST placed after JUNK (length 1, no padding), the reader skips 1 byte of padding, reads
    // LIST's header starting 1 byte to the right, and every later chunk boundary shifts.
    const tail = new Uint8Array([
      ...unpaddedChunk("JUNK", [7]),
      ...unpaddedChunk("LIST", [0, 0, 0, 0]),
    ]);
    assertThrows(() => decodeWav(appendRaw(single(), tail)), ModelInputError, "a fragment of");
  });

  it("accepts the form where only the last chunk lacks its padding (common among real writers, and nothing follows it)", () => {
    const decoded = decodeWav(appendRaw(single(), unpaddedChunk("JUNK", [7])));
    assertEquals(Array.from(decoded.data), [0.5]);
  });
});

test("decodeWav: rejects data that does not divide into frames", () => {
  assertThrows(
    () =>
      decodeWav(
        buildWav({
          format: 1,
          channels: 2,
          sampleRate: 48000,
          bits: 16,
          // 6 bytes although 2ch × 16bit = 4 bytes per frame.
          payload: new Uint8Array(6),
        }),
      ),
    ModelInputError,
    "not divisible",
  );
});

test("decodeWav: rejects a container whose RIFF declared size exceeds its physical length (truncated)", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([1, 2]),
  });
  // Declares 16 bytes more than the physical length. The chunks are all consistent, so an
  // implementation that does not read the declared size reads to the end without trouble.
  new DataView(complete.buffer).setUint32(4, complete.length - 8 + 16, true);
  assertThrows(
    () => decodeWav(complete),
    ModelInputError,
    `RIFF declares ${complete.length + 8} bytes`,
  );
});

test("decodeWav: rejects a file whose data declaration runs past RIFF's logical end", () => {
  const complete = buildWav({
    format: 1,
    channels: 1,
    sampleRate: 48000,
    bits: 16,
    payload: int16Payload([1, 2, 3, 4]),
  });
  // Leaves 8 extra physical bytes and extends data's declared length by as much. An implementation
  // that takes the boundary from the physical length lets it through and reads the 8 bytes outside
  // the container as 4 samples.
  const padded = new Uint8Array(complete.length + 8);
  padded.set(complete);
  const view = new DataView(padded.buffer);
  view.setUint32(40, view.getUint32(40, true) + 8, true); // the data chunk's length field (standard 44-byte header)
  assertThrows(() => decodeWav(padded), ModelInputError, "chunk 'data' declares 16 bytes");
});

test("decodeWav: rejects a file whose fmt block align / byte rate contradicts the derived values", () => {
  const build = () =>
    buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 16,
      payload: int16Payload([1, 2]),
    });
  // Breaks only the block align (2 bytes for 1ch × 16bit). Offset 32 in the standard fmt layout.
  const badAlign = build();
  new DataView(badAlign.buffer).setUint16(32, 4, true);
  assertThrows(
    () => decodeWav(badAlign),
    ModelInputError,
    "block align 4 does not match 1ch × 16bit = 2",
  );
  // Breaks only the byte rate (48000 × 2 = 96000). Offset 28 in the standard fmt layout.
  const badRate = build();
  new DataView(badRate.buffer).setUint32(28, 192000, true);
  assertThrows(
    () => decodeWav(badRate),
    ModelInputError,
    "byte rate 192000 does not match 48000Hz × block align 2 = 96000",
  );
});

test("encodeWav: rejects non-finite samples with their position and value (does not turn them into silence or clipping)", () => {
  // The clip's `Math.max` / `Math.min` pass NaN through, and `Math.round(NaN)` → `setInt16` writes
  // 0. ±Infinity sticks to full scale. Neither raises.
  for (const [value, text] of [
    [NaN, "NaN"],
    [Infinity, "Infinity"],
    [-Infinity, "-Infinity"],
  ] as const) {
    const samples = Float32Array.of(0.1, 0.2, value, 0.3);
    assertThrows(
      () => encodeWav(samples, 48000),
      ModelInputError,
      `Sample 2 (${text}) is non-finite`,
    );
  }
  // Out-of-range finite values are clipped and pass as before (the check does not reject everything).
  const clipped = new DataView(encodeWav(Float32Array.of(2, -2), 48000).buffer);
  assertEquals(clipped.getInt16(44, true), 32767);
  assertEquals(clipped.getInt16(46, true), -32767);
});

test("encodeWav: rejects a sampleRate / byte rate that does not fit in u32", () => {
  const samples = Float32Array.of(0, 0.5);
  // Exactly at the limit (byte rate = 0xffff_fffe) passes, and the header carries that value as is.
  const view = new DataView(encodeWav(samples, 0x7fff_ffff).buffer);
  assertEquals(view.getUint32(24, true), 0x7fff_ffff, "sample rate");
  assertEquals(view.getUint32(28, true), 0xffff_fffe, "byte rate");
  // One above, sampleRate itself fits in u32 but the byte rate (×2) overflows. Without the check,
  // `setUint32` wraps it mod 2^32 and a **valid WAV** declaring byte rate 0 comes out.
  assertThrows(() => encodeWav(samples, 0x8000_0000), ModelInputError, "byte rate 4294967296");
  // A sampleRate that itself exceeds u32 fails at the same check.
  assertThrows(() => encodeWav(samples, 0x1_0000_0000), ModelInputError, "does not fit in u32");
});

test("encodeWav: rejects a sample count whose RIFF chunk length exceeds u32", () => {
  // The limit is (0xffff_ffff − 36) / 2 = 2147483629 samples. A real one would need more than 8 GB,
  // so a stand-in that only claims a length exercises the bound computation (`encodeWav` checks the
  // length **before** allocating the output). The exactly-at-the-limit side would need a 4 GB
  // allocation, so it is not here (the existing round-trip test covers the passing side).
  const huge = { length: 2_147_483_630 } as unknown as Float32Array;
  assertThrows(
    () => encodeWav(huge, 48000),
    ModelInputError,
    "RIFF chunk length 4294967296 does not fit in u32",
  );
});

// ---- Classifying whether a failure is caused by the input (ADR 0107) ---------------------------

describe("when a host sorts failures of the shared audio layer into 400 / 500", () => {
  it("catches a parse failure of the bytes the host passed as ModelInputError", () => {
    const notRiff = new Uint8Array(64);
    ascii(notRiff, 0, "RIFX");
    ascii(notRiff, 8, "WAVE");
    assert(assertThrows(() => decodeWav(notRiff)) instanceof ModelInputError);
    // The format check (an unsupported bit depth) lands in the same branch.
    const unsupported = buildWav({
      format: 1,
      channels: 1,
      sampleRate: 48000,
      bits: 24,
      payload: new Uint8Array(6),
    });
    assert(assertThrows(() => decodeWav(unsupported)) instanceof ModelInputError);
  });

  it("catches violations in the waveform / sample rate the caller passed as ModelInputError too", () => {
    assert(assertThrows(() => encodeWav(new Float32Array(1), 0)) instanceof ModelInputError);
    assert(
      assertThrows(() => encodeWav(Float32Array.of(0, NaN), 48000)) instanceof ModelInputError,
    );
  });

  // NOTE: a broken internal invariant (a plain Error that is **not** ModelInputError) has no path
  // to reach this layer. All 17 throws in `wav.ts` look only at the caller's bytes / waveform /
  // sample rate, and none comes from assets or wiring. The "not ModelInputError" side is covered by
  // the shared image layer (`resizePlaneF32` in `image_preprocess_test.ts` — an internal helper not
  // on the public surface).
});

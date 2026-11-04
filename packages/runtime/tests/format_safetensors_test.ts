import { test } from "vite-plus/test";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import {
  parseSafetensors,
  parseSafetensorsHeader,
  SafetensorsError,
  safetensorsHeaderLength,
  tensorBytes,
} from "../src/format/safetensors.ts";
import {
  buildSafetensors,
  f32Bytes,
  packSafetensors,
  packSafetensorsRaw,
} from "./helpers/safetensors.ts";

const F32_2 = f32Bytes([1, 2]);

test("parseSafetensors: a valid file yields the tensor table and __metadata__", () => {
  const buffer = buildSafetensors(
    [
      { name: "a", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 2]) },
      { name: "b", dtype: "I8", shape: [3], data: new Uint8Array([7, 8, 9]) },
    ],
    { note: "{}", extra: "x" },
  );
  const file = parseSafetensors(buffer);

  assertEquals([...file.tensors.keys()].sort(), ["a", "b"]);
  assertEquals(file.metadata.get("note"), "{}");
  assertEquals(file.metadata.get("extra"), "x");

  const a = file.tensors.get("a");
  assertEquals(a?.dtype, "F32");
  assertEquals(a?.shape, [2, 1]);
  assertEquals(a?.byteLength, 8);
});

test("parseSafetensors: a view does not copy and refers to the original ArrayBuffer", () => {
  const buffer = buildSafetensors([{ name: "a", dtype: "F32", shape: [2], data: F32_2 }]);
  const file = parseSafetensors(buffer);
  const view = file.tensors.get("a");
  if (view === undefined) throw new Error("no tensor a");

  const bytes = tensorBytes(file, view);
  assertStrictEquals(bytes.buffer, buffer);
  assertEquals(bytes.byteOffset, view.byteOffset);
  assertEquals(new Float32Array(buffer, view.byteOffset, 2), new Float32Array([1, 2]));
});

test("parseSafetensors: rejects a file without even a header length", () => {
  assertThrows(() => parseSafetensors(new ArrayBuffer(4)), SafetensorsError, "is too short");
});

test("parseSafetensors: rejects a header length that exceeds the file length", () => {
  const buffer = new ArrayBuffer(16);
  new DataView(buffer).setBigUint64(0, 4096n, true);
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "the file length");
});

test("parseSafetensors: rejects broken header JSON", () => {
  assertThrows(
    () => parseSafetensors(packSafetensorsRaw("{not json", new Uint8Array(0))),
    SafetensorsError,
    "header JSON",
  );
});

test("parseSafetensors: rejects a header that is not an object", () => {
  assertThrows(
    () => parseSafetensors(packSafetensorsRaw("[]", new Uint8Array(0))),
    SafetensorsError,
    "is not an object",
  );
});

test("parseSafetensors: rejects an unsupported dtype", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F64", shape: [1], data_offsets: [0, 8] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "unsupported dtype");

  // The packed 4-bit / 2-bit that were dialects of the old distribution (spellings the official
  // safetensors does not have) MUST fail at the same check — accepting them silently would bring
  // the quantized storage that moved into the container (`krm`) back through the asset path.
  for (const dtype of ["I4", "I2"]) {
    assertThrows(
      () =>
        parseSafetensors(
          packSafetensors({ w: { dtype, shape: [8], data_offsets: [0, 4] } }, new Uint8Array(4)),
        ),
      SafetensorsError,
      "unsupported dtype",
    );
  }
});

test("parseSafetensors: rejects a size mismatch between shape and data_offsets", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [2], data_offsets: [0, 4] } },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "size mismatch");
});

test("parseSafetensors: rejects negative dimensions and non-integer declarations", () => {
  const negative = packSafetensors(
    { a: { dtype: "F32", shape: [-1], data_offsets: [0, 0] } },
    new Uint8Array(0),
  );
  assertThrows(() => parseSafetensors(negative), SafetensorsError, "shape element");

  const fractional = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [0.5, 4.5] } },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(fractional), SafetensorsError, "data_offsets");
});

test("parseSafetensors: rejects an out-of-range data section", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [2], data_offsets: [0, 8] } },
    new Uint8Array(4),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "outside the data section");
});

test("parseSafetensors: rejects overlapping regions", () => {
  const buffer = packSafetensors(
    {
      a: { dtype: "F32", shape: [2], data_offsets: [0, 8] },
      b: { dtype: "F32", shape: [2], data_offsets: [4, 12] },
    },
    new Uint8Array(12),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "overlaps");
});

test("parseSafetensors: rejects gaps between tensors", () => {
  const buffer = packSafetensors(
    {
      a: { dtype: "F32", shape: [1], data_offsets: [0, 4] },
      b: { dtype: "F32", shape: [1], data_offsets: [8, 12] },
    },
    new Uint8Array(12),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "unused range");
});

test("parseSafetensors: rejects an unused trailing region", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [0, 4] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "at the end of the data section");
});

test("parseSafetensors: rejects a tensor not aligned to its element size", () => {
  // Placing an F32 right after a 1-byte I8 gives an offset of 1 from the start of the data section,
  // and no view can be created.
  const buffer = packSafetensors(
    {
      a: { dtype: "I8", shape: [1], data_offsets: [0, 1] },
      b: { dtype: "F32", shape: [1], data_offsets: [1, 5] },
    },
    new Uint8Array(5),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "is not aligned");
});

test("parseSafetensors: rejects non-string values in __metadata__", () => {
  const buffer = packSafetensors({ __metadata__: { k: 1 } }, new Uint8Array(0));
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "is not a string");
});

/**
 * The remaining guard clauses (header length / header encoding / declaration types / element
 * count). Each would likely fail with a different message in a downstream check and would not be a
 * silent wrong value, but **the attribution of the diagnostic read by whoever rebuilds the
 * distribution** changes ("the header has the wrong type" reported as "size mismatch" sends them
 * to fix the wrong place).
 */
test("parseSafetensors: header length and header table type guards fail with their own messages", () => {
  // ① The header length exceeds the safe-integer range (checked before converting to Number)
  const huge = new ArrayBuffer(16);
  new DataView(huge).setBigUint64(0, 2n ** 60n, true);
  assertThrows(() => parseSafetensors(huge), SafetensorsError, "exceeds the safe integer range");

  // ② The header is invalid UTF-8 (0xff alone is an invalid byte)
  const invalidUtf8 = new ArrayBuffer(16);
  new DataView(invalidUtf8).setBigUint64(0, 8n, true);
  new Uint8Array(invalidUtf8, 8).fill(0xff);
  assertThrows(() => parseSafetensors(invalidUtf8), SafetensorsError, "invalid UTF-8");

  // ③ A header entry is not an object
  assertThrows(
    () => parseSafetensors(packSafetensors({ a: 5 }, new Uint8Array(0))),
    SafetensorsError,
    "header entry is not an object",
  );

  // ④ shape is not an array
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors({ a: { dtype: "F32", shape: "x", data_offsets: [0, 4] } }, F32_2),
      ),
    SafetensorsError,
    "shape is not an array",
  );

  // ⑤ data_offsets is not an array / does not have 2 elements
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors({ a: { dtype: "F32", shape: [1], data_offsets: 0 } }, new Uint8Array(4)),
      ),
    SafetensorsError,
    "data_offsets is not an array",
  );
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors({ a: { dtype: "F32", shape: [1], data_offsets: [0] } }, new Uint8Array(4)),
      ),
    SafetensorsError,
    "data_offsets has 1 elements",
  );

  // ⑥ The element count exceeds the safe-integer range (reachable from the declaration alone — no
  // real data needed)
  assertThrows(
    () =>
      parseSafetensors(
        packSafetensors(
          { a: { dtype: "F32", shape: [2 ** 30, 2 ** 30], data_offsets: [0, 0] } },
          new Uint8Array(0),
        ),
      ),
    SafetensorsError,
    "element count exceeds the safe integer range",
  );

  // ⑦ __metadata__ is not an object (an array is of type object in JSON but is not a table)
  assertThrows(
    () => parseSafetensors(packSafetensors({ __metadata__: [] }, new Uint8Array(0))),
    SafetensorsError,
    "__metadata__ is not an object",
  );
});

test("parseSafetensors: rejects reversed data_offsets", () => {
  const buffer = packSafetensors(
    { a: { dtype: "F32", shape: [1], data_offsets: [8, 4] } },
    new Uint8Array(8),
  );
  assertThrows(() => parseSafetensors(buffer), SafetensorsError, "reversed");
});

/**
 * The surface that parses only the header (a partial application for callers that read ranges).
 * Pins that it yields the same table as a full parse without holding a single byte of the data
 * section, and that mixing up prefix / file length does not pass silently.
 */
const HEADER_FIXTURE = buildSafetensors(
  [
    { name: "a", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 2]) },
    { name: "b", dtype: "I8", shape: [3], data: new Uint8Array([7, 8, 9]) },
  ],
  { note: "{}", extra: "x" },
);

test("parseSafetensorsHeader: returns the same table as a full parse from the header range alone", () => {
  const full = parseSafetensors(HEADER_FIXTURE);

  // The caller's two-step read: read 8 bytes → learn the header length N → read 8+N bytes again.
  const headerLength = safetensorsHeaderLength(new Uint8Array(HEADER_FIXTURE, 0, 8));
  const prefix = new Uint8Array(HEADER_FIXTURE, 0, 8 + headerLength);
  const header = parseSafetensorsHeader(prefix, HEADER_FIXTURE.byteLength);

  assertEquals(header.dataStart, 8 + headerLength);
  assertEquals([...header.metadata], [...full.metadata]);
  // byteOffset stays absolute from the start of the file — the caller passes it as is to range
  // reads.
  assertEquals([...header.tensors], [...full.tensors]);

  // A prefix over-read by a fixed-length read (e.g. 64 KiB) gives the same table — the slicing is
  // closed within 8+N.
  assertEquals(
    [
      ...parseSafetensorsHeader(
        new Uint8Array(HEADER_FIXTURE, 0, 8 + headerLength + 2),
        HEADER_FIXTURE.byteLength,
      ).tensors,
    ],
    [...full.tensors],
  );
  // The success path passing the prefix as an `ArrayBuffer` (the same table as the view version).
  assertEquals(
    [
      ...parseSafetensorsHeader(
        HEADER_FIXTURE.slice(0, 8 + headerLength),
        HEADER_FIXTURE.byteLength,
      ).tensors,
    ],
    [...full.tensors],
  );

  // A prefix placed in the middle of a container (through the view's byteOffset) gives the same
  // result.
  const vessel = new Uint8Array(new ArrayBuffer(16 + prefix.byteLength));
  vessel.set(prefix, 16);
  const offsetPrefix = vessel.subarray(16);
  assertEquals(safetensorsHeaderLength(offsetPrefix), headerLength);
  assertEquals(
    [...parseSafetensorsHeader(offsetPrefix, HEADER_FIXTURE.byteLength).tensors],
    [...full.tensors],
  );
});

test("parseSafetensorsHeader: a prefix cut off inside the header fails with the required length in the message", () => {
  const need = 8 + safetensorsHeaderLength(HEADER_FIXTURE);
  assertThrows(
    () =>
      parseSafetensorsHeader(
        new Uint8Array(HEADER_FIXTURE, 0, need - 1),
        HEADER_FIXTURE.byteLength,
      ),
    SafetensorsError,
    `needs the first ${need} bytes`,
  );
});

test("parseSafetensorsHeader: a wrong file length fails with the same message as a full parse", () => {
  // Longer than the real length = there is an uncovered tail / shorter = the declarations run past
  // the data section.
  assertThrows(
    () => parseSafetensorsHeader(HEADER_FIXTURE, HEADER_FIXTURE.byteLength + 8),
    SafetensorsError,
    "at the end of the data section",
  );
  assertThrows(
    () => parseSafetensorsHeader(HEADER_FIXTURE, HEADER_FIXTURE.byteLength - 4),
    SafetensorsError,
    "outside the data section",
  );
  assertThrows(
    () => parseSafetensors(HEADER_FIXTURE.slice(0, HEADER_FIXTURE.byteLength - 4)),
    SafetensorsError,
    "outside the data section",
  );

  assertThrows(
    () => parseSafetensorsHeader(HEADER_FIXTURE, -1),
    SafetensorsError,
    "is not a non-negative integer",
  );
  assertThrows(() => parseSafetensorsHeader(HEADER_FIXTURE, 4), SafetensorsError, "is too short");
});

test("safetensorsHeaderLength: rejects a prefix shorter than 8 bytes", () => {
  assertThrows(
    () => safetensorsHeaderLength(new ArrayBuffer(7)),
    SafetensorsError,
    "needs the first 8 bytes",
  );
  assertThrows(
    () => safetensorsHeaderLength(new Uint8Array(new ArrayBuffer(7))),
    SafetensorsError,
    "needs the first 8 bytes",
  );
});

/**
 * A buffer where extra bytes follow a valid file (read into the start of a larger container, say).
 * The whole buffer is checked as the file's real length; pins that the slack does not pass
 * silently.
 */
const PREFIX_FILE = buildSafetensors([
  { name: "a", dtype: "F32", shape: [2], data: f32Bytes([1, 2]) },
]);

test("parseSafetensors: a buffer with slack after a valid file is rejected as an unused trailing region", () => {
  // The real length without the slack is accepted on its own — the control showing the slack is
  // the only reason for the failure.
  assertEquals(parseSafetensors(PREFIX_FILE).tensors.get("a")?.shape, [2]);

  const buffer = new ArrayBuffer(PREFIX_FILE.byteLength + 64);
  const bytes = new Uint8Array(buffer);
  bytes.set(new Uint8Array(PREFIX_FILE));
  bytes.fill(0xab, PREFIX_FILE.byteLength);
  assertThrows(
    () => parseSafetensors(buffer),
    SafetensorsError,
    "Unused range of 64 bytes at the end",
  );
});

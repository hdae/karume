// The safetensors reader on files built here: every dtype, a zero-element tensor and metadata,
// opened from a Blob and from a ranged reader that records its reads; then one broken file per
// check. No GPU, no local file.

import { describe, it } from "vite-plus/test";
import { assertEquals, assertRejects } from "@std/assert";
import { openSafetensors, type RangedReader, SafetensorsError } from "../mod.ts";

type Spec = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly bytes: number;
};

const LENGTH_BYTES = 8;

/** `count` bytes counting up from `seed` (each tensor's bytes differ from its neighbours'). */
const filled = (count: number, seed: number): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length: count }, (_, i) => (seed + i) & 0xff);

/** A file of a header (padded with spaces to a multiple of 8 bytes, as the reference writer does) and a data section. */
const pack = (header: unknown, data: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const json = JSON.stringify(header);
  const padding = (LENGTH_BYTES - (json.length % LENGTH_BYTES)) % LENGTH_BYTES;
  return withHeaderBytes(new TextEncoder().encode(json + " ".repeat(padding)), data);
};

const withHeaderBytes = (
  header: Uint8Array<ArrayBuffer>,
  data: Uint8Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(LENGTH_BYTES + header.byteLength + data.byteLength));
  new DataView(out.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  out.set(header, LENGTH_BYTES);
  out.set(data, LENGTH_BYTES + header.byteLength);
  return out;
};

/** The tensors laid out back to back in the given order. */
const build = (
  specs: readonly Spec[],
  metadata?: Readonly<Record<string, string>>,
): Uint8Array<ArrayBuffer> => {
  const header: Record<string, unknown> = metadata === undefined ? {} : { __metadata__: metadata };
  let offset = 0;
  for (const spec of specs) {
    header[spec.name] = {
      dtype: spec.dtype,
      shape: spec.shape,
      data_offsets: [offset, offset + spec.bytes],
    };
    offset += spec.bytes;
  }
  const data = new Uint8Array(new ArrayBuffer(offset));
  offset = 0;
  for (const [index, spec] of specs.entries()) {
    data.set(filled(spec.bytes, 16 * index), offset);
    offset += spec.bytes;
  }
  return pack(header, data);
};

/** Every accepted dtype, largest element first (so every tensor is aligned), and a zero-element tensor. */
const SPECS: readonly Spec[] = [
  { name: "i64", dtype: "I64", shape: [2], bytes: 16 },
  { name: "f32", dtype: "F32", shape: [2, 2], bytes: 16 },
  { name: "empty", dtype: "F32", shape: [0, 4], bytes: 0 },
  { name: "i32", dtype: "I32", shape: [1], bytes: 4 },
  { name: "u32", dtype: "U32", shape: [3], bytes: 12 },
  { name: "f16", dtype: "F16", shape: [2], bytes: 4 },
  { name: "bf16", dtype: "BF16", shape: [3], bytes: 6 },
  { name: "i8", dtype: "I8", shape: [3], bytes: 3 },
  { name: "u8", dtype: "U8", shape: [2, 1], bytes: 2 },
  { name: "bool", dtype: "BOOL", shape: [1], bytes: 1 },
];
const METADATA = { format: "pt", note: "{}" };
const FILE = build(SPECS, METADATA);
const DATA_START = FILE.byteLength - 64;

/** A ranged reader over `bytes` that records every read as `[offset, length]`. */
const recording = (
  bytes: Uint8Array<ArrayBuffer>,
): { readonly reader: RangedReader; readonly reads: [number, number][] } => {
  const reads: [number, number][] = [];
  return {
    reads,
    reader: {
      size: bytes.byteLength,
      read: (offset, length) => {
        reads.push([offset, length]);
        return Promise.resolve(bytes.slice(offset, offset + length));
      },
    },
  };
};

/** A Blob over `bytes` whose slices are recorded as `[start, end]`. */
const recordingBlob = (
  bytes: Uint8Array<ArrayBuffer>,
): { readonly blob: Blob; readonly slices: [number, number][] } => {
  const blob = new Blob([bytes]);
  const slices: [number, number][] = [];
  const slice = blob.slice.bind(blob);
  blob.slice = (start?: number, end?: number): Blob => {
    slices.push([start ?? 0, end ?? blob.size]);
    return slice(start, end);
  };
  return { blob, slices };
};

describe("opening a safetensors file", () => {
  it("lists every tensor's dtype, shape and byte length in data order, and the metadata, from a Blob and from a ranged reader", async () => {
    for (const source of [new Blob([FILE]), recording(FILE).reader]) {
      const file = await openSafetensors(source);
      assertEquals(
        [...file.tensors].map(([name, tensor]) => [
          name,
          tensor.dtype,
          tensor.shape,
          tensor.byteLength,
        ]),
        SPECS.map((spec) => [spec.name, spec.dtype, spec.shape, spec.bytes]),
      );
      assertEquals(Object.fromEntries(file.metadata), METADATA);
    }
  });

  it("reads the 8-byte length and the header, and nothing else", async () => {
    const { reader, reads } = recording(FILE);
    await openSafetensors(reader);
    assertEquals(reads, [
      [0, LENGTH_BYTES],
      [LENGTH_BYTES, DATA_START - LENGTH_BYTES],
    ]);

    const { blob, slices } = recordingBlob(FILE);
    await openSafetensors(blob);
    assertEquals(slices, [
      [0, LENGTH_BYTES],
      [LENGTH_BYTES, DATA_START],
    ]);
  });

  it("reads a tensor with one ranged read of exactly its bytes, on every call", async () => {
    const { reader, reads } = recording(FILE);
    const file = await openSafetensors(reader);
    let offset = DATA_START;
    for (const [index, spec] of SPECS.entries()) {
      reads.length = 0;
      const tensor = file.tensors.get(spec.name);
      if (tensor === undefined) throw new Error(`no tensor ${spec.name}`);
      assertEquals(await tensor.read(), filled(spec.bytes, 16 * index));
      assertEquals(await tensor.read(), filled(spec.bytes, 16 * index));
      assertEquals(reads, [
        [offset, spec.bytes],
        [offset, spec.bytes],
      ]);
      offset += spec.bytes;
    }

    const blobFile = await openSafetensors(new Blob([FILE]));
    assertEquals(await blobFile.tensors.get("bf16")?.read(), filled(6, 16 * 6));
  });

  it("refuses a read that gives another length than the tensor's, naming the tensor", async () => {
    const file = await openSafetensors({
      size: FILE.byteLength,
      read: (offset, length) =>
        Promise.resolve(FILE.slice(offset, offset + (offset < DATA_START ? length : 1))),
    });
    const tensor = file.tensors.get("f32");
    if (tensor === undefined) throw new Error("no tensor f32");
    await assertRejects(
      () => tensor.read(),
      SafetensorsError,
      `tensor 'f32': the read of [${DATA_START + 16}, ${DATA_START + 32}) gave 1 bytes`,
    );
  });
});

const lengthOnly = (length: bigint, fileBytes: number): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(fileBytes));
  new DataView(out.buffer).setBigUint64(0, length, true);
  return out;
};

const F32_1 = filled(4, 0);
const one = (entry: unknown, data: Uint8Array<ArrayBuffer> = F32_1): Uint8Array<ArrayBuffer> =>
  pack({ a: entry }, data);

/** An F32 right after a 1-byte I8: its first byte is one past a multiple of 8. */
const MISALIGNED = pack(
  {
    a: { dtype: "I8", shape: [1], data_offsets: [0, 1] },
    b: { dtype: "F32", shape: [1], data_offsets: [1, 5] },
  },
  filled(5, 0),
);

/** One broken file per check, and the message it is refused with. */
const BROKEN: readonly {
  readonly check: string;
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly message: string;
}[] = [
  { check: "a file shorter than the length", bytes: filled(4, 0), message: "is too short" },
  {
    check: "a header length over the bound",
    bytes: lengthOnly(100_000_001n, 16),
    message: "Header length 100000001 exceeds the limit of 100000000 bytes",
  },
  {
    check: "a header length at the bound but past the end of the file",
    bytes: lengthOnly(100_000_000n, 16),
    message: "Header length 100000000 exceeds the file length 16",
  },
  {
    check: "a header that is not UTF-8",
    bytes: withHeaderBytes(new Uint8Array([0xff, 0xff]), new Uint8Array(0)),
    message: "Header is invalid UTF-8",
  },
  {
    check: "a header that is not JSON",
    bytes: withHeaderBytes(new TextEncoder().encode("{not json"), new Uint8Array(0)),
    message: "Cannot parse the header JSON",
  },
  {
    check: "a header that is not a JSON object",
    bytes: pack([], new Uint8Array(0)),
    message: "Header JSON is not an object",
  },
  {
    check: "metadata that is not an object",
    bytes: pack({ __metadata__: [] }, new Uint8Array(0)),
    message: "__metadata__ is not an object",
  },
  {
    check: "a metadata value that is not a string",
    bytes: pack({ __metadata__: { k: 1 } }, new Uint8Array(0)),
    message: "__metadata__.k is not a string",
  },
  {
    check: "an entry that is not an object",
    bytes: one(5),
    message: "tensor 'a': header entry is not an object",
  },
  {
    check: "a dtype outside the vocabulary",
    bytes: one({ dtype: "F64", shape: [1], data_offsets: [0, 8] }, filled(8, 0)),
    message: `tensor 'a': unsupported dtype "F64"`,
  },
  {
    check: "a packed dtype",
    bytes: one({ dtype: "I4", shape: [8], data_offsets: [0, 4] }),
    message: `tensor 'a': unsupported dtype "I4"`,
  },
  {
    check: "a shape that is not an array",
    bytes: one({ dtype: "F32", shape: "x", data_offsets: [0, 4] }),
    message: "tensor 'a': shape is not an array",
  },
  {
    check: "a negative dimension",
    bytes: one({ dtype: "F32", shape: [-1], data_offsets: [0, 4] }),
    message: "tensor 'a': shape element -1 is not a non-negative integer",
  },
  {
    check: "data_offsets that are not an array",
    bytes: one({ dtype: "F32", shape: [1], data_offsets: 0 }),
    message: "tensor 'a': data_offsets is not an array",
  },
  {
    check: "data_offsets of one element",
    bytes: one({ dtype: "F32", shape: [1], data_offsets: [0] }),
    message: "tensor 'a': data_offsets has 1 elements (2 required)",
  },
  {
    check: "a fractional offset",
    bytes: one({ dtype: "F32", shape: [1], data_offsets: [0.5, 4.5] }),
    message: "tensor 'a': data_offsets 0.5 is not a non-negative integer",
  },
  {
    check: "reversed data_offsets",
    bytes: one({ dtype: "F32", shape: [1], data_offsets: [4, 0] }),
    message: "tensor 'a': data_offsets are reversed [4, 0)",
  },
  {
    check: "an element count beyond the safe integers",
    bytes: one({ dtype: "F32", shape: [2 ** 30, 2 ** 30], data_offsets: [0, 4] }),
    message: "tensor 'a': element count exceeds the safe integer range",
  },
  {
    check: "a byte range other than numel × element size",
    bytes: one({ dtype: "F32", shape: [2], data_offsets: [0, 4] }),
    message: "tensor 'a': size mismatch offsets=4 expected=8 (F32 [2])",
  },
  {
    check: "a range past the data section",
    bytes: one({ dtype: "F32", shape: [2], data_offsets: [0, 8] }),
    message: "tensor 'a': outside the data section [0, 8) data section length=4",
  },
  {
    check: "two overlapping ranges",
    bytes: pack(
      {
        a: { dtype: "F32", shape: [2], data_offsets: [0, 8] },
        b: { dtype: "F32", shape: [2], data_offsets: [4, 12] },
      },
      filled(12, 0),
    ),
    message: "tensor 'b': range [4, 12) overlaps tensor 'a' [0, 8)",
  },
  {
    check: "a gap between two tensors",
    bytes: pack(
      {
        a: { dtype: "F32", shape: [1], data_offsets: [0, 4] },
        b: { dtype: "F32", shape: [1], data_offsets: [8, 12] },
      },
      filled(12, 0),
    ),
    message: "tensor 'b': unused range [4, 8) in the data section before it (after tensor 'a')",
  },
  {
    check: "a gap before the first tensor",
    bytes: one({ dtype: "F32", shape: [1], data_offsets: [4, 8] }, filled(8, 0)),
    message: "tensor 'a': unused range [0, 4) in the data section before it",
  },
  {
    check: "unused bytes after the last tensor",
    bytes: one({ dtype: "F32", shape: [1], data_offsets: [0, 4] }, filled(8, 0)),
    message: "Unused range of 4 bytes at the end of the data section after tensor 'a'",
  },
  {
    check: "a tensor whose first byte is not a multiple of its element size",
    bytes: MISALIGNED,
    message: `tensor 'b': absolute offset ${
      LENGTH_BYTES + Number(new DataView(MISALIGNED.buffer).getBigUint64(0, true)) + 1
    } is not aligned to the alignment unit of F32, 4 bytes`,
  },
];

describe("the checks of the tensor table", () => {
  for (const { check, bytes, message } of BROKEN) {
    it(`refuses ${check}`, async () => {
      await assertRejects(
        () => openSafetensors(recording(bytes).reader),
        SafetensorsError,
        message,
      );
    });
  }

  it("refuses a header length over the bound without reading the header", async () => {
    const { reader, reads } = recording(lengthOnly(2n ** 63n, 16));
    await assertRejects(() => openSafetensors(reader), SafetensorsError, "exceeds the limit");
    assertEquals(reads, [[0, LENGTH_BYTES]]);
  });

  it("refuses a ranged reader whose size is not a non-negative integer", async () => {
    await assertRejects(
      () => openSafetensors({ size: 1.5, read: () => Promise.resolve(new Uint8Array(0)) }),
      SafetensorsError,
      "File length 1.5 is not a non-negative integer",
    );
  });
});

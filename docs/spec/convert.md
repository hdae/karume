# Conversion: weights converted while they are read

What `@karume/convert` (`packages/convert`, private) does now. The direction is item 7 of the ADR
[0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md) list in the [backlog](../backlog.md);
the container format is [container-v1](../container-v1.md).

## The inputs

- **The target**: a `BlockSource` over the container of a distributed preset (`krm`). Its part 0 is the plan: per
  initializer, the codec, `rowAxis`, `groupSize`, the scale block and the piece row ranges (the model descriptor's
  binding), every block's part, offset, length and sha256, every part's length and sha256. It is read with the
  runtime's own reader (`openContainer`); the plan is not derived from the graph. Its model descriptor must be in
  canonical form (serializing it again gives its bytes), or the conversion is refused.
- **The tensor list**: tensor name → element type (`F32`, `F16` or `BF16`), shape, and a function that reads that
  tensor's bytes (little endian, row-major, `numel × element size`). Nothing is read when the list is built or when the
  converting source is built; a tensor is read when a block of it is read.
- **The name map**: initializer name → tensor name. It is the identity when omitted.

## What is served

`openConvertingSource` gives a `BlockSource` with the target's part count and part lengths, and `verified: true`.

| Part | Served bytes |
| --- | --- |
| Part 0 | The target's, except the sha256 of every converted block and of every part that holds one, which is 64 `0` characters. The document keeps its length; the header and the graph descriptor are the target's. |
| The const part (part 1) and the asset parts | The target's bytes, with their true hashes. When the target is unverified, every block a read touches is read whole and checked against its declared sha256 before the range is returned. |
| A part that holds converted blocks | Built from the encoded payloads: each block's payload at its offset, its tail padding and the gaps between blocks 0x00, for any offset and length inside the part. The target is never read for it. |

- A converted block is every data block and every scale block of an initializer the model descriptor binds; consts
  and assets are not converted. A part that holds both converted blocks and asset blocks is refused.
- Part 0 never states a hash its bytes do not have: the 64 `0` characters are not the digest of any bytes. A reader of
  this source does not compare block hashes because it is verified; the real hashes are written by the writer.
- The encoded payloads of one initializer (its data and its scale) are kept until a block of another initializer is
  asked for; nothing older is kept. Reading an earlier initializer's block again reads and encodes its tensor again.

## The encodings

Every source dtype is converted to f32 first; that is exact (F16 and BF16 go through the runtime's decoders, which do
not keep a NaN's payload). The target codec then decides:

| Codec | Stored | Refused |
| --- | --- | --- |
| `int8-sym` | Per index of `shape[rowAxis]` (row axis 0 or 1), over every other element: `scale = max(fl32(amax / 127), 2^-126)`, `q = clamp(round_half_even(fl32(w / scale)), −127, 127)`; the codes in the weight's memory order, one f32 scale per channel. `fl32(a / b)` is the f32 quotient. The same for a whole tensor and for one split into pieces (a piece is a row range of the payload). | A non-finite value. |
| `bf16` | The value itself: the upper half of its f32 bits. | A value that is not a bf16 value (the lower half of its f32 bits is not zero) and a NaN. A float is never rounded. |
| `f32` | The value itself. | Nothing. |

These are the exporter's storage conversions bit for bit (`channel_scale`, `quantize_to_int8` in
`tools/exporter/src/karume/quantize.py`; `_convert_for_storage` in `emit.py`, which refuses what the table refuses).
Any other target codec is refused.

## The checks when it is built

Before a byte of any tensor is read, and all reported together in one `ConvertError` (one line per problem, naming
the graph and the initializer):

- an initializer of the plan whose tensor the list does not have;
- a tensor whose shape is not the declared shape (the message gives both element counts);
- a target codec other than `int8-sym`, `bf16` and `f32`;
- a part that holds both converted blocks and asset blocks.

Tensors of the list that no initializer uses are not an error; the result lists them (`unusedTensors`).

When a tensor is read: a read that gives another byte length than its dtype and shape, and a value the codec refuses,
are a `ConvertError` naming the initializer, the tensor and the element.

## The writer

`writeContainer(source, sink)` writes the container a source serves:

- Each part from 1 on is read whole, the sha256 of the part and of every block in it is computed, and the part goes to
  `sink.write(part, 0, bytes)`, in index order (a part of length 0 as 0 bytes).
- Part 0 goes last: the model descriptor serialized again with the real hashes (the same length, so the header and the
  graph descriptor do not change).
- It returns every part's length and sha256 (part 0 included) and the length and sha256 of the two documents, in the
  shape of `DescriptorExpectation`: what a manifest entry (`container` of `karume/5`) needs.
- A part's sha256 is one Web Crypto digest over the whole part, so a whole part is held while it is hashed and written.

Guarantee: for the tensors of a distributed preset, the parts written against the plan of a preset made from the same
values by the exporter are byte-identical to that preset's files (checked on the Anima transformer, bf16 tensors
against the `i8` plan, and on a text conditioner's `bf16` copy — `packages/e2e/tests/e2e_anima_convert_test.ts`). The
written container opens with `openContainer` as an unverified source, every block hash checked.

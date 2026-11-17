# The safetensors reader: a file read one tensor at a time

What `@karume/safetensors` (`packages/safetensors`, private) does now. It is the format reader of item 7 of the ADR
[0152](../decisions/0152-maintained-specs-and-declared-numeric-requirements.md) list in the [backlog](../backlog.md):
it reads a user's weights file for [the conversion](convert.md). The format is upstream's
(<https://github.com/huggingface/safetensors>): an unsigned 64-bit little-endian header length N, N bytes of UTF-8 JSON,
then the data section, with `data_offsets` relative to its start.

The runtime's whole-buffer reader (`parseSafetensors`, `packages/runtime/src/format/safetensors.ts`) applies the same
checks to a file held in memory; the two are separate code until they are merged.

## Opening a file

- The input is a `Blob` (a browser `File` is one) or a ranged reader `{ size, read(offset, length) }`. A value with a
  `read` member is taken as a ranged reader; a `Blob` is read with `slice(…).arrayBuffer()`.
- Opening makes two reads: bytes `[0, 8)`, the header length N, then `[8, 8 + N)`, the header. No byte of the data
  section is read, and every check below runs.
- The result: `metadata` (`__metadata__`; empty when the header has none) and `tensors`, tensor name → `dtype`, `shape`,
  `byteLength` and `read`, in the order of the data section.
- `read()` makes one ranged read of exactly the tensor's bytes, `[8 + N + begin, 8 + N + end)`, on every call, and
  keeps nothing. A read that gives another length is refused, naming the tensor.
- Values are not interpreted. A tensor of `F32`, `F16` or `BF16` is usable as an entry of the conversion's tensor list
  as it is.
- A key the header JSON holds twice is read as its last occurrence (`JSON.parse`).

## The checks

All run when the file is opened. A refusal is a `SafetensorsError`; one about a tensor names it, an overlap or a gap
also names the tensor before it, and unused bytes at the end name the last tensor.

| What | Refused |
| --- | --- |
| The file length | A ranged reader's `size` that is not a non-negative integer; a file shorter than 8 bytes |
| The header length | N above 100,000,000 bytes (the reference implementation's bound; refused before the header is read); `8 + N` past the end of the file |
| The header | Not UTF-8 (strict decoding); not JSON; not a JSON object; `__metadata__` that is not an object, or a value of it that is not a string |
| A tensor's entry | Not an object; a `dtype` other than `F32` `F16` `BF16` `I8` `U8` `I32` `U32` `I64` `BOOL`; a `shape` that is not an array of non-negative safe integers; `data_offsets` that are not two non-negative safe integers, or are reversed; an element count beyond the safe integers; `end − begin` other than `numel × element size` |
| The data section (entries in order of `begin`, then `end`) | A range that overlaps the one before it; unused bytes before a range; a range past the end of the data section; a range whose first byte in the file (`8 + N + begin`) is not a multiple of its element size; unused bytes after the last range |

Element sizes: `I64` 8; `F32`, `I32`, `U32` 4; `F16`, `BF16` 2; `I8`, `U8`, `BOOL` 1. A zero-element tensor has an
empty range, which is checked like any other (its place and its alignment).

Where it differs from the runtime's reader: the bound on N (the runtime's reader bounds N by the file length only), and
the messages that name the neighbouring tensor. The accepted files are otherwise the same.

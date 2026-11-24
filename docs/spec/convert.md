# Conversion: weights converted while they are read

What `@karume/convert` (`packages/convert`, published on JSR) does now. The direction is item 7 of the ADR
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
- **The tensor names** (`tensorName`): initializer name → tensor name. It is the identity when omitted. A
  [name map](#the-name-map) gives this mapping per component of a file.
- **The 16-bit float codec** (`float16Codec`, `f16` or `bf16`; optional): the plan is the target's with this codec
  in place of every `f16` and `bf16` initializer's ([below](#a-plan-with-another-16-bit-float-codec)). Omitted:
  the target's plan as it is.

## What is served

`openConvertingSource` gives a `BlockSource` with the target's part count, the target's part lengths from part 1 on,
the length of the plan's part 0 (the target's, or [the one with another 16-bit float
codec](#a-plan-with-another-16-bit-float-codec)), and `verified: true`.

| Part | Served bytes |
| --- | --- |
| Part 0 | The plan's — the target's, or [the one with another 16-bit float codec](#a-plan-with-another-16-bit-float-codec) —, except the sha256 of every converted block and of every part that holds one, which is 64 `0` characters. The document keeps the plan's length; the header and the graph descriptor are the plan's. |
| The const part (part 1) and the asset parts | The target's bytes, with their true hashes. When the target is unverified, every block a read touches is read whole and checked against its declared sha256 before the range is returned. |
| A part that holds converted blocks | Built from the encoded payloads: each block's payload at its offset, its tail padding and the gaps between blocks 0x00, for any offset and length inside the part. The target is never read for it. |

- A converted block is every data block and every scale block of an initializer the model descriptor binds; consts
  and assets are not converted. A part that holds both converted blocks and asset blocks is refused.
- Part 0 never states a hash its bytes do not have: the 64 `0` characters are not the digest of any bytes. A reader of
  this source does not compare block hashes because it is verified; the real hashes are written by the writer.
- The source names the parts it builds (`convertedParts`, in index order): every part that holds a converted block.
  The target is never read for them, so a caller that fetches the target's parts ahead of the reads leaves them out.
- By default the encoded payloads of one initializer (its data and its scale) are kept until a block of another
  initializer is asked for; nothing older is kept. Reading an earlier initializer's block again reads and encodes its
  tensor again.
- With `retainEncoded: true` the encoded payloads of every initializer are kept once they are encoded: a block of an
  initializer encoded before is served from memory, and its tensor is not read or encoded again. What is held grows to
  the converted size of what was read (the payloads, without tail padding and gaps), and is held for as long as the
  source is reachable; the source has no call that drops it — it is released when the source is, by garbage
  collection. An encoding that fails is not kept, with or without the option: a later read tries again.

## The encodings

Every source dtype is converted to f32 first; that is exact (F16 and BF16 go through the runtime's decoders, which do
not keep a NaN's payload). The target codec then decides:

| Codec | Stored | Refused |
| --- | --- | --- |
| `int8-sym` | Per index of `shape[rowAxis]` (row axis 0 or 1), over every other element: `scale = max(fl32(amax / 127), 2^-126)`, `q = clamp(round_half_even(fl32(w / scale)), −127, 127)`; the codes in the weight's memory order, one f32 scale per channel. `fl32(a / b)` is the f32 quotient. The same for a whole tensor and for one split into pieces (a piece is a row range of the payload). | A non-finite value. |
| `f16` | The value itself, as IEEE binary16 (±0, the subnormals and ±inf included). | A value that is not an f16 value — one `Math.f16round` moves: a finer mantissa, a magnitude past 65504, or one between the points of the f16 grid near zero (below 2⁻²⁴, or not a multiple of it below 2⁻¹⁴) — and a NaN. A float is never rounded. |
| `bf16` | The value itself: the upper half of its f32 bits. | A value that is not a bf16 value (the lower half of its f32 bits is not zero) and a NaN. A float is never rounded. |
| `f32` | The value itself. | Nothing. |

These are the exporter's storage conversions bit for bit (`channel_scale`, `quantize_to_int8` in
`tools/exporter/src/karume/quantize.py`; `_convert_for_storage` in `emit.py`, whose `_is_f16_exact` and `_is_bf16_exact`
— a `torch.equal` of the round trip through the type — refuse what the table refuses and let ±inf through).
Any other target codec is refused.

What `f16` guarantees, and what it does not:

- An F16 tensor without a NaN is stored bit for bit (its f32 values are exact, and each is an f16 value). With an F16
  source, the parts written against an f16 plan — a distributed one, or a bf16 plan with `float16Codec: "f16"` —
  equal an f16 export of the same values and layout: the exporter stores the values the recipe rounded to f16, which
  for F16 values is a no-op, and an initializer the plan stores as `f32` beside them takes the value itself, as the
  exporter's does.
- A BF16 or F32 source is stored only when every value of the tensor is an f16 value; otherwise it is refused, not
  rounded. The exporter's f16 presets of a BF16 upstream round the model to f16 first (the recipe's rounding, before
  the export), and this conversion does not reproduce that: it never changes a value.

## The plan before a conversion

`readPlanCodecs(target)` reads the target's part 0 alone — the same reading and the same checks of part 0 as the
converting source — and gives every initializer the plan stores in its weight parts with its codec (`graph`,
`initializer`, `codec`), in the order of the graphs and of their binding tables. It is what a caller decides on before
it gives a tensor list: whether every codec is one encoded here (`ENCODED_CODECS`: `int8-sym`, `f16`, `bf16`, `f32`), and
whether a tensor's dtype can be stored as its initializer's codec. A codec is listed whether it is encoded here or not.
With `float16Codec` it reads the plan the converting source given the same option encodes against.

## A plan with another 16-bit float codec

With `float16Codec`, every initializer the target stores as `f16` or `bf16` is stored as the codec given; every
other initializer (`int8-sym`, `f32`) is stored as the target stores it. The two codecs store an element in two bytes,
with no scale and the same packing in the codec ledger, so the plan keeps every block of the target — its part, offset
and length —, every part past part 0 at its length, the const part, the assets and the graph descriptor.

- What changes is the model descriptor: the codec of those bindings and the codec set. With it change its length in
  the header and part 0's length (`bf16` is one character longer than `f16` at each binding it names).
- That part 0 is made here: the target's header with the new model descriptor length, the target's graph descriptor,
  and the model descriptor serialized canonically. The runtime's reader opens it as it opens the target's, so a part 0
  made wrongly is refused there; the target's own model descriptor must still be canonical.
- A plan whose 16-bit initializers already have the codec is the target's, part 0 included.
- Guarantee: with an F16 source and `f16` over a plan whose 16-bit weights are bf16 (or the other way round with a BF16
  source and `bf16`), the written container is, byte for byte, the one the exporter writes from those values with the
  16-bit weights stored as the source's type, provided that export differs from the target in the 16-bit codec alone
  (the same graph descriptor and the same blocks). Checked on a synthetic container in both directions
  (`packages/convert/tests/converting_source_test.ts`).

## The checks when it is built

Before a byte of any tensor is read, and all reported together in one `ConvertError` (one line per problem, naming
the graph and the initializer):

- an initializer of the plan whose tensor the list does not have;
- a tensor whose shape is not the declared shape (the message gives both element counts);
- a target codec other than `int8-sym`, `f16`, `bf16` and `f32`;
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
  shape of `DescriptorExpectation`: what a manifest entry (`container` of `karume/6`) needs.
- A part's sha256 is one Web Crypto digest over the whole part, so a whole part is held while it is hashed and written.

Guarantee: for the tensors of a distributed preset, the parts written against the plan of a preset made from the same
values by the exporter are byte-identical to that preset's files (checked on the Anima transformer, bf16 tensors
against the `i8` plan, and on a text conditioner's `bf16` copy — `packages/e2e/tests/e2e_anima_convert_test.ts`). The
written container opens with `openContainer` as an unverified source, every block hash checked.

## The name map

Data that says which tensor of a weights file is which initializer of which component: one JSON document per model,
written beside its export recipe, where both names are known (Anima: `tools/export-recipes/anima/name-map.json`). It
holds no function and no tensor operation (select, slice, reshape): a tensor goes to its initializer as it is.

A distribution carries its family's map: the recipe declares the file (`Pipeline.name_map` of the exporter), the
assembly checks that it is JSON with the format tag below, copies it to `name-map.json` at the repository root and
references it from the manifest as `nameMap` (`karume/6` — ADR [0153](../decisions/0153-manifest-v6-name-map.md)), a
reference the hub checks like every file reference. Anima's repositories (`karume-anima`, `karume-anima-extra` and the
unpublished `karume-anima-2.9b`) carry the map above. `fromSingleFile` of `@karume/models` reads it from the distribution ([single-file](single-file.md)):
fetched through the hub and checked with `parseNameMap` (below); `placeTensors` takes a map as a value.

| Field | What it is |
| --- | --- |
| `format` | `karume-name-map/1` |
| `layouts[]` | Every layout the model's files exist in upstream (a single file, a file per component) |
| `layouts[].name`, `.source` | The layout's name, and where its prefixes and markers come from (the recipe's lines, the upstream function) |
| `layouts[].stripPrefix` | Prefixes one of which every key of the file starts with, the same one for every key; it is removed from every key first. Empty: the keys are taken as they are |
| `layouts[].markers` | Keys, after that prefix is removed, that a file of the layout holds |
| `layouts[].components[]` | `role` (the component's role in the distribution), `source` (where its rule list comes from), `route`, `rules` and `initializerPrefix` |

A key goes to the component whose `route` is the longest prefix it starts with (`""` takes every key). The
component's `rules` apply in order, each to the key as the rules before it left it; the order is part of the data
(Anima's `adaln_modulation_self_attn.1` must be rewritten before `self_attn`, or it becomes
`adaln_modulation_attn1.1`).

| Rule | Effect |
| --- | --- |
| `{ "stripPrefix": p }` | Removes `p` when the key starts with it |
| `{ "replace": a, "with": b }` | Replaces every occurrence of `a`, left to right, with `b` taken literally (Python's `str.replace`) |
| `{ "exclude": s }` | Takes the key out when it contains `s`: the graph does not use that tensor |

The initializer name is `initializerPrefix` followed by what the rules leave.

### Reading a name map

`parseNameMap(value)` reads a map that arrives as unchecked JSON (a distribution's file) and gives a `NameMap` built from
the checked values (it shares no object with `value`). The checks are written by hand (no dependency):

- An object whose `format` is present and is not `karume-name-map/1` is refused for that alone: its other fields are not
  read against this format.
- Otherwise every problem is reported together in one `ConvertError`, one line each, naming where it is (`the name
  map.layouts[0].components[1].rules[2]`): a value that is not an object, a string or an array where the table above
  says so; a field missing; a key the format does not name, at every level; a `role` or a layout `name` that is empty;
  a rule that is none of the three kinds, or holds a key of another kind.
- A `replace` rule whose `replace` is empty, and an `exclude` rule whose `exclude` is empty, are refused: Python's
  `str.replace` puts the replacement between every two characters, which this interpreter does not, and an empty
  exclusion would take out every key.

### Placing a file's tensors

`placeTensors(map, keys)` reads a map against a file's keys and gives:

- `layout`: the one layout that recognises the file — every key carries the same one of its `stripPrefix` prefixes
  (when it lists some), and the file holds every one of its markers;
- `components`: per role of that layout (every role, possibly empty), initializer name → file key, in file order. The
  component's tensor list is the file's tensors under these initializer names: `openConvertingSource` then needs no
  `tensorName`, and an initializer no key is placed at is reported as missing, never read from a key of its name;
- `excluded`: the keys an exclusion rule took out, and `unplaced`: the keys no route takes (both in file order; neither
  is an error — whether the plan's initializers are all there is the converting source's check).

Refused with one `ConvertError`, every problem listed together:

- a map whose `format` is not `karume-name-map/1`;
- a file no layout recognises (each layout's reason: the prefix, or the marker keys the file does not hold), or one
  that more than one layout recognises;
- two keys that map to one initializer of a component (both keys and the initializer are named);
- a key whose longest route is the route of two components.

Anima's map has four layouts. The single file: `model.diffusion_model.` or `net.` is stripped, `llm_adapter.*` goes to
`text_conditioner` with that prefix removed, and every other key goes to `transformer` through diffusers' 23 Cosmos 2.0
renames and 8 exclusions, as `anima/single_file.py` runs them. The single file without a prefix: the same markers and
the same components, with no prefix stripped (a 40-block layer-expanded model is saved this way; the rules name no block index,
so they hold for any block count). The diffusers repository's `transformer` and `text_conditioner` files: their keys
as they are. Every initializer name carries `model.`, the attribute of the recipe's wrapper module.

Guarantee: on the upstream single files `anima-base-v1.0` and `anima-turbo-v1.1` (685 keys each), the single-file
layout places 567 keys in `transformer` and 118 in `text_conditioner`, excludes none and leaves none, and their tensors,
read with [the safetensors reader](safetensors.md) and written against the `i8` and `bf16` transformer plans and the
`bf16` text conditioner plan of the distributed models, are byte-identical to those presets' files
(`packages/e2e/tests/e2e_anima_single_file_test.ts`).

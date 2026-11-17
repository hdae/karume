# @karume/convert

Private workspace package (not published yet): weights converted **while a Session reads them**. A weights file read as
a tensor list is encoded against the plan of a distributed preset — that preset's part 0, which says how every
initializer is stored and where — and served as that preset's container; writing the result as a `krm` is optional.
The behaviour is specified in [docs/spec/convert.md](../../docs/spec/convert.md).

- **Depends on `@karume/runtime` only** (`openContainer` and the types of the main entry, and the format parts of
  `@karume/runtime/container`: the model descriptor's canonical serializer, the sha256, the f16 / bf16 decoders and
  the descriptor and supply types). No hub, no models, no `node:` import, no side effects at import.
- No file format reader is here: the input is a format-neutral tensor list (the tests read a distributed container as
  one).

| Export | What it is |
| --- | --- |
| `openConvertingSource({ target, tensors, tensorName? })` | The converting `BlockSource` (`verified: true`) over the target's container, and the tensors the plan does not use. Encodes `int8-sym`, `bf16` and `f32`, bit for bit as the exporter does; every check that needs no tensor byte runs when it is built and is reported together |
| `writeContainer(source, sink)` | Writes a source's container part by part to `sink.write(part, offset, bytes)` (part 0 last) with the real hashes, and returns each part's length and sha256 and the two descriptors' |
| `SourceTensor`, `TensorDtype`, `TensorList` | The tensor list |
| `ConvertingSourceInput`, `ConvertingSource`, `ContainerSink`, `Digest`, `WrittenContainer` | The inputs and results above |
| `INVALID_SHA256` | The 64 `0` characters the served part 0 states for converted blocks and their parts |
| `ConvertError` | Every refusal: a tensor missing or of another shape, a codec not encoded here, a value the codec cannot store as given |

## Tests

- `tests/encoders_test.ts` — the encoders against `tests/fixtures/encoders.json`, which the exporter's Python writes
  from seeded tensors (`tests/fixtures/generate_encoders.py`). Re-run it with
  `cd tools/exporter && uv run python ../../packages/convert/tests/fixtures/generate_encoders.py`.
- `tests/converting_source_test.ts` — the converting source and the writer on a synthetic container written by the
  runtime's test writer (`packages/runtime/tests/helpers/container-write.ts`).
- `packages/e2e/tests/e2e_anima_convert_test.ts` (lane `models:anima`, needs `models/karume-anima/`) — the Anima
  transformer's `bf16` tensors written against the `i8` plan give the `i8` files, byte for byte.

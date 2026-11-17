# @karume/convert

Private workspace package (not published yet): weights converted **while a Session reads them**. A weights file read as
a tensor list is encoded against the plan of a distributed preset — that preset's part 0, which says how every
initializer is stored and where — and served as that preset's container; writing the result as a `krm` is optional.
The behaviour is specified in [docs/spec/convert.md](../../docs/spec/convert.md).

- **Depends on `@karume/runtime` only** (`openContainer` and the types of the main entry, and the format parts of
  `@karume/runtime/container`: the model descriptor's canonical serializer, the sha256, the f16 / bf16 decoders and
  the descriptor and supply types). No hub, no models, no `node:` import, no side effects at import.
- No file format reader is here: the input is a format-neutral tensor list (the tests read a distributed container as
  one; `@karume/safetensors` reads a user's file as one). A name map — JSON beside a model's export recipe — says which
  key of a file is which initializer of which component.

| Export | What it is |
| --- | --- |
| `openConvertingSource({ target, tensors, tensorName? })` | The converting `BlockSource` (`verified: true`) over the target's container, the tensors the plan does not use, and the parts it builds (`convertedParts`, never read from the target). Encodes `int8-sym`, `bf16` and `f32`, bit for bit as the exporter does; every check that needs no tensor byte runs when it is built and is reported together |
| `writeContainer(source, sink)` | Writes a source's container part by part to `sink.write(part, offset, bytes)` (part 0 last) with the real hashes, and returns each part's length and sha256 and the two descriptors' |
| `readPlanCodecs(target)` | The codec of every initializer a target's plan stores, read from its part 0 alone — what a caller checks before it gives a tensor list (`ENCODED_CODECS`: the codecs encoded here) |
| `placeTensors(map, keys)` | Reads a name map against a file's keys: the layout that recognises the file, per component the initializer name → key mapping, and the keys excluded or not placed; refuses an unrecognised file, two keys on one initializer and a key two components route |
| `parseNameMap(value)` | Reads a name map that arrives as unchecked JSON (a distribution's `name-map.json`): every field checked by hand, every problem reported together, a key the format does not name refused |
| `NameMap`, `NameMapLayout`, `NameMapComponent`, `NameMapRule`, `TensorPlacement`, `NAME_MAP_FORMAT` | The name map's format (`karume-name-map/1`) and the placement |
| `SourceTensor`, `TensorDtype`, `TensorList` | The tensor list |
| `PlannedCodec`, `ENCODED_CODECS` | One initializer of a plan with its codec; the codecs encoded here |
| `ConvertingSourceInput`, `ConvertingSource`, `ContainerSink`, `Digest`, `WrittenContainer` | The inputs and results above |
| `INVALID_SHA256` | The 64 `0` characters the served part 0 states for converted blocks and their parts |
| `ConvertError` | Every refusal: a tensor missing or of another shape, a codec not encoded here, a value the codec cannot store as given, a file a name map cannot place |

## Tests

- `tests/encoders_test.ts` — the encoders against `tests/fixtures/encoders.json`, which the exporter's Python writes
  from seeded tensors (`tests/fixtures/generate_encoders.py`). Re-run it with
  `cd tools/exporter && uv run python ../../packages/convert/tests/fixtures/generate_encoders.py`.
- `tests/converting_source_test.ts` — the converting source and the writer on a synthetic container written by the
  runtime's test writer (`packages/runtime/tests/helpers/container-write.ts`).
- `tests/name_map_test.ts` — the name map's interpreter on small synthetic maps: each rule kind, the rule order, the
  routing and each refusal; `parseNameMap` on each kind of problem and on the Anima map of the recipe.
- `packages/e2e/tests/e2e_anima_convert_test.ts` (lane `models:anima`, needs `models/karume-anima/`) — the Anima
  transformer's `bf16` tensors written against the `i8` plan give the `i8` files, byte for byte.
- `packages/e2e/tests/e2e_anima_single_file_test.ts` (lane `models:anima`, also needs the upstream files under
  `inputs/anima/upstream-single-file/`) — the upstream single files read with `@karume/safetensors`, placed with
  `tools/export-recipes/anima/name-map.json` and converted give the distributed files, byte for byte.

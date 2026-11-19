# @karume/safetensors

The safetensors reader of [Karume](https://github.com/hdae/karume), a general-purpose neural-network inference stack
that runs on WebGPU. It opens a file **from its header alone** and reads **one tensor at a time**, so a file too large
to hold in memory (a 4 GB upstream checkpoint in a browser, where one ArrayBuffer stops near 2 GiB) is never read whole.
The behaviour is specified in [docs/spec/safetensors.md](https://github.com/hdae/karume/blob/main/docs/spec/safetensors.md).
The other packages of the stack are `@karume/runtime` (IR execution), `@karume/hub` (manifest resolution, download,
cache), `@karume/convert` (weights converted while a Session reads them) and `@karume/models` (pipelines and
tokenizers).

Status: pre-1.0, versioned in lockstep with the other `@karume/*` packages.

- **No dependency** (not even `@karume/runtime`), Web standard APIs only, no side effects at import.
- The input is a `Blob` (a browser `File` is one; Node's `fs.openAsBlob` gives one too) or any `{ size, read(offset,
  length) }`.
- It does not interpret values: a tensor is its dtype, shape, byte length and a function that reads its bytes. A float
  tensor (`F32` / `F16` / `BF16`) is usable as an entry of `@karume/convert`'s tensor list as it is.
- The checks are the runtime's whole-buffer reader's (`packages/runtime/src/format/safetensors.ts`), stated here again
  until the two readers are merged, plus the reference implementation's 100 MB bound on the header.

| Export | What it is |
| --- | --- |
| `openSafetensors(source)` | Reads the 8-byte length and the header (two reads), checks the tensor table, and returns `{ metadata, tensors }`: `__metadata__`, and per tensor name (in data order) its `dtype`, `shape`, `byteLength` and `read()`, one ranged read of exactly its bytes |
| `SafetensorsSource`, `RangedReader` | The input: a `Blob` or `{ size, read(offset, length) }` |
| `SafetensorsFile`, `SafetensorsTensor`, `SafetensorsDtype` | The result, and the accepted dtypes (`F32 F16 BF16 I8 U8 I32 U32 I64 BOOL`) |
| `SafetensorsError` | Every refusal; one about a tensor names it |

## Install

```sh
deno add jsr:@karume/safetensors
```

```sh
npx jsr add @karume/safetensors
pnpm dlx jsr add @karume/safetensors
bunx jsr add @karume/safetensors
```

## Minimal usage

Open a file from its header, list its tensors, and read one tensor's bytes.

```ts
import { openSafetensors } from "@karume/safetensors";

// file: a browser `File`, any `Blob`, or `{ size, read(offset, length) }`.
const { metadata, tensors } = await openSafetensors(file);
for (const [name, tensor] of tensors) console.log(name, tensor.dtype, tensor.shape, tensor.byteLength);

const [first] = tensors.values();
const bytes = await first?.read(); // one ranged read of exactly that tensor's bytes
```

## Tests

- `tests/open_safetensors_test.ts` — a file built in the test with every dtype, a zero-element tensor and metadata,
  opened from a `Blob` and from a ranged reader that records its reads (opening reads the length and the header only;
  a tensor is one read of exactly its bytes), and one broken file per check.
- `tests/public_surface_test.ts` — the names the entry exports, types included, against
  `tests/fixtures/public-surface.json` (`KARUME_SURFACE=write` rewrites it).
- `packages/e2e/tests/e2e_anima_single_file_test.ts` (lane `models:anima`, needs the upstream files under
  `inputs/anima/upstream-single-file/` and `models/karume-anima/`) — the 4.2 GB Anima single files read through it and
  converted.

## Documentation

- [Repository](https://github.com/hdae/karume) — overview and examples
- [docs/spec/safetensors.md](https://github.com/hdae/karume/blob/main/docs/spec/safetensors.md) — what this package
  does

## License

MIT — see [LICENSE](./LICENSE).

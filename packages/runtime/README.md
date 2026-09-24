# @karume/runtime

The execution engine of [Karume](https://github.com/hdae/karume), a general-purpose neural-network
inference stack that runs on WebGPU. This package opens a Karume container (`krm` — the IR graphs,
the binding table and the weight blocks), generates WGSL for it, uploads the weights block by block,
and executes the graph on a GPU device. It is written in pure TypeScript and WGSL, has zero external
dependencies, and takes the same code path in Deno and in the browser. The other packages of the
stack are `@karume/hub` (manifest resolution, download, cache) and `@karume/models` (pipelines and
tokenizers).

Status: pre-1.0. The public surface is a deliberately thin, explicitly designed API — internal
modules are never re-exported. Unsupported operators, contract violations and missing device
limits fail loudly with typed errors rather than silently falling back to an approximation.

## Install

```sh
deno add jsr:@karume/runtime
```

```sh
npx jsr add @karume/runtime
pnpm dlx jsr add @karume/runtime
bunx jsr add @karume/runtime
```

## Minimal usage

Acquire a GPU context, open a container, build a session for one of its graphs, run it, then release
the two lifetimes in order (session first, device last).

```ts
import { acquireGpu, createSessionFromContainer, openContainer } from "@karume/runtime";

const gpu = await acquireGpu();
// A split container is a numbered part sequence; part 0 holds the header and the two descriptors.
const parts = await Promise.all(
  ["model-00001-of-00003.krm", "model-00002-of-00003.krm", "model-00003-of-00003.krm"].map(
    (path) => Deno.readFile(path),
  ),
);
const opened = await openContainer({ kind: "parts", parts });
// The graph name is the component name the exporter wrote (the `weights` key in `karume.json`).
const session = await createSessionFromContainer(gpu, opened, "model");

const outputs = await session.run({
  pixel_values: {
    dtype: "f32",
    shape: [1, 3, 224, 224],
    data: new Float32Array(1 * 3 * 224 * 224),
  },
});
console.log(outputs.logits.shape, outputs.logits.data.length);

await session.dispose();
gpu.destroy();
```

Input tensors are borrowed, not copied: do not write to `data` until the promise returned by `run`
has settled.

`openContainer` checks the two descriptors against expected lengths and digests only when you pass
them as its second argument; without them nothing vouches for the descriptors' integrity. Pass them
whenever the bytes come from somewhere you do not control — `@karume/hub` supplies them from the
manifest (`container.descriptor`).

Admission is a separate step, so that "this device cannot run this model" is decided before a single
weight block is read. Given a lazy block source (such as `@karume/hub`'s `openContainerSource`),
`openContainer` reads only part 0. `prepareContainer` checks the graph against the runtime's
capabilities, `estimate` reports the memory the model will need, and `createContainerSession` reads
the weight blocks one at a time, uploading each and letting it go.

```ts
import { type BlockSource, openContainer, prepareContainer } from "@karume/runtime";

// A lazy source over the numbered parts a user picked in the browser, in order: slicing a `File`
// reads only that byte range.
const fileSource = (files: readonly File[]): BlockSource => ({
  partCount: files.length,
  verified: false, // nothing has checked these bytes, so every block read is digested
  partLength: (index) => files[index].size,
  read: async (part, offset, length) =>
    new Uint8Array(await files[part].slice(offset, offset + length).arrayBuffer()),
});

const lazy = await openContainer({ kind: "source", source: fileSource(files) });
const prepared = prepareContainer(lazy, "model");
console.log(prepared.estimate().resident.weights.totalBytes);
const session = await prepared.createContainerSession(gpu);
```

For weights built on the host rather than read from a file, `openMemoryContainer` returns the same
kind of opened container without writing one.

## Requirements

- A WebGPU device. Deno 2 (tested on 2.9.6) exposes WebGPU without a flag; in the browser, any
  engine with a working `navigator.gpu` will do.
- A real GPU adapter. There is no CPU fallback — `acquireGpu` throws `GpuUnavailableError` when no
  adapter can be obtained.
- Nothing else: the runtime uses Web-standard APIs only, and no build step or native module.

## Documentation

- [Repository](https://github.com/hdae/karume) — overview and examples
- [docs/ir-v2.md](https://github.com/hdae/karume/blob/main/docs/ir-v2.md) — the IR format
- [docs/container-v1.md](https://github.com/hdae/karume/blob/main/docs/container-v1.md) — the
  container format (`krm` / `krg`)
- [docs/op-vocabulary.md](https://github.com/hdae/karume/blob/main/docs/op-vocabulary.md) — the
  operator vocabulary
- [docs/decisions/](https://github.com/hdae/karume/tree/main/docs/decisions) — architecture
  decision records, the source of truth for the design

## License

MIT — see [LICENSE](./LICENSE).

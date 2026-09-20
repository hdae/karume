# @karume/runtime

The execution engine of [Karume](https://github.com/hdae/karume), a general-purpose neural-network
inference stack that runs on WebGPU. This package opens a model container (a Karume IR v1 graph and
its weights, stored as safetensors), generates WGSL for it, uploads the weights, and executes the
graph on a GPU device. It is written in pure TypeScript and WGSL, has zero external dependencies,
and takes the same code path in Deno and in the browser. The other packages of the stack are
`@karume/hub` (manifest resolution, download, cache) and `@karume/models` (pipelines and
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

Acquire a GPU context, open a model file, build a session, run it, then release the two lifetimes
in order (session first, device last).

```ts
import { acquireGpu, createSession, openModel } from "@karume/runtime";

const gpu = await acquireGpu();
const model = openModel((await Deno.readFile("model.safetensors")).buffer);
const session = await createSession(gpu, model);

const outputs = await session.run({
  pixel_values: {
    dtype: "f32",
    shape: [1, 3, 224, 224],
    data: new Float32Array(1 * 3 * 224 * 224),
  },
});
console.log(outputs.logits.shape, outputs.logits.data.length);

await session.dispose();
gpu.device.destroy();
```

Input tensors are borrowed, not copied: do not write to `data` until the promise returned by `run`
has settled.

For a model that is distributed as several shards, admission is a separate step, so that "this
device cannot run this model" is decided before a single weight byte is downloaded. `prepareModel`
takes only the graph shard, `estimate` reports the memory the model will need, and `createSession`
consumes the remaining weight shards one at a time.

```ts
import { prepareModel } from "@karume/runtime";

const prepared = prepareModel({ id: "model-00001-of-00003.safetensors", bytes: graphShardBytes });
console.log(prepared.estimate().resident.weights.totalBytes);
const session = await prepared.createSession(gpu, weightShards);
```

## Requirements

- A WebGPU device. Deno 2 (tested on 2.9.6) exposes WebGPU without a flag; in the browser, any
  engine with a working `navigator.gpu` will do.
- A real GPU adapter. There is no CPU fallback — `acquireGpu` throws `GpuUnavailableError` when no
  adapter can be obtained.
- Nothing else: the runtime uses Web-standard APIs only, and no build step or native module.

## Documentation

- [Repository](https://github.com/hdae/karume) — overview and examples
- [docs/ir-v1.md](https://github.com/hdae/karume/blob/main/docs/ir-v1.md) — the IR format
- [docs/op-vocabulary.md](https://github.com/hdae/karume/blob/main/docs/op-vocabulary.md) — the
  operator vocabulary
- [docs/decisions/](https://github.com/hdae/karume/tree/main/docs/decisions) — architecture
  decision records, the source of truth for the design

## License

MIT — see [LICENSE](./LICENSE).

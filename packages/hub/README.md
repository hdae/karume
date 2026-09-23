# @karume/hub

The distribution layer of [Karume](https://github.com/hdae/karume), a general-purpose
neural-network inference stack that runs on WebGPU. This package reads a Karume distribution
manifest (`karume.json`, format `karume/5`), picks one model and one quantization out of it,
fetches the referenced containers (`krm`) and assets with integrity checking, and caches them. Sources are Hugging Face
repositories (pinned to a resolved revision) or a local directory. It depends only on packages
built from Web-standard APIs, so the same code runs in Deno and in the browser. The other packages
of the stack are `@karume/runtime` (IR execution) and `@karume/models` (pipelines and tokenizers).

Status: pre-1.0. Manifests of an older major are not read — the JSR packages and the `karume`
exporter move in lockstep, so a distribution built by an older exporter has to be rebuilt.

## Install

```sh
deno add jsr:@karume/hub
```

```sh
npx jsr add @karume/hub
pnpm dlx jsr add @karume/hub
bunx jsr add @karume/hub
```

## Minimal usage

Load the manifest, resolve one selection, and fetch the assets it points at. The revision is
resolved once, in `loadManifest`, and every later call fetches from that same pinned generation.

```ts
import { fetchAssets, loadManifest, resolveSelection } from "@karume/hub";

const loaded = await loadManifest({ repo: "hdae/karume-birefnet-hr" });
const selection = resolveSelection(loaded.manifest, { model: "1024" });
const assets = await fetchAssets(loaded, selection.assets);
console.log(Object.keys(selection.containers), Object.keys(assets));
```

A weight component is a container (`krm`) split into parts. Warm the parts into the cache with
`prefetchAssets` (progress, cancellation, four downloads at a time), then open the container as a
`BlockSource` for `@karume/runtime`'s `openContainer`: `openContainerSource` reads byte ranges
out of the warmed cache and never digests a warm hit, so peak RAM stays at one block (or one part
where the source can only scan).

```ts
import { openContainerSource, prefetchAssets, selectionRefs } from "@karume/hub";

await prefetchAssets(loaded, selectionRefs(selection));
const container = selection.containers["matte"];
const source = openContainerSource(loaded, container);
// openContainer({ kind: "source", source }, container.descriptor) — see @karume/runtime
```

To keep host memory flat while reading plain assets one at a time, `streamAssets` yields verified
byte ranges in the order the refs were given, so peak RAM stays at the size of the largest file.

```ts
import { streamAssets } from "@karume/hub";

for await (const asset of streamAssets(loaded, Object.values(files))) {
  console.log(asset.id, asset.bytes.byteLength);
}
```

A distribution that already sits on disk is passed as a source handle instead of a repository
reference. The Deno reader lives on its own subpath, so the browser build never pulls it in.

```ts
import { loadManifest } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";

const local = await loadManifest(denoDirectory("./models/karume-gemma4"));
```

## Requirements

- Deno 2 (tested on 2.9.6) or a browser. Network access for Hugging Face sources; the local
  directory source needs read access only.
- The browser cache path uses the `CacheStorage` API, which is available in a secure context only.
- No WebGPU is needed here — this package never touches a GPU device. Executing what it fetches is
  `@karume/runtime`'s job.

## Documentation

- [Repository](https://github.com/hdae/karume) — overview and examples
- [docs/decisions/0038-manifest-v1.md](https://github.com/hdae/karume/blob/main/docs/decisions/0038-manifest-v1.md)
  and [docs/decisions/0041-manifest-v2.md](https://github.com/hdae/karume/blob/main/docs/decisions/0041-manifest-v2.md)
  — the manifest and fetch-layer specification
- [docs/decisions/](https://github.com/hdae/karume/tree/main/docs/decisions) — architecture
  decision records, the source of truth for the design

## License

MIT — see [LICENSE](./LICENSE).

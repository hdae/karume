# @karume/models

The model layer of [Karume](https://github.com/hdae/karume), a general-purpose neural-network
inference stack that runs on WebGPU. This package turns a published distribution into a working
pipeline: it resolves the manifest through `@karume/hub`, runs the graphs through
`@karume/runtime`, and adds the host-side pieces a model needs on top of raw graph execution —
image pre- and post-processing, audio encode and decode, tokenizers, chat templates and sampling.
Everything is pure TypeScript; there is no Python, no native module and no build step at runtime.

Status: pre-1.0. Every pipeline has the same shape: a `fromPretrained` (fetches a distribution) or
`fromAssets` (bytes you already hold) constructor, one verb, and `dispose`.

## Install

```sh
deno add jsr:@karume/models
```

```sh
npx jsr add @karume/models
pnpm dlx jsr add @karume/models
bunx jsr add @karume/models
```

## Minimal usage

Each family has its own subpath export, so importing one pipeline does not pull in the others. A
subpath exports bare names — `Pipeline`, `SOURCES`, `PipelineOptions`, … — with no family prefix.
The `SOURCES` table of a family lists the distributions this package version was verified against,
pinned to a commit SHA.

```ts
import { Pipeline, SOURCES } from "@karume/models/birefnet";

// Fetches the manifest, admits the model from the container descriptors, fetches the parts, opens a GPU session.
const pipeline = await Pipeline.fromPretrained(SOURCES["birefnet-hr"]);

// `rgb` is a tight RGB8 pixel buffer (width * height * 3), decoded by the caller.
const matte = await pipeline.segment({ data: rgb, width, height });
console.log(matte.width, matte.height, matte.data.length);

await pipeline.dispose();
```

The barrel export (`@karume/models`) is for callers that use several families at once. It exports
the helpers shared by every family (`encodePng`, `encodeWav`, `decodeWav`, `ModelInputError`,
`KARUME_SOURCES`, …) and one namespace per family, named after the subpath in camelCase (`anima`,
`depthAnything`, `gemma4Qat`, `sbv2`, …). A namespace is the subpath module itself, so its members are
documented on the subpath's page:

```ts
import { anima, sbv2 } from "@karume/models";
// or, per family: import * as sbv2 from "@karume/models/sbv2";

const tts = await sbv2.Pipeline.fromPretrained(sbv2.SOURCES["sbv2-jvnv"]);
const painter = await anima.Pipeline.fromPretrained(anima.SOURCES["anima"]);
```

The package's modules are side-effect free, so a bundler drops the pipelines and members of the
families you do not use. A barrel import still keeps every family's source table (about 1.5 KB
minified) until the package declares `"sideEffects": false`
([backlog](https://github.com/hdae/karume/blob/main/docs/backlog.md)); a subpath import does not.

## Families

| Subpath                         | Task                                                     |
| ------------------------------- | -------------------------------------------------------- |
| `@karume/models/gemma`          | text generation and chat (Gemma 4)                       |
| `@karume/models/gemma4-qat`     | the same API on fixed mobile QAT weights                 |
| `@karume/models/anima`          | text to image                                            |
| `@karume/models/moss`           | text to image (MOSS-Image)                               |
| `@karume/models/sbv2`           | text to speech                                           |
| `@karume/models/irodori`        | text to speech                                           |
| `@karume/models/birefnet`       | image to alpha matte (background removal)                |
| `@karume/models/depth-anything` | image to relative depth map                              |
| `@karume/models/siglip2`        | image to embedding                                       |
| `@karume/models/umt5`           | text to encoder hidden states (umT5-XXL)                 |
| `@karume/models/vowel-detector` | Japanese speech to lip-sync labels                       |
| `@karume/models/wan`            | text to video (Wan2.1 / Wan2.2), image to video (Wan2.2) |

## Requirements

- A WebGPU device with a real GPU adapter — the same requirement as `@karume/runtime`, since every
  pipeline opens a session. There is no CPU fallback.
- Deno 2 (tested on 2.9.6) or a browser. Deno exposes WebGPU without a flag.
- Network access when a pipeline is built with `fromPretrained`. A distribution that is already on
  disk is passed as a source handle instead, and then nothing goes over the network.

## Documentation

- [Repository](https://github.com/hdae/karume) — overview and runnable examples per family
- [docs/glossary.md](https://github.com/hdae/karume/blob/main/docs/glossary.md) — terms used across
  the stack
- [docs/quantization.md](https://github.com/hdae/karume/blob/main/docs/quantization.md) — the
  quantization schemes a distribution may declare
- [docs/decisions/](https://github.com/hdae/karume/tree/main/docs/decisions) — architecture
  decision records, the source of truth for the design

## License

MIT — see [LICENSE](./LICENSE).

Model weights are distributed separately and carry their own upstream licenses; the MIT license
here covers this package's code only. A few functions are ported from diffusers (Apache-2.0) and
Pillow (MIT-CMU); [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) lists them with the upstream
license texts.

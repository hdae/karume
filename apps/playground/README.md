# Karume playground

A browser application that runs Karume pipelines on the browser's own GPU, reading the local
distributions under `models/` through the dev server. It replaces the Deno-era `tools/gpu-lab` page
(ADR 0127 decision 5) and is the first slice of the common shell for such pages.

Today it has one model page: **Wan2.2 TI2V 5B** (`/wan22`), for checking the presets and the
32-latent VAE tile (`vaeTile: 32`, ADR 0134) on a real GPU.

The app is private (not published) and imports only the public surfaces of the workspace packages:
`@karume/runtime`, `@karume/hub`, `@karume/models/wan` (and `encodePng` from the `@karume/models`
barrel, loaded only when frames are saved).

## Run

```sh
export PATH=$HOME/.local/share/vite-plus/bin:$PATH   # Node 24 + pnpm from Vite+
pnpm install
pnpm --filter @karume/playground dev
```

The dev server listens on **http://127.0.0.1:8790** (loopback only). It serves:

| URL | What |
| --- | --- |
| `/`, `/wan22` | the pages (with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, so the page is cross-origin isolated) |
| `/models/wan22/…` | `models/karume-wan2.2` (read-only, HTTP `Range`, `Cross-Origin-Resource-Policy: same-origin`) |
| `/models/umt5/…` | `models/karume-umt5-xxl` (the umT5 text encoder Wan's manifest references) |
| `/models/index.json` | which mounts exist, whether their directory is there, and the HF repo / revision of a mount served under `/hf/` |
| `/hf/api/models/<repo>/revision/<ref>` | the HF revision API: `{ "sha": "<revision>" }` for any ref of a served repo |
| `/hf/<repo>/resolve/<revision>/<path>` | the HF resolve URL: a file of the repo at its current revision (404 for any other revision) |

### The HF-compatible route (`/hf/`)

`/hf/` speaks the two requests the hub's HF source makes, so the page can read a distribution with
`hubUrl` set to `<page origin>/hf` and get the hub's CacheStorage for free. Two repos are served:

- `hdae/karume-wan2.2` (the public name, `WAN_SOURCES["wan2.2"].repo`) from `models/karume-wan2.2`.
  Its revision is the first 40 hex digits of the sha256 of that directory's `karume.json`,
  recomputed on every request: re-assembling the mirror changes the revision at once, so the hub
  never serves a stale manifest from its cache (the manifest entry is keyed by the revision-pinned
  URL; asset entries are keyed by content and stay valid).
- each cross-repo reference of that manifest, at the (repo, revision) the manifest declares, from the
  mount whose directory is named after the last element of the repo name (`hdae/karume-umt5-xxl` →
  `models/karume-umt5-xxl`). A reference no mount holds is logged at startup and answers 404.

The same rules as `/models/` apply: loopback only, GET / HEAD only, 404 for traversal and for
symlinks that leave the directory, COOP / COEP / CORP on every response. Responses are `no-store`
(the hub keeps its own copy; a browser HTTP-cache copy would double the disk use). Which mount is
served under which repo is set in `vite.config.ts` (`{ directory, hfRepo }`).

The directories are relative to the repository root and are set in `vite.config.ts`
(`distributionsPlugin({ distributions })`). A missing directory does not stop the server: it logs a
warning, `/models/<name>/` answers 404, and the page shows a disabled state naming the directory.

### From another machine (SSH tunnel)

The server binds to loopback, so open it from another machine through a tunnel, for example from
the Windows PC with the RTX 5070 Ti:

```sh
ssh -N -L 8790:127.0.0.1:8790 <user>@<this-machine>
```

Then open http://127.0.0.1:8790/wan22 in Chrome on that machine. Use `127.0.0.1` or `localhost`:
WebGPU and cross-origin isolation need a secure context, and loopback counts as one.

## The Wan2.2 page

The page acquires the GPU once (`acquireGpu({ subgroups: "available" })`) and builds the pipeline
for the chosen source, preset and text encoder path (construction creates no Session; the GPU work
starts with **Run**). One generation runs at a time.

| Control | What it does |
| --- | --- |
| Preset | The manifest's `quants` (`quant` of `fromPretrained`); the manifest's `defaultQuant` is preselected and marked "(default)". Changing it rebuilds the pipeline. |
| Source | **Cached** (default): the HF-compatible route (`/hf/`) through the hub's CacheStorage — the first run stores the weights in the browser, later runs read them from its disk instead of the network (or the SSH tunnel). **Direct**: `localDirectory` over plain `fetch` from `/models/` on every run, nothing cached. Changing it rebuilds the pipeline. |
| VAE tile | `vaeTile` of the request: 16 (default) or 32. 32 decodes in 32-latent tiles (fewer, larger tiles) and needs about 9.4 GiB on the device during the decode; the 10 GB class keeps 16. |
| Size | 1280×704 or 704×1280. |
| Frames | 4n+1 from 5 to 121 (default 33). The video plays at 24 fps. |
| Steps | Denoise steps (default 50). **Quick 2-step check** sets 2 steps and 17 frames — enough to run every stage, VAE tiles included. |
| Seed | The seed of the initial noise (default 42). |
| Shift | The flow-matching shift; empty = the manifest's default (5). |
| Text encoder | `precomputed`: the text embedding asset's fixed prompts (choose one by name), no umT5. `gpu`: umT5 on the GPU (about 5.3 GiB of weights) with a free prompt and negative prompt (empty negative = the official default). Changing it rebuilds the pipeline. |
| Run / Cancel | Run one generation; Cancel stops at the next stage boundary, step or VAE tile. |

The status card also shows the **cache**: how much of the distribution the hub has stored (files and
GiB over every preset, a file shared by presets counted once). **Refresh cache size** reads it again
(it is also re-read after each run); **Clear cache** deletes everything the hub stored for this origin
(`clearHubCache`) and rebuilds the pipeline, so the next cached run reads the weights over the network
again. To compare a cold and a warm run: clear, run, run again, and compare the build cost of the two
records (`supplyWaitMs` is where reading the weights lands).

While running, the status card shows each stage's start and end (`text_encoder` / `transformer` /
`vae_decoder`), the denoise step counter, the VAE tile counter and the elapsed seconds. Errors are
shown in full (message, stack, `cause` chain).

The result card plays the frames in a canvas (play / pause, frame scrubber) and shows a summary of the
record: adapter and device, source, preset, VAE tile, size, frames, steps, seed, shift, per-stage
seconds, wall seconds, the number of non-finite values in the output, the sha256 of the RGBA pixels of
all frames (concatenated in order, the `wanFrameToRgba` rule), the first denoise step and the
min / median / max of the rest (likewise for VAE tiles), the build cost summed over the components,
and per component the runs, weight memory, submit count, build cost and the attention kernels of its
last run (`attention_online` for the online route, `attention_qk` / `attention_stats` /
`attention_pv` for the materialized one).

### The record (`format: "karume-playground-wan22/2"`)

The JSON keeps the full detail; the page shows only the summary above.

| Field | What |
| --- | --- |
| `format` | `karume-playground-wan22/2` (records without it are the first version) |
| `device` | enabled `features` and `wgslLanguageFeatures` (sorted), `subgroupLanesLinear`, `gpuTimingEnabled`, and `limits` (`maxComputeWorkgroupStorageSize`, `maxStorageBufferBindingSize`, `maxBufferSize`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `timeline.denoiseStepSeconds` | seconds of every denoise step in order; the first is measured from the transformer stage start, so it also holds the Session creation and the weight upload |
| `timeline.vaeTileSeconds` | seconds of every VAE tile in order; the first holds the VAE Sessions' creation |
| `diagnostics.<component>.build` | the Session build cost of the component's latest report (`buildStats`): `partCount`, `supplyWaitMs`, `decodeMs`, `bufferCreateMs`, `writeBufferIssueMs`, `uploadFenceMs`, `uploadedBytes`. The GPU transfer itself cannot be told apart from the host clock; it lands in `uploadFenceMs` |
| `diagnostics.<component>.lastRunPipelines` | `{ key, dispatchCount }` of every pipeline the component's last run planned |

Per-kernel GPU time is not recorded: `WanTi2vPipeline` refuses a device acquired with
`gpuTiming: true` (its VAE stage runs one tile as one batch, and the runtime opens no batch on a
timing device), so the page cannot offer a timing toggle.

### Where records land

Nothing is written on the server. The buttons save into the **browser's download folder** of the
machine running Chrome:

- **Download record (JSON)** → `wan22_<UTC timestamp>.json`
- **Download frames (zip of PNGs)** → `wan22_<UTC timestamp>_frames.zip` (store-only zip,
  `frame_000.png` …, lossless RGB from the same RGBA bytes the sha256 covers)
- **Copy record** → the clipboard (paste into an issue or a research note)

## Develop

```sh
pnpm --filter @karume/playground test        # vp test: the server's request handling (/models/ and /hf/), record and cache helpers, zip writer
pnpm --filter @karume/playground typecheck
pnpm --filter @karume/playground check       # vp check: format + lint + types
pnpm --filter @karume/playground build
```

The tests need no GPU. The GPU path is checked by hand on the target machine.

`src/routeTree.gen.ts` is generated by the TanStack Router plugin (file-based routing under
`src/routes/`); do not edit it. UI components under `src/components/ui/` come from the Shadcn CLI
(Base UI backend): add more with `pnpm dlx shadcn@latest add @shadcn/<name>` in this directory.

### Adding a model page

1. Add the distribution mount in `vite.config.ts` (`distributions: { <name>: "models/<dir>" }`, or
   `{ directory, hfRepo }` to serve it on the HF-compatible route as well).
2. Add `src/routes/<page>.tsx` (`createFileRoute("/<page>")({ component })`) and put the page's code
   under `src/models/<page>/`.
3. Add an entry to `MODEL_PAGES` in `src/models/registry.ts` (the index lists it as a card, the
   header links it). Widen `ModelPage["path"]` with the new route.

The shared pieces a page builds on:

| Piece | Where |
| --- | --- |
| Distribution source over the dev server (`localDirectory` with range reads, `crossRepo`), and the `HubRepoRef` of the HF-compatible route (`hfRepoRef`) | `src/lib/distributions.ts` |
| What the hub's cache holds of a distribution, over every preset | `src/lib/hub-cache.ts` |
| Which mounts exist (`/models/index.json`) | `src/hooks/use-distribution-index.ts` |
| One GPU per page, released on unmount, re-acquire after device loss | `src/hooks/use-gpu.ts` |
| Record helpers (non-finite count, RGBA stack + sha256, diagnostics and build cost per component, attention route, device facts, timeline) | `src/lib/record.ts` |
| Downloads and the zip writer | `src/lib/download.ts`, `src/lib/zip.ts` |
| Error alert, labelled fields, select, key-value table, frame player | `src/components/` |

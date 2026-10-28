# Karume playground

A browser application that runs Karume pipelines on the browser's own GPU, reading the local
distributions under `models/` through the dev server. It replaces the Deno-era `tools/gpu-lab` page
(ADR 0127 decision 5) and is the first slice of the common shell for such pages.

A left sidebar switches between the pages, grouped by kind:

| Group | Page | Route | State |
| --- | --- | --- | --- |
| Video | Wan2.2 TI2V 5B — the presets and the 32-latent VAE tile (`vaeTile: 32`, ADR 0134) on a real GPU | `/wan22` | done |
| Image | Anima — the official models and the third-party fine-tunes | `/anima` | placeholder |
| Vision | BiRefNet (HR / Lucida) | `/birefnet` | placeholder |
| Vision | Depth Anything V2 | `/depth-anything` | placeholder |
| Vision | SigLIP 2 | `/siglip2` | placeholder |
| Audio | Irodori-TTS (v4 / v4.1 small) | `/irodori` | placeholder |
| Audio | Style-Bert-VITS2 (JVNV) | `/sbv2` | placeholder |
| Text | Gemma 4 chat (`gemma4` and the `gemma4-qat` distribution) | `/gemma4` | placeholder |
| Benchmarks | Attention route bench — the two a8 attention routes per kernel at Wan2.2's shapes | `/attention-bench` | done |

A placeholder shows the page's title and whether its mounts are there.

The app is private (not published) and imports only the public surfaces of the workspace packages:
`@karume/runtime` (and its `/tune` subpath), `@karume/hub` and the family subpaths of
`@karume/models` (the barrel only for `encodePng` / `encodeWav`, loaded when a PNG or a WAV is
made).

## Run

```sh
export PATH=$HOME/.local/share/vite-plus/bin:$PATH   # Node 24 + pnpm from Vite+
pnpm install
pnpm --filter @karume/playground dev
```

The dev server listens on **http://127.0.0.1:8790** (loopback only). It serves:

| URL | What |
| --- | --- |
| `/` and the page routes | the pages (with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, so the page is cross-origin isolated) |
| `/models/<mount>/…` | the mount's directory (read-only, HTTP `Range`, `Cross-Origin-Resource-Policy: same-origin`) — the mounts are below |
| `/models/index.json` | which mounts exist, whether their directory is there, and the HF repo / revision of a mount served under `/hf/` |
| `/hf/api/models/<repo>/revision/<ref>` | the HF revision API: `{ "sha": "<revision>" }` with the repo's computed revision for any ref, except an alias revision, which answers as itself ([the `/hf/` route](#the-hf-compatible-route-hf)) |
| `/hf/<repo>/resolve/<revision>/<path>` | the HF resolve URL: a file of the repo at its computed revision, a referenced file at an alias revision, or a file of a cross-repo reference at its declared revision; 404 otherwise ([the `/hf/` route](#the-hf-compatible-route-hf)) |

### The mounts

The mount table is `src/mounts.ts` (`vite.config.ts` hands it to `distributionsPlugin`; the page
registry names mounts with its `MountName` type). The directories are relative to the repository
root. The HF-compatible repo is the public name from the family's `*_SOURCES` table.

| Mount | Directory | HF-compatible repo (`/hf/`) | Page |
| --- | --- | --- | --- |
| `wan22` | `models/karume-wan2.2` | `hdae/karume-wan2.2` | Wan2.2 |
| `umt5` | `models/karume-umt5-xxl` | — (served as the cross-repo reference of Wan's manifest) | Wan2.2 |
| `anima` | `models/karume-anima` | `hdae/karume-anima` (plus the revision `anima-extra` references — below) | Anima |
| `anima-extra` | `models/karume-anima-extra` | `hdae/karume-anima-extra` | Anima |
| `birefnet-hr` | `models/karume-birefnet-hr` | `hdae/karume-birefnet-hr` | BiRefNet |
| `lucida` | `models/karume-lucida` | `hdae/karume-lucida` | BiRefNet |
| `depth-anything` | `models/karume-depth-anything-v2` | `hdae/karume-depth-anything-v2` | Depth Anything V2 |
| `siglip2` | `models/karume-siglip2` | `hdae/karume-siglip2` | SigLIP 2 |
| `irodori-v4` | `models/karume-irodori-v4-small` | `hdae/karume-irodori-v4-small` | Irodori-TTS |
| `irodori-v4.1` | `models/karume-irodori-v4.1-small` | `hdae/karume-irodori-v4.1-small` | Irodori-TTS |
| `sbv2` | `models/karume-sbv2-jvnv` | `hdae/karume-sbv2-jvnv` | Style-Bert-VITS2 |
| `gemma4` | `models/karume-gemma4` | `hdae/karume-gemma4` | Gemma 4 chat |
| `gemma4-qat` | `models/karume-gemma4-qat` | — (no public repo: direct only) | Gemma 4 chat |

A missing directory does not stop the server: it logs a warning, `/models/<name>/` answers 404, and
the pages show a disabled state naming the directory.

### The HF-compatible route (`/hf/`)

`/hf/` speaks the two requests the hub's HF source makes, so the page can read a distribution with
`hubUrl` set to `<page origin>/hf` and get the hub's CacheStorage for free. It serves:

- each mount with an HF-compatible repo, under that name. Its revision is the first 40 hex digits of
  the sha256 of the directory's `karume.json`, recomputed on every request: re-assembling the mirror
  changes the revision at once, so the hub never serves a stale manifest from its cache (the
  manifest entry is keyed by the revision-pinned URL; asset entries are keyed by content and stay
  valid).
- each cross-repo reference of those manifests, at the (repo, revision) the manifest declares, from
  the mount whose directory is named after the last element of the repo name
  (`hdae/karume-umt5-xxl` → `models/karume-umt5-xxl`). A reference no mount holds is logged at
  startup and answers 404.

**A referenced revision of a served repo (the Anima rule).** `karume-anima-extra` takes the text
encoder, the VAE and the tokenizers from `hdae/karume-anima` at the revision its manifest declares,
while the `anima` mount is served at its own computed revision. One directory holds one revision, so
by default the server refuses that (the table fails loudly). The `anima` mount opts in with
`aliasReferencedRevisions: true`, which makes the declared revision an **alias** of the directory:

- every file a manifest takes from the repo at that revision must be listed in the mount's own
  manifest as its own file with the same path, size and sha256 — otherwise the table still fails
  loudly (the alias would serve different bytes);
- the alias serves only those files; anything else at that revision answers 404;
- the computed revision stays the repo's revision (the revision API answers it for `main`; an alias
  revision is answered as itself), so the official models read as before;
- the server logs each alias at startup (repo, revision, directory, how many files, which repo
  references it).

The direct route needs no alias: `localDirectory` maps cross-repo references to mounts explicitly
(`pretrainedRef(source, "anima-extra", { "hdae/karume-anima": "anima" })`) and follows no revision.

The same rules as `/models/` apply: loopback only, GET / HEAD only, 404 for traversal and for
symlinks that leave the directory, COOP / COEP / CORP on every response. Responses are `no-store`
(the hub keeps its own copy; a browser HTTP-cache copy would double the disk use).

### From another machine (SSH tunnel)

The server binds to loopback, so open it from another machine through a tunnel, for example from
the Windows PC with the RTX 5070 Ti:

```sh
ssh -N -L 8790:127.0.0.1:8790 <user>@<this-machine>
```

Then open http://127.0.0.1:8790/ in Chrome on that machine. Use `127.0.0.1` or `localhost`:
WebGPU and cross-origin isolation need a secure context, and loopback counts as one.

## The Wan2.2 page

The page acquires the GPU once (`acquireGpu({ subgroups: "available" })`, plus `gpuTiming: true`
while **GPU timing** is on) and builds the pipeline for the chosen source, preset and text encoder
path (construction creates no Session; the GPU work starts with **Run**). One generation runs at a
time.

| Control | What it does |
| --- | --- |
| Scenario | A fixed set of settings for runs that are compared with each other (below): **Load** fills the form and the source, **Run this scenario** fills them and runs. |
| Preset | The manifest's `quants` (`quant` of `fromPretrained`); the manifest's `defaultQuant` is preselected and marked "(default)". Changing it rebuilds the pipeline. |
| Source | **Cached** (default): the HF-compatible route (`/hf/`) through the hub's CacheStorage — the first run stores the weights in the browser, later runs read them from its disk instead of the network (or the SSH tunnel). **Direct**: `localDirectory` over plain `fetch` from `/models/` on every run, nothing cached. Changing it rebuilds the pipeline. |
| VAE tile | `vaeTile` of the request: 16 (default) or 32. 32 decodes in 32-latent tiles (fewer, larger tiles) and needs about 9.4 GiB on the device during the decode; the 10 GB class keeps 16. |
| Size | 1280×704 or 704×1280. |
| Frames | 4n+1 from 5 to 121 (default 33). The video plays at 24 fps. |
| Steps | Denoise steps (default 50). **Quick 2-step check** sets 2 steps and 17 frames on the current settings — enough to run every stage, VAE tiles included (the **Quick check** scenario does the same on the scenarios' fixed settings). |
| Seed | The seed of the initial noise (default 42). |
| Shift | The flow-matching shift; empty = the manifest's default (5). |
| Text encoder | `precomputed`: the text embedding asset's fixed prompts (choose one by name), no umT5. `gpu`: umT5 on the GPU (about 5.3 GiB of weights) with a free prompt and negative prompt (empty negative = the official default). Changing it rebuilds the pipeline. |
| Run / Cancel | Run one generation; Cancel stops at the next stage boundary, step or VAE tile (or drops a scenario run that is still waiting for its pipeline). |
| GPU timing | Off by default; the switch sits in the status card. On: the GPU is acquired again with `gpuTiming: true` (which rebuilds the pipeline) and every run records its GPU time per kernel (below). Disabled while a generation runs, and with the reason when the adapter has no `timestamp-query`. |

### Scenarios

A scenario is a complete set of settings fixed in code (`src/models/wan22/scenarios.ts`), so runs
recorded on different days or machines differ only where they are meant to. "Scenario", not
"preset": on this page a preset is a quant of the manifest.

All four use the manifest's `defaultQuant`, the GPU text encoder, the prompt of the text embedding
asset's positive row `boxing-cats` (its text goes into the free-text prompt), an empty negative
prompt (the official default), seed 42, the manifest's shift, 1280×704 and the **Cached** source.

| Id | Label | Frames | Steps | VAE tile |
| --- | --- | --- | --- | --- |
| `quick-check` | Quick check | 17 | 2 | 16 |
| `bench-33f` | 33 frames, 50 steps (tile 16) — the comparison point with the earlier records | 33 | 50 | 16 |
| `bench-33f-tile32` | 33 frames, 50 steps (tile 32) | 33 | 50 | 32 |
| `bench-121f` | 121 frames, 50 steps (tile 32) | 121 | 50 | 32 |

A scenario is refused (the page says why, nothing changes) when the dev server serves no
HF-compatible route for the distribution, when the umT5 distribution is missing, or when the text
embedding asset has no positive `boxing-cats` row (the prompt list is known once a pipeline has
opened). **Load** and **Run this scenario** are disabled while a generation runs or the manifest is
not loaded.

**Run this scenario** does not start at once: applying the settings can rebuild the pipeline (a
different preset, text encoder or source). It shows "Waiting for the pipeline…" and starts the
generation when the open pipeline was built for the applied source, preset and text encoder and the
form still equals the scenario. The waiting run is dropped (and the page says why) when the form is
edited, the source is changed, Cancel is pressed, or the pipeline for the scenario fails to build.

After **Load** the form is an ordinary form: any edit makes it a custom one. The hint under the
select shows which scenario the form equals now, or "custom", worked out from the form and the source
on every render (the precomputed prompt choice is not compared — the GPU path does not read it).

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
| `scenario` | the id of the scenario the run's settings equal (its pipeline's preset, text encoder and source, and the form's request), or `custom`; records without it predate the scenarios |
| `timeline.denoiseStepSeconds` | seconds of every denoise step in order; the first is measured from the transformer stage start, so it also holds the Session creation and the weight upload |
| `timeline.vaeTileSeconds` | seconds of every VAE tile in order; the first holds the VAE Sessions' creation |
| `diagnostics.<component>.build` | the Session build cost of the component's latest report (`buildStats`): `partCount`, `supplyWaitMs`, `decodeMs`, `bufferCreateMs`, `writeBufferIssueMs`, `uploadFenceMs`, `uploadedBytes`. The GPU transfer itself cannot be told apart from the host clock; it lands in `uploadFenceMs` |
| `diagnostics.<component>.lastRunPipelines` | `{ key, dispatchCount }` of every pipeline the component's last run planned |
| `diagnostics.<component>.timing` | only with GPU timing on, for a component with at least one timed run: `runs` (timed runs), `totalMs`, `dispatchCount`, `clampedNegativeSamples` (samples the runtime clamped to 0 — non-zero means the breakdown is questionable), `ops` (`{ op, ms, dispatchCount }` per op — the key's text before the first `:`) and `keys` (`{ key, ms, dispatchCount }` per pipeline key), each summed over every timed run, most expensive first, ms at 0.1 ms (rounded once, after summing the nanoseconds) |

### GPU timing

With **GPU timing** on, the text encoder (umT5) and the DiT report the GPU time of every kernel of
every run (`lastRunTiming` through `onRunDiagnostics`). The record sums them per component over all
runs; the result card shows, per timed component, the timed total next to its stage's wall seconds
and the five most expensive ops with their share of that total.

- **What is not timed**: the VAE decode stage. Each tile runs as one batch, and a batch section on a
  timing device runs untimed (the runtime collects per-pass timestamps only at a run's flush). The
  VAE decoder components carry no `timing`.
- **Timed runs are slower**: on a timing device every dispatch becomes its own pass, so the stage
  seconds, step seconds and wall seconds of a timed run are not comparable with untimed records.
  Compare timed records with timed records.
- **Scenarios**: timing is a device setting, not a form field, so a run with timing on still records
  the scenario its settings equal. `device.gpuTimingEnabled` tells a timed run of a scenario from an
  untimed one (the result card shows "· GPU timing on" next to the scenario).
- Chrome without developer flags rounds timestamps to 100 µs. Each dispatch is its own pass, so a
  dispatch shorter than that reads as 0 or 0.1 ms: the totals of ops made of many short dispatches
  are coarse, the ones made of long dispatches (`linear`, `attention_online`) much less so.

### Where records land

Nothing is written on the server. The buttons save into the **browser's download folder** of the
machine running Chrome:

- **Download record (JSON)** → `wan22_<UTC timestamp>.json`
- **Download frames (zip of PNGs)** → `wan22_<UTC timestamp>_frames.zip` (store-only zip,
  `frame_000.png` …, lossless RGB from the same RGBA bytes the sha256 covers)
- **Copy record** → the clipboard (paste into an issue or a research note)

## The attention route bench page

This page times the two attention routes on their own, apart from a generation, with the kernels,
keys, binding order, row blocks and variants production would use on this device
(`runAttentionRouteBench` of `@karume/runtime/tune`):

- **s16** — the a8 route with S stored in f16: ①QK, ② row statistics and ③PV, each timed on its own
  and summed;
- **online** — the online int8 kernel (`attention_online`, ADR 0132), one dispatch;
- **quantization** (optional) — the four dispatches both routes share (q and k per token, the
  permute to Vᵀ and its quantization), once per node and outside the route totals.

The page probes the adapter first and acquires its GPU with `subgroups: "available"` and, when the
adapter lists `timestamp-query`, `gpuTiming: true`. Without timestamp-query it says so and the
times are wall-clock (per pass, divided by the repetitions — they include the submit overhead).
Chrome without developer flags rounds timestamps to 100 µs; the bench stacks repetitions until one
pass is about 80 ms, so that rounding stays below 0.2%.

| Control | What it does |
| --- | --- |
| Shapes | Wan2.2 TI2V-5B (24 heads, D = 128): self-attention at 33 frames (N = 7,920), self-attention at 121 frames (N = 27,280), cross-attention over the 512 text tokens at 121 frames (M = 27,280). |
| Rows per dispatch | Empty (default) = the first row block production would dispatch: for s16 the first block of the even split under the device's `maxStorageBufferBindingSize`, for online the first window of its work budget. A number overrides both (capped at the shape's query count; the online kernel rounds up to whole 64-row workgroups). |
| Rounds | Measurement rounds per dispatch (default 5); the min is kept. |
| Quantization | Also time the shared quantization dispatches (default on). |
| Run / Cancel | Run the bench; Cancel stops before the next dispatch (the record keeps the finished shapes). |

The result table has one row per shape and route: rows per dispatch, the production split, ms per
dispatch, ms per 1,000 query rows, integer TOPS (`4 · heads · rows · N · D` operations per dispatch),
the node estimate (ms per row × the node's query rows, quantization excluded), the **online / s16**
ratio of the per-row times (below 1 = online faster), the s16 stages and the variant (the online
reduction form `sg32` / `wg` with its workgroup storage bytes, or the s16 row window). A route
production would not run on this device or shape shows the reason instead. The status card shows the
device facts: geometry profile, the dp4a variant (and the canary's verdict), the subgroup features
and the linear lane mapping, and the limits.

**Download record (JSON)** saves `attention-bench_<UTC timestamp>.json` (`format:
"karume-playground-attention-bench/1"`): the adapter, the device facts (as on the Wan2.2 page), the
settings, and under `bench` the runtime's record (`karume-attention-bench/1`) with every stage's
key, workgroups, reps, raw rounds and wall-clock rounds. **Copy record** copies the same JSON.

Absolute times compare only on one machine; across machines, compare the ratios.

## Develop

```sh
pnpm --filter @karume/playground test        # vp test: the server's request handling (/models/ and /hf/, aliases included), the page registry, record / cache / preset / image helpers, pretrainedRef, zip writer, attention bench table, Wan2.2 scenarios
pnpm --filter @karume/playground typecheck
pnpm --filter @karume/playground check       # vp check: format + lint + types
pnpm --filter @karume/playground build
```

The tests need no GPU. The GPU path is checked by hand on the target machine.

`src/routeTree.gen.ts` is generated by the TanStack Router plugin (file-based routing under
`src/routes/`); do not edit it. UI components under `src/components/ui/` come from the Shadcn CLI
(Base UI backend): add more with `pnpm dlx shadcn@latest add @shadcn/<name>` in this directory.

### The shell

`src/routes/__root.tsx` is the shell: Shadcn's `SidebarProvider`, the sidebar
(`src/components/app-sidebar.tsx` — the registry's pages by group, the current route highlighted,
collapsible to icons on a desktop and a sheet on a phone), a header with the sidebar trigger and the
current page's title, and the page in the inset. The index route lists the same pages as cards, with
the state of their mounts.

`src/models/registry.ts` drives both: `MODEL_PAGES` holds one entry per page (`path`, `title`,
`summary`, `group`, a lucide `icon`, the `mounts` it reads), and `PAGE_GROUPS` the order of
the groups. `path` is typed from the generated route tree, and `tests/registry_test.ts` fails when
a route file has no entry or an entry has no route file.

### Adding a model page

1. Add the mount to `src/mounts.ts` (`<name>: "models/<dir>"`, or
   `{ directory, hfRepo: <FAMILY>_SOURCES[...].repo }` to serve it on the HF-compatible route too).
2. Add `src/routes/<page>.tsx` (`createFileRoute("/<page>")({ component })`) and put the page's code
   under `src/models/<page>/` (`page.tsx` exports `<Pascal>Page`). Run `pnpm --filter
   @karume/playground build` (or the dev server) once so the router plugin regenerates
   `src/routeTree.gen.ts`.
3. Add an entry to `MODEL_PAGES` in `src/models/registry.ts` (the sidebar and the index list it).

The kit a page builds on:

| Piece | Where | What it does |
| --- | --- | --- |
| `useDistributionIndex`, `mountStatus` | `src/hooks/use-distribution-index.ts`, `src/lib/distributions.ts` | Which mounts exist (`/models/index.json`) and one mount's status. |
| `MountBadges` | `src/components/mount-badges.tsx` | A badge per mount: its directory, "(missing)" when absent. |
| `useManifest` | `src/hooks/use-manifest.ts` | The mount's manifest over `/models/` and the sha256 of its `karume.json` (`loading` / `missing` / `ready` / `error`). |
| `manifestPresets`, `presetChoices`, `modelChoices` | `src/lib/presets.ts` | A model's presets (quants with label and description) and its `defaultQuant`; select choices with "(default)". |
| `PresetSelect` | `src/components/preset-select.tsx` | The Preset field: the presets, the default marked, the description as the hint. |
| `usePipelineSource` | `src/hooks/use-pipeline-source.ts` | The cached / direct choice for a mount: a stable `PipelineSource`, the cached route's `HubRepoRef`, `cacheCleared()` after clearing the hub's cache. |
| `SourcePicker` | `src/components/source-picker.tsx` | The Source field (Cached / Direct, Cached disabled without an HF-compatible route). |
| `pretrainedRef` | `src/lib/distributions.ts` | The first argument of `fromPretrained` for a `PipelineSource`: the `HubRepoRef`, or the mount over plain fetch with `crossRepo` mapped to other mounts. |
| `distributionSource`, `hfRepoRef` | `src/lib/distributions.ts` | The two sources underneath (`localDirectory` with range reads; the `HubRepoRef` of `/hf/`). |
| `queryCacheInventory` | `src/lib/hub-cache.ts` | What the hub's cache holds of a distribution, over every preset. |
| `useGpu` | `src/hooks/use-gpu.ts` | One GPU per page, released on unmount, re-acquired after a device loss. |
| `useTimestampQuery` | `src/hooks/use-timestamp-query.ts` | Whether the adapter lists `timestamp-query`. |
| `useRunTask` | `src/hooks/use-run-task.ts` | One cancellable run: state (`idle` / `running` with stage and counter / `done` / `cancelled` / `error`), the elapsed-time ticker, Cancel, an abort when the page is left. |
| `RunStatusCard`, `RunStatus`, `GpuAlerts` | `src/components/run-status-card.tsx` | The Status card: GPU, cross-origin isolation, the page's rows and controls, the GPU's failure states, and the run (elapsed, stage, progress bar, Cancel, error in full). |
| `recordEnvironment` | `src/lib/environment.ts` | The fields every record starts with: `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device`. |
| Record helpers | `src/lib/record.ts` | Non-finite count, RGBA stack + sha256, diagnostics and build cost per component, attention route, GPU time per key and per op, device facts, `adapterText`, timeline. |
| `RecordActions` | `src/components/record-actions.tsx` | Download record (JSON) and Copy record, with room for the page's own buttons. |
| `downloadBlob`, `downloadJson`, `fileStamp`, `zipStored` | `src/lib/download.ts`, `src/lib/zip.ts` | Downloads and the zip writer. |
| `ImageInput`, `decodeImageFile`, `rgbaToRgb8` | `src/components/image-input.tsx`, `src/lib/image.ts` | An image file field with a preview; the decoded RGBA and the pipelines' `Rgb8Image` (alpha dropped; a canvas keeps pixels premultiplied, so a fully transparent pixel reads as black). |
| `RgbaImage` | `src/components/rgba-image.tsx` | RGBA pixels in a canvas, with Download PNG for an opaque image (`encodePng` refuses alpha other than 255; a refused encode is shown). |
| `AudioClip` | `src/components/audio-clip.tsx` | A mono f32 waveform as a WAV: a player and Download WAV. |
| `ErrorAlert`, `Field`, `ChoiceSelect`, `KeyValueTable`, `FramePlayer` | `src/components/` | Error in full, labelled field, select, key-value table, frame player. |

Chat pieces (`message`, `message-scroller`, `bubble`, `input-group`) and `tabs`, `scroll-area`,
`spinner`, `empty`, `tooltip` are installed under `src/components/ui/` from the Shadcn CLI.

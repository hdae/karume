# Karume playground

A browser application that runs Karume pipelines on the browser's own GPU, reading the local
distributions under `models/` through the dev server. It replaces the Deno-era `tools/gpu-lab` page
(ADR 0127 decision 5) and is the first slice of the common shell for such pages.

A left sidebar switches between the pages, grouped by kind:

| Group | Page | Route | State |
| --- | --- | --- | --- |
| Video | Wan2.2 TI2V 5B — the presets and the 32-latent VAE tile (`vaeTile: 32`, ADR 0134) on a real GPU | `/wan22` | done |
| Image | Anima — the official models and the third-party fine-tunes | `/anima` | done |
| Image | MOSS-Image — text → image | `/moss-image` | built; not yet checked in a browser |
| Vision | BiRefNet (HR / Lucida) | `/birefnet` | done |
| Vision | Depth Anything V2 | `/depth-anything` | done |
| Vision | SigLIP 2 | `/siglip2` | done |
| Audio | Irodori-TTS (v4 / v4.1 small) | `/irodori` | done |
| Audio | Style-Bert-VITS2 (JVNV) | `/sbv2` | done |
| Text | Gemma 4 chat (`gemma4`, the q4_0 QAT line `gemma4-qat` and the mobile QAT `gemma4-qat-mobile`) | `/gemma4` | done |
| Benchmarks | Attention route bench — the two a8 attention routes per kernel at Wan2.2's shapes | `/attention-bench` | done |
| Benchmarks | Expert transfer bench — the fence and router-readback round trips and the upload of Gemma 4 26B A4B experts (staging ring or `writeBuffer`) | `/expert-transfer` | done |
| Benchmarks | Geometry sweep — the GEMM tile geometries case by case, one merged record for `deriveGeometryProfile` | `/geometry-sweep` | done |
| Benchmarks | Session A/B — the wall-clock A/B of a Gemma 4 decode with and without the candidate fusion rule (`runSessionAb`, `judgeSessionAb`): token identity, hits, dispatches, per-round times and the verdict | `/session-ab` | built; not yet checked in a browser |
| Benchmarks | Kernel checks — the subgroup32 GEMV / RMS reductions, the rms_norm prefetch form, the K-55 / K-91 states-attention rewrites, the integer and byte-plane GEMVs, the f16 full state slots, the f16 attention scores, the bidirectional block, the expert linear and the built-in fusions against their unfused plans, word for word on the browser's backend | `/kernel-check` | done |

Every model page reads its distributions from the mounts below, runs on a GPU the page acquires
itself, and saves a JSON record of each run (the environment, the request, timings and Session
diagnostics, plus the output's sha256 where there is one) for comparing runs across presets, sources
and machines. The benchmark pages read no distribution; the geometry sweep page acquires no GPU of
its own (`runGeometrySweep` does) and saves the runtime's sweep record as it is.

The app is private (not published) and imports only the public surfaces of the workspace packages:
`@karume/runtime` (and its `/tune` subpath), `@karume/hub` and the family subpaths of
`@karume/models` (the barrel only for `encodePng` / `encodeWav` / `decodeWav`, loaded when a PNG or
a WAV is made or read, and for type-only imports). Besides the UI libraries, the one outside package
a pipeline input comes from is `@hdae/yomi`, a dependency of the app: the caller-side Japanese
analyser whose `sbv2.Utterance` the Style-Bert-VITS2 page hands to `sbv2.Pipeline` (the same analyser
the SBV2 WAV regression test uses).

One exception to the public surfaces: the kernel-check page imports `@karume/kernel-checks`, a
private workspace package (never published) that drives runtime internals — kernel WGSL generators
and the runtime tests' dispatch harness — so that the page runs the same check implementation as
the runtime's node GPU tests ([the kernel-check page](#the-kernel-check-page)). No other page may
import it, and nothing here imports runtime internals directly. The session A/B page imports
`@karume/candidate-rules`, another private workspace package: candidate fusion rules written against
`@karume/runtime/kernel` alone, which the page passes to the gemma pipeline to measure them
([the session A/B page](#the-session-ab-page)).

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
root. The HF-compatible repo is the public name from the family's `SOURCES` table (imported as a namespace, `anima.SOURCES`).

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
| `gemma4-qat` | `models/karume-gemma4-qat` | `hdae/karume-gemma4-qat` | Gemma 4 chat |
| `gemma4-qat-mobile` | `models/karume-gemma4-qat-mobile` | `hdae/karume-gemma4-qat-mobile` | Gemma 4 chat |
| `moss-image` | `models/karume-moss-image` | `hdae/karume-moss-image` | MOSS-Image |

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
all frames (concatenated in order, the `wan.frameToRgba` rule), the first denoise step and the
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

## The Anima page

Text → image with `anima.Pipeline` (`@karume/models/anima`), over both Anima distributions: the five
official models of `karume-anima` (mount `anima`) and the two third-party fine-tunes of
`karume-anima-extra` (mount `anima-extra`). The fine-tunes take their text encoder, VAE and
tokenizers from the official repository. On the cached route the hub follows that cross-repo
reference on `/hf/` (the alias revision above). On the direct route the page maps every repository
the manifest references to the mount that holds it (`hdae/karume-anima` → `anima`), and fails loudly
when that mount is missing.

**When the GPU and the pipeline are set up.** The page acquires its GPU without optional features.
While a preset that computes in f16 is chosen (`f16-c16`, the manifest's `gpuFeatures.shaderF16`),
it acquires the GPU with `shaderF16: true` instead. A feature can be requested only when the device
is created, and the runtime runs an f16 check at acquisition. A device that fails that check
therefore only blocks `f16-c16`, never the other presets. Switching to or from `f16-c16` acquires
the GPU again: the open pipeline (and a resident DiT) is disposed through `track` in the same
commit, so the old device is destroyed only after that dispose. Run is disabled until the new GPU is
ready. The pipeline is **opened on Run**, not when a select changes: on the cached route, opening
stores the weights in the hub's cache (several GiB for `bf16`), so browsing the selects must not
download them. The open pipeline stays for the next run while the GPU, source, model and preset stay
the same. It is disposed when one of them changes, on **Release pipeline**, and when the page is
left. One generation runs at a time.

| Control | What it does |
| --- | --- |
| Model | Every model of both manifests ("Official · …" / "Fine-tune · …"), each mount's `defaultModel` marked "(default)". The official default (`anima-turbo-v1.1`) is preselected; without the official distribution, the fine-tunes' default. |
| Preset | The model's `quants`, `defaultQuant` preselected and marked, the manifest's description as the hint: f16, the int8 presets, `f16-c16` (computes in f16: choosing it acquires the GPU with shader-f16, and a hint says so, or shows that the acquisition failed) and `bf16` (the upstream weights bit for bit, same size as f16). |
| Source | **Cached** (default, the HF-compatible route through the hub's CacheStorage) or **Direct** (`localDirectory` over plain fetch, nothing cached). |
| Prompt | Free text (Danbooru-style tags work best), with four sample prompts. |
| Negative prompt | Empty = the manifest's default. Disabled and not sent when the guidance in use is 1: the pipeline never computes the uncond side there and refuses a negative prompt. |
| Guidance | `guidanceScale`; empty = the manifest's default (1 for the Turbo models, 4 for the others). 1 = no CFG. |
| Steps | 2–1000; empty = the manifest's default (8 for Turbo, 20–30 for the others). |
| Resolution | `WxH` or one number for a square (`parseResolution`: each side a multiple of 16 from 512 to 2048), with quick choices; empty = the manifest's default (1024×1024). |
| Sampler | Manifest default (`euler`) or `euler` / `dpmpp-2m` (`anima.SAMPLER_TYPES`). Choosing one departs from the distributor's recommendation. |
| Seed | Integer ≥ 0 (default 42); **Random** draws one. |
| DiT residency | `residency` of the request. `per-stage` (default): every stage opens and closes its Session, and the peak is the largest stage. `transformer`: the DiT Session stays after the run, so the next run with the same open pipeline skips the DiT load. The text and VAE stages then stack on it (about +2.6 GiB at 1024² with the default preset), and the pipeline evicts it when a stage would not fit. A later `per-stage` run, **Release pipeline**, or changing the selection releases it. |
| Live preview | On by default: after every denoise step, `approximatePreview` of the step's latent (a linear projection to RGB at 1/8 of the image, not the VAE decode), drawn scaled up and blocky. |
| Run / Cancel | Run one generation (opening the pipeline first when needed). `generate` takes no signal: Cancel throws from the generation's `onEvent`, the API's way to stop it, so it takes effect at the next stage boundary, denoise step or VAE tile; a stage that is loading weights or encoding the prompt finishes first. While the pipeline opens, Cancel aborts `fromPretrained`. |
| Release pipeline | In the status card: disposes the open pipeline (and a resident DiT). A failed dispose (for example on a lost device) is shown in full under the button. |

The status card shows:
- the GPU and cross-origin isolation;
- whether shader-f16 is requested and enabled ("—" while the GPU is being acquired);
- the open pipeline: model · preset · source, seconds to open, and "DiT resident" while a DiT kept
  by `transformer` residency sits on the device;
- the run: the stage (`Text encoder (Qwen3)`, `Text conditioner`, `Transformer (DiT)`, `VAE decoder`),
  the step or VAE tile counter with the step's sigma, residency events, and MiB read while opening.

Errors are shown in full. If the GPU changes while the pipeline is opening (a device loss during a
long download), the new pipeline is released and the run fails with an error saying to press Run
again.

The result card shows the image (**Download PNG** — lossless, from the same RGBA bytes the sha256
covers) and a summary of the record.

### The record (`format: "karume-playground-anima/1"`)

| Field | What |
| --- | --- |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment(gpu)`; `device.features` lists `shader-f16` only for an f16 preset's run |
| `distribution` | `mount` (`anima` / `anima-extra`), `directory`, `manifestSha256` (sha256 of the mount's `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model` | `name`, `isDefault` (the mount's `defaultModel`) |
| `preset` | `quant`, `label`, `isDefault` |
| `request` | exactly the fields sent to `generate` (`prompt`, `seed`, `residency`, plus the fields the form set: `negativePrompt`, `guidanceScale`, `steps`, `resolution`, `sampler`) |
| `effective` | the values the run used once the manifest's defaults filled the rest: `steps`, `guidanceScale`, `resolution` (`WxH`), `sampler`, `negativePrompt` (`null` when not used) |
| `livePreview` | whether the preview was computed (it runs on the main thread between steps) |
| `pipeline` | `reused` (an earlier run's pipeline ran this one) and `openSeconds` (`null` when reused) |
| `residency.requested` / `residency.events` | the request's residency, and `{ action, reason }` of every `residency` event, in order (also the ones a cancelled run reports) |
| `residency.residentAtStart` | the open pipeline held a DiT kept resident by an earlier run when this run started |
| `residency.transformerBuildCarriedOver` | the DiT stage ran on that carried-over DiT (no eviction before the stage ended). Then `diagnostics.transformer.build` is the earlier run's build (a Session's build stats are fixed for its lifetime), not a cost of this run; the page's build total leaves it out |
| `stageSeconds` | start → end of `text_encoder`, `text_conditioner`, `transformer`, `vae_decoder` (a stage's Session build and release fall inside it) |
| `timeline.denoiseStepSeconds` / `timeline.vaeTileSeconds` | seconds of every step / tile; the first is measured from its stage start, so it holds the Session creation and the weight upload (unless the DiT was carried over) |
| `generateSeconds` | `generate` alone |
| `output` | `width`, `height`, `rgbaSha256` (sha256 of the RGBA bytes the pipeline returned; alpha is always 255) |
| `diagnostics.<component>` | per component: runs, weight GiB, submits, dispatches, geometry profile and the Session build cost (`build`) of its latest report |

**Download record (JSON)** saves `anima_<UTC timestamp>.json`, **Download PNG** saves `anima_<UTC timestamp>.png`,
**Copy record** copies the JSON.

### Caveats

- The page has no GPU-timing switch, so records carry no per-kernel `timing`.
- The page has no cache size or **Clear cache** controls. To compare a cold and a warm cached run,
  clear the cache from the Wan2.2 page (`clearHubCache` clears everything for this origin).
- When the pipeline re-runs a stage after evicting a resident DiT on an out-of-memory, it emits no
  second `stage` start. That stage's seconds, and its first step (or tile) interval, therefore also
  hold the failed attempt and the eviction.
- If the f16 check fails at acquisition, `f16-c16` cannot run on that machine (the error is shown);
  choosing any other preset acquires the GPU without the feature.

## The MOSS-Image page

Text → image with `moss.Pipeline` (`@karume/models/moss`) over the distribution
`karume-moss-image` (mount `moss-image`): MOSS-Image SFT with the Qwen3.5 text encoder, the int8
DiT and the f32 VAE decoder (ADR 0138 and its addenda of 2026-11-01). The mount is the local mirror
of `hdae/karume-moss-image` (`moss.SOURCES["moss-image"]`; `tools/export-recipes/moss/distribution.py`
assembles it), served on both routes: **Cached** (the HF-compatible route through the hub's
CacheStorage) and **Direct** (plain fetch from `/models/`).

**Memory and time.** `generate` opens and closes its stages one after another, every generation:
the text encoder (about 7.4 GiB of int8 weights) is uploaded, encodes the prompts and is released
before the text refiner, the DiT pair (about 4.5 GiB, `dit_skip22` borrowing `dit`'s weights) and
the VAE open. Each generation therefore reads and uploads about 12.3 GiB of weights (on Cached, from
the browser's cache after the first run). At 1024² × 50 steps (the defaults) a generation takes
minutes on a desktop GPU; a smaller size or fewer steps is a quicker check. The page says this in
its help text.

**When the GPU and the pipeline are set up.** The page acquires its GPU with `subgroups:
"available"` (the online attention kernel's subgroup form, never required). The pipeline is
**opened on Run** and stays open for the next run while the GPU, the source and the preset stay
the same. Opening reads the manifest, the container descriptors and the tokenizer and builds no
Session, so an open pipeline holds no GPU memory between runs. It is disposed when the preset, the
source or the GPU changes, on **Release pipeline**, and when the page is left. One generation runs
at a time. The distribution has one model (`sft`); the page runs the manifest's `defaultModel`
and offers no model choice.

| Control | What it does |
| --- | --- |
| Preset | The model's `quants` with the manifest's labels, `defaultQuant` (`f32+dit8-a8-attn8-online`) preselected and marked, the description as the hint. |
| Source | **Cached** (default): the HF-compatible route (`/hf/`) through the hub's CacheStorage. **Direct**: `localDirectory` over plain `fetch` from `/models/`. Changing it releases the open pipeline. |
| Prompt | Free text (a descriptive sentence works best; up to 4,096 tokens — a longer prompt is refused, the request's default `overflow`), with four sample prompts. |
| Negative prompt | Empty = the manifest's default (upstream's: the empty prompt). Disabled and not sent when the guidance in use is 1: no negative branch runs there and the pipeline refuses a negative prompt. |
| Width / Height | Each a positive multiple of 16; empty = the manifest's default (1024 × 1024), with quick choices (1024², 1152×896, 896×1152, 768², 512²). The pipeline also refuses an area over this device's cap of the untiled VAE. |
| Steps | A positive integer; empty = the manifest's default (50). The schedule snaps to the 1,000-point training grid, so it may run fewer. |
| Guidance | The CFG scale, ≥ 0; empty = the manifest's default (3). 1 = no negative branch. |
| Shift | The flow-matching shift, > 0; empty = the manifest's default (5). |
| Seed | Integer ≥ 0 (default 42); **Random** draws one. Karume's own noise generator: the same seed does not give upstream's image. |
| Run | Run one generation (opening the pipeline first when needed). |
| Release pipeline | In the status card: disposes the open pipeline. A failed dispose is shown in full under the button. |

The status card shows the GPU, cross-origin isolation, the open pipeline (model · preset · source,
seconds to open) and the run. **`generate` reports no stage events** (no progress hook, and it takes no
signal): while it runs the status shows "Generating" and the elapsed time only, and Cancel is
disabled with a note. Cancel works while the pipeline opens. A run whose GPU lease ends while it
generates (the page is left, the GPU is acquired again) drops its image when `generate` returns.

The result card shows the image (`imageToRgba` into a canvas; **Download PNG** — lossless, from the
same RGBA bytes the sha256 covers) and a summary of the record.

### The record (`format: "karume-playground-moss-image/1"`)

| Field | What |
| --- | --- |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment(gpu)` |
| `distribution` | `mount` (`moss-image`), `directory`, `manifestSha256` (sha256 of the mount's `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model` | `name`, `isDefault` (the manifest's `defaultModel`) |
| `preset` | `quant`, `label`, `isDefault` |
| `request` | exactly the fields sent to `generate` (`prompt`, `seed`, plus the fields the form set: `negativePrompt`, `width`, `height`, `steps`, `guidance`, `shift`) |
| `effective` | the values the run used once the manifest's defaults filled the rest: `width`, `height`, `steps`, `guidance`, `shift`, `negativePrompt` (`null` when not used) |
| `pipeline` | `reused` (an earlier run's pipeline ran this one) and `openSeconds` (`null` when reused) |
| `generateSeconds` | `generate` alone (every stage's weight upload, run and release fall inside it) |
| `output` | `width`, `height`, `rgbaSha256` (sha256 of the RGBA bytes of `imageToRgba`; alpha is always 255) |

There are no `stageSeconds`, `timeline` or `diagnostics`: the pipeline has no stage events and no
`onRunDiagnostics`.

**Download record (JSON)** saves `moss-image_<UTC timestamp>.json`, **Download PNG** saves
`moss-image_<UTC timestamp>.png`, **Copy record** copies the JSON.

### Caveats

- No per-stage time, no step counter and no GPU timing (the pipeline exposes none).
- No cancel during a generation: a 1024² × 50-step run occupies the GPU for minutes once started.
- `overflow` (`"truncate"` for a prompt over 4,096 tokens) and the `latents` form are not offered.

## The BiRefNet page

The page cuts the background out of one image with `birefnet.Pipeline` (`@karume/models/birefnet`) on
the page's GPU. The GPU is acquired with `acquireGpu({})`, since the family's presets declare no
features, and with `gpuTiming: true` while **GPU timing** is on.

The pipeline keeps one Session from construction to `dispose`. Construction reads the weights: about
432 MiB for HR's `f16` and 843 MiB for Lucida's `f32`. So the **first Run builds it**, and later
runs on the same selection reuse it. Changing the checkpoint, model, preset or source (or the GPU)
releases it, and so does leaving the page. A build that finishes after the GPU changed is disposed,
and the run fails saying so. One run at a time.

Before any weight is read, every selection is **estimated** on the page's GPU from the graph
descriptor alone (see "Estimate" below).

| Control | What it does |
| --- | --- |
| Checkpoint | **HR** (mount `birefnet-hr`, photos) or **Lucida** (mount `lucida`, illustrations, transparent objects, logos). Each is its own distribution. A missing one is disabled and marked "(missing)". |
| Model (input size) | The manifest's `models` (`model` of `fromPretrained`): `1024` (default) or `2048`. `segment` resizes the image to this square, and the matte comes back at the image's own size. |
| Preset | The model's `quants` (`quant` of `fromPretrained`), with `defaultQuant` preselected and marked "(default)". HR has `f16` only. Lucida has `f32` (default) and `f16`. |
| Source | **Cached** (default, the hub's cache over `/hf/`) or **Direct** (plain fetch from `/models/`), as on the Wan2.2 page. On Direct the load counter fills at once (nothing is downloaded into a cache), and the weights are read while the Session is built. |
| Image | Any format the browser decodes. Alpha is dropped, so a fully transparent pixel reads as black. The page does not resize it; the pipeline does. |
| Run / Cancel | Run builds the pipeline if needed, then segments. Run is enabled once the estimate has passed. Cancel works while the manifest loads and, on Cached, while the weights download into the cache. **The Session build is not cancellable**: it reads every weight block (from the cache, or from `/models/` on Direct) and uploads it, and takes no signal. The status shows it as "building the Session", with Cancel disabled and the reason under it. **The segmentation is not cancellable** either (`segment` takes no signal): Cancel stays disabled until it ends, with the reason under it; a Cancel during the hashing that follows still stops the run. |
| Estimate again | Shown only when the estimate failed. It runs the estimate again (for example after a network error). |
| GPU timing | Off by default, in the status card. When on, the GPU is acquired again with `gpuTiming: true` (which releases the pipeline), and the record keeps the matte Session's GPU time per kernel. Disabled while running and when the adapter has no `timestamp-query`. |

The status card shows:

- the GPU and cross-origin isolation
- whether the pipeline is resident (for which selection, seconds to open, number of runs)
- **Reads**: the bytes the selection reads (the manifest's parts for that model and preset, each
  file once)
- **Declared limits**: the preset's `requiredLimits` against the device
- **Estimate**
- the device's binding limits (`maxStorageBufferBindingSize` per binding, `maxBufferSize` per
  buffer)

**Estimate.** For every build (GPU, checkpoint, model, preset, source) the page runs these public
calls, which fetch only part 0 of the matte container (the header and the graph and model
descriptors, under 1 MiB):

1. `loadManifest` on the chosen source
2. `resolveSelection`
3. `prefetchAssets` on part 0 of the matte container
4. `openContainerSource` and `openContainer`
5. `prepareContainer(opened, "matte").estimate({ maxStorageBufferBindingSize, maxBufferSize })` with
   the GPU's limits

The estimate plans the intermediate buffers with the same planner as the Session (ADR 0093 decision
5).

- If an intermediate exceeds the binding limit, or a region exceeds the buffer limit, the estimate
  throws, naming every node. The page shows the error in full and keeps Run disabled.
- Without this check, the Session plans its intermediates when it first runs, so the same failure
  would come on the first segmentation, after the weights were read and uploaded.
- The largest intermediate binding is 320 MiB at 1024² and 1,280 MiB at 2048² (docs/limitations.md,
  "The BiRefNet family's distribution…").
- When it passes, the row shows the peak the runtime accounts for (`peakAccountedBytes`, not an
  upper bound), the resident weights and the largest run shape (its input and readback buffers plus
  its intermediates).

The **Declared limits** row compares the preset's `requiredLimits` with the device. A shortfall
lists every short limit and also disables Run. Both current distributions declare none, so the row
reads "the preset declares no device limits".

The result card has three tabs:

- **Cut-out**: the image with the matte as straight alpha, over a checkerboard. Display only, since
  `encodePng` refuses translucent pixels.
- **Matte**: opaque grayscale, white = foreground, with **Download PNG**.
- **On white**: the cut-out composited on white, `round(c · a + 255 · (1 − a))` with `a = α / 255`,
  opaque, with **Download PNG**.

Beside the tabs it shows a summary of the record.

### The record (`format: "karume-playground-birefnet/1"`)

| Field | What |
| --- | --- |
| `format`, `page` | `karume-playground-birefnet/1`, `birefnet` |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment(gpu)`, as on every page (`device.gpuTimingEnabled` tells a timed run apart) |
| `distribution` | `checkpoint` (`birefnet-hr` / `lucida`), `directory`, `manifestSha256` (sha256 of the mount's `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model` | `name` (the input size) and `isDefault` |
| `preset` | `quant`, `label` (the manifest's label, else the name), `isDefault` |
| `request.image` | `name`, `width`, `height` and `rgbSha256`: sha256 of the decoded RGB bytes the pipeline received, so two records show whether they ran on the same input |
| `pipeline` | `reused` (false = this run built it), `loadSeconds` (`fromPretrained`: manifest, weights from network or cache, Session build), `segmentRun` (1 = the pipeline's first segmentation, which also compiles the GPU pipelines and plans the intermediates; then 2, 3…) |
| `timings` | `segmentSeconds` (`segment`: preprocess, the Session run, readback, sigmoid, resize to the input size) and `wallSeconds` (from Run to the matte; when `pipeline.reused` is false it also holds the release of a previous pipeline, if any, and the load) |
| `output` | `width`, `height` and `matteSha256`: sha256 of the matte's raw bytes (one 8-bit alpha per pixel, row-major, at the input size) |
| `diagnostics.matte` | the Session's report of this run: `runs`, `weightsAllocatedGiB`, `submitCount`, `dispatchCount`, `geometryProfile`, `build` (as on the Wan2.2 page), `lastRunPipelines`, and with GPU timing on `timing` (`totalMs`, `ops`, `keys`, `clampedNegativeSamples`). Absent when the run reported none. |

### Where records land

- **Download record (JSON)** → `birefnet_<UTC timestamp>.json`
- **Download PNG** on the Matte tab → `birefnet_<checkpoint>-<model>-<preset>_<UTC timestamp>_matte.png`
- **Download PNG** on the On white tab → `birefnet_<checkpoint>-<model>-<preset>_<UTC timestamp>_white.png`
- **Copy record** → the clipboard

### Caveats

- Matte sha256 values match across machines only for the same input, checkpoint, model and preset.
  Even then, f32 rounding can differ by GPU (the repository's sha reference rows are per environment
  — ADR 0106). A mismatch between machines is a finding to look at, not a failure by itself.
- `segmentRun: 1` includes compiling the GPU pipelines. Compare speed on `segmentRun ≥ 2`.
- The estimate reads part 0 of the selection's container on every selection change, about 0.7 MiB.
  On Cached it lands in the hub's cache, as `fromPretrained` would do anyway.
- 2048² needs about 3.5 GiB on the device (Lucida `f32`; about 3.1 GiB for HR `f16`). It does not
  run on the Intel Arc B570 (Linux xe driver job timeout — docs/limitations.md).

## The Depth Anything V2 page

Monocular relative depth with `depthAnything.Pipeline` (`@karume/models/depth-anything`). Pick an
image, estimate its depth on the page's GPU, look at the map, save it as a PNG, and save a record of
the run.

The page acquires its GPU once. Untimed it calls `acquireGpu({})`: the family's presets declare no
GPU features, so this is the device the pipeline would acquire itself. With **GPU timing** on it
calls `acquireGpu({ gpuTiming: true })`.

The pipeline is **built on the first Run**, not when the page opens. Construction fetches the
weights and creates the one Session, which uploads them (about 99 MB for Small). Later runs reuse
the open pipeline: one Session stays loaded on the GPU and serves many images, as the family
intends. When the model, preset, source or GPU changes, the open pipeline is disposed at once, and
the next Run builds the new one. The pipeline is owned by a small slot outside React state
(`pipeline-slot.ts`). The slot also disposes a build that finishes after the settings changed or
after the page was left. One run at a time.

The image goes in at its own size. The pipeline resizes it to the model's input (518×518 for Small,
bicubic, aspect not kept). It returns raw f32 relative depth at the input size: larger = nearer,
with no unit and no origin. Every value is finite: `estimate` throws on a non-finite value instead
of returning it.

| Control | What it does |
| --- | --- |
| Model | The manifest's `models` (`model` of `fromPretrained`). `defaultModel` is preselected and marked "(default)". Choosing another model selects its `defaultQuant`. |
| Preset | The model's `quants` (`quant` of `fromPretrained`). `defaultQuant` is preselected and marked "(default)". |
| Source | **Cached** (default): the HF-compatible route (`/hf/`) through the hub's CacheStorage. **Direct**: `localDirectory` over plain `fetch` from `/models/`. |
| Image | Any format the browser decodes; alpha is dropped. The preview shows the decoded pixels. |
| Estimate depth | Builds the pipeline if needed, then runs `estimate` once. The hint next to it says why it is disabled. |
| Cancel | Stops the manifest and weight fetch (the signal reaches only the hub's fetch layer). The Session build (weight upload) and `estimate()` take no signal, so Cancel is disabled during both, with the reason under it. |
| GPU timing | In the Status card; off by default. When on, the GPU is acquired again with `gpuTiming: true`, which drops the open pipeline, and the record carries the GPU time per op and per key. Disabled while running. When the adapter has no `timestamp-query`, it is disabled with that reason. |
| Coloring | **Inferno** (default): matplotlib's perceptually uniform map, dark = far, light yellow = near. **Grayscale**: black = far, white = near. It recolours the current result without running again. |
| Download PNG | The map as shown (the current coloring) → `depth-anything_<UTC timestamp>_<coloring>.png`. |

The map is min-max normalized. Relative depth has no fixed scale, so the shades of two images do not
compare. A flat map draws as one colour.

The Status card shows:

- the GPU and cross-origin isolation;
- the manifest;
- the open pipeline: model · preset · source · build seconds, or "not built";
- the model's input size.

While running it also shows the stage:

1. `loading the pipeline`: manifest and admission checks; Cancel stops it.
2. `fetching the weights`: MiB read and the file path; Cancel stops it.
3. `building the Session`: not cancellable. The page switches to this stage once the hub's progress
   reports every file in.
4. `estimating`: not cancellable.
5. `hashing the output`.

### The record (`format: "karume-playground-depth-anything/1"`)

| Field | What |
| --- | --- |
| `format`, `page` | `karume-playground-depth-anything/1`, `depth-anything` |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment` (the device facts as on the Wan2.2 page) |
| `distribution` | `directory` and `manifestSha256` (sha256 of the mount's `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model`, `preset` | `{ name, isDefault }` and `{ quant, label, isDefault }` |
| `request.image` | `name`, `width`, `height` and `rgbSha256`: the sha256 of the RGB8 bytes given to `estimate` (only the image's own bytes; the same file gives the same hash on every machine) |
| `request.modelInput` | the square size the graph runs at (`pipelineConfig.imageWidth` / `imageHeight`), or `null` |
| `timings.pipelineLoadSeconds` | building the pipeline (fetch, checks, Session creation, weight upload) when this run built it; `null` when the run reused one an earlier run built |
| `timings.estimateSeconds` | the `estimate` call: host preprocessing, the one `depth` run, the resize back to the input size |
| `timings.postprocessSeconds`, `timings.wallSeconds` | the output range and the two sha256; the whole run |
| `output` | `width`, `height` (the input size), `min` / `max` of the raw depth, and `depthSha256`: the sha256 of the raw f32 depth (little-endian bytes, row-major). This is the value to compare across machines. |
| `diagnostics.depth` | The Session diagnostics of this run (one report per estimate, so `runs` is 1). It holds the weight GiB and `submitCount` / `dispatchCount` **of this run**: the Session counts over its whole life, and the page subtracts the previous run's counts. It also holds the geometry profile, `build` (`buildStats`), `lastRunPipelines`, and, with GPU timing on, `timing` (per op and per key, as on the Wan2.2 page). |

**Download record (JSON)** → `depth-anything_<UTC timestamp>.json`; **Copy record** copies the same
JSON.

### Caveats

- `diagnostics.depth.build` is fixed when the Session is built. A run that reused the pipeline
  therefore repeats the build of the run that built it; `pipelineLoadSeconds: null` tells them
  apart.
- The first run's submit / dispatch counts are counted from the Session's creation. Later runs count
  from the previous run.
- The first run's `estimateSeconds` includes creating the pipelines (shaders) on the device; compare
  the second and later runs.
- A timed run is slower (every dispatch is its own pass); compare timed records with timed records.
- `depthSha256` covers the raw f32 values, not the PNG: the PNG depends on the coloring and the
  min-max normalization.

## The SigLIP 2 page

The distribution holds only SigLIP 2's **vision tower**. `siglip2.Pipeline.embed` turns an image into
its raw `pooler_output` vector, which is not L2-normalized; the base vector has a norm of about 13.
There is no text tower, so there are no candidate labels and no zero-shot scores. The page embeds
one or two images. With two, it also gives their **cosine similarity**. That is an image-to-image
measure, not SigLIP's text-image score: that score needs the text tower's `logit_scale` /
`logit_bias`.

The page acquires the GPU once (`acquireGpu({ subgroups: "available" })`, plus `gpuTiming: true`
while **GPU timing** is on). `fromPretrained` creates the vision Session and uploads the weights
while it builds (base 372 MB, so400m 1.71 GB). So the page builds the pipeline on the **first Run**,
not when the page opens. It keeps the pipeline for later runs with the same GPU, source, model and
preset (`canReuse` in `request.ts`). A run with a different choice first disposes the old pipeline,
then builds the new one, so at most one Session is open at a time. A new GPU (device loss, or
switching GPU timing) and leaving the page dispose it too. A build that finishes after the GPU
changed is disposed, and the run fails saying so. One run happens at a time.

| Control | What it does |
| --- | --- |
| Model | The manifest's `models` (`model` of `fromPretrained`): `base` (224×224 input, 768-dim) and `so400m` (384×384, 1152-dim). The manifest's `defaultModel` is preselected and marked "(default)". The hint shows the input size and the embedding width the model's `pipelineConfig` declares. |
| Preset | The chosen model's `quants`. Its `defaultQuant` is preselected and marked "(default)". A preset the newly chosen model lacks falls back to that model's default. |
| Source | **Cached** (default): the HF-compatible route (`/hf/`) through the hub's CacheStorage. **Direct**: `localDirectory` over plain `fetch` from `/models/`. |
| Image A | Required. Any format the browser decodes; alpha is dropped. Every image is stretched to the model's input size, with no crop or pad, so the aspect ratio is not kept (SigLIP 2's preprocessing does the same). |
| Image B | Optional. With B, the run embeds both and compares them. **Remove image B** goes back to a single image and resets the file field, so the same file can be chosen again. |
| Run / Cancel | Run builds the pipeline if needed, then embeds A (and B). Cancel stops the **fetch** at once (the signal goes to `fromPretrained`'s manifest and weight fetches). The **Session build** (reading the cached parts and uploading the weights) and **each embedding** take no signal: Cancel is disabled during the Session build and during the last embedding, with the reason under it. During Image A of two, a Cancel takes effect when that embedding ends: Image B is skipped, the run ends as cancelled with no result, and the pipeline is kept. |
| GPU timing | Off by default; the switch sits in the status card. On: the GPU is acquired again with `gpuTiming: true`, so the next run rebuilds the pipeline, and every embedding records its GPU time per kernel. A timed run is slower; compare timed records only with timed records. Disabled while running and when the adapter has no `timestamp-query`. |

The status card shows the held pipeline (model · preset · source, or "none (built on Run)") and the
sha256 of `karume.json`. While running, it shows the stage:

- "releasing the previous pipeline"
- "fetching the weights": the file being read with its own MiB, then the load's total MiB
  (`fileLoaded / fileTotal` · `loaded / total`)
- "building the Session": once every file is fetched; not cancellable. No progress streams during
  the upload.
- "embedding Image A / B", with a counter; the last embedding is not cancellable
- "hashing"

The result card shows the cosine (with two images) and a table with one column per image: input
(name and size), dimension, non-finite count, L2 norm, min, max, mean, the first 8 values, the
sha256 of the embedding's f32 bytes and the sha256 of the input RGB8 pixels. Below the table come
the buttons, then a summary: adapter, source, model, preset, build seconds, embed seconds per image,
wall seconds, the vision Session's runs, weights, this run's submits and dispatches (with the
Session's totals), geometry profile, build cost, and GPU time per op when timed.

### The record (`format: "karume-playground-siglip2/1"`)

| Field | What |
| --- | --- |
| `format`, `page` | `karume-playground-siglip2/1`, `siglip2` |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment(gpu)`, as on the other pages |
| `distribution` | `directory` and `manifestSha256` (the sha256 of `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model` | `name`, `isDefault`, and the `imageWidth` / `imageHeight` / `hiddenDim` its `pipelineConfig` declares |
| `preset` | `quant`, `label`, `isDefault` |
| `request.images` | per image in order (A, then B): `slot`, `name`, `width`, `height`, `rgbSha256` (the sha256 of the RGB8 pixels the pipeline was given) |
| `timings.pipelineReused` | true when an earlier run's pipeline (and Session) was used as is |
| `timings.buildSeconds` | `fromPretrained`: manifest, weight download or cache read, Session build and weight upload; `null` when reused |
| `timings.embedSeconds` | one per image: preprocessing, one vision run and the readback (the first embedding of a fresh pipeline also creates the kernels) |
| `timings.wallSeconds` | from Run to the last embedding (hashing excluded) |
| `output.embeddings` | per image: `slot`, `dimension`, `nonFinite`, `first` (8 values at 6 decimals; a non-finite value is the string `"NaN"`, `"Infinity"` or `"-Infinity"`, since JSON has no such numbers), `summary` (`l2Norm`, `min`, `max`, `mean`; absent when there are non-finite values) and `sha256` of the raw f32 bytes (little-endian) |
| `output.comparison` | two images only: `{ "cosineSimilarity" }` (6 decimals, accumulated in f64), or `{ "cosineSimilarity": null, "reason" }` when it cannot be taken (non-finite values, a zero vector) |
| `diagnostics.vision` | the vision Session over this run's embeddings, one report per `embed`: `runs`, `weightsAllocatedGiB`, `submitCount` / `dispatchCount` (**this run's** share: the latest report minus the Session's counters when the run started, so fresh and reused runs compare), `sessionSubmitCount` / `sessionDispatchCount` (the Session's totals since it was built), `geometryProfile`, `build` (`buildStats`; from the run that built the Session when `pipelineReused`), `lastRunPipelines`, and `timing` with GPU timing on (as on the Wan2.2 page) |

**Download record (JSON)** saves `siglip2_<UTC timestamp>.json`. **Download Image A / B embedding
(raw f32)** saves `siglip2_<UTC timestamp>_<a|b>.f32`: the same bytes the sha256 covers, for a
value-level diff when two machines' hashes differ. **Copy record** copies the JSON.

### Caveats

- The vector is `pooler_output` itself. Normalize it yourself for cosine search; keep it raw as
  linear-probe input.
- The embedding sha256 compares runs exactly. Two GPUs, drivers or browsers can differ in the last
  bits while the cosine and the norm agree to many digits. When the hashes differ, compare the
  downloaded `.f32` files.
- A canvas keeps pixels premultiplied, so a fully transparent pixel reaches the pipeline as black.
  `rgbSha256` names exactly what was embedded.
- The per-run counts assume that the runtime's submit and dispatch counters start at 0 when the
  Session is built and only grow. That is implementation behavior, not a documented guarantee. If
  the counters ever go down, the page fails the record loudly instead of writing a negative count.
- The "building the Session" stage is inferred from the hub's progress: `loaded` reaches `total`
  once every file of the load is fetched. The upload itself streams no progress.

## The Irodori-TTS page

The page speaks Japanese text with `irodori.Pipeline` on the page's GPU. It reads one of two
distributions: v4.1-small (default; upstream retrained only the duration predictor) or v4-small. It
acquires the GPU once (`acquireGpu({})`, plus `gpuTiming: true` while **GPU timing** is on). The
first **Run** builds the pipeline. Later runs reuse it until the model, preset, linear compute,
source or GPU changes. A new GPU (switching GPU timing, a device loss) and leaving the page dispose
it at once; a build that finishes after the GPU changed is disposed, and the run fails saying so. The status card's **Pipeline** row compares the open pipeline with the current choices
on every render: "the next Run reuses it", or "the next Run rebuilds it (… changed)". Building opens
no Session: every stage opens and closes its own Session during a generation, so the held pipeline
keeps no weights on the GPU. One synthesis runs at a time.

| Control | What it does |
| --- | --- |
| Model | The mount: `irodori-v4.1` (v4.1-small, default) or `irodori-v4` (v4-small). The hint names its directory, with "(missing)" when it is absent. A switch keeps the chosen preset when the other manifest has it (both declare the same five), and keeps the decoded reference audio while the new manifest loads. |
| Preset | The manifest's `quants`; `defaultQuant` is preselected and marked "(default)". |
| Source | **Cached** (default): `/hf/` through the hub's CacheStorage. **Direct**: `localDirectory` over plain fetch from `/models/`. |
| Linear compute | `linearCompute` of `fromPretrained`, applied to the DiT and its context projection only. Choices: the preset's declaration (default), `f32`, or `a8`. The hint names the preset's DiT weights label, its declaration and the value the DiT is asked for. `a8` runs only on int8 / int4 weights: with weights labelled `f32` / `f16` the page shows a note, because the runtime sends float weights to the f32 path without an error. `f16` is not offered: it needs a device with `shaderF16: true`. |
| Text | The Japanese text, sent as typed (the pipeline normalizes it). It starts with the first voice-design example's text (`src/models/irodori/samples.ts`). Only the first `maxTextLen − 1` tokens after normalization are read (255 on the small distributions). The pipeline cuts the rest without an error (upstream behaviour). |
| Voice | Two exclusive tabs; only the active tab is sent. **Voice design**: a Japanese caption (`caption`) plus four example buttons, each filling the caption and a text that suits it. An empty caption means no voice design. The caption is cut at `maxCaptionLen − 1` tokens the same way. **Voice clone**: a reference audio file (`speaker.audio`), decoded to the distribution's `sampleRate` (below). The panel stays mounted, so the file input keeps its file across tab switches. The hint gives the minimum length and the first N seconds the model reads, both from `pipelineConfig`. |
| Seed | `seed` (default 0). |
| Duration (s) | `durationSeconds`. Empty = the duration predictor decides. A number skips the predictor, and the pipeline clamps it to the distribution's `minSeconds`–`maxSeconds`. |
| Codec tile (latent frames) | `codecTileFrames`. Empty = the pipeline's default. It only trades memory; the waveform does not change. |
| Steps | Read-only: `pipelineConfig.steps` (40). The API has no step knob. |
| Stage events | On (default): subscribes to `onEvent`. This gives per-stage seconds, the step counter, and **Cancel** at the next stage or step. The DiT loop then runs on the host path (one readback per step); the waveform is bit-identical. Off: no `onEvent`, so there are no stage seconds, and once synthesis starts the run cannot be cancelled. The status card's Cancel is then disabled, with the reason under it. The DiT loop is resident (one GPU batch, the fastest path) unless GPU timing is on, which forces the host loop. |
| Run / Cancel | Run one synthesis. Cancel stops a pipeline build at once, and a synthesis at its next stage or step (only with stage events on). A Cancel pressed while the finished run's record is being assembled still ends the run as cancelled. |
| GPU timing | As on the Wan2.2 page: times every kernel of every stage. A timed run uses the host DiT loop and is slower. Switching acquires the GPU again, so the next run rebuilds the pipeline. Disabled while running and when the adapter has no `timestamp-query`. |

**The reference audio.** The pipeline has no resampler, so the page decodes the file to the
distribution's `sampleRate` (48 kHz):

- A PCM-16 or float-32 WAV already at that rate goes through `decodeWav` of the models barrel
  (loaded on the first file). The samples are the file's own, so they compare across machines and
  browsers.
- Any other file (another rate, another format) is decoded by the browser's `decodeAudioData` on an
  `AudioContext` at that rate, which resamples it. The channels are averaged with `decodeWav`'s
  arithmetic.

The page shows the decoder (and why the browser was used), the length, the sha256 of the samples,
and a player with Download WAV for the decoded audio.

The result card plays the audio (AudioClip: player and **Download WAV**). It also shows a summary of
the record:

- adapter, source, model, preset, and linear compute (choice → resolved value)
- the DiT's linear dispatches (i8a8 / other), with a note when `a8` was asked for but no linear ran
  int8
- voice, and seed / duration / steps
- the DiT loop, and whether the pipeline was built (and how long it took) or reused
- length, samples, latent frames S and forwards
- generate and wall seconds, and the real-time factor
- per-stage seconds, the first denoise step, and the min / median / max of the rest
- the f32 sha256
- per component: the runs, weight GiB and build cost; with GPU timing on, also the five most
  expensive ops

### The record (`format: "karume-playground-irodori/1"`)

| Field | What |
| --- | --- |
| `format`, `page` | `karume-playground-irodori/1`, `irodori` |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment(gpu)` |
| `distribution` | `mount`, `directory`, `manifestSha256` (sha256 of the mount's `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model`, `preset`, `linearCompute` | the manifest model (`v4.1-small` / `v4-small`); `{ quant, label, isDefault }`; the page's choice `preset` / `f32` / `a8` |
| `linearComputeResolved` | what the DiT Sessions were asked for: the choice, else the preset's `session.linearCompute`, else `f32` |
| `ditLinearDispatches` | `{ i8a8, other }`: planned dispatches of linear kernels in the `dit` component's last run, counted from its pipeline keys (a runtime diagnostic label, outside semver). `null` when no `dit` diagnostics came. `a8` with `i8a8: 0` = float weights on the f32 path. |
| `request` | `text` (as typed, before truncation); `voice` = `{ kind: "caption", caption }` (`""` = no voice design) or `{ kind: "reference", file, decoder, sampleRate, samples, seconds, f32Sha256 }`; `seed`; `durationSeconds` (a number or `"predicted"`); `codecTileFrames` (a number or `"default"`); `steps` (from `pipelineConfig`) |
| `ditLoop`, `stageEvents` | `host` (stage events or GPU timing on) or `resident`; whether `onEvent` was subscribed |
| `pipeline` | `{ built: true, loadSeconds }` when this run built it, `{ built: false }` when it reused the open one |
| `stageSeconds` | seconds per stage (`backbone`, `text-proj`, `caption-proj`, `codec-encoder`, `speaker`, `duration`, `dit-context`, `dit`, `codec-decoder`), from each stage's start event (before its Session) to its end event (after release). Empty with stage events off. |
| `timeline.denoiseStepSeconds` | seconds of every DiT step. The first is measured from the `dit` stage start, so it also holds the Session creation and the weight upload. |
| `generateSeconds`, `wallSeconds` | the `generate` call alone; from Run to the end of `generate` (the pipeline build included when this run built it) |
| `output` | `sampleRate`, `samples`, `seconds`, `latentFrames` (S), `forwards`, `seed`, `realTimeFactor` (`generateSeconds / seconds`; below 1 = faster than real time), `f32Sha256` (sha256 of the output samples as little-endian f32 bytes) |
| `diagnostics.<component>` | as on the Wan2.2 page: runs, weight GiB, submit / dispatch counts, geometry profile, `build`, `lastRunPipelines`, and `timing` with GPU timing on |

### Caveats

- Seconds from runs with stage events on, off, or with GPU timing on are not comparable with each
  other (host vs resident DiT loop). Compare records with the same `ditLoop` and
  `device.gpuTimingEnabled`. The waveform, and so `output.f32Sha256`, is the same on both loops.
- A reference that went through the browser decoder depends on the browser's resampler. Compare
  `request.voice.f32Sha256` before comparing outputs across machines.
- Text past `maxTextLen − 1` tokens (and a caption past `maxCaptionLen − 1`) is cut without warning.
  A record of a long input keeps the full text, so its `request.text` can say more than the audio
  does.
- The page reads only these `pipelineConfig` numbers: `sampleRate`, `maxTextLen`, `maxCaptionLen`,
  `hopLength`, `speakerPatchSize`, `speakerRows`, `codecHaloFrames`, `minSeconds`, `maxSeconds` and
  `steps`. They are used for decoding and hints. The pipeline still checks every request (seed
  range, codec tile, reference rate and length) and throws a `ModelInputError`, shown in full.

**Download record (JSON)** saves `irodori_<UTC timestamp>.json`. **Download WAV** saves
`irodori_<UTC timestamp>.wav`: 16-bit PCM from `encodeWav`, so its bytes are not the f32 samples the
sha256 covers. **Copy record** copies the JSON.

## The Style-Bert-VITS2 page

The page acquires the GPU once (`acquireGpu({ subgroups: "available" })`, plus `gpuTiming: true`
while **GPU timing** is on). It builds `sbv2.Pipeline.fromPretrained` for the chosen source, model
and preset. Building reads the manifest, the three graphs' descriptors and the small assets. With
the **Cached** source it also downloads every weight part of the model / preset into Cache Storage,
so the first build of a model / preset fetches all of its weights and later builds find them there.
The **Direct** source has no such step. Building creates no Session. Every **Run** creates the three
graphs' Sessions one after another (`text_encoder` → `front` → `voice`), reads their weights from
the source (the cache the build filled, or plain fetch), and destroys them again. One synthesis runs
at a time.

Text analysis is the caller's job: since 0.6.0, `generate` takes an analysed `sbv2.Utterance`, so the
page does the analysis itself. `@hdae/yomi` (`analyzeWithWords`) turns the text into accent phrases
and the word alignment, and `sbv2.toUtterance` lowers them to the utterance. This is the same
analyser the SBV2 WAV regression test uses. The analyser is a caller-side dependency of this page,
not a workspace package. The analyser's dictionary (JTD1, about 6.4 MB gzip) is **not in a
distribution**. It is fetched from the analyser's Hugging Face dataset at the revision its version
pins, on first use only, and kept in this origin's Cache Storage (`yomi-dict`). The page starts
loading it when it opens.

Every list comes from the distribution:
- The speakers are the manifest's models (a jvnv model holds one speaker).
- The styles and the starting values come from the model's `pipelineConfig` (`styles`, `speakers`,
  `defaults`).
- The input limits come from `maxTokens` and `maxFrames`.

| Control | What it does |
| --- | --- |
| Speaker | The manifest's models with the speakers each holds (`F1 — jvnv-F1-jp` …). `defaultModel` is preselected and marked "(default)". Changing it rebuilds the pipeline and switches to that model's speaker; the style stays when the model has it. A **Speaker name** select appears only for a model with more than one speaker. |
| Preset | The model's `quants`. `defaultQuant` is preselected and marked. Changing it rebuilds the pipeline. |
| Source | **Cached** (default: `/hf/` through the hub's CacheStorage; the build downloads the weights) or **Direct** (`/models/sbv2/` over plain fetch, read again on every run). Changing it rebuilds the pipeline. |
| Sample text | Fills the text with the regression-test sentence (default), an expressive one or a longer narration (`samples.ts`). |
| Text (Japanese) | The text to synthesize. The hint gives its length and both limits of one synthesis: `maxTokens` (tokens and phonemes) and `maxFrames` (total predicted frames, which grow with the length scale). The pipeline rejects a request over either limit. |
| Reading | Live: the moras the analyser reads, with high tones overlined, and the mora count. |
| Style | `pipelineConfig.styles` in table-row order; the default is marked. |
| Style weight / Length scale / SDP ratio / Noise scale / Noise scale W | Sliders for `styleWeight` (0–2), `lengthScale` (0.5–2, larger is slower), `sdpRatio` (0–1), and `noiseScale` and `noiseScaleW` (0–1.5). They start at `pipelineConfig.defaults`. |
| Seed | `seed` (default 0). The page refuses a seed that is not a number. An out-of-range seed is rejected by the pipeline with `sbv2.InputError`. |
| Run | Analyses the text and synthesizes it. **Not cancellable**: `sbv2.Pipeline.generate` takes no AbortSignal, so the status card's Cancel stays disabled, with the reason under it. |
| Reset options to the distribution's defaults | Sets speaker, style, every knob and the seed back to `pipelineConfig.defaults` / 0. The text stays. |
| GPU timing | Off by default, in the status card (as on the Wan2.2 page). Turning it on acquires the GPU again with `gpuTiming: true`, and every graph records its GPU time per kernel. Timed runs are slower, so compare timed records only with timed records. The switch is disabled while a run is in flight. Switching while the pipeline is being built is fine: the abandoned build is disposed before the old device is destroyed. |

Every option is passed to `generate` explicitly, the defaults too, so the record is the full
request. The page refuses only an empty text and a seed that is not a number. Acceptance belongs to
the pipeline. An input it refuses shows as **Input rejected — sbv2.InputError**, with the message,
the stack and a hint that quotes both limits. A frame-limit rejection comes only after the
`text_encoder` and `front` graphs have run. Any other failure shows in full as **Run failed**.

While running, the status card shows the elapsed seconds and the graph being run (`n / 3 graphs`,
advanced by `onRunDiagnostics`). While the pipeline is being built, the **Pipeline** row shows
`fetching x / y MiB in total · <latest file>`. The result card plays the clip (**Download WAV**) and
summarizes the record:
- the adapter, source, model, preset, speaker, style × weight, knobs and reading
- the duration and the real-time factor
- analysis, per-stage, synthesis and wall seconds
- the sha256 of the f32 samples and of the WAV
- per graph: the weight memory, submits, dispatches and build cost (plus GPU time when timed)

### The record (`format: "karume-playground-sbv2/1"`)

`buildSbv2Record` in `src/models/sbv2/record.ts` assembles the record (CPU-tested).

| Field | What |
| --- | --- |
| `format`, `page` | `karume-playground-sbv2/1`, `sbv2` |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment(gpu)` (as on the other pages) |
| `distribution` | `directory` and `manifestSha256` (sha256 of `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `model`, `preset` | the manifest's model; `{ quant, label, isDefault }` (`isDefault` = the model's `defaultQuant`) |
| `request` | `{ text, options }`: `speaker`, `style`, `styleWeight`, `sdpRatio`, `noiseScale`, `noiseScaleW`, `lengthScale`, `seed`, exactly as passed to `generate` |
| `analysis` | `analyzer` / `analyzerVersion` (`@hdae/yomi` and its version, which pins the dictionary revision); `reading` (moras, high tones in brackets); `utterance` (the exact `sbv2.Utterance` that `generate` received) |
| `timings.pipelineLoadSeconds` | building the pipeline: manifest, descriptors, small assets and, with Cached, the weight download into Cache Storage (no Session) |
| `timings.analysisSeconds` | `analyzeWithWords` + `sbv2.toUtterance` on the host |
| `timings.stageSeconds` | Seconds per graph. Each graph is measured up to its diagnostics report, which comes after its run and before its Session is destroyed, starting from the previous graph's report (the first from the `generate` call). So a stage holds the previous graph's Session destruction, the host glue, its own Session creation (reading and uploading the weights) and its run. |
| `timings.synthesisSeconds`, `timings.wallSeconds` | the whole `generate` call, which also holds the last Session's destruction and the output check; analysis + synthesis + hashing |
| `output` | `sampleRate`, `samples`, `durationSeconds`; `realTimeFactor` (synthesis seconds / audio seconds; below 1 = faster than real time); `f32Sha256` (sha256 of the f32 samples' bytes as they sit in memory, little-endian in practice); `wavSha256` (sha256 of the 16-bit WAV that `encodeWav` makes, the hash the SBV2 regression test pins in `packages/models/tests/fixtures/references/sbv2.json`) |
| `diagnostics.<graph>` | per `text_encoder` / `front` / `voice`: the diagnostics summary, `build` (`buildStats`), `lastRunPipelines`, and with GPU timing `timing` (only for graphs that were timed, as on the Wan2.2 page) |

**Download record (JSON)** saves `sbv2_<UTC timestamp>.json`, **Download WAV** saves `sbv2_<UTC timestamp>.wav`,
and **Copy record** copies the JSON.

### Caveats

- **Not cancellable.** Once started, a synthesis runs to its end (`generate` has no AbortSignal).
  The speaker, preset, source and GPU timing controls are disabled while it runs. Leaving the page
  during a run drops its result; the GPU is destroyed only after the synthesis has finished.
- **The analyser decides the input.** The same text can give a different utterance with another
  `@hdae/yomi` version (another dictionary revision). When two runs' hashes differ, compare
  `analysis.utterance` first. To fix a wrong reading, write it in kana in the text.
- **The dictionary needs the network once** (Hugging Face, with CORS). Later visits read it from
  Cache Storage. Its download takes no AbortSignal.
- **The first Cached build downloads all weights** of the model / preset, and `pipelineLoadSeconds`
  includes that download. Later builds and every run read from Cache Storage. **Direct** downloads
  nothing at build and reads the weights over plain fetch on every run.
  `diagnostics.<graph>.build.supplyWaitMs` is where the per-run weight read shows. Which source is
  faster is not fixed: compare the two records.
- **Hashes compare one environment.** `f32Sha256` / `wavSha256` are bit-exact only on the same GPU,
  driver and browser back end. The regression test's reference rows are per environment key (Linux /
  Vulkan for the RTX 3080 Ti row), so a Chrome D3D12 run is not expected to match them. Compare it
  with other runs on the same machine.
- The regression test's request is model F1, preset `i8`, the regression-test sample text, seed 0
  and the distribution's defaults.

## The Gemma 4 chat page

A multi-turn chat with regular Gemma 4 (`gemma.Pipeline`, mount `gemma4`), its q4_0 QAT line
(`gemma.Pipeline`, mount `gemma4-qat`: E2B / E4B / 12B with their QAT MTP drafters) or a mobile QAT
model (`gemma4QatMobile.Pipeline`, mount `gemma4-qat-mobile`: E2B / E4B, experimental). The page
acquires the GPU once (`acquireGpu({ subgroups: "available" })`). **Load** builds the pipeline:
`fromPretrained` reads every weight and builds the Session (gigabytes), so nothing is read until
Load is pressed. The conversation is one `gemma.ChatSession`. It carries the KV between turns that
close with the distribution's EOS. After any other stop it re-renders the history on the next send.
One turn runs at a time.

A model whose manifest has the `vision` weights (12B of the q4_0 QAT line) loads with `vision: true`,
and only then the composer shows **Attach images**. The page decodes a file itself
(`createImageBitmap` without color-space conversion, `OffscreenCanvas`, the alpha dropped) to the RGB8
the session takes. The browser applies a JPEG's EXIF orientation, and a canvas stores premultiplied alpha, so a
transparent pixel arrives black where Pillow's `convert("RGB")` would keep its stored color. An
attachment can be removed before sending, and a user turn shows its images as thumbnails. The session
embeds each image once and keeps its rows for a re-render (ADR 0144). The record states each image's
size and soft tokens (`turns[].images`), never its pixels.

| Control | What it does |
| --- | --- |
| Model | Every model of the three manifests (`Gemma 4 — e2b` / `e4b`, `Gemma 4 QAT (q4_0) — e2b` / `e4b` / `12b`, `Gemma 4 mobile QAT — e2b` / `e4b`), each manifest's `defaultModel` marked "(default)". Applies on the next Load. |
| Preset | The model's `quants`; `defaultQuant` is preselected and marked "(default)". Applies on the next Load. |
| Source | **Cached** (the hub's cache over `/hf/`) or **Direct** (`/models/`) — Direct only for the two QAT mounts, which are not on `/hf/`. Applies on the next Load. |
| MTP drafter | A model of the regular entry point with drafter weights only (regular E2B, every model of the q4_0 QAT line): builds the drafter (`speculative: {}`). The session then speculates with the self-tuning gate. Off by default. Applies on the next Load. |
| f16 KV (full layers) | Passes `fullStateStorage: "f16"` to `fromPretrained`: the full-layer KV in half the memory, counted by **Fit** (ADR 0143). It changes the numbers slightly. With the MTP drafter the session passes `speculative: "always"`, because the self-tuning gate is refused under it. Off by default. Applies on the next Load. |
| Collect Session diagnostics | Passes `onRunDiagnostics` to `fromPretrained`. The record then gets per-run counts and the Session diagnostics. **It changes the decode path**: without it the pipeline picks greedy (temperature 0) tokens on the GPU (an 8-byte readback per token). With it every greedy token reads the whole logits row back and is picked on the CPU. Off by default; leave it off to measure decode rates. Applies on the next Load. |
| Load / Load again | Builds the pipeline for the selection and starts a new chat. A note says when the selection differs from the loaded pipeline. |
| Cancel load | Shown while loading, in place of Load. Aborts the download (the `signal` of `fromPretrained`). Building the Session after the download cannot be interrupted: a pipeline that finishes after Cancel is released at once. |
| System prompt | The session's system message (empty = none). Applies on New chat. |
| Capacity | KV positions the chat allocates (`capacity` of the session). Empty = the distribution's (4096). It must be from `chunkLength` to `maxPosition`. Applies on New chat. |
| When a turn does not fit | **Drop the oldest turns** (`dropOldestTurns`: the oldest user / assistant pair goes, the system message stays; checked before sending), or **Refuse the turn**: the policy returns the history unchanged, so the session fails the turn with `GenerationCapacityError` and the history stays as it was. Applies on New chat. |
| New chat | Disposes the session and builds a new one with the chat settings (the record starts over). |
| Temperature, Top-k, Top-p, Seed, Max new tokens | The turn's sampler and `maxNewTokens`, passed with every send. The form starts from the distribution's sampler (temperature 1, top-k 64, top-p 0.95), seed 0, 512 tokens. Empty top-k / top-p = no narrowing. Empty max new tokens = no limit: the turn runs until the model stops or the context is full (stop reason `capacity`), and the session's default `reserveNewTokens` (1,024) decides whether old turns are dropped first. **Distribution's sampler** resets the form; **Greedy** sets temperature 0. |
| Sample messages | Fill the input (one is Japanese, to exercise multi-byte streaming). |
| Send / Stop | Enter sends (Shift+Enter is a new line). Stop aborts the turn through the request's `signal`. A partial reply stays in the history and the turn shows "stopped". The status card's Cancel does the same. |

The transcript is rebuilt from the session's own history (`session.turns`) after every turn, so the
bubbles always show what the session will send next. Entries that `dropOldestTurns` removed stay
visible, dimmed, marked "dropped from the context". A turn the session does not keep is one that
produced no text and did not close with EOS: a Stop during prefill, or a failure. It leaves no
entry. The page says so in a notice and puts the message back into the input. Each assistant
bubble's footer shows how the turn stopped, the generated tokens, the prefill tokens with their
rate, the time to first token (TTFT), the decode tokens/s and the total seconds. The status card
shows the loaded pipeline, the chat settings and the **context**:

- after a turn that closed with EOS: the positions the session's KV holds, over the capacity;
- after any other stop: the token count of the history the session will re-render on the next send.

While a turn streams, the reply is drawn at most once per animation frame, and the status card's
counter updates every 250 ms. Page re-renders run on the thread that runs the decode loop, so they
are kept out of the window being measured.

How the figures are measured (from `performance.now()` in the page):

- **Prefill tokens**: the prompt the session passed to the sequence. That is the delta when the KV
  is carried, the whole history otherwise. A carried turn also prefills the previous turn's closing
  token (the sequence's pending token), which is not counted.
- **Prefill seconds**: from `send` to the last committed chunk (`onPrefill`). It includes the
  overflow check and, on a fresh sequence, its allocation.
- **TTFT**: from `send` to the first generated token (`onToken`).
- **Decode tokens/s**: (`tokens` − 1) / (decode end − first token). The first token is sampled from
  the prefill's logits. The decode end depends on the diagnostics switch:
  - with diagnostics: the last decode / verify / draft run's `onRunDiagnostics`, which includes the
    run that drew the stop token;
  - without: the end of the text stream.

### The record (`format: "karume-playground-gemma4/4"`)

| Field | What |
| --- | --- |
| `format`, `page` | `karume-playground-gemma4/4`, `gemma4` (version 3 always had `request.maxNewTokens` and never the stop reason `capacity`; version 2 recorded the mobile QAT line as mount `gemma4-qat` with pipeline `gemma4Qat.Pipeline` — since version 3 that mount is the q4_0 QAT line and the mobile line is `gemma4-qat-mobile` / `gemma4QatMobile.Pipeline`; version 1 spelled `pipeline` with the 0.14 class names `Gemma4Pipeline` / `Gemma4QatPipeline`) |
| `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device` | `recordEnvironment` |
| `distribution` | `mount`, `directory`, `manifestSha256` (sha256 of `karume.json`) |
| `source` | `{ "kind": "direct" }` or `{ "kind": "cached", "repo", "revision", "hubUrl" }` |
| `pipeline` | `gemma.Pipeline` or `gemma4QatMobile.Pipeline` |
| `model`, `preset` | the model name; `{ quant, label, isDefault }` |
| `speculative` | whether the MTP drafter was built |
| `fullStateStorage` | `"f16"` when the page's f16 KV switch was on, absent otherwise |
| `vision` | `true` when the image embedder was built, absent otherwise |
| `loadSeconds` | from Load to the pipeline being ready |
| `diagnosticsCollected` | whether the pipeline was loaded with `onRunDiagnostics`. When true, greedy turns ran without the on-GPU greedy selection, so their decode rates are not those of an app without diagnostics. |
| `chat` | `system` (absent = none), `capacity`, `overflow` (`drop-oldest` / `refuse`) |
| `turns[].index`, `userText`, `replyText` | the turn's number (failed and not-kept turns included), the message, the streamed reply |
| `turns[].images` | on a message with images: `width`, `height` and `softTokens` of each, in the order sent |
| `turns[].request` | `maxNewTokens` (absent on a turn without a limit) and the `sampler` exactly as passed |
| `turns[].outcome` | `{ kind: "stopped", stop: { reason, tokens, token?, stopString?, speculation? } }` or `{ kind: "failed", error }` |
| `turns[].prefill` | `promptTokens`, `carriedPositions` (positions the carried KV already held; 0 = rendered from the start), `chunks` |
| `turns[].timing` | `totalSeconds`, `prefillSeconds`, `prefillTokensPerSecond`, `ttftSeconds`, `decodeTokens`, `decodeSeconds`, `decodeTokensPerSecond` (a figure that could not be measured is absent) |
| `turns[].generatedTokenIds` | every generated token id except the stop token (`outcome.stop.token` holds that). Compare these across machines (temperature 0 or the same seed). |
| `turns[].droppedEntries` | history entries the overflow policy dropped before the turn was sent |
| `turns[].inHistory` | whether the session kept the turn (its message and reply) in its history |
| `turns[].kvUsedAfter` | positions the session holds after the turn; absent when it released the sequence |
| `turns[].runs` | with diagnostics only: runs reported by `onRunDiagnostics` during the turn (`prefill`, `decode`, `draft`, `verify`) |
| `diagnostics.target`, `diagnostics.drafter` | with diagnostics only: the Session diagnostics folded over the chat's runs, as on the Wan2.2 page (runs, weight GiB, submit / dispatch counts, geometry profile, `build` cost, `lastRunPipelines`). `drafter` is present only with the MTP drafter. |

**Download record (JSON)** saves `gemma4_<UTC timestamp>.json`; **Copy record** copies the same
JSON.

### Caveats

- Decode rates are comparable only between records with the same `diagnosticsCollected`. For
  production-like rates, leave diagnostics off. Speculative turns (MTP) do not use the GPU greedy
  path either way.
- No GPU timing switch. On a timing device every dispatch becomes its own pass, which would distort
  the decode rate this page exists to show. GPU-resident PLE also refuses timing. The record carries
  no per-kernel times.
- Only one turn can run at a time. Changing the model, preset, source, MTP or diagnostics needs
  Load. The system prompt, capacity and overflow policy need New chat. Both clear the chat.
- Load is not a `useRunTask` run (the status card's run is the chat turn). It has its own Cancel
  load button.
- A failed release of the previous pipeline or session is shown as an error on the page and logged
  to the console.
- The mobile QAT entry point is experimental (ADR 0097): fixed int2/int4/int8 with SRQ, no MTP.
- After a device loss, **Acquire the GPU again** rebuilds the pipeline for the same Load and starts
  an empty chat.

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

## The Gemma 4 decode bench page

A decode A/B of regular Gemma 4 and its q4_0 QAT line (`gemma.Pipeline`, mounts `gemma4` and
`gemma4-qat`) and the mobile QAT distribution (`gemma4QatMobile.Pipeline`, mount
`gemma4-qat-mobile`), kept as a page so each measurement wave reuses it
(ADR 0127 decision 5). An **arm** is a model of any of the three distributions at one preset, with the GPU tap
on or off. The page builds each arm's pipeline the way the chat page does (`buildGemmaPipeline`),
always over the **direct route** (the mounts also have a cached route, which the bench does not
offer): one route for every arm keeps the source out of the comparison.

**The GPU tap** (`src/models/gemma4-bench/tap.ts`). The page acquires its GPU through a wrapper of
`navigator.gpu` (`acquireGpu({ subgroups: "available", gpu: tap.gpu })`). The wrapper requests
`timestamp-query` when the adapter lists it and hides it from `device.features`, so the runtime sees
a device without timing (`gpuTimingEnabled` false) and keeps its own pass structure (one chunk = one
compute pass). With the tap on, every compute pass gets a pair of pass-boundary timestamps, resolved
in the same command buffer before it is finished, and passes, dispatches and submits are counted. karume's own timing mode is not
used: it encodes one dispatch per pass (ADR 0021), about 4 µs more per dispatch, and QAT runs about
210 more dispatches per token, so it would bias this very A/B. With the tap off, the same device's
calls pass straight through (a flag check per command encoder and per submit): a wall-clock control
that shows what the tap costs. The page refuses to run if the device reports `gpuTimingEnabled`.

**Split dispatches** (per arm, with the tap on; the arm's label shows "(split)") is the attribution
view. The tap ends the real compute pass before every dispatch after the first and begins a new
timed one on the same encoder, replaying the pipeline, the bind groups (with their dynamic offsets)
and any open debug groups, so every dispatch gets its own pass-boundary timestamps and the label of
its pipeline — the runtime's pipeline cache key (`linear_gemv_parallel:wi4g32:l32`,
`attention_state_qk:v1:f32:wg16x16:par:gqa`, …). Each chat's decode window is then broken down per
key (dispatches and GPU ms per decode token), and each key is put in an op family by its kernel
name (`family.ts`): attention core (every `attention*` kernel: QK, row statistics, PV, fused
statistics + PV), attention other (`state_append`, `rope`, `strided` copies), linear (`linear` and
every `linear_*` kernel, GEMVs that fuse their input's static quantize included), norm (every
`rms_norm*` kernel), elementwise (`ew:*`), static quantize (`static_quantize`,
`gelu_mul_static_quantize`), embedding (`embedding`, `gather`) and other. The research's PLE and
lm_head rows depend on a kernel's position in the run, which a key does not carry: their kernels
count in their kernel's family here. Every extra pass boundary costs GPU time of its own (about 4 µs
per dispatch on the RTX 3080 Ti under Dawn, research 2026-10-27 §3), which inflates small kernels
most and the arm's absolute GPU ms/token, wall − GPU and decode rate as well: **read the per-family
and per-key shares**, compare them across arms split alike, and take absolute times from an unsplit
arm. The tap times up to 262,144 passes per chat (mobile QAT E2B runs about 1,036 dispatches per decode
token, so 128 new tokens plus the prefill fit); later passes are counted but untimed, and a window
with an untimed pass reports no GPU time. Its timestamp buffers take about 8 MiB of GPU memory at
most (4 MiB of packed results plus the read-back copy).

**Integer GEMV** (per arm; the arm's label shows "(integer)") passes the Session option
`integerGemv: true` to `fromPretrained` as an explicit option, so the decode linears of a mobile QAT
model at the integer GEMV's measured shapes (mobile QAT E2B: the i2 gate / up and down, the i4 o and the
own-KV layers' down — ADR 0140) run on the integer GEMV (the runtime's
`src/kernels/linear-gemv-integer.ts`) instead of the parallel GEMV. The option needs packed int8
activations, which need the parallel GEMV: the arm also passes `packedStaticQuantize: true` and
`linearGemvReduce: "parallel"` where its preset does not declare them (mobile QAT `i4-fast` declares both,
so the arm passes `integerGemv` alone). Explicit options win over the preset's declaration. Regular
Gemma 4 has no linear at those shapes, so there the switch changes only what the packed activations
change.

**Planes GEMV** (per arm; the arm's label shows "(planes)") passes the Session option
`linearGemvPlanes: true` to `fromPretrained` as an explicit option, so the i2 linears of a mobile QAT model
at the byte-plane GEMV's measured shapes (mobile QAT E2B: the lm_head, 262144 × 1536) run on the byte-plane
GEMV (the runtime's `src/kernels/linear-gemv-planes.ts`) instead of the sequential GEMV. The option
needs no other option, so the arm passes it alone, next to what the Integer GEMV switch passes when
both are on. Its sums differ from the sequential GEMV's by the addition order only (contract class R
of ADR 0147 decision 2), so a greedy arm may decode other tokens than one without it. Regular Gemma 4 has no i2
linear at that shape, so there the switch changes nothing.

**PLE on GPU** (per arm; the arm's label shows "(PLE GPU)") passes the pipeline option
`pleResidency: "gpu"` to `fromPretrained`: the per-layer embeddings are uploaded once at load and
each run gathers its rows on the GPU instead of dequantizing them on the host (ADR 0085, GPU
residency). The `values` table goes in one binding — regular Gemma 4 `bf16` (its one preset) E2B 4.38 GiB /
E4B 5.25 GiB, the q4_0 QAT line E2B 1.09 GiB /
E4B 1.31 GiB (i4, plus a scale table of 280 / 336 MiB), mobile QAT E2B 1.09 GiB (i4) / E4B 672 MiB (i2)
— so the device must grant a `maxStorageBufferBindingSize` that large; the page acquires its GPU with the adapter's own limits (an Apple M2 offers 4 GiB), and a
device that cannot hold it fails the arm's load with the shortfall in bytes.

**Decode ahead** (per arm: 1, 2, 4 or 8; the arm's label shows "(ahead N)" for 2 and more) passes
the pipeline option `decodeAhead`: a greedy chat runs up to N decode steps in one GPU batch, each
step fed the token the previous step selected on the GPU, and the N selections are read back at the
batch's single fence — one fence, one readback and one host round trip per N tokens instead of per
token. The token ids are those of depth 1 (the same kernels run on the same inputs); a batch that
meets the end-of-turn token has already computed the steps after it, which are discarded. It needs
**PLE on GPU** (the arm reports the pair as an error otherwise), and with the distribution's sampler
the chats decode one step per batch as before. Every chat's `stop` records `decodeAhead` (the
depth, the batches it ran — 0 when it did not take the run-ahead — and the steps they computed).

| Control | What it does |
| --- | --- |
| Arms | One row per arm: Model (every model of the three manifests; a mobile QAT model `gemma4QatMobile.Pipeline` does not accept is disabled), Preset (the model's `defaultQuant` preselected), GPU tap, Split dispatches, Integer GEMV, Planes GEMV, PLE on GPU, Decode ahead. Starts with each distribution's default model at its default preset. Arm 1 is the one the summary compares the others with. |
| Prompts | Two fixed prompts, run in this order in every visit. The story request renders to 21 tokens (17 distinct ids); the GPU explainer to 60 tokens (48 distinct). A prompt with 32 or more distinct ids makes a PLE block whole-resident on the host: the condition under which the PLE row cache once stopped admitting rows (perf-ledger H-41). Every chat records the distinct count it measured, and the run table flags a prompt that falls on the wrong side. |
| Chats | Chats per prompt per visit (default 3). Chat 0 is the first after the build: recorded, and left out of the summary. |
| New tokens | `maxNewTokens` of every chat (default 128). |
| Rounds | Default 4. Even rounds visit the arms in order, odd rounds in reverse (ABBA for two arms). Every visit builds its arm's pipeline, runs its chats and releases it before the next visit, so two models are never resident at once. |
| Sampler | **Greedy** (default: every arm and round decodes the same way) or the distribution's sampler with seed 0. Always passed explicitly. The bench never passes `onRunDiagnostics`: with it every greedy token reads the whole logits row back (the chat page's "Collect Session diagnostics"), which changes the decode path. |
| Run | Runs every visit. Cancel (status card) stops at the next step boundary; the record keeps what finished. |

Per chat (one `chat` call — one sequence — from `performance.now()` in the page):

- **Decode tokens/s, ms/token**: (`tokens` − 1) over first token → end of the text stream, as on the
  chat page (the prefill and the first token are excluded; a stop token counts as one).
- **TTFT** and **prefill tokens/s** (the rendered prompt over `chat` → last committed chunk), apart.
- With the tap: **GPU ms/token** (Σ pass end − begin over the decode window ÷ decode tokens) and its
  median over the token-to-token intervals; **wall − GPU** per token; **dispatches, passes, submits
  per token**; the same for the prefill window. The decode window starts at what the tap had recorded
  when the first token arrived.
- The sha256 of the generated token ids (a greedy arm repeats it in every round).

The **summary** covers the warm chats of every round per arm and prompt: median and min (for rates,
median and max) — the development machine shows transient 5–45× slow kernels, which move a median
less than a mean and a min not at all — the ratio of each arm's median decode ms/token to arm 1's,
and how many distinct outputs the warm chats produced. For split arms, the **Split attribution**
card lists per arm and prompt the family totals and every key, sorted by GPU ms per token, each the
median over the warm chats of every round (a key missing from a chat counts 0), with its share of
the median total and its dispatches per token; `<pre id="split-summary">` holds the same as plain
text.

**Download record (JSON)** saves `gemma4-bench_<UTC timestamp>.json` (`format:
"karume-playground-gemma4-bench/2"` — version 1 recorded the mobile QAT line as mount `gemma4-qat`, which
since version 2 names the q4_0 QAT line; the mobile line is `gemma4-qat-mobile`): `recordEnvironment`, `environmentKey` (ADR 0106's
`chrome-<adapter name>` rule; absent outside Chromium), `timestampQuery`, `complete` (false when the
bench failed or was cancelled), `settings`, `arms` (mount, directory, manifest sha256, source,
pipeline, model, preset, tap, the sampler passed), `visits` (round, arm, load seconds), `chats` (every
chat, cold ones included, with its stop, prompt ids and measurement) and `summary`. Every arm records
`split`; a split arm adds `tap.perKey` (key, dispatches and GPU ms per decode token) on each of its
chats and `split` (the medians by family and by key) on its summary entries. Every arm also records
`integerGemv` and `sessionOptions` (the Session options it passed explicitly, `linearGemvPlanes`
among them; `{}` when none), and `pipelineOptions` (`pleResidency` and `decodeAhead` when the arm
passed them; `{}` when none). A chat of a run-ahead arm carries `decodeAhead` in its `stop`.
**Copy record** copies the same JSON.

Caveats:

- Chrome rounds GPU timestamps to 100 µs unless chrome://flags/#enable-webgpu-developer-features is
  on.
- The status card's elapsed-time ticker re-renders the page every 250 ms while a chat runs (the
  tables are memoized); the page's own updates land between chats.
- A tap-off arm still runs on the device with `timestamp-query` enabled (unused).
- Absolute times compare only on one machine.

## The geometry sweep page

This page measures the GEMM tile geometries on the browser's GPU and writes the record a geometry
profile is derived from. It is the successor of the sweep tab of the removed `tools/gpu-lab`. It
calls `runGeometrySweep` of `@karume/runtime/tune` **once per case**, idles between cases (cooling:
one case takes minutes, and heat moves the default geometry's re-measurement, which the derivation
checks), and merges the per-case records into **one sweep record** (`karume-geometry-sweep/2`, the
format `deriveGeometryProfile` reads, with no playground envelope). Its first job is perf-ledger
V-15: the conv3d-only sweep on the RTX 5070 Ti, from which the derivation chooses `i8a8.conv3d` of
the `nvidia-blackwell` table.

**Serve the page from the checkout that will derive the table.** The record carries `caseSet`,
`defaultKernels` and `candidateKernels`, the fingerprints of the case set and of the kernels of the
runtime the page was built from, and `deriveGeometryProfile` rejects a record whose values differ
from the deriving code. Do not edit `packages/runtime` of the serving checkout until the record has
been derived: the dev server reloads edited modules.

**The GPU.** `runGeometrySweep` acquires a dedicated device per call and destroys it before it
returns, so the page has no `useGpu`. The run joins the app's device queue with `runOnGpuQueue`
instead:

- it starts only after the previous page's device is destroyed ("waiting for the previous page's GPU
  device" in the status);
- the next acquisition of any page waits until the run has settled;
- leaving the page aborts the run at its next cancel point (below), and the queue stays held until
  the run has settled.

| Control | What it does |
| --- | --- |
| Preset | **V-15: conv3d, full, 5 rounds, 90 s idle** fills the form with every conv3d w8a8 case, the `full` set, 5 rounds and 90 s idle; while the form equals it, a badge next to the title names it. **Every case (reset)** is no preset: it goes back to the initial form (every case, `quick+`, 5 rounds, 90 s). |
| Op families | Which families are shown and swept (the runtime's `geometrySweepCases()`, grouped by op). |
| Cases | Every case of the chosen families, with **all** / **none** per family. A ticked case of a family that is not chosen is not swept. |
| Candidate set | `quick` (a few geometries — a dry run that checks the page and the device), `quick+` (quick plus the registered tables' geometries; the runtime's default), `full` (every geometry — the set a table is derived from). |
| Rounds | Measurement rounds per geometry (default 5); the min is kept. |
| Idle between cases (s) | Cooling before every case but the first (default 90; 0 = none). |
| Run / Cancel | Run sweeps the cases in the table's order. Cancel takes effect at once during the idle wait, before the next case starts, and inside a case once the current geometry is measured (`runGeometrySweep` keeps the rows measured so far, with `aborted: true`). While a case still acquires its device, Cancel waits for the device; that case measured nothing, and its record is dropped. A Cancel after the last case has measured everything leaves the run completed. |

The status card shows the adapter as the browser reports it (read without creating a device), whether
it lists `timestamp-query` (without it the sweep falls back to the wall clock, and the derivation
rejects a wall-clock record), cross-origin isolation, the runtime's case set id (`sweepCaseSetId`)
and, once the first case has started, dp4a and the timing unit. While running it shows the elapsed
seconds, the cases done, and the current case with its row count and the sweep's status line (the
geometry measured now), or the seconds left of the idle wait. A device loss stops the run after
that case.

The **Cases** table has one row per case, filled live: the default geometry's time per dispatch, the
drift (the default re-measured at the end of the case ÷ the first; outside 0.9–1.1 the derivation
drops the case's ratios), the fastest candidate whose output matched the default bit for bit, the
failed rows, the rows whose output differed from the default (never adopted), and the case's
seconds.

### The record

When the run ends with at least one case (also after Cancel, a device loss or an error), the page
merges the per-case records into one record in the shape one call over the planned cases writes
(`gpuTiming.quantized` aside, below):

| Field | What |
| --- | --- |
| `rows`, `cases` | every case's rows and summary, in run order |
| `startedAt`, `date` | the first case's start, the last case's end |
| `settings.cases` | every planned case (also the ones a cancel left out) |
| `candidateKernels` | recomputed over the merged rows with `sweepCandidateKernelsId` (each per-case record fingerprints only its own rows) |
| `aborted` | Cancel stopped a case midway or the run between cases (a run stopped by a device loss or an error has fewer cases than planned and `aborted: false`) |
| `gpuTiming.quantized` | true when any case's rounds looked quantized: the larger rounding bound, never the smaller. One call judges all the rows together (true only when every round looks quantized); the runtime does not export that judgement, and taking the larger bound only drops more observations from the ratio data |
| `deviceLost` | the first loss, or `null` |
| everything else | the per-case records' own values, which must agree (format, case set, default kernels, browser, adapter, dp4a, timing, the other settings) — the merge fails loudly otherwise |

The result card shows the file name, the sha256 of the bytes saved and a summary, then runs
`deriveGeometryProfile` on those very bytes and shows its verdicts and the derived table
(`geometryProfileJson`). The table is named `<vendor>-<architecture>` and matches the adapter's
vendor and architecture (an adapter without a vendor gets an injection-only table). It is derived
from this record alone: the repository's derivation adds the adapter's earlier records, so only the
fields this record has cases for (for V-15, `i8a8.conv3d`) say what that derivation will choose. A
record the derivation rejects shows the reason instead.

**Download record (JSON)** saves `geometry-sweep_<adapter>_<candidate set>_<UTC timestamp>.json`
(the adapter's description, or its vendor, architecture and device when the description is empty;
`quick+` is spelled `quick-plus`; the timestamp is the record's `date`). **Copy record** copies the
same JSON.

### Caveats

- A tab in the background is throttled: keep it in front, with nothing else on the GPU.
- The per-case records are kept only in the page's memory; a crashed tab loses them.
- `runGeometrySweep` consults no signal while it acquires its device. If the browser's adapter or
  device request never settles, the case never starts: Cancel and leaving the page do not end it,
  and since the run holds the app's GPU queue, no other page acquires a device either. Reload the
  tab to recover.
- The verdicts on the page are a preview. The table the repository registers comes from the
  derivation on the dev machine over every record of the adapter.

## The kernel-check page

The node lanes run on Dawn / Vulkan; a kernel rewrite that relies on the backend compiling the same
expression the same way must also be checked on the other backends — D3D12 (Chrome on Windows) and
Metal (macOS). This page runs those checks on the browser's own GPU. The checks are the shared
implementations of `@karume/kernel-checks`: the same cases and criteria as the runtime's node GPU
tests, not a copy.

| Check | Node GPU test | What must hold | Needs |
| --- | --- | --- | --- |
| GEMV subgroup32 | `gpu_linear_gemv_subgroup_test.ts` | `linearGemvReduce: "parallel-subgroup32"` equals the parallel reduction word for word (180 cases: i2 / i4 / i8, 5 lane counts, M = 1 / 4 / 8); rows past M stay unwritten; row 0 is the same at every M; outputs within the GEMM band of an FP64 reference (ADR 0101). | `subgroups`, `subgroup-size-control`, WGSL `subgroup_id` |
| RMS subgroup32 | `gpu_rms_norm_subgroup_test.ts` | `rmsNormReduce: "subgroup32"` through real Sessions: within the `rms_norm` tolerance of the CPU reference and row 0 identical at every M (48 cases); fused rms_norm + add identical to unfused, one fusion per run (12); 19 rows on 2 workgroups within tolerance (1) (ADR 0100). | the same |
| rms_norm: prefetch form vs runtime-trip form | `gpu_rms_norm_prefetch_test.ts` | The prefetch form of the 256-wide `rms_norm` writes the same words as the runtime-trip text, and both RMS → add orders the same words as their v1 WGSL, with NaN by payload (118 cases each: every multiple of 256 up to 4096 at rows 1 / 4 / 8 / 33 and 768 rows at three widths, one workgroup per row and three; adversarial rows) (perf-ledger K-99). | nothing optional |
| States attention rewrites (K-55 / K-91) | `gpu_state_attention_fold_test.ts` | The folded ①′ writes the same S words as ①′ (252 cases), the unrolled ③′ and fused stats + PV the same O words as their v1 WGSL (546 / 490); the reference outputs met the word classes the contract covers and no output word was left unwritten; every class the product folds is in the grid; 5 order-only mutants each change at least one word (else a zero-mismatch result would prove nothing). The page also runs each mutant on every case of its shape (the mutant matrix, not in the node test). | nothing optional |
| Integer GEMV | `gpu_linear_gemv_integer_test.ts` | The integer GEMV (`integerGemv`) writes every output word of its CPU reference (692 cases: i2 / i4 / i8, the 15 (lanes, rows) variants on small K and on the i2 / i8 widths in use, five of them on the i4 widths in use, plus a saturated fixture at each storage's largest K — every activation code −128 against the largest stored code, the i8 one offset at one element — on 2 variants, each at M = 1 / 8, with and without the SRQ epilogue); rows past M stay unwritten. | nothing optional |
| Byte-plane GEMV | `gpu_linear_gemv_planes_test.ts` | The byte-plane GEMV (`linearGemvPlanes`) meets contract class R (ADR 0147 decision 2; 240 cases: the 15 (lanes, rows) variants at K = 16 / 80 / 1536 and at 37 and 1029 columns — multiples of no workgroup's columns — with random codes, plus a saturated fixture at K = 1536 — every code −2 against activations of one sign — each at M = 1 / 8): every output within the GEMM band of an FP64 reference, and at M = 1 and K = 1536 within twice that band of the sequential GEMV; a second dispatch writes the same words; nothing is written past the M × n outputs; a NaN in one activation reaches every output of its row. The node test also shows the band is not vacuous (two WGSL mutants turn it red); the page does not run them. | nothing optional |
| f16 full state slots | `gpu_state_kv16_test.ts` | With `fullStateStorage: "f16"` (ADR 0143), the append writes what host rounding writes, word layout included (7 cases, one on every rounding boundary of f16), and every reader of an f16 slot writes the same words as its f32 kernel over the same slot rounded on the host (246 cases, 6 of them with subnormal slots). The node test also shows the check can fail (6 mutants each mismatch, one rounding toward zero); the page does not run them. | nothing optional |
| f16 attention scores (S) | `gpu_score_f16_store_test.ts` | The store of `attentionScoreStorage: "f16"` writes what host rounding writes (2 cases). It goes through `pack2x16float`, whose rounding WGSL leaves open: PASS means every word matches one rounding, to nearest or toward zero, and the `rounding` requirement names which. FAIL means anything else, a mix of the two included. | nothing optional |
| Bidirectional block (Gemma 4 12B images) | `gpu_state_bidirectional_test.ts` | The sliding ①ₜ with `GenerationRun.bidirectionalRows` (ADR 0144) writes every S word of the same kernel run with every score kept plus a host mirror of the predicate (33 cases: 12B's sliding shape and a small window, P = 0 / 17 / 1000 / 5000, four row layouts, plus one image chunk). The node test also runs predicate mutants that must mismatch; the page does not run them. | nothing optional |
| Expert linear (Gemma 4 26B A4B) | `gpu_expert_linear_test.ts` | `expert_linear` with f16 steps (ADR 0145) writes, for every pick, the words of `linear`'s sequential i4 GEMV on that expert with the steps widened to f32 (48 cases: gate_up and down × 3 tokens × 8 picks with repeated ids); every output starts as a NaN pattern, so an unwritten word mismatches too. | nothing optional |
| Fusion rules: fused vs unfused | the twin-graph tests of the fusion rules (`gpu_gated_residual_fusion_test.ts`, `gpu_silu_fusion_test.ts`, `gpu_adaln_fusion_test.ts`, `gpu_rope_fusion_test.ts`, `gpu_upsample2x_fusion_test.ts`, `gpu_rms_norm_rope_fusion_test.ts`, `gpu_linear_prologue_fusion_test.ts`, `gpu_attention_permute_fusion_test.ts`, `gpu_row_block_attention_test.ts`, `gpu_rms_norm_add_fusion_test.ts`, `gpu_gelu_mul_static_quantize_fusion_test.ts`, `gpu_linear_static_quantize_session_test.ts`) | Each built-in fusion's output equals the same values run node by node — a plain Session with an interposed zero-dispatch reshape, without the opt-in, or without the forced row blocks — word for word, with NaN by class or by payload as each case declares, and the fusion counter shows the expected hits on the fused side and 0 on the plain side, plus the dispatches where the test pins them (ADR 0147 decision 9, layer 1; 377 cases in 13 parts, one per rule: gated residual 9, SiLU 3, adaLN 11, RoPE 23, upsample x2 1, rmsNormRope 8, linear prologue 20, attention permutes 5 + 4 online sg32, row-block attention 15, RMS→add 200, gelu→mul→SRQ 54, linear→SRQ 24). The node tests also keep what is no twin on the given device (census, counterexamples, capped-limit grid-stride runs, the online wg routes); the page does not run those. | the linear prologue part: WGSL `packed_4x8_integer_dot_product`; the online sg32 part: a device whose online attention takes the sg32 form (subgroups and a linear lane mapping). Elsewhere that part reads as not run |

The page probes the adapter's features, then acquires its GPU with `subgroups: "available"`. **Run**
next to a check runs that check; **Run all** runs the ten in order; Cancel stops before the next
case. A check being run reads as not run until it ends, so a cancelled re-run does not keep its
earlier verdict. The status card shows the adapter's and the device's features, WGSL `subgroup_id` and the
linear lane mapping.

Each check gets a verdict:

- **PASS** — every part ran the number of cases its check exports (`GEMV_SUBGROUP_CASES`,
  `RMS_SUBGROUP_CASES`, `STATE_ATTENTION_CASES`, `GEMV_INTEGER_CASES`, `GEMV_PLANES_CASES`, `STATE_KV16_CASES`, `SCORE_F16_CASES`,
  `STATE_BIDIRECTIONAL_CASES`, `EXPERT_LINEAR_CASES`, `FUSION_TWIN_CASES`; the counts in the table above) and the
  states-attention check its `STATE_ATTENTION_MUTANTS` mutants and matrix rows, every criterion of
  every case has 0 failing words, every requirement is met, every mutant mismatched on its case and
  every mutant matrix row passed;
- **FAIL** — anything else; the part table shows cases, failing cases, mismatching words and the
  unmet requirements, below it any count that differs from the check's (a part with no cases
  compared nothing), and the first failure of each failing part (case, criterion, word index,
  expected vs actual as u32 hex — for a tolerance band the worst element, with the report);
- **not available** — the device lacks a feature the check needs (the subgroup checks without
  `subgroups` / `subgroup-size-control` / WGSL `subgroup_id`). This is not a pass: the page's
  overall verdict is then **INCOMPLETE** at best;
- **INCOMPLETE** — the check ran and nothing failed, but the device lacked what one of its parts
  needs (the fusion check's linear prologue or online sg32 part); the part table shows that part as
  "not run here" with the reason. Not a pass either: the overall verdict is **INCOMPLETE** at best;
- **ERROR** — the harness itself failed: a WebGPU error caught by the validation, out-of-memory or
  internal error scope every case runs under, an uncaptured WebGPU error, a lost device, or a
  mutation anchor that no longer matches the generated WGSL; the error is shown in full.

The overall verdict covers all eleven checks: a check that was not run counts like one that is not
available, so running one check alone gives **INCOMPLETE** at best (the result card lists the
checks not run). The release check is a **Run all**.

For the states-attention check the page also shows the mutant table (each mutant's kernel, case and
differing words), the mutant matrix and the folded ①′ classes. The matrix runs each mutant on every
case of its shape in its kernel's grid and classifies each case by whether the mutated code path
runs there: every acting case must mismatch, no idle case may, and the contraction-only cases (a
swap of two terms added to an exact 0, which shows only where the backend fuses the multiply-add)
are counted without a requirement. The folded ①′ classes: for each (shape, M) of the grid, the F the check runs
and the F the product selects (`stateQkFoldHeads`). The product folds only the classes of
`STATE_QK_FOLD_CLASSES` (E2B r = 8, D = 512, M = 1 / 4 / 8 and MiniCPM5 r = 8, D = 128, M = 1 —
observed identical on Vulkan and D3D12); E4B's F = 4 is still checked but **kept for K-95** (it
differed on D3D12 on 2026-10-27, perf-ledger K-95), so on D3D12 this check FAILs on the e4b-full
classes until K-95 lands. Next to the check's verdict the page shows its verdict **outside the K-95
classes** (the same evaluation without the cases of the classes kept only for K-95): it fails on
anything the product runs, so on D3D12 it is the one that must read PASS.

### The record (`format: "karume-playground-kernel-check/1"`)

**Download record (JSON)** saves `kernel-check_<UTC timestamp>.json` after a run ends — completed,
cancelled or aborted (it holds every check run on the page so far with the current device; a
reacquired device starts a new record): the environment fields (as on the other pages — adapter,
device features, WGSL language features, limits), `adapterFeatures`, the overall `verdict` and
every check with its `verdict` — `not-run` for one that was not run, `incomplete` for one with a part
not run here — and `seconds`, and either
the reason it was not available, the error, or `countMismatches` (the counts that differ from the
check's; empty when none do), the parts (summary — with `notRun`, the reason, for a part not run
here — requirements and `results`:
every case with each criterion's word count, failing words and first or worst failing word), the
mutants, the mutant matrix, the fold classes and, for the states-attention check, `productVerdict`
(the verdict outside the K-95 classes). While a run is in progress there is no record to download.
**Copy record** copies the same JSON.

## The session A/B page

The wall-clock A/B of `@karume/runtime/tune` (`runSessionAb`, judged by `judgeSessionAb`) on a real
Gemma 4 decode, with and without a candidate fusion rule: how a fusion is accepted or dropped on a
backend the node lanes do not reach (D3D12 in Chrome on Windows, Metal on macOS). Speed fusions are
adopted on the measured end-to-end time of untimed runs, not on per-dispatch GPU time
([ADR 0117](../../docs/decisions/0117-app-geometry-tuning.md), addendum of 2026-11-11).

**The pipeline.** The model and the preset are chosen as on the chat page (any model of the three
mounts), with the load options the chat page offers that the A/B needs: f16 KV and, for Gemma 4 26B
A4B, the expert cache (slots, host tier, self-speculation depth — 64 slots with the host tier `all`
runs 26B on a 16 GB GPU). Both arms load it with `buildGemmaPipeline` over the direct route.

**The arms.** `base` passes no rule; `rule` passes the candidate rule `candidate/gelu_mul` of
`@karume/candidate-rules` through the gemma pipeline's experimental `fusionRules` (the target
Session gets it). An arm's open builds the pipeline, close disposes it. A **timed run** is one
greedy generation (temperature 0, seed 0) of the chosen token count from one fixed prompt, in a
fresh sequence of capacity 1024; it returns the generated token ids, which the runner compares with
the base's first run, so identity is token identity. With **Read fusion and dispatch counts** on
(the default), the pipeline is built with `onRunDiagnostics` and each run reports the fusion counts
and the planned dispatches (`lastRunPipelines`) of its first one-row run (a decode run, or a draft
under self-speculation). Without the expert cache that callback moves greedy decoding from the GPU to
the CPU for both arms alike; under the cache decode is on the CPU either way. Off, the record has no
counts.

| Field | Default | Accepted |
| --- | --- | --- |
| Rounds | 4 | an integer ≥ 3 (one untimed warm-up round comes first) |
| Tokens per run | 96 | 2 to 512 (the first token comes from the prefill, so one token has no decode run) |
| Minimum gain | 0.01 (1% of the base's time) | a number ≥ 0 |
| Open and close per visit | on | on: every visit builds its pipeline, runs it once untimed and once timed, and disposes it (two pipelines of a large model do not fit together); off: both pipelines stay loaded |
| Backend | none (must be chosen) | `d3d12` / `metal` / `vulkan` / `other` — WebGPU does not expose it, so the record states what is chosen |

The runner visits the arms forward and in reverse in every round (base, rule, rule, base); the
status card shows the run in progress (round, visit, arm, timed or not) and how many runs have
ended. After a closed pipeline the page submits one empty command buffer and waits for it: Dawn
frees a destroyed buffer's memory only after a later submit completes, and without it the next
visit's build of 26B ran out of memory before its first submit (observed on Dawn / Vulkan).

When it ends the page shows, per arm, the identity (identical over N runs, or how many words of
which run differ and where first), the rule's hits, the dispatches and every timed round's two
visits; then the judgement of the rule arm: **Adopt**, **Reject** or **Inconclusive**, its reason in
words and its numbers (median paired difference, base median, gain, noise floor with the visit and
delta spreads, the paired differences per round). Identity is required: tokens that differ reject
the rule whatever the times. A verdict speaks for that device, backend and work only, and
inconclusive does not mean the difference is zero.

### The record (`format: "karume-playground-session-ab/1"`)

**Download record (JSON)** saves `session-ab_<UTC timestamp>.json` once the A/B has ended:
`recordEnvironment`, the `request` (model, preset, load options, rounds, tokens, minimum gain, open
per visit, counts, backend, the prompt, the sampler, the capacity and the two arms with the rules
they passed), `pipeline` (the entry point), `distribution` (directory and manifest sha256),
`source`, `rule` (name and package), the runtime's `record` (`karume-session-ab/1`) unchanged and
the `judgements`. A stored `record` can be judged again with `judgeSessionAb`, with another minimum
gain too. A failed or cancelled A/B leaves no record (the runner returns one only at the end).
**Copy record** copies the same JSON.

Caveats:

- Never run it while other work uses the same GPU: the times would measure the contention.
- The status card's ticker re-renders the page every 250 ms while runs go; nothing the page does
  runs inside a timed span but the generation.
- Building a pipeline per visit is not timed, but it heats the GPU between visits like any load;
  the A B B A order spreads that over both arms.

## Develop

```sh
pnpm --filter @karume/playground test        # vp test: the server's request handling (/models/ and /hf/, aliases included), the page registry, useGpu's teardown order, record / cache / preset / image helpers, pretrainedRef, zip writer, attention bench table, the geometry sweep's merge and run order, every page's pure helpers (settings, records, progress, pipeline slots)
pnpm --filter @karume/playground test        # vp test: the server's request handling (/models/ and /hf/, aliases included), the page registry, useGpu's teardown order, record / cache / preset / image helpers, pretrainedRef, zip writer, attention bench table, kernel-check verdicts and record, every page's pure helpers (settings, records, progress, pipeline slots)
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
   `{ directory, hfRepo: <family>.SOURCES[...].repo }` with `import * as <family> from "@karume/models/<subpath>"` to serve it on the HF-compatible route too).
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
| `useGpu` | `src/hooks/use-gpu.ts` | One GPU per page, released on unmount, re-acquired after a device loss (`reacquire`) or an options change (GPU timing). `track(work)` holds the device until `work` settles: the destroy waits one macrotask after the release and then for every tracked promise, and the next acquisition — of any page, through the app-wide queue the root layout provides (`GpuQueueContext`) — waits for that destroy (flush before destroy). A device released while still acquiring holds no work: it is destroyed when it arrives, and the next acquisition does not wait for it. A page MUST pass every pipeline / Session disposal and every build or run on the device through `track`, started no later than the effects of the commit that releases the device. `lease` names one acquisition (a new one on every options or attempt change, also back to earlier options); `state` shows a device only for the lease it was acquired for, so in the render that changes the lease it is already `acquiring` and effects keyed on the device dispose in the same commit. |
| `runOnGpuQueue`, `useGpuQueue` | `src/hooks/use-gpu.ts` | Work that acquires and destroys a device of its own (`runGeometrySweep`) as the app's GPU user: it starts after the queue's last released device is destroyed, and the next acquisition waits until it has settled. The queue is taken in the call itself (call it from the event handler). It MUST NOT be called while a `useGpu` device is live (the task would start beside it), so only a page without `useGpu` calls it. |
| `useTimestampQuery` | `src/hooks/use-timestamp-query.ts` | Whether the adapter lists `timestamp-query`. |
| `useRunTask` | `src/hooks/use-run-task.ts` | One cancellable run: state (`idle` / `running` with stage and counter / `done` / `cancelled` / `error`), the elapsed-time ticker, Cancel, and an abort when the GPU lease ends (the page is left or the GPU is acquired again). Takes the page GPU's `{ track, lease }`, so every run holds the device until it settles. A report with `cancellable: false` and its required `cancelNote` marks a stage that takes no signal. |
| `RunStatusCard`, `RunStatus`, `GpuAlerts` | `src/components/run-status-card.tsx` | The Status card: GPU, cross-origin isolation, the page's rows and controls, the GPU's failure states, and the run (elapsed, stage, progress bar, Cancel — disabled with the stage's `cancelNote` while it is not cancellable — error in full). |
| `recordEnvironment` | `src/lib/environment.ts` | The fields every record starts with: `recordedAt`, `userAgent`, `crossOriginIsolated`, `adapter`, `device`. |
| Record helpers | `src/lib/record.ts` | Non-finite count, RGBA stack + sha256, `f32Sha256` (raw f32 bytes), `rgb8Sha256` (the RGB8 input), `sourceRecord`, diagnostics and build cost per component, a run's share of a kept Session's submit / dispatch counts (`submitCountsSince`, `perRunSubmitCounts`), attention route, GPU time per key and per op, device facts, `adapterText`, the stage clock (`foldStageEvent`, `stageSecondsOf`), timeline. |
| `RecordActions` | `src/components/record-actions.tsx` | Download record (JSON) and Copy record, with room for the page's own buttons. |
| `downloadBlob`, `downloadJson`, `fileStamp`, `zipStored` | `src/lib/download.ts`, `src/lib/zip.ts` | Downloads and the zip writer. |
| `ImageInput`, `decodeImageFile`, `rgbaToRgb8` | `src/components/image-input.tsx`, `src/lib/image.ts` | An image file field with a preview; the decoded RGBA and the pipelines' `Rgb8Image` (alpha dropped; a canvas keeps pixels premultiplied, so a fully transparent pixel reads as black). |
| `RgbaImage` | `src/components/rgba-image.tsx` | RGBA pixels in a canvas, with Download PNG for an opaque image (`encodePng` refuses alpha other than 255; a refused encode is shown). |
| `AudioClip` | `src/components/audio-clip.tsx` | A mono f32 waveform as a WAV: a player and Download WAV. |
| `ErrorAlert`, `Field`, `ChoiceSelect`, `KeyValueTable`, `FramePlayer` | `src/components/` | Error in full, labelled field, select, key-value table, frame player. |

Chat pieces (`message`, `message-scroller`, `bubble`, `input-group`) and `tabs`, `scroll-area`,
`spinner`, `empty`, `tooltip` are installed under `src/components/ui/` from the Shadcn CLI.

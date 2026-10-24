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
| `/models/index.json` | which mounts exist and whether their directory is there |

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
for the chosen preset and text encoder path (construction creates no Session; the GPU work starts
with **Run**). One generation runs at a time.

| Control | What it does |
| --- | --- |
| Preset | The manifest's `quants` (`quant` of `fromPretrained`); the manifest's `defaultQuant` is preselected and marked "(default)". Changing it rebuilds the pipeline. |
| VAE tile | `vaeTile` of the request: 16 (default) or 32. 32 decodes in 32-latent tiles (fewer, larger tiles) and needs about 9.4 GiB on the device during the decode; the 10 GB class keeps 16. |
| Size | 1280×704 or 704×1280. |
| Frames | 4n+1 from 5 to 121 (default 33). The video plays at 24 fps. |
| Steps | Denoise steps (default 50). **Quick 2-step check** sets 2 steps and 17 frames — enough to run every stage, VAE tiles included. |
| Seed | The seed of the initial noise (default 42). |
| Shift | The flow-matching shift; empty = the manifest's default (5). |
| Text encoder | `precomputed`: the text embedding asset's fixed prompts (choose one by name), no umT5. `gpu`: umT5 on the GPU (about 5.3 GiB of weights) with a free prompt and negative prompt (empty negative = the official default). Changing it rebuilds the pipeline. |
| Run / Cancel | Run one generation; Cancel stops at the next stage boundary, step or VAE tile. |

While running, the status card shows each stage's start and end (`text_encoder` / `transformer` /
`vae_decoder`), the denoise step counter, the VAE tile counter and the elapsed seconds. Errors are
shown in full (message, stack, `cause` chain).

The result card plays the frames in a canvas (play / pause, frame scrubber) and shows the record:
adapter (vendor / architecture / device / description), preset, VAE tile, size, frames, steps, seed,
shift, per-stage seconds, wall seconds, the number of non-finite values in the output, the sha256 of
the RGBA pixels of all frames (concatenated in order, the `wanFrameToRgba` rule), and the Session
diagnostics per component (runs, weight memory allocated in GiB, submit count).

### Where records land

Nothing is written on the server. The buttons save into the **browser's download folder** of the
machine running Chrome:

- **Download record (JSON)** → `wan22_<UTC timestamp>.json`
- **Download frames (zip of PNGs)** → `wan22_<UTC timestamp>_frames.zip` (store-only zip,
  `frame_000.png` …, lossless RGB from the same RGBA bytes the sha256 covers)
- **Copy record** → the clipboard (paste into an issue or a research note)

## Develop

```sh
pnpm --filter @karume/playground test        # vp test: the server's request handling, record helpers, zip writer
pnpm --filter @karume/playground typecheck
pnpm --filter @karume/playground check       # vp check: format + lint + types
pnpm --filter @karume/playground build
```

The tests need no GPU. The GPU path is checked by hand on the target machine.

`src/routeTree.gen.ts` is generated by the TanStack Router plugin (file-based routing under
`src/routes/`); do not edit it. UI components under `src/components/ui/` come from the Shadcn CLI
(Base UI backend): add more with `pnpm dlx shadcn@latest add @shadcn/<name>` in this directory.

### Adding a model page

1. Add the distribution mount in `vite.config.ts` (`distributions: { <name>: "models/<dir>" }`).
2. Add `src/routes/<page>.tsx` (`createFileRoute("/<page>")({ component })`) and put the page's code
   under `src/models/<page>/`.
3. Add an entry to `MODEL_PAGES` in `src/models/registry.ts` (the index lists it as a card, the
   header links it). Widen `ModelPage["path"]` with the new route.

The shared pieces a page builds on:

| Piece | Where |
| --- | --- |
| Distribution source over the dev server (`localDirectory` with range reads, `crossRepo`) | `src/lib/distributions.ts` |
| Which mounts exist (`/models/index.json`) | `src/hooks/use-distribution-index.ts` |
| One GPU per page, released on unmount, re-acquire after device loss | `src/hooks/use-gpu.ts` |
| Record helpers (non-finite count, RGBA stack + sha256, diagnostics summary) | `src/lib/record.ts` |
| Downloads and the zip writer | `src/lib/download.ts`, `src/lib/zip.ts` |
| Error alert, labelled fields, select, key-value table, frame player | `src/components/` |

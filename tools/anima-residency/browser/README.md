# Anima DiT residency browser check

A local page for observing Anima's DiT residency (ADR 0112) in Chrome (WebGPU): consecutive
generates with residency on or off, VRAM held by dummy buffers between generates, and the
`residency` events the pipeline emits — `retained`, `released`, `evicted`, each with its reason,
including the pre-emptive `evicted` / `headroom`. It is a confirmation page, not a speed
comparison.

Run from the repository root (tested with Deno 2.9.6):

```sh
deno task bench:anima-browser
```

Open **http://localhost:8788** in Chrome. Stop the server with Ctrl+C.

The server bundles the local karume modules into a temporary directory at startup and serves the
distribution in `models/karume-anima` (the directory containing `karume.json`) read-only, with byte
ranges. It listens only on the loopback interface. Override the port or the distribution when
needed:

```sh
deno task bench:anima-browser --port 8788 --source /path/to/karume-anima
```

The page uses the distribution's default model. Only a single distribution directory is served,
so the cross-repository layout of `karume-anima-extra` is not supported.

## How the page drives the pipeline

- The pipeline is built **once** with `{ gpu, residency: "transformer" }` on the first generate and
  reused afterwards. The **次の generate の常駐** selector sets the per-generate `residency` of the
  request (`transformer` keeps the DiT after the generate; `per-stage` releases it).
- The pipeline and the dummy buffers share one GPU device from `acquireGpu()`.
- **VRAM を埋める** allocates the requested GiB as 1 GiB `STORAGE` buffers (or `maxBufferSize`
  pieces if smaller), each inside an `out-of-memory` error scope, and stops at the first
  out-of-memory error. The status line and the dummy line show how much was actually allocated.
  **ダミーを解放** destroys them.
- **pipeline を破棄** disposes the pipeline, destroys the dummies and the GPU device. The next
  generate builds everything again. Use it after a device loss.
- Leave **steps** and **guidance scale** blank to use the distribution defaults. The default model
  runs with guidance scale 1, where a negative prompt is rejected — leave the negative prompt
  blank unless you also set guidance scale above 1.

Each generate adds one table row: requested residency, dummy GiB held at the start, wall time
(`generate()` call to resolution, PNG encoding excluded), the start-to-end time of every stage
(a stage retried after an out-of-memory eviction appears twice), the residency events as
`action/reason @seconds (position)`, the first 12 hex digits of the PNG's SHA-256, and the error
name and message when the generate fails. A failed generate stops the remaining runs of that
batch. The latest image is shown below the table.

## What to confirm

1. **Residency makes the 2nd and later generates faster.** Run 3 generates with residency
   `transformer`. The 1st includes the DiT load; the 2nd and 3rd skip it, so their wall time and
   their `transformer` stage time are shorter. The PNG SHA-256 is identical across rows with the
   same prompt, seed, and resolution.
2. **Each residency-on generate reports `retained/request`.** Every row run with `transformer`
   lists `retained/request` (emitted before the `transformer` stage ends). A row run with
   `per-stage` after a resident one lists `released/request` instead.
3. **Held VRAM causes an eviction, not a failure.** With a resident DiT (after step 1), press
   **VRAM を埋める** with enough GiB that a text or VAE stage no longer fits next to the DiT (start
   with 6 GiB on a 10 GiB GPU and adjust). Run 1 generate. The row should list
   `evicted/headroom` (pre-emptive: before the first stage, or after `transformer` and before
   `vae_decoder`) — or `evicted/out-of-memory` (the reactive fallback) — and then complete with
   the **same PNG SHA-256** as before. After an eviction the pipeline is downgraded for good:
   later residency-on rows list `released/downgraded`. If no eviction appears, raise the dummy
   size; if the allocation stops early, the status line says where.

The pre-emptive check costs up to two trial allocations per generate (about 0.1 s; the measured numbers
are in `docs/research/2026-09-27-h35-oom-device-lost.md`, which also explains why an
out-of-memory error during upload can lose the device). Chrome's WebGPU implementation (Dawn)
appears to place upload staging in system RAM rather than VRAM (inferred from source, not
measured), so the reason Chrome reports may differ from the Deno
probe; record what the page shows.

## Export

**JSON を保存** downloads `anima-residency-browser-<quant>-<timestamp>.json` with the adapter
information, user agent, checkout revision and dirty flag, bundle hash, distribution name,
manifest SHA-256, pipeline build times, the dummy allocations (each request and how much was
allocated), any device loss, and every table row including the request and the full PNG
SHA-256. Save it to `outputs/bench-browser/` to keep it next to the other browser results (that
directory is not tracked by git).

## Per-op GPU timing (K-70)

The page doubles as a per-op GPU profiler for perf-ledger K-70 (why Anima is slow on Apple / Metal
in Chrome, across the int8, f16 and f32 paths). A Deno twin, `tools/anima-residency/profile.ts`,
writes the same JSON on a machine without Chrome.

### What is recorded

- **quant** — the selector lists the quants of the distribution's default model (default:
  `defaultQuant`). The quant and the **GPU 時間を採る（timestamp-query）** checkbox are fixed when the
  page takes the GPU (the first generate, or **VRAM を埋める**); both controls are locked while a GPU
  is held. To switch, press **pipeline を破棄** and choose again. Every row records its quant.
- **GPU time per stage** — with the checkbox on, the GPU is acquired with `gpuTiming: true` and the
  page sums `Session.diagnostics().lastRunTiming` over every run of each stage (the DiT runs once per
  step and per CFG branch, the text stages once per prompt, the VAE once per tile). The checkbox is
  on by default when the adapter lists `timestamp-query`, and disabled with a note when it does not.
- **Planned dispatches per stage** — `lastRunPipelines` is summed per key on every run, with or
  without timing.
- The **GPU 時間** column shows each stage's GPU total in ms; expand a stage for its top 10 keys
  (key, ms, dispatch count, share of the stage).

The int8 and f16 DiT paths are selected by the quant (`f16+dit8-a8-attn8-s16` and the other `dit8`
quants run int8 linear / attention, `f16` stores the DiT in f16 and computes in f32, `f16-c16`
computes in f16 and needs `shader-f16`). The text encoder, text conditioner and VAE always compute
in f32, so every run also profiles the f32 path.

Caveats:

- Timing splits every dispatch into its own compute pass, so wall times with timing on are **not
  comparable** to wall times with timing off.
- Chrome quantizes timestamps to 100 µs unless **WebGPU Developer Features** is enabled
  (`chrome://flags/#enable-webgpu-developer-features`). Enable it before profiling. When every key
  of a stage is a multiple of 100 µs, the stage summary says `100 µs 量子化の疑い`.
- Deno does not convert timestamps to nanoseconds; it returns raw GPU ticks (1 tick = 52.08 ns on
  the Arc B570 — see `docs/known-issues.md`). The Deno JSON marks this with `gpuTiming.unit:
  "deno-raw-tick"`; the page writes `"ns"`. Shares within a stage are unaffected.
- Deno on Metal loses the device when timing is on (`docs/limitations.md`, "Metal（Apple GPU）では GPU
  側 timestamp 計測が実用にならない"). That observation is from wgpu; Chrome uses Dawn and has not
  been tried. If the device is lost in Chrome, the row shows the error — press **pipeline を破棄**,
  clear the checkbox, and still record wall times and dispatch counts.

### Export format `karume-anima-residency-browser/2`

On top of the fields listed under [Export](#export), version 2 adds:

- `quant` (the current or next pipeline's quant) and `gpuTiming: { enabled, feature, unit }`;
- per row: `quant`;
- per stage (`rows[].stages[]`): `gpu: { totalNs, runs, clampedNegativeSamples, entries: [{ key,
  ns, dispatchCount }] }` (every key, sorted by `ns` descending; absent when timing is off) and
  `pipelines: [{ key, dispatchCount }]` (sorted by key);
- per pipeline build (`pipelineLoads[]`): `quant` and `gpuTiming`.

The downloaded file is named `anima-residency-browser-<quant>-<timestamp>.json`.

### Profiling on Apple / Metal (Chrome)

1. Enable `chrome://flags/#enable-webgpu-developer-features` and restart Chrome.
2. Start the server (`deno task bench:anima-browser`) and open http://localhost:8788.
3. Leave **GPU 時間を採る** on, select quant `f16` and resolution 512x512, set N to 2, and press
   **N 回生成** (the first generate loads the resident DiT; the second is the steady state).
4. Change the resolution to 1024x1024 and run 2 more generates (the resident DiT is kept).
5. Press **JSON を保存**, then **pipeline を破棄**.
6. Select the default quant (`f16+dit8-a8-attn8-s16`) and repeat steps 3 to 5.

### Profiling with Deno (Arc B570 or any Linux / Vulkan machine)

Run from the repository root, once per quant and resolution:

```sh
deno run -A tools/anima-residency/profile.ts --resolution 512x512 --count 2 --quant f16
deno run -A tools/anima-residency/profile.ts --resolution 512x512 --count 2
deno run -A tools/anima-residency/profile.ts --resolution 1024x1024 --count 2 --quant f16
deno run -A tools/anima-residency/profile.ts --resolution 1024x1024 --count 2
```

Flags: `--source` (default `models/karume-anima`), `--quant` (default: the distribution's
`defaultQuant`), `--resolution` (default `1024x1024`), `--count` (default 2), `--seed` (default 42),
`--prompt` (default: the page's prompt), `--out` (default `outputs/bench/karume/<date>_metal-recon/`)
and `--date` (the date in the default `--out`; default: today). The CLI builds the pipeline once with
`residency: "transformer"`, prints per generate each stage's wall and GPU time and the DiT stage's
top 5 keys, and writes `anima-profile-<adapter>-<quant>-<WxH>-<ISO time>.json` with the same shape as
the page export (`userAgent` is `{ deno }`, there is no `bundleSha256`, and no dummy buffers).

The page and CLI JSON files are compared offline; there is no comparison tool yet.

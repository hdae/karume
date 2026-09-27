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

**JSON を保存** downloads `anima-residency-browser-<timestamp>.json` with the adapter
information, user agent, checkout revision and dirty flag, bundle hash, distribution name,
manifest SHA-256, pipeline build times, the dummy allocations (each request and how much was
allocated), any device loss, and every table row including the request and the full PNG
SHA-256. Save it to `outputs/bench-browser/` to keep it next to the other browser results (that
directory is not tracked by git).

# Geometry sweep in Chrome

A local page that runs the GEMM tile geometry sweep (perf-ledger K-70) in Chrome's WebGPU, with the
same measurement core and the same JSON format as the Deno CLI. What is measured and the output
format are described in [../README.md](../README.md).

Run from the repository root. The page has been run on an Apple M2 (Chrome 153) and, served from a static host, on an RTX 5070 Ti (Windows, Chrome 153).

```sh
deno task bench:geometry-browser
```

Open **http://localhost:8789** in Chrome. Stop the server with Ctrl+C. Use `--port` to change the
port:

```sh
deno task bench:geometry-browser --port 8789
```

The server bundles the local karume modules into a temporary directory at startup and serves only
the page, the bundle, and `/config.json` (the checkout revision, dirty flag, and bundle hash). It
listens only on the loopback interface. The inputs are synthetic, so no model is served.

## Before measuring

1. Enable `chrome://flags/#enable-webgpu-developer-features` and restart Chrome. Without it,
   Chrome rounds timestamps to 100 µs; the page still runs, and the export marks the rounds as
   quantized.
2. Close other GPU-heavy tabs.

## Using the page

- **op** — the kernel families to sweep (all are checked by default): `linear`, `matmul` (mirrors
  of `linear` cases — no census has a rank-2 matmul), `bmm` (rows of the Anima op census),
  `i8a8-linear`, `attention`, `i8a8-attention`, and `conv2d`.
- **候補** (candidates) — `quick` runs the default geometry plus 4–5 others per case; `full` runs
  the whole grid and takes a long time on a slow GPU.
- **rounds** — timed passes per geometry (default 5; the minimum is kept).
- **GPU の timestamp で測る** — time passes with `timestamp-query` (on by default when the adapter
  lists it). If the device is lost with timestamps on, clear it and start again: the sweep then
  uses the wall clock.
- **開始** acquires a fresh GPU device and starts the sweep; **中断** stops after the current
  geometry.
- The table shows one block per case. The first row (bold, marked `*`) is the geometry the
  production code uses; **対既定** is its time divided by the row's time (above 1 is faster).
  **出力の一致** is red when the output differs from the default geometry's output — such a
  geometry is not usable.
- The last row of each block is the default geometry measured again at the end of the case
  (`既定の再測定 ×N.NN`, the repeat divided by the first measurement). Outside 0.9–1.1 it is shown
  in red as a hint to run that case again.
- **JSON を保存** downloads the last sweep as `geometry-sweep-browser-<timestamp>.json`. Save it to
  `outputs/bench-browser/` to keep it next to the other browser results (that directory is not
  tracked by git).
- **JSON を表示** / **JSON をコピー** put the same JSON into a text area on the page and copy it to
  the clipboard. Use them where the page is served from a static host that blocks downloads (the
  page, `runner.js` and a static `config.json` can be published as they are; all paths are relative).

## Suggested run on Apple / Metal

1. Start the server and open the page with the developer features flag enabled.
2. Select `quick`, keep all op families checked, and press **開始**.
3. Press **JSON を保存** when the status line says the sweep is complete.
4. If a family looks interesting, run it again with `full` and only that family checked.

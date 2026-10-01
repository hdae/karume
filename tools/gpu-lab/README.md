# GPU lab

A local Chrome (WebGPU) page that runs the whole geometry-profile loop on one machine, as a proof of
concept for ADR 0115 (addendum decision 6):

1. **Benchmark** — sweep GEMM tile geometries on this GPU (the **1. 掃引** tab).
2. **Optimize** — turn the sweep into a geometry profile, a static table of tile geometries for one
   adapter (the **2. プロファイル** tab).
3. **Run** — inject the table with `acquireGpu({ geometryProfile })` and run Anima on it (the
   **3. Anima** tab).

It replaces the two earlier pages (the Chrome page of `tools/geometry-sweep` and the Anima residency
check page). Each tab keeps what its page did: the same measurements, table columns, JSON fields, and
file names. The Deno CLIs stay where they were (see [Deno twins](#deno-twins)).

## Running

Run from the repository root (tested with Deno 2.9.6):

```sh
deno task bench:gpu-lab
```

Open **http://localhost:8790** in Chrome. Stop the server with Ctrl+C. Override the port or the
Anima distribution when needed:

```sh
deno task bench:gpu-lab --port 8790 --source /path/to/karume-anima
```

The server bundles the local karume modules into a temporary directory at startup and serves only
the page, the bundle (`/main.js`), `/config.json` (the checkout revision, dirty flag, bundle hash,
and distribution name), and the distribution in `models/karume-anima` (the directory containing
`karume.json`) read-only, with byte ranges. When `--source` is given and that directory has no
`karume.json` (or does not exist), the server stops with `--source <path> has no karume.json` and
exit code 1. When `--source` is omitted and `models/karume-anima` has no `karume.json`, it prints a
warning and still starts: the sweep and profile tabs do not need a model, `/config.json` has
`source: null`, `/models/anima/…` answers 404, and the Anima tab is disabled with
`配布形が無い（--source で指定）` on its status line. It listens
only on the loopback interface and sends the cross-origin isolation headers (COOP / COEP / CORP).
The page uses the distribution's default model. Only a single distribution directory is served, so
the cross-repository layout of `karume-anima-extra` is not supported.

### Using the page through port forwarding

The server only answers requests addressed to `localhost`, `127.0.0.1`, or `[::1]`. To use a GPU on
another machine (for example a laptop's Chrome with the checkout and models on a workstation),
forward the port and open the page on the machine with the GPU:

1. Start the server on the machine with the checkout: `deno task bench:gpu-lab`.
2. On the machine with the GPU and Chrome, forward the port:
   `ssh -N -L 8790:127.0.0.1:8790 <user>@<machine with the checkout>`.
3. Open **http://localhost:8790** in Chrome on the machine with the GPU.

The model weights then travel through the tunnel, so the first Anima generate spends most of its
text stage on the download; compare stage times only after the weights are cached.

## Before measuring

1. Enable `chrome://flags/#enable-webgpu-developer-features` and restart Chrome. Without it, Chrome
   rounds timestamps to 100 µs; the sweep still runs, but its JSON is marked `quantized` and the
   profile tab rejects it.
2. Close other GPU-heavy tabs.

## The header

- **Tabs** — only the selected tab is shown. Every result table scrolls inside its tab; the page
  itself does not grow.
- **GPU 設定** (GPU settings) — the geometry profile and **GPU の timestamp で測る（timestamp-query）**.
  Nothing changes until **適用** (apply); `未適用の変更があります` marks a pending change. The
  profile choices are:
  - `自動` (auto) — no injection: the runtime selects a built-in profile from the adapter's vendor,
    architecture, and description (the option shows which one this adapter gets).
  - `default` — inject the default table.
  - each built-in profile id (every entry of `BUILTIN_GEOMETRY_PROFILES`) — inject that table on any
    GPU (for A/B runs of another machine's table). It is recorded as `builtin:<id>`. The label
    says `（注入 — <description> 用）` for a table that matches a description, `（注入 — <vendor> /
    <architecture> 用）` otherwise, and `（注入専用）` for a table without `match`: the runtime never
    selects such a table by itself, so this list is the way to use it.
  - `生成した表 #<n>: <id>（注入）` — inject a table made in the profile tab. Every table derived
    successfully stays as its own option for the life of the page, numbered from 1 in the order
    they were made (deriving again adds an option and never removes one; a failed derivation adds
    none). The applied one is marked `（適用中）`; a table derived again with the same id is a
    different option, so the mark tells which values are applied.
  - `保存した表: <id>（<saved at>・<adapter>）（注入）` — the last table derived on an earlier
    visit. Each successful derivation also saves that table to `localStorage`
    (`karume-gpu-lab/last-generated-profile/1`, overwriting the previous one, with the adapter
    description or vendor/architecture and the checkout revision), and the page offers it again when
    it opens, because Chrome sometimes has to be reloaded and the derived table would otherwise be
    lost. A saved value that cannot be read or does not pass the runtime's profile check is deleted,
    and the GPU settings status line says so. The applied settings themselves are not saved. Both
    kinds are recorded as `generated:<id>`.

  **適用** disposes the Anima pipeline, its dummy buffers, and its GPU device; if the Anima tab held a
  device, it acquires a new one with the new settings at once, and the Anima status line says
  `GPU を取り直しました（quant <quant>・幾何プロファイル <requested>）`. The timestamp setting applies to the
  next sweep and to the Anima device.
- **Environment line** — the adapter, whether GPU time is taken, the applied geometry profile
  (`自動 → <id>` or `<requested>（注入）`, where `<requested>` is `default`, `builtin:<id>`, or
  `generated:<id>`), and the checkout revision. On the sweep tab it adds that
  the sweep measures explicit geometries, so the profile does not affect its results.

Only one GPU operation runs at a time: a sweep, an Anima action (generate, fill VRAM, release,
dispose), or applying GPU settings. Starting another while one runs is refused with a message; two
at once would make both timings meaningless.

## 1. 掃引 (sweep)

The same sweep as `tools/geometry-sweep`: both call `runGeometrySweep` of `@karume/runtime/tune`
(`packages/runtime/src/tune/`), which acquires the dedicated device, measures, and builds the
record (what is measured, the candidate grids, and the output format are described in
[../geometry-sweep/README.md](../geometry-sweep/README.md)).

- **op** — the kernel families to sweep (all checked by default).
- **候補** (candidate set) — `quick+` (default), `quick`, or `full`. `quick+` is `quick` plus every
  geometry a registered profile uses (`BUILTIN_GEOMETRY_PROFILES`): for f32, each profile's
  `gemmRows` rules and `attention.qk` / `attention.pv`; for conv2d, the f32 `quick+` geometries with
  n-tiles of 64 or 128 plus each profile's `conv2d.rows64` / `rows32`; for int8, each profile's three
  `i8a8` fields. Duplicates are measured once.
- **rounds** — timed passes per geometry (default 5; the minimum is kept).
- **開始** acquires a fresh GPU device (with timestamps when the GPU settings ask for them; no
  profile is injected — every geometry is explicit) and starts the sweep; **中断** stops after the
  current geometry.
- The table shows one block per case. The first row (bold, `*`) is the production default;
  **対既定** is its time divided by the row's time (above 1 is faster); **出力の一致** is red when the
  output differs from the default's. The last row of a block is the default measured again
  (`既定の再測定 ×N.NN`); outside 0.9–1.1 it is red as a hint to re-run that case.
- **JSON を保存** downloads `geometry-sweep-browser-<timestamp>.json` (`karume-geometry-sweep/2`).
  Save it to `outputs/bench-browser/`, which is where the profile tab's registration command
  expects it (not tracked by git). **JSON を表示** / **JSON をコピー** put the same JSON into a text
  area and the clipboard, for hosts that block downloads.

The JSON `settings` has `candidateSet` (`quick`, `quick+`, or `full`) and, for compatibility,
`quick` (true only for `quick`).

## 2. プロファイル (profile)

Builds a geometry profile from sweep results with the same pure functions as the CLI (the
generator in `packages/runtime/src/tune/derive.ts` — the same rules as `deriveGeometryProfile` of
`@karume/runtime/tune` — and the TypeScript rendering in `tools/geometry-sweep/render.ts`, both used
by `main.ts profile`).

- **Inputs** — **掃引タブの直近の結果を使う** takes the last sweep of the sweep tab from memory
  (checked whenever a new sweep is available; the label is updated when a sweep ends, even while
  this tab is open); **掃引の JSON を読み込む** adds files; the text area
  adds pasted JSON. Several sweeps of the same adapter can be combined. Each input is checked when
  it is added: a sweep without GPU timestamps (unit `wall` or no `gpuTiming`) or with quantized
  timestamps is rejected, and the status line shows the generator's reason as is. The SHA-256 is
  taken over the bytes given: for a file, the bytes read from it (the same value as the CLI); for
  the in-memory sweep, the bytes **JSON を保存** writes; for pasted text, its UTF-8 encoding.
- **id / vendor / architecture** — blank fields use the first input's adapter (the placeholders show
  the values); type to override. **vendor だけで当てる** leaves `architecture` out of `match`.
  **description でも照合する** (off by default) adds the first input's adapter description to `match`
  (shown next to the box; it fails when the description is empty), like the CLI's `--description`.
  **注入専用（自動選択しない）** leaves `match` out, like `--opt-in`: the runtime never selects the
  table, and the vendor, architecture, and description controls are disabled. The minimum
  speed-up is fixed at ×1.05 (the CLI default).
- **表を作る** derives the table. The table lists, per field, the cases, the result (`採用 <geometry>`
  with the geometric mean and range of its speed-ups, or `既定 <geometry> のまま` with the reason),
  every rejected geometry with its reason, and the cases dropped from a sweep because the default
  re-measurement was outside 0.9–1.1, failed, or is missing (`比の材料から外したケース` — their
  speed-ups are not used, but a mismatch or a failure there still rejects the geometry; the rule is
  the CLI's, see `tools/geometry-sweep/README.md`). Errors (adapters that differ, a sweep given twice, a
  default row that is not the current runtime default) stop the generation and are shown as is.
- **出力** — one of four texts, with **コピー** and **保存**:
  - **TS の生成物** — the source the CLI would write, before `deno fmt` (saved as `<id>.ts`).
  - **アプリ用 TS** — the table as a constant for an application, before formatting (saved as
    `geometry-profile-<id>.ts`): `import type { GeometryProfile } from "@karume/runtime";` and
    `export const <ID>: GeometryProfile = { … };` with the same values and `provenance` as the
    generated file, the last `maxRows` written as `Number.POSITIVE_INFINITY`. Pass the constant to
    `acquireGpu({ geometryProfile: <ID> })` to use the table without registering it.
  - **注入の JSON** — the `GeometryProfile` value (saved as `geometry-profile-<id>.json`). The last
    `gemmRows` rule's `maxRows` is written as `1e999`, which `JSON.parse` reads as `Infinity`;
    `JSON.parse` of the text can be passed to `acquireGpu({ geometryProfile })` as is.
  - **登録のコマンド** — where the sweep files must be (`outputs/bench-browser/<name>`) and the
    `deno run -A tools/geometry-sweep/main.ts profile …` command that writes the formatted file to
    `packages/runtime/src/kernels/geometry-profiles/<id>.ts` and prints the lines to add to
    `geometry-profiles/index.ts`.
- **この表を適用** selects that table's `生成した表 #<n>: <id>` option in the GPU settings and
  applies it. If another GPU operation is running, the selection is left as it was; if applying
  fails, the selection returns to the applied setting.

## 3. Anima

The Anima DiT residency check (ADR 0112) and per-op GPU profiler (perf-ledger K-70). It needs the
distribution; without one (see [Running](#running)) every control is disabled.

- The pipeline is built **once** with `{ gpu, residency: "transformer" }` on the first generate and
  reused. **次の generate の常駐** sets the per-generate `residency` (`transformer` keeps the DiT;
  `per-stage` releases it).
- The pipeline and the dummy buffers share one GPU device, acquired with the applied GPU settings
  (geometry profile and timestamps) and the quant's features (`shader-f16`). The quant is fixed
  when the device is acquired (the first generate or **VRAM を埋める**) and locked while it is held;
  to switch, press **pipeline を破棄** and choose again.
- **VRAM を埋める** allocates the requested GiB as 1 GiB `STORAGE` buffers (or `maxBufferSize`
  pieces), each inside an `out-of-memory` error scope, and stops at the first out-of-memory error.
  **ダミーを解放** destroys them. **pipeline を破棄** disposes the pipeline, the dummies, and the
  device; use it after a device loss.
- Leave **steps** and **guidance scale** blank to use the distribution defaults. The default model
  runs with guidance scale 1, where a negative prompt is rejected.
- Each generate adds a table row: quant, requested residency, dummy GiB held, wall time (PNG
  encoding excluded), each stage's time, each stage's GPU time (expand for the top 10 keys), the
  geometry profile id the sessions reported (`SessionDiagnostics.geometryProfile`; one id when all
  stages agree, otherwise `component id` per stage), the residency events, the first 12 hex digits
  of the PNG's SHA-256, and the error. A failed generate stops the batch. The latest image is shown
  next to the table.
- **A/B（既定 vs 表）** runs a self A/B of the applied geometry profile with the current settings
  (quant, resolution, seed, prompt, steps, guidance scale, residency, and N): interval A injects
  `default` and runs N generates, then interval B uses the applied profile choice (`自動`,
  `builtin:<id>`, or a derived or saved table) and runs N generates. A table is fixed per device, so
  before each interval the pipeline and the device are disposed and acquired again with that
  interval's choice (the resident DiT is not carried over). Interval A's `default` is internal to
  the tab: the header selection and the applied settings do not change, and when the run ends the
  tab holds a device with the applied settings (if interval A fails, its device is disposed and the
  next action acquires one with the applied settings). The rows go into the same table
  (`geometryProfileRequested` tells the intervals apart); progress reads
  `A/B 区間 A（default）generate 2/2`. A failed row stops the run, and a failure in interval A skips
  interval B. When it ends, the status line and a summary table below the rows show whether the PNG
  SHA-256 agrees within interval A, within interval B, and between them (with the first 12 hex
  digits when it does not), and, for the whole generate (`wallMs`) and for each stage
  (`text_encoder`, `text_conditioner`, `transformer`, `vae_decoder`), the wall time and GPU time
  (`stages[].gpu.totalNs`; `—` without GPU time) of each interval and B ÷ A (below 1 means
  interval B is faster). Each interval's time is the median of its 2nd and later generates, because
  the 1st loads the DiT (with N = 1, the only generate). A failed interval shows `失敗 — <error name>`
  and no times. The button does nothing but explain on the status line when the applied profile is
  `default` (interval B would equal interval A) or dummy buffers are held (acquiring again would
  drop them in the middle of the comparison). The summary is not written to the JSON; the rows
  carry everything it is computed from.
- The line above the status shows the adapter, distribution, quant, GPU time, the requested profile
  (`幾何プロファイルの要求`), and after a generate the reported profile id(s).
- **JSON を保存** downloads `anima-residency-browser-<quant>-<timestamp>.json`.

GPU time splits every dispatch into its own compute pass, so wall times with GPU time on are not
comparable to wall times with it off.

## JSON formats

- **Sweep** — `karume-geometry-sweep/2`, as described in
  [../geometry-sweep/README.md](../geometry-sweep/README.md#output-karume-geometry-sweep2), with
  `settings.candidateSet`.
- **Anima** — `karume-anima-residency-browser/2`: adapter, user agent, checkout revision and dirty
  flag, bundle hash, distribution name, manifest SHA-256, `defaultModel`, `quant` (current or
  next), `geometryProfileRequested`, `geometryProfileInjected`, `gpuTiming: { enabled, feature, unit }`,
  `pipelineResidency`, `pipelineLoads[]` (`at`, `ms`, `quant`, `gpuTiming`), `dummies` (held bytes,
  buffers, every request), `deviceLost`, and `rows[]`. Each row has `index`, `quant`,
  `geometryProfileRequested`, `geometryProfileInjected`, `residencyRequested`, `request`, `dummyBytesHeld`, `wallMs`,
  `stages[]` (`component`, `startMs`, `endMs`, `geometryProfile`, `gpu: { totalNs, runs,
  clampedNegativeSamples, entries }` when GPU time is on, `pipelines`), `residency[]`, `pngSha256`,
  and `error`.
  - `geometryProfileRequested` is what the device was asked for: `auto` (no injection), `default`,
    `builtin:<id>` (a built-in table), or `generated:<id>`. The stage's `geometryProfile` is the id
    of the table the session used; it does not tell an injected table from a built-in one with the
    same id, so the two fields are read together. The field was added within version 2; older
    files lack it.
  - `geometryProfileInjected` is the injected table itself (the `GeometryProfile` value, as in
    the profile JSON below, with `1e999` for the last `maxRows`); it is absent with `auto`. A
    generated table keeps its id when it is derived again, so only this field tells which values
    a row ran with. The top-level field is the table of the current (or next) device. The Deno
    twin never injects, so its files lack it.
- **Profile** — the `GeometryProfile` value (`id`, `match`, `gemmRows`, `attention`, `conv2d`,
  `i8a8`, `provenance`), with `1e999` for the last `maxRows`.

## What to confirm

1. **The sweep runs with `quick+`.** On the sweep tab, keep all op families, press **開始**, and
   wait for `完了`. Every case block lists the geometries of the registered profiles, no row fails,
   and **出力の一致** is `一致` everywhere. Save the JSON to `outputs/bench-browser/`.
2. **The profile tab derives a table from it.** Press **表を作る** with the latest sweep checked. The
   table has ten fields; each adopted field shows a speed-up of at least ×1.05 in every case. With a
   sweep taken without the developer features flag (quantized), the status line shows the rejection
   instead.
3. **The injected table is used.** Press **この表を適用**. The environment line shows
   `generated:<id>（注入）`. On the Anima tab, run 2 generates (512x512, seed 42). The
   **幾何プロファイル** column shows `<id>` for every stage, and the saved JSON has
   `geometryProfileRequested: "generated:<id>"` in each row.
4. **The table does not change the image.** Apply `自動`, run 2 generates with the same prompt, seed,
   and resolution, and compare: the PNG SHA-256 is the same as with the injected table (tile
   geometry does not change the result bits — ADR 0115), and the DiT stage time shows the effect
   of the table.
   **A/B（既定 vs 表）** does the same comparison against `default` in one press: with the table
   applied and N = 2, the summary shows `一致` within interval A, within interval B, and
   `A と B が一致`, and the `transformer` rows give the DiT stage's B ÷ A.
5. **Built-in tables can be injected on any GPU.** Apply `apple-metal-3` or `nvidia-blackwell` on a
   machine they do not match. The Anima rows report that id, and the PNG SHA-256 is still the same.
6. **Automatic selection picks the expected table.** With `自動`, the Anima rows report
   `apple-metal-3` on an Apple M2 in Chrome, `nvidia-blackwell` on an RTX 5070 Ti, and `default` on
   any other GPU.
7. **Residency (ADR 0112).** With residency `transformer`, the 2nd and later generates are faster
   than the 1st and list `retained/request`; a `per-stage` generate after a resident one lists
   `released/request`. After **VRAM を埋める** with enough GiB that a text or VAE stage no longer fits
   next to the DiT (start with 6 GiB on a 10 GiB GPU), one generate lists `evicted/headroom` (or
   the reactive `evicted/out-of-memory`) and still produces the same PNG SHA-256; later rows list
   `released/downgraded`.

## Caveats

- The sweep and Anima use separate GPU devices. A sweep does not release the Anima pipeline or its
  dummies, so on a GPU with little memory press **pipeline を破棄** before a large sweep.
- Chrome quantizes timestamps to 100 µs unless the developer features flag is enabled; the sweep
  JSON then has `gpuTiming.quantized: true`, and an Anima stage summary says `100 µs 量子化の疑い`.
- If a device is lost with timestamps on, clear **GPU の timestamp で測る**, press **適用**, and run
  again (the sweep then uses the wall clock; Anima records stage wall times and dispatch counts
  only).
- The TS shown on the profile tab is not formatted; register tables with the command, which formats
  them, so that `deno fmt --check` and `main.ts profile --check` pass.
- The pre-emptive residency check costs up to two trial allocations per generate (about 0.1 s;
  `docs/research/2026-09-27-h35-oom-device-lost.md`).

## Deno twins

The same measurements run without Chrome, and write the same JSON formats:

- The sweep: `deno run -A tools/geometry-sweep/main.ts [--set quick|quick+|full] [--op …] [--case …]`
  (see [../geometry-sweep/README.md](../geometry-sweep/README.md); `userAgent` is `{ deno }` and the
  time unit is `deno-raw-tick`).
- The Anima profiler, once per quant and resolution:

  ```sh
  deno run -A tools/anima-residency/profile.ts --resolution 512x512 --count 2 --quant f16
  deno run -A tools/anima-residency/profile.ts --resolution 512x512 --count 2
  ```

  Flags: `--source` (default `models/karume-anima`), `--quant` (default: the distribution's
  `defaultQuant`), `--resolution` (default `1024x1024`), `--count` (default 2), `--seed` (default
  42), `--prompt` (default: the page's prompt), `--out` (default
  `outputs/bench/karume/<date>_metal-recon/`), and `--date` (the date in the default `--out`). It
  builds the pipeline once with `residency: "transformer"`, prints each stage's wall and GPU time,
  and writes `anima-profile-<adapter>-<quant>-<WxH>-<ISO time>.json` in the page's format
  (`userAgent` is `{ deno }`, no `bundleSha256`, no dummy buffers, `gpuTiming.unit` is
  `deno-raw-tick` — Deno returns raw GPU ticks, 1 tick = 52.08 ns on the Arc B570 — and
  `geometryProfileRequested` is always `auto`). Deno on Metal loses the device when timing is on
  (`docs/limitations.md`).

# GPU lab

A local Chrome (WebGPU) page that runs the whole geometry-profile loop on one machine, as a proof of
concept for ADR 0115 (addendum decision 6):

1. **Benchmark** — sweep GEMM tile geometries on this GPU (the **1. 掃引** tab).
2. **Optimize** — turn the sweep into a geometry profile, a static table of tile geometries for one
   adapter (the **2. プロファイル** tab).
3. **Run** — inject the table with `acquireGpu({ geometryProfile })` and run Anima on it (the
   **3. Anima** tab).

A fourth tab, **4. Wan**, runs the Wan2.1 T2V 1.3B and Wan2.2 TI2V 5B text-to-video pipelines in Chrome
(ADR 0118 stage 9, ADR 0121 stage 8; see [4. Wan](#4-wan) and
[Checking Wan in Chrome](#checking-wan-in-chrome)).

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

The Wan distribution is served the same way from `models/karume-wan2.1` under `/models/wan/…`
(`/config.json` names it in `wanSource`). `--wan-source <path>` overrides the directory; the same
rules apply: an explicit `--wan-source` without `karume.json` stops the server with
`--wan-source <path> has no karume.json`, and a missing default only prints a warning, sets
`wanSource: null`, and answers 404 under `/models/wan/`:

```sh
deno task bench:gpu-lab --wan-source /path/to/karume-wan2.1
```

The Wan2.2 distribution follows the same rules from `models/karume-wan2.2` under `/models/wan22/…`
(`/config.json` names it in `wan22Source`; `--wan22-source <path>` overrides the directory):

```sh
deno task bench:gpu-lab --wan22-source /path/to/karume-wan2.2
```

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
   may round timestamps to 100 µs (observed not to on macOS Chrome 154 — the record's
   `gpuTiming.quantized` tells); when it does, the sweep still runs and its JSON is marked `quantized`. The
   profile tab accepts it, but leaves out of the speed-ups every row whose rounding error bound
   exceeds 1%, and lists those rows with the bound in the verdicts. With the flag the timestamps are
   not rounded, so no row is left out for rounding.
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
    (`karume-gpu-lab/last-generated-profile/2`, overwriting the previous one, with the save time and
    the checkout revision; the table itself is stored as the text of `geometryProfileJson`), and the
    page offers it again when it opens, because Chrome sometimes has to be reloaded and the derived
    table would otherwise be lost. `<adapter>` is the adapter description, or vendor/architecture
    when the description is empty, read from the table's `provenance.adapter` (the adapter the
    sweeps ran on — it is not stored a second time). A saved value that `parseGeometryProfileJson`
    rejects (including the runtime's profile check), or whose table has no `provenance`, is
    deleted, and the GPU settings status line says so. A table saved under the earlier `/1` key is
    not read. The applied settings themselves are not saved. Both kinds are recorded as
    `generated:<id>`.
  - `保存した表（照合して注入 — 一致しなければ自動）` — the application flow of
    `@karume/runtime/tune` instead of a plain injection (see
    [Stored table, matched and injected](#stored-table-matched-and-injected)). It is disabled while
    `localStorage` holds no saved table. It is recorded as `saved:<id>` (`saved:(読めない)` when
    the saved value cannot be read), whether or not the table was injected.

  **適用** disposes the Anima pipeline, its dummy buffers, and its GPU device; if the Anima tab held a
  device, it acquires a new one with the new settings at once, and the Anima status line says
  `GPU を取り直しました（quant <quant>・幾何プロファイル <requested>）`. It also disposes the Wan pipeline and
  its device, which the Wan tab acquires again on the next **読み込む**. The timestamp setting applies
  to the next sweep and to the Anima device (the Wan tab never asks for timestamps).
- **Environment line** — the adapter, whether GPU time is taken, the applied geometry profile
  (`自動 → <id>`, `<requested>（注入）` where `<requested>` is `default`, `builtin:<id>`, or
  `generated:<id>`, or `saved:<id>（GPU を取るときに adapter と照合 — …）`), and the checkout revision. On the sweep tab it adds that
  the sweep measures explicit geometries, so the profile does not affect its results.

### Stored table, matched and injected

The `保存した表（照合して注入）` choice runs the flow an application uses to apply a stored profile
(ADR 0117): sweep → derive → save → restart → match → inject through the callback form of
`acquireGpu({ geometryProfile })`.

1. **適用** reads the saved value from `localStorage` (the text the profile tab wrote; after a reload
   this is the table saved before it). The text is read here, before any GPU is acquired, because
   the callback must not wait for I/O.
2. When the Anima tab acquires its device, it passes a callback. The runtime calls it with the
   information of the adapter it actually got; the callback parses the saved text
   (`parseGeometryProfileJson`) and calls `geometryProfileMismatch(profile, adapterInfo)`.
3. On a match the callback returns the table and the runtime injects it: the rows report its id in
   the geometry profile column and carry it in `geometryProfileInjected`. On a mismatch it returns
   `undefined`, the runtime selects a profile from the adapter as with `自動`, the rows report that
   id and have no `geometryProfileInjected`, and the status line and the line above it say
   `保存した表を注入しない（自動で選ぶ）— <reason>` (for example
   `表 '<id>' の adapter の description が違う（…）`). A saved value that cannot be read is
   deleted from `localStorage` and the run uses automatic selection, with the reason shown the same
   way. On a match they say `保存した表 <id> は adapter と一致 → 注入`.

Deriving a new table after applying this choice marks the settings as pending: apply again to
use the new saved text.

Only one GPU operation runs at a time: a sweep, an Anima action (generate, fill VRAM, release,
dispose), a Wan action (load, generate, dispose), or applying GPU settings. Starting another while one runs is refused with a message; two
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
- The JSON records when the sweep started (`startedAt`), whether **中断** stopped it (`aborted`), and
  each case's wall clock (`cases[].elapsedMs`).
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
  it is added: a sweep without GPU timestamps (unit `wall` or no `gpuTiming`) is rejected, and
  the status line shows the generator's reason as is. A sweep with quantized timestamps is accepted;
  rows whose rounding error bound exceeds 1% are left out of the speed-ups and listed. The SHA-256 is
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
  (resolution, seed, prompt, steps, guidance scale, residency, and N): interval A injects
  `default` and runs N generates, then interval B uses the applied profile choice (`自動`,
  `builtin:<id>`, or a derived or saved table) and runs N generates. It runs this A/B for two
  quants (quant outer, interval inner): the distribution's default quant, then `f16` (the
  unquantized one; just once when the default is `f16`), so N = 3 gives 12 rows. The quant select
  is not consulted, and the press fails before acquiring a GPU when either quant is not in the
  distribution. A table is fixed per device, and so is the quant, so before each interval the
  pipeline and the device are disposed and acquired again with that interval's choice and quant
  (the resident DiT is not carried over). Interval A's `default` and the quant being run are
  internal to the tab: the header
  selection, the applied settings, and the quant select do not change, and when the run ends the
  tab holds a device with the applied settings and the selected quant (otherwise its device is
  disposed and the next action acquires one with them). The rows go into the same table
  (`quant` and `geometryProfileRequested` tell the runs apart); progress reads
  `A/B f16・区間 A（default）generate 2/3`. A failed row stops that quant's A/B, and a failure in
  interval A skips that quant's interval B; the next quant still runs. When it ends, the status line
  (one phrase per quant: `A/B 完了 — <quant>: <sha verdicts> · B ÷ A: <transformer ratios> / …`,
  with the whole-generate ratio too when only one quant ran) and one summary table per quant below
  the rows (its caption names the quant) show whether the PNG SHA-256 agrees within interval A, within interval B, and between them (with the first 12 hex
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

## 4. Wan

Runs Wan2.1 T2V 1.3B (`WanPipeline` of `@karume/models/wan`) in Chrome, to see whether a clip
completes on a browser device and, when it does not, where it stops (binding limits, the GPU
timeout, memory). The pipeline's rules are those of `examples/wan`: only the four prompts of the
embedding asset, 832x480 or 480x832, 4n+1 frames from 5 to 81 (Wan2.2: the same four prompts, and the
sizes and frame counts below).

The tab has a generation switch next to the source field: `Wan2.1 T2V 1.3B（karume-wan2.1）` (the default) or
`Wan2.2 TI2V 5B（karume-wan2.2）` (text-to-video only). It can be changed only before loading. Changing it rebuilds
the frame and size choices, the limits table, the quant choices (from that generation's distribution on this
server), the source placeholder, and the info line. Wan2.1 loads with `WanPipeline.fromPretrained` exactly as
before. Wan2.2 accepts 1280x704 or 704x1280 and 4n+1 frames from 5 to 49 in the product (default 1280x704, 33
frames), and its clips play at 24 fps. The page offers Wan2.2 up to 121 frames, the official default length: it
calls the family's internal loader and generator (`loadWanFromPretrained` / `generateWanVideo` of
`packages/models/src/wan/family.ts`) with a copy of the descriptor whose frame limit is 121. Frame counts above 49
are marked `製品の受理の外 — 開発機の sha 行を持たない` in the frame select: the product rejects them, and the
development machine keeps no reference rows for them (see [Wan2.2](#7-wan22-ti2v-5b)). The default frame count
and size are each generation's descriptor defaults.

The tab has a text-encoder route switch. `precomputed` (the tab default) uses the fixed-prompt embedding
asset; `gpu` runs umT5-XXL on the GPU and accepts any prompt — it needs the `karume-umt5-xxl` mirror,
which the server serves at `/models/umt5/` (`--umt5-source`, default `models/karume-umt5-xxl`). A quant
switch next to it lists the quants of the distribution this server serves; its first entry, `既定`,
resolves to the manifest's `defaultQuant` at load (for Wan2.1, `f16+dit8-a8-attn8-s16` since 2026-10-04 —
ADR 0120), and the resolved name is passed to the pipeline and recorded. The saved JSON is
`karume-wan-browser/4` and carries the generation (`wan2.1` / `wan2.2`), the route, and the quant.

- **取得元** (source) — blank reads the distribution this server serves for the chosen generation
  (`models/karume-wan2.1`, or `models/karume-wan2.2` with Wan2.2); an
  `owner/name` reads that Hugging Face repository at `main` (the distribution is not published
  yet, so the blank default is the normal case).
- **読み込む** (load) reads the manifest, acquires a GPU device, and builds the pipeline with
  `WanPipeline.fromPretrained(source, { gpu, quant, textEncoder })`. The device is acquired by the tab with the applied
  geometry profile of the GPU settings and without GPU time: `WanPipeline` refuses a timing device
  (the VAE decodes one tile per batch, and the runtime opens no batch on a timing device), so the
  timestamp setting does not apply to this tab. `acquireGpu` requests the adapter's own limits, so
  the device does not keep the WebGPU default of 128 MiB per storage binding. From this server, loading
  reads only the descriptors, and every generate reads the weights from the server again as each
  stage builds its session (about 2.6 GiB for the `f16` transformer and 0.27 GiB for the VAE); from
  Hugging Face, loading first downloads the weight parts into the browser cache. **pipeline を破棄** disposes the pipeline and
  the device; use it after a device loss. Applying the GPU settings also disposes them; load again
  to use the new settings.
- **The limits table** lists every limit `acquireGpu` requests, with the adapter's value and, once
  loaded, the device's. Two rows are judged for the selected frame count and size (red when short):
  `maxStorageBufferBindingSize` and `maxBufferSize` must hold the largest value the runtime cannot
  split (see [binding limits](#1-binding-limits)). The line above the table gives the verdict and
  the largest frame count those two limits allow at the selected size. The other rows are
  informational. Passing the table is necessary, not sufficient: memory and submit times are
  checked only by running.
- **Prompt, negative, seed, frames, size, steps, guidance, shift** — leave steps, guidance, and shift
  blank for the distribution defaults (50, 5.0, 3.0 for Wan2.1 and 50, 5.0, 5.0 for Wan2.2; the
  placeholders show them after loading). A
  blank negative uses the asset's `negative` row. **生成** (generate) reads them once.
- Each generate adds a row: the resolved request, the wall time, each stage's time
  (`transformer`, `vae_decoder`), the step times and the VAE tile times (the first one includes
  building that stage's sessions, i.e. uploading the weights; then the median and maximum of the
  rest — a step is two transformer forwards with CFG), the session diagnostics of each component
  (weights and slot backing allocated, geometry profile, submit count, `窓平均の最大` = the largest
  window mean, a lower bound of the longest submit, and the count of chunks over the time budget),
  the SHA-256 of the RGB bytes (every frame's 8-bit RGB from `wanFrameToRgba`, concatenated in frame
  order — the bytes the e2e reference rows hash), the reference verdict, and the error. The clip is
  drawn on the canvas next to the table: **前** / **次** step one frame, **再生** plays at the clip's
  frame rate (16 fps for Wan2.1, 24 fps for Wan2.2), and the slider seeks. The condition column starts
  with the generation.
- **JSON を保存** downloads `wan-browser-<timestamp>.json` (`karume-wan-browser/4`; the generation is in
  each `loads[]` entry, in each row, and next to the loaded source).

There is no way to stop a generate from the page (the pipeline has no `signal` yet); closing the tab
stops it. On the Intel Arc B570 under Deno, a 50-step clip with the `f16` quant takes about 30
minutes for 33 frames and about 2 hours for 81 frames; with `f16+dit8-a8-attn8-s16` it takes about
16 minutes for 33 frames and about 62 minutes for 81 frames (3,703 s, of which the transformer stage
is 3,383 s).

## Checking Wan in Chrome

The target of ADR 0118 stage 9 is a complete clip in Chrome on the user's machines (an RTX 5070 Ti
first: Chrome reports it as vendor `nvidia`, architecture `blackwell`). The numbers below come
from the B570 runs of stages 6 to 8 under Deno.

**Result on the RTX 5070 Ti (2026-10-04, Chrome 154 on Windows)**: a 50-step clip with the `f16`
quant and the precomputed embeddings (`boxing-cats`, seed 42, 832x480, 81 frames) completed without
a device loss in 2,767.9 s (46.1 minutes): 2,686.2 s for the transformer stage and 81.4 s for the
VAE stage. The adapter and the device both reported a `maxStorageBufferBindingSize` of
2,147,483,644 B, so the 81-frame FFN intermediate fit without splitting. The largest window mean was
102.3 ms for the transformer and 231.1 ms for `vae_decoder_first`, far from the watchdog. VRAM
peaked at about 8 GB (watched by eye — not in the record). The GPU text encoder and the int8 quants
have not been run in Chrome yet. No reference row was added for this key: a row that needs a
46-minute rerun cannot catch regressions, so a Chrome row, when one is needed, comes from a 2-step
case.

### 1. Binding limits

Chrome gives a device 128 MiB per storage binding unless more is requested; `acquireGpu` requests
the adapter's value. What a clip needs is the largest value the runtime cannot split:

| Value                                         | 33 frames               | 81 frames                  |
| --------------------------------------------- | ----------------------- | -------------------------- |
| Transformer FFN intermediate `[1,S,8960]` f32 | 503,193,600 B (480 MiB) | 1,174,118,400 B (1.09 GiB) |
| VAE intermediate (`vae_decoder_next`) f32     | 201,326,592 B (192 MiB) | 201,326,592 B (192 MiB)    |

S is the token count (14,040 at 33 frames, 32,760 at 81). So no clip runs at the 128 MiB default,
33 frames need at least 503,193,600 B, and 81 frames at least 1,174,118,400 B, for both
`maxStorageBufferBindingSize` and `maxBufferSize`.

The self-attention scores are not such a value: the attention op splits its query rows into
blocks that each fit the binding limit (ADR 0060). On the B570 (limit 2,147,483,644 B) that is 5
blocks of 1,892,367,360 B at 33 frames and 24 blocks of 2,146,435,200 B at 81 frames; a smaller limit
gives more, smaller blocks, not a failure, so an adapter below 2,146,435,200 B can still run 81
frames. The limits table shows the block count for the selected request.

### 2. GPU timeout (TDR and the watchdog)

- Windows resets a GPU job that runs longer than 2 s by default (TDR); Chrome has its own GPU
  watchdog (about 10 s, from a secondary source). The runtime cuts submits by a time budget, and the
  longest single submit measured on the B570 was 458.6 ms at 81 frames — but that was in the timing
  mode, which splits passes differently; the normal mode the page uses has not been measured. The
  RTX 5070 Ti is expected to be faster per submit (an expectation, not a measurement).
- The diagnostics column gives `窓平均の最大` per component: a lower bound of the longest submit. A
  value near or above 2,000 ms on Windows is a warning even when the run completes.
- A timeout shows as a device loss: the status line and the row say
  `GPU device lost（<reason>）: <message>`, and Chrome may also print `D3D12` or device-removed
  errors in the console. On Windows, a TDR also leaves a `Display` event 4101 in Event Viewer
  (Windows Logs → System). Record all three, then press **pipeline を破棄** before trying again.

### 3. Memory

On the B570 the transformer stage peaked at 6.19 GiB (33 frames) and 7.31 GiB (81 frames), the VAE
stage at 3.32 and 3.78 GiB, and the two stages never hold memory at the same time. A 16 GB card is
expected to fit both, but Chrome's own allocations on top of that have not been measured. The page
cannot read VRAM: watch it from outside (`nvidia-smi --query-gpu=memory.used --format=csv -l 1`, or
the dedicated GPU memory graph in the Windows Task Manager) and note the peak of each stage. The
diagnostics column gives the weights and slot backing of each session (on the B570 at 81 frames:
2.645 + 3.517 GiB for the transformer), a lower bound that leaves out the VAE caches (about
0.58 GiB).

### 4. The single ArrayBuffer limit

Chromium refuses any single ArrayBuffer above 2,145,386,496 bytes (`docs/limitations.md`). No host
buffer of a Wan run comes near it: the largest weight part is 264,705,024 B and parts are read one at
a time, the text embedding asset is 5,280,288 B, the f32 clip is 388,177,920 B at 81 frames, its RGB
bytes 97,044,480 B, and the largest VAE cache 50,331,648 B. The 2 GiB attention blocks live on the
GPU, not in an ArrayBuffer.

### 5. Environment keys and reference SHA-256

Reference hashes are kept per environment (ADR 0106) in
`packages/models/tests/fixtures/references/wan.json`. The page has no `KARUME_REFERENCE=write`:
it shows the environment key (on the line above the status), and, when the request is one of the
reference cases, the case id and whether this environment's row matches. When there is no row, report
the key, the case id, and the SHA-256; the row is then added to the file. The key is
`chrome-<vendor>-<architecture>` (`chrome-nvidia-blackwell` on the RTX 5070 Ti) without the
developer features flag, and is built from the adapter description with the flag, so the same
machine has two keys; keep the flag setting the same across runs. Rows of another key are never
compared (bit equality across machines is not guaranteed).

The reference cases are the e2e cases: `boxing-cats`, seed 42, the default negative, the default
guidance and shift, 832x480, and

- `2step-seed-boxing-cats-seed42` — steps 2, 33 frames (a quick run),
- `50step-boxing-cats-seed42` — steps 50, 33 frames,
- `50step-boxing-cats-seed42-81f` — steps 50, 81 frames.

These ids are the `f16` quant's. The id also depends on the quant that ran: the int8 quants put their
name in front (`f16+dit8-a8-attn8-s16-2step-seed-boxing-cats-seed42`), for the quants the e2e keeps rows
for — `f16+dit8` and `f16+dit8-a8-attn8-s16` at 2 steps, and `f16+dit8-a8-attn8-s16` at 50 steps.
Any other quant is not a reference case.

### 6. What to record

1. The adapter line (vendor, architecture, description), the environment key, and the user agent.
2. The limits table (adapter and device values) and its verdict line.
3. For each run: the row (stage, step, and tile times; diagnostics; SHA-256; reference verdict).
4. The VRAM peak of each stage, if you can watch it.
5. On a failure: the status line, the row's error, the console errors (copy them), and on Windows
   the Event Viewer entry.
6. **JSON を保存** after the last run (it holds all of the above that the page sees).

### Steps

1. Start the server on the machine with the GPU, with the checkout and `models/karume-wan2.1`
   (through port forwarding every generate reads the 2.9 GiB of weights through the tunnel again).
2. Open **http://localhost:8790/#wan** in Chrome and note whether the developer features flag is on.
3. Before loading, read the limits table at 33 and at 81 frames.
4. Press **読み込む**. The device column fills in; it should equal the adapter column.
5. Run the quick reference case: `boxing-cats`, seed 42, 33 frames, 832x480, steps 2, guidance and
   shift blank. Check that the clip plays and note the SHA-256. The quant is the one chosen before
   loading (`既定` = the manifest's `defaultQuant`); to compare with the `f16` rows, choose `f16`.
6. Run `50step-boxing-cats-seed42` (steps blank), then, if it completes, 81 frames.
7. Save the JSON and report it with the records above.

### 7. Wan2.2 (TI2V 5B)

Sections 1 to 6 and the steps above describe Wan2.1; this section gives what differs for Wan2.2. The
numbers marked as measured come from the B570 under Deno (ADR 0121, "B570 の 1280×704 のフレーム数の試走" and
stage 2); nothing has run in Chrome yet.

**Binding limits.** S is 7,920 tokens at 1280x704 and 33 frames, 11,440 at 49 frames, and 27,280 at 121
frames (32 pixels per token on each side):

| Value                                          | 33 frames               | 49 frames               | 121 frames                 |
| ---------------------------------------------- | ----------------------- | ----------------------- | -------------------------- |
| Transformer FFN intermediate `[1,S,14336]` f32 | 454,164,480 B (433 MiB) | 656,015,360 B (626 MiB) | 1,564,344,320 B (1.46 GiB) |
| VAE intermediate (`vae_decoder_next`) f32      | 201,326,592 B (192 MiB) | 201,326,592 B (192 MiB) | 201,326,592 B (192 MiB)    |

The self-attention scores of the 24 heads split into 3 blocks at 33 frames on the B570 limit (measured
in stage 2), and into 34 blocks of 2,102,960,640 B at 121 frames (computed).

**Memory and time (measured on the B570, 1280x704, 50 steps).**

| Quant                   | Frames | Transformer stage peak | Transformer diagnostics | Time (transformer / VAE)      |
| ----------------------- | -----: | ---------------------: | ----------------------: | ----------------------------- |
| `f16+dit8-a8-attn8-s16` |     33 |               7.72 GiB |                7.22 GiB | 23 min 40 s (967 s / 446 s)   |
| `f16+dit8-a8-attn8-s16` |     49 |               8.49 GiB |                7.93 GiB | 36 min 5 s (1,504 s / 655 s)  |
| `f16+dit8-a8-attn8-s16` |     57 |               8.74 GiB |                8.13 GiB | 43 min 19 s (1,834 s / 756 s) |
| `f16+dit8`              |     33 |               7.86 GiB |                7.35 GiB | 43 min 3 s (2,143 s / 439 s)  |
| `f16+dit8`              |     49 |               8.45 GiB |                7.64 GiB | 66 min 53 s (3,360 s / 647 s) |

The VAE stage peaked at 4.36 to 4.52 GiB with the practical quant, at any frame count. 121 frames has not
run anywhere. The ADR 0121 capacity table estimates (extrapolated from Wan2.1, before any measurement)
transformer diagnostics of 9.16 to 9.50 GiB with `f16+dit8` and 10.18 to 11.37 GiB with
`f16+dit8-a8-attn8-s16` at 1280x704 and 121 frames. With about 1.3 GiB on top, the practical quant is
expected to fit the RTX 5070 Ti (16 GB) at about 11.5 to 12.7 GiB. That is an expectation, not a
measurement. The same table estimates the transformer stage at about 78 minutes on the RTX with `f16+dit8`.
If the practical quant halves that as it does on the B570, it takes about 40 minutes. Both are estimates.

**Reference cases.** The rows are kept in `packages/models/tests/fixtures/references/wan-ti2v.json`, with
the same environment keys as Wan2.1. The common conditions are `boxing-cats`, seed 42, the default
negative, and the default guidance and shift (5.0 and 5.0):

- `f16+dit8-2step-boxing-cats-seed42-1280x704-17f-shift5` and
  `f16+dit8-2step-boxing-cats-seed42-704x1280-17f-shift5` — steps 2, 17 frames, only with `f16+dit8`;
- `<quant>-50step-boxing-cats-seed42-1280x704-33f-shift5` — steps 50, 33 frames, 1280x704, with `f16+dit8`
  or `f16+dit8-a8-attn8-s16`. The e2e observes these without keeping rows for now (ADR 0121 stage 7 decides
  them), and no row is added for a Chrome key (ADR 0121 stage 10), so the page reports a missing row: report
  the SHA-256 as a record only, unlike section 5.

Any other request, quant, or route is not a reference case.

**Steps for Wan2.2.**

1. Start the server with the Wan2.2 distribution: `deno task bench:gpu-lab --wan22-source
   models/karume-wan2.2` (without the flag, the default location is used when it holds `karume.json`).
2. Open **http://localhost:8790/#wan** in Chrome and choose `Wan2.2 TI2V 5B（karume-wan2.2）` in **世代**.
3. Choose the quant (it holds until **pipeline を破棄**). `既定` resolves to the manifest's `defaultQuant`.
   The 2-step reference case has rows only for `f16+dit8`, so choose `f16+dit8` explicitly for it, even when
   `既定` resolves to the practical quant. The practical quant `f16+dit8-a8-attn8-s16` takes about half the
   transformer time of `f16+dit8` on the B570.
4. Before loading, read the limits table at 33, 49, and 121 frames (1280x704).
5. Press **読み込む**. Run the 2-step reference case first: `boxing-cats`, seed 42, 17 frames, 1280x704,
   steps 2, guidance and shift blank, loaded with `f16+dit8` (step 3). Note the SHA-256.
6. Run the 50-step clip at 33 frames (steps blank). If it completes, run 49 frames, then 121 frames.
7. Watch VRAM from outside (`nvidia-smi --query-gpu=memory.used --format=csv -l 1`, or the Windows Task
   Manager) and note the peak of each stage.
8. Save the JSON and report it with the records of [6. What to record](#6-what-to-record).

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
- **Wan** — `karume-wan-browser/4`, as described in [4. Wan](#4-wan): the adapter, the environment
  key, the adapter and device limits, the loaded source / generation / quant / route, `loads[]`,
  `deviceLost`, and `rows[]` (each with the generation, the resolved request, the wall time, the
  stage / step / tile times, the session diagnostics, the RGB SHA-256, the reference verdict, and
  the error).

## What to confirm

1. **The sweep runs with `quick+`.** On the sweep tab, keep all op families, press **開始**, and
   wait for `完了`. Every case block lists the geometries of the registered profiles, no row fails,
   and **出力の一致** is `一致` everywhere. Save the JSON to `outputs/bench-browser/`.
2. **The profile tab derives a table from it.** Press **表を作る** with the latest sweep checked. The
   table has ten fields; each adopted field shows a speed-up of at least ×1.05 in every case. With a
   sweep taken without the developer features flag (quantized), the table is still made, and each
   row left out for its rounding error is listed under its field.
3. **The injected table is used.** Press **この表を適用**. The environment line shows
   `generated:<id>（注入）`. On the Anima tab, run 2 generates (512x512, seed 42). The
   **幾何プロファイル** column shows `<id>` for every stage, and the saved JSON has
   `geometryProfileRequested: "generated:<id>"` in each row.
4. **The table does not change the image.** Apply `自動`, run 2 generates with the same prompt, seed,
   and resolution, and compare: the PNG SHA-256 is the same as with the injected table (tile
   geometry does not change the result bits — ADR 0115), and the DiT stage time shows the effect
   of the table.
   **A/B（既定 vs 表）** does the same comparison against `default` in one press: with the table
   applied and N = 3, the table gains 12 rows (3 × 2 intervals × 2 quants: the default quant and
   `f16`), and one summary per quant shows `一致` within interval A,
   within interval B, and `A と B が一致`, with the `transformer` rows giving the DiT stage's
   B ÷ A. The quant select and the header still show what they showed before the press.
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
- Chrome may quantize timestamps to 100 µs without the developer features flag (not observed on
  macOS Chrome 154); the sweep JSON then has `gpuTiming.quantized: true`, and an Anima stage summary
  says `100 µs 量子化の疑い`.
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

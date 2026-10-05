# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The three JSR packages (`@karume/runtime`, `@karume/hub`, `@karume/models`) and the exporter are
versioned in lockstep: one version number covers all of them. While the project is at `0.x`,
breaking changes may land in a minor or patch release; every one of them is listed under
**Breaking** below.

Entries describe user-visible changes only. Design rationale lives in `docs/decisions/`,
measurements in `docs/research/`.

## [Unreleased]

### Added

- `SessionDiagnostics.lastRunPipelines`: per-pipeline-key dispatch counts of the last run as
  planned, filled with or without `gpuTiming` and on the resident path (`enqueue`) too; the key
  spelling is a diagnostic label outside semver.
- `AnimaPipelineOptions` accepts `linearCompute`, `attentionCompute` and `attentionScoreStorage`, and
  `IrodoriPipelineOptions` accepts `linearCompute`; in both families they apply to the DiT
  session only (text encoders and the VAE keep the quant declaration). An explicit value wins over
  the quant's `session` declaration, which wins over the runtime default; invalid values and
  combinations the runtime rejects throw `ModelInputError` before any weight bytes are fetched, and
  an `"f16"` choice adds `shader-f16` to the GPU requirements.
- `sessionOptionsViolation(options)` (`@karume/runtime`): the GPU-independent acceptance check of
  `SessionOptions` (types, spellings, combinations, ranges) that session construction applies,
  returned as a message so callers can reject options before loading weights.
- `fitsHeadroom(gpu, bytes)` (`@karume/runtime`): reports whether `bytes` can be allocated on the
  device right now. It makes a trial `createBuffer` allocation (the one out-of-memory path WebGPU
  guarantees not to lose the device), destroys it and waits for the release before answering
  `true` or `false`. The answer is a point-in-time fact, not a reservation; validation errors and
  device loss are thrown.
- `@karume/runtime/tune`: a second entry point of `@karume/runtime` for tuning the GEMM tile
  geometry inside an application. `runGeometrySweep(options)` acquires a dedicated GPU device, sweeps
  the candidate geometries (`quick`, `quick+` or `full`, optionally narrowed by op or case, with
  progress callbacks and an `AbortSignal`) and returns the sweep record
  (`karume-geometry-sweep/2`); `deriveGeometryProfile(reports, options)` derives a `GeometryProfile`
  and the per-field verdicts from one or more records with the same rules as the repository's
  profile generator; `geometryProfileJson(profile)` serializes a profile (`Infinity` as `1e999`) for
  storage and `acquireGpu({ geometryProfile })`. `deriveGeometryProfile` also accepts sweeps
  measured with Chrome's 100 µs-rounded timestamps (no developer flag): instead of rejecting them,
  it bounds the rounding error of every observation and drops from the ratio material only those
  whose bound exceeds 1% (they are listed in the verdicts); wall-clock-only sweeps are still
  rejected. The main entry (`@karume/runtime`) is unchanged.
- `@karume/runtime/tune` can tell whether a stored profile may still be used.
  `parseGeometryProfileJson(text)` reads the text of `geometryProfileJson` back into a
  `GeometryProfile`; it rejects unknown fields at every level, a last `maxRows` that lost its
  `Infinity` (plain `JSON.stringify` writes `null`), and anything the injection check rejects, with
  a `GeometryProfileParseError` that names the field. `geometryProfileMismatch(profile, adapterInfo)`
  returns `undefined` when the profile matches, otherwise the first mismatch as a message: the four
  adapter fields the profile was swept on (`vendor`, `architecture`, `device`, `description`, empty
  strings included), the kernel fingerprint (a hash of the pipeline keys, params, WGSL and
  workgroup counts that the profile and the default geometry give for the sweep's shapes, which the
  function derives again from the profile without a GPU) and the case-set id (`sweepCaseSetId()`,
  a hash of the sweep's cases and `gemmRows` bounds that leaves out the cases' descriptions). A
  profile derived from sweeps carries these in `GeometryProfile.provenance` (`candidateSet`,
  `adapter` as the four fields, `kernels` and `caseSet`, next to `sweep`, `sha256` and `date`), and
  `deriveGeometryProfile` requires all its records to agree on the four adapter fields. A runtime
  update that changes those kernels or the sweep's cases makes stored profiles mismatch: sweep
  again. Injection (`acquireGpu({ geometryProfile })`) still ignores `provenance`.
- The sweep record (`karume-geometry-sweep/2`, unchanged format) gains `startedAt`, `aborted`,
  `cases[].elapsedMs`, `caseSet` (the case-set id) and `defaultKernels` (the kernel fingerprint of
  the default profile). `deriveGeometryProfile` refuses a record whose `caseSet` or
  `defaultKernels` differs from the current runtime's; records without these fields are accepted
  as before. A derived profile's `provenance.userAgent` lists the records' `userAgent` values as an
  array of strings, distinct and in order of appearance (`{ deno }` becomes `Deno/<version>`; it is
  left out when any record lacks `userAgent`); it is informational and not part of
  `geometryProfileMismatch`.
- The sweep record also gains `candidateKernels`, a fingerprint of the kernels behind its measured
  rows (every row without `error`, rebuilt from its case, geometry and the record's `dp4a`).
  `deriveGeometryProfile` derives it again from the record's rows and refuses the record when it
  differs, so a runtime that changed only the candidate kernels (not the default ones that
  `defaultKernels` covers) no longer builds a profile from outdated speedups and output matches;
  records without the field are accepted as before.

- Same-machine A/B gates for the practical quant seats of anima (default seat), irodori `i8-a8` and
  sbv2 `i8-a8` (ADR 0110 decision 5): the reference seat is derived from the manifest (same weights,
  empty `session`), the observation point is where the seat first takes effect (step-1 latent / sbv2
  front outputs), and the gate checks determinism, census and a declared relRMS band. `results.json`
  gains the optional `comparisons` field (schema unchanged) and `tools/verify-diff` renders it.
- BiRefNet recipe: an `f16` weight series (`python -m birefnet.export --dtype f16`, exclusive with
  `--verify`) and an `f16` quant seat next to `f32` in the `birefnet` / `lucida` distributions
  (ADR 0113). The default seat stays `f32`. For BiRefNet_HR the `f16` seat is lossless — the
  upstream checkpoint is itself stored in f16 — while Lucida's is quantized; the NOTICE and model
  card say which. Each compressed series records its export-time quantization error against the
  unrounded f32 outputs in `quality.json`, and the export refuses to publish an `f16` series of an
  all-f16 checkpoint unless every case is bit-identical, or a lossy one whose worst alpha MAE
  reaches half an 8-bit alpha step (a provisional line). At 1024² the resident weights drop from
  919 MiB to 509 MiB. The published repositories pick the seat up at the next release.
- `tools/flag-bench`: a Deno benchmark that compares Gemma execution flags against a reference on the
  same machine (GPU time as the primary metric, wall clock secondary) and records which kernel keys
  actually ran.
- Device-keyed sha256 reference rows for six more test suites: the SigLIP 2 and Gemma 4 golden
  tests (f32 safetensors of the pooled output / the last-position logits row), the BiRefNet and
  Depth Anything real-image tests and the vowel-detector end-to-end chain (f32 safetensors of the
  raw output), and one row per Gemma 4 / Gemma 4 QAT quant seat, enumerated from the mirror's
  manifest (the token ids of a fixed 64-token greedy prompt). Runtime-side fixtures live in
  `packages/runtime/tests/fixtures/references/`. Where a row is an extra check on an existing
  test, it is created and compared only after that check passes, and a case without a row for the
  current environment still records its measured sha256 in `results.json`.
- `AnimaPipelineOptions.residency` and `AnimaGenerateRequest.residency` (`"per-stage"` | `"transformer"`,
  default `"per-stage"`, ADR 0112): `"transformer"` keeps the DiT session alive after a `generate`, so
  the next `generate` on the same pipeline skips reloading the DiT weights, re-deriving its plan and
  rebuilding its intermediate buffers (about 2.45 s, or 10.4% of the wall time, per image from the
  second call onward, measured on an Intel Arc B570 at 1024² with the default quant). The request
  value decides whether the DiT is kept after that call; a DiT that is already resident is always reused. Keeping it raises peak VRAM, because the
  text and VAE stages load on top of it (about +2.6 GiB at 1024² with the default quant). While a
  DiT is resident, the pipeline checks with a trial allocation (`fitsHeadroom`), before the text
  stages and again before the VAE stage, whether the next stage fits on top of it; if it does not,
  the pipeline drops the resident DiT before building that stage (reason `headroom`) and stays
  per-stage for the rest of its life. The check runs up front because on Deno an out-of-memory
  error raised while uploading weights can already have lost the device. As a second line, when
  another stage, or a DiT carried over from an earlier call, still runs out of memory (as a
  `GpuOutOfMemoryError`), the pipeline drops the resident DiT, waits for the release to reach the
  device, retries that stage once, and likewise stays per-stage. A DiT built in the
  same call that runs out of memory is not evicted; the error is thrown as on the per-stage path.
  A failed call with `"per-stage"` still releases a carried-over DiT. Unknown values throw
  `ModelInputError` before any weight bytes are fetched. The
  new `residency` event reports each change (`retained` / `released` / `evicted`, with the reason);
  the `AnimaResidency`, `AnimaResidencyAction` and `AnimaResidencyReason` types are exported from
  `@karume/models` and `@karume/models/anima`.
- `SessionDiagnostics.geometryProfile`: the id of the tile-geometry profile the session selected
  (`"default"` on adapters without a profile). The name is a diagnostic label outside semver.
- `BUILTIN_GEOMETRY_PROFILES` (`@karume/runtime`): the built-in tile-geometry profiles as values, so an
  application can pick one by id and inject it with `acquireGpu({ geometryProfile })` — for
  example `apple-metal-3` on an Apple M2 that Chrome reports without a description.
- `acquireGpu({ geometryProfile })`: inject one static tile-geometry profile for the device (ADR
  0115), for adapters without a built-in profile — typically the table `tools/geometry-sweep`
  generates from a sweep on that machine. When given, the adapter's vendor and architecture are
  not consulted, every session on the device and the dp4a canary use the table, and nothing is
  measured at run time. A malformed table throws `GpuFeatureError` before any device is created,
  including a value with missing or mistyped fields (for example one read from storage without
  `parseGeometryProfileJson`), and so does a table with a geometry that exceeds the adapter's
  workgroup limits (`maxComputeInvocationsPerWorkgroup`, `maxComputeWorkgroupSizeX` / `Y`,
  `maxComputeWorkgroupStorageSize`); the message names the field, the geometry and the limit.
  `geometryProfile` also accepts a synchronous callback `(adapterInfo) => GeometryProfile |
  undefined`, called once per `acquireGpu` after the adapter is obtained and before the device is
  created, with the same adapter information that becomes `GpuContext.adapterInfo`: a returned
  table goes through the same check, `undefined` means automatic selection, and an exception thrown
  by the callback propagates without creating a device. A callback that returns a Promise is
  rejected with `GpuFeatureError` (load stored tables before calling `acquireGpu`). The runtime
  does not compare the returned table with the adapter; call `geometryProfileMismatch` inside the
  callback for that. The adapter information handed to the callback (and `GpuContext.adapterInfo`)
  is frozen, including the empty-valued record used when the adapter has no `info`. The types
  `GeometryProfile`, `GemmRowsRule`, `GemmGeometry` and `I8a8Geometry` are exported for it; the
  built-in tables themselves are not.
- IR op `conv3d` (ADR 0118): an **unbatched** 3-D convolution
  `x[Cin,T,H,W] * W[Cout,Cin/groups,Kt,Kh,Kw] + b[Cout] → [Cout,Tout,Hout,Wout]` with `stride` /
  `padding` / `dilation` as `[T, H, W]` triples and a scalar `groups`, all four mandatory. The
  runtime runs it on the GPU as an implicit GEMM with `groups == 1` and f32 / f16 / i8 weight
  storage; a graph with `groups > 1` is rejected when the session is built. The exporter keeps
  `aten.conv3d` as one node for an unbatched input (normalizing scalar spatial arguments to
  triples and synthesizing a zero bias when there is none) and fails loudly on a batched rank-5
  input.
- `ChunkBudgetStats.submitGpuTime` (`SessionDiagnostics.submit.chunkBudget`): on a device with
  `gpuTiming`, the GPU-side span of every submit is measured from its own timestamps, and the
  session keeps the count, the largest span, the same two for the submits made before the
  adaptive budget had a measurement, and the heaviest single dispatch with its pipeline key. It
  adds no wait and is undefined without `gpuTiming`; values are in timestamp units like
  `GpuTimingEntry.ns`.
- `@karume/models/wan` (also in the barrel): Wan2.1 T2V 1.3B text-to-video (ADR 0118).
  `WanPipeline.fromPretrained(source, { model?, quant?, gpu?, onRunDiagnostics?, … })` loads a
  `karume/5` distribution with pipeline `wan/1` (the `karume-wan2.1` layout: model `t2v-1.3b`,
  quant `f16`, the transformer and the two VAE chunk-graph containers `vae_decoder_first` /
  `vae_decoder_next`, and the precomputed text-embedding asset `text_embeds`); the distribution is
  not published on Hugging Face yet, so there is no `WAN_SOURCES` table and a local copy is passed
  as a directory source handle. `WanPipeline.fromAssets({ manifest, assets })` builds the same
  pipeline from bytes already held. The manifest's `pipelineConfig` declares the defaults for
  `steps`, `guidance` and `shift`, and a quant that declares any session knob is rejected before
  weights are fetched. `generate({ prompt, negativePrompt?, seed | latents, steps?, guidance?,
  shift?, frames = 33, width = 832, height = 480, onEvent? })` returns `GeneratedVideo`
  (`[3, F, H, W]` f32 in `[-1, 1]`, and `fps` — 16 for Wan2.1, a fact of the upstream model rather
  than a knob): the DiT runs with classifier-free guidance as two batch-1 passes, the flow-matching
  UniPC scheduler runs on the host, and the VAE always decodes in tiles.
  `prompt` and `negativePrompt` must be one of the prompts stored in the embedding asset (original
  or normalized text — `WanPipeline.prompts` lists them); any other string, a size other than
  832×480 / 480×832, a frame count other than 4n+1 in 5–81, and out-of-range knobs throw
  `ModelInputError` before any weight reaches the GPU. A GPU with `gpuTiming` is rejected at construction (the VAE stage needs
  batches). `wanFrameToRgba(video, frame)` converts one frame to 8-bit RGBA for `encodePng`
  (it reads only `frames`, `width`, `height` and `data`, so a video assembled by hand needs no
  `fps`).
- Exporter recipes: `dist.py --pipeline wan` assembles the Wan2.1 series into the
  `karume-wan2.1` distribution (Apache-2.0 `LICENSE.md` and the change `NOTICE.md` at the root, a
  model card listing the pinned upstream revision and the fixed prompts). It refuses series whose
  storage is not f16, whose container provenance names another upstream revision or license, a
  transformer without its RoPE base tables, and an embedding asset made from another revision, with
  other prompts or with a width that does not match the transformer.
- Wan2.1 int8 transformer quants (ADR 0120): `wan.export_dit --dtype i8` writes the
  `wan2.1-t2v-1.3b-i8-dyn` series (the 307 linear weights as per-output-channel int8; bias, norm and
  modulation tables stay f32), and `dist.py --pipeline wan` adds two quants to `karume-wan2.1`:
  `f16+dit8` (int8 transformer weights, f32 compute — the reference quant) and
  `f16+dit8-a8-attn8-s16` (int8 weights with per-token int8 activations in linear and attention,
  f16 score storage). `f16` stays the default; `WanPipeline` accepts `linearCompute`,
  `attentionCompute` and `attentionScoreStorage` from the quant declaration. The GPU numerics gates
  and the visual check for these quants land with ADR 0120 stages 3–6.
- `karume-umt5-xxl` (ADR 0119): a distribution of the umT5-XXL text encoder alone (`dist.py --pipeline umt5`,
  pipeline `umt5-encoder/1`, quant `i8`, 26 parts). `karume-wan2.1` references its `text_encoder` across
  repositories and ships the tokenizer asset `umt5_tokenizer`.
- `WanPipelineOptions.textEncoder: "gpu" | "precomputed"` (default `"gpu"`): the GPU route runs umT5 and
  accepts any prompt (2–512 tokens; out-of-vocabulary characters, in-text special tokens, HTML entity
  candidates, mojibake, C1 controls and unassigned code points are rejected with a message that says how
  to fix the prompt); `"precomputed"` keeps the fixed-prompt embedding asset and does not fetch umT5.
  `WanPipelineOptions.signal` and `WanGenerateRequest.signal` cancel loading and generation (the signal's
  reason is thrown as is; the next `generate` works after a cancellation).
- Exporter recipes: third-party umT5-compatible text encoders can be converted for local experiments
  (ADR 0122). `python -m wan.umt5_intake --repo <owner/name> --revision <commit> --file <name>.safetensors
  --name <name>` fetches the weights at a pinned revision, checks their sha256 and records them under
  `inputs/umt5/<name>/` (`intake.json`, format `karume-umt5-intake/2`); `wan.umt5_export --intake` writes
  the `umt5-xxl-<name>-i8-dyn` series (BF16 weights are widened to F32 on read), and `dist.py --pipeline
  umt5 --intake` assembles a local mirror (`umt5-xxl-<name>-local`, default output
  `outputs/misc/local-dist/<name>`; paths under `models/` are refused). Weights whose license is not
  declared are recorded as `NOASSERTION` and need `--allow-undeclared-license`; the mirror then has no
  `LICENSE.md`.
- `WanTi2vPipeline` (`@karume/models/wan`, also in the barrel): Wan2.2 TI2V 5B text-to-video
  (ADR 0121). It has the same surface as `WanPipeline` (`fromPretrained`, `fromAssets`, `prompts`,
  `generate`, `dispose`) and shares its public types (`WanGenerateRequest`, `GeneratedVideo`,
  `WanGenerateEvent`, `WanPipelineOptions`, …); it reads manifests with pipeline `wan-ti2v/1`, and
  each class rejects the other's pipeline. Accepted sizes are 1280×704 and 704×1280 with 4n+1
  frames from 5 to 33 (default 1280×704, 33 frames); the latents are `[48, F', H/16, W/16]` and
  `fps` is 24. Image-to-video is not available yet. There is no distribution yet, so for now the
  pipeline is built with `fromAssets` from a manifest and the exported series containers.
- Release tooling: `tools/release/hf-upload.zsh upload` first reads `provenance.license` from every
  container of the directory (`tools/release/container_license.ts`, needs Deno) and uploads nothing when
  one carries the undeclared-license mark (`NOASSERTION`, or a value such as `unknown`, in any case), when
  a part 0 cannot be read, or when the directory has no container. A direct `hf upload` bypasses this
  check.

### Changed

- Every pipeline now composes its session options through one shared rule (explicit > quant
  declaration > runtime default) and rejects, before fetching weights, a manifest `session` key the
  family does not accept; previously the non-Gemma families passed any declared key through to the
  runtime. All published quants are accepted unchanged.
- Every pipeline now derives its GPU feature requirements from the effective session options (the
  quant's `gpuFeatures` plus `shader-f16` when an `"f16"` compute path is in effect), requests them
  when it acquires its own device, and checks them on a shared `gpu` before fetching weights.
  Gemma 4 pipelines previously ignored a quant's `gpuFeatures` declaration. Published distributions
  request the same features as before.
- Gemma 4 QAT recipe: E4B gains the `i4-gemvpar` and `i4-fast` quants (the same declarations as
  E2B) and now defaults to `i4-fast`; the E2B and E4B `i4-fast` declare
  `stateAttentionReduce: "parallel-fused"` instead of `"parallel"`. On an Intel Arc B570, `i4-fast`
  cut GPU decode time per step by 49.3% (E2B) and 41.7% (E4B) against `i4`, and no single flag was
  slower on its own. Rebuild the distribution to pick up the new quants; no requantization is needed.
- The BiRefNet 2048² GPU tests (`birefnet-hr-2048` / `lucida-2048`) are skipped explicitly on the
  Intel Arc B570 environment key (`deno-intel-graphics-bmg-g21`), where one `deform_conv2d`
  dispatch exceeds the Linux xe driver's 5 s job limit and the resulting device-lost panic would
  stop the whole test process. The skip is per environment key and announced at registration; the
  asset-completeness test still runs, and other machines run the series as before.
- Anima's `stage` event now marks the start and end of a stage. On the default per-stage path these
  are still the points before the session is built and after it is released; for a DiT carried over
  and kept (`"transformer"`), the `transformer` stage's start and end no longer bracket a weight load
  or a release. With `"per-stage"` the release of a carried-over DiT happens inside them, and after
  an out-of-memory eviction so do the release and the reload.
- Irodori generation is faster: the DiT's conditioning keys / values, which do not change across
  denoising steps, are now computed once per generation instead of in every forward pass. On an
  Intel Arc B570 with the default `i8-a8` quant, a voice-clone generation took 12.7% less wall time
  and a 30-second utterance 5.0% less; the waveform is bit-identical and the observed peak of GPU
  buffer memory is unchanged. This needs a distribution in the new layout (see **Breaking**).
- GEMM tile geometry is now selected per adapter from built-in static profiles (ADR 0115), chosen
  once at session construction from the adapter's vendor and architecture; nothing is measured at
  run time. Adapters without a profile get byte-identical shaders and pipeline keys as before.
  Apple `metal-3` under Chrome gets a profile measured on an M2, covering linear / matmul / bmm,
  fused attention, conv2d and the i8a8 linear and attention kernels; it is selected automatically
  only when the adapter also reports the description `Apple M2` (Chrome does so with
  `chrome://flags/#enable-webgpu-developer-features`; Deno always does). Without it, Apple adapters
  keep the default geometry, which is the best on an Apple M5, and the table can still be injected
  with `acquireGpu`. NVIDIA `blackwell` under
  Chrome gets a profile measured on an RTX 5070 Ti that changes only the i8a8 linear and attention
  tiles (the 512² DiT stage of the default quant took 1.14 s of GPU time instead of 1.36 s); its
  f32 kernels keep the default. Outputs stay
  bit-identical across geometries. Both built-in profiles now carry seven `gemmRows` rules
  (M ≤ 16 / 32 / 64 / 128 / 256 / 512 / > 512, ADR 0116) instead of the default table's three,
  each measured with linear, matmul and bmm cases; the default table is unchanged. On an M2
  under Chrome, the 512² DiT stage of Anima's `f16`
  quant took 45 s instead of 80 s (GPU time 44 s instead of 77 s) and the default quant 71 s
  instead of 82 s; the PNG bytes are unchanged. Measured again under Chrome 154 with the
  seven-rule table, the same stage took 43 s instead of 52 s (GPU time 42 s instead of 51 s):
  the default f32 linear and QK geometries on the M2 ran faster under Chrome 154 than under
  Chrome 153 (same WGSL), so the table's gain is smaller there; the PNG bytes are still
  unchanged. On an RTX 5070 Ti the same A/B leaves the `f16` DiT stage unchanged (2.88 s of GPU
  time either way, because the NVIDIA profile changes only the i8a8 tiles and the M 129〜512 rows)
  with identical PNG bytes.
- `karume-wan2.1`: the default quant is now `f16+dit8-a8-attn8-s16` (int8 transformer with int8
  activations, ADR 0120) instead of `f16`, so `WanPipeline` without `quant` runs it; pass
  `quant: "f16"` for the f16 transformer. On an Intel Arc B570 a 50-step, 33-frame clip took 952 s
  instead of about 30 minutes, and an 81-frame clip 3,703 s instead of about 2 hours. The gpu-lab
  Wan tab gains a quant choice, and its JSON is `karume-wan-browser/3`. `deno task demo:wan` gains
  `--quant` (a key of the manifest's `quants`); the output directory names the quant after the
  family, or `default` without it.
- `karume-umt5-xxl` (not published yet): the containers now name the original `google/umt5-xxl`
  commit as their upstream instead of the Wan2.1 checkpoint's `text_encoder` (its encoder weights
  are bit-identical in float32, ADR 0122), and the exporter series is renamed from
  `wan2.1-umt5-i8-dyn` to `umt5-xxl-i8-dyn`. The weight parts are byte-identical; only the first
  part (the model descriptor's provenance) and, in `karume-wan2.1`, the cross-repository reference
  to it change. `wan.umt5_export` selects the umT5 upstream with `--upstream` and the Wan text
  snapshot with `--model`, and rejects an option the subcommand does not use.

### Fixed

- `@karume/runtime`: the fused-attention i8a8 ①QK shader generator produced wrong scores (whole
  columns silently zero) for tile geometries whose per-thread K-side fill count is 5 or more: a
  generated fill variable (`k4`) shadowed the K pack count. The default geometry and the Apple
  profile were not affected. The fill variable is now named independently of the caller, and a
  codegen test rejects any shadowed declaration across all candidate geometries.

### Breaking

- `@karume/hub`: the manifest `session` vocabulary gains `stateAttentionReduce` (`"sequential"` /
  `"parallel"` / `"parallel-fused"`), composed like the other quant knobs (explicit > quant
  declaration > runtime default). Readers of 0.13.x reject a manifest that declares it as an unknown
  `session` key, so a distribution rebuilt with the new recipes needs this release's hub / models.
  Published distributions and the pinned revisions are unchanged until they are re-uploaded.
- `@karume/models`: Gemma 4 pipelines no longer apply a built-in `stateAttentionReduce: "parallel"`
  default. The attention reduction now comes from the explicit option, else the selected quant's
  declaration, else the runtime's reference `"sequential"`. As a result the `i4` quants (regular and
  QAT), `fromAssets`, and every quant of a distribution built before
  this change (one that does not declare the key) run the reference attention path; pass
  `stateAttentionReduce: "parallel"` to keep the previous behaviour. The rebuilt recipes declare it explicitly: `i4-gemvpar` declares
  `"parallel"` (unchanged behaviour), and the `i4-fast` quants (regular and QAT) now declare
  `"parallel-fused"` (same summation order as `"parallel"`; on an Intel Arc B570, regular E2B
  GPU decode time −47.9% against `i4` with the reference attention, versus −46.9% with
  `"parallel"`, with identical tokens).
- `@karume/models`: `AnimaGenerateEvent` gains a fourth member, `{ kind: "residency" }`. TypeScript code
  that handles the three previous kinds and narrows the remainder with `else` no longer type-checks;
  add a `residency` branch. The event is only emitted when `residency: "transformer"` is used or a
  resident DiT exists, so default-path event sequences are unchanged at runtime.
- Irodori distribution layout: the DiT is split into two graphs, and the manifest `weights` gain a
  `dit_context` component (in all four weight series, f32 / f16 / i8 / i4) that takes the three
  conditioning states; `dit` now takes the conditioning keys / values it produces instead. The
  `i8+dit4` quant takes both DiT graphs from the int4 series. This release's `@karume/models` rejects a
  distribution without `dit_context` as a missing component before any weight bytes are fetched,
  and older releases cannot run the new layout. The published Irodori repositories and the pinned
  revisions are updated at the next release.
- `@karume/models`: `IrodoriRunComponent`, and with it the Irodori `stage` event's stage names, gain
  `"dit-context"` (run once per generation, before the DiT loop). TypeScript code that switches
  exhaustively over the previous eight names no longer type-checks.
- `WanPipeline` defaults to the GPU text encoder: distributions without the umT5 cross-repository
  reference need `textEncoder: "precomputed"` (the error says so). Every Wan quant now declares
  `requiredLimits` of 1,050,148,864 bytes for `maxBufferSize` / `maxStorageBufferBindingSize` (the
  vocabulary embedding), also on the precomputed route. `WanGenerateEvent` stages and `WanRunComponent`
  gain `text_encoder`. `deno task demo:wan` names its output directory by route. The gpu-lab Wan JSON is
  `karume-wan-browser/2`.

## [0.13.0] - 2026-09-25

### Added

- Karume container format (`krm` / `krg`, ADR 0108, stage 1): the runtime opens a container
  (`openContainer`), admits one of its graphs (`prepareContainer`) and builds a session from its
  blocks (`createSessionFromContainer`), checking each block's sha256 when the source has not
  already verified the whole file (ADR 0109); `codecLayout` maps a
  codec name to its decode path. New types on the runtime surface: `ContainerInput`, `AssetReader`,
  `CodecLayout` and `CodecName`. The exporter gains `karume.container` (writer / reader, canonical
  JSON with ECMAScript number spelling) and `karume migrate` (old shards → `krm`). IR v2 replaces
  IR v1 (`docs/ir-v2.md`).
- Karume container format, stage 2 (ADR 0109): manifest `karume/5` puts a container behind every
  `weights.<component>.<dtype>` (`descriptor` expectations plus the part `FileRef`s); hub resolves a
  selection to containers and assets (`resolveSelection` / `selectionRefs`) and opens a container as a
  `BlockSource` for the runtime (`openContainerSource`, range reads over the warmed cache, no digest on
  a warm hit). Every pipeline's `fromPretrained` accepts `components` to swap one component for the
  same role in another `karume/5` repository (admitted before any weight bytes are fetched). Container
  assets carry a declared logical length and are read through `OpenedContainer.asset(name)`; Gemma's
  PLE sidecar becomes container assets (index schema 3 over row-aligned blocks). `karume migrate
  --manifest` converts a whole `karume/4` repository (including PLE and cross-repository references)
  and writes the `karume/5` manifest; `karume verify <container>` checks a container from the CLI.
  Every default source pin names a `karume/5` revision of its repository.
- Karume container format, stage 3 (ADR 0108): the exporter writes `krm` directly (`publish_model` /
  `export_to_file` with `provenance` and `graph_name`; PLE tables and `rope_base` become container
  assets), `karume dist` bakes a `karume/5` repository from container series,
  `karume migrate --part-bytes` picks the part length from the writer's choice set
  (256 / 512 / 768 / 1024 MiB; default 256), and the runtime gains an in-memory container
  (`openMemoryContainer`) that feeds host-built graphs and already-decoded tensors through the same
  admission and upload path as a `krm` without writing one —
  `parseIrDeclarationValue` / `IrDeclaration` are on the runtime surface so callers can build the
  declarations for it. Gemma's on-device PLE gather and Irodori's host graphs go through it. New types
  on the runtime surface: `MemoryContainerInput`, `MemoryEncoding`, `MemoryPiece`, `MemoryTensor`, and
  `BoundContainer` (the face both a `krm` and an in-memory container present to the session builder).
- Gemma 4 QAT family: `gemma4-qat` pipelines for E2B / E4B with fixed INT2 / INT4 storage, fixed
  static range quantization (SRQ) whose rounding is preserved on both CPU and GPU, PLE read back
  whole or row by row, and a chat CLI example.
- Speculative decoding for Gemma 4 (MTP): a drafter session, the `speculative` option, a
  draft/verify/commit loop whose token stream matches non-speculative decoding exactly, and a
  self-financing gate that falls back to plain decode in contexts where speculation loses.
- Range reads for assets: `openAsset` and `AssetRangeReader` in hub, a directory adapter that
  reads at an offset, and range reads over Hugging Face files once fetch-cache 0.8.0 has fetched and verified them whole
  (no HTTP Range requests).
- `Session.enqueueRead` reads graph outputs back at the batch's terminal fence; PLE can be kept
  resident on the GPU as an opt-in, with `per_layer_inputs` gathered on device.
- Chat CLI examples for MiniCPM5 and Qwen3, and a headless-Chrome benchmark page that compares
  TTFT and generation speed for Gemma 4 E2B.
- `glossary.md` and `quantization.md` under `docs/` as indexes for terminology and quantization
  methods.
- Test lanes: `deno task test:core` and `deno task test:models:<family>` split the full verify
  into a core lane and per-family lanes; `verify_lanes_test.ts` checks that every test file
  belongs to a lane. Full `deno task verify` is unchanged and remains the release gate.
- Device-keyed sha256 references: the PNG / WAV reference values live as per-environment rows
  in tracked JSON fixtures instead of constants. `KARUME_REFERENCE=write` creates the rows for
  a new machine (`rewrite` refreshes them), a machine without rows skips those cases explicitly
  and fails a reference gate (`KARUME_ALLOW_NO_REFERENCE=1` to opt out), and every run writes
  `results.json` plus the produced images / audio under `outputs/verify/<environment>/`.
- Per-environment golden tolerances and recorded measurements: the second tolerance band of the
  golden comparison (the WGSL spec band) is keyed by environment too, so a band widened for one GPU
  no longer loosens the regression net on machines that have no such row. Every golden comparison
  now records its `measurements` (largest absolute and relative difference, the band that accepted
  the output, and which stage accepted it) in `results.json` even when it passes, the eleven
  real-weight golden tests write their own `<family>-golden` results, and `tools/verify-diff` lays
  the `results.json` files collected from several machines side by side and prints what differs —
  a read-only tool, not a gate.
- A public-surface snapshot gate for the three packages (`deno doc --json` symbols against a
  tracked fixture; `KARUME_SURFACE=write` refreshes it).
- Each package now ships its README and LICENSE; this CHANGELOG.
- `ModelInputError` in `@karume/models`: one cross-family error for requests that cannot be
  accepted as given, exported from the barrel and from every pipeline subpath, so a host that
  loads several families tells a 400 from a 500 with a single `instanceof` instead of reading
  message strings. `Sbv2InputError` and `GenerationCapacityError` are now subclasses of it.
- `ANIMA_SAMPLER_TYPES` in `@karume/models` (barrel and `./anima`): the frozen list of values
  `AnimaGenerateRequest.sampler` accepts; `AnimaSamplerType` is now derived from it (same union).
- hub's `AssetProgress` carries the file's origin as optional `repo` / `revision` (the declared
  cross-repository target, or the session's repository and resolved SHA), so per-file progress can
  tell apart same-path files from different repositories; the aggregated progress stream in
  `@karume/models` keys files by (repo, revision, path) instead of `path` alone.
- English READMEs for the Anima, Irodori, SBV2 and vowel-detector example CLIs
  (`examples/<family>/README.md`: sources, options, output files and constraints), and a
  `--sampler <euler|dpmpp-2m>` option for `examples/anima`; the output file name carries the
  sampler when it is given, so sampler A/B runs no longer overwrite each other.
- Distribution legal files: re-baked Irodori repositories (`karume-irodori-v4-small` /
  `karume-irodori-v4.1-small`) ship `LICENSE.md` (MIT) and `NOTICE.md`, and `karume-sbv2-jvnv` ships
  `LICENSE.md` (CC BY-SA 4.0) and `NOTICE.md`. The export recipes split SBV2 into two pipelines by
  voice family: `--pipeline sbv2` (JVNV, with the attribution files) and `--pipeline sbv2-fn` (FN).
- The DeBERTa w8a8 mirror gate (`packages/runtime/tests/e2e_deberta_w8a8_test.ts`, ADR 0026
  decision 3): runs `linearCompute: "a8"` against the torch act-quant mirror goldens
  (`io-i8a8.<case>`) with a strict tolerance on output.0/1, a collapse ceiling on the deeper layers,
  and a pipeline-key census (192 i8a8 linears / 192 `quantize_rows` / no other linear kernel).
- More test gates: the mixed-codec miniature model (i4 group-scale linear → f16 add → i8
  per-channel linear, piece-split) in `gpu_memory_container_test.ts` is again checked against an
  independent CPU reference (codec decode + reference ops), not only krm vs in-memory container
  agreement, and a miswired-scale reference is asserted to fail; the Anima host glue
  (`sigmaSchedule` / `cfgEulerStep` / `denormalizeLatents` / `padSequence`) is checked bit for bit
  against all four `pipeline_ref.py` reference fixtures (`anima_host_glue_parity_test.ts`; skipped
  with a reason when the fixtures are absent); GPU tests pin the models-internal `withSession` scope
  (dispose on success and failure, observe ordering, pass-through of results and errors); CI lints
  and format-checks `tools/llm-baseline` with the export-recipes ruff settings, and `pack_int2` /
  `unpack_int2` are documented as the canonical i2 byte order (ADR 0097).
- New names on the `@karume/runtime` surface: `DEFAULT_PLAN_BACKING_BUDGET_BYTES`,
  `assertChunkBuckets`, and the types `AdmissionScenarioSpec`, `LinearGemvReduce`, `RmsNormReduce`,
  `EnqueueRead` and `SharedWeight`; `Session.exportWeight`; and the
  `SessionOptions` keys `linearGemvReduce`, `rmsNormReduce`, `fuseRmsNormAdd`,
  `fuseLinearStaticQuantize`, `packedStaticQuantize`, `linearGemvRowsThreadTarget`,
  `planBackingBudgetBytes` and `sharedWeights`, plus `GenerationContextSpec.chunkBuckets`.
- New names on the `@karume/models` surface: `GEMMA4_CHUNK_BUCKETS`, `gemma4QatRopeInputs`,
  `Gemma4QatPipeline`, and the types `Gemma4QatPipelineOptions`, `Gemma4QatFromPretrainedOptions`,
  `ComponentSource`, `FromPretrainedComponentOptions`, `SpeculationGateOptions` (the
  speculation gate's knobs under `speculative.gate`; `exploreMax` defaults to 512),
  `GenerationGateTrace`, `GenerationRunPhase` / `Gemma4RunPhase` (the `phase` passed to `onRun`)
  and `Gemma4PleResidency` (the `pleResidency` option). `speculative.policy: "always"` speculates on
  every cycle without the gate.
- New names on the `@karume/hub` surface: the types `ContainerRef`, `DocumentExpectation` and
  `ContainerBlockSource`, and `ResolveOptions.weights` (fetch a subset of a model's weights by
  name). The manifest `session` vocabulary gains four keys: `linearGemvReduce`, `fuseRmsNormAdd`,
  `fuseLinearStaticQuantize` and `packedStaticQuantize`.

### Changed

- anima distribution repositories: `NOTICE.md` and the model card now state that any rights to use
  the CircleStone Models and/or Derivatives are granted directly by CircleStone Labs LLC under the
  CircleStone Labs Non-Commercial License (license §3(a)).
- `GpuContext.beginBatch()` rejects with `GpuDeviceLostError` when the device is already lost or
  destroyed; `openContainer` rejects a `krm` whose expectation has `graph` but no `model`, and
  containers whose weight pieces are not in non-decreasing part order; `estimateGraphMemory` /
  `PreparedModel.estimate` pack intermediates within `options.maxBufferSize` when it is given.
- `opbench single` accepts every knob of the manifest `session` vocabulary (`--session
  <knob>=true|false` for booleans), records `gpu_wall_ratio` / `timing_warning` when GPU time
  contradicts the wall clock, and `opbench census` / `tools/fusion-hints` read manifests through
  `@karume/hub`; `tools/ram-peak/measure.ts` gains `--state cold|warm|local` and `--cache-dir` (a
  directory below `outputs/ram-peak/`); `tools/release/hf-upload.zsh` marks unreadable parts as `### FAILED` and exits
  non-zero; CI lints the workspace-external Python scripts and checks `deno.lock` for drift.
- Model cards: the Usage snippets declare the pipeline with `await using` (these pipelines only
  implement `Symbol.asyncDispose`); the sbv2 card analyzes text with `@hdae/yomi`, converts it
  with `toSbv2Utterance` and calls `generate(utterance, options)` (the claim that the pipeline
  fetches a dictionary is removed); the gemma4 card names each device limit once; SBV2 cards
  explain how every quant seat was rounded; the vowel-detector card states the minimum clip length.
- Export recipes: `gemma4.export_drafter` no longer accepts `--sym-max`; `embeddinggemma.export`
  fails loudly before publishing if SDPA was decomposed or the band mask is not a folded constant;
  anima export refuses `--lora` / `--num-layers` when no running target can use them.
- The opt-in `parallel-subgroup32` GEMV variant now spells its multiply-add with an explicit
  `fma()` like the parallel family (ADR 0101 addendum). Vulkan output is unchanged; on Metal the
  subgroup32 output now matches the parallel kernel bit for bit instead of the previous spelling.
  The browser benchmark server serves `/check.html`, which runs the parallel-vs-subgroup32 u32
  identity check in Chrome.
- The peak of JavaScript-side `ArrayBuffer` memory (V8 "external") while building a session from a
  container is lower (process RSS additionally includes wgpu staging and is not bounded by this): the
  runtime uploads container
  blocks one at a time and releases each as soon as `writeBuffer` returns (the WebGPU specification
  copies the bytes at call time), and hub's scan path hands out views of the held part instead of
  copies. Sources read by range (local directories, and browser blob reads by design) now peak at
  about one block plus the CPU expansion buffer regardless of part length; Hugging Face downloads
  under Deno still hold the current part, plus earlier parts until they are garbage-collected.
- Decode and prefill are substantially faster: the GEMV family covers 1 ≤ M ≤ 64 as row-block
  variants, parallel GEMV (with optional subgroup reduction) is selectable from the quant seat,
  argmax over long rows runs in two phases, top-k uses a bounded heap and top-p is linear in the
  vocabulary, and RMS+residual, RoPE, attention row-statistics+PV and linear→SRQ can be fused.
- The distributed Gemma 4 E2B defaults to the `i4-fast` quant seat, which declares parallel GEMV
  and the RMS-norm + residual fusion; the `gemma4_qat` recipe's `i4-fast` additionally declares
  the linear → SRQ fusion and packed INT8 activations (`packedStaticQuantize`) for the parallel
  GEMV path.
- Slot backing is held as an LRU set under a byte budget instead of a single slot, so changing
  shape no longer rebuilds it on every run.
- Gemma 4 prefill picks the smallest declared chunk bucket that covers the query length; the
  default bucket set is `[4, 8, 32, 64, 128, 256]`.
- Golden tolerances are two-tier: outputs are checked against Karume's own bound first and,
  where a WGSL accuracy bound is declared for that output, against the specification bound;
  exceeding only the first is recorded in `results.json` rather than failed.
- `karume` (PyPI) no longer imports torch at package import: the public names in `karume.__all__`
  resolve lazily on first access (PEP 562), so torch-free modules such as `karume.dist`,
  `karume.modelcard` and `karume.container` import in ~0.15 s instead of ~1.8 s. `__all__` and what
  each name resolves to are unchanged.
- `karume dist` takes the repository name in the model card's Usage example from the pipeline's
  declaration instead of deriving it from the `--out` directory name. The new `--repo OWNER/NAME`
  overrides the declaration (and then requires `--out`); when the models being bundled declare
  different names (e.g. the four JVNV voices of `karume-sbv2-jvnv`), `--repo` is required.

### Fixed

- The safe-softmax guard is no longer removed when scores can reach -inf via a negated +inf literal
  or an f32 overflow; `karume dist` refuses a series whose weight parts disagree with the
  descriptor's `parts[].sha256`; `tools/mtp-bench` derives the PLE budget from mirror paths with
  spaces or non-ASCII characters.
- Runtime (2026-09-24 full review, obvious fixes): `openContainer` / `openMemoryContainer` reject
  `rowAxis: 1` for `int4-sym-g` / `int2-off` / `ternary` (only `int8-sym` may declare it);
  container readers no longer resolve names such as `constructor` through `Object.prototype`;
  `OpenedContainer.asset(name).read` rejects non-integer / NaN ranges and short reads from a
  verified source; `Session.enqueueRead` rejects at admission when a graph output aliases a
  resident input and rejects an oversized batch readback before any dispatch; `Session.dispose()`
  rejects while a batch holding this session's `enqueueRead` is unsettled (call it again after
  `batch.finish()`); `Session.run` reports invalid generation arguments through the returned
  Promise instead of throwing synchronously, and symbols in numeric knobs raise `ExecutionError`
  instead of `TypeError`; `argmax` over zero outer rows with a last dimension ≥ 16384 returns an
  empty output instead of an internal error.
- Models: a failing Session dispose no longer replaces the original stage error (an
  `AggregateError` carries the stage error first); Anima prompt encoding no longer throws
  `RangeError` on very long unspaced prompts and stops tokenizing at the 512-token limit;
  `VowelDetector` rejects non-finite audio samples with `ModelInputError` before GPU execution;
  `resizeRgb8` rejects unknown filter names instead of returning a black image and
  `normalizeToNchw` rejects non-finite mean / non-positive std; Gemma tokenizer assets with unknown
  fields and PLE index assets without `storage` are rejected; `parseGemma4PipelineConfig` rejects
  `chunkLength` / `maxChunkLength` below 2 at declaration time; Gemma 4 QAT admission errors use
  the `Gemma4QatPipeline:` prefix; `decodeWav` rejects duplicated `fmt` / `data` chunks; SBV2
  phoneme errors refer to the caller's utterance.
- Exporter (`karume`): `karume verify` and the publish / migrate self-check reject optional
  provenance fields that are not non-empty strings (matching the TypeScript reader) and reject
  `ternary` payloads containing code 0; `publish_container` refuses such payloads before placing
  the container; `karume migrate --manifest` verifies every declared file against the legacy
  manifest's size / sha256 first and applies the safetensors checks to legacy PLE sidecars; f16
  rounding fails loudly on saturation even when the tensor already holds non-finite values;
  `karume dist` reports missing / mistyped `karume.json` fields as `DistError`; `awq_search_scale`
  raises `QuantizeError` when every alpha is non-finite; `assert_reader_layout` rejects the legacy
  I4 / I2 dtypes by default (`allow_legacy_dtypes=True` reads legacy shards); `karume verify`
  accepts integral float dimensions.
- Metal parity: the parallel GEMV family spells its products with explicit `fma()`, so the packed
  INT8 activation path matches the f32 path bit for bit on Metal as well.
- GEMM no longer reads past the end of an edge channel; maximum selection preserves the ordering
  and value bits of subnormals.
- Lifecycle defects in models: the generation stream's end notification and release are
  consistent, conversation history no longer shares message ownership, the PLE cache is not
  re-registered after dispose, and Irodori releases every resource on multiple failures.
- `sin` golden on Intel Arc (Mesa ANV): the output is within the WGSL accuracy bound
  (absolute 2^-11) and is now accepted under it; the resident over-limit test no longer assumes
  `maxBufferSize` is a multiple of 4.
- Batch lifecycle in runtime: a batch whose read-back fails before its fence no longer feeds
  the partial elapsed time into the submit chunk estimate; `enqueue` / `run` /
  `finishAndRead` re-check their admission after copying the caller's inputs, so a getter that
  re-enters the same Session or batch is rejected instead of reading another call's values,
  running after `dispose`, or deadlocking; `finish` rejects when the device is lost while its
  error scopes are still settling.
- The runtime README's minimal example tears the device down through `gpu.destroy()` (calling
  `device.destroy()` directly fires `onDeviceLost` as an unexpected loss).
- An i4 weight whose rows are zero-length (e.g. `[R, 0]`) now expands on the CPU path instead of
  failing with a scale-shape mismatch: the companion-scale shape `[shape[rowAxis], rowLength /
  groupSize]` comes from a single function shared by the container binder, the residency planner,
  the container-to-session path and `decodeI4`.

### Breaking

- `@karume/runtime`: `tensorBytes` is no longer exported (view `SafetensorsFile.buffer` with the
  tensor's `byteOffset` / `byteLength`); `SessionBuildStats.shardCount` / `shardWaitMs` are renamed
  to `partCount` / `supplyWaitMs`.
- `@karume/hub`: `LocalDirectoryOptions.fallback` is removed — no public value could act as a
  delegate. Map cross-repo references explicitly with `crossRepo`; an unmapped reference fails
  loudly.
- Exporter (`karume`): `IR_METADATA_KEY` moved to `karume.legacy` (used only by `karume migrate`);
  `write_model` / `verify_model` leave `karume.__all__` (the `karume.shards` / `karume.repack` modules
  go with the shard form; `verify_container` / `stored_model` / `Provenance` / `FixedQuantizedWeight`
  are the new public names); `publish_model` no longer deletes stale
  `<stem>-NNNNN-of-NNNNN.safetensors` siblings; `publish_model` / `export_to_file` raise `ValueError`
  for an output path whose suffix is not `.krm`; `dist.weight_components` returns
  `(parts, weights key)` pairs; `modelcard.from_pretrained` requires the `disposable` keyword.
- **Breaking:** the runtime's in-memory graph (`IrGraph`) now uses the merged storage vocabulary:
  `initializers[name]` is `{ storage: { codec, groupSize?, rowAxis? } }` or `{ shared: true }`,
  initializer names are the tensor keys (the exporter's FQN / `const.<hash>`), and the `tensor` /
  `storage.dtype` / `storage.scale` fields are gone, and capability diagnostics say
  `非対応 格納 '<layout>'`. Shared initializers
  are named after the lender's initializer.
- **Breaking:** hub reads manifest `karume/5` only (no `karume/4`); `resolveFiles` / `ResolvedFiles` /
  `WeightFiles` are replaced by `resolveSelection` / `ResolvedSelection` / `WeightContainer`, the
  `<weights>[i]` fetch-key convention and the 256 MiB shard limit are gone, and `fetchAssets` takes a
  plain `Record<string, FileRef>`. Published repositories keep working from the previously released
  packages; this main reads only repositories re-uploaded in the container format.
- **Breaking:** `from*Assets` take containers instead of safetensors — a component key maps to a
  single-form `krm` (`transformer`) or to its parts (`transformer[0]`, `transformer[1]`, …).
  Gemma's PLE lives in the `model` container: `Gemma4Assets` loses `pleIndex` / `readPleShard`, decode
  reads only the PLE rows it needs instead of whole sidecar shards, and the default PLE resident budget
  is now two blocks (about 64 MiB) instead of two shards.
- **Breaking:** container descriptors declare `assets[].length` (payload bytes; the block length is
  that rounded up to 4) — containers written by the stage-1 writer must be rewritten; `BlockSource`
  gains a required `verified` flag and `DescriptorExpectation.sha256` is a plain string.
- **Breaking:** the Gemma 4 product graph exits on the selected R rows as logits plus hidden
  state — distributions must be re-exported. `Gemma4Pipeline` prefills in buckets by default
  (`GEMMA4_CHUNK_BUCKETS` = 4, 8, 32, 64, 128, 256; `Gemma4PipelineOptions.chunkBuckets` overrides).
- **Breaking:** the Gemma 4 drafter's calling convention was aligned with the upstream layout and
  its goldens re-baked.
- **Breaking:** speculation stops enumerating acceptances at a stop token, and reports what was
  handed to the caller as `GenerationSpeculation.delivered`.
- **Breaking:** input-caused failures that used to throw `RangeError` now throw
  `ModelInputError` — the seed range, sampler settings, image and audio sizes, and WAV sample
  rate / sample values. Code branching on `instanceof RangeError` for these has to switch to
  `ModelInputError`; plain `catch` is unaffected, since `ModelInputError` extends `Error`.
  Branching on `err.name === "RangeError"` breaks the same way: the name is now
  `"ModelInputError"`.
- **Breaking:** the safetensors distribution form is gone from the runtime (ADR 0108 §18):
  `openModel`, `KarumeModel`, `ContainerError`, `createSession(gpu, model)`, `prepareModel`,
  `createSessionFromShards`, `ModelShard`, `PreparedModel.createSession` and
  `estimateSessionMemory` are removed, and IR v1 JSON (`version: 1` with `tensor` / `storage`
  initializers) is no longer parsed. Open a `krm` with `openContainer` or build one in memory with
  `openMemoryContainer`, then `prepareContainer` → `estimate()` → `createContainerSession()`;
  estimate a graph you already hold with `estimateGraphMemory`. Capability shortfalls throw
  `RuntimeSupportError`; descriptor, binding and supply violations throw `ContainerFormatError`.
  `parseSafetensors` stays for plain safetensors assets, but the packed `I4` dialect dtype is
  rejected (`SafetensorsDtype` is a subset of the official set).
- **Breaking:** hub's shard streaming face is gone: `streamAssets`, `StreamedAsset`,
  `StreamAssetsOptions` and `DirectoryAdapter.readFileInto` are removed
  (`fetchAssets`, `prefetchAssets`, `openAsset` and `openContainerSource` remain).
- **Breaking:** exporter: `karume repack` is removed and `karume verify` checks containers only;
  `publish_model` / `export_to_file` require `provenance` and `graph_name` (the graph name must equal
  the manifest `weights` key — `karume dist` refuses otherwise), `WeightFiles.extras` is gone, and
  `tools/llm-speed` profiles distributions only.
- **Breaking:** `parseSafetensors` no longer takes a second `byteLength` argument; the whole
  `ArrayBuffer` is treated as the file, so trailing bytes past the data section are rejected as
  unused space. Callers that read a file into a larger reusable buffer must pass a tight
  `ArrayBuffer` (e.g. `buffer.slice(0, length)`).

## [0.12.0] - 2026-09-06

### Changed

- Long-context Gemma 4 is faster: `lm_head` (INT8, M=1) moved from the GEMM skeleton to the GEMV
  family (×5.0 on its own, decode GPU −21…24%), QK gained a D-parallel reduction variant behind
  the opt-in seat `stateAttentionReduce: "parallel"` (decode wall at P=16K −9…15%), and prefill
  plans (M ≥ 16) take a tiled GEMM-skeleton path by default (prefill wall at P=16K −64%). Token
  streams and goldens are unchanged; intermediate values of `Gemma4Pipeline` prefill differ from
  0.11.0.
- `fromPretrained` hub options — `onRetry` among them — are passed through uniformly for all
  eight model families.

### Added

- `evictCachedAssets` gained `CacheInventoryOptions.protect` and reports `EvictedAssets.alsoEvicted`.

### Fixed

- `evictCachedAssets` no longer keeps a sibling seat alive when it holds exactly the same
  reference set as the seat being evicted.
- BiRefNet's `fromPretrained` points at `BIREFNET_SOURCES` when no source is given, and the
  documented maximum binding size for BiRefNet was corrected.

## [0.11.0] - 2026-09-06

### Added

- Cache maintenance for hub: `listCachedAssets` lists what is cached and `evictCachedAssets`
  deletes by selection, with reference counting scoped to a single manifest.
- `LoadManifestOptions.onRetry` reports the fetch layer's retries (`RetryDiagnostic`).

### Changed

- hub depends on `@hdae/fetch-cache` ^0.7.0, which retries 429 / 503 responses following
  `Retry-After` (five attempts by default).

### Breaking

- **Breaking:** exceeding the receive limit on the Hugging Face source now throws `HubFetchError`
  (with the fetch layer's error as `cause`) instead of `IntegrityError`, and the 1 MiB limit on
  `karume.json` is applied after the whole body is received rather than from `content-length`.

## [0.10.0] - 2026-09-05

### Added

- BiRefNet HR and Lucida distributions, published as one repository per family with the model name
  carrying the resolution (`1024` by default, `2048` also available), reachable through the new
  `BIREFNET_SOURCES` table. The source tables now cover 7 families in 10 entries.

### Changed

- Intermediate GPU buffers are placed by static liveness packing with a preflight against the
  device's limits; the size-bucketed pool is retired. BiRefNet 1024² intermediates drop from
  6,283 MiB to 749 MiB, and 2048² becomes possible at ≈4.1 GiB total.
- The BiRefNet decoder tail applies the 1×1 convolution before the bilinear upsample, which
  removes a `cat`; the 1024² and 2048² goldens were re-baked.
- Several entry points reject more without accepting less: the four runtime execution knobs check
  their spelling, the estimate entry range-checks its arguments, family-argument and manifest
  checks moved to the call site, and the exporter rejects a distribution plan that places a
  generated payload under `weights`.

## [0.9.0] - 2026-09-04

### Added

- SigLIP 2 (base and so400m in one repository, base by default) and Depth Anything V2
  distributions. The source tables cover 6 families in 8 entries.
- `tools/opbench` gained `single` / `graph` / `torch` sub-commands and `tools/fusion-hints` gained
  `inductor`, for per-op measurement and fusion discovery against a reference compiler.

### Breaking

- **Breaking:** the per-repository pin constants `*_CURRENT` are replaced by per-family tables
  `<FAMILY>_SOURCES`, which map a source key to a repository, model and revision.
- **Breaking:** the Gemma 4 distribution repository was renamed, and repository naming,
  co-habitation of several models in one repository, and LICENSE / NOTICE bundling now follow one
  rule for every family.

## [0.8.0] - 2026-09-03

### Added

- Text generation for Gemma 4: the generation API surface, tokenizer, detokenizer and chat
  template, `Gemma4Pipeline`, `Gemma4ChatSession` with an injectable overflow policy (oldest turns
  dropped by default), a prefill progress hook, and an interactive chat example.
- `DistributionSource` — a local mirror can be read directly and cross-repository references are
  declared explicitly; `localDirectory` and the `@karume/hub/deno` entry point.
- Memory admission: the hard limit on a single buffer is checked deterministically before
  allocation, load reuses one staging buffer, and the Hugging Face path writes `into` a caller's
  buffer.
- Irodori v4.1-small, the split of the anima distribution into an official and an
  additionally-trained repository, and an intake command for single-file Civitai models.

### Breaking

- **Breaking:** shard specification v2 and v3 — graph-only shards, a single size limit of 256 MiB
  measured as file length, and tensors split into row-range pieces that are written into one
  parent GPU buffer at an offset. All distributions were re-baked with every tensor bit-identical.
- **Breaking:** Gemma 4 RoPE tables left the distribution: cosine and sine are supplied by the
  host, `capacity` and `chunkLength` became runtime knobs, and `position_ids` is gone.
- **Breaking:** `generateGreedy` left the public surface, `ANIMA_TURBO_CURRENT` folded into
  `ANIMA_CURRENT`, PLE residency is counted in bytes (`maxResidentPleBytes`), and anima's VAE
  tiling dropped its divisibility constraint, restoring eight accepted resolutions.

### Fixed

- NaN preservation is unified across the softmax family (`nan_max`) and the empty-row guard was
  ported into fused attention; strict canaries cover saturation ranges; `DEFAULT_TOLERANCE` is
  retired in favour of per-op measured tolerances and a bit-identity gate.

## [0.7.0] - 2026-08-29

### Added

- `prepareModel → estimate → createSession` as an explicit two-stage boundary, with
  `ResidentWeight` as a discriminated union and `planWeightResidency` as a pure planner.
- hub `prefetchAssets`, so a load can fetch every weight shard before execution starts.
- Cross-repository references extended to split components, with one reference per shard.

### Breaking

- **Breaking:** the exporter splits components larger than 1 GiB into deterministic shards, and
  all seven pipelines load graph-first through a shard-sequential surface. A 3.9 GB f16 model now
  loads in browsers that cap a single `ArrayBuffer`, which was previously impossible.
- **Breaking:** the estimator is redesigned as `AdmissionReport`, which reports prefill and decode
  scenarios separately and includes `peakAccountedBytes`.
- **Breaking:** the shard-sequential surface carries real asset ids, so a failure is attributed to
  the asset that caused it.
- **Breaking:** hub follows fetch-cache 0.5.0 — verification moves to the fetch layer,
  `AssetPhase` loses `verifying`, and `clearHubCache` changes what it clears.

## [0.6.0] - 2026-08-25

### Breaking

- **Breaking:** SBV2 input is layered in two — `Sbv2Phrases` → `toSbv2Utterance` →
  `Sbv2Utterance`, which `generate` takes as its first argument. The injection and dictionary
  seats are gone.

### Removed

- The `@hdae/yomi` dependency: the published dependency graph drops from four packages to
  `@karume/hub` and `@karume/runtime` only. Text analysis now happens in the caller, which is free
  to use any analyser.

Distributions, manifests and pins are unchanged; the WAV gate's three sha256 values are identical
across the change.

## [0.5.1] - 2026-08-25

### Changed

- anima ships with the Euler sampler as its distributed default again; DPM++ 2M stays selectable
  through `AnimaGenerateRequest.sampler`. Weights are byte-identical to 0.5.0 (verified by sha256
  over every file); the two anima pins point at the re-baked revisions.

## [0.5.0] - 2026-08-24

### Added

- A `scheduler.type` seat for anima with the DPM++ 2M sampler, family-independent in the runtime.
- An `AbortSignal` for Irodori construction.

### Breaking

- **Breaking:** manifest format `karume/4` — presentation fields, `requiredLimits`, and cross-repo
  component references. Quant seat names were renamed project-wide, and the `linearCompute` /
  `attentionCompute` value `"i8a8"` became `"a8"`.
- **Breaking:** `fromPretrained` requires `ref`. The verified revisions are published as `*_CURRENT`
  pin constants, and an implicit `main` warns.
- **Breaking:** anima's accepted resolutions narrowed to eight, and the two i4 seats of anima base
  are excluded from the distribution.

### Fixed

- The estimator reproduces reshape and identity-expand aliasing instead of counting the alias as a
  separate allocation.

Four distribution repositories were re-uploaded for this release.

## [0.4.3] - 2026-08-24

### Added

- An `AbortSignal` for anima construction, `approximatePreview` for an RGB preview of a latent,
  `animaLatents()` as a copy-returning accessor for the latent normalization constants, and
  per-file progress in hub's `AssetProgress`.
- A w4 seat for Irodori, whose i4 series is baked from the shipped bytes with adaLN rows kept at
  i8.

### Breaking

- **Breaking:** anima's accepted set gained a condition on the number of VAE tiles.

### Fixed

- Import safety: WebGPU globals are no longer read at module scope, so importing the runtime is
  safe where WebGPU is absent.
- Contract gates in the runtime: empty IR names, the number of scale axes, f32 rounding of `eps`,
  rejection of i4 for convolution inputs, and the device state required to accept a resident
  tensor.
- hub passes aborts through on the cache path, keeps `loaded` consistent, and self-heals a
  manifest; models wrap input errors in their family's error type, reject non-finite
  vowel-detector logits with coordinates, and make an anima abort take effect at stage boundaries.

## [0.4.2] - 2026-08-22

### Added

- An entry point that rebuilds a single Civitai file into a diffusers layout before export.

### Changed

- anima's default source pin moves to a revision that contains the i4 seats.

### Breaking

- **Breaking:** anima's distribution recipe is per-model and the pipeline splits in two, and model
  names now carry the upstream version.

### Fixed

- Overlay validation failures are wrapped in `Sbv2InputError`; recipes no longer import
  group-level dependencies at module scope.

## [0.4.1] - 2026-08-21

### Added

- anima's i4 series is baked with GPTQ calibration, using a calibration corpus kept separate from
  the evaluation inputs.

### Changed

- SBV2's injection seat was re-tuned after feedback from a consuming application.

### Breaking

- **Breaking:** linear's integer dot-product path accepts i4-resident weights (w4a8), which
  changes the output bits for `linearCompute: "i8a8"` combined with i4 residency. No published
  manifest declared that combination, which is why the change shipped in a patch release.

## [0.4.0] - 2026-08-21

### Added

- Calibrated rounding in the exporter core (`quant_calib`, GPTQ and AWQ), applied to the DeBERTa,
  Irodori and anima i4 series.
- i4 storage and execution for embeddings and for `conv1d(groups == 1)`.
- A tone injection seat for SBV2: an overlay dictionary, `given_tone`, and `analyzeProsody`.
- Default source pins in models — the commit SHAs of the three published repositories are baked
  in, which makes `ref` optional.
- LICENSE and NOTICE files bundled into the anima distribution repository.

### Changed

- SBV2's default distributed quant is `w8-bert4`.

### Breaking

- **Breaking:** manifest format `karume/3` — each dtype entry carries a shard field.

### Fixed

- A batch self-deadlock and a create/dispose race are rejected as exceptions instead of hanging or
  corrupting state; argmax and topk gained fault-injection gates.

## [0.3.0] - 2026-08-16

### Added

- `onEvent` generation events for the anima and Irodori pipelines.

### Changed

- The exporter is split into a general-purpose core, published as `karume`, and per-model recipes
  that live outside the wheel in a shared uv workspace. A boundary test fails if the core reaches
  back into a recipe.

### Fixed

- A large review wave: session mapping is one path with gates on outputs, order, values and ids;
  surplus tensors in a container are rejected; `deform`'s NaN constant expression became a
  parameter; batch leases and residency lifetime are gated; u32 range checks are unified inside
  codegen.

### Removed

- 81 exports that nothing used.

## [0.2.2] - 2026-08-10

### Changed

- The f32 / f16 GEMM skeleton's tile geometry is a parameter and defaults to the measured best
  (f16 ×1.28, f32 ×1.15, bit-identical output).

## [0.2.1] - 2026-08-10

### Added

- `onRunDiagnostics`, an observation seat for `Session.run` diagnostics, on both pipelines.

### Changed

- The i8a8 GEMM family's tile geometry is a parameter and defaults to the measured best
  (DiT ×1.23); i8a8 attention accumulators are statically unrolled (QK ×1.37, PV ×1.35); adaLN
  fusion folds four nodes into one dispatch.

## [0.2.0] - 2026-08-09

### Added

- SBV2 text-to-speech: the pipeline in `@karume/models`, and exporter support for building and
  carding an SBV2 distribution.
- A fusion pass that folds RoPE, SiLU and upsample2x into single dispatches.

### Breaking

- **Breaking:** manifest format `karume/2` — model and quant become two independent axes, and
  `dist` emits v2 and takes `--model` to assemble a family.
- **Breaking:** a distribution layout is always an independent copy; hard links are gone. Default
  repository names carry the `karume-` prefix.

### Fixed

- A silent wrong-value bug on Metal: shared B-tile component writes are static.
- Error scopes report the underlying out-of-memory error rather than the first validation error;
  staging buffers are released by submitting after the weight upload.

## [0.1.0] - 2026-08-05

### Added

- First release. `@karume/runtime` executes IR v1 graphs on WebGPU, `@karume/hub` resolves a
  manifest and fetches and caches its assets, `@karume/models` ships `AnimaPipeline` and the
  shared image layer, and the `karume` exporter turns a `torch.export` program into IR v1 with
  `export`, `dist` and `verify` sub-commands.
- A strict safetensors reader on the runtime's public surface, `clearHubCache`, `caches` and
  `fetch` injection for `fromPretrained`, and `Symbol.dispose` support on `AnimaPipeline`.
- Model cards are generated from the manifest when a distribution is assembled.

[Unreleased]: https://github.com/hdae/karume/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/hdae/karume/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/hdae/karume/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/hdae/karume/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/hdae/karume/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/hdae/karume/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/hdae/karume/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/hdae/karume/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/hdae/karume/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/hdae/karume/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/hdae/karume/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/hdae/karume/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/hdae/karume/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/hdae/karume/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/hdae/karume/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/hdae/karume/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/hdae/karume/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/hdae/karume/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/hdae/karume/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hdae/karume/releases/tag/v0.1.0

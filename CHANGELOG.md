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

## [0.14.0] - 2026-10-16

### Added

- Conversation forking. `GenerationContext.fork()` (`@karume/runtime`) returns an independent
  context with the same capacity that holds a copy of every state slot and the logical length;
  `GenerationSequence.fork()` and `Gemma4ChatSession.fork()` (`@karume/models`) build on it and
  also carry over the pending frontier token, the speculation gate's observations and, for the
  chat session, the history (a session whose KV cache is empty copies only the history and makes
  no GPU copy). Each branch is owned and disposed by the caller and does not depend on the
  original. Regenerating a reply is written as "keep the base session, fork it each time and
  `send` on the branch"; there is no separate `regenerate`. The same seed gives the same reply: the
  sampler's random stream is rebuilt from its `seed` (default 0) on every generate and the copied
  KV cache is bit-identical, so vary the seed per attempt (for example
  `send(text, { sampler: { ...spec, seed: attempt } })`) with a temperature above 0 (a temperature of
  0 is greedy and ignores the seed). Every branch adds one context's state
  bytes of VRAM. `fork()` is rejected while a run (including a drafter's run that reads the KV
  cache), a batch reservation or a deferred commit is pending, on a borrowing context, and after
  dispose; the sequence and chat-session versions throw synchronously when already disposed (the
  chat session also while a turn is running). Token sequences after a fork match the unforked
  sequence of operations without speculation and with `speculative: "always"`; with the
  self-tuning gate they match on devices where the speculation equality check passes.
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
  `--verify`) and quant seats derived from each upstream checkpoint's storage type (ADR 0113): no
  seat is wider than the checkpoint, and the default is the seat of the same width. The `lucida`
  distribution gains an `f16` seat next to `f32`, which stays the default; Lucida is stored in
  f32, so its `f16` seat is quantized. The `birefnet` (BiRefNet_HR) distribution carries `f16`
  only (see Breaking): the upstream checkpoint is itself stored in f16, so that seat is lossless.
  The NOTICE and model card say which seat is lossless, and the card's GPU memory figures are
  measured per checkpoint on the seat they name. For BiRefNet_HR the export now writes the `f16`
  series by default and refuses `--dtype f32`. Each compressed series records its export-time
  quantization error against the unrounded f32 outputs in `quality.json`, and the export refuses to
  publish an `f16` series of an all-f16 checkpoint unless every case is bit-identical, or a lossy one
  whose worst alpha MAE reaches half an 8-bit alpha step (a provisional line). At 1024² the
  BiRefNet_HR resident weights drop from 919 MiB (`f32`) to 509 MiB (`f16`), and to 431 MiB with the
  window-mask change under Changed. This release republishes both repositories with these seats
  (pinned in `BIREFNET_SOURCES`).
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
  the transformer and the two VAE chunk-graph containers `vae_decoder_first` /
  `vae_decoder_next`, and the precomputed text-embedding asset `text_embeds`). `karume-wan2.1` is
  not published on Hugging Face (`WAN_SOURCES` has no Wan2.1 key), so a local copy is passed as a
  directory source handle. `WanPipeline.fromAssets({ manifest, assets })` builds the same
  pipeline from bytes already held. The manifest's `pipelineConfig` declares the defaults for
  `steps`, `guidance` and `shift`, and a quant `session` key other than `linearCompute`,
  `attentionCompute` and `attentionScoreStorage` is rejected before weights are fetched.
  `WanRunComponent` is `text_encoder`, `transformer`, `vae_decoder_first` / `vae_decoder_next`
  and, for Wan2.2 image-to-video only, `vae_encoder_pre` / `vae_encoder_attn` /
  `vae_encoder_post` (the `stage` event's `component` `vae_encoder`; Wan2.1 never emits them).
  `generate({ prompt, negativePrompt?, seed | latents, steps?, guidance?,
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
  f16 score storage). `f16+dit8-a8-attn8-s16` is the default, so `WanPipeline` without `quant`
  runs it; pass `quant: "f16"` for the f16 transformer. On an Intel Arc B570 a 50-step, 33-frame
  clip took 952 s instead of about 30 minutes with `f16`, and an 81-frame clip 3,703 s instead of
  about 2 hours. `WanPipeline` accepts `linearCompute`, `attentionCompute` and
  `attentionScoreStorage` from the quant declaration.
- `karume-umt5-xxl` (ADR 0119 / 0122): a distribution of the umT5-XXL text encoder alone (`dist.py --pipeline umt5`,
  pipeline `umt5-encoder/1`, quant `i8`, 26 parts) with the tokenizer of `google/umt5-xxl` as the asset
  `tokenizer` (format `karume-umt5-tokenizer/1`, no prompt cleaning). It is published as
  `hdae/karume-umt5-xxl` and pinned in `UMT5_SOURCES["umt5-xxl"]`. The containers name the original
  `google/umt5-xxl` commit as their upstream (the Wan2.1 checkpoint's `text_encoder` holds
  bit-identical float32 encoder weights, ADR 0122), and the exporter series is `umt5-xxl-i8-dyn`;
  `wan.umt5_export` selects the umT5 upstream with `--upstream` and the Wan text snapshot with
  `--model`, and rejects an option the subcommand does not use. The Wan distributions reference its
  `text_encoder` across repositories and ship their own tokenizer asset `umt5_tokenizer` with Wan's prompt
  cleaning.
- `@karume/models/umt5` (ADR 0122): `Umt5Encoder` runs the umT5-XXL encoder on its own —
  `Umt5Encoder.fromPretrained(ref)` / `fromAssets`, `tokenize(text)`, and `encode(text, { signal })`, which
  returns the token ids and the last hidden states `[L, 4096]` (f32, row-major). The session stays
  resident until `dispose()`. The text is tokenized as is: out-of-vocabulary characters, in-text special
  tokens, `▁` right after a space, whitespace other than U+0020, a trailing space, more than 512 tokens and
  an empty text are rejected with `ModelInputError` (on those inputs `transformers` and `tokenizer.json`
  disagree, or the graph cannot take them).
- `WanPipelineOptions.textEncoder: "gpu" | "precomputed"` (default `"gpu"`): the GPU route runs umT5 and
  accepts any prompt (2–512 tokens; out-of-vocabulary characters, in-text special tokens, HTML entity
  candidates, mojibake, C1 controls and unassigned code points are rejected with a message that says how
  to fix the prompt); `"precomputed"` keeps the fixed-prompt embedding asset and does not fetch umT5.
  A distribution without the umT5 cross-repository reference needs `textEncoder: "precomputed"` (the
  error says so). Every Wan quant declares `requiredLimits` of 1,050,148,864 bytes for
  `maxBufferSize` / `maxStorageBufferBindingSize` (the vocabulary embedding), also on the
  precomputed route. `WanPipelineOptions.signal` and `WanGenerateRequest.signal` cancel loading and generation (the signal's
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
  frames from 5 to 121 (default 1280×704, 33 frames); the latents are `[48, F', H/16, W/16]` and
  `fps` is 24. 121 frames at 1280×704 have completed on a 12 GiB GPU (RTX 3080 Ti, Deno). On a
  GPU with about 10 GB such as the Intel Arc B570, clips up to 57 frames have been checked; longer
  clips are not supported there, as none of them was run there (a memory estimate puts 81 and 121
  frames beyond that GPU's limit; the other lengths were not estimated).
  Image-to-video: pass `image` (an `Rgb8Image`, now also exported from `./wan`) to
  `WanTi2vPipeline.generate` (`WanTi2vGenerateRequest`); the first latent frame is conditioned on the
  image as in diffusers' `WanImageToVideoPipeline`. Without `width` / `height` the accepted size
  closest to the image's aspect ratio is chosen (landscape on a tie). `fit: "crop"` (the default —
  the official Wan2.2 preprocessing: resize to cover with LANCZOS, bit-exact with Pillow, then a
  center crop) or `fit: "stretch"` (diffusers' direct resize) picks how the image is fitted; with
  only two accepted sizes, more of a square or 4:3 image is cropped than in the official
  pipeline. `WanPipeline` (Wan2.1) rejects `image` and `fit`. The distribution now also carries the
  VAE encoder (three f16 graphs, 108 MB), fetched for text-to-video too. The `karume-wan2.2`
  distribution is published as `hdae/karume-wan2.2` and pinned in `WAN_SOURCES["wan2.2"]`;
  `dist.py --pipeline wan-ti2v` still assembles a local mirror for `fromPretrained`, and
  `fromAssets` builds the pipeline from a manifest and the exported series containers. The
  distribution has two quants, `f16+dit8-a8-attn8-s16` (the default: int8 transformer with int8
  activations) and `f16+dit8` (int8 transformer weights, f32 compute — the reference quant), and no
  `f16` quant; its manifest declares 50 steps, guidance 5.0 and shift 5.0, and like `karume-wan2.1`
  it references the `text_encoder` of `karume-umt5-xxl` across repositories.
  `deno task demo:wan --generation wan2.2` runs it (the default generation stays `wan2.1`), and
  `--image <PNG|JPEG>` with an optional `--fit <crop|stretch>` runs image-to-video; the example
  decodes the file with fast-png and jpeg-js, which are dependencies of the examples only (PNG gives
  Pillow's pixels exactly, JPEG differs slightly). The gpu-lab Wan tab gains a generation choice
  and, for Wan2.2, a condition image decoded by the browser (`createImageBitmap`); its JSON
  records the generation and the condition image of an image-to-video row. Both the example and the gpu-lab tab apply the EXIF orientation of a JPEG or PNG, so a
  phone photo tagged to be shown upright is used upright, and both pick the same "auto" size for
  it. This intentionally differs from the official `generate.py`, which does not apply the
  orientation; an image without an orientation tag gives the same pixels either way.
- Measuring the seams of the Wan transformer loop (the host work, uploads and readbacks between
  the transformer forwards). Each row of the gpu-lab Wan tab gains `seams` (per-step spans split at
  each forward's end, their medians from step 2 on, and the VAE tile-assembly time) and, for
  Wan2.2, `seamInputCheck` (the transformer's owned input bytes against the expected total); a new
  Wan2.2-only **切れ目の下見** (seam probe) button measures one step's host work on full-size arrays and an
  upload and readback of the same bytes as one forward, kept in the top-level `seamProbe[]`. The
  JSON is now `karume-wan-browser/6`. `tools/wan-seams/main.ts` is the Deno twin: it runs the probe
  and a text-to-video generate per frame count and writes `karume-wan-seams/1`.
- Release tooling: `tools/release/hf-upload.zsh upload` first reads `provenance.license` from every
  container of the directory (`tools/release/container_license.ts`, needs Deno) and uploads nothing when
  one carries the undeclared-license mark (`NOASSERTION`, or a value such as `unknown`, in any case), when
  a part 0 cannot be read, or when the directory has no container. A direct `hf upload` bypasses this
  check.
- `SessionOptions.attentionRowBlockMaxBytes` (`@karume/runtime`): an optional cap, in bytes, on each
  temporary buffer of the attention row blocks (the score matrix of one block and its row statistics,
  each — not their sum). Attention that materializes the score matrix (the fused `attention` op, the
  state form, and the decomposed attention window) already splits the query rows into blocks that fit
  the device's `maxStorageBufferBindingSize`; with this option the blocks also fit the given budget,
  which lowers the memory peak at the attention step in exchange for more dispatches. Outputs are
  bit-identical to the run without a budget, and omitting the option leaves pipelines, keys and
  memory unchanged. A budget smaller than one query row of a graph is rejected when that shape is first
  planned (a `run`), naming the budget; non-positive or non-integer values are rejected at session
  construction. The pipelines (`@karume/models`) do not expose it.
- `EstimateOptions` accepts `attentionRowBlockMaxBytes`, `attentionCompute` and
  `attentionScoreStorage` with the same values and checks as `SessionOptions`, so the estimate splits
  attention the way the session will (omitted seats are estimated as `"f32"`, the runtime default).
- `@karume/models`: `UMT5_SOURCES` (`@karume/models/umt5`) and `WAN_SOURCES` (`@karume/models/wan`),
  both also in the barrel and folded into `KARUME_SOURCES`, pin the distributions first published
  with this release: `UMT5_SOURCES["umt5-xxl"]` is `hdae/karume-umt5-xxl` and
  `WAN_SOURCES["wan2.2"]` is `hdae/karume-wan2.2`. Pass them to `Umt5Encoder.fromPretrained` /
  `WanTi2vPipeline.fromPretrained` to follow the revisions this package version was verified with;
  calling those without `ref` names the table in the error. `karume-wan2.1` is not published, so
  `WAN_SOURCES` has no Wan2.1 key and `WanPipeline` still takes a local copy.
- Exporter (`karume`): `UnsupportedAtenOpsError` gains `reports`, one `UnsupportedOpReport` per
  unsupported op. Each report gives the op's occurrence count and its Core ATen membership
  (`core_aten` / `non_core_aten` / `non_aten`, read from the installed torch's `torch.Tag.core`
  rather than a hand-kept list). It also gives the innermost modules the op came from
  (`nn_module_stack`, grouped by class with the first path), plus the number of nodes that have no
  module stack. The error message lists the same facts, with a legend that says when adding the op
  to the vocabulary needs an ADR (Core ATen ops included when they break an execution-model
  invariant). The existing `.ops` mapping keeps its shape.
- Gemma 4 recipe: the regular (non-QAT) recipe takes `--model e2b|e4b` like the QAT recipe (default
  `e2b`, unchanged). This builds E4B series and an `e4b` entry of `karume-gemma4`
  (`dist.py --pipeline gemma4 --model e2b --model e4b`). E4B ships no drafter, and it reuses E2B's
  tokenizer series once their upstream sha256 match. The checkpoint is now loaded into a model built
  on the meta device, so loading no longer holds a second f32 copy of the weights (about 35 GiB for
  E4B). Graphs and weights are unchanged. The published `karume-gemma4` still carries E2B only.
- `tools/gpu-lab` (`deno task bench:gpu-lab`): a local Chrome (WebGPU) page. **1. 掃引** (sweep)
  runs `runGeometrySweep` and saves the `karume-geometry-sweep/2` record; **2. プロファイル**
  (profile) derives a `GeometryProfile` from one or more sweeps of one adapter (the rules of
  `deriveGeometryProfile`) and exports it as TypeScript, injection JSON or a registration command;
  **3. Anima** generates on an injected, built-in or stored-and-matched profile, records per-stage
  wall and GPU time, the DiT residency and the PNG sha256, and A/Bs the default geometry against
  the applied table; **4. Wan** runs Wan2.1 / Wan2.2 and saves a clip as a zip of numbered PNGs
  (its JSON is `karume-wan-browser/6`). The server listens on loopback only, sends cross-origin
  isolation headers, serves the local `models/` mirrors read-only (`--source`, `--wan-source`,
  `--wan22-source`, `--umt5-source`) and refuses to start when an explicit directory has no
  `karume.json`.
- `tools/geometry-sweep`: the Deno CLI of the GEMM tile-geometry sweep
  (`deno run -A tools/geometry-sweep/main.ts [--set quick|quick+|full] [--op …] [--case …]`,
  default `quick+`; the record is `karume-geometry-sweep/2`) and its `profile` subcommand, which
  derives a geometry profile from sweep records and writes the formatted table to
  `packages/runtime/src/kernels/geometry-profiles/<id>.ts`. `tools/anima-residency/profile.ts` is
  the Deno twin of the gpu-lab Anima tab: it builds Anima once with `residency: "transformer"`,
  prints each stage's wall and GPU time and writes the tab's JSON format.
- `examples/wan` (`deno task demo:wan`): a one-shot CLI for `WanPipeline` and, with
  `--generation wan2.2`, `WanTi2vPipeline` that writes one clip as numbered PNG frames under
  `outputs/examples/`. Without `--source` it reads the local mirror (`models/karume-wan2.1` /
  `models/karume-wan2.2`) and prints the `dist.py` command when it is missing; `--source` takes
  another directory or a Hugging Face repository (its `main` revision). `--quant` takes a key of
  the manifest's `quants`, `--text-encoder gpu|precomputed` picks the route, `--prompt` /
  `--negative` take free text or a name from the embedding asset, `--umt5-source` points at a local
  umT5 mirror, and `--swap-text-encoder <dir | owner/name@<40-hex commit>>` replaces umT5 with a
  compatible encoder (a bare repository name is rejected). The output directory names the quant
  (`default` without `--quant`), the prompt and the route. Contradictory flags and out-of-range
  values are rejected before any weight reaches the GPU.

### Changed

- `karume-birefnet-hr` / `karume-lucida` (republished with this release): the Swin shifted-window
  attention masks are stored as a table of the 4 distinct windows plus a per-window index and built in
  the graph, and the zero padding is no longer baked as constants. Outputs are bit-identical. The
  resident weights drop to 431 MiB (HR `f16`, was 509 / 706 MiB at 1024² / 2048²) and 842 MiB (Lucida
  `f32`, was 919 / 1,116 MiB), and the 1024² and 2048² models of each repository share their weight
  parts, so a repository is less than half its former size.
- `@karume/models` ships `THIRD_PARTY_NOTICES.md`: the functions ported from diffusers (Apache-2.0)
  and Pillow (MIT-CMU) are listed with the upstream license texts, and each ported function carries
  a notice comment naming its upstream source.
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
- The BiRefNet 2048² GPU tests (`birefnet-hr-2048-f16`, `lucida-2048` and `lucida-2048-f16`) are
  skipped explicitly on the Intel Arc B570 environment key (`deno-intel-graphics-bmg-g21`), where
  one `deform_conv2d` dispatch exceeds the Linux xe driver's 5 s job limit and the resulting
  device-lost panic would stop the whole test process. The skip is per environment key and
  announced at registration; the asset-completeness test still runs, and other machines run the
  series as before.
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
  Chrome gets a profile measured on an RTX 5070 Ti that changes the i8a8 linear and attention
  tiles (the 512² DiT stage of the default quant took 1.14 s of GPU time instead of 1.36 s) and the
  linear / matmul / bmm rows for M 129–512; f32 attention, conv2d and M > 512 keep the default.
  Outputs stay
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
  time either way, because the NVIDIA profile changes only the i8a8 tiles and the M 129–512 rows)
  with identical PNG bytes.
- `@karume/runtime`: the memory estimate (`estimateGraphMemory` / `PreparedModel.estimate`) now
  counts the internal temporaries of the fused `attention` op — the score matrix and row statistics of
  each row block, and the int8 quantization buffers of `attentionCompute: "a8"` — with the same
  functions the session uses, so `workspaceBytes` and `peakAccountedBytes` can grow for graphs with
  that op (for example the Wan2.2 transformer). Combinations a session rejects (an additive mask or
  grouped query attention with `"a8"`, a head dimension or key length beyond the int32 accumulation
  bound of the int8 path, a non-f32 seat on the state form) are now rejected by the estimate too.
  Without `maxStorageBufferBindingSize`, some graphs with that op are now rejected (see **Breaking**).
  The second entry of `unaccounted` now names only the linear int8 quantization buffers. The estimates
  the Anima and Gemma pipelines use are unchanged.
- BiRefNet_HR: `segment()` now scales the matte back in the order its upstream uses — the sigmoid is
  rounded to 8 bits at the graph's resolution, then resized with 8-bit bicubic resampling that
  matches Pillow's `BICUBIC` bit for bit (`postprocess: "u8-pil-bicubic"`). It used to take the
  floating-point antialiased bilinear path that Lucida's upstream uses (`"float-bilinear"`, still
  Lucida's). Rounding stays half-up (upstream truncates). Only the contour changes, and only when the
  image size differs from the model's resolution: on the seven of eight test images with a detected
  foreground (the eighth had none and did not change), the alpha inside the contour band moved by
  0.3–2 steps on average and by up to 25 steps (8-bit), and all eight pairs were indistinguishable
  in a blind A/B. At the model's own resolution the output is effectively unchanged (ADR 0125).
- `@karume/runtime`: the parallel GEMV path now covers the linear shapes of Gemma 4 E4B, regular and
  QAT. This path is `linearGemvReduce: "parallel"`, together with the `fuseLinearStaticQuantize` /
  `packedStaticQuantize` paths that depend on it. Before, every E4B linear silently ran the
  sequential kernel, so these options had no effect on E4B. The E4B entries reuse the lane counts
  of E2B's matching layers; they were not tuned on E4B.
- `@karume/models`: the pinned revisions move. `ANIMA_SOURCES` (`anima`, `anima-extra`) points at
  revisions that only update `README.md` / `NOTICE.md` (which now carry the rights-grant text);
  `karume.json` and the weights are byte-identical. `IRODORI_SOURCES` (both repositories, in the
  `dit_context` layout), `GEMMA4_SOURCES["gemma4"]` (whose `i4-gemvpar` / `i4-fast` quants now
  declare `stateAttentionReduce` `"parallel"` / `"parallel-fused"`) and `BIREFNET_SOURCES`
  (`birefnet-hr`, `lucida`, as `birefnet/2`) point at the repositories republished with this
  release (see **Breaking**).
- Exporter (`karume`): with the default `include=None`, `fake_quant_int8` now also rounds
  `nn.Conv3d` weights (per output channel, like `Conv2d`), to go with the new `conv3d` op. If a
  model uses a `Conv3d` weight in any form other than the `conv3d` op (for example on the host),
  narrow the targets with `include`, or that weight is rounded too.
- BiRefNet recipe: `python -m birefnet.export` now finds the checkpoint by the revision in the
  `--model-dir` download record, and falls back to the directory name only when that revision is
  not in the table. It fails when the two point to different checkpoints. Before, a renamed
  BiRefNet_HR directory could still write the retired `f32` series. `dist.py --pipeline birefnet` /
  `--pipeline lucida` now require every container to name that checkpoint's pinned upstream
  revision. Before, the containers only had to agree with each other. So a series from the other
  checkpoint or from another revision is refused, even when the distribution has a single seat.
  The default seat must be the checkpoint's own storage type, and anything else raises
  `DistError`. The recipe functions `export_series`, `default_out_dir` and `birefnet_series_name`
  no longer default `dtype` to `f32`.
- Release tooling: the fragmentation table of `tools/release/hf-upload.zsh` gains `ranges=`, the
  number of byte ranges a download actually fetches (from the reconstruction's `fetch_info`), next
  to `terms=`, so repeated references such as all-zero chunks can be told from real
  fragmentation. A reconstruction response without `fetch_info` is now reported as `FAILED`; the
  MiB/term verdict is unchanged.

### Fixed

- BiRefNet model cards (`karume-birefnet-hr` / `karume-lucida`): the GPU total had been converted at
  1,000 MiB per GiB, so the published `f32` figures (about 1.7 / 4.1 GiB at 1024² / 2048²) were
  overstated. The figures were re-measured on 2026-10-16 after the window-mask change (Lucida `f32`
  about 1.6 / 3.8 GiB, BiRefNet_HR `f16` about 1.2 / 3.4 GiB at 1024² / 2048²) and now count
  weights, intermediates, input and output readback; the cards republished with this release carry
  them.
- `@karume/runtime`: a long `run` no longer lets the host queue an unbounded number of command
  buffers ahead of the GPU. Between steps it places a completion marker every 64 submits since the
  last fence and waits once more than two markers are outstanding (under Deno the marker itself
  blocks until the queue drains). `enqueue` is unchanged: it neither places markers nor waits, so
  non-awaited enqueues keep their call order. This fixes a device loss midway through very long runs
  (the Wan2.2 reference quant at 1280×704×121 on an RTX 3080 Ti under Deno). Outputs are
  bit-identical.
- `@karume/runtime`: work on a lost or destroy-requested device now fails at the call instead of
  appearing to succeed. Session construction checks the device on entry and again just before it
  resolves; a graph that uploads no weights (no initializers, or only `sharedWeights`) used to
  return a Session bound to a lost device. `run`, `enqueue` and `enqueueRead` reject with
  `GpuDeviceLostError` at admission; a run or enqueue without outputs could resolve successfully on
  a lost device. If the device is lost while `GpuContext.beginBatch` waits for an earlier batch, it
  no longer opens a batch that holds the batch lock. If the device is lost while
  `GpuContext.createResident` checks its allocation, it no longer returns an unusable tensor whose
  failure only showed at the first write or read.
- `@karume/runtime`: recovery after a non-fatal out-of-memory error. When the GPU work of a `run`
  or batch fails, for example because an allocation ran out of memory, the runtime now drops the
  params, execution plan and intermediate buffers that call newly cached. Before, the same `run`, or
  a later batch of the same shape, kept failing with validation errors after memory was freed,
  until the entry was evicted or the Session was rebuilt. Host-side failures (invalid inputs,
  pre-run checks) keep the cache as before. The dp4a canary behind `attentionCompute: "a8"`
  likewise no longer remembers a transient out-of-memory failure for the life of the `GpuContext`.
  A numerical mismatch or a device loss is still remembered.
- `@karume/runtime`: Session construction copies its `SessionOptions` (including `submitPolicy`) on
  entry and reads only the copy. Changing the options object while construction awaits can
  therefore no longer build a session with a combination the checks reject, such as
  `linearGemvReduce: "sequential"` with `packedStaticQuantize: true`. The submit scheduler also
  keeps its own copy of the policy. `DEFAULT_SUBMIT_POLICY` is now frozen: its `readonly` type
  already disallowed assignment, which now throws in strict mode. Before, an assignment silently
  changed every session that used the default and could remove the submit cap.
- `@karume/runtime`: a weight borrowed through `SessionOptions.sharedWeights` must now have the same
  declared dtype as the lender's weight. A mismatch throws `ExecutionError` at construction, before
  any buffer is allocated. Before, a borrower could declare an f32 weight as i32 or bool and get its
  bytes reinterpreted (1.0 read as 1065353216). The only published pair, the Gemma 4 E2B drafter's
  `lm_head`, is f32 on both sides and is unchanged.
- `@karume/runtime`: when a kernel's 32-bit index math would wrap, the session is no longer built
  and it fails with an error whose `name` is `CodegenError`, naming the limit; it used to read or write the wrong elements
  silently. This covers `linear` (including int4 `w4a8`) and `embedding` weights with ≥ 2^32
  logical elements; conv inputs, batched outputs or weights with ≥ 2^32 elements; `conv1d` /
  `conv2d` axes where input length + 2·padding > 2^31 − 1; and `conv_transpose1d` kernels, input
  lengths or output length + padding > 2^31 − 1. A grid-stride dispatch whose element count +
  stride > 2^32 (a GPU hang before) now throws `DispatchLimitError`. `upsample_bilinear2d` now
  clamps its last source index to `in − 1`, so an axis longer than about 2^24 stops reading the
  next plane at the edge. Published models are far below these limits, and their outputs and
  sha256 references are unchanged.
- `@karume/runtime`: three memory-estimate fixes. A decomposed attention window that the session
  runs in row blocks is now counted in its fused form. A graph whose unfused score matrix exceeds
  the binding limit therefore gets an estimate instead of an exception, and such a window adds its
  output to `workspaceBytes`. `peakAccountedBytes` now includes every prefill bucket shape allowed
  by `generation.chunkBuckets`; a bucket can need more than the full-length prefill and was
  underestimated. Gemma 4's `estimateSessionMemory` passes its buckets, and the default E2B
  estimate is unchanged. Attention row blocks, in the session and in the estimate, are now sized by
  the larger of a score row and its row statistics. Degenerate shapes, such as key length 1 in f32
  or below 4 in f16, that fit when split are no longer rejected at allocation.
- `@karume/runtime`: stricter container and IR input checks. `openContainer` rejects, before
  reading it, a `krg` whose graph descriptor declares a constant region above the 1 GiB part
  ceiling. The container binding behind `openContainer` / `openMemoryContainer` keeps graph,
  initializer and supply names in null-prototype maps, so a name like `__proto__` no longer
  replaces a map's prototype in browsers: a weighted graph used to fail with a missing supply, and
  a weightless graph vanished from the result. A quantized weight (`int8-sym` / `int4-sym-g`) whose
  row count is 0 is rejected with an explicit message; int4 used to fail later in CPU decoding with
  a NaN row length. `parseIrDeclarationValue` rejects arrays with holes (for example `new Array(1)`
  as a shape) with an `IrError` naming the path. Input from `JSON.parse` has no holes, so file input
  is unaffected.
- `@karume/hub`: `prefetchAssets` now emits `complete` exactly once for every ref it is given,
  including refs served by a local directory source, which it still does not read or cache. Model
  loading measures progress against the total size of every file it fetches. From a local mirror,
  or a setup mixing HF and local sources (swapped components, cross-repository references), the
  last `loaded` used to fall short of that total, and a local distribution without assets sent no
  progress at all. Progress now reaches the total.
- `@karume/hub`: an `onProgress` listener that throws no longer fails the load. Every progress
  notification of `fetchAssets` / `prefetchAssets` now isolates the listener and reports the
  exception with `console.warn`. This covers the progress that `@karume/models` loads aggregate
  through these calls. Before, an exception thrown on a `downloading` notification was only warned,
  but one thrown on `complete` rejected the whole load and stopped the other files' downloads.
- `@karume/hub` cancellation: `prefetchAssets` checks the `AbortSignal` again after each file is
  warmed, so during a warm-cache prefetch, files already in flight (up to four) no longer emit
  `complete` after the abort. `openAsset` also checks the signal after the file is opened. A
  warm-cache hit makes no network request, so an abort during the open used to still return a
  reader; it now rejects with the signal's reason.
- `@karume/hub`: the reader `openAsset` returns now rejects any `read` outside the ref's declared
  `size`, or with a negative or non-integer offset or length, from every source. Before, only the
  local directory source checked this. On HF, a ref that declared a smaller `size` for the same
  sha256 could read the rest of a warm cache entry beyond its declaration.
- `@karume/hub`: an HF source copies the `repo`, `hubUrl` and `revision` of the `HubRepoRef` given
  to `loadManifest` when it is created. Mutating that object afterwards, even while the revision is
  still resolving, no longer sends later fetches to another repository or host while `loaded.repo`
  still names the original.
- `@karume/hub`: building the `available` labels for a `parseManifest` error took time quadratic in
  the number of `weights` entries. It ran before the size-limit check, so one manifest under 1 MiB
  could block the main thread for a long time. The scan is now linear. The `models`, `quants` and
  `weights` lists in `available` are cut at the table size limit (32 each), and the error message
  says so when they are cut. A manifest that passes the checks always gets complete `available`
  lists.
- `@karume/hub` is now free of import-time side effects. `DistributionSource` and `LoadedManifest`
  no longer bind their internals in `static` blocks that run when the module is evaluated. A bundle
  that imports only `parseManifest` from the barrel used to keep both classes; now they are
  tree-shaken. A hand-built or copied handle that never went through the constructor now fails
  with an `Error` that says so, where it used to fail with a bare `TypeError`.
- `@karume/models`: Gemma sampler and stop settings are type-checked before use, which closes
  silent misbehavior with untyped JS / JSON input. `logitBias` must be an array of `[token, bias]`
  pairs whose bias is a finite number or `-Infinity`: a string bias such as `"0"` used to be
  concatenated into the logit (`1 + "0"` = `"10"`), and `null` / `true` were read as 0 / 1.
  `stopStrings` must be an array of strings without holes: `"END"` used to be split into
  characters, `[123]` matched `"123"`, and holes in a sparse array were dropped. Violations throw
  `ModelInputError`; `chat` and `Gemma4ChatSession.send` check before copying the settings.
- `@karume/models/anima`: classifier-free guidance multiplies by the scale rounded to f32, as torch
  does, in Euler and DPM++ 2M; a scale not exact in f32 (e.g. 1.1) now matches torch in the last
  bit, while the published defaults (1 / 4) give byte-identical images. A `guidanceScale` finite in
  f64 but infinite in f32 throws `ModelInputError` before GPU work, and each denoising step checks
  the updated latent: a non-finite value (e.g. guidance 1e38 overflowing in the CFG mix) throws
  naming the step instead of reaching the VAE, where it could be clamped into normal-looking
  pixels.
- `@karume/models`: `normalizeToNchw` rejects with `ModelInputError`, before allocating the output,
  a `mean` / `std` that is finite but overflows f32 during normalization (e.g. `std` =
  `Number.MIN_VALUE`, or `mean` = `Number.MAX_VALUE`). Such values used to return planes of
  `Infinity` / `NaN`. Ordinary model coefficients give the same values as before.
- `@karume/models`: tokenizer and symbol-table assets are validated more strictly at load, so a
  malformed or tampered distribution fails instead of producing wrong ids: Irodori added tokens
  must lie inside the vocabulary and match its spelling, and its 256 byte tokens must be
  `<0x00>`…`<0xFF>` in order; Gemma / EmbeddingGemma added tokens may not share an id or reuse an
  in-vocabulary id under another spelling; SBV2 `symbols.json` must keep `blankId` inside the tone
  and language tables, and its relative-position settings must define the log buckets (an
  undefined combination produced NaN gather indices that became 0). The published distributions
  pass every check.
- `@karume/models`: construction rejects more mismatched graphs and settings before any GPU work:
  SigLIP 2, BiRefNet and Depth Anything graphs whose `pixel_values` is not f32, a SigLIP 2 pooled
  output that is not f32 `[1, hidden]`, image mean / std declared in those manifests or an Irodori
  `initScale` that overflows f32, and generation graphs that reuse the input-length symbol for the
  selected rows. These used to fail only after the device, sessions and weights were set up (or
  after a GPU run, on non-finite output).
- `@karume/models`: `fromAssets` without a `gpu` now checks the quant's declared `requiredLimits`
  against the device it acquired, as `fromPretrained` and a shared `gpu` already did, in Anima,
  SBV2, Irodori, SigLIP 2, BiRefNet, Depth Anything and the vowel detector. A device below the
  declaration fails at construction with a message naming the limit (the acquired device is
  destroyed) instead of later in session construction with a low-level buffer error, or not at all
  when the declaration exceeded the real need.
- `@karume/models`: `decodeWav` rejects with `ModelInputError` a RIFF container whose chunk walk
  does not end exactly at the RIFF logical end: a 1–7 byte fragment left before the end (such as a
  4-byte `JUNK` header), which also catches headers misread after an odd-length chunk written
  without its pad byte. A last chunk missing only its pad byte is still accepted.
- `@karume/models`: text and audio front ends no longer block the thread for seconds on
  pathological inputs; outputs are unchanged. Anima's Qwen2 BPE uses Gemma's merge queue (an
  8,000-character unbroken piece took about 1.1 s); added-token matching uses a trie built once per
  tokenizer (Gemma, Irodori, Anima, Wan); Irodori strips nested outer brackets in linear time; SBV2
  spreads phonemes over characters in closed form (a 200,000-character word took about 19 s); the
  vowel detector rejects over-limit audio before feature extraction (a 600 s clip took about 3.5 s
  first). A single Gemma text fragment beyond about 130,000 tokens reaches the capacity check
  instead of throwing an engine `RangeError`.
- `@karume/models`: loading and failure paths. A manifest whose selected containers share a part
  file now loads (the prefetch plan listed it twice and failed with `ManifestReferenceError`);
  `components` replacements are looked up by own keys only, so inherited properties are ignored;
  when a Gemma 4 PLE row read fails, no new reads are issued and the error is raised after the
  in-flight reads settle, keeping the 16-read concurrency limit; a vowel-detector session dispose
  that fails after a run error no longer replaces it (an `AggregateError` carries the run error
  first).
- Exporter (`karume`): constant folding no longer bakes in values that depend on the length of a
  symbolic axis. Folding now excludes by structure the forms whose result depends on a position
  along a symbolic axis: indexing that axis with an index tensor (`arange(T)[tensor([-1])]`),
  slicing it with a negative start (`arange(T)[-2:]`), `cat` along it, `repeat` of it more than
  once, and `bmm` contracting over it. The two-point check can pass by coincidence on these forms,
  so such exports used to succeed and then silently give different values at other lengths. These
  forms now lower to runtime ops, or fail loudly as unsupported ops. As a result, `cat` / `bmm`
  along a symbolic axis, which the two-point check used to reject as non-commuting, now export.
  Indexing with a boolean / uint8 mask is also excluded from folding; it already failed before,
  and now fails as an unsupported op. In the exporter and recipe test suites, the folding of
  existing graphs is unchanged.
- Exporter (`karume`): after re-reading what it wrote, `publish_container` now applies the
  whole-graph IR acceptance rules (duplicate outputs, forward references, unbound symbols). This
  covers every path through it: `publish_model`, graph surgery and `karume migrate`. On a violation
  it raises `PublishError` and installs nothing. Before, a graph edited after export could be
  published as a `.krm` that the runtime rejects. Distributions were not affected, because
  `karume dist` and `karume verify` already ran these checks. Also, `to_external_states_form` now
  derives `symbols` again after it removes the K/V inputs. Before, a length symbol that only those
  inputs bound was left unbound, and both the Python and the runtime parser rejected the graph.
- Exporter (`karume`): the Python container reader and writer, and `karume verify`, now accept
  exactly what the runtime accepts. Newly rejected: `rowAxis: 1` for the `i4` / `i2` / ternary
  codecs (including when an explicit `Encoding` is given to the writer); piece part numbers that go
  down; `version: true` and `packing.blockElements: true`; the JSON key `__proto__` anywhere, and
  empty asset names; block counts (const plus weight blocks) above the runtime's limit; non-ASCII
  digits in dimension expressions (such as `'T+２'`). The runtime already rejected all of these, so
  such files used to pass in Python and fail only in the browser or Deno. Newly accepted:
  integer-valued floats in integer fields (`0.0`, `1e0` in `permute` dims and in container
  `index` / `offset` / `length` / `rowAxis`), which the runtime reads as integers and Python used to
  reject.
- Exporter (`karume`): `karume dist` and `karume migrate` now check names the way `@karume/hub`
  does. Python's `$` also matches before a final newline, so model names, path segments,
  repositories, revisions, graph names, sha256 digests, pipeline names and block ids with a
  trailing newline (`'owner/name\n'`) used to pass, and are now rejected. The Anima recipe's
  Civitai model names are checked the same way. Repository segments that start with `.` (including
  `.` and `..`) in `--repo`, `--cross-repo` and cross-repository references are now rejected.
  Quant `label` / `description` now follow the hub parser: `null` is rejected (it used to be read as
  "not set" and produced a manifest the hub rejects), and empty, whitespace-only and multi-line
  values are now accepted, as the hub accepts them.
- Exporter (`karume`): the model card that `karume dist` writes now identifies a file by
  repository, revision and path, as the hub does. Two components can reference the same relative
  path from different repositories or revisions (own vs. borrowed); the card used to count only one
  of the two files, so the download size was understated and a borrowed source could drop out of
  the attribution. The per-container asset sizes are fixed the same way. In the quant table, `|` in
  a label or description is escaped, and runs of whitespace (newlines included) are folded into
  one space, so a row no longer splits or shifts columns. A blank field is shown as `—`.
- Exporter (`karume`): `karume migrate` checks its input more strictly. Paths in an old
  (`karume/4`) manifest go through the hub's path grammar, and a declared file whose real path
  (after symlinks) is outside the input repository is refused, both before anything is written;
  before, `../` or absolute paths could place new `.krm` files outside the output directory. Each
  legacy tensor's safetensors dtype and shape are checked against the IR declaration (weights,
  scales, and PLE shard values and scales); before, 4 bytes of `I32` were accepted as an f32 `[1]`
  weight and silently reinterpreted. A stored dtype the old writer never produced is rejected with
  a message that names it. Whole-repository migration is assembled in a staging directory and
  installed only when every component, every copied file and the manifest succeed; before, a
  failure left a partial output that blocked the same command from running again.

### Breaking

- `@karume/models`: `GenerationSequence` gains a required `fork()` method. Code that implements the
  type itself (for example a `Gemma4ChatSessionHost` whose `sequence()` returns its own object) no
  longer type-checks until it provides `fork()`.
- `@karume/runtime`: `SessionDiagnostics` gains the required fields `geometryProfile` and
  `lastRunPipelines`. Code that receives the type is unaffected; code that builds a
  `SessionDiagnostics` itself (a test double, for example) must supply them.
- `@karume/models`: the non-Gemma families now reject, before fetching weights, a manifest `session`
  key their family does not accept, where 0.13.x passed it through to the runtime (see **Changed**).
  Every published quant is accepted unchanged; only a locally built or hand-edited distribution that
  declares such a key stops loading.
- `@karume/hub`: the manifest `session` vocabulary gains `stateAttentionReduce` (`"sequential"` /
  `"parallel"` / `"parallel-fused"`), composed like the other quant knobs (explicit > quant
  declaration > runtime default). Readers of 0.13.x reject a manifest that declares it as an unknown
  `session` key, so a distribution rebuilt with the new recipes needs this release's hub / models.
  `karume-gemma4` is republished with this release and declares the key (`GEMMA4_SOURCES` pins that
  revision), so 0.13.x readers reject the new revision; their own pinned revisions are unaffected.
  `karume-gemma4-qat` is not published.
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
  and older releases cannot run the new layout. This release republishes both Irodori repositories
  in the new layout and pins `IRODORI_SOURCES` to them; 0.13.x keeps its own pins to the old layout.
- `@karume/models`: `IrodoriRunComponent`, and with it the Irodori `stage` event's stage names, gain
  `"dit-context"` (run once per generation, before the DiT loop). TypeScript code that switches
  exhaustively over the previous eight names no longer type-checks.
- `@karume/models/birefnet`: the pipeline contract is now `birefnet/2`, and `pipelineConfig` gains a
  required `postprocess` (`"u8-pil-bicubic"` or `"float-bilinear"`; anything else is rejected).
  This release's `BirefnetPipeline` rejects a `birefnet/1` distribution — including the
  `karume-birefnet-hr` / `karume-lucida` revisions published with 0.13.0 and earlier — as an
  unsupported major; rebuild a local mirror with `dist.py --pipeline birefnet` /
  `--pipeline lucida`. This release republishes both repositories as `birefnet/2` and pins
  `BIREFNET_SOURCES` to the new revisions.
- `karume-birefnet-hr` (republished with this release) carries only the `f16` quant, which is also
  the default. The upstream BiRefNet_HR checkpoint is stored in f16, so the former `f32` seat held
  the same values at nearly twice the resident weight memory (919 MiB against 509 MiB at 1024²).
  `quant: "f32"` is rejected as an unknown quant; omit `quant` or pass `"f16"`. The matte logits
  of the `f16` seat are identical to the former `f32` seat's on the reference images.
  `karume-lucida` keeps `f32` (the default) and gains `f16`.
- `@karume/runtime`: `OpenedContainer.extractGraph()` now extracts only from a container opened from
  a `krm`. On a container opened from a `krg` it throws `ContainerFormatError`: a `krg` has no
  document that carries the hash of its whole constant region, so the copy cannot be verified.
  Extraction also checks, on every input, that the constant region read back has the declared
  length. Extraction from a `krm` is unchanged.
- `@karume/runtime`: the memory estimate (`estimateGraphMemory` / `PreparedModel.estimate`) now
  throws in three cases where it used to return numbers. (1) Without `maxStorageBufferBindingSize`, it
  counts the fused `attention` op only when its row-block count is the same on every device — the
  score matrix and the row statistics each fit one block under the WebGPU core default (128 MiB),
  or `attentionRowBlockMaxBytes` is at most that default — and otherwise rejects the graph, asking
  for the granted `maxStorageBufferBindingSize`. (2) A state graph whose fixed physical chunk row
  count is not in the set the generation context allows for the given `chunkLength` /
  `chunkBuckets`, including extra `generation.scenarios`, is rejected; every `run` of such a shape
  was already rejected. (3) A fused `attention` combination a session rejects (an additive mask or
  grouped query attention with `"a8"`, a head dimension or key length beyond the int32 accumulation
  bound of the int8 path, a non-f32 seat on the state form) is rejected by the estimate too; such a
  session was already rejected at construction. The estimates the Anima and Gemma pipelines use are
  not affected.
- `@karume/hub`: stricter input checks reject inputs that used to load. `loadManifest` with a
  `HubRepoRef` checks the `repo` before any request or cache cleanup: it must be exactly HF's
  `owner/name` (two segments of `[A-Za-z0-9._-]`, no leading dot), the same rule as
  cross-repository references. A value such as `"someone/anima?x=y"` used to load, with the query
  swallowing the rest of the URL; it now throws `HubError`. A ref without a string `repo` (from
  plain JS or an unchecked saved config) now also throws `HubError`, where it used to throw a bare
  `TypeError`. When an HF source resolves a branch or tag, the response must be a 40-character
  lowercase hex commit SHA: a non-conforming mirror or custom `fetch` that answers
  `{ sha: "main" }` used to load with `revisionSha: "main"` and re-resolve the branch on every
  later fetch; it now fails as a revision-resolution `HubFetchError` with the response in `cause`.
  On an HF load, a manifest whose cross-repository reference points at the loaded repository itself
  (at any revision) is rejected with `ManifestFormatError`; reference the file without `repo` /
  `revision` instead. `parseManifest` rejects a `pipeline` major that is not a safe integer (for
  example 400 digits, which used to parse as `Infinity`) with `ManifestFormatError`. The real
  Hugging Face Hub and the published distributions are not affected.
- `@karume/models`: a sampler spec (`SamplerSpec`, the `sampler` of Gemma 4 `generate` / `chat`,
  `Gemma4ChatSession.send` and the QAT pipeline) with a key other than `temperature`, `topK`,
  `topP`, `repetitionPenalty`, `logitBias` and `seed` now throws `ModelInputError`, naming the
  unknown key and listing the accepted ones. Unknown keys used to be ignored, so a saved JSON
  config with `minP` or a misspelled `top_p` silently sampled from the full distribution. Remove
  the extra keys.
- `@karume/models/anima`: `steps` must be an integer from 2 to 1,000. Larger values, which used to
  allocate the sigma ladder and run a loop that could not be cancelled, throw `ModelInputError`
  before GPU work; a manifest `pipelineConfig.defaults.steps` above 1,000 is rejected as an asset
  error (`Error` with the `ModelInputError` as `cause`). `generate` also rejects with `RangeError` a
  manifest `shift` whose f32 sigma ladder starts above 1 (55 of the 1,901 two-decimal values in
  [1, 20], e.g. 1.32), with either sampler: DPM++ 2M used to run every step on NaN, while Euler
  worked. The published `shift` of 3 starts exactly at 1 and is unaffected.
- `@karume/models/irodori`: a speaker reference must give at least two speaker tokens after
  patching: `IrodoriSpeakerInput.latent` needs `2 × speakerPatchSize` frames (8 on the published
  v4 / v4.1 small) and `audio` needs `(2 × speakerPatchSize − 1) × hopLength + 1` samples (13,441
  at 48 kHz). Shorter references throw `ModelInputError` at the start of `generate` /
  `generateLatent`. Latents of 4–7 frames and audio of 5,761–13,440 samples used to run with a
  one-token speaker sequence below the exported graph's minimum. Shape, row-count and sample-rate
  errors of `audio` / `latent` are now also raised there, before the text stages and the codec
  encoder run (same type and message).
- `@karume/models`: some input-caused failures that used to be a plain `Error` now throw
  `ModelInputError` (a subclass of `Error`; plain `catch` is unaffected): Gemma / EmbeddingGemma
  text with an unpaired surrogate (`encode`, and so chat rendering); an Irodori reference audio too
  short to give any speaker token, now raised before GPU work instead of after the codec encoder;
  and an SBV2 `noiseScaleW` that overflows f32, now `Sbv2InputError` (a `ModelInputError`) raised
  before the text encoder instead of after GPU execution (the noise is drawn earlier from the same
  stream, so the same seed gives the same WAV). Code that branches on `instanceof ModelInputError`
  or `err.name` classifies these differently.

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

[Unreleased]: https://github.com/hdae/karume/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/hdae/karume/compare/v0.13.0...v0.14.0
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

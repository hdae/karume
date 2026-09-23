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

- Karume container format (`krm` / `krg`, ADR 0108, stage 1): the runtime opens a container
  (`openContainer`), admits one of its graphs (`prepareContainer`) and builds a session from its
  blocks (`createSessionFromContainer`) with per-block sha256 verification; `codecLayout` maps a
  codec name to its decode path. The exporter gains `karume.container` (writer / reader, canonical
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
  and writes the `karume/5` manifest; `karume verify --container` checks a container from the CLI.
- Gemma 4 QAT family: `gemma4-qat` pipelines for E2B / E4B with fixed INT2 / INT4 storage, fixed
  static re-quantization (SRQ) whose rounding is preserved on both CPU and GPU, PLE read back
  whole or row by row, and a chat CLI example.
- Speculative decoding for Gemma 4 (MTP): a drafter session, the `speculative` option, a
  draft/verify/commit loop whose token stream matches non-speculative decoding exactly, and a
  self-financing gate that falls back to plain decode in contexts where speculation loses.
- Range reads for assets: `openAsset` and `AssetRangeReader` in hub, a directory adapter that
  reads at an offset, `parseSafetensorsHeader` / `safetensorsHeaderLength` on the runtime surface,
  and Hugging Face range reads through fetch-cache 0.8.0.
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

### Changed

- Decode and prefill are substantially faster: the GEMV family covers 1 ≤ M ≤ 64 as row-block
  variants, parallel GEMV (with optional subgroup reduction) is selectable from the quant seat,
  argmax over long rows runs in two phases, top-k uses a bounded heap and top-p is linear in the
  vocabulary, and RMS+residual, RoPE, attention row-statistics+PV and linear→SRQ can be fused.
- The distributed Gemma 4 E2B defaults to the `i4-fast` quant seat, which declares packed INT8
  activations (`packedStaticQuantize`) for the parallel GEMV path.
- Slot backing is held as an LRU set under a byte budget instead of a single slot, so changing
  shape no longer rebuilds it on every run.
- Gemma 4 prefill picks the smallest declared chunk bucket that covers the query length; the
  default bucket set is `[32, 64, 128, 256]`.
- Golden tolerances are two-tier: outputs are checked against Karume's own bound first and,
  where a WGSL accuracy bound is declared for that output, against the specification bound;
  exceeding only the first is recorded in `results.json` rather than failed.

### Fixed

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

### Breaking

- **Breaking:** the runtime's in-memory graph (`IrGraph`) now uses the merged storage vocabulary:
  `initializers[name]` is `{ storage: { codec, groupSize?, rowAxis? } }` or `{ shared: true }`,
  initializer names are the tensor keys (the exporter's FQN / `const.<hash>`), and the `tensor` /
  `storage.dtype` / `storage.scale` fields are gone. `openModel` / `extractIrGraph` return the
  legacy scale keys alongside the graph, `createShardValidator` takes them as a second argument,
  `ReadyInitializer` carries bytes instead of safetensors views, and capability diagnostics say
  `非対応 格納 '<layout>'`. Shared initializers are named after the lender's initializer.
- **Breaking:** the PLE read surface is a handle (`openPleShard`); decode reads the rows it needs
  instead of the whole shard.
- **Breaking:** hub reads manifest `karume/5` only (no `karume/4`); `resolveFiles` / `ResolvedFiles` /
  `WeightFiles` are replaced by `resolveSelection` / `ResolvedSelection` / `WeightContainer`, the
  `<weights>[i]` fetch-key convention and the 256 MiB shard limit are gone, and `fetchAssets` takes a
  plain `Record<string, FileRef>`. Published repositories keep working from the previously released
  packages; this main reads only repositories re-uploaded in the container format.
- **Breaking:** `from*Assets` take containers instead of safetensors — a component key maps to a
  single-form `krm` (`transformer`) or to its parts (`transformer[0]`, `transformer[1]`, …).
  `Gemma4Assets` loses `pleIndex` / `openPleShard` (the PLE lives in the `model` container) and the
  `Gemma4PleShardSource` / `Gemma4PleReadOptions` types are gone; the default PLE resident budget is
  now two blocks (about 64 MiB) instead of two shards.
- **Breaking:** container descriptors declare `assets[].length` (payload bytes; the block length is
  that rounded up to 4) — containers written by the stage-1 writer must be rewritten; `BlockSource`
  gains a required `verified` flag and `DescriptorExpectation.sha256` is a plain string.
- **Breaking:** the Gemma 4 product graph exits on the selected R rows as logits plus hidden
  state, and the default bucket set gains 4 and 8 — distributions must be re-exported.
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

[Unreleased]: https://github.com/hdae/karume/compare/v0.12.0...HEAD
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

# 0089: GPU memory fit becomes a deterministic comparison against absolute limits (memory management phase A)

- Status: accepted (2026-09-01); partly superseded by [0108](0108-container-format.md) (decision 4:
  the shard-size check became the part-length ceiling, effective 2026-09-24). The payload-only demand
  of decision 3 was re-graded to MAY on 2026-10-18
- Date: 2026-09-01
- Related: [0070](0070-shard-loading-admission.md) (decision 4 = reliance on errorScope; decision 5
  = never compare against free memory; decision 6 = the limits preflight this ADR implements) /
  [0066](0066-generation-context-state-slots.md) (addendum 5 = state binding-limit check) /
  [0038](0038-manifest-v1.md) (§7 addendum = the `requiredLimits` field) /
  [0081](0081-shard-spec-v2.md) (shard limit; later changed by [0090](0090-shard-spec-v3-tensor-pieces.md))

## Context

Allocation failures were detected only by the out-of-memory / validation errorScope per shard
(ADR 0070 decision 4). That net has three weaknesses:

1. **Implementation-dependent.** On an M2 the out-of-memory errorScope stays silent (the known-issues
   entry on the silent out-of-memory errorScope on Metal; wgpu-hal metal's `check_if_oom()` is a
   no-op). There a failed allocation becomes a no-op `writeBuffer` into an invalid buffer, and the run
   reads garbage.
2. **Late.** The failure shows up after gigabytes have been transferred.
3. **Coarse.** It names the failing shard at best, not which weight exceeded what by how many bytes.

Allocation sizes, however, are fixed by declarations alone: the residency plan
(`planWeightResidency`) is a pure function of the prepare phase, and state slots are fixed by their
bindings. The manifest's `requiredLimits` field was empty three times over: the exporter did not
write it (only the gemma4 recipe wrote it by hand), the hub only parsed it, and nothing read it. The
1 GiB shard limit (ADR 0081) was checked only on the writer side.

## Decision

**Everything that can be compared deterministically is promoted to an explicit check before
allocation or fetch.** The only comparison target is the device's **absolute limits**
(`maxBufferSize` / `maxStorageBufferBindingSize`). Free VRAM and the allocation total are not
compared (ADR 0070 decision 5 stands: WebGPU does not expose totals, and a guessed threshold
rejects healthy environments). The last line of detection for totals stays errorScope, which is
silent on Metal; the [limitations](../limitations.md) entry on the GPU memory preflight stopping at
the absolute device limits records this gap.

1. **Explicit check of the weight path** (runtime). The mapping from residency to allocated buffers
   has one implementation, `planWeightBuffers` (`weight-residency.ts`): eligible weights as payload,
   ineligible ones after f32 expansion, i8 / i4 with one more companion-scale buffer, sizes after
   `toSizeClass` alignment (the real `createBuffer` size). `assertWeightsWithinLimits` checks every
   entry against both limits at the entry of Session construction, before any byte is uploaded
   (today: before the per-part batch loop in `session-build.ts`). Every overrun is listed in **one**
   failure. The estimator (`estimate.ts`) consumes the same mapping, so check, estimate and real load
   cannot claim different sizes.
2. **State check completed** (runtime). The existing binding-limit check (ADR 0066 addendum 5) also
   checks `maxBufferSize` (`generation-context.ts`). `binding ≤ buffer` holds for a device the runtime
   plans itself; it is not guaranteed for a `GpuContext` passed in from outside.
3. **The writer of `requiredLimits` is derived in core** (exporter `karume/limits.py` and
   `bake_required_limits` in `dist.py`). Demand = the largest weight tensor payload and the largest
   state slot of the components the quant selects (capacity resolved from `pipelineConfig`; two or
   more symbols or a missing binding fail loudly). The demand MAY count resident sizes (payloads)
   only and leave out the f32-expanded size of weights that cannot stay compressed (re-graded
   2026-10-18 to MAY: the original reason, false rejection of devices that would run, does not hold
   — see the addendum of 2026-09-25; what remains is that Python has no mirror of the residency
   classification, and counting expanded sizes was left as a candidate for after the release, which
   0.14.0 satisfied). Only presets whose demand exceeds the WebGPU guaranteed defaults (256 MiB /
   128 MiB) get the field, so that "no field = runs on a default-spec device" keeps its meaning.
   Workgroup limits are not baked; they depend on kernel design. **Hand-written values on the
   planning side are rejected** as double bookkeeping (the gemma4 recipe's hand-written field was
   retired after it matched the derived value exactly).

   **Scope of the claim (ruling of 2026-09-05).** "No field = runs on a default-spec device" claims
   that **resident data (weights and state) fits the defaults**; it does not cover intermediate
   tensors (node outputs). Intermediates are not counted statically: after fusion their real demand
   depends on the granted limit (row-block attention does not materialise S at full width), so it
   cannot be read off the distribution, and baking the pre-fusion maximum would reject working
   devices before download. Example: BiRefNet 1024² carries no field, yet its largest binding at run
   time is 320 MiB (limitations, BiRefNet section, measured 2026-10-16). Intermediate overruns are
   caught by the plan-time preflight of ADR [0093](0093-transient-liveness-packing.md) decision 5
   (listed in full at Session construction); before that point the model is downloaded and then
   fails. Model cards state the resource needs.
4. **Reader-side check of the shard limit** (hub). At manifest parse each `size` in `shards` is
   checked (closed interval, same as the Python source `shards.py`). `assets` / `extras` are exempt:
   the limit is part of the shard-splitting contract, and a single oversized asset is legal.
   (Superseded by ADR 0108 decision 6 on 2026-09-24: `packages/hub/src/manifest.ts` applies a
   part-length ceiling of 1024 MiB, closed interval, to `container.parts` only; the Python source is
   `PART_MAX_BYTES` in `tools/exporter/src/karume/container.py`.)
5. **Next step (wave 2).** A reader of manifest `requiredLimits` (models admission compares it with
   the GPU limits and rejects before download, reading the adapter first) and the wiring of
   `estimateSessionMemory` into loading. This is the other half of ADR 0070 decision 6.

## Consequences

- Regenerating the distributions adds `requiredLimits`: all 30 anima quants (296.8 MiB), irodori f32
  (300 MiB, both limits) and f16 (150 MiB, binding only — an example of the band where only one limit
  is exceeded). gemma4 stays byte-identical (hand-written equals derived); sbv2 is within the
  defaults.
- No existing asset is newly rejected: a bind group always binds the whole buffer, so no weight above
  the binding limit ever ran.
- Remaining weaknesses of the same kind (filed only): `GpuContext.createResident` and run-time
  transients still rely on errorScope. The gemma4 PLE sidecar was shard-numbered but sat in `extras`,
  outside decision 4 (within the limit in practice).
  - Closed for run-time transients on 2026-09-05 by ADR [0093](0093-transient-liveness-packing.md)
    decision 5 (slot size > `maxStorageBufferBindingSize` and region > `maxBufferSize` are listed in
    full at plan time, before allocation). `createResident` remains.

## Addendum (2026-09-01) — wave 2: the models reader

- **Limits are obtained by reading the adapter and discarding it (option A).** The runtime adds
  `readAdapterLimits` (`requestAdapter` → `planRequiredLimits`, no caps, no device;
  `packages/runtime/src/gpu/acquire.ts`). `packages/models/src/session/gpu-features.ts` holds
  `assertRequiredLimitsSatisfied` (partial mapping, lists every shortfall) and
  `assertRequiredLimitsBeforeDownload` (`GpuContext.limits` with a shared GPU, the adapter's values
  otherwise). Call sites: the admission of each family given a shared GPU (which also guards
  `fromAssets`), and each family's `fromPretrained` before the weight prefetch. gemma4 checks only in
  `fromPretrained` (`gemma4ManifestConfig` returns the quant); its `fromAssets` has no manifest and is
  not covered (recorded in [limitations](../limitations.md), the entry on the pre-download GPU fit
  check).
  - 2026-09-05: `assertRequiredLimitsBeforeDownload` reads the adapter once without a shared GPU even
    when nothing is declared (the comparison is still a no-op without a declaration, so the meaning of
    "no field" is unchanged). This only detects a missing GPU early; it does not strengthen the check.
- Option B (acquire the device before download) and option C (carry the adapter around) were
  rejected. Both conflict with the ordering rule that every family parses its assets before it
  acquires a GPU (MUST). The WebGPU spec also lets an adapter expire at any time, and `requestDevice`
  on an expired adapter silently returns a device that is already lost; implementers advise
  requesting a fresh adapter right before the device. Option A's weakness (two `requestAdapter` calls
  need not return the same physical adapter) is absorbed by decision 1: the construction-time check
  stays the final check, and the pre-download check is a precheck.
- **Wiring `estimateSessionMemory` into loading (second half of decision 5) moves to phase B.** The
  estimate needs every symbolic input dimension bound (unbound fails loudly), and only gemma4 has the
  bound values at load time (capacity and chunkLength in its config). Wiring it now would bake an
  arbitrary representative shape per family. The limitations entry only gives the caller data to
  decide with; it does not reject, so the delay costs no safety.
- The five existing mirrors were regenerated with the values predicted above; shard sizes and sha256
  did not change (only `requiredLimits` differed in the manifests). Two unrelated drifts were noted
  (an sbv2 `symbols.json` metadata change, model-card wording).

## Addendum (2026-09-02) — phase B measurements; the estimate wiring stays deferred

- Phase B measured the host RAM peak as "constant + k × largest shard" (k ≈ 3 on Linux, 2.4 on Mac;
  [research/2026-09-02-shard-size-ram-peak.md](../research/2026-09-02-shard-size-ram-peak.md)). Phase
  C-1 (reusing the read buffer) followed and was retired on 2026-09-24 (ADR 0108 step 3d; ADR 0070
  addendum of 2026-09-24).
- **Wiring `estimateSessionMemory` into loading stays deferred.** The figure that matters for the
  decision is host-side (derivable from the manifest), not the GPU estimate. The GPU estimate remains
  a surface the caller uses after choosing the generation shape.

## Addendum (2026-09-24) — "the largest shard in the manifest" under the container format

With the container of ADR [0108](0108-container-format.md) the host-side figure depends on the source
type. Seek-type sources (local positional reads, a browser Blob) are bounded by the largest declared
block. Scan-type sources (Deno via HF) add buffers awaiting GC to the largest part, and how many of
those exist is not bounded by the declarations (0108 addendum 5, item 5). The counting rule lives in
[container-v1](../container-v1.md) §11. Tensor-level streaming (the old candidate C-2) reached every
path as block granularity at 0108 step 3e; it lowers host RAM only for seek-type sources. "Before the
shard loop" in decision 1 now reads "before the per-part batch loop" (where `assertWeightsWithinLimits`
is called); the meaning of the check is unchanged. Decision 4 now reads as the part-length ceiling
(see decision 4).

## Addendum (2026-09-25) — correction of the rationale of decision 3

Decision 3 limited the demand to resident sizes because "writing the f32-expanded worst case would
reject, before download, environments that would run". This does not hold in the current code.
`planWeightResidency(graph)` (`packages/runtime/src/runtime/weight-residency.ts`) is a pure function
of the graph and does not depend on the device. A weight that cannot stay resident in compressed form
(`expanded`) is expanded to f32 on every device and allocated as `numel × 4`. A distribution whose
expanded size exceeds the defaults therefore always fails the limit check in `createSession` on a
default-spec device; there is no "environment that would run".

The current semantics of the demand (code unchanged, ruling of 2026-09-25):

- The exporter's demand (`component_demand_bytes` in `tools/exporter/src/karume/dist.py`) is the
  maximum of the **payload lengths** in the merged supply plan (pieces summed, companion scales
  separately) and the largest state slot. Expanded sizes are not counted.
- "No field = runs on a default-spec device" therefore covers **resident presets (raw / f16 / i8 / i4
  / i2) and state** only. Overruns of expanded weights are caught by the check in `createSession`
  (decision 1).
- The shipped assets have no expanded weight (all initializers of the 110 distributed containers were
  classified in the review of 2026-09-24), so the gap is latent.

The docstring of `tools/exporter/src/karume/limits.py` already gives this reason (a limitation from
having no Python mirror of the residency classification, not an exclusion to avoid false rejection);
it still labels the rule MUST, while this ADR grades it MAY since 2026-10-18.

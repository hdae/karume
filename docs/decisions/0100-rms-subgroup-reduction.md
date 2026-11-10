# 0100: Opt-in 32-lane subgroup reduction for RMS normalization

- Status: accepted (2026-09-13; opt-in, not in a default preset); partly superseded by
  [0127](0127-node-dawn-toolchain.md) / [0129](0129-test-layout-and-lanes.md) (2026-10-16: Node +
  Dawn supports subgroups — decision 5's Deno / Chrome scope and the Deno SKIP) and by
  [0128](0128-default-preset-original-weights.md) decision 2 C1 (2026-10-17: now the reason it stays
  out of default presets); on 2026-10-18 decision 6 (the comparison page) was withdrawn;
  decision 2's rejection of missing device features replaced by [0147](0147-numerics-and-kernel-adoption.md) decision
  8 (2026-11-10): a warned switch to `workgroup`
- Date: 2026-09-13
- Related: [0017](0017-rms-norm-conv2d-clamp-min.md), [0058](0058-numerics-opt-in-contract.md),
  [0099](0099-rms-norm-add-fusion.md)

## Context and alternatives

Rejected on 2026-09-13: spreading the 256 partial sums over 128 / 64 threads (slower), vec2 / vec4
loads (width-dependent), the original tree via `subgroupShuffleXor` (little end-to-end gain). Scalar
`subgroupAdd` partial sums gained end to end on QAT E2B with no large quality loss in a small
evaluation. The summation order changes, so it does not replace the original key; it follows the
approved opt-in policy (ADR 0058). Numbers: [research](../research/2026-09-13-rms-subgroup-reduction.md).

## Decision

1. Add `AcquireGpuOptions.subgroups?: boolean`; omitted or false requests nothing new. True requires
   the features `subgroups` and `subgroup-size-control` and the WGSL language feature `subgroup_id`,
   and rejects their absence. It also runs a kernel that returns a known answer to all 256 lanes
   (32 lanes × 8 subgroups): the feature list is not proof that it works, and failures to create,
   run or read back, and device loss, raise. Acquiring it does not switch the inference summation.
2. Add `SessionOptions.rmsNormReduce?: "workgroup" | "subgroup32"` and the type `RmsNormReduce`,
   default `workgroup`. Invalid values are rejected before the weights are uploaded; without the
   subgroup features the Session warns once per device and runs `"workgroup"` (the reference value),
   shown in the pipeline key (amended 2026-11-10 by ADR [0147](0147-numerics-and-kernel-adoption.md)
   decision 8, replacing the rejection of missing device features). `subgroup32` applies to f32 RMS wider than 128 and to RMS → add pairs already
   eligible under `fuseRmsNormAdd: true`; widths up to 128 keep the existing kernel. The fusion
   conditions (shape, use count, public values, argument order) do not change.
3. The new kernel: 256 threads, `@subgroup_size(32)`, the same input scan per thread. The 8
   `subgroupAdd` partial sums go to shared memory; after a barrier each subgroup reads the same 8
   values and adds them. No mapping between subgroup ID and local ID is assumed; every lane takes
   part. The row grid-stride, the barrier after all reads of the shared sums, the integer
   rounding barrier and the argument order of RMS → add stay. The original RMS key and WGSL stay.
4. A separate path that accepts changes in low-order bits and generated sequences. Reference
   goldens, SHA values and tolerances do not change. The first rows for M = 1 / 4 / 8 are checked,
   but that alone does not accept whole speculative generation of a real model.
5. The regular and QAT Gemma pipelines accept `rmsNormReduce`; when they acquire the GPU they request
   the capability only for `subgroup32`, and a GPU passed in is checked at Session construction.
   Target and drafter get the same setting. The quant storage vocabulary, distributed weights,
   defaults and published revisions do not change. Deno 2.9.6 lacked the features, a shared default
   with this option would have kept users from starting, and M2 was unchecked, so the option was
   limited to explicit use on Chrome. (Superseded by ADR 0128 decision 2 C1 on 2026-10-17 together
   with the M2 measurement below: it stays out of default presets because a default preset runs on
   baseline WebGPU, and M2 gained 0.52%. Node + Dawn (ADR 0127) provides the features.)
6. Withdrawn 2026-10-18: the Chrome comparison page settings (not ported — ADR 0127 decision 5).

## Specification and acceptance

Follows [subgroupAdd](https://www.w3.org/TR/WGSL/#subgroupadd-builtin) and the `subgroup_size` /
`subgroup_id` conditions of the [WGSL specification](https://www.w3.org/TR/WGSL/). The measured fixed
32-lane design is adopted; an unmeasured variable-size design is not run in its place. No external
source code is copied.

- Snapshots: the existing reference WGSL, the three new WGSL texts, keys separate from fusion keys.
- Missing features, invalid values, a silent all-zero result, wrong lanes and device loss.
- CPU-reference tolerance, odd widths, 1 / 4 / 8 / 19 rows, forced grid-stride, both fusion
  argument orders and the later input's lifetime; real-model speed and quality runs (2026-09-13).
- Deno skipped the new feature and Chrome ran the same checks. (Superseded by ADR 0127 decision 1 /
  ADR 0129 on 2026-10-16: `packages/runtime/tests/gpu_rms_norm_subgroup_test.ts` skips by adapter
  features and runs on Node + Dawn.)
- Target models and scope are decided after the M2 check; not applied by default to unmeasured E4B,
  long contexts or speculative decoding.

## Addendum (2026-09-13) — after the M2 check

Not promoted for M2: decode 34.899 → 35.081 tok/s (about 0.52%, inside the repeat spread), same TTFT;
the option stays. Data: [M2 40 generations](../research/2026-09-13-gemv-subgroup.md).

## Addendum (2026-10-23) — `subgroups: "available"` (ADR 0132 decision 5)

`AcquireGpuOptions.subgroups` also takes `"available"`: the features are requested when the adapter
advertises `subgroups`, `subgroup-size-control` and the WGSL feature `subgroup_id`, and the device is
acquired without them otherwise (no error). The known-answer check of decision 1 still runs whenever
the features were requested and still rejects a failing device. The acquisition also measures whether
the device maps `local_invocation_index` linearly onto subgroups (`GpuContext.subgroupLanesLinear`);
the online attention's subgroup form depends on it and is chosen only where it holds. `true` keeps its
rejecting meaning; the three options of this ADR are unchanged.

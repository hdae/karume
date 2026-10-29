# 0005: Verification strategy (real-GPU tests as a first-class deliverable)

- Status: accepted (2026-08-01); partly superseded by [0127](0127-node-dawn-toolchain.md) decision 1
  (the real-GPU stage runs under Vitest on Node + Dawn, not `deno test`) and
  [0129](0129-test-layout-and-lanes.md) decision 2 (lanes come from one table), ruled 2026-10-18,
  in effect since both were accepted on 2026-10-29. The stages
  and rules below are in force (ADR 0129 decision 3 keeps them).
- Date: 2026-08-01
- Related: recon §1 / §8 / §9-2 — the prototype's largest gap was no automated numeric test of its
  WGSL kernels; Deno 2.9.4 + RTX 3080 Ti (WebGPU without flags, `shader-f16`, `timestamp-query`) was
  measured to run them.

## Decision

Verification has three stages plus acceptance, and **real-GPU numeric verification is part of the
automated suite** (`deno test` then — *superseded by ADR 0127 decision 1 on 2026-10-18: Vitest on
Node + Dawn*).

1. **WGSL snapshots** pin codegen determinism (byte-identical output).
2. **allclose against the CPU reference implementation**, always `|x−y| ≤ atol + rtol·|ref|`
   (relative error alone is forbidden); NaN / Inf fail. Needs no GPU.
3. **Real-GPU golden comparison** (Deno + real hardware — *superseded by ADR 0127 decision 1 on
   2026-10-18: Node + Dawn*), per op and per graph, against torch-derived goldens. A torch-derived
   golden per new op SHOULD be added; the required evidence for a new op is the TypeScript oracle
   (ADR [0129](0129-test-layout-and-lanes.md) decision 3) *(re-graded 2026-10-18 to SHOULD: ruling 5a)*.
4. **Real-browser acceptance lane**, run by hand; it MAY be automated *(re-graded 2026-10-18 to MAY:
   the browser is now the primary target and the `examples/` harness left the tree with ADR 0127
   decision 5; its new home is an ADR 0127 open point)*. Dawn / wgpu differences (rounding,
   features) are caught here.

### Rules (inherited from the prototype)

- An all-SKIP run is an explicit FAIL (no silent apparent success).
- The exporter's emitted set ⊆ the runtime's executable set is checked against a **contract table
  that includes dtypes** (op names alone missed the f32 `sum` incident).
- "It type-checks" ≠ "it computes the same numbers": every port and kernel change MUST pass stage 3.
- Without an adapter stage 3 shows as SKIP, but a release MUST have stage 3 green.

## Consequences

- Withdrawn 2026-10-18: the planned "backend × dtype path" CI matrix with software adapters
  (lavapipe) was never defined; CI runs the core lane without a GPU (`.github/workflows/ci.yml`).
- Added 2026-09-05: a release machine's adapter MUST list `shader-f16` and `timestamp-query`.
  Elsewhere `KARUME_ALLOW_NO_SHADER_F16=1` / `KARUME_ALLOW_NO_TIMESTAMP_QUERY=1` (and
  `KARUME_ALLOW_NO_GPU=1` without an adapter) state the intent; by default
  `packages/runtime/tests/gpu_gate_test.ts` fails the all-SKIP. Missing real-weight assets fail the
  same way in `packages/e2e/tests/e2e_assets_gate_test.ts` (`KARUME_ALLOW_NO_ASSETS=1`). A green run
  with any opt-out is not release evidence ([release-runbook](../release-runbook.md) §1).

## Addendum (2026-09-20) — test lanes (splitting the run into units)

The full run took over 30 minutes, mostly in per-family e2e tests (real weights, real GPU), so the
suite was split into lanes. **Only the unit of execution changed; the stages and rules did not.**

- **Granularity**: `core` (runtime, hub, shared layers, tools) and one lane per model family, named
  after the distribution spelling (list: `packages/test-support/src/lanes.ts`; run one with
  `vp test --project <lane>`).
- **The all-skip guards are in core only** (`gpu_gate`, the assets gate, `distribution_gate`): they
  require assets regardless of family, so a family lane would turn red on unrelated missing assets.
  `distribution_gate` derives the scope of its manifest check from every child directory of
  `models/` (ruling of 2026-10-11; a child without `karume.json` fails); its roster only detects a
  mirror that should exist and does not (`packages/runtime/tests/distribution_gate_test.ts`).
- **Coverage check**: the one danger of lanes is a test no lane runs, which passes silently.
  `packages/runtime/tests/verify_lanes_test.ts` checks core ∪ all lanes = every test file, core
  disjoint from the family lanes, and every lane non-empty. It read the `deno.json` task strings and
  threw on an unexpected form *(superseded by ADR 0129 decision 2 on 2026-10-18: it reads the lane
  table; the three conditions are unchanged)*.
- **Family lanes may overlap** (`e2e_gemma4_ple_gpu_test.ts` is in both gemma4 lanes): a gap is not
  allowed, an overlap only runs twice.
- **A release needs one green full run** (`vp test`, release-runbook §1): only the full run covers
  interaction within one process and run order. A union of lane runs does not replace it (ADR
  0129 addendum of 2026-10-29, open point 3).

## Addendum (2026-09-20) — reference guard (an all-SKIP of sha256 rows fails)

sha256 reference values are rows per environment key (ADR [0106](0106-device-keyed-references.md)),
so a machine can have no row yet; every sha256 regression check is then an explicit SKIP, and a
silent green would read "not verified" as "verified".

- **Reference guard** (`registerReferenceGate`, `packages/runtime/tests/helpers/reference.ts`; at the
  end of every e2e file with sha256 rows — 3 then, 13 on 2026-10-18): FAIL when the current
  environment has no row and the run is not writing rows (`KARUME_REFERENCE`). Since 2026-10-03,
  green means every registered case has a row here or its id is held (ADR 0106 addendum). Opt-out
  `KARUME_ALLOW_NO_REFERENCE=1`. Without a GPU and assets the sha256 checks and the guard do not run.
- **Relation to lanes**: the three core guards do not run in a family lane, but the reference guard
  lives in the family's e2e files, so `vp test --project models:<family>` always runs it.

## Addendum (2026-10-11) — upstream parity (the maintainer's policy)

- **Level**: agreement in the computed numbers, or in their intent, is enough. Upstream mistakes,
  bugs and precision choices made for speed are not copied (BiRefNet ports upstream's interpolation
  kernels and step order, not its truncation and half precision — ADR
  [0125](0125-birefnet-matte-postprocess.md)).
- **When upstream has a demonstrable problem** (quality loss, a bug, breakage in some environment),
  a workaround is offered as a public option and MAY become the default when better. Otherwise
  Karume follows upstream and adds no option (ADR [0008](0008-public-api.md)).
- **What follows upstream** is built to be compared with goldens from upstream's own functions
  where practical; otherwise correctness is shown another way (f64 reference with normalised-ratio
  bands — ADR [0118](0118-wan21-video-generation.md) decision 8).
- **Why**: exact agreement was a means to make an early, untrusted implementation testable; with
  speed weighing more now, 100 % agreement is worth little (done when possible).
- sha256 regression checks (ADR 0106) guard Karume's own output, separately from upstream parity,
  and MUST NOT become tolerances. In the suite, upstream-parity cases run only with
  `KARUME_UPSTREAM_PARITY=1` (ADR 0129 addendum 2026-10-17 (2)).

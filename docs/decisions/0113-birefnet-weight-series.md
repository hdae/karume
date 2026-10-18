# 0113: BiRefNet weight series — phase 1 (f16) layout, the rounding position and the quality check

- Status: accepted (2026-09-26, phase 1 = f16; phase 2 (i8) would extend this ADR and has not
  started); decision 7 and parts of decisions 1, 6 and 8 (marked inline) superseded by this ADR's
  addenda of 2026-09-27 and 2026-10-11 (presets and the default derived from the upstream storage
  dtype — the form [0128](0128-default-preset-original-weights.md) cites); the B570 held rows of
  decision 9 withdrawn 2026-10-18
- Date: 2026-09-26
- Related: [0006](0006-quantization.md) (fake-quant method) / [0018](0018-f16-weight-execution.md)
  (executing f16 storage) / [0027](0027-sbv2-f16-series.md) (the earlier f16 series; `--verify`
  exclusion of its decision 3) / [0029](0029-sbv2-i8-series-and-quant-quality.md) (series × storage
  dtype detection limits, dtype-set check) / [0050](0050-irodori-quant-series.md) (irodori phase 1 —
  the model for this ADR) / [0075](0075-quant-presentation.md) (preset presentation fields) /
  [0092](0092-distribution-repos-and-sources.md) (BiRefNet's 2 repositories × 2 resolutions). Input:
  deep dive E1 of the speed / memory plan (`.claude/reviews/2026-09-25_perf-recon/deep/E1-birefnet-series.md`,
  untracked).

## Context

The BiRefNet family (BiRefNet_HR and its fine-tune Lucida) shipped one f32 series; resident GPU weights
were 919 MiB at 1024² and 1,116 MiB at 2048². f32-only was not a kernel limit — direct conv2d and
implicit GEMM both have `w=f16` / `w=i8` (ADR 0018 / 0019); missing were the recipe's dtype axis and a
quality verdict. The investigation found three facts:

1. **The upstream BiRefNet_HR checkpoint is f16** (safetensors header: 687 F16 + 67 I64, 0 F32). The
   export read it with `dtype=torch.float32` and widened f16 values to f32. Lucida is 687 F32.
2. **Patch ⑦ (`BatchNorm2d` → per-channel `x·α + β`) does not fold BN into the conv weights.** α / β
   are f32 constants derived from BN statistics, inputs of mul / add (not weight slots, so not
   eligible), and are stored f32.
3. **Eligible weights (resident in compressed storage) are 89.2 % at 1024² and 73.5 % at 2048².**
   8 window masks, 16 zero-pad constants and 20 deform_conv2d weights (no weight slot) stay f32.

i8 cannot be copied from earlier families (the windowed-attention relative-position tables and TailConv
are not `nn.Linear` / `nn.Conv2d` modules and get no scale; deform's `regular_conv` is rounded yet stored
f32), so the maintainer approved (2026-09-26) **two phases**: phase 1 = f16 + the quality-check
harness; phase 2 = a BiRefNet-specific i8 plan + i8 series.

## Decision (phase 1)

1. **f16 is a separate series** (`outputs/series/<model>-<resolution>-f16/`, e.g.
   `birefnet-hr-1024-f16`); f32 series keep their unsuffixed names. Mixing them would silently apply
   the f32 series' net (per-series tolerances) to compressed assets (as in ADR 0018 / 0027).
   `--dtype {f32,f16}` (default f32) excludes `--verify` (the structural reason of ADR 0027 decision 3:
   `--verify` compares against the eager f32 weights before patching). The CLI accepts no i8.
   (Superseded by the addendum of 2026-10-11, item 2, on 2026-10-11: the writable dtypes and the default
   are per checkpoint — HR writes f16 only, default f16.)
2. **Rounding happens right after `load_model` and before `patch.apply`** (`birefnet.export.load_wrapper`).
   Rounding after apply would move ⑦'s α / β to f16 values while their storage stays f32: no VRAM saved,
   only numerics moved. Before apply, α / β derive in f32 from the rounded statistics. Rounding uses
   `karume.quantize.round_weights_to_f16` (all parameters and f32 buffers); emit's eligibility test
   stores only weight slots as f16.
3. **HR's f16 series is lossless against upstream** (rounding is the identity); its golden is
   **bit-identical** to the f32 series (inputs and outputs of the 8 cases at 1024², 2026-09-26). Lucida's
   f16 series is lossy.
4. **Quality-check harness (at export, torch CPU, no GPU)**: in the same process, before rounding, take
   the f32 outputs (a separately loaded f32 wrapper, patched graph), and write the differences to the
   rounded weights' outputs to `quality.json` in the series directory. Cases: 4 synthetic + 4 real
   images (real images run with or without `--real-images`). Metrics: logit max_abs / rel_rms, binary
   mask mismatch rate, mean absolute α difference after sigmoid, fraction of pixels whose 8-bit α moved
   and the largest move, bit-exactness.
5. **Quality checks (before publication — a failure removes the series)**:
   - If every float in the checkpoint is f16 (taken from the checkpoint header, not from the rounded
     result), the check fails unless every case is **bit-identical**. This catches a regression of the rounding position
     (rounding after apply). Bits are compared, not values (`torch.equal` treats `0.0` and `-0.0` as
     equal).
   - For a lossy checkpoint, a worst mean absolute α difference of **0.5 LSB or more (half a step of
     8-bit α = 0.5 / 255 ≈ 1.96e-3) fails** (`birefnet.export.ALPHA_MAE_LIMIT`). This is a provisional
     line from the kill criterion of deep dive E1 §6, raised from a record to a check by the review of
     2026-09-26 (A-2), and revisited with the phase 2 (i8) measurements; Lucida f16 (1024² 7.0e-5,
     2048² 4.8e-5) passes with about 30× margin. A difference of **0 in every case** also fails (the
     baseline and the compared side see the same weights — the harness has gone vacuous).
   - Compressed series also get a foreground-ratio order check on the rounded weights' 4 real-image
     outputs without `--real-images` (with it, the golden sanity check covers them).
6. **Distribution layout**: output path `matte/model.{f32,f16}.krm`, placement role `matte_{dtype}`.
   Required dtypes `matte_f32 → f32` / `matte_f16 → f16`; the forbidden list applies to the f32 preset
   only (the whole compressed vocabulary — a mixed-up f16 preset is caught by the required check).
   Checks (dimensions, provenance, `pipelineConfig` equality) apply to every dtype series.
   `karume dist --pipeline birefnet|lucida` needs both series (superseded by the addendum of
   2026-10-11, item 1, on 2026-10-11: the series follow the derived presets; only Lucida needs both).
   Every dtype series names the same upstream revision (superseded by the addendum of 2026-10-11,
   item 3, on 2026-10-11: every container is matched against the pinned revision). The reason stays:
   licence (MIT for both) and dimensions match across the two checkpoints, so without a revision check
   "HR's f32 series + Lucida's f16 series" would assemble as one model.
7. **Presets f32 / f16, default f32** — a default change alters the distribution's meaning and waits for
   the phase 2 quality measurements and a visual review (at least 4 real-image matte PNG pairs); HR's
   f16 is lossless, but a published default is not moved silently.
   (Superseded by the addenda of 2026-09-27 and 2026-10-11 (item 1) on 2026-10-11: the default is the
   preset of the upstream storage dtype, HR ships f16 only; released in 0.14.0 and listed by ADR 0128.)
8. **NOTICE and the card's storage description come from one place** (`birefnet.card.birefnet_storage_lines`),
   built from the checkpoint's storage dtype (a fact absent from the manifest —
   `BirefnetCheckpoint.stored_dtype`) and the shipped presets. `base_model_relation` is `quantized`
   only when the default quant's file rounds the checkpoint values (with defaults equal to the storage
   dtype, neither repository sets it). pytest checks `stored_dtype` against the header where real
   weights exist. The f16 preset's storage: linear / convolution / embedding (relative-position tables)
   weights f16; deform sampling weights (`regular_conv`), bias and norm f32; BatchNorm α / β f32,
   derived from statistics. The card's total GPU allocation (about 1.6 GiB at 1024², 4.0 GiB at 2048²;
   corrected 2026-10-11 from 1.7 / 4.1, which divided MiB by 1,000) is the f32 preset's measurement and
   the card says so. (Superseded by the addendum of 2026-10-11, item 4, on 2026-10-11: HR has no f32
   preset and resources are measured per checkpoint; "name only measured presets" stays.)
9. **TS checks**: `packages/e2e/tests/e2e_birefnet_test.ts` covers the f16 series (HR uses the f32
   series' tolerance — the goldens are bit-identical; Lucida f16 provisionally takes the f32 series'
   value and is derived from measurement), plus a GPU-free check of the container's **compressed storage
   dtype set** against the series' declaration (ADR 0029 decision 2: f32 = empty, f16 = `["f16"]`). The
   real-image e2e (`packages/e2e/tests/e2e_birefnet_real_test.ts`) covers the two 1024² f16 series
   (reference values written on a real GPU). The Lucida f16 2048² value is still provisional
   (`LUCIDA_F16_2048_SYNTHETIC_TOLERANCE`, `e2e_birefnet_test.ts` lines 271–277); the B570 that blocked
   the measurement is retired and the RTX 3080 Ti runs 2048², so it can be derived now.
   Withdrawn 2026-10-18: holding the 2048² f16 series on the B570 (`HELD_SERIES`, key
   `deno-intel-graphics-bmg-g21`) — that key matches no Node lane run; a Node row is made only after a
   B570 is observed under Node + Dawn.

## Measurements (2026-09-26, torch CPU export)

Resident weights 919.2 → 509.2 MiB (−44.6 %) at 1024² and 1,116.2 → 706.2 MiB (−36.7 %) at 2048²; HR f16
bit-exact; Lucida f16 α MAE 7.0e-5 / 4.8e-5, 8-bit α at most one step. GPU outputs are checked with
`vp test --project models:birefnet`. Containers shrank later (2026-10-16 addendum).

## Consequences

- At 1024² resident weights drop by 410 MiB and the download by about 45 %. At 2048² intermediate
  tensors dominate, so the effect is smaller (weights −36.7 %).
- Phase 2 (i8) plan: one BiRefNet-specific i8 plan function returns "what to round (include)" and
  "explicit f32" **from the same traversal** (drop `regular_conv`; relative-position tables (24) and
  TailConv (2) as f32; offset / modulator f32 at first). The harness is reused and sets the thresholds.
  **Three unknowns**: ① conv2d `w=i8` has never shipped in a real model; ② quality with offset /
  modulator in i8 (sampling positions move); ③ the effect on Lucida's saturated logits (up to 1078).
- Publication (re-uploading `karume-birefnet-hr` / `karume-lucida`, the `@karume/models` pin) happens
  at release (release-runbook).

## Addendum (2026-09-26 / 2026-09-27) — HR ships the f16 preset only; Lucida f32 default + f16

The maintainer's rulings (2026-09-26 "if it is lossless, f16 please" → 2026-09-27 "if upstream has no
f32, no f32 option is needed"): **BiRefNet_HR ships only the `f16` preset.** Upstream
`ZhengPeng7/BiRefNet_HR` has a single `model.safetensors` (444,473,596 B; 687 F16 + 67 I64, 0 F32 —
checked against the HF listing and the local header on 2026-09-27), so no f32 checkpoint exists. The
f32 series only widened f16 values (golden bit-identical to f16) and doubled residency (919 vs 509 MiB at
1024²). Reading f16 storage uses core WGSL `unpack2x16float` (ADR 0018) and needs no optional feature,
so no device is lost. **Lucida: `f32` default + optional `f16`** (upstream f32; f16 is lossy — α MAE up
to 7.0e-5 / 4.8e-5, 8-bit α at most one step, binary-mask mismatch up to 3.1e-5: small but not 0). The
recipe change (dist builds HR from one f16 preset, default preset, `base_model_relation`, pytest) and
the retirement of HR's f32 series, goldens, sha256 reference values and `HELD_SERIES` f32 rows go with
it (the f16 goldens hold the same values, so coverage does not shrink). Decision 6's "both series
needed" stays for Lucida only. The phase 2 (i8) verdict follows its quality measurements.

## Addendum (2026-10-11) — HR's f32 preset retired in the recipe and tests

The previous addendum's ruling enters the recipe and tests before publication (the HF re-upload and the
`BIREFNET_SOURCES` pin stayed with the next release; done in 0.14.0).

1. **Preset derivation**: presets come per checkpoint from the upstream storage dtype
   (`BirefnetCheckpoint.stored_dtype`) — **no preset wider than upstream storage, and the default is the
   preset of the storage dtype** (`birefnet_seat_dtypes` / `birefnet_default_quant` in
   `birefnet.distribution`). HR (f16) → `f16` only, default `f16`; Lucida (f32) → `f32` (default) /
   `f16`. Series acquisition, placement, weights, the quant table, the default, NOTICE and the
   `pipelineConfig` baseline all read this set; nothing is tabulated by hand. HR's placement requires
   only the f16 series and ignores a local f32 series (a plan given a series of a dtype outside the set
   fails). Lucida still needs both (decision 6). Presentation fields are looked up by (preset, upstream
   storage); HR's f16 preset does not claim a comparison with an absent f32 preset (`Upstream precision
   (f16)`). Card and NOTICE carry no text for presets wider than upstream storage; with one preset the
   heading is singular.
2. **Export follows the same rule**: the `--model-dir` checkpoint is identified by the revision of the
   acquisition record (when it matches a pinned value), then by the directory name (the same link as the
   series name); if both point to different checkpoints it fails (`birefnet.export.model_dir_checkpoint`).
   The same function decides writable dtypes and the default: HR defaults to f16 and `--dtype f32` is
   rejected by argparse with a reason; Lucida defaults to f32 and allows f16. A `--model-dir` absent
   from the attribution table gets the whole vocabulary and default f32 (`dist` does not build it).
   Series names do not change; `--verify` writes no series and is unchanged.
3. **Mix-ups are caught by pinned revisions**: each entry of `BIREFNET_CHECKPOINTS` (`birefnet.card`)
   pins its upstream revision (HR `a7a562f6fd16021180f2f4348f4de003a2d3d1e1`, Lucida
   `6cbedc9722652dc9a3df91dd871f0c4f3334e922` — from the `hf download` records of `inputs/birefnet`), and
   `dist` matches every container's `provenance.upstreamRevision` against it (`assert_upstream_provenance`,
   as for Wan). The "same revision across dtype series" check of decision 6 is removed: with one HR
   preset it compares nothing and cannot reject "a Lucida series in the HR repository"; the pinned value
   covers one or two presets and includes the cross-preset equality. pytest checks the pinned values
   against local acquisition records where they exist.
4. **Card resources are per checkpoint** (`BirefnetCheckpoint.resources` — the measured preset,
   conditions and figures per resolution). "No figures for unmeasured presets" holds by not rendering a
   card whose measured preset is absent from the manifest. Total allocation = the runtime estimate's
   `peakAccountedBytes` (weights + intermediates + inputs + output readback). The 2026-10-11 figures (HR
   f16, Lucida f32 re-measured, RTX 3080 Ti) were replaced on 2026-10-16 (next addendum).
5. Retired 2026-10-11: test rows reading HR's f32 series, HR's f32 tolerance constants (moved to f16),
   the `birefnet-hr-2048` `HELD_SERIES` row and the four `birefnet-hr-1024/*` sha256 reference values.
6. HR's f16 outputs equal the f32 series' on real GPUs as well (matte-logit sha256 of 4 real-image
   cases identical on the RTX 3080 Ti and the Arc B570, 2026-10-11).
7. Lucida's two f32 series were re-exported with the upstream revision (2026-10-11). **Goldens depend on
   the CPU of the machine that exported them** (torch 2.13.0+cpu; the same machine reproduces them bit
   for bit, the 2026-10-06 machine change moved them by f32 rounding); the GPU checks passed with the
   existing tolerances.

## Addendum (2026-10-16) — window masks as table + index, zero pads as `expand` (the root cause of xet fragmentation)

2048² parts were fragmented in HF xet reconstruction at the first publication (2026-09-05) and the 0.13.0
re-upload ([backlog](../backlog.md), [research 2026-08-09](../research/2026-08-09-xet-fragmentation.md),
2026-09-05 addendum). The cause was the container bytes, not the upload procedure.

1. **Cause = long zero runs in baked constants.** Patch ④ baked the shifted-window attention masks as
   whole `[nW, 144, 144]` f32 tensors (8 masks; 65 MiB at 1024², 248 MiB at 2048²; values {0, −100},
   mostly 0); constant folding baked patch ③'s zero pads into all-zero initializers (28 MiB at 2048²).
   xet's content-defined chunking cuts long zero runs into identical all-zero chunks, and reconstruction
   splits into terms pointing at the same range again and again.
2. Confirmed with a local CAS simulation using the same hf_xet 1.6.0 (old part 00003: 265 terms, 245
   repeated references; new: 4 terms). Details in `outputs/diag/birefnet-xet-2026-10-16/` (untracked).
3. **Fix** (`birefnet.patch` ③ and ④): bake only a table of window kinds (interior / right column /
   bottom row / bottom-right corner, at most 4: `[K ≤ 4, 144·144]`) and per-window indices (i32
   constants), and build the mask in the graph with `embedding` → `view`. **Export fails if the rebuilt
   mask is not bit-identical to the original.** Zero pads expand a one-element zero with `expand` (a
   constant-folding frontier, so only one element is baked). The mask table is an embedding table, so
   the f16 preset stores it in f16 (values {0, −100} are exact in f16).
4. Containers shrank (f16 series 534.7 / 741.3 MB → 452.9 MB; Lucida f32 964.6 / 1,171.2 MB →
   884.2 MB); GPU outputs did not change (`models:birefnet` green with unchanged sha256 reference values).
5. From part 00003 on, the 1024² and 2048² containers are byte-identical (weights, mask table included);
   only part 00001 (graph and descriptors) and part 00002 (graph constants, ≈ 0.1 MiB) differ. The
   mirror keeps the shared parts once under `shared/` (`karume-birefnet-hr` 9 → 6 files, 1,216.9 →
   432.8 MiB; `karume-lucida` 22 → 14 files, 3,253.8 → 1,276.8 MiB).
6. The masks moved from resident weights to runtime intermediates (largest `[1849, 144, 144]` f32 ≈
   153 MB at 2048²). Card figures re-measured on the RTX 3080 Ti (warm-up + 5 runs): total allocation
   HR f16 1,198 / 3,443 MiB and Lucida f32 1,609 / 3,854 MiB at 1024² / 2048² (values in
   `BIREFNET_CHECKPOINTS`, `birefnet/card.py`; raw data `outputs/misc/birefnet-resources-2026-10-16/`,
   untracked). Lucida was measured alone on a cooled GPU; measured right after HR, its 1024² runs slowed
   to 1.69–2.01 s.

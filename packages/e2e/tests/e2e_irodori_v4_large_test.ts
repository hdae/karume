/**
 * Real-GPU checks of the **v4-Large** distribution `models/karume-irodori-v4-large/` (ADR 0151) —
 * upstream's 3.3 B-parameter rectified-flow model with the T5Gemma 2 text backbone (a sliding
 * window that cuts captions beyond 256 tokens, and an end-of-image row selection), the Gemma
 * tokenizer, and no `f32` preset. It follows the structure of the MeanFlow checks
 * (`e2e_irodori_meanflow_test.ts`), for what this distribution makes new:
 *
 * - **per-graph parity** (`e2e_irodori_test.ts`'s form) for the seven graphs whose bytes are this
 *   model's own (`backbone`, `text-proj`, `caption-proj`, `speaker`, `duration`, `dit-context`,
 *   `dit`), on the mirror's **f16** graphs against the torch goldens of
 *   `goldens/irodori-v4-large-f16/<target>/` (written by `tools/export-recipes/irodori/export.py`).
 *   There is no `f32` preset, and the f16 goldens are baked from the f16-rounded weights (with the
 *   five non-persistent backbone buffers kept at f32 — ADR 0151 decision 6), so this compares the
 *   implementations only. The backbone cases include the three this model makes new:
 *   `caption-window` (T 388, beyond the window), `caption-long` (T 512, at the bound) and
 *   `caption-eoi` (the end-of-image id — decision 7). The codec pair is the published v4-small
 *   containers this distribution borrows, checked by v4-small's own suites;
 * - **latent parity** (`e2e_irodori_latent_test.ts`'s form) per storage series (`f16` / `i8` /
 *   `i8+dit4`) against the full-loop goldens `goldens/irodori-v4-large*\/pipeline/` (written by
 *   `irodori/pipeline_ref.py` only after its comparison with upstream's `sample_euler_rf_cfg`), over
 *   the three cases `full`, `no-ref` and `caption-window` (a 459-token caption the backbone's window
 *   cuts): S and the forward count exactly, z within a tolerance derived per series from
 *   measurement;
 * - the **premise** that the distribution's declaration and the golden's generation conditions agree
 *   (the RF sampler and its CFG knobs, the caps, the t schedule against upstream's `linspace`, and
 *   that the golden names a case the window cuts — `windowCutCases`, decision 1);
 * - **determinism** and **path identity**: two generations with the same seed are byte-identical,
 *   and the host path (`onEvent`) gives the same bytes as the resident path, with one `denoise-step`
 *   event per Euler step (the MUST of `runDitLoopResident`, on the RF loop with CFG).
 *
 * The parity cases are upstream parity, so they are opt-in (`KARUME_UPSTREAM_PARITY=1`); the
 * premise, determinism, path and golden-layout cases run by default. The WAV sha256 rows are
 * `e2e_irodori_v4_large_wav_test.ts`.
 *
 * MUST: the mirror and the goldens are local assets of a real-GPU machine. An environment without
 * the mirror (or the v4-small mirror it borrows the codec from), without the goldens, or without a
 * GPU SKIPs explicitly with the missing path (ADR 0005); a golden absent from a complete fetch of
 * the pinned test assets FAILs (ADR 0139).
 */

import { isNotFound, readDirSync, readFile } from "@karume/test-support/fs";
import { goldenDir, testAssetsFetched } from "@karume/test-support/test-assets";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  acquireGpu,
  parseSafetensors,
  prepareContainer,
  type SafetensorsFile,
  type Tensor,
} from "../../runtime/mod.ts";
import {
  compareTensors,
  formatAllclose,
  type Tolerance,
} from "../../runtime/src/reference/allclose.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { ioTensor } from "../../runtime/tests/helpers/golden-io.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { Pipeline } from "../../models/irodori.ts";
import { parseIrodoriPipelineConfig } from "../../models/src/irodori/config.ts";
import { tSchedule } from "../../models/src/irodori/host/sampler.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  mirrorDirectory,
  mirrorHasModel,
  mirrorPresent,
  mirrorRoot,
  openSelectedContainer,
  selectMirrorComponent,
} from "../../models/tests/helpers/mirror.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import {
  type GoldenCase,
  goldenPresent,
  pipelineGoldenDir,
  readCase,
  worstDifference,
} from "./helpers/irodori-assets.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/** The v4-Large distribution (it borrows its codec pair from the v4-small mirror). */
const MIRROR = mirrorRoot("karume-irodori-v4-large");
const BORROWED_MIRROR = mirrorRoot("karume-irodori-v4-small");
const MODEL = "v4-large";
const MIRROR_READY = mirrorHasModel(MIRROR, MODEL);
const FETCHED = testAssetsFetched();
const FETCH_HINT =
  "fetch: node tools/test-assets/fetch.ts (goldens/irodori-v4-large*/ — until they are in the " +
  "pinned dataset, stage them from outputs/series/ as tools/test-assets/README.md describes)";

if (!MIRROR_READY) {
  const missing = [
    `the mirror ${MIRROR.pathname} with model '${MODEL}'`,
    ...(mirrorPresent(BORROWED_MIRROR)
      ? []
      : [`the mirror it borrows the codec from, ${BORROWED_MIRROR.pathname}`]),
  ];
  console.warn(
    `[karume] Irodori v4-Large e2e SKIPs: missing ${missing.join(" / ")} — ` +
      "cd tools/export-recipes && uv run python dist.py --pipeline irodori-large --resolve-refs hf",
  );
}

const manifestText = MIRROR_READY
  ? await readTextIfPresent(new URL("karume.json", MIRROR))
  : undefined;

const lostAsset = (paths: readonly string[]): Error =>
  new Error(
    `${paths.join(" / ")} absent, but outputs/test-assets/ holds a complete fetch of the pinned ` +
      "revision: the test-asset dataset at the pin lacks it",
  );

// --- per-graph parity (the seven graphs of the model's own series) -------------------------

/*
 * The tolerances of the mirror's f16 graphs against the f16 series' torch CPU goldens, one per
 * graph output, each derived from this model's own measurement (2026-11-14, environment
 * `node-nvidia-geforce-rtx-3080-ti`; a plain comparison — the maxAbs / maxRel every case logs —
 * with \|ref\| read from the golden). A constant set back to `undefined` makes its cases log the
 * measurement and FAIL — they never turn green on a borrowed value.
 *
 * MUST: re-derive each from this model's own measurement, as 5–10 times the worst over every case
 * of the target, with the measurement table in the constant's doc (the form of
 * `e2e_irodori_test.ts`). Do not reuse v4-small's values: the widths (1152 / 1280 / 2048), the
 * depths (26 backbone layers, 24 DiT blocks), the ranges and the storage (f16, not f32) all differ
 * (ADR 0027).
 *
 * Every measured worst sits within 0.3–6.2 times its v4-small counterpart (v4-small's f32
 * graphs against f32 goldens), as an implementation-only difference should: the f16 goldens are
 * baked from the same f16-rounded weights the containers hold, so the storage rounding is not in
 * the comparison. As there, **atol drives** every target (each output straddles 0, so the smallest
 * nonzero \|ref\| reaches 1e-6–1e-8 and maxRel jumps there on elements whose absolute error is
 * tiny); rtol 1e-6 stays below atol at the top of every range.
 */

/**
 * `backbone` (T5Gemma 2 text encoder, 26 layers, `[1,T]` ids → `[1,T,1152]`).
 *
 * | Case           | T   | maxAbs  | maxRel  | \|ref\| max | \|ref\| min nonzero |
 * | -------------- | --- | ------- | ------- | ----------- | ------------------ |
 * | text-short     | 11  | 1.78e-5 | 1.02e-2 | 17.33       | 8.82e-5            |
 * | text-formal    | 19  | 2.43e-5 | 1.16e-2 | 17.25       | 1.40e-4            |
 * | text-emoji     | 24  | 3.34e-5 | 5.38e-2 | 17.23       | 2.40e-5            |
 * | caption-ja     | 30  | 2.94e-5 | 1.91e-2 | 17.25       | 4.25e-5            |
 * | caption-eoi    | 31  | 2.96e-5 | 5.68e-3 | 17.24       | 2.09e-4            |
 * | text-long      | 213 | 5.63e-5 | 4.85e-1 | 17.79       | 4.22e-6            |
 * | caption-window | 388 | 5.05e-5 | 9.46e-2 | 27.51       | 1.76e-5            |
 * | caption-long   | 512 | 5.75e-5 | 6.04e-1 | 27.62       | 2.21e-6            |
 *
 * atol 4e-4 is about 7.0 times the measured worst 5.75e-5 (caption-long); rtol 1e-6 contributes
 * 2.8e-5 (1/14 of atol) at the top \|ref\| = 27.62. Growing T 47-fold (11 → 512) grows maxAbs
 * 3.2-fold. The window cases (`caption-window`, `caption-long`) are no looser than `text-long`,
 * which the window does not cut — a window applied at the wrong width or on the wrong layers moves
 * the hidden state by O(1), 3–4 orders of magnitude above this threshold.
 */
const BACKBONE_TOLERANCE: Tolerance | undefined = { atol: 4e-4, rtol: 1e-6 };
/**
 * `text-proj` (`[1,T,1152]` → `[1,T,1280]`; the input is the torch backbone golden, so the
 * backbone's error does not enter).
 *
 * | Case           | T   | maxAbs  | maxRel  | \|ref\| max | \|ref\| min nonzero |
 * | -------------- | --- | ------- | ------- | ----------- | ------------------ |
 * | text-short     | 11  | 8.01e-5 | 1.98e-3 | 52.55       | 6.34e-4            |
 * | text-formal    | 19  | 4.96e-5 | 5.19e-3 | 52.56       | 3.29e-4            |
 * | text-emoji     | 24  | 6.10e-5 | 2.34e-2 | 52.08       | 1.58e-4            |
 * | caption-ja     | 30  | 5.72e-5 | 9.34e-3 | 58.07       | 1.86e-4            |
 * | caption-eoi    | 31  | 7.63e-5 | 5.82e-2 | 58.23       | 7.47e-5            |
 * | text-long      | 213 | 6.87e-5 | 9.32e-2 | 51.90       | 4.00e-5            |
 * | caption-window | 388 | 5.72e-5 | 1.07e-1 | 51.67       | 1.63e-5            |
 * | caption-long   | 512 | 5.34e-5 | 3.12e+0 | 51.71       | 7.60e-7            |
 *
 * atol 5e-4 is about 6.2 times the measured worst 8.01e-5 (text-short); rtol 1e-6 contributes
 * 5.8e-5 (1/8.6 of atol) at the top \|ref\| = 58.23. The range is about 4 times v4-small's (O(50)
 * against O(13)), and the absolute error grows with it (v4-small's worst 1.29e-5).
 */
const TEXT_PROJ_TOLERANCE: Tolerance | undefined = { atol: 5e-4, rtol: 1e-6 };
/**
 * `caption-proj` output 0 (the raw projector output, `[1,T,1280]`). Not `TEXT_PROJ_TOLERANCE`:
 * the weights and the range differ (\|ref\| max 108–140 against text-proj's 52–58).
 *
 * | Case           | T   | maxAbs  | maxRel  | \|ref\| max | \|ref\| min nonzero |
 * | -------------- | --- | ------- | ------- | ----------- | ------------------ |
 * | text-short     | 11  | 7.25e-5 | 5.89e-3 | 118.1       | 1.22e-3            |
 * | text-formal    | 19  | 5.72e-5 | 7.78e-3 | 116.8       | 1.58e-4            |
 * | text-emoji     | 24  | 7.63e-5 | 1.68e-2 | 111.8       | 1.71e-4            |
 * | caption-ja     | 30  | 1.18e-4 | 9.71e-3 | 140.3       | 3.44e-4            |
 * | caption-eoi    | 31  | 1.07e-4 | 1.43e-2 | 139.8       | 3.20e-4            |
 * | text-long      | 213 | 9.92e-5 | 4.60e-1 | 109.2       | 1.10e-5            |
 * | caption-window | 388 | 7.63e-5 | 8.79e-1 | 108.6       | 3.93e-6            |
 * | caption-long   | 512 | 8.39e-5 | 6.23e-1 | 108.5       | 3.64e-6            |
 *
 * atol 1e-3 is about 8.5 times the measured worst 1.18e-4 (caption-ja); rtol 1e-6 contributes
 * 1.4e-4 (1/7 of atol) at the top \|ref\| = 140.3. As on v4-small, the real short captions
 * (caption-ja / caption-eoi) widen the range and are the worst two.
 */
const CAPTION_PROJ_TOLERANCE: Tolerance | undefined = { atol: 1e-3, rtol: 1e-6 };
/**
 * `caption-proj` output 1 (`caption_norm` applied, `[1,T,1280]`). Not output 0's value: the norm
 * shrinks the range by 38–61 times (\|ref\| max 108–140 → 2.3–2.8), and the absolute error with
 * it.
 *
 * | Case           | T   | maxAbs  | maxRel  | \|ref\| max | \|ref\| min nonzero |
 * | -------------- | --- | ------- | ------- | ----------- | ------------------ |
 * | text-short     | 11  | 2.62e-6 | 5.89e-3 | 2.528       | 1.76e-4            |
 * | text-formal    | 19  | 2.74e-6 | 7.78e-3 | 2.355       | 5.79e-6            |
 * | text-emoji     | 24  | 2.62e-6 | 1.68e-2 | 2.796       | 1.55e-5            |
 * | caption-ja     | 30  | 3.10e-6 | 9.71e-3 | 2.306       | 1.80e-5            |
 * | caption-eoi    | 31  | 3.70e-6 | 1.43e-2 | 2.307       | 8.40e-6            |
 * | text-long      | 213 | 2.62e-6 | 4.60e-1 | 2.803       | 1.06e-6            |
 * | caption-window | 388 | 3.58e-6 | 8.79e-1 | 2.834       | 2.15e-7            |
 * | caption-long   | 512 | 3.93e-6 | 6.23e-1 | 2.838       | 3.79e-7            |
 *
 * atol 3e-5 is about 7.6 times the measured worst 3.93e-6 (caption-long); rtol 1e-6 contributes
 * 2.8e-6 (1/11 of atol) at the top \|ref\| = 2.838. maxRel matches output 0's to the digits in
 * every case (the error is carried over as a relative quantity, as on v4-small).
 */
const CAPTION_NORM_TOLERANCE: Tolerance | undefined = { atol: 3e-5, rtol: 1e-6 };
/**
 * `speaker` (`[1,S,128]` → `[1,S,1280]`; `ref-real-*` from the codec latent of real speech, the
 * others synthetic standard normal).
 *
 * | Case           | S   | maxAbs  | maxRel  | \|ref\| max | \|ref\| min nonzero |
 * | -------------- | --- | ------- | ------- | ----------- | ------------------ |
 * | ref-min        | 2   | 8.35e-6 | 2.49e-3 | 3.306       | 1.03e-3            |
 * | ref-1s         | 6   | 1.45e-5 | 1.01e-1 | 3.769       | 1.42e-5            |
 * | ref-real-short | 6   | 7.99e-6 | 2.03e-1 | 3.872       | 1.63e-5            |
 * | ref-5s         | 31  | 1.71e-5 | 1.64e-1 | 4.166       | 1.04e-5            |
 * | ref-real-full  | 47  | 1.55e-5 | 7.98e-2 | 4.529       | 8.75e-6            |
 * | ref-30s        | 187 | 3.16e-5 | 8.39e-1 | 4.028       | 3.43e-7            |
 * | ref-max        | 750 | 1.71e-5 | 4.11e-1 | 4.872       | 7.96e-8            |
 *
 * atol 2e-4 is about 6.3 times the measured worst 3.16e-5 (ref-30s); rtol 1e-6 contributes 4.9e-6
 * (1/41 of atol) at the top \|ref\| = 4.872. The real latents are not looser than the synthetic
 * ones (worst 1.55e-5 against 3.16e-5).
 */
const SPEAKER_TOLERANCE: Tolerance | undefined = { atol: 2e-4, rtol: 1e-6 };
/**
 * `duration` (log frames, `[1]`).
 *
 * | Case             | maxAbs  | maxRel  | \|ref\| |
 * | ---------------- | ------- | ------- | ------- |
 * | dur-both         | 4.77e-7 | 1.02e-7 | 4.690   |
 * | dur-caption-only | 4.77e-7 | 9.16e-8 | 5.207   |
 * | dur-neither      | 4.77e-7 | 8.88e-8 | 5.368   |
 * | dur-speaker-only | 0       | 0       | 5.342   |
 * | dur-long         | 4.77e-7 | 6.80e-8 | 7.013   |
 *
 * Every difference is 0 or **exactly 1 ulp** (f32, at values in [4, 8)). atol 3e-6 with rtol 1e-7
 * bounds the difference at 3.5e-6–3.7e-6 over these values, about 7.3–7.8 times the measured worst
 * 4.77e-7 (rtol is kept at 1e-7, not v4-small's 1e-6, which alone would contribute 7e-6 at
 * \|ref\| = 7 and put the bound beyond 10 times the measurement). In frames that is
 * `e^7 × 3.7e-6 ≈ 4e-3` — far from the host's rounding to integer frames; a mixed-up condition
 * vector moves it by O(0.1–1) (the 0.68 between `dur-both` and `dur-neither` above).
 */
const DURATION_TOLERANCE: Tolerance | undefined = { atol: 3e-6, rtol: 1e-7 };
/**
 * `dit` (one DiT step, 24 blocks — `v_pred[1,S,32]`; the condition K/V are torch inputs, so
 * `dit-context`'s error does not enter).
 *
 * | Case               | S   | maxAbs  | maxRel  | \|ref\| max | \|ref\| min nonzero |
 * | ------------------ | --- | ------- | ------- | ----------- | ------------------ |
 * | dit-cond-min       | 2   | 5.87e-6 | 4.88e-5 | 2.754       | 2.68e-2            |
 * | dit-cond-1s        | 25  | 7.63e-6 | 8.49e-4 | 3.293       | 4.59e-4            |
 * | dit-cond-late      | 25  | 7.27e-6 | 1.22e-2 | 3.066       | 4.94e-5            |
 * | dit-uncond-text    | 25  | 4.53e-6 | 5.03e-1 | 3.368       | 1.74e-6            |
 * | dit-uncond-speaker | 25  | 6.91e-6 | 7.39e-4 | 3.388       | 5.49e-4            |
 * | dit-uncond-caption | 25  | 5.60e-6 | 1.50e-2 | 3.257       | 1.54e-5            |
 * | dit-cond-max       | 750 | 9.78e-6 | 3.33e-2 | 4.400       | 7.42e-5            |
 *
 * atol 6e-5 is about 6.1 times the measured worst 9.78e-6 (dit-cond-max); rtol 1e-6 contributes
 * 4.4e-6 (1/14 of atol) at the top \|ref\| = 4.400. Twice v4-small's depth leaves the worst
 * within 1.2 times v4-small's (8.34e-6) — the condition side's 1519 tokens dominate the reduction
 * length, as there.
 */
const DIT_TOLERANCE: Tolerance | undefined = { atol: 6e-5, rtol: 1e-6 };
/**
 * `dit-context` (the 48 condition K/V `[1,1519,32,64]`; one value for all, as v4-small's — there
 * is one representative case, so per-output values would overfit its ranges).
 *
 * Measured on the one case `dit-cond-min` (the inputs: `text_state[1,256,1280]`,
 * `speaker_state[1,751,1280]`, `caption_state[1,512,1280]`):
 *
 * | Output       | maxAbs   | maxRel   | \|ref\| max | \|ref\| min nonzero |
 * | ------------ | -------- | -------- | ----------- | ------------------ |
 * | context_k_23 | 6.87e-5  | 3.84e+0  | 100.8       | 1.13e-8            |
 * | context_v_22 | 5.34e-5  | 1.84e-1  | 45.25       | 3.93e-6            |
 * | context_v_23 | 4.39e-5  | 1.26e+0  | 38.25       | 1.25e-6            |
 * | context_v_2  | 3.43e-5  | 1.01e+0  | 21.46       | 1.34e-7            |
 * | other 44     | ≤3.24e-5 | ≤2.78e+1 | 6.21–31.58  | ≥3.36e-8           |
 *
 * atol 5e-4 is about 7.3 times the measured worst 6.87e-5 (context_k_23, the widest range); rtol
 * 1e-6 contributes 1.0e-4 (1/5 of atol) at the top \|ref\| = 100.8. The largest maxRel
 * (context_k_15's 27.8) comes from an element near 0 (that output's smallest nonzero \|ref\| is
 * 3.36e-8) while that output's maxAbs is 9.30e-6. A swapped K / V, block order or condition order
 * moves an output by O(1–10).
 */
const DIT_CONTEXT_TOLERANCE: Tolerance | undefined = { atol: 5e-4, rtol: 1e-6 };

/** The output count of `dit-context` (24 blocks × K / V). */
const DIT_CONTEXT_OUTPUTS = 48;

/** The golden root of the per-graph goldens (the f16 series — there is no f32 preset). */
const GRAPH_GOLDENS = goldenDir("irodori-v4-large-f16");
/** The goldens are the f16 series, so the containers are the mirror's f16 graphs. */
const GRAPH_QUANT = "f16";

/** The cases of the text-side targets (`T5GEMMA2_GOLDEN_CASES` of `irodori/export.py`). */
const TEXT_CASES = [
  "caption-eoi",
  "caption-ja",
  "caption-long",
  "caption-window",
  "text-emoji",
  "text-formal",
  "text-long",
  "text-short",
] as const;

/**
 * Target directory → the manifest's weights key, the cases the export writes (sorted) and the
 * tolerance per output position. Fixed here, not taken from a listing — a listing alone makes
 * cases silently vanish where only part of the export ran; the golden-layout case below checks
 * the listing against this table.
 */
const TARGETS: Readonly<
  Record<
    string,
    {
      readonly graph: string;
      readonly cases: readonly string[];
      readonly tolerances: readonly (Tolerance | undefined)[];
    }
  >
> = {
  backbone: { graph: "backbone", cases: TEXT_CASES, tolerances: [BACKBONE_TOLERANCE] },
  "caption-proj": {
    graph: "caption_proj",
    cases: TEXT_CASES,
    tolerances: [CAPTION_PROJ_TOLERANCE, CAPTION_NORM_TOLERANCE],
  },
  dit: {
    graph: "dit",
    cases: [
      "dit-cond-1s",
      "dit-cond-late",
      "dit-cond-max",
      "dit-cond-min",
      "dit-uncond-caption",
      "dit-uncond-speaker",
      "dit-uncond-text",
    ],
    tolerances: [DIT_TOLERANCE],
  },
  "dit-context": {
    graph: "dit_context",
    cases: ["dit-cond-min"],
    tolerances: Array.from({ length: DIT_CONTEXT_OUTPUTS }, () => DIT_CONTEXT_TOLERANCE),
  },
  duration: {
    graph: "duration",
    cases: ["dur-both", "dur-caption-only", "dur-long", "dur-neither", "dur-speaker-only"],
    tolerances: [DURATION_TOLERANCE],
  },
  speaker: {
    graph: "speaker",
    cases: ["ref-1s", "ref-30s", "ref-5s", "ref-max", "ref-min", "ref-real-full", "ref-real-short"],
    tolerances: [SPEAKER_TOLERANCE],
  },
  "text-proj": { graph: "text_proj", cases: TEXT_CASES, tolerances: [TEXT_PROJ_TOLERANCE] },
};

/** The backbone cases this model makes new (beyond the window, at the bound, the end-of-image id). */
const BACKBONE_NEW_CASES = ["caption-window", "caption-long", "caption-eoi"] as const;

const IO_PREFIX = "io.";
const IO_SUFFIX = ".safetensors";

const ioUrl = (target: string, name: string): URL =>
  new URL(`${target}/${IO_PREFIX}${name}${IO_SUFFIX}`, GRAPH_GOLDENS);

const readIo = async (url: URL): Promise<SafetensorsFile> => {
  const bytes = await readFile(url);
  return parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};

/**
 * The `__metadata__` key of an io golden that does not carry some of its inputs
 * (`IO_INPUTS_FROM_KEY` of `irodori/export.py`). Its value is JSON `{ golden, tensors }`: the path
 * of another golden of this series (relative to the series root) and, per omitted input key, the
 * output key of that golden holding its bytes. This model's `dit` io names the `dit-context`
 * golden for the 48 condition K/V (they are the same bytes in all seven cases, 597 MB each); the
 * recipe checks the bytes at write time.
 */
const IO_INPUTS_FROM_KEY = "inputs_from";

/** A golden's reference, opened: omitted input key → the referenced golden's key, and that file. */
type OpenedReference = {
  readonly tensors: ReadonlyMap<string, string>;
  readonly url: URL;
  readonly file: SafetensorsFile;
};

/**
 * Open the golden an io references (`undefined` when it carries every input). A metadata value of
 * any other shape, or a path outside this series, FAILs.
 */
const openReference = async (
  io: SafetensorsFile,
  where: string,
): Promise<OpenedReference | undefined> => {
  const text = io.metadata.get(IO_INPUTS_FROM_KEY);
  if (text === undefined) return undefined;
  const value: unknown = JSON.parse(text);
  const malformed = new Error(
    `${where}: __metadata__.${IO_INPUTS_FROM_KEY} is not { golden, tensors }`,
  );
  if (typeof value !== "object" || value === null || !("golden" in value) || !("tensors" in value))
    throw malformed;
  const { golden, tensors } = value;
  if (typeof golden !== "string" || typeof tensors !== "object" || tensors === null)
    throw malformed;
  const map = new Map<string, string>();
  for (const [key, origin] of Object.entries(tensors)) {
    if (typeof origin !== "string") throw malformed;
    map.set(key, origin);
  }
  const url = new URL(golden, GRAPH_GOLDENS);
  assert(
    url.href.startsWith(GRAPH_GOLDENS.href),
    `${where} references ${url.pathname}, outside its series`,
  );
  return { tensors: map, url, file: await readIo(url) };
};

/**
 * The io goldens listed under one target directory (sorted). Only NotFound means absent — any
 * other failure propagates, so a broken root is reported as broken rather than skipped.
 */
const listCases = (target: string): readonly string[] => {
  try {
    return [...readDirSync(new URL(`${target}/`, GRAPH_GOLDENS))]
      .filter(
        (entry) =>
          entry.isFile && entry.name.startsWith(IO_PREFIX) && entry.name.endsWith(IO_SUFFIX),
      )
      .map((entry) => entry.name.slice(IO_PREFIX.length, entry.name.length - IO_SUFFIX.length))
      .sort();
  } catch (cause) {
    if (isNotFound(cause)) return [];
    throw cause;
  }
};

const ANY_GRAPH_GOLDEN = Object.keys(TARGETS).some((target) => listCases(target).length > 0);

// The golden layout runs where some per-graph golden is present (staged or fetched), and where the
// mirror is present beside a complete fetch (then none at all means the dataset at the pin lacks
// them). It needs no GPU and is not upstream parity.
test.skipIf(!ANY_GRAPH_GOLDEN && !(MIRROR_READY && FETCHED))(
  "Irodori v4-Large assets: the per-graph goldens hold exactly the expected targets and cases",
  () => {
    if (!ANY_GRAPH_GOLDEN) throw lostAsset([GRAPH_GOLDENS.pathname]);
    for (const name of BACKBONE_NEW_CASES) {
      assert(
        TARGETS.backbone.cases.includes(name),
        `the backbone case table lacks ${name} (ADR 0151 decisions 1 and 7)`,
      );
    }
    for (const [target, { cases }] of Object.entries(TARGETS)) {
      assertEquals(
        listCases(target),
        [...cases],
        `io goldens of ${GRAPH_GOLDENS.pathname}${target}/`,
      );
    }
  },
);

test.skipIf(!MIRROR_READY)(
  `Irodori v4-Large assets: the mirror declares every per-graph target at ${GRAPH_QUANT}`,
  async () => {
    const { models } = parseManifest(manifestText as string);
    assert(Object.hasOwn(models, MODEL), `the mirror declares no model '${MODEL}'`);
    const entry = models[MODEL];
    for (const [target, { graph }] of Object.entries(TARGETS)) {
      assert(
        Object.hasOwn(entry.weights, graph) && Object.hasOwn(entry.weights[graph], GRAPH_QUANT),
        `the mirror declares no '${graph}' at ${GRAPH_QUANT} (target ${target})`,
      );
    }
  },
);

/** `outputs/verify/<environment key>/<date>_irodori-v4-large-golden/` (safe to delete). */
const graphResults = openResults("irodori-v4-large-golden");

for (const [target, { graph, cases, tolerances }] of Object.entries(TARGETS)) {
  for (const name of cases) {
    const url = ioUrl(target, name);
    const present = goldenPresent(url);
    if (!present && !FETCHED) {
      console.warn(
        `[karume] Irodori v4-Large ${target}/${name} SKIPs: missing ${url.pathname} (${FETCH_HINT})`,
      );
    }
    test.skipIf(!GPU_AVAILABLE || !MIRROR_READY || (!present && !FETCHED) || !UPSTREAM_PARITY)(
      `Irodori v4-Large golden comparison: ${target} / ${name} (real GPU, ${GRAPH_QUANT} graphs / torch CPU expected values)${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        if (!present) throw lostAsset([url.pathname]);
        await runRecordedCase(
          graphResults,
          { id: `${target}/${name}` },
          async ({ measurements }) => {
            const opened = await openSelectedContainer(
              await selectMirrorComponent(MIRROR, graph, { model: MODEL, quant: GRAPH_QUANT }),
            );
            const parsed = prepareContainer(opened, graph);
            // The range differs per output (caption-proj's second output is normalised), so a
            // table with the wrong count would compare an output against another one's value.
            assertEquals(
              tolerances.length,
              parsed.graph.outputs.length,
              `${target}'s tolerance count differs from the IR output count`,
            );
            const io = await readIo(url);
            // An input the golden does not carry comes from the golden its metadata names (the
            // condition K/V of `dit`, from `dit-context`) — fed the same way as a carried one.
            const reference = await openReference(io, url.pathname);
            const omitted = [...(reference?.tensors.keys() ?? [])];
            const expectedKeys = [
              ...parsed.graph.inputs.map((spec) => `input.${spec.name}`),
              ...parsed.graph.outputs.map((_, index) => `output.${index}`),
            ].sort();
            // Every input and output exactly once: carried by the golden or named by its reference.
            for (const key of omitted) {
              assert(!io.tensors.has(key), `${url.pathname} both carries and references ${key}`);
            }
            assertEquals(
              [...io.tensors.keys(), ...omitted].sort(),
              expectedKeys,
              `tensor keys of ${url.pathname}`,
            );
            // The symbolic dimensions bind from the golden's input shapes: T up to 512 for the
            // text side (`caption-long` is the bound, `caption-window` beyond the window), S up to
            // 750 for `speaker` / `dit` (`ref-max` / `dit-cond-max`).
            const inputs: Record<string, Tensor> = {};
            for (const spec of parsed.graph.inputs) {
              const key = `input.${spec.name}`;
              const origin = reference?.tensors.get(key);
              const [file, stored, where] =
                reference !== undefined && origin !== undefined
                  ? [reference.file, origin, reference.url.pathname]
                  : [io, key, url.pathname];
              const view = file.tensors.get(stored);
              assert(view !== undefined, `${stored} (for ${key}) is not in ${where}`);
              inputs[spec.name] = ioTensor(file, view, spec.dtype);
            }
            const gpu = await acquireGpu();
            try {
              assertAdapterMatchesEnvironment(gpu);
              const session = await parsed.createContainerSession(gpu);
              try {
                const outputs = await session.run(inputs);
                assertEquals(Object.keys(outputs).sort(), [...parsed.graph.outputs].sort());
                const failures: string[] = [];
                const underived: string[] = [];
                parsed.graph.outputs.forEach((output, index) => {
                  const view = io.tensors.get(`output.${index}`);
                  assert(view !== undefined, `output.${index} is not in the golden`);
                  const declared = parsed.graph.values[output].dtype;
                  assertEquals(outputs[output].shape, view.shape, `${output}: shape`);
                  assertEquals(outputs[output].dtype, declared, `${output}: dtype`);
                  const tolerance = tolerances[index];
                  const report = compareTensors(
                    outputs[output],
                    ioTensor(io, view, declared),
                    tolerance ?? { atol: 0, rtol: 0 },
                  );
                  measurements.push({
                    output,
                    maxAbs: report.maxAbsError,
                    maxRel: report.maxRelError,
                    tolerance: tolerance ?? { atol: 0, rtol: 0 },
                    stage: "karume",
                  });
                  console.log(
                    `[e2e] irodori ${MODEL}/${GRAPH_QUANT} ${target}/${name} output.${index} ('${output}'): ` +
                      `maxAbs ${report.maxAbsError.toExponential(3)} / maxRel ${report.maxRelError.toExponential(3)}`,
                  );
                  if (tolerance === undefined) {
                    underived.push(`output.${index} maxAbs ${report.maxAbsError.toExponential(3)}`);
                  } else if (!report.pass) {
                    failures.push(`${output}: ${formatAllclose(report)}`);
                  }
                });
                if (underived.length > 0) {
                  throw new Error(
                    `${target}: the tolerance is not derived yet (${underived.join(" / ")}) — fill in ` +
                      "the target's tolerance constant from every case's log (do not reuse v4-small's)",
                  );
                }
                assert(failures.length === 0, `${target}/${name}: ${failures.join(" / ")}`);
              } finally {
                await session.dispose();
              }
            } finally {
              gpu.destroy();
            }
          },
        );
      },
    );
  }
}

// --- latent parity, premise, determinism, paths --------------------------------------------

/** The fields of a v4-Large golden `meta.json` this file reads (the exporter owns the shape). */
type V4LargeMeta = {
  readonly steps: number;
  readonly initScale: number;
  readonly cfgRange: readonly [number, number];
  readonly cfgScales: { readonly text: number; readonly speaker: number; readonly caption: number };
  readonly frameRate: number;
  readonly tSchedule: readonly number[];
  /** The exporter's max |closed form − upstream `linspace`| of the t schedule. */
  readonly tScheduleClosedFormMaxAbs: number;
  readonly tEmbedDim: number;
  readonly caps: { readonly text: number; readonly speaker: number; readonly caption: number };
  /** The cases transformers' sliding-window mask cuts (written for T5Gemma 2 only — ADR 0151). */
  readonly windowCutCases: readonly string[];
  readonly cases: Readonly<Record<string, GoldenCase>>;
};

/** One storage series (the quant preset × its full-loop golden series — one to one). */
type V4LargeSeries = {
  readonly name: string;
  readonly seriesRoot: string;
  /** The tolerance of `z` (derived per series from measurement — `undefined` = not derived yet). */
  readonly zAtol: number | undefined;
};

/**
 * The `z` tolerance of the **f16** series. Measured (a plain comparison — the `z maxAbs` this
 * check logs; 2026-11-14, `node-nvidia-geforce-rtx-3080-ti`):
 *
 * | Case           | S   | Elements | z maxAbs diff | \|z\| max | Ratio    |
 * | -------------- | --- | -------- | ------------- | --------- | -------- |
 * | full           | 157 | 5,024    | **8.2269e-4** | 5.592107  | 1/6,797  |
 * | no-ref         | 134 | 4,288    | 5.1826e-5     | 4.585699  | 1/88,483 |
 * | caption-window | 156 | 4,992    | 4.7643e-5     | 4.716795  | 1/99,003 |
 *
 * 5e-3 = **6.1 times** the measured worst 8.2269e-4 (full), about 1/1,118 of the \|z\| max. `full`
 * is the worst by an order of magnitude, as on v4-small's i8+dit4 series (the one case with a
 * reference and all three CFG branches, 4 forwards per step).
 */
const F16_Z_ATOL: number | undefined = 5e-3;

/**
 * The `z` tolerance of the **i8** series (its golden is baked from the shipped i8 bytes, so this
 * too is the implementation difference only). Measured (2026-11-14,
 * `node-nvidia-geforce-rtx-3080-ti`):
 *
 * | Case           | S   | Elements | z maxAbs diff | \|z\| max | Ratio     |
 * | -------------- | --- | -------- | ------------- | --------- | --------- |
 * | full           | 157 | 5,024    | **3.1525e-4** | 5.560662  | 1/17,639  |
 * | no-ref         | 135 | 4,320    | 3.6597e-5     | 4.58121   | 1/125,180 |
 * | caption-window | 155 | 4,960    | 2.6037e-4     | 4.921999  | 1/18,904  |
 *
 * 2e-3 = **6.3 times** the measured worst 3.1525e-4 (full), about 1/2,780 of the \|z\| max. S
 * differs from the f16 golden's in two cases (135 / 155 against 134 / 156) — the i8 weights move
 * the duration predictor's output, and each series' golden carries its own S, which matches
 * exactly here.
 */
const I8_Z_ATOL: number | undefined = 2e-3;

/**
 * The `z` tolerance of the **i8+dit4** series (series root `-i4`; `dit` alone is i4, the other
 * roles share the i8 bytes; the golden reads the shipped bytes back). Measured (2026-11-14,
 * `node-nvidia-geforce-rtx-3080-ti`):
 *
 * | Case           | S   | Elements | z maxAbs diff | \|z\| max | Ratio     |
 * | -------------- | --- | -------- | ------------- | --------- | --------- |
 * | full           | 157 | 5,024    | **1.2258e-3** | 5.504448  | 1/4,490   |
 * | no-ref         | 135 | 4,320    | 4.1008e-5     | 4.787208  | 1/116,738 |
 * | caption-window | 155 | 4,960    | 1.2532e-4     | 4.447559  | 1/35,490  |
 *
 * 8e-3 = **6.5 times** the measured worst 1.2258e-3 (full), about 1/688 of the \|z\| max. There
 * is no ordering between series (the same note as v4-small's i8+dit4).
 */
const I8_DIT4_Z_ATOL: number | undefined = 8e-3;

/**
 * The `z` tolerances per series. A case logs `S / forwards / z maxAbs` and FAILs while its series'
 * value is `undefined`.
 *
 * MUST: derived per series from the raw measurement (the pattern of ADR 0027), never reused from
 * v4-small, from MeanFlow or from another v4-Large series: to re-derive, set the constant back to
 * `undefined`, run with `KARUME_UPSTREAM_PARITY=1`, take 5–10 times the logged worst over the
 * three cases, and write the measurement table in its doc. The recipe measured the f16 latent
 * against the f32 golden at 43.9 / 59.5 / 56.0 dB (ADR 0151 Consequences) — a quantization figure,
 * not this check's (each series' golden is baked from its own rounded weights, so this is the
 * implementation difference only).
 *
 * NOTE: `i8-a8` shares the i8 bytes but quantizes activations, so it is not part of the
 * numeric-parity net (the note of `e2e_irodori_latent_test.ts`).
 */
const SERIES: readonly V4LargeSeries[] = [
  { name: "f16", seriesRoot: "irodori-v4-large-f16", zAtol: F16_Z_ATOL },
  { name: "i8", seriesRoot: "irodori-v4-large-i8", zAtol: I8_Z_ATOL },
  { name: "i8+dit4", seriesRoot: "irodori-v4-large-i4", zAtol: I8_DIT4_Z_ATOL },
];

/** The full-loop golden cases (`caption-window` is the one the backbone's window cuts). */
const LATENT_CASES = ["full", "no-ref", "caption-window"] as const;

/** The utterance length (seconds) of the determinism and path checks (`duration` does not run). */
const SHORT_SECONDS = 1;

const openPipeline = async (quant: string): Promise<Pipeline> =>
  await Pipeline.fromPretrained(mirrorDirectory(MIRROR), { model: MODEL, quant });

for (const series of SERIES) {
  const directory = pipelineGoldenDir(series.seriesRoot);
  const metaUrl = new URL("meta.json", directory);
  const metaText = await readTextIfPresent(metaUrl);
  const readMeta = (): V4LargeMeta => {
    if (metaText === undefined) throw lostAsset([metaUrl.pathname]);
    return JSON.parse(metaText) as V4LargeMeta;
  };
  const seated =
    manifestText !== undefined &&
    Object.hasOwn(parseManifest(manifestText).models[MODEL].quants, series.name);
  const available = seated && (metaText !== undefined || FETCHED);
  if (seated && metaText === undefined && !FETCHED) {
    console.warn(
      `[karume] Irodori v4-Large latent (${series.name}) SKIPs: missing ${metaUrl.pathname} (${FETCH_HINT})`,
    );
  }
  const runnable = GPU_AVAILABLE && seated;

  for (const caseName of LATENT_CASES) {
    test.skipIf(!(GPU_AVAILABLE && available && UPSTREAM_PARITY))(
      `e2e (real GPU): the v4-Large latent matches the full-loop golden (${series.name} / ${caseName})${UPSTREAM_PARITY_OPT_IN}`,
      async () => {
        const meta = readMeta();
        assert(
          Object.hasOwn(meta.cases, caseName),
          `${metaUrl.pathname} has no case '${caseName}'`,
        );
        const expected = meta.cases[caseName];
        const caseUrl = new URL(`case.${caseName}.safetensors`, directory);
        if (!goldenPresent(caseUrl)) {
          throw new Error(
            `inconsistent golden set: ${metaUrl.pathname} is present but ${caseUrl.pathname} is absent`,
          );
        }
        const golden = await readCase(directory, caseName);
        await using pipeline = await openPipeline(series.name);
        const started = performance.now();
        const latent = await pipeline.generateLatent({
          text: expected.text,
          ...(expected.caption === "" ? {} : { caption: expected.caption }),
          ...(expected.reference === null
            ? {}
            : { speaker: { latent: golden("reference_latent") } }),
          initialNoise: golden("noise"),
        });
        const z = golden("z");
        const { maxAbs, at } = worstDifference(latent.data, z);
        console.log(
          `[e2e] irodori ${MODEL}/${series.name} ${caseName}: ${((performance.now() - started) / 1000).toFixed(1)}s / ` +
            `S ${latent.frames} / forwards ${latent.forwards} / z maxAbs ${maxAbs.toExponential(4)} ` +
            `(atol ${series.zAtol ?? "not derived"} / |z| max ${expected.zAbsMax})`,
        );
        // The integer decisions (the formula deciding S, the t interval where CFG applies) must
        // match exactly whatever the series.
        assertEquals(latent.frames, expected.S, `${caseName}: S differs from the golden`);
        assertEquals(
          latent.forwards,
          expected.forwards,
          `${caseName}: dit's forward count differs`,
        );
        assertEquals(
          latent.latentDim,
          z.length / expected.S,
          `${caseName}: latentDim differs from the golden`,
        );
        if (series.zAtol === undefined) {
          throw new Error(
            `the z tolerance of ${series.name} is not derived yet (measured maxAbs ${maxAbs.toExponential(4)}) — ` +
              "fill in this series' zAtol from the log of every case (do not reuse another series' value)",
          );
        }
        if (maxAbs > series.zAtol) {
          throw new Error(
            `${caseName}: the latent differs from the golden by ${maxAbs.toExponential(4)} (tolerance ${series.zAtol}); ` +
              `worst at frame ${Math.floor(at / latent.latentDim)}: golden ${z[at]} → measured ${latent.data[at]} — ` +
              "check the CFG combination / t schedule / mask interval / the backbone's window before the tolerance",
          );
        }
      },
    );
  }

  test.skipIf(!available)(
    `e2e (assets): the v4-Large declaration matches the golden's generation conditions (${series.name})`,
    () => {
      // No GPU needed — the check's premise (the golden regeneration and the dist regeneration
      // have not drifted apart).
      const meta = readMeta();
      const config = parseIrodoriPipelineConfig(
        parseManifest(manifestText as string).models[MODEL].pipelineConfig,
      );
      if (config.flowParameterization !== "rf_velocity") {
        throw new Error(`${series.name}: the distribution does not declare the RF sampler`);
      }
      const drifted: string[] = [];
      const check = (key: string, expected: number, actual: number): void => {
        if (expected !== actual) drifted.push(`${key} golden ${expected} → distribution ${actual}`);
      };
      check("steps", meta.steps, config.steps);
      check("initScale", meta.initScale, config.initScale);
      check("cfgMinT", meta.cfgRange[0], config.cfgMinT);
      check("cfgMaxT", meta.cfgRange[1], config.cfgMaxT);
      check("cfgScales.text", meta.cfgScales.text, config.cfgScales.text);
      check("cfgScales.speaker", meta.cfgScales.speaker, config.cfgScales.speaker);
      check("cfgScales.caption", meta.cfgScales.caption, config.cfgScales.caption);
      check("frameRate", meta.frameRate, config.frameRate);
      check("timestepEmbedDim", meta.tEmbedDim, config.timestepEmbedDim);
      check("maxTextLen", meta.caps.text, config.maxTextLen);
      check("speakerRows", meta.caps.speaker, config.speakerRows);
      check("maxCaptionLen", meta.caps.caption, config.maxCaptionLen);
      if (drifted.length > 0) {
        throw new Error(
          `${series.name}: the distribution's pipelineConfig differs from the golden's generation conditions: ` +
            `${drifted.join(" / ")} (find out which side was re-baked before suspecting the z comparison)`,
        );
      }
      // The host's closed-form t schedule against upstream's f32 `linspace` values: no further
      // apart than the exporter measured for the same two sequences (the doc of `tSchedule`).
      const schedule = tSchedule(config.steps, config.initScale);
      assertEquals(meta.tSchedule.length, schedule.length, "t schedule length");
      const scheduleGap = Math.max(
        ...Array.from(schedule, (value, index) => Math.abs(value - meta.tSchedule[index])),
      );
      assert(
        scheduleGap <= meta.tScheduleClosedFormMaxAbs,
        `the host t schedule is ${scheduleGap} from the golden's, beyond the exporter's measured ` +
          `${meta.tScheduleClosedFormMaxAbs}`,
      );
      // The golden carries a case the backbone's window cuts, and this file runs it — otherwise
      // the window (ADR 0151 decision 1) would be outside the latent check.
      assert(meta.windowCutCases.length > 0, `${metaUrl.pathname}: windowCutCases is empty`);
      for (const name of meta.windowCutCases) {
        assert(
          Object.hasOwn(meta.cases, name),
          `windowCutCases names '${name}', not a golden case`,
        );
        assert(
          (LATENT_CASES as readonly string[]).includes(name),
          `windowCutCases names '${name}', which this file does not run`,
        );
      }
      assertEquals(
        Object.keys(meta.cases).sort(),
        [...LATENT_CASES].sort(),
        `the golden cases of ${metaUrl.pathname}`,
      );
    },
  );

  test.skipIf(!runnable)(
    `e2e (real GPU): v4-Large generations are deterministic and both loop paths give the same bytes (${series.name})`,
    async () => {
      await using pipeline = await openPipeline(series.name);
      const config = parseIrodoriPipelineConfig(
        parseManifest(manifestText as string).models[MODEL].pipelineConfig,
      );
      // Not passing `initialNoise` = the path through the internal `Randn`; S is fixed with
      // `durationSeconds` (no `duration` run). Text only: the text CFG runs inside its t interval.
      const request = {
        text: "本日はお越しいただき、誠にありがとうございます。",
        seed: 7,
        durationSeconds: SHORT_SECONDS,
      };
      const first = await pipeline.generateLatent(request);
      const second = await pipeline.generateLatent(request);
      const steps: number[] = [];
      const observed = await pipeline.generateLatent({
        ...request,
        onEvent: (event) => {
          if (event.kind === "denoise-step") steps.push(event.step);
        },
      });
      assertEquals(first.seed, 7);
      assertEquals(first.frames, SHORT_SECONDS * config.frameRate);
      assert(
        first.forwards > config.steps,
        `forwards ${first.forwards}: the text CFG added no unconditional forward over ${config.steps} steps`,
      );
      assertEquals(
        new Uint8Array(second.data.buffer),
        new Uint8Array(first.data.buffer),
        "the latent changed with the same seed and the same request",
      );
      assertEquals(
        steps,
        Array.from({ length: config.steps }, (_, index) => index + 1),
        "one denoise-step event per Euler step (not per forward)",
      );
      assertEquals(observed.forwards, first.forwards);
      assertEquals(
        new Uint8Array(observed.data.buffer),
        new Uint8Array(first.data.buffer),
        "the host path (onEvent) and the resident path differ (the MUST of runDitLoopResident)",
      );
    },
  );
}

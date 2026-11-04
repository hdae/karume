/**
 * End-to-end comparison of the Wan2.1 pipeline (`Pipeline` — the wiring of ADR 0118 step 6, the distribution of step
 * 7) (real GPU).
 *
 * The distribution mirror `models/karume-wan2.1/` (assembled by `dist.py --pipeline wan` — 3 `krm` files and the text
 * embedding asset) is passed through `nodeDirectory` to `Pipeline.fromPretrained`, and `generate` is run (the public
 * fetch surface — manifest resolution, family admission, sequential part reads). The references (`pipeline_steps.*`)
 * are goldens no distribution carries, read from `goldens/wan2.1-t2v-1.3b-f16-dyn/` of the test-asset root (ADR 0139
 * decision 2).
 *
 * On a machine without the distribution this e2e SKIPs explicitly (printing the reason and the assembly command). The
 * all-skip guard `packages/runtime/tests/distribution_gate_test.ts` turns the resulting all-SKIP into a FAIL (set
 * `KARUME_ALLOW_NO_DISTRIBUTION=1` to let it pass on purpose — the same split as the anima / gemma4 e2e).
 *
 * The comparison partner is the recipe's few-step reference (`tools/export-recipes/wan/few_step_ref.py`): the latents
 * of diffusers' `Pipeline` run plainly for 2 steps on CPU f32 (with CFG, guide 5.0, shift 3.0), and the frames of those
 * latents through step 5's tiled decode (the weights are the values rounded to f16 — ADR 0006). The initial noise is
 * the reference's `latents_init` (torch's `randn`), injected through `latents` (except for the one seed-path case —
 * {@link SEED_CASE}).
 *
 * ## Checks (default lanes — decision 8 "the lanes default to few steps")
 *
 * What this e2e owns are the **wiring** checks (prompt mix-ups, guidance drift, step order, hand-offs between stages),
 * not the numeric check of one DiT forward. The forward's numbers are owned by step 3 (`e2e_wan_dit_test.ts`) as the
 * normalized ratio r against an f64 reference. So the partner here can stay the CPU f32 reference — the band is wide
 * enough to include the rounding difference between GPU and CPU f32 (the attribution of step 3 — the GEMM's K
 * reduction), and wiring errors fall outside it (confirmed by fault injection).
 *
 * - **Host self-consistency** (no GPU): from the DiT outputs the reference recorded (cond / uncond), the host CFG +
 *   UniPC reproduces the reference's latents **bit-identically** (it both confirms the reference's precision and checks
 *   the host's wiring).
 * - **Band (opt-in upstream parity)** (decision 8's metric: ratio = max absolute difference ÷ the
 *   reference's max absolute value, against the CPU f32 reference): per observation point (the
 *   latents after each step and the frames) the band is the measured worst of the two deciding cases
 *   (`band-boxing-cats`, seed 20261030 / `band-cat-dog-baking`, seed 20261032) × 5
 *   ({@link LATENT_RATIO_BANDS} / {@link FRAME_RATIO_BAND}), and the acceptance case
 *   (`accept-ferret`, seed 20261033 — seed and prompt differ from the deciding cases) must fall
 *   inside it (decision 8's form — the same deciding / acceptance split as step 3). The judgement
 *   runs only with `KARUME_UPSTREAM_PARITY=1` ("Default run vs opt-in upstream parity" below); the
 *   three cases themselves run by default for their sha256 rows.
 * - **Fault injection** (the acceptance case's initial noise): swapping the positive and negative prompts and a 1%
 *   shift of guide fall outside the band (by a floor of {@link FAULT_MARGIN} ×).
 * - **sha256 environment rows** (ADR 0106): the output frames (the uint8 RGB of all frames concatenated — the rule of
 *   `frameToRgba`) are compared with the environment key's row in `fixtures/references/wan.json`. Rows are made with
 *   `KARUME_REFERENCE=write`. The B570 row (2026-10-02) holds the value written in step 6 by **reading the series
 *   directly** (`fromAssets`), and this file requires that the path through the distribution (`fromPretrained`)
 *   matches the same row — the distribution is an independent copy of the series' `krm`, and the manifest defaults (50
 *   / 5.0 / 3.0) equal step 6's constants, so a split of even 1 bit is a regression of the fetch surface or of the
 *   default resolution.
 * - **Every case names its preset explicitly** (the 2 presets {@link F16_QUANT} / i8): the default preset is decided by
 *   the manifest's `defaultQuant`, and on 2026-10-04 it moved from `f16` to the practical preset (ADR 0120 ruling
 *   2026-10-04, item 4). The rows, bands and CPU references of the existing IDs without a preset name are values of the
 *   `f16` preset, so omitting `quant` would run the same ID with another preset's values ({@link loadPipeline} makes
 *   the preset required and stops the omission at the type level).
 * - **Seed path** ({@link SEED_CASE}): one 2-step case that makes the initial noise from the seed without injecting
 *   `latents`. There is no CPU reference, so only completion, 0 non-finite values and the sha256 environment row hold
 *   it.
 * - **sha rows of the 2 i8 presets** (ADR 0120 decision 4 / step 5 — {@link SEAT_QUANTS}): one case each of the same
 *   request as the seed path with the reference partner `f16+dit8` and the practical preset `f16+dit8-a8-attn8-s16`.
 *   The ID puts the preset name first ({@link seatCaseId} — the same order as anima's `<quant>-<resolution>`; the IDs
 *   of the existing `f16` rows do not move). The row class comes neither from the ID nor from a separate table but from
 *   whether the manifest preset's `session` is empty (ADR 0110 decision 7 — {@link rowClassOf}): `f16+dit8` is a
 *   **reference row** (frozen — not touched by commits that change the practical path), the practical preset a
 *   **practical row** (the detector of regressions and determinism of the practical path — `rewrite` in the same commit
 *   as a commit that changes the practical path's numbers on purpose). The CPU reference was taken only with the f16
 *   weights, so the i8 presets have no band and only the sha256 environment rows hold them (the numeric checks are the
 *   DiT-only r check and the self A/B check — `e2e_wan_dit_test.ts` / `e2e_wan_ab_test.ts`).
 * - **Text encoder path** (ADR 0119 decision 7): the sha rows and band cases above all take the **precomputed
 *   embedding asset path** (`textEncoder: "precomputed"` is explicit — the default is `"gpu"`, so without it they would
 *   silently move to the umT5 path and a row's value would be compared with the sha of another path). The **GPU path**
 *   (umT5 i8 run on the GPU — {@link GPU_TEXT_CASES}) has sha rows under other case ids (the fixed prompt
 *   `boxing-cats` and one free prompt, 2 steps, seed 42), without a CPU reference (the umT5 numeric check is
 *   `e2e_wan_umt5_test.ts`). umT5 lives in another distribution repository (`models/karume-umt5-xxl/` — referenced
 *   across repositories by `text_encoder` of the Wan manifest) and is passed through the source's `crossRepo` mapping.
 *   On a machine without it, only those 2 cases SKIP explicitly.
 *
 * ## Default run vs opt-in upstream parity
 *
 * The band judgement (each reference case's latents and frames inside the band of the diffusers CPU
 * f32 reference) is upstream parity, so it is opt-in behind `KARUME_UPSTREAM_PARITY=1`
 * (helpers/upstream-parity.ts — ADR 0129 decision 4 and its addendum 2026-10-17 (2)). It is gated at
 * the assertion, not by skipping the cases: the three reference cases run by default because their
 * initial noise comes from the fixtures and their outputs carry sha256 rows. Without the opt-in they
 * log the ratios but neither assert the band nor record it as a measurement; with the opt-in the band
 * is asserted before the row is settled, so `KARUME_REFERENCE=write` run together with the opt-in
 * still refuses to write a row for an output outside the band. Everything else runs by default: the
 * GPU-free host self-consistency (bit identity, no tolerance), the admission faults, the sha256 rows
 * (the `f16` seat, the reference and practical i8 seats, the seed path and the GPU text path) and the
 * fault injections. The fault injections read the accept fixture and use the band only as the margin
 * a wiring fault must exceed (ratio > {@link FAULT_MARGIN} × band) — a check on the wiring, not on
 * parity.
 *
 * ## The 50-step end-to-end run (env opt-in — kept out of the default lanes)
 *
 * Only with `KARUME_WAN_FULL_PIPELINE=1`: the default settings (50 steps, CFG, 832×480, 33 frames, `boxing-cats`, seed
 * 42) and the 81 frames of the same settings (the upper limit of the accepted set — ADR 0118 step 8) run once each with
 * the `f16` preset and the practical preset `f16+dit8-a8-attn8-s16` (ADR 0120 step 5 — the 81-frame practical row and
 * the material for the visual check of step 6); completion, 0 non-finite values, times, the VRAM at stage switches,
 * the frame PNGs (all frames + a 4×8 contact sheet) and the RGB artifacts are written to
 * `outputs/verify/<environment key>/<date>_wan-pipeline-full/`, and the sha256 environment rows live in
 * `fixtures/references/wan.json` (the PNGs are for the user's visual check — material for the decision of perf-ledger
 * K-75). No CPU reference is made (decision 8 — accepted by sha256 and by eye; the 81-frame components are covered by
 * the full-size DiT e2e and the 21-chunk VAE e2e).
 * The results directory is kept apart from the default lanes' (`<date>_wan-pipeline/`) because `results.json` is
 * rewritten whole on every run — with the same directory, a later 2-step-only run would erase the 50-step record.
 *
 * ## Observation (not a check)
 *
 * The stage times (wall clock — between `stage` events) and the VRAM at each stage boundary (`drm-total-*` of
 * `/proc/self/fdinfo` — `helpers/drm-usage.ts`). The value at the DiT stage's `end` (right after the Session is torn
 * down, before the VAE stage is built) shows whether the lag of the B570's `destroy()` makes the DiT and VAE
 * allocations overlap (in the 2026-10-02 measurement it had returned to the value before the stage — the reason no
 * release wait was added; module doc of `src/wan/pipeline.ts`).
 * GPU time is not taken — on a measuring device the VAE batch cannot be opened (`Pipeline` construction refuses it).
 *
 * Environments without the distribution or the references SKIP explicitly; environments with only part of the
 * references FAIL (the same discipline as the e2e of steps 4 / 5).
 */

import { isNotFound, readFile, readTextFile, statSync, writeFile } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  type DistributionSource,
  localDirectory,
  parseManifest,
  resolveSelection,
} from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import {
  type GpuContext,
  parseSafetensors,
  type SafetensorsFile,
  type SessionDiagnostics,
} from "@karume/runtime";
import { encodePng } from "../../models/mod.ts";
import {
  type GeneratedVideo,
  frameToRgba,
  type GenerateEvent,
  type GenerateRequest,
  Pipeline,
  type PipelineOptions,
  type Prompt,
  type RunComponent,
} from "../../models/wan.ts";
import { parseWanTextEmbeds } from "../../models/src/wan/text-embeds.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../../models/src/wan/scheduler.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { WAN_ASSEMBLE_COMMAND } from "../../models/tests/helpers/wan-distribution.ts";
import {
  type DrmTimeline,
  formatDrmUsage,
  monitorDrmUsage,
} from "../../models/tests/helpers/drm-usage.ts";
import { verifyRunningGpu } from "../../runtime/tests/helpers/environment.ts";
import {
  openReferences,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleOrObserve,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY } from "./helpers/upstream-parity.ts";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";

/**
 * Bands of the latents of the 2-step run (after each step — `[16,9,60,104]`) (ratio = max absolute difference ÷ the
 * reference's max absolute value). The index is the step (0 = after t 999, 1 = after t 750).
 *
 * Measured (2026-10-02, `deno-intel-graphics-bmg-g21`, a plain comparison with `atol = rtol = 0`). The bands were
 * decided by running only the 2 deciding cases, and the acceptance case was run for the first time afterwards (its
 * values were not used to decide the bands):
 *
 * | Case                  | Role       | `latents.0` (t 999) | `latents.1` (t 750) | Frames (after clamp) |
 * | --------------------- | ---------- | ------------------- | ------------------- | -------------------- |
 * | `band-boxing-cats`    | deciding   | **1.700e-5**        | **1.815e-4**        | **1.691e-3**         |
 * | `band-cat-dog-baking` | deciding   | 1.228e-5            | 5.871e-5            | 3.094e-4             |
 * | Band (worst × 5)      |            | 8.5e-5              | 9.1e-4              | 8.5e-3               |
 * | `accept-ferret`       | acceptance | 1.393e-5            | 5.069e-4            | 5.698e-3             |
 *
 * The acceptance case is at 0.16 / 0.56 / 0.67 × the band.
 *
 * Band = the worst of the 2 deciding cases × 5 per observation point (rounded up to 2 significant digits). The
 * difference comes not from the host but from the DiT's GPU rounding (the attribution of step 3 — the GEMM's K
 * reduction, perf-ledger K-75): the host CFG + UniPC rebuilds the reference's latents **bit-identically** from the
 * reference's DiT outputs (the GPU-free test below), so the whole difference comes from the DiT's 4 forwards. CFG
 * (`5·cond − 4·uncond`) adds up the differences of cond / uncond, and step 1 inherits step 0's difference.
 *
 * Fault injection (the acceptance case's initial noise, `latents.1`): swapped positive and negative prompts 1.777
 * (about 1,950 × the band), guide 5.0 → 5.05 1.333e-1 (about 147 ×). The floor is {@link FAULT_MARGIN} ×.
 *
 * MUST: do not change this value or the deciding cases after seeing the result of the acceptance case
 * (`accept-ferret`) (that would break the independence of the band decision and the acceptance). If the acceptance
 * case falls outside the band, investigate the cause instead of widening the band.
 */
const LATENT_RATIO_BANDS = [8.5e-5, 9.1e-4] as const;

/**
 * Band of the output frames of the 2-step run (`[3,33,480,832]` after the clamp — the reference is clamped too before
 * the comparison).
 *
 * 8.5e-3 = the worst deciding case (1.691e-3 of `band-boxing-cats`) × 5 (the table of {@link LATENT_RATIO_BANDS}). Most
 * of the difference is the latent difference spread through the VAE (the ratio of step 5's tiled decode itself is
 * 2.65e-6).
 *
 * MUST: do not change this value after seeing the acceptance result (the same as {@link LATENT_RATIO_BANDS}).
 */
const FRAME_RATIO_BAND = 8.5e-3;

/**
 * Floor of the factor by which faults (swapped positive and negative prompts, a 1% shift of guide) must stay away from
 * the band (the floor of the addendum 2026-10-02 "an injection that does not turn red is a sign that the band is too
 * wide" — 2, the same as step 3).
 */
const FAULT_MARGIN = 2;

/** The references' place (goldens no distribution carries — the test-asset root, ADR 0139 decision 2). */
const GOLDEN_ROOT = goldenDir("wan2.1-t2v-1.3b-f16-dyn");
/** The command that fetches the goldens into the local test-asset root (ADR 0139 decision 3). */
const FETCH = "node tools/test-assets/fetch.ts";
/**
 * A complete fetch of the pinned test-asset dataset makes a missing reference a FAIL (the completeness test below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;
/** The distribution mirror (the default output directory of `dist.py --pipeline wan`). */
const DIST_ROOT = new URL("../../../models/karume-wan2.1/", import.meta.url);
/**
 * Key of the embedding asset in the manifest's `assets` (`WAN_TEXT_EMBEDS_ROLE` of the recipe `wan/distribution.py`).
 */
const TEXT_EMBEDS = "text_embeds";
const STEPS = 2;

/**
 * The reference cases (the canonical list is `FIXTURE_CASES` of the recipe `wan/few_step_ref.py`). 2 deciding
 * (`band`) + 1 acceptance (`accept`). MUST: do not change the deciding cases after seeing the acceptance result.
 */
const CASES: readonly { readonly name: string; readonly role: "band" | "accept" }[] = [
  { name: "band-boxing-cats", role: "band" },
  { name: "band-cat-dog-baking", role: "band" },
  { name: "accept-ferret", role: "accept" },
];
const ACCEPT_CASE = "accept-ferret";

/**
 * The 2-step seed path (no `latents` injected; the initial noise is made from the seed — it exercises the generator
 * `WanRandn` and the seed → latent layout). The generator's sequence differs from torch's `randn`, so no CPU reference
 * can be made, and only the sha256 environment row holds the value. It sits in the default lanes because the only
 * other row holding the seed path's value is the opt-in 50-step run (the default lanes and the full verify would let
 * a generator regression through).
 */
const SEED_CASE = { id: "2step-seed-boxing-cats-seed42", prompt: "boxing-cats", seed: 42 } as const;

/**
 * The `f16` preset (the reference side — the preset closest to the original weights; only this preset has a CPU
 * reference (weights rounded to f16) and bands). The rows of the existing IDs without a preset name hold this
 * preset's values (ADR 0120 ruling 2026-10-04, item 4 — explicit so the values hold even though the default preset
 * moved to the practical preset).
 */
const F16_QUANT = "f16";
/** The practical preset (ADR 0120 decision 1 — the manifest's default preset since 2026-10-04). */
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";
/**
 * The 2 i8 presets that run the 2-step seed path in addition to the `f16` preset (ADR 0120 decision 4 / step 5): the
 * reference partner (reference row) and the practical preset (practical row). The row class is derived from the
 * manifest by {@link rowClassOf} — this order claims no class.
 */
const SEAT_QUANTS = ["f16+dit8", PRACTICAL_QUANT] as const;

/** ID of a case that names its preset (preset name first — the existing `f16` rows keep their IDs without one). */
const seatCaseId = (quant: string, base: string): string => `${quant}-${base}`;

/**
 * The GPU-path cases (the sha rows of the ADR 0119 addendum "design of step 10d" — 2 steps, seed 42, the `f16` preset,
 * the default negative). The fixed prompt passes the original text of the asset's row (the same string with only the
 * path differing — comparable with {@link SEED_CASE} of the asset path), and the free prompt is a short English
 * sentence not in the asset (the ID does not spell it out but calls it by name — when the text changes, change the ID
 * too).
 *
 * MUST: when the text, seed or step count changes, change the ID too (a row's value would be compared with the sha of
 * another condition).
 */
const GPU_TEXT_CASES: readonly {
  readonly id: string;
  readonly prompt: { readonly asset: string } | { readonly text: string };
  readonly seed: number;
}[] = [
  { id: "gpu-text-2step-boxing-cats-seed42", prompt: { asset: "boxing-cats" }, seed: 42 },
  {
    id: "gpu-text-2step-free-fox-seed42",
    prompt: { text: "A red fox trots through fresh snow in a quiet birch forest at sunrise." },
    seed: 42,
  },
];

/** Opt-in for the 50-step end-to-end run (decision 8 — run only before a release and when regenerating references). */
const FULL_PIPELINE = process.env.KARUME_WAN_FULL_PIPELINE === "1";
/**
 * Requests of the 50-step end-to-end run: the default settings + the first fixed prompt, seed 42 (the first of step
 * 7's visual checks), and the 81 frames of the same request (ADR 0118 step 8), with the `f16` preset and the practical
 * preset (ADR 0120 step 5 — the 81-frame practical row and the material for step 6's visual check). Cases that omit
 * `frames` exercise the default ({@link DEFAULT_FRAMES}).
 * The order is by allocation size (`f16` preset 81 → practical preset 81 → `f16` preset 33 → practical preset 33 —
 * the practical preset's VRAM peak in the DiT stage is expected to be about 1 GiB smaller; the expected-value table of
 * ADR 0120). What is left of an earlier allocation can push a later large one into OOM — the lag of the B570's
 * `destroy()`, the same reason as putting the full size first in `e2e_wan_dit_test.ts`.
 */
const FULL_CASES: readonly {
  readonly id: string;
  readonly prompt: string;
  readonly seed: number;
  readonly frames?: number;
  /** Preset (required — when omitted it silently moves to the manifest's `defaultQuant`). */
  readonly quant: string;
}[] = [
  {
    id: "50step-boxing-cats-seed42-81f",
    prompt: "boxing-cats",
    seed: 42,
    frames: 81,
    quant: F16_QUANT,
  },
  {
    id: seatCaseId(PRACTICAL_QUANT, "50step-boxing-cats-seed42-81f"),
    prompt: "boxing-cats",
    seed: 42,
    frames: 81,
    quant: PRACTICAL_QUANT,
  },
  { id: "50step-boxing-cats-seed42", prompt: "boxing-cats", seed: 42, quant: F16_QUANT },
  {
    id: seatCaseId(PRACTICAL_QUANT, "50step-boxing-cats-seed42"),
    prompt: "boxing-cats",
    seed: 42,
    quant: PRACTICAL_QUANT,
  },
];
/** Frame count when `frames` is omitted (the default of `src/wan/pipeline.ts`). */
const DEFAULT_FRAMES = 33;
/**
 * Step count of the 50-step end-to-end run. The request does not specify it and leaves it to the manifest default, so
 * it is held by the number of denoise-steps observed (running `KARUME_REFERENCE=write` against a distribution whose
 * default changed would write the output of another step count into the `50step-…` rows).
 */
const FULL_STEPS = 50;

const ASSEMBLE_UMT5_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";

/**
 * The umT5 distribution mirror (the target `text_encoder` of the Wan manifest references across repositories — ADR 0119
 * addendum "design of step 10d" D).
 */
const UMT5_ROOT = new URL("../../../models/karume-umt5-xxl/", import.meta.url);
/** Component name in the manifest (`TEXT_ENCODER` of `src/wan/pipeline.ts`). */
const TEXT_ENCODER = "text_encoder";

const fixtureUrl = (name: string): URL =>
  new URL(`pipeline_steps.${name}.safetensors`, GOLDEN_ROOT);

const fileExists = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

const DIST_PRESENT = fileExists(new URL("karume.json", DIST_ROOT));
if (!DIST_PRESENT) {
  console.warn(
    `[karume] The distribution mirror ${DIST_ROOT.pathname} is missing, so the Wan pipeline comparison is SKIPPED. ` +
      `Assemble: ${WAN_ASSEMBLE_COMMAND} (the all-skip guard distribution_gate_test.ts turns an all-SKIP into a FAIL)`,
  );
}
const FIXTURES_PRESENT = CASES.map(({ name }) => fileExists(fixtureUrl(name)));
const ANY_FIXTURE = FIXTURES_PRESENT.some(Boolean);
if (!ANY_FIXTURE) {
  console.warn(
    `[karume] no few-step reference under ${GOLDEN_ROOT.pathname}, so Wan's pipeline checks are SKIPPED. ` +
      `(${MISSING_GOLDENS})`,
  );
}
/**
 * The comparison can run (both the distribution and the references are present — a machine with only part of the
 * references FAILs in the assets check below).
 */
const ANY_PRESENT = DIST_PRESENT && ANY_FIXTURE;

/**
 * Part 0 of the umT5 container in the distribution's default selection (undefined when absent — an old distribution
 * without umT5). The cross-repository repo is taken from this declaration (the key is not copied by hand — the mapping
 * follows when regenerating the distribution changes the repo).
 */
const TEXT_ENCODER_PART0 = DIST_PRESENT
  ? resolveSelection(parseManifest(await readTextFile(new URL("karume.json", DIST_ROOT))))
      .containers[TEXT_ENCODER]?.parts[0]
  : undefined;
/**
 * The repo of umT5's cross-repository target (undefined for a container in the same repository — no mapping needed).
 */
const UMT5_REPO = TEXT_ENCODER_PART0?.repo;
const UMT5_PRESENT = fileExists(new URL("karume.json", UMT5_ROOT));
/**
 * The GPU path can be built (the distribution has umT5 and, if it is cross-repository, the target's mirror is present
 * too).
 */
const GPU_TEXT_PRESENT =
  DIST_PRESENT && TEXT_ENCODER_PART0 !== undefined && (UMT5_REPO === undefined || UMT5_PRESENT);
if (DIST_PRESENT && !GPU_TEXT_PRESENT) {
  console.warn(
    TEXT_ENCODER_PART0 === undefined
      ? `[karume] The distribution mirror ${DIST_ROOT.pathname} has no umT5 (${TEXT_ENCODER}), so the Wan GPU-path` +
          ` cases and the umT5 admission faults are SKIPPED. Reassemble: ${WAN_ASSEMBLE_COMMAND}`
      : `[karume] The umT5 distribution mirror ${UMT5_ROOT.pathname} (cross-repository target ${UMT5_REPO}) is missing, so the Wan GPU-path` +
          ` cases and the umT5 admission faults are SKIPPED. Assemble: ${ASSEMBLE_UMT5_COMMAND}`,
  );
}

/**
 * The cross-repository mapping (empty when the umT5 container is in the same repository — it would not be used anyway).
 */
const crossRepoOf = (umt5: DistributionSource): Record<string, DistributionSource> =>
  UMT5_REPO === undefined ? {} : { [UMT5_REPO]: umt5 };

/**
 * The distribution's embedding asset (taken from `assets` of the manifest's default selection — the path is not spelled
 * again).
 */
const readDistributionEmbeds = async (): Promise<Uint8Array<ArrayBuffer>> => {
  const manifest = parseManifest(await readTextFile(new URL("karume.json", DIST_ROOT)));
  const ref = resolveSelection(manifest).assets[TEXT_EMBEDS];
  assert(ref !== undefined, `the distribution manifest's assets have no '${TEXT_EMBEDS}'`);
  return await readFile(new URL(ref.path, DIST_ROOT));
};

/**
 * Reads the distribution through a source handle (neither network nor CacheStorage). The caller always names the
 * path and the preset (neither the default `"gpu"` nor the manifest's default preset is silently taken — "Text encoder
 * path" and "Every case names its preset explicitly" of the module doc).
 * `"gpu"` passes umT5's cross-repository target through the `crossRepo` mapping.
 */
const loadPipeline = (
  gpu: GpuContext,
  diagnostics: Map<RunComponent, SessionDiagnostics>,
  textEncoder: NonNullable<PipelineOptions["textEncoder"]>,
  quant: string,
): Promise<Pipeline> =>
  Pipeline.fromPretrained(
    textEncoder === "gpu"
      ? nodeDirectory(DIST_ROOT, { crossRepo: crossRepoOf(nodeDirectory(UMT5_ROOT)) })
      : nodeDirectory(DIST_ROOT),
    {
      gpu,
      textEncoder,
      quant,
      onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
    },
  );

/**
 * Derives the sha row class (ADR 0110 decision 7) from the `session` of the preset of the distribution manifest's
 * default model (empty = reference row, non-empty = practical row). No separate table — when the preset's declaration
 * changes, the label changes too.
 */
const rowClassOf = async (quant: string): Promise<string> => {
  const manifest = parseManifest(await readTextFile(new URL("karume.json", DIST_ROOT)));
  const model = manifest.models[manifest.defaultModel];
  assert(
    model !== undefined,
    `the distribution manifest has no default model '${manifest.defaultModel}'`,
  );
  const seat = Object.hasOwn(model.quants, quant) ? model.quants[quant] : undefined;
  assert(seat !== undefined, `the distribution manifest has no preset '${quant}'`);
  return Object.keys(seat.session).length === 0
    ? `${quant}: reference row (empty session — frozen)`
    : `${quant}: practical row (non-empty session — detector of practical-path regressions and determinism)`;
};

const references = openReferences(
  new URL("../../models/tests/fixtures/references/wan.json", import.meta.url),
);
const results = openResults("wan-pipeline");
/** Results directory of the 50-step end-to-end run (kept apart from the default lanes — module doc). */
const fullResults = openResults("wan-pipeline-full");

/** A value a `beforeAll` assigns, read inside an `it` (fails loudly when the hook did not assign it). */
const assigned = <T>(value: T | undefined, what: string): T => {
  assert(value !== undefined, `${what} is not assigned (its beforeAll did not complete)`);
  return value;
};

test.skipIf(!ANY_FIXTURE && !FETCHED)(
  "Wan end-to-end references: the 3 few-step references are present",
  () => {
    assertEquals(
      FIXTURES_PRESENT,
      FIXTURES_PRESENT.map(() => true),
      `missing: ${CASES.filter((_, index) => !FIXTURES_PRESENT[index])
        .map(({ name }) => fixtureUrl(name).pathname)
        .join(", ")} (${MISSING_GOLDENS})`,
    );
  },
);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * A copy of the distribution manifest with its default model rewritten (fault injection that replaces only
 * `karume.json` — not a single byte of the weights changes). The faults also put in values outside the manifest's
 * type, so it does not go through `parseManifest` and descends from unknown with type predicates.
 */
const overrideModel = (
  original: string,
  patch: (model: Record<string, unknown>) => Record<string, unknown>,
): Record<string, unknown> => {
  const manifest: unknown = JSON.parse(original);
  assert(
    isRecord(manifest) && isRecord(manifest.models),
    "the distribution manifest has no models",
  );
  const name = manifest.defaultModel;
  assert(typeof name === "string", "the distribution manifest has no defaultModel");
  const model = manifest.models[name];
  assert(isRecord(model), `the distribution manifest has no default model '${name}'`);
  return { ...manifest, models: { ...manifest.models, [name]: patch(model) } };
};

/**
 * Prefix marking paths read from umT5's cross-repository target (so they are not confused with same-spelled paths of
 * this repository).
 */
const UMT5_PREFIX = "umt5:";

/**
 * A source that replaces only `karume.json` with `manifest` and reads the rest from the distribution (pushing the paths
 * read onto `requested` — to count the order and number of fetches). Reads from umT5's cross-repository target go onto
 * the same list with {@link UMT5_PREFIX} (on a machine without the target's mirror no mapping is passed — touching it
 * makes hub fail with "no cross-repository target").
 */
const countingSource = (manifest: Record<string, unknown>, requested: string[]) => {
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  const umt5 = localDirectory(
    {
      readFile: async (path) => {
        requested.push(`${UMT5_PREFIX}${path}`);
        return await readFile(new URL(path, UMT5_ROOT));
      },
    },
    { label: "wan-admission-fault-umt5" },
  );
  return localDirectory(
    {
      readFile: async (path) => {
        requested.push(path);
        if (path === "karume.json") return manifestBytes;
        return await readFile(new URL(path, DIST_ROOT));
      },
    },
    { label: "wan-admission-fault", ...(UMT5_PRESENT ? { crossRepo: crossRepoOf(umt5) } : {}) },
  );
};

describe.skipIf(!DIST_PRESENT)(
  "Wan distribution family admission (no GPU): a pipeline major, pipelineConfig, a mismatch in a quant's session and " +
    "missing tokenizer assets of the GPU path fail by name, without fetching a single weight part from part 1 on or " +
    "a single asset before that (the precomputed path does not touch a single byte of umT5)",
  () => {
    let original: string | undefined;
    let heavyPaths: ReadonlySet<string> | undefined;
    beforeAll(async () => {
      original = await readTextFile(new URL("karume.json", DIST_ROOT));
      const parsed = parseManifest(original);
      // part 0 = descriptor is the input of admission itself and, by contract, is fetched before the check
      // (phase 1 of packages/models/src/hub/components.ts). What must be touched only after the check is part 1
      // onward and the assets (embeddings, tokenizer, the raw RoPE table).
      const model = parsed.models[parsed.defaultModel];
      assert(
        model !== undefined,
        `the distribution manifest has no default model '${parsed.defaultModel}'`,
      );
      heavyPaths = new Set([
        ...Object.values(model.weights).flatMap((entry) =>
          Object.values(entry).flatMap((weights) =>
            weights.container.parts
              .slice(1)
              .map((ref) => (ref.repo === undefined ? ref.path : `${UMT5_PREFIX}${ref.path}`)),
          ),
        ),
        ...Object.values(model.assets).map((ref) => ref.path),
      ]);
      assert(heavyPaths.size > 0);
    });
    const faults: readonly {
      readonly label: string;
      readonly patch: (model: Record<string, unknown>) => Record<string, unknown>;
      /** Path (the existing faults use the precomputed path — they run whether or not umT5 is present). */
      readonly textEncoder: NonNullable<PipelineOptions["textEncoder"]>;
      readonly message: string;
    }[] = [
      {
        label: "pipeline major 2",
        patch: (model) => ({ ...model, pipeline: "wan/2" }),
        textEncoder: "precomputed",
        message: "unsupported major",
      },
      {
        label: "unknown key in pipelineConfig",
        patch: (model) => ({
          ...model,
          pipelineConfig: {
            scheduler: { shift: 3, type: "unipc" },
            defaults: { steps: 50, guidance: 5 },
          },
        }),
        textEncoder: "precomputed",
        message: "unknown key 'type'",
      },
      {
        // The acceptance table (WAN_SESSION_POLICY) accepts only the 3 fields of the linear / attention execution form
        // (ADR 0120 decision 7). A declaration of a knob that is valid manifest vocabulary but that Wan does not accept
        // (the reduction of the states-form attention — meant for a decoder's decode) fails by name before the weights
        // are fetched.
        label: "a quant declares an execution knob not in the acceptance table",
        patch: (model) => {
          const quants = model.quants;
          assert(isRecord(quants) && isRecord(quants.f16), "the distribution has no f16 quant");
          return {
            ...model,
            quants: {
              ...quants,
              f16: { ...quants.f16, session: { stateAttentionReduce: "parallel" } },
            },
          };
        },
        textEncoder: "precomputed",
        message: "session.stateAttentionReduce is not supported",
      },
      {
        // The GPU path reads the tokenizer assets — without their declaration it fails before fetching umT5's part 1
        // onward (about 5.3 GiB at i8). Part 0 of the umT5 container (the descriptor) is an admission input, so
        // fetching it is allowed.
        label: "GPU path with no umT5 tokenizer assets in the manifest",
        patch: (model) => {
          const assets = model.assets;
          assert(isRecord(assets), "the distribution manifest has no assets");
          const { umt5_tokenizer: _dropped, ...rest } = assets;
          return { ...model, assets: rest };
        },
        textEncoder: "gpu",
        message: "have no umT5 tokenizer asset 'umt5_tokenizer'",
      },
    ];
    for (const { label, patch, textEncoder, message } of faults) {
      it.skipIf(textEncoder === "gpu" && !GPU_TEXT_PRESENT)(
        `${label}（textEncoder: ${textEncoder}）`,
        async () => {
          const heavyPathSet = assigned(heavyPaths, "heavyPaths");
          const requested: string[] = [];
          const source = countingSource(
            overrideModel(assigned(original, "original"), patch),
            requested,
          );
          await assertRejects(
            () =>
              Pipeline.fromPretrained(source, {
                textEncoder,
                // One fault rewrites the `f16` preset's declaration — without naming the preset it would move to the
                // default preset and never touch the fault.
                quant: F16_QUANT,
              }),
            Error,
            message,
          );
          assert(requested.includes("karume.json"), `the manifest was not read: ${requested}`);
          const heavy = requested.filter((path) => heavyPathSet.has(path));
          assertEquals(heavy, [], "fetched weight parts from part 1 on or assets before admission");
          if (textEncoder === "precomputed") {
            const umt5 = requested.filter((path) => path.startsWith(UMT5_PREFIX));
            assertEquals(umt5, [], "the precomputed path touches the umT5 container");
          }
        },
      );
    }
  },
);

/** One reference (F32 tensors and metadata). */
type Fixture = {
  readonly tensor: (key: string) => Float32Array<ArrayBuffer>;
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await readFile(url);
  const file: SafetensorsFile = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    tensor: (key) => {
      const view = file.tensors.get(key);
      assert(view !== undefined && view.dtype === "F32", `${url.pathname}: '${key}' is not F32`);
      return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
    },
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: no metadata '${key}'`);
      return value;
    },
  };
};

/**
 * Summary of a difference (ratio = max absolute difference ÷ the reference's max absolute value — decision 8's metric).
 */
type Difference = {
  readonly maxAbs: number;
  /**
   * Maximum per-element relative difference (elements where the reference is 0 are excluded — for the results record).
   */
  readonly maxRel: number;
  readonly referenceMax: number;
  readonly ratio: number;
  readonly nonFinite: number;
};

const difference = (got: Float32Array, want: Float32Array): Difference => {
  assertEquals(got.length, want.length, "element count");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMax = 0;
  let nonFinite = 0;
  for (let index = 0; index < want.length; index += 1) {
    if (!Number.isFinite(got[index])) {
      nonFinite += 1;
      continue;
    }
    const absolute = Math.abs(got[index] - want[index]);
    maxAbs = Math.max(maxAbs, absolute);
    if (want[index] !== 0) maxRel = Math.max(maxRel, absolute / Math.abs(want[index]));
    referenceMax = Math.max(referenceMax, Math.abs(want[index]));
  }
  return { maxAbs, maxRel, referenceMax, ratio: maxAbs / referenceMax, nonFinite };
};

const formatDifference = (label: string, diff: Difference): string =>
  `${label} ratio ${diff.ratio.toExponential(3)} (maxAbs ${diff.maxAbs.toExponential(3)} / reference ${diff.referenceMax.toFixed(
    3,
  )})`;

/** Upstream's `clamp(-1, 1)` (the reference frames are written before the clamp). */
const clamped = (values: Float32Array): Float32Array =>
  values.map((value) => (value > 1 ? 1 : value < -1 ? -1 : value));

/** The uint8 RGB of all output frames concatenated (the sha256 artifact — decision 8). */
const rgbBytes = (video: GeneratedVideo): Uint8Array<ArrayBuffer> => {
  const plane = video.width * video.height;
  const out = new Uint8Array(video.frames * plane * 3);
  for (let frame = 0; frame < video.frames; frame += 1) {
    const rgba = frameToRgba(video, frame);
    for (let index = 0; index < plane; index += 1) {
      out.set(rgba.subarray(index * 4, index * 4 + 3), (frame * plane + index) * 3);
    }
  }
  return out;
};

/**
 * Observation of one generate (from the events — the latents after each step, stage times, VRAM at stage boundaries).
 */
type Observed = {
  readonly video: GeneratedVideo;
  readonly latents: readonly Float32Array[];
  /** Stage name → wall-clock ms (from `start` to `end`). */
  readonly stageMs: Readonly<Record<string, number>>;
  readonly wallMs: number;
  readonly drm: DrmTimeline;
  readonly diagnostics: ReadonlyMap<RunComponent, SessionDiagnostics>;
};

/**
 * Runs generate once and collects the observation from the events. With `abortAfterDenoise` it throws at the DiT
 * stage's `end` (after the Session is torn down) and skips the VAE stage (fault injection looks only at the latents —
 * it does not pay the VAE's 2 minutes).
 */
const observe = async (
  pipeline: Pipeline,
  diagnostics: Map<RunComponent, SessionDiagnostics>,
  request: Omit<GenerateRequest, "onEvent">,
  options: { readonly abortAfterDenoise?: boolean } = {},
): Promise<Observed | { readonly latents: readonly Float32Array[] }> => {
  const latents: Float32Array[] = [];
  const stageStarted = new Map<string, number>();
  const stageMs: Record<string, number> = {};
  const monitor = monitorDrmUsage();
  const abort = new Error("stop after the DiT stage (fault injection)");
  diagnostics.clear();
  const started = performance.now();
  const onEvent = (event: GenerateEvent): void => {
    if (event.kind === "denoise-step") latents.push(event.copyLatents().data);
    if (event.kind !== "stage") return;
    const label = `${event.component} ${event.at}`;
    monitor.mark(label);
    monitor.enter(label);
    if (event.at === "start") stageStarted.set(event.component, performance.now());
    else stageMs[event.component] = performance.now() - (stageStarted.get(event.component) ?? 0);
    if (
      options.abortAfterDenoise === true &&
      event.component === "transformer" &&
      event.at === "end"
    ) {
      throw abort;
    }
  };
  try {
    const video = await pipeline.generate({ ...request, onEvent });
    const wallMs = performance.now() - started;
    return {
      video,
      latents,
      stageMs,
      wallMs,
      drm: monitor.stop(),
      diagnostics: new Map(diagnostics),
    };
  } catch (error) {
    monitor.stop();
    if (error === abort) return { latents };
    throw error;
  }
};

/**
 * Lines of one observation (stage times, the VRAM peak per interval and the values at boundaries, the diagnostics
 * breakdown).
 */
const formatObserved = (observed: Observed): string[] => {
  const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(3)} GiB`;
  const lines = [
    `wall ${(observed.wallMs / 1000).toFixed(1)} s, stages ${Object.entries(observed.stageMs)
      .map(([stage, ms]) => `${stage} ${(ms / 1000).toFixed(1)} s`)
      .join(", ")}`,
    ...[...observed.drm.peaks].map(
      ([phase, peak]) => `VRAM peak [${phase}] ${formatDrmUsage(peak)}`,
    ),
    ...observed.drm.marks.map(
      ({ label, usage }) => `VRAM mark [${label}] ${formatDrmUsage(usage)}`,
    ),
  ];
  for (const [component, diagnostics] of observed.diagnostics) {
    lines.push(
      `${component}: weights ${gib(diagnostics.weights.allocatedBytes)}, backing ${gib(
        diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
      )}, geometry ${diagnostics.geometryProfile}, ${diagnostics.submit.submitCount} submits, max window mean ${
        diagnostics.submit.chunkBudget.maxWindowMeanMs?.toFixed(1) ?? "—"
      } ms, ${diagnostics.submit.chunkBudget.overBudgetChunks} over budget`,
    );
  }
  return lines;
};

const textOf = (
  prompts: readonly Prompt[],
  name: string,
  form: "prompt" | "normalized",
): string => {
  const entry = prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `the embedded assets have no '${name}'`);
  return entry[form];
};

test.skipIf(!ANY_PRESENT)(
  "Wan end-to-end (host, no GPU): the distribution's embedding asset and σ / timestep match the reference metadata, and " +
    "CFG + UniPC reproduces the reference latents bit-identically from the reference DiT outputs",
  async () => {
    const embedsBytes = await readDistributionEmbeds();
    const embeds = parseWanTextEmbeds(embedsBytes.buffer);
    assertEquals(
      embeds.entries.map(({ name, role, tokens }) => [name, role, tokens]),
      [
        ["boxing-cats", "positive", 28],
        ["ferret", "positive", 118],
        ["cat-dog-baking", "positive", 50],
        ["negative", "negative", 126],
      ],
    );
    assertEquals(embeds.width, 4096);
    // ftfy normalizes the full-width comma to the ASCII "," (decision 4 — the normalized string is kept in the metadata
    // too).
    assert(!textOf(embeds.entries, "negative", "normalized").includes("，"));
    assert(textOf(embeds.entries, "negative", "prompt").includes("，"));
    const embedsSha = await sha256Hex(embedsBytes);
    const schedule = wanUniPcSchedule(STEPS, 3, WAN_UNIPC_CONFIG.numTrainTimesteps);
    for (const { name } of CASES) {
      const fixture = await readFixture(fixtureUrl(name));
      assertEquals(
        fixture.meta("text_embeds_sha256"),
        embedsSha,
        `${name}: the embedding asset the reference was made from`,
      );
      assertEquals(
        [fixture.meta("steps"), fixture.meta("guidance_scale"), fixture.meta("flow_shift")],
        [String(STEPS), "5.0", "3.0"],
      );
      assertEquals(fixture.meta("negative"), "negative");
      const timesteps: unknown = JSON.parse(fixture.meta("timesteps"));
      const sigmas: unknown = JSON.parse(fixture.meta("sigmas"));
      assertEquals(timesteps, schedule.timesteps);
      assertEquals(sigmas, [...schedule.sigmas]);
      // Rebuild the reference latents with the host CFG + UniPC from the DiT outputs the reference recorded (forward
      // hooks).
      const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let latents: Float32Array<ArrayBuffer> = fixture.tensor("latents_init");
      for (let step = 0; step < STEPS; step += 1) {
        const velocity = wanClassifierFreeGuidance(
          fixture.tensor(`noise_cond.${step}`),
          fixture.tensor(`noise_uncond.${step}`),
          5,
        );
        latents = sampler.step(velocity, latents);
        const want = fixture.tensor(`latents.${step}`);
        const bits = new Uint32Array(latents.buffer, latents.byteOffset, latents.length);
        const wantBits = new Uint32Array(want.buffer, want.byteOffset, want.length);
        assert(
          bits.every((value, index) => value === wantBits[index]),
          `${name}: step ${step} latents are not bit-identical (${formatDifference(
            "latents",
            difference(latents, want),
          )})`,
        );
      }
    }
  },
);

describe.skipIf(!ANY_PRESENT || !GPU_AVAILABLE)(
  "Wan end-to-end 2 steps (real GPU): 832×480, 33 frames, fault injections outside the band, " +
    "sha256 environment rows (the `f16` preset and the 2 i8 presets: reference row / practical row) — the latents and frames inside the band " +
    "of the diffusers reference only with KARUME_UPSTREAM_PARITY=1",
  () => {
    let deviceLost: string | undefined;
    // One device and one `f16` pipeline shared by the its below (built before them, disposed after them all).
    let gpuContext: GpuContext | undefined;
    let f16Pipeline: Pipeline | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
      f16Pipeline = await loadPipeline(gpuContext, diagnostics, "precomputed", F16_QUANT);
    });
    afterAll(async () => {
      if (gpuContext === undefined) return;
      try {
        await f16Pipeline?.dispose();
      } finally {
        // MUST: wait for the releases before dropping the device (the B570's destroy lag — leaves budget for later
        // tests).
        await settleReleases(gpuContext);
        gpuContext.destroy();
      }
    });

    for (const { name, role } of CASES) {
      it(`${name}（${role}）`, async () => {
        const pipeline = assigned(f16Pipeline, "f16Pipeline");
        const fixture = await readFixture(fixtureUrl(name));
        const promptName = fixture.meta("prompt");
        // band cases look up the original text, accept the normalized string (exercising both spellings of the accepted
        // set).
        const prompt = textOf(
          pipeline.prompts,
          promptName,
          role === "band" ? "prompt" : "normalized",
        );
        const id = `2step-${name}`;
        let settlement: ReferenceSettlement | undefined;
        await runRecordedCase(results, { id }, async ({ measurements }) => {
          const observed = await observe(pipeline, diagnostics, {
            prompt,
            latents: fixture.tensor("latents_init"),
            steps: STEPS,
          });
          assert("video" in observed, "generate did not run to the end");
          assertEquals(observed.latents.length, STEPS, "number of denoise-steps");
          const diffs = [
            ...observed.latents.map((latents, step) => ({
              output: `latents.${step}`,
              band: LATENT_RATIO_BANDS[step],
              diff: difference(latents, fixture.tensor(`latents.${step}`)),
            })),
            {
              output: "frames",
              band: FRAME_RATIO_BAND,
              diff: difference(observed.video.data, clamped(fixture.tensor("frames"))),
            },
          ];
          // A measurement records the band it was judged against, so it exists only when the band
          // is judged (upstream parity is opt-in — module doc).
          if (UPSTREAM_PARITY) {
            for (const { output, band, diff } of diffs) {
              measurements.push({
                output,
                maxAbs: diff.maxAbs,
                maxRel: diff.maxRel,
                tolerance: { atol: band * diff.referenceMax, rtol: 0 },
                stage: "karume",
              });
            }
          }
          const notes = [
            ...diffs.map(({ output, diff }) => formatDifference(output, diff)),
            ...(UPSTREAM_PARITY
              ? []
              : ["band not judged (upstream parity is opt-in: KARUME_UPSTREAM_PARITY=1)"]),
            ...formatObserved(observed),
          ];
          console.log(`[wan-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
          for (const { output, diff } of diffs) {
            assertEquals(diff.nonFinite, 0, `${id}: ${output} non-finite`);
          }
          // MUST stay before the row is settled: with the opt-in, an output outside the band is
          // never written as a row under KARUME_REFERENCE=write.
          if (UPSTREAM_PARITY) {
            for (const { output, band, diff } of diffs) {
              assert(
                diff.ratio <= band,
                `${id}: ${formatDifference(output, diff)} outside band ${band}`,
              );
            }
          }
          assertEquals(deviceLost, undefined, "device lost");
          const outcome = await settleOrObserve(references, results, {
            id,
            artifact: `${id}.rgb`,
            bytes: rgbBytes(observed.video),
          });
          settlement = outcome.settlement;
          return { ...outcome.fields, note: notes.join(" / ") };
        });
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(id, settlement, references));
        }
      });
    }

    /**
     * One 2-step seed-path case (completion, 0 non-finite values, sha256 environment row). The `f16` preset and the 2
     * i8 presets all go through this same case (only the pipeline, the ID and the row-class label differ per preset).
     */
    const seedPath = async (target: Pipeline, id: string, quant: string): Promise<void> => {
      let settlement: ReferenceSettlement | undefined;
      await runRecordedCase(results, { id }, async () => {
        const observed = await observe(target, diagnostics, {
          prompt: textOf(target.prompts, SEED_CASE.prompt, "prompt"),
          seed: SEED_CASE.seed,
          steps: STEPS,
        });
        assert("video" in observed, "generate did not run to the end");
        assertEquals(observed.latents.length, STEPS, "number of denoise-steps");
        const { video } = observed;
        assertEquals([video.frames, video.height, video.width], [DEFAULT_FRAMES, 480, 832]);
        const nonFinite = video.data.reduce(
          (count, value) => count + (Number.isFinite(value) ? 0 : 1),
          0,
        );
        const notes = [
          await rowClassOf(quant),
          `non-finite ${nonFinite}`,
          ...formatObserved(observed),
        ];
        console.log(`[wan-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
        assertEquals(nonFinite, 0, `${id}: non-finite`);
        assertEquals(deviceLost, undefined, "device lost");
        const outcome = await settleOrObserve(references, results, {
          id,
          artifact: `${id}.rgb`,
          bytes: rgbBytes(video),
        });
        settlement = outcome.settlement;
        return { ...outcome.fields, note: notes.join(" / ") };
      });
      if (settlement?.check.status === "fail") {
        throw new Error(referenceMismatchMessage(id, settlement, references));
      }
    };

    it(`${SEED_CASE.id} (seed path, sha256 environment row)`, async () => {
      await seedPath(assigned(f16Pipeline, "f16Pipeline"), SEED_CASE.id, F16_QUANT);
    });

    // Fault injection (the acceptance case's initial noise — only the latents are looked at, so it stops before the VAE
    // stage). Each fault it reads the accept fixture and its prompts itself (the request is built from them).
    const faults: readonly {
      readonly label: string;
      readonly request: (prompts: {
        readonly ferret: string;
        readonly negative: string;
      }) => Omit<GenerateRequest, "onEvent" | "latents" | "steps">;
    }[] = [
      {
        label: "positive and negative prompts swapped",
        request: ({ ferret, negative }) => ({ prompt: negative, negativePrompt: ferret }),
      },
      { label: "guide 5.0 → 5.05", request: ({ ferret }) => ({ prompt: ferret, guidance: 5.05 }) },
    ];
    const lastBand = LATENT_RATIO_BANDS[STEPS - 1];
    for (const { label, request } of faults) {
      it(`fault injection: ${label} → ${ACCEPT_CASE} latents outside the band`, async () => {
        const pipeline = assigned(f16Pipeline, "f16Pipeline");
        const accept = await readFixture(fixtureUrl(ACCEPT_CASE));
        const ferret = textOf(pipeline.prompts, accept.meta("prompt"), "prompt");
        const negative = textOf(pipeline.prompts, accept.meta("negative"), "prompt");
        const observed = await observe(
          pipeline,
          diagnostics,
          {
            ...request({ ferret, negative }),
            latents: accept.tensor("latents_init"),
            steps: STEPS,
          },
          { abortAfterDenoise: true },
        );
        const diff = difference(observed.latents[STEPS - 1], accept.tensor(`latents.${STEPS - 1}`));
        console.log(
          `[wan-pipeline] fault ${label}: ${formatDifference(
            `latents.${STEPS - 1}`,
            diff,
          )} (${(diff.ratio / lastBand).toFixed(1)} × the band)`,
        );
        assert(
          diff.ratio > FAULT_MARGIN * lastBand,
          `${label}: ratio ${diff.ratio} does not exceed band ${lastBand} × ${FAULT_MARGIN}`,
        );
      });
    }

    // The 2 i8 presets (ADR 0120 step 5 — reference row / practical row). Each preset's pipeline is built on the same
    // device (construction builds no Session, so even next to the `f16` preset's pipeline the DiT / VAE allocations do
    // not overlap — a stage's Session exists only inside generate).
    for (const quant of SEAT_QUANTS) {
      const id = seatCaseId(quant, SEED_CASE.id);
      it(`${id} (${quant} preset, seed path, sha256 environment row)`, async () => {
        const gpu = assigned(gpuContext, "gpuContext");
        try {
          await using seatPipeline = await loadPipeline(gpu, diagnostics, "precomputed", quant);
          await seedPath(seatPipeline, id, quant);
        } finally {
          // Wait for the release of this preset's stage allocations before the next preset allocates (the B570's
          // `destroy()` lag).
          await settleReleases(gpu);
        }
      });
    }
  },
);

describe.skipIf(!GPU_TEXT_PRESENT || !GPU_AVAILABLE)(
  "Wan end-to-end 2 steps GPU path (real GPU): the umT5 text stage is torn down before the DiT and VAE stages are built, the fixed prompt and " +
    "the free prompt complete with 0 non-finite values, stage times and the VRAM at switches, sha256 environment rows",
  () => {
    let deviceLost: string | undefined;
    // One device and one GPU-path pipeline shared by the its below (built before them, disposed after them all).
    let gpuContext: GpuContext | undefined;
    let gpuTextPipeline: Pipeline | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
      // Construction builds no Session (umT5's weights are read in the text stage on every generate — ADR 0119 decision
      // 11).
      gpuTextPipeline = await loadPipeline(gpuContext, diagnostics, "gpu", F16_QUANT);
    });
    afterAll(async () => {
      if (gpuContext === undefined) return;
      try {
        await gpuTextPipeline?.dispose();
      } finally {
        await settleReleases(gpuContext);
        gpuContext.destroy();
      }
    });

    for (const spec of GPU_TEXT_CASES) {
      it(spec.id, async () => {
        const gpu = assigned(gpuContext, "gpuContext");
        const pipeline = assigned(gpuTextPipeline, "gpuTextPipeline");
        const prompt =
          "asset" in spec.prompt
            ? textOf(pipeline.prompts, spec.prompt.asset, "prompt")
            : spec.prompt.text;
        let settlement: ReferenceSettlement | undefined;
        try {
          await runRecordedCase(results, { id: spec.id }, async () => {
            const observed = await observe(pipeline, diagnostics, {
              prompt,
              seed: spec.seed,
              steps: STEPS,
            });
            assert("video" in observed, "generate did not run to the end");
            assertEquals(observed.latents.length, STEPS, "number of denoise-steps");
            // The text stage is torn down before the DiT stage is built (order of the stage boundaries — the VRAM marks
            // are taken in the same order).
            assertEquals(
              observed.drm.marks.slice(0, 3).map(({ label }) => label),
              ["text_encoder start", "text_encoder end", "transformer start"],
              "text stage → DiT stage order",
            );
            assert(
              observed.diagnostics.has("text_encoder"),
              "the diagnostics of the umT5 run did not arrive",
            );
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [DEFAULT_FRAMES, 480, 832]);
            const nonFinite = video.data.reduce(
              (count, value) => count + (Number.isFinite(value) ? 0 : 1),
              0,
            );
            const notes = [
              "textEncoder: gpu (umT5 i8, f32 activations)",
              `prompt ${JSON.stringify(prompt.trim().slice(0, 80))}`,
              `non-finite ${nonFinite}`,
              ...formatObserved(observed),
            ];
            console.log(`[wan-pipeline] ${spec.id}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, `${spec.id}: non-finite`);
            assertEquals(deviceLost, undefined, "device lost");
            const outcome = await settleOrObserve(references, results, {
              id: spec.id,
              artifact: `${spec.id}.rgb`,
              bytes: rgbBytes(video),
            });
            settlement = outcome.settlement;
            return { ...outcome.fields, note: notes.join(" / ") };
          });
        } finally {
          // Wait for the release of this case's stage allocations before the next case allocates (the B570's
          // `destroy()` lag).
          await settleReleases(gpu);
        }
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(spec.id, settlement, references));
        }
      });
    }
  },
);

/** Contact sheet (`rows × cols` cells, each a frame shrunk to 1/`factor` by box averaging). */
const contactSheet = (
  video: GeneratedVideo,
  rows: number,
  cols: number,
  factor: number,
): {
  readonly rgba: Uint8ClampedArray<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
} => {
  const cellWidth = video.width / factor;
  const cellHeight = video.height / factor;
  const width = cellWidth * cols;
  const height = cellHeight * rows;
  const rgba = new Uint8ClampedArray(width * height * 4);
  const cells = rows * cols;
  for (let cell = 0; cell < cells; cell += 1) {
    // Pick evenly spaced frames, including the first and the last (32 cells out of 33 / 81 frames).
    const frame = Math.round((cell * (video.frames - 1)) / (cells - 1));
    const source = frameToRgba(video, frame);
    const left = (cell % cols) * cellWidth;
    const top = Math.floor(cell / cols) * cellHeight;
    for (let y = 0; y < cellHeight; y += 1) {
      for (let x = 0; x < cellWidth; x += 1) {
        const at = ((top + y) * width + left + x) * 4;
        for (let channel = 0; channel < 3; channel += 1) {
          let sum = 0;
          for (let dy = 0; dy < factor; dy += 1) {
            for (let dx = 0; dx < factor; dx += 1) {
              sum += source[((y * factor + dy) * video.width + x * factor + dx) * 4 + channel];
            }
          }
          rgba[at + channel] = Math.round(sum / (factor * factor));
        }
        rgba[at + 3] = 255;
      }
    }
  }
  return { rgba, width, height };
};

describe.skipIf(!FULL_PIPELINE || !ANY_PRESENT || !GPU_AVAILABLE)(
  "Wan end-to-end 50 steps (real GPU, opt-in KARUME_WAN_FULL_PIPELINE=1): 33 / 81 frames of the default settings (the `f16` preset and the practical preset) " +
    "complete with 0 non-finite values, times and the VRAM at stage switches, PNG of every frame + contact sheet",
  () => {
    let deviceLost: string | undefined;
    // One device shared by the its below (acquired before them, released after them all).
    let gpuContext: GpuContext | undefined;
    const diagnostics = new Map<RunComponent, SessionDiagnostics>();
    beforeAll(async () => {
      gpuContext = verifyRunningGpu(
        await acquireTestGpu({
          onDeviceLost: (info) => {
            deviceLost = `${info.reason}: ${info.message}`;
          },
        }),
      );
    });
    afterAll(async () => {
      if (gpuContext === undefined) return;
      await settleReleases(gpuContext);
      gpuContext.destroy();
    });

    // Build each case's preset pipeline (construction builds no Session — the `f16` preset and the practical preset
    // alternate in order of allocation size).
    for (const spec of FULL_CASES) {
      it(spec.id, async () => {
        const gpu = assigned(gpuContext, "gpuContext");
        const frames = spec.frames ?? DEFAULT_FRAMES;
        let settlement: ReferenceSettlement | undefined;
        try {
          await using pipeline = await loadPipeline(gpu, diagnostics, "precomputed", spec.quant);
          await runRecordedCase(fullResults, { id: spec.id }, async () => {
            const observed = await observe(pipeline, diagnostics, {
              prompt: textOf(pipeline.prompts, spec.prompt, "prompt"),
              seed: spec.seed,
              frames: spec.frames,
            });
            assert("video" in observed, "generate did not run to the end");
            assertEquals(
              observed.latents.length,
              FULL_STEPS,
              "number of denoise-steps (the manifest default)",
            );
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [frames, 480, 832]);
            const nonFinite = video.data.reduce(
              (count, value) => count + (Number.isFinite(value) ? 0 : 1),
              0,
            );
            const notes = [
              await rowClassOf(spec.quant),
              `non-finite ${nonFinite}`,
              ...formatObserved(observed),
            ];
            console.log(`[wan-pipeline] ${spec.id}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, "non-finite");
            assertEquals(deviceLost, undefined, "device lost");
            for (let frame = 0; frame < video.frames; frame += 1) {
              await writeFile(
                fullResults.artifact(`${spec.id}-frame-${String(frame).padStart(2, "0")}.png`),
                await encodePng(frameToRgba(video, frame), video.width, video.height),
              );
            }
            const sheet = contactSheet(video, 4, 8, 4);
            await writeFile(
              fullResults.artifact(`${spec.id}-sheet.png`),
              await encodePng(sheet.rgba, sheet.width, sheet.height),
            );
            console.log(`[wan-pipeline] PNG: ${fullResults.dir.pathname}`);
            const outcome = await settleOrObserve(references, fullResults, {
              id: spec.id,
              artifact: `${spec.id}.rgb`,
              bytes: rgbBytes(video),
            });
            settlement = outcome.settlement;
            return { ...outcome.fields, note: notes.join(" / ") };
          });
        } finally {
          // Wait for the release of this case's stage allocations before the next case allocates (the B570's
          // `destroy()` lag).
          await settleReleases(gpu);
        }
        if (settlement?.check.status === "fail") {
          throw new Error(referenceMismatchMessage(spec.id, settlement, references));
        }
      });
    }
  },
);

/**
 * The cases to register (only those that can run — the precomputed path needs the distribution and the few-step
 * references, the GPU path the distribution and the umT5 mirror. The GPU path has no CPU reference, so it does not
 * depend on the few-step references).
 */
const CASE_IDS = [
  ...(ANY_PRESENT
    ? [
        ...CASES.map(({ name }) => `2step-${name}`),
        SEED_CASE.id,
        ...SEAT_QUANTS.map((quant) => seatCaseId(quant, SEED_CASE.id)),
        ...(FULL_PIPELINE ? FULL_CASES.map(({ id }) => id) : []),
      ]
    : []),
  ...(GPU_TEXT_PRESENT ? GPU_TEXT_CASES.map(({ id }) => id) : []),
];
const RUNNABLE = (ANY_PRESENT || GPU_TEXT_PRESENT) && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });

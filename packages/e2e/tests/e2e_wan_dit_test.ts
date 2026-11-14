/**
 * **Port check** of the Wan2.1 DiT (S-shaped graph) (real GPU; ADR 0118 step 2 — decisions 3 / 8).
 *
 * The exported IR (the f16 transformer of the local distribution mirror `models/karume-wan2.1/` — opened through its
 * manifest, ADR 0139 decision 1) runs on the karume runtime, and the latents after the host unpatchify
 * (`src/wan/dit-tokens.ts`) are compared with one forward of upstream's **plain** diffusers `WanTransformer3DModel`
 * (CPU f32, the same f16-rounded weights). The `Pipeline` e2e (`e2e_wan_pipeline_test.ts`) checks the whole
 * distribution path; this file checks the DiT's forward on its own up to full size, so it opens the transformer
 * container directly.
 *
 * The goldens are in the test-asset root (`goldens/wan2.1-t2v-1.3b-{f16,i8}-dyn/transformer/` — ADR 0139 decision 2;
 * written by `cd tools/export-recipes && uv run --group wan --inexact python -m wan.export_dit`, whose docstring
 * holds their content and the case table). Two per case:
 *
 * - `io.<case>` — the graph inputs (`input.*`) and the torch CPU output after patching (`output.0`)
 * - `reference.<case>` — the input latents, the timestep, the output (`[1,16,F,H,W]`) and each block's output of
 *   upstream's plain forward
 *
 * The patched eager run is **bit-identical** to upstream in every case (the `[eager]` lines of the export script — the
 * patch embedding turned into a Linear also matched on this machine's torch CPU), so the difference seen here is
 * exactly the difference between "the IR run on the GPU" and "upstream".
 *
 * ## Bands (decision 8; the addendum 2026-10-02 "separate the band decision from the acceptance")
 *
 * The metric is the per-case **ratio** = max absolute difference ÷ the reference's max absolute value (the form of
 * decision 8's guideline). {@link DIT_RATIO_BAND} is set at about 5 × the worst ratio of the 6 `band` cases and
 * accepted on the 3 `accept` cases (unseen inputs with seeds different from `band`). The 2 `growth` cases (S = 768)
 * record the growth with S and are not part of the band check (the full-size bands are derived independently per S by
 * steps 3 / 8 — decision 8). The check also requires that the 4 fault injections (swapped RoPE h / w, swapped
 * unpatchify order, the timestep's cos / sin reversed, the timestep off by 1) fall outside the band on the acceptance
 * cases.
 *
 * MUST: do not change the `band` cases after seeing the `accept` results (that would break the independence of the band
 * decision and the acceptance — the canonical cases are `CASES` of `wan/export_dit.py`). If the acceptance falls
 * outside the band, investigate the cause instead of widening the band.
 *
 * ## Full size (ADR 0118 steps 3 / 8 — 832×480, S = 14,040 for 33 frames and S = 32,760 for 81 frames)
 *
 * One full-size forward is checked by 3 tests. Unlike S = 192, the metric is the **normalized ratio** (the GPU error ÷
 * the error of the CPU f32 reference itself — {@link normalizedRatio}), and the band {@link DIT_FULL_NORMALIZED_BAND}
 * is a table per S. No S's band brings in another S's band: each is decided independently on that S's 6 `full-band`
 * cases and accepted on the 2 `full-accept` cases (other seeds) (decision 8's extrapolation). The error baseline is
 * **upstream run with f64 activations too** (`output.f64` of `reference.<case>` — the weights are the same f16
 * rounding).
 *
 * - Comparison in measuring mode (`gpuTiming`): the Session is rebuilt per S; the number of row blocks (the expected
 *   value derived from the device's binding limit — {@link expectedRowBlocks}) and the maximum GPU time per submit ≤
 *   {@link SUBMIT_GPU_LIMIT_MS} (including the first run's chunks issued before measured evidence existed —
 *   `ChunkBudgetStats.submitGpuTime` of the runtime) are checks, and it also checks that the 4 fault injections fall
 *   outside the band on the acceptance cases (the timestep off by 1 at least {@link SUBTLE_FAULT_MARGIN} × the band).
 *   The band check uses the max-based r; the p99.99-based r ({@link quantileRatio}) is only recorded (to see the
 *   weakness of the metric r — the maximum is decided by a single element of the tail — through the distribution; ADR
 *   0118 addendum 2026-10-02 "result of step 3 — the band's floor and the weakness of the metric r").
 * - Normal mode (no measurement): the time (wall clock) is recorded per S. In measuring mode 1 dispatch = 1 pass, so
 *   the time can differ. The check is that the output equals the measuring-mode output of the same case as Uint32
 *   (chunk boundaries do not move values — {@link measuredTokens}).
 *
 * ## i8 series (ADR 0120 steps 3 / 4 — the mirror's int8 transformer and `goldens/wan2.1-t2v-1.3b-i8-dyn/transformer/`)
 *
 * The series the same recipe writes with `--dtype i8` (307 linears int8 per-channel, bias / norm upstream's f32) is
 * checked by the same gates, with the int8 container and the int8 goldens in place of the f16 ones.
 *
 * - **r check of the reference partner `f16+dit8` (empty `session`)** (step 3): the same implementation as the f16
 *   preset's full size ({@link registerNormalizedRatioGate}), run on the S = 192 group (9 cases — including the 1 S =
 *   128 acceptance case) and on S = 14,040 (8 cases, measuring mode). Every case of the i8 series has an f64 reference
 *   (`output.f64` — values taken with the i8 fake-quant weights), so the S = 192 group is also checked by r. The band
 *   per group is the worst of the 6 deciding cases × 5 ({@link I8_NORMALIZED_BAND} — while not yet derived, it prints r
 *   and the band candidate and goes red). The fault injections are the same 4 as for the f16 preset plus 1 specific to
 *   i8 (the scale of one linear doubled — {@link scaledScaleContainer}). There is no S = 32,760 golden (ADR 0120
 *   decision 3). The io inputs are byte-identical to the f16 series, so the case names are shared.
 * - **Time and allocation of the practical preset `f16+dit8-a8-attn8-s16`** (step 4 — a record + the 1-submit check of
 *   measuring mode): the bundle is taken from the distribution manifest's preset (the manifest is canonical for the
 *   bundle — ADR 0110 decision 1). The inputs are the f16 series' `io.*` (S = 32,760 exists only in the f16 series;
 *   the inputs do not depend on the preset). The numeric check belongs to the self A/B check (`e2e_wan_ab_test.ts`) and
 *   is not done here.
 *
 * Without the mirror, the goldens a case reads or a GPU the case SKIPs explicitly, naming what is missing (ADR 0005). A
 * golden set present only in part is a FAIL, not a SKIP (the completeness tests below).
 *
 * ## Default run vs opt-in upstream parity
 *
 * Every band above (S = 192 ratio, full-size r, i8 r) and the fault injections that prove a band is
 * live compare against the diffusers goldens, so those cases are opt-in behind
 * `KARUME_UPSTREAM_PARITY=1` (helpers/upstream-parity.ts). The default run keeps the checks that
 * need no tolerance against upstream: asset completeness, the GPU-free row-block table, host
 * patchify / RoPE bit identity, per-forward finiteness and fusion counts, the full-size row-block
 * counts and the "1 submit <= 1 s" timing gate (per S, per series), and the Uint32 equality between
 * the measured and the normal mode.
 */

import { isNotFound, readFile, statSync } from "@karume/test-support/fs";
import { afterAll, beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  type BoundContainer,
  type FusionCounts,
  type GpuContext,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
  wanTokenWidth,
} from "../../models/src/wan/dit-tokens.ts";
import {
  parseWanRopeBase,
  type WanRopeBase,
  wanRopeTables,
} from "../../models/src/wan/dit-rope.ts";
import { timestepsProj } from "../../models/src/wan/dit-timestep.ts";
import { quantOf } from "./helpers/ab-gate.ts";
import { effectiveSessionOptions } from "../../models/tests/helpers/census-table.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import { goldenDir, TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import { settleReleases } from "../../models/tests/helpers/settle-releases.ts";
import { timestampUnitNs } from "../../models/tests/helpers/timestamp-unit.ts";
import {
  openWanComponent,
  WAN_ASSEMBLE_COMMAND,
  WAN_DIST_MODEL,
  WAN_DIST_ROOT,
} from "../../models/tests/helpers/wan-distribution.ts";
import {
  assertAdapterMatchesEnvironment,
  ENVIRONMENT,
} from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";

/**
 * Tolerance on the difference between one DiT forward on the GPU (B570, f16 preset) and upstream (CPU f32) (ratio = max
 * absolute difference ÷ the reference's max absolute value, latents after unpatchify).
 *
 * Measured (2026-10-02, `deno-intel-graphics-bmg-g21`, a plain comparison with `atol = rtol = 0`):
 *
 * | Role   | Case                  | Grid F'·H'·W' | Max abs diff | Reference max abs | Ratio (io)   | Ratio (TS proj)  |
 * | ------ | --------------------- | ------------- | ------------ | ----------------- | ------------ | ---------------- |
 * | band   | `band-s00192-t0999`   | 3·8·8         | 2.694e-5     | 4.024             | 6.70e-6      | 9.30e-6          |
 * | band   | `band-s00192-t0750`   | 2·8·12        | 5.460e-5     | 5.468             | 9.98e-6      | 1.20e-5          |
 * | band   | `band-s00192-t0500`   | 3·8·8         | 6.363e-5     | 3.569             | 1.78e-5      | 1.67e-5          |
 * | band   | `band-s00192-t0250`   | 1·8·24        | 2.694e-5     | 5.321             | 5.06e-6      | 4.70e-6          |
 * | band   | `band-s00192-t0600`   | 2·8·12        | 5.406e-5     | 5.594             | 9.66e-6      | 1.03e-5          |
 * | band   | `band-s00192-t0113`   | 3·8·8         | 8.273e-5     | 4.321             | **1.91e-5**  | 1.62e-5          |
 * | accept | `accept-s00192-t0600` | 2·8·12        | 3.247e-4     | 5.242             | 6.19e-5      | 4.81e-5          |
 * | accept | `accept-s00128-t0030` | 4·4·8         | 3.101e-5     | 4.784             | 6.48e-6      | 5.97e-6          |
 * | accept | `accept-s00192-t0400` | 2·12·8        | 6.151e-5     | 6.234             | 9.87e-6      | 9.87e-6          |
 * | growth | `growth-s00768-t0999` | 3·16·16       | 5.892e-5     | 4.426             | 1.33e-5      | 1.30e-5          |
 * | growth | `growth-s00768-t0500` | 3·16·16       | 4.804e-5     | 4.874             | 9.86e-6      | 8.98e-6          |
 *
 * "Ratio (io)" is the value with the golden's torch `timesteps_proj` fed in (the max-abs-diff column is from it too),
 * "Ratio (TS proj)" the value with the host's `timestepsProj` fed in.
 *
 * Ratios of the fault injections on the acceptance cases (RoPE h / w, unpatchify order, cos / sin reversed, **the
 * timestep off by 1**): `accept-s00192-t0600` 4.03e-1 / 1.28 / 2.88e-1 / **9.18e-3**, `accept-s00128-t0030` 1.06e-1 /
 * 1.43 / 3.95e-1 / **6.91e-3**, `accept-s00192-t0400` 2.48e-1 / 1.32 / 2.79e-1 / **1.93e-3**.
 *
 * The band 9.6e-5 is about 5.0 × the worst band-case ratio 1.915e-5 (io of `band-s00192-t0113`). The worst acceptance
 * value 6.19e-5 is 0.64 × the band. The smallest subtle fault (the timestep off by 1) 1.93e-3 is about 20 × the band,
 * and the 3 O(1) faults land 3–4 orders of magnitude above it.
 *
 * Why a ratio rather than an absolute difference: the old band (absolute atol 3.2e-4 = the worst 6.36e-5 of the then 4
 * deciding cases × 5) was exceeded by an unseen validation case (today's `accept-s00192-t0600`) at 3.247e-4. What
 * exceeded it was 1 element out of 12,288, and the ratio was 6.2e-5. The max absolute difference is decided by a
 * single tail element and moves with the reference's range (3.6–6.2), so the ratio divided by the range is more
 * consistent across inputs. Moreover, deciding cases had been added back then after seeing the acceptance results,
 * which also broke the independence of decision and acceptance.
 *
 * Why not a single deciding case (measured): a band decided on one t = 999 case was exceeded 1.6 × by an unseen t =
 * 500 case. Not a porting error but input sensitivity, on 2 grounds:
 * ① the patched eager run is bit-identical to upstream (the GPU vs eager difference = the GPU vs upstream difference —
 *    the `[eager]` lines of the export script).
 * ② even in the CPU eager run, a mid-range timestep amplifies the 1e-5-order wobble of `timesteps_proj` about 4 × in
 *    the output (0.2 × at t = 999). The GPU's rounding difference is amplified the same way.
 * So the deciding cases became 6 cases with timesteps spread over the whole range of the reference's schedule (999 →
 * 60), with different seeds and grids as well.
 *
 * Per-layer trend (measurement graph; max absolute difference / the reference's max absolute value of each block's
 * output; 11 cases × 30 blocks): it goes up and down within 6e-6 to 1.6e-4 and does not grow monotonically with depth
 * (per-case maxima such as 1.54e-4 at layer 18 of `band-s00192-t0500` and 1.25e-4 at layer 25 of
 * `growth-s00768-t0999`; most return to the 1e-5 order in the later layers). Going from S = 192 to 768 does not grow
 * the ratio of the final output either.
 */
const DIT_RATIO_BAND = 9.6e-5;

/** Patch of Wan2.1 T2V 1.3B (`(1,2,2)`, 16 latent channels — the transformer's config). */
const WAN_GEOMETRY: WanPatchGeometry = {
  channels: 16,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

/** The series the f16 goldens were exported with (their folder under `goldens/` of the test-asset root). */
const SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn";
/** The component: the manifest's weights key, which is also the container's graph name (container-v1 §2.1). */
const COMPONENT = "transformer";
const GOLDEN_DIR = new URL(`${COMPONENT}/`, goldenDir(SERIES_NAME));
const GRAPH = COMPONENT;
/** Name of the asset (`ROPE_BASE_ASSET` of `wan/export_dit.py`). */
const ROPE_BASE_ASSET = "rope_base";

/** The command that fetches the goldens into the local test-asset root (ADR 0139 decision 3). */
const FETCH = "node tools/test-assets/fetch.ts";
/**
 * A complete fetch of the pinned test-asset dataset makes a missing golden a FAIL (the completeness tests below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;

type CaseRole = "band" | "accept" | "growth";

/**
 * The cases that must have been generated. **Pinned here rather than enumerated** (so that tests do not silently
 * disappear in an environment where only part of the generation ran). The canonical list is `CASES` of
 * `wan/export_dit.py`.
 */
const CASES: readonly { readonly name: string; readonly role: CaseRole }[] = [
  { name: "band-s00192-t0999", role: "band" },
  { name: "band-s00192-t0750", role: "band" },
  { name: "band-s00192-t0500", role: "band" },
  { name: "band-s00192-t0250", role: "band" },
  { name: "band-s00192-t0600", role: "band" },
  { name: "band-s00192-t0113", role: "band" },
  { name: "accept-s00192-t0600", role: "accept" },
  { name: "accept-s00128-t0030", role: "accept" },
  { name: "accept-s00192-t0400", role: "accept" },
  { name: "growth-s00768-t0999", role: "growth" },
  { name: "growth-s00768-t0500", role: "growth" },
];

type FullRole = "full-band" | "full-accept";

/**
 * The full-size S (latents `[16,F',60,104]` → S = F'·30·52): 14,040 for 33 frames (F' = 9, ADR 0118 step 3) and 32,760
 * for 81 frames (F' = 21, step 8).
 */
type FullTokens = 14040 | 32760;

/**
 * The order of S in the GPU tests. The S = 32,760 with the largest allocation comes first (for the same reason the full
 * size comes before S = 192 — on the B570 what is left of an earlier allocation can push a later large one into OOM,
 * so the larger one runs on an empty device).
 */
const FULL_TOKENS_ORDER: readonly FullTokens[] = [32760, 14040];

/**
 * The full-size cases. The canonical list is `CASES` of `wan/export_dit.py`. Per S, the 6 `full-band` cases are the
 * deciding cases of that S's row of {@link DIT_FULL_NORMALIZED_BAND}, with timesteps spread over 999 (2 seeds) / 750 /
 * 500 / 250 / 113, and the 2 `full-accept` cases are for acceptance (their seeds differ from every step 2 case and
 * between deciding and acceptance cases). As in {@link FULL_TOKENS_ORDER}, S = 32,760 comes first.
 *
 * MUST: do not add or remove deciding cases after seeing the acceptance results (that would break the independence of
 * the band decision and the acceptance).
 */
const FULL_CASES: readonly {
  readonly name: string;
  readonly tokens: FullTokens;
  readonly role: FullRole;
}[] = [
  { name: "full-band-s32760-t0999", tokens: 32760, role: "full-band" },
  { name: "full-band-s32760-t0999-2", tokens: 32760, role: "full-band" },
  { name: "full-band-s32760-t0750", tokens: 32760, role: "full-band" },
  { name: "full-band-s32760-t0500", tokens: 32760, role: "full-band" },
  { name: "full-band-s32760-t0250", tokens: 32760, role: "full-band" },
  { name: "full-band-s32760-t0113", tokens: 32760, role: "full-band" },
  { name: "full-accept-s32760-t0999", tokens: 32760, role: "full-accept" },
  { name: "full-accept-s32760-t0600", tokens: 32760, role: "full-accept" },
  { name: "full-band-s14040-t0999", tokens: 14040, role: "full-band" },
  { name: "full-band-s14040-t0999-2", tokens: 14040, role: "full-band" },
  { name: "full-band-s14040-t0750", tokens: 14040, role: "full-band" },
  { name: "full-band-s14040-t0500", tokens: 14040, role: "full-band" },
  { name: "full-band-s14040-t0250", tokens: 14040, role: "full-band" },
  { name: "full-band-s14040-t0113", tokens: 14040, role: "full-band" },
  { name: "full-accept-s14040-t0999", tokens: 14040, role: "full-accept" },
  { name: "full-accept-s14040-t0600", tokens: 14040, role: "full-accept" },
];

/**
 * Cases per S (in the order of {@link FULL_TOKENS_ORDER}). `normalModeCase` is the one case run in normal mode (the
 * first of each S).
 */
const FULL_CASES_BY_TOKENS = FULL_TOKENS_ORDER.map((tokens) => {
  const cases = FULL_CASES.filter((spec) => spec.tokens === tokens);
  return { tokens, cases, normalModeCase: cases[0].name };
});

/**
 * The f64 reference held by `reference.<case>` of the full-size cases (the output of upstream's plain forward run with
 * f64 activations too, rounded to f32 — `REFERENCE_F64_KEY` of `wan/export_dit.py`).
 *
 * The difference of the rounding to f32 is at most 2⁻²⁴·|x| per element, at most 6e-8 as a ratio. How much it moves the
 * normalization's denominator (the ratio of the CPU f32 reference against f64 — {@link referenceErrorOf}) is decided
 * by the smallest denominator (measured values in the table of {@link DIT_FULL_NORMALIZED_BAND} and in results.json):
 *
 * - deciding cases (`full-band`) only: below 2% at the S = 14,040 minimum 3.72e-6 (`t0250`), about 2.9% at the S =
 *   32,760 minimum 2.07e-6 (`t0999-2`)
 * - including acceptance: about 3.3% at the minimum 1.83e-6 (`full-accept-s14040-t0600`)
 *
 * A shift of a few % in r does not change the direction of a verdict (the verdict closest to the band is the timestep
 * off by 1 of S = 32,760 t0600 at 2.65 × the band; that case's denominator is 6.77e-6, so the shift is below 1%).
 */
const REFERENCE_F64_KEY = "output.f64";

/**
 * Tolerance on the error of one full-size forward on the GPU (B570, f16 preset) (a table per S). The metric is the
 * **normalized ratio** ({@link normalizedRatio}):
 *
 *   r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|)
 *
 * = "how many times the error that the CPU f32 reference (upstream's plain forward, CPU torch f32) itself makes on the
 * same input the GPU error is". f64 is upstream run with f64 activations too (the weights are the same f16 rounding —
 * `output.f64`), CPU f32 is `output`. No S's band brings in S = 192 or another S's band: each is decided independently
 * as the worst r of that S's 6 `full-band` cases × 5 (ADR 0118 decision 8's extrapolation).
 *
 * Measured at S = 32,760 (2026-10-03, `deno-intel-graphics-bmg-g21`, ADR 0118 step 8). The band formula (worst r of
 * the 6 `full-band` cases × 5) was fixed by an ADR addendum before the measurement; the `full-accept` r came out in
 * the same run but was not used to decide the band:
 *
 * | Case                       | GPU / f64 | CPU f32 / f64 | r        |
 * | -------------------------- | --------- | ------------- | -------- |
 * | `full-band-s32760-t0999`   | 4.81e-5   | 2.32e-6       | **20.8** |
 * | `full-band-s32760-t0999-2` | 2.53e-5   | 2.07e-6       | 12.2     |
 * | `full-band-s32760-t0750`   | 5.70e-5   | 9.22e-6       | 6.18     |
 * | `full-band-s32760-t0500`   | 1.84e-5   | 4.48e-6       | 4.11     |
 * | `full-band-s32760-t0250`   | 2.30e-5   | 4.14e-6       | 5.56     |
 * | `full-band-s32760-t0113`   | 3.10e-5   | 7.86e-6       | 3.94     |
 *
 * Band 104 = the worst 20.8 (`full-band-s32760-t0999`) × 5. The 2 acceptance cases have r 13.1 (t0999) / 4.89 (t0600),
 * inside the band. The fault injections' r (RoPE h / w, unpatchify order, cos / sin reversed, the timestep off by 1)
 * are 5.65e4 / 3.44e5 / 1.72e5 / 1.91e4 at t0999 and 3.07e4 / 2.12e5 / 2.12e4 / **276** at t0600 (2.65 × the band ≥
 * {@link SUBTLE_FAULT_MARGIN}). All 4 are outside the band. The worst r being larger than at S = 14,040 (14.9 → 20.8)
 * is due to the one t = 999 case; the other 5 are similar (3.9–12.2).
 *
 * Measured at S = 14,040 (2026-10-02, `deno-intel-graphics-bmg-g21`, a plain comparison with `atol = rtol = 0`). The
 * band was decided from the 6 deciding cases alone, before the 2 acceptance cases (seeds 777006 / 777007) were
 * generated:
 *
 * | Case                       | GPU / f64  | CPU f32 / f64 | GPU / CPU f32 | r          |
 * | -------------------------- | ---------- | ------------- | ------------- | ---------- |
 * | `full-band-s14040-t0999`   | 2.120e-3   | 1.421e-4      | 1.978e-3      | **14.9**   |
 * | `full-band-s14040-t0999-2` | 4.053e-5   | 5.562e-6      | 3.908e-5      | 7.29       |
 * | `full-band-s14040-t0750`   | 6.837e-5   | 1.332e-5      | 7.142e-5      | 5.13       |
 * | `full-band-s14040-t0500`   | 1.673e-5   | 3.769e-6      | 1.596e-5      | 4.44       |
 * | `full-band-s14040-t0250`   | 9.844e-6   | 3.724e-6      | 9.486e-6      | 2.64       |
 * | `full-band-s14040-t0113`   | 8.921e-6   | 5.592e-6      | 9.277e-6      | 1.60       |
 *
 * Each column is "max absolute difference ÷ the reference's max absolute value" (the references of the left 3 columns
 * are f64 / f64 / CPU f32 respectively). Band 75 = the worst 14.92 (`full-band-s14040-t0999`) × 5 = 74.6, rounded up
 * to 2 significant digits.
 *
 * Why normalize: the ratio itself (GPU / f64) moves 240 × across inputs (8.9e-6 at t = 113 to 2.1e-3 at t = 999; 52 ×
 * by seed even at the same t = 999). What moves is the input's amplification of rounding, and the CPU f32 reference
 * moves away from f64 by just as much (on the same input it applies to any f32 implementation). Deciding the band as
 * the worst ratio × 5 gives 1.1e-2, which let even a subtle fault (the timestep off by 1 — ratio 1.77e-3 at t = 600)
 * inside the band. Dividing by the CPU f32 error cancels the amplification, and r stays within 1.6–15 (the 240 × spread
 * of the ratio becomes 9 ×).
 *
 * Why r exceeds 1 (the attribution of 2026-10-02 — a precision difference, not an error): the shape of the per-layer
 * ratio growth is the same on the GPU and on CPU f32, with the peak at the same layer. In the per-op local errors (the
 * outputs of the ops of blocks 0 / 28, with the GPU values as inputs, compared with f64; `full-band-s14040-t0999` /
 * `t0113`), linear (GEMM) is 4–7 × CPU f32 and the FFN (down with K = 8960) 11–14 ×. Reproducing the GEMM's K reduction
 * (K ascending, a single f32 accumulator — src/kernels/gemm.ts) on the CPU as sequential fma is bit-identical to the
 * GPU output (to_v and to_out of blocks 0 / 28, 512 rows). Norms and residuals are 1 ×, and block 28's SDPA is smaller
 * on the GPU (0.8 ×). A remedy (changing the reduction order affects the sha256 and the outputs of every series) is a
 * separate ruling.
 *
 * MUST: do not change this value, the `full-band` cases or the metric after seeing the `full-accept` results (the
 * acceptance seeds were renewed after the metric was decided — the docstring of `wan/export_dit.py`). If the
 * acceptance falls outside the band, investigate the cause instead of widening the band.
 *
 * NOTE (2026-10-02, the acceptance results — generated and run after the band was decided):
 *
 * | Case                       | GPU / f64 | CPU f32 / f64 | GPU / CPU f32 | r    | r (TS proj)     |
 * | -------------------------- | --------- | ------------- | ------------- | ---- | --------------- |
 * | `full-accept-s14040-t0999` | 2.00e-5   | 6.82e-6       | 2.49e-5       | 2.93 | 3.27            |
 * | `full-accept-s14040-t0600` | 8.45e-6   | 1.83e-6       | 8.70e-6       | 4.60 | 4.50            |
 *
 * Both inside the band. The fault injections' r (RoPE h / w, unpatchify order, cos / sin reversed, the timestep off by
 * 1) are 3.25e4 / 2.05e5 / 6.44e4 / 6.04e3 at t0999 (the timestep off by 1 at 80.5 × the band) and 3.24e4 / 6.65e5 /
 * 3.62e4 / **332** at t0600 (ratio 6.09e-4, **4.42 ×** the band). All 4 are outside the band. The smallest margin of
 * the timestep off by 1 is 4.42 ×, which satisfies {@link SUBTLE_FAULT_MARGIN} (2). Neither the band, the metric nor
 * the deciding cases were moved.
 */
const DIT_FULL_NORMALIZED_BAND: Readonly<Record<FullTokens, number>> = { 14040: 75, 32760: 104 };

/**
 * Factor by which a subtle fault (the timestep off by 1) must stay away from the band (the implementation of the
 * addendum 2026-10-02 "an injection that does not turn red is a sign that the band is too wide"). Merely falling
 * outside the band does not reveal when the margin of the band's worst × 5 and the size of the fault become the same
 * order of magnitude. The value is 2: the size of the fault itself moves nearly 3 × with the input (1.77e-3 and
 * 6.09e-4 for 2 seeds at t=600), so 2 ×, below the observed minimum of 4.42 × (acceptance t0600), 9.4 × (seed 888123 of
 * an independent validation) and 80 × (t0999), is the floor that detects a widened band. The first value of 5 was an
 * estimate not in the ADR and was lowered to 2 after the observation (disclosed — the band, the metric and the
 * deciding cases did not move).
 */
const SUBTLE_FAULT_MARGIN = 2;

/**
 * The i8 series (the weights of the reference partner `f16+dit8` / the practical preset — ADR 0120 decision 2) — the
 * folder of its goldens.
 */
const I8_SERIES_NAME = "wan2.1-t2v-1.3b-i8-dyn";
const I8_GOLDEN_DIR = new URL(`${COMPONENT}/`, goldenDir(I8_SERIES_NAME));
/**
 * Prefix of the result IDs and step names of the i8 reference-partner cases (the case names are shared with the f16
 * series, so they are separated by preset).
 */
const I8_ID_PREFIX = "f16+dit8/";

/** The int8 container's graph name (the same weights key as the f16 one). */
const i8Graph = (): string => COMPONENT;

/**
 * The i8 series' cases (a mirror of `series_cases("i8")` of `wan/export_dit.py` — the S = 192 group is the f16 series'
 * `band` / `accept` (including the 1 S = 128 acceptance case), the full size the 8 cases of S = 14,040. `growth` and S
 * = 32,760 are not taken — ADR 0120 decision 3). Seeds, timesteps and valid lengths are the same as the f16 series.
 */
const I8_SMALL_CASES: readonly GateCase[] = CASES.filter(({ role }) => role !== "growth").map(
  ({ name, role }) => ({ name, role, accept: role === "accept" }),
);
const I8_FULL_CASES: readonly GateCase[] = FULL_CASES.filter(({ tokens }) => tokens === 14040).map(
  ({ name, role }) => ({ name, role, accept: role === "full-accept" }),
);

/** The groups of the i8 reference partner (the S = 192 group includes the 1 S = 128 acceptance case). */
type I8Group = 192 | 14040;

/**
 * Band of the normalized ratio r of the reference partner `f16+dit8` (i8 series, empty `session`) (per group — ADR 0120
 * decision 3). The derivation rule is the same as the f16 preset's {@link DIT_FULL_NORMALIZED_BAND}: the worst r of the
 * group's 6 deciding cases × 5. No band is brought in from another group or from the f16 preset (ADR 0118 decision 8's
 * extrapolation). The S = 192 group is also checked by r — every case of the i8 series has an f64 reference, so the
 * metric differs from the f16 preset's S = 192 ratio band ({@link DIT_RATIO_BAND}).
 *
 * `undefined` = not yet derived: the check prints each case's r and the band candidate ({@link registerBandCandidate})
 * and **stops red** (no check without detection power is made by putting in a value that passes). After the GPU
 * measurement it is written here together with the measured table.
 *
 * MUST: do not change this value or the deciding cases after seeing the acceptance results (the same as the f16
 * preset). If the acceptance falls outside the band, investigate the cause instead of widening the band.
 *
 * Derivation (2026-10-03, B570, measuring mode (S = 14,040) / normal device (S = 192)):
 *
 * | Group   | Deciding r (t0999 / t0999-2 / t0750 / t0500 / t0250 / t0113)    | Worst | × 5   | Band | Acceptance r            |
 * | ------- | ---------------------------------------------------------------- | ----- | ----- | ---- | ----------------------- |
 * | 14,040  | 10.7 / 10.2 / 4.02 / 2.90 / 2.56 / 2.05                          | 10.7  | 53.5  | 54   | t0999 3.71 / t0600 4.64 |
 * | 192     | 2.59 / 2.83 / 3.66 / 4.12 / 3.30 + t0600 5.97                    | 5.97  | 29.9  | 30   | t0600 3.82 / t0400 3.43 |
 *
 * The scale × 2 fault injection gives r 1.66e+4 to 4.45e+4 (3 orders of magnitude above the band). The f16 preset's
 * band for the same S is 75 (S = 14,040); that the i8 t0999 r is smaller than f16's (14.9) is because the denominator
 * (the CPU f32 error) is of the same order while the numerator's maximum came out smaller — not a ranking of the
 * presets.
 */
const I8_NORMALIZED_BAND: Readonly<Record<I8Group, number | undefined>> = {
  192: 30,
  14040: 54,
};

/**
 * The linear whose per-channel scale the i8-specific fault injection doubles ({@link scaledScaleContainer}). The V of
 * the self-attention of a middle block is chosen — for q / k, the RMS norm right after them (`norm_q` / `norm_k`)
 * cancels the factor so it never reaches the values. The claim is that "the r band catches an i8 scale mix-up (axis,
 * value, binding)" (ADR 0120 acceptance, step 3).
 */
const I8_SCALE_FAULT_WEIGHT = "blocks.15.attn1.to_v.weight";
/** The scale-fault label in the it titles, the notes and the wrapping describe's title. */
const I8_SCALE_FAULT_LABEL = `scale of ${I8_SCALE_FAULT_WEIGHT} doubled`;

/**
 * The distribution mirror (the containers and the practical preset's bundle — the output directory of
 * `dist.py --pipeline wan`).
 */
const DIST_ROOT = WAN_DIST_ROOT;
/** The distribution's Wan model name (the key of the manifest's models). */
const DIST_MODEL = WAN_DIST_MODEL;
/** The practical preset (ADR 0120 decision 1). */
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";

/**
 * Inputs for the practical preset's time and allocation (one per S, the f16 series' `io.*`). The values do not affect
 * the time, so light cases whose goldens do not hold each block's output are chosen (the reference of
 * `full-band-s14040-t0999` is 2.6 GB). The S = 32,760 with the largest allocation comes first (the same reason as
 * {@link FULL_TOKENS_ORDER}).
 */
const PRACTICAL_TIMING_CASES: readonly { readonly tokens: FullTokens; readonly name: string }[] = [
  { tokens: 32760, name: "full-band-s32760-t0999" },
  { tokens: 14040, name: "full-band-s14040-t0999-2" },
];

/**
 * Upper limit on the GPU execution span of 1 submit (ADR 0118 decision 6 and the addendum 2026-10-02 — 1/5 of Linux
 * xe's `job_timeout_ms` of 5,000 ms. The check is on the maximum of each single submit, not on a window mean, and it
 * includes the first run's chunks issued before measured evidence existed).
 */
const SUBMIT_GPU_LIMIT_MS = 1000;

/** Number of attention heads of Wan2.1 T2V 1.3B (`num_attention_heads` of the transformer's config). */
const WAN_HEADS = 12;
/** Storage width of one element of the score matrix S (the default f32 storage). */
const SCORE_BYTES = 4;

/**
 * Expected number of row blocks of the full-size self-attention (ADR 0118 decision 6). Row blocks (ADR 0060) split the
 * query rows evenly into "the smallest count at which one score block S fits within the binding limit", so the count
 * can be derived independently of the runtime from the device's binding limit (`maxStorageBufferBindingSize`) alone:
 *
 *   rows per block = ⌊limit ÷ (heads × S × 4 B)⌋, count = ⌈S ÷ rows per block⌉
 *
 * On the B570 (binding limit {@link B570_STORAGE_BINDING_LIMIT}), S = 14,040 gives 5 blocks (2,808 rows each when split
 * evenly) and S = 32,760 gives 24 blocks × 1,365 rows (the table of decision 6 — the GPU-free test below reproduces it
 * with this formula).
 */
const expectedRowBlocks = (tokens: number, bindingLimit: number): number =>
  Math.ceil(tokens / Math.floor(bindingLimit / (WAN_HEADS * tokens * SCORE_BYTES)));

/** `maxStorageBufferBindingSize` of the B570 (the premise of the table of ADR 0118 decision 6). */
const B570_STORAGE_BINDING_LIMIT = 2_147_483_644;

/**
 * The fusions this IR gets at plan time (measured — `lastRunFusions`). MUST: a moved value is red
 * (a fusion silently drops out with one change of the exporter's node order, and the values stay
 * correct while only the speed changes — this is the only observation point).
 *
 * - `adaln` 0: Wan's modulation is `layer_norm → add(scale, one) → mul → add(shift)`, the same four
 *   computations as the rule, but the rule's broadcast form requires 2–3 reshapes (the modulation
 *   vectors' unsqueeze) right after the layer_norm. Wan slices the modulation vectors from the table
 *   **before** the layer_norm, so there are 0 reshapes in between and it is not grabbed (as ADR
 *   0118 decision 3 guessed; widening the rule is a separate perf item, perf-ledger K-73). 2.1 has
 *   no per-token `where` either, so the per-token form does not apply.
 * - `rmsNormRope` 60: the q and k prologues of the 30 self-attention blocks — `rms_norm` over the
 *   token row, two reshapes, then the pair-form RoPE (the complex rotation in real numbers), one
 *   dispatch each (2026-10-25).
 * - `rope` 0: those 60 pair-form RoPEs (ROPE_RULE's second exact form, the `rope` count of
 *   2026-10-24) now fold with their rms_norm.
 * - `gatedResidual` 0: 2.1's gates are broadcast vectors (no per-token `where`).
 * - `silu` 2: the two SiLUs of the time-embedding MLP (inside `time_embedder` and `act_fn`).
 */
const EXPECTED_FUSIONS: Partial<FusionCounts> = {
  adaln: 0,
  rmsNormRope: 60,
  rope: 0,
  gatedResidual: 0,
  silu: 2,
};

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

/**
 * Whether the file exists (anything other than NotFound propagates — a permission error is not read as "the asset is
 * missing").
 */
const filePresent = (url: URL): boolean => {
  try {
    return statSync(url).isFile;
  } catch (cause) {
    if (isNotFound(cause)) return false;
    throw cause;
  }
};

/**
 * The 2 goldens (`io.<case>` and `reference.<case>`). `dir` is the goldens' component folder (the f16 one by default).
 */
const caseFiles = (name: string, dir: URL = GOLDEN_DIR): readonly URL[] => [
  new URL(`io.${name}.safetensors`, dir),
  new URL(`reference.${name}.safetensors`, dir),
];

// The same script writes the full-size cases in the same run as well (an environment with only part of them FAILs in
// the completeness test).
const MIRROR_PRESENT = mirrorPresent(DIST_ROOT);
const goldensPresent = (names: readonly string[], dir: URL = GOLDEN_DIR): boolean =>
  names.every((name) => caseFiles(name, dir).every(filePresent));
const expectedFiles = [...CASES, ...FULL_CASES].flatMap(({ name }) => caseFiles(name));
const presentFiles = expectedFiles.filter(filePresent);
/** The host check reads every f16 golden (S = 192 and full size) and the container's `rope_base`. */
const ASSETS_AVAILABLE = MIRROR_PRESENT && presentFiles.length === expectedFiles.length;
const ANY_PRESENT = presentFiles.length > 0;
/** The S-form cases read the S = 192 goldens only. */
const SMALL_AVAILABLE = MIRROR_PRESENT && goldensPresent(CASES.map(({ name }) => name));
/**
 * The full-size cases read the full-size goldens only (the normal mode also reads the measured mode's output, which
 * runs on the same set).
 */
const FULL_AVAILABLE = MIRROR_PRESENT && goldensPresent(FULL_CASES.map(({ name }) => name));

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so Wan's DiT e2e is SKIPPED. Assemble it: ` +
      `${WAN_ASSEMBLE_COMMAND} (an all-SKIP is a FAIL of the gate distribution_gate_test.ts)`,
  );
} else if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] the f16 goldens under ${GOLDEN_DIR.pathname} are not all present, so the cases that read them ` +
      `are SKIPPED (${MISSING_GOLDENS})`,
  );
}

// The same script also writes the i8 series' S = 192 group and S = 14,040 group in the same run. A `--no-full` export
// (without S = 14,040) is an intermediate state for splitting the steps, so the completeness test FAILs until both are
// present. The r check runs only the groups that are complete (what is missing is reported by the completeness test's
// FAIL).
const i8FilesOf = (cases: readonly GateCase[]): readonly URL[] =>
  cases.flatMap(({ name }) => caseFiles(name, I8_GOLDEN_DIR));
const i8ExpectedFiles = i8FilesOf([...I8_SMALL_CASES, ...I8_FULL_CASES]);
const I8_ANY_PRESENT = i8ExpectedFiles.some(filePresent);
const I8_SMALL_AVAILABLE = MIRROR_PRESENT && i8FilesOf(I8_SMALL_CASES).every(filePresent);
const I8_FULL_AVAILABLE = MIRROR_PRESENT && i8FilesOf(I8_FULL_CASES).every(filePresent);

for (const [available, group] of [
  [I8_SMALL_AVAILABLE, "the S = 192 group"],
  [I8_FULL_AVAILABLE, "S = 14,040"],
] as const) {
  if (available || !MIRROR_PRESENT) continue;
  console.warn(
    `[karume] the int8 goldens of ${group} under ${I8_GOLDEN_DIR.pathname} are not all present, so the ` +
      `f16+dit8 r gate of ${group} is SKIPPED (${MISSING_GOLDENS})`,
  );
}

/**
 * The distribution manifest (canonical for the practical preset's bundle — without it the practical preset's time
 * record SKIPs).
 */
const distManifestText = await readTextIfPresent(new URL("karume.json", DIST_ROOT));
/**
 * The practical preset's time can be taken (inputs = the f16 goldens of {@link PRACTICAL_TIMING_CASES}, weights = the
 * mirror's int8 transformer, bundle = the distribution manifest).
 */
const PRACTICAL_RUNNABLE =
  MIRROR_PRESENT &&
  distManifestText !== undefined &&
  goldensPresent(PRACTICAL_TIMING_CASES.map(({ name }) => name));

/**
 * The practical preset's effective SessionOptions (the `session` of the distribution manifest's preset passed through
 * the family's acceptance table — the manifest's quant preset alone is canonical for the bundle, ADR 0110 decision 1.
 * The bundle's values are not copied into the test).
 */
const practicalSessionOptions = (): SessionOptions => {
  if (distManifestText === undefined) {
    throw new Error(
      `${DIST_ROOT.pathname}karume.json is missing (inconsistent with the skip condition)`,
    );
  }
  const where = `wan ${DIST_MODEL}/${PRACTICAL_QUANT}`;
  const declared = quantOf(parseManifest(distManifestText), DIST_MODEL, PRACTICAL_QUANT).session;
  return effectiveSessionOptions("wan", declared, {}, where);
};

const viewOf = (file: SafetensorsFile, key: string, where: string) => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`${where}: no '${key}'`);
  return view;
};

const floatsOf = (file: SafetensorsFile, key: string, where: string): Float32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "F32") throw new Error(`${where}: '${key}' is ${view.dtype}`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** The golden of one case (graph inputs; upstream's latent shape, timestep and output). */
type Golden = {
  readonly io: SafetensorsFile;
  readonly reference: SafetensorsFile;
  readonly latentShape: readonly number[];
  readonly timestep: number;
};

const loadGolden = async (name: string, dir: URL = GOLDEN_DIR): Promise<Golden> => {
  const [ioUrl, referenceUrl] = caseFiles(name, dir);
  const io = parseSafetensors(await readBuffer(ioUrl));
  const reference = parseSafetensors(await readBuffer(referenceUrl));
  const latents = viewOf(reference, "latents", name);
  const timestep = viewOf(reference, "timestep", name);
  if (timestep.dtype !== "I32" || latents.shape.length !== 5 || latents.shape[0] !== 1) {
    throw new Error(`${name}: unexpected shape of the reference's latents / timestep`);
  }
  return {
    io,
    reference,
    // The host latents are `[C,F,H,W]` without a batch axis (`src/wan/dit-tokens.ts`).
    latentShape: latents.shape.slice(1),
    timestep: new Int32Array(reference.buffer, timestep.byteOffset, 1)[0],
  };
};

/** Graph inputs (`input.*` of io — the declared names and shapes are used as they are). */
const graphInputs = (
  golden: Golden,
  inputs: readonly { readonly name: string }[],
  where: string,
): Record<string, Tensor> =>
  Object.fromEntries(
    inputs.map(({ name }) => {
      const view = viewOf(golden.io, `input.${name}`, where);
      return [
        name,
        {
          dtype: "f32",
          shape: view.shape,
          data: floatsOf(golden.io, `input.${name}`, where),
        },
      ];
    }),
  );

/** Index of the first element whose bits differ (-1 when identical). 0 when the lengths differ. */
const firstBitMismatch = (actual: Float32Array, expected: Float32Array): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

type Difference = {
  readonly maxAbs: number;
  readonly maxRel: number;
  readonly referenceMaxAbs: number;
  readonly nonFinite: number;
};

const difference = (actual: Float32Array, expected: Float32Array): Difference => {
  assertEquals(actual.length, expected.length, "element count");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMaxAbs = 0;
  let nonFinite = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const got = actual[index];
    if (!Number.isFinite(got)) nonFinite += 1;
    const error = Math.abs(got - expected[index]);
    const magnitude = Math.abs(expected[index]);
    maxAbs = Math.max(maxAbs, error);
    if (magnitude > 0) maxRel = Math.max(maxRel, error / magnitude);
    referenceMaxAbs = Math.max(referenceMaxAbs, magnitude);
  }
  return { maxAbs, maxRel, referenceMaxAbs, nonFinite };
};

/** The band metric (max absolute difference ÷ the reference's max absolute value — {@link DIT_RATIO_BAND}). */
const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/**
 * The absolute band equivalent to the ratio band for this case (band × the reference's max absolute value), put into
 * `tolerance` of results.json. The record's shape (`Measurement`) holds only an absolute band, so the same boundary as
 * the verdict is mapped to an absolute value and kept.
 */
const recordedTolerance = (diff: Difference, band = DIT_RATIO_BAND) => ({
  atol: band * diff.referenceMaxAbs,
  rtol: 0,
});

const formatDifference = (label: string, diff: Difference): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / reference max abs ` +
  `${diff.referenceMaxAbs.toFixed(3)} (ratio ${ratioOf(diff).toExponential(2)})`;

/**
 * The source of the full-size normalization's denominator = the difference between the CPU f32 reference (`output`)
 * and the f64 reference (`output.f64`). A ratio of 0 (the f32 reference bit-identical to f64) cannot divide, so it
 * fails loudly.
 */
const referenceErrorOf = (golden: Golden, name: string): Difference => {
  const diff = difference(
    floatsOf(golden.reference, "output", name),
    floatsOf(golden.reference, REFERENCE_F64_KEY, name),
  );
  if (!(ratioOf(diff) > 0)) {
    throw new Error(
      `${name}: ratio of the CPU f32 reference against f64 is ${ratioOf(diff)} (cannot normalize)`,
    );
  }
  return diff;
};

/**
 * Normalized ratio ({@link DIT_FULL_NORMALIZED_BAND}) = the GPU's ratio against f64 ÷ the CPU f32 reference's ratio
 * against f64 (`referenceRatio`).
 */
const normalizedRatio = (diff: Difference, referenceRatio: number): number =>
  ratioOf(diff) / referenceRatio;

const formatNormalized = (label: string, diff: Difference, referenceRatio: number): string =>
  `${formatDifference(label, diff)}, r ${normalizedRatio(diff, referenceRatio).toPrecision(3)}`;

/** Quantile of the p99.99-based r ({@link quantileRatio}). */
const RECORDED_QUANTILE = 0.9999;

/** q-quantile of |actual − expected| (nearest rank — the ceil(q·n)-th in ascending order). */
const absErrorQuantile = (actual: Float32Array, expected: Float32Array, q: number): number => {
  assertEquals(actual.length, expected.length, "element count");
  const errors = new Float64Array(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    errors[index] = Math.abs(actual[index] - expected[index]);
  }
  errors.sort();
  return errors[Math.max(0, Math.ceil(q * errors.length) - 1)];
};

/**
 * One record line of the p99.99-based r (= p99.99 of |GPU − f64| ÷ p99.99 of |CPU f32 reference − f64|
 * (`referenceQuantile`)) (it goes into the note of results.json).
 * **A record only**; the check stays the max-based r ({@link normalizedRatio}). The max denominator is decided by a
 * single outlier element, and r can swing on inputs with an extremely small denominator (ADR 0118 addendum 2026-10-02
 * "result of step 3 — the band's floor and the weakness of the metric r"), so this is kept as material for deciding,
 * from the distribution, whether to replace it. The division by the reference's max absolute value cancels between
 * numerator and denominator, so it is omitted.
 */
const quantileRatio = (
  label: string,
  actual: Float32Array,
  expected: Float32Array,
  referenceQuantile: number,
): string => {
  const numerator = absErrorQuantile(actual, expected, RECORDED_QUANTILE);
  const ratio = numerator / referenceQuantile;
  const parts = [
    `numerator ${numerator.toExponential(3)}`,
    `denominator ${referenceQuantile.toExponential(3)}`,
  ];
  return `${label}: p99.99-based r ${ratio.toPrecision(3)} (record only, ${parts.join(" / ")})`;
};

/** Reads the output's last dimension `(pt,ph,pw,c)` as `(c,pt,ph,pw)` (the fault injection of the unpatchify order). */
const transposeTokenAxes = (
  tokens: Float32Array,
  geometry: WanPatchGeometry,
): Float32Array<ArrayBuffer> => {
  const channels = geometry.channels;
  const width = wanTokenWidth(geometry);
  const volume = width / channels;
  const out = new Float32Array(tokens.length);
  for (let token = 0; token < tokens.length / width; token += 1) {
    for (let c = 0; c < channels; c += 1) {
      for (let v = 0; v < volume; v += 1) {
        out[token * width + v * channels + c] = tokens[token * width + c * volume + v];
      }
    }
  }
  return out;
};

/** Reads the container's asset `rope_base` (role `rope-base`) into the raw table. */
const readRopeBase = async (opened: OpenedContainer): Promise<WanRopeBase> => {
  const reader = opened.asset(ROPE_BASE_ASSET);
  assertEquals(reader.role, "rope-base", `role of asset '${ROPE_BASE_ASSET}'`);
  const bytes = await reader.read(0, reader.length);
  return parseWanRopeBase(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};

const results = openResults("wan-dit");

// Only a completely empty environment SKIPs as "not generated". If anything is present, what is missing FAILs.
// Only a root with no f16 golden that is not a complete fetch SKIPs; the mirror's completeness is the distribution
// gate's.
test.skipIf(!ANY_PRESENT && !FETCHED)(
  "Wan DiT goldens: the goldens of every case are present",
  () => {
    assertEquals(
      expectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `missing goldens (${MISSING_GOLDENS})`,
    );
  },
);

// Only a completely empty environment SKIPs as "not generated". If anything is present, what is missing FAILs (the same
// discipline as the f16 series).
test.skipIf(!I8_ANY_PRESENT && !FETCHED)(
  "Wan DiT i8 goldens: the goldens of every int8 case (9 of the S = 192 group, 8 of S = 14,040) are present",
  () => {
    assertEquals(
      i8ExpectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `missing int8 goldens (${MISSING_GOLDENS})`,
    );
  },
);

test("Wan DiT expected row blocks (no GPU): reproduces the table of ADR 0118 decision 6 (24 / 5 blocks) with the B570's binding limit", () => {
  assertEquals(
    FULL_TOKENS_ORDER.map((tokens) => expectedRowBlocks(tokens, B570_STORAGE_BINDING_LIMIT)),
    [24, 5],
  );
});

test.skipIf(!ASSETS_AVAILABLE)(
  "Wan DiT host: patchify and the RoPE tables from the asset rope_base are bit-identical to the golden inputs",
  async () => {
    const base = await readRopeBase(await openWanComponent(COMPONENT, "f16"));
    // The host reordering is bit-identical to torch at full size too (S = 14,040 / 32,760, non-square grids 9·30·52 /
    // 21·30·52).
    for (const { name } of [...CASES, ...FULL_CASES]) {
      const golden = await loadGolden(name);
      const tokens = patchifyLatents(
        floatsOf(golden.reference, "latents", name),
        golden.latentShape,
        WAN_GEOMETRY,
      );
      assertEquals(
        firstBitMismatch(tokens, floatsOf(golden.io, "input.tokens", name)),
        -1,
        `${name}: patchify`,
      );
      const tables = wanRopeTables(base, wanTokenGrid(golden.latentShape, WAN_GEOMETRY));
      assertEquals(
        firstBitMismatch(tables.cos, floatsOf(golden.io, "input.rope_cos", name)),
        -1,
        `${name}: rope_cos`,
      );
      assertEquals(
        firstBitMismatch(tables.sin, floatsOf(golden.io, "input.rope_sin", name)),
        -1,
        `${name}: rope_sin`,
      );
    }
  },
);

/**
 * The outputs produced by the measuring-mode comparison (only for the cases run in normal mode — case name → tokens
 * before unpatchify).
 *
 * The normal-mode test compares with these as Uint32. In measuring mode 1 dispatch = 1 pass, and the chunk boundaries
 * after measured evidence exists can also differ from a normal run. This directly pins that the values still do not
 * move by a single bit (chunk boundaries do not move values). The 2 modes use different devices (with or without the
 * measuring feature), so they are not loaded at once but run in turn in separate tests.
 *
 * MUST: register the measuring-mode comparison before the normal mode (Vitest runs the tests of one file in
 * registration order). A run that skipped the measuring mode (`-t` and the like) leaves the normal mode nothing
 * to compare with, so it fails instead of silently dropping the comparison. NOTE: the default measured-mode case
 * ({@link registerFullStructureGate}) fills this, so the Uint32 gate runs without the opt-in parity cases.
 */
const measuredTokens = new Map<string, Float32Array>();

/**
 * The normal-mode outputs (case name -> the last run's tokens before unpatchify). The default
 * normal-mode case fills it, and the opt-in case that holds those outputs to the full-size band
 * reads it, so the band check does not rerun the S = 32,760 forward. The same registration-order
 * MUST as {@link measuredTokens} applies.
 */
const normalModeTokens = new Map<string, Float32Array<ArrayBuffer>>();

const gib = (bytes: number): string => `${(bytes / 2 ** 30).toFixed(2)} GiB`;

/** Observation of one full-size run (time and live allocations — put into the results note). */
type FullRun = {
  readonly label: string;
  readonly wallMs: number;
  /**
   * Total GPU time of the dispatch passes (in raw timestamp units — {@link formatRun} converts to ns with
   * {@link TIMESTAMP_UNIT_NS}). undefined on a device without measurement.
   */
  readonly gpuTicks: number | undefined;
  /**
   * Allocations live right after the run = weights + that run's arena (intermediates, inputs, readback staging) + the
   * slot backing held (intermediates and inputs). Readback staging (`MAP_READ`) is included — on the B570 it sits on
   * the side not counted against the VRAM budget (measured 2026-10-02: 3 GiB of `MAP_READ` could still be allocated
   * after 9 GiB of STORAGE), so it is shown separately.
   */
  readonly liveBytes: number;
  /** Of which the weights (initializers and the params cache). */
  readonly weightBytes: number;
  /** Of which the readback staging (the sum of the graph outputs' bytes). */
  readonly stagingBytes: number;
};

const observeRun = (
  label: string,
  diagnostics: SessionDiagnostics,
  wallMs: number,
  stagingBytes: number,
): FullRun => ({
  label,
  wallMs,
  gpuTicks: diagnostics.lastRunTiming?.totalNs,
  liveBytes:
    diagnostics.weights.allocatedBytes +
    (diagnostics.lastRun?.allocatedBytes ?? 0) +
    diagnostics.planBacking.residentBytes +
    diagnostics.planBacking.inputBytes,
  weightBytes: diagnostics.weights.allocatedBytes,
  stagingBytes,
});

/** `unitNs` is the ns of one timestamp unit (when omitted, no GPU time is shown — a run without measurement). */
const formatRun = (run: FullRun, unitNs?: number): string =>
  `${run.label}: wall ${(run.wallMs / 1000).toFixed(2)} s` +
  (run.gpuTicks === undefined || unitNs === undefined
    ? ""
    : `, GPU ${((run.gpuTicks * unitNs) / 1e9).toFixed(2)} s`) +
  `, allocated ${gib(run.liveBytes)} (weights ${gib(run.weightBytes)}, readback staging ${gib(
    run.stagingBytes,
  )})`;

/**
 * Number of self-attention row blocks (from the dispatch count of ①QK in `lastRunPipelines` of the latest run). Keys
 * with a row window (`:rwa` — `attentionQkKey` of src/kernels/attention.ts) appear once per row block per layer; keys
 * without a row window are the cross-attention (N = 512 fits in one block), once per layer.
 */
const rowBlocksOf = (diagnostics: SessionDiagnostics): number => {
  const qk = (diagnostics.lastRunPipelines ?? []).filter(({ key }) =>
    key.startsWith("attention_qk:"),
  );
  const count = (windowed: boolean): number =>
    qk
      .filter(({ key }) => key.includes(":rwa") === windowed)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  const layers = count(false);
  assert(layers > 0, "no ①QK dispatch without a row window (cross-attention) at all");
  return count(true) / layers;
};

/** Record of the observed GPU time per submit, converted to ms (the check is {@link SUBMIT_GPU_LIMIT_MS}). */
const submitGpuNote = (
  diagnostics: SessionDiagnostics,
  unitNs: number,
): { readonly maxMs: number; readonly unbackedSubmits: number; readonly note: string } => {
  const budget = diagnostics.submit.chunkBudget;
  const observed = budget.submitGpuTime;
  if (observed === undefined) throw new Error("measuring mode but no submit GPU time");
  const ms = (ns: number): string => `${((ns * unitNs) / 1e6).toFixed(1)} ms`;
  const note = [
    `max GPU time per submit ${ms(observed.maxNs)} (${observed.submits} submits)`,
    `max of the ${observed.unbackedSubmits} submits before evidence ${
      observed.maxUnbackedNs === undefined ? "—" : ms(observed.maxUnbackedNs)
    }`,
    `max single dispatch ${ms(observed.maxDispatchNs)} (${observed.maxDispatchKey ?? "—"})`,
    `max window mean ${budget.maxWindowMeanMs?.toFixed(1) ?? "—"} ms, max estimate ${
      budget.maxEstimatedMs?.toFixed(1) ?? "—"
    } ms, ${budget.overBudgetChunks} over budget`,
    `${diagnostics.submit.submitCount} submits, ${diagnostics.submit.dispatchCount} dispatches`,
  ].join(", ");
  return {
    maxMs: (observed.maxNs * unitNs) / 1e6,
    unbackedSubmits: observed.unbackedSubmits,
    note,
  };
};

/** Sum of the graph outputs' bytes (the size of the readback staging). */
const outputBytes = (outputs: Readonly<Record<string, Tensor>>): number =>
  Object.values(outputs).reduce((total, tensor) => total + tensor.data.byteLength, 0);

/**
 * The default-run checks of one forward that need no upstream golden: the output is finite and the
 * plan-time fusion counts match {@link EXPECTED_FUSIONS}.
 */
const assertForwardStructure = (
  tokens: Float32Array,
  diagnostics: SessionDiagnostics,
  id: string,
): void => {
  const nonFinite = tokens.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);
  assertEquals(nonFinite, 0, `${id}: non-finite`);
  assertEquals(
    { ...diagnostics.lastRunFusions, ...EXPECTED_FUSIONS },
    diagnostics.lastRunFusions,
    `${id}: fusion counts (adaln / rope / silu)`,
  );
};

/**
 * What a describe's `beforeAll` assigns for the its of that describe (the device, the opened
 * container, the prepared graph, ...). The its and the hooks registered after it read the values
 * through {@link required} when they run, never at collection time: a skipped describe still runs
 * its factory, and nothing is assigned then.
 */
type Fixture = {
  gpu?: GpuContext;
  opened?: OpenedContainer;
  prepared?: PreparedModel;
  /** The i8 graph name, read when used ({@link i8Graph}). */
  graph?: string;
  base?: WanRopeBase;
  options?: SessionOptions;
  deviceLost?: string;
};

/** The fixture's value, or a loud failure when its `beforeAll` did not get that far. */
const required = <K extends keyof Fixture>(fixture: Fixture, key: K): NonNullable<Fixture[K]> => {
  const value = fixture[key];
  if (value === undefined) throw new Error(`fixture '${key}' was not assigned by beforeAll`);
  return value;
};

/**
 * The `afterAll` of a describe that acquired a device: wait for the releases, then destroy the
 * device (the `finally` of the former single test).
 */
const releaseDevice = async (fixture: Fixture): Promise<void> => {
  if (fixture.gpu === undefined) return;
  await settleReleases(fixture.gpu);
  fixture.gpu.destroy();
};

/**
 * Registers the hooks of one Session scope in the caller's describe: a `beforeAll` that opens the
 * Session on the fixture's device and prepared graph (`prepare` defaults to the fixture's), and an
 * `afterAll` that disposes it and waits for the releases. The caller registers its device hooks
 * first: Vitest runs `beforeAll` hooks in registration order and `afterAll` hooks in reverse
 * (`sequence.hooks: "stack"`, the default), so the Session opens after the device and is disposed
 * before the device goes. A describe holds at most one Session scope; a test that opened several
 * Sessions one after another wraps each scope in its own describe so two never coexist.
 */
const registerSessionScope = (
  fixture: Fixture,
  prepare: () => PreparedModel = () => required(fixture, "prepared"),
): { readonly session: () => Session; readonly prepared: () => PreparedModel } => {
  let prepared: PreparedModel | undefined;
  let session: Session | undefined;
  beforeAll(async () => {
    prepared = prepare();
    session = await prepared.createContainerSession(required(fixture, "gpu"));
  });
  afterAll(async () => {
    if (session === undefined) return;
    await session.dispose();
    // Wait for this group's allocations to be released before the next group's Session allocates
    // ({@link settleReleases}).
    await settleReleases(required(fixture, "gpu"));
  });
  return {
    session: () => {
      if (session === undefined) throw new Error("the Session was not opened by beforeAll");
      return session;
    },
    prepared: () => {
      if (prepared === undefined) throw new Error("the graph was not prepared by beforeAll");
      return prepared;
    },
  };
};

/**
 * Default run of an S = 192 group: one forward per case on one Session, held to
 * {@link assertForwardStructure}. The upstream bands of the same cases are the opt-in
 * {@link registerNormalizedRatioGate} / S = 192 ratio case. Registers one Session scope
 * ({@link registerSessionScope}) and one it per case in the caller's describe.
 */
const registerSmallStructureGate = (
  fixture: Fixture,
  seriesDir: URL,
  idPrefix: string,
  cases: readonly { readonly name: string; readonly role: string }[],
): void => {
  const scope = registerSessionScope(fixture);
  for (const { name, role } of cases) {
    const id = `${idPrefix}${name}`;
    it(`${id}（${role}）`, async () => {
      const prepared = scope.prepared();
      const session = scope.session();
      const golden = await loadGolden(name, seriesDir);
      const outputs = await session.run(graphInputs(golden, prepared.graph.inputs, name));
      const tensor = outputs[prepared.graph.outputs[0]];
      if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
      assertForwardStructure(tensor.data, session.diagnostics(), id);
    });
  }
};

/** Input of {@link registerFullStructureGate} (one series root × one full-size S). */
type FullStructureSpec = {
  readonly seriesDir: URL;
  /**
   * Prefix of the result IDs, step names and logs (empty for the f16 preset, {@link I8_ID_PREFIX} for the i8 reference
   * partner).
   */
  readonly idPrefix: string;
  readonly tokens: number;
  /** The case whose io inputs drive the two runs. */
  readonly name: string;
  /** Keep the first run's tokens for the normal-mode Uint32 gate ({@link measuredTokens}). */
  readonly keepMeasured: boolean;
};

/**
 * Default run of one full-size S in the measured mode (`gpuTiming`): a fresh Session runs one
 * case twice (the first run carries the pipeline build and the unbacked `initialChunkSize`
 * chunks, the second the backed chunks of later runs), and each run is held to the golden S,
 * {@link assertForwardStructure} and the row-block count derived from the binding limit
 * ({@link expectedRowBlocks}). The it then gates the per-submit GPU time at
 * {@link SUBMIT_GPU_LIMIT_MS}. None of this compares with upstream; the full-size bands are the
 * opt-in {@link registerNormalizedRatioGate}. Registers one it in the caller's describe; the
 * device, the prepared graph and the device-lost reason come from the fixture.
 */
const registerFullStructureGate = (fixture: Fixture, spec: FullStructureSpec): void => {
  const { idPrefix, tokens, name } = spec;
  const id = `${idPrefix}${name}`;
  it(`${idPrefix}S = ${tokens}: GPU time per submit, time, allocation`, async () => {
    const gpu = required(fixture, "gpu");
    const prepared = required(fixture, "prepared");
    // The expected value comes from the binding limit the device actually granted (the same value the runtime's row
    // blocks see).
    const bindingLimit = gpu.device.limits.maxStorageBufferBindingSize;
    const built = performance.now();
    const session = await prepared.createContainerSession(gpu);
    const buildMs = performance.now() - built;
    let note = "";
    try {
      await runRecordedCase(
        results,
        { id: `${idPrefix}full-s${tokens}/submit`, failureNote: () => note },
        async () => {
          const golden = await loadGolden(name, spec.seriesDir);
          assertEquals(
            wanTokenGrid(golden.latentShape, WAN_GEOMETRY).count,
            tokens,
            `${id}: golden S`,
          );
          const inputs = graphInputs(golden, prepared.graph.inputs, name);
          const runs: FullRun[] = [];
          const rowBlocks: number[] = [];
          for (const label of ["run 1", "run 2"]) {
            const started = performance.now();
            const outputs = await session.run(inputs);
            const wallMs = performance.now() - started;
            const tensor = outputs[prepared.graph.outputs[0]];
            if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
            // Kept before any assertion so the normal-mode Uint32 gate still runs on a red run here.
            if (spec.keepMeasured && runs.length === 0) measuredTokens.set(name, tensor.data);
            const diagnostics = session.diagnostics();
            runs.push(observeRun(`${name} ${label}`, diagnostics, wallMs, outputBytes(outputs)));
            rowBlocks.push(rowBlocksOf(diagnostics));
            // Wait for this run's intermediates to be released before the next run allocates (run 2 builds the slot
            // backing).
            await settleReleases(gpu);
            assertForwardStructure(tensor.data, diagnostics, `${id} ${label}`);
          }
          // The conversion table is looked up only in this step (in an environment missing from the table only the time
          // check fails).
          const unitNs = timestampUnitNs(ENVIRONMENT.key);
          const submit = submitGpuNote(session.diagnostics(), unitNs);
          note = [
            submit.note,
            `build ${(buildMs / 1000).toFixed(1)} s`,
            ...runs.map((observed) => formatRun(observed, unitNs)),
            `row blocks ${rowBlocks.join(" / ")}`,
          ].join(" / ");
          console.log(`[wan-dit] ${idPrefix}full size S = ${tokens}, measuring mode: ${note}`);
          for (const [index, observed] of rowBlocks.entries()) {
            assertEquals(
              observed,
              expectedRowBlocks(tokens, bindingLimit),
              `${id} ${runs[index].label}: number of self-attention row blocks (binding limit ${bindingLimit} B)`,
            );
          }
          assert(
            submit.unbackedSubmits > 0,
            "the first run's chunks issued before measured evidence existed (submits held at initialChunkSize) were not measured",
          );
          assert(
            submit.maxMs <= SUBMIT_GPU_LIMIT_MS,
            `max GPU time per submit ${submit.maxMs.toFixed(
              1,
            )} ms exceeded ${SUBMIT_GPU_LIMIT_MS} ms`,
          );
          assertEquals(fixture.deviceLost, undefined, "device lost");
          return { status: "pass", note };
        },
      );
    } finally {
      await session.dispose();
      // Wait for this group's allocations to be released before the next group's Session allocates
      // ({@link settleReleases}).
      await settleReleases(gpu);
    }
  });
};

// The full-size tests (S = 32,760 / 14,040) are placed **before** the S = 192 GPU tests. On the B570 the allocations of
// a destroyed device stay unreleased for a while ({@link settleReleases}), and what an earlier test left behind can
// push the full size's allocations of several GiB into OOM. The full-size tests wait for the releases before dropping
// the device, so they leave nothing to later tests. For the same reason the largest allocation runs first: in the order
// S = 32,760 → 14,040 ({@link FULL_TOKENS_ORDER}), with the Session rebuilt per S — switching shapes on one Session
// keeps the retired slot backing alive until the clean-up of the switching run, so the backings of 2 S values are
// resident at once (the runtime's `estimate.ts` — "the window from retirement to the actual destroy()" in the doc of
// `peakAccountedBytes`).

/** One case of {@link registerNormalizedRatioGate}. */
type GateCase = {
  readonly name: string;
  /** The role shown in the step name (`full-band` / `band` …). */
  readonly role: string;
  /** Acceptance (runs the TS timesteps_proj and the fault injections; not used to decide the band). */
  readonly accept: boolean;
};

/** Input of {@link registerNormalizedRatioGate} (one series root × one group of S). */
type GateSpec = {
  /** Location of the goldens (the series' component directory). */
  readonly seriesDir: URL;
  /**
   * Prefix of the result IDs, step names and logs (empty for the f16 preset, {@link I8_ID_PREFIX} for the i8 reference
   * partner).
   */
  readonly idPrefix: string;
  readonly cases: readonly GateCase[];
  /** Band of the normalized ratio r. `undefined` = not yet derived (r is recorded and the check stops red). */
  readonly band: number | undefined;
  /**
   * Full-size groups only: the golden S each case must have. The row-block count and the per-submit
   * GPU-time gate of the same groups are default-run checks ({@link registerFullStructureGate}).
   */
  readonly tokens?: number;
};

/**
 * One case's r (material for {@link registerBandCandidate} — pushed before the verdict, so runs outside the band are
 * kept too).
 */
type CaseRatio = { readonly name: string; readonly accept: boolean; readonly ratio: number };

/**
 * Runs the check of the normalized ratio r (ADR 0118 decision 8) for one series root and one group of S (builds a
 * Session, runs every case, tears it down and waits for the release). The f16 preset's full size (S = 32,760 / 14,040)
 * and the i8 reference partner (the S = 192 group / 14,040 — ADR 0120 decision 3) go through this same implementation —
 * "the same r check as decision 8" is not grown separately per preset.
 *
 * One case: 1 forward on the io inputs → r = the ratio against the f64 reference divided by the CPU f32 reference's own
 * ratio, judged against the band. Acceptance cases additionally run the TS timesteps_proj input and the 4 fault
 * injections ({@link faultInjections}). If the band is not yet derived, it prints r and stops red.
 * Full-size groups (`tokens`) also assert the golden S. Opt-in only (upstream parity); the
 * finiteness / fusion / row-block / submit-time checks run by default in
 * {@link registerFullStructureGate} and {@link registerSmallStructureGate}.
 *
 * Registers one Session scope ({@link registerSessionScope}) and one it per case in the caller's
 * describe; the device, the prepared graph and the RoPE base come from the fixture.
 *
 * Returns the r of the cases that produced one (material for the band candidate).
 * The returned list is filled by the its as they run (read it from an it registered after them).
 */
const registerNormalizedRatioGate = (fixture: Fixture, spec: GateSpec): readonly CaseRatio[] => {
  const { band, tokens: fullTokens, idPrefix } = spec;
  /** What the failure messages call the band (the full-size groups keep the f16 preset's existing wording). */
  const bandLabel = fullTokens === undefined ? "band" : "full-size band";
  const ratios: CaseRatio[] = [];
  const scope = registerSessionScope(fixture);
  const run = async (inputs: Record<string, Tensor>): Promise<Float32Array<ArrayBuffer>> => {
    const outputs = await scope.session().run(inputs);
    const tensor = outputs[scope.prepared().graph.outputs[0]];
    if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
    // Wait for this run's intermediates to be released before the next run allocates (run 2 builds the slot backing).
    await settleReleases(required(fixture, "gpu"));
    return tensor.data;
  };
  for (const { name, role, accept } of spec.cases) {
    const id = `${idPrefix}${name}`;
    it(`${id}（${role}）`, async () => {
      const prepared = scope.prepared();
      const base = required(fixture, "base");
      // Even a run that fails outside the band keeps the ratio and the fault-injection record in results.json.
      let caseNote = "";
      await runRecordedCase(
        results,
        {
          id,
          failureNote: () => caseNote,
        },
        async ({ measurements }) => {
          const golden = await loadGolden(name, spec.seriesDir);
          if (fullTokens !== undefined) {
            assertEquals(
              wanTokenGrid(golden.latentShape, WAN_GEOMETRY).count,
              fullTokens,
              `${id}: golden S`,
            );
          }
          const inputs = graphInputs(golden, prepared.graph.inputs, name);
          // The error baseline is the f64 reference, and the verdict uses the ratio normalized by the CPU f32
          // reference's own error ({@link DIT_FULL_NORMALIZED_BAND} / {@link I8_NORMALIZED_BAND}). The GPU vs CPU f32
          // reference difference is only recorded.
          const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
          const reference = floatsOf(golden.reference, "output", name);
          const referenceError = referenceErrorOf(golden, name);
          const referenceRatio = ratioOf(referenceError);
          const referenceQuantile = absErrorQuantile(reference, expected, RECORDED_QUANTILE);
          // A run with the band not yet derived is recorded with an infinite band (the same form as the record-only
          // fields — the verdict below turns it red).
          const toleranceOf = (diff: Difference) =>
            band === undefined
              ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
              : recordedTolerance(diff, band * referenceRatio);
          const tokens = await run(inputs);
          const latents = unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY);
          const diff = difference(latents, expected);
          const diffF32 = difference(latents, reference);
          measurements.push(
            {
              output: "latents",
              maxAbs: diff.maxAbs,
              maxRel: diff.maxRel,
              tolerance: toleranceOf(diff),
              stage: "karume",
            },
            {
              output: "latents@f32-reference",
              maxAbs: diffF32.maxAbs,
              maxRel: diffF32.maxRel,
              tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
              stage: "karume",
            },
            {
              output: "f32-reference@f64-reference",
              maxAbs: referenceError.maxAbs,
              maxRel: referenceError.maxRel,
              tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
              stage: "karume",
            },
          );
          ratios.push({ name, accept, ratio: normalizedRatio(diff, referenceRatio) });
          const notes = [
            formatNormalized("io inputs / f64 reference", diff, referenceRatio),
            formatDifference(
              "CPU f32 reference / f64 reference (normalization denominator)",
              referenceError,
            ),
            formatDifference("io inputs / CPU f32 reference (record)", diffF32),
            quantileRatio("io inputs / f64 reference", latents, expected, referenceQuantile),
          ];
          assertEquals(diff.nonFinite, 0, `${id}: non-finite`);

          let hostDiff: Difference | undefined;
          let faults: FaultResult[] = [];
          if (accept) {
            // The product path's input (the host timesteps_proj) is accepted under the same band too (the same form as
            // the S = 192 comparison).
            const width = viewOf(golden.io, "input.timesteps_proj", name).shape[1];
            const hostLatents = unpatchifyTokens(
              await run({
                ...inputs,
                timesteps_proj: {
                  dtype: "f32",
                  shape: [1, width],
                  data: timestepsProj(golden.timestep, width),
                },
              }),
              golden.latentShape,
              WAN_GEOMETRY,
            );
            hostDiff = difference(hostLatents, expected);
            measurements.push({
              output: "latents@host-timesteps-proj",
              maxAbs: hostDiff.maxAbs,
              maxRel: hostDiff.maxRel,
              tolerance: toleranceOf(hostDiff),
              stage: "karume",
            });
            notes.push(
              formatNormalized("TS timesteps_proj / f64 reference", hostDiff, referenceRatio),
              quantileRatio(
                "TS timesteps_proj / f64 reference",
                hostLatents,
                expected,
                referenceQuantile,
              ),
            );
            faults = await faultInjections(golden, inputs, tokens, expected, run, base);
            notes.push(
              ...faults.map(
                ({ note, ratio }) =>
                  `${note}, r ${(ratio / referenceRatio).toPrecision(3)} (${
                    band === undefined
                      ? "band not yet derived"
                      : `${(ratio / referenceRatio / band).toFixed(1)} × the band`
                  })`,
              ),
            );
          }
          // Emitted before the verdict (a run that fell outside the band keeps its ratio at hand too).
          caseNote = notes.join(" / ");
          console.log(`[wan-dit] ${id}: ${caseNote}`);

          if (band === undefined) {
            throw new Error(
              `${id}: band not yet derived (r ${normalizedRatio(diff, referenceRatio).toPrecision(
                3,
              )}) — write the worst deciding r × 5 into the band constant (the last step of the same group prints the band candidate)`,
            );
          }
          // The fault-injection verdicts come first (even when the acceptance falls outside the band, whether the band
          // catches the faults is judged).
          for (const { label, ratio } of faults) {
            const normalized = ratio / referenceRatio;
            assert(
              normalized > band,
              `${id}: fault injection ${label} r ${normalized.toPrecision(
                3,
              )} stayed inside the ${bandLabel} ${band}`,
            );
          }
          if (accept) {
            const subtle = faults.find(({ label }) => label === TIMESTEP_OFF_BY_ONE);
            if (subtle === undefined) {
              throw new Error(`${id}: no fault injection ${TIMESTEP_OFF_BY_ONE}`);
            }
            const margin = subtle.ratio / referenceRatio / band;
            assert(
              margin >= SUBTLE_FAULT_MARGIN,
              `${id}: fault injection ${TIMESTEP_OFF_BY_ONE} is at the ${bandLabel} × ${margin.toFixed(
                2,
              )} (should be at least ${SUBTLE_FAULT_MARGIN} × — a sign that the band is too wide)`,
            );
          }
          assert(
            normalizedRatio(diff, referenceRatio) <= band,
            `${id}: ${notes[0]} outside the ${bandLabel} ${band}`,
          );
          if (hostDiff !== undefined) {
            assert(
              normalizedRatio(hostDiff, referenceRatio) <= band,
              `${id}: TS timesteps_proj r ${normalizedRatio(hostDiff, referenceRatio).toPrecision(
                3,
              )} outside the ${bandLabel} ${band}`,
            );
          }
          return { status: "pass", note: notes.join(" / ") };
        },
      );
    });
  }
  return ratios;
};

/**
 * A step that prints the candidate for a band not yet derived (`undefined`) and stops red. The candidate is made from
 * the deciding cases alone, by the derivation rule (ADR 0118 decision 8 — the worst deciding r of the group × 5). The
 * acceptance r are listed but not used for the candidate (independence of the band decision and the acceptance). A
 * run where any deciding case failed before producing r prints no candidate (a worst with gaps is not the worst). Does
 * nothing once the band is derived.
 * Registers that it in the caller's describe after the gate's its (`ratiosOf` reads the list
 * {@link registerNormalizedRatioGate} returned when the it runs); the band is a constant, so
 * whether the it exists is known at collection time.
 */
const registerBandCandidate = (
  label: string,
  cases: readonly GateCase[],
  ratiosOf: () => readonly CaseRatio[],
  band: number | undefined,
): void => {
  if (band !== undefined) return;
  it(`${label}: band candidate (not yet derived)`, () => {
    const ratios = ratiosOf();
    const expected = cases.filter(({ accept }) => !accept).length;
    const decided = ratios.filter(({ accept }) => !accept);
    const worst = Math.max(...decided.map(({ ratio }) => ratio));
    const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
    const candidate =
      decided.length === expected
        ? `worst r of the ${expected} deciding cases ${worst.toPrecision(3)} × 5 = ${(
            worst * 5
          ).toPrecision(3)}` +
          ` (rounded up to 2 significant digits: ${Number((Math.ceil((worst * 5) / unit) * unit).toPrecision(2))})`
        : `of the ${expected} deciding cases, ${decided.length} produced r — no candidate (investigate the failed cases first)`;
    const accepted = ratios
      .filter(({ accept }) => accept)
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`);
    throw new Error(
      `${label}: band not yet derived — ${candidate}. Acceptance r (not used to decide the band): ${
        accepted.join(" / ") || "none"
      }`,
    );
  });
};

/**
 * A supply surface with the per-channel scale of one linear (`weight`) doubled (the i8-specific fault injection —
 * {@link I8_SCALE_FAULT_WEIGHT}). The container file is not rewritten: a **copy** of the fetched block (after the
 * sha256 verification, since it is an unverified source) is doubled and returned (the MUST of
 * `BoundContainer.readBlock` — the returned buffer is not modified). The scale payload is an f32 sequence from the
 * start of the block (the form Session construction reads with `subarray(0, payloadBytes)`).
 */
const scaledScaleContainer = (
  opened: OpenedContainer,
  graph: string,
  weight: string,
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const scale = bound?.supplies.get(weight)?.scale;
  if (scale === undefined) {
    throw new Error(`graph '${graph}': '${weight}' has no companion scale (not an i8 linear)`);
  }
  return {
    graphs: opened.graphs,
    readBlock: async (id) => {
      const bytes = await opened.readBlock(id);
      if (id !== scale.id) return bytes;
      const doubled = bytes.slice();
      const values = new Float32Array(doubled.buffer, 0, scale.payloadBytes / 4);
      for (let index = 0; index < values.length; index += 1) values[index] *= 2;
      return doubled;
    },
  };
};

/**
 * Runs the acceptance cases with the i8-specific fault injection ({@link scaledScaleContainer}) and checks that r falls
 * outside the band (ADR 0120 acceptance, step 3). The Session is built separately after the Session of
 * {@link registerNormalizedRatioGate} is torn down (the 2 are not resident at once — 2 at full size eat into the B570's
 * budget).
 * Registers one Session scope on the scaled container ({@link registerSessionScope}) and one it
 * per accept case; the caller wraps it in its own describe, after the ratio gate's describe.
 */
const registerI8ScaleFault = (
  fixture: Fixture,
  cases: readonly GateCase[],
  band: number | undefined,
): void => {
  const label = I8_SCALE_FAULT_LABEL;
  const scope = registerSessionScope(fixture, () => {
    const graph = required(fixture, "graph");
    return prepareContainer(
      scaledScaleContainer(required(fixture, "opened"), graph, I8_SCALE_FAULT_WEIGHT),
      graph,
    );
  });
  for (const { name } of cases.filter(({ accept }) => accept)) {
    const id = `${I8_ID_PREFIX}${name}/scale-fault`;
    it(`${id} (fault injection ${label} → outside the band)`, async () => {
      const gpu = required(fixture, "gpu");
      const prepared = scope.prepared();
      const session = scope.session();
      let note = "";
      await runRecordedCase(results, { id, failureNote: () => note }, async ({ measurements }) => {
        const golden = await loadGolden(name, I8_GOLDEN_DIR);
        const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
        const referenceRatio = ratioOf(referenceErrorOf(golden, name));
        const outputs = await session.run(graphInputs(golden, prepared.graph.inputs, name));
        const tensor = outputs[prepared.graph.outputs[0]];
        if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
        await settleReleases(gpu);
        const diff = difference(
          unpatchifyTokens(tensor.data, golden.latentShape, WAN_GEOMETRY),
          expected,
        );
        const normalized = normalizedRatio(diff, referenceRatio);
        measurements.push({
          output: "latents@scale-fault",
          maxAbs: diff.maxAbs,
          maxRel: diff.maxRel,
          tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
          stage: "karume",
        });
        note = `${formatNormalized(`fault injection ${label} / f64 reference`, diff, referenceRatio)}（${
          band === undefined
            ? "band not yet derived"
            : `${(normalized / band).toFixed(1)} × the band`
        }）`;
        console.log(`[wan-dit] ${id}: ${note}`);
        if (band === undefined) {
          throw new Error(`${id}: band not yet derived (fault-injection r ${normalized})`);
        }
        assert(
          normalized > band,
          `${id}: fault injection ${label} r ${normalized.toPrecision(3)} stayed inside band ${band}`,
        );
        return { status: "pass", note };
      });
    });
  }
};

/**
 * The opt-in parity of one i8 reference-seat group, registered in the caller's describe in the
 * order the former test ran it: the ratio gate ({@link registerNormalizedRatioGate}), the scale
 * fault ({@link registerI8ScaleFault}), then the band candidate ({@link registerBandCandidate}).
 * The ratio gate and the scale fault each hold one Session, so each gets its own describe and the
 * first Session is disposed before the second opens. `label` names the group (the ratio gate's
 * describe and the band candidate).
 */
const registerI8ParityGroups = (fixture: Fixture, label: string, spec: GateSpec): void => {
  let ratios: readonly CaseRatio[] = [];
  describe(label, () => {
    ratios = registerNormalizedRatioGate(fixture, spec);
  });
  describe(`${I8_ID_PREFIX}fault injection ${I8_SCALE_FAULT_LABEL}`, () => {
    registerI8ScaleFault(fixture, spec.cases, spec.band);
  });
  registerBandCandidate(label, spec.cases, () => ratios, spec.band);
};

describe.skipIf(!FULL_AVAILABLE || !GPU_AVAILABLE)(
  "Wan DiT full size, measuring mode (real GPU): 1 forward at S = 32,760 / 14,040 has the per-S row-block count, " +
    "fusion counts, 0 non-finite values, GPU time per submit ≤ 1 s",
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.prepared = prepareContainer(await openWanComponent(COMPONENT, "f16"), GRAPH);
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      // Measuring mode (timestamp-query): the GPU time per submit is measured by piggybacking on the existing
      // collection (no new waits — the runtime's `ChunkBudgetStats.submitGpuTime`). The computation is the same as a
      // normal run, but the window's wall clock changes because 1 dispatch = 1 pass. Chunk boundaries are decided by
      // estimates learned from wall-clock windows, so only the chunks before measured evidence exists
      // (initialChunkSize) are the same as in a normal run; the boundaries after the evidence (after the first window
      // closes) can differ.
      fixture.gpu = await acquireTestGpu({
        gpuTiming: true,
        onDeviceLost: (info) => {
          fixture.deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    afterAll(() => releaseDevice(fixture));
    for (const { tokens, normalModeCase } of FULL_CASES_BY_TOKENS) {
      registerFullStructureGate(fixture, {
        seriesDir: GOLDEN_DIR,
        idPrefix: "",
        tokens,
        name: normalModeCase,
        keepMeasured: true,
      });
    }
  },
);

describe.skipIf(!FULL_AVAILABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan DiT full size, comparison (real GPU, measuring mode / diffusers CPU f64): 1 forward at S = 32,760 / 14,040 is inside the per-S " +
    `full-size band, fault injections are outside the band${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      const opened = await openWanComponent(COMPONENT, "f16");
      fixture.prepared = prepareContainer(opened, GRAPH);
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.base = await readRopeBase(opened);
      // Same measured-mode device as the band derivation (ADR 0118 stage 3 / 8 measured there).
      fixture.gpu = await acquireTestGpu({ gpuTiming: true });
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    afterAll(() => releaseDevice(fixture));
    // One describe per S: each group holds its own Session, closed before the next group opens.
    for (const { tokens: fullTokens, cases } of FULL_CASES_BY_TOKENS) {
      describe(`S = ${fullTokens}`, () => {
        registerNormalizedRatioGate(fixture, {
          seriesDir: GOLDEN_DIR,
          idPrefix: "",
          cases: cases.map(({ name, role }) => ({ name, role, accept: role === "full-accept" })),
          band: DIT_FULL_NORMALIZED_BAND[fullTokens],
          tokens: fullTokens,
        });
      });
    }
  },
);

describe.skipIf(!FULL_AVAILABLE || !GPU_AVAILABLE)(
  "Wan DiT full size, normal mode (real GPU, no measurement): records the time of 1 forward per S (32,760 / 14,040)" +
    " (equal to the measuring-mode output as Uint32, 0 non-finite values)",
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.prepared = prepareContainer(await openWanComponent(COMPONENT, "f16"), GRAPH);
      // The time is taken from a normal run with measurement off (ADR 0118 addendum 2026-10-02 — measuring mode is 1
      // dispatch = 1 pass).
      fixture.gpu = await acquireTestGpu({
        onDeviceLost: (info) => {
          fixture.deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    afterAll(() => releaseDevice(fixture));
    // Rebuild the Session per S (for the same reason as the measuring-mode comparison — never 2 S backings resident at
    // once).
    for (const { tokens: fullTokens, normalModeCase: name } of FULL_CASES_BY_TOKENS) {
      it(`S = ${fullTokens}`, async () => {
        const gpu = required(fixture, "gpu");
        const prepared = required(fixture, "prepared");
        const built = performance.now();
        const session = await prepared.createContainerSession(gpu);
        const buildMs = performance.now() - built;
        try {
          await runRecordedCase(results, { id: `${name}/normal-mode` }, async () => {
            const golden = await loadGolden(name);
            const inputs = graphInputs(golden, prepared.graph.inputs, name);
            const runs: FullRun[] = [];
            const produced: Float32Array<ArrayBuffer>[] = [];
            // Run 1 is the first run (pipeline creation and the initialChunkSize chunks before measured evidence), run
            // 2 the shape of the second and later runs (slot backing, chunks cut by the time budget). Run 2 is the
            // closer one to one generation step.
            for (const label of ["run 1", "run 2"]) {
              const started = performance.now();
              const outputs = await session.run(inputs);
              const wallMs = performance.now() - started;
              const tensor = outputs[prepared.graph.outputs[0]];
              if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
              produced.push(tensor.data);
              runs.push(observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)));
              await settleReleases(gpu);
            }
            const tokens = produced[produced.length - 1];
            // Kept before any assertion so the opt-in band case can still report on a red run here.
            normalModeTokens.set(name, tokens);
            const budget = session.diagnostics().submit.chunkBudget;
            const note = [
              `build ${(buildMs / 1000).toFixed(1)} s`,
              ...runs.map((observed) => formatRun(observed)),
              `max window mean ${
                budget.maxWindowMeanMs?.toFixed(1) ?? "—"
              } ms, ${budget.overBudgetChunks} over budget`,
            ].join(" / ");
            console.log(`[wan-dit] full size S = ${fullTokens}, normal mode: ${note}`);
            assertEquals(
              tokens.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0),
              0,
              `${name}: non-finite`,
            );
            const measured = measuredTokens.get(name);
            if (measured === undefined) {
              throw new Error(
                `${name}: no output from the measuring-mode comparison (run the measuring-mode comparison first in the same run — ` +
                  "the MUST of measuredTokens)",
              );
            }
            for (const [index, data] of produced.entries()) {
              const { label } = runs[index];
              assertEquals(
                firstBitMismatch(data, measured),
                -1,
                `${name}: normal-mode ${label} does not equal the measuring-mode output as Uint32` +
                  " (the value is the index of the first differing element)",
              );
            }
            assertEquals(fixture.deviceLost, undefined, "device lost");
            return { status: "pass", note };
          });
        } finally {
          await session.dispose();
          await settleReleases(gpu);
        }
      });
    }
  },
);

describe.skipIf(!FULL_AVAILABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan DiT full size, normal mode (no measurement / diffusers CPU f64): 1 forward per S (32,760 / 14,040) is inside the full-size band" +
    UPSTREAM_PARITY_OPT_IN,
  () => {
    for (const { tokens: fullTokens, normalModeCase: name } of FULL_CASES_BY_TOKENS) {
      it(`S = ${fullTokens}`, async () => {
        const fullBand = DIT_FULL_NORMALIZED_BAND[fullTokens];
        await runRecordedCase(
          results,
          { id: `${name}/normal-mode/band` },
          async ({ measurements }) => {
            const tokens = normalModeTokens.get(name);
            if (tokens === undefined) {
              throw new Error(
                `${name}: no normal-mode output (the default normal-mode case must run first in the same ` +
                  "run — normalModeTokens)",
              );
            }
            const golden = await loadGolden(name);
            const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
            const referenceRatio = ratioOf(referenceErrorOf(golden, name));
            const diff = difference(
              unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY),
              expected,
            );
            measurements.push({
              output: "latents",
              maxAbs: diff.maxAbs,
              maxRel: diff.maxRel,
              tolerance: recordedTolerance(diff, fullBand * referenceRatio),
              stage: "karume",
            });
            const note = formatNormalized("io inputs / f64 reference", diff, referenceRatio);
            console.log(`[wan-dit] full size S = ${fullTokens}, normal mode: ${note}`);
            assertEquals(diff.nonFinite, 0, `${name}: non-finite`);
            assert(
              normalizedRatio(diff, referenceRatio) <= fullBand,
              `${name}: normal-mode r ${normalizedRatio(diff, referenceRatio).toPrecision(
                3,
              )} outside the full-size band ${fullBand}`,
            );
            return { status: "pass", note };
          },
        );
      });
    }
  },
);

// The practical preset's time and allocation (ADR 0120 step 4) and the i8 reference partner's r check (step 3) come
// after the f16 preset's full size and before S = 192. The order is by allocation size (practical preset S = 32,760 →
// 14,040 → i8 reference partner S = 14,040 → the S = 192 group) — for the same reason the f16 preset's full size comes
// first (the B570's `destroy()` lag, {@link settleReleases}).

/**
 * Runs 2 forwards of the practical preset (run 1 = pipeline creation and the chunks before measured evidence, run 2 =
 * the shape of the second and later runs, close to one generation pass — the same order as the f16 preset's normal
 * mode).
 */
const practicalRuns = async (
  session: Session,
  gpu: GpuContext,
  inputs: Record<string, Tensor>,
  output: string,
): Promise<{ readonly runs: readonly FullRun[]; readonly nonFinite: number }> => {
  const runs: FullRun[] = [];
  let nonFinite = 0;
  for (const label of ["run 1", "run 2"]) {
    const started = performance.now();
    const outputs = await session.run(inputs);
    const wallMs = performance.now() - started;
    const tensor = outputs[output];
    if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
    nonFinite += tensor.data.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);
    runs.push(observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)));
    // Wait for this run's intermediates to be released before the next run allocates.
    await settleReleases(gpu);
  }
  return { runs, nonFinite };
};

describe.skipIf(!PRACTICAL_RUNNABLE || !GPU_AVAILABLE)(
  `Wan DiT practical preset ${PRACTICAL_QUANT} (real GPU, measuring mode): records the GPU time of 1 pass at S = 32,760 / 14,040` +
    ", GPU time per submit ≤ 1 s, 0 non-finite values",
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.options = practicalSessionOptions();
      fixture.prepared = prepareContainer(await openWanComponent(COMPONENT, "i8"), i8Graph());
      // Measuring mode means the same as in the f16 preset's full-size comparison (1 dispatch = 1 pass — the normal
      // mode below takes the wall clock). The 1-submit check is re-measured per preset for the extra dispatches of a8
      // (3,087 → 3,311 / step on anima — ADR 0030) (ADR 0120 risk 8).
      fixture.gpu = await acquireTestGpu({
        gpuTiming: true,
        onDeviceLost: (info) => {
          fixture.deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    afterAll(() => releaseDevice(fixture));
    // Rebuild the Session per S (never 2 S backings resident at once — the same as the f16 preset's full size).
    for (const { tokens, name } of PRACTICAL_TIMING_CASES) {
      it(`${PRACTICAL_QUANT} S = ${tokens}`, async () => {
        const gpu = required(fixture, "gpu");
        const prepared = required(fixture, "prepared");
        const options = required(fixture, "options");
        const built = performance.now();
        const session = await prepared.createContainerSession(gpu, options);
        const buildMs = performance.now() - built;
        let note = "";
        try {
          await runRecordedCase(
            results,
            { id: `${PRACTICAL_QUANT}/full-s${tokens}/submit`, failureNote: () => note },
            async () => {
              const golden = await loadGolden(name);
              const { runs, nonFinite } = await practicalRuns(
                session,
                gpu,
                graphInputs(golden, prepared.graph.inputs, name),
                prepared.graph.outputs[0],
              );
              const unitNs = timestampUnitNs(ENVIRONMENT.key);
              const submit = submitGpuNote(session.diagnostics(), unitNs);
              note = [
                submit.note,
                `build ${(buildMs / 1000).toFixed(1)} s`,
                ...runs.map((observed) => formatRun(observed, unitNs)),
                `fusions ${JSON.stringify(session.diagnostics().lastRunFusions)}`,
                `non-finite ${nonFinite}`,
              ].join(" / ");
              console.log(
                `[wan-dit] practical preset ${PRACTICAL_QUANT} S = ${tokens}, measuring mode: ${note}`,
              );
              assertEquals(nonFinite, 0, `${name}: non-finite`);
              assert(
                submit.unbackedSubmits > 0,
                "the first run's chunks issued before measured evidence existed (submits held at initialChunkSize) were not measured",
              );
              assert(
                submit.maxMs <= SUBMIT_GPU_LIMIT_MS,
                `max GPU time per submit ${submit.maxMs.toFixed(
                  1,
                )} ms exceeded ${SUBMIT_GPU_LIMIT_MS} ms`,
              );
              assertEquals(fixture.deviceLost, undefined, "device lost");
              return { status: "pass", note };
            },
          );
        } finally {
          await session.dispose();
          await settleReleases(gpu);
        }
      });
    }
  },
);

describe.skipIf(!I8_FULL_AVAILABLE || !GPU_AVAILABLE)(
  "Wan DiT i8 reference partner f16+dit8 (real GPU, measuring mode): 1 forward at S = 14,040 has the row blocks, fusion counts, 0 non-finite values, " +
    "GPU time per submit ≤ 1 s",
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.prepared = prepareContainer(await openWanComponent(COMPONENT, "i8"), i8Graph());
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.gpu = await acquireTestGpu({
        gpuTiming: true,
        onDeviceLost: (info) => {
          fixture.deviceLost = `${info.reason}: ${info.message}`;
        },
      });
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    afterAll(() => releaseDevice(fixture));
    registerFullStructureGate(fixture, {
      seriesDir: I8_GOLDEN_DIR,
      idPrefix: I8_ID_PREFIX,
      tokens: 14040,
      name: I8_FULL_CASES[0].name,
      keepMeasured: false,
    });
  },
);

describe.skipIf(!I8_FULL_AVAILABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan DiT i8 reference partner f16+dit8 (real GPU, measuring mode / diffusers CPU f64): 1 forward at S = 14,040 is inside the r band, " +
    `the 5 fault injections (the 4 of the f16 preset + the doubled scale) are outside the band${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.graph = i8Graph();
      fixture.opened = await openWanComponent(COMPONENT, "i8");
      fixture.prepared = prepareContainer(fixture.opened, fixture.graph);
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.base = await readRopeBase(fixture.opened);
      // Same measured-mode device as the band derivation (I8_NORMALIZED_BAND: S = 14,040 measured there).
      fixture.gpu = await acquireTestGpu({ gpuTiming: true });
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    afterAll(() => releaseDevice(fixture));
    registerI8ParityGroups(fixture, `${I8_ID_PREFIX}S = 14,040`, {
      seriesDir: I8_GOLDEN_DIR,
      idPrefix: I8_ID_PREFIX,
      cases: I8_FULL_CASES,
      band: I8_NORMALIZED_BAND[14040],
      tokens: 14040,
    });
  },
);

describe.skipIf(!I8_SMALL_AVAILABLE || !GPU_AVAILABLE)(
  "Wan DiT i8 reference partner f16+dit8 (real GPU): fusion counts and 0 non-finite values of 1 forward of the S = 192 group",
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.prepared = prepareContainer(await openWanComponent(COMPONENT, "i8"), i8Graph());
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.gpu = await acquireTestGpu();
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    // Wait for the releases before dropping the device (leaves budget for later tests — {@link settleReleases}).
    afterAll(() => releaseDevice(fixture));
    registerSmallStructureGate(fixture, I8_GOLDEN_DIR, I8_ID_PREFIX, I8_SMALL_CASES);
  },
);

describe.skipIf(!I8_SMALL_AVAILABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  "Wan DiT i8 reference partner f16+dit8 (real GPU / diffusers CPU f64): 1 forward of the S = 192 group is inside the r band, " +
    `the 5 fault injections (the 4 of the f16 preset + the doubled scale) are outside the band${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.graph = i8Graph();
      fixture.opened = await openWanComponent(COMPONENT, "i8");
      fixture.prepared = prepareContainer(fixture.opened, fixture.graph);
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.base = await readRopeBase(fixture.opened);
      fixture.gpu = await acquireTestGpu();
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    // Wait for the releases before dropping the device (leaves budget for later tests — {@link settleReleases}).
    afterAll(() => releaseDevice(fixture));
    registerI8ParityGroups(fixture, `${I8_ID_PREFIX}S = 192 group`, {
      seriesDir: I8_GOLDEN_DIR,
      idPrefix: I8_ID_PREFIX,
      cases: I8_SMALL_CASES,
      band: I8_NORMALIZED_BAND[192],
    });
  },
);

describe.skipIf(!SMALL_AVAILABLE || !GPU_AVAILABLE)(
  "Wan DiT S shape (real GPU): fusion counts and 0 non-finite values of 1 forward of every case",
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      fixture.prepared = prepareContainer(await openWanComponent(COMPONENT, "f16"), GRAPH);
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.gpu = await acquireTestGpu();
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    // Wait for the releases before dropping the device (leaves budget for later tests — {@link settleReleases}).
    afterAll(() => releaseDevice(fixture));
    registerSmallStructureGate(fixture, GOLDEN_DIR, "", CASES);
  },
);

describe.skipIf(!SMALL_AVAILABLE || !GPU_AVAILABLE || !UPSTREAM_PARITY)(
  `Wan DiT comparison (real GPU / diffusers CPU f32): 1 forward of the S shape is inside the band, fault injections are outside the band${UPSTREAM_PARITY_OPT_IN}`,
  () => {
    const fixture: Fixture = {};
    beforeAll(async () => {
      const opened = await openWanComponent(COMPONENT, "f16");
      fixture.prepared = prepareContainer(opened, GRAPH);
      assertEquals(fixture.prepared.graph.outputs.length, 1, "the product graph has 1 output");
      fixture.base = await readRopeBase(opened);
      fixture.gpu = await acquireTestGpu();
      assertAdapterMatchesEnvironment(fixture.gpu);
    });
    // Wait for the releases before dropping the device (leaves budget for later tests — {@link settleReleases}).
    afterAll(() => releaseDevice(fixture));
    const scope = registerSessionScope(fixture);
    const run = async (inputs: Record<string, Tensor>): Promise<Float32Array<ArrayBuffer>> => {
      const outputs = await scope.session().run(inputs);
      const tensor = outputs[scope.prepared().graph.outputs[0]];
      if (tensor.dtype !== "f32") throw new Error(`output is ${tensor.dtype}`);
      return tensor.data;
    };
    for (const { name, role } of CASES) {
      it(`${name}（${role}）`, async () => {
        const prepared = scope.prepared();
        const base = required(fixture, "base");
        await runRecordedCase(results, { id: name }, async ({ measurements }) => {
          const golden = await loadGolden(name);
          const inputs = graphInputs(golden, prepared.graph.inputs, name);
          const expected = floatsOf(golden.reference, "output", name);
          const tokens = await run(inputs);
          const latents = unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY);
          const diff = difference(latents, expected);
          measurements.push({
            output: "latents",
            maxAbs: diff.maxAbs,
            maxRel: diff.maxRel,
            tolerance: recordedTolerance(diff),
            stage: "karume",
          });
          const notes = [formatDifference("io inputs", diff)];
          assertEquals(diff.nonFinite, 0, `${name}: non-finite`);

          // The values run with the host timesteps_proj (TS) (the product path's input) are checked under the same
          // ratio band too. The difference of the input itself (1 ULP of `exp` between torch and TS — worst 3.05e-5) is
          // covered by the absolute atol 1.5e-4 of `wan_dit_host_test.ts` (sin / cos range over [-1, 1], so absolute is
          // enough — kept as it is even though the band's metric became a ratio).
          const width = viewOf(golden.io, "input.timesteps_proj", name).shape[1];
          const hostTimestep = unpatchifyTokens(
            await run({
              ...inputs,
              timesteps_proj: {
                dtype: "f32",
                shape: [1, width],
                data: timestepsProj(golden.timestep, width),
              },
            }),
            golden.latentShape,
            WAN_GEOMETRY,
          );
          const hostDiff = difference(hostTimestep, expected);
          measurements.push({
            output: "latents@host-timesteps-proj",
            maxAbs: hostDiff.maxAbs,
            maxRel: hostDiff.maxRel,
            tolerance: recordedTolerance(hostDiff),
            stage: "karume",
          });
          notes.push(formatDifference("TS timesteps_proj", hostDiff));
          const faults =
            role === "accept"
              ? await faultInjections(golden, inputs, tokens, expected, run, base)
              : [];
          notes.push(...faults.map(({ note }) => note));
          // Emitted before the verdict (a run that fell outside the band keeps the ratios of every case at hand too).
          console.log(`[wan-dit] ${name}: ${notes.join(" / ")}`);

          // The fault-injection verdicts come first (even when the acceptance falls outside the band, whether the band
          // catches the faults is judged).
          for (const { label, ratio } of faults) {
            assert(
              ratio > DIT_RATIO_BAND,
              `${name}: fault injection ${label} stayed inside band ${DIT_RATIO_BAND}`,
            );
          }
          if (role !== "growth") {
            assert(
              ratioOf(diff) <= DIT_RATIO_BAND,
              `${name}: ${notes[0]} outside band ${DIT_RATIO_BAND}`,
            );
            assert(
              ratioOf(hostDiff) <= DIT_RATIO_BAND,
              `${name}: ${notes[1]} outside band ${DIT_RATIO_BAND}`,
            );
          }
          return { status: "pass", note: notes.join(" / ") };
        });
      });
    }
  },
);

/** Result of one fault injection (the caller judges after printing the ratios of every case). */
type FaultResult = { readonly label: string; readonly ratio: number; readonly note: string };

/**
 * Label of the subtle fault injection (the full-size comparison looks it up for the {@link SUBTLE_FAULT_MARGIN} check).
 */
const TIMESTEP_OFF_BY_ONE = "timestep off by 1";

/**
 * The 4 fault injections of the acceptance cases (addendum 2026-10-02 — evidence that the band is not too wide). Each
 * must fall outside the band (if an injection does not turn red, revisit how the band is decided).
 *
 * - RoPE h / w swapped (building the tables wrongly on a non-square grid — on a square grid the values coincide and it
 *   cannot be seen. So the acceptance grid is required to be non-square, and a square one fails instead of silently
 *   skipping the injection — never green with 4 reduced to 3)
 * - unpatchify order swapped (the output read in the input order `(c,pt,ph,pw)`)
 * - the timestep's cos / sin reversed
 * - **the timestep off by 1** (`timesteps_proj` built with t + 1 — TS's `timestepsProj`). The 3 above are O(1)
 *   differences that turn red wherever the band is, but this is a subtle fault with a ratio of 2e-3 to 9e-3 and
 *   grounds the upper side of the band.
 */
const faultInjections = async (
  golden: Golden,
  inputs: Record<string, Tensor>,
  tokens: Float32Array,
  expected: Float32Array,
  run: (inputs: Record<string, Tensor>) => Promise<Float32Array<ArrayBuffer>>,
  base: WanRopeBase,
): Promise<FaultResult[]> => {
  const grid = wanTokenGrid(golden.latentShape, WAN_GEOMETRY);
  const unpatchify = (data: Float32Array<ArrayBuffer>) =>
    unpatchifyTokens(data, golden.latentShape, WAN_GEOMETRY);
  const results: FaultResult[] = [];
  const measure = (label: string, actual: Float32Array) => {
    const diff = difference(actual, expected);
    const ratio = ratioOf(diff);
    results.push({
      label,
      ratio,
      note: `fault injection ${label}: maxAbs ${diff.maxAbs.toExponential(3)} (ratio ${ratio.toExponential(
        2,
      )})`,
    });
  };

  assert(
    grid.rows !== grid.cols,
    `acceptance case grid ${grid.frames}·${grid.rows}·${grid.cols} is square — swapped RoPE h / w would not show in the values` +
      " (choose a non-square grid for acceptance)",
  );
  const swapped = wanRopeTables(base, { ...grid, rows: grid.cols, cols: grid.rows });
  const shape = inputs.rope_cos.shape;
  measure(
    "RoPE h / w swapped",
    unpatchify(
      await run({
        ...inputs,
        rope_cos: { dtype: "f32", shape, data: swapped.cos },
        rope_sin: { dtype: "f32", shape, data: swapped.sin },
      }),
    ),
  );
  measure("unpatchify order", unpatchify(transposeTokenAxes(tokens, WAN_GEOMETRY)));
  const proj = inputs.timesteps_proj;
  if (proj.dtype !== "f32") throw new Error(`timesteps_proj is ${proj.dtype}`);
  const half = proj.data.length / 2;
  const flipped = Float32Array.from(
    { length: proj.data.length },
    (_, index) => proj.data[index < half ? half + index : index - half],
  );
  measure(
    "timestep cos / sin reversed",
    unpatchify(
      await run({ ...inputs, timesteps_proj: { dtype: "f32", shape: proj.shape, data: flipped } }),
    ),
  );
  measure(
    TIMESTEP_OFF_BY_ONE,
    unpatchify(
      await run({
        ...inputs,
        timesteps_proj: {
          dtype: "f32",
          shape: proj.shape,
          data: timestepsProj(golden.timestep + 1, proj.shape[1]),
        },
      }),
    ),
  );
  return results;
};

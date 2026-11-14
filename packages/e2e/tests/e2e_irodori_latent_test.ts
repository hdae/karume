/**
 * Irodori's **port check** (real GPU). It goes through distribution → `Pipeline.fromAssets` →
 * `generateLatent` and **compares every element** of the resulting latent **with the full-loop
 * golden's `z`** (the latent version of SBV2's WAV sha256 check `e2e_sbv2_wav_test.ts` — there
 * is no codec here yet, so it cannot close on the waveform).
 *
 * The golden is `outputs/test-assets/goldens/irodori-v4-small/pipeline/` (`case.full` /
 * `case.no-ref` / `meta.json`), written by `tools/export-recipes/irodori/pipeline_ref.py`.
 * That script writes them only **after passing a comparison with the upstream
 * `sample_euler_rf_cfg` every time**, so this check closing is the second stage of "the TS host
 * implementation emits the same latent as upstream".
 *
 * ## What is pinned by bit identity and what is checked with a tolerance
 *
 * - **S (the latent length) and the forward count must match exactly**. Both are integer decisions
 *   (expm1 → banker's rounding → clamp / the t interval where CFG applies), and being off by even
 *   1 means the formula differs.
 * - **`z` has a tolerance**. The graph path differs structurally from the golden side (torch's
 *   graph wrapper) in "whether the condition KV is recomputed every forward", "the reduction order"
 *   and "the rounding of WGSL vs ATen", so bit identity cannot arise in principle. See
 *   {@link Z_ATOL} for the threshold's derivation.
 * - **The golden's initial noise is injected** (`initialNoise`). TS's `Randn` and torch's Philox
 *   are different sequences, so matching the random numbers too would make the check demand "a
 *   transcription of the RNG". The determinism of the random sequence itself is checked by another
 *   test (two generations with the same seed → byte-identical).
 *
 * ## Effective-value drift (the check's premise)
 *
 * The golden and the distribution are **baked separately** (`tools/export-recipes/irodori/pipeline_ref.py` and `tools/export-recipes/dist.py`),
 * so a sampler knob moving on only one side would only show as "z does not match". To state first
 * that the premise broke, the comparison of `pipelineConfig` with `meta.json` is its own check.
 *
 * ## Parameterized by storage-dtype series (the pattern of ADR 0018 / 0027)
 *
 * The distribution's quant seats (`f32` / `f16` / `i8`) and the full-loop golden series are
 * **one to one**, and per series ① the golden is re-baked with the fake-quantized weights and ② the
 * tolerance is **derived independently from the raw measurement** (not shared across series — one
 * re-derivation would silently move the other). The compressed series' goldens are also taken with
 * the rounded weights, so every series looks at "implementation differences only"; the quality of
 * the quantization itself belongs to the listening check (the user's ruling).
 *
 * NOTE: the seat name and the series root are **not always spelled the same** (seat `i8+dit4` ↔
 * series `-i4` — a seat name names the execution configuration, a series root the storage dtype,
 * and `i8` / `i8-a8` share the same bytes). This check covers seats up to the storage dtype; the
 * activation-quantized `i8-a8` seat **cannot** be part of the numeric-parity net, so another check
 * owns it (`e2e_irodori_ab_test.ts` — ADR 0025 decision 6 / 0026).
 *
 * **The exact match of S and the forward count is required whatever the series** — lowering the
 * storage dtype has no axis that moves the duration output (it changes only the weight storage, and
 * the formula deciding S stays on the host), so if this breaks, suspect the path, not the
 * tolerance. **Whether S moves with i8 is a measurement on another axis** (the S drift table of
 * `tools/export-recipes/irodori/measure_quant.py`); read that first if it breaks.
 *
 * MUST: the assets are the mirror `models/karume-irodori-v4-small/` and the goldens above. An
 * environment without the mirror, without the goldens, or without a GPU SKIPs explicitly with the
 * missing path (never delete a test to stay green — ADR 0005). Two absences FAIL instead: a golden
 * absent from a complete fetch of the pinned test assets (the dataset at the pin lacks it — ADR
 * 0139), and a case file absent beside a present `meta.json` (an inconsistent golden set).
 *
 * NOTE: only the tests use `Deno.readFile` for local reads. The package itself uses Web standard
 * APIs only (a cross-cutting invariant), and `fromAssets` is a surface that only receives bytes.
 */

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { Pipeline } from "../../models/irodori.ts";
import { parseIrodoriPipelineConfig } from "../../models/src/irodori/config.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  ASSETS_DIR,
  FETCH_COMMAND,
  type GoldenCase,
  type GoldenMeta,
  goldenPresent,
  hasQuantSeat,
  loadLocalAssets,
  manifestText,
  MODEL,
  modelEntry,
  pipelineGoldenDir,
  readCase,
  readManifest,
  worstDifference,
} from "./helpers/irodori-assets.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import { UPSTREAM_PARITY, UPSTREAM_PARITY_OPT_IN } from "./helpers/upstream-parity.ts";
import { testAssetsFetched } from "@karume/test-support/test-assets";

/**
 * Whether `outputs/test-assets/` holds a complete fetch of the pin (ADR 0139). Then an absent golden
 * means the dataset at the pin lacks it, so its cases FAIL ({@link lostAsset}) instead of SKIPping.
 */
const FETCHED = testAssetsFetched();
const lostAsset = (paths: readonly string[]): Error =>
  new Error(
    `${paths.join(" / ")} absent, but outputs/test-assets/ holds a complete fetch of the pinned ` +
      "revision: the test-asset dataset at the pin lacks it",
  );
/** The SKIP hint for absent goldens. */
const FETCH_HINT = (seriesRoot: string): string =>
  `fetch: ${FETCH_COMMAND} — the assets gate accepts only a complete fetch; ` +
  `--include 'goldens/${seriesRoot}/pipeline/**' with KARUME_ALLOW_NO_ASSETS=1 runs this file alone`;

/**
 * The tolerance for comparing every element of `z` (**f32 series**).
 *
 * Measured (a plain comparison with `atol = rtol = 0` — the values this check logs every time;
 * 2026-08-12, real GPU):
 *
 * | Case     | S   | Elements | z maxAbs diff  | \|z\| max | Ratio     |
 * | -------- | --- | -------- | -------------- | ---------- | --------- |
 * | full     | 161 | 5,152    | **1.9050e-4**  | 5.10       | 1/26,800 of the range |
 * | no-ref   | 116 | 3,712    | **7.8994e-4**  | 4.33       | 1/5,480 of the range  |
 *
 * atol 5e-3 is **6.3 times** the measured worst 7.8994e-4 and 1/870 of the range (\|z\| max
 * 4.33). The difference is **structural**: the reduction order and rounding differ between the
 * golden side (torch's graph wrapper) and WGSL, accumulated over 40 Euler steps (S and the forward
 * count match exactly, so this is the only path difference left). **A wrong formula** (the CFG sign
 * or scale, the t schedule, the mask's interval split) shows at O(1), the same as the range — in
 * the golden side's measurement, CFG alone moves z by 5.56 / 3.23 (`cfgEffectMaxAbs` in
 * `meta.json`), more than 3 orders of magnitude away from this threshold.
 *
 * MUST: do not loosen this when it does not match. It is a number of the same character as the
 * upstream comparison's threshold (`EULER_REFERENCE_ATOL` = 1e-3 in
 * `tools/export-recipes/irodori/pipeline_ref.py`); moving it erases the meaning of "the port
 * works".
 */
const Z_ATOL = 5e-3;

/**
 * The tolerance of `z` for the **f16 series** (`--dtype f16` — only eligible weight slots stored
 * as f16, computed in f32).
 *
 * Measured (a plain comparison with `atol = rtol = 0` — the values this check logs every time;
 * 2026-08-12, real GPU):
 *
 * | Case     | S   | Elements | z maxAbs diff  | \|z\| max | Ratio    |
 * | -------- | --- | -------- | -------------- | ---------- | -------- |
 * | full     | 161 | 5,152    | 2.1321e-4      | 5.10272    | 1/23,933 |
 * | no-ref   | 116 | 3,712    | 1.8516e-4      | 4.33354    | 1/23,404 |
 *
 * 1.5e-3 = **7.0 times** the measured worst 2.1321e-4 (taken the same way as f32's 6.3 times),
 * about 1/3,400 of the \|z\| max. The golden is taken with the fake-quantized weights, so this
 * difference is also "implementation differences only", as in the f32 series — the measurement
 * being smaller than f32's (7.9e-4) is within chance; there is no ordering between series.
 *
 * MUST: **do not reuse the f32 value** (the pattern of ADR 0027 — derive independently from the raw
 * measurement per series; reusing it lets one re-derivation silently move the other). The
 * re-derivation procedure is the same as f32's: ① bake the f16 series' assets (the 3 exports with
 * `--dtype f16` + `karume dist`) ② set this constant back to `undefined`, run this check and
 * write the raw `maxAbs` from the log into the table ③ take 5–10 times the measured worst as the
 * threshold and add its ratio to the range.
 *
 * In an environment where only the assets are present with this still `undefined`, this check
 * **does not turn green** (it prints the measurement and stops red) — filling an underived
 * tolerance with "the same as f32 for now" and going green would silently add checks without
 * detection power.
 */
const F16_Z_ATOL: number | undefined = 1.5e-3;

/**
 * The tolerance of `z` for the **i8 seat** (`--dtype i8` — only eligible weight slots stored as
 * per-channel i8, computed in f32).
 *
 * Measured (a plain comparison with `atol = rtol = 0` — the values this check logs every time;
 * 2026-08-12, real GPU):
 *
 * | Case     | S   | Elements | z maxAbs diff  | \|z\| max | Ratio   |
 * | -------- | --- | -------- | -------------- | ---------- | ------- |
 * | full     | 161 | 5,152    | **1.8429e-3**  | 5.11886    | 1/2,777 |
 * | no-ref   | 116 | 3,712    | 1.6475e-4      | 4.33821    | 1/26,331 |
 *
 * 1e-2 = **5.4 times** the measured worst 1.8429e-3, about 1/512 of the \|z\| max. The difference
 * an order of magnitude larger than f16's is the coarseness of the i8 grid itself (the golden has
 * the same rounded weights, so the coefficient riding on the "implementation differences" grows) —
 * S / forwards match exactly, so the path is the same.
 *
 * MUST: **do not reuse the f32 / f16 values** (the pattern of ADR 0027 / 0029 — derive
 * independently from the raw measurement per series). The re-derivation procedure is the same as
 * f16's: ① bake the i8 series' assets (the 3 exports with `--dtype i8` + `karume dist`) ② set this
 * constant back to `undefined`, run this check and write the raw `maxAbs` from the log into the
 * table above ③ take 5–10 times the measured worst as the threshold and add its ratio to the range.
 *
 * In an environment where only the assets are present with this still `undefined`, this check
 * **does not turn green** (it prints the measurement and stops red — ADR 0050 decision 4).
 *
 * NOTE: **if S / forwards break, it is not a tolerance problem**, so read the S drift table of
 * `tools/export-recipes/irodori/measure_quant.py` before touching the threshold.
 */
const W8_Z_ATOL: number | undefined = 1e-2;

/**
 * The tolerance of `z` for the **i8+dit4 seat** (`--dtype i4` — `dit` alone is i4 g32 with GPTQ
 * calibration; the only mixed seat, whose other 7 roles share the same i8 bytes as `i8`).
 *
 * The golden is baked **from the shipped bytes** (`irodori.pipeline_ref` reads the exported i4
 * container back — calibration does not run twice), so this difference too is "implementation
 * differences only", as in the other series.
 *
 * Measured (a plain comparison with `atol = rtol = 0` — the values this check logs every time;
 * 2026-08-23, real GPU):
 *
 * | Case     | S   | Elements | z maxAbs diff  | \|z\| max | Ratio    |
 * | -------- | --- | -------- | -------------- | ---------- | -------- |
 * | full     | 161 | 5,152    | **1.1522e-3**  | 5.03657    | 1/4,371  |
 * | no-ref   | 116 | 3,712    | 2.8908e-5      | 4.99839    | 1/172,907 |
 *
 * 6e-3 = **5.2 times** the measured worst 1.1522e-3 (taken the same way as i8's 5.4 times), about
 * 1/839 of the \|z\| max. The measurement being smaller than i8's (1.8429e-3) is within chance,
 * not a consequence of the GPTQ calibration — there is no ordering between series (the same note
 * as i8's).
 *
 * MUST: **do not reuse the f32 / f16 / i8 values** (the pattern of ADR 0027 / 0050 — derive
 * independently from the raw measurement per series). The re-derivation procedure is the same as
 * i8's: ① bake the i4 series' assets (the export with `--dtype i4` + `pipeline_ref --dtype i4` +
 * `karume dist`) ② set this constant back to `undefined`, run this check and write the raw
 * `maxAbs` from the log into the table ③ take 5–10 times the measured worst as the threshold and
 * add its ratio to the range.
 *
 * NOTE: **if S / forwards break, it is not a tolerance problem** (the same as i8's note). That S
 * does not move with i4 rests on the measurement of J-2 stage 2 (the S prediction matched exactly
 * for all 3 gptq configurations — research 2026-08-20 §6), and S / forwards also matched exactly in
 * the measurement above.
 */
const W4_Z_ATOL: number | undefined = 6e-3;

/**
 * The utterance length (seconds) of the determinism check. S is fixed small without running
 * `duration`, and it generates twice.
 */
const DETERMINISM_SECONDS = 1;

/**
 * The series per storage dtype (the distribution's quant seat × the full-loop golden series — one
 * to one).
 *
 * MUST: keep the goldens in a separate place per series. Comparing f16 assets against the f32
 * golden mixes the quantization error itself into the difference and erases the tolerance's
 * meaning (and it only ever acts in the loosening direction, so the detection power drops while
 * staying green — the same trap as in `karume.quantize`'s module docstring).
 */
type IrodoriSeries = {
  /**
   * The name shown in test titles (= the spelling of the distribution's quant seat — a different
   * axis from the series root's spelling).
   */
  readonly name: string;
  /** The series root of its full-loop goldens (`goldens/<root>/pipeline/` of the test assets). */
  readonly seriesRoot: string;
  /**
   * The tolerance for comparing every element of `z` (**derived from measurement per series** —
   * `undefined` means not derived yet).
   */
  readonly zAtol: number | undefined;
};

const SERIES: readonly IrodoriSeries[] = [
  {
    name: "f32",
    seriesRoot: "irodori-v4-small",
    zAtol: Z_ATOL,
  },
  {
    name: "f16",
    seriesRoot: "irodori-v4-small-f16",
    zAtol: F16_Z_ATOL,
  },
  {
    // The seat is spelled `i8`, the series root `-i8` (the `i8-a8` seat points at the same bytes,
    // but activation quantization applies there, so it is not part of the numeric-parity net and
    // another check owns it).
    name: "i8",
    seriesRoot: "irodori-v4-small-i8",
    zAtol: W8_Z_ATOL,
  },
  {
    // The seat is spelled `i8+dit4`, the series root `-i4`. The only mixed seat: `dit` alone is
    // i4 and the other 7 roles share `i8`'s i8 bytes (the seat table's ruling 2026-08-23).
    // Generating the golden presupposes the exported series (`pipeline_ref --dtype i4` reads the
    // shipped bytes back).
    name: "i8+dit4",
    seriesRoot: "irodori-v4-small-i4",
    zAtol: W4_Z_ATOL,
  },
];

/**
 * Reproduces one golden case with the distribution and compares them.
 *
 * No knob is passed (the declarations of `pipelineConfig` are the effective values as is). Only
 * the reference latent and the initial noise come from the golden, and S is left to the
 * `duration` graph — a different S fails on the length of `initialNoise`, so this checks that
 * "the same path was taken, S included".
 */
const runCase = async (
  series: IrodoriSeries,
  pipeline: Pipeline,
  name: string,
  expected: GoldenCase,
): Promise<void> => {
  const goldenDir = pipelineGoldenDir(series.seriesRoot);
  const caseUrl = new URL(`case.${name}.safetensors`, goldenDir);
  if (!goldenPresent(caseUrl)) {
    throw new Error(
      `inconsistent golden set: ${new URL("meta.json", goldenDir).pathname} is present but ` +
        `${caseUrl.pathname} is absent`,
    );
  }
  const golden = await readCase(goldenDir, name);
  const started = performance.now();
  const latent = await pipeline.generateLatent({
    text: expected.text,
    ...(expected.caption === "" ? {} : { caption: expected.caption }),
    ...(expected.reference === null ? {} : { speaker: { latent: golden("reference_latent") } }),
    initialNoise: golden("noise"),
  });
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);
  const z = golden("z");
  const { maxAbs, at } = worstDifference(latent.data, z);
  console.log(
    `[e2e] irodori ${MODEL}/${series.name} ${name}: ${elapsed}s / S ${latent.frames} / ` +
      `forwards ${latent.forwards} / z maxAbs ${maxAbs.toExponential(4)} ` +
      `(atol ${series.zAtol ?? "not derived"} / |z| max ${expected.zAbsMax})`,
  );
  // The integer decisions (the formula deciding S, the t interval where CFG applies) must match
  // exactly whatever the series.
  assertEquals(latent.frames, expected.S, `${name}: S differs from the golden`);
  assertEquals(
    latent.forwards,
    expected.forwards,
    `${name}: dit's forward count differs from the golden`,
  );
  assertEquals(
    latent.latentDim,
    z.length / expected.S,
    `${name}: latentDim differs from the golden`,
  );
  if (series.zAtol === undefined) {
    throw new Error(
      `the tolerance of seat ${series.name} is not derived yet (measured maxAbs ${maxAbs.toExponential(4)} / ` +
        `|z| max ${expected.zAbsMax}) — collect the measurements in the log above for every case and fill ` +
        "in the table of this seat's `*_Z_ATOL` constant (do not reuse another seat's value — the pattern of ADR 0027)",
    );
  }
  if (maxAbs > series.zAtol) {
    const frame = Math.floor(at / latent.latentDim);
    throw new Error(
      `${name}: the latent differs from the golden by ${maxAbs.toExponential(4)} (tolerance ${series.zAtol})\n` +
        `  the worst is at frame ${frame} / column ${at % latent.latentDim}: ` +
        `golden ${z[at]} → measured ${latent.data[at]}\n` +
        "  instead of raising the tolerance, first check which of the CFG combination formula / t schedule / mask interval / " +
        "the conditions' right pad moved",
    );
  }
};

for (const series of SERIES) {
  const goldenDir = pipelineGoldenDir(series.seriesRoot);
  const metaUrl = new URL("meta.json", goldenDir);
  const metaText = await readTextIfPresent(metaUrl);
  const readMeta = (): GoldenMeta => {
    if (metaText === undefined) throw lostAsset([metaUrl.pathname]);
    return JSON.parse(metaText) as GoldenMeta;
  };
  /**
   * Whether the distribution has this seat (the seat spelling and the golden series are one to
   * one, but the spellings are separate axes).
   */
  const seated = hasQuantSeat(series.name);
  const mirrorReady = manifestText !== undefined && seated;
  // Every case reads the mirror's quant seat and `meta.json`; a parity case also reads its own
  // `case.<name>.safetensors`. An absent mirror SKIPs. An absent `meta.json` SKIPs unless the
  // test assets are a complete fetch of the pin, where `readMeta` FAILs instead (ADR 0139).
  const caseUrl = (name: string): URL => new URL(`case.${name}.safetensors`, goldenDir);
  const absentGoldens: readonly string[] =
    metaText === undefined
      ? [metaUrl, caseUrl("full"), caseUrl("no-ref")]
          .filter((url) => !goldenPresent(url))
          .map((url) => url.pathname)
      : [];
  const available = mirrorReady && (metaText !== undefined || FETCHED);
  if (!available) {
    const missing = [
      ...(mirrorReady
        ? []
        : [`the mirror ${ASSETS_DIR.pathname} with quant seat '${series.name}'`]),
      ...absentGoldens,
    ];
    console.warn(
      `[karume] Irodori latent e2e (${series.name}) SKIPs: missing ${missing.join(" / ")}` +
        (metaText === undefined ? ` (${FETCH_HINT(series.seriesRoot)})` : ""),
    );
  }

  const runnable = GPU_AVAILABLE && available;
  // The two golden cases compare the latent (S, forwards and z under a tolerance) with the torch
  // full-loop golden, so they are upstream parity and run only on opt-in. Determinism and the
  // pipelineConfig drift check stay in the default run. A case file absent beside a present
  // `meta.json` is an inconsistent golden set, so its parity case FAILs in `runCase`.
  const parityRunnable = runnable && UPSTREAM_PARITY;

  test.skipIf(!parityRunnable)(
    `e2e (real GPU): the distribution's latent matches the full-loop golden (${series.name} / with reference + caption)${UPSTREAM_PARITY_OPT_IN}`,
    async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, series.name);
      // `await using` also intends to check the release via [Symbol.asyncDispose] on this real-GPU
      // path.
      await using pipeline = await Pipeline.fromAssets(
        { manifest, assets },
        {
          model: MODEL,
          quant: series.name,
        },
      );
      await runCase(series, pipeline, "full", readMeta().cases["full"]);
    },
  );

  test.skipIf(!parityRunnable)(
    `e2e (real GPU): without a reference / with an empty caption the latent still matches the full-loop golden (${series.name})${UPSTREAM_PARITY_OPT_IN}`,
    async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, series.name);
      await using pipeline = await Pipeline.fromAssets(
        { manifest, assets },
        {
          model: MODEL,
          quant: series.name,
        },
      );
      // The path where the speaker's zero short-circuit and CFG off for the empty caption apply
      // (forwards drop from 100 to 60).
      await runCase(series, pipeline, "no-ref", readMeta().cases["no-ref"]);
    },
  );

  test.skipIf(!runnable)(
    `e2e (real GPU): two generations with the same seed are byte-identical (${series.name} / the RNG and the host glue are deterministic)`,
    async () => {
      const manifest = readManifest();
      const assets = await loadLocalAssets(manifest, series.name);
      await using pipeline = await Pipeline.fromAssets(
        { manifest, assets },
        {
          model: MODEL,
          quant: series.name,
        },
      );
      // Not passing `initialNoise` = the only path through the internal `Randn`. S is fixed with
      // `durationSeconds` to keep it short without running `duration` (what matters is
      // reproducibility, not length).
      const request = {
        text: readMeta().cases["full"].text,
        seed: 7,
        durationSeconds: DETERMINISM_SECONDS,
      };
      const first = await pipeline.generateLatent(request);
      const second = await pipeline.generateLatent(request);
      assertEquals(first.seed, 7);
      assertEquals(first.frames, DETERMINISM_SECONDS * readMeta().frameRate);
      assertEquals(
        new Uint8Array(second.data.buffer),
        new Uint8Array(first.data.buffer),
        "the latent changed with the same seed and the same request (the RNG or the host glue is non-deterministic)",
      );
    },
  );

  test.skipIf(!available)(
    `e2e (assets): the distribution's pipelineConfig matches the golden's generation conditions (${series.name})`,
    () => {
      // No GPU needed — it only checks the check's **premise** (that the golden regeneration and
      // the dist regeneration have not drifted apart).
      const meta = readMeta();
      const config = parseIrodoriPipelineConfig(modelEntry(readManifest()).pipelineConfig);
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
            `${drifted.join(" / ")}\n` +
            "(the check's premise is broken — before suspecting the z comparison, find out which side was re-baked)",
        );
      }
    },
  );
}

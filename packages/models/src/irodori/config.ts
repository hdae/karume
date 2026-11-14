/**
 * Schema validation of `pipelineConfig` (ADR 0038 §1 — each pipeline implementation owns and
 * validates its schema).
 *
 * The hub passes `pipelineConfig` through (it only strips forbidden keys and checks the size
 * limits). **This module is therefore the source of truth for the shape**, and every hand-written
 * check runs at parse time.
 *
 * MUST: unknown keys fail loudly (if a typo such as `{ steps: 40, step: 40 }` silently fell back to
 * the default, the run would differ from the default the distributor intended without anyone
 * noticing).
 * MUST: maps are read only through `Object.hasOwn` (cross-cutting invariant).
 *
 * ## MUST: model-specific numbers come from here (= the manifest), never from TS constants
 *
 * Irodori works correctly only when "the declared lengths of the graphs" and "the host formulas"
 * mesh (Tmax of the condition states, speaker row count, latent width, t_embed width, the clamp
 * range of S). Hard-coded in TS, a weight swap would leave only the host with old numbers,
 * producing a silent wrong value in a form where **the shapes still match**.
 * Matching against the real graphs is done on the assets by {@link Pipeline.fromAssets}.
 *
 * ## MUST: reject unsupported CFG modes at parse time, **keeping no value** (ADR 0047 decision 1)
 *
 * This implementation's uncond is "cond's state + an all-False mask over the segment", and it
 * matches upstream only with `speaker_uncond_mode="mask"` and `cfg_guidance_mode="independent"`.
 * The other modes (`"noise"` / `joint` / `alternating`) break the equivalence, so instead of
 * branching on them they are **not accepted** (the type can express only "mask" / "independent"
 * too).
 *
 * ## MUST: the sampler follows the declared flow parameterization (ADR 0150)
 *
 * `flowParameterization` names upstream's parameterization, spelled as upstream spells it:
 * `"rf_velocity"` (the rectified-flow Euler sampler with CFG — the v4 / v4.1 distributions, whose
 * manifests carry no such key, so **absent means `"rf_velocity"`**) or `"meanflow"` (the MeanFlow
 * student: one conditional forward per step with the interval embedding, no CFG). For
 * `"meanflow"` the RF-only keys (`initScale`, `cfgMinT`, `cfgMaxT`, `cfgScales`,
 * `speakerUncondMode`, `cfgGuidanceMode`) **must be absent** — upstream ignores them for such a
 * checkpoint, and karume does not accept a value it would ignore. The parsed type is a
 * discriminated union on `flowParameterization`, so the CFG fields cannot be read on the MeanFlow
 * branch.
 *
 * NOTE: the table of public distribution repositories ({@link SOURCES}) lives here too. It
 * is not one of the "model-specific numbers" the MUST above forbids but belongs to "which manifest
 * to fetch", a value a distribution cannot hold in the first place (ADR 0073).
 */

import type { HubRepoRef } from "@karume/hub";
import {
  assertAllowedKeys,
  isPositiveInteger,
  readNumber,
  readOnly,
  readRecord,
} from "../config/readers.ts";
import { maxSequenceLength } from "./host/round.ts";

/** The contract name of `pipeline`, and the major this implementation accepts (ADR 0038 §1). */
export const IRODORI_PIPELINE_NAME = "irodori";
export const IRODORI_PIPELINE_MAJOR = 1;

/**
 * **Table of public distribution repositories** of the Irodori family (ADR 0092 — one table per
 * family, **no default entry**). The values are **the sources this package version verified**
 * (pinned commit SHAs — ADR 0073).
 *
 * A key is the basename of the HF repository name without `karume-` (`"karume-" + key` gives the
 * repository basename back — the check in `tests/sources_test.ts` covers this invariant).
 * Generations and versions are separate repositories = separate entries, and old versions are
 * **kept** (the old repositories stay public as they are):
 *
 * - `"irodori-v4-small"` = `hdae/karume-irodori-v4-small`
 * - `"irodori-v4.1-small"` = `hdae/karume-irodori-v4.1-small` (upstream v4.1 = a follow-up version
 *   that retrained only the duration predictor)
 * - `"irodori-v4.1-small-mf"` = `hdae/karume-irodori-v4.1-small-mf` (upstream's MeanFlow student of
 *   v4.1 — 4 steps without CFG; ADR 0150)
 *
 * The utterance lengths come out differently, so v4 and v4.1 stand side by side as **separate
 * sources** rather than "old and new" (the example scripts defaulted to v4.1 — the examples were
 * removed in 0.15).
 *
 * Pass it **as the opt-in for following the package version automatically** — to pin
 * reproducibility yourself, write your own `{ repo, revision }` instead of this table
 * (`fromPretrained` has no default).
 *
 * MUST: a revision is pinned to a commit SHA — branches and tags can be moved by the distributor,
 * so the bytes this published package reads would change silently for network-side reasons (an
 * unrecoverable kind of accident). A SHA removes the revision-resolution request itself, so offline
 * startup with a complete cache also works (ADR 0038). A user who needs to follow main explicitly
 * chooses `{ ...SOURCES["irodori-v4.1-small"], revision: "main" }`.
 */
// NOTE: the revision is updated to the SHA of main after the upload, in §3 of the release runbook
// (docs/release-runbook.md) (inherits the maintenance duty of ADR 0073 decision 3 — by hand + the
// runbook check).
export const SOURCES = {
  "irodori-v4-small": {
    repo: "hdae/karume-irodori-v4-small",
    revision: "b90ab65f03c0198eb1561a83c6afd0624418af0d",
  },
  "irodori-v4.1-small": {
    repo: "hdae/karume-irodori-v4.1-small",
    revision: "8b54ca335c1ff16621ab5645fbf579ce2ec12ce4",
  },
  "irodori-v4.1-small-mf": {
    repo: "hdae/karume-irodori-v4.1-small-mf",
    revision: "fc4ba31e2f9ea5bc03564758fc98a17805216ace",
  },
} as const satisfies Record<string, HubRepoRef>;

/** The flow parameterizations a distribution can declare (upstream's spellings — ADR 0150). */
export const IRODORI_FLOW_RF = "rf_velocity";
export const IRODORI_FLOW_MEANFLOW = "meanflow";
type IrodoriFlow = typeof IRODORI_FLOW_RF | typeof IRODORI_FLOW_MEANFLOW;

const FLOW_KEY = "flowParameterization";

/** The keys every declaration has, whatever its flow parameterization. */
const COMMON_KEYS: readonly string[] = [
  "maxTextLen",
  "maxCaptionLen",
  "speakerRows",
  "ditSymMax",
  "frameRate",
  "sampleRate",
  "hopLength",
  "codecHaloFrames",
  "latentDim",
  "speakerPatchSize",
  "speakerDim",
  "textDim",
  "captionDim",
  "timestepEmbedDim",
  "steps",
  "minSeconds",
  "maxSeconds",
];

/**
 * The keys only the RF sampler reads (a MeanFlow declaration that carries one fails —
 * ADR 0150).
 */
const RF_KEYS: readonly string[] = [
  "initScale",
  "cfgMinT",
  "cfgMaxT",
  "cfgScales",
  "speakerUncondMode",
  "cfgGuidanceMode",
];

const CFG_SCALE_KEYS: readonly string[] = ["text", "speaker", "caption"];

/** The only way of building uncond this implementation accepts (ADR 0047 decision 1). */
const SPEAKER_UNCOND_MODE = "mask";
/** The only CFG combination this implementation accepts (`v_cond + Σ s_k(v_cond − v_k)`). */
const CFG_GUIDANCE_MODE = "independent";

/** CFG strength per condition. Combination order fixed: text → speaker → caption (ADR 0047). */
type IrodoriCfgScales = {
  readonly text: number;
  readonly speaker: number;
  readonly caption: number;
};

/** The fields every declaration has (model-specific numbers, step count and seconds clamp). */
type IrodoriCommonConfig = {
  /**
   * Maximum text token length (BOS included). Also the declared length of `text_state` of
   * `dit_context`.
   */
  readonly maxTextLen: number;
  /**
   * Maximum caption token length (BOS included). Also the declared length of `caption_state` of
   * `dit_context`.
   */
  readonly maxCaptionLen: number;
  /**
   * Declared row count of `speaker_state` of `dit_context` (the reference-latent maximum + 1 mean
   * token).
   */
  readonly speakerRows: number;
  /** Maximum of the symbolic dimension S of `dit`. */
  readonly ditSymMax: number;
  /** Codec frame rate (Hz) — the conversion between seconds and latent frames. */
  readonly frameRate: number;
  /** Sampling rate of the waveform (Hz). Needed for the conversion from seconds to samples. */
  readonly sampleRate: number;
  /** Number of samples one latent frame decodes into (the decoder's output ratio). */
  readonly hopLength: number;
  /**
   * Number of latent frames added on both sides of the kept span when the codec decoder is tiled.
   *
   * A **model constant** that comes from the decoder's receptive field (13,793 samples per side =
   * 7.19 frames), not a runtime knob. Discarding this much halo makes the inside of a tile
   * **bit-identical** to a full-length decode (measured by the real-GPU check
   * `packages/e2e/tests/e2e_irodori_codec_test.ts`).
   */
  readonly codecHaloFrames: number;
  /** Width of the patched latent (the last dimension of `x_t`). */
  readonly latentDim: number;
  /** Patch width along time when the reference latent is passed to the speaker encoder. */
  readonly speakerPatchSize: number;
  readonly speakerDim: number;
  readonly textDim: number;
  readonly captionDim: number;
  /**
   * Width of `t_embed` (first half cos / second half sin) — and of the MeanFlow step's
   * `delta_embed`, the same embedding of the interval length.
   */
  readonly timestepEmbedDim: number;
  /** Number of sampler steps (Euler steps for RF, MeanFlow steps for MeanFlow). */
  readonly steps: number;
  /** Seconds that decide the clamp range of S. */
  readonly minSeconds: number;
  readonly maxSeconds: number;
};

/** The rectified-flow Euler sampler with independent CFG (v4 / v4.1). */
type IrodoriRfSampler = {
  /** Absent in the manifest means this value (the published v4 / v4.1 manifests carry none). */
  readonly flowParameterization: typeof IRODORI_FLOW_RF;
  /** Initial scale of the t schedule (`t_i = initScale·(1 − i/steps)`). */
  readonly initScale: number;
  /** Lower / upper bound of the t where CFG applies (both ends included). */
  readonly cfgMinT: number;
  readonly cfgMaxT: number;
  readonly cfgScales: IrodoriCfgScales;
  /**
   * The only value ADR 0047 accepts. **Held to check the distribution's declaration, not to branch
   * on**.
   */
  readonly speakerUncondMode: typeof SPEAKER_UNCOND_MODE;
  readonly cfgGuidanceMode: typeof CFG_GUIDANCE_MODE;
};

/**
 * The MeanFlow sampler (ADR 0150): t runs from 1 to 0 in `steps` equal intervals of upstream's
 * f32 `linspace`, each step one conditional forward that also takes the interval embedding. It
 * has no knob of its own (no CFG, no initial scale).
 */
type IrodoriMeanFlowSampler = {
  readonly flowParameterization: typeof IRODORI_FLOW_MEANFLOW;
};

/** A parsed RF declaration (the branch whose CFG fields can be read). */
export type IrodoriRfPipelineConfig = IrodoriCommonConfig & IrodoriRfSampler;
/** A parsed MeanFlow declaration. */
export type IrodoriMeanFlowPipelineConfig = IrodoriCommonConfig & IrodoriMeanFlowSampler;
/** A parsed `pipelineConfig` — a discriminated union on `flowParameterization` (ADR 0150). */
export type IrodoriPipelineConfig = IrodoriRfPipelineConfig | IrodoriMeanFlowPipelineConfig;

const isPositiveFinite = (value: number): boolean => Number.isFinite(value) && value > 0;
const isNonNegativeFinite = (value: number): boolean => Number.isFinite(value) && value >= 0;
/**
 * Acceptance condition of the initial scale of the t schedule — a positive finite number that
 * **stays finite when rounded to f32**.
 *
 * MUST: being finite in double precision is not enough. The t sequence is built as
 * `f32(initScale · f32(1 − i/steps))` and passed to the GPU (`t_embed` and the step size), so a
 * value above the f32 maximum (e.g. 1e39) turns the head of the t sequence into ±Inf, which does
 * not surface until the non-finite check of the output (after the DiT has run to the end). Every
 * value of the t sequence is at most `f32(initScale)`, so one check here makes every element of the
 * sequence finite (no scan needed).
 * NOTE: a magnitude that is finite in f32 but overflows in the graph's arithmetic is a separate
 * problem and is not constrained here.
 */
const isPositiveF32Finite = (value: number): boolean =>
  isPositiveFinite(value) && Number.isFinite(Math.fround(value));
/**
 * Acceptance condition of a CFG strength — a non-negative finite number that is **exactly
 * representable in f32**.
 *
 * MUST: f32 exactness is required at the declaration step. JS numbers are f64; the host path's
 * `combineCfg` feeds the f64 scale into the multiply as is, `f32(scale * f32(base − v))`, while the
 * resident path rounds the scale to f32 and the GPU multiplies in f32. With a value that is not
 * exact in f32 (e.g. 1.3) the two split by 1 to 2 ulp, and the MUST "the outputs of the 2 paths are
 * the same" (the DiT loop branch in `pipeline.ts`) holds only conditionally. Rejecting it on the
 * distribution's declaration removes any room for f64 on the host side in the first place (the
 * real distributions' 3.0 / 5.0 / 3.0 are all exact in f32).
 */
const isNonNegativeF32 = (value: number): boolean =>
  isNonNegativeFinite(value) && value === Math.fround(value);

const parseCfgScales = (raw: unknown): IrodoriCfgScales => {
  const where = "pipelineConfig.cfgScales";
  const record = readRecord(raw, where);
  assertAllowedKeys(record, CFG_SCALE_KEYS, where);
  // 0 is a regular value meaning "do not run CFG for that condition" (upstream
  // `has_*_cfg = scale > 0`).
  const requirement = "not a non-negative finite number, or not exactly representable in f32";
  return {
    text: readNumber(record, "text", where, isNonNegativeF32, requirement),
    speaker: readNumber(record, "speaker", where, isNonNegativeF32, requirement),
    caption: readNumber(record, "caption", where, isNonNegativeF32, requirement),
  };
};

/**
 * Reads the declared flow parameterization (absent = {@link IRODORI_FLOW_RF}; anything else than
 * the two exact spellings fails).
 */
const readFlow = (raw: Readonly<Record<string, unknown>>, where: string): IrodoriFlow => {
  if (!Object.hasOwn(raw, FLOW_KEY)) return IRODORI_FLOW_RF;
  const value = raw[FLOW_KEY];
  if (value === IRODORI_FLOW_RF || value === IRODORI_FLOW_MEANFLOW) return value;
  throw new Error(
    `${where}.${FLOW_KEY}: '${String(value)}' is not '${IRODORI_FLOW_RF}' or '${IRODORI_FLOW_MEANFLOW}'`,
  );
};

/** Reads the RF sampler's knobs (the declaration has no `flowParameterization`, or `"rf_velocity"`). */
const parseRfSampler = (
  raw: Readonly<Record<string, unknown>>,
  where: string,
): IrodoriRfSampler => {
  const sampler: IrodoriRfSampler = {
    flowParameterization: IRODORI_FLOW_RF,
    initScale: readNumber(
      raw,
      "initScale",
      where,
      isPositiveF32Finite,
      "not a positive finite number, or not finite in f32",
    ),
    cfgMinT: readNumber(raw, "cfgMinT", where, Number.isFinite, "not a finite number"),
    cfgMaxT: readNumber(raw, "cfgMaxT", where, Number.isFinite, "not a finite number"),
    cfgScales: parseCfgScales(Object.hasOwn(raw, "cfgScales") ? raw["cfgScales"] : undefined),
    speakerUncondMode: readOnly(
      raw,
      "speakerUncondMode",
      where,
      SPEAKER_UNCOND_MODE,
      "uncond is expressed as 'the cond state + every mask of that segment False' (ADR 0047 decision 1)",
    ),
    cfgGuidanceMode: readOnly(
      raw,
      "cfgGuidanceMode",
      where,
      CFG_GUIDANCE_MODE,
      "the CFG combination is limited to the form `v_cond + Σ s_k(v_cond − v_k)` (ADR 0047 decision 1)",
    ),
  };
  // A declaration with an empty / reversed interval stays silent as "CFG never applies" (only the
  // forward count goes down).
  if (sampler.cfgMinT > sampler.cfgMaxT) {
    throw new Error(
      `${where}: cfgMinT ${sampler.cfgMinT} is greater than cfgMaxT ${sampler.cfgMaxT} (CFG would never apply)`,
    );
  }
  return sampler;
};

/** Checks and reads the manifest's `pipelineConfig` (the raw value the hub passed through). */
export const parseIrodoriPipelineConfig = (
  raw: Readonly<Record<string, unknown>>,
): IrodoriPipelineConfig => {
  const where = "pipelineConfig";
  const flow = readFlow(raw, where);
  if (flow === IRODORI_FLOW_MEANFLOW) {
    // MUST: a value MeanFlow would ignore is refused by name, before the generic unknown-key check
    // (whose message would not say why).
    const ignored = RF_KEYS.filter((key) => Object.hasOwn(raw, key));
    if (ignored.length > 0) {
      throw new Error(
        `${where}: a '${IRODORI_FLOW_MEANFLOW}' distribution declares the RF-only key(s)` +
          ` ${ignored.map((key) => `'${key}'`).join(", ")} — the MeanFlow sampler has no CFG and no` +
          " initial scale, so the value would be ignored (ADR 0150)",
      );
    }
  }
  assertAllowedKeys(
    raw,
    flow === IRODORI_FLOW_MEANFLOW
      ? [...COMMON_KEYS, FLOW_KEY]
      : [...COMMON_KEYS, ...RF_KEYS, FLOW_KEY],
    where,
  );
  const positive = "not a positive integer";
  // A field that loses one entry to a leading token (BOS for text / caption, the mean token for
  // speaker) needs at least 2 so that the body keeps a budget. Accepting 1 gives text a body of 0
  // tokens and speaker 0 rows of reference latent, and the failure surfaces with an unrelated
  // message such as "maximum reference-audio samples 0" → "the waveform is empty" from
  // `integratedLoudness`.
  const withLeadingToken = (value: number): boolean => Number.isInteger(value) && value >= 2;
  const common: IrodoriCommonConfig = {
    maxTextLen: readNumber(raw, "maxTextLen", where, withLeadingToken, "not an integer >= 2"),
    maxCaptionLen: readNumber(raw, "maxCaptionLen", where, withLeadingToken, "not an integer >= 2"),
    speakerRows: readNumber(raw, "speakerRows", where, withLeadingToken, "not an integer >= 2"),
    ditSymMax: readNumber(raw, "ditSymMax", where, isPositiveInteger, positive),
    frameRate: readNumber(raw, "frameRate", where, isPositiveInteger, positive),
    sampleRate: readNumber(raw, "sampleRate", where, isPositiveInteger, positive),
    hopLength: readNumber(raw, "hopLength", where, isPositiveInteger, positive),
    codecHaloFrames: readNumber(raw, "codecHaloFrames", where, isPositiveInteger, positive),
    latentDim: readNumber(raw, "latentDim", where, isPositiveInteger, positive),
    speakerPatchSize: readNumber(raw, "speakerPatchSize", where, isPositiveInteger, positive),
    speakerDim: readNumber(raw, "speakerDim", where, isPositiveInteger, positive),
    textDim: readNumber(raw, "textDim", where, isPositiveInteger, positive),
    captionDim: readNumber(raw, "captionDim", where, isPositiveInteger, positive),
    // Split into a cos half / a sin half, so an odd width cannot be built.
    timestepEmbedDim: readNumber(
      raw,
      "timestepEmbedDim",
      where,
      (value) => isPositiveInteger(value) && value % 2 === 0,
      "not a positive even number",
    ),
    steps: readNumber(raw, "steps", where, isPositiveInteger, positive),
    minSeconds: readNumber(
      raw,
      "minSeconds",
      where,
      isPositiveFinite,
      "not a positive finite number",
    ),
    maxSeconds: readNumber(
      raw,
      "maxSeconds",
      where,
      isPositiveFinite,
      "not a positive finite number",
    ),
  };
  const config: IrodoriPipelineConfig =
    flow === IRODORI_FLOW_MEANFLOW
      ? { ...common, flowParameterization: IRODORI_FLOW_MEANFLOW }
      : { ...common, ...parseRfSampler(raw, where) };
  if (config.minSeconds > config.maxSeconds) {
    throw new Error(
      `${where}: minSeconds ${config.minSeconds} is greater than maxSeconds ${config.maxSeconds}`,
    );
  }
  // The two conversions, seconds → frames (`frameRate`) and seconds → samples → frames
  // (`sampleRate` / `hopLength`), must not move independently. If they diverge, the way S is
  // decided changes per path, and only the cut length quietly disagrees (no exception).
  if (config.sampleRate !== config.frameRate * config.hopLength) {
    throw new Error(
      `${where}: sampleRate ${config.sampleRate} differs from frameRate ${config.frameRate} × hopLength` +
        ` ${config.hopLength} = ${config.frameRate * config.hopLength}`,
    );
  }
  // A declaration where the S reachable from the maximum seconds exceeds the declared maximum of
  // dit fails **only for long utterances**, with "the decided latent length exceeds the declared
  // maximum of dit" (after running up to the duration step). The reachable maximum differs by the
  // rounding of the 2 paths, so it is read from the one function in `host/round.ts` (the formula is
  // not copied).
  const reachable = maxSequenceLength(config);
  if (reachable > config.ditSymMax) {
    throw new Error(
      `${where}: maxSeconds ${config.maxSeconds} can reach latent length ${reachable}, which` +
        ` exceeds ditSymMax ${config.ditSymMax}`,
    );
  }
  return config;
};

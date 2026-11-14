// Behaviour tests of the schema validation of `pipelineConfig` (irodori). Needs neither a GPU nor
// real assets.
//
// Three things are pinned here:
//  ① unknown keys / missing keys / out-of-range values fail **at parse time** (no path where a
//     misspelling in the distribution silently degrades to a default — the doc of
//     `src/irodori/config.ts`)
//  ② `speakerUncondMode` / `cfgGuidanceMode` have **an accepted set of 1 value** (ADR 0047
//     decision 1). An unsupported mode is **rejected** rather than "kept and branched on", and it
//     cannot be expressed as a type either
//  ③ a reversed range declaration (cfgMinT/cfgMaxT, minSeconds/maxSeconds) fails — a reversed
//     range stays silent as "CFG is never applied", only reducing the forward count without any
//     exception

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import {
  IRODORI_PIPELINE_MAJOR,
  IRODORI_PIPELINE_NAME,
  type IrodoriPipelineConfig,
  type IrodoriRfPipelineConfig,
  parseIrodoriPipelineConfig,
} from "../src/irodori/config.ts";

/**
 * The values the real v4-small weights require (the skeleton the distribution is expected to
 * write).
 */
const BASE: Record<string, unknown> = {
  maxTextLen: 256,
  maxCaptionLen: 512,
  speakerRows: 751,
  ditSymMax: 750,
  frameRate: 25,
  sampleRate: 48000,
  hopLength: 1920,
  codecHaloFrames: 8,
  latentDim: 32,
  speakerPatchSize: 4,
  speakerDim: 768,
  textDim: 512,
  captionDim: 512,
  timestepEmbedDim: 512,
  steps: 40,
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgScales: { text: 3, speaker: 5, caption: 3 },
  minSeconds: 0.5,
  maxSeconds: 30,
  speakerUncondMode: "mask",
  cfgGuidanceMode: "independent",
};

const withPatch = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...BASE,
  ...patch,
});

const without = (key: string): Record<string, unknown> => {
  const { [key]: _removed, ...rest } = BASE;
  return rest;
};

/**
 * Narrows a parsed declaration to the RF branch (the shape of `BASE`, which declares no
 * `flowParameterization` — absent means RF, ADR 0150) and fails if it is not.
 */
const assertRf: (config: IrodoriPipelineConfig) => asserts config is IrodoriRfPipelineConfig = (
  config,
) => {
  assertEquals(config.flowParameterization, "rf_velocity");
};

test("the contract name and major are fixed by this implementation (what the manifest's spelling is checked against)", () => {
  assertEquals(IRODORI_PIPELINE_NAME, "irodori");
  assertEquals(IRODORI_PIPELINE_MAJOR, 1);
});

test("parseIrodoriPipelineConfig: reads the real weights' values completely", () => {
  const config = parseIrodoriPipelineConfig(BASE);
  assertRf(config);
  assertEquals(config.maxTextLen, 256);
  assertEquals(config.speakerRows, 751);
  assertEquals(config.cfgScales, { text: 3, speaker: 5, caption: 3 });
  assertEquals(config.speakerUncondMode, "mask");
  assertEquals(config.cfgGuidanceMode, "independent");
});

test("parseIrodoriPipelineConfig: rejects unknown keys (no silent acceptance of misspellings)", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ step: 40 })),
    Error,
    "unknown key 'step'",
  );
});

test("parseIrodoriPipelineConfig: also rejects unknown keys of cfgScales", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ cfgScales: { text: 3, speaker: 5, style: 3 } })),
    Error,
    "unknown key 'style'",
  );
});

test("parseIrodoriPipelineConfig: a missing key fails naming the field", () => {
  for (const key of Object.keys(BASE)) {
    assertThrows(
      () => parseIrodoriPipelineConfig(without(key)),
      Error,
      key === "cfgScales" ? "pipelineConfig.cfgScales" : `pipelineConfig.${key}`,
    );
  }
});

test("parseIrodoriPipelineConfig: rejects an unsupported speakerUncondMode (ADR 0047 decision 1)", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ speakerUncondMode: "noise" })),
    Error,
    "only 'mask'",
  );
});

test("parseIrodoriPipelineConfig: rejects an unsupported cfgGuidanceMode (ADR 0047 decision 1)", () => {
  for (const mode of ["joint", "alternating"]) {
    assertThrows(
      () => parseIrodoriPipelineConfig(withPatch({ cfgGuidanceMode: mode })),
      Error,
      "only 'independent'",
    );
  }
});

test("parseIrodoriPipelineConfig: checks the value ranges of widths and lengths", () => {
  // The token sequence always places one BOS, so 1, which leaves no budget for the text, cannot be
  // built.
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ maxTextLen: 1 })), Error, ">= 2");
  // t_embed splits into cos in the first half and sin in the second, so an odd width cannot be
  // built.
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ timestepEmbedDim: 511 })),
    Error,
    "not a positive even number",
  );
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ latentDim: 0 })),
    Error,
    "positive integer",
  );
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ initScale: 0 })),
    Error,
    "not a positive finite number",
  );
  assertThrows(
    () =>
      parseIrodoriPipelineConfig(withPatch({ cfgScales: { text: -1, speaker: 5, caption: 3 } })),
    Error,
    "not a non-negative finite number",
  );
});

test("parseIrodoriPipelineConfig: initScale accepts only values that stay finite when rounded to f32", () => {
  // Finite in double precision but above f32's maximum (about 3.4e38), the head of the t sequence
  // would reach the GPU as ±Inf and surface only at the output check after the DiT has run to the
  // end (each value of the t sequence is at most f32(initScale)).
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ initScale: 1e39 })),
    Error,
    "not finite in f32",
  );
  // f32's maximum itself is finite, so it passes (the boundary is "whether the rounded result is
  // finite").
  const largest = 3.4028234663852886e38;
  const config = parseIrodoriPipelineConfig(withPatch({ initScale: largest }));
  assertRf(config);
  assertEquals(config.initScale, largest);
});

test("parseIrodoriPipelineConfig: fails when the CFG window is reversed (CFG would vanish silently)", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ cfgMinT: 1, cfgMaxT: 0.5 })),
    Error,
    "CFG would never apply",
  );
});

test("parseIrodoriPipelineConfig: fails when the seconds range is reversed", () => {
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ minSeconds: 30, maxSeconds: 0.5 })),
    Error,
    "minSeconds 30",
  );
});

test("parseIrodoriPipelineConfig: the 3 codec fields are positive integers (0 and fractions fail)", () => {
  for (const key of ["sampleRate", "hopLength", "codecHaloFrames"]) {
    for (const value of [0, -1, 1.5, "48000"]) {
      assertThrows(
        () => parseIrodoriPipelineConfig(withPatch({ [key]: value })),
        Error,
        `pipelineConfig.${key}`,
      );
    }
  }
});

test("parseIrodoriPipelineConfig: fails when sampleRate ÷ hopLength differs from frameRate", () => {
  // No form where the two paths, seconds → frames and seconds → samples → frames, move
  // independently.
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ hopLength: 1000 })),
    Error,
    "frameRate 25 × hopLength 1000",
  );
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ frameRate: 24 })),
    Error,
    "sampleRate 48000",
  );
});

test("parseIrodoriPipelineConfig: rejects speakerRows 1 (the reference latent would have 0 rows)", () => {
  // The speaker condition is "1 mean token + the patched reference", so with 1 row no reference row
  // fits. Passed through, the reference audio's maximum sample count becomes 0, and generation
  // fails part-way with `integratedLoudness`'s **unrelated message** "the waveform is empty". The
  // same reasoning and the same accepted set as maxTextLen / maxCaptionLen.
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ speakerRows: 1 })), Error, ">= 2");
  assertThrows(() => parseIrodoriPipelineConfig(withPatch({ speakerRows: 0 })), Error, ">= 2");
});

test("parseIrodoriPipelineConfig: fails when the S reachable from maxSeconds exceeds ditSymMax", () => {
  // Passed through, only long utterances would run up to the duration stage before failing with
  // "the decided latent length exceeds the dit's declared limit" (short utterances never reveal
  // it).
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ maxSeconds: 30.5 })),
    Error,
    "can reach latent length 763",
  );
  // A declaration where only the manual-seconds path
  // (`ceil(trunc(seconds×sampleRate) / hopLength)`) comes out 1 frame higher. A formula that looks
  // only at the duration path's `floor(seconds×frameRate)` misses it.
  assertThrows(
    () => parseIrodoriPipelineConfig(withPatch({ maxSeconds: 30.02 })),
    Error,
    "can reach latent length 751",
  );
  // The real distribution's 30 seconds / ditSymMax 750 holds exactly (the boundary is not too
  // tight).
  assertEquals(parseIrodoriPipelineConfig(withPatch({})).ditSymMax, 750);
});

test("parseIrodoriPipelineConfig: cfgScales 0 is a valid value (no CFG for that condition)", () => {
  const config = parseIrodoriPipelineConfig(
    withPatch({ cfgScales: { text: 3, speaker: 0, caption: 0 } }),
  );
  assertRf(config);
  assertEquals(config.cfgScales.speaker, 0);
});

test("parseIrodoriPipelineConfig: a cfgScale not exactly representable in f32 fails at the declaration stage", () => {
  // 1.3 differs between f64 and f32. The host path (f64 multiplication) and the resident path (f32
  // multiplication) would split by 1–2 ulp and make the MUST "the 2 paths give the same output"
  // conditional, so it is rejected on the distribution's declaration side.
  assertThrows(
    () =>
      parseIrodoriPipelineConfig(withPatch({ cfgScales: { text: 1.3, speaker: 5, caption: 3 } })),
    Error,
    "not exactly representable in f32",
  );
  // The real distribution's 3 / 5 / 3, and non-integers exact in f32 (sums of powers of 2), pass.
  const config = parseIrodoriPipelineConfig(
    withPatch({ cfgScales: { text: 1.25, speaker: 5, caption: 3 } }),
  );
  assertRf(config);
  assertEquals(config.cfgScales.text, 1.25);
});

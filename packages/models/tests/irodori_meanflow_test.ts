// The MeanFlow sampler of irodori (ADR 0150) without a GPU: the declaration (`pipelineConfig` with
// `flowParameterization: "meanflow"`), the t sequence and the intervals the loop feeds `dit`, and the
// loop material (one conditional forward per step — no CFG variant).
//
// The schedule is pinned to the **f32 values torch produces** for several step counts, not only the
// default 4 — torch's f32 `linspace` is not the closed form `1 − i/steps`, and the TS side must
// reproduce upstream's `sample_euler_meanflow` for whatever step count a distribution declares.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type IrodoriPipelineConfig, parseIrodoriPipelineConfig } from "../src/irodori/config.ts";
import { meanflowIntervals, meanflowSchedule, tSchedule } from "../src/irodori/host/sampler.ts";
import { ditSampler } from "../src/irodori/pipeline.ts";

/** The common fields of the real v4.1-small-mf declaration (`models/karume-irodori-v4.1-small-mf`). */
const COMMON: Record<string, unknown> = {
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
  minSeconds: 0.5,
  maxSeconds: 30,
};

/** The MeanFlow declaration (4 steps, no RF knob). */
const MEANFLOW: Record<string, unknown> = { ...COMMON, flowParameterization: "meanflow", steps: 4 };

/** The RF knobs a MeanFlow declaration must not carry (with the values the v4 / v4.1 manifests hold). */
const RF_KNOBS: Record<string, unknown> = {
  initScale: 0.999,
  cfgMinT: 0.5,
  cfgMaxT: 1,
  cfgScales: { text: 3, speaker: 5, caption: 3 },
  speakerUncondMode: "mask",
  cfgGuidanceMode: "independent",
};

/** The RF declaration of the published v4 / v4.1 manifests (no `flowParameterization`). */
const RF: Record<string, unknown> = { ...COMMON, steps: 40, ...RF_KNOBS };

/**
 * `torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)` and its differences
 * `t[:-1] − t[1:]`, as f32 bit patterns — generated with torch 2.13.0 on the CPU (AVX512,
 * 2026-11-14) by the same expression `irodori.pipeline_ref.meanflow_schedule` uses.
 */
const TORCH: Readonly<
  Record<number, { readonly t: readonly number[]; readonly delta: readonly number[] }>
> = {
  1: {
    t: [0x3f800000, 0x00000000],
    delta: [0x3f800000],
  },
  2: {
    t: [0x3f800000, 0x3f000000, 0x00000000],
    delta: [0x3f000000, 0x3f000000],
  },
  3: {
    t: [0x3f800000, 0x3f2aaaaa, 0x3eaaaaab, 0x00000000],
    delta: [0x3eaaaaac, 0x3eaaaaa9, 0x3eaaaaab],
  },
  4: {
    t: [0x3f800000, 0x3f400000, 0x3f000000, 0x3e800000, 0x00000000],
    delta: [0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000],
  },
  5: {
    t: [0x3f800000, 0x3f4ccccd, 0x3f19999a, 0x3ecccccd, 0x3e4ccccd, 0x00000000],
    delta: [0x3e4ccccc, 0x3e4ccccc, 0x3e4cccce, 0x3e4ccccd, 0x3e4ccccd],
  },
  6: {
    t: [0x3f800000, 0x3f555555, 0x3f2aaaaa, 0x3f000000, 0x3eaaaaab, 0x3e2aaaab, 0x00000000],
    delta: [0x3e2aaaac, 0x3e2aaaac, 0x3e2aaaa8, 0x3e2aaaaa, 0x3e2aaaab, 0x3e2aaaab],
  },
  7: {
    t: [
      0x3f800000, 0x3f5b6db7, 0x3f36db6e, 0x3f124924, 0x3edb6db8, 0x3e924925, 0x3e124925,
      0x00000000,
    ],
    delta: [0x3e124924, 0x3e124924, 0x3e124928, 0x3e124920, 0x3e124926, 0x3e124925, 0x3e124925],
  },
  8: {
    t: [
      0x3f800000, 0x3f600000, 0x3f400000, 0x3f200000, 0x3f000000, 0x3ec00000, 0x3e800000,
      0x3e000000, 0x00000000,
    ],
    delta: [
      0x3e000000, 0x3e000000, 0x3e000000, 0x3e000000, 0x3e000000, 0x3e000000, 0x3e000000,
      0x3e000000,
    ],
  },
  10: {
    t: [
      0x3f800000, 0x3f666666, 0x3f4ccccd, 0x3f333333, 0x3f19999a, 0x3f000000, 0x3ecccccd,
      0x3e99999a, 0x3e4ccccd, 0x3dcccccd, 0x00000000,
    ],
    delta: [
      0x3dccccd0, 0x3dccccc8, 0x3dccccd0, 0x3dccccc8, 0x3dccccd0, 0x3dcccccc, 0x3dcccccc,
      0x3dccccce, 0x3dcccccd, 0x3dcccccd,
    ],
  },
  12: {
    t: [
      0x3f800000, 0x3f6aaaab, 0x3f555555, 0x3f400000, 0x3f2aaaaa, 0x3f155555, 0x3f000000,
      0x3ed55556, 0x3eaaaaab, 0x3e800000, 0x3e2aaaab, 0x3daaaaab, 0x00000000,
    ],
    delta: [
      0x3daaaaa8, 0x3daaaab0, 0x3daaaaa8, 0x3daaaab0, 0x3daaaaa8, 0x3daaaaa8, 0x3daaaaa8,
      0x3daaaaac, 0x3daaaaac, 0x3daaaaaa, 0x3daaaaab, 0x3daaaaab,
    ],
  },
  16: {
    t: [
      0x3f800000, 0x3f700000, 0x3f600000, 0x3f500000, 0x3f400000, 0x3f300000, 0x3f200000,
      0x3f100000, 0x3f000000, 0x3ee00000, 0x3ec00000, 0x3ea00000, 0x3e800000, 0x3e400000,
      0x3e000000, 0x3d800000, 0x00000000,
    ],
    delta: [
      0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000,
      0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000, 0x3d800000,
      0x3d800000, 0x3d800000,
    ],
  },
  25: {
    t: [
      0x3f800000, 0x3f75c28f, 0x3f6b851f, 0x3f6147ae, 0x3f570a3e, 0x3f4ccccd, 0x3f428f5c,
      0x3f3851ec, 0x3f2e147b, 0x3f23d70a, 0x3f19999a, 0x3f0f5c29, 0x3f051eb8, 0x3ef5c28f,
      0x3ee147ae, 0x3ecccccc, 0x3eb851eb, 0x3ea3d70a, 0x3e8f5c29, 0x3e75c28f, 0x3e4ccccc,
      0x3e23d70a, 0x3df5c28f, 0x3da3d70a, 0x3d23d70a, 0x00000000,
    ],
    delta: [
      0x3d23d710, 0x3d23d700, 0x3d23d710, 0x3d23d700, 0x3d23d710, 0x3d23d710, 0x3d23d700,
      0x3d23d710, 0x3d23d710, 0x3d23d700, 0x3d23d710, 0x3d23d710, 0x3d23d708, 0x3d23d708,
      0x3d23d710, 0x3d23d708, 0x3d23d708, 0x3d23d708, 0x3d23d70c, 0x3d23d70c, 0x3d23d708,
      0x3d23d70a, 0x3d23d70a, 0x3d23d70a, 0x3d23d70a,
    ],
  },
  40: {
    t: [
      0x3f800000, 0x3f79999a, 0x3f733333, 0x3f6ccccd, 0x3f666666, 0x3f600000, 0x3f59999a,
      0x3f533333, 0x3f4ccccd, 0x3f466666, 0x3f400000, 0x3f39999a, 0x3f333333, 0x3f2ccccd,
      0x3f266666, 0x3f200000, 0x3f19999a, 0x3f133333, 0x3f0ccccd, 0x3f066666, 0x3f000000,
      0x3ef33333, 0x3ee66667, 0x3ed9999a, 0x3ecccccd, 0x3ec00000, 0x3eb33333, 0x3ea66667,
      0x3e99999a, 0x3e8ccccd, 0x3e800000, 0x3e666667, 0x3e4ccccd, 0x3e333333, 0x3e19999a,
      0x3e000000, 0x3dcccccd, 0x3d99999a, 0x3d4ccccd, 0x3ccccccd, 0x00000000,
    ],
    delta: [
      0x3cccccc0, 0x3ccccce0, 0x3cccccc0, 0x3ccccce0, 0x3cccccc0, 0x3cccccc0, 0x3ccccce0,
      0x3cccccc0, 0x3ccccce0, 0x3cccccc0, 0x3cccccc0, 0x3ccccce0, 0x3cccccc0, 0x3ccccce0,
      0x3cccccc0, 0x3cccccc0, 0x3ccccce0, 0x3cccccc0, 0x3ccccce0, 0x3cccccc0, 0x3cccccd0,
      0x3cccccc0, 0x3cccccd0, 0x3cccccd0, 0x3cccccd0, 0x3cccccd0, 0x3cccccc0, 0x3cccccd0,
      0x3cccccd0, 0x3cccccd0, 0x3cccccc8, 0x3cccccd0, 0x3cccccd0, 0x3cccccc8, 0x3cccccd0,
      0x3ccccccc, 0x3ccccccc, 0x3cccccce, 0x3ccccccd, 0x3ccccccd,
    ],
  },
  64: {
    t: [
      0x3f800000, 0x3f7c0000, 0x3f780000, 0x3f740000, 0x3f700000, 0x3f6c0000, 0x3f680000,
      0x3f640000, 0x3f600000, 0x3f5c0000, 0x3f580000, 0x3f540000, 0x3f500000, 0x3f4c0000,
      0x3f480000, 0x3f440000, 0x3f400000, 0x3f3c0000, 0x3f380000, 0x3f340000, 0x3f300000,
      0x3f2c0000, 0x3f280000, 0x3f240000, 0x3f200000, 0x3f1c0000, 0x3f180000, 0x3f140000,
      0x3f100000, 0x3f0c0000, 0x3f080000, 0x3f040000, 0x3f000000, 0x3ef80000, 0x3ef00000,
      0x3ee80000, 0x3ee00000, 0x3ed80000, 0x3ed00000, 0x3ec80000, 0x3ec00000, 0x3eb80000,
      0x3eb00000, 0x3ea80000, 0x3ea00000, 0x3e980000, 0x3e900000, 0x3e880000, 0x3e800000,
      0x3e700000, 0x3e600000, 0x3e500000, 0x3e400000, 0x3e300000, 0x3e200000, 0x3e100000,
      0x3e000000, 0x3de00000, 0x3dc00000, 0x3da00000, 0x3d800000, 0x3d400000, 0x3d000000,
      0x3c800000, 0x00000000,
    ],
    delta: [
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000, 0x3c800000,
      0x3c800000,
    ],
  },
  100: {
    t: [
      0x3f800000, 0x3f7d70a4, 0x3f7ae148, 0x3f7851ec, 0x3f75c28f, 0x3f733333, 0x3f70a3d7,
      0x3f6e147b, 0x3f6b851f, 0x3f68f5c3, 0x3f666666, 0x3f63d70a, 0x3f6147ae, 0x3f5eb852,
      0x3f5c28f6, 0x3f59999a, 0x3f570a3e, 0x3f547ae1, 0x3f51eb85, 0x3f4f5c29, 0x3f4ccccd,
      0x3f4a3d71, 0x3f47ae15, 0x3f451eb8, 0x3f428f5c, 0x3f400000, 0x3f3d70a4, 0x3f3ae148,
      0x3f3851ec, 0x3f35c28f, 0x3f333333, 0x3f30a3d7, 0x3f2e147b, 0x3f2b851f, 0x3f28f5c3,
      0x3f266667, 0x3f23d70a, 0x3f2147ae, 0x3f1eb852, 0x3f1c28f6, 0x3f19999a, 0x3f170a3e,
      0x3f147ae1, 0x3f11eb85, 0x3f0f5c29, 0x3f0ccccd, 0x3f0a3d71, 0x3f07ae15, 0x3f051eb8,
      0x3f028f5c, 0x3f000000, 0x3efae147, 0x3ef5c28f, 0x3ef0a3d7, 0x3eeb851e, 0x3ee66666,
      0x3ee147ae, 0x3edc28f5, 0x3ed70a3d, 0x3ed1eb85, 0x3ecccccc, 0x3ec7ae14, 0x3ec28f5c,
      0x3ebd70a4, 0x3eb851eb, 0x3eb33333, 0x3eae147b, 0x3ea8f5c2, 0x3ea3d70a, 0x3e9eb852,
      0x3e999999, 0x3e947ae1, 0x3e8f5c29, 0x3e8a3d70, 0x3e851eb8, 0x3e800000, 0x3e75c28f,
      0x3e6b851e, 0x3e6147ae, 0x3e570a3d, 0x3e4ccccc, 0x3e428f5c, 0x3e3851eb, 0x3e2e147b,
      0x3e23d70a, 0x3e199999, 0x3e0f5c29, 0x3e051eb8, 0x3df5c28f, 0x3de147ae, 0x3dcccccc,
      0x3db851eb, 0x3da3d70a, 0x3d8f5c29, 0x3d75c28f, 0x3d4ccccc, 0x3d23d70a, 0x3cf5c28f,
      0x3ca3d70a, 0x3c23d70a, 0x00000000,
    ],
    delta: [
      0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d740, 0x3c23d700, 0x3c23d700, 0x3c23d700,
      0x3c23d700, 0x3c23d700, 0x3c23d740, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700,
      0x3c23d700, 0x3c23d700, 0x3c23d740, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700,
      0x3c23d700, 0x3c23d740, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700,
      0x3c23d740, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700,
      0x3c23d740, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d740,
      0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d700, 0x3c23d740, 0x3c23d700,
      0x3c23d700, 0x3c23d720, 0x3c23d700, 0x3c23d700, 0x3c23d720, 0x3c23d700, 0x3c23d700,
      0x3c23d720, 0x3c23d700, 0x3c23d700, 0x3c23d720, 0x3c23d700, 0x3c23d700, 0x3c23d700,
      0x3c23d720, 0x3c23d700, 0x3c23d700, 0x3c23d720, 0x3c23d700, 0x3c23d700, 0x3c23d720,
      0x3c23d700, 0x3c23d700, 0x3c23d720, 0x3c23d700, 0x3c23d700, 0x3c23d710, 0x3c23d710,
      0x3c23d700, 0x3c23d710, 0x3c23d710, 0x3c23d700, 0x3c23d710, 0x3c23d700, 0x3c23d710,
      0x3c23d710, 0x3c23d700, 0x3c23d710, 0x3c23d708, 0x3c23d708, 0x3c23d710, 0x3c23d708,
      0x3c23d708, 0x3c23d708, 0x3c23d70c, 0x3c23d70c, 0x3c23d708, 0x3c23d70a, 0x3c23d70a,
      0x3c23d70a, 0x3c23d70a,
    ],
  },
};

const bits = (values: Float32Array<ArrayBuffer>): number[] =>
  Array.from(new Uint32Array(values.buffer));

test("parseIrodoriPipelineConfig: a meanflow declaration parses to the MeanFlow branch with no RF field", () => {
  const config = parseIrodoriPipelineConfig(MEANFLOW);
  assertEquals(config.flowParameterization, "meanflow");
  assertEquals(config.steps, 4);
  for (const key of Object.keys(RF_KNOBS)) {
    assertEquals(Object.hasOwn(config, key), false, key);
  }
});

test("parseIrodoriPipelineConfig: an absent flowParameterization means rf_velocity (the published manifests)", () => {
  const config = parseIrodoriPipelineConfig(RF);
  assertEquals(config.flowParameterization, "rf_velocity");
  if (config.flowParameterization !== "rf_velocity") throw new Error("not the RF branch");
  assertEquals(config.cfgScales, { text: 3, speaker: 5, caption: 3 });
  assertEquals(parseIrodoriPipelineConfig({ ...RF, flowParameterization: "rf_velocity" }), config);
});

test("parseIrodoriPipelineConfig: every RF-only key next to meanflow fails by name (a value it would ignore)", () => {
  for (const [key, value] of Object.entries(RF_KNOBS)) {
    const error = assertThrows(
      () => parseIrodoriPipelineConfig({ ...MEANFLOW, [key]: value }),
      Error,
      "declares the RF-only key(s)",
    );
    assert(error.message.includes(`'${key}'`), error.message);
  }
});

test("parseIrodoriPipelineConfig: a flowParameterization outside the two spellings fails", () => {
  for (const value of ["MeanFlow", "rf", "", 1, null]) {
    assertThrows(
      () => parseIrodoriPipelineConfig({ ...MEANFLOW, flowParameterization: value }),
      Error,
      "flowParameterization",
    );
  }
});

test("parseIrodoriPipelineConfig: an RF declaration still needs every RF knob", () => {
  for (const key of Object.keys(RF_KNOBS)) {
    const { [key]: _removed, ...rest } = RF;
    assertThrows(() => parseIrodoriPipelineConfig(rest), Error, key);
  }
});

test("parseIrodoriPipelineConfig: the common checks still apply to a meanflow declaration", () => {
  assertThrows(() => parseIrodoriPipelineConfig({ ...MEANFLOW, steps: 0 }), Error, "steps");
  assertThrows(() => parseIrodoriPipelineConfig({ ...MEANFLOW, step: 4 }), Error, "unknown key");
  const { steps: _steps, ...withoutSteps } = MEANFLOW;
  assertThrows(() => parseIrodoriPipelineConfig(withoutSteps), Error, "steps: missing");
});

test("meanflowSchedule: reproduces torch's f32 linspace(1, 0) bit for bit for every pinned step count", () => {
  for (const [steps, expected] of Object.entries(TORCH)) {
    assertEquals(bits(meanflowSchedule(Number(steps))), [...expected.t], `steps ${steps}`);
  }
});

test("meanflowIntervals: reproduces torch's f32 t[:-1] − t[1:] bit for bit", () => {
  for (const [steps, expected] of Object.entries(TORCH)) {
    const intervals = meanflowIntervals(meanflowSchedule(Number(steps)));
    assertEquals(bits(intervals), [...expected.delta], `steps ${steps}`);
  }
});

test("meanflowSchedule: the closed form 1 − i/steps is not the same sequence (why the kernel's form is copied)", () => {
  // Fault injection on the formula: the obvious closed form already splits at steps 3.
  const closed = new Float32Array(4).map((_, index) => Math.fround(1 - index / 3));
  assert(bits(closed).some((value, index) => value !== TORCH[3].t[index]));
});

test("meanflowSchedule: rejects a step count that is not a positive integer", () => {
  for (const steps of [0, -1, 1.5, Number.NaN]) {
    assertThrows(() => meanflowSchedule(steps), RangeError, "steps");
  }
});

const CAPS = { text: 256, speaker: 751, caption: 512 } as const;
const FRAMES = 25;
const MASK_SHAPE = [1, 1, 1, FRAMES + CAPS.text + CAPS.speaker + CAPS.caption] as const;
const USED_ALL = { text: 12, speaker: 32, caption: 20 } as const;

const samplerOf = (config: IrodoriPipelineConfig) =>
  ditSampler(config, FRAMES, USED_ALL, CAPS, MASK_SHAPE);

test("ditSampler: a meanflow declaration gives the linspace schedule, one interval per step and no uncond variant", () => {
  const { schedule, sampler } = samplerOf(parseIrodoriPipelineConfig(MEANFLOW));
  assertEquals(bits(schedule), [...TORCH[4].t]);
  assertEquals(sampler.kind, "meanflow");
  if (sampler.kind !== "meanflow") throw new Error("not the MeanFlow sampler");
  assertEquals(bits(sampler.intervals), [...TORCH[4].delta]);
  // The sampler carries nothing CFG could read (every condition present, which would give RF three
  // variants).
  assertEquals(Object.keys(sampler).sort(), ["intervals", "kind"]);
});

test("ditSampler: the RF declaration keeps its closed-form schedule and one variant per condition with CFG", () => {
  const config = parseIrodoriPipelineConfig(RF);
  const { schedule, sampler } = samplerOf(config);
  assertEquals(bits(schedule), bits(tSchedule(40, 0.999)));
  if (sampler.kind !== "rf_velocity") throw new Error("not the RF sampler");
  assertEquals(
    sampler.uncondVariants.map((variant) => variant.segment),
    ["text", "speaker", "caption"],
  );
  assertEquals([sampler.cfgMinT, sampler.cfgMaxT], [0.5, 1]);
});

test("ditSampler: a meanflow declaration of another step count follows it (not only 4)", () => {
  const { schedule, sampler } = samplerOf(parseIrodoriPipelineConfig({ ...MEANFLOW, steps: 7 }));
  assertEquals(bits(schedule), [...TORCH[7].t]);
  if (sampler.kind !== "meanflow") throw new Error("not the MeanFlow sampler");
  assertEquals(sampler.intervals.length, 7);
});

// The hand-written schema validation of `pipelineConfig` (ADR 0038 §1 — models owns the schema).
//
// The hub passes `pipelineConfig` through untouched, so **this is the only check**. It mainly pins
// two things: ① a missing required field or a wrong type is rejected by name; ② an unknown key
// fails loudly (it is not silently degraded to a default).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { parseAnimaPipelineConfig } from "../src/anima/config.ts";
import { ModelInputError } from "../src/errors.ts";

/** The same shape as `models/karume-anima/karume.json` (a copy of the real distribution). */
const VALID = {
  scheduler: { shift: 3, numTrainTimesteps: 1000 },
  defaults: {
    steps: 10,
    guidanceScale: 1,
    resolution: { width: 1024, height: 1024 },
    negativePrompt: "low quality, worst quality, blurry, bad anatomy, jpeg artifacts",
  },
} as const;

const withDefaults = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...VALID,
  defaults: { ...VALID.defaults, ...patch },
});

test("parseAnimaPipelineConfig: reads the shape of the real distribution", () => {
  const config = parseAnimaPipelineConfig(VALID);
  // The real distribution has no `type`, so it falls back to the default `"euler"` (see also
  // "defaults to euler" below).
  assertEquals(config.scheduler, { type: "euler", shift: 3, numTrainTimesteps: 1000 });
  assertEquals(config.defaults.steps, 10);
  assertEquals(config.defaults.guidanceScale, 1);
  assertEquals(config.defaults.resolution, { width: 1024, height: 1024 });
  assertEquals(config.defaults.negativePrompt, VALID.defaults.negativePrompt);
});

test("parseAnimaPipelineConfig: negativePrompt is optional (the field may be absent)", () => {
  const config = parseAnimaPipelineConfig({
    scheduler: VALID.scheduler,
    defaults: {
      steps: 8,
      guidanceScale: 4,
      resolution: { width: 1344, height: 768 },
    },
  });
  assertEquals(config.defaults.negativePrompt, undefined);
  assertEquals(config.defaults.resolution, { width: 1344, height: 768 });
});

test("parseAnimaPipelineConfig: scheduler.type defaults to euler (pins the behaviour of distributed manifests)", () => {
  // `type` is an optional field added additively (ADR 0038 §1). Repositories already distributed
  // cannot be migrated, so the moment a manifest without the field moved to another update rule,
  // the output images would change silently.
  assertEquals(parseAnimaPipelineConfig(VALID).scheduler.type, "euler");
  assertEquals(
    parseAnimaPipelineConfig({
      ...VALID,
      scheduler: { ...VALID.scheduler, type: "euler" },
    }).scheduler.type,
    "euler",
  );
});

test("parseAnimaPipelineConfig: scheduler.type accepts exactly two words", () => {
  assertEquals(
    parseAnimaPipelineConfig({
      ...VALID,
      scheduler: { ...VALID.scheduler, type: "dpmpp-2m" },
    }).scheduler,
    { type: "dpmpp-2m", shift: 3, numTrainTimesteps: 1000 },
  );
  // An unknown value or a non-string is rejected with expected and actual side by side (not
  // silently degraded to the default euler).
  const unknown = assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { ...VALID.scheduler, type: "dpm++" } }),
    Error,
    "pipelineConfig.scheduler.type: expected 'euler' / 'dpmpp-2m'",
  );
  assertEquals(unknown.message.includes("actual 'dpm++'"), true, unknown.message);
  const notString = assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { ...VALID.scheduler, type: 2 } }),
    Error,
    "pipelineConfig.scheduler.type: expected",
  );
  assertEquals(notString.message.includes("actual 2"), true, notString.message);
  // Different case or surrounding whitespace is not accepted either (the accepted set is an exact
  // match of the two words).
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { ...VALID.scheduler, type: "Euler" } }),
    Error,
    "actual 'Euler'",
  );
});

test("parseAnimaPipelineConfig: an unknown key fails loudly (a misspelling is not degraded to a default)", () => {
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, sampler: {} }),
    Error,
    "pipelineConfig: unknown key 'sampler'",
  );
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { ...VALID.scheduler, sigmaMax: 1 } }),
    Error,
    "pipelineConfig.scheduler: unknown key 'sigmaMax'",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ step: 8 })),
    Error,
    "pipelineConfig.defaults: unknown key 'step'",
  );
  assertThrows(
    () =>
      parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1024, height: 1024, w: 1 } })),
    Error,
    "unknown key 'w'",
  );
});

test("parseAnimaPipelineConfig: rejects a missing required field by name", () => {
  assertThrows(() => parseAnimaPipelineConfig({}), Error, "pipelineConfig.scheduler: missing");
  assertThrows(
    () => parseAnimaPipelineConfig({ scheduler: VALID.scheduler }),
    Error,
    "pipelineConfig.defaults: missing",
  );
  assertThrows(
    () => parseAnimaPipelineConfig({ scheduler: { shift: 3 }, defaults: VALID.defaults }),
    Error,
    "pipelineConfig.scheduler.numTrainTimesteps: missing",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1024 } })),
    Error,
    "height: missing",
  );
});

test("parseAnimaPipelineConfig: checks types and ranges", () => {
  assertThrows(
    () => parseAnimaPipelineConfig({ ...VALID, scheduler: { shift: 0, numTrainTimesteps: 1000 } }),
    Error,
    "not a positive finite number",
  );
  assertThrows(
    () =>
      parseAnimaPipelineConfig({
        ...VALID,
        scheduler: { shift: 3, numTrainTimesteps: 1000.5 },
      }),
    Error,
    "not a positive integer",
  );
  // steps < 2 cannot build the sigma linspace (the same lower bound as the sampler's accepted set).
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ steps: 1 })),
    Error,
    "integer of at least 2",
  );
  // The upper bound is the sampler's accepted set too (if the default lay outside the check, a
  // generate whose request passes nothing would fail because of its input). The diagnostic is the
  // very wording of the generation check (`assertAcceptableSteps`) — evidence that it calls the
  // check instead of copying the condition.
  assertEquals(parseAnimaPipelineConfig(withDefaults({ steps: 1000 })).defaults.steps, 1000);
  const overLimit = assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ steps: 1001 })),
    Error,
    "is not at most 1000",
  );
  // As with the resolution, a manifest-side violation is an asset mismatch, so it is **not** a
  // `ModelInputError`.
  assert(!(overLimit instanceof ModelInputError));
  assert(overLimit.cause instanceof ModelInputError);
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ guidanceScale: "1" })),
    Error,
    "not a finite number",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ negativePrompt: 42 })),
    Error,
    "not a string",
  );
});

test("parseAnimaPipelineConfig: requires the default resolution to lie inside the accepted set too", () => {
  // If the distribution violates it, construction fails — no "found out only after running one
  // generation".
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1000, height: 1024 } })),
    Error,
    "is not a multiple of",
  );
  assertThrows(
    () => parseAnimaPipelineConfig(withDefaults({ resolution: { width: 256, height: 256 } })),
    Error,
    "lower bound",
  );
  // The other side: the accepted-set conditions are shared with the generation request, but a
  // manifest-side violation is an asset mismatch, so it is **not** a `ModelInputError` (ADR 0107
  // decision 2). The original exception remains in `cause`.
  const thrown = assertThrows(() =>
    parseAnimaPipelineConfig(withDefaults({ resolution: { width: 1000, height: 1024 } })),
  );
  assert(thrown instanceof Error);
  assert(!(thrown instanceof ModelInputError));
  assert(thrown.cause instanceof ModelInputError);
});

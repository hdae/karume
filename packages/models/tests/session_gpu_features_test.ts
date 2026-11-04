// Host-side tests of `src/session/gpu-features.ts`: the GPU prerequisites a `Quant` declares
// (`gpuFeatures` and `requiredLimits`), mapped to `acquireGpu` requests and checked against a GPU.
// No adapter is needed: the feature checks wrap a fake device in the real `GpuContext`, and the
// limits checks are pure functions.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { readTextFile } from "@karume/test-support/fs";
import type { GpuFeaturesSpec, RequiredLimitsSpec } from "@karume/hub";
import type { RequiredLimits } from "@karume/runtime";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  assertAcquiredGpuCapabilities,
  assertGpuFeaturesGranted,
  assertRequiredLimitsSatisfied,
  sessionGpuFeatures,
  toAcquireGpuOptions,
} from "../src/session/gpu-features.ts";

// ==== gpuFeatures ====================================================================
//
// The mapping from the manifest's `gpuFeatures` (features that can only be requested when the
// device is created) to the `acquireGpu` request. It is the spelling contract of ADR 0038 §3, and a
// gap is a **silent degradation** (running on a device without a capability the distribution asked
// for). All eight families use the same function, so its checks are gathered here in one place too
// (originally seven families each read `quant.gpuFeatures?.shaderF16 === true` on their own line —
// review M1-V9).
//
// The check after acquisition (`assertGpuFeaturesGranted`) is exercised without a GPU too. runtime
// does not expose `GpuContext` as a value (`acquireGpu` is the only entry point — ADR 0008), so
// this uses `fakeGpuContext` from the helper `fake-gpu.ts` over there (= a fake device wrapped in
// **the real `GpuContext`**) (precedent: ⑧ in `container_loading_test.ts`). A home-made fake would
// make the subject under test itself fake; wrapping the real one keeps the subject real.
//
// Only the positive control (passes on a device that has the feature) cannot be written yet,
// because the helper needs a way to pass features (requested from whoever owns the runtime tests —
// a third argument to `fakeGpuContext`).

test("toAcquireGpuOptions: requests only the declared features", () => {
  // Requesting a feature for a distribution with no declaration / an explicit false would make
  // `acquireGpu` itself fail on adapters without it (= it would stop working where it should work).
  assertEquals(toAcquireGpuOptions(undefined), {});
  assertEquals(toAcquireGpuOptions({}), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: false }), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: true }), { shaderF16: true });
});

test("toAcquireGpuOptions: the online attention preset asks for subgroups only as 'available' (ADR 0132 decision 5)", () => {
  // `attentionScoreStorage: "none"` runs faster with subgroups but is bit-identical without them,
  // so the request never becomes a requirement (a default preset declares no feature — ADR 0128
  // C1).
  assertEquals(toAcquireGpuOptions(undefined, { attentionScoreStorage: "none" }), {
    subgroups: "available",
  });
  assertEquals(
    toAcquireGpuOptions(
      { shaderF16: true },
      { attentionCompute: "a8", attentionScoreStorage: "none" },
    ),
    { subgroups: "available", shaderF16: true },
  );
  assertEquals(toAcquireGpuOptions(undefined, { attentionScoreStorage: "f16" }), {});
  assertEquals(toAcquireGpuOptions({ shaderF16: true }, {}), { shaderF16: true });
});

test("toAcquireGpuOptions: maps every key of GpuFeaturesSpec (detects a key left behind when one is added)", () => {
  // The mapping's coverage is pinned by type through `FEATURES` in `src/session/gpu-features.ts`
  // (the exhaustive table of `Required<GpuFeaturesSpec>`), so a new key is a **compile error**.
  // This is the counterpart that checks at run time that this type check is alive (a change that
  // loosens the type would pass the type check) — it passes a spec with every key set to true and
  // confirms that the resulting key set equals the input's.
  const full: Required<GpuFeaturesSpec> = { shaderF16: true };
  const requested = toAcquireGpuOptions(full) as Record<string, unknown>;
  assertEquals(Object.keys(requested).sort(), Object.keys(full).sort());
});

// ---- Checks on an acquired GpuContext (the only check on the shared-GPU path) ----

const FEATURES_WHERE = "TestPipeline: quant 'f16'";

/** A device without shader-f16 (`fakeGpuContext` creates features as an empty `Set`). */
const noFeatureGpu = () => fakeGpuContext(fakeDevice());

test("assertGpuFeaturesGranted: a shared GPU without a requested feature fails, naming it", () => {
  // On the path given a shared GPU (`options.gpu`), the device already exists, so the feature
  // cannot be requested again (ADR 0028). Letting it through here would fail later at Session
  // construction or silently fall back to another path.
  const error = assertThrows(
    () => assertGpuFeaturesGranted({ shaderF16: true }, noFeatureGpu(), FEATURES_WHERE),
    Error,
  );
  // The three parts of the diagnostic: which preset / which feature / how to request it again.
  assertStringIncludes(error.message, FEATURES_WHERE);
  assertStringIncludes(error.message, "shader-f16");
  assertStringIncludes(error.message, "acquireGpu({ shaderF16: true })");
});

test("assertGpuFeaturesGranted: a no-op for a distribution that requests nothing (the counterpart showing the check is not always true)", () => {
  const gpu = noFeatureGpu();
  assertGpuFeaturesGranted(undefined, gpu, FEATURES_WHERE);
  assertGpuFeaturesGranted({}, gpu, FEATURES_WHERE);
  assertGpuFeaturesGranted({ shaderF16: false }, gpu, FEATURES_WHERE);
});

// ---- Requests derived from the effective SessionOptions (presets of families with overrides) ----

test("sessionGpuFeatures: adds shader-f16 to the request when the effective settings choose f16 compute", () => {
  // An explicit `linearCompute: "f16"` layered over a quant declaration (that does not request
  // shader-f16). Without adding it, a self-acquired device lacks the feature and Session
  // construction fails after all the weights have been uploaded.
  assertEquals(sessionGpuFeatures(undefined, { linearCompute: "f16" }), { shaderF16: true });
  assertEquals(sessionGpuFeatures({ shaderF16: false }, { attentionCompute: "f16" }), {
    shaderF16: true,
  });
  // On the shared-GPU path, the added request goes straight into the admission check (it fails
  // before the weights are fetched).
  assertThrows(
    () =>
      assertGpuFeaturesGranted(
        sessionGpuFeatures(undefined, { linearCompute: "f16" }),
        noFeatureGpu(),
        FEATURES_WHERE,
      ),
    Error,
    "shader-f16",
  );
});

test("sessionGpuFeatures: returns the declaration as is when nothing is needed or it is already declared (no extra requests)", () => {
  // s16 (`attentionScoreStorage: "f16"`) is a storage form that does not depend on shader-f16 (ADR
  // 0030 decision 1).
  assertEquals(sessionGpuFeatures(undefined, {}), undefined);
  assertEquals(sessionGpuFeatures(undefined, { attentionScoreStorage: "f16" }), undefined);
  assertEquals(sessionGpuFeatures(undefined, { linearCompute: "a8" }), undefined);
  const declared = { shaderF16: true };
  assertEquals(sessionGpuFeatures(declared, { linearCompute: "f16" }) === declared, true);
});

// ==== requiredLimits =================================================================
//
// The check that compares the manifest's `requiredLimits` (the minimum device limits a quant
// declares — ADR 0038 §7) with the GPU's limits. It is ADR 0089 decision 5's check **before a
// single byte of weights is downloaded**, so both an omission (declared but checked by nobody =
// fails at allocation after downloading several GiB) and an excess (checking limits nobody wrote =
// wrongly rejecting an environment that would work) do real harm.
//
// The side that provides what to compare against (`assertRequiredLimitsBeforeDownload` — the shared
// GPU's limits or the adapter's measured values) needs a GPU / the load path, so ⑧ and ⑨ in
// `container_loading_test.ts` pin it.

const LIMITS_WHERE = "TestPipeline: quant 'i4'";

/**
 * The measured values to compare against. Everything except `maxBufferSize` is 0 so that a pass
 * shows that **limits not written in the spec are not examined**.
 */
const LIMITS: RequiredLimits = {
  maxBufferSize: 268435456,
  maxStorageBufferBindingSize: 0,
  maxUniformBufferBindingSize: 0,
  maxStorageBuffersPerShaderStage: 0,
  maxUniformBuffersPerShaderStage: 0,
  maxComputeWorkgroupStorageSize: 0,
  maxComputeInvocationsPerWorkgroup: 0,
  maxComputeWorkgroupSizeX: 0,
  maxComputeWorkgroupSizeY: 0,
  maxComputeWorkgroupSizeZ: 0,
  maxComputeWorkgroupsPerDimension: 0,
};

test("assertRequiredLimitsSatisfied: a distribution whose requirement equals the measured value passes", () => {
  // The boundary is "short = measured < required" — written as `<=`, every device at exactly the
  // requirement (the value the exporter bakes is the actual largest tensor itself) would be wrongly
  // rejected.
  const spec: RequiredLimitsSpec = { maxBufferSize: LIMITS.maxBufferSize };
  assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE);
});

test("assertRequiredLimitsSatisfied: a distribution short by even 1 fails", () => {
  const spec: RequiredLimitsSpec = { maxBufferSize: LIMITS.maxBufferSize + 1 };
  assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE),
    Error,
    "maxBufferSize",
  );
});

test("assertRequiredLimitsSatisfied: limits not written in the spec are not examined (a partial mapping)", () => {
  // Ten measured values are 0, but the distribution claims only `maxBufferSize`. Examining limits
  // that were not written would break the semantics "no field = runs on the WebGPU guaranteed
  // defaults" (ADR 0089 decision 3).
  const spec: RequiredLimitsSpec = { maxBufferSize: 1 };
  assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE);
});

test("assertRequiredLimitsSatisfied: a no-op for a distribution without a declaration", () => {
  assertRequiredLimitsSatisfied(undefined, LIMITS, LIMITS_WHERE);
});

test("assertRequiredLimitsSatisfied: the diagnostic carries where, the limit name, the required and the measured value", () => {
  // A user who hit the failure needs all four to decide the next step (another quant / another
  // machine). Every shortfall is reported **at once** (making them fix one at a time hides the full
  // picture of how far the environment falls short).
  const spec: RequiredLimitsSpec = {
    maxBufferSize: 1073741824,
    maxStorageBufferBindingSize: 134217728,
  };
  const error = assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE),
    Error,
  );
  for (const fragment of [
    LIMITS_WHERE,
    "maxBufferSize",
    "1073741824",
    "268435456",
    "maxStorageBufferBindingSize",
    "134217728",
    "0",
  ]) {
    if (!error.message.includes(fragment)) {
      throw new Error(`the diagnostic lacks '${fragment}': ${error.message}`);
    }
  }
});

test("assertRequiredLimitsSatisfied: a limit name runtime does not have fails as vocabulary drift", () => {
  // A name hub accepted that runtime does not have (the second line of defence). Silently skipping
  // it would bring back "a declared limit nobody looks at" = the silent cap. While
  // `limit_vocabulary_test.ts` is green this is structurally unreachable, but it catches the case
  // when that check comes off.
  //
  // The unknown key is built through an index signature (killing the type with `as unknown as`
  // would obscure that this check looks at what is said at run time, not at the type having been
  // loosened).
  const spec: Record<string, number> = { maxBufferSize: 1, karumeUnknownLimit: 1 };
  const error = assertThrows(
    () => assertRequiredLimitsSatisfied(spec, LIMITS, LIMITS_WHERE),
    Error,
  );
  for (const fragment of [LIMITS_WHERE, "karumeUnknownLimit", "tests/limit_vocabulary_test.ts"]) {
    if (!error.message.includes(fragment)) {
      throw new Error(`the diagnostic lacks '${fragment}': ${error.message}`);
    }
  }
});

// ==== The checks after GPU acquisition (one helper for every family) ===================

const ACQUIRED_WHERE = "TestPipeline: quant 'f16'";

test("assertAcquiredGpuCapabilities: names a missing feature with the family's subject", () => {
  const error = assertThrows(
    () =>
      assertAcquiredGpuCapabilities({ shaderF16: true }, undefined, noFeatureGpu(), ACQUIRED_WHERE),
    Error,
  );
  assertStringIncludes(error.message, ACQUIRED_WHERE);
  assertStringIncludes(error.message, "shader-f16");
});

test("assertAcquiredGpuCapabilities: matches the declared limits against the GPU in hand (not only a shared one)", () => {
  // The fake context's limits are all 0, so any declared minimum is a shortfall.
  const error = assertThrows(
    () =>
      assertAcquiredGpuCapabilities(
        undefined,
        { maxBufferSize: 1 },
        noFeatureGpu(),
        ACQUIRED_WHERE,
      ),
    Error,
  );
  for (const fragment of [ACQUIRED_WHERE, "maxBufferSize: required 1 / this GPU has 0"]) {
    assertStringIncludes(error.message, fragment);
  }
});

test("assertAcquiredGpuCapabilities: a preset that declares neither passes on any GPU", () => {
  assertAcquiredGpuCapabilities(undefined, undefined, noFeatureGpu(), ACQUIRED_WHERE);
  assertAcquiredGpuCapabilities({}, {}, noFeatureGpu(), ACQUIRED_WHERE);
});

/**
 * The construction code of the eight families that check the GPU after acquiring it. Each must go
 * through the helper after `options.gpu ?? (await acquireGpu(...))` — a family that writes the two
 * checks by hand can drop one of them for a device it acquired itself, and no type check notices.
 */
const ACQUIRING_FAMILIES = [
  "wan/family.ts",
  "anima/pipeline.ts",
  "sbv2/pipeline.ts",
  "irodori/pipeline.ts",
  "birefnet/pipeline.ts",
  "depth-anything/pipeline.ts",
  "siglip2/pipeline.ts",
  "vowel-detector/pipeline.ts",
] as const;

test("every family checks the GPU it acquired through assertAcquiredGpuCapabilities, not by hand", async () => {
  for (const file of ACQUIRING_FAMILIES) {
    const source = await readTextFile(new URL(`../src/${file}`, import.meta.url));
    const acquisition = source.indexOf("options.gpu ?? (await acquireGpu(");
    assert(acquisition !== -1, `${file}: the GPU acquisition expression is not found`);
    assertEquals(
      source.indexOf("options.gpu ?? (await acquireGpu(", acquisition + 1),
      -1,
      `${file}: more than one GPU acquisition`,
    );
    const after = source.slice(acquisition);
    assertStringIncludes(
      after,
      "assertAcquiredGpuCapabilities(",
      `${file}: the helper is not called after acquisition`,
    );
    for (const bare of ["assertGpuFeaturesGranted(", "assertRequiredLimitsSatisfied("]) {
      assert(!after.includes(bare), `${file}: '${bare}' is written by hand after acquisition`);
    }
  }
});

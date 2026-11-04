// The **construction guards** of `Pipeline`. Only the part that needs neither a GPU nor real assets
// (the E2E on a real GPU is P3, iteration 2).
//
// It covers six things:
//  ① `fromAssets` rejects manifest contract violations and asset parse failures **before**
//    acquiring a GPU (unknown model / pipeline name / unknown major / unknown quant / missing
//    asset). If the place of rejection moved, in an environment without a GPU it would turn into
//    another exception and the reader would not learn "what was wrong".
//  ② The entry guard of `generate` (`resolveNegativePrompt`) "does not silently accept a knob that
//    has no effect". A regression would leave the PNG check on a real GPU green, so it is pinned
//    directly as a pure function.
//  ③ `copyLatents` of `denoise-step` (`latentSnapshot`) returns "a copy of the latent at the time
//    it was made". If this broke, subscribers would **silently** receive **the latent of another
//    step** (the PNG check on a real GPU does not go through the observation hook, so it stays
//    green).
//  ④ The construction `signal` takes effect **both at the entry and after execution has started**
//    (if the assembly after the download finished could not be aborted, there would be a window in
//    which the UI's cancel button does nothing). The latter is checked up to "the first stage
//    boundary" with empty assets — the boundaries after that need real assets and a GPU, so they
//    cannot be checked here.
//  ⑤ The choice point of the denoise loop's update rule (`denoiseStep`) branches according to the
//    effective sampler (the request's `sampler` ↩ the manifest's `scheduler.type`), and `"euler"`
//    keeps no history (the PNG check on a real GPU goes through both update rules, but it needs
//    assets and a GPU).
//  ⑥ The spelling of `residency` (DiT residency — ADR 0112) is rejected at admission for the
//    construction option and at the entry of `generate` (before touching the GPU) for the request.
//    The latter is pinned together with its wiring, on a pipeline built with a fake GPU (unlike the
//    sampler, the e2e on a real GPU has no misspelling case). The state transitions themselves
//    (keep / release / evict) are checked by `anima_residency_test.ts`.
//
// NOTE: The mapping from the manifest's `session` to `SessionOptions` is now shared by seven
// families, so its check lives in `session_options_test.ts`.

import { test } from "vite-plus/test";
import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertNotStrictEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { parseManifest } from "@karume/hub";
import { SAMPLER_TYPES } from "../anima.ts";
import { assertAnimaSamplerType } from "../src/anima/config.ts";
import {
  type GenerateRequest,
  Pipeline,
  type PipelineOptions,
  denoiseStep,
  latentSnapshot,
  resolveNegativePrompt,
} from "../src/anima/pipeline.ts";
import { Randn } from "../src/anima/random.ts";
import { ModelInputError } from "../src/errors.ts";
import { assertAcceptableSeed } from "../src/request-gates.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

/**
 * This series' weight components (the same four as `COMPONENT_KEYS` in `src/anima/pipeline.ts`).
 */
const WEIGHT_NAMES = ["text_encoder", "text_conditioner", "transformer", "vae_decoder"] as const;

/**
 * Four containers that open (with minimal declarations). anima's admission does not compare graphs,
 * so the content does not matter. `tokenizer` is left out: it stays as the one that fails after
 * every check has passed.
 */
const COMPONENTS: Record<string, Uint8Array<ArrayBuffer>> = {};
for (const name of WEIGHT_NAMES) {
  Object.assign(
    COMPONENTS,
    partAssets(
      name,
      await tensorlessContainer(name, {
        inputs: [{ name: "x", shape: [1, 4] }],
        output: { name: "y", shape: [1, 4] },
      }),
    ),
  );
}

/** The skeleton of `models/karume-anima/karume.json` (only the fields the checks need). */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "anima-turbo",
    models: {
      "anima-turbo": {
        pipeline: "anima/1",
        weights: Object.fromEntries(
          WEIGHT_NAMES.map((name) => [name, { f16: declaredContainer(`${name}/model.f16`) }]),
        ),
        assets: {},
        quants: {
          "f16+dit8-a8-attn8-s16": {
            weights: Object.fromEntries(WEIGHT_NAMES.map((name) => [name, "f16"])),
            session: {},
          },
        },
        defaultQuant: "f16+dit8-a8-attn8-s16",
        pipelineConfig: {
          scheduler: { shift: 3, numTrainTimesteps: 1000 },
          defaults: {
            steps: 10,
            guidanceScale: 1,
            resolution: { width: 1024, height: 1024 },
          },
        },
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

test("fromAssets: rejects a manifest whose pipeline contract name is not anima", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "sbv2/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "'sbv2/1'",
  );
});

test("fromAssets: an unknown major fails loudly (checking it is models' responsibility — ADR 0038 §1)", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "anima/2" }));
  // The hub does not check the major of `pipeline` (only the pipeline implementation knows whether
  // it can read it).
  assertEquals(manifest.models["anima-turbo"].pipeline, { name: "anima", major: 2 });
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "the major of pipeline 'anima/2' is not supported",
  );
});

test("fromAssets: a pipelineConfig schema violation fails at construction", async () => {
  const manifest = parseManifest(manifestText({ pipelineConfig: {} }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig.scheduler: missing",
  );
});

/**
 * The same shape as the `pipelineConfig` that `manifestText` bakes in (to replace
 * `scheduler.type`).
 */
const pipelineConfig = (scheduler: Record<string, unknown>): Record<string, unknown> => ({
  scheduler,
  defaults: { steps: 10, guidanceScale: 1, resolution: { width: 1024, height: 1024 } },
});

test("fromAssets: a manifest that declares scheduler.type gets to the same point too (the order of the checks is unchanged)", async () => {
  // Checks that adding the field did not move the order of the construction stages (manifest
  // contract → assets → GPU). A manifest declaring a valid type passes the contract checks, goes on
  // to the assets, and fails with **the same wording** as "the contrapositive of the check order".
  const manifest = parseManifest(
    manifestText({
      pipelineConfig: pipelineConfig({ type: "dpmpp-2m", shift: 3, numTrainTimesteps: 1000 }),
    }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "asset 'tokenizer' is missing",
  );
});

test("fromAssets: an unknown scheduler.type fails before touching the assets", async () => {
  // If a misspelling silently degraded to the default euler, the update rule the distributor
  // declared and the one that runs would disagree unnoticed. It fails with **the type's wording**
  // even with empty assets = the contract check comes before the asset parse.
  const manifest = parseManifest(
    manifestText({
      pipelineConfig: pipelineConfig({ type: "dpm-solver++", shift: 3, numTrainTimesteps: 1000 }),
    }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "pipelineConfig.scheduler.type: expected 'euler' / 'dpmpp-2m'",
  );
});

test("fromAssets: a nonexistent quant fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { quant: "f16+dit8-a8" }),
    Error,
    "available: f16+dit8-a8-attn8-s16",
  );
});

test("fromAssets: a nonexistent model fails with the list of available ones", async () => {
  // An axis added in v2. When a model name is mistyped, it answers "then what is there" from the
  // primary information (ADR 0041 §8).
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { model: "anima-xl" }),
    Error,
    "available: anima-turbo",
  );
});

test("fromAssets: the assets are touched only after every manifest contract is met (the contrapositive of the check order)", async () => {
  // The flip side of the cases above, which "fail with the manifest's wording even with empty
  // assets". With a valid manifest the checks go on to the assets and fail on the missing
  // `tokenizer` (= both the contract checks and the asset parse come before the GPU). If this
  // turned into an `acquireGpu` failure in an environment without a GPU, this check would turn red.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }),
    Error,
    "asset 'tokenizer' is missing",
  );
});

test("fromAssets: without a component's container it fails with the spellings of both forms (the opener's diagnostic)", async () => {
  // A Record without all the containers fails on **the missing component** (not with the manifest's
  // wording) = the contrapositive showing that the check above is not dragged along by missing
  // assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'text_encoder' has no container",
  );
});

test("fromAssets: an already-aborted signal rejects with the reason itself before touching the assets", async () => {
  // The UI's cancel button must also act on the assembly after the download finished (asset parse →
  // acquireGpu). That the entry check comes **before** the asset parse is checked with the same
  // manifest + empty assets as the check above: a call that would fail with "asset 'tokenizer' is
  // missing" without a signal fails with the reason when already aborted.
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("cancel button");
  controller.abort(reason);
  const error = await assertRejects(() =>
    Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { signal: controller.signal }),
  );
  // Not wrapped (the consumer can identify its own abort with `error ===
  // controller.signal.reason`).
  assertStrictEquals(error, reason);
});

test("fromAssets: an abort that arrives after execution has started also takes effect at the first stage boundary", async () => {
  // The check above sees only "already aborted before the call". The cancel button is pressed
  // **while the assembly is running**, so the stage-boundary check is a dead letter unless it
  // yields to the event loop first (abort() is delivered as a task — during a synchronous parse it
  // is never observed). Scheduling it and then calling fails with the reason, not with the asset
  // error ("asset 'tokenizer' is missing").
  const manifest = parseManifest(manifestText());
  const controller = new AbortController();
  const reason = new Error("cancel button (while running)");
  setTimeout(() => controller.abort(reason), 0);
  const error = await assertRejects(() =>
    Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { signal: controller.signal }),
  );
  assertStrictEquals(error, reason);
});

// ---- Explicit execution knobs (explicit > quant declaration > runtime default) ------------------
// ADR 0058 addendum 2026-09-26.

/**
 * A manifest with only the quant's `session` replaced (the other fields are the skeleton of {@link
 * manifestText}).
 */
const manifestWithSession = (session: Record<string, unknown>) =>
  parseManifest(
    manifestText({
      quants: {
        "f16+dit8-a8-attn8-s16": {
          weights: Object.fromEntries(WEIGHT_NAMES.map((name) => [name, "f16"])),
          session,
        },
      },
    }),
  );

/** Reproduces the construction options of a caller from outside the types (a JS consumer). */
const optionsWith = (key: string, value: unknown): PipelineOptions => {
  const options: PipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

test("fromAssets: when combining the explicit choice and the quant declaration gives a combination the runtime does not accept, a ModelInputError before touching the assets", async () => {
  // The declaration is s16 only (it passes on its own). Only together with the explicit c16 does it
  // become a shape the runtime does not accept, so the remedy is "fix the choice" = input-caused.
  // The assets are empty — had it not failed at admission, it would fail on the missing component.
  await assertRejects(
    () =>
      Pipeline.fromAssets(
        { manifest: manifestWithSession({ attentionScoreStorage: "f16" }), assets: emptyAssets },
        { attentionCompute: "f16" },
      ),
    ModelInputError,
    "cannot be given together",
  );
});

test("fromAssets: an invalid explicit value and a knob Anima does not accept give a ModelInputError before touching the assets", async () => {
  const manifest = parseManifest(manifestText());
  // null falls back neither to the quant declaration nor to the runtime default.
  await assertRejects(
    () =>
      Pipeline.fromAssets({ manifest, assets: emptyAssets }, optionsWith("linearCompute", null)),
    ModelInputError,
    "linearCompute is invalid",
  );
  // Parallel GEMV is outside Anima's acceptance table — silently dropping it would mean "I chose it
  // but it has no effect".
  await assertRejects(
    () =>
      Pipeline.fromAssets(
        { manifest, assets: emptyAssets },
        optionsWith("linearGemvReduce", "parallel"),
      ),
    ModelInputError,
    "linearGemvReduce cannot be specified for this family",
  );
});

test("fromAssets: an accepted explicit choice passes admission and gets to the asset stage (the counterpart showing the check is not vacuous)", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () =>
      Pipeline.fromAssets(
        { manifest, assets: COMPONENTS },
        {
          linearCompute: "f16",
          attentionScoreStorage: "f16",
        },
      ),
    Error,
    "asset 'tokenizer' is missing",
  );
});

test("fromAssets: when the shared GPU lacks the explicit f16 compute, it fails by name before touching the assets", async () => {
  // The quant declaration does not require shader-f16. The requirement is derived from the
  // effective settings (no silent fallback to f32, and no deferring it to Session construction
  // after the weights are uploaded).
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () =>
      Pipeline.fromAssets(
        { manifest, assets: emptyAssets },
        {
          gpu: fakeGpuContext(fakeDevice()),
          linearCompute: "f16",
        },
      ),
    Error,
    "shader-f16",
  );
});

// ---- The DiT residency construction option (ADR 0112) ------------------------------------------

test("fromAssets: an unknown residency spelling gives a ModelInputError before touching the assets", async () => {
  // If a misspelling silently became per-stage operation, one would notice that the opt-in has no
  // effect only through speed. The assets are empty — had it not failed at admission, it would fail
  // on the missing component.
  const manifest = parseManifest(manifestText());
  for (const value of ["transformers", null]) {
    await assertRejects(
      () => Pipeline.fromAssets({ manifest, assets: emptyAssets }, optionsWith("residency", value)),
      ModelInputError,
      "anima.Pipeline: residency: expected 'per-stage' / 'transformer'",
    );
  }
});

test("fromAssets: an accepted residency passes admission and gets to the asset stage (the counterpart showing the check is not vacuous)", async () => {
  // Construction opens no Session (the resident DiT is built at the first generate), so even with
  // transformer the order of the construction stages does not change.
  const manifest = parseManifest(manifestText());
  for (const residency of ["per-stage", "transformer"] as const) {
    await assertRejects(
      () => Pipeline.fromAssets({ manifest, assets: COMPONENTS }, { residency }),
      Error,
      "asset 'tokenizer' is missing",
    );
  }
});

/**
 * A set of assets that builds up to just before the GPU. Construction opens no Session, so it
 * builds even with a fake GPU — all it needs is the assets that get parsed (the minimal JSON of the
 * two tokenizers and the rope base tables in the `transformer` container).
 */
const buildableAssets = async (): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const encode = (value: unknown): Uint8Array<ArrayBuffer> =>
    new TextEncoder().encode(JSON.stringify(value));
  // The same minimal shape as the createTokenizers check in `anima_tokenizer_test.ts` ("aba" is two
  // or more tokens in both).
  const tokenizer = encode({
    vocabText: "a\nb\nab",
    vocabCount: 3,
    mergesText: "a b",
    mergesCount: 1,
    addedTokens: [],
    classes: { letter: [[0x61, 0x7a]], number: [], space: [[0x20, 0x20]] },
    caseFold: [],
    nfcSegments: [],
    maxLength: 8,
  });
  const tokenizer2 = encode({
    vocabText: "<pad>\n</s>\n<unk>\n▁\na\nb",
    scores: [0, 0, 0, -1, -1, -2],
    unkId: 2,
    eosId: 1,
    addedTokens: [],
    space: [[0x20, 0x20]],
    normalizer: { single: [], multi: [], extend: [], breakAfter: [], prepend: [] },
    maxLength: 8,
  });
  const ropeTable = (): DumpTensor => ({ dtype: "F32", shape: [2, 1], data: new Float32Array(2) });
  const rope = writeSafetensors(
    new Map(
      ["t", "h", "w"].flatMap((axis) => [
        [`cos_${axis}`, ropeTable()] as const,
        [`sin_${axis}`, ropeTable()] as const,
      ]),
    ),
    {},
  );
  const transformer = await tensorlessContainer(
    "transformer",
    { inputs: [{ name: "x", shape: [1, 4] }], output: { name: "y", shape: [1, 4] } },
    [{ name: "rope_base", role: "rope-base", bytes: rope }],
  );
  const others = Object.fromEntries(
    Object.entries(COMPONENTS).filter(([key]) => !key.startsWith("transformer[")),
  );
  return {
    ...others,
    ...partAssets("transformer", transformer),
    tokenizer,
    tokenizer_2: tokenizer2,
  };
};

/**
 * A request with one field set to a value that does not pass the TS types (reproduces a JS caller's
 * misspelling).
 */
const requestWith = (key: string, value: unknown): GenerateRequest => {
  const request: GenerateRequest = { prompt: "aba" };
  Object.defineProperty(request, key, { value, enumerable: true });
  return request;
};

test("generate: an unknown residency spelling in the request gives a ModelInputError before touching the GPU (the entry wiring)", async () => {
  // If the misspelling "transformers" slipped past the entry, it would run as per-stage operation
  // (the residency decision only looks for a match with "transformer") with no notice = a silently
  // slow path. The fake GPU cannot build a Session, so once past the entry it would turn into
  // another exception at the first stage's construction (the counterpart test below).
  const manifest = parseManifest(manifestText());
  await using pipeline = await Pipeline.fromAssets(
    { manifest, assets: await buildableAssets() },
    { gpu: fakeGpuContext(fakeDevice()) },
  );
  const events: string[] = [];
  const request = requestWith("residency", "transformers");
  await assertRejects(
    () => pipeline.generate({ ...request, onEvent: (event) => void events.push(event.kind) }),
    ModelInputError,
    "residency: expected 'per-stage' / 'transformer' (actual 'transformers')",
  );
  assertEquals(events, [], "a stage was entered before the entry check");
});

test("generate: an accepted residency passes the entry and gets to the first stage (GPU) (the counterpart showing the check is not vacuous)", async () => {
  const manifest = parseManifest(manifestText());
  await using pipeline = await Pipeline.fromAssets(
    { manifest, assets: await buildableAssets() },
    { gpu: fakeGpuContext(fakeDevice()) },
  );
  for (const residency of ["per-stage", "transformer"] as const) {
    const events: string[] = [];
    const error = await assertRejects(() =>
      pipeline.generate({
        prompt: "aba",
        residency,
        onEvent: (event) => {
          if (event.kind === "stage") events.push(`${event.component}:${event.at}`);
        },
      }),
    );
    assert(!(error instanceof ModelInputError), `${residency}: rejected at the entry (${error})`);
    assertEquals(events, ["text_encoder:start"], `${residency}: the first stage was not entered`);
  }
});

test("generate: a guidanceScale that does not stay finite in f32 gives a ModelInputError before touching the GPU", async () => {
  // CFG multiplies by the factor in f32. A value finite in f64 but Infinity in f32 makes the
  // combination ±Inf / NaN.
  const manifest = parseManifest(manifestText());
  await using pipeline = await Pipeline.fromAssets(
    { manifest, assets: await buildableAssets() },
    { gpu: fakeGpuContext(fakeDevice()) },
  );
  for (const guidanceScale of [1e39, Number.MAX_VALUE, -1e39]) {
    const events: string[] = [];
    await assertRejects(
      () =>
        pipeline.generate({
          prompt: "aba",
          negativePrompt: "b",
          guidanceScale,
          onEvent: (event) => void events.push(event.kind),
        }),
      ModelInputError,
      `guidanceScale ${guidanceScale} is not a finite number that fits in f32`,
    );
    assertEquals(events, [], `${guidanceScale}: a stage was entered before the entry check`);
  }
});

test("resolveNegativePrompt: rejects a negativePrompt passed with guidanceScale 1", () => {
  // Silently accepting a knob with no effect means the user thinks they set it while not a single
  // character is used (guidance=1 skips the whole uncond branch — `needsUncond` in
  // `generation/dpm-solver-multistep.ts`).
  assertThrows(
    () => resolveNegativePrompt("worst quality", undefined, 1),
    ModelInputError,
    "negativePrompt has no effect",
  );
  // Only **an explicit choice in the request** is rejected. A state where only the manifest's
  // default is filled in is not "passing a knob with no effect", so it passes even at guidance=1.
  // This is **the very default path** of the `anima-turbo` distribution (defaults.guidanceScale = 1
  // + defaults.negativePrompt), and a refactor that tightened this would make turbo's default
  // generation throw outright.
  assertEquals(resolveNegativePrompt(undefined, "default negative", 1), "default negative");
  // Even with a fallback, a requested value fails = the branch between rejecting and not rejecting
  // is decided **only by whether requested is present**, not by whether a fallback is.
  assertThrows(
    () => resolveNegativePrompt("worst quality", "default negative", 1),
    ModelInputError,
    "negativePrompt has no effect",
  );
  assertEquals(resolveNegativePrompt(undefined, undefined, 1), undefined);
});

test("resolveNegativePrompt: rejects a missing negativePrompt under settings that compute uncond", () => {
  // The shape where it is in neither the request nor the manifest's defaults. Without rejecting it
  // here, the uncond side's text would enter the GPU path empty.
  assertThrows(
    () => resolveNegativePrompt(undefined, undefined, 7),
    ModelInputError,
    "negativePrompt is required",
  );
  // The source order is request > defaults. Either one is enough to pass.
  assertEquals(resolveNegativePrompt(undefined, "default negative", 7), "default negative");
  assertEquals(
    resolveNegativePrompt("requested negative", "default negative", 7),
    "requested negative",
  );
});

test("assertAcceptableSeed: the generation entry and Randn go through the same accepted set", () => {
  // `generate`'s entry calls this check (the `Randn` is created after running the text encoder /
  // conditioner and uploading all the DiT weights, so a check only on the generator side would fail
  // at the end of a GB-scale load). That the accepted set is not written in two places is confirmed
  // with the same input and the same wording.
  for (const seed of [-1, 1.5, Number.NaN, Number.MAX_SAFE_INTEGER + 1]) {
    const direct = assertThrows(
      () => assertAcceptableSeed(seed),
      ModelInputError,
      "non-negative safe integer",
    );
    const viaRandn = assertThrows(
      () => new Randn(seed),
      ModelInputError,
      "non-negative safe integer",
    );
    assertEquals(direct.message, viaRandn.message, `diagnostic for seed ${seed}`);
  }
  for (const seed of [0, 42, Number.MAX_SAFE_INTEGER]) assertAcceptableSeed(seed);
});

test("denoiseStep: scheduler.type selects the update rule (euler and dpmpp-2m diverge on the same input)", () => {
  // The ladder and the input are a hand-placed single element over 4 steps. `index 2` is **the only
  // position where the 2nd-order term actually takes effect** — both ends (0 / 3) fall to 1st
  // order, and at `index 1` λ_s1 = −inf (the leading σ is exactly 1), so the 2nd-order term becomes
  // exactly 0 (the ∓inf section of the update rule's doc).
  //
  // The expected values are a hand calculation solving diffusers' formulas in f64
  // (σ=[1,.5,.25,.125,0] / x=1 / v=2 / previous x0=0):
  //   x0 = x − σ_s0·v = 0.5
  //   1st-order term: x + (σ_t − σ_s0)·v = 0.75 — in flow matching DPM++'s 1st order is
  //                   identical to Euler
  //   2nd-order term: h = log(7/3), r0 = (λ_s0 − λ_s1)/h → 0.75 + 0.0964055 = 0.8464055
  // The difference between the two paths is **exactly the 2nd-order term**, so the value moves if
  // the choice is reversed or if the 2nd-order term is dropped.
  const input = {
    sample: Float32Array.from([1]),
    cond: Float32Array.from([2]),
    uncond: undefined,
    guidance: 1,
    sigmas: Float32Array.from([1, 0.5, 0.25, 0.125, 0]),
    index: 2,
    previousX0: Float32Array.from([0]),
  };

  // euler does not read the history even when given one (the default path of distributed manifests
  // does not move here).
  const euler = denoiseStep("euler", input);
  assertEquals([...euler.sample], [0.75]);
  assertEquals(euler.previousX0, undefined, "Euler keeps no history to pass to the next step");

  const dpm = denoiseStep("dpmpp-2m", input);
  assertAlmostEquals(dpm.sample[0], 0.8464055, 1e-6);
  // What is passed to the next step is this step's x0 prediction itself. If this were cut, DPM++ 2M
  // would silently fall to 1st order (= the same value as Euler).
  const carried = dpm.previousX0;
  assert(carried !== undefined, "DPM++ 2M carries x0 over to the next step");
  assertEquals([...carried], [0.5]);
});

test("denoiseStep: a non-finite latent after the update fails naming the step and is not passed to the next step (with either update rule)", () => {
  // Even with a factor of 1e38, finite in f32, the product with the difference 10 overflows f32,
  // the combination becomes +Inf and the updated value -Inf. A shape the entry's f32 check alone
  // cannot prevent (the overflow comes from the pair of the factor and the DiT output).
  const input = {
    sample: Float32Array.from([0, 0.5]),
    cond: Float32Array.from([0.25, 10]),
    uncond: Float32Array.from([0.25, 0]),
    guidance: 1e38,
    sigmas: Float32Array.from([1, 0.5, 0]),
    index: 0,
    previousX0: undefined,
  };
  for (const type of ["euler", "dpmpp-2m"] as const) {
    assertThrows(
      () => denoiseStep(type, input),
      Error,
      "step 1/2: element 1 of the updated latent is non-finite",
      type,
    );
  }
  // Control: with the same input and only an ordinary factor, it passes (the check is not vacuous).
  for (const type of ["euler", "dpmpp-2m"] as const) {
    const update = denoiseStep(type, { ...input, guidance: 4 });
    assert(update.sample.every(Number.isFinite), type);
  }
});

test("assertAnimaSamplerType: the request-side spelling also accepts exactly two words, and the message names the origin", () => {
  // The manifest side (`where = "pipelineConfig.scheduler.type"`) is checked by
  // `anima_config_test.ts`, but the request-side call (`where = "sampler"` in `pipeline.ts`) is
  // inside `generate()` and cannot be reached without a real GPU + real assets. The accepted set
  // itself is pinned directly as a pure function (the same approach as `resolveNegativePrompt`).
  // The wiring stays with the real-GPU check (the `sampler: "dpmpp-2m"` case of
  // `e2e_anima_test.ts`).
  assertEquals(assertAnimaSamplerType("euler", "sampler"), "euler");
  assertEquals(assertAnimaSamplerType("dpmpp-2m", "sampler"), "dpmpp-2m");

  const misspelled = assertThrows(
    () => assertAnimaSamplerType("dpm++", "sampler"),
    Error,
    "sampler: expected 'euler' / 'dpmpp-2m'",
  );
  assertStringIncludes(misspelled.message, "actual 'dpm++'");
  // Even for a non-string it shows what "actually" arrived (the diagnostic does not vanish at the
  // type's edge).
  for (const [value, expected] of [
    [2, "actual 2"],
    [undefined, "actual undefined"],
  ] as const) {
    const thrown = assertThrows(() => assertAnimaSamplerType(value, "sampler"), Error);
    assertStringIncludes(thrown.message, expected);
  }
});

test("SAMPLER_TYPES: the published list is the accepted set itself and consumers cannot rewrite it", () => {
  // A CLI / UI checks arguments and lists the choices with this list. If the list and the check had
  // separate tables, there would be "words you can choose that fail", so the public surface
  // (`./anima`) pins that every word of the list passes the check.
  assertEquals([...SAMPLER_TYPES], ["euler", "dpmpp-2m"]);
  for (const name of SAMPLER_TYPES) {
    assertEquals(assertAnimaSamplerType(name, "sampler"), name);
  }
  // The shape where a JS consumer that bypasses the types adds or replaces a word in the shared
  // list. If the freeze came off, the accepted set of every later pipeline would widen silently
  // (unknown spellings would no longer fail before the GPU).
  assertThrows(() => Reflect.apply(Array.prototype.push, SAMPLER_TYPES, ["dpm++"]), TypeError);
  assertEquals(Reflect.set(SAMPLER_TYPES, 0, "dpm++"), false);
  assertEquals([...SAMPLER_TYPES], ["euler", "dpmpp-2m"]);
  assertThrows(() => assertAnimaSamplerType("dpm++", "sampler"), Error, "actual 'dpm++'");
});

test("latentSnapshot: copies the latent at the time it was bound (the copy does not change as steps advance)", () => {
  const shape = [1, 2, 1, 1];
  // Reproduces the denoise loop: `cfgEulerStep` is a pure function, so `current` is replaced by **a
  // new array** at each step. The event hook is created and passed per step.
  const step1 = Float32Array.from([1, 2]);
  const first = latentSnapshot(step1, shape);
  const step2 = Float32Array.from([3, 4]);
  const second = latentSnapshot(step2, shape);

  // Even when called after the whole generation is over, it returns the value "at the time it was
  // made". An implementation that captured the loop variable would give [3, 4] for both (a mix-up
  // subscribers cannot detect).
  assertEquals(Array.from(first().data), [1, 2]);
  assertEquals(Array.from(second().data), [3, 4]);
  assertEquals(first().shape, shape);

  // Each call returns **a separate copy**. Even if a subscriber rewrites the array it received,
  // neither the next copy nor the pipeline's array is affected (structurally ruling out the
  // accident of a held reference).
  const copy = first();
  assertNotStrictEquals(copy.data, step1);
  copy.data[0] = 99;
  assertEquals(Array.from(first().data), [1, 2]);
  assertEquals(Array.from(step1), [1, 2]);
  assertNotStrictEquals(copy.data, first().data);

  // The same goes for shape. The actual argument is **the plain mutable array** `plan.latentShape`,
  // so returning it by reference would let a subscriber's rewrite break the patchify / rope
  // correspondence of the next step (the element count does not change, so the final "output image
  // dimensions == requested resolution" check also passes, and something else comes out silently).
  assertNotStrictEquals(copy.shape, shape);
  shape[2] = 99;
  assertEquals(copy.shape, [1, 2, 1, 1]);
});

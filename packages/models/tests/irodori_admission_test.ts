// The failure paths of the **graph check** of the family admission (`admitIrodori`). Needs neither
// a GPU nor real assets.
//
// The check covers `assertStaticDim` / `assertOutputScale` / `assertOutputDim` / exactly one
// symbolic dimension in `dit` / the `dit_context` → `dit` wiring (no symbolic dimensions, K / V
// pairs, the set of names, shapes and dtypes — ADR 0114), each a check whose doc says "MUST: do not
// drop this. … a **silent wrong value** results" (the shape still matches while the condition is
// read from another position). `irodori_pipeline_test.ts` deliberately leaves this layer out
// because it "has no vehicle to build synthetic containers", but that vehicle now lives in
// `tests/helpers/container-fixture.ts`, so the layer can be exercised without real assets.
//
// The observation setup: only **the 9 graphs** are passed as assets, without `tokenizer`.
//  - the correct 9 → what fails is `asset 'tokenizer' is missing` (= every check passed and it
//    moved on to the next stage)
//  - breaking exactly one axis → rejected with a message containing that axis name, the expected
//    value and the measured value
// and this contrapositive pins both the checks themselves and their position (before going to
// acquire the GPU). The GPU is never touched.

import { test } from "vite-plus/test";
import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { admitIrodori, assetOpener } from "../src/irodori/admission.ts";
import { Pipeline, type PipelineOptions } from "../src/irodori/pipeline.ts";
import { ModelInputError } from "../src/errors.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import type { ModelInput } from "../../runtime/tests/helpers/container-write.ts";
import {
  declaredContainer,
  parseIrDeclarationValue,
  partAssets,
  tensorlessContainer,
  type TensorlessGraphSpec,
  tensorlessInput,
  writeContainer,
} from "./helpers/container-fixture.ts";
import {
  HUB_URL,
  MANIFEST_PATH,
  REPO,
  serveContainer,
  serveRepos,
  SHA,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";

/** The fake repo components are replaced from. */
const OTHER_REPO = "karume-test/replacement";

/** The real `pipelineConfig` of `models/karume-irodori-v4-small/karume.json` (23 fields). */
const CONFIG = {
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

const COMPONENTS = [
  "backbone",
  "text_proj",
  "caption_proj",
  "speaker",
  "duration",
  "dit",
  "dit_context",
  "codec_decoder",
  "codec_encoder",
] as const;

type Component = (typeof COMPONENTS)[number];
/**
 * The components a single-output declaration is enough for (`dit_context` outputs K / V pairs, so
 * it has another vehicle — {@link ContextSpec}).
 */
type SingleOutput = Exclude<Component, "dit_context">;

/**
 * The real DiT's block count (12 for the v4 line). **A number for the test to copy the real
 * declarations**; the implementation does not bake it in (the block count is derived from the
 * outputs of `dit_context` — ADR 0114).
 */
const BLOCKS = 12;
/**
 * The row count of the context concatenating the 3 conditions (text + speaker + caption = 1519).
 */
const CONTEXT_ROWS = CONFIG.maxTextLen + CONFIG.speakerRows + CONFIG.maxCaptionLen;
/** The shape of one context K / V (20 heads × 64 head_dim — after reshape, before transpose). */
const CONTEXT_SHAPE = [1, CONTEXT_ROWS, 20, 64] as const;
/** The outputs of `dit_context` = the K / V inputs of `dit` (K, V in block order). */
const CONTEXT_NAMES: readonly string[] = Array.from({ length: BLOCKS }, (_, block) => [
  `context_k_${block}`,
  `context_v_${block}`,
]).flat();

/**
 * One declared value (dtype f32 when omitted — the same default as the check's vehicle
 * `TensorlessGraphSpec`).
 */
type ValueSpec = {
  readonly name: string;
  readonly shape: readonly (number | string)[];
  readonly dtype?: "f32" | "i32";
};

/**
 * The declaration of `dit_context` (several outputs — `TensorlessGraphSpec` is a single-output
 * vehicle, so it is kept separately).
 */
type ContextSpec = {
  readonly symbols?: readonly string[];
  readonly inputs: readonly ValueSpec[];
  readonly outputs: readonly ValueSpec[];
};

/**
 * The container inputs of `dit_context` (each output's node is "one `mul` taking the first input
 * of the output's dtype through 2 ports" — like `declarationOf` of `container-fixture.ts`, the
 * purpose is checking declarations, and it is never run. The dtypes match so that the IR's op
 * contract checks pass).
 */
const contextInput = (spec: ContextSpec): ModelInput => ({
  graphs: {
    dit_context: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: spec.symbols ?? [],
      inputs: spec.inputs.map((input) => ({
        name: input.name,
        dtype: input.dtype ?? "f32",
        shape: input.shape,
      })),
      outputs: spec.outputs.map((output) => output.name),
      initializers: {},
      values: Object.fromEntries(
        spec.outputs.map((output) => [
          output.name,
          { dtype: output.dtype ?? "f32", shape: output.shape },
        ]),
      ),
      states: {},
      nodes: spec.outputs.map((output) => {
        const source =
          spec.inputs.find((input) => (input.dtype ?? "f32") === (output.dtype ?? "f32")) ??
          spec.inputs[0];
        return { op: "mul", ins: [source.name, source.name], outs: [output.name], attrs: {} };
      }),
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

/** The same `dit_context` declaration as the real thing (3 conditions → 24 K / V). */
const contextSpec = (): ContextSpec => ({
  inputs: [
    { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
    { name: "speaker_state", shape: [1, CONFIG.speakerRows, CONFIG.speakerDim] },
    { name: "caption_state", shape: [1, CONFIG.maxCaptionLen, CONFIG.captionDim] },
  ],
  outputs: CONTEXT_NAMES.map((name) => ({ name, shape: CONTEXT_SHAPE })),
});

/**
 * The 8 minimal graphs that pass every check (the values of `CONFIG` appear in the declarations as
 * they are — `dit_context` is {@link contextSpec}).
 */
const graphSpecs = (): Record<SingleOutput, TensorlessGraphSpec> => ({
  // Outside the check (any declaration will do).
  backbone: {
    inputs: [{ name: "input_ids", shape: [1, 4] }],
    output: { name: "hidden", shape: [1, 4, CONFIG.textDim] },
  },
  text_proj: {
    inputs: [{ name: "hidden", shape: [1, 4, CONFIG.textDim] }],
    output: { name: "text_state", shape: [1, 4, CONFIG.textDim] },
  },
  caption_proj: {
    inputs: [{ name: "hidden", shape: [1, 4, CONFIG.captionDim] }],
    output: { name: "caption_state", shape: [1, 4, CONFIG.captionDim] },
  },
  // The patch width of the reference latent (latentDim × speakerPatchSize).
  speaker: {
    inputs: [{ name: "latent", shape: [1, 4, CONFIG.latentDim * CONFIG.speakerPatchSize] }],
    output: { name: "speaker_vec", shape: [1, CONFIG.speakerDim] },
  },
  duration: {
    inputs: [
      { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
      { name: "speaker_vec", shape: [1, CONFIG.speakerDim] },
      { name: "caption_vec", shape: [1, CONFIG.captionDim] },
    ],
    output: { name: "log_frames", shape: [1] },
  },
  // Holds no 3 condition states and receives the outputs of `dit_context` (24 K / V) under the same
  // names and shapes.
  dit: {
    symbols: ["S"],
    inputs: [
      { name: "x_t", shape: [1, "S", CONFIG.latentDim] },
      { name: "t_embed", shape: [1, CONFIG.timestepEmbedDim] },
      { name: "mask", shape: [1, 1, 1, `S+${CONTEXT_ROWS}`] },
      ...CONTEXT_NAMES.map((name) => ({ name, shape: CONTEXT_SHAPE })),
    ],
    output: { name: "v", shape: [1, "S", CONFIG.latentDim] },
  },
  // 1 latent frame → hopLength samples (the check goes as far as the **coefficient** of the
  // output's derived dimension).
  codec_decoder: {
    symbols: ["S"],
    inputs: [{ name: "latent", shape: [1, "S", CONFIG.latentDim] }],
    output: { name: "wav", shape: [1, 1, `${CONFIG.hopLength}S`] },
  },
  codec_encoder: {
    symbols: ["T"],
    inputs: [{ name: "wav", shape: [1, "T", CONFIG.hopLength] }],
    output: { name: "latent", shape: [1, "T", CONFIG.latentDim] },
  },
});

/**
 * Replacing components (single-output components are `TensorlessGraphSpec`, `dit_context` is
 * {@link ContextSpec}).
 */
type Patch = Partial<Record<SingleOutput, TensorlessGraphSpec>> & {
  readonly dit_context?: ContextSpec;
};

/**
 * Builds the 9 containers with exactly one replaced (`tokenizer` is not included — the observation
 * setup).
 */
const assetsWith = async (patch: Patch = {}): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const specs = { ...graphSpecs(), ...patch };
  const context = patch.dit_context ?? contextSpec();
  let assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const name of COMPONENTS) {
    const container =
      name === "dit_context"
        ? await writeContainer(contextInput(context))
        : await tensorlessContainer(name, specs[name]);
    assets = { ...assets, ...partAssets(name, container) };
  }
  return assets;
};

const FILE = { size: 16, sha256: "a".repeat(64) };

const manifestText = (pipelineConfig: Readonly<Record<string, unknown>> = CONFIG): string => {
  let weights: Record<string, unknown> = {};
  let mapping: Record<string, string> = {};
  for (const name of COMPONENTS) {
    weights = { ...weights, [name]: { f32: declaredContainer(`${name}/model.f32`) } };
    mapping = { ...mapping, [name]: "f32" };
  }
  return JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "v4-small",
    models: {
      "v4-small": {
        pipeline: "irodori/1",
        weights,
        assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
        quants: { f32: { weights: mapping, session: {} } },
        defaultQuant: "f32",
        pipelineConfig,
      },
    },
  });
};

const build = async (
  patch: Patch = {},
  pipelineConfig: Readonly<Record<string, unknown>> = CONFIG,
): Promise<unknown> =>
  await Pipeline.fromAssets({
    manifest: parseManifest(manifestText(pipelineConfig)),
    assets: await assetsWith(patch),
  });

/**
 * Breaks exactly one axis of one entry of an input list (independent of the component vehicle's
 * type).
 */
const breakAxis = <
  Input extends { readonly name: string; readonly shape: readonly (number | string)[] },
>(
  inputs: readonly Input[],
  inputName: string,
  axis: number,
  value: number | string,
): Input[] =>
  inputs.map((input) =>
    input.name === inputName
      ? { ...input, shape: input.shape.map((dim, index) => (index === axis ? value : dim)) }
      : input,
  );

/** Breaks exactly one axis of one input of one graph. */
const breakInput = (
  component: SingleOutput | "dit_context",
  inputName: string,
  axis: number,
  value: number | string,
): Patch => {
  if (component === "dit_context") {
    const spec = contextSpec();
    return { dit_context: { ...spec, inputs: breakAxis(spec.inputs, inputName, axis, value) } };
  }
  const spec = graphSpecs()[component];
  return { [component]: { ...spec, inputs: breakAxis(spec.inputs, inputName, axis, value) } };
};

/** Breaks exactly one axis of one graph's output. */
const breakOutput = (component: SingleOutput, axis: number, value: number | string): Patch => {
  const spec = graphSpecs()[component];
  return {
    [component]: {
      ...spec,
      output: {
        ...spec.output,
        shape: spec.output.shape.map((dim, index) => (index === axis ? value : dim)),
      },
    },
  };
};

test("admitIrodori: a graph meeting every check proceeds to the asset stage (the contrapositive of the check position)", async () => {
  // `tokenizer` is not passed, so once every check passes that is what fails. If this fails with
  // "graph input '…' axis …", a declaration that should be valid is caught by a check.
  await assertRejects(() => build(), Error, "irodori: asset 'tokenizer' is missing");
});

test("admitIrodori: each broken axis fails naming the axis, the expected value and the measured value (12 points)", async () => {
  // Each case moves exactly one axis to near its current value (keeping the shape's rank) —
  // changing the rank would hit another check and could not show that "this axis's check is alive".
  const cases: readonly {
    readonly where: string;
    readonly patch: Patch;
    readonly actual: string;
    readonly expected: string;
  }[] = [
    {
      where: "latentDim",
      patch: breakInput("dit", "x_t", 2, 33),
      actual: "'x_t' axis 2 is 33",
      expected: "pipelineConfig has 32",
    },
    {
      where: "timestepEmbedDim",
      patch: breakInput("dit", "t_embed", 1, 511),
      actual: "'t_embed' axis 1 is 511",
      expected: "pipelineConfig has 512",
    },
    {
      where: "maxTextLen",
      patch: breakInput("dit_context", "text_state", 1, 255),
      actual: "'text_state' axis 1 is 255",
      expected: "pipelineConfig has 256",
    },
    {
      where: "textDim",
      patch: breakInput("dit_context", "text_state", 2, 511),
      actual: "'text_state' axis 2 is 511",
      expected: "pipelineConfig has 512",
    },
    {
      where: "speakerRows",
      patch: breakInput("dit_context", "speaker_state", 1, 750),
      actual: "'speaker_state' axis 1 is 750",
      expected: "pipelineConfig has 751",
    },
    {
      where: "speakerDim",
      patch: breakInput("dit_context", "speaker_state", 2, 767),
      actual: "'speaker_state' axis 2 is 767",
      expected: "pipelineConfig has 768",
    },
    {
      where: "maxCaptionLen",
      patch: breakInput("dit_context", "caption_state", 1, 511),
      actual: "'caption_state' axis 1 is 511",
      expected: "pipelineConfig has 512",
    },
    {
      where: "captionDim",
      patch: breakInput("dit_context", "caption_state", 2, 511),
      actual: "'caption_state' axis 2 is 511",
      expected: "pipelineConfig has 512",
    },
    {
      where: "textDim",
      patch: breakInput("duration", "text_state", 2, 511),
      actual: "'text_state' axis 2 is 511",
      expected: "pipelineConfig has 512",
    },
    {
      where: "speakerDim",
      patch: breakInput("duration", "speaker_vec", 1, 767),
      actual: "'speaker_vec' axis 1 is 767",
      expected: "pipelineConfig has 768",
    },
    {
      where: "captionDim",
      patch: breakInput("duration", "caption_vec", 1, 511),
      actual: "'caption_vec' axis 1 is 511",
      expected: "pipelineConfig has 512",
    },
    {
      where: "latentDim × speakerPatchSize",
      patch: breakInput("speaker", "latent", 2, 64),
      actual: "'latent' axis 2 is 64",
      expected: "pipelineConfig has 128",
    },
  ];

  for (const testCase of cases) {
    const error = await assertRejects(() => build(testCase.patch), Error);
    assertStringIncludes(error.message, testCase.where);
    assertStringIncludes(error.message, testCase.actual);
    assertStringIncludes(error.message, testCase.expected);
  }
  assertEquals(cases.length, 12);
});

test("admitIrodori: checks codec_decoder's latent width and output scale separately", async () => {
  // The input width (latentDim).
  const width = await assertRejects(
    () => build(breakInput("codec_decoder", "latent", 2, 31)),
    Error,
    "latentDim",
  );
  assertStringIncludes(width.message, "'latent' axis 2 is 31");

  // The **coefficient of the output's derived dimension**. The shape stays "a waveform of a
  // plausible length", so only this catches a broken "1 frame → hopLength samples" (a form where
  // the seconds-based cut and the tail trim silently point at other sample positions).
  const scale = await assertRejects(
    () => build(breakOutput("codec_decoder", 2, "960S")),
    Error,
    "hopLength",
  );
  assertStringIncludes(scale.message, "axis 2 = 960S");
  assertStringIncludes(scale.message, "expects '1920S'");
});

test("admitIrodori: checks codec_encoder's input frame width and output latent width separately", async () => {
  const frame = await assertRejects(
    () => build(breakInput("codec_encoder", "wav", 2, 1919)),
    Error,
    "hopLength",
  );
  assertStringIncludes(frame.message, "'wav' axis 2 is 1919");

  const latent = await assertRejects(
    () => build(breakOutput("codec_encoder", 2, 31)),
    Error,
    "latentDim",
  );
  assertStringIncludes(latent.message, "graph output 'latent' has axis 2 = 31");
});

test("admitIrodori: rejects dit unless it has exactly one symbolic dimension (the same message whatever the path)", async () => {
  // The resident path binds S with this symbol name on every enqueue. Placing the check at run time
  // would ① fail only after the weights were downloaded and ② not run on the host path (devices
  // with `gpuTiming` enabled / `onEvent` subscriptions) (the same distribution would fail with a
  // different message per observation path).
  const specs = graphSpecs();
  const twoSymbols: TensorlessGraphSpec = {
    ...specs.dit,
    symbols: ["S", "B"],
    inputs: [{ name: "x_t", shape: ["B", "S", CONFIG.latentDim] }, ...specs.dit.inputs.slice(1)],
    output: { name: "v", shape: ["B", "S", CONFIG.latentDim] },
  };
  // The order is not the declaration order (`S, B`) because the container's graph description is
  // written in **canonical form** (symbols in code-point order — container-v1 /
  // `canonicalIrDocument`). The message names what was read back.
  await assertRejects(
    () => build({ dit: twoSymbols }),
    Error,
    "dit does not have exactly one symbolic dimension ([B, S])",
  );
});

// ---- The dit_context → dit wiring (ADR 0114) ---------------------------------------------------
//
// Resident inputs are checked by the runtime **only for size** (the doc of `RunInput`). A mismatch
// of shape or dtype passes silently as long as the byte count matches, so the admission check is
// the only detector here.

test("admitIrodori: derives the names of the 24 K / V and the resident bytes from the same declarations as the real thing (178 MiB)", async () => {
  // Calls the admission directly, which stops before the asset stage (tokenizer) — the derived
  // values are not visible from the public surface.
  const admitted = await admitIrodori(
    parseManifest(manifestText()),
    await assetOpener(await assetsWith()),
    {},
  );
  const bytes = CONTEXT_SHAPE.reduce((product, dim) => product * dim, 4);
  assertEquals(
    admitted.ditContextOutputs,
    CONTEXT_NAMES.map((name) => ({ name, byteLength: bytes })),
  );
  // The same number as the design estimate (24 × [1,1519,20,64] f32 = 178.0 MiB).
  assertEquals(
    admitted.ditContextOutputs.reduce((total, output) => total + output.byteLength, 0),
    186_654_720,
  );
});

test("admitIrodori: rejects a dit_context with a symbolic dimension (a resident tensor needs its size at allocation)", async () => {
  const spec = contextSpec();
  const symbolic: ContextSpec = {
    ...spec,
    symbols: ["T"],
    inputs: breakAxis(spec.inputs, "text_state", 0, "T"),
  };
  await assertRejects(
    () => build({ dit_context: symbolic }),
    Error,
    "dit_context has symbolic dimensions ([T])",
  );
});

test("admitIrodori: rejects dit_context outputs that do not form K / V pairs", async () => {
  const spec = contextSpec();
  // 23 with the last V dropped (the dit side is aligned to the same 23, to see that it fails here
  // before the name check).
  const odd = spec.outputs.slice(0, -1);
  const dit = graphSpecs().dit;
  await assertRejects(
    () =>
      build({
        dit_context: { ...spec, outputs: odd },
        dit: { ...dit, inputs: dit.inputs.filter((input) => input.name !== "context_v_11") },
      }),
    Error,
    "dit_context has 23 outputs",
  );
});

test("admitIrodori: fails with the differences on both sides when dit's input names differ from {x_t, t_embed, mask} ∪ the outputs of dit_context", async () => {
  const dit = graphSpecs().dit;
  // Mixes up one V on the dit side only (a mixed-up K / V pair = miswiring on the condition side).
  const renamed = await assertRejects(
    () =>
      build({
        dit: {
          ...dit,
          inputs: dit.inputs.map((input) =>
            input.name === "context_v_11" ? { ...input, name: "context_v_12" } : input,
          ),
        },
      }),
    Error,
    "the dit inputs do not match {x_t, t_embed, mask} ∪ the dit_context outputs",
  );
  assertStringIncludes(renamed.message, "missing from dit: [context_v_11]");
  assertStringIncludes(renamed.message, "only in dit: [context_v_12]");

  // The old form (dit takes the 3 condition states directly and has no K / V besides mask) fails by
  // name.
  const legacy = await assertRejects(
    () =>
      build({
        dit: {
          ...dit,
          inputs: [
            ...dit.inputs.filter((input) => !CONTEXT_NAMES.includes(input.name)),
            { name: "text_state", shape: [1, CONFIG.maxTextLen, CONFIG.textDim] },
          ],
        },
      }),
    Error,
    "only in dit: [text_state]",
  );
  assertStringIncludes(legacy.message, "context_k_0");
});

test("admitIrodori: fails when the shape / dtype of a dit_context output and dit's input of the same name differ (even at the same size)", async () => {
  const spec = contextSpec();
  // A shape with head and head_dim swapped (the same element count = a form that would pass the
  // resident input's size check).
  const transposed: ContextSpec = {
    ...spec,
    outputs: spec.outputs.map((output) =>
      output.name === "context_k_3" ? { ...output, shape: [1, CONTEXT_ROWS, 64, 20] } : output,
    ),
  };
  const shape = await assertRejects(
    () => build({ dit_context: transposed }),
    Error,
    "dit_context output 'context_k_3' (f32 [1, 1519, 64, 20]) does not match the dit input of the same name (f32 [1, 1519, 20, 64])",
  );
  assertStringIncludes(shape.message, "the same static shape and the same dtype are required");

  // Only the dtype differs (both 4 bytes, so this is invisible in the byte count too). To make an
  // i32 output valid as a declaration, one i32 input is added (the graph check looks only at the
  // axes of the 3 conditions).
  const retyped: ContextSpec = {
    ...spec,
    inputs: [...spec.inputs, { name: "ids", shape: [1], dtype: "i32" }],
    outputs: spec.outputs.map((output) =>
      output.name === "context_v_0" ? { ...output, dtype: "i32" } : output,
    ),
  };
  await assertRejects(
    () => build({ dit_context: retyped }),
    Error,
    "dit_context output 'context_v_0' (i32 [1, 1519, 20, 64]) does not match the dit input of the same name (f32 [1, 1519, 20, 64])",
  );
});

test("fromPretrained: a dit replaced through components with a wrong dimension fails before fetching weights (the replacement slot is wired)", async () => {
  // Only the replacement dit is built with latentDim 64 (the original repo's 9 pass every check).
  // If the replacement slot were left unwired, the original repo would pass the checks, proceed to
  // fetching assets and fail with another message.
  const specs = graphSpecs();
  let weights: Record<string, unknown> = {};
  let files: (readonly [string, Uint8Array<ArrayBuffer>])[] = [];
  for (const name of COMPONENTS) {
    const input =
      name === "dit_context" ? contextInput(contextSpec()) : tensorlessInput(name, specs[name]);
    const served = await serveContainer(`${name}/model.f32`, input);
    weights = { ...weights, [name]: { f32: served.entry } };
    files = [...files, ...served.files];
  }
  const wide = [1, "S", CONFIG.latentDim * 2];
  const widened: TensorlessGraphSpec = {
    ...specs.dit,
    inputs: specs.dit.inputs.map((input) =>
      input.name === "x_t" ? { ...input, shape: wide } : input,
    ),
    output: { ...specs.dit.output, shape: wide },
  };
  const replacement = await serveContainer("other/dit.f32", tensorlessInput("dit", widened));
  const modelOf = (entries: Record<string, unknown>) => ({
    test: {
      pipeline: "irodori/1",
      weights: entries,
      assets: { tokenizer: { ...FILE, path: "tokenizer.json" } },
      quants: {
        f32: {
          weights: Object.fromEntries(Object.keys(entries).map((key) => [key, "f32"])),
          session: {},
        },
      },
      defaultQuant: "f32",
      pipelineConfig: CONFIG,
    },
  });
  const mock = serveRepos([
    { repo: REPO, models: modelOf(weights), files },
    {
      repo: OTHER_REPO,
      models: modelOf({ dit: { f32: replacement.entry } }),
      files: replacement.files,
    },
  ]);

  const error = await assertRejects(
    () =>
      Pipeline.fromPretrained(
        { repo: REPO, revision: SHA, hubUrl: HUB_URL },
        {
          fetch: mock.fetch,
          caches: new MemoryCacheStorage(),
          components: { dit: { source: { repo: OTHER_REPO, revision: SHA, hubUrl: HUB_URL } } },
        },
      ),
    Error,
    "the graph descriptor does not match the manifest's declaration",
  );
  assertStringIncludes(error.message, "dit");
  // The replacement's manifest was looked up (= the slot reached the loader). The check needs only
  // the declarations of the 2 manifests, so no container of either repo was fetched (it did not
  // reach the tokenizer asset either).
  assertEquals(
    mock.requests.filter((request) => request.repo === OTHER_REPO),
    [{ repo: OTHER_REPO, path: MANIFEST_PATH }],
  );
  assertEquals(
    mock.paths.filter((path) => path !== MANIFEST_PATH),
    [],
  );
});

// ---- Explicit execution knobs (explicit > quant declaration > runtime default — ADR 0058 addendum
// 2026-09-26) ----------

/** Reproduces the build options of callers that come from outside the types (JS consumers). */
const optionsWith = (key: string, value: unknown): PipelineOptions => {
  const options: PipelineOptions = {};
  Object.defineProperty(options, key, { value, enumerable: true });
  return options;
};

test("admitIrodori: invalid explicit values and attention knobs are a ModelInputError before the graph check", async () => {
  // dit's declaration has one axis broken — if the merge came after the graph check, it would fail
  // with the check's message.
  const manifest = parseManifest(manifestText());
  const broken = await assetsWith(breakInput("dit", "x_t", 2, CONFIG.latentDim + 1));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: broken }, optionsWith("linearCompute", null)),
    ModelInputError,
    "linearCompute is invalid",
  );
  // dit's attention is not covered by the fused-attention contract (the decomposed path), so there
  // is no slot for it to act on.
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: broken }, optionsWith("attentionCompute", "f16")),
    ModelInputError,
    "attentionCompute cannot be specified for this family",
  );
});

test("admitIrodori: an accepted explicit linearCompute passes the checks and proceeds to the asset stage", async () => {
  await assertRejects(
    async () =>
      await Pipeline.fromAssets(
        {
          manifest: parseManifest(manifestText()),
          assets: await assetsWith(),
        },
        { linearCompute: "a8" },
      ),
    Error,
    "irodori: asset 'tokenizer' is missing",
  );
});

test("admitIrodori: an explicit f16 the shared GPU does not support fails by name before acquiring the GPU", async () => {
  // The quant declaration does not require shader-f16. The requirement is derived from the
  // effective settings (it neither falls back to f32 silently nor waits until the Session build
  // after the weights are uploaded).
  await assertRejects(
    async () =>
      await Pipeline.fromAssets(
        {
          manifest: parseManifest(manifestText()),
          assets: await assetsWith(),
        },
        { gpu: fakeGpuContext(fakeDevice()), linearCompute: "f16" },
      ),
    Error,
    "shader-f16",
  );
});

// --- MeanFlow (ADR 0150): the step takes the interval embedding too ----------------------

/** The MeanFlow declaration of the same model numbers (no RF knob — the real v4.1-small-mf's). */
const MEANFLOW_CONFIG: Readonly<Record<string, unknown>> = (() => {
  const {
    initScale: _initScale,
    cfgMinT: _cfgMinT,
    cfgMaxT: _cfgMaxT,
    cfgScales: _cfgScales,
    speakerUncondMode: _speakerUncondMode,
    cfgGuidanceMode: _cfgGuidanceMode,
    ...common
  } = CONFIG;
  return { ...common, flowParameterization: "meanflow", steps: 4 };
})();

/** The MeanFlow `dit`: the RF step's inputs with `delta_embed` (the width of `t_embed`) after `t_embed`. */
const meanflowDit = (width: number = CONFIG.timestepEmbedDim): TensorlessGraphSpec => {
  const dit = graphSpecs().dit;
  return {
    ...dit,
    inputs: dit.inputs.flatMap((input) =>
      input.name === "t_embed" ? [input, { name: "delta_embed", shape: [1, width] }] : [input],
    ),
  };
};

test("admitIrodori: a MeanFlow dit with delta_embed proceeds to the asset stage", async () => {
  await assertRejects(
    () => build({ dit: meanflowDit() }, MEANFLOW_CONFIG),
    Error,
    "irodori: asset 'tokenizer' is missing",
  );
});

test("admitIrodori: a MeanFlow distribution whose dit lacks delta_embed fails naming the input", async () => {
  const error = await assertRejects(() => build({}, MEANFLOW_CONFIG), Error, "delta_embed");
  assertStringIncludes(error.message, "irodori");
});

test("admitIrodori: an RF distribution whose dit takes delta_embed fails (it would never be fed)", async () => {
  const error = await assertRejects(
    () => build({ dit: meanflowDit() }),
    Error,
    "the dit inputs do not match {x_t, t_embed, mask} ∪ the dit_context outputs",
  );
  assertStringIncludes(error.message, "only in dit: [delta_embed]");
});

test("admitIrodori: a MeanFlow delta_embed of another width fails naming timestepEmbedDim", async () => {
  const error = await assertRejects(
    () => build({ dit: meanflowDit(CONFIG.timestepEmbedDim + 2) }, MEANFLOW_CONFIG),
    Error,
    "timestepEmbedDim",
  );
  assertStringIncludes(error.message, "delta_embed");
});

test("admitIrodori: a MeanFlow declaration with a CFG knob fails before the graph check", async () => {
  await assertRejects(
    () => build({ dit: meanflowDit() }, { ...MEANFLOW_CONFIG, cfgScales: CONFIG.cfgScales }),
    Error,
    "declares the RF-only key(s) 'cfgScales'",
  );
});

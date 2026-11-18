// `anima.Pipeline.fromSingleFile` without a GPU, on a tiny Anima-shaped distribution served by the
// fake HF fetch layer (four components stored as bf16, the transformer also as int8; three quants —
// `i8`, `bf16` and `bf16-c16` (bf16 storage, f16 compute) — and a name map) and on tiny safetensors
// files written here. Construction opens no Session, so a fake GPU builds the pipeline; no tensor
// byte of the file is read until a Session is, which is what lets each case below observe what was
// fetched and what was read.
//
// What it pins:
//  ① The success path: which component comes from where and against which container — the text
//    conditioner against the bf16 container's plan, stored as the file's dtype (BF16 as bf16, F16 as
//    f16), the transformer as an int8 quant's from any float dtype and the same way as the text
//    conditioner under a quant that stores it as a float; an f16 container the distribution may
//    still carry is never read for a float storage, and under a quant whose label is `f16` (not
//    the float label) it is the plan as it is — the converted parts are neither fetched nor
//    counted in the progress, the file is read up to its header only, and a role replaced by
//    `components` is not taken from the file (the file's text conditioner would be refused there).
//    Which parts are fetched shows the container; the served container shows the codec.
//  ② Each refusal, before a weight part is fetched (the fetched paths are the manifest, the name
//    map and descriptors only). Each case says which fault it injects.
//  ③ The abort signal at a boundary of the single-file steps.
//  ④ `retainConverted` reaches the converting source of every component from the file.
// The real files are converted on a GPU in `packages/e2e/tests/e2e_anima_from_single_file_test.ts`.

import { describe, it, vi } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
} from "@std/assert";
import type { AssetProgress, HubRepoRef } from "@karume/hub";
import { type IrDeclaration, openContainer } from "@karume/runtime";
import { ConvertError, NAME_MAP_FORMAT, openConvertingSource } from "@karume/convert";
import { SafetensorsError } from "@karume/safetensors";
import { ModelInputError, Pipeline, type SingleFileSource } from "../anima.ts";
import type {
  AssetInput,
  EncodingInput,
  ModelInput,
} from "../../runtime/tests/helpers/container-write.ts";
import { parseIrDeclarationValue } from "./helpers/container-fixture.ts";
import {
  fileRef,
  HUB_URL,
  MANIFEST_PATH,
  REPO,
  SHA,
  type ServedContainer,
  serveContainer,
  serveRepos,
} from "./helpers/container-loading-fixture.ts";
import { MemoryCacheStorage } from "./helpers/memory-cache.ts";
import { type DumpTensor, writeSafetensors } from "./helpers/safetensors-write.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";

// The loader calls the converting source through the package entry; the spy records every source
// it builds and runs the real one, so the case of ④ reads the very sources the pipeline holds.
vi.mock("@karume/convert", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@karume/convert")>();
  return { ...actual, openConvertingSource: vi.fn(actual.openConvertingSource) };
});

const NAME_MAP_PATH = "name-map.json";

// ---- The distribution ------------------------------------------------------------------------

/**
 * One `linear(x, w, b)` over two named initializers (static: `rows` × `k` input, a [2, `k`]
 * weight), the graph named by its role.
 */
const linearGraph = (w: string, b: string, rows = 2, k = 2): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [rows, k] }],
    outputs: ["y"],
    initializers: { [w]: {}, [b]: {} },
    values: {
      [w]: { dtype: "f32", shape: [2, k] },
      [b]: { dtype: "f32", shape: [2] },
      y: { dtype: "f32", shape: [rows, 2] },
    },
    states: {},
    nodes: [{ op: "linear", ins: ["x", w, b], outs: ["y"], attrs: {} }],
  });

const f32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Float32Array.from(values).buffer);

/**
 * The stored bytes of a [2, 2] weight in each codec (the plan is what the conversion reads); the
 * int2 one is [2, 16] (its packing takes 16 elements a word), a codec the conversion does not
 * encode.
 */
const WEIGHT: Readonly<Record<"int8-sym" | "f16" | "bf16" | "int2-off", () => TensorBytes>> = {
  "int8-sym": () => ({
    bytes: new Uint8Array(Int8Array.from([127, 64, -127, 32]).buffer),
    encoding: {
      codec: "int8-sym",
      rowAxis: 0,
      groupSize: 2,
      scale: { bytes: f32Bytes([1 / 127, 2 / 127]), dtype: "f32" },
    },
  }),
  f16: () => ({
    bytes: new Uint8Array(Uint16Array.from([0x3c00, 0x3800, 0xc000, 0x3400]).buffer),
    encoding: { codec: "f16" },
  }),
  bf16: () => ({
    bytes: new Uint8Array(Uint16Array.from([0x3f80, 0x3f00, 0xc000, 0x3e80]).buffer),
    encoding: { codec: "bf16" },
  }),
  "int2-off": () => ({
    bytes: new Uint8Array(8),
    encoding: {
      codec: "int2-off",
      rowAxis: 0,
      groupSize: 16,
      scale: { bytes: f32Bytes([0.5, 0.25]), dtype: "f32" },
    },
  }),
};
type TensorBytes = { readonly bytes: Uint8Array<ArrayBuffer>; readonly encoding: EncodingInput };

/** The rope base table `#build` parses from the transformer's asset (the shape it accepts). */
const ROPE = writeSafetensors(
  new Map(
    ["t", "h", "w"].flatMap((axis) => [
      [`cos_${axis}`, { dtype: "F32", shape: [2, 1], data: new Float32Array(2) }] as const,
      [`sin_${axis}`, { dtype: "F32", shape: [2, 1], data: new Float32Array(2) }] as const,
    ]),
  ),
  {},
);

/** One weighted component: a linear whose weight is stored in `codec` and whose bias is f32. */
const weighted = (
  role: string,
  w: string,
  b: string,
  codec: keyof typeof WEIGHT,
  assets: readonly AssetInput[] = [],
  graph: IrDeclaration = linearGraph(w, b, 2, codec === "int2-off" ? 16 : 2),
): ModelInput => ({
  graphs: { [role]: graph },
  consts: [],
  weights: [
    { graph: role, initializer: w, ...WEIGHT[codec]() },
    { graph: role, initializer: b, bytes: f32Bytes([0.25, -0.5]), encoding: { codec: "f32" } },
  ],
  assets: [...assets],
  provenance: { license: "test" },
});

const TRANSFORMER_W = "model.layers.0.w";
const TRANSFORMER_B = "model.layers.0.b";
const CONDITIONER_W = "model.embed";
const CONDITIONER_B = "model.bias";
const ROPE_ASSET: AssetInput = {
  name: "rope_base",
  role: "rope-base",
  bytes: ROPE,
  dedicatedPart: true,
};

const encode = (value: unknown): Uint8Array<ArrayBuffer> =>
  new TextEncoder().encode(JSON.stringify(value));

/** The minimal tokenizers `#build` parses (the shapes of `anima_pipeline_test.ts`). */
const TOKENIZER = encode({
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
const TOKENIZER_2 = encode({
  vocabText: "<pad>\n</s>\n<unk>\n▁\na\nb",
  scores: [0, 0, 0, -1, -1, -2],
  unkId: 2,
  eosId: 1,
  addedTokens: [],
  space: [[0x20, 0x20]],
  normalizer: { single: [], multi: [], extend: [], breakAfter: [], prepend: [] },
  maxLength: 8,
});

/**
 * The name map: the single-file layout (`net.` stripped, `llm_adapter.*` to the text conditioner,
 * the rest to the transformer through one rename and one exclusion), a transformer-only and a text
 * conditioner-only layout (a file of one component), a layout whose routes leave keys to no
 * component, and one that places keys in the VAE decoder (a component Anima does not take from a
 * file).
 */
const NAME_MAP = {
  format: NAME_MAP_FORMAT,
  layouts: [
    {
      name: "single-file",
      source: "the test",
      stripPrefix: ["net."],
      markers: ["blocks.0.w", "llm_adapter.embed"],
      components: [
        {
          role: "text_conditioner",
          source: "the test",
          route: "llm_adapter.",
          rules: [{ stripPrefix: "llm_adapter." }],
          initializerPrefix: "model.",
        },
        {
          role: "transformer",
          source: "the test",
          route: "",
          rules: [{ replace: "blocks.", with: "layers." }, { exclude: "_extra_state" }],
          initializerPrefix: "model.",
        },
      ],
    },
    {
      name: "transformer-file",
      source: "the test",
      stripPrefix: [],
      markers: ["layers.0.w"],
      components: [
        {
          role: "transformer",
          source: "the test",
          route: "",
          rules: [],
          initializerPrefix: "model.",
        },
      ],
    },
    {
      name: "text-conditioner-file",
      source: "the test",
      stripPrefix: [],
      markers: ["embed"],
      components: [
        {
          role: "text_conditioner",
          source: "the test",
          route: "",
          rules: [],
          initializerPrefix: "model.",
        },
      ],
    },
    {
      name: "routed",
      source: "the test",
      stripPrefix: ["routed."],
      markers: ["layers.0.w"],
      components: [
        {
          role: "transformer",
          source: "the test",
          route: "layers.",
          rules: [],
          initializerPrefix: "model.",
        },
      ],
    },
    {
      name: "with-vae",
      source: "the test",
      stripPrefix: ["vae-file."],
      markers: ["layers.0.w"],
      components: [
        {
          role: "transformer",
          source: "the test",
          route: "layers.",
          rules: [],
          initializerPrefix: "model.",
        },
        {
          role: "vae_decoder",
          source: "the test",
          route: "vae.",
          rules: [],
          initializerPrefix: "",
        },
      ],
    },
  ],
};

/** A declaration-only component (the text encoder and the VAE decoder come from the distribution). */
const declared = (role: string): ModelInput => ({
  graphs: {
    [role]: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["mul"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [1, 4] }],
      outputs: ["y"],
      initializers: {},
      values: { y: { dtype: "f32", shape: [1, 4] } },
      states: {},
      nodes: [{ op: "mul", ins: ["x", "x"], outs: ["y"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [],
  assets: [],
  provenance: { license: "test" },
});

const CONTAINERS = {
  textEncoder: await serveContainer("shared/text_encoder/model", declared("text_encoder")),
  vaeDecoder: await serveContainer("shared/vae_decoder/model", declared("vae_decoder")),
  conditionerF16: await serveContainer(
    "test/text_conditioner/model.f16",
    weighted("text_conditioner", CONDITIONER_W, CONDITIONER_B, "f16"),
  ),
  conditionerBf16: await serveContainer(
    "test/text_conditioner/model.bf16",
    weighted("text_conditioner", CONDITIONER_W, CONDITIONER_B, "bf16"),
  ),
  transformerI8: await serveContainer(
    "test/transformer/model.i8",
    weighted("transformer", TRANSFORMER_W, TRANSFORMER_B, "int8-sym", [ROPE_ASSET]),
  ),
  transformerF16: await serveContainer(
    "test/transformer/model.f16",
    weighted("transformer", TRANSFORMER_W, TRANSFORMER_B, "f16", [ROPE_ASSET]),
  ),
  transformerBf16: await serveContainer(
    "test/transformer/model.bf16",
    weighted("transformer", TRANSFORMER_W, TRANSFORMER_B, "bf16", [ROPE_ASSET]),
  ),
  // Referenced only by the entries of the refusals that need them.
  transformerI2: await serveContainer(
    "test/transformer/model.i2",
    weighted("transformer", TRANSFORMER_W, TRANSFORMER_B, "int2-off", [ROPE_ASSET]),
  ),
} satisfies Record<string, ServedContainer>;

const quant = (
  labels: Readonly<
    Record<"text_encoder" | "text_conditioner" | "transformer" | "vae_decoder", string>
  >,
) => ({ weights: labels, session: {} });

const ASSETS = {
  tokenizer: await fileRef("shared/tokenizer/qwen2-tokenizer.json", TOKENIZER),
  tokenizer_2: await fileRef("shared/tokenizer_2/t5-tokenizer.json", TOKENIZER_2),
};

/** Every role stored as bf16, the transformer as `transformer`. */
const storedAs = (transformer: string) =>
  quant({
    text_encoder: "bf16",
    text_conditioner: "bf16",
    transformer,
    vae_decoder: "bf16",
  });

/** The model entry; `patch` replaces fields of it. */
const modelEntry = (patch: Record<string, unknown> = {}) => ({
  pipeline: "anima/1",
  weights: {
    text_encoder: { bf16: CONTAINERS.textEncoder.entry },
    text_conditioner: { bf16: CONTAINERS.conditionerBf16.entry },
    transformer: { i8: CONTAINERS.transformerI8.entry, bf16: CONTAINERS.transformerBf16.entry },
    vae_decoder: { bf16: CONTAINERS.vaeDecoder.entry },
  },
  assets: ASSETS,
  quants: {
    i8: storedAs("i8"),
    bf16: storedAs("bf16"),
    "bf16-c16": {
      ...storedAs("bf16"),
      session: { linearCompute: "f16", attentionCompute: "f16" },
      gpuFeatures: { shaderF16: true },
    },
  },
  defaultQuant: "i8",
  pipelineConfig: {
    scheduler: { shift: 3, numTrainTimesteps: 1000 },
    defaults: { steps: 2, guidanceScale: 1, resolution: { width: 512, height: 512 } },
  },
  ...patch,
});

const SERVED_FILES = [
  ...Object.values(CONTAINERS).flatMap((container) => container.files),
  [ASSETS.tokenizer.path, TOKENIZER] as const,
  [ASSETS.tokenizer_2.path, TOKENIZER_2] as const,
];

/**
 * The model entry with the f16 containers a distribution may still carry beside its bf16 ones (an
 * `f16` label for the text conditioner and the transformer, and a quant that stores every role as
 * `f16`): a component from a file stored as a float never reads them. Quant `f16`'s labels are not
 * the float label, so under it the transformer is a quantized storage, converted against the f16
 * container as it is.
 */
const withF16Containers = () => {
  const entry = modelEntry();
  return {
    ...entry,
    weights: {
      ...entry.weights,
      text_conditioner: { ...entry.weights.text_conditioner, f16: CONTAINERS.conditionerF16.entry },
      transformer: { ...entry.weights.transformer, f16: CONTAINERS.transformerF16.entry },
    },
    quants: {
      ...entry.quants,
      f16: quant({
        text_encoder: "bf16",
        text_conditioner: "f16",
        transformer: "f16",
        vae_decoder: "bf16",
      }),
    },
  };
};

/** Every served path → its size. */
const SIZES = new Map(SERVED_FILES.map(([path, bytes]) => [path, bytes.byteLength]));

/** The paths of part 0 of every container (descriptors: fetching them is not a weight download). */
const DESCRIPTORS = new Set(Object.values(CONTAINERS).map((container) => container.parts[0].path));

/** The fake HF repository; `withNameMap: false` serves a distribution without a name map. */
const serve = async (
  options: { readonly entry?: Record<string, unknown>; readonly withNameMap?: boolean } = {},
) => {
  const mapBytes = encode(NAME_MAP);
  const withNameMap = options.withNameMap ?? true;
  const mock = serveRepos([
    {
      repo: REPO,
      models: { test: options.entry ?? modelEntry() },
      files: [...SERVED_FILES, ...(withNameMap ? [[NAME_MAP_PATH, mapBytes] as const] : [])],
      members: withNameMap ? { nameMap: await fileRef(NAME_MAP_PATH, mapBytes) } : {},
    },
  ]);
  const ref: HubRepoRef = { repo: REPO, revision: SHA, hubUrl: HUB_URL };
  return {
    mock,
    ref,
    options: {
      fetch: mock.fetch,
      caches: new MemoryCacheStorage(),
      gpu: fakeGpuContext(fakeDevice()),
    },
  };
};

// ---- The files -------------------------------------------------------------------------------

const BF16_WEIGHT = Uint16Array.from([0x3f80, 0x3f00, 0xc000, 0x3e80]);
const BF16_BIAS = Uint16Array.from([0x3e80, 0xbf00]);

/** The same values as f16 bits: [1, 0.5, -2, 0.25] and [0.25, -0.5]. */
const F16_WEIGHT = Uint16Array.from([0x3c00, 0x3800, 0xc000, 0x3400]);
const F16_BIAS = Uint16Array.from([0x3400, 0xb800]);

/** An F16 file's weight bf16 cannot hold: [1 + 2⁻¹⁰, 0.5, -2, 0.25]. */
const F16_FINE_WEIGHT = Uint16Array.from([0x3c01, 0x3800, 0xc000, 0x3400]);

const bf16 = (shape: readonly number[], data: Uint16Array): DumpTensor => ({
  dtype: "BF16",
  shape,
  data,
});
const f16 = (shape: readonly number[], data: Uint16Array): DumpTensor => ({
  dtype: "F16",
  shape,
  data,
});
const f32 = (shape: readonly number[], values: readonly number[]): DumpTensor => ({
  dtype: "F32",
  shape,
  data: Float32Array.from(values),
});

/** The keys of an upstream-shaped single file and their tensors (BF16, as upstream's are). */
const singleFileTensors = (): Map<string, DumpTensor> =>
  new Map([
    ["net.blocks.0.w", bf16([2, 2], BF16_WEIGHT)],
    ["net.blocks.0.b", bf16([2], BF16_BIAS)],
    ["net.llm_adapter.embed", bf16([2, 2], BF16_WEIGHT)],
    ["net.llm_adapter.bias", bf16([2], BF16_BIAS)],
  ]);

/**
 * The same keys with F16 tensors (a fine-tune saved in F16), the weights with a value bf16 cannot
 * hold: stored as bf16 they would be refused (the conversion never rounds).
 */
const f16SingleFileTensors = (): Map<string, DumpTensor> =>
  new Map([
    ["net.blocks.0.w", f16([2, 2], F16_FINE_WEIGHT)],
    ["net.blocks.0.b", f16([2], F16_BIAS)],
    ["net.llm_adapter.embed", f16([2, 2], F16_FINE_WEIGHT)],
    ["net.llm_adapter.bias", f16([2], F16_BIAS)],
  ]);

/** A ranged reader over `bytes` that records every read. */
const recordedFile = (
  bytes: Uint8Array<ArrayBuffer>,
): { readonly file: SingleFileSource; readonly reads: (readonly [number, number])[] } => {
  const reads: (readonly [number, number])[] = [];
  return {
    reads,
    file: {
      size: bytes.byteLength,
      read: (offset, length) => {
        reads.push([offset, length]);
        return Promise.resolve(bytes.slice(offset, offset + length));
      },
    },
  };
};

const fileOf = (tensors: ReadonlyMap<string, DumpTensor>) =>
  recordedFile(writeSafetensors(tensors, {}));

/** The header's end in a safetensors file: the reads up to it read no tensor byte. */
const headerEnd = (bytes: Uint8Array<ArrayBuffer>): number =>
  8 + Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));

/** The served paths fetched other than the manifest, the name map and descriptors. */
const weightDownloads = (paths: readonly string[]): string[] =>
  paths.filter(
    (path) => path !== MANIFEST_PATH && path !== NAME_MAP_PATH && !DESCRIPTORS.has(path),
  );

/** Asserts a refusal and that it came before any weight part (or asset) was fetched. */
const refused = async (
  run: () => Promise<unknown>,
  paths: readonly string[],
  text: string,
  type: new (message: string) => Error = Error,
): Promise<string> => {
  const error = await assertRejects(run, type);
  assertStringIncludes(error.message, text);
  assertEquals(weightDownloads(paths), [], "a weight part was fetched before the refusal");
  return error.message;
};

const pathsOf = (container: ServedContainer): string[] => container.files.map(([path]) => path);

/** The paths of `container` that were fetched, in its order. */
const fetchedOf = (paths: readonly string[], container: ServedContainer): string[] => {
  const fetched = new Set(paths);
  return pathsOf(container).filter((path) => fetched.has(path));
};

/**
 * What a transformer container converted against shows: its part 0 and its asset part (the rope
 * table) fetched, never its converted weight part.
 */
const convertedTransformer = (container: ServedContainer): string[] => [
  container.parts[0].path,
  container.parts[3].path,
];

/**
 * Per role, what the converting sources of the last build serve (the spy's results since its last
 * `mockClear`): the codec set of the served container and the stored payload of one initializer.
 */
const servedByRole = async () => {
  const served = new Map<
    string,
    {
      readonly codecs: readonly string[];
      readonly stored: (initializer: string) => Promise<Uint8Array<ArrayBuffer>>;
    }
  >();
  for (const result of vi.mocked(openConvertingSource).mock.results) {
    assert(result.type === "return");
    const { source } = await result.value;
    const view = await openContainer({ kind: "source", source });
    const [role] = Object.keys(view.graphs);
    served.set(role, {
      codecs: view.model?.codecs ?? [],
      stored: async (initializer) => {
        const block = view.graphs[role].supplies.get(initializer)?.blocks[0];
        assert(block !== undefined, `no block of '${initializer}'`);
        const bytes = await source.read(block.part, block.offset, block.length);
        return bytes.slice(0, block.payloadBytes);
      },
    });
  }
  return served;
};

const u16Bytes = (bits: Uint16Array<ArrayBuffer>): Uint8Array<ArrayBuffer> =>
  new Uint8Array(bits.buffer.slice(0));

// ---- ① The success path ----------------------------------------------------------------------

describe("fromSingleFile builds an Anima pipeline from a single file and a distribution", () => {
  it("takes the transformer and the text conditioner from the file: the converted parts are not fetched, and the progress counts only what is fetched", async () => {
    const { mock, ref, options } = await serve();
    const bytes = writeSafetensors(singleFileTensors(), {});
    const { file, reads } = recordedFile(bytes);
    const progress: AssetProgress[] = [];
    await using pipeline = await Pipeline.fromSingleFile(file, ref, {
      ...options,
      onProgress: (event) => void progress.push(event),
    });
    assert(pipeline instanceof Pipeline);

    const fetched = new Set(mock.paths);
    const { transformerI8, conditionerBf16, textEncoder, vaeDecoder } = CONTAINERS;
    // The transformer: part 0 and the kept parts (the asset part with the rope table), never the
    // part of the converted weights (part 2: the weight, its scale and the bias).
    assertEquals(
      pathsOf(transformerI8).filter((path) => fetched.has(path)),
      [transformerI8.parts[0].path, transformerI8.parts[3].path],
    );
    // The text conditioner from the file is converted against the bf16 container: its descriptor
    // only.
    assertEquals(
      pathsOf(conditionerBf16).filter((path) => fetched.has(path)),
      [conditionerBf16.parts[0].path],
    );
    // The text encoder and the VAE decoder come from the distribution, whole.
    for (const container of [textEncoder, vaeDecoder]) {
      assertEquals(
        pathsOf(container).filter((path) => fetched.has(path)),
        pathsOf(container),
      );
    }
    // No byte of a tensor was read: the format's first bytes and the header only.
    const end = headerEnd(bytes);
    assert(reads.length > 0);
    for (const [offset, length] of reads)
      assert(offset + length <= end, `read ${offset}+${length}`);
    // One progress stream whose total is what was fetched (the name map is not part of it).
    const total = [...fetched]
      .filter((path) => path !== MANIFEST_PATH && path !== NAME_MAP_PATH)
      .reduce((sum, path) => sum + (SIZES.get(path) ?? 0), 0);
    const last = progress.at(-1);
    assert(last !== undefined, "no progress was reported");
    assertEquals([last.loaded, last.total], [total, total]);
  });

  it("a role replaced by components is not taken from the file: the file's text conditioner, which could not be stored, is not refused, and the quant's container is fetched", async () => {
    const { mock, ref, options } = await serve();
    // Fault: the file's text conditioner is F32 — refused were it taken from the file (no container
    // stores F32 as it is; the case "an F32 text conditioner" below).
    const tensors = singleFileTensors();
    tensors.set("net.llm_adapter.embed", f32([2, 2], [1, 0.5, -2, 0.25]));
    tensors.set("net.llm_adapter.bias", f32([2], [0.25, -0.5]));
    const { file } = fileOf(tensors);
    await using pipeline = await Pipeline.fromSingleFile(file, ref, {
      ...options,
      components: { text_conditioner: { source: ref } },
    });
    assert(pipeline instanceof Pipeline);
    const fetched = new Set(mock.paths);
    assertEquals(
      pathsOf(CONTAINERS.conditionerBf16).filter((path) => fetched.has(path)),
      pathsOf(CONTAINERS.conditionerBf16),
    );
  });

  it("an F16 file under a quant that stores the transformer as a float: both roles are stored as f16, the file's own bits, against the plans of the bf16 containers — an f16 container the distribution carries is never fetched", async () => {
    // Fault the second entry injects: f16 containers of both roles (and a quant naming them) beside
    // the bf16 ones; reading either would show in the fetched paths.
    for (const entry of [modelEntry(), withF16Containers()]) {
      const { mock, ref, options } = await serve({ entry });
      vi.mocked(openConvertingSource).mockClear();
      await using pipeline = await Pipeline.fromSingleFile(
        fileOf(f16SingleFileTensors()).file,
        ref,
        {
          ...options,
          quant: "bf16",
        },
      );
      assert(pipeline instanceof Pipeline);
      const { transformerF16, transformerBf16, conditionerF16, conditionerBf16 } = CONTAINERS;
      assertEquals(fetchedOf(mock.paths, transformerBf16), convertedTransformer(transformerBf16));
      assertEquals(fetchedOf(mock.paths, conditionerBf16), [conditionerBf16.parts[0].path]);
      assertEquals(
        [...fetchedOf(mock.paths, transformerF16), ...fetchedOf(mock.paths, conditionerF16)],
        [],
      );
      // The served containers store the weights as f16 — the file's bits, which bf16 cannot hold —
      // and the f32 bias as the plan does.
      const served = await servedByRole();
      for (const [role, weight, bias] of [
        ["transformer", TRANSFORMER_W, TRANSFORMER_B],
        ["text_conditioner", CONDITIONER_W, CONDITIONER_B],
      ] as const) {
        const view = served.get(role);
        assert(view !== undefined, `no converting source of '${role}'`);
        assertEquals(view.codecs, ["f16", "f32"]);
        assertEquals(await view.stored(weight), u16Bytes(F16_FINE_WEIGHT));
        assertEquals(await view.stored(bias), f32Bytes([0.25, -0.5]));
      }
    }
  });

  it("an F16 file under f16 linear compute (bf16-c16): stored as f16 against the bf16 plans, the f16 storage under f16 compute that runs", async () => {
    const { mock, ref, options } = await serve();
    vi.mocked(openConvertingSource).mockClear();
    const { file } = fileOf(f16SingleFileTensors());
    await using pipeline = await Pipeline.fromSingleFile(file, ref, {
      ...options,
      gpu: fakeGpuContext(fakeDevice({ features: ["shader-f16"] })),
      quant: "bf16-c16",
    });
    assert(pipeline instanceof Pipeline);
    const { transformerBf16, conditionerBf16 } = CONTAINERS;
    assertEquals(fetchedOf(mock.paths, transformerBf16), convertedTransformer(transformerBf16));
    assertEquals(fetchedOf(mock.paths, conditionerBf16), [conditionerBf16.parts[0].path]);
    assertEquals((await servedByRole()).get("transformer")?.codecs, ["f16", "f32"]);
  });

  it("a BF16 file under a quant that stores the transformer as a float: stored as bf16, the plan of the bf16 containers as it is", async () => {
    const { mock, ref, options } = await serve({ entry: withF16Containers() });
    vi.mocked(openConvertingSource).mockClear();
    const { file } = fileOf(singleFileTensors());
    await using pipeline = await Pipeline.fromSingleFile(file, ref, { ...options, quant: "bf16" });
    assert(pipeline instanceof Pipeline);
    const { transformerF16, transformerBf16 } = CONTAINERS;
    assertEquals(fetchedOf(mock.paths, transformerBf16), convertedTransformer(transformerBf16));
    assertEquals(fetchedOf(mock.paths, transformerF16), []);
    const view = (await servedByRole()).get("transformer");
    assert(view !== undefined);
    assertEquals(view.codecs, ["bf16", "f32"]);
    assertEquals(await view.stored(TRANSFORMER_W), u16Bytes(BF16_WEIGHT));
  });

  it("an F16 file under a quant whose transformer label is f16, not the float label: the transformer is converted against that quant's f16 container as it is, the text conditioner against the bf16 plan as f16", async () => {
    // A label other than the family's float label is a quantized storage, whatever its container
    // stores: its own plan, no 16-bit float codec put in its place.
    const { mock, ref, options } = await serve({ entry: withF16Containers() });
    vi.mocked(openConvertingSource).mockClear();
    const { file } = fileOf(f16SingleFileTensors());
    await using pipeline = await Pipeline.fromSingleFile(file, ref, { ...options, quant: "f16" });
    assert(pipeline instanceof Pipeline);
    const { transformerF16, transformerBf16, conditionerF16, conditionerBf16 } = CONTAINERS;
    assertEquals(fetchedOf(mock.paths, transformerF16), convertedTransformer(transformerF16));
    assertEquals(fetchedOf(mock.paths, transformerBf16), []);
    assertEquals(fetchedOf(mock.paths, conditionerBf16), [conditionerBf16.parts[0].path]);
    assertEquals(fetchedOf(mock.paths, conditionerF16), []);
    const served = await servedByRole();
    for (const [role, weight] of [
      ["transformer", TRANSFORMER_W],
      ["text_conditioner", CONDITIONER_W],
    ] as const) {
      const view = served.get(role);
      assert(view !== undefined, `no converting source of '${role}'`);
      assertEquals(view.codecs, ["f16", "f32"]);
      assertEquals(await view.stored(weight), u16Bytes(F16_FINE_WEIGHT));
    }
  });

  it("under an int8 quant the transformer is the quant's container, converted from any float dtype, a mix of them included", async () => {
    // The f16 and bf16 transformers the distribution carries beside it are not fetched.
    const { mock, ref, options } = await serve({ entry: withF16Containers() });
    const tensors = singleFileTensors();
    tensors.set("net.blocks.0.w", f16([2, 2], F16_WEIGHT));
    tensors.set("net.blocks.0.b", f32([2], [0.25, -0.5]));
    const { file } = fileOf(tensors);
    await using pipeline = await Pipeline.fromSingleFile(file, ref, options);
    assert(pipeline instanceof Pipeline);
    const { transformerI8, transformerF16, transformerBf16 } = CONTAINERS;
    assertEquals(fetchedOf(mock.paths, transformerI8), convertedTransformer(transformerI8));
    assertEquals(
      [...fetchedOf(mock.paths, transformerF16), ...fetchedOf(mock.paths, transformerBf16)],
      [],
    );
  });

  it("takes a file of the transformer alone, with the text conditioner of the quant", async () => {
    const { mock, ref, options } = await serve();
    const { file } = fileOf(
      new Map([
        ["layers.0.w", f32([2, 2], [1, 0.5, -2, 0.25])],
        ["layers.0.b", f32([2], [0.25, -0.5])],
      ]),
    );
    await using pipeline = await Pipeline.fromSingleFile(file, ref, options);
    assert(pipeline instanceof Pipeline);
    const fetched = new Set(mock.paths);
    assertEquals(
      pathsOf(CONTAINERS.conditionerBf16).filter((path) => fetched.has(path)),
      pathsOf(CONTAINERS.conditionerBf16),
    );
  });
});

// ---- ② The refusals --------------------------------------------------------------------------

describe("fromSingleFile refuses before a weight part is fetched", () => {
  it("a file that is not safetensors, naming what was seen", async () => {
    // Faults: a GGUF magic, a zip archive (torch.save), a pickle, and bytes of no known format.
    for (const [bytes, seen] of [
      [new TextEncoder().encode("GGUF and the rest"), "the file is a GGUF file"],
      [Uint8Array.from([0x50, 0x4b, 3, 4, 0, 0, 0, 0, 0, 0]), "the file is a zip archive"],
      [Uint8Array.from([0x80, 2, 0x7d, 0x71, 0, 0, 0, 0, 0, 0]), "the file is a Python pickle"],
      [Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), "the file's first bytes are 01 02 03"],
      [Uint8Array.from([1, 2, 3]), "the file is 3 bytes long"],
    ] as const) {
      const { mock, ref, options } = await serve();
      await refused(
        () => Pipeline.fromSingleFile(new Blob([bytes]), ref, options),
        mock.paths,
        `only safetensors files are read, and ${seen}`,
      );
    }
  });

  it("a safetensors file the reader refuses, with the reader's error", async () => {
    // Fault: a header length past the end of the file, behind a valid first byte.
    const { mock, ref, options } = await serve();
    const bytes = new Uint8Array(new ArrayBuffer(16));
    new DataView(bytes.buffer).setBigUint64(0, 1000n, true);
    bytes[8] = "{".charCodeAt(0);
    await refused(
      () => Pipeline.fromSingleFile(new Blob([bytes]), ref, options),
      mock.paths,
      "exceeds the file length",
      SafetensorsError,
    );
  });

  it("a distribution that carries no name map", async () => {
    // Fault: the manifest has no `nameMap`.
    const { mock, ref, options } = await serve({ withNameMap: false });
    const { file, reads } = fileOf(singleFileTensors());
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the distribution carries no name map",
    );
    assertEquals(reads, [], "the file was read although the map is missing");
  });

  it("a file no layout of the map recognises, with placeTensors' error", async () => {
    // Fault: keys of another model.
    const { mock, ref, options } = await serve();
    const { file } = fileOf(new Map([["encoder.w", f32([1], [1])]]));
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "no layout of the name map recognises the file",
      ConvertError,
    );
  });

  it("a key no component takes, listing it", async () => {
    // Fault: a key outside every route of the `routed` layout.
    const { mock, ref, options } = await serve();
    const { file } = fileOf(
      new Map([
        ["routed.layers.0.w", f32([2, 2], [1, 0.5, -2, 0.25])],
        ["routed.layers.0.b", f32([2], [0.25, -0.5])],
        ["routed.extra.scale", f32([1], [1])],
      ]),
    );
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "1 key(s) of the file go to no component of the name map's layout 'routed' (nothing of the file is dropped): 'routed.extra.scale'",
    );
  });

  it("tensors of a component Anima does not take from a file", async () => {
    // Fault: the `with-vae` layout places a key in the VAE decoder.
    const { mock, ref, options } = await serve();
    const { file } = fileOf(
      new Map([
        ["vae-file.layers.0.w", f32([2, 2], [1, 0.5, -2, 0.25])],
        ["vae-file.layers.0.b", f32([2], [0.25, -0.5])],
        ["vae-file.vae.w", f32([1], [1])],
      ]),
    );
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the file holds tensors of 'vae_decoder', which this pipeline does not take from a file",
    );
  });

  it("a tensor placed in the transformer that its plan does not use, listing it", async () => {
    // Fault: one more block than the plan has.
    const { mock, ref, options } = await serve();
    const tensors = singleFileTensors();
    tensors.set("net.blocks.9.w", bf16([2, 2], BF16_WEIGHT));
    const { file } = fileOf(tensors);
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the file holds 1 tensor(s) placed in the transformer that its plan does not use (nothing of the file is dropped): 'net.blocks.9.w' (as 'model.layers.9.w')",
    );
  });

  it("an initializer of the plan the file has no tensor for, with the converting source's error", async () => {
    // Fault: the bias of the transformer is missing.
    const { mock, ref, options } = await serve();
    const tensors = singleFileTensors();
    tensors.delete("net.blocks.0.b");
    const { file } = fileOf(tensors);
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the tensor list has no tensor 'model.layers.0.b'",
      ConvertError,
    );
  });

  it("a file that holds no transformer", async () => {
    // Fault: a file of the text conditioner alone.
    const { mock, ref, options } = await serve();
    const { file } = fileOf(
      new Map([
        ["embed", bf16([2, 2], BF16_WEIGHT)],
        ["bias", bf16([2], BF16_BIAS)],
      ]),
    );
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the file holds no transformer (the name map's layout 'text-conditioner-file' places none of its 2 keys there)",
    );
  });

  it("a call that leaves no component to come from the file", async () => {
    // Fault: both roles the file holds are replaced.
    const { mock, ref, options } = await serve();
    const { file } = fileOf(singleFileTensors());
    await refused(
      () =>
        Pipeline.fromSingleFile(file, ref, {
          ...options,
          components: { transformer: { source: ref }, text_conditioner: { source: ref } },
        }),
      mock.paths,
      "no component is left to come from the file (components replaces 'transformer', 'text_conditioner'); without one, load the distribution with fromPretrained",
    );
  });

  it("a quant whose transformer is stored in a codec that is not converted: one message with the quant, the codec and the quants that work", async () => {
    // Fault: quant `i2`, whose transformer is int2. The quants that work are those whose container
    // for this file encodes: `bf16-c16` is left out, as it refuses a BF16 transformer.
    const entry = modelEntry();
    const { mock, ref, options } = await serve({
      entry: {
        ...entry,
        weights: {
          ...entry.weights,
          transformer: { ...entry.weights.transformer, i2: CONTAINERS.transformerI2.entry },
        },
        quants: { ...entry.quants, i2: storedAs("i2") },
      },
    });
    const { file } = fileOf(singleFileTensors());
    const message = await refused(
      () => Pipeline.fromSingleFile(file, ref, { ...options, quant: "i2" }),
      mock.paths,
      "the 'i2' transformer of model 'test', which the file's transformer is converted against under quant 'i2', stores codec 'int2-off', which the conversion of a file does not encode (it encodes int8-sym / f16 / bf16 / f32); the quants that work: i8 / bf16",
    );
    assertEquals(message.split("\n").length, 1, "one line per initializer");
  });

  it("a tensor whose dtype is not the one a quantized container's float weight takes, naming the role, the tensor, its dtype and what to do", async () => {
    // Fault: the model's `i8` transformer label holds a bf16 plan. A quant that stores the
    // transformer quantized is converted against as it is, so an F16 file lands on a bf16 weight.
    const entry = modelEntry();
    const { mock, ref, options } = await serve({
      entry: {
        ...entry,
        weights: {
          ...entry.weights,
          transformer: { ...entry.weights.transformer, i8: CONTAINERS.transformerBf16.entry },
        },
      },
    });
    const { file } = fileOf(f16SingleFileTensors());
    await refused(
      () => Pipeline.fromSingleFile(file, ref, { ...options, quant: "i8" }),
      mock.paths,
      "the transformer stores 'model.layers.0.w' as bf16, which takes BF16 values as they are and never rounds, but the file's 'net.blocks.0.w' is F16; choose a quant that stores the transformer quantized, which converts from any float dtype: i8",
    );
  });

  it("a BF16 file under a quant whose transformer label is f16, not the float label: refused by the dtype check of that quant's f16 container", async () => {
    // Fault: quant `f16` of a distribution that carries f16 containers. Its transformer label is a
    // quantized storage, so its f16 plan is the plan as it is, and an f16 weight takes F16 only.
    const { mock, ref, options } = await serve({ entry: withF16Containers() });
    await refused(
      () =>
        Pipeline.fromSingleFile(fileOf(singleFileTensors()).file, ref, {
          ...options,
          quant: "f16",
        }),
      mock.paths,
      "the transformer stores 'model.layers.0.w' as f16, which takes F16 values as they are and never rounds, but the file's 'net.blocks.0.w' is BF16; ",
    );
  });

  it("a text conditioner whose tensors mix dtypes, naming both", async () => {
    // Fault: one tensor of the file's text conditioner is F32, the other BF16.
    const { mock, ref, options } = await serve();
    const tensors = singleFileTensors();
    tensors.set("net.llm_adapter.bias", f32([2], [0.25, -0.5]));
    const { file } = fileOf(tensors);
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the file's text_conditioner mixes dtypes: BF16 (1 tensor(s), e.g. 'net.llm_adapter.embed'), F32 (1 tensor(s), e.g. 'net.llm_adapter.bias'); it is stored as the float type of its tensors, so they must all have one; pass components: { text_conditioner: … } to take the distribution's text_conditioner",
    );
  });

  it("an F32 text conditioner, which no container stores as it is", async () => {
    // Fault: the file's text conditioner is F32 throughout.
    const { mock, ref, options } = await serve();
    const tensors = singleFileTensors();
    tensors.set("net.llm_adapter.embed", f32([2, 2], [1, 0.5, -2, 0.25]));
    tensors.set("net.llm_adapter.bias", f32([2], [0.25, -0.5]));
    const { file } = fileOf(tensors);
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the file's text_conditioner is F32, and no container stores F32 as it is (the float storages: BF16 → 'bf16', F16 → 'f16'); pass components: { text_conditioner: … } to take the distribution's text_conditioner",
    );
  });

  it("under a quant that stores the transformer as a float: a transformer whose tensors mix dtypes, and an F32 one, each with the int8 quants to choose", async () => {
    // Faults: the transformer's weight F16 beside its BF16 bias; the transformer F32 throughout
    // (quant `bf16` both times).
    const { mock, ref, options } = await serve();
    const mixed = singleFileTensors();
    mixed.set("net.blocks.0.w", f16([2, 2], F16_WEIGHT));
    await refused(
      () => Pipeline.fromSingleFile(fileOf(mixed).file, ref, { ...options, quant: "bf16" }),
      mock.paths,
      "the file's transformer mixes dtypes: F16 (1 tensor(s), e.g. 'net.blocks.0.w'), BF16 (1 tensor(s), e.g. 'net.blocks.0.b'); it is stored as the float type of its tensors (quant 'bf16' stores the transformer as a float), so they must all have one; choose a quant that stores the transformer quantized, which converts from any float dtype: i8",
    );
    const wide = singleFileTensors();
    wide.set("net.blocks.0.w", f32([2, 2], [1, 0.5, -2, 0.25]));
    wide.set("net.blocks.0.b", f32([2], [0.25, -0.5]));
    await refused(
      () => Pipeline.fromSingleFile(fileOf(wide).file, ref, { ...options, quant: "bf16" }),
      mock.paths,
      "the file's transformer is F32 (quant 'bf16' stores the transformer as a float), and no container stores F32 as it is (the float storages: BF16 → 'bf16', F16 → 'f16'); choose a quant that stores the transformer quantized, which converts from any float dtype: i8",
    );
  });

  it("a BF16 transformer under f16 linear compute — the quant's or an explicit one — in one message naming the quant and the dtype", async () => {
    // Faults: quant `bf16-c16` (linearCompute 'f16') with a BF16 file; quant `bf16` with an
    // explicit `linearCompute: "f16"`.
    const { mock, ref, options } = await serve();
    const gpu = fakeGpuContext(fakeDevice({ features: ["shader-f16"] }));
    const declared = await refused(
      () =>
        Pipeline.fromSingleFile(fileOf(singleFileTensors()).file, ref, {
          ...options,
          gpu,
          quant: "bf16-c16",
        }),
      mock.paths,
      "quant 'bf16-c16' computes the transformer's linear layers in f16 (linearCompute 'f16'), and the file's transformer is BF16, stored as bf16: bf16 storage under f16 compute is not run from a file",
    );
    assertEquals(declared.split("\n").length, 1);
    await refused(
      () =>
        Pipeline.fromSingleFile(fileOf(singleFileTensors()).file, ref, {
          ...options,
          gpu,
          quant: "bf16",
          linearCompute: "f16",
        }),
      mock.paths,
      "quant 'bf16' computes the transformer's linear layers in f16 (linearCompute 'f16', set explicitly), and the file's transformer is BF16",
    );
  });

  it("a tensor of a dtype the conversion does not take", async () => {
    // Fault: an I8 tensor where a weight is placed.
    const { mock, ref, options } = await serve();
    const tensors = singleFileTensors();
    tensors.set("net.blocks.0.w", {
      dtype: "I8",
      shape: [2, 2],
      data: Int8Array.from([1, 2, 3, 4]),
    });
    const { file } = fileOf(tensors);
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "'net.blocks.0.w' is I8, not a float tensor (F32 / F16 / BF16)",
    );
  });

  it("a model with no bf16 text conditioner, whose plan the file's is converted against — an F16 file too, although the model has an f16 one", async () => {
    // Fault: the model's text conditioner exists as f16 only (every quant names it).
    const entry = modelEntry();
    const conditionerF16 = (stored: ReturnType<typeof storedAs>) => ({
      ...stored,
      weights: { ...stored.weights, text_conditioner: "f16" },
    });
    const { mock, ref, options } = await serve({
      entry: {
        ...entry,
        weights: { ...entry.weights, text_conditioner: { f16: CONTAINERS.conditionerF16.entry } },
        quants: { i8: conditionerF16(storedAs("i8")), bf16: conditionerF16(storedAs("bf16")) },
      },
    });
    for (const [tensors, dtype] of [
      [singleFileTensors(), "BF16"],
      [f16SingleFileTensors(), "F16"],
    ] as const) {
      await refused(
        () => Pipeline.fromSingleFile(fileOf(tensors).file, ref, options),
        mock.paths,
        `model 'test' has no 'bf16' text_conditioner, whose plan the file's text_conditioner is converted against (its tensors are ${dtype})`,
      );
    }
  });

  it("the family's admission still runs on the converted components before a weight part", async () => {
    // Fault: the manifest's pipeline is another family's.
    const { mock, ref, options } = await serve({ entry: modelEntry({ pipeline: "sbv2/1" }) });
    const { file } = fileOf(singleFileTensors());
    await refused(
      () => Pipeline.fromSingleFile(file, ref, options),
      mock.paths,
      "the manifest's pipeline is 'sbv2/1'",
    );
  });
});

// ---- ③ The abort signal ----------------------------------------------------------------------

describe("fromSingleFile and the abort signal", () => {
  it("an already-aborted signal rejects with its reason before anything is fetched or read", async () => {
    const { mock, ref, options } = await serve();
    const { file, reads } = fileOf(singleFileTensors());
    const controller = new AbortController();
    const reason = new Error("cancel button");
    controller.abort(reason);
    const error = await assertRejects(() =>
      Pipeline.fromSingleFile(file, ref, { ...options, signal: controller.signal }),
    );
    assertStrictEquals(error, reason);
    assertEquals([mock.paths, reads], [[], []]);
  });

  it("an abort while the name map is fetched stops before the file is read", async () => {
    const { mock, ref, options } = await serve();
    const { file, reads } = fileOf(singleFileTensors());
    const controller = new AbortController();
    const reason = new Error("cancel button (while fetching)");
    const fetch: typeof globalThis.fetch = (input, init) => {
      const href =
        typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (href.endsWith(`/${NAME_MAP_PATH}`)) controller.abort(reason);
      return options.fetch(input, init);
    };
    const error = await assertRejects(() =>
      Pipeline.fromSingleFile(file, ref, { ...options, fetch, signal: controller.signal }),
    );
    assertStrictEquals(error, reason);
    assertEquals(reads, []);
    assertEquals(weightDownloads(mock.paths), []);
  });
});

// ---- ④ Keeping the converted weights ---------------------------------------------------------

describe("fromSingleFile with retainConverted", () => {
  /** The initializers of each from-file role in the order the case reads them: the weight again last. */
  const READ_ORDER: Readonly<Record<string, readonly string[]>> = {
    transformer: [TRANSFORMER_W, TRANSFORMER_B, TRANSFORMER_W],
    text_conditioner: [CONDITIONER_W, CONDITIONER_B, CONDITIONER_W],
  };

  /**
   * Builds the pipeline from the upstream-shaped file (both roles from it), then reads, through
   * every converting source the loader built, every block of the role's weight, of its bias, then
   * of the weight again (no Session: the reads a Session build makes); returns per role how many
   * reads of the file read tensor bytes (one ranged read per tensor, past the header).
   */
  const tensorReadsPerRole = async (
    retainConverted: boolean | undefined,
  ): Promise<Record<string, number>> => {
    const { ref, options } = await serve();
    const bytes = writeSafetensors(singleFileTensors(), {});
    const { file, reads } = recordedFile(bytes);
    const spy = vi.mocked(openConvertingSource);
    spy.mockClear();
    await using pipeline = await Pipeline.fromSingleFile(file, ref, {
      ...options,
      ...(retainConverted === undefined ? {} : { retainConverted }),
    });
    assert(pipeline instanceof Pipeline);
    const end = headerEnd(bytes);
    const tensorReads = (): number => reads.filter(([offset]) => offset >= end).length;
    const counts: Record<string, number> = {};
    for (const result of spy.mock.results) {
      assert(result.type === "return");
      const { source } = await result.value;
      const view = await openContainer({ kind: "source", source });
      const [role] = Object.keys(view.graphs);
      const order = READ_ORDER[role];
      assert(order !== undefined, `a converting source of '${role}'`);
      const before = tensorReads();
      for (const initializer of order) {
        const supply = view.graphs[role].supplies.get(initializer);
        assert(supply !== undefined, `no supply '${initializer}'`);
        const scale = supply.scale === undefined ? [] : (supply.scale.pieces ?? [supply.scale]);
        for (const block of [...supply.blocks, ...scale]) {
          await source.read(block.part, block.offset, block.length);
        }
      }
      counts[role] = tensorReads() - before;
    }
    return counts;
  };

  it("on: every component from the file reads each tensor once; the weight read again comes from memory", async () => {
    assertEquals(await tensorReadsPerRole(true), { transformer: 2, text_conditioner: 2 });
  });

  it("off (the default, and false): the weight read again is read from the file again", async () => {
    const again = { transformer: 3, text_conditioner: 3 };
    assertEquals(await tensorReadsPerRole(undefined), again);
    assertEquals(await tensorReadsPerRole(false), again);
  });

  it("a value that is not a boolean is refused before anything is fetched or read", async () => {
    const { mock, ref, options } = await serve();
    const { file, reads } = fileOf(singleFileTensors());
    await assertRejects(
      () =>
        Pipeline.fromSingleFile(file, ref, {
          ...options,
          retainConverted: "yes" as unknown as boolean,
        }),
      ModelInputError,
      'retainConverted "yes" is not a boolean',
    );
    assertEquals([mock.paths, reads], [[], []]);
  });
});

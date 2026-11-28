// The grid-fill rule of the GEMM (`gemmGrid` — src/kernels/geometry-profile.ts) on a real GPU.
//
// The level `exact` admits a tile chosen by the rule only because, and only where, its output words
// equal the reference GEMM's: the geometry decides which thread computes which output, never the
// order of a K reduction (src/kernels/gemm-geometry.ts). That is a fact of the compiled kernels, not
// a guarantee of WGSL, so it is checked here on this device's backend:
//
// 1. Through a Session, for every weight storage the GEMM reads (f32, f16, bf16, i8, i4 at group 32,
//    i2 under f32 compute, f16 under f16 compute) and for matmul and bmm, on shapes that reach every
//    step of the ladder — the full grid that keeps the largest tile and the under-filled grids that
//    walk down to the last —, the output of a Session whose injected table carries the rule equals,
//    word for word, the output of the default profile's Session (the M-bucket tile). Equal words
//    alone would also pass if the rule were ignored, so the key that ran must name the rule's tile,
//    and on these shapes it differs from the bucket's.
// 2. The split-K partial (`LINEAR_SPLIT_K`) at one slice equals the reference GEMM word for word on
//    the tile the rule gives each shape (its one-slice identity, on the tiles it now takes).
//
// Fault injection (recorded 2026-11-28 on an RTX 3080 Ti, Dawn / Vulkan): with the linear GEMM's
// dispatch x axis taken from the M-bucket tile instead of the rule's (the key and WGSL kept), part 1
// turns red at the first storage, f32, on the 65 × 40 shape: the rule's 16 × 16 tile needs three
// column tiles where the bucket's 64 × 32 gives two, so columns 32 to 39 stay unwritten. (On the
// four shapes before it the bucket's tile is at most as wide, and the extra workgroups write nothing.)

import { platformObject } from "@karume/test-support/webgpu";
import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu, GpuContext } from "../src/gpu/device.ts";
import { gemmGeometryTileKeyPart, type GemmGeometry } from "../src/kernels/gemm-geometry.ts";
import {
  DEFAULT_GEOMETRY_PROFILE,
  type GemmGridRule,
  gemmGridStep,
  gemmRowsGeometry,
  type GemmShape,
  gemmShapeGeometry,
  type GeometryProfile,
} from "../src/kernels/geometry-profile.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import type { EncodingInput, TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";
import { splitKRunner } from "./helpers/linear-splitk-dispatch.ts";
import {
  type EncodedLinearWeight,
  encodeLinearWeight,
  type LinearWeightStorage,
  storageLabel,
} from "./helpers/linear-storage-weights.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openGraphModel,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";

/** The recon's ladder (largest first): 128×128, 64×64 / 256, 64×32 / 128, 32×32 / 128, 16×16 / 64. */
const LADDER: readonly GemmGeometry[] = [
  { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  { regM: 4, regN: 4, wgX: 8, wgY: 16 },
  { regM: 2, regN: 4, wgX: 8, wgY: 16 },
  { regM: 1, regN: 4, wgX: 4, wgY: 16 },
];

/**
 * A rule with small counts, so that small shapes (fast to run, every edge of a tile met) reach
 * every step of the ladder.
 */
const RULE: GemmGridRule = {
  ladder: LADDER,
  keepLargest: { minWorkgroups: 8, minK: 64 },
  minWorkgroups: 8,
  maxPadding: 1.5,
};

const GRID_PROFILE: GeometryProfile = {
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "grid-test",
  gemmGrid: RULE,
};

/**
 * One linear shape per step of the ladder, M and N off the tiles where the padding bound allows it
 * (M ≥ 65, so the integer storages take the GEMM rather than the GEMV family), and one scalar shape
 * (N not a multiple of 4) that the padding sends to the last tile.
 */
const LINEAR_SHAPES: readonly GemmShape[] = [
  { m: 500, n: 384, k: 64, batch: 1 },
  { m: 100, n: 520, k: 96, batch: 1 },
  { m: 136, n: 96, k: 64, batch: 1 },
  { m: 70, n: 272, k: 96, batch: 1 },
  { m: 65, n: 40, k: 64, batch: 1 },
  { m: 136, n: 98, k: 64, batch: 1 },
];

/** A storage of the weight and the GEMM's compute. */
type Storage = { readonly weight: LinearWeightStorage; readonly compute: "f32" | "f16" };

const STORAGES: readonly Storage[] = [
  { weight: { storage: "f32" }, compute: "f32" },
  { weight: { storage: "f16" }, compute: "f32" },
  { weight: { storage: "bf16" }, compute: "f32" },
  { weight: { storage: "i8" }, compute: "f32" },
  { weight: { storage: "i4", groupSize: 32 }, compute: "f32" },
  { weight: { storage: "i2" }, compute: "f32" },
  { weight: { storage: "f16" }, compute: "f16" },
];

const SIGNED = (index: number): number => ((index % 11) - 5) * 0.375 + 0.125;
const WEIGHT = (index: number): number => ((index % 7) - 3) * 0.5 - 0.0625;

/** Activations of mixed signs and magnitudes, so the sums cancel and round. */
const activations = (m: number, k: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75);

/** The container encoding of an encoded weight (its exact bytes, its scale). */
const weightTensor = (weight: EncodedLinearWeight, n: number, k: number): TensorInput => {
  const bitsPer = { f32: 32, f16: 16, bf16: 16, i8: 8, i4: 4, i2: 2 }[weight.storage];
  const bytes = new Uint8Array(weight.words.buffer, 0, (n * k * bitsPer) / 8).slice();
  const scale =
    weight.scale === undefined
      ? undefined
      : { bytes: f32Bytes([...weight.scale]), dtype: "f32" as const };
  const encoding: EncodingInput =
    weight.storage === "f32" || weight.storage === "f16" || weight.storage === "bf16"
      ? { codec: weight.storage }
      : weight.storage === "i4"
        ? { codec: "int4-sym-g", groupSize: weight.groupSize, scale }
        : { codec: weight.storage === "i8" ? "int8-sym" : "int2-off", groupSize: k, scale };
  return { graph: GRAPH_NAME, initializer: "w", bytes, encoding };
};

/**
 * A GpuContext that keeps the real device with an empty adapter identity (so no embedded table
 * matches and the default profile is the baseline whatever this machine is) and, when given, an
 * injected table. MUST NOT be destroyed: the device belongs to the original context.
 */
const contextWith = (gpu: GpuContext, injected?: GeometryProfile): GpuContext =>
  new GpuContext(
    gpu.device,
    platformObject<GPUAdapterInfo>({
      vendor: "",
      architecture: "",
      device: gpu.adapterInfo.device,
      description: "",
      subgroupMinSize: gpu.adapterInfo.subgroupMinSize,
      subgroupMaxSize: gpu.adapterInfo.subgroupMaxSize,
      isFallbackAdapter: gpu.adapterInfo.isFallbackAdapter,
    }),
    gpu.limits,
    gpu.wgslLanguageFeatures,
    undefined,
    injected,
  );

type Run = { readonly words: Uint32Array<ArrayBuffer>; readonly keys: readonly string[] };

const words = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4).slice();

const runSession = async (
  gpu: GpuContext,
  model: () => ReturnType<typeof openGraphModel>,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions,
): Promise<Run> => {
  const session = await createSessionFromContainer(gpu, await model(), GRAPH_NAME, options);
  try {
    const y = (await session.run(inputs))["y"];
    const keys = (session.diagnostics().lastRunPipelines ?? []).map((row) => row.key);
    return { words: words(y), keys };
  } finally {
    await session.dispose();
  }
};

/** A one-linear model `[m, k] × [n, k]ᵀ + [n]` over a weight of the given storage. */
const linearModel = ({ m, n, k }: GemmShape, weight: EncodedLinearWeight) => {
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x0", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x0", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const bias = Float32Array.from({ length: n }, (_, i) => ((i % 5) - 2) * 0.25);
  const tensors: readonly TensorInput[] = [
    weightTensor(weight, n, k),
    { graph: GRAPH_NAME, initializer: "b", bytes: f32Bytes(bias), encoding: { codec: "f32" } },
  ];
  return () => openModelBytes(graph, tensors);
};

/**
 * Runs one case with the default profile and with the rule, and checks the words and that the
 * rule's Session ran the rule's tile (a key starting with `prefix` naming it).
 */
const assertSameWordsUnderRule = async (
  gpu: GpuContext,
  where: string,
  prefix: string,
  shape: GemmShape,
  model: () => ReturnType<typeof openGraphModel>,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
): Promise<void> => {
  const expected = await runSession(contextWith(gpu), model, inputs, options);
  const actual = await runSession(contextWith(gpu, GRID_PROFILE), model, inputs, options);
  const part = gemmGeometryTileKeyPart(gemmShapeGeometry(GRID_PROFILE, shape));
  assert(
    actual.keys.some((key) => key.startsWith(prefix) && key.includes(part)),
    `${where}: did not run a ${prefix}…${part} key (${actual.keys.join(" / ")})`,
  );
  assertEquals(actual.words, expected.words, `${where}: words differ from the M-bucket tile's`);
};

describe("the shapes of the GPU checks", () => {
  it("reach every step of the ladder, with tiles the M bucket would not give", () => {
    const steps = LINEAR_SHAPES.map((shape) => gemmGridStep(RULE, shape));
    assertEquals(steps, [0, 1, 2, 3, 4, 4]);
    const differing = LINEAR_SHAPES.filter(
      (shape) =>
        gemmShapeGeometry(GRID_PROFILE, shape) !==
        gemmRowsGeometry(DEFAULT_GEOMETRY_PROFILE, shape.m),
    );
    assert(
      differing.length >= 4,
      `only ${differing.length} shapes take another tile than the bucket's`,
    );
  });
});

describe.skipIf(!GPU_AVAILABLE)(
  "the grid-fill rule's tile writes the M-bucket tile's words (real GPU)",
  () => {
    it("linear, every storage the GEMM reads, every step of the ladder", async () => {
      const gpu = await acquireGpu(SHADER_F16_AVAILABLE ? { shaderF16: true } : {});
      try {
        for (const { weight: storage, compute } of STORAGES) {
          if (compute === "f16" && !SHADER_F16_AVAILABLE) {
            console.warn("SKIP the f16-compute GEMM: the adapter does not list shader-f16");
            continue;
          }
          for (const shape of LINEAR_SHAPES) {
            const weight = encodeLinearWeight(storage, shape.n, shape.k, "irregular");
            const where = `linear ${storageLabel(storage)} ${compute} ${shape.m}×${shape.n}×${shape.k}`;
            await assertSameWordsUnderRule(
              gpu,
              where,
              "linear:",
              shape,
              linearModel(shape, weight),
              {
                x0: {
                  dtype: "f32",
                  shape: [shape.m, shape.k],
                  data: activations(shape.m, shape.k),
                },
              },
              compute === "f16" ? { linearCompute: "f16" } : {},
            );
          }
        }
      } finally {
        gpu.destroy();
      }
    });

    it("matmul and bmm, the batch counted on the grid's z axis", async () => {
      const gpu = await acquireGpu();
      try {
        for (const shape of LINEAR_SHAPES) {
          const { m, n, k } = shape;
          await assertSameWordsUnderRule(
            gpu,
            `matmul ${m}×${n}×${k}`,
            "matmul:",
            shape,
            async () =>
              await openGraphModel(
                singleOpDeclaration(
                  "matmul",
                  [
                    [m, k],
                    [k, n],
                  ],
                  [[m, n]],
                ),
              ),
            { x0: fill([m, k], SIGNED), x1: fill([k, n], WEIGHT) },
          );
        }
        for (const shape of [
          { m: 64, n: 64, k: 128, batch: 16 },
          { m: 96, n: 72, k: 40, batch: 6 },
          { m: 300, n: 136, k: 64, batch: 2 },
          { m: 130, n: 100, k: 64, batch: 3 },
        ]) {
          const { m, n, k, batch } = shape;
          await assertSameWordsUnderRule(
            gpu,
            `bmm ${batch}×${m}×${n}×${k}`,
            "bmm:",
            shape,
            async () =>
              await openGraphModel(
                singleOpDeclaration(
                  "bmm",
                  [
                    [batch, m, k],
                    [batch, k, n],
                  ],
                  [[batch, m, n]],
                ),
              ),
            { x0: fill([batch, m, k], SIGNED), x1: fill([batch, k, n], WEIGHT) },
          );
        }
      } finally {
        gpu.destroy();
      }
    });

    it("the split-K partial at one slice equals the reference GEMM on the rule's tile of every shape", async () => {
      const gpu = await acquireGpu();
      try {
        const run = splitKRunner(gpu);
        for (const { weight: storage, compute } of STORAGES) {
          if (compute === "f16") continue;
          for (const shape of LINEAR_SHAPES) {
            const { m, n, k } = shape;
            const result = await run({
              geometry: gemmShapeGeometry(GRID_PROFILE, shape),
              m,
              n,
              k,
              slices: 1,
              weight: encodeLinearWeight(storage, n, k, "irregular"),
              x: activations(m, k),
              bias: Float32Array.from({ length: n }, (_, i) => ((i % 5) - 2) * 0.25),
            });
            assertEquals(
              result.split,
              result.reference,
              `split-K ${storageLabel(storage)} ${m}×${n}×${k} at one slice`,
            );
          }
        }
      } finally {
        gpu.destroy();
      }
    });
  },
);

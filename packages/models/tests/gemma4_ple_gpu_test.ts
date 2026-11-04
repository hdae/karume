/**
 * Checks of the **GPU-free part** of the PLE GPU-resident option (ADR 0085 addendum (GPU-resident option)).
 *
 * Four things are checked — ① accepting and rejecting the option's setting ② the shape of the gather IR (node list,
 * shapes, initializer declarations) and the graph-only rules check (`parseIrDeclarationValue`) ③ the container's
 * assets → the in-memory container (storage dtype → codec, the order of pieces and their part split, and that the
 * reassembled bytes match the original) ④ the supply form of an index with a single block. Value agreement on a
 * real GPU is owned by `e2e_gemma4_ple_gpu_test.ts`.
 */
import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  type IrDeclaration,
  IrError,
  parseIrDeclarationValue,
  prepareContainer,
} from "@karume/runtime";
import {
  buildGemma4PleGatherContainer,
  gemma4PleGatherGraph,
  gemma4PleGpuBytes,
} from "../src/gemma/ple-gpu.ts";
import { ModelInputError } from "../src/errors.ts";
import { resolveGemma4PleResidency } from "../src/gemma/pipeline.ts";
import { pleFixture, type PleFixtureSpec, scaleOf } from "./helpers/ple-fixture.ts";

/** A made-up single index (the same spelling and relations as the real assets, only smaller). */
const SPEC: PleFixtureSpec = {
  storage: "i4",
  tokens: 6,
  layers: 2,
  dim: 16,
  embedScale: 16,
  valueRows: 4,
};

/** Bytes per layer (dim / 2 since it is i4). */
const ROW_BYTES = SPEC.dim / 2;
/** Rows of the weight (token × layer). */
const WEIGHT_ROWS = SPEC.tokens * SPEC.layers;

const rig = pleFixture(SPEC);
const index = rig.index;

test("pleResidency: acceptance, and rejection of knobs that cannot be combined", () => {
  assertEquals(resolveGemma4PleResidency("T", {}), "host");
  assertEquals(resolveGemma4PleResidency("T", { pleResidency: "host" }), "host");
  assertEquals(resolveGemma4PleResidency("T", { pleResidency: "gpu" }), "gpu");
  // A misspelling is not turned into the default (a setting that comes from outside the types).
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "GPU" as "gpu" }),
    Error,
    "pleResidency",
  );
  // A combination of knobs that cannot be used together is decided by the caller's options alone, so it is caused by the input (ADR 0107 decision 2).
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", maxResidentPleBytes: 0 }),
    ModelInputError,
    "maxResidentPleBytes",
  );
  assertThrows(
    () => resolveGemma4PleResidency("T", { pleResidency: "gpu", speculative: {} }),
    ModelInputError,
    "speculative",
  );
  // The other side: a misspelling belongs to "re-reading the accepted set", so it is not classified as caused by the input (the exclusion of decision 3).
  const misspelled = assertThrows(() =>
    resolveGemma4PleResidency("T", { pleResidency: "GPU" as "gpu" }),
  );
  assert(!(misspelled instanceof ModelInputError));
});

test("gather IR: two nodes, embedding → mul, and the index shape becomes the per-layer shape as is", () => {
  const graph = gemma4PleGatherGraph(index);
  assertEquals(
    graph.nodes.map((node) => node.op),
    ["embedding", "mul"],
  );
  assertEquals(graph.nodes[0].ins, ["ple", "ple_index"]);
  assertEquals(graph.nodes[1].ins, ["ple_raw", "embed_scale"]);
  assertEquals(graph.inputs, [{ name: "ple_index", dtype: "i32", shape: [1, "M", 2] }]);
  // The weight is folded into rows of (token, layer) — 1 row = 1 layer, so the order of the scales lines up with the index.
  assertEquals(graph.values.ple.shape, [WEIGHT_ROWS, SPEC.dim]);
  assertEquals(graph.values[graph.outputs[0]].shape, [1, "M", 2, SPEC.dim]);
  // An IR v2 declaration holds only names (the storage is in the binding table = the in-memory container's encoding side).
  assertEquals(Object.keys(graph.initializers), ["ple", "embed_scale"]);
  assertEquals(graph.initializers.ple, { shared: false });
});

test("in-memory container: the storage dtype maps to the codec ledger's registered name and group length", async () => {
  const mapping = [
    ["i8", "int8-sym"],
    ["i4", "int4-sym-g"],
    ["i2", "int2-off"],
  ] as const;
  for (const [storage, codec] of mapping) {
    const fixture = pleFixture({ ...SPEC, storage });
    const bound = await buildGemma4PleGatherContainer(fixture.index, fixture.openBlock, "T");
    const encoding = bound.graphs.ple.supplies.get("ple")?.encoding;
    assertEquals(encoding?.codec, codec);
    // i4's group = the row length; i8 / i2's per-channel = the row length. Either way the group count is 1, so the
    // companion scale is exactly the f32 column `[tokens × layers, 1]`.
    assertEquals(encoding?.groupSize, SPEC.dim);
    assertEquals(encoding?.rowAxis, 0);
    // Checks through to the storage after the merge (all 3 pass the capability check — the GPU is not touched).
    assertEquals(prepareContainer(bound, "ple").graph.initializers.ple.storage, {
      codec,
      groupSize: SPEC.dim,
      rowAxis: 0,
    });
  }
});

test("in-memory container: an index with a single block is supplied whole, not split into pieces", async () => {
  // Omitting `valueRows` gives a single block (`pieces` MUST have 2 or more, so it takes the `bytes` branch).
  // The old implementation rejected this as "fewer than 2 PLE rows", so this pins that it passes.
  const fixture = pleFixture({ ...SPEC, storage: "i8", valueRows: undefined });
  assertEquals(fixture.index.values.blocks.length, 1);
  const bound = await buildGemma4PleGatherContainer(fixture.index, fixture.openBlock, "T");
  const supply = bound.graphs.ple.supplies.get("ple");
  assert(supply !== undefined, "no supply for initializer 'ple'");
  assertEquals(supply.blocks.length, 1);
  assertEquals(supply.blocks[0].rows, [0, WEIGHT_ROWS]);
  assertEquals(
    await bound.readBlock(supply.blocks[0].id),
    fixture.blocks.get(fixture.index.values.blocks[0].asset),
  );
  // The companion scale is placed in the same part as the single whole supply too (rule ③).
  assertEquals(supply.scale?.part, supply.blocks[0].part);
  assertEquals(prepareContainer(bound, "ple").graph.outputs, ["per_layer"]);
});

/**
 * Turns a parsed declaration back into a JSON document (`shared: false` cannot be spelled, so the field is dropped)
 * — the base the doctored inputs are put on.
 */
const documentOf = (declaration: IrDeclaration): Record<string, unknown> => ({
  ...declaration,
  initializers: Object.fromEntries(Object.keys(declaration.initializers).map((name) => [name, {}])),
});

test("gather IR: passes the graph-only rules (a false requires.ops fails at the declaration step)", () => {
  const document = documentOf(gemma4PleGatherGraph(index));
  // Negative control: a document without doctoring passes as is and returns to the same declaration.
  assertEquals(parseIrDeclarationValue(document), gemma4PleGatherGraph(index));
  // Declaring an unused op is seen neither by the merge (declaration × supply) nor by `prepareContainer` (capability +
  // op contracts) — only this check, which looks at the graph-only rules, can reject it.
  assertThrows(
    () => parseIrDeclarationValue({ ...document, requires: { ops: ["embedding", "mul", "topk"] } }),
    IrError,
    "requires.ops",
  );
});

test("gather IR: passes the runtime's op contracts and storage", async () => {
  const bound = await buildGemma4PleGatherContainer(index, rig.openBlock, "T");
  // prepareContainer checks every unsupported op / storage codec and the IR contracts (the GPU is not touched).
  const prepared = prepareContainer(bound, "ple");
  assertEquals(prepared.graph.outputs, ["per_layer"]);
  assertEquals(prepared.graph.symbols, ["M"]);
});

test("resident bytes: decided by the index alone (the actual values of E2B QAT)", () => {
  const e2b = pleFixture({
    storage: "i4",
    tokens: 4096,
    layers: 35,
    dim: 256,
    embedScale: 16,
  }).index;
  assertEquals(gemma4PleGpuBytes(e2b), {
    values: (4096 * 35 * 256) / 2,
    scales: 4096 * 35 * 4,
    total: (4096 * 35 * 256) / 2 + 4096 * 35 * 4,
  });
  // i8 (regular Gemma 4) doubles at the same dimensions.
  const i8 = pleFixture({
    storage: "i8",
    tokens: 4096,
    layers: 35,
    dim: 256,
    embedScale: 16,
  }).index;
  assertEquals(gemma4PleGpuBytes(i8).values, 4096 * 35 * 256);
  const i2 = pleFixture({
    storage: "i2",
    tokens: 4096,
    layers: 35,
    dim: 256,
    embedScale: 16,
  }).index;
  assertEquals(gemma4PleGpuBytes(i2).values, (4096 * 35 * 256) / 4);
});

test("in-memory container: pieces are in order, and reassembling them matches the original values", async () => {
  const bound = await buildGemma4PleGatherContainer(index, rig.openBlock, "T");
  const supply = bound.graphs.ple.supplies.get("ple");
  assert(supply !== undefined, "no supply for initializer 'ple'");
  // Pieces are 1:1 with the `values` blocks (the row range is the token interval × the layer count — 1 row = 1 layer).
  assertEquals(
    supply.blocks.map((block) => block.rows),
    [
      [0, 8],
      [8, 12],
    ],
  );
  // Part split: each piece k is in its own part (= one fence at build time), and the companion scale is in the same
  // part as piece 1 (rule ③ of container-v1 §13.3).
  const parts = supply.blocks.map((block) => block.part);
  assertEquals(new Set(parts).size, parts.length, "pieces share a part");
  assertEquals(supply.scale?.part, parts[0]);

  // Joining the pieces re-read in block order gives exactly the concatenation of the original values.
  const collected: Uint8Array[] = [];
  for (const block of supply.blocks) collected.push(await bound.readBlock(block.id));
  assertEquals(
    collected.map((piece) => piece.byteLength / ROW_BYTES),
    [8, 4],
  );
  const joined = new Uint8Array(WEIGHT_ROWS * ROW_BYTES);
  let cursor = 0;
  for (const piece of collected) {
    joined.set(piece, cursor);
    cursor += piece.byteLength;
  }
  const expected = new Uint8Array(WEIGHT_ROWS * ROW_BYTES);
  let at = 0;
  for (const block of index.values.blocks) {
    const bytes = rig.blocks.get(block.asset)!;
    expected.set(bytes, at);
    at += bytes.byteLength;
  }
  assertEquals(joined, expected);

  // The scale block is exactly the scale table reordered to the flat index `id * layers + layer`. The fixture's
  // scales also depend on the token (`scaleOf`), so this claim also fails on a shift in the token direction (rotated
  // rows, mixed-up blocks).
  const scaleBytes = await bound.readBlock(supply.scale!.id);
  assertEquals(scaleBytes.byteLength, WEIGHT_ROWS * 4);
  const scales = new Float32Array(scaleBytes.buffer, scaleBytes.byteOffset, WEIGHT_ROWS);
  assertEquals(
    [...scales],
    Array.from({ length: WEIGHT_ROWS }, (_row, row) =>
      scaleOf(Math.floor(row / SPEC.layers), row % SPEC.layers),
    ),
  );

  // embed_scale is supplied whole (f32, no padding).
  const embedScale = bound.graphs.ple.supplies.get("embed_scale");
  assertEquals(embedScale?.encoding.codec, "f32");
  const embedBytes = await bound.readBlock(embedScale!.blocks[0].id);
  assertEquals(new Float32Array(embedBytes.buffer, embedBytes.byteOffset, 1)[0], SPEC.embedScale);
});

// GPU-free checks of Wan2.2 TI2V-5B's two VAE decoder chunk graphs (the f16 containers of the mirror
// `models/karume-wan2.2/`, and the fixtures of the series `wan2.2-ti2v-5b-f16-dyn` in the test-asset root — ADR 0139
// decisions 1 / 2; ADR 0121 stage 4).
//
// Three things are checked:
//
// - **Container declaration → chunk-sequence contract**: the same host function as 2.1 (`wanVaeChunkLayout` in
//   `src/wan/vae-chunks.ts`) derives from the 2.2 declaration the 48 latent channels, tile 16, 12 output channels, the
//   cache table (32 entries; first has 30, without the two of `time_conv`) and the frame shapes. The table is in
//   `helpers/wan-ti2v-vae.ts` (frozen twice, together with the recipe's table).
// - **IR claims as read by the runtime**: the rank (values are rank 4 or less; rank 5 only for initializers that are a
//   conv3d's second input) and the op set (within the same 15-op vocabulary as 2.1) are the same claims as the recipe's
//   `export_vae.assert_chunk_graph`, counted by a different reader (the declaration of the runtime that opened the
//   container). One attention node per graph is an extra claim the recipe does not check (the single-head mid attention
//   of ADR 0121 decision 6). Declarations rewritten per claim are each rejected, to show the check is not vacuous.
//   NOTE: the node count and the conv3d count are not pinned (they change whenever the exporter's decomposition
//   changes harmlessly; the three claims above carry the structure).
// - **Fixtures**: the metadata (tile, chunk count, role, weight rounding, reference provenance) and shapes match the
//   table.
//
// The container checks need the mirror; the fixture check needs the three fixtures. Without them the cases SKIP
// explicitly, naming what is missing (ADR 0005). Fixtures present only in part are a FAIL, not a SKIP (the three are
// written together). No GPU is used (the real-GPU chunk-sequence check is `e2e_wan_ti2v_vae_tiles_test.ts`).
//
// NOTE: the containers' graph names are the manifest's weights keys (container-v1 §2.1).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { prepareContainer, type PreparedModel } from "@karume/runtime";
import { wanVaeChunkLayout, wanVaeFrameCount } from "../../models/src/wan/vae-chunks.ts";
import { readFixture } from "../../models/tests/helpers/wan-vae-chunk-loop.ts";
import { TEST_ASSETS_FETCH as FETCH } from "../../models/tests/helpers/wan-ti2v-dit.ts";
import {
  openWanTi2vComponent,
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import { mirrorPresent } from "../../models/tests/helpers/mirror.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";
import {
  ti2vVaeChunkFixtures,
  ti2vVaeFixtureUrl,
  WAN_TI2V_VAE_CACHE_BYTES as CACHE_BYTES,
  WAN_TI2V_VAE_CACHE_TABLE as CACHE_TABLE,
  WAN_TI2V_VAE_CASES as CASES,
  WAN_TI2V_VAE_FIRST as FIRST,
  WAN_TI2V_VAE_FIRST_CACHE_BYTES as FIRST_CACHE_BYTES,
  WAN_TI2V_VAE_GOLDEN_DIR as GOLDEN_DIR,
  WAN_TI2V_VAE_LATENT_CHANNELS as LATENT_CHANNELS,
  WAN_TI2V_VAE_NEXT as NEXT,
  WAN_TI2V_VAE_SAMPLE_CHANNELS as SAMPLE_CHANNELS,
  WAN_TI2V_VAE_SAMPLE_TILE as SAMPLE_TILE,
  WAN_TI2V_VAE_TILE as TILE,
  WAN_TI2V_VAE_TIME_CONV_CACHES as TIME_CONV_CACHES,
} from "../../models/tests/helpers/wan-ti2v-vae.ts";

type Graph = PreparedModel["graph"];

/**
 * The ops allowed in a chunk graph's IR (the recipe's `export_vae.EXPECTED_OPS` — the same 15 as 2.1). SiLU is
 * decomposed into `x · sigmoid(x)`, and the mid attention is SDPA kept as is (ADR 0023).
 */
const CHUNK_GRAPH_OPS: ReadonlySet<string> = new Set([
  "add",
  "attention",
  "cat",
  "clamp_min",
  "conv2d",
  "conv3d",
  "div",
  "expand",
  "mul",
  "permute",
  "reshape",
  "sigmoid",
  "slice",
  "sqrt",
  "sum",
]);

/**
 * The upper bound of a value's rank (the bound of the strided-copy ops — ADR 0118 decision 2). Rank 5 is allowed only
 * for conv3d weights.
 */
const MAX_VALUE_RANK = 4;

/** The number of single-head mid attention nodes (per graph — ADR 0121 decision 6). */
const ATTENTION_NODES = 1;

/**
 * The expected fixture metadata (verbatim from the writer `export_vae.emit_targets` — for 2.2 the reference is
 * before unpatchify).
 */
const FIXTURE_WEIGHTS = "f16-rounded";
const FIXTURE_REFERENCE =
  "diffusers AutoencoderKLWan._decode chunk loop before unpatchify and clamp (CPU f32)";

const MIRROR_PRESENT = mirrorPresent(DIST_ROOT);
const ASSETS = ti2vVaeChunkFixtures();
const ANY_PRESENT = ASSETS.some(({ present }) => present);
const ALL_PRESENT = ASSETS.every(({ present }) => present);
/**
 * A complete fetch of the pinned test-asset dataset makes a missing fixture a FAIL (the completeness test below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_FIXTURES = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the GPU-free checks of Wan2.2 TI2V's VAE chunk graphs are SKIPPED. Assemble it: ${ASSEMBLE}`,
  );
}
if (!ANY_PRESENT) {
  console.warn(
    `[karume] no VAE chunk fixture under ${GOLDEN_DIR.pathname}, so the fixture check is SKIPPED ` +
      `(${MISSING_FIXTURES})`,
  );
}

/** Opens and prepares the mirror's f16 container (the graph name is the component key). */
const prepared = async (component: string): Promise<PreparedModel> =>
  prepareContainer(await openWanTi2vComponent(component, "f16"), component);

const elements = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** The set of names of the conv3d second inputs (weights). */
const conv3dWeights = (graph: Graph): ReadonlySet<string> =>
  new Set(graph.nodes.filter(({ op }) => op === "conv3d").map(({ ins }) => ins[1]));

/**
 * The list of violations of the IR claims (the second item at the top of the file; empty when they hold). Values are
 * checked both in the graph inputs (not listed in `values`) and in `values` (initializers included).
 */
const irViolations = (graph: Graph): readonly string[] => {
  const violations: string[] = [];
  const unexpected = [...new Set(graph.nodes.map(({ op }) => op))]
    .filter((op) => !CHUNK_GRAPH_OPS.has(op))
    .sort();
  if (unexpected.length > 0)
    violations.push(`ops outside the vocabulary: ${unexpected.join(" / ")}`);
  const weights = conv3dWeights(graph);
  const shapes = [
    ...graph.inputs.map(({ name, shape }) => [name, shape] as const),
    ...Object.entries(graph.values).map(([name, { shape }]) => [name, shape] as const),
  ];
  for (const [name, shape] of shapes) {
    if (shape.length <= MAX_VALUE_RANK) continue;
    const conv3dWeight =
      shape.length === 5 && weights.has(name) && Object.hasOwn(graph.initializers, name);
    if (!conv3dWeight) {
      violations.push(
        `rank-${shape.length} value '${name}' (rank 5 is allowed only for conv3d weights)`,
      );
    }
  }
  const attention = graph.nodes.filter(({ op }) => op === "attention").length;
  if (attention !== ATTENTION_NODES) {
    violations.push(`${attention} attention nodes (expected ${ATTENTION_NODES})`);
  }
  return violations;
};

test.skipIf(!ANY_PRESENT && !FETCHED)(
  "Wan2.2 TI2V VAE chunk fixtures: the three fixtures are present",
  () => {
    assertEquals(
      ASSETS.filter(({ present }) => !present).map(({ path }) => path),
      [],
      `missing fixtures (${MISSING_FIXTURES})`,
    );
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE chunk declaration (no GPU): 48 latent channels, tile 16, 12 output channels, the table's 32 caches" +
    " (first has 30, without the two of time_conv), frames [12,1,128,128] / [12,4,128,128]",
  async () => {
    const layout = wanVaeChunkLayout(await prepared(FIRST), await prepared(NEXT));
    assertEquals(
      [layout.latentChannels, layout.tile, layout.sampleTile, layout.sampleChannels],
      [LATENT_CHANNELS, TILE, SAMPLE_TILE, SAMPLE_CHANNELS],
      "latent channels, tile, frame edge, output channels",
    );
    assertEquals(
      [...layout.cacheShapes.entries()],
      CACHE_TABLE.map(({ name, shape }) => [name, shape]),
      "cache table (in next's input order)",
    );
    assertEquals(
      layout.next.caches,
      CACHE_TABLE.map(({ name }) => name),
    );
    assertEquals(
      layout.first.caches,
      CACHE_TABLE.filter(({ timeConv }) => !timeConv).map(({ name }) => name),
      "first's caches = table − time_conv",
    );
    assertEquals(
      layout.next.caches.filter((name) => !layout.first.caches.includes(name)),
      [...TIME_CONV_CACHES],
    );
    assertEquals(TIME_CONV_CACHES, ["cache_11", "cache_18"]);
    assertEquals(layout.first.frameShape, [SAMPLE_CHANNELS, 1, SAMPLE_TILE, SAMPLE_TILE]);
    assertEquals(layout.next.frameShape, [SAMPLE_CHANNELS, 4, SAMPLE_TILE, SAMPLE_TILE]);
    assertEquals(
      [...layout.cacheShapes.values()].reduce((sum, shape) => sum + 4 * elements(shape), 0),
      CACHE_BYTES,
      "total cache bytes (f32)",
    );
    assertEquals(
      layout.first.caches.reduce((sum, name) => {
        const shape = layout.cacheShapes.get(name);
        assert(
          shape !== undefined,
          `the shape of first's cache '${name}' is missing from the layout`,
        );
        return sum + 4 * elements(shape);
      }, 0),
      FIRST_CACHE_BYTES,
      "total bytes of first's caches (f32)",
    );
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE chunk IR (no GPU): rank 5 only for conv3d weights, ops within the 15-op vocabulary, one" +
    " attention per graph",
  async () => {
    for (const component of [FIRST, NEXT]) {
      const { graph } = await prepared(component);
      assertEquals(irViolations(graph), [], component);
      // The rank-5 exception actually applies (without a single conv3d weight the claim above would be vacuous).
      const rank5 = Object.entries(graph.values)
        .filter(([, { shape }]) => shape.length === 5)
        .map(([name]) => name)
        .sort();
      assert(rank5.length > 0, `${component}: no rank-5 value`);
      assertEquals(
        rank5,
        [...conv3dWeights(graph)].sort(),
        `${component}: the set of rank-5 values`,
      );
    }
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V VAE chunk IR check itself (no GPU): rejects each rewritten declaration" +
    " (a conv3d weight that is not an initializer, a rank-5 graph input, a rank-5 conv2d weight, an op outside the vocabulary, zero attention nodes)",
  async () => {
    // Break one point of a declaration that holds (the test above) — exactly one violation, worded as that claim.
    const { graph } = await prepared(NEXT);
    const conv2d = graph.nodes.find(({ op }) => op === "conv2d");
    assert(conv2d !== undefined, "no conv2d node");
    const weight = conv2d.ins[1];
    assert(
      Object.hasOwn(graph.initializers, weight),
      `the conv2d weight '${weight}' is not an initializer`,
    );
    const declared = graph.values[weight];
    // The node rewritten to an op outside the vocabulary is not a conv3d (rewriting a conv3d would also drop the rank-5
    // exception of its weight).
    const multiply = graph.nodes.findIndex(({ op }) => op === "mul");
    const attention = graph.nodes.findIndex(({ op }) => op === "attention");
    assert(multiply >= 0 && attention >= 0, "no mul / attention node");
    // The remaining branches of the rank-5 exception: a conv3d weight that is not an initializer is rejected; graph
    // inputs are checked too.
    const [conv3dWeight] = [...conv3dWeights(graph)];
    assert(conv3dWeight !== undefined, "no conv3d node");
    const [input] = graph.inputs;
    assert(input !== undefined && input.shape.length === 4, "no rank-4 graph input");
    const rejected: readonly [string, Graph, string][] = [
      [
        "a conv3d weight that is not an initializer",
        {
          ...graph,
          initializers: Object.fromEntries(
            Object.entries(graph.initializers).filter(([name]) => name !== conv3dWeight),
          ),
        },
        `rank-5 value '${conv3dWeight}'`,
      ],
      [
        "a rank-5 graph input",
        {
          ...graph,
          inputs: graph.inputs.map((declared, index) =>
            index === 0 ? { ...declared, shape: [1, ...declared.shape] } : declared,
          ),
        },
        `rank-5 value '${input.name}'`,
      ],
      [
        "a rank-5 non-conv3d value (the initializer of a conv2d weight)",
        {
          ...graph,
          values: { ...graph.values, [weight]: { ...declared, shape: [1, ...declared.shape] } },
        },
        `rank-5 value '${weight}'`,
      ],
      [
        "an op outside the vocabulary",
        {
          ...graph,
          nodes: graph.nodes.map((node, index) =>
            index === multiply ? { ...node, op: "pad" } : node,
          ),
        },
        "ops outside the vocabulary: pad",
      ],
      [
        "zero attention nodes",
        {
          ...graph,
          nodes: graph.nodes.map((node, index) =>
            index === attention ? { ...node, op: "mul" } : node,
          ),
        },
        "0 attention nodes",
      ],
    ];
    for (const [label, broken, message] of rejected) {
      const violations = irViolations(broken);
      assertEquals(violations.length, 1, `${label}: ${violations.join(" / ")}`);
      assertStringIncludes(violations[0], message, label);
    }
  },
);

test.skipIf(!ALL_PRESENT)(
  "Wan2.2 TI2V VAE chunk fixtures (no GPU): metadata (tile 16, chunk count, role, f16 rounding, reference provenance)" +
    " and shapes (latents [48,F,16,16], frames [12,1+4(F−1),128,128]) match the table; the three seeds all differ",
  async () => {
    const seeds: string[] = [];
    for (const { name, role, chunks } of CASES) {
      const fixture = await readFixture(ti2vVaeFixtureUrl(name));
      const meta = (key: string): string | undefined => fixture.metadata.get(key);
      assertEquals(
        {
          tile: meta("tile"),
          chunks: meta("chunks"),
          role: meta("role"),
          weights: meta("weights"),
          reference: meta("reference"),
        },
        {
          tile: String(TILE),
          chunks: String(chunks),
          role,
          weights: FIXTURE_WEIGHTS,
          reference: FIXTURE_REFERENCE,
        },
        `${name}: metadata`,
      );
      assertEquals(fixture.latentShape, [LATENT_CHANNELS, chunks, TILE, TILE], `${name}: latents`);
      assertEquals(
        fixture.frameShape,
        [SAMPLE_CHANNELS, wanVaeFrameCount(chunks), SAMPLE_TILE, SAMPLE_TILE],
        `${name}: frames`,
      );
      const seed = meta("seed");
      assert(seed !== undefined, `${name}: metadata 'seed' is missing`);
      seeds.push(seed);
    }
    // The band-setting and the acceptance cases use different latents (setting and acceptance stay independent — ADR
    // 0118 addendum 2026-10-02).
    assertEquals(new Set(seeds).size, CASES.length, `duplicate seeds (${seeds.join(" / ")})`);
  },
);

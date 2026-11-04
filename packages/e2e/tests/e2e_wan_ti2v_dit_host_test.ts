// GPU-free comparisons of the Wan2.2 TI2V-5B DiT (i8 series `wan2.2-ti2v-5b-i8-dyn`) (ADR 0121 steps 1 / 2).
//
// Two things are checked:
//
// - **Host functions × golden inputs**: the 2.1 host functions (`src/wan/dit-*.ts`) reproduce the golden graph inputs
//   for the 5B with 48 latent channels too (patchify and the RoPE tables from the asset `rope_base` bit-identical;
//   `timesteps_proj` within atol). Of the 2 inputs for I2V (`timesteps_proj_condition`, `condition_mask`),
//   `src/wan/dit-loop.ts` builds only the T2V form (step 6) and there is no TS implementation of the I2V form yet
//   (step 9), so the golden's convention is pinned here: the condition-side timestep is `condition_timestep` of
//   `reference` (0 for I2V; the same as the generation side for T2V), and the condition mask is 1 (u32) on the H'·W'
//   tokens of the first latent frame for I2V and all 0 for T2V. The host condition mask inside the tests
//   (`conditionMask` of `helpers/wan-ti2v-dit.ts` — used by the production-path input and the fault injection of the
//   real-GPU r check) is also held bit-identical to the golden. Both the S = 192 goldens (step 1) and the goldens of
//   the 2 full-size shapes (step 2 — latents `[48,21,30,52]` / `[48,9,44,80]`, P = 390 / 880) are checked.
// - **The container's declaration as the runtime sees it**: 7 inputs (the condition mask is bool `[1,S,1]`), 310
//   linear nodes (the 3 of the timestep MLP run twice at M = 1 — the dispatch count of decision 2), 182 `where` (30
//   layers × 6 + 2 in the head), 307 i8 weights. The same counts as the recipe's IR check (`wan/ti2v_export_dit.py`),
//   counted through the runtime's reader.
//
// The container is the int8 transformer of the distribution mirror `models/karume-wan2.2/` (through its manifest —
// ADR 0139 decision 1) and the goldens are in the test-asset root (decision 2). Without them the cases SKIP
// explicitly, naming what is missing (ADR 0005). A golden group present only in part is a FAIL, not a SKIP (the
// completeness tests below — the S = 192 group and the full-size group are written in separate runs, so each group is
// checked on its own). No GPU is used (the real-GPU r gate is `e2e_wan_ti2v_dit_test.ts`).
//
// NOTE: the container's graph name is the manifest's weights key (`transformer` — container-v1 §2.1).

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { codecLayout, type OpenedContainer, prepareContainer } from "@karume/runtime";
import { WAN22_TI2V_GENERATION } from "../../models/src/wan/descriptor.ts";
import { ditContract } from "../../models/src/wan/dit-loop.ts";
import { patchifyLatents, wanTokenGrid } from "../../models/src/wan/dit-tokens.ts";
import { wanRopeTables } from "../../models/src/wan/dit-rope.ts";
import { timestepsProj } from "../../models/src/wan/dit-timestep.ts";
import {
  caseFiles,
  conditionMask,
  filePresent,
  firstBitMismatch,
  loadTi2vGolden,
  openTi2vTransformer,
  readRopeBase,
  storedConditionMask,
  TEST_ASSETS_FETCH as FETCH,
  type Ti2vCase,
  WAN22_GEOMETRY,
  WAN_TI2V_CASES as CASES,
  WAN_TI2V_COMPONENT as COMPONENT,
  WAN_TI2V_FULL_CASES as FULL_CASES,
  WAN_TI2V_GOLDEN_DIR as GOLDEN_DIR,
  wanTi2vMirrorPresent,
} from "../../models/tests/helpers/wan-ti2v-dit.ts";
import { floatsOf } from "../../models/tests/helpers/safetensors.ts";
import {
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";

/**
 * Tolerance (absolute) between the TS implementation of `timesteps_proj` and upstream (torch CPU f32).
 *
 * The same function and the same width (`freq_dim` 256) as 2.1, so the measurement (worst over all 1,001 values
 * 3.0517e-5 — 1 ULP of `exp`) and the basis of the band stay those of `TIMESTEPS_PROJ_ATOL` in `wan_dit_host_test.ts`.
 */
const TIMESTEPS_PROJ_ATOL = 1.5e-4;

/** The counts of decisions 2 / 3 (5B — 30 layers). */
const EXPECTED_LINEAR_NODES = 310;
const EXPECTED_WHERE_NODES = 30 * 6 + 2;
const EXPECTED_I8_WEIGHTS = 307;

/** Declaration of the graph inputs (names and shapes of `INPUT_NAMES` in `wan/ti2v_export_dit.py`). */
const EXPECTED_INPUTS = [
  { name: "tokens", dtype: "f32", shape: [1, "S", 192] },
  { name: "timesteps_proj", dtype: "f32", shape: [1, 256] },
  { name: "encoder_hidden_states", dtype: "f32", shape: [1, 512, 4096] },
  { name: "rope_cos", dtype: "f32", shape: [1, "S", 1, 128] },
  { name: "rope_sin", dtype: "f32", shape: [1, "S", 1, 128] },
  { name: "timesteps_proj_condition", dtype: "f32", shape: [1, 256] },
  { name: "condition_mask", dtype: "bool", shape: [1, "S", 1] },
] as const;

const expectedFiles = CASES.flatMap(({ name }) => caseFiles(name));
/**
 * A complete fetch of the pinned test-asset dataset makes a missing golden a FAIL (the completeness tests below): the
 * dataset at the pin lacks a file the cases read. Without one, the cases SKIP (the assets gate is red then).
 */
const FETCHED = testAssetsFetched();
const MISSING_GOLDENS = FETCHED
  ? `the test-asset dataset at revision ${TEST_ASSETS.revision} lacks them`
  : `fetch: ${FETCH}`;
const MIRROR_PRESENT = wanTi2vMirrorPresent();
const ANY_PRESENT = expectedFiles.some(filePresent);
const ASSETS_AVAILABLE = MIRROR_PRESENT && expectedFiles.every(filePresent);

if (!MIRROR_PRESENT) {
  console.warn(
    `[karume] the mirror ${DIST_ROOT.pathname} is absent (or not karume/5, or a sibling it references is ` +
      `absent), so the GPU-free checks of Wan2.2 TI2V's DiT are SKIPPED. Assemble it: ${ASSEMBLE}`,
  );
} else if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] the S = 192 goldens under ${GOLDEN_DIR.pathname} are not all present, so the GPU-free ` +
      `host checks are SKIPPED (${MISSING_GOLDENS})`,
  );
}

// The full-size goldens are a group of their own (written in a separate run of the recipe — `write-full`, hours on
// the CPU): none present SKIPs unless the root is a complete fetch, any one present makes a missing one a FAIL.
const fullExpectedFiles = FULL_CASES.flatMap(({ name }) => caseFiles(name));
const FULL_ANY_PRESENT = fullExpectedFiles.some(filePresent);
const FULL_AVAILABLE = MIRROR_PRESENT && fullExpectedFiles.every(filePresent);

if (MIRROR_PRESENT && !FULL_ANY_PRESENT) {
  console.warn(
    `[karume] no full-size golden under ${GOLDEN_DIR.pathname}, so the full-size GPU-free host checks are ` +
      `SKIPPED (${MISSING_GOLDENS})`,
  );
}

const maxAbsDiff = (actual: Float32Array, expected: Float32Array): number => {
  assertEquals(actual.length, expected.length, "element count");
  let worst = 0;
  for (let index = 0; index < actual.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  return worst;
};

/** The container's graph (named by the manifest's weights key — the NOTE at the top of the file). */
const containerGraphOf = (opened: OpenedContainer) => {
  assertEquals(Object.keys(opened.graphs), [COMPONENT], "container graph names");
  return opened.graphs[COMPONENT];
};

// Only a root with no S = 192 golden that is not a complete fetch SKIPs; the mirror's completeness is the distribution
// gate's.
test.skipIf(!ANY_PRESENT && !FETCHED)(
  "Wan2.2 TI2V DiT goldens: the goldens of every case (S = 192, 5 T2V + 5 I2V) are present",
  () => {
    assertEquals(
      expectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `missing goldens (${MISSING_GOLDENS})`,
    );
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V DiT container (no GPU): 7 inputs, condition mask is bool [1,S,1], linear 310, where 182, i8 307",
  async () => {
    const graph = containerGraphOf(await openTi2vTransformer());
    const { declaration } = graph;
    assertEquals(
      declaration.inputs.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
      EXPECTED_INPUTS.map(({ name, dtype, shape }) => ({ name, dtype, shape: [...shape] })),
    );
    assertEquals(declaration.symbols, ["S"]);
    const count = (op: string): number => declaration.nodes.filter((node) => node.op === op).length;
    assertEquals(count("linear"), EXPECTED_LINEAR_NODES, "number of linear nodes (dispatches)");
    assertEquals(count("where"), EXPECTED_WHERE_NODES, "number of where nodes");
    const i8 = [...graph.supplies.values()].filter(
      ({ encoding }) => codecLayout(encoding.codec) === "i8",
    );
    assertEquals(i8.length, EXPECTED_I8_WEIGHTS, "number of i8 weights");
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V DiT container (no GPU): passes ditContract with the input shapes of the 2.2 generation descriptor, form is ti2v",
  async () => {
    const opened = await openTi2vTransformer();
    const prepared = prepareContainer(opened, COMPONENT);
    const contract = ditContract(
      prepared,
      await readRopeBase(opened),
      WAN22_GEOMETRY,
      WAN22_TI2V_GENERATION.ditInputForm,
      "wan.Ti2vPipeline",
    );
    assertEquals(contract.form, "ti2v");
  },
);

/**
 * Checks per case that the host functions (patchify, the RoPE tables from the asset `rope_base`, the 2
 * `timesteps_proj`, the condition mask inside the tests) reproduce the golden graph inputs (the first item at the top
 * of the file).
 */
const assertHostInputs = async (cases: readonly Ti2vCase[]): Promise<void> => {
  const base = await readRopeBase(await openTi2vTransformer());
  for (const { name, form } of cases) {
    const golden = await loadTi2vGolden(name);
    const { io, reference, latentShape } = golden;
    const tokens = patchifyLatents(
      floatsOf(reference, "latents", name),
      latentShape,
      WAN22_GEOMETRY,
    );
    assertEquals(
      firstBitMismatch(tokens, floatsOf(io, "input.tokens", name)),
      -1,
      `${name}: patchify`,
    );
    const grid = wanTokenGrid(latentShape, WAN22_GEOMETRY);
    const tables = wanRopeTables(base, grid);
    assertEquals(
      firstBitMismatch(tables.cos, floatsOf(io, "input.rope_cos", name)),
      -1,
      `${name}: rope_cos`,
    );
    assertEquals(
      firstBitMismatch(tables.sin, floatsOf(io, "input.rope_sin", name)),
      -1,
      `${name}: rope_sin`,
    );

    const { timestep, conditionTimestep: condition } = golden;
    assertEquals(condition, form === "i2v" ? 0 : timestep, `${name}: condition-side timestep`);
    for (const [key, value] of [
      ["input.timesteps_proj", timestep],
      ["input.timesteps_proj_condition", condition],
    ] as const) {
      const expected = floatsOf(io, key, name);
      const diff = maxAbsDiff(timestepsProj(value, expected.length), expected);
      assert(diff <= TIMESTEPS_PROJ_ATOL, `${name}: ${key} difference ${diff}`);
    }

    // Fail unless the reader of the condition mask's storage (bool as u32 0 / 1) is U32.
    const mask = storedConditionMask(golden, name);
    const conditioned = form === "i2v" ? grid.rows * grid.cols : 0;
    assertEquals(mask.length, grid.count, `${name}: condition mask length`);
    assertEquals(
      mask.findIndex((bit, index) => bit !== (index < conditioned ? 1 : 0)),
      -1,
      `${name}: condition mask (${form === "i2v" ? "first latent frame" : "all 0"})`,
    );
    assertEquals(
      firstBitMismatch(conditionMask(grid, form), mask),
      -1,
      `${name}: the host condition mask inside the tests`,
    );
  }
};

test.skipIf(!ASSETS_AVAILABLE)(
  "Wan2.2 TI2V DiT host: patchify, the RoPE tables of rope_base, the 2 timesteps_proj and the condition mask match the golden inputs",
  () => assertHostInputs(CASES),
);

test.skipIf(!FULL_ANY_PRESENT && !FETCHED)(
  "Wan2.2 TI2V DiT full-size goldens: the goldens of every case (2 shapes, 6 band + 8 accept) are present",
  () => {
    assertEquals(
      fullExpectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `missing full-size goldens (${MISSING_GOLDENS})`,
    );
  },
);

test.skipIf(!FULL_AVAILABLE)(
  "Wan2.2 TI2V DiT full-size host (S = 8,190 / 7,920, condition frames P = 390 / 880): patchify, RoPE tables, " +
    "the 2 timesteps_proj and the condition mask match the golden inputs",
  () => assertHostInputs(FULL_CASES),
);

// Wan2.2 TI2V-5B の DiT（i8 系列 `wan2.2-ti2v-5b-i8-dyn`）の GPU 不要の突き合わせ（ADR 0121 段 1 / 段 2）。
//
// 見るのは 2 つ:
//
// - **ホスト関数 × golden の入力**: 2.1 のホスト関数（`src/wan/dit-*.ts`）が潜在 48 チャネルの 5B でも
//   golden のグラフ入力を再現すること（patchify・資産 `rope_base` からの RoPE 表はビット一致・
//   `timesteps_proj` は atol）。I2V 対応の 2 入力（`timesteps_proj_condition`・`condition_mask`）は、T2V の
//   形だけ `src/wan/dit-loop.ts` が組み（段 6）、I2V の形の TS の実装はまだ無い（段 9）ので、ここで golden の
//   規約を固定する: 条件側の時刻は `reference` の `condition_timestep`（I2V は 0・T2V は生成側と
//   同じ）、条件マスクは I2V なら先頭の潜在フレームの H'·W' トークンが 1（u32）・T2V は全て 0。
//   テストの中のホストの条件マスク（`helpers/wan-ti2v-dit.ts` の `conditionMask` — 実 GPU の r 門の
//   製品の経路の入力と故障注入が使う）も golden とビット一致させる。
//   S = 192 の golden（段 1）と、実寸の 2 つの形の golden（段 2 — 潜在 `[48,21,30,52]` / `[48,9,44,80]`・
//   P = 390 / 880）の両方で見る。
// - **runtime から見た容器の宣言**: 入力 7 本（条件マスクは bool `[1,S,1]`）・linear のノード 310 本
//   （時刻の MLP の 3 本を M = 1 で 2 回 — 決定 2 の dispatch の本数）・`where` 182 本（30 層 × 6 + head 2）・
//   i8 の重み 307 本。recipe の IR の検査（`wan/ti2v_export_dit.py`）と同じ数を、runtime の読み口で数える。
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
  floatsOf,
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
import {
  WAN_TI2V_ASSEMBLE_COMMAND as ASSEMBLE,
  WAN_TI2V_DIST_ROOT as DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import { TEST_ASSETS, testAssetsFetched } from "@karume/test-support/test-assets";

/**
 * `timesteps_proj` の TS 実装と上流（torch CPU f32）の許容差（絶対）。
 *
 * 2.1 と同じ関数・同じ幅（`freq_dim` 256）なので、実測（全 1,001 通りの最悪 3.0517e-5 — `exp` の 1 ULP）と
 * 帯の根拠は `wan_dit_host_test.ts` の `TIMESTEPS_PROJ_ATOL` のまま。
 */
const TIMESTEPS_PROJ_ATOL = 1.5e-4;

/** 決定 2 / 3 の本数（5B — 30 層）。 */
const EXPECTED_LINEAR_NODES = 310;
const EXPECTED_WHERE_NODES = 30 * 6 + 2;
const EXPECTED_I8_WEIGHTS = 307;

/** グラフ入力の宣言（`wan/ti2v_export_dit.py` の `INPUT_NAMES` と形）。 */
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
  assertEquals(actual.length, expected.length, "要素数");
  let worst = 0;
  for (let index = 0; index < actual.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  return worst;
};

/** The container's graph (named by the manifest's weights key — the NOTE at the top of the file). */
const containerGraphOf = (opened: OpenedContainer) => {
  assertEquals(Object.keys(opened.graphs), [COMPONENT], "容器のグラフ名");
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
  "Wan2.2 TI2V DiT 容器（GPU 不要）: 入力 7 本・条件マスクは bool [1,S,1]・linear 310・where 182・i8 307",
  async () => {
    const graph = containerGraphOf(await openTi2vTransformer());
    const { declaration } = graph;
    assertEquals(
      declaration.inputs.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
      EXPECTED_INPUTS.map(({ name, dtype, shape }) => ({ name, dtype, shape: [...shape] })),
    );
    assertEquals(declaration.symbols, ["S"]);
    const count = (op: string): number => declaration.nodes.filter((node) => node.op === op).length;
    assertEquals(count("linear"), EXPECTED_LINEAR_NODES, "linear のノード（dispatch）の本数");
    assertEquals(count("where"), EXPECTED_WHERE_NODES, "where のノードの本数");
    const i8 = [...graph.supplies.values()].filter(
      ({ encoding }) => codecLayout(encoding.codec) === "i8",
    );
    assertEquals(i8.length, EXPECTED_I8_WEIGHTS, "i8 の重みの本数");
  },
);

test.skipIf(!MIRROR_PRESENT)(
  "Wan2.2 TI2V DiT 容器（GPU 不要）: 2.2 の世代の記述子の入力の形で ditContract を通り、form は ti2v",
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
 * ホスト関数（patchify・資産 `rope_base` からの RoPE 表・`timesteps_proj` 2 本・テストの中の条件マスク）が
 * golden のグラフ入力を再現することを、ケースごとに見る（ファイル冒頭の 1 つ目）。
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
    assertEquals(condition, form === "i2v" ? 0 : timestep, `${name}: 条件側の timestep`);
    for (const [key, value] of [
      ["input.timesteps_proj", timestep],
      ["input.timesteps_proj_condition", condition],
    ] as const) {
      const expected = floatsOf(io, key, name);
      const diff = maxAbsDiff(timestepsProj(value, expected.length), expected);
      assert(diff <= TIMESTEPS_PROJ_ATOL, `${name}: ${key} の差 ${diff}`);
    }

    // 条件マスクの格納（bool は u32 の 0 / 1）は読み口が U32 でなければ落とす。
    const mask = storedConditionMask(golden, name);
    const conditioned = form === "i2v" ? grid.rows * grid.cols : 0;
    assertEquals(mask.length, grid.count, `${name}: 条件マスクの長さ`);
    assertEquals(
      mask.findIndex((bit, index) => bit !== (index < conditioned ? 1 : 0)),
      -1,
      `${name}: 条件マスク（${form === "i2v" ? "先頭の潜在フレーム" : "全て 0"}）`,
    );
    assertEquals(
      firstBitMismatch(conditionMask(grid, form), mask),
      -1,
      `${name}: テストの中のホストの条件マスク`,
    );
  }
};

test.skipIf(!ASSETS_AVAILABLE)(
  "Wan2.2 TI2V DiT ホスト: patchify・rope_base の RoPE 表・timesteps_proj 2 本・条件マスクが golden の入力と一致する",
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
  "Wan2.2 TI2V DiT 実寸のホスト（S = 8,190 / 7,920・条件フレーム P = 390 / 880）: patchify・RoPE 表・" +
    "timesteps_proj 2 本・条件マスクが golden の入力と一致する",
  () => assertHostInputs(FULL_CASES),
);

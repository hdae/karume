// Wan2.2 TI2V-5B の DiT（i8 系列 `wan2.2-ti2v-5b-i8-dyn`）の GPU 不要の突き合わせ（ADR 0121 段 1）。
//
// 見るのは 2 つ:
//
// - **ホスト関数 × golden の入力**: 2.1 のホスト関数（`src/wan/dit-*.ts`）が潜在 48 チャネルの 5B でも
//   golden のグラフ入力を再現すること（patchify・資産 `rope_base` からの RoPE 表はビット一致・
//   `timesteps_proj` は atol）。I2V 対応の 2 入力（`timesteps_proj_condition`・`condition_mask`）は TS の
//   実装がまだ無い（段 6）ので、ここで golden の規約を固定する: 条件側の時刻は `reference` の
//   `condition_timestep`（I2V は 0・T2V は生成側と同じ）、条件マスクは I2V なら先頭の潜在フレームの
//   H'·W' トークンが 1（u32）・T2V は全て 0。
// - **runtime から見た容器の宣言**: 入力 7 本（条件マスクは bool `[1,S,1]`）・linear のノード 310 本
//   （時刻の MLP の 3 本を M = 1 で 2 回 — 決定 2 の dispatch の本数）・`where` 182 本（30 層 × 6 + head 2）・
//   i8 の重み 307 本。recipe の IR の検査（`wan/ti2v_export_dit.py`）と同じ数を、runtime の読み口で数える。
//
// 資産が無い環境は生成コマンド付きで**明示 SKIP**（ADR 0005）。資産が**一部だけ**ある環境は SKIP ではなく
// FAIL にする（下の完全性テスト）。GPU は使わない（実 GPU の r 門は段 2）。
//
// NOTE: 容器のグラフ名は綴らず、系列のグラフ名の表（`runtime/tests/helpers/series-graphs.ts` — 門番
// `assets_gate_test.ts` と同じ正本）から引く。

import { assert, assertEquals } from "@std/assert";
import {
  codecLayout,
  type OpenedContainer,
  parseSafetensors,
  type SafetensorsFile,
} from "@karume/runtime";
import { patchifyLatents, type WanPatchGeometry, wanTokenGrid } from "../src/wan/dit-tokens.ts";
import { parseWanRopeBase, type WanRopeBase, wanRopeTables } from "../src/wan/dit-rope.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";

const SERIES_NAME = "wan2.2-ti2v-5b-i8-dyn";
const COMPONENT = "transformer";
const SERIES_DIR = new URL(`../../../outputs/series/${SERIES_NAME}/${COMPONENT}/`, import.meta.url);
const MODEL_FILE = "model.krm";
/** 資産の名前（`wan/export_dit.py` の `ROPE_BASE_ASSET` — 5B も同じ席）。 */
const ROPE_BASE_ASSET = "rope_base";

const GENERATE =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.ti2v_export_dit write";

/** Wan2.2 TI2V-5B の patch（`(1,2,2)`・潜在 48 チャネル — transformer の config）。 */
const WAN22_GEOMETRY: WanPatchGeometry = {
  channels: 48,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

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

type Form = "t2v" | "i2v";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する**（生成を一部だけ流した環境でテストが黙って
 * 消える形にしない）。正本は `wan/ti2v_export_dit.py` の `CASES`。
 */
const CASES: readonly { readonly name: string; readonly form: Form }[] = [
  { name: "band-t2v-s00192-t0999", form: "t2v" },
  { name: "band-t2v-s00192-t0500", form: "t2v" },
  { name: "band-t2v-s00192-t0250", form: "t2v" },
  { name: "band-i2v-s00192-t0999", form: "i2v" },
  { name: "band-i2v-s00192-t0750", form: "i2v" },
  { name: "band-i2v-s00192-t0113", form: "i2v" },
  { name: "accept-t2v-s00192-t0600", form: "t2v" },
  { name: "accept-t2v-s00192-t0030", form: "t2v" },
  { name: "accept-i2v-s00192-t0400", form: "i2v" },
  { name: "accept-i2v-s00192-t0900", form: "i2v" },
];

/** ファイルの有無（NotFound 以外は伝播させる — 権限エラーを「資産が無い」に読み替えない）。 */
const filePresent = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const caseFiles = (name: string): readonly URL[] => [
  new URL(`io.${name}.safetensors`, SERIES_DIR),
  new URL(`reference.${name}.safetensors`, SERIES_DIR),
];

const expectedFiles = CASES.flatMap(({ name }) => caseFiles(name));
const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, SERIES_DIR));
const ANY_PRESENT = MODEL_PRESENT || expectedFiles.some(filePresent);
const ASSETS_AVAILABLE = MODEL_PRESENT && expectedFiles.every(filePresent);

if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_DIR.pathname} に Wan2.2 TI2V の DiT の容器と golden が揃っていないため、` +
      `GPU 不要の突き合わせを SKIP する（重み 4.7GB につきリポジトリ管理外）。生成: ${GENERATE}`,
  );
}

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

const viewOf = (file: SafetensorsFile, key: string, where: string) => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`${where}: '${key}' が無い`);
  return view;
};

const floatsOf = (file: SafetensorsFile, key: string, where: string): Float32Array => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "F32") throw new Error(`${where}: '${key}' が ${view.dtype}`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

const scalarOf = (file: SafetensorsFile, key: string, where: string): number => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "I32" || view.byteLength !== 4) throw new Error(`${where}: '${key}' が想定外`);
  return new Int32Array(file.buffer, view.byteOffset, 1)[0];
};

/** 最初にビットが割れる要素の添字（一致なら -1）。長さが違えば 0。 */
const firstBitMismatch = (actual: Float32Array, expected: Float32Array): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

const maxAbsDiff = (actual: Float32Array, expected: Float32Array): number => {
  assertEquals(actual.length, expected.length, "要素数");
  let worst = 0;
  for (let index = 0; index < actual.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  return worst;
};

const readRopeBase = async (opened: OpenedContainer): Promise<WanRopeBase> => {
  const reader = opened.asset(ROPE_BASE_ASSET);
  assertEquals(reader.role, "rope-base", `資産 '${ROPE_BASE_ASSET}' の役割`);
  const bytes = await reader.read(0, reader.length);
  return parseWanRopeBase(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
};

/**
 * 容器のグラフ（名前は表から引く — ファイル冒頭の NOTE）。**使うときに引く** — 表に行が無いときにモジュール
 * の評価で落ちて、資産の無い機の SKIP まで巻き込まない。
 */
const seriesGraphOf = (opened: OpenedContainer) => {
  const name = seriesGraph(SERIES_NAME, COMPONENT);
  assertEquals(Object.keys(opened.graphs), [name], "容器のグラフ名");
  return opened.graphs[name];
};

Deno.test({
  name:
    "Wan2.2 TI2V DiT 資産: 容器と全ケース（S = 192・T2V 5 本 + I2V 5 本）の golden が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。何か 1 つでもあれば欠けは FAIL。
  ignore: !ANY_PRESENT,
  fn: () => {
    assert(MODEL_PRESENT, `${SERIES_DIR.pathname}${MODEL_FILE} が無い`);
    assertEquals(
      expectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `golden の欠け（生成: ${GENERATE}）`,
    );
  },
});

Deno.test({
  name:
    "Wan2.2 TI2V DiT 容器（GPU 不要）: 入力 7 本・条件マスクは bool [1,S,1]・linear 310・where 182・i8 307",
  ignore: !MODEL_PRESENT,
  fn: async () => {
    const graph = seriesGraphOf(await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)));
    const { declaration } = graph;
    assertEquals(
      declaration.inputs.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
      EXPECTED_INPUTS.map(({ name, dtype, shape }) => ({ name, dtype, shape: [...shape] })),
    );
    assertEquals(declaration.symbols, ["S"]);
    const count = (op: string): number => declaration.nodes.filter((node) => node.op === op).length;
    assertEquals(count("linear"), EXPECTED_LINEAR_NODES, "linear のノード（dispatch）の本数");
    assertEquals(count("where"), EXPECTED_WHERE_NODES, "where のノードの本数");
    const i8 = [...graph.supplies.values()].filter(({ encoding }) =>
      codecLayout(encoding.codec) === "i8"
    );
    assertEquals(i8.length, EXPECTED_I8_WEIGHTS, "i8 の重みの本数");
  },
});

Deno.test({
  name:
    "Wan2.2 TI2V DiT ホスト: patchify・rope_base の RoPE 表・timesteps_proj 2 本・条件マスクが golden の入力と一致する",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const base = await readRopeBase(await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)));
    for (const { name, form } of CASES) {
      const [ioUrl, referenceUrl] = caseFiles(name);
      const io = parseSafetensors(await readBuffer(ioUrl));
      const reference = parseSafetensors(await readBuffer(referenceUrl));
      const latents = viewOf(reference, "latents", name);
      assertEquals(latents.shape.length, 5, `${name}: latents の rank`);
      // ホストの潜在はバッチ軸を持たない `[C,F,H,W]`（`src/wan/dit-tokens.ts`）。
      const latentShape = latents.shape.slice(1);
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

      const timestep = scalarOf(reference, "timestep", name);
      const condition = scalarOf(reference, "condition_timestep", name);
      assertEquals(condition, form === "i2v" ? 0 : timestep, `${name}: 条件側の timestep`);
      for (
        const [key, value] of [
          ["input.timesteps_proj", timestep],
          ["input.timesteps_proj_condition", condition],
        ] as const
      ) {
        const expected = floatsOf(io, key, name);
        const diff = maxAbsDiff(timestepsProj(value, expected.length), expected);
        assert(diff <= TIMESTEPS_PROJ_ATOL, `${name}: ${key} の差 ${diff}`);
      }

      const maskView = viewOf(io, "input.condition_mask", name);
      assertEquals(maskView.dtype, "U32", `${name}: 条件マスクの格納（bool は u32 の 0 / 1）`);
      const mask = new Uint32Array(io.buffer, maskView.byteOffset, maskView.byteLength / 4);
      const conditioned = form === "i2v" ? grid.rows * grid.cols : 0;
      assertEquals(mask.length, grid.count, `${name}: 条件マスクの長さ`);
      assertEquals(
        mask.findIndex((bit, index) => bit !== (index < conditioned ? 1 : 0)),
        -1,
        `${name}: 条件マスク（${form === "i2v" ? "先頭の潜在フレーム" : "全て 0"}）`,
      );
    }
  },
});

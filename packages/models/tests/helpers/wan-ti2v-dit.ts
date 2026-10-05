/**
 * Wan2.2 TI2V-5B の DiT（i8 系列 `wan2.2-ti2v-5b-i8-dyn`）のテストが共有するもの（ADR 0121 段 1 / 段 2）:
 * 系列の置き場・golden のケースの表・golden の読み口・ホスト側の条件マスク。
 *
 * GPU 不要の突き合わせ（`wan_ti2v_dit_host_test.ts`）と実 GPU の r 門（`e2e_wan_ti2v_dit_test.ts`）が同じ
 * ケースの表を読む — 表を 2 か所で育てると、片方だけに足したケースが黙って片方の門を外れる。
 *
 * ケースの正本は `tools/export-recipes/wan/ti2v_export_dit.py` の `CASES`（S = 192 — 段 1 の `write`）と
 * `FULL_CASES`（実寸 — 段 2 の `write-full`）。**列挙結果ではなくここで固定する**（生成を一部だけ流した環境で
 * テストが黙って消える形にしない）。
 *
 * ## ホスト側の条件マスク（{@link conditionMask}）
 *
 * T2V の条件入力（全て偽の条件マスクと、生成側と同じ条件側の時刻）は `src/wan/dit-loop.ts` が組む（ADR 0121
 * 段 6）。I2V の条件マスクを組む関数は `src/wan/` にまだ無い（段 9）ので、ここが I2V の形も持つ。規約は recipe の
 * `dit_patch.dit_condition_mask` と同じで、I2V は先頭の潜在フレームの P = H'·W' トークンが 1（bool は u32 の
 * 0 / 1）、T2V は全て 0。golden の `input.condition_mask` とのビット一致はホストテストが縛る。
 */

import { assertEquals } from "@std/assert";
import { type OpenedContainer, parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import type { WanPatchGeometry, WanTokenGrid } from "../../src/wan/dit-tokens.ts";
import { parseWanRopeBase, type WanRopeBase } from "../../src/wan/dit-rope.ts";

export const WAN_TI2V_SERIES_NAME = "wan2.2-ti2v-5b-i8-dyn";
export const WAN_TI2V_COMPONENT = "transformer";
export const WAN_TI2V_SERIES_DIR = new URL(
  `../../../../outputs/series/${WAN_TI2V_SERIES_NAME}/${WAN_TI2V_COMPONENT}/`,
  import.meta.url,
);
export const WAN_TI2V_MODEL_FILE = "model.krm";
/** 資産の名前（`wan/export_dit.py` の `ROPE_BASE_ASSET` — 5B も同じ席）。 */
const ROPE_BASE_ASSET = "rope_base";

const TI2V_EXPORT =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.ti2v_export_dit";
/** 容器と S = 192 の golden の生成（段 1）。 */
export const WAN_TI2V_GENERATE = `${TI2V_EXPORT} write`;
/** 実寸の golden を据えた系列へ足す（段 2 — CPU で数時間・途中から再開できる）。 */
export const WAN_TI2V_GENERATE_FULL = `${TI2V_EXPORT} write-full`;

/** Wan2.2 TI2V-5B の patch（`(1,2,2)`・潜在 48 チャネル — transformer の config）。 */
export const WAN22_GEOMETRY: WanPatchGeometry = {
  channels: 48,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

export type Ti2vForm = "t2v" | "i2v";

/** golden の 1 ケース（役割は帯の決定 / 受入れ — 帯の決定と受入れの独立は recipe の表の MUST）。 */
export type Ti2vCase = {
  readonly name: string;
  readonly role: "band" | "accept" | "full-band" | "full-accept";
  readonly form: Ti2vForm;
};

/** S = 192 の golden（段 1 — 決定用 T2V 3 + I2V 3・受入れ T2V 2 + I2V 2）。 */
export const WAN_TI2V_CASES: readonly Ti2vCase[] = [
  { name: "band-t2v-s00192-t0999", role: "band", form: "t2v" },
  { name: "band-t2v-s00192-t0500", role: "band", form: "t2v" },
  { name: "band-t2v-s00192-t0250", role: "band", form: "t2v" },
  { name: "band-i2v-s00192-t0999", role: "band", form: "i2v" },
  { name: "band-i2v-s00192-t0750", role: "band", form: "i2v" },
  { name: "band-i2v-s00192-t0113", role: "band", form: "i2v" },
  { name: "accept-t2v-s00192-t0600", role: "accept", form: "t2v" },
  { name: "accept-t2v-s00192-t0030", role: "accept", form: "t2v" },
  { name: "accept-i2v-s00192-t0400", role: "accept", form: "i2v" },
  { name: "accept-i2v-s00192-t0900", role: "accept", form: "i2v" },
];

/**
 * 実寸の S（ADR 0121 追記「裁定 1 の確定」の 2 つの形 — 832×480・81 フレーム = 潜在 `[48,21,30,52]` の 8,190 と
 * 1280×704・33 フレーム = `[48,9,44,80]` の 7,920）。
 */
export type Ti2vFullTokens = 8190 | 7920;

/**
 * 実寸の golden（段 2 — 形ごとに決定用 3 本〈2 つの形で T2V 3 + I2V 3〉・受入れ T2V 2 + I2V 2）。並びは確保の
 * 大きい S = 8,190 が先（GPU テストはこの順で回す）。
 */
export const WAN_TI2V_FULL_CASES: readonly (Ti2vCase & { readonly tokens: Ti2vFullTokens })[] = [
  { name: "full-band-t2v-s08190-t0999", role: "full-band", form: "t2v", tokens: 8190 },
  { name: "full-band-i2v-s08190-t0500", role: "full-band", form: "i2v", tokens: 8190 },
  { name: "full-band-t2v-s08190-t0113", role: "full-band", form: "t2v", tokens: 8190 },
  { name: "full-accept-t2v-s08190-t0999", role: "full-accept", form: "t2v", tokens: 8190 },
  { name: "full-accept-t2v-s08190-t0600", role: "full-accept", form: "t2v", tokens: 8190 },
  { name: "full-accept-i2v-s08190-t0900", role: "full-accept", form: "i2v", tokens: 8190 },
  { name: "full-accept-i2v-s08190-t0400", role: "full-accept", form: "i2v", tokens: 8190 },
  { name: "full-band-i2v-s07920-t0999", role: "full-band", form: "i2v", tokens: 7920 },
  { name: "full-band-t2v-s07920-t0750", role: "full-band", form: "t2v", tokens: 7920 },
  { name: "full-band-i2v-s07920-t0250", role: "full-band", form: "i2v", tokens: 7920 },
  { name: "full-accept-t2v-s07920-t0999", role: "full-accept", form: "t2v", tokens: 7920 },
  { name: "full-accept-t2v-s07920-t0300", role: "full-accept", form: "t2v", tokens: 7920 },
  { name: "full-accept-i2v-s07920-t0800", role: "full-accept", form: "i2v", tokens: 7920 },
  { name: "full-accept-i2v-s07920-t0050", role: "full-accept", form: "i2v", tokens: 7920 },
];

/** 受入れ（帯の決定に使わない側）か。 */
export const isAccept = ({ role }: Ti2vCase): boolean =>
  role === "accept" || role === "full-accept";

/** ファイルの有無（NotFound 以外は伝播させる — 権限エラーを「資産が無い」に読み替えない）。 */
export const filePresent = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/** golden 2 本（`io.<case>` と `reference.<case>`）。 */
export const caseFiles = (name: string): readonly URL[] => [
  new URL(`io.${name}.safetensors`, WAN_TI2V_SERIES_DIR),
  new URL(`reference.${name}.safetensors`, WAN_TI2V_SERIES_DIR),
];

export const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

export const viewOf = (file: SafetensorsFile, key: string, where: string) => {
  const view = file.tensors.get(key);
  if (view === undefined) throw new Error(`${where}: '${key}' が無い`);
  return view;
};

export const floatsOf = (
  file: SafetensorsFile,
  key: string,
  where: string,
): Float32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "F32") throw new Error(`${where}: '${key}' が ${view.dtype}`);
  return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

export const scalarOf = (file: SafetensorsFile, key: string, where: string): number => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "I32" || view.byteLength !== 4) throw new Error(`${where}: '${key}' が想定外`);
  return new Int32Array(file.buffer, view.byteOffset, 1)[0];
};

/** 最初にビットが割れる要素の添字（一致なら -1）。長さが違えば 0。 */
export const firstBitMismatch = (
  actual: Float32Array | Uint32Array,
  expected: Float32Array | Uint32Array,
): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
};

/** 1 ケースぶんの golden（グラフ入力・上流の潜在形と時刻 2 本と参照）。 */
export type Ti2vGolden = {
  readonly io: SafetensorsFile;
  readonly reference: SafetensorsFile;
  /** ホストの潜在の形 `[C,F,H,W]`（バッチ軸を持たない — `src/wan/dit-tokens.ts`）。 */
  readonly latentShape: readonly number[];
  /** 生成側の timestep。 */
  readonly timestep: number;
  /** 条件側の timestep（I2V は 0・T2V は生成側と同じ — `reference.<case>` の `condition_timestep`）。 */
  readonly conditionTimestep: number;
};

export const loadTi2vGolden = async (name: string): Promise<Ti2vGolden> => {
  const [ioUrl, referenceUrl] = caseFiles(name);
  const io = parseSafetensors(await readBuffer(ioUrl));
  const reference = parseSafetensors(await readBuffer(referenceUrl));
  const latents = viewOf(reference, "latents", name);
  assertEquals(latents.shape.length, 5, `${name}: latents の rank`);
  assertEquals(latents.shape[0], 1, `${name}: latents のバッチ`);
  return {
    io,
    reference,
    latentShape: latents.shape.slice(1),
    timestep: scalarOf(reference, "timestep", name),
    conditionTimestep: scalarOf(reference, "condition_timestep", name),
  };
};

/** golden の `input.condition_mask`（bool は u32 の 0 / 1 で格納 — `docs/ir-v2.md`）。 */
export const storedConditionMask = (
  golden: Ti2vGolden,
  where: string,
): Uint32Array<ArrayBuffer> => {
  const view = viewOf(golden.io, "input.condition_mask", where);
  if (view.dtype !== "U32") {
    throw new Error(`${where}: 条件マスクの格納が ${view.dtype}（U32 のはず）`);
  }
  return new Uint32Array(golden.io.buffer, view.byteOffset, view.byteLength / 4);
};

/**
 * ホスト側の条件マスク `[1,S,1]`（u32 の 0 / 1 — モジュール doc）。I2V は先頭の潜在フレームの H'·W' トークンが
 * 1。トークン添字は `(f·H' + h)·W' + w`（`src/wan/dit-tokens.ts`）なので先頭に連続して並ぶ。
 */
export const conditionMask = (grid: WanTokenGrid, form: Ti2vForm): Uint32Array<ArrayBuffer> => {
  const mask = new Uint32Array(grid.count);
  if (form === "i2v") mask.fill(1, 0, grid.rows * grid.cols);
  return mask;
};

/** 容器の資産 `rope_base`（役割 `rope-base`）を読んで素表にする。 */
export const readRopeBase = async (opened: OpenedContainer): Promise<WanRopeBase> => {
  const reader = opened.asset(ROPE_BASE_ASSET);
  assertEquals(reader.role, "rope-base", `資産 '${ROPE_BASE_ASSET}' の役割`);
  const bytes = await reader.read(0, reader.length);
  return parseWanRopeBase(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
  );
};

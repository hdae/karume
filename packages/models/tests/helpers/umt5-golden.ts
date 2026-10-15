/**
 * umT5-XXL の i8 系列の golden（`outputs/series/umt5-xxl-i8-dyn/text_encoder/reference.<case>.safetensors` — recipe
 * `wan/umt5_reference.py`）の読み口と、GPU の出力をそれに突き合わせる判定（正規化比 r と帯 — ADR 0119 決定 8 ①）。
 *
 * 2 本の e2e が同じ golden を同じ判定で読む: Wan の umT5 の移植の門（`e2e_wan_umt5_test.ts` — 系列の容器を runtime で
 * 直接回す）と、単体の `Umt5Encoder` の通し（`e2e_umt5_encoder_test.ts` — 配布形 `karume-umt5-xxl` を
 * `fromPretrained` で組む）。帯・指標・ケースの正本はここ 1 本（2 本へ写すと、片方だけが動く）。
 *
 * golden の中身（1 ケース 1 本）:
 *
 * - `input_ids` I32 `[L]` / `relative_position_buckets` I32 `[L,L]` — Python の経路（上流の `prompt_clean` →
 *   transformers 5.14.1・上流のバケットの式）の入力
 * - `output.f64` F32 `[1,L,4096]` — 活性も f64 で回した参照を f32 へ丸めた値（誤差の基準）
 * - `output.f32` F32 `[1,L,4096]` — CPU f32 の参照（上流の eager とビット一致する書き下し — 正規化の分母）
 * - `output.unquantized.f64` / `output.unquantized.f32` F32 `[1,L,4096]`（**受入れだけ**）— 量子化しない重みで回した
 *   同じ書き下し（品質の記録の基準 — 門には使わない）
 * - メタ（キー 1 つの JSON）— プロンプトの原文と前処理の後の文字列・L・役割・容器の part 0 の sha256
 *
 * 副作用ゼロ（定数と関数だけ — ファイルは呼ばれたときに読む）。
 */

import { assert, assertEquals } from "@std/assert";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";

/**
 * 正規化比 r の帯（ADR 0119 決定 8 ① — 導き方は ADR 0118 決定 8 と同じ）。決定用 6 本（`band-*`）の最悪 r × 5 を
 * 有効数字 2 桁へ切り上げた値で、受入れ 4 本（`accept-*`）は帯の決定に使わない。
 *
 * `undefined` = 未導出: 各ケースは r と故障注入の r を出して**赤で止まり**、最後の step が帯の候補を出して赤に
 * なる（通る値を仮置きして検出力の無い門を作らない — DiT の `I8_NORMALIZED_BAND` と同じ扱い）。GPU の実測の後に、
 * 実測の表と一緒にここへ書く。
 *
 * MUST: 受入れの結果を見てこの値も決定用のケースも指標も変えない（ケースの正本は recipe
 * `wan/umt5_reference.py` の `band_cases` / `accept_cases`）。受入れが帯を外れたら、帯を広げずに原因を調べる。
 * 導出（2026-10-03・B570・通常の device・CPU 層逐次 f64 参照）:
 *
 * | 決定用（L）  | r     |   | 受入れ          | r    |
 * | ------------ | ----- | - | --------------- | ---- |
 * | band-l0008   | 8.25  |   | boxing-cats     | 2.78 |
 * | band-l0022   | 5.92  |   | ferret          | 8.87 |
 * | band-l0036   | 2.20  |   | cat-dog-baking  | 4.36 |
 * | band-l0163   | 4.93  |   | negative        | 3.74 |
 * | band-l0327   | 0.904 |   |                 |      |
 * | band-l0488   | 15.8  |   |                 |      |
 *
 * 最悪 15.8 × 5 = 79（有効数字 2 桁へ切り上げ）。故障注入: 層 0 / 1 の表の取り違え r 5.0e+5〜8.0e+5・wo.12 の scale × 2 は
 * 5.0e+5〜9.7e+5（帯の 4 桁上）。scale × 1.0001（記録だけ）は 22.6 / 72.6 / 80.2 / 40.0 で帯の縁（r は δ に比例 — 1e-4 の
 * scale のずれはこの門では区別できない。微妙な故障の床は持たない）。
 */
export const UMT5_NORMALIZED_BAND: number | undefined = 79;

export const SERIES_NAME = "umt5-xxl-i8-dyn";
export const COMPONENT = "text_encoder";
export const SERIES_DIR = new URL(
  `../../../../outputs/series/${SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);

export const GENERATE =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_export reference";

/** golden のキー（recipe `wan/umt5_reference.py` と同じ綴り）。 */
export const INPUT_IDS_KEY = "input_ids";
export const BUCKETS_KEY = "relative_position_buckets";
const OUTPUT_F64_KEY = "output.f64";
const OUTPUT_F32_KEY = "output.f32";
const OUTPUT_UNQUANTIZED_F64_KEY = "output.unquantized.f64";
const OUTPUT_UNQUANTIZED_F32_KEY = "output.unquantized.f32";
const METADATA_KEY = "karume.wan.umt5_reference";
/** /2 = 受入れに量子化なしの参照を足した形（受入れでは {@link OUTPUT_UNQUANTIZED_F32_KEY} などが必須）。 */
const REFERENCE_FORMAT = "karume-wan-umt5-reference/2";

export type CaseRole = "band" | "accept";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する**（生成を一部だけ流した環境でテストが黙って
 * 消える形にしない）。正本は recipe `wan/umt5_reference.py`（決定用は 10a のパリティ fixture の受理した乱択から
 * seed 20261003 で選んだ単体 3 本〈L = 8 / 22 / 36〉と、乱択を連ねた合成 3 本〈L = 163 / 327 / 488〉・受入れは固定
 * 4 プロンプト）。`embedding` は bf16 の事前計算資産のテンソル名（受入れだけ）。
 */
export const CASES: readonly {
  readonly name: string;
  readonly role: CaseRole;
  readonly embedding?: string;
}[] = [
  { name: "band-l0008", role: "band" },
  { name: "band-l0022", role: "band" },
  { name: "band-l0036", role: "band" },
  { name: "band-l0163", role: "band" },
  { name: "band-l0327", role: "band" },
  { name: "band-l0488", role: "band" },
  { name: "accept-boxing-cats", role: "accept", embedding: "boxing-cats" },
  { name: "accept-ferret", role: "accept", embedding: "ferret" },
  { name: "accept-cat-dog-baking", role: "accept", embedding: "cat-dog-baking" },
  { name: "accept-negative", role: "accept", embedding: "negative" },
];

export const goldenUrl = (name: string): URL =>
  new URL(`reference.${name}.safetensors`, SERIES_DIR);

/** ファイルの有無（NotFound 以外は伝播させる — 権限エラーを「資産が無い」に読み替えない）。 */
export const filePresent = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

export const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

const viewOf = (file: SafetensorsFile, key: string, where: string) => {
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

const intsOf = (file: SafetensorsFile, key: string, where: string): Int32Array<ArrayBuffer> => {
  const view = viewOf(file, key, where);
  if (view.dtype !== "I32") throw new Error(`${where}: '${key}' が ${view.dtype}`);
  return new Int32Array(file.buffer, view.byteOffset, view.byteLength / 4);
};

/** golden のメタ（recipe の `golden_metadata` のうち、ここで使う欄）。 */
export type GoldenMeta = {
  readonly role: string;
  readonly prompt: string;
  /**
   * 上流の前処理（`prompt_clean`）の後の文字列 — golden の `input_ids` はこれを transformers 5.14.1 に通した列。
   * 前処理を持たない単体のトークナイザはこの文字列を受ける。
   */
  readonly cleaned: string;
  readonly length: number;
  readonly part0Sha256: string;
};

const parseMeta = (file: SafetensorsFile, where: string): GoldenMeta => {
  const raw = file.metadata.get(METADATA_KEY);
  if (raw === undefined || file.metadata.size !== 1) {
    throw new Error(`${where}: メタはキー ${METADATA_KEY} 1 つのはず`);
  }
  const value: unknown = JSON.parse(raw);
  const isRecord = (item: unknown): item is Record<string, unknown> =>
    typeof item === "object" && item !== null && !Array.isArray(item);
  const record = (item: unknown, label: string): Record<string, unknown> => {
    if (!isRecord(item)) throw new Error(`${where}: メタの ${label} がオブジェクトでない`);
    return item;
  };
  const meta = record(value, "根");
  const container = record(meta["container"], "container");
  const { format, role, prompt, cleaned, length } = meta;
  const part0Sha256 = container["part0Sha256"];
  if (
    format !== REFERENCE_FORMAT || typeof role !== "string" || typeof prompt !== "string" ||
    typeof cleaned !== "string" || typeof length !== "number" || typeof part0Sha256 !== "string"
  ) {
    throw new Error(`${where}: メタの形が ${REFERENCE_FORMAT} でない`);
  }
  return { role, prompt, cleaned, length, part0Sha256 };
};

/** 量子化なしの参照（受入れだけ — 品質の記録の基準）。 */
type UnquantizedReference = {
  readonly expected: Float32Array<ArrayBuffer>;
  readonly reference: Float32Array<ArrayBuffer>;
};

/** 1 ケースぶんの golden。 */
export type Golden = {
  readonly meta: GoldenMeta;
  readonly ids: Int32Array<ArrayBuffer>;
  readonly buckets: Int32Array<ArrayBuffer>;
  readonly expected: Float32Array<ArrayBuffer>;
  readonly reference: Float32Array<ArrayBuffer>;
  readonly outputShape: readonly number[];
  /** 受入れだけが持つ（決定用で在れば・受入れで欠ければ fail loudly — recipe の `_check_unquantized` の鏡）。 */
  readonly unquantized?: UnquantizedReference;
};

export const loadGolden = async (name: string): Promise<Golden> => {
  const file = parseSafetensors(await readBuffer(goldenUrl(name)));
  const meta = parseMeta(file, name);
  const ids = intsOf(file, INPUT_IDS_KEY, name);
  assertEquals(meta.length, ids.length, `${name}: メタの L と input_ids の長さ`);
  const accept = meta.role === "accept";
  assertEquals(
    file.tensors.has(OUTPUT_UNQUANTIZED_F32_KEY),
    accept,
    `${name}: 量子化なしの参照は受入れだけが持つ（役割 ${meta.role}）`,
  );
  return {
    meta,
    ids,
    buckets: intsOf(file, BUCKETS_KEY, name),
    expected: floatsOf(file, OUTPUT_F64_KEY, name),
    reference: floatsOf(file, OUTPUT_F32_KEY, name),
    outputShape: viewOf(file, OUTPUT_F64_KEY, name).shape,
    ...(accept
      ? {
        unquantized: {
          expected: floatsOf(file, OUTPUT_UNQUANTIZED_F64_KEY, name),
          reference: floatsOf(file, OUTPUT_UNQUANTIZED_F32_KEY, name),
        },
      }
      : {}),
  };
};

/** 最初に割れる要素の添字（一致なら -1・長さが違えば 0）。 */
export const firstMismatch = (actual: ArrayLike<number>, expected: ArrayLike<number>): number => {
  if (actual.length !== expected.length) return 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (actual[index] !== expected[index]) return index;
  }
  return -1;
};

export type Difference = {
  readonly maxAbs: number;
  readonly maxRel: number;
  readonly referenceMaxAbs: number;
  readonly nonFinite: number;
};

export const difference = (actual: Float32Array, expected: Float32Array): Difference => {
  assertEquals(actual.length, expected.length, "要素数");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMaxAbs = 0;
  let nonFinite = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const got = actual[index];
    if (!Number.isFinite(got)) nonFinite += 1;
    const error = Math.abs(got - expected[index]);
    const magnitude = Math.abs(expected[index]);
    maxAbs = Math.max(maxAbs, error);
    if (magnitude > 0) maxRel = Math.max(maxRel, error / magnitude);
    referenceMaxAbs = Math.max(referenceMaxAbs, magnitude);
  }
  return { maxAbs, maxRel, referenceMaxAbs, nonFinite };
};

/** 比（最大絶対差 ÷ 参照の最大絶対値 — recipe の `max_ratio` と同じ形）。 */
export const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/**
 * 正規化の分母 = CPU f32 の参照の f64 の参照に対する比。0（f32 の参照が f64 とビット一致）だと割れないので
 * fail loudly。
 */
export const referenceRatioOf = (golden: Golden, name: string): number => {
  const ratio = ratioOf(difference(golden.reference, golden.expected));
  if (!(ratio > 0)) {
    throw new Error(`${name}: CPU f32 の参照の f64 に対する比が ${ratio}（正規化できない）`);
  }
  return ratio;
};

export const formatRatio = (label: string, diff: Difference, referenceRatio: number): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / 参照の最大絶対値 ${
    diff.referenceMaxAbs.toFixed(3)
  }（比 ${ratioOf(diff).toExponential(2)}）・r ${(ratioOf(diff) / referenceRatio).toPrecision(3)}`;

/** 帯の判定（未導出なら r を出して赤 — {@link UMT5_NORMALIZED_BAND}）。 */
export const judgeBand = (
  band: number | undefined,
  id: string,
  ratio: number,
  faults: readonly { readonly label: string; readonly ratio: number }[],
): void => {
  if (band === undefined) {
    throw new Error(
      `${id}: 帯が未導出（r ${
        ratio.toPrecision(3)
      }）— 決定用の最悪 r × 5 を UMT5_NORMALIZED_BAND へ書く` +
        "（候補は最後の step が出す）",
    );
  }
  // 故障注入の判定を先に置く（受入れが帯の外でも、帯が故障を拾えるかは判定される）。
  for (const fault of faults) {
    assert(
      fault.ratio > band,
      `${id}: 故障注入 ${fault.label} の r ${
        fault.ratio.toPrecision(3)
      } が帯 ${band} の内に収まった`,
    );
  }
  assert(ratio <= band, `${id}: r ${ratio.toPrecision(3)} が帯 ${band} の外`);
};

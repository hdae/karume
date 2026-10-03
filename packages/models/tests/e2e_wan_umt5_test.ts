/**
 * Wan2.1 の umT5（i8 系列のテキストエンコーダ）の**移植の門**（実 GPU・ADR 0119 段 10c — 決定 3 / 4 / 5 / 8）。
 *
 * 系列の容器（`outputs/series/wan2.1-umt5-i8-dyn/text_encoder/` — 重み i8 per-channel・活性 f32・入力は token id
 * `[1,L]` とバケット表 `[L,L]`・L は 2〜512 の記号次元）を karume runtime で回し、CPU の層逐次の参照（同じ i8 の
 * fake-quant 重み — recipe `wan/umt5_reference.py`）と突き合わせる。パイプラインへの結線は段 10d で、ここは
 * text_encoder 単体の forward を見る。
 *
 * 生成は `cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_export reference`（容器は
 * 段 10b の書き出し）。golden は 1 ケース 1 本の `reference.<case>.safetensors`:
 *
 * - `input_ids` I32 `[L]` / `relative_position_buckets` I32 `[L,L]` — Python の経路（上流の `prompt_clean` →
 *   transformers 5.14.1・上流のバケットの式）の入力
 * - `output.f64` F32 `[1,L,4096]` — 活性も f64 で回した参照を f32 へ丸めた値（誤差の基準）
 * - `output.f32` F32 `[1,L,4096]` — CPU f32 の参照（上流の eager とビット一致する書き下し — 正規化の分母）
 * - メタ（キー 1 つの JSON）— プロンプトの原文・L・役割・容器の part 0 の sha256
 *
 * ## 門
 *
 * 1. **入力の両経路の一致**（GPU 不要）: golden の原文を TS のトークナイザ（`WanPromptEncoder` — 10a の資産）に
 *    通した id 列と、TS のバケット表（`buildUmt5RelativePositionBuckets`）が golden の入力と**ビット一致**。GPU の
 *    門はこの TS の入力（製品の経路）で回す。golden が今の容器から採られたこと（part 0 の sha256）も見る。
 * 2. **正規化比 r の帯**（決定 8 ①）: r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|)（DiT の
 *    実寸の門と同じ指標 — `e2e_wan_dit_test.ts` の `DIT_FULL_NORMALIZED_BAND`）。帯は決定用 6 本の最悪 r × 5 を
 *    有効数字 2 桁へ切り上げた値で、受入れ 4 本は別に判定する（{@link UMT5_NORMALIZED_BAND}）。
 * 3. **故障注入**（受入れの 4 本で・帯の外へ出ること）: 相対位置の表を 1 ずらす・id 列の 1 トークンを別の id に・
 *    層 0 / 1 の相対位置の表の取り違え（容器の供給面で block を入れ替える — 別の Session）。
 *
 * ## 記録（門ではない）
 *
 * - **品質の記録**（決定 8 ②）: 受入れ 4 本で GPU i8・CPU f32 の参照（`output.f32`）・bf16 の事前計算資産
 *   （`outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors`）の 3 点を、行ごとのコサイン類似度の最小・
 *   相対フロベニウス誤差・最大絶対差で比べて results.json の note へ（f32 の参照を基準にする — ADR 0119 追記
 *   「10b の結果」）。
 * - **資源**: 構築と 1 forward ごとの壁時間・確保（重み・アリーナ・常駐）・fdinfo の VRAM の山（区間ごと）を、
 *   ADR 0119 の容量の見立て（重み i8 5.30 GiB）と並べる。
 *
 * 資産が無い環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。資産が**一部だけ**ある環境は
 * SKIP ではなく FAIL にする（完全性テスト）。
 */

import { assert, assertEquals } from "@std/assert";
import {
  acquireGpu,
  type BoundContainer,
  type GpuContext,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { parseWanTokenizerAsset, WanPromptEncoder } from "../src/wan/text/tokenizer.ts";
import {
  buildUmt5RelativePositionBuckets,
  type I32Tensor,
} from "../src/wan/umt5/relative-position.ts";
import { type Umt5SessionInputs, umt5SessionInputs } from "../src/wan/umt5/session-io.ts";
import { formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

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
 */
const UMT5_NORMALIZED_BAND: number | undefined = undefined;

const SERIES_NAME = "wan2.1-umt5-i8-dyn";
const COMPONENT = "text_encoder";
const SERIES_DIR = new URL(
  `../../../outputs/series/${SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);
const MODEL_FILE = "model.krm";
/**
 * 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本）。**使うときに引く** — 表に行が無いときにモジュール
 * 評価で落とすと、GPU 不要のテストまで巻き添えで落ちる。
 */
const graphName = (): string => seriesGraph(SERIES_NAME, COMPONENT);

/** 10a のトークナイザ資産（recipe `wan/umt5_tokenizer.py` が焼く）。 */
const TOKENIZER_URL = new URL(
  "../../../outputs/series/wan2.1-umt5-tokenizer/tokenizer.json",
  import.meta.url,
);
/** bf16 の事前計算資産（固定 4 本 — recipe `wan/text_embeds.py`・品質の記録の 3 点目）。 */
const TEXT_EMBEDS_URL = new URL(
  "../../../outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors",
  import.meta.url,
);

const GENERATE =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_export reference";
const GENERATE_TOKENIZER =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_tokenizer";

/** golden のキー（recipe `wan/umt5_reference.py` と同じ綴り）。 */
const INPUT_IDS_KEY = "input_ids";
const BUCKETS_KEY = "relative_position_buckets";
const OUTPUT_F64_KEY = "output.f64";
const OUTPUT_F32_KEY = "output.f32";
const METADATA_KEY = "karume.wan.umt5_reference";
const REFERENCE_FORMAT = "karume-wan-umt5-reference/1";

/**
 * 故障注入で取り違える 2 層の相対位置の表（層の添字の 1 ずれ — 先頭の 2 層）。容器の initializer 名は recipe の
 * `umt5_reference.layer_keys` と同じ綴り。
 */
const relativeBiasTable = (layer: number): string =>
  `encoder.block.${layer}.layer.0.SelfAttention.relative_attention_bias.weight`;
const SWAPPED_TABLES: readonly [string, string] = [relativeBiasTable(0), relativeBiasTable(1)];

/** ADR 0119 の容量の見立て（i8 の重み 5,686,617,600 B = 5.30 GiB — 調査 §1.2・§5）。 */
const ESTIMATED_WEIGHT_BYTES = 5_686_617_600;

type CaseRole = "band" | "accept";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する**（生成を一部だけ流した環境でテストが黙って
 * 消える形にしない）。正本は recipe `wan/umt5_reference.py`（決定用は 10a のパリティ fixture の受理した乱択から
 * seed 20261003 で選んだ単体 3 本〈L = 8 / 22 / 36〉と、乱択を連ねた合成 3 本〈L = 163 / 327 / 488〉・受入れは固定
 * 4 プロンプト）。`embedding` は bf16 の事前計算資産のテンソル名（受入れだけ）。
 */
const CASES: readonly {
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

const goldenUrl = (name: string): URL => new URL(`reference.${name}.safetensors`, SERIES_DIR);

/** ファイルの有無（NotFound 以外は伝播させる — 権限エラーを「資産が無い」に読み替えない）。 */
const filePresent = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, SERIES_DIR));
const expectedFiles = CASES.map(({ name }) => goldenUrl(name));
const ASSETS_AVAILABLE = MODEL_PRESENT && expectedFiles.every(filePresent);
const ANY_PRESENT = MODEL_PRESENT || expectedFiles.some(filePresent);
const tokenizerText = await readTextIfPresent(TOKENIZER_URL);

if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_DIR.pathname} に umT5 の容器と golden が揃っていないため Wan の umT5 の e2e を SKIP ` +
      `する（重み 5.3 GB につきリポジトリ管理外）。生成: ${GENERATE}`,
  );
} else if (tokenizerText === undefined) {
  console.warn(
    `[karume] ${TOKENIZER_URL.pathname} が無いため Wan の umT5 の e2e（入力は TS のトークナイザで作る）を ` +
      `SKIP する。生成: ${GENERATE_TOKENIZER}`,
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

const floatsOf = (
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
type GoldenMeta = {
  readonly role: string;
  readonly prompt: string;
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
  const { format, role, prompt, length } = meta;
  const part0Sha256 = container["part0Sha256"];
  if (
    format !== REFERENCE_FORMAT || typeof role !== "string" || typeof prompt !== "string" ||
    typeof length !== "number" || typeof part0Sha256 !== "string"
  ) {
    throw new Error(`${where}: メタの形が ${REFERENCE_FORMAT} でない`);
  }
  return { role, prompt, length, part0Sha256 };
};

/** 1 ケースぶんの golden。 */
type Golden = {
  readonly meta: GoldenMeta;
  readonly ids: Int32Array<ArrayBuffer>;
  readonly buckets: Int32Array<ArrayBuffer>;
  readonly expected: Float32Array<ArrayBuffer>;
  readonly reference: Float32Array<ArrayBuffer>;
  readonly outputShape: readonly number[];
};

const loadGolden = async (name: string): Promise<Golden> => {
  const file = parseSafetensors(await readBuffer(goldenUrl(name)));
  const meta = parseMeta(file, name);
  const ids = intsOf(file, INPUT_IDS_KEY, name);
  assertEquals(meta.length, ids.length, `${name}: メタの L と input_ids の長さ`);
  return {
    meta,
    ids,
    buckets: intsOf(file, BUCKETS_KEY, name),
    expected: floatsOf(file, OUTPUT_F64_KEY, name),
    reference: floatsOf(file, OUTPUT_F32_KEY, name),
    outputShape: viewOf(file, OUTPUT_F64_KEY, name).shape,
  };
};

/** 最初に割れる要素の添字（一致なら -1・長さが違えば 0）。 */
const firstMismatch = (actual: ArrayLike<number>, expected: ArrayLike<number>): number => {
  if (actual.length !== expected.length) return 0;
  for (let index = 0; index < expected.length; index += 1) {
    if (actual[index] !== expected[index]) return index;
  }
  return -1;
};

const hex = (bytes: ArrayBuffer): string =>
  [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");

/** 容器の part 0（ヘッダ + 2 文書 — 全 block の sha256 を宣言する）の sha256（golden のメタと突き合わせる）。 */
const containerPart0Sha256 = async (): Promise<string> => {
  const [part0] = resolveParts(new URL(MODEL_FILE, SERIES_DIR));
  return hex(await crypto.subtle.digest("SHA-256", await Deno.readFile(part0)));
};

/** TS のトークナイザ（資産が無い回は ignore で SKIP 済み — ここへ来たら条件の食い違いとして落とす）。 */
const tokenizerEncoder = (): WanPromptEncoder => {
  if (tokenizerText === undefined) {
    throw new Error(`${TOKENIZER_URL.pathname} が無い（ignore の条件と食い違っている）`);
  }
  return new WanPromptEncoder(parseWanTokenizerAsset(JSON.parse(tokenizerText)));
};

/**
 * 製品の経路（TS のトークナイザ + TS のバケット表）でグラフ入力を作り、golden の入力（Python の経路）と
 * **ビット一致**することを見てから返す（門 1 — GPU の門は一致した入力でしか回さない）。
 */
const hostInputs = (encoder: WanPromptEncoder, golden: Golden, name: string) => {
  const ids = encoder.encode(golden.meta.prompt, name);
  const at = firstMismatch(ids, golden.ids);
  assertEquals(
    at,
    -1,
    `${name}: TS の id 列が golden と ${at} 番目で割れる（TS ${ids.length} 本）`,
  );
  const buckets = buildUmt5RelativePositionBuckets(ids.length);
  const cell = firstMismatch(buckets.data, golden.buckets);
  assertEquals(cell, -1, `${name}: TS のバケット表が golden と要素 ${cell} で割れる`);
  return { ids, buckets, inputs: umt5SessionInputs(ids, buckets) };
};

type Difference = {
  readonly maxAbs: number;
  readonly maxRel: number;
  readonly referenceMaxAbs: number;
  readonly nonFinite: number;
};

const difference = (actual: Float32Array, expected: Float32Array): Difference => {
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
const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/**
 * 正規化の分母 = CPU f32 の参照の f64 の参照に対する比。0（f32 の参照が f64 とビット一致）だと割れないので
 * fail loudly。
 */
const referenceRatioOf = (golden: Golden, name: string): number => {
  const ratio = ratioOf(difference(golden.reference, golden.expected));
  if (!(ratio > 0)) {
    throw new Error(`${name}: CPU f32 の参照の f64 に対する比が ${ratio}（正規化できない）`);
  }
  return ratio;
};

const formatRatio = (label: string, diff: Difference, referenceRatio: number): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / 参照の最大絶対値 ${
    diff.referenceMaxAbs.toFixed(3)
  }（比 ${ratioOf(diff).toExponential(2)}）・r ${(ratioOf(diff) / referenceRatio).toPrecision(3)}`;

/** 品質の記録の 3 指標（決定 8 ② — `base` を基準に）。 */
const rowQuality = (actual: Float32Array, base: Float32Array, width: number): string => {
  assertEquals(actual.length, base.length, "要素数");
  let minCosine = Number.POSITIVE_INFINITY;
  let errorSquares = 0;
  let baseSquares = 0;
  let maxAbs = 0;
  for (let row = 0; row < base.length / width; row += 1) {
    let dot = 0;
    let left = 0;
    let right = 0;
    for (let column = row * width; column < (row + 1) * width; column += 1) {
      const a = actual[column];
      const b = base[column];
      dot += a * b;
      left += a * a;
      right += b * b;
      errorSquares += (a - b) ** 2;
      maxAbs = Math.max(maxAbs, Math.abs(a - b));
    }
    baseSquares += right;
    minCosine = Math.min(minCosine, dot / Math.sqrt(left * right));
  }
  return `行ごとのコサイン類似度の最小 ${minCosine.toFixed(6)}・相対フロベニウス ${
    Math.sqrt(errorSquares / baseSquares).toExponential(3)
  }・maxAbs ${maxAbs.toExponential(3)}`;
};

/** 相対位置の表を 1 ずらした表（`table'[i][j] = bucket(j + 1 − i)` — ホストの距離の 1 ずれの故障注入）。 */
const shiftedBuckets = (length: number): I32Tensor => {
  const wide = buildUmt5RelativePositionBuckets(length + 1);
  const data = new Int32Array(length * length);
  for (let row = 0; row < length; row += 1) {
    const start = row * (length + 1) + 1;
    data.set(wide.data.subarray(start, start + length), row * length);
  }
  return { dtype: "i32", shape: [length, length], data };
};

/**
 * id 列の中ほどの 1 トークンを、同じ列の別の id に置き換えた列（語彙の内の id を使う — 範囲外の id で落ちる
 * 形ではなく、別の語として黙って通る形を注入する）。
 */
const replacedToken = (ids: Int32Array<ArrayBuffer>): Int32Array<ArrayBuffer> => {
  const at = Math.floor((ids.length - 1) / 2);
  const other = ids.find((id, index) => index < ids.length - 1 && id !== ids[at]);
  if (other === undefined) throw new Error("置き換える別の id が列に無い");
  const replaced = ids.slice();
  replaced[at] = other;
  return replaced;
};

/**
 * 2 層の相対位置の表の block を入れ替えた供給面（故障注入 — 層の表の取り違え）。容器のファイルは書き換えず、
 * `readBlock` が返す block を入れ替える（どちらも sha256 の検証を通った本物の block — `BoundContainer.readBlock`
 * の MUST どおり、返された器は書き換えない）。
 */
const swappedTablesContainer = (
  opened: OpenedContainer,
  graph: string,
  [first, second]: readonly [string, string],
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const blockOf = (weight: string): string => {
    const supply = bound?.supplies.get(weight);
    if (supply === undefined || supply.blocks.length !== 1 || supply.scale !== undefined) {
      throw new Error(`グラフ '${graph}' の '${weight}' が丸ごと 1 block の F32 でない`);
    }
    return supply.blocks[0].id;
  };
  const swap = new Map([[blockOf(first), blockOf(second)], [blockOf(second), blockOf(first)]]);
  return {
    graphs: opened.graphs,
    readBlock: (id) => opened.readBlock(swap.get(id) ?? id),
  };
};

const gib = (bytes: number): string => `${(bytes / 2 ** 30).toFixed(2)} GiB`;

/** 1 forward の観測（壁時間と、run の直後に生きている確保）。 */
type Run = {
  readonly label: string;
  readonly wallMs: number;
  readonly weightBytes: number;
  readonly arenaBytes: number;
  readonly backingBytes: number;
};

const observeRun = (label: string, diagnostics: SessionDiagnostics, wallMs: number): Run => ({
  label,
  wallMs,
  weightBytes: diagnostics.weights.allocatedBytes,
  arenaBytes: diagnostics.lastRun?.allocatedBytes ?? 0,
  backingBytes: diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
});

const formatRun = (run: Run): string =>
  `${run.label}: 壁 ${(run.wallMs / 1000).toFixed(2)} s・確保 ${
    gib(run.weightBytes + run.arenaBytes + run.backingBytes)
  }（重み ${gib(run.weightBytes)}・アリーナ ${gib(run.arenaBytes)}・常駐 ${
    gib(run.backingBytes)
  }）`;

/** 1 forward を回して出力（f32 `[1, L, 4096]`）と観測を返す（次の run の前に解放を待つ）。 */
const forward = async (
  session: Session,
  prepared: PreparedModel,
  gpu: GpuContext,
  inputs: Umt5SessionInputs,
  label: string,
): Promise<{ readonly output: Extract<Tensor, { readonly dtype: "f32" }>; readonly run: Run }> => {
  const started = performance.now();
  const outputs = await session.run(inputs);
  const wallMs = performance.now() - started;
  const output = outputs[prepared.graph.outputs[0]];
  if (output.dtype !== "f32") throw new Error(`${label}: 出力が ${output.dtype}`);
  const run = observeRun(label, session.diagnostics(), wallMs);
  await settleReleases(gpu);
  return { output, run };
};

const results = openResults("wan-umt5");

Deno.test({
  name: "Wan umT5 資産: i8 系列の容器と全ケース（決定用 6 本・受入れ 4 本）の golden が揃っている",
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
    "Wan umT5 ホスト（GPU 不要）: TS のトークナイザの id 列と TS のバケット表が golden の入力（Python の経路）と" +
    "ビット一致し、golden が今の容器から採られている",
  ignore: !ASSETS_AVAILABLE || tokenizerText === undefined,
  fn: async () => {
    const encoder = tokenizerEncoder();
    const part0 = await containerPart0Sha256();
    for (const { name, role } of CASES) {
      const golden = await loadGolden(name);
      assertEquals(golden.meta.role, role, `${name}: 役割`);
      assertEquals(
        golden.meta.part0Sha256,
        part0,
        `${name}: golden を採った容器（part 0 の sha256）が今の容器と違う — 生成し直す: ${GENERATE}`,
      );
      hostInputs(encoder, golden, name);
    }
  },
});

/** 1 ケースの r（帯の候補の材料 — 判定の前に積むので帯の外の回も残る）。 */
type CaseRatio = { readonly name: string; readonly role: CaseRole; readonly ratio: number };

/**
 * 未導出の帯の候補を出して赤で止める（決定用の最悪 r × 5 を有効数字 2 桁へ切り上げ — 受入れの r は並べるが
 * 候補には使わない）。決定用のどれかが r を出す前に落ちた回は候補を出さない（欠けた最悪は最悪ではない）。
 */
const bandCandidate = (ratios: readonly CaseRatio[]): string => {
  const expected = CASES.filter(({ role }) => role === "band").length;
  const decided = ratios.filter(({ role }) => role === "band");
  const accepted = ratios.filter(({ role }) => role === "accept")
    .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`).join(" / ") || "なし";
  if (decided.length !== expected) {
    return `決定用 ${expected} 本のうち r が出たのは ${decided.length} 本 — 候補にしない（落ちたケースを先に調べる）。` +
      `受入れの r: ${accepted}`;
  }
  const worst = Math.max(...decided.map(({ ratio }) => ratio));
  const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
  const candidate = Number((Math.ceil(worst * 5 / unit) * unit).toPrecision(2));
  return `決定用 ${expected} 本（${
    decided.map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`).join(" / ")
  }）の最悪 r ${worst.toPrecision(3)} × 5 = ${
    (worst * 5).toPrecision(3)
  }（有効数字 2 桁へ切り上げ ${candidate}）。` +
    `受入れの r（帯の決定に使わない）: ${accepted}`;
};

/** 帯の判定（未導出なら r を出して赤 — {@link UMT5_NORMALIZED_BAND}）。 */
const judgeBand = (
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

Deno.test({
  name:
    "Wan umT5 移植の門（実 GPU / CPU 層逐次 f64）: TS の入力で回した 1 forward の正規化比 r が帯の内・故障注入 3 件は" +
    "帯の外・品質の記録（GPU i8 / CPU f32 / bf16 資産）と資源（壁時間・確保・fdinfo の VRAM の山）を残す",
  ignore: !ASSETS_AVAILABLE || tokenizerText === undefined || !GPU_AVAILABLE,
  fn: async (t) => {
    const band = UMT5_NORMALIZED_BAND;
    const encoder = tokenizerEncoder();
    const embeds = filePresent(TEXT_EMBEDS_URL)
      ? parseSafetensors(await readBuffer(TEXT_EMBEDS_URL))
      : undefined;
    if (embeds === undefined) {
      console.warn(
        `[karume] ${TEXT_EMBEDS_URL.pathname} が無いため品質の記録の bf16 の 3 点目を省く（生成: cd ` +
          "tools/export-recipes && uv run --group wan --inexact python -m wan.text_embeds）",
      );
    }
    const graph = graphName();
    const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR));
    const prepared = prepareContainer(opened, graph);
    assertEquals(
      prepared.graph.inputs.map(({ name }) => name),
      [INPUT_IDS_KEY, BUCKETS_KEY],
      "グラフ入力",
    );
    assertEquals(prepared.graph.outputs.length, 1, "グラフの出力は 1 本");
    let deviceLost: string | undefined;
    const gpu = await acquireGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    // 区間ごとの VRAM の山（fdinfo の `drm-total-*` — 10 ms の標本で短い山は取りこぼしうる・門ではない）。
    // `start` の区間は Session を張る前の値（この process の他の確保）。
    const monitor = monitorDrmUsage();
    const ratios: CaseRatio[] = [];
    const runs: Run[] = [];
    let buildMs = 0;
    try {
      assertAdapterMatchesEnvironment(gpu);
      monitor.enter("構築");
      const built = performance.now();
      const session = await prepared.createContainerSession(gpu);
      buildMs = performance.now() - built;
      try {
        for (const { name, role, embedding } of CASES) {
          await t.step(`${name}（${role}）`, async () => {
            let caseNote = "";
            await runRecordedCase(results, {
              id: name,
              failureNote: () => caseNote,
            }, async ({ measurements }) => {
              const golden = await loadGolden(name);
              const { ids, inputs } = hostInputs(encoder, golden, name);
              const referenceRatio = referenceRatioOf(golden, name);
              monitor.enter(`run ${name}`);
              const { output, run } = await forward(session, prepared, gpu, inputs, name);
              runs.push(run);
              assertEquals(output.shape, golden.outputShape, `${name}: 出力の形`);
              const diff = difference(output.data, golden.expected);
              const diffF32 = difference(output.data, golden.reference);
              const referenceDiff = difference(golden.reference, golden.expected);
              const ratio = ratioOf(diff) / referenceRatio;
              ratios.push({ name, role, ratio });
              const unbounded = { atol: Number.POSITIVE_INFINITY, rtol: 0 };
              measurements.push(
                {
                  output: "output@f64-reference",
                  maxAbs: diff.maxAbs,
                  maxRel: diff.maxRel,
                  // 判定と同じ境界を絶対値へ写す（未導出の回は無限の帯 — 判定は下で赤にする）。
                  tolerance: band === undefined
                    ? unbounded
                    : { atol: band * referenceRatio * diff.referenceMaxAbs, rtol: 0 },
                  stage: "karume",
                },
                {
                  output: "output@f32-reference",
                  maxAbs: diffF32.maxAbs,
                  maxRel: diffF32.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                },
                {
                  output: "f32-reference@f64-reference",
                  maxAbs: referenceDiff.maxAbs,
                  maxRel: referenceDiff.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                },
              );
              const notes = [
                `L ${ids.length}`,
                formatRatio("GPU / f64 参照", diff, referenceRatio),
                `CPU f32 参照 / f64 参照（正規化の分母）: 比 ${referenceRatio.toExponential(3)}`,
                formatRun(run),
              ];
              assertEquals(diff.nonFinite, 0, `${name}: 非有限`);

              const faults: { label: string; ratio: number }[] = [];
              if (role === "accept") {
                const width = golden.outputShape[golden.outputShape.length - 1];
                notes.push(
                  `品質（基準 CPU f32 参照）GPU i8: ${
                    rowQuality(output.data, golden.reference, width)
                  }`,
                );
                if (embeds !== undefined && embedding !== undefined) {
                  const asset = floatsOf(embeds, embedding, `text_embeds の ${embedding}`);
                  notes.push(
                    `品質（基準 CPU f32 参照）bf16 資産: ${
                      rowQuality(asset, golden.reference, width)
                    }`,
                    `品質（基準 bf16 資産）GPU i8: ${rowQuality(output.data, asset, width)}`,
                  );
                }
                const injected: readonly (readonly [string, Umt5SessionInputs])[] = [
                  ["相対位置の表を 1 ずらす", umt5SessionInputs(ids, shiftedBuckets(ids.length))],
                  [
                    "id 列の中ほどの 1 トークンを別の id に",
                    umt5SessionInputs(
                      replacedToken(ids),
                      buildUmt5RelativePositionBuckets(ids.length),
                    ),
                  ],
                ];
                for (const [label, faulty] of injected) {
                  const { output: faultOutput } = await forward(
                    session,
                    prepared,
                    gpu,
                    faulty,
                    `${name} ${label}`,
                  );
                  const faultDiff = difference(faultOutput.data, golden.expected);
                  faults.push({ label, ratio: ratioOf(faultDiff) / referenceRatio });
                  notes.push(`故障注入 ${formatRatio(label, faultDiff, referenceRatio)}`);
                }
              }
              // 判定の前に出す（帯の外へ出た回も比が手元に残る）。
              caseNote = notes.join(" / ");
              console.log(`[wan-umt5] ${name}: ${caseNote}`);
              assertEquals(deviceLost, undefined, "device lost");
              judgeBand(band, name, ratio, faults);
              return { status: "pass", note: caseNote };
            });
          });
        }
      } finally {
        monitor.enter("破棄");
        await session.dispose();
        await settleReleases(gpu);
      }
      monitor.enter("破棄後");

      await t.step("資源の記録（壁時間・確保・fdinfo の VRAM の山 — 門ではない）", async () => {
        const { peaks } = monitor.stop();
        const weightBytes = runs.length === 0 ? 0 : runs[0].weightBytes;
        const note = [
          `構築 ${(buildMs / 1000).toFixed(1)} s`,
          `重みの確保 ${gib(weightBytes)}（ADR 0119 の見立て i8 ${gib(ESTIMATED_WEIGHT_BYTES)} の ${
            (weightBytes / ESTIMATED_WEIGHT_BYTES).toFixed(3)
          } 倍）`,
          ...runs.map(formatRun),
          ...[...peaks].map(([phase, peak]) => `VRAM 山 [${phase}] ${formatDrmUsage(peak)}`),
        ].join(" / ");
        console.log(`[wan-umt5] 資源: ${note}`);
        await runRecordedCase(
          results,
          { id: "umt5/resources" },
          () => Promise.resolve({ status: "pass", note }),
        );
      });

      const swapLabel = "層 0 / 1 の相対位置の表の取り違え";
      const swapped = prepareContainer(
        swappedTablesContainer(opened, graph, SWAPPED_TABLES),
        graph,
      );
      // 本体の Session を畳んだ後に張る（2 本を同時に載せない — 重み 5.3 GiB × 2 は B570 の天井
      // 9,600 MiB を越える）。
      const faultSession = await swapped.createContainerSession(gpu);
      try {
        for (const { name } of CASES.filter(({ role }) => role === "accept")) {
          const id = `${name}/table-swap`;
          await t.step(`${id}（故障注入 ${swapLabel} → 帯の外）`, async () => {
            let note = "";
            await runRecordedCase(results, { id, failureNote: () => note }, async () => {
              const golden = await loadGolden(name);
              const { inputs } = hostInputs(encoder, golden, name);
              const referenceRatio = referenceRatioOf(golden, name);
              const { output } = await forward(faultSession, swapped, gpu, inputs, id);
              const faultDiff = difference(output.data, golden.expected);
              const ratio = ratioOf(faultDiff) / referenceRatio;
              note = `故障注入 ${formatRatio(swapLabel, faultDiff, referenceRatio)}（${
                band === undefined ? "帯は未導出" : `帯の ${(ratio / band).toFixed(1)} 倍`
              }）`;
              console.log(`[wan-umt5] ${id}: ${note}`);
              if (band === undefined) throw new Error(`${id}: 帯が未導出（故障注入の r ${ratio}）`);
              assert(ratio > band, `${id}: r ${ratio.toPrecision(3)} が帯 ${band} の内に収まった`);
              return { status: "pass", note };
            });
          });
        }
      } finally {
        await faultSession.dispose();
        await settleReleases(gpu);
      }

      if (band === undefined) {
        await t.step("帯の候補（未導出）", () => {
          throw new Error(`帯が未導出 — ${bandCandidate(ratios)}`);
        });
      }
    } finally {
      monitor.stop();
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

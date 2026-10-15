/**
 * Wan2.1 の umT5（i8 系列のテキストエンコーダ）の**移植の門**（実 GPU・ADR 0119 段 10c — 決定 3 / 4 / 5 / 8）。
 *
 * 系列の容器（`outputs/series/umt5-xxl-i8-dyn/text_encoder/` — 重み i8 per-channel・活性 f32・入力は token id
 * `[1,L]` とバケット表 `[L,L]`・L は 2〜512 の記号次元）を karume runtime で回し、CPU の層逐次の参照（同じ i8 の
 * fake-quant 重み — recipe `wan/umt5_reference.py`）と突き合わせる。パイプラインへの結線は段 10d で、ここは
 * text_encoder 単体の forward を見る。
 *
 * 生成は `cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_export reference`（容器は
 * 段 10b の書き出し）。golden は 1 ケース 1 本の `reference.<case>.safetensors` で、中身と読み口・帯・判定は
 * `helpers/umt5-golden.ts`（単体の `Umt5Encoder` の通しの e2e〈`e2e_umt5_encoder_test.ts`〉と共有の 1 本）。
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
 *    層 0 / 1 の相対位置の表の取り違え（容器の供給面で block を入れ替える — 別の Session）・i8 に固有の 1 件
 *    （中ほどの層の linear 1 本の per-channel scale を 2 倍 — {@link scaledScaleContainer}・別の Session）。同じ linear の
 *    scale を {@link SUBTLE_SCALE_FACTOR} 倍にした微妙な故障は r を記録だけする（帯との距離 — DiT の
 *    `SUBTLE_FAULT_MARGIN` に当たる床を決める材料）。
 *
 * ## 記録（門ではない）
 *
 * - **品質の記録**（決定 8 ②）: 受入れ 4 本で GPU i8・bf16 の事前計算資産
 *   （`outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors`）を、**量子化なしの CPU f32 の参照**
 *   （`output.unquantized.f32` — ADR 0119 追記「10b の結果」の「f32 参照を基準にする」）を基準に、行ごとのコサイン
 *   類似度の最小・相対フロベニウス誤差・最大絶対差で比べて results.json の note へ。「i8 の丸めの影響」（GPU i8 vs
 *   量子化なし f32）と「bf16 の影響」（bf16 資産 vs 量子化なし f32）を分けて出す。i8 の CPU f32 の参照（`output.f32`）
 *   との比較も並べる（GPU 自身の誤差）。
 * - **資源**: 構築と 1 forward ごとの壁時間・確保（重み・アリーナ・常駐）・fdinfo の VRAM の山（区間ごと）を、
 *   ADR 0119 の容量の見立て（重み i8 5.30 GiB）と並べる。計測モード（`gpuTiming` — 通常モードとは別の device）で
 *   1 forward の GPU 時間と 1 submit の GPU 時間の最大（ADR 0118 決定 6 の 1 s を目安）も記録する。timestamp の
 *   換算表（`helpers/timestamp-unit.ts`）に無い環境では換算を飛ばし、そう note に書く（r の門は換算に依らない）。
 *
 * 資産が無い環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。資産が**一部だけ**ある環境は
 * SKIP ではなく FAIL にする（完全性テスト）。
 */

import { assert, assertEquals } from "@std/assert";
import {
  type BoundContainer,
  type GpuContext,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { parseWanTokenizerAsset, WanPromptEncoder } from "../src/wan/text/tokenizer.ts";
import { buildUmt5RelativePositionBuckets, type I32Tensor } from "../src/umt5/relative-position.ts";
import { type Umt5SessionInputs, umt5SessionInputs } from "../src/umt5/session-io.ts";
import { formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { lookupTimestampUnitNs } from "./helpers/timestamp-unit.ts";
import {
  BUCKETS_KEY,
  type CaseRole,
  CASES,
  COMPONENT,
  difference,
  filePresent,
  firstMismatch,
  floatsOf,
  formatRatio,
  GENERATE,
  type Golden,
  goldenUrl,
  INPUT_IDS_KEY,
  judgeBand,
  loadGolden,
  ratioOf,
  readBuffer,
  referenceRatioOf,
  SERIES_DIR,
  SERIES_NAME,
  UMT5_NORMALIZED_BAND,
} from "./helpers/umt5-golden.ts";
import {
  modelPresent,
  openSeriesContainer,
  resolveParts,
} from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import {
  assertAdapterMatchesEnvironment,
  ENVIRONMENT,
} from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

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

const GENERATE_TOKENIZER =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.umt5_tokenizer";

/**
 * 故障注入で取り違える 2 層の相対位置の表（層の添字の 1 ずれ — 先頭の 2 層）。容器の initializer 名は recipe の
 * `umt5_reference.layer_keys` と同じ綴り。
 */
const relativeBiasTable = (layer: number): string =>
  `encoder.block.${layer}.layer.0.SelfAttention.relative_attention_bias.weight`;
const SWAPPED_TABLES: readonly [string, string] = [relativeBiasTable(0), relativeBiasTable(1)];

/**
 * i8 に固有の故障注入で per-channel scale を倍にする linear（{@link scaledScaleContainer}）。24 層の中ほど（層 12）の
 * FFN の出口 `wo` を選ぶ: 出力は残差へ線形に足されるので、倍率 (1 + δ) はその層の FFN の寄与を δ だけずらす形で出て、
 * δ と r の関係が読める。q / k は避けた — umT5 は q / k の直後に norm もスケールも無く（上流
 * `UMT5Attention.forward` — `scores = q·kᵀ` に相対位置のバイアスを足して softmax）倍率は打ち消されないが、softmax の
 * 温度として非線形に効くので、微妙な水準で δ と r の関係が読みにくい（DiT が V を選んだ理由 — q / k の直後の
 * RMS norm — とは事情が違う）。
 */
const I8_SCALE_FAULT_WEIGHT = "encoder.block.12.layer.1.DenseReluDense.wo.weight";
/** 帯の外を門にする倍率（i8 の scale の取り違え — 軸・値・束縛 — の大きさ）。 */
const GROSS_SCALE_FACTOR = 2;
/**
 * r を記録だけする微妙な倍率（帯との距離を見る — DiT の `SUBTLE_FAULT_MARGIN` に当たる床を、帯を決めた後に
 * ここの観測から決める材料）。
 *
 * 値は CPU の見積りで帯の桁に寄せた（2026-10-03・GPU 不使用 — i8 の CPU f32 の層逐次でこの linear の重みを倍にし、
 * 受入れ 4 本の r を採った）: r は δ にほぼ比例し、×1.01 で 2.17e3〜1.04e4・×1.001 で 217〜897・×1.0001 で 20.9〜86.4
 * （×2 は 5.04e5〜9.70e5）。帯は未導出だが、決定用の最悪 r × 5 は DiT の前例（帯 30〜104）と同じ桁と見込み、帯に近い
 * ×1.0001 を採る。GPU の r は GPU 自身の誤差（DiT の i8 では r にして 2〜11）が乗る分だけこの見積りからずれうる。
 */
const SUBTLE_SCALE_FACTOR = 1.0001;

/** ADR 0119 の容量の見立て（i8 の重み 5,686,617,600 B = 5.30 GiB — 調査 §1.2・§5）。 */
const ESTIMATED_WEIGHT_BYTES = 5_686_617_600;

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

/**
 * linear 1 本（`weight`）の per-channel scale を `factor` 倍にした供給面（i8 に固有の故障注入 —
 * {@link I8_SCALE_FAULT_WEIGHT}・DiT の `scaledScaleContainer` に倍率の引数を足した形）。容器のファイルは書き換えない:
 * 取得した block（sha256 の検証の後）の**写し**を倍にして返す（`BoundContainer.readBlock` の MUST — 返された器は書き換え
 * ない）。scale の payload は block の先頭からの f32 の列（Session 構築が `subarray(0, payloadBytes)` で読む形）。
 */
const scaledScaleContainer = (
  opened: OpenedContainer,
  graph: string,
  weight: string,
  factor: number,
): BoundContainer => {
  const bound = Object.hasOwn(opened.graphs, graph) ? opened.graphs[graph] : undefined;
  const scale = bound?.supplies.get(weight)?.scale;
  if (scale === undefined) {
    throw new Error(
      `グラフ '${graph}' の '${weight}' に companion scale が無い（i8 の linear でない）`,
    );
  }
  return {
    graphs: opened.graphs,
    readBlock: async (id) => {
      const bytes = await opened.readBlock(id);
      if (id !== scale.id) return bytes;
      const scaled = bytes.slice();
      const values = new Float32Array(scaled.buffer, 0, scale.payloadBytes / 4);
      for (let index = 0; index < values.length; index += 1) values[index] *= factor;
      return scaled;
    },
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

Deno.test({
  name:
    "Wan umT5 移植の門（実 GPU / CPU 層逐次 f64）: TS の入力で回した 1 forward の正規化比 r が帯の内・故障注入 4 件は" +
    "帯の外（scale の微妙な倍率は r を記録）・品質の記録（基準 量子化なし CPU f32 — GPU i8 / i8 の CPU f32 / bf16 資産）と" +
    "資源（壁時間・確保・fdinfo の VRAM の山）を残す",
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
    const gpu = await acquireTestGpu({
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
                const unquantized = golden.unquantized;
                if (unquantized === undefined) {
                  throw new Error(`${name}: 受入れなのに量子化なしの参照が無い`);
                }
                const unquantizedDiff = difference(output.data, unquantized.reference);
                measurements.push({
                  output: "output@unquantized-f32-reference",
                  maxAbs: unquantizedDiff.maxAbs,
                  maxRel: unquantizedDiff.maxRel,
                  tolerance: unbounded,
                  stage: "karume",
                });
                const asset = embeds !== undefined && embedding !== undefined
                  ? floatsOf(embeds, embedding, `text_embeds の ${embedding}`)
                  : undefined;
                // 基準（量子化なし f32）自身の誤差 — 下の差がこれより桁で大きいことを読むための床。
                notes.push(
                  `量子化なし CPU f32 参照 / 量子化なし f64 参照: 比 ${
                    ratioOf(difference(unquantized.reference, unquantized.expected))
                      .toExponential(3)
                  }`,
                  `品質 i8 の丸めの影響（GPU i8 / 基準 量子化なし CPU f32）: ${
                    rowQuality(output.data, unquantized.reference, width)
                  }`,
                );
                if (asset !== undefined) {
                  notes.push(
                    `品質 bf16 の影響（bf16 資産 / 基準 量子化なし CPU f32）: ${
                      rowQuality(asset, unquantized.reference, width)
                    }`,
                  );
                }
                notes.push(
                  `品質（基準 CPU f32 参照）GPU i8: ${
                    rowQuality(output.data, golden.reference, width)
                  }`,
                );
                if (asset !== undefined) {
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

      // i8 に固有の故障注入（scale の倍率 2 水準）。倍率ごとに別の Session を、前の Session を畳んでから張る
      // （表の取り違えと同じ理由 — 2 本を同時に載せない）。
      for (const factor of [GROSS_SCALE_FACTOR, SUBTLE_SCALE_FACTOR]) {
        const gated = factor === GROSS_SCALE_FACTOR;
        const label = `${I8_SCALE_FAULT_WEIGHT} の scale を ${factor} 倍`;
        const scaled = prepareContainer(
          scaledScaleContainer(opened, graph, I8_SCALE_FAULT_WEIGHT, factor),
          graph,
        );
        const scaledSession = await scaled.createContainerSession(gpu);
        try {
          for (const { name } of CASES.filter(({ role }) => role === "accept")) {
            const id = `${name}/${gated ? "scale-fault" : "scale-subtle"}`;
            const purpose = gated ? "帯の外" : "r を記録だけ — 帯との距離";
            await t.step(`${id}（故障注入 ${label} → ${purpose}）`, async () => {
              let note = "";
              await runRecordedCase(results, { id, failureNote: () => note }, async () => {
                const golden = await loadGolden(name);
                const { inputs } = hostInputs(encoder, golden, name);
                const referenceRatio = referenceRatioOf(golden, name);
                const { output } = await forward(scaledSession, scaled, gpu, inputs, id);
                const faultDiff = difference(output.data, golden.expected);
                const ratio = ratioOf(faultDiff) / referenceRatio;
                note = `故障注入 ${formatRatio(label, faultDiff, referenceRatio)}（${
                  band === undefined ? "帯は未導出" : `帯の ${(ratio / band).toPrecision(3)} 倍`
                }）`;
                console.log(`[wan-umt5] ${id}: ${note}`);
                assertEquals(deviceLost, undefined, "device lost");
                if (!gated) return { status: "pass", note };
                if (band === undefined) {
                  throw new Error(`${id}: 帯が未導出（故障注入の r ${ratio}）`);
                }
                assert(
                  ratio > band,
                  `${id}: r ${ratio.toPrecision(3)} が帯 ${band} の内に収まった`,
                );
                return { status: "pass", note };
              });
            });
          }
        } finally {
          await scaledSession.dispose();
          await settleReleases(gpu);
        }
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

/** 1 submit の GPU 実行の幅の目安（ADR 0118 決定 6 の 1 s — ここは記録だけで門にしない）。 */
const SUBMIT_GPU_GUIDE_MS = 1000;

Deno.test({
  name:
    "Wan umT5 資源の記録（実 GPU・計測モード）: 1 forward の GPU 時間と 1 submit の GPU 時間の最大（目安 ≤ 1 s）を" +
    "残す（門ではない — timestamp の換算表に無い環境は換算を飛ばす）",
  ignore: !ASSETS_AVAILABLE || tokenizerText === undefined || !GPU_AVAILABLE,
  fn: async () => {
    const encoder = tokenizerEncoder();
    const graph = graphName();
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)),
      graph,
    );
    let deviceLost: string | undefined;
    // 計測モード（timestamp-query）の device は通常モードと別に取る（DiT の前例 — 2 つの device を同時に載せない。
    // 移植の門のテストは device を破棄して終わる）。submit ごとの GPU 時間は既存の回収に相乗りして測る（runtime の
    // `ChunkBudgetStats.submitGpuTime`）。
    const gpu = await acquireTestGpu({
      gpuTiming: true,
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      const unitNs = lookupTimestampUnitNs(ENVIRONMENT.key);
      const lines: string[] = unitNs === undefined
        ? [
          `環境キー '${ENVIRONMENT.key}' は timestamp の換算表（helpers/timestamp-unit.ts）に無い — ` +
          "GPU 時間は記録しない（推測で換算しない）",
        ]
        : [];
      const session = await prepared.createContainerSession(gpu);
      try {
        for (const { name } of CASES) {
          const golden = await loadGolden(name);
          const { ids, inputs } = hostInputs(encoder, golden, name);
          const { run } = await forward(session, prepared, gpu, inputs, name);
          const ticks = session.diagnostics().lastRunTiming?.totalNs;
          if (ticks === undefined) {
            throw new Error(`${name}: 計測モードなのに run の GPU 時間が無い`);
          }
          lines.push(
            `${name}（L ${ids.length}）: 壁 ${(run.wallMs / 1000).toFixed(2)} s` +
              (unitNs === undefined ? "" : `・GPU ${(ticks * unitNs / 1e9).toFixed(3)} s`),
          );
        }
        const submit = session.diagnostics().submit;
        const observed = submit.chunkBudget.submitGpuTime;
        if (observed === undefined) throw new Error("計測モードなのに submit の GPU 時間が無い");
        if (unitNs !== undefined) {
          const ms = (ns: number): string => `${(ns * unitNs / 1e6).toFixed(1)} ms`;
          const maxMs = observed.maxNs * unitNs / 1e6;
          lines.push(
            `1 submit の GPU 時間の最大 ${
              ms(observed.maxNs)
            }（${observed.submits} 本・目安 ${SUBMIT_GPU_GUIDE_MS} ms ${
              maxMs <= SUBMIT_GPU_GUIDE_MS ? "以内" : "を越えた"
            }）`,
            `裏付け前 ${observed.unbackedSubmits} 本の最大 ${
              observed.maxUnbackedNs === undefined ? "—" : ms(observed.maxUnbackedNs)
            }`,
            `単発 dispatch の最大 ${ms(observed.maxDispatchNs)}（${
              observed.maxDispatchKey ?? "—"
            }）`,
          );
        }
        lines.push(`submit ${submit.submitCount} 本・dispatch ${submit.dispatchCount} 本`);
      } finally {
        await session.dispose();
        await settleReleases(gpu);
      }
      assertEquals(deviceLost, undefined, "device lost");
      const note = lines.join(" / ");
      console.log(`[wan-umt5] 資源（計測モード）: ${note}`);
      await runRecordedCase(
        results,
        { id: "umt5/resources/gpu-timing" },
        () => Promise.resolve({ status: "pass", note }),
      );
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

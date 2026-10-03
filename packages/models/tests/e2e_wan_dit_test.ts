/**
 * Wan2.1 の DiT（S 形グラフ）の**移植の門**（実 GPU・ADR 0118 段 2 — 決定 3 / 8）。
 *
 * export した IR（f16 席・`outputs/series/wan2.1-t2v-1.3b-f16-dyn/transformer/`）を karume runtime で
 * 回し、ホストの unpatchify（`src/wan/dit-tokens.ts`）を通した潜在を、上流の**素の** diffusers
 * `WanTransformer3DModel`（CPU f32・同じ f16 丸めの重み）の 1 forward と突き合わせる。配布形（段 7）を
 * 通す検証は `WanPipeline` の e2e（`e2e_wan_pipeline_test.ts`）が持ち、ここは DiT 単体の forward を実寸まで
 * 検証するので、系列の容器（`krm`）を直接開く（計測用の層別グラフも系列にしか無い）。
 *
 * 生成は `cd tools/export-recipes && uv run --group wan --inexact python -m wan.export_dit`（golden の
 * 中身とケースの表は同ファイルの docstring）。golden は 2 本ずつ:
 *
 * - `io.<case>` — グラフの入力（`input.*`）と、パッチ後の torch CPU の出力（`output.0`）
 * - `reference.<case>` — 上流の素の forward の入力の潜在・timestep・出力（`[1,16,F,H,W]`）・各ブロックの出力
 *
 * パッチ後の eager は全ケースで上流と**ビット一致**（export 台本の `[eager]` 行 — patch 埋め込みの Linear 化も
 * この機の torch CPU では一致した）なので、ここで見る差はそのまま「GPU で回した IR」と「上流」の差。
 *
 * ## 帯（決定 8・追記 2026-10-02 の「帯の決定と受入れを分ける」）
 *
 * 指標はケースごとの**比** = 最大絶対差 ÷ 参照の最大絶対値（決定 8 の目安の形）。{@link DIT_RATIO_BAND} は
 * `band` 6 ケースの最悪の比の約 5 倍で決め、`accept` 3 ケース（`band` と seed が違う未見の入力）で受け入れる。
 * `growth` 2 ケース（S = 768）は S に対する伸びの記録で、帯の門には入れない（実寸の帯は段 3 / 段 8 が S ごとに
 * 独立に導く — 決定 8）。故障注入 4 件（RoPE の h / w の取り違え・unpatchify の並びの取り違え・timestep の
 * cos / sin の前後反転・timestep の 1 ずれ）が受入れケースで帯の外へ出ることも門にする。
 *
 * MUST: `accept` の結果を見て `band` のケースを変えない（帯の決定と受入れの独立が崩れる — ケースの正本は
 * `wan/export_dit.py` の `CASES`）。受入れが帯を外れたら、帯を広げずに原因を調べる。
 *
 * 層数に対する伸びは計測用のグラフ（層別の出口 31 本 — `…-f16-dyn-probe/`・`export_dit --layers`）で各
 * ブロックの出力を上流の forward hook の値と突き合わせて記録する（門ではない）。
 *
 * ## 実寸（ADR 0118 段 3 / 段 8 — 832×480・33 フレームの S = 14,040 と 81 フレームの S = 32,760）
 *
 * 実寸の 1 forward を 3 本のテストで見る。指標は S = 192 と違い**正規化した比**（GPU の誤差 ÷ CPU f32 の参照
 * 自身の誤差 — {@link normalizedRatio}）で、帯 {@link DIT_FULL_NORMALIZED_BAND} は S ごとの表。どの S の帯も
 * 他の S の帯を持ち込まず、その S の `full-band` 6 ケースで独立に決め、`full-accept` 2 ケース（別 seed）で
 * 受け入れる（決定 8 の外挿）。誤差の基準は**活性も f64 で回した上流**（`reference.<case>` の `output.f64` —
 * 重みは同じ f16 丸め）。
 *
 * - 計測モード（`gpuTiming`）の照合: S ごとに Session を張り直し、行ブロックの枚数（device の束縛上限から導いた
 *   期待値 — {@link expectedRowBlocks}）と、1 submit ごとの GPU 時間の最大 ≤ {@link SUBMIT_GPU_LIMIT_MS}（最初の
 *   run の裏付け前のチャンクを含む — runtime の `ChunkBudgetStats.submitGpuTime`）を門にし、受入れケースで故障注入 4 件が帯の外へ出ること
 *   （timestep の 1 ずれは帯の {@link SUBTLE_FAULT_MARGIN} 倍以上）も見る。帯の門は max ベースの r で、
 *   p99.99 ベースの r（{@link quantileRatio}）は記録だけ（指標 r の弱点 — 最大は裾の 1 要素で決まる — を
 *   分布で見るため。ADR 0118 追記 2026-10-02「段 3 の結果 — 帯の床と指標 r の弱点」）。
 * - 層別: S = 192 / 768 と同じ計測用グラフで、各ブロックの比を記録する（門ではない・相手は f32 の参照の
 *   `block.NN` — f64 のブロック出力は golden に持たない）。ブロックの出力を golden に持つ 2 ケース
 *   （{@link FULL_CASES} の `blocks` — どちらも S = 14,040）だけで回す。S = 32,760 は出口 31 本の readback
 *   staging が約 6 GB になり B570 に入らないので置かない。
 * - 通常モード（計測なし）: S ごとに所要（壁時計）を記録する。計測モードは 1 dispatch = 1 pass で所要が
 *   変わりうる。出力は計測モードの同じケースの出力と Uint32 で一致することを門にする（チャンクの切れ目は
 *   値を動かさない — {@link measuredTokens}）。
 *
 * ## i8 系列（ADR 0120 段 3 / 段 4 — `outputs/series/wan2.1-t2v-1.3b-i8-dyn/transformer/`）
 *
 * 同じ台本の `--dtype i8` が書く系列（linear 307 本が i8 per-channel・bias / norm は上流の f32）を、系列 root を
 * 切り替えて同じ門で見る。
 *
 * - **参照席 `f16+dit8`（`session` 空）の r 門**（段 3）: f16 席の実寸と同じ 1 本（{@link normalizedRatioGate}）を、
 *   S = 192 の組（9 本 — S = 128 の受入れ 1 本を含む）と S = 14,040（8 本・計測モード）で回す。i8 系列は全ケースが
 *   f64 の参照（`output.f64` — i8 の fake-quant 重みで採った値）を持つので、S = 192 の組も r で見る。帯は組ごとに
 *   決定用 6 本の最悪 × 5（{@link I8_NORMALIZED_BAND} — 未導出の間は r と帯の候補を出して赤）。故障注入は f16 席と
 *   同じ 4 件に、i8 に固有の 1 件（linear 1 本の scale を 2 倍 — {@link scaledScaleContainer}）を足す。S = 32,760 の
 *   golden は持たない（ADR 0120 決定 3）。io の入力は f16 系列とバイト同一なのでケース名は共通。
 * - **実用席 `f16+dit8-a8-attn8-s16` の時間と確保**（段 4 — 記録 + 計測モードの 1 submit の門）: 束は配布形の
 *   manifest の席から引く（束の正本は manifest — ADR 0110 決定 1）。入力は f16 系列の `io.*`（S = 32,760 は f16 系列に
 *   しか無い・入力は席に依らない）。数値の門は自機 A/B 門（`e2e_wan_ab_test.ts`）が持ち、ここでは見ない。
 *
 * 資産が無い環境と GPU 無し環境は生成コマンド付きで**明示 SKIP**する（ADR 0005）。資産が**一部だけ**ある
 * 環境は SKIP ではなく FAIL にする（下の完全性テスト）。
 */

import { assert, assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import {
  acquireGpu,
  type BoundContainer,
  type FusionCounts,
  type GpuContext,
  type OpenedContainer,
  parseSafetensors,
  prepareContainer,
  type PreparedModel,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type SessionOptions,
  type Tensor,
} from "@karume/runtime";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
  wanTokenWidth,
} from "../src/wan/dit-tokens.ts";
import { parseWanRopeBase, type WanRopeBase, wanRopeTables } from "../src/wan/dit-rope.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { quantOf } from "./helpers/ab-gate.ts";
import { effectiveSessionOptions } from "./helpers/census-table.ts";
import { formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { timestampUnitNs } from "./helpers/timestamp-unit.ts";
import { modelPresent, openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { seriesGraph } from "../../runtime/tests/helpers/series-graphs.ts";
import {
  assertAdapterMatchesEnvironment,
  ENVIRONMENT,
} from "../../runtime/tests/helpers/environment.ts";
import {
  type Measurement,
  openResults,
  runRecordedCase,
} from "../../runtime/tests/helpers/results.ts";

/**
 * GPU（B570・f16 席）の DiT 1 forward と上流（CPU f32）の差の許容（比 = 最大絶対差 ÷ 参照の最大絶対値・
 * unpatchify 後の潜在）。
 *
 * 実測（2026-10-02・`deno-intel-graphics-bmg-g21`・`atol = rtol = 0` の素の突合）:
 *
 * | 役割   | ケース                | 格子 F'·H'·W' | 最大絶対差 | 参照の最大絶対値 | 比（io）     | 比（TS の proj） |
 * | ------ | --------------------- | ------------- | ---------- | ---------------- | ------------ | ---------------- |
 * | band   | `band-s00192-t0999`   | 3·8·8         | 2.694e-5   | 4.024            | 6.70e-6      | 9.30e-6          |
 * | band   | `band-s00192-t0750`   | 2·8·12        | 5.460e-5   | 5.468            | 9.98e-6      | 1.20e-5          |
 * | band   | `band-s00192-t0500`   | 3·8·8         | 6.363e-5   | 3.569            | 1.78e-5      | 1.67e-5          |
 * | band   | `band-s00192-t0250`   | 1·8·24        | 2.694e-5   | 5.321            | 5.06e-6      | 4.70e-6          |
 * | band   | `band-s00192-t0600`   | 2·8·12        | 5.406e-5   | 5.594            | 9.66e-6      | 1.03e-5          |
 * | band   | `band-s00192-t0113`   | 3·8·8         | 8.273e-5   | 4.321            | **1.91e-5**  | 1.62e-5          |
 * | accept | `accept-s00192-t0600` | 2·8·12        | 3.247e-4   | 5.242            | 6.19e-5      | 4.81e-5          |
 * | accept | `accept-s00128-t0030` | 4·4·8         | 3.101e-5   | 4.784            | 6.48e-6      | 5.97e-6          |
 * | accept | `accept-s00192-t0400` | 2·12·8        | 6.151e-5   | 6.234            | 9.87e-6      | 9.87e-6          |
 * | growth | `growth-s00768-t0999` | 3·16·16       | 5.892e-5   | 4.426            | 1.33e-5      | 1.30e-5          |
 * | growth | `growth-s00768-t0500` | 3·16·16       | 4.804e-5   | 4.874            | 9.86e-6      | 8.98e-6          |
 *
 * 「比（io）」は golden の torch の `timesteps_proj` を入れた値（最大絶対差の列もこちら）、「比（TS の proj）」は
 * ホストの `timestepsProj` を入れた値。
 *
 * 受入れケースの故障注入の比（RoPE の h / w・unpatchify の並び・cos / sin 反転・**timestep の 1 ずれ**）:
 * `accept-s00192-t0600` 4.03e-1 / 1.28 / 2.88e-1 / **9.18e-3**・`accept-s00128-t0030` 1.06e-1 / 1.43 / 3.95e-1 /
 * **6.91e-3**・`accept-s00192-t0400` 2.48e-1 / 1.32 / 2.79e-1 / **1.93e-3**。
 *
 * 帯 9.6e-5 は band の最悪の比 1.915e-5（`band-s00192-t0113` の io）の約 5.0 倍。受入れの最悪 6.19e-5 は帯の
 * 0.64 倍。微妙な故障（timestep の 1 ずれ）の最小 1.93e-3 は帯の約 20 倍で、O(1) の故障 3 件は 3〜4 桁上に出る。
 *
 * 絶対差でなく比で見る理由: 旧帯（絶対 atol 3.2e-4 = 当時の決定用 4 ケースの最悪 6.36e-5 × 5）を、検証の未見
 * ケース（今の `accept-s00192-t0600`）が 3.247e-4 で超えた。超えたのは 12,288 要素中 1 要素で、比は 6.2e-5。
 * 最大絶対差は裾の 1 要素で決まり参照の振れ幅（3.6〜6.2）ごと動くので、振れ幅で割った比の方が入力をまたいで
 * 揃う。しかも当時は決定用のケースを受入れの結果を見て足していたので、決定と受入れの独立も崩れていた。
 *
 * 決定用を 1 ケースにしない理由（実測）: t = 999 の 1 ケースで決めた帯を、t = 500 の未見ケースが 1.6 倍
 * 超えた。移植の誤りではなく入力の感度で、根拠は 2 つ:
 * ① パッチ後の eager は上流とビット一致（GPU と eager の差 = GPU と上流の差 — export 台本の `[eager]` 行）。
 * ② CPU の eager でも、中ほどの timestep は `timesteps_proj` の 1e-5 級の揺れを出力で約 4 倍に増幅する
 *    （t = 999 は 0.2 倍）。GPU の丸めの差も同じ増幅を受ける。
 * そこで決定用は timestep を参照の設定の列（999 → 60）の全域に散らし、seed と格子も変えた 6 ケースにした。
 *
 * 層の向き（計測用グラフ・各ブロックの出力の 最大絶対差 / 参照の最大絶対値・11 ケース × 30 ブロック）: 6e-6〜
 * 1.6e-4 の範囲で上下し、層数に対して単調には伸びない（ケースごとの最大は `band-s00192-t0500` の 18 層目
 * 1.54e-4・`growth-s00768-t0999` の 25 層目 1.25e-4 など。多くは後ろの層で 1e-5 級へ戻る）。S = 192 → 768
 * でも最終出力の比は伸びていない。
 */
const DIT_RATIO_BAND = 9.6e-5;

/** Wan2.1 T2V 1.3B の patch（`(1,2,2)`・潜在 16 チャネル — transformer の config）。 */
const WAN_GEOMETRY: WanPatchGeometry = {
  channels: 16,
  patchFrames: 1,
  patchHeight: 2,
  patchWidth: 2,
};

const SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn";
const PROBE_SERIES_NAME = "wan2.1-t2v-1.3b-f16-dyn-probe";
const COMPONENT = "transformer";
const SERIES_DIR = new URL(
  `../../../outputs/series/${SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);
const PROBE_DIR = new URL(
  `../../../outputs/series/${PROBE_SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);
/** 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本 — 門番と同じ正本から引く）。 */
const GRAPH = seriesGraph(SERIES_NAME, COMPONENT);
const PROBE_GRAPH = seriesGraph(PROBE_SERIES_NAME, COMPONENT);
const MODEL_FILE = "model.krm";
/** 資産の名前（`wan/export_dit.py` の `ROPE_BASE_ASSET`）。 */
const ROPE_BASE_ASSET = "rope_base";

const GENERATE = "cd tools/export-recipes && uv run --group wan --inexact python -m wan.export_dit";

type CaseRole = "band" | "accept" | "growth";

/**
 * 生成されているはずのケース。**列挙結果ではなくここで固定する**（生成を一部だけ流した環境でテストが
 * 黙って消える形にしない）。正本は `wan/export_dit.py` の `CASES`。
 */
const CASES: readonly { readonly name: string; readonly role: CaseRole }[] = [
  { name: "band-s00192-t0999", role: "band" },
  { name: "band-s00192-t0750", role: "band" },
  { name: "band-s00192-t0500", role: "band" },
  { name: "band-s00192-t0250", role: "band" },
  { name: "band-s00192-t0600", role: "band" },
  { name: "band-s00192-t0113", role: "band" },
  { name: "accept-s00192-t0600", role: "accept" },
  { name: "accept-s00128-t0030", role: "accept" },
  { name: "accept-s00192-t0400", role: "accept" },
  { name: "growth-s00768-t0999", role: "growth" },
  { name: "growth-s00768-t0500", role: "growth" },
];

type FullRole = "full-band" | "full-accept";

/**
 * 実寸の S（潜在 `[16,F',60,104]` → S = F'·30·52）: 33 フレーム（F' = 9・ADR 0118 段 3）の 14,040 と
 * 81 フレーム（F' = 21・段 8）の 32,760。
 */
type FullTokens = 14040 | 32760;

/**
 * GPU テストで回す S の順。確保が最大の S = 32,760 を先に置く（理由は S = 192 より前に実寸を置くのと同じ —
 * B570 では前の確保の残りが後ろの大きな確保を OOM にしうるので、大きい方を空の device で回す）。
 */
const FULL_TOKENS_ORDER: readonly FullTokens[] = [32760, 14040];

/**
 * 実寸のケース。正本は `wan/export_dit.py` の `CASES`。S ごとに `full-band` 6 本は
 * {@link DIT_FULL_NORMALIZED_BAND} のその S の行の決定用で timestep を 999〈seed 2 本〉/ 750 / 500 / 250 / 113 に
 * 散らし、`full-accept` 2 本は受入れ用（seed は段 2 のどのケースとも、決定用と受入れの間でも別）。`blocks` は
 * 各ブロックの出力（`block.NN` — 1 ケース 2.6 GB）を golden に持つケース（`CaseSpec.blocks` — 層ごとの記録は
 * この 2 本だけで回す）。並びは {@link FULL_TOKENS_ORDER} と同じく S = 32,760 が先。
 *
 * MUST: 受入れの結果を見て決定用のケースを足し引きしない（帯の決定と受入れの独立が崩れる）。
 */
const FULL_CASES: readonly {
  readonly name: string;
  readonly tokens: FullTokens;
  readonly role: FullRole;
  readonly blocks: boolean;
}[] = [
  { name: "full-band-s32760-t0999", tokens: 32760, role: "full-band", blocks: false },
  { name: "full-band-s32760-t0999-2", tokens: 32760, role: "full-band", blocks: false },
  { name: "full-band-s32760-t0750", tokens: 32760, role: "full-band", blocks: false },
  { name: "full-band-s32760-t0500", tokens: 32760, role: "full-band", blocks: false },
  { name: "full-band-s32760-t0250", tokens: 32760, role: "full-band", blocks: false },
  { name: "full-band-s32760-t0113", tokens: 32760, role: "full-band", blocks: false },
  { name: "full-accept-s32760-t0999", tokens: 32760, role: "full-accept", blocks: false },
  { name: "full-accept-s32760-t0600", tokens: 32760, role: "full-accept", blocks: false },
  { name: "full-band-s14040-t0999", tokens: 14040, role: "full-band", blocks: true },
  { name: "full-band-s14040-t0999-2", tokens: 14040, role: "full-band", blocks: false },
  { name: "full-band-s14040-t0750", tokens: 14040, role: "full-band", blocks: false },
  { name: "full-band-s14040-t0500", tokens: 14040, role: "full-band", blocks: false },
  { name: "full-band-s14040-t0250", tokens: 14040, role: "full-band", blocks: false },
  { name: "full-band-s14040-t0113", tokens: 14040, role: "full-band", blocks: false },
  { name: "full-accept-s14040-t0999", tokens: 14040, role: "full-accept", blocks: true },
  { name: "full-accept-s14040-t0600", tokens: 14040, role: "full-accept", blocks: false },
];

/**
 * S ごとのケース（{@link FULL_TOKENS_ORDER} の順）。`normalModeCase` は通常モードで回す 1 本（S ごとの先頭）。
 */
const FULL_CASES_BY_TOKENS = FULL_TOKENS_ORDER.map((tokens) => {
  const cases = FULL_CASES.filter((spec) => spec.tokens === tokens);
  return { tokens, cases, normalModeCase: cases[0].name };
});

/**
 * 実寸のケースの `reference.<case>` が持つ f64 の参照（上流の素の forward を活性も f64 で回した出力を f32 へ
 * 丸めた値 — `wan/export_dit.py` の `REFERENCE_F64_KEY`）。
 *
 * f32 へ丸めた差は要素ごとに 2⁻²⁴·|x| 以下で、比にして 6e-8 以下。これが正規化の分母（CPU f32 の参照の f64 に
 * 対する比 — {@link referenceErrorOf}）を動かす割合は分母の最小で決まる（実測は
 * {@link DIT_FULL_NORMALIZED_BAND} の表と results.json）:
 *
 * - 決定用（`full-band`）だけ: S = 14,040 の最小 3.72e-6（`t0250`）で 2% 未満・S = 32,760 の最小 2.07e-6
 *   （`t0999-2`）で約 2.9%
 * - 受入れを含めると: 最小 1.83e-6（`full-accept-s14040-t0600`）で約 3.3%
 *
 * r が数 % 動いても判定の向きは変わらない（判定が帯と最も近いのは S = 32,760 の t0600 の timestep の 1 ずれで
 * 帯の 2.65 倍。そのケースの分母は 6.77e-6 なので、動きは 1% 未満）。
 */
const REFERENCE_F64_KEY = "output.f64";

/**
 * 実寸の GPU（B570・f16 席）の 1 forward の誤差の許容（S ごとの表）。指標は**正規化した比**
 * （{@link normalizedRatio}）:
 *
 *   r = (max|GPU − f64| ÷ max|f64|) ÷ (max|CPU f32 − f64| ÷ max|f64|)
 *
 * = 「GPU の誤差が、同じ入力で CPU f32 の参照（上流の素の forward・CPU torch f32）自身が出す誤差の何倍か」。
 * f64 は活性も f64 で回した上流（重みは同じ f16 丸め — `output.f64`）、CPU f32 は `output`。どの S の帯も S = 192
 * や他の S の帯を持ち込まず、その S の `full-band` 6 ケースの最悪の r × 5 で独立に決める（ADR 0118 決定 8 の外挿）。
 *
 * S = 32,760 の実測（2026-10-03・`deno-intel-graphics-bmg-g21`・ADR 0118 段 8）。帯の式（`full-band` 6 本の最悪
 * r × 5）は実測の前に ADR の追記で固定し、`full-accept` の r は同じ走行で出たが帯の決定には使っていない:
 *
 * | ケース                     | GPU / f64 | CPU f32 / f64 | r        |
 * | -------------------------- | --------- | ------------- | -------- |
 * | `full-band-s32760-t0999`   | 4.81e-5   | 2.32e-6       | **20.8** |
 * | `full-band-s32760-t0999-2` | 2.53e-5   | 2.07e-6       | 12.2     |
 * | `full-band-s32760-t0750`   | 5.70e-5   | 9.22e-6       | 6.18     |
 * | `full-band-s32760-t0500`   | 1.84e-5   | 4.48e-6       | 4.11     |
 * | `full-band-s32760-t0250`   | 2.30e-5   | 4.14e-6       | 5.56     |
 * | `full-band-s32760-t0113`   | 3.10e-5   | 7.86e-6       | 3.94     |
 *
 * 帯 104 = 最悪 20.8（`full-band-s32760-t0999`）× 5。受入れ 2 本は r 13.1（t0999）/ 4.89（t0600）で帯の内。故障注入の
 * r（RoPE の h / w・unpatchify の並び・cos / sin 反転・timestep の 1 ずれ）は t0999 で 5.65e4 / 3.44e5 / 1.72e5 /
 * 1.91e4、t0600 で 3.07e4 / 2.12e5 / 2.12e4 / **276**（帯の 2.65 倍 ≥ {@link SUBTLE_FAULT_MARGIN}）。4 件とも帯の外。
 * S = 14,040 より r の最悪が大きい（14.9 → 20.8）のは t = 999 の 1 本で、他の 5 本は同程度（3.9〜12.2）。
 *
 * S = 14,040 の実測（2026-10-02・`deno-intel-graphics-bmg-g21`・`atol = rtol = 0` の素の突合）。帯は受入れ 2 ケース
 * （seed 777006 / 777007）を生成する前に、決定用 6 ケースだけから決めた:
 *
 * | ケース                     | GPU / f64  | CPU f32 / f64 | GPU / CPU f32 | r          |
 * | -------------------------- | ---------- | ------------- | ------------- | ---------- |
 * | `full-band-s14040-t0999`   | 2.120e-3   | 1.421e-4      | 1.978e-3      | **14.9**   |
 * | `full-band-s14040-t0999-2` | 4.053e-5   | 5.562e-6      | 3.908e-5      | 7.29       |
 * | `full-band-s14040-t0750`   | 6.837e-5   | 1.332e-5      | 7.142e-5      | 5.13       |
 * | `full-band-s14040-t0500`   | 1.673e-5   | 3.769e-6      | 1.596e-5      | 4.44       |
 * | `full-band-s14040-t0250`   | 9.844e-6   | 3.724e-6      | 9.486e-6      | 2.64       |
 * | `full-band-s14040-t0113`   | 8.921e-6   | 5.592e-6      | 9.277e-6      | 1.60       |
 *
 * 各列は「最大絶対差 ÷ 参照の最大絶対値」（左 3 列の参照はそれぞれ f64 / f64 / CPU f32）。帯 75 = 最悪 14.92
 * （`full-band-s14040-t0999`）× 5 = 74.6 を有効数字 2 桁へ切り上げた値。
 *
 * 正規化する理由: 比そのもの（GPU / f64）は入力で 240 倍動く（t = 113 の 8.9e-6 〜 t = 999 の 2.1e-3・同じ
 * t = 999 でも seed で 52 倍）。動いているのは入力による丸めの増幅で、CPU f32 の参照も同じだけ f64 から離れる
 * （同じ入力なら f32 のどの実装にも掛かる）。比の最悪 × 5 で帯を決めると 1.1e-2 になり、微妙な故障（timestep の
 * 1 ずれ — t = 600 で比 1.77e-3）まで帯の内に入った。CPU f32 の誤差で割れば増幅が打ち消され、r は 1.6〜15 に
 * 収まる（比の 240 倍の幅が 9 倍になる）。
 *
 * r が 1 を超える理由（2026-10-02 の帰属・誤りではなく精度差）: 層ごとの比の伸びの形は GPU と CPU f32 で同じで、
 * 峰も同じ層。op ごとの局所誤差（ブロック 0 / 28 の op の出口を GPU の値を入力にして f64 と比べた値・
 * `full-band-s14040-t0999` / `t0113`）では linear（GEMM）が CPU f32 の 4〜7 倍、FFN（down は K = 8960）が 11〜14 倍。
 * GEMM の K 縮約（K 昇順・1 本の f32 累積 — src/kernels/gemm.ts）を CPU で逐次の fma として再現すると GPU の出力と
 * ビット一致する（block 0 / 28 の to_v・to_out・512 行）。norm・残差は 1 倍で、block 28 の SDPA は GPU の方が
 * 小さい（0.8 倍）。手当て（縮約順の変更は全系列の sha256 と生成物に響く）は別の裁定。
 *
 * MUST: `full-accept` の結果を見てこの値も、`full-band` のケースも、指標も変えない（受入れの seed は指標を
 * 決めた後に新しくした — `wan/export_dit.py` の docstring）。受入れが帯を外れたら、帯を広げずに原因を調べる。
 *
 * NOTE（2026-10-02・受入れの結果 — 帯を決めた後に生成して回した）:
 *
 * | ケース                     | GPU / f64 | CPU f32 / f64 | GPU / CPU f32 | r    | r（TS の proj） |
 * | -------------------------- | --------- | ------------- | ------------- | ---- | --------------- |
 * | `full-accept-s14040-t0999` | 2.00e-5   | 6.82e-6       | 2.49e-5       | 2.93 | 3.27            |
 * | `full-accept-s14040-t0600` | 8.45e-6   | 1.83e-6       | 8.70e-6       | 4.60 | 4.50            |
 *
 * どちらも帯の内。故障注入の r（RoPE の h / w・unpatchify の並び・cos / sin 反転・timestep の 1 ずれ）は t0999 で
 * 3.25e4 / 2.05e5 / 6.44e4 / 6.04e3（timestep の 1 ずれは帯の 80.5 倍）、t0600 で 3.24e4 / 6.65e5 / 3.62e4 / **332**
 * （比 6.09e-4・帯の **4.42 倍**）。4 件とも帯の外。timestep の 1 ずれの余裕は最小 4.42 倍で、
 * {@link SUBTLE_FAULT_MARGIN}（2）を満たす。帯も指標も決定用も動かしていない。
 */
const DIT_FULL_NORMALIZED_BAND: Readonly<Record<FullTokens, number>> = { 14040: 75, 32760: 104 };

/**
 * 微妙な故障（timestep の 1 ずれ）が帯から離れているべき倍率（追記 2026-10-02 の「赤にならない注入は帯が広すぎる
 * 兆候」の実装）。帯の外に出るだけでは、帯の最悪 × 5 の余裕と故障の大きさが同じ桁になったときに気づけない。
 * 値は 2: 故障の大きさ自体が入力で 3 倍近く動く（t=600 の 2 seed で 1.77e-3 と 6.09e-4）ので、観測の最小 4.42 倍
 * （受入れ t0600）と 9.4 倍（独立検証の seed 888123）・80 倍（t0999）を下回る 2 倍を、帯の広がりを検出する床とする。
 * 最初に置いた 5 は ADR に無い見積りで、観測の後に 2 へ下げた（開示 — 帯・指標・決定用は動かしていない）。
 */
const SUBTLE_FAULT_MARGIN = 2;

/** i8 系列（参照席 `f16+dit8` / 実用席の重み — ADR 0120 決定 2）。 */
const I8_SERIES_NAME = "wan2.1-t2v-1.3b-i8-dyn";
const I8_SERIES_DIR = new URL(
  `../../../outputs/series/${I8_SERIES_NAME}/${COMPONENT}/`,
  import.meta.url,
);
const I8_GENERATE = `${GENERATE} --dtype i8`;
/** i8 の参照席のケースの結果 ID と step 名の接頭辞（ケース名は f16 系列と共通なので席で分ける）。 */
const I8_ID_PREFIX = "f16+dit8/";

/**
 * 容器の中のグラフ名（表は helpers/series-graphs.ts の 1 本）。**使うときに引く** — 表に行が無いときに
 * モジュール評価で落とすと、f16 席の門まで全部が巻き添えで落ちる。
 */
const i8Graph = (): string => seriesGraph(I8_SERIES_NAME, COMPONENT);

/**
 * i8 系列のケース（`wan/export_dit.py` の `series_cases("i8")` の鏡像 — S = 192 の組は f16 系列の `band` /
 * `accept`〈S = 128 の受入れ 1 本を含む〉、実寸は S = 14,040 の 8 本。`growth` と S = 32,760 は採らない — ADR 0120
 * 決定 3）。seed・timestep・有効長は f16 系列と同じ。
 */
const I8_SMALL_CASES: readonly GateCase[] = CASES.filter(({ role }) => role !== "growth").map(
  ({ name, role }) => ({ name, role, accept: role === "accept" }),
);
const I8_FULL_CASES: readonly GateCase[] = FULL_CASES.filter(({ tokens }) => tokens === 14040)
  .map(({ name, role }) => ({ name, role, accept: role === "full-accept" }));

/** i8 の参照席の組（S = 192 の組は S = 128 の受入れ 1 本を含む）。 */
type I8Group = 192 | 14040;

/**
 * 参照席 `f16+dit8`（i8 系列・`session` 空）の正規化した比 r の帯（組ごと — ADR 0120 決定 3）。導出規則は f16 席の
 * {@link DIT_FULL_NORMALIZED_BAND} と同じで、その組の決定用 6 本の最悪 r × 5。組どうし・f16 席の帯は持ち込まない
 * （ADR 0118 決定 8 の外挿）。S = 192 の組も r で見る — i8 系列は全ケースが f64 の参照を持つので、f16 席の S = 192 の
 * 比の帯（{@link DIT_RATIO_BAND}）とは指標が違う。
 *
 * `undefined` = 未導出: 門は各ケースの r と帯の候補（{@link bandCandidate}）を出して**赤で止まる**（通る値を
 * 仮置きして検出力の無い門を作らない）。GPU の実測の後に、実測の表と一緒にここへ書く。
 *
 * MUST: 受入れの結果を見てこの値も決定用のケースも変えない（f16 席と同じ）。受入れが帯を外れたら、帯を広げずに
 * 原因を調べる。
 *
 * 導出（2026-10-03・B570・計測モード〈S = 14,040〉/ 通常の device〈S = 192〉）:
 *
 * | 組      | 決定用の r（t0999 / t0999-2 / t0750 / t0500 / t0250 / t0113） | 最悪  | × 5   | 帯 | 受入れの r              |
 * | ------- | ---------------------------------------------------------------- | ----- | ----- | -- | ----------------------- |
 * | 14,040  | 10.7 / 10.2 / 4.02 / 2.90 / 2.56 / 2.05                          | 10.7  | 53.5  | 54 | t0999 3.71 / t0600 4.64 |
 * | 192     | 2.59 / 2.83 / 3.66 / 4.12 / 3.30 + t0600 5.97                    | 5.97  | 29.9  | 30 | t0600 3.82 / t0400 3.43 |
 *
 * scale × 2 の故障注入は r 1.66e+4〜4.45e+4（帯の 3 桁上）。f16 席の同じ S の帯は 75（S = 14,040）で、i8 の t0999 の r が
 * f16（14.9）より小さいのは分母（CPU f32 の誤差）が同じ桁で分子の最大が小さく出たため — 席の優劣ではない。
 */
const I8_NORMALIZED_BAND: Readonly<Record<I8Group, number | undefined>> = {
  192: 30,
  14040: 54,
};

/**
 * i8 に固有の故障注入で per-channel scale を 2 倍にする linear（{@link scaledScaleContainer}）。中ほどのブロックの
 * self-attention の V を選ぶ — q / k は直後の RMS norm（`norm_q` / `norm_k`）が倍率を打ち消して値に出ない。
 * 主張は「i8 の scale の取り違え（軸・値・束縛）を r の帯が掴める」こと（ADR 0120 検収 段 3）。
 */
const I8_SCALE_FAULT_WEIGHT = "blocks.15.attn1.to_v.weight";

/** 配布形ミラー（実用席の束の正本 — `dist.py --pipeline wan` の出力先）。 */
const DIST_ROOT = new URL("../../../models/karume-wan2.1/", import.meta.url);
/** 配布形の Wan のモデル名（manifest の models のキー）。 */
const DIST_MODEL = "t2v-1.3b";
/** 実用席（ADR 0120 決定 1）。 */
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";

/**
 * 実用席の時間と確保を採る入力（S ごとに 1 本・f16 系列の `io.*`）。値は時間に効かないので、各ブロックの出力を
 * golden に持たない軽いケースを選ぶ（`full-band-s14040-t0999` の reference は 2.6 GB）。並びは確保が最大の
 * S = 32,760 を先に置く（{@link FULL_TOKENS_ORDER} と同じ理由）。
 */
const PRACTICAL_TIMING_CASES: readonly { readonly tokens: FullTokens; readonly name: string }[] = [
  { tokens: 32760, name: "full-band-s32760-t0999" },
  { tokens: 14040, name: "full-band-s14040-t0999-2" },
];

/**
 * 1 submit の GPU 実行の幅の上限（ADR 0118 決定 6 と追記 2026-10-02 — Linux xe の `job_timeout_ms`
 * 5,000 ms の 1/5。門は窓平均ではなく 1 本ずつの最大で、最初の run の裏付け前のチャンクも含む）。
 */
const SUBMIT_GPU_LIMIT_MS = 1000;

/** Wan2.1 T2V 1.3B の attention の head 数（transformer の config `num_attention_heads`）。 */
const WAN_HEADS = 12;
/** スコア行列 S の 1 要素の格納幅（既定の f32 格納）。 */
const SCORE_BYTES = 4;

/**
 * 実寸の self-attention の行ブロック枚数の期待値（ADR 0118 決定 6）。行ブロック（ADR 0060）はクエリ行を
 * 「1 枚のスコア S が束縛上限に収まる最小枚数」へ等分するので、device の束縛上限
 * （`maxStorageBufferBindingSize`）だけから runtime と独立に導ける:
 *
 *   1 枚の行数 = ⌊上限 ÷ (heads × S × 4 B)⌋・枚数 = ⌈S ÷ 1 枚の行数⌉
 *
 * B570（束縛上限 {@link B570_STORAGE_BINDING_LIMIT}）では S = 14,040 が 5 枚（等分で 2,808 行）・S = 32,760 が
 * 24 枚 × 1,365 行（決定 6 の表 — 下の GPU 不要のテストがこの式で再現する）。
 */
const expectedRowBlocks = (tokens: number, bindingLimit: number): number =>
  Math.ceil(tokens / Math.floor(bindingLimit / (WAN_HEADS * tokens * SCORE_BYTES)));

/** B570 の `maxStorageBufferBindingSize`（ADR 0118 決定 6 の表の前提）。 */
const B570_STORAGE_BINDING_LIMIT = 2_147_483_644;

/**
 * この IR で計画時に掛かる融合（実測 — `lastRunFusions`）。MUST: 値が動いたら赤にする（融合は
 * エクスポータのノード順 1 つで黙って外れ、値は正しいまま性能だけが変わる — 唯一の観測点）。
 *
 * - `adaln` 0: Wan の変調は `layer_norm → add(scale, one) → mul → add(shift)` で計算 4 本の並びはルールと
 *   同じだが、ルールは layer_norm の直後に reshape 2〜3 本（変調ベクトルの unsqueeze）を要求する。Wan は
 *   変調ベクトルを layer_norm の**前**に表から切り出すので、間に reshape が 0 本で掴まれない（ADR 0118
 *   決定 3 の推測どおり。ルールの拡張は perf の別起票）。
 * - `rope` 0: 融合ルールは half-split 形だけを掴む。Wan は interleave 形（実数化パッチ — 非融合の
 *   primitive 列で走る）。
 * - `silu` 2: 時刻埋め込みの MLP の SiLU 2 本（`time_embedder` の中と `act_fn`）。
 */
const EXPECTED_FUSIONS: Partial<FusionCounts> = { adaln: 0, rope: 0, silu: 2 };

const readBuffer = async (url: URL): Promise<ArrayBuffer> => {
  const bytes = await Deno.readFile(url);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
};

/** ファイルの有無（NotFound 以外は伝播させる — 権限エラーを「資産が無い」に読み替えない）。 */
const filePresent = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

/** golden 2 本（`io.<case>` と `reference.<case>`）。`dir` は系列の部品ディレクトリ（既定は f16 系列）。 */
const caseFiles = (name: string, dir: URL = SERIES_DIR): readonly URL[] => [
  new URL(`io.${name}.safetensors`, dir),
  new URL(`reference.${name}.safetensors`, dir),
];

// 実寸のケースも同じ台本が同じ回に書く（一部だけある環境は完全性テストで FAIL にする）。
const expectedFiles = [...CASES, ...FULL_CASES].flatMap(({ name }) => caseFiles(name));
const MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, SERIES_DIR));
const presentFiles = expectedFiles.filter(filePresent);
const ASSETS_AVAILABLE = MODEL_PRESENT && presentFiles.length === expectedFiles.length;
const ANY_PRESENT = MODEL_PRESENT || presentFiles.length > 0;
const PROBE_AVAILABLE = ASSETS_AVAILABLE && modelPresent(new URL(MODEL_FILE, PROBE_DIR));

if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${SERIES_DIR.pathname} に DiT の容器と golden が揃っていないため Wan の DiT の e2e を ` +
      `SKIP する（重み 2.8GB につきリポジトリ管理外）。生成: ${GENERATE}`,
  );
} else if (!PROBE_AVAILABLE) {
  console.warn(
    `[karume] ${PROBE_DIR.pathname} に計測用の層別グラフが無いため層ごとの記録を SKIP する。` +
      `生成: ${GENERATE} --layers`,
  );
}

// i8 系列も S = 192 の組と S = 14,040 の組を同じ台本が同じ回に書く。`--no-full` の書き出し（S = 14,040 を除く）は
// 段の分割用の途中の席なので、揃うまで完全性テストが FAIL にする。r 門は組ごとに揃った組だけを回す（f16 系列の
// probe と同じ扱い — 欠けは完全性テストの FAIL が言う）。
const i8FilesOf = (cases: readonly GateCase[]): readonly URL[] =>
  cases.flatMap(({ name }) => caseFiles(name, I8_SERIES_DIR));
const i8ExpectedFiles = i8FilesOf([...I8_SMALL_CASES, ...I8_FULL_CASES]);
const I8_MODEL_PRESENT = modelPresent(new URL(MODEL_FILE, I8_SERIES_DIR));
const I8_ANY_PRESENT = I8_MODEL_PRESENT || i8ExpectedFiles.some(filePresent);
const I8_SMALL_AVAILABLE = I8_MODEL_PRESENT && i8FilesOf(I8_SMALL_CASES).every(filePresent);
const I8_FULL_AVAILABLE = I8_MODEL_PRESENT && i8FilesOf(I8_FULL_CASES).every(filePresent);

for (
  const [available, group] of [
    [I8_SMALL_AVAILABLE, "S = 192 の組"],
    [I8_FULL_AVAILABLE, "S = 14,040"],
  ] as const
) {
  if (available) continue;
  console.warn(
    `[karume] ${I8_SERIES_DIR.pathname} に i8 の DiT の容器と ${group} の golden が揃っていないため、参照席 ` +
      `f16+dit8 の ${group} の r 門を SKIP する。生成: ${I8_GENERATE}`,
  );
}

/** 配布形の manifest（実用席の束の正本 — 無ければ実用席の時間の記録を SKIP する）。 */
const distManifestText = await readTextIfPresent(new URL("karume.json", DIST_ROOT));
/** 実用席の時間を採れる（入力 = f16 系列の golden・重み = i8 系列・束 = 配布形の manifest）。 */
const PRACTICAL_RUNNABLE = ASSETS_AVAILABLE && I8_MODEL_PRESENT && distManifestText !== undefined;

if (ASSETS_AVAILABLE && I8_MODEL_PRESENT && distManifestText === undefined) {
  console.warn(
    `[karume] 配布形ミラー ${DIST_ROOT.pathname} の karume.json が無いため実用席 ${PRACTICAL_QUANT} の時間と` +
      "確保の記録を SKIP する（組み立て: cd tools/export-recipes && uv run python dist.py --pipeline wan）",
  );
}

/**
 * 実用席の実効 SessionOptions（配布形の manifest の席の `session` を家族の受理表で通した値 — 束の正本は
 * manifest の quant 席だけ・ADR 0110 決定 1。テストに束の値を書き写さない）。
 */
const practicalSessionOptions = (): SessionOptions => {
  if (distManifestText === undefined) {
    throw new Error(`${DIST_ROOT.pathname}karume.json が無い（ignore の条件と食い違っている）`);
  }
  const where = `wan ${DIST_MODEL}/${PRACTICAL_QUANT}`;
  const declared = quantOf(parseManifest(distManifestText), DIST_MODEL, PRACTICAL_QUANT).session;
  return effectiveSessionOptions("wan", declared, {}, where);
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

/** 1 ケースぶんの golden（グラフ入力・上流の潜在形と timestep と出力）。 */
type Golden = {
  readonly io: SafetensorsFile;
  readonly reference: SafetensorsFile;
  readonly latentShape: readonly number[];
  readonly timestep: number;
};

const loadGolden = async (name: string, dir: URL = SERIES_DIR): Promise<Golden> => {
  const [ioUrl, referenceUrl] = caseFiles(name, dir);
  const io = parseSafetensors(await readBuffer(ioUrl));
  const reference = parseSafetensors(await readBuffer(referenceUrl));
  const latents = viewOf(reference, "latents", name);
  const timestep = viewOf(reference, "timestep", name);
  if (timestep.dtype !== "I32" || latents.shape.length !== 5 || latents.shape[0] !== 1) {
    throw new Error(`${name}: reference の latents / timestep の形が想定外`);
  }
  return {
    io,
    reference,
    // ホストの潜在はバッチ軸を持たない `[C,F,H,W]`（`src/wan/dit-tokens.ts`）。
    latentShape: latents.shape.slice(1),
    timestep: new Int32Array(reference.buffer, timestep.byteOffset, 1)[0],
  };
};

/** グラフ入力（io の `input.*` — 宣言の名前と shape をそのまま使う）。 */
const graphInputs = (
  golden: Golden,
  inputs: readonly { readonly name: string }[],
  where: string,
): Record<string, Tensor> =>
  Object.fromEntries(
    inputs.map(({ name }) => {
      const view = viewOf(golden.io, `input.${name}`, where);
      return [name, {
        dtype: "f32",
        shape: view.shape,
        data: floatsOf(golden.io, `input.${name}`, where),
      }];
    }),
  );

/** 最初にビットが割れる要素の添字（一致なら -1）。長さが違えば 0。 */
const firstBitMismatch = (actual: Float32Array, expected: Float32Array): number => {
  if (actual.length !== expected.length) return 0;
  const left = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const right = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  return left.findIndex((bits, index) => bits !== right[index]);
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

/** 帯の指標（最大絶対差 ÷ 参照の最大絶対値 — {@link DIT_RATIO_BAND}）。 */
const ratioOf = (diff: Difference): number => diff.maxAbs / diff.referenceMaxAbs;

/**
 * results.json の `tolerance` に載せる、このケースで比の帯と同値な絶対の帯（帯 × 参照の最大絶対値）。
 * 記録の形（`Measurement`）は絶対の帯しか持たないので、判定と同じ境界を絶対値へ写して残す。
 */
const recordedTolerance = (diff: Difference, band = DIT_RATIO_BAND) => ({
  atol: band * diff.referenceMaxAbs,
  rtol: 0,
});

const formatDifference = (label: string, diff: Difference): string =>
  `${label}: maxAbs ${diff.maxAbs.toExponential(3)} / 参照の最大絶対値 ` +
  `${diff.referenceMaxAbs.toFixed(3)}（比 ${ratioOf(diff).toExponential(2)}）`;

/**
 * 実寸の正規化の分母の元 = CPU f32 の参照（`output`）と f64 の参照（`output.f64`）の差。比が 0（f32 の参照が
 * f64 とビット一致）だと割れないので fail loudly にする。
 */
const referenceErrorOf = (golden: Golden, name: string): Difference => {
  const diff = difference(
    floatsOf(golden.reference, "output", name),
    floatsOf(golden.reference, REFERENCE_F64_KEY, name),
  );
  if (!(ratioOf(diff) > 0)) {
    throw new Error(
      `${name}: CPU f32 の参照の f64 に対する比が ${ratioOf(diff)}（正規化できない）`,
    );
  }
  return diff;
};

/**
 * 正規化した比（{@link DIT_FULL_NORMALIZED_BAND}）= GPU の f64 に対する比 ÷ CPU f32 の参照の f64 に対する比
 * （`referenceRatio`）。
 */
const normalizedRatio = (diff: Difference, referenceRatio: number): number =>
  ratioOf(diff) / referenceRatio;

const formatNormalized = (label: string, diff: Difference, referenceRatio: number): string =>
  `${formatDifference(label, diff)}・r ${normalizedRatio(diff, referenceRatio).toPrecision(3)}`;

/** p99.99 ベースの r の分位（{@link quantileRatio}）。 */
const RECORDED_QUANTILE = 0.9999;

/** |actual − expected| の q 分位（最近順位法 — 昇順に並べた ceil(q·n) 番目）。 */
const absErrorQuantile = (actual: Float32Array, expected: Float32Array, q: number): number => {
  assertEquals(actual.length, expected.length, "要素数");
  const errors = new Float64Array(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    errors[index] = Math.abs(actual[index] - expected[index]);
  }
  errors.sort();
  return errors[Math.max(0, Math.ceil(q * errors.length) - 1)];
};

/**
 * p99.99 ベースの r（= |GPU − f64| の p99.99 ÷ |CPU f32 の参照 − f64| の p99.99〈`referenceQuantile`〉）の
 * 記録の 1 行（results.json の note に載る）。
 * **記録だけ**で、門は max ベースの r（{@link normalizedRatio}）のまま。max の分母は外れ値 1 要素で決まり、
 * 分母が極端に小さい入力で r が暴れうる（ADR 0118 追記 2026-10-02「段 3 の結果 — 帯の床と指標 r の弱点」）
 * ので、置き換えるかを分布で判断する材料として残す。参照の最大絶対値での割り算は分子と分母で打ち消すので省く。
 */
const quantileRatio = (
  label: string,
  actual: Float32Array,
  expected: Float32Array,
  referenceQuantile: number,
): string => {
  const numerator = absErrorQuantile(actual, expected, RECORDED_QUANTILE);
  const ratio = numerator / referenceQuantile;
  const parts = [
    `分子 ${numerator.toExponential(3)}`,
    `分母 ${referenceQuantile.toExponential(3)}`,
  ];
  return `${label}: p99.99 ベースの r ${ratio.toPrecision(3)}（記録だけ・${parts.join(" / ")}）`;
};

/** 出口の最終次元を `(pt,ph,pw,c)` → `(c,pt,ph,pw)` と読み替える（unpatchify の並びの故障注入）。 */
const transposeTokenAxes = (tokens: Float32Array, geometry: WanPatchGeometry): Float32Array => {
  const channels = geometry.channels;
  const width = wanTokenWidth(geometry);
  const volume = width / channels;
  const out = new Float32Array(tokens.length);
  for (let token = 0; token < tokens.length / width; token += 1) {
    for (let c = 0; c < channels; c += 1) {
      for (let v = 0; v < volume; v += 1) {
        out[token * width + v * channels + c] = tokens[token * width + c * volume + v];
      }
    }
  }
  return out;
};

/** 容器の資産 `rope_base`（役割 `rope-base`）を読んで素表にする。 */
const readRopeBase = async (opened: OpenedContainer): Promise<WanRopeBase> => {
  const reader = opened.asset(ROPE_BASE_ASSET);
  assertEquals(reader.role, "rope-base", `資産 '${ROPE_BASE_ASSET}' の役割`);
  const bytes = await reader.read(0, reader.length);
  return parseWanRopeBase(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
};

const results = openResults("wan-dit");

Deno.test({
  name: "Wan DiT 資産: 容器と全ケースの golden が揃っている",
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
    "Wan DiT i8 資産: i8 系列の容器と全ケース（S = 192 の組 9 本・S = 14,040 の 8 本）の golden が揃っている",
  // 完全に空の環境だけ「生成していない」として SKIP。何か 1 つでもあれば欠けは FAIL（f16 系列と同じ規律）。
  ignore: !I8_ANY_PRESENT,
  fn: () => {
    assert(I8_MODEL_PRESENT, `${I8_SERIES_DIR.pathname}${MODEL_FILE} が無い`);
    assertEquals(
      i8ExpectedFiles.filter((url) => !filePresent(url)).map((url) => url.pathname),
      [],
      `i8 の golden の欠け（生成: ${I8_GENERATE}）`,
    );
  },
});

Deno.test(
  "Wan DiT 行ブロックの期待値（GPU 不要）: B570 の束縛上限で ADR 0118 決定 6 の表（24 枚 / 5 枚）を再現する",
  () => {
    assertEquals(
      FULL_TOKENS_ORDER.map((tokens) => expectedRowBlocks(tokens, B570_STORAGE_BINDING_LIMIT)),
      [24, 5],
    );
  },
);

Deno.test({
  name: "Wan DiT ホスト: patchify と資産 rope_base からの RoPE 表が golden の入力とビット一致する",
  ignore: !ASSETS_AVAILABLE,
  fn: async () => {
    const base = await readRopeBase(await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)));
    // 実寸（S = 14,040 / 32,760・非正方の格子 9·30·52 / 21·30·52）でもホストの並べ替えが torch とビット一致すること。
    for (const { name } of [...CASES, ...FULL_CASES]) {
      const golden = await loadGolden(name);
      const tokens = patchifyLatents(
        floatsOf(golden.reference, "latents", name),
        golden.latentShape,
        WAN_GEOMETRY,
      );
      assertEquals(
        firstBitMismatch(tokens, floatsOf(golden.io, "input.tokens", name)),
        -1,
        `${name}: patchify`,
      );
      const tables = wanRopeTables(base, wanTokenGrid(golden.latentShape, WAN_GEOMETRY));
      assertEquals(
        firstBitMismatch(tables.cos, floatsOf(golden.io, "input.rope_cos", name)),
        -1,
        `${name}: rope_cos`,
      );
      assertEquals(
        firstBitMismatch(tables.sin, floatsOf(golden.io, "input.rope_sin", name)),
        -1,
        `${name}: rope_sin`,
      );
    }
  },
});

/**
 * 計測モードの照合が出した出力（通常モードで回すケースだけ — ケース名 → unpatchify 前のトークン）。
 *
 * 通常モードのテストがこれと Uint32 で突き合わせる。計測モードは 1 dispatch = 1 pass で、裏付け後のチャンクの
 * 切れ目も通常の実行と変わりうる。それでも値が 1 ビットも動かないこと（チャンクの切れ目は値を動かさない）を
 * 直接縛る。2 つのモードは device が違う（計測の feature の有無）ので、同時に載せず別のテストで順に回す。
 *
 * MUST: 計測モードの照合を通常モードより前に登録する（Deno は同じファイルのテストを登録順に回す）。計測モードを
 * 飛ばした回（`--filter` など）は通常モードに比べる相手が無いので、比較を黙って省かず落とす。
 */
const measuredTokens = new Map<string, Float32Array>();

const gib = (bytes: number): string => `${(bytes / 2 ** 30).toFixed(2)} GiB`;

/** 実寸の 1 run の観測（所要と生きている確保 — 結果の note に載せる）。 */
type FullRun = {
  readonly label: string;
  readonly wallMs: number;
  /**
   * dispatch の pass の GPU 時間の合計（timestamp の生の単位 — ns への換算は {@link formatRun} が
   * {@link TIMESTAMP_UNIT_NS} で行う）。計測が無効な device では undefined。
   */
  readonly gpuTicks: number | undefined;
  /**
   * run の直後に生きている確保 = 重み + その run のアリーナ（中間・入力・readback staging）+ 保持中の
   * slot backing（中間と入力）。readback staging（`MAP_READ`）も含む — B570 では VRAM の予算に数えられない
   * 側に載る（2026-10-02 実測: STORAGE 9 GiB の後でも `MAP_READ` 3 GiB が取れた）ので別に出す。
   */
  readonly liveBytes: number;
  /** そのうちの重み（initializer と params キャッシュ）。 */
  readonly weightBytes: number;
  /** そのうちの readback staging（グラフ出力のバイト数の和）。 */
  readonly stagingBytes: number;
};

const observeRun = (
  label: string,
  diagnostics: SessionDiagnostics,
  wallMs: number,
  stagingBytes: number,
): FullRun => ({
  label,
  wallMs,
  gpuTicks: diagnostics.lastRunTiming?.totalNs,
  liveBytes: diagnostics.weights.allocatedBytes + (diagnostics.lastRun?.allocatedBytes ?? 0) +
    diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes,
  weightBytes: diagnostics.weights.allocatedBytes,
  stagingBytes,
});

/** `unitNs` は timestamp の 1 単位の ns（省くと GPU 時間を出さない — 計測なしの run）。 */
const formatRun = (run: FullRun, unitNs?: number): string =>
  `${run.label}: 壁 ${(run.wallMs / 1000).toFixed(2)} s` +
  (run.gpuTicks === undefined || unitNs === undefined
    ? ""
    : `・GPU ${(run.gpuTicks * unitNs / 1e9).toFixed(2)} s`) +
  `・確保 ${gib(run.liveBytes)}（うち重み ${gib(run.weightBytes)}・readback staging ${
    gib(run.stagingBytes)
  }）`;

/**
 * self-attention の行ブロック枚数（直近 run の `lastRunPipelines` の ①QK の dispatch 本数から）。行窓のキー
 * （`:rwa` — src/kernels/attention.ts の `attentionQkKey`）は層ごとにブロック枚数ぶん、行窓の無いキーは
 * cross-attention（N = 512 で 1 枚に収まる）で層ごとに 1 本。
 */
const rowBlocksOf = (diagnostics: SessionDiagnostics): number => {
  const qk = (diagnostics.lastRunPipelines ?? []).filter(({ key }) =>
    key.startsWith("attention_qk:")
  );
  const count = (windowed: boolean): number =>
    qk.filter(({ key }) => key.includes(":rwa") === windowed)
      .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
  const layers = count(false);
  assert(layers > 0, "①QK の行窓の無い dispatch（cross-attention）が 1 本も無い");
  return count(true) / layers;
};

/** 1 submit の GPU 時間の観測を ms に直した記録（門は {@link SUBMIT_GPU_LIMIT_MS}）。 */
const submitGpuNote = (
  diagnostics: SessionDiagnostics,
  unitNs: number,
): { readonly maxMs: number; readonly unbackedSubmits: number; readonly note: string } => {
  const budget = diagnostics.submit.chunkBudget;
  const observed = budget.submitGpuTime;
  if (observed === undefined) throw new Error("計測モードなのに submit の GPU 時間が無い");
  const ms = (ns: number): string => `${(ns * unitNs / 1e6).toFixed(1)} ms`;
  const note = [
    `1 submit の GPU 時間の最大 ${ms(observed.maxNs)}（${observed.submits} 本）`,
    `裏付け前 ${observed.unbackedSubmits} 本の最大 ${
      observed.maxUnbackedNs === undefined ? "—" : ms(observed.maxUnbackedNs)
    }`,
    `単発 dispatch の最大 ${ms(observed.maxDispatchNs)}（${observed.maxDispatchKey ?? "—"}）`,
    `窓平均の最大 ${budget.maxWindowMeanMs?.toFixed(1) ?? "—"} ms・推定の最大 ${
      budget.maxEstimatedMs?.toFixed(1) ?? "—"
    } ms・予算超過 ${budget.overBudgetChunks} 本`,
    `submit ${diagnostics.submit.submitCount} 本・dispatch ${diagnostics.submit.dispatchCount} 本`,
  ].join("・");
  return {
    maxMs: observed.maxNs * unitNs / 1e6,
    unbackedSubmits: observed.unbackedSubmits,
    note,
  };
};

/** グラフ出力のバイト数の和（readback staging の大きさ）。 */
const outputBytes = (outputs: Readonly<Record<string, Tensor>>): number =>
  Object.values(outputs).reduce((total, tensor) => total + tensor.data.byteLength, 0);

/**
 * 計測用グラフの出力（各ブロックの出力・最後は最終出力）を上流の forward hook の値と突き合わせ、
 * 層ごとの比を `measurements` に積んで返す（門ではない — 非有限だけを落とす）。
 */
const layerRatios = (
  produced: Readonly<Record<string, Tensor>>,
  outputs: readonly string[],
  golden: Golden,
  name: string,
  measurements: Measurement[],
): string[] => {
  const blocks = outputs.length - 1;
  const rows: string[] = [];
  for (let index = 0; index < blocks; index += 1) {
    const tensor = produced[outputs[index]];
    if (tensor.dtype !== "f32") throw new Error(`出力 ${index} が ${tensor.dtype}`);
    const key = `block.${String(index).padStart(2, "0")}`;
    const diff = difference(tensor.data, floatsOf(golden.reference, key, name));
    assertEquals(diff.nonFinite, 0, `${name}: ${key} が非有限`);
    measurements.push({
      output: key,
      maxAbs: diff.maxAbs,
      maxRel: diff.maxRel,
      tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
      stage: "karume",
    });
    rows.push(`${index}:${(diff.maxAbs / diff.referenceMaxAbs).toExponential(2)}`);
  }
  console.log(
    `[wan-dit] ${name} 層ごとの比（最大絶対差 / 参照の最大絶対値）: ${rows.join(" ")}`,
  );
  return rows;
};

// 実寸（S = 32,760 / 14,040）の 4 本は、S = 192 の GPU テストより**前**に置く。B570 では破棄した device の確保が
// 解放されないまま残り（{@link settleReleases}）、前のテストの残りが実寸の数 GiB の確保を OOM にしうる。
// 実寸の 4 本は device を捨てる前に解放を待つので、後ろのテストへは残りを渡さない。同じ理由で確保が最大のものを
// 先に回す: 層別の probe（S = 14,040・出口 31 本の readback staging を含めて確保 9.85 GiB）を先頭に、照合は
// S = 32,760 → 14,040 の順（{@link FULL_TOKENS_ORDER}）で S ごとに Session を張り直す — 1 本の Session で形を
// 切り替えると、退役した slot backing が切り替えの run の後始末まで残り、2 つの S の backing が同時に載る
// （runtime の `estimate.ts` — `peakAccountedBytes` の doc にある「退役から実際の destroy() までの窓」）。
// 2026-10-03: probe を照合の後に置いた並びでは、S = 32,760 の照合の後で probe が OOM になった（確保の残り）。

Deno.test({
  name:
    "Wan DiT 実寸 層別（実 GPU・計測用グラフ）: S = 14,040 の各ブロックの出力の差を記録する（門ではない）",
  ignore: !PROBE_AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, PROBE_DIR)),
      PROBE_GRAPH,
    );
    const outputs = prepared.graph.outputs;
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const session = await prepared.createContainerSession(gpu);
      try {
        // ブロックの出力を golden に持つ 2 ケースだけ（{@link FULL_CASES} の `blocks`）。
        for (const { name } of FULL_CASES.filter(({ blocks }) => blocks)) {
          await runRecordedCase(results, { id: `${name}/layers` }, async ({ measurements }) => {
            const golden = await loadGolden(name);
            const started = performance.now();
            const produced = await session.run(graphInputs(golden, prepared.graph.inputs, name));
            const observed = observeRun(
              "層別",
              session.diagnostics(),
              performance.now() - started,
              outputBytes(produced),
            );
            // 出口 31 本（2.4 GiB）の中間を持つ run なので、次の run の確保の前に解放を待つ。
            await settleReleases(gpu);
            const rows = layerRatios(produced, outputs, golden, name, measurements);
            return { status: "pass", note: `${rows.join(" ")} / ${formatRun(observed)}` };
          });
        }
      } finally {
        await session.dispose();
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

/** {@link normalizedRatioGate} の 1 ケース。 */
type GateCase = {
  readonly name: string;
  /** step 名に出す役割（`full-band` / `band` …）。 */
  readonly role: string;
  /** 受入れ（TS の timesteps_proj と故障注入を回す・帯の決定には使わない）。 */
  readonly accept: boolean;
};

/** {@link normalizedRatioGate} の入力（系列 root 1 つ × S の組 1 つ）。 */
type GateSpec = {
  readonly t: Deno.TestContext;
  readonly gpu: GpuContext;
  readonly prepared: PreparedModel;
  readonly base: WanRopeBase;
  /** golden の置き場（系列の部品ディレクトリ）。 */
  readonly seriesDir: URL;
  /** 結果の ID・step 名・ログの接頭辞（f16 席は空・i8 の参照席は {@link I8_ID_PREFIX}）。 */
  readonly idPrefix: string;
  readonly cases: readonly GateCase[];
  /** 正規化した比 r の帯。`undefined` = 未導出（r を記録して赤で止める）。 */
  readonly band: number | undefined;
  /**
   * 実寸の組だけ: golden の S・行ブロックの枚数（束縛上限から導く期待）・計測モードの 1 submit の GPU 時間の門。
   * S = 192 の組は持たない（行窓が無い・1 submit の門は実寸の主張）。
   */
  readonly full?: {
    readonly tokens: number;
    readonly bindingLimit: number;
    readonly deviceLost: () => string | undefined;
  };
  /** 通常モードの比較相手（{@link measuredTokens}）を残すケース（f16 席だけ — ケース名は席をまたいで共通）。 */
  readonly normalModeCase?: string;
};

/** 1 ケースの r（{@link bandCandidate} の材料 — 判定の前に積むので帯の外の回も残る）。 */
type CaseRatio = { readonly name: string; readonly accept: boolean; readonly ratio: number };

/**
 * 正規化した比 r の門（ADR 0118 決定 8）を、系列 root 1 つ・S の組 1 つについて回す（Session を張って全ケースを
 * 回し、畳んで解放を待つ）。f16 席の実寸（S = 32,760 / 14,040）と i8 の参照席（S = 192 の組 / 14,040 — ADR 0120
 * 決定 3）が同じこの 1 本を通る — 「決定 8 と同じ r 門」を席ごとに別々に育てない。
 *
 * 1 ケース: io の入力で 1 forward → f64 の参照に対する比を CPU f32 の参照自身の比で割った r を帯で判定する。受入れは
 * 加えて TS の timesteps_proj の入力と故障注入 4 件（{@link faultInjections}）。実寸の組（`full`）だけ golden の S・
 * 行ブロックの枚数・計測モードの 1 submit の GPU 時間も門にする。帯が未導出なら r を出して赤で止める。
 *
 * 返すのは r を出せたケースの r（帯の候補の材料）。
 */
const normalizedRatioGate = async (spec: GateSpec): Promise<readonly CaseRatio[]> => {
  const { t, gpu, prepared, base, band, full, idPrefix } = spec;
  /** 失敗文言の帯の呼び名（実寸の組は f16 席の既存の文言のまま）。 */
  const bandLabel = full === undefined ? "帯" : "実寸の帯";
  const ratios: CaseRatio[] = [];
  const built = performance.now();
  const session = await prepared.createContainerSession(gpu);
  const buildMs = performance.now() - built;
  const runs: FullRun[] = [];
  try {
    const run = async (
      inputs: Record<string, Tensor>,
      label = "故障注入",
    ): Promise<Float32Array> => {
      const started = performance.now();
      const outputs = await session.run(inputs);
      const wallMs = performance.now() - started;
      const tensor = outputs[prepared.graph.outputs[0]];
      if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
      runs.push(observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)));
      // 次の run の確保（2 本目は slot backing の構築）の前に、この run の中間の解放を待つ。
      await settleReleases(gpu);
      return tensor.data;
    };
    for (const { name, role, accept } of spec.cases) {
      const id = `${idPrefix}${name}`;
      await t.step(`${id}（${role}）`, async () => {
        // 帯の外で落ちた回も、比と故障注入の記録を results.json に残す。
        let caseNote = "";
        await runRecordedCase(results, {
          id,
          failureNote: () => caseNote,
        }, async ({ measurements }) => {
          const golden = await loadGolden(name, spec.seriesDir);
          if (full !== undefined) {
            assertEquals(
              wanTokenGrid(golden.latentShape, WAN_GEOMETRY).count,
              full.tokens,
              `${id}: golden の S`,
            );
          }
          const inputs = graphInputs(golden, prepared.graph.inputs, name);
          // 誤差の基準は f64 の参照で、判定は CPU f32 の参照自身の誤差で正規化した比
          // （{@link DIT_FULL_NORMALIZED_BAND} / {@link I8_NORMALIZED_BAND}）。GPU と CPU f32 の参照の差は記録だけ。
          const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
          const reference = floatsOf(golden.reference, "output", name);
          const referenceError = referenceErrorOf(golden, name);
          const referenceRatio = ratioOf(referenceError);
          const referenceQuantile = absErrorQuantile(reference, expected, RECORDED_QUANTILE);
          // 帯が未導出の回は無限の帯として記録する（記録だけの欄と同じ書き方 — 判定は下で赤にする）。
          const toleranceOf = (diff: Difference) =>
            band === undefined
              ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
              : recordedTolerance(diff, band * referenceRatio);
          const tokens = await run(inputs, `${name} io`);
          // 判定の前に残す（帯の外へ出た回も、通常モードのビット一致の門は走る）。
          if (name === spec.normalModeCase) measuredTokens.set(name, tokens);
          const latents = unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY);
          const diff = difference(latents, expected);
          const diffF32 = difference(latents, reference);
          measurements.push({
            output: "latents",
            maxAbs: diff.maxAbs,
            maxRel: diff.maxRel,
            tolerance: toleranceOf(diff),
            stage: "karume",
          }, {
            output: "latents@f32-reference",
            maxAbs: diffF32.maxAbs,
            maxRel: diffF32.maxRel,
            tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
            stage: "karume",
          }, {
            output: "f32-reference@f64-reference",
            maxAbs: referenceError.maxAbs,
            maxRel: referenceError.maxRel,
            tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
            stage: "karume",
          });
          ratios.push({ name, accept, ratio: normalizedRatio(diff, referenceRatio) });
          const diagnostics = session.diagnostics();
          const rowBlocks = full === undefined ? undefined : rowBlocksOf(diagnostics);
          const notes = [
            formatNormalized("io の入力 / f64 参照", diff, referenceRatio),
            formatDifference("CPU f32 参照 / f64 参照（正規化の分母）", referenceError),
            formatDifference("io の入力 / CPU f32 参照（記録）", diffF32),
            quantileRatio("io の入力 / f64 参照", latents, expected, referenceQuantile),
            ...(rowBlocks === undefined ? [] : [`行ブロック ${rowBlocks} 枚`]),
          ];
          assertEquals(diff.nonFinite, 0, `${id}: 非有限`);
          if (full !== undefined) {
            assertEquals(
              rowBlocks,
              expectedRowBlocks(full.tokens, full.bindingLimit),
              `${id}: self-attention の行ブロック枚数（束縛上限 ${full.bindingLimit} B）`,
            );
          }
          assertEquals(
            { ...diagnostics.lastRunFusions, ...EXPECTED_FUSIONS },
            diagnostics.lastRunFusions,
            `${id}: 融合の件数（adaln / rope / silu）`,
          );

          let hostDiff: Difference | undefined;
          let faults: FaultResult[] = [];
          if (accept) {
            // 製品の経路の入力（ホストの timesteps_proj）も同じ帯で受け入れる（S = 192 の照合と同じ形）。
            const width = viewOf(golden.io, "input.timesteps_proj", name).shape[1];
            const hostLatents = unpatchifyTokens(
              await run({
                ...inputs,
                timesteps_proj: {
                  dtype: "f32",
                  shape: [1, width],
                  data: timestepsProj(golden.timestep, width),
                },
              }, `${name} TS の timesteps_proj`),
              golden.latentShape,
              WAN_GEOMETRY,
            );
            hostDiff = difference(hostLatents, expected);
            measurements.push({
              output: "latents@host-timesteps-proj",
              maxAbs: hostDiff.maxAbs,
              maxRel: hostDiff.maxRel,
              tolerance: toleranceOf(hostDiff),
              stage: "karume",
            });
            notes.push(
              formatNormalized("TS の timesteps_proj / f64 参照", hostDiff, referenceRatio),
              quantileRatio(
                "TS の timesteps_proj / f64 参照",
                hostLatents,
                expected,
                referenceQuantile,
              ),
            );
            faults = await faultInjections(golden, inputs, tokens, expected, run, base);
            notes.push(
              ...faults.map(({ note, ratio }) =>
                `${note}・r ${(ratio / referenceRatio).toPrecision(3)}（${
                  band === undefined
                    ? "帯は未導出"
                    : `帯の ${(ratio / referenceRatio / band).toFixed(1)} 倍`
                }）`
              ),
            );
          }
          // 判定の前に出す（帯の外へ出た回も比が手元に残る）。
          caseNote = notes.join(" / ");
          console.log(`[wan-dit] ${id}: ${caseNote}`);

          if (band === undefined) {
            throw new Error(
              `${id}: 帯が未導出（r ${
                normalizedRatio(diff, referenceRatio).toPrecision(3)
              }）— 決定用の最悪 r × 5 を帯の定数へ書く（帯の候補は同じ組の最後の step が出す）`,
            );
          }
          // 故障注入の判定を先に置く（受入れが帯の外でも、帯が故障を拾えるかは判定される）。
          for (const { label, ratio } of faults) {
            const normalized = ratio / referenceRatio;
            assert(
              normalized > band,
              `${id}: 故障注入 ${label} の r ${
                normalized.toPrecision(3)
              } が${bandLabel} ${band} の内に収まった`,
            );
          }
          if (accept) {
            const subtle = faults.find(({ label }) => label === TIMESTEP_OFF_BY_ONE);
            if (subtle === undefined) {
              throw new Error(`${id}: 故障注入 ${TIMESTEP_OFF_BY_ONE} が無い`);
            }
            const margin = subtle.ratio / referenceRatio / band;
            assert(
              margin >= SUBTLE_FAULT_MARGIN,
              `${id}: 故障注入 ${TIMESTEP_OFF_BY_ONE} が${bandLabel}の ${
                margin.toFixed(2)
              } 倍（${SUBTLE_FAULT_MARGIN} 倍以上のはず — 帯が広すぎる兆候）`,
            );
          }
          assert(
            normalizedRatio(diff, referenceRatio) <= band,
            `${id}: ${notes[0]} が${bandLabel} ${band} の外`,
          );
          if (hostDiff !== undefined) {
            assert(
              normalizedRatio(hostDiff, referenceRatio) <= band,
              `${id}: TS の timesteps_proj の r ${
                normalizedRatio(hostDiff, referenceRatio).toPrecision(3)
              } が${bandLabel} ${band} の外`,
            );
          }
          return { status: "pass", note: notes.join(" / ") };
        });
      });
    }
    if (full !== undefined) {
      await t.step(
        `${idPrefix}S = ${full.tokens} の 1 submit の GPU 時間・所要・確保`,
        async () => {
          let note = "";
          await runRecordedCase(
            results,
            { id: `${idPrefix}full-s${full.tokens}/submit`, failureNote: () => note },
            () => {
              // 換算の表はこのステップでだけ引く（表に無い環境で落ちるのは時間門だけ）。
              const unitNs = timestampUnitNs(ENVIRONMENT.key);
              const submit = submitGpuNote(session.diagnostics(), unitNs);
              note = [
                submit.note,
                `構築 ${(buildMs / 1000).toFixed(1)} s`,
                ...runs.map((observed) => formatRun(observed, unitNs)),
              ].join(" / ");
              console.log(`[wan-dit] ${idPrefix}実寸 S = ${full.tokens} の計測モード: ${note}`);
              assert(
                submit.unbackedSubmits > 0,
                "最初の run の裏付け前のチャンク（initialChunkSize で据え置いた submit）を測れていない",
              );
              assert(
                submit.maxMs <= SUBMIT_GPU_LIMIT_MS,
                `1 submit の GPU 時間の最大 ${
                  submit.maxMs.toFixed(1)
                } ms が ${SUBMIT_GPU_LIMIT_MS} ms を超えた`,
              );
              assertEquals(full.deviceLost(), undefined, "device lost");
              return Promise.resolve({ status: "pass", note });
            },
          );
        },
      );
    }
  } finally {
    await session.dispose();
    // 次の組の Session の確保の前に、この組の確保の解放を待つ（{@link settleReleases}）。
    await settleReleases(gpu);
  }
  return ratios;
};

/**
 * 未導出の帯（`undefined`）の候補を出して赤で止める step。候補は導出規則（ADR 0118 決定 8 — その組の決定用の
 * 最悪 r × 5）どおりに決定用のケースだけから作る。受入れの r は並べて出すが候補には使わない（帯の決定と受入れの
 * 独立）。決定用のどれかが r を出す前に落ちた回は候補を出さない（欠けた最悪は最悪ではない）。帯が導出済みなら
 * 何もしない。
 */
const bandCandidate = async (
  t: Deno.TestContext,
  label: string,
  cases: readonly GateCase[],
  ratios: readonly CaseRatio[],
  band: number | undefined,
): Promise<void> => {
  if (band !== undefined) return;
  await t.step(`${label}: 帯の候補（未導出）`, () => {
    const expected = cases.filter(({ accept }) => !accept).length;
    const decided = ratios.filter(({ accept }) => !accept);
    const worst = Math.max(...decided.map(({ ratio }) => ratio));
    const unit = 10 ** (Math.floor(Math.log10(worst * 5)) - 1);
    const candidate = decided.length === expected
      ? `決定用 ${expected} 本の最悪 r ${worst.toPrecision(3)} × 5 = ${
        (worst * 5).toPrecision(3)
      }` +
        `（有効数字 2 桁へ切り上げ ${Number((Math.ceil(worst * 5 / unit) * unit).toPrecision(2))}）`
      : `決定用 ${expected} 本のうち r が出たのは ${decided.length} 本 — 候補にしない（落ちたケースを先に調べる）`;
    const accepted = ratios.filter(({ accept }) => accept)
      .map(({ name, ratio }) => `${name} ${ratio.toPrecision(3)}`);
    throw new Error(
      `${label}: 帯が未導出 — ${candidate}。受入れの r（帯の決定に使わない）: ${
        accepted.join(" / ") || "なし"
      }`,
    );
  });
};

/**
 * linear 1 本（`weight`）の per-channel scale を 2 倍にした供給面（i8 に固有の故障注入 — {@link I8_SCALE_FAULT_WEIGHT}）。
 * 容器のファイルは書き換えない: 取得した block（未検証の取得元なので sha256 の検証の後）の**写し**を 2 倍にして返す
 * （`BoundContainer.readBlock` の MUST — 返された器は書き換えない）。scale の payload は block の先頭からの f32 の列
 * （Session 構築が `subarray(0, payloadBytes)` で読む形）。
 */
const scaledScaleContainer = (
  opened: OpenedContainer,
  graph: string,
  weight: string,
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
      const doubled = bytes.slice();
      const values = new Float32Array(doubled.buffer, 0, scale.payloadBytes / 4);
      for (let index = 0; index < values.length; index += 1) values[index] *= 2;
      return doubled;
    },
  };
};

/**
 * i8 に固有の故障注入（{@link scaledScaleContainer}）で受入れケースを回し、r が帯の外へ出ることを門にする
 * （ADR 0120 検収 段 3）。Session は {@link normalizedRatioGate} の Session を畳んだ後に別に張る（2 本を同時に
 * 載せない — 実寸で 2 本は B570 の予算を食う）。
 */
const i8ScaleFault = async (
  t: Deno.TestContext,
  gpu: GpuContext,
  opened: OpenedContainer,
  graph: string,
  cases: readonly GateCase[],
  band: number | undefined,
): Promise<void> => {
  const label = `${I8_SCALE_FAULT_WEIGHT} の scale を 2 倍`;
  const prepared = prepareContainer(
    scaledScaleContainer(opened, graph, I8_SCALE_FAULT_WEIGHT),
    graph,
  );
  const session = await prepared.createContainerSession(gpu);
  try {
    for (const { name } of cases.filter(({ accept }) => accept)) {
      const id = `${I8_ID_PREFIX}${name}/scale-fault`;
      await t.step(`${id}（故障注入 ${label} → 帯の外）`, async () => {
        let note = "";
        await runRecordedCase(
          results,
          { id, failureNote: () => note },
          async ({ measurements }) => {
            const golden = await loadGolden(name, I8_SERIES_DIR);
            const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
            const referenceRatio = ratioOf(referenceErrorOf(golden, name));
            const outputs = await session.run(graphInputs(golden, prepared.graph.inputs, name));
            const tensor = outputs[prepared.graph.outputs[0]];
            if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
            await settleReleases(gpu);
            const diff = difference(
              unpatchifyTokens(tensor.data, golden.latentShape, WAN_GEOMETRY),
              expected,
            );
            const normalized = normalizedRatio(diff, referenceRatio);
            measurements.push({
              output: "latents@scale-fault",
              maxAbs: diff.maxAbs,
              maxRel: diff.maxRel,
              tolerance: { atol: Number.POSITIVE_INFINITY, rtol: 0 },
              stage: "karume",
            });
            note = `${formatNormalized(`故障注入 ${label} / f64 参照`, diff, referenceRatio)}（${
              band === undefined ? "帯は未導出" : `帯の ${(normalized / band).toFixed(1)} 倍`
            }）`;
            console.log(`[wan-dit] ${id}: ${note}`);
            if (band === undefined) {
              throw new Error(`${id}: 帯が未導出（故障注入の r ${normalized}）`);
            }
            assert(
              normalized > band,
              `${id}: 故障注入 ${label} の r ${
                normalized.toPrecision(3)
              } が帯 ${band} の内に収まった`,
            );
            return { status: "pass", note };
          },
        );
      });
    }
  } finally {
    await session.dispose();
    await settleReleases(gpu);
  }
};

Deno.test({
  name:
    "Wan DiT 実寸 照合（実 GPU・計測モード / diffusers CPU f64）: S = 32,760 / 14,040 の 1 forward が S ごとの" +
    "実寸の帯の内・故障注入は帯の外・行ブロックは S ごとの枚数・1 submit の GPU 時間 ≤ 1 s",
  ignore: !ASSETS_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR));
    const prepared = prepareContainer(opened, GRAPH);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    const base = await readRopeBase(opened);
    let deviceLost: string | undefined;
    // 計測モード（timestamp-query）: submit ごとの GPU 時間は既存の回収に相乗りして測る（新しい待ちを
    // 足さない — runtime の `ChunkBudgetStats.submitGpuTime`）。計算は通常の実行と同じだが、1 dispatch
    // = 1 pass になる分だけ窓の壁時計が変わる。チャンクの切れ目は壁時計の窓から学んだ推定で決まるので、
    // 通常の実行と同じなのは裏付け前のチャンク（initialChunkSize）だけで、裏付けが付いた後（最初の窓を
    // 閉じた後）の切れ目は変わりうる。
    const gpu = await acquireGpu({
      gpuTiming: true,
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      // 期待値の元は device が実際に許した束縛上限（runtime の行ブロックが見るのと同じ値）。
      const bindingLimit = gpu.device.limits.maxStorageBufferBindingSize;
      for (const { tokens: fullTokens, cases, normalModeCase } of FULL_CASES_BY_TOKENS) {
        await normalizedRatioGate({
          t,
          gpu,
          prepared,
          base,
          seriesDir: SERIES_DIR,
          idPrefix: "",
          cases: cases.map(({ name, role }) => ({ name, role, accept: role === "full-accept" })),
          band: DIT_FULL_NORMALIZED_BAND[fullTokens],
          full: { tokens: fullTokens, bindingLimit, deviceLost: () => deviceLost },
          normalModeCase,
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "Wan DiT 実寸 通常モード（実 GPU・計測なし）: S ごと（32,760 / 14,040）の 1 forward の所要を記録する（帯の内）",
  ignore: !ASSETS_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR)),
      GRAPH,
    );
    let deviceLost: string | undefined;
    // 所要は計測を切った通常の実行で採る（ADR 0118 追記 2026-10-02 — 計測モードは 1 dispatch = 1 pass）。
    const gpu = await acquireGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      // S ごとに Session を張り直す（理由は計測モードの照合と同じ — 2 つの S の backing を同時に載せない）。
      for (const { tokens: fullTokens, normalModeCase: name } of FULL_CASES_BY_TOKENS) {
        await t.step(`S = ${fullTokens}`, async () => {
          const fullBand = DIT_FULL_NORMALIZED_BAND[fullTokens];
          const built = performance.now();
          const session = await prepared.createContainerSession(gpu);
          const buildMs = performance.now() - built;
          try {
            await runRecordedCase(
              results,
              { id: `${name}/normal-mode` },
              async ({ measurements }) => {
                const golden = await loadGolden(name);
                const inputs = graphInputs(golden, prepared.graph.inputs, name);
                const expected = floatsOf(golden.reference, REFERENCE_F64_KEY, name);
                const referenceRatio = ratioOf(referenceErrorOf(golden, name));
                const runs: FullRun[] = [];
                const produced: Float32Array[] = [];
                // 1 本目は最初の run（パイプラインの生成と、裏付け前の initialChunkSize のチャンク）、2 本目は
                // 2 回目以降の形（slot backing・時間予算で切ったチャンク）。生成の 1 ステップに近いのは 2 本目。
                for (const label of ["1 本目", "2 本目"]) {
                  const started = performance.now();
                  const outputs = await session.run(inputs);
                  const wallMs = performance.now() - started;
                  const tensor = outputs[prepared.graph.outputs[0]];
                  if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
                  produced.push(tensor.data);
                  runs.push(
                    observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)),
                  );
                  await settleReleases(gpu);
                }
                const tokens = produced[produced.length - 1];
                const diff = difference(
                  unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY),
                  expected,
                );
                measurements.push({
                  output: "latents",
                  maxAbs: diff.maxAbs,
                  maxRel: diff.maxRel,
                  tolerance: recordedTolerance(diff, fullBand * referenceRatio),
                  stage: "karume",
                });
                const budget = session.diagnostics().submit.chunkBudget;
                const note = [
                  formatNormalized("io の入力 / f64 参照", diff, referenceRatio),
                  `構築 ${(buildMs / 1000).toFixed(1)} s`,
                  ...runs.map(formatRun),
                  `窓平均の最大 ${
                    budget.maxWindowMeanMs?.toFixed(1) ?? "—"
                  } ms・予算超過 ${budget.overBudgetChunks} 本`,
                ].join(" / ");
                console.log(`[wan-dit] 実寸 S = ${fullTokens} の通常モード: ${note}`);
                assertEquals(diff.nonFinite, 0, `${name}: 非有限`);
                assert(
                  normalizedRatio(diff, referenceRatio) <= fullBand,
                  `${name}: 通常モードの r ${
                    normalizedRatio(diff, referenceRatio).toPrecision(3)
                  } が実寸の帯 ${fullBand} の外`,
                );
                const measured = measuredTokens.get(name);
                if (measured === undefined) {
                  throw new Error(
                    `${name}: 計測モードの照合の出力が無い（同じ回で計測モードの照合を先に回す — ` +
                      "measuredTokens の MUST）",
                  );
                }
                for (const [index, data] of produced.entries()) {
                  const { label } = runs[index];
                  assertEquals(
                    firstBitMismatch(data, measured),
                    -1,
                    `${name}: 通常モードの ${label}が計測モードの出力と Uint32 で一致しない` +
                      "（値は最初に割れる要素の添字）",
                  );
                }
                assertEquals(deviceLost, undefined, "device lost");
                return { status: "pass", note };
              },
            );
          } finally {
            await session.dispose();
            await settleReleases(gpu);
          }
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

// 実用席の時間と確保（ADR 0120 段 4）と i8 の参照席の r 門（段 3）は、f16 席の実寸の後・S = 192 の前に置く。並びは
// 確保の大きい順（実用席の S = 32,760 → 14,040 → i8 の参照席の S = 14,040 → S = 192 の組）— 理由は f16 席の実寸を
// 先に置くのと同じ（B570 の `destroy()` の遅れ・{@link settleReleases}）。

/**
 * 実用席の 1 forward を 2 本回す（1 本目 = パイプラインの生成と裏付け前のチャンク・2 本目 = 生成の 1 パスに近い
 * 2 回目以降の形 — f16 席の通常モードと同じ並び）。`onRun` は各 run の前に呼ぶ（fdinfo の区間の切り替え）。
 */
const practicalRuns = async (
  session: Session,
  gpu: GpuContext,
  inputs: Record<string, Tensor>,
  output: string,
  onRun: (label: string) => void = () => {},
): Promise<{ readonly runs: readonly FullRun[]; readonly nonFinite: number }> => {
  const runs: FullRun[] = [];
  let nonFinite = 0;
  for (const label of ["1 本目", "2 本目"]) {
    onRun(label);
    const started = performance.now();
    const outputs = await session.run(inputs);
    const wallMs = performance.now() - started;
    const tensor = outputs[output];
    if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
    nonFinite += tensor.data.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);
    runs.push(observeRun(label, session.diagnostics(), wallMs, outputBytes(outputs)));
    // 次の run の確保の前に、この run の中間の解放を待つ。
    await settleReleases(gpu);
  }
  return { runs, nonFinite };
};

Deno.test({
  name:
    `Wan DiT 実用席 ${PRACTICAL_QUANT}（実 GPU・計測モード）: S = 32,760 / 14,040 の 1 パスの GPU 時間を` +
    "記録し、1 submit の GPU 時間 ≤ 1 s・非有限 0",
  ignore: !PRACTICAL_RUNNABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const options = practicalSessionOptions();
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, I8_SERIES_DIR)),
      i8Graph(),
    );
    let deviceLost: string | undefined;
    // 計測モードの意味は f16 席の実寸の照合と同じ（1 dispatch = 1 pass — 壁時計は下の通常モードが採る）。1 submit の
    // 門は a8 で dispatch が増える分（anima で 3,087 → 3,311 / step — ADR 0030）を席ごとに測り直す（ADR 0120 リスク 8）。
    const gpu = await acquireGpu({
      gpuTiming: true,
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      // S ごとに Session を張り直す（2 つの S の backing を同時に載せない — f16 席の実寸と同じ）。
      for (const { tokens, name } of PRACTICAL_TIMING_CASES) {
        await t.step(`${PRACTICAL_QUANT} S = ${tokens}`, async () => {
          const built = performance.now();
          const session = await prepared.createContainerSession(gpu, options);
          const buildMs = performance.now() - built;
          let note = "";
          try {
            await runRecordedCase(
              results,
              { id: `${PRACTICAL_QUANT}/full-s${tokens}/submit`, failureNote: () => note },
              async () => {
                const golden = await loadGolden(name);
                const { runs, nonFinite } = await practicalRuns(
                  session,
                  gpu,
                  graphInputs(golden, prepared.graph.inputs, name),
                  prepared.graph.outputs[0],
                );
                const unitNs = timestampUnitNs(ENVIRONMENT.key);
                const submit = submitGpuNote(session.diagnostics(), unitNs);
                note = [
                  submit.note,
                  `構築 ${(buildMs / 1000).toFixed(1)} s`,
                  ...runs.map((observed) => formatRun(observed, unitNs)),
                  `融合 ${JSON.stringify(session.diagnostics().lastRunFusions)}`,
                  `非有限 ${nonFinite}`,
                ].join(" / ");
                console.log(
                  `[wan-dit] 実用席 ${PRACTICAL_QUANT} S = ${tokens} の計測モード: ${note}`,
                );
                assertEquals(nonFinite, 0, `${name}: 非有限`);
                assert(
                  submit.unbackedSubmits > 0,
                  "最初の run の裏付け前のチャンク（initialChunkSize で据え置いた submit）を測れていない",
                );
                assert(
                  submit.maxMs <= SUBMIT_GPU_LIMIT_MS,
                  `1 submit の GPU 時間の最大 ${
                    submit.maxMs.toFixed(1)
                  } ms が ${SUBMIT_GPU_LIMIT_MS} ms を超えた`,
                );
                assertEquals(deviceLost, undefined, "device lost");
                return { status: "pass", note };
              },
            );
          } finally {
            await session.dispose();
            await settleReleases(gpu);
          }
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    `Wan DiT 実用席 ${PRACTICAL_QUANT}（実 GPU・通常モード）: S = 32,760 / 14,040 の 1 forward の壁時間と確保` +
    "（fdinfo）を記録する・非有限 0",
  ignore: !PRACTICAL_RUNNABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const options = practicalSessionOptions();
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, I8_SERIES_DIR)),
      i8Graph(),
    );
    let deviceLost: string | undefined;
    const gpu = await acquireGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      for (const { tokens, name } of PRACTICAL_TIMING_CASES) {
        await t.step(`${PRACTICAL_QUANT} S = ${tokens}`, async () => {
          // 区間ごとの VRAM の山（fdinfo の `drm-total-*` — 10 ms の標本で短い山は取りこぼしうる・門ではない）。
          // `start` の区間は Session を張る前の値（この process の他の確保）。
          const monitor = monitorDrmUsage();
          let note = "";
          try {
            await runRecordedCase(
              results,
              { id: `${PRACTICAL_QUANT}/${name}/normal-mode`, failureNote: () => note },
              async () => {
                const golden = await loadGolden(name);
                const inputs = graphInputs(golden, prepared.graph.inputs, name);
                monitor.enter("構築");
                const built = performance.now();
                const session = await prepared.createContainerSession(gpu, options);
                const buildMs = performance.now() - built;
                let observed: Awaited<ReturnType<typeof practicalRuns>>;
                try {
                  observed = await practicalRuns(
                    session,
                    gpu,
                    inputs,
                    prepared.graph.outputs[0],
                    (label) => monitor.enter(label),
                  );
                } finally {
                  monitor.enter("破棄");
                  await session.dispose();
                  await settleReleases(gpu);
                }
                const { peaks } = monitor.stop();
                note = [
                  `構築 ${(buildMs / 1000).toFixed(1)} s`,
                  ...observed.runs.map((run) => formatRun(run)),
                  ...[...peaks].map(([phase, peak]) =>
                    `VRAM 山 [${phase}] ${formatDrmUsage(peak)}`
                  ),
                  `非有限 ${observed.nonFinite}`,
                ].join(" / ");
                console.log(
                  `[wan-dit] 実用席 ${PRACTICAL_QUANT} S = ${tokens} の通常モード: ${note}`,
                );
                assertEquals(observed.nonFinite, 0, `${name}: 非有限`);
                assertEquals(deviceLost, undefined, "device lost");
                return { status: "pass", note };
              },
            );
          } finally {
            // 成功の経路では止め済み（2 度目は標本を 1 つ足すだけ）。失敗の経路で interval を残さない。
            monitor.stop();
          }
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "Wan DiT i8 参照席 f16+dit8（実 GPU・計測モード / diffusers CPU f64）: S = 14,040 の 1 forward が r の帯の内・" +
    "故障注入 5 件（f16 席と同じ 4 件 + scale の 2 倍）は帯の外・行ブロック・1 submit の GPU 時間 ≤ 1 s",
  ignore: !I8_FULL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const graph = i8Graph();
    const opened = await openSeriesContainer(new URL(MODEL_FILE, I8_SERIES_DIR));
    const prepared = prepareContainer(opened, graph);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    const base = await readRopeBase(opened);
    let deviceLost: string | undefined;
    const gpu = await acquireGpu({
      gpuTiming: true,
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    try {
      assertAdapterMatchesEnvironment(gpu);
      const band = I8_NORMALIZED_BAND[14040];
      const ratios = await normalizedRatioGate({
        t,
        gpu,
        prepared,
        base,
        seriesDir: I8_SERIES_DIR,
        idPrefix: I8_ID_PREFIX,
        cases: I8_FULL_CASES,
        band,
        full: {
          tokens: 14040,
          bindingLimit: gpu.device.limits.maxStorageBufferBindingSize,
          deviceLost: () => deviceLost,
        },
      });
      await i8ScaleFault(t, gpu, opened, graph, I8_FULL_CASES, band);
      await bandCandidate(t, `${I8_ID_PREFIX}S = 14,040`, I8_FULL_CASES, ratios, band);
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "Wan DiT i8 参照席 f16+dit8（実 GPU / diffusers CPU f64）: S = 192 の組の 1 forward が r の帯の内・" +
    "故障注入 5 件（f16 席と同じ 4 件 + scale の 2 倍）は帯の外",
  ignore: !I8_SMALL_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const graph = i8Graph();
    const opened = await openSeriesContainer(new URL(MODEL_FILE, I8_SERIES_DIR));
    const prepared = prepareContainer(opened, graph);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    const base = await readRopeBase(opened);
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const band = I8_NORMALIZED_BAND[192];
      const ratios = await normalizedRatioGate({
        t,
        gpu,
        prepared,
        base,
        seriesDir: I8_SERIES_DIR,
        idPrefix: I8_ID_PREFIX,
        cases: I8_SMALL_CASES,
        band,
      });
      await i8ScaleFault(t, gpu, opened, graph, I8_SMALL_CASES, band);
      await bandCandidate(t, `${I8_ID_PREFIX}S = 192 の組`, I8_SMALL_CASES, ratios, band);
    } finally {
      // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name: "Wan DiT 照合（実 GPU / diffusers CPU f32）: S 形の 1 forward が帯の内・故障注入は帯の外",
  ignore: !ASSETS_AVAILABLE || !GPU_AVAILABLE,
  fn: async (t) => {
    const opened = await openSeriesContainer(new URL(MODEL_FILE, SERIES_DIR));
    const prepared = prepareContainer(opened, GRAPH);
    assertEquals(prepared.graph.outputs.length, 1, "製品のグラフの出力は 1 本");
    const base = await readRopeBase(opened);
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const session = await prepared.createContainerSession(gpu);
      try {
        const run = async (inputs: Record<string, Tensor>): Promise<Float32Array> => {
          const outputs = await session.run(inputs);
          const tensor = outputs[prepared.graph.outputs[0]];
          if (tensor.dtype !== "f32") throw new Error(`出力が ${tensor.dtype}`);
          return tensor.data;
        };
        for (const { name, role } of CASES) {
          await t.step(`${name}（${role}）`, async () => {
            await runRecordedCase(results, { id: name }, async ({ measurements }) => {
              const golden = await loadGolden(name);
              const inputs = graphInputs(golden, prepared.graph.inputs, name);
              const expected = floatsOf(golden.reference, "output", name);
              const tokens = await run(inputs);
              const latents = unpatchifyTokens(tokens, golden.latentShape, WAN_GEOMETRY);
              const diff = difference(latents, expected);
              measurements.push({
                output: "latents",
                maxAbs: diff.maxAbs,
                maxRel: diff.maxRel,
                tolerance: recordedTolerance(diff),
                stage: "karume",
              });
              const notes = [formatDifference("io の入力", diff)];
              assertEquals(diff.nonFinite, 0, `${name}: 非有限`);
              assertEquals(
                { ...session.diagnostics().lastRunFusions, ...EXPECTED_FUSIONS },
                session.diagnostics().lastRunFusions,
                `${name}: 融合の件数（adaln / rope / silu）`,
              );

              // ホストの timesteps_proj（TS）で回した値（製品の経路の入力）も同じ比の帯で見る。入力そのものの
              // 差（torch と TS の `exp` の 1 ULP — 最悪 3.05e-5）は `wan_dit_host_test.ts` の絶対 atol 1.5e-4 が
              // 押さえる（値域が [-1, 1] の sin / cos なので絶対で足りる — 帯の指標を比に変えても据え置く）。
              const width = viewOf(golden.io, "input.timesteps_proj", name).shape[1];
              const hostTimestep = unpatchifyTokens(
                await run({
                  ...inputs,
                  timesteps_proj: {
                    dtype: "f32",
                    shape: [1, width],
                    data: timestepsProj(golden.timestep, width),
                  },
                }),
                golden.latentShape,
                WAN_GEOMETRY,
              );
              const hostDiff = difference(hostTimestep, expected);
              measurements.push({
                output: "latents@host-timesteps-proj",
                maxAbs: hostDiff.maxAbs,
                maxRel: hostDiff.maxRel,
                tolerance: recordedTolerance(hostDiff),
                stage: "karume",
              });
              notes.push(formatDifference("TS の timesteps_proj", hostDiff));
              const faults = role === "accept"
                ? await faultInjections(golden, inputs, tokens, expected, run, base)
                : [];
              notes.push(...faults.map(({ note }) => note));
              // 判定の前に出す（帯の外へ出た回も全ケースの比が手元に残る）。
              console.log(`[wan-dit] ${name}: ${notes.join(" / ")}`);

              // 故障注入の判定を先に置く（受入れが帯の外でも、帯が故障を拾えるかは判定される）。
              for (const { label, ratio } of faults) {
                assert(
                  ratio > DIT_RATIO_BAND,
                  `${name}: 故障注入 ${label} が帯 ${DIT_RATIO_BAND} の内に収まった`,
                );
              }
              if (role !== "growth") {
                assert(
                  ratioOf(diff) <= DIT_RATIO_BAND,
                  `${name}: ${notes[0]} が帯 ${DIT_RATIO_BAND} の外`,
                );
                assert(
                  ratioOf(hostDiff) <= DIT_RATIO_BAND,
                  `${name}: ${notes[1]} が帯 ${DIT_RATIO_BAND} の外`,
                );
              }
              return { status: "pass", note: notes.join(" / ") };
            });
          });
        }
      } finally {
        await session.dispose();
      }
    } finally {
      // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

/** 故障注入 1 件の結果（判定は呼び手が全ケースの比を出してから行う）。 */
type FaultResult = { readonly label: string; readonly ratio: number; readonly note: string };

/** 微妙な故障の注入のラベル（実寸の照合が {@link SUBTLE_FAULT_MARGIN} の門で引く）。 */
const TIMESTEP_OFF_BY_ONE = "timestep の 1 ずれ";

/**
 * 受入れケースの故障注入 4 件（追記 2026-10-02 — 帯が広すぎないことの裏取り）。どれも帯の外へ出ることを
 * 門にする（赤にならない注入があれば帯の決め方を見直す）。
 *
 * - RoPE の h / w の取り違え（非正方の格子で表を組み違える — 正方では値が一致して見えない。なので受入れの格子は
 *   非正方であることを要求し、正方なら注入を黙って飛ばさず落とす — 4 件が 3 件に減ったまま緑にしない）
 * - unpatchify の並びの取り違え（出口を入口の並び `(c,pt,ph,pw)` で読む）
 * - timestep の cos / sin の前後反転
 * - **timestep の 1 ずれ**（`timesteps_proj` を t + 1 で組む — TS の `timestepsProj`）。上の 3 件は O(1) の差で
 *   帯がどこにあっても赤になるが、これは比 2e-3〜9e-3 の微妙な故障で、帯の上限側の根拠になる。
 */
const faultInjections = async (
  golden: Golden,
  inputs: Record<string, Tensor>,
  tokens: Float32Array,
  expected: Float32Array,
  run: (inputs: Record<string, Tensor>) => Promise<Float32Array>,
  base: WanRopeBase,
): Promise<FaultResult[]> => {
  const grid = wanTokenGrid(golden.latentShape, WAN_GEOMETRY);
  const unpatchify = (data: Float32Array) =>
    unpatchifyTokens(data, golden.latentShape, WAN_GEOMETRY);
  const results: FaultResult[] = [];
  const measure = (label: string, actual: Float32Array) => {
    const diff = difference(actual, expected);
    const ratio = ratioOf(diff);
    results.push({
      label,
      ratio,
      note: `故障注入 ${label}: maxAbs ${diff.maxAbs.toExponential(3)}（比 ${
        ratio.toExponential(2)
      }）`,
    });
  };

  assert(
    grid.rows !== grid.cols,
    `受入れケースの格子 ${grid.frames}·${grid.rows}·${grid.cols} が正方 — RoPE の h / w の取り違えが値に出ない` +
      "（受入れには非正方の格子を選ぶ）",
  );
  const swapped = wanRopeTables(base, { ...grid, rows: grid.cols, cols: grid.rows });
  const shape = inputs.rope_cos.shape;
  measure(
    "RoPE の h / w 取り違え",
    unpatchify(
      await run({
        ...inputs,
        rope_cos: { dtype: "f32", shape, data: swapped.cos },
        rope_sin: { dtype: "f32", shape, data: swapped.sin },
      }),
    ),
  );
  measure("unpatchify の並び", unpatchify(transposeTokenAxes(tokens, WAN_GEOMETRY)));
  const proj = inputs.timesteps_proj;
  if (proj.dtype !== "f32") throw new Error(`timesteps_proj が ${proj.dtype}`);
  const half = proj.data.length / 2;
  const flipped = Float32Array.from(
    { length: proj.data.length },
    (_, index) => proj.data[index < half ? half + index : index - half],
  );
  measure(
    "timestep の cos / sin 反転",
    unpatchify(
      await run({ ...inputs, timesteps_proj: { dtype: "f32", shape: proj.shape, data: flipped } }),
    ),
  );
  measure(
    TIMESTEP_OFF_BY_ONE,
    unpatchify(
      await run({
        ...inputs,
        timesteps_proj: {
          dtype: "f32",
          shape: proj.shape,
          data: timestepsProj(golden.timestep + 1, proj.shape[1]),
        },
      }),
    ),
  );
  return results;
};

Deno.test({
  name: "Wan DiT 層別（実 GPU・計測用グラフ）: 各ブロックの出力の差を記録する（門ではない）",
  ignore: !PROBE_AVAILABLE || !GPU_AVAILABLE,
  fn: async () => {
    const prepared = prepareContainer(
      await openSeriesContainer(new URL(MODEL_FILE, PROBE_DIR)),
      PROBE_GRAPH,
    );
    const outputs = prepared.graph.outputs;
    const gpu = await acquireGpu();
    try {
      assertAdapterMatchesEnvironment(gpu);
      const session = await prepared.createContainerSession(gpu);
      try {
        for (const { name } of CASES) {
          await runRecordedCase(results, { id: `${name}/layers` }, async ({ measurements }) => {
            const golden = await loadGolden(name);
            const produced = await session.run(graphInputs(golden, prepared.graph.inputs, name));
            const rows = layerRatios(produced, outputs, golden, name, measurements);
            return { status: "pass", note: rows.join(" ") };
          });
        }
      } finally {
        await session.dispose();
      }
    } finally {
      // device を捨てる前に解放を待つ（後続のテストの予算を残す — {@link settleReleases}）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

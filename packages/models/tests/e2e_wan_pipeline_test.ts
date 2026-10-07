/**
 * Wan2.1 のパイプライン（`WanPipeline` — ADR 0118 段 6 の結線・段 7 の配布形）の通しの照合（実 GPU）。
 *
 * 配布形ミラー `models/karume-wan2.1/`（`dist.py --pipeline wan` が組む — `krm` 3 本とテキスト埋め込み
 * 資産）を `denoDirectory` で `WanPipeline.fromPretrained` へ渡して組み、`generate` を通す（公開面の
 * 取得面 — manifest の解決・家族 admission・part の逐次読み）。参照（`pipeline_steps.*`）は配布しない
 * golden なので系列 `outputs/series/wan2.1-t2v-1.3b-f16-dyn/` から読む。
 *
 * 配布形が無い機ではこの e2e は明示 SKIP する（理由と組み立てのコマンドを出す）。それで全 SKIP に
 * なるのを FAIL にするのは門番 `packages/runtime/tests/distribution_gate_test.ts`（意図して通すなら
 * `KARUME_ALLOW_NO_DISTRIBUTION=1` — anima / gemma4 の e2e と同じ分担）。
 *
 * 比べる相手は recipe の少ステップの参照（`tools/export-recipes/wan/few_step_ref.py`）: diffusers の
 * `WanPipeline` を CPU f32 で素のまま 2 ステップ（CFG あり・guide 5.0・shift 3.0）回した潜在と、それを
 * 段 5 のタイル decode に通したフレーム（重みは f16 へ丸めた値 — ADR 0006）。初期ノイズは参照の
 * `latents_init`（torch の `randn`）を `latents` で注入する（seed 経路の 1 本 — {@link SEED_CASE} — を除く）。
 *
 * ## 門（既定のレーン — 決定 8「レーンの既定は少ステップ」）
 *
 * この e2e が持つのは**結線**の門（プロンプトの取り違え・guidance の揺れ・step の順・段の受け渡し）で
 * あって、DiT の forward 1 本の数値の門ではない。forward の数値は段 3（`e2e_wan_dit_test.ts`）が f64 の
 * 参照に対する正規化比 r で持つ。なのでここの相手は CPU f32 の参照のままでよい — 帯は GPU と CPU f32 の
 * 丸めの差（段 3 の帰属 — GEMM の K 縮約）を含んだ幅で、結線の誤りはその外へ出る（故障注入で確かめる）。
 *
 * - **ホストの自己整合**（GPU 不要）: 参照が記録した DiT の出力（cond / uncond）から、ホストの CFG +
 *   UniPC が参照の潜在を**ビット一致**で再現する（参照の精度の確認と、ホストの結線の門を兼ねる）。
 * - **帯**（決定 8 の指標 = 比 = 最大絶対差 ÷ 参照の最大絶対値 — CPU f32 の参照に対する比）: 決定用 2 本
 *   （`band-boxing-cats`・seed 20261030 / `band-cat-dog-baking`・seed 20261032）の実測最悪 × 5 で
 *   観測点ごと（各 step の後の潜在・フレーム）に帯を決め（{@link LATENT_RATIO_BANDS} /
 *   {@link FRAME_RATIO_BAND}）、受入れ 1 本（`accept-ferret`・seed 20261033 — 決定用と seed も
 *   プロンプトも違う）で受け入れる（決定 8 の形 — 段 3 と同じ決定用 / 受入れの分け方）。
 * - **故障注入**（受入れの初期ノイズ）: プロンプトの正負の取り違え・guide の 1% のずれ が帯の外
 *   （床 {@link FAULT_MARGIN} 倍）。
 * - **sha256 の環境行**（ADR 0106）: 出力フレーム（uint8 の RGB を全フレーム連結したバイト列 —
 *   `wanFrameToRgba` の規則）を `fixtures/references/wan.json` の環境キーの行と突き合わせる。
 *   行は `KARUME_REFERENCE=write` で作る。B570 の行（2026-10-02）は段 6 で**系列を直読み**
 *   （`fromAssets`）して書いた値で、配布形経由（`fromPretrained`）でも同じ行と一致することを
 *   ここで要求する — 配布形は系列の `krm` の独立コピーで、manifest の既定（50 / 5.0 / 3.0）も段 6 の
 *   定数と同じ値なので、1 ビットでも割れたら取得面か既定の解決の退行。
 * - **席は全ケースで明示する**（{@link F16_QUANT} / i8 の 2 席）: 既定席は manifest の `defaultQuant` が決め、
 *   2026-10-04 に `f16` から実用席へ移った（ADR 0120 裁定 2026-10-04 の 4）。席名を持たない既存の ID の行・帯・
 *   CPU の参照は `f16` 席の値なので、`quant` を省くと同じ ID が別の席の値で回る（{@link loadPipeline} は席を
 *   必須にして、省略を型で止める）。
 * - **seed 経路**（{@link SEED_CASE}）: `latents` を注入せず seed から初期ノイズを作る 2 ステップ 1 本。CPU の
 *   参照は無いので、完走・非有限 0 と sha256 の環境行だけで縛る。
 * - **i8 の 2 席の sha 行**（ADR 0120 決定 4 / 段 5 — {@link SEAT_QUANTS}）: seed 経路と同じ要求を参照席 `f16+dit8` と
 *   実用席 `f16+dit8-a8-attn8-s16` で 1 本ずつ。ID は席名を先頭に置く（{@link seatCaseId} — anima の `<quant>-<解像度>`
 *   と同じ並び。既存の `f16` 席の行の ID は動かさない）。行のクラスは ID でも別表でもなく、manifest の席の `session` が
 *   空か否かから導く（ADR 0110 決定 7 — {@link rowClassOf}）: `f16+dit8` は**参照行**（凍結 — 実用層の変更コミットで
 *   触らない）、実用席は**実用行**（実用層の退行と決定性の検出器 — 実用層の数値を意図して変えるコミットで同じコミットの
 *   `rewrite`）。CPU の参照は f16 の重みでしか採っていないので、i8 の席は帯を持たず sha256 の環境行だけで縛る
 *   （数値の門は DiT 単体の r 門と自機 A/B 門 — `e2e_wan_dit_test.ts` / `e2e_wan_ab_test.ts`）。
 * - **テキストエンコーダの経路**（ADR 0119 決定 7）: 上の sha 行と帯のケースは全部**事前計算の埋め込み資産の経路**
 *   （`textEncoder: "precomputed"` を明示 — 既定は `"gpu"` なので、明示しないと umT5 の経路へ黙って移り、行の値が
 *   別の経路の sha と突き合わさる）。**GPU 経路**（umT5 i8 を GPU で回す — {@link GPU_TEXT_CASES}）は別の case id の
 *   sha 行（固定プロンプト `boxing-cats` と自由プロンプト 1 本・2 ステップ・seed 42）で、CPU の参照は持たない（umT5 の
 *   数値の門は `e2e_wan_umt5_test.ts`）。umT5 は別の配布リポ（`models/karume-umt5-xxl/` — Wan の manifest の
 *   `text_encoder` が越境参照する）で、取得元の `crossRepo` の mapping で渡す。無い機ではその 2 本だけ明示 SKIP。
 *
 * ## 50 ステップの通し（env の opt-in — 既定のレーンに入れない）
 *
 * `KARUME_WAN_FULL_PIPELINE=1` のときだけ、既定の設定（50 ステップ・CFG・832×480・33 フレーム・
 * `boxing-cats`・seed 42）と、同じ設定の 81 フレーム（受理集合の上限 — ADR 0118 段 8）を、`f16` 席と実用席
 * `f16+dit8-a8-attn8-s16`（ADR 0120 段 5 — 81 フレームの実用行と段 6 の視認の素材）で 1 本ずつ回し、
 * 完走・非有限 0・所要・段の切り替えの VRAM・フレームの PNG（全フレーム + 4×8 の一覧図）と RGB の実物を
 * `outputs/verify/<環境キー>/<日付>_wan-pipeline-full/` に書き、sha256 の環境行は
 * `fixtures/references/wan.json` に持つ（PNG は利用者の視認 — perf-ledger K-75 の判断材料）。CPU の参照は作らない（決定 8 — sha256 と目視で受ける。81 フレームの部品は DiT の実寸・VAE の
 * 21 chunk の e2e が覆う）。
 * 結果の席を既定のレーン（`<日付>_wan-pipeline/`）と分けるのは、`results.json` が走行ごとに丸ごと
 * 書き直されるため — 同じ席だと、後から回した 2 ステップだけの走行が 50 ステップの記録を消す。
 *
 * ## 観測（門ではない）
 *
 * 段の所要（壁時計 — `stage` イベントの間）と、段の境目ごとの VRAM（`/proc/self/fdinfo` の
 * `drm-total-*` — `helpers/drm-usage.ts`）。DiT の段の `end`（Session を畳んだ直後・VAE の段を張る前）の
 * 値で、B570 の `destroy()` の遅れが DiT と VAE の確保を重ねていないかを見る（2026-10-02 の実測では
 * 段の前の値へ戻っていた — 解放待ちを入れなかった根拠。`src/wan/pipeline.ts` のモジュール doc）。
 * GPU 時間は採らない — 計測の device では VAE の batch を開けない（`WanPipeline` の構築が拒む）。
 *
 * 配布形か参照が無い環境は明示 SKIP、参照が一部だけある環境は FAIL（段 4 / 5 の e2e と同じ規律）。
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  type DistributionSource,
  localDirectory,
  parseManifest,
  resolveSelection,
} from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import {
  type GpuContext,
  parseSafetensors,
  type SafetensorsFile,
  type SessionDiagnostics,
} from "@karume/runtime";
import { encodePng } from "../mod.ts";
import {
  type GeneratedVideo,
  wanFrameToRgba,
  type WanGenerateEvent,
  type WanGenerateRequest,
  WanPipeline,
  type WanPipelineOptions,
  type WanPrompt,
  type WanRunComponent,
} from "../wan.ts";
import { parseWanTextEmbeds } from "../src/wan/text-embeds.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../src/wan/scheduler.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { WAN_ASSEMBLE_COMMAND } from "./helpers/wan-distribution.ts";
import { type DrmTimeline, formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import {
  openReferences,
  referenceMismatchMessage,
  type ReferenceSettlement,
  registerReferenceGate,
  settleOrObserve,
  sha256Hex,
} from "../../runtime/tests/helpers/reference.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";

/**
 * 2 ステップの通しの潜在（各 step の後 — `[16,9,60,104]`）の帯（比 = 最大絶対差 ÷ 参照の最大絶対値）。
 * 添字が step（0 = t 999 の後・1 = t 750 の後）。
 *
 * 実測（2026-10-02・`deno-intel-graphics-bmg-g21`・`atol = rtol = 0` の素の突合）。帯は決定用 2 本だけを
 * 回して決め、受入れはその後に初めて回した（受入れの値は帯の決定に使っていない）:
 *
 * | ケース                | 役割   | `latents.0`（t 999） | `latents.1`（t 750） | フレーム（クランプ後） |
 * | --------------------- | ------ | -------------------- | -------------------- | ---------------------- |
 * | `band-boxing-cats`    | 決定用 | **1.700e-5**         | **1.815e-4**         | **1.691e-3**           |
 * | `band-cat-dog-baking` | 決定用 | 1.228e-5             | 5.871e-5             | 3.094e-4               |
 * | 帯（最悪 × 5）        |        | 8.5e-5               | 9.1e-4               | 8.5e-3                 |
 * | `accept-ferret`       | 受入れ | 1.393e-5             | 5.069e-4             | 5.698e-3               |
 *
 * 受入れは帯の 0.16 / 0.56 / 0.67 倍。
 *
 * 帯 = 観測点ごとに決定用 2 本の最悪 × 5（有効数字 2 桁へ切り上げ）。差はホストではなく DiT の GPU の
 * 丸めから来る（段 3 の帰属 — GEMM の K 縮約・perf-ledger K-75）: ホストの CFG + UniPC は参照の DiT
 * 出力から参照の潜在を**ビット一致**で作り直す（下の GPU 不要のテスト）ので、差は全部 DiT の 4 forward
 * から来る。CFG（`5·cond − 4·uncond`）が cond / uncond の 2 本の差を足し合わせ、step 1 は step 0 の差を
 * 引き継ぐ。
 *
 * 故障注入（受入れの初期ノイズ・`latents.1`）: プロンプトの正負の取り違え 1.777（帯の約 1,950 倍）・
 * guide 5.0 → 5.05 1.333e-1（約 147 倍）。床は {@link FAULT_MARGIN} 倍。
 *
 * MUST: 受入れ（`accept-ferret`）の結果を見てこの値も、決定用のケースも変えない（帯の決定と受入れの
 * 独立が崩れる）。受入れが帯を外れたら帯を広げずに原因を調べる。
 */
const LATENT_RATIO_BANDS = [8.5e-5, 9.1e-4] as const;

/**
 * 2 ステップの通しの出力フレーム（クランプ後の `[3,33,480,832]` — 参照もクランプして比べる）の帯。
 *
 * 8.5e-3 = 決定用の最悪（`band-boxing-cats` の 1.691e-3）× 5（{@link LATENT_RATIO_BANDS} の表）。差の
 * 大半は潜在の差が VAE を通って広がった分（段 5 のタイル decode そのものの比は 2.65e-6）。
 *
 * MUST: 受入れの結果を見てこの値を変えない（{@link LATENT_RATIO_BANDS} と同じ）。
 */
const FRAME_RATIO_BAND = 8.5e-3;

/**
 * 故障（プロンプトの正負の取り違え・guide の 1% のずれ）が帯から離れているべき倍率の床（追記
 * 2026-10-02 の「赤にならない注入は帯が広すぎる兆候」の床 — 段 3 と同じ 2）。
 */
const FAULT_MARGIN = 2;

/** 参照（配布しない golden）の置き場。 */
const SERIES_ROOT = new URL("../../../outputs/series/wan2.1-t2v-1.3b-f16-dyn/", import.meta.url);
/** 配布形ミラー（`dist.py --pipeline wan` の既定の出力先）。 */
const DIST_ROOT = new URL("../../../models/karume-wan2.1/", import.meta.url);
/** manifest の `assets` の埋め込み資産のキー（recipe `wan/distribution.py` の `WAN_TEXT_EMBEDS_ROLE`）。 */
const TEXT_EMBEDS = "text_embeds";
const STEPS = 2;

/**
 * 参照のケース（正本は recipe `wan/few_step_ref.py` の `FIXTURE_CASES`）。決定用（`band`）2 本 +
 * 受入れ（`accept`）1 本。MUST: 受入れの結果を見て決定用のケースを変えない。
 */
const CASES: readonly { readonly name: string; readonly role: "band" | "accept" }[] = [
  { name: "band-boxing-cats", role: "band" },
  { name: "band-cat-dog-baking", role: "band" },
  { name: "accept-ferret", role: "accept" },
];
const ACCEPT_CASE = "accept-ferret";

/**
 * seed 経路の 2 ステップ（`latents` を注入せず、seed から初期ノイズを作る — 生成器 `WanRandn` と seed → 潜在の
 * 並べ方を通す）。生成器の列は torch の `randn` と別なので CPU の参照は作れず、値は sha256 の環境行だけで縛る。
 * 既定のレーンに置くのは、seed 経路の値を縛る行がほかに opt-in の 50 ステップにしか無いため（既定のレーンと
 * フル verify が生成器の退行を素通しする）。
 */
const SEED_CASE = { id: "2step-seed-boxing-cats-seed42", prompt: "boxing-cats", seed: 42 } as const;

/**
 * `f16` 席（参照側 — 元の重みにいちばん近い席。CPU の参照〈f16 へ丸めた重み〉と帯を持つのはこの席だけ）。席名を
 * 持たない既存の ID の行はこの席の値（ADR 0120 裁定 2026-10-04 の 4 — 既定席が実用席へ移っても値を保つため明示する）。
 */
const F16_QUANT = "f16";
/** 実用席（ADR 0120 決定 1 — 2026-10-04 から manifest の既定席）。 */
const PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";
/**
 * 2 ステップの seed 経路を `f16` 席に加えて回す i8 の 2 席（ADR 0120 決定 4 / 段 5）: 参照席（参照行）と実用席（実用行）。
 * 行のクラスは {@link rowClassOf} が manifest から導く — この並びはクラスを主張しない。
 */
const SEAT_QUANTS = ["f16+dit8", PRACTICAL_QUANT] as const;

/** 席を名乗るケースの ID（席名が先頭 — `f16` 席の既存の行の ID は席名を持たないまま動かさない）。 */
const seatCaseId = (quant: string, base: string): string => `${quant}-${base}`;

/**
 * GPU 経路のケース（ADR 0119 追記「段 10d の設計」の sha 行 — 2 ステップ・seed 42・`f16` 席・既定の negative）。
 * 固定プロンプトは資産の行の原文を渡し（同じ文字列で経路だけが違う — 資産の経路の {@link SEED_CASE} と比べられる）、
 * 自由プロンプトは資産に無い短い英文（ID は綴りを持たず名前で呼ぶ — 文面を変えたら ID も変える）。
 *
 * MUST: 文面・seed・step 数を変えたら ID も変える（行の値が別の条件の sha と突き合わさる）。
 */
const GPU_TEXT_CASES: readonly {
  readonly id: string;
  readonly prompt: { readonly asset: string } | { readonly text: string };
  readonly seed: number;
}[] = [
  { id: "gpu-text-2step-boxing-cats-seed42", prompt: { asset: "boxing-cats" }, seed: 42 },
  {
    id: "gpu-text-2step-free-fox-seed42",
    prompt: { text: "A red fox trots through fresh snow in a quiet birch forest at sunrise." },
    seed: 42,
  },
];

/** 50 ステップの通しの opt-in（決定 8 — リリース前と参照値の焼き直しのときだけ回す）。 */
const FULL_PIPELINE = Deno.env.get("KARUME_WAN_FULL_PIPELINE") === "1";
/**
 * 50 ステップの通しの要求: 既定の設定 + 1 本目の固定プロンプト・seed 42（段 7 の目視の 1 本目）と、同じ要求の
 * 81 フレーム（ADR 0118 段 8）を、`f16` 席と実用席（ADR 0120 段 5 — 81 フレームの実用行と段 6 の
 * 視認の素材）で。`frames` を省いたケースは既定（{@link DEFAULT_FRAMES}）を通す。
 * 並びは確保の大きい順（`f16` 席の 81 → 実用席の 81 → `f16` 席の 33 → 実用席の 33 — DiT 段の VRAM の山は実用席が
 * 約 1 GiB 小さい見込み・ADR 0120 期待値の表）。前の確保の残りが後ろの大きな確保を OOM にしうる — B570 の
 * `destroy()` の遅れ・`e2e_wan_dit_test.ts` で実寸を先に置くのと同じ理由。
 */
const FULL_CASES: readonly {
  readonly id: string;
  readonly prompt: string;
  readonly seed: number;
  readonly frames?: number;
  /** 席（必須 — 省略すると manifest の `defaultQuant` へ黙って移る）。 */
  readonly quant: string;
}[] = [
  {
    id: "50step-boxing-cats-seed42-81f",
    prompt: "boxing-cats",
    seed: 42,
    frames: 81,
    quant: F16_QUANT,
  },
  {
    id: seatCaseId(PRACTICAL_QUANT, "50step-boxing-cats-seed42-81f"),
    prompt: "boxing-cats",
    seed: 42,
    frames: 81,
    quant: PRACTICAL_QUANT,
  },
  { id: "50step-boxing-cats-seed42", prompt: "boxing-cats", seed: 42, quant: F16_QUANT },
  {
    id: seatCaseId(PRACTICAL_QUANT, "50step-boxing-cats-seed42"),
    prompt: "boxing-cats",
    seed: 42,
    quant: PRACTICAL_QUANT,
  },
];
/** `frames` を省いたときのフレーム数（`src/wan/pipeline.ts` の既定）。 */
const DEFAULT_FRAMES = 33;
/**
 * 50 ステップの通しの step 数。要求では指定せず manifest の既定に任せるので、観測した denoise-step の数で縛る
 * （既定が変わった配布形で `KARUME_REFERENCE=write` を回すと、別の step 数の出力が `50step-…` の行に書かれる）。
 */
const FULL_STEPS = 50;

const GENERATE_COMMAND = "cd tools/export-recipes && uv run --group wan --inexact " +
  "python -m wan.text_embeds && uv run --group wan --inexact python -m wan.few_step_ref";
const ASSEMBLE_UMT5_COMMAND = "cd tools/export-recipes && uv run python dist.py --pipeline umt5";

/** umT5 の配布形ミラー（Wan の manifest の `text_encoder` が越境参照する先 — ADR 0119 追記「段 10d の設計」D）。 */
const UMT5_ROOT = new URL("../../../models/karume-umt5-xxl/", import.meta.url);
/** manifest の部品名（`src/wan/pipeline.ts` の `TEXT_ENCODER`）。 */
const TEXT_ENCODER = "text_encoder";

const fixtureUrl = (name: string): URL =>
  new URL(`pipeline_steps.${name}.safetensors`, SERIES_ROOT);

const fileExists = (url: URL): boolean => {
  try {
    return Deno.statSync(url).isFile;
  } catch (cause) {
    if (cause instanceof Deno.errors.NotFound) return false;
    throw cause;
  }
};

const DIST_PRESENT = fileExists(new URL("karume.json", DIST_ROOT));
if (!DIST_PRESENT) {
  console.warn(
    `[karume] 配布形ミラー ${DIST_ROOT.pathname} が無いため Wan のパイプラインの照合を SKIP する。` +
      `組み立て: ${WAN_ASSEMBLE_COMMAND}（全 SKIP は門番 distribution_gate_test.ts が FAIL にする）`,
  );
}
const FIXTURES_PRESENT = CASES.map(({ name }) => fileExists(fixtureUrl(name)));
const ANY_FIXTURE = FIXTURES_PRESENT.some(Boolean);
if (!ANY_FIXTURE) {
  console.warn(
    `[karume] ${SERIES_ROOT.pathname} に少ステップの参照が無いため Wan のパイプラインの照合を SKIP ` +
      `する。生成: ${GENERATE_COMMAND}`,
  );
}
/** 照合を回せる（配布形と参照の両方がある — 参照が一部だけの機は下の資産の門が FAIL にする）。 */
const ANY_PRESENT = DIST_PRESENT && ANY_FIXTURE;

/**
 * 配布形の既定の選択が持つ umT5 の容器の part 0（無ければ undefined — umT5 を持たない旧い配布形）。越境先の repo は
 * この宣言から引く（キーを写経しない — 配布形の再生成で repo が変わっても mapping が追従する）。
 */
const TEXT_ENCODER_PART0 = DIST_PRESENT
  ? resolveSelection(parseManifest(await Deno.readTextFile(new URL("karume.json", DIST_ROOT))))
    .containers[TEXT_ENCODER]?.parts[0]
  : undefined;
/** umT5 の越境先の repo（自リポの容器なら undefined — mapping が要らない）。 */
const UMT5_REPO = TEXT_ENCODER_PART0?.repo;
const UMT5_PRESENT = fileExists(new URL("karume.json", UMT5_ROOT));
/** GPU 経路を組める（配布形が umT5 を持ち、越境なら越境先のミラーもある）。 */
const GPU_TEXT_PRESENT = DIST_PRESENT && TEXT_ENCODER_PART0 !== undefined &&
  (UMT5_REPO === undefined || UMT5_PRESENT);
if (DIST_PRESENT && !GPU_TEXT_PRESENT) {
  console.warn(
    TEXT_ENCODER_PART0 === undefined
      ? `[karume] 配布形ミラー ${DIST_ROOT.pathname} が umT5（${TEXT_ENCODER}）を持たないため、Wan の GPU 経路の` +
        `ケースと umT5 の admission の故障を SKIP する。組み直し: ${WAN_ASSEMBLE_COMMAND}`
      : `[karume] umT5 の配布形ミラー ${UMT5_ROOT.pathname}（${UMT5_REPO} の越境先）が無いため、Wan の GPU 経路の` +
        `ケースと umT5 の admission の故障を SKIP する。組み立て: ${ASSEMBLE_UMT5_COMMAND}`,
  );
}

/** 越境先の mapping（umT5 の容器が自リポなら空 — 渡しても使われない）。 */
const crossRepoOf = (umt5: DistributionSource): Record<string, DistributionSource> =>
  UMT5_REPO === undefined ? {} : { [UMT5_REPO]: umt5 };

/** 配布形の埋め込み資産（manifest の既定の選択の `assets` から引く — path を綴り直さない）。 */
const readDistributionEmbeds = async (): Promise<Uint8Array<ArrayBuffer>> => {
  const manifest = parseManifest(await Deno.readTextFile(new URL("karume.json", DIST_ROOT)));
  const ref = resolveSelection(manifest).assets[TEXT_EMBEDS];
  assert(ref !== undefined, `配布形の manifest の assets に '${TEXT_EMBEDS}' が無い`);
  return await Deno.readFile(new URL(ref.path, DIST_ROOT));
};

/**
 * 配布形を取得元ハンドルで読む（network も CacheStorage も通らない）。経路と席は呼び手が必ず名乗る（既定の
 * `"gpu"` にも manifest の既定席にも黙って乗せない — モジュール doc の「テキストエンコーダの経路」と「席は全ケースで
 * 明示する」）。
 * `"gpu"` は umT5 の越境先を `crossRepo` の mapping で渡す。
 */
const loadPipeline = (
  gpu: GpuContext,
  diagnostics: Map<WanRunComponent, SessionDiagnostics>,
  textEncoder: NonNullable<WanPipelineOptions["textEncoder"]>,
  quant: string,
): Promise<WanPipeline> =>
  WanPipeline.fromPretrained(
    textEncoder === "gpu"
      ? denoDirectory(DIST_ROOT, { crossRepo: crossRepoOf(denoDirectory(UMT5_ROOT)) })
      : denoDirectory(DIST_ROOT),
    {
      gpu,
      textEncoder,
      quant,
      onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
    },
  );

/**
 * sha 行のクラス（ADR 0110 決定 7）を、配布形の manifest の既定モデルの席の `session` から導く（空 = 参照行・非空 =
 * 実用行）。別表に持たない — 席の宣言が変われば表示も変わる。
 */
const rowClassOf = async (quant: string): Promise<string> => {
  const manifest = parseManifest(await Deno.readTextFile(new URL("karume.json", DIST_ROOT)));
  const model = manifest.models[manifest.defaultModel];
  assert(model !== undefined, `配布形の manifest に既定モデル '${manifest.defaultModel}' が無い`);
  const seat = Object.hasOwn(model.quants, quant) ? model.quants[quant] : undefined;
  assert(seat !== undefined, `配布形の manifest に席 '${quant}' が無い`);
  return Object.keys(seat.session).length === 0
    ? `${quant}: 参照行（session 空 — 凍結）`
    : `${quant}: 実用行（session 非空 — 実用層の退行と決定性の検出器）`;
};

const references = openReferences(new URL("fixtures/references/wan.json", import.meta.url));
const results = openResults("wan-pipeline");
/** 50 ステップの通しの結果の席（既定のレーンと分ける — モジュール doc）。 */
const fullResults = openResults("wan-pipeline-full");

Deno.test({
  name: "Wan 通しの参照: 少ステップの参照 3 本が揃っている",
  ignore: !ANY_FIXTURE,
  fn: () => {
    assertEquals(
      FIXTURES_PRESENT,
      FIXTURES_PRESENT.map(() => true),
      `${SERIES_ROOT.pathname} の欠け`,
    );
  },
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * 配布形の manifest の既定モデルを書き換えた写し（`karume.json` だけを差し替える故障注入 — 重みは
 * 1 バイトも書き換えない）。故障は manifest の型の外の値も入れるので、`parseManifest` を通さず
 * unknown のまま型述語で降りる。
 */
const overrideModel = (
  original: string,
  patch: (model: Record<string, unknown>) => Record<string, unknown>,
): Record<string, unknown> => {
  const manifest: unknown = JSON.parse(original);
  assert(isRecord(manifest) && isRecord(manifest.models), "配布形の manifest に models が無い");
  const name = manifest.defaultModel;
  assert(typeof name === "string", "配布形の manifest に defaultModel が無い");
  const model = manifest.models[name];
  assert(isRecord(model), `配布形の manifest に既定モデル '${name}' が無い`);
  return { ...manifest, models: { ...manifest.models, [name]: patch(model) } };
};

/** umT5 の越境先から読んだ path の印（自リポの path と同じ綴りでも取り違えない）。 */
const UMT5_PREFIX = "umt5:";

/**
 * `karume.json` だけを `manifest` に差し替え、残りは配布形から読む取得元（読んだ path を
 * `requested` に積む — 取得の順と回数を数えるため）。umT5 の越境先も同じ列へ {@link UMT5_PREFIX} を付けて積む
 * （越境先のミラーが無い機では mapping を渡さない — 触れば hub が「越境先が無い」で落とす）。
 */
const countingSource = (manifest: Record<string, unknown>, requested: string[]) => {
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  const umt5 = localDirectory({
    readFile: async (path) => {
      requested.push(`${UMT5_PREFIX}${path}`);
      return await Deno.readFile(new URL(path, UMT5_ROOT));
    },
  }, { label: "wan-admission-fault-umt5" });
  return localDirectory({
    readFile: async (path) => {
      requested.push(path);
      if (path === "karume.json") return manifestBytes;
      return await Deno.readFile(new URL(path, DIST_ROOT));
    },
  }, { label: "wan-admission-fault", ...(UMT5_PRESENT ? { crossRepo: crossRepoOf(umt5) } : {}) });
};

Deno.test({
  name:
    "Wan 配布形の家族 admission（GPU 不要）: pipeline の major・pipelineConfig・quant の session の齟齬・" +
    "計測（gpuTiming）の共有 device・GPU 経路のトークナイザ資産の欠落は名指しで落ち、それまでに重みの part 1 以降と" +
    "資産を 1 本も取らない（事前計算の経路は umT5 に 1 バイトも触らない）",
  ignore: !DIST_PRESENT,
  fn: async (t) => {
    const original = await Deno.readTextFile(new URL("karume.json", DIST_ROOT));
    const parsed = parseManifest(original);
    // part 0 = descriptor は admission の入力そのもので、門より前に取る契約
    // （packages/models/src/hub/components.ts の相 1）。門の後にしか触れてはいけないのは part 1
    // 以降と assets（埋め込み・トークナイザ・RoPE の素表）。
    const model = parsed.models[parsed.defaultModel];
    assert(model !== undefined, `配布形の manifest に既定モデル '${parsed.defaultModel}' が無い`);
    const heavyPaths = new Set([
      ...Object.values(model.weights).flatMap((entry) =>
        Object.values(entry).flatMap((weights) =>
          weights.container.parts.slice(1).map((ref) =>
            ref.repo === undefined ? ref.path : `${UMT5_PREFIX}${ref.path}`
          )
        )
      ),
      ...Object.values(model.assets).map((ref) => ref.path),
    ]);
    assert(heavyPaths.size > 0);
    const faults: readonly {
      readonly label: string;
      readonly patch: (model: Record<string, unknown>) => Record<string, unknown>;
      /** 経路（既存の故障は事前計算の経路 — umT5 の有無に依らずに回す）。 */
      readonly textEncoder: NonNullable<WanPipelineOptions["textEncoder"]>;
      /** 共有で渡す GPU（manifest ではなく構築のオプション側の齟齬）。 */
      readonly gpu?: GpuContext;
      readonly message: string;
    }[] = [
      {
        label: "pipeline の major 2",
        patch: (model) => ({ ...model, pipeline: "wan/2" }),
        textEncoder: "precomputed",
        message: "major に未対応",
      },
      {
        label: "pipelineConfig の未知キー",
        patch: (model) => ({
          ...model,
          pipelineConfig: {
            scheduler: { shift: 3, type: "unipc" },
            defaults: { steps: 50, guidance: 5 },
          },
        }),
        textEncoder: "precomputed",
        message: "未知キー 'type'",
      },
      {
        // 受理表（WAN_SESSION_POLICY）が受けるのは linear / attention の実行形 3 欄だけ
        // （ADR 0120 決定 7）。manifest 語彙としては正しいが Wan が受けないノブ（states 形 attention の
        // 縮約 — decoder の decode 向け）の宣言は、重みを取る前に名指しで落ちる。
        label: "quant が受理表に無い実行ノブを宣言",
        patch: (model) => {
          const quants = model.quants;
          assert(isRecord(quants) && isRecord(quants.f16), "配布形に f16 の quant が無い");
          return {
            ...model,
            quants: {
              ...quants,
              f16: { ...quants.f16, session: { stateAttentionReduce: "parallel" } },
            },
          };
        },
        textEncoder: "precomputed",
        message: "session.stateAttentionReduceは未対応",
      },
      {
        // VAE の段は 1 タイル = 1 batch で、runtime は計測の device で batch を開かない。DiT の段を
        // 回し終えてから落ちる形にせず、重みを取る前に落とす（`WanPipelineOptions.gpu` の MUST）。
        // device は timestamp-query だけを持つフェイク（GPU を取らない）。
        label: "計測（gpuTiming）の device を共有で渡す",
        patch: (model) => model,
        textEncoder: "precomputed",
        gpu: fakeGpuContext(fakeDevice({ features: ["timestamp-query"] })),
        message: "gpuTiming が有効な device",
      },
      {
        // GPU 経路はトークナイザ資産を読む — 宣言が無ければ umT5（i8 で約 5.3 GiB）の part 1 以降を取る前に落とす。
        // umT5 の容器の part 0（descriptor）は admission の入力なので取ってよい。
        label: "GPU 経路で manifest に umT5 のトークナイザ資産が無い",
        patch: (model) => {
          const assets = model.assets;
          assert(isRecord(assets), "配布形の manifest に assets が無い");
          const { umt5_tokenizer: _dropped, ...rest } = assets;
          return { ...model, assets: rest };
        },
        textEncoder: "gpu",
        message: "umT5 のトークナイザ資産 'umt5_tokenizer' が無い",
      },
    ];
    for (const { label, patch, textEncoder, gpu, message } of faults) {
      await t.step({
        name: `${label}（textEncoder: ${textEncoder}）`,
        ignore: textEncoder === "gpu" && !GPU_TEXT_PRESENT,
        fn: async () => {
          const requested: string[] = [];
          const source = countingSource(overrideModel(original, patch), requested);
          await assertRejects(
            () =>
              WanPipeline.fromPretrained(source, {
                textEncoder,
                // 故障の 1 本は `f16` 席の宣言を書き換える — 席を名乗らないと既定席へ移って故障に触れない。
                quant: F16_QUANT,
                ...(gpu === undefined ? {} : { gpu }),
              }),
            Error,
            message,
          );
          assert(requested.includes("karume.json"), `manifest を読んでいない: ${requested}`);
          const heavy = requested.filter((path) => heavyPaths.has(path));
          assertEquals(heavy, [], "admission の前に重みの part 1 以降・資産を取っている");
          if (textEncoder === "precomputed") {
            const umt5 = requested.filter((path) => path.startsWith(UMT5_PREFIX));
            assertEquals(umt5, [], "事前計算の経路が umT5 の容器に触っている");
          }
        },
      });
    }
  },
});

/** 参照 1 本（F32 のテンソルとメタ）。 */
type Fixture = {
  readonly tensor: (key: string) => Float32Array<ArrayBuffer>;
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await Deno.readFile(url);
  const file: SafetensorsFile = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  return {
    tensor: (key) => {
      const view = file.tensors.get(key);
      assert(view !== undefined && view.dtype === "F32", `${url.pathname}: '${key}' が F32 で無い`);
      return new Float32Array(file.buffer, view.byteOffset, view.byteLength / 4);
    },
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: メタ '${key}' が無い`);
      return value;
    },
  };
};

/** 差の要約（比 = 最大絶対差 ÷ 参照の最大絶対値 — 決定 8 の指標）。 */
type Difference = {
  readonly maxAbs: number;
  /** 要素ごとの相対差の最大（参照が 0 の要素は除く — 結果の記録用）。 */
  readonly maxRel: number;
  readonly referenceMax: number;
  readonly ratio: number;
  readonly nonFinite: number;
};

const difference = (got: Float32Array, want: Float32Array): Difference => {
  assertEquals(got.length, want.length, "要素数");
  let maxAbs = 0;
  let maxRel = 0;
  let referenceMax = 0;
  let nonFinite = 0;
  for (let index = 0; index < want.length; index += 1) {
    if (!Number.isFinite(got[index])) {
      nonFinite += 1;
      continue;
    }
    const absolute = Math.abs(got[index] - want[index]);
    maxAbs = Math.max(maxAbs, absolute);
    if (want[index] !== 0) maxRel = Math.max(maxRel, absolute / Math.abs(want[index]));
    referenceMax = Math.max(referenceMax, Math.abs(want[index]));
  }
  return { maxAbs, maxRel, referenceMax, ratio: maxAbs / referenceMax, nonFinite };
};

const formatDifference = (label: string, diff: Difference): string =>
  `${label} 比 ${diff.ratio.toExponential(3)}（maxAbs ${diff.maxAbs.toExponential(3)} / 参照 ${
    diff.referenceMax.toFixed(3)
  }）`;

/** 上流の `clamp(-1, 1)`（参照のフレームはクランプ前で書かれている）。 */
const clamped = (values: Float32Array): Float32Array =>
  values.map((value) => (value > 1 ? 1 : value < -1 ? -1 : value));

/** 出力フレームの uint8 の RGB を全フレーム連結したバイト列（sha256 の実物 — 決定 8）。 */
const rgbBytes = (video: GeneratedVideo): Uint8Array<ArrayBuffer> => {
  const plane = video.width * video.height;
  const out = new Uint8Array(video.frames * plane * 3);
  for (let frame = 0; frame < video.frames; frame += 1) {
    const rgba = wanFrameToRgba(video, frame);
    for (let index = 0; index < plane; index += 1) {
      out.set(rgba.subarray(index * 4, index * 4 + 3), (frame * plane + index) * 3);
    }
  }
  return out;
};

/** 1 回の generate の観測（イベントから — 各 step の後の潜在・段の所要・段の境目の VRAM）。 */
type Observed = {
  readonly video: GeneratedVideo;
  readonly latents: readonly Float32Array[];
  /** 段名 → 壁時計の ms（`start` から `end` まで）。 */
  readonly stageMs: Readonly<Record<string, number>>;
  readonly wallMs: number;
  readonly drm: DrmTimeline;
  readonly diagnostics: ReadonlyMap<WanRunComponent, SessionDiagnostics>;
};

/**
 * generate を 1 回まわし、イベントから観測を集める。`abortAfterDenoise` なら DiT の段の `end`
 * （Session を畳んだ後）で投げて VAE の段を飛ばす（故障注入は潜在だけを見る — VAE の 2 分を払わない）。
 */
const observe = async (
  pipeline: WanPipeline,
  diagnostics: Map<WanRunComponent, SessionDiagnostics>,
  request: Omit<WanGenerateRequest, "onEvent">,
  options: { readonly abortAfterDenoise?: boolean } = {},
): Promise<Observed | { readonly latents: readonly Float32Array[] }> => {
  const latents: Float32Array[] = [];
  const stageStarted = new Map<string, number>();
  const stageMs: Record<string, number> = {};
  const monitor = monitorDrmUsage();
  const abort = new Error("DiT の段の後で止める（故障注入）");
  diagnostics.clear();
  const started = performance.now();
  const onEvent = (event: WanGenerateEvent): void => {
    if (event.kind === "denoise-step") latents.push(event.copyLatents().data);
    if (event.kind !== "stage") return;
    const label = `${event.component} ${event.at}`;
    monitor.mark(label);
    monitor.enter(label);
    if (event.at === "start") stageStarted.set(event.component, performance.now());
    else stageMs[event.component] = performance.now() - (stageStarted.get(event.component) ?? 0);
    if (
      options.abortAfterDenoise === true && event.component === "transformer" && event.at === "end"
    ) {
      throw abort;
    }
  };
  try {
    const video = await pipeline.generate({ ...request, onEvent });
    const wallMs = performance.now() - started;
    return {
      video,
      latents,
      stageMs,
      wallMs,
      drm: monitor.stop(),
      diagnostics: new Map(diagnostics),
    };
  } catch (error) {
    monitor.stop();
    if (error === abort) return { latents };
    throw error;
  }
};

/** 観測の 1 行（段の所要・VRAM の区間ごとの山と境目の値・診断の内訳）。 */
const formatObserved = (observed: Observed): string[] => {
  const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(3)} GiB`;
  const lines = [
    `壁 ${(observed.wallMs / 1000).toFixed(1)} s・段 ${
      Object.entries(observed.stageMs).map(([stage, ms]) => `${stage} ${(ms / 1000).toFixed(1)} s`)
        .join("・")
    }`,
    ...[...observed.drm.peaks].map(([phase, peak]) => `VRAM 山 [${phase}] ${formatDrmUsage(peak)}`),
    ...observed.drm.marks.map(({ label, usage }) => `VRAM 点 [${label}] ${formatDrmUsage(usage)}`),
  ];
  for (const [component, diagnostics] of observed.diagnostics) {
    lines.push(
      `${component}: 重み ${gib(diagnostics.weights.allocatedBytes)}・backing ${
        gib(diagnostics.planBacking.residentBytes + diagnostics.planBacking.inputBytes)
      }・幾何 ${diagnostics.geometryProfile}・submit ${diagnostics.submit.submitCount} 本・窓平均の最大 ${
        diagnostics.submit.chunkBudget.maxWindowMeanMs?.toFixed(1) ?? "—"
      } ms・予算超過 ${diagnostics.submit.chunkBudget.overBudgetChunks} 本`,
    );
  }
  return lines;
};

const textOf = (
  prompts: readonly WanPrompt[],
  name: string,
  form: "prompt" | "normalized",
): string => {
  const entry = prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `埋め込み資産に '${name}' が無い`);
  return entry[form];
};

Deno.test({
  name:
    "Wan 通し（ホスト・GPU 不要）: 配布形の埋め込み資産と σ / timestep が参照のメタと一致し、参照の DiT " +
    "出力から CFG + UniPC が参照の潜在をビット一致で再現する",
  ignore: !ANY_PRESENT,
  fn: async () => {
    const embedsBytes = await readDistributionEmbeds();
    const embeds = parseWanTextEmbeds(embedsBytes.buffer);
    assertEquals(
      embeds.entries.map(({ name, role, tokens }) => [name, role, tokens]),
      [
        ["boxing-cats", "positive", 28],
        ["ferret", "positive", 118],
        ["cat-dog-baking", "positive", 50],
        ["negative", "negative", 126],
      ],
    );
    assertEquals(embeds.width, 4096);
    // ftfy は全角の読点を ASCII の「,」へ正規化する（決定 4 — 正規化後の文字列もメタに持つ）。
    assert(!textOf(embeds.entries, "negative", "normalized").includes("，"));
    assert(textOf(embeds.entries, "negative", "prompt").includes("，"));
    const embedsSha = await sha256Hex(embedsBytes);
    const schedule = wanUniPcSchedule(STEPS, 3, WAN_UNIPC_CONFIG.numTrainTimesteps);
    for (const { name } of CASES) {
      const fixture = await readFixture(fixtureUrl(name));
      assertEquals(
        fixture.meta("text_embeds_sha256"),
        embedsSha,
        `${name}: 参照を作った埋め込み資産`,
      );
      assertEquals(
        [fixture.meta("steps"), fixture.meta("guidance_scale"), fixture.meta("flow_shift")],
        [String(STEPS), "5.0", "3.0"],
      );
      assertEquals(fixture.meta("negative"), "negative");
      const timesteps: unknown = JSON.parse(fixture.meta("timesteps"));
      const sigmas: unknown = JSON.parse(fixture.meta("sigmas"));
      assertEquals(timesteps, schedule.timesteps);
      assertEquals(sigmas, [...schedule.sigmas]);
      // 参照が記録した DiT の出力（forward の hook）から、ホストの CFG + UniPC で参照の潜在を作り直す。
      const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let latents: Float32Array = fixture.tensor("latents_init");
      for (let step = 0; step < STEPS; step += 1) {
        const velocity = wanClassifierFreeGuidance(
          fixture.tensor(`noise_cond.${step}`),
          fixture.tensor(`noise_uncond.${step}`),
          5,
        );
        latents = sampler.step(velocity, latents);
        const want = fixture.tensor(`latents.${step}`);
        const bits = new Uint32Array(latents.buffer, latents.byteOffset, latents.length);
        const wantBits = new Uint32Array(want.buffer, want.byteOffset, want.length);
        assert(
          bits.every((value, index) => value === wantBits[index]),
          `${name}: step ${step} の潜在がビット一致しない（${
            formatDifference("潜在", difference(latents, want))
          }）`,
        );
      }
    }
  },
});

Deno.test({
  name:
    "Wan 通し 2 ステップ（実 GPU）: 832×480・33 フレームの潜在とフレームが参照と帯の内・故障注入は帯の外・" +
    "sha256 の環境行（`f16` 席と i8 の 2 席〈参照行 / 実用行〉）",
  ignore: !ANY_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    const diagnostics = new Map<WanRunComponent, SessionDiagnostics>();
    try {
      const pipeline = await loadPipeline(gpu, diagnostics, "precomputed", F16_QUANT);
      const { prompts } = pipeline;
      try {
        for (const { name, role } of CASES) {
          await t.step(`${name}（${role}）`, async () => {
            const fixture = await readFixture(fixtureUrl(name));
            const promptName = fixture.meta("prompt");
            // band は原文・accept は正規化後の文字列で引く（受理集合の 2 つの綴りを両方通す）。
            const prompt = textOf(prompts, promptName, role === "band" ? "prompt" : "normalized");
            const id = `2step-${name}`;
            let settlement: ReferenceSettlement | undefined;
            await runRecordedCase(results, { id }, async ({ measurements }) => {
              const observed = await observe(pipeline, diagnostics, {
                prompt,
                latents: fixture.tensor("latents_init"),
                steps: STEPS,
              });
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(observed.latents.length, STEPS, "denoise-step の数");
              const diffs = [
                ...observed.latents.map((latents, step) => ({
                  output: `latents.${step}`,
                  band: LATENT_RATIO_BANDS[step],
                  diff: difference(latents, fixture.tensor(`latents.${step}`)),
                })),
                {
                  output: "frames",
                  band: FRAME_RATIO_BAND,
                  diff: difference(observed.video.data, clamped(fixture.tensor("frames"))),
                },
              ];
              for (const { output, band, diff } of diffs) {
                measurements.push({
                  output,
                  maxAbs: diff.maxAbs,
                  maxRel: diff.maxRel,
                  tolerance: { atol: band * diff.referenceMax, rtol: 0 },
                  stage: "karume",
                });
              }
              const notes = [
                ...diffs.map(({ output, diff }) => formatDifference(output, diff)),
                ...formatObserved(observed),
              ];
              console.log(`[wan-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
              for (const { output, band, diff } of diffs) {
                assertEquals(diff.nonFinite, 0, `${id}: ${output} の非有限`);
                assert(
                  diff.ratio <= band,
                  `${id}: ${formatDifference(output, diff)} が帯 ${band} の外`,
                );
              }
              assertEquals(deviceLost, undefined, "device lost");
              const outcome = await settleOrObserve(references, results, {
                id,
                artifact: `${id}.rgb`,
                bytes: rgbBytes(observed.video),
              });
              settlement = outcome.settlement;
              return { ...outcome.fields, note: notes.join(" / ") };
            });
            if (settlement?.check.status === "fail") {
              throw new Error(referenceMismatchMessage(id, settlement, references));
            }
          });
        }

        /**
         * seed 経路の 2 ステップ 1 本（完走・非有限 0・sha256 の環境行）。`f16` 席と i8 の 2 席が同じこの 1 本を通る
         * （席ごとに違うのは pipeline と ID と行のクラスの表示だけ）。
         */
        const seedPath = async (target: WanPipeline, id: string, quant: string): Promise<void> => {
          let settlement: ReferenceSettlement | undefined;
          await runRecordedCase(results, { id }, async () => {
            const observed = await observe(target, diagnostics, {
              prompt: textOf(target.prompts, SEED_CASE.prompt, "prompt"),
              seed: SEED_CASE.seed,
              steps: STEPS,
            });
            assert("video" in observed, "generate が最後まで回っていない");
            assertEquals(observed.latents.length, STEPS, "denoise-step の数");
            const { video } = observed;
            assertEquals([video.frames, video.height, video.width], [DEFAULT_FRAMES, 480, 832]);
            const nonFinite = video.data.reduce(
              (count, value) => count + (Number.isFinite(value) ? 0 : 1),
              0,
            );
            const notes = [
              await rowClassOf(quant),
              `非有限 ${nonFinite}`,
              ...formatObserved(observed),
            ];
            console.log(`[wan-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
            assertEquals(nonFinite, 0, `${id}: 非有限`);
            assertEquals(deviceLost, undefined, "device lost");
            const outcome = await settleOrObserve(references, results, {
              id,
              artifact: `${id}.rgb`,
              bytes: rgbBytes(video),
            });
            settlement = outcome.settlement;
            return { ...outcome.fields, note: notes.join(" / ") };
          });
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(id, settlement, references));
          }
        };

        await t.step(`${SEED_CASE.id}（seed 経路・sha256 の環境行）`, async () => {
          await seedPath(pipeline, SEED_CASE.id, F16_QUANT);
        });

        // 故障注入（受入れの初期ノイズ — 潜在だけを見るので VAE の段の前で止める）。
        const accept = await readFixture(fixtureUrl(ACCEPT_CASE));
        const ferret = textOf(prompts, accept.meta("prompt"), "prompt");
        const negative = textOf(prompts, accept.meta("negative"), "prompt");
        const faults: readonly {
          readonly label: string;
          readonly request: Omit<WanGenerateRequest, "onEvent" | "latents" | "steps">;
        }[] = [
          {
            label: "プロンプトの正負の取り違え",
            request: { prompt: negative, negativePrompt: ferret },
          },
          { label: "guide 5.0 → 5.05", request: { prompt: ferret, guidance: 5.05 } },
        ];
        const lastBand = LATENT_RATIO_BANDS[STEPS - 1];
        for (const { label, request } of faults) {
          await t.step(`故障注入: ${label} → ${ACCEPT_CASE} の潜在が帯の外`, async () => {
            const observed = await observe(pipeline, diagnostics, {
              ...request,
              latents: accept.tensor("latents_init"),
              steps: STEPS,
            }, { abortAfterDenoise: true });
            const diff = difference(
              observed.latents[STEPS - 1],
              accept.tensor(`latents.${STEPS - 1}`),
            );
            console.log(
              `[wan-pipeline] fault ${label}: ${
                formatDifference(`latents.${STEPS - 1}`, diff)
              }（帯の ${(diff.ratio / lastBand).toFixed(1)} 倍）`,
            );
            assert(
              diff.ratio > FAULT_MARGIN * lastBand,
              `${label}: 比 ${diff.ratio} が帯 ${lastBand} の ${FAULT_MARGIN} 倍を超えない`,
            );
          });
        }

        // i8 の 2 席（ADR 0120 段 5 — 参照行 / 実用行）。席ごとに pipeline を同じ device の上に張る（構築では Session を
        // 張らないので、`f16` 席の pipeline と並べても DiT / VAE の確保は重ならない — 段の Session は generate の中だけ）。
        for (const quant of SEAT_QUANTS) {
          const id = seatCaseId(quant, SEED_CASE.id);
          await t.step(`${id}（${quant} の席・seed 経路・sha256 の環境行）`, async () => {
            try {
              await using seatPipeline = await loadPipeline(gpu, diagnostics, "precomputed", quant);
              await seedPath(seatPipeline, id, quant);
            } finally {
              // 次の席の確保の前に、この席の段の確保の解放を待つ（B570 の `destroy()` の遅れ）。
              await settleReleases(gpu);
            }
          });
        }
      } finally {
        await pipeline.dispose();
      }
    } finally {
      // MUST: device を捨てる前に解放を待つ（B570 の destroy の遅れ — 後続のテストの予算を残す）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

Deno.test({
  name:
    "Wan 通し 2 ステップ GPU 経路（実 GPU）: umT5 の text 段を畳んでから DiT・VAE の段を張り、固定プロンプトと" +
    "自由プロンプトが完走・非有限 0・段の所要と切り替えの VRAM・sha256 の環境行",
  ignore: !GPU_TEXT_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    const diagnostics = new Map<WanRunComponent, SessionDiagnostics>();
    try {
      // 構築では Session を張らない（umT5 の重みは generate ごとに text 段で読む — ADR 0119 決定 11）。
      await using pipeline = await loadPipeline(gpu, diagnostics, "gpu", F16_QUANT);
      for (const spec of GPU_TEXT_CASES) {
        await t.step(spec.id, async () => {
          const prompt = "asset" in spec.prompt
            ? textOf(pipeline.prompts, spec.prompt.asset, "prompt")
            : spec.prompt.text;
          let settlement: ReferenceSettlement | undefined;
          try {
            await runRecordedCase(results, { id: spec.id }, async () => {
              const observed = await observe(pipeline, diagnostics, {
                prompt,
                seed: spec.seed,
                steps: STEPS,
              });
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(observed.latents.length, STEPS, "denoise-step の数");
              // text 段は DiT 段を張る前に畳む（段の境目の並び — VRAM の点も同じ並びで採る）。
              assertEquals(
                observed.drm.marks.slice(0, 3).map(({ label }) => label),
                ["text_encoder start", "text_encoder end", "transformer start"],
                "text 段 → DiT 段の順",
              );
              assert(observed.diagnostics.has("text_encoder"), "umT5 の run の診断が届いていない");
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [DEFAULT_FRAMES, 480, 832]);
              const nonFinite = video.data.reduce(
                (count, value) => count + (Number.isFinite(value) ? 0 : 1),
                0,
              );
              const notes = [
                "textEncoder: gpu（umT5 i8・活性 f32）",
                `プロンプト ${JSON.stringify(prompt.trim().slice(0, 80))}`,
                `非有限 ${nonFinite}`,
                ...formatObserved(observed),
              ];
              console.log(`[wan-pipeline] ${spec.id}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, `${spec.id}: 非有限`);
              assertEquals(deviceLost, undefined, "device lost");
              const outcome = await settleOrObserve(references, results, {
                id: spec.id,
                artifact: `${spec.id}.rgb`,
                bytes: rgbBytes(video),
              });
              settlement = outcome.settlement;
              return { ...outcome.fields, note: notes.join(" / ") };
            });
          } finally {
            // 次のケースの確保の前に、このケースの段の確保の解放を待つ（B570 の `destroy()` の遅れ）。
            await settleReleases(gpu);
          }
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(spec.id, settlement, references));
          }
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

/** 一覧図（`rows × cols` のマス・各マスはフレームを `factor` 分の 1 に箱平均で縮めたもの）。 */
const contactSheet = (
  video: GeneratedVideo,
  rows: number,
  cols: number,
  factor: number,
): {
  readonly rgba: Uint8ClampedArray<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
} => {
  const cellWidth = video.width / factor;
  const cellHeight = video.height / factor;
  const width = cellWidth * cols;
  const height = cellHeight * rows;
  const rgba = new Uint8ClampedArray(width * height * 4);
  const cells = rows * cols;
  for (let cell = 0; cell < cells; cell += 1) {
    // 先頭と末尾のフレームを含めて等間隔に引く（33 / 81 枚から 32 マス）。
    const frame = Math.round((cell * (video.frames - 1)) / (cells - 1));
    const source = wanFrameToRgba(video, frame);
    const left = (cell % cols) * cellWidth;
    const top = Math.floor(cell / cols) * cellHeight;
    for (let y = 0; y < cellHeight; y += 1) {
      for (let x = 0; x < cellWidth; x += 1) {
        const at = ((top + y) * width + left + x) * 4;
        for (let channel = 0; channel < 3; channel += 1) {
          let sum = 0;
          for (let dy = 0; dy < factor; dy += 1) {
            for (let dx = 0; dx < factor; dx += 1) {
              sum += source[((y * factor + dy) * video.width + x * factor + dx) * 4 + channel];
            }
          }
          rgba[at + channel] = Math.round(sum / (factor * factor));
        }
        rgba[at + 3] = 255;
      }
    }
  }
  return { rgba, width, height };
};

Deno.test({
  name:
    "Wan 通し 50 ステップ（実 GPU・opt-in KARUME_WAN_FULL_PIPELINE=1）: 既定の設定の 33 / 81 フレーム（`f16` 席と実用席）が" +
    "完走し非有限 0・所要と段の切り替えの VRAM・PNG 全フレーム + 一覧図",
  ignore: !FULL_PIPELINE || !ANY_PRESENT || !GPU_AVAILABLE,
  fn: async (t) => {
    await assertRunningAdapter();
    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    const diagnostics = new Map<WanRunComponent, SessionDiagnostics>();
    try {
      // ケースごとに席の pipeline を張る（構築では Session を張らない — `f16` 席と実用席を確保の大きい順に交互に回す）。
      for (const spec of FULL_CASES) {
        await t.step(spec.id, async () => {
          const frames = spec.frames ?? DEFAULT_FRAMES;
          let settlement: ReferenceSettlement | undefined;
          try {
            await using pipeline = await loadPipeline(gpu, diagnostics, "precomputed", spec.quant);
            await runRecordedCase(fullResults, { id: spec.id }, async () => {
              const observed = await observe(pipeline, diagnostics, {
                prompt: textOf(pipeline.prompts, spec.prompt, "prompt"),
                seed: spec.seed,
                frames: spec.frames,
              });
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(
                observed.latents.length,
                FULL_STEPS,
                "denoise-step の数（manifest の既定）",
              );
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [frames, 480, 832]);
              const nonFinite = video.data.reduce(
                (count, value) => count + (Number.isFinite(value) ? 0 : 1),
                0,
              );
              const notes = [
                await rowClassOf(spec.quant),
                `非有限 ${nonFinite}`,
                ...formatObserved(observed),
              ];
              console.log(`[wan-pipeline] ${spec.id}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, "非有限");
              assertEquals(deviceLost, undefined, "device lost");
              for (let frame = 0; frame < video.frames; frame += 1) {
                await Deno.writeFile(
                  fullResults.artifact(`${spec.id}-frame-${String(frame).padStart(2, "0")}.png`),
                  await encodePng(wanFrameToRgba(video, frame), video.width, video.height),
                );
              }
              const sheet = contactSheet(video, 4, 8, 4);
              await Deno.writeFile(
                fullResults.artifact(`${spec.id}-sheet.png`),
                await encodePng(sheet.rgba, sheet.width, sheet.height),
              );
              console.log(`[wan-pipeline] PNG: ${fullResults.dir.pathname}`);
              const outcome = await settleOrObserve(references, fullResults, {
                id: spec.id,
                artifact: `${spec.id}.rgb`,
                bytes: rgbBytes(video),
              });
              settlement = outcome.settlement;
              return { ...outcome.fields, note: notes.join(" / ") };
            });
          } finally {
            // 次のケースの確保の前に、このケースの段の確保の解放を待つ（B570 の `destroy()` の遅れ）。
            await settleReleases(gpu);
          }
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(spec.id, settlement, references));
          }
        });
      }
    } finally {
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

/**
 * 登録するケース（回せるものだけ — 事前計算の経路は配布形と少ステップの参照、GPU 経路は配布形と umT5 のミラーが要る。
 * GPU 経路は CPU の参照を持たないので少ステップの参照に依らない）。
 */
const CASE_IDS = [
  ...(ANY_PRESENT
    ? [
      ...CASES.map(({ name }) => `2step-${name}`),
      SEED_CASE.id,
      ...SEAT_QUANTS.map((quant) => seatCaseId(quant, SEED_CASE.id)),
      ...(FULL_PIPELINE ? FULL_CASES.map(({ id }) => id) : []),
    ]
    : []),
  ...(GPU_TEXT_PRESENT ? GPU_TEXT_CASES.map(({ id }) => id) : []),
];
const RUNNABLE = (ANY_PRESENT || GPU_TEXT_PRESENT) && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });

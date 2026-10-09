/**
 * Wan2.2 TI2V-5B の I2V の通し（`WanTi2vPipeline` に条件画像を渡す — ADR 0121 決定 11・段 9b）の照合（実 GPU）。
 *
 * 配布形ミラー `models/karume-wan2.2/`（encoder の 3 部品を含む — `dist.py --pipeline wan-ti2v` が組む）を `denoDirectory` で
 * 読んで組み、条件画像からの生成を通す。参照（`pipeline_steps_i2v.*`）は配布しない golden なので系列から読む。配布形が無い機は
 * 明示 SKIP する（理由と組み立てのコマンドを出す — 全 SKIP を FAIL にするのは門番 `distribution_gate_test.ts`）。
 *
 * 比べる相手は recipe の 2 ステップの参照（`tools/export-recipes/wan/ti2v_i2v_few_step_ref.py` — 置き場は i8 の DiT の系列の根の
 * `pipeline_steps_i2v.<case>.safetensors`）: diffusers の `WanImageToVideoPipeline` の `expand_timesteps` の分岐を CPU f32 で
 * 2 ステップ（CFG あり・guidance 5.0・shift 5.0・時刻の MLP は値ごとに M = 1 — 決定 4）回した潜在と、それを段 5 のタイル
 * decode → unpatchify → クランプに通したフレーム（重みは参照席 `f16+dit8` と同じ値）。形は 1280×704・9 フレーム（潜在
 * `[48,3,44,80]`・S = 2,640・条件のトークン 880）で、3 本とも同じ形。初期ノイズは参照の `latents_init` を `latents` で注入し、
 * 条件画像は参照に埋め込まれた元画像の RGB8（`source` 832×480 — PNG の decode にも別の資産にも頼らない）を `image` で渡す
 * （寸法の選択と crop は製品の前処理が行う）。
 *
 * ## 門（既定のレーン）
 *
 * - **ホストの自己整合**（GPU 不要 — 3 本とも）: 参照のメタ（σ / timestep・shift・時刻の経路・条件のトークン数・画像の来歴・
 *   埋め込み資産）と統計 48 値が TS と一致し、埋め込みの `source` を TS の crop（`fitWanI2vImage`）に通すと参照の `rgb8` と
 *   ビット一致する。参照が記録した DiT の出力から、ホストの CFG + UniPC（置き換え無し — diffusers の callback と同じ）が参照の
 *   `latents.<i>` をビット一致で作り直し、製品の置き換えの関数（`withWanFirstFrameCondition`）が `latents.1` と条件の潜在から
 *   参照の `latents_final` をビット一致で作り直す。
 * - **帯**（比 = 最大絶対差 ÷ 参照の最大絶対値 — T2V と同じ指標）: 観測点は各 step の後のスケジューラの潜在（`latents.0` /
 *   `latents.1` — `denoise-step` の `copyLatents`・置き換える前）・最後の潜在（`latents_final` — 置き換えた後・VAE へ渡す値。
 *   下の「最後の潜在の観測」）・クランプ後のフレーム。帯は決定用 2 本（`band-boxing-cats` / `band-cat-dog-baking`）の観測点ごとの
 *   最悪 × 5（有効数字 2 桁へ切り上げ）で、受入れ（`accept-ferret`）が帯の内であることを見る。帯が未導出（`undefined`）の間は、
 *   各ケースは比を記録して赤で止まり、「帯の候補」の step が決定用の最悪 × 5 を出す（決定用の 2 本目は opt-in なので、候補は
 *   opt-in を付けた走行でだけ出る）。MUST: 受入れの結果を見て帯もケースも変えない。受入れが帯を外れたら、帯を広げずに原因を
 *   調べる。
 * - **DiT の入力の結線**（帯のケースごと — 実走の入力そのもの）: DiT の run は CFG の 2 パス × 2 step。各 run の条件マスクは
 *   先頭 880 トークンだけが真、条件側の時刻は t = 0 の proj（`timestepsProj(0)`）とビット一致、モデル入力の先頭の潜在フレームは
 *   全 run で同じ値（= 条件の潜在）で、残りのフレームはその step のスケジューラの状態（step 0 は注入した初期ノイズ・step 1 は
 *   `latents.0`）と一致する。段の並びは encoder → DiT（事前計算の経路は text の段を持たない）で、encoder の 3 グラフの診断が届く。
 * - **条件の潜在**（帯のケースごと）: DiT が受けた先頭フレーム（encoder の段の出口）と参照の `condition_latents` の最大絶対差が
 *   encoder の GPU の門と同じ帯（`WAN_I2V_ENCODER_BAND` — 帯の決定と受入れの独立は encoder の門の側で保たれている）の内。
 *   encoder の退行を潜在の帯に任せず、encoder の段を名指しで落とす。
 * - **VAE の入力の結線**（帯のケースごと）: 下の「最後の潜在の観測」。
 * - **故障注入**（決定用の既定のケース `band-boxing-cats` の初期ノイズ・DiT の段で打ち切り `latents.1` で比べる・床は帯の
 *   {@link FAULT_MARGIN} 倍）: 条件マスクを全て偽・条件マスクを 1 トークンずらす・時刻の入力 2 本を取り違える・モデル入力の
 *   先頭フレームを置き換えない・条件の潜在を正規化しない。下の「観測と故障注入の口」の形で、製品のコードに口を足さずに入れる。
 *   各注入は帯のほかに、名指しの検査（DiT の入力の結線か条件の潜在の門）が同じ走行の記録で赤になることも見る。VAE の入力の
 *   結線の検査が赤になることは、帯のケースの記録で期待を故障の値に替えて見る（GPU の時間 0）。
 * - **sha256 の環境行**（ADR 0106 — `fixtures/references/wan-ti2v.json` を T2V と共有・ケース ID が別）: 出力フレーム（uint8 の
 *   RGB を全フレーム連結したバイト列 — `wanFrameToRgba` の規則）。既定のレーンは 2 本 = 参照席の既定のケース
 *   （{@link DEFAULT_CASE}）と、実用席の同じ要求（{@link PRACTICAL_CASE_ID}）。実用席の行は**実用行**（ADR 0110 決定 7 — 実用層の
 *   数値を意図して変えるコミットで同じコミットの `rewrite`）で、突合と書き込みの前に、同じ要求の参照席の実物と 1 bit 以上違う
 *   こと（床 — ADR 0110 決定 5）を見る。参照席の行は**参照行**（凍結）。ID は席・I2V と fit・step 数・ケース・寸法・フレーム数・
 *   shift を全て持つ（{@link caseIdOf} — 裁定 F12）。行が無い機はそのケースの sha の突合を飛ばし（実物と実測の sha は残す）、
 *   参照門が赤になる。行は帯が緑になった後に `KARUME_REFERENCE=write` をこのファイル単独で回して作る。
 * - **席は全ケースで明示する**: manifest の `defaultQuant` は品質の裁定で動きうる値で、既定席に乗ると同じ ID が別の席の値で回る。
 *
 * ## opt-in（env `KARUME_WAN_TI2V_I2V_FULL=1` の 1 つ — 既定のレーンに入れない）
 *
 * - 2 ステップの残りの参照ケース 2 本（`band-cat-dog-baking` — 帯の導出に要る決定用の 2 本目・`accept-ferret` — 受入れ）。
 *   どちらも参照席の行を持つ。
 * - 縦長 704×1280（参照席・`boxing-cats` の横長の元画像を明示の縦長へ crop・seed 42・CPU の参照は無い — 完走・非有限 0・
 *   sha256 の環境行）。
 * - 50 ステップの通し（別の `Deno.test` — 下の節）。
 *
 * opt-in の行は opt-in のときだけ参照門に登録する（既定のレーンでは回らないので、登録すると行が無い機の参照門を赤にする — T2V の
 * 121 フレーム・50 ステップと同じ扱い）。2 ステップの opt-in を回すコマンド（50 ステップを同じプロセスに載せない）:
 *
 * ```
 * KARUME_WAN_TI2V_I2V_FULL=1 deno test -A --v8-flags=--expose-gc packages/models/tests/e2e_wan_ti2v_i2v_pipeline_test.ts --filter "I2V 通し 2 ステップ"
 * ```
 *
 * ## 50 ステップの通し（同じ opt-in・`--filter "50 ステップ"` で単独のプロセス）
 *
 * 事前計算の経路・`boxing-cats` の画像とプロンプト・seed 42・shift 5.0・1280×704（画像の縦横比で選ぶ）・33 フレーム・manifest の
 * 既定の 50 ステップを、参照席と実用席で 1 本ずつ回し、全フレームの PNG・一覧図・RGB の実物・段ごとの所要・段の境目の VRAM を
 * `outputs/verify/<環境キー>/<日付>_wan-ti2v-i2v-pipeline-full/` に書き、非有限 0 と device lost が無いことを見る（利用者の視認の
 * 素材）。先頭フレームと条件画像の差は記録だけ（門ではない）。sha256 の環境行は持たない（段 9b で決めた sha 行の列挙に
 * 入っていない — 実物の RGB とその sha256 は結果の席に残す）。
 *
 * ## 観測と故障注入の口（製品のコードに口を足さない）
 *
 * DiT の入力（モデル入力・条件マスク・条件側の時刻）と VAE の段の入力は公開面の外なので、DiT の入力を見る回（参照席の帯の
 * ケースと故障注入）は**1 層内側**で回す: 構築は `loadWanFromPretrained`、生成は `generateWanVideo`（公開 class の
 * `fromPretrained` / `generate` が直列化鎖に載せる本体そのもの — 取得面・家族 admission・段の順序は同じ 1 本を通る）。状態の
 * `transformer` / `vae_decoder_first` / `vae_decoder_next` の部品だけを、Session を Proxy で包む写し（{@link withSessionHook}）に差し替え、`run` /
 * `enqueue` の入力を写す（観測の回は入力のオブジェクトをそのまま渡す — 値は変えない）。故障注入は同じ口で DiT の入力を
 * 書き換えるか（マスク・時刻の入力・モデル入力）、`generateWanVideo` へ渡す家族の値の統計を mean 0・std 1 にする（正規化の
 * 式 `(x − mean)·(1/std)` が mu をそのまま返す — encoder の段だけが読む値で、VAE の段の前で打ち切る）。実用席・縦長・50 ステップ
 * は公開面（`WanTi2vPipeline.fromPretrained` + `generate`）で回す（条件画像を受ける公開の入口の実 GPU の通し）。
 *
 * ### 最後の潜在の観測
 *
 * `copyLatents` はスケジューラの状態（置き換える前）なので、最後の潜在は次の 2 つから組む: DiT の step 0 の入力から読み出した
 * 条件の潜在（モデル入力の先頭フレーム — patchify を製品の `patchifyLatents` の置換で戻す）と、観測した `latents.1`。組んだ値を
 * 製品の置き換えの関数に通したものが VAE へ渡った値であることは、VAE の段の入力（逆正規化済みのタイル — 先頭 chunk 28 枚・
 * 後続 chunk 56 枚）が、置き換えの関数を通さずに組んだ期待 — 先頭フレームは条件の潜在・フレーム 1 / 2 は `latents.1` のそのフレーム
 * — の逆正規化を参照のタイルの窓（メタの `rows_starts` × `cols_starts`）で切ったものと値で一致することで縛る（ループの後の置き換えが
 * 落ちると先頭フレームの窓が、最後のモデル入力の写しを渡すとフレーム 1 / 2 の窓が 1 枚も合わない）。窓は潜在の全体を覆うので、
 * `latents_final` の帯は VAE が実際に受けた値の帯でもある。
 *
 * ## device の使い方
 *
 * 2 ステップの `Deno.test` 1 本は device を 1 つだけ取り、全ケースを `t.step` で回す（ケースの間に `settleReleases`）。構築は配布形の
 * part を読むだけで Session を張らない（段ごとに張って畳む）ので、1 層内側の状態と公開の pipeline を同じ device で順に使う。
 * 50 ステップは別の `Deno.test`（device 1 つ）で、`--filter "50 ステップ"` で**単独のプロセス**として回す。
 *
 * ## 観測（門ではない）
 *
 * 段の所要（壁時計 — `stage` イベントの間）・段の境目の VRAM（`helpers/drm-usage.ts`）・条件の潜在の差の encoder の帯に対する倍率・先頭フレームと
 * 条件画像の差（`[-1, 1]`・VAE の往復）・`latents.<i>` のフレームごとの最大絶対差と、step 0 の DiT の出力（uncond / cond）と参照の
 * `noise_uncond.0` / `noise_cond.0` のフレームごとの最大絶対差（step 0 のモデル入力は参照と条件の潜在の差しか違わないので、
 * 差が条件のトークンに集まるか生成側に出るかを切り分ける手掛かり — {@link LATENT_RATIO_BANDS} の導出の注記）。
 *
 * 参照（`pipeline_steps_i2v.*`）が無い環境では GPU の全ケース（条件画像も参照から読む）とホストの自己整合を明示 SKIP する。
 * 参照が一部だけある環境は FAIL（T2V の通しの e2e と同じ規律）。
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { denoDirectory } from "@karume/hub/deno";
import {
  parseSafetensors,
  type RunInputs,
  type RunOutputs,
  type SafetensorsFile,
  type Session,
  type SessionDiagnostics,
  type Tensor,
} from "@karume/runtime";
import { encodePng } from "../mod.ts";
import {
  type GeneratedVideo,
  type Rgb8Image,
  wanFrameToRgba,
  type WanGenerateEvent,
  type WanPrompt,
  type WanRunComponent,
  type WanTi2vGenerateRequest,
  WanTi2vPipeline,
} from "../wan.ts";
import type { ModelComponent } from "../src/hub/components.ts";
import {
  generateWanVideo,
  loadWanFromPretrained,
  WAN22_TI2V_FAMILY,
  type WanFamilySpec,
  type WanState,
} from "../src/wan/family.ts";
import { wanConditionMask, wanDitPatch, withWanFirstFrameCondition } from "../src/wan/dit-loop.ts";
import {
  patchifyLatents,
  unpatchifyTokens,
  type WanPatchGeometry,
  wanTokenGrid,
} from "../src/wan/dit-tokens.ts";
import { timestepsProj } from "../src/wan/dit-timestep.ts";
import { fitWanI2vImage, preprocessWanI2vImage } from "../src/wan/i2v-preprocess.ts";
import {
  denormalizeWanLatents,
  WAN22_LATENTS_MEAN,
  WAN22_LATENTS_STD,
} from "../src/wan/latents.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../src/wan/scheduler.ts";
import { WAN_VAE_LATENT_INPUT } from "../src/wan/vae-chunks.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { type DrmTimeline, formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { filePresent, WAN_TI2V_SERIES_NAME } from "./helpers/wan-ti2v-dit.ts";
import { WAN_I2V_ENCODER_BAND, WAN_I2V_IMAGE_SHA256 } from "./helpers/wan-i2v-image.ts";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_ASSEMBLE_COMMAND,
  WAN_TI2V_DIST_ROOT,
  WAN_TI2V_PRACTICAL_QUANT,
  WAN_TI2V_REFERENCE_QUANT,
} from "./helpers/wan-ti2v-pipeline.ts";
import { contactSheet, writeWanFrames } from "./helpers/wan-video-artifacts.ts";
import { assertRunningAdapter } from "../../runtime/tests/helpers/environment.ts";
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
 * 2 ステップの通しのスケジューラの潜在（各 step の後 — `[48,3,44,80]`）の帯（比 = 最大絶対差 ÷ 参照の最大絶対値）。添字が
 * step。`undefined` = 未導出（モジュール doc「門」— 各ケースは比を記録して赤で止まり、「帯の候補」の step が候補を出す）。
 *
 * 導出の規則: 観測点ごとに決定用 2 本の最悪 × 5（有効数字 2 桁へ切り上げ）。最初の実走が出した候補を、実測の表（決定用 2 本・
 * 帯・受入れ）と一緒にここへ書く（T2V の `e2e_wan_ti2v_pipeline_test.ts` の同名の定数の doc と同じ形）。
 *
 * 導出（2026-10-09・RTX 3080 Ti・Deno・参照席・opt-in で 3 本 — 同じ日に 2 回走らせ、全ての比が同値）:
 *
 * | 観測点        | band-boxing-cats | band-cat-dog-baking | 最悪 × 5 | 帯     | 受入れ accept-ferret |
 * | ------------- | ---------------: | ------------------: | -------: | -----: | -------------------: |
 * | latents.0     |         9.564e-5 |            5.786e-4 | 2.893e-3 | 2.9e-3 |             5.517e-6 |
 * | latents.1     |         7.108e-5 |            6.985e-4 | 3.492e-3 | 3.5e-3 |             1.160e-4 |
 * | latents_final |         1.117e-4 |            7.716e-4 | 3.858e-3 | 3.9e-3 |             4.390e-5 |
 * | frames        |         1.412e-3 |            1.766e-3 | 8.832e-3 | 8.9e-3 |             7.367e-4 |
 *
 * 潜在の帯は T2V（latents.1 9.5e-4）より約 4 倍広い。決めているのは cat-dog-baking で、差は条件のトークン（先頭フレーム・t = 0・
 * きれいな画像の潜在）の DiT の出力に集まる（step 0 の cond パスのフレームごとの maxAbs 3.21e-3 / 2.81e-4 / 2.42e-4）。出所は
 * DiT の forward の精度ではなく、encoder の出口の差（条件の潜在の maxAbs 1.26e-5 — encoder の帯 7.8e-5 の内）を DiT が条件の
 * トークンで約 250 倍に増幅すること: モデル入力の先頭フレームを参照の `condition_latents` に差し替えて同じ step 0 を回すと
 * （使い捨ての診断・2026-10-09）、cond パスのフレーム 0 の差は 2.41e-4、`latents.1` の差は 4.12e-4 に下がり、T2V と同じ桁に
 * 戻る（boxing-cats は差し替えても `latents.1` 4.72e-4 で変わらない — 増幅の大きさは画像で違う）。段 2 の DiT の門の golden は
 * 潜在を乱数で作るので、この感度は通しで初めて現れる。
 *
 * 故障注入の比（latents.1・床は帯の 2 倍 = 7.0e-3）: 条件マスクを全て偽 1.220・1 トークンずらす 4.844e-1・時刻の入力 2 本を
 * 取り違える 1.252・モデル入力の先頭フレームを置き換えない 1.043・条件の潜在を正規化しない 1.098（最小でも帯の 138 倍 — 時刻の
 * 取り違えは同じ日の 2 回目の走行で測った）。初回の「条件側の時刻を生成側と同じにする」は 1.220 で、条件マスクを全て偽と同じ値
 * だった（どちらも全トークンが生成側の時刻を受ける同じ入力）ので、時刻の取り違えに替えた。
 *
 * 既定のレーンの検出力（記録 — 規則は変えない）: 帯を決めているのは opt-in の cat-dog-baking で、既定のレーンで回る boxing-cats
 * から見た帯の余裕は latents.0 30 倍・latents.1 49 倍・latents_final 35 倍・frames 6.3 倍（故障注入の床は boxing-cats の
 * latents.1 の約 98 倍）。boxing-cats の潜在の誤差を数十倍にする DiT や UniPC の退行は、既定のレーンでは帯を出ず、opt-in で
 * 見える。encoder の退行は条件の潜在の門（encoder の帯 — 条件の潜在の maxAbs は boxing-cats 1.150e-5・cat-dog-baking
 * 1.264e-5・ferret 5.394e-6 で、encoder の GPU の門の 1280×704 の値と同じ）が名指しで落とす。
 *
 * MUST: 受入れ（`accept-ferret`）の結果を見てこの値も、決定用のケースも変えない（帯の決定と受入れの独立が崩れる）。
 * 受入れが帯を外れたら帯を広げずに原因を調べる。
 */
const LATENT_RATIO_BANDS: readonly [number | undefined, number | undefined] = [
  2.9e-3,
  3.5e-3,
];

/**
 * 最後の潜在（置き換えた後 — VAE へ渡す値・`[48,3,44,80]`）の帯。`undefined` = 未導出（{@link LATENT_RATIO_BANDS} と同じ規則）。
 *
 * MUST: 受入れの結果を見てこの値を変えない（{@link LATENT_RATIO_BANDS} と同じ）。
 */
const FINAL_RATIO_BAND: number | undefined = 3.9e-3;

/**
 * 出力フレーム（クランプ後の `[3,9,704,1280]` — 参照もクランプ後で書かれている）の帯。`undefined` = 未導出
 * （{@link LATENT_RATIO_BANDS} と同じ規則）。
 *
 * MUST: 受入れの結果を見てこの値を変えない（{@link LATENT_RATIO_BANDS} と同じ）。
 */
const FRAME_RATIO_BAND: number | undefined = 8.9e-3;

/** 故障が帯から離れているべき倍率の床（T2V・2.1・段 2 と同じ 2 — 赤にならない注入は帯が広すぎる兆候）。 */
const FAULT_MARGIN = 2;

/** 2 ステップの通しの条件（参照の `ti2v_i2v_few_step_ref.py` の `STEPS` / `GUIDANCE_SCALE` / `FLOW_SHIFT` / `LATENT_SHAPE`）。 */
const STEPS = 2;
const GUIDANCE = 5;
const SHIFT = 5;
const WIDTH = 1280;
const HEIGHT = 704;
const FRAMES = 9;
const LATENT_SHAPE: readonly [number, number, number, number] = [48, 3, 44, 80];
/** 条件の潜在の形（先頭の潜在フレーム 1 枚）。 */
const CONDITION_SHAPE: readonly number[] = [48, 1, 44, 80];
/** 条件のトークン数（先頭の潜在フレームのトークン `(44/2)·(80/2)` — 参照のメタ `condition_tokens`）。 */
const CONDITION_TOKENS = 880;
/** 参照に埋め込まれた元画像の形（`inputs/wan-i2v/<名前>-832x480.png` の画素 — HWC）。 */
const SOURCE_SHAPE: readonly number[] = [480, 832, 3];
/** 参照の寸法の合わせ方（公式 — 要求では省いて製品の既定に乗せ、ID には綴る）。 */
const FIT = "crop";

/** 参照（配布しない golden）の置き場 — i8 の DiT の系列の根（T2V の参照と同じ根・接頭辞が別）。 */
const STEPS_ROOT = new URL(`../../../outputs/series/${WAN_TI2V_SERIES_NAME}/`, import.meta.url);

/** opt-in（モジュール doc「opt-in」— 2 ステップの残りの参照ケース・縦長・50 ステップ）。 */
const I2V_FULL = Deno.env.get("KARUME_WAN_TI2V_I2V_FULL") === "1";

/**
 * 参照のケース（正本は recipe `wan/ti2v_i2v_few_step_ref.py` の `FIXTURE_CASES` — プロンプトと画像は同じ名前）。決定用
 * （`band`）2 本 + 受入れ（`accept`）1 本。既定のレーンは決定用の 1 本目だけ（裁定 8 = b）で、残りは opt-in。
 * MUST: 受入れの結果を見て決定用のケースを変えない。
 */
const CASES: readonly {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly optIn: boolean;
}[] = [
  { name: "band-boxing-cats", role: "band", optIn: false },
  { name: "band-cat-dog-baking", role: "band", optIn: true },
  { name: "accept-ferret", role: "accept", optIn: true },
];
/** 既定のレーンのケース（故障注入と実用席の要求もこのケース）。 */
const DEFAULT_CASE = "band-boxing-cats";

/**
 * sha 行のケース ID（席・I2V と fit・step 数・ケース・寸法・フレーム数・shift を全て持つ — 裁定 F12）。`head` は
 * {@link i2vHead} が綴る。
 *
 * MUST: 要求（画像・fit・プロンプト・seed / 初期ノイズ・step 数・寸法・フレーム数・shift・席）を変えたら ID も変える（行の値が
 * 別の条件の sha と突き合わさる）。
 */
const caseIdOf = (head: string, width: number, height: number, frames: number): string =>
  `${head}-${width}x${height}-${frames}f-shift${SHIFT}`;

/** ID の頭（`<席>-i2v-<fit>-<step 数>-<ケース>` — T2V の ID と `-i2v-` で割れる）。 */
const i2vHead = (quant: string, steps: string, name: string): string =>
  `${quant}-i2v-${FIT}-${steps}-${name}`;

/** 帯のケースの ID（参照席・事前計算の経路）。 */
const bandCaseId = (name: string): string =>
  caseIdOf(i2vHead(WAN_TI2V_REFERENCE_QUANT, "2step", name), WIDTH, HEIGHT, FRAMES);

/**
 * 実用席のケース（{@link DEFAULT_CASE} と同じ要求で、席だけ実用席 — 既定のレーンの実用行）。帯は持たない（実用席の数値の門は
 * 自機 A/B 門 — I2V の A/B は段 9d）。公開面で回す。
 */
const PRACTICAL_CASE_ID = caseIdOf(
  i2vHead(WAN_TI2V_PRACTICAL_QUANT, "2step", DEFAULT_CASE),
  WIDTH,
  HEIGHT,
  FRAMES,
);

/** seed 経路の画像とプロンプト（同じ名前 — {@link DEFAULT_CASE} の参照に埋め込まれた元画像）と seed。 */
const SEED_IMAGE = "boxing-cats";
const SEED = 42;

/**
 * 縦長のケース（opt-in・参照席・公開面）。横長の元画像を明示の 704×1280 へ crop する（段 9a の encoder の golden の縦長と同じ
 * 扱い）。CPU の参照は無いので、完走・非有限 0 と sha256 の環境行だけで縛る。
 */
const PORTRAIT_CASE_ID = caseIdOf(
  i2vHead(WAN_TI2V_REFERENCE_QUANT, "2step", `${SEED_IMAGE}-seed${SEED}`),
  HEIGHT,
  WIDTH,
  FRAMES,
);

/** 50 ステップの通しのフレーム数（既定のフレーム数）。 */
const FULL_FRAMES = 33;
/**
 * 50 ステップの通しの step 数。要求では指定せず manifest の既定に任せるので、観測した denoise-step の数で縛る（manifest の
 * 既定が変わったら ID の `50step` と食い違う）。
 */
const FULL_STEPS = 50;
/** 50 ステップの通しの席（並びは確保の大きい参照席が先 — T2V の 50 ステップと同じ理由）。 */
const FULL_CASES: readonly { readonly id: string; readonly quant: string }[] = [
  WAN_TI2V_REFERENCE_QUANT,
  WAN_TI2V_PRACTICAL_QUANT,
].map((quant) => ({
  id: caseIdOf(i2vHead(quant, "50step", `${SEED_IMAGE}-seed${SEED}`), WIDTH, HEIGHT, FULL_FRAMES),
  quant,
}));

const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.ti2v_i2v_few_step_ref";

/**
 * 参照を作った埋め込み資産（2.1 の系列のファイル — ADR 0121 決定 9）。ホストの自己整合はこのバイトの sha を参照のメタ
 * `text_embeds_sha256` と突き合わせる（T2V の通しの e2e と同じ扱い — 配布形の写しは同じ系列のファイル）。
 */
const TEXT_EMBEDS_URL = new URL(
  "../../../outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors",
  import.meta.url,
);

const fixtureUrl = (name: string): URL =>
  new URL(`pipeline_steps_i2v.${name}.safetensors`, STEPS_ROOT);

/** 配布形の manifest（無ければ通しの照合を SKIP する — 中身はこの e2e では読まない）。 */
const DIST_PRESENT = (await readWanTi2vDistributionManifestText()) !== undefined;
if (!DIST_PRESENT) {
  console.warn(
    `[karume] 配布形ミラー ${WAN_TI2V_DIST_ROOT.pathname} が無いため Wan2.2 の I2V の通しの照合を SKIP する。` +
      `組み立て: ${WAN_TI2V_ASSEMBLE_COMMAND}（全 SKIP は門番 distribution_gate_test.ts が FAIL にする）`,
  );
}
const FIXTURES_PRESENT = CASES.map(({ name }) => filePresent(fixtureUrl(name)));
const ANY_FIXTURE = FIXTURES_PRESENT.some(Boolean);
if (!ANY_FIXTURE) {
  console.warn(
    `[karume] ${STEPS_ROOT.pathname} に I2V の 2 ステップの参照（pipeline_steps_i2v.*）が無いため Wan2.2 の I2V の通し` +
      `（条件画像も参照から読む）とホストの自己整合を SKIP する。生成（CPU で約 42 分・RAM の山 約 24 GiB）: ${GENERATE_COMMAND}`,
  );
}

const references = openReferences(new URL("fixtures/references/wan-ti2v.json", import.meta.url));
const results = openResults("wan-ti2v-i2v-pipeline");
/** 50 ステップの通しの結果の席（`results.json` は走行ごとに丸ごと書き直されるので、別のプロセスで回す回と分ける）。 */
const fullResults = openResults("wan-ti2v-i2v-pipeline-full");

Deno.test({
  name: "Wan2.2 I2V 通しの参照: 2 ステップの参照 3 本が揃っている",
  ignore: !ANY_FIXTURE,
  fn: () => {
    assertEquals(
      FIXTURES_PRESENT,
      FIXTURES_PRESENT.map(() => true),
      `${STEPS_ROOT.pathname} の欠け`,
    );
  },
});

/** 参照 1 本（F32 / U8 のテンソル・形・メタ）。 */
type Fixture = {
  readonly tensor: (key: string) => Float32Array<ArrayBuffer>;
  readonly bytes: (key: string) => Uint8Array<ArrayBuffer>;
  readonly shape: (key: string) => readonly number[];
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await Deno.readFile(url);
  const file: SafetensorsFile = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string, dtype: "F32" | "U8") => {
    const found = file.tensors.get(key);
    assert(
      found !== undefined && found.dtype === dtype,
      `${url.pathname}: '${key}' が ${dtype} で無い`,
    );
    return found;
  };
  return {
    tensor: (key) => {
      const found = view(key, "F32");
      return new Float32Array(file.buffer, found.byteOffset, found.byteLength / 4);
    },
    bytes: (key) => {
      const found = view(key, "U8");
      return new Uint8Array(file.buffer, found.byteOffset, found.byteLength);
    },
    shape: (key) => {
      const found = file.tensors.get(key);
      assert(found !== undefined, `${url.pathname}: '${key}' が無い`);
      return found.shape;
    },
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: メタ '${key}' が無い`);
      return value;
    },
  };
};

/** 参照に埋め込まれた元画像（RGB8・HWC — 製品の `image` に渡す形）。 */
const sourceImageOf = (fixture: Fixture): Rgb8Image => {
  assertEquals(fixture.shape("source"), SOURCE_SHAPE, "元画像の形");
  return { width: SOURCE_SHAPE[1], height: SOURCE_SHAPE[0], data: fixture.bytes("source") };
};

/** 差の要約（比 = 最大絶対差 ÷ 参照の最大絶対値）。 */
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

/** f32 の列どうしのビット一致（NaN の符号や -0 も区別する）。 */
const bitsEqual = (got: Float32Array, want: Float32Array): boolean => {
  if (got.length !== want.length) return false;
  const gotBits = new Uint32Array(got.buffer, got.byteOffset, got.length);
  const wantBits = new Uint32Array(want.buffer, want.byteOffset, want.length);
  return gotBits.every((value, index) => value === wantBits[index]);
};

/**
 * 値の一致（`===` — -0 と +0 を同じに見る）で最初に食い違う添字（一致なら -1・長さ違いは 0）。結線の検査は「同じ値が渡った」
 * ことを見るので、置き換えの式 `(1 − m)·cond + m·latents` が符号付きゼロだけを動かす場合を割れとしない。
 */
const firstValueMismatch = (
  got: Float32Array | Uint32Array,
  want: Float32Array | Uint32Array,
): number => {
  if (got.length !== want.length) return 0;
  return got.findIndex((value, index) => value !== want[index]);
};

/** 出力フレームの uint8 の RGB を全フレーム連結したバイト列（sha256 の実物）。 */
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

const countNonFinite = (values: Float32Array): number =>
  values.reduce((count, value) => count + (Number.isFinite(value) ? 0 : 1), 0);

/**
 * 出力の先頭フレームと条件画像（crop の後の RGB8 を `[-1, 1]` へ）の差（観測 — VAE の往復の誤差を含む・門ではない）。
 * `video.data` は `[3, F, H, W]`、`rgb8` は `[H, W, 3]`。
 */
const firstFrameDifference = (
  video: GeneratedVideo,
  rgb8: Uint8Array,
): { readonly maxAbs: number; readonly meanAbs: number } => {
  const plane = video.width * video.height;
  assertEquals(rgb8.length, plane * 3, "条件画像の要素数");
  let maxAbs = 0;
  let sum = 0;
  for (let channel = 0; channel < 3; channel += 1) {
    const base = channel * video.frames * plane;
    for (let index = 0; index < plane; index += 1) {
      const want = Math.fround(2 * Math.fround(rgb8[index * 3 + channel] / 255) - 1);
      const absolute = Math.abs(video.data[base + index] - want);
      maxAbs = Math.max(maxAbs, absolute);
      sum += absolute;
    }
  }
  return { maxAbs, meanAbs: sum / (plane * 3) };
};

const formatFirstFrame = (diff: { readonly maxAbs: number; readonly meanAbs: number }): string =>
  `先頭フレームと条件画像の差（[-1,1]・観測）maxAbs ${diff.maxAbs.toFixed(3)}・meanAbs ${
    diff.meanAbs.toFixed(4)
  }`;

/** 潜在 `[C, F, H, W]`（`LATENT_SHAPE`）のフレーム 1 枚を `[C, 1, H, W]` に切り出す。 */
const latentFrame = (latents: Float32Array, frame: number): Float32Array<ArrayBuffer> => {
  const [channels, frames, height, width] = LATENT_SHAPE;
  const plane = height * width;
  assertEquals(latents.length, channels * frames * plane, "潜在の要素数");
  const out = new Float32Array(channels * plane);
  for (let channel = 0; channel < channels; channel += 1) {
    const from = (channel * frames + frame) * plane;
    out.set(latents.subarray(from, from + plane), channel * plane);
  }
  return out;
};

/**
 * 潜在のフレームごとの最大絶対差（観測 — 先頭フレームはスケジューラの状態では置き換えられない値で、DiT の条件のトークンの出力から
 * 進む。差がどのフレームにあるかを記録して、帯を読むときの手掛かりにする）。
 */
const frameMaxAbs = (got: Float32Array, want: Float32Array): string =>
  Array.from(
    { length: LATENT_SHAPE[1] },
    (_, frame) =>
      difference(latentFrame(got, frame), latentFrame(want, frame)).maxAbs.toExponential(2),
  ).join(", ");

const textOf = (
  prompts: readonly WanPrompt[],
  name: string,
  form: "prompt" | "normalized",
): string => {
  const entry = prompts.find((candidate) => candidate.name === name);
  assert(entry !== undefined, `埋め込み資産に '${name}' が無い`);
  return entry[form];
};

// ---- 観測と故障注入の口（モジュール doc「観測と故障注入の口」）------------------------------------------

/**
 * DiT の S 形グラフの入力名（recipe `wan/export_dit.py` / `wan/ti2v_export_dit.py` の forward の引数名 — 製品の `dit-loop.ts`
 * の非公開の定数と同じ綴り。違えば {@link f32Input} / {@link boolInput} が名指しで落とす）。
 */
const DIT_TOKENS = "tokens";
const DIT_TIMESTEPS_PROJ = "timesteps_proj";
const DIT_TIMESTEPS_PROJ_CONDITION = "timesteps_proj_condition";
const DIT_CONDITION_MASK = "condition_mask";

/** run の入力の f32 のホストテンソル（無い・常駐・dtype 違いは fail loudly）。 */
const f32Input = (inputs: RunInputs, name: string): Extract<Tensor, { readonly dtype: "f32" }> => {
  const input = inputs[name];
  if (input === undefined || !("data" in input) || input.dtype !== "f32") {
    throw new Error(
      `入力 '${name}' が f32 のホストテンソルでない（入力: ${Object.keys(inputs).join(", ")}）`,
    );
  }
  return input;
};

/** run の入力の bool のホストテンソル（{@link f32Input} と同じ規律）。 */
const boolInput = (
  inputs: RunInputs,
  name: string,
): Extract<Tensor, { readonly dtype: "bool" }> => {
  const input = inputs[name];
  if (input === undefined || !("data" in input) || input.dtype !== "bool") {
    throw new Error(
      `入力 '${name}' が bool のホストテンソルでない（入力: ${Object.keys(inputs).join(", ")}）`,
    );
  }
  return input;
};

/**
 * 部品の Session を、`method` の入力を `hook` に通す Proxy で包んだ写し（観測と故障注入の口 — 製品のコードに口を足さない）。
 * `run` の出力は `onOutputs` が受ける（観測 — 値はそのまま返す）。他のメソッドは元の Session に束ねて渡す（Session は `#` の
 * private を持つので、`this` を Proxy にすると落ちる）。
 */
const withSessionHook = (
  component: ModelComponent,
  method: "run" | "enqueue",
  hook: (inputs: RunInputs) => RunInputs,
  onOutputs?: (outputs: RunOutputs) => void,
): ModelComponent => ({
  ...component,
  createSession: async (gpu, options) => {
    const session = await component.createSession(gpu, options);
    return new Proxy<Session>(session, {
      get: (target, property) => {
        if (property === "run" && method === "run") {
          return (...args: Parameters<Session["run"]>) => {
            const [inputs, ...rest] = args;
            return target.run(hook(inputs), ...rest).then((outputs) => {
              onOutputs?.(outputs);
              return outputs;
            });
          };
        }
        if (property === "enqueue" && method === "enqueue") {
          return (...args: Parameters<Session["enqueue"]>) => {
            const [inputs, ...rest] = args;
            return target.enqueue(hook(inputs), ...rest);
          };
        }
        const value: unknown = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  },
});

/**
 * patchify の置換（トークンの並びの各要素が潜在のどの添字から来たか）。製品の `patchifyLatents` に添字の列を通して作る
 * （潜在の要素数 506,880 は f32 で正確に表せる）。置換であること（各添字がちょうど 1 回）もここで確かめる。
 */
const tokenSourcesOf = (patch: WanPatchGeometry): Int32Array<ArrayBuffer> => {
  const count = LATENT_SHAPE.reduce((product, dim) => product * dim, 1);
  const indices = new Float32Array(count);
  for (let index = 0; index < count; index += 1) indices[index] = index;
  const sources = Int32Array.from(patchifyLatents(indices, LATENT_SHAPE, patch));
  const seen = new Uint8Array(count);
  for (const source of sources) seen[source] += 1;
  assert(seen.every((times) => times === 1), "patchify が潜在の添字の置換になっていない");
  return sources;
};

/** DiT の 1 run の入力（観測 — `modelInput` は patchify を戻したモデル入力の潜在 `[C, F, H, W]`）。 */
type DitRunInputs = {
  readonly modelInput: Float32Array<ArrayBuffer>;
  readonly proj: Float32Array<ArrayBuffer>;
  readonly conditionProj: Float32Array<ArrayBuffer>;
  readonly mask: Uint32Array<ArrayBuffer>;
};

/** 1 回の generate で Session に渡った入力の記録。 */
type SessionRecord = {
  /** DiT の run ごと（改変の後 — DiT が実際に受けた値）。 */
  readonly dit: DitRunInputs[];
  /** DiT の run ごとの出力（unpatchify の後の `[C, F, H, W]` — 参照の `noise_uncond.<i>` / `noise_cond.<i>` と同じ並び）。 */
  readonly ditOutputs: Float32Array<ArrayBuffer>[];
  /** VAE の先頭 chunk（`vae_decoder_first`）の enqueue ごとの `latent`（逆正規化済みのタイル `[C, 1, t, t]`）。 */
  readonly vaeFirstLatents: Float32Array<ArrayBuffer>[];
  /** VAE の後続 chunk（`vae_decoder_next` — 潜在のフレーム 1 以降）の enqueue ごとの `latent`（同じ形）。 */
  readonly vaeNextLatents: Float32Array<ArrayBuffer>[];
};

const emptyRecord = (): SessionRecord => ({
  dit: [],
  ditOutputs: [],
  vaeFirstLatents: [],
  vaeNextLatents: [],
});

/**
 * 状態の `transformer` と `vae_decoder_first` / `vae_decoder_next` の Session を包んだ写し（入力を `record` へ写す）。`alter` は
 * DiT の run の入力の改変（故障注入 — 省けば入力のオブジェクトをそのまま渡す）。
 */
const probedState = (
  state: WanState,
  record: SessionRecord,
  sources: Int32Array,
  alter?: (inputs: RunInputs) => RunInputs,
): WanState => ({
  ...state,
  transformer: withSessionHook(state.transformer, "run", (inputs) => {
    const altered = alter === undefined ? inputs : alter(inputs);
    const tokens = f32Input(altered, DIT_TOKENS).data;
    assertEquals(tokens.length, sources.length, "DiT の tokens の要素数");
    const modelInput = new Float32Array(sources.length);
    for (let index = 0; index < sources.length; index += 1) {
      modelInput[sources[index]] = tokens[index];
    }
    record.dit.push({
      modelInput,
      proj: Float32Array.from(f32Input(altered, DIT_TIMESTEPS_PROJ).data),
      conditionProj: Float32Array.from(f32Input(altered, DIT_TIMESTEPS_PROJ_CONDITION).data),
      mask: Uint32Array.from(boolInput(altered, DIT_CONDITION_MASK).data),
    });
    return altered;
  }, (outputs) => {
    record.ditOutputs.push(
      unpatchifyTokens(f32Input(outputs, state.dit.output).data, LATENT_SHAPE, state.dit.patch),
    );
  }),
  vaeFirst: withSessionHook(state.vaeFirst, "enqueue", (inputs) => {
    record.vaeFirstLatents.push(Float32Array.from(f32Input(inputs, WAN_VAE_LATENT_INPUT).data));
    return inputs;
  }),
  vaeNext: withSessionHook(state.vaeNext, "enqueue", (inputs) => {
    record.vaeNextLatents.push(Float32Array.from(f32Input(inputs, WAN_VAE_LATENT_INPUT).data));
    return inputs;
  }),
});

/**
 * DiT の入力の結線（モジュール doc「門」）を実走の記録で見て、条件の潜在（モデル入力の先頭フレーム）を返す。
 * `stepLatents` は観測した各 step の後のスケジューラの潜在。
 */
const assertDitWiring = (
  runs: readonly DitRunInputs[],
  latentsInit: Float32Array,
  stepLatents: readonly Float32Array[],
  where: string,
): Float32Array<ArrayBuffer> => {
  assertEquals(
    runs.length,
    STEPS * 2,
    `${where}: DiT の run の数（CFG の 2 パス × ${STEPS} step）`,
  );
  const [, frames] = LATENT_SHAPE;
  const condition = latentFrame(runs[0].modelInput, 0);
  const expectedMask = new Uint32Array(runs[0].mask.length);
  expectedMask.fill(1, 0, CONDITION_TOKENS);
  // 条件側の時刻の期待（上流の `first_frame_mask · t` — 条件のトークンは t = 0。幅は生成側の proj と同じ）。
  const zeroProj = timestepsProj(0, runs[0].proj.length);
  runs.forEach((run, index) => {
    const step = Math.floor(index / 2);
    const label = `${where}: step ${step} の run ${index}`;
    assertEquals(
      firstValueMismatch(run.mask, expectedMask),
      -1,
      `${label} の条件マスクが先頭 ${CONDITION_TOKENS} トークンだけ真でない`,
    );
    // 全 run で t = 0 の proj なので、step に依らないことと生成側（t > 0）と違うことも含む。
    assert(
      bitsEqual(run.conditionProj, zeroProj),
      `${label} の条件側の時刻が t = 0 の proj（timestepsProj(0)）とビット一致しない`,
    );
    assertEquals(
      firstValueMismatch(latentFrame(run.modelInput, 0), condition),
      -1,
      `${label} のモデル入力の先頭フレームが run 0 と違う（条件の潜在で置き換わっていない）`,
    );
    const current = step === 0 ? latentsInit : stepLatents[step - 1];
    for (let frame = 1; frame < frames; frame += 1) {
      assertEquals(
        firstValueMismatch(latentFrame(run.modelInput, frame), latentFrame(current, frame)),
        -1,
        `${label} のモデル入力のフレーム ${frame} がその step のスケジューラの状態と違う`,
      );
    }
  });
  return condition;
};

/**
 * VAE の段の入力（逆正規化済みのタイル）が、置き換えの関数を通さずに組んだ期待と値で一致する（モジュール doc「最後の潜在の
 * 観測」）: 先頭 chunk（潜在のフレーム 0）は条件の潜在、後続 chunk（フレーム 1 以降）は最後のスケジューラの状態
 * `schedulerLatents` のそのフレーム — どちらも逆正規化を参照のタイルの窓で切ったもの。ループの後の置き換えが VAE に
 * 届いたことと、置き換えないフレームが最後の状態のまま渡ったことを、VAE が実際に受けた値で縛る。窓の順には依らない
 * （chunk ごとに多重集合で突き合わせる）。
 */
const assertVaeReceivesLatents = (
  record: Pick<SessionRecord, "vaeFirstLatents" | "vaeNextLatents">,
  condition: Float32Array,
  schedulerLatents: Float32Array,
  fixture: Fixture,
  where: string,
): void => {
  const tile = Number(fixture.meta("tile"));
  const rows = fixture.meta("rows_starts").split(",").map(Number);
  const cols = fixture.meta("cols_starts").split(",").map(Number);
  const [channels, frames, height, width] = LATENT_SHAPE;
  const stats = WAN22_TI2V_FAMILY.generation.latents;
  const expected = Array.from(
    { length: frames },
    (_, frame) =>
      denormalizeWanLatents(frame === 0 ? condition : latentFrame(schedulerLatents, frame), stats),
  );
  const chunks = [
    {
      label: "先頭 chunk（vae_decoder_first）",
      recorded: record.vaeFirstLatents,
      frames: [0],
      cause: "ループの後の置き換えが VAE に届いていない",
    },
    {
      label: "後続 chunk（vae_decoder_next）",
      recorded: record.vaeNextLatents,
      frames: Array.from({ length: frames - 1 }, (_, index) => index + 1),
      cause: "置き換えないフレームが最後のスケジューラの状態でない（最後のモデル入力の写しなど）",
    },
  ];
  for (const chunk of chunks) {
    assertEquals(
      chunk.recorded.length,
      rows.length * cols.length * chunk.frames.length,
      `${where}: VAE の${chunk.label}の enqueue の数（タイルの数 × フレーム数）`,
    );
    const unmatched = [...chunk.recorded];
    for (const frame of chunk.frames) {
      for (const top of rows) {
        for (const left of cols) {
          assert(
            top + tile <= height && left + tile <= width,
            `${where}: 窓 (${top}, ${left}) が潜在の外`,
          );
          const window = new Float32Array(channels * tile * tile);
          for (let channel = 0; channel < channels; channel += 1) {
            for (let y = 0; y < tile; y += 1) {
              const from = (channel * height + top + y) * width + left;
              window.set(
                expected[frame].subarray(from, from + tile),
                (channel * tile + y) * tile,
              );
            }
          }
          const at = unmatched.findIndex((latent) => firstValueMismatch(latent, window) === -1);
          assert(
            at !== -1,
            `${where}: VAE の${chunk.label}の入力に、潜在のフレーム ${frame}・窓 (row ${top}, col ${left}) の` +
              `期待の逆正規化が無い（${chunk.cause}か、タイルの窓が参照の計画と違う）`,
          );
          unmatched.splice(at, 1);
        }
      }
    }
  }
};

/**
 * 条件の潜在の門（encoder の段の出口 — 通しの中で DiT が受けた先頭フレームと参照の `condition_latents` の最大絶対差に、
 * encoder の GPU の門と同じ帯 {@link WAN_I2V_ENCODER_BAND} を当てる）。encoder の退行（束縛の取り違え・正規化の統計の
 * 取り違え・f16 の丸め）は通しの中では「条件」として自己整合するので、潜在の帯に任せると真因の段を名指しできない
 * （既定のケースの boxing-cats は DiT が encoder の差をほとんど増幅しない — {@link LATENT_RATIO_BANDS} の導出の注記）。
 */
const assertConditionInBand = (diff: Difference, where: string): void => {
  assertEquals(diff.nonFinite, 0, `${where}: 条件の潜在の非有限`);
  assert(
    diff.maxAbs <= WAN_I2V_ENCODER_BAND,
    `${where}: 条件の潜在（VAE encoder の段の出口・正規化の後）と参照の condition_latents の maxAbs ` +
      `${diff.maxAbs.toExponential(3)} が encoder の帯 ${WAN_I2V_ENCODER_BAND} の外`,
  );
};

// ---- 生成 1 回の観測 ------------------------------------------------------------------------------------

/** 生成の口（1 層内側の `generateWanVideo` か、公開の `WanTi2vPipeline.generate`）。 */
type Generate = (request: WanTi2vGenerateRequest) => Promise<GeneratedVideo>;

const viaState =
  (state: WanState, spec: WanFamilySpec = WAN22_TI2V_FAMILY): Generate => (request) =>
    generateWanVideo(spec, state, request);

const viaPipeline = (pipeline: WanTi2vPipeline): Generate => (request) =>
  pipeline.generate(request);

/** 1 回の generate の観測（イベントから — 各 step の後の潜在・段の所要・段の境目の VRAM）。 */
type Observed = {
  readonly video: GeneratedVideo;
  readonly latents: readonly Float32Array<ArrayBuffer>[];
  /** 段名 → 壁時計の ms（`start` から `end` まで）。 */
  readonly stageMs: Readonly<Record<string, number>>;
  readonly wallMs: number;
  readonly drm: DrmTimeline;
  readonly diagnostics: ReadonlyMap<WanRunComponent, SessionDiagnostics>;
};

/**
 * generate を 1 回まわし、イベントから観測を集める。`abortAfterDenoise` なら DiT の段の `end`（Session を畳んだ後）で投げて
 * VAE の段を飛ばす（故障注入は潜在だけを見る — 28 タイルの VAE を払わない）。`onLatents` は各 step の後のスケジューラの潜在を
 * 受ける（故障注入「置き換えない」が次の step のモデル入力を作る）。
 */
const observe = async (
  generate: Generate,
  diagnostics: Map<WanRunComponent, SessionDiagnostics>,
  request: Omit<WanTi2vGenerateRequest, "onEvent">,
  options: {
    readonly abortAfterDenoise?: boolean;
    readonly onLatents?: (latents: Float32Array<ArrayBuffer>) => void;
  } = {},
): Promise<Observed | { readonly latents: readonly Float32Array<ArrayBuffer>[] }> => {
  const latents: Float32Array<ArrayBuffer>[] = [];
  const stageStarted = new Map<string, number>();
  const stageMs: Record<string, number> = {};
  const monitor = monitorDrmUsage();
  const abort = new Error("DiT の段の後で止める（故障注入）");
  diagnostics.clear();
  const started = performance.now();
  const onEvent = (event: WanGenerateEvent): void => {
    if (event.kind === "denoise-step") {
      const data = event.copyLatents().data;
      latents.push(data);
      options.onLatents?.(data);
    }
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
    const video = await generate({ ...request, onEvent });
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
      }・幾何 ${diagnostics.geometryProfile}・submit ${diagnostics.submit.submitCount} 本`,
    );
  }
  return lines;
};

/**
 * 段の並び（encoder → DiT — 事前計算の経路は text の段を持たない）と encoder の 3 グラフの診断（条件画像の要求の実走の結線）。
 */
const assertEncoderStage = (observed: Observed, where: string): void => {
  assertEquals(
    observed.drm.marks.slice(0, 3).map(({ label }) => label),
    ["vae_encoder start", "vae_encoder end", "transformer start"],
    `${where}: encoder の段 → DiT の段の順`,
  );
  for (const component of ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"] as const) {
    assert(
      observed.diagnostics.has(component),
      `${where}: ${component} の run の診断が届いていない`,
    );
  }
};

/** 2 ステップの通しの要求の共通部分（プロンプト・画像・初期ノイズはケースが足す）。 */
const TWO_STEP = { steps: STEPS, guidance: GUIDANCE, shift: SHIFT, frames: FRAMES } as const;

Deno.test({
  name:
    "Wan2.2 I2V 通し（ホスト・GPU 不要）: 参照のメタ・画像の来歴と crop・統計 48 値が TS と一致し、参照の DiT 出力から " +
    "CFG + UniPC が参照の潜在を、置き換えの関数が参照の最後の潜在をビット一致で作り直す",
  ignore: !ANY_FIXTURE,
  fn: async () => {
    const embedsSha = await sha256Hex(await Deno.readFile(TEXT_EMBEDS_URL));
    const schedule = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
    const mean = Float32Array.from(WAN22_LATENTS_MEAN);
    const std = Float32Array.from(WAN22_LATENTS_STD);
    const patch = wanDitPatch(LATENT_SHAPE[0]);
    const mask = wanConditionMask("ti2v", wanTokenGrid(LATENT_SHAPE, patch), patch, true, "test");
    assert(mask !== undefined, "ti2v の条件マスクが無い");
    // 製品の条件マスクの真のトークン数（参照のメタ `condition_tokens` と下で突き合わせる）。
    assertEquals(
      mask.reduce((count, value) => count + value, 0),
      CONDITION_TOKENS,
      "製品の条件マスク",
    );
    for (const { name, role } of CASES) {
      const fixture = await readFixture(fixtureUrl(name));
      assertEquals(fixture.meta("role"), role, `${name}: 役割`);
      assertEquals(
        fixture.meta("text_embeds_sha256"),
        embedsSha,
        `${name}: 参照を作った埋め込み資産`,
      );
      assertEquals(
        [
          fixture.meta("steps"),
          fixture.meta("guidance_scale"),
          fixture.meta("flow_shift"),
          fixture.meta("expand_timesteps"),
          fixture.meta("timestep_mlp"),
          fixture.meta("condition_tokens"),
          fixture.meta("condition_timestep"),
        ],
        [String(STEPS), "5.0", "5.0", "true", "per-value-m1", String(CONDITION_TOKENS), "0"],
        `${name}: step 数・guidance・shift・時刻の経路・条件のトークン数と時刻`,
      );
      assertEquals(fixture.meta("negative"), "negative", `${name}: negative の行名`);
      // 画像の来歴: プロンプトと画像は同じ名前・PNG の sha は段 9a の golden の来歴と同じ値。
      const image = fixture.meta("image");
      assertEquals(fixture.meta("prompt"), image, `${name}: プロンプトと画像が同じ名前でない`);
      assertEquals(
        fixture.meta("image_sha256"),
        WAN_I2V_IMAGE_SHA256[image],
        `${name}: 画像の sha256`,
      );
      assertEquals(
        [
          fixture.meta("width"),
          fixture.meta("height"),
          fixture.meta("fit"),
          fixture.meta("resample"),
        ],
        [String(WIDTH), String(HEIGHT), FIT, "LANCZOS"],
        `${name}: 寸法・fit・resample`,
      );
      const source = sourceImageOf(fixture);
      const rgb8 = fixture.bytes("rgb8");
      assertEquals(fixture.shape("rgb8"), [HEIGHT, WIDTH, 3], `${name}: 切った RGB8 の形`);
      assertEquals(
        await sha256Hex(fixture.bytes("source")),
        fixture.meta("source_rgb8_sha256"),
        `${name}: 元画像の RGB8 の sha256`,
      );
      assertEquals(
        await sha256Hex(rgb8),
        fixture.meta("rgb8_sha256"),
        `${name}: 切った RGB8 の sha256`,
      );
      // 製品の前処理: 寸法は画像の縦横比で 1280×704、crop は参照の crop とビット一致。
      const selected = preprocessWanI2vImage(source, {}, WAN22_TI2V_FAMILY.generation);
      assertEquals([selected.width, selected.height], [WIDTH, HEIGHT], `${name}: 寸法の選択`);
      const cropped = fitWanI2vImage(source, { width: WIDTH, height: HEIGHT }, "crop");
      const mismatch = cropped.data.findIndex((value, index) => value !== rgb8[index]);
      assertEquals(mismatch, -1, `${name}: TS の crop が参照の rgb8 と画素 ${mismatch} で割れる`);
      assertEquals(fixture.shape("latents_init"), LATENT_SHAPE, `${name}: 潜在の形`);
      assertEquals(fixture.shape("condition_latents"), CONDITION_SHAPE, `${name}: 条件の潜在の形`);
      assertEquals(fixture.shape("frames"), [3, FRAMES, HEIGHT, WIDTH], `${name}: フレームの形`);
      assert(
        bitsEqual(fixture.tensor("latents_mean"), mean),
        `${name}: latents_mean が WAN22 と違う`,
      );
      assert(bitsEqual(fixture.tensor("latents_std"), std), `${name}: latents_std が WAN22 と違う`);
      const timesteps: unknown = JSON.parse(fixture.meta("timesteps"));
      const sigmas: unknown = JSON.parse(fixture.meta("sigmas"));
      assertEquals(timesteps, schedule.timesteps, `${name}: timesteps`);
      assert(
        Array.isArray(sigmas) && sigmas.every((value) => typeof value === "number"),
        `${name}: sigmas が数の列でない`,
      );
      assert(
        bitsEqual(Float32Array.from(sigmas), schedule.sigmas),
        `${name}: sigmas ${JSON.stringify(sigmas)} が f32 で ${[...schedule.sigmas]} と一致しない`,
      );
      // 参照が記録した DiT の出力から、ホストの CFG + UniPC（置き換え無し）で参照の潜在を作り直す。
      const sampler = new WanUniPcSampler(schedule, WAN_UNIPC_CONFIG);
      let latents: Float32Array = fixture.tensor("latents_init");
      for (let step = 0; step < STEPS; step += 1) {
        assertEquals(
          fixture.shape(`latents.${step}`),
          LATENT_SHAPE,
          `${name}: latents.${step} の形`,
        );
        const velocity = wanClassifierFreeGuidance(
          fixture.tensor(`noise_cond.${step}`),
          fixture.tensor(`noise_uncond.${step}`),
          GUIDANCE,
        );
        latents = sampler.step(velocity, latents);
        const want = fixture.tensor(`latents.${step}`);
        assert(
          bitsEqual(latents, want),
          `${name}: step ${step} の潜在がビット一致しない（${
            formatDifference("潜在", difference(latents, want))
          }）`,
        );
      }
      // 最後の潜在 = 最後のスケジューラの潜在の先頭フレームを条件の潜在で置き換えた値（製品の関数で作り直す）。
      const final = withWanFirstFrameCondition(
        fixture.tensor(`latents.${STEPS - 1}`),
        fixture.tensor("condition_latents"),
        LATENT_SHAPE,
      );
      assert(
        bitsEqual(final, fixture.tensor("latents_final")),
        `${name}: 最後の潜在がビット一致しない（${
          formatDifference("latents_final", difference(final, fixture.tensor("latents_final")))
        }）`,
      );
    }
  },
});

/** 観測点（帯を持つ出力 — 各 step の後のスケジューラの潜在・最後の潜在・クランプ後のフレーム）。 */
const OBSERVATION_POINTS = ["latents.0", "latents.1", "latents_final", "frames"] as const;
type ObservationPoint = typeof OBSERVATION_POINTS[number];

/** 観測点ごとの帯（未導出は `undefined`）。 */
const bandOf = (point: ObservationPoint): number | undefined => {
  switch (point) {
    case "latents.0":
      return LATENT_RATIO_BANDS[0];
    case "latents.1":
      return LATENT_RATIO_BANDS[1];
    case "latents_final":
      return FINAL_RATIO_BAND;
    case "frames":
      return FRAME_RATIO_BAND;
  }
};

const BANDS_DERIVED = OBSERVATION_POINTS.every((point) => bandOf(point) !== undefined);

/** 1 ケースの観測点ごとの比（帯の候補の材料 — 非有限と device lost の検査を通った後・帯の判定の前に積む）。 */
type CaseRatios = {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly ratios: Readonly<Record<ObservationPoint, number>>;
};

/**
 * 帯の候補（最悪 × 5 を有効数字 2 桁へ切り上げ — T2V の通しの e2e の同名の関数と同じ規則）。割り算の誤差で、ちょうど 2 桁の値を
 * 1 単位上へ切り上げないよう、商を 12 桁へ丸めてから切り上げる。
 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(Number((value / unit).toPrecision(12))) * unit).toPrecision(2));
};

Deno.test({
  name:
    "Wan2.2 I2V 通し 2 ステップ（実 GPU）: 1280×704・9 フレームの潜在・最後の潜在・フレームが参照と帯の内・DiT の入力の結線・" +
    "故障注入は帯の外・sha256 の環境行（参照席の帯のケース・実用席・opt-in の縦長）",
  ignore: !DIST_PRESENT || !GPU_AVAILABLE || !ANY_FIXTURE,
  fn: async (t) => {
    await assertRunningAdapter();
    let deviceLost: string | undefined;
    const gpu = await acquireTestGpu({
      onDeviceLost: (info) => {
        deviceLost = `${info.reason}: ${info.message}`;
      },
    });
    const diagnostics = new Map<WanRunComponent, SessionDiagnostics>();
    const onRunDiagnostics = (component: WanRunComponent, diagnosed: SessionDiagnostics): void => {
      diagnostics.set(component, diagnosed);
    };
    /** 参照席の既定のケースの実物（RGB のバイト列）— 同じ要求の実用席のケースの床の比べる相手。 */
    let referenceRgb: Uint8Array<ArrayBuffer> | undefined;
    /** 既定のケースの参照（故障注入と実用席・縦長が読む）。 */
    let defaultFixture: Fixture | undefined;
    const defaultFixtureOf = async (): Promise<Fixture> =>
      defaultFixture ??= await readFixture(fixtureUrl(DEFAULT_CASE));
    /** 既定のケースの要求（実用席も同じ要求 — 席だけが違う）。 */
    const defaultRequest = async (
      prompts: readonly WanPrompt[],
    ): Promise<Omit<WanTi2vGenerateRequest, "onEvent">> => {
      const fixture = await defaultFixtureOf();
      return {
        ...TWO_STEP,
        prompt: textOf(prompts, fixture.meta("prompt"), "prompt"),
        negativePrompt: textOf(prompts, fixture.meta("negative"), "prompt"),
        latents: fixture.tensor("latents_init"),
        image: sourceImageOf(fixture),
      };
    };
    /**
     * sha256 の環境行を突き合わせる（または書く）1 本の末端。`beforeSettle` は突合と書き込みの前に実物を受ける
     * （`KARUME_REFERENCE=write` の走行でも、行を書く前に効かせる検査の置き場 — 実用席の床）。
     */
    const settleCase = async (
      id: string,
      video: GeneratedVideo,
      beforeSettle?: (bytes: Uint8Array<ArrayBuffer>) => void,
    ) => {
      const bytes = rgbBytes(video);
      beforeSettle?.(bytes);
      return await settleOrObserve(references, results, { id, artifact: `${id}.rgb`, bytes });
    };
    try {
      // 参照席の 1 層内側の状態（DiT の入力を観測する回 — モジュール doc「観測と故障注入の口」）。
      const state = await loadWanFromPretrained(
        WAN22_TI2V_FAMILY,
        denoDirectory(WAN_TI2V_DIST_ROOT),
        { gpu, textEncoder: "precomputed", quant: WAN_TI2V_REFERENCE_QUANT, onRunDiagnostics },
      );
      const prompts: readonly WanPrompt[] = state.textEmbeds.entries;
      const sources = tokenSourcesOf(state.dit.patch);
      const caseRatios: CaseRatios[] = [];

      for (const { name, role, optIn } of CASES) {
        const id = bandCaseId(name);
        await t.step({
          name: `${id}（${role}${optIn ? "・opt-in" : ""}）`,
          ignore: optIn && !I2V_FULL,
          fn: async () => {
            const fixture = await readFixture(fixtureUrl(name));
            assertEquals(fixture.meta("role"), role, `${name}: 役割`);
            // band は原文・accept は正規化後の文字列で引く（受理集合の 2 つの綴りを両方通す — T2V と同じ）。
            const prompt = textOf(
              prompts,
              fixture.meta("prompt"),
              role === "band" ? "prompt" : "normalized",
            );
            const latentsInit = fixture.tensor("latents_init");
            let settlement: ReferenceSettlement | undefined;
            let caseNote = "";
            try {
              await runRecordedCase(
                results,
                { id, failureNote: () => caseNote },
                async ({ measurements }) => {
                  const record = emptyRecord();
                  const observed = await observe(
                    viaState(probedState(state, record, sources)),
                    diagnostics,
                    {
                      ...TWO_STEP,
                      prompt,
                      negativePrompt: textOf(prompts, fixture.meta("negative"), "prompt"),
                      latents: latentsInit,
                      // 寸法は省く（画像の縦横比で 1280×704 を選ぶ — 既定の使い方）・fit も省く（既定の crop）。
                      image: sourceImageOf(fixture),
                    },
                  );
                  assert("video" in observed, "generate が最後まで回っていない");
                  assertEquals(observed.latents.length, STEPS, "denoise-step の数");
                  const { video } = observed;
                  assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
                  assertEncoderStage(observed, id);
                  const condition = assertDitWiring(record.dit, latentsInit, observed.latents, id);
                  const last = observed.latents[STEPS - 1];
                  assertVaeReceivesLatents(record, condition, last, fixture, id);
                  // VAE の結線の検査が実データで赤になること（GPU の時間 0）。突き合わせは値の一致なので、期待をその故障で
                  // VAE が受けたはずの値に替えて赤になることは、VAE がその値を受けたときに赤になることと同じ: ループの後の
                  // 置き換えが落ちた形（先頭フレームがスケジューラの状態のまま）と、最後のモデル入力の写しを渡す形
                  // （置き換えないフレームが 1 つ前の状態）。
                  assertThrows(
                    () => assertVaeReceivesLatents(record, latentFrame(last, 0), last, fixture, id),
                    Error,
                    "先頭 chunk（vae_decoder_first）",
                  );
                  assertThrows(
                    () =>
                      assertVaeReceivesLatents(
                        record,
                        condition,
                        observed.latents[STEPS - 2],
                        fixture,
                        id,
                      ),
                    Error,
                    "後続 chunk（vae_decoder_next）",
                  );
                  const final = withWanFirstFrameCondition(
                    observed.latents[STEPS - 1],
                    condition,
                    LATENT_SHAPE,
                  );
                  const diffs = OBSERVATION_POINTS.map((point) => ({
                    point,
                    band: bandOf(point),
                    diff: point === "frames"
                      // 参照のフレームはクランプ後で書かれている（`ti2v_few_step_ref.decode_frames` の clamp）。
                      ? difference(video.data, fixture.tensor("frames"))
                      : point === "latents_final"
                      ? difference(final, fixture.tensor("latents_final"))
                      : difference(
                        observed.latents[point === "latents.0" ? 0 : 1],
                        fixture.tensor(point),
                      ),
                  }));
                  for (const { point, band, diff } of diffs) {
                    measurements.push({
                      output: point,
                      maxAbs: diff.maxAbs,
                      maxRel: diff.maxRel,
                      // 未導出の帯は無限の帯として記録する（判定は下で赤にする — T2V と同じ扱い）。
                      tolerance: band === undefined
                        ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
                        : { atol: band * diff.referenceMax, rtol: 0 },
                      stage: "karume",
                    });
                  }
                  const conditionDiff = difference(condition, fixture.tensor("condition_latents"));
                  const notes = [
                    ...diffs.map(({ point, diff }) => formatDifference(point, diff)),
                    ...(["latents.0", "latents.1"] as const).map((point, step) =>
                      `${point} のフレームごとの maxAbs（観測）[${
                        frameMaxAbs(observed.latents[step], fixture.tensor(point))
                      }]`
                    ),
                    `${
                      formatDifference("条件の潜在", conditionDiff)
                    }・encoder の帯 ${WAN_I2V_ENCODER_BAND} の ${
                      (conditionDiff.maxAbs / WAN_I2V_ENCODER_BAND).toFixed(2)
                    } 倍`,
                    // step 0 のモデル入力は参照と条件の潜在の差しか違わないので、DiT の forward の差をここで切り出せる。
                    ...(["noise_uncond.0", "noise_cond.0"] as const).map((key, pass) =>
                      `DiT の出力 ${key}（観測・フレームごとの maxAbs [${
                        frameMaxAbs(record.ditOutputs[pass], fixture.tensor(key))
                      }]） ${
                        formatDifference(
                          "",
                          difference(record.ditOutputs[pass], fixture.tensor(key)),
                        )
                      }`
                    ),
                    formatFirstFrame(firstFrameDifference(video, fixture.bytes("rgb8"))),
                    ...formatObserved(observed),
                  ];
                  caseNote = notes.join(" / ");
                  console.log(`[wan-ti2v-i2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
                  for (const { point, diff } of diffs) {
                    assertEquals(diff.nonFinite, 0, `${id}: ${point} の非有限`);
                  }
                  // 条件の門は帯の候補の材料を積む前（encoder が壊れた回の比で帯を導かない）。
                  assertConditionInBand(conditionDiff, id);
                  assertEquals(deviceLost, undefined, "device lost");
                  caseRatios.push({
                    name,
                    role,
                    ratios: {
                      "latents.0": diffs[0].diff.ratio,
                      "latents.1": diffs[1].diff.ratio,
                      "latents_final": diffs[2].diff.ratio,
                      frames: diffs[3].diff.ratio,
                    },
                  });
                  for (const { point, band, diff } of diffs) {
                    if (band === undefined) {
                      throw new Error(
                        `${id}: ${point} の帯が未導出（比 ${diff.ratio.toExponential(3)}）— ` +
                          "帯の候補は「帯の候補」の step が出す（opt-in で決定用の 2 本を回す）",
                      );
                    }
                    assert(
                      diff.ratio <= band,
                      `${id}: ${formatDifference(point, diff)} が帯 ${band} の外`,
                    );
                  }
                  const outcome = await settleCase(id, video, (bytes) => {
                    if (name === DEFAULT_CASE) referenceRgb = bytes;
                  });
                  settlement = outcome.settlement;
                  return { ...outcome.fields, note: caseNote };
                },
              );
            } finally {
              // 次のケースの確保の前に、このケースの段の確保の解放を待つ（B570 の `destroy()` の遅れ）。
              await settleReleases(gpu);
            }
            if (settlement?.check.status === "fail") {
              throw new Error(referenceMismatchMessage(id, settlement, references));
            }
          },
        });
      }

      // 未導出の帯の候補（決定用だけから作る — 受入れの比は並べるが候補に使わない。決定用のどれかが比を出す前に落ちた回・
      // opt-in を付けずに決定用の 2 本目が回っていない回は候補を出さない）。帯が導出済みなら step を作らない。
      if (!BANDS_DERIVED) {
        await t.step("帯の候補（未導出）", () => {
          const expected = CASES.filter(({ role }) => role === "band").length;
          const decided = caseRatios.filter(({ role }) => role === "band");
          const candidate = decided.length === expected
            ? OBSERVATION_POINTS.map((point) => {
              const worst = Math.max(...decided.map(({ ratios }) => ratios[point]));
              return `${point}: 決定用 ${expected} 本の最悪 ${worst.toExponential(3)} × 5 = ${
                (worst * 5).toExponential(3)
              }（有効数字 2 桁へ切り上げ ${roundUpTwoDigits(worst * 5)}）`;
            }).join(" / ")
            : `決定用 ${expected} 本のうち比が出たのは ${decided.length} 本 — 候補にしない` +
              "（opt-in KARUME_WAN_TI2V_I2V_FULL=1 で決定用の全ケースを回す・落ちたケースは先に調べる）";
          const accepted = caseRatios.filter(({ role }) => role === "accept").map((
            { name, ratios },
          ) =>
            `${name} ${
              OBSERVATION_POINTS.map((point) => `${point} ${ratios[point].toExponential(3)}`).join(
                "・",
              )
            }`
          );
          const lines = caseRatios.filter(({ role }) => role === "band").map(({ name, ratios }) =>
            `${name} ${
              OBSERVATION_POINTS.map((point) => `${point} ${ratios[point].toExponential(3)}`).join(
                "・",
              )
            }`
          );
          throw new Error(
            `帯が未導出 — ${candidate}。決定用の比: ${
              lines.join(" / ") || "なし"
            }。受入れの比（帯の決定に使わない）: ${accepted.join(" / ") || "なし"}`,
          );
        });
      }

      // 故障注入（既定のケースの初期ノイズ — 潜在だけを見るので VAE の段の前で止める）。
      const patch = state.dit.patch;
      const channels = WAN22_TI2V_FAMILY.generation.latents.mean.length;
      /** 統計を mean 0・std 1 にした家族の値（正規化の式が mu をそのまま返す — 「正規化しない」の注入）。 */
      const unnormalized: WanFamilySpec = {
        ...WAN22_TI2V_FAMILY,
        generation: {
          ...WAN22_TI2V_FAMILY.generation,
          latents: {
            mean: Array.from({ length: channels }, () => 0),
            std: Array.from({ length: channels }, () => 1),
          },
        },
      };
      const faults: readonly {
        readonly label: string;
        /** DiT の run の入力の改変（`current` は今の step のスケジューラの状態 — 置き換える前）。 */
        readonly alter?: (inputs: RunInputs, current: Float32Array) => RunInputs;
        /** 家族の値（省けば製品の値）。 */
        readonly spec?: WanFamilySpec;
        /**
         * 帯のほかに名指しで赤になるべき検査（DiT の入力の結線 — 文言の断片 / 条件の潜在の門）。結線と条件の門が
         * 実データで赤になることを、GPU の時間を足さずに同じ走行の記録で示す。
         */
        readonly named: { readonly kind: "wiring"; readonly fragment: string } | {
          readonly kind: "condition";
        };
      }[] = [
        {
          label: "条件マスクを全て偽",
          alter: (inputs) => {
            const mask = boolInput(inputs, DIT_CONDITION_MASK);
            return {
              ...inputs,
              [DIT_CONDITION_MASK]: { ...mask, data: new Uint32Array(mask.data.length) },
            };
          },
          named: { kind: "wiring", fragment: "の条件マスクが" },
        },
        {
          label: "条件マスクを 1 トークンずらす",
          alter: (inputs) => {
            const mask = boolInput(inputs, DIT_CONDITION_MASK);
            const shifted = new Uint32Array(mask.data.length);
            shifted.set(mask.data.subarray(0, mask.data.length - 1), 1);
            return { ...inputs, [DIT_CONDITION_MASK]: { ...mask, data: shifted } };
          },
          named: { kind: "wiring", fragment: "の条件マスクが" },
        },
        {
          // 時刻の入力 2 本の束ね違い（生成側のトークンが t = 0・条件のトークンが生成側の t を受ける）。「条件側の時刻を
          // 生成側と同じにする」は DiT が受ける値が「条件マスクを全て偽」と同じ（全トークンが生成側の時刻）で、帯の上では
          // 同じ走行の繰り返しになるので置かない（その故障は結線の検査の t = 0 の照合が名指しで落とす）。
          label: "時刻の入力 2 本を取り違える",
          alter: (inputs) => ({
            ...inputs,
            [DIT_TIMESTEPS_PROJ]: f32Input(inputs, DIT_TIMESTEPS_PROJ_CONDITION),
            [DIT_TIMESTEPS_PROJ_CONDITION]: f32Input(inputs, DIT_TIMESTEPS_PROJ),
          }),
          named: { kind: "wiring", fragment: "の条件側の時刻が t = 0 の proj" },
        },
        {
          label: "モデル入力の先頭フレームを置き換えない",
          alter: (inputs, current) => {
            const tokens = f32Input(inputs, DIT_TOKENS);
            return {
              ...inputs,
              [DIT_TOKENS]: { ...tokens, data: patchifyLatents(current, LATENT_SHAPE, patch) },
            };
          },
          named: { kind: "wiring", fragment: "のモデル入力の先頭フレームが run 0 と違う" },
        },
        { label: "条件の潜在を正規化しない", spec: unnormalized, named: { kind: "condition" } },
      ];
      const lastBand = LATENT_RATIO_BANDS[STEPS - 1];
      for (const { label, alter, spec, named } of faults) {
        await t.step({
          name: `故障注入: ${label} → ${DEFAULT_CASE} の latents.${STEPS - 1} が帯の外`,
          fn: async () => {
            try {
              const fixture = await defaultFixtureOf();
              const record = emptyRecord();
              let current: Float32Array = fixture.tensor("latents_init");
              const altered = alter === undefined
                ? undefined
                : (inputs: RunInputs): RunInputs => alter(inputs, current);
              const observed = await observe(
                viaState(probedState(state, record, sources, altered), spec),
                diagnostics,
                await defaultRequest(prompts),
                {
                  abortAfterDenoise: true,
                  onLatents: (latents) => {
                    current = latents;
                  },
                },
              );
              // 段のイベント名が変わって打ち切りが効かないと、28 タイルの VAE を黙って払う（値の判定には出ない）。
              assert(!("video" in observed), "DiT の段の後で止まっていない");
              assertEquals(observed.latents.length, STEPS, "denoise-step の数");
              // 改変が全 run に掛かったこと（口が空振りしていない）。
              assertEquals(record.dit.length, STEPS * 2, "DiT の run の数");
              const diff = difference(
                observed.latents[STEPS - 1],
                fixture.tensor(`latents.${STEPS - 1}`),
              );
              console.log(
                `[wan-ti2v-i2v-pipeline] fault ${label}: ${
                  formatDifference(`latents.${STEPS - 1}`, diff)
                }（${
                  lastBand === undefined
                    ? "帯は未導出"
                    : `帯の ${(diff.ratio / lastBand).toFixed(1)} 倍`
                }）`,
              );
              if (lastBand === undefined) {
                throw new Error(
                  `${label}: 帯が未導出（故障注入の比 ${diff.ratio.toExponential(3)}）`,
                );
              }
              assert(
                diff.ratio > FAULT_MARGIN * lastBand,
                `${label}: 比 ${diff.ratio} が帯 ${lastBand} の ${FAULT_MARGIN} 倍を超えない`,
              );
              // 名指しの検査も同じ記録で赤になること（真因の段・結線を帯より先に指す口が効いている）。
              if (named.kind === "wiring") {
                assertThrows(
                  () =>
                    assertDitWiring(
                      record.dit,
                      fixture.tensor("latents_init"),
                      observed.latents,
                      label,
                    ),
                  Error,
                  named.fragment,
                );
              } else {
                assertThrows(
                  () =>
                    assertConditionInBand(
                      difference(
                        latentFrame(record.dit[0].modelInput, 0),
                        fixture.tensor("condition_latents"),
                      ),
                      label,
                    ),
                  Error,
                  "が encoder の帯",
                );
              }
            } finally {
              await settleReleases(gpu);
            }
          },
        });
      }

      // 縦長（opt-in・参照席・公開面 — 横長の元画像を明示の 704×1280 へ crop・seed 経路）。
      await t.step({
        name: `${PORTRAIT_CASE_ID}（縦長・公開面・sha256 の環境行・opt-in）`,
        ignore: !I2V_FULL,
        fn: async () => {
          let settlement: ReferenceSettlement | undefined;
          try {
            await using pipeline = await WanTi2vPipeline.fromPretrained(
              denoDirectory(WAN_TI2V_DIST_ROOT),
              {
                gpu,
                textEncoder: "precomputed",
                quant: WAN_TI2V_REFERENCE_QUANT,
                onRunDiagnostics,
              },
            );
            const fixture = await defaultFixtureOf();
            await runRecordedCase(results, { id: PORTRAIT_CASE_ID }, async () => {
              const observed = await observe(viaPipeline(pipeline), diagnostics, {
                ...TWO_STEP,
                prompt: textOf(pipeline.prompts, SEED_IMAGE, "prompt"),
                seed: SEED,
                image: sourceImageOf(fixture),
                width: HEIGHT,
                height: WIDTH,
                fit: "crop",
              });
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(observed.latents.length, STEPS, "denoise-step の数");
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [FRAMES, WIDTH, HEIGHT]);
              assertEncoderStage(observed, PORTRAIT_CASE_ID);
              const nonFinite = countNonFinite(video.data);
              const notes = [`非有限 ${nonFinite}`, ...formatObserved(observed)];
              console.log(`[wan-ti2v-i2v-pipeline] ${PORTRAIT_CASE_ID}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, `${PORTRAIT_CASE_ID}: 非有限`);
              assertEquals(deviceLost, undefined, "device lost");
              const outcome = await settleCase(PORTRAIT_CASE_ID, video);
              settlement = outcome.settlement;
              return { ...outcome.fields, note: notes.join(" / ") };
            });
          } finally {
            await settleReleases(gpu);
          }
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(PORTRAIT_CASE_ID, settlement, references));
          }
        },
      });

      // 実用席（公開面 — 既定のケースと同じ要求で席だけが違う）。
      await t.step({
        name: `${PRACTICAL_CASE_ID}（実用席・公開面・sha256 の環境行）`,
        fn: async () => {
          let settlement: ReferenceSettlement | undefined;
          try {
            await using practical = await WanTi2vPipeline.fromPretrained(
              denoDirectory(WAN_TI2V_DIST_ROOT),
              {
                gpu,
                textEncoder: "precomputed",
                quant: WAN_TI2V_PRACTICAL_QUANT,
                onRunDiagnostics,
              },
            );
            const request = await defaultRequest(practical.prompts);
            await runRecordedCase(results, { id: PRACTICAL_CASE_ID }, async () => {
              const observed = await observe(viaPipeline(practical), diagnostics, request);
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(observed.latents.length, STEPS, "denoise-step の数");
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
              assertEncoderStage(observed, PRACTICAL_CASE_ID);
              const nonFinite = countNonFinite(video.data);
              const fixture = await defaultFixtureOf();
              const notes = [
                `非有限 ${nonFinite}`,
                formatFirstFrame(firstFrameDifference(video, fixture.bytes("rgb8"))),
                ...formatObserved(observed),
              ];
              console.log(`[wan-ti2v-i2v-pipeline] ${PRACTICAL_CASE_ID}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, `${PRACTICAL_CASE_ID}: 非有限`);
              assertEquals(deviceLost, undefined, "device lost");
              const outcome = await settleCase(PRACTICAL_CASE_ID, video, (bytes) => {
                // 床（ADR 0110 決定 5 — 実用席は同じ要求の参照席と 1 bit 以上違う）: 席の `session` が Session に届かない
                // 退行では、実用席は参照席と同じバイトを出す。その実物を実用行として凍結しないよう、行を書く前に落とす。
                if (referenceRgb === undefined) {
                  throw new Error(
                    `${PRACTICAL_CASE_ID}: 同じ要求の参照席の実物が無い（参照席の ${DEFAULT_CASE} が回っていない — ` +
                      "床を確かめられない）",
                  );
                }
                const reference = referenceRgb;
                assert(
                  bytes.length !== reference.length ||
                    bytes.some((value, index) => value !== reference[index]),
                  `${PRACTICAL_CASE_ID}: 実用席の出力が参照席の同じ要求とバイト単位で一致した（席の session が届いていない）`,
                );
              });
              settlement = outcome.settlement;
              return { ...outcome.fields, note: notes.join(" / ") };
            });
          } finally {
            await settleReleases(gpu);
          }
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(PRACTICAL_CASE_ID, settlement, references));
          }
        },
      });
    } finally {
      // MUST: device を捨てる前に解放を待つ（B570 の destroy の遅れ — 後続のテストの予算を残す）。1 層内側の状態は GPU を
      // 自前で取っていない（`gpu` を渡した）ので、畳む資源を持たない。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

// NOTE: opt-in の env を付けて `--filter` なしで回すと、2 ステップの `Deno.test` と合わせて 1 プロセスで device を 2 つ順に
// 取る（B570 の `destroy()` が VRAM を返さない件に当たる）。`--filter "50 ステップ"` で単独のプロセスとして回す。
Deno.test({
  name:
    "Wan2.2 I2V 通し 50 ステップ（実 GPU・opt-in KARUME_WAN_TI2V_I2V_FULL=1）: 1280×704・33 フレーム（参照席と実用席）が" +
    "完走し非有限 0・所要と段の切り替えの VRAM・PNG 全フレーム + 一覧図",
  ignore: !I2V_FULL || !DIST_PRESENT || !GPU_AVAILABLE || !ANY_FIXTURE,
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
      const fixture = await readFixture(fixtureUrl(DEFAULT_CASE));
      for (const { id, quant } of FULL_CASES) {
        await t.step(id, async () => {
          try {
            await using pipeline = await WanTi2vPipeline.fromPretrained(
              denoDirectory(WAN_TI2V_DIST_ROOT),
              {
                gpu,
                textEncoder: "precomputed",
                quant,
                onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
              },
            );
            await runRecordedCase(fullResults, { id }, async () => {
              const observed = await observe(viaPipeline(pipeline), diagnostics, {
                prompt: textOf(pipeline.prompts, SEED_IMAGE, "prompt"),
                seed: SEED,
                shift: SHIFT,
                frames: FULL_FRAMES,
                image: sourceImageOf(fixture),
              });
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(
                observed.latents.length,
                FULL_STEPS,
                "denoise-step の数（manifest の既定）",
              );
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [FULL_FRAMES, HEIGHT, WIDTH]);
              assertEncoderStage(observed, id);
              const nonFinite = countNonFinite(video.data);
              const notes = [
                `席 ${quant}`,
                `非有限 ${nonFinite}`,
                formatFirstFrame(firstFrameDifference(video, fixture.bytes("rgb8"))),
                ...formatObserved(observed),
              ];
              console.log(`[wan-ti2v-i2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, "非有限");
              assertEquals(deviceLost, undefined, "device lost");
              await writeWanFrames(fullResults, id, video);
              const sheet = contactSheet(video, 4, 8, 4);
              await Deno.writeFile(
                fullResults.artifact(`${id}-sheet.png`),
                await encodePng(sheet.rgba, sheet.width, sheet.height),
              );
              const bytes = rgbBytes(video);
              await Deno.writeFile(fullResults.artifact(`${id}.rgb`), bytes);
              console.log(`[wan-ti2v-i2v-pipeline] PNG: ${fullResults.dir.pathname}`);
              return {
                status: "pass",
                actual: await sha256Hex(bytes),
                artifact: `${id}.rgb`,
                note: notes.join(" / "),
              };
            });
          } finally {
            // 次の席の確保の前に、この席の段の確保の解放を待つ（B570 の `destroy()` の遅れ）。
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

/**
 * 参照門に登録するケース（回せるものだけ — 全ケースが参照〈条件画像も参照から読む〉と配布形を要る）。opt-in のケースは
 * opt-in のときだけ登録する（既定のレーンでは回らないので、登録すると行が無い機の参照門を赤にする）。
 */
const CASE_IDS = DIST_PRESENT && ANY_FIXTURE
  ? [
    ...CASES.filter(({ optIn }) => !optIn || I2V_FULL).map(({ name }) => bandCaseId(name)),
    PRACTICAL_CASE_ID,
    ...(I2V_FULL ? [PORTRAIT_CASE_ID] : []),
  ]
  : [];
const RUNNABLE = DIST_PRESENT && GPU_AVAILABLE && ANY_FIXTURE;
if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });

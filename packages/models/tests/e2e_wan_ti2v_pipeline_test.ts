/**
 * Wan2.2 TI2V-5B の通し（`WanTi2vPipeline` の T2V — ADR 0121 段 6）の照合（実 GPU）。
 *
 * 配布形（`models/karume-wan2.2/` — 段 8）はまだ無いので、系列の容器から組んだ manifest と資産の Record
 * （`helpers/wan-ti2v-pipeline.ts`）を `WanTi2vPipeline.fromAssets` へ渡して組む（2.1 の段 6 と同じ入口）。資産が無い機は
 * 生成コマンドを出して明示 SKIP する。
 *
 * 比べる相手は recipe の 2 ステップの参照（`tools/export-recipes/wan/ti2v_few_step_ref.py` — 置き場は i8 の DiT の系列の根の
 * `pipeline_steps.<case>.safetensors`）: diffusers の `WanPipeline(expand_timesteps=False)` を CPU f32 で素のまま 2 ステップ
 * （CFG あり・guidance 5.0・shift 5.0）回した潜在と、それを段 5 のタイル decode → unpatchify → クランプに通したフレーム（重みは
 * DiT が i8 の RTN の fake-quant・VAE が f16 表現可能値 = 参照席 `f16+dit8` と同じ値）。形は 1280×704・17 フレーム（潜在
 * `[48,5,44,80]`・S = 4,400）で、決定用 2 本と受入れ 1 本は同じ形（受入れの形を変えて帯を外させない — 利用者の裁定 2026-10-05）。
 * 初期ノイズは参照の `latents_init`（torch の `randn`）を `latents` で注入する。
 *
 * ## 門（既定のレーン）
 *
 * この e2e が持つのは**結線**の門（プロンプトの取り違え・guidance の揺れ・shift の結線・step の順・段の受け渡し・48 値の統計の
 * 向き）であって、DiT の forward 1 本の数値の門ではない（それは段 2 の `e2e_wan_ti2v_dit_test.ts` の r 門）。
 *
 * - **ホストの自己整合**（GPU 不要）: 参照が記録した DiT の出力（cond / uncond）から、ホストの CFG + UniPC（shift 5.0）が参照の
 *   潜在を**ビット一致**で作り直す。メタの timesteps が `wanUniPcSchedule(2, 5.0)` と完全一致し、sigmas が f32 で一致する。
 *   参照の `latents_mean` / `latents_std` が `WAN22_LATENTS_*` とビット一致する（逆正規化の統計の向きの縛り）。
 * - **帯**（比 = 最大絶対差 ÷ 参照の最大絶対値 — 2.1 と同じ指標）: 観測点は各 step の後の潜在（`latents.0` / `latents.1` —
 *   `denoise-step` の `copyLatents`）とクランプ後のフレーム。帯は決定用 2 本（`band-boxing-cats` / `band-cat-dog-baking`）の
 *   観測点ごとの最悪 × 5（有効数字 2 桁へ切り上げ）で、受入れ（`accept-ferret` — seed もプロンプトも決定用と違う）が帯の内で
 *   あることを見る。帯が未導出（{@link LATENT_RATIO_BANDS} / {@link FRAME_RATIO_BAND} が `undefined`）の間は、各ケースは比を記録して
 *   赤で止まり、「帯の候補」の step が決定用の最悪 × 5 を出す（通る値を仮置きして検出力の無い門を作らない — 段 2 の前例）。
 *   MUST: 受入れの結果を見て帯もケースも変えない。受入れが帯を外れたら、帯を広げずに原因を調べる。
 * - **故障注入**（受入れの初期ノイズ・`latents.1` で比べる・床は帯の {@link FAULT_MARGIN} 倍）: プロンプトの正負の取り違え・
 *   guidance 5.0 → 5.05・shift 5.0 → 3.0（shift の結線）。どれも DiT の段の `end` のイベントで打ち切って VAE を回さない。T2V では
 *   条件マスクと条件側の時刻は原理的に値に出ない（2 本が同じ値・マスクは全て偽 — 段 2 と同じ理由）ので注入に入れない。
 * - **sha256 の環境行**（ADR 0106 — `fixtures/references/wan-ti2v.json`。2.1 の `wan.json` には触らない）: 出力フレーム（uint8 の
 *   RGB を全フレーム連結したバイト列 — `wanFrameToRgba` の規則）。参照席の 6 本 = 帯のケース 3 本・seed 経路の 1280×704 と
 *   704×1280・GPU 経路（umT5 を GPU で回す — umT5 のミラーかトークナイザ資産が無い機は明示 SKIP）。ID は席・経路・step 数・
 *   ケース・寸法・フレーム数・shift を全て持つ（{@link caseIdOf} — 裁定 F12）。行が無い機はそのケースの sha の突合を飛ばし
 *   （実物と実測の sha は残す）、参照門が赤になる（既存の規律）。行は帯が緑になった後に `KARUME_REFERENCE=write` をこのファイル
 *   単独で回して作る（裁定 F8）。
 *
 * ## 50 ステップの通し（env の opt-in `KARUME_WAN_TI2V_FULL_PIPELINE=1` — 既定のレーンに入れない）
 *
 * 事前計算の経路・`boxing-cats`・seed 42・shift 5.0・1280×704・33 フレーム・manifest の既定の 50 ステップを、参照席と実用席で
 * 1 本ずつ回し、全フレームの PNG・一覧図（4×8 マス・1/4 縮小）・RGB の実物・段ごとの所要・段の境目の VRAM の山を
 * `outputs/verify/<環境キー>/<日付>_wan-ti2v-pipeline-full/` に書き、非有限 0 と device lost が無いことを見る（利用者の視認の
 * 素材 — 生成スクリプトの同じ要求の PNG とバイトが一致することもここで見られる）。sha は**観測だけ**で行は書かない（参照席の
 * 行は 2 ステップの帯が緑になった後に別のコミットで・実用席の行は参照席との A/B の門を通った後の段 7 — 数値の正しさを確かめる前の
 * 出力を凍結しない）。結果の席を既定のレーン（`<日付>_wan-ti2v-pipeline/`）と分けるのは、`results.json` が走行ごとに丸ごと
 * 書き直されるため。
 *
 * NOTE: 実用席 `f16+dit8-a8-attn8-s16` の 2.2 の DiT は、参照席との自機 A/B の門（段 7）をまだ通っていない。opt-in の実用席の
 * 映像は、数値の正しさが未確認の席の出力（裁定 F6）。
 *
 * ## device の使い方（B570 の `destroy()` が VRAM を返さない件 — docs/known-issues.md）
 *
 * 既定のレーンの `Deno.test` 1 本は device を 1 つだけ取り、全ケースを `t.step` で回す（ケースの間に `settleReleases`）。
 * opt-in の 50 ステップも同じく device 1 つで、`--filter "50 ステップ"` で**単独のプロセス**として回す（既定のレーンの device の
 * 残りを背負わない）。資産の Record は 1 本の `Deno.test` で 1 回だけ読み、席ごとの構築で使い回す（`fromAssets` は Record を
 * パイプラインの寿命の間ずっと持つ）。
 *
 * ## 観測（門ではない）
 *
 * 段の所要（壁時計 — `stage` イベントの間）と、段の境目ごとの VRAM（`/proc/self/fdinfo` の `drm-total-*` —
 * `helpers/drm-usage.ts`）。GPU 時間は採らない — 計測の device では VAE の batch を開けない（構築が拒む）。
 *
 * 事前計算の経路の資産が無い環境は明示 SKIP。参照（`pipeline_steps.*`）が無い環境では帯・故障注入・ホストの自己整合だけを
 * 明示 SKIP する（seed 経路と GPU 経路は CPU の参照に依らないので回る）。参照が一部だけある環境は FAIL（2.1 の通しの e2e と
 * 同じ規律）。
 */

import { assert, assertEquals } from "@std/assert";
import {
  type GpuContext,
  parseSafetensors,
  type SafetensorsFile,
  type SessionDiagnostics,
} from "@karume/runtime";
import { encodePng } from "../mod.ts";
import {
  type GeneratedVideo,
  type WanAssets,
  wanFrameToRgba,
  type WanGenerateEvent,
  type WanGenerateRequest,
  type WanPrompt,
  type WanRunComponent,
  WanTi2vPipeline,
} from "../wan.ts";
import { WAN22_LATENTS_MEAN, WAN22_LATENTS_STD } from "../src/wan/latents.ts";
import {
  WAN_UNIPC_CONFIG,
  wanClassifierFreeGuidance,
  WanUniPcSampler,
  wanUniPcSchedule,
} from "../src/wan/scheduler.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import { type DrmTimeline, formatDrmUsage, monitorDrmUsage } from "./helpers/drm-usage.ts";
import { filePresent, WAN_TI2V_GENERATE, WAN_TI2V_SERIES_NAME } from "./helpers/wan-ti2v-dit.ts";
import { WAN_TI2V_VAE_GENERATE } from "./helpers/wan-ti2v-vae.ts";
import {
  missingWanTi2vSeriesAssets,
  readWanTi2vSeriesAssets,
  readWanTi2vTextEncoderAssets,
  WAN_TI2V_PRACTICAL_QUANT,
  WAN_TI2V_REFERENCE_QUANT,
  WAN_TI2V_TEXT_EMBEDS_URL,
  wanTi2vSeriesManifest,
  wanTi2vSeriesPresent,
  type WanTi2vSeriesRoute,
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
 * 2 ステップの通しの潜在（各 step の後 — `[48,5,44,80]`）の帯（比 = 最大絶対差 ÷ 参照の最大絶対値）。添字が step。
 * `undefined` = 未導出（モジュール doc「門」— 各ケースは比を記録して赤で止まり、「帯の候補」の step が候補を出す）。
 *
 * 導出の規則: 観測点ごとに決定用 2 本の最悪 × 5（有効数字 2 桁へ切り上げ）。最初の実走が出した候補を、実測の表（決定用 2 本・
 * 帯・受入れ）と一緒にここへ書く（2.1 の `e2e_wan_pipeline_test.ts` の同名の定数の doc と同じ形）。
 *
 * 導出（2026-10-05・B570・参照席）:
 *
 * | 観測点    | band-boxing-cats | band-cat-dog-baking | 最悪 × 5 | 帯     | 受入れ accept-ferret |
 * | --------- | ---------------: | ------------------: | -------: | -----: | -------------------: |
 * | latents.0 |         7.185e-6 |            3.315e-6 | 3.593e-5 | 3.6e-5 |             2.968e-6 |
 * | latents.1 |         1.035e-4 |            1.885e-4 | 9.424e-4 | 9.5e-4 |             1.199e-4 |
 * | frames    |         2.276e-3 |            1.839e-4 | 1.138e-2 | 1.2e-2 |             1.781e-3 |
 *
 * 故障注入の比（latents.1・床は帯の 2 倍 = 1.9e-3）: プロンプトの正負の取り違え 1.321・guidance 5.0 → 5.05 は 8.181e-2・
 * shift 5.0 → 3.0 は 5.512e-1。
 *
 * MUST: 受入れ（`accept-ferret`）の結果を見てこの値も、決定用のケースも変えない（帯の決定と受入れの独立が崩れる）。
 * 受入れが帯を外れたら帯を広げずに原因を調べる。
 */
const LATENT_RATIO_BANDS: readonly [number | undefined, number | undefined] = [
  3.6e-5,
  9.5e-4,
];

/**
 * 2 ステップの通しの出力フレーム（クランプ後の `[3,17,704,1280]` — 参照もクランプ後で書かれている）の帯。`undefined` = 未導出
 * （{@link LATENT_RATIO_BANDS} と同じ規則）。
 *
 * MUST: 受入れの結果を見てこの値を変えない（{@link LATENT_RATIO_BANDS} と同じ）。
 */
const FRAME_RATIO_BAND: number | undefined = 1.2e-2;

/**
 * 故障（プロンプトの正負の取り違え・guidance の 1% のずれ・shift の取り違え）が帯から離れているべき倍率の床（2.1 と段 2 と
 * 同じ 2 — 赤にならない注入は帯が広すぎる兆候）。
 */
const FAULT_MARGIN = 2;

/** 2 ステップの通しの条件（参照の `ti2v_few_step_ref.py` の `STEPS` / `GUIDANCE_SCALE` / `FLOW_SHIFT` / `LATENT_SHAPE`）。 */
const STEPS = 2;
const GUIDANCE = 5;
const SHIFT = 5;
const WIDTH = 1280;
const HEIGHT = 704;
const FRAMES = 17;
const LATENT_SHAPE: readonly number[] = [48, 5, 44, 80];

/** 参照（配布しない golden）の置き場 — i8 の DiT の系列の根（ADR 0121 段 6 の裁定 10）。 */
const STEPS_ROOT = new URL(`../../../outputs/series/${WAN_TI2V_SERIES_NAME}/`, import.meta.url);

/**
 * 参照のケース（正本は recipe `wan/ti2v_few_step_ref.py` の `FIXTURE_CASES`）。決定用（`band`）2 本 + 受入れ（`accept`）1 本。
 * MUST: 受入れの結果を見て決定用のケースを変えない。
 */
const CASES: readonly { readonly name: string; readonly role: "band" | "accept" }[] = [
  { name: "band-boxing-cats", role: "band" },
  { name: "band-cat-dog-baking", role: "band" },
  { name: "accept-ferret", role: "accept" },
];
const ACCEPT_CASE = "accept-ferret";

/**
 * sha 行のケース ID（席・経路・step 数・ケース・寸法・フレーム数・shift を全て持つ — 裁定 F12）。`head` は
 * `<席>[-gpu-text]-<step 数>-<ケース>`。
 *
 * MUST: 要求（プロンプト・seed・step 数・寸法・フレーム数・shift・席・経路）を変えたら ID も変える（行の値が別の条件の sha と
 * 突き合わさる）。寸法・フレーム数・shift はこの関数が要求と同じ値から綴る。
 */
const caseIdOf = (head: string, width: number, height: number, frames: number): string =>
  `${head}-${width}x${height}-${frames}f-shift${SHIFT}`;

/** 帯のケースの ID（参照席・事前計算の経路）。 */
const bandCaseId = (name: string): string =>
  caseIdOf(`${WAN_TI2V_REFERENCE_QUANT}-2step-${name}`, WIDTH, HEIGHT, FRAMES);

/** seed 経路の固定の要求（資産の行の原文・seed 42）。 */
const SEED_PROMPT = "boxing-cats";
const SEED = 42;

/**
 * seed 経路の 2 ステップ（`latents` を注入せず seed から初期ノイズを作る）。CPU の参照は無いので、完走・非有限 0 と sha256 の
 * 環境行だけで縛る。2 寸法（横長と縦長 — 受理集合の 2 つ）を 1 本ずつ。
 */
const SEED_CASES: readonly {
  readonly id: string;
  readonly width: number;
  readonly height: number;
}[] = [
  { width: WIDTH, height: HEIGHT },
  { width: HEIGHT, height: WIDTH },
].map(({ width, height }) => ({
  id: caseIdOf(
    `${WAN_TI2V_REFERENCE_QUANT}-2step-${SEED_PROMPT}-seed${SEED}`,
    width,
    height,
    FRAMES,
  ),
  width,
  height,
}));

/**
 * GPU 経路のケース（umT5 i8 を GPU で回す — 固定プロンプトの原文・seed 42・2 ステップ・1280×704）。CPU の参照は持たない（umT5 の
 * 数値の門は `e2e_wan_umt5_test.ts`）。資産の経路の seed のケースと同じ文字列で、経路だけが違う。
 */
const GPU_TEXT_CASE_ID = caseIdOf(
  `${WAN_TI2V_REFERENCE_QUANT}-gpu-text-2step-${SEED_PROMPT}-seed${SEED}`,
  WIDTH,
  HEIGHT,
  FRAMES,
);

/** 50 ステップの通しの opt-in（モジュール doc）。 */
const FULL_PIPELINE = Deno.env.get("KARUME_WAN_TI2V_FULL_PIPELINE") === "1";
/** 50 ステップの通しのフレーム数（既定のフレーム数）。 */
const FULL_FRAMES = 33;
/**
 * 50 ステップの通しの step 数。要求では指定せず manifest の既定に任せるので、観測した denoise-step の数で縛る（manifest の
 * 既定が変わったら ID の `50step` と食い違う）。
 */
const FULL_STEPS = 50;
/**
 * 50 ステップの通しの席。並びは確保の大きい順: S = 7,920（1280×704・33 フレーム）の DiT の diag は参照席 7.36 GiB > 実用席
 * 7.25 GiB（ADR 0121 段 2 の追記の容量の表）。前の確保の残りが後ろの大きな確保を OOM にしうる（B570 の `destroy()` の遅れ）ので
 * 大きい側を先に置く。
 */
const FULL_CASES: readonly { readonly id: string; readonly quant: string }[] = [
  WAN_TI2V_REFERENCE_QUANT,
  WAN_TI2V_PRACTICAL_QUANT,
].map((quant) => ({
  id: caseIdOf(`${quant}-50step-${SEED_PROMPT}-seed${SEED}`, WIDTH, HEIGHT, FULL_FRAMES),
  quant,
}));

const GENERATE_COMMAND =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.ti2v_few_step_ref";
const TEXT_EMBEDS_COMMAND =
  "cd tools/export-recipes && uv run --group wan --inexact python -m wan.text_embeds";
const UMT5_COMMANDS = "cd tools/export-recipes && uv run python dist.py --pipeline umt5 && " +
  "uv run --group wan --inexact python -m wan.umt5_tokenizer";

const fixtureUrl = (name: string): URL => new URL(`pipeline_steps.${name}.safetensors`, STEPS_ROOT);

/** 事前計算の経路の資産（DiT・VAE 2 本・埋め込み資産）が揃っている。 */
const SERIES_PRESENT = wanTi2vSeriesPresent("precomputed");
if (!SERIES_PRESENT) {
  console.warn(
    `[karume] Wan2.2 の通しの資産が揃っていない（${
      missingWanTi2vSeriesAssets("precomputed").join(" / ")
    }）ため通しの照合を SKIP する。生成: DiT ${WAN_TI2V_GENERATE}・VAE ${WAN_TI2V_VAE_GENERATE}・` +
      `埋め込み資産 ${TEXT_EMBEDS_COMMAND}`,
  );
}
/** GPU 経路の資産（上に加えて umT5 のミラーとトークナイザ資産）が揃っている。 */
const GPU_TEXT_PRESENT = wanTi2vSeriesPresent("gpu");
if (SERIES_PRESENT && !GPU_TEXT_PRESENT) {
  console.warn(
    `[karume] umT5 の資産が無い（${
      missingWanTi2vSeriesAssets("gpu").join(" / ")
    }）ため Wan2.2 の GPU 経路のケースを SKIP する。組み立て: ${UMT5_COMMANDS}`,
  );
}
const FIXTURES_PRESENT = CASES.map(({ name }) => filePresent(fixtureUrl(name)));
const ANY_FIXTURE = FIXTURES_PRESENT.some(Boolean);
if (!ANY_FIXTURE) {
  console.warn(
    `[karume] ${STEPS_ROOT.pathname} に 2 ステップの参照（pipeline_steps.*）が無いため Wan2.2 の通しの帯・故障注入・` +
      `ホストの自己整合を SKIP する。生成（CPU で約 3 時間・RAM の山 約 24 GiB）: ${GENERATE_COMMAND}`,
  );
}

const references = openReferences(new URL("fixtures/references/wan-ti2v.json", import.meta.url));
const results = openResults("wan-ti2v-pipeline");
/** 50 ステップの通しの結果の席（既定のレーンと分ける — モジュール doc）。 */
const fullResults = openResults("wan-ti2v-pipeline-full");

Deno.test({
  name: "Wan2.2 通しの参照: 2 ステップの参照 3 本が揃っている",
  ignore: !ANY_FIXTURE,
  fn: () => {
    assertEquals(
      FIXTURES_PRESENT,
      FIXTURES_PRESENT.map(() => true),
      `${STEPS_ROOT.pathname} の欠け`,
    );
  },
});

/** 参照 1 本（F32 のテンソル・形・メタ）。 */
type Fixture = {
  readonly tensor: (key: string) => Float32Array<ArrayBuffer>;
  readonly shape: (key: string) => readonly number[];
  readonly meta: (key: string) => string;
};

const readFixture = async (url: URL): Promise<Fixture> => {
  const bytes = await Deno.readFile(url);
  const file: SafetensorsFile = parseSafetensors(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  const view = (key: string) => {
    const found = file.tensors.get(key);
    assert(found !== undefined && found.dtype === "F32", `${url.pathname}: '${key}' が F32 で無い`);
    return found;
  };
  return {
    tensor: (key) => {
      const found = view(key);
      return new Float32Array(file.buffer, found.byteOffset, found.byteLength / 4);
    },
    shape: (key) => view(key).shape,
    meta: (key) => {
      const value = file.metadata.get(key);
      assert(value !== undefined, `${url.pathname}: メタ '${key}' が無い`);
      return value;
    },
  };
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
 * generate を 1 回まわし、イベントから観測を集める。`abortAfterDenoise` なら DiT の段の `end`（Session を畳んだ後）で投げて
 * VAE の段を飛ばす（故障注入は潜在だけを見る — 28 タイルの VAE を払わない）。
 */
const observe = async (
  pipeline: WanTi2vPipeline,
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

/**
 * 系列の資産から参照席 / 実用席の pipeline を組む。経路と席は呼び手が必ず名乗る（既定の `"gpu"` にも manifest の既定席にも
 * 黙って乗せない — 2.1 の e2e と同じ規律）。`assets` は呼び手が 1 回だけ読んだ Record（`"gpu"` は umT5 の分を足したもの）。
 */
const loadPipeline = (
  gpu: GpuContext,
  diagnostics: Map<WanRunComponent, SessionDiagnostics>,
  textEncoder: WanTi2vSeriesRoute,
  quant: string,
  assets: WanAssets["assets"],
): Promise<WanTi2vPipeline> =>
  WanTi2vPipeline.fromAssets(
    { manifest: wanTi2vSeriesManifest(textEncoder), assets },
    {
      gpu,
      textEncoder,
      quant,
      onRunDiagnostics: (component, diagnosed) => diagnostics.set(component, diagnosed),
    },
  );

/** 2 ステップの通しの要求の共通部分（寸法とプロンプトと初期ノイズはケースが足す）。 */
const TWO_STEP = { steps: STEPS, guidance: GUIDANCE, shift: SHIFT, frames: FRAMES } as const;

Deno.test({
  name:
    "Wan2.2 通し（ホスト・GPU 不要）: 参照のメタ（埋め込み資産・σ / timestep・shift 5.0）と統計 48 値が TS と一致し、" +
    "参照の DiT 出力から CFG + UniPC が参照の潜在をビット一致で再現する",
  ignore: !ANY_FIXTURE,
  fn: async () => {
    const embedsSha = await sha256Hex(await Deno.readFile(WAN_TI2V_TEXT_EMBEDS_URL));
    const schedule = wanUniPcSchedule(STEPS, SHIFT, WAN_UNIPC_CONFIG.numTrainTimesteps);
    const mean = Float32Array.from(WAN22_LATENTS_MEAN);
    const std = Float32Array.from(WAN22_LATENTS_STD);
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
        ],
        [String(STEPS), "5.0", "5.0", "false"],
        `${name}: step 数・guidance・shift・時刻の経路`,
      );
      assertEquals(fixture.meta("negative"), "negative", `${name}: negative の行名`);
      assertEquals(fixture.shape("latents_init"), LATENT_SHAPE, `${name}: 潜在の形`);
      assertEquals(fixture.shape("frames"), [3, FRAMES, HEIGHT, WIDTH], `${name}: フレームの形`);
      // 48 値の統計（逆正規化の向き）— 参照が上流の VAE の config から書いた値と TS の定数のビット一致。
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
      // 参照が記録した DiT の出力（forward の hook）から、ホストの CFG + UniPC で参照の潜在を作り直す。
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
    }
  },
});

/** 観測点（帯を持つ出力 — 各 step の後の潜在とクランプ後のフレーム）。 */
const OBSERVATION_POINTS = ["latents.0", "latents.1", "frames"] as const;
type ObservationPoint = typeof OBSERVATION_POINTS[number];

/** 観測点ごとの帯（未導出は `undefined`）。 */
const bandOf = (point: ObservationPoint): number | undefined =>
  point === "frames" ? FRAME_RATIO_BAND : LATENT_RATIO_BANDS[point === "latents.0" ? 0 : 1];

const BANDS_DERIVED = OBSERVATION_POINTS.every((point) => bandOf(point) !== undefined);

/**
 * 1 ケースの観測点ごとの比（帯の候補の材料）。非有限と device lost の検査を通った後・帯の判定の前に積む（帯の外の回は残し、
 * 非有限の要素を飛ばして出た比や device lost の後の比は候補に混ぜない）。
 */
type CaseRatios = {
  readonly name: string;
  readonly role: "band" | "accept";
  readonly ratios: Readonly<Record<ObservationPoint, number>>;
};

/**
 * 帯の候補（最悪 × 5 を有効数字 2 桁へ切り上げ — 段 2 の `bandCandidate` と同じ規則）。割り算の誤差（例 1.1e-6 / 1e-7 が
 * 11 をわずかに超える）で、ちょうど 2 桁の値を 1 単位上へ切り上げないよう、商を 12 桁へ丸めてから切り上げる。
 */
const roundUpTwoDigits = (value: number): number => {
  const unit = 10 ** (Math.floor(Math.log10(value)) - 1);
  return Number((Math.ceil(Number((value / unit).toPrecision(12))) * unit).toPrecision(2));
};

Deno.test({
  name:
    "Wan2.2 通し 2 ステップ（実 GPU）: 1280×704・17 フレームの潜在とフレームが参照と帯の内・故障注入は帯の外・" +
    "sha256 の環境行（参照席の帯のケース・seed 経路の 2 寸法・GPU 経路）",
  ignore: !SERIES_PRESENT || !GPU_AVAILABLE,
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
      // 資産の Record は 1 回だけ読む（事前計算の経路 — GPU 経路のケースは umT5 の分を足して同じ Record を使う）。
      const assets = await readWanTi2vSeriesAssets("precomputed");
      const pipeline = await loadPipeline(
        gpu,
        diagnostics,
        "precomputed",
        WAN_TI2V_REFERENCE_QUANT,
        assets,
      );
      try {
        const { prompts } = pipeline;
        const caseRatios: CaseRatios[] = [];

        for (const { name, role } of CASES) {
          const id = bandCaseId(name);
          await t.step({
            name: `${id}（${role}）`,
            ignore: !ANY_FIXTURE,
            fn: async () => {
              const fixture = await readFixture(fixtureUrl(name));
              assertEquals(fixture.meta("role"), role, `${name}: 役割`);
              // band は原文・accept は正規化後の文字列で引く（受理集合の 2 つの綴りを両方通す）。
              const prompt = textOf(
                prompts,
                fixture.meta("prompt"),
                role === "band" ? "prompt" : "normalized",
              );
              let settlement: ReferenceSettlement | undefined;
              let caseNote = "";
              try {
                await runRecordedCase(
                  results,
                  { id, failureNote: () => caseNote },
                  async ({ measurements }) => {
                    const observed = await observe(pipeline, diagnostics, {
                      ...TWO_STEP,
                      prompt,
                      latents: fixture.tensor("latents_init"),
                      width: WIDTH,
                      height: HEIGHT,
                    });
                    assert("video" in observed, "generate が最後まで回っていない");
                    assertEquals(observed.latents.length, STEPS, "denoise-step の数");
                    const { video } = observed;
                    assertEquals([video.frames, video.height, video.width], [
                      FRAMES,
                      HEIGHT,
                      WIDTH,
                    ]);
                    const diffs = OBSERVATION_POINTS.map((point) => ({
                      point,
                      band: bandOf(point),
                      // 参照のフレームはクランプ後で書かれている（`ti2v_few_step_ref.decode_frames` の clamp）。
                      diff: point === "frames"
                        ? difference(video.data, fixture.tensor("frames"))
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
                        // 未導出の帯は無限の帯として記録する（判定は下で赤にする — 段 2 の `toleranceOf` と同じ扱い）。
                        tolerance: band === undefined
                          ? { atol: Number.POSITIVE_INFINITY, rtol: 0 }
                          : { atol: band * diff.referenceMax, rtol: 0 },
                        stage: "karume",
                      });
                    }
                    const notes = [
                      ...diffs.map(({ point, diff }) => formatDifference(point, diff)),
                      ...formatObserved(observed),
                    ];
                    caseNote = notes.join(" / ");
                    console.log(`[wan-ti2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
                    for (const { point, diff } of diffs) {
                      assertEquals(diff.nonFinite, 0, `${id}: ${point} の非有限`);
                    }
                    assertEquals(deviceLost, undefined, "device lost");
                    caseRatios.push({
                      name,
                      role,
                      ratios: {
                        "latents.0": diffs[0].diff.ratio,
                        "latents.1": diffs[1].diff.ratio,
                        frames: diffs[2].diff.ratio,
                      },
                    });
                    for (const { point, band, diff } of diffs) {
                      if (band === undefined) {
                        throw new Error(
                          `${id}: ${point} の帯が未導出（比 ${diff.ratio.toExponential(3)}）— ` +
                            "帯の候補は「帯の候補」の step が出す",
                        );
                      }
                      assert(
                        diff.ratio <= band,
                        `${id}: ${formatDifference(point, diff)} が帯 ${band} の外`,
                      );
                    }
                    const outcome = await settleOrObserve(references, results, {
                      id,
                      artifact: `${id}.rgb`,
                      bytes: rgbBytes(video),
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

        // 未導出の帯の候補（決定用だけから作る — 受入れの比は並べるが候補に使わない。決定用のどれかが比を出す前に落ちた回は
        // 候補を出さない）。帯が導出済みなら step を作らない。
        if (!BANDS_DERIVED) {
          await t.step({
            name: "帯の候補（未導出）",
            ignore: !ANY_FIXTURE,
            fn: () => {
              const expected = CASES.filter(({ role }) => role === "band").length;
              const decided = caseRatios.filter(({ role }) => role === "band");
              const candidate = decided.length === expected
                ? OBSERVATION_POINTS.map((point) => {
                  const worst = Math.max(...decided.map(({ ratios }) => ratios[point]));
                  return `${point}: 決定用 ${expected} 本の最悪 ${worst.toExponential(3)} × 5 = ${
                    (worst * 5).toExponential(3)
                  }（有効数字 2 桁へ切り上げ ${roundUpTwoDigits(worst * 5)}）`;
                }).join(" / ")
                : `決定用 ${expected} 本のうち比が出たのは ${decided.length} 本 — 候補にしない（落ちたケースを先に調べる）`;
              const accepted = caseRatios.filter(({ role }) => role === "accept").map((
                { name, ratios },
              ) =>
                `${name} ${
                  OBSERVATION_POINTS.map((point) => `${point} ${ratios[point].toExponential(3)}`)
                    .join("・")
                }`
              );
              throw new Error(
                `帯が未導出 — ${candidate}。受入れの比（帯の決定に使わない）: ${
                  accepted.join(" / ") || "なし"
                }`,
              );
            },
          });
        }

        // 故障注入（受入れの初期ノイズ — 潜在だけを見るので VAE の段の前で止める）。
        let acceptFixture: Fixture | undefined;
        const faults: readonly {
          readonly label: string;
          readonly request: (
            ferret: string,
            negative: string,
          ) => Omit<WanGenerateRequest, "onEvent" | "latents">;
        }[] = [
          {
            label: "プロンプトの正負の取り違え",
            request: (ferret, negative) => ({
              ...TWO_STEP,
              prompt: negative,
              negativePrompt: ferret,
            }),
          },
          {
            label: "guidance 5.0 → 5.05",
            request: (ferret) => ({ ...TWO_STEP, prompt: ferret, guidance: 5.05 }),
          },
          {
            label: "shift 5.0 → 3.0",
            request: (ferret) => ({ ...TWO_STEP, prompt: ferret, shift: 3 }),
          },
        ];
        const lastBand = LATENT_RATIO_BANDS[STEPS - 1];
        for (const { label, request } of faults) {
          await t.step({
            name: `故障注入: ${label} → ${ACCEPT_CASE} の潜在が帯の外`,
            ignore: !ANY_FIXTURE,
            fn: async () => {
              try {
                acceptFixture ??= await readFixture(fixtureUrl(ACCEPT_CASE));
                const accept = acceptFixture;
                const ferret = textOf(prompts, accept.meta("prompt"), "prompt");
                const negative = textOf(prompts, accept.meta("negative"), "prompt");
                const observed = await observe(pipeline, diagnostics, {
                  ...request(ferret, negative),
                  latents: accept.tensor("latents_init"),
                  width: WIDTH,
                  height: HEIGHT,
                }, { abortAfterDenoise: true });
                // 段のイベント名が変わって打ち切りが効かないと、28 タイルの VAE を黙って払う（値の判定には出ない）。
                assert(!("video" in observed), "DiT の段の後で止まっていない");
                assertEquals(observed.latents.length, STEPS, "denoise-step の数");
                const diff = difference(
                  observed.latents[STEPS - 1],
                  accept.tensor(`latents.${STEPS - 1}`),
                );
                console.log(
                  `[wan-ti2v-pipeline] fault ${label}: ${
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
              } finally {
                await settleReleases(gpu);
              }
            },
          });
        }

        // seed 経路の 2 ステップ（2 寸法 — 完走・非有限 0・sha256 の環境行）。
        for (const { id, width, height } of SEED_CASES) {
          await t.step(`${id}（seed 経路・sha256 の環境行）`, async () => {
            let settlement: ReferenceSettlement | undefined;
            try {
              await runRecordedCase(results, { id }, async () => {
                const observed = await observe(pipeline, diagnostics, {
                  ...TWO_STEP,
                  prompt: textOf(prompts, SEED_PROMPT, "prompt"),
                  seed: SEED,
                  width,
                  height,
                });
                assert("video" in observed, "generate が最後まで回っていない");
                assertEquals(observed.latents.length, STEPS, "denoise-step の数");
                const { video } = observed;
                assertEquals([video.frames, video.height, video.width], [FRAMES, height, width]);
                const nonFinite = countNonFinite(video.data);
                const notes = [`非有限 ${nonFinite}`, ...formatObserved(observed)];
                console.log(`[wan-ti2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
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
            } finally {
              await settleReleases(gpu);
            }
            if (settlement?.check.status === "fail") {
              throw new Error(referenceMismatchMessage(id, settlement, references));
            }
          });
        }
      } finally {
        await pipeline.dispose();
      }

      // GPU 経路（umT5 の text 段を畳んでから DiT・VAE の段を張る）。umT5 の資産はこのケースの間だけ持つ。
      await t.step({
        name: `${GPU_TEXT_CASE_ID}（GPU 経路・sha256 の環境行）`,
        ignore: !GPU_TEXT_PRESENT,
        fn: async () => {
          let settlement: ReferenceSettlement | undefined;
          try {
            const gpuAssets = { ...assets, ...await readWanTi2vTextEncoderAssets() };
            // 構築では Session を張らない（umT5 の重みは generate ごとに text 段で読む — ADR 0119 決定 11）。
            await using gpuPipeline = await loadPipeline(
              gpu,
              diagnostics,
              "gpu",
              WAN_TI2V_REFERENCE_QUANT,
              gpuAssets,
            );
            const prompt = textOf(gpuPipeline.prompts, SEED_PROMPT, "prompt");
            await runRecordedCase(results, { id: GPU_TEXT_CASE_ID }, async () => {
              const observed = await observe(gpuPipeline, diagnostics, {
                ...TWO_STEP,
                prompt,
                seed: SEED,
                width: WIDTH,
                height: HEIGHT,
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
              assertEquals([video.frames, video.height, video.width], [FRAMES, HEIGHT, WIDTH]);
              const nonFinite = countNonFinite(video.data);
              const notes = [
                "textEncoder: gpu（umT5 i8・活性 f32）",
                `非有限 ${nonFinite}`,
                ...formatObserved(observed),
              ];
              console.log(`[wan-ti2v-pipeline] ${GPU_TEXT_CASE_ID}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, `${GPU_TEXT_CASE_ID}: 非有限`);
              assertEquals(deviceLost, undefined, "device lost");
              const outcome = await settleOrObserve(references, results, {
                id: GPU_TEXT_CASE_ID,
                artifact: `${GPU_TEXT_CASE_ID}.rgb`,
                bytes: rgbBytes(video),
              });
              settlement = outcome.settlement;
              return { ...outcome.fields, note: notes.join(" / ") };
            });
          } finally {
            await settleReleases(gpu);
          }
          if (settlement?.check.status === "fail") {
            throw new Error(referenceMismatchMessage(GPU_TEXT_CASE_ID, settlement, references));
          }
        },
      });
    } finally {
      // MUST: device を捨てる前に解放を待つ（B570 の destroy の遅れ — 後続のテストの予算を残す）。
      await settleReleases(gpu);
      gpu.destroy();
    }
  },
});

// NOTE: opt-in の env を付けて `--filter` なしで回すと、既定のレーンの `Deno.test` と合わせて 1 プロセスで device を 2 つ順に
// 取る（B570 の `destroy()` が VRAM を返さない件に当たる）。`--filter "50 ステップ"` で単独のプロセスとして回す（モジュール doc
// 「device の使い方」）。
Deno.test({
  name:
    "Wan2.2 通し 50 ステップ（実 GPU・opt-in KARUME_WAN_TI2V_FULL_PIPELINE=1）: 1280×704・33 フレーム（参照席と実用席）が" +
    "完走し非有限 0・所要と段の切り替えの VRAM・PNG 全フレーム + 一覧図・sha は観測だけ",
  ignore: !FULL_PIPELINE || !SERIES_PRESENT || !GPU_AVAILABLE,
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
      // 資産の Record は 1 回だけ読み、席ごとの構築で使い回す（構築では Session を張らない）。
      const assets = await readWanTi2vSeriesAssets("precomputed");
      for (const { id, quant } of FULL_CASES) {
        await t.step(id, async () => {
          try {
            await using pipeline = await loadPipeline(
              gpu,
              diagnostics,
              "precomputed",
              quant,
              assets,
            );
            await runRecordedCase(fullResults, { id }, async () => {
              const observed = await observe(pipeline, diagnostics, {
                prompt: textOf(pipeline.prompts, SEED_PROMPT, "prompt"),
                seed: SEED,
                shift: SHIFT,
                width: WIDTH,
                height: HEIGHT,
                frames: FULL_FRAMES,
              });
              assert("video" in observed, "generate が最後まで回っていない");
              assertEquals(
                observed.latents.length,
                FULL_STEPS,
                "denoise-step の数（manifest の既定）",
              );
              const { video } = observed;
              assertEquals([video.frames, video.height, video.width], [FULL_FRAMES, HEIGHT, WIDTH]);
              const nonFinite = countNonFinite(video.data);
              const notes = [
                `${quant}: sha は観測だけ（行は書かない — モジュール doc）`,
                `非有限 ${nonFinite}`,
                ...formatObserved(observed),
              ];
              console.log(`[wan-ti2v-pipeline] ${id}:\n  ${notes.join("\n  ")}`);
              assertEquals(nonFinite, 0, "非有限");
              assertEquals(deviceLost, undefined, "device lost");
              await writeWanFrames(fullResults, id, video);
              const sheet = contactSheet(video, 4, 8, 4);
              await Deno.writeFile(
                fullResults.artifact(`${id}-sheet.png`),
                await encodePng(sheet.rgba, sheet.width, sheet.height),
              );
              console.log(`[wan-ti2v-pipeline] PNG: ${fullResults.dir.pathname}`);
              // sha は参照値の fixture に触らずに観測だけする（`KARUME_REFERENCE=write` を付けた走行でも行を書かない）。
              const bytes = rgbBytes(video);
              const artifact = `${id}.rgb`;
              await Deno.writeFile(fullResults.artifact(artifact), bytes);
              return {
                status: "pass",
                actual: await sha256Hex(bytes),
                artifact,
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
 * 参照門に登録するケース（回せるものだけ — 帯のケースは参照と事前計算の経路の資産、seed のケースは事前計算の経路の資産、
 * GPU 経路は加えて umT5 の資産が要る）。50 ステップの通しは sha を観測だけするので登録しない。
 */
const CASE_IDS = [
  ...(SERIES_PRESENT && ANY_FIXTURE ? CASES.map(({ name }) => bandCaseId(name)) : []),
  ...(SERIES_PRESENT ? SEED_CASES.map(({ id }) => id) : []),
  ...(GPU_TEXT_PRESENT ? [GPU_TEXT_CASE_ID] : []),
];
const RUNNABLE = SERIES_PRESENT && GPU_AVAILABLE;
if (RUNNABLE) references.warnMissing(CASE_IDS);
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });

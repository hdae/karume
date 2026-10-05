/**
 * Wan2.2 TI2V-5B の通し（`WanTi2vPipeline`）を、配布形の無い間に系列の容器から組む口（ADR 0121 段 6 — 設計 §3.4）。
 *
 * 配布形（`models/karume-wan2.2/` — 段 8）はまだ無い。それまでの入口は `WanTi2vPipeline.fromAssets({ manifest, assets })`
 * で、manifest（{@link wanTi2vSeriesManifest} — 宣言だけ）と資産の Record（{@link readWanTi2vSeriesAssets} — 系列の part 列を
 * そのまま読む）をここで組む。段 8 で配布形ができたら、e2e は `fromPretrained(denoDirectory(…))` へ移る（2.1 の段 6 → 段 7 と
 * 同じ流れ）。sha 行は、配布形経由でも同じ値になることを段 8 で要求する。
 *
 * 席の定数（{@link WAN_TI2V_REFERENCE_QUANT} / {@link WAN_TI2V_PRACTICAL_QUANT} / {@link WAN_TI2V_PRACTICAL_SESSION}）も
 * ここに置き、DiT の e2e（`e2e_wan_ti2v_dit_test.ts`）と通しの e2e（`e2e_wan_ti2v_pipeline_test.ts`）が同じ値を読む。
 *
 * ## 資産の置き場（全て git 追跡外 — `docs/assets-layout.md`）
 *
 * | 部品 / 資産          | 置き場                                                              | 経路         |
 * | -------------------- | ------------------------------------------------------------------- | ------------ |
 * | `transformer`        | `outputs/series/wan2.2-ti2v-5b-i8-dyn/transformer/model.krm`（part 列） | 両方         |
 * | `vae_decoder_first`  | `outputs/series/wan2.2-ti2v-5b-f16-dyn/vae_decoder_first/model.krm`  | 両方         |
 * | `vae_decoder_next`   | `outputs/series/wan2.2-ti2v-5b-f16-dyn/vae_decoder_next/model.krm`   | 両方         |
 * | `text_embeds`        | `outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors` | 両方         |
 * | `text_encoder`       | `models/karume-umt5-xxl/xxl/text_encoder/model.i8.krm`（part 列）    | `"gpu"` だけ |
 * | `umt5_tokenizer`     | `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json`               | `"gpu"` だけ |
 *
 * 埋め込み資産は 2.1 の資産のバイトそのもの（ADR 0121 決定 9 — 同じ umT5）。
 *
 * ## ホストの RAM
 *
 * `fromAssets` は資産の Record をパイプラインの寿命の間ずっと持つ（写しは作らない — `hub/asset-readers.ts`）。
 * `"precomputed"` で約 7 GB（DiT 4.67 GiB + VAE 2 本）、`"gpu"` で umT5 i8 の 5.30 GiB が乗る。呼び手は Record を 1 回だけ
 * 読み、席ごとの構築で使い回す（席ごとに読み直すと、前の Record が回収されるまで 2 本ぶん持つ）。
 */

import { type Manifest, parseManifest, type SessionSpec } from "@karume/hub";
import type { WanAssets, WanPipelineOptions } from "../../wan.ts";
import { modelPresent, resolveParts } from "../../../runtime/tests/helpers/container-files.ts";
import { declaredContainer } from "./container-fixture.ts";
import {
  filePresent,
  WAN_TI2V_COMPONENT,
  WAN_TI2V_MODEL_FILE,
  WAN_TI2V_SERIES_DIR,
} from "./wan-ti2v-dit.ts";
import {
  WAN_TI2V_VAE_FIRST,
  WAN_TI2V_VAE_MODEL_FILE,
  WAN_TI2V_VAE_NEXT,
  WAN_TI2V_VAE_ROOT,
} from "./wan-ti2v-vae.ts";

/** テキストエンコーダの経路（`WanPipelineOptions.textEncoder` と同じ 2 値）。 */
export type WanTi2vSeriesRoute = NonNullable<WanPipelineOptions["textEncoder"]>;

/** 参照席（ADR 0121 決定 2 — 重み i8・計算 f32・`session` 空。段 7 までの開発用の仮の既定）。 */
export const WAN_TI2V_REFERENCE_QUANT = "f16+dit8";
/** 実用席（ADR 0121 決定 2 — 2.1 の実用席と同じ名前と束）。 */
export const WAN_TI2V_PRACTICAL_QUANT = "f16+dit8-a8-attn8-s16";

/**
 * 実用席の束（ADR 0121 決定 2 の表 — `linearCompute: "a8"`・`attentionCompute: "a8"`・`attentionScoreStorage: "f16"`）。
 *
 * NOTE: 束の正本は配布形の manifest の quant 席（ADR 0110 決定 1）だが、2.2 の配布形は段 8 まで無い。それまでは ADR の宣言を
 * ここに書き、家族の受理表（`WAN_SESSION_POLICY` — 決定 2 の「受理表の 3 キーは ADR 0120 と同じ」）に通す。段 8 で manifest
 * から引く形に替える。
 */
export const WAN_TI2V_PRACTICAL_SESSION: SessionSpec = {
  linearCompute: "a8",
  attentionCompute: "a8",
  attentionScoreStorage: "f16",
};

/** manifest の既定モデル（ADR 0121 — 配布形の段 8 でも同じ名前の見込み）。 */
const MODEL_NAME = "ti2v-5b";
/** manifest の pipeline（`config.ts` の `WAN_TI2V_PIPELINE_NAME` / `WAN_TI2V_PIPELINE_MAJOR`）。 */
const PIPELINE = "wan-ti2v/1";

/** 部品名（manifest の weights のキー = 資産の Record の part 列のキーの接頭辞）。 */
const TEXT_ENCODER = "text_encoder";
const TEXT_EMBEDS = "text_embeds";
const UMT5_TOKENIZER = "umt5_tokenizer";

const REPO_ROOT = new URL("../../../../", import.meta.url);

/** 容器の代表 path（part 列は {@link resolveParts} が引く）と、その格納型（manifest の weights の dtype）。 */
const CONTAINERS: readonly {
  readonly component: string;
  readonly dtype: "i8" | "f16";
  readonly representative: URL;
  readonly route: "both" | "gpu";
}[] = [
  {
    component: WAN_TI2V_COMPONENT,
    dtype: "i8",
    representative: new URL(WAN_TI2V_MODEL_FILE, WAN_TI2V_SERIES_DIR),
    route: "both",
  },
  {
    component: WAN_TI2V_VAE_FIRST,
    dtype: "f16",
    representative: new URL(`${WAN_TI2V_VAE_FIRST}/${WAN_TI2V_VAE_MODEL_FILE}`, WAN_TI2V_VAE_ROOT),
    route: "both",
  },
  {
    component: WAN_TI2V_VAE_NEXT,
    dtype: "f16",
    representative: new URL(`${WAN_TI2V_VAE_NEXT}/${WAN_TI2V_VAE_MODEL_FILE}`, WAN_TI2V_VAE_ROOT),
    route: "both",
  },
  {
    component: TEXT_ENCODER,
    dtype: "i8",
    representative: new URL("models/karume-umt5-xxl/xxl/text_encoder/model.i8.krm", REPO_ROOT),
    route: "gpu",
  },
];

/**
 * 埋め込み資産（2.1 と同じバイト — 決定 9）。通しの参照（`pipeline_steps.*`）のメタ `text_embeds_sha256` と突き合わせる
 * テストもこの置き場を読む。
 */
export const WAN_TI2V_TEXT_EMBEDS_URL = new URL(
  "outputs/series/wan2.1-t2v-1.3b-text-embeds/text_embeds.safetensors",
  REPO_ROOT,
);

/** manifest の assets のキー → ファイル（{@link CONTAINERS} と同じく経路で絞る）。 */
const FILE_ASSETS: readonly {
  readonly key: string;
  readonly file: URL;
  readonly route: "both" | "gpu";
}[] = [
  { key: TEXT_EMBEDS, file: WAN_TI2V_TEXT_EMBEDS_URL, route: "both" },
  {
    key: UMT5_TOKENIZER,
    file: new URL("outputs/series/wan2.1-umt5-tokenizer/tokenizer.json", REPO_ROOT),
    route: "gpu",
  },
];

const onRoute = (route: WanTi2vSeriesRoute) => (entry: { readonly route: "both" | "gpu" }) =>
  entry.route === "both" || route === "gpu";

/**
 * 系列の通しを組む manifest（karume/5 — **宣言だけ**）。`fromAssets` は part の sha256 も descriptor の値も資産の中身と
 * 突き合わせない（`hub/components.ts` の全量面）ので、weights の 3 点セットと assets の size / sha256 は綴りだけ合わせた
 * 値（`declaredContainer` と同じ扱い）。中身の正しさは容器の block ごとの sha256（`verified: false` の経路）と家族 admission が見る。
 *
 * - 席: 参照席と実用席の 2 つ（`session` は 2.1 の配布形と同じ 3 キー）・既定は参照席（ADR 0121 決定 2「段 7 までの開発用は
 *   参照席を仮の既定」）
 * - `pipelineConfig`: shift 5.0（上流の scheduler の `flow_shift` = 公式の 720p の値 — 利用者の裁定 2026-10-05）・50 ステップ・
 *   guidance 5.0
 * - `requiredLimits` は宣言しない（配布の時点で `dist.py` が導く — 段 8）
 *
 * `"gpu"` の経路だけ umT5 の部品（`text_encoder`）とトークナイザ資産（`umt5_tokenizer`）を宣言する。
 */
export const wanTi2vSeriesManifest = (route: WanTi2vSeriesRoute): Manifest => {
  const containers = CONTAINERS.filter(onRoute(route));
  const seatWeights = Object.fromEntries(
    containers.map(({ component, dtype }) => [component, dtype]),
  );
  return parseManifest(JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: MODEL_NAME,
    models: {
      [MODEL_NAME]: {
        pipeline: PIPELINE,
        weights: Object.fromEntries(
          containers.map(({ component, dtype }) => [
            component,
            { [dtype]: declaredContainer(`${component}/model.${dtype}`) },
          ]),
        ),
        assets: Object.fromEntries(
          FILE_ASSETS.filter(onRoute(route)).map(({ key, file }) => [
            key,
            { path: file.pathname.split("/").pop(), size: 8, sha256: "e".repeat(64) },
          ]),
        ),
        quants: {
          [WAN_TI2V_REFERENCE_QUANT]: { weights: seatWeights, session: {} },
          [WAN_TI2V_PRACTICAL_QUANT]: { weights: seatWeights, session: WAN_TI2V_PRACTICAL_SESSION },
        },
        defaultQuant: WAN_TI2V_REFERENCE_QUANT,
        pipelineConfig: { scheduler: { shift: 5 }, defaults: { steps: 50, guidance: 5 } },
      },
    },
  }));
};

/** 容器の part 列を Record のキー（`<部品>[<添字>]` — part 0 から添字順）で読む。 */
const readContainerParts = async (
  component: string,
  representative: URL,
): Promise<Record<string, Uint8Array<ArrayBuffer>>> => {
  const parts = resolveParts(representative);
  const entries: [string, Uint8Array<ArrayBuffer>][] = [];
  for (const [index, part] of parts.entries()) {
    entries.push([`${component}[${index}]`, await Deno.readFile(part)]);
  }
  return Object.fromEntries(entries);
};

/** 表の行のうち `pick` が選んだ経路の容器と資産を読む。 */
const readSelected = async (
  pick: (route: "both" | "gpu") => boolean,
): Promise<WanAssets["assets"]> => {
  const assets: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const { component, representative, route } of CONTAINERS) {
    if (pick(route)) Object.assign(assets, await readContainerParts(component, representative));
  }
  for (const { key, file, route } of FILE_ASSETS) {
    if (pick(route)) assets[key] = await Deno.readFile(file);
  }
  return assets;
};

/**
 * 系列の資産の Record（`WanAssets["assets"]`）を読む。`"precomputed"` は DiT・VAE 2 本・埋め込み資産、`"gpu"` は加えて
 * umT5 の part 列とトークナイザ資産（{@link readWanTi2vTextEncoderAssets} の分）。
 *
 * MUST: 1 回だけ読んで使い回す（モジュール doc「ホストの RAM」）。無いファイルは `Deno.readFile` の NotFound で落ちる
 * （SKIP の判定は {@link wanTi2vSeriesPresent} が先に持つ — ここで黙って欠けた Record を返さない）。
 */
export const readWanTi2vSeriesAssets = (route: WanTi2vSeriesRoute): Promise<WanAssets["assets"]> =>
  readSelected((entry) => entry === "both" || route === "gpu");

/**
 * `"gpu"` の経路だけが要る資産（umT5 の part 列とトークナイザ資産）。`"precomputed"` の Record を読んだ呼び手が、GPU 経路の
 * ケースの前に足す口（DiT と VAE を読み直さない — umT5 だけで 5.30 GiB あるので、要るケースの間だけ持つ）。
 */
export const readWanTi2vTextEncoderAssets = (): Promise<WanAssets["assets"]> =>
  readSelected((entry) => entry === "gpu");

/**
 * その経路の資産が揃っているか（同期 — モジュールの先頭の SKIP の判定用）。`"gpu"` は `"precomputed"` の資産を含む。
 * 容器は part 列の解決まで見る（欠番・単一形との同居は {@link resolveParts} が fail loudly）。
 */
export const wanTi2vSeriesPresent = (route: WanTi2vSeriesRoute): boolean =>
  CONTAINERS.filter(onRoute(route)).every(({ representative }) => modelPresent(representative)) &&
  FILE_ASSETS.filter(onRoute(route)).every(({ file }) => filePresent(file));

/** 経路ごとに欠けている資産の置き場（SKIP の警告に出す — 何を作れば回るかを言う）。 */
export const missingWanTi2vSeriesAssets = (route: WanTi2vSeriesRoute): readonly string[] => [
  ...CONTAINERS.filter(onRoute(route))
    .filter(({ representative }) => !modelPresent(representative))
    .map(({ representative }) => representative.pathname),
  ...FILE_ASSETS.filter(onRoute(route))
    .filter(({ file }) => !filePresent(file))
    .map(({ file }) => file.pathname),
];

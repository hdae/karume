/**
 * Wan の VAE の段 — chunk グラフ 2 本（first / next）の**常時タイル** decode（ADR 0118 決定 2 —
 * `vae-tiles.ts`）と、受理する寸法が全部タイルで覆えるかの家族 admission の門
 * （{@link assertWanVaeTilesCover}）。
 *
 * 段の順序と Session の寿命（text → DiT → VAE を 1 段ずつ張って畳む）は `./pipeline.ts` 冒頭の doc が
 * 正本で、ここは VAE の段 1 本を回すだけ。Wan2.1 / 2.2 の class が共有する（ADR 0121 決定 10）。
 * 世代ごとの値（受理集合・逆正規化の統計）は世代の記述子（`./descriptor.ts`）から引数で受ける。
 * `owner` は文言の接頭辞（Wan2.1 は `"WanPipeline"`）。
 *
 * NOTE: 公開型（`WanGenerateEvent` / `WanRunComponent`）は `./pipeline.ts` から `import type` で取る
 * （型だけの参照は消去されるので循環 import にならない）。
 *
 * MUST: 全モジュール副作用ゼロ（import 時実行・グローバル可変状態の禁止 — CLAUDE.md）。
 */

import type { GpuContext, Session, SessionDiagnostics } from "@karume/runtime";

import { settleAbort } from "../concurrency/abort.ts";
import { disposeSteps } from "../session/dispose-steps.ts";
import type { ModelComponent } from "../hub/components.ts";
import type { WanGenerateEvent, WanRunComponent } from "./pipeline.ts";
import type { PlanLayout, WanGenerationKnobs } from "./plan.ts";
import type { WanGenerationDescriptor } from "./descriptor.ts";
import { firstNonFinite } from "./graph-io.ts";
import { denormalizeWanLatents } from "./latents.ts";
import { WanVaeChunkCaches, type WanVaeChunkLayout } from "./vae-chunks.ts";
import {
  clampWanVaeFrames,
  decodeWanVaeTiled,
  planWanVaeTiles,
  wanVaeTileCount,
} from "./vae-tiles.ts";

/** 部品のキー（系列のグラフ名 = 段 7 の manifest の weights 名）。 */
export const VAE_DECODER_FIRST = "vae_decoder_first";
export const VAE_DECODER_NEXT = "vae_decoder_next";

/**
 * VAE の chunk グラフの幾何で、受理する寸法（`generation.acceptedSizes`）が全部タイルで覆えることを
 * 見る（家族 admission の門 — 寸法は有限なので全数を計画する）。
 *
 * MUST: admission で呼ぶ。タイル辺は配布物だけで差し替えられる（ADR 0118 決定 2）ので、潜在の短辺
 * より大きいタイル（例 64 > 60）や重なりの下限を満たせないタイル（例 8）の資産も chunk グラフの
 * 検査（`wanVaeChunkLayout`）は通る。ここで落とさないと、DiT の段を全部払った後の VAE の段で
 * 初めて落ちる。
 */
export const assertWanVaeTilesCover = (
  layout: PlanLayout,
  generation: WanGenerationDescriptor,
  owner: string,
): void => {
  const scale = layout.sampleTile / layout.tile;
  for (const { width, height } of generation.acceptedSizes) {
    const latentHeight = height / scale;
    const latentWidth = width / scale;
    if (!Number.isInteger(latentHeight) || !Number.isInteger(latentWidth)) {
      throw new Error(
        `${owner}: VAE の縮尺 ${layout.sampleTile} / ${layout.tile} では ${width}×${height} の潜在が` +
          "整数にならない",
      );
    }
    try {
      planWanVaeTiles(layout, latentHeight, latentWidth);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(
        `${owner}: VAE のタイル（潜在 ${layout.tile}）では ${width}×${height}` +
          `（潜在 ${latentHeight}×${latentWidth}）をタイル decode できない — ${reason}`,
        { cause: error },
      );
    }
  }
};

/** VAE の段が読む構築済みの材料（パイプラインの内部状態のうちこの段の分 — 構造で受ける）。 */
type WanVaeDecodeState = {
  readonly gpu: GpuContext;
  readonly vaeFirst: ModelComponent;
  readonly vaeNext: ModelComponent;
  readonly layout: WanVaeChunkLayout;
  readonly onRunDiagnostics?: (
    component: WanRunComponent,
    diagnostics: SessionDiagnostics,
  ) => void;
};

/**
 * VAE の段（chunk グラフ 2 本の Session と常駐の cache を張り、タイル decode して畳む）。
 *
 * MUST: クランプの**前**に有限性を見る。上流と同じクランプ（{@link clampWanVaeFrames}）は ±Inf を
 * ±1 に変えるので、後では検出できない（クランプ自体は上流の写しなので変えない）。
 *
 * NOTE: 出力の検査は要素数だけで、VAE の unpatchify の有無は見分けない（patchify 空間の要素数
 * `3·p²·(H/p)·(W/p)` は `3·H·W` と等しい）。unpatchify を入れる段 5 で、形で見る検査に変える。
 */
export const decodeWanVaeStage = async (
  state: WanVaeDecodeState,
  plan: WanGenerationKnobs,
  latents: Float32Array,
  generation: WanGenerationDescriptor,
  emit: (event: WanGenerateEvent) => Promise<void>,
  signal: AbortSignal | undefined,
  owner: string,
): Promise<Float32Array<ArrayBuffer>> => {
  const { layout } = state;
  const tilePlan = plan.tiles;
  const tiles = wanVaeTileCount(tilePlan);
  const denormalized = denormalizeWanLatents(latents, generation.latents);
  const observe = state.onRunDiagnostics;

  await emit({ kind: "stage", component: "vae_decoder", at: "start" });
  let first: Session | undefined;
  let next: Session | undefined;
  let caches: WanVaeChunkCaches | undefined;
  let frames: Float32Array<ArrayBuffer>;
  let failure: { readonly error: unknown } | undefined;
  try {
    // VAE は quant の session を受けない（`{}` — anima の `withStage` と同じ取り決め。
    // `sessionOptions` は DiT の Session へ渡す実効設定 — {@link WanAdmission.sessionOptions}）。
    first = await state.vaeFirst.createSession(state.gpu, {});
    next = await state.vaeNext.createSession(state.gpu, {});
    caches = await WanVaeChunkCaches.create(state.gpu, layout);
    const sessions = { first, next };
    frames = await decodeWanVaeTiled(
      state.gpu,
      sessions,
      caches,
      tilePlan,
      denormalized,
      async (tile) => {
        observe?.("vae_decoder_first", sessions.first.diagnostics());
        observe?.("vae_decoder_next", sessions.next.diagnostics());
        await emit({ kind: "vae-tile", tile, tiles });
        // タイルの間（1 タイル = 1 batch で不可分 — 中断は次のタイルの前で効く）。
        await settleAbort(signal);
      },
    );
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    // MUST: Session → 常駐の順で畳む（段 4 / 5 の e2e と同じ）。畳む失敗で本体の失敗を上書きしない。
    const opened = { first, next, caches };
    await disposeSteps([
      () => {
        if (failure !== undefined) throw failure.error;
      },
      () => opened.first?.dispose(),
      () => opened.next?.dispose(),
      () => opened.caches?.dispose(),
    ]);
  }
  const expected = 3 * plan.frames * plan.height * plan.width;
  if (frames.length !== expected) {
    throw new Error(
      `${owner}: VAE の出力 ${frames.length} 要素が [3, ${plan.frames}, ${plan.height}, ${plan.width}] と違う`,
    );
  }
  const broken = firstNonFinite(frames);
  if (broken !== -1) {
    const plane = plan.height * plan.width;
    const pixel = broken % plane;
    const sheet = Math.floor(broken / plane);
    throw new Error(
      `${owner}: VAE の出力（クランプ前）の channel ${Math.floor(sheet / plan.frames)}・` +
        `フレーム ${sheet % plan.frames}・画素 (x=${pixel % plan.width}, ` +
        `y=${Math.floor(pixel / plan.width)}) が非有限（${frames[broken]}）`,
    );
  }
  clampWanVaeFrames(frames);
  await emit({ kind: "stage", component: "vae_decoder", at: "end" });
  return frames;
};

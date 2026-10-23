/**
 * Wan2.1 / Wan2.2 の受理集合 × 融合 attention の経路（runtime の `attentionRoute` — 実行と同じ判定）。
 * GPU も資産も要らない。
 *
 * 実用席（`f16+dit8-a8-attn8-s16` — tools/export-recipes/wan/distribution.py）は DiT の Session に a8 + s16 を
 * 宣言する。受理する全形（寸法 × フレーム数）で self / cross とも宣言どおりの経路になることを断言する —
 * 将来、self のキー長が 4 の倍数でない寸法を受理に足すと、ここが赤になる（ADR 0058 決定 3 の縮退を
 * 黙って受理集合へ持ち込まない）。キー長は本番の入口（`planWanRequest` の潜在の形 → `wanTokenGrid`）から
 * 導く — 式をここに二重に書かない。
 */

import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  type AttentionDeviceFacts,
  type AttentionRoute,
  attentionRoute,
  type AttentionSeats,
} from "../../runtime/src/runtime/attention-route.ts";
import type { WanPipelineConfig } from "../src/wan/config.ts";
import {
  WAN21_GENERATION,
  WAN22_TI2V_GENERATION,
  type WanGenerationDescriptor,
} from "../src/wan/descriptor.ts";
import { wanDitPatch } from "../src/wan/dit-loop.ts";
import { wanTokenGrid } from "../src/wan/dit-tokens.ts";
import {
  type PlanLayout,
  planWanRequest,
  type PromptGate,
  TEMPORAL_COMPRESSION,
} from "../src/wan/plan.ts";

/** 実用席の attention の宣言（a8 + s16）。 */
const PRACTICAL_SEATS: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f16" };
/** 宣言どおりの経路。 */
const REQUESTED: AttentionRoute = { qk: "i8a8", pv: "i8a8", score: "f16" };
/** Device facts: the a8 + s16 seats never read them (only `attentionScoreStorage: "none"` does). */
const DEVICE: AttentionDeviceFacts = { subgroups: true, maxComputeWorkgroupStorageSize: 32768 };

/**
 * 配布形の DiT のグラフ宣言の写し（両世代で同じ）: `rope_cos [1,S,1,128]` の head_dim 128 = D・
 * `encoder_hidden_states [1,512,4096]` の 512 = cross のキー長。
 */
const HEAD_DIM = 128;
const CROSS_KEYS = 512;

/** 世代ごとの配布形の VAE の chunk グラフの幾何（`latent` の宣言 `[C,1,tile,tile]` と出口）の写し。 */
const GENERATIONS: readonly {
  readonly name: string;
  readonly generation: WanGenerationDescriptor;
  readonly layout: PlanLayout;
  /** 既定の 33 フレームの self のキー長（ADR 0118 / 0121 の census の記録と同じ値 — 導出の取り違えの検出器）。 */
  readonly defaultKeys: number;
}[] = [
  {
    name: "Wan2.1",
    generation: WAN21_GENERATION,
    layout: { latentChannels: 16, tile: 32, sampleTile: 256, sampleChannels: 3 },
    defaultKeys: 14040,
  },
  {
    name: "Wan2.2",
    generation: WAN22_TI2V_GENERATION,
    layout: { latentChannels: 48, tile: 16, sampleTile: 128, sampleChannels: 12 },
    defaultKeys: 7920,
  },
];

/** 計画の門に渡す最小の設定とプロンプトの門（キー長はプロンプトと step に依らない）。 */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };
const GATE: PromptGate<string> = { resolve: (text) => text, defaultNegative: () => "negative" };

/** 受理する全形の self のキー長（本番の入口 `planWanRequest` を通す — 受理外なら門が投げる）。 */
const acceptedSelfKeys = (
  generation: WanGenerationDescriptor,
  layout: PlanLayout,
): readonly { readonly shape: string; readonly keys: number }[] =>
  generation.acceptedSizes.flatMap(({ width, height }) => {
    const shapes = [];
    for (
      let frames = generation.minFrames;
      frames <= generation.maxFrames;
      frames += TEMPORAL_COMPRESSION
    ) {
      const plan = planWanRequest(
        { prompt: "p", width, height, frames },
        GATE,
        layout,
        CONFIG,
        generation,
      );
      const keys = wanTokenGrid(plan.latentShape, wanDitPatch(layout.latentChannels)).count;
      shapes.push({ shape: `${width}×${height}×${frames}`, keys });
    }
    return shapes;
  });

describe("Wan の受理集合 × 実用席の attention の経路", () => {
  for (const { name, generation, layout, defaultKeys } of GENERATIONS) {
    describe(name, () => {
      it("既定の形（33 フレーム）の self のキー長が記録の値と一致する", () => {
        const { width, height, frames } = generation.defaults;
        const plan = planWanRequest({ prompt: "p" }, GATE, layout, CONFIG, generation);
        assertEquals([plan.width, plan.height, plan.frames], [width, height, frames]);
        assertEquals(
          wanTokenGrid(plan.latentShape, wanDitPatch(layout.latentChannels)).count,
          defaultKeys,
        );
      });

      it("受理する全形で self は宣言どおり（③PV も S の f16 格納も落ちない）", () => {
        const shapes = acceptedSelfKeys(generation, layout);
        const frameCount = (generation.maxFrames - generation.minFrames) / TEMPORAL_COMPRESSION + 1;
        assertEquals(shapes.length, generation.acceptedSizes.length * frameCount);
        for (const { shape, keys } of shapes) {
          assertEquals(
            attentionRoute(PRACTICAL_SEATS, HEAD_DIM, keys, DEVICE),
            REQUESTED,
            `${shape}（S=${keys}）`,
          );
        }
      });

      it("cross（キー長 512）は宣言どおり", () => {
        assertEquals(attentionRoute(PRACTICAL_SEATS, HEAD_DIM, CROSS_KEYS, DEVICE), REQUESTED);
      });
    });
  }
});

/** The online-attention preset's attention seats (`f16+dit8-a8-attn8-online` — ADR 0132 decision 6). */
const ONLINE_SEATS: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "none" };

/**
 * The online seats over Wan2.2's whole accepted set (Wan2.1 has no online preset — ADR 0132 decision 6).
 * The route depends on the device facts (ADR 0132 decision 5): `sg32` with subgroups, `wg` without them when the
 * workgroup-storage limit holds the portable form's scratch (D = 128: 33,536 B), and otherwise the three dispatches
 * of a8 with S in f32 — never `"none"`, which the three kernels cannot store.
 */
const ONLINE_DEVICES: readonly {
  readonly name: string;
  readonly device: AttentionDeviceFacts;
  readonly route: AttentionRoute;
}[] = [
  {
    name: "subgroups, 32 KiB of workgroup storage",
    device: { subgroups: true, maxComputeWorkgroupStorageSize: 32768 },
    route: { qk: "i8a8", pv: "i8a8", score: "none", online: "sg32" },
  },
  {
    name: "no subgroups, 48 KiB of workgroup storage",
    device: { subgroups: false, maxComputeWorkgroupStorageSize: 49152 },
    route: { qk: "i8a8", pv: "i8a8", score: "none", online: "wg" },
  },
  {
    name: "no subgroups, 32 KiB of workgroup storage (falls back)",
    device: { subgroups: false, maxComputeWorkgroupStorageSize: 32768 },
    route: { qk: "i8a8", pv: "i8a8", score: "f32" },
  },
];

describe("Wan2.2's accepted set × the online preset's attention route", () => {
  const wan22 = GENERATIONS.find(({ generation }) => generation === WAN22_TI2V_GENERATION);
  if (wan22 === undefined) throw new Error("GENERATIONS has no Wan2.2 entry");
  const { generation, layout } = wan22;
  for (const { name, device, route } of ONLINE_DEVICES) {
    describe(name, () => {
      it("self takes the same route over every accepted shape", () => {
        const shapes = acceptedSelfKeys(generation, layout);
        const frameCount = (generation.maxFrames - generation.minFrames) / TEMPORAL_COMPRESSION + 1;
        assertEquals(shapes.length, generation.acceptedSizes.length * frameCount);
        for (const { shape, keys } of shapes) {
          assertEquals(
            attentionRoute(ONLINE_SEATS, HEAD_DIM, keys, device),
            route,
            `${shape} (S=${keys})`,
          );
        }
      });

      it("cross (512 keys) takes the same route", () => {
        assertEquals(attentionRoute(ONLINE_SEATS, HEAD_DIM, CROSS_KEYS, device), route);
      });
    });
  }
});

/**
 * The accepted set of Wan2.1 / Wan2.2 × the fused attention route (the runtime's `attentionRoute` —
 * the same decision as at execution). Needs neither a GPU nor assets.
 *
 * The practical preset (`f16+dit8-a8-attn8-s16` — tools/export-recipes/wan/distribution.py) declares
 * a8 + s16 on the DiT Session. This asserts that for every accepted shape (size × frame count) both
 * self and cross take the declared route — if a size whose self key length is not a multiple of 4 is
 * ever added to the accepted set, this turns red (the degradation of ADR 0058 decision 3 is not
 * brought into the accepted set silently). The key length is derived from the production entry (the
 * latent shape of `planWanRequest` → `wanTokenGrid`) — the formula is not written a second time here.
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

/** The attention declaration of the practical preset (a8 + s16). */
const PRACTICAL_SEATS: AttentionSeats = { attentionCompute: "a8", attentionScoreStorage: "f16" };
/** The declared route. */
const REQUESTED: AttentionRoute = { qk: "i8a8", pv: "i8a8", score: "f16" };
/** Device facts: the a8 + s16 seats never read them (only `attentionScoreStorage: "none"` does). */
const DEVICE: AttentionDeviceFacts = { subgroups: true, maxComputeWorkgroupStorageSize: 32768 };

/**
 * A copy of the distribution's DiT graph declaration (the same in both generations): head_dim 128 of
 * `rope_cos [1,S,1,128]` = D, and 512 of `encoder_hidden_states [1,512,4096]` = the cross key length.
 */
const HEAD_DIM = 128;
const CROSS_KEYS = 512;

/** A copy of the geometry of each generation's distributed VAE chunk graph (the `latent` declaration `[C,1,tile,tile]` and the output). */
const GENERATIONS: readonly {
  readonly name: string;
  readonly generation: WanGenerationDescriptor;
  readonly layout: PlanLayout;
  /** The self key length at the default 33 frames (the same value as recorded by the census of ADR 0118 / 0121 — a detector for a wrong derivation). */
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

/** The smallest config passed to the plan check, and the prompt gate (the key length depends on neither the prompt nor the steps). */
const CONFIG: WanPipelineConfig = { scheduler: { shift: 3 }, defaults: { steps: 50, guidance: 5 } };
const GATE: PromptGate<string> = { resolve: (text) => text, defaultNegative: () => "negative" };

/** The self key length of every accepted shape (through the production entry `planWanRequest` — outside the accepted set its check throws). */
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

describe("the accepted set of Wan × the attention route of the practical preset", () => {
  for (const { name, generation, layout, defaultKeys } of GENERATIONS) {
    describe(name, () => {
      it("the self key length of the default shape (33 frames) matches the recorded value", () => {
        const { width, height, frames } = generation.defaults;
        const plan = planWanRequest({ prompt: "p" }, GATE, layout, CONFIG, generation);
        assertEquals([plan.width, plan.height, plan.frames], [width, height, frames]);
        assertEquals(
          wanTokenGrid(plan.latentShape, wanDitPatch(layout.latentChannels)).count,
          defaultKeys,
        );
      });

      it("self takes the declared route for every accepted shape (neither ③PV nor the f16 storage of S falls back)", () => {
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

      it("cross (key length 512) takes the declared route", () => {
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
 * workgroup-storage limit holds the portable form's scratch (D = 128: 35,328 B), and otherwise the three dispatches
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
    route: { qk: "i8a8", pv: "i8a8", score: "none", online: "sg32", keyWords: "whole" },
  },
  {
    name: "no subgroups, 48 KiB of workgroup storage",
    device: { subgroups: false, maxComputeWorkgroupStorageSize: 49152 },
    route: { qk: "i8a8", pv: "i8a8", score: "none", online: "wg", keyWords: "whole" },
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

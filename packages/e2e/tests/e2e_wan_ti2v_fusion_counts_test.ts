/**
 * Fusion hit counts of the Wan2.2 TI2V-5B distribution graphs (no GPU — `planFusions` on part 0
 * only; the shared planning side and the rationale live in `helpers/fusion-counts.ts`).
 *
 * The last case pins the hits of the candidate rules `candidate/vae_rms_silu_g8` / `_g4`, which
 * no preset passes (`@karume/candidate-rules`).
 *
 * MUST: the asset is `models/karume-wan2.2/` (untracked local asset); where it is missing the case
 * SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { vaeRmsSiluRuleName } from "@karume/candidate-rules";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { exists, fusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";
import { vaeRmsSiluPlans } from "./helpers/vae-rms-silu-plan.ts";

const WAN22_DIR = new URL("../../../models/karume-wan2.2/", import.meta.url);
const WAN22_AVAILABLE = await exists(new URL("karume.json", WAN22_DIR));
if (!WAN22_AVAILABLE) {
  console.warn(`[karume] SKIP the Wan2.2 fusion hit counts: ${WAN22_DIR.pathname} is missing`);
}

/**
 * Only the DiT and the VAE graphs are read: the text encoder is a cross-repo reference to the umT5
 * distribution, and this gate knows no mirror for it (a cross-repo reference fails loudly instead
 * of being read from somewhere else).
 */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/** A component graph of the default model (`ti2v-5b`) at one seat. */
const readWan22Graph = async (component: string, dtype: string): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", WAN22_DIR)),
  );
  return await readDistributionGraph(
    WAN22_DIR,
    manifest,
    manifest.defaultModel,
    component,
    dtype,
    NO_MIRRORS,
  );
};

/**
 * The input shapes with every symbolic dimension solved to `extent` (the DiT's `S`; the VAE
 * encoder's `8h` / `8w` / `h` / `w` / `S` — each graph is bound on its own, so one value per graph
 * is consistent).
 */
const solvedShapes = (
  graph: IrGraph,
  extent: number,
): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : extent)),
    ]),
  );

/**
 * Device facts that hold the online kernel at the DiT's D 128 (subgroups and 48 KiB of workgroup
 * storage — the sg32 form needs 31,232 B).
 */
const ONLINE_DEVICE = { subgroups: true, maxComputeWorkgroupStorageSize: 48 * 1024 } as const;

/**
 * The Wan2.2 DiT (30 blocks + the head), measured 2026-10-24:
 *
 * - **rmsNormRope 60** (counted 2026-10-25): the q and k prologues of the 30 self-attention blocks,
 *   `rms_norm` over the 3,072-wide token row → two reshapes → the pair-form RoPE, each folded into
 *   one dispatch (perf-ledger F-3 stage 3). So **rope 0**: every pair-form RoPE of this DiT (60 —
 *   the `rope` count before 2026-10-25) now folds with its rms_norm; the rms_norm takes the 256-wide
 *   workgroup kernel here (`rmsNormReduce` is not set), and under `"subgroup32"` the RoPEs would
 *   fold alone again (rope 60).
 * - **adaln 61**: the per-token adaLN (`where` picks the condition / generation vector per token)
 *   before the self-attention and before the FFN of the 30 blocks, plus the head's.
 * - **gatedResidual 60**: the per-token gated residual after the self-attention output projection
 *   and after the FFN of the 30 blocks (the cross-attention residual has no gate).
 * - **silu 4**: the time-embedding MLP's two SiLUs, run once per timestep stream (generation and
 *   condition — ADR 0121 decision 3).
 * - **attentionPermute 180, under `attentionCompute: "a8"` only** (counted 2026-10-25): each of
 *   the 60 attentions (30 self, 30 cross) absorbs the `[0, 2, 1, 3]` permutes of q, k and v right
 *   before it (perf-ledger F-3 A4 / F-8). On the seats that store S (s16, and S in f32) the permute
 *   of O after it stays a node. The f32 attention seat absorbs nothing.
 * - **attentionPermute 240 on the online seat** (`attentionScoreStorage: "none"` on a device that
 *   holds the online kernel; counted 2026-10-26): every attention takes the online route (D 128,
 *   and its key count — S for self-attention, the text tokens for cross-attention — is a multiple of
 *   4 in every case checked here), so it also absorbs the permute of O (perf-ledger F-3 A4).
 * - **linearPrologue 31, under `linearCompute: "a8"` only** (counted 2026-10-25): the gelu_tanh
 *   folded into the step ① of the i8 linear that reads it (perf-ledger F-4) — the FFN down
 *   projection of the 30 blocks (K 14,336) and the text embedder's second linear (K 3,072). The
 *   default f32 linear has no step ① to fold into.
 *
 * The counts do not depend on S (S = 192 and the 1280×704×121 size 7,920 are both checked). If a
 * count moves, the exporter's node order or a rule's accepted set changed: the values stay correct
 * (an unmatched window runs as plain nodes) and only the speed changes, so this is the only place
 * that notices.
 */
test.skipIf(!WAN22_AVAILABLE)(
  "the real Wan2.2 DiT grabs rmsNormRope 60 / adaln 61 / gatedResidual 60 / silu 4 per run (plus linearPrologue 31 under a8, and attentionPermute 180 on the a8 attention seats that store S, 240 on the online seat), and the VAE chunk graphs keep their counts",
  async () => {
    const transformer = await readWan22Graph("transformer", "i8");
    for (const sequence of [192, 7920]) {
      const shapes = solvedShapes(transformer, sequence);
      const counts = { ...NONE, silu: 4, rmsNormRope: 60, adaln: 61, gatedResidual: 60 };
      assertEquals(fusionCounts(transformer, shapes), counts, `transformer S=${sequence}`);
      assertEquals(
        fusionCounts(transformer, shapes, { linearCompute: "a8" }),
        { ...counts, linearPrologue: 31 },
        `transformer S=${sequence} a8`,
      );
      // The attn8 presets (s16 and online) and the a8 seat with S in f32: linearCompute and
      // attentionCompute "a8".
      for (const [seat, storage, absorbed] of [
        ["S in f32", "f32", 180],
        ["s16", "f16", 180],
        ["online", "none", 240],
      ] as const) {
        assertEquals(
          fusionCounts(
            transformer,
            shapes,
            { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: storage },
            ONLINE_DEVICE,
          ),
          { ...counts, linearPrologue: 31, attentionPermute: absorbed },
          `transformer S=${sequence} a8 attention, ${seat}`,
        );
      }
    }
    // The VAE graphs have none of the DiT forms: SiLU everywhere, the spatial nearest ×2 in the
    // decoder chunks. Both seats of the decoder carry the same node order.
    const vae: readonly (readonly [string, readonly string[], Partial<FusionCounts>])[] = [
      ["vae_encoder_pre", ["f16"], { silu: 18 }],
      ["vae_encoder_attn", ["f16"], {}],
      ["vae_encoder_post", ["f16"], { silu: 3 }],
      ["vae_decoder_first", ["f16", "i8"], { silu: 29, upsample2x: 3 }],
      ["vae_decoder_next", ["f16", "i8"], { silu: 29, upsample2x: 3 }],
      ["vae_decoder_first_32", ["f16", "i8"], { silu: 29, upsample2x: 3 }],
      ["vae_decoder_next_32", ["f16", "i8"], { silu: 29, upsample2x: 3 }],
    ];
    for (const [component, seats, counts] of vae) {
      for (const seat of seats) {
        const graph = await readWan22Graph(component, seat);
        assertEquals(
          fusionCounts(graph, solvedShapes(graph, 64)),
          { ...NONE, ...counts },
          `${component}/${seat}`,
        );
      }
    }
  },
);

/**
 * The candidate rules `candidate/vae_rms_silu_g8` / `_g4` (`@karume/candidate-rules`, the channel
 * RMS norm and the SiLU after it in one dispatch), which no preset passes, planned on every VAE
 * graph on 2026-11-11, per tail of the rule's key (the two rules plan alike):
 *
 * - **the decoder chunk graphs** (first / next, both seats): **29** norms with the SiLU (every
 *   resnet norm and the head's) and **1** without it (the mid-block attention's norm,
 *   `[1, 1024, 16, 16]` over axis 1, which a conv2d reads). The built-in silu's 29 go to the rule
 *   (silu 29 → 0), upsample2x keeps its 3, and the steps fall by 29 × 8 + 7 = 239 (the ten nodes of
 *   a site were nine steps, the silu pair being one): 437 → 198 on first, 447 → 208 on next.
 * - **the tile-32 pair** (`_32`): **0**. Every weight of those graphs is borrowed from the
 *   16-latent pair (`shared`), and the runtime does not bind a borrowed weight to an external rule
 *   (ADR 0148), so the windows are declined and silu keeps its 29.
 * - **the encoder** (I2V): pre **18** and post **3**, all with the SiLU (silu → 0); attn **0** — its
 *   norm reduces the last axis of `[S, 640]`, whose leading axis is not 1.
 */
test.skipIf(!WAN22_AVAILABLE)(
  "the candidate rules candidate/vae_rms_silu_g8 / _g4 take 29 + 1 norms of each 16-latent decoder chunk graph, 18 / 3 of the encoder and none of the tile-32 pair, and move only silu",
  async () => {
    const vae: readonly (readonly [
      component: string,
      seats: readonly string[],
      tails: Readonly<Record<string, number>>,
      steps: readonly [number, number],
    ])[] = [
      ["vae_encoder_pre", ["f16"], { silu: 18 }, [248, 104]],
      ["vae_encoder_attn", ["f16"], {}, [19, 19]],
      ["vae_encoder_post", ["f16"], { silu: 3 }, [33, 9]],
      ["vae_decoder_first", ["f16", "i8"], { silu: 29, none: 1 }, [437, 198]],
      ["vae_decoder_next", ["f16", "i8"], { silu: 29, none: 1 }, [447, 208]],
      ["vae_decoder_first_32", ["f16", "i8"], {}, [437, 437]],
      ["vae_decoder_next_32", ["f16", "i8"], {}, [447, 447]],
    ];
    for (const [component, seats, tails, steps] of vae) {
      for (const seat of seats) {
        const graph = await readWan22Graph(component, seat);
        for (const groups of [8, 4] as const) {
          const label = `${component}/${seat} g${groups}`;
          const planned = vaeRmsSiluPlans(graph, solvedShapes(graph, 64), groups);
          assertEquals(planned.tails, tails, `${label}: hits per tail`);
          const hits = (tails.silu ?? 0) + (tails.none ?? 0);
          const { counts } = planned.plain.plan;
          assertEquals(
            planned.fused.plan.counts,
            {
              ...counts,
              silu: counts.silu - (tails.silu ?? 0),
              [vaeRmsSiluRuleName(groups)]: hits,
            },
            `${label}: the counters`,
          );
          assertEquals(
            [planned.plain.plan.steps.length, planned.fused.plan.steps.length],
            [...steps],
            `${label}: steps`,
          );
        }
      }
    }
  },
);

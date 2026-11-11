/**
 * Fusion hit counts of the Wan2.1 distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the asset is `models/karume-wan2.1/` (untracked local asset); where it is missing the case
 * SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { opSequenceCount, stableFusionCandidates } from "./helpers/fusion-candidates.ts";
import {
  exists,
  fusionCounts,
  NONE,
  planDistributionFusions,
  readDistributionGraph,
} from "./helpers/fusion-counts.ts";

/**
 * The Wan2.1 T2V 1.3B distribution (`models/karume-wan2.1/` — ADR 0118 step 7; not published on
 * HF). The DiT is one S-shaped graph, the VAE two chunk graphs (static shapes at tile 32).
 */
const WAN_DIR = new URL("../../../models/karume-wan2.1/", import.meta.url);
const WAN_AVAILABLE = await exists(new URL("karume.json", WAN_DIR));
if (!WAN_AVAILABLE) {
  console.warn(
    `[karume] ${WAN_DIR.pathname} is missing, so the Wan2.1 fusion hit counts are SKIPPED`,
  );
}

/**
 * The Wan2.1 distribution is self-contained, so no cross-repo mirror is known: a cross-repo
 * reference in its manifest fails loudly instead of being read from somewhere else.
 */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * A component graph of the default model (`t2v-1.3b`). Every component has an f16 seat; the
 * transformer also has the i8 seat the a8 presets run.
 */
const readWanGraph = async (component: string, dtype = "f16"): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", WAN_DIR)),
  );
  return await readDistributionGraph(
    WAN_DIR,
    manifest,
    manifest.defaultModel,
    component,
    dtype,
    NO_MIRRORS,
  );
};

/**
 * Shape with the input's symbolic dimension (the DiT's `S`) solved as `sequence` (the VAE is all
 * static, so passed through).
 */
const wanShapes = (graph: IrGraph, sequence: number): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : sequence)),
    ]),
  );

/**
 * Device facts that hold the online kernel at the DiT's D 128 (subgroups and 48 KiB of workgroup
 * storage — the sg32 form needs 31,232 B).
 */
const ONLINE_DEVICE = { subgroups: true, maxComputeWorkgroupStorageSize: 48 * 1024 } as const;

/**
 * The three Wan2.1 components. The counts are frozen from measurements (2026-10-03; rope from
 * 2026-10-24; attentionPermute and rmsNormRope from 2026-10-25):
 *
 * - **transformer: silu 2 and rmsNormRope 60**. The two `sigmoid`s are the time-embedding MLP's
 *   (the SiLU inside `time_embedder` and the one before `time_proj`); both are grabbed as SiLU.
 *   - rmsNormRope 60: the q and k prologues of the 30 self-attention blocks — `rms_norm` over the
 *     1,536-wide token row, two reshapes, then the RoPE written as the complex rotation in real
 *     numbers (the **pair form** — swap of adjacent pairs + elementwise products with cos / sin;
 *     `recipe wan/dit_patch.py`) — each folded into one dispatch (perf-ledger F-3 stage 3).
 *   - rope 0: those 60 pair-form RoPEs (ROPE_RULE's second form, the `rope` count of 2026-10-24)
 *     now fold with their rms_norm; with `rmsNormReduce: "subgroup32"` they would fold alone again.
 *   - adaln 0: the modulation vectors are sliced **before** the layer_norm, so the 2–3 reshapes
 *     right after the layer_norm that ADALN_RULE's broadcast form requires are absent, and there is
 *     no `where` of the per-token form (perf-ledger K-73).
 *   - gatedResidual 0: the gate is a broadcast vector (no per-token `where`).
 *   - rowBlockAttention / identityExpand 0: attention is the fused op (`attention`, 60 of them),
 *     with no decomposed spelling.
 *   - **attentionPermute 180 under `attentionCompute: "a8"` only**, on both weight seats: each of
 *     the 60 attentions (30 self, 30 cross) absorbs the `[0, 2, 1, 3]` permutes of q, k and v right
 *     before it (perf-ledger F-3 A4 / F-8 — H = 12, D = 128). On the seats that store S (s16, and S
 *     in f32) the permute of O after it stays a node. The f32 attention seat absorbs nothing.
 *   - **attentionPermute 240 on a user-chosen online seat** (`attentionScoreStorage: "none"` on a
 *     device that holds the online kernel; counted 2026-10-26), on both weight seats: every
 *     attention takes the online route (D 128; the key count — S for self-attention, the 512 text
 *     tokens for cross-attention — is a multiple of 4 at every S checked here), so it also absorbs
 *     the permute of O (perf-ledger F-3 A4).
 *   - **linearPrologue 31 on the i8 seat under `linearCompute: "a8"`** (counted 2026-10-25): the
 *     gelu_tanh folded into the step ① of the i8 linear that reads it (perf-ledger F-4) — the FFN
 *     down projection of the 30 blocks (K 8,960) and the text embedder's second linear (K 1,536).
 *     The f16 seat has no i8a8 linear (0 under a8 too), and the default f32 linear has no step ①.
 * - **The VAE chunk graphs (first / next): silu 29 + upsample2x 3**. All 29 `sigmoid`s are SiLU
 *   (the resnet activations and before the head), and upsample2x is the three spatial ×2 stages
 *   (nearest ×2 + conv2d). The temporal upsample (`time_conv` of next) is not the upsample2x form.
 *
 * The counts do not depend on S (the DiT is checked at S = 192 and the full size 14,040). If a
 * count moves, first check whether a rule's accepted set widened or narrowed unintentionally (the
 * same reading as the Irodori DiT gate).
 */
test.skipIf(!WAN_AVAILABLE)(
  "the real Wan2.1 grabs DiT silu 2 / rmsNormRope 60 (plus linearPrologue 31 on the i8 seat under a8, and attentionPermute 180 on the a8 attention seats that store S, 240 on the online seat) and the VAE chunk graphs' silu 29 / upsample2x 3 per run (adaln is a different spelling: 0)",
  async () => {
    const transformer = await readWanGraph("transformer");
    const transformerI8 = await readWanGraph("transformer", "i8");
    for (const sequence of [192, 14040]) {
      const counts = { ...NONE, silu: 2, rmsNormRope: 60 };
      const f32 = (graph: IrGraph) => fusionCounts(graph, wanShapes(graph, sequence));
      const a8 = (graph: IrGraph) =>
        fusionCounts(graph, wanShapes(graph, sequence), { linearCompute: "a8" });
      // The f16+dit8-a8-attn8-s16 preset: linearCompute and attentionCompute "a8".
      const attn8 = (graph: IrGraph) =>
        fusionCounts(graph, wanShapes(graph, sequence), {
          linearCompute: "a8",
          attentionCompute: "a8",
        });
      assertEquals(f32(transformer), counts, `transformer S=${sequence}`);
      assertEquals(a8(transformer), counts, `transformer S=${sequence} f16 seat a8`);
      assertEquals(f32(transformerI8), counts, `transformer S=${sequence} i8 seat`);
      assertEquals(
        a8(transformerI8),
        { ...counts, linearPrologue: 31 },
        `transformer S=${sequence} i8 seat a8`,
      );
      assertEquals(
        attn8(transformerI8),
        { ...counts, linearPrologue: 31, attentionPermute: 180 },
        `transformer S=${sequence} i8 seat a8 attention`,
      );
      assertEquals(
        attn8(transformer),
        { ...counts, attentionPermute: 180 },
        `transformer S=${sequence} f16 seat a8 attention`,
      );
      // A user-chosen online seat: attentionCompute "a8" × attentionScoreStorage "none".
      const online = (graph: IrGraph) =>
        fusionCounts(
          graph,
          wanShapes(graph, sequence),
          { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
          ONLINE_DEVICE,
        );
      assertEquals(
        online(transformerI8),
        { ...counts, linearPrologue: 31, attentionPermute: 240 },
        `transformer S=${sequence} i8 seat a8 attention, online`,
      );
      assertEquals(
        online(transformer),
        { ...counts, attentionPermute: 240 },
        `transformer S=${sequence} f16 seat a8 attention, online`,
      );
    }
    for (const name of ["vae_decoder_first", "vae_decoder_next"]) {
      const graph = await readWanGraph(name);
      assertEquals(
        fusionCounts(graph, wanShapes(graph, 0)),
        { ...NONE, silu: 29, upsample2x: 3 },
        name,
      );
    }
  },
);

/**
 * The DiT's two known misses (adaln 0 and gatedResidual 0 above) as fusion candidates, named by the
 * op sequence of their chain (passthroughs excluded):
 *
 * - adaLN: `layer_norm, add, mul, add` — the norm, `1 + scale` (the modulation vector is sliced
 *   before the norm, so ADALN_RULE's broadcast form with its reshapes after the norm does not
 *   match), the product and the shift: **61** = the self-attention and FFN norms of the 30 blocks
 *   and the head's norm (the same 61 the Wan2.2 DiT grabs as adaln).
 * - Gated residual: `linear, mul, add` — the output projection of the self-attention or the FFN,
 *   its broadcast gate and the residual add (no per-token `where`): **60** = 30 blocks × 2. The
 *   cross-attention's projection is added without a gate.
 *
 * A candidate is not a rule to write: a mul→add gated residual rule was measured on anima and
 * Irodori (−0.2 to −0.6% of the GPU total, nothing in the wall) and rejected (perf-ledger K-7).
 * The check records that the candidate search finds these sites.
 *
 * The search is the stable one: it widens the window limit until the rows stop changing. At the
 * default limit of 12 the block chains are cut into overlapping windows marked "hit the limit" and
 * sites inside an overlap are counted twice (the adaLN form came out 62 times, the gated residual
 * 64); the widest maximal window is 18 nodes (2026-11-11).
 */
test.skipIf(!WAN_AVAILABLE)(
  "the Wan2.1 DiT's known misses come out as fusion candidates: the adaLN form 61 times, the gated residual 60",
  async () => {
    const transformer = await readWanGraph("transformer");
    for (const sequence of [192, 14040]) {
      const label = `transformer S=${sequence}`;
      const { plan, context } = planDistributionFusions(
        transformer,
        wanShapes(transformer, sequence),
      );
      const { rows, stable } = stableFusionCandidates(plan.steps, context);
      assertEquals(stable, true, `${label}: precondition — the stable search stabilised`);
      assertEquals(
        rows.filter((row) => row.hitLimit).map((row) => row.form),
        [],
        `${label}: precondition — no window at the limit`,
      );
      assertEquals(
        {
          adaln: opSequenceCount(rows, ["layer_norm", "add", "mul", "add"]),
          gatedResidual: opSequenceCount(rows, ["linear", "mul", "add"]),
        },
        { adaln: 61, gatedResidual: 60 },
        label,
      );
    }
  },
);

/**
 * Fusion hit counts of the Anima distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the assets are `models/karume-anima/` (official repo, 5 variants, default =
 * anima-turbo-v1.1 — ADR 0087) and `models/karume-anima-extra/` (extra variants whose shared
 * components are cross-repo references into the official repo). Both are untracked local assets;
 * where they are missing the cases SKIP explicitly with a reason (ADR 0005).
 *
 * NOTE: what this file pins are the values **per run** (the same lifetime as `lastRunFusions`). The
 * per-predict totals in the measurement column of ADR 0040 are these folded over the pipeline's run
 * counts: rope 56×8step + 56 = **504** / silu 2×8 + 28 + 29×9tile = **305** /
 * upsample2x 3×9tile = **27** / identityExpand 112 + 48 = **160** / adaln 85×8step = **680**
 * (the ADR's "adaln 85" is the number of static chains = one DiT run in this file).
 * The 9 VAE tiles are 3 per axis × 2 axes at 1024px (latent 128, tile 64, minimum overlap 8).
 * NOTE: the text encoder's rope is 56 (the record at the time ADR 0040 was written says 55). Once
 * an in-window passthrough could be grabbed, the **one first occurrence** where the sin table's
 * `sym_prefix_slice` falls into a gap of the chain started to be picked up as well.
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import {
  type ContainerManifest,
  containerPart0,
} from "../../runtime/tests/helpers/container-graph.ts";
import { DISABLED_FUSION_RULES } from "../../runtime/src/runtime/session-types.ts";
import { fusionCandidates, opSequenceCount } from "./helpers/fusion-candidates.ts";
import {
  exists,
  fusionCounts,
  NONE,
  planDistributionFusions,
  readDistributionGraph,
} from "./helpers/fusion-counts.ts";

const ANIMA_DIR = new URL("../../../models/karume-anima/", import.meta.url);

const ASSETS_AVAILABLE = await exists(new URL("karume.json", ANIMA_DIR));
if (!ASSETS_AVAILABLE) {
  console.warn(
    `[karume] ${ANIMA_DIR.pathname} has no karume.json, so the fusion hit counts of the real` +
      " assets SKIP (a regression in the exporter's node emission order can only be detected on the real assets)",
  );
}

/**
 * The distribution of the extra variants (`models/karume-anima-extra/` — anima-wai-v1.0 /
 * anima-copycat-20260610). It bakes text_encoder / vae_decoder as **cross-repository references**
 * into the official repository (ADR 0038 §7), so it is readable only once the official mirror is in
 * place.
 */
const EXTRA_DIR = new URL("../../../models/karume-anima-extra/", import.meta.url);

/**
 * The repo → local mirror map of cross-repository references (ADR 0038 §7). The official repository
 * is self-contained (ADR 0087); only the shared components of extra are read across repositories.
 */
const MIRRORS: ReadonlyMap<string, URL> = new Map([["hdae/karume-anima", ANIMA_DIR]]);

const EXTRA_AVAILABLE = ASSETS_AVAILABLE && (await exists(new URL("karume.json", EXTRA_DIR)));
if (!EXTRA_AVAILABLE) {
  console.warn(
    `[karume] ${EXTRA_DIR.pathname} has no karume.json (or the referenced ${ANIMA_DIR.pathname} is` +
      " absent), so the fusion hit counts of the extra variants SKIP (bake it with the cross-repository reference steps of the release-runbook)",
  );
}

// Where the assets are absent not a single byte is read (only tests that get past ignore touch
// these constants). **The distribution owns the spelling** (the part count, the paths and whether a
// reference crosses repositories are the exporter's choice), so the test reads them from the
// manifest instead of baking them in — baked in, the fusion hit-count check would fail with
// NotFound the moment the split count changes, and regression detection itself would stop.
const ANIMA_MANIFEST: ContainerManifest | undefined = ASSETS_AVAILABLE
  ? JSON.parse(await readTextFile(new URL("karume.json", ANIMA_DIR)))
  : undefined;
const EXTRA_MANIFEST: ContainerManifest | undefined = EXTRA_AVAILABLE
  ? JSON.parse(await readTextFile(new URL("karume.json", EXTRA_DIR)))
  : undefined;

/** The graph of a component of the official repository's default model (turbo). */
const readAnimaGraph = (component: string, dtype: string): Promise<IrGraph> => {
  const manifest = ANIMA_MANIFEST as ContainerManifest;
  return readDistributionGraph(
    ANIMA_DIR,
    manifest,
    manifest.defaultModel,
    component,
    dtype,
    MIRRORS,
  );
};

/**
 * The DiT's input shapes (S = the number of patch tokens; 1024px = 64×64 = 4096). The test below
 * checks at two points that the hit counts do not depend on S.
 */
const ditShapes = (sequence: number): Readonly<Record<string, readonly number[]>> => ({
  tokens: [1, sequence, 68],
  timesteps_proj: [1, 2048],
  encoder_hidden_states: [1, 512, 1024],
  rope_cos: [1, 1, sequence, 128],
  rope_sin: [1, 1, sequence, 128],
});

/**
 * The DiT's counts under `linearCompute: "a8"` (the `f16+dit8-a8*` presets; counted 2026-10-25):
 * the f32 counts plus **linearPrologue 28 on the i8 seat** — the erf gelu folded into the step ① of
 * the i8 linear that reads it (perf-ledger F-4), the FFN down projection of the 28 blocks (gelu
 * [S, 8192] → linear [2048, 8192]). The f16 seat has no i8a8 linear, so it stays at 0 under a8.
 */
const ditA8 = (quant: "i8" | "f16"): FusionCounts => ({
  ...NONE,
  silu: 2,
  rope: 56,
  adaln: 85,
  linearPrologue: quant === "i8" ? 28 : 0,
});

const a8Counts = (graph: IrGraph, shapes: Readonly<Record<string, readonly number[]>>) =>
  fusionCounts(graph, shapes, { linearCompute: "a8" });

/**
 * The DiT's counts on the a8 attention seat (the `f16+dit8-a8-attn8*` presets — linearCompute and
 * attentionCompute "a8"; counted 2026-10-25): the a8 counts plus **attentionPermute 56** — each of
 * the 56 attentions (28 self, 28 cross) absorbs the `[0, 2, 1, 3]` permute of v right before it
 * (perf-ledger F-3 A4 / F-8). Its q / k reach it through the per-head rms_norm and the RoPE (self)
 * or the per-head rms_norm alone (cross), so their permutes are not adjacent. On these seats, which
 * store S, the permute of O after it stays a node.
 */
const ditAttn8 = (quant: "i8" | "f16"): FusionCounts => ({ ...ditA8(quant), attentionPermute: 56 });

const attn8Counts = (graph: IrGraph, shapes: Readonly<Record<string, readonly number[]>>) =>
  fusionCounts(graph, shapes, { linearCompute: "a8", attentionCompute: "a8" });

/**
 * Device facts that hold the online kernel at the DiT's D 128 (subgroups and 48 KiB of workgroup
 * storage — the sg32 form needs 31,232 B).
 */
const ONLINE_DEVICE = { subgroups: true, maxComputeWorkgroupStorageSize: 48 * 1024 } as const;

/**
 * The DiT's counts on a user-chosen online seat (attentionCompute "a8" × attentionScoreStorage
 * "none" on a device that holds the online kernel; counted 2026-10-26): **attentionPermute 112**.
 * Every attention takes the online route (D 128; the key count — S for self-attention, the 512 text
 * tokens for cross-attention — is a multiple of 4 at every S checked here), so each of the 56 also
 * absorbs the permute of O right after it (perf-ledger F-3 A4).
 */
const ditOnline = (quant: "i8" | "f16"): FusionCounts => ({
  ...ditA8(quant),
  attentionPermute: 112,
});

const onlineCounts = (graph: IrGraph, shapes: Readonly<Record<string, readonly number[]>>) =>
  fusionCounts(
    graph,
    shapes,
    { linearCompute: "a8", attentionCompute: "a8", attentionScoreStorage: "none" },
    ONLINE_DEVICE,
  );

test.skipIf(!ASSETS_AVAILABLE)(
  "the real DiT grabs adaln 85 / rope 56 / silu 2 per run (the same on w8a8 and f16; under a8, linearPrologue 28 on the i8 seat only; attentionPermute 56 on the a8 attention seat that stores S, 112 on the online seat)",
  async () => {
    const expected: FusionCounts = { ...NONE, silu: 2, rope: 56, adaln: 85 };
    for (const quant of ["i8", "f16"] as const) {
      const graph = await readAnimaGraph("transformer", quant);
      // Fusion looks only at the f32 compute path, so the hit counts do not move when the weight
      // storage changes.
      assertEquals(fusionCounts(graph, ditShapes(4096)), expected, `${quant}: 1024px（S=4096）`);
      assertEquals(fusionCounts(graph, ditShapes(1024)), expected, `${quant}: 512px（S=1024）`);
      for (const sequence of [4096, 1024]) {
        assertEquals(
          a8Counts(graph, ditShapes(sequence)),
          ditA8(quant),
          `${quant} a8: S=${sequence}`,
        );
        assertEquals(
          attn8Counts(graph, ditShapes(sequence)),
          ditAttn8(quant),
          `${quant} a8 attention: S=${sequence}`,
        );
        assertEquals(
          onlineCounts(graph, ditShapes(sequence)),
          ditOnline(quant),
          `${quant} a8 attention, online: S=${sequence}`,
        );
      }
    }
  },
);

/**
 * A window limit no window of the unfused DiT reaches (its widest maximal window is 43 nodes —
 * 2026-11-11). At the default limit of 12 the block chains are cut into overlapping windows marked
 * "hit the limit", and a site inside an overlap is counted more than once.
 */
const UNCUT_WINDOW = 64;

/**
 * **The control of the fusion candidate search**: the DiT planned with every rule off
 * (`DISABLED_FUSION_RULES: "all"`) must show each site the rules grab today as a candidate — the
 * check that the enumerator neither loses a window nor counts one chain twice. Each rule's sites
 * are named by the op sequence of their chain in the unfused plan (passthroughs excluded):
 *
 * - silu 2: `sigmoid, mul`.
 * - rope 56: the half-split form, `slice, slice, neg, cat, mul, mul, add` — q and k of the 28
 *   self-attention blocks. Both sit in one maximal window that runs on through the attention.
 * - adaln 85: `layer_norm`, the reshapes of the modulation vectors, then `add, mul, add`. The rule
 *   places those reshapes as in-window passthroughs; unfused they are private values that flow into
 *   the chain, so they are chain nodes here: two at the 28 self-attention norms (the window also
 *   holds the reshape of the block's gate vector, a passthrough here too) and at the head's norm
 *   (no gate), three at the other 56.
 *
 * At the default window limit of 12 the same count gives rope 140 and the two-reshape adaLN 141:
 * the overlapping cut windows count one site several times.
 */
test.skipIf(!ASSETS_AVAILABLE)(
  "with fusion off, the real DiT's silu 2 / rope 56 / adaln 85 all come out as fusion candidates",
  async () => {
    const graph = await readAnimaGraph("transformer", "f16");
    for (const sequence of [4096, 1024]) {
      const label = `S=${sequence}`;
      const { plan, context } = planDistributionFusions(graph, ditShapes(sequence), {
        [DISABLED_FUSION_RULES]: "all",
      });
      assertEquals(plan.counts, NONE, `${label}: precondition — no rule grabs anything`);
      const rows = fusionCandidates(plan.steps, context, UNCUT_WINDOW);
      assertEquals(
        rows.filter((row) => row.hitLimit).map((row) => row.form),
        [],
        `${label}: precondition — no window at the limit`,
      );
      const adaln = (reshapes: number) => [
        "layer_norm",
        ...Array.from({ length: reshapes }, () => "reshape"),
        "add",
        "mul",
        "add",
      ];
      assertEquals(
        {
          silu: opSequenceCount(rows, ["sigmoid", "mul"]),
          rope: opSequenceCount(rows, ["slice", "slice", "neg", "cat", "mul", "mul", "add"]),
          adaln: opSequenceCount(rows, adaln(2)) + opSequenceCount(rows, adaln(3)),
        },
        { silu: 2, rope: 56, adaln: 85 },
        label,
      );
    }
  },
);

test.skipIf(!ASSETS_AVAILABLE)(
  "fusion hit counts of the real text encoder / conditioner / VAE decoder",
  async () => {
    const textEncoder = await readAnimaGraph("text_encoder", "f16");
    assertEquals(
      fusionCounts(textEncoder, { input_ids: [1, 64] }),
      { ...NONE, silu: 28, rope: 56, identityExpand: 112 },
      "text encoder",
    );
    const conditionerShapes = {
      source_hidden_states: [1, 64, 1024],
      target_input_ids: [1, 512],
    };
    const conditioner = await readAnimaGraph("text_conditioner", "f16");
    assertEquals(
      fusionCounts(conditioner, conditionerShapes),
      { ...NONE, rope: 24, identityExpand: 48 },
      "conditioner (the 12 BSHD q/k chains each are fused too)",
    );
    // Its 6 erf gelu → linear pairs do not fold even under a8: the conditioner's weights are f16 /
    // bf16 on every preset (no i8a8 linear to absorb them; the quant's session reaches only the
    // DiT).
    for (const quant of ["f16", "bf16"] as const) {
      assertEquals(
        a8Counts(await readAnimaGraph("text_conditioner", quant), conditionerShapes),
        { ...NONE, rope: 24, identityExpand: 48 },
        `conditioner ${quant} a8: no linearPrologue`,
      );
    }
    const vae = await readAnimaGraph("vae_decoder", "f16");
    assertEquals(
      fusionCounts(vae, { latents: [1, 16, 64, 64] }),
      { ...NONE, silu: 29, upsample2x: 3 },
      "VAE decoder (one tile)",
    );
  },
);

/**
 * The fusion hit counts of the extra variants. The values were measured and frozen on 2026-09-24 on
 * extra (karume/0.13.0, referencing hdae/karume-anima@adb9dcf0) — they are fine-tunes with the same
 * architecture as the official ones, so they end up equal to the official values, but they are kept
 * as **the values of actually planning this distribution's graphs** (extra bakes its own
 * transformer / text_conditioner per variant, so an emission-order regression can show up here
 * independently).
 *
 * text_encoder / vae_decoder are cross-repository references into the official repository, so this
 * check also serves as the check that the {@link MIRRORS} routing works on the real assets.
 */
const EXTRA_MODELS = ["anima-copycat-20260610", "anima-wai-v1.0"] as const;
const EXTRA_CROSS_REPO_COMPONENTS = ["text_encoder", "vae_decoder"] as const;

test.skipIf(!EXTRA_AVAILABLE)(
  "the real extra assets match the manifest's variant table for both variants and hold the shared components as cross-repository references",
  () => {
    const manifest = EXTRA_MANIFEST as ContainerManifest;
    // Pin the whole table, so that the day a variant is added, one that skips the fusion check is
    // not shipped silently.
    assertEquals(Object.keys(manifest.models).sort(), [...EXTRA_MODELS]);
    for (const model of EXTRA_MODELS) {
      // So that this check does not silently turn into "a check that crosses no repository" the day
      // extra becomes self-contained again.
      const crossed = ["text_encoder", "text_conditioner", "transformer", "vae_decoder"].filter(
        (component) =>
          Object.keys(manifest.models[model].weights[component]).some(
            (dtype) => containerPart0(manifest, model, component, dtype).repo !== undefined,
          ),
      );
      assertEquals(
        crossed,
        [...EXTRA_CROSS_REPO_COMPONENTS],
        `${model}: components held as cross-repository references`,
      );
    }
  },
);

for (const model of EXTRA_MODELS) {
  test.skipIf(!EXTRA_AVAILABLE)(
    `fusion hit counts of the real extra ${model} (including the shared components it references)`,
    async () => {
      const manifest = EXTRA_MANIFEST as ContainerManifest;
      const read = (component: string, dtype: string): Promise<IrGraph> =>
        readDistributionGraph(EXTRA_DIR, manifest, model, component, dtype, MIRRORS);
      const dit: FusionCounts = { ...NONE, silu: 2, rope: 56, adaln: 85 };
      for (const quant of ["i8", "f16"] as const) {
        const graph = await read("transformer", quant);
        assertEquals(fusionCounts(graph, ditShapes(4096)), dit, `${quant}: 1024px（S=4096）`);
        assertEquals(fusionCounts(graph, ditShapes(1024)), dit, `${quant}: 512px（S=1024）`);
        assertEquals(a8Counts(graph, ditShapes(4096)), ditA8(quant), `${quant} a8: S=4096`);
        assertEquals(
          attn8Counts(graph, ditShapes(4096)),
          ditAttn8(quant),
          `${quant} a8 attention: S=4096`,
        );
        assertEquals(
          onlineCounts(graph, ditShapes(4096)),
          ditOnline(quant),
          `${quant} a8 attention, online: S=4096`,
        );
      }
      assertEquals(
        fusionCounts(await read("text_encoder", "f16"), { input_ids: [1, 64] }),
        { ...NONE, silu: 28, rope: 56, identityExpand: 112 },
        "text encoder (cross-repository)",
      );
      assertEquals(
        fusionCounts(await read("text_conditioner", "f16"), {
          source_hidden_states: [1, 64, 1024],
          target_input_ids: [1, 512],
        }),
        { ...NONE, rope: 24, identityExpand: 48 },
        "conditioner",
      );
      assertEquals(
        fusionCounts(await read("vae_decoder", "f16"), { latents: [1, 16, 64, 64] }),
        { ...NONE, silu: 29, upsample2x: 3 },
        "VAE decoder (cross-repository, one tile)",
      );
    },
  );
}

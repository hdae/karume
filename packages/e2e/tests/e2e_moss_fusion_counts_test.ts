/**
 * Fusion hit counts and fusion candidates of the MOSS-Image DiT stage graphs (no GPU —
 * `planFusions` on part 0 only; the shared planning side lives in `./helpers/fusion-counts.ts`, the
 * candidate search in `./helpers/fusion-candidates.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-moss-image/` (untracked local asset);
 * where it is missing the cases SKIP explicitly with the command that gets it (ADR 0005).
 *
 * The DiT stage is three graphs: `dit` (42 blocks — the positive branch), `dit_skip22` (the same
 * blocks without block 22 — the negative branch) and `dit_text` (the text refiner, f32 compute in
 * every preset — ADR 0138 decision 3, so it is planned without knobs only). Their counts were
 * frozen from planning the mirror on 2026-11-11 and do not depend on the sequence length (each is
 * checked at two lengths: S = N + T with N = 256 / 4,096 image tokens at 256² / 1024² and T = 64 /
 * 128 text tokens).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { builtInCounts } from "../../runtime/tests/helpers/fusion-count-record.ts";
import {
  MOSS_PRACTICAL_PRESET,
  MOSS_REFERENCE_PRESET,
  mossPreset,
} from "../../models/src/moss/preset.ts";
import { MOSS_GET_MIRROR, MOSS_MIRROR } from "../../models/tests/helpers/moss-dit.ts";
import { opSequenceCount, stableFusionCandidates } from "./helpers/fusion-candidates.ts";
import {
  exists,
  fusionCounts,
  NONE,
  planDistributionFusions,
  readDistributionGraph,
} from "./helpers/fusion-counts.ts";

const MOSS_AVAILABLE = await exists(new URL("karume.json", MOSS_MIRROR));
if (!MOSS_AVAILABLE) {
  console.warn(
    `[karume] ${MOSS_MIRROR.pathname} has no karume.json, so the MOSS-Image DiT fusion hit counts` +
      ` and candidates SKIP (get it with: ${MOSS_GET_MIRROR})`,
  );
}

/**
 * The MOSS-Image distribution is self-contained, so no cross-repo mirror is known: a cross-repo
 * reference in its manifest fails loudly instead of being read from somewhere else.
 */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/** A DiT-stage graph of the default model (every DiT-stage component has the i8 seat only). */
const readMossGraph = async (component: string): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", MOSS_MIRROR)),
  );
  return await readDistributionGraph(
    MOSS_MIRROR,
    manifest,
    manifest.defaultModel,
    component,
    "i8",
    NO_MIRRORS,
  );
};

/** The input shapes with the graph's one symbol (`S` of the DiT pair, `T` of `dit_text`) bound. */
const boundShapes = (graph: IrGraph, length: number): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => (typeof dim === "number" ? dim : length)),
    ]),
  );

/** S = N + T at 256² (256 image tokens, 64 text tokens) and at 1024² (4,096 + 128). */
const PAIR_LENGTHS = [256 + 64, 4096 + 128] as const;

/** The knobs of the two presets (the preset table owns them — `packages/models/src/moss/preset.ts`). */
const PRESETS = [MOSS_REFERENCE_PRESET, MOSS_PRACTICAL_PRESET] as const;

/**
 * Device facts that hold the online kernel at the DiT's head width (subgroups and 48 KiB of
 * workgroup storage) — the practical preset's online seat.
 */
const ONLINE_DEVICE = { subgroups: true, maxComputeWorkgroupStorageSize: 48 * 1024 } as const;

/**
 * The hit counts of the DiT pair:
 *
 * - **silu 1**: the time-embedding MLP's SiLU. The other 42 sigmoids (one per block) sit inside
 *   the FFN activation's chain, not in the SiLU form.
 * - **adaln 0 / gatedResidual 0 / rope 0** — the three known misses (the candidates below):
 *   ADALN_RULE's leading op is `layer_norm` and the DiT normalises with `rms_norm`; GATED_RESIDUAL_RULE
 *   takes a per-token gate picked by `where` and the DiT's gate is a broadcast `mul`; ROPE_RULE's
 *   half-split form wants `neg, cat` before the products and the DiT emits the `x · cos` product
 *   before the `neg` (`slice, slice, mul, neg, cat, mul, add`).
 * - **attentionPermute 84 on the practical preset** (`dit`; 82 on `dit_skip22`, one block fewer):
 *   each attention absorbs the permute of v right before it and, on the online route, the permute
 *   of O right after it. Its q / k reach it through the RoPE, so their permutes are not adjacent.
 * - linearPrologue 0 under `linearCompute: "a8"`: the prologue folds `gelu` only, and the FFN's
 *   activation is a different chain.
 */
test.skipIf(!MOSS_AVAILABLE)(
  "the real MOSS-Image DiT stage grabs silu 1 per run of the pair (plus attentionPermute 84 / 82 on the practical preset), nothing in dit_text, and none of adaln / gatedResidual / rope",
  async () => {
    for (const [component, permutes] of [
      ["dit", 84],
      ["dit_skip22", 82],
    ] as const) {
      const graph = await readMossGraph(component);
      for (const length of PAIR_LENGTHS) {
        for (const name of PRESETS) {
          const practical = name === MOSS_PRACTICAL_PRESET;
          assertEquals(
            fusionCounts(graph, boundShapes(graph, length), mossPreset(name).dit, ONLINE_DEVICE),
            builtInCounts({ silu: 1, ...(practical ? { attentionPermute: permutes } : {}) }),
            `${component} S=${length} ${name}`,
          );
        }
      }
    }
    const text = await readMossGraph("dit_text");
    for (const length of [1, 64, 128]) {
      assertEquals(fusionCounts(text, boundShapes(text, length)), NONE, `dit_text T=${length}`);
    }
  },
);

/**
 * The three known misses as fusion candidates, each at its **84 sites** (42 blocks × 2 — the
 * applicability table of 2026-11-10 counted the same 84 each). They are named by the op sequence
 * of their chain (passthroughs excluded), and one maximal window may hold more than one site:
 *
 * - RoPE: `slice, slice, mul, neg, cat, mul, add` — q and k per block. On the reference preset both
 *   sit in one maximal window that runs from the q projection through the attention to the gated
 *   residual; on the practical preset the attention's step absorbs permutes and ends the plain run,
 *   so each stands alone (`slice, slice, mul, neg, cat, mul, add, permute`, 84 windows).
 * - adaLN: `rms_norm, mul, add` — the norm, the broadcast `1 + scale` product and the shift, before
 *   the attention and before the FFN of each block.
 * - Gated residual: `linear, mul, add` — the output projection (attention or FFN), its broadcast gate
 *   and the residual add.
 *
 * The search is the stable one: it widens the window limit until the rows stop changing. At the
 * default limit of 12 the long chains of the blocks are cut into overlapping windows marked "hit the
 * limit", and a site inside the overlap is counted twice (the gated residual of the FFN came out 86
 * times). The widest maximal window is 69 nodes on the reference preset (the search stops at 192)
 * and 25 on the practical one (it stops at 96 — 2026-11-11). A fixed limit of 32 leaves no window at
 * the limit on either preset: the 69-node window only appears once the
 * limit spans it, which the limit mark cannot tell.
 *
 * If a count moves, first check whether the exporter's emission order changed or a rule started to
 * grab a site (its hit count above moves with it).
 */
test.skipIf(!MOSS_AVAILABLE)(
  "the MOSS-Image DiT's three known misses come out as fusion candidates at 84 sites each (RoPE, adaLN, gated residual)",
  async () => {
    const graph = await readMossGraph("dit");
    for (const name of PRESETS) {
      const label = `dit ${name}`;
      const { plan, context } = planDistributionFusions(
        graph,
        boundShapes(graph, 4096 + 128),
        mossPreset(name).dit,
        ONLINE_DEVICE,
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
          rope: opSequenceCount(rows, ["slice", "slice", "mul", "neg", "cat", "mul", "add"]),
          adaln: opSequenceCount(rows, ["rms_norm", "mul", "add"]),
          gatedResidual: opSequenceCount(rows, ["linear", "mul", "add"]),
        },
        { rope: 84, adaln: 84, gatedResidual: 84 },
        label,
      );
    }
  },
);

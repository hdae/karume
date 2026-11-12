/**
 * Fusion hit counts and fusion candidates of the MOSS-Image DiT stage graphs, and the hits of the
 * family's fusion rule on every image graph (no GPU — `planFusions` on part 0 only; the shared
 * planning side lives in `./helpers/fusion-counts.ts`, the candidate search in
 * `./helpers/fusion-candidates.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-moss-image/` (untracked local asset);
 * where it is missing the cases SKIP explicitly with the command that gets it (ADR 0005).
 *
 * The DiT stage is three graphs: `dit` (42 blocks — the positive branch), `dit_skip22` (the same
 * blocks without block 22 — the negative branch) and `dit_text` (the text refiner, f32 compute in
 * every preset — ADR 0138 decision 3, so it is planned without knobs only). Their counts were
 * frozen from planning the mirror on 2026-11-11 and do not depend on the sequence length (each is
 * checked at two lengths: S = N + T with N = 256 / 4,096 image tokens at 256² / 1024² and T = 64 /
 * 128 text tokens). The first case pins the built-in rules' hits (the plan without the family's
 * rule); the last cases pin the hits of `models/ew_chain` (`@karume/models/moss`), which the pipeline
 * passes by default to `dit_text`, the DiT pair and the three VAE decoder graphs.
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { EW_CHAIN_RULE_NAME } from "@karume/models/moss";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import { DISABLED_FUSION_RULES } from "../../runtime/src/runtime/session-types.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { builtInCounts } from "../../runtime/tests/helpers/fusion-count-record.ts";
import { mossFusionRules } from "../../models/src/moss/fusion-rules.ts";
import {
  MOSS_PRACTICAL_PRESET,
  MOSS_REFERENCE_PRESET,
  mossPreset,
} from "../../models/src/moss/preset.ts";
import { MOSS_GET_MIRROR, MOSS_MIRROR } from "../../models/tests/helpers/moss-dit.ts";
import { opSequenceCount, stableFusionCandidates } from "./helpers/fusion-candidates.ts";
import {
  exists,
  type FusionCountOptions,
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

/**
 * A graph of the default model at its one seat: `i8` for the DiT-stage components, `f32` for the VAE
 * decoder graphs.
 */
const readMossGraph = async (component: string, seat = "i8"): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", MOSS_MIRROR)),
  );
  return await readDistributionGraph(
    MOSS_MIRROR,
    manifest,
    manifest.defaultModel,
    component,
    seat,
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

/** The sequence lengths of the 1024² fridge-note request: S₊ = 4,096 + 63, S₋ = 4,096 + 10. */
const REQUEST_LENGTHS = { dit: 4096 + 63, dit_skip22: 4096 + 10 } as const;

/** The input shapes with each symbol bound by name (the VAE decoder graphs have two: `h`, `w`). */
const shapesOf = (
  graph: IrGraph,
  binding: Readonly<Record<string, number>>,
): Readonly<Record<string, readonly number[]>> =>
  Object.fromEntries(
    graph.inputs.map((spec) => [
      spec.name,
      spec.shape.map((dim) => {
        if (typeof dim === "number") return dim;
        if (!Object.hasOwn(binding, dim)) throw new Error(`no binding for the symbol '${dim}'`);
        return binding[dim];
      }),
    ]),
  );

/** The rule's forms (its dispatch key without the rule's prefix and suffix). */
const GATE_FORM =
  "f.s.f.s:div.x0.x1-tanh.t0-mul.t1.x1-sigmoid.x0-mul.t2.t3-div.x2.x3-tanh.t5-mul.t6.x3-mul.t4.t7";
const ROPE_FORM = "f.b0101.f:mul.x0.x1-add.x2.t0";
const GATED_FORM = "f.b001.f:mul.x0.x1-add.x2.t0";
const ADALN_FORM = "f.b001.b001:mul.x0.x1-add.t0.x2";
const RESIDUAL_FORM = "f.f.s:add.x0.x1-div.t0.x2";

/**
 * Plans `graph` with the knobs and the pipeline's default rule list (`mossFusionRules` — the list
 * the MOSS-Image stages give every image graph's Session and estimate), and with the same options and
 * `models/ew_chain` disabled by name (the test-only `DISABLED_FUSION_RULES`, the plain side), and
 * asserts the rule's hits per form, that no other counter moves, and that the steps fall by the
 * nodes folded. Returns the plain and fused step counts and the plain side's silu hits.
 */
const assertDefaultRuleHits = (
  graph: IrGraph,
  shapes: Readonly<Record<string, readonly number[]>>,
  knobs: FusionCountOptions,
  expected: Readonly<Record<string, number>>,
  label: string,
): { readonly steps: readonly [number, number]; readonly silu: number } => {
  const options = { ...knobs, fusionRules: mossFusionRules(undefined) };
  const fused = planDistributionFusions(graph, shapes, options, ONLINE_DEVICE);
  const plain = planDistributionFusions(
    graph,
    shapes,
    { ...options, [DISABLED_FUSION_RULES]: [EW_CHAIN_RULE_NAME] },
    ONLINE_DEVICE,
  );
  const seen: Record<string, number> = {};
  let folded = 0;
  for (const step of fused.plan.steps) {
    if (step.kind !== "fused" || step.rule !== EW_CHAIN_RULE_NAME) continue;
    const form = step.dispatches[0].key.replace(/^ext:models:ew_chain:v1:|:f32:wg128$/g, "");
    seen[form] = (seen[form] ?? 0) + 1;
    folded += step.nodeCount - 1;
  }
  assertEquals(seen, expected, `${label}: hits per form`);
  const hits = Object.values(expected).reduce((sum, count) => sum + count, 0);
  assertEquals(plain.plan.counts[EW_CHAIN_RULE_NAME], 0, `${label}: the plain side`);
  assertEquals(
    fused.plan.counts,
    { ...plain.plan.counts, [EW_CHAIN_RULE_NAME]: hits },
    `${label}: the counters`,
  );
  assertEquals(
    fused.plan.steps.length,
    plain.plan.steps.length - folded,
    `${label}: the steps fall by the nodes folded`,
  );
  return {
    steps: [plain.plan.steps.length, fused.plan.steps.length],
    silu: plain.plan.counts.silu,
  };
};

/**
 * The pipeline's default rule `models/ew_chain` (a run of f32 elementwise nodes in one dispatch)
 * planned on the DiT pair with each preset's knobs, per form (its dispatch key names the inputs'
 * classes and the node list):
 *
 * - **the nine-node FFN gate** (`div, tanh, mul, sigmoid, mul, div, tanh, mul, mul` over
 *   `[1, S, 8192]` with two `[1]` constants), 42 on `dit` — **0 on `dit_skip22`**, whose constants
 *   are weights borrowed from `dit`, which the runtime declines to bind (ADR 0148 decisions 5 and 7);
 * - **the RoPE tail** (`mul(·, sin [1, S, 1, D]) → add(x·cos, ·)` over `[1, S, 24, 128]`), 84 / 82;
 * - **the gated residual's tail** (`mul(·, gate [1, 1, C]) → add(residual, ·)`), 84 / 82;
 * - **adaLN's tail** (`mul(·, 1 + scale [1, 1, C]) → add(·, shift [1, 1, C])`), 82 / 80 — block 0's
 *   pair sits around an aliasing `reshape` and an independent `add` (the candidate rows), which a
 *   run of consecutive nodes does not cross.
 *
 * 292 on `dit` and 244 on `dit_skip22`, on both presets, at the request's length and at 256² alike.
 * The built-in silu stays at 1 (the rule leaves the exact `sigmoid(x) → mul(x, ·)` pair to it),
 * every other counter is the plan's without the rule, and the steps fall by the nodes folded (at
 * the request's length 2,310 → 1,724 and 2,256 → 2,012 on the practical preset, 2,394 → 1,808 and
 * 2,338 → 2,094 on the reference preset).
 */
test.skipIf(!MOSS_AVAILABLE)(
  "the pipeline's default rule models/ew_chain takes 292 runs of dit and 244 of dit_skip22 on both presets, per form, and moves no other counter",
  async () => {
    const forms = (gate: number, rope: number, gated: number, adaln: number) => ({
      ...(gate === 0 ? {} : { [GATE_FORM]: gate }),
      [ROPE_FORM]: rope,
      [GATED_FORM]: gated,
      [ADALN_FORM]: adaln,
    });
    const STEPS = {
      [MOSS_PRACTICAL_PRESET]: { dit: [2310, 1724], dit_skip22: [2256, 2012] },
      [MOSS_REFERENCE_PRESET]: { dit: [2394, 1808], dit_skip22: [2338, 2094] },
    } as const;
    for (const [component, expected] of [
      ["dit", forms(42, 84, 84, 82)],
      ["dit_skip22", forms(0, 82, 82, 80)],
    ] as const) {
      const graph = await readMossGraph(component);
      for (const name of PRESETS) {
        for (const length of [REQUEST_LENGTHS[component], 256 + 64]) {
          const label = `${component} ${name} S=${length}`;
          const { steps, silu } = assertDefaultRuleHits(
            graph,
            boundShapes(graph, length),
            mossPreset(name).dit,
            expected,
            label,
          );
          assertEquals(silu, 1, `${label}: silu`);
          if (length === REQUEST_LENGTHS[component]) {
            assertEquals(steps, STEPS[name][component], `${label}: steps`);
          }
        }
      }
    }
  },
);

/**
 * The same default rule on the other image graphs the pipeline gives it to, per form:
 *
 * - **`dit_text`** (the text refiner, planned without knobs — f32 compute in every preset): 6 — the
 *   RoPE tails of q and k in each of its two blocks and the two FFN gates (the DiT's forms);
 * - **the VAE decoder graphs**: every residual connection `add(x, h) → div(·, scale)` with the
 *   block's output scale as a one-element constant — 1 in `vae_decoder_pre`, 1 in
 *   `vae_decoder_attn` (the attention's), 13 in `vae_decoder_post`.
 *
 * The counts do not depend on the bound sizes (each graph is planned at two); every other counter
 * is the plan's without the rule, and the steps fall by the nodes folded (103 → 83, 10 → 9,
 * 18 → 17 and 115 → 102).
 */
test.skipIf(!MOSS_AVAILABLE)(
  "the pipeline's default rule models/ew_chain takes 6 runs of dit_text and 1 / 1 / 13 of the VAE decoder graphs, per form, and moves no other counter",
  async () => {
    /** A graph, its seat, two bindings of its symbols, the hits per form and the plain / fused steps. */
    type Row = readonly [
      component: string,
      seat: string,
      bindings: readonly Readonly<Record<string, number>>[],
      expected: Readonly<Record<string, number>>,
      steps: readonly [number, number],
    ];
    const rows: readonly Row[] = [
      ["dit_text", "i8", [{ T: 64 }, { T: 128 }], { [ROPE_FORM]: 4, [GATE_FORM]: 2 }, [103, 83]],
      [
        "vae_decoder_pre",
        "f32",
        [
          { h: 32, w: 32 },
          { h: 128, w: 128 },
        ],
        { [RESIDUAL_FORM]: 1 },
        [10, 9],
      ],
      ["vae_decoder_attn", "f32", [{ S: 1024 }, { S: 16384 }], { [RESIDUAL_FORM]: 1 }, [18, 17]],
      [
        "vae_decoder_post",
        "f32",
        [
          { h: 32, w: 32 },
          { h: 128, w: 128 },
        ],
        { [RESIDUAL_FORM]: 13 },
        [115, 102],
      ],
    ];
    for (const [component, seat, bindings, expected, steps] of rows) {
      const graph = await readMossGraph(component, seat);
      for (const binding of bindings) {
        const label = `${component} ${JSON.stringify(binding)}`;
        assertEquals(
          assertDefaultRuleHits(graph, shapesOf(graph, binding), {}, expected, label).steps,
          steps,
          `${label}: steps`,
        );
      }
    }
  },
);

/**
 * Fusion hit counts of the Irodori distribution graphs (no GPU — `planFusions` on part 0 only; the
 * shared planning side and the rationale live in `runtime/tests/helpers/fusion-counts.ts`).
 *
 * MUST: the assets are `models/karume-irodori-v4-small/` and the distributions built on it —
 * `karume-irodori-v4.1-small`, `karume-irodori-v4.1-small-mf` and `karume-irodori-v4-large`
 * (untracked local assets); where one is missing its cases SKIP explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { IRODORI_SESSION_POLICY } from "../../models/src/irodori/admission.ts";
import { resolveSessionOptions } from "../../models/src/session/options.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import {
  type ContainerManifest,
  containerPart0,
  readContainerGraph,
} from "../../runtime/tests/helpers/container-graph.ts";
import { opSequenceCount, stableFusionCandidates } from "./helpers/fusion-candidates.ts";
import {
  boundFusionCounts,
  exists,
  fusionCounts,
  NONE,
  planDistributionFusions,
  readDistributionGraph,
  type SymbolBinding,
} from "./helpers/fusion-counts.ts";

const IRODORI_DIR = new URL("../../../models/karume-irodori-v4-small/", import.meta.url);

/**
 * Fusion hit counts depend **only on the node list** (the storage dtype changes only which
 * initializer is read and how), so looking at the single f32 one is enough however many series
 * there are — what is looked at is the exporter's node emission order, which comes from the same
 * single conversion across series.
 */
const readIrodoriGraph = async (name: string): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", IRODORI_DIR)),
  );
  const head = containerPart0(manifest, "v4-small", name, "f32");
  return await readContainerGraph(new URL(head.path, IRODORI_DIR), name);
};

// Look at a **file** (not the directory) — when the distribution's spelling moves, judging "assets
// present" from a non-empty directory alone ends in a FAIL on an unreadable path.
const IRODORI_AVAILABLE = await exists(new URL("karume.json", IRODORI_DIR));
if (!IRODORI_AVAILABLE) {
  console.warn(`[karume] ${IRODORI_DIR.pathname} is absent, so Irodori's fusion hit counts SKIP`);
}

/**
 * The input shapes of the DiT's condition-side K/V projection `dit_context` (3 condition states —
 * right-padded to Tmax, static).
 */
const IRODORI_CONTEXT_SHAPES: Readonly<Record<string, readonly number[]>> = {
  text_state: [1, 256, 512],
  speaker_state: [1, 751, 768],
  caption_state: [1, 512, 512],
};

/**
 * The DiT's input shapes (S = the latent frame count; mask is the derived dimension `S+1519`). The
 * condition side takes `dit_context`'s 24 outputs (12 blocks × K / V, each `[1,1519,20,64]` — ADR
 * 0114) as inputs of the same names.
 */
const irodoriDitShapes = (sequence: number): Readonly<Record<string, readonly number[]>> => ({
  x_t: [1, sequence, 32],
  t_embed: [1, 512],
  mask: [1, 1, 1, sequence + 1519],
  ...Object.fromEntries(
    Array.from({ length: 12 }, (_, block) => block).flatMap((block) => [
      [`context_k_${block}`, [1, 1519, 20, 64]],
      [`context_v_${block}`, [1, 1519, 20, 64]],
    ]),
  ),
});

/**
 * The Irodori DiT (12 blocks). The record of what the matchers' accepted sets cover in this
 * spelling:
 *
 * - **rope 23**: the DiT's RoPE is the **pair form** (view as `[1,S,H·D/2,2]`, slice / neg / cat
 *   on the last axis, reshape back — on the first 10 of the 20 heads), the same spelling as Wan's
 *   DiT, which ROPE_RULE accepts exactly as its second form (2026-10-24). 12 blocks × q / k = 24
 *   sites; the first use of the shared cos / sin tables puts their two `sym_prefix_slice`s inside
 *   the window, which the pair form does not accept, so that one site stays plain (23 = 24 − 1).
 *   The fused kernel performs the same f32 operations in the same order (the WAV sha256 rows of
 *   `e2e_irodori_wav_test.ts` held on the RTX 3080 Ti with it — 2026-10-24).
 * - adaln 0: ADALN_RULE's leading op is `layer_norm`, but the DiT normalises with `rms_norm` (87)
 *   and has no `layer_norm` at all.
 * - silu 17: the timestep MLP's 5 + one per block (12). The remaining 12 sigmoids are the gates of
 *   `mul(v, sigmoid(u))` (not applied to themselves, so not SiLU).
 * - **rowBlockAttention 12**: every decomposed attention of the 12 blocks (`bmm → reshape →
 *   add(mask) → safe_softmax → expand → reshape → expand → reshape → bmm`). Among the real assets
 *   only the DiT emits this spelling; the other 7 parts carry the `attention` op (ADR 0023).
 * - identityExpand 24: of the 12 blocks × 4, the two inside the window (P side and V side) are
 *   swallowed by the fused step and no longer count as aliases. **48 → 24 is that move, not a
 *   regression**; if `rowBlockAttention` went back to 0 it would be 48 again (only one of the two
 *   moving means an accident in an accepted set).
 *
 * So this gate guards both "the grabbed kinds do not drop out" and "**it moves when an accepted
 * set widens**". If a count changes, first check whether ROPE_RULE / ADALN_RULE accepted something
 * unintentionally (the moment a set widens past an exact form, the fallback's correctness
 * guarantee is gone).
 */
test.skipIf(!IRODORI_AVAILABLE)(
  "the real Irodori DiT grabs rope 23 / rowBlockAttention 12 / silu 17 / identityExpand 24 per run (adaln is a different spelling: 0)",
  async () => {
    const graph = await readIrodoriGraph("dit");
    const expected: FusionCounts = {
      ...NONE,
      silu: 17,
      rope: 23,
      rowBlockAttention: 12,
      identityExpand: 24,
    };
    // The hit counts do not depend on S (2 points inside `ditSymMax` 750). **The row-block count
    // changes with S** (with the 128MiB limit, S=125 is 1 block and S=750 is 2), but the hit counts
    // do not move.
    for (const sequence of [125, 750]) {
      assertEquals(fusionCounts(graph, irodoriDitShapes(sequence)), expected, `S=${sequence}`);
    }
    // The condition-side K/V projection (the 72 linears + rms_norm + cat ADR 0114 split out of
    // `dit`) grabs no fusion at all — the paired observation point that `dit`'s 3 kinds did not
    // move here.
    assertEquals(
      fusionCounts(await readIrodoriGraph("dit_context"), IRODORI_CONTEXT_SHAPES),
      NONE,
      "dit_context",
    );
  },
);

test.skipIf(!IRODORI_AVAILABLE)(
  "fusion hit counts of the real Irodori condition path and codec",
  async () => {
    // The backbone (ModernBERT-ja, 25 layers) is the only half-split RoPE: 25 layers × q / k = 50.
    // It is the counterpart of Irodori's own blocks (dit / speaker — the pair form): **if only one
    // side drops out, this tells which spelling changed**.
    assertEquals(
      fusionCounts(await readIrodoriGraph("backbone"), { input_ids: [1, 256] }),
      { ...NONE, rope: 50 },
      "backbone",
    );
    assertEquals(
      fusionCounts(await readIrodoriGraph("text_proj"), { hidden: [1, 256, 768] }),
      { ...NONE, silu: 1 },
      "text_proj",
    );
    assertEquals(
      fusionCounts(await readIrodoriGraph("caption_proj"), { hidden: [1, 512, 768] }),
      { ...NONE, silu: 1 },
      "caption_proj",
    );
    // speaker has 8 blocks (silu 8 + 8 gates). Its RoPE is the DiT's pair form on all 12 heads:
    // 8 blocks × q / k = 16 sites, of which the first use of the tables stays plain (its two
    // sym_prefix_slices sit inside the window) — 15.
    assertEquals(
      fusionCounts(await readIrodoriGraph("speaker"), { latent: [1, 750, 128] }),
      { ...NONE, silu: 8, rope: 15 },
      "speaker",
    );
    assertEquals(
      fusionCounts(await readIrodoriGraph("duration"), {
        text_state: [1, 256, 512],
        speaker_vec: [1, 768],
        has_speaker: [1, 1],
        caption_vec: [1, 512],
        has_caption: [1, 1],
      }),
      { ...NONE, silu: 5 },
      "duration",
    );
    // The codec has Snake activations (29 `sin` — not sigmoid) and 4 `conv_transpose1d`, and
    // neither the silu nor the upsample2x shape. All 0 records "nothing to grab"; a change to non-0
    // means an accepted set widened.
    for (const [name, shapes] of [
      ["codec_decoder", { latent: [1, 750, 32] }],
      ["codec_encoder", { wav: [1, 750, 1920] }],
    ] as const) {
      assertEquals(fusionCounts(await readIrodoriGraph(name), shapes), NONE, name);
    }
  },
);

/**
 * The DiT's adaLN (adaln 0 above — it normalises with `rms_norm`) as a fusion candidate, named by
 * the op sequence of its chain: `rms_norm, add, mul, add` — the norm, `1 + scale`, the product and
 * the shift. **24** = the 12 blocks × 2 (before the attention and before the MLP), the same 24 as
 * the record of the removed 0.14 tool (`docs/research/2026-09-03-op-census-fusion-hints.md` §2.2).
 * 23 of them end a longer maximal window (`linear, linear, add, …` before the norm); one stands
 * alone.
 *
 * The search is the stable one: it widens the window limit until the rows stop changing (the widest
 * maximal window is 13 nodes — 2026-11-11; at the default limit of 12 some windows are marked "hit
 * the limit").
 */
test.skipIf(!IRODORI_AVAILABLE)(
  "the Irodori DiT's adaLN form comes out as a fusion candidate 24 times (rms_norm, add, mul, add)",
  async () => {
    const graph = await readIrodoriGraph("dit");
    for (const sequence of [125, 750]) {
      const label = `S=${sequence}`;
      const { plan, context } = planDistributionFusions(graph, irodoriDitShapes(sequence));
      const { rows, stable } = stableFusionCandidates(plan.steps, context);
      assertEquals(stable, true, `${label}: precondition — the stable search stabilised`);
      assertEquals(
        rows.filter((row) => row.hitLimit).map((row) => row.form),
        [],
        `${label}: precondition — no window at the limit`,
      );
      assertEquals(opSequenceCount(rows, ["rms_norm", "add", "mul", "add"]), 24, label);
    }
  },
);

/**
 * The distributions built on v4-small: v4.1-small, its MeanFlow variant v4.1-small-mf and
 * v4-Large. Each shares some containers with an earlier mirror through cross-repository
 * references (v4.1-small: everything but `duration` and the i4 `dit`; v4.1-small-mf: everything
 * but `dit` and `dit_context`; v4-Large: the two codec graphs), read here from the sibling mirror.
 */
const IRODORI_VARIANTS = [
  "karume-irodori-v4.1-small",
  "karume-irodori-v4.1-small-mf",
  "karume-irodori-v4-large",
] as const;

/** The repo → local mirror map of the variants' cross-repository references (ADR 0038 §7). */
const IRODORI_MIRRORS: ReadonlyMap<string, URL> = new Map(
  ["karume-irodori-v4-small", "karume-irodori-v4.1-small"].map((name) => [
    `hdae/${name}`,
    mirrorRoot(name),
  ]),
);

/** A binding of each graph's symbol (`dit_context` has none; the DiT at two S, as above). */
const IRODORI_BINDINGS: Readonly<Record<string, readonly SymbolBinding[]>> = {
  backbone: [{ T: 256 }],
  text_proj: [{ T: 256 }],
  caption_proj: [{ T: 512 }],
  speaker: [{ S: 750 }],
  duration: [{ T: 256 }],
  dit: [{ S: 125 }, { S: 750 }],
  dit_context: [{}],
  codec_decoder: [{ S: 750 }],
  codec_encoder: [{ T: 750 }],
};

/** The hits of the graphs v4-small and the two v4.1 distributions share in form (the cases above). */
const SMALL_COUNTS: Readonly<Record<string, Partial<FusionCounts>>> = {
  backbone: { rope: 50 },
  text_proj: { silu: 1 },
  caption_proj: { silu: 1 },
  speaker: { silu: 8, rope: 15 },
  duration: { silu: 5 },
  dit: { silu: 17, rope: 23, rowBlockAttention: 12, identityExpand: 24 },
  dit_context: {},
  codec_decoder: {},
  codec_encoder: {},
};

/**
 * Per distribution and graph, on every preset (the Session options of the pipeline: `dit` and
 * `dit_context` take the preset's — `i8-a8`: `linearCompute: "a8"`, which folds nothing here: the
 * linear prologue folds a `gelu` node and the DiT has none — and every other graph the defaults):
 *
 * - **v4.1-small**: v4-small's counts above (its own `duration` and i4 `dit` included).
 * - **v4.1-small-mf**: the same, but the DiT's silu is **19** — the interval embedding of a
 *   MeanFlow step (`delta_embed`) adds a second two-SiLU MLP to the head (7 + 12 blocks).
 * - **v4-Large** (24 DiT blocks, a 26-layer backbone, 14 speaker blocks): backbone rope 52
 *   (26 × q / k); speaker silu 14 and rope 27 (14 × 2 − the first use of the tables, as on
 *   v4-small); duration silu 6; the DiT silu 29 (the head's 5 + 24), rope 47 (24 × 2 − 1),
 *   rowBlockAttention 24 and identityExpand 48 (24 × 2, the pair swallowed by each fused
 *   attention).
 */
const IRODORI_VARIANT_COUNTS: Readonly<
  Record<(typeof IRODORI_VARIANTS)[number], Readonly<Record<string, Partial<FusionCounts>>>>
> = {
  "karume-irodori-v4.1-small": SMALL_COUNTS,
  "karume-irodori-v4.1-small-mf": { ...SMALL_COUNTS, dit: { ...SMALL_COUNTS.dit, silu: 19 } },
  "karume-irodori-v4-large": {
    backbone: { rope: 52 },
    text_proj: { silu: 1 },
    caption_proj: { silu: 1 },
    speaker: { silu: 14, rope: 27 },
    duration: { silu: 6 },
    dit: { silu: 29, rope: 47, rowBlockAttention: 24, identityExpand: 48 },
    dit_context: {},
    codec_decoder: {},
    codec_encoder: {},
  },
};

for (const name of IRODORI_VARIANTS) {
  const mirror = mirrorRoot(name);
  const available = mirrorPresent(mirror);
  if (!available) {
    console.warn(
      `[karume] ${mirror.pathname} (or a mirror it references) has no karume.json, so its Irodori` +
        " fusion hit counts SKIP (build it with dist.py in tools/export-recipes)",
    );
  }
  test.skipIf(!available)(
    `the real ${name} graphs plan the same fusions on every preset`,
    async () => {
      const text = await readTextFile(new URL("karume.json", mirror));
      const manifest: ContainerManifest = JSON.parse(text);
      const parsed = parseManifest(text);
      const expected = IRODORI_VARIANT_COUNTS[name];
      for (const [model, entry] of Object.entries(parsed.models)) {
        assertEquals(
          Object.keys(entry.weights),
          Object.keys(expected),
          `${model}: the graphs planned`,
        );
        for (const [preset, quant] of Object.entries(entry.quants)) {
          const ditOptions = resolveSessionOptions(
            IRODORI_SESSION_POLICY,
            quant.session,
            {},
            preset,
          );
          for (const [component, counts] of Object.entries(expected)) {
            const storage = quant.weights[component];
            const graph = await readDistributionGraph(
              mirror,
              manifest,
              model,
              component,
              storage,
              IRODORI_MIRRORS,
            );
            const options = component === "dit" || component === "dit_context" ? ditOptions : {};
            for (const binding of IRODORI_BINDINGS[component]) {
              assertEquals(
                boundFusionCounts(graph, binding, options),
                { ...NONE, ...counts },
                `${model} ${preset} ${component}/${storage} ${JSON.stringify(binding)}`,
              );
            }
          }
        }
      }
    },
  );
}

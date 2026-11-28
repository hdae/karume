/**
 * Fusion hit counts of the Qwen3.5 text encoder graph — MOSS-Image's text encoder, the one
 * distribution that ships it (no GPU — `planFusions` on part 0 only; the shared planning side and
 * the rationale live in `./helpers/fusion-counts.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-moss-image/` (untracked local asset);
 * where it is missing the case SKIPs explicitly with the command that gets it (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { MOSS_PRESET_NAMES } from "../../models/src/moss/preset.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import {
  MOSS_GET_MIRROR,
  MOSS_MIRROR,
  mossMirrorPresent,
} from "../../models/tests/helpers/moss-dit.ts";
import { boundFusionCounts, NONE, readDistributionGraph } from "./helpers/fusion-counts.ts";

const AVAILABLE = mossMirrorPresent();
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MOSS_MIRROR.pathname} has no karume.json, so the Qwen3.5 text encoder's fusion hit` +
      ` counts SKIP (get it with: ${MOSS_GET_MIRROR})`,
  );
}

/** The MOSS-Image distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * Qwen3.5-9B's 32 layers — 24 linear-attention (Gated DeltaNet) layers and 8 full-attention layers
 * — at two token lengths (the counts do not depend on `T`). The encoder's Session takes no preset
 * knob (`Qwen3_5TextEncoder.open` is given no options), so both MOSS-Image presets plan it alike.
 *
 * - **silu 80**: the SiLU of each layer's MLP (32), and in each linear-attention layer the SiLU
 *   after its short convolution and the one of its gated norm's gate (24 + 24). The other 32
 *   sigmoids are not SiLU: 24 feed the delta scan directly, 8 gate the full attention's output.
 * - **rope 16**: q and k of the 8 full-attention layers.
 * - Nothing else. Its norms follow the residual adds (`add → rms_norm → linear`; RMS→add takes a
 *   norm followed by the add, and only at the measured widths — the model's 4,096 is not one), and
 *   its 48 `expand` nodes broadcast (they change the shape), so none is an identity expand.
 */
test.skipIf(!AVAILABLE)(
  "the real Qwen3.5 text encoder grabs silu 80 / rope 16 on both MOSS-Image presets",
  async () => {
    const text = await readTextFile(new URL("karume.json", MOSS_MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const model = manifest.defaultModel;
    const quants = parseManifest(text).models[model].quants;
    assertEquals(Object.keys(quants), [...MOSS_PRESET_NAMES], "the presets planned");
    const storages = new Set(Object.values(quants).map((quant) => quant.weights.text_encoder));
    assertEquals([...storages], ["i8"], "the text encoder's storage on every preset");
    const graph = await readDistributionGraph(
      MOSS_MIRROR,
      manifest,
      model,
      "text_encoder",
      "i8",
      NO_MIRRORS,
    );
    for (const length of [64, 128]) {
      assertEquals(
        boundFusionCounts(graph, { T: length }),
        { ...NONE, silu: 80, rope: 16 },
        `text_encoder T=${length}`,
      );
    }
  },
);

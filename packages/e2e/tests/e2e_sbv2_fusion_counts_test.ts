/**
 * Fusion hit counts of the Style-Bert-VITS2 distribution graphs — the DeBERTa text encoder, `front`
 * and `voice` (no GPU — `planFusions` on part 0 only; the shared planning side and the rationale
 * live in `./helpers/fusion-counts.ts`).
 *
 * MUST: the asset is the distribution mirror `models/karume-sbv2-jvnv/` (untracked local asset);
 * where it is missing the case SKIPs explicitly with a reason (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { SBV2_SESSION_POLICY } from "../../models/src/sbv2/pipeline.ts";
import { resolveSessionOptions } from "../../models/src/session/options.ts";
import type { FusionCounts } from "../../runtime/src/runtime/fusion.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { mirrorPresent, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import {
  boundFusionCounts,
  NONE,
  readDistributionGraph,
  type SymbolBinding,
} from "./helpers/fusion-counts.ts";

const MIRROR = mirrorRoot("karume-sbv2-jvnv");
const AVAILABLE = mirrorPresent(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} has no karume.json, so the sbv2 fusion hit counts SKIP` +
      " (build it with dist.py --pipeline sbv2 in tools/export-recipes)",
  );
}

/** The sbv2 distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

/**
 * Per component: two bindings of its symbol (the counts do not depend on it) and its hits. The
 * pipeline runs the text encoder with the Session defaults on every preset and gives `front` and
 * `voice` the preset's options (`i8-a8`: `linearCompute: "a8"`).
 *
 * - **text_encoder** (DeBERTa-v2, 22 layers): nothing. It normalises with `layer_norm` straight into
 *   a `linear` (adaLN wants the modulation chain after the norm), and its 22 `linear → gelu →
 *   linear` pairs would fold only into an i8a8 linear (the linear prologue under `linearCompute:
 *   "a8"`), which the text encoder never gets. Its three `expand` nodes change the shape.
 * - **front**: identityExpand 42 — 42 of its 56 `expand` nodes keep their input's shape and are
 *   aliases. Its 24 `gelu` feed `conv1d` and `add`, never a `linear`, so `i8-a8` moves nothing.
 * - **voice**: identityExpand 168 of its 218 `expand` nodes. It has no `gelu` at all.
 */
const COMPONENTS: readonly (readonly [string, readonly SymbolBinding[], Partial<FusionCounts>])[] =
  [
    ["text_encoder", [{ T: 64 }, { T: 128 }], {}],
    ["front", [{ P: 64 }, { P: 128 }], { identityExpand: 42 }],
    ["voice", [{ T: 256 }, { T: 512 }], { identityExpand: 168 }],
  ];

test.skipIf(!AVAILABLE)(
  "the real sbv2 graphs plan the same fusions on every speaker and preset (nothing in the DeBERTa text encoder; identityExpand 42 in front, 168 in voice)",
  async () => {
    const text = await readTextFile(new URL("karume.json", MIRROR));
    const manifest: ContainerManifest = JSON.parse(text);
    const parsed = parseManifest(text);
    for (const [model, entry] of Object.entries(parsed.models)) {
      assertEquals(
        Object.keys(entry.weights),
        COMPONENTS.map(([component]) => component),
        `${model}: the graphs planned`,
      );
      for (const [preset, quant] of Object.entries(entry.quants)) {
        const options = resolveSessionOptions(SBV2_SESSION_POLICY, quant.session, {}, preset);
        for (const [component, bindings, expected] of COMPONENTS) {
          const storage = quant.weights[component];
          const graph = await readDistributionGraph(
            MIRROR,
            manifest,
            model,
            component,
            storage,
            NO_MIRRORS,
          );
          for (const binding of bindings) {
            assertEquals(
              boundFusionCounts(graph, binding, component === "text_encoder" ? {} : options),
              { ...NONE, ...expected },
              `${model} ${preset} ${component}/${storage} ${JSON.stringify(binding)}`,
            );
          }
        }
      }
    }
  },
);

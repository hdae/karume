/**
 * The MOSS-Image shape of the attention bench's set (`ATTENTION_BENCH_SHAPES` of
 * `@karume/runtime/tune`, id `moss-image-1024` — the shapes attention-bench records must cover for
 * the profile derivation, ADR 0117 addendum 2026-11-12) is the default preset's online attention at
 * the family's 1024² request, pinned against the family itself: the image tokens of a 1024² request
 * (`assertMossImageSize` / `mossImageTokens`), the text tokens of the fridge-note prompt (the ids of
 * upstream's text-embedding asset — the prompt of the 1024² sha rows), and the `dit` graph's
 * attention nodes in the distribution mirror (heads, depth, half scale).
 *
 * The runtime keeps the shape as numbers (it cannot import a model family), so a family change that
 * moves any of them turns this red instead of leaving the bench measuring another shape. Without the
 * mirror or the asset, the case SKIPs with the command that gets it (ADR 0005). No GPU.
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { ATTENTION_BENCH_SHAPES } from "@karume/runtime/tune";
import { mossImageTokens } from "../src/moss/dit-tokens.ts";
import { assertMossImageSize } from "../src/moss/request.ts";
import {
  MOSS_FETCH_GOLDENS,
  MOSS_GET_MIRROR,
  MOSS_TEXT_EMBEDS,
  mossMirrorPresent,
  openMossMirrorContainer,
  testAssetsRunnable,
} from "./helpers/moss-dit.ts";
import { readMossUpstreamPrompts } from "./helpers/moss-text.ts";

const MIRROR_PRESENT = mossMirrorPresent();
const ASSET_PRESENT = testAssetsRunnable([MOSS_TEXT_EMBEDS]);
if (!MIRROR_PRESENT) console.warn(`[karume] SKIP: no MOSS-Image mirror (${MOSS_GET_MIRROR})`);
if (!ASSET_PRESENT) console.warn(`[karume] SKIP: no text-embedding asset (${MOSS_FETCH_GOLDENS})`);

describe.skipIf(!MIRROR_PRESENT || !ASSET_PRESENT)(
  "the attention bench's MOSS-Image shape is the family's 1024² attention",
  () => {
    it("S = the image tokens of 1024² + the fridge-note prompt's tokens, and the dit graph's attention is [1, heads, S, depth] at the shape's scale", async () => {
      const shape = ATTENTION_BENCH_SHAPES.find(({ id }) => id === "moss-image-1024");
      assert(shape !== undefined, "the bench's set has no MOSS-Image shape");
      const prompt = (await readMossUpstreamPrompts()).find(({ name }) => name === "fridge-note");
      assert(prompt !== undefined, "the asset has no fridge-note prompt");
      const sequence = mossImageTokens(assertMossImageSize(1024, 1024)) + prompt.ids.length;
      assertEquals([shape.queries, shape.keys], [sequence, sequence]);
      const opened = await openMossMirrorContainer("dit");
      const [graph] = Object.values(opened.graphs);
      const signatures = new Set(
        graph.declaration.nodes
          .filter(({ op }) => op === "attention")
          .map(({ ins, attrs }) =>
            JSON.stringify([ins.map((name) => graph.declaration.values[name]?.shape), attrs]),
          ),
      );
      const operand = [1, shape.heads, "S", shape.depth];
      assertEquals(
        [...signatures],
        [JSON.stringify([[operand, operand, operand], { scale: shape.scale }])],
      );
    });
  },
);

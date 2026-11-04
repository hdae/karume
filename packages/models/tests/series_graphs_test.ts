// Checks of the lookup functions of the series → graph name table (helpers/series-graphs.ts). Needs
// neither a GPU nor real assets.
//
// The table was gathered in one place so that the gatekeeper and the e2e tests do not hold separate
// tables, but if a lookup silently returned `undefined`, half of that purpose would be lost — the
// caller would pass `undefined` to `prepareContainer` or open **a different graph** through a
// default. Neither lets a misspelling be caught by the numbers (`prepareContainer` fails on a wrong
// name, but passes if another graph named by the default exists). This pins that the lookup rejects
// a misspelling on the spot.
//
// Whether the table's contents match the real files is checked with real assets by
// `e2e_assets_gate_test.ts` (outside this check's reach).

import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { GRAPH_NAME_PATTERN } from "../../runtime/src/format/container/limits.ts";
import { SERIES_GRAPHS, seriesComponents, seriesGraph } from "./helpers/series-graphs.ts";

test("seriesGraph returns the graph name from the table (both directly in the root and in component directories)", () => {
  assertEquals(seriesGraph("gemma4-e2b-product"), "model");
  assertEquals(seriesGraph("depth-anything-v2-small-hf"), "depth");
  // Two typical cases whose spelling differs from the directory name (exactly why the table is
  // needed).
  assertEquals(seriesGraph("dacvae-32dim", "decoder"), "codec_decoder");
  assertEquals(seriesGraph("irodori-v4-small", "caption-proj"), "caption_proj");
});

test("seriesGraph rejects a series not in the table, naming it", () => {
  const error = assertThrows(() => seriesGraph("gemma4-e2b-produkt"), Error, "gemma4-e2b-produkt");
  // The existing spellings are listed too (a typo cannot be fixed without the list).
  assertEquals(error.message.includes("gemma4-e2b-product"), true, error.message);
});

test("seriesGraph rejects a component not in the table, even for a series that is", () => {
  const error = assertThrows(() => seriesGraph("dacvae-32dim", "decoders"), Error, "decoders");
  assertEquals(error.message.includes("decoder"), true, error.message);
});

test("seriesGraph rejects a lookup that gets the presence of components wrong", () => {
  // Looking up a series with component directories as if directly in the root (`""` is not in the
  // table).
  assertThrows(() => seriesGraph("dacvae-32dim"), Error, "<根直下>");
  // Passing a component name for a series directly in the root (lists `<根直下>` (root) as the
  // existing component).
  assertThrows(() => seriesGraph("gemma4-e2b-product", "model"), Error, "<根直下>");
});

test("seriesComponents rejects a series not in the table and hands out the component table as is for one that is", () => {
  assertThrows(() => seriesComponents("sbv2-F9"), Error, "sbv2-F9");
  assertEquals(Object.keys(seriesComponents("deberta-i8")).sort(), [
    "full-24layer",
    "sbv2-22layer",
  ]);
});

test("every graph name in the table fits the container's graph-name vocabulary", () => {
  // A spelling with a space such as `codec decoder` would not fail until `prepareContainer` (and
  // getting that far needs real assets). Make it fail as soon as the table entry is added.
  const outside = Object.entries(SERIES_GRAPHS).flatMap(([series, components]) =>
    Object.entries(components)
      .filter(([, graph]) => !GRAPH_NAME_PATTERN.test(graph))
      .map(([component, graph]) => `${series}/${component}: ${graph}`),
  );
  assertEquals(outside, [], `rows outside the graph-name vocabulary (${GRAPH_NAME_PATTERN})`);
});

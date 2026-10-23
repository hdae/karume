/**
 * The query-row windows of the online attention kernel (`planAttentionOnlineWindows` in
 * src/runtime/recipe-builders/attention.ts — ADR 0132 decision 2). No GPU: the window is a
 * dispatch-size device (a work budget per dispatch), so the plan is a pure function of the shape.
 * Expected values are computed by hand here (rebuilding them with the same formula would make the
 * test tautological).
 */

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ATTENTION_ONLINE_ROWS } from "../src/kernels/attention-online-i8a8.ts";
import {
  ATTENTION_ONLINE_WINDOW_MACS,
  planAttentionOnlineWindows,
} from "../src/runtime/recipe-builders/attention.ts";

describe("planAttentionOnlineWindows", () => {
  it("the Wan2.2 121-frame shape: 6,528-row windows, 5 of them, the last one the remainder", () => {
    // 2⁴⁰ / (24 · 27,280 · 128 · 2) = 6,543.6… rows → rounded down to a multiple of 64 = 6,528.
    const windows = planAttentionOnlineWindows(27280, { batch: 24, cols: 27280, depth: 128 });
    assertEquals(windows.length, 5);
    for (const [index, window] of windows.slice(0, -1).entries()) {
      assertEquals(window, { offset: index * 6528, rows: 6528 }, `window ${index}`);
    }
    // 4 · 6,528 = 26,112; 27,280 − 26,112 = 1,168
    assertEquals(windows.at(-1), { offset: 26112, rows: 1168 });
  });

  it("every window but the last is a multiple of 64 rows and within the work budget", () => {
    for (const extent of [
      { batch: 24, cols: 27280, depth: 128 },
      { batch: 24, cols: 7920, depth: 128 },
      { batch: 2, cols: 512, depth: 64 },
      { batch: 1, cols: 16384, depth: 128 },
    ]) {
      const rows = 33333;
      const windows = planAttentionOnlineWindows(rows, extent);
      let next = 0;
      for (const window of windows) {
        assertEquals(window.offset, next, "windows are contiguous");
        next += window.rows;
        const macs = extent.batch * window.rows * extent.cols * extent.depth * 2;
        assert(macs <= ATTENTION_ONLINE_WINDOW_MACS, `${JSON.stringify(extent)}: ${macs} MACs`);
      }
      assertEquals(next, rows, "the windows cover every row once");
      for (const window of windows.slice(0, -1)) {
        assertEquals(window.rows % ATTENTION_ONLINE_ROWS, 0, JSON.stringify(extent));
      }
    }
  });

  it("a shape inside the budget is one window over every row", () => {
    assertEquals(planAttentionOnlineWindows(150, { batch: 2, cols: 200, depth: 128 }), [
      { offset: 0, rows: 150 },
    ]);
    assertEquals(planAttentionOnlineWindows(1, { batch: 1, cols: 4, depth: 64 }), [
      { offset: 0, rows: 1 },
    ]);
  });

  it("splits at the budget: 2¹⁶ rows per window at B·H 1, N 65,536, D 128", () => {
    // 2⁴⁰ / (65,536 · 128 · 2) = 65,536 rows exactly — one more workgroup row makes a second window.
    assertEquals(planAttentionOnlineWindows(65536, { batch: 1, cols: 65536, depth: 128 }), [
      { offset: 0, rows: 65536 },
    ]);
    assertEquals(planAttentionOnlineWindows(65600, { batch: 1, cols: 65536, depth: 128 }), [
      { offset: 0, rows: 65536 },
      { offset: 65536, rows: 64 },
    ]);
  });

  it("never goes below one workgroup row (64) when even 64 rows exceed the budget", () => {
    // 64 rows · 4,096 heads · 65,536 keys · 128 · 2 = 2⁴⁴ MACs > 2⁴⁰ — still 64 rows per window.
    const windows = planAttentionOnlineWindows(200, { batch: 4096, cols: 65536, depth: 128 });
    assertEquals(windows, [
      { offset: 0, rows: 64 },
      { offset: 64, rows: 64 },
      { offset: 128, rows: 64 },
      { offset: 192, rows: 8 },
    ]);
  });
});

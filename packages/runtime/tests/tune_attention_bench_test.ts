// The attention route bench (`runAttentionRouteBench` of `@karume/runtime/tune`): the pure helpers,
// the binding-role check with storage params, and one GPU run on small shapes (both routes timed,
// a finite ratio, an N % 4 != 0 shape timed online with its padded Vᵀ quantisation, and a shape the
// online kernel cannot run reported as not eligible), and the opt-in candidate `onlineTiled`
// (perf-ledger K-106): the paired timing's pure summary, and one GPU run where the candidate is timed
// in the same rounds as `online`, its producer beside today's, and its identity block holds.

import { describe, test } from "vite-plus/test";
import { assert, assertEquals, assertMatch, assertThrows } from "@std/assert";
import { acquireGpu } from "../mod.ts";
import {
  ATTENTION_BENCH_FORMAT,
  type AttentionBenchRoute,
  type AttentionBenchShape,
  runAttentionRouteBench,
  WAN22_ATTENTION_SHAPES,
} from "../tune.ts";
import {
  attentionIntegerOps,
  checkAttentionBenchShape,
  finiteRatio,
  medianOf,
  onlineDispatchRows,
  pairedRoundRatio,
  positionalBindings,
  summarizePair,
  withProducerRatio,
} from "../src/tune/attention-bench.ts";
import { assertBindingRoles, PAIR_ORDER, type PairedRound } from "../src/tune/harness.ts";
import { parseStorageRoles } from "../src/gpu/pipeline-cache.ts";
import { stridedWgsl } from "../src/codegen/strided.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const shape = (overrides: Partial<AttentionBenchShape>): AttentionBenchShape => ({
  id: "test",
  heads: 2,
  queries: 256,
  keys: 512,
  depth: 128,
  scale: 0.2973017692565918,
  ...overrides,
});

describe("attention bench helpers", () => {
  test("the Wan2.2 shapes are 24 heads × D 128 at N 7,920 / 27,280 / 512 with the exporter's half scale", () => {
    assertEquals(
      WAN22_ATTENTION_SHAPES.map(({ heads, queries, keys, depth }) => [
        heads,
        queries,
        keys,
        depth,
      ]),
      [
        [24, 7920, 7920, 128],
        [24, 27280, 27280, 128],
        [24, 27280, 512, 128],
      ],
    );
    // f32(128^-1/4) — the value the exporter writes for SDPA's default scale (aten_handlers.py)
    for (const { scale } of WAN22_ATTENTION_SHAPES) assertEquals(scale, 0.2973017692565918);
  });

  test("positional bindings order the pairs by binding number and refuse a gap", () => {
    assertEquals(
      positionalBindings("qk", [
        [4, "qscale"],
        [1, "qq"],
        [3, "s"],
        [5, "kscale"],
        [2, "kq"],
      ]),
      ["qq", "kq", "s", "qscale", "kscale"],
    );
    assertThrows(
      () =>
        positionalBindings("pv", [
          [1, "s"],
          [3, "stats"],
        ]),
      Error,
      "not exactly 1..2",
    );
  });

  test("the online route measures whole 64-row workgroups, clamped at the node's rows", () => {
    assertEquals(onlineDispatchRows(6528, 27280), 6528);
    assertEquals(onlineDispatchRows(3032, 27280), 3072);
    assertEquals(onlineDispatchRows(150, 150), 150);
    assertEquals(onlineDispatchRows(1, 512), 64);
  });

  test("integer operations count the QK and PV multiply-accumulates twice each", () => {
    assertEquals(attentionIntegerOps(24, 6528, 27280, 128), 4 * 24 * 6528 * 27280 * 128);
  });

  test("the ratio is given only for two positive times", () => {
    assertEquals(finiteRatio(2, 4), 0.5);
    assertEquals(finiteRatio(2, 0), undefined);
    assertEquals(finiteRatio(0, 4), undefined);
    assertEquals(finiteRatio(undefined, 4), undefined);
  });

  test("a shape with a non-integer extent or rows outside 1..M is refused before the GPU", () => {
    checkAttentionBenchShape(shape({}));
    assertThrows(() => checkAttentionBenchShape(shape({ keys: 0 })), Error, "keys");
    assertThrows(() => checkAttentionBenchShape(shape({ heads: 1.5 })), Error, "heads");
    assertThrows(() => checkAttentionBenchShape(shape({ rowsPerDispatch: 257 })), Error, "1..256");
    assertThrows(() => checkAttentionBenchShape(shape({ scale: Number.NaN })), Error, "scale");
  });

  test("storage params put binding 0 among the reads the role check expects", () => {
    const roles = parseStorageRoles(stridedWgsl({ dtype: "f32" }));
    assertBindingRoles(
      { key: "strided", bindings: ["v", "vt"], writes: ["vt"], paramsUsage: "storage" },
      roles,
    );
    assertThrows(
      () => assertBindingRoles({ key: "strided", bindings: ["v", "vt"], writes: ["vt"] }, roles),
      Error,
      "storage bindings",
    );
  });
});

describe("the paired timing of the candidate (pure)", () => {
  const round = (a: [number, number], b: [number, number], clamped = 0): PairedRound => ({
    a,
    b,
    wallA: [a[0] + 10, a[1] + 10],
    wallB: [b[0] + 10, b[1] + 10],
    clamped,
  });

  test("a round runs A, B, B, A, so a drift that is linear in time weighs both arms alike", () => {
    assertEquals([...PAIR_ORDER], ["a", "b", "b", "a"]);
    // Pass i runs at clock factor 1 + 0.1·i: A at 1.0 and 1.3, B at 1.1 and 1.2 — the same sum
    const drift = [1, 1.1, 1.2, 1.3];
    const sums = { a: 0, b: 0 };
    PAIR_ORDER.forEach((arm, pass) => {
      sums[arm] += drift[pass];
    });
    assertEquals(sums.a.toFixed(12), sums.b.toFixed(12));
  });

  test("the round ratio is B ÷ A over both passes, and absent when a pass was clamped", () => {
    assertEquals(pairedRoundRatio(round([100, 100], [90, 100])), 0.95);
    assertEquals(pairedRoundRatio(round([100, 100], [0, 100], 1)), undefined);
  });

  test("the median takes the middle value, or the mean of the middle two", () => {
    assertEquals(medianOf([3, 1, 2]), 2);
    assertEquals(medianOf([4, 1, 3, 2]), 2.5);
    assertEquals(medianOf([]), undefined);
  });

  test("the summary keeps every round with its ratio, the median ratio and each arm's min per dispatch", () => {
    const summary = summarizePair("ns", {
      reps: 10,
      rounds: [
        round([1e7, 1.1e7], [0.95e7, 1.0e7]),
        round([1.2e7, 1.2e7], [1.08e7, 1.08e7]),
        round([0.9e7, 1e7], [0, 0.8e7], 1),
      ],
    });
    assertEquals(summary.reps, 10);
    assertEquals(
      summary.rounds?.map(({ ratio }) => ratio?.toFixed(4)),
      ["0.9286", "0.9000", undefined],
    );
    // The median of the two unclamped rounds
    assertEquals(summary.ratio?.toFixed(4), ((0.9286 + 0.9) / 2).toFixed(4));
    // min over the arm's passes ÷ reps, in ms; a clamped 0 is never the min
    assertEquals(summary.aMsPerDispatch, 0.9e7 / 10 / 1e6);
    assertEquals(summary.bMsPerDispatch, 0.8e7 / 10 / 1e6);
    // Without GPU timestamps the wall clock gives the per-dispatch times
    const wall = summarizePair("wall", { reps: 10, rounds: [round([1e7, 1e7], [2e7, 2e7])] });
    assertEquals(wall.aMsPerDispatch, (1e7 + 10) / 10 / 1e6);
  });

  test("kernel and producer per node need all four times", () => {
    assertEquals(
      withProducerRatio(
        { onlineNodeMs: 100, tiledNodeMs: 90 },
        { aMsPerDispatch: 2, bMsPerDispatch: 3 },
      ),
      93 / 102,
    );
    assertEquals(withProducerRatio({ onlineNodeMs: 100, tiledNodeMs: 90 }, undefined), undefined);
    assertEquals(
      withProducerRatio({ onlineNodeMs: 100 }, { aMsPerDispatch: 2, bMsPerDispatch: 3 }),
      undefined,
    );
  });
});

const measured = (route: AttentionBenchRoute, where: string): number => {
  assert(route.eligible, `${where}: not eligible (${route.eligible ? "" : route.reason})`);
  assertEquals(route.error, undefined, `${where}: ${route.error}`);
  for (const stage of route.stages) {
    assertEquals(stage.error, undefined, `${where} ${stage.stage}: ${stage.error}`);
    assert((stage.msPerDispatch ?? 0) > 0, `${where} ${stage.stage}: ${stage.msPerDispatch}`);
    assert((stage.reps ?? 0) >= 1);
  }
  assert((route.msPerDispatch ?? 0) > 0);
  assert((route.msPer1000Rows ?? 0) > 0);
  assert((route.tops ?? 0) > 0);
  return route.msPer1000Rows ?? 0;
};

describe.skipIf(!GPU_AVAILABLE)("attention route bench on the GPU", () => {
  // ADR 0132 addendum 2026-10-27: N % 4 != 0 became eligible online (a padded Vᵀ). Before it, the
  // ineligible case was N = 1,022 at D = 128; it is now D = 96 (D % 64 != 0) at the same N.
  test("both routes are timed at small shapes, N % 4 != 0 runs online, and D = 96 is not eligible online", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const timestamps = adapter?.features.has("timestamp-query") === true;
    const gpu = await acquireGpu({
      subgroups: "available",
      ...(timestamps ? { gpuTiming: true } : {}),
    });
    try {
      const statuses: string[] = [];
      const report = await runAttentionRouteBench(gpu, {
        shapes: [
          shape({ id: "n512", keys: 512 }),
          shape({ id: "n1024", keys: 1024, rowsPerDispatch: 128 }),
          shape({ id: "d96n1022", keys: 1022, depth: 96 }),
          shape({ id: "n1022", keys: 1022 }),
        ],
        rounds: 1,
        onProgress: (progress) => {
          if (progress.kind === "status") statuses.push(progress.message);
        },
      });
      assertEquals(report.format, ATTENTION_BENCH_FORMAT);
      assertEquals(report.aborted, false);
      assertEquals(report.timing.unit, timestamps ? "ns" : "wall");
      assertMatch(report.variants.dp4a.dot, /^(dp4a|emu)$/);
      assert(statuses.length > 0);
      assertEquals(
        report.shapes.map(({ shape: { id } }) => id),
        ["n512", "n1024", "d96n1022", "n1022"],
      );

      for (const result of report.shapes.slice(0, 2)) {
        const { id } = result.shape;
        const s16 = measured(result.s16, `${id} s16`);
        const online = measured(result.online, `${id} online`);
        assertEquals(result.s16.eligible ? result.s16.stages.map(({ stage }) => stage) : [], [
          "qk",
          "stats",
          "pv",
        ]);
        assert(result.online.eligible && result.online.reduce !== undefined);
        assert((result.online.workgroupStorageBytes ?? 0) > 0);
        const ratio = result.onlineVsS16 ?? Number.NaN;
        assert(Number.isFinite(ratio) && ratio > 0, `${id}: ratio ${ratio}`);
        assertEquals(ratio, online / s16);
        assertEquals(
          (result.quantization?.stages ?? []).map(({ stage, error }) => [stage, error]),
          [
            ["quantize_q", undefined],
            ["quantize_k", undefined],
            ["permute_v", undefined],
            ["quantize_vt", undefined],
          ],
        );
        assert((result.quantization?.msPerNode ?? 0) > 0);
      }
      // rowsPerDispatch: the s16 block takes it as is (and becomes a row window of the node); the
      // online dispatch rounds it up to whole 64-row workgroups
      const [, windowed] = report.shapes;
      assert(windowed.s16.eligible && windowed.online.eligible);
      assertEquals([windowed.s16.rows, windowed.s16.rowWindow], [128, true]);
      assertEquals(windowed.online.rows, 128);

      const [, , odd, padded] = report.shapes;
      assertEquals(odd.online.eligible, false);
      assert(!odd.online.eligible && odd.online.reason.includes("not eligible"));
      assertEquals(odd.onlineVsS16, undefined);
      assertEquals(odd.quantization, undefined);
      // N % 4 != 0: online with the padded key axis; s16 is not eligible (③PV keeps N % 4 == 0)
      measured(padded.online, "n1022 online");
      assertEquals(padded.s16.eligible, false);
      assertEquals(padded.onlineVsS16, undefined);
      assertEquals(
        (padded.quantization?.stages ?? []).map(({ stage, error }) => [stage, error]),
        [
          ["quantize_q", undefined],
          ["quantize_k", undefined],
          ["permute_v", undefined],
          ["quantize_vt", undefined],
        ],
      );
      // The record is plain JSON (no Infinity / NaN / undefined-only fields lost in meaning)
      assertEquals(JSON.parse(JSON.stringify(report)).format, ATTENTION_BENCH_FORMAT);
      // The candidate is opt-in: nothing of it without the option
      assertEquals(report.settings.onlineTiled, false);
      for (const result of report.shapes) {
        assertEquals(result.onlineTiled, undefined);
        assertEquals(result.quantization?.tiledVt, undefined);
      }
    } finally {
      gpu.destroy();
    }
  });

  test("the candidate onlineTiled is timed with online in the same rounds, its producer beside today's, and its O is identical", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const timestamps = adapter?.features.has("timestamp-query") === true;
    const gpu = await acquireGpu({
      subgroups: "available",
      ...(timestamps ? { gpuTiming: true } : {}),
    });
    try {
      const report = await runAttentionRouteBench(gpu, {
        shapes: [
          shape({ id: "n512", keys: 512 }),
          shape({ id: "n1022", keys: 1022, queries: 200 }),
          shape({ id: "d96n1022", keys: 1022, depth: 96 }),
        ],
        rounds: 2,
        onlineTiled: true,
      });
      assertEquals(report.settings.onlineTiled, true);
      const [whole, padded, odd] = report.shapes;
      for (const result of [whole, padded]) {
        const { id } = result.shape;
        const tiled = result.onlineTiled;
        assert(tiled?.eligible === true, `${id}: the candidate did not run`);
        assert(result.online.eligible);
        assertEquals(tiled.rows, result.online.rows);
        const { kernel } = tiled;
        assertEquals(kernel.error, undefined, `${id}: ${kernel.error}`);
        assertEquals(kernel.stages, ["online", "onlineTiled"]);
        assertMatch(kernel.keys[1], /:ktile$/);
        assertEquals(kernel.keys[0], result.online.stages[0]?.key);
        assertEquals(kernel.rounds?.length, 2);
        assert((kernel.reps ?? 0) >= 1);
        assert((kernel.aMsPerDispatch ?? 0) > 0 && (kernel.bMsPerDispatch ?? 0) > 0);
        assert(Number.isFinite(kernel.ratio ?? Number.NaN), `${id}: ratio ${kernel.ratio}`);
        assertEquals(result.tiledVsOnline, kernel.ratio);
        // The identity block: both producers, both kernels, every O word and scale word equal
        assertEquals(tiled.identity, {
          identical: true,
          words: result.shape.heads * result.shape.queries * result.shape.depth,
          differingWords: 0,
          differingScaleWords: 0,
        });
        // The producer pair in the quantization line, after today's four stages
        const pair = result.quantization?.tiledVt;
        assert(pair !== undefined, `${id}: no producer pair`);
        assertEquals(pair.error, undefined, `${id}: ${pair.error}`);
        assertEquals(pair.stages, ["quantize_vt", "quantize_vt_tiled"]);
        assertEquals(pair.keys[0], result.quantization?.stages[3]?.key);
        assertMatch(pair.keys[1], /:tile64$/);
        assert(Number.isFinite(pair.ratio ?? Number.NaN));
        assert(Number.isFinite(result.tiledVsOnlineWithProducer ?? Number.NaN));
      }
      // Where the online route does not run, neither does the candidate
      assertEquals(odd.onlineTiled?.eligible, false);
      assertEquals(odd.tiledVsOnline, undefined);
      assertEquals(JSON.parse(JSON.stringify(report)).shapes.length, 3);
    } finally {
      gpu.destroy();
    }
  });
});

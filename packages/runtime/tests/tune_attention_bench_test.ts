// The attention route bench (`runAttentionRouteBench` of `@karume/runtime/tune`): the pure helpers,
// the binding-role check with storage params, and one GPU run on small shapes (both routes timed,
// a finite ratio, and a shape the online kernel cannot run reported as not eligible).

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
  onlineDispatchRows,
  positionalBindings,
} from "../src/tune/attention-bench.ts";
import { assertBindingRoles } from "../src/tune/harness.ts";
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
  test("both routes are timed at small shapes and an N % 4 != 0 shape is not eligible online", async () => {
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
        ["n512", "n1024", "n1022"],
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

      const [, , odd] = report.shapes;
      assertEquals(odd.online.eligible, false);
      assert(!odd.online.eligible && odd.online.reason.includes("not eligible"));
      assertEquals(odd.onlineVsS16, undefined);
      assertEquals(odd.quantization, undefined);
      // The record is plain JSON (no Infinity / NaN / undefined-only fields lost in meaning)
      assertEquals(JSON.parse(JSON.stringify(report)).format, ATTENTION_BENCH_FORMAT);
    } finally {
      gpu.destroy();
    }
  });
});

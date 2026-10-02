// 公開面 `@karume/runtime/tune` の入口（ADR 0117 決定 2）の門 — runGeometrySweep（専用の device を取って
// 掃引し、記録を返す）と deriveGeometryProfile（記録から表と採否の行）。入口は公開面（../tune.ts）から
// 引き、突き合わせの相手（CLI / GPU lab が通る生成器の内部）だけを src から引く。GPU の節はアダプタが
// 無い環境で明示 SKIP。

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  deriveGeometryProfile,
  type GeometrySweepCaseSummary,
  type GeometrySweepOptions,
  type GeometrySweepProgress,
  type GeometrySweepReport,
  type GeometrySweepRow,
  runGeometrySweep,
} from "../tune.ts";
import {
  buildGeometryProfile,
  deriveProfile,
  parseSweepReport,
  verdictLines,
} from "../src/tune/derive.ts";
import { timestampUnitFor } from "../src/tune/sweep.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import { geometryProfileKernelsId, sweepCaseSetId } from "../src/tune/fingerprint.ts";
import { A, B, BIG, caseRows, linearCase, report } from "./helpers/sweep-records.ts";

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();
const timestampQuery = adapter !== null && adapter.features.has("timestamp-query");

/**
 * 記録に足した欄（ADR 0117 決定 8）のうち掃引の中身に依らない検査: 開始時刻が ISO 8601 で終了（`date`）以前・
 * ケース集合の版と既定の表のカーネルの指紋が今の runtime の値・ケースごとの壁時計が有限の非負。
 */
const assertRecordFields = (swept: GeometrySweepReport): void => {
  assertEquals(new Date(swept.startedAt).toISOString(), swept.startedAt);
  assert(swept.startedAt <= swept.date, `startedAt ${swept.startedAt} > date ${swept.date}`);
  assertEquals(swept.caseSet, sweepCaseSetId());
  assertEquals(swept.defaultKernels, geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE));
  for (const summary of swept.cases) {
    assert(
      Number.isFinite(summary.elapsedMs) && summary.elapsedMs >= 0,
      `${summary.caseId}: elapsedMs ${summary.elapsedMs}`,
    );
  }
};

describe("deriveGeometryProfile", () => {
  const inputs = [{
    report: report(
      caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }], [B, { speedup: 1.01 }]]),
    ),
    path: "a.json",
    sha256: "sha-a",
  }];

  it("記録から表と採否の行を返し、CLI / GPU lab の経路（deriveProfile → buildGeometryProfile）と同じ値になる", () => {
    const options = { id: "test-gpu", vendor: "apple", architecture: "metal-3" } as const;
    const { profile, verdicts } = deriveGeometryProfile(inputs, options);
    const sources = inputs.map(({ report, path, sha256 }) =>
      parseSweepReport(report, { path, sha256 })
    );
    const internal = deriveProfile(sources, { ...options, minSpeedup: 1.05 });
    assertEquals(profile, buildGeometryProfile(options, sources, internal));
    assertEquals(verdicts, verdictLines(internal));
    assertEquals(profile.gemmRows.at(-1)?.geometry, A);
    assertEquals(profile.provenance?.sweep, "a.json");
  });

  it("minSpeedup の既定は ×1.05（×1.04 の勝ちは採らない）で、1 未満と非有限は投げる", () => {
    const close = [{
      report: report(caseRows(linearCase(1024), BIG, [[A, { speedup: 1.04 }]])),
      path: "a.json",
      sha256: "sha-a",
    }];
    const largest = (minSpeedup?: number) =>
      deriveGeometryProfile(close, {
        id: "x",
        optIn: true,
        ...(minSpeedup === undefined ? {} : { minSpeedup }),
      }).profile.gemmRows.at(-1)?.geometry;
    assertEquals(largest(), BIG);
    assertEquals(largest(1.03), A);
    for (const minSpeedup of [0.99, Number.NaN, Number.POSITIVE_INFINITY]) {
      assertThrows(() => largest(minSpeedup), Error, "minSpeedup は 1 以上");
    }
  });

  it("門に落ちる記録（壁時計で測った掃引）は生成器の理由のまま投げる", () => {
    const wall = [{
      report: {
        ...report(caseRows(linearCase(1024), BIG, [])),
        gpuTiming: { feature: false, unit: "wall", quantized: false },
      },
      path: "wall.json",
      sha256: "sha-wall",
    }];
    assertThrows(
      () => deriveGeometryProfile(wall, { id: "x", optIn: true }),
      Error,
      'wall.json: gpuTiming.unit が "wall"',
    );
  });
});

describe("runGeometrySweep: options の検査（GPU に触れる前）", () => {
  // 型の外から来る値（JS の呼び手・UI の文字列）の門 — JSON.parse の戻りは型検査を経ない
  const untyped = (json: string): GeometrySweepOptions => JSON.parse(json);

  it("知らない候補集合・op・ケース id と、正の整数でない rounds は投げる（既定で走らせない）", async () => {
    await assertRejects(
      () => runGeometrySweep(untyped('{ "candidateSet": "quik" }')),
      Error,
      "候補集合 quik",
    );
    await assertRejects(() => runGeometrySweep(untyped('{ "ops": ["gemv"] }')), Error, "op gemv");
    await assertRejects(
      () => runGeometrySweep({ cases: ["linear-m1-n1-k1"] }),
      Error,
      "ケース linear-m1-n1-k1 が形状表に無い",
    );
    for (const rounds of [0, 1.5]) {
      await assertRejects(() => runGeometrySweep({ rounds }), Error, "rounds は 1 以上の整数");
    }
  });

  it("op とケース id の両方に合うケースが無ければ投げる", async () => {
    await assertRejects(
      () => runGeometrySweep({ ops: ["linear"], cases: ["conv2d-c96-512x512"] }),
      Error,
      "に合うケースが無い",
    );
  });
});

describe("timestampUnitFor", () => {
  it("Deno（wgpu は raw tick を換算しない）は deno-raw-tick、それ以外（Chrome / Dawn）は ns", () => {
    assertEquals(timestampUnitFor("Deno/2.9.6"), "deno-raw-tick");
    assertEquals(
      timestampUnitFor(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
      ),
      "ns",
    );
  });
});

describe("runGeometrySweep: 実 GPU", () => {
  it({
    name: "中断済みの signal なら行を作らず、設定と adapter を載せた記録を返す",
    ignore: adapter === null,
    fn: async () => {
      const kinds: GeometrySweepProgress["kind"][] = [];
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        ops: ["linear"],
        rounds: 1,
        signal: AbortSignal.abort(),
        onProgress: (progress) => kinds.push(progress.kind),
      });
      assertEquals(swept.format, "karume-geometry-sweep/2");
      assertEquals(kinds, ["started"]);
      assertEquals(swept.aborted, true);
      assertRecordFields(swept);
      assertEquals(swept.rows, []);
      assertEquals(swept.cases, []);
      assertEquals(swept.deviceLost, null);
      assertEquals(swept.userAgent, navigator.userAgent);
      assertEquals(swept.adapter.vendor, adapter?.info.vendor);
      assertEquals(swept.settings.candidateSet, "quick");
      assertEquals(swept.settings.quick, true);
      assertEquals(swept.settings.ops, ["linear"]);
      assertEquals(swept.settings.rounds, 1);
      assert(!("cases" in swept.settings), JSON.stringify(swept.settings));
    },
  });

  it({
    name:
      "1 ケースを quick で測ると既定の行が先頭・全行が既定と出力一致し、通知は started → 行 → ケースの順",
    ignore: adapter === null,
    fn: async () => {
      const id = "linear-m16-n3072-k1024";
      const kinds: GeometrySweepProgress["kind"][] = [];
      const notified: GeometrySweepRow[] = [];
      const summaries: GeometrySweepCaseSummary[] = [];
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        cases: [id],
        rounds: 1,
        onProgress: (progress) => {
          if (progress.kind !== "status") kinds.push(progress.kind);
          if (progress.kind === "row") notified.push(progress.row);
          if (progress.kind === "case") summaries.push(progress.summary);
        },
      });
      assertEquals(swept.settings.cases, [id]);
      assertEquals(swept.gpuTiming.unit, timestampQuery ? "deno-raw-tick" : "wall");
      assert(swept.rows.length >= 2, `${swept.rows.length} 行`);
      assertEquals(swept.rows[0].isDefault, true);
      for (const row of swept.rows) {
        assertEquals(row.caseId, id);
        assertEquals(row.error, undefined, row.error);
      }
      for (const row of swept.rows.slice(1)) {
        assertEquals(row.identicalToDefault, true, row.geometry);
      }
      assertEquals(notified, swept.rows);
      assertEquals(kinds, ["started", ...swept.rows.map(() => "row" as const), "case"]);
      assertEquals(swept.cases.map((summary) => summary.caseId), [id]);
      assert((swept.cases[0].defaultRepeat?.driftRatio ?? 0) > 0);
      assertEquals(swept.aborted, false);
      assertRecordFields(swept);
      // 通知の case と記録の cases[] は同じ値（elapsedMs を含む）
      assertEquals(summaries, swept.cases);
    },
  });
});

describe("runGeometrySweep: 実 GPU — 中断の記録（aborted）", () => {
  const id = "linear-m16-n3072-k1024";

  it({
    name:
      "ケースの途中（既定の行の直後）で中断すると aborted が true で、そのケースは再測定を持たず elapsedMs を持つ",
    ignore: adapter === null,
    fn: async () => {
      const abort = new AbortController();
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        cases: [id],
        rounds: 1,
        signal: abort.signal,
        onProgress: (progress) => {
          if (progress.kind === "row") abort.abort();
        },
      });
      assertEquals(swept.aborted, true);
      assertEquals(swept.rows.length, 1);
      assertEquals(swept.rows[0].isDefault, true);
      assertEquals(swept.cases.map((summary) => summary.caseId), [id]);
      assertEquals(swept.cases[0].defaultRepeat, undefined);
      assertRecordFields(swept);
    },
  });

  it({
    name:
      "全て測り終えた後（最後のケースの通知）に届いた中断は数えない（aborted が false — 記録は欠けていない）",
    ignore: adapter === null,
    fn: async () => {
      const abort = new AbortController();
      const swept = await runGeometrySweep({
        candidateSet: "quick",
        cases: [id],
        rounds: 1,
        signal: abort.signal,
        onProgress: (progress) => {
          if (progress.kind === "case") abort.abort();
        },
      });
      assert(abort.signal.aborted);
      assertEquals(swept.aborted, false);
      assert(swept.cases[0].defaultRepeat !== undefined, "再測定まで測り終えていない");
      assertRecordFields(swept);
    },
  });
});

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import type { AnimaRunComponent } from "../../../packages/models/anima.ts";
import type { Row, StageRecord } from "../../anima-residency/record.ts";
import { AB_STAGES, abStatusLine, abTableRows, summarizeAb } from "./ab-summary.ts";

const SHA_X = "a".repeat(64);
const SHA_Y = "b".repeat(64);

type StageTimes = Partial<Record<AnimaRunComponent, { wall: number; gpuNs?: number }>>;

/** 段の記録（`gpuNs` を省くと GPU 時間の無い段 = timestamp を採らなかった走行）。 */
const stagesOf = (times: StageTimes): StageRecord[] => {
  let at = 0;
  return AB_STAGES.flatMap((component) => {
    const time = times[component];
    if (time === undefined) return [];
    const startMs = at;
    at += time.wall;
    return {
      component,
      startMs,
      endMs: at,
      ...(time.gpuNs === undefined
        ? {}
        : { gpu: { totalNs: time.gpuNs, runs: 1, clampedNegativeSamples: 0, entries: [] } }),
      pipelines: [],
    };
  });
};

let serial = 0;

const row = (
  options: {
    wallMs: number;
    sha?: string;
    stages?: StageTimes;
    error?: string;
    requested?: Row["geometryProfileRequested"];
  },
): Row => ({
  index: ++serial,
  quant: "f16",
  geometryProfileRequested: options.requested ?? "default",
  residencyRequested: "transformer",
  request: { prompt: "p", resolution: { width: 512, height: 512 }, seed: 42 },
  dummyBytesHeld: 0,
  wallMs: options.wallMs,
  stages: stagesOf(options.stages ?? {}),
  residency: [],
  ...(options.error === undefined
    ? { pngSha256: options.sha ?? SHA_X }
    : { error: { name: options.error, message: "boom" } }),
});

describe("summarizeAb", () => {
  describe("両区間が同じ PNG を出したとき", () => {
    const summary = summarizeAb(
      [row({ wallMs: 9000 }), row({ wallMs: 4000 })],
      [row({ wallMs: 8000, requested: "auto" }), row({ wallMs: 3000, requested: "auto" })],
    );

    it("区間の中も区間の間も一致と判定する", () => {
      assertEquals(summary.a, {
        kind: "ok",
        count: 2,
        sha: { match: true, distinct: [SHA_X.slice(0, 12)] },
      });
      assertEquals(summary.b.kind === "ok" && summary.b.sha.match, true);
      assertEquals(summary.comparison?.shaMatch, true);
    });

    it("全体の壁時計は 2 回目以降の値で比べる（1 回目の DiT 読み込みを外す）", () => {
      assertEquals(summary.comparison?.wallMs, { a: 4000, b: 3000, ratio: 0.75 });
    });
  });

  describe("PNG が割れたとき", () => {
    it("区間の中の不一致は出た sha を先頭 12 桁で最初に出た順に並べる", () => {
      const summary = summarizeAb(
        [row({ wallMs: 1, sha: SHA_X }), row({ wallMs: 1, sha: SHA_Y }), row({ wallMs: 1 })],
        [row({ wallMs: 1 })],
      );
      assertEquals(summary.a, {
        kind: "ok",
        count: 3,
        sha: { match: false, distinct: [SHA_X.slice(0, 12), SHA_Y.slice(0, 12)] },
      });
      assertEquals(summary.comparison?.shaMatch, false);
    });

    it("区間の中で揃っていても A と B が違えば区間の間は不一致", () => {
      const summary = summarizeAb([row({ wallMs: 1, sha: SHA_X })], [
        row({ wallMs: 1, sha: SHA_Y }),
      ]);
      assertEquals(summary.a.kind === "ok" && summary.a.sha.match, true);
      assertEquals(summary.b.kind === "ok" && summary.b.sha.match, true);
      assertEquals(summary.comparison?.shaMatch, false);
    });
  });

  describe("段ごとの時間", () => {
    it("壁時計と GPU 時間を段ごとに 2 回目以降の中央値で比べる（偶数個は真ん中 2 つの平均）", () => {
      const stages = (dit: number, ditGpu: number): StageTimes => ({
        text_encoder: { wall: 100, gpuNs: 50e6 },
        text_conditioner: { wall: 10, gpuNs: 5e6 },
        transformer: { wall: dit, gpuNs: ditGpu },
        vae_decoder: { wall: 200, gpuNs: 150e6 },
      });
      const summary = summarizeAb(
        [
          row({ wallMs: 1, stages: stages(99999, 99999e6) }),
          row({ wallMs: 1, stages: stages(1000, 900e6) }),
          row({ wallMs: 1, stages: stages(3000, 2700e6) }),
        ],
        [
          row({ wallMs: 1, stages: stages(88888, 88888e6) }),
          row({ wallMs: 1, stages: stages(1000, 900e6) }),
        ],
      );
      const transformer = summary.comparison?.stages.find((stage) =>
        stage.component === "transformer"
      );
      assertEquals(transformer?.wallMs, { a: 2000, b: 1000, ratio: 0.5 });
      assertEquals(transformer?.gpuNs, { a: 1800e6, b: 900e6, ratio: 0.5 });
      assertEquals(summary.comparison?.stages.map((stage) => stage.component), [
        "text_encoder",
        "text_conditioner",
        "transformer",
        "vae_decoder",
      ]);
    });

    it("同じ段が 2 回出た行（OOM 退避のやり直し）はその合計を段の時間にする", () => {
      const doubled: Row = {
        ...row({ wallMs: 1 }),
        stages: [
          ...stagesOf({ text_encoder: { wall: 30, gpuNs: 3 } }),
          ...stagesOf({ text_encoder: { wall: 70, gpuNs: 7 } }),
        ],
      };
      const summary = summarizeAb([doubled], [doubled]);
      const encoder = summary.comparison?.stages[0];
      assertEquals(encoder?.wallMs, { a: 100, b: 100, ratio: 1 });
      assertEquals(encoder?.gpuNs, { a: 10, b: 10, ratio: 1 });
    });
  });

  describe("N = 1 のとき", () => {
    it("1 回目しか無いのでその値で比べる", () => {
      const summary = summarizeAb(
        [row({ wallMs: 5000, stages: { transformer: { wall: 4000, gpuNs: 2e9 } } })],
        [row({ wallMs: 2500, stages: { transformer: { wall: 1000, gpuNs: 1e9 } } })],
      );
      assertEquals(summary.comparison?.wallMs, { a: 5000, b: 2500, ratio: 0.5 });
      assertEquals(summary.comparison?.stages[2].wallMs, { a: 4000, b: 1000, ratio: 0.25 });
    });
  });

  describe("GPU 時間が無いとき", () => {
    it("壁時計だけ比べ、GPU 時間は値も比も出さない", () => {
      const summary = summarizeAb(
        [row({ wallMs: 1, stages: { transformer: { wall: 400 } } })],
        [row({ wallMs: 1, stages: { transformer: { wall: 200 } } })],
      );
      const transformer = summary.comparison?.stages[2];
      assertEquals(transformer?.wallMs, { a: 400, b: 200, ratio: 0.5 });
      assertEquals(transformer?.gpuNs, {});
      assertEquals(abTableRows(summary).find(([title]) => title === "transformer GPU 時間 (ms)"), [
        "transformer GPU 時間 (ms)",
        "—",
        "—",
        "—",
      ]);
    });

    it("片方の区間だけ GPU 時間を持つ行があれば、その区間の値を出さない（欠けた行を黙って外さない）", () => {
      const summary = summarizeAb(
        [
          row({ wallMs: 1, stages: { transformer: { wall: 400, gpuNs: 1 } } }),
          row({ wallMs: 1, stages: { transformer: { wall: 400 } } }),
        ],
        [row({ wallMs: 1, stages: { transformer: { wall: 200, gpuNs: 1 } } })],
      );
      assertEquals(summary.comparison?.stages[2].gpuNs, { b: 1 });
    });
  });

  describe("失敗行があるとき", () => {
    it("区間 A が失敗したら区間 A は失敗・区間 B は回していない・比べない", () => {
      const summary = summarizeAb(
        [row({ wallMs: 1 }), row({ wallMs: 1, error: "GpuOutOfMemoryError" })],
        [],
      );
      assertEquals(summary, {
        a: { kind: "failed", errorName: "GpuOutOfMemoryError" },
        b: { kind: "skipped" },
      });
      assertEquals(abTableRows(summary), [[
        "PNG sha256（区間の中）",
        "失敗 — GpuOutOfMemoryError",
        "回していない",
        "—",
      ]]);
      assertEquals(
        abStatusLine(summary),
        "A/B 未完 — 区間 A: 失敗 — GpuOutOfMemoryError · 区間 B: 回していない",
      );
    });

    it("区間 B が失敗したら区間 B は失敗・比べない", () => {
      const summary = summarizeAb([row({ wallMs: 1 })], [
        row({ wallMs: 1, error: "GpuDeviceLostError" }),
      ]);
      assertEquals(summary.a.kind, "ok");
      assertEquals(summary.b, { kind: "failed", errorName: "GpuDeviceLostError" });
      assertEquals(summary.comparison, undefined);
    });
  });

  it("区間 A の行が無ければ投げる（A/B は区間 A から回す）", () => {
    assertThrows(() => summarizeAb([], []), Error, "区間 A の行が無い");
  });
});

describe("abStatusLine", () => {
  it("sha の判定と、全体と transformer の B ÷ A を 1 行に並べる", () => {
    const summary = summarizeAb(
      [row({ wallMs: 4000, stages: { transformer: { wall: 2000, gpuNs: 2e9 } } })],
      [row({ wallMs: 3000, stages: { transformer: { wall: 1000, gpuNs: 1.5e9 } } })],
    );
    assertEquals(
      abStatusLine(summary),
      "A/B 完了 — sha: 区間 A 一致 · 区間 B 一致 · A と B 一致 · B ÷ A: 全体 ×0.750 · transformer 壁時計 ×0.500 / GPU ×0.750",
    );
  });
});

/**
 * flag-bench の走行順・GPU 積算・要約の単体検証（純関数だけ・GPU 不要）。
 *
 * 見る点:
 *
 * 1. **ABBA** — 1 round = S0..Sk → Sk..S0・各 set は 1 round に 2 回・通し番号と round 番号
 * 2. **GPU 積算** — 種別ごとに分かれ、分母はその種別の run 本数・0 本の種別は欄ごと無い
 * 3. **要約** — 暖機を除いた中央値・基準比・token 列の一致（visit 間 / 基準と）
 * 4. **混ぜない** — 計測 ON と OFF の行が混ざった要約は落とす・計測 ON の行に内訳が欠けたら落とす
 * 5. **census** — decode run のキー別本数を記録し、基準とキー集合が同じ set に `noKeyChange`
 *
 * NOTE: リポの慣習に合わせて `Deno.test`（文脈）+ `t.step`（振る舞い）で書く。
 */

import { assert, assertAlmostEquals, assertEquals, assertFalse, assertThrows } from "@std/assert";
import {
  abbaOrder,
  addRunPipelines,
  addRunTiming,
  emptyRunGpuTally,
  gpuRecord,
  median,
  pipelineCensus,
  type RunRow,
  sameCensus,
  sameKeySet,
  summarizeSets,
} from "./summary.ts";

Deno.test("abbaOrder: 走行順", async (t) => {
  await t.step("1 round = 往路 S0..Sk → 復路 Sk..S0", () => {
    assertEquals(abbaOrder(3, 1).map((visit) => visit.setIndex), [0, 1, 2, 2, 1, 0]);
  });
  await t.step("round を重ねても同じ形を繰り返し、通し番号と round が付く", () => {
    const visits = abbaOrder(2, 2);
    assertEquals(visits.map((visit) => visit.setIndex), [0, 1, 1, 0, 0, 1, 1, 0]);
    assertEquals(visits.map((visit) => visit.visit), [1, 2, 3, 4, 5, 6, 7, 8]);
    assertEquals(visits.map((visit) => visit.round), [1, 1, 1, 1, 2, 2, 2, 2]);
  });
  await t.step("各 set の位置の平均が揃う（線形の漂流が set に偏らない）", () => {
    const visits = abbaOrder(4, 1);
    const mean = (setIndex: number): number => {
      const at = visits.filter((visit) => visit.setIndex === setIndex).map((visit) => visit.visit);
      return at.reduce((sum, value) => sum + value, 0) / at.length;
    };
    for (let setIndex = 1; setIndex < 4; setIndex++) assertEquals(mean(setIndex), mean(0));
  });
  await t.step("set 数・rounds が正の整数でなければ落とす", () => {
    assertThrows(() => abbaOrder(0, 1));
    assertThrows(() => abbaOrder(2, 0));
  });
});

Deno.test("gpuRecord: 1 走行ぶんの GPU 積算", async (t) => {
  await t.step("種別ごとに分かれ、分母はその種別の run 本数", () => {
    const tally = emptyRunGpuTally();
    addRunTiming(tally, "prefill", { totalNs: 8e6, dispatchCount: 400, clampedNegativeSamples: 0 });
    addRunTiming(tally, "decode", { totalNs: 2e6, dispatchCount: 300, clampedNegativeSamples: 1 });
    addRunTiming(tally, "decode", { totalNs: 4e6, dispatchCount: 300, clampedNegativeSamples: 0 });
    const record = gpuRecord(tally);
    assertEquals(record.prefill, {
      runs: 1,
      msPerRun: 8,
      dispatchesPerRun: 400,
      clampedNegativeSamples: 0,
    });
    assertEquals(record.decode, {
      runs: 2,
      msPerRun: 3,
      dispatchesPerRun: 300,
      clampedNegativeSamples: 1,
    });
  });
  await t.step("run が 0 本の種別は欄ごと無い（0 ms と書かない）", () => {
    const tally = emptyRunGpuTally();
    addRunTiming(tally, "decode", { totalNs: 1e6, dispatchCount: 10, clampedNegativeSamples: 0 });
    assertFalse(Object.hasOwn(gpuRecord(tally), "prefill"));
    assertEquals(gpuRecord(emptyRunGpuTally()), {});
  });
});

Deno.test("median", async (t) => {
  await t.step("奇数本は真ん中・偶数本は中央 2 本の平均・入力順に依らない", () => {
    assertEquals(median([3, 1, 2]), 2);
    assertEquals(median([4, 1, 3, 2]), 2.5);
  });
  await t.step("空の列は落とす", () => {
    assertThrows(() => median([]));
  });
});

type RowSpec = Partial<RunRow> & Pick<RunRow, "set">;
const row = (spec: RowSpec): RunRow => ({
  type: "run",
  visit: 1,
  round: 1,
  prompt: "english-list",
  rep: 1,
  warmup: false,
  gpuTiming: false,
  promptTokens: 31,
  delivered: 96,
  stopReason: "max-tokens",
  ttftMs: 100,
  decodeMsPerToken: 20,
  tokensSha256: "aaa",
  ...spec,
});

Deno.test("summarizeSets: 計測 OFF の行", async (t) => {
  const rows: RunRow[] = [
    // 暖機は巨大な値でも中央値に入らない（フォールト注入）。
    row({ set: "ref", rep: 0, warmup: true, decodeMsPerToken: 999, ttftMs: 9999 }),
    row({ set: "ref", decodeMsPerToken: 20, ttftMs: 100 }),
    row({ set: "ref", decodeMsPerToken: 22, ttftMs: 110, visit: 4 }),
    row({
      set: "ref",
      decodeMsPerToken: 21,
      ttftMs: 90,
      prompt: "japanese-list",
      tokensSha256: "bbb",
    }),
    row({ set: "fast", rep: 0, warmup: true, decodeMsPerToken: 999, visit: 2 }),
    row({ set: "fast", decodeMsPerToken: 18, ttftMs: 100, visit: 2 }),
    row({ set: "fast", decodeMsPerToken: 19, ttftMs: 95, visit: 3, tokensSha256: "ccc" }),
    row({
      set: "fast",
      decodeMsPerToken: 17,
      ttftMs: 105,
      visit: 2,
      prompt: "japanese-list",
      tokensSha256: "bbb",
    }),
  ];
  const [reference, fast] = summarizeSets(rows, ["ref", "fast"]);

  await t.step("暖機を除いた中央値", () => {
    assertEquals(reference.measuredRuns, 3);
    assertEquals(reference.decodeMsPerToken, 21);
    assertEquals(reference.ttftMs, 100);
    assertEquals(fast.decodeMsPerToken, 18);
  });
  await t.step("基準比は (値 / 基準 − 1) × 100・基準自身は 0", () => {
    assertEquals(reference.deltaPercent.decodeMsPerToken, 0);
    assertAlmostEquals(fast.deltaPercent.decodeMsPerToken, (18 / 21 - 1) * 100, 1e-9);
  });
  await t.step("計測 OFF では GPU の欄が無い", () => {
    assertFalse(Object.hasOwn(fast, "gpuDecodeMsPerStep"));
    assertFalse(Object.hasOwn(fast.deltaPercent, "gpuDecodeMsPerStep"));
  });
  await t.step("token 列: visit 間の揺れと基準との食い違いを別々に報せる", () => {
    assert(reference.tokensIdenticalAcrossVisits);
    assert(reference.tokensMatchReference);
    // fast の english-list は aaa と ccc の 2 種類 = visit 間で不一致・基準とも不一致。
    assertFalse(fast.tokensIdenticalAcrossVisits);
    assertFalse(fast.tokensMatchReference);
  });
  await t.step("列が揃っていて基準と同じなら両方 true", () => {
    const [, same] = summarizeSets(
      rows.filter((one) => one.tokensSha256 !== "ccc"),
      ["ref", "fast"],
    );
    assert(same.tokensIdenticalAcrossVisits);
    assert(same.tokensMatchReference);
  });
  await t.step("visit 間で揃っていても基準と違えば tokensMatchReference だけが false", () => {
    const shifted = rows.map((one) =>
      one.set === "fast" && one.prompt === "english-list" ? { ...one, tokensSha256: "ddd" } : one
    );
    const [, changed] = summarizeSets(shifted, ["ref", "fast"]);
    assert(changed.tokensIdenticalAcrossVisits);
    assertFalse(changed.tokensMatchReference);
  });
});

Deno.test("summarizeSets: 計測 ON の行", async (t) => {
  const gpu = (decodeMs: number, dispatches: number): RunRow["gpu"] => ({
    prefill: { runs: 1, msPerRun: 30, dispatchesPerRun: 500, clampedNegativeSamples: 0 },
    decode: {
      runs: 95,
      msPerRun: decodeMs,
      dispatchesPerRun: dispatches,
      clampedNegativeSamples: 0,
    },
  });
  const rows: RunRow[] = [
    row({ set: "ref", gpuTiming: true, gpu: gpu(10, 400) }),
    row({ set: "ref", gpuTiming: true, gpu: gpu(12, 400), visit: 4 }),
    row({ set: "fused", gpuTiming: true, gpu: gpu(9, 300), visit: 2 }),
    row({ set: "fused", gpuTiming: true, gpu: gpu(9, 300), visit: 3 }),
  ];

  await t.step("GPU decode ms/step と dispatch/step の中央値と基準比", () => {
    const [reference, fused] = summarizeSets(rows, ["ref", "fused"]);
    assertEquals(reference.gpuDecodeMsPerStep, 11);
    assertEquals(reference.gpuPrefillMsPerRun, 30);
    assertEquals(fused.gpuDispatchesPerStep, 300);
    assertAlmostEquals(
      fused.deltaPercent.gpuDecodeMsPerStep ?? Number.NaN,
      (9 / 11 - 1) * 100,
      1e-9,
    );
    assertAlmostEquals(fused.deltaPercent.gpuDispatchesPerStep ?? Number.NaN, -25, 1e-9);
  });
  await t.step(
    "計測 ON の計測走行に decode の内訳が無ければ落とす（欠けた中央値を出さない）",
    () => {
      const broken = [...rows, row({ set: "fused", gpuTiming: true, visit: 3, rep: 2 })];
      assertThrows(() => summarizeSets(broken, ["ref", "fused"]), Error, "gpu.decode");
    },
  );
  await t.step("計測 ON と OFF の行が混ざっていたら落とす", () => {
    const mixed = [...rows, row({ set: "ref", visit: 5 })];
    assertThrows(() => summarizeSets(mixed, ["ref", "fused"]), Error, "混ざって");
  });
});

Deno.test("summarizeSets: 簿記の破れ", async (t) => {
  await t.step("計測走行の無い set・未知の set の行は落とす", () => {
    assertThrows(
      () => summarizeSets([row({ set: "ref" })], ["ref", "empty"]),
      Error,
      "計測走行が無い",
    );
    assertThrows(() => summarizeSets([row({ set: "other" })], ["ref"]), Error, "未知の set");
  });
});

Deno.test("pipelineCensus / addRunPipelines: decode run のキー別本数", async (t) => {
  await t.step("行をキーの辞書順の キー → 本数 に畳み、同じキーの 2 行は落とす", () => {
    const census = pipelineCensus([
      { key: "rms_norm", dispatchCount: 43 },
      { key: "linear_gemv_parallel_i4_g512:l4", dispatchCount: 126 },
    ]);
    assertEquals(Object.keys(census), ["linear_gemv_parallel_i4_g512:l4", "rms_norm"]);
    assertEquals(census["rms_norm"], 43);
    assertThrows(
      () => pipelineCensus([{ key: "a", dispatchCount: 1 }, { key: "a", dispatchCount: 2 }]),
      Error,
      "2 行",
    );
  });
  await t.step("decode run だけを記録し、prefill の census は器に入れない", () => {
    const tally = emptyRunGpuTally();
    addRunPipelines(tally, "prefill", [{ key: "linear_gemm", dispatchCount: 300 }]);
    assertEquals(tally.decodePipelines, undefined);
    addRunPipelines(tally, "decode", [{ key: "linear_gemv", dispatchCount: 300 }]);
    assertEquals(tally.decodePipelines, { linear_gemv: 300 });
  });
  await t.step("1 生成の中で decode の census が変わったら落とす（本数だけの違いも）", () => {
    const tally = emptyRunGpuTally();
    addRunPipelines(tally, "decode", [{ key: "linear_gemv", dispatchCount: 300 }]);
    addRunPipelines(tally, "decode", [{ key: "linear_gemv", dispatchCount: 300 }]);
    assertThrows(
      () => addRunPipelines(tally, "decode", [{ key: "linear_gemv", dispatchCount: 299 }]),
      Error,
      "変わった",
    );
  });
  await t.step("decode run の census が無い・空なら落とす", () => {
    assertThrows(() => addRunPipelines(emptyRunGpuTally(), "decode", undefined), Error, "空");
    assertThrows(() => addRunPipelines(emptyRunGpuTally(), "decode", []), Error, "空");
  });
  await t.step("sameKeySet は本数を見ず、sameCensus は本数まで見る", () => {
    assert(sameKeySet({ a: 1, b: 2 }, { b: 5, a: 1 }));
    assertFalse(sameCensus({ a: 1, b: 2 }, { b: 5, a: 1 }));
    assertFalse(sameKeySet({ a: 1 }, { a: 1, b: 1 }));
    assertFalse(sameKeySet({ a: 1, b: 1 }, { a: 1, c: 1 }));
  });
});

Deno.test("summarizeSets: 計測 ON の census（appliedKeys / noKeyChange）", async (t) => {
  const gpu: RunRow["gpu"] = {
    prefill: { runs: 1, msPerRun: 30, dispatchesPerRun: 500, clampedNegativeSamples: 0 },
    decode: { runs: 95, msPerRun: 10, dispatchesPerRun: 400, clampedNegativeSamples: 0 },
  };
  const sequential = { linear_gemv_i4_g32: 300, rms_norm: 100 };
  const parallel = { "linear_gemv_parallel_i4_g32:l4": 300, rms_norm: 100 };
  const timed = (set: string, decodePipelines: RunRow["decodePipelines"], visit = 1): RunRow =>
    row({ set, gpuTiming: true, gpu, decodePipelines, visit });
  const rows: RunRow[] = [
    timed("ref", sequential),
    timed("ref", sequential, 4),
    timed("gemvpar", parallel, 2),
    timed("gemvpar", parallel, 3),
    // 適格表に無い形で縮退した想定 — キー集合は基準と同じ（本数だけ違っても no-op 扱い）。
    timed("noop", { linear_gemv_i4_g32: 300, rms_norm: 99 }, 5),
  ];

  await t.step("set ごとに appliedKeys を出し、基準とキー集合が同じ set にだけ noKeyChange", () => {
    const [reference, gemvpar, noop] = summarizeSets(rows, ["ref", "gemvpar", "noop"]);
    assertEquals(reference.appliedKeys, sequential);
    assertEquals(gemvpar.appliedKeys, parallel);
    assertFalse(Object.hasOwn(reference, "noKeyChange"));
    assertEquals(gemvpar.noKeyChange, false);
    assertEquals(noop.noKeyChange, true);
  });
  await t.step("census を記録していない行（計測 OFF・記録前）では欄ごと無い", () => {
    const [reference] = summarizeSets([row({ set: "ref" })], ["ref"]);
    assertFalse(Object.hasOwn(reference, "appliedKeys"));
    assertFalse(Object.hasOwn(reference, "noKeyChange"));
  });
  await t.step("set の中で census が一部の走行にだけ在るなら落とす", () => {
    const broken = [...rows, row({ set: "gemvpar", gpuTiming: true, gpu, visit: 3, rep: 2 })];
    assertThrows(
      () => summarizeSets(broken, ["ref", "gemvpar", "noop"]),
      Error,
      "decodePipelines が無い",
    );
  });
  await t.step("set の中で census が走行ごとに違えば落とす", () => {
    const drifted = [...rows, timed("gemvpar", sequential, 3)];
    assertThrows(() => summarizeSets(drifted, ["ref", "gemvpar", "noop"]), Error, "走行ごとに違う");
  });
  await t.step("census を持つ set と持たない set が混ざっていたら落とす", () => {
    const mixed = [
      timed("ref", sequential),
      row({ set: "gemvpar", gpuTiming: true, gpu, visit: 2 }),
    ];
    assertThrows(() => summarizeSets(mixed, ["ref", "gemvpar"]), Error, "混ざって");
  });
});

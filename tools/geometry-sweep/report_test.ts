import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { compareToDefault, driftOutOfRange, roundsLookQuantized, type SweepRow } from "./report.ts";

const ROW: SweepRow = {
  caseId: "c",
  op: "linear",
  shape: "M1 N1 K1",
  censusCount: 1,
  geometry: "g",
  geometryParams: { regM: 1, regN: 4, wgX: 4, wgY: 16 },
  isDefault: false,
  reps: 2,
  rounds: [400_000, 300_000],
  perDispatch: 150_000,
  outputSha256: "a",
};

describe("compareToDefault", () => {
  it("既定より速い行は 1 より大きい比になり、digest の一致を記録する", () => {
    const row = compareToDefault(ROW, { perDispatch: 300_000, outputSha256: "a" });
    assertEquals(row.speedupVsDefault, 2);
    assertEquals(row.identicalToDefault, true);
  });

  it("digest が違えば不一致を記録する（幾何が値を変えた — 採用不可の印）", () => {
    assertEquals(
      compareToDefault(ROW, { perDispatch: 150_000, outputSha256: "b" }).identicalToDefault,
      false,
    );
  });

  it("既定の行が失敗していれば比較を足さない（不一致と読ませない）", () => {
    const row = compareToDefault(ROW, {});
    assertEquals(row.speedupVsDefault, undefined);
    assertEquals(row.identicalToDefault, undefined);
  });
});

describe("roundsLookQuantized", () => {
  it("単位 ns で全 round が 100 µs の倍数なら量子化の疑いにする", () => {
    assertEquals(roundsLookQuantized([ROW], "ns"), true);
    assertEquals(roundsLookQuantized([{ ...ROW, rounds: [400_000, 300_001] }], "ns"), false);
  });

  it("raw tick と壁時計は判定しない", () => {
    assertEquals(roundsLookQuantized([ROW], "deno-raw-tick"), false);
    assertEquals(roundsLookQuantized([ROW], "wall"), false);
  });
});

describe("driftOutOfRange", () => {
  it("既定の再測定の比が 0.9〜1.1 の内なら警告しない（境界を含む）", () => {
    for (const ratio of [0.9, 1, 1.1]) assertEquals(driftOutOfRange(ratio), false, String(ratio));
  });

  it("範囲の外と、比が数でないときは測り直しの目安として警告する", () => {
    for (const ratio of [0.89, 1.11, Number.NaN]) {
      assertEquals(driftOutOfRange(ratio), true, String(ratio));
    }
  });
});

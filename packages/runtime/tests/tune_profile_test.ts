// 幾何プロファイルの生成器（src/tune/derive.ts の deriveProfile / parseSweepReport / profileRowsSegments）の
// 門。GPU を使わない — 掃引の記録は小さな合成 JSON で作る（helpers/sweep-records.ts）。
//
// 固定するのは ① 規則の抽出（既定比の門・出力一致の門・クラスの全ケースで勝つこと・幾何平均での
// 比較・掃引にケースが無いクラスの既定）② adapter を混ぜないこと。生成物の TS の描画と CLI
// （`main.ts profile --check`）は tools/geometry-sweep/profile_test.ts が固定する。

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import { GEMM_ROWS_BUCKETS, type GemmGeometry } from "../src/kernels/gemm-geometry.ts";
import { defaultI8a8Geometry, type I8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { PROFILE_GEMM_ROWS_BOUNDS } from "../src/tune/cases.ts";
import { conv2dCandidate, gemmCandidate, i8a8Candidate } from "../src/tune/geometries.ts";
import {
  deriveProfile,
  parseSweepReport,
  profileRowsSegments,
  type SlotVerdict,
  type SweepSource,
  verdictLines,
} from "../src/tune/derive.ts";
import {
  A,
  ADAPTER,
  B,
  BIG,
  bmmCase,
  type CaseRef,
  caseRows,
  linearCase,
  matmulCase,
  OPTIONS,
  rejectionOf,
  type Repeats,
  report,
  row,
  ROWS_SLOT_NAMES,
  SMALL,
  source,
  verdictOf,
} from "./helpers/sweep-records.ts";

describe("deriveProfile: 規則の抽出", () => {
  it("既定比が --min-speedup 未満の幾何しか無いクラスは、掃引の既定の行の幾何を書く", () => {
    const verdicts = deriveProfile([
      source([
        ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.04 }]]),
        ...caseRows(linearCase(4096), BIG, [[A, { speedup: 1.3 }]]),
      ]),
    ], OPTIONS);
    const verdict = verdictOf(verdicts, "gemmRows > 512");
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertStringIncludes(rejectionOf(verdict, A), "linear-m1024 で ×1.040 < ×1.050");
  });

  it("出力が既定と不一致の幾何は、最も速くても採らない", () => {
    const verdict = verdictOf(
      deriveProfile([
        source(caseRows(linearCase(1024), BIG, [
          [A, { speedup: 2, identical: false }],
          [B, { speedup: 1.3 }],
        ])),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(rejectionOf(verdict, A), "不一致");
  });

  it("クラスの 1 ケースでも既定比の門を越えない幾何は、幾何平均が最大でも採らない", () => {
    const verdict = verdictOf(
      deriveProfile([
        source([
          ...caseRows(linearCase(1024), BIG, [[A, { speedup: 3 }], [B, { speedup: 1.2 }]]),
          ...caseRows(linearCase(4096), BIG, [[A, { speedup: 1.01 }], [B, { speedup: 1.2 }]]),
        ]),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(rejectionOf(verdict, A), "linear-m4096 で ×1.010");
  });

  it("クラスのどれかのケースで測っていない・失敗した幾何は採らない", () => {
    const verdict = verdictOf(
      deriveProfile([
        source([
          ...caseRows(linearCase(1024), BIG, [[A, { speedup: 2 }], [B, { speedup: 1.5 }]]),
          ...caseRows(linearCase(4096), BIG, [[B, { speedup: 1.5, error: "device lost" }]]),
        ]),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "default");
    assertStringIncludes(rejectionOf(verdict, A), "linear-m4096 で測っていない");
    assertStringIncludes(rejectionOf(verdict, B), "linear-m4096 で失敗（device lost）");
  });

  it("候補どうしは幾何平均で比べる（算術平均なら逆を選ぶ組）", () => {
    // A: (1.6, 1.06) 算術 1.33・幾何 1.302 / B: (1.31, 1.31) 算術 1.31・幾何 1.31
    const verdict = verdictOf(
      deriveProfile([
        source([
          ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.6 }], [B, { speedup: 1.31 }]]),
          ...caseRows(linearCase(4096), BIG, [[A, { speedup: 1.06 }], [B, { speedup: 1.31 }]]),
        ]),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(rejectionOf(verdict, A), "幾何平均 ×1.302（採用 ×1.310 に届かない）");
  });

  it("行数は段の境界 ±1 で正しい段に入り（rows <= maxRows で最初に当たる段）、conv2d は既定の行の tileM で分ける", () => {
    // 既定の行は M を覆う既定の段の幾何（掃引は既定の 3 段の表で組む）— 段の既定がずれると生成が止まる
    const defaultFor = (m: number): GemmGeometry => m <= 64 ? SMALL : m <= 512 ? A : BIG;
    const expected: readonly (readonly [number, string])[] = [
      [16, "gemmRows ≤ 16"],
      [17, "gemmRows 17〜32"],
      [32, "gemmRows 17〜32"],
      [33, "gemmRows 33〜64"],
      [64, "gemmRows 33〜64"],
      [65, "gemmRows 65〜128"],
      [128, "gemmRows 65〜128"],
      [129, "gemmRows 129〜256"],
      [256, "gemmRows 129〜256"],
      [257, "gemmRows 257〜512"],
      [512, "gemmRows 257〜512"],
      [513, "gemmRows > 512"],
    ];
    const conv = (channels: number, mTile: number): CaseRef => ({
      caseId: `conv2d-c${channels}-m${mTile}`,
      op: "conv2d",
      shape: `Cin${channels} Cout${channels} 8x8 3x3`,
    });
    const conv2dRows = (ref: CaseRef, mTile: number): Record<string, unknown>[] => [
      row(ref, conv2dCandidate(gemmMTileGeometry(mTile)), true),
      row(ref, conv2dCandidate(B), false, { speedup: 1.4, identical: true }),
    ];
    const verdicts = deriveProfile([
      source([
        ...expected.flatMap(([m]) =>
          caseRows(
            linearCase(m),
            defaultFor(m),
            m === 33 || m === 64 ? [[B, { speedup: 1.2 }]] : [],
          )
        ),
        ...conv2dRows(conv(96, 32), 32),
        ...conv2dRows(conv(192, 64), 64),
      ]),
    ], OPTIONS);
    for (const slot of ROWS_SLOT_NAMES) {
      const cases = expected.filter(([, name]) => name === slot).map(([m]) => `linear-m${m}`);
      assertEquals(verdictOf(verdicts, slot).cases, cases.sort(), slot);
    }
    assertEquals(verdictOf(verdicts, "gemmRows 33〜64").outcome.geometry, B);
    assertEquals(verdictOf(verdicts, "conv2d.rows32").cases, ["conv2d-c96-m32"]);
    assertEquals(verdictOf(verdicts, "conv2d.rows64").cases, ["conv2d-c192-m64"]);
  });

  it("matmul / bmm の行は shape の M で gemmRows の欄に入る（bmm はバッチ数に依らず行列 1 枚の M）", () => {
    const verdicts = deriveProfile([
      source([
        ...caseRows(matmulCase(16), SMALL, []),
        ...caseRows(bmmCase(600, 17), SMALL, []),
        ...caseRows(matmulCase(64), SMALL, []),
        ...caseRows(bmmCase(600, 64), SMALL, []),
        ...caseRows(matmulCase(65), A, []),
        ...caseRows(bmmCase(16, 512), A, []),
        ...caseRows(matmulCase(512), A, []),
        ...caseRows(matmulCase(4096), BIG, []),
      ]),
    ], OPTIONS);
    assertEquals(verdictOf(verdicts, "gemmRows ≤ 16").cases, ["matmul-m16"]);
    assertEquals(verdictOf(verdicts, "gemmRows 17〜32").cases, ["bmm-b600-m17"]);
    assertEquals(verdictOf(verdicts, "gemmRows 33〜64").cases, ["bmm-b600-m64", "matmul-m64"]);
    assertEquals(verdictOf(verdicts, "gemmRows 65〜128").cases, ["matmul-m65"]);
    assertEquals(verdictOf(verdicts, "gemmRows 257〜512").cases, ["bmm-b16-m512", "matmul-m512"]);
    assertEquals(verdictOf(verdicts, "gemmRows > 512").cases, ["matmul-m4096"]);
  });

  it("matmul / bmm の観測も欄の全ケースに数える（そこで ×1.05 未満なら linear で速くても採らない）", () => {
    const verdicts = deriveProfile([
      source([
        ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.5 }], [B, { speedup: 1.2 }]]),
        ...caseRows(matmulCase(4096), BIG, [[A, { speedup: 1.02 }], [B, { speedup: 1.2 }]]),
        ...caseRows(linearCase(512), A, [[B, { speedup: 1.5 }]]),
        ...caseRows(bmmCase(16, 512), A, [[B, { speedup: 1.04 }]]),
      ]),
    ], OPTIONS);
    const large = verdictOf(verdicts, "gemmRows > 512");
    assertEquals(large.cases, ["linear-m1024", "matmul-m4096"]);
    assertEquals(large.outcome.kind, "adopted");
    assertEquals(large.outcome.geometry, B);
    assertStringIncludes(rejectionOf(large, A), "matmul-m4096 で ×1.020 < ×1.050");
    const middle = verdictOf(verdicts, "gemmRows 257〜512");
    assertEquals(middle.outcome.kind, "default");
    assertEquals(middle.outcome.geometry, A);
    assertStringIncludes(rejectionOf(middle, B), "bmm-b16-m512 で ×1.040 < ×1.050");
  });

  it("掃引にケースが無いクラスは runtime の既定を書き、理由を残す", () => {
    const verdicts = deriveProfile([source(caseRows(linearCase(1024), BIG, []))], OPTIONS);
    const conv = verdictOf(verdicts, "conv2d.rows64");
    assertEquals(conv.outcome.geometry, gemmMTileGeometry(64));
    assert(conv.outcome.kind === "default");
    assertEquals(conv.outcome.reason, "掃引にこのクラスのケースが無い");
    assertEquals(
      verdictOf(verdicts, "i8a8.attentionPv").outcome.geometry,
      defaultI8a8Geometry("attention_pv"),
    );
  });

  it("同じケースを複数の掃引が測れば比はその幾何平均、1 本でも不一致なら採らない", () => {
    const ref = linearCase(1024);
    const verdict = verdictOf(
      deriveProfile([
        source(caseRows(ref, BIG, [[A, { speedup: 1.2 }], [B, { speedup: 1.5 }]]), "quick.json"),
        source(
          caseRows(ref, BIG, [[A, { speedup: 1 }], [B, { speedup: 1.5, identical: false }]]),
          "full.json",
        ),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean.toFixed(4), Math.sqrt(1.2).toFixed(4));
    assertStringIncludes(rejectionOf(verdict, B), "不一致");
  });

  it("i8a8 は tileK 込みの幾何を op と段ごとに選ぶ", () => {
    const tile: I8a8Geometry = { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 };
    const ref: CaseRef = {
      caseId: "i8a8-attention-pv",
      op: "i8a8-attention",
      shape: "pv BH16 M64 N64 D128",
    };
    const verdicts = deriveProfile([
      source([
        row(ref, i8a8Candidate(defaultI8a8Geometry("attention_pv")), true),
        row(ref, i8a8Candidate(tile), false, { speedup: 1.1, identical: true }),
      ]),
    ], OPTIONS);
    assertEquals(verdictOf(verdicts, "i8a8.attentionPv").outcome.geometry, tile);
    assertEquals(
      verdictOf(verdicts, "i8a8.attentionQk").outcome.geometry,
      defaultI8a8Geometry("attention_qk"),
    );
  });
});

describe("deriveProfile: 既定の再測定比が範囲外のケースは、その掃引の材料から外す", () => {
  const ref = linearCase(1024);
  const wide = linearCase(4096);
  const drift = (caseId: string, driftRatio: number): Record<string, unknown> => ({
    caseId,
    defaultRepeat: { perDispatch: 1, driftRatio },
  });
  /** `cases[]` を指定した掃引（指定しないケースは比 1）。 */
  const sweepWith = (
    rows: readonly Record<string, unknown>[],
    name: string,
    repeats: Repeats,
  ): SweepSource =>
    parseSweepReport(report(rows, ADAPTER, repeats), { path: name, sha256: `sha-${name}` });

  it("範囲外の掃引の観測は比に数えず、他の掃引の同じケースで判定する", () => {
    // 外さなければ A の比は √(0.5 × 1.3) ≈ ×0.806 で退けられる
    const drifted = sweepWith(caseRows(ref, BIG, [[A, { speedup: 0.5 }]]), "drifted.json", {
      [ref.caseId]: drift(ref.caseId, 1.2),
    });
    const steady = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "steady.json");
    const verdict = verdictOf(deriveProfile([drifted, steady], OPTIONS), "gemmRows > 512");
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean, 1.3);
    assertEquals(verdict.cases, [ref.caseId]);
    assertEquals(verdict.excluded, [
      { path: "drifted.json", caseId: ref.caseId, reason: "既定の再測定比 ×1.200 が範囲外" },
    ]);
  });

  it("外した掃引の出力の不一致と失敗は、他の掃引で一致・成功していても候補を落とす", () => {
    // 外すのは比だけ — 出力の一致は正しさの門で熱に依らない
    const drifted = sweepWith(
      caseRows(ref, BIG, [
        [A, { speedup: 1.3, identical: false }],
        [B, { speedup: 1.3, error: "device lost" }],
      ]),
      "drifted.json",
      { [ref.caseId]: drift(ref.caseId, 1.2) },
    );
    const steady = source(
      caseRows(ref, BIG, [[A, { speedup: 1.3 }], [B, { speedup: 1.3 }]]),
      "steady.json",
    );
    const verdict = verdictOf(deriveProfile([drifted, steady], OPTIONS), "gemmRows > 512");
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(rejectionOf(verdict, A), `${ref.caseId} で出力が既定と不一致`);
    assertEquals(rejectionOf(verdict, B), `${ref.caseId} で失敗（device lost）`);
  });

  it("全掃引で外したケースは測っていない扱いで、欄は既定のまま（ケースの一覧には残る）", () => {
    const rows = [
      ...caseRows(ref, BIG, [[A, { speedup: 1.3 }]]),
      ...caseRows(wide, BIG, [[A, { speedup: 1.3 }]]),
    ];
    const verdict = verdictOf(
      deriveProfile([
        sweepWith(rows, "first.json", { [wide.caseId]: drift(wide.caseId, 1.15) }),
        sweepWith(rows, "second.json", { [wide.caseId]: drift(wide.caseId, 0.8) }),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.cases, [ref.caseId, wide.caseId]);
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(
      rejectionOf(verdict, A),
      `${wide.caseId} で測っていない（全掃引で比の材料から外した）`,
    );
    assertEquals(verdict.excluded.map((entry) => [entry.path, entry.caseId]), [
      ["first.json", wide.caseId],
      ["second.json", wide.caseId],
    ]);
  });

  it("クラスの全ケースを外したら、欄は既定でその理由を残す", () => {
    const verdict = verdictOf(
      deriveProfile([
        sweepWith(caseRows(ref, BIG, [[A, { speedup: 1.5 }]]), "sweep.json", {
          [ref.caseId]: drift(ref.caseId, 1.5),
        }),
      ], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(verdict.outcome.reason, "クラスの全ケースを全掃引で比の材料から外した");
  });

  it("ケースが 1 本の段でそれを全掃引で外したら、速い幾何があっても段の既定のまま", () => {
    const only = linearCase(128);
    const rows = caseRows(only, A, [[B, { speedup: 1.5 }]]);
    const verdict = verdictOf(
      deriveProfile([
        sweepWith(rows, "first.json", { [only.caseId]: drift(only.caseId, 1.15) }),
        sweepWith(rows, "second.json", { [only.caseId]: drift(only.caseId, 0.8) }),
      ], OPTIONS),
      "gemmRows 65〜128",
    );
    assertEquals(verdict.cases, [only.caseId]);
    assert(verdict.outcome.kind === "default");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.reason, "クラスの全ケースを全掃引で比の材料から外した");
    assertEquals(
      rejectionOf(verdict, B),
      `${only.caseId} で測っていない（全掃引で比の材料から外した）`,
    );
  });

  it("範囲の両端（×0.9 / ×1.1）は外さず、再測定の失敗・欄なし・ケースの記録なしは外す", () => {
    const cases = [900, 1100, 2048, 3072, 4096].map(linearCase);
    const built = report(
      cases.flatMap((entry) => caseRows(entry, BIG, [[A, { speedup: 1.2 }]])),
      ADAPTER,
      {
        [cases[0].caseId]: drift(cases[0].caseId, 0.9),
        [cases[1].caseId]: drift(cases[1].caseId, 1.1),
        [cases[2].caseId]: { caseId: cases[2].caseId, defaultRepeatError: "device lost" },
        [cases[3].caseId]: { caseId: cases[3].caseId },
      },
    );
    // cases[4] はケースの記録（cases[] の要素）ごと無い — 掃引が途中で止まった記録の形
    const truncated = {
      ...built,
      cases: (built.cases as readonly Record<string, unknown>[])
        .filter((entry) => entry.caseId !== cases[4].caseId),
    };
    const verdict = verdictOf(
      deriveProfile([parseSweepReport(truncated, { path: "sweep.json", sha256: "0" })], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.excluded.map(({ caseId, reason }) => [caseId, reason]), [
      [cases[2].caseId, "既定の再測定が失敗（device lost）"],
      [cases[3].caseId, "既定の再測定が無い"],
      [cases[4].caseId, "既定の再測定が無い"],
    ]);
  });
});

describe("deriveProfile: 量子化した timestamp は観測ごとの丸め誤差の上界で判定する", () => {
  const ref = linearCase(1024);
  /** 100 µs（Chrome の刻み）— rounds は全てこの倍数で書く。 */
  const QUANTUM = 100_000;
  /** 量子化した掃引（`gpuTiming.quantized` true・既定の再測定比は 1）。 */
  const quantizedSource = (rows: readonly Record<string, unknown>[], name: string): SweepSource =>
    parseSweepReport({
      ...report(rows),
      gpuTiming: { feature: true, unit: "ns", quantized: true },
    }, { path: name, sha256: `sha-${name}` });
  /** min の round が `ms` ミリ秒の rounds（e = 100 µs ÷ min）。 */
  const roundsOf = (
    ms: number,
  ): readonly number[] => [ms * 10 * QUANTUM, ms * 10 * QUANTUM + QUANTUM];
  const lines = (verdict: SlotVerdict): string[] =>
    verdictLines([verdict]).filter((line) => line.startsWith("  - 掃引 "));

  it("E ≤ 1% の観測だけなら、量子化した掃引から採否が出て何も外さない", () => {
    // e(既定) = 0.1 / 100 = 0.1%・e(A) = 0.1 / 60 ≈ 0.17% → E ≈ 0.27%
    const rows = caseRows(ref, BIG, [[A, { speedup: 1.3, rounds: roundsOf(60), reps: 3 }]], {
      rounds: roundsOf(100),
      reps: 5,
    });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean, 1.3);
    assertEquals(verdict.excluded, []);
  });

  it("E > 1% の観測だけを外し、同じケースの他の幾何は残す（E は既定の行の e も足す）", () => {
    // e(既定) = 0.1 / 20 = 0.5%。A: e = 0.1 / 12.5 = 0.8% で E = 1.3% → 外す（e だけなら 1% 以下）。
    // B: e = 0.1 / 50 = 0.2% で E = 0.7% → 残す。外さなければ A（×1.5）が B（×1.2）に勝つ
    const rows = caseRows(ref, BIG, [
      [A, { speedup: 1.5, rounds: roundsOf(12.5) }],
      [B, { speedup: 1.2, rounds: roundsOf(50) }],
    ], { rounds: roundsOf(20) });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertEquals(rejectionOf(verdict, A), `${ref.caseId} で測っていない`);
    const name = gemmCandidate(A).name;
    assertEquals(verdict.excluded, [
      {
        path: "q.json",
        caseId: ref.caseId,
        geometry: name,
        reason: "丸め誤差の上界 E 1.30%（> 1%）",
      },
    ]);
    assertEquals(lines(verdict), [
      `  - 掃引 q.json: ${ref.caseId} の ${name} は丸め誤差の上界 E 1.30%（> 1%）のため比の材料から外した（出力の一致と失敗は見る）`,
    ]);
  });

  it("E がちょうど 1% の観測は残す（門は E > 1% で外す）", () => {
    // e(既定) = 0.1 / 20 = 0.5%・A: e = 0.1 / 20 = 0.5% で E = 1.00% → 残す
    const rows = caseRows(ref, BIG, [
      [A, { speedup: 1.5, rounds: roundsOf(20) }],
    ], { rounds: roundsOf(20) });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.excluded, []);
  });

  it("外した観測の出力の不一致は、判定に効く（正しさの門は丸めに依らない）", () => {
    const rows = caseRows(ref, BIG, [
      [A, { speedup: 1.5, identical: false, rounds: roundsOf(2) }],
    ], { rounds: roundsOf(100) });
    const steady = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "steady.json");
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json"), steady], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.outcome.kind, "default");
    assertEquals(rejectionOf(verdict, A), `${ref.caseId} で出力が既定と不一致`);
    assertEquals(verdict.excluded.map((entry) => entry.geometry), [gemmCandidate(A).name]);
  });

  it("既定の行の e が 1% を超えるケースは全観測を外し、採否の行には 1 行だけ載る", () => {
    // e(既定) = 0.1 / 5 = 2% → どの幾何も E > 1%（A・B とも e は小さい）
    const rows = caseRows(ref, BIG, [
      [A, { speedup: 1.5, rounds: roundsOf(80) }],
      [B, { speedup: 1.2, rounds: roundsOf(80) }],
    ], { rounds: roundsOf(5) });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "default");
    assertEquals(verdict.outcome.geometry, BIG);
    assertEquals(verdict.outcome.reason, "クラスの全ケースを全掃引で比の材料から外した");
    assertEquals(verdict.excluded, [
      {
        path: "q.json",
        caseId: ref.caseId,
        reason: "既定の行の丸め誤差の上界 e 2.00%（> 1%）",
      },
    ]);
    assertEquals(lines(verdict), [
      `  - 掃引 q.json: ${ref.caseId} は既定の行の丸め誤差の上界 e 2.00%（> 1%）のため比の材料から外した（出力の一致と失敗は見る）`,
    ]);
  });

  it("同じ (ケース, 幾何) を他の掃引が測っていれば、そちらの比で判定する（quick と full の重ね）", () => {
    // quick（量子化）の A は E > 1% で外す — 数えれば √(0.5 × 1.3) ≈ ×0.806 で退けられる
    const quick = quantizedSource(
      caseRows(ref, BIG, [[A, { speedup: 0.5, rounds: roundsOf(2) }]], { rounds: roundsOf(100) }),
      "quick.json",
    );
    const full = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "full.json");
    const verdict = verdictOf(deriveProfile([quick, full], OPTIONS), "gemmRows > 512");
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.outcome.geomean, 1.3);
    assertEquals(verdict.excluded.map(({ path, geometry }) => [path, geometry]), [
      ["quick.json", gemmCandidate(A).name],
    ]);
  });

  it("rounds が無い・正の round が無い行は上界を出せないので外す（既定の行ならケースごと）", () => {
    const wide = linearCase(4096);
    const rows = [
      ...caseRows(ref, BIG, [
        [A, { speedup: 1.3 }],
        [B, { speedup: 1.3, rounds: [0, 0] }],
      ], { rounds: roundsOf(100) }),
      ...caseRows(wide, BIG, [[A, { speedup: 1.3, rounds: roundsOf(80) }]]),
    ];
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(
      verdict.excluded.map(({ caseId, geometry, reason }) => [caseId, geometry, reason]),
      [
        [ref.caseId, gemmCandidate(A).name, "丸め誤差の上界が不明（rounds が無い）"],
        [ref.caseId, gemmCandidate(B).name, "丸め誤差の上界が不明（正の round が無い）"],
        [wide.caseId, undefined, "既定の行の丸め誤差の上界が不明（rounds が無い）"],
      ],
    );
  });

  it("失敗した行は比を持たないので上界を見ず（外さず）、失敗として退ける", () => {
    const rows = caseRows(ref, BIG, [[A, { error: "device lost" }]], { rounds: roundsOf(100) });
    const verdict = verdictOf(
      deriveProfile([quantizedSource(rows, "q.json")], OPTIONS),
      "gemmRows > 512",
    );
    assertEquals(verdict.excluded, []);
    assertEquals(rejectionOf(verdict, A), `${ref.caseId} で失敗（device lost）`);
  });

  it("量子化していない掃引は rounds を見ない（rounds の無い記録も、短い pass も外さない）", () => {
    const rows = caseRows(ref, BIG, [
      [A, { speedup: 1.3, rounds: [1_000] }],
      [B, { speedup: 1.2 }],
    ]);
    const verdict = verdictOf(
      deriveProfile([source(rows, "plain.json")], OPTIONS),
      "gemmRows > 512",
    );
    assert(verdict.outcome.kind === "adopted");
    assertEquals(verdict.outcome.geometry, A);
    assertEquals(verdict.excluded, []);
  });
});

describe("parseSweepReport: 既定の再測定（cases[]）", () => {
  const rows = caseRows(linearCase(1024), BIG, []);
  const meta = { path: "sweep.json", sha256: "0" };

  it("cases が無い記録は落とす（黙って全ケースを外さない）", () => {
    const { cases: _, ...withoutCases } = report(rows);
    assertThrows(() => parseSweepReport(withoutCases, meta), Error, "cases が配列でない");
  });

  it("同じケースの記録が 2 本あれば落とす", () => {
    const entry = { caseId: "linear-m1024", defaultRepeat: { perDispatch: 1, driftRatio: 1 } };
    assertThrows(
      () => parseSweepReport({ ...report(rows), cases: [entry, entry] }, meta),
      Error,
      "linear-m1024 が 2 本",
    );
  });
});

describe("deriveProfile: description で adapter の機種まで揃える", () => {
  const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);

  it("description を指定したら、adapter の description が違う掃引は落ちる", () => {
    assertThrows(
      () =>
        deriveProfile([source(rows, "m5.json", { ...ADAPTER, description: "Apple M5" })], {
          ...OPTIONS,
          description: "Test GPU",
        }),
      Error,
      "合わない",
    );
    deriveProfile([source(rows)], { ...OPTIONS, description: "Test GPU" });
  });

  it("description は architecture と組でだけ・空文字は落ちる（runtime の門と同じ条件）", () => {
    assertThrows(
      () =>
        deriveProfile([source(rows)], {
          vendor: "apple",
          description: "Test GPU",
          minSpeedup: 1.05,
        }),
      Error,
      "architecture の両方と組",
    );
    assertThrows(
      () => deriveProfile([source(rows)], { ...OPTIONS, description: "" }),
      Error,
      "空文字",
    );
  });

  it("掃引どうしは description まで揃える（--opt-in・--description 省略でも別機種を混ぜない）", () => {
    const m2 = source(rows, "m2.json", { ...ADAPTER, description: "Apple M2" });
    const m5 = source(rows, "m5.json", { ...ADAPTER, description: "Apple M5" });
    const blank = source(rows, "blank.json", { ...ADAPTER, description: "" });
    for (const target of [{ optIn: true } as const, OPTIONS]) {
      assertThrows(
        () => deriveProfile([m2, m5], { ...target, minSpeedup: 1.05 }),
        Error,
        "と違う",
      );
      assertThrows(
        () => deriveProfile([m2, blank], { ...target, minSpeedup: 1.05 }),
        Error,
        "と違う",
      );
    }
    deriveProfile([blank, source(rows, "blank2.json", { ...ADAPTER, description: "" })], OPTIONS);
  });

  it("--opt-in は vendor / architecture を照合しない（掃引どうしの adapter は揃える）", () => {
    deriveProfile([source(rows)], { optIn: true, minSpeedup: 1.05 });
    assertThrows(
      () =>
        deriveProfile([
          source(rows, "m2.json"),
          source(rows, "other.json", { ...ADAPTER, architecture: "metal-2" }),
        ], { optIn: true, minSpeedup: 1.05 }),
      Error,
      "と違う",
    );
  });
});

describe("deriveProfile: runtime の既定プロファイルとの整合", () => {
  it("ケースが無いクラスに書く既定: gemmRows の段はその範囲を覆う既定の段の幾何、他の欄は DEFAULT_GEOMETRY_PROFILE と同じ（14 欄・この順）", () => {
    const profile = DEFAULT_GEOMETRY_PROFILE;
    // 既定の表は 3 段のまま（≤ 64 / 65〜512 / > 512 — ADR 0116 決定 3）
    assertEquals(profile.gemmRows.map((rule) => rule.maxRows), [64, 512, Infinity]);
    const [small, middle, large] = profile.gemmRows.map((rule) => rule.geometry);
    const expected: Readonly<Record<string, GemmGeometry | I8a8Geometry>> = {
      "gemmRows ≤ 16": small,
      "gemmRows 17〜32": small,
      "gemmRows 33〜64": small,
      "gemmRows 65〜128": middle,
      "gemmRows 129〜256": middle,
      "gemmRows 257〜512": middle,
      "gemmRows > 512": large,
      "attention.qk": profile.attention.qk,
      "attention.pv": profile.attention.pv,
      "conv2d.rows64": profile.conv2d.rows64,
      "conv2d.rows32": profile.conv2d.rows32,
      "i8a8.linear": profile.i8a8.linear,
      "i8a8.attentionQk": profile.i8a8.attentionQk,
      "i8a8.attentionPv": profile.i8a8.attentionPv,
    };
    const verdicts = deriveProfile([source([])], OPTIONS);
    assertEquals(verdicts.map((verdict) => verdict.slot), Object.keys(expected));
    for (const verdict of verdicts) {
      assertEquals(verdict.outcome.geometry, expected[verdict.slot], verdict.slot);
    }
  });
});

describe("profileRowsSegments: gemmRows の段の境界の検査", () => {
  it("生成器の境界（PROFILE_GEMM_ROWS_BOUNDS）は 7 段で、欄名と効く範囲を行数の範囲で綴る（添字で綴らない）", () => {
    const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
    assertEquals(segments.map((segment) => segment.maxRows), [16, 32, 64, 128, 256, 512, Infinity]);
    assertEquals(segments.map((segment) => segment.slot), [...ROWS_SLOT_NAMES]);
    assertEquals(segments[1].scope, "linear / matmul / bmm の行数 17〜32");
    assertEquals(segments[6].scope, "linear / matmul / bmm の行数 > 512");
  });

  it("段の既定はその段の範囲を覆う既定の段の幾何（≤ 64 の 3 段 → ≤ 64・65〜512 の 3 段 → 65〜512・> 512 → > 512）", () => {
    const [small, middle, large] = GEMM_ROWS_BUCKETS.map((rule) => rule.geometry);
    assertEquals(
      profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS).map((segment) => segment.fallback),
      [small, small, small, middle, middle, middle, large],
    );
  });

  it("綴りと段の既定は渡した境界から導く（既定の境界そのものなら既定の表と同じ 3 段）", () => {
    assertEquals(
      profileRowsSegments([64, 512, Infinity]).map((
        segment,
      ) => [segment.maxRows, segment.fallback]),
      GEMM_ROWS_BUCKETS.map((rule) => [rule.maxRows, rule.geometry]),
    );
    assertEquals(
      profileRowsSegments([64, 100, 512, Infinity]).map((segment) => segment.slot),
      ["gemmRows ≤ 64", "gemmRows 65〜100", "gemmRows 101〜512", "gemmRows > 512"],
    );
  });

  it("既定の境界（64 / 512）を含まない境界の集合は生成を止める（段が既定の境界をまたぐ）", () => {
    assertThrows(
      () => profileRowsSegments([32, 128, Infinity]),
      Error,
      "既定の表の境界 64 / 512 を含まない",
    );
    assertThrows(
      () => profileRowsSegments([16, 32, 64, 128, 256, Infinity]),
      Error,
      "既定の表の境界 512 を含まない",
    );
  });

  it("狭義昇順でない・末尾が Infinity でない・正整数でない境界の集合は生成を止める", () => {
    for (const bounds of [[64, 32, 512, Infinity], [64, 64, 512, Infinity]]) {
      assertThrows(() => profileRowsSegments(bounds), Error, "狭義昇順でない");
    }
    for (const bounds of [[64, 512], []]) {
      assertThrows(() => profileRowsSegments(bounds), Error, "末尾が Infinity でない");
    }
    for (const bounds of [[0, 64, 512, Infinity], [16.5, 64, 512, Infinity]]) {
      assertThrows(() => profileRowsSegments(bounds), Error, "正整数");
    }
  });
});

describe("deriveProfile: ケースが 1 本の段", () => {
  it("段の 1 ケースで ×1.05 以上の幾何を採り、未満なら段の既定のまま", () => {
    const verdicts = deriveProfile([
      source([
        ...caseRows(linearCase(16), SMALL, [[A, { speedup: 1.05 }], [B, { speedup: 1.04 }]]),
        ...caseRows(linearCase(32), SMALL, [[A, { speedup: 1.049 }]]),
      ]),
    ], OPTIONS);
    const adopted = verdictOf(verdicts, "gemmRows ≤ 16");
    assertEquals(adopted.cases, ["linear-m16"]);
    assert(adopted.outcome.kind === "adopted");
    assertEquals(adopted.outcome.geometry, A);
    assertEquals(adopted.outcome.geomean, 1.05);
    assertStringIncludes(rejectionOf(adopted, B), "linear-m16 で ×1.040 < ×1.050");
    const kept = verdictOf(verdicts, "gemmRows 17〜32");
    assertEquals(kept.cases, ["linear-m32"]);
    assertEquals(kept.outcome.kind, "default");
    assertEquals(kept.outcome.geometry, SMALL);
    assertStringIncludes(rejectionOf(kept, A), "linear-m32 で ×1.049 < ×1.050");
  });
});

describe("deriveProfile: 掃引の既定の行は今の runtime の既定であること", () => {
  // 比の土台（既定の行）を 1 つずらした fixture — どの欄でも生成を止める
  const shifted: readonly {
    readonly slot: SlotVerdict["slot"];
    readonly rows: readonly Record<string, unknown>[];
  }[] = [
    { slot: "gemmRows ≤ 16", rows: caseRows(linearCase(16), A, [[B, { speedup: 1.2 }]]) },
    { slot: "gemmRows 257〜512", rows: caseRows(linearCase(512), BIG, [[A, { speedup: 1.2 }]]) },
    {
      slot: "attention.qk",
      rows: [
        row(
          { caseId: "attention-qk", op: "attention", shape: "qk BH16 M64 N64 D128" },
          gemmCandidate(A),
          true,
        ),
      ],
    },
    {
      slot: "conv2d.rows64",
      rows: [
        row(
          { caseId: "conv2d-c192", op: "conv2d", shape: "Cin192 Cout192 8x8 3x3" },
          // tileM 64 のまま（= 同じクラス）で幾何だけが runtime の既定と違う
          conv2dCandidate({ regM: 4, regN: 4, wgX: 16, wgY: 16 }),
          true,
        ),
      ],
    },
    {
      slot: "i8a8.linear",
      rows: [
        row(
          { caseId: "i8a8-linear", op: "i8a8-linear", shape: "M64 N64 K64" },
          i8a8Candidate(defaultI8a8Geometry("attention_pv")),
          true,
        ),
      ],
    },
  ];
  for (const { slot, rows } of shifted) {
    it(`${slot} の既定の行の幾何が runtime の既定と違えば落ちる`, () => {
      const error = assertThrows(() => deriveProfile([source(rows)], OPTIONS), Error);
      assertStringIncludes(error.message, `${slot}: 掃引の既定の行`);
      assertStringIncludes(error.message, "今の runtime の既定");
    });
  }
});

describe("deriveProfile: adapter を混ぜない", () => {
  const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);

  it("掃引どうしの adapter（vendor / architecture）が違えば落ちる", () => {
    assertThrows(
      () =>
        deriveProfile([
          source(rows, "m2.json"),
          source(rows, "other.json", { ...ADAPTER, architecture: "metal-2" }),
        ], { vendor: "apple", minSpeedup: 1.05 }),
      Error,
      "と違う",
    );
  });

  it("--vendor / --architecture と掃引の adapter が違えば落ちる", () => {
    assertThrows(
      () => deriveProfile([source(rows)], { ...OPTIONS, vendor: "intel" }),
      Error,
      "合わない",
    );
    assertThrows(
      () => deriveProfile([source(rows)], { ...OPTIONS, architecture: "metal-2" }),
      Error,
      "合わない",
    );
  });

  it("同じ掃引（同じ sha256）を 2 度渡すと落ちる", () => {
    assertThrows(
      () => deriveProfile([source(rows), source(rows)], OPTIONS),
      Error,
      "2 度",
    );
  });

  it("掃引どうしの adapter の device が違えば、他の 3 欄が同じでも落ちる（表は 4 欄の 1 組だけを持つ）", () => {
    const m2 = source(rows, "m2.json", { ...ADAPTER, device: "0x0000" });
    const other = source(rows, "other.json", { ...ADAPTER, device: "0x1234" });
    const blank = source(rows, "blank.json", { ...ADAPTER, device: "" });
    for (const target of [{ optIn: true } as const, OPTIONS]) {
      for (const mixed of [other, blank]) {
        const error = assertThrows(
          () => deriveProfile([m2, mixed], { ...target, minSpeedup: 1.05 }),
          Error,
          "と違う",
        );
        assertStringIncludes(
          error.message,
          `adapter の device ${JSON.stringify(mixed.adapter.device)} が m2.json の "0x0000"`,
        );
      }
    }
    deriveProfile([m2, source(rows, "m2-again.json", { ...ADAPTER, device: "0x0000" })], OPTIONS);
  });
});

describe("parseSweepReport", () => {
  const meta = { path: "sweep.json", sha256: "0" };

  it("adapter の device は必須の文字列（空文字は値として受ける）", () => {
    const rows = caseRows(linearCase(1024), BIG, []);
    const { device: _, ...withoutDevice } = ADAPTER;
    assertThrows(
      () => parseSweepReport(report(rows, withoutDevice), meta),
      Error,
      "adapter: device が文字列でない",
    );
    assertEquals(
      parseSweepReport(report(rows, { ...ADAPTER, device: "" }), meta).adapter.device,
      "",
    );
  });

  it("候補集合は settings.candidateSet、欄の無い古い記録は settings.quick から読み、どちらも無ければ落ちる", () => {
    const base = report(caseRows(linearCase(1024), BIG, []));
    const read = (settings: unknown) => parseSweepReport({ ...base, settings }, meta).candidateSet;
    assertEquals(read({ candidateSet: "quick+", quick: false }), "quick+");
    assertEquals(read({ quick: true }), "quick");
    assertEquals(read({ quick: false }), "full");
    assertThrows(() => read({ candidateSet: "quik" }), Error, "settings.candidateSet が");
    assertThrows(() => read({}), Error, "candidateSet も quick も無い");
    assertThrows(() => read(undefined), Error, "settings が無い");
  });

  it("既定の行が 1 本でないケースがあれば落ちる（比の土台が決まらない）", () => {
    const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);
    assertThrows(
      () => parseSweepReport(report([...rows, rows[0]]), meta),
      Error,
      "既定の行が 2 本",
    );
  });

  it("timestamp が量子化された掃引も受け、刻み 100 µs を観測ごとの上界に使う", () => {
    const parsed = parseSweepReport({
      ...report(caseRows(linearCase(1024), BIG, [])),
      gpuTiming: { feature: true, unit: "ns", quantized: true },
    }, meta);
    assertEquals(parsed.timestampQuantum, 100_000);
    assertEquals(
      parseSweepReport(report(caseRows(linearCase(1024), BIG, [])), meta).timestampQuantum,
      0,
    );
  });

  it("gpuTiming.quantized が真偽値でない・単位 ns 以外で true の掃引は落とす", () => {
    const rows = caseRows(linearCase(1024), BIG, []);
    for (const quantized of [undefined, "true", 1]) {
      assertThrows(
        () =>
          parseSweepReport({
            ...report(rows),
            gpuTiming: { feature: true, unit: "ns", quantized },
          }, meta),
        Error,
        "gpuTiming.quantized が真偽値でない",
      );
    }
    assertThrows(
      () =>
        parseSweepReport({
          ...report(rows),
          gpuTiming: { feature: true, unit: "deno-raw-tick", quantized: true },
        }, meta),
      Error,
      "単位が deno-raw-tick",
    );
  });

  it("rounds に負の値・数でない値がある行と、reps が正整数でない行は落とす", () => {
    const ref = linearCase(1024);
    for (
      const measured of [{ rounds: [1, -1] }, { rounds: [Number.NaN] }, { reps: 1.5 }, { reps: 0 }]
    ) {
      assertThrows(
        () =>
          parseSweepReport(report(caseRows(ref, BIG, [[A, { speedup: 1.2, ...measured }]])), meta),
        Error,
        "の型が違う",
      );
    }
  });

  it("壁時計（unit wall）の掃引は材料にしない", () => {
    assertThrows(
      () =>
        parseSweepReport({
          ...report(caseRows(linearCase(1024), BIG, [])),
          gpuTiming: { feature: false, unit: "wall", quantized: false },
        }, meta),
      Error,
      'gpuTiming.unit が "wall"',
    );
  });

  it("gpuTiming が無い掃引は材料にしない（単位が分からない）", () => {
    const { gpuTiming: _, ...withoutTiming } = report(caseRows(linearCase(1024), BIG, []));
    assertThrows(() => parseSweepReport(withoutTiming, meta), Error, "gpuTiming が無い");
  });

  it("GPU の timestamp で測った掃引（ns / deno-raw-tick）は受ける", () => {
    for (const unit of ["ns", "deno-raw-tick"]) {
      const parsed = parseSweepReport({
        ...report(caseRows(linearCase(1024), BIG, [])),
        gpuTiming: { feature: true, unit, quantized: false },
      }, meta);
      assertEquals(parsed.rows.length, 1);
    }
  });
});

// 幾何プロファイルの生成器（profile.ts）の門。GPU を使わない — 掃引の記録は小さな合成 JSON で作る。
//
// 固定するのは ① 規則の抽出（既定比の門・出力一致の門・クラスの全ケースで勝つこと・幾何平均での
// 比較・掃引にケースが無いクラスの既定）② adapter を混ぜないこと ③ 生成の決定性（整形込み）
// ④ `main.ts profile --check` の終了コード。

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemmMTileGeometry } from "../../packages/runtime/src/kernels/gemm.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../../packages/runtime/src/kernels/geometry-profile.ts";
import {
  defaultGemmGeometry,
  type GemmGeometry,
} from "../../packages/runtime/src/kernels/gemm-geometry.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
} from "../../packages/runtime/src/kernels/i8a8-geometry.ts";
import type { SweepOp } from "./cases.ts";
import { conv2dCandidate, gemmCandidate, i8a8Candidate } from "./geometries.ts";
import {
  deriveProfile,
  displayPath,
  formatTypeScript,
  parseProfileFlags,
  parseSweepReport,
  type ProfileFlags,
  renderProfileSource,
  ROWS_BUCKETS,
  type SlotVerdict,
  type SweepSource,
} from "./profile.ts";
import { REPORT_FORMAT } from "./report.ts";

const ENTRY = new URL("./main.ts", import.meta.url);
const DECODER = new TextDecoder();

const BIG = defaultGemmGeometry();
const A: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };
const B: GemmGeometry = { regM: 8, regN: 4, wgX: 16, wgY: 8 };

type CaseRef = { readonly caseId: string; readonly op: SweepOp; readonly shape: string };

const linearCase = (m: number): CaseRef => ({
  caseId: `linear-m${m}`,
  op: "linear",
  shape: `M${m} N64 K64`,
});

type Measured = {
  readonly speedup?: number;
  readonly identical?: boolean;
  readonly error?: string;
};

/** 掃引の 1 行（`report.ts` の `SweepRow` のうち生成が読む欄 + 数欄）。 */
const row = (
  ref: CaseRef,
  candidate: { readonly name: string; readonly geometry: GemmGeometry | I8a8Geometry },
  isDefault: boolean,
  measured: Measured = {},
): Record<string, unknown> => ({
  ...ref,
  censusCount: 1,
  geometry: candidate.name,
  geometryParams: candidate.geometry,
  isDefault,
  ...(isDefault ? { speedupVsDefault: 1, identicalToDefault: true } : {}),
  ...(measured.speedup === undefined ? {} : { speedupVsDefault: measured.speedup }),
  ...(measured.identical === undefined ? {} : { identicalToDefault: measured.identical }),
  ...(measured.error === undefined ? {} : { error: measured.error }),
});

/** ケース 1 本ぶんの行（既定 + 候補ごとの測定）。 */
const caseRows = (
  ref: CaseRef,
  defaultGeometry: GemmGeometry,
  candidates: readonly (readonly [GemmGeometry, Measured])[],
): Record<string, unknown>[] => [
  row(ref, gemmCandidate(defaultGeometry), true),
  ...candidates.map(([geometry, measured]) =>
    row(ref, gemmCandidate(geometry), false, { identical: true, ...measured })
  ),
];

const ADAPTER = { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" };

const report = (
  rows: readonly Record<string, unknown>[],
  adapter: Record<string, string> = ADAPTER,
): Record<string, unknown> => ({
  format: REPORT_FORMAT,
  date: "2026-09-27T00:00:00.000Z",
  adapter,
  gpuTiming: { feature: true, unit: "ns", quantized: false },
  rows,
});

const source = (
  rows: readonly Record<string, unknown>[],
  name = "sweep.json",
  adapter?: Record<string, string>,
): SweepSource => parseSweepReport(report(rows, adapter), { path: name, sha256: `sha-${name}` });

const OPTIONS = { vendor: "apple", architecture: "metal-3", minSpeedup: 1.05 } as const;

const verdictOf = (verdicts: readonly SlotVerdict[], slot: SlotVerdict["slot"]): SlotVerdict => {
  const found = verdicts.find((verdict) => verdict.slot === slot);
  if (found === undefined) throw new Error(`${slot} が無い`);
  return found;
};

const rejectionOf = (verdict: SlotVerdict, geometry: GemmGeometry): string => {
  const name = gemmCandidate(geometry).name;
  const found = verdict.rejected.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`${name} が退けた一覧に無い`);
  return found.reason;
};

describe("deriveProfile: 規則の抽出", () => {
  it("既定比が --min-speedup 未満の幾何しか無いクラスは、掃引の既定の行の幾何を書く", () => {
    const verdicts = deriveProfile([
      source([
        ...caseRows(linearCase(1024), BIG, [[A, { speedup: 1.04 }]]),
        ...caseRows(linearCase(4096), BIG, [[A, { speedup: 1.3 }]]),
      ]),
    ], OPTIONS);
    const verdict = verdictOf(verdicts, "gemmRows[2]");
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
      "gemmRows[2]",
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
      "gemmRows[2]",
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
      "gemmRows[2]",
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
      "gemmRows[2]",
    );
    assertEquals(verdict.outcome.kind, "adopted");
    assertEquals(verdict.outcome.geometry, B);
    assertStringIncludes(rejectionOf(verdict, A), "幾何平均 ×1.302（採用 ×1.310 に届かない）");
  });

  it("行数 64 / 512 を境にクラスを分け、conv2d は既定の行の tileM で分ける", () => {
    const small: GemmGeometry = { regM: 1, regN: 4, wgX: 4, wgY: 16 };
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
        ...caseRows(linearCase(64), small, [[B, { speedup: 1.2 }]]),
        ...caseRows(linearCase(65), A, []),
        ...caseRows(linearCase(512), A, []),
        ...caseRows(linearCase(513), BIG, []),
        ...conv2dRows(conv(96, 32), 32),
        ...conv2dRows(conv(192, 64), 64),
      ]),
    ], OPTIONS);
    assertEquals(verdictOf(verdicts, "gemmRows[0]").cases, ["linear-m64"]);
    assertEquals(verdictOf(verdicts, "gemmRows[0]").outcome.geometry, B);
    assertEquals(verdictOf(verdicts, "gemmRows[1]").cases, ["linear-m512", "linear-m65"]);
    assertEquals(verdictOf(verdicts, "gemmRows[2]").cases, ["linear-m513"]);
    assertEquals(verdictOf(verdicts, "conv2d.rows32").cases, ["conv2d-c96-m32"]);
    assertEquals(verdictOf(verdicts, "conv2d.rows64").cases, ["conv2d-c192-m64"]);
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
      "gemmRows[2]",
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

describe("deriveProfile: runtime の既定プロファイルとの整合", () => {
  it("行数の境界と、ケースが無いクラスに書く既定は DEFAULT_GEOMETRY_PROFILE と同じ", () => {
    const profile = DEFAULT_GEOMETRY_PROFILE;
    assertEquals([...ROWS_BUCKETS], profile.gemmRows.map((rule) => rule.maxRows));
    const expected: Record<SlotVerdict["slot"], GemmGeometry | I8a8Geometry> = {
      "gemmRows[0]": profile.gemmRows[0].geometry,
      "gemmRows[1]": profile.gemmRows[1].geometry,
      "gemmRows[2]": profile.gemmRows[2].geometry,
      "attention.qk": profile.attention.qk,
      "attention.pv": profile.attention.pv,
      "conv2d.rows64": profile.conv2d.rows64,
      "conv2d.rows32": profile.conv2d.rows32,
      "i8a8.linear": profile.i8a8.linear,
      "i8a8.attentionQk": profile.i8a8.attentionQk,
      "i8a8.attentionPv": profile.i8a8.attentionPv,
    };
    for (const verdict of deriveProfile([source([])], OPTIONS)) {
      assertEquals(verdict.outcome.geometry, expected[verdict.slot], verdict.slot);
    }
  });
});

describe("deriveProfile: 掃引の既定の行は今の runtime の既定であること", () => {
  // 比の土台（既定の行）を 1 つずらした fixture — どの欄でも生成を止める
  const shifted: readonly {
    readonly slot: SlotVerdict["slot"];
    readonly rows: readonly Record<string, unknown>[];
  }[] = [
    { slot: "gemmRows[1]", rows: caseRows(linearCase(512), BIG, [[A, { speedup: 1.2 }]]) },
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
});

describe("parseSweepReport", () => {
  const meta = { path: "sweep.json", sha256: "0" };

  it("既定の行が 1 本でないケースがあれば落ちる（比の土台が決まらない）", () => {
    const rows = caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]);
    assertThrows(
      () => parseSweepReport(report([...rows, rows[0]]), meta),
      Error,
      "既定の行が 2 本",
    );
  });

  it("timestamp が量子化された掃引は材料にしない", () => {
    assertThrows(
      () =>
        parseSweepReport({
          ...report(caseRows(linearCase(1024), BIG, [])),
          gpuTiming: { feature: true, unit: "ns", quantized: true },
        }, meta),
      Error,
      "量子化",
    );
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

describe("parseProfileFlags", () => {
  it("--out のファイル名が <id>.ts でなければ落ちる", () => {
    assertThrows(
      () =>
        parseProfileFlags([
          "--from",
          "a.json",
          "--id",
          "apple-metal-3",
          "--vendor",
          "apple",
          "--out",
          "dir/apple.ts",
        ]),
      Error,
      "apple-metal-3.ts",
    );
  });

  it("--min-speedup は 1 以上（既定より遅い幾何を採らせない）", () => {
    assertThrows(
      () =>
        parseProfileFlags([
          "--from",
          "a.json",
          "--id",
          "x",
          "--vendor",
          "v",
          "--out",
          "x.ts",
          "--min-speedup",
          "0.9",
        ]),
      Error,
      "1 以上",
    );
  });
});

describe("displayPath", () => {
  it("cwd の下の絶対 path は相対にし、先頭の ./ を落とす（綴りが違っても生成物は同じ）", () => {
    assertEquals(displayPath("/repo/outputs/a.json", "/repo"), "outputs/a.json");
    assertEquals(displayPath("./outputs/a.json", "/repo"), "outputs/a.json");
    assertEquals(displayPath("/elsewhere/a.json", "/repo"), "/elsewhere/a.json");
  });
});

describe("生成物", () => {
  const flags: ProfileFlags = {
    from: ["sweep.json"],
    id: "test-gpu",
    vendor: "apple",
    architecture: "metal-3",
    out: "profiles/test-gpu.ts",
    minSpeedup: 1.05,
    check: false,
  };
  const build = (): string => {
    const sources = [
      source(caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2, error: "x */ y" }]])),
    ];
    return renderProfileSource(flags, sources, deriveProfile(sources, flags));
  };

  it("同じ入力からは同じ文字列になり、整形は冪等（生成 → 再生成でバイト同一）", async () => {
    assertEquals(build(), build());
    const formatted = await formatTypeScript(build());
    assertEquals(await formatTypeScript(formatted), formatted);
    assertStringIncludes(formatted, "export const TEST_GPU: GeometryProfile = {");
  });

  it("掃引の記録由来の文字列がコメントを閉じない", () => {
    const source = build();
    assertEquals(source.indexOf("*/"), source.lastIndexOf("*/"));
  });
});

describe("main.ts profile --check", () => {
  const run = async (args: readonly string[]): Promise<{ code: number; stdout: string }> => {
    const output = await new Deno.Command(Deno.execPath(), {
      args: ["run", "-A", ENTRY.href, "profile", ...args],
      stdout: "piped",
      stderr: "piped",
    }).output();
    return { code: output.code, stdout: DECODER.decode(output.stdout) };
  };

  it("既存の生成物が再生成とバイト同一なら 0、違えば差分の要約を出して 1", async () => {
    const root = await Deno.makeTempDir({ prefix: "geometry-profile-" });
    try {
      const sweep = `${root}/sweep.json`;
      await Deno.writeTextFile(
        sweep,
        JSON.stringify(report(caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]]))),
      );
      const out = `${root}/test-gpu.ts`;
      const args = ["--from", sweep, "--id", "test-gpu", "--vendor", "apple", "--out", out];
      assertEquals((await run(args)).code, 0);
      assertEquals((await run([...args, "--check"])).code, 0);
      await Deno.writeTextFile(out, `${await Deno.readTextFile(out)}// 手で編集\n`);
      const mismatched = await run([...args, "--check"]);
      assertEquals(mismatched.code, 1);
      assertStringIncludes(mismatched.stdout, "- // 手で編集");
    } finally {
      await Deno.remove(root, { recursive: true });
    }
  });
});

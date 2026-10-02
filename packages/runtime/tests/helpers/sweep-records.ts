// 掃引の記録（`karume-geometry-sweep/2`）の合成 fixture — 生成器（src/tune/derive.ts）のテストが共有する。
//
// runtime の生成器のテスト（tune_profile_test.ts・tune_derive_test.ts）と、道具の生成物の描画・CLI の
// テスト（tools/geometry-sweep/profile_test.ts・render_test.ts — 道具は tests/helpers を掴んでよい:
// ADR 0008 追記 2026-09-05）が同じ記録から表を作る。GPU を使わない。

import { defaultGemmGeometry, type GemmGeometry } from "../../src/kernels/gemm-geometry.ts";
import type { I8a8Geometry } from "../../src/kernels/i8a8-geometry.ts";
import type { SweepOp } from "../../src/tune/cases.ts";
import { gemmCandidate } from "../../src/tune/geometries.ts";
import { parseSweepReport, type SlotVerdict, type SweepSource } from "../../src/tune/derive.ts";
import { REPORT_FORMAT } from "../../src/tune/report.ts";

// --- 規則の抽出のテスト（tune_profile_test.ts）の記録: ケースと候補を行で組む ---

export const BIG = defaultGemmGeometry();
export const A: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };
export const B: GemmGeometry = { regM: 8, regN: 4, wgX: 16, wgY: 8 };
/** 既定の表の ≤ 64 の段の幾何（A は 65〜512・BIG は > 512 の段の幾何）。 */
export const SMALL: GemmGeometry = { regM: 1, regN: 4, wgX: 4, wgY: 16 };

/** 7 段の欄名（生成器の綴りの期待値 — 手で書いたもの）。 */
export const ROWS_SLOT_NAMES = [
  "gemmRows ≤ 16",
  "gemmRows 17〜32",
  "gemmRows 33〜64",
  "gemmRows 65〜128",
  "gemmRows 129〜256",
  "gemmRows 257〜512",
  "gemmRows > 512",
] as const;

export type CaseRef = { readonly caseId: string; readonly op: SweepOp; readonly shape: string };

export const linearCase = (m: number): CaseRef => ({
  caseId: `linear-m${m}`,
  op: "linear",
  shape: `M${m} N64 K64`,
});

/** matmul の shape は linear と同じ綴り（cases.ts の caseShape）。 */
export const matmulCase = (m: number): CaseRef => ({
  caseId: `matmul-m${m}`,
  op: "matmul",
  shape: `M${m} N64 K64`,
});

/** bmm の shape はバッチを前に置く（cases.ts の caseShape — バケットは行列 1 枚の M）。 */
export const bmmCase = (batch: number, m: number): CaseRef => ({
  caseId: `bmm-b${batch}-m${m}`,
  op: "bmm",
  shape: `B${batch} M${m} N64 K64`,
});

export type Measured = {
  readonly speedup?: number;
  readonly identical?: boolean;
  readonly error?: string;
  /** 計測 round ごとの pass の時間（丸め誤差の上界の材料 — 省くと行に載せない）。 */
  readonly rounds?: readonly number[];
  readonly reps?: number;
};

/** 掃引の 1 行（`report.ts` の `SweepRow` のうち生成が読む欄 + 数欄）。 */
export const row = (
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
  ...(measured.rounds === undefined ? {} : { rounds: measured.rounds }),
  ...(measured.reps === undefined ? {} : { reps: measured.reps }),
});

/** ケース 1 本ぶんの行（既定 + 候補ごとの測定・既定の行の rounds などは `defaultMeasured`）。 */
export const caseRows = (
  ref: CaseRef,
  defaultGeometry: GemmGeometry,
  candidates: readonly (readonly [GemmGeometry, Measured])[],
  defaultMeasured: Measured = {},
): Record<string, unknown>[] => [
  row(ref, gemmCandidate(defaultGeometry), true, defaultMeasured),
  ...candidates.map(([geometry, measured]) =>
    row(ref, gemmCandidate(geometry), false, { identical: true, ...measured })
  ),
];

export const ADAPTER = {
  vendor: "apple",
  architecture: "metal-3",
  device: "",
  description: "Test GPU",
};

/** ケースごとの既定の再測定（`cases[]`）。既定は全ケースで比 1（材料から外さない）。 */
export type Repeats = Readonly<Record<string, Record<string, unknown>>>;

export const report = (
  rows: readonly Record<string, unknown>[],
  adapter: Record<string, string> = ADAPTER,
  repeats: Repeats = {},
): Record<string, unknown> => ({
  format: REPORT_FORMAT,
  date: "2026-09-27T00:00:00.000Z",
  adapter,
  gpuTiming: { feature: true, unit: "ns", quantized: false },
  cases: [...new Set(rows.map((entry) => String(entry.caseId)))].map((caseId) =>
    repeats[caseId] ?? { caseId, defaultRepeat: { perDispatch: 1, driftRatio: 1 } }
  ),
  rows,
});

export const source = (
  rows: readonly Record<string, unknown>[],
  name = "sweep.json",
  adapter?: Record<string, string>,
): SweepSource => parseSweepReport(report(rows, adapter), { path: name, sha256: `sha-${name}` });

export const OPTIONS = { vendor: "apple", architecture: "metal-3", minSpeedup: 1.05 } as const;

export const verdictOf = (
  verdicts: readonly SlotVerdict[],
  slot: SlotVerdict["slot"],
): SlotVerdict => {
  const found = verdicts.find((verdict) => verdict.slot === slot);
  if (found === undefined) throw new Error(`${slot} が無い`);
  return found;
};

export const rejectionOf = (verdict: SlotVerdict, geometry: GemmGeometry): string => {
  const name = gemmCandidate(geometry).name;
  const found = verdict.rejected.find((entry) => entry.name === name);
  if (found === undefined) throw new Error(`${name} が退けた一覧に無い`);
  return found.reason;
};

// --- 注入の表のテスト（tune_derive_test.ts）の記録: linear 1 ケース・2 本の掃引 ---

export const DEFAULT = defaultGemmGeometry();
export const FAST: GemmGeometry = { regM: 4, regN: 4, wgX: 8, wgY: 16 };

/** linear M = 1024 の 1 ケース（既定と、既定より 1.2 倍速く出力の一致する幾何 1 つ）。 */
export const sweep = (name: string, sha256: string): SweepSource =>
  parseSweepReport({
    format: REPORT_FORMAT,
    date: "2026-09-29T00:00:00.000Z",
    adapter: { vendor: "apple", architecture: "metal-3", device: "", description: "Test GPU" },
    gpuTiming: { feature: true, unit: "ns", quantized: false },
    cases: [{ caseId: "linear-m1024", defaultRepeat: { perDispatch: 1, driftRatio: 1 } }],
    rows: [
      {
        caseId: "linear-m1024",
        op: "linear",
        shape: "M1024 N64 K64",
        geometry: gemmCandidate(DEFAULT).name,
        geometryParams: DEFAULT,
        isDefault: true,
        speedupVsDefault: 1,
        identicalToDefault: true,
      },
      {
        caseId: "linear-m1024",
        op: "linear",
        shape: "M1024 N64 K64",
        geometry: gemmCandidate(FAST).name,
        geometryParams: FAST,
        isDefault: false,
        speedupVsDefault: 1.2,
        identicalToDefault: true,
      },
    ],
  }, { path: name, sha256 });

export const SPEC = {
  from: ["a.json", "b.json"],
  id: "test-gpu",
  vendor: "apple",
  architecture: "metal-3",
  out: "profiles/test-gpu.ts",
  minSpeedup: 1.05,
} as const;

export const SOURCES = [sweep("a.json", "sha-a"), sweep("b.json", "sha-b")];

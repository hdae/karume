// 掃引の記録に足した欄（ADR 0117 決定 8）の読み手と、表の `provenance.userAgent`（ADR 0117 追記 2026-10-02）の門。
// 書き手（runGeometrySweep）が欄を書くことは実 GPU の tune_entry_test.ts が固定する。GPU を使わない。
//
// 固定するのは ① 記録の `caseSet` / `defaultKernels` が今の runtime の値と違えば生成器が両方の値を名指して止まり、
// 欄の無い記録は今までどおり受ける ② `aborted` は生成器の判定に効かない ③ `userAgent` を表の
// `provenance.userAgent`（文字列の配列）へ写す規則（`{ deno }` は `Deno/<版>`・同じ値は 1 要素・違えば現れた順に
// 重複を除いて並べる・欄の無い記録が混ざれば書かない）と、parse の往復・照合に使わないこと ④ 記録の
// `candidateKernels`（測った候補のカーネルの指紋）が今の runtime で記録の行から導き直した値と違えば止まる（候補の
// WGSL だけが変わった runtime の古い記録を拒む）⑤ 生成器は返す表を注入の門に通す（空の id の表を作らない）。

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  deriveGeometryProfile,
  geometryProfileJson,
  geometryProfileMismatch,
  type GeometrySweepInput,
  parseGeometryProfileJson,
  sweepCaseSetId,
} from "../tune.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import {
  candidateKernelsId,
  type CasePlanner,
  geometryProfileKernelsId,
  sweepCandidateKernelsId,
} from "../src/tune/fingerprint.ts";
import { SWEEP_CASES } from "../src/tune/cases.ts";
import { casePlan } from "../src/tune/harness.ts";
import {
  A,
  ADAPTER,
  BIG,
  type CaseRef,
  caseRows,
  linearCase,
  report,
} from "./helpers/sweep-records.ts";

const CHROME_153 =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36";
const CHROME_154 =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36";

/** linear M = 1024 の 1 ケース（A が既定より ×1.2 速く出力一致）の記録に、`fields` の欄を足したもの。 */
const record = (fields: Record<string, unknown> = {}): Record<string, unknown> => ({
  ...report(caseRows(linearCase(1024), BIG, [[A, { speedup: 1.2 }]])),
  ...fields,
});

/** 記録の列を生成器の入力にする（sha256 は記録ごとに別 — 同じ値の 2 本は生成器が拒む）。 */
const inputs = (...records: readonly Record<string, unknown>[]): GeometrySweepInput[] =>
  records.map((value, index) => ({
    report: value,
    path: `sweep-${index}.json`,
    sha256: `sha-${index}`,
  }));

const OPTIONS = { id: "test-gpu", vendor: "apple", architecture: "metal-3" } as const;

const derive = (...records: readonly Record<string, unknown>[]) =>
  deriveGeometryProfile(inputs(...records), OPTIONS);

describe("記録の照合キー caseSet / defaultKernels（ADR 0117 決定 8）", () => {
  const current = {
    caseSet: sweepCaseSetId(),
    defaultKernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
  };

  it("今の runtime の値を持つ記録は受け、欄の無い記録（この欄より前の記録）と同じ表になる", () => {
    const withKeys = derive(record(current));
    const without = derive(record());
    assertEquals(withKeys.profile, without.profile);
    assertEquals(withKeys.verdicts, without.verdicts);
    assertEquals(withKeys.profile.gemmRows.at(-1)?.geometry, A);
  });

  for (const key of ["caseSet", "defaultKernels"] as const) {
    it(`${key} が今の runtime の値と違えば、記録の値と今の値の両方を名指して止まる`, () => {
      const stale = "0123456789abcdef";
      const error = assertThrows(
        () => derive(record({ ...current, [key]: stale })),
        Error,
        `sweep-0.json: ${key} ${stale} が今の runtime の ${current[key]} と違う`,
      );
      assertStringIncludes(error.message, "掃引し直す");
    });

    it(`${key} が文字列でなければ止まる（黙って照合を飛ばさない）`, () => {
      assertThrows(
        () => derive(record({ ...current, [key]: 1 })),
        Error,
        `sweep-0.json: ${key} が文字列でない（1）`,
      );
    });
  }

  it("2 本目の記録だけが違っても、その記録を名指して止まる", () => {
    assertThrows(
      () => derive(record(current), record({ ...current, caseSet: "ffffffffffffffff" })),
      Error,
      "sweep-1.json: caseSet ffffffffffffffff",
    );
  });

  it("aborted・startedAt・cases[].elapsedMs は生成器の判定に効かない", () => {
    const plain = derive(record(current));
    const base = record(current);
    const cases = base.cases;
    if (!Array.isArray(cases)) throw new Error("fixture の cases が配列でない");
    const annotated = derive({
      ...base,
      startedAt: "2026-09-26T23:00:00.000Z",
      aborted: true,
      cases: cases.map((summary: Record<string, unknown>) => ({ ...summary, elapsedMs: 1234.5 })),
    });
    assertEquals(annotated.profile, plain.profile);
    assertEquals(annotated.verdicts, plain.verdicts);
  });
});

describe("表の provenance.userAgent（ADR 0117 追記 2026-10-02）", () => {
  it("記録の userAgent の文字列をそのまま 1 要素の配列で写す", () => {
    assertEquals(
      derive(record({ userAgent: CHROME_154 })).profile.provenance?.userAgent,
      [CHROME_154],
    );
  });

  it("Deno の CLI の記録（{ deno }）は Deno の navigator.userAgent と同じ綴り Deno/<版> にする", () => {
    assertEquals(
      derive(record({ userAgent: { deno: "2.9.6" } })).profile.provenance?.userAgent,
      ["Deno/2.9.6"],
    );
  });

  it("全ての記録で同じ値なら 1 要素だけ書く（今のブラウザとそのまま比べられる）", () => {
    const { profile } = derive(
      record({ userAgent: CHROME_154 }),
      record({ userAgent: CHROME_154 }),
    );
    assertEquals(profile.provenance?.userAgent, [CHROME_154]);
  });

  it("記録どうしで違えば、現れた順に重複を除いて並べる（', ' を含む値でも要素ごとに分かれたまま）", () => {
    const { profile } = derive(
      record({ userAgent: CHROME_153 }),
      record({ userAgent: CHROME_154 }),
      record({ userAgent: CHROME_153 }),
      record({ userAgent: { deno: "2.9.6" } }),
    );
    assertEquals(profile.provenance?.userAgent, [CHROME_153, CHROME_154, "Deno/2.9.6"]);
  });

  it("userAgent の無い記録が混ざれば欄ごと書かない（一部の記録のブラウザを表の値にしない）", () => {
    const { profile } = derive(record({ userAgent: CHROME_154 }), record());
    assertEquals(profile.provenance !== undefined && "userAgent" in profile.provenance, false);
  });

  it("文字列でも { deno: 文字列 } でもない userAgent は記録を名指して止まる", () => {
    for (const userAgent of [1, { deno: 2 }, { deno: "2.9.6", v8: "14" }, null]) {
      assertThrows(
        () => derive(record({ userAgent })),
        Error,
        "sweep-0.json: userAgent が文字列でも { deno: 文字列 } でもない",
        JSON.stringify(userAgent),
      );
    }
  });

  it("保存の JSON（geometryProfileJson → parseGeometryProfileJson）の往復で残る", () => {
    const { profile } = derive(
      record({ userAgent: CHROME_153 }),
      record({ userAgent: CHROME_154 }),
    );
    const parsed = parseGeometryProfileJson(geometryProfileJson(profile));
    assertEquals(parsed, profile);
    assertEquals(parsed.provenance?.userAgent, [CHROME_153, CHROME_154]);
  });

  it("照合（geometryProfileMismatch）には使わない — 掃引と別のブラウザでも一致のまま", () => {
    const { profile } = derive(record({ userAgent: CHROME_153 }));
    assertEquals(geometryProfileMismatch(profile, ADAPTER), undefined);
    const otherBrowser = {
      ...profile,
      provenance: profile.provenance === undefined
        ? undefined
        : { ...profile.provenance, userAgent: [CHROME_154] },
    };
    assertEquals(geometryProfileMismatch(otherBrowser, ADAPTER), undefined);
  });
});

describe("記録の照合キー candidateKernels（測った候補のカーネル — ADR 0117 決定 8）", () => {
  // 指紋は記録の行を今の runtime の case plan で組み直すので、ケースは今の形状表にあるもの（合成の id では導けない）
  const LINEAR: CaseRef = {
    caseId: "linear-m1024-n2048-k2048",
    op: "linear",
    shape: "M1024 N2048 K2048",
  };
  const MEASURED = [
    { caseId: LINEAR.caseId, geometryParams: BIG },
    { caseId: LINEAR.caseId, geometryParams: A },
  ];
  const measuredRecord = (fields: Record<string, unknown> = {}): Record<string, unknown> => ({
    ...report(caseRows(LINEAR, BIG, [[A, { speedup: 1.2 }]])),
    dp4a: false,
    ...fields,
  });

  it("今の runtime の値を持つ記録は受け、欄の無い記録と同じ表になる", () => {
    const current = sweepCandidateKernelsId(MEASURED, false);
    const withKey = derive(measuredRecord({ candidateKernels: current }));
    const without = derive(measuredRecord());
    assertEquals(withKey.profile, without.profile);
    assertEquals(withKey.profile.gemmRows.at(-1)?.geometry, A);
  });

  it("既定を変えず候補の WGSL だけが違う runtime で焼いた記録は、両方の値を名指して止まる", () => {
    // 記録を書いた runtime: 既定の幾何以外の dispatch の WGSL だけが今と違う（defaultKernels は同じ値のまま）
    const candidatesOnly: CasePlanner = (sweepCase, limit, dp4a) => {
      const plan = casePlan(sweepCase, limit, dp4a);
      const defaultGeometry = JSON.stringify(plan.defaultCandidate.geometry);
      return {
        ...plan,
        launch: (candidate) => {
          const launch = plan.launch(candidate);
          return JSON.stringify(candidate.geometry) === defaultGeometry
            ? launch
            : { ...launch, wgsl: `${launch.wgsl}\n// older` };
        },
      };
    };
    const stale = candidateKernelsId(MEASURED, false, SWEEP_CASES, candidatesOnly);
    const current = sweepCandidateKernelsId(MEASURED, false);
    const error = assertThrows(
      () =>
        derive(measuredRecord({
          defaultKernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
          candidateKernels: stale,
        })),
      Error,
      `sweep-0.json: candidateKernels ${stale} が今の runtime の ${current} と違う（測った候補のカーネル`,
    );
    assertStringIncludes(error.message, "掃引し直す");
  });

  it("candidateKernels が文字列でない・dp4a が真偽値でない記録は止まる（黙って照合を飛ばさない）", () => {
    const current = sweepCandidateKernelsId(MEASURED, false);
    assertThrows(
      () => derive(measuredRecord({ candidateKernels: 1 })),
      Error,
      "sweep-0.json: candidateKernels が文字列でない（1）",
    );
    assertThrows(
      () => derive(measuredRecord({ candidateKernels: current, dp4a: undefined })),
      Error,
      "sweep-0.json: dp4a が真偽値でない",
    );
  });
});

describe("deriveGeometryProfile は返す表を注入の門に通す", () => {
  it("id が空の表は作らず、注入の門の文言で止まる（match のある表・注入専用の表とも）", () => {
    for (const options of [{ ...OPTIONS, id: "" }, { id: "", optIn: true } as const]) {
      assertThrows(
        () => deriveGeometryProfile(inputs(record()), options),
        Error,
        "生成した表 '' が注入の門を通らない — 幾何プロファイル: id が空",
      );
    }
  });
});

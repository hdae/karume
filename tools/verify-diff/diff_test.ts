// verify-diff の純関数部の門（GPU も outputs/ も触らない — 入力は全て合成）。
//
// この道具の値打ちは「割れた所を指す」ことなので、固定するのは ① 席の選び方（最新の日付 /
// --date / 系列の絞り込み）② 差異の拾い方（status / 実物の sha / 片方にしか無いケース）
// ③ 読めない文書で黙って空にしないこと の 3 面である。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { buildDiff, type LoadedResults, parseResultsDocument } from "./diff.ts";
import { parseArgs, single } from "./main.ts";

/** 席 1 本ぶんの合成入力（`results.json` の実物と同じ形を境界の検査に通す）。 */
const seat = (
  environment: string,
  date: string,
  family: string,
  cases: readonly unknown[],
  checkout: unknown = { sha: "1111111111111111111111111111111111111111", dirty: false },
): LoadedResults => ({
  path: `outputs/verify/${environment}/${date}_${family}/results.json`,
  date,
  document: parseResultsDocument({
    schema: 1,
    family,
    environment: { key: environment, runtime: { name: "deno" } },
    checkout,
    startedAt: `${date}T00:00:00.000Z`,
    cases,
  }),
});

const passing = (id: string, actual?: string): Record<string, unknown> => ({
  id,
  status: "pass",
  elapsedMs: 1,
  ...(actual === undefined ? {} : { actual }),
});

describe("results.json の境界の検査", () => {
  it("schema が 1 でない文書は throw する", () => {
    assertThrows(
      () => parseResultsDocument({ schema: 2, family: "anima" }, "a.json"),
      Error,
      "a.json: schema が 1 でない（2）",
    );
  });

  it("必須欄が欠けた文書は throw する（空の行列へ落とさない）", () => {
    assertThrows(
      () =>
        parseResultsDocument({
          schema: 1,
          family: "anima",
          environment: { key: "deno-a" },
          checkout: null,
          startedAt: "2026-09-21T00:00:00.000Z",
        }),
      Error,
      "cases が配列でない",
    );
    assertThrows(
      () =>
        parseResultsDocument({
          schema: 1,
          family: "anima",
          environment: {},
          checkout: null,
          startedAt: "2026-09-21T00:00:00.000Z",
          cases: [],
        }),
      Error,
      "environment: key が文字列でない",
    );
    assertThrows(
      () => seat("deno-a", "2026-09-21", "anima", [{ id: "x", status: "ok", elapsedMs: 1 }]),
      Error,
      "未知の status 'ok'",
    );
  });

  it("measurements を読む（非有限が null で来ても落ちない）", () => {
    const loaded = seat("deno-a", "2026-09-21", "golden", [{
      id: "activations",
      status: "pass",
      elapsedMs: 3,
      measurements: [
        {
          output: "y",
          maxAbs: 0.00002,
          maxRel: null,
          tolerance: { atol: 0.00048828125, rtol: 0 },
          stage: "spec",
        },
      ],
    }]);
    assertEquals(loaded.document.cases[0].measurements, [{
      output: "y",
      maxAbs: 0.00002,
      maxRel: null,
      tolerance: { atol: 0.00048828125, rtol: 0 },
      stage: "spec",
    }]);
    // 行列にもそのまま載る（--json と実測の小表が読む所）。
    const report = buildDiff([loaded]);
    assertEquals(report.families[0].rows[0].cells["deno-a"].measurements?.[0].maxRel, null);
  });
});

describe("席の選び方", () => {
  it("同じ系列・同じ環境に複数の日付があれば最新の日付を採る", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-20", "anima", [passing("base")]),
      seat("deno-a", "2026-09-21", "anima", [passing("base"), passing("extra")]),
    ]);
    assertEquals(report.families[0].selected, [{
      environment: "deno-a",
      date: "2026-09-21",
      path: "outputs/verify/deno-a/2026-09-21_anima/results.json",
      checkout: { sha: "1111111111111111111111111111111111111111", dirty: false },
    }]);
    assertEquals(report.families[0].rows.map((row) => row.id), ["base", "extra"]);
  });

  it("--date を指定するとその日付だけを採り、持たない環境は「無し」になる", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-20", "anima", [passing("base")]),
      seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base")]),
    ], { date: "2026-09-20" });
    assertEquals(report.families[0].environments, ["deno-a"]);
    assertEquals(report.families[0].selected[0].date, "2026-09-20");
    assertEquals(report.families[0].absent, ["deno-b"]);
  });

  it("--family で指定した系列だけを行列にし、当たらない指定は警告にする", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      seat("deno-a", "2026-09-21", "sbv2", [passing("jvnv")]),
    ], { families: ["sbv2", "irodori"] });
    assertEquals(report.families.map((matrix) => matrix.family), ["sbv2"]);
    assertEquals(report.warnings, ["--family irodori に当たる結果が無い"]);
  });
});

describe("環境間の差異", () => {
  it("全環境で一致していれば差異は空", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base", "aa")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base", "aa")]),
    ]);
    assertEquals(report.families[0].differences, []);
    assertEquals(report.families[0].environments, ["deno-a", "deno-b"]);
  });

  it("status が食い違うケースを挙げる", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      seat("deno-b", "2026-09-21", "anima", [{ id: "base", status: "fail", elapsedMs: 2 }]),
    ]);
    assertEquals(report.families[0].differences, [{
      caseId: "base",
      kind: "status",
      values: { "deno-a": "pass", "deno-b": "fail" },
    }]);
  });

  it("実物の sha が食い違うケースを挙げる（赤にはしない = 差異として並ぶだけ）", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base", "aa")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base", "bb")]),
    ]);
    assertEquals(report.families[0].differences, [{
      caseId: "base",
      kind: "actual",
      values: { "deno-a": "aa", "deno-b": "bb" },
    }]);
  });

  it("片方にしか無いケースを挙げる", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base"), passing("only-a")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base")]),
    ]);
    assertEquals(report.families[0].differences, [{
      caseId: "only-a",
      kind: "missing",
      values: { "deno-a": "pass" },
      missing: ["deno-b"],
    }]);
  });
});

describe("チェックアウトの警告", () => {
  it("選んだ結果の sha が環境間で食い違えば警告する", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")]),
      seat("deno-b", "2026-09-21", "anima", [passing("base")], {
        sha: "2222222222222222222222222222222222222222",
        dirty: false,
      }),
    ]);
    assertEquals(report.families[0].warnings, [
      "checkout の sha が環境間で食い違う（別のコミットの結果を並べている）: " +
      "deno-a=1111111 / deno-b=2222222",
    ]);
  });

  it("dirty な作業木と checkout の無い結果も警告する", () => {
    const report = buildDiff([
      seat("deno-a", "2026-09-21", "anima", [passing("base")], {
        sha: "1111111111111111111111111111111111111111",
        dirty: true,
      }),
      seat("deno-b", "2026-09-21", "anima", [passing("base")], null),
    ]);
    assertEquals(report.families[0].warnings, [
      "deno-a: 作業木が dirty なチェックアウトで採られた結果（1111111）",
      "deno-b: checkout が無い（git の無い機で採られた結果）",
    ]);
  });
});

describe("CLI の引数", () => {
  it("--json は値を取らず、以降の対をずらさない", () => {
    const args = parseArgs(["--json", "--family", "anima", "--family", "sbv2"]);
    assertEquals(args.flags.has("json"), true);
    assertEquals(args.values.get("family"), ["anima", "sbv2"]);
  });

  it("未知のオプション・値の書き忘れ・2 度渡しは落ちる", () => {
    assertThrows(() => parseArgs(["--families", "anima"]), Error, "未知のオプション --families");
    assertThrows(() => parseArgs(["--date"]), Error, "'--キー 値' の対になっていない");
    assertThrows(() => parseArgs(["--date", "--json"]), Error, "値が無い（'--json' はオプション）");
    assertThrows(
      () => single(parseArgs(["--date", "2026-09-20", "--date", "2026-09-21"]), "date"),
      Error,
      "--date は 1 度しか指定できない",
    );
  });
});

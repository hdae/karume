// 幾何プロファイルの生成物（render.ts）と CLI（profile.ts = `main.ts profile`）の門。GPU を使わない —
// 掃引の記録は小さな合成 JSON で作る（packages/runtime/tests/helpers/sweep-records.ts）。
//
// 固定するのは ① 採否の行が生成物のコメントに載ること ② 生成の決定性（整形込み）③ CLI の引数
// ④ `main.ts profile --check` の終了コード。規則の抽出そのものは runtime の
// packages/runtime/tests/tune_profile_test.ts が固定する。

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { gemmCandidate } from "../../packages/runtime/src/tune/geometries.ts";
import {
  deriveProfile,
  parseSweepReport,
  type SweepSource,
  verdictLines,
} from "../../packages/runtime/src/tune/derive.ts";
import {
  A,
  ADAPTER,
  B,
  BIG,
  caseRows,
  linearCase,
  type Repeats,
  report,
  ROWS_SLOT_NAMES,
  SMALL,
  source,
} from "../../packages/runtime/tests/helpers/sweep-records.ts";
import { displayPath, formatTypeScript, parseProfileFlags, type ProfileFlags } from "./profile.ts";
import { renderProfileSource } from "./render.ts";

const ENTRY = new URL("./main.ts", import.meta.url);
const DECODER = new TextDecoder();

describe("deriveProfile: 既定の再測定比が範囲外のケースは、その掃引の材料から外す", () => {
  const ref = linearCase(1024);
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

  it("外したケースは採否の行と生成物のコメントに理由つきで出る", () => {
    const drifted = sweepWith(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "drifted.json", {
      [ref.caseId]: drift(ref.caseId, 1.16),
    });
    const steady = source(caseRows(ref, BIG, [[A, { speedup: 1.3 }]]), "steady.json");
    const flags = {
      from: ["drifted.json", "steady.json"],
      id: "test-gpu",
      vendor: "apple",
      architecture: "metal-3",
      out: "profiles/test-gpu.ts",
      minSpeedup: 1.05,
    };
    const verdicts = deriveProfile([drifted, steady], flags);
    const line =
      "掃引 drifted.json: linear-m1024 は既定の再測定比 ×1.160 が範囲外のため比の材料から外した（出力の一致と失敗は見る）";
    assert(verdictLines(verdicts).includes(`  - ${line}`), verdictLines(verdicts).join("\n"));
    assertStringIncludes(
      renderProfileSource(flags, [drifted, steady], verdicts),
      ` *   - ${line}\n`,
    );
  });
});

describe("deriveProfile: ケースが 1 本の段", () => {
  it("採否の行と生成物は段を範囲で綴り、添字（gemmRows[n]）を含まない", () => {
    const flags = {
      from: ["sweep.json"],
      id: "test-gpu",
      vendor: "apple",
      architecture: "metal-3",
      out: "profiles/test-gpu.ts",
      minSpeedup: 1.05,
    };
    const sources = [
      source(caseRows(linearCase(32), SMALL, [[A, { speedup: 1.2 }], [B, { speedup: 1.01 }]])),
    ];
    const verdicts = deriveProfile(sources, flags);
    const lines = verdictLines(verdicts);
    assert(
      lines.includes(
        "- gemmRows 17〜32（linear / matmul / bmm の行数 17〜32・1 ケース）: 採用 " +
          `${gemmCandidate(A).name} ×1.200（×1.200〜×1.200）`,
      ),
      lines.join("\n"),
    );
    for (const slot of ROWS_SLOT_NAMES) {
      assert(lines.some((line) => line.startsWith(`- ${slot}（`)), slot);
    }
    const rendered = renderProfileSource(flags, sources, verdicts);
    assert(!lines.join("\n").includes("gemmRows["), lines.join("\n"));
    assert(!rendered.includes("gemmRows["), rendered);
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

  it("--description は match に足し、--opt-in は vendor 無しの注入専用の指定になる", () => {
    const base = ["--from", "a.json", "--id", "x", "--out", "x.ts"];
    const described = parseProfileFlags([
      ...base,
      "--vendor",
      "apple",
      "--architecture",
      "metal-3",
      "--description",
      "Apple M2",
    ]);
    assert(described.optIn !== true);
    assertEquals(described.description, "Apple M2");
    const optIn = parseProfileFlags([...base, "--opt-in"]);
    assertEquals(optIn.optIn, true);
    assertEquals(optIn.vendor, undefined);
  });

  it("--opt-in と --vendor / --architecture / --description の同時指定は落ちる", () => {
    const base = ["--from", "a.json", "--id", "x", "--out", "x.ts", "--opt-in"];
    for (const extra of [["--vendor", "v"], ["--architecture", "a"], ["--description", "d"]]) {
      assertThrows(() => parseProfileFlags([...base, ...extra]), Error, "同時に指定しない");
    }
  });

  it("--vendor も --opt-in も無ければ落ちる（注入専用を黙って作らない）", () => {
    assertThrows(
      () => parseProfileFlags(["--from", "a.json", "--id", "x", "--out", "x.ts"]),
      Error,
      "--opt-in",
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

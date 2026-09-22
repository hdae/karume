// verify-diff の CLI 面の門（GPU も outputs/ も触らない — 席は毎回 Deno.makeTempDir に合成する）。
//
// 純関数部は diff_test.ts が押さえるので、ここで固定するのは**プロセスとして走らせたときだけ
// 観測できる面**である: ① 根の歩き方（席の読み込み・`--root` の綴りの解釈）② 終了コードの契約
// （差異は 0・読めない文書だけ 1 — ADR 0106）③ Markdown と `--json` の出力の形。
//
// 道具立ては tools/opbench/main_test.ts:44 と同じ（Deno.Command で main.ts を起動する）。

import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "@std/testing/bdd";

const ENTRY = new URL("./main.ts", import.meta.url);
const DECODER = new TextDecoder();

type Run = { readonly code: number; readonly stdout: string; readonly stderr: string };

const run = async (args: readonly string[]): Promise<Run> => {
  const output = await new Deno.Command(Deno.execPath(), {
    args: ["run", "-A", ENTRY.href, ...args],
    stdout: "piped",
    stderr: "piped",
  }).output();
  return {
    code: output.code,
    stdout: DECODER.decode(output.stdout),
    stderr: DECODER.decode(output.stderr),
  };
};

/** `results.json` 1 本ぶんの中身（書き手 `packages/runtime/tests/helpers/results.ts` と同じ形）。 */
const document = (
  family: string,
  environmentKey: string,
  cases: readonly unknown[],
): Record<string, unknown> => ({
  schema: 1,
  family,
  environment: { key: environmentKey, runtime: { name: "deno" } },
  checkout: { sha: "1111111111111111111111111111111111111111", dirty: false },
  startedAt: "2026-09-21T00:00:00.000Z",
  cases,
});

const passing = (id: string): Record<string, unknown> => ({ id, status: "pass", elapsedMs: 1 });

/** 席 1 本を置く。`body` を文字列で渡すと（壊れた JSON のために）そのまま書く。 */
const writeSeat = async (
  root: string,
  environmentDir: string,
  seat: string,
  body: unknown,
): Promise<void> => {
  const directory = `${root}/${environmentDir}/${seat}`;
  await Deno.mkdir(directory, { recursive: true });
  await Deno.writeTextFile(
    `${directory}/results.json`,
    typeof body === "string" ? body : JSON.stringify(body),
  );
};

/** 席を置いた一時の根で 1 本走らせる（後始末込み）。 */
const withRoot = async (
  place: (root: string) => Promise<void>,
  use: (root: string) => Promise<void>,
): Promise<void> => {
  const root = await Deno.makeTempDir({ prefix: "verify-diff-" });
  try {
    await place(root);
    await use(root);
  } finally {
    await Deno.remove(root, { recursive: true });
  }
};

describe("verify-diff CLI: 読める根", () => {
  it("差異があっても終了コードは 0 で、行列と差異を Markdown に出す", async () => {
    await withRoot(
      async (root) => {
        await writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [
            passing("base"),
          ]),
        );
        await writeSeat(
          root,
          "deno-b",
          "2026-09-21_anima",
          document("anima", "deno-b", [
            { id: "base", status: "fail", elapsedMs: 2 },
          ]),
        );
      },
      async (root) => {
        const result = await run(["--root", root]);
        // 門ではないので差異があっても 0（ADR 0106）。
        assertEquals(result.code, 0, result.stderr);
        assertStringIncludes(result.stdout, "# verify-diff");
        assertStringIncludes(result.stdout, `根: ${root}`);
        assertStringIncludes(result.stdout, "## anima");
        assertStringIncludes(result.stdout, "| base | pass | fail |");
        assertStringIncludes(result.stdout, "status が割れている — deno-a=pass / deno-b=fail");
      },
    );
  });

  it("--json は行列と警告を JSON で出す", async () => {
    await withRoot(
      async (root) => {
        await writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [
            passing("base"),
          ]),
        );
      },
      async (root) => {
        const result = await run(["--root", root, "--json"]);
        assertEquals(result.code, 0, result.stderr);
        const report = JSON.parse(result.stdout);
        assertEquals(report.warnings, []);
        assertEquals(report.families.length, 1);
        assertEquals(report.families[0].family, "anima");
        assertEquals(report.families[0].environments, ["deno-a"]);
        assertEquals(report.families[0].rows, [{
          id: "base",
          cells: { "deno-a": { status: "pass", elapsedMs: 1 } },
        }]);
      },
    );
  });

  it("同じ系列・環境キー・日付の席が 2 つあると警告を出す（終了コードは 0 のまま）", async () => {
    await withRoot(
      async (root) => {
        // ディレクトリ名は違うが、文書が名乗る環境キーは同じ（手コピーで起こる形）。
        await writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [
            passing("base"),
          ]),
        );
        await writeSeat(
          root,
          "copied",
          "2026-09-21_anima",
          document("anima", "deno-a", [
            { id: "base", status: "fail", elapsedMs: 2 },
          ]),
        );
      },
      async (root) => {
        const result = await run(["--root", root, "--json"]);
        assertEquals(result.code, 0, result.stderr);
        const report = JSON.parse(result.stdout);
        assertEquals(report.warnings.length, 1);
        assertStringIncludes(
          report.warnings[0],
          "同じ系列・環境キー・日付の席が 2 つある（anima / deno-a / 2026-09-21）",
        );
        // 両方の path が読み手に渡る（どちらが採られたか分からないまま黙らない）。
        assertStringIncludes(report.warnings[0], `${root}/deno-a/2026-09-21_anima/results.json`);
        assertStringIncludes(report.warnings[0], `${root}/copied/2026-09-21_anima/results.json`);
      },
    );
  });
});

describe("verify-diff CLI: 読めない根", () => {
  it("席に results.json が無ければ終了コード 1", async () => {
    await withRoot(
      async (root) => {
        await Deno.mkdir(`${root}/deno-a/2026-09-21_anima`, { recursive: true });
      },
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "席に results.json が無い（壊れた席）");
        assertEquals(result.stdout, "");
      },
    );
  });

  it("JSON として読めない文書は終了コード 1", async () => {
    await withRoot(
      (root) => writeSeat(root, "deno-a", "2026-09-21_anima", "{ 壊れている"),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "JSON として読めない");
        assertEquals(result.stdout, "");
      },
    );
  });

  it("schema が違う文書は終了コード 1", async () => {
    await withRoot(
      (root) =>
        writeSeat(root, "deno-a", "2026-09-21_anima", {
          ...document("anima", "deno-a", [passing("base")]),
          schema: 2,
        }),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "schema が 1 でない（2）");
      },
    );
  });

  it("ケース ID が重複した文書は終了コード 1", async () => {
    await withRoot(
      (root) =>
        writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [passing("base"), {
            id: "base",
            status: "fail",
            elapsedMs: 2,
          }]),
        ),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "ケース ID が重複 'base'");
      },
    );
  });

  it("根が無ければ終了コード 1", async () => {
    const result = await run(["--root", "/nonexistent-verify-diff-root"]);
    assertEquals(result.code, 1);
    assertStringIncludes(result.stderr, "結果の根が無い");
  });
});

describe("verify-diff CLI: --root の綴り", () => {
  it("`%` を含む根は実名のまま読む（`%41` を `A` に解かない）", async () => {
    const parent = await Deno.makeTempDir({ prefix: "verify-diff-escape-" });
    try {
      // 実体 `x%41y` と、`%41` を解いた先の `xAy` を並べて置く（別の場所を黙って読む形）。
      await writeSeat(
        `${parent}/x%41y`,
        "deno-a",
        "2026-09-21_anima",
        document(
          "anima",
          "deno-a",
          [passing("literal")],
        ),
      );
      await writeSeat(
        `${parent}/xAy`,
        "deno-a",
        "2026-09-21_anima",
        document("anima", "deno-a", [
          passing("decoded"),
        ]),
      );

      const result = await run(["--root", `${parent}/x%41y`]);

      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "literal");
      assert(
        !result.stdout.includes("decoded"),
        `%41 を解いた別の根を読んでいる:\n${result.stdout}`,
      );
    } finally {
      await Deno.remove(parent, { recursive: true });
    }
  });

  it("`#` を含む根も読める（URL の素片として切り落とさない）", async () => {
    const parent = await Deno.makeTempDir({ prefix: "verify-diff-hash-" });
    try {
      const root = `${parent}/hash#dir`;
      await writeSeat(
        root,
        "deno-a",
        "2026-09-21_anima",
        document("anima", "deno-a", [
          passing("base"),
        ]),
      );

      const result = await run(["--root", root]);

      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "| base | pass |");
    } finally {
      await Deno.remove(parent, { recursive: true });
    }
  });
});

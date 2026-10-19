// Checks of the CLI side of verify-diff (touches neither the GPU nor outputs/ — results directories are
// synthesized in a temporary directory every time).
//
// diff_test.ts covers the pure part, so what is pinned here is **what can only be observed when it
// runs as a process**: ① how the root is walked (loading results directories, how the spelling of
// `--root` is interpreted) ② the exit code contract (0 for differences, 1 only for unreadable
// documents — ADR 0106) ③ the shape of the Markdown and `--json` output.
//
// Setup: node:child_process's execFile starts `node main.ts` as a child process.

import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";

import { makeTempDir, mkdir, remove, writeTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { describe, it } from "vite-plus/test";

const ENTRY = fileURLToPath(new URL("../main.ts", import.meta.url));

type Run = { readonly code: number; readonly stdout: string; readonly stderr: string };

/** Returns a non-zero exit code as a result too (only a failure to start rejects). */
const run = (args: readonly string[]): Promise<Run> =>
  new Promise((resolve, reject) => {
    execFile(process.execPath, [ENTRY, ...args], { encoding: "utf8" }, (error, stdout, stderr) => {
      const code = error === null ? 0 : error.code;
      if (typeof code !== "number") {
        reject(error);
        return;
      }
      resolve({ code, stdout, stderr });
    });
  });

/**
 * The contents of one `results.json` (the same shape as the writer,
 * `packages/runtime/tests/helpers/results.ts`).
 */
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

/** Places one results directory. A string `body` is written as is (for broken JSON). */
const writeSeat = async (
  root: string,
  environmentDir: string,
  seat: string,
  body: unknown,
): Promise<void> => {
  const directory = `${root}/${environmentDir}/${seat}`;
  await mkdir(directory, { recursive: true });
  await writeTextFile(
    `${directory}/results.json`,
    typeof body === "string" ? body : JSON.stringify(body),
  );
};

/** Runs once on a temporary root with results directories placed in it (cleanup included). */
const withRoot = async (
  place: (root: string) => Promise<void>,
  use: (root: string) => Promise<void>,
): Promise<void> => {
  const root = await makeTempDir({ prefix: "verify-diff-" });
  try {
    await place(root);
    await use(root);
  } finally {
    await remove(root, { recursive: true });
  }
};

describe("verify-diff CLI: a readable root", () => {
  it("exits with 0 even with differences, and prints the matrix and the differences as Markdown", async () => {
    await withRoot(
      async (root) => {
        await writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [passing("base")]),
        );
        await writeSeat(
          root,
          "deno-b",
          "2026-09-21_anima",
          document("anima", "deno-b", [{ id: "base", status: "fail", elapsedMs: 2 }]),
        );
      },
      async (root) => {
        const result = await run(["--root", root]);
        // Not a check, so 0 even with differences (ADR 0106).
        assertEquals(result.code, 0, result.stderr);
        assertStringIncludes(result.stdout, "# verify-diff");
        assertStringIncludes(result.stdout, `Root: ${root}`);
        assertStringIncludes(result.stdout, "## anima");
        assertStringIncludes(result.stdout, "| base | pass | fail |");
        assertStringIncludes(result.stdout, "status differs — deno-a=pass / deno-b=fail");
      },
    );
  });

  it("--json prints the matrix and the warnings as JSON", async () => {
    await withRoot(
      async (root) => {
        await writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [passing("base")]),
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
        assertEquals(report.families[0].rows, [
          {
            id: "base",
            cells: { "deno-a": { status: "pass", elapsedMs: 1 } },
          },
        ]);
      },
    );
  });

  it("lists the self A/B observations (comparisons) in a small table per case", async () => {
    await withRoot(
      (root) =>
        writeSeat(
          root,
          "deno-a",
          "2026-09-26_anima-ab",
          document("anima-ab", "deno-a", [
            {
              id: "f16+dit8-a8-attn8-s16-512",
              status: "fail",
              elapsedMs: 1,
              comparisons: [
                {
                  output: "step1-latent",
                  reference: "f16+dit8",
                  practical: "f16+dit8-a8-attn8-s16",
                  relRms: 0.0123,
                  maxAbs: 0.5,
                  band: { metric: "relRms", floor: 0.001, ceiling: 0.025 },
                },
                {
                  output: "zz-unbanded",
                  reference: "f16+dit8",
                  practical: "f16+dit8-a8-attn8-s16",
                  relRms: null,
                  maxAbs: null,
                },
              ],
            },
          ]),
        ),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 0, result.stderr);
        assertStringIncludes(result.stdout, "### A/B of f16+dit8-a8-attn8-s16-512");
        assertStringIncludes(
          result.stdout,
          "| step1-latent | deno-a | f16+dit8 → f16+dit8-a8-attn8-s16 | 0.0123 | 0.5 | 0.001〜0.025 |",
        );
        assertStringIncludes(
          result.stdout,
          "| zz-unbanded | deno-a | f16+dit8 → f16+dit8-a8-attn8-s16 | null | null | not derived |",
        );
      },
    );
  });

  it("warns on two results directories with the same family, environment key and date (exit code stays 0)", async () => {
    await withRoot(
      async (root) => {
        // The directory names differ, but the documents report the same environment key (what a
        // hand copy produces).
        await writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [passing("base")]),
        );
        await writeSeat(
          root,
          "copied",
          "2026-09-21_anima",
          document("anima", "deno-a", [{ id: "base", status: "fail", elapsedMs: 2 }]),
        );
      },
      async (root) => {
        const result = await run(["--root", root, "--json"]);
        assertEquals(result.code, 0, result.stderr);
        const report = JSON.parse(result.stdout);
        assertEquals(report.warnings.length, 1);
        assertStringIncludes(
          report.warnings[0],
          "two results directories with the same family, environment key and date (anima / deno-a / 2026-09-21)",
        );
        // Both paths reach the reader (it does not stay silent about which one was taken).
        assertStringIncludes(report.warnings[0], `${root}/deno-a/2026-09-21_anima/results.json`);
        assertStringIncludes(report.warnings[0], `${root}/copied/2026-09-21_anima/results.json`);
      },
    );
  });
});

describe("verify-diff CLI: an unreadable root", () => {
  it("exits with 1 when a results directory has no results.json", async () => {
    await withRoot(
      async (root) => {
        await mkdir(`${root}/deno-a/2026-09-21_anima`, { recursive: true });
      },
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "the results directory has no results.json (broken)");
        assertEquals(result.stdout, "");
      },
    );
  });

  it("exits with 1 on a document that cannot be read as JSON", async () => {
    await withRoot(
      (root) => writeSeat(root, "deno-a", "2026-09-21_anima", "{ broken"),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "cannot be read as JSON");
        assertEquals(result.stdout, "");
      },
    );
  });

  it("exits with 1 on a document with a different schema", async () => {
    await withRoot(
      (root) =>
        writeSeat(root, "deno-a", "2026-09-21_anima", {
          ...document("anima", "deno-a", [passing("base")]),
          schema: 2,
        }),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "schema is not 1 (2)");
      },
    );
  });

  it("exits with 1 on a document with a duplicate case ID", async () => {
    await withRoot(
      (root) =>
        writeSeat(
          root,
          "deno-a",
          "2026-09-21_anima",
          document("anima", "deno-a", [
            passing("base"),
            {
              id: "base",
              status: "fail",
              elapsedMs: 2,
            },
          ]),
        ),
      async (root) => {
        const result = await run(["--root", root]);
        assertEquals(result.code, 1);
        assertStringIncludes(result.stderr, "duplicate case ID 'base'");
      },
    );
  });

  it("exits with 1 when the root does not exist", async () => {
    const result = await run(["--root", "/nonexistent-verify-diff-root"]);
    assertEquals(result.code, 1);
    assertStringIncludes(result.stderr, "the results root does not exist");
  });
});

describe("verify-diff CLI: the spelling of --root", () => {
  it("reads a root containing `%` by its real name (does not decode `%41` to `A`)", async () => {
    const parent = await makeTempDir({ prefix: "verify-diff-escape-" });
    try {
      // Place the real `x%41y` next to `xAy`, where `%41` decodes to (the shape that silently reads
      // another place).
      await writeSeat(
        `${parent}/x%41y`,
        "deno-a",
        "2026-09-21_anima",
        document("anima", "deno-a", [passing("literal")]),
      );
      await writeSeat(
        `${parent}/xAy`,
        "deno-a",
        "2026-09-21_anima",
        document("anima", "deno-a", [passing("decoded")]),
      );

      const result = await run(["--root", `${parent}/x%41y`]);

      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "literal");
      assert(
        !result.stdout.includes("decoded"),
        `reads another root, with %41 decoded:\n${result.stdout}`,
      );
    } finally {
      await remove(parent, { recursive: true });
    }
  });

  it("also reads a root containing `#` (does not cut it off as a URL fragment)", async () => {
    const parent = await makeTempDir({ prefix: "verify-diff-hash-" });
    try {
      const root = `${parent}/hash#dir`;
      await writeSeat(
        root,
        "deno-a",
        "2026-09-21_anima",
        document("anima", "deno-a", [passing("base")]),
      );

      const result = await run(["--root", root]);

      assertEquals(result.code, 0, result.stderr);
      assertStringIncludes(result.stdout, "| base | pass |");
    } finally {
      await remove(parent, { recursive: true });
    }
  });
});

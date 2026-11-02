// Checks of the fetch command against a fake `hf` (touches neither the network nor outputs/ — the
// local root is a temporary directory every time).
//
// What is pinned here is the contract the assets gate relies on: `.revision` stands for "a
// download of that revision completed". So the record is written only after `hf` exited 0, never
// after a failure or a kill, and a stale record is gone before the download starts. The fake also
// proves the arguments `hf` receives (repo, dataset type, pinned revision, local dir, includes).

import { fileURLToPath } from "node:url";

import {
  isNotFound,
  makeTempDir,
  readTextFile,
  remove,
  stat,
  writeTextFile,
} from "@karume/test-support/fs";
import { TEST_ASSETS } from "@karume/test-support/test-assets";
import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";

import { downloadArgs, fetchTestAssets, hfCommand, parseArgs } from "../fetch.ts";

const PIN = {
  repo: "hdae/karume-test-assets",
  revision: "0123456789abcdef0123456789abcdef01234567",
};

/**
 * A stand-in for `hf`: `node fake-hf.mjs <record> <mode> <hf args>…`. It records the `hf`
 * arguments and whether `<local dir>/.revision` existed when it started, writes one "downloaded"
 * file, then exits 0 (`ok`), exits 3 (`fail`) or kills itself with SIGTERM (`signal`).
 */
const FAKE_HF = `
import { existsSync, writeFileSync } from "node:fs";
const [record, mode, ...args] = process.argv.slice(2);
const localDir = args[args.indexOf("--local-dir") + 1];
writeFileSync(record, JSON.stringify({ args, revisionAtStart: existsSync(localDir + "/.revision") }));
writeFileSync(localDir + "/downloaded.txt", "asset");
if (mode === "fail") process.exit(3);
if (mode === "signal") process.kill(process.pid, "SIGTERM");
`;

/** Whether `url` exists (NotFound is false; any other failure propagates). */
const exists = (url: URL): Promise<boolean> =>
  stat(url).then(
    () => true,
    (cause: unknown) => {
      if (isNotFound(cause)) return false;
      throw cause;
    },
  );

type Recorded = { readonly args: readonly string[]; readonly revisionAtStart: boolean };

type Rig = {
  readonly root: URL;
  readonly revisionFile: URL;
  readonly command: (mode: "ok" | "fail" | "signal") => readonly string[];
  readonly recorded: () => Promise<Recorded>;
};

/** Runs `body` on a temporary directory holding the fake `hf` and an empty local root. */
const withRig = async (body: (rig: Rig) => Promise<void>): Promise<void> => {
  const dir = await makeTempDir({ prefix: "karume-test-assets-" });
  const base = new URL(`file://${dir}/`);
  const fake = new URL("fake-hf.mjs", base);
  const record = new URL("record.json", base);
  await writeTextFile(fake, FAKE_HF);
  const root = new URL("test-assets/", base);
  try {
    await body({
      root,
      revisionFile: new URL(".revision", root),
      command: (mode) => [process.execPath, fileURLToPath(fake), fileURLToPath(record), mode],
      recorded: async () => JSON.parse(await readTextFile(record)) as Recorded,
    });
  } finally {
    await remove(base, { recursive: true });
  }
};

describe("fetch: a download that completes", () => {
  it("passes the pinned repo, the dataset type, the revision, the local root and every include to hf", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({
        pin: PIN,
        root: rig.root,
        includes: ["goldens/lucida-1024/**", "corpus/anima/**"],
        command: rig.command("ok"),
      });
      assertEquals((await rig.recorded()).args, [
        "download",
        PIN.repo,
        "--repo-type",
        "dataset",
        "--revision",
        PIN.revision,
        "--local-dir",
        fileURLToPath(rig.root),
        "--exclude",
        ".revision",
        "--include",
        "goldens/lucida-1024/**",
        "--include",
        "corpus/anima/**",
      ]);
    });
  });

  it("records the fetched revision in .revision after hf exited 0", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("ok") });
      assertEquals(await readTextFile(rig.revisionFile), `${PIN.revision}\n`);
    });
  });

  it("removes a stale .revision before hf starts (hf sees none while it downloads)", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("ok") });
      await writeTextFile(rig.revisionFile, `${"f".repeat(40)}\n`);
      await fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("ok") });
      assertEquals((await rig.recorded()).revisionAtStart, false);
      assertEquals(await readTextFile(rig.revisionFile), `${PIN.revision}\n`);
    });
  });
});

describe("fetch: a partial download (--include) is not recorded as complete", () => {
  const includes = ["goldens/lucida-1024/**"];

  it("writes no .revision on a root without one (the gate stays red)", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({ pin: PIN, root: rig.root, includes, command: rig.command("ok") });
      assertEquals(await exists(rig.revisionFile), false);
    });
  });

  it("keeps the record of a complete fetch of the same revision", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("ok") });
      await fetchTestAssets({ pin: PIN, root: rig.root, includes, command: rig.command("ok") });
      assertEquals(await readTextFile(rig.revisionFile), `${PIN.revision}\n`);
    });
  });

  it("drops the record of another revision before hf starts", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("ok") });
      await writeTextFile(rig.revisionFile, `${"f".repeat(40)}\n`);
      await fetchTestAssets({ pin: PIN, root: rig.root, includes, command: rig.command("ok") });
      assertEquals((await rig.recorded()).revisionAtStart, false);
      assertEquals(await exists(rig.revisionFile), false);
    });
  });
});

describe("fetch: a download that does not complete leaves no .revision", () => {
  it("hf exits non-zero: the fetch fails naming the exit code, and the stale record is gone", async () => {
    await withRig(async (rig) => {
      await fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("ok") });
      await assertRejects(
        () => fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("fail") }),
        Error,
        "exited with 3",
      );
      assertEquals(await exists(rig.revisionFile), false);
    });
  });

  it("hf is killed: the fetch fails naming the signal and writes no record", async () => {
    await withRig(async (rig) => {
      await assertRejects(
        () => fetchTestAssets({ pin: PIN, root: rig.root, command: rig.command("signal") }),
        Error,
        "killed by SIGTERM",
      );
      assertEquals(await exists(rig.revisionFile), false);
    });
  });

  it("hf is missing: the fetch fails naming the command and KARUME_HF_BIN", async () => {
    await withRig(async (rig) => {
      const missing = fileURLToPath(new URL("no-such-hf", rig.root));
      const error = await assertRejects(
        () => fetchTestAssets({ pin: PIN, root: rig.root, command: [missing] }),
        Error,
        `'${missing}' was not found`,
      );
      assertEquals(error.message.includes("KARUME_HF_BIN"), true);
      assertEquals(await exists(rig.revisionFile), false);
    });
  });
});

describe("fetch: the pin", () => {
  it("refuses the placeholder revision before running hf", async () => {
    await withRig(async (rig) => {
      await assertRejects(
        () =>
          fetchTestAssets({
            pin: { repo: PIN.repo, revision: "0".repeat(40) },
            root: rig.root,
            command: rig.command("ok"),
          }),
        Error,
        "placeholder revision",
      );
      assertEquals(await exists(new URL("downloaded.txt", rig.root)), false);
    });
  });

  it("refuses a revision that is not a commit SHA (a branch would move under the tests)", async () => {
    await withRig(async (rig) => {
      await assertRejects(
        () =>
          fetchTestAssets({
            pin: { repo: PIN.repo, revision: "main" },
            root: rig.root,
            command: rig.command("ok"),
          }),
        Error,
        "40-hex commit SHA",
      );
    });
  });

  it("the default arguments name the tracked pin", () => {
    const args = downloadArgs(TEST_ASSETS, new URL("file:///root/"), []);
    assertEquals(args.slice(0, 6), [
      "download",
      TEST_ASSETS.repo,
      "--repo-type",
      "dataset",
      "--revision",
      TEST_ASSETS.revision,
    ]);
  });
});

describe("fetch: the command line", () => {
  it("collects every --include in order", () => {
    assertEquals(parseArgs(["--include", "a/**", "--include", "b/**"]), ["a/**", "b/**"]);
    assertEquals(parseArgs([]), []);
  });

  it("fails on an unknown argument and on --include without a glob", () => {
    assertThrows(() => parseArgs(["--revision", "x"]), Error, "unknown argument '--revision'");
    assertThrows(() => parseArgs(["--include"]), Error, "--include needs a glob");
    assertThrows(() => parseArgs(["--include", "--include"]), Error, "--include needs a glob");
  });

  it("hf is KARUME_HF_BIN when set, otherwise hf on PATH", () => {
    assertEquals(hfCommand({ KARUME_HF_BIN: "/opt/hf/bin/hf" }), ["/opt/hf/bin/hf"]);
    assertEquals(hfCommand({}), ["hf"]);
    assertEquals(hfCommand({ KARUME_HF_BIN: "" }), ["hf"]);
  });
});

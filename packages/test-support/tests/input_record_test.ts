/**
 * input-record: the files a test file reads, the directories it lists and the commands it starts are
 * seen where they pass through `node:fs` and `node:child_process` — whichever way the caller imported
 * the function — and the wrapped functions behave as before.
 */

import { assert, assertEquals } from "@std/assert";
import { execFile, execFileSync } from "node:child_process";
import fs, { readdirSync, readFileSync } from "node:fs";
import { open, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { afterAll, describe, it } from "vite-plus/test";
import { inputsOfFile, opensForReading, pathOfArgument } from "../src/input-record.ts";

const dir = fs.mkdtempSync(join(tmpdir(), "karume-input-record-"));
const fileAt = (name: string): string => {
  const path = join(dir, name);
  fs.writeFileSync(path, name);
  return path;
};

afterAll(() => {
  fs.rmSync(dir, { recursive: true });
});

describe("the input record (installed by the setup file)", () => {
  it("sees a read through a named import, the default import and the promises API", async () => {
    const named = fileAt("named.txt");
    const viaDefault = fileAt("default.txt");
    const viaPromise = fileAt("promise.txt");
    const viaHandle = fileAt("handle.txt");
    assertEquals(readFileSync(named, "utf8"), "named.txt");
    assertEquals(fs.readFileSync(pathToFileURL(viaDefault), "utf8"), "default.txt");
    assertEquals(await readFile(viaPromise, "utf8"), "promise.txt");
    const handle = await open(viaHandle);
    assertEquals(await handle.readFile("utf8"), "handle.txt");
    await handle.close();
    const { files } = inputsOfFile();
    for (const path of [named, viaDefault, viaPromise, viaHandle]) {
      assert(files.includes(path), `${path} is not in the record`);
    }
  });

  it("records an attempt on a file that does not exist, and leaves out a file opened only for writing", () => {
    const missing = join(dir, "missing.txt");
    const written = join(dir, "written.txt");
    let thrown = false;
    try {
      readFileSync(missing);
    } catch {
      thrown = true;
    }
    assert(thrown, "reading a missing file still throws");
    fs.closeSync(fs.openSync(written, "w"));
    const { files } = inputsOfFile();
    assert(files.includes(missing));
    assert(!files.includes(written));
  });

  it("records a directory listing", () => {
    fileAt("listed.txt");
    assert(readdirSync(dir).includes("listed.txt"));
    assert(inputsOfFile().directories.includes(dir));
  });

  it("records a command with its arguments, and promisify(execFile) still resolves with stdout and stderr", async () => {
    const script = "process.stdout.write('out')";
    assertEquals(execFileSync(process.execPath, ["-e", script], { encoding: "utf8" }), "out");
    const result = await promisify(execFile)(process.execPath, ["-e", script, "again"]);
    assertEquals(result.stdout, "out");
    assertEquals(result.stderr, "");
    const { commands } = inputsOfFile();
    assert(commands.some((command) => command.join(" ") === `${process.execPath} -e ${script}`));
    assert(commands.some((command) => command.at(-1) === "again"));
  });
});

describe("pathOfArgument / opensForReading", () => {
  it("turns a string, a file URL and a buffer into an absolute path, and nothing else", () => {
    assertEquals(pathOfArgument("/a/b"), "/a/b");
    assertEquals(pathOfArgument(new URL("file:///a/b%20c")), "/a/b c");
    assertEquals(pathOfArgument(new TextEncoder().encode("/a/b")), "/a/b");
    assertEquals(pathOfArgument(3), undefined);
    assertEquals(pathOfArgument(new URL("https://example.com/a")), undefined);
  });

  it("counts the default and every flag that can read, and not a write-only flag", () => {
    for (const flags of [
      undefined,
      "r",
      "r+",
      "w+",
      "a+",
      fs.constants.O_RDONLY,
      fs.constants.O_RDWR,
    ]) {
      assert(opensForReading(flags), `${String(flags)} can read`);
    }
    for (const flags of ["w", "a", "wx", fs.constants.O_WRONLY | fs.constants.O_CREAT]) {
      assert(!opensForReading(flags), `${String(flags)} cannot read`);
    }
  });
});

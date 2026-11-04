// Behaviour tests of the read-if-present helpers (`helpers/read-if-present.ts`). Needs neither a
// GPU nor real assets.
//
// These helpers are where the asset tests decide whether to SKIP, so confusing "absent" with
// "unreadable" would turn a check that never ran into a silent green SKIP. The tests pin that only
// NotFound is folded into `undefined` and that anything else (here, a path to a directory) throws.
//
// NOTE: the error type for reading a directory differs between operating systems (some environments
// have no `Deno.errors.IsADirectory`), so the tests observe "threw instead of returning
// `undefined`" rather than the type.

import { makeTempDir, mkdir, remove, writeTextFile } from "@karume/test-support/fs";
import { assertEquals, assertRejects } from "@std/assert";
import { afterAll, beforeAll, describe, it } from "vite-plus/test";
import { readFileIfPresent, readTextIfPresent } from "./helpers/read-if-present.ts";

describe("readTextIfPresent / readFileIfPresent", () => {
  let root: URL;
  beforeAll(async () => {
    const dir = await makeTempDir({ prefix: "karume-read-if-present-" });
    root = new URL(`file://${dir}/`);
    await writeTextFile(new URL("present.txt", root), "中身");
    await mkdir(new URL("directory/", root));
  });
  afterAll(async () => {
    await remove(root, { recursive: true });
  });

  describe("when the path does not exist", () => {
    it("the text read returns undefined", async () => {
      assertEquals(await readTextIfPresent(new URL("absent.txt", root)), undefined);
    });
    it("the byte read returns undefined", async () => {
      assertEquals(await readFileIfPresent(new URL("absent.bin", root)), undefined);
    });
  });

  describe("when the path points to a directory (a failure other than NotFound)", () => {
    it("the text read throws instead of folding into undefined", async () => {
      await assertRejects(() => readTextIfPresent(new URL("directory", root)));
    });
    it("the byte read throws instead of folding into undefined", async () => {
      await assertRejects(() => readFileIfPresent(new URL("directory", root)));
    });
  });

  describe("when a file exists at the path", () => {
    it("the text read returns the contents", async () => {
      assertEquals(await readTextIfPresent(new URL("present.txt", root)), "中身");
    });
    it("the byte read returns the content bytes", async () => {
      assertEquals(
        await readFileIfPresent(new URL("present.txt", root)),
        new TextEncoder().encode("中身"),
      );
    });
  });
});

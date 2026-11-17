import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vite-plus/test";

import { nodeInputs } from "../node-inputs.ts";
import { inputSegments } from "../src/inputs.ts";

const BYTES = Uint8Array.from({ length: 32 }, (_, index) => index * 3);

let repository = "";

beforeAll(async () => {
  repository = await mkdtemp(join(tmpdir(), "karume-workbench-inputs-"));
  await mkdir(join(repository, "inputs", "anima"), { recursive: true });
  await writeFile(join(repository, "inputs", "anima", "model.safetensors"), BYTES);
  await writeFile(join(repository, "secret.txt"), "outside");
  await symlink(join(repository, "secret.txt"), join(repository, "inputs", "escape.txt"));
});

afterAll(async () => {
  await rm(repository, { recursive: true, force: true });
});

describe("the path of an input file", () => {
  it("is split into its segments", () => {
    expect(inputSegments("anima/a b/model.safetensors")).toEqual([
      "anima",
      "a b",
      "model.safetensors",
    ]);
  });

  it.each(["", "/abs.bin", "a//b", "../x", "a/./b", "a/..", "a\\b", "a\u0000b"])(
    "refuses %j, naming it",
    (path) => {
      expect(() => inputSegments(path)).toThrow(`input file '${path}': not a path inside inputs/`);
    },
  );
});

describe("the Node run's openInput", () => {
  it("reads a file under inputs/ by ranges and records its repository path, size and how it was read", async () => {
    const opened = await nodeInputs(repository)("anima/model.safetensors");
    expect(opened.facts).toEqual({
      path: "inputs/anima/model.safetensors",
      size: 32,
      read: "directory",
    });
    expect(await opened.file.read(4, 8)).toEqual(BYTES.slice(4, 12));
    expect(await opened.file.read(0, 0)).toEqual(new Uint8Array(0));
  });

  it("refuses a range past the end of the file", async () => {
    const opened = await nodeInputs(repository)("anima/model.safetensors");
    await expect(opened.file.read(30, 4)).rejects.toThrow("read 2 bytes");
  });

  it("refuses a link whose real path leaves inputs/, and a missing file", async () => {
    await expect(nodeInputs(repository)("escape.txt")).rejects.toThrow("leaves inputs/");
    await expect(nodeInputs(repository)("anima/missing.bin")).rejects.toThrow();
  });
});

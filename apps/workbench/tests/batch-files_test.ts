import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vite-plus/test";

import { UNTRACKED_BATCHES_DIRECTORY } from "../src/batch.ts";
import { handleBatchesGet, MAX_BATCH_BYTES } from "../src/batch-files.ts";

let root: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "workbench-batches-"));
});

afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

const directory = (): string => join(root, UNTRACKED_BATCHES_DIRECTORY);

const put = async (name: string, text: string): Promise<void> => {
  await mkdir(directory(), { recursive: true });
  await writeFile(join(directory(), name), text);
};

const get = (method = "GET", maxBytes = MAX_BATCH_BYTES) =>
  handleBatchesGet({ method }, { repositoryRoot: root, maxBytes });

const listedOf = async (maxBytes = MAX_BATCH_BYTES): Promise<unknown> => {
  const response = await get("GET", maxBytes);
  expect(response.status).toBe(200);
  const body: unknown = JSON.parse(response.body);
  return body;
};

describe("GET /workbench/batches", () => {
  it("answers every .json file of inputs/workbench/batches/ in file-name order, with its text, uncached", async () => {
    await put("b-second.json", '{"title":"b"}');
    await put("a-first.json", '{"title":"a"}');
    await put("Not_Kebab.json", "{}");
    const response = await get();
    expect(response.status).toBe(200);
    expect(response.headers).toEqual({
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    });
    expect(JSON.parse(response.body)).toEqual({
      batches: [
        { id: "Not_Kebab", text: "{}" },
        { id: "a-first", text: '{"title":"a"}' },
        { id: "b-second", text: '{"title":"b"}' },
      ],
    });
  });

  it("answers an empty list when the directory is missing", async () => {
    expect(await listedOf()).toEqual({ batches: [] });
  });

  it("leaves out what is not a .json file directly in the directory", async () => {
    await put("notes.txt", "x");
    await put("batch.json.bak", "x");
    await put("kept.json", "{}");
    await mkdir(join(directory(), "nested.json"));
    await mkdir(join(directory(), "deeper"));
    await writeFile(join(directory(), "deeper", "inner.json"), "{}");
    expect(await listedOf()).toEqual({ batches: [{ id: "kept", text: "{}" }] });
  });

  it("lists a file over the cap with that problem in place of its text, and the cap itself with its text", async () => {
    await put("at-cap.json", "x".repeat(16));
    await put("over-cap.json", "x".repeat(17));
    expect(await listedOf(16)).toEqual({
      batches: [
        { id: "at-cap", text: "x".repeat(16) },
        { id: "over-cap", problem: "the file is larger than 16 bytes, so it was not read" },
      ],
    });
  });

  it("lists a file that is not UTF-8 with that problem", async () => {
    await mkdir(directory(), { recursive: true });
    await writeFile(join(directory(), "latin.json"), Uint8Array.of(0x7b, 0xe9, 0x7d));
    expect(await listedOf()).toEqual({
      batches: [{ id: "latin", problem: "the file is not UTF-8 text" }],
    });
  });

  it("answers 405 with Allow for anything but GET", async () => {
    for (const method of ["POST", "PUT", "HEAD"]) {
      const response = await get(method);
      expect(response.status).toBe(405);
      expect(response.headers["Allow"]).toBe("GET");
    }
  });

  it("refuses a file one byte over 1 MiB by default", async () => {
    await put("huge.json", " ".repeat(1024 * 1024 + 1));
    expect(await listedOf()).toEqual({
      batches: [
        { id: "huge", problem: "the file is larger than 1048576 bytes, so it was not read" },
      ],
    });
  });
});

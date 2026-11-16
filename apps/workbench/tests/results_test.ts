import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vite-plus/test";

import { RECORD_FORMAT } from "../src/record.ts";
import {
  adapterDirectory,
  handleRecordPost,
  type RecordPostRequest,
  type RecordTarget,
  RESULTS_DIRECTORY,
  writeRecord,
} from "../src/results.ts";

let root: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "workbench-results-"));
});

afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

const NOW = new Date("2026-11-16T09:30:12.345Z");
const bytesOf = (text: string): Uint8Array => new TextEncoder().encode(text);

describe("adapterDirectory", () => {
  it("names a browser's adapter by its description: lower case, runs outside [a-z0-9] as one hyphen, trimmed", () => {
    expect(
      adapterDirectory("browser", {
        vendor: "nvidia",
        architecture: "blackwell",
        device: "0x2c05",
        description: "  NVIDIA GeForce RTX 5070 Ti  ",
      }),
    ).toBe("browser-nvidia-geforce-rtx-5070-ti");
  });

  it("names Node's adapter by its device, because Dawn puts the driver version in the description", () => {
    expect(
      adapterDirectory("node", {
        vendor: "nvidia",
        architecture: "ampere",
        device: "nvidia-geforce-rtx-3080-ti",
        description: "NVIDIA: 615.78.08 615.78.8.0",
      }),
    ).toBe("node-nvidia-geforce-rtx-3080-ti");
  });

  it("falls back to vendor-architecture, then unknown-adapter", () => {
    expect(
      adapterDirectory("browser", {
        description: "",
        device: "0x0000",
        vendor: "apple",
        architecture: "metal-3",
      }),
    ).toBe("browser-apple-metal-3");
    expect(adapterDirectory("node", { device: "", vendor: "intel", architecture: "" })).toBe(
      "node-intel",
    );
    expect(adapterDirectory("browser", {})).toBe("browser-unknown-adapter");
    expect(adapterDirectory("node", { description: "named", device: 42, vendor: null })).toBe(
      "node-unknown-adapter",
    );
  });

  it("keeps at most 80 characters of the name and never ends on a hyphen", () => {
    expect(adapterDirectory("browser", { description: `${"a".repeat(79)} b` })).toBe(
      `browser-${"a".repeat(79)}`,
    );
    expect(adapterDirectory("browser", { description: "x".repeat(200) })).toBe(
      `browser-${"x".repeat(80)}`,
    );
  });
});

describe("writeRecord", () => {
  const target: RecordTarget = {
    entryId: "kernel-check",
    host: "browser",
    adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
  };

  it("writes the bytes unchanged to <host>-<adapter name>/<entry id>_<UTC stamp>.json and returns the relative path", async () => {
    const bytes = bytesOf('{"x": 1}\n');
    const path = await writeRecord(root, target, bytes, NOW);
    expect(path).toBe(
      `${RESULTS_DIRECTORY}/browser-nvidia-blackwell/kernel-check_2026-11-16T09-30-12.json`,
    );
    expect(new Uint8Array(await readFile(join(root, path)))).toEqual(bytes);
  });

  it("adds -1, -2 when the name is taken and never replaces an existing file", async () => {
    const first = await writeRecord(root, target, bytesOf("first"), NOW);
    const second = await writeRecord(root, target, bytesOf("second"), NOW);
    const third = await writeRecord(root, target, bytesOf("third"), NOW);
    expect(second).toBe(first.replace(".json", "-1.json"));
    expect(third).toBe(first.replace(".json", "-2.json"));
    expect(await readFile(join(root, first), "utf8")).toBe("first");
    expect(await readFile(join(root, second), "utf8")).toBe("second");
  });

  it("creates the file exclusively: a file someone else put there first stays as it was", async () => {
    const path = await writeRecord(root, target, bytesOf("mine"), NOW);
    await writeFile(join(root, path), "theirs");
    const next = await writeRecord(root, target, bytesOf("again"), NOW);
    expect(await readFile(join(root, path), "utf8")).toBe("theirs");
    expect(next).not.toBe(path);
  });

  it("refuses an entry id that is not kebab-case", async () => {
    await expect(
      writeRecord(root, { ...target, entryId: "../x" }, bytesOf("{}"), NOW),
    ).rejects.toThrow("kebab-case");
  });
});

const ORIGIN = "http://127.0.0.1:8791";

const validRecord = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: RECORD_FORMAT,
    entry: { id: "kernel-check", parameters: {} },
    environment: {
      host: "browser",
      adapter: { vendor: "nvidia", architecture: "blackwell", device: "", description: "" },
    },
    ...overrides,
  });

const chunks = async function* (...parts: readonly Uint8Array[]): AsyncGenerator<Uint8Array> {
  for (const part of parts) yield part;
};

const post = (
  overrides: Partial<Omit<RecordPostRequest, "headers">> & {
    readonly headers?: Readonly<Record<string, string | undefined>>;
    readonly text?: string;
  } = {},
  maxBytes = 1024,
) =>
  handleRecordPost(
    {
      method: overrides.method ?? "POST",
      scheme: overrides.scheme ?? "http",
      headers: {
        host: "127.0.0.1:8791",
        origin: ORIGIN,
        "content-type": "application/json",
        ...overrides.headers,
      },
      body: overrides.body ?? chunks(bytesOf(overrides.text ?? validRecord())),
    },
    { repositoryRoot: root, now: () => NOW, maxBytes },
  );

/** The `path` of a 201 body. */
const pathOf = (body: string): string => {
  const parsed: unknown = JSON.parse(body);
  if (typeof parsed !== "object" || parsed === null || !("path" in parsed)) {
    throw new Error(`no path in ${body}`);
  }
  if (typeof parsed.path !== "string") throw new Error(`no path in ${body}`);
  return parsed.path;
};

const filesUnder = async (directory: string): Promise<string[]> =>
  (await readdir(directory, { recursive: true, withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name));

describe("handleRecordPost", () => {
  it("writes the posted bytes unchanged and answers 201 with the repository-relative path", async () => {
    const text = validRecord();
    const response = await post({ text });
    expect(response.status).toBe(201);
    const path = pathOf(response.body);
    expect(path).toBe(
      `${RESULTS_DIRECTORY}/browser-nvidia-blackwell/kernel-check_2026-11-16T09-30-12.json`,
    );
    expect(await readFile(join(root, path), "utf8")).toBe(text);
  });

  it("answers 405 with Allow for anything but POST", async () => {
    const response = await post({ method: "PUT" });
    expect(response.status).toBe(405);
    expect(response.headers["Allow"]).toBe("POST");
  });

  it("answers 415 without Content-Type application/json", async () => {
    expect((await post({ headers: { "content-type": "text/plain" } })).status).toBe(415);
    expect((await post({ headers: { "content-type": undefined } })).status).toBe(415);
    expect(
      (await post({ headers: { "content-type": "application/json; charset=utf-8" } })).status,
    ).toBe(201);
  });

  it("answers 403 when the Origin is missing or another site's, and writes nothing", async () => {
    expect((await post({ headers: { origin: undefined } })).status).toBe(403);
    expect((await post({ headers: { origin: "https://example.com" } })).status).toBe(403);
    expect((await post({ headers: { origin: "http://127.0.0.1:8790" } })).status).toBe(403);
    expect((await post({ scheme: "https" })).status).toBe(403);
    expect(await filesUnder(root)).toEqual([]);
  });

  it("answers 413 for a body one byte over the cap, without reading past it", async () => {
    const text = validRecord();
    const size = bytesOf(text).byteLength;
    expect((await post({ text }, size)).status).toBe(201);
    let pulled = 0;
    const counted = async function* (): AsyncGenerator<Uint8Array> {
      for (const byte of bytesOf(`${text} `)) {
        pulled += 1;
        yield Uint8Array.of(byte);
      }
    };
    const response = await post({ body: counted() }, size);
    expect(response.status).toBe(413);
    expect(response.headers["Connection"]).toBe("close");
    expect(pulled).toBe(size + 1);
  });

  it("answers 422 with a one-line reason for a body that is not a workbench record", async () => {
    const reasons = [
      await post({ text: "{not json" }),
      await post({ text: validRecord({ format: "karume-playground-kernel-check/1" }) }),
      await post({ text: validRecord({ entry: { id: "Kernel_Check" } }) }),
      await post({ text: validRecord({ entry: {} }) }),
      await post({ text: validRecord({ environment: { host: "browser" } }) }),
      await post({
        text: validRecord({ environment: { host: "../x", adapter: { description: "gpu" } } }),
      }),
      await post({ text: validRecord({ environment: { adapter: { description: "gpu" } } }) }),
      await post({ text: "[]" }),
    ];
    for (const response of reasons) {
      expect(response.status).toBe(422);
      expect(response.body.trim().split("\n")).toHaveLength(1);
    }
    expect(await filesUnder(root)).toEqual([]);
  });

  it("cannot be steered out of outputs/workbench by the entry id or the adapter strings", async () => {
    expect((await post({ text: validRecord({ entry: { id: "../../../escape" } }) })).status).toBe(
      422,
    );
    const response = await post({
      text: validRecord({
        environment: {
          host: "browser",
          adapter: {
            description: "../../../../tmp/escape",
            device: "/etc/passwd",
            vendor: "..",
            architecture: "\\..",
          },
        },
      }),
    });
    expect(response.status).toBe(201);
    const path = pathOf(response.body);
    const resultsRoot = resolve(root, RESULTS_DIRECTORY) + sep;
    expect(resolve(root, path).startsWith(resultsRoot)).toBe(true);
    const written = await filesUnder(root);
    expect(written).toHaveLength(1);
    expect(written.every((file) => resolve(file).startsWith(resultsRoot))).toBe(true);
  });
});

import { afterEach, describe, expect, it, vi } from "vite-plus/test";

import { saveRecord } from "../src/shell/save.ts";

/** Answers the record's POST with `answer` (a function, so a test can make the request fail). */
const serve = (answer: () => Promise<Response>): string[] => {
  const posted: string[] = [];
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    posted.push(`${init?.method ?? "GET"} ${url}`);
    return answer();
  });
  return posted;
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("saveRecord under the dev server", () => {
  it("posts the record and answers where the dev server wrote it", async () => {
    const posted = serve(() =>
      Promise.resolve(Response.json({ path: "outputs/workbench/x/a.json" }, { status: 201 })),
    );
    expect(import.meta.env.DEV).toBe(true);
    expect(await saveRecord("{}\n")).toEqual({
      kind: "saved",
      path: "outputs/workbench/x/a.json",
    });
    expect(posted).toEqual(["POST /workbench/results"]);
  });

  it("names the status and the text of an answer other than 201", async () => {
    serve(() => Promise.resolve(new Response("not a record\n", { status: 422 })));
    expect(await saveRecord("{}\n")).toEqual({
      kind: "not-saved",
      reason: "the dev server answered 422: not a record",
    });
  });

  it("says that the dev server could not be reached when the request fails", async () => {
    serve(() => Promise.reject(new TypeError("Failed to fetch")));
    expect(await saveRecord("{}\n")).toEqual({
      kind: "not-saved",
      reason: "the dev server could not be reached (TypeError: Failed to fetch)",
    });
  });

  it("says that the dev server answered no path for a 201 without one", async () => {
    serve(() => Promise.resolve(Response.json({ written: true }, { status: 201 })));
    expect(await saveRecord("{}\n")).toEqual({
      kind: "not-saved",
      reason: "the dev server answered no path",
    });
  });

  it("reports an answer it cannot read as not saved, with the reason, instead of rejecting", async () => {
    serve(() => Promise.resolve(new Response("written", { status: 201 })));
    const saved = await saveRecord("{}\n");
    expect(saved.kind).toBe("not-saved");
    expect(saved.kind === "not-saved" ? saved.reason : "").toMatch(
      /^the dev server answered 201, but its answer could not be read \(SyntaxError: .+\)$/,
    );
  });

  it("reports an error answer whose body cannot be read as not saved, with the reason", async () => {
    const body = new ReadableStream<Uint8Array<ArrayBuffer>>({
      start: (controller) => controller.error(new Error("connection reset")),
    });
    serve(() => Promise.resolve(new Response(body, { status: 500 })));
    expect(await saveRecord("{}\n")).toEqual({
      kind: "not-saved",
      reason:
        "the dev server answered 500, but its answer could not be read (Error: connection reset)",
    });
  });
});

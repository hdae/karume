import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { containedPath, createHandler, fileResponse } from "./server.ts";

describe("browser benchmark asset server", () => {
  it("serves exact byte ranges and rejects ranges outside the asset", async () => {
    const dir = await Deno.makeTempDir();
    try {
      const path = `${dir}/weights.bin`;
      await Deno.writeFile(path, new Uint8Array([10, 20, 30, 40, 50]));
      const range = await fileResponse(
        new Request("http://localhost/model", { headers: { Range: "bytes=1-3" } }),
        path,
      );
      assertEquals(range.status, 206);
      assertEquals(range.headers.get("Content-Range"), "bytes 1-3/5");
      assertEquals(new Uint8Array(await range.arrayBuffer()), new Uint8Array([20, 30, 40]));
      for (
        const value of [
          "bytes=0-5",
          "bytes=3-1",
          "bytes=0-1,3-4",
          "bytes=9007199254740992-9007199254740993",
        ]
      ) {
        const response = await fileResponse(
          new Request("http://localhost/model", { headers: { Range: value } }),
          path,
        );
        assertEquals(response.status, 416);
        assertEquals(response.headers.get("Content-Range"), "bytes */5");
        await response.body?.cancel();
      }
      const head = await fileResponse(
        new Request("http://localhost/model", { method: "HEAD" }),
        path,
      );
      assertEquals(head.headers.get("Content-Length"), "5");
      assertEquals(head.body, null);
      const cancelled = await fileResponse(new Request("http://localhost/model"), path);
      await cancelled.body?.cancel();
      const inFlight = await fileResponse(new Request("http://localhost/model"), path);
      const reader = inFlight.body!.getReader();
      await Promise.all([reader.read(), reader.cancel()]);
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
  });
  it("rejects traversal and symlinks escaping the selected model directory", async () => {
    const dir = await Deno.makeTempDir();
    try {
      await Deno.mkdir(`${dir}/model`);
      await Deno.writeTextFile(`${dir}/private`, "not served");
      await Deno.writeTextFile(`${dir}/model/karume.json`, "{}");
      await Deno.symlink(`${dir}/private`, `${dir}/model/link`);
      assertEquals(
        await containedPath(`${dir}/model`, "karume.json"),
        `${await Deno.realPath(dir)}/model/karume.json`,
      );
      for (
        const path of ["../private", "%2e%2e/private", "a/%2e%2e/private", "a\\private", "link"]
      ) await assertRejects(() => containedPath(`${dir}/model`, path));
      const handler = createHandler(
        { normal: `${dir}/model`, qat: `${dir}/model` },
        new Uint8Array(),
        "test",
      );
      assertEquals((await handler(new Request("http://evil.example/config.json"))).status, 403);
      assertEquals(
        (await handler(new Request("http://localhost/config.json", { method: "POST" }))).status,
        405,
      );
      assertEquals((await handler(new Request("http://localhost/private"))).status, 404);
      const config = await handler(new Request("http://localhost/config.json"));
      assertEquals(config.headers.get("Cross-Origin-Opener-Policy"), "same-origin");
      assertEquals(config.headers.get("Cross-Origin-Embedder-Policy"), "require-corp");
      await config.body?.cancel();
      const frame = await handler(new Request("http://localhost/frame.html"));
      const html = await frame.text();
      assert(!html.includes("{{onnxruntime}}"));
      assert(html.includes("ort.webgpu.min.mjs?v="));
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
  });
});

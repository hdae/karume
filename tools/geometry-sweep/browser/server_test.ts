import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { createHandler, type ServerConfig } from "./server.ts";

const CONFIG: ServerConfig = {
  revision: "test-revision",
  dirty: true,
  bundleSha256: "0".repeat(64),
};

const handler = createHandler(new Uint8Array([1, 2, 3]), CONFIG);

describe("geometry sweep browser server", () => {
  it("answers only localhost hosts and read-only methods", async () => {
    assertEquals((await handler(new Request("http://evil.example/config.json"))).status, 403);
    assertEquals((await handler(new Request("http://192.168.1.2:8789/"))).status, 403);
    const post = await handler(new Request("http://localhost/config.json", { method: "POST" }));
    assertEquals(post.status, 405);
  });

  it("serves config.json with the checkout identity and cross-origin isolation headers", async () => {
    const response = await handler(new Request("http://127.0.0.1:8789/config.json"));
    assertEquals(response.status, 200);
    assertEquals(response.headers.get("Cross-Origin-Opener-Policy"), "same-origin");
    assertEquals(response.headers.get("Cross-Origin-Embedder-Policy"), "require-corp");
    assertEquals(response.headers.get("Cross-Origin-Resource-Policy"), "same-origin");
    assertEquals(await response.json(), CONFIG);
  });

  it("serves the page and the bundled runner", async () => {
    const page = await handler(new Request("http://localhost/"));
    assertEquals(page.status, 200);
    assertEquals(page.headers.get("Content-Type"), "text/html; charset=utf-8");
    assertEquals((await page.text()).includes('src="runner.js"'), true);
    const runner = await handler(new Request("http://localhost/runner.js"));
    assertEquals(runner.headers.get("Content-Type"), "text/javascript");
    assertEquals(new Uint8Array(await runner.arrayBuffer()), new Uint8Array([1, 2, 3]));
  });

  it("serves nothing else — no file path reaches the disk", async () => {
    for (
      const path of ["/index.html", "/server.ts", "/models/anima/karume.json", "/../../deno.json"]
    ) {
      const response = await handler(new Request(`http://localhost${path}`));
      assertEquals(response.status, 404, path);
      await response.body?.cancel();
    }
  });
});

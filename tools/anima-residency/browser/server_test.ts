import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { createHandler, type ServerConfig } from "./server.ts";

const CONFIG: ServerConfig = {
  revision: "test-revision",
  dirty: true,
  bundleSha256: "0".repeat(64),
  source: "karume-anima",
};

const withModelRoot = async (
  body: (root: string, handler: (req: Request) => Promise<Response>) => Promise<void>,
): Promise<void> => {
  const dir = await Deno.makeTempDir();
  try {
    await Deno.mkdir(`${dir}/model/shared`, { recursive: true });
    await Deno.writeTextFile(`${dir}/private`, "not served");
    await Deno.writeFile(`${dir}/model/shared/part.krm`, new Uint8Array([10, 20, 30, 40, 50]));
    await Deno.symlink(`${dir}/private`, `${dir}/model/link`);
    const root = await Deno.realPath(`${dir}/model`);
    await body(root, createHandler(root, new Uint8Array([1, 2, 3]), CONFIG));
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
};

describe("anima residency browser server", () => {
  it("answers only localhost hosts and read-only methods", async () => {
    await withModelRoot(async (_root, handler) => {
      assertEquals((await handler(new Request("http://evil.example/config.json"))).status, 403);
      assertEquals((await handler(new Request("http://192.168.1.2:8788/"))).status, 403);
      const post = await handler(new Request("http://localhost/config.json", { method: "POST" }));
      assertEquals(post.status, 405);
    });
  });

  it("serves config.json with the checkout identity and cross-origin isolation headers", async () => {
    await withModelRoot(async (_root, handler) => {
      const response = await handler(new Request("http://127.0.0.1:8788/config.json"));
      assertEquals(response.status, 200);
      assertEquals(response.headers.get("Cross-Origin-Opener-Policy"), "same-origin");
      assertEquals(response.headers.get("Cross-Origin-Embedder-Policy"), "require-corp");
      assertEquals(response.headers.get("Cross-Origin-Resource-Policy"), "same-origin");
      assertEquals(await response.json(), CONFIG);
    });
  });

  it("serves the page, the bundled runner and model byte ranges inside the source root", async () => {
    await withModelRoot(async (_root, handler) => {
      const page = await handler(new Request("http://localhost/"));
      assertEquals(page.status, 200);
      assertEquals(page.headers.get("Content-Type"), "text/html; charset=utf-8");
      assertEquals((await page.text()).includes('src="/runner.js"'), true);
      const runner = await handler(new Request("http://localhost/runner.js"));
      assertEquals(runner.headers.get("Content-Type"), "text/javascript");
      assertEquals(new Uint8Array(await runner.arrayBuffer()), new Uint8Array([1, 2, 3]));
      const range = await handler(
        new Request("http://localhost/models/anima/shared/part.krm", {
          headers: { Range: "bytes=1-3" },
        }),
      );
      assertEquals(range.status, 206);
      assertEquals(new Uint8Array(await range.arrayBuffer()), new Uint8Array([20, 30, 40]));
    });
  });

  it("rejects traversal and symlinks escaping the source root", async () => {
    await withModelRoot(async (_root, handler) => {
      for (const path of ["shared%2f..%2f..%2fprivate", "a%5cprivate", "link"]) {
        const response = await handler(new Request(`http://localhost/models/anima/${path}`));
        assertEquals(response.status, 400, path);
        await response.body?.cancel();
      }
      // URL の正規化で根の外へ出る綴りは /models/anima/ の外の経路になり、何も配らない。
      const normalized = await handler(new Request("http://localhost/models/anima/../../private"));
      assertEquals(normalized.status, 404);
      await normalized.body?.cancel();
      const missing = await handler(new Request("http://localhost/models/anima/absent.krm"));
      assertEquals(missing.status, 404);
      await missing.body?.cancel();
    });
  });
});

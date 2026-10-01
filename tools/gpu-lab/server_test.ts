import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  createHandler,
  MissingDistributionError,
  resolveDistribution,
  type ServerConfig,
} from "./server.ts";

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

describe("gpu lab server", () => {
  it("answers only localhost hosts and read-only methods", async () => {
    await withModelRoot(async (_root, handler) => {
      assertEquals((await handler(new Request("http://evil.example/config.json"))).status, 403);
      assertEquals((await handler(new Request("http://192.168.1.2:8790/"))).status, 403);
      const post = await handler(new Request("http://localhost/config.json", { method: "POST" }));
      assertEquals(post.status, 405);
    });
  });

  it("serves config.json with the checkout identity and cross-origin isolation headers", async () => {
    await withModelRoot(async (_root, handler) => {
      const response = await handler(new Request("http://127.0.0.1:8790/config.json"));
      assertEquals(response.status, 200);
      assertEquals(response.headers.get("Cross-Origin-Opener-Policy"), "same-origin");
      assertEquals(response.headers.get("Cross-Origin-Embedder-Policy"), "require-corp");
      assertEquals(response.headers.get("Cross-Origin-Resource-Policy"), "same-origin");
      assertEquals(await response.json(), CONFIG);
    });
  });

  it("serves the page, the bundle and model byte ranges inside the source root", async () => {
    await withModelRoot(async (_root, handler) => {
      const page = await handler(new Request("http://localhost/"));
      assertEquals(page.status, 200);
      assertEquals(page.headers.get("Content-Type"), "text/html; charset=utf-8");
      assertEquals((await page.text()).includes('src="main.js"'), true);
      const bundle = await handler(new Request("http://localhost/main.js"));
      assertEquals(bundle.headers.get("Content-Type"), "text/javascript");
      assertEquals(new Uint8Array(await bundle.arrayBuffer()), new Uint8Array([1, 2, 3]));
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

  it("starts without a distribution: the page and config are served, the model path is 404", async () => {
    const config: ServerConfig = { ...CONFIG, source: null };
    const handler = createHandler(undefined, new Uint8Array([1, 2, 3]), config);
    const page = await handler(new Request("http://localhost/"));
    assertEquals(page.status, 200);
    await page.body?.cancel();
    const served = await handler(new Request("http://localhost/config.json"));
    assertEquals(served.status, 200);
    assertEquals(await served.json(), config);
    const manifest = await handler(new Request("http://localhost/models/anima/karume.json"));
    assertEquals(manifest.status, 404);
    await manifest.body?.cancel();
  });

  it("serves no page source or repository file — only the page, bundle, config and model", async () => {
    await withModelRoot(async (_root, handler) => {
      for (
        const path of [
          "/index.html",
          "/browser/index.html",
          "/browser/main.ts",
          "/server.ts",
          "/../../deno.json",
        ]
      ) {
        const response = await handler(new Request(`http://localhost${path}`));
        assertEquals(response.status, 404, path);
        await response.body?.cancel();
      }
    });
  });
});

describe("gpu lab distribution resolution", () => {
  /** console.warn に出た行を集める（本物の warn には流さない）。 */
  const capturingWarnings = async (
    body: (warnings: unknown[][]) => Promise<void>,
  ): Promise<void> => {
    const original = console.warn;
    const warnings: unknown[][] = [];
    console.warn = (...args: unknown[]) => void warnings.push(args);
    try {
      await body(warnings);
    } finally {
      console.warn = original;
    }
  };

  const withDirectory = async (body: (dir: string) => Promise<void>): Promise<void> => {
    const dir = await Deno.makeTempDir();
    try {
      await body(dir);
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
  };

  it("refuses to start when --source is given but has no karume.json", async () => {
    await withDirectory(async (dir) => {
      await capturingWarnings(async (warnings) => {
        const error = await assertRejects(
          () => resolveDistribution({ path: dir, explicit: true }),
          MissingDistributionError,
        );
        assertEquals(error.message.startsWith(`--source ${dir} has no karume.json`), true);
        assertEquals(warnings.length, 0);
      });
    });
  });

  it("refuses to start when --source names a directory that does not exist", async () => {
    await withDirectory(async (dir) => {
      await assertRejects(
        () => resolveDistribution({ path: `${dir}/absent`, explicit: true }),
        MissingDistributionError,
      );
    });
  });

  it("warns and starts without the distribution when the default location has none", async () => {
    await withDirectory(async (dir) => {
      await capturingWarnings(async (warnings) => {
        assertEquals(
          await resolveDistribution({ path: `${dir}/absent`, explicit: false }),
          undefined,
        );
        assertEquals(warnings.length, 1);
        assertEquals(String(warnings[0][0]).includes("has no karume.json"), true);
      });
    });
  });

  it("serves the directory holding karume.json, given explicitly or by default", async () => {
    await withDirectory(async (dir) => {
      await Deno.writeTextFile(`${dir}/karume.json`, "{}");
      const real = await Deno.realPath(dir);
      assertEquals(await resolveDistribution({ path: dir, explicit: true }), real);
      assertEquals(await resolveDistribution({ path: dir, explicit: false }), real);
    });
  });
});

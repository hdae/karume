import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  createHandler,
  MissingDistributionError,
  parseServerArgs,
  resolveDistribution,
  type ServerConfig,
} from "./server.ts";

const CONFIG: ServerConfig = {
  revision: "test-revision",
  dirty: true,
  bundleSha256: "0".repeat(64),
  source: "karume-anima",
  wanSource: "karume-wan2.1",
  wan22Source: "karume-wan2.2",
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
    await body(root, createHandler({ anima: root }, new Uint8Array([1, 2, 3]), CONFIG));
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
    const config: ServerConfig = { ...CONFIG, source: null, wanSource: null };
    const handler = createHandler({}, new Uint8Array([1, 2, 3]), config);
    const page = await handler(new Request("http://localhost/"));
    assertEquals(page.status, 200);
    await page.body?.cancel();
    const served = await handler(new Request("http://localhost/config.json"));
    assertEquals(served.status, 200);
    assertEquals(await served.json(), config);
    for (const name of ["anima", "wan"]) {
      const manifest = await handler(new Request(`http://localhost/models/${name}/karume.json`));
      assertEquals(manifest.status, 404, name);
      await manifest.body?.cancel();
    }
  });

  it("serves each distribution only under its own route", async () => {
    const dir = await Deno.makeTempDir();
    try {
      for (const name of ["anima", "wan"]) {
        await Deno.mkdir(`${dir}/${name}`);
        await Deno.writeTextFile(`${dir}/${name}/karume.json`, `{"name":"${name}"}`);
      }
      await Deno.writeTextFile(`${dir}/wan/only-wan.krm`, "wan part");
      const anima = await Deno.realPath(`${dir}/anima`);
      const wan = await Deno.realPath(`${dir}/wan`);
      const handler = createHandler({ anima, wan }, new Uint8Array([1, 2, 3]), CONFIG);
      for (const name of ["anima", "wan"]) {
        const manifest = await handler(new Request(`http://localhost/models/${name}/karume.json`));
        assertEquals(manifest.status, 200, name);
        assertEquals(await manifest.json(), { name });
      }
      const crossed = await handler(new Request("http://localhost/models/anima/only-wan.krm"));
      assertEquals(crossed.status, 404);
      await crossed.body?.cancel();
      const escaped = await handler(
        new Request("http://localhost/models/wan/..%2fanima%2fkarume.json"),
      );
      assertEquals(escaped.status, 400);
      await escaped.body?.cancel();
      // Wan だけ無い起動: Anima は配り、Wan の経路は 404
      const animaOnly = createHandler({ anima }, new Uint8Array([1, 2, 3]), CONFIG);
      const missing = await animaOnly(new Request("http://localhost/models/wan/karume.json"));
      assertEquals(missing.status, 404);
      await missing.body?.cancel();
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
  });

  it("serves the Wan2.2 distribution only under /models/wan22/, apart from the Wan2.1 one", async () => {
    const dir = await Deno.makeTempDir();
    try {
      for (const name of ["wan", "wan22"]) {
        await Deno.mkdir(`${dir}/${name}`);
        await Deno.writeTextFile(`${dir}/${name}/karume.json`, `{"name":"${name}"}`);
      }
      await Deno.writeTextFile(`${dir}/wan22/only-wan22.krm`, "wan22 part");
      const wan = await Deno.realPath(`${dir}/wan`);
      const wan22 = await Deno.realPath(`${dir}/wan22`);
      const handler = createHandler({ wan, wan22 }, new Uint8Array([1, 2, 3]), CONFIG);
      for (const name of ["wan", "wan22"]) {
        const manifest = await handler(new Request(`http://localhost/models/${name}/karume.json`));
        assertEquals(manifest.status, 200, name);
        assertEquals(await manifest.json(), { name });
      }
      for (const path of ["/models/wan/only-wan22.krm", "/models/umt5/only-wan22.krm"]) {
        const crossed = await handler(new Request(`http://localhost${path}`));
        assertEquals(crossed.status, 404, path);
        await crossed.body?.cancel();
      }
      // Wan2.2 だけ無い起動: Wan2.1 は配り、/models/wan22/ は 404
      const wanOnly = createHandler({ wan }, new Uint8Array([1, 2, 3]), {
        ...CONFIG,
        wan22Source: null,
      });
      const missing = await wanOnly(new Request("http://localhost/models/wan22/karume.json"));
      assertEquals(missing.status, 404);
      await missing.body?.cancel();
      const served = await wanOnly(new Request("http://localhost/models/wan/karume.json"));
      assertEquals(served.status, 200);
      await served.body?.cancel();
    } finally {
      await Deno.remove(dir, { recursive: true });
    }
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
  const ANIMA = { option: "--source", tab: "Anima" } as const;
  const WAN = { option: "--wan-source", tab: "Wan" } as const;

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
          () => resolveDistribution({ path: dir, explicit: true, ...ANIMA }),
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
        () => resolveDistribution({ path: `${dir}/absent`, explicit: true, ...ANIMA }),
        MissingDistributionError,
      );
    });
  });

  it("names the option and the tab of the distribution that is missing", async () => {
    await withDirectory(async (dir) => {
      const error = await assertRejects(
        () => resolveDistribution({ path: dir, explicit: true, ...WAN }),
        MissingDistributionError,
      );
      assertEquals(error.message.startsWith(`--wan-source ${dir} has no karume.json`), true);
      assertEquals(error.message.endsWith("start without the Wan tab)"), true);
      await capturingWarnings(async (warnings) => {
        await resolveDistribution({ path: dir, explicit: false, ...WAN });
        assertEquals(
          String(warnings[0][0]).includes("the Wan tab is disabled; pass --wan-source"),
          true,
        );
      });
    });
  });

  it("names --wan22-source and the Wan2.2 generation of the Wan tab when it is missing", async () => {
    await withDirectory(async (dir) => {
      const { locations } = parseServerArgs(["--wan22-source", dir]);
      const error = await assertRejects(
        () => resolveDistribution(locations.wan22),
        MissingDistributionError,
      );
      assertEquals(error.message.startsWith(`--wan22-source ${dir} has no karume.json`), true);
      assertEquals(error.message.endsWith("start without the Wan (Wan2.2) tab)"), true);
    });
  });

  it("warns and starts without the distribution when the default location has none", async () => {
    await withDirectory(async (dir) => {
      await capturingWarnings(async (warnings) => {
        assertEquals(
          await resolveDistribution({ path: `${dir}/absent`, explicit: false, ...ANIMA }),
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
      assertEquals(await resolveDistribution({ path: dir, explicit: true, ...ANIMA }), real);
      assertEquals(await resolveDistribution({ path: dir, explicit: false, ...WAN }), real);
    });
  });
});

describe("gpu lab command line", () => {
  it("reads every distribution from its default location when no option is given", () => {
    const { port, locations } = parseServerArgs([]);
    assertEquals(port, 8790);
    assertEquals(locations, {
      anima: { path: "models/karume-anima", explicit: false, option: "--source", tab: "Anima" },
      wan: { path: "models/karume-wan2.1", explicit: false, option: "--wan-source", tab: "Wan" },
      wan22: {
        path: "models/karume-wan2.2",
        explicit: false,
        option: "--wan22-source",
        tab: "Wan (Wan2.2)",
      },
      umt5: {
        path: "models/karume-umt5-xxl",
        explicit: false,
        option: "--umt5-source",
        tab: "Wan GPU text encoder",
      },
    });
  });

  it("takes --wan22-source as the explicit Wan2.2 location and leaves the others at their defaults", () => {
    const { locations } = parseServerArgs(["--wan22-source", "/data/wan22", "--port", "8791"]);
    assertEquals(locations.wan22.path, "/data/wan22");
    assertEquals(locations.wan22.explicit, true);
    assertEquals(locations.wan, {
      path: "models/karume-wan2.1",
      explicit: false,
      option: "--wan-source",
      tab: "Wan",
    });
    assertEquals(parseServerArgs(["--wan-source", "/data/wan21"]).locations.wan22.explicit, false);
  });

  it("refuses an unknown, repeated, or valueless option and a port out of range", () => {
    for (
      const argv of [
        ["--wan23-source", "x"],
        ["--wan22-source", "a", "--wan22-source", "b"],
        ["--wan22-source"],
        ["--wan22-source", "--port"],
      ]
    ) {
      assertThrows(() => parseServerArgs(argv), Error, "Invalid option", argv.join(" "));
    }
    for (const port of ["80", "70000", "8790.5"]) {
      assertThrows(() => parseServerArgs(["--port", port]), Error, "Invalid port", port);
    }
  });
});

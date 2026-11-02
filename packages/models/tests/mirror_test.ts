// Checks of the shared mirror helpers (`helpers/mirror.ts`) on tiny synthetic distributions in a
// temporary `models/` (no GPU, no real weights).
//
// What is pinned: a mirror is recognised by its manifest format, a component is opened through the
// manifest (the graph named by the weights key, the documents checked against the declared
// sha256), a declared asset is read through the manifest, and a cross-repo reference
// `hdae/<name>` is read from the sibling mirror `<name>` — failing by name when that mirror is
// absent rather than at the first file read.

import { makeTempDir, mkdir, remove, writeFile, writeTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";

import { linearComponent } from "./helpers/container-fixture.ts";
import {
  fileRef,
  serveContainer,
  type ServedContainer,
} from "./helpers/container-loading-fixture.ts";
import {
  MODELS_ROOT,
  mirrorDirectory,
  mirrorPresent,
  mirrorRoot,
  openMirrorComponent,
  readMirrorAsset,
} from "./helpers/mirror.ts";

const REVISION = "0123456789abcdef0123456789abcdef01234567";
const TOKENIZER = new TextEncoder().encode('{"vocab":["a","b"]}');

/** The manifest text of one model `test` with the component `dit` and the asset `tokenizer`. */
const manifestText = (dit: unknown, tokenizer: unknown, format = "karume/5"): string =>
  JSON.stringify({
    format,
    generator: "karume-test/0",
    defaultModel: "test",
    models: {
      test: {
        pipeline: "test/1",
        weights: { dit: { f32: dit } },
        assets: { tokenizer },
        quants: { f32: { weights: { dit: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
  });

/**
 * The manifest entry of a served container — its parts referenced from `repo` when given, its graph
 * descriptor's expected sha256 replaced when `graphSha256` is given.
 */
const entryOf = (
  served: ServedContainer,
  options: { readonly repo?: string; readonly graphSha256?: string } = {},
): Record<string, unknown> => {
  const { descriptor } = served.written;
  const { repo, graphSha256 } = options;
  return {
    container: {
      descriptor:
        graphSha256 === undefined
          ? descriptor
          : { ...descriptor, graph: { ...descriptor.graph, sha256: graphSha256 } },
      parts:
        repo === undefined
          ? served.parts
          : served.parts.map((part) => ({ ...part, repo, revision: REVISION })),
    },
  };
};

type Models = {
  readonly models: URL;
  /** The container both mirrors declare (written under `karume-base`). */
  readonly served: ServedContainer;
  /** `karume-base`: holds its container and tokenizer itself. */
  readonly base: URL;
  /** `karume-derived`: references both from `hdae/karume-base`. */
  readonly derived: URL;
};

/** Builds `models/karume-base/` and `models/karume-derived/` in a temporary directory. */
const withModels = async (body: (models: Models) => Promise<void>): Promise<void> => {
  const dir = new URL(`file://${await makeTempDir({ prefix: "karume-mirror-" })}/`);
  const models = new URL("models/", dir);
  const base = new URL("karume-base/", models);
  const derived = new URL("karume-derived/", models);
  try {
    const served = await serveContainer("test/dit/model.f32", linearComponent("dit"));
    await mkdir(new URL("test/dit/", base), { recursive: true });
    for (const [path, bytes] of served.files) await writeFile(new URL(path, base), bytes);
    await writeFile(new URL("tokenizer.json", base), TOKENIZER);
    const tokenizer = await fileRef("tokenizer.json", TOKENIZER);
    await writeTextFile(new URL("karume.json", base), manifestText(entryOf(served), tokenizer));
    await mkdir(derived, { recursive: true });
    await writeTextFile(
      new URL("karume.json", derived),
      manifestText(entryOf(served, { repo: "hdae/karume-base" }), {
        ...tokenizer,
        repo: "hdae/karume-base",
        revision: REVISION,
      }),
    );
    await body({ models, served, base, derived });
  } finally {
    await remove(dir, { recursive: true });
  }
};

describe("mirror: the root and the presence check", () => {
  it("a mirror's root is models/<repo dir>/", () => {
    assertEquals(
      mirrorRoot("karume-birefnet-hr").href,
      new URL("karume-birefnet-hr/", MODELS_ROOT).href,
    );
    assert(MODELS_ROOT.pathname.endsWith("/models/"));
  });

  it("is present only when karume.json declares karume/5", async () => {
    await withModels(async ({ models, base }) => {
      assert(mirrorPresent(base));
      assertEquals(mirrorPresent(new URL("karume-missing/", models)), false);
      await writeTextFile(new URL("karume.json", base), JSON.stringify({ format: "karume/4" }));
      assertEquals(mirrorPresent(base), false);
    });
  });
});

describe("mirror: opening a component through the manifest", () => {
  it("opens the container whose graph is the weights key", async () => {
    await withModels(async ({ base }) => {
      const component = await openMirrorComponent(base, "dit");
      assertEquals(Object.keys(component.opened.graphs), ["dit"]);
      assertEquals(component.graph.outputs, ["y"]);
    });
  });

  it("fails naming the key when the manifest has no such component", async () => {
    await withModels(async ({ base }) => {
      await assertRejects(() => openMirrorComponent(base, "vae"), Error, "vae");
    });
  });

  it("fails when part 0's documents differ from the declared sha256", async () => {
    await withModels(async ({ served, base }) => {
      const tokenizer = await fileRef("tokenizer.json", TOKENIZER);
      await writeTextFile(
        new URL("karume.json", base),
        manifestText(entryOf(served, { graphSha256: "e".repeat(64) }), tokenizer),
      );
      await assertRejects(() => openMirrorComponent(base, "dit"), Error, "sha256");
    });
  });
});

describe("mirror: reading a declared asset through the manifest", () => {
  it("returns the asset's bytes", async () => {
    await withModels(async ({ base }) => {
      assertEquals(await readMirrorAsset(base, "tokenizer"), TOKENIZER);
    });
  });

  it("fails listing the declared assets when the name is not declared", async () => {
    await withModels(async ({ base }) => {
      await assertRejects(
        () => readMirrorAsset(base, "text_embeds"),
        Error,
        "declares no asset 'text_embeds' (declared: tokenizer)",
      );
    });
  });
});

describe("mirror: cross-repo references resolve to sibling mirrors", () => {
  it("opens a component whose parts live in hdae/<name> from models/<name>", async () => {
    await withModels(async ({ derived }) => {
      const component = await openMirrorComponent(derived, "dit");
      assertEquals(Object.keys(component.opened.graphs), ["dit"]);
    });
  });

  it("reads a cross-repo asset from the sibling mirror", async () => {
    await withModels(async ({ derived }) => {
      assertEquals(await readMirrorAsset(derived, "tokenizer"), TOKENIZER);
    });
  });

  it("fails naming the repo and the expected directory when the sibling mirror is absent", async () => {
    await withModels(async ({ base, derived }) => {
      await remove(base, { recursive: true });
      const error = assertThrows(() => mirrorDirectory(derived), Error, "'hdae/karume-base'");
      assert(error.message.includes("/models/karume-base/"), error.message);
    });
  });

  it("fails on a cross-repo reference outside hdae/", async () => {
    await withModels(async ({ served, base, derived }) => {
      const tokenizer = await fileRef("tokenizer.json", TOKENIZER);
      await writeTextFile(
        new URL("karume.json", derived),
        manifestText(entryOf(served, { repo: "someone/karume-base" }), tokenizer),
      );
      assert(mirrorPresent(base));
      assertThrows(() => mirrorDirectory(derived), Error, "not a hdae/<name> mirror");
    });
  });
});

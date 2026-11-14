// The check that the barrel (`mod.ts`) **exports no values** for these names (the same shape as
// models' `models_barrel_surface_test.ts`). Needs neither the network nor real assets.
//
// `DistributionSource` is an **opaque** handle, and telling it apart is confined to
// `isDistributionSource` (the doc of `mod.ts` is the source of truth for why — telling it apart by
// structure would let a misspelling pass as a source). That opacity holds only through
// `export type`, yet changing it to `export` still passes the type check, and the public surface
// snapshot check shows no diff either (its collector, `runtime/tests/helpers/public-surface.ts`,
// counts a type-only re-export by its target's kind, as `deno doc --json` did — measured on
// `deno doc` 2026-09-21). That nothing is exported as a value can be observed
// only at run time.

import { fileURLToPath } from "node:url";
import { makeTempDir, remove, writeTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertMatch,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import { build } from "vite-plus";
import * as hub from "../mod.ts";
import type { DistributionSource, LoadedManifest } from "../mod.ts";
import { fetchManifest } from "./helpers/fixture.ts";

/** Names exported as types only (the surface of the opaque handle). */
const TYPE_ONLY = ["DistributionSource"];

test("barrel: the opaque handle types are not exported as values", () => {
  const surface = Object.keys(hub);
  // Positive control — first make sure this is not green because it looks at an empty namespace.
  assert(
    surface.includes("isDistributionSource"),
    `the barrel import is broken (${surface.length} entries)`,
  );
  assertEquals(
    surface.filter((name) => TYPE_ONLY.includes(name)),
    [],
    "a name exported as a type only is exported as a value (telling the opaque handle apart would depend on structure)",
  );
});

test("barrel: the opaque handle and LoadedManifest do not accept hand-built values", async () => {
  // Not accepted in the types either (nominal types) — with structural types a misspelled
  // `HubRepoRef` or a hand-built value would pass the type check. If the two lines below stop being
  // type errors, the type check of this test file fails.
  // @ts-expect-error a structural object does not satisfy the source handle type
  const source: DistributionSource = {};
  // @ts-expect-error a value that only has the public fields does not satisfy the LoadedManifest type
  const loaded: LoadedManifest = { manifest: fetchManifest };
  // Not accepted at run time either — a source is bound only to what loadManifest returns.
  assertEquals(hub.isDistributionSource(source), false);
  const error = await assertRejects(() => hub.fetchAssets(loaded, {}), Error);
  assertStringIncludes(error.message, "no source is bound to the LoadedManifest");
});

/**
 * Bundles an entry that contains only `source` in memory with Vite's (Rolldown's) build API and
 * returns the JS (not minified — the test observes names). Nothing is external — the whole barrel
 * is
 * bundled so that tree-shaking can be observed.
 */
const bundle = async (source: string): Promise<string> => {
  const dir = await makeTempDir();
  try {
    const input = `${dir}/entry.ts`;
    await writeTextFile(input, source);
    const result = await build({
      configFile: false,
      logLevel: "silent",
      root: dir,
      publicDir: false,
      build: {
        write: false,
        minify: false,
        outDir: `${dir}/dist`,
        lib: { entry: input, formats: ["es"] },
      },
    });
    // A build with write: false and no watch returns a RolldownOutput (or an array of them).
    if (!Array.isArray(result) && !("output" in result)) {
      throw new Error("build returned a watcher instead of the bundle output");
    }
    return (Array.isArray(result) ? result : [result])
      .flatMap(({ output }) => output)
      .flatMap((file) => (file.type === "chunk" ? [file.code] : []))
      .join("\n");
  } finally {
    await remove(dir, { recursive: true });
  }
};

test("barrel: importing alone leaves no hub class in the bundle (nothing runs at import)", async () => {
  // The cross-cutting invariant "zero side effects in every module" is the condition for
  // tree-shaking through the barrel. A class with code that runs when the class is evaluated (a
  // `static` block) is kept by the bundler as a side effect even when it is unused.
  const barrel = fileURLToPath(new URL("../mod.ts", import.meta.url));
  // Positive control — the using side's bundle does contain the names (first make sure they can be
  // observed by name).
  const used = await bundle(
    `import { isDistributionSource, loadManifest } from "${barrel}";\n` +
      `console.log(isDistributionSource, loadManifest);\n`,
  );
  for (const name of ["DistributionSource", "LoadedManifest"]) {
    // Rolldown may emit either the `class Name` or the `Name = class` form.
    assertMatch(
      used,
      new RegExp(`\\bclass ${name}\\b|\\b${name} = class\\b`),
      `positive control: ${name} missing from the using side's bundle`,
    );
  }
  const bare = await bundle(`import "${barrel}";\n`);
  for (const name of ["DistributionSource", "LoadedManifest"]) {
    assert(!bare.includes(name), `unused ${name} remains in the bundle (code runs at import)`);
  }
});

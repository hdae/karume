// The barrel's tree-shaking gate (ADR 0037 §4 and its addendum of 2026-11-04). Needs neither a GPU,
// real assets nor the network.
//
// Every module MUST be free of side effects, and a bundler MUST be able to see that without a
// `"sideEffects": false` package flag: JSR publishes from `deno.json`, and the `package.json` of its
// npm layer carries no such field (checked on the 0.16.0 tarball). So a module-scope expression the
// bundler cannot prove pure — an unannotated call such as `Object.freeze([...])`, an object or array
// spread, a property read — stays in every barrel bundle, whatever the consumer imports. The fix at
// each site is a `/* @__PURE__ */` call; this gate catches a new site.
//
// Observed with Vite's build API (Rolldown — the bundler of Vite+), minified, with `@karume/*` and
// `@hdae/*` external and treated as free of side effects: the runtime and hub barrels are not under
// test here, only what this package adds.

import { fileURLToPath } from "node:url";
import { makeTempDir, remove, writeTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { build } from "vite-plus";

const BARREL = fileURLToPath(new URL("../mod.ts", import.meta.url));
const WAV = fileURLToPath(new URL("../src/audio/wav.ts", import.meta.url));

/** Bundles an entry module of `source` in memory and returns the minified JS. */
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
        minify: true,
        outDir: `${dir}/dist`,
        lib: { entry: input, formats: ["es"] },
        rolldownOptions: {
          external: [/^@karume\//, /^@hdae\//],
          treeshake: { moduleSideEffects: "no-external" },
        },
      },
    });
    // A build without watch and with write: false returns the RolldownOutput (or an array of them).
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

test("barrel tree-shaking: a bare barrel import leaves no code in the bundle", async () => {
  // Positive control — a bundle that reads the source tables does carry them, so the empty
  // output below is not an artefact of the bundling setup.
  const used = await bundle(`export { KARUME_SOURCES } from "${BARREL}";\n`);
  assert(used.includes("hdae/karume-"), "control: the source tables are missing from the bundle");
  const bare = await bundle(`import "${BARREL}";\n`);
  // Rolldown still writes its `//#region` / `//#endregion` markers into an empty chunk.
  const code = bare.replaceAll(/^\/\/#(?:end)?region.*$/gm, "").trim();
  assertEquals(code, "", "module-scope code of the barrel survives tree-shaking");
});

test("barrel tree-shaking: a shared helper costs through the barrel what it costs from its module", async () => {
  const direct = await bundle(`export { encodeWav } from "${WAV}";\n`);
  assert(direct.length > 0, "control: the direct bundle is empty");
  const viaBarrel = await bundle(`export { encodeWav } from "${BARREL}";\n`);
  assertEquals(viaBarrel, direct);
});

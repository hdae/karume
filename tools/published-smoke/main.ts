// Connectivity check against the published JSR packages (docs/release-runbook.md §5).
//
// The same file runs on both Node (primary) and Deno (secondary — checks that the published packages
// can be imported from Deno).
// The published packages are imported by bare specifier (`@karume/hub` / `@karume/models`), and this
// directory's own configuration pins their resolution to the published packages on the registry:
//   - Node: the `jsr:^<version>` dependencies of package.json (pnpm installs them from npm.jsr.io —
//     not workspace:*)
//   - Deno: the import map of deno.json (`jsr:@karume/*@<version>`). Under the workspace, Deno
//     **resolves `jsr:@karume/*` to the local workspace members**, so it runs with its own deno.json
//     (not a workspace member; the one-line "ignoring the parent workspace" warning at startup is
//     expected). Right after a release the dependencies are caught by the minimum dependency age
//     check, so the same deno.json excludes `jsr:@karume/*` and `jsr:@hdae/fetch-cache` from it.
// The resolved version is compared with the expected version (packages/models/package.json), and a
// mismatch fails.
//
// Check = resolve the manifest of every entry of the models barrel's source table
// (`KARUME_SOURCES` — the union of every family. ADR 0092), and take one family (sbv2) through
// `fromPretrained` (needs a GPU — skipped with `--manifests-only`).
// Run: node tools/published-smoke/main.ts [--manifests-only]
//      deno run -A main.ts [--manifests-only] (in tools/published-smoke)

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isPinRef, parseSmokeArgs, type PinRef } from "./cli.ts";

type HubModule = {
  loadManifest: (ref: PinRef) => Promise<{ manifest: { format?: unknown } }>;
};
type Sbv2Module = {
  Sbv2Pipeline: {
    fromPretrained: (ref: PinRef) => Promise<{ dispose: () => Promise<void> }>;
  };
  SBV2_SOURCES: Record<string, unknown>;
};
type DawnModule = { create: (flags: string[]) => GPU };

// The dynamic import specifiers are passed as variables of type `string` (no types attach — only the
// surface in use is declared structurally). What is typed is the published package, not the
// workspace version; with a literal, Deno's type check would resolve to the workspace members and tsc
// to node_modules that are not installed.
const HUB: string = "@karume/hub";
const MODELS: string = "@karume/models";
const MODELS_SBV2: string = "@karume/models/sbv2";
const DAWN: string = "webgpu";

/**
 * Narrows a value received from a dynamic import **to a keyed table that can be looked up**. With
 * `typeof === "object"` alone the type is only `object`, the result of `Object.entries` falls to
 * `any`, and the checks after it are not type-checked.
 */
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const readVersion = async (path: string): Promise<string> => {
  const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
  const version = isRecord(parsed) ? parsed.version : undefined;
  if (typeof version !== "string") {
    throw new Error(`cannot read version: ${path}`);
  }
  return version;
};

/**
 * The published version a bare specifier actually resolves to.
 *
 * - Deno returns the jsr specifier of the import map as spelled, before version resolution
 *   (`jsr:@karume/hub@0.14.0`) — the imports of deno.json name the exact version (with `^` it does
 *   not match the expected version and fails).
 * - Node returns a file URL inside the installed package — the nearest package.json holds the
 *   version.
 */
const resolvedVersion = async (specifier: string): Promise<string> => {
  const resolved = import.meta.resolve(specifier);
  const jsr = /^jsr:\/?@karume\/[^@/]+@([^/]+)/.exec(resolved)?.[1];
  if (jsr !== undefined) return jsr;
  if (resolved.startsWith("file:")) {
    let dir = dirname(fileURLToPath(resolved));
    while (!existsSync(join(dir, "package.json"))) {
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
    return await readVersion(join(dir, "package.json"));
  }
  throw new Error(`cannot read the version from the resolution of ${specifier}: ${resolved}`);
};

/**
 * Node has no `navigator.gpu`, and the published `fromPretrained` takes its adapter from
 * `navigator.gpu`. On Node only, Dawn (npm `webgpu`) is installed as `navigator.gpu` (the shape
 * browsers and Deno have — installed the same way as the test-support setup).
 */
const ensureNavigatorGpu = async (): Promise<void> => {
  const current: GPU | undefined = navigator.gpu;
  if (current !== undefined) return;
  const { create } = (await import(DAWN)) as DawnModule;
  Object.defineProperty(navigator, "gpu", {
    value: create([]),
    configurable: true,
    enumerable: true,
  });
};

const { manifestsOnly } = parseSmokeArgs(process.argv.slice(2));
const version = await readVersion(
  fileURLToPath(new URL("../../packages/models/package.json", import.meta.url)),
);
console.log(`published smoke: @karume/* @ ${version}`);

for (const specifier of [HUB, MODELS]) {
  const resolved = await resolvedVersion(specifier);
  if (resolved !== version) {
    throw new Error(
      `${specifier} resolved to ${resolved} (expected ${version} — packages/models/package.json). ` +
        "On Node, raise the dependencies of tools/published-smoke/package.json and run pnpm install; " +
        "on Deno, align the imports of tools/published-smoke/deno.json",
    );
  }
}

const models = (await import(MODELS)) as Record<string, unknown>;
const hub = (await import(HUB)) as HubModule;

const sources: unknown = models.KARUME_SOURCES;
if (!isRecord(sources)) {
  throw new Error("KARUME_SOURCES not found in the models barrel");
}
// filter cannot narrow the second element of a tuple, so flatMap narrows the type too (no `as`).
const pins = Object.entries(sources).flatMap(([name, value]) =>
  isPinRef(value) ? [[name, value] as const] : [],
);
if (pins.length === 0) {
  throw new Error("KARUME_SOURCES has no entry");
}

for (const [name, ref] of pins) {
  const started = performance.now();
  const loaded = await hub.loadManifest(ref);
  const format = loaded.manifest.format;
  if (typeof format !== "string") {
    throw new Error(`${name}: manifest.format is not a string`);
  }
  console.log(
    `  manifest OK ${name} ${ref.repo}@${ref.revision.slice(0, 8)} format=${format} (${(
      performance.now() - started
    ).toFixed(0)}ms)`,
  );
}

if (manifestsOnly) {
  console.log(`published smoke OK: ${pins.length} manifests (fromPretrained skipped)`);
} else {
  await ensureNavigatorGpu();
  const { Sbv2Pipeline, SBV2_SOURCES } = (await import(MODELS_SBV2)) as Sbv2Module;
  const pin: unknown = SBV2_SOURCES["sbv2-jvnv"];
  if (!isPinRef(pin)) throw new Error('SBV2_SOURCES["sbv2-jvnv"] is not a pin entry');
  const started = performance.now();
  const pipeline = await Sbv2Pipeline.fromPretrained(pin);
  console.log(
    `  fromPretrained OK sbv2-jvnv (${((performance.now() - started) / 1000).toFixed(1)}s)`,
  );
  await pipeline.dispose();
  console.log(`published smoke OK: ${pins.length} manifests + sbv2 fromPretrained`);
}

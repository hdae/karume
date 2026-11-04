/**
 * Measures how many container bytes stay resident (a script `container_loading_test.ts` launches in
 * a separate process).
 *
 * MUST: it is a separate process because it needs `--v8-flags=--expose-gc` — `deno test` has no way
 * to set that flag, and without it "bytes that became unreachable but are not collected yet" cannot
 * be told apart from "bytes still held" (gc always runs once before measuring).
 *
 * What is measured is `process.memoryUsage().external` (the total of ArrayBuffer backing stores).
 * Measuring **after explicitly dropping the cache's contents** leaves the remaining external memory
 * = only the bytes models holds. `open(key)` is kept while measuring (to reproduce the original
 * situation: the bytes stay resident as long as a component's supply entry is alive). On the
 * container path the supply entry holds only **a range reader**, and the weight blocks are read
 * only at the moment the Session is built — so one whole weight staying resident is a regression.
 *
 * On failure it writes the reason to stderr and exits non-zero (the caller shows that output as
 * is).
 */

import { loadContainerComponents } from "../../src/hub/components.ts";
import { parseIrDeclarationValue } from "./container-fixture.ts";
import { type RepoSpec, serveContainer, serveRepos } from "./container-loading-fixture.ts";
import { MemoryCacheStorage } from "./memory-cache.ts";
import { loadManifest, resolveSelection } from "@karume/hub";
import type { ModelInput } from "../../../runtime/tests/helpers/container-write.ts";

const HUB_URL = "https://hub.test";
const REPO = "karume-test/retention";
const SHA = "0123456789abcdef0123456789abcdef01234567";

/** The size of one weight (2048² f32 = 16MiB). */
const WIDTH = 2048;
const WEIGHT_BYTES = WIDTH * WIDTH * 4;
/**
 * The pass line: the external memory allowed to remain after gc. Holding the weight leaves the
 * whole 16MiB.
 */
const ALLOWED_RESIDENT_BYTES = 4 * 1024 * 1024;

/** One `linear` stage. `w` (16MiB) goes in a weight block. */
const component = (): ModelInput => ({
  graphs: {
    dit: parseIrDeclarationValue({
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [2, WIDTH] }],
      outputs: ["y"],
      initializers: { w: {}, b: {} },
      values: {
        w: { dtype: "f32", shape: [WIDTH, WIDTH] },
        b: { dtype: "f32", shape: [WIDTH] },
        y: { dtype: "f32", shape: [2, WIDTH] },
      },
      states: {},
      nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
    }),
  },
  consts: [],
  weights: [
    {
      graph: "dit",
      initializer: "w",
      bytes: new Uint8Array(new ArrayBuffer(WEIGHT_BYTES)),
      encoding: { codec: "f32" },
    },
    {
      graph: "dit",
      initializer: "b",
      bytes: new Uint8Array(new ArrayBuffer(WIDTH * 4)),
      encoding: { codec: "f32" },
    },
  ],
  assets: [],
  provenance: { license: "test" },
});

const mib = (bytes: number): string => `${(bytes / 1024 / 1024).toFixed(1)}MiB`;

/** Reads external memory after making sure unreachable ArrayBuffers are collected. */
const externalAfterGc = (): number => {
  const collect = (globalThis as { gc?: () => void }).gc;
  if (collect === undefined) {
    console.error("weight-block-retention: not launched with --v8-flags=--expose-gc");
    process.exit(2);
  }
  collect();
  collect();
  return process.memoryUsage().external;
};

/**
 * Builds the table of served files. A separate function so that **the fixture's bytes are not
 * placed in `main`'s frame** — a running function's locals are not collected while its frame is
 * alive (even when nothing inside refers to them), and the 16MiB fixture would land in the
 * measurement as is.
 */
const buildServed = async (): Promise<RepoSpec> => {
  const container = await serveContainer("dit/model.f32", component(), {
    partBytes: 32 * 1024 * 1024,
    blockBytes: 32 * 1024 * 1024,
  });
  return {
    repo: REPO,
    models: {
      test: {
        pipeline: "test/1",
        weights: { dit: { f32: container.entry } },
        assets: {},
        quants: { f32: { weights: { dit: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: {},
      },
    },
    files: container.files,
  };
};

const main = async (): Promise<void> => {
  // Node keeps the imported module sources as external strings (about 10 MiB for hub + models), so
  // the retained bytes are measured as the growth over this baseline, not as an absolute amount.
  const baseline = externalAfterGc();
  // MUST: do not bind `RepoSpec` in `main`'s frame (the fixture's bytes would land in the
  // measurement).
  const mock = serveRepos([await buildServed()]);
  const { fetch } = mock;

  const caches = new MemoryCacheStorage();
  const hubOptions = { fetch, caches };
  const loaded = await loadManifest({ repo: REPO, revision: SHA, hubUrl: HUB_URL }, hubOptions);
  const { open } = await loadContainerComponents(
    "test.fromPretrained",
    loaded,
    resolveSelection(loaded.manifest),
    ["dit"],
    // The family admission slot (this script measures residency, so it places no check).
    () => undefined,
    hubOptions,
  );

  // Drop the bytes outside models (the fixture, the cache) while keeping the supply entry — so that
  // the equation "remaining external memory = what models holds" holds (the fake HF table is held
  // by the `fetch` closure, so it does not vanish on its own while the fetch surface is alive).
  mock.clear();
  for (const cache of caches.namespaces.values()) cache.entries.clear();

  const resident = externalAfterGc() - baseline;
  // Keep `open` alive until after gc (reachability is cut only here).
  if (open("dit").graph.outputs[0] !== "y") {
    console.error("weight-block-retention: cannot read the graph declaration (fixture error)");
    process.exit(2);
  }

  if (resident > ALLOWED_RESIDENT_BYTES) {
    console.error(
      `weight-block-retention: container bytes stay resident` +
        ` (external growth=${mib(resident)} > allowed ${mib(ALLOWED_RESIDENT_BYTES)}` +
        ` / one weight ${mib(WEIGHT_BYTES)})`,
    );
    process.exit(1);
  }
  console.log(
    `weight-block-retention: external growth=${mib(resident)} (baseline ${mib(baseline)})`,
  );
};

await main();

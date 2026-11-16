/**
 * The Vite plugin of the verification cache's `reuse` mode (see `./verify-cache.ts`): it serves a stub
 * module in place of a test file whose recorded result is reused, so the file's own code never loads
 * and its one case reports the reuse as skipped.
 *
 * The reporter (`./verify-cache-reporter.ts`) decides before any file runs and registers each reused
 * file with its stub; this plugin answers the test runner's request for the file's module. The Vitest
 * config (which creates the plugin) and the reporter are loaded as separate module instances, so the
 * registry they share lives on `globalThis` under a registered symbol, in the main process only.
 */

import type { Plugin } from "vite-plus";
import { cacheModeOf } from "./verify-cache.ts";

const REGISTRY = Symbol.for("karume.verifyCache.reusedStubs");

type RegistryHost = typeof globalThis & { [REGISTRY]?: Map<string, string> };

/** Reused test files of the current run: absolute module path → the stub module's text. */
export const reusedStubs = (): Map<string, string> => {
  const host = globalThis as RegistryHost;
  host[REGISTRY] ??= new Map();
  return host[REGISTRY];
};

const stubOf = (id: string): string | undefined => {
  const queryAt = id.indexOf("?");
  return reusedStubs().get(queryAt === -1 ? id : id.slice(0, queryAt));
};

/**
 * The plugin. Creating it checks `KARUME_VERIFY_CACHE`, so a value that is not a mode stops the run
 * before anything starts.
 */
export const verifyCacheReuse = (): Plugin => {
  cacheModeOf(process.env);
  return {
    name: "karume:verify-cache-reuse",
    // Before the TypeScript transform and any other loader: the stub replaces the file's source.
    enforce: "pre",
    configureVitest({ defineCacheKeyGenerator }) {
      // MUST: Vitest keeps transformed modules on disk keyed by the file's own source and serves them
      // without asking the plugins, so a reused file bypasses that cache (`false`): its stub is
      // neither served from it nor written to it. Every other file keeps its usual key (`undefined`).
      defineCacheKeyGenerator(({ id }) => (stubOf(id) === undefined ? undefined : false));
    },
    load(id) {
      return stubOf(id);
    },
  };
};

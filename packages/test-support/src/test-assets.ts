/**
 * The pinned private test-asset dataset (ADR 0139 decisions 2, 3 and 5) and its local root.
 *
 * Test assets that no distribution carries — recipe outputs (`goldens/<series>/<path>`) and the
 * hand-frozen inputs (`corpus/<kind>/`) — live in a private Hugging Face dataset and are fetched
 * into `outputs/test-assets/` at the revision pinned here (`node tools/test-assets/fetch.ts`). The
 * dataset root and the local root have the same layout. The fetch records the revision it
 * downloaded in `outputs/test-assets/.revision` only after the download completed; the assets
 * gate (`packages/e2e/tests/e2e_assets_gate_test.ts`) is red while that file is absent or differs
 * from the pin.
 *
 * MUST: {@link TEST_ASSETS} is the single source of truth for the pin. A change that needs new or
 * changed assets uploads them first and moves the revision here in the same commit as the tests that
 * read them (decision 3) — the fetch command, the gate and the README all read it from here.
 *
 * No side effects at import: every path is computed, nothing is read until a function is called.
 */

import { isNotFound, readTextFileSync } from "./fs.ts";

/** The dataset repository and the commit SHA the tests read. */
export const TEST_ASSETS = {
  repo: "hdae/karume-test-assets",
  revision: "b4b86d2d6868a273f6c752ae0342c0efef7b9bb9",
} as const;

/** The shape of a pin (the tests of the fetch command and of the gate pass their own). */
export type TestAssetsPin = { readonly repo: string; readonly revision: string };

/** The local root `outputs/test-assets/` (resolved from this module, independent of the cwd). */
export const TEST_ASSETS_ROOT: URL = new URL("../../../outputs/test-assets/", import.meta.url);

/** The file the fetch writes after a completed download (the revision it fetched, one line). */
export const REVISION_FILE = ".revision";

/** The kinds of hand-frozen inputs under `corpus/` (decision 2). */
export type CorpusKind = "anima" | "wan-i2v";

/**
 * `goldens/<name>/` — the recipe outputs of one series, at the relative paths they had under
 * `outputs/series/<name>/`.
 */
export const goldenDir = (name: string): URL => new URL(`goldens/${name}/`, TEST_ASSETS_ROOT);

/** `corpus/<kind>/` — the test inputs made by this project and frozen by hand. */
export const corpusDir = (kind: CorpusKind): URL => new URL(`corpus/${kind}/`, TEST_ASSETS_ROOT);

/**
 * The revision recorded by the last completed fetch into `root` (trimmed), or `undefined` when no
 * fetch completed there.
 *
 * MUST: only NotFound means "not fetched" — any other failure (permissions, `.revision` being a
 * directory) propagates, so a broken root is reported as broken rather than as "absent".
 */
export const readFetchedRevision = (root: URL = TEST_ASSETS_ROOT): string | undefined => {
  try {
    return readTextFileSync(new URL(REVISION_FILE, root)).trim();
  } catch (cause) {
    if (isNotFound(cause)) return undefined;
    throw cause;
  }
};

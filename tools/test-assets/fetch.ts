/**
 * fetch — download the pinned private test-asset dataset into `outputs/test-assets/` (ADR 0139
 * decision 3).
 *
 *     node tools/test-assets/fetch.ts
 *     node tools/test-assets/fetch.ts --include 'goldens/birefnet-hr-1024-f16/**'
 *
 * It runs `hf download <repo> --repo-type dataset --revision <sha> --local-dir <root> --exclude
 * .revision` for the pin in `@karume/test-support/test-assets`, passing each `--include <glob>`
 * through, and records the revision in `<root>/.revision` **only after a complete download** (no
 * `--include`) exited 0. A stale `.revision` is removed before the download starts, so a failed or
 * interrupted fetch leaves no record that the assets gate could mistake for a completed one. A partial
 * fetch never writes the record: it keeps one only when a complete fetch of the same revision is
 * already there, so the gate stays red on a root that holds part of the assets.
 *
 * The `hf` command is `hf` on PATH (huggingface_hub 1.x), or the executable named by
 * `KARUME_HF_BIN`. The dataset is private: `hf` reads the maintainer's token (`hf auth login`).
 */

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { isNotFound, mkdir, remove, writeTextFile } from "@karume/test-support/fs";
import {
  readFetchedRevision,
  REVISION_FILE,
  TEST_ASSETS,
  TEST_ASSETS_ROOT,
  type TestAssetsPin,
} from "@karume/test-support/test-assets";

const USAGE = `Usage: node tools/test-assets/fetch.ts [--include <glob>]...
  --include <glob>   fetch only the matching paths (repeatable), e.g. 'goldens/<dir>/**'`;

/** A commit SHA (the pin must name one — a branch would move under the tests). */
const COMMIT_SHA = /^[0-9a-f]{40}$/;

/** The revision of a pin that was never set (see `TEST_ASSETS`). */
const PLACEHOLDER_REVISION = "0".repeat(40);

/** Reads `--include <glob>` pairs; anything else fails loudly. */
export const parseArgs = (argv: readonly string[]): readonly string[] => {
  const includes: string[] = [];
  for (let at = 0; at < argv.length; at += 2) {
    const key = argv[at];
    const value = argv[at + 1];
    if (key !== "--include") throw new Error(`unknown argument '${key}'\n${USAGE}`);
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`--include needs a glob\n${USAGE}`);
    }
    includes.push(value);
  }
  return includes;
};

/** The command that stands for `hf`: `KARUME_HF_BIN` when set, otherwise `hf` on PATH. */
export const hfCommand = (
  env: Readonly<Record<string, string | undefined>> = process.env,
): readonly string[] => {
  const bin = env.KARUME_HF_BIN;
  return [bin === undefined || bin === "" ? "hf" : bin];
};

/** The `hf download` arguments for `pin` into `root`, with each include passed through. */
export const downloadArgs = (
  pin: TestAssetsPin,
  root: URL,
  includes: readonly string[],
): readonly string[] => [
  "download",
  pin.repo,
  "--repo-type",
  "dataset",
  "--revision",
  pin.revision,
  "--local-dir",
  fileURLToPath(root),
  // MUST: never take a `.revision` from the dataset — it would stand for a completed fetch while
  // the download is still running (or after it was interrupted).
  "--exclude",
  REVISION_FILE,
  ...includes.flatMap((glob) => ["--include", glob]),
];

export type FetchOptions = {
  readonly pin?: TestAssetsPin;
  readonly root?: URL;
  /** `--include` globs; any include makes the fetch partial (no `.revision` is written). */
  readonly includes?: readonly string[];
  /** The executable (and leading arguments) standing for `hf` — tests pass a fake. */
  readonly command?: readonly string[];
};

/** Runs `command` with stdio inherited; resolves on exit 0, rejects naming the exit otherwise. */
const runToCompletion = (command: readonly string[], args: readonly string[]): Promise<void> =>
  new Promise((resolve, reject) => {
    const [bin, ...leading] = command;
    const child = spawn(bin, [...leading, ...args], { stdio: "inherit" });
    child.on("error", (cause) => {
      reject(
        new Error(
          isNotFound(cause)
            ? `'${bin}' was not found: install the Hugging Face CLI (huggingface_hub 1.x, \`hf\`) or ` +
                "point KARUME_HF_BIN at it"
            : `could not start '${bin}': ${cause.message}`,
          { cause },
        ),
      );
    });
    child.on("close", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `'${bin} download' ${signal === null ? `exited with ${code}` : `was killed by ${signal}`}` +
            ` — the test assets were not recorded as fetched (no ${REVISION_FILE})`,
        ),
      );
    });
  });

/**
 * Downloads the pinned dataset into `root` and, for a complete fetch, records the revision once
 * `hf` exited 0.
 *
 * MUST: the stale `.revision` is removed before the download and the new one is written only after
 * a complete download — otherwise an interrupted fetch would leave the previous revision's record on
 * a root that holds a mix of two revisions, and a partial one would stand for assets it did not
 * fetch; the gate would read either as fetched. A partial fetch keeps the record of a complete fetch
 * of the same revision (re-fetching part of a complete root leaves it complete).
 */
export const fetchTestAssets = async (options: FetchOptions = {}): Promise<void> => {
  const pin = options.pin ?? TEST_ASSETS;
  const root = options.root ?? TEST_ASSETS_ROOT;
  const includes = options.includes ?? [];
  const partial = includes.length > 0;
  if (pin.revision === PLACEHOLDER_REVISION) {
    throw new Error(
      `the pin of ${pin.repo} is still the placeholder revision ${PLACEHOLDER_REVISION}: ` +
        "set revision in packages/test-support/src/test-assets.pin.json to the dataset's commit SHA",
    );
  }
  if (!COMMIT_SHA.test(pin.revision)) {
    throw new Error(
      `the pin of ${pin.repo} must be a 40-hex commit SHA, got '${pin.revision}' ` +
        "(a branch or tag would move under the tests)",
    );
  }
  await mkdir(root, { recursive: true });
  const revisionFile = new URL(REVISION_FILE, root);
  const keepRecord = partial && readFetchedRevision(root) === pin.revision;
  if (!keepRecord) {
    await remove(revisionFile).catch((cause: unknown) => {
      if (!isNotFound(cause)) throw cause;
    });
  }
  await runToCompletion(options.command ?? hfCommand(), downloadArgs(pin, root, includes));
  if (!partial) await writeTextFile(revisionFile, `${pin.revision}\n`);
};

if (import.meta.main) {
  try {
    const includes = parseArgs(process.argv.slice(2));
    await fetchTestAssets({ includes });
    console.log(
      `fetched ${TEST_ASSETS.repo}@${TEST_ASSETS.revision} into ${fileURLToPath(TEST_ASSETS_ROOT)}` +
        (includes.length === 0
          ? ""
          : " (partial: not recorded as a complete fetch unless one of this revision was already there)"),
    );
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    // process.exit() is not used: it cuts off stderr queued on an asynchronous pipe.
    process.exitCode = 1;
  }
}

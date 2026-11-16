/**
 * The checkout a record was made on, read from git and `packages/runtime/package.json` at the
 * moment it is asked for. Node only: the dev server answers `GET /workbench/checkout` with it, a
 * production build bakes it in when its config loads, and the Node run reads it directly.
 */

import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";

import type { Checkout } from "./record.ts";

const run = promisify(execFile);

const git = async (repositoryRoot: string, ...args: string[]): Promise<string> =>
  (await run("git", ["-C", repositoryRoot, ...args], { maxBuffer: 64 * 1024 * 1024 })).stdout;

export const readCheckout = async (repositoryRoot: string): Promise<Checkout> => {
  const [head, status, manifest] = await Promise.all([
    git(repositoryRoot, "rev-parse", "HEAD"),
    git(repositoryRoot, "status", "--porcelain"),
    readFile(join(repositoryRoot, "packages/runtime/package.json"), "utf8"),
  ]);
  const parsed: unknown = JSON.parse(manifest);
  const version =
    typeof parsed === "object" && parsed !== null && "version" in parsed
      ? parsed.version
      : undefined;
  if (typeof version !== "string") {
    throw new Error("packages/runtime/package.json has no version");
  }
  return { commit: head.trim(), dirty: status.trim() !== "", runtimeVersion: version };
};

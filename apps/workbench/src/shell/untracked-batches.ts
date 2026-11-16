/**
 * The untracked batches as the page gets them: asked of the dev server once at start
 * (`GET /workbench/batches`). A production build has none and does not ask.
 *
 * SPEC: docs/spec/workbench.md#batches
 */

import type { BatchFile } from "../batch.ts";

const batchFileOf = (value: unknown): BatchFile | undefined => {
  if (typeof value !== "object" || value === null || !("id" in value)) return undefined;
  if (typeof value.id !== "string") return undefined;
  if ("text" in value && typeof value.text === "string") return { id: value.id, text: value.text };
  if ("problem" in value && typeof value.problem === "string") {
    return { id: value.id, problem: value.problem };
  }
  return undefined;
};

/** The dev server's untracked batch files; throws when it does not answer with a list of them. */
export const fetchUntrackedBatches = async (): Promise<readonly BatchFile[]> => {
  if (!import.meta.env.DEV) return [];
  const response = await fetch("/workbench/batches");
  if (!response.ok) throw new Error(`GET /workbench/batches answered ${response.status}`);
  const body: unknown = await response.json();
  const items: readonly unknown[] | undefined =
    typeof body === "object" && body !== null && "batches" in body && Array.isArray(body.batches)
      ? body.batches
      : undefined;
  const files = items?.map(batchFileOf);
  if (files === undefined || files.includes(undefined)) {
    throw new Error("GET /workbench/batches answered no list of batch files");
  }
  return files.filter((file) => file !== undefined);
};

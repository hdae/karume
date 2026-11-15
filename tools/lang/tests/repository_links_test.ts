// The link gate: every relative Markdown link and every `DECIDED:` / `SPEC:` pointer of the tracked files
// resolves — the file exists and, with a `#fragment`, so does the heading.
//
// Documents are moved and their headings reworded as the specs under `docs/spec/` take over from the
// decision records, and code points at a rule by path and heading. Running the check on the repository
// in the core lane turns a pointer left behind by such a move into a failing test instead of a dead link
// found by a reader. The scan reads about two thousand files in well under a second.

import { fileURLToPath } from "node:url";
import { assertEquals } from "@std/assert";
import { test } from "vite-plus/test";
import { checkRepository, describeBroken } from "../check-doc-links.ts";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

test("every Markdown link and every DECIDED: / SPEC: pointer in the tracked files resolves", () => {
  const report = checkRepository(REPO_ROOT);
  assertEquals(describeBroken(report.broken), []);
});

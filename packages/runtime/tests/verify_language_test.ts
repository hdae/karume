// The language gate (ADR 0130, step 2-9h — the enforcement of the repository language).
//
// The repository is English; the files that keep Japanese — dated research snapshots, the Japanese
// term table, superseded ADRs, data such as vocabularies and TTS inputs, and the areas the ADR's
// staging leaves for 0.15.x — are listed with their reasons in `tools/lang/language-allow-list.ts`.
// This test scans every tracked text file (`git ls-files`) and fails on Japanese outside that list,
// on a file above its line budget, and on an allowance that covers nothing any more, so the list
// shrinks as the remaining areas are translated and never keeps a dead exception.
//
// It runs in the core lane like `verify_lanes_test.ts`: the scan reads about two thousand files in
// well under a second.

import { fileURLToPath } from "node:url";
import { assert } from "@std/assert";
import { test } from "vite-plus/test";
import { describeReport, isGreen, runGate } from "../../../tools/lang/language-gate.ts";

const REPO_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

test("every tracked file with Japanese text is covered by the allow-list of ADR 0130", () => {
  const report = runGate(REPO_ROOT);
  assert(isGreen(report), `language gate:\n${describeReport(report).join("\n")}`);
});

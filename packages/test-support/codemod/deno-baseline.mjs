#!/usr/bin/env node
// Per-file counts from the Deno JUnit baseline (outputs/misc/0.15-baseline-2026-10-16/deno-junit).
//   node packages/test-support/codemod/deno-baseline.mjs <test file>...
// Prints, per file: top-level tests (Deno `Deno.test` / bdd `it`, steps excluded), how many of
// them were ignored (skipped) and failed. Vitest's summary for the ported file must show the same
// passed / skipped numbers.
import { readdirSync, readFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";

const root = resolve(new URL("../../../", import.meta.url).pathname);
const junitDir = join(root, "outputs/misc/0.15-baseline-2026-10-16/deno-junit");
const suites = new Map();
for (const name of readdirSync(junitDir).filter((n) => n.endsWith(".xml"))) {
  const xml = readFileSync(join(junitDir, name), "utf8");
  const re = /<testsuite name="([^"]+)"[^>]*>([\s\S]*?)<\/testsuite>/g;
  for (const m of xml.matchAll(re)) {
    const file = m[1].replace(/^\.\//, "");
    const body = m[2];
    const cases = [...body.matchAll(/<testcase name="([^"]*)"[^>]*>([\s\S]*?)<\/testcase>/g)];
    const top = cases.filter((c) => !c[1].includes(" &gt; "));
    const entry = suites.get(file) ?? { lane: name.replace(/\.xml$/, ""), tests: 0, skipped: 0, failed: 0 };
    entry.tests += top.length;
    entry.skipped += top.filter((c) => /<skipped/.test(c[2])).length;
    entry.failed += top.filter((c) => /<failure/.test(c[2])).length;
    suites.set(file, entry);
  }
}
const files = process.argv.slice(2);
if (files.length === 0) {
  console.error("usage: deno-baseline.mjs <test file>...");
  process.exit(2);
}
let totals = { tests: 0, skipped: 0, failed: 0 };
for (const f of files) {
  const rel = relative(root, resolve(f));
  const e = suites.get(rel);
  if (!e) { console.log(`${rel}: (not in the Deno baseline)`); continue; }
  console.log(`${rel}: lane=${e.lane} tests=${e.tests} passed=${e.tests - e.skipped - e.failed} skipped=${e.skipped} failed=${e.failed}`);
  totals.tests += e.tests; totals.skipped += e.skipped; totals.failed += e.failed;
}
console.log(`total: tests=${totals.tests} passed=${totals.tests - totals.skipped - totals.failed} skipped=${totals.skipped} failed=${totals.failed}`);

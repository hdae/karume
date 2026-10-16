#!/usr/bin/env node
// Type-check gate for the port. Writes a private tsconfig (extends the root one) whose `include`
// covers the package sources, test-support, and every test file that no longer references Deno,
// runs tsc on it, and hides errors located in files that still reference Deno (those are reached
// through imports from ported files). Exit 1 when any error remains in the ported scope.
//
//   node packages/test-support/codemod/check-ported.mjs            # whole ported scope
//   node packages/test-support/codemod/check-ported.mjs <file>...  # only errors in these files
import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";

const root = resolve(new URL("../../../", import.meta.url).pathname);
const only = new Set(process.argv.slice(2).map((p) => relative(root, resolve(p))));
const walk = (dir) => readdirSync(dir).flatMap((name) => {
  const path = join(dir, name);
  return statSync(path).isDirectory() ? walk(path) : path.endsWith(".ts") ? [path] : [];
});
const usesDeno = (text) =>
  text.split("\n").some((line) => /\bDeno\./.test(line) && !/^\s*(\*|\/\/|\/\*)/.test(line)) ||
  /from "@std\/testing\/bdd"/.test(text);
const ported = [];
const pending = new Set();
for (const pkg of readdirSync(join(root, "packages"))) {
  let files;
  try { files = walk(join(root, "packages", pkg, "tests")); } catch { continue; }
  for (const file of files) {
    const rel = relative(root, file);
    (usesDeno(readFileSync(file, "utf8")) ? pending : ported).push?.(rel) ?? pending.add(rel);
  }
}
const config = join(root, `tsconfig.check-${process.pid}.json`);
writeFileSync(config, JSON.stringify({
  extends: "./tsconfig.json",
  include: ["packages/*/*.ts", "packages/*/src/**/*.ts", "packages/test-support/src/**/*.ts", "vite.config.ts", ...ported.sort()],
}, null, 2));
let lines;
try {
  const result = spawnSync("pnpm", ["exec", "tsc", "-p", config, "--noEmit"], { cwd: root, encoding: "utf8" });
  lines = (result.stdout + result.stderr).split("\n").filter((l) => l.trim() !== "");
} finally {
  rmSync(config, { force: true });
}
const kept = lines.filter((line) => {
  const m = line.match(/^([^(]+)\(\d+,\d+\): error/);
  if (!m) return only.size === 0; // non-location lines (e.g. config errors) only in whole-scope mode
  if (pending.has(m[1])) return false;
  return only.size === 0 || only.has(m[1]);
});
for (const line of kept) console.log(line);
console.log(`check-ported: ${kept.length} error line(s)${only.size > 0 ? " in the given files" : " in ported scope"} (${lines.length - kept.length} hidden; ${ported.length} ported / ${pending.size} pending test files)`);
process.exit(kept.length === 0 ? 0 : 1);

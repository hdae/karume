#!/usr/bin/env node
// Moves test files into packages/e2e/tests/ (git mv) and re-relativizes the relative module
// specifiers (`from "./x"`, `import "./x"`, `import("./x")`) and `new URL("./x", import.meta.url)`
// literals so they keep pointing at the same files. Other relative-looking string literals are
// listed for manual review.
//   node packages/test-support/codemod/move-e2e.mjs [--write] <file.ts>...
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, relative, resolve, sep } from "node:path";

const root = resolve(new URL("../../../", import.meta.url).pathname);
const destDir = join(root, "packages/e2e/tests");
const write = process.argv.includes("--write");
const files = process.argv.slice(2).filter((a) => a !== "--write");
if (files.length === 0) { console.error("usage: move-e2e.mjs [--write] <file.ts>..."); process.exit(2); }

const relSpec = (fromDir, target) => {
  let r = relative(fromDir, target).split(sep).join("/");
  if (!r.startsWith(".")) r = `./${r}`;
  return r;
};

if (write) mkdirSync(destDir, { recursive: true });
let total = 0;
for (const file of files) {
  const abs = resolve(file);
  const oldDir = dirname(abs);
  const newAbs = join(destDir, basename(abs));
  if (existsSync(newAbs)) throw new Error(`${newAbs} already exists`);
  let text = readFileSync(abs, "utf8");
  let rewritten = 0;
  const rewrite = (spec) => {
    const out = relSpec(destDir, resolve(oldDir, spec));
    if (out !== spec) rewritten++;
    return out;
  };
  text = text.replace(/((?:from|import)\s*")(\.\.?\/[^"\n]+)(")/g, (_, a, s, b) => a + rewrite(s) + b);
  text = text.replace(/(import\(\s*")(\.\.?\/[^"\n]+)(")/g, (_, a, s, b) => a + rewrite(s) + b);
  text = text.replace(/(new URL\(\s*")(\.\.?\/[^"\n]+)("\s*,\s*import\.meta\.url\s*\))/g, (_, a, s, b) => a + rewrite(s) + b);
  const leftovers = [];
  text.split("\n").forEach((line, i) => {
    if (/^\s*(\*|\/\/|\/\*)/.test(line)) return;
    for (const m of line.matchAll(/"(\.\.?\/[^"\n]*)"/g)) {
      if (/(?:from|import)\s*"$/.test(line.slice(0, m.index)) || /import\(\s*"$/.test(line.slice(0, m.index))) continue;
      if (/new URL\(\s*"$/.test(line.slice(0, m.index)) && /import\.meta\.url/.test(line)) continue;
      leftovers.push(`${i + 1}: ${line.trim().slice(0, 120)}`);
    }
  });
  total += rewritten;
  console.log(`${relative(root, abs)} → packages/e2e/tests/${basename(abs)}: ${rewritten} specifier(s) rewritten`);
  for (const l of leftovers) console.log(`  REVIEW ${l}`);
  if (write) {
    execFileSync("git", ["mv", abs, newAbs], { cwd: root });
    writeFileSync(newAbs, text);
  }
}
console.log(`\n${files.length} file(s), ${total} specifiers${write ? " (written)" : " (dry run)"}`);

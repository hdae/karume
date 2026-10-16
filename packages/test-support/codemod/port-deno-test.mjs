#!/usr/bin/env node
// Codemod for the Deno → Node (Vitest) port of the test suite. Mechanical, count-reported, idempotent.
//
//   node packages/test-support/codemod/port-deno-test.mjs [--write] <file.ts>...
//
// Without --write it only reports. Rewrites:
//   Deno.test(            → test(            (import from @karume/test-support/test)
//   Deno.TestContext      → TestContext      (type import from @karume/test-support/test)
//   "@std/testing/bdd"    → "@karume/test-support/test"
//   Deno.env.get("X")     → process.env.X     / Deno.env.get(expr) → process.env[expr]
//   Deno.<fs api>(        → <fs api>(         (import from @karume/test-support/fs; see FS_NAMES)
//   Deno.DirEntry         → DirEntry          (type import from @karume/test-support/fs)
//   x instanceof Deno.errors.NotFound → isNotFound(x)   (IsADirectory / FilesystemLoop likewise)
//   Deno.exit( / Deno.memoryUsage() / Deno.build.os / Deno.build.arch → process.*
// Everything else that still mentions `Deno.` is listed as REMAINING for manual porting
// (Deno.Command / Deno.open / Deno.serve / Deno.execPath / Deno.version ...).

import { readFileSync, writeFileSync } from "node:fs";

const FS_NAMES = [
  "readFile",
  "readFileSync",
  "readTextFile",
  "readTextFileSync",
  "writeFile",
  "writeFileSync",
  "writeTextFile",
  "writeTextFileSync",
  "mkdir",
  "mkdirSync",
  "remove",
  "removeSync",
  "makeTempDir",
  "makeTempDirSync",
  "stat",
  "statSync",
  "readDir",
  "readDirSync",
  "symlink",
  "symlinkSync",
  "readLinkSync",
  "realPathSync",
];
const ERROR_PREDICATES = {
  NotFound: "isNotFound",
  IsADirectory: "isADirectory",
  FilesystemLoop: "isFilesystemLoop",
};
const TEST_MODULE = "@karume/test-support/test";
const FS_MODULE = "@karume/test-support/fs";

const args = process.argv.slice(2);
const write = args.includes("--write");
const files = args.filter((a) => a !== "--write");
if (files.length === 0) {
  console.error("usage: port-deno-test.mjs [--write] <file.ts>...");
  process.exit(2);
}

const count = (text, re) => (text.match(re) ?? []).length;

/** Merge `names` into an existing `import { ... } from "<module>";` or insert a new import at the top. */
const ensureImport = (text, module, names) => {
  if (names.size === 0) return text;
  const re = new RegExp(
    `^import \\{([^}]*)\\} from "${module.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&")}";\\n`,
    "m",
  );
  const existing = text.match(re);
  if (existing) {
    const present = new Set(
      existing[1]
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    );
    for (const n of names) present.add(n);
    const sorted = [...present].sort((a, b) =>
      a.replace(/^type /, "").localeCompare(b.replace(/^type /, "")),
    );
    return text.replace(re, `import { ${sorted.join(", ")} } from "${module}";\n`);
  }
  const sorted = [...names].sort((a, b) =>
    a.replace(/^type /, "").localeCompare(b.replace(/^type /, "")),
  );
  const line = `import { ${sorted.join(", ")} } from "${module}";\n`;
  const at = text.search(/^import /m);
  return at < 0 ? line + text : text.slice(0, at) + line + text.slice(at);
};

let totalRemaining = 0;
for (const file of files) {
  const original = readFileSync(file, "utf8");
  let text = original;
  const report = [];
  const testNames = new Set();
  const fsNames = new Set();

  // bdd → test shim
  if (/from "@std\/testing\/bdd";/.test(text)) {
    text = text.replace(/from "@std\/testing\/bdd";/g, `from "${TEST_MODULE}";`);
    report.push("bdd import → test-support");
  }
  // Deno.test(
  const nTest = count(text, /\bDeno\.test\(/g);
  if (nTest > 0) {
    // A file that declares its own `test` helper (e.g. `const test = (name, fn) => Deno.test({...})`)
    // gets the shim under the alias `denoTest` so the local name keeps its meaning.
    const localTest = /^(?:export )?(?:const|let|var|function|async function) test\b/m.test(text);
    const name = localTest ? "denoTest" : "test";
    text = text.replace(/\bDeno\.test\(/g, `${name}(`);
    testNames.add(localTest ? "test as denoTest" : "test");
    report.push(`Deno.test ×${nTest}${localTest ? " (aliased denoTest)" : ""}`);
  }
  const nCtx = count(text, /\bDeno\.TestContext\b/g);
  if (nCtx > 0) {
    text = text.replace(/\bDeno\.TestContext\b/g, "TestContext");
    testNames.add("type TestContext");
    report.push(`Deno.TestContext ×${nCtx}`);
  }
  // env
  const nEnvLit = count(text, /\bDeno\.env\.get\("([A-Z0-9_]+)"\)/g);
  if (nEnvLit > 0) {
    text = text.replace(/\bDeno\.env\.get\("([A-Z0-9_]+)"\)/g, "process.env.$1");
    report.push(`Deno.env.get(literal) ×${nEnvLit}`);
  }
  const nEnvExpr = count(text, /\bDeno\.env\.get\(([^()"]+)\)/g);
  if (nEnvExpr > 0) {
    text = text.replace(/\bDeno\.env\.get\(([^()"]+)\)/g, "process.env[$1]");
    report.push(`Deno.env.get(expr) ×${nEnvExpr}`);
  }
  // fs
  for (const name of FS_NAMES) {
    const re = new RegExp(`\\bDeno\\.${name}\\(`, "g");
    const n = count(text, re);
    if (n > 0) {
      text = text.replace(re, `${name}(`);
      fsNames.add(name);
      report.push(`Deno.${name} ×${n}`);
    }
  }
  const nDirEntry = count(text, /\bDeno\.DirEntry\b/g);
  if (nDirEntry > 0) {
    text = text.replace(/\bDeno\.DirEntry\b/g, "DirEntry");
    fsNames.add("type DirEntry");
    report.push(`Deno.DirEntry ×${nDirEntry}`);
  }
  for (const [cls, predicate] of Object.entries(ERROR_PREDICATES)) {
    const re = new RegExp(`\\b([A-Za-z_$][\\w$]*) instanceof Deno\\.errors\\.${cls}\\b`, "g");
    const n = count(text, re);
    if (n > 0) {
      text = text.replace(re, `${predicate}($1)`);
      fsNames.add(predicate);
      report.push(`instanceof Deno.errors.${cls} ×${n}`);
    }
  }
  // process
  const simple = [
    [/\bDeno\.exit\(/g, "process.exit(", "Deno.exit"],
    [/\bDeno\.memoryUsage\(\)/g, "process.memoryUsage()", "Deno.memoryUsage"],
    [/\bDeno\.build\.os\b/g, "process.platform", "Deno.build.os"],
    [/\bDeno\.build\.arch\b/g, "process.arch", "Deno.build.arch"],
  ];
  for (const [re, to, label] of simple) {
    const n = count(text, re);
    if (n > 0) {
      text = text.replace(re, to);
      report.push(`${label} ×${n}`);
    }
  }

  text = ensureImport(text, TEST_MODULE, testNames);
  text = ensureImport(text, FS_MODULE, fsNames);

  const remaining = [];
  text.split("\n").forEach((line, i) => {
    if (/\bDeno\./.test(line) && !/^\s*(\*|\/\/|\/\*)/.test(line))
      remaining.push(`${i + 1}: ${line.trim().slice(0, 110)}`);
  });
  totalRemaining += remaining.length;

  const changed = text !== original;
  console.log(`${file}: ${changed ? report.join(", ") : "no change"}`);
  for (const r of remaining) console.log(`  REMAINING ${r}`);
  if (write && changed) writeFileSync(file, text);
}
console.log(
  `\n${files.length} file(s), ${totalRemaining} remaining Deno reference(s)${write ? "" : " (dry run — nothing written)"}`,
);

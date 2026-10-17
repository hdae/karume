#!/usr/bin/env node
// Phase-2 codemod: replace the Deno-shaped registration forms of @karume/test-support/test with
// Vitest's own. Object forms `test({ name, ignore, fn })` / `it({...})` / `describe({...})` become
// `test.skipIf(ignore)(name, fn)` (plain `test(name, fn)` without `ignore`), the three-argument
// forms `it(name, { ignore }, fn)` likewise, and the import moves to "vite-plus/test". Files that
// use `t.step` (or the shim's TestContext / StepFailureError / TestFunction types) are left for the
// manual conversion and reported.
//   node packages/test-support/codemod/port-shim.mjs [--write] <file.ts>...
import { parseSync } from "oxc-parser";
import { readFileSync, writeFileSync } from "node:fs";

const write = process.argv.includes("--write");
const files = process.argv.slice(2).filter((a) => a !== "--write");
if (files.length === 0) {
  console.error("usage: port-shim.mjs [--write] <file.ts>...");
  process.exit(2);
}
const SHIM = "@karume/test-support/test";
const CALLEES = new Set(["test", "it", "describe"]);
const STEP_TYPES = /\b(TestContext|StepFailureError|TestFunction)\b|\bt\.step\(/;

let totalObject = 0,
  totalThree = 0,
  skippedStep = 0,
  changedFiles = 0;
const warnings = [];

const walk = (node, visit) => {
  if (node === null || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const child of node) walk(child, visit);
    return;
  }
  if (typeof node.type === "string") visit(node);
  for (const key of Object.keys(node)) {
    if (key === "type" || key === "span" || key === "start" || key === "end") continue;
    walk(node[key], visit);
  }
};

for (const file of files) {
  const source = readFileSync(file, "utf8");
  if (!source.includes(SHIM)) continue;
  if (STEP_TYPES.test(source)) {
    skippedStep++;
    warnings.push(`${file}: uses steps or shim types — manual conversion`);
    continue;
  }
  const { program, errors } = parseSync(file, source, { sourceType: "module", lang: "ts" });
  if (errors.length > 0) {
    warnings.push(`${file}: parse errors ${errors.length}`);
    continue;
  }
  const edits = [];
  const src = (n) => source.slice(n.start, n.end);
  walk(program, (node) => {
    if (
      node.type !== "CallExpression" ||
      node.callee.type !== "Identifier" ||
      !CALLEES.has(node.callee.name)
    )
      return;
    const callee = node.callee.name;
    const args = node.arguments;
    if (args.length === 1 && args[0].type === "ObjectExpression") {
      const props = Object.fromEntries(
        args[0].properties
          .filter((p) => p.type === "ObjectProperty" || p.type === "Property")
          .map((p) => [p.key.name ?? p.key.value, p]),
      );
      const name = props.name,
        fn = props.fn,
        ignore = props.ignore;
      const extra = Object.keys(props).filter((k) => !["name", "fn", "ignore"].includes(k));
      if (!name || !fn || extra.length) {
        warnings.push(
          `${file}:${node.start}: object form with keys ${Object.keys(props).join(",")} left as is`,
        );
        return;
      }
      let fnSrc;
      if (
        fn.method ||
        (fn.kind === "init" &&
          fn.value.type === "FunctionExpression" &&
          source.slice(fn.start, fn.value.start).trim() !== "fn:")
      ) {
        // method shorthand `fn(t) {…}` / `async fn() {…}` → function expression
        const v = fn.value;
        fnSrc = `${v.async ? "async " : ""}function ${source.slice(v.start, v.end)}`;
        if (!source.slice(v.start, v.end).startsWith("("))
          fnSrc = `${v.async ? "async " : ""}function ${source.slice(v.start, v.end).replace(/^[^(]*/, "")}`;
      } else fnSrc = src(fn.value);
      const head = ignore ? `${callee}.skipIf(${src(ignore.value)})` : callee;
      edits.push({
        start: node.start,
        end: node.end,
        text: `${head}(${src(name.value)}, ${fnSrc})`,
      });
      totalObject++;
      return;
    }
    if (args.length === 3 && args[1].type === "ObjectExpression") {
      const props = Object.fromEntries(
        args[1].properties.map((p) => [p.key.name ?? p.key.value, p]),
      );
      const keys = Object.keys(props);
      if (keys.length !== 1 || !props.ignore) {
        warnings.push(
          `${file}:${node.start}: three-arg form with options ${keys.join(",")} left as is`,
        );
        return;
      }
      edits.push({
        start: node.start,
        end: node.end,
        text: `${callee}.skipIf(${src(props.ignore.value)})(${src(args[0])}, ${src(args[2])})`,
      });
      totalThree++;
    }
  });
  // nested calls: apply inner edits first by sorting descending and skipping overlaps
  edits.sort((a, b) => b.start - a.start);
  let out = source;
  let last = Infinity;
  for (const e of edits) {
    if (e.end > last) {
      warnings.push(`${file}: overlapping edit skipped at ${e.start}`);
      continue;
    }
    out = out.slice(0, e.start) + e.text + out.slice(e.end);
    last = e.start;
  }
  out = out.split(`"${SHIM}"`).join('"vite-plus/test"');
  if (out !== source) {
    changedFiles++;
    if (write) writeFileSync(file, out);
  }
}
console.log(
  `port-shim: ${changedFiles} file(s) ${write ? "written" : "would change"}; object forms ${totalObject}, three-arg forms ${totalThree}; ${skippedStep} step file(s) left for manual conversion`,
);
for (const w of warnings) console.log("  " + w);

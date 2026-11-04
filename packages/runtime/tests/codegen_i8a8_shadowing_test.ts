/**
 * The check of **overlapping declaration names** (shadowing) in the i8a8 kernel generators.
 *
 * In WGSL a `let` / `var` in a block may legally hide a same-named declaration of an outer scope;
 * it compiles and passes validation. When a generator builds slot-numbered names
 * (`<prefix><number>`), some geometries produce a name that collides with the function-scope `k4`
 * / `n4` (pack counts), and a range test silently becomes false, giving a wrong value (measured in
 * the fill of fused attention ①QK — 2026-09-27, 16 geometries with K-side fill slot 5 or above).
 * The test scans the body of `fn main` in the generated WGSL and checks, over every candidate
 * geometry × 3 kernels × variants, that no inner scope declares a name already in an outer scope
 * (parameters included).
 */

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { attentionPvI8a8Wgsl, attentionQkI8a8Wgsl } from "../src/kernels/attention-i8a8.ts";
import {
  ATTENTION_ONLINE_REDUCES,
  attentionOnlineI8a8Wgsl,
} from "../src/kernels/attention-online-i8a8.ts";
import { conv3dI8a8Wgsl } from "../src/kernels/conv3d-i8a8.ts";
import { linearI8a8Wgsl } from "../src/kernels/linear-i8a8.ts";
import {
  assertI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
} from "../src/kernels/i8a8-geometry.ts";

/**
 * Filters the same grid as the sweep's candidates (regM / regN ∈ {4,8}, wgX ∈ {8,16}, wgY ∈
 * {4,8,16}, tileK ∈ {16,32}) through the check.
 */
const candidateGeometries = (): I8a8Geometry[] => {
  const found: I8a8Geometry[] = [];
  for (const regM of [4, 8]) {
    for (const regN of [4, 8]) {
      for (const wgX of [8, 16]) {
        for (const wgY of [4, 8, 16]) {
          for (const tileK of [16, 32]) {
            const geometry: I8a8Geometry = { regM, regN, wgX, wgY, tileK };
            try {
              assertI8a8Geometry(geometry, "candidate");
              found.push(geometry);
            } catch {
              // A geometry the check rejects is not a candidate
            }
          }
        }
      }
    }
  }
  return found;
};

/**
 * The position of the `)` matching the `(` at `open`. It counts parenthesis depth —— stopping at
 * the first `)` would end the parameter list at the closing parenthesis of a parameter attribute
 * `@builtin(...)`.
 */
const matchingParen = (text: string, open: number): number => {
  assertEquals(text[open], "(", "position of the parameter list's opening parenthesis");
  let depth = 0;
  for (let at = open; at < text.length; at += 1) {
    if (text[at] === "(") depth += 1;
    if (text[at] !== ")") continue;
    depth -= 1;
    if (depth === 0) return at;
  }
  throw new Error("the parameter list of fn main is not closed");
};

/**
 * Returns the places in the body of `fn main` where `let` / `var` declares a name already in an
 * outer scope (parameters, enclosing blocks) (`name@depth`). The header variable of
 * `for (var i …)` goes into the scope of the block that follows.
 */
export const shadowedDeclarations = (wgsl: string): string[] => {
  const start = wgsl.indexOf("fn main(");
  assert(start >= 0, "no fn main");
  const paramsEnd = matchingParen(wgsl, start + "fn main".length);
  const params = [...wgsl.slice(start, paramsEnd).matchAll(/(\w+)\s*:/g)].map((m) => m[1]);
  const body = wgsl.slice(paramsEnd);
  const loopVars = new Set(
    [...body.matchAll(/\bfor\s*\(\s*var\s+(\w+)/g)].map((m) => m.index + m[0].length),
  );
  const scopes: Set<string>[] = [];
  let pending: string[] = [];
  const shadowed: string[] = [];
  for (const match of body.matchAll(/[{}]|\b(let|var)\s+([A-Za-z_]\w*)/g)) {
    if (match[0] === "{") {
      const scope = new Set(scopes.length === 0 ? params : []);
      for (const name of pending) scope.add(name);
      pending = [];
      scopes.push(scope);
      continue;
    }
    if (match[0] === "}") {
      scopes.pop();
      if (scopes.length === 0) break;
      continue;
    }
    const name = match[2];
    if (scopes.some((scope) => scope.has(name))) shadowed.push(`${name}@${scopes.length}`);
    if (loopVars.has(match.index + match[0].length)) pending.push(name);
    else scopes[scopes.length - 1].add(name);
  }
  return shadowed;
};

test("shadowedDeclarations finds shadowing and allows the same name in sibling blocks", () => {
  const shadowing = `fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let k4 = 4u;
  for (var t = 0u; t < 2u; t = t + 1u) {
    var k4 = 0u;
    if (t < k4) { k4 = 1u; }
  }
}`;
  assertEquals(shadowedDeclarations(shadowing), ["k4@2"]);
  const siblings = `fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  for (var i = 0u; i < 2u; i = i + 1u) { var x = i; }
  for (var i = 0u; i < 2u; i = i + 1u) { var x = i; }
  { let lid2 = lid; }
}`;
  assertEquals(shadowedDeclarations(siblings), []);
  assertEquals(shadowedDeclarations(`fn main(a: u32) { var a = 1u; }`), ["a@1"]);
});

test("shadowedDeclarations also puts attributed parameters (the shape real kernels use) into the outer scope", () => {
  // A real kernel's `fn main` starts with `@builtin(workgroup_id) wid`. Ending the parameter list
  // at the first `)` stops at the attribute's closing parenthesis and records no parameter name
  // at all (missing every shadowing of a parameter).
  const attributed = (body: string): string =>
    `fn main(
  @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
) { ${body} }`;
  assertEquals(shadowedDeclarations(attributed("let lid = 1u;")), ["lid@1"]);
  assertEquals(shadowedDeclarations(attributed("let wid = 0u; { var lid = 2u; }")), [
    "wid@1",
    "lid@2",
  ]);
  // Control: a declaration with a name different from the parameters reports nothing.
  assertEquals(shadowedDeclarations(attributed("let tile = wid.x + lid.y;")), []);
});

test("the 3 i8a8 kernels never redeclare an outer-scope name for any candidate geometry × variant", () => {
  const geometries = candidateGeometries();
  assert(geometries.length >= 48, `only ${geometries.length} candidates`);
  // The grid must contain the shapes that hit the defect (K-side fill slots = regN · tileK / 4 /
  // wgY of 5 or above)
  assert(
    geometries.some((g) => (g.regN * (g.tileK / 4)) / g.wgY >= 5),
    "no candidate geometry with 5 or more fill slots (the check cannot hit the defect)",
  );
  const failures: string[] = [];
  for (const geometry of geometries) {
    const tag = i8a8GeometryKeyPart(geometry, false);
    for (const dp4a of [true, false]) {
      for (const v4 of [true, false]) {
        const variants: [string, () => string][] = [
          [`qk v4=${v4} dp4a=${dp4a}`, () => attentionQkI8a8Wgsl(v4, dp4a, "f32", geometry)],
          [
            `qk rw v4=${v4} dp4a=${dp4a}`,
            () => attentionQkI8a8Wgsl(v4, dp4a, "f32", geometry, true),
          ],
          [`pv v4=${v4} dp4a=${dp4a}`, () => attentionPvI8a8Wgsl(v4, dp4a, "f32", geometry)],
          [
            `pv rw v4=${v4} dp4a=${dp4a}`,
            () => attentionPvI8a8Wgsl(v4, dp4a, "f32", geometry, true),
          ],
          [`linear v4=${v4} dp4a=${dp4a}`, () => linearI8a8Wgsl(v4, dp4a, geometry)],
          [`conv3d v4=${v4} dp4a=${dp4a}`, () => conv3dI8a8Wgsl(v4, dp4a, geometry)],
        ];
        if (v4) {
          variants.push([
            `qk s16 dp4a=${dp4a}`,
            () => attentionQkI8a8Wgsl(true, dp4a, "f16", geometry),
          ]);
          variants.push([
            `pv s16 dp4a=${dp4a}`,
            () => attentionPvI8a8Wgsl(true, dp4a, "f16", geometry),
          ]);
        }
        for (const [label, generate] of variants) {
          const shadowed = shadowedDeclarations(generate());
          if (shadowed.length > 0) failures.push(`${tag} ${label}: ${shadowed.join(", ")}`);
        }
      }
    }
  }
  assertEquals(failures, []);
});

test("the online int8 attention generator declares no shadowed name (both depths × reductions × dot)", () => {
  for (const depth of [64, 128]) {
    for (const reduce of ATTENTION_ONLINE_REDUCES) {
      for (const dp4a of [true, false]) {
        assertEquals(
          shadowedDeclarations(attentionOnlineI8a8Wgsl(dp4a, depth, reduce)),
          [],
          `attention_online d${depth} ${reduce} ${dp4a ? "dp4a" : "emu"}`,
        );
      }
    }
  }
});

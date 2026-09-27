/**
 * i8a8 カーネル生成器の**宣言名の重なり**（シャドーイング）の門。
 *
 * WGSL ではブロック内の `let` / `var` が外側スコープの同名を覆い隠すのが合法で、コンパイルも
 * validation も通る。生成器がスロット番号付きの名前（`<接頭辞><番号>`）を組むと、関数スコープの
 * `k4` / `n4`（パック数）と重なる形が幾何によって現れ、範囲判定が黙って偽になる誤値になる
 * （融合 attention ①QK の充填で実測 — 2026-09-27・K 側の充填スロット 5 以上の 16 幾何）。
 * 生成 WGSL の `fn main` 本体を走査し、外側スコープ（引数を含む）に既にある名前を内側で宣言して
 * いないことを、候補の全幾何 × 3 カーネル × 変種で見る。
 */

import { assert, assertEquals } from "@std/assert";
import { attentionPvI8a8Wgsl, attentionQkI8a8Wgsl } from "../src/kernels/attention-i8a8.ts";
import { linearI8a8Wgsl } from "../src/kernels/linear-i8a8.ts";
import {
  assertI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
} from "../src/kernels/i8a8-geometry.ts";

/** 掃引の候補と同じ格子（regM / regN ∈ {4,8}・wgX ∈ {8,16}・wgY ∈ {4,8,16}・tileK ∈ {16,32}）を門で濾す。 */
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
              // 門が落とす幾何は候補ではない
            }
          }
        }
      }
    }
  }
  return found;
};

/**
 * `fn main` 本体で、外側スコープ（引数・囲むブロック）に既にある名前を `let` / `var` で宣言している
 * 箇所を返す（`名前@深さ`）。`for (var i …)` のヘッダ変数は続くブロックのスコープに入れる。
 */
export const shadowedDeclarations = (wgsl: string): string[] => {
  const start = wgsl.indexOf("fn main(");
  assert(start >= 0, "fn main が無い");
  const paramsEnd = wgsl.indexOf(")", start);
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

Deno.test("shadowedDeclarations は覆い隠しを見つけ、兄弟ブロックの同名は許す", () => {
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

Deno.test("i8a8 の 3 カーネルは候補の全幾何 × 変種で外側スコープの名前を宣言し直さない", () => {
  const geometries = candidateGeometries();
  assert(geometries.length >= 48, `候補が ${geometries.length} 本しか無い`);
  // 欠陥を踏む形（K 側の充填スロット = regN · tileK / 4 / wgY が 5 以上）が格子に含まれていること
  assert(
    geometries.some((g) => g.regN * (g.tileK / 4) / g.wgY >= 5),
    "充填スロット 5 以上の幾何が候補に無い（門が欠陥を踏めない）",
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

// barrel（`mod.ts`）の**値非公開**の門（models の `models_barrel_surface_test.ts` と同型）。
// ネットワークも実資産も要らない。
//
// `DistributionSource` は**中身が不透明**なハンドルで、判別は `isDistributionSource` に閉じる
// （`mod.ts` の doc が理由の正本 — 構造で見分けようとすると綴り間違いが取得元として通る）。
// その不透明さは `export type` でしか保てないのに、`export` へ変えても型検査は通り、公開面
// スナップショット門も差分を出さない（`deno doc --json` は type-only 再輸出と値再輸出を
// 区別しない — 2026-09-21 実測）。値として出ていないことは実行時にしか観測できない。

import { assert, assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import * as hub from "../mod.ts";
import type { DistributionSource, LoadedManifest } from "../mod.ts";
import { fetchManifest } from "./helpers/fixture.ts";

/** 型としてのみ公開する綴り（不透明ハンドルの面）。 */
const TYPE_ONLY = ["DistributionSource"];

Deno.test("barrel: 不透明ハンドルの型は値として出ていない", () => {
  const surface = Object.keys(hub);
  // 陽性対照 — 空の名前空間を見て緑になっている形でないことを先に確かめる。
  assert(
    surface.includes("isDistributionSource"),
    `barrel の import が壊れている（${surface.length} 件）`,
  );
  assertEquals(
    surface.filter((name) => TYPE_ONLY.includes(name)),
    [],
    "型としてのみ公開する綴りが値として出ている（不透明ハンドルの判別が構造依存になる）",
  );
});

Deno.test("barrel: 不透明ハンドルと LoadedManifest は手で組んだ値を受けない", async () => {
  // 型の上でも受けない（名目型）— 構造的な型だと `HubRepoRef` の綴り間違いや手で組んだ値が
  // 型検査を通る。下の 2 行が型エラーにならなくなれば、この test ファイルの型検査が落ちる。
  // @ts-expect-error 構造的なオブジェクトは取得元ハンドルの型を満たさない
  const source: DistributionSource = {};
  // @ts-expect-error 公開欄を揃えただけの値は LoadedManifest の型を満たさない
  const loaded: LoadedManifest = { manifest: fetchManifest };
  // 実行時も受けない — 取得元は loadManifest の返り値にしか結び付いていない。
  assertEquals(hub.isDistributionSource(source), false);
  const error = await assertRejects(() => hub.fetchAssets(loaded, {}), Error);
  assertStringIncludes(error.message, "取得元が結び付いていない");
});

/**
 * `source` だけを書いた入口を `deno bundle`（minify しない — 名前で観測する）で束ねた JS を返す。
 * 設定（import map）はリポ直下の `deno.json` から読むので、cwd をリポ直下に置く。
 */
const bundle = async (source: string): Promise<string> => {
  const dir = await Deno.makeTempDir();
  try {
    const input = `${dir}/entry.ts`;
    const output = `${dir}/bundle.js`;
    await Deno.writeTextFile(input, source);
    const { success, stderr } = await new Deno.Command(Deno.execPath(), {
      args: ["bundle", "--quiet", "-o", output, input],
      cwd: new URL("../../../", import.meta.url),
      stdout: "null",
      stderr: "piped",
    }).output();
    if (!success) {
      throw new Error(`deno bundle が失敗した: ${new TextDecoder().decode(stderr)}`);
    }
    return await Deno.readTextFile(output);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
};

Deno.test("barrel: import するだけでは hub のクラスが bundle に残らない（import 時実行が無い）", async () => {
  // 横断の不変条件「全モジュール副作用ゼロ」は barrel 経由 tree-shaking の成立条件。クラス評価時に
  // 走るコード（`static` ブロック）を持つクラスは、使われていなくても bundler が副作用と見て残す。
  const barrel = new URL("../mod.ts", import.meta.url).href;
  // 陽性対照 — 使う側の bundle には同じ名前が現れる（名前で観測できる形であることを先に確かめる）。
  const used = await bundle(
    `import { isDistributionSource, loadManifest } from "${barrel}";\n` +
      `console.log(isDistributionSource, loadManifest);\n`,
  );
  for (const name of ["DistributionSource", "LoadedManifest"]) {
    assertStringIncludes(
      used,
      `var ${name} = class`,
      `陽性対照: 使う側の bundle に ${name} が無い`,
    );
  }
  const bare = await bundle(`import "${barrel}";\n`);
  for (const name of ["DistributionSource", "LoadedManifest"]) {
    assert(!bare.includes(name), `使われていない ${name} が bundle に残っている（import 時実行）`);
  }
});

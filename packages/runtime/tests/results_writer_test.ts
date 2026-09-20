// 検証結果の書き出し（`outputs/verify/<環境キー>/<日付>_<系列>/`）。
//
// ここで見るのは 3 つ: 席が環境キーで分かれること、`record` のたびに `results.json` が
// **その時点までの全件**で書き直されること（途中で落ちても直前までが残る）、実物の置き場が
// 同じディレクトリ配下に閉じること。実 GPU も `outputs/` も要らない — 根と環境は注入する。

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { Environment } from "./helpers/environment.ts";
import { openResults, type ResultEntry } from "./helpers/results.ts";

const KEY = "deno-intel-graphics-bmg-g21";

const ENVIRONMENT: Environment = {
  key: KEY,
  runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
  adapter: { vendor: "32902", architecture: "", device: "57868", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
};

type ResultsDocument = {
  readonly schema: number;
  readonly family: string;
  readonly environment: Environment;
  readonly checkout: { readonly sha: string; readonly dirty: boolean } | null;
  readonly startedAt: string;
  readonly cases: readonly ResultEntry[];
};

Deno.test("結果の書き出し: record のたびに全件で書き直され、実物は同じ席に置かれる", async () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const root = new URL(`file://${temporary}/`);
    const results = openResults("irodori", { root, environment: ENVIRONMENT });
    const today = new Date().toISOString().slice(0, 10);
    assertEquals(results.dir.href, new URL(`${KEY}/${today}_irodori/`, root).href);

    const artifact = results.artifact("voice-clone.wav");
    assert(artifact.href.startsWith(results.dir.href), "実物が結果の席の外を指している");
    await Deno.writeFile(artifact, new Uint8Array([1, 2, 3]));

    await results.record({
      id: "voice-clone",
      status: "pass",
      expected: "a".repeat(64),
      actual: "a".repeat(64),
      artifact: "voice-clone.wav",
      elapsedMs: 12,
    });
    const afterFirst = JSON.parse(
      await Deno.readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(afterFirst.cases.length, 1, "1 件目が残っていない");

    await results.record({ id: "no-ref", status: "written", actual: "b".repeat(64), elapsedMs: 8 });
    const document = JSON.parse(
      await Deno.readTextFile(new URL("results.json", results.dir)),
    ) as ResultsDocument;
    assertEquals(document.schema, 1);
    assertEquals(document.family, "irodori");
    assertEquals(document.environment.key, KEY);
    assertEquals(document.cases.map((entry) => entry.id), ["voice-clone", "no-ref"]);
    assertEquals(document.cases[1].status, "written");
    // 参照値を作った回は expected を持たない（JSON.stringify が undefined の欄を落とす）。
    assertEquals(Object.hasOwn(document.cases[1], "expected"), false);
    assert(document.startedAt.endsWith("Z"), "startedAt が ISO 文字列でない");
    // このリポは git 管理下なのでチェックアウトは採れる（git が無い機でだけ null）。
    assert(
      document.checkout === null || /^[0-9a-f]{40}$/.test(document.checkout.sha),
      `checkout の sha が commit SHA でない: ${JSON.stringify(document.checkout)}`,
    );
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

Deno.test("結果の書き出し: ディレクトリ区切りを含む実物の名前は throw する", () => {
  const temporary = Deno.makeTempDirSync({ prefix: "karume-verify-" });
  try {
    const results = openResults("anima", {
      root: new URL(`file://${temporary}/`),
      environment: ENVIRONMENT,
    });
    assertThrows(() => results.artifact("F1/i8.wav"), Error, "ディレクトリ区切り");
  } finally {
    Deno.removeSync(temporary, { recursive: true });
  }
});

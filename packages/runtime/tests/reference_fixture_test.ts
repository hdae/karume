// 環境別の参照値 fixture（`packages/models/tests/fixtures/references/*.json`）の読み書き。
//
// ここで縛るのは**モードの意味論**そのもの: 未設定は比較だけ、`write` は無い行を足すだけ、
// `rewrite` は現環境の行だけを上書きする。この 3 つのどれかが緩むと、門が「他の機の参照値と
// 突き合わせて緑」「割れた値を黙って焼き直して緑」のどちらかに化ける。
//
// 実 GPU も環境変数も要らない — 環境とモードは引数で注入する。

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { Environment } from "./helpers/environment.ts";
import { openReferences, parseReferenceMode } from "./helpers/reference.ts";

const CURRENT = "deno-intel-graphics-bmg-g21";
const OTHER = "deno-nvidia-geforce-rtx-3080-ti";

const environment = (key: string): Environment => ({
  key,
  runtime: { name: "deno", version: "2.9.6", v8: "15.0.245.2", typescript: "6.0.3" },
  adapter: { vendor: "32902", architecture: "", device: "57868", description: "test" },
  os: { platform: "linux", arch: "x86_64" },
});

/** 1 件だけ他環境の行を持つ fixture を一時ディレクトリに作る。 */
const withFixture = (
  cases: Record<string, Record<string, string>>,
  body: (fixtureUrl: URL) => void,
): void => {
  const dir = Deno.makeTempDirSync({ prefix: "karume-reference-" });
  try {
    const fixtureUrl = new URL("references.json", `file://${dir}/`);
    Deno.writeTextFileSync(
      fixtureUrl,
      `${JSON.stringify({ schema: 1, kind: "sha256", cases }, undefined, 2)}\n`,
    );
    body(fixtureUrl);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
};

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);

Deno.test("参照 fixture: lookup は現環境の行だけを返す", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, environment(CURRENT), undefined);
    assertEquals(references.lookup("case-1"), undefined, "他環境の行が現環境として見えている");
    assertEquals(references.hasCurrent(), false);
    assertEquals(references.lacksReference("case-1"), true);
    const other = openReferences(fixtureUrl, environment(OTHER), undefined);
    assertEquals(other.lookup("case-1"), SHA_A);
    assertEquals(other.hasCurrent(), true);
    assertEquals(other.lacksReference("case-1"), false);
  });
});

Deno.test("参照 fixture: write は無い行を足し、既存の行は上書きしない", () => {
  withFixture({ "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, environment(CURRENT), "write");
    // 行が無い ⇒ 実測を足して緑。
    assertEquals(references.check("case-1", SHA_B), { status: "written" });
    assertEquals(references.lookup("case-1"), SHA_B);
    // 行がある ⇒ 通常どおり比較（write でも上書きしない）。
    assertEquals(references.check("case-1", SHA_B), { status: "pass", expected: SHA_B });
    assertEquals(references.check("case-1", SHA_A), { status: "fail", expected: SHA_B });
    assertEquals(references.lookup("case-1"), SHA_B, "write モードが既存の行を書き換えた");
    // 書き戻した内容は次の読みでも同じ（他環境の行は生きたまま）。
    const reopened = openReferences(fixtureUrl, environment(OTHER), undefined);
    assertEquals(reopened.lookup("case-1"), SHA_A, "他環境の行が write で動いた");
  });
});

Deno.test("参照 fixture: rewrite は現環境の行だけを上書きする", () => {
  withFixture({ "case-1": { [CURRENT]: SHA_A, [OTHER]: SHA_A } }, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, environment(CURRENT), "rewrite");
    assertEquals(references.check("case-1", SHA_B), { status: "rewritten", previous: SHA_A });
    assertEquals(references.lookup("case-1"), SHA_B);
    const other = openReferences(fixtureUrl, environment(OTHER), undefined);
    assertEquals(other.lookup("case-1"), SHA_A, "rewrite が他環境の行を巻き込んだ");
  });
});

Deno.test("参照 fixture: 未設定モードで行が無いケースを突き合わせると throw する", () => {
  withFixture({}, (fixtureUrl) => {
    const references = openReferences(fixtureUrl, environment(CURRENT), undefined);
    assertThrows(
      () => references.check("case-1", SHA_A),
      Error,
      "KARUME_REFERENCE=write",
    );
  });
});

Deno.test("参照 fixture: 書き出しは辞書順で安定する（2 度書いてバイト同一）", () => {
  withFixture({ "case-2": { [OTHER]: SHA_A }, "case-1": { [OTHER]: SHA_A } }, (fixtureUrl) => {
    openReferences(fixtureUrl, environment(CURRENT), "write").check("case-2", SHA_B);
    const first = Deno.readTextFileSync(fixtureUrl);
    openReferences(fixtureUrl, environment(CURRENT), "rewrite").check("case-2", SHA_B);
    const second = Deno.readTextFileSync(fixtureUrl);
    assertEquals(second, first, "同じ内容を書き直したのにバイトが動いた");
    // JSON の鍵は挿入順で読み戻るので、並びの検査は parse した鍵の列で足りる。
    const document = JSON.parse(first) as { cases: Record<string, Record<string, string>> };
    assertEquals(Object.keys(document.cases), ["case-1", "case-2"], "ケース名が辞書順でない");
    assertEquals(
      Object.keys(document.cases["case-2"]),
      [CURRENT, OTHER].sort(),
      "環境キーが辞書順でない",
    );
    assert(first.endsWith("}\n"), "末尾改行が無い");
  });
});

Deno.test("参照 fixture: 読めない fixture は空として扱わず throw する", () => {
  withFixture({}, (fixtureUrl) => {
    assertThrows(
      () => openReferences(new URL("missing.json", fixtureUrl), environment(CURRENT), undefined),
      Error,
      "fixture が読めない",
    );
    Deno.writeTextFileSync(fixtureUrl, "{ broken");
    assertThrows(
      () => openReferences(fixtureUrl, environment(CURRENT), undefined),
      Error,
      "JSON として壊れている",
    );
    Deno.writeTextFileSync(fixtureUrl, JSON.stringify({ schema: 2, kind: "sha256", cases: {} }));
    assertThrows(
      () => openReferences(fixtureUrl, environment(CURRENT), undefined),
      Error,
      "schema が 2",
    );
  });
});

Deno.test("参照 fixture: KARUME_REFERENCE の未知の綴りは throw する", () => {
  assertEquals(parseReferenceMode(undefined), undefined);
  assertEquals(parseReferenceMode("write"), "write");
  assertEquals(parseReferenceMode("rewrite"), "rewrite");
  assertThrows(() => parseReferenceMode("1"), Error, "KARUME_REFERENCE");
  assertThrows(() => parseReferenceMode(""), Error, "KARUME_REFERENCE");
});

// `fromPretrained` の取得元指定（`src/hub/repo-ref.ts`）の門。**ネットワークには一切出ない**。
//
// 0.5.0 で `ref` は全ファミリ必須になった（既定ソースの廃止）。TS 側は引数必須なので型検査が
// 受け持つが、型を持たない JS からの引数なし呼び出しは実行時にしか止まらない。そこで落とせないと
// `undefined` が hub まで滑り、「repo が undefined の URL を叩いた」という**原因の遠い**失敗に
// 化ける。ここで押さえるのは 3 つ:
//  ① 取得元が無い呼び出しは「repo が必須」で落ちる。
//  ② その文言に**正しい記述例**が載る（`{ repo, revision }` の綴りと、そのファミリの
//     `*_CURRENT` 定数名の 2 択）— 読んだ人がそのまま直せない診断は「落ちた」だけで役に立たない。
//  ③ 公開配布リポを持たないファミリには存在しない定数を案内しない。
// 正常系（文字列 = main 追従 / オブジェクトはそのまま）も同じ 1 本が担うので合わせて縛る。

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { toRepoRef } from "../src/hub/repo-ref.ts";
import { AnimaPipeline } from "../src/anima/pipeline.ts";

Deno.test("toRepoRef: 取得元が無ければ『repo が必須』+ 記述例 2 択で落ちる", () => {
  // JS からの引数なし呼び出しの形（TS では型検査が先に落とす）。
  const error = assertThrows(
    () => toRepoRef(undefined, "AnimaPipeline.fromPretrained", "ANIMA_TURBO_CURRENT"),
    Error,
    "repo が必須",
  );
  // 主語（どの入口が落ちたか）と記述例 2 択が揃っていること。
  assertStringIncludes(error.message, "AnimaPipeline.fromPretrained");
  assertStringIncludes(error.message, '{ repo: "owner/name", revision: "<40 桁の commit SHA>" }');
  assertStringIncludes(error.message, "ANIMA_TURBO_CURRENT");
});

Deno.test("toRepoRef: 公開配布リポを持たないファミリには定数を案内しない", () => {
  // 存在しない識別子を案内すると、読んだ人は import できないものを探しに行く。
  const error = assertThrows(
    () => toRepoRef(undefined, "BirefnetPipeline.fromPretrained"),
    Error,
    "repo が必須",
  );
  assertStringIncludes(error.message, '{ repo: "owner/name", revision: "<40 桁の commit SHA>" }');
  assertStringIncludes(error.message, "リポ名の文字列");
  assertEquals(error.message.includes("_CURRENT"), false);
});

Deno.test("toRepoRef: 空の repo も同じ門で落ちる（空文字の URL を組み立てない）", () => {
  assertThrows(() => toRepoRef("", "Sbv2Pipeline.fromPretrained"), Error, "repo が必須");
  assertThrows(() => toRepoRef({ repo: "" }, "Sbv2Pipeline.fromPretrained"), Error, "repo が必須");
});

Deno.test("toRepoRef: 文字列は `{ repo }`（= main 追従）・オブジェクトはそのまま通す", () => {
  // 文字列 ref の意味は ref 必須化でも変えない（revision を書かない = main 追従）。
  assertEquals(toRepoRef("someone/karume-fork", "X.fromPretrained"), {
    repo: "someone/karume-fork",
  });
  // revision / hubUrl は hub の語彙なので、ここで削らない・足さない。
  const pinned = { repo: "someone/karume-fork", revision: "a".repeat(40), hubUrl: "https://m" };
  assertEquals(toRepoRef(pinned, "X.fromPretrained"), pinned);
});

Deno.test("fromPretrained: 取得元の綴りが空なら fetch を 1 度も呼ばずに落ちる", async () => {
  // 公開面から見た対。門が取得層より**前**にあることを、fetch 席へ「呼ばれたら失敗する」
  // スタブを置いて見る（呼ばれてしまうと文言がネットワーク側の失敗に化ける）。
  let calls = 0;
  const fetchStub: typeof globalThis.fetch = (input) => {
    calls += 1;
    return Promise.reject(new Error(`repo_ref_test: 取得層まで進んだ（${String(input)}）`));
  };
  await assertRejects(
    () => AnimaPipeline.fromPretrained({ repo: "" }, { fetch: fetchStub }),
    Error,
    "AnimaPipeline.fromPretrained: repo が必須",
  );
  assertEquals(calls, 0);
});

// `fromPretrained` の取得元指定（`src/hub/repo-ref.ts`）の門。**ネットワークには一切出ない**。
//
// 0.5.0 で `ref` は全ファミリ必須になった（既定ソースの廃止）。TS 側は引数必須なので型検査が
// 受け持つが、型を持たない JS からの引数なし呼び出しは実行時にしか止まらない。そこで落とせないと
// `undefined` が hub まで滑り、「repo が undefined の URL を叩いた」という**原因の遠い**失敗に
// 化ける。ここで押さえるのは 4 つ:
//  ① 取得元が無い呼び出しは「repo が必須」で落ちる。
//  ② その文言に**正しい記述例**が載る（`{ repo, revision }` の綴りと、そのファミリの
//     取得元対応表を引く綴りの 2 択）— 読んだ人がそのまま直せない診断は「落ちた」だけで
//     役に立たない。
//  ③ 公開配布リポを持たないファミリには存在しない識別子を案内しない。
//  ④ **綴りが HF の `owner/name` であること**（パスにしか見えない文字列を URL へ綴り込まない）。
//     救えない綴り（`models/karume-gemma4-e2b` のような合法な `owner/name`）も対で固定する。
//  ⑤ **取得元ハンドル**（`localDirectory` / `denoDirectory`）はこの門を通さず素通りし、かつ
//     素通りの席を作ったことで HF 側の門が緩んでいないこと。
// 正常系（文字列 = main 追従 / オブジェクトはそのまま）も同じ 1 本が担うので合わせて縛る。

import {
  assertEquals,
  assertRejects,
  assertStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { localDirectory } from "@karume/hub";
import { toManifestSource, toRepoRef } from "../src/hub/repo-ref.ts";
import { AnimaPipeline } from "../src/anima/pipeline.ts";

Deno.test("toRepoRef: 取得元が無ければ『repo が必須』+ 記述例 2 択で落ちる", () => {
  // JS からの引数なし呼び出しの形（TS では型検査が先に落とす）。
  const error = assertThrows(
    () => toRepoRef(undefined, "AnimaPipeline.fromPretrained", 'ANIMA_SOURCES["anima"]'),
    Error,
    "repo が必須",
  );
  // 主語（どの入口が落ちたか）と記述例 2 択が揃っていること。
  assertStringIncludes(error.message, "AnimaPipeline.fromPretrained");
  assertStringIncludes(error.message, '{ repo: "owner/name", revision: "<40 桁の commit SHA>" }');
  assertStringIncludes(error.message, 'ANIMA_SOURCES["anima"]');
});

Deno.test("toRepoRef: 公開配布リポを持たないファミリには対応表を案内しない", () => {
  // 存在しない識別子を案内すると、読んだ人は import できないものを探しに行く。
  const error = assertThrows(
    () => toRepoRef(undefined, "BirefnetPipeline.fromPretrained"),
    Error,
    "repo が必須",
  );
  assertStringIncludes(error.message, '{ repo: "owner/name", revision: "<40 桁の commit SHA>" }');
  assertStringIncludes(error.message, "リポ名の文字列");
  assertEquals(error.message.includes("_SOURCES"), false);
});

Deno.test("toRepoRef: 空の repo も同じ門で落ちる（空文字の URL を組み立てない）", () => {
  assertThrows(() => toRepoRef("", "Sbv2Pipeline.fromPretrained"), Error, "repo が必須");
  assertThrows(() => toRepoRef({ repo: "" }, "Sbv2Pipeline.fromPretrained"), Error, "repo が必須");
});

Deno.test("toRepoRef: パスにしか見えない綴りは HF へ投げる前に落ちる", () => {
  // ローカルのディレクトリを渡した呼び出しは、門が無いとその文字列が URL へ綴り込まれ、
  // 返るのは 401 / 404 —「取得先が存在しない」という原因の遠い診断になる。
  const paths = [
    "./models/karume-gemma4-e2b", // 先頭 './'
    "../karume-gemma4-e2b", // 先頭 '../'
    "/home/me/models/dist", // 絶対パス
    "hdae/karume-anima/", // 末尾 '/'
    "a/b/c", // スラッシュ 2 個以上
    "hdae//karume-anima", // 空セグメント
    "hdae/karume anima", // 許可外の文字
    ".hidden/name", // 先頭ドット
    "karume-gemma4-e2b", // セグメント 1 つ（`owner/name` でない）
  ];
  for (const repo of paths) {
    assertThrows(
      () => toRepoRef(repo, "Gemma4Pipeline.fromPretrained"),
      Error,
      "'owner/name' でない",
      repo,
    );
  }
});

Deno.test("toRepoRef: `owner/name` に見えるローカルパスは救えない（仕様）", () => {
  // 綴りとして完全に合法な HF repo 名なので、この引数の中に区別できる情報が無い。
  // 「実在するか」は取得層の仕事で、ここが見るのは綴りだけである。
  assertEquals(toRepoRef("models/karume-gemma4-e2b", "Gemma4Pipeline.fromPretrained"), {
    repo: "models/karume-gemma4-e2b",
  });
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

Deno.test("toManifestSource: 取得元ハンドルは綴りの門を通さずそのまま渡る", () => {
  // ローカルの配布形は HF の repo ではないので、`owner/name` を要求する門に掛けてはならない
  // （掛けると「repo が必須」で落ち、手元のディレクトリからは 1 バイトも読めない）。
  // 実体は 1 度も読まないので、アダプターは読まれたら落ちる形で十分。
  const source = localDirectory({
    readFile: () => Promise.reject(new Error("repo_ref_test: 判別だけで実体を読んだ")),
  }, { label: "./models/karume-test" });
  assertStrictEquals(toManifestSource(source, "Gemma4Pipeline.fromPretrained"), source);
});

Deno.test("toManifestSource: 取得元ハンドルでない値は従来どおり綴りの門を通る", () => {
  // union を足しても HF 側の門が緩まないこと（取得元ハンドルの判別は同一性なので、
  // 「repo を持つオブジェクト」が取得元として素通りする形にはならない）。
  assertEquals(toManifestSource("someone/karume-fork", "X.fromPretrained"), {
    repo: "someone/karume-fork",
  });
  assertThrows(
    () => toManifestSource("./models/karume-gemma4-e2b", "X.fromPretrained"),
    Error,
    "'owner/name' でない",
  );
  assertThrows(() => toManifestSource(undefined, "X.fromPretrained"), Error, "repo が必須");
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
